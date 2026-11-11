// Package whisper turns audio into timed words. It is only used to find WHEN speech happens;
// the transcribed text is never written anywhere.
//
//   - Local: whisper.cpp's whisper-server, started on demand and stopped after an idle period
//     so the GPU memory is handed back. The default small-q5_1 model needs well under 1 GB of VRAM.
//   - OpenAI: any OpenAI-compatible /audio/transcriptions endpoint (OpenAI, Groq, self-hosted).
package whisper

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"regexp"
	"strings"
	"time"

	"github.com/jorgentuck/captionSync/internal/media"
)

type Word struct {
	Text       string
	Start, End float64
	// Reliable marks a timestamp we trust to the ~0.1-0.2 s level. whisper.cpp only gets segment
	// START times right (inner word times are spread evenly over the segment), so only the first
	// word of each segment is reliable there. OpenAI-style word timestamps are all reliable.
	Reliable bool
}

type Chunk struct {
	Words    []Word
	Language string
	// SegmentLevel: only segment start times came back (whisper.cpp, or an API without word
	// timestamps). The first word of each segment carries the segment start, which still needs to
	// be refined against the audio; the other words' times are rough.
	SegmentLevel bool
}

// ErrEngine marks failures of the transcription engine itself (server won't start, keeps crashing,
// hangs, API down) as opposed to problems with one file. Files hitting it are retried later
// instead of being marked failed.
var ErrEngine = errors.New("transcription engine unavailable")

type Engine interface {
	Name() string
	Transcribe(ctx context.Context, pcm []byte, language string) (*Chunk, error)
	Status() map[string]any
	Close()
	// Thorough returns the engine to use for thorough re-checks (a stronger model, if configured).
	Thorough() Engine
}

// wav wraps 16 kHz mono s16le PCM in a WAV header.
func wav(pcm []byte) []byte {
	var b bytes.Buffer
	w := func(v any) { _ = binary.Write(&b, binary.LittleEndian, v) }
	b.WriteString("RIFF")
	w(uint32(36 + len(pcm)))
	b.WriteString("WAVEfmt ")
	w(uint32(16))
	w(uint16(1))
	w(uint16(1))
	w(uint32(media.SampleRate))
	w(uint32(media.SampleRate * 2))
	w(uint16(2))
	w(uint16(16))
	b.WriteString("data")
	w(uint32(len(pcm)))
	b.Write(pcm)
	return b.Bytes()
}

func multipartBody(fields [][2]string, audio []byte) (*bytes.Buffer, string, error) {
	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	for _, f := range fields {
		if err := mw.WriteField(f[0], f[1]); err != nil {
			return nil, "", err
		}
	}
	fw, err := mw.CreateFormFile("file", "audio.wav")
	if err != nil {
		return nil, "", err
	}
	if _, err := fw.Write(wav(audio)); err != nil {
		return nil, "", err
	}
	if err := mw.Close(); err != nil {
		return nil, "", err
	}
	return &body, mw.FormDataContentType(), nil
}

type jsonWord = struct {
	Word  string  `json:"word"`
	Start float64 `json:"start"`
	End   float64 `json:"end"`
}

var (
	// non-speech annotations whisper emits: [BLANK_AUDIO], [Music], (laughs), *sighs*, ♪ ... ♪
	nonSpeech = regexp.MustCompile(`\[[^\]]*\]|\([^)]*\)|\*[^*]*\*|♪[^♪]*♪?`)
	hasWordRe = regexp.MustCompile(`[\p{L}\p{N}]`)
)

// segmentWords splits a segment's text into real words (no non-speech tags, no bare punctuation).
// whisper.cpp's per-segment "words" are BPE tokens ("BL","ANK"...), so the text is used instead.
func segmentWords(text string) []string {
	var out []string
	for _, w := range strings.Fields(nonSpeech.ReplaceAllString(text, " ")) {
		if hasWordRe.MatchString(w) {
			out = append(out, w)
		}
	}
	return out
}

// verboseJSON is the OpenAI "verbose_json" shape (whisper.cpp server uses the same).
type verboseJSON struct {
	Language string     `json:"language"`
	Text     string     `json:"text"`
	Words    []jsonWord `json:"words"`
	Segments []struct {
		Text  string     `json:"text"`
		Start float64    `json:"start"`
		End   float64    `json:"end"`
		Words []jsonWord `json:"words"`
	} `json:"segments"`
}

func (v *verboseJSON) chunk() *Chunk {
	c := &Chunk{Language: v.Language}
	for _, w := range v.Words { // real word timestamps (OpenAI-style)
		if hasWordRe.MatchString(w.Word) && !nonSpeech.MatchString(w.Word) {
			c.Words = append(c.Words, Word{w.Word, w.Start, w.End, true})
		}
	}
	if len(c.Words) > 0 {
		return c
	}
	c.SegmentLevel = true
	for _, s := range v.Segments {
		toks := segmentWords(s.Text)
		if len(toks) == 0 || s.End <= s.Start {
			continue
		}
		// The segment start is the only timestamp we use; inner words are spread evenly. If the
		// segment opens with a non-speech tag ("(laughs) Okay"), its start is the laugh, not the word.
		// ...but punctuation (">>", "...", a dash) before the first word is just text, not a sound.
		lead := s.Text
		if i := strings.Index(s.Text, toks[0]); i >= 0 {
			lead = s.Text[:i]
		}
		leadsWithWord := !nonSpeech.MatchString(lead)
		step := (s.End - s.Start) / float64(len(toks))
		for i, t := range toks {
			st := s.Start + float64(i)*step
			c.Words = append(c.Words, Word{t, st, st + step, i == 0 && leadsWithWord})
		}
	}
	return c
}

func postTranscription(ctx context.Context, client *http.Client, url string, headers map[string]string,
	fields [][2]string, pcm []byte, attempts int) (*Chunk, error) {
	var lastErr error
	for attempt := 0; attempt < attempts; attempt++ {
		if attempt > 0 {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(time.Duration(2<<attempt) * time.Second):
			}
		}
		body, ctype, err := multipartBody(fields, pcm)
		if err != nil {
			return nil, err
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, body)
		if err != nil {
			return nil, err
		}
		req.Header.Set("Content-Type", ctype)
		for k, v := range headers {
			req.Header.Set(k, v)
		}
		resp, err := client.Do(req)
		if err != nil {
			lastErr = err
			continue
		}
		data, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.StatusCode == 429 || resp.StatusCode >= 500 {
			lastErr = fmt.Errorf("HTTP %d: %s", resp.StatusCode, truncate(string(data), 200))
			continue
		}
		if resp.StatusCode != 200 {
			err := fmt.Errorf("HTTP %d: %s", resp.StatusCode, truncate(string(data), 300))
			switch resp.StatusCode {
			case 400, 413, 415, 422: // this request (this audio) was rejected
				return nil, err
			}
			return nil, fmt.Errorf("%w: %v", ErrEngine, err) // auth, missing model, ...: affects every file
		}
		var v verboseJSON
		if err := json.Unmarshal(data, &v); err != nil {
			return nil, fmt.Errorf("bad transcription response: %w", err)
		}
		return v.chunk(), nil
	}
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}
	return nil, fmt.Errorf("%w: transcription failed: %v", ErrEngine, lastErr)
}

func truncate(s string, n int) string {
	if len(s) > n {
		return s[:n] + "..."
	}
	return s
}

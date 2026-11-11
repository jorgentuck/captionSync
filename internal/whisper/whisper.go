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
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
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
}

type Engine interface {
	Name() string
	Transcribe(ctx context.Context, pcm []byte, language string) (*Chunk, error)
	Status() map[string]any
	Close()
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

// wordsInside reports whether word times are plausible for their segment. With VAD enabled,
// whisper.cpp reports word times on the VAD-compressed timeline, so they must be ignored.
func wordsInside(ws []jsonWord, start, end float64) bool {
	for _, w := range ws {
		if w.Start < start-1 || w.Start > end+1 {
			return false
		}
	}
	return true
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
	for _, w := range v.Words {
		c.Words = append(c.Words, Word{w.Word, w.Start, w.End, true})
	}
	if len(c.Words) > 0 {
		return c
	}
	for _, s := range v.Segments {
		if len(s.Words) > 0 && wordsInside(s.Words, s.Start, s.End) {
			for i, w := range s.Words {
				st := w.Start
				if i == 0 {
					st = s.Start
				}
				c.Words = append(c.Words, Word{w.Word, st, w.End, i == 0})
			}
			continue
		}
		// no word timings: spread the segment's words evenly over it
		toks := strings.Fields(s.Text)
		if len(toks) == 0 {
			continue
		}
		step := (s.End - s.Start) / float64(len(toks))
		for i, t := range toks {
			st := s.Start + float64(i)*step
			c.Words = append(c.Words, Word{t, st, st + step, i == 0})
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
			return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, truncate(string(data), 300))
		}
		var v verboseJSON
		if err := json.Unmarshal(data, &v); err != nil {
			return nil, fmt.Errorf("bad transcription response: %w", err)
		}
		return v.chunk(), nil
	}
	return nil, fmt.Errorf("transcription failed: %w", lastErr)
}

func truncate(s string, n int) string {
	if len(s) > n {
		return s[:n] + "..."
	}
	return s
}

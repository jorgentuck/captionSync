// Package media wraps ffprobe/ffmpeg: probing, audio decoding and (embedded.go) embedded subtitles.
// Video files are only written by Remux, which is used only when EMBEDDED_SUBS=remux.
package media

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

const SampleRate = 16000

type Stream struct {
	Index       int               `json:"index"`
	CodecType   string            `json:"codec_type"`
	CodecName   string            `json:"codec_name"`
	StartTime   string            `json:"start_time"`
	Duration    string            `json:"duration"`
	Tags        map[string]string `json:"tags"`
	Disposition map[string]int    `json:"disposition"`
}

func (s Stream) Tag(k string) string {
	for tk, v := range s.Tags {
		if strings.EqualFold(tk, k) {
			return v
		}
	}
	return ""
}

func (s Stream) Language() string { return strings.ToLower(s.Tag("language")) }

type Probe struct {
	Streams []Stream `json:"streams"`
	Format  struct {
		Duration   string `json:"duration"`
		FormatName string `json:"format_name"`
		StartTime  string `json:"start_time"`
	} `json:"format"`
}

func run(ctx context.Context, timeout time.Duration, name string, args ...string) ([]byte, error) {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, name, args...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	if err := cmd.Run(); err != nil {
		lines := strings.Split(strings.TrimSpace(stderr.String()), "\n")
		if len(lines) > 5 {
			lines = lines[len(lines)-5:]
		}
		if ctx.Err() == context.DeadlineExceeded {
			return nil, fmt.Errorf("%s timed out", name)
		}
		return nil, fmt.Errorf("%s failed: %v: %s", name, err, strings.Join(lines, " | "))
	}
	return stdout.Bytes(), nil
}

func ProbeFile(ctx context.Context, path string) (*Probe, error) {
	out, err := run(ctx, 2*time.Minute, "ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path)
	if err != nil {
		return nil, err
	}
	var p Probe
	if err := json.Unmarshal(out, &p); err != nil {
		return nil, fmt.Errorf("parse ffprobe output: %w", err)
	}
	return &p, nil
}

func (p *Probe) Duration() float64 {
	if d, err := strconv.ParseFloat(p.Format.Duration, 64); err == nil && d > 0 {
		return d
	}
	best := 0.0
	for _, s := range p.Streams {
		if d, err := strconv.ParseFloat(s.Duration, 64); err == nil {
			best = math.Max(best, d)
		}
	}
	return best
}

// AudioDelay is how late audio stream N (relative index) starts compared with the file's start.
// ffmpeg's raw PCM output does not pad that gap, so audio decoded from before it begins late.
func (p *Probe) AudioDelay(audioRel int) float64 {
	fs, _ := strconv.ParseFloat(p.Format.StartTime, 64)
	i := 0
	for _, s := range p.Streams {
		if s.CodecType != "audio" {
			continue
		}
		if i == audioRel {
			as, err := strconv.ParseFloat(s.StartTime, 64)
			if err != nil || as-fs < 0.01 {
				return 0
			}
			return as - fs
		}
		i++
	}
	return 0
}

// ChooseAudio returns the relative audio stream index (for -map 0:a:N), or -1.
func (p *Probe) ChooseAudio(preferred []string) int {
	var audio []Stream
	for _, s := range p.Streams {
		if s.CodecType == "audio" {
			audio = append(audio, s)
		}
	}
	if len(audio) == 0 {
		return -1
	}
	for _, lang := range preferred {
		for i, s := range audio {
			if s.Language() == lang {
				return i
			}
		}
	}
	for i, s := range audio {
		if s.Disposition["default"] == 1 {
			return i
		}
	}
	return 0
}

// ReadAudio decodes a slice of one audio stream to 16 kHz mono signed 16-bit little-endian PCM.
func ReadAudio(ctx context.Context, path string, audioRel int, start, length float64) ([]byte, error) {
	timeout := time.Duration(math.Max(120, length*4)) * time.Second
	return run(ctx, timeout, "ffmpeg", "-nostdin", "-hide_banner", "-v", "error",
		"-ss", fmt.Sprintf("%.3f", math.Max(0, start)), "-i", path, "-t", fmt.Sprintf("%.3f", length),
		"-map", fmt.Sprintf("0:a:%d", audioRel), "-vn", "-sn", "-dn",
		"-ac", "1", "-ar", strconv.Itoa(SampleRate), "-f", "s16le", "-")
}

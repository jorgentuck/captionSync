package media

import (
	"context"
	"fmt"
	"math"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// TextSubCodecs maps ffmpeg codec names to the text format we extract them to.
// Anything not listed (hdmv_pgs_subtitle, dvd_subtitle, dvb_subtitle, ...) is image-based.
var TextSubCodecs = map[string]string{
	"subrip": "srt", "srt": "srt", "text": "srt", "mov_text": "srt",
	"ass": "ass", "ssa": "ass", "webvtt": "vtt",
}

var subEncoder = map[string]string{"srt": "srt", "ass": "ass", "vtt": "webvtt"}

type SubStream struct {
	Stream
	Rel    int    // index among subtitle streams (-map 0:s:Rel)
	Format string // srt | ass | vtt; "" = image-based, cannot be retimed by text
}

// Label is a short human description, e.g. "#3 eng subrip (SDH)".
func (s SubStream) Label() string {
	parts := []string{fmt.Sprintf("#%d", s.Index)}
	if l := s.Language(); l != "" {
		parts = append(parts, l)
	}
	parts = append(parts, s.CodecName)
	if t := s.Tag("title"); t != "" {
		parts = append(parts, "("+t+")")
	}
	if s.Disposition["forced"] == 1 {
		parts = append(parts, "[forced]")
	}
	return strings.Join(parts, " ")
}

func (p *Probe) SubtitleStreams() []SubStream {
	var out []SubStream
	rel := 0
	for _, s := range p.Streams {
		if s.CodecType != "subtitle" {
			continue
		}
		out = append(out, SubStream{Stream: s, Rel: rel, Format: TextSubCodecs[s.CodecName]})
		rel++
	}
	return out
}

type Extract struct {
	Rel    int
	Format string
	Out    string
}

// ExtractSubtitles pulls several text subtitle streams out in a single pass over the file.
func ExtractSubtitles(ctx context.Context, path string, tracks []Extract) error {
	if len(tracks) == 0 {
		return nil
	}
	args := []string{"-nostdin", "-hide_banner", "-v", "error", "-y", "-i", path}
	for _, t := range tracks {
		args = append(args, "-map", fmt.Sprintf("0:s:%d", t.Rel), "-c:s", subEncoder[t.Format], t.Out)
	}
	_, err := run(ctx, time.Hour, "ffmpeg", args...)
	return err
}

var mp4Like = map[string]bool{".mp4": true, ".m4v": true, ".mov": true}

var keepDisposition = []string{"default", "forced", "hearing_impaired", "visual_impaired", "comment",
	"lyrics", "karaoke", "original", "dub", "captions", "descriptions"}

// Remux writes `out` with every stream of src stream-copied, except the subtitle streams listed in
// replacements (absolute stream index -> retimed subtitle file). Language, title and default/forced
// flags are carried over. The result is checked before returning.
func Remux(ctx context.Context, src string, p *Probe, replacements map[int]string, out string) error {
	mp4 := mp4Like[strings.ToLower(filepath.Ext(out))]
	args := []string{"-nostdin", "-hide_banner", "-v", "error", "-y", "-i", src}
	idxs := make([]int, 0, len(replacements))
	for idx := range replacements {
		idxs = append(idxs, idx)
	}
	sort.Ints(idxs)
	inputs := map[int]int{}
	for n, idx := range idxs {
		args = append(args, "-i", replacements[idx])
		inputs[idx] = n + 1
	}
	var maps, opts []string
	outI := 0
	for _, st := range p.Streams {
		if mp4 && st.CodecType == "data" {
			continue // timecode/data tracks often can't be written back into mp4
		}
		if in, ok := inputs[st.Index]; ok {
			maps = append(maps, "-map", fmt.Sprintf("%d:0", in))
			codec := "copy"
			if mp4 {
				codec = "mov_text"
			}
			opts = append(opts, fmt.Sprintf("-c:%d", outI), codec)
			if l := st.Tag("language"); l != "" {
				opts = append(opts, fmt.Sprintf("-metadata:s:%d", outI), "language="+l)
			}
			if t := st.Tag("title"); t != "" {
				opts = append(opts, fmt.Sprintf("-metadata:s:%d", outI), "title="+t)
			}
			var flags []string
			for _, k := range keepDisposition {
				if st.Disposition[k] == 1 {
					flags = append(flags, k)
				}
			}
			disp := "0"
			if len(flags) > 0 {
				disp = strings.Join(flags, "+")
			}
			opts = append(opts, fmt.Sprintf("-disposition:%d", outI), disp)
		} else {
			maps = append(maps, "-map", fmt.Sprintf("0:%d", st.Index))
		}
		outI++
	}
	args = append(args, maps...)
	args = append(args, "-map_metadata", "0", "-map_chapters", "0", "-c", "copy")
	args = append(args, opts...)
	if mp4 {
		args = append(args, "-movflags", "+faststart")
	}
	args = append(args, out)
	if _, err := run(ctx, 6*time.Hour, "ffmpeg", args...); err != nil {
		return err
	}
	np, err := ProbeFile(ctx, out)
	if err != nil {
		return fmt.Errorf("probe remuxed file: %w", err)
	}
	// compare per stream type: the mp4 muxer may recreate chapter/timecode tracks we skipped
	count := func(ss []Stream, mp4Data bool) map[string]int {
		c := map[string]int{}
		for _, s := range ss {
			if s.CodecType == "video" || s.CodecType == "audio" || s.CodecType == "subtitle" || (!mp4Data && s.CodecType == "attachment") {
				c[s.CodecType]++
			}
		}
		return c
	}
	want, got := count(p.Streams, mp4), count(np.Streams, mp4)
	for k, v := range want {
		if got[k] != v {
			return fmt.Errorf("remux produced %d %s streams, expected %d", got[k], k, v)
		}
	}
	if d0, d1 := p.Duration(), np.Duration(); d0 > 0 && math.Abs(d0-d1) > math.Max(2, d0*0.01) {
		return fmt.Errorf("remux duration mismatch (%.1fs vs %.1fs)", d0, d1)
	}
	return nil
}

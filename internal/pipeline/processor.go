// Package pipeline processes one video at a time (Processor) and runs the queue (Service).
package pipeline

import (
	"context"
	"fmt"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"syscall"
	"time"

	"github.com/jorgentuck/captionSync/internal/align"
	"github.com/jorgentuck/captionSync/internal/config"
	"github.com/jorgentuck/captionSync/internal/library"
	"github.com/jorgentuck/captionSync/internal/media"
	"github.com/jorgentuck/captionSync/internal/store"
	"github.com/jorgentuck/captionSync/internal/subtitle"
	"github.com/jorgentuck/captionSync/internal/whisper"
)

const minCues = 10

type Processor struct {
	cfg    *config.Config
	lib    *library.Library
	engine whisper.Engine
}

func NewProcessor(cfg *config.Config, lib *library.Library, engine whisper.Engine) *Processor {
	return &Processor{cfg: cfg, lib: lib, engine: engine}
}

type Outcome struct {
	Status    string
	Message   string
	OffsetMs  *int64
	Details   map[string]any
	Signature string
}

type track struct {
	name     string // shown in the UI
	path     string // sidecar file, or the extracted copy of an embedded track
	embedded *media.SubStream
	file     *subtitle.File
	cues     []align.Cue
	result   align.Result
	written  bool
	changed  int
	err      string
	short    bool
}

type transcript struct {
	words    []align.Word
	speech   [][2]float64
	spans    [][2]float64
	language string
	sampled  bool
	seconds  float64
}

func (p *Processor) params() align.Params {
	pr := align.DefaultParams()
	pr.MaxOffset = p.cfg.MaxOffset
	pr.MatchThreshold = float64(p.cfg.MatchThreshold)
	pr.MinAnchors = p.cfg.MinAnchors
	pr.SyncThreshold = float64(p.cfg.SyncThresholdMs) / 1000
	pr.AllowActivity = p.cfg.AllowActivity
	return pr
}

// Process checks the subtitles of one video and fixes the ones that are out of sync.
func (p *Processor) Process(ctx context.Context, path string, progress func(string)) Outcome {
	started := time.Now()
	sigBefore := p.lib.Signature(path)
	if sigBefore == "" {
		return Outcome{Status: store.Missing, Message: "File no longer exists"}
	}
	out := Outcome{Signature: sigBefore, Details: map[string]any{}}

	progress("probing")
	pr, err := media.ProbeFile(ctx, path)
	if err != nil {
		return fail(out, err)
	}
	dur := pr.Duration()

	// 1. collect subtitle tracks: sidecar files, plus embedded text tracks when enabled
	var tracks []*track
	for _, sp := range p.lib.Sidecars(path) {
		tracks = append(tracks, &track{name: filepath.Base(sp), path: sp})
	}
	var embText, embImage []media.SubStream
	for _, ss := range pr.SubtitleStreams() {
		if ss.Format != "" {
			embText = append(embText, ss)
		} else {
			embImage = append(embImage, ss)
		}
	}
	out.Details["embedded"] = embeddedInfo(embText, embImage)
	checkEmbedded := len(embText) > 0 && (p.cfg.EmbeddedMode == "remux" || p.cfg.EmbeddedMode == "sidecar")
	if checkEmbedded {
		tmp, err := os.MkdirTemp(p.tmpRoot(), "job-")
		if err != nil {
			return fail(out, err)
		}
		defer os.RemoveAll(tmp)
		progress("extracting embedded subtitles")
		var ex []media.Extract
		for i := range embText {
			ss := embText[i]
			if p.cfg.EmbeddedMode == "sidecar" {
				if _, err := os.Stat(embeddedSidecarPath(path, &ss)); err == nil {
					continue // exported on an earlier run; that sidecar is checked like any other
				}
			}
			t := &track{name: "embedded " + ss.Label(), embedded: &ss,
				path: filepath.Join(tmp, fmt.Sprintf("s%d.%s", ss.Index, ss.Format))}
			tracks = append(tracks, t)
			ex = append(ex, media.Extract{Rel: ss.Rel, Format: ss.Format, Out: t.path})
		}
		if err := media.ExtractSubtitles(ctx, path, ex); err != nil {
			return fail(out, fmt.Errorf("extract embedded subtitles: %w", err))
		}
	}

	if len(tracks) == 0 {
		out.Status = store.NoSubs
		switch {
		case len(embText) > 0 && p.cfg.EmbeddedMode != "off":
			out.Status = store.Embedded
			out.Message = "Only embedded subtitles (" + labels(embText) +
				"), not checked. Set EMBEDDED_SUBS=remux or sidecar to fix them."
		case len(embImage) > 0:
			out.Message = "Only image-based embedded subtitles (" + labels(embImage) + "), which can't be retimed"
		default:
			out.Message = "No subtitles"
		}
		return out
	}

	for _, t := range tracks {
		f, err := subtitle.Load(t.path)
		if err != nil {
			t.err = err.Error()
			continue
		}
		t.file = f
		for _, c := range f.Cues() {
			t.cues = append(t.cues, align.Cue{Start: c.Start, End: c.End, Tokens: align.Tokenize(c.Text, true)})
		}
		sort.Slice(t.cues, func(i, j int) bool { return t.cues[i].Start < t.cues[j].Start })
		if len(t.cues) < minCues {
			t.short = true // e.g. forced subs with a handful of lines: skip, not an error
			t.err = fmt.Sprintf("skipped: only %d lines, too few to analyse", len(t.cues))
		}
	}
	usable, short := 0, 0
	for _, t := range tracks {
		if t.err == "" {
			usable++
		} else if t.short {
			short++
		}
	}
	if usable == 0 {
		out.Status, out.Message = store.Failed, "No usable subtitles: "+tracks[0].err
		if short == len(tracks) {
			out.Status, out.Message = store.Uncertain, "Subtitles too short to analyse"
		}
		out.Details["tracks"] = trackDetails(tracks)
		return out
	}

	// 2. listen + analyse (sampled first, full pass only when needed)
	audio := pr.ChooseAudio(p.cfg.PreferredAudioLang)
	if audio < 0 || dur <= 0 {
		return fail(out, fmt.Errorf("no audio stream / unknown duration"))
	}
	sampled := p.cfg.AnalysisMode != "full"
	tr, err := p.transcribe(ctx, path, audio, dur, sampled, progress)
	if err != nil {
		return fail(out, err)
	}
	p.analyse(tracks, tr, dur, progress)
	if tr.sampled && p.cfg.AutoFull && wantsFull(tracks) {
		slog.Info("sampled analysis inconclusive, running full pass", "path", path)
		if full, err := p.transcribe(ctx, path, audio, dur, false, progress); err == nil {
			tr = full
			p.analyse(tracks, tr, dur, progress)
		} else if ctx.Err() != nil {
			return fail(out, err)
		} else {
			slog.Warn("full pass failed, keeping sampled result", "err", err)
		}
	}

	// 3. write fixes
	var fixed, uncertain, failed int
	var firstShift *int64
	remux := map[int]string{}
	var remuxTracks []*track
	for _, t := range tracks {
		if t.err != "" {
			continue
		}
		switch t.result.Status {
		case "needs_fix":
			ms := t.result.MedianShiftMs
			if firstShift == nil {
				firstShift = &ms
			}
			if p.cfg.DryRun {
				fixed++
				continue
			}
			progress("writing " + t.name)
			if p.cfg.Backup && t.embedded != nil {
				_ = p.backup(path+fmt.Sprintf(".track%d.%s", t.embedded.Index, t.embedded.Format), t.path)
			}
			t.changed = t.file.Retime(t.result.Mapping.Apply)
			var err error
			switch {
			case t.embedded == nil:
				err = p.writeSidecar(t)
			case p.cfg.EmbeddedMode == "sidecar":
				err = p.writeEmbeddedAsSidecar(path, t)
			default: // remux: collect, rewrite the video once below
				if err = os.WriteFile(t.path, t.file.Bytes(), 0o644); err == nil {
					remux[t.embedded.Index] = t.path
					remuxTracks = append(remuxTracks, t)
					continue
				}
			}
			if err != nil {
				t.err = "write failed: " + err.Error()
				failed++
				continue
			}
			t.written = true
			fixed++
		case "uncertain":
			uncertain++
		}
	}
	if len(remux) > 0 {
		progress("remuxing video with corrected subtitles")
		err := p.remuxVideo(ctx, path, pr, remux)
		for _, t := range remuxTracks {
			if err != nil {
				t.err = "remux failed: " + err.Error()
				failed++
			} else {
				t.written = true
				fixed++
			}
		}
	}

	out.OffsetMs = firstShift
	out.Signature = p.lib.Signature(path) // includes our own rewrites, so they don't re-trigger
	out.Details["tracks"] = trackDetails(tracks)
	out.Details["duration_s"] = math.Round(dur)
	out.Details["audio_language"] = tr.language
	out.Details["analysis"] = map[bool]string{true: "sample", false: "full"}[tr.sampled]
	out.Details["audio_analysed_s"] = math.Round(tr.seconds)
	out.Details["elapsed_s"] = math.Round(time.Since(started).Seconds()*10) / 10
	out.Details["dry_run"] = p.cfg.DryRun

	var msgs []string
	for _, t := range tracks {
		m := t.result.Message
		if t.err != "" {
			m = t.err
		}
		msgs = append(msgs, t.name+": "+m)
	}
	if !checkEmbedded && len(embText) > 0 && p.cfg.EmbeddedMode != "off" {
		msgs = append(msgs, fmt.Sprintf("%d embedded text track(s) not checked", len(embText)))
	}
	out.Message = strings.Join(msgs, "; ")
	switch {
	case failed > 0:
		out.Status = store.Failed
	case fixed > 0 && p.cfg.DryRun:
		out.Status = store.NeedsFix
	case fixed > 0:
		out.Status = store.Fixed
	case uncertain > 0:
		out.Status = store.Uncertain
	default:
		out.Status = store.InSync
	}
	return out
}

func labels(ss []media.SubStream) string {
	var l []string
	for _, s := range ss {
		l = append(l, s.Label())
	}
	return strings.Join(l, ", ")
}

func embeddedInfo(text, image []media.SubStream) []map[string]any {
	out := []map[string]any{}
	for _, s := range text {
		out = append(out, map[string]any{"label": s.Label(), "format": s.Format, "text": true})
	}
	for _, s := range image {
		out = append(out, map[string]any{"label": s.Label(), "text": false})
	}
	return out
}

func fail(out Outcome, err error) Outcome {
	out.Status, out.Message = store.Failed, err.Error()
	return out
}

func wantsFull(tracks []*track) bool {
	for _, t := range tracks {
		if t.err == "" && t.result.WantsFull {
			return true
		}
	}
	return false
}

func (p *Processor) analyse(tracks []*track, tr *transcript, dur float64, progress func(string)) {
	progress("aligning")
	for _, t := range tracks {
		if t.err == "" {
			t.result = align.Analyze(t.cues, tr.words, tr.speech, tr.spans, dur, p.params(), tr.sampled)
		}
	}
}

func trackDetails(tracks []*track) []map[string]any {
	out := make([]map[string]any, 0, len(tracks))
	for _, t := range tracks {
		d := map[string]any{"file": t.name, "error": t.err, "written": t.written,
			"changed_lines": t.changed}
		if t.err == "" || t.written {
			d["status"] = t.result.Status
			d["message"] = t.result.Message
			d["cues"] = t.result.Cues
			d["anchors"] = t.result.Anchors
			d["median_shift_ms"] = t.result.MedianShiftMs
			d["max_shift_ms"] = t.result.MaxShiftMs
			d["mapping"] = t.result.Mapping
		}
		out = append(out, d)
	}
	return out
}

// --- listening -------------------------------------------------------------

func windows(dur float64, sampled bool, n, w, chunk int) [][2]float64 {
	if sampled && dur > float64(n*w)*1.2 {
		margin := math.Min(120, dur*0.03)
		usable := dur - 2*margin - float64(w)
		out := make([][2]float64, n)
		for i := range out {
			out[i] = [2]float64{margin + usable*float64(i)/float64(n-1), float64(w)}
		}
		return out
	}
	var out [][2]float64
	for s := 0.0; s < dur; s += float64(chunk) {
		out = append(out, [2]float64{s, math.Min(float64(chunk), dur-s)})
	}
	return out
}

type pcmChunk struct {
	start, length float64
	pcm           []byte
	err           error
}

func (p *Processor) transcribe(ctx context.Context, path string, audio int, dur float64, sampled bool,
	progress func(string)) (*transcript, error) {
	wins := windows(dur, sampled, p.cfg.SampleWindows, p.cfg.SampleWindowSec, p.cfg.FullChunkSec)
	tr := &transcript{sampled: sampled && len(wins) == p.cfg.SampleWindows && dur > float64(p.cfg.SampleWindows*p.cfg.SampleWindowSec)*1.2}
	lang := p.cfg.AudioLanguage

	// decode the next window with ffmpeg while the current one is being transcribed
	ch := make(chan pcmChunk, 2)
	cctx, cancel := context.WithCancel(ctx)
	defer cancel()
	go func() {
		defer close(ch)
		for _, w := range wins {
			pcm, err := media.ReadAudio(cctx, path, audio, w[0], w[1])
			select {
			case ch <- pcmChunk{w[0], w[1], pcm, err}:
			case <-cctx.Done():
				return
			}
			if err != nil {
				return
			}
		}
	}()

	i := 0
	for c := range ch {
		i++
		if c.err != nil {
			return nil, fmt.Errorf("decode audio: %w", c.err)
		}
		if len(c.pcm) < media.SampleRate*2 { // < 1 s of audio
			continue
		}
		kind := "sample"
		if !tr.sampled {
			kind = "chunk"
		}
		progress(fmt.Sprintf("listening (%s %d/%d)", kind, i, len(wins)))
		res, err := p.engine.Transcribe(ctx, c.pcm, lang)
		if err != nil {
			return nil, err
		}
		actual := float64(len(c.pcm)) / (2 * media.SampleRate)
		tr.spans = append(tr.spans, [2]float64{c.start, c.start + actual})
		tr.seconds += actual
		firstSeg := true
		for _, w := range res.Words {
			if strings.TrimSpace(w.Text) == "" || strings.HasPrefix(strings.TrimSpace(w.Text), "[") {
				continue // [BLANK_AUDIO], [Music] ...
			}
			reliable := w.Reliable
			if reliable && firstSeg && p.engine.Name() == "local" {
				// whisper pins a chunk's first segment to 0.0 when the chunk opens on silence or
				// mid-sentence, so that start time can't be trusted
				firstSeg = false
				if w.Start < 1.0 {
					reliable = false
				}
			}
			tr.words = append(tr.words, align.Word{Text: w.Text, Start: c.start + w.Start, End: c.start + w.End,
				Reliable: reliable})
		}
		// lock the language once detected so every window is transcribed consistently
		if lang == "" && res.Language != "" && len(res.Words) >= 20 {
			lang = res.Language
		}
	}
	tr.language = lang
	tr.speech = speechSpans(tr.words)
	return tr, nil
}

func speechSpans(words []align.Word) [][2]float64 {
	ws := append([]align.Word(nil), words...)
	sort.Slice(ws, func(i, j int) bool { return ws[i].Start < ws[j].Start })
	var out [][2]float64
	for _, w := range ws {
		end := math.Max(w.End, w.Start+0.2)
		if n := len(out); n > 0 && w.Start-out[n-1][1] < 0.6 {
			out[n-1][1] = math.Max(out[n-1][1], end)
			continue
		}
		out = append(out, [2]float64{w.Start, end})
	}
	return out
}

// --- writing ---------------------------------------------------------------

func (p *Processor) tmpRoot() string {
	d := filepath.Join(p.cfg.ConfigDir, "tmp")
	_ = os.MkdirAll(d, 0o755)
	return d
}

// backup copies src into /config/backups/<date>/<name>.<time>.
func (p *Processor) backup(name, src string) error {
	orig, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	bp := filepath.Join(p.cfg.ConfigDir, "backups", time.Now().Format("2006-01-02"),
		strings.TrimPrefix(name, "/")+"."+time.Now().Format("150405"))
	if err := os.MkdirAll(filepath.Dir(bp), 0o755); err != nil {
		return fmt.Errorf("backup: %w", err)
	}
	if err := os.WriteFile(bp, orig, 0o644); err != nil {
		return fmt.Errorf("backup: %w", err)
	}
	return nil
}

// matchOwner gives tmp the permissions and owner of ref (keeps Unraid's nobody:users).
func matchOwner(tmp string, ref os.FileInfo) {
	_ = os.Chmod(tmp, ref.Mode().Perm())
	if sys, ok := ref.Sys().(*syscall.Stat_t); ok {
		_ = os.Chown(tmp, int(sys.Uid), int(sys.Gid))
	}
}

// atomicWrite writes data next to dst and renames it over dst.
func atomicWrite(dst string, data []byte, ref os.FileInfo) error {
	tmp := filepath.Join(filepath.Dir(dst), "."+filepath.Base(dst)+".captionsync.tmp")
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		os.Remove(tmp)
		return err
	}
	matchOwner(tmp, ref)
	if err := os.Rename(tmp, dst); err != nil {
		os.Remove(tmp)
		return err
	}
	return nil
}

func (p *Processor) writeSidecar(t *track) error {
	st, err := os.Stat(t.path)
	if err != nil {
		return err
	}
	if p.cfg.Backup {
		if err := p.backup(t.path, t.path); err != nil {
			return err
		}
	}
	return atomicWrite(t.path, t.file.Bytes(), st)
}

// embeddedSidecarPath is where EMBEDDED_SUBS=sidecar writes a fixed embedded track,
// e.g. "Movie.eng.forced.srt".
func embeddedSidecarPath(video string, ss *media.SubStream) string {
	stem := strings.TrimSuffix(video, filepath.Ext(video))
	lang := ss.Language()
	if lang == "" {
		lang = "und"
	}
	name := stem + "." + lang
	if ss.Disposition["forced"] == 1 {
		name += ".forced"
	} else if ss.Disposition["hearing_impaired"] == 1 {
		name += ".sdh"
	}
	return name + "." + ss.Format
}

func (p *Processor) writeEmbeddedAsSidecar(video string, t *track) error {
	dst := embeddedSidecarPath(video, t.embedded)
	if _, err := os.Stat(dst); err == nil {
		return fmt.Errorf("%s already exists", filepath.Base(dst))
	}
	st, err := os.Stat(video)
	if err != nil {
		return err
	}
	return atomicWrite(dst, t.file.Bytes(), st)
}

func freeBytes(dir string) uint64 {
	var fs syscall.Statfs_t
	if err := syscall.Statfs(dir, &fs); err != nil {
		return math.MaxUint64 // unknown: don't block
	}
	return uint64(fs.Bavail) * uint64(fs.Bsize)
}

// remuxVideo rewrites the video with corrected embedded subtitles. All other streams are copied
// untouched; the new file is verified before it replaces the original.
func (p *Processor) remuxVideo(ctx context.Context, video string, pr *media.Probe, repl map[int]string) error {
	st, err := os.Stat(video)
	if err != nil {
		return err
	}
	dir := filepath.Dir(video)
	if need := uint64(float64(st.Size()) * 1.05); freeBytes(dir) < need {
		return fmt.Errorf("not enough free space in %s for a temporary copy", dir)
	}
	ext := filepath.Ext(video)
	tmp := filepath.Join(dir, "."+strings.TrimSuffix(filepath.Base(video), ext)+".captionsync.tmp"+ext)
	defer os.Remove(tmp) // no-op after a successful rename
	if err := media.Remux(ctx, video, pr, repl, tmp); err != nil {
		return err
	}
	if cur, err := os.Stat(video); err != nil || cur.Size() != st.Size() || !cur.ModTime().Equal(st.ModTime()) {
		return fmt.Errorf("video changed while remuxing, not replacing it")
	}
	matchOwner(tmp, st)
	return os.Rename(tmp, video)
}

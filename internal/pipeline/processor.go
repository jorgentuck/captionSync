// Package pipeline processes one video at a time (Processor) and runs the queue (Service).
package pipeline

import (
	"context"
	"errors"
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
	"github.com/jorgentuck/captionSync/internal/onset"
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
	// Requeue: nothing final was decided (files changed underneath us, or the engine is down);
	// check this file again later instead of recording a result.
	Requeue bool
	// EngineDown: the transcription engine itself failed; workers should back off.
	EngineDown bool
}

type track struct {
	name     string // shown in the UI
	path     string // sidecar file, or the extracted copy of an embedded track
	embedded *media.SubStream
	dst      string // EMBEDDED_SUBS=sidecar: where the fixed copy is written
	stamp    stamp  // sidecar size/mtime when it was read
	file     *subtitle.File
	cues     []align.Cue
	result   align.Result
	written  bool
	changed  int
	err      string
	short    bool // too few lines: skipped, not an error
	broken   bool // could not be read/parsed
}

// stamp identifies a file version cheaply (size + modification time).
type stamp struct {
	size int64
	mod  time.Time
	ok   bool
}

func stampOf(p string) stamp {
	st, err := os.Stat(p)
	if err != nil {
		return stamp{}
	}
	return stamp{st.Size(), st.ModTime(), true}
}

func (a stamp) same(b stamp) bool { return a.ok == b.ok && a.size == b.size && a.mod.Equal(b.mod) }

// errChanged: a file we were about to replace changed after we read it (Bazarr/Sonarr/Radarr).
var errChanged = errors.New("changed while it was being analysed")

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
//
// Nothing is written unless every file involved is still exactly as it was when it was read:
// a subtitle Bazarr replaced, or a video Sonarr/Radarr upgraded mid-run, sends the file back to
// the queue instead of being overwritten with stale content.
func (p *Processor) Process(ctx context.Context, path string, progress func(string)) Outcome {
	started := time.Now()
	sigBefore := p.lib.Signature(path)
	if sigBefore == "" {
		return Outcome{Status: store.Missing, Message: "File no longer exists"}
	}
	videoStamp := stampOf(path)
	if p.lib.SharedStem(path) {
		return Outcome{Status: store.Uncertain, Signature: sigBefore,
			Message: "Another video in this folder has the same name; its subtitle files can't be matched to one of them safely, left unchanged"}
	}
	sidecarList := p.lib.Sidecars(path)
	out := Outcome{Signature: sigBefore, Details: map[string]any{}}
	p.cleanStaleTemps(path)

	progress("probing")
	pr, err := media.ProbeFile(ctx, path)
	if err != nil {
		return fail(out, err)
	}
	dur := pr.Duration()

	// 1. collect subtitle tracks: sidecar files, plus embedded text tracks when enabled
	var tracks []*track
	for _, sp := range sidecarList {
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
		dsts := embeddedSidecarPaths(path, embText)
		var ex []media.Extract
		for i := range embText {
			ss := embText[i]
			if p.cfg.EmbeddedMode == "sidecar" {
				if _, err := os.Stat(dsts[ss.Index]); err == nil {
					continue // exported on an earlier run; that sidecar is checked like any other
				}
			}
			t := &track{name: "embedded " + ss.Label(), embedded: &ss,
				path: filepath.Join(tmp, fmt.Sprintf("s%d.%s", ss.Index, ss.Format))}
			if p.cfg.EmbeddedMode == "sidecar" {
				t.dst = dsts[ss.Index]
			}
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
		if t.embedded == nil {
			t.stamp = stampOf(t.path)
		}
		f, err := subtitle.Load(t.path)
		if err != nil {
			t.err = "could not read: " + err.Error()
			t.broken = true
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
	delay := pr.AudioDelay(audio)
	sampled := p.cfg.AnalysisMode != "full"
	tr, err := p.transcribe(ctx, path, audio, dur, delay, sampled, progress)
	if err != nil {
		return fail(out, err)
	}
	p.analyse(tracks, tr, dur, progress)
	if tr.sampled && wantsFull(tracks) {
		why := "AUTO_FULL is off"
		if p.cfg.AutoFull {
			slog.Info("sampled analysis not conclusive, running full pass", "path", path)
			full, err := p.transcribe(ctx, path, audio, dur, delay, false, progress)
			switch {
			case err == nil:
				tr = full
				p.analyse(tracks, tr, dur, progress)
			case ctx.Err() != nil || errors.Is(err, whisper.ErrEngine):
				return fail(out, err)
			default:
				why = "the full pass failed: " + err.Error()
			}
		}
		// a sampled result that asked for a full pass is never written
		for _, t := range tracks {
			if t.err != "" || !tr.sampled || !t.result.WantsFull {
				continue
			}
			if t.result.Status == "needs_fix" {
				t.result.Status = "uncertain"
				t.result.Message += " - needs a full analysis (" + why + "), left unchanged"
			} else if t.result.Status == "in_sync" {
				t.result.Message += " (sampled only; " + why + ")"
			}
		}
	}

	// 3. write fixes - only if nothing changed underneath us
	if !p.cfg.DryRun && needsWrite(tracks) {
		if what := p.changedSince(path, videoStamp, sidecarList, tracks, nil); what != "" {
			return requeue(out, what)
		}
	}
	var fixed, uncertain, failed int
	var firstShift *int64
	remux := map[int]string{}
	var remuxTracks []*track
	for _, t := range tracks {
		if t.broken {
			failed++
			continue
		}
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
				if err := p.backup(path+fmt.Sprintf(".track%d.%s", t.embedded.Index, t.embedded.Format), t.path); err != nil {
					t.err = "not written, backup failed: " + err.Error()
					failed++
					continue
				}
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
			if errors.Is(err, errChanged) {
				return requeue(out, t.name+" "+errChanged.Error())
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
	remuxed := false
	if len(remux) > 0 {
		progress("remuxing video with corrected subtitles")
		err := p.remuxVideo(ctx, path, pr, remux, videoStamp)
		if errors.Is(err, errChanged) {
			return requeue(out, "the video "+errChanged.Error())
		}
		for _, t := range remuxTracks {
			if err != nil {
				t.err = "remux failed: " + err.Error()
				failed++
			} else {
				t.written = true
				fixed++
			}
		}
		remuxed = err == nil
	}

	out.OffsetMs = firstShift
	// The stored signature must never hide a change someone else made during the run: keep the
	// pre-run one if we wrote nothing; after writing, only accept the new one if everything we
	// did not write is still as we read it.
	out.Signature = sigBefore
	if anyWritten(tracks) {
		var mine []string
		for _, t := range tracks {
			if t.written && t.dst != "" {
				mine = append(mine, t.dst)
			}
		}
		if p.changedSince(path, map[bool]stamp{true: {}, false: videoStamp}[remuxed], sidecarList, tracks, mine) == "" {
			out.Signature = p.lib.Signature(path)
		} else {
			out.Signature = "" // something else changed too: let the next scan/watch event re-check it
		}
	}
	out.Details["tracks"] = trackDetails(tracks)
	out.Details["duration_s"] = math.Round(dur)
	out.Details["audio_language"] = tr.language
	out.Details["audio_delay_s"] = math.Round(delay*1000) / 1000
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

func needsWrite(tracks []*track) bool {
	for _, t := range tracks {
		if t.err == "" && t.result.Status == "needs_fix" {
			return true
		}
	}
	return false
}

func anyWritten(tracks []*track) bool {
	for _, t := range tracks {
		if t.written {
			return true
		}
	}
	return false
}

// changedSince reports (as a description) anything that differs from what was read at the start:
// the video (unless videoStamp is zero = we replaced it ourselves), the set of sidecar files
// (ignoring `mine`, files we created), and every sidecar we did not write ourselves.
func (p *Processor) changedSince(video string, videoStamp stamp, sidecars []string, tracks []*track, mine []string) string {
	if videoStamp.ok && !stampOf(video).same(videoStamp) {
		return "the video " + errChanged.Error()
	}
	now := map[string]bool{}
	for _, s := range p.lib.Sidecars(video) {
		now[s] = true
	}
	for _, s := range mine {
		delete(now, s)
	}
	if len(now) != len(sidecars) {
		return "subtitle files were added or removed during the analysis"
	}
	for _, s := range sidecars {
		if !now[s] {
			return "subtitle files were added or removed during the analysis"
		}
	}
	for _, t := range tracks {
		if t.embedded == nil && !t.written && !stampOf(t.path).same(t.stamp) {
			return t.name + " " + errChanged.Error()
		}
	}
	return ""
}

func requeue(out Outcome, why string) Outcome {
	out.Requeue = true
	out.Message = why + "; checking it again"
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
	if errors.Is(err, whisper.ErrEngine) {
		// not this file's fault: try it again once the engine is back
		out.Requeue, out.EngineDown = true, true
	}
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

// transcribe listens to the windows (sampled) or the whole file in chunks. delay is the audio
// stream's start delay: ffmpeg's PCM for a window that begins before the audio does starts at
// the audio's first sample, not at the window start.
func (p *Processor) transcribe(ctx context.Context, path string, audio int, dur, delay float64, sampled bool,
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
		start := math.Max(c.start, delay) // where PCM sample 0 really is
		actual := float64(len(c.pcm)) / (2 * media.SampleRate)
		tr.spans = append(tr.spans, [2]float64{start, start + actual})
		tr.seconds += actual
		words := res.Words
		if res.SegmentLevel {
			refineStarts(words, c.pcm, actual)
		}
		for _, w := range words {
			tr.words = append(tr.words, align.Word{Text: w.Text, Start: start + w.Start, End: start + w.End,
				Reliable: w.Reliable})
		}
		// lock the language once detected so every window is transcribed consistently
		if lang == "" && res.Language != "" && len(res.Words) >= 20 {
			lang = config.NormalizeLang(res.Language)
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

// refineStarts replaces whisper's segment start times (often in the silence before a line, up to
// seconds early) with the speech onset measured in the audio. A segment whose onset can't be found
// clearly is no longer treated as a trustworthy timing anchor. Afterwards each segment's other
// words are spread between its (refined) start and the next segment's (refined) start, so the text
// stays in time order.
func refineStarts(words []whisper.Word, pcm []byte, length float64) {
	det := onset.New(pcm)
	var idx []int
	for i, w := range words {
		if w.Reliable {
			idx = append(idx, i)
		}
	}
	if len(idx) == 0 {
		return
	}
	// pass 1: refine every segment start against whisper's (unrefined) next start
	starts := make([]float64, len(idx))
	for k, i := range idx {
		until := length
		if k+1 < len(idx) {
			until = words[idx[k+1]].Start
		}
		t, ok := det.Refine(words[i].Start, until)
		starts[k] = words[i].Start
		switch {
		case !ok || t < 0.3:
			words[i].Reliable = false // no clear onset, or at the chunk's very first moment
		case k == 0 && words[i].Start < 1.0 && det.ActiveAtStart():
			words[i].Reliable = false // the chunk opened mid-sentence: this "start" isn't one
		default:
			starts[k] = t
		}
	}
	// pass 2: move every segment's words into [its start, next start)
	for k, i := range idx {
		next, end := len(words), length
		if k+1 < len(idx) {
			next, end = idx[k+1], starts[k+1]
		}
		n := next - i
		if n <= 0 || end <= starts[k] {
			continue
		}
		step := math.Min((end-starts[k])/float64(n), 1.0)
		for j := 0; j < n; j++ {
			words[i+j].Start = starts[k] + float64(j)*step
			words[i+j].End = words[i+j].Start + step
		}
	}
}

// --- writing ---------------------------------------------------------------

func (p *Processor) tmpRoot() string {
	d := filepath.Join(p.cfg.ConfigDir, "tmp")
	_ = os.MkdirAll(d, 0o755)
	return d
}

// cleanStaleTemps removes our own leftover temp files next to this video (from a crash or a
// killed container). Only files older than 6 hours, so a remux in progress is never touched.
func (p *Processor) cleanStaleTemps(video string) {
	dir := filepath.Dir(video)
	entries, _ := os.ReadDir(dir) // no Glob: folder names like "Movie [imdbid-tt1]" are not patterns
	for _, e := range entries {
		n := e.Name()
		if e.IsDir() || !strings.HasPrefix(n, ".") || !strings.Contains(n, ".captionsync.tmp") {
			continue
		}
		m := filepath.Join(dir, n)
		if st, err := os.Stat(m); err == nil && time.Since(st.ModTime()) > 6*time.Hour {
			if os.Remove(m) == nil {
				slog.Info("removed stale temp file", "path", m)
			}
		}
	}
}

// backup copies src into /config/backups/<date>/<name>.<time>.
func (p *Processor) backup(name, src string) error {
	orig, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	now := time.Now()
	bp := filepath.Join(p.cfg.ConfigDir, "backups", now.Format("2006-01-02"),
		strings.TrimPrefix(name, "/")+"."+now.Format("150405.000000000"))
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

func syncFile(path string) error {
	f, err := os.OpenFile(path, os.O_RDWR, 0)
	if err != nil {
		return err
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

func syncDir(dir string) {
	if d, err := os.Open(dir); err == nil {
		_ = d.Sync()
		d.Close()
	}
}

// atomicWrite writes data next to dst, flushes it to disk and renames it over dst. If check is
// given, it runs right before the rename and can veto it (e.g. dst changed meanwhile).
func atomicWrite(dst string, data []byte, ref os.FileInfo, check func() error) error {
	tmp := filepath.Join(filepath.Dir(dst), "."+filepath.Base(dst)+".captionsync.tmp")
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		os.Remove(tmp)
		return err
	}
	if err := syncFile(tmp); err != nil {
		os.Remove(tmp)
		return err
	}
	matchOwner(tmp, ref)
	if check != nil {
		if err := check(); err != nil {
			os.Remove(tmp)
			return err
		}
	}
	if err := os.Rename(tmp, dst); err != nil {
		os.Remove(tmp)
		return err
	}
	syncDir(filepath.Dir(dst))
	return nil
}

func (p *Processor) writeSidecar(t *track) error {
	st, err := os.Stat(t.path)
	if err != nil {
		return err
	}
	unchanged := func() error {
		if !stampOf(t.path).same(t.stamp) {
			return errChanged
		}
		return nil
	}
	if err := unchanged(); err != nil {
		return err
	}
	if p.cfg.Backup {
		if err := p.backup(t.path, t.path); err != nil {
			return err
		}
	}
	return atomicWrite(t.path, t.file.Bytes(), st, unchanged)
}

// embeddedSidecarPaths names the files EMBEDDED_SUBS=sidecar writes for each embedded text track,
// e.g. "Movie.eng.forced.srt". Tracks that would share a name get ".track<N>" added.
func embeddedSidecarPaths(video string, tracks []media.SubStream) map[int]string {
	stem := strings.TrimSuffix(video, filepath.Ext(video))
	out := map[int]string{}
	used := map[string]bool{}
	for _, ss := range tracks {
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
		if used[name] {
			name += fmt.Sprintf(".track%d", ss.Index)
		}
		used[name] = true
		out[ss.Index] = name + "." + ss.Format
	}
	return out
}

func (p *Processor) writeEmbeddedAsSidecar(video string, t *track) error {
	if _, err := os.Stat(t.dst); err == nil {
		return fmt.Errorf("%s already exists", filepath.Base(t.dst))
	}
	st, err := os.Stat(video)
	if err != nil {
		return err
	}
	return atomicWrite(t.dst, t.file.Bytes(), st, func() error {
		if _, err := os.Stat(t.dst); err == nil {
			return errChanged // someone created it meanwhile
		}
		return nil
	})
}

func freeBytes(dir string) uint64 {
	var fs syscall.Statfs_t
	if err := syscall.Statfs(dir, &fs); err != nil {
		return math.MaxUint64 // unknown: don't block
	}
	return uint64(fs.Bavail) * uint64(fs.Bsize)
}

// remuxVideo rewrites the video with corrected embedded subtitles. All other streams are copied
// untouched; the new file is verified before it replaces the original, and nothing is replaced if
// the video differs from the version that was analysed (videoStamp).
func (p *Processor) remuxVideo(ctx context.Context, video string, pr *media.Probe, repl map[int]string, videoStamp stamp) error {
	st, err := os.Stat(video)
	if err != nil {
		return err
	}
	if !stampOf(video).same(videoStamp) {
		return errChanged
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
	if err := syncFile(tmp); err != nil {
		return err
	}
	if !stampOf(video).same(videoStamp) {
		return errChanged
	}
	matchOwner(tmp, st)
	if err := os.Rename(tmp, video); err != nil {
		return err
	}
	syncDir(dir)
	return nil
}

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
	"sync"
	"sync/atomic"
	"time"

	"github.com/jorgentuck/captionSync/internal/config"
	"github.com/jorgentuck/captionSync/internal/library"
	"github.com/jorgentuck/captionSync/internal/store"
)

// Service owns the worker pool and the "queue things up" operations used by the API and watcher.
type Service struct {
	cfg   *config.Config
	store *store.Store
	lib   *library.Library
	proc  *Processor

	paused     atomic.Bool
	thorough   atomic.Int32    // thorough re-checks running (they're slow: one at a time)
	claimMu    sync.Mutex      // makes "claim + count thorough" one step across workers
	engineDown atomic.Int64    // unix nanos until which workers back off (engine unavailable)
	lastOK     atomic.Int64    // unix nanos of the last job the engine got through
	lastOKID   atomic.Int64    // ...and which file that was
	engFailAt  map[int64]int64 // when each file last failed the engine (guarded by mu)
	recent     []jobStat       // last finished jobs, for throughput/ETA (guarded by mu)
	wg         sync.WaitGroup

	mu      sync.Mutex
	current map[int]*Current
	scans   []*Scan
	nextID  int
}

type Current struct {
	Reason  string    `json:"reason"`
	ID      int64     `json:"id"`
	Path    string    `json:"path"`
	Stage   string    `json:"stage"`
	Started time.Time `json:"started"`
}

type Scan struct {
	ID       string     `json:"id"`
	Label    string     `json:"label"`
	Found    int        `json:"found"`
	Queued   int        `json:"queued"`
	Skipped  int        `json:"skipped"`
	Done     bool       `json:"done"`
	Error    string     `json:"error,omitempty"`
	Started  time.Time  `json:"started"`
	Finished *time.Time `json:"finished,omitempty"`
}

func NewService(cfg *config.Config, st *store.Store, lib *library.Library, proc *Processor) *Service {
	return &Service{cfg: cfg, store: st, lib: lib, proc: proc, current: map[int]*Current{}, engFailAt: map[int64]int64{}}
}

// Run starts the workers; returns immediately.
func (s *Service) Run(ctx context.Context) {
	if n, err := s.store.ResetProcessing(); err == nil && n > 0 {
		slog.Info("re-queued interrupted files", "count", n)
	}
	go s.sweep(ctx)
	for i := 0; i < s.cfg.Workers; i++ {
		s.wg.Add(1)
		go func(i int) {
			defer s.wg.Done()
			s.worker(ctx, i)
		}(i)
	}
}

// Wait blocks until the workers have stopped (after the context is cancelled), up to timeout,
// so a write or remux in progress can finish or clean up its temp files.
func (s *Service) Wait(timeout time.Duration) {
	done := make(chan struct{})
	go func() { s.wg.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(timeout):
		slog.Warn("workers did not stop in time")
	}
}

func (s *Service) worker(ctx context.Context, id int) {
	for {
		if ctx.Err() != nil {
			return
		}
		if s.paused.Load() {
			sleep(ctx, 2*time.Second)
			continue
		}
		if until := time.Unix(0, s.engineDown.Load()); time.Now().Before(until) {
			sleep(ctx, time.Until(until))
			continue
		}
		s.claimMu.Lock()
		f, err := s.store.ClaimNext(s.thorough.Load() > 0)
		if err == nil && f != nil && strings.HasPrefix(f.Reason, "thorough") {
			s.thorough.Add(1)
		}
		s.claimMu.Unlock()
		if err != nil {
			slog.Error("claim job", "err", err)
			sleep(ctx, 5*time.Second)
			continue
		}
		if f == nil {
			select {
			case <-ctx.Done():
				return
			case <-s.store.Wake():
			case <-time.After(5 * time.Second):
			}
			continue
		}
		s.handle(ctx, id, f)
	}
}

func sleep(ctx context.Context, d time.Duration) {
	select {
	case <-ctx.Done():
	case <-time.After(d):
	}
}

func (s *Service) handle(ctx context.Context, wid int, f *store.File) {
	if strings.HasPrefix(f.Reason, "thorough") {
		defer s.thorough.Add(-1) // counted when it was claimed
	}
	cur := &Current{ID: f.ID, Path: f.Path, Stage: "starting", Started: time.Now(), Reason: f.Reason}
	s.mu.Lock()
	s.current[wid] = cur
	s.mu.Unlock()
	defer func() {
		s.mu.Lock()
		delete(s.current, wid)
		s.mu.Unlock()
	}()
	progress := func(stage string) {
		s.mu.Lock()
		cur.Stage = stage
		s.mu.Unlock()
	}

	slog.Info("processing", "path", f.Path, "reason", f.Reason)
	var out Outcome
	func() {
		defer func() {
			if r := recover(); r != nil {
				out = Outcome{Status: store.Failed, Message: fmt.Sprintf("internal error: %v", r),
					Signature: s.lib.Signature(f.Path)}
			}
		}()
		out = s.proc.Process(ctx, f.Path, f.Reason, progress)
	}()
	if ctx.Err() != nil {
		return // shutting down: leave it 'processing' so it is re-queued on start
	}
	if out.Requeue {
		s.requeue(f, out)
		return
	}
	switch out.Status {
	case store.Fixed, store.InSync, store.NeedsFix, store.Uncertain:
		s.lastOK.Store(time.Now().UnixNano())
		s.lastOKID.Store(f.ID)
	}
	s.record(cur.Started, out)
	s.mu.Lock()
	delete(s.engFailAt, f.ID)
	s.mu.Unlock()
	var err error
	for i := 0; i < 3; i++ {
		if err = s.store.Finish(f.ID, out.Status, out.Message, out.Signature, out.OffsetMs, out.Details); err == nil {
			break
		}
		sleep(ctx, 2*time.Second)
	}
	if err != nil {
		slog.Error("could not save result; re-queuing", "path", f.Path, "err", err)
		if err := s.store.Requeue(f.ID, "result could not be saved: "+err.Error(), true, time.Minute); err != nil {
			slog.Error("requeue after failed save also failed; the sweeper will retry", "path", f.Path, "err", err)
		}
		return
	}
	slog.Info("done", "path", f.Path, "status", out.Status, "msg", out.Message,
		"took", time.Since(cur.Started).Round(time.Second))
}

// requeue sends a file back to the queue without a result.
//   - files changed during the run: wait a settle period, don't count it as an attempt;
//   - engine failure while the engine has been working for other files: probably this file's
//     audio (crashes/hangs whisper) - count it, so it ends up failed instead of looping;
//   - engine failure with no recent success anywhere: an outage - pause all workers, don't count.
func (s *Service) requeue(f *store.File, out Outcome) {
	countAttempt, delay := false, s.cfg.Settle
	if out.EngineDown {
		delay = 0
		// Blame the file only with evidence: the engine served a DIFFERENT file successfully after
		// this file's previous engine failure. Otherwise (first failure, or nothing has worked
		// since) treat it as an outage: pause the workers and don't count it.
		now := time.Now().UnixNano()
		s.mu.Lock()
		prev, seen := s.engFailAt[f.ID]
		s.engFailAt[f.ID] = now
		s.mu.Unlock()
		if seen && s.lastOK.Load() > prev && s.lastOKID.Load() != f.ID {
			countAttempt = true
		} else {
			s.engineDown.Store(time.Now().Add(time.Minute).UnixNano())
			slog.Warn("transcription engine unavailable; pausing workers for a minute", "err", out.Message)
		}
	}
	if err := s.store.Requeue(f.ID, out.Message, countAttempt, delay); err != nil {
		slog.Error("requeue", "err", err)
	}
	slog.Info("requeued", "path", f.Path, "why", out.Message, "counted", countAttempt)
}

// sweep re-queues rows stuck in 'processing' with no worker on them (e.g. a result that could not
// be saved while /config was full), so they don't stay unprocessable until a restart.
func (s *Service) sweep(ctx context.Context) {
	t := time.NewTicker(2 * time.Minute)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		}
		ids, err := s.store.ProcessingIDs()
		if err != nil {
			continue
		}
		s.mu.Lock()
		active := map[int64]bool{}
		for _, c := range s.current {
			active[c.ID] = true
		}
		s.mu.Unlock()
		for _, id := range ids {
			if !active[id] {
				if err := s.store.Requeue(id, "found stuck in processing; retrying", false, 0); err != nil {
					slog.Error("sweep requeue", "id", id, "err", err)
				}
			}
		}
	}
}

type jobStat struct {
	done             time.Time
	took             float64
	listened         bool // needed the speech model (not just "no subtitles" etc.)
	whisperS, audioS float64
	full, extra      bool
}

func (s *Service) record(started time.Time, out Outcome) {
	st := jobStat{done: time.Now(), took: time.Since(started).Seconds()}
	if tm, ok := out.Details["timing"].(map[string]any); ok {
		st.listened = true
		st.whisperS, _ = tm["whisper_s"].(float64)
		st.audioS, _ = tm["audio_s"].(float64)
		if ps, ok := tm["passes"].([]string); ok {
			for _, p := range ps {
				st.full = st.full || p == "chunk"
				st.extra = st.extra || p == "extra sample"
			}
		}
	}
	s.mu.Lock()
	s.recent = append(s.recent, st)
	if len(s.recent) > 300 {
		s.recent = s.recent[len(s.recent)-300:]
	}
	s.mu.Unlock()
}

// Perf summarises recent throughput: files/hour over the last hour, average time per analysed
// file, how often the extra/full passes were needed, how fast whisper ran, and an ETA.
func (s *Service) Perf(queued int) map[string]any {
	s.mu.Lock()
	defer s.mu.Unlock()
	var lastHour, listened, full, extra int
	var took, whisperS, audioS float64
	for _, j := range s.recent {
		if time.Since(j.done) < time.Hour {
			lastHour++
		}
		if j.listened {
			listened++
			took += j.took
			whisperS += j.whisperS
			audioS += j.audioS
			if j.full {
				full++
			}
			if j.extra {
				extra++
			}
		}
	}
	out := map[string]any{"files_last_hour": lastHour, "sample_size": listened}
	if listened > 0 {
		out["avg_s_per_file"] = math.Round(took / float64(listened))
		out["full_pass_pct"] = math.Round(100 * float64(full) / float64(listened))
		out["extra_pass_pct"] = math.Round(100 * float64(extra) / float64(listened))
	}
	if whisperS > 0 {
		out["whisper_realtime_x"] = math.Round(audioS/whisperS*10) / 10
	}
	// rate from the recent window (up to the last hour), so it reflects current settings
	if n := len(s.recent); n >= 3 {
		first := s.recent[0].done
		for _, j := range s.recent {
			if time.Since(j.done) < time.Hour {
				first = j.done
				break
			}
		}
		cnt := 0
		for _, j := range s.recent {
			if !j.done.Before(first) {
				cnt++
			}
		}
		if span := time.Since(first).Hours(); span > 0.02 && cnt > 1 {
			rate := float64(cnt) / span
			out["files_per_hour"] = math.Round(rate)
			if queued > 0 {
				out["eta_hours"] = math.Round(float64(queued)/rate*10) / 10
			}
		}
	}
	return out
}

func (s *Service) Pause(p bool) { s.paused.Store(p) }
func (s *Service) Paused() bool { return s.paused.Load() }

func (s *Service) Current() []Current {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]Current, 0, len(s.current))
	for _, c := range s.current {
		out = append(out, *c)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Started.Before(out[j].Started) })
	return out
}

func (s *Service) Scans() []Scan {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]Scan, 0, len(s.scans))
	for i := len(s.scans) - 1; i >= 0; i-- {
		out = append(out, *s.scans[i])
	}
	return out
}

var ErrNotAllowed = errors.New("path is outside MEDIA_ROOTS")

// Resolve maps a path (already translated with MapPath if it came from outside) to the video it
// refers to.
func (s *Service) Resolve(path string) (string, error) {
	p := filepath.Clean(path)
	if !s.lib.Allowed(p) {
		return "", ErrNotAllowed
	}
	if s.lib.IsSubtitle(p) {
		v := s.lib.VideoForSubtitle(p)
		if v == "" {
			return "", fmt.Errorf("no video found for subtitle %s", filepath.Base(p))
		}
		return v, nil
	}
	return p, nil
}

// EnqueueFile queues one video (or the video a subtitle file belongs to).
// Returns "queued", "skipped", "already" or "missing".
func (s *Service) EnqueueFile(path, reason string, force bool, priority int) (string, error) {
	p, err := s.Resolve(path)
	if err != nil {
		return "", err
	}
	if !s.lib.IsVideo(p) {
		return "", fmt.Errorf("not a video file: %s", filepath.Base(p))
	}
	if s.lib.Ignored(p) {
		return "", fmt.Errorf("ignored by IGNORE_PATTERNS: %s", p)
	}
	sig := s.lib.Signature(p)
	if sig == "" {
		s.store.MarkMissing(p)
		return "missing", nil
	}
	return s.store.Enqueue(p, sig, reason, force, priority)
}

func (s *Service) newScan(label string) *Scan {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.nextID++
	sc := &Scan{ID: fmt.Sprintf("scan-%d", s.nextID), Label: label, Started: time.Now()}
	s.scans = append(s.scans, sc)
	if len(s.scans) > 20 {
		s.scans = s.scans[len(s.scans)-20:]
	}
	return sc
}

func (s *Service) count(sc *Scan, res string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	sc.Found++
	if res == "queued" {
		sc.Queued++
	} else {
		sc.Skipped++
	}
}

func (s *Service) done(sc *Scan, err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	t := time.Now()
	sc.Done, sc.Finished = true, &t
	if err != nil {
		sc.Error = err.Error()
	}
	slog.Info("scan finished", "scan", sc.Label, "found", sc.Found, "queued", sc.Queued, "skipped", sc.Skipped)
}

// ScanPath walks a folder (or a root) in the background, queuing new/changed videos (or all, with force).
func (s *Service) ScanPath(root, reason string, force bool, priority int) (*Scan, error) {
	p := filepath.Clean(root)
	if !s.lib.Allowed(p) {
		return nil, ErrNotAllowed
	}
	if _, err := os.Stat(p); err != nil {
		return nil, err
	}
	label := reason + ": " + p
	if force {
		label += " (force)"
	}
	sc := s.newScan(label)
	go func() {
		err := s.lib.WalkVideos(p, func(v string) error {
			res, err := s.EnqueueFile(v, reason, force, priority)
			if err != nil {
				slog.Warn("enqueue", "path", v, "err", err)
			}
			s.count(sc, res)
			return nil
		})
		s.done(sc, err)
	}()
	return sc, nil
}

// ScanAll scans every media root at bulk priority, so new arrivals and hand-picked work go first.
func (s *Service) ScanAll(reason string, force bool) []*Scan {
	var out []*Scan
	for _, r := range s.cfg.MediaRoots {
		if sc, err := s.ScanPath(r, reason, force, store.PriorityBulk); err == nil {
			out = append(out, sc)
		} else {
			slog.Warn("scan root", "root", r, "err", err)
		}
	}
	return out
}

// RequeueStatus re-checks every file previously recorded with one of the given statuses.
func (s *Service) RequeueStatus(statuses []string, reason string) (*Scan, error) {
	paths, err := s.store.PathsWithStatus(statuses)
	if err != nil {
		return nil, err
	}
	sc := s.newScan(fmt.Sprintf("%s: %v", reason, statuses))
	go func() {
		for _, p := range paths {
			res, err := s.EnqueueFile(p, reason, true, store.PriorityUser)
			if err != nil {
				slog.Warn("requeue", "path", p, "err", err)
			}
			s.count(sc, res)
		}
		s.done(sc, nil)
	}()
	return sc, nil
}

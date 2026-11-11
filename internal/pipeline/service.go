package pipeline

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
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

	paused atomic.Bool

	mu      sync.Mutex
	current map[int]*Current
	scans   []*Scan
	nextID  int
}

type Current struct {
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
	return &Service{cfg: cfg, store: st, lib: lib, proc: proc, current: map[int]*Current{}}
}

// Run starts the workers; returns immediately.
func (s *Service) Run(ctx context.Context) {
	if n, err := s.store.ResetProcessing(); err == nil && n > 0 {
		slog.Info("re-queued interrupted files", "count", n)
	}
	for i := 0; i < s.cfg.Workers; i++ {
		go s.worker(ctx, i)
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
		f, err := s.store.ClaimNext()
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
	cur := &Current{ID: f.ID, Path: f.Path, Stage: "starting", Started: time.Now()}
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
		out = s.proc.Process(ctx, f.Path, progress)
	}()
	if ctx.Err() != nil {
		return // shutting down: leave it 'processing' so it is re-queued on start
	}
	if err := s.store.Finish(f.ID, out.Status, out.Message, out.Signature, out.OffsetMs, out.Details); err != nil {
		slog.Error("save result", "err", err)
	}
	slog.Info("done", "path", f.Path, "status", out.Status, "msg", out.Message,
		"took", time.Since(cur.Started).Round(time.Second))
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

// Resolve maps an incoming path (possibly from Tdarr/Bazarr) to the video it refers to.
func (s *Service) Resolve(path string) (string, error) {
	p := filepath.Clean(s.lib.MapPath(path))
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
func (s *Service) EnqueueFile(path, reason string, force bool) (string, error) {
	p, err := s.Resolve(path)
	if err != nil {
		return "", err
	}
	if !s.lib.IsVideo(p) {
		return "", fmt.Errorf("not a video file: %s", filepath.Base(p))
	}
	sig := s.lib.Signature(p)
	if sig == "" {
		s.store.MarkMissing(p)
		return "missing", nil
	}
	return s.store.Enqueue(p, sig, reason, force)
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
func (s *Service) ScanPath(root, reason string, force bool) (*Scan, error) {
	p := filepath.Clean(s.lib.MapPath(root))
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
			res, err := s.EnqueueFile(v, reason, force)
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

// ScanAll scans every media root.
func (s *Service) ScanAll(reason string, force bool) []*Scan {
	var out []*Scan
	for _, r := range s.cfg.MediaRoots {
		if sc, err := s.ScanPath(r, reason, force); err == nil {
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
			res, err := s.EnqueueFile(p, reason, true)
			if err != nil {
				slog.Warn("requeue", "path", p, "err", err)
			}
			s.count(sc, res)
		}
		s.done(sc, nil)
	}()
	return sc, nil
}

// Package watch notices new or changed videos/subtitles (Tdarr finishing a transcode, Bazarr
// downloading a subtitle) and queues them once the files have stopped changing.
package watch

import (
	"context"
	"io/fs"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/fsnotify/fsnotify"

	"github.com/jorgentuck/captionSync/internal/config"
	"github.com/jorgentuck/captionSync/internal/library"
	"github.com/jorgentuck/captionSync/internal/store"
)

type Enqueuer interface {
	EnqueueFile(path, reason string, force bool, priority int) (string, error)
}

type pending struct {
	last  time.Time
	size  int64
	mtime time.Time
}

type Watcher struct {
	cfg *config.Config
	lib *library.Library
	q   Enqueuer

	mu      sync.Mutex
	pending map[string]*pending
}

func New(cfg *config.Config, lib *library.Library, q Enqueuer) *Watcher {
	return &Watcher{cfg: cfg, lib: lib, q: q, pending: map[string]*pending{}}
}

func (w *Watcher) Pending() int {
	w.mu.Lock()
	defer w.mu.Unlock()
	return len(w.pending)
}

func (w *Watcher) Run(ctx context.Context) {
	switch w.cfg.WatchMode {
	case "off", "none", "":
		slog.Info("file watcher disabled")
		return
	case "poll":
		go w.poll(ctx)
	default:
		if err := w.inotify(ctx); err != nil {
			slog.Warn("inotify unavailable, falling back to polling", "err", err)
			go w.poll(ctx)
		}
	}
	go w.settle(ctx)
}

func (w *Watcher) relevant(p string) bool {
	return w.lib.IsVideo(p) || w.lib.IsSubtitle(p)
}

func (w *Watcher) mark(p string) {
	if !w.relevant(p) || w.lib.Ignored(p) {
		return
	}
	st, err := os.Stat(p)
	if err != nil || st.IsDir() {
		return
	}
	w.mu.Lock()
	w.pending[p] = &pending{last: time.Now(), size: st.Size(), mtime: st.ModTime()}
	w.mu.Unlock()
}

// settle queues files once they have been untouched for SETTLE_SECONDS.
func (w *Watcher) settle(ctx context.Context) {
	t := time.NewTicker(5 * time.Second)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		}
		var ready []string
		w.mu.Lock()
		for p, pe := range w.pending {
			if time.Since(pe.last) < w.cfg.Settle {
				continue
			}
			st, err := os.Stat(p)
			if err != nil {
				delete(w.pending, p)
				continue
			}
			if st.Size() != pe.size || !st.ModTime().Equal(pe.mtime) {
				pe.size, pe.mtime, pe.last = st.Size(), st.ModTime(), time.Now()
				continue // still being written
			}
			delete(w.pending, p)
			ready = append(ready, p)
		}
		w.mu.Unlock()

		for _, p := range ready {
			res, err := w.q.EnqueueFile(p, "watcher", false, store.PriorityNow)
			switch {
			case err != nil:
				slog.Debug("watcher skip", "path", p, "err", err)
			case res == "already":
				// being processed right now: check again later in case it changed meanwhile
				w.mu.Lock()
				if _, ok := w.pending[p]; !ok {
					if st, err := os.Stat(p); err == nil {
						w.pending[p] = &pending{last: time.Now(), size: st.Size(), mtime: st.ModTime()}
					}
				}
				w.mu.Unlock()
			case res == "queued":
				slog.Info("watcher queued", "path", p)
			}
		}
	}
}

// --- inotify ---------------------------------------------------------------

func (w *Watcher) inotify(ctx context.Context) error {
	fw, err := fsnotify.NewWatcher()
	if err != nil {
		return err
	}
	dirs := 0
	add := func(root string) {
		_ = filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
			if err != nil {
				return nil
			}
			if d.IsDir() {
				if p != root && (strings.HasPrefix(d.Name(), ".") || w.lib.Ignored(d.Name())) {
					return filepath.SkipDir
				}
				if err := fw.Add(p); err != nil {
					slog.Warn("cannot watch directory (raise fs.inotify.max_user_watches or use WATCH_MODE=poll)",
						"dir", p, "err", err)
					return filepath.SkipAll
				}
				dirs++
			}
			return nil
		})
	}
	for _, root := range w.cfg.WatchPaths {
		add(root)
	}
	if dirs == 0 {
		fw.Close()
		return os.ErrNotExist
	}
	slog.Info("watching for changes (inotify)", "dirs", dirs, "settle", w.cfg.Settle)

	go func() {
		defer fw.Close()
		for {
			select {
			case <-ctx.Done():
				return
			case err, ok := <-fw.Errors:
				if !ok {
					return
				}
				slog.Warn("watcher error", "err", err)
			case ev, ok := <-fw.Events:
				if !ok {
					return
				}
				if !ev.Has(fsnotify.Create) && !ev.Has(fsnotify.Write) {
					continue
				}
				if st, err := os.Stat(ev.Name); err == nil && st.IsDir() {
					if !w.lib.Ignored(ev.Name) {
						add(ev.Name)
						// files may have landed before the watch was added
						_ = filepath.WalkDir(ev.Name, func(p string, d fs.DirEntry, err error) error {
							if err == nil && !d.IsDir() {
								w.mark(p)
							}
							return nil
						})
					}
					continue
				}
				w.mark(ev.Name)
			}
		}
	}()
	return nil
}

// --- polling (for filesystems without inotify, e.g. some network shares) ----

type stamp struct {
	size  int64
	mtime time.Time
}

func (w *Watcher) snapshot() map[string]stamp {
	out := map[string]stamp{}
	for _, root := range w.cfg.WatchPaths {
		_ = filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
			if err != nil {
				return nil
			}
			if d.IsDir() {
				if p != root && (strings.HasPrefix(d.Name(), ".") || w.lib.Ignored(d.Name())) {
					return filepath.SkipDir
				}
				return nil
			}
			if w.relevant(d.Name()) {
				if info, err := d.Info(); err == nil {
					out[p] = stamp{info.Size(), info.ModTime()}
				}
			}
			return nil
		})
	}
	return out
}

func (w *Watcher) poll(ctx context.Context) {
	prev := w.snapshot()
	slog.Info("watching for changes (polling)", "files", len(prev), "interval", w.cfg.PollInterval)
	t := time.NewTicker(w.cfg.PollInterval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		}
		cur := w.snapshot()
		for p, s := range cur {
			if old, ok := prev[p]; !ok || old != s {
				w.mark(p)
			}
		}
		prev = cur
	}
}

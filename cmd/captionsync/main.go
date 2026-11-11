// Command captionsync fixes the timing of existing subtitle files (e.g. from Bazarr) by listening
// to the video's audio with a small speech model and shifting cues to match the dialogue.
package main

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"syscall"
	"time"

	"github.com/jorgentuck/captionSync/internal/config"
	"github.com/jorgentuck/captionSync/internal/library"
	"github.com/jorgentuck/captionSync/internal/pipeline"
	"github.com/jorgentuck/captionSync/internal/store"
	"github.com/jorgentuck/captionSync/internal/watch"
	"github.com/jorgentuck/captionSync/internal/web"
	"github.com/jorgentuck/captionSync/internal/whisper"
)

var version = "dev"

func main() {
	cfg := config.Load()
	level := slog.LevelInfo
	_ = level.UnmarshalText([]byte(cfg.LogLevel))
	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: level})))

	if err := os.MkdirAll(cfg.ConfigDir, 0o755); err != nil {
		slog.Error("config dir", "err", err)
		os.Exit(1)
	}
	st, err := store.Open(filepath.Join(cfg.ConfigDir, "captionsync.db"))
	if err != nil {
		slog.Error("open database", "err", err)
		os.Exit(1)
	}
	defer st.Close()

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	lib := library.New(cfg)
	engine := whisper.New(cfg)
	defer engine.Close()
	if l, ok := engine.(*whisper.Local); ok {
		go func() { // fetch the model up-front so the first job doesn't wait on it
			if err := l.EnsureModel(ctx); err != nil {
				slog.Error("whisper model", "err", err)
			}
		}()
	}

	svc := pipeline.NewService(cfg, st, lib, pipeline.NewProcessor(cfg, lib, engine))
	svc.Run(ctx)

	w := watch.New(cfg, lib, svc)
	w.Run(ctx)

	if cfg.ScanOnStart {
		svc.ScanAll("startup scan", false)
	}
	if cfg.ScanInterval > 0 {
		go func() {
			t := time.NewTicker(cfg.ScanInterval)
			defer t.Stop()
			for {
				select {
				case <-ctx.Done():
					return
				case <-t.C:
					svc.ScanAll("scheduled scan", false)
				}
			}
		}()
	}

	srv := &http.Server{
		Addr:              ":" + strconv.Itoa(cfg.Port),
		Handler:           web.New(cfg, st, lib, svc, engine, w.Pending, version).Handler(),
		ReadHeaderTimeout: 10 * time.Second,
	}
	go func() {
		<-ctx.Done()
		sctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = srv.Shutdown(sctx)
	}()
	slog.Info("captionsync started", "version", version, "port", cfg.Port, "roots", cfg.MediaRoots,
		"engine", engine.Name(), "workers", cfg.Workers, "dry_run", cfg.DryRun)
	if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		slog.Error("http server", "err", err)
	}
}

// Package web serves the JSON API and the single-page UI.
package web

import (
	"embed"
	"encoding/json"
	"errors"
	"io/fs"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/jorgentuck/captionSync/internal/config"
	"github.com/jorgentuck/captionSync/internal/library"
	"github.com/jorgentuck/captionSync/internal/pipeline"
	"github.com/jorgentuck/captionSync/internal/store"
	"github.com/jorgentuck/captionSync/internal/whisper"
)

//go:embed static
var staticFS embed.FS

type Server struct {
	cfg     *config.Config
	store   *store.Store
	lib     *library.Library
	svc     *pipeline.Service
	engine  whisper.Engine
	pending func() int
	version string
	started time.Time
}

func New(cfg *config.Config, st *store.Store, lib *library.Library, svc *pipeline.Service,
	engine whisper.Engine, pending func() int, version string) *Server {
	return &Server{cfg: cfg, store: st, lib: lib, svc: svc, engine: engine, pending: pending,
		version: version, started: time.Now()}
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	sub, _ := fs.Sub(staticFS, "static")
	mux.Handle("GET /", http.FileServer(http.FS(sub)))

	mux.HandleFunc("GET /api/health", func(w http.ResponseWriter, r *http.Request) { writeJSON(w, 200, map[string]string{"ok": "true"}) })
	mux.HandleFunc("GET /api/status", s.status)
	mux.HandleFunc("GET /api/files", s.files)
	mux.HandleFunc("GET /api/files/{id}", s.file)
	mux.HandleFunc("GET /api/browse", s.browse)
	mux.HandleFunc("POST /api/process", s.process)
	mux.HandleFunc("POST /api/process/all", s.processAll)
	mux.HandleFunc("POST /api/process/status", s.processStatus)
	mux.HandleFunc("POST /api/queue/clear", s.clearQueue)
	mux.HandleFunc("POST /api/worker/pause", func(w http.ResponseWriter, r *http.Request) {
		s.svc.Pause(true)
		writeJSON(w, 200, map[string]bool{"paused": true})
	})
	mux.HandleFunc("POST /api/worker/resume", func(w http.ResponseWriter, r *http.Request) {
		s.svc.Pause(false)
		writeJSON(w, 200, map[string]bool{"paused": false})
	})
	mux.HandleFunc("POST /api/hooks/{source}", s.hook)
	mux.HandleFunc("GET /api/hooks/{source}", s.hook)
	return logRequests(mux)
}

func logRequests(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/api/") && r.Method != http.MethodGet {
			slog.Info("api", "method", r.Method, "path", r.URL.Path)
		}
		h.ServeHTTP(w, r)
	})
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func writeErr(w http.ResponseWriter, code int, err error) {
	writeJSON(w, code, map[string]string{"error": err.Error()})
}

func readJSON(r *http.Request, v any) error {
	if r.ContentLength == 0 {
		return nil
	}
	return json.NewDecoder(http.MaxBytesReader(nil, r.Body, 1<<20)).Decode(v)
}

func (s *Server) status(w http.ResponseWriter, r *http.Request) {
	stats, err := s.store.Stats()
	if err != nil {
		writeErr(w, 500, err)
		return
	}
	queue, _ := s.store.Queue(25)
	writeJSON(w, 200, map[string]any{
		"version": s.version, "uptime_s": int(time.Since(s.started).Seconds()),
		"stats": stats, "statuses": store.AllStatuses,
		"paused": s.svc.Paused(), "current": s.svc.Current(), "queue": queue, "scans": s.svc.Scans(),
		"watcher_pending": s.pending(), "engine": s.engine.Status(), "settings": s.cfg.Public(),
	})
}

func (s *Server) files(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	limit, _ := strconv.Atoi(q.Get("limit"))
	if limit <= 0 || limit > 500 {
		limit = 100
	}
	offset, _ := strconv.Atoi(q.Get("offset"))
	rows, total, err := s.store.List(q.Get("status"), q.Get("q"), limit, offset)
	if err != nil {
		writeErr(w, 500, err)
		return
	}
	writeJSON(w, 200, map[string]any{"files": rows, "total": total})
}

func (s *Server) file(w http.ResponseWriter, r *http.Request) {
	id, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if err != nil {
		writeErr(w, 400, err)
		return
	}
	f, err := s.store.Get(id)
	if err != nil {
		writeErr(w, 500, err)
		return
	}
	if f == nil {
		writeErr(w, 404, errors.New("not found"))
		return
	}
	hist, _ := s.store.History(id, 50)
	var subs []string
	for _, p := range s.lib.Sidecars(f.Path) {
		subs = append(subs, filepath.Base(p))
	}
	writeJSON(w, 200, map[string]any{"file": f, "history": hist, "subtitles": subs})
}

type processReq struct {
	Path  string `json:"path"`
	Force bool   `json:"force"`
}

// process handles one file or one folder.
func (s *Server) process(w http.ResponseWriter, r *http.Request) {
	var req processReq
	if err := readJSON(r, &req); err != nil || strings.TrimSpace(req.Path) == "" {
		writeErr(w, 400, errors.New("path is required"))
		return
	}
	p := filepath.Clean(s.lib.MapPath(req.Path))
	st, err := os.Stat(p)
	if err != nil {
		writeErr(w, 404, err)
		return
	}
	if st.IsDir() {
		sc, err := s.svc.ScanPath(p, "manual folder", req.Force, store.PriorityUser)
		if err != nil {
			writeErr(w, 400, err)
			return
		}
		writeJSON(w, 202, map[string]any{"kind": "folder", "scan": sc.ID})
		return
	}
	res, err := s.svc.EnqueueFile(p, "manual", req.Force, store.PriorityNow)
	if err != nil {
		writeErr(w, 400, err)
		return
	}
	writeJSON(w, 200, map[string]any{"kind": "file", "result": res})
}

func (s *Server) processAll(w http.ResponseWriter, r *http.Request) {
	var req processReq
	_ = readJSON(r, &req)
	scans := s.svc.ScanAll("manual all", req.Force)
	writeJSON(w, 202, map[string]any{"scans": len(scans)})
}

func (s *Server) processStatus(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Statuses []string `json:"statuses"`
	}
	if err := readJSON(r, &req); err != nil || len(req.Statuses) == 0 {
		writeErr(w, 400, errors.New("statuses is required"))
		return
	}
	for _, st := range req.Statuses {
		if !store.IsStatus(st) || st == store.Queued || st == store.Processing {
			writeErr(w, 400, errors.New("invalid status: "+st))
			return
		}
	}
	sc, err := s.svc.RequeueStatus(req.Statuses, "recheck")
	if err != nil {
		writeErr(w, 500, err)
		return
	}
	writeJSON(w, 202, map[string]any{"scan": sc.ID})
}

func (s *Server) clearQueue(w http.ResponseWriter, r *http.Request) {
	n, err := s.store.ClearQueue()
	if err != nil {
		writeErr(w, 500, err)
		return
	}
	writeJSON(w, 200, map[string]any{"cleared": n})
}

// hook accepts calls from Tdarr flows, Bazarr post-processing, scripts, ...
// The file can be given as ?path=..., a form field or JSON: path | file | filePath | episode | subtitles | video.
func (s *Server) hook(w http.ResponseWriter, r *http.Request) {
	keys := []string{"path", "file", "filePath", "episode", "video", "subtitles"}
	var path string
	for _, k := range keys {
		if v := r.URL.Query().Get(k); v != "" {
			path = v
			break
		}
	}
	if path == "" && r.Method == http.MethodPost {
		ct := r.Header.Get("Content-Type")
		if strings.Contains(ct, "json") || ct == "" {
			var body map[string]any
			if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20)).Decode(&body); err == nil {
				for _, k := range keys {
					if v, ok := body[k].(string); ok && v != "" {
						path = v
						break
					}
				}
			}
		} else if err := r.ParseForm(); err == nil {
			for _, k := range keys {
				if v := r.PostForm.Get(k); v != "" {
					path = v
					break
				}
			}
		}
	}
	if path == "" {
		writeErr(w, 400, errors.New("no file path supplied (use path, file, episode or subtitles)"))
		return
	}
	res, err := s.svc.EnqueueFile(path, "hook:"+r.PathValue("source"), false, store.PriorityNow)
	if err != nil {
		writeErr(w, 400, err)
		return
	}
	writeJSON(w, 200, map[string]any{"result": res})
}

type entry struct {
	Name   string `json:"name"`
	Path   string `json:"path"`
	Dir    bool   `json:"dir"`
	Status string `json:"status,omitempty"`
}

func (s *Server) browse(w http.ResponseWriter, r *http.Request) {
	p := r.URL.Query().Get("path")
	if p == "" {
		var out []entry
		for _, root := range s.cfg.MediaRoots {
			out = append(out, entry{Name: root, Path: root, Dir: true})
		}
		writeJSON(w, 200, map[string]any{"path": "", "parent": "", "entries": out})
		return
	}
	p = filepath.Clean(p)
	if !s.lib.Allowed(p) {
		writeErr(w, 400, pipeline.ErrNotAllowed)
		return
	}
	items, err := os.ReadDir(p)
	if err != nil {
		writeErr(w, 404, err)
		return
	}
	out := []entry{}
	for _, it := range items {
		if strings.HasPrefix(it.Name(), ".") {
			continue
		}
		full := filepath.Join(p, it.Name())
		if it.IsDir() {
			out = append(out, entry{Name: it.Name(), Path: full, Dir: true})
		} else if s.lib.IsVideo(it.Name()) {
			e := entry{Name: it.Name(), Path: full}
			e.Status = s.store.StatusOf(full)
			out = append(out, e)
		}
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Dir != out[j].Dir {
			return out[i].Dir
		}
		return strings.ToLower(out[i].Name) < strings.ToLower(out[j].Name)
	})
	parent := filepath.Dir(p)
	if !s.lib.Allowed(parent) {
		parent = ""
	}
	writeJSON(w, 200, map[string]any{"path": p, "parent": parent, "entries": out})
}

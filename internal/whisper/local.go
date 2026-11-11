package whisper

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/jorgentuck/captionSync/internal/config"
)

const modelBaseURL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/"

// server manages one whisper.cpp server subprocess for one model on one port.
type server struct {
	cfg    *config.Config
	client *http.Client
	model  string
	port   int

	reqMu    sync.Mutex   // one request at a time (the server runs one anyway), so timeouts measure real work
	stopping atomic.Bool  // set while we stop the server on purpose
	lastExit atomic.Value // string: why the server last exited unexpectedly
	mu       sync.Mutex
	cmd      *exec.Cmd
	exited   chan struct{}
	device   string
	lastUsed time.Time
	inflight int
}

func newServer(cfg *config.Config, model string, port int) *server {
	l := &server{cfg: cfg, model: model, port: port,
		client: &http.Client{Timeout: 3 * time.Hour}} // per-request deadlines come from ctx
	go l.reaper()
	return l
}

func modelPath(cfg *config.Config, m string) string {
	if strings.HasSuffix(m, ".bin") {
		return m
	}
	return filepath.Join(cfg.ConfigDir, "models", "ggml-"+m+".bin")
}

func (l *server) modelPath() string { return modelPath(l.cfg, l.model) }

const vadBaseURL = "https://huggingface.co/ggml-org/whisper-vad/resolve/main/"

func (l *server) vadPath() string { return modelPath(l.cfg, l.cfg.WhisperVADModel) }

var dlMu sync.Mutex // serialises model downloads across all servers

// ensureModel downloads a speech model (and the VAD model) on first use. Safe to call concurrently.
func ensureModel(ctx context.Context, cfg *config.Config, model string) error {
	dlMu.Lock()
	defer dlMu.Unlock()
	if err := download(ctx, modelPath(cfg, model), modelBaseURL+"ggml-"+model+".bin", model); err != nil {
		return err
	}
	if cfg.WhisperVAD {
		return download(ctx, modelPath(cfg, cfg.WhisperVADModel), vadBaseURL+"ggml-"+cfg.WhisperVADModel+".bin", cfg.WhisperVADModel)
	}
	return nil
}

func download(ctx context.Context, path, url, name string) error {
	if _, err := os.Stat(path); err == nil {
		return nil
	}
	if strings.HasSuffix(name, ".bin") {
		return fmt.Errorf("model file %s not found", path)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	slog.Info("downloading model", "url", url)
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	resp, err := (&http.Client{Timeout: time.Hour}).Do(req)
	if err != nil {
		return fmt.Errorf("download model: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return fmt.Errorf("download model %s: HTTP %d (check the model name)", url, resp.StatusCode)
	}
	tmp := path + ".download"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	if _, err := io.Copy(f, resp.Body); err != nil {
		f.Close()
		os.Remove(tmp)
		return fmt.Errorf("download model: %w", err)
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func gpuPresent() bool {
	for _, p := range []string{"/dev/nvidiactl", "/dev/nvidia0"} {
		if _, err := os.Stat(p); err == nil {
			return true
		}
	}
	return false
}

func (l *server) binary(device string) string {
	return filepath.Join(l.cfg.WhisperBinDir, "whisper-server-"+device)
}

func (l *server) url(path string) string {
	return "http://127.0.0.1:" + strconv.Itoa(l.port) + path
}

// start launches the server; caller holds l.mu.
func (l *server) start(ctx context.Context) error {
	if l.cmd != nil {
		select {
		case <-l.exited:
			l.cmd = nil // crashed - restart it
		default:
			return nil
		}
	}
	if err := ensureModel(ctx, l.cfg, l.model); err != nil {
		return err
	}
	var devices []string
	switch l.cfg.WhisperDevice {
	case "cuda", "gpu":
		devices = []string{"cuda"}
	case "cpu":
		devices = []string{"cpu"}
	default:
		if gpuPresent() {
			devices = append(devices, "cuda")
		}
		devices = append(devices, "cpu")
	}
	var lastErr error
	for _, dev := range devices {
		bin := l.binary(dev)
		if _, err := os.Stat(bin); err != nil {
			lastErr = fmt.Errorf("%s not found", bin)
			continue
		}
		if err := l.launch(ctx, bin); err != nil {
			slog.Warn("whisper server failed to start", "device", dev, "err", err)
			lastErr = err
			continue
		}
		l.device = dev
		slog.Info("whisper server ready", "device", dev, "model", l.model, "port", l.port)
		return nil
	}
	return fmt.Errorf("could not start whisper server: %w", lastErr)
}

func (l *server) launch(ctx context.Context, bin string) error {
	args := []string{"-m", l.modelPath(), "--host", "127.0.0.1", "--port", strconv.Itoa(l.port),
		"-t", strconv.Itoa(l.cfg.WhisperThreads),
		"-nc",      // never carry context from one request (another file / window) into the next
		"-nf",      // no temperature fallback: re-decoding music/noise is slow and useless for timing
		"-bo", "1", // greedy: one candidate
	}
	if l.cfg.WhisperFlashAttn {
		args = append(args, "-fa")
	}
	if l.cfg.WhisperVAD {
		// Optional. Measured on whisper.cpp 1.7.6: VAD timestamps are sometimes perfect and
		// sometimes seconds off, depending on tiny changes to the audio, so it is off by default.
		args = append(args, "--vad", "-vm", l.vadPath())
	}
	// make sure the port is ours: otherwise the readiness probe could talk to someone else
	if ln, err := net.Listen("tcp", "127.0.0.1:"+strconv.Itoa(l.port)); err != nil {
		return fmt.Errorf("port %d is already in use (set WHISPER_PORT to a free range): %w", l.port, err)
	} else {
		ln.Close()
	}
	cmd := exec.Command(bin, args...)
	logw := &prefixWriter{}
	cmd.Stdout, cmd.Stderr = logw, logw
	if err := cmd.Start(); err != nil {
		return err
	}
	exited := make(chan struct{})
	l.stopping.Store(false)
	go func() {
		err := cmd.Wait()
		if !l.stopping.Load() {
			why := fmt.Sprintf("%v; last output: %s", err, logw.tail())
			l.lastExit.Store(why)
			slog.Warn("whisper server exited unexpectedly (will restart on next request)", "why", why)
		}
		close(exited)
	}()

	deadline := time.Now().Add(3 * time.Minute)
	for time.Now().Before(deadline) {
		select {
		case <-exited:
			return fmt.Errorf("exited during startup: %s", logw.tail())
		case <-ctx.Done():
			_ = cmd.Process.Kill()
			return ctx.Err()
		case <-time.After(500 * time.Millisecond):
		}
		req, _ := http.NewRequestWithContext(ctx, http.MethodGet, l.url("/"), nil)
		resp, err := (&http.Client{Timeout: 2 * time.Second}).Do(req)
		if err == nil {
			resp.Body.Close()
			select {
			case <-exited: // answered by something else, and ours died
				return fmt.Errorf("exited during startup: %s", logw.tail())
			default:
			}
			if resp.StatusCode == 200 {
				l.cmd, l.exited = cmd, exited
				return nil
			}
		}
	}
	_ = cmd.Process.Kill()
	return fmt.Errorf("timed out waiting for whisper server")
}

func (l *server) stop() {
	if l.cmd == nil {
		return
	}
	l.stopping.Store(true)
	_ = l.cmd.Process.Signal(os.Interrupt)
	select {
	case <-l.exited:
	case <-time.After(10 * time.Second):
		_ = l.cmd.Process.Kill()
		<-l.exited
	}
	l.cmd, l.device = nil, ""
	slog.Info("whisper server stopped (GPU memory released)")
}

func (l *server) reaper() {
	for range time.Tick(15 * time.Second) {
		l.mu.Lock()
		if l.cmd != nil && l.inflight == 0 && l.cfg.ModelIdleUnload > 0 && time.Since(l.lastUsed) > l.cfg.ModelIdleUnload {
			l.stop()
		}
		l.mu.Unlock()
	}
}

// crashed reports whether the running server process has died.
func (l *server) crashed() bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.cmd == nil {
		return true
	}
	select {
	case <-l.exited:
		return true
	default:
		return false
	}
}

// kill forcibly ends a hung server; the next request starts a fresh one.
func (l *server) kill() {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.cmd == nil {
		return
	}
	l.stopping.Store(true)
	_ = l.cmd.Process.Kill()
	<-l.exited
	l.cmd, l.device = nil, ""
}

// Transcribe sends one chunk to the server. If the server dies (out of GPU memory, a whisper.cpp
// crash) or hangs, it is restarted and the chunk retried, so one bad moment doesn't fail the file.
// Failures of the server itself are reported as ErrEngine.
func (l *server) Transcribe(ctx context.Context, pcm []byte, language string) (*Chunk, error) {
	if language == "" {
		language = "auto"
	}
	fields := [][2]string{
		// Plain segments: their start times are what we use. max_len stops whisper.cpp from cutting
		// long segments at 60 characters, which gives the extra pieces guessed mid-speech starts.
		{"response_format", "verbose_json"}, {"temperature", "0.0"}, {"language", language},
		{"max_len", "100000"},
	}
	audioSec := float64(len(pcm)) / (2 * 16000)
	timeout := time.Duration(math.Max(120, 60+3*audioSec)) * time.Second
	var lastErr error
	for attempt := 0; attempt < 3; attempt++ {
		l.mu.Lock()
		if err := l.start(ctx); err != nil {
			l.mu.Unlock()
			if ctx.Err() != nil {
				return nil, ctx.Err()
			}
			return nil, fmt.Errorf("%w: %v", ErrEngine, err)
		}
		l.inflight++
		l.lastUsed = time.Now()
		l.mu.Unlock()

		l.reqMu.Lock()
		rctx, cancel := context.WithTimeout(ctx, timeout)
		chunk, err := postTranscription(rctx, l.client, l.url("/inference"), nil, fields, pcm, 2)
		hung := rctx.Err() == context.DeadlineExceeded && ctx.Err() == nil
		cancel()
		l.reqMu.Unlock()

		l.mu.Lock()
		l.inflight--
		l.lastUsed = time.Now()
		l.mu.Unlock()
		if err == nil || ctx.Err() != nil {
			return chunk, err
		}
		lastErr = err
		switch {
		case hung:
			slog.Warn("whisper server did not answer in time; restarting it", "timeout", timeout, "attempt", attempt+1)
			l.kill()
		case l.crashed():
			why, _ := l.lastExit.Load().(string)
			slog.Warn("whisper server died during a request; restarting and retrying", "attempt", attempt+1, "why", why)
		default:
			// the server is alive and answered: a problem with this request, not an engine outage
			return nil, fmt.Errorf("whisper rejected this audio: %v", err)
		}
	}
	why, _ := l.lastExit.Load().(string)
	return nil, fmt.Errorf("%w: whisper server keeps failing (%v). Server said: %s", ErrEngine, lastErr, why)
}

// state never blocks: while the model downloads or the server starts, it says so.
func (l *server) state() (loaded bool, device string) {
	if !l.mu.TryLock() {
		return false, "starting"
	}
	defer l.mu.Unlock()
	return l.cmd != nil, l.device
}

func (l *server) Close() {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.stop()
}

// prefixWriter forwards whisper-server output to the debug log and keeps a short tail for errors.
type prefixWriter struct {
	mu  sync.Mutex
	buf []string
}

func (p *prefixWriter) Write(b []byte) (int, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	for _, line := range strings.Split(strings.TrimRight(string(b), "\n"), "\n") {
		if line = strings.TrimSpace(line); line != "" {
			slog.Debug("whisper", "out", line)
			p.buf = append(p.buf, line)
			if len(p.buf) > 8 {
				p.buf = p.buf[1:]
			}
		}
	}
	return len(b), nil
}

func (p *prefixWriter) tail() string {
	p.mu.Lock()
	defer p.mu.Unlock()
	return strings.Join(p.buf, " | ")
}

// ---------------------------------------------------------------------------
// Local: pools of servers. Requests borrow a free server, so WHISPER_INSTANCES servers transcribe
// in parallel (the GPU is far from saturated by one). An optional THOROUGH_MODEL gets one server
// of its own, used only for files re-checked in thorough mode.
// ---------------------------------------------------------------------------

type pool struct {
	servers []*server
	free    chan *server
}

type Local struct {
	cfg   *config.Config
	mu    sync.Mutex
	pools map[string]*pool
}

func NewLocal(cfg *config.Config) *Local { return &Local{cfg: cfg, pools: map[string]*pool{}} }

func (l *Local) Name() string { return "local" }

func (l *Local) pool(model string) *pool {
	l.mu.Lock()
	defer l.mu.Unlock()
	if p, ok := l.pools[model]; ok {
		return p
	}
	n, port := l.cfg.WhisperInstances, l.cfg.WhisperPort
	if model != l.cfg.WhisperModel {
		n, port = 1, l.cfg.WhisperPort+20
	}
	p := &pool{free: make(chan *server, n)}
	for i := 0; i < n; i++ {
		s := newServer(l.cfg, model, port+i)
		p.servers = append(p.servers, s)
		p.free <- s
	}
	l.pools[model] = p
	return p
}

// EnsureModel fetches the main model up front so the first job doesn't wait for the download.
func (l *Local) EnsureModel(ctx context.Context) error {
	return ensureModel(ctx, l.cfg, l.cfg.WhisperModel)
}

func (l *Local) Transcribe(ctx context.Context, pcm []byte, language string) (*Chunk, error) {
	return l.transcribeWith(ctx, l.cfg.WhisperModel, pcm, language)
}

func (l *Local) transcribeWith(ctx context.Context, model string, pcm []byte, language string) (*Chunk, error) {
	p := l.pool(model)
	select {
	case s := <-p.free:
		defer func() { p.free <- s }()
		return s.Transcribe(ctx, pcm, language)
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// Thorough returns an engine using THOROUGH_MODEL (or this one, if none is configured).
func (l *Local) Thorough() Engine {
	if l.cfg.ThoroughModel == "" || l.cfg.ThoroughModel == l.cfg.WhisperModel {
		return l
	}
	return &modelView{l: l, model: l.cfg.ThoroughModel}
}

func (l *Local) Status() map[string]any {
	l.mu.Lock()
	defer l.mu.Unlock()
	loaded, device, servers := 0, "", 0
	for _, p := range l.pools {
		for _, s := range p.servers {
			servers++
			if ok, d := s.state(); ok || d == "starting" {
				if ok {
					loaded++
				}
				device = d
			}
		}
	}
	return map[string]any{"engine": "local", "model": l.cfg.WhisperModel, "thorough_model": l.cfg.ThoroughModel,
		"instances": l.cfg.WhisperInstances, "loaded": loaded > 0, "running": loaded, "device": device}
}

func (l *Local) Close() {
	l.mu.Lock()
	defer l.mu.Unlock()
	for _, p := range l.pools {
		for _, s := range p.servers {
			s.Close()
		}
	}
}

// modelView is the Local engine pinned to another model (thorough re-checks).
type modelView struct {
	l     *Local
	model string
}

func (m *modelView) Name() string { return "local" }
func (m *modelView) Transcribe(ctx context.Context, pcm []byte, language string) (*Chunk, error) {
	return m.l.transcribeWith(ctx, m.model, pcm, language)
}
func (m *modelView) Status() map[string]any { return m.l.Status() }
func (m *modelView) Close()                 {}
func (m *modelView) Thorough() Engine       { return m }

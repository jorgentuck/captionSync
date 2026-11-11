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

// Local manages a whisper.cpp server subprocess.
type Local struct {
	cfg    *config.Config
	client *http.Client

	dlMu     sync.Mutex   // serialises model downloads
	stopping atomic.Bool  // set while we stop the server on purpose
	lastExit atomic.Value // string: why the server last exited unexpectedly
	mu       sync.Mutex
	cmd      *exec.Cmd
	exited   chan struct{}
	device   string
	lastUsed time.Time
	inflight int
}

func NewLocal(cfg *config.Config) *Local {
	l := &Local{cfg: cfg, client: &http.Client{Timeout: 3 * time.Hour}} // per-request deadlines come from ctx
	go l.reaper()
	return l
}

func (l *Local) Name() string { return "local" }

func (l *Local) modelPath() string {
	m := l.cfg.WhisperModel
	if strings.HasSuffix(m, ".bin") {
		return m
	}
	return filepath.Join(l.cfg.ConfigDir, "models", "ggml-"+m+".bin")
}

const vadBaseURL = "https://huggingface.co/ggml-org/whisper-vad/resolve/main/"

func (l *Local) vadPath() string {
	m := l.cfg.WhisperVADModel
	if strings.HasSuffix(m, ".bin") {
		return m
	}
	return filepath.Join(l.cfg.ConfigDir, "models", "ggml-"+m+".bin")
}

// EnsureModel downloads the speech model (and VAD model) on first use. Safe to call concurrently.
func (l *Local) EnsureModel(ctx context.Context) error {
	l.dlMu.Lock()
	defer l.dlMu.Unlock()
	if err := download(ctx, l.modelPath(), modelBaseURL+"ggml-"+l.cfg.WhisperModel+".bin", l.cfg.WhisperModel); err != nil {
		return err
	}
	if l.cfg.WhisperVAD {
		return download(ctx, l.vadPath(), vadBaseURL+"ggml-"+l.cfg.WhisperVADModel+".bin", l.cfg.WhisperVADModel)
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

func (l *Local) binary(device string) string {
	return filepath.Join(l.cfg.WhisperBinDir, "whisper-server-"+device)
}

func (l *Local) url(path string) string {
	return "http://127.0.0.1:" + strconv.Itoa(l.cfg.WhisperPort) + path
}

// start launches the server; caller holds l.mu.
func (l *Local) start(ctx context.Context) error {
	if l.cmd != nil {
		select {
		case <-l.exited:
			l.cmd = nil // crashed - restart it
		default:
			return nil
		}
	}
	if err := l.EnsureModel(ctx); err != nil {
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
		slog.Info("whisper server ready", "device", dev, "model", l.cfg.WhisperModel)
		return nil
	}
	return fmt.Errorf("could not start whisper server: %w", lastErr)
}

func (l *Local) launch(ctx context.Context, bin string) error {
	args := []string{"-m", l.modelPath(), "--host", "127.0.0.1", "--port", strconv.Itoa(l.cfg.WhisperPort),
		"-t", strconv.Itoa(l.cfg.WhisperThreads),
		"-nc", // never carry context from one request (another file / window) into the next
	}
	if l.cfg.WhisperVAD {
		// Optional. Measured on whisper.cpp 1.7.6: VAD timestamps are sometimes perfect and
		// sometimes seconds off, depending on tiny changes to the audio, so it is off by default.
		args = append(args, "--vad", "-vm", l.vadPath())
	}
	// make sure the port is ours: otherwise the readiness probe could talk to someone else
	if ln, err := net.Listen("tcp", "127.0.0.1:"+strconv.Itoa(l.cfg.WhisperPort)); err != nil {
		return fmt.Errorf("port %d is already in use (set WHISPER_PORT to a free port): %w", l.cfg.WhisperPort, err)
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
		resp, err := http.Get(l.url("/"))
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

func (l *Local) stop() {
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

func (l *Local) reaper() {
	for range time.Tick(15 * time.Second) {
		l.mu.Lock()
		if l.cmd != nil && l.inflight == 0 && l.cfg.ModelIdleUnload > 0 && time.Since(l.lastUsed) > l.cfg.ModelIdleUnload {
			l.stop()
		}
		l.mu.Unlock()
	}
}

// crashed reports whether the running server process has died.
func (l *Local) crashed() bool {
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
func (l *Local) kill() {
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
func (l *Local) Transcribe(ctx context.Context, pcm []byte, language string) (*Chunk, error) {
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

		rctx, cancel := context.WithTimeout(ctx, timeout)
		chunk, err := postTranscription(rctx, l.client, l.url("/inference"), nil, fields, pcm, 2)
		hung := rctx.Err() == context.DeadlineExceeded && ctx.Err() == nil
		cancel()

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
			return nil, err // an answer from a healthy server
		}
	}
	why, _ := l.lastExit.Load().(string)
	return nil, fmt.Errorf("%w: whisper server keeps failing (%v). Server said: %s", ErrEngine, lastErr, why)
}

// Status never blocks: while the model downloads or the server starts, it says so.
func (l *Local) Status() map[string]any {
	if !l.mu.TryLock() {
		return map[string]any{"engine": "local", "model": l.cfg.WhisperModel, "loaded": false, "device": "starting"}
	}
	defer l.mu.Unlock()
	return map[string]any{"engine": "local", "model": l.cfg.WhisperModel, "loaded": l.cmd != nil, "device": l.device}
}

func (l *Local) Close() {
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

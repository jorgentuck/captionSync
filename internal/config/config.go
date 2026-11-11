// Package config reads all runtime settings from environment variables.
package config

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

type Config struct {
	MediaRoots []string
	ConfigDir  string
	Port       int
	LogLevel   string

	// Transcription
	Transcriber     string // local | openai
	WhisperModel    string // ggml model name (e.g. small-q5_1) or path to a .bin
	WhisperDevice   string // auto | cuda | cpu
	WhisperThreads  int
	WhisperVAD      bool
	WhisperVADModel string
	WhisperBinDir   string
	WhisperPort     int
	AudioLanguage   string // "" = auto-detect
	ModelIdleUnload time.Duration
	OpenAIBaseURL   string
	OpenAIKey       string
	OpenAIModel     string

	// Analysis
	AnalysisMode       string // sample | full
	SampleWindows      int
	SampleWindowSec    int
	FullChunkSec       int
	AutoFull           bool
	MaxOffset          float64
	SyncThresholdMs    int
	MinAnchors         int
	MatchThreshold     int
	AllowActivity      bool
	PreferredAudioLang []string

	// Output
	EmbeddedMode string // report | remux | sidecar | off
	DryRun       bool
	Backup       bool
	Workers      int

	// Watching
	WatchMode      string // inotify | poll | off
	WatchPaths     []string
	Settle         time.Duration
	PollInterval   time.Duration
	ScanInterval   time.Duration
	ScanOnStart    bool
	IgnorePatterns []string
	VideoExt       []string
	SubExt         []string
	PathMappings   [][2]string
}

var (
	defaultVideoExt = []string{".mkv", ".mp4", ".m4v", ".avi", ".mov", ".ts", ".m2ts", ".wmv", ".webm"}
	defaultSubExt   = []string{".srt", ".ass", ".ssa", ".vtt"}
	defaultIgnore   = []string{"*TdarrCache*", "*.partial*", "*.tmp", "*-sample.*", "*.sample.*", ".*"}
	// InternalIgnore matches our own temporary files; always applied.
	InternalIgnore = []string{"*.captionsync.tmp*"}
)

func str(name, def string) string {
	if v := strings.TrimSpace(os.Getenv(name)); v != "" {
		return v
	}
	return def
}

func boolean(name string, def bool) bool {
	v := strings.ToLower(strings.TrimSpace(os.Getenv(name)))
	if v == "" {
		return def
	}
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

func integer(name string, def int) int {
	if v, err := strconv.Atoi(strings.TrimSpace(os.Getenv(name))); err == nil {
		return v
	}
	return def
}

func float(name string, def float64) float64 {
	if v, err := strconv.ParseFloat(strings.TrimSpace(os.Getenv(name)), 64); err == nil {
		return v
	}
	return def
}

func list(name string, def []string) []string {
	v := strings.TrimSpace(os.Getenv(name))
	if v == "" {
		return append([]string(nil), def...)
	}
	var out []string
	for _, p := range strings.Split(v, ",") {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}

func cleanPaths(ps []string) []string {
	out := make([]string, 0, len(ps))
	for _, p := range ps {
		out = append(out, filepath.Clean(p))
	}
	return out
}

func exts(ps []string) []string {
	out := make([]string, 0, len(ps))
	for _, e := range ps {
		e = strings.ToLower(e)
		if !strings.HasPrefix(e, ".") {
			e = "." + e
		}
		out = append(out, e)
	}
	return out
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// Load builds the configuration from the environment.
func Load() *Config {
	roots := cleanPaths(list("MEDIA_ROOTS", []string{"/media"}))
	c := &Config{
		MediaRoots: roots,
		ConfigDir:  str("CONFIG_DIR", "/config"),
		Port:       integer("PORT", 8080),
		LogLevel:   strings.ToUpper(str("LOG_LEVEL", "INFO")),

		Transcriber:     strings.ToLower(str("TRANSCRIBER", "local")),
		WhisperModel:    str("WHISPER_MODEL", "small-q5_1"),
		WhisperDevice:   strings.ToLower(str("WHISPER_DEVICE", "auto")),
		WhisperThreads:  integer("WHISPER_THREADS", 4),
		WhisperVAD:      boolean("WHISPER_VAD", false), // whisper.cpp 1.7.6 VAD timestamps are unstable
		WhisperVADModel: str("WHISPER_VAD_MODEL", "silero-v5.1.2"),
		WhisperBinDir:   str("WHISPER_BIN_DIR", "/usr/local/bin"),
		WhisperPort:     integer("WHISPER_PORT", 8178),
		AudioLanguage:   str("AUDIO_LANGUAGE", ""),
		ModelIdleUnload: time.Duration(integer("MODEL_IDLE_UNLOAD", 300)) * time.Second,
		OpenAIBaseURL:   strings.TrimRight(str("OPENAI_BASE_URL", "https://api.openai.com/v1"), "/"),
		OpenAIKey:       str("OPENAI_API_KEY", ""),
		OpenAIModel:     str("OPENAI_MODEL", "whisper-1"),

		AnalysisMode:       strings.ToLower(str("ANALYSIS_MODE", "sample")),
		SampleWindows:      max(3, integer("SAMPLE_WINDOWS", 12)),
		SampleWindowSec:    max(45, integer("SAMPLE_WINDOW_SECONDS", 60)), // shorter windows give poor timestamps
		FullChunkSec:       max(60, integer("FULL_CHUNK_SECONDS", 300)),
		AutoFull:           boolean("AUTO_FULL", true),
		MaxOffset:          float("MAX_OFFSET_SECONDS", 120),
		SyncThresholdMs:    integer("SYNC_THRESHOLD_MS", 300),
		MinAnchors:         integer("MIN_ANCHORS", 8),
		MatchThreshold:     integer("MATCH_THRESHOLD", 80),
		AllowActivity:      boolean("ALLOW_ACTIVITY_FALLBACK", true),
		PreferredAudioLang: list("PREFERRED_AUDIO_LANG", nil),

		EmbeddedMode: strings.ToLower(str("EMBEDDED_SUBS", "report")),
		DryRun:       boolean("DRY_RUN", false),
		Backup:       boolean("BACKUP", true),
		Workers:      max(1, integer("WORKERS", 2)),

		WatchMode:      strings.ToLower(str("WATCH_MODE", "inotify")),
		Settle:         time.Duration(integer("SETTLE_SECONDS", 90)) * time.Second,
		PollInterval:   time.Duration(max(10, integer("POLL_INTERVAL", 60))) * time.Second,
		ScanInterval:   time.Duration(integer("SCAN_INTERVAL_MINUTES", 0)) * time.Minute,
		ScanOnStart:    boolean("SCAN_ON_START", false),
		IgnorePatterns: append(list("IGNORE_PATTERNS", defaultIgnore), InternalIgnore...),
		VideoExt:       exts(list("VIDEO_EXTENSIONS", defaultVideoExt)),
		SubExt:         exts(list("SUBTITLE_EXTENSIONS", defaultSubExt)),
	}
	c.WatchPaths = cleanPaths(list("WATCH_PATHS", roots))
	for i, l := range c.PreferredAudioLang {
		c.PreferredAudioLang[i] = strings.ToLower(l)
	}
	// PATH_MAPPINGS=/mnt/user/media:/media  (path as Tdarr sees it -> path inside this container)
	for _, m := range list("PATH_MAPPINGS", nil) {
		if i := strings.Index(m, ":"); i > 0 {
			c.PathMappings = append(c.PathMappings, [2]string{
				strings.TrimRight(m[:i], "/"), strings.TrimRight(m[i+1:], "/")})
		}
	}
	return c
}

// Public returns settings safe to show in the UI.
func (c *Config) Public() map[string]any {
	key := ""
	if c.OpenAIKey != "" {
		key = "***"
	}
	return map[string]any{
		"media_roots": c.MediaRoots, "transcriber": c.Transcriber, "whisper_model": c.WhisperModel,
		"whisper_device": c.WhisperDevice, "whisper_vad": c.WhisperVAD, "audio_language": c.AudioLanguage,
		"model_idle_unload_s": c.ModelIdleUnload.Seconds(), "openai_base_url": c.OpenAIBaseURL,
		"openai_model": c.OpenAIModel, "openai_api_key": key,
		"analysis_mode": c.AnalysisMode, "sample_windows": c.SampleWindows,
		"sample_window_seconds": c.SampleWindowSec, "auto_full": c.AutoFull,
		"max_offset_s": c.MaxOffset, "sync_threshold_ms": c.SyncThresholdMs, "min_anchors": c.MinAnchors,
		"match_threshold": c.MatchThreshold, "allow_activity_fallback": c.AllowActivity,
		"embedded_subs": c.EmbeddedMode, "dry_run": c.DryRun, "backup": c.Backup, "workers": c.Workers,
		"watch_mode": c.WatchMode, "watch_paths": c.WatchPaths, "settle_s": c.Settle.Seconds(),
		"scan_interval_min": c.ScanInterval.Minutes(), "ignore_patterns": c.IgnorePatterns,
		"path_mappings": c.PathMappings,
	}
}

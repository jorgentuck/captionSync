package whisper

import (
	"context"
	"net/http"
	"time"

	"github.com/jorgentuck/captionSync/internal/config"
)

// OpenAI talks to any OpenAI-compatible speech-to-text endpoint.
type OpenAI struct {
	cfg    *config.Config
	client *http.Client
}

func NewOpenAI(cfg *config.Config) *OpenAI {
	return &OpenAI{cfg: cfg, client: &http.Client{Timeout: 10 * time.Minute}}
}

func (o *OpenAI) Name() string { return "openai" }

func (o *OpenAI) Transcribe(ctx context.Context, pcm []byte, language string) (*Chunk, error) {
	fields := [][2]string{
		{"model", o.cfg.OpenAIModel}, {"response_format", "verbose_json"},
		{"timestamp_granularities[]", "word"}, {"timestamp_granularities[]", "segment"},
	}
	if len(language) == 2 { // ISO-639-1 only; the API reports names like "english"
		fields = append(fields, [2]string{"language", language})
	}
	headers := map[string]string{}
	if o.cfg.OpenAIKey != "" {
		headers["Authorization"] = "Bearer " + o.cfg.OpenAIKey
	}
	return postTranscription(ctx, o.client, o.cfg.OpenAIBaseURL+"/audio/transcriptions", headers, fields, pcm, 4)
}

func (o *OpenAI) Status() map[string]any {
	return map[string]any{"engine": "openai", "model": o.cfg.OpenAIModel, "base_url": o.cfg.OpenAIBaseURL}
}

func (o *OpenAI) Close() {}

func (o *OpenAI) Thorough() Engine { return o }

func New(cfg *config.Config) Engine {
	if cfg.Transcriber == "openai" {
		return NewOpenAI(cfg)
	}
	return NewLocal(cfg)
}

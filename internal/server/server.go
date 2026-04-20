package server

import (
	"context"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"time"

	"github.com/ThatCatDev/tanrenai-gpu/internal/config"
	"github.com/ThatCatDev/tanrenai-gpu/internal/gguf"
	"github.com/ThatCatDev/tanrenai-gpu/internal/models"
	"github.com/ThatCatDev/tanrenai-gpu/internal/runner"
	"github.com/ThatCatDev/tanrenai-gpu/internal/training"
)

// Server is the tanrenai GPU server — pure inference + training API.
type Server struct {
	cfg             *config.Config
	http            *http.Server
	store           *models.Store
	runner          runner.Runner
	embeddingRunner *EmbeddingSubprocess
	trainingManager *training.Manager
	templateCleanup func() // removes temp template file on model switch/shutdown
}

// EmbeddingSubprocess wraps an embedding server subprocess.
type EmbeddingSubprocess struct {
	Sub     *runner.Subprocess
	BaseURL string
}

// New creates a new GPU Server.
func New(cfg *config.Config) *Server {
	s := &Server{
		cfg:   cfg,
		store: models.NewStore(cfg.ModelsDir),
	}

	mux := http.NewServeMux()
	s.registerRoutes(mux)

	s.http = &http.Server{
		Addr:    fmt.Sprintf("%s:%d", cfg.Host, cfg.Port),
		Handler: withLogging(withCORS(mux)),
	}

	return s
}

// Start starts the server and blocks until the context is cancelled.
func (s *Server) Start(ctx context.Context) error {
	ln, err := net.Listen("tcp", s.http.Addr)
	if err != nil {
		return fmt.Errorf("listen: %w", err)
	}

	slog.Info("Tanrenai GPU server listening", "addr", s.http.Addr)
	slog.Info("directories configured", "models_dir", s.cfg.ModelsDir, "bin_dir", s.cfg.BinDir)

	errCh := make(chan error, 1)
	go func() {
		errCh <- s.http.Serve(ln)
	}()

	select {
	case <-ctx.Done():
		slog.Info("shutting down GPU server")
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := s.http.Shutdown(shutdownCtx); err != nil {
			slog.Error("server shutdown error", "error", err)
		}
		if s.runner != nil {
			_ = s.runner.Close()
		}
		if s.embeddingRunner != nil {
			_ = s.embeddingRunner.Sub.GracefulStop()
		}
		if s.templateCleanup != nil {
			s.templateCleanup()
		}

		return nil
	case err := <-errCh:
		return err
	}
}

// SetTrainingManager sets the training manager for fine-tuning API endpoints.
func (s *Server) SetTrainingManager(m *training.Manager) {
	s.trainingManager = m
}

// SetEmbeddingRunner sets the embedding subprocess for the /v1/embeddings endpoint.
func (s *Server) SetEmbeddingRunner(er *EmbeddingSubprocess) {
	s.embeddingRunner = er
}

// StartEmbeddingSubprocess resolves the model and spawns a llama-server in embedding mode.
func (s *Server) StartEmbeddingSubprocess(ctx context.Context, modelName string) (*EmbeddingSubprocess, error) {
	modelPath, err := s.store.Resolve(modelName)
	if err != nil {
		return nil, err
	}

	args := []string{
		"--model", modelPath,
		"--embedding",
		"--ctx-size", "512",
		"--host", "127.0.0.1",
		"--n-gpu-layers", "999",
	}

	sub, err := runner.NewSubprocess(runner.SubprocessConfig{
		BinDir:        s.cfg.BinDir,
		Args:          args,
		Label:         "embedding",
		HealthTimeout: 60 * time.Second,
	})
	if err != nil {
		return nil, err
	}

	if err := sub.Start(ctx); err != nil {
		return nil, err
	}

	slog.Info("embedding server ready", "url", sub.BaseURL(), "model", modelName)

	return &EmbeddingSubprocess{Sub: sub, BaseURL: sub.BaseURL()}, nil
}

// LoadResult contains information about a loaded model.
type LoadResult struct {
	CtxSize int
}

// LoadModel loads a model by name into the runner.
func (s *Server) LoadModel(ctx context.Context, modelName string) (*LoadResult, error) {
	modelPath, err := s.store.Resolve(modelName)
	if err != nil {
		return nil, err
	}

	// Close existing runner if any
	if s.runner != nil {
		_ = s.runner.Close()
		s.runner = nil
	}

	// Clean up any previous auto-detected template.
	if s.templateCleanup != nil {
		s.templateCleanup()
		s.templateCleanup = nil
	}

	// Read GGUF metadata once for both template and context length detection.
	var meta *gguf.Metadata
	meta, err = gguf.ReadMetadata(modelPath)
	if err != nil {
		slog.Warn("could not read GGUF metadata", "error", err)
		meta = nil
	}

	r := runner.NewProcessRunner()
	opts := runner.DefaultOptions()
	opts.BinDir = s.cfg.BinDir
	opts.GPULayers = s.cfg.GPULayers
	opts.CtxSize = s.cfg.CtxSize
	opts.ChatTemplateFile = s.cfg.ChatTemplateFile
	opts.FlashAttention = s.cfg.FlashAttention
	opts.ReasoningFormat = s.cfg.ReasoningFormat
	opts.CPUMoE = s.cfg.CPUMoE
	opts.CPUMoELayers = s.cfg.CPUMoELayers
	opts.NoKVOffload = s.cfg.NoKVOffload
	opts.FitVRAM = s.cfg.FitVRAM
	opts.TensorSplit = s.cfg.TensorSplit
	opts.SplitMode = s.cfg.SplitMode
	opts.OverrideTensor = s.cfg.OverrideTensor

	// Auto-detect context length from GGUF when config uses the default.
	if meta != nil && meta.Architecture.ContextLength > 0 && opts.CtxSize == runner.DefaultOptions().CtxSize {
		opts.CtxSize = int(meta.Architecture.ContextLength)
		slog.Info("auto-detected context length from GGUF", "ctx_size", opts.CtxSize)
	}

	// Log MoE architecture when detected. We don't auto-enable any
	// special offload flags here — the default path (`--n-gpu-layers
	// 999`) works fine on cards large enough to hold the model (for
	// a 65 GB Q4 on an 80 GB A100, there's plenty of headroom).
	//
	// Two earlier attempts were both wrong:
	//   - `--fit on`: llama.cpp's heuristic counts total MoE params
	//     (122B) instead of active (10B), decides it won't fit, and
	//     offloads zero layers — model ends up on CPU.
	//   - `--cpu-moe`: in this llama.cpp version it also pushes the
	//     KV cache to CPU, killing inference speed.
	//
	// Users who genuinely don't have enough VRAM can opt in via
	// `cpu_moe` or `gpu_layers` per request.
	if meta != nil && meta.Architecture.ExpertCount > 0 {
		slog.Info("MoE model detected",
			"experts", meta.Architecture.ExpertCount,
			"active", meta.Architecture.ExpertUsedCount)
	}

	// Auto-detect chat template from GGUF metadata when no explicit template is set.
	if opts.ChatTemplateFile == "" && !s.cfg.NoAutoTemplate {
		if res, err := runner.ResolveTemplate(modelPath, meta); err != nil {
			slog.Warn("template auto-detection failed", "error", err)
		} else if res != nil {
			opts.ChatTemplateFile = res.TemplatePath
			s.templateCleanup = res.Cleanup
			slog.Info("auto-detected chat template", "source", res.Source)
		}
	}

	// Always enable reasoning format — it's harmless for non-thinking models
	// but required for thinking models where <think> tokens would
	// silently consume the output budget without producing visible content.
	if opts.ReasoningFormat == "" {
		opts.ReasoningFormat = "deepseek"
	}

	if err := r.Load(ctx, modelPath, opts); err != nil {
		return nil, err
	}

	s.runner = r

	return &LoadResult{CtxSize: opts.CtxSize}, nil
}

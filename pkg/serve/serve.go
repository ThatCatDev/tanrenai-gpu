// Package serve provides a public API for starting the tanrenai GPU server in-process.
package serve

import (
	"context"
	"fmt"
	"os"

	"github.com/ThatCatDev/tanrenai-gpu/internal/config"
	"github.com/ThatCatDev/tanrenai-gpu/internal/models"
	"github.com/ThatCatDev/tanrenai-gpu/internal/runner"
	"github.com/ThatCatDev/tanrenai-gpu/internal/server"
)

// Config holds the public configuration for the GPU server.
type Config struct {
	Host             string
	Port             int
	ModelsDir        string
	BinDir           string
	GPULayers        int
	CtxSize          int
	FlashAttention   bool
	EmbeddingModel   string
	ChatTemplate     string // named template shortcut (e.g. "chatml")
	ChatTemplateFile string // path to custom Jinja chat template file
	ReasoningFormat  string // reasoning format (e.g. "deepseek")
	NoAutoTemplate   bool   // disable automatic template detection from GGUF metadata
	CPUMoE           bool   // keep all MoE expert weights on CPU
	CPUMoELayers     int    // keep first N layers' MoE experts on CPU
	NoKVOffload      bool   // keep KV cache on CPU to save VRAM
	FitVRAM          bool   // auto-adjust parameters to fit device memory
	TensorSplit      string // per-GPU VRAM fractions
	SplitMode        string // multi-GPU split strategy
	OverrideTensor   string // fine-grained tensor buffer type overrides
}

// Start starts the GPU server and blocks until ctx is cancelled.
func Start(ctx context.Context, cfg Config) error {
	icfg := config.DefaultConfig()
	if cfg.Host != "" {
		icfg.Host = cfg.Host
	}
	if cfg.Port != 0 {
		icfg.Port = cfg.Port
	}
	if cfg.ModelsDir != "" {
		icfg.ModelsDir = cfg.ModelsDir
	}
	if cfg.BinDir != "" {
		icfg.BinDir = cfg.BinDir
	}
	if cfg.GPULayers != 0 {
		icfg.GPULayers = cfg.GPULayers
	}
	if cfg.CtxSize != 0 {
		icfg.CtxSize = cfg.CtxSize
	}
	icfg.FlashAttention = cfg.FlashAttention
	if cfg.EmbeddingModel != "" {
		icfg.EmbeddingModel = cfg.EmbeddingModel
	}
	if cfg.ReasoningFormat != "" {
		icfg.ReasoningFormat = cfg.ReasoningFormat
	}
	icfg.NoAutoTemplate = cfg.NoAutoTemplate
	icfg.CPUMoE = cfg.CPUMoE
	icfg.CPUMoELayers = cfg.CPUMoELayers
	icfg.NoKVOffload = cfg.NoKVOffload
	icfg.FitVRAM = cfg.FitVRAM
	icfg.TensorSplit = cfg.TensorSplit
	icfg.SplitMode = cfg.SplitMode
	icfg.OverrideTensor = cfg.OverrideTensor

	// Handle chat template: explicit file takes precedence, then named template.
	if cfg.ChatTemplateFile != "" {
		icfg.ChatTemplateFile = cfg.ChatTemplateFile
	} else if cfg.ChatTemplate != "" {
		switch cfg.ChatTemplate {
		case "chatml":
			tpl := runner.GenerateChatML(runner.DefaultChatMLConfig)
			path, err := runner.WriteTemplateFile("chatml", tpl)
			if err != nil {
				return fmt.Errorf("write chat template: %w", err)
			}
			defer func() { _ = os.Remove(path) }()
			icfg.ChatTemplateFile = path
		default:
			return fmt.Errorf("unknown chat template %q (available: chatml; or use ChatTemplateFile for custom templates)", cfg.ChatTemplate)
		}
	}

	if err := config.EnsureDirs(); err != nil {
		return fmt.Errorf("ensure dirs: %w", err)
	}

	srv := server.New(icfg)

	// Start embedding subprocess if configured.
	if icfg.EmbeddingModel != "" {
		er, err := srv.StartEmbeddingSubprocess(ctx, icfg.EmbeddingModel)
		if err != nil {
			return fmt.Errorf("embedding subprocess: %w", err)
		}
		srv.SetEmbeddingRunner(er)
	}

	return srv.Start(ctx)
}

// ModelsDir returns the default directory where models are stored.
func ModelsDir() string {
	return config.ModelsDir()
}

// DownloadProgress is called periodically during a model download.
type DownloadProgress = models.DownloadProgress

// DownloadModel downloads a GGUF model from a URL to the models directory.
func DownloadModel(url, destDir string, progress DownloadProgress) (string, error) {
	return models.Download(url, destDir, progress)
}

// ResolveModel resolves a model name to its file path in the models directory.
func ResolveModel(name string) (string, error) {
	store := models.NewStore(config.ModelsDir())

	return store.Resolve(name)
}

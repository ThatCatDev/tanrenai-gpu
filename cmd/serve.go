package cmd

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/ThatCatDev/tanrenai-gpu/internal/config"
	"github.com/ThatCatDev/tanrenai-gpu/internal/runner"
	"github.com/ThatCatDev/tanrenai-gpu/internal/server"
	"github.com/spf13/cobra"
)

var serveCmd = &cobra.Command{
	Use:   "serve",
	Short: "Start the tanrenai GPU API server",
	RunE: func(cmd *cobra.Command, args []string) error {
		cfg := config.DefaultConfig()

		if host, _ := cmd.Flags().GetString("host"); host != "" {
			cfg.Host = host
		}
		if port, _ := cmd.Flags().GetInt("port"); port != 0 {
			cfg.Port = port
		}
		if dir, _ := cmd.Flags().GetString("models-dir"); dir != "" {
			cfg.ModelsDir = dir
		}
		if gpu, _ := cmd.Flags().GetInt("gpu-layers"); cmd.Flags().Changed("gpu-layers") {
			cfg.GPULayers = gpu
		}
		if ctx, _ := cmd.Flags().GetInt("ctx-size"); ctx != 0 {
			cfg.CtxSize = ctx
		}
		if tpl, _ := cmd.Flags().GetString("chat-template-file"); tpl != "" {
			cfg.ChatTemplateFile = tpl
		}
		// Named template shortcut — generates a generic ChatML template.
		if name, _ := cmd.Flags().GetString("chat-template"); name != "" && cfg.ChatTemplateFile == "" {
			switch name {
			case "chatml":
				// OK
			default:
				return fmt.Errorf("unknown chat template %q (available: chatml; or use --chat-template-file for custom templates)", name)
			}
			tpl := runner.GenerateChatML(runner.DefaultChatMLConfig)
			path, err := runner.WriteTemplateFile("chatml", tpl)
			if err != nil {
				return fmt.Errorf("failed to write chat template: %w", err)
			}
			defer func() { _ = os.Remove(path) }()
			cfg.ChatTemplateFile = path
			_, _ = fmt.Fprintf(os.Stdout, "Using %s chat template (generated)\n", name)
		}

		if embModel, _ := cmd.Flags().GetString("embedding-model"); embModel != "" {
			cfg.EmbeddingModel = embModel
		}

		if rf, _ := cmd.Flags().GetString("reasoning-format"); rf != "" {
			cfg.ReasoningFormat = rf
		}
		if cmd.Flags().Changed("flash-attn") {
			fa, _ := cmd.Flags().GetBool("flash-attn")
			cfg.FlashAttention = fa
		}
		if cmd.Flags().Changed("no-auto-template") {
			nat, _ := cmd.Flags().GetBool("no-auto-template")
			cfg.NoAutoTemplate = nat
		}

		// MoE / multi-GPU flags
		if cmd.Flags().Changed("cpu-moe") {
			cfg.CPUMoE, _ = cmd.Flags().GetBool("cpu-moe")
		}
		if n, _ := cmd.Flags().GetInt("n-cpu-moe"); n > 0 {
			cfg.CPUMoELayers = n
		}
		if cmd.Flags().Changed("no-kv-offload") {
			cfg.NoKVOffload, _ = cmd.Flags().GetBool("no-kv-offload")
		}
		if cmd.Flags().Changed("fit") {
			cfg.FitVRAM, _ = cmd.Flags().GetBool("fit")
		}
		if ts, _ := cmd.Flags().GetString("tensor-split"); ts != "" {
			cfg.TensorSplit = ts
		}
		if sm, _ := cmd.Flags().GetString("split-mode"); sm != "" {
			cfg.SplitMode = sm
		}
		if ot, _ := cmd.Flags().GetString("override-tensor"); ot != "" {
			cfg.OverrideTensor = ot
		}

		if err := config.EnsureDirs(); err != nil {
			return err
		}

		ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
		defer stop()

		srv := server.New(cfg)

		// Start embedding subprocess if configured
		if cfg.EmbeddingModel != "" {
			er, err := srv.StartEmbeddingSubprocess(ctx, cfg.EmbeddingModel)
			if err != nil {
				return fmt.Errorf("embedding subprocess: %w", err)
			}
			srv.SetEmbeddingRunner(er)
		}

		return srv.Start(ctx)
	},
}

func init() {
	serveCmd.Flags().String("host", "127.0.0.1", "bind address")
	serveCmd.Flags().Int("port", 11435, "listen port")
	serveCmd.Flags().Int("gpu-layers", -1, "GPU layers to offload (-1 = auto)")
	serveCmd.Flags().Int("ctx-size", 4096, "context window size")
	serveCmd.Flags().String("chat-template", "", "named chat template (e.g. chatml)")
	serveCmd.Flags().String("chat-template-file", "", "path to custom Jinja chat template file")
	serveCmd.Flags().String("embedding-model", "", "embedding model name (e.g. nomic-embed-text)")
	serveCmd.Flags().String("reasoning-format", "", "reasoning format for thinking mode (e.g. deepseek)")
	serveCmd.Flags().Bool("flash-attn", true, "enable flash attention")
	serveCmd.Flags().Bool("no-auto-template", false, "disable automatic chat template detection from GGUF metadata")
	serveCmd.Flags().Bool("cpu-moe", false, "keep all MoE expert weights on CPU")
	serveCmd.Flags().Int("n-cpu-moe", 0, "keep first N layers' MoE experts on CPU")
	serveCmd.Flags().Bool("no-kv-offload", false, "keep KV cache on CPU to save VRAM")
	serveCmd.Flags().Bool("fit", false, "auto-adjust parameters to fit device memory")
	serveCmd.Flags().String("tensor-split", "", "per-GPU VRAM fractions (e.g. 0.7,0.3)")
	serveCmd.Flags().String("split-mode", "", "multi-GPU split: none, layer, row")
	serveCmd.Flags().String("override-tensor", "", "fine-grained tensor buffer type overrides")
	rootCmd.AddCommand(serveCmd)
}

package runner

import "time"

// Options configures how a runner loads and serves a model.
type Options struct {
	// Port for the llama-server subprocess to listen on.
	// 0 means auto-allocate a free port.
	Port int

	// GPULayers is the number of layers to offload to GPU (-1 = auto/all).
	GPULayers int

	// CtxSize is the context window size in tokens.
	CtxSize int

	// BinDir is the directory containing llama-server binaries.
	BinDir string

	// Threads is the number of CPU threads to use (0 = auto).
	Threads int

	// FlashAttention enables flash attention if supported.
	FlashAttention bool

	// ChatTemplateFile is an optional path to a Jinja chat template file.
	// When set, llama-server uses this template instead of the GGUF-embedded one.
	ChatTemplateFile string

	// ReasoningFormat specifies the reasoning/thinking format for llama-server
	// (e.g. "deepseek" for thinking/reasoning mode).
	ReasoningFormat string

	// CPUMoE keeps all MoE expert weights on CPU when true.
	// Useful when VRAM is limited — only attention/router layers use GPU.
	CPUMoE bool

	// CPUMoELayers keeps MoE expert weights of the first N layers on CPU.
	// 0 = disabled. Mutually exclusive with CPUMoE.
	CPUMoELayers int

	// NoKVOffload keeps the KV cache on CPU to save VRAM for model weights.
	NoKVOffload bool

	// FitVRAM enables llama-server's auto-fit mode which adjusts
	// parameters to fit available device memory.
	FitVRAM bool

	// TensorSplit specifies per-GPU VRAM fractions for multi-GPU setups.
	// E.g., "0.7,0.3" for 70%/30% split across 2 GPUs.
	TensorSplit string

	// SplitMode controls multi-GPU splitting strategy: "none", "layer", "row".
	SplitMode string

	// OverrideTensor allows fine-grained tensor buffer type overrides.
	// Format: "pattern=type,pattern=type"
	OverrideTensor string

	// Quiet suppresses subprocess stdout/stderr output.
	Quiet bool

	// HealthTimeout is how long to wait for the subprocess to become healthy.
	// 0 means use the default (120s for inference, 60s for embedding).
	HealthTimeout time.Duration
}

// DefaultOptions returns Options with sensible defaults.
// Port defaults to 0 (auto-allocate) to avoid conflicts when running
// multiple instances (e.g., serve + run, inference + embedding).
func DefaultOptions() Options {
	return Options{
		Port:           0,
		GPULayers:      -1,
		CtxSize:        4096,
		Threads:        0,
		FlashAttention: true,
	}
}

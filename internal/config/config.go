package config

// Config holds the GPU server configuration.
type Config struct {
	Host             string
	Port             int
	ModelsDir        string
	BinDir           string
	GPULayers        int
	CtxSize          int
	ChatTemplateFile string // optional Jinja chat template override
	EmbeddingModel   string // optional embedding model name/path
	ReasoningFormat  string // optional reasoning format (e.g. "deepseek" for thinking/reasoning mode)
	FlashAttention   bool   // enable flash attention (default true)
	NoAutoTemplate   bool   // disable automatic template detection from GGUF metadata
	CPUMoE           bool   // keep all MoE expert weights on CPU
	CPUMoELayers     int    // keep first N layers' MoE experts on CPU (0 = disabled)
	NoKVOffload      bool   // keep KV cache on CPU to save VRAM
	FitVRAM          bool   // auto-adjust parameters to fit device memory
	TensorSplit      string // per-GPU VRAM fractions (e.g. "0.7,0.3")
	SplitMode        string // multi-GPU split strategy: none, layer, row
	OverrideTensor   string // fine-grained tensor buffer type overrides
}

// DefaultConfig returns a Config with sensible defaults.
func DefaultConfig() *Config {
	return &Config{
		Host:           "127.0.0.1",
		Port:           11435,
		ModelsDir:      ModelsDir(),
		BinDir:         BinDir(),
		GPULayers:      -1, // auto
		CtxSize:        4096,
		FlashAttention: true,
	}
}

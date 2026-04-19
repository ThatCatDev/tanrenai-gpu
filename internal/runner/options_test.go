package runner

import (
	"testing"
)

func TestDefaultOptions(t *testing.T) {
	opts := DefaultOptions()

	if opts.Port != 0 {
		t.Errorf("Port = %d, want 0 (auto-allocate)", opts.Port)
	}
	if opts.GPULayers != -1 {
		t.Errorf("GPULayers = %d, want -1 (auto)", opts.GPULayers)
	}
	if opts.CtxSize != 4096 {
		t.Errorf("CtxSize = %d, want 4096", opts.CtxSize)
	}
	if opts.Threads != 0 {
		t.Errorf("Threads = %d, want 0 (auto)", opts.Threads)
	}
	if !opts.FlashAttention {
		t.Error("FlashAttention = false, want true")
	}
	// Optional fields that should have zero values by default
	if opts.BinDir != "" {
		t.Errorf("BinDir = %q, want empty", opts.BinDir)
	}
	if opts.ChatTemplateFile != "" {
		t.Errorf("ChatTemplateFile = %q, want empty", opts.ChatTemplateFile)
	}
	if opts.ReasoningFormat != "" {
		t.Errorf("ReasoningFormat = %q, want empty", opts.ReasoningFormat)
	}
	if opts.Quiet {
		t.Error("Quiet = true, want false")
	}
	if opts.HealthTimeout != 0 {
		t.Errorf("HealthTimeout = %v, want 0", opts.HealthTimeout)
	}
}

func TestDefaultOptions_IsValue(t *testing.T) {
	// DefaultOptions returns a value, not a pointer — modifying it shouldn't
	// affect subsequent calls
	opts1 := DefaultOptions()
	opts1.GPULayers = 100
	opts1.CtxSize = 8192

	opts2 := DefaultOptions()
	if opts2.GPULayers != -1 {
		t.Errorf("second DefaultOptions().GPULayers = %d, expected -1 (not affected by modification of first)", opts2.GPULayers)
	}
	if opts2.CtxSize != 4096 {
		t.Errorf("second DefaultOptions().CtxSize = %d, expected 4096", opts2.CtxSize)
	}
}

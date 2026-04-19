package runner

import (
	"context"
	"testing"
)

// ---- NewProcessRunner ----

func TestNewProcessRunner(t *testing.T) {
	r := NewProcessRunner()
	if r == nil {
		t.Fatal("NewProcessRunner returned nil")
	}
	if r.crashNotify == nil {
		t.Fatal("crashNotify channel is nil")
	}
	if r.stopMonitor == nil {
		t.Fatal("stopMonitor channel is nil")
	}
}

func TestProcessRunner_ModelName_Empty(t *testing.T) {
	r := NewProcessRunner()
	if r.ModelName() != "" {
		t.Errorf("ModelName() = %q, want empty string before Load", r.ModelName())
	}
}

func TestProcessRunner_CrashNotify_ReturnsChannel(t *testing.T) {
	r := NewProcessRunner()
	ch := r.CrashNotify()
	if ch == nil {
		t.Fatal("CrashNotify returned nil channel")
	}
	// Should be the same channel as crashNotify
	if ch != r.crashNotify {
		t.Error("CrashNotify should return the internal crashNotify channel")
	}
}

func TestProcessRunner_Close_NoSubprocess(t *testing.T) {
	r := NewProcessRunner()
	// Close without ever loading — should not panic
	if err := r.Close(); err != nil {
		t.Errorf("Close with no subprocess: %v", err)
	}
}

func TestProcessRunner_Close_IdempotentStopMonitor(t *testing.T) {
	r := NewProcessRunner()
	// First close should succeed
	if err := r.Close(); err != nil {
		t.Fatalf("first Close: %v", err)
	}
	// Second close should not panic (stopMonitor already closed)
	if err := r.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
}

func TestProcessRunner_Health_NoSubprocess(t *testing.T) {
	r := NewProcessRunner()
	err := r.Health(context.Background())
	if err == nil {
		t.Fatal("Health without subprocess should return error")
	}
}

// ---- buildArgs ----

func TestBuildArgs_DefaultOptions(t *testing.T) {
	r := &ProcessRunner{
		modelPath: "/models/mymodel.gguf",
		opts: Options{
			CtxSize:   4096,
			GPULayers: -1, // means "all" → 999
		},
	}

	args := r.buildArgs()

	// Must include --model, --ctx-size, --host, --n-gpu-layers, --jinja
	mustContainPair(t, args, "--model", "/models/mymodel.gguf")
	mustContainPair(t, args, "--ctx-size", "4096")
	mustContainPair(t, args, "--host", "127.0.0.1")
	mustContainPair(t, args, "--n-gpu-layers", "999")
	mustContainFlag(t, args, "--jinja")
}

func TestBuildArgs_ExplicitGPULayers(t *testing.T) {
	r := &ProcessRunner{
		modelPath: "/models/mymodel.gguf",
		opts: Options{
			CtxSize:   2048,
			GPULayers: 32,
		},
	}

	args := r.buildArgs()
	mustContainPair(t, args, "--n-gpu-layers", "32")
}

func TestBuildArgs_Threads(t *testing.T) {
	r := &ProcessRunner{
		modelPath: "/models/mymodel.gguf",
		opts: Options{
			CtxSize: 4096,
			Threads: 8,
		},
	}

	args := r.buildArgs()
	mustContainPair(t, args, "--threads", "8")
}

func TestBuildArgs_NoThreads(t *testing.T) {
	r := &ProcessRunner{
		modelPath: "/models/mymodel.gguf",
		opts: Options{
			CtxSize: 4096,
			Threads: 0, // should not add --threads
		},
	}

	args := r.buildArgs()
	for _, a := range args {
		if a == "--threads" {
			t.Error("--threads should not be in args when Threads=0")
		}
	}
}

func TestBuildArgs_FlashAttention(t *testing.T) {
	r := &ProcessRunner{
		modelPath: "/models/mymodel.gguf",
		opts: Options{
			CtxSize:        4096,
			FlashAttention: true,
		},
	}

	args := r.buildArgs()
	mustContainPair(t, args, "--flash-attn", "on")
}

func TestBuildArgs_NoFlashAttention(t *testing.T) {
	r := &ProcessRunner{
		modelPath: "/models/mymodel.gguf",
		opts: Options{
			CtxSize:        4096,
			FlashAttention: false,
		},
	}

	args := r.buildArgs()
	for _, a := range args {
		if a == "--flash-attn" {
			t.Error("--flash-attn should not be in args when FlashAttention=false")
		}
	}
}

func TestBuildArgs_ChatTemplateFile(t *testing.T) {
	r := &ProcessRunner{
		modelPath: "/models/mymodel.gguf",
		opts: Options{
			CtxSize:          4096,
			ChatTemplateFile: "/tmp/template.jinja",
		},
	}

	args := r.buildArgs()
	mustContainPair(t, args, "--chat-template-file", "/tmp/template.jinja")
}

func TestBuildArgs_ReasoningFormat(t *testing.T) {
	r := &ProcessRunner{
		modelPath: "/models/mymodel.gguf",
		opts: Options{
			CtxSize:         4096,
			ReasoningFormat: "deepseek",
		},
	}

	args := r.buildArgs()
	mustContainPair(t, args, "--reasoning-format", "deepseek")
}

func TestBuildArgs_NoOptionalFlags(t *testing.T) {
	r := &ProcessRunner{
		modelPath: "/models/mymodel.gguf",
		opts: Options{
			CtxSize: 4096,
		},
	}

	args := r.buildArgs()
	for i, a := range args {
		if a == "--chat-template-file" {
			t.Errorf("unexpected --chat-template-file at index %d", i)
		}
		if a == "--reasoning-format" {
			t.Errorf("unexpected --reasoning-format at index %d", i)
		}
	}
}

// ---- ModelName from Load (without subprocess) ----

func TestProcessRunner_ModelName_SplitGGUF(t *testing.T) {
	// Simulate what Load does for a split GGUF file name
	r := NewProcessRunner()
	r.modelPath = "/models/mymodel-00001-of-00003.gguf"
	name := "mymodel-00001-of-00003.gguf"
	// Apply same logic as Load:
	if splitModelSuffix.MatchString(name) {
		name = splitModelSuffix.ReplaceAllString(name, "")
	}
	r.modelName = name

	if r.ModelName() != "mymodel" {
		t.Errorf("ModelName() = %q, want %q", r.ModelName(), "mymodel")
	}
}

func TestProcessRunner_ModelName_RegularGGUF(t *testing.T) {
	r := NewProcessRunner()
	r.modelName = "Qwen2.5-7B-Instruct"

	if r.ModelName() != "Qwen2.5-7B-Instruct" {
		t.Errorf("ModelName() = %q, want %q", r.ModelName(), "Qwen2.5-7B-Instruct")
	}
}

// ---- helpers ----

func mustContainPair(t *testing.T, args []string, flag, value string) {
	t.Helper()
	for i := 0; i < len(args)-1; i++ {
		if args[i] == flag && args[i+1] == value {
			return
		}
	}
	t.Errorf("args missing pair (%q, %q); full args: %v", flag, value, args)
}

func mustContainFlag(t *testing.T, args []string, flag string) {
	t.Helper()
	for _, a := range args {
		if a == flag {
			return
		}
	}
	t.Errorf("args missing flag %q; full args: %v", flag, args)
}

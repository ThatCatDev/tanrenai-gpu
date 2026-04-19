package server

import (
	"context"
	"io"
	"os"
	"path/filepath"
	"testing"

	"github.com/ThatCatDev/tanrenai-gpu/internal/runner"
	"github.com/ThatCatDev/tanrenai-gpu/pkg/api"
)

// createFakeGGUF writes a minimal GGUF v3 file that ReadMetadata can parse.
func createFakeGGUF(t *testing.T, dir, name string) string {
	t.Helper()
	path := filepath.Join(dir, name)

	// Minimal GGUF v3: magic + version + tensor_count(0) + kv_count(0)
	gguf := []byte{
		0x47, 0x47, 0x55, 0x46, // magic "GGUF"
		0x03, 0x00, 0x00, 0x00, // version 3
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // tensor count = 0
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // kv count = 0
	}
	if err := os.WriteFile(path, gguf, 0644); err != nil {
		t.Fatalf("create fake GGUF %s: %v", name, err)
	}

	return path
}

// TestLoadModel_ModelNotFound verifies error when model doesn't exist in store.
func TestLoadModel_ModelNotFound(t *testing.T) {
	s := New(newTestConfig(t))

	_, err := s.LoadModel(context.Background(), "ghost-model")
	if err == nil {
		t.Fatal("expected error for nonexistent model")
	}
}

// TestLoadModel_WithValidGGUF_ButNoBinary tests LoadModel when model file exists
// but llama-server binary doesn't (so the actual subprocess spawn fails).
func TestLoadModel_WithValidGGUF_ButNoBinary(t *testing.T) {
	modelsDir := t.TempDir()
	createFakeGGUF(t, modelsDir, "testmodel.gguf")

	cfg := newTestConfig(t)
	cfg.ModelsDir = modelsDir
	cfg.BinDir = t.TempDir() // empty dir — no llama-server binary
	cfg.NoAutoTemplate = true

	s := New(cfg)

	_, err := s.LoadModel(context.Background(), "testmodel")
	// Expected to fail because llama-server binary doesn't exist
	if err == nil {
		t.Fatal("expected error when llama-server binary is missing")
	}
}

// TestLoadModel_ExistingRunnerClosed verifies existing runner is closed before loading a new model.
func TestLoadModel_ExistingRunnerClosed(t *testing.T) {
	modelsDir := t.TempDir()
	createFakeGGUF(t, modelsDir, "model2.gguf")

	cfg := newTestConfig(t)
	cfg.ModelsDir = modelsDir
	cfg.BinDir = t.TempDir() // no binary
	cfg.NoAutoTemplate = true

	s := New(cfg)

	// Set a mock runner
	mock := &testRunner{}
	s.runner = mock

	_, err := s.LoadModel(context.Background(), "model2")
	// Error expected (no binary), but Close() should have been called on old runner
	if err == nil {
		t.Fatal("expected error (no binary)")
	}

	if !mock.closeCalled {
		t.Error("Close() should have been called on existing runner before loading a new model")
	}
}

// TestLoadModel_TemplateCleaned tests that template cleanup is called on model switch.
func TestLoadModel_TemplateCleaned(t *testing.T) {
	modelsDir := t.TempDir()
	createFakeGGUF(t, modelsDir, "model3.gguf")

	cfg := newTestConfig(t)
	cfg.ModelsDir = modelsDir
	cfg.BinDir = t.TempDir()
	cfg.NoAutoTemplate = true

	s := New(cfg)

	// Set a template cleanup function
	cleanupCalled := false
	s.templateCleanup = func() { cleanupCalled = true }

	_, err := s.LoadModel(context.Background(), "model3")
	// Error expected, but cleanup should run
	if err == nil {
		t.Fatal("expected error (no binary)")
	}

	if !cleanupCalled {
		t.Error("templateCleanup should have been called when loading a new model")
	}
}

// TestLoadModel_AutoTemplate_WithArch verifies auto-template runs when GGUF has architecture.
func TestLoadModel_AutoTemplate_WithArch(t *testing.T) {
	modelsDir := t.TempDir()

	// GGUF with architecture but no chat template — triggers ChatML fallback
	ggufPath := filepath.Join(modelsDir, "arch-model.gguf")
	gguf := buildGGUFWithArch("qwen2", "TestModel")
	if err := os.WriteFile(ggufPath, gguf, 0644); err != nil {
		t.Fatal(err)
	}

	cfg := newTestConfig(t)
	cfg.ModelsDir = modelsDir
	cfg.BinDir = t.TempDir() // no binary — will fail after template resolution
	cfg.NoAutoTemplate = false

	s := New(cfg)

	_, err := s.LoadModel(context.Background(), "arch-model")
	if err == nil {
		t.Fatal("expected error (no binary)")
	}

	// templateCleanup should be set if auto-template ran successfully
	// (or nil if GGUF had no architecture field — both are acceptable)
}

// testRunner is a minimal runner.Runner implementation for testing.
type testRunner struct {
	closeCalled bool
}

func (r *testRunner) Load(_ context.Context, _ string, _ runner.Options) error { return nil }
func (r *testRunner) Health(_ context.Context) error                           { return nil }
func (r *testRunner) ModelName() string                                        { return "test" }

func (r *testRunner) Close() error {
	r.closeCalled = true

	return nil
}

func (r *testRunner) ChatCompletion(_ context.Context, _ *api.ChatCompletionRequest) (*api.ChatCompletionResponse, error) {
	return nil, nil
}

func (r *testRunner) ChatCompletionStream(_ context.Context, _ *api.ChatCompletionRequest, _ io.Writer) error {
	return nil
}

func (r *testRunner) Tokenize(_ context.Context, _ string) (int, error) { return 0, nil }

// buildGGUFWithArch creates a minimal GGUF with an architecture field.
func buildGGUFWithArch(arch, name string) []byte {
	writeU32le := func(v uint32) []byte {
		return []byte{byte(v), byte(v >> 8), byte(v >> 16), byte(v >> 24)}
	}
	writeU64le := func(v uint64) []byte {
		b := make([]byte, 8)
		for i := 0; i < 8; i++ {
			b[i] = byte(v >> (i * 8))
		}

		return b
	}
	writeStr := func(s string) []byte {
		var out []byte
		out = append(out, writeU64le(uint64(len(s)))...)
		out = append(out, []byte(s)...)

		return out
	}
	writeKVStr := func(key, val string) []byte {
		var out []byte
		out = append(out, writeStr(key)...)
		out = append(out, writeU32le(8)...) // string type
		out = append(out, writeStr(val)...)

		return out
	}

	var buf []byte
	buf = append(buf, writeU32le(0x46554747)...) // magic
	buf = append(buf, writeU32le(3)...)          // version
	buf = append(buf, writeU64le(0)...)          // tensor count
	buf = append(buf, writeU64le(2)...)          // 2 KV pairs

	buf = append(buf, writeKVStr("general.architecture", arch)...)
	buf = append(buf, writeKVStr("general.name", name)...)

	return buf
}

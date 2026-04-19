package models

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestNewStore_Dir(t *testing.T) {
	tmp := t.TempDir()
	s := NewStore(tmp)
	if s.Dir() != tmp {
		t.Errorf("Dir() = %q, want %q", s.Dir(), tmp)
	}
}

func TestList_EmptyDir(t *testing.T) {
	tmp := t.TempDir()
	s := NewStore(tmp)

	entries := s.List()
	if len(entries) != 0 {
		t.Errorf("List() on empty dir: got %d entries, want 0", len(entries))
	}
}

func TestList_NonExistentDir(t *testing.T) {
	s := NewStore("/nonexistent/path/that/does/not/exist")
	entries := s.List()
	// Should return empty, not panic
	if entries == nil {
		t.Error("List() on non-existent dir returned nil, want empty slice")
	}
}

func TestList_SingleGGUF(t *testing.T) {
	tmp := t.TempDir()
	ggufPath := filepath.Join(tmp, "mymodel.gguf")
	if err := os.WriteFile(ggufPath, []byte("fake gguf content"), 0644); err != nil {
		t.Fatal(err)
	}

	s := NewStore(tmp)
	entries := s.List()

	if len(entries) != 1 {
		t.Fatalf("List() = %d entries, want 1", len(entries))
	}
	if entries[0].Name != "mymodel" {
		t.Errorf("Name = %q, want %q", entries[0].Name, "mymodel")
	}
	if entries[0].Path != ggufPath {
		t.Errorf("Path = %q, want %q", entries[0].Path, ggufPath)
	}
	if entries[0].Size != int64(len("fake gguf content")) {
		t.Errorf("Size = %d, want %d", entries[0].Size, len("fake gguf content"))
	}
}

func TestList_MultipleGGUFs(t *testing.T) {
	tmp := t.TempDir()
	models := []string{"llama-3.gguf", "qwen2.5.gguf", "mistral.gguf"}
	for _, m := range models {
		if err := os.WriteFile(filepath.Join(tmp, m), []byte("data"), 0644); err != nil {
			t.Fatal(err)
		}
	}

	s := NewStore(tmp)
	entries := s.List()

	if len(entries) != 3 {
		t.Errorf("List() = %d entries, want 3", len(entries))
	}
}

func TestList_IgnoresNonGGUF(t *testing.T) {
	tmp := t.TempDir()
	// Write a mix of .gguf and non-.gguf files
	files := map[string]bool{
		"model.gguf":      true,  // should appear
		"model.bin":       false, // should NOT appear
		"config.json":     false,
		"tokenizer.model": false,
		"another.GGUF":    true, // uppercase extension, should also appear (case-insensitive)
	}
	for name := range files {
		if err := os.WriteFile(filepath.Join(tmp, name), []byte("x"), 0644); err != nil {
			t.Fatal(err)
		}
	}

	s := NewStore(tmp)
	entries := s.List()

	expected := 0
	for _, v := range files {
		if v {
			expected++
		}
	}
	if len(entries) != expected {
		names := make([]string, len(entries))
		for i, e := range entries {
			names[i] = e.Name
		}
		t.Errorf("List() = %d entries (%v), want %d", len(entries), names, expected)
	}
}

func TestList_SplitGGUFs_Grouped(t *testing.T) {
	tmp := t.TempDir()
	// Simulate a 3-part split GGUF
	parts := []string{
		"bigmodel-00001-of-00003.gguf",
		"bigmodel-00002-of-00003.gguf",
		"bigmodel-00003-of-00003.gguf",
	}
	for _, p := range parts {
		if err := os.WriteFile(filepath.Join(tmp, p), []byte("partdata"), 0644); err != nil {
			t.Fatal(err)
		}
	}

	s := NewStore(tmp)
	entries := s.List()

	// All parts should be grouped into one entry
	if len(entries) != 1 {
		names := make([]string, len(entries))
		for i, e := range entries {
			names[i] = e.Name
		}
		t.Fatalf("List() = %d entries %v, want 1 (split GGUFs should be grouped)", len(entries), names)
	}
	if entries[0].Name != "bigmodel" {
		t.Errorf("grouped entry Name = %q, want %q", entries[0].Name, "bigmodel")
	}
	// Size should be sum of all parts (8 bytes each * 3)
	wantSize := int64(len("partdata") * 3)
	if entries[0].Size != wantSize {
		t.Errorf("grouped entry Size = %d, want %d", entries[0].Size, wantSize)
	}
}

func TestList_SplitAndSingleMixed(t *testing.T) {
	tmp := t.TempDir()
	// One split model + one standalone
	splitParts := []string{
		"splitmodel-00001-of-00002.gguf",
		"splitmodel-00002-of-00002.gguf",
	}
	for _, p := range splitParts {
		if err := os.WriteFile(filepath.Join(tmp, p), []byte("x"), 0644); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(tmp, "standalone.gguf"), []byte("y"), 0644); err != nil {
		t.Fatal(err)
	}

	s := NewStore(tmp)
	entries := s.List()

	if len(entries) != 2 {
		t.Errorf("List() = %d entries, want 2", len(entries))
	}
}

func TestResolve_ExactFilename(t *testing.T) {
	tmp := t.TempDir()
	ggufPath := filepath.Join(tmp, "mymodel.gguf")
	if err := os.WriteFile(ggufPath, []byte("data"), 0644); err != nil {
		t.Fatal(err)
	}

	s := NewStore(tmp)

	// Resolve by full name with extension
	got, err := s.Resolve("mymodel.gguf")
	if err != nil {
		t.Fatalf("Resolve(mymodel.gguf): %v", err)
	}
	if got != ggufPath {
		t.Errorf("Resolve(mymodel.gguf) = %q, want %q", got, ggufPath)
	}
}

func TestResolve_WithoutExtension(t *testing.T) {
	tmp := t.TempDir()
	ggufPath := filepath.Join(tmp, "mymodel.gguf")
	if err := os.WriteFile(ggufPath, []byte("data"), 0644); err != nil {
		t.Fatal(err)
	}

	s := NewStore(tmp)

	// Resolve by name without extension
	got, err := s.Resolve("mymodel")
	if err != nil {
		t.Fatalf("Resolve(mymodel): %v", err)
	}
	// Should return the .gguf path
	if !strings.HasSuffix(got, ".gguf") {
		t.Errorf("Resolve(mymodel) = %q, expected path ending in .gguf", got)
	}
}

func TestResolve_FuzzyMatch(t *testing.T) {
	tmp := t.TempDir()
	ggufPath := filepath.Join(tmp, "qwen2.5-coder-32b-q4_k_m.gguf")
	if err := os.WriteFile(ggufPath, []byte("data"), 0644); err != nil {
		t.Fatal(err)
	}

	s := NewStore(tmp)

	// Fuzzy: partial substring match
	got, err := s.Resolve("qwen2.5")
	if err != nil {
		t.Fatalf("Resolve(qwen2.5): %v", err)
	}
	if got == "" {
		t.Error("Resolve(qwen2.5) returned empty path")
	}
}

func TestResolve_CaseInsensitiveExact(t *testing.T) {
	tmp := t.TempDir()
	ggufPath := filepath.Join(tmp, "MyModel.gguf")
	if err := os.WriteFile(ggufPath, []byte("data"), 0644); err != nil {
		t.Fatal(err)
	}

	s := NewStore(tmp)

	got, err := s.Resolve("MyModel")
	if err != nil {
		t.Fatalf("Resolve(MyModel): %v", err)
	}
	if got == "" {
		t.Error("Resolve(MyModel) returned empty path")
	}
}

func TestResolve_NotFound(t *testing.T) {
	tmp := t.TempDir()
	// Put one model in the dir so List() works
	if err := os.WriteFile(filepath.Join(tmp, "existing.gguf"), []byte("data"), 0644); err != nil {
		t.Fatal(err)
	}

	s := NewStore(tmp)

	_, err := s.Resolve("nonexistent-model-xyz")
	if err == nil {
		t.Fatal("Resolve on missing model: expected error, got nil")
	}
	if !strings.Contains(err.Error(), "not found") {
		t.Errorf("error = %q, expected it to contain 'not found'", err.Error())
	}
}

func TestResolve_AbsolutePath(t *testing.T) {
	tmp := t.TempDir()
	ggufPath := filepath.Join(tmp, "absolute.gguf")
	if err := os.WriteFile(ggufPath, []byte("data"), 0644); err != nil {
		t.Fatal(err)
	}

	s := NewStore(tmp)

	// Passing an absolute path that exists should return it directly
	got, err := s.Resolve(ggufPath)
	if err != nil {
		t.Fatalf("Resolve(abs path): %v", err)
	}
	if got != ggufPath {
		t.Errorf("Resolve(abs path) = %q, want %q", got, ggufPath)
	}
}

func TestResolve_AbsolutePathMissing(t *testing.T) {
	tmp := t.TempDir()
	s := NewStore(tmp)

	nonExistent := filepath.Join(tmp, "ghost.gguf")
	_, err := s.Resolve(nonExistent)
	if err == nil {
		t.Fatal("Resolve on missing absolute path: expected error, got nil")
	}
}

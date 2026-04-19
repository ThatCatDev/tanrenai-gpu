package models

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadMetadata_FileNotExist(t *testing.T) {
	meta, err := LoadMetadata("/nonexistent/model.gguf")
	if err != nil {
		t.Fatalf("expected nil error for missing file, got: %v", err)
	}
	if meta != nil {
		t.Errorf("expected nil meta for missing file, got: %+v", meta)
	}
}

func TestLoadMetadata_ValidFile(t *testing.T) {
	dir := t.TempDir()
	ggufPath := filepath.Join(dir, "model.gguf")
	// Create a dummy gguf file
	if err := os.WriteFile(ggufPath, []byte("fake"), 0644); err != nil {
		t.Fatal(err)
	}

	want := &ModelMetadata{
		HFRepo:   "owner/repo",
		HFBranch: "main",
		Source:   "huggingface",
	}
	if err := SaveMetadata(ggufPath, want); err != nil {
		t.Fatalf("SaveMetadata: %v", err)
	}

	got, err := LoadMetadata(ggufPath)
	if err != nil {
		t.Fatalf("LoadMetadata: %v", err)
	}
	if got == nil {
		t.Fatal("expected non-nil metadata")
	}
	if got.HFRepo != want.HFRepo {
		t.Errorf("HFRepo = %q, want %q", got.HFRepo, want.HFRepo)
	}
	if got.HFBranch != want.HFBranch {
		t.Errorf("HFBranch = %q, want %q", got.HFBranch, want.HFBranch)
	}
	if got.Source != want.Source {
		t.Errorf("Source = %q, want %q", got.Source, want.Source)
	}
}

func TestLoadMetadata_BadJSON(t *testing.T) {
	dir := t.TempDir()
	ggufPath := filepath.Join(dir, "model.gguf")
	// Write bad JSON to the meta file
	if err := os.WriteFile(ggufPath+".meta.json", []byte("{invalid"), 0644); err != nil {
		t.Fatal(err)
	}

	_, err := LoadMetadata(ggufPath)
	if err == nil {
		t.Fatal("expected error for bad JSON")
	}
}

func TestSaveMetadata_CreatesFile(t *testing.T) {
	dir := t.TempDir()
	ggufPath := filepath.Join(dir, "model.gguf")

	meta := &ModelMetadata{
		HFRepo: "owner/myrepo",
		Source: "huggingface",
	}
	if err := SaveMetadata(ggufPath, meta); err != nil {
		t.Fatalf("SaveMetadata: %v", err)
	}

	metaPath := ggufPath + ".meta.json"
	if _, err := os.Stat(metaPath); err != nil {
		t.Errorf("meta file not created: %v", err)
	}
}

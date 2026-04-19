package models

import (
	"os"
	"path/filepath"
	"testing"
)

func TestMetadataPath(t *testing.T) {
	tests := []struct {
		ggufPath string
		want     string
	}{
		{"/models/foo.gguf", "/models/foo.gguf.meta.json"},
		{"/tmp/my-model.gguf", "/tmp/my-model.gguf.meta.json"},
		{"relative/path.gguf", "relative/path.gguf.meta.json"},
	}
	for _, tt := range tests {
		got := MetadataPath(tt.ggufPath)
		if got != tt.want {
			t.Errorf("MetadataPath(%q) = %q, want %q", tt.ggufPath, got, tt.want)
		}
	}
}

func TestLoadMetadata_NotExist(t *testing.T) {
	tmp := t.TempDir()
	ggufPath := filepath.Join(tmp, "nonexistent.gguf")

	meta, err := LoadMetadata(ggufPath)
	if err != nil {
		t.Fatalf("LoadMetadata on missing file: unexpected error: %v", err)
	}
	if meta != nil {
		t.Errorf("LoadMetadata on missing file: got %+v, want nil", meta)
	}
}

func TestSaveAndLoadMetadata_RoundTrip(t *testing.T) {
	tmp := t.TempDir()
	ggufPath := filepath.Join(tmp, "model.gguf")

	original := &ModelMetadata{
		HFRepo:   "owner/repo",
		HFBranch: "main",
		Source:   "https://huggingface.co/owner/repo/resolve/main/model.gguf",
	}

	if err := SaveMetadata(ggufPath, original); err != nil {
		t.Fatalf("SaveMetadata: %v", err)
	}

	// Sidecar file should exist at the expected path
	metaPath := MetadataPath(ggufPath)
	if _, err := os.Stat(metaPath); err != nil {
		t.Fatalf("metadata file not created at %q: %v", metaPath, err)
	}

	loaded, err := LoadMetadata(ggufPath)
	if err != nil {
		t.Fatalf("LoadMetadata: %v", err)
	}
	if loaded == nil {
		t.Fatal("LoadMetadata returned nil after save")
	}

	if loaded.HFRepo != original.HFRepo {
		t.Errorf("HFRepo = %q, want %q", loaded.HFRepo, original.HFRepo)
	}
	if loaded.HFBranch != original.HFBranch {
		t.Errorf("HFBranch = %q, want %q", loaded.HFBranch, original.HFBranch)
	}
	if loaded.Source != original.Source {
		t.Errorf("Source = %q, want %q", loaded.Source, original.Source)
	}
}

func TestSaveAndLoadMetadata_EmptyFields(t *testing.T) {
	tmp := t.TempDir()
	ggufPath := filepath.Join(tmp, "model.gguf")

	original := &ModelMetadata{
		Source: "manual",
	}

	if err := SaveMetadata(ggufPath, original); err != nil {
		t.Fatalf("SaveMetadata: %v", err)
	}

	loaded, err := LoadMetadata(ggufPath)
	if err != nil {
		t.Fatalf("LoadMetadata: %v", err)
	}
	if loaded == nil {
		t.Fatal("LoadMetadata returned nil")
	}

	if loaded.Source != "manual" {
		t.Errorf("Source = %q, want %q", loaded.Source, "manual")
	}
	if loaded.HFRepo != "" {
		t.Errorf("HFRepo = %q, want empty", loaded.HFRepo)
	}
	if loaded.HFBranch != "" {
		t.Errorf("HFBranch = %q, want empty", loaded.HFBranch)
	}
}

func TestLoadMetadata_CorruptJSON(t *testing.T) {
	tmp := t.TempDir()
	ggufPath := filepath.Join(tmp, "model.gguf")
	metaPath := MetadataPath(ggufPath)

	// Write invalid JSON
	if err := os.WriteFile(metaPath, []byte("{invalid json"), 0644); err != nil {
		t.Fatalf("write corrupt file: %v", err)
	}

	_, err := LoadMetadata(ggufPath)
	if err == nil {
		t.Fatal("LoadMetadata on corrupt JSON: expected error, got nil")
	}
}

func TestParseHFURL_Valid(t *testing.T) {
	tests := []struct {
		url        string
		wantRepo   string
		wantBranch string
	}{
		{
			url:        "https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct-GGUF/resolve/main/model.gguf",
			wantRepo:   "Qwen/Qwen2.5-Coder-32B-Instruct-GGUF",
			wantBranch: "main",
		},
		{
			url:        "https://huggingface.co/bartowski/Qwen3-32B-GGUF/resolve/dev/Qwen3-32B-Q4_K_M.gguf",
			wantRepo:   "bartowski/Qwen3-32B-GGUF",
			wantBranch: "dev",
		},
		{
			url:        "http://huggingface.co/owner/repo/resolve/feature-branch/file.gguf",
			wantRepo:   "owner/repo",
			wantBranch: "feature-branch",
		},
		{
			// Leading/trailing whitespace should be trimmed
			url:        "  https://huggingface.co/owner/repo/resolve/main/file.gguf  ",
			wantRepo:   "owner/repo",
			wantBranch: "main",
		},
	}
	for _, tt := range tests {
		repo, branch, ok := ParseHFURL(tt.url)
		if !ok {
			t.Errorf("ParseHFURL(%q): ok = false, want true", tt.url)

			continue
		}
		if repo != tt.wantRepo {
			t.Errorf("ParseHFURL(%q): repo = %q, want %q", tt.url, repo, tt.wantRepo)
		}
		if branch != tt.wantBranch {
			t.Errorf("ParseHFURL(%q): branch = %q, want %q", tt.url, branch, tt.wantBranch)
		}
	}
}

func TestParseHFURL_Invalid(t *testing.T) {
	tests := []string{
		"",
		"   ",
		"https://example.com/model.gguf",
		"https://huggingface.co/owner/repo/blob/main/file.gguf", // not /resolve/
		"not-a-url",
		"https://huggingface.co/owner",              // incomplete
		"https://huggingface.co/owner/repo/resolve", // missing branch
	}
	for _, url := range tests {
		_, _, ok := ParseHFURL(url)
		if ok {
			t.Errorf("ParseHFURL(%q): ok = true, want false", url)
		}
	}
}

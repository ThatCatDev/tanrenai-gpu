package runner

import (
	"os"
	"strings"
	"testing"
)

func TestResolveTemplate_NonExistentFile(t *testing.T) {
	res, err := ResolveTemplate("/nonexistent/model.gguf", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Should return nil when GGUF can't be read and no metadata exists.
	if res != nil {
		res.Cleanup()
		t.Error("expected nil resolution for nonexistent file")
	}
}

func TestResolveTemplate_Cleanup(t *testing.T) {
	// Write a generated template file and verify cleanup removes it.
	cfg := DefaultChatMLConfig
	tpl := GenerateChatML(cfg)
	path, err := WriteTemplateFile("test-cleanup", tpl)
	if err != nil {
		t.Fatalf("write template: %v", err)
	}

	// File should exist.
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("template file not created: %v", err)
	}

	// Cleanup should remove it.
	os.Remove(path)
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Error("cleanup did not remove template file")
	}
}

func TestSanitizeName(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"qwen2", "qwen2"},
		{"Qwen/Qwen2.5-32B-GGUF", "qwen-qwen2-5-32b-gguf"},
		{"foo bar/baz.qux", "foo-bar-baz-qux"},
	}
	for _, tt := range tests {
		got := sanitizeName(tt.input)
		if got != tt.want {
			t.Errorf("sanitizeName(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

func TestWriteTemplateFile(t *testing.T) {
	content := "{{ test template }}"
	path, err := WriteTemplateFile("test", content)
	if err != nil {
		t.Fatalf("write: %v", err)
	}
	defer os.Remove(path)

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if string(data) != content {
		t.Errorf("content = %q, want %q", string(data), content)
	}
	if !strings.HasSuffix(path, ".jinja") {
		t.Errorf("path %q should end with .jinja", path)
	}
}

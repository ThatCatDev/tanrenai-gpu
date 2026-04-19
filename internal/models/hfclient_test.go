package models

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestFetchChatTemplate_String(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"chat_template": "{{ messages }}", "other": "field"}`))
	}))
	defer srv.Close()

	c := NewHFClient()
	c.BaseURL = srv.URL

	tpl, err := c.FetchChatTemplate("owner/repo", "main")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if tpl != "{{ messages }}" {
		t.Errorf("got %q, want %q", tpl, "{{ messages }}")
	}
}

func TestFetchChatTemplate_Array(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"chat_template": [
			{"name": "tool_use", "template": "tool template"},
			{"name": "default", "template": "default template"}
		]}`))
	}))
	defer srv.Close()

	c := NewHFClient()
	c.BaseURL = srv.URL

	tpl, err := c.FetchChatTemplate("owner/repo", "main")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if tpl != "default template" {
		t.Errorf("got %q, want %q", tpl, "default template")
	}
}

func TestFetchChatTemplate_ArrayNoDefault(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"chat_template": [{"name": "custom", "template": "first template"}]}`))
	}))
	defer srv.Close()

	c := NewHFClient()
	c.BaseURL = srv.URL

	tpl, err := c.FetchChatTemplate("owner/repo", "main")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if tpl != "first template" {
		t.Errorf("got %q, want %q", tpl, "first template")
	}
}

func TestFetchChatTemplate_NoChatTemplate(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"model_type": "gpt2"}`))
	}))
	defer srv.Close()

	c := NewHFClient()
	c.BaseURL = srv.URL

	_, err := c.FetchChatTemplate("owner/repo", "main")
	if err == nil {
		t.Fatal("expected error for missing chat_template")
	}
}

func TestFetchChatTemplate_404(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer srv.Close()

	c := NewHFClient()
	c.BaseURL = srv.URL

	_, err := c.FetchChatTemplate("owner/repo", "main")
	if err == nil {
		t.Fatal("expected error for 404")
	}
}

func TestParseHFURL(t *testing.T) {
	tests := []struct {
		url    string
		repo   string
		branch string
		ok     bool
	}{
		{
			url:    "https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct-GGUF/resolve/main/qwen2.5-coder-32b-instruct-q4_k_m.gguf",
			repo:   "Qwen/Qwen2.5-Coder-32B-Instruct-GGUF",
			branch: "main",
			ok:     true,
		},
		{
			url:    "https://huggingface.co/bartowski/Qwen3-32B-GGUF/resolve/main/Qwen3-32B-Q4_K_M.gguf",
			repo:   "bartowski/Qwen3-32B-GGUF",
			branch: "main",
			ok:     true,
		},
		{
			url: "https://example.com/model.gguf",
			ok:  false,
		},
		{
			url: "",
			ok:  false,
		},
	}

	for _, tt := range tests {
		repo, branch, ok := ParseHFURL(tt.url)
		if ok != tt.ok {
			t.Errorf("ParseHFURL(%q): ok = %v, want %v", tt.url, ok, tt.ok)

			continue
		}
		if ok {
			if repo != tt.repo {
				t.Errorf("ParseHFURL(%q): repo = %q, want %q", tt.url, repo, tt.repo)
			}
			if branch != tt.branch {
				t.Errorf("ParseHFURL(%q): branch = %q, want %q", tt.url, branch, tt.branch)
			}
		}
	}
}

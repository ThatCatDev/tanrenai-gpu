package runner

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/ThatCatDev/tanrenai-gpu/pkg/api"
)

// setupRunnerWithServer creates a ProcessRunner with an internal Client wired
// to the given httptest server. This lets us test ChatCompletion, Tokenize,
// and ChatCompletionStream without spawning an actual subprocess.
func setupRunnerWithServer(srv *httptest.Server) *ProcessRunner {
	r := NewProcessRunner()
	r.client = NewClient(srv.URL)
	r.modelName = "testmodel"

	return r
}

func TestProcessRunner_ChatCompletion_SetsStreamFalse(t *testing.T) {
	var gotStream *bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req api.ChatCompletionRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "bad request", http.StatusBadRequest)

			return
		}
		gotStream = &req.Stream
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(api.ChatCompletionResponse{
			ID: "test",
			Choices: []api.Choice{
				{Message: api.Message{Role: "assistant", Content: "hi"}},
			},
		})
	}))
	defer srv.Close()

	r := setupRunnerWithServer(srv)

	req := &api.ChatCompletionRequest{
		Model:    "testmodel",
		Messages: []api.Message{{Role: "user", Content: "hello"}},
		Stream:   true, // will be overridden to false
	}

	resp, err := r.ChatCompletion(context.Background(), req)
	if err != nil {
		t.Fatalf("ChatCompletion: %v", err)
	}
	if resp == nil {
		t.Fatal("expected non-nil response")
	}
	if gotStream == nil {
		t.Fatal("stream field not captured")
	}
	if *gotStream {
		t.Error("ChatCompletion should set Stream=false")
	}
}

func TestProcessRunner_ChatCompletion_Error(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	r := setupRunnerWithServer(srv)
	req := &api.ChatCompletionRequest{
		Messages: []api.Message{{Role: "user", Content: "hello"}},
	}

	_, err := r.ChatCompletion(context.Background(), req)
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestProcessRunner_ChatCompletionStream_SetsStreamTrue(t *testing.T) {
	var gotStream bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req api.ChatCompletionRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err == nil {
			gotStream = req.Stream
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer srv.Close()

	r := setupRunnerWithServer(srv)
	req := &api.ChatCompletionRequest{
		Model:    "testmodel",
		Messages: []api.Message{{Role: "user", Content: "hello"}},
		Stream:   false, // will be overridden to true
	}

	var buf strings.Builder
	if err := r.ChatCompletionStream(context.Background(), req, &buf); err != nil {
		t.Fatalf("ChatCompletionStream: %v", err)
	}
	if !gotStream {
		t.Error("ChatCompletionStream should set Stream=true")
	}
}

func TestProcessRunner_ChatCompletionStream_Error(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer srv.Close()

	r := setupRunnerWithServer(srv)
	req := &api.ChatCompletionRequest{
		Messages: []api.Message{{Role: "user", Content: "hello"}},
	}

	var buf strings.Builder
	err := r.ChatCompletionStream(context.Background(), req, &buf)
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestProcessRunner_Tokenize_Success(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"tokens": []int{1, 2, 3},
		})
	}))
	defer srv.Close()

	r := setupRunnerWithServer(srv)
	count, err := r.Tokenize(context.Background(), "hello world")
	if err != nil {
		t.Fatalf("Tokenize: %v", err)
	}
	if count != 3 {
		t.Errorf("count = %d, want 3", count)
	}
}

func TestProcessRunner_Tokenize_Error(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	r := setupRunnerWithServer(srv)
	_, err := r.Tokenize(context.Background(), "hello")
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestProcessRunner_Health_WithSubprocess(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.WriteHeader(http.StatusOK)

			return
		}
		http.Error(w, "not found", http.StatusNotFound)
	}))
	defer srv.Close()

	r := NewProcessRunner()
	doneCh := make(chan struct{})
	r.sub = &Subprocess{
		baseURL: srv.URL,
		label:   "test",
		doneCh:  doneCh,
	}

	if err := r.Health(context.Background()); err != nil {
		t.Errorf("Health with healthy server: %v", err)
	}
}

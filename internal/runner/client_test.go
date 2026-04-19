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

func TestNewClient(t *testing.T) {
	c := NewClient("http://localhost:11435")
	if c == nil {
		t.Fatal("NewClient returned nil")
	}
	if c.baseURL != "http://localhost:11435" {
		t.Errorf("baseURL = %q, want %q", c.baseURL, "http://localhost:11435")
	}
	if c.httpClient == nil {
		t.Fatal("httpClient is nil")
	}
}

func TestClient_ChatCompletion_Success(t *testing.T) {
	want := &api.ChatCompletionResponse{
		ID:    "chatcmpl-abc",
		Model: "testmodel",
		Choices: []api.Choice{
			{Message: api.Message{Role: "assistant", Content: "Hello!"}},
		},
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			http.Error(w, "unexpected path", http.StatusBadRequest)

			return
		}
		if r.Header.Get("Content-Type") != "application/json" {
			http.Error(w, "bad content-type", http.StatusBadRequest)

			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(want)
	}))
	defer srv.Close()

	c := NewClient(srv.URL)
	req := &api.ChatCompletionRequest{
		Model:    "testmodel",
		Messages: []api.Message{{Role: "user", Content: "hi"}},
	}

	got, err := c.ChatCompletion(context.Background(), req)
	if err != nil {
		t.Fatalf("ChatCompletion error: %v", err)
	}
	if got.ID != want.ID {
		t.Errorf("ID = %q, want %q", got.ID, want.ID)
	}
	if len(got.Choices) != 1 {
		t.Fatalf("len(Choices) = %d, want 1", len(got.Choices))
	}
	if got.Choices[0].Message.Content != "Hello!" {
		t.Errorf("Content = %q, want %q", got.Choices[0].Message.Content, "Hello!")
	}
}

func TestClient_ChatCompletion_ErrorStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte("internal error"))
	}))
	defer srv.Close()

	c := NewClient(srv.URL)
	req := &api.ChatCompletionRequest{
		Model:    "testmodel",
		Messages: []api.Message{{Role: "user", Content: "hi"}},
	}

	_, err := c.ChatCompletion(context.Background(), req)
	if err == nil {
		t.Fatal("expected error for non-200 status")
	}
	if !strings.Contains(err.Error(), "500") {
		t.Errorf("error %q should contain '500'", err.Error())
	}
}

func TestClient_ChatCompletion_BadJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("{invalid json"))
	}))
	defer srv.Close()

	c := NewClient(srv.URL)
	req := &api.ChatCompletionRequest{
		Model:    "testmodel",
		Messages: []api.Message{{Role: "user", Content: "hi"}},
	}

	_, err := c.ChatCompletion(context.Background(), req)
	if err == nil {
		t.Fatal("expected error for bad JSON response")
	}
}

func TestClient_ChatCompletion_NetworkError(t *testing.T) {
	// Use a port that is not listening
	c := NewClient("http://127.0.0.1:1")
	req := &api.ChatCompletionRequest{
		Model:    "testmodel",
		Messages: []api.Message{{Role: "user", Content: "hi"}},
	}

	_, err := c.ChatCompletion(context.Background(), req)
	if err == nil {
		t.Fatal("expected network error")
	}
}

func TestClient_ChatCompletion_ContextCancelled(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Never respond
		<-r.Context().Done()
	}))
	defer srv.Close()

	c := NewClient(srv.URL)
	req := &api.ChatCompletionRequest{
		Model:    "testmodel",
		Messages: []api.Message{{Role: "user", Content: "hi"}},
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately

	_, err := c.ChatCompletion(ctx, req)
	if err == nil {
		t.Fatal("expected error for cancelled context")
	}
}

func TestClient_Tokenize_Success(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/tokenize" {
			http.Error(w, "unexpected path", http.StatusBadRequest)

			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"tokens": []int{1, 2, 3, 4, 5},
		})
	}))
	defer srv.Close()

	c := NewClient(srv.URL)
	count, err := c.Tokenize(context.Background(), "hello world")
	if err != nil {
		t.Fatalf("Tokenize error: %v", err)
	}
	if count != 5 {
		t.Errorf("count = %d, want 5", count)
	}
}

func TestClient_Tokenize_EmptyTokens(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"tokens": []int{},
		})
	}))
	defer srv.Close()

	c := NewClient(srv.URL)
	count, err := c.Tokenize(context.Background(), "")
	if err != nil {
		t.Fatalf("Tokenize error: %v", err)
	}
	if count != 0 {
		t.Errorf("count = %d, want 0", count)
	}
}

func TestClient_Tokenize_ErrorStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte("tokenize error"))
	}))
	defer srv.Close()

	c := NewClient(srv.URL)
	_, err := c.Tokenize(context.Background(), "hello")
	if err == nil {
		t.Fatal("expected error for non-200 status")
	}
	if !strings.Contains(err.Error(), "500") {
		t.Errorf("error %q should contain '500'", err.Error())
	}
}

func TestClient_Tokenize_BadJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("{invalid"))
	}))
	defer srv.Close()

	c := NewClient(srv.URL)
	_, err := c.Tokenize(context.Background(), "hello")
	if err == nil {
		t.Fatal("expected error for bad JSON response")
	}
}

func TestClient_Tokenize_NetworkError(t *testing.T) {
	c := NewClient("http://127.0.0.1:1")
	_, err := c.Tokenize(context.Background(), "hello")
	if err == nil {
		t.Fatal("expected network error")
	}
}

func TestClient_ChatCompletionStream_Success(t *testing.T) {
	sseData := "data: {\"id\":\"1\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n\n"

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			http.Error(w, "unexpected path", http.StatusBadRequest)

			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(sseData))
	}))
	defer srv.Close()

	c := NewClient(srv.URL)
	req := &api.ChatCompletionRequest{
		Model:    "testmodel",
		Messages: []api.Message{{Role: "user", Content: "hi"}},
		Stream:   true,
	}

	var buf strings.Builder
	err := c.ChatCompletionStream(context.Background(), req, &buf)
	if err != nil {
		t.Fatalf("ChatCompletionStream error: %v", err)
	}
	if buf.String() != sseData {
		t.Errorf("stream output = %q, want %q", buf.String(), sseData)
	}
}

func TestClient_ChatCompletionStream_ErrorStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte("unavailable"))
	}))
	defer srv.Close()

	c := NewClient(srv.URL)
	req := &api.ChatCompletionRequest{
		Model:    "testmodel",
		Messages: []api.Message{{Role: "user", Content: "hi"}},
		Stream:   true,
	}

	var buf strings.Builder
	err := c.ChatCompletionStream(context.Background(), req, &buf)
	if err == nil {
		t.Fatal("expected error for non-200 status")
	}
	if !strings.Contains(err.Error(), "503") {
		t.Errorf("error %q should contain '503'", err.Error())
	}
}

func TestClient_ChatCompletionStream_NetworkError(t *testing.T) {
	c := NewClient("http://127.0.0.1:1")
	req := &api.ChatCompletionRequest{
		Model:    "testmodel",
		Messages: []api.Message{{Role: "user", Content: "hi"}},
		Stream:   true,
	}

	var buf strings.Builder
	err := c.ChatCompletionStream(context.Background(), req, &buf)
	if err == nil {
		t.Fatal("expected network error")
	}
}

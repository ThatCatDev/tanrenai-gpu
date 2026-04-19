package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/ThatCatDev/tanrenai-gpu/internal/models"
	"github.com/ThatCatDev/tanrenai-gpu/internal/runner"
	"github.com/ThatCatDev/tanrenai-gpu/pkg/api"
)

// mockRunner implements runner.Runner for testing.
type mockRunner struct {
	modelName     string
	chatResp      *api.ChatCompletionResponse
	chatErr       error
	streamErr     error
	tokenizeCount int
	tokenizeErr   error
}

func (m *mockRunner) Load(_ context.Context, _ string, _ runner.Options) error { return nil }
func (m *mockRunner) Health(_ context.Context) error                           { return nil }
func (m *mockRunner) ModelName() string                                        { return m.modelName }
func (m *mockRunner) Close() error                                             { return nil }
func (m *mockRunner) ChatCompletion(_ context.Context, _ *api.ChatCompletionRequest) (*api.ChatCompletionResponse, error) {
	return m.chatResp, m.chatErr
}
func (m *mockRunner) ChatCompletionStream(_ context.Context, _ *api.ChatCompletionRequest, w io.Writer) error {
	if m.streamErr != nil {
		return m.streamErr
	}
	_, _ = w.Write([]byte("data: {}\n\n"))

	return nil
}
func (m *mockRunner) Tokenize(_ context.Context, _ string) (int, error) {
	return m.tokenizeCount, m.tokenizeErr
}

// ---- Health handler ----

func TestHealth_Returns200WithOKStatus(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	w := httptest.NewRecorder()

	Health(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("status = %d, want 200", resp.StatusCode)
	}

	var body map[string]string
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatalf("decode body: %v", err)
	}
	if body["status"] != "ok" {
		t.Errorf("body[status] = %q, want %q", body["status"], "ok")
	}
}

func TestHealth_ContentType(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	w := httptest.NewRecorder()

	Health(w, req)

	ct := w.Header().Get("Content-Type")
	if ct != "application/json" {
		t.Errorf("Content-Type = %q, want %q", ct, "application/json")
	}
}

// ---- Chat handler ----

func TestChatHandler_NoModelLoaded_NoModelInRequest(t *testing.T) {
	h := &ChatHandler{
		GetRunner: func() runner.Runner { return nil },
		LoadFunc:  nil, // should not be called
	}

	body := mustMarshal(t, api.ChatCompletionRequest{
		Messages: []api.Message{{Role: "user", Content: "hello"}},
		// Model is empty
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", w.Code)
	}
	assertErrorType(t, w, "invalid_request")
}

func TestChatHandler_BadJSON(t *testing.T) {
	h := &ChatHandler{
		GetRunner: func() runner.Runner { return nil },
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader([]byte("{invalid")))
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", w.Code)
	}
	assertErrorType(t, w, "invalid_request")
}

func TestChatHandler_EmptyMessages(t *testing.T) {
	h := &ChatHandler{
		GetRunner: func() runner.Runner { return nil },
	}

	body := mustMarshal(t, api.ChatCompletionRequest{
		Model:    "mymodel",
		Messages: []api.Message{}, // empty
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", w.Code)
	}
	assertErrorType(t, w, "invalid_request")
}

func TestChatHandler_ModelLoadFailure(t *testing.T) {
	h := &ChatHandler{
		GetRunner: func() runner.Runner { return nil },
		LoadFunc: func(_ context.Context, _ string) (*LoadResult, error) {
			return nil, errors.New("model not found")
		},
	}

	body := mustMarshal(t, api.ChatCompletionRequest{
		Model:    "nonexistent",
		Messages: []api.Message{{Role: "user", Content: "hi"}},
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusInternalServerError {
		t.Errorf("status = %d, want 500", w.Code)
	}
	assertErrorType(t, w, "model_error")
}

func TestChatHandler_NonStreamSuccess(t *testing.T) {
	mr := &mockRunner{
		modelName: "testmodel",
		chatResp: &api.ChatCompletionResponse{
			ID:    "test-id",
			Model: "testmodel",
			Choices: []api.Choice{
				{Message: api.Message{Role: "assistant", Content: "Hello!"}},
			},
		},
	}
	h := &ChatHandler{
		GetRunner: func() runner.Runner { return mr },
	}

	body := mustMarshal(t, api.ChatCompletionRequest{
		Model:    "testmodel",
		Messages: []api.Message{{Role: "user", Content: "hi"}},
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}

	var resp api.ChatCompletionResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if resp.ID != "test-id" {
		t.Errorf("resp.ID = %q, want %q", resp.ID, "test-id")
	}
}

func TestChatHandler_NonStreamError(t *testing.T) {
	mr := &mockRunner{
		modelName: "testmodel",
		chatErr:   errors.New("inference failed"),
	}
	h := &ChatHandler{
		GetRunner: func() runner.Runner { return mr },
	}

	body := mustMarshal(t, api.ChatCompletionRequest{
		Model:    "testmodel",
		Messages: []api.Message{{Role: "user", Content: "hi"}},
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusInternalServerError {
		t.Errorf("status = %d, want 500", w.Code)
	}
	assertErrorType(t, w, "inference_error")
}

func TestChatHandler_StreamSuccess(t *testing.T) {
	mr := &mockRunner{modelName: "testmodel"}
	h := &ChatHandler{
		GetRunner: func() runner.Runner { return mr },
	}

	body := mustMarshal(t, api.ChatCompletionRequest{
		Model:    "testmodel",
		Messages: []api.Message{{Role: "user", Content: "hi"}},
		Stream:   true,
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	ct := w.Header().Get("Content-Type")
	if ct != "text/event-stream" {
		t.Errorf("Content-Type = %q, want text/event-stream", ct)
	}
}

// ---- Tokenize handler ----

func TestTokenizeHandler_NoModelLoaded(t *testing.T) {
	h := &TokenizeHandler{
		GetRunner: func() runner.Runner { return nil },
	}

	body := mustMarshal(t, map[string]string{"content": "hello world"})
	req := httptest.NewRequest(http.MethodPost, "/tokenize", bytes.NewReader(body))
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want 503", w.Code)
	}
	assertErrorType(t, w, "no_model")
}

func TestTokenizeHandler_BadJSON(t *testing.T) {
	mr := &mockRunner{modelName: "testmodel", tokenizeCount: 5}
	h := &TokenizeHandler{
		GetRunner: func() runner.Runner { return mr },
	}

	req := httptest.NewRequest(http.MethodPost, "/tokenize", bytes.NewReader([]byte("{invalid")))
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", w.Code)
	}
}

func TestTokenizeHandler_Success(t *testing.T) {
	mr := &mockRunner{modelName: "testmodel", tokenizeCount: 5}
	h := &TokenizeHandler{
		GetRunner: func() runner.Runner { return mr },
	}

	body := mustMarshal(t, map[string]string{"content": "hello world"})
	req := httptest.NewRequest(http.MethodPost, "/tokenize", bytes.NewReader(body))
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}

	var resp map[string]any
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	tokens, ok := resp["tokens"]
	if !ok {
		t.Fatal("response missing 'tokens' field")
	}
	// JSON arrays decode as []interface{}
	tArr, ok := tokens.([]interface{})
	if !ok {
		t.Fatalf("tokens is %T, want []interface{}", tokens)
	}
	if len(tArr) != 5 {
		t.Errorf("len(tokens) = %d, want 5", len(tArr))
	}
}

func TestTokenizeHandler_RunnerError(t *testing.T) {
	mr := &mockRunner{modelName: "testmodel", tokenizeErr: errors.New("tokenize failed")}
	h := &TokenizeHandler{
		GetRunner: func() runner.Runner { return mr },
	}

	body := mustMarshal(t, map[string]string{"content": "hello"})
	req := httptest.NewRequest(http.MethodPost, "/tokenize", bytes.NewReader(body))
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusInternalServerError {
		t.Errorf("status = %d, want 500", w.Code)
	}
	assertErrorType(t, w, "tokenize_error")
}

// ---- Models handler ----

func TestModelsHandler_EmptyStore(t *testing.T) {
	tmp := t.TempDir()
	store := models.NewStore(tmp)
	h := &ModelsHandler{Store: store}

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}

	var resp api.ModelListResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if resp.Object != "list" {
		t.Errorf("Object = %q, want %q", resp.Object, "list")
	}
	if len(resp.Data) != 0 {
		t.Errorf("len(Data) = %d, want 0", len(resp.Data))
	}
}

func TestModelsHandler_WithModels(t *testing.T) {
	tmp := t.TempDir()
	// Create two fake GGUF files
	for _, name := range []string{"model-a.gguf", "model-b.gguf"} {
		if err := os.WriteFile(filepath.Join(tmp, name), []byte("fake"), 0644); err != nil {
			t.Fatal(err)
		}
	}

	store := models.NewStore(tmp)
	h := &ModelsHandler{Store: store}

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}

	var resp api.ModelListResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if len(resp.Data) != 2 {
		t.Errorf("len(Data) = %d, want 2", len(resp.Data))
	}
	for _, m := range resp.Data {
		if m.Object != "model" {
			t.Errorf("model Object = %q, want %q", m.Object, "model")
		}
		if m.OwnedBy != "local" {
			t.Errorf("model OwnedBy = %q, want %q", m.OwnedBy, "local")
		}
	}
}

// ---- LoadHandler ----

func TestLoadHandler_BadJSON(t *testing.T) {
	h := &LoadHandler{
		LoadFunc: func(_ context.Context, _ string) (*LoadResult, error) { return nil, nil },
	}

	req := httptest.NewRequest(http.MethodPost, "/api/load", bytes.NewReader([]byte("{invalid")))
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", w.Code)
	}
}

func TestLoadHandler_MissingModel(t *testing.T) {
	h := &LoadHandler{
		LoadFunc: func(_ context.Context, _ string) (*LoadResult, error) { return nil, nil },
	}

	body := mustMarshal(t, map[string]string{"model": ""})
	req := httptest.NewRequest(http.MethodPost, "/api/load", bytes.NewReader(body))
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", w.Code)
	}
	assertErrorType(t, w, "invalid_request")
}

func TestLoadHandler_LoadError(t *testing.T) {
	h := &LoadHandler{
		LoadFunc: func(_ context.Context, _ string) (*LoadResult, error) {
			return nil, errors.New("model not found")
		},
	}

	body := mustMarshal(t, map[string]string{"model": "missing-model"})
	req := httptest.NewRequest(http.MethodPost, "/api/load", bytes.NewReader(body))
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusInternalServerError {
		t.Errorf("status = %d, want 500", w.Code)
	}
	assertErrorType(t, w, "model_error")
}

func TestLoadHandler_Success(t *testing.T) {
	h := &LoadHandler{
		LoadFunc: func(_ context.Context, _ string) (*LoadResult, error) {
			return &LoadResult{CtxSize: 4096}, nil
		},
	}

	body := mustMarshal(t, map[string]string{"model": "mymodel"})
	req := httptest.NewRequest(http.MethodPost, "/api/load", bytes.NewReader(body))
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}

	var resp api.LoadResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if resp.Status != "loaded" {
		t.Errorf("Status = %q, want %q", resp.Status, "loaded")
	}
	if resp.Model != "mymodel" {
		t.Errorf("Model = %q, want %q", resp.Model, "mymodel")
	}
	if resp.CtxSize != 4096 {
		t.Errorf("CtxSize = %d, want 4096", resp.CtxSize)
	}
}

// ---- EmbeddingsHandler ----

func TestEmbeddingsHandler_NotConfigured(t *testing.T) {
	h := &EmbeddingsHandler{EmbeddingBaseURL: ""}

	body := mustMarshal(t, api.EmbeddingRequest{Input: "hello"})
	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", bytes.NewReader(body))
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want 503", w.Code)
	}
	assertErrorType(t, w, "no_embedding")
}

func TestEmbeddingsHandler_BadJSON(t *testing.T) {
	h := &EmbeddingsHandler{EmbeddingBaseURL: "http://localhost:9999"}

	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", bytes.NewReader([]byte("{invalid")))
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", w.Code)
	}
}

func TestEmbeddingsHandler_EmptyInput(t *testing.T) {
	h := &EmbeddingsHandler{EmbeddingBaseURL: "http://localhost:9999"}

	body := mustMarshal(t, api.EmbeddingRequest{Input: ""})
	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", bytes.NewReader(body))
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", w.Code)
	}
	assertErrorType(t, w, "invalid_request")
}

func TestEmbeddingsHandler_ProxiesRequest(t *testing.T) {
	// Start a fake embedding server that returns a valid response
	embeddingServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/embeddings" {
			http.Error(w, "unexpected path", http.StatusBadRequest)

			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(api.EmbeddingResponse{
			Data: []api.EmbeddingData{{Embedding: []float32{0.1, 0.2, 0.3}, Index: 0}},
		})
	}))
	defer embeddingServer.Close()

	h := &EmbeddingsHandler{EmbeddingBaseURL: embeddingServer.URL}

	body := mustMarshal(t, api.EmbeddingRequest{Input: "hello world"})
	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", bytes.NewReader(body))
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}
}

func TestEmbeddingsHandler_UpstreamError(t *testing.T) {
	// Upstream server returns 500
	embeddingServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte("internal error"))
	}))
	defer embeddingServer.Close()

	h := &EmbeddingsHandler{EmbeddingBaseURL: embeddingServer.URL}

	body := mustMarshal(t, api.EmbeddingRequest{Input: "hello"})
	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", bytes.NewReader(body))
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusInternalServerError {
		t.Errorf("status = %d, want 500", w.Code)
	}
}

// ---- normalizeModelName ----

func TestNormalizeModelName(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"model.gguf", "model"},
		{"model", "model"},
		{"my-model.gguf", "my-model"},
		{"path/to/model.gguf", "path/to/model"},
	}
	for _, tt := range tests {
		got := normalizeModelName(tt.input)
		if got != tt.want {
			t.Errorf("normalizeModelName(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

// ---- helpers ----

func mustMarshal(t *testing.T, v any) []byte {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	return b
}

func assertErrorType(t *testing.T, w *httptest.ResponseRecorder, wantType string) {
	t.Helper()
	var errResp api.ErrorResponse
	if err := json.NewDecoder(w.Body).Decode(&errResp); err != nil {
		t.Fatalf("decode error response: %v", err)
	}
	if errResp.Error.Type != wantType {
		t.Errorf("error.type = %q, want %q", errResp.Error.Type, wantType)
	}
}

package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/ThatCatDev/tanrenai-gpu/pkg/api"
)

// TestServer_Routes_Load_BadJSON exercises handleLoadModel.
func TestServer_Routes_Load_BadJSON(t *testing.T) {
	s := New(newTestConfig(t))

	req := httptest.NewRequest(http.MethodPost, "/api/load", bytes.NewReader([]byte("{invalid")))
	w := httptest.NewRecorder()
	s.http.Handler.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", w.Code)
	}
}

// TestServer_Routes_Load_MissingModel exercises handleLoadModel with empty model.
func TestServer_Routes_Load_MissingModel(t *testing.T) {
	s := New(newTestConfig(t))

	body, _ := json.Marshal(map[string]string{"model": ""})
	req := httptest.NewRequest(http.MethodPost, "/api/load", bytes.NewReader(body))
	w := httptest.NewRecorder()
	s.http.Handler.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", w.Code)
	}
}

// TestServer_Routes_Load_ModelNotFound exercises the load path where model doesn't exist.
func TestServer_Routes_Load_ModelNotFound(t *testing.T) {
	s := New(newTestConfig(t))

	body, _ := json.Marshal(map[string]string{"model": "nonexistent"})
	req := httptest.NewRequest(http.MethodPost, "/api/load", bytes.NewReader(body))
	w := httptest.NewRecorder()
	s.http.Handler.ServeHTTP(w, req)

	if w.Code != http.StatusInternalServerError {
		t.Errorf("status = %d, want 500 (model not found)", w.Code)
	}
}

// TestServer_Routes_Pull_BadJSON exercises handlePullModel.
func TestServer_Routes_Pull_BadJSON(t *testing.T) {
	s := New(newTestConfig(t))

	req := httptest.NewRequest(http.MethodPost, "/api/pull", bytes.NewReader([]byte("{invalid")))
	w := newFlushableTestRecorder()
	s.http.Handler.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", w.Code)
	}
}

// TestServer_Routes_Tokenize_NoModel exercises handleTokenize with no model loaded.
func TestServer_Routes_Tokenize_NoModel(t *testing.T) {
	s := New(newTestConfig(t))

	body, _ := json.Marshal(map[string]string{"content": "hello world"})
	req := httptest.NewRequest(http.MethodPost, "/tokenize", bytes.NewReader(body))
	w := httptest.NewRecorder()
	s.http.Handler.ServeHTTP(w, req)

	if w.Code != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want 503 (no model)", w.Code)
	}
}

// TestServer_Routes_Embeddings_NotConfigured exercises handleEmbeddings without embedding runner.
func TestServer_Routes_Embeddings_NotConfigured(t *testing.T) {
	s := New(newTestConfig(t))
	// No embedding runner set

	body, _ := json.Marshal(api.EmbeddingRequest{Input: "hello"})
	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", bytes.NewReader(body))
	w := httptest.NewRecorder()
	s.http.Handler.ServeHTTP(w, req)

	if w.Code != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want 503 (no embedding)", w.Code)
	}
}

// TestServer_Routes_Embeddings_WithRunner exercises handleEmbeddings with a runner.
func TestServer_Routes_Embeddings_WithRunner(t *testing.T) {
	// Start a fake embedding server
	embSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(api.EmbeddingResponse{
			Data: []api.EmbeddingData{{Embedding: []float32{0.1, 0.2}, Index: 0}},
		})
	}))
	defer embSrv.Close()

	s := New(newTestConfig(t))
	s.SetEmbeddingRunner(&EmbeddingSubprocess{BaseURL: embSrv.URL})

	body, _ := json.Marshal(api.EmbeddingRequest{Input: "hello"})
	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", bytes.NewReader(body))
	w := httptest.NewRecorder()
	s.http.Handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200 (body: %s)", w.Code, w.Body.String())
	}
}

// TestServer_Routes_CORS_Options exercises the CORS preflight handler.
func TestServer_Routes_CORS_Options(t *testing.T) {
	s := New(newTestConfig(t))

	req := httptest.NewRequest(http.MethodOptions, "/v1/chat/completions", nil)
	w := httptest.NewRecorder()
	s.http.Handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("OPTIONS status = %d, want 200", w.Code)
	}
	if w.Header().Get("Access-Control-Allow-Origin") != "*" {
		t.Errorf("CORS origin header missing")
	}
}

// TestServer_Routes_CORS_Headers verifies CORS headers on normal requests.
func TestServer_Routes_CORS_Headers(t *testing.T) {
	s := New(newTestConfig(t))

	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	w := httptest.NewRecorder()
	s.http.Handler.ServeHTTP(w, req)

	if w.Header().Get("Access-Control-Allow-Origin") != "*" {
		t.Error("CORS origin header should be set on all responses")
	}
}

// TestResponseWriter_WriteHeader verifies statusCode is captured.
func TestResponseWriter_WriteHeader(t *testing.T) {
	inner := httptest.NewRecorder()
	rw := &responseWriter{ResponseWriter: inner, statusCode: http.StatusOK}

	rw.WriteHeader(http.StatusNotFound)

	if rw.statusCode != http.StatusNotFound {
		t.Errorf("statusCode = %d, want 404", rw.statusCode)
	}
	if inner.Code != http.StatusNotFound {
		t.Errorf("inner.Code = %d, want 404", inner.Code)
	}
}

// TestResponseWriter_Flush verifies Flush works when underlying writer supports it.
func TestResponseWriter_Flush(t *testing.T) {
	inner := &flushRecorder{}
	rw := &responseWriter{ResponseWriter: inner}

	rw.Flush()

	if !inner.flushed {
		t.Error("Flush should delegate to underlying writer")
	}
}

// TestResponseWriter_Flush_NotFlusher verifies no panic when underlying writer doesn't support Flush.
func TestResponseWriter_Flush_NotFlusher(t *testing.T) {
	inner := httptest.NewRecorder() // does not implement http.Flusher
	rw := &responseWriter{ResponseWriter: inner}

	// Should not panic
	rw.Flush()
}

// TestServer_WrapLoadFunc exercises wrapLoadFunc.
func TestServer_WrapLoadFunc(t *testing.T) {
	s := New(newTestConfig(t))
	fn := s.wrapLoadFunc()

	// Call with a nonexistent model — should return error
	_, err := fn(nil, "nonexistent-model")
	if err == nil {
		t.Fatal("expected error for nonexistent model")
	}
}

// TestServer_RegisterRoutes_WithTrainingManager verifies finetune routes are registered
// when training manager is set at New() time.
func TestServer_RegisterRoutes_WithTrainingManager(t *testing.T) {
	cfg := newTestConfig(t)
	s := &Server{
		cfg:   cfg,
		store: nil, // intentionally nil for this test
	}
	// Set training manager before registerRoutes
	from := training_new_manager_for_test(t)
	s.trainingManager = from

	// Register routes manually
	mux := http.NewServeMux()
	s.store = newModelsStore(t)
	s.registerRoutes(mux)

	// Test that finetune routes exist
	req := httptest.NewRequest(http.MethodGet, "/v1/finetune/runs", nil)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code == http.StatusNotFound {
		t.Error("finetune/runs should be registered when training manager is set")
	}
}

// flushRecorder is an http.ResponseWriter that tracks Flush calls.
type flushRecorder struct {
	flushed bool
	header  http.Header
	code    int
	body    bytes.Buffer
}

func (f *flushRecorder) Header() http.Header {
	if f.header == nil {
		f.header = make(http.Header)
	}

	return f.header
}

func (f *flushRecorder) Write(b []byte) (int, error) { return f.body.Write(b) }

func (f *flushRecorder) WriteHeader(code int) { f.code = code }

func (f *flushRecorder) Flush() { f.flushed = true }

// flushableTestRecorder is a flushable ResponseRecorder for server route tests.
type flushableTestRecorder struct {
	*httptest.ResponseRecorder
}

func newFlushableTestRecorder() *flushableTestRecorder {
	return &flushableTestRecorder{httptest.NewRecorder()}
}

func (f *flushableTestRecorder) Flush() {}

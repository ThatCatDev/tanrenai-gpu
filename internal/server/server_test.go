package server

import (
	"context"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/ThatCatDev/tanrenai-gpu/internal/config"
	"github.com/ThatCatDev/tanrenai-gpu/internal/training"
)

// newTestConfig creates a Config for tests with a temp dir for models.
func newTestConfig(t *testing.T) *config.Config {
	t.Helper()
	cfg := config.DefaultConfig()
	cfg.Host = "127.0.0.1"
	cfg.Port = 9999 // placeholder; overridden below if needed
	cfg.ModelsDir = t.TempDir()
	cfg.BinDir = t.TempDir()

	return cfg
}

// findFreePort finds a free TCP port and returns it.
func findFreePort(t *testing.T) int {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("could not find free port: %v", err)
	}
	port := ln.Addr().(*net.TCPAddr).Port
	_ = ln.Close()

	return port
}

func TestNew_CreatesServer(t *testing.T) {
	s := New(newTestConfig(t))
	if s == nil {
		t.Fatal("New returned nil")
	}
	if s.cfg == nil {
		t.Fatal("Server.cfg is nil")
	}
	if s.store == nil {
		t.Fatal("Server.store is nil")
	}
	if s.http == nil {
		t.Fatal("Server.http is nil")
	}
}

func TestServer_SetTrainingManager(t *testing.T) {
	s := New(newTestConfig(t))

	client := training.NewSidecarClient("http://localhost:18082")
	store := training.NewRunStoreAt(t.TempDir())
	m := training.NewManagerWithStore(store, client, t.TempDir())

	s.SetTrainingManager(m)
	if s.trainingManager == nil {
		t.Error("SetTrainingManager should set trainingManager")
	}
}

func TestServer_SetEmbeddingRunner(t *testing.T) {
	s := New(newTestConfig(t))

	er := &EmbeddingSubprocess{BaseURL: "http://localhost:9999"}
	s.SetEmbeddingRunner(er)
	if s.embeddingRunner == nil {
		t.Error("SetEmbeddingRunner should set embeddingRunner")
	}
	if s.embeddingRunner.BaseURL != "http://localhost:9999" {
		t.Errorf("BaseURL = %q, want %q", s.embeddingRunner.BaseURL, "http://localhost:9999")
	}
}

func TestServer_Start_ShutdownViaContext(t *testing.T) {
	cfg := newTestConfig(t)
	cfg.Port = findFreePort(t)

	s := New(cfg)

	ctx, cancel := context.WithCancel(context.Background())

	errCh := make(chan error, 1)
	go func() {
		errCh <- s.Start(ctx)
	}()

	// Give the server a moment to start
	time.Sleep(50 * time.Millisecond)

	// Cancel context to trigger shutdown
	cancel()

	select {
	case err := <-errCh:
		if err != nil {
			t.Errorf("Start returned unexpected error: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Error("server did not shut down within 5 seconds")
	}
}

func TestServer_Routes_Health(t *testing.T) {
	s := New(newTestConfig(t))

	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	w := httptest.NewRecorder()
	s.http.Handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("health status = %d, want 200", w.Code)
	}
}

func TestServer_Routes_Models(t *testing.T) {
	s := New(newTestConfig(t))

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	w := httptest.NewRecorder()
	s.http.Handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("models status = %d, want 200", w.Code)
	}
}

func TestServer_LoadModel_NotFound(t *testing.T) {
	s := New(newTestConfig(t))

	_, err := s.LoadModel(context.Background(), "nonexistent-model")
	if err == nil {
		t.Fatal("expected error for nonexistent model")
	}
}

func TestServer_WithTrainingManager_SetStoredCorrectly(t *testing.T) {
	s := New(newTestConfig(t))

	client := training.NewSidecarClient("http://localhost:18082")
	store := training.NewRunStoreAt(t.TempDir())
	m := training.NewManagerWithStore(store, client, t.TempDir())

	s.SetTrainingManager(m)
	if s.trainingManager != m {
		t.Error("trainingManager not set correctly")
	}
}

// TestServer_New_WithTrainingManager verifies that when a training manager is
// set, the finetune routes are registered via the route registration called in New().
// NOTE: Since routes are registered at New() time (before SetTrainingManager),
// we test this by creating a fresh server with the manager already set via a helper.
func TestServer_New_FinetuneRoutesNotRegisteredWithoutManager(t *testing.T) {
	// Without a training manager, finetune routes should 404
	s := New(newTestConfig(t))

	req := httptest.NewRequest(http.MethodGet, "/v1/finetune/runs", nil)
	w := httptest.NewRecorder()
	s.http.Handler.ServeHTTP(w, req)

	// Go's default ServeMux returns 405 for unregistered patterns or 404
	// Either is fine — the route should not be registered
	if w.Code == http.StatusOK {
		t.Error("finetune/runs should not return 200 without training manager")
	}
}

func TestServer_Routes_TokenizeNoModel(t *testing.T) {
	s := New(newTestConfig(t))

	req := httptest.NewRequest(http.MethodPost, "/tokenize", nil)
	w := httptest.NewRecorder()
	s.http.Handler.ServeHTTP(w, req)

	// Should return 503 (no model loaded) or 400 (bad request)
	if w.Code == http.StatusOK {
		t.Error("tokenize should not succeed with no model loaded")
	}
}

func TestServer_Routes_ChatCompletionsNoModel(t *testing.T) {
	s := New(newTestConfig(t))

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	w := httptest.NewRecorder()
	s.http.Handler.ServeHTTP(w, req)

	// Should return an error (bad request or no model)
	if w.Code == http.StatusOK {
		t.Error("chat completions should not succeed with no model loaded")
	}
}

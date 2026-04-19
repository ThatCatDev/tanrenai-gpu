package training

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

// TestSidecarRunner_BaseURL verifies BaseURL returns the correct URL.
func TestSidecarRunner_BaseURL(t *testing.T) {
	r := &SidecarRunner{
		port:    18082,
		baseURL: "http://127.0.0.1:18082",
	}

	if r.BaseURL() != "http://127.0.0.1:18082" {
		t.Errorf("BaseURL() = %q, want %q", r.BaseURL(), "http://127.0.0.1:18082")
	}
}

// TestSidecarRunner_Close_NilCmd verifies Close works when cmd is nil.
func TestSidecarRunner_Close_NilCmd(t *testing.T) {
	r := &SidecarRunner{}
	if err := r.Close(); err != nil {
		t.Errorf("Close with nil cmd: unexpected error %v", err)
	}
}

// TestSidecarRunner_HealthCheck_OK verifies healthCheck returns nil for 200.
func TestSidecarRunner_HealthCheck_OK(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.WriteHeader(http.StatusOK)

			return
		}
		w.WriteHeader(http.StatusNotFound)
	}))
	defer srv.Close()

	r := &SidecarRunner{
		baseURL: srv.URL,
	}

	ctx := context.Background()
	if err := r.healthCheck(ctx); err != nil {
		t.Errorf("healthCheck: unexpected error %v", err)
	}
}

// TestSidecarRunner_HealthCheck_NonOK verifies healthCheck returns error for non-200.
func TestSidecarRunner_HealthCheck_NonOK(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer srv.Close()

	r := &SidecarRunner{
		baseURL: srv.URL,
	}

	if err := r.healthCheck(context.Background()); err == nil {
		t.Error("healthCheck: expected error for non-200 status, got nil")
	}
}

// TestSidecarRunner_HealthCheck_NetworkError verifies healthCheck returns error when server is unreachable.
func TestSidecarRunner_HealthCheck_NetworkError(t *testing.T) {
	r := &SidecarRunner{
		baseURL: "http://127.0.0.1:1", // unreachable port
	}

	if err := r.healthCheck(context.Background()); err == nil {
		t.Error("healthCheck: expected error for unreachable server, got nil")
	}
}

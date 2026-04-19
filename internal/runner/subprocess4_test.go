package runner

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os/exec"
	"testing"
)

// TestSubprocess_GracefulStop_LiveProcess verifies GracefulStop terminates
// a real running process cleanly via SIGTERM.
func TestSubprocess_GracefulStop_LiveProcess(t *testing.T) {
	// Start a process that stays alive until signaled
	cmd := exec.Command("sleep", "30")
	if err := cmd.Start(); err != nil {
		t.Skipf("cannot start sleep: %v", err)
	}

	doneCh := make(chan struct{})
	sub := &Subprocess{
		cmd:    cmd,
		label:  "test-graceful",
		doneCh: doneCh,
	}

	// Monitor the process in a goroutine to close doneCh when it exits
	go func() {
		_ = cmd.Wait()
		close(doneCh)
	}()

	if err := sub.GracefulStop(); err != nil {
		t.Errorf("GracefulStop: unexpected error %v", err)
	}

	if !sub.WasStopped() {
		t.Error("GracefulStop should set stopped=true")
	}
}

// TestSubprocess_HealthCheck_OK verifies healthCheck returns nil when server responds 200.
func TestSubprocess_HealthCheck_OK(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.WriteHeader(http.StatusOK)

			return
		}
		w.WriteHeader(http.StatusNotFound)
	}))
	defer srv.Close()

	sub := &Subprocess{
		baseURL: srv.URL,
	}

	if err := sub.healthCheck(context.Background()); err != nil {
		t.Errorf("healthCheck: unexpected error %v", err)
	}
}

// TestSubprocess_HealthCheck_NonOK verifies healthCheck returns error for non-200.
func TestSubprocess_HealthCheck_NonOK(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer srv.Close()

	sub := &Subprocess{
		baseURL: srv.URL,
	}

	if err := sub.healthCheck(context.Background()); err == nil {
		t.Error("healthCheck: expected error for non-200")
	}
}

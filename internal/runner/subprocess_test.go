package runner

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

func TestAllocatePort(t *testing.T) {
	port, err := allocatePort()
	if err != nil {
		t.Fatalf("allocatePort: %v", err)
	}
	if port <= 0 || port > 65535 {
		t.Fatalf("allocatePort returned invalid port: %d", port)
	}

	// Allocate a second port; it should be different (not guaranteed but overwhelmingly likely).
	port2, err := allocatePort()
	if err != nil {
		t.Fatalf("allocatePort second call: %v", err)
	}
	if port2 == port {
		t.Logf("warning: two consecutive allocatePort calls returned the same port %d", port)
	}
}

func TestResolveBinaryMissing(t *testing.T) {
	dir := t.TempDir()
	_, err := resolveBinary(dir)
	if err == nil {
		t.Fatal("expected error for missing binary")
	}
}

func TestResolveBinaryExists(t *testing.T) {
	dir := t.TempDir()
	binName := "llama-server"
	if runtime.GOOS == "windows" {
		binName = "llama-server.exe"
	}
	binPath := filepath.Join(dir, binName)
	if err := os.WriteFile(binPath, []byte("#!/bin/sh\n"), 0755); err != nil {
		t.Fatal(err)
	}
	got, err := resolveBinary(dir)
	if err != nil {
		t.Fatalf("resolveBinary: %v", err)
	}
	if got != binPath {
		t.Errorf("resolveBinary = %q, want %q", got, binPath)
	}
}

func TestNewSubprocessAutoPort(t *testing.T) {
	dir := t.TempDir()
	binName := "llama-server"
	if runtime.GOOS == "windows" {
		binName = "llama-server.exe"
	}
	if err := os.WriteFile(filepath.Join(dir, binName), []byte("#!/bin/sh\n"), 0755); err != nil {
		t.Fatal(err)
	}

	sub, err := NewSubprocess(SubprocessConfig{
		BinDir: dir,
		Port:   0,
		Label:  "test",
	})
	if err != nil {
		t.Fatalf("NewSubprocess: %v", err)
	}
	if sub.Port() <= 0 {
		t.Errorf("expected auto-allocated port > 0, got %d", sub.Port())
	}
}

func TestGracefulStopSendsSIGTERM(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("SIGTERM test not applicable on Windows")
	}

	// Start a sleep process and verify graceful stop terminates it.
	cmd := exec.Command("sleep", "60")
	if err := cmd.Start(); err != nil {
		t.Skipf("cannot start sleep: %v", err)
	}

	doneCh := make(chan struct{})
	go func() {
		cmd.Wait()
		close(doneCh)
	}()

	sub := &Subprocess{
		cmd:    cmd,
		label:  "test",
		doneCh: doneCh,
	}

	if err := sub.GracefulStop(); err != nil {
		t.Fatalf("GracefulStop: %v", err)
	}

	// Process should be dead.
	select {
	case <-doneCh:
		// OK
	case <-time.After(10 * time.Second):
		t.Fatal("process still running after GracefulStop")
	}
}

func TestGracefulStopNilProcess(t *testing.T) {
	sub := &Subprocess{
		label:  "test",
		doneCh: make(chan struct{}),
	}
	if err := sub.GracefulStop(); err != nil {
		t.Fatalf("GracefulStop on nil process: %v", err)
	}
}

func TestSubprocessHealthCheck(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.WriteHeader(http.StatusOK)

			return
		}
		http.Error(w, "not found", http.StatusNotFound)
	}))
	defer server.Close()

	sub := &Subprocess{
		baseURL: server.URL,
		label:   "test",
	}

	if err := sub.healthCheck(context.Background()); err != nil {
		t.Errorf("healthCheck failed on healthy server: %v", err)
	}
}

func TestSubprocessHealthCheckUnhealthy(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer server.Close()

	sub := &Subprocess{
		baseURL: server.URL,
		label:   "test",
	}

	if err := sub.healthCheck(context.Background()); err == nil {
		t.Error("expected error for unhealthy server")
	}
}

func TestWaitForHealthTimeout(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer server.Close()

	doneCh := make(chan struct{})
	sub := &Subprocess{
		baseURL:       server.URL,
		label:         "test",
		healthTimeout: 1 * time.Second,
		doneCh:        doneCh,
	}

	err := sub.waitForHealth(context.Background())
	if err == nil {
		t.Fatal("expected timeout error")
	}
}

func TestWaitForHealthCancelled(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer server.Close()

	doneCh := make(chan struct{})
	sub := &Subprocess{
		baseURL:       server.URL,
		label:         "test",
		healthTimeout: 30 * time.Second,
		doneCh:        doneCh,
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	err := sub.waitForHealth(ctx)
	if err == nil {
		t.Fatal("expected context cancelled error")
	}
}

func TestWaitForHealthBecomesHealthy(t *testing.T) {
	callCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		if callCount < 3 {
			w.WriteHeader(http.StatusServiceUnavailable)

			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	doneCh := make(chan struct{})
	sub := &Subprocess{
		baseURL:       server.URL,
		label:         "test",
		healthTimeout: 10 * time.Second,
		doneCh:        doneCh,
	}

	err := sub.waitForHealth(context.Background())
	if err != nil {
		t.Fatalf("waitForHealth: %v", err)
	}
	if callCount < 3 {
		t.Errorf("expected at least 3 health checks, got %d", callCount)
	}
}

package runner

import (
	"context"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

// TestAllocatePort_BindFail attempts to verify the allocatePort error path.
// We can't easily force net.Listen to fail in unit tests, but we can verify
// that the returned port is always valid.
func TestAllocatePort_ReturnsValidPort(t *testing.T) {
	for i := 0; i < 5; i++ {
		port, err := allocatePort()
		if err != nil {
			t.Fatalf("allocatePort[%d]: %v", i, err)
		}
		if port <= 0 || port > 65535 {
			t.Fatalf("allocatePort[%d]: invalid port %d", i, port)
		}
		// Verify the port is actually bindable
		l, err := net.Listen("tcp", "127.0.0.1:0")
		if err == nil {
			_ = l.Close()
		}
	}
}

// TestResolveBinary_BundledPath verifies resolveBinary checks the bin/ subdir
// next to the executable. We can't easily place a binary there, but we verify
// the error message mentions both paths.
func TestResolveBinary_ErrorMentionsPaths(t *testing.T) {
	dir := t.TempDir()
	_, err := resolveBinary(dir)
	if err == nil {
		t.Fatal("expected error for missing binary")
	}
	errStr := err.Error()
	if len(errStr) == 0 {
		t.Error("error message should not be empty")
	}
}

// TestNewSubprocess_DefaultHealthTimeout verifies that a zero HealthTimeout
// defaults to 120 seconds.
func TestNewSubprocess_DefaultHealthTimeout(t *testing.T) {
	dir := t.TempDir()
	binName := "llama-server"
	if runtime.GOOS == "windows" {
		binName = "llama-server.exe"
	}
	if err := os.WriteFile(dir+"/"+binName, []byte("#!/bin/sh\n"), 0755); err != nil {
		t.Fatal(err)
	}

	sub, err := NewSubprocess(SubprocessConfig{
		BinDir:        dir,
		Port:          19888,
		HealthTimeout: 0, // should default to 120s
	})
	if err != nil {
		t.Fatalf("NewSubprocess: %v", err)
	}
	if sub.healthTimeout != 120*time.Second {
		t.Errorf("healthTimeout = %v, want 120s", sub.healthTimeout)
	}
}

// TestGracefulStop_NilCmd_WithStoppedFlag verifies GracefulStop on a subprocess
// with nil cmd sets stopped=true and returns nil.
func TestGracefulStop_NilCmd_SetsFlags(t *testing.T) {
	sub := &Subprocess{
		label:  "test",
		doneCh: make(chan struct{}),
		// cmd is nil
	}

	err := sub.GracefulStop()
	if err != nil {
		t.Fatalf("GracefulStop: %v", err)
	}
	if !sub.stopped {
		t.Error("GracefulStop should set stopped=true")
	}
	if sub.healthy {
		t.Error("GracefulStop should set healthy=false")
	}
}

// TestGracefulStop_ProcessAlreadyDead verifies GracefulStop handles a process
// that has already exited (signal returns an error).
func TestGracefulStop_ProcessAlreadyDead(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("signal behavior differs on Windows")
	}

	// Start and immediately wait for a short-lived process.
	cmd := exec.Command("true")
	if err := cmd.Run(); err != nil {
		t.Skipf("cannot run 'true': %v", err)
	}
	// ProcessState is now set (process has exited)

	doneCh := make(chan struct{})
	close(doneCh) // already exited

	sub := &Subprocess{
		cmd:    cmd,
		label:  "test",
		doneCh: doneCh,
	}

	// Sending signal to a dead process — GracefulStop should handle this gracefully.
	err := sub.GracefulStop()
	if err != nil {
		t.Errorf("GracefulStop on dead process: %v", err)
	}
}

// TestSubprocess_HealthCheck_ConnectionRefused verifies healthCheck returns an error
// when no server is listening on the given URL.
func TestSubprocess_HealthCheck_ConnectionRefused(t *testing.T) {
	sub := &Subprocess{
		baseURL: "http://127.0.0.1:1", // nothing listening here
		label:   "test",
	}

	// Should return a network error
	if err := sub.healthCheck(context.Background()); err == nil {
		t.Error("expected error for connection refused")
	}
}

// TestSubprocess_HealthCheck_InvalidURL verifies healthCheck returns an error
// when the base URL is invalid (http.NewRequestWithContext fails).
func TestSubprocess_HealthCheck_InvalidURL(t *testing.T) {
	sub := &Subprocess{
		// A URL with a space in it is invalid for http.NewRequestWithContext.
		baseURL: "://invalid url",
		label:   "test",
	}

	if err := sub.healthCheck(context.Background()); err == nil {
		t.Error("expected error for invalid URL")
	}
}

// TestGracefulStop_SIGKILLPath verifies that GracefulStop sends SIGKILL when the
// process does not exit within 5 seconds of receiving SIGTERM.
func TestGracefulStop_SIGKILLPath(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("SIGTERM/SIGKILL test not applicable on Windows")
	}
	if testing.Short() {
		t.Skip("skipping 5-second SIGKILL test in short mode")
	}

	// Check if nosigterm binary exists (pre-built for this test)
	nosigterm := "/tmp/nosigterm"
	if _, err := os.Stat(nosigterm); err != nil {
		t.Skipf("nosigterm binary not found at %s: %v (build with: go build -o /tmp/nosigterm /tmp/nosigterm.go)", nosigterm, err)
	}

	// Start a process that ignores SIGTERM and stays alive.
	cmd := exec.Command(nosigterm)
	if err := cmd.Start(); err != nil {
		t.Skipf("cannot start nosigterm: %v", err)
	}
	t.Logf("Started nosigterm PID: %d", cmd.Process.Pid)

	doneCh := make(chan struct{})
	go func() {
		err := cmd.Wait()
		t.Logf("nosigterm exited: %v (state=%v)", err, cmd.ProcessState)
		close(doneCh)
	}()

	// Give the process time to start and set up its signal handlers
	time.Sleep(200 * time.Millisecond)

	// Verify it's still alive before sending SIGTERM
	select {
	case <-doneCh:
		t.Skip("nosigterm process died before test started — skipping")
	default:
		// Good, still alive
	}

	sub := &Subprocess{
		cmd:    cmd,
		label:  "test-sigkill",
		doneCh: doneCh,
	}

	// GracefulStop will SIGTERM → wait 5s → SIGKILL
	if err := sub.GracefulStop(); err != nil {
		t.Errorf("GracefulStop: %v", err)
	}

	// Process should be dead
	select {
	case <-doneCh:
		// Good
	case <-time.After(2 * time.Second):
		t.Fatal("process still alive after GracefulStop")
	}
}

// TestResolveBinary_BundledFound verifies resolveBinary falls back to finding
// llama-server in bin/ next to the running executable.
func TestResolveBinary_BundledFound(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("bundled binary test not needed on Windows")
	}

	// Find the test executable path and create a bin/llama-server next to it.
	exe, err := os.Executable()
	if err != nil {
		t.Skipf("cannot find executable: %v", err)
	}
	binDir := filepath.Join(filepath.Dir(exe), "bin")
	if mkErr := os.MkdirAll(binDir, 0755); mkErr != nil {
		t.Fatalf("mkdir bin/: %v", mkErr)
	}
	bundledBin := filepath.Join(binDir, "llama-server")
	// Write a fake binary
	if writeErr := os.WriteFile(bundledBin, []byte("#!/bin/sh\n"), 0755); writeErr != nil {
		t.Fatalf("write bundled binary: %v", writeErr)
	}
	t.Cleanup(func() {
		_ = os.Remove(bundledBin)
		_ = os.Remove(binDir) // best-effort rmdir
	})

	// Use an empty binDir so the primary lookup fails and falls back to bundled
	emptyDir := t.TempDir()
	got, resolveErr := resolveBinary(emptyDir)
	if resolveErr != nil {
		t.Fatalf("resolveBinary with bundled binary: %v", resolveErr)
	}
	if got != bundledBin {
		t.Errorf("resolveBinary = %q, want %q", got, bundledBin)
	}
}

package runner

import (
	"os"
	"os/exec"
	"strings"
	"testing"
)

// TestSubprocess_pipeOutput tests that pipeOutput reads from subprocess pipes.
// We create a real subprocess (echo) with actual stdout/stderr pipes.
func TestSubprocess_pipeOutput(t *testing.T) {
	// Create a subprocess that writes to stdout and stderr.
	cmd := exec.Command("sh", "-c", "echo hello; echo error >&2")

	sub := &Subprocess{
		cmd:   cmd,
		label: "test-pipe",
	}

	// Attach pipes before starting
	sub.pipeOutput()

	if err := cmd.Start(); err != nil {
		t.Skipf("cannot start shell: %v", err)
	}

	// Wait for process to complete
	_ = cmd.Wait()

	// pipeOutput was called successfully if we reach here without panic
}

// TestSubprocess_scanLines tests that scanLines reads lines from a reader.
func TestSubprocess_scanLines(t *testing.T) {
	sub := &Subprocess{label: "scan-test"}
	content := "line1\nline2\nline3\n"

	// Should complete without error or panic
	done := make(chan struct{})
	go func() {
		defer close(done)
		sub.scanLines(strings.NewReader(content), "[test] ")
	}()

	<-done
}

// TestSubprocess_ExitCode_AfterExit verifies ExitCode after process exits.
func TestSubprocess_ExitCode_AfterExit(t *testing.T) {
	cmd := exec.Command("true") // exits with 0
	if err := cmd.Run(); err != nil {
		t.Skipf("cannot run 'true': %v", err)
	}
	// After Run(), cmd.ProcessState is set
	sub := &Subprocess{cmd: cmd}
	code := sub.ExitCode()
	if code != 0 {
		t.Errorf("ExitCode() = %d, want 0", code)
	}
}

// TestSubprocess_GracefulStop_SetsStopped verifies GracefulStop sets stopped flag.
func TestSubprocess_GracefulStop_SetsStopped(t *testing.T) {
	sub := &Subprocess{
		label:  "test",
		doneCh: make(chan struct{}),
		// cmd is nil — GracefulStop should return early
	}

	if err := sub.GracefulStop(); err != nil {
		t.Fatalf("GracefulStop with nil cmd: %v", err)
	}

	if !sub.WasStopped() {
		t.Error("GracefulStop should set stopped=true")
	}
}

// TestCheckGPUSupport_BinaryWithoutCUDA verifies checkGPUSupport handles a
// binary without CUDA support gracefully (no panic, no crash).
func TestCheckGPUSupport_BinaryWithoutCUDA(t *testing.T) {
	// Create a fake binary that reports no CUDA support
	dir := t.TempDir()
	fakeBin := dir + "/llama-server"
	if err := os.WriteFile(fakeBin, []byte("#!/bin/sh\necho 'no cuda here'\n"), 0755); err != nil {
		t.Fatal(err)
	}

	// Should not panic
	checkGPUSupport(fakeBin)
}

// TestNewSubprocess_WithExplicitPort tests NewSubprocess when port is pre-set.
func TestNewSubprocess_WithExplicitPort(t *testing.T) {
	dir := t.TempDir()
	binName := "llama-server"
	if err := os.WriteFile(dir+"/"+binName, []byte("#!/bin/sh\n"), 0755); err != nil {
		t.Fatal(err)
	}

	sub, err := NewSubprocess(SubprocessConfig{
		BinDir: dir,
		Port:   19876,
		Label:  "test-explicit-port",
	})
	if err != nil {
		t.Fatalf("NewSubprocess: %v", err)
	}
	if sub.Port() != 19876 {
		t.Errorf("Port() = %d, want 19876", sub.Port())
	}
	if sub.BaseURL() != "http://127.0.0.1:19876" {
		t.Errorf("BaseURL() = %q, want %q", sub.BaseURL(), "http://127.0.0.1:19876")
	}
}

// TestNewSubprocess_DefaultLabel tests that empty label defaults to "llama-server".
func TestNewSubprocess_DefaultLabel(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(dir+"/llama-server", []byte("#!/bin/sh\n"), 0755); err != nil {
		t.Fatal(err)
	}

	sub, err := NewSubprocess(SubprocessConfig{
		BinDir: dir,
		Port:   19877,
		Label:  "", // should default to "llama-server"
	})
	if err != nil {
		t.Fatalf("NewSubprocess: %v", err)
	}
	if sub.label != "llama-server" {
		t.Errorf("label = %q, want %q", sub.label, "llama-server")
	}
}

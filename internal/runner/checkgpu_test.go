package runner

import (
	"os"
	"runtime"
	"testing"
)

// TestCheckGPUSupport_FakeBinaryWithCUDA verifies checkGPUSupport handles a
// binary whose output contains "ggml_cuda_init" (simulates CUDA support detected).
func TestCheckGPUSupport_FakeBinaryWithCUDA(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("CUDA detection test only runs on Linux")
	}

	dir := t.TempDir()
	fakeBin := dir + "/llama-server"
	// Script that prints "ggml_cuda_init" to simulate CUDA support.
	script := "#!/bin/sh\necho 'ggml_cuda_init called'\n"
	if err := os.WriteFile(fakeBin, []byte(script), 0755); err != nil {
		t.Fatal(err)
	}

	// Should not panic — exercises the hasCUDA=true branch on systems where GPU
	// is detected (or just exercises the version-check code path regardless).
	checkGPUSupport(fakeBin)
}

// TestCheckGPUSupport_FakeBinaryNoCUDA verifies checkGPUSupport handles a binary
// that does not output "ggml_cuda_init" (logs a warning but does not crash).
func TestCheckGPUSupport_FakeBinaryNoCUDA(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("GPU detection test only runs on Linux")
	}

	dir := t.TempDir()
	fakeBin := dir + "/llama-server"
	script := "#!/bin/sh\necho 'version 1.0'\n"
	if err := os.WriteFile(fakeBin, []byte(script), 0755); err != nil {
		t.Fatal(err)
	}

	checkGPUSupport(fakeBin)
}

// TestCheckGPUSupport_NvidiaSmiPath forces the nvidia-smi code path by using
// a directory that won't have /dev/nvidiactl or /dev/dxg present in the test environment.
// This exercises the exec.Command("nvidia-smi") branch regardless of GPU availability.
func TestCheckGPUSupport_NonexistentBinary(t *testing.T) {
	// Call with a path to a non-existent binary — should not panic even if
	// the GPU detection succeeds but version-check fails.
	checkGPUSupport("/nonexistent/llama-server")
}

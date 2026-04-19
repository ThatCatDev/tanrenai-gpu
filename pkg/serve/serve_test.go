package serve

import (
	"context"
	"net"
	"os"
	"testing"
	"time"
)

// findFreePort finds an available TCP port.
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

func TestModelsDir(t *testing.T) {
	dir := ModelsDir()
	if dir == "" {
		t.Error("ModelsDir() returned empty string")
	}
}

func TestResolveModel_NotFound(t *testing.T) {
	_, err := ResolveModel("nonexistent-model-xyz")
	if err == nil {
		t.Fatal("expected error for nonexistent model")
	}
}

func TestDownloadModel_InvalidURL(t *testing.T) {
	// .invalid is reserved (RFC 2606) — fails fast at DNS rather than
	// burning each retry's OS connect timeout.
	_, err := DownloadModel("http://no-such-host.invalid/model.gguf", t.TempDir(), nil)
	if err == nil {
		t.Fatal("expected error for unreachable URL")
	}
}

func TestDownloadModel_NonGGUF(t *testing.T) {
	_, err := DownloadModel("http://example.com/model.bin", t.TempDir(), nil)
	if err == nil {
		t.Fatal("expected error for non-.gguf URL")
	}
}

func TestStart_UnknownChatTemplate(t *testing.T) {
	port := findFreePort(t)
	_ = port
	ctx := context.Background()
	err := Start(ctx, Config{
		Host:         "127.0.0.1",
		Port:         findFreePort(t),
		ChatTemplate: "unknown-template-xyz",
	})
	if err == nil {
		t.Fatal("expected error for unknown chat template")
	}
}

func TestStart_ShutdownViaContext(t *testing.T) {
	port := findFreePort(t)
	ctx, cancel := context.WithCancel(context.Background())

	errCh := make(chan error, 1)
	go func() {
		errCh <- Start(ctx, Config{
			Host: "127.0.0.1",
			Port: port,
		})
	}()

	// Give server time to start
	time.Sleep(50 * time.Millisecond)
	cancel()

	select {
	case err := <-errCh:
		if err != nil {
			t.Errorf("Start returned unexpected error: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Error("server did not shut down in time")
	}
}

func TestStart_WithAllConfigOptions(t *testing.T) {
	port := findFreePort(t)
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()

	err := Start(ctx, Config{
		Host:            "127.0.0.1",
		Port:            port,
		ModelsDir:       t.TempDir(),
		BinDir:          t.TempDir(),
		GPULayers:       32,
		CtxSize:         4096,
		FlashAttention:  true,
		ReasoningFormat: "deepseek",
		NoAutoTemplate:  true,
	})
	// Context timeout or nil is expected
	if err != nil && err != context.DeadlineExceeded {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestStart_ChatTemplateFileDirect(t *testing.T) {
	// Create a temp Jinja template file
	tplFile := t.TempDir() + "/template.jinja"
	if err := os.WriteFile(tplFile, []byte("{{ messages }}"), 0644); err != nil {
		t.Fatalf("create template file: %v", err)
	}

	port := findFreePort(t)
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()

	err := Start(ctx, Config{
		Host:             "127.0.0.1",
		Port:             port,
		ChatTemplateFile: tplFile,
	})
	if err != nil && err != context.DeadlineExceeded {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestStart_ChatMLTemplate(t *testing.T) {
	port := findFreePort(t)
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()

	err := Start(ctx, Config{
		Host:         "127.0.0.1",
		Port:         port,
		ChatTemplate: "chatml",
	})
	if err != nil && err != context.DeadlineExceeded {
		t.Errorf("unexpected error: %v", err)
	}
}

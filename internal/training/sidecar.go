package training

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"time"
)

// SidecarConfig configures the Python training sidecar.
type SidecarConfig struct {
	PythonPath string // path to python3 binary (default: "python3")
	SidecarDir string // directory containing main.py
	Port       int    // port to listen on (default: 18082)
}

// SidecarRunner manages the Python sidecar subprocess.
type SidecarRunner struct {
	cmd     *exec.Cmd
	port    int
	baseURL string
}

// NewSidecarRunner starts the Python training sidecar.
func NewSidecarRunner(ctx context.Context, cfg SidecarConfig) (*SidecarRunner, error) {
	if cfg.PythonPath == "" {
		cfg.PythonPath = "python3"
	}
	if cfg.Port == 0 {
		cfg.Port = 18082
	}

	args := []string{
		"-m", "uvicorn", "main:app",
		"--host", "127.0.0.1",
		"--port", fmt.Sprintf("%d", cfg.Port),
	}

	cmd := exec.CommandContext(ctx, cfg.PythonPath, args...)
	cmd.Dir = cfg.SidecarDir
	cmd.Stdout = io.Discard
	cmd.Stderr = io.Discard
	cmd.Env = os.Environ()

	r := &SidecarRunner{
		cmd:     cmd,
		port:    cfg.Port,
		baseURL: fmt.Sprintf("http://127.0.0.1:%d", cfg.Port),
	}

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("failed to start training sidecar: %w", err)
	}

	if err := r.waitForHealth(ctx, 30*time.Second); err != nil {
		_ = r.Close()

		return nil, fmt.Errorf("training sidecar failed to start: %w", err)
	}

	return r, nil
}

// BaseURL returns the base URL of the sidecar.
func (r *SidecarRunner) BaseURL() string {
	return r.baseURL
}

// Close stops the sidecar subprocess.
func (r *SidecarRunner) Close() error {
	if r.cmd != nil && r.cmd.Process != nil {
		if err := r.cmd.Process.Kill(); err != nil {
			return fmt.Errorf("failed to kill training sidecar: %w", err)
		}
		_ = r.cmd.Wait()
	}

	return nil
}

func (r *SidecarRunner) waitForHealth(ctx context.Context, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			if time.Now().After(deadline) {
				return fmt.Errorf("timeout waiting for training sidecar after %s", timeout)
			}
			if r.healthCheck(ctx) == nil {
				return nil
			}
		}
	}
}

func (r *SidecarRunner) healthCheck(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, r.baseURL+"/health", nil)
	if err != nil {
		return err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("health check returned %d", resp.StatusCode)
	}

	return nil
}

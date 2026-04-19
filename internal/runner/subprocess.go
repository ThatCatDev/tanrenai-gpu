package runner

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

// Subprocess manages the lifecycle of a llama-server child process.
// It handles binary resolution, environment setup, process start/stop,
// structured logging, health polling, and graceful shutdown.
type Subprocess struct {
	cmd  *exec.Cmd
	mu   sync.Mutex
	port int

	binPath       string
	args          []string
	env           []string
	label         string // log prefix, e.g. "llama-server" or "embedding"
	quiet         bool
	baseURL       string
	healthy       bool
	stopped       bool          // true after explicit Close()
	doneCh        chan struct{} // closed when the process exits
	healthTimeout time.Duration

	// tailBuf keeps the last N lines of the subprocess's merged
	// stdout/stderr so we can attach them to crash errors. llama-server's
	// OOM / SIGSEGV / loader messages arrive here; without this buffer
	// the caller only ever sees "exit code -1" and has to guess at why.
	tailBuf   []string
	tailMu    sync.Mutex
	tailLimit int
}

// SubprocessConfig holds everything needed to start a llama-server subprocess.
type SubprocessConfig struct {
	BinDir        string
	Args          []string      // args to pass after the binary path
	Port          int           // 0 = auto-allocate
	Label         string        // log prefix (default "llama-server")
	Quiet         bool          // suppress subprocess stdout/stderr
	HealthTimeout time.Duration // how long to wait for /health (default 120s)
}

// allocatePort finds a free TCP port by binding to :0 and releasing it.
func allocatePort() (int, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, fmt.Errorf("allocate port: %w", err)
	}
	port := l.Addr().(*net.TCPAddr).Port
	_ = l.Close()

	return port, nil
}

// resolveBinary finds llama-server by checking multiple locations:
//  1. The configured binDir (e.g. ~/.local/share/tanrenai/bin/)
//  2. A bin/ subdirectory next to the running executable (bundled releases)
func resolveBinary(binDir string) (string, error) {
	binName := "llama-server"
	if runtime.GOOS == "windows" {
		binName = "llama-server.exe"
	}

	// Check configured binDir first.
	binPath := filepath.Join(binDir, binName)
	if _, err := os.Stat(binPath); err == nil {
		return binPath, nil
	}

	// Check bin/ next to the running executable (bundled release layout).
	if exe, err := os.Executable(); err == nil {
		bundled := filepath.Join(filepath.Dir(exe), "bin", binName)
		if _, err := os.Stat(bundled); err == nil {
			return bundled, nil
		}
	}

	return "", fmt.Errorf("llama-server not found at %s — place it in the bin/ directory next to the tanrenai executable, or in %s", binPath, binDir)
}

// checkGPUSupport detects whether GPU acceleration will be used and logs the result.
func checkGPUSupport(binPath string) {
	hasGPU := false
	hasCUDA := false

	switch runtime.GOOS {
	case "linux":
		// Check native Linux and WSL GPU devices.
		if _, err := os.Stat("/dev/nvidiactl"); err == nil {
			hasGPU = true
		} else if _, err := os.Stat("/dev/dxg"); err == nil {
			// WSL2 exposes GPU via /dev/dxg
			hasGPU = true
		} else if err := exec.Command("nvidia-smi").Run(); err == nil {
			hasGPU = true
		}
	case "darwin":
		// macOS with Apple Silicon always has Metal GPU support.
		if runtime.GOARCH == "arm64" {
			slog.Info("Using GPU acceleration (Metal)")

			return
		}
	case "windows":
		// Check for NVIDIA GPU via nvidia-smi.
		if err := exec.Command("nvidia-smi").Run(); err == nil {
			hasGPU = true
		}
	}

	if !hasGPU {
		slog.Info("No GPU detected — using CPU inference")

		return
	}

	// Run llama-server --version to check if it actually loads the CUDA backend.
	binDir := filepath.Dir(binPath)
	cmd := exec.Command(binPath, "--version")
	cmd.Dir = binDir
	env := os.Environ()
	env = append(env, "LD_LIBRARY_PATH="+binDir+":"+os.Getenv("LD_LIBRARY_PATH"))
	cmd.Env = env
	out, err := cmd.CombinedOutput()
	if err == nil && strings.Contains(string(out), "ggml_cuda_init") {
		hasCUDA = true
	}

	if hasCUDA {
		slog.Info("Using GPU acceleration (CUDA)")
	} else {
		slog.Warn("NVIDIA GPU detected but llama-server lacks CUDA support — using CPU only. See docs for GPU setup.")
	}
}

// NewSubprocess creates a Subprocess but does not start it. Call Start() next.
func NewSubprocess(cfg SubprocessConfig) (*Subprocess, error) {
	binPath, err := resolveBinary(cfg.BinDir)
	if err != nil {
		return nil, err
	}

	checkGPUSupport(binPath)

	port := cfg.Port
	if port == 0 {
		port, err = allocatePort()
		if err != nil {
			return nil, err
		}
	}

	label := cfg.Label
	if label == "" {
		label = "llama-server"
	}

	healthTimeout := cfg.HealthTimeout
	if healthTimeout == 0 {
		healthTimeout = 120 * time.Second
	}

	// Set library path so CUDA/other shared libs next to llama-server are found.
	binDir := filepath.Dir(binPath)
	env := os.Environ()
	switch runtime.GOOS {
	case "windows":
		env = append(env, "PATH="+binDir+";"+os.Getenv("PATH"))
	case "darwin":
		env = append(env, "DYLD_LIBRARY_PATH="+binDir+":"+os.Getenv("DYLD_LIBRARY_PATH"))
	default:
		env = append(env, "LD_LIBRARY_PATH="+binDir+":"+os.Getenv("LD_LIBRARY_PATH"))
	}

	return &Subprocess{
		binPath:       binPath,
		args:          cfg.Args,
		env:           env,
		port:          port,
		label:         label,
		quiet:         cfg.Quiet,
		baseURL:       fmt.Sprintf("http://127.0.0.1:%d", port),
		healthTimeout: healthTimeout,
		doneCh:        make(chan struct{}),
		tailLimit:     50,
	}, nil
}

// pushTail appends a line to the ring buffer, evicting the oldest when
// over tailLimit. Thread-safe. Falls back to a sensible default when the
// caller constructed a Subprocess directly without setting tailLimit.
func (s *Subprocess) pushTail(line string) {
	s.tailMu.Lock()
	defer s.tailMu.Unlock()
	limit := s.tailLimit
	if limit <= 0 {
		limit = 50
	}
	if len(s.tailBuf) >= limit {
		// Drop as many oldest entries as needed to fit within limit-1
		// (then append will bring us back to limit).
		keep := limit - 1
		if keep < 0 {
			keep = 0
		}
		s.tailBuf = append(s.tailBuf[:0], s.tailBuf[len(s.tailBuf)-keep:]...)
	}
	s.tailBuf = append(s.tailBuf, line)
}

// snapshotTail returns a copy of the current tail buffer as a single
// newline-separated string, suitable for embedding in an error message.
func (s *Subprocess) snapshotTail() string {
	s.tailMu.Lock()
	defer s.tailMu.Unlock()
	if len(s.tailBuf) == 0 {
		return ""
	}
	return strings.Join(s.tailBuf, "\n")
}

// Port returns the port the subprocess is listening on.
func (s *Subprocess) Port() int {
	return s.port
}

// BaseURL returns the HTTP base URL of the subprocess.
func (s *Subprocess) BaseURL() string {
	return s.baseURL
}

// Healthy returns whether the subprocess last passed a health check.
func (s *Subprocess) Healthy() bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.healthy
}

// Start launches the subprocess and waits for it to become healthy.
// The provided ctx controls only the health-check wait; the subprocess
// itself runs with a background lifetime.
func (s *Subprocess) Start(ctx context.Context) error {
	s.mu.Lock()
	s.stopped = false
	s.healthy = false
	s.doneCh = make(chan struct{})
	s.mu.Unlock()

	// Inject --port into args.
	args := append([]string(nil), s.args...)
	portFound := false
	for i, a := range args {
		if a == "--port" && i+1 < len(args) {
			args[i+1] = strconv.Itoa(s.port)
			portFound = true

			break
		}
	}
	if !portFound {
		args = append(args, "--port", strconv.Itoa(s.port))
	}

	s.cmd = exec.Command(s.binPath, args...)
	s.cmd.Dir = filepath.Dir(s.binPath) // backends are discovered relative to cwd
	s.cmd.Env = s.env
	setSysProcAttr(s.cmd)

	if s.quiet {
		s.cmd.Stdout = io.Discard
		s.cmd.Stderr = io.Discard
	} else {
		s.pipeOutput()
	}

	slog.Info("subprocess starting", "label", s.label, "bin", s.binPath, "port", s.port)

	if err := s.cmd.Start(); err != nil {
		return fmt.Errorf("failed to start %s: %w", s.label, err)
	}

	// On Windows, assign to a Job Object so the subprocess is killed if we crash.
	afterStart(s.cmd)

	// Background goroutine to detect process exit.
	go func() {
		_ = s.cmd.Wait()
		close(s.doneCh)
	}()

	if err := s.waitForHealth(ctx); err != nil {
		_ = s.GracefulStop()

		return fmt.Errorf("%s failed to become healthy: %w", s.label, err)
	}

	s.mu.Lock()
	s.healthy = true
	s.mu.Unlock()

	slog.Info("subprocess ready", "label", s.label, "port", s.port)

	return nil
}

// Done returns a channel that is closed when the subprocess exits.
func (s *Subprocess) Done() <-chan struct{} {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.doneCh
}

// ExitCode returns the process exit code, or -1 if not yet exited.
func (s *Subprocess) ExitCode() int {
	if s.cmd == nil || s.cmd.ProcessState == nil {
		return -1
	}

	return s.cmd.ProcessState.ExitCode()
}

// GracefulStop sends SIGTERM, waits up to 5 seconds, then SIGKILL.
func (s *Subprocess) GracefulStop() error {
	s.mu.Lock()
	s.stopped = true
	s.healthy = false
	s.mu.Unlock()

	if s.cmd == nil || s.cmd.Process == nil {
		return nil
	}

	pid := s.cmd.Process.Pid
	slog.Info("subprocess sending SIGTERM", "label", s.label, "pid", pid)

	// Send SIGTERM (SIGINT on Windows for graceful shutdown).
	var sigErr error
	if runtime.GOOS == "windows" {
		sigErr = s.cmd.Process.Signal(os.Interrupt)
	} else {
		sigErr = s.cmd.Process.Signal(syscall.SIGTERM)
	}

	if sigErr != nil {
		// Process may already be dead.
		slog.Warn("subprocess signal failed, process may have exited", "label", s.label, "error", sigErr)

		return nil
	}

	// Wait up to 5 seconds for clean exit.
	select {
	case <-s.doneCh:
		slog.Info("subprocess exited cleanly", "label", s.label)
		cleanupProcAttr(s.cmd)

		return nil
	case <-time.After(5 * time.Second):
		slog.Warn("subprocess did not exit after SIGTERM, sending SIGKILL", "label", s.label, "pid", pid)
		if err := s.cmd.Process.Kill(); err != nil {
			return fmt.Errorf("failed to kill %s: %w", s.label, err)
		}
		<-s.doneCh
		cleanupProcAttr(s.cmd)

		return nil
	}
}

// WasStopped returns true if GracefulStop was called (i.e., this was an intentional shutdown).
func (s *Subprocess) WasStopped() bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.stopped
}

// waitForHealth polls /health until it returns 200, with progress logging.
func (s *Subprocess) waitForHealth(ctx context.Context) error {
	deadline := time.Now().Add(s.healthTimeout)
	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()

	progressTicker := time.NewTicker(5 * time.Second)
	defer progressTicker.Stop()

	start := time.Now()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-s.doneCh:
			tail := s.snapshotTail()
			if tail != "" {
				return fmt.Errorf("%s process exited during startup (exit code %d); last output:\n%s", s.label, s.ExitCode(), tail)
			}
			return fmt.Errorf("%s process exited during startup (exit code %d); no output captured", s.label, s.ExitCode())
		case <-progressTicker.C:
			slog.Info("subprocess still loading model", "label", s.label, "elapsed_s", int(time.Since(start).Seconds()))
		case <-ticker.C:
			if time.Now().After(deadline) {
				return fmt.Errorf("timeout waiting for %s to become ready after %s", s.label, s.healthTimeout)
			}
			if s.healthCheck(ctx) == nil {
				return nil
			}
		}
	}
}

func (s *Subprocess) healthCheck(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, s.baseURL+"/health", nil)
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

// pipeOutput connects subprocess stdout+stderr to Go's logger with a prefix.
func (s *Subprocess) pipeOutput() {
	prefix := fmt.Sprintf("[%s] ", s.label)

	stdoutPipe, err := s.cmd.StdoutPipe()
	if err == nil {
		go s.scanLines(stdoutPipe, prefix)
	}

	stderrPipe, err := s.cmd.StderrPipe()
	if err == nil {
		go s.scanLines(stderrPipe, prefix)
	}
}

func (s *Subprocess) scanLines(r io.Reader, prefix string) {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), 256*1024)
	for scanner.Scan() {
		line := scanner.Text()
		s.pushTail(line)
		// Info level so llama-server OOM / load errors actually reach the
		// GPU container logs where an operator can grep them later.
		slog.Info("subprocess output", "label", s.label, "line", line)
	}
}

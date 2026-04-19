package runner

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/ThatCatDev/tanrenai-gpu/pkg/api"
)

// splitSuffix matches split GGUF filenames like "model-00001-of-00003.gguf"
var splitModelSuffix = regexp.MustCompile(`-\d{5}-of-\d{5}\.gguf$`)

const maxRestartAttempts = 3

// ProcessRunner manages a llama-server subprocess for model inference.
type ProcessRunner struct {
	sub       *Subprocess
	modelPath string
	modelName string
	opts      Options
	client    *Client
	baseURL   string

	// Crash detection and auto-restart.
	mu          sync.Mutex
	restarts    int
	crashNotify chan error // receives an error each time the process crashes
	stopMonitor chan struct{}
}

// NewProcessRunner creates a new ProcessRunner.
func NewProcessRunner() *ProcessRunner {
	return &ProcessRunner{
		crashNotify: make(chan error, 4),
		stopMonitor: make(chan struct{}),
	}
}

// CrashNotify returns a channel that receives an error each time the
// subprocess crashes unexpectedly. The channel is buffered; if the consumer
// falls behind, notifications are dropped.
func (r *ProcessRunner) CrashNotify() <-chan error {
	return r.crashNotify
}

func (r *ProcessRunner) Load(ctx context.Context, modelPath string, opts Options) error {
	r.modelPath = modelPath
	name := filepath.Base(modelPath)
	// Strip split suffix (e.g. "-00001-of-00003.gguf" → "") and .gguf extension
	if splitModelSuffix.MatchString(name) {
		name = splitModelSuffix.ReplaceAllString(name, "")
	} else {
		name = strings.TrimSuffix(name, ".gguf")
	}
	r.modelName = name
	r.opts = opts

	if err := r.startSubprocess(ctx); err != nil {
		return err
	}

	// Start crash monitoring goroutine.
	go r.monitorCrashes()

	return nil
}

// startSubprocess creates and starts the llama-server subprocess.
func (r *ProcessRunner) startSubprocess(ctx context.Context) error {
	args := r.buildArgs()

	healthTimeout := r.opts.HealthTimeout
	if healthTimeout == 0 {
		healthTimeout = 120 * time.Second
	}

	sub, err := NewSubprocess(SubprocessConfig{
		BinDir:        r.opts.BinDir,
		Args:          args,
		Port:          r.opts.Port,
		Label:         "llama-server",
		Quiet:         r.opts.Quiet,
		HealthTimeout: healthTimeout,
	})
	if err != nil {
		return err
	}

	if err := sub.Start(ctx); err != nil {
		return err
	}

	r.sub = sub
	r.baseURL = sub.BaseURL()
	r.client = NewClient(r.baseURL)
	// Update opts.Port so restarts reuse the same allocated port.
	r.opts.Port = sub.Port()

	slog.Info("llama-server ready", "port", sub.Port(), "model", r.modelName)

	return nil
}

func (r *ProcessRunner) buildArgs() []string {
	args := []string{
		"--model", r.modelPath,
		"--ctx-size", strconv.Itoa(r.opts.CtxSize),
		"--host", "127.0.0.1",
	}

	if r.opts.GPULayers >= 0 {
		args = append(args, "--n-gpu-layers", strconv.Itoa(r.opts.GPULayers))
	} else if r.opts.FitVRAM {
		args = append(args, "--n-gpu-layers", "auto")
	} else {
		args = append(args, "--n-gpu-layers", "999")
	}

	// MoE-specific flags
	if r.opts.CPUMoE {
		args = append(args, "--cpu-moe")
	} else if r.opts.CPUMoELayers > 0 {
		args = append(args, "--n-cpu-moe", strconv.Itoa(r.opts.CPUMoELayers))
	}

	if r.opts.NoKVOffload {
		args = append(args, "--no-kv-offload")
	}

	if r.opts.FitVRAM {
		args = append(args, "--fit", "on")
	}

	if r.opts.TensorSplit != "" {
		args = append(args, "--tensor-split", r.opts.TensorSplit)
	}

	if r.opts.SplitMode != "" {
		args = append(args, "--split-mode", r.opts.SplitMode)
	}

	if r.opts.OverrideTensor != "" {
		args = append(args, "--override-tensor", r.opts.OverrideTensor)
	}

	if r.opts.Threads > 0 {
		args = append(args, "--threads", strconv.Itoa(r.opts.Threads))
	}

	if r.opts.FlashAttention {
		args = append(args, "--flash-attn", "on")
	}

	args = append(args, "--jinja")

	if r.opts.ChatTemplateFile != "" {
		args = append(args, "--chat-template-file", r.opts.ChatTemplateFile)
	}

	if r.opts.ReasoningFormat != "" {
		args = append(args, "--reasoning-format", r.opts.ReasoningFormat)
	}

	return args
}

// monitorCrashes watches for unexpected process exits and restarts.
func (r *ProcessRunner) monitorCrashes() {
	for {
		select {
		case <-r.stopMonitor:
			return
		case <-r.sub.Done():
			if r.sub.WasStopped() {
				return
			}

			exitCode := r.sub.ExitCode()
			slog.Error("llama-server process crashed", "exit_code", exitCode)

			r.mu.Lock()
			r.restarts++
			attempt := r.restarts
			r.mu.Unlock()

			crashErr := fmt.Errorf("llama-server crashed (exit code %d, restart %d/%d)", exitCode, attempt, maxRestartAttempts)

			// Notify consumers (non-blocking).
			select {
			case r.crashNotify <- crashErr:
			default:
			}

			if attempt > maxRestartAttempts {
				slog.Error("llama-server max restart attempts reached, giving up", "max_attempts", maxRestartAttempts)

				return
			}

			slog.Info("llama-server restarting", "attempt", attempt, "max_attempts", maxRestartAttempts)
			// Use a timeout context for restart health check.
			restartCtx, cancel := context.WithTimeout(context.Background(), r.sub.healthTimeout)
			if err := r.startSubprocess(restartCtx); err != nil {
				slog.Error("llama-server restart failed", "error", err)
				cancel()

				return
			}
			cancel()
			slog.Info("llama-server restart successful")
		}
	}
}

func (r *ProcessRunner) Health(ctx context.Context) error {
	if r.sub == nil {
		return fmt.Errorf("llama-server not started")
	}

	return r.sub.healthCheck(ctx)
}

func (r *ProcessRunner) ChatCompletion(ctx context.Context, req *api.ChatCompletionRequest) (*api.ChatCompletionResponse, error) {
	req.Stream = false

	return r.client.ChatCompletion(ctx, req)
}

func (r *ProcessRunner) ChatCompletionStream(ctx context.Context, req *api.ChatCompletionRequest, w io.Writer) error {
	req.Stream = true

	return r.client.ChatCompletionStream(ctx, req, w)
}

func (r *ProcessRunner) Tokenize(ctx context.Context, text string) (int, error) {
	return r.client.Tokenize(ctx, text)
}

func (r *ProcessRunner) ModelName() string {
	return r.modelName
}

func (r *ProcessRunner) Close() error {
	// Stop the crash monitor so it doesn't try to restart.
	select {
	case <-r.stopMonitor:
		// Already closed.
	default:
		close(r.stopMonitor)
	}

	if r.sub != nil {
		return r.sub.GracefulStop()
	}

	return nil
}

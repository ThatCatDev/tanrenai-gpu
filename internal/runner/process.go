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

	// lastCrash records the most recent unexpected subprocess exit, so a request
	// that hits the box while llama-server is down/restarting carries the reason
	// it died (its stderr tail) instead of a bare "connection refused". Guarded
	// by mu.
	lastCrashCode int
	lastCrashTail string
	lastCrashAt   time.Time
}

// crashRecencyWindow bounds how long after a crash we still attribute a request
// failure to it. Beyond this the box has had time to restart/reap, so an old
// crash is unlikely to be the cause and we don't want to misattribute.
const crashRecencyWindow = 2 * time.Minute

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

	// Always pass --fit explicitly. llama.cpp's recent builds run the
	// device-memory introspection step even without --fit on, and the
	// introspection itself has segfaulted on at least one Ampere host
	// (RTX A6000) in master b1-1e5ad35 — crashing inside
	// common_params_fit_impl before the model finished loading. Sizing
	// is already enforced upstream (the platform picks an offer with
	// enough VRAM for the chosen model); we don't need llama.cpp to
	// re-derive it. Forcing --fit off skips the buggy code path.
	if r.opts.FitVRAM {
		args = append(args, "--fit", "on")
	} else {
		args = append(args, "--fit", "off")
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

	// Quantize the KV cache (both K and V) to shrink its VRAM footprint at large
	// contexts — the dominant consumer for multi-slot serving. Requires flash
	// attention (above). "f16"/"" leaves it at full precision.
	if r.opts.KVCacheType != "" && r.opts.KVCacheType != "f16" {
		args = append(args, "--cache-type-k", r.opts.KVCacheType, "--cache-type-v", r.opts.KVCacheType)
	}

	if r.opts.ContextShift {
		args = append(args, "--context-shift")
	}

	if r.opts.Parallel > 1 {
		args = append(args, "--parallel", strconv.Itoa(r.opts.Parallel))
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
			tail := r.sub.snapshotTail()
			// stderr_tail carries llama-server's last output — the actual kill
			// reason (e.g. "CUDA error: out of memory", a SIGILL/segfault). exit
			// code alone ("-1") never says why.
			slog.Error("llama-server process crashed", "exit_code", exitCode, "stderr_tail", tail)

			r.mu.Lock()
			r.restarts++
			attempt := r.restarts
			r.lastCrashCode = exitCode
			r.lastCrashTail = tail
			r.lastCrashAt = time.Now()
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

	resp, err := r.client.ChatCompletion(ctx, req)

	return resp, r.annotateCrash(err)
}

func (r *ProcessRunner) ChatCompletionStream(ctx context.Context, req *api.ChatCompletionRequest, w io.Writer) error {
	req.Stream = true

	return r.annotateCrash(r.client.ChatCompletionStream(ctx, req, w))
}

// annotateCrash appends the most recent llama-server crash reason to a request
// error when the crash was recent enough to plausibly be the cause. This turns
// a bare "dial tcp ...: connection refused" into one that also says WHY the
// subprocess was down (its stderr tail), so the reason flows downstream into
// the platform's error log. Returns err unchanged when there's no recent crash.
func (r *ProcessRunner) annotateCrash(err error) error {
	if err == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.lastCrashAt.IsZero() || time.Since(r.lastCrashAt) > crashRecencyWindow {
		return err
	}
	reason := extractCrashReason(r.lastCrashTail)
	if reason == "" {
		reason = "(no stderr captured)"
	}
	if len(reason) > 400 {
		reason = reason[:400] + "…"
	}
	return fmt.Errorf("%w; last llama-server crash exit=%d: %s", err, r.lastCrashCode, reason)
}

// backtraceFrame matches a C/C++ stack-trace line like
// "/usr/local/lib/libfoo.so(_Zsym+0x12)[0x7f1a2b3c]" — pure stack noise that
// crowds out the actual error message in a crash dump.
var backtraceFrame = regexp.MustCompile(`\[0x[0-9a-fA-F]+\]\s*$`)

// crashSignal flags lines that actually name a failure cause.
var crashSignal = regexp.MustCompile(`(?i)(cuda error|out of memory|oom|ggml_assert|ggml[ _]|assert|terminate called|what\(\):|segmentation|sigsegv|sigabrt|abort|exception|error:|failed)`)

// extractCrashReason pulls the meaningful reason out of a crash dump. llama.cpp
// prints the cause (e.g. "CUDA error: out of memory", "GGML_ASSERT(...) failed")
// BEFORE a long backtrace, so naively keeping the tail surfaces only libc stack
// frames. We instead return the first line that names a cause and isn't a raw
// backtrace frame; failing that, the first non-frame line (the dump header).
func extractCrashReason(tail string) string {
	if tail == "" {
		return ""
	}
	lines := strings.Split(tail, "\n")
	for _, ln := range lines {
		t := strings.TrimSpace(ln)
		if t == "" || backtraceFrame.MatchString(t) {
			continue
		}
		if crashSignal.MatchString(t) {
			return t
		}
	}
	for _, ln := range lines {
		t := strings.TrimSpace(ln)
		if t != "" && !backtraceFrame.MatchString(t) {
			return t
		}
	}
	return strings.TrimSpace(tail)
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

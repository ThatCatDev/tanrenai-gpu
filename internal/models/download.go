package models

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	neturl "net/url"
	"os"
	"path"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// DownloadProgress is called periodically during a download.
type DownloadProgress func(downloaded, total int64)

// Tunables. Exposed as package variables so tests can shorten them.
var (
	// DownloadMaxAttempts is how many times we'll retry a failing download
	// before giving up. Each retry picks up from the existing .partial via
	// Range, so a 90%-done shard resumes rather than starts over.
	DownloadMaxAttempts = 5

	// DownloadBackoffBase is the base for exponential backoff between
	// retries: 1×, 2×, 4×, 8×, 16× = ~31s total across 5 attempts.
	DownloadBackoffBase = 1 * time.Second

	// DownloadStallThreshold is how long we'll tolerate zero bytes flowing
	// before concluding the TCP connection is dead. Deliberately generous —
	// HF and R2 both occasionally pause for tens of seconds mid-stream.
	DownloadStallThreshold = 60 * time.Second
)

// permanentError wraps errors we don't want to retry — non-gguf URLs,
// 4xx responses, parse failures. Bubble up on the first occurrence.
type permanentError struct{ err error }

func (e *permanentError) Error() string { return e.err.Error() }
func (e *permanentError) Unwrap() error { return e.err }

func permanent(err error) error { return &permanentError{err: err} }

func isRetryable(err error) bool {
	var p *permanentError
	return !errors.As(err, &p)
}

// Download downloads a GGUF file from HuggingFace (or any direct URL
// including presigned S3/R2 URLs). Retries up to DownloadMaxAttempts on
// transient errors — connection drops, TCP stalls, truncated streams —
// resuming from the existing .partial file via Range on each retry.
//
//	https://huggingface.co/<repo>/resolve/main/<filename>.gguf
func Download(url, destDir string, progress DownloadProgress) (string, error) {
	parsed, err := neturl.Parse(url)
	if err != nil {
		return "", fmt.Errorf("parse URL: %w", err)
	}
	filename := path.Base(parsed.Path)
	if !strings.HasSuffix(strings.ToLower(filename), ".gguf") {
		return "", fmt.Errorf("URL does not point to a .gguf file: %s", filename)
	}

	destPath := filepath.Join(destDir, filename)
	partialPath := destPath + ".partial"

	var lastErr error
	for attempt := 1; attempt <= DownloadMaxAttempts; attempt++ {
		err := downloadAttempt(url, partialPath, progress)
		if err == nil {
			lastErr = nil
			break
		}
		lastErr = err
		if !isRetryable(err) {
			return "", err
		}
		if attempt >= DownloadMaxAttempts {
			break
		}
		backoff := DownloadBackoffBase << uint(attempt-1)
		time.Sleep(backoff)
	}
	if lastErr != nil {
		return "", fmt.Errorf("download failed after %d attempts: %w", DownloadMaxAttempts, lastErr)
	}

	if err := os.Rename(partialPath, destPath); err != nil {
		return "", fmt.Errorf("rename file: %w", err)
	}

	// Save provenance metadata if this is a HuggingFace URL.
	if repo, branch, ok := ParseHFURL(url); ok {
		meta := &ModelMetadata{
			HFRepo:   repo,
			HFBranch: branch,
			Source:   "huggingface",
		}
		if err := SaveMetadata(destPath, meta); err != nil {
			fmt.Fprintf(os.Stderr, "warning: could not save model metadata: %v\n", err)
		}
	}

	return destPath, nil
}

// downloadAttempt runs a single GET-with-Range attempt, writing into
// partialPath. Returns nil when the file on disk matches the expected size
// for the first time, a *permanentError for non-retryable conditions, and
// a plain error for everything else (retryable).
//
// A stall watchdog runs alongside the body read: if no bytes flow for
// DownloadStallThreshold, the request context is cancelled, unblocking
// resp.Body.Read with a context-cancelled error. The outer retry loop then
// picks up where we left off via Range.
func downloadAttempt(url, partialPath string, progress DownloadProgress) error {
	var startByte int64
	if info, err := os.Stat(partialPath); err == nil {
		startByte = info.Size()
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return permanent(fmt.Errorf("create request: %w", err))
	}
	if startByte > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", startByte))
	}
	if token := os.Getenv("HF_TOKEN"); token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("download request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	switch {
	case resp.StatusCode == http.StatusRequestedRangeNotSatisfiable:
		// Our resume position is past the server's end-of-file. Probably a
		// stale or corrupt .partial. Wipe it so the next attempt starts fresh.
		_ = os.Remove(partialPath)
		return fmt.Errorf("range not satisfiable; partial reset")
	case resp.StatusCode >= 400 && resp.StatusCode < 500:
		return permanent(fmt.Errorf("download failed with status %d", resp.StatusCode))
	case resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusPartialContent:
		return fmt.Errorf("download failed with status %d", resp.StatusCode)
	}

	expectedTotal := resp.ContentLength + startByte

	flags := os.O_CREATE | os.O_WRONLY
	if startByte > 0 {
		flags |= os.O_APPEND
	} else {
		flags |= os.O_TRUNC
	}
	f, err := os.OpenFile(partialPath, flags, 0644)
	if err != nil {
		return permanent(fmt.Errorf("open file: %w", err))
	}
	defer func() { _ = f.Close() }()

	// Stall watchdog. Updates `lastRead` on every successful Read; if more
	// than DownloadStallThreshold passes between reads, we cancel ctx and
	// the next resp.Body.Read returns a context error.
	var (
		lastMu   sync.Mutex
		lastRead = time.Now()
		stalled  bool
	)
	watchDone := make(chan struct{})
	go func() {
		defer close(watchDone)
		ticker := time.NewTicker(DownloadStallThreshold / 4)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				lastMu.Lock()
				since := time.Since(lastRead)
				lastMu.Unlock()
				if since > DownloadStallThreshold {
					lastMu.Lock()
					stalled = true
					lastMu.Unlock()
					cancel()
					return
				}
			}
		}
	}()

	downloaded := startByte
	buf := make([]byte, 32*1024)
	for {
		n, readErr := resp.Body.Read(buf)
		if n > 0 {
			if _, writeErr := f.Write(buf[:n]); writeErr != nil {
				cancel()
				<-watchDone
				return permanent(fmt.Errorf("write file: %w", writeErr))
			}
			downloaded += int64(n)
			lastMu.Lock()
			lastRead = time.Now()
			lastMu.Unlock()
			if progress != nil {
				progress(downloaded, expectedTotal)
			}
		}
		if readErr != nil {
			cancel()
			<-watchDone
			if readErr == io.EOF {
				break
			}
			lastMu.Lock()
			wasStalled := stalled
			lastMu.Unlock()
			if wasStalled {
				return fmt.Errorf("stalled: no bytes received for %s (got %d / %d)", DownloadStallThreshold, downloaded, expectedTotal)
			}
			return fmt.Errorf("read body: %w", readErr)
		}
	}
	cancel()
	<-watchDone

	if err := f.Close(); err != nil {
		return permanent(fmt.Errorf("close file: %w", err))
	}

	// Validate byte count. ContentLength is -1 when the server doesn't
	// advertise a size — skip validation in that case; nothing we can
	// compare against.
	if resp.ContentLength >= 0 && downloaded != expectedTotal {
		return fmt.Errorf("truncated: got %d / %d bytes", downloaded, expectedTotal)
	}

	return nil
}

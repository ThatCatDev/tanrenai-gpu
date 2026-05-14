package models

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"sync"
	"sync/atomic"
	"time"
)

// Tunables for the parallel range downloader. Exposed for tests.
var (
	// DisableParallelDownload, when true, skips the HEAD probe and parallel
	// path entirely. The single-stream sequential downloader handles every
	// request. Used by the legacy download tests, which were written against
	// the pre-parallel call pattern and assert on exact request counts.
	DisableParallelDownload = false

	// ParallelDownloadMinSize is the file size below which we don't bother
	// splitting into ranges — the per-connection overhead and the HEAD
	// roundtrip aren't worth it for small files.
	ParallelDownloadMinSize int64 = 50 * 1024 * 1024

	// ParallelDownloadConcurrency is how many concurrent range GETs we
	// issue against a large file. 8 saturates most R2/CF edge paths without
	// causing rate-limiting; higher values give diminishing returns.
	ParallelDownloadConcurrency = 8

	// ParallelDownloadChunkBuffer is the read buffer per in-flight range.
	// Memory ceiling per download is roughly Concurrency * ChunkBuffer.
	ParallelDownloadChunkBuffer = 64 * 1024

	// ParallelDownloadProbeTimeout caps the initial HEAD (+ optional GET
	// range probe) used to learn the file size and verify range support.
	// A misbehaving origin shouldn't block the whole download — bail and
	// fall through to sequential.
	ParallelDownloadProbeTimeout = 10 * time.Second
)

// errNoRangeSupport signals that the server can't (or won't) honor Range
// requests, so the caller should fall through to a single-stream download.
// Distinct from a normal retryable error: there's no point retrying with
// parallel ranges when the server fundamentally doesn't support them.
type errNoRangeSupport struct{ reason string }

func (e *errNoRangeSupport) Error() string {
	return "server does not support range requests: " + e.reason
}

// tryParallel attempts a parallel range download. Returns true on success
// (partialPath is fully populated and ready to rename), false when the
// caller should fall through to the sequential downloader — either because
// the file is too small to bother splitting, the server doesn't support
// ranges, or the parallel attempt failed in a way we can't usefully retry
// in parallel. Permanent errors (4xx etc.) are returned to the caller as
// false so the sequential path can surface them with its own retry logic
// and we never silently mask a real error.
func tryParallel(partialPath, url string, progress DownloadProgress) bool {
	if DisableParallelDownload {
		return false
	}
	ctx, cancel := context.WithTimeout(context.Background(), ParallelDownloadProbeTimeout)
	size, supportsRanges, err := probeRanges(ctx, url)
	cancel()
	if err != nil {
		slog.Debug("parallel download: probe failed, falling back to sequential", "url", url, "error", err)
		return false
	}
	if !supportsRanges {
		slog.Debug("parallel download: server lacks Accept-Ranges, using sequential", "url", url)
		return false
	}
	if size < ParallelDownloadMinSize {
		slog.Debug("parallel download: file smaller than threshold, using sequential", "size", size, "threshold", ParallelDownloadMinSize)
		return false
	}

	slog.Info("parallel download starting", "size", size, "concurrency", ParallelDownloadConcurrency)
	start := time.Now()
	err = parallelDownload(context.Background(), url, partialPath, size, ParallelDownloadConcurrency, progress)
	if err == nil {
		slog.Info("parallel download finished", "size", size, "duration", time.Since(start).Round(time.Second))
		return true
	}

	// errNoRangeSupport mid-flight is rare (probe said yes, but a range
	// got a 200) — fall through cleanly.
	var rangeErr *errNoRangeSupport
	if errors.As(err, &rangeErr) {
		slog.Warn("parallel download: server rejected ranges mid-flight, falling back", "error", err)
		_ = os.Remove(partialPath)
		return false
	}

	// Anything else: log and fall through. The sequential path's own
	// retry will surface the real error if it's truly broken.
	slog.Warn("parallel download failed, falling back to sequential", "error", err)
	_ = os.Remove(partialPath)
	return false
}

// probeClient is a dedicated HTTP client for HEAD/range probes. Keep-alive
// is disabled so the underlying TCP connection closes the moment our probe
// returns (or its context expires) — important because some origins keep
// streaming after we get headers, and a pooled connection would leave them
// hanging until idle eviction.
var probeClient = &http.Client{
	Transport: &http.Transport{
		DisableKeepAlives: true,
	},
}

// probeRanges issues a HEAD against the URL to figure out (a) whether the
// server supports byte ranges and (b) the file's Content-Length. Both are
// required to safely split the download.
//
// Some object stores (notably HF's CDN) return Accept-Ranges only on GET
// not HEAD; we fall back to a 1-byte GET with Range probe in that case.
func probeRanges(ctx context.Context, url string) (size int64, supportsRanges bool, err error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodHead, url, nil)
	if err != nil {
		return 0, false, err
	}
	if token := os.Getenv("HF_TOKEN"); token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}

	resp, err := probeClient.Do(req)
	if err != nil {
		return 0, false, err
	}
	_ = resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return 0, false, fmt.Errorf("HEAD returned status %d", resp.StatusCode)
	}

	size = resp.ContentLength
	if resp.Header.Get("Accept-Ranges") == "bytes" {
		return size, true, nil
	}

	// HEAD didn't advertise ranges; many CDN-fronted origins still honor
	// them on GET. Probe with a single-byte range to be sure.
	getReq, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return size, false, nil
	}
	getReq.Header.Set("Range", "bytes=0-0")
	if token := os.Getenv("HF_TOKEN"); token != "" {
		getReq.Header.Set("Authorization", "Bearer "+token)
	}
	getResp, err := probeClient.Do(getReq)
	if err != nil {
		return size, false, nil
	}
	defer func() { _ = getResp.Body.Close() }()
	if getResp.StatusCode == http.StatusPartialContent {
		// Confirm with the range probe: Content-Range tells us the true size.
		if cr := getResp.Header.Get("Content-Range"); cr != "" {
			var total int64
			if _, scanErr := fmt.Sscanf(cr, "bytes 0-0/%d", &total); scanErr == nil && total > 0 {
				size = total
			}
		}
		return size, true, nil
	}
	return size, false, nil
}

// parallelDownload pulls a file in N concurrent ranges, writing each to its
// offset in partialPath. Returns errNoRangeSupport when the server doesn't
// honor ranges (caller should fall back to sequential); returns a normal
// error for everything else.
//
// Per-range retry resumes from where each chunk left off when a TCP stream
// breaks mid-range, identical in spirit to the sequential downloader's
// `.partial` resume. The global progress callback sees the sum across all
// concurrent ranges.
func parallelDownload(ctx context.Context, url, partialPath string, totalSize int64, concurrency int, progress DownloadProgress) error {
	if totalSize <= 0 {
		return fmt.Errorf("totalSize must be > 0, got %d", totalSize)
	}
	if concurrency < 1 {
		concurrency = 1
	}

	// Pre-size the file. WriteAt at offset N requires the file to either
	// already extend to N or grow to accommodate; truncating to totalSize
	// gives every chunk a fixed slot to fill in any order.
	f, err := os.OpenFile(partialPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
	if err != nil {
		return permanent(fmt.Errorf("open partial: %w", err))
	}
	defer func() { _ = f.Close() }()
	if err := f.Truncate(totalSize); err != nil {
		return permanent(fmt.Errorf("truncate partial to %d: %w", totalSize, err))
	}

	chunkSize := totalSize / int64(concurrency)
	if chunkSize < 1 {
		// File smaller than concurrency — degenerate, run as one chunk.
		chunkSize = totalSize
		concurrency = 1
	}

	var (
		downloaded atomic.Int64
		wg         sync.WaitGroup
		errMu      sync.Mutex
		firstErr   error
	)

	// Cancel all in-flight ranges as soon as any one fails permanently.
	rangeCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	for i := 0; i < concurrency; i++ {
		start := int64(i) * chunkSize
		end := start + chunkSize - 1
		if i == concurrency-1 {
			end = totalSize - 1
		}

		wg.Add(1)
		go func(start, end int64) {
			defer wg.Done()
			if err := downloadRangeWithRetry(rangeCtx, url, f, start, end, &downloaded, progress, totalSize); err != nil {
				errMu.Lock()
				if firstErr == nil {
					firstErr = err
					cancel()
				}
				errMu.Unlock()
			}
		}(start, end)
	}
	wg.Wait()
	return firstErr
}

// downloadRangeWithRetry pulls bytes [start, end] (inclusive) into f at the
// corresponding offsets, retrying up to DownloadMaxAttempts on transient
// failures. On a mid-stream failure the next attempt resumes from where the
// last one stopped (tracked via cumulative `received`), so a flaky range
// doesn't have to start over.
func downloadRangeWithRetry(ctx context.Context, url string, f *os.File, start, end int64, downloaded *atomic.Int64, progress DownloadProgress, total int64) error {
	received := int64(0)
	var lastErr error

	for attempt := 1; attempt <= DownloadMaxAttempts; attempt++ {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		err := doRangeAttempt(ctx, url, f, start+received, end, &received, downloaded, progress, total)
		if err == nil {
			return nil
		}
		if !isRetryable(err) {
			return err
		}
		lastErr = err
		if attempt < DownloadMaxAttempts {
			backoff := DownloadBackoffBase << uint(attempt-1)
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(backoff):
			}
		}
	}
	return fmt.Errorf("range %d-%d failed after %d attempts: %w", start, end, DownloadMaxAttempts, lastErr)
}

// doRangeAttempt is one TCP-level pass at downloading [start, end] into f.
// Increments `received` as it writes so the outer retry loop can resume at
// the new start position. A stall watchdog cancels the request if no bytes
// flow for DownloadStallThreshold; the caller's outer retry then picks up
// where we left off.
func doRangeAttempt(parent context.Context, url string, f *os.File, start, end int64, received *int64, downloaded *atomic.Int64, progress DownloadProgress, total int64) error {
	if start > end {
		// Already fully received — caller's bookkeeping says we're done.
		return nil
	}

	ctx, cancel := context.WithCancel(parent)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return permanent(fmt.Errorf("build range request: %w", err))
	}
	req.Header.Set("Range", fmt.Sprintf("bytes=%d-%d", start, end))
	if token := os.Getenv("HF_TOKEN"); token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("range request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	switch {
	case resp.StatusCode == http.StatusPartialContent:
		// expected — proceed
	case resp.StatusCode == http.StatusOK:
		// Server ignored Range and sent the full body. We could read+
		// discard the prefix and write the suffix, but that defeats
		// the whole point of parallel ranges. Bail to fallback.
		return &errNoRangeSupport{reason: "server returned 200 for a Range request"}
	case resp.StatusCode == http.StatusRequestedRangeNotSatisfiable:
		// Stale boundary — shouldn't happen mid-attempt but be defensive.
		return permanent(fmt.Errorf("range not satisfiable for bytes=%d-%d", start, end))
	case resp.StatusCode >= 400 && resp.StatusCode < 500:
		return permanent(fmt.Errorf("range request status %d", resp.StatusCode))
	default:
		return fmt.Errorf("range request status %d", resp.StatusCode)
	}

	// Stall watchdog. Same pattern as the sequential downloader: if no
	// bytes flow for DownloadStallThreshold, cancel ctx so resp.Body.Read
	// unblocks with a context-cancelled error and the outer retry resumes.
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

	buf := make([]byte, ParallelDownloadChunkBuffer)
	offset := start
	for {
		n, readErr := resp.Body.Read(buf)
		if n > 0 {
			if _, writeErr := f.WriteAt(buf[:n], offset); writeErr != nil {
				cancel()
				<-watchDone
				return permanent(fmt.Errorf("write at offset %d: %w", offset, writeErr))
			}
			offset += int64(n)
			atomic.AddInt64(received, int64(n))
			downloaded.Add(int64(n))
			lastMu.Lock()
			lastRead = time.Now()
			lastMu.Unlock()
			if progress != nil {
				progress(downloaded.Load(), total)
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
				return fmt.Errorf("stalled: no bytes for %s on range bytes=%d-%d", DownloadStallThreshold, start, end)
			}
			return fmt.Errorf("read body bytes=%d-%d: %w", start, end, readErr)
		}
	}
	cancel()
	<-watchDone

	expected := end - start + 1
	if offset-start != expected {
		return fmt.Errorf("range bytes=%d-%d: got %d / %d bytes", start, end, offset-start, expected)
	}
	return nil
}

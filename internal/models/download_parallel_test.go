package models

import (
	"bytes"
	"context"
	"crypto/sha256"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// withParallelEnabled re-enables parallel downloads (disabled in TestMain
// via the shared init() to keep legacy tests stable) and lowers the size
// threshold + per-chunk target so tests can drive the parallel path with
// small fixtures.
//
// chooseConcurrency() will pick `ceil(fileSize / chunkSize)` ranges,
// capped at maxConcurrency — so calling withParallelEnabled(t, 1024, 4, 16)
// with a 64 KB fixture yields 4 ranges of 16 KB each.
//
// Restores all package vars on cleanup so subsequent tests aren't affected.
func withParallelEnabled(t *testing.T, minSize int64, chunkSize int64, maxConcurrency int) {
	t.Helper()
	prevDisabled := DisableParallelDownload
	prevMin := ParallelDownloadMinSize
	prevChunk := ParallelDownloadMinChunkSize
	prevMax := ParallelDownloadMaxConcurrency
	prevBuf := ParallelDownloadChunkBuffer
	DisableParallelDownload = false
	ParallelDownloadMinSize = minSize
	ParallelDownloadMinChunkSize = chunkSize
	ParallelDownloadMaxConcurrency = maxConcurrency
	ParallelDownloadChunkBuffer = 4096 // small so tests exercise multiple read loops
	t.Cleanup(func() {
		DisableParallelDownload = prevDisabled
		ParallelDownloadMinSize = prevMin
		ParallelDownloadMinChunkSize = prevChunk
		ParallelDownloadMaxConcurrency = prevMax
		ParallelDownloadChunkBuffer = prevBuf
	})
}

// rangeAwareHandler serves `content` with Accept-Ranges: bytes and honors
// Range requests with proper 206 Partial Content responses + Content-Range
// header. Mirrors what R2 / HF CDN do for presigned URLs.
func rangeAwareHandler(content []byte, requestCount *atomic.Int32) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if requestCount != nil {
			requestCount.Add(1)
		}
		w.Header().Set("Accept-Ranges", "bytes")
		w.Header().Set("Content-Length", strconv.Itoa(len(content)))

		rangeHeader := r.Header.Get("Range")
		if rangeHeader == "" || r.Method == http.MethodHead {
			if r.Method != http.MethodHead {
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write(content)
			} else {
				w.WriteHeader(http.StatusOK)
			}
			return
		}

		var start, end int64
		if _, err := fmt.Sscanf(rangeHeader, "bytes=%d-%d", &start, &end); err != nil {
			http.Error(w, "bad range", http.StatusBadRequest)
			return
		}
		if end >= int64(len(content)) {
			end = int64(len(content)) - 1
		}
		w.Header().Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", start, end, len(content)))
		w.Header().Set("Content-Length", strconv.FormatInt(end-start+1, 10))
		w.WriteHeader(http.StatusPartialContent)
		_, _ = w.Write(content[start : end+1])
	})
}

func makeBlob(n int) []byte {
	b := make([]byte, n)
	for i := range b {
		// pseudo-random so corruption shows up in checksum comparisons
		b[i] = byte((i*2654435761 + 17) & 0xff)
	}
	return b
}

func TestParallelDownload_HappyPath(t *testing.T) {
	// 1 KB threshold, 16 KB chunks, max 4 ranges → 64 KB fixture yields 4 ranges
	withParallelEnabled(t, 1024, 16*1024, 4)

	content := makeBlob(64 * 1024)
	var calls atomic.Int32
	srv := httptest.NewServer(rangeAwareHandler(content, &calls))
	defer srv.Close()

	destDir := t.TempDir()
	path, err := Download(srv.URL+"/big.gguf", destDir, "", nil)
	if err != nil {
		t.Fatalf("Download: %v", err)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	if !bytes.Equal(got, content) {
		t.Errorf("downloaded content mismatch: sha %x vs %x", sha256.Sum256(got), sha256.Sum256(content))
	}
	// 1 HEAD probe + N range GETs (one per chunk).
	if n := calls.Load(); n < 2 {
		t.Errorf("expected ≥2 HTTP calls (probe + ranges), got %d", n)
	}
}

func TestParallelDownload_FallsBackWhenNoRangeSupport(t *testing.T) {
	withParallelEnabled(t, 1024, 2*1024, 4)

	content := makeBlob(8 * 1024)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Deliberately no Accept-Ranges header; refuse Range requests.
		if r.Header.Get("Range") != "" {
			// Mimic an origin that ignores Range and sends the full body.
			w.Header().Set("Content-Length", strconv.Itoa(len(content)))
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write(content)
			return
		}
		w.Header().Set("Content-Length", strconv.Itoa(len(content)))
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(content)
	}))
	defer srv.Close()

	destDir := t.TempDir()
	path, err := Download(srv.URL+"/x.gguf", destDir, "", nil)
	if err != nil {
		t.Fatalf("Download: %v", err)
	}
	got, _ := os.ReadFile(path)
	if !bytes.Equal(got, content) {
		t.Errorf("fallback content mismatch")
	}
}

func TestParallelDownload_SkipsSmallFiles(t *testing.T) {
	// Threshold above the fixture size — parallel should be skipped and
	// only the sequential path's single GET should fire (no HEAD probe in
	// the result either, since we shortcut on size after probing).
	withParallelEnabled(t, 1<<30, 1<<20, 4) // 1 GB threshold

	content := makeBlob(8 * 1024)
	var calls atomic.Int32
	srv := httptest.NewServer(rangeAwareHandler(content, &calls))
	defer srv.Close()

	destDir := t.TempDir()
	path, err := Download(srv.URL+"/small.gguf", destDir, "", nil)
	if err != nil {
		t.Fatalf("Download: %v", err)
	}
	got, _ := os.ReadFile(path)
	if !bytes.Equal(got, content) {
		t.Errorf("small-file fallback content mismatch")
	}
}

func TestParallelDownload_ProgressMonotonic(t *testing.T) {
	withParallelEnabled(t, 1024, 8*1024, 4)

	content := makeBlob(32 * 1024)
	srv := httptest.NewServer(rangeAwareHandler(content, nil))
	defer srv.Close()

	var (
		mu       sync.Mutex
		samples  []int64
		lastSeen int64
	)
	progress := func(downloaded, total int64) {
		mu.Lock()
		defer mu.Unlock()
		// Concurrent ranges write to the same atomic counter; values
		// should be monotonically non-decreasing.
		if downloaded < lastSeen {
			t.Errorf("non-monotonic progress: %d < %d", downloaded, lastSeen)
		}
		lastSeen = downloaded
		samples = append(samples, downloaded)
	}

	destDir := t.TempDir()
	_, err := Download(srv.URL+"/m.gguf", destDir, "", progress)
	if err != nil {
		t.Fatalf("Download: %v", err)
	}
	if len(samples) == 0 {
		t.Fatal("progress never called")
	}
	if samples[len(samples)-1] != int64(len(content)) {
		t.Errorf("final progress = %d, want %d", samples[len(samples)-1], len(content))
	}
}

func TestParallelDownload_RetriesPerRangeOnTransientFailure(t *testing.T) {
	withParallelEnabled(t, 1024, 4*1024, 2)

	content := makeBlob(8 * 1024)

	// Track per-range attempt counts so we can fail the first attempt of
	// exactly one specific range. Subsequent attempts succeed.
	var attempts sync.Map // key: range header string, value: *atomic.Int32

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rangeHeader := r.Header.Get("Range")
		if r.Method == http.MethodHead {
			w.Header().Set("Accept-Ranges", "bytes")
			w.Header().Set("Content-Length", strconv.Itoa(len(content)))
			w.WriteHeader(http.StatusOK)
			return
		}
		if rangeHeader == "" {
			http.Error(w, "want range", http.StatusBadRequest)
			return
		}

		v, _ := attempts.LoadOrStore(rangeHeader, &atomic.Int32{})
		n := v.(*atomic.Int32).Add(1)

		var start, end int64
		_, _ = fmt.Sscanf(rangeHeader, "bytes=%d-%d", &start, &end)

		// Fail the first attempt of the FIRST chunk (start=0) mid-stream.
		if start == 0 && n == 1 {
			w.Header().Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", start, end, len(content)))
			w.WriteHeader(http.StatusPartialContent)
			// Write half the chunk then hijack to force a mid-stream close.
			half := (end - start + 1) / 2
			_, _ = w.Write(content[start : start+half])
			w.(http.Flusher).Flush()
			if hj, ok := w.(http.Hijacker); ok {
				conn, _, _ := hj.Hijack()
				_ = conn.Close()
			}
			return
		}

		if end >= int64(len(content)) {
			end = int64(len(content)) - 1
		}
		w.Header().Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", start, end, len(content)))
		w.Header().Set("Content-Length", strconv.FormatInt(end-start+1, 10))
		w.WriteHeader(http.StatusPartialContent)
		_, _ = w.Write(content[start : end+1])
	}))
	defer srv.Close()

	destDir := t.TempDir()
	path, err := Download(srv.URL+"/flaky.gguf", destDir, "", nil)
	if err != nil {
		t.Fatalf("Download: %v", err)
	}
	got, _ := os.ReadFile(path)
	if !bytes.Equal(got, content) {
		t.Errorf("reassembled content mismatch after retry: len=%d want=%d", len(got), len(content))
	}
}

func TestParallelDownload_SaveAsOverrideStillApplies(t *testing.T) {
	withParallelEnabled(t, 1024, 2*1024, 4)

	content := makeBlob(8 * 1024)
	srv := httptest.NewServer(rangeAwareHandler(content, nil))
	defer srv.Close()

	destDir := t.TempDir()
	path, err := Download(srv.URL+"/Some-Source-Name.gguf", destDir, "my-chosen-name", nil)
	if err != nil {
		t.Fatalf("Download: %v", err)
	}
	want := filepath.Join(destDir, "my-chosen-name.gguf")
	if path != want {
		t.Errorf("path = %q, want %q", path, want)
	}
	if _, err := os.Stat(want); err != nil {
		t.Errorf("file not at expected location: %v", err)
	}
}

func TestChooseConcurrency(t *testing.T) {
	// Pin the tunables so this test isn't sensitive to package-level mutation
	// from other tests running concurrently.
	prevChunk := ParallelDownloadMinChunkSize
	prevMax := ParallelDownloadMaxConcurrency
	ParallelDownloadMinChunkSize = 128 * 1024 * 1024 // 128 MB
	ParallelDownloadMaxConcurrency = 16
	t.Cleanup(func() {
		ParallelDownloadMinChunkSize = prevChunk
		ParallelDownloadMaxConcurrency = prevMax
	})

	cases := []struct {
		name string
		size int64
		want int
	}{
		{"100MB rounds up to 1 chunk", 100 * 1024 * 1024, 1},
		{"128MB exactly is 1 chunk", 128 * 1024 * 1024, 1},
		{"129MB rounds up to 2 chunks", 129 * 1024 * 1024, 2},
		{"500MB → 4 chunks", 500 * 1024 * 1024, 4},
		{"1GB → 8 chunks", 1024 * 1024 * 1024, 8},
		{"2GB → 16 chunks (cap)", 2 * 1024 * 1024 * 1024, 16},
		{"20GB → 16 chunks (cap)", 20 * 1024 * 1024 * 1024, 16},
		{"1 byte → 1 chunk (floor)", 1, 1},
	}
	for _, c := range cases {
		got := chooseConcurrency(c.size)
		if got != c.want {
			t.Errorf("%s: chooseConcurrency(%d) = %d, want %d", c.name, c.size, got, c.want)
		}
	}
}

func TestProbeRanges_HEADWithAcceptRanges(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Accept-Ranges", "bytes")
		w.Header().Set("Content-Length", "12345")
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	size, ranges, err := probeRanges(ctx, srv.URL+"/x.gguf")
	if err != nil {
		t.Fatalf("probeRanges: %v", err)
	}
	if size != 12345 {
		t.Errorf("size = %d, want 12345", size)
	}
	if !ranges {
		t.Errorf("expected Accept-Ranges: bytes -> supportsRanges=true")
	}
}

func TestProbeRanges_RangeGetFallback(t *testing.T) {
	// HEAD omits Accept-Ranges (some CDNs do this), but the GET Range probe
	// returns 206 with Content-Range. probeRanges should recover the true
	// size from Content-Range and report supportsRanges=true.
	totalSize := int64(98765)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodHead {
			w.Header().Set("Content-Length", strconv.FormatInt(totalSize, 10))
			w.WriteHeader(http.StatusOK)
			return
		}
		if !strings.HasPrefix(r.Header.Get("Range"), "bytes=") {
			http.Error(w, "want range", http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Range", fmt.Sprintf("bytes 0-0/%d", totalSize))
		w.Header().Set("Content-Length", "1")
		w.WriteHeader(http.StatusPartialContent)
		_, _ = w.Write([]byte{0})
	}))
	defer srv.Close()

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	size, ranges, err := probeRanges(ctx, srv.URL+"/x.gguf")
	if err != nil {
		t.Fatalf("probeRanges: %v", err)
	}
	if !ranges {
		t.Errorf("expected GET-range probe to confirm support")
	}
	if size != totalSize {
		t.Errorf("size = %d, want %d", size, totalSize)
	}
}

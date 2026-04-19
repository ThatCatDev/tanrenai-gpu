package models

import (
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// Shrink backoff/attempt count for the whole test binary. Production flows
// use the defaults (1s base, 5 attempts ~= 31s worst case); tests want
// near-instant retries so the NetworkError/ServerError cases don't run for
// 30 seconds each.
func init() {
	DownloadBackoffBase = 1 * time.Millisecond
	DownloadStallThreshold = 500 * time.Millisecond
	// 2 attempts so dial-fail tests (NetworkError) don't take 5× the OS
	// connect timeout. Production still uses 5.
	DownloadMaxAttempts = 2
}

func TestDownload_NonGGUFURL(t *testing.T) {
	_, err := Download("http://example.com/model.bin", t.TempDir(), nil)
	if err == nil {
		t.Fatal("expected error for non-.gguf URL")
	}
}

func TestDownload_HappyPath(t *testing.T) {
	content := []byte("fake gguf content for testing")

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(content)
	}))
	defer srv.Close()

	destDir := t.TempDir()
	url := srv.URL + "/model.gguf"

	var lastDownloaded, lastTotal int64
	progress := func(downloaded, total int64) {
		lastDownloaded = downloaded
		lastTotal = total
	}

	path, err := Download(url, destDir, progress)
	if err != nil {
		t.Fatalf("Download error: %v", err)
	}

	if path != filepath.Join(destDir, "model.gguf") {
		t.Errorf("path = %q, want %q", path, filepath.Join(destDir, "model.gguf"))
	}

	// File should exist and have the correct content
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	if string(got) != string(content) {
		t.Errorf("file content = %q, want %q", string(got), string(content))
	}

	// Partial file should be cleaned up
	if _, err := os.Stat(path + ".partial"); !os.IsNotExist(err) {
		t.Error(".partial file should be removed after successful download")
	}

	// Progress should have been called
	if lastDownloaded == 0 {
		t.Error("progress callback was never called or downloaded=0")
	}
	_ = lastTotal
}

func TestDownload_NilProgress(t *testing.T) {
	content := []byte("small gguf data")

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(content)
	}))
	defer srv.Close()

	destDir := t.TempDir()
	url := srv.URL + "/model.gguf"

	// progress=nil should not panic
	path, err := Download(url, destDir, nil)
	if err != nil {
		t.Fatalf("Download error: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Errorf("output file not found: %v", err)
	}
}

func TestDownload_ServerError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	destDir := t.TempDir()
	_, err := Download(srv.URL+"/model.gguf", destDir, nil)
	if err == nil {
		t.Fatal("expected error for server 500")
	}
	if !strings.Contains(err.Error(), "500") {
		t.Errorf("error should mention status 500: %v", err)
	}
}

func TestDownload_PartialResume(t *testing.T) {
	// Simulate a partial download that resumes with 206 Partial Content
	fullContent := []byte("0123456789abcdefghij gguf file content here")
	existingBytes := fullContent[:10]
	remainingBytes := fullContent[10:]

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rangeHeader := r.Header.Get("Range")
		if rangeHeader == "bytes=10-" {
			w.WriteHeader(http.StatusPartialContent)
			_, _ = w.Write(remainingBytes)
		} else {
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write(fullContent)
		}
	}))
	defer srv.Close()

	destDir := t.TempDir()
	url := srv.URL + "/model.gguf"

	// Pre-create a .partial file simulating a partial download
	partialPath := filepath.Join(destDir, "model.gguf.partial")
	if err := os.WriteFile(partialPath, existingBytes, 0644); err != nil {
		t.Fatalf("setup partial file: %v", err)
	}

	path, err := Download(url, destDir, nil)
	if err != nil {
		t.Fatalf("Download error: %v", err)
	}

	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	if string(got) != string(fullContent) {
		t.Errorf("resumed content = %q, want %q", string(got), string(fullContent))
	}
}

func TestDownload_PresignedURLWithQuery(t *testing.T) {
	// Presigned S3/R2 URLs have a long ?X-Amz-... query string. The
	// filename extraction must ignore it, otherwise the .gguf suffix
	// check rejects the URL even though the underlying object is valid.
	content := []byte("gguf bytes from r2")

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(content)
	}))
	defer srv.Close()

	destDir := t.TempDir()
	// Mirror what modelcache.Cache.Lookup returns.
	url := srv.URL + "/models/unsloth/Qwen-GGUF/Q4_K_M/Qwen-Q4_K_M-00001-of-00003.gguf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=deadbeef"

	path, err := Download(url, destDir, nil)
	if err != nil {
		t.Fatalf("Download error on presigned URL: %v", err)
	}
	if !strings.HasSuffix(path, "Qwen-Q4_K_M-00001-of-00003.gguf") {
		t.Errorf("path should land without the query string: got %q", path)
	}
	if _, err := os.Stat(path); err != nil {
		t.Errorf("file missing after successful download: %v", err)
	}
}

func TestDownload_NetworkError(t *testing.T) {
	// .invalid is reserved by RFC 2606 — DNS fails fast, no slow connect
	// attempt on any real port. We want a connect-level failure that
	// doesn't take 30s in WSL.
	_, err := Download("http://no-such-host.invalid/model.gguf", t.TempDir(), nil)
	if err == nil {
		t.Fatal("expected network error")
	}
}

func TestDownload_HFProvenance(t *testing.T) {
	content := []byte("gguf content")

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Simulate HuggingFace URL pattern:
		// /Qwen/Qwen2.5-7B/resolve/main/model.gguf
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(content)
	}))
	defer srv.Close()

	destDir := t.TempDir()

	// Use a real HF-style URL so ParseHFURL returns true, but point at test server
	// We can't fully test HF provenance without a real HF URL, but we can test
	// that a non-HF URL doesn't error.
	url := srv.URL + "/model.gguf"
	path, err := Download(url, destDir, nil)
	if err != nil {
		t.Fatalf("Download error: %v", err)
	}
	if path == "" {
		t.Error("expected non-empty path")
	}
}

func TestDownload_StalledConnection(t *testing.T) {
	// Server advertises a 1 MB file but writes only the first 32 KB, then
	// hangs forever without closing. Without a stall detector the client
	// would block indefinitely on resp.Body.Read.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "1048576")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(make([]byte, 32*1024))
		w.(http.Flusher).Flush()
		// Block forever. The watchdog should cancel the context.
		<-r.Context().Done()
	}))
	defer srv.Close()

	destDir := t.TempDir()
	_, err := Download(srv.URL+"/stall.gguf", destDir, nil)
	if err == nil {
		t.Fatal("expected stall error; got success")
	}
	if !strings.Contains(err.Error(), "stall") && !strings.Contains(err.Error(), "context canceled") {
		t.Errorf("expected stall-related error, got: %v", err)
	}
}

func TestDownload_RetriesAndResumes(t *testing.T) {
	// First GET drops after half the bytes (simulates flaky mid-stream close).
	// Second GET serves the remainder via Range. Download should succeed
	// with the full file assembled.
	full := []byte(strings.Repeat("X", 2048))
	var calls atomic.Int32

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := calls.Add(1)
		rangeHeader := r.Header.Get("Range")
		if n == 1 {
			// First attempt: claim full size, deliver half, then close with partial.
			w.Header().Set("Content-Length", "2048")
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write(full[:1024])
			w.(http.Flusher).Flush()
			// Hijack to force-close mid-stream.
			hj, _ := w.(http.Hijacker)
			if hj != nil {
				conn, _, _ := hj.Hijack()
				_ = conn.Close()
			}
			return
		}
		// Second attempt: resume via Range.
		if rangeHeader != "bytes=1024-" {
			t.Errorf("expected Range: bytes=1024-, got %q", rangeHeader)
		}
		w.Header().Set("Content-Length", "1024")
		w.WriteHeader(http.StatusPartialContent)
		_, _ = w.Write(full[1024:])
	}))
	defer srv.Close()

	destDir := t.TempDir()
	path, err := Download(srv.URL+"/retry.gguf", destDir, nil)
	if err != nil {
		t.Fatalf("Download error: %v", err)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	if string(got) != string(full) {
		t.Errorf("reassembled content length = %d, want %d", len(got), len(full))
	}
	if calls.Load() < 2 {
		t.Errorf("expected at least 2 server calls (initial + Range), got %d", calls.Load())
	}
}

func TestDownload_TruncatedStreamIsRetryable(t *testing.T) {
	// Server advertises 2048 but returns only 1024 and cleanly closes.
	// Old behavior: EOF treated as success, rename with 1024 bytes — silent corruption.
	// New behavior: size validation catches it, retry kicks in. Since the
	// server is deterministic (always truncates), the retry also fails —
	// but we should see "truncated" in the error.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "2048")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(make([]byte, 1024))
	}))
	defer srv.Close()

	destDir := t.TempDir()
	_, err := Download(srv.URL+"/trunc.gguf", destDir, nil)
	if err == nil {
		t.Fatal("expected truncation error; got success")
	}
	// Either path is acceptable: Go's http stack surfaces the body-shorter
	// -than-Content-Length case as io.ErrUnexpectedEOF ("read body"), and
	// for the rare case where the stream closes cleanly with a short size
	// our own "truncated" check fires. Both are retryable and both
	// prevent the old silent-success bug.
	if !strings.Contains(err.Error(), "truncated") && !strings.Contains(err.Error(), "read body") {
		t.Errorf("expected truncation-style error, got: %v", err)
	}
}

func TestDownload_LargeContent_ProgressTracking(t *testing.T) {
	// Generate content larger than the 32KB read buffer
	content := make([]byte, 100*1024) // 100KB
	for i := range content {
		content[i] = byte(i % 256)
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "102400")
		w.WriteHeader(http.StatusOK)
		_, _ = io.Copy(w, strings.NewReader(string(content)))
	}))
	defer srv.Close()

	destDir := t.TempDir()
	url := srv.URL + "/bigmodel.gguf"

	callCount := 0
	progress := func(downloaded, total int64) {
		callCount++
	}

	path, err := Download(url, destDir, progress)
	if err != nil {
		t.Fatalf("Download error: %v", err)
	}

	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	if len(got) != len(content) {
		t.Errorf("downloaded %d bytes, want %d", len(got), len(content))
	}
	if callCount == 0 {
		t.Error("progress callback never called")
	}
}

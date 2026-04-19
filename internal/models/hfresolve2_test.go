package models

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// newHFMockServer creates a test HuggingFace API server that returns the given files.
func newHFMockServer(t *testing.T, files []HFFileInfo) *httptest.Server {
	t.Helper()

	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasPrefix(r.URL.Path, "/api/models/") {
			http.Error(w, "unexpected path", http.StatusNotFound)

			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(struct {
			Siblings []HFFileInfo `json:"siblings"`
		}{Siblings: files})
	}))
}

// withHFBaseURL overrides hfAPIBaseURL for a test and restores it after.
func withHFBaseURL(t *testing.T, url string) {
	t.Helper()
	orig := hfAPIBaseURL
	hfAPIBaseURL = url
	t.Cleanup(func() { hfAPIBaseURL = orig })
}

// TestResolveHFModel_WithMock_NoBestQuant verifies ResolveHFModel with
// auto-selection when no preferred quant exists.
func TestResolveHFModel_WithMock_AutoSelect(t *testing.T) {
	files := []HFFileInfo{
		{Filename: "model-Q4_K_M.gguf", Size: 4000000000},
		{Filename: "model-Q8_0.gguf", Size: 8000000000},
		{Filename: "README.md", Size: 1000},
	}
	srv := newHFMockServer(t, files)
	defer srv.Close()
	withHFBaseURL(t, srv.URL)

	urls, err := ResolveHFModel("hf://owner/repo")
	if err != nil {
		t.Fatalf("ResolveHFModel: %v", err)
	}
	if len(urls) == 0 {
		t.Fatal("expected at least one URL")
	}
	// Should have picked Q4_K_M
	if !strings.Contains(urls[0], "Q4_K_M") {
		t.Errorf("expected Q4_K_M in URL, got %q", urls[0])
	}
}

// TestResolveHFModel_WithMock_QuantFilter verifies quant subfolder filtering.
func TestResolveHFModel_WithMock_QuantFilter(t *testing.T) {
	files := []HFFileInfo{
		{Filename: "Q8_0/model.gguf", Size: 8000000000},
		{Filename: "Q4_K_M/model.gguf", Size: 4000000000},
	}
	srv := newHFMockServer(t, files)
	defer srv.Close()
	withHFBaseURL(t, srv.URL)

	urls, err := ResolveHFModel("hf://owner/repo/Q8_0")
	if err != nil {
		t.Fatalf("ResolveHFModel with quant: %v", err)
	}
	if len(urls) != 1 {
		t.Fatalf("expected 1 URL, got %d: %v", len(urls), urls)
	}
	if !strings.Contains(urls[0], "Q8_0") {
		t.Errorf("URL should contain Q8_0, got %q", urls[0])
	}
}

// TestResolveHFModel_WithMock_QuantNotFound verifies error when quant doesn't match.
func TestResolveHFModel_WithMock_QuantNotFound(t *testing.T) {
	files := []HFFileInfo{
		{Filename: "model-Q4_K_M.gguf"},
	}
	srv := newHFMockServer(t, files)
	defer srv.Close()
	withHFBaseURL(t, srv.URL)

	_, err := ResolveHFModel("hf://owner/repo/nonexistent-quant")
	if err == nil {
		t.Fatal("expected error for nonexistent quant")
	}
}

// TestResolveHFModel_WithMock_NoGGUFFiles verifies error when no GGUFs in repo.
func TestResolveHFModel_WithMock_NoGGUFFiles(t *testing.T) {
	files := []HFFileInfo{
		{Filename: "README.md", Size: 1000},
		{Filename: "config.json", Size: 500},
	}
	srv := newHFMockServer(t, files)
	defer srv.Close()
	withHFBaseURL(t, srv.URL)

	_, err := ResolveHFModel("hf://owner/repo")
	if err == nil {
		t.Fatal("expected error when no GGUF files in repo")
	}
}

// TestResolveHFModel_WithMock_SplitFiles verifies split GGUF files are returned together.
func TestResolveHFModel_WithMock_SplitFiles(t *testing.T) {
	files := []HFFileInfo{
		{Filename: "model-Q4_K_M-00001-of-00002.gguf"},
		{Filename: "model-Q4_K_M-00002-of-00002.gguf"},
		{Filename: "model-Q8_0.gguf"},
	}
	srv := newHFMockServer(t, files)
	defer srv.Close()
	withHFBaseURL(t, srv.URL)

	urls, err := ResolveHFModel("hf://owner/repo")
	if err != nil {
		t.Fatalf("ResolveHFModel: %v", err)
	}
	// Q4_K_M split files should both be returned
	if len(urls) != 2 {
		t.Errorf("expected 2 URLs for split Q4_K_M, got %d: %v", len(urls), urls)
	}
}

// TestResolveHFModel_WithMock_FilenameSubstringMatch verifies filename substring matching
// as fallback when subfolder match fails.
func TestResolveHFModel_WithMock_FilenameSubstringMatch(t *testing.T) {
	files := []HFFileInfo{
		{Filename: "Mixtral-8x7B-Q4_K_M.gguf"},
		{Filename: "Mixtral-8x7B-Q8_0.gguf"},
	}
	srv := newHFMockServer(t, files)
	defer srv.Close()
	withHFBaseURL(t, srv.URL)

	urls, err := ResolveHFModel("hf://owner/repo/Q8_0")
	if err != nil {
		t.Fatalf("ResolveHFModel with substring match: %v", err)
	}
	if len(urls) != 1 {
		t.Fatalf("expected 1 URL, got %d: %v", len(urls), urls)
	}
	if !strings.Contains(urls[0], "Q8_0") {
		t.Errorf("URL should contain Q8_0, got %q", urls[0])
	}
}

// TestListHFFiles_DirectCall verifies listHFFiles via the overridable base URL.
func TestListHFFiles_DirectCall_Success(t *testing.T) {
	files := []HFFileInfo{
		{Filename: "model-Q4_K_M.gguf", Size: 4000000000},
	}
	srv := newHFMockServer(t, files)
	defer srv.Close()
	withHFBaseURL(t, srv.URL)

	result, err := listHFFiles("owner/repo")
	if err != nil {
		t.Fatalf("listHFFiles: %v", err)
	}
	if len(result) != 1 {
		t.Errorf("expected 1 file, got %d", len(result))
	}
}

// TestListHFFiles_DirectCall_Error verifies listHFFiles returns error on non-200.
func TestListHFFiles_DirectCall_Error(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer srv.Close()
	withHFBaseURL(t, srv.URL)

	_, err := listHFFiles("private/repo")
	if err == nil {
		t.Fatal("expected error for 401 response")
	}
}

// TestListHFFiles_DirectCall_BadJSON verifies listHFFiles returns error for bad JSON.
func TestListHFFiles_DirectCall_BadJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("{invalid"))
	}))
	defer srv.Close()
	withHFBaseURL(t, srv.URL)

	_, err := listHFFiles("owner/repo")
	if err == nil {
		t.Fatal("expected error for bad JSON")
	}
}

// TestListHFFiles_DirectCall_NetworkError verifies listHFFiles returns error on network failure.
func TestListHFFiles_DirectCall_NetworkError(t *testing.T) {
	withHFBaseURL(t, "http://127.0.0.1:1")

	_, err := listHFFiles("owner/repo")
	if err == nil {
		t.Fatal("expected error for network failure")
	}
}

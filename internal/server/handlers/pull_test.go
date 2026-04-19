package handlers

import (
	"bufio"
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/ThatCatDev/tanrenai-gpu/internal/models"
)

// ---- PullHandler ----

func TestPullHandler_BadJSON(t *testing.T) {
	store := models.NewStore(t.TempDir())
	h := &PullHandler{Store: store}

	req := httptest.NewRequest(http.MethodPost, "/api/pull", bytes.NewReader([]byte("{invalid")))
	w := newFlushableRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", w.Code)
	}
	assertErrorType(t, w.ResponseRecorder, "invalid_request")
}

func TestPullHandler_MissingURL(t *testing.T) {
	store := models.NewStore(t.TempDir())
	h := &PullHandler{Store: store}

	body := mustMarshal(t, map[string]string{"url": ""})
	req := httptest.NewRequest(http.MethodPost, "/api/pull", bytes.NewReader(body))
	w := newFlushableRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", w.Code)
	}
	assertErrorType(t, w.ResponseRecorder, "invalid_request")
}

func TestPullHandler_InvalidHFRef(t *testing.T) {
	store := models.NewStore(t.TempDir())
	h := &PullHandler{Store: store}

	body := mustMarshal(t, map[string]string{"url": "hf://invalid"})
	req := httptest.NewRequest(http.MethodPost, "/api/pull", bytes.NewReader(body))
	w := newFlushableRecorder()

	h.ServeHTTP(w, req)

	// Should stream an error event
	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200 (SSE error is streamed)", w.Code)
	}

	ct := w.Header().Get("Content-Type")
	if ct != "text/event-stream" {
		t.Errorf("Content-Type = %q, want text/event-stream", ct)
	}

	// Parse SSE events and find error
	events := parseSSEEvents(t, w.Body.String())
	if len(events) == 0 {
		t.Fatal("expected at least one SSE event")
	}
	lastEvent := events[len(events)-1]
	if lastEvent["status"] != "error" {
		t.Errorf("last event status = %q, want %q", lastEvent["status"], "error")
	}
}

func TestPullHandler_DirectURLDownload(t *testing.T) {
	// Start a fake GGUF download server
	ggufContent := []byte("fake gguf model data")
	dlServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(ggufContent)
	}))
	defer dlServer.Close()

	store := models.NewStore(t.TempDir())
	h := &PullHandler{Store: store}

	// Use a direct URL pointing to our test server
	dlURL := dlServer.URL + "/model.gguf"
	body := mustMarshal(t, map[string]string{"url": dlURL})
	req := httptest.NewRequest(http.MethodPost, "/api/pull", bytes.NewReader(body))
	w := newFlushableRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}

	ct := w.Header().Get("Content-Type")
	if ct != "text/event-stream" {
		t.Errorf("Content-Type = %q, want text/event-stream", ct)
	}

	events := parseSSEEvents(t, w.Body.String())
	if len(events) == 0 {
		t.Fatal("expected SSE events")
	}

	// First event should be "resolving"
	if events[0]["status"] != "resolving" {
		t.Errorf("first event status = %q, want %q", events[0]["status"], "resolving")
	}

	// Last event should be "downloaded"
	last := events[len(events)-1]
	if last["status"] != "downloaded" {
		t.Errorf("last event status = %q, want %q (events: %v)", last["status"], "downloaded", events)
	}
}

func TestPullHandler_DownloadError(t *testing.T) {
	// Server that returns an error
	dlServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusForbidden)
	}))
	defer dlServer.Close()

	store := models.NewStore(t.TempDir())
	h := &PullHandler{Store: store}

	dlURL := dlServer.URL + "/model.gguf"
	body := mustMarshal(t, map[string]string{"url": dlURL})
	req := httptest.NewRequest(http.MethodPost, "/api/pull", bytes.NewReader(body))
	w := newFlushableRecorder()

	h.ServeHTTP(w, req)

	events := parseSSEEvents(t, w.Body.String())
	if len(events) == 0 {
		t.Fatal("expected SSE events")
	}

	last := events[len(events)-1]
	if last["status"] != "error" {
		t.Errorf("last event status = %q, want %q (events: %v)", last["status"], "error", events)
	}
}

// flushableRecorder is a ResponseRecorder that also implements http.Flusher.
type flushableRecorder struct {
	*httptest.ResponseRecorder
}

func newFlushableRecorder() *flushableRecorder {
	return &flushableRecorder{httptest.NewRecorder()}
}

func (f *flushableRecorder) Flush() {
	// no-op — data is already in the buffer
}

// parseSSEEvents parses the SSE body and returns a slice of event data maps.
func parseSSEEvents(t *testing.T, body string) []map[string]any {
	t.Helper()
	var events []map[string]any
	scanner := bufio.NewScanner(strings.NewReader(body))
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		var ev map[string]any
		if err := json.Unmarshal([]byte(data), &ev); err != nil {
			t.Logf("could not parse SSE event %q: %v", data, err)

			continue
		}
		events = append(events, ev)
	}

	return events
}

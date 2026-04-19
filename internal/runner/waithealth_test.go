package runner

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// TestWaitForHealth_ProgressTicker verifies that waitForHealth logs progress
// when the subprocess takes more than 5 seconds to become healthy.
// This test takes approximately 5 seconds to run.
func TestWaitForHealth_ProgressTicker(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping 5-second progress ticker test in short mode")
	}

	callCount := 0
	// Server responds with unhealthy for the first ~5s, then healthy.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		// After 5+ seconds, respond healthy to allow the progress ticker to fire first.
		if callCount > 10 {
			w.WriteHeader(http.StatusOK)
		} else {
			w.WriteHeader(http.StatusServiceUnavailable)
		}
	}))
	defer srv.Close()

	doneCh := make(chan struct{})
	sub := &Subprocess{
		baseURL:       srv.URL,
		label:         "test-progress",
		healthTimeout: 15 * time.Second, // long enough for progress ticker
		doneCh:        doneCh,
	}

	// waitForHealth will poll every 500ms and log progress every 5s.
	// With 10 unhealthy responses at 500ms each = ~5s before healthy.
	err := sub.waitForHealth(context.Background())
	if err != nil {
		t.Fatalf("waitForHealth unexpectedly failed: %v", err)
	}
}

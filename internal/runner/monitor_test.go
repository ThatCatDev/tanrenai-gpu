package runner

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// TestMonitorCrashes_WasStoppedExits verifies that monitorCrashes exits cleanly
// when the subprocess doneCh is closed but WasStopped() is true (intentional stop).
func TestMonitorCrashes_WasStoppedExits(t *testing.T) {
	doneCh := make(chan struct{})
	sub := &Subprocess{
		doneCh:        doneCh,
		stopped:       true, // WasStopped() returns true
		healthTimeout: 5 * time.Second,
	}

	r := NewProcessRunner()
	r.sub = sub

	done := make(chan struct{})
	go func() {
		defer close(done)
		r.monitorCrashes()
	}()

	// Trigger doneCh so the select fires
	close(doneCh)

	select {
	case <-done:
		// monitorCrashes exited because WasStopped() was true
	case <-time.After(2 * time.Second):
		t.Fatal("monitorCrashes did not exit when WasStopped() was true")
	}
}

// TestMonitorCrashes_StopMonitorExits verifies that monitorCrashes exits when
// stopMonitor is closed.
func TestMonitorCrashes_StopMonitorExits(t *testing.T) {
	doneCh := make(chan struct{})
	sub := &Subprocess{
		doneCh:        doneCh,
		healthTimeout: 5 * time.Second,
	}

	r := NewProcessRunner()
	r.sub = sub

	done := make(chan struct{})
	go func() {
		defer close(done)
		r.monitorCrashes()
	}()

	// Close stopMonitor to stop the monitor goroutine
	close(r.stopMonitor)

	select {
	case <-done:
		// monitorCrashes exited cleanly
	case <-time.After(2 * time.Second):
		t.Fatal("monitorCrashes did not exit when stopMonitor was closed")
	}
}

// TestMonitorCrashes_CrashNotify verifies that when the subprocess crashes
// (doneCh closed, WasStopped=false), monitorCrashes sends to crashNotify
// and then exits when max restarts is exceeded.
func TestMonitorCrashes_CrashNotify(t *testing.T) {
	// Set up a mock subprocess with a closed doneCh to simulate a crash.
	// The subprocess was NOT stopped intentionally.
	doneCh := make(chan struct{})
	close(doneCh) // already crashed

	sub := &Subprocess{
		doneCh:        doneCh,
		stopped:       false,
		healthTimeout: 1 * time.Second,
	}

	r := NewProcessRunner()
	r.sub = sub
	// Pre-set restarts to maxRestartAttempts so the monitor gives up immediately.
	r.restarts = maxRestartAttempts

	done := make(chan struct{})
	go func() {
		defer close(done)
		r.monitorCrashes()
	}()

	// Should receive a crash notification
	select {
	case err := <-r.crashNotify:
		if err == nil {
			t.Error("expected non-nil crash error")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("expected crash notification but got none")
	}

	// monitorCrashes should exit after exceeding max restarts
	select {
	case <-done:
		// exited as expected
	case <-time.After(2 * time.Second):
		t.Fatal("monitorCrashes did not exit after max restarts")
	}
}

// TestProcessRunner_Close_WithSubprocess verifies that Close calls GracefulStop
// on the subprocess when one exists.
func TestProcessRunner_Close_WithSubprocess(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	// Build a runner with a subprocess that has no real process (nil cmd).
	doneCh := make(chan struct{})
	sub := &Subprocess{
		baseURL:       srv.URL,
		label:         "test",
		doneCh:        doneCh,
		healthTimeout: 5 * time.Second,
	}

	r := NewProcessRunner()
	r.sub = sub

	if err := r.Close(); err != nil {
		t.Errorf("Close with subprocess (nil cmd): %v", err)
	}

	if !sub.WasStopped() {
		t.Error("GracefulStop should have set stopped=true")
	}
}

// TestMonitorCrashes_CrashRestartFails verifies the path where restart fails
// and the monitor exits.
func TestMonitorCrashes_CrashRestartFails(t *testing.T) {
	doneCh := make(chan struct{})
	close(doneCh) // already crashed

	sub := &Subprocess{
		doneCh:        doneCh,
		stopped:       false,
		healthTimeout: 1 * time.Second,
		// binPath is empty → Start will fail
	}

	r := NewProcessRunner()
	r.sub = sub
	// With restarts=0 and maxRestartAttempts=3, the first crash will attempt restart.
	// startSubprocess will fail because sub.binPath is empty.
	// But startSubprocess calls NewSubprocess which calls resolveBinary which
	// will fail since r.opts.BinDir is empty. So restart attempt will fail → exit.
	r.restarts = 0

	done := make(chan struct{})
	go func() {
		defer close(done)
		r.monitorCrashes()
	}()

	// Should get a crash notification
	select {
	case <-r.crashNotify:
		// received notification
	case <-time.After(3 * time.Second):
		t.Fatal("expected crash notification")
	}

	// Monitor should exit after failed restart
	select {
	case <-done:
		// exited as expected
	case <-time.After(3 * time.Second):
		t.Fatal("monitorCrashes did not exit after failed restart")
	}
}

// TestMonitorCrashes_WaitForHealthDoneCh verifies waitForHealth returns an error
// when doneCh is closed (process exited during startup).
func TestMonitorCrashes_WaitForHealthDoneCh(t *testing.T) {
	// Use an httptest server that always fails health check
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer srv.Close()

	doneCh := make(chan struct{})
	sub := &Subprocess{
		baseURL:       srv.URL,
		label:         "test",
		doneCh:        doneCh,
		healthTimeout: 10 * time.Second,
	}

	// Close doneCh immediately to simulate process exit during startup
	close(doneCh)

	err := sub.waitForHealth(context.Background())
	if err == nil {
		t.Fatal("expected error when doneCh is closed during waitForHealth")
	}
}

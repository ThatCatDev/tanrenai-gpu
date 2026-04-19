package runner

import (
	"testing"
)

// TestSubprocess_BaseURL covers BaseURL accessor.
func TestSubprocess_BaseURL(t *testing.T) {
	sub := &Subprocess{baseURL: "http://127.0.0.1:11435"}
	if sub.BaseURL() != "http://127.0.0.1:11435" {
		t.Errorf("BaseURL() = %q, want %q", sub.BaseURL(), "http://127.0.0.1:11435")
	}
}

// TestSubprocess_Healthy_Default covers initial false state.
func TestSubprocess_Healthy_Default(t *testing.T) {
	sub := &Subprocess{}
	if sub.Healthy() {
		t.Error("Healthy() should be false by default")
	}
}

// TestSubprocess_Healthy_True covers setting healthy to true.
func TestSubprocess_Healthy_True(t *testing.T) {
	sub := &Subprocess{healthy: true}
	if !sub.Healthy() {
		t.Error("Healthy() should be true when set")
	}
}

// TestSubprocess_WasStopped_False covers default stopped=false.
func TestSubprocess_WasStopped_False(t *testing.T) {
	sub := &Subprocess{}
	if sub.WasStopped() {
		t.Error("WasStopped() should be false by default")
	}
}

// TestSubprocess_WasStopped_True covers stopped=true.
func TestSubprocess_WasStopped_True(t *testing.T) {
	sub := &Subprocess{stopped: true}
	if !sub.WasStopped() {
		t.Error("WasStopped() should be true when set")
	}
}

// TestSubprocess_ExitCode_NilCmd covers ExitCode when cmd is nil.
func TestSubprocess_ExitCode_NilCmd(t *testing.T) {
	sub := &Subprocess{}
	if sub.ExitCode() != -1 {
		t.Errorf("ExitCode() = %d, want -1 for nil cmd", sub.ExitCode())
	}
}

// TestSubprocess_Done_Channel covers Done returning the doneCh.
func TestSubprocess_Done_Channel(t *testing.T) {
	doneCh := make(chan struct{})
	sub := &Subprocess{doneCh: doneCh}

	ch := sub.Done()
	if ch == nil {
		t.Fatal("Done() returned nil channel")
	}

	// Close the done channel and verify it's readable
	close(doneCh)
	select {
	case <-ch:
		// OK
	default:
		t.Error("Done() channel should be closed after close(doneCh)")
	}
}

// TestSubprocess_Port covers the Port accessor.
func TestSubprocess_Port(t *testing.T) {
	sub := &Subprocess{port: 11435}
	if sub.Port() != 11435 {
		t.Errorf("Port() = %d, want 11435", sub.Port())
	}
}

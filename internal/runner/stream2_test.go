package runner

import (
	"errors"
	"strings"
	"testing"
)

// errorReader is an io.Reader that returns a fixed number of bytes then an error.
type errorReader struct {
	data []byte
	pos  int
	err  error
}

func (r *errorReader) Read(p []byte) (int, error) {
	if r.pos < len(r.data) {
		n := copy(p, r.data[r.pos:])
		r.pos += n

		return n, nil
	}

	return 0, r.err
}

// TestParseSSEStream_ScannerError verifies that ParseSSEStream sends a StreamEvent
// with the underlying scanner error when the reader fails mid-stream (after valid lines).
func TestParseSSEStream_ScannerError(t *testing.T) {
	// To reach scanner.Err(), we need the reader to fail AFTER providing complete lines
	// (so the JSON parse doesn't fail first and return early).
	// We use a non-data line first (which is skipped), then fail on the next read.
	// The scanner will buffer the non-data line, process it (skip), then try to read more,
	// hit the error, and set scanner.Err().
	customErr := errors.New("simulated read failure")
	// Provide a non-data line (skipped by ParseSSEStream), then fail.
	// The scanner will process the line, then try to read more and get the error.
	r := &errorReader{
		data: []byte("event: ping\n"),
		err:  customErr,
	}

	ch := ParseSSEStream(r)
	var gotErr error
	for ev := range ch {
		if ev.Err != nil {
			gotErr = ev.Err
		}
	}
	if gotErr == nil {
		t.Fatal("expected scanner error event, got none")
	}
}

// TestParseSSEStream_InvalidJSON2 verifies that ParseSSEStream sends a StreamEvent
// with an error when it encounters invalid JSON after "data: ".
func TestParseSSEStream_InvalidJSON2(t *testing.T) {
	input := "data: {invalid json here}\n\n"
	ch := ParseSSEStream(strings.NewReader(input))

	ev, ok := <-ch
	if !ok {
		t.Fatal("channel closed without sending an event")
	}
	if ev.Err == nil {
		t.Fatal("expected error event for invalid JSON, got none")
	}
}

// TestParseSSEStream_SkipsNonDataLines2 verifies that ParseSSEStream handles a reader
// that returns lines normally then stops.
func TestParseSSEStream_SkipsNonDataLines2(t *testing.T) {
	// Lines without "data: " prefix should be skipped.
	input := "event: message\nretry: 1000\ndata: [DONE]\n\n"
	ch := ParseSSEStream(strings.NewReader(input))

	ev, ok := <-ch
	if !ok {
		t.Fatal("channel closed without sending an event")
	}
	if !ev.Done {
		t.Errorf("expected Done event, got: %+v", ev)
	}
}

// TestParseSSEStream_EmptyInput2 verifies the channel is closed cleanly on empty input.
func TestParseSSEStream_EmptyInput2(t *testing.T) {
	ch := ParseSSEStream(strings.NewReader(""))
	// Channel should be closed with no events.
	for range ch {
		// drain
	}
}

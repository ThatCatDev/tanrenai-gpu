package runner

import (
	"bytes"
	"io"
	"strings"
	"testing"
)

func TestStreamForwardingSSEDetectsDone(t *testing.T) {
	src := strings.NewReader("data: {\"choices\":[{}]}\n\ndata: [DONE]\n\n")
	var dst bytes.Buffer
	sawDone, _, err := streamForwardingSSE(&dst, src)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !sawDone {
		t.Error("expected [DONE] to be detected on a clean stream")
	}
	if dst.String() != "data: {\"choices\":[{}]}\n\ndata: [DONE]\n\n" {
		t.Errorf("forwarded bytes mismatch: %q", dst.String())
	}
}

func TestStreamForwardingSSEFlagsMissingDone(t *testing.T) {
	// llama-server aborts mid-generation: a chunk, then a clean EOF, no [DONE].
	src := strings.NewReader("data: {\"choices\":[{\"delta\":{\"content\":\"hel\"}}]}\n\n")
	var dst bytes.Buffer
	sawDone, tail, err := streamForwardingSSE(&dst, src)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if sawDone {
		t.Error("must NOT report [DONE] when the stream ended without it")
	}
	if !strings.Contains(string(tail), "hel") {
		t.Errorf("tail should carry the last bytes seen, got %q", tail)
	}
}

// chunkedReader yields its data in fixed-size pieces to force [DONE] across a
// read boundary.
type chunkedReader struct {
	data []byte
	size int
	pos  int
}

func (c *chunkedReader) Read(p []byte) (int, error) {
	if c.pos >= len(c.data) {
		return 0, io.EOF
	}
	end := c.pos + c.size
	if end > len(c.data) {
		end = len(c.data)
	}
	n := copy(p, c.data[c.pos:end])
	c.pos += n
	return n, nil
}

func TestStreamForwardingSSEDoneAcrossBoundary(t *testing.T) {
	// One byte at a time guarantees [DONE] is split across reads.
	src := &chunkedReader{data: []byte("data: {}\n\ndata: [DONE]\n\n"), size: 1}
	var dst bytes.Buffer
	sawDone, _, err := streamForwardingSSE(&dst, src)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !sawDone {
		t.Error("split [DONE] across read boundaries must still be detected")
	}
}

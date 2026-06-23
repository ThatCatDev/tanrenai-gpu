package runner

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"time"

	"github.com/ThatCatDev/tanrenai-gpu/pkg/api"
)

// Client is an HTTP client for communicating with a llama-server subprocess.
type Client struct {
	baseURL    string
	httpClient *http.Client
}

// NewClient creates a new Client for the given base URL, optimized for
// loopback communication with a llama-server subprocess.
func NewClient(baseURL string) *Client {
	return &Client{
		baseURL: baseURL,
		httpClient: &http.Client{
			Transport: &http.Transport{
				DialContext: (&net.Dialer{
					Timeout:   5 * time.Second,
					KeepAlive: 30 * time.Second,
				}).DialContext,
				MaxIdleConns:          20,
				MaxIdleConnsPerHost:   10,
				IdleConnTimeout:       120 * time.Second,
				ResponseHeaderTimeout: 300 * time.Second,
			},
		},
	}
}

// ChatCompletion sends a non-streaming chat completion request.
func (c *Client) ChatCompletion(ctx context.Context, req *api.ChatCompletionRequest) (*api.ChatCompletionResponse, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("send request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		slog.Error("llama-server returned non-200 (chat)", "status", resp.StatusCode, "body", string(respBody))

		return nil, fmt.Errorf("llama-server returned %d: %s", resp.StatusCode, string(respBody))
	}

	var result api.ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return &result, nil
}

// Tokenize sends text to the /tokenize endpoint and returns the token count.
func (c *Client) Tokenize(ctx context.Context, text string) (int, error) {
	payload := struct {
		Content string `json:"content"`
	}{Content: text}

	body, err := json.Marshal(payload)
	if err != nil {
		return 0, fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/tokenize", bytes.NewReader(body))
	if err != nil {
		return 0, fmt.Errorf("create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return 0, fmt.Errorf("send request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)

		return 0, fmt.Errorf("tokenize returned %d: %s", resp.StatusCode, string(respBody))
	}

	var result struct {
		Tokens []int `json:"tokens"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return 0, fmt.Errorf("decode response: %w", err)
	}

	return len(result.Tokens), nil
}

// ChatCompletionStream sends a streaming chat completion request and writes SSE chunks to the writer.
func (c *Client) ChatCompletionStream(ctx context.Context, req *api.ChatCompletionRequest, w io.Writer) error {
	body, err := json.Marshal(req)
	if err != nil {
		return fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("send request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		slog.Error("llama-server returned non-200 (chat stream)", "status", resp.StatusCode, "body", string(respBody))

		return fmt.Errorf("llama-server returned %d: %s", resp.StatusCode, string(respBody))
	}

	// Pipe the SSE stream to the response writer while tracking whether
	// llama-server sent its terminal [DONE]. llama-server already formats proper
	// SSE (data: {...}\n\n) and ends a healthy completion with `data: [DONE]`.
	//
	// A stream that ends WITHOUT [DONE] — even via a clean EOF (which io.Copy
	// would report as success) — means llama-server aborted the generation
	// mid-flight (slot/KV pressure, an internal error, a dropped connection).
	// We surface that as an error, with the last bytes seen, so the handler can
	// tell the caller instead of letting the stream go silently dead.
	sawDone, tail, err := streamForwardingSSE(w, resp.Body)
	if err != nil {
		return fmt.Errorf("stream copy failed: %w (last_bytes=%q)", err, tail)
	}
	if !sawDone {
		return fmt.Errorf("llama-server closed stream without [DONE] (last_bytes=%q)", tail)
	}

	return nil
}

// sseDoneMarker is the terminator llama-server sends on a clean completion.
var sseDoneMarker = []byte("[DONE]")

// streamTailKeep bounds the rolling tail: enough to span a split [DONE] marker
// across reads and to carry the last SSE event(s) for diagnostics.
const streamTailKeep = 512

// streamForwardingSSE copies src to dst, flushing each chunk so SSE events reach
// the caller promptly, and reports whether the terminal [DONE] marker was seen
// plus the trailing bytes of the stream (the last event(s) before it ended).
// The rolling tail makes [DONE] detection robust to the marker being split
// across read boundaries.
func streamForwardingSSE(dst io.Writer, src io.Reader) (sawDone bool, tail []byte, err error) {
	flusher, _ := dst.(http.Flusher)
	buf := make([]byte, 4096)
	for {
		n, rerr := src.Read(buf)
		if n > 0 {
			combined := append(tail, buf[:n]...)
			if len(combined) > streamTailKeep {
				combined = combined[len(combined)-streamTailKeep:]
			}
			tail = make([]byte, len(combined))
			copy(tail, combined)
			if bytes.Contains(tail, sseDoneMarker) {
				sawDone = true
			}
			if _, werr := dst.Write(buf[:n]); werr != nil {
				return sawDone, tail, werr
			}
			if flusher != nil {
				flusher.Flush()
			}
		}
		if rerr != nil {
			if rerr == io.EOF {
				return sawDone, tail, nil
			}
			return sawDone, tail, rerr
		}
	}
}

package handlers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/ThatCatDev/tanrenai-gpu/pkg/api"
)

// EmbeddingsHandler handles POST /v1/embeddings.
// It proxies embedding requests to the embedding llama-server subprocess.
type EmbeddingsHandler struct {
	EmbeddingBaseURL string // base URL of the embedding subprocess
}

func (h *EmbeddingsHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20) // 1MB limit

	if h.EmbeddingBaseURL == "" {
		writeError(w, http.StatusServiceUnavailable, "no_embedding", "embedding server not configured")

		return
	}

	var req api.EmbeddingRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", "failed to parse request body: "+err.Error())

		return
	}

	if req.Input == "" {
		writeError(w, http.StatusBadRequest, "invalid_request", "input must not be empty")

		return
	}

	// Forward to the embedding subprocess
	body, err := json.Marshal(req)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "internal_error", "failed to marshal request")

		return
	}

	proxyReq, err := http.NewRequestWithContext(r.Context(), http.MethodPost, h.EmbeddingBaseURL+"/v1/embeddings", bytes.NewReader(body))
	if err != nil {
		writeError(w, http.StatusInternalServerError, "internal_error", fmt.Sprintf("failed to create request: %v", err))

		return
	}
	proxyReq.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(proxyReq)
	if err != nil {
		writeError(w, http.StatusBadGateway, "embedding_error", fmt.Sprintf("embedding server error: %v", err))

		return
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		writeError(w, resp.StatusCode, "embedding_error", string(respBody))

		return
	}

	// Stream the response through
	w.Header().Set("Content-Type", "application/json")
	_, _ = io.Copy(w, resp.Body)
}

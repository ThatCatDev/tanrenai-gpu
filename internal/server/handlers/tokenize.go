package handlers

import (
	"encoding/json"
	"net/http"

	"github.com/ThatCatDev/tanrenai-gpu/internal/runner"
)

// TokenizeHandler handles POST /tokenize.
type TokenizeHandler struct {
	GetRunner func() runner.Runner
}

func (h *TokenizeHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20) // 1MB limit

	rn := h.GetRunner()
	if rn == nil {
		writeError(w, http.StatusServiceUnavailable, "no_model", "no model loaded")

		return
	}

	var req struct {
		Content string `json:"content"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", "failed to parse request body")

		return
	}

	count, err := rn.Tokenize(r.Context(), req.Content)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "tokenize_error", err.Error())

		return
	}

	// Return a tokens array of the right length (matching llama-server format)
	tokens := make([]int, count)
	for i := range tokens {
		tokens[i] = i
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"tokens": tokens})
}

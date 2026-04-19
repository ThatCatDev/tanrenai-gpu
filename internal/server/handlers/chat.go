package handlers

import (
	"context"
	"encoding/json"
	"net/http"

	"github.com/ThatCatDev/tanrenai-gpu/internal/runner"
	"github.com/ThatCatDev/tanrenai-gpu/pkg/api"
)

// ChatHandler handles POST /v1/chat/completions.
type ChatHandler struct {
	GetRunner func() runner.Runner
	LoadFunc  func(ctx context.Context, model string) (*LoadResult, error)
}

func (h *ChatHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, 10<<20) // 10MB limit

	var req api.ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", "failed to parse request body: "+err.Error())

		return
	}

	if len(req.Messages) == 0 {
		writeError(w, http.StatusBadRequest, "invalid_request", "messages must not be empty")

		return
	}

	// Auto-load the model if not already loaded or if a different model is requested
	currentRunner := h.GetRunner()
	if currentRunner == nil || (req.Model != "" && normalizeModelName(currentRunner.ModelName()) != normalizeModelName(req.Model)) {
		if req.Model == "" {
			writeError(w, http.StatusBadRequest, "invalid_request", "no model specified and no model loaded")

			return
		}
		if _, err := h.LoadFunc(r.Context(), req.Model); err != nil {
			writeError(w, http.StatusInternalServerError, "model_error", "failed to load model: "+err.Error())

			return
		}
		currentRunner = h.GetRunner()
	}

	if req.Stream {
		h.handleStream(w, r, &req, currentRunner)
	} else {
		h.handleComplete(w, r, &req, currentRunner)
	}
}

func (h *ChatHandler) handleComplete(w http.ResponseWriter, r *http.Request, req *api.ChatCompletionRequest, rn runner.Runner) {
	resp, err := rn.ChatCompletion(r.Context(), req)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "inference_error", err.Error())

		return
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func (h *ChatHandler) handleStream(w http.ResponseWriter, r *http.Request, req *api.ChatCompletionRequest, rn runner.Runner) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, ok := w.(http.Flusher)
	if ok {
		flusher.Flush()
	}

	if err := rn.ChatCompletionStream(r.Context(), req, w); err != nil {
		return
	}
}

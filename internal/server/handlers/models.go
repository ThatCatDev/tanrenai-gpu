package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/ThatCatDev/tanrenai-gpu/internal/models"
	"github.com/ThatCatDev/tanrenai-gpu/pkg/api"
)

// ModelsHandler handles GET /v1/models.
type ModelsHandler struct {
	Store *models.Store
}

func (h *ModelsHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	available := h.Store.List()

	data := make([]api.ModelInfo, 0, len(available))
	for _, m := range available {
		data = append(data, api.ModelInfo{
			ID:      m.Name,
			Object:  "model",
			Created: m.ModifiedAt,
			OwnedBy: "local",
		})
	}

	resp := api.ModelListResponse{
		Object: "list",
		Data:   data,
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

// LoadResult contains information returned by a model load.
type LoadResult struct {
	CtxSize int
}

// LoadHandler handles POST /api/load.
type LoadHandler struct {
	LoadFunc func(ctx context.Context, model string) (*LoadResult, error)
}

func (h *LoadHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20) // 1MB limit

	var req struct {
		Model string `json:"model"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", "failed to parse request body")

		return
	}

	if req.Model == "" {
		writeError(w, http.StatusBadRequest, "invalid_request", "model field is required")

		return
	}

	result, err := h.LoadFunc(r.Context(), req.Model)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "model_error", err.Error())

		return
	}

	resp := api.LoadResponse{
		Status: "loaded",
		Model:  req.Model,
	}
	if result != nil {
		resp.CtxSize = result.CtxSize
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

// PullHandler handles POST /api/pull — download a model.
type PullHandler struct {
	Store *models.Store
}

func (h *PullHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20) // 1MB limit

	var req struct {
		URL string `json:"url"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", "failed to parse request body")

		return
	}

	if req.URL == "" {
		writeError(w, http.StatusBadRequest, "invalid_request", "url field is required")

		return
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "server_error", "streaming not supported")

		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	sendEvent := func(evt any) {
		data, _ := json.Marshal(evt)
		_, _ = fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}

	// Resolve hf:// references into direct URLs
	urls, err := models.ResolveHFModel(req.URL)
	if err != nil {
		sendEvent(map[string]string{"status": "error", "error": err.Error()})

		return
	}

	sendEvent(map[string]any{"status": "resolving", "files": len(urls)})

	var lastPath string
	for i, dlURL := range urls {
		var lastPercent int
		fileNum := i + 1
		progress := func(downloaded, total int64) {
			if total <= 0 {
				return
			}
			percent := int(downloaded * 100 / total)
			if percent == lastPercent && percent != 100 {
				return
			}
			lastPercent = percent
			sendEvent(map[string]any{
				"status":      "downloading",
				"file":        fileNum,
				"total_files": len(urls),
				"downloaded":  downloaded,
				"total":       total,
				"percent":     percent,
			})
		}

		path, err := models.Download(dlURL, h.Store.Dir(), progress)
		if err != nil {
			sendEvent(map[string]string{"status": "error", "error": err.Error()})

			return
		}
		lastPath = path
	}

	sendEvent(map[string]string{"status": "downloaded", "path": lastPath})
}

// normalizeModelName strips common extensions (.gguf, etc.) for comparison.
func normalizeModelName(name string) string {
	name = strings.TrimSuffix(name, ".gguf")

	return name
}

func writeError(w http.ResponseWriter, status int, errType, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(api.ErrorResponse{
		Error: api.ErrorDetail{
			Message: message,
			Type:    errType,
		},
	})
}

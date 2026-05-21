package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/ThatCatDev/tanrenai-gpu/internal/models"
	"github.com/ThatCatDev/tanrenai-gpu/pkg/api"
	"github.com/ThatCatDev/tanrenai-gpu/pkg/naming"
)

// deriveSaveAs picks the destination basename to pass into models.Download.
// An explicit caller-supplied name always wins. Otherwise, if the URL is a
// canonical hf://<org>/<repo>-GGUF/<quant> pull URI, derive `<repo>-<quant>`
// so the on-disk filename matches the bare name a caller would round-trip
// through naming.ResolveBareNameToURI. Returns "" for anything else, which
// preserves the legacy "use the source URL's filename" behavior.
func deriveSaveAs(url, name string) string {
	if name != "" {
		return name
	}
	return naming.DeriveBareNameFromURI(url)
}

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

	// `name` is optional. When set, the downloaded file is saved as
	// `<name>.gguf` (or `<name>-NNNNN-of-MMMMM.gguf` for sharded models),
	// regardless of the source URL's filename. Lets callers control the
	// on-disk identity that /v1/models and /api/load see, so a user-typed
	// model name flows through cache + pull + load unchanged.
	//
	// When `name` is empty and the URL is a canonical hf:// pull URI
	// (`hf://<org>/<repo>-GGUF/<quant>`), we derive `<repo>-<quant>` and
	// use that as the destination basename — the actual .gguf file in the
	// HF repo often omits the repo-level variant tag (e.g. unsloth's
	// `Qwen3.6-35B-A3B-MTP-GGUF` ships `Qwen3.6-35B-A3B-Q8_0.gguf` with no
	// `MTP`), so without this step the on-disk name no longer matches what
	// the caller would resolve from the repo URI.
	var req struct {
		URL  string `json:"url"`
		Name string `json:"name"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", "failed to parse request body")

		return
	}

	if req.URL == "" {
		writeError(w, http.StatusBadRequest, "invalid_request", "url field is required")

		return
	}

	saveAs := deriveSaveAs(req.URL, req.Name)

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

		path, err := models.Download(dlURL, h.Store.Dir(), saveAs, progress)
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

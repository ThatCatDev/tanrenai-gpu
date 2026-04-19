package handlers

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/ThatCatDev/tanrenai-gpu/internal/training"
)

// FinetuneHandler handles fine-tuning API endpoints.
type FinetuneHandler struct {
	Manager *training.Manager
}

// Prepare handles POST /v1/finetune/prepare.
func (h *FinetuneHandler) Prepare(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20) // 1MB limit

	var req struct {
		BaseModel   string              `json:"base_model"`
		DatasetPath string              `json:"dataset_path"`
		SampleCount int                 `json:"sample_count"`
		Config      *training.RunConfig `json:"config,omitempty"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", "failed to parse request body: "+err.Error())

		return
	}

	if req.BaseModel == "" {
		writeError(w, http.StatusBadRequest, "invalid_request", "base_model is required")

		return
	}

	if req.DatasetPath == "" {
		writeError(w, http.StatusBadRequest, "invalid_request", "dataset_path is required")

		return
	}

	cfg := training.DefaultRunConfig()
	if req.Config != nil {
		cfg = *req.Config
	}

	run, err := h.Manager.Prepare(r.Context(), req.BaseModel, req.DatasetPath, req.SampleCount, cfg)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "finetune_error", err.Error())

		return
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(run)
}

// Train handles POST /v1/finetune/train.
func (h *FinetuneHandler) Train(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20) // 1MB limit

	var req struct {
		RunID string `json:"run_id"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", "failed to parse request body: "+err.Error())

		return
	}

	if req.RunID == "" {
		writeError(w, http.StatusBadRequest, "invalid_request", "run_id is required")

		return
	}

	if err := h.Manager.Train(r.Context(), req.RunID); err != nil {
		writeError(w, http.StatusInternalServerError, "finetune_error", err.Error())

		return
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]string{"status": "training", "run_id": req.RunID})
}

// Status handles GET /v1/finetune/status/{run_id}.
func (h *FinetuneHandler) Status(w http.ResponseWriter, r *http.Request) {
	parts := strings.Split(r.URL.Path, "/")
	if len(parts) < 5 {
		writeError(w, http.StatusBadRequest, "invalid_request", "run_id is required in path")

		return
	}
	runID := parts[len(parts)-1]

	run, err := h.Manager.Status(r.Context(), runID)
	if err != nil {
		writeError(w, http.StatusNotFound, "not_found", err.Error())

		return
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(run)
}

// Merge handles POST /v1/finetune/merge.
func (h *FinetuneHandler) Merge(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20) // 1MB limit

	var req struct {
		RunID      string `json:"run_id"`
		OutputName string `json:"output_name,omitempty"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", "failed to parse request body: "+err.Error())

		return
	}

	if req.RunID == "" {
		writeError(w, http.StatusBadRequest, "invalid_request", "run_id is required")

		return
	}

	outputPath, err := h.Manager.Merge(r.Context(), req.RunID, req.OutputName)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "finetune_error", err.Error())

		return
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]string{"status": "done", "output_path": outputPath})
}

// ListRuns handles GET /v1/finetune/runs.
func (h *FinetuneHandler) ListRuns(w http.ResponseWriter, r *http.Request) {
	runs, err := h.Manager.List(r.Context())
	if err != nil {
		writeError(w, http.StatusInternalServerError, "finetune_error", err.Error())

		return
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"runs": runs})
}

// DeleteRun handles DELETE /v1/finetune/runs/{run_id}.
func (h *FinetuneHandler) DeleteRun(w http.ResponseWriter, r *http.Request) {
	parts := strings.Split(r.URL.Path, "/")
	if len(parts) < 5 {
		writeError(w, http.StatusBadRequest, "invalid_request", "run_id is required in path")

		return
	}
	runID := parts[len(parts)-1]

	if err := h.Manager.Delete(r.Context(), runID); err != nil {
		writeError(w, http.StatusInternalServerError, "finetune_error", err.Error())

		return
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]string{"status": "deleted", "run_id": runID})
}

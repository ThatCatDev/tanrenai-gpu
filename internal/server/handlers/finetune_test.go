package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/ThatCatDev/tanrenai-gpu/internal/training"
)

// mockTrainingManager implements the subset of training.Manager needed by FinetuneHandler.
// We use a concrete struct mirroring the real Manager's API, injected via a thin wrapper.
type mockTrainingManager struct {
	trainErr  error
	statusRun *training.TrainingRun
	mergeErr  error
}

// prepareMockManager creates a training Manager backed by a fake sidecar server.
func prepareMockManager(t *testing.T, mock *mockTrainingManager) *training.Manager {
	t.Helper()
	// Build a sidecar server that returns mock responses
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/train":
			if mock.trainErr != nil {
				http.Error(w, mock.trainErr.Error(), http.StatusInternalServerError)

				return
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(training.TrainResponse{RunID: "test-run", Status: "started"})
		case "/merge":
			if mock.mergeErr != nil {
				http.Error(w, mock.mergeErr.Error(), http.StatusInternalServerError)

				return
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
		case "/convert":
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
		default:
			// Status endpoints like /status/{runID}
			if mock.statusRun != nil {
				w.Header().Set("Content-Type", "application/json")
				_ = json.NewEncoder(w).Encode(training.StatusResponse{
					Status: string(mock.statusRun.Status),
				})
			} else {
				http.Error(w, "not found", http.StatusNotFound)
			}
		}
	}))
	t.Cleanup(srv.Close)

	client := training.NewSidecarClient(srv.URL)
	tmp := t.TempDir()
	store := training.NewRunStoreAt(tmp)
	modelsDir := t.TempDir()

	return training.NewManagerWithStore(store, client, modelsDir)
}

// ---- Prepare handler ----

func TestFinetuneHandler_Prepare_BadJSON(t *testing.T) {
	m := prepareMockManager(t, &mockTrainingManager{})
	h := &FinetuneHandler{Manager: m}

	req := httptest.NewRequest(http.MethodPost, "/v1/finetune/prepare", bytes.NewReader([]byte("{invalid")))
	w := httptest.NewRecorder()
	h.Prepare(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", w.Code)
	}
	assertErrorType(t, w, "invalid_request")
}

func TestFinetuneHandler_Prepare_MissingBaseModel(t *testing.T) {
	m := prepareMockManager(t, &mockTrainingManager{})
	h := &FinetuneHandler{Manager: m}

	body := mustMarshal(t, map[string]any{
		"dataset_path": "/data/ds.jsonl",
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/finetune/prepare", bytes.NewReader(body))
	w := httptest.NewRecorder()
	h.Prepare(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", w.Code)
	}
	assertErrorType(t, w, "invalid_request")
}

func TestFinetuneHandler_Prepare_MissingDatasetPath(t *testing.T) {
	m := prepareMockManager(t, &mockTrainingManager{})
	h := &FinetuneHandler{Manager: m}

	body := mustMarshal(t, map[string]any{
		"base_model": "mymodel",
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/finetune/prepare", bytes.NewReader(body))
	w := httptest.NewRecorder()
	h.Prepare(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", w.Code)
	}
	assertErrorType(t, w, "invalid_request")
}

func TestFinetuneHandler_Prepare_Success(t *testing.T) {
	m := prepareMockManager(t, &mockTrainingManager{})
	h := &FinetuneHandler{Manager: m}

	body := mustMarshal(t, map[string]any{
		"base_model":   "mymodel",
		"dataset_path": "/data/ds.jsonl",
		"sample_count": 100,
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/finetune/prepare", bytes.NewReader(body))
	w := httptest.NewRecorder()
	h.Prepare(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200 (body: %s)", w.Code, w.Body.String())
	}

	var run training.TrainingRun
	if err := json.NewDecoder(w.Body).Decode(&run); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if run.BaseModel != "mymodel" {
		t.Errorf("BaseModel = %q, want %q", run.BaseModel, "mymodel")
	}
	if run.Status != training.StatusPending {
		t.Errorf("Status = %q, want %q", run.Status, training.StatusPending)
	}
}

func TestFinetuneHandler_Prepare_WithCustomConfig(t *testing.T) {
	m := prepareMockManager(t, &mockTrainingManager{})
	h := &FinetuneHandler{Manager: m}

	cfg := training.DefaultRunConfig()
	cfg.Epochs = 5
	body := mustMarshal(t, map[string]any{
		"base_model":   "mymodel",
		"dataset_path": "/data/ds.jsonl",
		"config":       cfg,
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/finetune/prepare", bytes.NewReader(body))
	w := httptest.NewRecorder()
	h.Prepare(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}
}

// ---- Train handler ----

func TestFinetuneHandler_Train_BadJSON(t *testing.T) {
	m := prepareMockManager(t, &mockTrainingManager{})
	h := &FinetuneHandler{Manager: m}

	req := httptest.NewRequest(http.MethodPost, "/v1/finetune/train", bytes.NewReader([]byte("{invalid")))
	w := httptest.NewRecorder()
	h.Train(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", w.Code)
	}
	assertErrorType(t, w, "invalid_request")
}

func TestFinetuneHandler_Train_MissingRunID(t *testing.T) {
	m := prepareMockManager(t, &mockTrainingManager{})
	h := &FinetuneHandler{Manager: m}

	body := mustMarshal(t, map[string]string{})
	req := httptest.NewRequest(http.MethodPost, "/v1/finetune/train", bytes.NewReader(body))
	w := httptest.NewRecorder()
	h.Train(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", w.Code)
	}
	assertErrorType(t, w, "invalid_request")
}

func TestFinetuneHandler_Train_RunNotFound(t *testing.T) {
	m := prepareMockManager(t, &mockTrainingManager{})
	h := &FinetuneHandler{Manager: m}

	body := mustMarshal(t, map[string]string{"run_id": "nonexistent"})
	req := httptest.NewRequest(http.MethodPost, "/v1/finetune/train", bytes.NewReader(body))
	w := httptest.NewRecorder()
	h.Train(w, req)

	if w.Code != http.StatusInternalServerError {
		t.Errorf("status = %d, want 500 (run not found)", w.Code)
	}
	assertErrorType(t, w, "finetune_error")
}

func TestFinetuneHandler_Train_Success(t *testing.T) {
	mock := &mockTrainingManager{}
	m := prepareMockManager(t, mock)

	// Create a pending run first
	run, err := m.Prepare(context.Background(), "mymodel", "/data/ds.jsonl", 10, training.DefaultRunConfig())
	if err != nil {
		t.Fatalf("Prepare: %v", err)
	}

	h := &FinetuneHandler{Manager: m}
	body := mustMarshal(t, map[string]string{"run_id": run.ID})
	req := httptest.NewRequest(http.MethodPost, "/v1/finetune/train", bytes.NewReader(body))
	w := httptest.NewRecorder()
	h.Train(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200 (body: %s)", w.Code, w.Body.String())
	}

	var resp map[string]string
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if resp["status"] != "training" {
		t.Errorf("status = %q, want %q", resp["status"], "training")
	}
	if resp["run_id"] != run.ID {
		t.Errorf("run_id = %q, want %q", resp["run_id"], run.ID)
	}
}

// ---- Status handler ----

func TestFinetuneHandler_Status_MissingRunID(t *testing.T) {
	m := prepareMockManager(t, &mockTrainingManager{})
	h := &FinetuneHandler{Manager: m}

	// Path too short
	req := httptest.NewRequest(http.MethodGet, "/v1/finetune/status", nil)
	w := httptest.NewRecorder()
	h.Status(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", w.Code)
	}
	assertErrorType(t, w, "invalid_request")
}

func TestFinetuneHandler_Status_NotFound(t *testing.T) {
	m := prepareMockManager(t, &mockTrainingManager{})
	h := &FinetuneHandler{Manager: m}

	req := httptest.NewRequest(http.MethodGet, "/v1/finetune/status/ghost-run", nil)
	w := httptest.NewRecorder()
	h.Status(w, req)

	if w.Code != http.StatusNotFound {
		t.Errorf("status = %d, want 404", w.Code)
	}
	assertErrorType(t, w, "not_found")
}

func TestFinetuneHandler_Status_Success(t *testing.T) {
	m := prepareMockManager(t, &mockTrainingManager{})

	// Create a pending run
	run, err := m.Prepare(context.Background(), "mymodel", "/data/ds.jsonl", 10, training.DefaultRunConfig())
	if err != nil {
		t.Fatalf("Prepare: %v", err)
	}

	h := &FinetuneHandler{Manager: m}
	req := httptest.NewRequest(http.MethodGet, "/v1/finetune/status/"+run.ID, nil)
	w := httptest.NewRecorder()
	h.Status(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200 (body: %s)", w.Code, w.Body.String())
	}

	var got training.TrainingRun
	if err := json.NewDecoder(w.Body).Decode(&got); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if got.ID != run.ID {
		t.Errorf("ID = %q, want %q", got.ID, run.ID)
	}
}

// ---- Merge handler ----

func TestFinetuneHandler_Merge_BadJSON(t *testing.T) {
	m := prepareMockManager(t, &mockTrainingManager{})
	h := &FinetuneHandler{Manager: m}

	req := httptest.NewRequest(http.MethodPost, "/v1/finetune/merge", bytes.NewReader([]byte("{invalid")))
	w := httptest.NewRecorder()
	h.Merge(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", w.Code)
	}
	assertErrorType(t, w, "invalid_request")
}

func TestFinetuneHandler_Merge_MissingRunID(t *testing.T) {
	m := prepareMockManager(t, &mockTrainingManager{})
	h := &FinetuneHandler{Manager: m}

	body := mustMarshal(t, map[string]string{})
	req := httptest.NewRequest(http.MethodPost, "/v1/finetune/merge", bytes.NewReader(body))
	w := httptest.NewRecorder()
	h.Merge(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", w.Code)
	}
	assertErrorType(t, w, "invalid_request")
}

func TestFinetuneHandler_Merge_RunNotFound(t *testing.T) {
	m := prepareMockManager(t, &mockTrainingManager{})
	h := &FinetuneHandler{Manager: m}

	body := mustMarshal(t, map[string]string{"run_id": "ghost"})
	req := httptest.NewRequest(http.MethodPost, "/v1/finetune/merge", bytes.NewReader(body))
	w := httptest.NewRecorder()
	h.Merge(w, req)

	if w.Code != http.StatusInternalServerError {
		t.Errorf("status = %d, want 500", w.Code)
	}
	assertErrorType(t, w, "finetune_error")
}

func TestFinetuneHandler_Merge_WrongStatus(t *testing.T) {
	m := prepareMockManager(t, &mockTrainingManager{})

	// Create a pending run (not merging)
	run, err := m.Prepare(context.Background(), "mymodel", "/data/ds.jsonl", 10, training.DefaultRunConfig())
	if err != nil {
		t.Fatalf("Prepare: %v", err)
	}

	h := &FinetuneHandler{Manager: m}
	body := mustMarshal(t, map[string]string{"run_id": run.ID})
	req := httptest.NewRequest(http.MethodPost, "/v1/finetune/merge", bytes.NewReader(body))
	w := httptest.NewRecorder()
	h.Merge(w, req)

	if w.Code != http.StatusInternalServerError {
		t.Errorf("status = %d, want 500 (wrong status)", w.Code)
	}
	assertErrorType(t, w, "finetune_error")
}

// ---- ListRuns handler ----

func TestFinetuneHandler_ListRuns_Empty(t *testing.T) {
	m := prepareMockManager(t, &mockTrainingManager{})
	h := &FinetuneHandler{Manager: m}

	req := httptest.NewRequest(http.MethodGet, "/v1/finetune/runs", nil)
	w := httptest.NewRecorder()
	h.ListRuns(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}

	var resp map[string]any
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if _, ok := resp["runs"]; !ok {
		t.Fatal("response missing 'runs' key")
	}
	// runs may be null/nil for empty list — both are valid empty
	switch v := resp["runs"].(type) {
	case nil:
		// ok — null JSON maps to nil
	case []interface{}:
		if len(v) != 0 {
			t.Errorf("len(runs) = %d, want 0", len(v))
		}
	default:
		t.Fatalf("runs is %T, want nil or []interface{}", resp["runs"])
	}
}

func TestFinetuneHandler_ListRuns_WithRuns(t *testing.T) {
	m := prepareMockManager(t, &mockTrainingManager{})

	// Create runs with slightly different timestamps to ensure unique IDs
	run1, err := m.Prepare(context.Background(), "model1", "/data/d1.jsonl", 10, training.DefaultRunConfig())
	if err != nil {
		t.Fatalf("Prepare run1: %v", err)
	}
	_ = run1

	h := &FinetuneHandler{Manager: m}

	req := httptest.NewRequest(http.MethodGet, "/v1/finetune/runs", nil)
	w := httptest.NewRecorder()
	h.ListRuns(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}

	var resp map[string]any
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	arr, ok := resp["runs"].([]interface{})
	if !ok {
		t.Fatalf("runs should be an array, got %T: %v", resp["runs"], resp["runs"])
	}
	if len(arr) != 1 {
		t.Errorf("len(runs) = %d, want 1", len(arr))
	}
}

// ---- DeleteRun handler ----

func TestFinetuneHandler_DeleteRun_MissingRunID(t *testing.T) {
	m := prepareMockManager(t, &mockTrainingManager{})
	h := &FinetuneHandler{Manager: m}

	req := httptest.NewRequest(http.MethodDelete, "/v1/finetune/runs", nil)
	w := httptest.NewRecorder()
	h.DeleteRun(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", w.Code)
	}
	assertErrorType(t, w, "invalid_request")
}

func TestFinetuneHandler_DeleteRun_NotFound(t *testing.T) {
	m := prepareMockManager(t, &mockTrainingManager{})
	h := &FinetuneHandler{Manager: m}

	req := httptest.NewRequest(http.MethodDelete, "/v1/finetune/runs/ghost-run", nil)
	w := httptest.NewRecorder()
	h.DeleteRun(w, req)

	if w.Code != http.StatusInternalServerError {
		t.Errorf("status = %d, want 500 (not found)", w.Code)
	}
	assertErrorType(t, w, "finetune_error")
}

func TestFinetuneHandler_DeleteRun_Success(t *testing.T) {
	m := prepareMockManager(t, &mockTrainingManager{})

	run, err := m.Prepare(context.Background(), "mymodel", "/data/ds.jsonl", 10, training.DefaultRunConfig())
	if err != nil {
		t.Fatalf("Prepare: %v", err)
	}

	h := &FinetuneHandler{Manager: m}
	req := httptest.NewRequest(http.MethodDelete, "/v1/finetune/runs/"+run.ID, nil)
	w := httptest.NewRecorder()
	h.DeleteRun(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200 (body: %s)", w.Code, w.Body.String())
	}

	var resp map[string]string
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if resp["status"] != "deleted" {
		t.Errorf("status = %q, want %q", resp["status"], "deleted")
	}
	if resp["run_id"] != run.ID {
		t.Errorf("run_id = %q, want %q", resp["run_id"], run.ID)
	}
}

// Verify time is used to avoid import issues.
var _ = time.Now

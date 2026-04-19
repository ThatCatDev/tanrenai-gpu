package training

import (
	"context"
	"encoding/json"
	"net/http"
	"testing"
	"time"
)

// TestManager_Merge_StatusTraining_SidecarDone verifies that a run in StatusTraining
// can be merged when the sidecar reports "done".
func TestManager_Merge_StatusTraining_SidecarDone(t *testing.T) {
	handlers := map[string]http.HandlerFunc{
		"/status/run-trn-merge": func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(StatusResponse{Status: "done"})
		},
		"/merge": func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
		},
		"/convert": func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
		},
	}
	client, stop := newTestSidecar(t, handlers)
	defer stop()

	m, store, _ := newTestManager(t, client)

	now := time.Now()
	run := &TrainingRun{
		ID:        "run-trn-merge",
		BaseModel: "mymodel",
		Status:    StatusTraining,
		CreatedAt: now,
		UpdatedAt: now,
	}
	if err := store.Save(run); err != nil {
		t.Fatal(err)
	}

	_, err := m.Merge(context.Background(), "run-trn-merge", "out.gguf")
	if err != nil {
		t.Fatalf("Merge with StatusTraining+sidecar done: %v", err)
	}
}

// TestManager_Merge_StatusTraining_SidecarNotDone verifies that a run in StatusTraining
// is rejected when the sidecar reports training is not done.
func TestManager_Merge_StatusTraining_SidecarNotDone(t *testing.T) {
	handlers := map[string]http.HandlerFunc{
		"/status/run-trn-notdone": func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(StatusResponse{Status: "training"})
		},
	}
	client, stop := newTestSidecar(t, handlers)
	defer stop()

	m, store, _ := newTestManager(t, client)

	now := time.Now()
	run := &TrainingRun{
		ID:        "run-trn-notdone",
		BaseModel: "mymodel",
		Status:    StatusTraining,
		CreatedAt: now,
		UpdatedAt: now,
	}
	if err := store.Save(run); err != nil {
		t.Fatal(err)
	}

	_, err := m.Merge(context.Background(), "run-trn-notdone", "")
	if err == nil {
		t.Fatal("expected error when sidecar reports training not done")
	}
}

// TestManager_Merge_StatusDone_Success verifies that a run already in StatusDone can be re-merged.
func TestManager_Merge_StatusDone_Success(t *testing.T) {
	handlers := map[string]http.HandlerFunc{
		"/merge": func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte("{}"))
		},
		"/convert": func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte("{}"))
		},
	}
	client, stop := newTestSidecar(t, handlers)
	defer stop()

	m, store, _ := newTestManager(t, client)

	now := time.Now()
	run := &TrainingRun{
		ID: "run-done-remerge", BaseModel: "mymodel", Status: StatusDone,
		CreatedAt: now, UpdatedAt: now,
	}
	if err := store.Save(run); err != nil {
		t.Fatal(err)
	}

	_, err := m.Merge(context.Background(), "run-done-remerge", "output2.gguf")
	if err != nil {
		t.Fatalf("Merge with StatusDone: %v", err)
	}
}

// TestManager_Merge_ConvertError_MarksRunFailed verifies that a convert error marks run as failed.
func TestManager_Merge_ConvertError_MarksRunFailed(t *testing.T) {
	handlers := map[string]http.HandlerFunc{
		"/merge": func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte("{}"))
		},
		"/convert": func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte("convert failed"))
		},
	}
	client, stop := newTestSidecar(t, handlers)
	defer stop()

	m, store, _ := newTestManager(t, client)

	now := time.Now()
	run := &TrainingRun{
		ID: "run-conv-err", BaseModel: "m", Status: StatusMerging,
		CreatedAt: now, UpdatedAt: now,
	}
	if err := store.Save(run); err != nil {
		t.Fatal(err)
	}

	_, err := m.Merge(context.Background(), "run-conv-err", "")
	if err == nil {
		t.Fatal("expected error from convert failure")
	}

	loaded, _ := store.Load("run-conv-err")
	if loaded.Status != StatusFailed {
		t.Errorf("Status = %q, want %q", loaded.Status, StatusFailed)
	}
}

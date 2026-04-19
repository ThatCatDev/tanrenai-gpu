package training

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// newTestSidecar creates a fake sidecar server and a SidecarClient pointing at it.
// handlers maps URL path -> handler func.
func newTestSidecar(t *testing.T, handlers map[string]http.HandlerFunc) (*SidecarClient, func()) {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if h, ok := handlers[r.URL.Path]; ok {
			h(w, r)

			return
		}
		http.Error(w, "not found", http.StatusNotFound)
	}))
	client := NewSidecarClient(srv.URL)

	return client, srv.Close
}

func newTestManager(t *testing.T, client *SidecarClient) (*Manager, *RunStore, string) {
	t.Helper()
	tmp := t.TempDir()
	store := NewRunStoreAt(tmp)
	modelsDir := t.TempDir()
	m := NewManagerWithStore(store, client, modelsDir)

	return m, store, modelsDir
}

// ---- Prepare ----

func TestManager_Prepare_Creates_PendingRun(t *testing.T) {
	client, stop := newTestSidecar(t, nil)
	defer stop()

	m, store, _ := newTestManager(t, client)

	run, err := m.Prepare(context.Background(), "mymodel", "/data/dataset.jsonl", 100, DefaultRunConfig())
	if err != nil {
		t.Fatalf("Prepare: %v", err)
	}

	if run.Status != StatusPending {
		t.Errorf("Status = %q, want %q", run.Status, StatusPending)
	}
	if run.BaseModel != "mymodel" {
		t.Errorf("BaseModel = %q, want %q", run.BaseModel, "mymodel")
	}
	if run.DatasetPath != "/data/dataset.jsonl" {
		t.Errorf("DatasetPath = %q, want %q", run.DatasetPath, "/data/dataset.jsonl")
	}
	if run.Metrics.SamplesUsed != 100 {
		t.Errorf("SamplesUsed = %d, want 100", run.Metrics.SamplesUsed)
	}
	if run.ID == "" {
		t.Error("run.ID is empty")
	}

	// Run should be persisted to store
	loaded, err := store.Load(run.ID)
	if err != nil {
		t.Fatalf("Load from store: %v", err)
	}
	if loaded.ID != run.ID {
		t.Errorf("persisted ID = %q, want %q", loaded.ID, run.ID)
	}
}

// ---- List / Delete via Manager ----

func TestManager_List_Empty(t *testing.T) {
	client, stop := newTestSidecar(t, nil)
	defer stop()

	m, _, _ := newTestManager(t, client)

	runs, err := m.List(context.Background())
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(runs) != 0 {
		t.Errorf("List() = %d runs, want 0", len(runs))
	}
}

func TestManager_Delete_Existing(t *testing.T) {
	client, stop := newTestSidecar(t, nil)
	defer stop()

	m, store, _ := newTestManager(t, client)

	// Create a run first
	run, err := m.Prepare(context.Background(), "mymodel", "", 0, DefaultRunConfig())
	if err != nil {
		t.Fatalf("Prepare: %v", err)
	}

	// Delete it
	if err := m.Delete(context.Background(), run.ID); err != nil {
		t.Fatalf("Delete: %v", err)
	}

	// Should no longer exist in store
	_, err = store.Load(run.ID)
	if err == nil {
		t.Fatal("Load after Delete: expected error, got nil")
	}
}

func TestManager_Delete_NotFound(t *testing.T) {
	client, stop := newTestSidecar(t, nil)
	defer stop()

	m, _, _ := newTestManager(t, client)

	err := m.Delete(context.Background(), "nonexistent-run")
	if err == nil {
		t.Fatal("Delete nonexistent run: expected error, got nil")
	}
}

// ---- Train ----

func TestManager_Train_NotFound(t *testing.T) {
	client, stop := newTestSidecar(t, nil)
	defer stop()

	m, _, _ := newTestManager(t, client)

	err := m.Train(context.Background(), "ghost-run")
	if err == nil {
		t.Fatal("Train on nonexistent run: expected error, got nil")
	}
}

func TestManager_Train_WrongStatus(t *testing.T) {
	client, stop := newTestSidecar(t, nil)
	defer stop()

	m, store, _ := newTestManager(t, client)

	// Create a run with "training" status (not pending)
	now := time.Now()
	run := &TrainingRun{
		ID: "run-training", BaseModel: "m", Status: StatusTraining,
		CreatedAt: now, UpdatedAt: now,
	}
	if err := store.Save(run); err != nil {
		t.Fatal(err)
	}

	err := m.Train(context.Background(), "run-training")
	if err == nil {
		t.Fatal("Train with wrong status: expected error, got nil")
	}
}

func TestManager_Train_SidecarError_MarksRunFailed(t *testing.T) {
	handlers := map[string]http.HandlerFunc{
		"/train": func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte("training failed"))
		},
	}
	client, stop := newTestSidecar(t, handlers)
	defer stop()

	m, store, _ := newTestManager(t, client)

	// Create a pending run
	run, err := m.Prepare(context.Background(), "mymodel", "/data/d.jsonl", 10, DefaultRunConfig())
	if err != nil {
		t.Fatalf("Prepare: %v", err)
	}

	// Train should fail
	if err := m.Train(context.Background(), run.ID); err == nil {
		t.Fatal("Train with sidecar error: expected error, got nil")
	}

	// Run should be marked as failed
	loaded, err := store.Load(run.ID)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if loaded.Status != StatusFailed {
		t.Errorf("Status = %q, want %q", loaded.Status, StatusFailed)
	}
	if loaded.Error == "" {
		t.Error("Error field should be set on failed run")
	}
}

func TestManager_Train_Success(t *testing.T) {
	handlers := map[string]http.HandlerFunc{
		"/train": func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(TrainResponse{RunID: "run-ok", Status: "started"})
		},
	}
	client, stop := newTestSidecar(t, handlers)
	defer stop()

	m, store, _ := newTestManager(t, client)

	run, err := m.Prepare(context.Background(), "mymodel", "/data/d.jsonl", 10, DefaultRunConfig())
	if err != nil {
		t.Fatalf("Prepare: %v", err)
	}

	if err := m.Train(context.Background(), run.ID); err != nil {
		t.Fatalf("Train: %v", err)
	}

	loaded, err := store.Load(run.ID)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if loaded.Status != StatusTraining {
		t.Errorf("Status = %q, want %q", loaded.Status, StatusTraining)
	}
}

// ---- Status ----

func TestManager_Status_NotFound(t *testing.T) {
	client, stop := newTestSidecar(t, nil)
	defer stop()

	m, _, _ := newTestManager(t, client)

	_, err := m.Status(context.Background(), "ghost")
	if err == nil {
		t.Fatal("Status on nonexistent run: expected error, got nil")
	}
}

func TestManager_Status_PendingRun_NoSidecarCall(t *testing.T) {
	sidecarCalled := false
	handlers := map[string]http.HandlerFunc{
		"/status/run-pend": func(w http.ResponseWriter, r *http.Request) {
			sidecarCalled = true
			w.WriteHeader(http.StatusOK)
		},
	}
	client, stop := newTestSidecar(t, handlers)
	defer stop()

	m, store, _ := newTestManager(t, client)

	now := time.Now()
	run := &TrainingRun{ID: "run-pend", BaseModel: "m", Status: StatusPending, CreatedAt: now, UpdatedAt: now}
	if err := store.Save(run); err != nil {
		t.Fatal(err)
	}

	got, err := m.Status(context.Background(), "run-pend")
	if err != nil {
		t.Fatalf("Status: %v", err)
	}
	if got.Status != StatusPending {
		t.Errorf("Status = %q, want %q", got.Status, StatusPending)
	}
	if sidecarCalled {
		t.Error("sidecar should not be called for pending run")
	}
}

func TestManager_Status_TrainingRun_QueriesSidecar(t *testing.T) {
	handlers := map[string]http.HandlerFunc{
		"/status/run-trn": func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(StatusResponse{
				Status:  "training",
				Metrics: RunMetrics{Progress: 0.5},
			})
		},
	}
	client, stop := newTestSidecar(t, handlers)
	defer stop()

	m, store, _ := newTestManager(t, client)

	now := time.Now()
	run := &TrainingRun{ID: "run-trn", BaseModel: "m", Status: StatusTraining, CreatedAt: now, UpdatedAt: now}
	if err := store.Save(run); err != nil {
		t.Fatal(err)
	}

	got, err := m.Status(context.Background(), "run-trn")
	if err != nil {
		t.Fatalf("Status: %v", err)
	}
	if got.Metrics.Progress != 0.5 {
		t.Errorf("Metrics.Progress = %v, want 0.5", got.Metrics.Progress)
	}
}

func TestManager_Status_TrainingDone_TransitionsToMerging(t *testing.T) {
	handlers := map[string]http.HandlerFunc{
		"/status/run-done": func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(StatusResponse{Status: "done"})
		},
	}
	client, stop := newTestSidecar(t, handlers)
	defer stop()

	m, store, _ := newTestManager(t, client)

	now := time.Now()
	run := &TrainingRun{ID: "run-done", BaseModel: "m", Status: StatusTraining, CreatedAt: now, UpdatedAt: now}
	if err := store.Save(run); err != nil {
		t.Fatal(err)
	}

	got, err := m.Status(context.Background(), "run-done")
	if err != nil {
		t.Fatalf("Status: %v", err)
	}
	if got.Status != StatusMerging {
		t.Errorf("Status = %q, want %q", got.Status, StatusMerging)
	}

	// Check persisted status too
	loaded, _ := store.Load("run-done")
	if loaded.Status != StatusMerging {
		t.Errorf("persisted Status = %q, want %q", loaded.Status, StatusMerging)
	}
}

func TestManager_Status_TrainingFailed_TransitionsToFailed(t *testing.T) {
	handlers := map[string]http.HandlerFunc{
		"/status/run-fail": func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(StatusResponse{Status: "failed", Error: "OOM"})
		},
	}
	client, stop := newTestSidecar(t, handlers)
	defer stop()

	m, store, _ := newTestManager(t, client)

	now := time.Now()
	run := &TrainingRun{ID: "run-fail", BaseModel: "m", Status: StatusTraining, CreatedAt: now, UpdatedAt: now}
	if err := store.Save(run); err != nil {
		t.Fatal(err)
	}

	got, err := m.Status(context.Background(), "run-fail")
	if err != nil {
		t.Fatalf("Status: %v", err)
	}
	if got.Status != StatusFailed {
		t.Errorf("Status = %q, want %q", got.Status, StatusFailed)
	}
	if got.Error != "OOM" {
		t.Errorf("Error = %q, want %q", got.Error, "OOM")
	}
}

// ---- Merge ----

func TestManager_Merge_WrongStatus(t *testing.T) {
	client, stop := newTestSidecar(t, nil)
	defer stop()

	m, store, _ := newTestManager(t, client)

	now := time.Now()
	run := &TrainingRun{ID: "run-pending", BaseModel: "m", Status: StatusPending, CreatedAt: now, UpdatedAt: now}
	if err := store.Save(run); err != nil {
		t.Fatal(err)
	}

	_, err := m.Merge(context.Background(), "run-pending", "")
	if err == nil {
		t.Fatal("Merge with wrong status: expected error, got nil")
	}
}

func TestManager_Merge_NotFound(t *testing.T) {
	client, stop := newTestSidecar(t, nil)
	defer stop()

	m, _, _ := newTestManager(t, client)

	_, err := m.Merge(context.Background(), "ghost-run", "")
	if err == nil {
		t.Fatal("Merge on nonexistent run: expected error, got nil")
	}
}

func TestManager_Merge_Success(t *testing.T) {
	handlers := map[string]http.HandlerFunc{
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

	m, store, modelsDir := newTestManager(t, client)

	now := time.Now()
	run := &TrainingRun{
		ID: "run-merge", BaseModel: "mymodel", Status: StatusMerging,
		CreatedAt: now, UpdatedAt: now, AdapterDir: "/tmp/adapter",
	}
	if err := store.Save(run); err != nil {
		t.Fatal(err)
	}

	ggufPath, err := m.Merge(context.Background(), "run-merge", "output.gguf")
	if err != nil {
		t.Fatalf("Merge: %v", err)
	}
	if ggufPath == "" {
		t.Error("Merge returned empty gguf path")
	}

	// Should use the provided output name and modelsDir
	expectedPath := modelsDir + "/output.gguf"
	if ggufPath != expectedPath {
		t.Errorf("ggufPath = %q, want %q", ggufPath, expectedPath)
	}

	// Run should be marked done
	loaded, _ := store.Load("run-merge")
	if loaded.Status != StatusDone {
		t.Errorf("Status = %q, want %q", loaded.Status, StatusDone)
	}
	if loaded.OutputModel != ggufPath {
		t.Errorf("OutputModel = %q, want %q", loaded.OutputModel, ggufPath)
	}
}

func TestManager_Merge_DefaultOutputName(t *testing.T) {
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
		ID: "run-auto-name", BaseModel: "/models/mymodel.gguf", Status: StatusMerging,
		CreatedAt: now, UpdatedAt: now,
	}
	if err := store.Save(run); err != nil {
		t.Fatal(err)
	}

	ggufPath, err := m.Merge(context.Background(), "run-auto-name", "")
	if err != nil {
		t.Fatalf("Merge: %v", err)
	}
	if ggufPath == "" {
		t.Error("Merge returned empty path with auto-generated name")
	}
}

func TestManager_Merge_MergeError_MarksRunFailed(t *testing.T) {
	handlers := map[string]http.HandlerFunc{
		"/merge": func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte("merge failed"))
		},
	}
	client, stop := newTestSidecar(t, handlers)
	defer stop()

	m, store, _ := newTestManager(t, client)

	now := time.Now()
	run := &TrainingRun{
		ID: "run-merge-err", BaseModel: "m", Status: StatusMerging,
		CreatedAt: now, UpdatedAt: now,
	}
	if err := store.Save(run); err != nil {
		t.Fatal(err)
	}

	_, err := m.Merge(context.Background(), "run-merge-err", "")
	if err == nil {
		t.Fatal("Merge with sidecar error: expected error, got nil")
	}

	loaded, _ := store.Load("run-merge-err")
	if loaded.Status != StatusFailed {
		t.Errorf("Status = %q, want %q", loaded.Status, StatusFailed)
	}
}

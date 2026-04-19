package training

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestNewSidecarClient(t *testing.T) {
	c := NewSidecarClient("http://localhost:9999")
	if c == nil {
		t.Fatal("NewSidecarClient returned nil")
	}
}

func TestSidecarClient_Train_Success(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/train" {
			http.Error(w, "unexpected", http.StatusBadRequest)

			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(TrainResponse{RunID: "run-123", Status: "started"})
	}))
	defer srv.Close()

	c := NewSidecarClient(srv.URL)
	runID, err := c.Train(context.Background(), TrainRequest{
		DatasetPath: "/data/d.jsonl", BaseModelPath: "model",
		OutputDir: "/out", Epochs: 3, LearningRate: 2e-4,
		LoraRank: 16, LoraAlpha: 32, BatchSize: 2,
	})
	if err != nil {
		t.Fatalf("Train: %v", err)
	}
	if runID != "run-123" {
		t.Errorf("runID = %q, want %q", runID, "run-123")
	}
}

func TestSidecarClient_Train_HTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte("internal error"))
	}))
	defer srv.Close()

	c := NewSidecarClient(srv.URL)
	_, err := c.Train(context.Background(), TrainRequest{})
	if err == nil {
		t.Fatal("expected error for HTTP 500, got nil")
	}
}

func TestSidecarClient_Status_Success(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/status/run-abc" {
			http.Error(w, "unexpected", http.StatusBadRequest)

			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(StatusResponse{
			Status:  "training",
			Metrics: RunMetrics{Progress: 0.75},
		})
	}))
	defer srv.Close()

	c := NewSidecarClient(srv.URL)
	resp, err := c.Status(context.Background(), "run-abc")
	if err != nil {
		t.Fatalf("Status: %v", err)
	}
	if resp.Status != "training" {
		t.Errorf("Status = %q, want %q", resp.Status, "training")
	}
	if resp.Metrics.Progress != 0.75 {
		t.Errorf("Progress = %v, want 0.75", resp.Metrics.Progress)
	}
}

func TestSidecarClient_Status_NotFound(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		_, _ = w.Write([]byte("not found"))
	}))
	defer srv.Close()

	c := NewSidecarClient(srv.URL)
	_, err := c.Status(context.Background(), "missing-run")
	if err == nil {
		t.Fatal("expected error for 404, got nil")
	}
}

func TestSidecarClient_Merge_Success(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/merge" {
			http.Error(w, "unexpected", http.StatusBadRequest)

			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"status":"ok"}`))
	}))
	defer srv.Close()

	c := NewSidecarClient(srv.URL)
	err := c.Merge(context.Background(), MergeRequest{
		BaseModelPath: "/models/base.gguf",
		AdapterDir:    "/adapters/lora",
		OutputPath:    "/merged",
	})
	if err != nil {
		t.Fatalf("Merge: %v", err)
	}
}

func TestSidecarClient_Merge_HTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte("bad request"))
	}))
	defer srv.Close()

	c := NewSidecarClient(srv.URL)
	err := c.Merge(context.Background(), MergeRequest{})
	if err == nil {
		t.Fatal("expected error for HTTP 400, got nil")
	}
}

func TestSidecarClient_Convert_Success(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/convert" {
			http.Error(w, "unexpected", http.StatusBadRequest)

			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"status":"ok"}`))
	}))
	defer srv.Close()

	c := NewSidecarClient(srv.URL)
	err := c.Convert(context.Background(), ConvertRequest{
		ModelDir:     "/merged",
		OutputPath:   "/models/out.gguf",
		Quantization: "Q4_K_M",
	})
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}
}

func TestSidecarClient_Convert_HTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte("conversion failed"))
	}))
	defer srv.Close()

	c := NewSidecarClient(srv.URL)
	err := c.Convert(context.Background(), ConvertRequest{})
	if err == nil {
		t.Fatal("expected error for HTTP 500, got nil")
	}
}

func TestSidecarClient_InvalidURL(t *testing.T) {
	// An invalid URL should produce an error on request creation
	c := NewSidecarClient("://invalid-url")
	_, err := c.Train(context.Background(), TrainRequest{})
	if err == nil {
		t.Fatal("expected error for invalid URL, got nil")
	}
}

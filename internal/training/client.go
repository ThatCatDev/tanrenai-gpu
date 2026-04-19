package training

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// SidecarClient is an HTTP client for the Python training sidecar.
type SidecarClient struct {
	baseURL string
	http    *http.Client
}

// NewSidecarClient creates a client pointing at the given sidecar URL.
func NewSidecarClient(baseURL string) *SidecarClient {
	return &SidecarClient{
		baseURL: baseURL,
		http:    &http.Client{},
	}
}

// TrainRequest is the request body for POST /train.
type TrainRequest struct {
	DatasetPath   string  `json:"dataset_path"`
	BaseModelPath string  `json:"base_model_path"`
	OutputDir     string  `json:"output_dir"`
	RunID         string  `json:"run_id,omitempty"`
	Epochs        int     `json:"epochs"`
	LearningRate  float64 `json:"learning_rate"`
	LoraRank      int     `json:"lora_rank"`
	LoraAlpha     int     `json:"lora_alpha"`
	BatchSize     int     `json:"batch_size"`
}

// TrainResponse is returned from POST /train.
type TrainResponse struct {
	RunID  string `json:"run_id"`
	Status string `json:"status"`
}

// StatusResponse is returned from GET /status/{run_id}.
type StatusResponse struct {
	Status  string     `json:"status"`
	Metrics RunMetrics `json:"metrics,omitempty"`
	Error   string     `json:"error,omitempty"`
}

// MergeRequest is the request body for POST /merge.
type MergeRequest struct {
	BaseModelPath string `json:"base_model_path"`
	AdapterDir    string `json:"adapter_dir"`
	OutputPath    string `json:"output_path"`
}

// ConvertRequest is the request body for POST /convert.
type ConvertRequest struct {
	ModelDir     string `json:"model_dir"`
	OutputPath   string `json:"output_path"`
	Quantization string `json:"quantization"`
}

// Train starts a training job on the sidecar.
func (c *SidecarClient) Train(ctx context.Context, req TrainRequest) (string, error) {
	var resp TrainResponse
	if err := c.post(ctx, "/train", req, &resp); err != nil {
		return "", err
	}

	return resp.RunID, nil
}

// Status returns the current status of a training run.
func (c *SidecarClient) Status(ctx context.Context, runID string) (StatusResponse, error) {
	var resp StatusResponse
	if err := c.get(ctx, "/status/"+runID, &resp); err != nil {
		return StatusResponse{}, err
	}

	return resp, nil
}

// Merge merges a LoRA adapter into a base model.
func (c *SidecarClient) Merge(ctx context.Context, req MergeRequest) error {
	var resp map[string]any

	return c.post(ctx, "/merge", req, &resp)
}

// Convert converts a merged model to GGUF format.
func (c *SidecarClient) Convert(ctx context.Context, req ConvertRequest) error {
	var resp map[string]any

	return c.post(ctx, "/convert", req, &resp)
}

func (c *SidecarClient) post(ctx context.Context, path string, body any, result any) error {
	data, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+path, bytes.NewReader(data))
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(req)
	if err != nil {
		return fmt.Errorf("send request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode >= 400 {
		respBody, _ := io.ReadAll(resp.Body)

		return fmt.Errorf("sidecar returned %d: %s", resp.StatusCode, string(respBody))
	}

	if result != nil {
		if err := json.NewDecoder(resp.Body).Decode(result); err != nil {
			return fmt.Errorf("decode response: %w", err)
		}
	}

	return nil
}

func (c *SidecarClient) get(ctx context.Context, path string, result any) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+path, nil)
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}

	resp, err := c.http.Do(req)
	if err != nil {
		return fmt.Errorf("send request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode >= 400 {
		respBody, _ := io.ReadAll(resp.Body)

		return fmt.Errorf("sidecar returned %d: %s", resp.StatusCode, string(respBody))
	}

	if result != nil {
		if err := json.NewDecoder(resp.Body).Decode(result); err != nil {
			return fmt.Errorf("decode response: %w", err)
		}
	}

	return nil
}

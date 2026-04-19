package training

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/ThatCatDev/tanrenai-gpu/internal/config"
)

// Manager orchestrates the fine-tuning pipeline: train, status, merge.
// Dataset preparation (export from memory) happens on the backend;
// the GPU server receives pre-exported dataset paths.
type Manager struct {
	runStore  *RunStore
	client    *SidecarClient
	modelsDir string
}

// NewManager creates a training Manager.
func NewManager(client *SidecarClient) *Manager {
	return &Manager{
		runStore:  NewRunStore(),
		client:    client,
		modelsDir: config.ModelsDir(),
	}
}

// NewManagerWithStore creates a Manager with a custom RunStore (for testing).
func NewManagerWithStore(store *RunStore, client *SidecarClient, modelsDir string) *Manager {
	return &Manager{
		runStore:  store,
		client:    client,
		modelsDir: modelsDir,
	}
}

// Prepare creates a pending training run with a pre-exported dataset.
// The dataset file is provided by the caller (backend exports from memory).
func (m *Manager) Prepare(ctx context.Context, baseModel, datasetPath string, sampleCount int, cfg RunConfig) (*TrainingRun, error) {
	runID := fmt.Sprintf("%s-%d", "ft", time.Now().Unix())

	now := time.Now()
	run := &TrainingRun{
		ID:          runID,
		BaseModel:   baseModel,
		Status:      StatusPending,
		CreatedAt:   now,
		UpdatedAt:   now,
		Config:      cfg,
		DatasetPath: datasetPath,
		Metrics: RunMetrics{
			SamplesUsed: sampleCount,
		},
	}

	if err := m.runStore.Save(run); err != nil {
		return nil, fmt.Errorf("save run: %w", err)
	}

	return run, nil
}

// Train starts a training job for the given run.
func (m *Manager) Train(ctx context.Context, runID string) error {
	run, err := m.runStore.Load(runID)
	if err != nil {
		return fmt.Errorf("load run: %w", err)
	}

	if run.Status != StatusPending {
		return fmt.Errorf("run %s has status %s, expected %s", runID, run.Status, StatusPending)
	}

	outputDir := filepath.Join(config.TrainingRunsDir(), runID)

	_, err = m.client.Train(ctx, TrainRequest{
		DatasetPath:   run.DatasetPath,
		BaseModelPath: run.BaseModel,
		OutputDir:     outputDir,
		RunID:         runID,
		Epochs:        run.Config.Epochs,
		LearningRate:  run.Config.LearningRate,
		LoraRank:      run.Config.LoraRank,
		LoraAlpha:     run.Config.LoraAlpha,
		BatchSize:     run.Config.BatchSize,
	})
	if err != nil {
		run.Status = StatusFailed
		run.Error = err.Error()
		run.UpdatedAt = time.Now()
		_ = m.runStore.Save(run)

		return fmt.Errorf("start training: %w", err)
	}

	run.Status = StatusTraining
	run.AdapterDir = filepath.Join(outputDir, "adapter")
	run.UpdatedAt = time.Now()
	_ = m.runStore.Save(run)

	return nil
}

// Status returns the current state of a training run, querying the sidecar
// for live metrics if training is in progress.
func (m *Manager) Status(ctx context.Context, runID string) (*TrainingRun, error) {
	run, err := m.runStore.Load(runID)
	if err != nil {
		return nil, fmt.Errorf("load run: %w", err)
	}

	if run.Status == StatusTraining {
		status, err := m.client.Status(ctx, runID)
		if err == nil {
			run.Metrics = status.Metrics
			switch status.Status {
			case "done":
				run.Status = StatusMerging // ready for merge
				run.UpdatedAt = time.Now()
				_ = m.runStore.Save(run)
			case "failed":
				run.Status = StatusFailed
				run.Error = status.Error
				run.UpdatedAt = time.Now()
				_ = m.runStore.Save(run)
			}
		}
	}

	return run, nil
}

// Merge merges the LoRA adapter into the base model and converts to GGUF.
func (m *Manager) Merge(ctx context.Context, runID string, outputName string) (string, error) {
	run, err := m.runStore.Load(runID)
	if err != nil {
		return "", fmt.Errorf("load run: %w", err)
	}

	if run.Status != StatusMerging && run.Status != StatusDone {
		if run.Status == StatusTraining {
			status, err := m.client.Status(ctx, runID)
			if err != nil || status.Status != "done" {
				return "", fmt.Errorf("run %s not ready for merge (status: %s)", runID, run.Status)
			}
			run.Status = StatusMerging
		} else {
			return "", fmt.Errorf("run %s has status %s, expected merging or done", runID, run.Status)
		}
	}

	run.Status = StatusMerging
	run.UpdatedAt = time.Now()
	_ = m.runStore.Save(run)

	// Merge adapter into base model
	mergedDir := filepath.Join(config.TrainingRunsDir(), runID, "merged")
	if err := m.client.Merge(ctx, MergeRequest{
		BaseModelPath: run.BaseModel,
		AdapterDir:    run.AdapterDir,
		OutputPath:    mergedDir,
	}); err != nil {
		run.Status = StatusFailed
		run.Error = fmt.Sprintf("merge failed: %v", err)
		run.UpdatedAt = time.Now()
		_ = m.runStore.Save(run)

		return "", fmt.Errorf("merge: %w", err)
	}

	// Convert to GGUF
	if outputName == "" {
		outputName = fmt.Sprintf("%s-ft-%s.gguf", filepath.Base(run.BaseModel), runID)
	}
	ggufPath := filepath.Join(m.modelsDir, outputName)

	if err := m.client.Convert(ctx, ConvertRequest{
		ModelDir:     mergedDir,
		OutputPath:   ggufPath,
		Quantization: "Q4_K_M",
	}); err != nil {
		run.Status = StatusFailed
		run.Error = fmt.Sprintf("convert failed: %v", err)
		run.UpdatedAt = time.Now()
		_ = m.runStore.Save(run)

		return "", fmt.Errorf("convert: %w", err)
	}

	run.Status = StatusDone
	run.OutputModel = ggufPath
	run.UpdatedAt = time.Now()
	_ = m.runStore.Save(run)

	return ggufPath, nil
}

// List returns all training runs.
func (m *Manager) List(ctx context.Context) ([]*TrainingRun, error) {
	return m.runStore.List()
}

// Delete removes a training run and its artifacts.
func (m *Manager) Delete(ctx context.Context, runID string) error {
	run, err := m.runStore.Load(runID)
	if err != nil {
		return fmt.Errorf("load run: %w", err)
	}

	// Remove the dataset file if it exists
	if run.DatasetPath != "" {
		_ = os.Remove(run.DatasetPath)
	}

	return m.runStore.Delete(runID)
}

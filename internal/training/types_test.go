package training

import (
	"testing"
)

func TestDefaultRunConfig(t *testing.T) {
	cfg := DefaultRunConfig()

	if cfg.Epochs != 3 {
		t.Errorf("Epochs = %d, want 3", cfg.Epochs)
	}
	if cfg.LearningRate != 2e-4 {
		t.Errorf("LearningRate = %v, want 2e-4", cfg.LearningRate)
	}
	if cfg.LoraRank != 16 {
		t.Errorf("LoraRank = %d, want 16", cfg.LoraRank)
	}
	if cfg.LoraAlpha != 32 {
		t.Errorf("LoraAlpha = %d, want 32", cfg.LoraAlpha)
	}
	if cfg.BatchSize != 2 {
		t.Errorf("BatchSize = %d, want 2", cfg.BatchSize)
	}
	if cfg.MaxSamples != 0 {
		t.Errorf("MaxSamples = %d, want 0 (use all)", cfg.MaxSamples)
	}
}

func TestDefaultRunConfig_IsValue(t *testing.T) {
	// DefaultRunConfig returns a value — mutating one should not affect another
	cfg1 := DefaultRunConfig()
	cfg1.Epochs = 99
	cfg1.LearningRate = 1.0

	cfg2 := DefaultRunConfig()
	if cfg2.Epochs != 3 {
		t.Errorf("second DefaultRunConfig().Epochs = %d, expected 3", cfg2.Epochs)
	}
	if cfg2.LearningRate != 2e-4 {
		t.Errorf("second DefaultRunConfig().LearningRate = %v, expected 2e-4", cfg2.LearningRate)
	}
}

func TestRunStatus_Constants(t *testing.T) {
	// Verify the status constants have the expected string values
	tests := []struct {
		status RunStatus
		want   string
	}{
		{StatusPending, "pending"},
		{StatusPreparing, "preparing"},
		{StatusTraining, "training"},
		{StatusMerging, "merging"},
		{StatusDone, "done"},
		{StatusFailed, "failed"},
	}
	for _, tt := range tests {
		if string(tt.status) != tt.want {
			t.Errorf("RunStatus %q = %q, want %q", tt.status, string(tt.status), tt.want)
		}
	}
}

func TestTrainingRun_ZeroValue(t *testing.T) {
	var run TrainingRun
	if run.ID != "" {
		t.Errorf("zero TrainingRun.ID = %q, want empty", run.ID)
	}
	if run.Status != "" {
		t.Errorf("zero TrainingRun.Status = %q, want empty", run.Status)
	}
}

func TestRunConfig_LoraAlphaDoubleRank(t *testing.T) {
	cfg := DefaultRunConfig()
	// Conventionally LoraAlpha should be 2x LoraRank
	if cfg.LoraAlpha != cfg.LoraRank*2 {
		t.Errorf("LoraAlpha (%d) != LoraRank*2 (%d)", cfg.LoraAlpha, cfg.LoraRank*2)
	}
}

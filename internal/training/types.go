package training

import (
	"time"

	"github.com/ThatCatDev/tanrenai-gpu/pkg/api"
)

// RunStatus represents the state of a training run.
type RunStatus string

const (
	StatusPending   RunStatus = "pending"
	StatusPreparing RunStatus = "preparing"
	StatusTraining  RunStatus = "training"
	StatusMerging   RunStatus = "merging"
	StatusDone      RunStatus = "done"
	StatusFailed    RunStatus = "failed"
)

// RunConfig configures a training run.
type RunConfig struct {
	Epochs       int     `json:"epochs"`
	LearningRate float64 `json:"learning_rate"`
	LoraRank     int     `json:"lora_rank"`
	LoraAlpha    int     `json:"lora_alpha"`
	BatchSize    int     `json:"batch_size"`
	MaxSamples   int     `json:"max_samples"`
}

// DefaultRunConfig returns sensible defaults for fine-tuning.
func DefaultRunConfig() RunConfig {
	return RunConfig{
		Epochs:       3,
		LearningRate: 2e-4,
		LoraRank:     16,
		LoraAlpha:    32,
		BatchSize:    2,
		MaxSamples:   0, // 0 = use all available
	}
}

// RunMetrics contains training metrics.
type RunMetrics struct {
	TrainLoss   float64 `json:"train_loss,omitempty"`
	EvalLoss    float64 `json:"eval_loss,omitempty"`
	Duration    string  `json:"duration,omitempty"`
	SamplesUsed int     `json:"samples_used,omitempty"`
	Progress    float64 `json:"progress,omitempty"` // 0.0–1.0
}

// TrainingRun represents a single fine-tuning run.
type TrainingRun struct {
	ID          string     `json:"id"`
	BaseModel   string     `json:"base_model"`
	Status      RunStatus  `json:"status"`
	CreatedAt   time.Time  `json:"created_at"`
	UpdatedAt   time.Time  `json:"updated_at"`
	Config      RunConfig  `json:"config"`
	Metrics     RunMetrics `json:"metrics"`
	DatasetPath string     `json:"dataset_path,omitempty"`
	AdapterDir  string     `json:"adapter_dir,omitempty"`
	OutputModel string     `json:"output_model,omitempty"`
	Error       string     `json:"error,omitempty"`
}

// DatasetEntry is a single training sample in ChatML format.
type DatasetEntry struct {
	Messages []api.Message `json:"messages"`
}

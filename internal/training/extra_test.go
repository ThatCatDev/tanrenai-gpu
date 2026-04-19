package training

import (
	"testing"
)

// TestNewRunStore verifies the default constructor compiles and creates a store.
func TestNewRunStore(t *testing.T) {
	// NewRunStore uses config.TrainingRunsDir() — it just constructs a store.
	// We can't easily test it without side effects, but we can verify it doesn't panic.
	store := NewRunStore()
	if store == nil {
		t.Fatal("NewRunStore returned nil")
	}
}

// TestNewManager verifies the default constructor creates a manager.
func TestNewManager(t *testing.T) {
	client := NewSidecarClient("http://localhost:18082")
	// NewManager uses config.ModelsDir() — avoid side effects by just checking non-nil.
	m := NewManager(client)
	if m == nil {
		t.Fatal("NewManager returned nil")
	}
}

// TestDefaultRunConfig verifies sensible defaults.
func TestDefaultRunConfig_Values(t *testing.T) {
	cfg := DefaultRunConfig()
	if cfg.Epochs <= 0 {
		t.Errorf("Epochs = %d, want > 0", cfg.Epochs)
	}
	if cfg.LearningRate <= 0 {
		t.Errorf("LearningRate = %v, want > 0", cfg.LearningRate)
	}
	if cfg.LoraRank <= 0 {
		t.Errorf("LoraRank = %d, want > 0", cfg.LoraRank)
	}
}

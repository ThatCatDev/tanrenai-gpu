package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()

	if cfg.Host != "127.0.0.1" {
		t.Errorf("Host = %q, want %q", cfg.Host, "127.0.0.1")
	}
	if cfg.Port != 11435 {
		t.Errorf("Port = %d, want 11435", cfg.Port)
	}
	if cfg.GPULayers != -1 {
		t.Errorf("GPULayers = %d, want -1 (auto)", cfg.GPULayers)
	}
	if cfg.CtxSize != 4096 {
		t.Errorf("CtxSize = %d, want 4096", cfg.CtxSize)
	}
	if !cfg.FlashAttention {
		t.Error("FlashAttention = false, want true")
	}
	if cfg.ModelsDir == "" {
		t.Error("ModelsDir must not be empty")
	}
	if cfg.BinDir == "" {
		t.Error("BinDir must not be empty")
	}
}

func TestDefaultConfig_ModelsAndBinDirConsistency(t *testing.T) {
	cfg := DefaultConfig()

	// ModelsDir and BinDir should be siblings under the same data directory
	modelsParent := filepath.Dir(cfg.ModelsDir)
	binParent := filepath.Dir(cfg.BinDir)
	if modelsParent != binParent {
		t.Errorf("ModelsDir parent %q != BinDir parent %q; expected same data dir", modelsParent, binParent)
	}
}

func TestDataDir_EnvOverride(t *testing.T) {
	tmp := t.TempDir()
	t.Setenv("TANRENAI_DATA_DIR", tmp)

	got := DataDir()
	if got != tmp {
		t.Errorf("DataDir() = %q, want %q", got, tmp)
	}
}

func TestDataDir_DefaultContainsTanrenai(t *testing.T) {
	// Unset override so we get the default path
	t.Setenv("TANRENAI_DATA_DIR", "")

	got := DataDir()
	if !strings.Contains(got, "tanrenai") {
		t.Errorf("DataDir() = %q, expected it to contain 'tanrenai'", got)
	}
}

func TestModelsDir_EnvOverride(t *testing.T) {
	tmp := t.TempDir()
	t.Setenv("TANRENAI_MODELS_DIR", tmp)
	// Ensure data dir env doesn't interfere
	t.Setenv("TANRENAI_DATA_DIR", "")

	got := ModelsDir()
	if got != tmp {
		t.Errorf("ModelsDir() = %q, want %q", got, tmp)
	}
}

func TestModelsDir_DefaultUnderDataDir(t *testing.T) {
	t.Setenv("TANRENAI_DATA_DIR", "")
	t.Setenv("TANRENAI_MODELS_DIR", "")

	dataDir := DataDir()
	modelsDir := ModelsDir()

	if !strings.HasPrefix(modelsDir, dataDir) {
		t.Errorf("ModelsDir() = %q not under DataDir() = %q", modelsDir, dataDir)
	}
	if filepath.Base(modelsDir) != "models" {
		t.Errorf("ModelsDir() base = %q, want %q", filepath.Base(modelsDir), "models")
	}
}

func TestBinDir_UnderDataDir(t *testing.T) {
	t.Setenv("TANRENAI_DATA_DIR", "")

	dataDir := DataDir()
	binDir := BinDir()

	if !strings.HasPrefix(binDir, dataDir) {
		t.Errorf("BinDir() = %q not under DataDir() = %q", binDir, dataDir)
	}
	if filepath.Base(binDir) != "bin" {
		t.Errorf("BinDir() base = %q, want %q", filepath.Base(binDir), "bin")
	}
}

func TestTrainingDir_UnderDataDir(t *testing.T) {
	t.Setenv("TANRENAI_DATA_DIR", "")

	dataDir := DataDir()
	trainingDir := TrainingDir()

	if !strings.HasPrefix(trainingDir, dataDir) {
		t.Errorf("TrainingDir() = %q not under DataDir() = %q", trainingDir, dataDir)
	}
	if filepath.Base(trainingDir) != "training" {
		t.Errorf("TrainingDir() base = %q, want %q", filepath.Base(trainingDir), "training")
	}
}

func TestTrainingDatasetsDir_UnderTrainingDir(t *testing.T) {
	t.Setenv("TANRENAI_DATA_DIR", "")

	trainingDir := TrainingDir()
	datasetsDir := TrainingDatasetsDir()

	if !strings.HasPrefix(datasetsDir, trainingDir) {
		t.Errorf("TrainingDatasetsDir() = %q not under TrainingDir() = %q", datasetsDir, trainingDir)
	}
	if filepath.Base(datasetsDir) != "datasets" {
		t.Errorf("TrainingDatasetsDir() base = %q, want %q", filepath.Base(datasetsDir), "datasets")
	}
}

func TestTrainingRunsDir_UnderTrainingDir(t *testing.T) {
	t.Setenv("TANRENAI_DATA_DIR", "")

	trainingDir := TrainingDir()
	runsDir := TrainingRunsDir()

	if !strings.HasPrefix(runsDir, trainingDir) {
		t.Errorf("TrainingRunsDir() = %q not under TrainingDir() = %q", runsDir, trainingDir)
	}
	if filepath.Base(runsDir) != "runs" {
		t.Errorf("TrainingRunsDir() base = %q, want %q", filepath.Base(runsDir), "runs")
	}
}

func TestSidecarDir_UnderDataDir(t *testing.T) {
	t.Setenv("TANRENAI_DATA_DIR", "")

	dataDir := DataDir()
	sidecarDir := SidecarDir()

	if !strings.HasPrefix(sidecarDir, dataDir) {
		t.Errorf("SidecarDir() = %q not under DataDir() = %q", sidecarDir, dataDir)
	}
	if filepath.Base(sidecarDir) != "sidecar" {
		t.Errorf("SidecarDir() base = %q, want %q", filepath.Base(sidecarDir), "sidecar")
	}
}

func TestEnsureDirs_CreatesDirs(t *testing.T) {
	tmp := t.TempDir()
	t.Setenv("TANRENAI_DATA_DIR", tmp)
	t.Setenv("TANRENAI_MODELS_DIR", "")

	if err := EnsureDirs(); err != nil {
		t.Fatalf("EnsureDirs() error: %v", err)
	}

	// Verify expected directories exist
	expectedDirs := []string{
		DataDir(),
		ModelsDir(),
		BinDir(),
		TrainingDir(),
		TrainingDatasetsDir(),
		TrainingRunsDir(),
	}
	for _, dir := range expectedDirs {
		info, err := os.Stat(dir)
		if err != nil {
			t.Errorf("expected dir %q to exist: %v", dir, err)

			continue
		}
		if !info.IsDir() {
			t.Errorf("%q exists but is not a directory", dir)
		}
	}
}

func TestEnsureDirs_Idempotent(t *testing.T) {
	tmp := t.TempDir()
	t.Setenv("TANRENAI_DATA_DIR", tmp)
	t.Setenv("TANRENAI_MODELS_DIR", "")

	// Calling twice should not fail
	if err := EnsureDirs(); err != nil {
		t.Fatalf("first EnsureDirs() error: %v", err)
	}
	if err := EnsureDirs(); err != nil {
		t.Fatalf("second EnsureDirs() error: %v", err)
	}
}

func TestDataDir_CustomPathUsedByBinDir(t *testing.T) {
	tmp := t.TempDir()
	t.Setenv("TANRENAI_DATA_DIR", tmp)

	binDir := BinDir()
	if !strings.HasPrefix(binDir, tmp) {
		t.Errorf("BinDir() = %q, expected it to start with custom data dir %q", binDir, tmp)
	}
}

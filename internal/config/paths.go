package config

import (
	"os"
	"path/filepath"
	"runtime"
)

// DataDir returns the default data directory for tanrenai.
// Windows: %LOCALAPPDATA%\tanrenai
// Linux/Mac: ~/.local/share/tanrenai
func DataDir() string {
	if dir := os.Getenv("TANRENAI_DATA_DIR"); dir != "" {
		return dir
	}
	if runtime.GOOS == "windows" {
		return filepath.Join(os.Getenv("LOCALAPPDATA"), "tanrenai")
	}
	home, _ := os.UserHomeDir()

	return filepath.Join(home, ".local", "share", "tanrenai")
}

// ModelsDir returns the directory where models are stored.
func ModelsDir() string {
	if dir := os.Getenv("TANRENAI_MODELS_DIR"); dir != "" {
		return dir
	}

	return filepath.Join(DataDir(), "models")
}

// BinDir returns the directory where llama-server binaries are stored.
func BinDir() string {
	return filepath.Join(DataDir(), "bin")
}

// TrainingDir returns the base directory for fine-tuning data.
func TrainingDir() string {
	return filepath.Join(DataDir(), "training")
}

// TrainingDatasetsDir returns the directory for exported training datasets.
func TrainingDatasetsDir() string {
	return filepath.Join(TrainingDir(), "datasets")
}

// TrainingRunsDir returns the directory for training run metadata and artifacts.
func TrainingRunsDir() string {
	return filepath.Join(TrainingDir(), "runs")
}

// SidecarDir returns the directory containing the Python training sidecar.
func SidecarDir() string {
	return filepath.Join(DataDir(), "sidecar")
}

// EnsureDirs creates the required directories if they don't exist.
func EnsureDirs() error {
	dirs := []string{DataDir(), ModelsDir(), BinDir(), TrainingDir(), TrainingDatasetsDir(), TrainingRunsDir()}
	for _, dir := range dirs {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return err
		}
	}

	return nil
}

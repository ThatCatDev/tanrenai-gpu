package training

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"

	"github.com/ThatCatDev/tanrenai-gpu/internal/config"
)

// RunStore persists TrainingRun metadata as JSON files.
type RunStore struct {
	baseDir string
}

// NewRunStore creates a RunStore using the default training runs directory.
func NewRunStore() *RunStore {
	return &RunStore{baseDir: config.TrainingRunsDir()}
}

// NewRunStoreAt creates a RunStore at a custom directory (for testing).
func NewRunStoreAt(dir string) *RunStore {
	return &RunStore{baseDir: dir}
}

func (s *RunStore) runDir(id string) string {
	return filepath.Join(s.baseDir, id)
}

func (s *RunStore) configPath(id string) string {
	return filepath.Join(s.runDir(id), "config.json")
}

// Save persists a training run to disk.
func (s *RunStore) Save(run *TrainingRun) error {
	dir := s.runDir(run.ID)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("create run dir: %w", err)
	}

	data, err := json.MarshalIndent(run, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal run: %w", err)
	}

	if err := os.WriteFile(s.configPath(run.ID), data, 0644); err != nil {
		return fmt.Errorf("write run config: %w", err)
	}

	return nil
}

// Load reads a training run from disk.
func (s *RunStore) Load(id string) (*TrainingRun, error) {
	data, err := os.ReadFile(s.configPath(id))
	if err != nil {
		return nil, fmt.Errorf("read run config: %w", err)
	}

	var run TrainingRun
	if err := json.Unmarshal(data, &run); err != nil {
		return nil, fmt.Errorf("unmarshal run: %w", err)
	}

	return &run, nil
}

// List returns all training runs, sorted by creation time (newest first).
func (s *RunStore) List() ([]*TrainingRun, error) {
	entries, err := os.ReadDir(s.baseDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}

		return nil, fmt.Errorf("read runs dir: %w", err)
	}

	var runs []*TrainingRun
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		run, err := s.Load(entry.Name())
		if err != nil {
			continue // skip corrupt entries
		}
		runs = append(runs, run)
	}

	sort.Slice(runs, func(i, j int) bool {
		return runs[i].CreatedAt.After(runs[j].CreatedAt)
	})

	return runs, nil
}

// Delete removes a training run and all its artifacts.
func (s *RunStore) Delete(id string) error {
	dir := s.runDir(id)
	if _, err := os.Stat(dir); os.IsNotExist(err) {
		return fmt.Errorf("run %s not found", id)
	}

	return os.RemoveAll(dir)
}

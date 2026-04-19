package training

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestNewRunStoreAt(t *testing.T) {
	tmp := t.TempDir()
	s := NewRunStoreAt(tmp)
	if s == nil {
		t.Fatal("NewRunStoreAt returned nil")
	}
}

func TestRunStore_SaveAndLoad_RoundTrip(t *testing.T) {
	tmp := t.TempDir()
	s := NewRunStoreAt(tmp)

	now := time.Now().UTC().Truncate(time.Second)
	run := &TrainingRun{
		ID:        "run-001",
		BaseModel: "mymodel",
		Status:    StatusPending,
		CreatedAt: now,
		UpdatedAt: now,
		Config:    DefaultRunConfig(),
	}

	if err := s.Save(run); err != nil {
		t.Fatalf("Save: %v", err)
	}

	loaded, err := s.Load("run-001")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	if loaded.ID != run.ID {
		t.Errorf("ID = %q, want %q", loaded.ID, run.ID)
	}
	if loaded.BaseModel != run.BaseModel {
		t.Errorf("BaseModel = %q, want %q", loaded.BaseModel, run.BaseModel)
	}
	if loaded.Status != run.Status {
		t.Errorf("Status = %q, want %q", loaded.Status, run.Status)
	}
	if loaded.Config.Epochs != run.Config.Epochs {
		t.Errorf("Config.Epochs = %d, want %d", loaded.Config.Epochs, run.Config.Epochs)
	}
}

func TestRunStore_Load_NotFound(t *testing.T) {
	tmp := t.TempDir()
	s := NewRunStoreAt(tmp)

	_, err := s.Load("nonexistent-run")
	if err == nil {
		t.Fatal("Load on nonexistent run: expected error, got nil")
	}
}

func TestRunStore_List_Empty(t *testing.T) {
	tmp := t.TempDir()
	s := NewRunStoreAt(tmp)

	runs, err := s.List()
	if err != nil {
		t.Fatalf("List on empty dir: %v", err)
	}
	if len(runs) != 0 {
		t.Errorf("List() = %d runs, want 0", len(runs))
	}
}

func TestRunStore_List_NonExistentDir(t *testing.T) {
	s := NewRunStoreAt("/nonexistent/path/xyz")

	runs, err := s.List()
	if err != nil {
		t.Fatalf("List on non-existent dir: expected nil error, got %v", err)
	}
	if runs != nil {
		t.Errorf("List() = %v, want nil", runs)
	}
}

func TestRunStore_List_Multiple_SortedNewestFirst(t *testing.T) {
	tmp := t.TempDir()
	s := NewRunStoreAt(tmp)

	base := time.Now().UTC().Truncate(time.Second)
	runsToSave := []*TrainingRun{
		{ID: "run-a", BaseModel: "model1", Status: StatusDone, CreatedAt: base, UpdatedAt: base},
		{ID: "run-b", BaseModel: "model2", Status: StatusTraining, CreatedAt: base.Add(time.Minute), UpdatedAt: base.Add(time.Minute)},
		{ID: "run-c", BaseModel: "model3", Status: StatusFailed, CreatedAt: base.Add(-time.Minute), UpdatedAt: base},
	}

	for _, r := range runsToSave {
		if err := s.Save(r); err != nil {
			t.Fatalf("Save(%s): %v", r.ID, err)
		}
	}

	listed, err := s.List()
	if err != nil {
		t.Fatalf("List: %v", err)
	}

	if len(listed) != 3 {
		t.Errorf("List() = %d runs, want 3", len(listed))
	}

	// Should be sorted newest first — run-b (base+1m) should come first
	if len(listed) >= 1 && listed[0].ID != "run-b" {
		t.Errorf("listed[0].ID = %q, want %q (newest first)", listed[0].ID, "run-b")
	}
	if len(listed) >= 3 && listed[2].ID != "run-c" {
		t.Errorf("listed[2].ID = %q, want %q (oldest last)", listed[2].ID, "run-c")
	}
}

func TestRunStore_Save_UpdatesExisting(t *testing.T) {
	tmp := t.TempDir()
	s := NewRunStoreAt(tmp)

	now := time.Now().UTC().Truncate(time.Second)
	run := &TrainingRun{
		ID:        "run-upd",
		BaseModel: "model",
		Status:    StatusPending,
		CreatedAt: now,
		UpdatedAt: now,
	}

	if err := s.Save(run); err != nil {
		t.Fatalf("first Save: %v", err)
	}

	// Update status and save again
	run.Status = StatusDone
	if err := s.Save(run); err != nil {
		t.Fatalf("second Save: %v", err)
	}

	loaded, err := s.Load("run-upd")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if loaded.Status != StatusDone {
		t.Errorf("Status after update = %q, want %q", loaded.Status, StatusDone)
	}
}

func TestRunStore_Delete_Exists(t *testing.T) {
	tmp := t.TempDir()
	s := NewRunStoreAt(tmp)

	now := time.Now().UTC()
	run := &TrainingRun{ID: "run-del", BaseModel: "m", Status: StatusDone, CreatedAt: now, UpdatedAt: now}

	if err := s.Save(run); err != nil {
		t.Fatalf("Save: %v", err)
	}

	if err := s.Delete("run-del"); err != nil {
		t.Fatalf("Delete: %v", err)
	}

	// Should no longer be loadable
	_, err := s.Load("run-del")
	if err == nil {
		t.Fatal("Load after Delete: expected error, got nil")
	}
}

func TestRunStore_Delete_NotFound(t *testing.T) {
	tmp := t.TempDir()
	s := NewRunStoreAt(tmp)

	err := s.Delete("ghost-run")
	if err == nil {
		t.Fatal("Delete on nonexistent run: expected error, got nil")
	}
}

func TestRunStore_List_SkipsCorruptEntries(t *testing.T) {
	tmp := t.TempDir()
	s := NewRunStoreAt(tmp)

	// Save one valid run
	now := time.Now().UTC().Truncate(time.Second)
	valid := &TrainingRun{ID: "run-valid", BaseModel: "m", Status: StatusDone, CreatedAt: now, UpdatedAt: now}
	if err := s.Save(valid); err != nil {
		t.Fatalf("Save valid: %v", err)
	}

	// Manually create a corrupt run directory (invalid JSON)
	corruptDir := filepath.Join(tmp, "run-corrupt")
	if err := os.MkdirAll(corruptDir, 0755); err != nil {
		t.Fatalf("mkdir corrupt: %v", err)
	}
	corruptConfig := filepath.Join(corruptDir, "config.json")
	if err := os.WriteFile(corruptConfig, []byte("{invalid json"), 0644); err != nil {
		t.Fatalf("write corrupt config: %v", err)
	}

	listed, err := s.List()
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	// Only the valid run should appear; corrupt entry is silently skipped
	if len(listed) != 1 {
		t.Errorf("List() = %d runs, want 1 (corrupt entry should be skipped)", len(listed))
	}
	if len(listed) == 1 && listed[0].ID != "run-valid" {
		t.Errorf("listed[0].ID = %q, want %q", listed[0].ID, "run-valid")
	}
}

func TestRunStore_List_SkipsNonDirectories(t *testing.T) {
	tmp := t.TempDir()
	s := NewRunStoreAt(tmp)

	// Place a plain file in the runs dir — it should be ignored
	if err := os.WriteFile(filepath.Join(tmp, "somefile.txt"), []byte("data"), 0644); err != nil {
		t.Fatal(err)
	}

	now := time.Now().UTC().Truncate(time.Second)
	run := &TrainingRun{ID: "run-real", BaseModel: "m", Status: StatusDone, CreatedAt: now, UpdatedAt: now}
	if err := s.Save(run); err != nil {
		t.Fatalf("Save: %v", err)
	}

	listed, err := s.List()
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(listed) != 1 {
		t.Errorf("List() = %d runs, want 1 (non-dir files ignored)", len(listed))
	}
}

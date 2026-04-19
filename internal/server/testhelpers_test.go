package server

import (
	"testing"

	"github.com/ThatCatDev/tanrenai-gpu/internal/models"
	"github.com/ThatCatDev/tanrenai-gpu/internal/training"
)

func training_new_manager_for_test(t *testing.T) *training.Manager {
	t.Helper()
	client := training.NewSidecarClient("http://localhost:18082")
	store := training.NewRunStoreAt(t.TempDir())

	return training.NewManagerWithStore(store, client, t.TempDir())
}

func newModelsStore(t *testing.T) *models.Store {
	t.Helper()

	return models.NewStore(t.TempDir())
}

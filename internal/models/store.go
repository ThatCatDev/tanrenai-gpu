package models

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// Store manages locally available GGUF model files.
type Store struct {
	dir string
}

// NewStore creates a new Store for the given directory.
func NewStore(dir string) *Store {
	return &Store{dir: dir}
}

// Dir returns the models directory path.
func (s *Store) Dir() string {
	return s.dir
}

// splitSuffix matches split GGUF filenames like "model-00001-of-00003.gguf"
var splitSuffix = regexp.MustCompile(`-\d{5}-of-\d{5}\.gguf$`)

// List returns all available models by scanning the models directory for .gguf files.
// Split GGUFs (multi-part) are grouped into a single entry using the first part's path.
func (s *Store) List() []ModelEntry {
	type modelInfo struct {
		entry ModelEntry
		parts int
	}
	models := map[string]*modelInfo{}

	// Resolve symlinks so filepath.Walk can traverse
	dir := s.dir
	if resolved, err := filepath.EvalSymlinks(dir); err == nil {
		dir = resolved
	}

	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() || !strings.HasSuffix(strings.ToLower(info.Name()), ".gguf") {
			return nil
		}

		name := info.Name()
		// Group split GGUFs by base name
		baseName := splitSuffix.ReplaceAllString(name, "")
		if baseName == name {
			// Single file — strip .gguf
			baseName = strings.TrimSuffix(name, filepath.Ext(name))
		}

		if m, ok := models[baseName]; ok {
			m.entry.Size += info.Size()
			m.parts++
			// Keep the first part's path for loading
			if path < m.entry.Path {
				m.entry.Path = path
			}
		} else {
			models[baseName] = &modelInfo{
				entry: ModelEntry{
					Name:       baseName,
					Path:       path,
					Size:       info.Size(),
					ModifiedAt: info.ModTime().Unix(),
				},
				parts: 1,
			}
		}

		return nil
	})
	_ = err // Directory may not exist yet, that's OK

	entries := make([]ModelEntry, 0, len(models))
	for _, m := range models {
		entries = append(entries, m.entry)
	}

	return entries
}

// Resolve finds a model by name and returns its full path.
// It searches for an exact filename match (with or without .gguf extension),
// or a partial name match.
func (s *Store) Resolve(name string) (string, error) {
	// Try exact path first
	if filepath.IsAbs(name) {
		if _, err := os.Stat(name); err == nil {
			return name, nil
		}
	}

	// Try with .gguf extension in models dir
	candidate := filepath.Join(s.dir, name)
	if _, err := os.Stat(candidate); err == nil {
		return candidate, nil
	}
	candidate = candidate + ".gguf"
	if _, err := os.Stat(candidate); err == nil {
		return candidate, nil
	}

	// Search by partial name match
	entries := s.List()
	for _, e := range entries {
		if strings.EqualFold(e.Name, name) {
			return e.Path, nil
		}
		if strings.Contains(strings.ToLower(e.Name), strings.ToLower(name)) {
			return e.Path, nil
		}
	}

	return "", fmt.Errorf("model %q not found in %s", name, s.dir)
}

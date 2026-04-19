package runner

import (
	"os"
	"path/filepath"
)

// WriteTemplateFile writes a template string to a temp file and returns its path.
// The caller should clean up the file when done.
func WriteTemplateFile(name, content string) (string, error) {
	dir := os.TempDir()
	path := filepath.Join(dir, "tanrenai-"+name+"-chat.jinja")
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		return "", err
	}

	return path, nil
}

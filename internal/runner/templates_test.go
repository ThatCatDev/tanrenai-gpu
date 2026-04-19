package runner

import (
	"os"
	"path/filepath"
	"testing"
)

// TestWriteTemplateFile_WriteError verifies that WriteTemplateFile returns an error
// when os.WriteFile fails (e.g., target path is a directory).
func TestWriteTemplateFile_WriteError(t *testing.T) {
	// Create a directory at the exact path that WriteTemplateFile would write to.
	// WriteTemplateFile writes to os.TempDir()/tanrenai-<name>-chat.jinja
	name := "testerror-dir-block"
	dir := os.TempDir()
	targetPath := filepath.Join(dir, "tanrenai-"+name+"-chat.jinja")

	// Create a directory at that path so os.WriteFile fails (is a directory).
	if err := os.MkdirAll(targetPath, 0755); err != nil {
		t.Skipf("cannot create blocking directory: %v", err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(targetPath) })

	_, err := WriteTemplateFile(name, "{{ test }}")
	if err == nil {
		t.Fatal("expected error when WriteTemplateFile target is a directory")
	}
}

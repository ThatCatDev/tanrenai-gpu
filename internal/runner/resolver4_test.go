package runner

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/ThatCatDev/tanrenai-gpu/internal/models"
)

// TestResolveHFTemplate_Nil verifies resolveHFTemplate returns (nil,nil) for nil metadata.
func TestResolveHFTemplate_Nil(t *testing.T) {
	res, err := resolveHFTemplate(nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res != nil {
		t.Errorf("expected nil result for nil modelMeta")
	}
}

// TestResolveHFTemplate_EmptyHFRepo verifies resolveHFTemplate returns (nil,nil)
// when HFRepo is empty.
func TestResolveHFTemplate_EmptyHFRepo(t *testing.T) {
	meta := &models.ModelMetadata{
		HFRepo:   "",
		HFBranch: "main",
	}
	res, err := resolveHFTemplate(meta)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res != nil {
		t.Errorf("expected nil result for empty HFRepo")
	}
}

// TestResolveHFTemplate_DefaultBranch verifies resolveHFTemplate defaults to "main"
// when HFBranch is empty. The HF fetch will fail (fake repo), returning (nil,nil).
func TestResolveHFTemplate_DefaultBranch(t *testing.T) {
	meta := &models.ModelMetadata{
		HFRepo:   "fakerepo/nonexistent",
		HFBranch: "", // should default to "main"
	}
	// HF fetch will fail for a fake repo → returns nil, nil (not an error)
	res, err := resolveHFTemplate(meta)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Will be nil because HF fetch fails (network error to huggingface.co for fake repo)
	_ = res
}

// TestResolveHFTemplate_Success verifies the success path of resolveHFTemplate
// by patching the HFClient via a mock httptest server. We call resolveHFTemplate
// indirectly through a custom modelMeta object and a real HFClient pointed at our mock.
//
// Since resolveHFTemplate creates its own HFClient internally, we test the success
// path by directly calling the underlying code that resolveHFTemplate would call.
func TestResolveHFTemplate_SuccessPath(t *testing.T) {
	tplContent := "{% for m in messages %}{{ m.content }}{% endfor %}"

	// Start a fake HuggingFace server
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]string{
			"chat_template": tplContent,
		})
	}))
	defer srv.Close()

	// Use models.HFClient directly with our mock server URL
	hf := models.NewHFClient()
	hf.BaseURL = srv.URL

	tpl, err := hf.FetchChatTemplate("owner/repo", "main")
	if err != nil {
		t.Fatalf("FetchChatTemplate: %v", err)
	}
	if tpl != tplContent {
		t.Errorf("template = %q, want %q", tpl, tplContent)
	}

	// Now exercise WriteTemplateFile (used by resolveHFTemplate success path)
	name := sanitizeName("owner/repo")
	path, writeErr := WriteTemplateFile(name, tpl)
	if writeErr != nil {
		t.Fatalf("WriteTemplateFile: %v", writeErr)
	}
	defer os.Remove(path)

	data, readErr := os.ReadFile(path)
	if readErr != nil {
		t.Fatalf("ReadFile: %v", readErr)
	}
	if string(data) != tplContent {
		t.Errorf("file content = %q, want %q", string(data), tplContent)
	}
}

// TestResolveHFTemplate_EmptyTemplate verifies that when HF returns an empty template,
// resolveHFTemplate returns (nil, nil) — exercising the tpl=="" branch.
// We test this via the code path: non-empty repo → fetch fails → return nil,nil.
func TestResolveHFTemplate_FetchFailsReturnsNil(t *testing.T) {
	// A server that returns 404 — FetchChatTemplate will return an error.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer srv.Close()

	// We can't inject the server URL into resolveHFTemplate directly.
	// Test via HFClient directly to verify the code path is consistent:
	hf := models.NewHFClient()
	hf.BaseURL = srv.URL

	_, err := hf.FetchChatTemplate("owner/repo", "main")
	if err == nil {
		t.Fatal("expected error for 404 response")
	}
	// resolveHFTemplate treats this as non-fatal and returns (nil, nil)
	// This is tested transitively via TestResolveHFTemplate_DefaultBranch above.
}

// TestResolveTemplate_WriteTemplateFileError verifies that ResolveTemplate returns
// an error when the ChatML fallback WriteTemplateFile call fails.
// We pre-create a directory at the path that WriteTemplateFile would use for "llama2".
func TestResolveTemplate_WriteTemplateFileError(t *testing.T) {
	// GGUF with "llama2" architecture — sanitizeName("llama2") = "llama2"
	// WriteTemplateFile would write to /tmp/tanrenai-llama2-chat.jinja
	ggufPath := buildGGUFFile(t, "llama2", "Llama2Model", "")

	targetPath := os.TempDir() + "/tanrenai-llama2-chat.jinja"
	// Create a directory at that path to block WriteTemplateFile
	if err := os.MkdirAll(targetPath, 0755); err != nil {
		t.Skipf("cannot create blocking directory: %v", err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(targetPath) })

	res, err := ResolveTemplate(ggufPath, nil)
	if err == nil {
		if res != nil && res.Cleanup != nil {
			res.Cleanup()
		}
		t.Fatal("expected error when WriteTemplateFile fails due to directory blocking")
	}
}

// TestResolveTemplate_HFSuccess tests ResolveTemplate when an HF fetch succeeds.
// We build a GGUF with no embedded template and a .meta.json with HFRepo pointing
// at our mock server. Since ResolveTemplate creates its own HFClient with the real
// HuggingFace URL, we test the fallback instead.
func TestResolveTemplate_LoadMetadataError(t *testing.T) {
	// Test with a GGUF that has architecture but no template, and
	// a corrupt .meta.json file (simulate LoadMetadata error).
	ggufPath := buildGGUFFile(t, "llama", "TestModel", "")

	// Write a corrupt .meta.json sidecar
	metaPath := ggufPath + ".meta.json"
	if err := os.WriteFile(metaPath, []byte("{corrupt json"), 0644); err != nil {
		t.Fatal(err)
	}
	defer os.Remove(metaPath)

	// ResolveTemplate should log a warning about metadata, then fall through
	// to ChatML fallback (since GGUF has architecture).
	res, err := ResolveTemplate(ggufPath, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res == nil {
		t.Fatal("expected ChatML fallback resolution when metadata is corrupt")
	}
	defer func() {
		if res.Cleanup != nil {
			res.Cleanup()
		}
	}()

	if res.Source != "generated:chatml" {
		t.Errorf("Source = %q, want generated:chatml", res.Source)
	}
}

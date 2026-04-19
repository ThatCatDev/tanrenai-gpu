package runner

import (
	"fmt"
	"log/slog"
	"os"
	"strings"

	"github.com/ThatCatDev/tanrenai-gpu/internal/gguf"
	"github.com/ThatCatDev/tanrenai-gpu/internal/models"
)

// TemplateResolution holds the result of automatic template detection.
type TemplateResolution struct {
	// TemplatePath is the path to the generated/fetched template file.
	TemplatePath string

	// Source describes where the template came from.
	// Examples: "generated:chatml", "huggingface:Qwen/Qwen2.5-32B-GGUF", "none"
	Source string

	// Cleanup removes the temporary template file. May be nil.
	Cleanup func()
}

// ResolveTemplate attempts to auto-detect and generate a chat template for the
// given GGUF model file. If meta is non-nil it is used directly; otherwise the
// GGUF header is read from modelPath. Returns nil (no error) if no template
// could be determined.
//
// Resolution chain:
//  1. If GGUF has embedded tokenizer.chat_template → return nil (llama-server reads it with --jinja)
//  2. If .meta.json has HF repo → fetch from HuggingFace (model-specific, higher quality)
//  3. If GGUF has architecture metadata but no template → generic ChatML fallback with warning
//  4. Return nil if no strategy works
func ResolveTemplate(modelPath string, meta *gguf.Metadata) (*TemplateResolution, error) {
	// Step 1: Read GGUF metadata — check for embedded chat template.
	if meta == nil {
		var err error
		meta, err = gguf.ReadMetadata(modelPath)
		if err != nil {
			slog.Warn("template resolver: could not read GGUF metadata", "error", err)
			// Non-fatal — continue to fallback strategies.
		}
	}
	if meta != nil && meta.Tokenizer.ChatTemplate != "" {
		// The GGUF already contains an embedded chat template — llama-server
		// reads it directly with --jinja. No need to generate or override.
		slog.Info("template resolver: GGUF has embedded chat_template, skipping generation")

		return nil, nil
	}

	// Step 2: Try HuggingFace via .meta.json sidecar.
	modelMeta, err := models.LoadMetadata(modelPath)
	if err != nil {
		slog.Warn("template resolver: could not load model metadata", "error", err)
	}
	if res, err := resolveHFTemplate(modelMeta); res != nil || err != nil {
		return res, err
	}

	// Step 3: Generic ChatML fallback when the GGUF has architecture metadata
	// but no embedded template and no HuggingFace source.
	if meta != nil && meta.General.Architecture != "" {
		slog.Warn("template resolver: no embedded template or HuggingFace source, falling back to generic ChatML",
			"architecture", meta.General.Architecture, "name", meta.General.Name)
		cfg := DefaultChatMLConfig
		tpl := GenerateChatML(cfg)
		name := sanitizeName(meta.General.Architecture)
		path, err := WriteTemplateFile(name, tpl)
		if err != nil {
			return nil, fmt.Errorf("template resolver: write generated template: %w", err)
		}

		return &TemplateResolution{
			TemplatePath: path,
			Source:       "generated:chatml",
			Cleanup:      func() { _ = os.Remove(path) },
		}, nil
	}

	// Step 4: No template found — return nil (caller uses whatever llama-server defaults to).
	return nil, nil
}

// resolveHFTemplate attempts to fetch a chat template from HuggingFace using
// the model's .meta.json sidecar. Returns (nil, nil) if modelMeta is nil or
// has no HFRepo set, so the caller can fall through to the next strategy.
func resolveHFTemplate(modelMeta *models.ModelMetadata) (*TemplateResolution, error) {
	if modelMeta == nil || modelMeta.HFRepo == "" {
		return nil, nil
	}
	branch := modelMeta.HFBranch
	if branch == "" {
		branch = "main"
	}
	hf := models.NewHFClient()
	tpl, err := hf.FetchChatTemplate(modelMeta.HFRepo, branch)
	if err != nil {
		slog.Warn("template resolver: HuggingFace fetch failed", "error", err)

		return nil, nil
	}
	if tpl == "" {
		return nil, nil
	}
	name := sanitizeName(modelMeta.HFRepo)
	path, err := WriteTemplateFile(name, tpl)
	if err != nil {
		return nil, fmt.Errorf("template resolver: write HF template: %w", err)
	}

	return &TemplateResolution{
		TemplatePath: path,
		Source:       "huggingface:" + modelMeta.HFRepo,
		Cleanup:      func() { _ = os.Remove(path) },
	}, nil
}

// sanitizeName produces a safe filename component from an architecture or repo name.
func sanitizeName(s string) string {
	r := strings.NewReplacer("/", "-", " ", "-", ".", "-")

	return strings.ToLower(r.Replace(s))
}

package models

import (
	"encoding/json"
	"os"
	"regexp"
	"strings"
)

// ModelMetadata stores provenance information alongside a GGUF file.
// Saved as <model>.gguf.meta.json.
type ModelMetadata struct {
	HFRepo   string `json:"hf_repo,omitempty"`
	HFBranch string `json:"hf_branch,omitempty"`
	Source   string `json:"source"`
}

// MetadataPath returns the sidecar metadata path for a GGUF file.
func MetadataPath(ggufPath string) string {
	return ggufPath + ".meta.json"
}

// LoadMetadata reads the sidecar metadata file for the given GGUF path.
// Returns nil, nil if the file does not exist.
func LoadMetadata(ggufPath string) (*ModelMetadata, error) {
	data, err := os.ReadFile(MetadataPath(ggufPath))
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}

		return nil, err
	}
	var meta ModelMetadata
	if err := json.Unmarshal(data, &meta); err != nil {
		return nil, err
	}

	return &meta, nil
}

// SaveMetadata writes the sidecar metadata file for the given GGUF path.
func SaveMetadata(ggufPath string, meta *ModelMetadata) error {
	data, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(MetadataPath(ggufPath), data, 0644)
}

// hfURLPattern matches HuggingFace resolve URLs:
// https://huggingface.co/{owner}/{repo}/resolve/{branch}/{filename}
var hfURLPattern = regexp.MustCompile(
	`^https?://huggingface\.co/([^/]+/[^/]+)/resolve/([^/]+)/`,
)

// ParseHFURL extracts the repo and branch from a HuggingFace download URL.
// Returns ("", "", false) if the URL doesn't match the expected pattern.
func ParseHFURL(rawURL string) (repo, branch string, ok bool) {
	rawURL = strings.TrimSpace(rawURL)
	m := hfURLPattern.FindStringSubmatch(rawURL)
	if m == nil {
		return "", "", false
	}

	return m[1], m[2], true
}

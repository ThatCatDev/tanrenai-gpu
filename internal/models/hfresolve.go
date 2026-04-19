package models

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"sort"
	"strings"
	"time"
)

// HFFileInfo represents a file in a HuggingFace repo.
type HFFileInfo struct {
	Filename string `json:"rfilename"`
	Size     int64  `json:"size"`
}

// hfAPIBaseURL is the base URL for the HuggingFace API.
// It can be overridden in tests.
var hfAPIBaseURL = "https://huggingface.co"

// ResolveHFModel resolves a HuggingFace model reference into direct download URLs.
// Accepts formats:
//   - hf://owner/repo                          → find best single GGUF
//   - hf://owner/repo/quant                    → find GGUF files in quant subfolder or matching quant name
//   - https://huggingface.co/owner/repo/...    → pass through as-is
//
// Returns a list of URLs to download (multiple for split GGUFs).
func ResolveHFModel(ref string) ([]string, error) {
	// Pass through direct URLs
	if strings.HasPrefix(ref, "https://") || strings.HasPrefix(ref, "http://") {
		return []string{ref}, nil
	}

	// Parse hf:// format
	if !strings.HasPrefix(ref, "hf://") {
		return nil, fmt.Errorf("unsupported model reference: %s (use hf://owner/repo or a direct URL)", ref)
	}

	path := strings.TrimPrefix(ref, "hf://")
	parts := strings.SplitN(path, "/", 3)
	if len(parts) < 2 {
		return nil, fmt.Errorf("invalid hf:// reference: need at least owner/repo")
	}

	repo := parts[0] + "/" + parts[1]
	quant := ""
	if len(parts) == 3 {
		quant = parts[2]
	}

	// List files in the repo
	files, err := listHFFiles(repo)
	if err != nil {
		return nil, fmt.Errorf("list repo %s: %w", repo, err)
	}

	// Filter to GGUF files
	var ggufFiles []HFFileInfo
	for _, f := range files {
		if strings.HasSuffix(strings.ToLower(f.Filename), ".gguf") {
			ggufFiles = append(ggufFiles, f)
		}
	}

	if len(ggufFiles) == 0 {
		return nil, fmt.Errorf("no GGUF files found in %s", repo)
	}

	// If quant specified, filter by subfolder or filename match
	var filtered []HFFileInfo
	var filterErr error
	if quant != "" {
		filtered, filterErr = filterByQuant(ggufFiles, quant, repo)
		if filterErr != nil {
			return nil, filterErr
		}
	} else {
		// No quant specified — pick the best single file or smallest quant
		filtered = pickBestQuant(ggufFiles)
	}
	ggufFiles = filtered

	// Build download URLs
	sort.Slice(ggufFiles, func(i, j int) bool {
		return ggufFiles[i].Filename < ggufFiles[j].Filename
	})

	var urls []string
	for _, f := range ggufFiles {
		urls = append(urls, fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s", repo, f.Filename))
	}

	return urls, nil
}

// filterByQuant filters ggufFiles to those matching the given quant specifier.
// It first tries a subfolder prefix match (e.g. "UD-Q4_K_XL/"), then falls
// back to a case-insensitive filename substring match. Returns an error listing
// available quants if no match is found.
func filterByQuant(ggufFiles []HFFileInfo, quant, repo string) ([]HFFileInfo, error) {
	var matched []HFFileInfo
	// Try subfolder match first (e.g. "UD-Q4_K_XL/file.gguf")
	for _, f := range ggufFiles {
		if strings.HasPrefix(f.Filename, quant+"/") {
			matched = append(matched, f)
		}
	}
	// Fall back to filename substring match
	if len(matched) == 0 {
		needle := strings.ToLower(quant)
		for _, f := range ggufFiles {
			if strings.Contains(strings.ToLower(f.Filename), needle) {
				matched = append(matched, f)
			}
		}
	}
	if len(matched) == 0 {
		quants := availableQuants(ggufFiles)

		return nil, fmt.Errorf("no GGUF files matching %q in %s\navailable: %s", quant, repo, strings.Join(quants, ", "))
	}

	return matched, nil
}

func listHFFiles(repo string) ([]HFFileInfo, error) {
	url := fmt.Sprintf("%s/api/models/%s", hfAPIBaseURL, repo)

	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	if token := os.Getenv("HF_TOKEN"); token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HuggingFace API returned %d", resp.StatusCode)
	}

	var result struct {
		Siblings []HFFileInfo `json:"siblings"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return result.Siblings, nil
}

// availableQuants extracts unique quant names from file paths.
func availableQuants(files []HFFileInfo) []string {
	seen := map[string]bool{}
	for _, f := range files {
		parts := strings.Split(f.Filename, "/")
		if len(parts) > 1 {
			seen[parts[0]] = true
		} else {
			// Extract quant from filename like "Model-Q4_K_M.gguf"
			name := strings.TrimSuffix(f.Filename, ".gguf")
			// Remove split suffix
			for _, suffix := range []string{"-00001-of-00002", "-00001-of-00003", "-00002-of-00003"} {
				name = strings.TrimSuffix(name, suffix)
			}
			seen[name] = true
		}
	}
	var quants []string
	for q := range seen {
		quants = append(quants, q)
	}
	sort.Strings(quants)

	return quants
}

// pickBestQuant selects Q4_K_M or the smallest available quant if no specific one requested.
func pickBestQuant(files []HFFileInfo) []HFFileInfo {
	// Prefer Q4_K_M
	for _, preferred := range []string{"Q4_K_M", "Q4_K_S", "Q4_0"} {
		var matched []HFFileInfo
		for _, f := range files {
			if strings.Contains(strings.ToUpper(f.Filename), preferred) {
				matched = append(matched, f)
			}
		}
		if len(matched) > 0 {
			return matched
		}
	}
	// Fall back to first file
	return files[:1]
}

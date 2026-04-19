package models

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

// HFClient fetches metadata from HuggingFace model repositories.
type HFClient struct {
	BaseURL    string
	HTTPClient *http.Client
}

// NewHFClient creates a new HuggingFace client with sensible defaults.
func NewHFClient() *HFClient {
	return &HFClient{
		BaseURL: "https://huggingface.co",
		HTTPClient: &http.Client{
			Timeout: 10 * time.Second,
		},
	}
}

// FetchChatTemplate downloads the tokenizer_config.json from a HuggingFace repo
// and extracts the chat_template field. Handles both string and array-of-objects
// formats (where the default template is selected).
func (c *HFClient) FetchChatTemplate(repo, branch string) (string, error) {
	url := fmt.Sprintf("%s/%s/raw/%s/tokenizer_config.json", c.BaseURL, repo, branch)

	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return "", fmt.Errorf("hf: create request: %w", err)
	}

	if token := os.Getenv("HF_TOKEN"); token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("hf: request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("hf: status %d for %s", resp.StatusCode, url)
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, 10*1024*1024)) // 10 MB limit
	if err != nil {
		return "", fmt.Errorf("hf: read body: %w", err)
	}

	return extractChatTemplate(body)
}

// extractChatTemplate parses the chat_template field from tokenizer_config.json.
// The field can be either a plain string or an array of objects with "name" and
// "template" fields (where we pick the "default" entry, or the first one).
func extractChatTemplate(data []byte) (string, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return "", fmt.Errorf("hf: parse config: %w", err)
	}

	tplRaw, ok := raw["chat_template"]
	if !ok {
		return "", fmt.Errorf("hf: no chat_template field in tokenizer_config.json")
	}

	// Try string first.
	var tplString string
	if err := json.Unmarshal(tplRaw, &tplString); err == nil {
		return tplString, nil
	}

	// Try array of objects.
	var tplArray []struct {
		Name     string `json:"name"`
		Template string `json:"template"`
	}
	if err := json.Unmarshal(tplRaw, &tplArray); err != nil {
		return "", fmt.Errorf("hf: chat_template is neither string nor array: %w", err)
	}

	if len(tplArray) == 0 {
		return "", fmt.Errorf("hf: empty chat_template array")
	}

	// Prefer "default" entry.
	for _, entry := range tplArray {
		if entry.Name == "default" {
			return entry.Template, nil
		}
	}

	// Fall back to first entry.
	return tplArray[0].Template, nil
}

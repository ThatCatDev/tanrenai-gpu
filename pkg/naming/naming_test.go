package naming

import "testing"

func TestResolveBareNameToURI(t *testing.T) {
	tests := []struct {
		name string
		want string
	}{
		{"Qwen3.5-122B-A10B-UD-Q4_K_XL", "hf://unsloth/Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL"},
		{"Qwen2.5-7B-Instruct-Q4_K_M", "hf://unsloth/Qwen2.5-7B-Instruct-GGUF/Q4_K_M"},
		{"Llama-3-8B-Instruct-Q5_K_S", "hf://unsloth/Llama-3-8B-Instruct-GGUF/Q5_K_S"},
		{"Mistral-7B-Instruct-v0.3-IQ2_XS", "hf://unsloth/Mistral-7B-Instruct-v0.3-GGUF/IQ2_XS"},
		{"gemma-2-27b-it-BF16", "hf://unsloth/gemma-2-27b-it-GGUF/BF16"},
		{"Qwen2.5-0.5B-UD-Q2_K_XL", "hf://unsloth/Qwen2.5-0.5B-GGUF/UD-Q2_K_XL"},
		{"Qwen3.6-35B-A3B-MTP-Q8_0", "hf://unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Q8_0"},
		// URIs pass through unchanged.
		{"hf://unsloth/foo/Q4_K_M", "hf://unsloth/foo/Q4_K_M"},
		{"https://huggingface.co/x/y.gguf", "https://huggingface.co/x/y.gguf"},
		{"http://example.com/m.gguf", "http://example.com/m.gguf"},
		// No quant suffix — caller should error rather than misroute.
		{"Qwen2.5-7B-Instruct", ""},
		{"just-a-plain-name", ""},
		{"", ""},
	}
	for _, tc := range tests {
		got := ResolveBareNameToURI(tc.name)
		if got != tc.want {
			t.Errorf("ResolveBareNameToURI(%q) = %q, want %q", tc.name, got, tc.want)
		}
	}
}

func TestDeriveBareNameFromURI(t *testing.T) {
	tests := []struct {
		uri  string
		want string
	}{
		// The MTP variant tag must survive the round-trip: this is the
		// bug that motivated the helper — the .gguf file in the HF repo
		// drops MTP, so the puller must derive the bare name from the
		// URI rather than from the file.
		{"hf://unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Q8_0", "Qwen3.6-35B-A3B-MTP-Q8_0"},
		// Canonical unsloth shapes round-trip with ResolveBareNameToURI.
		{"hf://unsloth/Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL", "Qwen3.5-122B-A10B-UD-Q4_K_XL"},
		{"hf://unsloth/Qwen2.5-7B-Instruct-GGUF/Q4_K_M", "Qwen2.5-7B-Instruct-Q4_K_M"},
		{"hf://unsloth/gemma-2-27b-it-GGUF/BF16", "gemma-2-27b-it-BF16"},
		// Non-unsloth orgs that follow the same convention.
		{"hf://someuser/MyModel-GGUF/Q5_K_M", "MyModel-Q5_K_M"},
		// Empty / non-matching: caller falls back to source filename.
		{"hf://unsloth/Qwen3.6-35B-A3B-MTP-GGUF", ""},
		{"hf://unsloth/no-suffix/Q4_K_M", ""},
		{"https://huggingface.co/x/resolve/main/y.gguf", ""},
		{"http://example.com/m.gguf", ""},
		{"", ""},
	}
	for _, tc := range tests {
		got := DeriveBareNameFromURI(tc.uri)
		if got != tc.want {
			t.Errorf("DeriveBareNameFromURI(%q) = %q, want %q", tc.uri, got, tc.want)
		}
	}
}

// TestRoundTrip pins the invariant the PullHandler relies on: pulling a
// bare name and pulling its expanded hf:// URI must land at the same
// on-disk basename.
func TestRoundTrip(t *testing.T) {
	names := []string{
		"Qwen3.6-35B-A3B-MTP-Q8_0",
		"Qwen3.5-122B-A10B-UD-Q4_K_XL",
		"Qwen2.5-7B-Instruct-Q4_K_M",
		"gemma-2-27b-it-BF16",
	}
	for _, name := range names {
		uri := ResolveBareNameToURI(name)
		got := DeriveBareNameFromURI(uri)
		if got != name {
			t.Errorf("round-trip %q -> %q -> %q", name, uri, got)
		}
	}
}

func TestIsURI(t *testing.T) {
	tests := []struct {
		in   string
		want bool
	}{
		{"hf://unsloth/x/q", true},
		{"https://example.com", true},
		{"http://example.com", true},
		{"Qwen3-Q4_K_M", false},
		{"", false},
	}
	for _, tc := range tests {
		if got := IsURI(tc.in); got != tc.want {
			t.Errorf("IsURI(%q) = %v, want %v", tc.in, got, tc.want)
		}
	}
}

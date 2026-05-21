// Package naming owns the bare-name ↔ hf:// URI mapping used to keep a
// user-typed model identifier (e.g. `Qwen3.6-35B-A3B-MTP-Q8_0`) in
// agreement with the on-disk basename across pull → /v1/models → /api/load.
//
// This is the single source of truth — clients (the tanrenai CLI, the
// hosted platform) and the GPU server itself should both import from here
// rather than maintaining parallel copies. The PullHandler uses it to
// auto-derive the destination basename when a caller passes an hf:// URI
// without an explicit `name` field, so the actual file in the HF repo
// (often missing the repo-level variant tag like `MTP`) doesn't end up
// becoming the on-disk identity the user has to remember.
package naming

import (
	"fmt"
	"regexp"
	"strings"
)

// unslothQuantSuffix matches a trailing GGUF quant suffix using the Unsloth
// naming convention. Captures the root (pre-quant part) and quant (with any
// `UD-` dynamic prefix kept so it can be passed verbatim in the hf:// URI).
//
//	Qwen3.5-122B-A10B-UD-Q4_K_XL   -> root=Qwen3.5-122B-A10B  quant=UD-Q4_K_XL
//	Qwen2.5-7B-Instruct-Q4_K_M     -> root=Qwen2.5-7B-Instruct quant=Q4_K_M
//	gemma-2-27b-it-BF16            -> root=gemma-2-27b-it      quant=BF16
var unslothQuantSuffix = regexp.MustCompile(`(?i)^(.+?)-((?:UD-)?(?:I?Q\d+(?:_\w+)*|F16|BF16|F32))$`)

// hfPullURI matches the canonical hf://<org>/<repo>-GGUF/<quant> pull URI
// produced by ResolveBareNameToURI. Captures the repo root (pre-`-GGUF`)
// and quant so callers can round-trip the URI back to a bare name.
var hfPullURI = regexp.MustCompile(`^hf://[^/]+/(.+?)-GGUF/(.+)$`)

// IsURI reports whether the input is already a pullable URI/URL rather
// than a bare model name needing resolution.
func IsURI(s string) bool {
	return strings.HasPrefix(s, "hf://") ||
		strings.HasPrefix(s, "https://") ||
		strings.HasPrefix(s, "http://")
}

// ResolveBareNameToURI guesses an hf:// URI for a bare model name using the
// Unsloth GGUF repo convention `<name>-GGUF/<quant>`. URIs/URLs pass through
// unchanged. Returns "" when the input has no recognizable quant suffix —
// callers should surface that as an error so the user gets a useful message
// instead of a silent misroute.
func ResolveBareNameToURI(name string) string {
	if IsURI(name) {
		return name
	}
	m := unslothQuantSuffix.FindStringSubmatch(name)
	if m == nil {
		return ""
	}
	root, quant := m[1], m[2]
	return fmt.Sprintf("hf://unsloth/%s-GGUF/%s", root, quant)
}

// DeriveBareNameFromURI is the inverse of ResolveBareNameToURI: given an
// hf://<org>/<repo>-GGUF/<quant> URI, returns the bare name
// `<repo>-<quant>` that callers should use as the on-disk basename.
//
// This matters because the actual GGUF inside the HF repo is often named
// without the repo-level variant tag (e.g. `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`
// ships `Qwen3.6-35B-A3B-Q8_0.gguf`, dropping the `MTP`). If the puller
// defaults to that filename, a subsequent /api/load by the user-typed name
// (`Qwen3.6-35B-A3B-MTP-Q8_0`) won't find the file. Deriving the bare name
// from the URI restores the round trip.
//
// Returns "" for non-hf:// URIs and for hf:// URIs that don't follow the
// canonical `-GGUF/<quant>` shape — callers should leave the destination
// name empty in that case and fall back to the source URL's filename.
func DeriveBareNameFromURI(uri string) string {
	m := hfPullURI.FindStringSubmatch(uri)
	if m == nil {
		return ""
	}
	return m[1] + "-" + m[2]
}

package handlers

import (
	"encoding/json"
	"net/http"

	"github.com/ThatCatDev/tanrenai-gpu/internal/buildinfo"
)

func Health(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]string{
		"status":  "ok",
		"version": buildinfo.Version,
	})
}

// Version reports the binary's build-time version identifier. Same string
// the /health endpoint returns under "version", but exposed on its own
// path so callers that want just the version don't have to parse health.
func Version(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]string{
		"version": buildinfo.Version,
	})
}

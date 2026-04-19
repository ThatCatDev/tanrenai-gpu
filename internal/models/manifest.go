package models

// ModelEntry represents a locally available model.
type ModelEntry struct {
	Name       string // display name (filename without .gguf)
	Path       string // full path to the GGUF file
	Size       int64  // file size in bytes
	ModifiedAt int64  // unix timestamp
}

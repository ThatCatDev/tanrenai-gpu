package server

import (
	"testing"

	"github.com/ThatCatDev/tanrenai-gpu/internal/gguf"
)

func TestKVCacheBytesPerToken(t *testing.T) {
	tests := []struct {
		name string
		arch gguf.Architecture
		want uint64
	}{
		{
			name: "GQA model",
			// 32 layers, 32 heads, 8 KV heads, head_dim = 4096/32 = 128.
			// 2 * 32 * 8 * 128 * 2 = 131072 bytes/token.
			arch: gguf.Architecture{
				BlockCount:      32,
				HeadCount:       32,
				HeadCountKV:     8,
				EmbeddingLength: 4096,
			},
			want: 2 * 32 * 8 * 128 * 2,
		},
		{
			name: "MHA falls back to head count for kv heads",
			arch: gguf.Architecture{
				BlockCount:      32,
				HeadCount:       32,
				HeadCountKV:     0, // unset → equals HeadCount
				EmbeddingLength: 4096,
			},
			want: 2 * 32 * 32 * 128 * 2,
		},
		{
			name: "missing fields yields zero",
			arch: gguf.Architecture{BlockCount: 32},
			want: 0,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := kvCacheBytesPerToken(&gguf.Metadata{Architecture: tc.arch})
			if got != tc.want {
				t.Errorf("kvCacheBytesPerToken = %d, want %d", got, tc.want)
			}
		})
	}
}

func TestStaticCap(t *testing.T) {
	if got := staticCap(262144, "test"); got != maxAutoCtxSize {
		t.Errorf("over-cap ctx = %d, want %d", got, maxAutoCtxSize)
	}
	if got := staticCap(8192, "test"); got != 8192 {
		t.Errorf("under-cap ctx = %d, want 8192", got)
	}
}

// autoDetectCtxSize must use the static cap (not the VRAM path) when the KV
// cache lives on CPU or when metadata is too sparse to size the cache.
func TestAutoDetectCtxSizeFallbacks(t *testing.T) {
	bigCtx := &gguf.Metadata{Architecture: gguf.Architecture{
		ContextLength:   262144,
		BlockCount:      32,
		HeadCount:       32,
		HeadCountKV:     8,
		EmbeddingLength: 4096,
	}}

	if got := autoDetectCtxSize(bigCtx, "/nonexistent.gguf", true); got != maxAutoCtxSize {
		t.Errorf("kv-on-cpu ctx = %d, want static cap %d", got, maxAutoCtxSize)
	}

	sparse := &gguf.Metadata{Architecture: gguf.Architecture{ContextLength: 262144}}
	if got := autoDetectCtxSize(sparse, "/nonexistent.gguf", false); got != maxAutoCtxSize {
		t.Errorf("sparse-metadata ctx = %d, want static cap %d", got, maxAutoCtxSize)
	}
}

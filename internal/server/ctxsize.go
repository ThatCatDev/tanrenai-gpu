package server

import (
	"bufio"
	"log/slog"
	"os"
	"os/exec"
	"strconv"
	"strings"

	"github.com/ThatCatDev/tanrenai-gpu/internal/gguf"
)

// maxAutoCtxSize is the fallback ceiling for auto-detected context length when
// we can't size it against real VRAM (no GPU, nvidia-smi unavailable, KV cache
// kept on CPU, or incomplete GGUF metadata). Modern models advertise 256K+
// training contexts whose KV cache is tens of GB; 32K is a safe default for
// chat workloads. The VRAM-aware path below raises this when the card can
// actually hold a larger KV cache.
const maxAutoCtxSize = 32768

// minAutoCtxSize is the floor we won't shrink below even when VRAM is tight.
// If the card genuinely can't hold this, llama-server surfaces the OOM itself.
const minAutoCtxSize = 4096

// ctxGranularity rounds the computed context length down to a clean multiple.
const ctxGranularity = 256

// kvCacheReserveFraction holds back a slice of free VRAM for llama.cpp's
// compute buffers, CUDA context, and fragmentation headroom — i.e. everything
// that isn't model weights or the KV cache itself.
const kvCacheReserveFraction = 0.10

// autoDetectCtxSize picks a context length for a model whose request left the
// context at the default. It starts from the GGUF-advertised training context
// and, when the KV cache lives on GPU and VRAM can be measured, clamps to what
// actually fits in free VRAM after weights load. Otherwise it falls back to the
// static maxAutoCtxSize ceiling. The returned value is logged with its reason.
func autoDetectCtxSize(meta *gguf.Metadata, modelPath string, kvOnCPU bool) int {
	ggufCtx := int(meta.Architecture.ContextLength)

	// KV on CPU (--no-kv-offload) means the cache spends RAM, not VRAM, so the
	// VRAM budget doesn't bound it. Keep the conservative static cap.
	if kvOnCPU {
		return staticCap(ggufCtx, "kv cache on cpu")
	}

	perToken := kvCacheBytesPerToken(meta)
	if perToken == 0 {
		// Metadata too sparse to size the KV cache — fall back.
		return staticCap(ggufCtx, "incomplete gguf metadata")
	}

	freeVRAM, ok := freeVRAMBytes()
	if !ok {
		return staticCap(ggufCtx, "vram unmeasured")
	}

	// Weights occupy roughly the on-disk file size once loaded onto the GPU.
	var weightBytes uint64
	if fi, err := os.Stat(modelPath); err == nil {
		weightBytes = uint64(fi.Size())
	}

	reserve := uint64(float64(freeVRAM) * kvCacheReserveFraction)
	used := weightBytes + reserve
	if used >= freeVRAM {
		// No room for any KV cache after weights + overhead. Let llama-server
		// try the floor and report the OOM itself rather than silently picking
		// something even smaller.
		slog.Warn("insufficient VRAM for KV cache after weights — using floor",
			"free_vram_bytes", freeVRAM, "weight_bytes", weightBytes, "ctx_size", minAutoCtxSize)

		return minAutoCtxSize
	}

	kvBudget := freeVRAM - used
	fitCtx := int(kvBudget / perToken)
	fitCtx -= fitCtx % ctxGranularity
	if fitCtx < minAutoCtxSize {
		fitCtx = minAutoCtxSize
	}

	// Never exceed the model's trained context.
	ctxSize := fitCtx
	if ctxSize > ggufCtx {
		ctxSize = ggufCtx
	}

	slog.Info("sized context length to fit VRAM",
		"gguf_ctx_size", ggufCtx,
		"ctx_size", ctxSize,
		"free_vram_bytes", freeVRAM,
		"weight_bytes", weightBytes,
		"kv_bytes_per_token", perToken,
		"kv_cache_bytes", uint64(ctxSize)*perToken)

	return ctxSize
}

// staticCap clamps to the conservative maxAutoCtxSize ceiling and logs why the
// VRAM-aware path was skipped.
func staticCap(ggufCtx int, reason string) int {
	if ggufCtx > maxAutoCtxSize {
		slog.Info("clamping auto-detected context length",
			"gguf_ctx_size", ggufCtx, "ctx_size", maxAutoCtxSize, "reason", reason)

		return maxAutoCtxSize
	}

	slog.Info("auto-detected context length from GGUF", "ctx_size", ggufCtx)

	return ggufCtx
}

// kvCacheBytesPerToken returns the VRAM cost of one token of KV cache across all
// layers, assuming an f16 cache (llama.cpp's default). It returns 0 when the
// metadata lacks the fields needed to compute it.
//
//	bytes/token = 2 (K and V) × n_layers × n_kv_heads × head_dim × 2 (f16)
//	head_dim    = embedding_length / n_heads
func kvCacheBytesPerToken(meta *gguf.Metadata) uint64 {
	a := meta.Architecture
	nLayers := a.BlockCount
	nHeads := a.HeadCount
	nKVHeads := a.HeadCountKV
	if nKVHeads == 0 {
		// Multi-head attention (no GQA): KV heads equal query heads.
		nKVHeads = nHeads
	}
	if nLayers == 0 || nHeads == 0 || nKVHeads == 0 || a.EmbeddingLength == 0 {
		return 0
	}

	headDim := a.EmbeddingLength / nHeads
	if headDim == 0 {
		return 0
	}

	const kvElemBytes = 2 // f16
	const kAndV = 2

	return kAndV * nLayers * nKVHeads * headDim * kvElemBytes
}

// freeVRAMBytes returns the total free VRAM across all NVIDIA GPUs, queried via
// nvidia-smi. The second return is false when no GPU memory could be read (no
// nvidia-smi, no NVIDIA GPU, or a parse failure), signalling callers to fall
// back to a static policy.
func freeVRAMBytes() (uint64, bool) {
	out, err := exec.Command("nvidia-smi",
		"--query-gpu=memory.free", "--format=csv,noheader,nounits").Output()
	if err != nil {
		return 0, false
	}

	var total uint64
	var any bool
	scanner := bufio.NewScanner(strings.NewReader(string(out)))
	for scanner.Scan() {
		field := strings.TrimSpace(scanner.Text())
		if field == "" {
			continue
		}
		mib, err := strconv.ParseUint(field, 10, 64)
		if err != nil {
			continue
		}
		total += mib * 1024 * 1024 // MiB → bytes
		any = true
	}

	if !any {
		return 0, false
	}

	return total, true
}

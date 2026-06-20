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

// maxAutoParallel caps how many concurrent slots the auto-slot path will pack
// into one instance, so a huge card doesn't spawn an unreasonable number of
// sequences (each carries fixed per-slot overhead in llama-server).
const maxAutoParallel = 64

// kvCacheReserveFraction holds back a slice of free VRAM for llama.cpp's
// compute buffers, CUDA context, and fragmentation headroom — i.e. everything
// that isn't model weights or the KV cache itself.
//
// This is held back BEFORE the slot count is computed, so a bigger reserve =
// fewer slots packed = more runtime headroom. 0.10 proved too thin: an A100
// PCIE 80GB packed to 13 slots left only ~8GB for compute buffers, which scale
// with the number of concurrent slots, and llama-server crash-looped under load
// (CUDA abort in server_queue::start_loop). 0.20 roughly halves the slot count
// on a large card and leaves real margin. Tune upward (toward 0.25) if OOM
// crashes persist on a given deployment.
const kvCacheReserveFraction = 0.20

// ctxPlan is the chosen context configuration for a model load: the total
// context window passed to llama-server (--ctx-size) and the number of
// concurrent sequence slots (--parallel). In llama-server, --ctx-size is the
// total shared across slots, so per-user context = CtxSize / Parallel.
type ctxPlan struct {
	CtxSize  int
	Parallel int
}

// planContext decides the context window (and, in multi-user mode, the number
// of concurrent slots) for a model whose request left the context at the
// default. When ctxPerUser is 0 it falls back to single-slot VRAM-aware sizing;
// when ctxPerUser > 0 it packs as many per-user slots as free VRAM allows.
func planContext(meta *gguf.Metadata, modelPath string, kvOnCPU bool, ctxPerUser int) ctxPlan {
	if ctxPerUser <= 0 {
		return ctxPlan{CtxSize: autoDetectCtxSize(meta, modelPath, kvOnCPU), Parallel: 1}
	}

	return planMultiSlot(meta, modelPath, kvOnCPU, ctxPerUser)
}

// planMultiSlot implements "fixed per-user context, auto slots": it holds each
// user's context window at ctxPerUser and fits as many slots as the VRAM KV
// budget allows, so the per-user window stays predictable while the supported
// user count flexes with the GPU and model.
func planMultiSlot(meta *gguf.Metadata, modelPath string, kvOnCPU bool, ctxPerUser int) ctxPlan {
	ggufCtx := int(meta.Architecture.ContextLength)

	// A single slot can't exceed the model's trained context.
	if ctxPerUser > ggufCtx {
		slog.Info("clamping per-user context to trained context",
			"requested", ctxPerUser, "ctx_per_user", ggufCtx)
		ctxPerUser = ggufCtx
	}
	ctxPerUser -= ctxPerUser % ctxGranularity
	if ctxPerUser < minAutoCtxSize {
		ctxPerUser = minAutoCtxSize
	}

	// Fall back to safe single-slot sizing whenever we can't compute a slot
	// count from real VRAM.
	singleSlot := func(reason string) ctxPlan {
		slog.Info("auto-slot disabled — serving a single slot",
			"reason", reason, "ctx_size", autoDetectCtxSize(meta, modelPath, kvOnCPU))

		return ctxPlan{CtxSize: autoDetectCtxSize(meta, modelPath, kvOnCPU), Parallel: 1}
	}

	if kvOnCPU {
		return singleSlot("kv cache on cpu")
	}
	perToken := kvCacheBytesPerToken(meta)
	if perToken == 0 {
		return singleSlot("incomplete gguf metadata")
	}
	budget, ok := kvBudgetBytes(modelPath)
	if !ok {
		return singleSlot("vram unmeasured")
	}

	perSlotBytes := uint64(ctxPerUser) * perToken
	slots := int(budget / perSlotBytes)
	if slots < 1 {
		// Not even one full per-user window fits — fall back to a single slot
		// sized to whatever VRAM can hold.
		return singleSlot("vram fits less than one per-user context")
	}
	if slots > maxAutoParallel {
		slots = maxAutoParallel
	}

	total := slots * ctxPerUser
	slog.Info("sized parallel slots to fit VRAM",
		"parallel", slots,
		"ctx_per_user", ctxPerUser,
		"ctx_size", total,
		"kv_bytes_per_token", perToken,
		"kv_budget_bytes", budget,
		"kv_cache_bytes", uint64(total)*perToken)

	return ctxPlan{CtxSize: total, Parallel: slots}
}

// autoDetectCtxSize picks a single-slot context length: it starts from the
// GGUF-advertised training context and, when the KV cache lives on GPU and VRAM
// can be measured, clamps to what actually fits in free VRAM after weights
// load. Otherwise it falls back to the static maxAutoCtxSize ceiling.
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

	budget, ok := kvBudgetBytes(modelPath)
	if !ok {
		return staticCap(ggufCtx, "vram unmeasured")
	}
	if budget == 0 {
		// No room for any KV cache after weights + overhead. Let llama-server
		// try the floor and report the OOM itself rather than silently picking
		// something even smaller.
		slog.Warn("insufficient VRAM for KV cache after weights — using floor",
			"ctx_size", minAutoCtxSize)

		return minAutoCtxSize
	}

	fitCtx := int(budget / perToken)
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

// kvBudgetBytes returns the VRAM available for the KV cache: free VRAM minus the
// model's weight footprint (≈ on-disk size) minus a reserve for compute buffers
// and fragmentation. The second return is false when VRAM can't be measured; a
// zero budget (measured, but no room left after weights) returns (0, true).
func kvBudgetBytes(modelPath string) (uint64, bool) {
	freeVRAM, ok := freeVRAMBytes()
	if !ok {
		return 0, false
	}

	var weightBytes uint64
	if fi, err := os.Stat(modelPath); err == nil {
		weightBytes = uint64(fi.Size())
	}

	reserve := uint64(float64(freeVRAM) * kvCacheReserveFraction)
	used := weightBytes + reserve
	if used >= freeVRAM {
		return 0, true
	}

	return freeVRAM - used, true
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

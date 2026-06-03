# Serving multiple users from one GPU instance

How `tanrenai-gpu` handles concurrency today, and the plan to let several
users share a single GPU instance.

## How it works today

`tanrenai-gpu` is an Ollama/OpenAI-compatible server, but it is **not** Ollama —
it's a thin server wrapping a single `llama-server` subprocess. Two structural
facts drive everything about multi-user:

1. **One model loaded at a time.** The server holds a single `s.runner`. A chat
   request naming a different model triggers `LoadModel`, which *closes the
   current subprocess and starts a new one*
   (`internal/server/handlers/chat.go`, `internal/server/server.go`).
2. **One inference slot.** The runner does not pass `--parallel` to
   `llama-server` (`internal/runner/process.go`), so it runs with the default
   single slot — no continuous batching. Concurrent requests serialize.
3. **No lock around the model swap.** `LoadModel` mutates `s.runner` with no
   mutex, so concurrent loads can race.

Consequences:

| Scenario | Result today |
|---|---|
| Many users, **same model**, light load | Works, but requests **serialize** through one slot |
| Many users, **same model**, heavy load | Bottlenecked to one-at-a-time; latency climbs |
| Users want **different models** at once | ❌ Server thrashes — each differing request reloads the model |

## Platform context (`tanrenai-platform`)

The platform is **per-user by design**: `instance.Manager` "manages per-user GPU
instance lifecycles" — each user gets their own vast.ai instance, provisioned on
demand (`Provision(ctx, user, modelName)`), with a per-user idle timer that
tears it down. The serve command is hard-coded in
`internal/network/headscale.go`:

```
nohup tanrenai-gpu serve --host 0.0.0.0 --port %d > /var/log/tanrenai-gpu.log 2>&1 &
```

So "multiple users sharing one GPU server" is a real shift from the current
one-instance-per-user model. We split it into phases.

## The model: just split the context

In `llama-server`, `--ctx-size` is the **total** context shared across
`--parallel N` slots — each slot (≈ one concurrent user) gets `ctx_size / N`.
So the design is simply:

- Pick a per-user context size `C` (good default: **16K** for chat; 8K for
  short Q&A; 32K if VRAM is plentiful).
- Pick a number of concurrent slots `N`.
- Run with `--parallel N --ctx-size N×C`.

VRAM only has to hold the total KV cache. KV per token is
`2 (K+V) × layers × kv_heads × head_dim × 2 bytes (f16)`, so:

```
max_users (N) = free_KV_VRAM ÷ (per_user_ctx × KV_per_token)
```

Worked examples on an 80 GB card (Q4 weights, f16 KV):

| Model | KV/token | Free for KV | Users @16K | Users @32K |
|---|---|---|---|---|
| Llama-3-8B | 128 KiB | ~73 GB | ~36 | ~18 |
| Qwen3-32B | 256 KiB | ~58 GB | ~14 | ~7 |
| 70B | 320 KiB | ~38 GB | ~7 | ~3 |

This composes with the VRAM-aware context sizing already in the server (PR #11):
the auto-sizer measures the KV budget; we just cap the per-model total at
`trained_context × parallel` so each slot can still reach the model's full
trained window when VRAM allows.

## Phase 1 — cheapest win: continuous batching on one instance

Goal: one instance serves `N` concurrent requests **on the same model**.

**`tanrenai-gpu`**
- Add a `Parallel int` option → emit `--parallel N` to `llama-server`
  (`runner.Options`, `process.go`, `config.Config`, `cmd/serve.go`,
  `pkg/serve`).
- Optionally add an explicit `--ctx-size-per-slot` knob, or keep the
  "split total ctx across slots" behavior and let the VRAM sizer pick the total.
- Fold `parallel` into `autoDetectCtxSize`: ceiling becomes
  `trained_context × parallel`, still bounded by VRAM.
- Add a mutex around the `LoadModel` swap so concurrent loads can't double-spawn.

**`tanrenai-platform`**
- Thread `--parallel N` (and optionally per-slot ctx) into the serve command in
  `headscale.go`, driven by a config/env knob (e.g. `GPU_PARALLEL_SLOTS`).

This helps immediately when a single user fires concurrent requests (multiple
tabs, an app making parallel calls). True cross-user sharing needs Phase 2.

## Phase 2 — cross-user sharing (larger change)

To let *different users* hit *one* instance, `tanrenai-platform`'s `Manager`
must stop being strictly per-user: introduce a shared/pooled instance that
multiple users route to (by model), with the idle timer keyed on the pool
rather than a single user. This is the bigger lift and is intentionally
deferred.

## Phase 3 — different models at once

Either run multiple instances (one model each) behind a router (LiteLLM / nginx)
that dispatches by model name, or implement multi-model loading inside one
instance (a `model → runner` map with LRU/VRAM-budgeted eviction).

## Open decision

How to parametrize Phase 1 (see chat thread):
- **(A) Fixed per-user context, derive slots:** operator sets `C` (e.g. 16K);
  the server fits as many slots as VRAM allows. Predictable per-user window.
- **(B) Fixed slot count, derive per-user:** operator sets `N` (`--parallel`);
  total ctx is auto-sized to VRAM and split → per-user = total/N.
- **(C) Both explicit:** operator sets `N` and `C`; total = `N×C`; error if it
  won't fit.

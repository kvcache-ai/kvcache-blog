---
title: "HiCache L3 for Hybrid Attention Models: SWA Checkpoints, Group Semantics, and 78% Storage Reduction"
summary: "Hybrid attention architectures couple compressed MLA with a large sliding-window cache — and when the full SWA state is offloaded page-by-page to Mooncake Store, it dominates L3 cost at 5x the MLA footprint. This post covers the full HiCache L3 optimization stack: periodic SWA checkpoints in the Unified Radix Tree, window-aware LRU refresh, leaf-level lock pruning, C4/C128 compress-state offload, draft KV pool registration, and Mooncake group semantics for coherent multi-object eviction — together reducing storage from 18 GB to 4 GB per million tokens while preserving 90%+ cache hit rates on agent workloads."
date: 2026-06-16
authors:
  - Mooncake community
tags:
  - HiCache
  - Mooncake
  - Mooncake Store
  - KVCache
  - L3
  - SWA
  - Group Semantics

draft: false
showathome: true
commentable: false
home_weight: 202606164
image:
  alt_text: "HiCache L3 SWA checkpoint and group semantics for hybrid attention models"
---

When you offload a million tokens of KVCache to distributed storage for DeepSeek V4, most of the bytes are not the compressed MLA attention state. They are the sliding-window attention (SWA) cache — the local-window KV that the model uses for its first few layers.

On current hardware, with the default per-page offload strategy, storing one million tokens of DeepSeek V4 Flash KVCache in the L3 tier (Mooncake Store) costs roughly 18 GB. The compressed MLA portion — the part that actually captures long-range context — accounts for about 3.6 GB of that. The rest is SWA state that exists only to serve the local attention window.

That ratio is uncomfortable. It means the L3 tier is dominated by state that, by design, only matters for a small neighborhood of tokens. If you could recompute that state on cache hit instead of storing it, the storage cost drops dramatically.

This post covers the full set of changes — from the periodic checkpoint mechanism itself, through the window-aware eviction fixes that make it work in practice, to the agent-workload benchmarks that validate the tradeoff.

## Background: DeepSeek V4's hybrid KV architecture

DeepSeek V4 uses a hybrid attention design that combines two fundamentally different KV storage patterns:

**Full-attention layers** use Multi-head Latent Attention (MLA), which compresses the KV cache into a low-rank latent space. For a million tokens, the MLA cache is approximately 3.6 GB — compact because the per-token state is a short latent vector rather than full key-value heads.

**Sliding-window layers** (the first few transformer layers) use standard multi-head attention over a fixed window. These layers maintain full-precision KV for the last `sliding_window_size` tokens (128 tokens for DeepSeek V4). However, in the radix tree, the *stored* SWA state for a prefix spans the entire prefix, not just the active window — because on cache hit, you need the window relative to the hit point, which is different from the window relative to the latest token.

The asymmetry is stark: the SWA cache is approximately 5x larger than the MLA cache for the same prefix. At million-token scale, that 5x turns 3.6 GB into 18 GB of L3 consumption per sequence.

For operators running shared Mooncake Store pools across many instances, this is the difference between "L3 is affordable" and "L3 is a storage capacity problem."

## The Unified Radix Tree and SWA

SGLang's `UnifiedRadixCache` manages KVCache as a radix tree where each node holds a range of token positions and their associated GPU memory blocks. For hybrid models like DeepSeek V4, the tree tracks *two* pools simultaneously:

- A **full-attention KV pool** for MLA blocks — these follow the standard paged KV layout.
- A **SWA KV pool** for sliding-window blocks — these have their own allocator (`SWATokenToKVPoolAllocator`), their own lock semantics, and their own eviction LRU.

The `SWAComponent` within the unified tree owns the sliding-window-specific logic: which nodes have live SWA state on device, which are backed up to host, which are evicted. It maintains a separate `swa_protected_size` counter and its own eviction tail.

This separation is critical because the correct eviction policy for SWA is *not* the same as for full-attention KV. Full-attention state is valuable for as long as the prefix is alive. SWA state is only valuable for the last `sliding_window_size` tokens relative to any active decode position. A vanilla LRU that treats both the same will make the wrong choice.

## Problem 1: SWA nodes that should be dead stay alive

The first problem (fixed in [#26615](https://github.com/sgl-project/sglang/pull/26615)) is subtle. On every prefix-cache match — and on every walk-down during insert — all matched ancestors are bumped to MRU position in the LRU. This is correct for full-attention KV, where any matched prefix is genuinely useful. But for SWA, only the last `sliding_window_size` tokens are reachable from any current decode step.

The consequence: very old SWA nodes (outside the active window of every live request) keep getting refreshed to MRU and never get evicted. Genuinely-hot SWA cache from other branches gets pushed toward the LRU tail and is evicted first. Under multi-turn workloads, the SWA pool fills with cache that can no longer contribute to any hit, hit rate degrades, and eviction churn goes up.

The fix is **window-aware LRU refresh**: when walking matched ancestors during a prefix-cache hit, only bump SWA nodes that fall within `sliding_window_size` tokens of the match point. Nodes outside the window do not get their LRU timestamp refreshed and naturally age out.

## Problem 2: leaf over-protection at 512x

The second problem (fixed in [#26777](https://github.com/sgl-project/sglang/pull/26777)) is an interaction between chunked prefill and SWA lock accounting.

When a request locks a radix tree leaf, `inc_lock_ref` protects `len(leaf.value)` SWA tokens for that leaf — even though SWA only needs the last `sliding_window_size` tokens. With `--chunked-prefill-size 65536` and `sliding_window=128` (DeepSeek V4's parameters), a single leaf could lock **65,536 SWA slots instead of 128** — a 512x over-protection.

This inflates `swa_protected_size` by approximately `chunked_prefill_size / sliding_window_size`, causing premature SWA pool exhaustion and retract thrashing. The fix is `SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT`: when enabled, the tree splits leaves so that the locked SWA region is at most `sliding_window_size` tokens, matching the actual SWA requirement.

## Problem 3: locked SWA leaves stranded in eviction

A third issue (fixed in [#28161](https://github.com/sgl-project/sglang/pull/28161)) appears under hybrid SWA+Mamba workloads. The early-release optimization (`SGLANG_OPT_SWA_RELEASE_LEAF_LOCK_AFTER_WINDOW`) was silently a no-op on `UnifiedRadixCache` because the `dec_swa_lock_only` method only existed on the legacy backend. This left full-locked leaves stranded in the SWA LRU; once SWA eviction picked one, the `_cascade_evict` assert `cd.lock_ref == 0` would fire.

The fix ports `dec_swa_lock_only` to the unified cache: walk ancestors to the `swa_uuid_for_lock` boundary, free SWA slots, detach from SWA LRU on leaves, and ensure the `full_lock_ref >= swa_lock_ref >= mamba_lock_ref` invariant survives across co-located lower-priority locks.

## The mechanism: periodic SWA checkpoints

With the above eviction correctness issues resolved, the L3 checkpoint mechanism ([#27557](https://github.com/sgl-project/sglang/pull/27557)) becomes viable.

The insight is that SWA state does not need to be stored at page granularity. It only needs to be stored at *checkpoint* granularity — at intervals large enough that on cache hit, the system can recompute the SWA state for the tokens between the last checkpoint and the hit point.

The implementation lives in `SWAComponent` within `swa_component.py`:

- A new `--swa-checkpoint-interval` server argument (exposed through `server_args.py` and `cache_init_params.py`) sets the checkpoint period in tokens.
- Instead of offloading SWA state for every page, `SWAComponent` writes a full SWA snapshot to L3 only every N tokens.
- On cache hit, the system loads the compressed MLA state (always stored per-page) and the most recent SWA checkpoint that precedes the hit point.
- For the tokens between the checkpoint and the hit point, the system aligns the hit to the checkpoint boundary — effectively requiring the matched prefix to be a multiple of the checkpoint interval. Tokens in the gap are re-prefilled with only the SWA layers active.

The first version (v1) does *not* implement SWA-only partial prefill for the gap. It simply aligns cache hits to checkpoint boundaries, which means a hit that falls between two checkpoints backs up to the previous checkpoint and re-runs full prefill for the gap tokens. This is simpler to implement correctly and already delivers the storage savings; SWA-only recomputation is a follow-up optimization.

## The numbers

Measured on DeepSeek V4 Flash with context-parallel 4 on Hopper GPUs, using Mooncake Store as the L3 backend, with a 1-million-token input:

| SWA Checkpoint Interval | L3 Memory | Reduction | TTFT on Hit (avg@5) | TTFT Delta |
|-------------------------|-----------|-----------|---------------------|------------|
| 256 tokens (default)    | 18.25 GB  | baseline  | 3039 ms             | baseline   |
| 2,048 tokens            | 5.74 GB   | −68%      | 3536 ms             | +16%       |
| 16,384 tokens           | 4.03 GB   | −78%      | 4660 ms             | +53%       |

At 2K interval: 12 GB saved per million tokens; half a second more TTFT on a hit that was already 3 seconds. For most serving scenarios, this is a strong default.

At 16K interval: approaching the theoretical minimum (just MLA cache); practical for workloads where storage cost dominates and cache hits are infrequent.

## Agent workloads: where SWA checkpoints transform the picture

The L3 checkpoint numbers above are for a single long-context fill. The more dramatic results come from agent workloads, where the combination of SWA checkpoint + suffix preservation ([#26907](https://github.com/sgl-project/sglang/pull/26907)) changes the regime entirely.

PR #26907 introduces a complementary mechanism: **preserving a trailing SWA suffix during device eviction**. Instead of evicting all SWA state when memory pressure hits, the eviction pass trims internal nodes to a page-aligned trailing sliding-window suffix — keeping just enough SWA on device to service a subsequent cache hit without a full reload.

The agent benchmark results on DeepSeek V4 (GB300, 1P/1D, concurrency 30, HiCache off) are striking:

| SWA Ratio | Variant | Cache Hit | Output tok/s | TTFT p90 |
|-----------|---------|-----------|--------------|----------|
| 0.015 | Baseline | 8.18% | 206.67 | 163.55 s |
| 0.015 | Checkpoint interval=81920 | 94.49% | 833.02 | 5.68 s |
| 0.015 | Structural checkpoints only | 89.36% | 706.04 | 21.25 s |

The cache hit rate jumps from **8% to 94%**. Output throughput quadruples. TTFT p90 drops from **163 seconds to 5.68 seconds**. This is not a marginal improvement — it is a regime change.

The mechanism: by keeping sparse SWA checkpoints (rather than dense per-page state), more of the limited SWA pool is available for genuinely-hot cache entries. The entries that get evicted now have checkpoint stubs that allow fast reload. The net effect is that almost all agent turn-by-turn requests hit cache instead of doing full prefill.

With HiCache enabled (3P/1D, concurrency 500), the improvement is more modest because baseline cache hit is already high, but still meaningful:

| SWA Ratio | Variant | Cache Hit | Output tok/s | TTFT p90 |
|-----------|---------|-----------|--------------|----------|
| 0.01 | Baseline | 89.47% | 5,216.81 | 127.55 s |
| 0.01 | Checkpoint interval=81920 | 92.19% | 6,488.80 | 91.53 s |

Even at 90%+ baseline hit rate, checkpoint + suffix preservation squeezes out another 24% throughput and 28% TTFT reduction.

## The SWA event metadata for distributed coordination

For disaggregated deployments where the L3 tier is a shared Mooncake Store pool, the system needs to communicate SWA checkpoint state across nodes. PR [#26579](https://github.com/sgl-project/sglang/pull/26579) extends the KV event protocol with SWA-aware metadata:

- `BlockStored` events gain optional `swa_sliding_window_size` and block-scoped `swa_valid_from` fields.
- When SWA cache state changes (including tombstone splits during eviction), the unified cache emits metadata refresh events.
- `swa_valid_from` is block/node-scoped: set when that block's own SWA state is missing or tombstoned; cleared when a refresh restores it.

The consumer contract: matching a longer prefix should aggregate metadata along the matched block path and use the deepest non-empty `swa_valid_from` together with `swa_sliding_window_size` to know how far back recomputation is needed. Consumers that do not understand the new metadata ignore the trailing fields and keep existing full-attention-only behavior.

This is what lets the Mooncake Store conductor, or a remote decode instance in a PD-disaggregated deployment, know: "this prefix is cached in L3, but the SWA state is only stored at checkpoint 491520 — if you hit at token 524288, you need to recompute SWA for 32K tokens after reload."

## PD disaggregation interaction

DeepSeek V4's hybrid SWA creates a specific constraint for prefill-decode (PD) disaggregation. PR [#26689](https://github.com/sgl-project/sglang/pull/26689) addresses the admission issue:

Before the fix, `PrefillBootstrapQueue` unconditionally clamped `max_total_num_tokens` to `swa_max_total_num_tokens` for hybrid-SWA models. That made valid long-context DeepSeek V4 PD requests fail admission before chunked prefill could run.

The fix: when `chunked_prefill_size > 0`, cap PD prefill admission by `full_max_total_num_tokens` (the MLA pool budget), because the prefill worker only needs SWA capacity for the active chunk, not the full token budget. The SWA pool serves a sliding window; it never holds the entire prefix simultaneously.

This is a prerequisite for L3 checkpoints in disaggregated mode: without it, million-token prefills on the prefill side would be rejected at admission.

## The complete picture: what controls what

For operators, the tuning knobs are:

**`--swa-checkpoint-interval N`** (server arg): write SWA snapshot to L3 every N tokens. Default 256 (dense). Set to 2048 for the recommended storage/TTFT tradeoff; set to 16384 for maximum storage savings.

**`SGLANG_SWA_CACHE_CHECKPOINT_MIN_TOKEN_INTERVAL`** (env var from #26907): controls on-device SWA suffix retention during eviction. Set to 0 to skip the SWA checkpoint soft pass (dense eviction behavior). Set to -1 for structural checkpoints only. Set to a positive token interval (e.g., 81920) to retain sparse trailing window checkpoints under device memory pressure.

**`SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT`** (env var, default True since #26777): prevent 512x SWA over-protection from chunked prefill leaves.

**`SGLANG_OPT_SWA_RELEASE_LEAF_LOCK_AFTER_WINDOW`** (env var from #28161): early-release SWA locks on leaves once the sliding window has passed them, freeing SWA pool capacity sooner.

These interact as a stack:
1. Leaf split + early release → SWA pool utilization improves → more room for hot entries.
2. Window-aware LRU → stale SWA nodes evict correctly → hot entries stay.
3. Suffix preservation → evicted nodes keep a checkpoint stub → faster reload.
4. L3 periodic checkpoint → storage footprint drops 68–78% → L3 becomes affordable at scale.

Each layer is independently useful. Together they are what makes million-token DeepSeek V4 serving work economically.

## Beyond SWA: C4 and C128 compress states in L3

SWA is the largest contributor to L3 storage cost, but it is not the only piece of DeepSeek V4 state that interacts with offload. DeepSeek V4's sparse attention (DSA) produces two additional per-request state buffers:

- **C4 state**: a compressed per-token attention score buffer used by the sparse indexer. It tracks which blocks are "hot" in the attention pattern and drives the top-k selection on each decode step. C4 lives in a sliding-window-like memory pool, indexed through `full_to_swa_index_mapping`.
- **C128 state**: a per-request compressed state accumulated over 128-token chunks. It holds the coarse-grained indexer context needed by the DSA decode path.

Both C4 and C128 were originally stored in FP32. On HBM-constrained hardware (H20 96 GB), this is expensive. PR [#24041](https://github.com/sgl-project/sglang/pull/24041) introduced `SGLANG_DSV4_COMPRESS_STATE_DTYPE=bf16`, halving C4 and C128 storage with negligible accuracy impact (GSM8K 0.948, MMLU 0.896 — matching FP32 baseline).

For L3 offloading, the critical question is: *can these states be offloaded alongside the KV cache?*

The answer depends on how the states are indexed. PR [#28612](https://github.com/sgl-project/sglang/pull/28612) exposed a subtle correctness bug: the online C128 + MTP path used `full_to_swa_index_mapping` to derive C128 state slots:

```python
full_loc = req_to_token[rid][chunk_start]
swa_loc = full_to_swa[full_loc]
main_slot = swa_loc / swa_page_size
```

This was unsafe because the radix cache can keep the full KV prefix alive while the SWA sidecar — and therefore the `full_to_swa` mapping — has already been tombstoned or freed. On a multi-turn cache hit, the C128 path could read slot 0, an old slot, or a reused slot.

The fix decouples C128 state from SWA mapping entirely:

- **C128** now uses request-scoped layout: `req_pool_idx * ring_size + position % ring_size`. This makes C128 state self-contained per request and safe to serialize independently.
- **C4** remains on the SWA-based sliding-window path (its window semantics align with SWA's lifecycle).

This decoupling is a prerequisite for clean L3 offload of C128 state: because C128 is now request-scoped rather than SWA-indexed, the L3 storage layer can serialize it as a contiguous buffer alongside the MLA cache without depending on SWA liveness.

For the no-prefix ragged prefill path ([#25400](https://github.com/sgl-project/sglang/pull/25400)), the system builds unified BF16 KV and packed ragged indices across C0/C4/C128 layers — demonstrating that all three state types can coexist in a single serializable metadata format.

## NSA indexer and L3: the serialization frontier

DeepSeek V3.2 introduced Native Sparse Attention (NSA) — a model-native sparsity pattern where the model itself decides at training time which tokens to attend to. The NSA indexer maintains a `index_k_with_scale_buffer` that records per-layer indexer state.

Early work on L3 offloading for NSA models ([#18637](https://github.com/sgl-project/sglang/pull/18637)) added MoonCake Store integration for DeepSeek V3.2, including indexer key serialization. However, a safety concern emerged ([#20880](https://github.com/sgl-project/sglang/pull/20880)): the L3 storage layer only handles MHA K/V and MLA latent-K formats but does not generically serialize the NSA indexer cache. Without proper handling, L3 prefetch could cause silent data corruption and shape mismatch crashes.

For DeepSeek V4, the situation is different from V3.2:

- V4 uses DSA (DeepSeek Sparse Attention) rather than V3.2's NSA. DSA's indexer state is captured through C4 and C128 compress states, which are structured buffers with known shapes.
- The hierarchical indexer work (HISA, [#24672](https://github.com/sgl-project/sglang/pull/24672)) introduces a pool-K cache (`HisaNSATokenToKVPool`) with per-layer mean-pooled blocks — another structured state that needs L3 serialization.
- L2 (host DRAM) offload is unaffected — it copies raw GPU pages without format awareness.

The path forward for V4 L3 is clear: C4 slides with SWA (already handled by SWA checkpoints), C128 is now request-scoped (serializable independently), and the MLA latent cache is already the primary L3 payload. The remaining gap is the HISA pool-K cache for deployments using the hierarchical indexer — that requires extending the L3 storage format to carry pool-block metadata.

## Speculative decoding: draft KV pool and HiCache

DeepSeek V4 supports speculative decoding via NextN (built-in MTP heads) and EAGLE. Both create a *draft KV pool* that lives alongside the target model's KV cache.

PR [#27805](https://github.com/sgl-project/sglang/pull/27805) fixes a gap: when speculative decoding is enabled together with HiCache on the `UnifiedRadixCache` path, the draft KV pool was not registered into `HostPoolGroup`. This caused HiCache backup/load to skip draft KV data entirely — the warning appeared but no data was saved.

The fix introduces a strategy-pattern dispatch:

- `DraftSwaStrategy` handles `DeepSeekV4TokenToKVPool` — creates a host pool matching the target SWA type.
- `DraftPlainKvStrategy` handles MHA/MLA pools as a fallback.
- `attach_draft_pool_to_unified_cache()` registers the draft pool as a `PoolEntry` in `HostPoolGroup`, enabling automatic participation in PoolTransfer-based backup/load and L3 storage.

For multi-step EAGLE, the interaction with HiSparse is more complex ([#25409](https://github.com/sgl-project/sglang/pull/25409)). Speculative decoding preallocates a logical KV window for several draft/verify steps, but only a subset is active in any given step. The fix separates speculative logical allocation from HiSparse device mapping:

- `prepare_for_decode()` only reserves logical KV locations.
- `prepare_for_v2_draft()` binds the draft slots for the current step.
- `prepare_for_v2_verify()` binds the target-verify slots for the current window.

For DeepSeek V4 specifically, full logical KV locations must be translated into compressed HiSparse locations before updating device mappings. Stale or overly broad mappings can cause later steps to read from the wrong slot.

The L3 implication: once the draft pool is registered in `HostPoolGroup`, HiCache's periodic SWA checkpoints naturally cover the draft KV as well. A cache hit that restores the target model's KV state also restores the draft model's state, enabling speculative decoding to resume without re-priming the draft model.

## Group semantics: keeping L3 objects coherent

When HiCache offloads one logical page of DeepSeek V4 KV cache to Mooncake Store, the physical reality is multiple objects: MLA K and V tensors, SWA K and V (when checkpointed), C4/C128 state buffers, possibly indexer sidecar objects, and draft KV — potentially split further across TP ranks. A TP=2 deployment with SWA checkpoint and draft pool can easily produce 8+ Mooncake objects per logical page.

Mooncake Store manages lifecycle at object granularity. Without coordination, its eviction policy can remove some objects from a group while leaving others. The result is a partial group: K without V, or rank-0 data without rank-1. For the KV cache system, a partial group is useless — a cache hit against it cannot produce correct model output.

The [group semantics RFC](https://github.com/kvcache-ai/Mooncake/issues/2127) quantified this problem. Under real HiCache traffic (Qwen3-8B, TP=2, `bench_mix.py` workload), object-level eviction creates eviction-fragmented partial groups:

| L3 capacity | Eviction cycles | Bad-cache live-byte ratio |
|-------------|----------------|---------------------------|
| 8 GB | 75 | 4.09% |
| 16 GB | 81 | 2.44% |
| 64 GB | 74 | 1.30% |
| 256 GB | 85 | 0.88% |

At 8 GB the fragmentation is material. At production scale (64 GB+) it drops below 1.5%, but any partial group is a wasted cache miss that triggers full recomputation.

Mooncake Store addresses this with opt-in grouped object routing ([kvcache-ai/Mooncake#2180](https://github.com/kvcache-ai/Mooncake/pull/2180), merged). The core mechanism:

- Objects carry a `group_id` via `ReplicateConfig.group_ids`.
- Grouped objects are routed to the same metadata shard by `hash(group_id)`.
- When any group member is accessed, the lease is refreshed for *all current group members*.
- When eviction selects a grouped candidate, it expands to evict all currently safe-to-evict group members together.

The SGLang integration ([sgl-project/sglang#26574](https://github.com/sgl-project/sglang/pull/26574)) maps the logical HiCache page key to the Mooncake group id:

```text
sglang-hicache:{tagged_logical_page_key}
```

This means all sub-objects derived from the same radix tree page — MHA/MLA K+V, SWA checkpoint, indexer sidecar, draft KV, across all TP ranks — share one group identity. The opt-in flag:

```json
{
  "enable_group_semantics": true
}
```

The implementation preserves backward compatibility: when running against older Mooncake packages without `ReplicateConfig.group_ids`, SGLang falls back to the existing ungrouped `batch_put_from(keys, ptrs, sizes)` path. No configuration change is needed for non-Mooncake backends.

For DeepSeek V4 specifically, group semantics closes an important gap. The hybrid architecture produces more sub-objects per logical page than a pure MHA model: MLA latent KV, SWA KV checkpoint, C4 state (on the SWA path), C128 state (request-scoped but tied to the same prefix), and optionally draft KV. Without group semantics, independent eviction of any one piece renders the entire cached prefix useless. With it, the full constellation is evicted or refreshed as a unit, ensuring that every cache hit returns a complete, usable state.

## What comes next

The current L3 checkpoint implementation backs up to the nearest checkpoint boundary on hit and re-runs full model prefill for the gap. The next step is **SWA-only partial prefill**: on cache hit, re-run *only* the SWA attention layers for the gap tokens, leaving the MLA layers untouched (their output is already in the loaded cache).

This cuts recomputation cost in proportion to the fraction of layers that use SWA (the first few layers in DeepSeek V4's architecture). With SWA-only partial prefill, the TTFT delta at 16K checkpoint interval would drop substantially, making even aggressive intervals practical.

Beyond that, several extensions are in flight:

- **C128 state in L3**: with request-scoped indexing, C128 state can be serialized alongside MLA cache in Mooncake Store. This enables full-state restoration on cache hit without recomputing the DSA indexer warm-up.
- **HISA pool-K offload**: for deployments using the hierarchical sparse indexer, the mean-pooled block cache needs L3 serialization — a format extension to the existing storage protocol.
- **Draft KV in L3**: with the draft pool registered in `HostPoolGroup`, speculative decoding state follows the same offload/reload path as the target model.
- **Cross-node checkpoint-aware routing**: the `swa_valid_from` event metadata enables the conductor to route requests to the instance holding both MLA cache and the closest SWA checkpoint, minimizing recomputation distance.

- **Group completeness guarantees**: the current group semantics are best-effort (evict-together, refresh-together). Future work includes atomic group visibility (a group is either fully present or fully absent in cache lookup), `group_size` tracking for completeness checks, and prefix-aware eviction policies that account for group membership when selecting eviction candidates.

The combined picture: a million-token DeepSeek V4 cache entry in L3, stored at ~4 GB (MLA + sparse SWA checkpoints), optionally including C128 state and draft KV, group-coherent across TP ranks, reloadable in under 4 seconds with SWA-only partial prefill, routable via workflow-affinity. That is the target state.

If you are running DeepSeek V4 with HiCache and Mooncake Store today, start with `--swa-checkpoint-interval 2048` and `SGLANG_SWA_CACHE_CHECKPOINT_MIN_TOKEN_INTERVAL=81920`. Watch your L3 utilization drop and your agent cache hit rate climb.

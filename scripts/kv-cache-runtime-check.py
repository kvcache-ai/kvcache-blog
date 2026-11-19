#!/usr/bin/env python3
"""Compare calculator component bytes with a real SGLang hybrid KV pool.

Run on the target host with its installed SGLang and an already-local HF config.
This script allocates cache buffers only; it does not load model weights or run
an attention forward. Expected bytes must come from the project calculator at
the same token count and cache dtype.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def tensor_info(tensor):
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": tensor.numel(),
        "element_size_bytes": tensor.element_size(),
        "logical_bytes": tensor.numel() * tensor.element_size(),
        "storage_bytes": tensor.untyped_storage().nbytes(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Local HF config.json path")
    parser.add_argument("--model-id", required=True, help="Official model ID represented by this local weight variant")
    parser.add_argument("--official-revision", required=True, help="Pinned official HF config revision used for comparison")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--kv-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--tokens", type=int, required=True, help="Full-attention sequence length to allocate")
    parser.add_argument("--window", type=int, required=True, help="SWA tail retained by the runtime")
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--full-layer", type=int, required=True, help="One config layer whose hybrid pattern is 0")
    parser.add_argument("--swa-layer", type=int, required=True, help="One config layer whose hybrid pattern is 1")
    parser.add_argument("--expected-full-bytes", type=int, required=True, help="One full layer's calculator bytes at --tokens")
    parser.add_argument("--expected-swa-bytes", type=int, required=True, help="One SWA layer's calculator bytes at min(tokens, window)")
    args = parser.parse_args()

    if args.tokens <= 0 or args.window <= 0 or args.page_size <= 1:
        parser.error("tokens/window must be positive and page-size must exceed 1 for SGLang SWA-tail allocation")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    pattern = config.get("hybrid_layer_pattern")
    if not isinstance(pattern, list):
        parser.error("config.json must contain hybrid_layer_pattern")
    if len(pattern) != int(config["num_hidden_layers"]):
        parser.error("hybrid_layer_pattern length does not match num_hidden_layers")
    if pattern[args.full_layer] != 0 or pattern[args.swa_layer] != 1:
        parser.error("selected layers do not match full=0 and SWA=1 in hybrid_layer_pattern")

    try:
        import torch
        from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
        import sglang
    except Exception as exc:  # Report a concrete incompatible environment.
        print(json.dumps({"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}))
        return 2

    if not torch.cuda.is_available():
        print(json.dumps({"status": "unavailable", "error": "CUDA is unavailable"}))
        return 2

    dtype = torch.bfloat16 if args.kv_dtype == "bf16" else torch.float16
    page = args.page_size
    tail = min(args.tokens, args.window)
    full_capacity = ((args.tokens + page - 1) // page) * page
    swa_capacity = ((tail + page - 1) // page) * page
    free_before, device_total = torch.cuda.mem_get_info(torch.device(args.device))
    allocated_before = torch.cuda.memory_allocated(torch.device(args.device))

    # Use SGLang's static hybrid fallback: it owns separate full/SWA tensors and
    # supports this model's unequal K/V dimensions. No formula builds the tensors.
    cache = SWAKVPool(
        size=full_capacity,
        size_swa=swa_capacity,
        page_size=page,
        dtype=dtype,
        head_num=int(config["num_key_value_heads"]),
        head_dim=int(config["head_dim"]),
        swa_head_num=int(config["swa_num_key_value_heads"]),
        swa_head_dim=int(config["swa_head_dim"]),
        v_head_dim=int(config["v_head_dim"]),
        swa_v_head_dim=int(config["swa_v_head_dim"]),
        full_attention_layer_ids=[args.full_layer],
        swa_attention_layer_ids=[args.swa_layer],
        device=args.device,
        kv_cache_layout="nhd",
    )
    allocator = SWATokenToKVPoolAllocator(
        size=full_capacity,
        size_swa=swa_capacity,
        page_size=page,
        dtype=dtype,
        device=args.device,
        kvcache=cache,
        need_sort=False,
    )

    prefix_lens = torch.tensor([0], dtype=torch.int64, device=args.device)
    prefix_lens_cpu = torch.tensor([0], dtype=torch.int64)
    seq_lens = torch.tensor([args.tokens], dtype=torch.int64, device=args.device)
    seq_lens_cpu = torch.tensor([args.tokens], dtype=torch.int64)
    last_loc = torch.tensor([-1], dtype=torch.int64, device=args.device)
    kv_loc = allocator.alloc_extend_swa_tail(
        prefix_lens=prefix_lens,
        prefix_lens_cpu=prefix_lens_cpu,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens_cpu,
        last_loc=last_loc,
        extend_num_tokens=args.tokens,
        swa_tail_len=tail,
    )
    if kv_loc is None:
        print(json.dumps({"status": "allocation_failed", "error": "SGLang allocator returned None"}))
        return 2

    full_k = cache.full_kv_pool.k_buffer[0]
    full_v = cache.full_kv_pool.v_buffer[0]
    swa_k = cache.swa_kv_pool.k_buffer[0]
    swa_v = cache.swa_kv_pool.v_buffer[0]
    mapping = allocator.full_to_swa_index_mapping[kv_loc]
    prefix_live = int((mapping[: args.tokens - tail] > 0).sum().item())
    tail_live = int((mapping[args.tokens - tail :] > 0).sum().item())

    full_row_bytes = (
        full_k.shape[1] * full_k.shape[2] + full_v.shape[1] * full_v.shape[2]
    ) * full_k.element_size()
    swa_row_bytes = (
        swa_k.shape[1] * swa_k.shape[2] + swa_v.shape[1] * swa_v.shape[2]
    ) * swa_k.element_size()
    full_live_bytes = int(kv_loc.numel()) * full_row_bytes
    swa_live_bytes = tail_live * swa_row_bytes
    full_storage_bytes = tensor_info(full_k)["logical_bytes"] + tensor_info(full_v)["logical_bytes"]
    swa_storage_bytes = tensor_info(swa_k)["logical_bytes"] + tensor_info(swa_v)["logical_bytes"]

    torch.cuda.synchronize(torch.device(args.device))
    allocated_after = torch.cuda.memory_allocated(torch.device(args.device))
    free_after, _ = torch.cuda.mem_get_info(torch.device(args.device))
    checks = {
        "full_bytes_match_calculator": full_live_bytes == args.expected_full_bytes,
        "swa_bytes_match_calculator": swa_live_bytes == args.expected_swa_bytes,
        "discarded_prefix_has_no_swa_mapping": prefix_live == 0,
        "window_tail_has_swa_mapping": tail_live == tail,
    }
    report = {
        "status": "pass" if all(checks.values()) else "mismatch",
        "versions": {"torch": torch.__version__, "sglang": sglang.__version__},
        "model_id": args.model_id,
        "official_config_revision": args.official_revision,
        "weight_config_path": str(args.config),
        "cache_dtype": str(dtype),
        "device": str(torch.cuda.get_device_name(torch.device(args.device))),
        "page_size": page,
        "tokens": args.tokens,
        "sliding_window": args.window,
        "retained_swa_tokens": tail_live,
        "pool_capacity_tokens": {"full": full_capacity, "swa": swa_capacity},
        "tensor_views": {
            "full_k": tensor_info(full_k),
            "full_v": tensor_info(full_v),
            "swa_k": tensor_info(swa_k),
            "swa_v": tensor_info(swa_v),
        },
        "components": {
            "full": {
                "calculator_bytes": args.expected_full_bytes,
                "runtime_live_bytes": full_live_bytes,
                "pool_storage_bytes": full_storage_bytes,
                "reserved_bytes": full_storage_bytes - full_live_bytes,
                "bytes_per_token": full_row_bytes,
            },
            "swa": {
                "calculator_bytes": args.expected_swa_bytes,
                "runtime_live_bytes": swa_live_bytes,
                "pool_storage_bytes": swa_storage_bytes,
                "reserved_bytes": swa_storage_bytes - swa_live_bytes,
                "bytes_per_token": swa_row_bytes,
            },
        },
        "swa_mapping": {
            "prefix_tokens": args.tokens - tail,
            "prefix_tokens_with_live_swa_mapping": prefix_live,
            "tail_tokens": tail,
            "tail_tokens_with_live_swa_mapping": tail_live,
        },
        "gpu_memory": {
            "device_total_bytes": device_total,
            "device_free_before_bytes": free_before,
            "device_free_after_bytes": free_after,
            "device_free_delta_bytes": free_before - free_after,
            "torch_allocated_delta_bytes": allocated_after - allocated_before,
            "allocator_and_runtime_overhead_bytes": allocated_after - allocated_before - full_storage_bytes - swa_storage_bytes,
        },
        "checks": checks,
        "limits": [
            "One representative full layer and one representative SWA layer only.",
            "Cache buffers and allocator mappings are measured; no model weights or attention forward are loaded.",
            "Pool storage includes SGLang page rounding and its reserved sink page; calculator comparison uses live logical tokens.",
            "The FP8/MXFP4 metadata describes weights; this run explicitly measures BF16 KV cache.",
        ],
    }
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

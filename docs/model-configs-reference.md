# KV Cache Calculator - Model Configuration Reference

> This document records all model configurations added to `data/kv_cache_calculator/models.yaml`.
> Parameters are sourced from official HuggingFace config.json files.

## Formulas

| Formula | Description | Key Fields |
|---------|-------------|------------|
| `standard_gqa` | Standard Grouped Query Attention | num_hidden_layers, num_attention_heads, num_key_value_heads, head_dim |
| `mla` | Multi-head Latent Attention (DeepSeek) | num_hidden_layers, kv_lora_rank, qk_rope_head_dim, qk_nope_head_dim, v_head_dim, num_key_value_heads |
| `dsa_mla` | DSA + MLA with indexer | Same as mla + index_head_dim, index_n_heads, index_topk |
| `deepseek_v4_hybrid` | DeepSeek V4 hybrid (sliding window + compression) | num_hidden_layers, head_dim, sliding_window, compress_ratios, index_head_dim, index_topk |

---

## Models Already Submitted (PR #7)

### DeepSeek V3 / R1 (MLA)

| Model | Layers | kv_lora_rank | qk_rope_head_dim | qk_nope_head_dim | v_head_dim | KV Heads | Max Ctx |
|-------|--------|-------------|-------------------|------------------|-----------|----------|---------|
| DeepSeek V3 | 61 | 512 | 64 | 128 | 128 | 128 | 163840 |
| DeepSeek R1 | 61 | 512 | 64 | 128 | 128 | 128 | 163840 |

Source: `https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json`

### Llama 3.1 / 3.3 (Standard GQA)

| Model | Layers | Attn Heads | KV Heads | head_dim | Max Ctx |
|-------|--------|-----------|----------|----------|---------|
| Llama 3.1 8B | 32 | 32 | 8 | 128 | 131072 |
| Llama 3.1 70B | 80 | 64 | 8 | 128 | 131072 |
| Llama 3.1 405B | 126 | 128 | 8 | 128 | 131072 |
| Llama 3.3 70B | 80 | 64 | 8 | 128 | 131072 |

Source: `https://huggingface.co/meta-llama/Llama-3.1-*-Instruct/raw/main/config.json`

### Qwen2.5 (Standard GQA)

| Model | Layers | Attn Heads | KV Heads | head_dim | Max Ctx |
|-------|--------|-----------|----------|----------|---------|
| Qwen2.5-72B | 80 | 64 | 8 | 128 | 131072 |
| Qwen2.5-32B | 64 | 40 | 8 | 128 | 131072 |
| Qwen2.5-14B | 48 | 40 | 8 | 128 | 131072 |
| Qwen2.5-7B | 28 | 28 | 4 | 128 | 131072 |

Source: `https://huggingface.co/Qwen/Qwen2.5-*-Instruct/raw/main/config.json`

### Mistral (Standard GQA)

| Model | Layers | Attn Heads | KV Heads | head_dim | Max Ctx |
|-------|--------|-----------|----------|----------|---------|
| Mistral Large (2411) | 88 | 96 | 8 | 128 | 131072 |
| Mistral Small 24B | 40 | 32 | 8 | 128 | 32768 |

Source: `https://huggingface.co/mistralai/Mistral-Large-Instruct-2411/raw/main/config.json`

### Gemma 2 (Standard GQA)

| Model | Layers | Attn Heads | KV Heads | head_dim | Max Ctx |
|-------|--------|-----------|----------|----------|---------|
| Gemma 2 27B | 46 | 32 | 16 | 128 | 8192 |
| Gemma 2 9B | 42 | 16 | 8 | 256 | 8192 |

Source: `https://huggingface.co/google/gemma-2-*-it/raw/main/config.json`

### Gemma 3 (Standard GQA)

| Model | Layers | Attn Heads | KV Heads | head_dim | Max Ctx |
|-------|--------|-----------|----------|----------|---------|
| Gemma 3 27B | 62 | 32 | 16 | 128 | 131072 |
| Gemma 3 12B | 48 | 16 | 8 | 256 | 131072 |

Note: Gemma 3 uses interleaved local/global attention (sliding_window=1024, global every ~4th layer). The `standard_gqa` formula gives a conservative (upper-bound) estimate.

Source: `https://huggingface.co/google/gemma-3-*-it/raw/main/config.json`

### Phi-4 (Standard GQA)

| Model | Layers | Attn Heads | KV Heads | head_dim | Max Ctx |
|-------|--------|-----------|----------|----------|---------|
| Phi-4 14B | 40 | 40 | 10 | 128 | 16384 |

Source: `https://huggingface.co/microsoft/phi-4/raw/main/config.json`

### Cohere Command R (Standard GQA)

| Model | Layers | Attn Heads | KV Heads | head_dim | Max Ctx |
|-------|--------|-----------|----------|----------|---------|
| Command R+ | 64 | 96 | 8 | 128 | 131072 |
| Command R | 40 | 64 | 8 | 128 | 131072 |

Source: `https://huggingface.co/CohereForAI/c4ai-command-r-*/raw/main/config.json`

---

## Models Pending Addition (Researched)

### Llama 4 (Standard GQA, iRoPE chunked attention)

| Model | Layers | Attn Heads | KV Heads | head_dim | Max Ctx | Notes |
|-------|--------|-----------|----------|----------|---------|-------|
| Llama 4 Scout 17B-16E | 48 | 40 | 8 | 128 | 10485760 | Chunked attn (chunk=8192), 16 MoE experts |
| Llama 4 Maverick 17B-128E | 48 | 40 | 8 | 128 | 1048576 | Chunked attn (chunk=8192), 128 MoE experts |

### Mixtral (Standard GQA, MoE)

| Model | Layers | Attn Heads | KV Heads | head_dim | Max Ctx | Notes |
|-------|--------|-----------|----------|----------|---------|-------|
| Mixtral 8x7B | 32 | 32 | 8 | 128 | 32768 | 8 experts, top-2 |
| Mixtral 8x22B | 56 | 48 | 8 | 128 | 65536 | 8 experts, top-2 |

### Yi (Standard GQA)

| Model | Layers | Attn Heads | KV Heads | head_dim | Max Ctx |
|-------|--------|-----------|----------|----------|---------|
| Yi-1.5-34B | 60 | 56 | 8 | 128 | 4096 |
| Yi-1.5-9B | 48 | 32 | 4 | 128 | 4096 |

### InternLM (Standard GQA)

| Model | Layers | Attn Heads | KV Heads | head_dim | Max Ctx |
|-------|--------|-----------|----------|----------|---------|
| InternLM2.5-20B | 48 | 48 | 8 | 128 | 32768 |
| InternLM3-8B | 32 | 32 | 8 | 128 | 32768 |

### DBRX (Standard GQA, MoE)

| Model | Layers | Attn Heads | KV Heads | head_dim | Max Ctx | Notes |
|-------|--------|-----------|----------|----------|---------|-------|
| DBRX 132B | 40 | 48 | 8 | 128 | 32768 | 16 experts, top-4 |

### Phi-3 (Standard GQA)

| Model | Layers | Attn Heads | KV Heads | head_dim | Max Ctx |
|-------|--------|-----------|----------|----------|---------|
| Phi-3-medium 14B | 40 | 40 | 10 | 128 | 131072 |
| Phi-3-small 7B | 32 | 32 | 8 | 128 | 131072 |
| Phi-3-mini 3.8B | 32 | 32 | 32 | 96 | 131072 |

### Qwen2-MoE (Standard GQA)

| Model | Layers | Attn Heads | KV Heads | head_dim | Max Ctx |
|-------|--------|-----------|----------|----------|---------|
| Qwen2-57B-A14B | 28 | 28 | 4 | 128 | 32768 |
| Qwen2.5-Coder-32B | 64 | 40 | 8 | 128 | 32768 |

### Falcon (Standard GQA)

| Model | Layers | Attn Heads | KV Heads | head_dim | Max Ctx |
|-------|--------|-----------|----------|----------|---------|
| Falcon-180B | 80 | 232 | 8 | 64 | 2048 |
| Falcon-40B | 60 | 128 | 8 | 64 | 2048 |

### Nemotron (Standard GQA)

| Model | Layers | Attn Heads | KV Heads | head_dim | Max Ctx |
|-------|--------|-----------|----------|----------|---------|
| Nemotron-4-340B | 96 | 96 | 8 | 192 | 4096 |

---

## Notes

- **MoE models**: KV cache is independent of MoE — only attention parameters matter.
- **Jamba (hybrid SSM+Attention)**: Only attention layers produce KV cache. Jamba 1.5 Large has 8 attention layers out of 64 total; Jamba 1.5 Mini has 4 out of 32. This requires a special formula (`num_attention_layers` instead of `num_hidden_layers`).
- **Llama 4 (iRoPE)**: Uses chunked attention (chunk_size=8192). Local-attention layers only cache chunk_size tokens. `standard_gqa` gives upper-bound estimate.
- **Gemma 3**: Uses interleaved sliding window (1024) + global attention. `standard_gqa` gives upper-bound estimate.

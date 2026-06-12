# KVCache.ai

Official website source for [KVCache.ai](https://kvcache.ai/) — the home of open-source projects and research on KV cache management and LLM serving optimization.

## About KVCache.ai

KVCache.ai advances the state of the art in Large Language Model (LLM) inference optimization. In decoder-only Transformer models, data from diverse modalities can ultimately be transformed into KV cache, making it a central component of modern LLM serving systems. As a result, KV cache has become a key focus for improving inference efficiency through techniques such as caching, scheduling, compression, offloading, and disaggregated serving architectures.

Through open-source projects and academic research, KVCache.ai develops effective, practical, and high-performance solutions for KV cache management and LLM serving optimization. The goal is to make LLM deployment more accessible, efficient, and cost-effective for organizations of all sizes.

### Featured Projects

- **[Mooncake](https://github.com/kvcache-ai/Mooncake)** — A KV cache-centric disaggregated architecture for LLM serving.
- **[KTransformers](https://ktransformers.net/en)** — A CPU/GPU heterogeneous LLM inference and fine-tuning framework for running and tuning 100B+ models on accessible workstation hardware.
- **[TrEnv-X](https://github.com/kvcache-ai/TrEnv-X)** — An open-source runtime platform designed for AI Agent applications.

### Tools

- **[KV Cache Size Calculator](/tools/kv-cache-size-calculator/)** — Estimate KV cache capacity for common production LLM families, including DeepSeek, GLM, Kimi, Qwen, MiniMax, MiMo, and others.

## Tech Stack

This site is built with:

- [Hugo](https://gohugo.io/) (v0.126.3) — static site generator
- [Hugo Blox](https://docs.hugoblox.com/) — theme and page builder (Tailwind CSS)
- [Pagefind](https://pagefind.app/) — static search (used in production builds on Netlify)

Content is written in Markdown with YAML front matter. Custom layouts and shortcodes live under `layouts/`.

## Links

- Website: [https://kvcache.ai](https://kvcache.ai)
- GitHub Organization: [https://github.com/kvcache-ai](https://github.com/kvcache-ai)
- X (Twitter): [@KVCache_AI](https://x.com/KVCache_AI)

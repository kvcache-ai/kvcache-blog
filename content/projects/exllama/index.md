---

title: Exllama
summary: An inference library for running local LLMs on modern consumer GPUs. 

  ExLlamaV2 supports the same 4-bit GPTQ models as V1, but also a new "EXL2" format. EXL2 is based on the same optimization method as GPTQ and supports 2, 3, 4, 5, 6 and 8-bit quantization. The format allows for mixing quantization levels within a model to achieve any average bitrate between 2 and 8 bits per weight.

date: 2024-06-15
showData: false
# authors:
#   - admin
tags:
  - tag

home_weight: 10
showathome: true

external_link: https://github.com/kvcache-ai/Lexllama
doc_link: /docs/exllama


draft: true

---



ExLlamaV2 is an inference library for running local LLMs on modern consumer GPUs.


## New in v0.1.0:

- ExLlamaV2 now supports paged attention via [Flash Attention](https://github.com/Dao-AILab/flash-attention) 2.5.7+
- New generator with dynamic batching, smart prompt caching, K/V cache deduplication and simplified API


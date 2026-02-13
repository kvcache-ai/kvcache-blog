---
title: "Mooncake Joins the PyTorch Ecosystem"
summary: "Mooncake is now part of the PyTorch Ecosystem, complementing PyTorch-native LLM serving with high-performance disaggregated data transfer and storage."
date: 2026-02-02
authors:
  - Mooncake community
tags:
  - Mooncake
  - PyTorch
  - LLM Serving

draft: false
showathome: true

commentable: false
home_weight: 100

---

We're excited to announce that **Mooncake has officially joined the PyTorch Ecosystem**.

For Mooncake, this is more than a badge—it represents a commitment to open governance, long-term collaboration with the PyTorch community, and a shared goal of making disaggregated inference architectures easier to adopt and better supported in the PyTorch-native LLM serving stack.

As context lengths grow and models continue to scale, LLM serving is rapidly evolving from **monolithic, single-stack deployments** to **heterogeneous, disaggregated architectures**. In these systems, different stages of inference run on different resources, and overall performance is determined not only by raw compute, but increasingly by the efficiency of the *data plane*.

In modern serving architectures, **communication and storage** often become the key bottlenecks and the dominant design constraints. For example: how quickly KVCache can be transferred from prefill nodes to decode nodes, or how efficiently KVCache can be reused across requests at scale.

Mooncake is built to address this missing layer: **a communication and storage infrastructure for large-model serving**, designed to make disaggregated LLM architectures practical, scalable, and composable in real-world production environments.

Concretely, Mooncake enables:

- **Prefill–decode disaggregation**: Separate high-throughput prefill or encoder workloads from latency-sensitive decode stages, while efficiently transferring KVCache across clusters.

- **Global KVCache reuse**: Treat KVCache blocks as shared, reusable state across requests and engine instances, improving cache hit rates and reducing redundant prefill work.

- **Elastic expert parallelism (MoE)**: Decouple experts from fixed workers, enabling more elastic and resilient serving under partial failures or shifting traffic patterns.

- **Fault-tolerant distributed backends**: Provide PyTorch distributed primitives designed to continue operating even in the presence of rank failures.

- **Fast weight updates**: Support rapid model and weight updates for RL, checkpointing, and iterative deployment workflows via tensor-native, zero-copy APIs.

Mooncake began as a research collaboration between Moonshot AI and Tsinghua University, emerging from the need to overcome the "memory wall" when serving large-scale models such as Kimi. Since its open-source release, it has grown into a thriving community-driven project.

Today, Mooncake is already being wired into PyTorch-native serving workflows through integrations with engines such as **SGLang**, **vLLM**, and **TensorRT-LLM**. Together, Mooncake and its integrations have led to wide adoption across leading organizations, including Moonshot AI (Kimi), Alibaba Cloud, Ant Group, JD.com, Tencent, Meituan, Approaching.AI, and LightSeek Foundation, ensuring smooth serving for millions of concurrent users.

Looking ahead, we are excited to collaborate even more closely across the ecosystem and to push the state of LLM serving toward a future that is more scalable, efficient, and production-ready.

For more details, see the official announcement on the PyTorch blog: https://pytorch.org/blog/mooncake-joins-pytorch/

---
title: "AgentENV Joins Forces with Miles to Accelerate Agentic RL at Scale"
summary: "AgentENV is now integrated with Miles as a sandbox backend, bringing isolated, fast-starting, and scalable Firecracker microVMs to large-scale agentic reinforcement learning."
date: 2026-08-18
authors:
  - AgentENV team
tags:
  - AgentENV
  - Miles
  - Agentic RL
  - RL Infrastructure

draft: false
showathome: true
commentable: false
home_weight: 202608180

image:
  preview_only: true
  alt_text: "AgentENV integration with Miles for agentic RL"
---

AgentENV is now integrated with Miles as a sandbox backend, bringing efficient, self-hosted execution environments to agentic reinforcement learning.

In agentic RL, rollouts require isolated environments where agents can perform actions such as writing and executing code and calling tools. Traditional sandboxes can be expensive to start and inflexible to build. At scale, downloading and unpacking complete OCI images adds further latency, making environment infrastructure a potential bottleneck in the training pipeline.

Miles is a high-performance, enterprise-ready reinforcement learning framework for large-scale model post-training. It offers fully asynchronous RL with decoupled rollout and training workers, fast agentic rollouts and in-loop weight updates, optimized low-precision training, and support for both LoRA and multi-LoRA.

Together, Miles manages the RL training workflow, while AgentENV supplies isolated, fast-starting, and scalable Firecracker microVMs during rollout. This integration has been validated with a sustained GRPO run over the full Terminal-Bench-2 task set. Using GLM-4.7-Flash on 8×H200 GPUs, the run completed 55 rollouts and approximately 3,400 episodes, with every episode executing in a fresh microVM warm-started from a pre-baked snapshot. A single `m7i.metal-24xl` AgentENV server sustained 64 concurrent microVMs throughout the run.

Related links:

- [AgentENV](https://github.com/kvcache-ai/AgentENV)
- [Miles](https://github.com/radixark/miles)

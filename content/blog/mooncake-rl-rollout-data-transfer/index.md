---
title: "Moving RL Rollout Data with Mooncake Structured-Object Transfer"
summary: "Mooncake now has a concrete RL rollout-data integration in Miles, using structured-object transfer to move rollout batches without hiding their field structure."
date: 2026-08-06
authors:
  - Mooncake community
tags:
  - Mooncake
  - Miles
  - RL
  - Rollout Data
  - Data Transfer
draft: false
showathome: true
commentable: false
home_weight: 202608060
image:
  preview_only: true
  alt_text: "Mooncake structured-object transfer for RL rollout data"
---

In RL training, rollout data is the handoff between acting and learning. Rollout workers run the current policy, collect prompts, generated tokens, masks, log probabilities, rewards, and metadata, and pass that batch to the trainer. The trainer uses it to compute losses and update the next policy version. If this handoff is slow, expensive, or lossy, the training loop pays for it directly: trainers wait for data, rollout workers hold memory longer, and the policy update consumes a distorted view of the samples.

Miles has merged Mooncake rollout data transfer support in [radixark/miles#591](https://github.com/radixark/miles/pull/591). The change adds Mooncake as a transfer backend between rollout producers and training workers.

This handoff is not shaped like KV cache. A rollout batch is a Python-level object with tensor-like arrays, ragged per-sample data, scalar lists, metadata, and sometimes multimodal fields. Framework-level object-store serialization can move the object, but the nested rollout structure is opaque to Mooncake. Mooncake's structured-object path keeps that structure visible while still moving the heavy bytes through the data plane.

![RL rollout data movement](rollout-data-plane.svg)

_Figure 1. The framework scheduler carries lightweight references, while Mooncake owns structured rollout payload movement and cleanup._

## Rollout Data Belongs on the Data Plane

In asynchronous RL training, rollout workers and trainer workers run at their own pace. Rollout workers produce the samples that define a training step; trainer workers consume those samples to update the policy. Once those two roles live in different processes or on different machines, rollout transfer becomes part of the critical path between generation and learning.

The scheduler still decides which trainer receives which batch. The batch payload should take a separate path:

- the framework scheduler passes lightweight references;
- Mooncake stores and transfers the payload;
- the training worker reconstructs the original object before running the step;
- cleanup releases the short-lived rollout object after use.

Different RL frameworks name this object differently, but the transfer shape is similar.

## What Miles Sends

Miles is a good concrete example because its rollout object is neither a single tensor nor a random Python object. It has a stable field layout for a given rollout configuration, but the leaves have different storage needs.

The benchmark payloads were built from captured Miles rollout-source data and scaled by repeating the same source layout. That keeps the field types close to the real transfer input instead of replacing the workload with synthetic dense tensors.

Each sample is about 3,422 bytes of logical payload before transfer encoding in this capture. The benchmark scales the same rollout layout from 128 MiB to 4 GiB so that the field mix stays stable while the object count grows.

Most bytes are not stored as one dense tensor object. They are per-sample arrays stored as Python lists of NumPy arrays:

| Field | Layout Before Transfer | Data Type | Size Per Sample | Discreteness |
| --- | --- | --- | ---: | --- |
| `tokens` | `list[np.ndarray]` | `int32` | ~1,338 B | Ragged per-sample token arrays. |
| `loss_masks` | `list[np.ndarray]` | `int32` | 1,024 B | One array per sample; fixed length 256 in this data set. |
| `rollout_log_probs` | `list[np.ndarray]` | `float32` | 1,024 B | One array per sample; fixed length 256 in this data set. |
| `partition` | Python list | `int` | small | High-cardinality scalar field; one distinct value per sample in the generated benchmark. |
| `sample_indices` | Python list | `int` | small | High-cardinality scalar field; one distinct value per sample. |
| `response_lengths` | Python list | `int` | small | Low-cardinality scalar field; all values are 256. |
| `rewards` | Python list | `float` | small | Low-cardinality scalar field in this capture; all values are 0.0. |
| `truncated` | Python list | `int` | small | Low-cardinality scalar field in this capture; all values are 1. |
| `weight_versions` | Python list of objects | list-like object | ~25 B serialized | Low-cardinality object field in this capture; still needs object-preserving encoding. |
| `raw_reward` | Python list | `float` | small | Metadata-style list; all values are 0.0 in this capture. |
| `total_lengths` | Python list | `int` | small | Per-sample total sequence length; values vary with prompt length in this capture. |

The three array-list fields add up to about 3,386 bytes per sample, so they dominate transfer volume. The scalar and object fields are smaller, but they still carry routing, length, reward, and bookkeeping state.

Two details matter for transfer. First, the object is highly fragmented before Mooncake sees it: a 4 GiB payload contains more than one million samples, and each large field starts as one small array per sample. Second, value cardinality varies by field. Some scalar fields are identifiers; others are almost constant in this capture. Mooncake needs to preserve both the typed payloads and the Python-level rollout state.

![Miles rollout object anatomy](rollout-object-anatomy.svg)

_Figure 2. In this Miles capture, most bytes come from typed array-list fields, while smaller scalar and metadata fields preserve training semantics._

### What Changes Object Size

The schema is chosen by the framework, not by the model size. For this Miles layout, the main per-sample bytes come from token IDs, loss masks, and rollout log probabilities. The object size grows with the number of samples in one transfer round and with sequence length, especially response length.

A larger model may still lead to larger transfer objects indirectly. Large-model RL jobs often use more rollout workers, larger rollout batches, more samples per prompt, or longer response/context limits. Optional fields also matter: teacher log probabilities or KL fields add float arrays, MoE/speculative paths may add routed-expert or top-k arrays, and multimodal runs may add media-related fields. Model size matters here only through the rollout data the framework emits.

## DataProto and Dict Semantics

Mooncake handles two object styles that show up in RL frameworks. Miles uses a plain rollout dictionary. Other RL stacks often use DataProto-style containers. The transfer layer supports both because they carry different semantics.

A plain dict preserves Python object structure. Mooncake expands the dict by key path, transfers the leaves, and reconstructs a dict on GET. Frameworks that already represent rollout data as nested or flat Python dictionaries can use this path without adopting DataProto semantics. Mooncake does not assign special meaning to a dict key named `batch` or `meta_info`; it preserves the structure the framework gives it.

A DataProto-style object carries stronger rollout semantics. It usually separates data into `batch`, `non_tensor_batch`, and `meta_info`. The `batch` fields are tensor-like or ndarray-like fields with a batch dimension. The `non_tensor_batch` fields are per-sample non-tensor values. The `meta_info` fields describe the whole rollout object rather than one sample.

That distinction matters for partial reads. A trainer may only need selected batch fields or selected rows. DataProto transfer can preserve those semantics instead of flattening the object into an untyped dictionary.

A dict preserves object structure; a DataProto preserves rollout semantics. The dict path is not Miles-specific. The same schema and manifest mechanism can also describe other structured objects, such as COO-style sparse tensors represented by separate `indices`, `values`, `shape`, and metadata fields.

## Tensor, Non-Tensor, and Multimodal Payloads

Mooncake does not force every field through one representation.

Tensor fields are handled as payloads where byte movement dominates. When the native tensor path is available, Mooncake can use store and transfer APIs that are closer to the underlying tensor buffer.

Non-tensor fields are handled through structured codecs. These include scalar values, strings, bytes, lists, JSON-like values, object arrays, and ragged fields. For these fields, correctness depends on preserving more than bytes: the receiver needs to know the field path, dtype, shape, row boundaries, and whether a value is missing or explicitly `None`.

Multimodal fields fit naturally into the non-tensor side of the structured-object path. A rollout may carry image bytes, media metadata, or references to media stored elsewhere. Some of those fields are binary payloads; others are metadata or references. The codec follows the field representation, not the category name.

Lifecycle also belongs in the API. Rollout data is usually consumed once. After the trainer has reconstructed the object, cleanup and release paths remove the transferred payload and release temporary buffers.

## How Mooncake Transfers a Rollout Object

Mooncake keeps the framework boundary small. The framework calls `put` with a dict or DataProto-style object and later calls `get` with the returned reference. Scheduling, trainer placement, retries, and object lifetime remain framework decisions. Mooncake owns the structured payload path underneath that reference.

The main path is:

1. `put` receives a dict or DataProto-style object from the framework.
2. Mooncake expands the object into typed leaves.
3. The schema chooses how each leaf is encoded.
4. Payloads move through the store data plane, using BufferPool staging when that is cheaper than registering many small buffers.
5. `get` reads the manifest, fetches the leaves, decodes them, reconstructs the object, and cleanup releases the short-lived payload.

The core mechanism is schema-driven transfer. Mooncake should not infer a field's meaning from one batch. The schema gives each field a stable interpretation: dense tensor, ragged typed array, scalar list, object field, or bytes. The manifest records what was actually written for this rollout object, including payload keys and missing or explicit null values. GET uses both pieces to rebuild the object.

On the PUT side, Mooncake first expands the input object into leaves. For a dict, the leaf identity comes from the key path. For a DataProto-style object, the identity also carries rollout semantics such as `batch`, `non_tensor_batch`, and `meta_info`. Mooncake then uses the schema to choose the right encoding path for each leaf and writes a manifest with the field paths, dtype, shape, row boundaries, payload keys, and explicit null markers needed for reconstruction.

The field handling then follows the schema:

| Field Shape | Mooncake Handling | Why It Matters |
| --- | --- | --- |
| Dense tensor or ndarray | Store as typed payload with dtype and shape metadata. | Avoids turning large numeric buffers into one opaque Python object. |
| Ragged typed arrays | Store values with row-boundary metadata. | Preserves per-sample boundaries for fields like Miles `tokens`. |
| Scalar lists and object fields | Use structured codecs that preserve Python-level values. | Keeps routing IDs, rewards, lengths, and object metadata intact. |
| Bytes, strings, and multimodal metadata | Encode according to the field representation. | Supports media payloads, media references, and metadata without forcing them into tensor form. |
| Missing or explicit `None` values | Preserve null state in the manifest. | Prevents data loss when partial reads or sparse fields are reconstructed. |

The optimized structured-object path used here also has to care about registered memory. Registering and unregistering every small leaf is expensive, especially for rollout objects made of many small arrays and scalar fields. Mooncake uses a BufferPool for transfer staging: registered buffers are reused across operations, and batches of small or non-contiguous leaves are copied into those buffers before transfer instead of being registered one by one.

That extra copy is intentional. For small objects, one bulk copy into a pooled registered buffer is usually cheaper than thousands of register/unregister operations. For large contiguous tensor payloads, Mooncake can still use a more direct path when the backend and memory layout allow it. The transfer path chooses between these cases based on the field representation rather than forcing every field through the same mechanism.

The copy path is tuned for this rollout shape. Typed ragged fields are packed into contiguous layouts with row-boundary metadata. Small fields are grouped into transfer chunks so the store sees fewer tiny operations. The hot copy path avoids unnecessary temporary objects and uses bulk memory copies where possible, which keeps BufferPool staging fast even though it performs a memcpy before transfer.

On the GET side, the consumer starts from the same lightweight reference. Mooncake reads the manifest, derives the payload keys for the requested object or slice, fetches the selected leaves from the store, decodes them, and reconstructs the original dict or DataProto-style shape. This is also where partial reads fit naturally: GET can select fields or rows while still using the same schema and manifest to rebuild a valid object.

Cleanup and release are part of the lifecycle. Rollout data is usually short-lived and consumed once by a trainer. After the training step no longer needs the object, the framework calls cleanup so Mooncake can remove store-side payloads and release temporary buffers.

![Mooncake structured-object transfer architecture](structured-transfer-architecture.svg)

_Figure 3. The framework passes a small reference; Mooncake uses schema and manifest metadata to connect PUT-side encoding with GET-side reconstruction._

## Performance Results

We compare the Miles Ray backend with Mooncake structured-object transfer on the captured layout above. The source rollout was generated by Miles with Qwen3-0.6B on math prompts. The seed capture is `rollout_id=0` with 8 real rollout-source samples. Larger benchmark payloads repeat the captured field layout into one transfer object per round; the table reports the expanded logical sample count for each round.

The producer measures the backend PUT call after loading the payload. The remote consumer measures GET after one warmup run. These numbers cover rollout payload transfer and reconstruction, not full training throughput.

| Payload | Samples / Transfer Round | Mooncake PUT | Ray PUT | Mooncake GET | Ray GET | GET Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 MiB | 39,221 | 583 ms | 930 ms | 47.4 ms | 501.3 ms | 10.6x |
| 512 MiB | 156,888 | 2,830 ms | 3,489 ms | 173.2 ms | 2,134.2 ms | 12.3x |
| 1 GiB | 313,776 | 4,636 ms | 7,469 ms | 356.3 ms | 5,035.4 ms | 14.1x |
| 4 GiB | 1,255,104 | 20,320 ms | 31,830 ms | 1,425.9 ms | 20,034.3 ms | 14.1x |

![Miles GET latency benchmark](miles-get-latency.svg)

_Figure 4. Mooncake keeps the GET path close to data-plane throughput as the Miles rollout payload grows, while Ray spends more time in object-store transfer and reconstruction._

The GET side shows the data-plane effect most clearly. Mooncake stays around 2.6 to 2.9 GiB/s on these Miles payloads, while Ray is around 0.20 to 0.25 GiB/s in the same setup. The gap widens as the payload grows because the Ray path moves and reconstructs a large Python object through the object store, while Mooncake moves typed leaves and reconstructs at the boundary.

PUT also improves, but the result is more mixed because this Miles layout starts as many Python lists and ragged arrays. Some of the PUT time is not network transfer; it is the cost of walking the object, encoding ragged fields, and building the manifest. That cost is still worth paying for this workload because rollout data is consumed remotely, and the trainer-side GET path is on the critical path before the training step can run.

## Miles as the First Merged Integration

Miles is the first merged RL framework integration for this structured rollout transfer path. Miles remains responsible for the RL workflow; Mooncake is the transfer substrate.

That boundary is important. Mooncake does not replace the framework scheduler, rollout manager, or trainer. It gives those components a data plane for moving structured rollout objects without collapsing them into a single opaque serialization format.

The same transfer model applies to other RL frameworks that move DataProto-style or dict-style rollout data between rollout and training stages. It is not limited to RL either. A dict plus schema can describe data layouts that are more structured than a blob but not naturally represented as one dense tensor. COO-format sparse tensors are one example: indices, values, shape, and auxiliary metadata can be transferred as separate typed leaves while still reconstructing one logical sparse object at the boundary.

## Related Links

- Miles PR: [radixark/miles#591](https://github.com/radixark/miles/pull/591)
- Mooncake project: [https://github.com/kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake)
- Mooncake documentation: [https://kvcache-ai.github.io/Mooncake/](https://kvcache-ai.github.io/Mooncake/)

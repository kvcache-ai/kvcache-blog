---
title: "Miles Rollout Data Transfer with Mooncake"
summary: "Miles now supports Mooncake as a rollout data-transfer backend for heterogeneous, fragmented Python rollout batches."
date: 2026-08-11
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
home_weight: 202608110
image:
  preview_only: true
  alt_text: "Miles rollout data transfer with Mooncake"
---

## Rollout Data in Disaggregated RL Systems

Reinforcement learning for large language models combines two very different workloads: **rollout generation** and **model training**.

During rollout, inference workers run the current policy on a set of prompts and generate responses. Along the way, they produce the information required by the learning algorithm, including generated tokens, masks, log probabilities, rewards, sequence lengths, sample identifiers, and other metadata. Together, these outputs form the **rollout data** that will be consumed by the next training step.

At small scale, generation and training can share the same execution environment. At larger scale, however, modern RL systems increasingly adopt a **disaggregated architecture**, where rollout generation and training are deployed as separate worker groups, often across different processes, GPUs, or machines.

The two stages have fundamentally different resource and execution characteristics. Rollout is an inference-heavy workload whose throughput depends on decoding efficiency, batching, and request scheduling, while training relies on large, highly synchronized tensor computations. Separating them allows each side to be scaled and scheduled independently instead of forcing both workloads into the same execution pattern.

This separation also enables **pipeline-level concurrency**. Systems such as **Miles and slime** can train on rollout *N* while rollout workers are already generating rollout *N+1*. Asynchronous RL systems can therefore allow rollout and training workers to progress at different rates rather than making one stage wait for the other after every operation.
The benefit is better resource utilization and greater flexibility in how inference and training capacity are provisioned. But disaggregation also introduces a new systems boundary: **the data produced by rollout workers must now move to a different set of workers before training can consume it**.

That handoff sits directly between generation and the next policy update. A slow transfer can leave trainers waiting for data and keep memory occupied on rollout workers longer than necessary. For distributed RL systems such as Miles and slime, efficiently moving this **structured rollout data from the inference side to the training side** therefore becomes an important part of the end-to-end RL pipeline.

## What Makes RL Rollout Data Transfer Challenging

RL rollout data differs significantly from the large, regular tensors commonly moved in distributed training.

A rollout batch is usually a **heterogeneous structured object** rather than a single contiguous tensor. Depending on the framework and algorithm, it may contain generated tokens, loss masks, log probabilities, rewards, sequence lengths, sample identifiers, routing information, metadata, and other auxiliary fields. These values can be represented as tensors, NumPy arrays, Python scalar lists, variable-length per-sample arrays, bytes, or arbitrary Python objects.

Several properties make this data particularly challenging to move efficiently.

**Challenge 1: Heterogeneous Data Types and Complex Semantics**

Different rollout fields have fundamentally different representations and semantics. Dense tensors and numeric arrays can be transferred efficiently as typed buffers, while ragged sequences need row-boundary information, scalar lists need their original values preserved, and metadata or Python objects may require more general encoding. At the same time, the trainer must reconstruct the exact structure expected by the RL framework, including dtype, shape, row order, null state, and metadata. A generic serializer can handle these objects functionally, but often at the cost of extra conversion, copying, and reconstruction. Efficient rollout transfer therefore needs to understand both the physical representation and the logical structure of each field.

**Challenge 2: Highly Fragmented Memory Layout**

Rollout data can contain a very large number of small memory allocations. In the captured Miles workload, major fields such as tokens, loss_masks, and rollout_log_probs are represented as list[np.ndarray], with one NumPy array allocated for each sample. As batch size grows, transferring these fragments individually introduces repeated memory registration and Store operations, while serializing the full Python object requires walking, copying, and rebuilding a large object graph. The challenge is to turn fragmented logical data into efficient bulk transfers without losing its original structure.

**Challenge 3: Incremental Production and Partial Consumption**

In asynchronous RL pipelines, rollout data may not become ready all at once. Different DataProto fields may be completed at different times for the same rows, while new row groups may also arrive later. An efficient data path should allow producers to publish newly completed fields or rows without rewriting payloads that have already been transferred. On the consumer side, different workers may only need a subset of the available fields or rows, so they should be able to fetch only the data they actually need instead of materializing the full object every time. This enables pipeline-level asynchrony even when individual transfer operations themselves remain synchronous.

Together, these challenges make rollout data movement more than a bandwidth problem: the system must efficiently move fragmented, heterogeneous data while preserving its structure and supporting incremental production and selective consumption.

An effective data path needs to satisfy several requirements at the same time:

* **Efficiency:** avoid excessive serialization, copying, object reconstruction, and per-allocation transfer overhead.
* **Correctness:** preserve field types, shapes, row boundaries, null state, metadata, and Python-level values through the round trip.
* **Scalability:** continue to perform well as the number of samples, object fragments, and total payload size increase.
* **Flexibility:** support heterogeneous fields without forcing the RL framework to flatten or rewrite its native rollout representation.
* **Predictable handoff latency:** deliver the batch quickly enough that the trainer does not stall waiting for rollout data.

## Powering Miles Rollout Data Transfer with Mooncake

**Miles** is a high-performance reinforcement learning framework for large-scale model post-training. It combines **SGLang for high-throughput rollout generation** with **Megatron-LM for scalable training**, and also provides a PyTorch FSDP2 backend for workloads that prefer to train Hugging Face model implementations directly. Miles supports fully asynchronous RL, where rollout and training workers are decoupled and can progress independently, together with features such as fast in-loop weight updates, agentic rollout, low-precision training, and fault tolerance for large-scale production RL workloads.

This disaggregated and asynchronous design makes the rollout-to-training data path a critical part of the RL pipeline. Rollout batches are structured and heterogeneous, often containing fragmented per-sample data and framework-specific metadata, making efficient transfer and reconstruction increasingly important as workload scale grows.

**Mooncake** provides a high-performance data plane for distributed AI workloads. For RL rollout data, it extends this data plane with structured-object transfer, allowing heterogeneous and fragmented rollout objects to be moved while preserving their original structure and semantics.

Mooncake has now been integrated into Miles as a rollout data-transfer backend. On rollout data captured from Miles, the integration delivers substantially lower transfer latency than the existing Ray path: Mooncake achieves **10.6× to 14.1× faster remote GET**, while also improving PUT performance across payload sizes from 128 MiB to 4 GiB.

The result is a faster rollout-to-training handoff without changing the RL programming model, providing Miles with a more efficient data path for large, structured rollout workloads.

## How Rollout Data Moves Through the RL Pipeline

In asynchronous RL systems, rollout generation and training can progress at different speeds. Once rollout workers and trainers are deployed in separate processes or on separate machines, each completed rollout batch must cross that boundary before it can be consumed by the next training step.

A useful design principle is to separate the **control plane** from the **data plane**. The framework scheduler decides where a rollout batch should go, but it should not carry the bulk payload itself. Instead, it passes a lightweight transfer reference that excludes tensor payloads and Store chunk layouts, while still allowing JSON-safe metadata when needed. The actual rollout payload takes a separate path:

* the framework scheduler passes the transfer reference;
* Mooncake stores and transfers the payload;
* the training worker reconstructs the original object before running the training step;
* after all readers finish, the framework removes the short-lived Store object.

This separation keeps scheduling decisions lightweight while allowing the bulk rollout payload to move through a dedicated data path.

### What Miles Actually Transfers

The rollout data captured from **Miles** makes this data path concrete. For a given framework configuration, the set of fields is stable, but those fields do not share a single convenient in-memory representation.

The benchmark uses rollout-source data captured directly from Miles. Each additional sample contributes approximately **3,422 logical bytes** to the rollout dictionary. Larger benchmark cases repeat the captured samples while preserving their original field types and memory fragmentation, rather than replacing them with synthetic dense tensors.

Most of the payload is stored as per-sample NumPy arrays inside Python lists instead of as one contiguous tensor:

| Field               | Layout Before Transfer | Data Type        |  Size Per Sample | Layout Characteristic                                                                    |
| ------------------- | ---------------------- | ---------------- | ---------------: | ---------------------------------------------------------------------------------------- |
| `tokens`            | `list[np.ndarray]`     | `int32`          |         ~1,338 B | Ragged per-sample token arrays.                                                          |
| `loss_masks`        | `list[np.ndarray]`     | `int32`          |          1,024 B | One array per sample; fixed length 256 in this data set.                                 |
| `rollout_log_probs` | `list[np.ndarray]`     | `float32`        |          1,024 B | One array per sample; fixed length 256 in this data set.                                 |
| `partition`         | Python list            | `int`            |            small | High-cardinality scalar field; one distinct value per sample in the generated benchmark. |
| `sample_indices`    | Python list            | `int`            |            small | High-cardinality scalar field; one distinct value per sample.                            |
| `response_lengths`  | Python list            | `int`            |            small | Low-cardinality scalar field; all values are 256.                                        |
| `rewards`           | Python list            | `float`          |            small | Low-cardinality scalar field in this capture; all values are 0.0.                        |
| `truncated`         | Python list            | `int`            |            small | Low-cardinality scalar field in this capture; all values are 1.                          |
| `weight_versions`   | Python list of objects | list-like object | ~25 B serialized | Low-cardinality object field in this capture; still needs object-preserving encoding.    |
| `raw_reward`        | Python list            | `float`          |            small | Metadata-style list; all values are 0.0 in this capture.                                 |
| `total_lengths`     | Python list            | `int`            |            small | Per-sample total sequence length; values vary with prompt length in this capture.        |

The three array-list fields—`tokens`, `loss_masks`, and `rollout_log_probs`—contribute about **3,386 bytes per sample** and dominate the transfer volume. The smaller scalar and object fields account for far fewer bytes, but they still carry routing, length, reward, and bookkeeping state required by training.

Fragmentation matters as much as total byte count. Each of the three array-list fields contains one allocation per sample, so the number of objects grows with the batch size. At the same time, the smaller fields cannot simply be dropped or normalized away: some are identifiers, while others only happen to be constant in this particular capture. Both the typed payload bytes and the original Python values must survive the transfer and reconstruction process.

![Miles rollout object anatomy](rollout-object-anatomy.svg)

*Figure 1. One sample adds about 3,422 logical bytes in this Miles capture. Three array-list fields contribute about 98.9% of those bytes; the remaining fields are smaller but still required by training.*

### Two Rollout Handoff Protocols

Above the Store data plane, rollout data can follow different handoff protocols. Here, **protocol** refers to when an object becomes visible to consumers and what a reader is allowed to request—not whether the underlying bytes move over RDMA or TCP.

#### Flat Dict: Completed-Object Handoff

In **Miles and slime**, the current handoff begins after the complete rollout dictionary is ready. The producer calls `put(data, type="dict")`, the scheduler carries the returned reference, and the trainer calls `get` to reconstruct the original dictionary.

![Synchronous rollout transfer for Miles and slime](rollout-data-plane.svg)

*Figure 2. Miles and slime use synchronous `put` and `get` for a completed rollout dict. The reference travels through the scheduler, while Mooncake moves the payload through the Store data plane.*

The individual transfer calls are synchronous. The producer returns the reference only after `put` completes, and the trainer waits for `get` to finish before consuming the object.

This does not prevent the RL pipeline itself from being asynchronous. Miles and slime can generate rollout *N+1* while training on rollout *N*. In other words, the concurrency sits above the transfer operation: each individual handoff is synchronous, while different rollout and training stages can overlap at the pipeline level.

#### DataProto: Incremental Publication and Partial Reads

Some RL pipelines need finer-grained handoff than a completed rollout dictionary. The **DataProto** path preserves three explicit sections:

* `batch` for row-aligned tensor fields;
* `non_tensor_batch` for row-aligned non-tensor fields;
* `meta_info` for object-level values.

Unlike the completed-object handoff, DataProto allows rollout data to become visible incrementally in two dimensions.

**Fields may arrive later for the same rows.** `append_dataproto_fields()` writes a new structured object containing the newly completed fields and returns an updated handle. Existing payloads are not rewritten, and appended row-aligned fields must use the original batch size.

**New rows may arrive later.** Each completed row group is stored as a separate DataProto fragment using `put(..., type="dataproto")`. A framework index or `DataProtoCatalog` maps logical keys to fragment rows, allowing readers to compose the requested logical row order without physically extending an existing handle.

Consumers can also select **fields and rows together**. A worker that needs only two fields for part of a batch does not have to materialize the rest of the DataProto object. Mooncake skips unrequested members and, where the stored layout supports it, reads only the byte ranges corresponding to the selected rows.

![DataProto incremental publication and partial reads](dataproto-staged-transfer.svg?v=8)

*Figure 3. DataProto can grow logically by adding fields to existing rows or by publishing new row fragments. These are different operations: only the first uses `append_dataproto_fields()`. A partial GET materializes the intersection of the requested fields and rows.*

The individual operations remain synchronous. A newly published group becomes visible only after its PUT and index update complete, and GET returns only after the requested data has been materialized.

Pipeline-level asynchrony comes from publishing completed field groups or row groups at different times while other work continues. This allows rollout production, data movement, and downstream consumption to overlap without requiring each individual Store operation to become asynchronous.

## How Mooncake Preserves and Transfers a Rollout Object

Take one completed Miles rollout batch. At the producer it is a flat Python dict. Miles passes that dict to `put(data, type="dict")`; at the trainer, `get(ref, type="dict")` returns the same mapping. Four pieces of work connect those two calls.

### 1. Identify Fields and Expand Leaves

At the API boundary, `type="dict"` fixes the contract. Every top-level key remains an ordinary Miles field, even if it happens to be named `batch` or `meta_info`.

The first pass does not touch the network. It works out what is in the dict. A dense ndarray can be described directly by its dtype and shape. The Miles `tokens` field is different: it is one Python list on paper, but one ndarray allocation per sample in memory. Object arrays and nested containers may hide another level of typed data. Mooncake opens those containers, identifies the leaves, and records enough path, row, and null information to put the original structure back together on GET.

Bandwidth is not the bottleneck yet. The expensive mistake here is to give up on the structure and serialize the whole Python graph as an opaque object. Mooncake first uses a framework-provided schema, then the native type information carried by ndarrays and tensors, and only then falls back to runtime inference. The schema is particularly valuable for ambiguous fields: a rare value or an extra `None` should not send the next batch down a different codec path.

### 2. Encode Fields and Plan Their Physical Layout

Once the leaves are known, Mooncake chooses how each field should live in the Store. Treating every field alike would throw away information that is already available: tensors know their shape, ragged rows have boundaries, and Python metadata still needs its original value semantics.

| Field Shape | Stored Layout | What It Avoids |
| --- | --- | --- |
| Dense ndarray | Typed bytes plus dtype and shape. | Generic serialization and an extra decode copy. |
| Tensor | Native tensor payload when supported; a correctness fallback otherwise. | Flattening a large contiguous tensor into a Python blob. |
| List of typed ndarray rows | Values plus offsets, shapes, dimensions, and null bits. | Thousands of small serialized arrays and lost row boundaries. |
| Numeric scalar list | One typed numeric buffer. | Per-item Python serialization. |
| Byte strings | Packed bytes plus row offsets and null bits. | Per-row serialization and a whole-field temporary buffer. |
| PIL images | Pixel payload plus reconstruction metadata. | Encoding decoded pixels through a generic Python serializer. |
| Variable-length media lists | Flat media items plus row and item offsets. | Padding every sample to the same media count or losing item boundaries. |
| Per-sample tensor dictionaries | One ragged tensor sub-payload per dictionary key, plus row and null metadata. | Pickling the dictionary list or padding all media tensors to one shape. |
| Strings | Packed UTF-8 bytes plus row offsets and null bits. | Per-item Python serialization. |
| Nested or general Python values | Recursive leaves, then MessagePack or JSON only where needed. | Pickling typed children together with unrelated Python structure. |

The three large Miles fields show why this choice matters. `tokens`, `loss_masks`, and `rollout_log_probs` each arrive as one ndarray allocation per sample. Serializing every row separately creates thousands of small objects. Calling `np.concatenate` first swings too far in the other direction: it builds a second, full-size copy before the Store can send anything.

Mooncake keeps the rows where they are and builds a copy plan instead. The plan describes how to fill each Store chunk when that chunk is needed. Row boundaries, shapes, and nulls become compact metadata; the bulk values remain typed bytes. One logical field can therefore produce several physical members, such as `data`, `offsets`, `shapes`, and `nulls`, without losing its identity as one dict field. Bytes-like fields use a similar multi-buffer plan rather than one large `bytes.join`.

Multimodal data makes the value of this field-level choice easier to see. The same image may reach a framework before or after preprocessing, and Mooncake does not force every representation through one media serializer. An encoded image held as `bytes` stays a byte string. A decoded PIL image uses the `media_bytes` path: Mooncake transfers its pixel payload with the reconstruction metadata needed on GET, avoiding another PNG or JPEG encode-decode cycle. A sample may also contain a variable number of PIL images or byte strings; `media_list_ragged` flattens those items while separate row and item offsets preserve which media belong to which sample. Null rows remain explicit in each layout.

Miles currently uses another common representation: tensors produced by its multimodal processor. `multimodal_train_inputs` holds one `dict[str, Tensor] | None` per sample, with keys such as `pixel_values` and model-specific grid tensors. Its field schema selects the ragged-tensor-dict codec. Mooncake stores each dictionary key as an independent ragged tensor sub-payload, preserving dtype, shape, sample order, missing keys, and samples without multimodal input. GET rebuilds the per-sample dictionaries for Miles to assemble the model inputs. No image is converted back to PIL on this path because the trainer asked for processed tensors, not image objects.

### 3. PUT Payloads and Publish the Manifest

The encoder now knows what bytes to send, but those bytes are still scattered across memory. A large Miles batch can contain tens of thousands of ndarray allocations. Sending them in place would mean registering and unregistering a long list of small memory regions, and that setup cost can exceed the transfer itself.

Mooncake pays for one controlled copy instead. It borrows registered memory from BufferPool, fills it in chunks of up to 64 MiB, and sends those chunks through the Store. The native copy path writes ndarray rows directly into the borrowed buffer while releasing the Python GIL. It replaces a Python row-by-row loop without first creating a full concatenated array. Multi-buffer payloads take the same route, so a bytes-like field does not need a whole-field temporary either.

Large bundles can keep several chunk PUTs in flight, with a fixed limit so staging does not consume memory without bound. Small bundles stay on the simpler path and avoid paying for unnecessary parallel work. In this workload, the BufferPool copy is deliberate: one sequential memory copy is cheaper than thousands of registrations, temporary concatenations, and Store requests. True source-buffer zero-copy remains available when the caller supplies a suitable tensor-object buffer in registered memory, while native tensors have their own direct path. A backend without the pooled fast path still uses the same object contract and falls back to ordinary Store operations.

The last write is the manifest: a small map from every physical member to its Store keys and chunk ranges. Mooncake publishes it only after the metadata and payload chunks are in place. A reader therefore finds a complete bundle, not a reference to chunks that are still being written. If a payload write fails, the writer removes the keys it has already created and never publishes the manifest.

That manifest is all Miles needs because it publishes one completed dict at a time.

### 4. GET Payloads, Decode Fields, and Rebuild the Dict

The read side reverses the same path. The manifest points to the required chunks; the structured metadata says how those bytes become fields again. A straightforward implementation would still waste most of the gain by fetching every member into a new Python `bytes` object, copying it again into an ndarray, and decoding ragged rows one at a time.

On the fast path, Mooncake reads into BufferPool-backed or caller-provided destinations instead. Members that belong together are fetched together. Typed-ragged rows become views over the contiguous result buffer, and MessagePack ragged values are decoded as one stream rather than sliced and reparsed for every sample. Because the caller asked for a dict, the decoder rebuilds that dict directly; it does not construct a DataProto-shaped intermediate object and then convert it back.

Most of the GET win comes from work that no longer happens: fewer temporary objects, fewer copies, and no per-row decode loop on the typed hot path. That matters for Miles because the trainer needs both halves of the object, the large numeric fields and the small Python fields, before it can begin the next step.

The returned arrays may still point into BufferPool memory, so their lease follows the result into the trainer. Once training is finished with the dict, Miles calls `release_result()` to return that memory. Miles removes the short-lived Store reference only after its consumers are done, at which point Mooncake can reclaim the payload chunks and manifest.

![Mooncake structured-object transfer architecture](structured-transfer-architecture.svg)

_Figure 2. A Miles dict is expanded into typed members and metadata, copied into registered BufferPool chunks where needed, and published through a bundle manifest. GET follows the same structure in reverse._

## Performance Results

For a fixed framework configuration, switching models does not by itself change the field contract. Transfer size instead follows the number of samples, prompt and response lengths, and optional fields such as teacher log probabilities, routing information, or multimodal inputs.

### Benchmark Payload

Miles generated the source data with Qwen3-0.6B on math prompts (`rollout_id=0`, 8 source samples). Every response in this capture has 256 tokens. Averaged across those samples, the logical payload breaks down as follows:

| Part of One Captured Sample | Calculation | Logical Bytes |
| --- | ---: | ---: |
| `tokens` | Average prompt-plus-response token count x 4 B (`int32`) | ~1,338 B |
| `loss_masks` | 256 entries x 4 B (`int32`) | 1,024 B |
| `rollout_log_probs` | 256 entries x 4 B (`float32`) | 1,024 B |
| Scalar and object fields | IDs, lengths, rewards, flags, and version metadata | ~36 B |
| **Total** | | **~3,422 B** |

The first three fields account for about 3,386 bytes, or 98.9% of this particular sample layout. Larger benchmark payloads repeat the eight captured samples, preserving their field types and fragmented allocation pattern while increasing the logical sample count. The calculation describes this capture; it does not define a fixed Miles sample size.

![Miles rollout object anatomy](rollout-object-anatomy.svg)

_Figure 3. The measured composition of one sample in the Qwen3-0.6B benchmark capture._

### Transfer Results

The benchmark measures the completed flat-dict handoff used by Miles and compares the Miles Ray backend with Mooncake structured-object transfer.

PUT is one timed backend call after the producer loads the payload. GET is the mean of three remote-consumer trials after one warmup. Reference serialization and scheduler handoff are outside the timed region. These numbers cover payload transfer and reconstruction, not end-to-end training throughput.

Across the tested payload sizes, Mooncake makes Miles GET roughly 10–14x faster than the Ray backend. PUT improves by about 1.2–1.6x.

![Miles GET latency benchmark](miles-get-latency.svg)

_Figure 4. Mooncake provides roughly 10–14x faster GET for this fragmented Miles rollout layout. The vertical axis uses a logarithmic scale._

PUT shows a smaller gain. Its timed path includes Python-object traversal, ragged-row packing, metadata and manifest construction, and payload transfer. GET improves more for this workload, and the trainer must finish it before starting the training step.

## Beyond the Miles Integration

The same structured-object layer also supports DataProto-shaped batches. It preserves the `batch`, `non_tensor_batch`, and `meta_info` sections, while adding staged publication and partial reads for frameworks whose rollout contract carries row and field semantics.

DataProto producers do not have to wait for one complete batch. They can add fields for existing rows with `append_dataproto_fields()` or publish newly completed row groups as separate fragments. Earlier payloads are not rewritten, and readers use the resulting handle to select only the fields and rows they need. A Catalog locates those logical row and field fragments; each fragment's manifest still describes its physical Store layout. Each Store operation is synchronous; the asynchronous pipeline comes from publishing completed stages at different times while rollout and training continue elsewhere.

This is a different handoff protocol from the completed flat dict used by Miles, so it remains a secondary topic here. The same machinery is also useful outside RL: it has already been validated and used for COO sparse tensors, where typed `indices` and `values` travel with `shape` and structural metadata.

## What Comes Next

The Miles integration establishes the basic data path. The main priority now is to validate, optimize, and integrate it across a wider range of RL workloads.

- **Validate and integrate more RL workloads.** Multimodal and agentic RL, VLA and world-model training, and RL for video-generation or diffusion models produce different combinations of media, trajectories, actions, rewards, and intermediate state. The next step is to capture those real rollout objects, verify their contracts in end-to-end training, and connect the frameworks that produce them.
- **Optimize for their actual data shapes.** Media-heavy samples, long or incrementally growing trajectories, and batches with many small fields stress different parts of the path. Profiling real workloads will guide codec, packing, request-count, metadata, and partial-read improvements instead of relying on dense synthetic buffers.
- **Isolate rollout data from KV cache workloads.** Mooncake needs separate accounting, quotas, and eviction policy for short-lived rollout data and KV cache data, so a burst of rollout traffic cannot evict latency-sensitive cache entries.
- **Explore other structured AI data paths.** Multimodal preprocessing outputs and other intermediate artifacts may fit the same typed-payload-plus-metadata model. They should be added only after their real layout, access pattern, and lifetime are understood.

## Acknowledgements

Xinpeng Zhao ([@zxpdemonio](https://github.com/zxpdemonio)) led the Mooncake structured-transfer design and implementation. Yufeng He ([@he-yufeng](https://github.com/he-yufeng)) and [@yokinoshitayoki](https://github.com/yokinoshitayoki) reviewed the Mooncake implementation.

The Miles integration was developed and refined with Teng Ma ([@stmatengss](https://github.com/stmatengss)) and [@fzyzcjy](https://github.com/fzyzcjy). Thanks to [@guapisolo](https://github.com/guapisolo) and Xuchun Shang ([@XucSh](https://github.com/XucSh)) for CI work and review, and to Bo Gao ([@Bo-Vincent](https://github.com/Bo-Vincent)) for testing and validation.

## Related Links

- Mooncake project: [https://github.com/kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake)
- Mooncake documentation: [https://kvcache-ai.github.io/Mooncake/](https://kvcache-ai.github.io/Mooncake/)

---
title: "Moving RL Rollout Data with Mooncake Structured-Object Transfer"
summary: "Mooncake moves dict and DataProto rollout batches as structured objects, with incremental DataProto publication and field- or row-level partial reads."
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
  alt_text: "Mooncake structured-object transfer for RL rollout data"
---

Every policy update starts with data produced by rollout workers. They run the current policy and collect prompts, generated tokens, masks, log probabilities, rewards, and metadata. That batch then has to reach a trainer before the next learning step can begin. A slow handoff leaves the trainer waiting and keeps rollout-worker memory occupied; an incorrect one changes the training input itself.

This path is now available in Miles: [radixark/miles#591](https://github.com/radixark/miles/pull/591) adds Mooncake as a rollout data-transfer backend without changing the framework's scheduler, rollout manager, or trainer.

## Why Rollout Data Needs Structured Transfer

A rollout batch is not one contiguous tensor or a convenient byte string. It is a dict or DataProto object containing tensors, ndarrays, ragged per-sample values, scalar lists, metadata, and sometimes multimodal fields. Putting a generic serializer in front of the Store misses three parts of the problem.

1. **The object may be highly fragmented in memory.** What looks like one field in Python can be a list of thousands of small NumPy arrays. Pickle has to walk, copy, and later rebuild that object graph. Writing every leaf separately simply turns the same fragmentation into many Store operations and repeated memory registration. Mooncake keeps the structure visible and, when the fast path is available, packs fragmented data into reusable registered buffers.
2. **Different field types need different performance paths.** Dense tensors, ragged rows, scalar lists, bytes, and Python objects do not benefit from the same encoding. Sending all of them through one generic serializer adds conversion and reconstruction work and gives up bulk copies, typed buffers, and compact ragged layouts. Mooncake chooses a codec for each field while retaining the dtype, shape, row boundaries, null state, and Python values needed on the other side.
3. **DataProto fields or row groups may be ready at different times.** A pipeline should be able to publish completed fields for existing rows or publish newly completed rows without rewriting earlier payloads, then read only the fields and rows it needs. Each Mooncake call is synchronous; a framework can still build an asynchronous pipeline by publishing completed groups at different times.

## References on the Control Plane, Payloads on the Data Plane

Asynchronous RL lets rollout and trainer workers run at different speeds. Once they live in separate processes or on separate machines, each completed batch has to cross that boundary before training can use it.

The scheduler should decide where the batch goes, not carry the batch itself. It passes a transfer reference that excludes tensor payloads and Store chunk layouts, although the reference may still contain JSON-safe metadata. The payload takes a separate route:

- the framework scheduler passes the transfer reference;
- Mooncake stores and transfers the payload;
- the training worker reconstructs the original object before running the step;
- framework cleanup removes the short-lived Store object after its readers finish.

## Two Rollout Handoff Protocols

Two handoff patterns sit above the Store data plane. Here, protocol means when an object becomes visible and what a reader may request, not whether the bytes move over RDMA or TCP.

### Flat Dict: Completed-Object Handoff

In Miles and slime, the handoff starts after the complete rollout dict is ready. The producer calls `put(data, type="dict")`, the scheduler carries the returned reference, and the trainer calls `get` to rebuild the dict.

![Synchronous rollout transfer for Miles and slime](rollout-data-plane.svg)

_Figure 1. Miles and slime use synchronous `put` and `get` for a completed rollout dict. The reference travels through the scheduler, while Mooncake moves the payload through the Store data plane._

Both frameworks also provide asynchronous RL loops, but that concurrency sits above the transfer call. They can generate rollout *N+1* while training on rollout *N*; the producer returns the reference only after `put` completes, and the trainer waits for `get` before consuming the object.

### DataProto: Incremental Publication and Partial Reads

The DataProto path preserves three sections: row-aligned tensor fields in `batch`, row-aligned non-tensor fields in `non_tensor_batch`, and object-level values in `meta_info`. Incremental publication can happen in two dimensions:

- **Fields arrive later for the same rows.** `append_dataproto_fields()` writes a structured object for the new fields and returns an updated handle. Earlier payloads are not rewritten, and the appended row-aligned fields must use the original batch size.
- **New rows arrive later.** Each completed row group is stored as a separate DataProto fragment with `put(..., type="dataproto")`. A framework index or `DataProtoCatalog` maps logical keys to fragment rows, so readers can compose the requested row order without physically extending an existing handle.

A consumer can select fields and rows together, so a worker that needs two columns for part of the batch does not have to materialize the rest. Mooncake skips unrequested members and, where the stored layout supports it, reads only the byte ranges for the selected rows.

![DataProto incremental publication and partial reads](dataproto-staged-transfer.svg?v=8)

_Figure 2. DataProto can grow logically by adding fields to existing rows or by publishing new row fragments. These are different operations: only the first uses `append_dataproto_fields()`. A partial GET materializes the intersection of the requested fields and rows._

The calls themselves are synchronous: a group becomes visible after its PUT and index update finish, and GET returns after the requested data is materialized. Pipeline-level asynchrony comes from the framework publishing completed field or row groups at different times while other work continues.

## What Miles Sends

_A fragmented, heterogeneous rollout batch_

The captured Miles batch makes the layout problem concrete. Its fields are stable for a given configuration, but they do not share one useful in-memory representation.

The benchmark uses rollout-source data captured from Miles. Each additional sample contributes about 3,422 logical bytes to the dict. Larger cases repeat those samples, preserving the original field types and fragmentation instead of replacing them with synthetic dense tensors.

Most bytes arrive as per-sample arrays held in Python lists, not as one dense tensor:

| Field | Layout Before Transfer | Data Type | Size Per Sample | Layout Characteristic |
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

The three array-list fields add up to about 3,386 bytes per sample and dominate transfer volume. The smaller scalar and object fields still carry routing, length, reward, and bookkeeping state.

Fragmentation matters as much as byte count. Each array-list field contains one allocation per sample, so the object count grows with the batch. The smaller fields cannot be dropped or normalized: some are identifiers, while others merely happen to be constant in this capture. Both the typed bytes and the original Python values must survive the round trip.

![Miles rollout object anatomy](rollout-object-anatomy.svg)

_Figure 3. One sample adds about 3,422 logical bytes in this Miles capture. Three array-list fields contribute about 98.9% of those bytes; the remaining fields are smaller but still required by training._

## How Mooncake Preserves and Transfers a Rollout Object

Mooncake accepts the dict or DataProto object already used by the framework; callers do not flatten it first. The object contract, field meaning, and physical payload layout stay separate.

### Schema-Driven Field Encoding

A Python list carries too little information on its own: it might hold scalars, typed ragged rows, bytes, or arbitrary objects. Mooncake can infer an encoding from runtime values, so a schema is not required for every field. The recommended integration is still for the framework to provide schemas for ambiguous or performance-sensitive fields because the framework owns the stable data contract. Numeric ndarrays and tensors already expose native type information, although native tensor transport still depends on backend support.

An explicit schema fixes the codec and can also declare the intended section while batch size, sequence length, and values change. This prevents an unusual batch from selecting a different inferred path. Structured metadata records how one instance was encoded, and the bundle manifest records its Store keys and chunk layout. A DataProto handle maps row-aligned fields to stages and sections.

### Dict and DataProto Contracts

The `type` argument selects the object contract, not just a codec. Miles uses `type="dict"` for a flat rollout dictionary. DataProto-based frameworks use `type="dataproto"` because their fields carry batch semantics.

The dict path preserves a flat mapping and reconstructs the same mapping on GET. Its keys do not acquire DataProto meaning: a field named `batch` or `meta_info` remains an ordinary dict field.

A DataProto-like object has three explicit sections:

- `batch` contains tensor or ndarray fields indexed by batch row;
- `non_tensor_batch` contains per-row non-tensor fields;
- `meta_info` contains object-level metadata.

The DataProto handle records the section of each row-aligned field in its field index, while `meta_info` is carried separately. Mooncake enforces one batch size across `batch` and `non_tensor_batch`. GET can return the three-section envelope as a dictionary or reconstruct the framework's DataProto class.

### Tensor, Non-Tensor, and Multimodal Payloads

Field representation and schema determine the transfer path. Numeric ndarrays use typed members. Tensors use native Store paths when the backend supports them and a serialized fallback otherwise.

Scalar values, strings, bytes, lists, JSON-like values, object arrays, and ragged fields use structured codecs. Their metadata preserves the field path and any dtype, shape, row boundary, or null state needed to reconstruct the value.

Multimodal is not a third contract. Decoded image tensors follow the tensor path; image bytes, media metadata, and external references use non-tensor codecs.

### PUT and GET Data Path

The public API stays small: `put(data, type=...)` and `get(ref, type=...)`. DataProto producers may append fields, and consumers may select `fields` and `rows` during GET. Scheduling, trainer placement, and the decision that all readers have finished remain framework responsibilities.

The transfer path is:

1. `put` receives a flat dict or DataProto object.
2. Mooncake walks the object into fields and leaves, then applies the field schema or native type handling.
3. The encoder creates payload members and the structured metadata needed to reconstruct them.
4. The bundle manifest records the Store keys and chunk layout for those payloads.
5. Mooncake writes the payloads through the Store, using BufferPool staging when the capability is available and the transfer policy permits it.
6. `get` resolves the requested object, reads the required payloads, decodes them, and reconstructs the result.

Each flat-dict key remains one logical field, but its codec may use several physical members, such as values, offsets, and a null mask for ragged data. DataProto locations also record whether a row-aligned field belongs to `batch` or `non_tensor_batch`; object-level `meta_info` remains separate. The returned handle carries routing and field-location metadata instead of the bulk tensor payloads.

| Field Shape | Mooncake Handling | Why It Matters |
| --- | --- | --- |
| Numeric ndarray | Store as a typed member with dtype and shape metadata. | Avoids turning numeric buffers into one opaque Python object. |
| Tensor | Use the native Store path when available, with a serialized fallback. | Preserves correctness across backends while retaining the fast path where supported. |
| Ragged typed arrays | Pack values with row-boundary metadata. | Preserves per-sample boundaries for fields such as Miles `tokens`. |
| Scalar lists and object fields | Use structured codecs that preserve Python values. | Keeps routing IDs, rewards, lengths, and object metadata intact. |
| Bytes, strings, and multimodal metadata | Encode according to the field representation. | Supports media payloads, references, and metadata without forcing them into tensor form. |
| Missing or explicit `None` values | Preserve null state in structured metadata. | Prevents data loss during reconstruction and partial reads. |

### DataProto Handles, Catalog, and Partial Reads

Incremental DataProto publication is optional and is not the transfer mode currently used by Miles or slime. `append_dataproto_fields()` adds fields without rewriting existing payloads and returns a new handle for the expanded view. The previous handle remains a valid snapshot of the earlier view; it does not gain the appended fields.

Pipelines that address samples by logical key can first write an immutable, single-stage fragment and then register its handle with `DataProtoCatalog`. The Catalog maps `(partition, key, field)` to fragment rows, and a reader resolves those mappings before fetching data. It stores metadata only; data transfer and payload lifetime remain outside the Catalog.

Each Store PUT or GET remains synchronous. Pipeline-level asynchrony comes from field or row groups being published at different times. During a partial read, the consumer selects fields and supplies rows as a slice or index list. Mooncake skips unrequested members and uses range reads when the stored layout permits; the reconstructed result retains the original DataProto sections and requested row order.

### BufferPool and Fragmented Payloads

Registering every small allocation can cost more than copying the data once. When BufferPool and the Store fast-path APIs are available, and policy permits them, Mooncake stages fragmented payloads in reusable registered buffers. Otherwise it uses the ordinary Store path.

That one copy can replace many register/unregister calls with a few bulk copies and Store writes. Typed ragged rows are packed into contiguous chunks within each field, with row boundaries stored beside the values. Eligible large contiguous tensors still take the direct path.

During GET, Mooncake uses structured metadata and bundle manifests to derive the payload keys and ranges required by the selection, fetches and decodes those payloads, and reconstructs the result. The optional DataProto Catalog identifies the published fragment for a logical field; it does not describe the fragment's physical layout.

Rollout data is short-lived, but release order still matters. A consumer calls `release_result()` after it is finished with a BufferPool-backed GET result. Once no reader can use a direct DataProto handle, the framework calls `cleanup_dataproto()` to remove the referenced Store payloads and manifests. Removing Catalog entries does not perform that physical cleanup.

![Mooncake structured-object transfer architecture](structured-transfer-architecture.svg)

_Figure 4. The codec hands one structured bundle to the Bundle Store. The manifest is the small entry index, while typed payloads carry most bytes. The payload transport uses reusable BufferPool memory for eligible fragmented PUT and GET paths, avoiding repeated buffer registration._

## Performance Results

For a fixed framework configuration, switching models does not by itself change the field contract. Transfer size instead follows the number of samples, prompt and response lengths, and optional fields such as teacher log probabilities, routing information, or multimodal inputs.

The benchmark measures the completed flat-dict protocol used by Miles; it does not cover staged DataProto publication or partial reads. It compares the Miles Ray backend with Mooncake structured-object transfer. The source data was generated by Miles with Qwen3-0.6B on math prompts (`rollout_id=0`, 8 captured rollout-source samples). Larger cases repeat the same field layout and per-sample representation, varying only the logical sample count shown in the table.

PUT is one timed backend call after the producer loads the payload. GET is the mean of three remote-consumer trials after one warmup. Reference serialization and scheduler handoff are outside the timed region. These numbers cover payload transfer and reconstruction, not end-to-end training throughput.

| Payload | Samples / Transfer Round | Mooncake PUT | Ray PUT | Mooncake GET | Ray GET | GET Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 MiB | 39,221 | 583 ms | 930 ms | 47.4 ms | 501.3 ms | 10.6x |
| 512 MiB | 156,888 | 2,830 ms | 3,489 ms | 173.2 ms | 2,134.2 ms | 12.3x |
| 1 GiB | 313,776 | 4,636 ms | 7,469 ms | 356.3 ms | 5,035.4 ms | 14.1x |
| 4 GiB | 1,255,104 | 20,320 ms | 31,830 ms | 1,425.9 ms | 20,034.3 ms | 14.1x |

![Miles GET latency benchmark](miles-get-latency.svg)

_Figure 5. Mooncake provides 10.6x to 14.1x faster GET for this fragmented Miles rollout layout. The vertical axis uses a logarithmic scale._

Mooncake GET sustains roughly 2.6 to 2.9 GiB/s on these payloads; Ray reaches about 0.20 to 0.25 GiB/s in the same setup. For this fragmented Python layout, the Ray object-store GET path is slower than fetching Mooncake's typed members and reconstructing the framework object at the consumer.

PUT shows a smaller gain. Its timed path includes Python-object traversal, ragged-row packing, metadata and manifest construction, and payload transfer. GET improves more for this workload, and the trainer must finish it before starting the training step.

The structured-object path is not limited to rollout data. It can also carry layouts such as a COO sparse tensor represented by typed `indices` and `values` plus `shape` and metadata.

## Acknowledgements

This work crossed several repository boundaries, and so did its review and validation. The Miles integration has merged. The slime and ROLL contributors are acknowledged for design feedback, review, and testing during development; those downstream integrations had not merged when this post was written. Some contributors appear in more than one group:

- **Mooncake implementation and review:** Xinpeng Zhao ([@zxpdemonio](https://github.com/zxpdemonio)), Yufeng He ([@he-yufeng](https://github.com/he-yufeng)), [@yokinoshitayoki](https://github.com/yokinoshitayoki), and Teng Ma ([@stmatengss](https://github.com/stmatengss)).
- **Miles integration and validation:** Xinpeng Zhao ([@zxpdemonio](https://github.com/zxpdemonio)), Teng Ma ([@stmatengss](https://github.com/stmatengss)), [@fzyzcjy](https://github.com/fzyzcjy), [@guapisolo](https://github.com/guapisolo), and Xuchun Shang ([@XucSh](https://github.com/XucSh)).
- **slime design feedback and testing:** Xinpeng Zhao ([@zxpdemonio](https://github.com/zxpdemonio)), Zilin Zhu ([@zhuzilin](https://github.com/zhuzilin)), Teng Ma ([@stmatengss](https://github.com/stmatengss)), Bo Gao ([@Bo-Vincent](https://github.com/Bo-Vincent)), and Lei Li ([@lilei199908](https://github.com/lilei199908)).
- **ROLL design feedback and testing:** Xinpeng Zhao ([@zxpdemonio](https://github.com/zxpdemonio)), Haizhou Zhao ([@hydrozhao](https://github.com/hydrozhao)), Zhiyuan Cheng ([@SendoRay](https://github.com/SendoRay)), and Wei Gao ([@gaow0007](https://github.com/gaow0007)).

## Related Links

- Miles PR: [radixark/miles#591](https://github.com/radixark/miles/pull/591)
- DataProto structured-object usage: [Mooncake documentation](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/api-reference/python/dataproto-structured-object-transfer.md)
- Mooncake project: [https://github.com/kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake)
- Mooncake documentation: [https://kvcache-ai.github.io/Mooncake/](https://kvcache-ai.github.io/Mooncake/)

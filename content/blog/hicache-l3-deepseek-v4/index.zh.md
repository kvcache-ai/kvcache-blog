---
title: "HiCache L3 面向混合注意力模型：SWA 检查点、Group Semantics 与 78% 存储缩减"
summary: "混合注意力架构将压缩 MLA 与大型滑动窗口缓存耦合——当完整 SWA 状态逐页卸载到 Mooncake Store 时，其存储成本是 MLA 的 5 倍。本文覆盖完整的 HiCache L3 优化栈：统一 Radix 树中的周期性 SWA 检查点、窗口感知 LRU 刷新、叶节点锁裁剪、C4/C128 压缩状态卸载、draft KV 池注册，以及 Mooncake group semantics 保证多对象一致性驱逐——合力将存储从每百万 token 18 GB 降至 4 GB，同时在 Agent 工作负载上保持 90% 以上缓存命中率。"
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
  alt_text: "HiCache L3 面向混合注意力模型的 SWA 检查点与 Group Semantics"
---

当你把 DeepSeek V4 的一百万 token KVCache 卸载到分布式存储时，大部分字节不是压缩后的 MLA 注意力状态。它们是滑动窗口注意力（SWA）缓存——模型在前几层使用的局部窗口 KV。

在当前硬件上，使用默认的逐页卸载策略，在 L3 层（Mooncake Store）存储一百万 token 的 DeepSeek V4 Flash KVCache 大约需要 18 GB。压缩后的 MLA 部分——真正捕获长距离上下文的部分——只占其中约 3.6 GB。其余都是仅为局部注意力窗口服务的 SWA 状态。

这个比例令人不安。它意味着 L3 层被设计上只对一小片 token 邻域有意义的状态所主导。如果你能在缓存命中时重新计算这些状态而不是存储它，存储成本就会大幅下降。

本文覆盖了完整的改动集合——从周期性检查点机制本身，到使其在实践中工作的窗口感知驱逐修复，再到验证权衡的 Agent 工作负载 benchmark。

## 背景：DeepSeek V4 的混合 KV 架构

DeepSeek V4 使用混合注意力设计，结合了两种根本不同的 KV 存储模式：

**全注意力层**使用 Multi-head Latent Attention（MLA），将 KV cache 压缩为低秩潜在空间。对于一百万 token，MLA 缓存约 3.6 GB——紧凑是因为每个 token 的状态是一个短潜在向量而非完整的 key-value head。

**滑动窗口层**（前几个 transformer 层）对固定窗口使用标准多头注意力。这些层为最近 `sliding_window_size` 个 token（DeepSeek V4 为 128 token）维护全精度 KV。但在 radix 树中，一个前缀的*存储* SWA 状态跨越整个前缀，而不仅是活跃窗口——因为缓存命中时，你需要相对于命中点的窗口，与相对于最新 token 的窗口不同。

不对称性是明显的：SWA 缓存大约是 MLA 缓存的 5 倍。在百万 token 规模下，这个 5 倍将 3.6 GB 变成每序列 18 GB 的 L3 消耗。

## 统一 Radix 树与 SWA

SGLang 的 `UnifiedRadixCache` 以 radix 树管理 KVCache，每个节点持有一段 token 位置及其关联的 GPU 内存块。对于 DeepSeek V4 这样的混合模型，树同时追踪*两个*池：

- **全注意力 KV 池**用于 MLA 块——遵循标准分页 KV 布局。
- **SWA KV 池**用于滑动窗口块——有自己的分配器（`SWATokenToKVPoolAllocator`）、自己的锁语义和自己的驱逐 LRU。

统一树中的 `SWAComponent` 拥有滑动窗口特定的逻辑：哪些节点在设备上有存活的 SWA 状态、哪些已备份到宿主机、哪些已被驱逐。它维护独立的 `swa_protected_size` 计数器和自己的驱逐尾部。

这种分离至关重要，因为 SWA 的正确驱逐策略与全注意力 KV *不同*。全注意力状态只要前缀存活就有价值。SWA 状态只对相对于任何活跃 decode 位置的最后 `sliding_window_size` 个 token 有价值。一个将两者同等对待的原版 LRU 会做出错误选择。

## 问题 1：应该死亡的 SWA 节点继续存活

第一个问题（在 [#26615](https://github.com/sgl-project/sglang/pull/26615) 修复）很微妙。每次前缀缓存匹配——以及插入时的每次下行遍历——所有匹配的祖先都会被提升到 MRU 位置。这对全注意力 KV 是正确的，但对 SWA 来说，只有最后 `sliding_window_size` 个 token 是可达的。

后果：非常老的 SWA 节点（在所有存活请求的活跃窗口之外）持续被刷新到 MRU 而永不被驱逐。来自其他分支的真正热的 SWA 缓存被推向 LRU 尾部并首先被驱逐。在多轮工作负载下，SWA 池被无法再贡献任何命中的缓存填满，命中率退化，驱逐抖动上升。

修复是**窗口感知 LRU 刷新**：在前缀缓存命中期间遍历匹配祖先时，只提升落在匹配点 `sliding_window_size` token 范围内的 SWA 节点。窗口外的节点不刷新 LRU 时间戳，自然老化淘汰。

## 问题 2：叶节点 512 倍过度保护

第二个问题（在 [#26777](https://github.com/sgl-project/sglang/pull/26777) 修复）是分块 prefill 与 SWA 锁计数之间的交互。

当一个请求锁定 radix 树叶节点时，`inc_lock_ref` 保护该叶 `len(leaf.value)` 个 SWA token——即使 SWA 只需要最后 `sliding_window_size` 个 token。在 `--chunked-prefill-size 65536` 和 `sliding_window=128`（DeepSeek V4 参数）下，单个叶节点可以锁住 **65,536 个 SWA 槽位而不是 128 个**——512 倍的过度保护。

这使 `swa_protected_size` 膨胀约 `chunked_prefill_size / sliding_window_size` 倍，导致 SWA 池过早耗尽和回退抖动。修复是 `SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT`：启用后，树会拆分叶节点使锁定的 SWA 区域最多为 `sliding_window_size` token，匹配实际 SWA 需求。

## 问题 3：被锁 SWA 叶节点搁浅在驱逐中

第三个问题（在 [#28161](https://github.com/sgl-project/sglang/pull/28161) 修复）出现在混合 SWA+Mamba 工作负载下。提前释放优化（`SGLANG_OPT_SWA_RELEASE_LEAF_LOCK_AFTER_WINDOW`）在 `UnifiedRadixCache` 上静默无效，因为 `dec_swa_lock_only` 方法只存在于遗留后端。这导致被全锁定的叶节点搁浅在 SWA LRU 中；一旦 SWA 驱逐选中它们，`_cascade_evict` 的 `cd.lock_ref == 0` 断言就会触发。

修复将 `dec_swa_lock_only` 移植到统一缓存：沿祖先遍历到 `swa_uuid_for_lock` 边界，释放 SWA 槽位，在叶节点上从 SWA LRU 分离，并确保 `full_lock_ref >= swa_lock_ref >= mamba_lock_ref` 不变量在共驻的低优先级锁之间成立。

## 机制：周期性 SWA 检查点

上述驱逐正确性问题解决后，L3 检查点机制（[#27557](https://github.com/sgl-project/sglang/pull/27557)）变得可行。

洞察是 SWA 状态不需要以页粒度存储。它只需要以*检查点*粒度存储——间隔大到缓存命中时系统可以重计算最后一个检查点与命中点之间的 SWA 状态。

实现位于 `swa_component.py` 中的 `SWAComponent`：

- 新的 `--swa-checkpoint-interval` 服务器参数（通过 `server_args.py` 和 `cache_init_params.py` 暴露）设置以 token 为单位的检查点周期。
- 不再为每个页面卸载 SWA 状态，`SWAComponent` 只在每 N 个 token 写一次完整 SWA 快照到 L3。
- 缓存命中时，系统加载压缩的 MLA 状态（始终按页存储）和命中点之前最近的 SWA 检查点。
- 对于检查点与命中点之间的 token，系统将命中对齐到检查点边界——实际上要求匹配的前缀是检查点间隔的倍数。间隙中的 token 以仅 SWA 层活跃的方式重新 prefill。

第一个版本（v1）*不*实现间隙的仅 SWA 部分 prefill。它简单地将缓存命中对齐到检查点边界，意味着落在两个检查点之间的命中会回退到前一个检查点并对间隙 token 重跑完整 prefill。这实现起来更简单且已能交付存储节省；仅 SWA 重计算是后续优化。

## 数据

在 DeepSeek V4 Flash 上用 context-parallel 4、Hopper GPU、Mooncake Store 作为 L3 后端、输入一百万 token 测量：

| SWA 检查点间隔 | L3 存储 | 降幅 | 命中时 TTFT (avg@5) | TTFT 增量 |
|---------------|---------|------|---------------------|-----------|
| 256 token（默认）| 18.25 GB | 基线 | 3039 ms | 基线 |
| 2,048 token | 5.74 GB | −68% | 3536 ms | +16% |
| 16,384 token | 4.03 GB | −78% | 4660 ms | +53% |

2K 间隔：每百万 token 节省 12 GB；命中时多半秒 TTFT，而本来就要 3 秒。对大多数服务场景这是强默认值。

16K 间隔：逼近理论最小值（仅 MLA 缓存）；适合存储成本主导且缓存命中不频繁的工作负载。

## Agent 工作负载：SWA 检查点改变了整个局面

上面的 L3 检查点数据是针对单次长上下文填充的。更戏剧性的结果来自 Agent 工作负载，其中 SWA 检查点 + 后缀保留（[#26907](https://github.com/sgl-project/sglang/pull/26907)）的组合彻底改变了运行区间。

#26907 引入了一个互补机制：**驱逐时保留尾部 SWA 后缀**。不再在内存压力下驱逐所有 SWA 状态，驱逐 pass 将内部节点修剪为页对齐的尾部滑动窗口后缀——保留刚好足够的 SWA 在设备上服务后续缓存命中而无需完整重载。

DeepSeek V4（GB300, 1P/1D, 并发 30, HiCache 关）上的 Agent benchmark 结果非常惊人：

| SWA Ratio | 变体 | 缓存命中 | 输出 tok/s | TTFT p90 |
|-----------|------|---------|-----------|----------|
| 0.015 | 基线 | 8.18% | 206.67 | 163.55 s |
| 0.015 | 检查点间隔=81920 | 94.49% | 833.02 | 5.68 s |
| 0.015 | 仅结构性检查点 | 89.36% | 706.04 | 21.25 s |

缓存命中率从 **8% 跃升至 94%**。输出吞吐量翻了四倍。TTFT p90 从 **163 秒降至 5.68 秒**。这不是渐进改善——这是量级变化。

机理：通过保留稀疏 SWA 检查点（而非密集的逐页状态），有限的 SWA 池有更多空间给真正热的缓存条目。被驱逐的条目现在有检查点桩位允许快速重载。净效果是几乎所有 Agent 逐轮请求都命中缓存而不是做完整 prefill。

HiCache 启用后（3P/1D, 并发 500），改善更温和因为基线命中率已经很高，但仍有意义：

| SWA Ratio | 变体 | 缓存命中 | 输出 tok/s | TTFT p90 |
|-----------|------|---------|-----------|----------|
| 0.01 | 基线 | 89.47% | 5,216.81 | 127.55 s |
| 0.01 | 检查点间隔=81920 | 92.19% | 6,488.80 | 91.53 s |

即使基线命中率 90%+，检查点 + 后缀保留仍然多挤出 24% 吞吐和 28% TTFT 降低。

## 分布式协调的 SWA 事件元数据

对于 L3 层为共享 Mooncake Store 池的分离式部署，系统需要跨节点通信 SWA 检查点状态。[#26579](https://github.com/sgl-project/sglang/pull/26579) 扩展了 KV 事件协议的 SWA 感知元数据：

- `BlockStored` 事件增加可选的 `swa_sliding_window_size` 和 block 作用域的 `swa_valid_from` 字段。
- 当 SWA 缓存状态变化（包括驱逐时的 tombstone 拆分），统一缓存发出元数据刷新事件。
- `swa_valid_from` 是 block/节点作用域的：当该 block 自身的 SWA 状态缺失或被 tombstone 时设置；刷新恢复时清除。

消费者契约：匹配更长前缀时应沿匹配的 block 路径聚合元数据，使用最深的非空 `swa_valid_from` 配合 `swa_sliding_window_size` 来知道需要重计算多远。不理解新元数据的消费者忽略尾部字段，保持现有的仅全注意力行为。

这让 Mooncake Store conductor 或 PD 分离式部署中的远程 decode 实例知道："这个前缀缓存在 L3 中，但 SWA 状态只存储在检查点 491520——如果你在 token 524288 命中，需要重载后重计算 32K token 的 SWA。"

## PD 分离式交互

DeepSeek V4 的混合 SWA 对 prefill-decode（PD）分离创建了特定约束。[#26689](https://github.com/sgl-project/sglang/pull/26689) 解决了准入问题：

修复前，`PrefillBootstrapQueue` 对混合 SWA 模型无条件将 `max_total_num_tokens` 限制在 `swa_max_total_num_tokens`。这使有效的长上下文 DeepSeek V4 PD 请求在分块 prefill 能运行之前就被拒绝准入。

修复：当 `chunked_prefill_size > 0` 时，按 `full_max_total_num_tokens`（MLA 池预算）限制 PD prefill 准入，因为 prefill worker 只需要活跃 chunk 的 SWA 容量，而非完整 token 预算。SWA 池服务滑动窗口；它永远不会同时持有整个前缀。

这是 L3 检查点在分离式模式下工作的前提：没有它，prefill 侧的百万 token prefill 会在准入时被拒绝。

## 完整图景：什么控制什么

对运营者来说，调优旋钮是：

**`--swa-checkpoint-interval N`**（服务器参数）：每 N 个 token 写一次 SWA 快照到 L3。默认 256（密集）。推荐设为 2048 获得存储/TTFT 的最佳权衡；设为 16384 获得最大存储节省。

**`SGLANG_SWA_CACHE_CHECKPOINT_MIN_TOKEN_INTERVAL`**（来自 #26907 的环境变量）：控制驱逐时设备上的 SWA 后缀保留。设为 0 跳过 SWA 检查点柔性 pass（密集驱逐行为）。设为 -1 仅做结构性检查点。设为正 token 间隔（如 81920）在设备内存压力下保留稀疏的尾部窗口检查点。

**`SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT`**（环境变量，自 #26777 起默认 True）：防止分块 prefill 叶节点的 512 倍 SWA 过度保护。

**`SGLANG_OPT_SWA_RELEASE_LEAF_LOCK_AFTER_WINDOW`**（来自 #28161 的环境变量）：滑动窗口经过叶节点后提前释放 SWA 锁，更快释放 SWA 池容量。

这些作为一个栈交互：
1. 叶拆分 + 提前释放 → SWA 池利用率改善 → 热条目有更多空间。
2. 窗口感知 LRU → 过时 SWA 节点正确驱逐 → 热条目保持。
3. 后缀保留 → 被驱逐节点保留检查点桩位 → 更快重载。
4. L3 周期性检查点 → 存储占用降低 68–78% → L3 在规模上变得可负担。

每一层独立有用。合在一起，它们是让百万 token DeepSeek V4 服务在经济上工作的条件。

## 不止 SWA：L3 中的 C4 和 C128 压缩状态

SWA 是 L3 存储成本的最大贡献者，但它不是唯一与卸载交互的 DeepSeek V4 状态。DeepSeek V4 的稀疏注意力（DSA）产生两个额外的 per-request 状态缓冲区：

- **C4 状态**：稀疏索引器使用的压缩 per-token 注意力分数缓冲区。它追踪注意力模式中哪些块是"热"的，并在每个 decode 步骤驱动 top-k 选择。C4 位于类似滑动窗口的内存池中，通过 `full_to_swa_index_mapping` 索引。
- **C128 状态**：per-request 的压缩状态，以 128 token 为单位累积。它持有 DSA decode 路径所需的粗粒度索引器上下文。

两者最初都以 FP32 存储。在 HBM 受限的硬件上（H20 96 GB），这很昂贵。[#24041](https://github.com/sgl-project/sglang/pull/24041) 引入了 `SGLANG_DSV4_COMPRESS_STATE_DTYPE=bf16`，将 C4 和 C128 存储减半，精度影响可忽略（GSM8K 0.948，MMLU 0.896——与 FP32 基线匹配）。

对 L3 卸载来说，关键问题是：*这些状态能否与 KV 缓存一起卸载？*

答案取决于状态如何被索引。[#28612](https://github.com/sgl-project/sglang/pull/28612) 暴露了一个微妙的正确性 bug：online C128 + MTP 路径使用 `full_to_swa_index_mapping` 来推导 C128 状态槽位：

```python
full_loc = req_to_token[rid][chunk_start]
swa_loc = full_to_swa[full_loc]
main_slot = swa_loc / swa_page_size
```

这不安全，因为 radix 缓存可以在全注意力 KV 前缀仍然存活的同时，SWA 侧车——以及 `full_to_swa` 映射——已被 tombstone 或释放。在多轮缓存命中时，C128 路径可能读到槽 0、旧槽或重用的槽。

修复将 C128 状态与 SWA 映射完全解耦：

- **C128** 现在使用请求作用域布局：`req_pool_idx * ring_size + position % ring_size`。这使 C128 状态自包含于每个请求且可以独立序列化。
- **C4** 保留在基于 SWA 的滑动窗口路径上（其窗口语义与 SWA 的生命周期对齐）。

这种解耦是干净 L3 卸载 C128 状态的前提：因为 C128 现在是请求作用域而非 SWA 索引的，L3 存储层可以将其作为连续缓冲区与 MLA 缓存一起序列化，不依赖 SWA 存活性。

对于无前缀 ragged prefill 路径（[#25400](https://github.com/sgl-project/sglang/pull/25400)），系统跨 C0/C4/C128 层构建统一的 BF16 KV 和打包的 ragged 索引——证明所有三种状态类型可以共存于单一可序列化的元数据格式中。

## NSA 索引器与 L3：序列化前沿

DeepSeek V3.2 引入了原生稀疏注意力（NSA）——一种模型原生的稀疏模式，模型在训练时自行决定对哪些 token 做注意力。NSA 索引器维护一个 `index_k_with_scale_buffer`，记录逐层的索引器状态。

早期对 NSA 模型的 L3 卸载工作（[#18637](https://github.com/sgl-project/sglang/pull/18637)）为 DeepSeek V3.2 添加了 MoonCake Store 集成，包括索引器 key 的序列化。但出现了安全顾虑（[#20880](https://github.com/sgl-project/sglang/pull/20880)）：L3 存储层仅处理 MHA K/V 和 MLA latent-K 格式，不通用地序列化 NSA 索引器缓存。不正确处理时，L3 预取可能导致静默数据损坏和形状不匹配崩溃。

对于 DeepSeek V4，情况与 V3.2 不同：

- V4 使用 DSA（DeepSeek Sparse Attention）而非 V3.2 的 NSA。DSA 的索引器状态通过 C4 和 C128 压缩状态捕获，这些是具有已知形状的结构化缓冲区。
- 层级索引器工作（HISA，[#24672](https://github.com/sgl-project/sglang/pull/24672)）引入了 pool-K 缓存（`HisaNSATokenToKVPool`），带有逐层均值池化块——另一种需要 L3 序列化的结构化状态。
- L2（宿主机 DRAM）卸载不受影响——它复制原始 GPU 页面而无需格式感知。

V4 L3 的前进路径很清晰：C4 随 SWA 滑动（已由 SWA 检查点处理），C128 现在是请求作用域的（可独立序列化），MLA 潜在缓存已是 L3 的主要负载。剩余的空白是使用层级索引器的部署中的 HISA pool-K 缓存——这需要扩展 L3 存储格式以携带 pool-block 元数据。

## 投机解码：draft KV 池与 HiCache

DeepSeek V4 通过 NextN（内建 MTP 头）和 EAGLE 支持投机解码。两者都创建一个与目标模型 KV 缓存并行存在的 *draft KV 池*。

[#27805](https://github.com/sgl-project/sglang/pull/27805) 修复了一个空白：当投机解码与 HiCache 在 `UnifiedRadixCache` 路径上同时启用时，draft KV 池未注册到 `HostPoolGroup`。这导致 HiCache 备份/加载完全跳过 draft KV 数据——警告出现了但数据没保存。

修复引入了策略模式分发：

- `DraftSwaStrategy` 处理 `DeepSeekV4TokenToKVPool`——创建匹配目标 SWA 类型的宿主机池。
- `DraftPlainKvStrategy` 处理 MHA/MLA 池作为回退。
- `attach_draft_pool_to_unified_cache()` 将 draft 池注册为 `HostPoolGroup` 中的 `PoolEntry`，使其自动参与基于 PoolTransfer 的备份/加载和 L3 存储。

对多步 EAGLE，与 HiSparse 的交互更复杂（[#25409](https://github.com/sgl-project/sglang/pull/25409)）。投机解码为多个 draft/verify 步骤预分配逻辑 KV 窗口，但任何给定步骤中只有子集是活跃的。修复将投机逻辑分配与 HiSparse 设备映射分离：

- `prepare_for_decode()` 仅预留逻辑 KV 位置。
- `prepare_for_v2_draft()` 绑定当前步骤的 draft 槽位。
- `prepare_for_v2_verify()` 绑定当前窗口的 target-verify 槽位。

对 DeepSeek V4 具体来说，完整逻辑 KV 位置必须在更新设备映射前转换为压缩的 HiSparse 位置。过时或过宽的映射会导致后续步骤从错误槽位读取。

L3 含义：一旦 draft 池注册到 `HostPoolGroup`，HiCache 的周期性 SWA 检查点自然也覆盖 draft KV。恢复目标模型 KV 状态的缓存命中同时恢复 draft 模型的状态，使投机解码无需重新启动 draft 模型即可继续。

## Group Semantics：保持 L3 对象的一致性

当 HiCache 将一个逻辑页的 DeepSeek V4 KV 缓存卸载到 Mooncake Store 时，物理上是多个对象：MLA K 和 V 张量、SWA K 和 V（检查点时）、C4/C128 状态缓冲区、可能的索引器 sidecar 对象、以及 draft KV——还可能跨 TP rank 进一步拆分。一个 TP=2 且开启 SWA 检查点和 draft 池的部署，每个逻辑页轻松产生 8+ 个 Mooncake 对象。

Mooncake Store 以对象粒度管理生命周期。没有协调的情况下，其驱逐策略可以从一个组中移除部分对象而保留其他的。结果是部分组：有 K 没 V，或者有 rank-0 数据没有 rank-1。对 KV 缓存系统来说，部分组是无用的——对它的缓存命中无法产生正确的模型输出。

[Group Semantics RFC](https://github.com/kvcache-ai/Mooncake/issues/2127) 量化了这个问题。在真实 HiCache 流量下（Qwen3-8B, TP=2, `bench_mix.py` 负载），对象级驱逐产生驱逐碎片化的部分组：

| L3 容量 | 驱逐轮数 | 坏缓存存活字节比 |
|---------|---------|-----------------|
| 8 GB | 75 | 4.09% |
| 16 GB | 81 | 2.44% |
| 64 GB | 74 | 1.30% |
| 256 GB | 85 | 0.88% |

在 8 GB 时碎片化是实质性的。在生产规模（64 GB+）下降到 1.5% 以下，但任何部分组都是浪费的缓存未命中，触发完整重计算。

Mooncake Store 通过可选的分组对象路由解决此问题（[kvcache-ai/Mooncake#2180](https://github.com/kvcache-ai/Mooncake/pull/2180)，已合并）。核心机制：

- 对象通过 `ReplicateConfig.group_ids` 携带 `group_id`。
- 分组对象按 `hash(group_id)` 路由到同一个元数据分片。
- 当任何组成员被访问时，为*所有当前组成员*刷新租约。
- 当驱逐选择了分组候选者时，扩展为一起驱逐所有当前可安全驱逐的组成员。

SGLang 集成（[sgl-project/sglang#26574](https://github.com/sgl-project/sglang/pull/26574)）将逻辑 HiCache 页 key 映射为 Mooncake group id：

```text
sglang-hicache:{tagged_logical_page_key}
```

这意味着从同一个 radix tree 页派生的所有子对象——MHA/MLA K+V、SWA 检查点、索引器 sidecar、draft KV，跨所有 TP rank——共享一个组身份。启用标志：

```json
{
  "enable_group_semantics": true
}
```

实现保持向后兼容：当运行在没有 `ReplicateConfig.group_ids` 的老版本 Mooncake 包上时，SGLang 回退到现有的非分组 `batch_put_from(keys, ptrs, sizes)` 路径。非 Mooncake 后端无需配置变更。

对 DeepSeek V4 具体来说，group semantics 填补了一个重要空白。混合架构比纯 MHA 模型每个逻辑页产生更多子对象：MLA 潜在 KV、SWA KV 检查点、C4 状态（在 SWA 路径上）、C128 状态（请求作用域但绑定到同一前缀）、以及可选的 draft KV。没有 group semantics，任何一个片段的独立驱逐会使整个缓存前缀无用。有了它，完整的对象集合作为一个单元被驱逐或刷新，确保每次缓存命中返回完整可用的状态。

## 下一步

当前 L3 检查点实现在命中时回退到最近的检查点边界并对间隙重跑完整模型 prefill。下一步是**仅 SWA 部分 prefill**：缓存命中时，只对间隙 token 重跑 SWA 注意力层，MLA 层不动（它们的输出已在加载的缓存中）。

这将重计算代价按使用 SWA 的层占比（DeepSeek V4 架构中的前几层）降低。有了仅 SWA 部分 prefill，16K 检查点间隔下的 TTFT 增量将大幅下降，使更激进的间隔也变得实际可行。

之后还有几项扩展在推进中：

- **L3 中的 C128 状态**：有了请求作用域索引，C128 状态可以与 MLA 缓存一起序列化到 Mooncake Store。这使缓存命中时能完整恢复状态而无需重计算 DSA 索引器预热。
- **HISA pool-K 卸载**：对使用层级稀疏索引器的部署，均值池化 block 缓存需要 L3 序列化——对现有存储协议的格式扩展。
- **L3 中的 Draft KV**：draft 池注册到 `HostPoolGroup` 后，投机解码状态遵循与目标模型相同的卸载/重载路径。
- **跨节点检查点感知路由**：`swa_valid_from` 事件元数据使 conductor 可以将请求路由到同时持有 MLA 缓存和最近 SWA 检查点的实例，最小化重计算距离。

- **组完整性保证**：当前 group semantics 是尽力而为的（一起驱逐、一起刷新）。未来工作包括原子组可见性（组在缓存查找中要么完整存在要么完全不存在）、`group_size` 跟踪用于完整性检查、以及在选择驱逐候选者时考虑组成员关系的前缀感知驱逐策略。

组合图景：一百万 token 的 DeepSeek V4 缓存条目在 L3 中，存储约 4 GB（MLA + 稀疏 SWA 检查点），可选包含 C128 状态和 draft KV，跨 TP rank 组一致，通过仅 SWA 部分 prefill 在 4 秒内可重载，通过工作流亲和性可路由。这是目标状态。

如果你今天在用 HiCache 和 Mooncake Store 运行 DeepSeek V4，从 `--swa-checkpoint-interval 2048` 和 `SGLANG_SWA_CACHE_CHECKPOINT_MIN_TOKEN_INTERVAL=81920` 开始。观察你的 L3 利用率下降和 Agent 缓存命中率攀升。

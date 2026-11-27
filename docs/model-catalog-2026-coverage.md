# 2026-06-01 之后 KV cache calculator 覆盖与跳过清单

**盘点日：** 2026-11-18 UTC。**日期门槛：** 2026-06-01（含）之后。**upstream 基线：** [`85ad63fd97899a183fda37288441bcc45f69c9c5`](https://github.com/kvcache-ai/kvcache-blog/commit/85ad63fd97899a183fda37288441bcc45f69c9c5)。

本文只记录本轮新范围：模型在门槛日或之后正式发布，或官方权重在门槛日或之后公开，且 cache 结构可由官方配置/文档核实、可由项目 calculator 计算。`Hub createdAt` 和 `lastModified` 是 repo 时间，不能替代模型发布或权重公开日期。发布日期或权重日期未知时标为未知，不从 repo 创建时间推断。

## 决策口径

- 上游基线中已有的 canonical preset 不作为本轮新增；相关 after-cutoff 型号单列为“upstream 已支持”。DeepSeek V4 Pro (0813) 和 V4 Flash (0731) 仅收紧 KV precision 选项，去掉无对应普通 FP4 KV layout 的通用选项，保留 BF16、普通 FP8 和 packed FP8（[SGLang](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/arg_groups/overrides.py)、[vLLM](https://github.com/vllm-project/vllm/blob/main/vllm/models/deepseek_v4/attention.py)）。
- canonical preset 表按项目内部 `id` 核算；exact alias 表按模型选择器的官方 Hugging Face ID 核算。alias 的 `skip` 表示不额外保留一条独立映射；若该 exact ID 同时是上方保留 canonical 的官方 ID，则以 canonical 行为准，不重复计数。
- 2026-06-01 后新增且可算的条目共 14 个 canonical preset 和 4 个独立 official ID alias。决策输入中另有 14 个 canonical 候选因日期不合格或日期证据不足而跳过、122 个 alias 不作为独立新增；另有 9 个 after-cutoff canonical 已存在于 upstream。
- 主表 18/18 official HF repo 元数据公开可读；14/14 新增 canonical 的固定 revision `config.json` 均经 HTTP HEAD 验证为 200。本轮未下载完整模型权重。
- 多模态条目只报告其表中明确限定的 decoder/cache 组件。视觉/audio encoder、扩散、流匹配、VAE、planner 等 state 不会被当成 decoder KV；需要额外解释的范围见每行说明。

## 新增且已由 calculator 计算

每行的大小是 1024 tokens、1 sequence、TP=1 的单序列总 cache。KV、indexer 和 recurrent-state precision 使用页面默认值；Draft KV 关闭；线性 attention/SConv state 开关按对应模型配置的页面默认值；checkpoint policy 为页面默认的 prompt-end。数值表示逻辑 cache payload，不包括 allocator page padding 或 page metadata。`GB = bytes / 10^9`，`GiB = bytes / 2^30`。表中结果保留页面的 5 位小数格式；下方三条精度示例也列出精确字节数。

| 状态 | 精确 official model ID | Preset / 页面选择 ID | 日期证据 | Formula / cache 范围 | 默认精度与状态 | 1024 tokens × 1 sequence |
|---|---|---|---|---|---|---:|
| 新增并已计算 | [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) | `deepseek-v4.1-flash-1017` / `deepseek-v4.1-flash-1017` | release 2026-10-17; weights 2026-10-17<br>[发布公告/官方说明](https://api-docs.deepseek.com/news/news261017/)<br>[权重公开证据](https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4.1-Flash-1017/commits/main) | DeepSeek V4.1 CED/CSA2 shared cache<br>[pinned 官方 config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/raw/08c8657f02291e11b6f6b63e0e16014f8e748518/config.json)<br>官方说明此日发布并公开HF权重；V4.1 shared-cache decoder字段可按当前preset计算。9/10发布的是不同模型ID DeepSeek-V4.1-Flash。 | KV FP4 (E2M1); indexer MXFP4; optional sequence state off | **0.00091 GB / 0.00085 GiB** |
| 新增并已计算 | [`moonshotai/Kimi-K3.1-Pro`](https://huggingface.co/moonshotai/Kimi-K3.1-Pro) | `kimi-k3.1-pro` / `kimi-k3.1-pro` | release 2026-10-23; weights 2026-11-02<br>[发布公告/官方说明](https://forum.moonshot.ai/t/kimi-k3-1-pro-has-launched-start-exploring-now/629)<br>[权重公开证据](https://huggingface.co/api/models/moonshotai/Kimi-K3.1-Pro/commits/main) | Kimi KDA/DSA/MLA with indexer hybrid<br>[pinned 官方 config](https://huggingface.co/moonshotai/Kimi-K3.1-Pro/raw/d9e99b0792a5e93b78ea9bea62a68167e36ddd9e/config.json)<br>Moonshot官方公告称其为open-weight并于10/23发布；HF首个权重repo提交为11/2。默认MLA KV为SGLang packed FP8，656 B/entry（512 latent + 16 scale + 128 BF16 RoPE）；indexer FP8每128维使用4 B scale。KDA convolution/recurrent state计入，recurrent state为FP32；大小为逻辑payload，不含page padding/metadata。 | 默认 KV FP8 SGLang packed（656 B/MLA entry）; indexer FP8（128维/4 B scale）; KDA state included (FP32 recurrent) | **468,738,048 B; 0.46874 GB / 0.43655 GiB** |
| 新增并已计算 | [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | `qwen3.8-flash-next` / `qwen3.8-flash-next` | release 2026-08-26; weights 2026-08-26<br>[发布公告/官方说明](https://github.com/QwenLM/Qwen3.8-Flash-Next)<br>[权重公开证据](https://huggingface.co/api/models/Qwen/Qwen3.8-Flash-Next/commits/main) | Qwen GDN/QSA hybrid with PLE<br>[pinned 官方 config](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/812d96e10a0dd82b69d749e68693698cf6c36daa/config.json)<br>Qwen官方README列出8/26发布并提供HF权重；GDN/QSA/PLE hybrid cache有当前calculator公式。 | KV BF16 / FP16; indexer BF16 / FP16; recurrent FP32; linear state included | **0.14161 GB / 0.13188 GiB** |
| 新增并已计算 | [`Qwen/Qwen-Image-2.1-PE-T2I`](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-T2I) | `qwen-image-2.1-pe-decoder` / `qwen-image-2.1-pe-decoder` | release 2026-09-20; weights 2026-09-20<br>[发布公告/官方说明](https://github.com/QwenLM/Qwen-Image-2.1)<br>[权重公开证据](https://huggingface.co/api/models/Qwen/Qwen-Image-2.1-PE-T2I/commits/main) | Qwen linear/full hybrid<br>[pinned 官方 config](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-T2I/raw/f3ed7985c788ad75b3ab7223e0c4c51e2a43545b/config.json)<br>Qwen官方repo记载9/20发布并列出T2I/I2I PE权重；这是可逐token生成文本的prompt rewriter，估算其Qwen3.5-VL decoder KV，排除图像扩散部分。 | KV BF16 / FP16; recurrent FP32; linear state included | **0.08507 GB / 0.07922 GiB** |
| 新增并已计算 | [`CohereLabs/North-Mini-Code-1.0`](https://huggingface.co/CohereLabs/North-Mini-Code-1.0) | `cohere-north-mini-code-1.0` / `cohere-north-mini-code-1.0` | release 2026-06-09; weights 2026-06-05<br>[发布公告/官方说明](https://docs.cohere.com/changelog/north-mini-code-1-0)<br>[权重公开证据](https://huggingface.co/api/models/CohereLabs/North-Mini-Code-1.0/commits/main) | Mixed full/sliding GQA<br>[pinned 官方 config](https://huggingface.co/CohereLabs/North-Mini-Code-1.0/raw/d11e61a842617a22dc328552fa5bb86231ee4f37/config.json)<br>Cohere官方公告6/9发布并开放HF权重；Hybrid full/sliding GQA decoder可估。FP8/W4A16属于同canonical的量化weight alias，不另列新模型。 | KV BF16 / FP16; optional sequence state off | **0.10276 GB / 0.09570 GiB** |
| 新增并已计算 | [`CohereLabs/North-Micro-Vision-Instruct`](https://huggingface.co/CohereLabs/North-Micro-Vision-Instruct) | `cohere-north-micro-vision-instruct` / `cohere-north-micro-vision-instruct` | release 2026-08-12; weights 2026-08-12<br>[发布公告/官方说明](https://huggingface.co/blog/CohereLabs/meet-north-micro-vision-instruct) | Mixed full/sliding GQA<br>[pinned 官方 config](https://huggingface.co/CohereLabs/North-Micro-Vision-Instruct/raw/fa548d9aa62c1220bfd21fc6cfc465f1bd2a1a40/config.json)<br>Cohere共同作者官方HF文章写明8/12发布并链接权重；只估混合注意力的LLM text decoder，vision encoder状态排除。 | KV BF16 / FP16; optional sequence state off | **0.11744 GB / 0.10938 GiB** |
| 新增并已计算 | [`google/gemma-4-12B`](https://huggingface.co/google/gemma-4-12B) | `gemma-4-12b` / `gemma-4-12b` | release 2026-06-03; weights 2026-06-03<br>[发布公告/官方说明](https://blog.google/innovation-and-ai/technology/developers-tools/introducing-gemma-4-12B/)<br>[权重公开证据](https://huggingface.co/api/models/google/gemma-4-12B/commits/main) | Mixed full/sliding GQA<br>[pinned 官方 config](https://huggingface.co/google/gemma-4-12B/raw/1a2c7729392586f5ecddae8807ee2ef741eabb9a/config.json)<br>Google官方公告6/3发布Gemma 4 12B并开放模型；decoder的mixed full/sliding GQA可估。E2B/E4B/26B/31B是4月发布的独立尺寸，不属于本轮新加项。 | KV BF16 / FP16; optional sequence state off | **0.35232 GB / 0.32813 GiB** |
| 新增并已计算 | [`thinkingmachines/Inkling-2-Small`](https://huggingface.co/thinkingmachines/Inkling-2-Small) | `inkling-2-small` / `inkling-2-small` | release 2026-11-05; weights 2026-11-12<br>[发布公告/官方说明](https://thinkingmachines.ai/news/inkling-2-small/)<br>[权重公开证据](https://huggingface.co/api/models/thinkingmachines/Inkling-2-Small/commits/main) | Inkling global/SWA with SConv<br>[pinned 官方 config](https://huggingface.co/thinkingmachines/Inkling-2-Small/raw/1df2d97d45210d94f781f42a6521392e5a539e46/config.json)<br>Thinking Machines 11/5公告，说明开放权重将在安全测试后发布；11/12官方更新称HF权重可下载，API首个权重提交也是11/12。Hybrid attention与SConv state可单独估。 | KV BF16 / FP16; SConv included | **0.20781 GB / 0.19354 GiB** |
| 新增并已计算 | [`XiaomiMiMo/MiMo-V2.6-Flash-RL`](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL) | `mimo-v2.6-flash` / `mimo-v2.6-flash` | release 2026-09-22; weights 2026-09-22<br>[发布公告/官方说明](https://mimo.mi.com/docs/updates/model)<br>[权重公开证据](https://mimo.mi.com/docs/en-US/news/latest/v2-6) | Mixed full/sliding GQA<br>[pinned 官方 config](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL/raw/f657ba2f7d24c9adc5730675baaa2d34de878a49/config.json)<br>Xiaomi官方模型release记录9/22发布并开源V2.6 Flash/Pro；mixed global/sliding GQA decoder可估。MOPD是同模型的不同recipe/weight repo，不重复作为新模型。 | KV BF16 / FP16; optional sequence state off | **0.04915 GB / 0.04578 GiB** |
| 新增并已计算 | [`XiaomiMiMo/MiMo-V2.6-Pro-RL`](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL) | `mimo-v2.6-pro` / `mimo-v2.6-pro` | release 2026-09-22; weights 2026-09-22<br>[发布公告/官方说明](https://mimo.mi.com/docs/updates/model)<br>[权重公开证据](https://mimo.mi.com/docs/en-US/news/latest/v2-6) | Mixed full/sliding GQA<br>[pinned 官方 config](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL/raw/a066ba0057192126bd5d255f37138265fc1a435e/config.json)<br>Xiaomi官方模型release记录9/22发布并开源V2.6 Pro；mixed global/sliding GQA decoder可估。MOPD是同模型的不同recipe/weight repo，不重复作为新模型。 | KV BF16 / FP16; optional sequence state off | **0.09175 GB / 0.08545 GiB** |
| 新增并已计算 | [`XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B`](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B) | `mimo-v2.6-distill-qwen-9b` / `mimo-v2.6-distill-qwen-9b` | release 2026-09-22; weights 2026-09-21<br>[发布公告/官方说明](https://mimo.mi.com/docs/en-US/news/latest/v2-6)<br>[权重公开证据](https://huggingface.co/api/models/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B/commits/main) | Qwen linear/full hybrid<br>[pinned 官方 config](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B/raw/1f09ddcc1a0ea45751ae0dfe2af20d533168c46f/config.json)<br>V2.6官方开源文章说同时发布Distill-Qwen-9B；HF首个权重提交9/21。它是独立distilled checkpoint，cache几何复用Qwen3.5-9B linear/full profile。 | KV BF16 / FP16; recurrent FP32; linear state included | **0.08507 GB / 0.07922 GiB** |
| 新增并已计算 | [`XiaomiMiMo/MiMo-V2.5-ASR`](https://huggingface.co/XiaomiMiMo/MiMo-V2.5-ASR) | `mimo-v2.5-asr-text-decoder` / `mimo-v2.5-asr-text-decoder` | release 2026-06-02; weights 2026-04-23<br>[发布公告/官方说明](https://mimo.mi.com/docs/updates/model)<br>[权重公开证据](https://huggingface.co/api/models/XiaomiMiMo/MiMo-V2.5-ASR/commits/main) | Standard MHA/GQA<br>[pinned 官方 config](https://huggingface.co/XiaomiMiMo/MiMo-V2.5-ASR/raw/227ce4c5e64e356007ad948781f0e8b55a1beba0/config.json)<br>Xiaomi官方release表将MiMo-V2.5-ASR列为6/2发布，因此满足模型发布日期门槛；注意HF权重实际已于4/23公开，未把该日期冒充6月权重发布。仅估启用cache的global Qwen2 text decoder，其他音频组件排除。 | KV BF16 / FP16; optional sequence state off | **0.15099 GB / 0.14063 GiB** |
| 新增并已计算 | [`MiniMaxAI/MiniMax-M3.1-Flash`](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash) | `minimax-m3.1-flash` / `minimax-m3.1-flash` | release 2026-10-19; weights 2026-10-23<br>[发布公告/官方说明](https://www.minimax.io/blog/minimax-m31f)<br>[权重公开证据](https://huggingface.co/api/models/MiniMaxAI/MiniMax-M3.1-Flash/commits/main) | MiniMax MSA sparse attention<br>[pinned 官方 config](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/raw/ce7f4df87d2e05262f5775f373dac4772f134041/config.json)<br>MiniMax 10/19公告明确当时模型先在API上线、权重稍后发布；HF首个含48个主权重shard与DSpark draft shard的提交是10/23。KV与key-only indexer可分别选NVFP4；每128维含64 B数据与8 B scale（16维/group、每group 1 B），双NVFP4配置为39,813,120 B（0.03981 GB / 0.03708 GiB）。实际backend支持的cache格式可能与checkpoint权重精度不同；官方MXFP8/NVFP4量化元数据描述的是权重，不是KV。图像/视频encoder state与MTP draft cache、page padding及metadata不计。 | 默认 KV BF16 / FP16; indexer BF16 / FP16; optional KV/indexer NVFP4 independently; sequence state off | **0.14156 GB / 0.13184 GiB** |
| 新增并已计算 | [`MiniMaxAI/MiniMax-H3-Image-PE`](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image-PE) | `minimax-h3-image-pe` / `minimax-h3-image-pe` | release 未确认; weights 2026-11-10<br>[发布公告/官方说明](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image-PE)<br>[权重公开证据](https://huggingface.co/api/models/MiniMaxAI/MiniMax-H3-Image-PE/commits/main) | Qwen linear/full hybrid<br>[pinned 官方 config](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image-PE/raw/35603d4b63949afa12cb098227fffefedf081afa/config.json)<br>没有找到PE子模型独立发布公告日；官方MiniMax HF API于11/10初始commit已经含4个safetensors权重shard，满足公开权重门槛。该PE生成文本；估其Qwen3.5 text decoder KV，H3扩散pipeline不计。 | KV BF16 / FP16; recurrent FP32; linear state included | **0.22105 GB / 0.20587 GiB** |
| 新增并已计算 | [`Qwen/Qwen-AgentWorld-35B-A3B`](https://huggingface.co/Qwen/Qwen-AgentWorld-35B-A3B) | `qwen3.5-35b-a3b` / `Qwen/Qwen-AgentWorld-35B-A3B` | 官方日期证据 2026-06-24<br>[发布公告/官方说明](https://github.com/QwenLM/Qwen-AgentWorld) | Qwen linear/full hybrid<br>[pinned 官方 config](https://huggingface.co/Qwen/Qwen-AgentWorld-35B-A3B/raw/60d2b0434a53d2e62a7c00a489586815d94ebffb/config.json)<br>独立after-cutoff finetuned world model；HF权重6/23上传，复用现有qwen3.5-35b-a3b decoder几何。 | KV BF16 / FP16; recurrent FP32; optional sequence state off | **0.02097 GB / 0.01953 GiB** |
| 新增并已计算 | [`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp) | `deepseek-v4-flash` / `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` | 官方日期证据 2026-08-21<br>[发布公告/官方说明](https://api-docs.deepseek.com/news/news260821/) | DeepSeek V4 hybrid sparse attention<br>[pinned 官方 config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp/raw/6821d6ad3681a4b137b066b76094fa82ebd0a380/config.json)<br>官方8/21 Vision-Exp型号，独立于base/DSpark的视觉任务；只估decoder cache，视觉encoder state不计。默认uniform FP8 KV + MXFP4 indexer；indexer每128维含64 B FP4数据与4 B scale（128/2 + 4 = 68 B）。可选SGLang packed KV为584 B/entry；两种估算均不含allocator page padding或page metadata。`tokens`应包含decoder实际处理的所有位置，包括图像位置。 | 默认 KV uniform FP8; indexer MXFP4（128维/68 B，含scale）; optional SGLang packed FP8 KV（584 B/entry） | **6,018,048 B; 0.00602 GB / 0.00560 GiB**；可选packed KV为**6,812,928 B** |
| 新增并已计算 | [`Qwen/Qwen-Image-2.1-PE-I2I`](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-I2I) | `qwen-image-2.1-pe-decoder` / `Qwen/Qwen-Image-2.1-PE-I2I` | 官方日期证据 2026-09-20<br>[发布公告/官方说明](https://github.com/QwenLM/Qwen-Image-2.1) | Qwen linear/full hybrid<br>[pinned 官方 config](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-I2I/raw/72927bc08afc99b7888ceb7d7d51a12db3700bbd/config.json)<br>独立I2I prompt-rewriter任务与canonical T2I不同；9/20官方发布且有权重，复用同一Qwen3.5-VL decoder几何，扩散部分排除。 | KV BF16 / FP16; recurrent FP32; linear state included | **0.08507 GB / 0.07922 GiB** |
| 新增并已计算 | [`Qwen/Qwen-Drive-1.0-4B`](https://huggingface.co/Qwen/Qwen-Drive-1.0-4B) | `qwen3.5-4b` / `Qwen/Qwen-Drive-1.0-4B` | 官方日期证据 2026-08-28<br>[发布公告/官方说明](https://github.com/QwenLM/Qwen-Drive-1.0) | Qwen linear/full hybrid<br>[pinned 官方 config](https://huggingface.co/Qwen/Qwen-Drive-1.0-4B/raw/28484089a7cc8c335cf5089fb0745cf7c49b6eaa/config.json)<br>HF API checkpoint-bearing上传为8/28；官方README明确Qwen3.5-4B LLM decoder不变，外接BEV/planning head排除，复用qwen3.5-4b。 | KV BF16 / FP16; recurrent FP32; optional sequence state off | **0.03355 GB / 0.03125 GiB** |

> 多模态范围：上表的 North Micro、Qwen Image-2.1-PE、DeepSeek Vision-Exp、Qwen AgentWorld、Qwen-Drive 与 MiniMax H3-Image-PE 都按表内官方配置/`cache_note` 只算文本 decoder cache；视觉 encoder、扩散与规划 state 不计。MiMo V2.5 ASR 只算可复用的 global text decoder。DeepSeek Vision-Exp 的 `tokens` 应包含 decoder 实际处理的所有位置，包括图像位置。

> 默认值与可选精度：三条上文数值均按各模型页面默认的KV/indexer precision计算。MiniMax的双NVFP4结果与DeepSeek Vision-Exp的packed KV结果是可选组合，不替代主表默认值；量化权重精度不代表KV-cache backend必然支持相同布局。所有数字均不包含多模态encoder state、allocator page padding或page metadata。

## Upstream 已支持，不重复新增

下列 after-cutoff canonical 已在基线 `85ad63f` 中，因此本轮状态是“upstream 已支持”，不是本轮新增或跳过计算。它们的 exact official ID、发布日期和来源保留在表中，避免把基线已有误写成本轮覆盖。

| 状态 | Official model ID | Upstream preset | 官方日期 | 官方证据 | 处理原因 |
|---|---|---|---|---|---|
| Upstream 已支持 | [`Qwen/Qwen3.8-2.4T-A95B`](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B) | `qwen3.8-2.4t-a95b` | release 2026-08-12 | [官方发布日期/模型说明](https://github.com/QwenLM/Qwen3.8) | 符合日期且可计算，但该canonical已在本地HEAD中，属已支持基线，不重复新增。 |
| Upstream 已支持 | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) | `qwen3.8-27b` | release 2026-08-14 | [官方发布日期/模型说明](https://github.com/QwenLM/Qwen3.8) | 符合日期且可计算，但该canonical已在本地HEAD中，属已支持基线，不重复新增。 |
| Upstream 已支持 | [`dots-studio/dots3-note-prev`](https://huggingface.co/dots-studio/dots3-note-prev) | `dots3-note-prev` | weights 2026-08-14 | [官方发布日期/模型说明](https://huggingface.co/api/models/dots-studio/dots3-note-prev/commits/main) | 首个包含safetensors的HF API提交为8/14，符合日期并已有hybrid DSA/SWA profile；该canonical已在本地HEAD中，不重复新增。 |
| Upstream 已支持 | [`zai-org/GLM-5.3`](https://huggingface.co/zai-org/GLM-5.3) | `glm-5.3` | release 2026-08-14 | [官方发布日期/模型说明](https://z.ai/blog/glm-5.3) | after-cutoff型号，当前canonical已存在；保持已有profile，不重复新增。 |
| Upstream 已支持 | [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) | `glm-5.3-flash` | release 2026-08-26 | [官方发布日期/模型说明](https://x.com/Zai_org/status/2092616204787626030) | after-cutoff型号，当前canonical已存在；保持已有profile，不重复新增。 |
| Upstream 已支持 | [`thinkingmachines/Inkling-Small`](https://huggingface.co/thinkingmachines/Inkling-Small) | `inkling-small` | release 2026-07-30 | [官方发布日期/模型说明](https://thinkingmachines.ai/news/inkling-small/) | after-cutoff可计算模型，但旧canonical已在本地HEAD并已被Inkling 2 Small取代；不重复新增。 |
| Upstream 已支持 | [`moonshotai/Kimi-K2.7-Code`](https://huggingface.co/moonshotai/Kimi-K2.7-Code) | `kimi-k2.7-code` | release 2026-06-16 | [官方发布日期/模型说明](https://forum.moonshot.ai/t/here-comes-kimi-k2-7-code-better-coding-with-more-efficiency/441) | 符合日期且权重链接存在，但kimi-k2.7-code canonical已在本地HEAD；不重复添加。 |
| Upstream 已支持 | [`moonshotai/Kimi-K3`](https://huggingface.co/moonshotai/Kimi-K3) | `kimi-k3` | release 2026-07-22 | [官方发布日期/模型说明](https://forum.moonshot.ai/t/kimi-k3-is-here-our-most-capable-model/480) | 符合日期且本地已有kimi-k3 canonical；不重复添加。 |
| Upstream 已支持 | [`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | `deepseek-v4.1-flash` | release 2026-09-10 | [官方发布日期/模型说明](https://api-docs.deepseek.com/news/news260910/) | 9/10版本与1017是不同模型ID，但本地HEAD已有deepseek-v4.1-flash canonical；不重复新增。 |

## Skip：未新增或未计算的窗口候选

本节逐项保留 after-cutoff/upstream、restricted、日期未确认及 cache 结构未核实的候选。盘点只覆盖筛选输入和下表列出的补充候选，不声称穷尽所有官方 Hub repo。

### Upstream 已支持

上方“Upstream 已支持，不重复新增”表逐项列出 9 个已确认在基线中的 after-cutoff canonical。它们继续可用，但不作为本轮新增结果重复计算。

### 日期证据不足，无法确认进入窗口

| Skip 候选 official ID | Canonical preset | 发布/权重日期 | 官方证据 | 跳过原因 |
|---|---|---|---|---|
| [`zai-org/GLM-4.7-Flash`](https://huggingface.co/zai-org/GLM-4.7-Flash) | `glm-4.7-flash` | 未确认 | [模型/官方说明](https://huggingface.co/zai-org/GLM-4.7-Flash)<br>[HF 提交记录](https://huggingface.co/api/models/zai-org/GLM-4.7-Flash/commits/main) | 未找到发布方可核验的官方发布日期；HF API最初repo提交在1/19但该日期不是权重公开日期。无6/1后权重提交证据，按日期不明跳过。 |
| [`zai-org/GLM-OCR`](https://huggingface.co/zai-org/GLM-OCR) | `glm-ocr-text-decoder` | 未确认 | [模型/官方说明](https://github.com/zai-org/GLM-OCR)<br>[HF 提交记录](https://huggingface.co/api/models/zai-org/GLM-OCR/commits/main) | 当前官方repo只核实到1/30的初始配置提交及3/12的技术报告更新；未找到精确发布日期或6/1后权重公开时间。虽然可估OCR text decoder GQA KV，日期未核实，跳过。 |
| [`Qwen/WebWorld-14B`](https://huggingface.co/Qwen/WebWorld-14B) | `webworld-14b` | 未确认 | [模型/官方说明](https://github.com/QwenLM/WebWorld)<br>[HF 提交记录](https://huggingface.co/api/models/Qwen/WebWorld-14B/commits/main) | 官方repo确认Qwen3-14B causal-LM finetune；未找到发布方日期公告或精确首个权重提交日，现有官方材料指向2月发布而非6/1后；按日期未核实跳过。官方config设use_cache=false，需服务端显式启用cache。 |
| [`Qwen/WebWorld-32B`](https://huggingface.co/Qwen/WebWorld-32B) | `webworld-32b` | 未确认 | [模型/官方说明](https://github.com/QwenLM/WebWorld)<br>[HF 提交记录](https://huggingface.co/api/models/Qwen/WebWorld-32B/commits/main) | 官方repo确认Qwen3-32B causal-LM finetune；未找到发布方日期公告或精确首个权重提交日，现有官方材料指向2月发布而非6/1后；按日期未核实跳过。官方config设use_cache=false，需服务端显式启用cache。 |

### 窗口前的候选仅作数量汇总

筛选输入中另有 10 个已证实在 2026-06-01 前发布/公开权重的 canonical 候选，本轮按日期门槛跳过，不在这份只看 6 月以后范围的主体表逐项展开：DeepSeek 3 个（V4 Pro/Flash preview、OCR-2）、GLM 1 个（GLM-Image）、Qwen 4 个（Coder-Next、WebWorld-8B、ASR、TTS）、Dots3 2 个（mocr、mocr-svg）。另有 Llama 4（2025）、旧版 Cohere Command（2024/2025）和 Cohere Transcribe 03-2026，均属窗口前项目；其 exact IDs 不列作本轮窗口候选。

筛选输入中的 122 个 alias `skip` 也不等于 122 个模型跳过：它们是已有 canonical 的精度/量化或 repo 格式变体、canonical official ID 的重复映射，或没有独立 after-cutoff 发布证据的 alias。本轮不把它们逐个列为模型结果；筛选后 catalog 只保留 4 个独立 after-cutoff alias，另 14 个 canonical 的官方 ID 由 canonical 结果表承接。

### 日期在窗口内，但配置受限或 cache profile 未核实

| Official model ID / exact repo IDs | 日期与官方来源 | 状态和跳过原因 |
|---|---|---|
| [`CohereLabs/North-Small-Translate-1.0`](https://huggingface.co/CohereLabs/North-Small-Translate-1.0)；[`CohereLabs/North-Small-Translate-1.0-fp8`](https://huggingface.co/CohereLabs/North-Small-Translate-1.0-fp8)；[`CohereLabs/North-Small-Translate-1.0-w4a16`](https://huggingface.co/CohereLabs/North-Small-Translate-1.0-w4a16) | Cohere 官方 changelog 记载 2026-09-09 发布，提供 BF16/FP8/W4A16 权重；[model card](https://huggingface.co/CohereLabs/North-Small-Translate-1.0) 描述 decoder-only MoE。 | 三个 repo 的 `config.json` 无凭据 HEAD 均返回 HTTP 401；无法核对精确层数、head/window 字段，不新增。精度变体没有独立型号身份。 |
| [`CohereLabs/cohere-transcribe-arabic-07-2026`](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026) | Cohere 官方文章记载 2026-07-07 发布权重；[model card](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026) 说明 Conformer encoder-decoder 和 Transformer token decoder。 | 官方 config 无凭据 HEAD 返回 HTTP 401；项目也没有该 encoder-decoder 的 decoder/cross-attention cache profile，本轮跳过。 |
| [`dots-studio/dots.tts-base`](https://huggingface.co/dots-studio/dots.tts-base)、[`dots-studio/dots.tts-soar`](https://huggingface.co/dots-studio/dots.tts-soar)、[`dots-studio/dots.tts-mf`](https://huggingface.co/dots-studio/dots.tts-mf) | Dots 官方 [repo README](https://github.com/studio-dots-ai/dots.tts) 标为 2026-06 发布（仅精确到月）。 | config 可读取；官方结构是 continuous autoregressive TTS，由 semantic encoder、LLM、acoustic flow-matching head 和 AudioVAE 组成。没有经核实的 text-token KV profile，跳过。 |
| [`dots-studio/dots.tts-mf-2steps`](https://huggingface.co/dots-studio/dots.tts-mf-2steps)、[`dots-studio/dots.tts-mf-1step`](https://huggingface.co/dots-studio/dots.tts-mf-1step)、[`dots-studio/dots.tts-mf-2steps-stts`](https://huggingface.co/dots-studio/dots.tts-mf-2steps-stts)、[`dots-studio/dots.tts.edit`](https://huggingface.co/dots-studio/dots.tts.edit) | Dots 官方 README 标为 2026-08 发布。 | 同系列采样/streaming checkpoint 或 audio-edit 任务；共享 backbone 但未核实可供本 calculator 使用的 token KV geometry，不新增独立 profile。 |
| [`MiniMaxAI/MiniMax-Music3`](https://huggingface.co/MiniMaxAI/MiniMax-Music3) | HF 官方 revision [`1ead5ad`](https://huggingface.co/MiniMaxAI/MiniMax-Music3/tree/1ead5ad6708943c855aa166979d5cff8e6b5f893) 于 2026-08-07 首次包含 checkpoint 权重。 | 官方卡说明 Global LLM 8B + Local LLM 0.6B、RVQ、Flow Matching/Flow-VAE；两个 LLM 的 `use_cache` 路径、token 轴和组合 cache 未核实，不能套单一 profile，跳过。 |

以上补充候选不是完整 Hub census。窗口前项目只作范围说明；公开日期精确到月或仍未知时，文档保留原有精度，不推导具体日期。


## 1024-token 结果复算命令

本表的输出由项目 `calculate()` 对 apply 后的 YAML/JSON 调用得到。主表循环复用页面默认设置：1024 tokens、1 sequence、TP=1；precision 不传 override，由模型默认字段选择；linear/SConv/SWA state 按各 preset 的 `default_include_*` 字段；Draft KV 关闭；checkpoint 为 prompt-end。脚本另用两个显式 override 复算Vision-Exp packed KV与MiniMax双NVFP4可选组合。`yaml` 是 Node 可解析 YAML 的模块；此控制节点的 Node 18 环境中可直接 `require('yaml')`。选择 canonical 时传项目 `id`，选择新增 alias 时传 exact official ID。

```js
const fs = require('node:fs');
const YAML = require('yaml');
const calculator = require('./assets/js/kv-cache-calculator.js');
const data = YAML.parse(fs.readFileSync('data/kv_cache_calculator/models.yaml', 'utf8'));
const catalog = JSON.parse(fs.readFileSync('data/kv_cache_calculator/model_catalog.json', 'utf8'));
const models = calculator.expandModelCatalog(data.models, catalog);
const ids = [
  'deepseek-v4.1-flash-1017', 'kimi-k3.1-pro', 'qwen3.8-flash-next',
  'qwen-image-2.1-pe-decoder', 'cohere-north-mini-code-1.0',
  'cohere-north-micro-vision-instruct', 'gemma-4-12b', 'inkling-2-small',
  'mimo-v2.6-flash', 'mimo-v2.6-pro', 'mimo-v2.6-distill-qwen-9b',
  'mimo-v2.5-asr-text-decoder', 'minimax-m3.1-flash', 'minimax-h3-image-pe',
  'Qwen/Qwen-AgentWorld-35B-A3B', 'deepseek-ai/DeepSeek-V4-Flash-Vision-Exp',
  'Qwen/Qwen-Image-2.1-PE-I2I', 'Qwen/Qwen-Drive-1.0-4B',
];
const options = {
  precisionOptions: data.precision_options,
  indexerPrecisionOptions: data.indexer_precision_options,
  recurrentStatePrecisionOptions: data.recurrent_state_precision_options,
};
for (const id of ids) {
  const model = models.find((item) => item.id === id);
  const input = {
    tokens: 1024, sequences: 1, tensorParallel: 1,
    includeDraftKvCache: false,
    includeSwaCache: Boolean(model.fields?.default_include_swa_cache),
    includeLinearAttentionState: Boolean(model.fields?.default_include_linear_attention_state),
    includeSconvState: Boolean(model.fields?.default_include_sconv_state),
    qwenCheckpointPolicy: 'prompt_end',
    kdaCheckpointPolicy: 'prompt_end',
    sconvCheckpointPolicy: 'prompt_end',
  };
  const result = calculator.calculate(model, input, options);
  const kvDefault = model.fields?.kv_fixed_precision_id || model.fields?.default_precision_id || '(calculator fallback)';
  const indexerDefault = model.fields?.indexer_fixed_precision_id || model.fields?.default_indexer_precision_id || '(calculator fallback)';
  console.log(`${id}\t${result.totalBytes} B\t${result.totalGB.toFixed(5)} GB\t${result.totalGiB.toFixed(5)} GiB\tKV=${kvDefault}\tIndexer=${indexerDefault}`);
}
for (const [id, overrides] of [
  ['deepseek-ai/DeepSeek-V4-Flash-Vision-Exp', { precision: 'fp8_sglang_packed', indexerPrecision: 'fp4_int4_g32_s1' }],
  ['minimax-m3.1-flash', { precision: 'nvfp4_g16_s1', indexerPrecision: 'nvfp4_g16_s1' }],
]) {
  const model = models.find((item) => item.id === id);
  const result = calculator.calculate(model, {
    tokens: 1024, sequences: 1, tensorParallel: 1,
    includeDraftKvCache: false,
    includeSwaCache: Boolean(model.fields?.default_include_swa_cache),
    includeLinearAttentionState: Boolean(model.fields?.default_include_linear_attention_state),
    includeSconvState: Boolean(model.fields?.default_include_sconv_state),
    qwenCheckpointPolicy: 'prompt_end',
    kdaCheckpointPolicy: 'prompt_end',
    sconvCheckpointPolicy: 'prompt_end',
    ...overrides,
  }, options);
  console.log(`${id} (optional)\t${result.totalBytes} B\t${result.totalGB.toFixed(5)} GB\t${result.totalGiB.toFixed(5)} GiB`);
}
```

项目依据：[`calculator page controls`](../layouts/shortcodes/kv-cache-calculator.html)、[`calculator defaults and formulas`](../assets/js/kv-cache-calculator.js)、[`canonical profiles`](../data/kv_cache_calculator/models.yaml)、[`official ID aliases`](../data/kv_cache_calculator/model_catalog.json)。

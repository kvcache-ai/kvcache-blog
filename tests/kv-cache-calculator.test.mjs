import assert from "node:assert/strict";
import { createRequire } from "node:module";
import test from "node:test";

const require = createRequire(import.meta.url);
const { calculate, calculateElementsPerSequence, formatBytes, modelFamily, modelsForFamily } = require("../assets/js/kv-cache-calculator.js");

const bf16 = { precision: "bf16_fp16", indexerPrecision: "bf16_fp16", sequences: 1, tensorParallel: 1 };

test("standard GQA formula matches Qwen3-32B at 128k tokens", () => {
  const model = {
    id: "qwen3-32b",
    label: "Qwen3-32B",
    formula: "standard_gqa",
    fields: { num_hidden_layers: 64, num_key_value_heads: 8, head_dim: 128 },
  };

  const result = calculate(model, { ...bf16, tokens: 128000 });
  assert.equal(result.elementPlan.elementsPerToken, 131072);
  assert.match(result.elementPlan.formulaText, /tokens \* sequences/);
  assert.match(result.elementPlan.formulaText, /precision_bytes/);
  assert.doesNotMatch(result.elementPlan.formulaText, /overhead_multiplier/);
  assert.ok(Math.abs(result.totalGiB - 31.25) < 1e-9);
});

test("Qwen3.6 27B counts only full-attention KV layers", () => {
  const model = {
    id: "qwen3.6-27b",
    label: "Qwen3.6-27B",
    formula: "qwen_linear_full_hybrid",
    fields: {
      num_hidden_layers: 64,
      full_attention_layers: 16,
      linear_attention_layers: 48,
      num_key_value_heads: 4,
      head_dim: 256,
      linear_num_key_heads: 16,
      linear_key_head_dim: 128,
      linear_num_value_heads: 48,
      linear_value_head_dim: 128,
      linear_conv_kernel_dim: 4,
      mtp_num_hidden_layers: 1,
    },
  };

  const result = calculate(model, { ...bf16, tokens: 128000 });

  assert.equal(result.elementPlan.elementsPerToken, 32768);
  assert.equal(result.elementPlan.components.find(([label]) => label === "Linear-attention layers")[1], 48);
  assert.equal(result.elementPlan.components.find(([label]) => label === "Linear state included")[1], "No");
  assert.equal(result.elementPlan.components.find(([label]) => label === "MTP layers not included")[1], 1);
  assert.match(result.elementPlan.note, /excluded by default/);
  assert.ok(Math.abs(result.totalGiB - 7.8125) < 1e-9);
});

test("Qwen3.6 27B optional linear-attention state adds fixed conv and recurrent state", () => {
  const model = {
    id: "qwen3.6-27b",
    label: "Qwen3.6-27B",
    formula: "qwen_linear_full_hybrid",
    fields: {
      num_hidden_layers: 64,
      full_attention_layers: 16,
      linear_attention_layers: 48,
      num_key_value_heads: 4,
      head_dim: 256,
      linear_num_key_heads: 16,
      linear_key_head_dim: 128,
      linear_num_value_heads: 48,
      linear_value_head_dim: 128,
      linear_conv_kernel_dim: 4,
      mtp_num_hidden_layers: 1,
    },
  };

  const result = calculate(model, { ...bf16, tokens: 128000, includeLinearAttentionState: true });
  const fullBytes = 128000 * 16 * 2 * 4 * 256 * 2;
  const convBytes = 48 * 4 * (2 * 16 * 128 + 48 * 128) * 2;
  const recurrentBytes = 48 * 48 * 128 * 128 * 4;

  assert.equal(result.cacheGroups.find((group) => group.label === "Full-attention KV cache").bytes, fullBytes);
  assert.equal(result.cacheGroups.find((group) => group.label === "Linear-attention state").bytes, convBytes + recurrentBytes);
  assert.equal(result.elementPlan.components.find(([label]) => label === "Linear state included")[1], "Yes");
  assert.match(result.elementPlan.formulaRows.find((row) => row.name === "total_bytes").expression, /linear_recurrent_state_bytes/);
  assert.equal(result.totalBytes, fullBytes + convBytes + recurrentBytes);
});

test("Qwen3.6 35B-A3B counts only full-attention KV layers", () => {
  const model = {
    id: "qwen3.6-35b-a3b",
    label: "Qwen3.6-35B-A3B",
    formula: "qwen_linear_full_hybrid",
    fields: {
      num_hidden_layers: 40,
      full_attention_layers: 10,
      linear_attention_layers: 30,
      num_key_value_heads: 2,
      head_dim: 256,
      linear_num_key_heads: 16,
      linear_key_head_dim: 128,
      linear_num_value_heads: 32,
      linear_value_head_dim: 128,
      linear_conv_kernel_dim: 4,
      mtp_num_hidden_layers: 1,
    },
  };

  const result = calculate(model, { ...bf16, tokens: 128000 });

  assert.equal(result.elementPlan.elementsPerToken, 10240);
  assert.equal(result.elementPlan.components.find(([label]) => label === "Full-attention layers")[1], 10);
  assert.equal(result.elementPlan.components.find(([label]) => label === "Linear-attention layers")[1], 30);
  assert.ok(Math.abs(result.totalGiB - 2.44140625) < 1e-9);
});

test("Qwen3.5 small models count only full-attention KV layers", () => {
  const model = {
    id: "qwen3.5-0.8b",
    label: "Qwen3.5-0.8B",
    formula: "qwen_linear_full_hybrid",
    fields: {
      num_hidden_layers: 24,
      full_attention_layers: 6,
      linear_attention_layers: 18,
      num_key_value_heads: 2,
      head_dim: 256,
      linear_num_key_heads: 16,
      linear_key_head_dim: 128,
      linear_num_value_heads: 16,
      linear_value_head_dim: 128,
      linear_conv_kernel_dim: 4,
      mtp_num_hidden_layers: 1,
    },
  };

  const result = calculate(model, { ...bf16, tokens: 128000 });

  assert.equal(result.elementPlan.elementsPerToken, 6144);
  assert.equal(result.elementPlan.components.find(([label]) => label === "Full-attention layers")[1], 6);
  assert.equal(result.elementPlan.components.find(([label]) => label === "Linear-attention layers")[1], 18);
  assert.ok(Math.abs(result.totalGiB - 1.46484375) < 1e-9);
});

test("Qwen3.5 0.8B linear-attention state can dominate short prompts", () => {
  const model = {
    id: "qwen3.5-0.8b",
    label: "Qwen3.5-0.8B",
    formula: "qwen_linear_full_hybrid",
    fields: {
      num_hidden_layers: 24,
      full_attention_layers: 6,
      linear_attention_layers: 18,
      num_key_value_heads: 2,
      head_dim: 256,
      linear_num_key_heads: 16,
      linear_key_head_dim: 128,
      linear_num_value_heads: 16,
      linear_value_head_dim: 128,
      linear_conv_kernel_dim: 4,
      mtp_num_hidden_layers: 1,
    },
  };

  const result = calculate(model, { ...bf16, tokens: 128, includeLinearAttentionState: true });
  const fullBytes = 128 * 6 * 2 * 2 * 256 * 2;
  const convBytes = 18 * 4 * (2 * 16 * 128 + 16 * 128) * 2;
  const recurrentBytes = 18 * 16 * 128 * 128 * 4;
  const linearState = result.cacheGroups.find((group) => group.label === "Linear-attention state");

  assert.equal(linearState.bytes, convBytes + recurrentBytes);
  assert.ok(linearState.bytes > fullBytes);
  assert.equal(result.totalBytes, fullBytes + convBytes + recurrentBytes);
});

test("Qwen3.5 large MoE models count only full-attention KV layers", () => {
  const model = {
    id: "qwen3.5-397b-a17b",
    label: "Qwen3.5-397B-A17B",
    formula: "qwen_linear_full_hybrid",
    fields: {
      num_hidden_layers: 60,
      full_attention_layers: 15,
      linear_attention_layers: 45,
      num_key_value_heads: 2,
      head_dim: 256,
      linear_num_key_heads: 16,
      linear_key_head_dim: 128,
      linear_num_value_heads: 64,
      linear_value_head_dim: 128,
      linear_conv_kernel_dim: 4,
      mtp_num_hidden_layers: 1,
    },
  };

  const result = calculate(model, { ...bf16, tokens: 128000 });

  assert.equal(result.elementPlan.elementsPerToken, 15360);
  assert.equal(result.elementPlan.components.find(([label]) => label === "Full-attention layers")[1], 15);
  assert.equal(result.elementPlan.components.find(([label]) => label === "Linear-attention layers")[1], 45);
  assert.ok(Math.abs(result.totalGiB - 3.662109375) < 1e-9);
});

test("Gemma 4 E2B mixed formula applies KV sharing before sliding/full counts", () => {
  const model = {
    id: "gemma-4-e2b",
    label: "Gemma 4 E2B",
    formula: "mixed_full_sliding_gqa",
    fields: {
      num_hidden_layers: 35,
      stored_layers: 15,
      full_attention_layers: 3,
      sliding_attention_layers: 12,
      num_key_value_heads: 1,
      head_dim: 256,
      global_head_dim: 512,
      sliding_window: 512,
    },
  };

  const result = calculate(model, { ...bf16, tokens: 128000 });

  assert.equal(result.elementPlan.components.find(([label]) => label === "Stored layers")[1], 15);
  assert.equal(result.elementPlan.components.find(([label]) => label === "Retained sliding tokens")[1], 512);
  assert.equal(result.elementPlan.byteGroups.find((group) => group.label === "Full-attention KV cache").elements, 393216000);
  assert.equal(result.elementPlan.byteGroups.find((group) => group.label === "Sliding-window KV cache").elements, 3145728);
  assert.ok(Math.abs(result.totalGiB - 0.73828125) < 1e-9);
});

test("Gemma 4 31B mixed formula uses global full-attention heads and sliding window", () => {
  const model = {
    id: "gemma-4-31b",
    label: "Gemma 4 31B",
    formula: "mixed_full_sliding_gqa",
    fields: {
      num_hidden_layers: 60,
      stored_layers: 60,
      full_attention_layers: 10,
      sliding_attention_layers: 50,
      num_key_value_heads: 16,
      num_global_key_value_heads: 4,
      head_dim: 256,
      global_head_dim: 512,
      sliding_window: 1024,
    },
  };

  const result = calculate(model, { ...bf16, tokens: 128000 });

  assert.equal(result.elementPlan.byteGroups.find((group) => group.label === "Full-attention KV cache").elements, 5242880000);
  assert.equal(result.elementPlan.byteGroups.find((group) => group.label === "Sliding-window KV cache").elements, 419430400);
  assert.ok(Math.abs(result.totalGiB - 10.546875) < 1e-9);
});

test("Cohere Command R standard formula uses full MHA KV heads", () => {
  const model = {
    id: "cohere-command-r-v01",
    label: "Cohere Command R v01",
    formula: "standard_gqa",
    fields: {
      num_hidden_layers: 40,
      num_key_value_heads: 64,
      head_dim: 128,
    },
  };

  const result = calculate(model, { ...bf16, tokens: 128000 });

  assert.equal(result.elementPlan.elementsPerToken, 655360);
  assert.ok(Math.abs(result.totalGiB - 156.25) < 1e-9);
});

test("Cohere Command R+ standard formula uses GQA KV heads", () => {
  const model = {
    id: "cohere-command-r-plus",
    label: "Cohere Command R+",
    formula: "standard_gqa",
    fields: {
      num_hidden_layers: 64,
      num_key_value_heads: 8,
      head_dim: 128,
    },
  };

  const result = calculate(model, { ...bf16, tokens: 128000 });

  assert.equal(result.elementPlan.elementsPerToken, 131072);
  assert.ok(Math.abs(result.totalGiB - 31.25) < 1e-9);
});

test("Cohere Command R7B mixed formula caps sliding-attention KV", () => {
  const model = {
    id: "cohere-command-r7b-12-2024",
    label: "Cohere Command R7B 12-2024",
    formula: "mixed_full_sliding_gqa",
    fields: {
      num_hidden_layers: 32,
      full_attention_layers: 8,
      sliding_attention_layers: 24,
      num_key_value_heads: 8,
      head_dim: 128,
      sliding_window: 4096,
    },
  };

  const result = calculate(model, { ...bf16, tokens: 128000 });

  assert.equal(result.elementPlan.components.find(([label]) => label === "Retained sliding tokens")[1], 4096);
  assert.equal(result.elementPlan.byteGroups.find((group) => group.label === "Full-attention KV cache").elements, 2097152000);
  assert.equal(result.elementPlan.byteGroups.find((group) => group.label === "Sliding-window KV cache").elements, 201326592);
  assert.ok(Math.abs(result.totalGiB - 4.28125) < 1e-9);
});

test("Cohere Command A mixed formula caps sliding-attention KV", () => {
  const model = {
    id: "cohere-command-a-03-2025",
    label: "Cohere Command A 03-2025",
    formula: "mixed_full_sliding_gqa",
    fields: {
      num_hidden_layers: 64,
      full_attention_layers: 16,
      sliding_attention_layers: 48,
      num_key_value_heads: 8,
      head_dim: 128,
      sliding_window: 4096,
    },
  };

  const result = calculate(model, { ...bf16, tokens: 128000 });

  assert.equal(result.elementPlan.components.find(([label]) => label === "Retained sliding tokens")[1], 4096);
  assert.equal(result.elementPlan.byteGroups.find((group) => group.label === "Full-attention KV cache").elements, 4194304000);
  assert.equal(result.elementPlan.byteGroups.find((group) => group.label === "Sliding-window KV cache").elements, 402653184);
  assert.ok(Math.abs(result.totalGiB - 8.5625) < 1e-9);
});

test("Cohere Command A Plus mixed formula caps sliding-attention KV", () => {
  const model = {
    id: "cohere-command-a-plus-05-2026",
    label: "Cohere Command A Plus 05-2026",
    formula: "mixed_full_sliding_gqa",
    fields: {
      num_hidden_layers: 32,
      full_attention_layers: 8,
      sliding_attention_layers: 24,
      num_key_value_heads: 8,
      head_dim: 128,
      sliding_window: 4096,
    },
  };

  const result = calculate(model, { ...bf16, tokens: 128000 });

  assert.equal(result.elementPlan.components.find(([label]) => label === "Retained sliding tokens")[1], 4096);
  assert.equal(result.elementPlan.byteGroups.find((group) => group.label === "Full-attention KV cache").elements, 2097152000);
  assert.equal(result.elementPlan.byteGroups.find((group) => group.label === "Sliding-window KV cache").elements, 201326592);
  assert.ok(Math.abs(result.totalGiB - 4.28125) < 1e-9);
});

test("MiMo V2.5 mixed formula uses separate K and V dimensions", () => {
  const model = {
    id: "mimo-v2.5",
    label: "MiMo-V2.5",
    formula: "mixed_full_sliding_gqa",
    fields: {
      num_hidden_layers: 48,
      full_attention_layers: 9,
      sliding_attention_layers: 39,
      num_key_value_heads: 4,
      head_dim: 192,
      v_head_dim: 128,
      swa_num_key_value_heads: 8,
      swa_head_dim: 192,
      swa_v_head_dim: 128,
      sliding_window: 128,
    },
  };

  const result = calculate(model, { ...bf16, tokens: 128000 });

  assert.equal(result.elementPlan.components.find(([label]) => label === "Retained sliding tokens")[1], 128);
  assert.equal(result.elementPlan.components.find(([label]) => label === "Full K+V dims")[1], 320);
  assert.equal(result.elementPlan.components.find(([label]) => label === "Sliding K+V dims")[1], 320);
  assert.equal(result.elementPlan.byteGroups.find((group) => group.label === "Full-attention KV cache").elements, 1474560000);
  assert.equal(result.elementPlan.byteGroups.find((group) => group.label === "Sliding-window KV cache").elements, 12779520);
  assert.match(result.elementPlan.formulaText, /full_head_dim \+ full_v_head_dim/);
  assert.ok(Math.abs(result.totalGiB - 2.7703857421875) < 1e-9);
});

test("MiMo V2.5 Pro mixed formula caps SWA tokens at 128", () => {
  const model = {
    id: "mimo-v2.5-pro",
    label: "MiMo-V2.5-Pro",
    formula: "mixed_full_sliding_gqa",
    fields: {
      num_hidden_layers: 70,
      full_attention_layers: 10,
      sliding_attention_layers: 60,
      num_key_value_heads: 8,
      head_dim: 192,
      v_head_dim: 128,
      swa_num_key_value_heads: 8,
      swa_head_dim: 192,
      swa_v_head_dim: 128,
      sliding_window: 128,
    },
  };

  const result = calculate(model, { ...bf16, tokens: 128000 });

  assert.equal(result.elementPlan.components.find(([label]) => label === "Retained sliding tokens")[1], 128);
  assert.equal(result.elementPlan.byteGroups.find((group) => group.label === "Full-attention KV cache").elements, 3276800000);
  assert.equal(result.elementPlan.byteGroups.find((group) => group.label === "Sliding-window KV cache").elements, 19660800);
  assert.ok(Math.abs(result.totalGiB - 6.14013671875) < 1e-9);
});

test("MLA formula matches Kimi K2.5 latent KV cache", () => {
  const model = {
    id: "kimi-k2.5",
    label: "Kimi K2.5",
    formula: "mla",
    fields: { num_hidden_layers: 61, kv_lora_rank: 512, qk_rope_head_dim: 64 },
  };

  const result = calculate(model, { ...bf16, tokens: 128000 });
  assert.equal(result.elementPlan.elementsPerToken, 35136);
  assert.ok(Math.abs(result.totalGiB - 8.3770751953125) < 1e-9);
});

test("DSA optimized formula includes latent cache and indexer state", () => {
  const model = {
    id: "glm-5",
    label: "GLM-5",
    formula: "dsa_mla",
    fields: {
      num_hidden_layers: 78,
      kv_lora_rank: 512,
      qk_rope_head_dim: 64,
      index_head_dim: 128,
      num_key_value_heads: 64,
      qk_head_dim: 256,
      v_head_dim: 256,
    },
  };

  const result = calculate(model, { ...bf16, tokens: 128000 });
  assert.equal(result.elementPlan.elementsPerToken, 54912);
  assert.match(result.elementPlan.note, /Production estimate/);
  assert.ok(Math.abs(result.totalGiB - 13.092041015625) < 1e-9);
});

test("GLM-5.2 DSA formula counts shared indexer layers only once", () => {
  const model = {
    id: "glm-5.2",
    label: "GLM-5.2",
    formula: "dsa_mla",
    fields: {
      num_hidden_layers: 78,
      kv_lora_rank: 512,
      qk_rope_head_dim: 64,
      index_head_dim: 128,
      indexer_full_layers: 21,
      indexer_shared_layers: 57,
      num_nextn_predict_layers: 1,
      draft_indexer_layers: 1,
    },
  };

  const result = calculate(model, {
    tokens: 128000,
    sequences: 1,
    precision: "fp8_int8",
    indexerPrecision: "fp4_int4",
  });

  assert.equal(result.elementPlan.components.find(([label]) => label === "Main indexer layers")[1], 21);
  assert.equal(result.elementPlan.components.find(([label]) => label === "Shared indexer layers")[1], 57);
  assert.equal(result.elementPlan.byteGroups.find((group) => group.role === "kv").elements, 5750784000);
  assert.equal(result.elementPlan.byteGroups.find((group) => group.role === "indexer").elements, 344064000);
  assert.ok(Math.abs(result.kvGiB - 5.3558349609375) < 1e-9);
  assert.ok(Math.abs(result.indexerGiB - 0.16021728515625) < 1e-9);
  assert.ok(Math.abs(result.totalGiB - 5.51605224609375) < 1e-9);
  assert.match(result.elementPlan.note, /shared indexer layers/);
});

test("GLM-5.2 draft option adds one KV layer and one full indexer layer", () => {
  const model = {
    id: "glm-5.2",
    label: "GLM-5.2",
    formula: "dsa_mla",
    fields: {
      num_hidden_layers: 78,
      kv_lora_rank: 512,
      qk_rope_head_dim: 64,
      index_head_dim: 128,
      indexer_full_layers: 21,
      indexer_shared_layers: 57,
      num_nextn_predict_layers: 1,
      draft_indexer_layers: 1,
    },
  };

  const withoutDraft = calculate(model, {
    tokens: 128000,
    sequences: 1,
    precision: "fp8_int8",
    indexerPrecision: "fp4_int4",
    includeDraftKvCache: false,
  });
  const withDraft = calculate(model, {
    tokens: 128000,
    sequences: 1,
    precision: "fp8_int8",
    indexerPrecision: "fp4_int4",
    includeDraftKvCache: true,
  });

  assert.equal(withDraft.elementPlan.components.find(([label]) => label === "Draft layers included")[1], 1);
  assert.equal(
    withDraft.elementPlan.components.find(([label]) => label === "Draft indexer layers included")[1],
    1,
  );
  assert.equal(
    withDraft.elementPlan.byteGroups.find((group) => group.role === "kv").elements -
      withoutDraft.elementPlan.byteGroups.find((group) => group.role === "kv").elements,
    73728000,
  );
  assert.equal(
    withDraft.elementPlan.byteGroups.find((group) => group.role === "indexer").elements -
      withoutDraft.elementPlan.byteGroups.find((group) => group.role === "indexer").elements,
    16384000,
  );
  assert.ok(Math.abs(withDraft.totalGiB - withoutDraft.totalGiB - 0.0762939453125) < 1e-9);
});

test("DeepSeek V4 hybrid formula uses sliding window, compression ratios, and ratio-4 indexer", () => {
  const compressRatios = [
    128, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0,
  ];
  const model = {
    id: "deepseek-v4-pro",
    label: "DeepSeek V4 Pro",
    formula: "deepseek_v4_hybrid",
    fields: {
      head_dim: 512,
      sliding_window: 128,
      num_hidden_layers: 61,
      index_head_dim: 128,
      compress_ratios: compressRatios,
    },
  };

  const plan = calculateElementsPerSequence(model, 128000);
  assert.equal(plan.components.find(([label]) => label === "Main layers")[1], 61);
  assert.equal(plan.components.find(([label]) => label === "Draft layers included")[1], 0);
  assert.equal(plan.components.find(([label]) => label === "Ratio=4 layers")[1], 30);
  assert.equal(plan.components.find(([label]) => label === "Ratio=128 layers")[1], 31);
  assert.equal(plan.components.find(([label]) => label === "Ratio=0 layers")[1], 0);
  assert.equal(plan.components.find(([label]) => label === "Ratio=0 KV elements")[1], 0);
  assert.equal(plan.components.find(([label]) => label === "Sliding-window elements")[1], 3997696);
  assert.ok(plan.formulaRows.some((row) => row.name === "sliding_kv_bytes"));
  assert.ok(plan.formulaRows.some((row) => row.name === "compressed_kv_bytes"));
  assert.ok(plan.formulaRows.some((row) => row.name === "kv_bytes"));
  assert.match(
    plan.formulaRows.find((row) => row.name === "total_bytes").expression,
    /kv_bytes \+ indexer_bytes/,
  );
  assert.equal(plan.byteGroups.find((group) => group.role === "kv").elements, 511389696);
  assert.equal(plan.byteGroups.find((group) => group.role === "indexer").elements, 122880000);

  const result = calculate(model, { ...bf16, tokens: 128000 });
  assert.ok(Math.abs(result.totalGiB - 1.1814193725585938) < 1e-9);
});

test("DeepSeek V4 defaults to FP8 attention and FP4 indexer cache", () => {
  const compressRatios = [
    128, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0,
  ];
  const model = {
    id: "deepseek-v4-pro",
    label: "DeepSeek V4 Pro",
    formula: "deepseek_v4_hybrid",
    fields: {
      head_dim: 512,
      sliding_window: 128,
      num_hidden_layers: 61,
      index_head_dim: 128,
      compress_ratios: compressRatios,
    },
  };

  const result = calculate(model, { tokens: 128000, sequences: 1, tensorParallel: 1 });

  assert.equal(result.precisionLabel, "FP8 / INT8");
  assert.equal(result.indexerPrecisionLabel, "FP4 / INT4");
  assert.ok(Math.abs(result.totalGiB - 0.5334892272949219) < 1e-9);
});

test("DeepSeek V4 can calculate explicit FP8 attention and FP4 indexer cache", () => {
  const compressRatios = [
    128, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0,
  ];
  const model = {
    id: "deepseek-v4-pro",
    label: "DeepSeek V4 Pro",
    formula: "deepseek_v4_hybrid",
    fields: {
      head_dim: 512,
      sliding_window: 128,
      num_hidden_layers: 61,
      index_head_dim: 128,
      compress_ratios: compressRatios,
    },
  };

  const result = calculate(
    model,
    { tokens: 128000, sequences: 1, precision: "fp8_int8", indexerPrecision: "fp4_int4", tensorParallel: 1 },
    {
      precisionOptions: [
        { id: "bf16_fp16", label: "BF16 / FP16", bytes_per_element: 2 },
        { id: "fp8_int8", label: "FP8 / INT8", bytes_per_element: 1 },
      ],
      indexerPrecisionOptions: [
        { id: "bf16_fp16", label: "BF16 / FP16", bytes_per_element: 2 },
        { id: "fp4_int4", label: "FP4 / INT4", bytes_per_element: 0.5 },
      ],
    },
  );

  assert.ok(Math.abs(result.totalGiB - 0.5334892272949219) < 1e-9);
  assert.equal(result.components.find(([label]) => label === "KV precision bytes")[1], 1);
  assert.equal(result.components.find(([label]) => label === "Indexer precision bytes")[1], 0.5);
});

test("DSA indexer models can use separate KV and indexer precision", () => {
  const model = {
    id: "deepseek-v3.2",
    label: "DeepSeek V3.2",
    formula: "dsa_mla",
    fields: {
      num_hidden_layers: 61,
      kv_lora_rank: 512,
      qk_rope_head_dim: 64,
      index_head_dim: 128,
      num_nextn_predict_layers: 1,
    },
  };

  const result = calculate(model, {
    tokens: 128000,
    sequences: 1,
    precision: "fp8_int8",
    indexerPrecision: "fp4_int4",
  });

  assert.equal(result.elementPlan.byteGroups.find((group) => group.role === "kv").elements, 4497408000);
  assert.equal(result.elementPlan.byteGroups.find((group) => group.role === "indexer").elements, 999424000);
  assert.ok(Math.abs(result.kvGiB - 4.18853759765625) < 1e-9);
  assert.ok(Math.abs(result.indexerGiB - 0.46539306640625) < 1e-9);
  assert.ok(Math.abs(result.totalGiB - 4.6539306640625) < 1e-9);
  assert.equal(result.elementPlan.components.find(([label]) => label === "Draft layers included")[1], 0);
  assert.match(result.elementPlan.formulaText, /active_layers/);
});

test("DSA draft option adds one latent KV and indexer layer", () => {
  const model = {
    id: "deepseek-v3.2",
    label: "DeepSeek V3.2",
    formula: "dsa_mla",
    fields: {
      num_hidden_layers: 61,
      kv_lora_rank: 512,
      qk_rope_head_dim: 64,
      index_head_dim: 128,
      num_nextn_predict_layers: 1,
    },
  };

  const withoutDraft = calculate(model, {
    tokens: 128000,
    sequences: 1,
    precision: "fp8_int8",
    indexerPrecision: "fp4_int4",
    includeDraftKvCache: false,
  });
  const withDraft = calculate(model, {
    tokens: 128000,
    sequences: 1,
    precision: "fp8_int8",
    indexerPrecision: "fp4_int4",
    includeDraftKvCache: true,
  });

  assert.equal(withDraft.elementPlan.components.find(([label]) => label === "Draft layers included")[1], 1);
  assert.equal(
    withDraft.elementPlan.byteGroups.find((group) => group.role === "kv").elements -
      withoutDraft.elementPlan.byteGroups.find((group) => group.role === "kv").elements,
    73728000,
  );
  assert.equal(
    withDraft.elementPlan.byteGroups.find((group) => group.role === "indexer").elements -
      withoutDraft.elementPlan.byteGroups.find((group) => group.role === "indexer").elements,
    16384000,
  );
  assert.ok(Math.abs(withDraft.totalGiB - withoutDraft.totalGiB - 0.0762939453125) < 1e-9);
});

test("DeepSeek V4 draft KV cache option adds the ratio-0 draft layer", () => {
  const compressRatios = [
    128, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0,
  ];
  const model = {
    id: "deepseek-v4-pro",
    label: "DeepSeek V4 Pro",
    formula: "deepseek_v4_hybrid",
    fields: {
      head_dim: 512,
      sliding_window: 128,
      num_hidden_layers: 61,
      index_head_dim: 128,
      compress_ratios: compressRatios,
    },
  };

  const withoutDraft = calculateElementsPerSequence(model, 128000, { includeDraftKvCache: false });
  const withDraft = calculateElementsPerSequence(model, 128000, { includeDraftKvCache: true });

  assert.equal(withDraft.components.find(([label]) => label === "Draft layers included")[1], 1);
  assert.equal(withDraft.components.find(([label]) => label === "Ratio=0 layers")[1], 1);
  assert.equal(withDraft.components.find(([label]) => label === "Ratio=0 KV elements")[1], 65536);
  assert.equal(withDraft.components.find(([label]) => label === "Sliding-window elements")[1], 4063232);
  assert.equal(
    withDraft.byteGroups.find((group) => group.role === "kv").elements -
      withoutDraft.byteGroups.find((group) => group.role === "kv").elements,
    65536,
  );
});

test("standard models ignore indexer precision and keep single precision scaling", () => {
  const model = {
    id: "minimax-m2",
    label: "MiniMax M2",
    formula: "standard_gqa",
    fields: { num_hidden_layers: 62, num_key_value_heads: 8, head_dim: 128 },
  };

  const result = calculate(model, {
    tokens: 128000,
    precision: "fp8_int8",
    indexerPrecision: "fp4_int4",
    sequences: 4,
    tensorParallel: 2,
  });

  assert.ok(Math.abs(result.totalGiB - 60.546875) < 1e-9);
  assert.equal(result.indexerBytes, 0);
});

test("DeepSeek V3 MLA draft option adds one latent KV layer", () => {
  const model = {
    id: "deepseek-v3",
    label: "DeepSeek V3",
    formula: "mla",
    fields: {
      num_hidden_layers: 61,
      kv_lora_rank: 512,
      qk_rope_head_dim: 64,
      num_nextn_predict_layers: 1,
    },
  };

  const withoutDraft = calculateElementsPerSequence(model, 128000, { includeDraftKvCache: false });
  const withDraft = calculateElementsPerSequence(model, 128000, { includeDraftKvCache: true });

  assert.equal(withoutDraft.elementsPerToken, 35136);
  assert.equal(withDraft.elementsPerToken, 35712);
  assert.equal(withDraft.components.find(([label]) => label === "Draft layers included")[1], 1);
  assert.match(withDraft.formulaText, /active_layers/);
});

test("MiniMax M2 draft option adds three standard GQA KV layers", () => {
  const model = {
    id: "minimax-m2",
    label: "MiniMax M2",
    formula: "standard_gqa",
    fields: {
      num_hidden_layers: 62,
      num_key_value_heads: 8,
      head_dim: 128,
      use_mtp: true,
      num_mtp_modules: 3,
      mtp_transformer_layers: 1,
    },
  };

  const withoutDraft = calculateElementsPerSequence(model, 128000, { includeDraftKvCache: false });
  const withDraft = calculateElementsPerSequence(model, 128000, { includeDraftKvCache: true });

  assert.equal(withoutDraft.elementsPerToken, 126976);
  assert.equal(withDraft.elementsPerToken, 133120);
  assert.equal(withDraft.components.find(([label]) => label === "Draft layers included")[1], 3);
});

test("Llama 3.1 70B standard GQA formula matches config fields", () => {
  const model = {
    id: "llama-3.1-70b",
    label: "Llama 3.1 70B",
    formula: "standard_gqa",
    fields: { num_hidden_layers: 80, num_key_value_heads: 8, head_dim: 128 },
  };

  const result = calculate(model, { ...bf16, tokens: 128000 });
  assert.equal(result.elementPlan.elementsPerToken, 163840);
  assert.ok(Math.abs(result.totalGiB - 39.0625) < 1e-9);
});

test("Qwen2.5 72B standard GQA formula ignores draft input", () => {
  const model = {
    id: "qwen2.5-72b",
    label: "Qwen2.5-72B",
    formula: "standard_gqa",
    fields: { num_hidden_layers: 80, num_key_value_heads: 8, head_dim: 128 },
  };

  const withoutDraft = calculate(model, { ...bf16, tokens: 128000, includeDraftKvCache: false });
  const withDraft = calculate(model, { ...bf16, tokens: 128000, includeDraftKvCache: true });

  assert.equal(withoutDraft.elementPlan.elementsPerToken, 163840);
  assert.equal(withDraft.elementPlan.elementsPerToken, withoutDraft.elementPlan.elementsPerToken);
  assert.ok(Math.abs(withDraft.totalGiB - 39.0625) < 1e-9);
});

test("standard GQA formula ignores Qwen linear-attention state input", () => {
  const model = {
    id: "llama-3.1-70b",
    label: "Llama 3.1 70B",
    formula: "standard_gqa",
    fields: { num_hidden_layers: 80, num_key_value_heads: 8, head_dim: 128 },
  };

  const withoutLinearState = calculate(model, { ...bf16, tokens: 4096, includeLinearAttentionState: false });
  const withLinearState = calculate(model, { ...bf16, tokens: 4096, includeLinearAttentionState: true });

  assert.equal(withLinearState.totalBytes, withoutLinearState.totalBytes);
  assert.equal(withLinearState.cacheGroups.length, withoutLinearState.cacheGroups.length);
});

test("model family grouping keeps Qwen generations under one family", () => {
  const models = [
    { id: "qwen3.6-27b", label: "Qwen3.6-27B", family: "Qwen3.6" },
    { id: "qwen3.5-397b-a17b", label: "Qwen3.5-397B-A17B", family: "Qwen3.5" },
    { id: "qwen3-32b", label: "Qwen3-32B", family: "Qwen3" },
    { id: "qwen2.5-72b", label: "Qwen2.5-72B", family: "Qwen2.5" },
    { id: "deepseek-v3", label: "DeepSeek V3", family: "DeepSeek" },
  ];

  assert.equal(modelFamily(models[0]), "Qwen");
  assert.deepEqual(modelsForFamily(models, "Qwen").map((model) => model.id), [
    "qwen3.6-27b",
    "qwen3.5-397b-a17b",
    "qwen3-32b",
    "qwen2.5-72b",
  ]);
});

test("display byte formatter keeps five decimal places", () => {
  assert.equal(formatBytes(1024), "1.00000 KiB");
  assert.equal(formatBytes(1024 ** 3), "1.00000 GiB");
});

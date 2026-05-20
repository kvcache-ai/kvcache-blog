import assert from "node:assert/strict";
import { createRequire } from "node:module";
import test from "node:test";

const require = createRequire(import.meta.url);
const { calculate, calculateElementsPerSequence, formatBytes } = require("../assets/js/kv-cache-calculator.js");

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
  assert.equal(plan.components.find(([label]) => label === "Layers")[1], 61);
  assert.equal(plan.components.find(([label]) => label === "Ratio=4 layers")[1], 30);
  assert.equal(plan.components.find(([label]) => label === "Ratio=128 layers")[1], 31);
  assert.equal(plan.components.find(([label]) => label === "Ratio=0 layers")[1], 0);
  assert.equal(plan.byteGroups.find((group) => group.role === "attention").elements, 511389696);
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
  assert.equal(result.components.find(([label]) => label === "Attention precision bytes")[1], 1);
  assert.equal(result.components.find(([label]) => label === "Indexer precision bytes")[1], 0.5);
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

  assert.equal(result.totalCachedTokens, 512000);
  assert.ok(Math.abs(result.totalGiB - 60.546875) < 1e-9);
  assert.ok(Math.abs(result.perDeviceGiB - 30.2734375) < 1e-9);
});

test("display byte formatter keeps five decimal places", () => {
  assert.equal(formatBytes(1024), "1.00000 KiB");
  assert.equal(formatBytes(1024 ** 3), "1.00000 GiB");
});

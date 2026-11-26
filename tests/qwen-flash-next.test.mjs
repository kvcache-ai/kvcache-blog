import assert from "node:assert/strict";
import { createRequire } from "node:module";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { loadModelsData } from "../scripts/lib/kv-cache-lab-traces.mjs";

const require = createRequire(import.meta.url);
const { calculate, modelFamily } = require("../assets/js/kv-cache-calculator.js");
const models = loadModelsData(
  fileURLToPath(new URL("../data/kv_cache_calculator/models.yaml", import.meta.url)),
).models;
const model = models.find((entry) => entry.id === "qwen3.8-flash-next");

test("Qwen3.8-Flash-Next preset preserves its official Qwen4-Exp architecture", () => {
  assert.ok(model);
  assert.equal(model.source_url, "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/812d96e10a0dd82b69d749e68693698cf6c36daa/config.json");
  assert.equal(model.fields.architectures[0], "Qwen4ExpForConditionalGeneration");
  assert.equal(model.fields.model_type, "qwen4_exp_text");
  assert.equal(model.fields.num_hidden_layers, 48);
  assert.equal(model.fields.full_attention_interval, 4);
  assert.equal(model.fields.full_attention_layers, 12);
  assert.equal(model.fields.linear_attention_layers, 36);
  assert.equal(model.max_position_embeddings, 262144);
  assert.equal(model.hit_rate_supported, false);
  assert.equal(modelFamily(model), "Qwen");
});

test("QSA base payload counts completed compressed groups and ordinary GQA KV", () => {
  const expected = new Map([
    [1, 24576],
    [3, 73728],
    [4, 101376],
    [5, 125952],
  ]);

  for (const [tokens, totalBytes] of expected) {
    const result = calculate(model, {
      tokens,
      precision: "bf16_fp16",
      indexerPrecision: "bf16_fp16",
      includeLinearAttentionState: false,
    });
    assert.equal(result.kvBytes, tokens * 12 * 2 * 2 * 256 * 2);
    assert.equal(result.indexerBytes, Math.floor(tokens / 4) * 12 * 128 * 2);
    assert.equal(result.totalBytes, totalBytes);
  }
});

test("legacy Qwen3.8 GDN element accounting does not gain QSA-only state", () => {
  const legacy = models.find((entry) => entry.id === "qwen3.8-27b");
  const result = calculate(legacy, {
    tokens: 1,
    includeLinearAttentionState: true,
  });

  assert.equal(result.elementPlan.elementsPerSequence, 39256064);
});

test("one continuation snapshot separates GDN, QSA pending ring, and PLE state", () => {
  const result = calculate(model, {
    tokens: 1,
    precision: "bf16_fp16",
    indexerPrecision: "bf16_fp16",
    includeLinearAttentionState: true,
    recurrentStatePrecision: "fp32",
  });
  const stateBytes = Object.fromEntries(
    result.cacheGroups
      .filter((group) => group.role === "linear_state")
      .map((group) => [group.label, group.bytes]),
  );

  assert.equal(stateBytes["GDN recurrent/conv checkpoint"], 115458048);
  assert.equal(stateBytes["QSA pending index state"], 12384);
  assert.equal(stateBytes["PLE continuation state"], 184336);
  assert.equal(Object.values(stateBytes).reduce((sum, bytes) => sum + bytes, 0), 115654768);
  assert.match(result.elementPlan.note, /Logical payload only/);
  assert.match(result.elementPlan.note, /replicates QSA indexer state/);
});

test("fixed intervals count final partial snapshots and recurrent precision remains independent", () => {
  const input = {
    tokens: 10241,
    sequences: 3,
    precision: "fp8_int8",
    indexerPrecision: "fp8_int8",
    includeLinearAttentionState: true,
    qwenCheckpointPolicy: "fixed_interval",
    qwenCheckpointInterval: 10240,
  };
  const fp32 = calculate(model, { ...input, recurrentStatePrecision: "fp32" });
  const bf16 = calculate(model, { ...input, recurrentStatePrecision: "bf16_fp16" });
  const stateBytes = (result) =>
    result.cacheGroups
      .filter((group) => group.role === "linear_state")
      .reduce((sum, group) => sum + group.bytes, 0);

  assert.equal(fp32.elementPlan.components.find(([label]) => label === "GDN checkpoints per sequence")[1], 2);
  assert.equal(stateBytes(fp32), 3 * 2 * 115654768);
  assert.equal(stateBytes(bf16), 3 * 2 * 59031664);
  assert.equal(fp32.kvBytes, bf16.kvBytes);
  assert.equal(fp32.indexerBytes, bf16.indexerBytes);
});

test("model precision allowlists reject FP4 fallback and expose the Kimi TP cache note", () => {
  assert.deepEqual(model.fields.kv_precision_ids, ["bf16_fp16", "fp8_int8"]);
  assert.deepEqual(model.fields.indexer_precision_ids, ["bf16_fp16", "fp8_int8"]);
  const result = calculate(model, {
    tokens: 4,
    precision: "fp4_int4",
    indexerPrecision: "fp4_int4",
    includeLinearAttentionState: false,
  });
  assert.equal(result.precisionLabel, "BF16 / FP16");
  assert.equal(result.indexerPrecisionLabel, "BF16 / FP16");

  const kimi = models.find((entry) => entry.id === "kimi-k3.1-pro");
  assert.match(calculate(kimi, { tokens: 1 }).elementPlan.note, /TP=1 sizing basis/);
});

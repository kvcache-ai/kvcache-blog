import assert from "node:assert/strict";
import { createRequire } from "node:module";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { loadModelsData } from "../scripts/lib/kv-cache-lab-traces.mjs";

const require = createRequire(import.meta.url);
const { calculate } = require("../assets/js/kv-cache-calculator.js");
const models = loadModelsData(
  fileURLToPath(new URL("../data/kv_cache_calculator/models.yaml", import.meta.url)),
).models;

function model(id) {
  const result = models.find((entry) => entry.id === id);
  assert.ok(result, `Missing expanded 2026 model profile: ${id}`);
  return result;
}

test("expanded family presets keep official IDs, pinned config sources and distinct records", () => {
  const cases = [
    ["cohere-north-mini-code-1.0", "CohereLabs/North-Mini-Code-1.0", "mixed_full_sliding_gqa"],
    ["cohere-north-micro-vision-instruct", "CohereLabs/North-Micro-Vision-Instruct", "mixed_full_sliding_gqa"],
    ["gemma-4-12b", "google/gemma-4-12B", "mixed_full_sliding_gqa"],
    ["inkling-2-small", "thinkingmachines/Inkling-2-Small", "inkling_hybrid"],
    ["mimo-v2.6-flash", "XiaomiMiMo/MiMo-V2.6-Flash-RL", "mixed_full_sliding_gqa"],
    ["mimo-v2.6-pro", "XiaomiMiMo/MiMo-V2.6-Pro-RL", "mixed_full_sliding_gqa"],
    ["mimo-v2.6-distill-qwen-9b", "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B", "qwen_linear_full_hybrid"],
    ["minimax-m3.1-flash", "MiniMaxAI/MiniMax-M3.1-Flash", "minimax_msa"],
    ["minimax-h3-image-pe", "MiniMaxAI/MiniMax-H3-Image-PE", "qwen_linear_full_hybrid"],
  ];

  assert.equal(new Set(cases.map(([id]) => id)).size, cases.length);
  assert.equal(new Set(cases.map(([, repo]) => repo)).size, cases.length);
  for (const [id, repo, formula] of cases) {
    const entry = model(id);
    assert.equal(entry.official_model_id, repo);
    assert.equal(entry.formula, formula);
    assert.match(entry.source_url, new RegExp(`huggingface\\.co/${repo.replaceAll("/", "/")}/raw/[a-f0-9]{40}/config\\.json`));
    assert.equal(entry.source_retrieved_at, "2026-11-18");
    assert.equal(typeof entry.hub_repo_created_at, "string");
  }
});

test("Gemma 4 12B counts both cached K and V despite attention_k_eq_v weight sharing", () => {
  const entry = model("gemma-4-12b");
  assert.equal(entry.fields.attention_k_eq_v, true);
  assert.equal(entry.fields.num_global_key_value_heads, 1);
  assert.equal(entry.fields.global_head_dim, 512);
  assert.equal(entry.fields.num_key_value_heads, 8);
  assert.deepEqual(
    [entry.fields.full_attention_layers, entry.fields.sliding_attention_layers],
    [8, 40],
  );
  const result = calculate(entry, { tokens: 1, precision: "bf16_fp16" });
  assert.equal(result.kvBytes, (8 * 1 * (512 + 512) + 40 * 8 * (256 + 256)) * 2);
  assert.match(entry.cache_note, /both cached, so the cache is not halved/);
});

test("Inkling 2 Small estimates a FP32 logical SConv snapshot and omits unverified MTP cache", () => {
  const entry = model("inkling-2-small");
  assert.equal(entry.fields.sconv_state_bytes_per_element, 4);
  assert.equal(entry.fields.sconv_kernel_size, 4);
  assert.equal(entry.fields.default_include_sconv_state, true);
  assert.equal(entry.fields.disable_draft_kv_cache, true);
  const result = calculate(entry, { tokens: 1025, precision: "bf16_fp16" });
  assert.equal(result.kvBytes, 201359360);
  assert.equal(result.cacheGroups.find((group) => group.role === "sconv_state").bytes, 6488064);
  assert.equal(result.totalBytes, 207847424);
  assert.match(entry.cache_note, /four SConv slots/);
});

test("MiMo V2.6 profiles use distinct full/SWA KV heads and K/V dimensions", () => {
  const flash = model("mimo-v2.6-flash");
  const pro = model("mimo-v2.6-pro");
  assert.deepEqual(
    [flash.fields.num_key_value_heads, flash.fields.head_dim, flash.fields.v_head_dim,
      flash.fields.swa_num_key_value_heads, flash.fields.swa_head_dim, flash.fields.swa_v_head_dim],
    [4, 192, 128, 8, 192, 128],
  );
  assert.deepEqual(
    [pro.fields.num_key_value_heads, pro.fields.head_dim, pro.fields.v_head_dim,
      pro.fields.swa_num_key_value_heads, pro.fields.swa_head_dim, pro.fields.swa_v_head_dim],
    [8, 192, 128, 8, 192, 128],
  );
  assert.equal(calculate(flash, { tokens: 1024, precision: "bf16_fp16" }).kvBytes, 49152000);
  assert.equal(calculate(pro, { tokens: 1024, precision: "bf16_fp16" }).kvBytes, 91750400);
});

test("MiMo distill and H3 Image PE use the Qwen hybrid decoder profile", () => {
  const distill = model("mimo-v2.6-distill-qwen-9b");
  assert.deepEqual(
    [distill.fields.num_hidden_layers, distill.fields.full_attention_layers,
      distill.fields.linear_attention_layers, distill.fields.mtp_num_hidden_layers],
    [32, 8, 24, 1],
  );
  const distillResult = calculate(distill, {
    tokens: 1024,
    precision: "bf16_fp16",
    includeLinearAttentionState: true,
    recurrentStatePrecision: "fp32",
  });
  assert.equal(distillResult.kvBytes, 33554432);
  assert.equal(distillResult.totalBytes, 85065728);
  assert.equal(distill.fields.disable_draft_kv_cache, true);

  const h3 = model("minimax-h3-image-pe");
  const h3Result = calculate(h3, {
    tokens: 1024,
    precision: "bf16_fp16",
    includeLinearAttentionState: true,
    recurrentStatePrecision: "fp32",
  });
  assert.equal(h3.fields.num_hidden_layers, 64);
  assert.equal(h3.fields.full_attention_layers, 16);
  assert.equal(h3.fields.linear_attention_layers, 48);
  assert.equal(h3Result.totalBytes, 221052928);
  assert.match(h3.cache_note, /diffusion\/image model/);
});

test("MiniMax M3.1 Flash exposes independent KV and indexer NVFP4 while keeping BF16 defaults", () => {
  const entry = model("minimax-m3.1-flash");
  assert.equal(entry.fields.sparse_attention_layers, 60);
  assert.deepEqual(entry.fields.kv_precision_ids, ["bf16_fp16", "fp8_int8", "nvfp4_g16_s1"]);
  assert.deepEqual(entry.fields.indexer_precision_ids, ["bf16_fp16", "nvfp4_g16_s1"]);
  assert.equal(entry.fields.indexer_fixed_precision_id, undefined);
  const bf16 = calculate(entry, {
    tokens: 1024,
  });
  const kvOnlyNvfp4 = calculate(entry, {
    tokens: 1024,
    precision: "nvfp4_g16_s1",
    indexerPrecision: "bf16_fp16",
  });
  const allNvfp4 = calculate(entry, {
    tokens: 1024,
    precision: "nvfp4_g16_s1",
    indexerPrecision: "nvfp4_g16_s1",
  });
  const indexerOnlyNvfp4 = calculate(entry, {
    tokens: 1024,
    precision: "bf16_fp16",
    indexerPrecision: "nvfp4_g16_s1",
  });
  assert.equal(bf16.precisionLabel, "BF16 / FP16");
  assert.equal(bf16.indexerPrecisionLabel, "BF16 / FP16");
  assert.equal(bf16.totalBytes, 141557760);
  assert.equal(kvOnlyNvfp4.totalBytes, 51118080);
  assert.equal(indexerOnlyNvfp4.totalBytes, 130252800);
  assert.equal(allNvfp4.totalBytes, 39813120);
  assert.equal(allNvfp4.perDeviceBytes, allNvfp4.totalBytes);
  assert.match(entry.cache_note, /weights; KV precision is selected independently/);
  assert.match(entry.cache_note, /allocator page padding is excluded/);
  const formulaRows = allNvfp4.elementPlan.formulaRows;
  assert.match(formulaRows.find((row) => row.name === "total_bytes").expression, /kv_scale_bytes/);
  assert.match(formulaRows.find((row) => row.name === "total_bytes").expression, /indexer_scale_bytes/);

  for (const tokens of [1, 3, 17]) {
    const batch = calculate(entry, {
      tokens,
      sequences: 2,
      tensorParallel: 4,
      precision: "nvfp4_g16_s1",
      indexerPrecision: "nvfp4_g16_s1",
    });
    assert.equal(batch.totalBytes, tokens * 2 * (60 * 4 * 2 * 72 + 60 * 72));
    assert.equal(batch.perDeviceBytes, batch.totalBytes / 4);
  }
});

test("Cohere vision profile discloses decoder-only cache scope", () => {
  const north = model("cohere-north-micro-vision-instruct");
  assert.match(north.cache_note, /text decoder only/);
  assert.equal(calculate(north, { tokens: 1024, precision: "bf16_fp16" }).kvBytes, 117440512);
});

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import test from "node:test";

import { loadModelsData } from "../scripts/lib/kv-cache-lab-traces.mjs";

const require = createRequire(import.meta.url);
const {
  calculate,
  expandModelCatalog,
  modelFamily,
  modelsForFamily,
} = require("../assets/js/kv-cache-calculator.js");

const canonicalModels = loadModelsData(
  new URL("../data/kv_cache_calculator/models.yaml", import.meta.url).pathname,
).models;
const catalog = JSON.parse(
  readFileSync(new URL("../data/kv_cache_calculator/model_catalog.json", import.meta.url), "utf8"),
);
const expandedModels = expandModelCatalog(canonicalModels, catalog);
const qwenCatalog = catalog.filter((entry) => entry.official_model_id.startsWith("Qwen/"));

function assertPinnedConfigSource(entry) {
  const source = new URL(entry.source_url);
  assert.equal(source.origin, "https://huggingface.co");
  const sourceConfigOwner = entry.config_source_id || entry.official_model_id;
  const rawPrefix = `/${sourceConfigOwner}/raw/`;
  assert.ok(source.pathname.startsWith(rawPrefix));
  assert.match(source.pathname.slice(rawPrefix.length), /^[a-f0-9]{40}\/(?:.*\/)?config\.json$/);
}

test("catalog aliases have unique official IDs, pinned sources, and valid presets", () => {
  assert.equal(new Set(catalog.map((entry) => entry.official_model_id)).size, catalog.length);
  assert.deepEqual(expandedModels.slice(0, canonicalModels.length), canonicalModels);
  assert.equal(expandedModels.length, canonicalModels.length + catalog.length);

  for (const entry of catalog) {
    const preset = canonicalModels.find((model) => model.id === entry.preset_id);
    const alias = expandedModels.find((model) => model.id === entry.official_model_id);
    assert.ok(preset, `Missing preset: ${entry.preset_id}`);
    assert.ok(alias, `Missing alias: ${entry.official_model_id}`);
    assert.equal(alias.label, entry.label ?? entry.official_model_id.split("/").pop());
    assert.equal(alias.source_url, entry.source_url);
    assert.equal(alias.official_release_date, entry.official_release_date);
    assert.equal(alias.weights_public_date, entry.weights_public_date);
    assertPinnedConfigSource(entry);
    const expectedNote = [...new Set([preset.cache_note, entry.cache_note].filter(Boolean))].join(" ");
    assert.equal(alias.cache_note || "", expectedNote);
    assert.equal(alias.formula, preset.formula);
    assert.equal(alias.max_position_embeddings, entry.max_position_embeddings ?? preset.max_position_embeddings);
    assert.deepEqual(alias.fields, { ...(preset.fields || {}), ...(entry.fields || {}) });
  }
  assert.ok(expandedModels.every((model) => !model.label.includes("/")), "Model labels should not contain Hub owner prefixes");
});

test("DeepSeek Vision-Exp alias keeps ordinary FP8 default and exposes the SGLang packed row layout", () => {
  const alias = expandedModels.find((model) => model.id === "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp");
  assert.equal(alias.label, "DeepSeek V4 Flash Vision Exp");
  assert.equal(alias.fields.default_precision_id, "fp8_int8");
  assert.equal(alias.fields.kv_entry_bytes_fp8_sglang_packed, 584);
  assert.match(alias.cache_note, /448 FP8 \+ 128 BF16 RoPE \+ 8 scale\/pad/);

  const ordinary = calculate(alias, { tokens: 1024, sequences: 1 });
  const packed = calculate(alias, {
    tokens: 1024,
    sequences: 1,
    precision: "fp8_sglang_packed",
    indexerPrecision: "fp4_int4_g32_s1",
  });
  assert.equal(ordinary.precisionLabel, "FP8 / INT8");
  assert.equal(ordinary.totalBytes, 6018048);
  assert.equal(packed.totalBytes, 6812928);
  assert.equal(packed.cacheGroups.find((group) => group.role === "kv").bytes, 6447360);
});

test("pinned config sources support a subfolder and a separate config owner", () => {
  assertPinnedConfigSource({
    official_model_id: "Qwen/Example-GGUF",
    config_source_id: "Qwen/Example-Base",
    source_url: "https://huggingface.co/Qwen/Example-Base/raw/0123456789abcdef0123456789abcdef01234567/text_config/config.json",
  });
});

test("Qwen catalog keeps canonical models and exposes every verified Qwen repository", () => {
  assert.equal(qwenCatalog.length, 3);
  for (const entry of qwenCatalog) {
    const alias = expandedModels.find((model) => model.id === entry.official_model_id);
    const preset = canonicalModels.find((model) => model.id === entry.preset_id);
    assert.equal(modelFamily(alias), modelFamily(preset));
  }
  assert.deepEqual(modelsForFamily(expandedModels, "Qwen").slice(0, 5).map((model) => model.id), [
    "qwen-image-2.1-pe-decoder",
    "Qwen/Qwen-Image-2.1-PE-I2I",
    "Qwen/Qwen-Drive-1.0-4B",
    "qwen3.8-flash-next",
    "qwen3.8-27b",
  ]);
});

test("selecting each Qwen repository uses its mapped preset cache geometry", () => {
  const input = {
    tokens: 257,
    sequences: 2,
    precision: "bf16_fp16",
    indexerPrecision: "bf16_fp16",
    includeLinearAttentionState: true,
    includeDraftKvCache: false,
  };

  for (const entry of qwenCatalog) {
    const alias = expandedModels.find((model) => model.id === entry.official_model_id);
    const preset = canonicalModels.find((model) => model.id === entry.preset_id);
    const familyModels = modelsForFamily(expandedModels, modelFamily(alias));
    assert.ok(familyModels.some((model) => model.id === entry.official_model_id), `Could not select ${entry.official_model_id} in ${modelFamily(alias)} family`);
    const result = calculate(alias, input);
    const expected = calculate(preset, input);
    assert.equal(result.totalBytes, expected.totalBytes, entry.official_model_id);
    assert.equal(alias.source_url, entry.source_url);
  }
});

test("accepted post-cutoff Qwen aliases map to their verified decoder profiles", () => {
  const expected = new Map([
    ["Qwen/Qwen-AgentWorld-35B-A3B", "qwen3.5-35b-a3b"],
    ["Qwen/Qwen-Image-2.1-PE-I2I", "qwen-image-2.1-pe-decoder"],
    ["Qwen/Qwen-Drive-1.0-4B", "qwen3.5-4b"],
  ]);

  assert.deepEqual(
    qwenCatalog.map((entry) => entry.official_model_id).sort(),
    [...expected.keys()].sort(),
  );
  for (const [officialModelId, presetId] of expected) {
    assert.equal(qwenCatalog.find((entry) => entry.official_model_id === officialModelId)?.preset_id, presetId);
  }

  const driveEntry = qwenCatalog.find((entry) => entry.official_model_id === "Qwen/Qwen-Drive-1.0-4B");
  const drive = expandedModels.find((model) => model.id === driveEntry.official_model_id);
  assert.equal(drive.max_position_embeddings, 32768);
  assert.match(drive.cache_note, /excludes vision-encoder state, the separate trajectory-planning expert\/cache, and perception modules/i);

  assert.equal(qwenCatalog.some((entry) => entry.official_model_id === "Qwen/Qwen-Image-Bench"), false);
});

test("Inkling SConv component text follows the configured state precision", () => {
  for (const [modelId, bytesPerElement, precisionName] of [
    ["inkling", 2, "BF16"],
    ["inkling-2-small", 4, "FP32"],
  ]) {
    const model = canonicalModels.find((item) => item.id === modelId);
    assert.equal(model.fields.sconv_state_bytes_per_element, bytesPerElement);
    const result = calculate(model, {
      tokens: 64,
      precision: "bf16_fp16",
      includeSconvState: true,
    });
    const row = result.elementPlan.formulaRows.find((item) => item.name === "sconv_checkpoint_bytes");
    const bytesComponent = result.elementPlan.components.find(([label]) => label === "SConv bytes per checkpoint");
    const stateComponent = result.elementPlan.components.find(([label]) => label === "SConv state bytes");

    assert.match(row.description, new RegExp(`Logical ${precisionName}`));
    assert.match(result.elementPlan.note, new RegExp(`logical ${precisionName} SConv checkpoints`));
    assert.match(bytesComponent[2], new RegExp(`logical ${precisionName} short-convolution checkpoint`));
    assert.match(stateComponent[2], new RegExp(`retain this state in ${precisionName}`));
  }
});

test("Qwen-Drive calculation note states decoder context and planner exclusions", () => {
  const drive = expandedModels.find((model) => model.id === "Qwen/Qwen-Drive-1.0-4B");
  const result = calculate(drive, {
    tokens: 64,
    sequences: 1,
    precision: "bf16_fp16",
    recurrentStatePrecision: "fp32",
    includeLinearAttentionState: true,
  });

  assert.equal(drive.max_position_embeddings, 32768);
  assert.equal(result.precisionLabel, "BF16 / FP16");
  assert.equal(result.recurrentStatePrecisionLabel, "FP32");
  assert.equal(result.totalBytes, 53608448);
  assert.match(result.elementPlan.note, /32768-token context/);
  assert.match(result.elementPlan.note, /trajectory-planning expert\/cache/);
  assert.match(result.elementPlan.note, /perception modules/);
});

test("selecting every family alias reproduces its canonical cache geometry", () => {
  const input = {
    tokens: 257,
    sequences: 2,
    precision: "bf16_fp16",
    indexerPrecision: "bf16_fp16",
    includeLinearAttentionState: true,
    includeDraftKvCache: true,
  };

  for (const entry of catalog) {
    const alias = expandedModels.find((model) => model.id === entry.official_model_id);
    const preset = canonicalModels.find((model) => model.id === entry.preset_id);
    assert.equal(calculate(alias, input).totalBytes, calculate(preset, input).totalBytes, entry.official_model_id);
  }
});

test("catalog aliases preserve and extend required canonical cache notes", () => {
  const preset = {
    id: "kimi-shared-preset",
    family: "Kimi",
    cache_note: "Canonical TP placement caveat.",
  };
  const [alias] = expandModelCatalog([preset], [
    {
      official_model_id: "moonshotai/Kimi-Variant",
      preset_id: preset.id,
      source_url: "https://huggingface.co/moonshotai/Kimi-Variant/raw/0123456789abcdef0123456789abcdef01234567/config.json",
      cache_note: "Variant architecture uses the same cache fields.",
    },
  ]).slice(1);

  assert.equal(alias.cache_note, "Canonical TP placement caveat. Variant architecture uses the same cache fields.");
  assert.equal(preset.cache_note, "Canonical TP placement caveat.");

  const duplicateNote = expandModelCatalog([preset], [
    {
      official_model_id: "moonshotai/Kimi-Variant-Duplicate-Note",
      preset_id: preset.id,
      source_url: "https://huggingface.co/moonshotai/Kimi-Variant-Duplicate-Note/raw/0123456789abcdef0123456789abcdef01234567/config.json",
      cache_note: preset.cache_note,
    },
  ])[1];
  assert.equal(duplicateNote.cache_note, preset.cache_note);
});

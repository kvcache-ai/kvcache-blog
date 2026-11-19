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
  const entry = models.find((candidate) => candidate.id === id);
  assert.ok(entry, `Missing Qwen 2026 profile: ${id}`);
  return entry;
}

test("Image-2.1 PE uses its Qwen3.5-VL 9B hybrid text config with no MTP layer", () => {
  const entry = model("qwen-image-2.1-pe-decoder");
  assert.equal(entry.source_url, "https://huggingface.co/Qwen/Qwen-Image-2.1-PE-T2I/raw/f3ed7985c788ad75b3ab7223e0c4c51e2a43545b/config.json");
  assert.equal(entry.max_position_embeddings, 262144);
  assert.equal(entry.fields.num_hidden_layers, 32);
  assert.equal(entry.fields.full_attention_layers, 8);
  assert.equal(entry.fields.linear_attention_layers, 24);
  assert.equal(entry.fields.mtp_num_hidden_layers, 0);
  assert.match(entry.cache_note, /mtp_num_hidden_layers=null/);

  const result = calculate(entry, {
    tokens: 4,
    precision: "bf16_fp16",
    includeLinearAttentionState: true,
    recurrentStatePrecision: "fp32",
  });
  assert.equal(result.kvBytes, 4 * 8 * 2 * 4 * 256 * 2);
  assert.equal(result.cacheGroups.find((group) => group.role === "linear_state").bytes, 51511296);
});

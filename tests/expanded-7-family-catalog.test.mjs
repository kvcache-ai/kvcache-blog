import assert from "node:assert/strict";
import { createRequire } from "node:module";
import test from "node:test";

import { loadModelsData } from "../scripts/lib/kv-cache-lab-traces.mjs";

const require = createRequire(import.meta.url);
const { calculate } = require("../assets/js/kv-cache-calculator.js");
const canonical = loadModelsData(
  new URL("../data/kv_cache_calculator/models.yaml", import.meta.url).pathname,
).models;

test("MiMo V2.5 ASR counts only the cached global Qwen2 decoder", () => {
  const preset = canonical.find((entry) => entry.id === "mimo-v2.5-asr-text-decoder");
  assert.ok(preset);
  assert.equal(preset.source_url, "https://huggingface.co/XiaomiMiMo/MiMo-V2.5-ASR/raw/227ce4c5e64e356007ad948781f0e8b55a1beba0/config.json");
  assert.equal(preset.max_position_embeddings, 8192);
  assert.deepEqual(
    [preset.fields.num_hidden_layers, preset.fields.full_attention_layers, preset.fields.num_key_value_heads, preset.fields.head_dim],
    [36, 36, 8, 128],
  );
  assert.equal(preset.fields.local_layers, 16);
  assert.equal(preset.fields.input_local_layers, 6);
  assert.match(preset.cache_note, /use_cache/);
  assert.match(preset.cache_note, /local audio-token transformer KV/);

  const bytes = calculate(preset, { tokens: 64, sequences: 2, precision: "bf16_fp16" });
  assert.equal(bytes.kvBytes, 64 * 2 * 36 * 2 * 8 * 128 * 2);
});

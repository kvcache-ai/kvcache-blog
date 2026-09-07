import assert from "node:assert/strict";
import { createRequire } from "node:module";
import test from "node:test";

const require = createRequire(import.meta.url);
const { calculate, calculateElementsPerSequence, formatBytes, modelFamily, modelsForFamily } = require("../assets/js/kv-cache-calculator.js");

const bf16 = { precision: "bf16_fp16", indexerPrecision: "bf16_fp16", sequences: 1, tensorParallel: 1 };

const dots3Note = {
  id: "dots3-note-prev",
  label: "dots3-note Preview",
  formula: "dots3_note_hybrid",
  fields: {
    num_hidden_layers: 46,
    full_attention_layers: 13,
    sliding_attention_layers: 33,
    kv_lora_rank: 512,
    qk_rope_head_dim: 64,
    swa_kv_lora_rank: 1024,
    swa_qk_rope_head_dim: 64,
    sliding_window: 512,
    sliding_window_size: 513,
    index_head_dim: 128,
    index_n_heads: 64,
    indexer_scale_bytes: 4,
    indexer_fixed_precision_id: "fp8_int8",
    default_precision_id: "bf16_fp16",
  },
};

test("Dots3 Note hybrid formula includes full MLA, capped SWA MLA, and FP8 indexer scales", () => {
  const result = calculate(dots3Note, { tokens: 1024, sequences: 1 });

  assert.equal(result.precisionLabel, "BF16 / FP16");
  assert.equal(result.indexerPrecisionLabel, "FP8 / INT8");
  assert.equal(result.totalBytes, 53858304);
  assert.equal(result.kvBytes, 52101120);
  assert.equal(result.indexerBytes, 1757184);
  assert.equal(result.bytesPerToken, 52596);
  assert.equal(
    result.elementPlan.components.find(([label]) => label === "Retained sliding-window tokens")[1],
    512,
  );
  assert.equal(
    result.cacheGroups.find((group) => group.label === "Indexer FP32 scale cache").bytes,
    53248,
  );
  assert.match(result.elementPlan.formulaText, /indexer_scale_bytes/);
  assert.match(result.elementPlan.note, /sliding_window_size=513 includes the current token/);
});

test("Dots3 Note grows linearly before the SWA window fills", () => {
  const result = calculate(dots3Note, { tokens: 256, sequences: 2 });

  assert.equal(result.totalBytes, 45312000);
  assert.equal(result.bytesPerSequence, 22656000);
  assert.equal(result.bytesPerToken, 88500);
  assert.equal(
    result.elementPlan.components.find(([label]) => label === "Retained sliding-window tokens")[1],
    256,
  );
});

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

const kimiK3 = {
  id: "kimi-k3",
  label: "Kimi K3",
  formula: "kimi_kda_mla_hybrid",
  fields: {
    num_hidden_layers: 93,
    full_attention_layers: 24,
    kda_layers: 69,
    default_precision_id: "bf16_fp16",
    kv_lora_rank: 512,
    qk_rope_head_dim: 64,
    kda_num_heads: 96,
    kda_head_dim: 128,
    kda_num_key_heads: 96,
    kda_key_head_dim: 128,
    kda_num_value_heads: 96,
    kda_value_head_dim: 128,
    kda_conv_kernel_size: 4,
    kda_conv_state_bytes_per_element: 2,
    kda_recurrent_state_bytes_per_element: 4,
    default_kda_checkpoint_interval: "infinity",
  },
};

test("Kimi K3 defaults to BF16 KV cache", () => {
  const result = calculate(kimiK3, {
    tokens: 1,
    sequences: 1,
    includeLinearAttentionState: false,
  });

  assert.equal(result.precisionLabel, "BF16 / FP16");
  assert.equal(result.totalBytes, 27648);
});

test("Kimi K3 counts the FP8 MLA latent payload", () => {
  const result = calculate(kimiK3, {
    tokens: 1048576,
    sequences: 1,
    precision: "fp8_int8",
  });

  assert.equal(result.tensorParallel, 1);
  assert.equal(result.elementPlan.elementsPerToken, 24 * (512 + 64));
  assert.equal(result.bytesPerToken, 13824);
  assert.equal(result.hitRateBytesPerToken, 13824);
  assert.equal(result.totalGiB, 13.5);
  assert.equal(
    result.cacheGroups.find((group) =>
      group.label.startsWith("MLA latent KV cache"),
    ).bytes,
    14495514624,
  );
  assert.equal(
    result.elementPlan.components.find(([label]) => label === "KDA state included")[1],
    "No",
  );
  assert.match(result.elementPlan.note, /69 KDA layers/);
});

test("Kimi K3 defaults to one checkpoint at an infinite interval", () => {
  const result = calculate(kimiK3, {
    tokens: 1048576,
    sequences: 1,
    precision: "fp8_int8",
    includeLinearAttentionState: true,
  });
  const convBytes = 69 * (4 - 1) * (3 * 96 * 128) * 2;
  const recurrentBytes = 69 * 96 * 128 * 128 * 4;
  const state = result.cacheGroups.find(
    (group) => group.label.startsWith("KDA checkpoint state"),
  );

  assert.equal(convBytes, 15261696);
  assert.equal(recurrentBytes, 434110464);
  assert.equal(
    result.elementPlan.components.find(
      ([label]) => label === "KDA checkpoints per sequence",
    )[1],
    1,
  );
  assert.equal(
    result.elementPlan.components.find(
      ([label]) => label === "KDA checkpoint interval",
    )[1],
    "∞",
  );
  assert.equal(state.bytes, convBytes + recurrentBytes);
  assert.equal(
    result.elementPlan.components.find(
      ([label]) => label === "KDA checkpoint bytes per sequence",
    )[1],
    convBytes + recurrentBytes,
  );
  assert.equal(
    result.elementPlan.components.find(([label]) => label === "KDA state included")[1],
    "Yes",
  );
  assert.match(
    result.elementPlan.formulaRows.find(
      (row) => row.name === "kda_recurrent_state_bytes",
    ).expression,
    /kda_checkpoint_count/,
  );
});

test("Kimi K3 stores a final checkpoint for a partial interval", () => {
  const result = calculate(kimiK3, {
    tokens: 1025,
    sequences: 1,
    precision: "fp8_int8",
    includeLinearAttentionState: true,
    kdaCheckpointPolicy: "fixed_interval",
    kdaCheckpointInterval: 1024,
  });
  const checkpointBytes =
    69 * (4 - 1) * (3 * 96 * 128) * 2 +
    69 * 96 * 128 * 128 * 4;
  const state = result.cacheGroups.find(
    (group) => group.label.startsWith("KDA checkpoint state"),
  );

  assert.equal(
    result.elementPlan.components.find(
      ([label]) => label === "KDA checkpoints per sequence",
    )[1],
    2,
  );
  assert.equal(state.bytes, 2 * checkpointBytes);
  assert.match(
    result.elementPlan.formulaRows.find(
      (row) => row.name === "kda_checkpoint_count",
    ).expression,
    /ceil/,
  );
});

test("Kimi K3 checkpoint storage scales with intervals and sequences", () => {
  const sequences = 3;
  const result = calculate(kimiK3, {
    tokens: 4096,
    sequences,
    precision: "fp8_int8",
    includeLinearAttentionState: true,
    kdaCheckpointPolicy: "fixed_interval",
    kdaCheckpointInterval: 1024,
  });
  const checkpointBytes =
    69 * (4 - 1) * (3 * 96 * 128) * 2 +
    69 * 96 * 128 * 128 * 4;
  const state = result.cacheGroups.find(
    (group) => group.label.startsWith("KDA checkpoint state"),
  );

  assert.equal(
    result.elementPlan.components.find(
      ([label]) => label === "KDA checkpoints per sequence",
    )[1],
    4,
  );
  assert.equal(state.bytes, 4 * sequences * checkpointBytes);
});

test("Kimi K3 ignores checkpoint interval when linear state is excluded", () => {
  const result = calculate(kimiK3, {
    tokens: 4096,
    sequences: 2,
    precision: "fp8_int8",
    includeLinearAttentionState: false,
    kdaCheckpointPolicy: "fixed_interval",
    kdaCheckpointInterval: 1,
  });

  assert.equal(
    result.cacheGroups.some((group) =>
      group.label.startsWith("KDA checkpoint state"),
    ),
    false,
  );
  assert.equal(
    result.elementPlan.components.find(
      ([label]) => label === "KDA checkpoints per sequence",
    )[1],
    0,
  );
});

test("Kimi K3 prompt-end policy ignores a finite interval", () => {
  const result = calculate(kimiK3, {
    tokens: 4096,
    sequences: 1,
    precision: "fp8_int8",
    includeLinearAttentionState: true,
    kdaCheckpointPolicy: "prompt_end",
    kdaCheckpointInterval: 1,
  });

  assert.equal(
    result.elementPlan.components.find(
      ([label]) => label === "KDA checkpoint interval",
    )[1],
    "∞",
  );
  assert.equal(
    result.elementPlan.components.find(
      ([label]) => label === "KDA checkpoints per sequence",
    )[1],
    1,
  );
});

test("Kimi K3 treats a blank fixed interval as prompt-end state", () => {
  const result = calculate(kimiK3, {
    tokens: 4096,
    sequences: 1,
    precision: "fp8_int8",
    includeLinearAttentionState: true,
    kdaCheckpointPolicy: "fixed_interval",
    kdaCheckpointInterval: "",
  });

  assert.equal(
    result.elementPlan.components.find(
      ([label]) => label === "KDA checkpoint interval",
    )[1],
    "∞",
  );
  assert.equal(
    result.elementPlan.components.find(
      ([label]) => label === "KDA checkpoints per sequence",
    )[1],
    1,
  );
});

test("Kimi K3 MLA cache scales from the exact logical token count", () => {
  const fp8Result = calculate(kimiK3, {
    tokens: 1,
    sequences: 1,
    precision: "fp8_int8",
    includeLinearAttentionState: false,
  });
  const bf16Result = calculate(kimiK3, {
    tokens: 1,
    sequences: 1,
    precision: "bf16_fp16",
    includeLinearAttentionState: false,
  });

  assert.equal(fp8Result.totalBytes, 13824);
  assert.equal(fp8Result.totalGiB * 1024 ** 2, 13.5);
  assert.equal(fp8Result.hitRateBytesPerToken, 13824);
  assert.equal(bf16Result.totalBytes, 27648);
  assert.equal(bf16Result.totalGiB * 1024 ** 2, 27);
  assert.equal(bf16Result.hitRateBytesPerToken, 27648);
});

const glm53Flash = {
  id: "glm-5.3-flash",
  label: "GLM-5.3-Flash",
  formula: "kimi_kda_dsa_mla_hybrid",
  fields: {
    num_hidden_layers: 45,
    full_attention_layers: 11,
    kda_layers: 34,
    indexer_full_layers: 45,
    indexer_shared_layers: 0,
    default_precision_id: "bf16_fp16",
    kv_lora_rank: 512,
    qk_rope_head_dim: 0,
    index_head_dim: 128,
    num_nextn_predict_layers: 1,
    draft_indexer_layers: 1,
    kda_num_heads: 64,
    kda_head_dim: 128,
    kda_num_key_heads: 64,
    kda_key_head_dim: 128,
    kda_num_value_heads: 64,
    kda_value_head_dim: 128,
    kda_conv_kernel_size: 4,
    kda_conv_state_bytes_per_element: 4,
    kda_recurrent_state_bytes_per_element: 2,
    default_kda_checkpoint_interval: "infinity",
  },
};

// Keep the unpooled 45-layer fixture above as generic backward compatibility coverage.
const glm53Pooled = {
  ...glm53Flash,
  fields: {
    ...glm53Flash.fields,
    indexer_full_layers: 11,
    index_kpool: 4,
  },
};

const glm53Quantized = {
  ...glm53Pooled,
  fields: {
    ...glm53Pooled.fields,
    default_indexer_precision_id: "fp8_int8",
    indexer_fp8_quant_block_size: 128,
    indexer_fp8_scale_bytes_per_element: 4,
    indexer_mxfp4_quant_block_size: 32,
    indexer_mxfp4_scale_bytes_per_element: 1,
  },
};

test("GLM indexer scales use fixed widths for FP8 and FP4, with no BF16 scales", () => {
  for (const [indexerPrecision, width, scales, scaleWidth] of [
    [undefined, 1, 1, 4],
    ["fp8_int8", 1, 1, 4],
    ["fp4_int4", 0.5, 4, 1],
    ["bf16_fp16", 2, 0, 0],
  ]) {
    for (const precision of ["bf16_fp16", "fp8_int8", "fp4_int4"]) {
      for (const tokens of [1, 3, 4, 5, 8, 9]) {
        for (const includeDraftKvCache of [false, true]) {
          const sequences = 3;
          const layers = includeDraftKvCache ? 12 : 11;
          const vectors = Math.floor(tokens / 4) * layers * sequences;
          const result = calculate(glm53Quantized, {
            tokens, sequences, precision, indexerPrecision, includeDraftKvCache,
          });
          const scale = result.cacheGroups.find((group) => group.label === "Indexer quantization scale cache");
          const scaleBytes = vectors * scales * scaleWidth;
          assert.equal(scale?.bytes ?? 0, scaleBytes);
          assert.equal(scale?.elements ?? 0, vectors * scales);
          assert.equal(result.indexerBytes, vectors * 128 * width + scaleBytes);
          const tailBytes = tokens % 4 ? layers * 4 * 2 * 128 * 2 * sequences : 0;
          assert.equal(result.totalBytes, result.kvBytes + result.indexerBytes + tailBytes);
          assert.match(result.elementPlan.formulaText, /indexer_scale_bytes/);
          assert.match(result.elementPlan.formulaRows.find((row) => row.name === "total_bytes").expression, /indexer_scale_bytes/);
        }
      }
    }
  }
});

test("GLM scale block counts floor per vector and honor configured scale widths", () => {
  const model = {
    ...glm53Quantized,
    fields: {
      ...glm53Quantized.fields,
      index_head_dim: 150,
      indexer_fp8_quant_block_size: 64,
      indexer_fp8_scale_bytes_per_element: 8,
      indexer_mxfp4_scale_bytes_per_element: 1,
    },
  };
  for (const [indexerPrecision, scaleBytes] of [["fp8_int8", 16], ["fp4_int4", 4]]) {
    const result = calculate(model, { tokens: 8, indexerPrecision });
    assert.equal(result.cacheGroups.find((group) => group.label === "Indexer quantization scale cache").bytes, 2 * 11 * scaleBytes);
  }
});

test("GLM scales follow resolved indexer precision for invalid and fixed selections", () => {
  const input = { tokens: 8, indexerPrecision: "invalid" };
  const result = calculate(glm53Quantized, input);
  assert.equal(result.indexerBytes, 2 * 11 * (128 + 4));
  const model = {
    ...glm53Quantized,
    fields: { ...glm53Quantized.fields, indexer_fixed_precision_id: "fp4_int4" },
  };
  const fixed = calculate(model, { tokens: 8, indexerPrecision: "fp8_int8" });
  assert.equal(fixed.indexerBytes, 2 * 11 * (64 + 4));
  const plan = calculateElementsPerSequence(glm53Quantized, 8);
  assert.equal(plan.byteGroups.find((group) => group.label === "Indexer quantization scale cache").bytesPerSequence, 2 * 11 * 4);
});

const glm53CheckpointBytes =
  34 * (4 - 1) * (3 * 64 * 128) * 4 +
  34 * 64 * 128 * 128 * 2;

test("GLM-5.3-Flash counts 11 MLA layers and 45 independent indexer layers without state", () => {
  const tokens = 1024;
  const result = calculate(glm53Flash, { ...bf16, tokens });

  assert.equal(result.kvBytes, tokens * 11 * 512 * 2);
  assert.equal(result.indexerBytes, tokens * 45 * 128 * 2);
  assert.equal(result.totalBytes, tokens * (11 * 512 + 45 * 128) * 2);
  assert.equal(result.bytesPerToken, (11 * 512 + 45 * 128) * 2);
  assert.equal(result.hitRateBytesPerToken, 11 * 512 * 2);
  assert.equal(result.elementPlan.byteGroups.find((group) => group.role === "kv").elements, tokens * 11 * 512);
  assert.equal(result.elementPlan.byteGroups.find((group) => group.role === "indexer").elements, tokens * 45 * 128);
  assert.equal(result.elementPlan.components.find(([label]) => label === "KDA state included")[1], "No");
  assert.equal(result.cacheGroups.some((group) => group.label.startsWith("KDA checkpoint state")), false);
});

test("GLM-5.3-Flash includes one configured KDA checkpoint only when explicitly enabled", () => {
  const input = { ...bf16, tokens: 4096 };
  const withoutState = calculate(glm53Flash, input);
  const result = calculate(glm53Flash, { ...input, includeLinearAttentionState: true });

  assert.equal(glm53CheckpointBytes, 81330176);
  assert.equal(result.cacheGroups.find((group) => group.label.startsWith("KDA checkpoint state")).bytes, glm53CheckpointBytes);
  assert.equal(result.totalBytes - withoutState.totalBytes, glm53CheckpointBytes);
  assert.equal(result.kvBytes, withoutState.kvBytes);
  assert.equal(result.indexerBytes, withoutState.indexerBytes);
  assert.equal(result.hitRateBytesPerToken, 11 * 512 * 2);
  assert.equal(result.elementPlan.components.find(([label]) => label === "KDA checkpoints per sequence")[1], 1);
});

test("GLM-5.3-Flash scales KV and indexer precision independently without scaling fixed state", () => {
  for (const [precision, kvBytesPerElement] of [["bf16_fp16", 2], ["fp8_int8", 1]]) {
    for (const [indexerPrecision, indexerBytesPerElement] of [["bf16_fp16", 2], ["fp4_int4", 0.5]]) {
      const tokens = 1025;
      const result = calculate(glm53Flash, {
        ...bf16, tokens, precision, indexerPrecision,
        includeLinearAttentionState: true,
        kdaCheckpointPolicy: "fixed_interval",
        kdaCheckpointInterval: 1024,
      });
      const kvBytes = tokens * 11 * 512 * kvBytesPerElement;
      const indexerBytes = tokens * 45 * 128 * indexerBytesPerElement;

      assert.equal(result.kvBytes, kvBytes);
      assert.equal(result.indexerBytes, indexerBytes);
      assert.equal(result.cacheGroups.find((group) => group.label.startsWith("KDA checkpoint state")).bytes, 2 * glm53CheckpointBytes);
      assert.equal(result.totalBytes, kvBytes + indexerBytes + 2 * glm53CheckpointBytes);
      assert.equal(result.hitRateBytesPerToken, 11 * 512 * kvBytesPerElement);
    }
  }
});

test("GLM-5.3-Flash draft adds one MLA and one indexer layer but no KDA state", () => {
  const input = { ...bf16, tokens: 1024, includeLinearAttentionState: true };
  const withoutDraft = calculate(glm53Flash, { ...input, includeDraftKvCache: false });
  const result = calculate(glm53Flash, { ...input, includeDraftKvCache: true });

  assert.equal(result.kvBytes, 1024 * 12 * 512 * 2);
  assert.equal(result.indexerBytes, 1024 * 46 * 128 * 2);
  assert.equal(result.totalBytes - withoutDraft.totalBytes, 1024 * (512 + 128) * 2);
  assert.equal(result.hitRateBytesPerToken, 12 * 512 * 2);
  assert.equal(result.cacheGroups.find((group) => group.label.startsWith("KDA checkpoint state")).bytes, glm53CheckpointBytes);
  assert.equal(result.elementPlan.components.find(([label]) => label === "Draft layers included")[1], 1);
  assert.equal(result.elementPlan.components.find(([label]) => label === "Draft indexer layers included")[1], 1);
});

test("GLM-5.3-Flash disable_draft_kv_cache suppresses both draft cache groups", () => {
  const model = { ...glm53Flash, fields: { ...glm53Flash.fields, disable_draft_kv_cache: true } };
  const input = { ...bf16, tokens: 1024, includeLinearAttentionState: true };
  const baseline = calculate(model, { ...input, includeDraftKvCache: false });
  const result = calculate(model, { ...input, includeDraftKvCache: true });

  assert.equal(result.kvBytes, baseline.kvBytes);
  assert.equal(result.indexerBytes, baseline.indexerBytes);
  assert.equal(result.totalBytes, baseline.totalBytes);
  assert.equal(result.elementPlan.components.find(([label]) => label === "Draft layers included")[1], 0);
  assert.equal(result.elementPlan.components.find(([label]) => label === "Draft indexer layers included")[1], 0);
});

test("GLM-5.3-Flash prompt-end and unspecified policies ignore a finite checkpoint interval", () => {
  for (const policy of [undefined, "prompt_end"]) {
    const result = calculate(glm53Flash, {
      ...bf16, tokens: 4096, includeLinearAttentionState: true,
      kdaCheckpointPolicy: policy, kdaCheckpointInterval: 1,
    });

    assert.equal(result.elementPlan.components.find(([label]) => label === "KDA checkpoints per sequence")[1], 1);
    assert.equal(result.elementPlan.components.find(([label]) => label === "KDA checkpoint interval")[1], "∞");
    assert.equal(result.cacheGroups.find((group) => group.label.startsWith("KDA checkpoint state")).bytes, glm53CheckpointBytes);
  }
});

test("GLM-5.3-Flash fixed checkpoints handle exact and partial interval boundaries", () => {
  for (const [tokens, checkpoints] of [[1, 1], [1023, 1], [1024, 1], [1025, 2], [2048, 2], [2049, 3]]) {
    const result = calculate(glm53Flash, {
      ...bf16, tokens, includeLinearAttentionState: true,
      kdaCheckpointPolicy: "fixed_interval", kdaCheckpointInterval: 1024,
    });

    assert.equal(result.elementPlan.components.find(([label]) => label === "KDA checkpoints per sequence")[1], checkpoints, `tokens=${tokens}`);
    assert.equal(result.cacheGroups.find((group) => group.label.startsWith("KDA checkpoint state")).bytes, checkpoints * glm53CheckpointBytes);
    assert.equal(result.totalBytes, tokens * (11 * 512 + 45 * 128) * 2 + checkpoints * glm53CheckpointBytes);
  }
});

test("GLM-5.3-Flash excluded state ignores fixed checkpoint policy", () => {
  const result = calculate(glm53Flash, {
    ...bf16, tokens: 4096, includeLinearAttentionState: false,
    kdaCheckpointPolicy: "fixed_interval", kdaCheckpointInterval: 1,
  });

  assert.equal(result.totalBytes, 4096 * (11 * 512 + 45 * 128) * 2);
  assert.equal(result.elementPlan.components.find(([label]) => label === "KDA checkpoints per sequence")[1], 0);
  assert.equal(result.cacheGroups.some((group) => group.label.startsWith("KDA checkpoint state")), false);
});

test("GLM-5.3-Flash sequence doubling scales all cache groups but not per-token hit-rate bytes", () => {
  for (const includeLinearAttentionState of [false, true]) {
    const input = {
      ...bf16, tokens: 1025, includeLinearAttentionState,
      kdaCheckpointPolicy: "fixed_interval", kdaCheckpointInterval: 1024,
    };
    const single = calculate(glm53Flash, input);
    const doubled = calculate(glm53Flash, { ...input, sequences: 2 });

    assert.equal(doubled.totalBytes, 2 * single.totalBytes);
    assert.equal(doubled.kvBytes, 2 * single.kvBytes);
    assert.equal(doubled.indexerBytes, 2 * single.indexerBytes);
    assert.equal(doubled.bytesPerSequence, single.bytesPerSequence);
    assert.equal(doubled.hitRateBytesPerToken, single.hitRateBytesPerToken);
    for (const group of single.cacheGroups) {
      assert.equal(doubled.cacheGroups.find((candidate) => candidate.label === group.label).bytes, 2 * group.bytes);
    }
  }
});

test("GLM-5.3-Flash token doubling leaves prompt-end state fixed", () => {
  const input = { ...bf16, tokens: 1024, includeLinearAttentionState: true };
  const single = calculate(glm53Flash, input);
  const doubled = calculate(glm53Flash, { ...input, tokens: 2048 });

  assert.equal(doubled.kvBytes, 2 * single.kvBytes);
  assert.equal(doubled.indexerBytes, 2 * single.indexerBytes);
  assert.equal(doubled.totalBytes, 2 * single.totalBytes - glm53CheckpointBytes);
});

test("GLM-5.3-Flash shared indexer layers do not allocate duplicate indexer cache", () => {
  const model = {
    ...glm53Flash,
    fields: { ...glm53Flash.fields, indexer_full_layers: 9, indexer_shared_layers: 36 },
  };
  const result = calculate(model, { ...bf16, tokens: 1024 });

  assert.equal(result.kvBytes, 1024 * 11 * 512 * 2);
  assert.equal(result.indexerBytes, 1024 * 9 * 128 * 2);
  assert.equal(result.elementPlan.components.find(([label]) => label === "Main indexer layers")[1], 9);
  assert.equal(result.elementPlan.components.find(([label]) => label === "Shared indexer layers")[1], 36);
});

test("GLM-5.3-Flash respects draft_indexer_layers independently of MLA draft layers", () => {
  for (const draftIndexerLayers of [0, 2]) {
    const model = {
      ...glm53Flash,
      fields: {
        ...glm53Flash.fields, indexer_full_layers: 9, indexer_shared_layers: 36,
        draft_indexer_layers: draftIndexerLayers,
      },
    };
    const result = calculate(model, { ...bf16, tokens: 1024, includeDraftKvCache: true });

    assert.equal(result.kvBytes, 1024 * 12 * 512 * 2);
    assert.equal(result.indexerBytes, 1024 * (9 + draftIndexerLayers) * 128 * 2);
    assert.equal(result.elementPlan.components.find(([label]) => label === "Draft indexer layers included")[1], draftIndexerLayers);
  }
});

test("GLM-5.3-Flash defaults missing indexer_full_layers to full_attention_layers, not total layers", () => {
  const fields = { ...glm53Flash.fields };
  delete fields.indexer_full_layers;
  const model = { ...glm53Flash, fields };

  for (const includeDraftKvCache of [false, true]) {
    const result = calculate(model, { ...bf16, tokens: 1024, includeDraftKvCache });
    const activeLayers = includeDraftKvCache ? 12 : 11;

    assert.equal(result.kvBytes, 1024 * activeLayers * 512 * 2);
    assert.equal(result.indexerBytes, 1024 * activeLayers * 128 * 2);
    assert.equal(result.elementPlan.components.find(([label]) => label === "Main indexer layers")[1], 11);
  }
});

test("GLM-5.3-Flash formula rows expose indexer precision and configurable checkpoint state", () => {
  const result = calculate(glm53Flash, {
    ...bf16, tokens: 1025, includeLinearAttentionState: true,
    kdaCheckpointPolicy: "fixed_interval", kdaCheckpointInterval: 1024,
  });
  const rows = result.elementPlan.formulaRows;

  assert.match(rows.find((row) => row.name === "indexer_bytes").expression, /indexer_precision_bytes/);
  assert.match(rows.find((row) => row.name === "total_bytes").expression, /indexer_bytes/);
  assert.match(rows.find((row) => row.name === "total_bytes").expression, /kda_recurrent_state_bytes/);
  assert.match(rows.find((row) => row.name === "kda_checkpoint_count").expression, /ceil/);
  assert.match(rows.find((row) => row.name === "kda_conv_state_bytes").expression, /kda_conv_state_bytes_per_element/);
  assert.match(rows.find((row) => row.name === "kda_recurrent_state_bytes").expression, /kda_recurrent_state_bytes_per_element/);
});

test("GLM-5.3-Flash descriptions do not inherit Kimi layer counts or reversed state precisions", () => {
  for (const includeLinearAttentionState of [false, true]) {
    const result = calculate(glm53Flash, { ...bf16, tokens: 1024, includeLinearAttentionState });
    const description = [
      result.elementPlan.note,
      result.elementPlan.formulaText,
      ...result.elementPlan.formulaRows.map((row) => `${row.name} ${row.expression}`),
      ...result.cacheGroups.map((group) => group.label),
    ].join("\n");

    assert.doesNotMatch(description, /\bKimi\b|\b(?:24|69)\b/i);
    assert.doesNotMatch(description, /BF16[^.\n;]*conv|conv[^.\n;]*BF16|FP32[^.\n;]*recurrent|recurrent[^.\n;]*FP32/i);
  }
});

test("GLM pooled indexer counts in-progress tail without a selection flag", () => {
  for (const [tokens, groups] of [[1, 0], [2, 0], [3, 0], [4, 1], [5, 1], [7, 1], [8, 2], [9, 2]]) {
    const result = calculate(glm53Pooled, { ...bf16, tokens });
    const hasTail = tokens % 4 !== 0;
    const tailElements = hasTail ? 11 * 4 * 2 * 128 : 0;
    const tail = result.elementPlan.byteGroups.find((group) => group.role === "index_tail");

    assert.equal(result.elementPlan.byteGroups.find((group) => group.role === "indexer").elements, groups * 11 * 128, `tokens=${tokens}`);
    if (hasTail) {
      assert.equal(tail.label, "Index tail cache");
      assert.equal(tail.elements, tailElements, `in-progress pool at tokens=${tokens}`);
      assert.equal(result.cacheGroups.find((group) => group.role === "index_tail").bytes, tailElements * 2);
    } else {
      assert.equal(tail, undefined);
      assert.match(result.elementPlan.formulaRows.find((row) => row.name === "index_tail_bytes").expression, /^0/);
    }
    assert.equal(result.cacheGroups.filter((group) => group.role === "index_tail").length, hasTail ? 1 : 0);
    assert.equal(result.kvBytes, tokens * 11 * 512 * 2);
    assert.equal(result.indexerBytes, groups * 11 * 128 * 2);
    assert.equal(result.totalBytes, result.kvBytes + result.indexerBytes + tailElements * 2);
    assert.equal(result.hitRateBytesPerToken, 11 * 512 * 2);
  }
});

test("GLM missing pool defaults to one and complete pools allocate no tail", () => {
  for (const pool of [undefined, 1, 4]) {
    const fields = { ...glm53Flash.fields };
    if (pool !== undefined) fields.index_kpool = pool;
    const result = calculate({ ...glm53Flash, fields }, { ...bf16, tokens: 8 });

    assert.equal(result.indexerBytes, Math.floor(8 / (pool ?? 1)) * 45 * 128 * 2);
    assert.equal(result.totalBytes, result.kvBytes + result.indexerBytes);
    assert.equal(result.cacheGroups.some((group) => group.role === "index_tail"), false);
  }
  const fields = { ...glm53Pooled.fields };
  delete fields.index_kpool;
  const result = calculate({ ...glm53Pooled, fields }, { ...bf16, tokens: 3 });
  assert.equal(result.indexerBytes, 3 * 11 * 128 * 2);
  assert.equal(result.cacheGroups.some((group) => group.role === "index_tail"), false);
  assert.equal(result.totalBytes, result.kvBytes + result.indexerBytes);
});

test("GLM tail width defaults to BF16 independently of KV and indexer precision and is configurable", () => {
  for (const tailWidth of [undefined, 1, 4]) {
    const fields = { ...glm53Pooled.fields };
    if (tailWidth !== undefined) fields.index_tail_bytes_per_element = tailWidth;
    for (const [precision, kvWidth] of [["bf16_fp16", 2], ["fp8_int8", 1]]) {
      for (const [indexerPrecision, indexerWidth] of [["bf16_fp16", 2], ["fp4_int4", 0.5]]) {
        const result = calculate({ ...glm53Pooled, fields }, {
          ...bf16, tokens: 9, precision, indexerPrecision, includeLinearAttentionState: true,
        });
        const tailBytes = 11 * 4 * 2 * 128 * (tailWidth ?? 2);

        assert.equal(result.kvBytes, 9 * 11 * 512 * kvWidth);
        assert.equal(result.indexerBytes, 2 * 11 * 128 * indexerWidth);
        assert.equal(result.cacheGroups.find((group) => group.role === "index_tail").bytes, tailBytes);
        assert.equal(result.cacheGroups.find((group) => group.label.startsWith("KDA checkpoint state")).bytes, glm53CheckpointBytes);
        assert.equal(result.totalBytes, result.kvBytes + result.indexerBytes + tailBytes + glm53CheckpointBytes);
        assert.equal(result.hitRateBytesPerToken, 11 * 512 * kvWidth);
      }
    }
  }
});

test("GLM pooled indexer and tail share active draft layers and sequence scaling, not shared layers", () => {
  for (const draftIndexerLayers of [0, 1, 2]) {
    for (const disableDraft of [false, true]) {
      const model = {
        ...glm53Pooled,
        fields: {
          ...glm53Pooled.fields, indexer_full_layers: 9, indexer_shared_layers: 36,
          draft_indexer_layers: draftIndexerLayers, disable_draft_kv_cache: disableDraft,
        },
      };
      for (const includeDraftKvCache of [false, true]) {
        const activeIndexerLayers = 9 + (includeDraftKvCache && !disableDraft ? draftIndexerLayers : 0);
        const activeMlaLayers = 11 + (includeDraftKvCache && !disableDraft ? 1 : 0);
        const input = { ...bf16, tokens: 9, includeDraftKvCache, includeLinearAttentionState: true };
        const single = calculate(model, input);
        const doubled = calculate(model, { ...input, sequences: 2 });

        assert.equal(single.indexerBytes, 2 * activeIndexerLayers * 128 * 2);
        assert.equal(single.cacheGroups.find((group) => group.role === "index_tail").bytes, activeIndexerLayers * 4 * 2 * 128 * 2);
        assert.equal(single.kvBytes, 9 * activeMlaLayers * 512 * 2);
        assert.equal(single.totalBytes, single.kvBytes + single.indexerBytes + activeIndexerLayers * 4 * 2 * 128 * 2 + glm53CheckpointBytes);
        assert.equal(doubled.totalBytes, 2 * single.totalBytes);
        assert.equal(doubled.bytesPerSequence, single.bytesPerSequence);
        assert.equal(single.hitRateBytesPerToken, activeMlaLayers * 512 * 2);
        assert.equal(doubled.hitRateBytesPerToken, single.hitRateBytesPerToken);
        for (const group of single.cacheGroups) {
          assert.equal(doubled.cacheGroups.find((candidate) => candidate.label === group.label).bytes, 2 * group.bytes);
        }
      }
    }
  }
});

test("GLM pooled formula rows expose floor groups, independent tail width, and tail in the total", () => {
  const result = calculate(glm53Pooled, { ...bf16, tokens: 9, includeLinearAttentionState: true });
  const rows = result.elementPlan.formulaRows;
  const expression = (name) => rows.find((row) => row.name === name).expression;

  assert.match(expression("indexer_pools"), /floor/);
  assert.match(expression("indexer_pools"), /tokens\s*\/\s*index_kpool/);
  assert.match(expression("indexer_bytes"), /indexer_pools/);
  assert.match(expression("indexer_bytes"), /indexer_precision_bytes/);
  assert.match(expression("index_tail_bytes"), /index_tail_bytes_per_element/);
  assert.doesNotMatch(expression("index_tail_bytes"), /indexer_precision_bytes|kv_precision_bytes|tokens\s*%/);
  assert.match(expression("total_bytes"), /index_tail_bytes/);
  assert.match(expression("total_bytes"), /indexer_bytes/);
  assert.match(expression("total_bytes"), /kda_recurrent_state_bytes/);
});

test("pool and tail fields do not change other MLA or KDA formulas", () => {
  for (const model of [kimiK3, { ...glm53Flash, formula: "dsa_mla" }]) {
    const configured = {
      ...model,
      fields: { ...model.fields, index_kpool: 4, index_tail_bytes_per_element: 4 },
    };
    const input = { ...bf16, tokens: 9, includeLinearAttentionState: true };
    const baseline = calculate(model, input);
    const result = calculate(configured, input);

    assert.equal(result.totalBytes, baseline.totalBytes);
    assert.equal(result.indexerBytes, baseline.indexerBytes);
    assert.deepEqual(result.cacheGroups, baseline.cacheGroups);
    assert.equal(result.elementPlan.byteGroups.some((group) => group.role === "index_tail"), false);
  }
});

const inkling = {
  id: "inkling",
  label: "Inkling",
  family: "Inkling",
  formula: "inkling_hybrid",
  default_tokens: 1024,
  fields: {
    num_hidden_layers: 66,
    full_attention_layers: 11,
    sliding_attention_layers: 55,
    hidden_size: 6144,
    num_key_value_heads: 8,
    head_dim: 128,
    swa_num_key_value_heads: 16,
    swa_head_dim: 128,
    sliding_window: 512,
    sconv_kernel_size: 4,
    sconv_state_bytes_per_element: 2,
    default_precision_id: "bf16_fp16",
    default_include_sconv_state: true,
    default_sconv_checkpoint_interval: "infinity",
    num_nextn_predict_layers: 8,
    draft_full_attention_layers: 2,
    draft_sliding_attention_layers: 6,
  },
};

const inklingSmall = {
  id: "inkling-small",
  label: "Inkling-Small",
  family: "Inkling",
  formula: "inkling_hybrid",
  default_tokens: 1024,
  fields: {
    num_hidden_layers: 42,
    full_attention_layers: 7,
    sliding_attention_layers: 35,
    hidden_size: 4096,
    num_key_value_heads: 8,
    head_dim: 128,
    swa_num_key_value_heads: 8,
    swa_head_dim: 128,
    sliding_window: 512,
    sconv_kernel_size: 4,
    sconv_state_bytes_per_element: 2,
    default_precision_id: "bf16_fp16",
    default_include_sconv_state: true,
    default_sconv_checkpoint_interval: "infinity",
    num_nextn_predict_layers: 8,
    draft_full_attention_layers: 2,
    draft_sliding_attention_layers: 6,
  },
};

test("Inkling counts global KV, capped SWA KV, and one BF16 SConv checkpoint", () => {
  const result = calculate(inkling, {
    tokens: 1048576,
    sequences: 1,
    includeSconvState: true,
  });
  const global = result.cacheGroups.find(
    (group) => group.label === "Full-attention KV cache",
  );
  const sliding = result.cacheGroups.find(
    (group) => group.label === "Sliding-window KV cache",
  );
  const sconv = result.cacheGroups.find(
    (group) => group.label === "SConv checkpoint state",
  );

  assert.equal(result.precisionLabel, "BF16 / FP16");
  assert.equal(global.bytes, 47244640256);
  assert.equal(sliding.bytes, 230686720);
  assert.equal(sconv.bytes, 6352896);
  assert.equal(result.totalBytes, 47481679872);
  assert.ok(Math.abs(result.totalGiB - 44.220760345458984) < 1e-12);
  assert.equal(
    result.elementPlan.components.find(
      ([label]) => label === "SConv checkpoints per sequence",
    )[1],
    1,
  );
});

test("Inkling-Small uses 7 global and 35 capped SWA layers", () => {
  const result = calculate(inklingSmall, {
    tokens: 1048576,
    sequences: 1,
    includeSconvState: true,
  });

  assert.equal(
    result.cacheGroups.find(
      (group) => group.label === "Full-attention KV cache",
    ).bytes,
    30064771072,
  );
  assert.equal(
    result.cacheGroups.find(
      (group) => group.label === "Sliding-window KV cache",
    ).bytes,
    73400320,
  );
  assert.equal(
    result.cacheGroups.find(
      (group) => group.label === "SConv checkpoint state",
    ).bytes,
    2580480,
  );
  assert.equal(result.totalBytes, 30140751872);
  assert.ok(Math.abs(result.totalGiB - 28.070762634277344) < 1e-12);
});

test("Inkling fixed interval counts the final partial interval and sequences", () => {
  const sequences = 3;
  const result = calculate(inkling, {
    ...bf16,
    tokens: 10241,
    sequences,
    includeSconvState: true,
    sconvCheckpointPolicy: "fixed_interval",
    sconvCheckpointInterval: 10240,
  });
  const sconv = result.cacheGroups.find(
    (group) => group.label === "SConv checkpoint state",
  );

  assert.equal(
    result.elementPlan.components.find(
      ([label]) => label === "SConv checkpoints per sequence",
    )[1],
    2,
  );
  assert.equal(sconv.bytes, 2 * sequences * 6352896);
});

test("Inkling treats a blank fixed interval as one prompt-end checkpoint", () => {
  const result = calculate(inkling, {
    ...bf16,
    tokens: 40960,
    includeSconvState: true,
    sconvCheckpointPolicy: "fixed_interval",
    sconvCheckpointInterval: "",
  });

  assert.equal(
    result.elementPlan.components.find(
      ([label]) => label === "SConv checkpoint interval",
    )[1],
    "∞",
  );
  assert.equal(
    result.elementPlan.components.find(
      ([label]) => label === "SConv checkpoints per sequence",
    )[1],
    1,
  );
});

test("Inkling can exclude SConv state without changing attention KV", () => {
  const withState = calculate(inkling, {
    ...bf16,
    tokens: 4096,
    includeSconvState: true,
  });
  const withoutState = calculate(inkling, {
    ...bf16,
    tokens: 4096,
    includeSconvState: false,
  });

  assert.equal(
    withoutState.cacheGroups.some(
      (group) => group.label === "SConv checkpoint state",
    ),
    false,
  );
  assert.equal(withState.kvBytes, withoutState.kvBytes);
  assert.equal(withState.totalBytes - withoutState.totalBytes, 6352896);
});

test("Inkling SConv checkpoint stays BF16 when attention KV uses FP8", () => {
  const bf16Result = calculate(inkling, {
    ...bf16,
    tokens: 4096,
    includeSconvState: true,
  });
  const fp8Result = calculate(inkling, {
    ...bf16,
    tokens: 4096,
    precision: "fp8_int8",
    includeSconvState: true,
  });
  const bf16Sconv = bf16Result.cacheGroups.find(
    (group) => group.label === "SConv checkpoint state",
  );
  const fp8Sconv = fp8Result.cacheGroups.find(
    (group) => group.label === "SConv checkpoint state",
  );

  assert.equal(fp8Result.kvBytes * 2, bf16Result.kvBytes);
  assert.equal(fp8Sconv.bytes, bf16Sconv.bytes);
});

test("Inkling draft option adds all 2 global and 6 SWA MTP layers", () => {
  const result = calculate(inkling, {
    ...bf16,
    tokens: 1048576,
    includeSconvState: true,
    includeDraftKvCache: true,
  });

  assert.equal(
    result.elementPlan.components.find(
      ([label]) => label === "Draft layers included",
    )[1],
    8,
  );
  assert.equal(
    result.elementPlan.components.find(
      ([label]) => label === "Active global-attention layers",
    )[1],
    13,
  );
  assert.equal(
    result.elementPlan.components.find(
      ([label]) => label === "Active sliding-window layers",
    )[1],
    61,
  );
  assert.equal(
    result.cacheGroups.find(
      (group) => group.label === "SConv checkpoint state",
    ).bytes,
    7114752,
  );
  assert.ok(Math.abs(result.totalGiB - 52.24490737915039) < 1e-12);
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
  assert.match(result.elementPlan.note, /is excluded/);
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
  const convBytes = 48 * 3 * (2 * 16 * 128 + 48 * 128) * 2;
  const recurrentBytes = 48 * 48 * 128 * 128 * 4;

  assert.equal(result.cacheGroups.find((group) => group.label === "Full-attention KV cache").bytes, fullBytes);
  assert.equal(result.cacheGroups.find((group) => group.label === "Linear-attention checkpoint state").bytes, convBytes + recurrentBytes);
  assert.equal(result.elementPlan.components.find(([label]) => label === "Linear state included")[1], "Yes");
  assert.match(result.elementPlan.formulaRows.find((row) => row.name === "total_bytes").expression, /linear_recurrent_state_bytes/);
  assert.equal(result.totalBytes, fullBytes + convBytes + recurrentBytes);
});

const qwen38 = {
  id: "qwen3.8-2.4t-a95b",
  label: "Qwen3.8-2.4T-A95B",
  family: "Qwen3.8",
  formula: "qwen_linear_full_hybrid",
  fields: {
    num_hidden_layers: 92,
    full_attention_layers: 23,
    linear_attention_layers: 69,
    num_key_value_heads: 4,
    head_dim: 256,
    linear_num_key_heads: 16,
    linear_key_head_dim: 128,
    linear_num_value_heads: 128,
    linear_value_head_dim: 128,
    linear_conv_kernel_dim: 4,
    mtp_num_hidden_layers: 1,
    default_recurrent_state_precision_id: "bf16_fp16",
    default_linear_state_checkpoint_interval: "infinity",
  },
};

test("Qwen3.8 counts 23 full-attention layers and one BF16 GDN checkpoint", () => {
  const tokens = 262144;
  const result = calculate(qwen38, {
    ...bf16,
    tokens,
    includeLinearAttentionState: true,
    recurrentStatePrecision: "bf16_fp16",
  });
  const fullBytes = tokens * 23 * 2 * 4 * 256 * 2;
  const convBytes = 69 * 3 * (2 * 16 * 128 + 128 * 128) * 2;
  const recurrentBytes = 69 * 128 * 128 * 128 * 2;
  const state = result.cacheGroups.find(
    (group) => group.label === "Linear-attention checkpoint state",
  );

  assert.equal(fullBytes, 23 * 1024 ** 3);
  assert.equal(state.bytes, convBytes + recurrentBytes);
  assert.equal(result.totalBytes, fullBytes + convBytes + recurrentBytes);
  assert.equal(result.recurrentStatePrecisionLabel, "BF16 / FP16");
  assert.equal(
    result.elementPlan.components.find(
      ([label]) => label === "GDN checkpoints per sequence",
    )[1],
    1,
  );
});

test("Qwen3.8 fixed intervals retain the final partial GDN checkpoint", () => {
  const tokens = 10241;
  const sequences = 2;
  const result = calculate(qwen38, {
    ...bf16,
    tokens,
    sequences,
    includeLinearAttentionState: true,
    recurrentStatePrecision: "bf16_fp16",
    qwenCheckpointPolicy: "fixed_interval",
    qwenCheckpointInterval: 10240,
  });
  const fullBytesPerSequence = tokens * 23 * 2 * 4 * 256 * 2;
  const checkpointBytes =
    69 * 3 * (2 * 16 * 128 + 128 * 128) * 2 +
    69 * 128 * 128 * 128 * 2;

  assert.equal(
    result.elementPlan.components.find(
      ([label]) => label === "GDN checkpoints per sequence",
    )[1],
    2,
  );
  assert.equal(
    result.totalBytes,
    sequences * (fullBytesPerSequence + 2 * checkpointBytes),
  );
});

test("Qwen3.8 recurrent precision changes recurrent state but not conv history", () => {
  const input = {
    ...bf16,
    tokens: 1024,
    includeLinearAttentionState: true,
  };
  const bf16Result = calculate(qwen38, {
    ...input,
    recurrentStatePrecision: "bf16_fp16",
  });
  const fp32Result = calculate(qwen38, {
    ...input,
    recurrentStatePrecision: "fp32",
  });
  const recurrentElements = 69 * 128 * 128 * 128;

  assert.equal(fp32Result.totalBytes - bf16Result.totalBytes, recurrentElements * 2);
  assert.equal(fp32Result.recurrentStatePrecisionLabel, "FP32");
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
  const convBytes = 18 * 3 * (2 * 16 * 128 + 16 * 128) * 2;
  const recurrentBytes = 18 * 16 * 128 * 128 * 4;
  const linearState = result.cacheGroups.find(
    (group) => group.label === "Linear-attention checkpoint state",
  );

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

test("MiniMax M3 MSA formula counts main KV plus key-only indexer cache", () => {
  const model = {
    id: "minimax-m3",
    label: "MiniMax M3",
    formula: "minimax_msa",
    fields: {
      num_hidden_layers: 60,
      full_attention_layers: 3,
      sparse_attention_layers: 57,
      num_key_value_heads: 4,
      head_dim: 128,
      index_head_dim: 128,
      index_n_heads: 4,
      index_block_size: 128,
      index_topk_blocks: 16,
      index_local_blocks: 1,
      indexer_fixed_precision_id: "bf16_fp16",
      num_mtp_modules: 7,
      num_nextn_predict_layers: 1,
      disable_draft_kv_cache: true,
    },
  };

  const plan = calculateElementsPerSequence(model, 1048576, { includeDraftKvCache: true });

  assert.equal(plan.elementsPerToken, 68736);
  assert.equal(plan.byteGroups.find((group) => group.role === "kv").elements, 64424509440);
  assert.equal(plan.byteGroups.find((group) => group.role === "indexer").elements, 7650410496);
  assert.equal(plan.components.find(([label]) => label === "Full-attention layers")[1], 3);
  assert.equal(plan.components.find(([label]) => label === "Sparse-attention layers")[1], 57);
  assert.equal(plan.components.find(([label]) => label === "MTP modules not included")[1], 7);
  assert.match(plan.note, /lightweight indexer/);
  assert.match(plan.note, /relevant KV blocks/);
});

test("MiniMax M3 fixes indexer precision to BF16 while KV precision can use FP8", () => {
  const model = {
    id: "minimax-m3",
    label: "MiniMax M3",
    formula: "minimax_msa",
    fields: {
      num_hidden_layers: 60,
      full_attention_layers: 3,
      sparse_attention_layers: 57,
      num_key_value_heads: 4,
      head_dim: 128,
      index_head_dim: 128,
      index_n_heads: 4,
      index_block_size: 128,
      index_topk_blocks: 16,
      index_local_blocks: 1,
      indexer_fixed_precision_id: "bf16_fp16",
      num_mtp_modules: 7,
      num_nextn_predict_layers: 1,
      disable_draft_kv_cache: true,
    },
  };

  const result = calculate(model, {
    tokens: 1048576,
    precision: "fp8_int8",
    indexerPrecision: "fp4_int4",
    includeDraftKvCache: true,
    sequences: 1,
    tensorParallel: 1,
  });

  assert.equal(result.indexerPrecisionLabel, "BF16 / FP16");
  assert.equal(result.components.find(([label]) => label === "KV precision bytes")[1], 1);
  assert.equal(result.components.find(([label]) => label === "Indexer precision bytes")[1], 2);
  assert.ok(Math.abs(result.kvGiB - 60) < 1e-9);
  assert.ok(Math.abs(result.indexerGiB - 14.25) < 1e-9);
  assert.ok(Math.abs(result.totalGiB - 74.25) < 1e-9);
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

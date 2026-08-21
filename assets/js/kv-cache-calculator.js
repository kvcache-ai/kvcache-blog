(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.KVCacheCalculator = factory();
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const BYTES_PER_GB = 1e9;
  const BYTES_PER_GIB = 1024 ** 3;
  const RESULT_DIGITS = 5;
  const QWEN_LINEAR_CONV_BYTES_PER_ELEMENT = 2;
  const QWEN_LINEAR_RECURRENT_BYTES_PER_ELEMENT = 4;
  const KIMI_KDA_CONV_BYTES_PER_ELEMENT = 2;
  const KIMI_KDA_RECURRENT_BYTES_PER_ELEMENT = 4;
  const INKLING_SCONV_BYTES_PER_ELEMENT = 2;
  const STATE_CHECKPOINT_INFINITY = "∞";
  const STATE_CUSTOM_INTERVAL_DEFAULT = 10240;
  const STATE_CHECKPOINT_POLICY_FIXED_INTERVAL = "fixed_interval";

  const DEFAULT_PRECISIONS = {
    bf16_fp16: { label: "BF16 / FP16", bytesPerElement: 2 },
    fp8_int8: { label: "FP8 / INT8", bytesPerElement: 1 },
    fp4_int4: { label: "FP4 / INT4", bytesPerElement: 0.5 },
  };

  const DEFAULT_RECURRENT_STATE_PRECISIONS = {
    bf16_fp16: { label: "BF16 / FP16", bytesPerElement: 2 },
    fp32: { label: "FP32", bytesPerElement: 4 },
  };

  const FORMULA_LABELS = {
    standard_gqa: "Standard MHA/GQA",
    mla: "MLA latent KV",
    dsa_mla: "DSA/MLA with indexer",
    kimi_kda_mla_hybrid: "Kimi KDA/MLA hybrid",
    inkling_hybrid: "Inkling global/SWA with SConv",
    qwen_linear_full_hybrid: "Qwen linear/full hybrid",
    mixed_full_sliding_gqa: "Mixed full/sliding GQA",
    minimax_msa: "MiniMax MSA sparse attention",
    deepseek_v4_hybrid: "DeepSeek V4 hybrid sparse attention",
  };

  function toPositiveNumber(value, fallback) {
    const parsed = Number(value);
    return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
  }

  function toPositiveInteger(value, fallback) {
    return Math.max(1, Math.floor(toPositiveNumber(value, fallback)));
  }

  function normalizePrecisionOptions(precisionOptions, fallback) {
    if (!Array.isArray(precisionOptions)) return fallback;
    return Object.fromEntries(
      precisionOptions.map((option) => [
        option.id,
        {
          label: option.label,
          bytesPerElement: Number(option.bytes_per_element),
        },
      ]),
    );
  }

  function precisionOptions(options) {
    return normalizePrecisionOptions(options && options.precisionOptions, DEFAULT_PRECISIONS);
  }

  function recurrentStatePrecisionOptions(options) {
    return normalizePrecisionOptions(
      options && options.recurrentStatePrecisionOptions,
      DEFAULT_RECURRENT_STATE_PRECISIONS,
    );
  }

  function isDeepSeekV4(model) {
    return model && model.formula === "deepseek_v4_hybrid";
  }

  function isInkling(model) {
    return Boolean(model && model.formula === "inkling_hybrid");
  }

  function hasIndexerCache(model) {
    return Boolean(model && model.fields && Number.isFinite(Number(model.fields.index_head_dim)));
  }

  function safeNumber(value, fallback) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  }

  function draftLayerCount(model) {
    if (!model || !model.fields) return 0;
    if (model.fields.disable_draft_kv_cache === true) return 0;
    const nextnLayers = safeNumber(model.fields.num_nextn_predict_layers, 0);
    if (nextnLayers > 0) return nextnLayers;
    if (model.fields.use_mtp === true) {
      return (
        safeNumber(model.fields.num_mtp_modules, 0) *
        safeNumber(model.fields.mtp_transformer_layers, 0)
      );
    }
    return 0;
  }

  function hasDraftKvCache(model) {
    if (!model || !model.fields) return false;
    if (isDeepSeekV4(model)) {
      const layers = safeNumber(model.fields.num_hidden_layers, 0);
      return Array.isArray(model.fields.compress_ratios) && model.fields.compress_ratios.length > layers;
    }
    return draftLayerCount(model) > 0;
  }

  function hasLinearAttentionState(model) {
    return Boolean(
      model &&
        (model.formula === "qwen_linear_full_hybrid" ||
          model.formula === "kimi_kda_mla_hybrid"),
    );
  }

  function hasKdaCheckpointInterval(model) {
    return Boolean(model && model.formula === "kimi_kda_mla_hybrid");
  }

  function hasQwenCheckpointInterval(model) {
    return Boolean(model && model.formula === "qwen_linear_full_hybrid");
  }

  function hasSconvState(model) {
    return isInkling(model);
  }

  function parseStateCheckpointInterval(value, fallback) {
    if (
      value === Infinity ||
      value === STATE_CHECKPOINT_INFINITY ||
      (typeof value === "string" && value.toLowerCase() === "infinity")
    ) {
      return Infinity;
    }
    const parsed = Number(value);
    return Number.isFinite(parsed) && parsed > 0
      ? Math.max(1, Math.floor(parsed))
      : fallback;
  }

  function defaultKdaCheckpointInterval(model) {
    const value =
      model && model.fields
        ? model.fields.default_kda_checkpoint_interval
        : undefined;
    return parseStateCheckpointInterval(value, Infinity);
  }

  function defaultSconvCheckpointInterval(model) {
    const value =
      model && model.fields
        ? model.fields.default_sconv_checkpoint_interval
        : undefined;
    return parseStateCheckpointInterval(value, Infinity);
  }

  function defaultQwenCheckpointInterval(model) {
    const value =
      model && model.fields
        ? model.fields.default_linear_state_checkpoint_interval
        : undefined;
    return parseStateCheckpointInterval(value, Infinity);
  }

  function defaultStateCheckpointInterval(model) {
    if (hasKdaCheckpointInterval(model)) return defaultKdaCheckpointInterval(model);
    if (hasQwenCheckpointInterval(model)) return defaultQwenCheckpointInterval(model);
    return defaultSconvCheckpointInterval(model);
  }

  function formatStateCheckpointInterval(value) {
    return Number.isFinite(value)
      ? String(value)
      : STATE_CHECKPOINT_INFINITY;
  }

  function toBoolean(value) {
    return value === true || value === "true" || value === "on" || value === "1";
  }

  function defaultPrecisionId(model, options) {
    const optionsById = precisionOptions(options || {});
    const modelDefault =
      model && model.fields && typeof model.fields.default_precision_id === "string"
        ? model.fields.default_precision_id
        : undefined;
    if (modelDefault && optionsById[modelDefault]) return modelDefault;
    if (isDeepSeekV4(model) && optionsById.fp8_int8) return "fp8_int8";
    return optionsById.bf16_fp16 ? "bf16_fp16" : Object.keys(optionsById)[0];
  }

  function defaultRecurrentStatePrecisionId(model, options) {
    const optionsById = recurrentStatePrecisionOptions(options || {});
    const modelDefault =
      model &&
      model.fields &&
      typeof model.fields.default_recurrent_state_precision_id === "string"
        ? model.fields.default_recurrent_state_precision_id
        : "fp32";
    if (optionsById[modelDefault]) return modelDefault;
    return optionsById.fp32 ? "fp32" : Object.keys(optionsById)[0];
  }

  function indexerPrecisionOptions(options) {
    return normalizePrecisionOptions(
      options && options.indexerPrecisionOptions,
      precisionOptions(options || {}),
    );
  }

  function fixedIndexerPrecisionId(model) {
    return model && model.fields && typeof model.fields.indexer_fixed_precision_id === "string"
      ? model.fields.indexer_fixed_precision_id
      : undefined;
  }

  function defaultIndexerPrecisionId(model, options, fallbackPrecisionId) {
    const optionsById = indexerPrecisionOptions(options || {});
    const fixedPrecisionId = fixedIndexerPrecisionId(model);
    if (fixedPrecisionId && optionsById[fixedPrecisionId]) return fixedPrecisionId;
    if (isDeepSeekV4(model) && optionsById.fp4_int4) return "fp4_int4";
    if (fallbackPrecisionId && optionsById[fallbackPrecisionId]) return fallbackPrecisionId;
    if (optionsById.bf16_fp16) return "bf16_fp16";
    return optionsById.fp4_int4 ? "fp4_int4" : Object.keys(optionsById)[0];
  }

  function getPrecisionProfile(precisionId, options, fallbackId) {
    const optionsById = precisionOptions(options || {});
    const selected = optionsById[precisionId] || optionsById[fallbackId] || DEFAULT_PRECISIONS.bf16_fp16;
    return {
      label: selected.label,
      bytesPerElement: selected.bytesPerElement,
    };
  }

  function getIndexerPrecisionProfile(precisionId, options, model, fallbackPrecisionId) {
    const optionsById = indexerPrecisionOptions(options || {});
    const fixedPrecisionId = fixedIndexerPrecisionId(model);
    const selected =
      (fixedPrecisionId && optionsById[fixedPrecisionId]) ||
      optionsById[precisionId] ||
      optionsById[defaultIndexerPrecisionId(model, options, fallbackPrecisionId)] ||
      DEFAULT_PRECISIONS.fp4_int4;
    return {
      label: selected.label,
      bytesPerElement: selected.bytesPerElement,
    };
  }

  function getRecurrentStatePrecisionProfile(precisionId, options, model) {
    const optionsById = recurrentStatePrecisionOptions(options || {});
    const selected =
      optionsById[precisionId] ||
      optionsById[defaultRecurrentStatePrecisionId(model, options)] ||
      DEFAULT_RECURRENT_STATE_PRECISIONS.fp32;
    return {
      label: selected.label,
      bytesPerElement: selected.bytesPerElement,
    };
  }

  function getField(model, name) {
    if (!model || !model.fields || !Number.isFinite(Number(model.fields[name]))) {
      throw new Error(`Model ${model ? model.id : ""} is missing numeric field ${name}`);
    }
    return Number(model.fields[name]);
  }

  function optionalField(model, name, fallback) {
    if (model && model.fields && Number.isFinite(Number(model.fields[name]))) {
      return Number(model.fields[name]);
    }
    return fallback;
  }

  function fieldList(model, names) {
    const fields = model && model.fields;
    if (!fields || typeof fields !== "object") return "";
    return names
      .filter((name) => Object.prototype.hasOwnProperty.call(fields, name))
      .map((name) => `${name}=${fields[name]}`)
      .join(", ");
  }

  function indexerLayerPlan(model, layers, draftLayers) {
    const mainIndexerLayers = optionalField(model, "indexer_full_layers", layers);
    const sharedIndexerLayers = optionalField(
      model,
      "indexer_shared_layers",
      Math.max(0, layers - mainIndexerLayers),
    );
    const draftIndexerLayers =
      draftLayers > 0 ? optionalField(model, "draft_indexer_layers", draftLayers) : 0;
    return {
      mainIndexerLayers,
      sharedIndexerLayers,
      draftIndexerLayers,
      activeIndexerLayers: mainIndexerLayers + draftIndexerLayers,
    };
  }

  function countByValue(values, target) {
    return values.filter((value) => Number(value) === target).length;
  }

  function calculateElementsPerSequence(model, tokens, settings) {
    const formula = model.formula;
    const includeDraftKvCache = toBoolean(settings && settings.includeDraftKvCache);
    const includeLinearAttentionState = toBoolean(settings && settings.includeLinearAttentionState);
    const includeSconvState = toBoolean(settings && settings.includeSconvState);

    if (formula === "standard_gqa") {
      const layers = getField(model, "num_hidden_layers");
      const draftLayers = includeDraftKvCache ? draftLayerCount(model) : 0;
      const activeLayers = layers + draftLayers;
      const kvHeads = getField(model, "num_key_value_heads");
      const headDim = getField(model, "head_dim");
      const elementsPerToken = activeLayers * 2 * kvHeads * headDim;
      return {
        elementsPerSequence: elementsPerToken * tokens,
        elementsPerToken,
        formulaLabel: FORMULA_LABELS[formula],
        formulaText:
          "active_layers = main_layers + draft_layers_if_enabled\ntotal_bytes = tokens * sequences * active_layers * 2 * num_key_value_heads * head_dim * precision_bytes",
        formulaRows: [
          {
            name: "active_layers",
            expression: "main_layers + draft_layers_if_enabled",
            description: "Draft layers are counted only when Include draft KV cache is enabled for models that define an MTP/draft stack.",
          },
          {
            name: "total_bytes",
            expression: "tokens x sequences x active_layers x 2 x num_key_value_heads x head_dim x precision_bytes",
            description: "Total KV cache bytes for all cached tokens and concurrent sequences.",
          },
        ],
        note: "Production estimate of base KV payload; allocator and memory-pool bytes are excluded. Draft KV is included only when the checkbox is enabled.",
        byteGroups: [{ role: "kv", label: "KV cache", elements: elementsPerToken * tokens }],
        components: [
          ["Main layers", layers],
          ["Draft layers included", draftLayers, "Extra MTP/draft layers included in KV capacity when the checkbox is enabled."],
          ["Per-token elements", elementsPerToken, "Number of scalar KV elements needed for one token before multiplying by precision bytes."],
          ["Model fields", fieldList(model, ["num_hidden_layers", "num_key_value_heads", "head_dim"])],
        ],
      };
    }

    if (formula === "mla") {
      const layers = getField(model, "num_hidden_layers");
      const draftLayers = includeDraftKvCache ? draftLayerCount(model) : 0;
      const activeLayers = layers + draftLayers;
      const kvRank = getField(model, "kv_lora_rank");
      const ropeDim = getField(model, "qk_rope_head_dim");
      const elementsPerToken = activeLayers * (kvRank + ropeDim);
      return {
        elementsPerSequence: elementsPerToken * tokens,
        elementsPerToken,
        formulaLabel: FORMULA_LABELS[formula],
        formulaText:
          "active_layers = main_layers + draft_layers_if_enabled\ntotal_bytes = tokens * sequences * active_layers * (kv_lora_rank + qk_rope_head_dim) * precision_bytes",
        formulaRows: [
          {
            name: "active_layers",
            expression: "main_layers + draft_layers_if_enabled",
            description: "Draft layers are counted only when Include draft KV cache is enabled for models that define an MTP/draft stack.",
          },
          {
            name: "total_bytes",
            expression: "tokens x sequences x active_layers x (kv_lora_rank + qk_rope_head_dim) x precision_bytes",
            description: "Total latent KV bytes for all cached tokens and concurrent sequences.",
          },
        ],
        note: "Production estimate of MLA latent KV payload; allocator and memory-pool bytes are excluded. Draft KV is included only when the checkbox is enabled.",
        byteGroups: [{ role: "kv", label: "KV cache", elements: elementsPerToken * tokens }],
        components: [
          ["Main layers", layers],
          ["Draft layers included", draftLayers, "Extra MTP/draft layers included in KV capacity when the checkbox is enabled."],
          ["Per-token elements", elementsPerToken, "Number of scalar latent KV elements needed for one token before multiplying by precision bytes."],
          ["Model fields", fieldList(model, ["num_hidden_layers", "kv_lora_rank", "qk_rope_head_dim"])],
        ],
      };
    }

    if (formula === "dsa_mla") {
      const layers = getField(model, "num_hidden_layers");
      const draftLayers = includeDraftKvCache ? draftLayerCount(model) : 0;
      const activeLayers = layers + draftLayers;
      const indexerPlan = indexerLayerPlan(model, layers, draftLayers);
      const indexDim = getField(model, "index_head_dim");
      const kvRank = getField(model, "kv_lora_rank");
      const ropeDim = getField(model, "qk_rope_head_dim");
      const kvElementsPerLayer = kvRank + ropeDim;
      const indexerElementsPerLayer = indexDim;

      const kvElementsPerToken = activeLayers * kvElementsPerLayer;
      const indexerElementsPerToken =
        indexerPlan.activeIndexerLayers * indexerElementsPerLayer;
      const elementsPerToken = kvElementsPerToken + indexerElementsPerToken;
      return {
        elementsPerSequence: elementsPerToken * tokens,
        elementsPerToken,
        formulaLabel: FORMULA_LABELS[formula],
        formulaText:
          "active_layers = main_layers + draft_layers_if_enabled\nactive_indexer_layers = main_indexer_layers + draft_indexer_layers_if_enabled\nkv_bytes = tokens * sequences * active_layers * (kv_lora_rank + qk_rope_head_dim) * kv_precision_bytes\nindexer_bytes = tokens * sequences * active_indexer_layers * index_head_dim * indexer_precision_bytes\ntotal_bytes = kv_bytes + indexer_bytes",
        formulaRows: [
          {
            name: "active_layers",
            expression: "main_layers + draft_layers_if_enabled",
            description: "Draft layers are counted only when Include draft KV cache is enabled for models that define a next-token prediction stack.",
          },
          {
            name: "active_indexer_layers",
            expression: "main_indexer_layers + draft_indexer_layers_if_enabled",
            description: "For DSA models with shared indexer layers, only full indexer layers allocate independent indexer key cache.",
          },
          {
            name: "kv_bytes",
            expression: "tokens x sequences x active_layers x (kv_lora_rank + qk_rope_head_dim) x kv_precision_bytes",
            description: "Latent KV payload stored by the production MLA/DSA path.",
          },
          {
            name: "indexer_bytes",
            expression: "tokens x sequences x active_indexer_layers x index_head_dim x indexer_precision_bytes",
            description: "Additional per-token indexer state used by the indexer attention path.",
          },
          {
            name: "total_bytes",
            expression: "kv_bytes + indexer_bytes",
            description: "Combined cache payload after applying the selected KV and indexer precisions.",
          },
        ],
        note: indexerPlan.sharedIndexerLayers > 0
          ? "Production estimate uses latent KV plus independently stored indexer state; shared indexer layers reuse the full indexer layers' selection. Expanded HF-compatible cache is not included."
          : "Production estimate uses latent KV plus indexer state; expanded HF-compatible cache is not included.",
        byteGroups: [
          { role: "kv", label: "KV cache", elements: kvElementsPerToken * tokens },
          { role: "indexer", label: "Indexer cache", elements: indexerElementsPerToken * tokens },
        ],
        components: [
          ["Main layers", layers],
          ["Draft layers included", draftLayers, "Extra next-token prediction layers included in KV capacity when the checkbox is enabled."],
          ["Main indexer layers", indexerPlan.mainIndexerLayers, "Full indexer layers that allocate independent indexer key cache."],
          ["Shared indexer layers", indexerPlan.sharedIndexerLayers, "DSA layers that reuse the previous full indexer layer's top-k selection."],
          ["Draft indexer layers included", indexerPlan.draftIndexerLayers, "Draft/MTP indexer layers counted when Include draft KV cache is enabled."],
          ["KV elements per token", kvElementsPerToken, "Latent KV elements per token before applying KV precision."],
          ["Indexer elements per token", indexerElementsPerToken, "Indexer elements per token before applying indexer precision."],
          ["Per-token elements", elementsPerToken, "KV plus indexer scalar elements per token before multiplying by precision bytes."],
          ["Model fields", fieldList(model, ["num_hidden_layers", "kv_lora_rank", "qk_rope_head_dim", "index_head_dim", "indexer_full_layers", "indexer_shared_layers", "draft_indexer_layers"])],
        ],
      };
    }

    if (formula === "kimi_kda_mla_hybrid") {
      const layers = getField(model, "num_hidden_layers");
      const fullLayers = getField(model, "full_attention_layers");
      const kdaLayers = getField(model, "kda_layers");
      const kdaCheckpointInterval = parseStateCheckpointInterval(
        settings && settings.kdaCheckpointInterval,
        defaultKdaCheckpointInterval(model),
      );
      const kdaCheckpointCount = includeLinearAttentionState
        ? Number.isFinite(kdaCheckpointInterval)
          ? Math.ceil(tokens / kdaCheckpointInterval)
          : 1
        : 0;
      const kvRank = getField(model, "kv_lora_rank");
      const ropeDim = getField(model, "qk_rope_head_dim");
      const kdaHeads = getField(model, "kda_num_heads");
      const kdaHeadDim = getField(model, "kda_head_dim");
      const kdaKeyHeads = optionalField(model, "kda_num_key_heads", kdaHeads);
      const kdaKeyDim = optionalField(model, "kda_key_head_dim", kdaHeadDim);
      const kdaValueHeads = optionalField(model, "kda_num_value_heads", kdaHeads);
      const kdaValueDim = optionalField(model, "kda_value_head_dim", kdaHeadDim);
      const kdaConvKernel = getField(model, "kda_conv_kernel_size");
      const convBytesPerElement = optionalField(
        model,
        "kda_conv_state_bytes_per_element",
        KIMI_KDA_CONV_BYTES_PER_ELEMENT,
      );
      const recurrentBytesPerElement = optionalField(
        model,
        "kda_recurrent_state_bytes_per_element",
        KIMI_KDA_RECURRENT_BYTES_PER_ELEMENT,
      );

      const mlaElementsPerToken = fullLayers * (kvRank + ropeDim);
      const mlaElements = mlaElementsPerToken * tokens;
      const kdaConvElements =
        kdaLayers *
        (kdaConvKernel - 1) *
        (kdaHeads * kdaHeadDim +
          kdaKeyHeads * kdaKeyDim +
          kdaValueHeads * kdaValueDim);
      const kdaRecurrentElements =
        kdaLayers * kdaValueHeads * kdaValueDim * kdaKeyDim;
      const kdaStateBytes =
        kdaConvElements * convBytesPerElement +
        kdaRecurrentElements * recurrentBytesPerElement;
      const kdaCheckpointBytesPerSequence =
        kdaCheckpointCount * kdaStateBytes;
      const byteGroups = [
        {
          role: "kv",
          label: "MLA latent KV cache",
          elements: mlaElements,
        },
      ];
      const formulaRows = [
        {
          name: "mla_kv_bytes",
          expression:
            "tokens x sequences x full_attention_layers x (kv_lora_rank + qk_rope_head_dim) x precision_bytes",
          description:
            "Stores one compressed MLA latent vector plus the RoPE key segment for every cached token.",
        },
      ];

      if (includeLinearAttentionState) {
        byteGroups.push({
          role: "linear_state",
          label: "KDA checkpoint state",
          bytesPerSequence: kdaCheckpointBytesPerSequence,
        });
        formulaRows.push(
          {
            name: "kda_checkpoint_count",
            expression:
              "interval is infinity ? 1 : ceil(tokens / kda_checkpoint_interval)",
            description:
              "The infinity default stores one final checkpoint; finite intervals also count a final partial interval.",
          },
          {
            name: "kda_conv_state_bytes",
            expression:
              "sequences x kda_checkpoint_count x kda_layers x (conv_kernel - 1) x (q_dim + k_dim + v_dim) x conv_state_bytes",
            description:
              "BF16 KDA short-convolution history stored in every retained checkpoint.",
          },
          {
            name: "kda_recurrent_state_bytes",
            expression:
              "sequences x kda_checkpoint_count x kda_layers x value_heads x value_head_dim x key_head_dim x recurrent_state_bytes",
            description:
              "FP32 KDA recurrent matrices stored in every retained checkpoint.",
          },
          {
            name: "total_bytes",
            expression:
              "mla_kv_bytes + kda_conv_state_bytes + kda_recurrent_state_bytes",
            description:
              "Combined token-linear MLA cache and retained KDA checkpoint states.",
          },
        );
      } else {
        formulaRows.push(
          {
            name: "kda_linear_attention_state",
            expression:
              "excluded unless Include linear-attention state is enabled",
            description:
              "KDA layers keep fixed convolution and recurrent state rather than ordinary token-addressable KV blocks.",
          },
          {
            name: "total_bytes",
            expression: "mla_kv_bytes",
            description:
              "Reusable token-addressable MLA latent KV payload only.",
          },
        );
      }

      return {
        elementsPerSequence:
          mlaElements +
          (includeLinearAttentionState
            ? kdaCheckpointCount *
              (kdaConvElements + kdaRecurrentElements)
            : 0),
        elementsPerToken: mlaElementsPerToken,
        hitRateElementsPerToken: mlaElementsPerToken,
        formulaLabel: FORMULA_LABELS[formula],
        formulaText:
          "mla_kv_bytes = tokens * sequences * full_attention_layers * (kv_lora_rank + qk_rope_head_dim) * precision_bytes\nkda_checkpoint_count = interval_is_infinity ? 1 : ceil(tokens / kda_checkpoint_interval)\ntotal_bytes = mla_kv_bytes + optional_kda_checkpoint_bytes",
        formulaRows,
        note: includeLinearAttentionState
          ? "Includes the 24-layer token-addressable MLA latent cache and retained BF16-convolution/FP32-recurrent KDA checkpoints. Active, ping-pong, and speculative runtime buffers are excluded."
          : "Includes the 24-layer token-addressable MLA latent cache. The 69 KDA layers' sequence-level state is excluded.",
        byteGroups,
        components: [
          ["Main layers", layers],
          [
            "MLA full-attention layers",
            fullLayers,
            "Layers that allocate token-addressable compressed MLA latent KV plus RoPE key cache.",
          ],
          [
            "KDA linear-attention layers",
            kdaLayers,
            "Layers whose convolution and recurrent state is stored in each KDA checkpoint.",
          ],
          [
            "KDA state included",
            includeLinearAttentionState ? "Yes" : "No",
            "When enabled, adds retained KDA convolution and recurrent checkpoints.",
          ],
          [
            "KDA checkpoint interval",
            formatStateCheckpointInterval(kdaCheckpointInterval),
            "The infinity default stores one final checkpoint; otherwise this is the number of tokens represented by each retained checkpoint.",
          ],
          [
            "KDA checkpoints per sequence",
            kdaCheckpointCount,
            "One when the interval is infinity; otherwise ceil(tokens / kda_checkpoint_interval).",
          ],
          [
            "MLA elements per token",
            mlaElementsPerToken,
            "full_attention_layers x (kv_lora_rank + qk_rope_head_dim).",
          ],
          [
            "KDA conv elements per checkpoint",
            kdaConvElements,
            "All KDA Q/K/V short-convolution history stored in one checkpoint.",
          ],
          [
            "KDA recurrent elements per checkpoint",
            kdaRecurrentElements,
            "All KDA recurrent matrices stored in one checkpoint.",
          ],
          [
            "KDA bytes per checkpoint",
            kdaStateBytes,
            "One BF16-convolution/FP32-recurrent KDA checkpoint.",
          ],
          [
            "KDA checkpoint bytes per sequence",
            kdaCheckpointBytesPerSequence,
            "Retained checkpoint count multiplied by bytes per checkpoint.",
          ],
          ["KDA conv-state bytes", convBytesPerElement],
          ["KDA recurrent-state bytes", recurrentBytesPerElement],
          [
            "Model fields",
            fieldList(model, [
              "num_hidden_layers",
              "full_attention_layers",
              "kda_layers",
              "kv_lora_rank",
              "qk_rope_head_dim",
              "kda_num_heads",
              "kda_head_dim",
              "kda_conv_kernel_size",
              "default_kda_checkpoint_interval",
            ]),
          ],
        ],
      };
    }

    if (formula === "inkling_hybrid") {
      const layers = getField(model, "num_hidden_layers");
      const mainGlobalLayers = getField(model, "full_attention_layers");
      const mainSwaLayers = getField(model, "sliding_attention_layers");
      const draftGlobalLayers = includeDraftKvCache
        ? getField(model, "draft_full_attention_layers")
        : 0;
      const draftSwaLayers = includeDraftKvCache
        ? getField(model, "draft_sliding_attention_layers")
        : 0;
      const activeGlobalLayers = mainGlobalLayers + draftGlobalLayers;
      const activeSwaLayers = mainSwaLayers + draftSwaLayers;
      const hiddenSize = getField(model, "hidden_size");
      const globalKvHeads = getField(model, "num_key_value_heads");
      const globalHeadDim = getField(model, "head_dim");
      const swaKvHeads = getField(model, "swa_num_key_value_heads");
      const swaHeadDim = getField(model, "swa_head_dim");
      const slidingWindow = getField(model, "sliding_window");
      const retainedSwaTokens = Math.min(tokens, slidingWindow);
      const sconvKernel = getField(model, "sconv_kernel_size");
      const sconvBytesPerElement = optionalField(
        model,
        "sconv_state_bytes_per_element",
        INKLING_SCONV_BYTES_PER_ELEMENT,
      );
      const sconvCheckpointInterval = parseStateCheckpointInterval(
        settings && settings.sconvCheckpointInterval,
        defaultSconvCheckpointInterval(model),
      );
      const sconvCheckpointCount = includeSconvState
        ? Number.isFinite(sconvCheckpointInterval)
          ? Math.ceil(tokens / sconvCheckpointInterval)
          : 1
        : 0;

      const globalElements =
        tokens * activeGlobalLayers * globalKvHeads * 2 * globalHeadDim;
      const swaElements =
        retainedSwaTokens * activeSwaLayers * swaKvHeads * 2 * swaHeadDim;
      const globalSconvElementsPerLayer =
        (sconvKernel - 1) *
        (2 * globalKvHeads * globalHeadDim + 2 * hiddenSize);
      const swaSconvElementsPerLayer =
        (sconvKernel - 1) *
        (2 * swaKvHeads * swaHeadDim + 2 * hiddenSize);
      const sconvElementsPerCheckpoint =
        activeGlobalLayers * globalSconvElementsPerLayer +
        activeSwaLayers * swaSconvElementsPerLayer;
      const sconvBytesPerCheckpoint =
        sconvElementsPerCheckpoint * sconvBytesPerElement;
      const sconvCheckpointBytesPerSequence =
        sconvCheckpointCount * sconvBytesPerCheckpoint;
      const byteGroups = [
        {
          role: "kv",
          label: "Full-attention KV cache",
          elements: globalElements,
        },
        {
          role: "kv",
          label: "Sliding-window KV cache",
          elements: swaElements,
        },
      ];
      const formulaRows = [
        {
          name: "active_global_layers",
          expression: "main_global_layers + draft_global_layers_if_enabled",
          description:
            "Global-attention MTP layers are counted only when Include draft KV cache is enabled.",
        },
        {
          name: "active_swa_layers",
          expression: "main_swa_layers + draft_swa_layers_if_enabled",
          description:
            "Sliding-window MTP layers are counted only when Include draft KV cache is enabled.",
        },
        {
          name: "global_kv_bytes",
          expression:
            "tokens x sequences x active_global_layers x global_kv_heads x 2 x head_dim x kv_precision_bytes",
          description:
            "Global-attention layers retain ordinary K and V for every cached token.",
        },
        {
          name: "swa_kv_bytes",
          expression:
            "min(tokens, sliding_window) x sequences x active_swa_layers x swa_kv_heads x 2 x swa_head_dim x kv_precision_bytes",
          description:
            "Sliding-window layers retain K and V only for the configured local window.",
        },
      ];

      if (includeSconvState) {
        byteGroups.push({
          role: "sconv_state",
          label: "SConv checkpoint state",
          bytesPerSequence: sconvCheckpointBytesPerSequence,
        });
        formulaRows.push(
          {
            name: "sconv_checkpoint_count",
            expression:
              "interval is infinity ? 1 : ceil(tokens / sconv_checkpoint_interval)",
            description:
              "The infinity default stores one final checkpoint; finite intervals also count a final partial interval.",
          },
          {
            name: "sconv_checkpoint_bytes",
            expression:
              "sequences x checkpoint_count x (kernel_size - 1) x 2 x [active_global_layers x (2 x global_kv_heads x head_dim + 2 x hidden_size) + active_swa_layers x (2 x swa_kv_heads x swa_head_dim + 2 x hidden_size)]",
            description:
              "Logical BF16 K, V, attention-output, and MLP-output convolution history stored at each retained prefix boundary.",
          },
          {
            name: "total_bytes",
            expression:
              "global_kv_bytes + swa_kv_bytes + sconv_checkpoint_bytes",
            description:
              "Combined reusable attention KV and retained short-convolution checkpoint payload.",
          },
        );
      } else {
        formulaRows.push(
          {
            name: "sconv_checkpoint_state",
            expression: "excluded unless Include SConv state is enabled",
            description:
              "Exact prefix continuation requires matching short-convolution state or recomputation from an earlier checkpoint.",
          },
          {
            name: "total_bytes",
            expression: "global_kv_bytes + swa_kv_bytes",
            description: "Reusable attention KV payload without SConv state.",
          },
        );
      }

      return {
        elementsPerSequence:
          globalElements +
          swaElements +
          sconvCheckpointCount * sconvElementsPerCheckpoint,
        elementsPerToken: (globalElements + swaElements) / tokens,
        formulaLabel: FORMULA_LABELS[formula],
        formulaText:
          "global_kv_bytes = tokens * sequences * active_global_layers * global_kv_heads * 2 * head_dim * kv_precision_bytes\nswa_kv_bytes = min(tokens, sliding_window) * sequences * active_swa_layers * swa_kv_heads * 2 * swa_head_dim * kv_precision_bytes\ntotal_bytes = global_kv_bytes + swa_kv_bytes + optional_sconv_checkpoint_bytes",
        formulaRows,
        note: includeSconvState
          ? "Includes reusable text-generation KV and logical BF16 SConv checkpoints. vLLM page padding, active/ping-pong/speculative buffers, and vision/audio encoder activations are excluded."
          : "Includes reusable text-generation KV only. SConv state required for exact prefix continuation is excluded.",
        byteGroups,
        components: [
          ["Main layers", layers],
          ["Main global-attention layers", mainGlobalLayers],
          ["Main sliding-window layers", mainSwaLayers],
          [
            "Draft layers included",
            draftGlobalLayers + draftSwaLayers,
            "All eight configured MTP layers are counted as a capacity upper bound when draft KV is enabled.",
          ],
          ["Draft global-attention layers", draftGlobalLayers],
          ["Draft sliding-window layers", draftSwaLayers],
          ["Active global-attention layers", activeGlobalLayers],
          ["Active sliding-window layers", activeSwaLayers],
          [
            "Retained sliding-window tokens",
            retainedSwaTokens,
            "min(tokens, sliding_window) for every SWA layer.",
          ],
          [
            "SConv state included",
            includeSconvState ? "Yes" : "No",
            "Adds the short-convolution state required to resume a cached Inkling prefix.",
          ],
          [
            "SConv checkpoint interval",
            formatStateCheckpointInterval(sconvCheckpointInterval),
            "The infinity default stores one final checkpoint; otherwise this is the number of tokens represented by each retained checkpoint.",
          ],
          [
            "SConv checkpoints per sequence",
            sconvCheckpointCount,
            "One when the interval is infinity; otherwise ceil(tokens / sconv_checkpoint_interval).",
          ],
          [
            "SConv elements per checkpoint",
            sconvElementsPerCheckpoint,
            "Logical K, V, attention-output, and MLP-output history across active main and draft layers.",
          ],
          [
            "SConv bytes per checkpoint",
            sconvBytesPerCheckpoint,
            "One logical BF16 short-convolution checkpoint.",
          ],
          [
            "SConv checkpoint bytes per sequence",
            sconvCheckpointBytesPerSequence,
            "Retained checkpoint count multiplied by bytes per checkpoint.",
          ],
          ["SConv state bytes", sconvBytesPerElement, "Current production serving implementations retain this state in BF16."],
          [
            "Model fields",
            fieldList(model, [
              "num_hidden_layers",
              "full_attention_layers",
              "sliding_attention_layers",
              "hidden_size",
              "num_key_value_heads",
              "head_dim",
              "swa_num_key_value_heads",
              "swa_head_dim",
              "sliding_window",
              "sconv_kernel_size",
              "num_nextn_predict_layers",
              "draft_full_attention_layers",
              "draft_sliding_attention_layers",
              "default_sconv_checkpoint_interval",
            ]),
          ],
        ],
      };
    }

    if (formula === "qwen_linear_full_hybrid") {
      const layers = getField(model, "num_hidden_layers");
      const fullLayers = getField(model, "full_attention_layers");
      const linearLayers = getField(model, "linear_attention_layers");
      const kvHeads = getField(model, "num_key_value_heads");
      const headDim = getField(model, "head_dim");
      const linearKeyHeads = getField(model, "linear_num_key_heads");
      const linearKeyDim = getField(model, "linear_key_head_dim");
      const linearValueHeads = getField(model, "linear_num_value_heads");
      const linearValueDim = getField(model, "linear_value_head_dim");
      const linearConvKernel = getField(model, "linear_conv_kernel_dim");
      const mtpLayers = optionalField(model, "mtp_num_hidden_layers", 0);
      const recurrentBytesPerElement = toPositiveNumber(
        settings && settings.qwenRecurrentStateBytesPerElement,
        QWEN_LINEAR_RECURRENT_BYTES_PER_ELEMENT,
      );
      const recurrentPrecisionLabel =
        (settings && settings.qwenRecurrentStatePrecisionLabel) || "FP32";
      const linearCheckpointInterval = parseStateCheckpointInterval(
        settings && settings.qwenCheckpointInterval,
        defaultQwenCheckpointInterval(model),
      );
      const linearCheckpointCount = includeLinearAttentionState
        ? Number.isFinite(linearCheckpointInterval)
          ? Math.ceil(tokens / linearCheckpointInterval)
          : 1
        : 0;
      const elementsPerToken = fullLayers * 2 * kvHeads * headDim;
      const fullElements = elementsPerToken * tokens;
      const linearConvElements =
        linearLayers *
        (linearConvKernel - 1) *
        (2 * linearKeyHeads * linearKeyDim + linearValueHeads * linearValueDim);
      const linearRecurrentElements = linearLayers * linearValueHeads * linearKeyDim * linearValueDim;
      const linearStateBytesPerCheckpoint =
        linearConvElements * QWEN_LINEAR_CONV_BYTES_PER_ELEMENT +
        linearRecurrentElements * recurrentBytesPerElement;
      const linearCheckpointBytesPerSequence =
        linearCheckpointCount * linearStateBytesPerCheckpoint;
      const byteGroups = [{ role: "kv", label: "Full-attention KV cache", elements: fullElements }];
      const formulaRows = [
        {
          name: "full_kv_bytes",
          expression: "tokens x sequences x full_attention_layers x 2 x num_key_value_heads x head_dim x precision_bytes",
          description: "Only Qwen full-attention layers are counted as ordinary token-linear KV cache.",
        },
      ];

      if (includeLinearAttentionState) {
        byteGroups.push({
          role: "linear_state",
          label: "Linear-attention checkpoint state",
          bytesPerSequence: linearCheckpointBytesPerSequence,
        });
        formulaRows.push(
          {
            name: "linear_state_checkpoint_count",
            expression:
              "interval is infinity ? 1 : ceil(tokens / linear_state_checkpoint_interval)",
            description:
              "The Prompt-End default stores one final GDN state; finite intervals also count a final partial interval.",
          },
          {
            name: "linear_conv_state_bytes",
            expression:
              "sequences x checkpoint_count x linear_attention_layers x (linear_conv_kernel_dim - 1) x (2 x linear_num_key_heads x linear_key_head_dim + linear_num_value_heads x linear_value_head_dim) x 2",
            description:
              "Each retained checkpoint stores the three-token BF16/FP16 short-convolution history when kernel size is four.",
          },
          {
            name: "linear_recurrent_state_bytes",
            expression:
              "sequences x checkpoint_count x linear_attention_layers x linear_num_value_heads x linear_value_head_dim x linear_key_head_dim x recurrent_state_bytes",
            description:
              "Each retained checkpoint stores the Qwen Gated DeltaNet recurrent matrices at the selected recurrent-state precision.",
          },
          {
            name: "total_bytes",
            expression: "full_kv_bytes + linear_conv_state_bytes + linear_recurrent_state_bytes",
            description:
              "Ordinary full-attention KV plus retained Qwen linear-attention checkpoints.",
          },
        );
      } else {
        formulaRows.push(
          {
            name: "linear_attention_state",
            expression: "excluded unless Include linear-attention state is enabled",
            description: "Qwen linear-attention / Gated DeltaNet layers keep non-standard recurrent and convolution state rather than ordinary per-token K/V tensors.",
          },
          {
            name: "total_bytes",
            expression: "full_kv_bytes",
            description: "Capacity-planning estimate for reusable ordinary KV payload only.",
          },
        );
      }

      return {
        elementsPerSequence:
          fullElements +
          linearCheckpointCount * (linearConvElements + linearRecurrentElements),
        elementsPerToken,
        formulaLabel: FORMULA_LABELS[formula],
        formulaText:
          "full_kv_bytes = tokens * sequences * full_attention_layers * 2 * num_key_value_heads * head_dim * precision_bytes\nlinear_state_checkpoint_count = interval_is_infinity ? 1 : ceil(tokens / interval)\ntotal_bytes = full_kv_bytes + optional_linear_state_checkpoint_bytes",
        formulaRows,
        note: includeLinearAttentionState
          ? "Includes retained Qwen3.5/3.6/3.8 Gated DeltaNet checkpoints. Active, ping-pong, and speculative runtime buffers are excluded."
          : "Qwen3.5/3.6/3.8 linear-attention recurrent/conv state is not ordinary per-token KV and is excluded. Enable the linear-attention state option to include retained checkpoints.",
        byteGroups,
        components: [
          ["Main layers", layers],
          ["Full-attention layers", fullLayers, "Layers counted as ordinary token-linear KV cache."],
          ["Linear-attention layers", linearLayers, "Qwen Gated DeltaNet layers whose runtime state is optional and does not grow linearly with token count."],
          ["Linear state included", includeLinearAttentionState ? "Yes" : "No", "When enabled, adds retained convolution and recurrent checkpoints for Qwen linear-attention layers."],
          ["GDN checkpoint interval", formatStateCheckpointInterval(linearCheckpointInterval), "Prompt-End stores one final checkpoint; otherwise this is the number of tokens represented by each retained checkpoint."],
          ["GDN checkpoints per sequence", linearCheckpointCount, "One for Prompt-End; otherwise ceil(tokens / linear_state_checkpoint_interval)."],
          ["Linear conv elements per checkpoint", linearConvElements, "Convolution-state scalar elements across all GDN layers, using kernel size minus one history positions."],
          ["Linear recurrent elements per checkpoint", linearRecurrentElements, "Recurrent-state scalar elements across all GDN layers."],
          ["Linear state bytes per checkpoint", linearStateBytesPerCheckpoint, "One convolution plus recurrent GDN checkpoint."],
          ["Linear checkpoint bytes per sequence", linearCheckpointBytesPerSequence, "Retained checkpoint count multiplied by bytes per checkpoint."],
          ["Linear conv-state bytes", QWEN_LINEAR_CONV_BYTES_PER_ELEMENT, "Short-convolution history remains BF16 / FP16."],
          ["Linear recurrent-state precision", recurrentPrecisionLabel, "Serving stacks can retain the recurrent matrices in BF16 / FP16 or FP32."],
          ["Linear recurrent-state bytes", recurrentBytesPerElement],
          ["MTP layers not included", mtpLayers, "Qwen3.5/3.6/3.8 MTP state belongs to the optional speculative path and is excluded from this base cache estimate."],
          ["Per-token elements", elementsPerToken, "Ordinary full-attention KV scalar elements per token before multiplying by precision bytes."],
          ["Model fields", fieldList(model, ["num_hidden_layers", "full_attention_layers", "linear_attention_layers", "num_key_value_heads", "head_dim", "linear_num_key_heads", "linear_key_head_dim", "linear_num_value_heads", "linear_value_head_dim", "linear_conv_kernel_dim", "mamba_ssm_dtype", "mtp_num_hidden_layers", "default_linear_state_checkpoint_interval"])],
        ],
      };
    }

    if (formula === "mixed_full_sliding_gqa") {
      const layers = getField(model, "num_hidden_layers");
      const fullLayers = getField(model, "full_attention_layers");
      const slidingLayers = getField(model, "sliding_attention_layers");
      const kvHeads = getField(model, "num_key_value_heads");
      const headDim = getField(model, "head_dim");
      const fullKvHeads = optionalField(model, "num_global_key_value_heads", kvHeads);
      const fullHeadDim = optionalField(model, "global_head_dim", headDim);
      const fullVHeadDim = optionalField(
        model,
        "global_v_head_dim",
        optionalField(model, "v_head_dim", fullHeadDim),
      );
      const slidingKvHeads = optionalField(
        model,
        "swa_num_key_value_heads",
        optionalField(model, "sliding_num_key_value_heads", kvHeads),
      );
      const slidingHeadDim = optionalField(
        model,
        "swa_head_dim",
        optionalField(model, "sliding_head_dim", headDim),
      );
      const slidingVHeadDim = optionalField(
        model,
        "swa_v_head_dim",
        optionalField(model, "sliding_v_head_dim", optionalField(model, "v_head_dim", slidingHeadDim)),
      );
      const slidingWindow = getField(model, "sliding_window");
      const retainedSlidingTokens = Math.min(tokens, slidingWindow);
      const fullElements = tokens * fullLayers * fullKvHeads * (fullHeadDim + fullVHeadDim);
      const slidingElements = retainedSlidingTokens * slidingLayers * slidingKvHeads * (slidingHeadDim + slidingVHeadDim);
      const elementsPerSequence = fullElements + slidingElements;
      return {
        elementsPerSequence,
        elementsPerToken: elementsPerSequence / tokens,
        formulaLabel: FORMULA_LABELS[formula],
        formulaText:
          "full_kv_bytes = tokens * sequences * full_layers * full_kv_heads * (full_head_dim + full_v_head_dim) * precision_bytes\nsliding_kv_bytes = min(tokens, sliding_window) * sequences * sliding_layers * sliding_kv_heads * (sliding_head_dim + sliding_v_head_dim) * precision_bytes\ntotal_bytes = full_kv_bytes + sliding_kv_bytes",
        formulaRows: [
          {
            name: "full_kv_bytes",
            expression: "tokens x sequences x full_layers x full_kv_heads x (full_head_dim + full_v_head_dim) x precision_bytes",
            description: "Full-attention layers retain ordinary KV for all cached tokens.",
          },
          {
            name: "sliding_kv_bytes",
            expression: "min(tokens, sliding_window) x sequences x sliding_layers x sliding_kv_heads x (sliding_head_dim + sliding_v_head_dim) x precision_bytes",
            description: "Sliding-attention layers retain only the local window for each sequence.",
          },
          {
            name: "total_bytes",
            expression: "full_kv_bytes + sliding_kv_bytes",
            description: "Combined reusable full-attention and sliding-window KV payload.",
          },
        ],
        note: "Production estimate counts text-generation KV payload only. Vision/audio encoder activations and allocator memory are excluded.",
        byteGroups: [
          { role: "kv", label: "Full-attention KV cache", elements: fullElements },
          { role: "kv", label: "Sliding-window KV cache", elements: slidingElements },
        ],
        components: [
          ["Main layers", layers],
          ["Stored layers", optionalField(model, "stored_layers", fullLayers + slidingLayers), "KV-producing layers counted after model-specific KV sharing, when configured."],
          ["Full-attention layers", fullLayers, "Layers whose KV grows with total cached tokens."],
          ["Sliding-attention layers", slidingLayers, "Layers whose KV is capped by the sliding window."],
          ["Retained sliding tokens", retainedSlidingTokens, "min(tokens, sliding_window) for sliding-attention layers."],
          ["Full K+V dims", fullHeadDim + fullVHeadDim, "Key plus value dimensions per full-attention KV head."],
          ["Sliding K+V dims", slidingHeadDim + slidingVHeadDim, "Key plus value dimensions per sliding-window KV head."],
          ["Full-attention elements", fullElements, "Full-attention scalar KV elements before applying precision bytes."],
          ["Sliding-window elements", slidingElements, "Sliding-window scalar KV elements before applying precision bytes."],
          ["Model fields", fieldList(model, ["num_hidden_layers", "full_attention_layers", "sliding_attention_layers", "num_key_value_heads", "num_global_key_value_heads", "head_dim", "global_head_dim", "v_head_dim", "global_v_head_dim", "swa_num_key_value_heads", "swa_head_dim", "swa_v_head_dim", "sliding_window"])],
        ],
      };
    }

    if (formula === "minimax_msa") {
      const layers = getField(model, "num_hidden_layers");
      const fullLayers = getField(model, "full_attention_layers");
      const sparseLayers = getField(model, "sparse_attention_layers");
      const kvHeads = getField(model, "num_key_value_heads");
      const headDim = getField(model, "head_dim");
      const indexDim = getField(model, "index_head_dim");
      const indexHeads = optionalField(model, "index_n_heads", kvHeads);
      const blockSize = getField(model, "index_block_size");
      const topkBlocks = getField(model, "index_topk_blocks");
      const localBlocks = optionalField(model, "index_local_blocks", 0);
      const mtpModules = optionalField(model, "num_mtp_modules", 0);
      const nextnLayers = optionalField(model, "num_nextn_predict_layers", 0);
      const kvElementsPerToken = layers * 2 * kvHeads * headDim;
      const indexerElementsPerToken = sparseLayers * indexDim;
      const elementsPerToken = kvElementsPerToken + indexerElementsPerToken;
      const kvElements = kvElementsPerToken * tokens;
      const indexerElements = indexerElementsPerToken * tokens;

      return {
        elementsPerSequence: elementsPerToken * tokens,
        elementsPerToken,
        formulaLabel: FORMULA_LABELS[formula],
        formulaText:
          "kv_bytes = tokens * sequences * layers * 2 * num_key_value_heads * head_dim * kv_precision_bytes\nindexer_bytes = tokens * sequences * sparse_attention_layers * index_head_dim * indexer_precision_bytes\ntotal_bytes = kv_bytes + indexer_bytes",
        formulaRows: [
          {
            name: "kv_bytes",
            expression: "tokens x sequences x layers x 2 x num_key_value_heads x head_dim x kv_precision_bytes",
            description: "MiniMax M3 stores ordinary main K/V cache for both full-attention and sparse-attention layers.",
          },
          {
            name: "indexer_bytes",
            expression: "tokens x sequences x sparse_attention_layers x index_head_dim x indexer_precision_bytes",
            description: "MSA sparse layers keep a key-only side cache for the lightning indexer; index_n_heads affects scoring, not the stored index-key cache width.",
          },
          {
            name: "total_bytes",
            expression: "kv_bytes + indexer_bytes",
            description: "Combined main KV cache and MSA index-key side cache.",
          },
        ],
        note: "MiniMax Sparse Attention (MSA) uses a lightweight indexer to pick the most relevant KV blocks for each query, so long-context attention can read a sparse subset of the cached tokens while keeping a separate indexer cache for block selection.",
        byteGroups: [
          { role: "kv", label: "KV cache", elements: kvElements },
          { role: "indexer", label: "Indexer cache", elements: indexerElements },
        ],
        components: [
          ["Main layers", layers],
          ["Full-attention layers", fullLayers, "Dense/full-attention layers without the MSA indexer branch."],
          ["Sparse-attention layers", sparseLayers, "MSA layers that add a key-only indexer side cache."],
          ["KV elements per token", kvElementsPerToken, "Main K/V elements per token before applying KV precision."],
          ["Indexer elements per token", indexerElementsPerToken, "MSA index-key elements per token before applying indexer precision."],
          ["Index heads", indexHeads, "Indexer query heads used for scoring selected KV blocks; the stored index-key cache is key-only."],
          ["Index block size", blockSize, "Number of tokens per sparse-selection block. This is not a sliding-window cache cap."],
          ["Top-k blocks", topkBlocks, "Sparse blocks selected by the indexer for each query."],
          ["Local blocks", localBlocks, "Recent local blocks always visible to the sparse attention path."],
          ["MTP modules not included", mtpModules, "The public MiniMax M3 checkpoint/config exposes MTP fields, but bundled MTP weights are not modeled in the base serving path."],
          ["Next-N layers not included", nextnLayers, "Config field retained for traceability; draft KV is not included for MiniMax M3."],
          ["Model fields", fieldList(model, ["num_hidden_layers", "full_attention_layers", "sparse_attention_layers", "num_key_value_heads", "head_dim", "index_head_dim", "index_n_heads", "index_block_size", "index_topk_blocks", "index_local_blocks", "indexer_fixed_precision_id"])],
        ],
      };
    }

    if (formula === "deepseek_v4_hybrid") {
      const headDim = getField(model, "head_dim");
      const indexDim = getField(model, "index_head_dim");
      const slidingWindow = getField(model, "sliding_window");
      const layers = getField(model, "num_hidden_layers");
      const allRatios = Array.isArray(model.fields.compress_ratios)
        ? model.fields.compress_ratios.map((ratio) => Number(ratio))
        : [];
      const mainRatios = allRatios.slice(0, layers);
      const draftRatios = allRatios.slice(layers);
      const activeRatios = includeDraftKvCache ? mainRatios.concat(draftRatios) : mainRatios;

      if (!activeRatios.length) {
        throw new Error(`Model ${model.id} is missing compress_ratios`);
      }

      let windowElements = 0;
      let compressedElements = 0;
      let indexerElements = 0;
      const ratioZeroLayers = countByValue(activeRatios, 0);
      const ratioFourLayers = countByValue(activeRatios, 4);
      const ratio128Layers = countByValue(activeRatios, 128);
      const ratioZeroElements = ratioZeroLayers * slidingWindow * headDim;

      activeRatios.forEach((ratio) => {
        windowElements += slidingWindow * headDim;
        if (ratio > 0) {
          compressedElements += Math.floor(tokens / ratio) * headDim;
        }
        if (ratio === 4) {
          indexerElements += Math.floor(tokens / 4) * indexDim;
        }
      });

      const attentionElements = windowElements + compressedElements;
      const elementsPerSequence = attentionElements + indexerElements;
      return {
        elementsPerSequence,
        elementsPerToken: elementsPerSequence / tokens,
        formulaLabel: FORMULA_LABELS[formula],
        formulaText:
          "sliding_kv_bytes = active_layers * sliding_window * head_dim * kv_precision_bytes\ncompressed_kv_bytes = sum_ratio>0(floor(tokens / compress_ratio) * head_dim) * kv_precision_bytes\nkv_bytes = sliding_kv_bytes + compressed_kv_bytes\nindexer_bytes = ratio4_layers * floor(tokens / 4) * index_head_dim * indexer_precision_bytes\ntotal_bytes = sequences * (kv_bytes + indexer_bytes)",
        formulaRows: [
          {
            name: "sliding_kv_bytes",
            expression: "active_layers x sliding_window x head_dim x kv_precision_bytes",
            description: "Includes ratio=0 layers. Ratio=0 layers only contribute this fixed sliding-window KV and do not add compressed KV slots.",
          },
          {
            name: "compressed_kv_bytes",
            expression: "sum over ratio>0 layers: floor(tokens / compress_ratio) x head_dim x kv_precision_bytes",
            description: "Compressed KV cache from layers whose compress_ratio is greater than zero; each layer keeps floor(tokens / compress_ratio) compressed slots.",
          },
          {
            name: "kv_bytes",
            expression: "sliding_kv_bytes + compressed_kv_bytes",
            description: "Main DeepSeek V4 KV cache before adding the separate indexer cache.",
          },
          {
            name: "indexer_bytes",
            expression:
              "ratio4_layers x floor(tokens / 4) x index_head_dim x indexer_precision_bytes",
            description: "Ratio=4 layers keep an extra compressed indexer cache that can use a separate precision.",
          },
          {
            name: "total_bytes",
            expression: "sequences x (kv_bytes + indexer_bytes)",
            description: "Combined DeepSeek V4 cache payload for all concurrent sequences.",
          },
        ],
        note: "Production estimate uses the official sliding-window/compressed-cache layout. The default DeepSeek V4 setting uses FP8 attention cache and FP4 indexer cache.",
        byteGroups: [
          { role: "kv", label: "KV cache", elements: attentionElements },
          { role: "indexer", label: "Indexer cache", elements: indexerElements },
        ],
        components: [
          ["Main layers", mainRatios.length],
          ["Draft layers included", includeDraftKvCache ? draftRatios.length : 0, "Extra MTP/draft layers after the main transformer layers. In DeepSeek V4 configs these are ratio=0 layers."],
          ["Ratio=4 layers", ratioFourLayers, "Layers whose compressed cache ratio is 4; these layers also carry indexer cache."],
          ["Ratio=128 layers", ratio128Layers, "Layers whose compressed cache keeps floor(tokens / 128) compressed KV slots."],
          ["Ratio=0 layers", ratioZeroLayers, "Layers with no compressed KV segment; they keep only the sliding-window KV cache."],
          ["Ratio=0 KV elements", ratioZeroElements, "The ratio=0 contribution: ratio0_layers x sliding_window x head_dim."],
          ["Sliding-window elements", windowElements, "Per-layer local KV reserve: sliding_window x head_dim, summed across active layers."],
          ["Compressed elements", compressedElements, "Compressed KV elements from layers with compress_ratio greater than zero."],
          ["KV elements", attentionElements, "Sliding-window plus compressed attention cache elements before applying KV precision."],
          ["Indexer elements", indexerElements, "Compressed indexer elements from ratio=4 layers before applying indexer precision."],
        ],
      };
    }

    throw new Error(`Unsupported formula: ${formula}`);
  }

  function bytesPerElementForGroup(precision, role) {
    if ((role === "kv" || role === "attention") && Number.isFinite(precision.kvBytesPerElement)) {
      return precision.kvBytesPerElement;
    }
    if (role === "indexer" && Number.isFinite(precision.indexerBytesPerElement)) {
      return precision.indexerBytesPerElement;
    }
    if (Number.isFinite(precision.bytesPerElement)) return precision.bytesPerElement;
    throw new Error(`Precision ${precision.label} does not define bytes for ${role} cache`);
  }

  function calculateCacheGroups(elementPlan, precision) {
    const groups = elementPlan.byteGroups || [{ role: "cache", elements: elementPlan.elementsPerSequence }];
    return groups.map((group) => ({
      role: group.role,
      label: group.label || "KV cache",
      elements: group.elements,
      bytesPerSequence: Number.isFinite(group.bytesPerSequence)
        ? group.bytesPerSequence
        : group.elements * bytesPerElementForGroup(precision, group.role),
    }));
  }

  function precisionComponents(precision) {
    if (
      Number.isFinite(precision.kvBytesPerElement) ||
      Number.isFinite(precision.indexerBytesPerElement)
    ) {
      return [
        ["KV precision bytes", precision.kvBytesPerElement],
        ["Indexer precision bytes", precision.indexerBytesPerElement],
      ];
    }
    return [["Precision bytes", precision.bytesPerElement]];
  }

  function calculate(model, input, options) {
    const tokens = toPositiveInteger(input.tokens, model.default_tokens || 4096);
    const sequences = toPositiveInteger(input.sequences, 1);
    const tensorParallel = toPositiveInteger(input.tensorParallel, 1);
    const precisionId = input.precision || defaultPrecisionId(model, options);
    const precision = getPrecisionProfile(
      precisionId,
      options,
      defaultPrecisionId(model, options),
    );
    const indexerPrecision = hasIndexerCache(model)
      ? getIndexerPrecisionProfile(
          input.indexerPrecision || defaultIndexerPrecisionId(model, options, precisionId),
          options,
          model,
          precisionId,
        )
      : null;
    const recurrentStatePrecision = hasQwenCheckpointInterval(model)
      ? getRecurrentStatePrecisionProfile(
          input.recurrentStatePrecision ||
            defaultRecurrentStatePrecisionId(model, options),
          options,
          model,
        )
      : null;
    const cachePrecision = indexerPrecision
      ? {
          label: precision.label,
          bytesPerElement: precision.bytesPerElement,
          kvBytesPerElement: precision.bytesPerElement,
          indexerBytesPerElement: indexerPrecision.bytesPerElement,
        }
      : precision;
    const elementPlan = calculateElementsPerSequence(model, tokens, {
      includeDraftKvCache: hasDraftKvCache(model) && toBoolean(input.includeDraftKvCache),
      includeLinearAttentionState: hasLinearAttentionState(model) && toBoolean(input.includeLinearAttentionState),
      includeSconvState:
        hasSconvState(model) &&
        (typeof input.includeSconvState === "undefined"
          ? Boolean(model.fields.default_include_sconv_state)
          : toBoolean(input.includeSconvState)),
      qwenRecurrentStateBytesPerElement: recurrentStatePrecision
        ? recurrentStatePrecision.bytesPerElement
        : undefined,
      qwenRecurrentStatePrecisionLabel: recurrentStatePrecision
        ? recurrentStatePrecision.label
        : undefined,
      qwenCheckpointInterval: hasQwenCheckpointInterval(model)
        ? input.qwenCheckpointPolicy === STATE_CHECKPOINT_POLICY_FIXED_INTERVAL
          ? parseStateCheckpointInterval(
              input.qwenCheckpointInterval,
              defaultQwenCheckpointInterval(model),
            )
          : Infinity
        : undefined,
      kdaCheckpointInterval: hasKdaCheckpointInterval(model)
        ? input.kdaCheckpointPolicy === STATE_CHECKPOINT_POLICY_FIXED_INTERVAL
          ? parseStateCheckpointInterval(
              input.kdaCheckpointInterval,
              defaultKdaCheckpointInterval(model),
            )
          : Infinity
        : undefined,
      sconvCheckpointInterval: hasSconvState(model)
        ? input.sconvCheckpointPolicy === STATE_CHECKPOINT_POLICY_FIXED_INTERVAL
          ? parseStateCheckpointInterval(
              input.sconvCheckpointInterval,
              defaultSconvCheckpointInterval(model),
            )
          : Infinity
        : undefined,
    });
    const cacheGroupsPerSequence = calculateCacheGroups(elementPlan, cachePrecision);
    const bytesPerSequence = cacheGroupsPerSequence.reduce((total, group) => total + group.bytesPerSequence, 0);
    const totalBytes = bytesPerSequence * sequences;
    const cacheGroups = cacheGroupsPerSequence.map((group) => ({
      role: group.role,
      label: group.label,
      elements: Number.isFinite(group.elements) ? group.elements * sequences : undefined,
      bytes: group.bytesPerSequence * sequences,
    }));
    const kvBytes = cacheGroups
      .filter((group) => group.role === "kv" || group.role === "attention" || group.role === "cache")
      .reduce((total, group) => total + group.bytes, 0);
    const indexerBytes = cacheGroups
      .filter((group) => group.role === "indexer")
      .reduce((total, group) => total + group.bytes, 0);
    const hitRateBytesPerToken = Number.isFinite(
      elementPlan.hitRateElementsPerToken,
    )
      ? elementPlan.hitRateElementsPerToken *
        bytesPerElementForGroup(cachePrecision, "kv")
      : undefined;

    return {
      modelId: model.id,
      modelLabel: model.label,
      precisionLabel: precision.label,
      indexerPrecisionLabel: indexerPrecision ? indexerPrecision.label : undefined,
      recurrentStatePrecisionLabel: recurrentStatePrecision
        ? recurrentStatePrecision.label
        : undefined,
      bytesPerElement: precision.bytesPerElement,
      tokens,
      sequences,
      totalCachedTokens: tokens * sequences,
      tensorParallel,
      totalBytes,
      totalGB: totalBytes / BYTES_PER_GB,
      totalGiB: totalBytes / BYTES_PER_GIB,
      kvBytes,
      kvGiB: kvBytes / BYTES_PER_GIB,
      indexerBytes,
      indexerGiB: indexerBytes / BYTES_PER_GIB,
      bytesPerSequence,
      bytesPerToken: bytesPerSequence / tokens,
      perDeviceBytes: totalBytes / tensorParallel,
      perDeviceGiB: totalBytes / tensorParallel / BYTES_PER_GIB,
      hitRateBytesPerToken,
      cacheGroups,
      elementPlan,
      components: elementPlan.components.concat(precisionComponents(cachePrecision)),
    };
  }

  function formatNumber(value, digits) {
    return Number(value).toLocaleString(undefined, {
      maximumFractionDigits: digits,
      minimumFractionDigits: digits,
    });
  }

  function formatBytes(bytes) {
    if (bytes >= BYTES_PER_GIB) return `${formatNumber(bytes / BYTES_PER_GIB, RESULT_DIGITS)} GiB`;
    if (bytes >= 1024 ** 2) return `${formatNumber(bytes / 1024 ** 2, RESULT_DIGITS)} MiB`;
    if (bytes >= 1024) return `${formatNumber(bytes / 1024, RESULT_DIGITS)} KiB`;
    return `${formatNumber(bytes, RESULT_DIGITS)} B`;
  }

  function modelFamily(model) {
    const family = model.family || "Other";
    return family.indexOf("Qwen") === 0 ? "Qwen" : family;
  }

  function groupModels(models) {
    return models.reduce((groups, model) => {
      const key = modelFamily(model);
      if (!groups[key]) groups[key] = [];
      groups[key].push(model);
      return groups;
    }, {});
  }

  function modelById(models, id) {
    return models.find((model) => model.id === id) || models[0];
  }

  function setText(root, selector, value) {
    const node = root.querySelector(selector);
    if (node) node.textContent = value;
  }

  function appendHelp(parent, description) {
    if (!description) return;
    const help = document.createElement("button");
    help.type = "button";
    help.className = "kv-help";
    help.textContent = "?";
    help.setAttribute("aria-label", description);
    help.dataset.kvTooltip = description;
    help.addEventListener("click", (event) => {
      event.preventDefault();
      const wasOpen = help.dataset.kvOpen === "true";
      document.querySelectorAll(".kv-help[data-kv-open='true']").forEach((node) => {
        delete node.dataset.kvOpen;
      });
      if (!wasOpen) help.dataset.kvOpen = "true";
    });
    help.addEventListener("blur", () => {
      delete help.dataset.kvOpen;
    });
    help.addEventListener("mouseleave", () => {
      delete help.dataset.kvOpen;
    });
    parent.appendChild(help);
  }

  function updateInlineHelp(parent, description) {
    if (!parent || !description) return;
    const help = parent.querySelector(".kv-help");
    if (!help) {
      appendHelp(parent, description);
      return;
    }
    help.setAttribute("aria-label", description);
    help.dataset.kvTooltip = description;
  }

  function renderMetricCard(label, value) {
    const item = document.createElement("div");
    const key = document.createElement("span");
    key.textContent = label;
    const val = document.createElement("strong");
    val.textContent = value;
    item.append(key, val);
    return item;
  }

  function renderMetrics(root, result) {
    const list = root.querySelector("[data-kv-metrics]");
    if (!list) return;
    list.innerHTML = "";

    const metrics = [];
    if (result.indexerPrecisionLabel) {
      metrics.push([
        "KV cache size",
        formatBytes(result.kvBytes),
      ]);
      metrics.push([
        "Indexer cache size",
        formatBytes(result.indexerBytes),
      ]);
    } else if (result.cacheGroups.length > 1) {
      result.cacheGroups.forEach((group) => {
        metrics.push([
          `${group.label} size`,
          formatBytes(group.bytes),
        ]);
      });
    }
    const includesFixedState = result.cacheGroups.some(
      (group) => group.role === "linear_state" || group.role === "sconv_state",
    );
    metrics.push([
      includesFixedState ? "Amortized size per token" : "Per token size",
      formatBytes(result.bytesPerToken),
    ]);
    if (Number.isFinite(result.hitRateBytesPerToken)) {
      metrics.push([
        "Reusable MLA per token",
        formatBytes(result.hitRateBytesPerToken),
      ]);
    }

    metrics.forEach(([label, value]) => {
      list.appendChild(renderMetricCard(label, value));
    });
  }

  function renderComponents(root, result) {
    const list = root.querySelector("[data-kv-components]");
    if (!list) return;
    list.innerHTML = "";
    result.components.forEach(([label, value, description]) => {
      const item = document.createElement("div");
      item.className = "kv-breakdown-row";
      const key = document.createElement("span");
      key.textContent = label;
      appendHelp(key, description);
      const val = document.createElement("strong");
      val.textContent = typeof value === "number" ? formatNumber(value, Number.isInteger(value) ? 0 : 2) : value;
      item.append(key, val);
      list.appendChild(item);
    });
  }

  function renderFormulaRows(root, elementPlan) {
    const list = root.querySelector("[data-kv-formula-rows]");
    if (!list) return;
    const rows = Array.isArray(elementPlan.formulaRows) && elementPlan.formulaRows.length
      ? elementPlan.formulaRows
      : [{ name: "total_bytes", expression: elementPlan.formulaText }];

    list.innerHTML = "";
    rows.forEach((row) => {
      const item = document.createElement("div");
      item.className = "kv-formula-row";

      const name = document.createElement("span");
      name.className = "kv-formula-name";
      name.textContent = row.name;
      appendHelp(name, row.description);

      const equals = document.createElement("span");
      equals.className = "kv-formula-equals";
      equals.textContent = "=";

      const expression = document.createElement("span");
      expression.className = "kv-formula-expression";
      expression.textContent = row.expression;

      item.append(name, equals, expression);
      list.appendChild(item);
    });
  }

  function sortedModelFamilies(models) {
    return Object.keys(groupModels(models)).sort();
  }

  function modelsForFamily(models, family) {
    return models
      .filter((model) => modelFamily(model) === family);
  }

  function populateModelFamilies(select, models, preferredFamily) {
    if (!select) return;
    const families = sortedModelFamilies(models);
    select.innerHTML = "";
    families.forEach((family) => {
      const item = document.createElement("option");
      item.value = family;
      item.textContent = family;
      select.appendChild(item);
    });
    select.value = families.includes(preferredFamily) ? preferredFamily : families[0];
  }

  function populateModelsForFamily(select, models, family, preferredModelId) {
    if (!select) return;
    const familyModels = modelsForFamily(models, family);
    select.innerHTML = "";
    familyModels.forEach((model) => {
      const item = document.createElement("option");
      item.value = model.id;
      item.textContent = model.label;
      select.appendChild(item);
    });
    const ids = familyModels.map((model) => model.id);
    select.value = ids.includes(preferredModelId) ? preferredModelId : ids[0];
  }

  function rawPrecisionOptions(data) {
    return data.precision_options || [];
  }

  function rawIndexerPrecisionOptions(data) {
    return data.indexer_precision_options || data.precision_options || [];
  }

  function rawRecurrentStatePrecisionOptions(data) {
    return data.recurrent_state_precision_options || [
      { id: "bf16_fp16", label: "BF16 / FP16", bytes_per_element: 2 },
      { id: "fp32", label: "FP32", bytes_per_element: 4 },
    ];
  }

  function populateSelect(select, options, preferredValue) {
    if (!select) return;
    select.innerHTML = "";
    options.forEach((option) => {
      const item = document.createElement("option");
      item.value = option.id;
      item.textContent = option.label;
      select.appendChild(item);
    });
    const values = options.map((option) => option.id);
    select.value = values.includes(preferredValue) ? preferredValue : values[0];
  }

  function populatePrecisionOptions(root, data, model) {
    const select = root.querySelector("[data-kv-input='precision']");
    const preferredValue = defaultPrecisionId(model, {
      precisionOptions: data.precision_options,
    });
    populateSelect(select, rawPrecisionOptions(data), preferredValue);
  }

  function populateIndexerPrecisionOptions(root, data, model) {
    const control = root.querySelector("[data-kv-indexer-control]");
    const select = root.querySelector("[data-kv-input='indexerPrecision']");
    const precisionSelect = root.querySelector("[data-kv-input='precision']");
    const showIndexerPrecision = hasIndexerCache(model);
    if (control) control.hidden = !showIndexerPrecision;
    if (select) select.disabled = false;
    if (showIndexerPrecision) {
      const fixedPrecisionId = fixedIndexerPrecisionId(model);
      const options = fixedPrecisionId
        ? rawIndexerPrecisionOptions(data).filter((option) => option.id === fixedPrecisionId)
        : rawIndexerPrecisionOptions(data);
      const preferredValue = defaultIndexerPrecisionId(
        model,
        { indexerPrecisionOptions: data.indexer_precision_options, precisionOptions: data.precision_options },
        precisionSelect ? precisionSelect.value : undefined,
      );
      populateSelect(select, options, preferredValue);
      if (select && fixedPrecisionId) select.disabled = true;
    }
  }

  function populateRecurrentStatePrecisionOptions(root, data, model) {
    const select = root.querySelector(
      "[data-kv-input='recurrentStatePrecision']",
    );
    const preferredValue = defaultRecurrentStatePrecisionId(model, {
      recurrentStatePrecisionOptions: data.recurrent_state_precision_options,
    });
    populateSelect(
      select,
      rawRecurrentStatePrecisionOptions(data),
      preferredValue,
    );
  }

  function syncDraftControl(root, model) {
    const control = root.querySelector("[data-kv-draft-control]");
    const checkbox = root.querySelector("[data-kv-input='includeDraftKvCache']");
    const showDraftControl = hasDraftKvCache(model);
    if (control) control.hidden = !showDraftControl;
    if (checkbox && !showDraftControl) checkbox.checked = false;
    if (checkbox && showDraftControl) checkbox.checked = false;
  }

  function syncLinearStateControl(root, model) {
    const control = root.querySelector("[data-kv-linear-state-control]");
    const checkbox = root.querySelector("[data-kv-input='includeLinearAttentionState']");
    const showLinearStateControl = hasLinearAttentionState(model);
    if (control) control.hidden = !showLinearStateControl;
    if (checkbox) {
      checkbox.checked = Boolean(
        showLinearStateControl &&
          model &&
          model.fields &&
          model.fields.default_include_linear_attention_state === true,
      );
    }
  }

  function syncRecurrentStatePrecisionControl(root, model) {
    const control = root.querySelector(
      "[data-kv-recurrent-state-precision-control]",
    );
    const includeState = checkboxValue(
      root.querySelector("[data-kv-input='includeLinearAttentionState']"),
    );
    if (control) {
      control.hidden = !(hasQwenCheckpointInterval(model) && includeState);
    }
  }

  function syncSconvStateControl(root, model) {
    const control = root.querySelector("[data-kv-sconv-state-control]");
    const checkbox = root.querySelector("[data-kv-input='includeSconvState']");
    const showSconvStateControl = hasSconvState(model);
    if (control) control.hidden = !showSconvStateControl;
    if (checkbox) {
      checkbox.checked = Boolean(
        showSconvStateControl &&
          model &&
          model.fields &&
          model.fields.default_include_sconv_state === true,
      );
    }
  }

  function stateCheckpointProfile(root, model) {
    if (hasKdaCheckpointInterval(model)) {
      return {
        label: "KDA",
        enabled: checkboxValue(
          root.querySelector("[data-kv-input='includeLinearAttentionState']"),
        ),
        promptHelp:
          "Assumes one KDA state is saved only at the end of each sequence.",
        intervalHelp:
          "Stores one KDA state checkpoint every N tokens. The final partial interval counts as one checkpoint.",
      };
    }
    if (hasQwenCheckpointInterval(model)) {
      return {
        label: "GDN",
        enabled: checkboxValue(
          root.querySelector("[data-kv-input='includeLinearAttentionState']"),
        ),
        promptHelp:
          "Assumes one Gated DeltaNet state is saved only at the end of each sequence.",
        intervalHelp:
          "Stores one Gated DeltaNet state checkpoint every N tokens. The final partial interval counts as one checkpoint.",
      };
    }
    if (hasSconvState(model)) {
      return {
        label: "SConv",
        enabled: checkboxValue(
          root.querySelector("[data-kv-input='includeSconvState']"),
        ),
        promptHelp:
          "Assumes one SConv state is saved only at the end of each sequence.",
        intervalHelp:
          "Stores one SConv state checkpoint every N tokens. The final partial interval counts as one checkpoint.",
      };
    }
    return null;
  }

  function syncStateCheckpointControl(root, model) {
    const policyControl = root.querySelector("[data-kv-state-policy-control]");
    const policyLabel = root.querySelector("[data-kv-state-policy-label]");
    const promptEndRadio = root.querySelector(
      "[data-kv-input='stateCheckpointPolicyPromptEnd']",
    );
    const fixedIntervalRadio = root.querySelector(
      "[data-kv-input='stateCheckpointPolicyFixedInterval']",
    );
    const intervalControl = root.querySelector("[data-kv-state-checkpoint-control]");
    const intervalLabel = root.querySelector("[data-kv-state-interval-label]");
    const promptHelp = root.querySelector("[data-kv-state-prompt-help]");
    const intervalHelp = root.querySelector("[data-kv-state-interval-help]");
    const input = root.querySelector("[data-kv-input='stateCheckpointInterval']");
    const profile = stateCheckpointProfile(root, model);
    const showPolicyControl = Boolean(profile && profile.enabled);
    if (policyControl) policyControl.hidden = !showPolicyControl;
    if (!promptEndRadio || !fixedIntervalRadio || !input) return;

    if (profile) {
      if (policyLabel) policyLabel.textContent = `${profile.label} Checkpoint Policy:`;
      if (intervalLabel) intervalLabel.textContent = `${profile.label} checkpoint interval`;
      updateInlineHelp(promptHelp, profile.promptHelp);
      updateInlineHelp(intervalHelp, profile.intervalHelp);
    }

    if (showPolicyControl && input.dataset.kvModelId !== model.id) {
      promptEndRadio.checked = true;
      fixedIntervalRadio.checked = false;
      input.value = "";
      input.dataset.kvModelId = model.id;
    }
    if (!showPolicyControl) {
      promptEndRadio.checked = true;
      fixedIntervalRadio.checked = false;
    }
    if (showPolicyControl && !promptEndRadio.checked && !fixedIntervalRadio.checked) {
      promptEndRadio.checked = true;
    }

    const showIntervalControl =
      showPolicyControl && fixedIntervalRadio.checked;
    if (intervalControl) intervalControl.hidden = !showIntervalControl;
    if (!showIntervalControl) input.value = "";
    input.disabled = !showIntervalControl;
  }

  function setCheckboxHelp(root) {
    root.querySelectorAll("[data-kv-inline-help]").forEach((node) => {
      appendHelp(node, node.getAttribute("data-kv-inline-help"));
    });
  }

  function hasInputValue(input) {
    return input && typeof input.value !== "undefined";
  }

  function inputValue(input, fallback) {
    return hasInputValue(input) ? input.value : fallback;
  }

  function checkboxValue(input) {
    return input && input.checked;
  }

  function addInputListeners(inputs, update) {
    Object.values(inputs).forEach((input) => {
      if (
        !input ||
        input === inputs.model ||
        input === inputs.modelFamily ||
        input === inputs.stateCheckpointPolicyPromptEnd ||
        input === inputs.stateCheckpointPolicyFixedInterval
      ) {
        return;
      }
      input.addEventListener("input", update);
      input.addEventListener("change", update);
    });
  }

  function initialize(root, data) {
    const models = data.models || [];
    if (!root || !models.length) return;
    setCheckboxHelp(root);

    const inputs = {
      modelFamily: root.querySelector("[data-kv-input='modelFamily']"),
      model: root.querySelector("[data-kv-input='model']"),
      tokens: root.querySelector("[data-kv-input='tokens']"),
      sequences: root.querySelector("[data-kv-input='sequences']"),
      precision: root.querySelector("[data-kv-input='precision']"),
      indexerPrecision: root.querySelector("[data-kv-input='indexerPrecision']"),
      recurrentStatePrecision: root.querySelector(
        "[data-kv-input='recurrentStatePrecision']",
      ),
      includeDraftKvCache: root.querySelector("[data-kv-input='includeDraftKvCache']"),
      includeLinearAttentionState: root.querySelector("[data-kv-input='includeLinearAttentionState']"),
      includeSconvState: root.querySelector("[data-kv-input='includeSconvState']"),
      stateCheckpointPolicyPromptEnd: root.querySelector(
        "[data-kv-input='stateCheckpointPolicyPromptEnd']",
      ),
      stateCheckpointPolicyFixedInterval: root.querySelector(
        "[data-kv-input='stateCheckpointPolicyFixedInterval']",
      ),
      stateCheckpointInterval: root.querySelector("[data-kv-input='stateCheckpointInterval']"),
    };

    function selectedModel() {
      return modelById(models, inputs.model.value);
    }

    function selectedFamily() {
      const model = selectedModel();
      return inputValue(inputs.modelFamily, modelFamily(model));
    }

    function syncModelDefaults() {
      const model = selectedModel();
      populatePrecisionOptions(root, data, model);
      populateIndexerPrecisionOptions(root, data, model);
      populateRecurrentStatePrecisionOptions(root, data, model);
      syncDraftControl(root, model);
      syncLinearStateControl(root, model);
      syncRecurrentStatePrecisionControl(root, model);
      syncSconvStateControl(root, model);
      syncStateCheckpointControl(root, model);
    }

    function update() {
      try {
        const model = selectedModel();
        syncRecurrentStatePrecisionControl(root, model);
        syncStateCheckpointControl(root, model);
        const stateCheckpointPolicy = checkboxValue(
          inputs.stateCheckpointPolicyFixedInterval,
        )
          ? STATE_CHECKPOINT_POLICY_FIXED_INTERVAL
          : "prompt_end";
        const stateCheckpointInterval = checkboxValue(
          inputs.stateCheckpointPolicyFixedInterval,
        )
          ? inputValue(inputs.stateCheckpointInterval, "")
          : Infinity;
        const result = calculate(
          model,
          {
            tokens: inputValue(inputs.tokens, model.default_tokens || 4096),
            sequences: inputValue(inputs.sequences, 1),
            precision: inputValue(inputs.precision, undefined),
            indexerPrecision: inputValue(inputs.indexerPrecision, undefined),
            recurrentStatePrecision: inputValue(
              inputs.recurrentStatePrecision,
              undefined,
            ),
            includeDraftKvCache: checkboxValue(inputs.includeDraftKvCache),
            includeLinearAttentionState: checkboxValue(inputs.includeLinearAttentionState),
            includeSconvState: checkboxValue(inputs.includeSconvState),
            qwenCheckpointPolicy: stateCheckpointPolicy,
            qwenCheckpointInterval: stateCheckpointInterval,
            kdaCheckpointPolicy: stateCheckpointPolicy,
            kdaCheckpointInterval: stateCheckpointInterval,
            sconvCheckpointPolicy: stateCheckpointPolicy,
            sconvCheckpointInterval: stateCheckpointInterval,
            tensorParallel: 1,
          },
          {
            precisionOptions: data.precision_options,
            indexerPrecisionOptions: data.indexer_precision_options,
            recurrentStatePrecisionOptions:
              data.recurrent_state_precision_options,
          },
        );

        setText(root, "[data-kv-output='totalGiB']", `${formatNumber(result.totalGiB, RESULT_DIGITS)} GiB`);
        setText(root, "[data-kv-output='totalGB']", `= ${formatNumber(result.totalGB, RESULT_DIGITS)} GB`);
        renderMetrics(root, result);
        setText(root, "[data-kv-output='formulaLabel']", result.elementPlan.formulaLabel);
        renderFormulaRows(root, result.elementPlan);
        setText(root, "[data-kv-output='cacheNote']", result.elementPlan.note);
        setText(root, "[data-kv-output='source']", model.source_url);
        const source = root.querySelector("[data-kv-source-link]");
        if (source) source.href = model.source_url;
        renderComponents(root, result);
        root.dataset.state = "ready";
      } catch (error) {
        root.dataset.state = "error";
        setText(root, "[data-kv-output='cacheNote']", error.message);
      }
    }

    const defaultModelId = inputs.model.value || models[0].id;
    const defaultModel = modelById(models, defaultModelId);
    populateModelFamilies(inputs.modelFamily, models, modelFamily(defaultModel));
    populateModelsForFamily(inputs.model, models, selectedFamily(), defaultModelId);

    inputs.modelFamily.addEventListener("change", () => {
      populateModelsForFamily(inputs.model, models, inputValue(inputs.modelFamily, undefined));
      syncModelDefaults();
      update();
    });
    inputs.model.addEventListener("change", () => {
      syncModelDefaults();
      update();
    });
    inputs.stateCheckpointPolicyPromptEnd.addEventListener("change", update);
    inputs.stateCheckpointPolicyFixedInterval.addEventListener("change", () => {
      if (
        inputs.stateCheckpointPolicyFixedInterval.checked &&
        inputs.stateCheckpointInterval.value.trim() === ""
      ) {
        inputs.stateCheckpointInterval.value = String(
          STATE_CUSTOM_INTERVAL_DEFAULT,
        );
      }
      update();
    });
    inputs.stateCheckpointInterval.addEventListener("blur", () => {
      const interval = parseStateCheckpointInterval(
        inputs.stateCheckpointInterval.value,
        defaultStateCheckpointInterval(selectedModel()),
      );
      inputs.stateCheckpointInterval.value = Number.isFinite(interval)
        ? String(interval)
        : "";
      update();
    });
    addInputListeners(inputs, update);

    syncModelDefaults();
    update();
  }

  function mount(rootId, data) {
    initialize(document.getElementById(rootId), data);
  }

  return {
    BYTES_PER_GB,
    BYTES_PER_GIB,
    calculate,
    calculateElementsPerSequence,
    formatBytes,
    modelFamily,
    modelsForFamily,
    mount,
  };
});

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
  const QSA_PENDING_KEY_BYTES_PER_ELEMENT = 2;
  const QSA_ROPE_POSITION_BYTES = 8;
  const QSA_ROPE_POSITION_AXES = 3;
  const PLE_CONTEXT_TOKEN_BYTES = 8;
  const KIMI_KDA_CONV_BYTES_PER_ELEMENT = 2;
  const KIMI_KDA_RECURRENT_BYTES_PER_ELEMENT = 4;
  const INKLING_SCONV_BYTES_PER_ELEMENT = 2;
  const STATE_CHECKPOINT_INFINITY = "∞";
  const STATE_CUSTOM_INTERVAL_DEFAULT = 10240;
  const STATE_CHECKPOINT_POLICY_FIXED_INTERVAL = "fixed_interval";
  const DEFAULT_MODEL_ID = "kimi-k3";

  const DEFAULT_PRECISIONS = {
    bf16_fp16: { label: "BF16 / FP16", bytesPerElement: 2 },
    fp8_int8: { label: "FP8 / INT8", bytesPerElement: 1 },
    fp4_int4: { label: "FP4 / INT4", bytesPerElement: 0.5 },
    fp8_sglang_packed: {
      label: "FP8 SGLang packed KV",
      bytesPerElement: 1,
      modelOnly: true,
      formulas: ["deepseek_v4_hybrid"],
      bytesPerEntry: 584,
    },
    fp4_int4_g32_s1: {
      label: "MXFP4 (32D groups + 1 B scale)",
      bytesPerElement: 0.5,
      modelOnly: true,
      formulas: ["deepseek_v4_hybrid"],
      groupSize: 32,
      scaleBytesPerGroup: 1,
    },
    fp8_int8_g128_s4: {
      label: "FP8 (128D groups + 4 B scale)",
      bytesPerElement: 1,
      modelOnly: true,
      formulas: ["deepseek_v4_hybrid"],
      groupSize: 128,
      scaleBytesPerGroup: 4,
    },
    nvfp4_g16_s1: {
      label: "NVFP4 (16D groups + 1 B scale)",
      bytesPerElement: 0.5,
      modelOnly: true,
      groupSize: 16,
      scaleBytesPerGroup: 1,
    },
  };
  const INDEXER_ONLY_PRECISION_IDS = new Set([
    "fp4_int4_g32_s1",
    "fp8_int8_g128_s4",
  ]);
  const KV_ONLY_PRECISION_IDS = new Set(["fp8_sglang_packed"]);
  const DEFAULT_KV_PRECISIONS = Object.fromEntries(
    Object.entries(DEFAULT_PRECISIONS).filter(([id]) => !INDEXER_ONLY_PRECISION_IDS.has(id)),
  );
  const DEFAULT_INDEXER_PRECISIONS = Object.fromEntries(
    Object.entries(DEFAULT_PRECISIONS).filter(([id]) => !KV_ONLY_PRECISION_IDS.has(id)),
  );

  const DEFAULT_RECURRENT_STATE_PRECISIONS = {
    bf16_fp16: { label: "BF16 / FP16", bytesPerElement: 2 },
    fp32: { label: "FP32", bytesPerElement: 4 },
  };

  const FORMULA_LABELS = {
    standard_gqa: "Standard MHA/GQA",
    mla: "MLA latent KV",
    dsa_mla: "DSA/MLA with indexer",
    dots3_note_hybrid: "Dots3 Note DSA-MLA/SWA-MLA hybrid",
    kimi_kda_mla_hybrid: "Kimi KDA/MLA hybrid",
    inkling_hybrid: "Inkling global/SWA with SConv",
    qwen_linear_full_hybrid: "Qwen linear/full hybrid",
    qwen4_exp_hybrid: "Qwen GDN/QSA hybrid with PLE",
    mixed_full_sliding_gqa: "Mixed full/sliding GQA",
    minimax_msa: "MiniMax MSA sparse attention",
    deepseek_v4_hybrid: "DeepSeek V4 hybrid sparse attention",
    deepseek_v41_shared: "DeepSeek V4.1 CED/CSA2 shared cache",
    kimi_kda_dsa_mla_hybrid: "KDA/DSA/MLA with indexer hybrid",
  };

  function toPositiveNumber(value, fallback) {
    const parsed = Number(value);
    return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
  }

  function toPositiveInteger(value, fallback) {
    return Math.max(1, Math.floor(toPositiveNumber(value, fallback)));
  }

  function validateCalculatorInputs(model, input) {
    const errors = {};
    const positiveInteger = (value) => Number.isSafeInteger(Number(value)) && Number(value) > 0;
    if (!positiveInteger(input.tokens)) errors.tokens = "Enter a positive whole number of tokens.";
    if (!positiveInteger(input.sequences)) errors.sequences = "Enter a positive whole number of sequences.";
    if (input.fixedInterval && !positiveInteger(input.stateCheckpointInterval)) {
      errors.stateCheckpointInterval = "Enter a positive whole number of tokens per checkpoint.";
    }
    const contextLimit = positiveInteger(model.max_position_embeddings) ? Number(model.max_position_embeddings) : null;
    return {
      errors,
      contextLimit,
      contextExceeded: !errors.tokens && contextLimit !== null && Number(input.tokens) > contextLimit,
    };
  }

  function normalizePrecisionOptions(precisionOptions, fallback) {
    if (!Array.isArray(precisionOptions)) return fallback;
    return Object.fromEntries(
      precisionOptions.map((option) => [
        option.id,
        {
          label: option.label,
          bytesPerElement: Number(option.bytes_per_element),
          bytesPerEntry: Number(option.bytes_per_entry),
          groupSize: Number(option.group_size),
          scaleBytesPerGroup: Number(option.scale_bytes_per_group),
          modelOnly: option.model_only === true,
          formulas: Array.isArray(option.formulas) ? option.formulas : [],
        },
      ]),
    );
  }

  function precisionOptions(options) {
    const all = normalizePrecisionOptions(options && options.precisionOptions, DEFAULT_KV_PRECISIONS);
    return Object.fromEntries(
      Object.entries(all).filter(([id]) => !INDEXER_ONLY_PRECISION_IDS.has(id)),
    );
  }

  function optionsForRole(options, role) {
    const isIndexer = role === "indexer";
    const primary = options && options[isIndexer ? "indexerPrecisionOptions" : "precisionOptions"];
    const source = Array.isArray(primary)
      ? primary
      : isIndexer && options && Array.isArray(options.precisionOptions)
        ? options.precisionOptions
        : undefined;
    const fallback = isIndexer ? DEFAULT_INDEXER_PRECISIONS : DEFAULT_KV_PRECISIONS;
    const all = normalizePrecisionOptions(source, fallback);
    const excluded = isIndexer ? KV_ONLY_PRECISION_IDS : INDEXER_ONLY_PRECISION_IDS;
    return Object.fromEntries(
      Object.entries(all).filter(([id]) => !excluded.has(id)),
    );
  }

  function allowedPrecisionIds(model, fieldName) {
    const ids = model && model.fields && model.fields[fieldName];
    return Array.isArray(ids) ? ids : null;
  }

  function precisionOptionAllowed(model, fieldName, option) {
    const ids = allowedPrecisionIds(model, fieldName);
    if (ids) return ids.includes(option.id);
    if (isDeepSeekV4(model) && fieldName === "indexer_precision_ids") {
      if (option.id === "fp4_int4" || option.id === "fp8_int8") return false;
    }
    if (!(option.model_only === true || option.modelOnly === true)) return true;
    if (option.formulas && option.formulas.includes(model && model.formula)) {
      return option.id !== "fp8_sglang_packed" ||
        (Number(model && model.fields && model.fields.head_dim) === 512);
    }
    return false;
  }

  function precisionOptionsForModel(options, model) {
    const all = precisionOptions(options || {});
    return Object.fromEntries(
      Object.entries(all).filter(([id, option]) =>
        precisionOptionAllowed(model, "kv_precision_ids", { ...option, id }),
      ),
    );
  }

  function precisionOptionIdsForModel(model, options, role) {
    const fieldName = role === "indexer" ? "indexer_precision_ids" : "kv_precision_ids";
    const roleDefaults = role === "indexer" ? DEFAULT_INDEXER_PRECISIONS : DEFAULT_KV_PRECISIONS;
    const fallbackOptions = Object.entries(roleDefaults).map(([id, option]) => ({
      id,
      model_only: option.modelOnly,
      formulas: option.formulas,
    }));
    const ids = (Array.isArray(options) && options.length ? options : fallbackOptions)
      .map((option) => {
        const rawId = typeof option === "string" ? option : option && option.id;
        const id = role === "indexer" ? canonicalIndexerPrecisionId(model, rawId) : rawId;
        if (!id || (role === "indexer" ? KV_ONLY_PRECISION_IDS : INDEXER_ONLY_PRECISION_IDS).has(id)) {
          return null;
        }
        const defaults = DEFAULT_PRECISIONS[id] || {};
        return {
          id,
          modelOnly: typeof option === "string"
            ? defaults.modelOnly
            : option.model_only === undefined
              ? defaults.modelOnly
              : option.model_only === true,
          formulas: (typeof option === "string" || !Array.isArray(option.formulas))
            ? defaults.formulas || []
            : option.formulas,
        };
      })
      .filter(Boolean)
      .filter((option) => precisionOptionAllowed(model, fieldName, option))
      .map((option) => option.id);
    return Array.from(new Set(ids));
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

  function isDeepSeekV41(model) {
    return Boolean(model && model.formula === "deepseek_v41_shared");
  }

  function isInkling(model) {
    return Boolean(model && model.formula === "inkling_hybrid");
  }

  function hasIndexerCache(model) {
    return Boolean(
      model &&
        model.fields &&
        (Number.isFinite(Number(model.fields.index_head_dim)) ||
          (model.formula === "qwen4_exp_hybrid" &&
            Number.isFinite(Number(model.fields.indexer_head_dim)))),
    );
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
          model.formula === "qwen4_exp_hybrid" ||
          model.formula === "kimi_kda_mla_hybrid" ||
          model.formula === "kimi_kda_dsa_mla_hybrid"),
    );
  }

  function hasKdaCheckpointInterval(model) {
    return Boolean(
      model &&
        (model.formula === "kimi_kda_mla_hybrid" ||
          model.formula === "kimi_kda_dsa_mla_hybrid"),
    );
  }

  function hasQwenCheckpointInterval(model) {
    return Boolean(
      model &&
        (model.formula === "qwen_linear_full_hybrid" ||
          model.formula === "qwen4_exp_hybrid"),
    );
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

  function formatStateCheckpointInterval(value) {
    return Number.isFinite(value)
      ? String(value)
      : STATE_CHECKPOINT_INFINITY;
  }

  function toBoolean(value) {
    return value === true || value === "true" || value === "on" || value === "1";
  }

  function defaultPrecisionId(model, options) {
    const optionsById = precisionOptionsForModel(options || {}, model);
    const fixedPrecisionId = fixedKvPrecisionId(model);
    if (fixedPrecisionId && optionsById[fixedPrecisionId]) return fixedPrecisionId;
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
    const fields = (model && model.fields) || {};
    const fieldBytes = Number(fields.kda_recurrent_state_bytes_per_element);
    const modelDefault =
      typeof fields.default_recurrent_state_precision_id === "string"
        ? fields.default_recurrent_state_precision_id
        : fieldBytes === 2
          ? "bf16_fp16"
          : "fp32";
    if (optionsById[modelDefault]) return modelDefault;
    return optionsById.fp32 ? "fp32" : Object.keys(optionsById)[0];
  }

  function indexerPrecisionOptions(options, model) {
    const all = optionsForRole(options || {}, "indexer");
    const allowed = allowedPrecisionIds(model, "indexer_precision_ids") || [];
    if (isDeepSeekV4(model) || allowed.length) {
      for (const id of ["fp4_int4_g32_s1", "fp8_int8_g128_s4"]) {
        if (!isDeepSeekV4(model) && !allowed.includes(id)) continue;
        if (!all[id]) all[id] = DEFAULT_PRECISIONS[id];
      }
    }
    return Object.fromEntries(
      Object.entries(all).filter(([id, option]) =>
        precisionOptionAllowed(model, "indexer_precision_ids", { ...option, id }),
      ),
    );
  }

  function fixedIndexerPrecisionId(model) {
    return model && model.fields && typeof model.fields.indexer_fixed_precision_id === "string"
      ? model.fields.indexer_fixed_precision_id
      : undefined;
  }

  function fixedKvPrecisionId(model) {
    return model && model.fields && typeof model.fields.kv_fixed_precision_id === "string"
      ? model.fields.kv_fixed_precision_id
      : undefined;
  }

  function canonicalIndexerPrecisionId(model, precisionId) {
    const allowed = allowedPrecisionIds(model, "indexer_precision_ids") || [];
    if (!isDeepSeekV4(model) && !allowed.length) return precisionId;
    if (precisionId === "fp4_int4" && (isDeepSeekV4(model) || allowed.includes("fp4_int4_g32_s1"))) {
      return "fp4_int4_g32_s1";
    }
    if (precisionId === "fp8_int8" && (isDeepSeekV4(model) || allowed.includes("fp8_int8_g128_s4"))) {
      return "fp8_int8_g128_s4";
    }
    return precisionId;
  }

  function defaultIndexerPrecisionId(model, options, fallbackPrecisionId) {
    const optionsById = indexerPrecisionOptions(options || {}, model);
    const fixedPrecisionId = fixedIndexerPrecisionId(model);
    if (fixedPrecisionId && optionsById[fixedPrecisionId]) return fixedPrecisionId;
    const modelDefault = model && model.fields && model.fields.default_indexer_precision_id;
    if (modelDefault && optionsById[modelDefault]) return modelDefault;
    if (isDeepSeekV4(model) && optionsById.fp4_int4_g32_s1) return "fp4_int4_g32_s1";
    if (isDeepSeekV4(model) && optionsById.fp4_int4) return "fp4_int4";
    if (fallbackPrecisionId && optionsById[fallbackPrecisionId]) return fallbackPrecisionId;
    if (optionsById.bf16_fp16) return "bf16_fp16";
    return optionsById.fp4_int4 ? "fp4_int4" : Object.keys(optionsById)[0];
  }

  function getPrecisionProfile(precisionId, options, fallbackId, model) {
    const optionsById = precisionOptionsForModel(options || {}, model);
    const selectedId = [
      precisionId,
      fallbackId,
      "bf16_fp16",
      ...Object.keys(optionsById),
    ].find((id) => optionsById[id]) || "bf16_fp16";
    const selected = optionsById[selectedId] || DEFAULT_PRECISIONS.bf16_fp16;
    const fields = (model && model.fields) || {};
    const modelEntryBytes = Number(fields[`kv_entry_bytes_${selectedId}`]);
    const bytesPerEntry = Number.isFinite(modelEntryBytes) && modelEntryBytes > 0
      ? modelEntryBytes
      : Number.isFinite(selected.bytesPerEntry) && selected.bytesPerEntry > 0
        ? selected.bytesPerEntry
        : isDeepSeekV4(model) && selectedId === "fp8_sglang_packed" && Number(fields.head_dim) === 512
          ? 584
          : undefined;
    return {
      label: (fixedKvPrecisionId(model) && model.fields.kv_precision_label) ||
        fields[`kv_precision_label_${selectedId}`] || selected.label,
      bytesPerElement: selected.bytesPerElement,
      bytesPerEntry,
      groupSize: selected.groupSize,
      scaleBytesPerGroup: selected.scaleBytesPerGroup,
    };
  }

  function getIndexerPrecisionProfile(precisionId, options, model, fallbackPrecisionId) {
    const optionsById = indexerPrecisionOptions(options || {}, model);
    const fixedPrecisionId = fixedIndexerPrecisionId(model);
    const selectedId = [
      fixedPrecisionId,
      precisionId,
      defaultIndexerPrecisionId(model, options, fallbackPrecisionId),
    ].find((id) => optionsById[id]) || "fp4_int4";
    const selected = optionsById[selectedId] || DEFAULT_PRECISIONS.fp4_int4;
    return {
      id: selectedId,
      label: (fixedPrecisionId && model.fields.indexer_precision_label) ||
        (model && model.fields && model.fields[`indexer_precision_label_${selectedId}`]) ||
        selected.label,
      bytesPerElement: selected.bytesPerElement,
      groupSize: selected.groupSize,
      scaleBytesPerGroup: selected.scaleBytesPerGroup,
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

    if (formula === "dots3_note_hybrid") {
      const layers = getField(model, "num_hidden_layers");
      const fullLayers = getField(model, "full_attention_layers");
      const slidingLayers = getField(model, "sliding_attention_layers");
      const fullKvRank = getField(model, "kv_lora_rank");
      const fullRopeDim = getField(model, "qk_rope_head_dim");
      const slidingKvRank = getField(model, "swa_kv_lora_rank");
      const slidingRopeDim = getField(model, "swa_qk_rope_head_dim");
      const slidingWindow = getField(model, "sliding_window");
      const indexDim = getField(model, "index_head_dim");
      const indexerScaleBytes = optionalField(model, "indexer_scale_bytes", 4);
      const retainedSlidingTokens = Math.min(tokens, slidingWindow);

      const fullElements = tokens * fullLayers * (fullKvRank + fullRopeDim);
      const slidingElements =
        retainedSlidingTokens * slidingLayers * (slidingKvRank + slidingRopeDim);
      const indexerElements = tokens * fullLayers * indexDim;
      const indexerScaleCount = tokens * fullLayers;
      const indexerScaleBytesPerSequence = indexerScaleCount * indexerScaleBytes;

      return {
        elementsPerSequence:
          fullElements + slidingElements + indexerElements + indexerScaleCount,
        elementsPerToken:
          (fullElements + slidingElements + indexerElements + indexerScaleCount) /
          tokens,
        formulaLabel: FORMULA_LABELS[formula],
        formulaText:
          "full_mla_bytes = tokens * sequences * full_layers * (kv_lora_rank + qk_rope_head_dim) * kv_precision_bytes\nswa_mla_bytes = min(tokens, sliding_window) * sequences * sliding_layers * (swa_kv_lora_rank + swa_qk_rope_head_dim) * kv_precision_bytes\nindexer_vector_bytes = tokens * sequences * full_layers * index_head_dim * indexer_precision_bytes\nindexer_scale_bytes = tokens * sequences * full_layers * fp32_scale_bytes\ntotal_bytes = full_mla_bytes + swa_mla_bytes + indexer_vector_bytes + indexer_scale_bytes",
        formulaRows: [
          {
            name: "full_mla_bytes",
            expression:
              "tokens x sequences x full_layers x (kv_lora_rank + qk_rope_head_dim) x kv_precision_bytes",
            description:
              "Dots3 Note full-attention layers retain compressed MLA latent and RoPE cache for every token.",
          },
          {
            name: "swa_mla_bytes",
            expression:
              "min(tokens, sliding_window) x sequences x sliding_layers x (swa_kv_lora_rank + swa_qk_rope_head_dim) x kv_precision_bytes",
            description:
              "Sliding-attention layers retain their wider SWA-MLA latent and RoPE cache only for the effective local window.",
          },
          {
            name: "indexer_vector_bytes",
            expression:
              "tokens x sequences x full_layers x index_head_dim x indexer_precision_bytes",
            description:
              "Each DSA full-attention layer stores one index key vector per token; index_n_heads affects scoring rather than stored cache width.",
          },
          {
            name: "indexer_scale_bytes",
            expression:
              "tokens x sequences x full_layers x fp32_scale_bytes",
            description:
              "The FP8 index vector has one FP32 quantization scale per token and DSA layer.",
          },
          {
            name: "total_bytes",
            expression:
              "full_mla_bytes + swa_mla_bytes + indexer_vector_bytes + indexer_scale_bytes",
            description:
              "Combined full-attention MLA, sliding-window MLA, and DSA indexer payload.",
          },
        ],
        note:
          "Payload estimate for 13 DSA-MLA and 33 SWA-MLA layers. The SWA cache retains 512 prior tokens because the source configuration's sliding_window_size=513 includes the current token. Allocator blocks, page metadata, fragmentation, temporary workspaces, activations, model weights, and multimodal encoder state are excluded.",
        byteGroups: [
          { role: "kv", label: "Full-attention MLA cache", elements: fullElements },
          { role: "kv", label: "Sliding-window MLA cache", elements: slidingElements },
          { role: "indexer", label: "Indexer vector cache", elements: indexerElements },
          {
            role: "indexer",
            label: "Indexer FP32 scale cache",
            bytesPerSequence: indexerScaleBytesPerSequence,
          },
        ],
        components: [
          ["Main layers", layers],
          ["Full-attention layers", fullLayers, "DSA-MLA layers whose cache grows with the full context."],
          ["Sliding-attention layers", slidingLayers, "SWA-MLA layers whose persistent cache is capped by the local window."],
          ["Retained sliding-window tokens", retainedSlidingTokens, "min(tokens, 512); the configured size 513 includes the current token."],
          ["Full MLA elements", fullElements, "Full-attention latent plus RoPE scalar elements before applying KV precision."],
          ["Sliding MLA elements", slidingElements, "SWA latent plus RoPE scalar elements before applying KV precision."],
          ["Indexer vector elements", indexerElements, "One index_head_dim vector per token and full-attention layer."],
          ["Indexer scale values", indexerScaleCount, "One FP32 scale per token and full-attention layer."],
          ["Indexer scale bytes", indexerScaleBytes, "FP32 scale width; independent of the selected KV precision."],
          ["Model fields", fieldList(model, ["num_hidden_layers", "full_attention_layers", "sliding_attention_layers", "kv_lora_rank", "qk_rope_head_dim", "swa_kv_lora_rank", "swa_qk_rope_head_dim", "sliding_window", "sliding_window_size", "index_head_dim", "index_n_heads", "indexer_scale_bytes"])],
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
      const recurrentBytesPerElement = toPositiveNumber(
        settings && settings.recurrentStateBytesPerElement,
        toPositiveNumber(
          settings && settings.qwenRecurrentStateBytesPerElement,
          optionalField(
            model,
            "kda_recurrent_state_bytes_per_element",
            KIMI_KDA_RECURRENT_BYTES_PER_ELEMENT,
          ),
        ),
      );
      const recurrentPrecisionLabel =
        (settings &&
          (settings.recurrentStatePrecisionLabel ||
            settings.qwenRecurrentStatePrecisionLabel)) ||
        (recurrentBytesPerElement === 2 ? "BF16 / FP16" : "FP32");

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
              `${recurrentPrecisionLabel} KDA recurrent matrices stored in every retained checkpoint.`,
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
          ? `Includes the 24-layer token-addressable MLA latent cache and retained BF16-convolution/${recurrentPrecisionLabel}-recurrent KDA checkpoints. Active, ping-pong, and speculative runtime buffers are excluded.`
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
            "KDA recurrent-state precision",
            recurrentPrecisionLabel,
            "Serving stacks keep the recurrent matrices in FP32 or BF16 / FP16: SGLang --mamba-ssm-dtype, vLLM --mamba-ssm-cache-dtype.",
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
            `One BF16-convolution/${recurrentPrecisionLabel}-recurrent KDA checkpoint.`,
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

    if (formula === "kimi_kda_dsa_mla_hybrid") {
      const layers = getField(model, "num_hidden_layers");
      const fullLayers = getField(model, "full_attention_layers");
      const draftLayers = includeDraftKvCache ? draftLayerCount(model) : 0;
      const activeFullLayers = fullLayers + draftLayers;
      const kvRank = getField(model, "kv_lora_rank");
      const ropeDim = getField(model, "qk_rope_head_dim");

      const mlaElementsPerToken = activeFullLayers * (kvRank + ropeDim);
      const mlaElements = mlaElementsPerToken * tokens;

      const indexerPlan = indexerLayerPlan(model, fullLayers, draftLayers);
      const indexDim = getField(model, "index_head_dim");
      const indexKpool = toPositiveInteger(optionalField(model, "index_kpool", 1), 1);
      const indexerPools = Math.floor(tokens / indexKpool);
      const indexerElements = indexerPools * indexerPlan.activeIndexerLayers * indexDim;
      const indexerElementsPerToken = indexerElements / tokens;
      const indexerPrecisionId = (settings && settings.indexerPrecisionId) ||
        defaultIndexerPrecisionId(model);
      const indexerPrecisionProfile = (settings && settings.indexerPrecisionProfile) || {};
      const indexerScalePrefix = indexerPrecisionId === "fp8_int8"
        ? "indexer_fp8"
        : indexerPrecisionId === "fp4_int4" ? "indexer_mxfp4" : null;
      const indexerQuantBlockSize = indexerPrecisionProfile.groupSize ||
        (indexerScalePrefix
          ? optionalField(model, `${indexerScalePrefix}_quant_block_size`, 0)
          : 0);
      const indexerScaleBytesPerElement = indexerPrecisionProfile.scaleBytesPerGroup ||
        (indexerScalePrefix
          ? optionalField(model, `${indexerScalePrefix}_scale_bytes_per_element`, 0)
          : 0);
      const indexerScalesPerBlock = indexerQuantBlockSize > 0
        ? indexerPrecisionProfile.groupSize > 0
          ? Math.ceil(indexDim / indexerQuantBlockSize)
          : Math.floor(indexDim / indexerQuantBlockSize)
        : 0;
      const indexerScaleElements =
        indexerPools * indexerPlan.activeIndexerLayers * indexerScalesPerBlock;
      const indexerScaleBytesPerSequence =
        indexerScaleElements * indexerScaleBytesPerElement;
      const indexTailTokens = tokens % indexKpool;
      const includeIndexTail = indexTailTokens > 0;
      const indexTailBytesPerElement = optionalField(model, "index_tail_bytes_per_element", 2);
      const indexTailElements = includeIndexTail
        ? indexerPlan.activeIndexerLayers * indexKpool * 2 * indexDim
        : 0;
      const indexTailBytesPerSequence = indexTailElements * indexTailBytesPerElement;

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
      const recurrentBytesPerElement = toPositiveNumber(
        settings && settings.recurrentStateBytesPerElement,
        toPositiveNumber(
          settings && settings.qwenRecurrentStateBytesPerElement,
          optionalField(
            model,
            "kda_recurrent_state_bytes_per_element",
            KIMI_KDA_RECURRENT_BYTES_PER_ELEMENT,
          ),
        ),
      );
      const recurrentPrecisionLabel =
        (settings &&
          (settings.recurrentStatePrecisionLabel ||
            settings.qwenRecurrentStatePrecisionLabel)) ||
        (recurrentBytesPerElement === 2 ? "BF16 / FP16" : "FP32");

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
          entries: activeFullLayers * tokens,
          entryElements: kvRank + ropeDim,
        },
        { role: "indexer", label: "Indexer cache", elements: indexerElements },
      ];
      if (indexerScaleBytesPerSequence > 0) {
        byteGroups.push({
          role: "indexer",
          label: "Indexer quantization scale cache",
          elements: indexerScaleElements,
          bytesPerSequence: indexerScaleBytesPerSequence,
        });
      }
      if (includeIndexTail) {
        byteGroups.push({
          role: "index_tail",
          label: "Index tail cache",
          elements: indexTailElements,
          bytesPerSequence: indexTailBytesPerSequence,
        });
      }
      const formulaRows = [
        {
          name: "active_full_attention_layers",
          expression: "full_attention_layers + draft_layers_if_enabled",
          description:
            "Draft layers are counted only when Include draft KV cache is enabled for models that define a next-token prediction stack.",
        },
        {
          name: "active_indexer_layers",
          expression: "main_indexer_layers + draft_indexer_layers_if_enabled",
          description:
            "For DSA models with shared indexer layers, only full indexer layers allocate independent indexer key cache.",
        },
        {
          name: "mla_kv_bytes",
          expression:
            (settings && settings.kvPrecisionProfile &&
              Number.isFinite(settings.kvPrecisionProfile.bytesPerEntry))
              ? "tokens x sequences x active_full_attention_layers x fp8_sglang_packed_bytes_per_entry"
              : "tokens x sequences x active_full_attention_layers x (kv_lora_rank + qk_rope_head_dim) x kv_precision_bytes",
          description:
            (settings && settings.kvPrecisionProfile &&
              Number.isFinite(settings.kvPrecisionProfile.bytesPerEntry))
              ? `Packed SGLang MLA KV uses ${settings.kvPrecisionProfile.bytesPerEntry} logical payload bytes per row; allocator page padding is excluded.`
              : "Latent KV payload stored by the production MLA/DSA path.",
        },
        {
          name: "indexer_pools",
          expression: "floor(tokens / index_kpool)",
          description:
            "Each complete token pool shares one indexer state.",
        },
        {
          name: "indexer_bytes",
          expression:
            "indexer_pools x sequences x active_indexer_layers x index_head_dim x indexer_precision_bytes",
          description:
            "Additional per-pool indexer state used by independent indexer layer.",
        },
        {
          name: "indexer_scale_bytes",
          expression: indexerQuantBlockSize > 0 && indexerScaleBytesPerElement > 0
            ? `indexer_pools x sequences x active_indexer_layers x ceil(index_head_dim / ${indexerQuantBlockSize}) x ${indexerScaleBytesPerElement}`
            : "0",
          description:
            "Quantization scales are counted per vector with ceil(index_head_dim / group_size); a partial final group still needs a scale.",
        },
        {
          name: "index_tail_bytes",
          expression: includeIndexTail
            ? "sequences x active_indexer_layers x index_kpool x 2 x index_head_dim x index_tail_bytes_per_element"
            : "0 (tokens % index_kpool = 0)",
          description:
            `Per-token indexer state while a pool is in progress. There are 2 head slots: one for 'K' and the other for gate score as the 'V'.`,
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
              "sequences x kda_checkpoint_count x kda_layers x (conv_kernel - 1) x (q_dim + k_dim + v_dim) x kda_conv_state_bytes_per_element",
            description:
              `KDA short-convolution history stored in every retained checkpoint at ${convBytesPerElement} bytes per element.`,
          },
          {
            name: "kda_recurrent_state_bytes",
            expression:
              "sequences x kda_checkpoint_count x kda_layers x value_heads x value_head_dim x key_head_dim x kda_recurrent_state_bytes_per_element",
            description:
              `KDA recurrent matrices stored in every retained checkpoint at ${recurrentBytesPerElement} bytes per element.`,
          },
          {
            name: "total_bytes",
            expression:
              "mla_kv_bytes + indexer_bytes + indexer_scale_bytes + index_tail_bytes + kda_conv_state_bytes + kda_recurrent_state_bytes",
            description:
              "Combined DSA/MLA cache and retained KDA checkpoint states.",
          },
        );
      } else {
        formulaRows.push(
          {
            name: "kda_linear_attention_state",
            expression:
              "excluded unless Include linear-attention state is enabled",
            description:
              "KDA layers keep convolution and recurrent state rather than ordinary token-addressable KV blocks.",
          },
          {
            name: "total_bytes",
            expression: "mla_kv_bytes + indexer_bytes + indexer_scale_bytes + index_tail_bytes",
            description: "DSA/MLA cache, compressed indexer payload, and fixed index tail cache without KDA checkpoint state.",
          },
        );
      }

      return {
        elementsPerSequence:
          mlaElements + indexerElements + indexerScaleElements + indexTailElements +
          kdaCheckpointCount * (kdaConvElements + kdaRecurrentElements),
        elementsPerToken: mlaElementsPerToken + indexerElementsPerToken + indexerScaleElements / tokens,
        hitRateElementsPerToken: mlaElementsPerToken,
        hitRateEntriesPerToken: activeFullLayers,
        hitRateEntryElements: kvRank + ropeDim,
        formulaLabel: FORMULA_LABELS[formula],
        formulaText: formulaRows.map((row) => `${row.name} = ${row.expression}`).join("\n"),
        formulaRows,
        note:
          `Production estimate uses latent KV plus indexer state, with one index state per ${indexKpool} tokens in each complete pool. ` +
          (settings && settings.kvPrecisionProfile &&
            Number.isFinite(settings.kvPrecisionProfile.bytesPerEntry)
            ? `SGLang packed MLA stores ${settings.kvPrecisionProfile.bytesPerEntry} logical payload bytes per row; allocator page padding is excluded. `
            : "") +
          (indexerScaleElements > 0
            ? `Includes ${indexerScaleBytesPerElement}-byte quantization scales for the indexer. `
            : "") +
          (includeIndexTail
            ? `Includes an uncompressed index tail cache for the in-progress pool (${indexTailTokens} tokens), at ${indexTailBytesPerElement} bytes per element. `
            : "") +
          (includeLinearAttentionState
            ? `Includes retained checkpoints for ${kdaLayers} KDA layers, with ${convBytesPerElement}-byte convolution and ${recurrentBytesPerElement}-byte recurrent elements. `
            : `The ${kdaLayers} KDA layers' sequence-level state is excluded. `),
        byteGroups,
        components: [
          ["Main layers", layers, "Main transformer layers, including DSA/MLA full-attention and KDA linear-attention layers."],
          [
            "DSA/MLA full-attention layers",
            fullLayers,
            "Layers that allocate token-addressable compressed DSA/MLA latent KV plus RoPE key cache.",
          ],
          ["Draft layers included", draftLayers, "Extra MTP/draft layers after the main transformer layers."],
          ["Active full-attention layers", activeFullLayers, "Main DSA/MLA full-attention layers plus the draft layers included in KV capacity."],
          ["Main indexer layers", indexerPlan.mainIndexerLayers, "Full indexer layers that allocate independent indexer key cache."],
          ["Shared indexer layers", indexerPlan.sharedIndexerLayers, "DSA layers that reuse the previous full indexer layer's top-k selection."],
          ["Draft indexer layers included", indexerPlan.draftIndexerLayers, "Draft/MTP indexer layers counted when Include draft KV cache is enabled."],
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
            "KDA recurrent-state precision",
            recurrentPrecisionLabel,
            "Serving stacks keep the recurrent matrices in FP32 or BF16 / FP16: SGLang --mamba-ssm-dtype, vLLM --mamba-ssm-cache-dtype.",
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
            "DSA/MLA elements per token",
            mlaElementsPerToken,
            "active_full_attention_layers x (kv_lora_rank + qk_rope_head_dim). Latent KV scalar elements per token before applying KV precision.",
          ],
          ["Indexer elements per token", indexerElementsPerToken, "Indexer elements per token before applying indexer precision."],
          ["Indexer scales per block", indexerScalesPerBlock, "Scales per quantized indexer block"],
          ["Indexer scale bytes per element", indexerScaleBytesPerElement, "Fixed scale width based on indexer quantization configuration."],
          ["Indexer scale bytes per sequence", indexerScaleBytesPerSequence, "pools x active indexer layers x scales per block x scale bytes per element."],
          ["Index pool size", indexKpool, "Number of tokens sharing one compressed index key. Defaults to one when index_kpool is not configured."],
          ["Indexer pools", indexerPools, "floor(tokens / index_kpool); partial pools do not allocate an additional compressed index key."],
          ["Index tail tokens", indexTailTokens, "tokens % index_kpool; zero means there is no in-progress pool and no index tail cache."],
          ["Index tail cache included", includeIndexTail ? "Yes" : "No", "Requires an in-progress pool (tokens % index_kpool > 0), independently of the KDA state checkbox."],
          ["Index tail elements per sequence", indexTailElements, "active_indexer_layers x index_kpool x 2 x index_head_dim while tail cache is present; zero when all pools are complete."],
          ["Index tail bytes per sequence", indexTailBytesPerSequence, "Tail-cache elements multiplied by the configured tail element width."],
          ["Index tail precision bytes", indexTailBytesPerElement, "Bytes per tail-cache element, set by index_tail_bytes_per_element."],
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
            `One ${convBytesPerElement}-byte convolution/${recurrentBytesPerElement}-byte recurrent elements KDA checkpoint.`,
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
              "index_head_dim",
              "index_kpool",
              "index_tail_bytes_per_element",
              "indexer_fp8_quant_block_size",
              "indexer_fp8_scale_bytes_per_element",
              "indexer_mxfp4_quant_block_size",
              "indexer_mxfp4_scale_bytes_per_element",
              "indexer_full_layers",
              "indexer_shared_layers",
              "num_nextn_predict_layers",
              "draft_indexer_layers",
              "kda_num_heads",
              "kda_head_dim",
              "kda_num_key_heads",
              "kda_key_head_dim",
              "kda_num_value_heads",
              "kda_value_head_dim",
              "kda_conv_kernel_size",
              "kda_conv_state_bytes_per_element",
              "kda_recurrent_state_bytes_per_element",
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
      const sconvPrecisionName = sconvBytesPerElement === 4 ? "FP32" : "BF16";
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
              "sequences x checkpoint_count x (kernel_size - 1) x sconv_state_bytes_per_element x [active_global_layers x (2 x global_kv_heads x head_dim + 2 x hidden_size) + active_swa_layers x (2 x swa_kv_heads x swa_head_dim + 2 x hidden_size)]",
            description:
              `Logical ${sconvPrecisionName} K, V, attention-output, and MLP-output convolution history stored at each retained prefix boundary.`,
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
          ? `Includes reusable text-generation KV and logical ${sconvPrecisionName} SConv checkpoints. vLLM page padding, active/ping-pong/speculative buffers, and vision/audio encoder activations are excluded.`
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
            `One logical ${sconvPrecisionName} short-convolution checkpoint.`,
          ],
          [
            "SConv checkpoint bytes per sequence",
            sconvCheckpointBytesPerSequence,
            "Retained checkpoint count multiplied by bytes per checkpoint.",
          ],
          ["SConv state bytes", sconvBytesPerElement, `Current production serving implementations retain this state in ${sconvPrecisionName}.`],
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

    if (formula === "qwen4_exp_hybrid") {
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
      const qsaCompressRatio = getField(model, "indexer_compress_ratio");
      const qsaKvHeads = getField(model, "indexer_kv_heads");
      const qsaHeadDim = getField(model, "indexer_head_dim");
      const qsaCompressedTokens = Math.floor(tokens / qsaCompressRatio);
      const qsaIndexElements = qsaCompressedTokens * fullLayers * qsaKvHeads * qsaHeadDim;
      const qsaPendingKeyElements = fullLayers * qsaCompressRatio * qsaKvHeads * qsaHeadDim;
      const qsaPendingBytes =
        qsaPendingKeyElements * QSA_PENDING_KEY_BYTES_PER_ELEMENT +
        qsaCompressRatio * QSA_ROPE_POSITION_AXES * QSA_ROPE_POSITION_BYTES;
      const pleLayerIds = model.fields.ple_layer_ids;
      const pleLayers = Array.isArray(pleLayerIds)
        ? pleLayerIds.length
        : optionalField(model, "ple_num_layers", 0);
      const pleConvElements =
        pleLayers *
        getField(model, "hidden_size") *
        getField(model, "hc_count") *
        (getField(model, "ple_conv_kernel_size") - 1) *
        getField(model, "ngram_size");
      const pleContextTokens = Math.max(0, getField(model, "ngram_size") - 1);
      const pleStateBytes =
        pleConvElements * QWEN_LINEAR_CONV_BYTES_PER_ELEMENT +
        pleContextTokens * PLE_CONTEXT_TOKEN_BYTES;
      const recurrentBytesPerElement = toPositiveNumber(
        settings && settings.qwenRecurrentStateBytesPerElement,
        QWEN_LINEAR_RECURRENT_BYTES_PER_ELEMENT,
      );
      const recurrentPrecisionLabel =
        (settings && settings.qwenRecurrentStatePrecisionLabel) || "FP32";
      const checkpointInterval = parseStateCheckpointInterval(
        settings && settings.qwenCheckpointInterval,
        defaultQwenCheckpointInterval(model),
      );
      const checkpointCount = includeLinearAttentionState
        ? Number.isFinite(checkpointInterval)
          ? Math.ceil(tokens / checkpointInterval)
          : 1
        : 0;
      const elementsPerToken = fullLayers * 2 * kvHeads * headDim;
      const fullElements = elementsPerToken * tokens;
      const qsaElements = qsaIndexElements;
      const gdnConvElements =
        linearLayers *
        (linearConvKernel - 1) *
        (2 * linearKeyHeads * linearKeyDim + linearValueHeads * linearValueDim);
      const gdnRecurrentElements =
        linearLayers * linearValueHeads * linearKeyDim * linearValueDim;
      const gdnBytes =
        gdnConvElements * QWEN_LINEAR_CONV_BYTES_PER_ELEMENT +
        gdnRecurrentElements * recurrentBytesPerElement;
      const gdnCheckpointBytes = checkpointCount * gdnBytes;
      const qsaCheckpointBytes = checkpointCount * qsaPendingBytes;
      const pleCheckpointBytes = checkpointCount * pleStateBytes;
      const byteGroups = [
        { role: "kv", label: "Full-attention KV cache", elements: fullElements },
        { role: "indexer", label: "QSA compressed indexer cache", elements: qsaElements },
      ];
      if (includeLinearAttentionState) {
        byteGroups.push(
          { role: "linear_state", label: "GDN recurrent/conv checkpoint", bytesPerSequence: gdnCheckpointBytes },
          { role: "linear_state", label: "QSA pending index state", bytesPerSequence: qsaCheckpointBytes },
          { role: "linear_state", label: "PLE continuation state", bytesPerSequence: pleCheckpointBytes },
        );
      }
      const formulaRows = [
        {
          name: "full_kv_bytes",
          expression: "tokens x sequences x full_attention_layers x 2 x num_key_value_heads x head_dim x kv_precision_bytes",
          description: "Only the 12 QSA full-attention layers keep ordinary token-linear K/V cache.",
        },
        {
          name: "qsa_indexer_bytes",
          expression: "floor(tokens / indexer_compress_ratio) x sequences x full_attention_layers x indexer_kv_heads x indexer_head_dim x indexer_precision_bytes",
          description: "One compressed key is stored per complete four-token group; query heads affect scoring, not stored key width.",
        },
        {
          name: "checkpoint_count",
          expression: includeLinearAttentionState
            ? "interval is infinity ? 1 : ceil(tokens / interval)"
            : "0 (Include prefix continuation snapshot is off)",
          description: "Each retained logical snapshot includes GDN, the full four-slot QSA ring, and PLE continuation state.",
        },
        {
          name: "gdn_checkpoint_bytes",
          expression: "sequences x checkpoint_count x (BF16_conv_state_bytes + recurrent_state_elements x recurrent_state_bytes)",
          description: "GDN stores BF16/FP16 convolution history and recurrent matrices at the selected recurrent-state precision.",
        },
        {
          name: "qsa_pending_state_bytes",
          expression: "sequences x checkpoint_count x (full_attention_layers x compress_ratio x indexer_kv_heads x indexer_head_dim x 2 + compress_ratio x 3 x 8)",
          description: "Counts the full four-slot BF16 key ring per QSA layer, even if some slots are empty, plus shared three-axis int64 RoPE positions.",
        },
        {
          name: "ple_continuation_bytes",
          expression: "sequences x checkpoint_count x (ple_conv_state_elements x 2 + (ngram_size - 1) x 8)",
          description: "PLE short-convolution state plus preceding token IDs; the PLE embedding table is model weights and is excluded.",
        },
        {
          name: "total_bytes",
          expression: includeLinearAttentionState ? "full_kv_bytes + qsa_indexer_bytes + checkpoint_bytes" : "full_kv_bytes + qsa_indexer_bytes",
          description: "Logical cache payload; retained state is an explicit complete-prefix snapshot assumption.",
        },
      ];
      const checkpointElements =
        gdnConvElements +
        gdnRecurrentElements +
        qsaPendingKeyElements +
        qsaCompressRatio * QSA_ROPE_POSITION_AXES +
        pleConvElements +
        pleContextTokens;

      return {
        elementsPerSequence:
          fullElements + qsaElements + checkpointCount * checkpointElements,
        elementsPerToken: (fullElements + qsaElements) / tokens,
        formulaLabel: FORMULA_LABELS[formula],
        formulaText:
          "full_kv_bytes = tokens * sequences * full_attention_layers * 2 * num_key_value_heads * head_dim * kv_precision_bytes\nqsa_indexer_bytes = floor(tokens / indexer_compress_ratio) * sequences * full_attention_layers * indexer_kv_heads * indexer_head_dim * indexer_precision_bytes\noptional_checkpoint_bytes = checkpoint_count * (gdn_state_bytes + qsa_pending_state_bytes + ple_continuation_bytes)\ntotal_bytes = full_kv_bytes + qsa_indexer_bytes + optional_checkpoint_bytes",
        formulaRows,
        note: "Logical payload only. The state option adds one complete prefix snapshot containing GDN state, the full four-slot QSA ring per indexer layer, and PLE history. SGLang preallocates runtime pools and replicates QSA indexer state across TP ranks; runtime slots, page padding, TP placement, PLE embedding weights, MTP cache, and workspaces are excluded.",
        byteGroups,
        components: [
          ["Main layers", layers],
          ["Full-attention layers", fullLayers],
          ["Linear-attention layers", linearLayers],
          ["QSA compression ratio", qsaCompressRatio],
          ["QSA compressed tokens", qsaCompressedTokens, "floor(tokens / indexer_compress_ratio); incomplete groups are in the optional pending snapshot."],
          ["QSA pending bytes per checkpoint", qsaPendingBytes, "Counts all four BF16 key slots per QSA layer plus shared MRoPE positions, including empty slots."],
          ["GDN checkpoints per sequence", checkpointCount],
          ["GDN state bytes per checkpoint", gdnBytes],
          ["PLE continuation bytes per checkpoint", pleStateBytes],
          ["Linear recurrent-state precision", recurrentPrecisionLabel],
          ["MTP layers not included", optionalField(model, "mtp_num_hidden_layers", 0), "Speculative MTP cache is excluded."],
          ["Model fields", fieldList(model, ["architectures", "model_type", "num_hidden_layers", "full_attention_interval", "full_attention_layers", "linear_attention_layers", "num_key_value_heads", "head_dim", "indexer_budget", "indexer_compress_ratio", "indexer_n_heads", "indexer_kv_heads", "indexer_head_dim", "linear_num_key_heads", "linear_key_head_dim", "linear_num_value_heads", "linear_value_head_dim", "linear_conv_kernel_dim", "mamba_ssm_dtype", "hc_count", "ple_layer_ids", "ple_conv_kernel_size", "ngram_size"])],
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
      const kvScaleBytes =
        settings && settings.kvPrecisionProfile &&
        settings.kvPrecisionProfile.groupSize > 0 &&
        settings.kvPrecisionProfile.scaleBytesPerGroup > 0
          ? tokens * layers * 2 * kvHeads *
            Math.ceil(headDim / settings.kvPrecisionProfile.groupSize) *
            settings.kvPrecisionProfile.scaleBytesPerGroup
          : 0;
      const indexerScaleBytes =
        settings && settings.indexerPrecisionProfile &&
        settings.indexerPrecisionProfile.groupSize > 0 &&
        settings.indexerPrecisionProfile.scaleBytesPerGroup > 0
          ? tokens * sparseLayers *
            Math.ceil(indexDim / settings.indexerPrecisionProfile.groupSize) *
            settings.indexerPrecisionProfile.scaleBytesPerGroup
          : 0;

      return {
        elementsPerSequence: elementsPerToken * tokens,
        elementsPerToken,
        formulaLabel: FORMULA_LABELS[formula],
        formulaText:
          "kv_bytes = tokens * sequences * layers * 2 * num_key_value_heads * head_dim * kv_precision_bytes\nkv_scale_bytes = tokens * sequences * layers * 2 * num_key_value_heads * ceil(head_dim / group_size) * scale_bytes_per_group\nindexer_bytes = tokens * sequences * sparse_attention_layers * index_head_dim * indexer_precision_bytes\nindexer_scale_bytes = tokens * sequences * sparse_attention_layers * ceil(index_head_dim / group_size) * scale_bytes_per_group\ntotal_bytes = kv_bytes + kv_scale_bytes + indexer_bytes + indexer_scale_bytes",
        formulaRows: [
          {
            name: "kv_bytes",
            expression: "tokens x sequences x layers x 2 x num_key_value_heads x head_dim x kv_precision_bytes",
            description: "MiniMax M3 stores ordinary main K/V cache for both full-attention and sparse-attention layers.",
          },
          {
            name: "kv_scale_bytes",
            expression: "tokens x sequences x layers x 2 x num_key_value_heads x ceil(head_dim / group_size) x scale_bytes_per_group",
            description: "Grouped KV formats count scale bytes independently for each K or V head; ceil preserves a partial final group.",
          },
          {
            name: "indexer_bytes",
            expression: "tokens x sequences x sparse_attention_layers x index_head_dim x indexer_precision_bytes",
            description: "MSA sparse layers keep a key-only side cache for the lightning indexer; index_n_heads affects scoring, not the stored index-key cache width.",
          },
          {
            name: "indexer_scale_bytes",
            expression: "tokens x sequences x sparse_attention_layers x ceil(index_head_dim / group_size) x scale_bytes_per_group",
            description: "Grouped indexer formats count scale bytes per key vector; ceil preserves a partial final group.",
          },
          {
            name: "total_bytes",
            expression: "kv_bytes + kv_scale_bytes + indexer_bytes + indexer_scale_bytes",
            description: "Combined main KV cache and MSA index-key side cache.",
          },
        ],
        note: "MiniMax Sparse Attention (MSA) uses a lightweight indexer to pick the most relevant KV blocks for each query, so long-context attention can read a sparse subset of the cached tokens while keeping a separate indexer cache for block selection. NVFP4 uses 16-dimension groups with one scale byte each (64 data + 8 scale bytes per 128-dimension head); allocator page padding is excluded.",
        byteGroups: [
          {
            role: "kv",
            label: "KV cache",
            elements: kvElements,
            vectorCount: tokens * layers * 2 * kvHeads,
            vectorDim: headDim,
          },
          {
            role: "indexer",
            label: "Indexer cache",
            elements: indexerElements,
            vectorCount: tokens * sparseLayers,
            vectorDim: indexDim,
          },
        ],
        components: [
          ["Main layers", layers],
          ["Full-attention layers", fullLayers, "Dense/full-attention layers without the MSA indexer branch."],
          ["Sparse-attention layers", sparseLayers, "MSA layers that add a key-only indexer side cache."],
          ["KV elements per token", kvElementsPerToken, "Main K/V elements per token before applying KV precision."],
          ["KV scale bytes per sequence", kvScaleBytes, "Grouped KV formats count one scale byte for each 16-dimension group in every separate K/V head."],
          ["Indexer elements per token", indexerElementsPerToken, "MSA index-key elements per token before applying indexer precision."],
          ["Indexer scale bytes per sequence", indexerScaleBytes, "Grouped indexer formats count one scale byte for each 16-dimension group per key vector."],
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

    if (formula === "deepseek_v41_shared") {
      const layers = getField(model, "num_hidden_layers");
      const headDim = getField(model, "head_dim");
      const indexDim = getField(model, "index_head_dim");
      const slidingWindow = getField(model, "sliding_window");
      const sourceLayers = model.fields.kv_source_layer_ids;
      const ratios = model.fields.compress_ratios;
      if (
        !Array.isArray(sourceLayers) || !sourceLayers.length ||
        new Set(sourceLayers).size !== sourceLayers.length ||
        !Array.isArray(ratios) ||
        sourceLayers.some((id) =>
          !Number.isInteger(id) || id < 0 || id >= layers ||
          !Number.isInteger(ratios[id]) || ratios[id] < 1)
      ) {
        throw new Error(`Model ${model.id} has invalid KV source layers or compression ratios`);
      }

      // Reindex layers share the source's K store; they do not allocate another cache.
      const sourceRatios = sourceLayers.map((id) => ratios[id]);
      const entries = sourceRatios.reduce((sum, ratio) => sum + Math.floor(tokens / ratio), 0);
      const entryExpression = [...new Set(sourceRatios)].map((ratio) => {
        const count = countByValue(sourceRatios, ratio);
        const term = ratio === 1 ? "tokens" : `floor(tokens / ${ratio})`;
        return `${count > 1 ? `${count} x ` : ""}${term}`;
      }).join(" + ");
      const kvScaleBytes = Math.ceil(headDim / getField(model, "kv_quant_group_size")) * getField(model, "kv_scale_bytes");
      const indexerScaleBytes = Math.ceil(indexDim / getField(model, "indexer_quant_group_size")) * getField(model, "indexer_scale_bytes");
      const swaScaleBytes = Math.ceil(headDim / getField(model, "swa_quant_group_size")) * getField(model, "swa_scale_bytes");
      const kvEntryBytes = headDim * 0.5 + kvScaleBytes;
      const indexerEntryBytes = indexDim * 0.5 + indexerScaleBytes;
      const swaEntryBytes = headDim * getField(model, "swa_bytes_per_element") + swaScaleBytes;
      const includeSwaCache = toBoolean(settings && settings.includeSwaCache);
      const draftLayers = includeSwaCache && includeDraftKvCache ? draftLayerCount(model) : 0;
      const swaLayers = includeSwaCache ? layers + draftLayers : 0;
      const retainedSwaTokens = Math.min(tokens, slidingWindow);
      const globalElements = entries * headDim;
      const indexerElements = entries * indexDim;
      const swaElements = swaLayers * retainedSwaTokens * headDim;
      const byteGroups = [
        { role: "kv", label: "Global KV cache", elements: globalElements, bytesPerSequence: entries * kvEntryBytes },
        { role: "indexer", label: "Indexer cache", elements: indexerElements, bytesPerSequence: entries * indexerEntryBytes },
      ];
      if (includeSwaCache) {
        byteGroups.push({
          role: "kv", label: "SWA cache", elements: swaElements,
          bytesPerSequence: swaLayers * retainedSwaTokens * swaEntryBytes,
        });
      }
      const formulaRows = [
        {
          name: "cache_entries",
          expression: entryExpression,
          description: "Per-sequence entries summed only over KV-source layers. Each source owns one global KV cache and one indexer K cache, shared by its consumers.",
        },
        {
          name: "global_kv_bytes",
          expression: "sequences x cache_entries x kv_entry_bytes",
          description: "Each 512-dimensional latent uses 256 FP4 data bytes and 32 E4M3 scale bytes. This shared latent is not a separate K/V pair.",
        },
        {
          name: "indexer_bytes",
          expression: "sequences x cache_entries x indexer_entry_bytes",
          description: "Each 128-dimensional indexer key uses 64 FP4 data bytes and 4 E8M0 scale bytes. Reindex layers only compute new queries and selections.",
        },
        {
          name: "swa_bytes",
          expression: "sequences x swa_layers x min(tokens, sliding_window) x swa_entry_bytes",
          description: "One end-of-sequence snapshot per included layer. SWA layers are zero when excluded, 40 for the main model, or 43 with the three DSpark draft layers. Each record is 512 FP8 bytes plus 16 scale bytes.",
        },
        {
          name: "total_bytes",
          expression: "global_kv_bytes + indexer_bytes + swa_bytes",
          description: "Reusable logical cache payload including quantization scales, without runtime allocations or cache-page padding.",
        },
      ];
      return {
        elementsPerSequence: globalElements + indexerElements + swaElements,
        elementsPerToken: (globalElements + indexerElements + swaElements) / tokens,
        formulaLabel: FORMULA_LABELS[formula],
        formulaText: formulaRows.map((row) => `${row.name} = ${row.expression}`).join("\n"),
        formulaRows,
        note: "Official FP4 global KV / MXFP4 indexer payload, including scales. " +
          (includeSwaCache
            ? "Includes one FP8 SWA snapshot per sequence. "
            : "SWA is excluded, assuming approximate bounded replay on cache reuse. ") +
          "Excludes incomplete compressor state, Engram history, runtime buffers, page padding and vision encoder activations.",
        byteGroups,
        showCacheGroups: true,
        includesFixedState: includeSwaCache,
        components: [
          ["Main layers", layers],
          ["KV source layers", sourceLayers.join(", "), "Zero-based Full-mode layer IDs. Only these layers own global KV and indexer K stores."],
          ["Indexer K stores", sourceLayers.length, "Reindex and Reuse layers share these stores; the number of index-producing layers is not the number of K caches."],
          ["Cache entries per sequence", entries, "Only completed compression groups are stored as global entries; incomplete compressor state is excluded."],
          ["KV entry bytes", kvEntryBytes, `${headDim} x 0.5 FP4 data bytes + ${kvScaleBytes} E4M3 scale bytes.`],
          ["Indexer entry bytes", indexerEntryBytes, `${indexDim} x 0.5 FP4 data bytes + ${indexerScaleBytes} E8M0 scale bytes.`],
          ["SWA layers included", swaLayers, "SWA is optional storage payload, not optional model attention. Bounded replay reconstructs approximate local states when they are not retained."],
          ["Draft layers included", draftLayers, "The three DSpark layers add only SWA snapshots, and only when both cache options are enabled."],
          ["Sliding window", slidingWindow, "Maximum local KV entries per layer in one end-of-sequence snapshot."],
          ["SWA entry bytes", swaEntryBytes, `${headDim} FP8 data bytes + ${swaScaleBytes} E8M0 scale bytes; independent of global KV precision.`],
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
      const kvPrecisionProfile = (settings && settings.kvPrecisionProfile) || {};
      const indexerPrecisionProfile = (settings && settings.indexerPrecisionProfile) || {};
      const kvEntryBytes = kvPrecisionProfile.bytesPerEntry;
      const indexerEntries = ratioFourLayers * Math.floor(tokens / 4);
      const indexerScaleBytesPerEntry = indexerPrecisionProfile.groupSize > 0 &&
        indexerPrecisionProfile.scaleBytesPerGroup > 0
        ? Math.ceil(indexDim / indexerPrecisionProfile.groupSize) *
          indexerPrecisionProfile.scaleBytesPerGroup
        : 0;
      const indexerScaleBytes = indexerEntries * indexerScaleBytesPerEntry;
      const formulaRows = [
        {
          name: "sliding_kv_bytes",
          expression: Number.isFinite(kvEntryBytes)
            ? "active_layers x sliding_window x kv_entry_bytes"
            : "active_layers x sliding_window x head_dim x kv_precision_bytes",
          description: Number.isFinite(kvEntryBytes)
            ? `Each logical SGLang packed row uses ${kvEntryBytes} bytes, including BF16 RoPE and scale/padding payload.`
            : "Includes ratio=0 layers. Ratio=0 layers only contribute this fixed sliding-window KV and do not add compressed KV slots.",
        },
        {
          name: "compressed_kv_bytes",
          expression: Number.isFinite(kvEntryBytes)
            ? "sum_ratio>0(floor(tokens / compress_ratio)) x kv_entry_bytes"
            : "sum_ratio>0(floor(tokens / compress_ratio) x head_dim) x kv_precision_bytes",
          description: "Compressed KV cache from layers whose compress_ratio is greater than zero; each layer keeps floor(tokens / compress_ratio) compressed slots.",
        },
        {
          name: "kv_bytes",
          expression: "sliding_kv_bytes + compressed_kv_bytes",
          description: "Main DeepSeek V4 KV cache before adding the separate indexer cache.",
        },
        {
          name: "indexer_bytes",
          expression: "ratio4_layers x floor(tokens / 4) x index_head_dim x indexer_precision_bytes",
          description: "Ratio=4 layers keep an extra compressed indexer cache at its selected payload precision.",
        },
        {
          name: "indexer_scale_bytes",
          expression: indexerScaleBytesPerEntry > 0
            ? `ratio4_layers x floor(tokens / 4) x ceil(index_head_dim / ${indexerPrecisionProfile.groupSize}) x ${indexerPrecisionProfile.scaleBytesPerGroup}`
            : "0",
          description: "Quantized indexer scale payload uses ceil(index_head_dim / group_size) per stored vector; unquantized BF16 has no scale bytes.",
        },
        {
          name: "total_bytes",
          expression: "sequences x (kv_bytes + indexer_bytes + indexer_scale_bytes)",
          description: "Combined logical cache payload; allocator page padding is excluded.",
        },
      ];
      return {
        elementsPerSequence,
        elementsPerToken: elementsPerSequence / tokens,
        formulaLabel: FORMULA_LABELS[formula],
        formulaText: formulaRows.map((row) => `${row.name} = ${row.expression}`).join("\n"),
        formulaRows,
        note: "Production estimate uses the official sliding-window/compressed-cache layout. FP4 indexer vectors count one 1-byte scale per 32 dimensions; FP8 indexer vectors count one 4-byte scale per 128 dimensions; BF16 has no scale. SGLang packed FP8 KV rows use 584 logical payload bytes (448 FP8 + 128 BF16 RoPE + 8 scale/pad); allocator page padding is excluded.",
        byteGroups: [
          {
            role: "kv",
            label: "KV cache",
            elements: attentionElements,
            entries: attentionElements / headDim,
            entryElements: headDim,
          },
          {
            role: "indexer",
            label: "Indexer cache",
            elements: indexerElements,
            vectorCount: indexerEntries,
            vectorDim: indexDim,
          },
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

  function precisionForCacheRole(precision, role) {
    if ((role === "kv" || role === "attention") && precision.kvPrecisionProfile) {
      return precision.kvPrecisionProfile;
    }
    if (role === "indexer" && precision.indexerPrecisionProfile) {
      return precision.indexerPrecisionProfile;
    }
    return precision;
  }

  function calculateCacheGroups(elementPlan, precision) {
    const groups = elementPlan.byteGroups || [{ role: "cache", elements: elementPlan.elementsPerSequence }];
    return groups.map((group) => {
      const rolePrecision = precisionForCacheRole(precision, group.role);
      let bytesPerSequence = group.bytesPerSequence;
      if (!Number.isFinite(bytesPerSequence)) {
        if (
          Number.isFinite(group.entries) &&
          Number.isFinite(rolePrecision.bytesPerEntry)
        ) {
          bytesPerSequence = group.entries * rolePrecision.bytesPerEntry;
        } else {
          bytesPerSequence = group.elements * bytesPerElementForGroup(precision, group.role);
          if (
            Number.isFinite(group.vectorCount) &&
            Number.isFinite(group.vectorDim) &&
            rolePrecision.groupSize > 0 &&
            rolePrecision.scaleBytesPerGroup > 0
          ) {
            bytesPerSequence +=
              group.vectorCount *
              Math.ceil(group.vectorDim / rolePrecision.groupSize) *
              rolePrecision.scaleBytesPerGroup;
          }
        }
      }
      return {
        role: group.role,
        label: group.label || "KV cache",
        elements: group.elements,
        bytesPerSequence,
      };
    });
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
    const precisionId = fixedKvPrecisionId(model) || input.precision || defaultPrecisionId(model, options);
    const precision = getPrecisionProfile(
      precisionId,
      options,
      defaultPrecisionId(model, options),
      model,
    );
    const requestedIndexerPrecisionId =
      input.indexerPrecision || defaultIndexerPrecisionId(model, options, precisionId);
    const indexerPrecision = hasIndexerCache(model)
      ? getIndexerPrecisionProfile(
          canonicalIndexerPrecisionId(model, requestedIndexerPrecisionId),
          options,
          model,
          precisionId,
        )
      : null;
    const recurrentStatePrecision = hasLinearAttentionState(model)
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
          kvPrecisionProfile: precision,
          indexerPrecisionProfile: indexerPrecision,
        }
      : { ...precision, kvPrecisionProfile: precision };
    const elementPlan = calculateElementsPerSequence(model, tokens, {
      indexerPrecisionId: indexerPrecision ? indexerPrecision.id : undefined,
      indexerPrecisionProfile: indexerPrecision || undefined,
      kvPrecisionProfile: precision,
      includeDraftKvCache: hasDraftKvCache(model) && toBoolean(input.includeDraftKvCache),
      includeSwaCache: isDeepSeekV41(model) && (
        typeof input.includeSwaCache === "undefined"
          ? toBoolean(model.fields.default_include_swa_cache)
          : toBoolean(input.includeSwaCache)
      ),
      includeLinearAttentionState: hasLinearAttentionState(model) && toBoolean(input.includeLinearAttentionState),
      includeSconvState:
        hasSconvState(model) &&
        (typeof input.includeSconvState === "undefined"
          ? Boolean(model.fields.default_include_sconv_state)
          : toBoolean(input.includeSconvState)),
      recurrentStateBytesPerElement: recurrentStatePrecision
        ? recurrentStatePrecision.bytesPerElement
        : undefined,
      recurrentStatePrecisionLabel: recurrentStatePrecision
        ? recurrentStatePrecision.label
        : undefined,
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
    if (model.cache_note) {
      elementPlan.note = [elementPlan.note, model.cache_note].filter(Boolean).join(" ");
    }
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
    const hitRateBytesPerToken = Number.isFinite(elementPlan.hitRateEntriesPerToken) &&
      Number.isFinite(precision.bytesPerEntry)
      ? elementPlan.hitRateEntriesPerToken * precision.bytesPerEntry
      : Number.isFinite(elementPlan.hitRateElementsPerToken)
        ? elementPlan.hitRateElementsPerToken * bytesPerElementForGroup(cachePrecision, "kv")
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

  function expandModelCatalog(models, catalog) {
    const expanded = models.slice();
    const presets = new Map(expanded.map((model) => [model.id, model]));
    const seenIds = new Set(presets.keys());
    (catalog || []).forEach((entry) => {
      const preset = presets.get(entry.preset_id);
      if (!preset || !entry.official_model_id || seenIds.has(entry.official_model_id)) return;
      const cacheNote = [...new Set([preset.cache_note, entry.cache_note].filter(Boolean))].join(" ");
      expanded.push({
        ...preset,
        id: entry.official_model_id,
        label: entry.label ?? entry.official_model_id.split("/").pop(),
        source_url: entry.source_url,
        max_position_embeddings: entry.max_position_embeddings ?? preset.max_position_embeddings,
        fields: entry.fields ? { ...preset.fields, ...entry.fields } : preset.fields,
        cache_note: cacheNote || undefined,
      });
      seenIds.add(entry.official_model_id);
    });
    return expanded;
  }

  function modelById(models, id) {
    return models.find((model) => model.id === id) || models[0];
  }

  function setText(root, selector, value) {
    const node = root.querySelector(selector);
    if (node) node.textContent = value;
  }

  function renderInputValidation(root, inputs, validation) {
    for (const name of ["tokens", "sequences", "stateCheckpointInterval"]) {
      const message = validation.errors[name] || "";
      const node = root.querySelector(`[data-kv-validation='${name}']`);
      if (node) {
        node.textContent = message;
        node.hidden = !message;
      }
      if (inputs[name]) inputs[name].setAttribute("aria-invalid", String(Boolean(message)));
    }
    const context = root.querySelector("[data-kv-context-limit]");
    const limit = validation.contextLimit === null ? null : formatNumber(validation.contextLimit, 0);
    if (context) {
      context.textContent = limit === null ? "Model context limit not specified." : `Configured max context: ${limit} tokens.${validation.contextExceeded ? " Requested tokens exceed this limit." : ""}`;
      context.classList.toggle("kv-field-warning", validation.contextExceeded);
    }
    const warning = root.querySelector("[data-kv-context-warning]");
    if (warning) {
      warning.textContent = validation.contextExceeded ? `Requested tokens exceed the configured ${limit}-token context limit. This is a hypothetical estimate only.` : "";
      warning.hidden = !validation.contextExceeded;
    }
  }

  function setResultStatus(root, state, message = "") {
    root.dataset.state = state;
    const status = root.querySelector("[data-kv-result-status]");
    if (status) {
      status.textContent = message;
      status.hidden = !message;
    }
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
    if (result.elementPlan.showCacheGroups) {
      result.cacheGroups.forEach((group) => {
        metrics.push([`${group.label} size`, formatBytes(group.bytes)]);
      });
    } else if (result.indexerPrecisionLabel) {
      metrics.push([
        "KV cache size",
        formatBytes(result.kvBytes),
      ]);
      metrics.push([
        "Indexer cache size",
        formatBytes(result.indexerBytes),
      ]);
      result.cacheGroups
        .filter((group) =>
          (group.role === "linear_state" || group.role === "sconv_state") ||
          group.role === "index_tail"
        )
        .forEach((group) => {
          metrics.push([
            `${group.label} size`,
            formatBytes(group.bytes),
          ]);
        });
    } else if (result.cacheGroups.length > 1) {
      result.cacheGroups.forEach((group) => {
        metrics.push([
          `${group.label} size`,
          formatBytes(group.bytes),
        ]);
      });
    }
    metrics.push([
      "Average cache size per token",
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
    const fixedPrecisionId = fixedKvPrecisionId(model);
    const preferredValue = defaultPrecisionId(model, {
      precisionOptions: data.precision_options,
    });
    const available = rawPrecisionOptions(data).filter((option) =>
      precisionOptionAllowed(model, "kv_precision_ids", {
        ...option,
        modelOnly: option.model_only === true,
      }),
    );
    const options = fixedPrecisionId
      ? available
        .filter((option) => option.id === fixedPrecisionId)
        .map((option) => ({ ...option, label: model.fields.kv_precision_label || option.label }))
      : available.map((option) => ({
          ...option,
          label: model.fields[`kv_precision_label_${option.id}`] || option.label,
        }));
    populateSelect(select, options, preferredValue);
    if (select) select.disabled = Boolean(fixedPrecisionId);
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
      const available = rawIndexerPrecisionOptions(data).filter((option) =>
        precisionOptionAllowed(model, "indexer_precision_ids", {
          ...option,
          modelOnly: option.model_only === true,
        }),
      );
      const options = fixedPrecisionId
        ? available
          .filter((option) => option.id === fixedPrecisionId)
          .map((option) => ({ ...option, label: model.fields.indexer_precision_label || option.label }))
        : available.map((option) => ({
            ...option,
            label: model.fields[`indexer_precision_label_${option.id}`] || option.label,
          }));
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

  function syncSwaControl(root, model, reset) {
    const control = root.querySelector("[data-kv-swa-control]");
    const checkbox = root.querySelector("[data-kv-input='includeSwaCache']");
    const draft = root.querySelector("[data-kv-input='includeDraftKvCache']");
    const draftHelp = root.querySelector("[data-kv-draft-help]");
    const supported = isDeepSeekV41(model);
    if (control) control.hidden = !supported;
    if (checkbox) {
      checkbox.disabled = !supported;
      if (reset || !supported) {
        checkbox.checked = supported && toBoolean(model.fields.default_include_swa_cache);
      }
    }
    if (draft) {
      draft.disabled = supported && !checkboxValue(checkbox);
      if (draft.disabled) draft.checked = false;
    }
    if (draftHelp) {
      updateInlineHelp(draftHelp, supported
        ? "Adds the three DSpark SWA snapshots when Include SWA cache is enabled. Global KV and indexer sizes do not change; speculative runtime buffers are excluded."
        : draftHelp.getAttribute("data-kv-inline-help"));
    }
  }

  function syncLinearStateControl(root, model) {
    const control = root.querySelector("[data-kv-linear-state-control]");
    const checkbox = root.querySelector("[data-kv-input='includeLinearAttentionState']");
    const showLinearStateControl = hasLinearAttentionState(model);
    const isQwenFlashNext = model && model.formula === "qwen4_exp_hybrid";
    if (control) control.hidden = !showLinearStateControl;
    const label = control && control.querySelector("[data-kv-linear-state-label]");
    if (label) {
      label.textContent = isQwenFlashNext
        ? "Include prefix continuation snapshot"
        : "Include linear-attention state";
    }
    const help = control && control.querySelector("[data-kv-inline-help]");
    if (help) {
      const description = isQwenFlashNext
        ? "Adds one logical prefix snapshot containing GDN recurrent/conv state, all four pending-key slots per QSA layer, and PLE convolution plus prior-token history. Empty QSA ring slots are counted. The PLE embedding table is model weights and is excluded; this snapshot layout is not SGLang's preallocated runtime-pool size."
        : "Adds model-specific linear-attention recurrent and convolution state. The checkpoint policy controls how many retained states are counted.";
      help.setAttribute("data-kv-inline-help", description);
      updateInlineHelp(help, description);
    }
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
      control.hidden = !(hasLinearAttentionState(model) && includeState);
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
    if (model && model.formula === "qwen4_exp_hybrid") {
      return {
        label: "GDN + QSA + PLE",
        enabled: checkboxValue(
          root.querySelector("[data-kv-input='includeLinearAttentionState']"),
        ),
        promptHelp:
          "Assumes one logical prefix snapshot containing GDN state, all four QSA pending-key slots per layer, and PLE history, even if ring slots are empty.",
        intervalHelp:
          "Stores that complete logical continuation snapshot every N tokens. The final partial interval counts as one checkpoint.",
      };
    }
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
    const models = expandModelCatalog(data.models || [], data.model_catalog || []);
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
      includeSwaCache: root.querySelector("[data-kv-input='includeSwaCache']"),
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
      syncSwaControl(root, model, true);
      syncLinearStateControl(root, model);
      syncRecurrentStatePrecisionControl(root, model);
      syncSconvStateControl(root, model);
      syncStateCheckpointControl(root, model);
    }

    function update() {
      try {
        const model = selectedModel();
        syncSwaControl(root, model, false);
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
        const formInput = {
          tokens: inputValue(inputs.tokens, ""),
          sequences: inputValue(inputs.sequences, ""),
          fixedInterval: stateCheckpointPolicy === STATE_CHECKPOINT_POLICY_FIXED_INTERVAL,
          stateCheckpointInterval,
        };
        const validation = validateCalculatorInputs(model, formInput);
        renderInputValidation(root, inputs, validation);
        if (Object.keys(validation.errors).length) {
          setResultStatus(root, "invalid", "Calculation paused. Correct the highlighted inputs to continue.");
          return;
        }
        const result = calculate(
          model,
          {
            tokens: formInput.tokens,
            sequences: formInput.sequences,
            precision: inputValue(inputs.precision, undefined),
            indexerPrecision: inputValue(inputs.indexerPrecision, undefined),
            recurrentStatePrecision: inputValue(
              inputs.recurrentStatePrecision,
              undefined,
            ),
            includeDraftKvCache: checkboxValue(inputs.includeDraftKvCache),
            includeSwaCache: checkboxValue(inputs.includeSwaCache),
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
        setResultStatus(root, "ready");
      } catch (error) {
        setResultStatus(root, "error", `Calculation failed: ${error.message}`);
      }
    }

    const defaultModelId = inputs.model.value || DEFAULT_MODEL_ID;
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
    expandModelCatalog,
    formatBytes,
    modelFamily,
    modelsForFamily,
    precisionOptionIdsForModel,
    validateCalculatorInputs,
    mount,
  };
});

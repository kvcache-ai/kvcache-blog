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

  const DEFAULT_PRECISIONS = {
    bf16_fp16: { label: "BF16 / FP16", bytesPerElement: 2 },
    fp8_int8: { label: "FP8 / INT8", bytesPerElement: 1 },
    fp4_int4: { label: "FP4 / INT4", bytesPerElement: 0.5 },
  };

  const FORMULA_LABELS = {
    standard_gqa: "Standard MHA/GQA",
    mla: "MLA latent KV",
    dsa_mla: "DSA/MLA with indexer",
    qwen_linear_full_hybrid: "Qwen linear/full hybrid",
    mixed_full_sliding_gqa: "Mixed full/sliding GQA",
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

  function isDeepSeekV4(model) {
    return model && model.formula === "deepseek_v4_hybrid";
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
    return Boolean(model && model.formula === "qwen_linear_full_hybrid");
  }

  function toBoolean(value) {
    return value === true || value === "true" || value === "on" || value === "1";
  }

  function defaultPrecisionId(model, options) {
    const optionsById = precisionOptions(options || {});
    if (isDeepSeekV4(model) && optionsById.fp8_int8) return "fp8_int8";
    return optionsById.bf16_fp16 ? "bf16_fp16" : Object.keys(optionsById)[0];
  }

  function indexerPrecisionOptions(options) {
    return normalizePrecisionOptions(
      options && options.indexerPrecisionOptions,
      precisionOptions(options || {}),
    );
  }

  function defaultIndexerPrecisionId(model, options, fallbackPrecisionId) {
    const optionsById = indexerPrecisionOptions(options || {});
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
    const selected =
      optionsById[precisionId] ||
      optionsById[defaultIndexerPrecisionId(model, options, fallbackPrecisionId)] ||
      DEFAULT_PRECISIONS.fp4_int4;
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

  function countByValue(values, target) {
    return values.filter((value) => Number(value) === target).length;
  }

  function calculateElementsPerSequence(model, tokens, settings) {
    const formula = model.formula;
    const includeDraftKvCache = toBoolean(settings && settings.includeDraftKvCache);
    const includeLinearAttentionState = toBoolean(settings && settings.includeLinearAttentionState);

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
      const indexDim = getField(model, "index_head_dim");
      const kvRank = getField(model, "kv_lora_rank");
      const ropeDim = getField(model, "qk_rope_head_dim");
      const kvElementsPerLayer = kvRank + ropeDim;
      const indexerElementsPerLayer = indexDim;
      const elementsPerLayer = kvElementsPerLayer + indexerElementsPerLayer;

      const kvElementsPerToken = activeLayers * kvElementsPerLayer;
      const indexerElementsPerToken = activeLayers * indexerElementsPerLayer;
      const elementsPerToken = activeLayers * elementsPerLayer;
      return {
        elementsPerSequence: elementsPerToken * tokens,
        elementsPerToken,
        formulaLabel: FORMULA_LABELS[formula],
        formulaText:
          "active_layers = main_layers + draft_layers_if_enabled\nkv_bytes = tokens * sequences * active_layers * (kv_lora_rank + qk_rope_head_dim) * kv_precision_bytes\nindexer_bytes = tokens * sequences * active_layers * index_head_dim * indexer_precision_bytes\ntotal_bytes = kv_bytes + indexer_bytes",
        formulaRows: [
          {
            name: "active_layers",
            expression: "main_layers + draft_layers_if_enabled",
            description: "Draft layers are counted only when Include draft KV cache is enabled for models that define a next-token prediction stack.",
          },
          {
            name: "kv_bytes",
            expression: "tokens x sequences x active_layers x (kv_lora_rank + qk_rope_head_dim) x kv_precision_bytes",
            description: "Latent KV payload stored by the production MLA/DSA path.",
          },
          {
            name: "indexer_bytes",
            expression: "tokens x sequences x active_layers x index_head_dim x indexer_precision_bytes",
            description: "Additional per-token indexer state used by the indexer attention path.",
          },
          {
            name: "total_bytes",
            expression: "kv_bytes + indexer_bytes",
            description: "Combined cache payload after applying the selected KV and indexer precisions.",
          },
        ],
        note: "Production estimate uses latent KV plus indexer state; expanded HF-compatible cache is not included.",
        byteGroups: [
          { role: "kv", label: "KV cache", elements: kvElementsPerToken * tokens },
          { role: "indexer", label: "Indexer cache", elements: indexerElementsPerToken * tokens },
        ],
        components: [
          ["Main layers", layers],
          ["Draft layers included", draftLayers, "Extra next-token prediction layers included in KV capacity when the checkbox is enabled."],
          ["KV elements per token", kvElementsPerToken, "Latent KV elements per token before applying KV precision."],
          ["Indexer elements per token", indexerElementsPerToken, "Indexer elements per token before applying indexer precision."],
          ["Per-token elements", elementsPerToken, "KV plus indexer scalar elements per token before multiplying by precision bytes."],
          ["Model fields", fieldList(model, ["num_hidden_layers", "kv_lora_rank", "qk_rope_head_dim", "index_head_dim"])],
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
      const elementsPerToken = fullLayers * 2 * kvHeads * headDim;
      const fullElements = elementsPerToken * tokens;
      const linearConvElements =
        linearLayers *
        linearConvKernel *
        (2 * linearKeyHeads * linearKeyDim + linearValueHeads * linearValueDim);
      const linearRecurrentElements = linearLayers * linearValueHeads * linearKeyDim * linearValueDim;
      const linearStateBytesPerSequence =
        includeLinearAttentionState
          ? linearConvElements * QWEN_LINEAR_CONV_BYTES_PER_ELEMENT +
            linearRecurrentElements * QWEN_LINEAR_RECURRENT_BYTES_PER_ELEMENT
          : 0;
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
          label: "Linear-attention state",
          bytesPerSequence: linearStateBytesPerSequence,
        });
        formulaRows.push(
          {
            name: "linear_conv_state_bytes",
            expression:
              "sequences x linear_attention_layers x linear_conv_kernel_dim x (2 x linear_num_key_heads x linear_key_head_dim + linear_num_value_heads x linear_value_head_dim) x 2",
            description: "Fixed-size Qwen linear-attention convolution state, estimated at BF16/FP16 precision.",
          },
          {
            name: "linear_recurrent_state_bytes",
            expression:
              "sequences x linear_attention_layers x linear_num_value_heads x linear_key_head_dim x linear_value_head_dim x 4",
            description: "Fixed-size Qwen Gated DeltaNet recurrent state, estimated at FP32 precision.",
          },
          {
            name: "total_bytes",
            expression: "full_kv_bytes + linear_conv_state_bytes + linear_recurrent_state_bytes",
            description: "Ordinary full-attention KV plus optional Qwen linear-attention runtime state.",
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
          fullElements + (includeLinearAttentionState ? linearConvElements + linearRecurrentElements : 0),
        elementsPerToken,
        formulaLabel: FORMULA_LABELS[formula],
        formulaText:
          "full_kv_bytes = tokens * sequences * full_attention_layers * 2 * num_key_value_heads * head_dim * precision_bytes\ntotal_bytes = full_kv_bytes + optional_linear_attention_state_bytes",
        formulaRows,
        note: includeLinearAttentionState
          ? "Qwen3.5/3.6 linear-attention state is sequence-level runtime state, not per-token KV. It does not grow linearly with tokens, so it matters more for short prompts and is diluted by full-attention KV at long context."
          : "Qwen3.5/3.6 linear-attention recurrent/conv state is not ordinary per-token KV and is excluded by default. Enable the linear-attention state option to add a fixed runtime-state estimate.",
        byteGroups,
        components: [
          ["Main layers", layers],
          ["Full-attention layers", fullLayers, "Layers counted as ordinary token-linear KV cache."],
          ["Linear-attention layers", linearLayers, "Qwen Gated DeltaNet layers whose runtime state is optional and does not grow linearly with token count."],
          ["Linear state included", includeLinearAttentionState ? "Yes" : "No", "When enabled, adds fixed convolution and recurrent state for Qwen linear-attention layers."],
          ["Linear conv elements", linearConvElements, "Fixed convolution-state scalar elements per sequence before applying the 2-byte estimate."],
          ["Linear recurrent elements", linearRecurrentElements, "Fixed recurrent-state scalar elements per sequence before applying the 4-byte estimate."],
          ["MTP layers not included", mtpLayers, "Qwen3.5/3.6 configs expose MTP layers, but the cache shape is not explicit enough to include defensibly."],
          ["Per-token elements", elementsPerToken, "Ordinary full-attention KV scalar elements per token before multiplying by precision bytes."],
          ["Model fields", fieldList(model, ["num_hidden_layers", "full_attention_layers", "linear_attention_layers", "num_key_value_heads", "head_dim", "linear_num_key_heads", "linear_key_head_dim", "linear_num_value_heads", "linear_value_head_dim", "linear_conv_kernel_dim"])],
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

    return {
      modelId: model.id,
      modelLabel: model.label,
      precisionLabel: precision.label,
      indexerPrecisionLabel: indexerPrecision ? indexerPrecision.label : undefined,
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
    parent.appendChild(help);
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
    metrics.push([
      "Per token size",
      formatBytes(result.bytesPerToken),
    ]);

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
    const preferredValue = isDeepSeekV4(model) ? "fp8_int8" : "bf16_fp16";
    populateSelect(select, rawPrecisionOptions(data), preferredValue);
  }

  function populateIndexerPrecisionOptions(root, data, model) {
    const control = root.querySelector("[data-kv-indexer-control]");
    const select = root.querySelector("[data-kv-input='indexerPrecision']");
    const precisionSelect = root.querySelector("[data-kv-input='precision']");
    const showIndexerPrecision = hasIndexerCache(model);
    if (control) control.hidden = !showIndexerPrecision;
    if (showIndexerPrecision) {
      const preferredValue = defaultIndexerPrecisionId(
        model,
        { indexerPrecisionOptions: data.indexer_precision_options, precisionOptions: data.precision_options },
        precisionSelect ? precisionSelect.value : undefined,
      );
      populateSelect(select, rawIndexerPrecisionOptions(data), preferredValue);
    }
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
    if (checkbox) checkbox.checked = false;
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
      if (!input || input === inputs.model || input === inputs.modelFamily) return;
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
      includeDraftKvCache: root.querySelector("[data-kv-input='includeDraftKvCache']"),
      includeLinearAttentionState: root.querySelector("[data-kv-input='includeLinearAttentionState']"),
      tensorParallel: root.querySelector("[data-kv-input='tensorParallel']"),
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
      syncDraftControl(root, model);
      syncLinearStateControl(root, model);
    }

    function update() {
      try {
        const model = selectedModel();
        const result = calculate(
          model,
          {
            tokens: inputValue(inputs.tokens, model.default_tokens || 4096),
            sequences: inputValue(inputs.sequences, 1),
            precision: inputValue(inputs.precision, undefined),
            indexerPrecision: inputValue(inputs.indexerPrecision, undefined),
            includeDraftKvCache: checkboxValue(inputs.includeDraftKvCache),
            includeLinearAttentionState: checkboxValue(inputs.includeLinearAttentionState),
            tensorParallel: inputValue(inputs.tensorParallel, 1),
          },
          {
            precisionOptions: data.precision_options,
            indexerPrecisionOptions: data.indexer_precision_options,
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

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

  const DEFAULT_PRECISIONS = {
    bf16_fp16: { label: "BF16 / FP16", bytesPerElement: 2 },
    fp8_int8: { label: "FP8 / INT8", bytesPerElement: 1 },
    fp4_int4: { label: "FP4 / INT4", bytesPerElement: 0.5 },
  };

  const FORMULA_LABELS = {
    standard_gqa: "Standard MHA/GQA",
    mla: "MLA latent KV",
    dsa_mla: "DSA/MLA with indexer",
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

  function defaultPrecisionId(model, options) {
    const optionsById = precisionOptions(options || {});
    if (model.formula === "deepseek_v4_hybrid" && optionsById.fp8_int8) return "fp8_int8";
    return optionsById.bf16_fp16 ? "bf16_fp16" : Object.keys(optionsById)[0];
  }

  function indexerPrecisionOptions(options) {
    return normalizePrecisionOptions(
      options && options.indexerPrecisionOptions,
      precisionOptions(options || {}),
    );
  }

  function defaultIndexerPrecisionId(options) {
    const optionsById = indexerPrecisionOptions(options || {});
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

  function getIndexerPrecisionProfile(precisionId, options) {
    const optionsById = indexerPrecisionOptions(options || {});
    const selected = optionsById[precisionId] || optionsById[defaultIndexerPrecisionId(options)] || DEFAULT_PRECISIONS.fp4_int4;
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

  function fieldList(model, names) {
    return names.map((name) => `${name}=${model.fields[name]}`).join(", ");
  }

  function countByValue(values, target) {
    return values.filter((value) => Number(value) === target).length;
  }

  function calculateElementsPerSequence(model, tokens) {
    const formula = model.formula;

    if (formula === "standard_gqa") {
      const layers = getField(model, "num_hidden_layers");
      const kvHeads = getField(model, "num_key_value_heads");
      const headDim = getField(model, "head_dim");
      const elementsPerToken = layers * 2 * kvHeads * headDim;
      return {
        elementsPerSequence: elementsPerToken * tokens,
        elementsPerToken,
        formulaLabel: FORMULA_LABELS[formula],
        formulaText:
          "total_bytes = tokens * sequences * layers * 2 * num_key_value_heads * head_dim * precision_bytes",
        formulaRows: [
          {
            name: "total_bytes",
            expression: "tokens x sequences x layers x 2 x num_key_value_heads x head_dim x precision_bytes",
          },
        ],
        note: "Production estimate of base KV payload; allocator and memory-pool bytes are excluded.",
        byteGroups: [{ role: "cache", elements: elementsPerToken * tokens }],
        components: [
          ["Per-token elements", elementsPerToken],
          ["Model fields", fieldList(model, ["num_hidden_layers", "num_key_value_heads", "head_dim"])],
        ],
      };
    }

    if (formula === "mla") {
      const layers = getField(model, "num_hidden_layers");
      const kvRank = getField(model, "kv_lora_rank");
      const ropeDim = getField(model, "qk_rope_head_dim");
      const elementsPerToken = layers * (kvRank + ropeDim);
      return {
        elementsPerSequence: elementsPerToken * tokens,
        elementsPerToken,
        formulaLabel: FORMULA_LABELS[formula],
        formulaText:
          "total_bytes = tokens * sequences * layers * (kv_lora_rank + qk_rope_head_dim) * precision_bytes",
        formulaRows: [
          {
            name: "total_bytes",
            expression: "tokens x sequences x layers x (kv_lora_rank + qk_rope_head_dim) x precision_bytes",
          },
        ],
        note: "Production estimate of MLA latent KV payload; allocator and memory-pool bytes are excluded.",
        byteGroups: [{ role: "cache", elements: elementsPerToken * tokens }],
        components: [
          ["Per-token elements", elementsPerToken],
          ["Model fields", fieldList(model, ["num_hidden_layers", "kv_lora_rank", "qk_rope_head_dim"])],
        ],
      };
    }

    if (formula === "dsa_mla") {
      const layers = getField(model, "num_hidden_layers");
      const indexDim = getField(model, "index_head_dim");
      const kvRank = getField(model, "kv_lora_rank");
      const ropeDim = getField(model, "qk_rope_head_dim");
      const elementsPerLayer = kvRank + ropeDim + indexDim;

      const elementsPerToken = layers * elementsPerLayer;
      return {
        elementsPerSequence: elementsPerToken * tokens,
        elementsPerToken,
        formulaLabel: FORMULA_LABELS[formula],
        formulaText:
          "total_bytes = tokens * sequences * layers * (kv_lora_rank + qk_rope_head_dim + index_head_dim) * precision_bytes",
        formulaRows: [
          {
            name: "total_bytes",
            expression:
              "tokens x sequences x layers x (kv_lora_rank + qk_rope_head_dim + index_head_dim) x precision_bytes",
          },
        ],
        note: "Production estimate uses latent KV plus indexer state; expanded HF-compatible cache is not included.",
        byteGroups: [{ role: "cache", elements: elementsPerToken * tokens }],
        components: [
          ["Per-layer elements", elementsPerLayer],
          ["Per-token elements", elementsPerToken],
          ["Model fields", fieldList(model, ["num_hidden_layers", "kv_lora_rank", "qk_rope_head_dim", "index_head_dim"])],
        ],
      };
    }

    if (formula === "deepseek_v4_hybrid") {
      const headDim = getField(model, "head_dim");
      const indexDim = getField(model, "index_head_dim");
      const slidingWindow = getField(model, "sliding_window");
      const layers = getField(model, "num_hidden_layers");
      const ratios = Array.isArray(model.fields.compress_ratios)
        ? model.fields.compress_ratios.map((ratio) => Number(ratio)).slice(0, layers)
        : [];

      if (!ratios.length) {
        throw new Error(`Model ${model.id} is missing compress_ratios`);
      }

      let windowElements = 0;
      let compressedElements = 0;
      let indexerElements = 0;

      ratios.forEach((ratio) => {
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
          "attention_bytes = sum_layers(sliding_window * head_dim + if compress_ratio > 0 then floor(tokens / compress_ratio) * head_dim) * attention_precision_bytes\nindexer_bytes = sum_ratio4_layers(floor(tokens / 4) * index_head_dim) * indexer_precision_bytes\ntotal_bytes = sequences * (attention_bytes + indexer_bytes)",
        formulaRows: [
          {
            name: "attention_bytes",
            expression:
              "sum over layers: [sliding_window x head_dim + compressed_tokens(layer) x head_dim] x attention_precision_bytes",
          },
          {
            name: "compressed_tokens(layer)",
            expression: "compress_ratio > 0 ? floor(tokens / compress_ratio) : 0",
          },
          {
            name: "indexer_bytes",
            expression:
              "sum over ratio=4 layers: floor(tokens / 4) x index_head_dim x indexer_precision_bytes",
          },
          {
            name: "total_bytes",
            expression: "sequences x (attention_bytes + indexer_bytes)",
          },
        ],
        note: "Production estimate uses the official sliding-window/compressed-cache layout. The default DeepSeek V4 setting uses FP8 attention cache and FP4 indexer cache.",
        byteGroups: [
          { role: "attention", elements: attentionElements },
          { role: "indexer", elements: indexerElements },
        ],
        components: [
          ["Layers", ratios.length],
          ["Ratio=4 layers", countByValue(ratios, 4)],
          ["Ratio=128 layers", countByValue(ratios, 128)],
          ["Ratio=0 layers", countByValue(ratios, 0)],
          ["Sliding-window elements", windowElements],
          ["Compressed elements", compressedElements],
          ["Attention elements", attentionElements],
          ["Indexer elements", indexerElements],
        ],
      };
    }

    throw new Error(`Unsupported formula: ${formula}`);
  }

  function bytesPerElementForGroup(precision, role) {
    if (role === "attention" && Number.isFinite(precision.attentionBytesPerElement)) {
      return precision.attentionBytesPerElement;
    }
    if (role === "indexer" && Number.isFinite(precision.indexerBytesPerElement)) {
      return precision.indexerBytesPerElement;
    }
    if (Number.isFinite(precision.bytesPerElement)) return precision.bytesPerElement;
    throw new Error(`Precision ${precision.label} does not define bytes for ${role} cache`);
  }

  function calculateBytesPerSequence(elementPlan, precision) {
    const groups = elementPlan.byteGroups || [{ role: "cache", elements: elementPlan.elementsPerSequence }];
    return groups.reduce((total, group) => {
      return total + group.elements * bytesPerElementForGroup(precision, group.role);
    }, 0);
  }

  function precisionComponents(precision) {
    if (
      Number.isFinite(precision.attentionBytesPerElement) ||
      Number.isFinite(precision.indexerBytesPerElement)
    ) {
      return [
        ["Attention precision bytes", precision.attentionBytesPerElement],
        ["Indexer precision bytes", precision.indexerBytesPerElement],
      ];
    }
    return [["Precision bytes", precision.bytesPerElement]];
  }

  function calculate(model, input, options) {
    const tokens = toPositiveInteger(input.tokens, model.default_tokens || 4096);
    const sequences = toPositiveInteger(input.sequences, 1);
    const tensorParallel = toPositiveInteger(input.tensorParallel, 1);
    const precision = getPrecisionProfile(
      input.precision || defaultPrecisionId(model, options),
      options,
      defaultPrecisionId(model, options),
    );
    const indexerPrecision = model.formula === "deepseek_v4_hybrid"
      ? getIndexerPrecisionProfile(input.indexerPrecision || defaultIndexerPrecisionId(options), options)
      : null;
    const cachePrecision = indexerPrecision
      ? {
          label: precision.label,
          bytesPerElement: precision.bytesPerElement,
          attentionBytesPerElement: precision.bytesPerElement,
          indexerBytesPerElement: indexerPrecision.bytesPerElement,
        }
      : precision;
    const elementPlan = calculateElementsPerSequence(model, tokens);
    const bytesPerSequence = calculateBytesPerSequence(elementPlan, cachePrecision);
    const totalBytes = bytesPerSequence * sequences;

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
      bytesPerSequence,
      bytesPerToken: bytesPerSequence / tokens,
      perDeviceBytes: totalBytes / tensorParallel,
      perDeviceGiB: totalBytes / tensorParallel / BYTES_PER_GIB,
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

  function groupModels(models) {
    return models.reduce((groups, model) => {
      const key = model.family || "Other";
      if (!groups[key]) groups[key] = [];
      groups[key].push(model);
      return groups;
    }, {});
  }

  function setText(root, selector, value) {
    const node = root.querySelector(selector);
    if (node) node.textContent = value;
  }

  function renderComponents(root, result) {
    const list = root.querySelector("[data-kv-components]");
    if (!list) return;
    list.innerHTML = "";
    result.components.forEach(([label, value]) => {
      const item = document.createElement("div");
      item.className = "kv-breakdown-row";
      const key = document.createElement("span");
      key.textContent = label;
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

  function populateModels(root, models) {
    const select = root.querySelector("[data-kv-input='model']");
    if (!select) return;
    const groups = groupModels(models);
    select.innerHTML = "";
    Object.keys(groups)
      .sort()
      .forEach((family) => {
        const optgroup = document.createElement("optgroup");
        optgroup.label = family;
        groups[family].forEach((model) => {
          const option = document.createElement("option");
          option.value = model.id;
          option.textContent = model.label;
          optgroup.appendChild(option);
        });
        select.appendChild(optgroup);
      });
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
    const preferredValue = model.formula === "deepseek_v4_hybrid" ? "fp8_int8" : "bf16_fp16";
    populateSelect(select, rawPrecisionOptions(data), preferredValue);
  }

  function populateIndexerPrecisionOptions(root, data, model) {
    const control = root.querySelector("[data-kv-indexer-control]");
    const select = root.querySelector("[data-kv-input='indexerPrecision']");
    const showIndexerPrecision = model.formula === "deepseek_v4_hybrid";
    if (control) control.hidden = !showIndexerPrecision;
    if (showIndexerPrecision) {
      populateSelect(select, rawIndexerPrecisionOptions(data), "fp4_int4");
    }
  }

  function initialize(root, data) {
    const models = data.models || [];
    if (!root || !models.length) return;
    populateModels(root, models);

    const inputs = {
      model: root.querySelector("[data-kv-input='model']"),
      tokens: root.querySelector("[data-kv-input='tokens']"),
      sequences: root.querySelector("[data-kv-input='sequences']"),
      precision: root.querySelector("[data-kv-input='precision']"),
      indexerPrecision: root.querySelector("[data-kv-input='indexerPrecision']"),
      tensorParallel: root.querySelector("[data-kv-input='tensorParallel']"),
    };

    function selectedModel() {
      return models.find((model) => model.id === inputs.model.value) || models[0];
    }

    function syncModelDefaults() {
      const model = selectedModel();
      inputs.tokens.value = model.default_tokens || 4096;
      populatePrecisionOptions(root, data, model);
      populateIndexerPrecisionOptions(root, data, model);
    }

    function update() {
      try {
        const model = selectedModel();
        const result = calculate(
          model,
          {
            tokens: inputs.tokens.value,
            sequences: inputs.sequences.value,
            precision: inputs.precision.value,
            indexerPrecision: inputs.indexerPrecision ? inputs.indexerPrecision.value : undefined,
            tensorParallel: inputs.tensorParallel.value,
          },
          {
            precisionOptions: data.precision_options,
            indexerPrecisionOptions: data.indexer_precision_options,
          },
        );

        setText(root, "[data-kv-output='totalGiB']", `${formatNumber(result.totalGiB, RESULT_DIGITS)} GiB`);
        setText(root, "[data-kv-output='totalGB']", `${formatNumber(result.totalGB, RESULT_DIGITS)} GB`);
        setText(root, "[data-kv-output='totalTokens']", formatNumber(result.totalCachedTokens, 0));
        setText(root, "[data-kv-output='perDevice']", `${formatNumber(result.perDeviceGiB, RESULT_DIGITS)} GiB`);
        setText(root, "[data-kv-output='perSequence']", formatBytes(result.bytesPerSequence));
        setText(root, "[data-kv-output='perToken']", formatBytes(result.bytesPerToken));
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

    inputs.model.addEventListener("change", () => {
      syncModelDefaults();
      update();
    });
    Object.values(inputs).forEach((input) => {
      if (input && input !== inputs.model) input.addEventListener("input", update);
      if (input && input !== inputs.model) input.addEventListener("change", update);
    });

    if (!inputs.model.value) inputs.model.value = models[0].id;
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
    mount,
  };
});

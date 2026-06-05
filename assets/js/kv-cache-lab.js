(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory(require("./kv-cache-calculator.js"));
  } else {
    root.KVCacheLab = factory(root.KVCacheCalculator);
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function (calculator) {
  "use strict";

  if (!calculator || typeof calculator.calculate !== "function") {
    throw new Error("KVCacheLab requires KVCacheCalculator");
  }

  const BYTES_PER_GIB = calculator.BYTES_PER_GIB || 1024 ** 3;
  const DEFAULT_SEED = 20260528;
  const DEFAULT_WARMUP_FRACTION = 0.3;
  const DEFAULT_BLOCK_SIZE = 64;
  const DEFAULT_REQUESTS = 4000;
  const DEFAULT_CAPACITY_GIB_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384];
  const INPUT_DEBOUNCE_MS = 250;
  const FALLBACK_STEP_DELAY_MS = 16;
  const MAX_CACHE_ENTRIES = 12;
  const POLICIES = ["fifo", "lru", "optimal"];
  const POLICY_LABELS = { fifo: "FIFO", lru: "LRU", optimal: "Optimal" };
  const POLICY_COLORS = { fifo: "#2563eb", lru: "#059669", optimal: "#d97706" };
  const POLICY_HELP = {
    fifo: "Evicts the oldest cached block first.",
    lru: "Evicts the least recently used cached block first.",
    optimal: "Belady/MIN computed with full knowledge of the future trace. Evicts the block whose next use is farthest away, or a block that will never be used again. Theoretical upper bound, not an online policy.",
  };
  const POLICY_TOOLTIP_LINES = {
    fifo: ["Evicts the oldest cached", "block first."],
    lru: ["Evicts the least recently used", "cached block first."],
    optimal: ["Belady/MIN: evict farthest future/unused block; theoretical upper bound."],
  };
  const SOURCE_LABELS = {
    mooncake_fast25: "Mooncake FAST25",
    nvidia_aiperf_mooncake: "NVIDIA AIPerf Mooncake traces",
    qwen_bailian_trace: "Qwen Bailian traces",
    ragpulse: "RAGPulse",
    lmcache_agentic_traces: "LMCache agentic traces",
    semianalysis_weka_no_subagents: "SemiAnalysis Weka Claude Code traces",
    semianalysis_weka_with_subagents_256k: "SemiAnalysis Weka Claude Code sub-agent traces",
    kv_cache_tester: "kv-cache-tester",
    exgentic_agent_traces: "Exgentic agent traces",
    burstgpt: "BurstGPT",
    swissai_serving_trace: "SwissAI serving trace",
    sglang_hicache: "SGLang HiCache docs",
  };
  const METRIC_HELP = {
    "Trace requests": "Number of normalized requests in the real trace used to precompute this curve. Each request contributes prefill input blocks.",
    "Warmup skipped": "Requests used only to warm the cache before measuring hit rate. They still populate and evict cache blocks, but their hits and misses are excluded.",
    "Avg input tokens": "Average prefill input tokens per request in the normalized trace. Output tokens are counted only when they appear in later request input/history.",
    "Unique blocks": "Distinct block identities in the normalized trace. This is the trace working-set size before converting the selected model budget into cache-block capacity.",
    "Hit rate ceiling": "Infinite-cache prefill token hit rate after warmup. A finite FIFO/LRU/Optimal cache cannot exceed this trace-level reuse upper bound.",
    "Native block size": "Source-native or declared block granularity used by the normalized trace. Real trace mode keeps this fixed instead of reinterpreting the trace at another block size.",
    "Max cache blocks": "Number of cache blocks that fit at the largest GiB budget on the x axis, after converting model precision and block size into bytes per block.",
  };

  function toNumber(value, fallback) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  }

  function toPositiveNumber(value, fallback) {
    const parsed = toNumber(value, fallback);
    return parsed > 0 ? parsed : fallback;
  }

  function toInteger(value, fallback) {
    return Math.max(0, Math.floor(toNumber(value, fallback)));
  }

  function toPositiveInteger(value, fallback) {
    return Math.max(1, Math.floor(toPositiveNumber(value, fallback)));
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function createRng(seed) {
    let state = toInteger(seed, DEFAULT_SEED) >>> 0;
    return function rng() {
      state += 0x6d2b79f5;
      let value = state;
      value = Math.imul(value ^ (value >>> 15), value | 1);
      value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
      return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
    };
  }

  function jitter(base, spread, rng) {
    const scale = 1 + (rng() * 2 - 1) * spread;
    return Math.max(1, Math.round(base * scale));
  }

  function buildWeightedSampler(count, skew) {
    const weights = [];
    let total = 0;
    for (let index = 0; index < count; index += 1) {
      const weight = 1 / Math.pow(index + 1, Math.max(0, skew));
      weights.push(weight);
      total += weight;
    }
    let cumulative = 0;
    const cdf = weights.map((weight) => {
      cumulative += weight / total;
      return cumulative;
    });
    return function sample(rng) {
      const value = rng();
      return cdf.findIndex((entry) => value <= entry);
    };
  }

  function makeBlocks(prefix, tokens, blockSize) {
    const safeTokens = toInteger(tokens, 0);
    if (safeTokens <= 0) return [];
    const size = toPositiveInteger(blockSize, DEFAULT_BLOCK_SIZE);
    const count = Math.ceil(safeTokens / size);
    const blocks = [];
    for (let index = 0; index < count; index += 1) {
      const remaining = safeTokens - index * size;
      blocks.push({ id: `${prefix}:${index}`, tokens: Math.min(size, remaining) });
    }
    return blocks;
  }

  function cloneBlocks(blocks) {
    return blocks.map((block) => ({ id: block.id, tokens: block.tokens }));
  }

  function normalizePreset(preset) {
    if (!preset || typeof preset !== "object") {
      return { id: "custom", label: "Custom", defaults: {} };
    }
    return {
      id: preset.id || "custom",
      label: preset.label || preset.id || "Custom",
      summary: preset.summary || "",
      sources: Array.isArray(preset.sources) ? preset.sources : [],
      defaults: preset.defaults || {},
    };
  }

  function sessionTurnLimit(averageTurns, rng) {
    const spread = 0.65 + rng() * 0.9;
    return Math.max(1, Math.round(averageTurns * spread));
  }

  function createSession(index, presetId, params, topicCount, rng) {
    return {
      id: index,
      epoch: 0,
      turn: 0,
      maxTurns: sessionTurnLimit(params.average_turns, rng),
      topic: Math.floor(rng() * topicCount),
      history: [],
      prefix: `trace:${presetId}:session:${index}`,
    };
  }

  function resetSession(session, params, topicCount, rng) {
    session.epoch += 1;
    session.turn = 0;
    session.maxTurns = sessionTurnLimit(params.average_turns, rng);
    session.topic = Math.floor(rng() * topicCount);
    session.history = [];
  }

  function generateTrace(presetInput, inputParams, seed) {
    const preset = normalizePreset(presetInput);
    const defaults = preset.defaults || {};
    const params = {
      requests: toPositiveInteger(inputParams && inputParams.requests, DEFAULT_REQUESTS),
      blockSize: toPositiveInteger(inputParams && inputParams.blockSize, DEFAULT_BLOCK_SIZE),
      sessions: toPositiveInteger(inputParams && inputParams.sessions, defaults.sessions || 120),
      average_turns: toPositiveInteger(inputParams && inputParams.average_turns, defaults.average_turns || 8),
      shared_prefix_tokens: toInteger(inputParams && inputParams.shared_prefix_tokens, defaults.shared_prefix_tokens || 0),
      document_tokens: toInteger(inputParams && inputParams.document_tokens, defaults.document_tokens || 0),
      per_turn_input_tokens: toPositiveInteger(inputParams && inputParams.per_turn_input_tokens, defaults.per_turn_input_tokens || 128),
      output_tokens: toInteger(inputParams && inputParams.output_tokens, defaults.output_tokens || 256),
      reuse_skew: toNumber(inputParams && inputParams.reuse_skew, defaults.reuse_skew || 1),
      burstiness: clamp(toNumber(inputParams && inputParams.burstiness, defaults.burstiness || 0), 0, 1),
    };
    const rng = createRng(seed || DEFAULT_SEED);
    const topicCount = Math.max(4, Math.round(Math.sqrt(params.sessions)));
    const sampleSession = buildWeightedSampler(params.sessions, params.reuse_skew);
    const sharedBlocks = makeBlocks(`trace:${preset.id}:shared`, params.shared_prefix_tokens, params.blockSize);
    const topicBlocks = Array.from({ length: topicCount }, (_, index) =>
      makeBlocks(`trace:${preset.id}:topic:${index}`, params.document_tokens, params.blockSize)
    );
    const sessions = Array.from({ length: params.sessions }, (_, index) =>
      createSession(index, preset.id, params, topicCount, rng)
    );
    const requests = [];
    let lastSession = null;
    let totalInputTokens = 0;

    for (let requestIndex = 0; requestIndex < params.requests; requestIndex += 1) {
      let session = null;
      if (lastSession && lastSession.turn < lastSession.maxTurns && rng() < params.burstiness) {
        session = lastSession;
      } else {
        session = sessions[sampleSession(rng)];
      }
      if (session.turn >= session.maxTurns) resetSession(session, params, topicCount, rng);

      const turnInputTokens = jitter(params.per_turn_input_tokens, 0.35, rng);
      const outputTokens = params.output_tokens > 0 ? jitter(params.output_tokens, 0.3, rng) : 0;
      const userBlocks = makeBlocks(
        `${session.prefix}:epoch:${session.epoch}:turn:${session.turn}:user`,
        turnInputTokens,
        params.blockSize,
      );
      const outputBlocks = makeBlocks(
        `${session.prefix}:epoch:${session.epoch}:turn:${session.turn}:output`,
        outputTokens,
        params.blockSize,
      );
      const inputBlocks = []
        .concat(cloneBlocks(sharedBlocks))
        .concat(cloneBlocks(topicBlocks[session.topic] || []))
        .concat(cloneBlocks(session.history))
        .concat(cloneBlocks(userBlocks));
      const inputTokens = inputBlocks.reduce((total, block) => total + block.tokens, 0);
      totalInputTokens += inputTokens;

      requests.push({
        id: requestIndex,
        sessionId: session.id,
        turn: session.turn,
        inputBlocks,
        appendBlocks: cloneBlocks(outputBlocks),
        inputTokens,
        outputTokens,
      });

      session.history = session.history.concat(cloneBlocks(userBlocks), cloneBlocks(outputBlocks));
      session.turn += 1;
      lastSession = session;
    }

    return {
      presetId: preset.id,
      presetLabel: preset.label,
      blockSize: params.blockSize,
      params,
      requests,
      summary: {
        requests: requests.length,
        sessions: params.sessions,
        totalInputTokens,
        averageInputTokens: requests.length ? totalInputTokens / requests.length : 0,
      },
    };
  }

  function normalizeBlock(block, fallbackTokens) {
    if (typeof block === "string") return { id: block, tokens: fallbackTokens || 1 };
    return { id: String(block.id), tokens: toPositiveInteger(block.tokens, fallbackTokens || 1) };
  }

  function normalizeRequests(trace) {
    if (trace && Array.isArray(trace.normalizedRequests)) return trace.normalizedRequests;
    const rawRequests = Array.isArray(trace) ? trace : trace && trace.requests;
    if (!Array.isArray(rawRequests)) return [];
    const fallbackTokens = trace && trace.blockSize ? trace.blockSize : DEFAULT_BLOCK_SIZE;
    return rawRequests.map((request, index) => ({
      id: request.id || index,
      inputBlocks: (request.inputBlocks || request.input || []).map((block) => normalizeBlock(block, fallbackTokens)),
      appendBlocks: (request.appendBlocks || request.append || []).map((block) => normalizeBlock(block, fallbackTokens)),
    }));
  }

  function buildFutureUseQueues(requests) {
    const queues = new Map();
    requests.forEach((request, requestIndex) => {
      const seen = new Set();
      request.inputBlocks.forEach((block) => {
        if (seen.has(block.id)) return;
        seen.add(block.id);
        if (!queues.has(block.id)) queues.set(block.id, []);
        queues.get(block.id).push(requestIndex);
      });
    });
    return queues;
  }

  function consumeCurrentUses(queues, request, requestIndex) {
    const seen = new Set();
    request.inputBlocks.forEach((block) => {
      if (seen.has(block.id)) return;
      seen.add(block.id);
      const queue = queues.get(block.id);
      if (queue && queue[0] === requestIndex) queue.shift();
    });
  }

  function totalMeasuredTokens(requests, warmupRequests) {
    return requests
      .slice(warmupRequests)
      .reduce((total, request) => total + request.inputBlocks.reduce((sum, block) => sum + block.tokens, 0), 0);
  }

  function simulatePolicy(trace, cacheBlocks, policy, options) {
    const requests = normalizeRequests(trace);
    const normalizedPolicy = policy || "lru";
    const capacity = Math.max(0, Math.floor(toNumber(cacheBlocks, 0)));
    const warmupFraction = toNumber(
      options && options.warmupFraction,
      trace && trace.warmupFraction ? trace.warmupFraction : DEFAULT_WARMUP_FRACTION,
    );
    const warmupRequests = clamp(
      toInteger(
        options && Number.isFinite(Number(options.warmupRequests))
          ? options.warmupRequests
          : requests.length * warmupFraction,
        0,
      ),
      0,
      requests.length,
    );
    const measuredTokens = totalMeasuredTokens(requests, warmupRequests);
    if (!requests.length || capacity <= 0) {
      return { policy: normalizedPolicy, cacheBlocks: capacity, warmupRequests, hitTokens: 0, totalTokens: measuredTokens, hitRate: 0 };
    }

    const cache = new Map();
    const fifoQueue = [];
    const futureQueues = normalizedPolicy === "optimal" ? buildFutureUseQueues(requests) : null;
    let hitTokens = 0;
    let totalTokens = 0;

    function evictFifo() {
      while (cache.size >= capacity && fifoQueue.length) {
        const victim = fifoQueue.shift();
        if (cache.delete(victim)) return;
      }
    }

    function rememberFifo(id) {
      if (cache.has(id)) return;
      evictFifo();
      if (cache.size < capacity) {
        cache.set(id, true);
        fifoQueue.push(id);
      }
    }

    function rememberLru(id) {
      if (cache.has(id)) {
        cache.delete(id);
        cache.set(id, true);
        return;
      }
      while (cache.size >= capacity) cache.delete(cache.keys().next().value);
      cache.set(id, true);
    }

    function nextUse(id) {
      const queue = futureQueues.get(id);
      return queue && queue.length ? queue[0] : Infinity;
    }

    function rememberOptimal(id) {
      if (cache.has(id)) return;
      if (cache.size < capacity) {
        cache.set(id, true);
        return;
      }
      const candidateNext = nextUse(id);
      let victim = null;
      let victimNext = -1;
      cache.forEach((_, cachedId) => {
        const cachedNext = nextUse(cachedId);
        if (cachedNext > victimNext) {
          victim = cachedId;
          victimNext = cachedNext;
        }
      });
      if (candidateNext < victimNext) {
        cache.delete(victim);
        cache.set(id, true);
      }
    }

    function remember(id) {
      if (normalizedPolicy === "fifo") rememberFifo(id);
      else if (normalizedPolicy === "optimal") rememberOptimal(id);
      else rememberLru(id);
    }

    requests.forEach((request, requestIndex) => {
      if (futureQueues) consumeCurrentUses(futureQueues, request, requestIndex);
      const measured = requestIndex >= warmupRequests;
      request.inputBlocks.forEach((block) => {
        const hit = cache.has(block.id);
        if (measured) {
          totalTokens += block.tokens;
          if (hit) hitTokens += block.tokens;
        }
        remember(block.id);
      });
      request.appendBlocks.forEach((block) => remember(block.id));
    });

    return { policy: normalizedPolicy, cacheBlocks: capacity, warmupRequests, hitTokens, totalTokens, hitRate: totalTokens ? hitTokens / totalTokens : 0 };
  }

  function estimateBytesPerToken(model, settings) {
    const estimateTokens = toPositiveInteger(settings && settings.estimateTokens, model.default_tokens || 4096);
    const result = calculator.calculate(
      model,
      {
        tokens: estimateTokens,
        sequences: 1,
        precision: settings && settings.precision,
        indexerPrecision: settings && settings.indexerPrecision,
        includeDraftKvCache: settings && settings.includeDraftKvCache,
        includeLinearAttentionState: false,
      },
      {
        precisionOptions: settings && settings.precisionOptions,
        indexerPrecisionOptions: settings && settings.indexerPrecisionOptions,
      },
    );
    return result.bytesPerToken;
  }

  function traceEstimateTokens(trace, model) {
    if (trace && trace.summary && Number.isFinite(Number(trace.summary.averageInputTokens))) {
      return toPositiveInteger(trace.summary.averageInputTokens, model.default_tokens || 4096);
    }
    const requests = trace && Array.isArray(trace.requests) ? trace.requests : [];
    if (requests.length) {
      const total = requests.reduce((sum, request) => {
        if (Number.isFinite(Number(request.inputTokens))) return sum + Number(request.inputTokens);
        const blocks = Array.isArray(request.inputBlocks) ? request.inputBlocks : [];
        return sum + blocks.reduce((blockSum, block) => blockSum + toNumber(block.tokens, 0), 0);
      }, 0);
      return toPositiveInteger(total / requests.length, model.default_tokens || 4096);
    }
    return toPositiveInteger(model.default_tokens || 4096, 4096);
  }

  function cacheBlocksForGiB(gib, bytesPerBlock) {
    if (!Number.isFinite(bytesPerBlock) || bytesPerBlock <= 0) return 0;
    return Math.max(0, Math.floor((gib * BYTES_PER_GIB) / bytesPerBlock));
  }

  function capacityValues(settings) {
    if (settings && Array.isArray(settings.capacityGiBValues) && settings.capacityGiBValues.length) {
      return settings.capacityGiBValues.map((value) => Math.max(0, toNumber(value, 0)));
    }
    const minGiB = Math.max(0, toNumber(settings && settings.minGiB, 1));
    const maxGiB = Math.max(minGiB, toNumber(settings && settings.maxGiB, 64));
    const steps = clamp(toPositiveInteger(settings && settings.steps, 24), 2, 80);
    if (settings && (Object.prototype.hasOwnProperty.call(settings, "minGiB") || Object.prototype.hasOwnProperty.call(settings, "maxGiB") || Object.prototype.hasOwnProperty.call(settings, "steps"))) {
      return Array.from({ length: steps }, (_, index) => minGiB + ((maxGiB - minGiB) * index) / (steps - 1));
    }
    return DEFAULT_CAPACITY_GIB_VALUES.slice();
  }

  function sweepCapacity(trace, model, settings) {
    const blockSize = toPositiveInteger(settings && settings.blockSize, trace && trace.blockSize ? trace.blockSize : DEFAULT_BLOCK_SIZE);
    const accountingSettings = Object.assign({}, settings || {}, { estimateTokens: traceEstimateTokens(trace, model) });
    const bytesPerToken = estimateBytesPerToken(model, accountingSettings);
    const bytesPerBlock = bytesPerToken * blockSize;
    const policies = (settings && settings.policies) || POLICIES;
    const simulationTrace = Object.assign({}, trace, { normalizedRequests: normalizeRequests(trace) });
    const points = capacityValues(settings || {}).map((gib) => {
      const cacheBlocks = cacheBlocksForGiB(gib, bytesPerBlock);
      const results = {};
      policies.forEach((policy) => {
        results[policy] = simulatePolicy(simulationTrace, cacheBlocks, policy, { warmupFraction: settings && settings.warmupFraction });
      });
      return { gib, cacheBlocks, results };
    });
    return { blockSize, bytesPerToken, bytesPerBlock, points, policies };
  }

  function stableValue(value) {
    if (Array.isArray(value)) return value.map(stableValue);
    if (value && typeof value === "object") {
      const result = {};
      Object.keys(value)
        .sort()
        .forEach((key) => {
          result[key] = stableValue(value[key]);
        });
      return result;
    }
    return value;
  }

  function createCacheKey(input) {
    return JSON.stringify(stableValue(input));
  }

  function shouldApplyJobResult(latestJobId, result) {
    return !!result && Number(result.jobId) === Number(latestJobId);
  }

  function modelSweepKey(model, settings) {
    return [
      model && model.id,
      `precision=${settings && settings.precision ? settings.precision : ""}`,
      `indexer=${settings && settings.indexerPrecision ? settings.indexerPrecision : ""}`,
      `draft=${settings && settings.includeDraftKvCache ? "1" : "0"}`,
    ].join("|");
  }

  function precomputedTrace(precomputed, preset) {
    if (!precomputed || !precomputed.traces || !preset) return null;
    return precomputed.traces[preset.id] || null;
  }

  function parseSweepKey(key) {
    const parts = String(key || "").split("|");
    const setting = { modelId: parts[0] || "" };
    parts.slice(1).forEach((part) => {
      const equalIndex = part.indexOf("=");
      if (equalIndex < 0) return;
      const name = part.slice(0, equalIndex);
      const value = part.slice(equalIndex + 1);
      if (name === "precision") setting.precision = value || undefined;
      if (name === "indexer") setting.indexerPrecision = value || undefined;
      if (name === "draft") setting.includeDraftKvCache = value === "1";
    });
    return setting;
  }

  function availableSettingsFor(precomputed, preset, model) {
    const trace = precomputedTrace(precomputed, preset);
    if (!trace || !trace.modelSweeps || !model) return [];
    return Object.keys(trace.modelSweeps)
      .map(parseSweepKey)
      .filter((setting) => setting.modelId === model.id);
  }

  function availableModelIdsFor(precomputed, preset) {
    const trace = precomputedTrace(precomputed, preset);
    if (!trace || !trace.modelSweeps) return null;
    return new Set(Object.keys(trace.modelSweeps).map((key) => parseSweepKey(key).modelId));
  }

  function precomputedResultFor(precomputed, preset, model, settings) {
    const trace = precomputedTrace(precomputed, preset);
    if (!trace || !trace.modelSweeps) return null;
    const key = modelSweepKey(model, settings || {});
    const sweep = trace.modelSweeps[key];
    if (!sweep) return null;
    return {
      preset,
      trace: {
        presetId: preset.id,
        presetLabel: preset.label,
        blockSize: trace.nativeBlockSize || sweep.blockSize,
        sourceKind: trace.sourceKind,
        sourceBlockSizeNote: trace.sourceBlockSizeNote,
        requests: null,
        summary: trace.summary || {},
      },
      sweep: Object.assign({}, sweep, {
        precomputed: true,
        reuseCeiling:
          Number.isFinite(Number(sweep.reuseCeiling))
            ? Number(sweep.reuseCeiling)
            : trace.summary && Number.isFinite(Number(trace.summary.infiniteHitRate))
              ? Number(trace.summary.infiniteHitRate)
              : undefined,
      }),
    };
  }

  function runLabComputation(input) {
    const trace = generateTrace(input.preset, input.params, input.seed || DEFAULT_SEED);
    const sweep = sweepCapacity(trace, input.model, input.settings || {});
    return {
      jobId: input.jobId,
      cacheKey: input.cacheKey,
      preset: input.preset,
      trace,
      sweep,
    };
  }

  function rememberCachedResult(cache, key, result) {
    if (cache.has(key)) cache.delete(key);
    cache.set(key, result);
    while (cache.size > MAX_CACHE_ENTRIES) cache.delete(cache.keys().next().value);
  }

  function modelFamily(model) {
    return calculator.modelFamily ? calculator.modelFamily(model) : model.family || "Other";
  }

  function sortedModelFamilies(models) {
    return Array.from(new Set(models.map(modelFamily))).sort();
  }

  function modelsForFamily(models, family) {
    return models.filter((model) => modelFamily(model) === family);
  }

  function modelById(models, id) {
    return models.find((model) => model.id === id) || models[0];
  }

  function modelFieldNumber(model, name, fallback) {
    return toNumber(model && model.fields ? model.fields[name] : undefined, fallback);
  }

  function isDeepSeekV4(model) {
    return Boolean(model && model.formula === "deepseek_v4_hybrid");
  }

  function hasIndexerCache(model) {
    return Number.isFinite(modelFieldNumber(model, "index_head_dim", NaN));
  }

  function draftLayerCount(model) {
    const nextnLayers = modelFieldNumber(model, "num_nextn_predict_layers", 0);
    if (nextnLayers > 0) return nextnLayers;
    if (model && model.fields && model.fields.use_mtp === true) {
      return modelFieldNumber(model, "num_mtp_modules", 0) * modelFieldNumber(model, "mtp_transformer_layers", 0);
    }
    return 0;
  }

  function hasDraftKvCache(model) {
    if (!model || !model.fields) return false;
    if (isDeepSeekV4(model)) {
      const layers = modelFieldNumber(model, "num_hidden_layers", 0);
      return Array.isArray(model.fields.compress_ratios) && model.fields.compress_ratios.length > layers;
    }
    return draftLayerCount(model) > 0;
  }

  function optionIds(options) {
    return (options || []).map((option) => option.id || option);
  }

  function defaultPrecisionId(model, options) {
    const ids = optionIds(options);
    if (isDeepSeekV4(model) && ids.includes("fp8_int8")) return "fp8_int8";
    if (ids.includes("bf16_fp16")) return "bf16_fp16";
    return ids[0];
  }

  function defaultIndexerPrecisionId(model, indexerOptions, fallbackPrecisionId) {
    const ids = optionIds(indexerOptions);
    if (isDeepSeekV4(model) && ids.includes("fp4_int4")) return "fp4_int4";
    if (fallbackPrecisionId && ids.includes(fallbackPrecisionId)) return fallbackPrecisionId;
    if (ids.includes("bf16_fp16")) return "bf16_fp16";
    return ids.includes("fp4_int4") ? "fp4_int4" : ids[0];
  }

  function presetById(presets, id) {
    return presets.find((preset) => preset.id === id) || presets[0];
  }

  function sourceLabel(sourceId) {
    return SOURCE_LABELS[sourceId] || String(sourceId).replace(/_/g, "-");
  }

  function formatPresetShape(defaults) {
    if (!defaults) return "";
    const parts = [];
    if (defaults.sessions) parts.push(`${formatInteger(defaults.sessions)} sessions`);
    if (defaults.average_turns) parts.push(`${formatInteger(defaults.average_turns)} avg turns`);
    if (defaults.shared_prefix_tokens) parts.push(`${formatInteger(defaults.shared_prefix_tokens)} shared-prefix tokens`);
    if (defaults.document_tokens) parts.push(`${formatInteger(defaults.document_tokens)} document-prefix tokens`);
    if (defaults.per_turn_input_tokens) parts.push(`${formatInteger(defaults.per_turn_input_tokens)} per-turn input tokens`);
    if (defaults.output_tokens) parts.push(`${formatInteger(defaults.output_tokens)} output tokens added to later history`);
    return parts.join("; ");
  }

  function tracePresetHelpText(preset) {
    if (!preset) return "Select a public real trace with block or message identity.";
    const sources = (preset.sources || []).map(sourceLabel).join(", ");
    const nativeBlockSize = preset.native_block_size || preset.nativeBlockSize;
    const sourceKind = preset.source_kind || preset.sourceKind;
    const sections = [
      `${preset.label}: ${preset.summary || "Public trace converted into a block-request stream."}`,
    ];
    if (sourceKind === "hash") sections.push("Uses published hash/block identities directly, so cache hits are computed from repeated block ids.");
    if (sourceKind === "agent_text") sections.push("Source does not publish block ids; message/span text is converted offline into approximate hashed token buckets.");
    if (nativeBlockSize) sections.push(`Native block size: ${formatInteger(nativeBlockSize)} tokens.`);
    if (sources) sections.push(`Sources: ${sources}.`);
    return sections.join(" ");
  }

  function formatNumber(value, digits) {
    return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits });
  }

  function formatInteger(value) {
    return Number(value).toLocaleString(undefined, { maximumFractionDigits: 0 });
  }

  function formatPercent(value) {
    return `${formatNumber(value * 100, 1)}%`;
  }

  function formatCapacityGiB(gib) {
    const numeric = Number(gib);
    if (Number.isFinite(numeric) && numeric >= 1024 && numeric % 1024 === 0) return `${formatNumber(numeric / 1024, 0)} TiB`;
    return `${formatNumber(numeric, 0)} GiB`;
  }

  function setText(root, selector, text) {
    const node = root.querySelector(selector);
    if (node) node.textContent = text;
  }

  function checkboxValue(input) {
    return Boolean(input && input.checked);
  }

  function setStatus(root, text) {
    setText(root, "[data-lab-status]", text);
  }

  function createHelpButton(text, className) {
    const help = document.createElement("button");
    help.type = "button";
    help.className = className || "kv-lab-help";
    help.textContent = "?";
    help.dataset.labTooltip = text;
    help.setAttribute("aria-label", text);
    return help;
  }

  function startSyncJob(input) {
    let timer = null;
    let cancelled = false;
    const promise = new Promise((resolve, reject) => {
      let trace = null;
      let simulationTrace = null;
      let point = null;
      let pointIndex = 0;
      let policyIndex = 0;
      let sweep = null;

      function step() {
        if (cancelled) return;
        try {
          if (!trace) {
            trace = generateTrace(input.preset, input.params, input.seed || DEFAULT_SEED);
            simulationTrace = Object.assign({}, trace, { normalizedRequests: normalizeRequests(trace) });
            const blockSize = toPositiveInteger(input.settings && input.settings.blockSize, trace.blockSize || DEFAULT_BLOCK_SIZE);
            const accountingSettings = Object.assign({}, input.settings || {}, { estimateTokens: traceEstimateTokens(trace, input.model) });
            const bytesPerToken = estimateBytesPerToken(input.model, accountingSettings);
            sweep = {
              blockSize,
              bytesPerToken,
              bytesPerBlock: bytesPerToken * blockSize,
              points: [],
              policies: (input.settings && input.settings.policies) || POLICIES,
              values: capacityValues(input.settings || {}),
            };
          }

          if (!point) {
            const gib = sweep.values[pointIndex];
            point = { gib, cacheBlocks: cacheBlocksForGiB(gib, sweep.bytesPerBlock), results: {} };
          }

          const policy = sweep.policies[policyIndex];
          point.results[policy] = simulatePolicy(simulationTrace, point.cacheBlocks, policy, {
            warmupFraction: input.settings && input.settings.warmupFraction,
          });
          policyIndex += 1;

          if (policyIndex >= sweep.policies.length) {
            sweep.points.push(point);
            point = null;
            policyIndex = 0;
            pointIndex += 1;
          }

          if (pointIndex >= sweep.values.length) {
            delete sweep.values;
            resolve({
              jobId: input.jobId,
              cacheKey: input.cacheKey,
              preset: input.preset,
              trace,
              sweep,
            });
            return;
          }
          timer = setTimeout(step, FALLBACK_STEP_DELAY_MS);
        } catch (error) {
          reject(error);
        }
      }

      timer = setTimeout(step, 0);
    });
    return {
      promise,
      cancel() {
        cancelled = true;
        if (timer) clearTimeout(timer);
      },
    };
  }

  function startWorkerJob(input, options) {
    const worker = new Worker(options.workerUrl);
    let settled = false;
    const promise = new Promise((resolve, reject) => {
      worker.onmessage = (event) => {
        settled = true;
        worker.terminate();
        const message = event.data || {};
        if (message.error) reject(new Error(message.error));
        else resolve(message.result || message);
      };
      worker.onerror = (event) => {
        settled = true;
        worker.terminate();
        reject(new Error(event.message || "Worker failed"));
      };
      worker.postMessage(
        Object.assign({}, input, {
          type: "run",
          calculatorScriptUrl: options.calculatorUrl,
          labScriptUrl: options.labUrl,
        }),
      );
    });
    return {
      promise,
      cancel() {
        if (!settled) worker.terminate();
      },
    };
  }

  function setSelectOptions(select, options, preferredValue, labelForOption) {
    if (!select) return;
    select.innerHTML = "";
    options.forEach((option) => {
      const item = document.createElement("option");
      item.value = option.id || option;
      item.textContent = labelForOption ? labelForOption(option) : option.label || String(option);
      select.appendChild(item);
    });
    const values = options.map((option) => option.id || option);
    select.value = values.includes(preferredValue) ? preferredValue : values[0];
  }

  function renderMetric(label, value) {
    const item = document.createElement("div");
    const key = document.createElement("span");
    key.className = "kv-lab-metric-label";
    const text = document.createElement("span");
    text.textContent = label;
    key.appendChild(text);
    if (METRIC_HELP[label]) key.appendChild(createHelpButton(METRIC_HELP[label], "kv-lab-help kv-lab-metric-help"));
    const val = document.createElement("strong");
    val.textContent = value;
    item.append(key, val);
    return item;
  }

  function renderMetrics(root, trace, sweep) {
    const list = root.querySelector("[data-lab-metrics]");
    if (!list) return;
    const maxPoint = sweep.points[sweep.points.length - 1];
    const summary = (trace && trace.summary) || {};
    const requestCount = Array.isArray(trace && trace.requests) ? trace.requests.length : summary.requests || 0;
    const warmupRequests =
      Number.isFinite(Number(sweep.warmupRequests))
        ? Number(sweep.warmupRequests)
        : maxPoint && maxPoint.results && maxPoint.results.lru
          ? maxPoint.results.lru.warmupRequests
          : summary.warmupRequests || 0;
    const averageInputTokens =
      Number.isFinite(Number(summary.averageInputTokens))
        ? Number(summary.averageInputTokens)
        : 0;
    const metrics = [
      ["Trace requests", formatInteger(requestCount)],
      ["Warmup skipped", formatInteger(warmupRequests)],
      ["Avg input tokens", formatNumber(averageInputTokens, 0)],
    ];
    if (Number.isFinite(Number(summary.uniqueBlocks))) metrics.push(["Unique blocks", formatInteger(summary.uniqueBlocks)]);
    if (Number.isFinite(Number(sweep.reuseCeiling))) metrics.push(["Hit rate ceiling", formatPercent(Number(sweep.reuseCeiling))]);
    if (Number.isFinite(Number(sweep.blockSize || trace.blockSize))) metrics.push(["Native block size", `${formatInteger(sweep.blockSize || trace.blockSize)} tokens`]);
    if (maxPoint) metrics.push(["Max cache blocks", formatInteger(maxPoint.cacheBlocks)]);
    list.innerHTML = "";
    metrics.forEach(([label, value]) => list.appendChild(renderMetric(label, value)));
  }

  function clearSvg(svg) {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
  }

  function svgNode(name, attrs) {
    const node = document.createElementNS("http://www.w3.org/2000/svg", name);
    Object.entries(attrs || {}).forEach(([key, value]) => node.setAttribute(key, value));
    return node;
  }

  function renderChart(root, sweep) {
    const svg = root.querySelector("[data-lab-chart]");
    if (!svg) return;
    clearSvg(svg);
    const width = 780;
    const height = 380;
    const margin = { top: 42, right: 32, bottom: 54, left: 64 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    const xScale = (index) => margin.left + (index / Math.max(1, sweep.points.length - 1)) * plotWidth;
    const yScale = (value) => margin.top + (1 - clamp(value, 0, 1)) * plotHeight;

    svg.appendChild(svgNode("rect", { x: 0, y: 0, width, height, fill: "#ffffff" }));
    [0, 0.25, 0.5, 0.75, 1].forEach((tick) => {
      const y = yScale(tick);
      svg.appendChild(svgNode("line", { x1: margin.left, y1: y, x2: width - margin.right, y2: y, stroke: "#e2e8f0", "stroke-width": 1 }));
      const label = svgNode("text", { x: margin.left - 12, y: y + 4, "text-anchor": "end", "font-size": 12, fill: "#64748b" });
      label.textContent = formatPercent(tick);
      svg.appendChild(label);
    });

    sweep.points.forEach((point, index) => {
      const x = xScale(index);
      svg.appendChild(svgNode("line", { x1: x, y1: margin.top, x2: x, y2: height - margin.bottom, stroke: "#f1f5f9", "stroke-width": 1 }));
      const label = svgNode("text", { x, y: height - 20, "text-anchor": "middle", "font-size": 12, fill: "#64748b" });
      label.textContent = formatCapacityGiB(point.gib);
      svg.appendChild(label);
    });

    svg.appendChild(svgNode("line", { x1: margin.left, y1: height - margin.bottom, x2: width - margin.right, y2: height - margin.bottom, stroke: "#94a3b8", "stroke-width": 1.2 }));
    svg.appendChild(svgNode("line", { x1: margin.left, y1: margin.top, x2: margin.left, y2: height - margin.bottom, stroke: "#94a3b8", "stroke-width": 1.2 }));

    if (Number.isFinite(Number(sweep.reuseCeiling))) {
      const ceilingY = yScale(Number(sweep.reuseCeiling));
      svg.appendChild(svgNode("line", {
        x1: margin.left,
        y1: ceilingY,
        x2: width - margin.right,
        y2: ceilingY,
        stroke: "#475569",
        "stroke-width": 1.4,
        "stroke-dasharray": "6 5",
      }));
      const ceilingLabel = svgNode("text", {
        x: width - margin.right - 12,
        y: Math.max(30, ceilingY - 10),
        "text-anchor": "end",
        "font-size": 11,
        fill: "#475569",
        "font-weight": 700,
      });
      ceilingLabel.textContent = `hit rate ceiling ${formatPercent(Number(sweep.reuseCeiling))}`;
      svg.appendChild(ceilingLabel);
    }

    const tooltip = svgNode("g", { visibility: "hidden", display: "none", "pointer-events": "none" });
    const tooltipBox = svgNode("rect", { width: 176, height: 76, rx: 6, fill: "#0f172a", opacity: 0.94 });
    const tooltipTitle = svgNode("text", { fill: "#ffffff", "font-size": 12, "font-weight": 700 });
    const tooltipRate = svgNode("text", { fill: "#dbeafe", "font-size": 12 });
    const tooltipBlocks = svgNode("text", { fill: "#cbd5e1", "font-size": 12 });
    tooltip.append(tooltipBox, tooltipTitle, tooltipRate, tooltipBlocks);

    function showTooltip(point, pointIndex, policy) {
      const x = xScale(pointIndex);
      const y = yScale(point.results[policy].hitRate);
      const boxWidth = 176;
      const boxHeight = 76;
      const boxX = Math.min(width - margin.right - boxWidth, Math.max(margin.left, x + 12));
      const boxY = Math.max(10, y - boxHeight - 12);
      tooltip.removeAttribute("display");
      tooltip.setAttribute("visibility", "visible");
      tooltipBox.setAttribute("width", boxWidth);
      tooltipBox.setAttribute("height", boxHeight);
      tooltipBox.setAttribute("x", boxX);
      tooltipBox.setAttribute("y", boxY);
      tooltipTitle.setAttribute("x", boxX + 12);
      tooltipTitle.setAttribute("y", boxY + 22);
      tooltipRate.setAttribute("x", boxX + 12);
      tooltipRate.setAttribute("y", boxY + 43);
      tooltipBlocks.setAttribute("x", boxX + 12);
      tooltipBlocks.setAttribute("y", boxY + 63);
      tooltipBlocks.setAttribute("visibility", "visible");
      tooltipTitle.textContent = `${POLICY_LABELS[policy]} @ ${formatCapacityGiB(point.gib)}`;
      tooltipRate.textContent = `Hit rate: ${formatPercent(point.results[policy].hitRate)}`;
      tooltipBlocks.textContent = `Cache blocks: ${formatInteger(point.cacheBlocks)}`;
    }

    function hideTooltip() {
      tooltip.setAttribute("visibility", "hidden");
      tooltip.setAttribute("display", "none");
      tooltipTitle.textContent = "";
      tooltipRate.textContent = "";
      tooltipBlocks.textContent = "";
    }

    function isTooltipTarget(node) {
      return Boolean(node && node.getAttribute && node.getAttribute("data-lab-tooltip-target") === "true");
    }

    function showPolicyHelp(policy, x, y) {
      const lines = POLICY_TOOLTIP_LINES[policy] || [POLICY_HELP[policy] || "", ""];
      const hasSecondLine = Boolean(lines[1]);
      const boxWidth = policy === "optimal" ? 430 : 220;
      const boxHeight = hasSecondLine ? 72 : 52;
      const boxX = Math.min(width - margin.right - boxWidth, Math.max(margin.left, x + 10));
      const boxY = Math.max(10, y + 10);
      tooltip.removeAttribute("display");
      tooltip.setAttribute("visibility", "visible");
      tooltipBox.setAttribute("width", boxWidth);
      tooltipBox.setAttribute("height", boxHeight);
      tooltipBox.setAttribute("x", boxX);
      tooltipBox.setAttribute("y", boxY);
      tooltipTitle.setAttribute("x", boxX + 12);
      tooltipTitle.setAttribute("y", boxY + 22);
      tooltipRate.setAttribute("x", boxX + 12);
      tooltipRate.setAttribute("y", boxY + 44);
      tooltipBlocks.setAttribute("x", boxX + 12);
      tooltipBlocks.setAttribute("y", boxY + 64);
      tooltipTitle.textContent = POLICY_LABELS[policy];
      tooltipRate.textContent = lines[0] || "";
      tooltipBlocks.textContent = lines[1] || "";
      tooltipBlocks.setAttribute("visibility", hasSecondLine ? "visible" : "hidden");
    }

    sweep.policies.forEach((policy, index) => {
      const points = sweep.points.map((point, pointIndex) => `${xScale(pointIndex)},${yScale(point.results[policy].hitRate)}`).join(" ");
      svg.appendChild(svgNode("polyline", { points, fill: "none", stroke: POLICY_COLORS[policy], "stroke-width": 3, "stroke-linejoin": "round", "stroke-linecap": "round" }));
      sweep.points.forEach((point, pointIndex) => {
        const tooltipLabel = `${POLICY_LABELS[policy]} @ ${formatCapacityGiB(point.gib)}; Hit rate: ${formatPercent(point.results[policy].hitRate)}; Cache blocks: ${formatInteger(point.cacheBlocks)}`;
        const marker = svgNode("circle", {
          class: "kv-lab-point",
          cx: xScale(pointIndex),
          cy: yScale(point.results[policy].hitRate),
          r: 5,
          fill: POLICY_COLORS[policy],
          stroke: "#ffffff",
          "stroke-width": 2,
          tabindex: 0,
          focusable: "true",
          "aria-label": tooltipLabel,
          "data-lab-tooltip-target": "true",
          cursor: "pointer",
        });
        marker.addEventListener("pointerenter", () => showTooltip(point, pointIndex, policy));
        marker.addEventListener("pointermove", () => showTooltip(point, pointIndex, policy));
        marker.addEventListener("mouseenter", () => showTooltip(point, pointIndex, policy));
        marker.addEventListener("mousemove", () => showTooltip(point, pointIndex, policy));
        marker.addEventListener("focus", () => showTooltip(point, pointIndex, policy));
        marker.addEventListener("click", () => showTooltip(point, pointIndex, policy));
        marker.addEventListener("pointerleave", hideTooltip);
        marker.addEventListener("mouseleave", hideTooltip);
        marker.addEventListener("blur", hideTooltip);
        svg.appendChild(marker);
      });
      const legendX = margin.left + index * 130;
      svg.appendChild(svgNode("line", { x1: legendX, y1: 22, x2: legendX + 28, y2: 22, stroke: POLICY_COLORS[policy], "stroke-width": 4, "stroke-linecap": "round" }));
      const label = svgNode("text", {
        x: legendX + 36,
        y: 26,
        "font-size": 13,
        fill: "#0f172a",
        "font-weight": 700,
        tabindex: 0,
        focusable: "true",
        "aria-label": `${POLICY_LABELS[policy]}: ${POLICY_HELP[policy]}`,
        "data-lab-tooltip-target": "true",
        cursor: "help",
      });
      label.textContent = POLICY_LABELS[policy];
      label.addEventListener("pointerenter", () => showPolicyHelp(policy, legendX + 36, 26));
      label.addEventListener("pointermove", () => showPolicyHelp(policy, legendX + 36, 26));
      label.addEventListener("mouseenter", () => showPolicyHelp(policy, legendX + 36, 26));
      label.addEventListener("mousemove", () => showPolicyHelp(policy, legendX + 36, 26));
      label.addEventListener("focus", () => showPolicyHelp(policy, legendX + 36, 26));
      label.addEventListener("click", () => showPolicyHelp(policy, legendX + 36, 26));
      label.addEventListener("pointerleave", hideTooltip);
      label.addEventListener("mouseleave", hideTooltip);
      label.addEventListener("blur", hideTooltip);
      svg.appendChild(label);
    });

    svg.onpointermove = (event) => {
      if (!isTooltipTarget(event.target)) hideTooltip();
    };
    svg.onmousemove = (event) => {
      if (!isTooltipTarget(event.target)) hideTooltip();
    };
    svg.onpointerleave = hideTooltip;
    svg.onmouseleave = hideTooltip;
    svg.appendChild(tooltip);

    const yLabelX = 13;
    const yLabel = svgNode("text", { x: yLabelX, y: margin.top + plotHeight / 2, transform: `rotate(-90 ${yLabelX} ${margin.top + plotHeight / 2})`, "text-anchor": "middle", "font-size": 12, fill: "#475569", "font-weight": 700 });
    yLabel.textContent = "Prefill token hit rate";
    svg.appendChild(yLabel);

    const xLabel = svgNode("text", { x: margin.left + plotWidth / 2, y: height - 5, "text-anchor": "middle", "font-size": 12, fill: "#475569", "font-weight": 700 });
    xLabel.textContent = "KV cache budget";
    svg.appendChild(xLabel);
  }

  function renderSources(root, preset, metadata) {
    const node = root.querySelector("[data-lab-sources]");
    if (!node) return;
    const sources = metadata && metadata.sources ? metadata.sources : {};
    node.innerHTML = "";
    const links = (preset.sources || []).map((sourceId) => [sourceId, sources[sourceId]]).filter(([, href]) => href);
    if (!links.length) return;
    node.appendChild(document.createTextNode("Sources: "));
    links.forEach(([sourceId, href], index) => {
      if (index > 0) node.appendChild(document.createTextNode(", "));
      const link = document.createElement("a");
      link.href = href;
      link.target = "_blank";
      link.rel = "noopener";
      link.textContent = sourceLabel(sourceId);
      node.appendChild(link);
    });
  }

  function renderResults(root, preset, trace, sweep, metadata) {
    const isPrecomputed = sweep && sweep.precomputed;
    const sourceKind = trace && trace.sourceKind;
    const modeNote = isPrecomputed
      ? "This chart uses offline precomputed curves from a normalized real trace; the browser does not replay the full trace."
      : "Deterministic simulation converts the selected preset into block requests, skips the warmup window, and reports prefill input-token hits.";
    const identityNote =
      sourceKind === "agent_text"
        ? "This agent trace is approximated from message/span text because the source does not publish block ids."
        : sourceKind === "hash"
          ? "Repeated hash/block ids define cache hits."
          : "";
    setText(
      root,
      "[data-lab-output='note']",
      `${preset.summary || ""} ${modeNote} ${identityNote} ${trace.sourceBlockSizeNote || metadata.block_size_note || ""}`,
    );
    renderMetrics(root, trace, sweep);
    renderChart(root, sweep);
    renderSources(root, preset, metadata);
  }

  function renderUnavailable(root, preset, model, settings) {
    const metrics = root.querySelector("[data-lab-metrics]");
    const svg = root.querySelector("[data-lab-chart]");
    const sources = root.querySelector("[data-lab-sources]");
    if (metrics) metrics.innerHTML = "";
    if (svg) clearSvg(svg);
    if (sources) sources.innerHTML = "";
    const precision = settings && settings.precision ? settings.precision : "default";
    const indexer = settings && settings.indexerPrecision ? `, indexer ${settings.indexerPrecision}` : "";
    setText(
      root,
      "[data-lab-output='note']",
      `No precomputed curve is available for ${preset.label} with ${model.label} (${precision}${indexer}). Run scripts/kv-cache-lab-precompute-curves.mjs for this model/precision combination, then rebuild the Hugo data.`,
    );
  }

  function initialize(root, data, options) {
    const models = data.models || [];
    const precomputed = data.precomputed || null;
    const rawPresets = data.lab && data.lab.presets ? data.lab.presets : [];
    const presets = precomputed && precomputed.traces
      ? rawPresets.filter((preset) => precomputedTrace(precomputed, preset))
      : rawPresets;
    if (!root || !models.length || !presets.length) return;
    const runtimeOptions = options || {};
    const defaults = (data.lab && data.lab.simulation_defaults) || {};
    const metadata = (data.lab && data.lab.metadata) || {};
    const resultCache = new Map();
    let debounceTimer = null;
    let latestJobId = 0;
    let activeJob = null;
    const inputs = {
      modelFamily: root.querySelector("[data-lab-input='modelFamily']"),
      model: root.querySelector("[data-lab-input='model']"),
      precision: root.querySelector("[data-lab-input='precision']"),
      indexerPrecision: root.querySelector("[data-lab-input='indexerPrecision']"),
      includeDraftKvCache: root.querySelector("[data-lab-input='includeDraftKvCache']"),
      preset: root.querySelector("[data-lab-input='preset']"),
    };
    const paramInputs = Array.from(root.querySelectorAll("[data-lab-param]"));

    function selectedModel() {
      return modelById(models, inputs.model.value);
    }

    function selectedPreset() {
      return presetById(presets, inputs.preset.value);
    }

    function syncTraceHelp() {
      const help = root.querySelector("[data-lab-trace-help]");
      if (!help) return;
      const text = tracePresetHelpText(selectedPreset());
      help.dataset.labTooltip = text;
      help.setAttribute("aria-label", text);
      help.removeAttribute("title");
    }

    function setupTraceHelp() {
      const help = root.querySelector("[data-lab-trace-help]");
      if (!help) return;
      help.addEventListener("click", (event) => {
        event.preventDefault();
      });
    }

    function populateFamilies(preferredFamily) {
      const availableIds = availableModelIdsFor(precomputed, selectedPreset());
      const familyModels = availableIds ? models.filter((model) => availableIds.has(model.id)) : models;
      setSelectOptions(inputs.modelFamily, sortedModelFamilies(familyModels), preferredFamily, (family) => family);
    }

    function populateModels(preferredModelId) {
      const availableIds = availableModelIdsFor(precomputed, selectedPreset());
      const familyModels = modelsForFamily(models, inputs.modelFamily.value).filter((model) => !availableIds || availableIds.has(model.id));
      setSelectOptions(inputs.model, familyModels, preferredModelId, (model) => model.label);
    }

    function syncPrecisionControls(model) {
      const availableSettings = availableSettingsFor(precomputed, selectedPreset(), model);
      const availablePrecisions = availableSettings.length
        ? Array.from(new Set(availableSettings.map((setting) => setting.precision).filter(Boolean)))
        : [];
      const precisionOptions = availablePrecisions.length
        ? (data.precision_options || []).filter((option) => availablePrecisions.includes(option.id))
        : data.precision_options || [];
      const precisionDefault = availablePrecisions[0] || defaultPrecisionId(model, data.precision_options || []);
      setSelectOptions(inputs.precision, precisionOptions, precisionDefault, (option) => option.label);

      const indexerControl = root.querySelector("[data-lab-indexer-control]");
      const availableIndexerPrecisions = availableSettings.length
        ? Array.from(new Set(availableSettings.map((setting) => setting.indexerPrecision).filter(Boolean)))
        : [];
      const showIndexer = hasIndexerCache(model) && (!availableSettings.length || availableIndexerPrecisions.length > 0);
      if (indexerControl) indexerControl.hidden = !showIndexer;
      if (showIndexer) {
        const indexerOptions = availableIndexerPrecisions.length
          ? (data.indexer_precision_options || []).filter((option) => availableIndexerPrecisions.includes(option.id))
          : data.indexer_precision_options || [];
        const indexerDefault = availableIndexerPrecisions[0] || defaultIndexerPrecisionId(model, data.indexer_precision_options || [], inputs.precision ? inputs.precision.value : undefined);
        setSelectOptions(inputs.indexerPrecision, indexerOptions, indexerDefault, (option) => option.label);
      }
    }

    function syncDraftControl(model) {
      const control = root.querySelector("[data-lab-draft-control]");
      const availableSettings = availableSettingsFor(precomputed, selectedPreset(), model);
      const showDraft = hasDraftKvCache(model) && (!availableSettings.length || availableSettings.some((setting) => setting.includeDraftKvCache));
      if (control) control.hidden = !showDraft;
      if (inputs.includeDraftKvCache) inputs.includeDraftKvCache.checked = false;
    }

    function syncModelControls() {
      const model = selectedModel();
      syncPrecisionControls(model);
      syncDraftControl(model);
    }

    function applyPresetDefaults() {
      const preset = selectedPreset();
      paramInputs.forEach((input) => {
        const key = input.dataset.labParam;
        if (key === "requests") input.value = defaults.requests || DEFAULT_REQUESTS;
        else if (preset.defaults && Object.prototype.hasOwnProperty.call(preset.defaults, key)) input.value = preset.defaults[key];
      });
    }

    function readParams() {
      const params = { requests: defaults.requests || DEFAULT_REQUESTS };
      paramInputs.forEach((input) => {
        params[input.dataset.labParam] = input.value;
      });
      params.blockSize = selectedPreset().native_block_size || selectedPreset().nativeBlockSize || defaults.block_size || DEFAULT_BLOCK_SIZE;
      return params;
    }

    function readSettings(blockSize) {
      const model = selectedModel();
      return {
        precision: inputs.precision.value,
        indexerPrecision: hasIndexerCache(model) && inputs.indexerPrecision ? inputs.indexerPrecision.value : undefined,
        includeDraftKvCache: hasDraftKvCache(model) && checkboxValue(inputs.includeDraftKvCache),
        precisionOptions: data.precision_options,
        indexerPrecisionOptions: data.indexer_precision_options,
        blockSize,
        capacityGiBValues: defaults.capacity_gib_values || DEFAULT_CAPACITY_GIB_VALUES,
        warmupFraction: defaults.warmup_fraction || DEFAULT_WARMUP_FRACTION,
      };
    }

    function computationInput() {
      const preset = selectedPreset();
      const blockSize = toPositiveInteger(preset.native_block_size || preset.nativeBlockSize || defaults.block_size, DEFAULT_BLOCK_SIZE);
      return {
        preset,
        model: selectedModel(),
        params: readParams(),
        settings: readSettings(blockSize),
        seed: defaults.seed || DEFAULT_SEED,
      };
    }

    function startJob(input) {
      const canUseWorker =
        runtimeOptions.workerUrl &&
        runtimeOptions.calculatorUrl &&
        runtimeOptions.labUrl &&
        typeof Worker === "function";
      if (canUseWorker) {
        root.dataset.computeMode = "worker";
        try {
          return startWorkerJob(input, runtimeOptions);
        } catch (error) {
          root.dataset.computeMode = "fallback";
          return startSyncJob(input);
        }
      }
      root.dataset.computeMode = "fallback";
      return startSyncJob(input);
    }

    function applyResult(result, fromCache) {
      renderResults(root, result.preset || selectedPreset(), result.trace, result.sweep, metadata);
      root.dataset.state = "ready";
      setStatus(root, fromCache ? "Cached" : result.sweep && result.sweep.precomputed ? "Precomputed" : "Ready");
    }

    function update() {
      try {
        const baseInput = computationInput();
        const precomputedResult = precomputedResultFor(precomputed, baseInput.preset, baseInput.model, baseInput.settings);
        if (precomputed && precomputed.traces) {
          if (precomputedResult) {
            if (activeJob) {
              activeJob.cancel();
              activeJob = null;
            }
            latestJobId += 1;
            applyResult(precomputedResult, false);
          } else {
            if (activeJob) {
              activeJob.cancel();
              activeJob = null;
            }
            latestJobId += 1;
            root.dataset.state = "error";
            setStatus(root, "Unavailable");
            renderUnavailable(root, baseInput.preset, baseInput.model, baseInput.settings);
          }
          return;
        }
        const cacheKey = createCacheKey(baseInput);
        const jobId = latestJobId + 1;
        latestJobId = jobId;
        if (activeJob) {
          activeJob.cancel();
          activeJob = null;
        }
        if (resultCache.has(cacheKey)) {
          applyResult(resultCache.get(cacheKey), true);
          return;
        }

        root.dataset.state = "calculating";
        setStatus(root, "Calculating...");
        activeJob = startJob(Object.assign({ jobId, cacheKey }, baseInput));
        activeJob.promise
          .then((result) => {
            if (!shouldApplyJobResult(latestJobId, result)) return;
            activeJob = null;
            rememberCachedResult(resultCache, cacheKey, result);
            applyResult(result, false);
          })
          .catch((error) => {
            if (jobId !== latestJobId) return;
            activeJob = null;
            root.dataset.state = "error";
            setStatus(root, "Calculation failed");
            setText(root, "[data-lab-output='note']", error.message);
          });
      } catch (error) {
        root.dataset.state = "error";
        setStatus(root, "Calculation failed");
        setText(root, "[data-lab-output='note']", error.message);
      }
    }

    function scheduleUpdate(delay) {
      if (debounceTimer) clearTimeout(debounceTimer);
      const wait = Math.max(0, delay || 0);
      if (wait === 0) {
        update();
        return;
      }
      debounceTimer = setTimeout(() => {
        debounceTimer = null;
        update();
      }, wait);
    }

    const defaultModel = models[0];
    populateFamilies(modelFamily(defaultModel));
    setSelectOptions(inputs.preset, presets, presets[0].id, (preset) => preset.label);
    populateModels(defaultModel.id);
    syncModelControls();
    applyPresetDefaults();
    setupTraceHelp();
    syncTraceHelp();

    inputs.modelFamily.addEventListener("change", () => {
      populateModels();
      syncModelControls();
      scheduleUpdate(0);
    });
    inputs.model.addEventListener("change", () => {
      syncModelControls();
      scheduleUpdate(0);
    });
    inputs.preset.addEventListener("change", () => {
      applyPresetDefaults();
      populateFamilies(inputs.modelFamily.value);
      populateModels(inputs.model.value);
      syncModelControls();
      syncTraceHelp();
      scheduleUpdate(0);
    });
    Object.values(inputs).forEach((input) => {
      if (!input || input === inputs.modelFamily || input === inputs.model || input === inputs.preset) return;
      input.addEventListener("input", () => scheduleUpdate(INPUT_DEBOUNCE_MS));
      input.addEventListener("change", () => scheduleUpdate(0));
    });
    paramInputs.forEach((input) => {
      input.addEventListener("input", () => scheduleUpdate(INPUT_DEBOUNCE_MS));
      input.addEventListener("change", () => scheduleUpdate(0));
    });

    scheduleUpdate(0);
  }

  function mount(rootId, data, options) {
    const rootNode = document.getElementById(rootId);
    const runtimeOptions = options || {};
    if (
      runtimeOptions.precomputedUrl &&
      data &&
      !data.precomputed &&
      typeof fetch === "function"
    ) {
      if (rootNode) {
        rootNode.dataset.state = "loading";
        setStatus(rootNode, "Loading trace data...");
      }
      fetch(runtimeOptions.precomputedUrl)
        .then((response) => {
          if (!response.ok) {
            throw new Error(`Failed to load precomputed trace data (${response.status})`);
          }
          return response.json();
        })
        .then((precomputed) => {
          initialize(rootNode, Object.assign({}, data, { precomputed }), runtimeOptions);
        })
        .catch((error) => {
          if (!rootNode) return;
          rootNode.dataset.state = "error";
          setStatus(rootNode, "Trace data failed");
          setText(rootNode, "[data-lab-output='note']", error.message);
        });
      return;
    }
    initialize(rootNode, data, runtimeOptions);
  }

  return {
    BYTES_PER_GIB,
    DEFAULT_CAPACITY_GIB_VALUES,
    POLICY_LABELS,
    cacheBlocksForGiB,
    createCacheKey,
    estimateBytesPerToken,
    generateTrace,
    mount,
    modelSweepKey,
    precomputedResultFor,
    runLabComputation,
    simulatePolicy,
    shouldApplyJobResult,
    sweepCapacity,
  };
});

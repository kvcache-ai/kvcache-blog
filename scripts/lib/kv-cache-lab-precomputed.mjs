export const PRECOMPUTED_SCHEMA_VERSION = 2;
export const SIMULATION_SEMANTICS_VERSION =
  "prefix-hit-context-aware-fifo-block-lru-trie-optimal-belady-bypass-v1";

const DEFAULT_POLICIES = ["fifo", "lru", "optimal"];

function finiteNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : undefined;
}

function parseSweepKey(key) {
  const parts = String(key || "").split("|");
  const setting = { modelId: parts[0] || "" };
  for (const part of parts.slice(1)) {
    const separator = part.indexOf("=");
    if (separator < 0) continue;
    const name = part.slice(0, separator);
    const value = part.slice(separator + 1);
    if (name === "precision") setting.precision = value || undefined;
    if (name === "indexer") setting.indexerPrecision = value || undefined;
    if (name === "draft") setting.includeDraftKvCache = value === "1";
  }
  return setting;
}

export function isCompactTrace(trace) {
  return Boolean(trace && trace.settings && trace.capacityResults && !trace.modelSweeps);
}

export function decodeCompactSetting(value) {
  if (!Array.isArray(value) || !Array.isArray(value[1])) return null;
  const bytesPerToken = finiteNumber(value[0]);
  if (!(bytesPerToken > 0)) return null;
  const pairs = value[1];
  if (pairs.length % 2 !== 0) return null;
  const points = [];
  for (let index = 0; index < pairs.length; index += 2) {
    const gib = finiteNumber(pairs[index]);
    const cacheBlocks = finiteNumber(pairs[index + 1]);
    if (!(gib >= 0) || !(cacheBlocks >= 0)) return null;
    points.push({ gib, cacheBlocks: Math.floor(cacheBlocks) });
  }
  return { bytesPerToken, points };
}

function policyHitTokens(point, policies) {
  return policies.map((policy) => {
    const result = point && point.results && point.results[policy];
    const hitTokens = result && finiteNumber(result.hitTokens);
    if (!(hitTokens >= 0)) throw new Error(`Missing ${policy} hitTokens for cacheBlocks=${point && point.cacheBlocks}`);
    return hitTokens;
  });
}

function sameNumberArray(left, right) {
  return Array.isArray(left)
    && Array.isArray(right)
    && left.length === right.length
    && left.every((value, index) => Number(value) === Number(right[index]));
}

export function compactTrace(trace, options = {}) {
  if (!trace || typeof trace !== "object") return trace;
  if (isCompactTrace(trace)) {
    const summary = trace.summary || {};
    const totalMeasuredTokens = finiteNumber(summary.totalMeasuredTokens);
    const infiniteHitRate = finiteNumber(summary.infiniteHitRate);
    const infiniteHitTokens = finiteNumber(summary.infiniteHitTokens)
      ?? (infiniteHitRate !== undefined && totalMeasuredTokens !== undefined
        ? Math.round(infiniteHitRate * totalMeasuredTokens)
        : undefined);
    return {
      ...trace,
      summary: {
        ...summary,
        ...(infiniteHitTokens !== undefined ? { infiniteHitTokens } : {}),
      },
    };
  }
  if (trace.blockCapacityCurve) return trace;
  const modelSweeps = trace.modelSweeps || {};
  const policies = options.policies || DEFAULT_POLICIES;
  const settings = {};
  const capacityResults = {};
  let totalMeasuredTokens = finiteNumber(trace.summary && trace.summary.totalMeasuredTokens);

  for (const [key, sweep] of Object.entries(modelSweeps)) {
    const points = Array.isArray(sweep.points) ? sweep.points : [];
    const bytesPerToken = finiteNumber(sweep.bytesPerToken);
    if (!(bytesPerToken > 0)) throw new Error(`Missing bytesPerToken for ${key}`);
    settings[key] = [bytesPerToken, points.flatMap((point) => [Number(point.gib), Math.floor(Number(point.cacheBlocks))])];
    for (const point of points) {
      const cacheBlocks = Math.floor(Number(point.cacheBlocks));
      const hits = policyHitTokens(point, policies);
      const existing = capacityResults[cacheBlocks];
      if (existing && !sameNumberArray(existing, hits)) {
        throw new Error(`Conflicting simulation results for cacheBlocks=${cacheBlocks}`);
      }
      capacityResults[cacheBlocks] = hits;
      if (totalMeasuredTokens === undefined) {
        for (const policy of policies) {
          const result = point.results && point.results[policy];
          const candidate = result && finiteNumber(result.totalTokens);
          if (candidate !== undefined) {
            totalMeasuredTokens = candidate;
            break;
          }
        }
      }
    }
  }

  const { modelSweeps: _modelSweeps, ...rest } = trace;
  const infiniteHitRate = finiteNumber(trace.summary && trace.summary.infiniteHitRate);
  const infiniteHitTokens = finiteNumber(trace.summary && trace.summary.infiniteHitTokens)
    ?? (infiniteHitRate !== undefined && totalMeasuredTokens !== undefined
      ? Math.round(infiniteHitRate * totalMeasuredTokens)
      : undefined);
  return {
    ...rest,
    summary: {
      ...(trace.summary || {}),
      ...(totalMeasuredTokens !== undefined ? { totalMeasuredTokens } : {}),
      ...(infiniteHitTokens !== undefined ? { infiniteHitTokens } : {}),
    },
    settings,
    capacityResults,
  };
}

export function compactPrecomputed(data) {
  const source = data || {};
  const sourceSemantics = source.metadata && source.metadata.simulation_semantics;
  if (sourceSemantics && sourceSemantics !== SIMULATION_SEMANTICS_VERSION) {
    throw new Error(`Unsupported simulation semantics: ${sourceSemantics}`);
  }
  const sourceSchema = finiteNumber(source.metadata && source.metadata.schema_version);
  if (sourceSchema !== undefined && sourceSchema > PRECOMPUTED_SCHEMA_VERSION) {
    throw new Error(`Unsupported precomputed schema version: ${sourceSchema}`);
  }
  const metadata = {
    ...(source.metadata || {}),
    schema_version: PRECOMPUTED_SCHEMA_VERSION,
    simulation_semantics: SIMULATION_SEMANTICS_VERSION,
  };
  const policies = metadata.policies || DEFAULT_POLICIES;
  const traces = {};
  for (const [traceId, trace] of Object.entries(source.traces || {})) {
    traces[traceId] = compactTrace(trace, { policies });
  }
  return { metadata, traces };
}

export function mergeCompactTraces(existingTrace, incomingTrace, policies = DEFAULT_POLICIES) {
  const existing = compactTrace(existingTrace || {}, { policies });
  const incoming = compactTrace(incomingTrace || {}, { policies });
  if (!isCompactTrace(existing)) return incoming;
  if (!isCompactTrace(incoming)) return existing;
  const capacityResults = { ...(existing.capacityResults || {}) };
  for (const [capacity, hits] of Object.entries(incoming.capacityResults || {})) {
    if (capacityResults[capacity] && !sameNumberArray(capacityResults[capacity], hits)) {
      throw new Error(`Conflicting simulation results for cacheBlocks=${capacity}`);
    }
    capacityResults[capacity] = hits;
  }
  return {
    ...existing,
    ...incoming,
    summary: { ...(existing.summary || {}), ...(incoming.summary || {}) },
    settings: { ...(existing.settings || {}), ...(incoming.settings || {}) },
    capacityResults,
  };
}

function runtimeMetadataFromCompactTrace(trace, metadata) {
  const summary = (trace && trace.summary) || {};
  return {
    blockSize: trace && trace.nativeBlockSize,
    requestCount: summary.requests,
    totalInputTokens: summary.totalInputTokens,
    totalBlocks: summary.totalBlocks,
    warmupRequests: summary.warmupRequests,
    warmupFraction: metadata && metadata.warmup_fraction,
  };
}

export function mergeCompactPrecomputed(baseData, inputData) {
  const base = compactPrecomputed(baseData || { metadata: {}, traces: {} });
  const input = compactPrecomputed(inputData || { metadata: {}, traces: {} });
  const baseWarmup = finiteNumber(base.metadata && base.metadata.warmup_fraction);
  const inputWarmup = finiteNumber(input.metadata && input.metadata.warmup_fraction);
  if (baseWarmup !== undefined && inputWarmup !== undefined && baseWarmup !== inputWarmup) {
    throw new Error(`Cannot merge precomputed data with different warmup fractions: ${baseWarmup} vs ${inputWarmup}`);
  }
  const output = {
    metadata: {
      ...base.metadata,
      ...input.metadata,
      sources: { ...(base.metadata.sources || {}), ...(input.metadata.sources || {}) },
      reference_sources: input.metadata.reference_sources || base.metadata.reference_sources,
    },
    traces: { ...base.traces },
  };
  for (const [traceId, trace] of Object.entries(input.traces || {})) {
    const existing = output.traces[traceId];
    if (existing && !compactTraceMatchesSimulation(
      existing,
      base.metadata,
      runtimeMetadataFromCompactTrace(trace, input.metadata),
    )) {
      throw new Error(`Cannot merge incompatible precomputed trace: ${traceId}`);
    }
    output.traces[traceId] = existing
      ? mergeCompactTraces(existing, trace, output.metadata.policies)
      : trace;
  }
  return output;
}

export function compactPolicyResults(trace, cacheBlocks, policies = DEFAULT_POLICIES) {
  if (!trace || !trace.capacityResults) return null;
  const hits = trace.capacityResults[String(Math.floor(Number(cacheBlocks)))];
  const totalTokens = finiteNumber(trace.summary && trace.summary.totalMeasuredTokens);
  if (!Array.isArray(hits) || hits.length < policies.length || !(totalTokens >= 0)) return null;
  const warmupRequests = finiteNumber(trace.summary && trace.summary.warmupRequests) || 0;
  const results = {};
  policies.forEach((policy, index) => {
    const hitTokens = finiteNumber(hits[index]);
    if (!(hitTokens >= 0)) return;
    results[policy] = {
      policy,
      cacheBlocks: Math.floor(Number(cacheBlocks)),
      warmupRequests,
      measurementStartRequest: warmupRequests,
      measurementMode: "fixed_window",
      hitTokens,
      totalTokens,
      hitRate: totalTokens ? hitTokens / totalTokens : 0,
    };
  });
  return Object.keys(results).length === policies.length ? results : null;
}

export function expandCompactTrace(trace, metadata = {}) {
  if (!isCompactTrace(trace)) return trace;
  const policies = metadata.policies || DEFAULT_POLICIES;
  const blockSize = finiteNumber(trace.nativeBlockSize) || 64;
  const modelSweeps = {};
  for (const [key, encoded] of Object.entries(trace.settings || {})) {
    const setting = decodeCompactSetting(encoded);
    if (!setting) continue;
    const parsed = parseSweepKey(key);
    const points = [];
    for (const point of setting.points) {
      const results = compactPolicyResults(trace, point.cacheBlocks, policies);
      if (!results) break;
      points.push({ ...point, results });
    }
    modelSweeps[key] = {
      modelId: parsed.modelId,
      precision: parsed.precision,
      indexerPrecision: parsed.indexerPrecision || null,
      includeDraftKvCache: Boolean(parsed.includeDraftKvCache),
      blockSize,
      bytesPerToken: setting.bytesPerToken,
      bytesPerBlock: setting.bytesPerToken * blockSize,
      points,
      policies,
      reuseCeiling: finiteNumber(trace.summary && trace.summary.infiniteHitRate),
      warmupRequests: finiteNumber(trace.summary && trace.summary.warmupRequests) || 0,
      sourceKind: trace.sourceKind,
    };
  }
  const { settings: _settings, capacityResults: _capacityResults, ...rest } = trace;
  return { ...rest, modelSweeps };
}

function equalIfPresent(left, right) {
  const a = finiteNumber(left);
  const b = finiteNumber(right);
  return a === undefined || b === undefined || a === b;
}

export function compactTraceMatchesSimulation(trace, datasetMetadata, simulationMetadata) {
  if (!isCompactTrace(trace)) return false;
  if (Number(datasetMetadata && datasetMetadata.schema_version) !== PRECOMPUTED_SCHEMA_VERSION) return false;
  if ((datasetMetadata && datasetMetadata.simulation_semantics) !== SIMULATION_SEMANTICS_VERSION) return false;
  if (!equalIfPresent(datasetMetadata && datasetMetadata.warmup_fraction, simulationMetadata && simulationMetadata.warmupFraction)) return false;
  const summary = trace.summary || {};
  const runtimeSummary = (simulationMetadata && simulationMetadata.summary) || {};
  return equalIfPresent(trace.nativeBlockSize, simulationMetadata && simulationMetadata.blockSize)
    && equalIfPresent(summary.requests, simulationMetadata && (simulationMetadata.requestCount ?? runtimeSummary.requests))
    && equalIfPresent(summary.totalInputTokens, simulationMetadata && (simulationMetadata.totalInputTokens ?? runtimeSummary.totalInputTokens))
    && equalIfPresent(summary.totalBlocks, simulationMetadata && (simulationMetadata.totalBlocks ?? runtimeSummary.totalBlocks))
    && equalIfPresent(summary.warmupRequests, simulationMetadata && simulationMetadata.warmupRequests)
    && finiteNumber(summary.totalMeasuredTokens) !== undefined;
}

export function simulationResultsFromCompactTrace(trace, policies = DEFAULT_POLICIES) {
  const results = new Map();
  for (const capacity of Object.keys((trace && trace.capacityResults) || {})) {
    const policyResults = compactPolicyResults(trace, Number(capacity), policies);
    if (!policyResults) continue;
    for (const policy of policies) {
      results.set(`${policy}|${capacity}`, policyResults[policy]);
    }
  }
  return results;
}

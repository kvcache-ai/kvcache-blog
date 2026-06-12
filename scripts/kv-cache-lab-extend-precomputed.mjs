#!/usr/bin/env node
import fsp from "node:fs/promises";
import path from "node:path";

import {
  DEFAULT_CAPACITY_GIB_VALUES,
  DEFAULT_OUTPUT_PATH,
  DEFAULT_WARMUP_FRACTION,
  TRACE_SOURCES,
  buildExecutionPlan,
  infiniteCacheReuse,
  loadModelsData,
  normalizeTraceSource,
  precomputeSweep,
} from "./lib/kv-cache-lab-traces.mjs";

function parseArgs(argv) {
  const args = {
    inputPath: DEFAULT_OUTPUT_PATH,
    outputPath: DEFAULT_OUTPUT_PATH,
    capacityGiBValues: DEFAULT_CAPACITY_GIB_VALUES,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--input") args.inputPath = argv[++index];
    else if (arg === "--output") args.outputPath = argv[++index];
    else if (arg === "--models") args.modelsPath = argv[++index];
    else if (arg === "--cache-dir") args.cacheDir = argv[++index];
    else if (arg === "--trace") {
      args.traceIds = args.traceIds || [];
      args.traceIds.push(argv[++index]);
    } else if (arg === "--capacity-gib-values") {
      args.capacityGiBValues = argv[++index]
        .split(",")
        .map((value) => Number(value.trim()))
        .filter((value) => Number.isFinite(value) && value > 0);
    }
  }
  return args;
}

function pointByGiB(points) {
  return new Map((points || []).map((point) => [Number(point.gib), point]));
}

function sortPoints(points) {
  return points.slice().sort((left, right) => Number(left.gib) - Number(right.gib));
}

const options = parseArgs(process.argv.slice(2));
const inputPath = path.resolve(options.inputPath);
const outputPath = path.resolve(options.outputPath);
const data = JSON.parse(await fsp.readFile(inputPath, "utf8"));
const modelsData = loadModelsData(options.modelsPath);
const modelById = new Map(modelsData.models.map((model) => [model.id, model]));
const wantedTraceIds = new Set(options.traceIds || Object.keys(data.traces || {}));
const generatedAt = new Date().toISOString();
const updated = [];

for (const source of TRACE_SOURCES) {
  if (!wantedTraceIds.has(source.id) || !data.traces[source.id]) continue;
  const traceData = data.traces[source.id];
  const requestLimit = Number(traceData.summary?.requests);
  const trace = await normalizeTraceSource(source, {
    ...options,
    requestLimit: Number.isFinite(requestLimit) && requestLimit > 0 ? requestLimit : options.requestLimit,
  });
  const warmupFraction = data.metadata?.warmup_fraction ?? DEFAULT_WARMUP_FRACTION;
  const plan = buildExecutionPlan(trace, { warmupFraction });
  const ceiling = infiniteCacheReuse(trace, { warmupFraction });
  const simulationCache = new Map();
  let traceAdded = 0;
  console.error(`[extend] ${source.id}: ${Object.keys(traceData.modelSweeps || {}).length} sweeps`);

  for (const sweep of Object.values(traceData.modelSweeps || {})) {
    const model = modelById.get(sweep.modelId);
    if (!model) continue;
    const existingByGiB = pointByGiB(sweep.points);
    const missingGiBValues = options.capacityGiBValues.filter((gib) => !existingByGiB.has(gib));
    if (!missingGiBValues.length) continue;
    console.error(`[extend] ${source.id}: ${sweep.modelId} +${missingGiBValues.join(",")}GiB`);
    const additional = precomputeSweep(
      trace,
      model,
      {
        precision: sweep.precision,
        indexerPrecision: sweep.indexerPrecision || undefined,
        includeDraftKvCache: Boolean(sweep.includeDraftKvCache),
        blockSize: trace.blockSize,
        capacityGiBValues: missingGiBValues,
        warmupFraction,
        precisionOptions: modelsData.precision_options,
        indexerPrecisionOptions: modelsData.indexer_precision_options,
      },
      { generatedAt, plan, simulationCache, ceiling },
    );
    sweep.points = sortPoints([...(sweep.points || []), ...additional.points]);
    sweep.generatedAt = generatedAt;
    traceAdded += additional.points.length;
  }

  if (traceAdded) updated.push({ trace: source.id, points: traceAdded });
}

data.metadata = {
  ...(data.metadata || {}),
  generated_at: data.metadata?.generated_at || generatedAt,
  extended_at: generatedAt,
  capacity_gib_values: options.capacityGiBValues,
};

await fsp.mkdir(path.dirname(outputPath), { recursive: true });
await fsp.writeFile(outputPath, `${JSON.stringify(data, null, 2)}\n`);
console.log(JSON.stringify({ outputPath, updated }, null, 2));

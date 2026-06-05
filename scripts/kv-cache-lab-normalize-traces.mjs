#!/usr/bin/env node
import { TRACE_SOURCES, normalizeTraceSource, parseArgs } from "./lib/kv-cache-lab-traces.mjs";

const options = parseArgs(process.argv.slice(2));
const selected = new Set(options.traceIds || TRACE_SOURCES.map((source) => source.id));
const summaries = [];
for (const source of TRACE_SOURCES) {
  if (!selected.has(source.id)) continue;
  const trace = await normalizeTraceSource(source, options);
  summaries.push({
    id: trace.id,
    label: trace.label,
    blockSize: trace.blockSize,
    summary: trace.summary,
  });
}
console.log(JSON.stringify({ traces: summaries }, null, 2));

#!/usr/bin/env node
import { parseArgs, writePrecomputedCurves } from "./lib/kv-cache-lab-traces.mjs";

const options = parseArgs(process.argv.slice(2));
const { outputPath, data } = await writePrecomputedCurves(options);
console.log(JSON.stringify({ outputPath, traces: Object.keys(data.traces) }, null, 2));

#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import path from "node:path";

import { DEFAULT_TRACE_CACHE_DIR } from "./lib/kv-cache-lab-traces.mjs";

const SMALL_TRACES = ["mooncake_fast25", "ragpulse", "lmcache_agentic_sample"];
const FULL_TRACES = [
  "bailian_qwen_trace_a",
  "semianalysis_weka_no_subagents",
  "semianalysis_weka_with_subagents_256k",
  "kv_cache_tester_claude_code",
];

function parseArgs(argv) {
  const options = {
    cacheDir: DEFAULT_TRACE_CACHE_DIR,
    nativeJobs: 8,
    nativeSimPath: "/tmp/kv-cache-lab-native-sim-context-window",
    outputPath: "data/kv_cache_lab/precomputed.json",
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--cache-dir") options.cacheDir = argv[++index];
    else if (arg === "--native-jobs") options.nativeJobs = Math.max(1, Math.floor(Number(argv[++index])));
    else if (arg === "--native-sim") options.nativeSimPath = argv[++index];
    else if (arg === "--output") options.outputPath = argv[++index];
    else throw new Error(`Unknown argument: ${arg}`);
  }
  return options;
}

function run(script, traceIds, options, extraArgs = []) {
  const args = [
    script,
    "--all-models",
    "--all-precisions",
    "--dedupe-kv-architecture",
    "--no-draft",
    "--cache-dir",
    path.resolve(options.cacheDir),
    "--output",
    path.resolve(options.outputPath),
    ...extraArgs,
  ];
  for (const traceId of traceIds) args.push("--trace", traceId);
  console.error(`$ node ${args.join(" ")}`);
  const result = spawnSync("node", args, { stdio: "inherit" });
  if (result.status !== 0) throw new Error(`${script} exited with status ${result.status}`);
}

const options = parseArgs(process.argv.slice(2));
for (const traceId of SMALL_TRACES) {
  run("scripts/kv-cache-lab-precompute-curves.mjs", [traceId], options);
}
run("scripts/kv-cache-lab-full-precompute.mjs", FULL_TRACES, options, [
  "--native-sim",
  path.resolve(options.nativeSimPath),
  "--native-jobs",
  String(options.nativeJobs),
]);

console.log(JSON.stringify({
  outputPath: path.resolve(options.outputPath),
  models: "all",
  precisions: "all",
  includeDraftKvCache: false,
  traces: [...SMALL_TRACES, ...FULL_TRACES],
}, null, 2));

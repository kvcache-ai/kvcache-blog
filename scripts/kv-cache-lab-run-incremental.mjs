#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

import {
  allModelSettings,
  filterPrecisionOptions,
  kvArchitectureGroups,
  loadModelsData,
  modelSweepKey,
  selectModels,
} from "./lib/kv-cache-lab-traces.mjs";

const SMALL_TRACES = ["mooncake_fast25", "ragpulse", "lmcache_agentic_sample"];
const FULL_TRACES = [
  "bailian_qwen_trace_a",
  "semianalysis_weka_no_subagents",
  "semianalysis_weka_with_subagents_256k",
  "kv_cache_tester_claude_code",
];

function parseArgs(argv) {
  const args = {
    includeFamilies: [],
    excludeFamilies: [],
    modelIds: [],
    outputPath: "data/kv_cache_lab/precomputed.json",
    nativeJobs: 6,
    nativeSimPath: "/tmp/kv-cache-lab-native-sim-context-window",
    tempDir: "/tmp",
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--include-family") args.includeFamilies.push(argv[++index]);
    else if (arg === "--exclude-family") args.excludeFamilies.push(argv[++index]);
    else if (arg === "--model") args.modelIds.push(argv[++index]);
    else if (arg === "--output") args.outputPath = argv[++index];
    else if (arg === "--native-sim") args.nativeSimPath = argv[++index];
    else if (arg === "--native-jobs") args.nativeJobs = Math.max(1, Math.floor(Number(argv[++index])));
    else if (arg === "--temp-dir") args.tempDir = argv[++index];
  }
  if (!args.includeFamilies.length && !args.modelIds.length) args.includeFamilies = ["DeepSeek"];
  if (!args.modelIds.length) delete args.modelIds;
  if (!args.excludeFamilies.length) delete args.excludeFamilies;
  return args;
}

function run(command, args) {
  console.error(`$ ${[command, ...args].join(" ")}`);
  const result = spawnSync(command, args, { stdio: "inherit" });
  if (result.status !== 0) {
    throw new Error(`${command} exited with status ${result.status}`);
  }
}

function slugPart(value) {
  return String(value || "none").replace(/[^a-zA-Z0-9_.-]+/g, "-");
}

function settingLabel(group, setting) {
  const models = group.models.map((model) => model.id).join("+");
  return [
    models,
    setting.precision,
    setting.indexerPrecision || "no-indexer",
  ].join("|");
}

function buildSettingTasks(modelsData, options) {
  const selectedModels = selectModels(modelsData.models, options);
  const groups = kvArchitectureGroups(selectedModels);
  const precisionOptions = filterPrecisionOptions(modelsData.precision_options, options.precisionIds);
  const indexerPrecisionOptions = filterPrecisionOptions(modelsData.indexer_precision_options, options.indexerPrecisionIds);
  const tasks = [];
  for (const group of groups) {
    const canonical = group.models[0];
    const settings = allModelSettings([canonical], {
      precisionOptions,
      indexerPrecisionOptions,
      includeDraftKvCache: false,
    });
    for (const setting of settings) {
      tasks.push({ group, setting });
    }
  }
  return tasks;
}

function precomputeArgs(script, traceIds, task, outputPath, options) {
  const args = [script];
  for (const traceId of traceIds) args.push("--trace", traceId);
  for (const model of task.group.models) args.push("--model", model.id);
  args.push(
    "--dedupe-kv-architecture",
    "--no-draft",
    "--precision",
    task.setting.precision,
    "--output",
    outputPath,
  );
  if (task.setting.indexerPrecision) args.push("--indexer-precision", task.setting.indexerPrecision);
  if (script.endsWith("kv-cache-lab-full-precompute.mjs")) {
    args.push("--native-sim", options.nativeSimPath, "--native-jobs", String(options.nativeJobs));
  }
  return args;
}

const options = parseArgs(process.argv.slice(2));
const modelsData = loadModelsData();
const tasks = buildSettingTasks(modelsData, options);
console.error(`[incremental] ${tasks.length} setting tasks`);

for (let index = 0; index < tasks.length; index += 1) {
  const task = tasks[index];
  const label = settingLabel(task.group, task.setting);
  const slug = slugPart(`${index + 1}-${label}`);
  const smallOutput = path.join(options.tempDir, `kv-cache-incremental-${slug}-small.json`);
  const fullOutput = path.join(options.tempDir, `kv-cache-incremental-${slug}-full.json`);
  const mergedOutput = path.join(options.tempDir, `kv-cache-incremental-${slug}-merged.json`);
  const started = Date.now();
  console.error(`[incremental] start ${index + 1}/${tasks.length}: ${label} at ${new Date().toISOString()}`);

  run("node", precomputeArgs("scripts/kv-cache-lab-precompute-curves.mjs", SMALL_TRACES, task, smallOutput, options));
  run("node", precomputeArgs("scripts/kv-cache-lab-full-precompute.mjs", FULL_TRACES, task, fullOutput, options));
  run("node", [
    "scripts/kv-cache-lab-merge-precomputed.mjs",
    "--base",
    options.outputPath,
    "--input",
    smallOutput,
    "--input",
    fullOutput,
    "--output",
    mergedOutput,
  ]);
  fs.copyFileSync(mergedOutput, options.outputPath);

  const sweepKeys = task.group.models.map((model) => modelSweepKey({
    modelId: model.id,
    precision: task.setting.precision,
    indexerPrecision: task.setting.indexerPrecision,
    includeDraftKvCache: false,
  }));
  const elapsedSeconds = ((Date.now() - started) / 1000).toFixed(1);
  console.error(`[incremental] recorded ${label} (${sweepKeys.join(", ")}) in ${elapsedSeconds}s`);
}

console.log(JSON.stringify({ outputPath: path.resolve(options.outputPath), tasks: tasks.length }, null, 2));

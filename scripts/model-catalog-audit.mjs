#!/usr/bin/env node

import { writeFile } from "node:fs/promises";

const API_ROOT = "https://huggingface.co/api/models";
const USER_AGENT = "kvcache-blog-model-catalog-audit/1.0";
const PAGE_SIZE = 100;
const SHAPE_KEYS = [
  "model_type",
  "architectures",
  "hidden_size",
  "num_hidden_layers",
  "num_attention_heads",
  "num_key_value_heads",
  "head_dim",
  "num_global_key_value_heads",
  "global_head_dim",
  "global_v_head_dim",
  "num_kv_shared_layers",
  "attention_k_eq_v",
  "sliding_window",
  "sliding_window_size",
  "layer_types",
  "hybrid_layer_pattern",
  "full_attention_interval",
  "kv_lora_rank",
  "qk_rope_head_dim",
  "linear_conv_kernel_dim",
  "linear_key_head_dim",
  "linear_num_key_heads",
  "linear_value_head_dim",
  "linear_num_value_heads",
  "sconv_kernel_size",
  "conv_kernel_size",
  "max_position_embeddings",
  "max_sequence_length",
];

function usage() {
  return `Usage:
  node scripts/model-catalog-audit.mjs --scope AUTHOR[:FAMILY_QUERY] [options]

Examples:
  node scripts/model-catalog-audit.mjs --scope CohereLabs --scope dots-studio \\
    --scope google:gemma --scope meta-llama:llama \\
    --created-from 2026-01-01 --created-through 2026-12-31 --output ./model-catalog-audit.json

Options:
  --scope AUTHOR[:FAMILY_QUERY]  Repeat for each official Hub owner. FAMILY_QUERY is
                                 sent as Hugging Face search and matched against repo IDs/tags.
  --created-from YYYY-MM-DD      Inclusive Hub repo creation date lower bound.
  --created-through YYYY-MM-DD   Inclusive Hub repo creation date upper bound.
  --metadata-only                Skip fetching each repo's pinned config.json.
  --component-configs            Also fetch nested config.json files exposed by the Hub.
  --concurrency N                Config fetch concurrency (default: 6).
  --max-pages N                  Pagination safety limit per scope (default: 1000).
  --output PATH                  Write JSON to PATH instead of stdout.
  --help                         Show this help.

The Hub createdAt value is repository creation time, not model publication time. Repos
without createdAt remain with an unknown creation-date status. Each record keeps
official_release_date/source unknown until a human verifies an official announcement or
model card. Set HF_TOKEN in the environment to inspect gated configs
for repositories your account can access; the token is never written to the output.`;
}

function parseArgs(argv) {
  const options = {
    scopes: [],
    createdFrom: null,
    createdThrough: null,
    withConfig: true,
    withComponentConfigs: false,
    concurrency: 6,
    maxPages: 1000,
    output: null,
    help: false,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--help" || arg === "-h") options.help = true;
    else if (arg === "--scope") options.scopes.push(parseScope(takeValue(argv, i++, arg)));
    else if (arg === "--created-from") options.createdFrom = takeValue(argv, i++, arg);
    else if (arg === "--created-through") options.createdThrough = takeValue(argv, i++, arg);
    else if (arg === "--metadata-only") options.withConfig = false;
    else if (arg === "--component-configs") options.withComponentConfigs = true;
    else if (arg === "--concurrency") options.concurrency = positiveInteger(takeValue(argv, i++, arg), arg);
    else if (arg === "--max-pages") options.maxPages = positiveInteger(takeValue(argv, i++, arg), arg);
    else if (arg === "--output") options.output = takeValue(argv, i++, arg);
    else throw new Error(`Unknown option: ${arg}`);
  }

  if (!options.help && options.scopes.length === 0) {
    throw new Error("At least one --scope is required.");
  }
  if (options.createdFrom) validateDate(options.createdFrom, "--created-from");
  if (options.createdThrough) validateDate(options.createdThrough, "--created-through");
  if (options.createdFrom && options.createdThrough && options.createdFrom > options.createdThrough) {
    throw new Error("--created-from must be on or before --created-through.");
  }
  return options;
}

function takeValue(argv, index, option) {
  const value = argv[index + 1];
  if (!value || value.startsWith("--")) throw new Error(`${option} needs a value.`);
  return value;
}

function parseScope(value) {
  const separator = value.indexOf(":");
  const author = separator < 0 ? value : value.slice(0, separator);
  const familyQuery = separator < 0 ? null : value.slice(separator + 1);
  if (!author || familyQuery === "") {
    throw new Error(`Invalid scope "${value}". Use AUTHOR or AUTHOR:FAMILY_QUERY.`);
  }
  return { author, familyQuery };
}

function positiveInteger(value, option) {
  const number = Number(value);
  if (!Number.isInteger(number) || number < 1) throw new Error(`${option} must be a positive integer.`);
  return number;
}

function validateDate(value, option) {
  const parsed = new Date(`${value}T00:00:00Z`);
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value) || Number.isNaN(parsed.valueOf())
    || parsed.toISOString().slice(0, 10) !== value) {
    throw new Error(`${option} must use YYYY-MM-DD.`);
  }
}

function makeListUrl(scope) {
  const url = new URL(API_ROOT);
  url.searchParams.set("author", scope.author);
  url.searchParams.set("limit", String(PAGE_SIZE));
  url.searchParams.set("full", "true");
  url.searchParams.set("sort", "createdAt");
  if (scope.familyQuery) url.searchParams.set("search", scope.familyQuery);
  return url;
}

function nextPageUrl(linkHeader) {
  if (!linkHeader) return null;
  const next = linkHeader
    .split(/,(?=\s*<)/)
    .find((part) => /rel\s*=\s*"?next"?/.test(part));
  const match = next?.match(/<([^>]+)>/);
  return match ? new URL(match[1]) : null;
}

function withinCreationWindow(createdAt, options) {
  if (!createdAt) return true;
  const date = createdAt.slice(0, 10);
  return (!options.createdFrom || date >= options.createdFrom)
    && (!options.createdThrough || date <= options.createdThrough);
}

function candidateConfigUrl(repo) {
  if (!repo.sha) return null;
  return `https://huggingface.co/${repo.id}/raw/${repo.sha}/config.json`;
}

function componentConfigPaths(repo) {
  return (repo.siblings ?? [])
    .map((file) => file.rfilename)
    .filter((path) => path && path !== "config.json" && path.endsWith("/config.json"))
    .sort();
}

function configSummary(config) {
  if (!config || typeof config !== "object") return null;
  const shapeKeys = [
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "kv_lora_rank",
    "linear_num_key_heads",
  ];
  const candidates = [];
  function visit(node, path) {
    if (!node || typeof node !== "object" || Array.isArray(node)) return;
    const fields = {};
    for (const key of SHAPE_KEYS) {
      if (node[key] !== undefined) fields[key] = node[key];
    }
    if (node.model_type !== undefined) fields.model_type = node.model_type;
    if (node.architectures !== undefined) fields.architectures = node.architectures;
    const observableShapeFields = shapeKeys.filter((key) => fields[key] !== undefined).length;
    if (observableShapeFields > 0 || path === "$") {
      candidates.push({ path, fields, observableShapeFields });
    }
    for (const [key, value] of Object.entries(node)) {
      if (value && typeof value === "object" && !Array.isArray(value)) {
        visit(value, `${path}.${key}`);
      }
    }
  }
  visit(config, "$");
  const priority = (path) => {
    let score = 0;
    if (/(^|\.)(text_config)(\.|$)/.test(path)) score += 100;
    if (/(^|\.)(language_config)(\.|$)/.test(path)) score += 90;
    if (/(^|\.)(decoder_config|llm_config)(\.|$)/.test(path)) score += 70;
    if (/vision|visual|image|audio|speech/i.test(path)) score -= 40;
    return score;
  };
  candidates.sort((a, b) => priority(b.path) - priority(a.path)
    || b.observableShapeFields - a.observableShapeFields
    || b.path.length - a.path.length);
  const selected = candidates[0] ?? { path: "$", fields: {}, observableShapeFields: 0 };
  const nestedConfigs = candidates
    .filter((candidate) => candidate.path !== "$" && candidate.observableShapeFields > 0)
    .map(({ path, fields, observableShapeFields }) => ({ path, fields, observableShapeFields }));
  return {
    root_model_type: config.model_type ?? null,
    root_architectures: config.architectures ?? null,
    config_path: selected.path,
    shape_status: selected.observableShapeFields >= 3 ? "shape_fields_present" : "shape_incomplete_or_not_a_text_decoder",
    observable_shape_field_count: selected.observableShapeFields,
    fields: selected.fields,
    nested_config_candidates: nestedConfigs,
  };
}

async function fetchJson(url, token) {
  const headers = { "User-Agent": USER_AGENT, Accept: "application/json" };
  if (token) headers.Authorization = `Bearer ${token}`;
  const response = await fetch(url, { headers });
  const body = await response.text();
  if (!response.ok) {
    return { ok: false, status: response.status, error: body.slice(0, 500) };
  }
  try {
    return { ok: true, status: response.status, value: JSON.parse(body) };
  } catch {
    return { ok: false, status: response.status, error: "Response was not valid JSON." };
  }
}

async function collectScope(scope, options) {
  const rows = [];
  let url = makeListUrl(scope);
  let pages = 0;
  while (url) {
    if (pages >= options.maxPages) throw new Error(`Page limit reached for ${scope.author}.`);
    const headers = { "User-Agent": USER_AGENT, Accept: "application/json" };
    if (process.env.HF_TOKEN) headers.Authorization = `Bearer ${process.env.HF_TOKEN}`;
    const response = await fetch(url, { headers });
    if (!response.ok) throw new Error(`Hub API ${response.status} for ${url}`);
    const page = await response.json();
    if (!Array.isArray(page)) throw new Error(`Hub API returned a non-list page for ${url}`);
    pages += 1;
    for (const repo of page) {
      const familyCorpus = [repo.id, ...(repo.tags ?? [])].join(" ").toLowerCase();
      if (scope.familyQuery && !familyCorpus.includes(scope.familyQuery.toLowerCase())) continue;
      if (!withinCreationWindow(repo.createdAt, options)) continue;
      rows.push({ repo, scope, page_count: pages });
    }
    url = nextPageUrl(response.headers.get("link"));
  }
  return { rows, pages };
}

async function mapLimit(items, concurrency, mapper) {
  const output = new Array(items.length);
  let nextIndex = 0;
  async function worker() {
    while (nextIndex < items.length) {
      const index = nextIndex;
      nextIndex += 1;
      output[index] = await mapper(items[index]);
    }
  }
  await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, worker));
  return output;
}

async function buildRecord(entry, options, token) {
  const { repo, scope } = entry;
  const configUrl = candidateConfigUrl(repo);
  const record = {
    official_model_id: repo.id,
    author: repo.author ?? scope.author,
    family_query: scope.familyQuery,
    repo_created_at: repo.createdAt ?? null,
    repo_created_at_status: repo.createdAt ? "known_hub_repository_creation_time" : "unknown",
    repo_created_at_is_model_release_date: false,
    official_release_date: null,
    official_release_date_status: "unknown_until_official_evidence_is_added",
    official_release_source_url: null,
    gated: repo.gated ?? null,
    access_status: repo.gated ? "gated_or_restricted" : "public_or_unknown",
    task: repo.pipeline_tag ?? null,
    family_related_tags: (repo.tags ?? []).filter((tag) => /^(cohere|dots|gemma|inkling|llama|mimo|qwen|minimax)/i.test(tag)),
    repo_sha: repo.sha ?? null,
    model_card_url: `https://huggingface.co/${repo.id}`,
    config_url: configUrl,
    config_status: options.withConfig ? "not_fetched" : "skipped_metadata_only",
    config_summary: null,
    component_config_paths: componentConfigPaths(repo),
    component_configs_status: options.withComponentConfigs ? "not_fetched" : "not_requested",
    component_configs: [],
  };

  if (!options.withConfig && !options.withComponentConfigs) return record;
  if (options.withConfig && configUrl) {
    const fetched = await fetchJson(configUrl, token);
    if (!fetched.ok) {
      record.config_status = fetched.status === 401 || fetched.status === 403
        ? "gated_or_auth_required"
        : fetched.status === 404 ? "config_not_found" : `http_error_${fetched.status}`;
      record.config_error = fetched.error;
    } else {
      record.config_status = "fetched";
      record.config_summary = configSummary(fetched.value);
      record.raw_config = fetched.value;
      if (repo.gated && record.config_summary?.shape_status !== "shape_fields_present") {
        record.config_status = "gated_metadata_or_partial_config";
      }
    }
  }
  if (options.withComponentConfigs) {
    record.component_configs_status = "fetched";
    record.component_configs = await Promise.all(record.component_config_paths.map(async (path) => {
      const url = `https://huggingface.co/${repo.id}/raw/${repo.sha}/${path}`;
      const component = await fetchJson(url, token);
      if (!component.ok) {
        return {
          path,
          source_url: url,
          status: component.status === 401 || component.status === 403
            ? "gated_or_auth_required"
            : component.status === 404 ? "config_not_found" : `http_error_${component.status}`,
        };
      }
      return {
        path,
        source_url: url,
        status: "fetched",
        config_summary: configSummary(component.value),
        raw_config: component.value,
      };
    }));
  }
  return record;
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    console.log(usage());
    return;
  }

  const scopeResults = [];
  for (const scope of options.scopes) {
    const result = await collectScope(scope, options);
    scopeResults.push({ ...result, scope });
  }
  const repoEntries = new Map();
  for (const result of scopeResults) {
    for (const entry of result.rows) {
      const existing = repoEntries.get(entry.repo.id);
      if (existing) {
        existing.scope_labels.push(entry.scope.familyQuery
          ? `${entry.scope.author}:${entry.scope.familyQuery}`
          : entry.scope.author);
      } else {
        repoEntries.set(entry.repo.id, { ...entry, scope_labels: [
          entry.scope.familyQuery ? `${entry.scope.author}:${entry.scope.familyQuery}` : entry.scope.author,
        ] });
      }
    }
  }

  const token = process.env.HF_TOKEN ?? null;
  const records = await mapLimit([...repoEntries.values()], options.concurrency, async (entry) => {
    const record = await buildRecord(entry, options, token);
    record.matched_scopes = entry.scope_labels;
    return record;
  });
  records.sort((a, b) => a.author.localeCompare(b.author)
    || (a.repo_created_at ?? "").localeCompare(b.repo_created_at ?? "")
    || a.official_model_id.localeCompare(b.official_model_id));

  const output = {
    generated_at: new Date().toISOString(),
    source_api: API_ROOT,
    pagination: "Followed Hugging Face Link rel=next until the final page for each scope.",
    creation_date_semantics: "createdAt is the Hub repository creation time, not the model release date.",
    release_date_policy: "Release dates stay unknown unless a human adds an official announcement or model-card URL.",
    query: {
      scopes: options.scopes,
      created_from: options.createdFrom,
      created_through: options.createdThrough,
      config_fetched: options.withConfig,
      component_configs_fetched: options.withComponentConfigs,
    },
    pages_fetched: scopeResults.map(({ scope, pages }) => ({ author: scope.author, family_query: scope.familyQuery, pages })),
    repository_count: records.length,
    repositories: records,
  };
  const json = `${JSON.stringify(output, null, 2)}\n`;
  if (options.output) await writeFile(options.output, json, "utf8");
  else process.stdout.write(json);
}

main().catch((error) => {
  console.error(`model-catalog-audit: ${error.message}`);
  process.exitCode = 1;
});

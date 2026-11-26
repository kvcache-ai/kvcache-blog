#!/usr/bin/env node

import { readFile, rename, writeFile, chmod, stat, unlink } from 'node:fs/promises';
import { basename, dirname, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const USAGE = `Usage:
  node scripts/filter-calculator-scope.mjs \\
    --head-yaml /path/to/HEAD-models.yaml \\
    --yaml data/kv_cache_calculator/models.yaml \\
    --yaml packages/kvcache-simulator/src/kvcache_sim/resources/models.yaml \\
    --catalog data/kv_cache_calculator/model_catalog.json \\
    --decisions /path/to/decisions.json [--apply]

Without --apply, the command only prints the planned changes.

Decision JSON:
{
  "models": [
    { "id": "new-preset-id", "action": "keep", "reason": "..." },
    { "id": "another-preset-id", "action": "skip", "reason": "..." }
  ],
  "aliases": [
    { "official_model_id": "Org/Model", "action": "keep", "reason": "..." },
    { "official_model_id": "Org/Other", "action": "skip", "reason": "..." }
  ]
}

Every non-HEAD YAML model and every catalog row must have exactly one decision.
`;

function fail(message) {
  throw new Error(message);
}

function splitLines(text) {
  const lines = [];
  let start = 0;
  while (start < text.length) {
    const newline = text.indexOf('\n', start);
    const end = newline === -1 ? text.length : newline + 1;
    const contentEnd = newline === -1 ? end : newline;
    lines.push({
      start,
      end,
      content: text.slice(start, contentEnd).replace(/\r$/, ''),
    });
    start = end;
  }
  return lines;
}

function parseYamlId(line, lineNumber) {
  const match = line.match(/^  - id:\s*(?:"((?:[^"\\]|\\.)*)"|'((?:[^']|'')*)'|([^#\r\n]+?))\s*(?:#.*)?\r?$/);
  if (!match) {
    fail(`models.yaml line ${lineNumber}: expected a model entry beginning with "  - id:"`);
  }
  let id = match[1] ?? match[2] ?? match[3]?.trim();
  if (match[1] !== undefined) {
    try {
      id = JSON.parse(`"${match[1]}"`);
    } catch {
      fail(`models.yaml line ${lineNumber}: invalid quoted model id`);
    }
  } else if (match[2] !== undefined) {
    id = match[2].replaceAll("''", "'");
  }
  if (!id) fail(`models.yaml line ${lineNumber}: empty model id`);
  return id;
}

function parseModelBlocks(text, sourceLabel) {
  const lines = splitLines(text);
  const headerIndex = lines.findIndex(({ content }) => /^models:\s*(?:#.*)?$/.test(content));
  if (headerIndex === -1) fail(`${sourceLabel}: top-level models: section not found`);

  const header = lines[headerIndex];
  let sectionEnd = text.length;
  for (let i = headerIndex + 1; i < lines.length; i += 1) {
    if (/^[A-Za-z_][A-Za-z0-9_.-]*\s*:/.test(lines[i].content)) {
      sectionEnd = lines[i].start;
      break;
    }
  }

  const bodyStart = header.end;
  const body = text.slice(bodyStart, sectionEnd);
  const bodyLines = splitLines(body);
  const starts = [];
  for (const line of bodyLines) {
    if (/^  -\s/.test(line.content)) {
      const id = parseYamlId(line.content, lines.findIndex((candidate) => candidate.start === bodyStart + line.start) + 1);
      starts.push({ start: line.start, id });
    }
  }

  let bodyContentEnd = body.length;
  let bodyTailStart = body.length;
  for (let i = bodyLines.length - 1; i >= 0; i -= 1) {
    const line = bodyLines[i];
    if (line.content.trim() === '' || line.content.startsWith('#')) {
      bodyTailStart = line.start;
    } else {
      break;
    }
  }
  bodyContentEnd = bodyTailStart;

  const ids = new Set();
  const blocks = [];
  for (let i = 0; i < starts.length; i += 1) {
    const entry = starts[i];
    if (ids.has(entry.id)) fail(`${sourceLabel}: duplicate model id "${entry.id}"`);
    ids.add(entry.id);
    const end = i + 1 < starts.length ? starts[i + 1].start : bodyContentEnd;
    if (end < entry.start) fail(`${sourceLabel}: malformed models section near "${entry.id}"`);
    blocks.push({ id: entry.id, raw: body.slice(entry.start, end) });
  }

  if (starts.length === 0 && body.trim() !== '') {
    const nonCommentLine = bodyLines.find((line) => line.content.trim() !== '' && !line.content.startsWith('#'));
    if (nonCommentLine && /^\s*-\s/.test(nonCommentLine.content)) {
      fail(`${sourceLabel}: models section contains a list entry without an id`);
    }
  }

  const prefixEnd = starts.length ? starts[0].start : body.length;
  return {
    blocks,
    ids,
    section: {
      start: header.start,
      end: sectionEnd,
      prefix: text.slice(header.start, bodyStart + prefixEnd),
      tail: text.slice(bodyStart + bodyContentEnd, sectionEnd),
      bodyStart,
    },
  };
}

function scanJsonArrayRows(text, sourceLabel) {
  const open = text.search(/\S/);
  if (open === -1 || text[open] !== '[') fail(`${sourceLabel}: expected a JSON array`);

  const rows = [];
  let cursor = open + 1;
  let close = -1;
  while (cursor < text.length) {
    while (/\s/.test(text[cursor] ?? '') && cursor < text.length) cursor += 1;
    if (text[cursor] === ']') {
      close = cursor;
      break;
    }
    if (text[cursor] !== '{') fail(`${sourceLabel}: each catalog row must be a JSON object`);

    const rowStart = cursor;
    const stack = [];
    let inString = false;
    let escaped = false;
    for (; cursor < text.length; cursor += 1) {
      const ch = text[cursor];
      if (inString) {
        if (escaped) escaped = false;
        else if (ch === '\\') escaped = true;
        else if (ch === '"') inString = false;
        continue;
      }
      if (ch === '"') inString = true;
      else if (ch === '{' || ch === '[') stack.push(ch);
      else if (ch === '}' || ch === ']') {
        const expected = ch === '}' ? '{' : '[';
        if (stack.pop() !== expected) fail(`${sourceLabel}: mismatched JSON bracket`);
        if (stack.length === 0) {
          cursor += 1;
          break;
        }
      }
    }
    if (stack.length !== 0 || inString) fail(`${sourceLabel}: unterminated JSON object`);
    rows.push({ start: rowStart, end: cursor });

    while (/\s/.test(text[cursor] ?? '') && cursor < text.length) cursor += 1;
    if (text[cursor] === ',') {
      cursor += 1;
      continue;
    }
    if (text[cursor] === ']') {
      close = cursor;
      break;
    }
    fail(`${sourceLabel}: expected a comma or closing bracket after catalog row`);
  }
  if (close === -1) fail(`${sourceLabel}: unterminated JSON array`);
  if (text.slice(close + 1).trim() !== '') fail(`${sourceLabel}: unexpected content after JSON array`);

  let separator = ',\n  ';
  if (rows.length > 1) separator = text.slice(rows[0].end, rows[1].start);
  return {
    rows,
    prefix: rows.length ? text.slice(0, rows[0].start) : text.slice(0, close),
    suffix: rows.length ? text.slice(rows.at(-1).end) : text.slice(close),
    separator,
  };
}

function parseCatalog(text, sourceLabel) {
  let catalog;
  try {
    catalog = JSON.parse(text);
  } catch (error) {
    fail(`${sourceLabel}: invalid JSON (${error.message})`);
  }
  if (!Array.isArray(catalog)) fail(`${sourceLabel}: expected a top-level JSON array`);
  const spans = scanJsonArrayRows(text, sourceLabel);
  if (catalog.length !== spans.rows.length) fail(`${sourceLabel}: could not map JSON rows to catalog entries`);

  const seen = new Set();
  const entries = catalog.map((entry, index) => {
    if (!entry || typeof entry !== 'object' || Array.isArray(entry)) {
      fail(`${sourceLabel}: catalog row ${index + 1} must be an object`);
    }
    if (typeof entry.official_model_id !== 'string' || !entry.official_model_id.trim()) {
      fail(`${sourceLabel}: catalog row ${index + 1} is missing official_model_id`);
    }
    if (typeof entry.preset_id !== 'string' || !entry.preset_id.trim()) {
      fail(`${sourceLabel}: catalog row ${index + 1} is missing preset_id`);
    }
    if (seen.has(entry.official_model_id)) {
      fail(`${sourceLabel}: duplicate official_model_id "${entry.official_model_id}"`);
    }
    seen.add(entry.official_model_id);
    return { ...entry, raw: text.slice(spans.rows[index].start, spans.rows[index].end) };
  });
  return { entries, spans };
}

function parseDecisions(decisions, modelIds, aliasIds) {
  if (!decisions || typeof decisions !== 'object' || Array.isArray(decisions)) {
    fail('decisions JSON must be an object');
  }
  if (!Array.isArray(decisions.models) || !Array.isArray(decisions.aliases)) {
    fail('decisions JSON must contain models and aliases arrays');
  }

  const readRows = (rows, idField, label, expectedIds) => {
    const result = new Map();
    for (const row of rows) {
      if (!row || typeof row !== 'object' || Array.isArray(row)) fail(`${label}: each decision must be an object`);
      const id = row[idField];
      if (typeof id !== 'string' || !id.trim()) fail(`${label}: missing ${idField}`);
      if (result.has(id)) fail(`${label}: duplicate decision for "${id}"`);
      if (!expectedIds.has(id)) fail(`${label}: unexpected decision for "${id}"`);
      if (row.action !== 'keep' && row.action !== 'skip') {
        fail(`${label} "${id}": action must be "keep" or "skip"`);
      }
      if (typeof row.reason !== 'string' || !row.reason.trim() || /[\r\n]/.test(row.reason)) {
        fail(`${label} "${id}": a single-line reason is required`);
      }
      result.set(id, { action: row.action, reason: row.reason.trim() });
    }
    for (const id of expectedIds) {
      if (!result.has(id)) fail(`${label}: missing decision for "${id}"`);
    }
    return result;
  };

  const modelDecisions = readRows(decisions.models, 'id', 'model decision', modelIds);
  const aliasDecisions = readRows(decisions.aliases, 'official_model_id', 'alias decision', aliasIds);
  return { modelDecisions, aliasDecisions };
}

function renderCatalog(catalog, retainedAliasIds) {
  const keptRaw = catalog.entries
    .filter((entry) => retainedAliasIds.has(entry.official_model_id))
    .map((entry) => entry.raw);
  return catalog.spans.prefix + keptRaw.join(catalog.spans.separator) + catalog.spans.suffix;
}

function renderYaml(currentText, headModelBlocks, selectedAddedIds, sourceLabel) {
  const current = parseModelBlocks(currentText, sourceLabel);
  const blocks = [
    ...headModelBlocks,
    ...current.blocks.filter((block) => selectedAddedIds.has(block.id)),
  ];
  const body = blocks.map((block) => block.raw).join('');
  return currentText.slice(0, current.section.start)
    + current.section.prefix
    + body
    + current.section.tail
    + currentText.slice(current.section.end);
}

export function filterCalculatorScope({ headYaml, currentYamls, catalogText, decisions }) {
  if (typeof headYaml !== 'string') fail('headYaml must be text');
  if (!Array.isArray(currentYamls) || currentYamls.length < 1) fail('at least one current YAML input is required');
  if (typeof catalogText !== 'string') fail('catalogText must be text');

  const head = parseModelBlocks(headYaml, 'HEAD models.yaml');
  const current = currentYamls.map((item, index) => {
    if (!item || typeof item.text !== 'string') fail(`current YAML ${index + 1} must contain text`);
    return {
      path: item.path ?? `models-${index + 1}.yaml`,
      text: item.text,
      parsed: parseModelBlocks(item.text, item.path ?? `current YAML ${index + 1}`),
    };
  });
  const headIds = head.ids;
  const headModelBlocks = head.blocks;

  const firstCurrentIds = current[0].parsed.ids;
  for (let i = 1; i < current.length; i += 1) {
    const ids = current[i].parsed.ids;
    if (ids.size !== firstCurrentIds.size || [...ids].some((id) => !firstCurrentIds.has(id))) {
      fail(`models.yaml inputs are out of sync: ${current[0].path} and ${current[i].path} have different model id sets`);
    }
  }

  const candidateModelIds = new Set([...firstCurrentIds].filter((id) => !headIds.has(id)));
  const catalog = parseCatalog(catalogText, 'model catalog');
  const candidateAliasIds = new Set(catalog.entries.map((entry) => entry.official_model_id));
  const { modelDecisions, aliasDecisions } = parseDecisions(decisions, candidateModelIds, candidateAliasIds);

  const retainedAliasIds = new Set(
    [...aliasDecisions].filter(([, decision]) => decision.action === 'keep').map(([id]) => id),
  );
  const directlyKeptModelIds = new Set(
    [...modelDecisions].filter(([, decision]) => decision.action === 'keep').map(([id]) => id),
  );
  const aliasPresetIds = new Set(
    catalog.entries
      .filter((entry) => retainedAliasIds.has(entry.official_model_id))
      .map((entry) => entry.preset_id),
  );
  for (const presetId of aliasPresetIds) {
    if (headIds.has(presetId)) continue;
    if (!candidateModelIds.has(presetId)) {
      fail(`retained alias references unknown preset "${presetId}"`);
    }
    if (modelDecisions.get(presetId).action !== 'keep') {
      fail(`retained alias references skipped preset "${presetId}"`);
    }
  }

  const aliasRequiredAddedIds = new Set([...aliasPresetIds].filter((id) => !headIds.has(id)));
  const selectedAddedIds = new Set([...directlyKeptModelIds, ...aliasRequiredAddedIds]);
  const modelBlocksByYaml = current.map(({ parsed }) => new Map(parsed.blocks.map((block) => [block.id, block])));
  for (const id of selectedAddedIds) {
    const firstRaw = modelBlocksByYaml[0].get(id)?.raw;
    if (firstRaw === undefined) fail(`selected model "${id}" has no YAML block`);
    for (let i = 1; i < modelBlocksByYaml.length; i += 1) {
      if (modelBlocksByYaml[i].get(id)?.raw !== firstRaw) {
        fail(`retained model "${id}" differs between ${current[0].path} and ${current[i].path}`);
      }
    }
  }

  const headPresetIds = headIds;
  for (const entry of catalog.entries) {
    if (retainedAliasIds.has(entry.official_model_id)
      && !headPresetIds.has(entry.preset_id)
      && !selectedAddedIds.has(entry.preset_id)) {
      fail(`retained alias "${entry.official_model_id}" would have dangling preset_id "${entry.preset_id}"`);
    }
  }

  const yamlOutputs = current.map((item) => ({
    path: item.path,
    text: renderYaml(item.text, headModelBlocks, selectedAddedIds, item.path),
  }));
  const outputCatalog = renderCatalog(catalog, retainedAliasIds);

  return {
    yamlOutputs,
    catalogText: outputCatalog,
    stats: {
      headModelsPreserved: head.blocks.length,
      candidateModels: candidateModelIds.size,
      retainedModels: selectedAddedIds.size,
      removedModels: candidateModelIds.size - selectedAddedIds.size,
      catalogAliases: catalog.entries.length,
      retainedAliases: retainedAliasIds.size,
      removedAliases: catalog.entries.length - retainedAliasIds.size,
    },
    decisions: {
      models: [...modelDecisions].map(([id, decision]) => ({ id, ...decision })),
      aliases: [...aliasDecisions].map(([official_model_id, decision]) => ({ official_model_id, ...decision })),
    },
  };
}

function parseArgs(argv) {
  const args = { yaml: [], apply: false };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--apply') args.apply = true;
    else if (arg === '--help' || arg === '-h') args.help = true;
    else if (['--head-yaml', '--yaml', '--catalog', '--decisions'].includes(arg)) {
      const value = argv[i + 1];
      if (!value || value.startsWith('--')) fail(`${arg} requires a path`);
      i += 1;
      if (arg === '--yaml') args.yaml.push(value);
      else args[arg.slice(2).replaceAll('-', '')] = value;
    } else {
      fail(`unknown argument: ${arg}`);
    }
  }
  if (args.help) return args;
  if (!args.headyaml || !args.catalog || !args.decisions || args.yaml.length === 0) {
    fail('missing required inputs; pass --help for usage');
  }
  const outputs = [...args.yaml, args.catalog].map((path) => resolve(path));
  if (new Set(outputs).size !== outputs.length) fail('output paths must be unique');
  if (outputs.includes(resolve(args.headyaml))) fail('--head-yaml must not be one of the output paths');
  return args;
}

function printTable(title, rows, idField) {
  console.log(`\n${title}`);
  if (rows.length === 0) {
    console.log('  (none)');
    return;
  }
  const idWidth = Math.max(idField.length, ...rows.map((row) => row[idField].length));
  for (const row of rows) {
    console.log(`  ${row.action.padEnd(4)}  ${row[idField].padEnd(idWidth)}  ${row.reason}`);
  }
}

async function stageAndApply(outputs) {
  const staged = [];
  try {
    for (let index = 0; index < outputs.length; index += 1) {
      const output = outputs[index];
      const originalStat = await stat(output.path);
      const tempPath = resolve(dirname(output.path), `.${basename(output.path)}.scope-${process.pid}-${index}.tmp`);
      await writeFile(tempPath, output.text, { encoding: 'utf8', flag: 'wx' });
      await chmod(tempPath, originalStat.mode & 0o777);
      staged.push({ tempPath, output });
    }
    for (const { tempPath, output } of staged) await rename(tempPath, output.path);
  } catch (error) {
    await Promise.all(staged.map(({ tempPath }) => unlink(tempPath).catch(() => {})));
    throw error;
  }
}

async function main(argv) {
  const args = parseArgs(argv);
  if (args.help) {
    console.log(USAGE);
    return;
  }

  const [headYaml, ...rest] = await Promise.all([
    readFile(args.headyaml, 'utf8'),
    ...args.yaml.map((path) => readFile(path, 'utf8')),
    readFile(args.catalog, 'utf8'),
    readFile(args.decisions, 'utf8'),
  ]);
  const yamlTexts = rest.slice(0, args.yaml.length);
  const catalogText = rest[args.yaml.length];
  let decisions;
  try {
    decisions = JSON.parse(rest[args.yaml.length + 1]);
  } catch (error) {
    fail(`invalid decisions JSON (${error.message})`);
  }

  const plan = filterCalculatorScope({
    headYaml,
    currentYamls: args.yaml.map((path, index) => ({ path, text: yamlTexts[index] })),
    catalogText,
    decisions,
  });
  const outputs = [
    ...plan.yamlOutputs.map((item) => ({ ...item })),
    { path: args.catalog, text: plan.catalogText },
  ];
  const changed = outputs.filter((output, index) => {
    const original = index < yamlTexts.length ? yamlTexts[index] : catalogText;
    return output.text !== original;
  });

  console.log(`${args.apply ? 'Apply plan' : 'Dry-run plan'}:`);
  console.log(`  HEAD canonical models preserved: ${plan.stats.headModelsPreserved}`);
  console.log(`  new presets: ${plan.stats.retainedModels} kept, ${plan.stats.removedModels} removed`);
  console.log(`  catalog aliases: ${plan.stats.retainedAliases} kept, ${plan.stats.removedAliases} removed`);
  console.log(`  changed files: ${changed.length}/${outputs.length}`);
  for (const output of changed) console.log(`  ${args.apply ? 'write' : 'would update'} ${output.path}`);
  printTable('New model decisions (id / action / reason):', plan.decisions.models, 'id');
  printTable('Catalog alias decisions (official_model_id / action / reason):', plan.decisions.aliases, 'official_model_id');

  if (args.apply && changed.length > 0) {
    await stageAndApply(changed);
    console.log(`\nApplied ${changed.length} file update(s).`);
  } else if (!args.apply) {
    console.log('\nDry-run only; no files were changed. Pass --apply to write these changes.');
  } else {
    console.log('\nNo file changes were needed.');
  }
}

const invokedPath = process.argv[1] ? resolve(process.argv[1]) : '';
if (invokedPath && pathToFileURL(invokedPath).href === import.meta.url) {
  main(process.argv.slice(2)).catch((error) => {
    console.error(`Error: ${error.message}`);
    process.exitCode = 1;
  });
}

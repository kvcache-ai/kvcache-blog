import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

import { filterCalculatorScope } from '../scripts/filter-calculator-scope.mjs';

const scriptPath = resolve(fileURLToPath(new URL('../scripts/filter-calculator-scope.mjs', import.meta.url)));

const headYaml = `metadata:\n  retrieved_at: "upstream"\nmodels:\n  - id: base-a\n    label: "HEAD A"\n    fields:\n      layers: 2\n  - id: base-b\n    label: "HEAD B"\n    fields:\n      layers: 4\nprecision_options:\n  - id: bf16\n`;

function currentYaml(metadata, extraBlockOrder = '') {
  return `metadata:\n  retrieved_at: "${metadata}"\n  user_note: keep this metadata\nmodels:\n  - id: base-a\n    label: "modified A"\n    fields:\n      layers: 9\n  - id: base-b\n    label: "modified B"\n    fields:\n      layers: 8\n  - id: keep-direct\n    label: "Kept directly"\n    fields:\n      layers: 6\n  - id: alias-only\n    label: "Kept for alias"\n    fields:\n      layers: 7\n  - id: skip-this\n    label: "Must be removed"\n    fields:\n      layers: 10\n${extraBlockOrder}precision_options:\n  - id: fp16\ncustom_root: preserve me\n`;
}

const catalogText = `[
  {"official_model_id":"Org/alias-only","preset_id":"alias-only","cache_note":"preserve alias metadata"},
  {"official_model_id":"Org/skipped","preset_id":"skip-this","cache_note":"remove this row"},
  {"official_model_id":"Org/base","preset_id":"base-a","custom_metadata":{"keep":true}}
]\n`;

const decisions = {
  models: [
    { id: 'keep-direct', action: 'keep', reason: 'official post-cutoff model with supported geometry' },
    { id: 'alias-only', action: 'keep', reason: 'required by accepted catalog alias' },
    { id: 'skip-this', action: 'skip', reason: 'published before cutoff' },
  ],
  aliases: [
    { official_model_id: 'Org/alias-only', action: 'keep', reason: 'official post-cutoff checkpoint' },
    { official_model_id: 'Org/skipped', action: 'skip', reason: 'preset is outside the requested scope' },
    { official_model_id: 'Org/base', action: 'keep', reason: 'official post-cutoff checkpoint using an existing preset' },
  ],
};

test('keeps HEAD model blocks byte-for-byte, retains accepted presets and aliases, and preserves other metadata', () => {
  const firstYaml = currentYaml('current-A');
  const secondYaml = currentYaml('current-B');
  const result = filterCalculatorScope({
    headYaml,
    currentYamls: [
      { path: 'site/models.yaml', text: firstYaml },
      { path: 'package/models.yaml', text: secondYaml },
    ],
    catalogText,
    decisions,
  });

  assert.equal(result.stats.headModelsPreserved, 2);
  assert.equal(result.stats.candidateModels, 3);
  assert.equal(result.stats.retainedModels, 2);
  assert.equal(result.stats.removedModels, 1);
  assert.equal(result.stats.catalogAliases, 3);
  assert.equal(result.stats.retainedAliases, 2);
  assert.equal(result.stats.removedAliases, 1);

  for (const output of result.yamlOutputs) {
    assert.ok(output.text.includes('retrieved_at: "current-'));
    assert.ok(output.text.includes('user_note: keep this metadata'));
    assert.ok(output.text.includes('custom_root: preserve me'));
    assert.ok(output.text.includes('  - id: base-a\n    label: "HEAD A"\n    fields:\n      layers: 2\n'));
    assert.ok(output.text.includes('  - id: base-b\n    label: "HEAD B"\n    fields:\n      layers: 4\n'));
    assert.ok(!output.text.includes('modified A'));
    assert.ok(!output.text.includes('Must be removed'));
    assert.ok(output.text.includes('  - id: keep-direct\n'));
    assert.ok(output.text.includes('  - id: alias-only\n'));
  }

  const keptCatalog = JSON.parse(result.catalogText);
  assert.deepEqual(keptCatalog.map((entry) => entry.official_model_id), ['Org/alias-only', 'Org/base']);
  assert.deepEqual(keptCatalog[1].custom_metadata, { keep: true });
  assert.ok(result.catalogText.includes('preserve alias metadata'));
  assert.ok(!result.catalogText.includes('remove this row'));
});

test('rejects missing, duplicate, unknown and contradictory decisions', () => {
  const input = {
    headYaml,
    currentYamls: [{ path: 'models.yaml', text: currentYaml('one') }],
    catalogText,
  };

  assert.throws(
    () => filterCalculatorScope({ ...input, decisions: { ...decisions, models: decisions.models.slice(0, 2) } }),
    /missing decision for "skip-this"/,
  );
  assert.throws(
    () => filterCalculatorScope({ ...input, decisions: { ...decisions, models: [...decisions.models, decisions.models[0]] } }),
    /duplicate decision for "keep-direct"/,
  );
  assert.throws(
    () => filterCalculatorScope({ ...input, decisions: { ...decisions, models: [...decisions.models, { id: 'not-present', action: 'skip', reason: 'unknown' }] } }),
    /unexpected decision for "not-present"/,
  );
  const contradictory = structuredClone(decisions);
  contradictory.models.find((row) => row.id === 'alias-only').action = 'skip';
  assert.throws(() => filterCalculatorScope({ ...input, decisions: contradictory }), /retained alias references skipped preset "alias-only"/);
});

test('rejects YAML model id drift between the two synchronized files', () => {
  const first = currentYaml('one');
  const second = currentYaml('two').replace('id: alias-only', 'id: changed-id');
  assert.throws(() => filterCalculatorScope({
    headYaml,
    currentYamls: [{ path: 'one.yaml', text: first }, { path: 'two.yaml', text: second }],
    catalogText,
    decisions,
  }), /different model id sets/);
});

test('dry-run leaves files untouched and --apply writes only after explicit opt-in', async (t) => {
  const dir = await mkdtemp(join(tmpdir(), 'kvcache-scope-'));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const headPath = join(dir, 'head.yaml');
  const yamlPath = join(dir, 'models.yaml');
  const catalogPath = join(dir, 'catalog.json');
  const decisionsPath = join(dir, 'decisions.json');
  const oneModelHead = `models:\n  - id: base\n    label: "HEAD"\n`;
  const current = `metadata:\n  note: keep\nmodels:\n  - id: base\n    label: "changed"\n  - id: new\n    label: "new"\n`;
  const catalog = `[{"official_model_id":"Org/New","preset_id":"new"}, {"official_model_id":"Org/Old","preset_id":"base"}]\n`;
  const decisionSet = {
    models: [{ id: 'new', action: 'keep', reason: 'supported' }],
    aliases: [
      { official_model_id: 'Org/New', action: 'keep', reason: 'accepted official checkpoint' },
      { official_model_id: 'Org/Old', action: 'skip', reason: 'outside requested release window' },
    ],
  };
  await Promise.all([
    writeFile(headPath, oneModelHead),
    writeFile(yamlPath, current),
    writeFile(catalogPath, catalog),
    writeFile(decisionsPath, JSON.stringify(decisionSet)),
  ]);

  const baseArgs = ['--head-yaml', headPath, '--yaml', yamlPath, '--catalog', catalogPath, '--decisions', decisionsPath];
  const dryRun = spawnSync(process.execPath, [scriptPath, ...baseArgs], { encoding: 'utf8' });
  assert.equal(dryRun.status, 0, dryRun.stderr);
  assert.match(dryRun.stdout, /Dry-run plan/);
  assert.match(dryRun.stdout, /Dry-run only; no files were changed/);
  assert.equal(await readFile(yamlPath, 'utf8'), current);
  assert.equal(await readFile(catalogPath, 'utf8'), catalog);

  const apply = spawnSync(process.execPath, [scriptPath, ...baseArgs, '--apply'], { encoding: 'utf8' });
  assert.equal(apply.status, 0, apply.stderr);
  assert.match(apply.stdout, /Applied 2 file update\(s\)/);
  const writtenYaml = await readFile(yamlPath, 'utf8');
  assert.ok(writtenYaml.includes('label: "HEAD"'));
  assert.ok(writtenYaml.includes('id: new'));
  assert.ok(!writtenYaml.includes('label: "changed"'));
  assert.deepEqual(JSON.parse(await readFile(catalogPath, 'utf8')).map((row) => row.official_model_id), ['Org/New']);
});

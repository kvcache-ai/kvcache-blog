#!/usr/bin/env bash
# Refresh the model catalog bundled in packages/kvcache-simulator from data/kv_cache_calculator/models.yaml.
# The package tests check that the two files match; run this after changing the catalog.
set -euo pipefail
cd "$(dirname "$0")/.."
cp data/kv_cache_calculator/models.yaml packages/kvcache-simulator/src/kvcache_sim/resources/models.yaml
echo "synced packages/kvcache-simulator/src/kvcache_sim/resources/models.yaml"

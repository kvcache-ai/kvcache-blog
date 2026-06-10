#!/usr/bin/env bash
# Build the KV Cache Lab WASM trace processor and copy the artifact into assets/
# so Hugo fingerprints and serves it. The committed wasm means no toolchain is
# needed at deploy time; re-run this only when wasm/kvcache-sim/ changes.
#
#   rustup target add wasm32-unknown-unknown   # one-time
#   scripts/build-wasm.sh
set -euo pipefail
cd "$(dirname "$0")/.."

cargo build --release --target wasm32-unknown-unknown --manifest-path wasm/kvcache-sim/Cargo.toml

SRC=wasm/kvcache-sim/target/wasm32-unknown-unknown/release/kvcache_sim.wasm
DST=assets/wasm/kvcache-sim.wasm
mkdir -p "$(dirname "$DST")"
cp "$SRC" "$DST"
echo "built $DST ($(du -h "$DST" | cut -f1))"

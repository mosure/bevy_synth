#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../" && pwd)"
TEST_DIR="${ROOT_DIR}/crates/burn_synth/tests/web_playwright"
OUT_DIR="${ROOT_DIR}/www/out"

unset RUSTFLAGS
unset CARGO_ENCODED_RUSTFLAGS
unset CARGO_BUILD_RUSTFLAGS
unset CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS
unset RUSTDOCFLAGS

mkdir -p "${OUT_DIR}"

echo "[web-e2e] build burn_synth wasm (wasm-api + wasm-api-wgpu)"
cargo build \
  -p burn_synth \
  --target wasm32-unknown-unknown \
  --profile wasm-release \
  --features wasm-api,wasm-api-wgpu

echo "[web-e2e] wasm-bindgen"
wasm-bindgen \
  "${ROOT_DIR}/target/wasm32-unknown-unknown/wasm-release/burn_synth.wasm" \
  --out-dir "${OUT_DIR}" \
  --target web \
  --no-typescript

cd "${TEST_DIR}"
echo "[web-e2e] playwright deps"
npm install --no-audit --no-fund
npx playwright install chromium

echo "[web-e2e] run synth_api integration test"
npx playwright test --config playwright.config.mjs --workers=1 --reporter=list

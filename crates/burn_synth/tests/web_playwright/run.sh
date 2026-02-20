#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../" && pwd)"
TEST_DIR="${ROOT_DIR}/crates/burn_synth/tests/web_playwright"
OUT_DIR="${ROOT_DIR}/www/out"
TMP_DIR="${TEST_DIR}/tmp"
NATIVE_REF_GLB="${TMP_DIR}/native_reference.glb"
LOCK_DIR="${ROOT_DIR}/target/.webgpu-test.lockdir"
LOCK_TIMEOUT_SEC="${BURN_SYNTH_WEBGPU_LOCK_TIMEOUT_SEC:-7200}"
LOCK_WAIT_SEC=2

unset RUSTFLAGS
unset CARGO_ENCODED_RUSTFLAGS
unset CARGO_BUILD_RUSTFLAGS
unset CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS
unset RUSTDOCFLAGS

mkdir -p "${OUT_DIR}"
mkdir -p "${TMP_DIR}"
mkdir -p "${ROOT_DIR}/target"

LOCK_FALLBACK_HELD=0
cleanup_webgpu_lock() {
  if [[ "${LOCK_FALLBACK_HELD}" == "1" ]]; then
    rmdir "${LOCK_DIR}" 2>/dev/null || true
  fi
}
trap cleanup_webgpu_lock EXIT

echo "[web-e2e] acquire exclusive WebGPU test lock"
lock_deadline=$((SECONDS + LOCK_TIMEOUT_SEC))
until mkdir "${LOCK_DIR}" 2>/dev/null; do
  if (( SECONDS >= lock_deadline )); then
    echo "[web-e2e] timed out waiting for WebGPU lock after ${LOCK_TIMEOUT_SEC}s: ${LOCK_DIR}" >&2
    exit 1
  fi
  sleep "${LOCK_WAIT_SEC}"
done
LOCK_FALLBACK_HELD=1

echo "[web-e2e] build burn_synth wasm (wasm-api + wasm-api-wgpu)"
MODEL_BASE_URL=assets/models \
cargo build \
  -p burn_synth \
  --lib \
  --no-default-features \
  --target wasm32-unknown-unknown \
  --profile wasm-release \
  --features wasm-api,wasm-api-wgpu

echo "[web-e2e] wasm-bindgen burn_synth"
wasm-bindgen \
  "${ROOT_DIR}/target/wasm32-unknown-unknown/wasm-release/burn_synth.wasm" \
  --out-dir "${OUT_DIR}" \
  --target web \
  --no-typescript

echo "[web-e2e] build bevy_synth wasm"
MODEL_BASE_URL=assets/models \
cargo build \
  -p bevy_synth \
  --target wasm32-unknown-unknown \
  --profile wasm-release \
  --features wgpu,triposg

echo "[web-e2e] wasm-bindgen bevy_synth"
wasm-bindgen \
  "${ROOT_DIR}/target/wasm32-unknown-unknown/wasm-release/bevy_synth.wasm" \
  --out-dir "${OUT_DIR}" \
  --target web \
  --no-typescript

cd "${TEST_DIR}"
echo "[web-e2e] playwright deps"
npm install --no-audit --no-fund
npx playwright install chromium

if [[ "${BURN_SYNTH_WEB_SKIP_ARTIFACT_ENSURE:-0}" != "1" ]]; then
  echo "[web-e2e] ensure f32+f16 parts artifacts for web models"
  cargo run \
    -p burn_synth_import \
    --bin ensure_web_burnpack_artifacts \
    -- \
    --root "${ROOT_DIR}/www/assets/models/MIDI-3D" \
    --root "${ROOT_DIR}/www/assets/models/RMBG-1.4" \
    --part-size-mib 64
fi

if [[ "${BURN_SYNTH_WEB_SKIP_NATIVE_REF:-0}" != "1" ]]; then
  echo "[web-e2e] build native reference GLB (burn_synth CLI)"
  cargo run \
    -p burn_synth \
    --features runtime,wgpu \
    -- \
    --backend wgpu \
    --dino-backend auto \
    --num-steps 2 \
    --num-tokens 512 \
    --flash-min-resolution 15 \
    --faces 1000 \
    --seed 42 \
    --progress off \
    mesh \
    --input "${ROOT_DIR}/docs/input_chair.jpg" \
    --output "${NATIVE_REF_GLB}"
else
  rm -f "${NATIVE_REF_GLB}"
fi

echo "[web-e2e] run web integration tests"
BURN_SYNTH_WEBGPU_LOCK_HELD=1 \
BURN_SYNTH_NATIVE_REF_GLB="${NATIVE_REF_GLB}" \
  BURN_SYNTH_WEB_TMP_DIR="${TMP_DIR}" \
  npx playwright test --config playwright.config.mjs --workers=1 --reporter=list

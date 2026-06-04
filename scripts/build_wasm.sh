#!/usr/bin/env bash
set -euo pipefail

unset RUSTFLAGS
unset CARGO_ENCODED_RUSTFLAGS
unset CARGO_BUILD_RUSTFLAGS
unset CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS
unset RUSTDOCFLAGS

cargo build --target wasm32-unknown-unknown --profile wasm-release

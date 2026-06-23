#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CANONICAL_FILES=(
  "crates/burn_trellis/src/runtime_model/sparse_structure_decoder.rs"
  "crates/burn_trellis/src/runtime_model/sparse_structure_flow.rs"
  "crates/burn_trellis/src/runtime_model/sparse_decoder_runtime_impl.rs"
  "crates/burn_trellis/src/runtime_model/sparse_decoder_wgpu_ops.rs"
  "crates/burn_trellis/src/runtime_model/fdg_decoder.rs"
  "crates/burn_trellis/src/runtime_model/sparse_unet_vae_decoder.rs"
  "crates/burn_trellis/src/staged_pipeline_runtime_helpers.rs"
  "crates/burn_trellis/src/staged_pipeline_runtime_decode.rs"
  "crates/burn_trellis/src/staged_pipeline_sampling.rs"
)

EXPECTED_INTO_DATA_OCCURRENCES="$(cat <<'EOF'
crates/burn_trellis/src/runtime_model/sparse_decoder_wgpu_ops.rs#1
crates/burn_trellis/src/runtime_model/sparse_decoder_wgpu_ops.rs#2
crates/burn_trellis/src/runtime_model/sparse_decoder_wgpu_ops.rs#3
crates/burn_trellis/src/runtime_model/sparse_structure_flow.rs#1
crates/burn_trellis/src/runtime_model/sparse_structure_flow.rs#2
crates/burn_trellis/src/runtime_model/sparse_structure_flow.rs#3
crates/burn_trellis/src/runtime_model/sparse_structure_flow.rs#4
crates/burn_trellis/src/runtime_model/sparse_structure_flow.rs#5
crates/burn_trellis/src/runtime_model/sparse_structure_flow.rs#6
crates/burn_trellis/src/runtime_model/sparse_structure_flow.rs#7
crates/burn_trellis/src/runtime_model/sparse_structure_flow.rs#8
crates/burn_trellis/src/runtime_model/sparse_structure_flow.rs#9
crates/burn_trellis/src/runtime_model/sparse_structure_flow.rs#10
EOF
)"

EXPECTED_ENV_VAR_OCCURRENCES="$(cat <<'EOF'
crates/burn_trellis/src/runtime_model/sparse_decoder_wgpu_ops.rs#1
crates/burn_trellis/src/runtime_model/sparse_decoder_wgpu_ops.rs#2
crates/burn_trellis/src/runtime_model/sparse_decoder_wgpu_ops.rs#3
crates/burn_trellis/src/runtime_model/sparse_decoder_wgpu_ops.rs#4
crates/burn_trellis/src/runtime_model/sparse_structure_flow.rs#1
crates/burn_trellis/src/runtime_model/sparse_structure_flow.rs#2
crates/burn_trellis/src/runtime_model/sparse_structure_flow.rs#3
crates/burn_trellis/src/runtime_model/sparse_structure_flow.rs#4
crates/burn_trellis/src/runtime_model/sparse_structure_flow.rs#5
crates/burn_trellis/src/runtime_model/sparse_structure_flow.rs#6
crates/burn_trellis/src/runtime_model/sparse_structure_flow.rs#7
EOF
)"

collect_occurrences() {
  local needle="$1"
  local output_path="$2"
  local search_output
  local search_path
  search_output="$(
    for search_path in "${CANONICAL_FILES[@]}"; do
      awk -v file="${search_path}" -v needle="${needle}" '
        /^[[:space:]]*#\[cfg\(test\)\][[:space:]]*$/ {
          pending_test_cfg = 1
          next
        }
        pending_test_cfg && /^[[:space:]]*mod tests[[:space:]]*\{/ {
          exit
        }
        {
          pending_test_cfg = 0
        }
        index($0, needle) > 0 {
          printf "%s:%d:%s\n", file, NR, $0
        }
      ' "${search_path}"
    done
  )"
  if [[ -z "${search_output}" ]]; then
    : > "${output_path}"
    return 0
  fi
  search_output="$(printf '%s\n' "${search_output}" | sort -t: -k1,1 -k2,2n)"
  awk -F: '{
    count[$1]++
    printf "%s#%d\n", $1, count[$1]
  }' <<<"${search_output}" > "${output_path}"
}

check_baseline() {
  local label="$1"
  local needle="$2"
  local expected="$3"
  local actual_path
  local expected_path
  actual_path="$(mktemp)"
  expected_path="$(mktemp)"
  collect_occurrences "${needle}" "${actual_path}"
  printf '%s\n' "${expected}" > "${expected_path}"
  if ! diff -u "${expected_path}" "${actual_path}"; then
    echo
    echo "canonical runtime guard failed for ${label}"
    echo "If this change is intentional, update the inline allowlist in scripts/guard_canonical_runtime.sh."
    rm -f "${actual_path}" "${expected_path}"
    return 1
  fi
  rm -f "${actual_path}" "${expected_path}"
}

check_baseline \
  "into_data" \
  ".into_data(" \
  "${EXPECTED_INTO_DATA_OCCURRENCES}"

check_baseline \
  "std::env::var" \
  "std::env::var(" \
  "${EXPECTED_ENV_VAR_OCCURRENCES}"

run_strict_benchmark_invariants_if_configured() {
  local strict_log="${TRELLIS2_STRICT_BENCH_LOG:-}"
  if [[ -z "${strict_log}" ]]; then
    return 0
  fi

  if [[ ! -f "${strict_log}" ]]; then
    echo "strict benchmark log not found: ${strict_log}"
    return 1
  fi

  local py_bin="python3"
  if ! command -v "${py_bin}" >/dev/null 2>&1; then
    py_bin="python"
  fi
  if ! command -v "${py_bin}" >/dev/null 2>&1; then
    echo "python interpreter not found (tried python3, python)"
    return 1
  fi

  local cmd=(
    "${py_bin}"
    "${ROOT_DIR}/scripts/check_trellis_strict_benchmark_invariants.py"
    "${strict_log}"
    "--min-shape-dispatches"
    "${TRELLIS2_STRICT_BENCH_MIN_SHAPE_DISPATCHES:-1}"
    "--min-tex-dispatches"
    "${TRELLIS2_STRICT_BENCH_MIN_TEX_DISPATCHES:-1}"
  )

  if [[ -n "${TRELLIS2_STRICT_BENCH_BASELINE_LOG:-}" ]]; then
    cmd+=(
      "--baseline-log"
      "${TRELLIS2_STRICT_BENCH_BASELINE_LOG}"
      "--max-regression-pct"
      "${TRELLIS2_STRICT_BENCH_MAX_REGRESSION_PCT:-20}"
    )
  fi

  if [[ -n "${TRELLIS2_STRICT_BENCH_MAX_TOTAL_MS:-}" ]]; then
    cmd+=("--max-total-ms" "${TRELLIS2_STRICT_BENCH_MAX_TOTAL_MS}")
  fi
  if [[ -n "${TRELLIS2_STRICT_BENCH_MAX_SPARSE_MS:-}" ]]; then
    cmd+=("--max-sparse-ms" "${TRELLIS2_STRICT_BENCH_MAX_SPARSE_MS}")
  fi
  if [[ -n "${TRELLIS2_STRICT_BENCH_MAX_SHAPE_SLAT_MS:-}" ]]; then
    cmd+=("--max-shape-slat-ms" "${TRELLIS2_STRICT_BENCH_MAX_SHAPE_SLAT_MS}")
  fi
  if [[ -n "${TRELLIS2_STRICT_BENCH_MAX_TEX_SLAT_MS:-}" ]]; then
    cmd+=("--max-tex-slat-ms" "${TRELLIS2_STRICT_BENCH_MAX_TEX_SLAT_MS}")
  fi
  if [[ -n "${TRELLIS2_STRICT_BENCH_MAX_DECODE_MS:-}" ]]; then
    cmd+=("--max-decode-ms" "${TRELLIS2_STRICT_BENCH_MAX_DECODE_MS}")
  fi

  echo "running strict benchmark invariant guard for ${strict_log}"
  "${cmd[@]}"
}

run_strict_benchmark_invariants_if_configured

echo "canonical runtime guards passed"

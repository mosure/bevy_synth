#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

run_id="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_triposplat_cuda_reference_parity}"
run_dir="${RUN_DIR:-tmp/runs/${run_id}}"
reference_run="${TRIPOSPLAT_REFERENCE_RUN:-tmp/runs/20260604T074500Z_triposplat_cuda_alpha_reference}"
stage_tensors="${TRIPOSPLAT_STAGE_TENSORS:-${reference_run}/stage_tensors_f32.safetensors}"
reference_splat="${TRIPOSPLAT_REFERENCE_SPLAT:-${reference_run}/reference_32768.splat}"
weights_root="${TRIPOSPLAT_WEIGHTS_ROOT:-crates/burn_triposplat/assets/models/TripoSplat}"
weights_precision="${TRIPOSPLAT_WEIGHTS_PRECISION:-f32}"
compute_dtype="${TRIPOSPLAT_COMPUTE_DTYPE:-f32}"
python_bin="${PYTHON:-python3}"

mkdir -p "$run_dir"

if [[ -z "${CUDA_PATH:-}" && -d /usr/local/cuda-12.9 ]]; then
  export CUDA_PATH=/usr/local/cuda-12.9
  export PATH="${CUDA_PATH}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_PATH}/lib64:${LD_LIBRARY_PATH:-}"
fi

cuda_compute_cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ')"
nvcc_version="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\)\.\([0-9][0-9]*\).*/\1.\2/p' | head -n1)"
if [[ "${TRIPOSPLAT_SKIP_CUDA_TOOLKIT_PREFLIGHT:-0}" != "1" ]]; then
  compute_major="${cuda_compute_cap%%.*}"
  nvcc_major="${nvcc_version%%.*}"
  nvcc_minor="${nvcc_version#*.}"
  if [[ -n "$compute_major" && "$compute_major" -ge 12 ]]; then
    toolkit_too_old=0
    if [[ -z "$nvcc_version" ]]; then
      toolkit_too_old=1
    elif [[ "$nvcc_major" -lt 12 ]]; then
      toolkit_too_old=1
    elif [[ "$nvcc_major" -eq 12 && "$nvcc_minor" -lt 9 ]]; then
      toolkit_too_old=1
    fi
    if [[ "$toolkit_too_old" -eq 1 ]]; then
      cat >"${run_dir}/cuda_toolkit_blocker.txt" <<EOF
TripoSplat CUDA parity is blocked before model execution.

GPU compute capability: ${cuda_compute_cap:-unknown}
Visible nvcc/NVRTC toolkit: ${nvcc_version:-unknown}

This Blackwell GPU requires CUDA/NVRTC 12.9+ for CubeCL/Burn runtime kernel compilation.
Install CUDA 12.9+ or set CUDA_PATH and LD_LIBRARY_PATH to a matching toolkit, then retry.
Set TRIPOSPLAT_SKIP_CUDA_TOOLKIT_PREFLIGHT=1 only if you intentionally want to bypass this guard.
EOF
      cat "${run_dir}/cuda_toolkit_blocker.txt" >&2
      exit 86
    fi
  fi
fi

cat >"${run_dir}/config.json" <<JSON
{
  "run_id": "${run_id}",
  "stage_tensors": "${stage_tensors}",
  "reference_splat": "${reference_splat}",
  "weights_root": "${weights_root}",
  "weights_precision": "${weights_precision}",
  "compute_dtype": "${compute_dtype}",
  "cuda_compute_cap": "${cuda_compute_cap}",
  "nvcc_version": "${nvcc_version}",
  "cuda_path": "${CUDA_PATH:-}",
  "stage_assert": "${TRIPOSPLAT_CUDA_STAGE_ASSERT:-0}"
}
JSON

echo "[triposplat_cuda_reference_parity] run_dir=${run_dir}"
echo "[triposplat_cuda_reference_parity] stage_tensors=${stage_tensors}"
echo "[triposplat_cuda_reference_parity] reference_splat=${reference_splat}"

TRIPOSPLAT_CUDA_STAGE_PARITY=1 \
TRIPOSPLAT_DECODE_REPLAY=1 \
TRIPOSPLAT_RUST_CONDITION_FLOW=1 \
TRIPOSPLAT_REPLAY_SPLAT_DIR="$run_dir" \
TRIPOSPLAT_STAGE_TENSORS="$stage_tensors" \
TRIPOSPLAT_WEIGHTS_ROOT="$weights_root" \
TRIPOSPLAT_WEIGHTS_PRECISION="$weights_precision" \
TRIPOSPLAT_COMPUTE_DTYPE="$compute_dtype" \
cargo test -p burn_synth --features cuda triposplat_cuda_stage_parity_reference_tensors -- --nocapture \
  2>&1 | tee "${run_dir}/01_cuda_stage_parity.log"

if [[ -f "$reference_splat" && -f "${run_dir}/reference_condition_32768.splat" ]]; then
  "$python_bin" scripts/triposplat_compare_splat.py \
    "$reference_splat" \
    "${run_dir}/reference_condition_32768.splat" \
    --report "${run_dir}/reference_condition_splat_compare.json" \
    2>&1 | tee "${run_dir}/02_reference_condition_splat_compare.log"
fi

if [[ -f "$reference_splat" && -f "${run_dir}/rust_condition_32768.splat" ]]; then
  "$python_bin" scripts/triposplat_compare_splat.py \
    "$reference_splat" \
    "${run_dir}/rust_condition_32768.splat" \
    --report "${run_dir}/rust_condition_splat_compare.json" \
    2>&1 | tee "${run_dir}/03_rust_condition_splat_compare.log"
fi

echo "[triposplat_cuda_reference_parity] complete run_dir=${run_dir}"

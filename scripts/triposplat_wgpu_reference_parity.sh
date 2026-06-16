#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

run_id="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_triposplat_wgpu_reference_parity}"
run_dir="${RUN_DIR:-tmp/runs/${run_id}}"
reference_run="${TRIPOSPLAT_REFERENCE_RUN:-tmp/runs/20260604T120916Z_triposplat_cuda_reference_true_f32_no_tf32}"
stage_tensors="${TRIPOSPLAT_STAGE_TENSORS:-${reference_run}/stage_tensors_f32.safetensors}"
weights_root="${TRIPOSPLAT_WEIGHTS_ROOT:-crates/burn_triposplat/assets/models/TripoSplat}"
precision="${TRIPOSPLAT_WEIGHTS_PRECISION:-f32}"
steps="${TRIPOSPLAT_STEPS:-20}"
guidance_scale="${TRIPOSPLAT_GUIDANCE_SCALE:-3.0}"
shift="${TRIPOSPLAT_SHIFT:-3.0}"
gaussians="${TRIPOSPLAT_GAUSSIANS:-32768}"
stop_after="${TRIPOSPLAT_STOP_AFTER:-encode}"
cfg_mode="${TRIPOSPLAT_CFG_MODE:-batched}"
max_abs="${TRIPOSPLAT_STAGE_MAX_ABS:-1.0e-2}"
mean_abs="${TRIPOSPLAT_STAGE_MEAN_ABS:-1.0e-3}"
rms="${TRIPOSPLAT_STAGE_RMS:-2.0e-3}"
timeout_seconds="${TRIPOSPLAT_TIMEOUT_SECONDS:-1200}"
gpu_sample_ms="${GPU_SAMPLE_MS:-1000}"
candidate="${TRIPOSPLAT_CANDIDATE_STAGES:-${run_dir}/stage_tensors_wgpu_${precision}_${stop_after}.safetensors}"

mkdir -p "$run_dir"

cat >"${run_dir}/config.json" <<JSON
{
  "run_id": "${run_id}",
  "backend": "wgpu",
  "reference_stage_tensors": "${stage_tensors}",
  "candidate_stage_tensors": "${candidate}",
  "weights_root": "${weights_root}",
  "precision": "${precision}",
  "steps": ${steps},
  "guidance_scale": ${guidance_scale},
  "shift": ${shift},
  "gaussians": ${gaussians},
  "stop_after": "${stop_after}",
  "cfg_mode": "${cfg_mode}",
  "thresholds": {
    "max_abs": ${max_abs},
    "mean_abs": ${mean_abs},
    "rms": ${rms}
  },
  "timeout_seconds": ${timeout_seconds},
  "gpu_sample_ms": ${gpu_sample_ms}
}
JSON

if [[ ! -f "$stage_tensors" ]]; then
  echo "missing reference stage tensors: ${stage_tensors}" >&2
  exit 66
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader \
    >"${run_dir}/gpu_info.csv" || true
  nvidia-smi \
    --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total \
    --format=csv,noheader,nounits \
    -lms "$gpu_sample_ms" >"${run_dir}/gpu.csv" &
  gpu_pid=$!
else
  gpu_pid=""
  echo "nvidia-smi not found; gpu telemetry unavailable" >"${run_dir}/gpu_unavailable.txt"
fi

set +e
timeout "${timeout_seconds}s" cargo run -p burn_triposplat --features import,backend_wgpu --bin triposplat_stage_export -- \
  --backend wgpu \
  --weights-root "$weights_root" \
  --precision "$precision" \
  --input-stages "$stage_tensors" \
  --output "$candidate" \
  --seed 42 \
  --steps "$steps" \
  --guidance-scale "$guidance_scale" \
  --shift "$shift" \
  --gaussians "$gaussians" \
  --stop-after "$stop_after" \
  --cfg-mode "$cfg_mode" \
  2>&1 | tee "${run_dir}/01_wgpu_stage_export.log"
export_status=${PIPESTATUS[0]}
set -e

if [[ -n "${gpu_pid}" ]]; then
  kill "$gpu_pid" >/dev/null 2>&1 || true
  wait "$gpu_pid" >/dev/null 2>&1 || true
fi

if [[ "$export_status" -ne 0 ]]; then
  echo "[triposplat_wgpu_reference_parity] export failed status=${export_status} run_dir=${run_dir}" >&2
  exit "$export_status"
fi

compare_tensors=(image_rgb_0_1 dinov3_raw feature1 vae_mean vae_logvar feature2)
if [[ "$stop_after" == "sample" || "$stop_after" == "decode" ]]; then
  compare_tensors+=(latent camera)
fi
compare_args=()
for tensor in "${compare_tensors[@]}"; do
  compare_args+=(--tensor "$tensor")
done

set +e
cargo run -p burn_triposplat --features import --bin triposplat_stage_compare -- \
  "$stage_tensors" \
  "$candidate" \
  --report "${run_dir}/triposplat_wgpu_stage_compare.json" \
  --max-abs "$max_abs" \
  --mean-abs "$mean_abs" \
  --rms "$rms" \
  "${compare_args[@]}" \
  2>&1 | tee "${run_dir}/02_stage_compare.log"
compare_status=${PIPESTATUS[0]}
set -e

output_bytes=0
if [[ -f "$candidate" ]]; then
  output_bytes="$(wc -c <"$candidate" | tr -d ' ')"
fi

gpu_summary=""
if [[ -f "${run_dir}/gpu.csv" ]]; then
  gpu_summary="$(
    python3 - "${run_dir}/gpu.csv" <<'PY'
from pathlib import Path
import csv
import sys

rows = []
for row in csv.reader(Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()):
    if len(row) >= 5:
        try:
            rows.append((float(row[1]), float(row[2]), float(row[3]), float(row[4])))
        except ValueError:
            pass
if rows:
    print(
        f"gpu_mean={sum(r[0] for r in rows)/len(rows):.1f}% "
        f"gpu_max={max(r[0] for r in rows):.0f}% "
        f"mem_max={max(r[2] for r in rows):.0f}MiB "
        f"samples={len(rows)}"
    )
PY
  )"
fi

{
  echo "# TripoSplat WGPU Reference Parity"
  echo
  echo "- run_id: \`${run_id}\`"
  echo "- reference: \`${stage_tensors}\`"
  echo "- candidate: \`${candidate}\`"
  echo "- candidate_bytes: \`${output_bytes}\`"
  echo "- stop_after: \`${stop_after}\`"
  echo "- cfg_mode: \`${cfg_mode}\`"
  echo "- compare_report: \`${run_dir}/triposplat_wgpu_stage_compare.json\`"
  if [[ -n "$gpu_summary" ]]; then
    echo "- gpu_summary: \`${gpu_summary}\`"
  fi
  echo
  echo "## Compare"
  echo '```json'
  cat "${run_dir}/triposplat_wgpu_stage_compare.json"
  echo '```'
} >"${run_dir}/summary.md"

echo "[triposplat_wgpu_reference_parity] complete status=${compare_status} run_dir=${run_dir}"
exit "$compare_status"

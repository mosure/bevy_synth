#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

run_id="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_triposplat_wgpu_native_smoke}"
run_dir="${RUN_DIR:-tmp/runs/${run_id}}"
input="${TRIPOSPLAT_INPUT:-tmp/runs/20260604T074500Z_triposplat_cuda_alpha_reference/input_chair_alpha.png}"
weights_root="${TRIPOSPLAT_WEIGHTS_ROOT:-crates/burn_triposplat/assets/models/TripoSplat}"
precision="${TRIPOSPLAT_WEIGHTS_PRECISION:-f32}"
steps="${TRIPOSPLAT_STEPS:-5}"
guidance_scale="${TRIPOSPLAT_GUIDANCE_SCALE:-3.0}"
shift="${TRIPOSPLAT_SHIFT:-3.0}"
gaussians="${TRIPOSPLAT_GAUSSIANS:-32768}"
rmbg_model="${TRIPOSPLAT_RMBG_MODEL:-rmbg14}"
timeout_seconds="${TRIPOSPLAT_TIMEOUT_SECONDS:-900}"
gpu_sample_ms="${GPU_SAMPLE_MS:-1000}"
output="${TRIPOSPLAT_OUTPUT:-${run_dir}/wgpu_${precision}_${gaussians}.splat}"

mkdir -p "$run_dir"

cat >"${run_dir}/config.json" <<JSON
{
  "run_id": "${run_id}",
  "backend": "wgpu",
  "input": "${input}",
  "output": "${output}",
  "weights_root": "${weights_root}",
  "weights_precision": "${precision}",
  "steps": ${steps},
  "guidance_scale": ${guidance_scale},
  "shift": ${shift},
  "gaussians": ${gaussians},
  "rmbg_model": "${rmbg_model}",
  "timeout_seconds": ${timeout_seconds},
  "gpu_sample_ms": ${gpu_sample_ms}
}
JSON

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
timeout "${timeout_seconds}s" cargo run -p burn_synth --no-default-features --features wgpu -- \
  --backend wgpu \
  --synthesis-models triposplat \
  --rmbg-model "$rmbg_model" \
  --triposplat-weights-root "$weights_root" \
  --triposplat-weights-precision "$precision" \
  --num-steps "$steps" \
  --guidance-scale "$guidance_scale" \
  --triposplat-shift "$shift" \
  --gaussians "$gaussians" \
  --progress stages \
  splat \
  --input "$input" \
  --output "$output" \
  2>&1 | tee "${run_dir}/01_wgpu_cli.log"
status=${PIPESTATUS[0]}
set -e

if [[ -n "${gpu_pid}" ]]; then
  kill "$gpu_pid" >/dev/null 2>&1 || true
  wait "$gpu_pid" >/dev/null 2>&1 || true
fi

output_bytes=0
if [[ -f "$output" ]]; then
  output_bytes="$(wc -c <"$output" | tr -d ' ')"
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
  echo "# TripoSplat Native WGPU Smoke"
  echo
  echo "- run_id: \`${run_id}\`"
  echo "- status: \`${status}\`"
  echo "- output: \`${output}\`"
  echo "- output_bytes: \`${output_bytes}\`"
  echo "- log: \`${run_dir}/01_wgpu_cli.log\`"
  if [[ -f "${run_dir}/gpu.csv" ]]; then
    echo "- gpu_samples: \`${run_dir}/gpu.csv\`"
  fi
  if [[ -n "$gpu_summary" ]]; then
    echo "- gpu_summary: \`${gpu_summary}\`"
  fi
  echo
  echo "## Stage Lines"
  echo '```'
  grep 'burn_synth.progress' "${run_dir}/01_wgpu_cli.log" || true
  echo '```'
} >"${run_dir}/summary.md"

echo "[triposplat_wgpu_native_smoke] complete status=${status} run_dir=${run_dir}"
exit "$status"

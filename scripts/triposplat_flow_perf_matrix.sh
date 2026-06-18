#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

run_id="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_triposplat_flow_perf_matrix}"
run_dir="${RUN_DIR:-tmp/runs/${run_id}}"
reference_run="${TRIPOSPLAT_REFERENCE_RUN:-tmp/runs/20260604T120916Z_triposplat_cuda_reference_true_f32_no_tf32}"
stage_tensors="${TRIPOSPLAT_STAGE_TENSORS:-${reference_run}/stage_tensors_f32.safetensors}"
weights_root="${TRIPOSPLAT_WEIGHTS_ROOT:-crates/burn_triposplat/assets/models/TripoSplat}"
upstream_code="${TRIPOSPLAT_UPSTREAM_CODE:-tmp/upstream/TripoSplat/main}"
ckpt_root="${TRIPOSPLAT_CKPT_ROOT:-tmp/upstream/TripoSplat/VAST-AI-TripoSplat}"
torch_venv="${TRIPOSPLAT_TORCH_VENV:-${HOME}/.venvs/torch}"
quality_matrix="${TRIPOSPLAT_FLOW_MATRIX:-low:5:30000,balanced:20:131072,high:50:300000}"
burn_backends="${TRIPOSPLAT_BURN_BACKENDS:-wgpu}"
precision="${TRIPOSPLAT_PRECISION:-f32}"
torch_dtype="${TRIPOSPLAT_TORCH_DTYPE:-f32}"
torch_cfg_mode="${TRIPOSPLAT_TORCH_CFG_MODE:-batched}"
burn_cfg_mode="${TRIPOSPLAT_BURN_CFG_MODE:-batched-main}"
guidance_scale="${TRIPOSPLAT_GUIDANCE_SCALE:-3.0}"
shift="${TRIPOSPLAT_SHIFT:-3.0}"
torch_warmup_steps="${TRIPOSPLAT_TORCH_WARMUP_STEPS:-1}"
burn_warmup_steps="${TRIPOSPLAT_BURN_WARMUP_STEPS:-1}"
burn_timing_repeats="${TRIPOSPLAT_BURN_TIMING_REPEATS:-1}"
timeout_seconds="${TRIPOSPLAT_TIMEOUT_SECONDS:-2400}"
gpu_sample_ms="${GPU_SAMPLE_MS:-1000}"
max_abs="${TRIPOSPLAT_FLOW_MAX_ABS:-2.0e-2}"
mean_abs="${TRIPOSPLAT_FLOW_MEAN_ABS:-2.0e-4}"
rms="${TRIPOSPLAT_FLOW_RMS:-5.0e-4}"
run_torch="${TRIPOSPLAT_RUN_TORCH:-1}"
strict="${TRIPOSPLAT_FLOW_MATRIX_STRICT:-1}"

mkdir -p "$run_dir"

if [[ ! -f "$stage_tensors" ]]; then
  echo "missing stage tensors: ${stage_tensors}" >&2
  exit 66
fi

normalized_backends="${burn_backends//,/ }"
feature_parts=("import")
if [[ " ${normalized_backends} " == *" wgpu "* ]]; then
  feature_parts+=("backend_wgpu")
fi
if [[ " ${normalized_backends} " == *" cuda "* ]]; then
  feature_parts+=("backend_cuda")
  export CUDARC_CUDA_VERSION="${CUDARC_CUDA_VERSION:-13010}"
fi
features="$(IFS=,; echo "${feature_parts[*]}")"
export CUBECL_AUTOTUNE_LEVEL="${CUBECL_AUTOTUNE_LEVEL:-full}"

cat >"${run_dir}/config.json" <<JSON
{
  "run_id": "${run_id}",
  "stage_tensors": "${stage_tensors}",
  "weights_root": "${weights_root}",
  "upstream_code": "${upstream_code}",
  "ckpt_root": "${ckpt_root}",
  "torch_venv": "${torch_venv}",
  "quality_matrix": "${quality_matrix}",
  "burn_backends": "${normalized_backends}",
  "features": "${features}",
  "precision": "${precision}",
  "torch_dtype": "${torch_dtype}",
  "torch_cfg_mode": "${torch_cfg_mode}",
  "burn_cfg_mode": "${burn_cfg_mode}",
  "guidance_scale": ${guidance_scale},
  "shift": ${shift},
  "torch_warmup_steps": ${torch_warmup_steps},
  "burn_warmup_steps": ${burn_warmup_steps},
  "burn_timing_repeats": ${burn_timing_repeats},
  "timeout_seconds": ${timeout_seconds},
  "gpu_sample_ms": ${gpu_sample_ms},
  "thresholds": {
    "max_abs": ${max_abs},
    "mean_abs": ${mean_abs},
    "rms": ${rms}
  },
  "cubecl_autotune_level": "${CUBECL_AUTOTUNE_LEVEL}",
  "cudarc_cuda_version": "${CUDARC_CUDA_VERSION:-}"
}
JSON

cargo build --release -p burn_triposplat --features "$features" --bin triposplat_stage_export \
  2>&1 | tee "${run_dir}/00_build_stage_export.log"
cargo build -p burn_triposplat --features import --bin triposplat_stage_compare \
  2>&1 | tee "${run_dir}/00_build_stage_compare.log"

gpu_pid=""
cleanup() {
  if [[ -n "${gpu_pid}" ]]; then
    kill "$gpu_pid" >/dev/null 2>&1 || true
    wait "$gpu_pid" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader \
    >"${run_dir}/gpu_info.csv" || true
  nvidia-smi \
    --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total \
    --format=csv,noheader,nounits \
    -lms "$gpu_sample_ms" >"${run_dir}/gpu.csv" &
  gpu_pid=$!
else
  echo "nvidia-smi not found; gpu telemetry unavailable" >"${run_dir}/gpu_unavailable.txt"
fi

failures=0
IFS=',' read -r -a quality_entries <<< "$quality_matrix"
read -r -a backend_entries <<< "$normalized_backends"

for quality_entry in "${quality_entries[@]}"; do
  IFS=':' read -r quality steps gaussians <<< "$quality_entry"
  if [[ -z "${quality:-}" || -z "${steps:-}" || -z "${gaussians:-}" ]]; then
    echo "invalid quality entry '${quality_entry}', expected name:steps:gaussians" >&2
    exit 64
  fi

  torch_dir="${run_dir}/torch/${quality}"
  torch_sample="${torch_dir}/sample.safetensors"
  mkdir -p "$torch_dir"

  torch_status=0
  if [[ "$run_torch" == "1" ]]; then
    set +e
    (
      if [[ -f "${torch_venv}/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source "${torch_venv}/bin/activate"
      fi
      timeout "${timeout_seconds}s" python3 scripts/triposplat_torch_flow_bench.py \
        --upstream-code "$upstream_code" \
        --ckpt-root "$ckpt_root" \
        --stage-tensors "$stage_tensors" \
        --output-dir "$torch_dir" \
        --sample-output "$torch_sample" \
        --device cuda \
        --dtype "$torch_dtype" \
        --steps "$steps" \
        --guidance-scale "$guidance_scale" \
        --shift "$shift" \
        --cfg-mode "$torch_cfg_mode" \
        --warmup-steps "$torch_warmup_steps" \
        --disable-tf32
    ) 2>&1 | tee "${torch_dir}/torch_flow.log"
    torch_status=${PIPESTATUS[0]}
    set -e
  fi
  echo "$torch_status" >"${torch_dir}/status"
  if [[ "$torch_status" -ne 0 ]]; then
    echo "[triposplat_flow_perf_matrix] torch failed quality=${quality} status=${torch_status}" >&2
    failures=$((failures + 1))
  fi

  for backend in "${backend_entries[@]}"; do
    burn_dir="${run_dir}/burn/${backend}/${quality}"
    burn_sample="${burn_dir}/stage_tensors.safetensors"
    burn_timing="${burn_dir}/flow_timing.json"
    burn_compare="${burn_dir}/flow_compare.json"
    mkdir -p "$burn_dir"

    set +e
    timeout "${timeout_seconds}s" target/release/triposplat_stage_export \
      --backend "$backend" \
      --weights-root "$weights_root" \
      --precision "$precision" \
      --input-stages "$stage_tensors" \
      --output "$burn_sample" \
      --seed 42 \
      --steps "$steps" \
      --guidance-scale "$guidance_scale" \
      --shift "$shift" \
      --gaussians "$gaussians" \
      --stop-after sample \
      --cfg-mode "$burn_cfg_mode" \
      --use-reference-condition \
      --flow-warmup-steps "$burn_warmup_steps" \
      --flow-timing-repeats "$burn_timing_repeats" \
      --flow-timing-output "$burn_timing" \
      2>&1 | tee "${burn_dir}/burn_flow.log"
    burn_status=${PIPESTATUS[0]}
    set -e
    echo "$burn_status" >"${burn_dir}/status"

    compare_status=125
    if [[ "$burn_status" -eq 0 && -f "$torch_sample" ]]; then
      set +e
      target/debug/triposplat_stage_compare \
        "$torch_sample" \
        "$burn_sample" \
        --report "$burn_compare" \
        --tensor latent \
        --tensor camera \
        --max-abs "$max_abs" \
        --mean-abs "$mean_abs" \
        --rms "$rms" \
        2>&1 | tee "${burn_dir}/flow_compare.log"
      compare_status=${PIPESTATUS[0]}
      set -e
    fi
    echo "$compare_status" >"${burn_dir}/compare_status"

    if [[ "$burn_status" -ne 0 || "$compare_status" -ne 0 ]]; then
      echo "[triposplat_flow_perf_matrix] burn backend=${backend} quality=${quality} burn_status=${burn_status} compare_status=${compare_status}" >&2
      if [[ "$backend" == "wgpu" || "$strict" == "1" ]]; then
        failures=$((failures + 1))
      fi
    fi
  done
done

python3 - "$run_dir" <<'PY'
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
qualities = [entry.split(":")[0] for entry in config["quality_matrix"].split(",")]
backends = config["burn_backends"].split()


def read_status(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except FileNotFoundError:
        return None


def read_json(path: Path) -> object | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


gpu_rows: list[tuple[float, float, float, float]] = []
gpu_csv = run_dir / "gpu.csv"
if gpu_csv.exists():
    for row in csv.reader(gpu_csv.read_text(encoding="utf-8").splitlines()):
        if len(row) >= 5:
            try:
                gpu_rows.append((float(row[1]), float(row[2]), float(row[3]), float(row[4])))
            except ValueError:
                pass

gpu_summary = None
if gpu_rows:
    gpu_summary = {
        "samples": len(gpu_rows),
        "gpu_util_mean": sum(row[0] for row in gpu_rows) / len(gpu_rows),
        "gpu_util_max": max(row[0] for row in gpu_rows),
        "memory_used_mib_max": max(row[2] for row in gpu_rows),
        "memory_total_mib_max": max(row[3] for row in gpu_rows),
    }

results = []
for quality in qualities:
    torch_meta = read_json(run_dir / "torch" / quality / "metadata.json")
    torch_status = read_status(run_dir / "torch" / quality / "status")
    torch_ms = None
    if isinstance(torch_meta, dict):
        torch_ms = torch_meta.get("sample_ms_wall")
    for backend in backends:
        burn_dir = run_dir / "burn" / backend / quality
        timing = read_json(burn_dir / "flow_timing.json")
        compare = read_json(burn_dir / "flow_compare.json")
        burn_status = read_status(burn_dir / "status")
        compare_status = read_status(burn_dir / "compare_status")
        burn_ms = timing.get("sample_ms_avg") if isinstance(timing, dict) else None
        speed_ratio = None
        if isinstance(torch_ms, (int, float)) and isinstance(burn_ms, (int, float)) and torch_ms > 0:
            speed_ratio = burn_ms / torch_ms
        results.append(
            {
                "quality": quality,
                "backend": backend,
                "torch_status": torch_status,
                "burn_status": burn_status,
                "compare_status": compare_status,
                "torch_sample_ms_wall": torch_ms,
                "burn_sample_ms_avg": burn_ms,
                "burn_over_torch_ratio": speed_ratio,
                "burn_precision_policy": timing.get("backend_precision_policy")
                if isinstance(timing, dict)
                else None,
                "strict_reference_parity_supported": timing.get(
                    "strict_reference_parity_supported"
                )
                if isinstance(timing, dict)
                else None,
                "compare_passed": compare.get("passed") if isinstance(compare, dict) else None,
                "compare_failures": compare.get("failures") if isinstance(compare, dict) else None,
            }
        )

summary = {
    "schema": "triposplat_flow_perf_matrix_v1",
    "config": config,
    "gpu_summary": gpu_summary,
    "results": results,
}
(run_dir / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)

lines = [
    "# TripoSplat Flow Performance Matrix",
    "",
    f"- run_id: `{config['run_id']}`",
    f"- stage_tensors: `{config['stage_tensors']}`",
    f"- torch_cfg_mode: `{config['torch_cfg_mode']}`",
    f"- burn_cfg_mode: `{config['burn_cfg_mode']}`",
    f"- summary_json: `{run_dir / 'summary.json'}`",
]
if gpu_summary:
    lines.append(
        "- gpu_summary: "
        f"`gpu_mean={gpu_summary['gpu_util_mean']:.1f}% "
        f"gpu_max={gpu_summary['gpu_util_max']:.0f}% "
        f"mem_max={gpu_summary['memory_used_mib_max']:.0f}MiB "
        f"samples={gpu_summary['samples']}`"
    )
lines += ["", "| quality | backend | torch ms | burn ms | burn/torch | compare | precision policy |", "| --- | --- | ---: | ---: | ---: | --- | --- |"]
for row in results:
    torch_ms = row["torch_sample_ms_wall"]
    burn_ms = row["burn_sample_ms_avg"]
    ratio = row["burn_over_torch_ratio"]
    lines.append(
        "| {quality} | {backend} | {torch_ms} | {burn_ms} | {ratio} | {compare} | `{policy}` |".format(
            quality=row["quality"],
            backend=row["backend"],
            torch_ms=f"{torch_ms:.1f}" if isinstance(torch_ms, (int, float)) else "n/a",
            burn_ms=f"{burn_ms:.1f}" if isinstance(burn_ms, (int, float)) else "n/a",
            ratio=f"{ratio:.2f}x" if isinstance(ratio, (int, float)) and math.isfinite(ratio) else "n/a",
            compare="pass" if row["compare_passed"] else f"fail({row['compare_status']})",
            policy=row["burn_precision_policy"] or "unknown",
        )
    )
(run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(run_dir)
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "[triposplat_flow_perf_matrix] complete failures=${failures} run_dir=${run_dir}"
if [[ "$strict" == "1" && "$failures" -ne 0 ]]; then
  exit 1
fi

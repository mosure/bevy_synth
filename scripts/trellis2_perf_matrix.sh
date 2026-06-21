#!/usr/bin/env bash
set -euo pipefail

run_id="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_trellis2_python_burn_perf}"
run_dir="${RUN_DIR:-tmp/runs/${run_id}}"
input="${TRELLIS2_INPUT:-docs/output_chair_bg_removed.png}"
quality="${TRELLIS2_QUALITY:-medium}"
backend="${TRELLIS2_BACKEND:-wgpu}"
compute_profile="${TRELLIS2_COMPUTE_PROFILE:-wgpu-fast-f16}"
seed="${TRELLIS2_SEED:-42}"
repeat="${TRELLIS2_REPEAT:-1}"
strict="${TRELLIS2_STRICT:-1}"
run_python="${TRELLIS2_RUN_PYTHON:-1}"
run_burn="${TRELLIS2_RUN_BURN:-1}"
capture_reference_hook="${TRELLIS2_CAPTURE_REFERENCE_HOOK:-0}"
capture_row_noise="${TRELLIS2_CAPTURE_ROW_NOISE:-0}"
reference_hook="${TRELLIS2_REFERENCE_HOOK:-}"
python_bin="${TRELLIS2_PYTHON_BIN:-${HOME}/.venvs/torch/bin/python3}"
timeout_s="${TRELLIS2_TIMEOUT_S:-3600}"
gpu_sample_ms="${TRELLIS2_GPU_SAMPLE_MS:-1000}"
ratio_limit="${TRELLIS2_RATIO_LIMIT:-2.0}"

case "${quality}" in
  low)
    pipeline_type="512_base"
    max_num_tokens="49152"
    ;;
  medium | high)
    pipeline_type="1024_cascade"
    max_num_tokens="49152"
    ;;
  *)
    echo "unsupported TRELLIS2_QUALITY='${quality}'" >&2
    exit 2
    ;;
esac

mkdir -p "${run_dir}/python" "${run_dir}/burn"

cat > "${run_dir}/config.json" <<JSON
{
  "run_id": "${run_id}",
  "input": "${input}",
  "quality": "${quality}",
  "pipeline_type": "${pipeline_type}",
  "max_num_tokens": ${max_num_tokens},
  "backend": "${backend}",
  "compute_profile": "${compute_profile}",
  "seed": ${seed},
  "repeat": ${repeat},
  "strict": ${strict},
  "ratio_limit": ${ratio_limit},
  "capture_reference_hook": ${capture_reference_hook},
  "capture_row_noise": ${capture_row_noise},
  "reference_hook": "${reference_hook}"
}
JSON

gpu_monitor_pid=""
if [[ "${run_python}" != "1" && "${run_burn}" != "1" ]]; then
  :
elif command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=timestamp,index,name,driver_version,memory.used,memory.total,utilization.gpu,utilization.memory --format=csv > "${run_dir}/gpu_info.csv" || true
  nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,utilization.memory --format=csv --loop-ms="${gpu_sample_ms}" > "${run_dir}/gpu_samples.csv" &
  gpu_monitor_pid="$!"
else
  printf 'nvidia-smi not found; no GPU utilization samples captured\n' > "${run_dir}/gpu_samples_missing.txt"
fi

cleanup() {
  if [[ -n "${gpu_monitor_pid}" ]]; then
    kill "${gpu_monitor_pid}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if [[ "${run_python}" == "1" ]]; then
  if [[ ! -x "${python_bin}" ]]; then
    echo "Python executable not found or not executable: ${python_bin}" >&2
    exit 2
  fi
  if [[ "${capture_reference_hook}" == "1" ]]; then
    reference_hook="${run_dir}/python/reference_hook.safetensors"
    reference_hook_args=()
    if [[ "${capture_row_noise}" != "1" ]]; then
      reference_hook_args+=(--skip-row-noise-capture)
    fi
    set +e
    timeout "${timeout_s}s" "${python_bin}" scripts/trellis2_python_reference.py \
      --input "${input}" \
      --artifacts-dir "${run_dir}/python/reference_artifacts" \
      --output-hook "${reference_hook}" \
      --pipeline-type "${pipeline_type}" \
      --max-num-tokens "${max_num_tokens}" \
      --seed "${seed}" \
      --texture-size 1024 \
      --decimation-target 1000000 \
      "${reference_hook_args[@]}" \
      > "${run_dir}/python/reference_stdout.log" \
      2> "${run_dir}/python/reference_stderr.log"
    status="$?"
    set -e
    printf '%s\n' "${status}" > "${run_dir}/python/reference_status.txt"
    if [[ "${status}" != "0" ]]; then
      echo "Python reference hook capture failed; see ${run_dir}/python/reference_stderr.log" >&2
      if [[ "${strict}" == "1" ]]; then
        exit "${status}"
      fi
    fi
  fi
  set +e
  timeout "${timeout_s}s" "${python_bin}" scripts/trellis2_python_reference.py \
    --input "${input}" \
    --artifacts-dir "${run_dir}/python/artifacts" \
    --pipeline-type "${pipeline_type}" \
    --max-num-tokens "${max_num_tokens}" \
    --seed "${seed}" \
    --texture-size 1024 \
    --decimation-target 1000000 \
    --skip-hook-capture \
    > "${run_dir}/python/stdout.log" \
    2> "${run_dir}/python/stderr.log"
  status="$?"
  set -e
  printf '%s\n' "${status}" > "${run_dir}/python/status.txt"
  if [[ "${status}" != "0" ]]; then
    echo "Python reference failed; see ${run_dir}/python/stderr.log" >&2
    if [[ "${strict}" == "1" ]]; then
      exit "${status}"
    fi
  fi
fi

if [[ "${run_burn}" == "1" ]]; then
  cargo build --release -p burn_trellis --features runtime-model-wgpu --bin trellis2_run \
    > "${run_dir}/burn/build.log" \
    2> "${run_dir}/burn/build.err"
  burn_args=(
    --input "${input}" \
    --backend "${backend}" \
    --quality "${quality}" \
    --compute-profile "${compute_profile}" \
    --seed "${seed}" \
    --repeat "${repeat}" \
    --strict-benchmark \
    --require-runtime-model \
    --report-json "${run_dir}/burn/report.json"
  )
  if [[ -n "${reference_hook}" ]]; then
    burn_args+=(--noise-overrides-hook "${reference_hook}")
  fi
  set +e
  timeout "${timeout_s}s" target/release/trellis2_run "${burn_args[@]}" \
    > "${run_dir}/burn/stdout.log" \
    2> "${run_dir}/burn/stderr.log"
  status="$?"
  set -e
  printf '%s\n' "${status}" > "${run_dir}/burn/status.txt"
  if [[ "${status}" != "0" ]]; then
    echo "Burn run failed; see ${run_dir}/burn/stderr.log" >&2
    if [[ "${strict}" == "1" ]]; then
      exit "${status}"
    fi
  fi
fi

"${python_bin}" - "${run_dir}" "${ratio_limit}" "${strict}" <<'PY'
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

run_dir = Path(sys.argv[1])
ratio_limit = float(sys.argv[2])
strict = sys.argv[3] == "1"

def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

def load_status(path: Path) -> int | None:
    if not path.exists():
        return None
    return int(path.read_text(encoding="utf-8").strip())

def norm_pipeline(value: Any) -> str:
    text = str(value or "")
    return "512" if text == "512_base" else text

def get_path(payload: dict[str, Any], *path: str) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value

def ms(value: Any) -> float | None:
    if value is None:
        return None
    out = float(value)
    return out if math.isfinite(out) and out > 0 else None

def ratio(burn_ms: float | None, python_s: float | None) -> float | None:
    if burn_ms is None or python_s is None or python_s <= 0:
        return None
    return burn_ms / (python_s * 1000.0)

def max_gpu_sample(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False}
    max_util = 0.0
    max_mem = 0.0
    rows = 0
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows += 1
            util = row.get(" utilization.gpu [%]") or row.get("utilization.gpu [%]") or "0 %"
            mem = row.get(" memory.used [MiB]") or row.get("memory.used [MiB]") or "0 MiB"
            max_util = max(max_util, float(util.replace("%", "").strip() or 0))
            max_mem = max(max_mem, float(mem.replace("MiB", "").strip() or 0))
    return {"available": True, "samples": rows, "max_gpu_util_percent": max_util, "max_memory_mib": max_mem}

config = load_json(run_dir / "config.json") or {}
py_summary = load_json(run_dir / "python/artifacts/python_reference_summary.json")
burn_report = load_json(run_dir / "burn/report.json")
py_status = load_status(run_dir / "python/status.txt")
burn_status = load_status(run_dir / "burn/status.txt")

issues: list[str] = []
if py_status not in (None, 0):
    issues.append(f"python_status={py_status}")
if burn_status not in (None, 0):
    issues.append(f"burn_status={burn_status}")

shape_checks: dict[str, dict[str, Any]] = {}
if py_summary is not None and burn_report is not None:
    py_pipe = norm_pipeline(py_summary.get("pipeline_type"))
    burn_pipe = norm_pipeline(get_path(burn_report, "effective_config", "pipeline_type"))
    if py_pipe != burn_pipe:
        issues.append(f"pipeline_type mismatch python={py_pipe} burn={burn_pipe}")
    py_tokens = py_summary.get("max_num_tokens")
    burn_tokens = get_path(burn_report, "effective_config", "max_num_tokens")
    if burn_tokens is None:
        burn_tokens = 49152
    if int(py_tokens) != int(burn_tokens):
        issues.append(f"max_num_tokens mismatch python={py_tokens} burn={burn_tokens}")

    pairs = {
        "sparse_coords": ("sparse_coords", "sparse_coords"),
        "shape_slat_rows": ("decode_shape_input_rows", "shape_slat_rows"),
        "tex_slat_rows": ("decode_tex_input_rows", "tex_slat_rows"),
        "cond_512_tokens": ("cond_512_tokens", "cond_512_tokens"),
        "cond_1024_tokens": ("cond_1024_tokens", "cond_1024_tokens"),
    }
    py_shapes = py_summary.get("shapes", {})
    burn_shapes = burn_report.get("shapes", {})
    if "last" in burn_report:
        burn_shapes = get_path(burn_report, "last", "shapes") or burn_shapes
    for name, (py_key, burn_key) in pairs.items():
        py_value = int(py_shapes.get(py_key) or 0)
        burn_value = int(burn_shapes.get(burn_key) or 0)
        ok = py_value == burn_value
        shape_checks[name] = {"python": py_value, "burn": burn_value, "ok": ok}
        if not ok:
            issues.append(f"shape mismatch {name}: python={py_value} burn={burn_value}")

stage_ratios: dict[str, dict[str, Any]] = {}
if py_summary is not None and burn_report is not None:
    py_stage = py_summary.get("stage_timings_seconds", {})
    burn_timings = burn_report.get("timings_ms", {})
    if "last" in burn_report:
        burn_timings = get_path(burn_report, "last", "timings_ms") or burn_timings
    comparisons = {
        "preprocess": (ms(burn_timings.get("preprocess")), py_stage.get("preprocess")),
        "conditioning": (
            ms(burn_timings.get("sparse_cond")),
            (py_stage.get("get_cond_512") or 0.0) + (py_stage.get("get_cond_1024") or 0.0),
        ),
        "sparse": (ms(burn_timings.get("sparse")), py_stage.get("sparse_structure")),
        "shape_slat": (
            ms(burn_timings.get("shape_slat")),
            py_stage.get("shape_slat_cascade")
            or py_stage.get("shape_slat_32")
            or py_stage.get("shape_slat_512")
            or py_stage.get("shape_slat_1024"),
        ),
        "tex_slat": (ms(burn_timings.get("tex_slat")), py_stage.get("tex_slat")),
        "decode_shape": (ms(burn_timings.get("decode_shape_decoder")), py_stage.get("decode_shape")),
        "decode_tex": (ms(burn_timings.get("decode_tex_decoder")), py_stage.get("decode_tex")),
        "decode_total": (ms(burn_timings.get("decode")), py_stage.get("decode_latent_total")),
        "total": (ms(burn_timings.get("total")), get_path(py_summary, "timing_seconds", "infer")),
    }
    for name, (burn_ms, python_s) in comparisons.items():
        value = ratio(burn_ms, python_s)
        stage_ratios[name] = {
            "burn_ms": burn_ms,
            "python_ms": None if python_s is None else float(python_s) * 1000.0,
            "ratio": value,
            "within_limit": value is None or value <= ratio_limit,
        }
        if value is not None and value > ratio_limit:
            issues.append(f"stage ratio > {ratio_limit:g}: {name}={value:.3f}x")

summary = {
    "status": "ok" if not issues else "issues",
    "config": config,
    "python_status": py_status,
    "burn_status": burn_status,
    "shape_checks": shape_checks,
    "stage_ratios": stage_ratios,
    "gpu": max_gpu_sample(run_dir / "gpu_samples.csv"),
    "issues": issues,
    "python_summary": str(run_dir / "python/artifacts/python_reference_summary.json") if py_summary else None,
    "burn_report": str(run_dir / "burn/report.json") if burn_report else None,
}
(run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

lines = ["# TRELLIS.2 Python/Burn Perf Matrix", ""]
lines.append(f"- status: `{summary['status']}`")
lines.append(f"- run_dir: `{run_dir}`")
lines.append(f"- input: `{config.get('input')}`")
lines.append(f"- quality: `{config.get('quality')}`")
lines.append(f"- backend/profile: `{config.get('backend')}` / `{config.get('compute_profile')}`")
lines.append("")
lines.append("## Shape Checks")
for name, row in shape_checks.items():
    lines.append(f"- {name}: python={row['python']} burn={row['burn']} ok={row['ok']}")
lines.append("")
lines.append("## Stage Ratios")
for name, row in stage_ratios.items():
    ratio_value = row["ratio"]
    ratio_text = "n/a" if ratio_value is None else f"{ratio_value:.3f}x"
    lines.append(
        f"- {name}: burn_ms={row['burn_ms']} python_ms={row['python_ms']} ratio={ratio_text} within_limit={row['within_limit']}"
    )
lines.append("")
lines.append("## GPU")
lines.append(f"- {summary['gpu']}")
if issues:
    lines.append("")
    lines.append("## Issues")
    for issue in issues:
        lines.append(f"- {issue}")
(run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

print(json.dumps(summary, indent=2, sort_keys=True))
if strict and issues:
    raise SystemExit(1)
PY

echo "summary: ${run_dir}/summary.json"

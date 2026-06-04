#!/usr/bin/env bash
set -euo pipefail

if ! command -v cargo >/dev/null 2>&1; then
  echo "cargo not found" >&2
  exit 127
fi

run_id="${1:-}"
if [ -z "$run_id" ]; then
  run_id="$(date -u +%Y%m%dT%H%M%SZ)_w7_pbr_stage_matrix"
fi
out_dir="tmp/runs/$run_id"
mkdir -p "$out_dir"

warmup="${WARMUP:-1}"
iters="${ITERS:-4}"
fallback_res="${FALLBACK_RES:-64}"
grids="${GRIDS:-64 96 128}"
wgpu_sampling="${WGPU:-0}"

build_log="$out_dir/01_build.log"
timeout 360s cargo test -p burn_trellis --features runtime-model-wgpu pbr_bake_benchmark_report --no-run 2>&1 | tee "$build_log" >/dev/null

summary_csv="$out_dir/summary.csv"
echo "grid,vertices,triangles,voxels,fallback_res,iters,mean_ms,min_ms,p50_ms,p90_ms,covered_texels" > "$summary_csv"

for grid in $grids; do
  case_log="$out_dir/g${grid}.log"
  env \
    TRELLIS2_PBR_BENCH=1 \
    TRELLIS2_PBR_BENCH_GRID="$grid" \
    TRELLIS2_PBR_BENCH_WARMUP="$warmup" \
    TRELLIS2_PBR_BENCH_ITERS="$iters" \
    TRELLIS2_PBR_BENCH_FALLBACK_RES="$fallback_res" \
    TRELLIS2_PBR_BENCH_WGPU="$wgpu_sampling" \
    timeout 600s cargo test -p burn_trellis --features runtime-model-wgpu pbr_bake_benchmark_report -- --nocapture 2>&1 | tee "$case_log" >/dev/null

  line="$(grep -m1 '^PBR_BENCH_RESULT,' "$case_log" || true)"
  if [ -z "$line" ]; then
    echo "missing PBR_BENCH_RESULT in $case_log" >&2
    tail -n 120 "$case_log" >&2 || true
    exit 1
  fi

  row="$(echo "$line" | sed -E \
    -e 's/^PBR_BENCH_RESULT,//' \
    -e 's/grid=//' \
    -e 's/,vertices=/,/' \
    -e 's/,triangles=/,/' \
    -e 's/,voxels=/,/' \
    -e 's/,fallback_res=/,/' \
    -e 's/,iters=/,/' \
    -e 's/,mean_ms=/,/' \
    -e 's/,min_ms=/,/' \
    -e 's/,p50_ms=/,/' \
    -e 's/,p90_ms=/,/' \
    -e 's/,covered_texels=/,/')"
  echo "$row" >> "$summary_csv"
done

echo "RUN_ID=$run_id"
echo "OUT_DIR=$out_dir"
echo "SUMMARY=$summary_csv"
cat "$summary_csv"

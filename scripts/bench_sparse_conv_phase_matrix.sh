#!/usr/bin/env bash
set -euo pipefail

if ! command -v cargo >/dev/null 2>&1; then
  echo "cargo not found" >&2
  exit 127
fi

warmup="${WARMUP:-2}"
iters="${ITERS:-4}"
neighbor="${NEIGHBOR:-sorted}"

run_id="${1:-}"
if [ -z "$run_id" ]; then
  run_id="$(date -u +%Y%m%dT%H%M%SZ)_w6_sparse_conv_phase_matrix"
fi
out_dir="tmp/runs/$run_id"
mkdir -p "$out_dir"

configs=(
  "4096 3 64 128"
  "8192 3 64 128"
  "16384 3 64 128"
  "4096 3 64 256"
  "8192 3 64 256"
  "4096 3 128 256"
)
variants=(
  "auto auto"
  "baseline auto"
  "fused auto"
  "baseline 1"
  "baseline 2"
  "baseline 4"
  "fused 1"
  "fused 2"
  "fused 4"
)

build_log="$out_dir/01_build.log"
timeout 240s cargo build -p burn_flex_gmm --features wgpu-kernel --bin sparse_conv_stage_bench >"$build_log" 2>&1
bin="target/debug/sparse_conv_stage_bench"

for cfg in "${configs[@]}"; do
  read -r rows kernel in_ch out_ch <<< "$cfg"
  for v in "${variants[@]}"; do
    read -r variant split <<< "$v"
    name="r${rows}_k${kernel}_ic${in_ch}_oc${out_ch}_v${variant}_s${split}"
    "$bin" \
      --rows "$rows" \
      --kernel "$kernel" \
      --in-ch "$in_ch" \
      --out-ch "$out_ch" \
      --warmup "$warmup" \
      --iters "$iters" \
      --variant "$variant" \
      --split-k "$split" \
      --neighbor "$neighbor" \
      >"$out_dir/${name}.json" \
      2>"$out_dir/${name}.stderr"
  done
done

summary_csv="$out_dir/summary.csv"
{
  echo "case,rows,kernel,in_ch,out_ch,variant,split_arg,mean_ms,min_ms,p50_ms,p90_ms,resolved_variant,resolved_split_k,splitk_calls,fused_calls,single_group_specialized_calls"
  for f in "$out_dir"/*.json; do
    case_name="$(basename "$f" .json)"
    rows="$(echo "$case_name" | sed -E 's/^r([0-9]+)_k.*/\1/')"
    kernel="$(echo "$case_name" | sed -E 's/^r[0-9]+_k([0-9]+)_ic.*/\1/')"
    in_ch="$(echo "$case_name" | sed -E 's/^r[0-9]+_k[0-9]+_ic([0-9]+)_oc.*/\1/')"
    out_ch="$(echo "$case_name" | sed -E 's/^r[0-9]+_k[0-9]+_ic[0-9]+_oc([0-9]+)_v.*/\1/')"
    variant="$(echo "$case_name" | sed -E 's/^.*_v([^_]+)_s.*$/\1/')"
    split_arg="$(echo "$case_name" | sed -E 's/^.*_s([^_]+)$/\1/')"
    mean="$(grep -m1 '"mean_ms"' "$f" | sed -E 's/.*: ([0-9.]+).*/\1/')"
    min="$(grep -m1 '"min_ms"' "$f" | sed -E 's/.*: ([0-9.]+).*/\1/')"
    p50="$(grep -m1 '"p50_ms"' "$f" | sed -E 's/.*: ([0-9.]+).*/\1/')"
    p90="$(grep -m1 '"p90_ms"' "$f" | sed -E 's/.*: ([0-9.]+).*/\1/')"
    rv="$(grep -m1 '"resolved_variant"' "$f" | sed -E 's/.*: "([^"]+)".*/\1/')"
    rs="$(grep -m1 '"resolved_split_k"' "$f" | sed -E 's/.*: ([0-9]+).*/\1/')"
    sk="$(grep -m1 '"splitk_calls"' "$f" | sed -E 's/.*: ([0-9]+).*/\1/')"
    fc="$(grep -m1 '"fused_calls"' "$f" | sed -E 's/.*: ([0-9]+).*/\1/')"
    sg="$(grep -m1 '"single_group_specialized_calls"' "$f" | sed -E 's/.*: ([0-9]+).*/\1/')"
    echo "$case_name,$rows,$kernel,$in_ch,$out_ch,$variant,$split_arg,$mean,$min,$p50,$p90,$rv,$rs,$sk,$fc,$sg"
  done | sort
} > "$summary_csv"

best_csv="$out_dir/best_by_case.csv"
awk -F',' 'NR==1 {next}
{
  key=$2","$3","$4","$5;
  if (!(key in best) || $9+0 < best_min[key]+0) {
    best[key]=$0;
    best_min[key]=$9;
  }
}
END {
  print "rows,kernel,in_ch,out_ch,best_case,best_min_ms";
  for (k in best) {
    split(best[k],a,",");
    print a[2]","a[3]","a[4]","a[5]","a[1]","a[9];
  }
}' "$summary_csv" | sort -t',' -k1,1n -k2,2n -k3,3n -k4,4n > "$best_csv"

auto_vs_best_csv="$out_dir/auto_vs_best.csv"
awk -F',' '
NR==1 {next}
{
  key=$2","$3","$4","$5;
  if ($6=="auto" && $7=="auto") {
    auto[key]=$0;
  }
  if (!(key in best_row) || $10+0 < best_p50[key]+0) {
    best_row[key]=$0;
    best_p50[key]=$10;
  }
}
END {
  print "rows,kernel,in_ch,out_ch,auto_case,auto_mean_ms,auto_p50_ms,auto_resolved_variant,auto_resolved_split_k,best_case,best_mean_ms,best_p50_ms,best_resolved_variant,best_resolved_split_k,p50_delta_ms";
  for (k in auto) {
    if (!(k in best_row)) {
      continue;
    }
    split(auto[k],a,",");
    split(best_row[k],b,",");
    delta=(a[10]+0)-(b[10]+0);
    print a[2]","a[3]","a[4]","a[5]","a[1]","a[8]","a[10]","a[12]","a[13]","b[1]","b[8]","b[10]","b[12]","b[13]","delta;
  }
}' "$summary_csv" | sort -t',' -k1,1n -k2,2n -k3,3n -k4,4n > "$auto_vs_best_csv"

echo "RUN_ID=$run_id"
echo "OUT_DIR=$out_dir"
echo "SUMMARY=$summary_csv"
echo "BEST=$best_csv"
echo "AUTO_VS_BEST=$auto_vs_best_csv"
cat "$best_csv"

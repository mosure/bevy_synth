# W6 Sparse Conv Phase Matrix + Hotshape Tune (2026-02-28)

## Scope

Workstream: `W6 Sparse conv hotspot closure`.

This pass adds reproducible phase-level sparse-conv benchmarking artifacts and performs a targeted scheduler tune from matrix evidence.

## Implementation

### 1) Added reusable phase benchmark script

File:
- `scripts/bench_sparse_conv_phase_matrix.sh`

Behavior:
- builds `sparse_conv_stage_bench`
- runs a fixed matrix over rows/channels/variant/split combinations
- writes per-case JSON outputs under `tmp/runs/<run_id>/`
- emits:
  - `summary.csv`
  - `best_by_case.csv`

Defaults:
- `WARMUP=2`, `ITERS=4`, `NEIGHBOR=sorted`

### 2) Sparse-conv selector tuning from phase matrix

File:
- `crates/burn_flex_gmm/src/wgpu.rs`

Tuning changes:
- Added split-k cap for large row counts:
  - `rows >= 16384 => split_k <= 2`
- Added hot-shape fused auto route for single-group low-inner-work shape:
  - `rows == 4096`
  - `single-group ownership`
  - `inner_work <= 2048`
  - `output_work >= 500000`
  - route to `FusedOc4SingleGroup`

Rationale:
- bounded phase matrix indicated the `4096` single-group low-inner-work shape repeatedly favored fused paths.
- high-row split-4 often showed unstable overhead; cap added to avoid excessive split factor.

### 3) Added selector tests for new behavior

File:
- `crates/burn_flex_gmm/src/wgpu.rs`

New tests:
- `sparse_conv_auto_schedule_uses_single_group_fused_hot_shape_variant`
- `sparse_conv_auto_schedule_keeps_baseline_for_rows4096_when_inner_work_is_high`

## Bench Artifacts

### Matrix baseline run

- `tmp/runs/20260228T004120Z_w6_phase_matrix_v1/summary.csv`

### Post-tune focused rerun

- `tmp/runs/20260228T004541Z_w6_phase_matrix_v2_after_tune_rebuild/summary.csv`

### Reusable script full run (post-tune)

- `tmp/runs/20260228T004120Z_w6_sparse_conv_phase_matrix_tool_after/summary.csv`
- `tmp/runs/20260228T004120Z_w6_sparse_conv_phase_matrix_tool_after/best_by_case.csv`

Best-by-case snapshot (post-tune script run):
- `4096,3,64,128 -> r4096_k3_ic64_oc128_vfused_s4 (min_ms=3.803)`
- `4096,3,64,256 -> r4096_k3_ic64_oc256_vbaseline_s4 (min_ms=5.227)`
- `4096,3,128,256 -> r4096_k3_ic128_oc256_vbaseline_s2 (min_ms=15.257)`
- `8192,3,64,128 -> r8192_k3_ic64_oc128_vfused_sauto (min_ms=6.541)`
- `8192,3,64,256 -> r8192_k3_ic64_oc256_vfused_s1 (min_ms=18.117)`
- `16384,3,64,128 -> r16384_k3_ic64_oc128_vauto_sauto (min_ms=20.371)`

## Validation

Commands:
- `cargo fmt --all`
- `cargo check -p burn_flex_gmm --features wgpu-kernel`
- `cargo test -p burn_flex_gmm --features wgpu-kernel sparse_conv_auto_schedule_ -- --nocapture`
- `cargo test -p burn_flex_gmm --features wgpu-kernel wgpu_single_group_specialized_kernel_matches_cpu_flex_path -- --nocapture`
- `cargo test -p burn_flex_gmm --features wgpu-kernel wgpu_fused_oc4_matches_baseline_output -- --nocapture`
- `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run`
- `./scripts/guard_canonical_runtime.sh`

Results:
- all commands passed.
- existing unrelated warning persists in `burn_trellis` (`from_device_tensors` dead code).

## Notes

- Stage microbench variance is currently high on shared GPU runtime; use `summary.csv` and `best_by_case.csv` as directional tuning evidence, not final perf claims.
- W6 remains in progress; next closure step is to run decode-stage-integrated hotspots and lock selector policy against those real stage traces.

# W7 PBR Stage Matrix Bench (2026-02-28)

## Scope

Workstream: `W7 Decode/PBR GPU kernel closure`.

This pass adds a bounded stage-benchmark harness for decode/PBR CPU path and captures initial matrix evidence after the lookup-hasher optimization.

## Implementation

Files:
- `crates/burn_trellis/src/staged_pipeline_tests.rs`
- `scripts/bench_trellis_pbr_stage_matrix.sh`

### 1) Added benchmark-report test

New test:
- `pbr_bake_benchmark_report`

Behavior:
- gated by `TRELLIS2_PBR_BENCH=1`
- synthesizes deterministic plane mesh + voxel attributes
- runs `bake_pbr_from_voxels_with_options(..., capture_debug=false)`
- prints machine-readable `PBR_BENCH_RESULT,...` line with timings and coverage

Inputs via env:
- `TRELLIS2_PBR_BENCH_GRID`
- `TRELLIS2_PBR_BENCH_WARMUP`
- `TRELLIS2_PBR_BENCH_ITERS`
- `TRELLIS2_PBR_BENCH_FALLBACK_RES`

### 2) Added matrix driver script

Script:
- `scripts/bench_trellis_pbr_stage_matrix.sh`

Behavior:
- builds benchmark test binary once (`--no-run`)
- runs benchmark test for grid matrix values
- captures case logs in `tmp/runs/<run_id>/g<grid>.log`
- emits machine-readable `summary.csv`

Operational note:
- script uses `tee` + `pipefail` for capture because direct redirection path did not reliably surface cargo/test output in this environment.

## Run Artifacts

Run id:
- `20260228T011803Z_w7_pbr_stage_matrix_v1`

Files:
- `tmp/runs/20260228T011803Z_w7_pbr_stage_matrix_v1/summary.csv`
- `tmp/runs/20260228T011803Z_w7_pbr_stage_matrix_v1/00_driver.log`
- `tmp/runs/20260228T011803Z_w7_pbr_stage_matrix_v1/g64.log`
- `tmp/runs/20260228T011803Z_w7_pbr_stage_matrix_v1/g96.log`
- `tmp/runs/20260228T011803Z_w7_pbr_stage_matrix_v1/g128.log`

Summary snapshot:
- `64`: mean `5.675` ms, p50 `5.668` ms
- `96`: mean `6.269` ms, p50 `6.240` ms
- `128`: mean `6.795` ms, p50 `6.773` ms

Common settings:
- `warmup=1`
- `iters=4`
- `fallback_res=64`
- `voxels=4096`
- `covered_texels=65536`

## Validation

Commands:
- `cargo fmt --all`
- `cargo test -p burn_trellis --features runtime-model-wgpu pbr_bake_benchmark_report -- --nocapture`
- `cargo test -p burn_trellis --features runtime-model-wgpu pbr_bake_produces_textures_and_uvs -- --nocapture`
- `cargo test -p burn_trellis --features runtime-model-wgpu pbr_quantization_tracks_float_buffers -- --nocapture`
- `bash -n scripts/bench_trellis_pbr_stage_matrix.sh`
- `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run`
- `./scripts/guard_canonical_runtime.sh`

Result:
- all commands passed.
- existing unrelated warning persists (`SparseSubdivisionLogits::from_device_tensors` dead code).

## Next Step

- Use this harness to compare upcoming device-native decode/PBR kernels against current CPU path at identical synthetic stage inputs and bounded run budgets.

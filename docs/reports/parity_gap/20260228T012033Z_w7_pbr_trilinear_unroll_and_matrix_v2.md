# W7 PBR Trilinear Sampling Unroll + Matrix V2 (2026-02-28)

## Scope

Workstream: `W7 Decode/PBR GPU kernel closure`.

This pass optimizes the CPU-side PBR sampling hotspot further by replacing branchy nested corner loops with a fixed 8-corner trilinear path, then reruns the PBR stage matrix harness.

## Implementation

File:
- `crates/burn_trellis/src/staged_pipeline_decode.rs`

Change:
- Rewrote `sample_voxel_attr` corner accumulation:
  - from nested `for dz/dy/dx` loops with per-corner branch/clamp checks
  - to explicit 8-corner accumulation with precomputed clamped endpoints and trilinear weights
- Preserved semantics:
  - sparse holes still return `Ok(None)`
  - weighted normalization unchanged (`weight_sum > 1e-8`)
  - same coordinate convention and clamping behavior

Rationale:
- `sample_voxel_attr` is in the inner loop of decode/PBR raster sampling; reducing loop/branch overhead improves CPU stage throughput while kernel work is in progress.

## Validation

Commands:
- `cargo fmt --all`
- `cargo test -p burn_trellis --features runtime-model-wgpu sample_voxel_attr_returns_none_for_sparse_holes -- --nocapture`
- `cargo test -p burn_trellis --features runtime-model-wgpu sample_voxel_attr_returns_value_when_supported -- --nocapture`
- `cargo test -p burn_trellis --features runtime-model-wgpu pbr_bake_produces_textures_and_uvs -- --nocapture`
- `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run`
- `./scripts/guard_canonical_runtime.sh`

Result:
- all commands passed.
- existing unrelated warning persists (`SparseSubdivisionLogits::from_device_tensors` dead code).

## Stage Bench Evidence

Harness:
- `scripts/bench_trellis_pbr_stage_matrix.sh`

### Baseline matrix (v1)

Run:
- `tmp/runs/20260228T011803Z_w7_pbr_stage_matrix_v1/summary.csv`

p50 (ms):
- grid64: `5.668`
- grid96: `6.240`
- grid128: `6.773`

### Post-unroll matrix (v2)

Run:
- `tmp/runs/20260228T012033Z_w7_pbr_stage_matrix_v2_unrolled_sample/summary.csv`

p50 (ms):
- grid64: `4.905`
- grid96: `5.603`
- grid128: `6.243`

Directional deltas (v1 -> v2, p50):
- grid64: `-0.763 ms` (~13.5%)
- grid96: `-0.637 ms` (~10.2%)
- grid128: `-0.530 ms` (~7.8%)

Notes:
- These are bounded stage runs and should be treated as directional perf evidence.
- V2 still remains CPU-path evidence; full W7 closure still requires device-native decode/PBR kernels.

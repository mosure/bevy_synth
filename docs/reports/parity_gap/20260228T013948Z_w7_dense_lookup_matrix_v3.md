# W7 Dense Voxel Lookup + Matrix V3 (2026-02-28)

## Scope

Workstream: `W7 Decode/PBR GPU kernel closure`.

This pass adds an adaptive dense voxel lookup backend for PBR attribute sampling and captures stage-matrix evidence.

## Implementation

File:
- `crates/burn_trellis/src/staged_pipeline_decode.rs`

Changes:
1. Added adaptive lookup backend selection:
   - `VoxelAttrLookup::Dense` for bounded spatial volume
   - `VoxelAttrLookup::Sparse` (FxHashMap) for larger volumes
2. Added dense lookup builder:
   - `build_voxel_attr_lookup(...)`
3. Added lookup-dispatch sampler:
   - `sample_voxel_attr_from_lookup(...)`
4. Added dense trilinear sampler:
   - `sample_voxel_attr_dense(...)`
5. Integrated lookup backend into `bake_pbr_from_voxels_with_options`.

Bound:
- `DENSE_VOXEL_LOOKUP_MAX_CELLS = 2_500_000`

Semantics:
- unchanged sparse-hole semantics (`Ok(None)`)
- unchanged no-rescue canonical behavior
- unchanged coordinate convention and normalization logic

## Tests

File:
- `crates/burn_trellis/src/staged_pipeline_tests.rs`

Added:
- `dense_voxel_lookup_sampling_matches_sparse_hash_sampling`

Purpose:
- verifies dense lookup sampling matches sparse hash sampling for representative positions.

## Validation

Commands:
- `cargo fmt --all`
- `cargo test -p burn_trellis --features runtime-model-wgpu dense_voxel_lookup_sampling_matches_sparse_hash_sampling -- --nocapture`
- `cargo test -p burn_trellis --features runtime-model-wgpu pbr_bake_produces_textures_and_uvs -- --nocapture`
- `cargo test -p burn_trellis --features runtime-model-wgpu pbr_bake_benchmark_report -- --nocapture`
- `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run`
- `./scripts/guard_canonical_runtime.sh`

Result:
- all commands passed.
- existing unrelated warning persists (`SparseSubdivisionLogits::from_device_tensors` dead code).

## Stage Bench Evidence

Run:
- `tmp/runs/20260228T013922Z_w7_pbr_stage_matrix_v3_dense_lookup/summary.csv`

v3 p50 (ms):
- grid64: `4.554`
- grid96: `4.926`
- grid128: `5.624`

Comparison across W7 stage matrices (p50 ms):
- grid64: v1 `5.668` -> v2 `4.905` -> v3 `4.554`
- grid96: v1 `6.240` -> v2 `5.603` -> v3 `4.926`
- grid128: v1 `6.773` -> v2 `6.243` -> v3 `5.624`

Directional deltas:
- v1 -> v2: `-0.763`, `-0.637`, `-0.530` ms
- v2 -> v3: `-0.351`, `-0.677`, `-0.619` ms
- v1 -> v3: `-1.114`, `-1.314`, `-1.149` ms

Interpretation:
- dense lookup path provides additional stage-level reductions over the unrolled sparse-hash sampler in this bounded matrix.
- this remains CPU-path optimization evidence; W7 closure still requires device-native decode/PBR kernels.

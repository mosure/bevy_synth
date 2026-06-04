# W7 WGPU Dense Sampler Persistent Lookup (v5) (2026-02-28)

## Scope

Workstream: `W7 Decode/PBR GPU kernel closure`.

This pass removed repeated dense lookup uploads from the WGPU dense sampler path by introducing a per-bake sampler context that uploads occupancy/attrs once and reuses those tensors across batch sampling calls.

## Implementation

File:
- `crates/burn_trellis/src/staged_pipeline_decode.rs`

Changes:
1. Added `DenseVoxelWgpuSampler` context:
   - stores device, occupancy tensor, attrs tensor, and stride/axis metadata.
2. Added constructor:
   - `DenseVoxelWgpuSampler::new(...)` performs one-time validation + upload.
3. Refactored batch sampler signature:
   - `sample_voxel_attr_dense_wgpu_batch(positions, &sampler)`.
4. Updated deferred WGPU bake path:
   - builds sampler once per bake call and reuses it for all deferred position batches.
5. Increased WGPU batch size cap:
   - `DENSE_VOXEL_WGPU_SAMPLE_BATCH` from `16_384` to `65_536`.

Design note:
- Canonical runtime decode remains CPU-sampler gated (`prefer_wgpu_sampling=false`) because the WGPU path is still slower in bounded stage benches despite this improvement.

## Validation

Commands:
- `cargo fmt --all`
- `timeout 180s cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run`
- `timeout 240s cargo test -p burn_trellis --features runtime-model-wgpu dense_voxel_lookup_sampling_matches_sparse_hash_sampling -- --nocapture`
- `timeout 300s env BURN_WGPU_SMOKE=1 cargo test -p burn_trellis --features runtime-model-wgpu pbr_bake_wgpu_dense_sampling_matches_cpu_sampling -- --nocapture`
- `./scripts/guard_canonical_runtime.sh`

Result:
- All commands passed.
- Existing unrelated warning persists: `SparseSubdivisionLogits::from_device_tensors` dead code.

## Stage Bench Evidence

Runs:
- v4 CPU baseline: `tmp/runs/20260228T020112Z_w7_pbr_stage_matrix_v4_cpu_regression/summary.csv`
- v4 WGPU probe: `tmp/runs/20260228T020121Z_w7_pbr_stage_matrix_v4_wgpu_dense_sampler/summary.csv`
- v5 CPU post-refactor: `tmp/runs/20260228T021136Z_w7_pbr_stage_matrix_v5_cpu_post_refactor/summary.csv`
- v5 WGPU post-refactor: `tmp/runs/20260228T021147Z_w7_pbr_stage_matrix_v5_wgpu_post_refactor/summary.csv`

p50 comparison (ms):
- grid64: v4 WGPU `48.570` -> v5 WGPU `18.087` (delta `-30.483`); v5 CPU `4.338`
- grid96: v4 WGPU `32.918` -> v5 WGPU `14.666` (delta `-18.252`); v5 CPU `5.006`
- grid128: v4 WGPU `33.586` -> v5 WGPU `18.683` (delta `-14.903`); v5 CPU `5.584`

Interpretation:
- Persistent dense lookup tensors removed the dominant repeated-upload overhead and materially improved WGPU path latency.
- WGPU path is still slower than the optimized CPU path for this bounded stage matrix, so canonical runtime gate remains off.
- Remaining W7 closure requires deeper GPU path changes (tensor-native raster/sample accumulation without host-centric intermediate loops/readbacks).

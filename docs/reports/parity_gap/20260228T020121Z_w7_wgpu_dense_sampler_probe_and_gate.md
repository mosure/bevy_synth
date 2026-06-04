# W7 WGPU Dense Sampler Probe + Canonical Gate (2026-02-28)

## Scope

Workstream: `W7 Decode/PBR GPU kernel closure`.

This pass introduced and evaluated a tensor-batched WGPU dense voxel sampler for PBR bake sampling, then gated canonical runtime behavior based on measured phase evidence.

## Implementation

Files:
- `crates/burn_trellis/src/staged_pipeline_decode.rs`
- `crates/burn_trellis/src/staged_pipeline_runtime_decode.rs`
- `crates/burn_trellis/src/staged_pipeline_tests.rs`
- `scripts/bench_trellis_pbr_stage_matrix.sh`

Changes:
1. Added explicit sampler-control parameter to PBR bake path:
   - `bake_pbr_from_voxels_with_options(..., prefer_wgpu_sampling)`.
2. Added tensor-batched WGPU dense sampler implementation:
   - `sample_voxel_attr_dense_wgpu_batch(...)`.
3. Added deferred sample stream in bake loop for WGPU path, preserving first-hit semantics by replaying sampled results in original raster order.
4. Added bounded benchmark toggle:
   - `TRELLIS2_PBR_BENCH_WGPU` consumed by `pbr_bake_benchmark_report` and script passthrough `WGPU=0/1`.
5. Added smoke parity test:
   - `pbr_bake_wgpu_dense_sampling_matches_cpu_sampling` (guarded by `BURN_WGPU_SMOKE=1`).
6. Kept canonical runtime decode on CPU sampler:
   - `staged_pipeline_runtime_decode.rs` now sets `prefer_wgpu_sampling = false` with inline rationale comment.

Rationale for canonical gate:
- Current WGPU prototype uploads dense occupancy/attr tensors per batch and does not yet remove upload/sync overhead.
- Bounded stage matrix shows this path is slower than the optimized CPU path at current workload sizes.
- Canonical path remains on the faster proven route until kernel/dataflow redesign removes this overhead.

## Validation

Commands:
- `cargo fmt --all`
- `timeout 180s cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run`
- `timeout 240s cargo test -p burn_trellis --features runtime-model-wgpu sample_voxel_attr_ -- --nocapture`
- `timeout 240s cargo test -p burn_trellis --features runtime-model-wgpu dense_voxel_lookup_sampling_matches_sparse_hash_sampling -- --nocapture`
- `timeout 300s cargo test -p burn_trellis --features runtime-model-wgpu pbr_ -- --nocapture`
- `timeout 300s env BURN_WGPU_SMOKE=1 cargo test -p burn_trellis --features runtime-model-wgpu pbr_bake_wgpu_dense_sampling_matches_cpu_sampling -- --nocapture`
- `./scripts/guard_canonical_runtime.sh`

Result:
- All commands passed.
- Existing unrelated warning remains: `SparseSubdivisionLogits::from_device_tensors` dead code.

## Stage Bench Evidence

Runs:
- CPU regression run: `tmp/runs/20260228T020112Z_w7_pbr_stage_matrix_v4_cpu_regression/summary.csv`
- WGPU sampler run: `tmp/runs/20260228T020121Z_w7_pbr_stage_matrix_v4_wgpu_dense_sampler/summary.csv`

p50 comparison (ms):
- grid64: CPU `4.336` vs WGPU `48.570`
- grid96: CPU `4.972` vs WGPU `32.918`
- grid128: CPU `5.561` vs WGPU `33.586`

Interpretation:
- Prototype WGPU dense sampler is numerically correct (smoke parity test passes) but currently not performance-viable.
- Immediate next W7 kernel work should move dense lookup residency across calls (persistent device tensors / kernel-side lookup without per-batch upload) before re-enabling canonical runtime use.

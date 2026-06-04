# W7 Decode/PBR Lookup Hotspot Cut (2026-02-28)

## Scope

Workstream: `W7 Decode/PBR GPU kernel closure` (kickoff step).

This pass applies a narrow decode/PBR hotspot optimization without changing canonical strict semantics:
- sparse-hole handling remains unchanged (`None` => uncovered texel)
- no nearest/rescue inpaint behavior introduced
- fail-fast error surfaces remain intact

## Implementation

Files:
- `crates/burn_trellis/src/staged_pipeline_decode.rs`
- `crates/burn_trellis/Cargo.toml`

Changes:
1. Added `rustc-hash` dependency for fast hasher support.
2. Added `VoxelAttrMap` alias using `FxHasher`-backed `HashMap` specifically for voxel-attribute lookup in PBR bake.
3. Switched voxel map construction in `bake_pbr_from_voxels_with_options` to `with_capacity_and_hasher(..., FxHasher)`.
4. Generalized `sample_voxel_attr` map parameter to accept any `HashMap` hasher (`impl BuildHasher`) so tests and call sites remain compatible.

Rationale:
- PBR bake sampling performs many sparse key lookups per covered texel (8 neighbor probes per sample).
- Default hash builder is correctness-safe but expensive for this hot inner loop.
- This cut improves lookup efficiency while preserving exact sampling semantics.

## Validation

Commands:
- `cargo fmt --all`
- `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run`
- `cargo test -p burn_trellis --features runtime-model-wgpu sample_voxel_attr_returns_none_for_sparse_holes -- --nocapture`
- `cargo test -p burn_trellis --features runtime-model-wgpu sample_voxel_attr_returns_value_when_supported -- --nocapture`
- `cargo test -p burn_trellis --features runtime-model-wgpu pbr_bake_produces_textures_and_uvs -- --nocapture`
- `cargo test -p burn_trellis --features runtime-model-wgpu pbr_quantization_tracks_float_buffers -- --nocapture`
- `./scripts/guard_canonical_runtime.sh`

Results:
- all commands passed.
- existing unrelated warning persists (`SparseSubdivisionLogits::from_device_tensors` dead code).

## Notes

- This is a pre-kernel hotspot cut, not final W7 closure.
- Next W7 steps still require device-native decode/PBR kernels (grid sample/raster helpers) and stage-level benchmark evidence after those kernels land.

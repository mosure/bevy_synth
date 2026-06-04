# W4 Runtime Decode Device-Requirement Tightening

Date: 2026-02-28

## Scope

Removed remaining canonical WGPU decode/sampling host tensorization fallback surfaces in staged runtime path.

Files changed:

- `crates/burn_trellis/src/staged_pipeline_runtime_decode.rs`
- `crates/burn_trellis/src/staged_pipeline_sampling.rs`

## Changes

1. Runtime decode now fails fast when canonical WGPU tensors are missing:

- shape decode requires `shape.coords_wgpu` and `shape.features_wgpu`
- tex decode requires `tex.coords_wgpu` and `tex.features_wgpu`
- removed tex coord fallback that previously reused shape coords
- removed host-row -> WGPU tensorization fallback in runtime decode

2. Staged sampling now requires device trace rows for WGPU feature handoff:

- removed `trace.samples_wgpu == None` fallback that rebuilt WGPU feature tensors from host `[f32;32]` rows
- canonical WGPU path now errors if device trace rows are missing

## Why this change

This closes parity-critical fallback behavior around decode boundaries and prevents silent host materialization from re-entering canonical WGPU flow.

## Validation

Commands run:

```bash
cargo fmt --all
cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run
env BURN_WGPU_SMOKE=1 cargo test -p burn_trellis --features runtime-model-wgpu pbr_bake_wgpu_dense_sampling_matches_cpu_sampling -- --nocapture
cargo test -p burn_trellis --features runtime-model-wgpu cascade_token_budget_accepts_equal_token_count_without_backoff -- --nocapture
cargo test -p burn_trellis --features runtime-model-wgpu runtime_decode_tests::sanitize_mesh_geometry_removes_non_finite_vertices_and_reindexes_faces -- --nocapture
./scripts/guard_canonical_runtime.sh
```

Results:

- all commands passed
- existing unrelated warning persists: `SparseSubdivisionLogits::from_device_tensors` dead-code warning

## Outcome

Canonical WGPU staged decode/sampling path is stricter:

- no host row tensorization at runtime decode boundaries
- no shape->tex coord fallback in runtime decode
- no host-row feature tensor rebuild in WGPU staged sampling

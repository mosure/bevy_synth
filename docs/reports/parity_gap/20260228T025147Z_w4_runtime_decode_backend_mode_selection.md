# W4 Runtime Decode Backend-Mode Selection Fix

Date: 2026-02-28

## Scope

Fix staged runtime decode mode selection so canonical strictness is chosen from runtime tensor residency rather than compile-time `runtime-model-wgpu` feature selection.

Files changed:

- `crates/burn_trellis/src/staged_pipeline_runtime_decode.rs`

## Problem

The prior pass selected strict tensor-native decode path purely by `#[cfg(feature = "runtime-model-wgpu")]`.

That made strict canonical checks apply to all builds with WGPU support compiled in, including explicit host decode runs that intentionally did not carry device tensors.

## Changes

1. Added runtime decode mode gate:

- `using_device_decode_inputs` is true when any decode input tensor is device-resident (`shape/tex coords/features`).
- Added inline comment documenting why this gate must be runtime-driven.

2. Shape decode path selection now uses runtime gate:

- if `using_device_decode_inputs == true`: strict canonical tensor-native shape decode with fail-fast on missing device coords/rows
- else: host decode path (`decode_*_result` with host coords/rows)

3. Tex decode path selection now uses runtime gate:

- if `using_device_decode_inputs == true`: strict canonical tensor-native tex decode with fail-fast on missing device coords/rows
- else: host decode path (`decode_with_guidance_result` with host coords/rows)

4. Removed compile-time-only host slice availability assumptions:

- host slice views (`shape_rows_host`, `tex_rows_host`, `shape_coords_host`, `tex_coords_host`) are always prepared and then used by the selected runtime path.

## Why this change

Canonical WGPU must remain strict fail-fast once the staged flow has crossed into device-owned sparse tensors.

At the same time, compile-time feature flags must not force canonical WGPU decode behavior onto explicit non-WGPU decode runs.

This fix keeps both invariants:

- strict canonical device path remains strict
- host decode runs remain valid when no device decode tensors are present

## Validation

Commands run:

```bash
cargo fmt --all
cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run
cargo test -p burn_trellis --features runtime-model-wgpu runtime_decode_tests::runtime_decode_device_gate_allows_host_only_inputs -- --nocapture
cargo test -p burn_trellis --features runtime-model-wgpu runtime_decode_tests::sanitize_mesh_geometry_removes_non_finite_vertices_and_reindexes_faces -- --nocapture
cargo test -p burn_trellis --features runtime-model-wgpu cascade_token_budget_accepts_equal_token_count_without_backoff -- --nocapture
./scripts/guard_canonical_runtime.sh
```

Results:

- all commands passed
- existing unrelated warning persists: `SparseSubdivisionLogits::from_device_tensors` dead code warning

## Outcome

`staged_pipeline_runtime_decode` now selects decode mode using runtime tensor residency semantics, eliminating the compile-time gating regression while preserving canonical strict-fail behavior for device-native decode flow.

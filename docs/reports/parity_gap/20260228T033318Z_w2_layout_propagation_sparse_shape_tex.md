# W2 Layout Propagation Through Sparse -> Shape -> Tex

Date: 2026-02-28

## Scope

Propagate sparse batch layout metadata through staged samples so canonical WGPU shape/tex sparse-flow handoff no longer re-derives layout from `coords_wgpu` in the normal path.

Files changed:

- `crates/burn_trellis/src/staged_pipeline.rs`
- `crates/burn_trellis/src/staged_pipeline_sampling.rs`
- `crates/burn_trellis/src/staged_pipeline_tests.rs`

## Problem

Earlier pass replaced single-batch assumptions with layout derivation, but shape/tex WGPU path still extracted batch ids from `coords_wgpu` to build layout.

That introduced an avoidable host extraction surface in staged sparse ownership flow.

## Changes

1. Added layout ownership on staged samples:

- `SparseStructureSample.layout`
- `ShapeSLatSample.layout`
- `TexSLatSample.layout`

2. Sparse stage now emits layout metadata with coords:

- host override path computes layout from override coords
- non-WGPU sampled path computes layout from sampled host coords
- canonical WGPU sampled path keeps strict device coords and uses explicit layout metadata (`vec![0..rows]` for current decoder-single-batch emission when host coords are intentionally not materialized)

3. Shape stage now consumes propagated layout:

- `sample_shape_slat` now accepts `sparse_layout`
- `sample_shape_slat_with_model` uses provided layout on device path (no layout extraction from `coords_wgpu` in normal path)
- added explicit layout-row validation (`validate_sparse_layout_rows`)

4. Tex stage now consumes propagated shape layout:

- `sample_tex_slat_with_model` uses `shape_slat.layout` on device path
- added explicit layout-row validation
- no normal-path `coords_wgpu` layout extraction needed

5. Cascade path remains explicitly handled:

- when cascade produces fresh HR coord tensor, layout extraction helper remains used for that derived tensor path

6. Updated staged test fixtures for new layout fields.

## Why this change

This advances device-backed sparse ownership parity by treating batch layout as owned stage metadata rather than repeatedly reconstructing it from device coord readback in canonical shape/tex WGPU handoff.

## Validation

Commands run:

```bash
cargo fmt --all
cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run
cargo test -p burn_trellis --features runtime-model-wgpu sparse_layout_from_batch_ids_tracks_real_batched_ranges -- --nocapture
cargo test -p burn_trellis --features runtime-model-wgpu sparse_layout_from_coords_tracks_real_batched_ranges -- --nocapture
cargo test -p burn_trellis --features runtime-model-wgpu runtime_decode_tests::runtime_decode_device_gate_allows_host_only_inputs -- --nocapture
./scripts/guard_canonical_runtime.sh
```

Results:

- all commands passed
- existing unrelated warning persists: `SparseSubdivisionLogits::from_device_tensors` dead code warning

## Outcome

Canonical WGPU staged shape/tex sparse-flow path now receives batch layout from prior stages and validates it, reducing host-materialization pressure in sparse ownership flow while preserving strict fail-fast behavior.

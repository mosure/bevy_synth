# W2 Device Layout Derivation From WGPU Sparse Coords

Date: 2026-02-28

## Scope

Remove single-batch layout assumption on canonical WGPU sparse flow handoff by deriving sparse batch layout from device coord tensors.

Files changed:

- `crates/burn_trellis/src/staged_pipeline.rs`
- `crates/burn_trellis/src/staged_pipeline_sampling.rs`
- `crates/burn_trellis/src/staged_pipeline_tests.rs`

## Problem

When `coords_wgpu` was present, shape/tex staged sampling used `vec![0..rows]` as sparse layout.

That implicitly forced single-batch semantics for device-resident sparse flow handoff and blocked real batched ownership parity.

## Changes

1. Added shared layout builder for grouped batch ids:

- `sparse_layout_from_batch_ids(batch_ids, context)`
- reused by host `sparse_layout_from_coords`

2. Added WGPU coord-layout extraction helper:

- `sparse_layout_from_coords_wgpu(coords_t, context)`
- validates `[rows,4]` coord shape
- extracts batch column via canonical extraction boundary helper (`tensor_i32_to_vec`)
- validates non-negative batch ids and grouped-by-batch order semantics

3. Switched canonical device layout selection in staged sampling:

- `sample_shape_slat_with_model`: device path now derives layout from `coords_wgpu`
- `sample_tex_slat_with_model`: same
- removed single-batch `0..rows` assumption in these device paths

4. Added unit tests for new layout helper semantics:

- `sparse_layout_from_batch_ids_tracks_real_batched_ranges`
- `sparse_layout_from_batch_ids_rejects_non_grouped_rows`

## Why this change

This advances sparse ownership parity toward true batched semantics on device-resident flows and prevents hidden single-batch behavior from persisting in canonical WGPU staged sampling.

## Validation

Commands run:

```bash
cargo fmt --all
cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run
cargo test -p burn_trellis --features runtime-model-wgpu sparse_layout_from_batch_ids_tracks_real_batched_ranges -- --nocapture
cargo test -p burn_trellis --features runtime-model-wgpu sparse_layout_from_batch_ids_rejects_non_grouped_rows -- --nocapture
cargo test -p burn_trellis --features runtime-model-wgpu runtime_decode_tests::runtime_decode_device_gate_allows_host_only_inputs -- --nocapture
./scripts/guard_canonical_runtime.sh
```

Results:

- all commands passed
- existing unrelated warning persists: `SparseSubdivisionLogits::from_device_tensors` dead code warning

## Outcome

Canonical WGPU shape/tex sparse-flow handoff no longer assumes single-batch layout and now uses explicit batch-range derivation from device coords, preserving fail-fast semantics for invalid batch ordering.

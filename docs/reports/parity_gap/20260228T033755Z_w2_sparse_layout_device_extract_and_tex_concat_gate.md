# W2 Sparse Layout Device Extraction + Tex Concat Device Gate

Date: 2026-02-28

## Scope

Tighten staged sparse ownership semantics on canonical WGPU path by removing two remaining host-oriented seams:

- sparse-structure staged layout fallback that defaulted to `0..rows` when host coords were absent
- tex-slat concat build fallback that allowed host concat row materialization when device shape rows were missing

Files changed:

- `crates/burn_trellis/src/staged_pipeline_sampling.rs`

## Changes

1. Sparse-structure sampled layout now derives from device coords when host coords are intentionally not materialized.

- previous behavior: `sampled_layout` fell back to `vec![0..sampled_rows]`
- new behavior: when `sampled_host` is empty and a coord tensor exists, layout is derived via `sparse_layout_from_coords_wgpu(...)`
- fail-fast: if neither host coords nor device coord tensor is available, stage returns an explicit error

2. Sparse-structure host coord materialization gate is now tied to runtime backend pairing.

- canonical pair (`sparse_flow=wgpu` + `sparse_structure_decoder=wgpu`) keeps host coords disabled
- non-canonical pair retains caller-controlled host materialization to prevent empty host-surface regressions

3. Tex-slat canonical WGPU concat path now requires device shape rows.

- added strict gate: if `tex_flow` backend is WGPU and `shape_slat.features_wgpu` is missing, return explicit fail-fast error
- removed WGPU-path host concat fallback branch; concat tensor is now built only from `shape_rows_wgpu`

## Why this change

These two branches were still enabling host-first behavior at parity-critical sparse ownership boundaries.

The updated gates preserve strict canonical WGPU semantics and reduce accidental host materialization surfaces that can hide parity/perf regressions.

## Validation

Commands run:

```bash
cargo fmt --all
cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run
cargo test -p burn_trellis --features runtime-model-wgpu sparse_layout_from_batch_ids_tracks_real_batched_ranges -- --nocapture
cargo test -p burn_trellis --features runtime-model-wgpu sparse_layout_from_batch_ids_rejects_non_grouped_rows -- --nocapture
./scripts/guard_canonical_runtime.sh
```

Results:

- all commands passed
- existing unrelated warning persists: `SparseSubdivisionLogits::from_device_tensors` dead code warning

## Outcome

Canonical WGPU sparse staging no longer assumes single-batch `0..rows` layout in sampled sparse-structure output and no longer permits host concat rescue on tex-slat concat assembly.

This closes another host-materialization seam in W2 while keeping fail-fast runtime behavior explicit.

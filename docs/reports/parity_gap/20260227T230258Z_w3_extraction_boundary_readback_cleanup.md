# Parity Gap Run Report

- `run_id`: `20260227T230258Z_w3_extraction_boundary_readback_cleanup`
- `date_utc`: `2026-02-27`
- `git_ref`: `dirty_worktree_uncommitted`
- `dirty_worktree`: `true`

## Scope

- Workstream(s): `W3`, `W4` (increment)
- Goal: further reduce canonical-module host readback surfaces by routing sparse decoder WGPU tensor readbacks through extraction-boundary helpers and locking the stricter `.into_data()` guard baseline.
- Backend: `runtime-model-wgpu`
- Input(s): `targeted compile/runtime-decode slice`

## Command(s)

```bash
cargo fmt --all
./scripts/guard_canonical_runtime.sh
timeout 180s cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run
timeout 180s cargo test -p burn_trellis --features runtime-model-wgpu runtime_decode_tests::sanitize_mesh_geometry_removes_non_finite_vertices_and_reindexes_faces -- --nocapture
```

## Invariant Summary

- Canonical WGPU fail-fast only: `unchanged`
- Pre-extraction host readbacks: `reduced in canonical modules (sparse_decoder_wgpu_ops no longer does direct .into_data; extraction helper boundary used)`
- Decode dispatch presence: `not measured in this run`
- Runtime source identity: `not measured in this run`
- Canonical guard baseline lock after cleanup: `pass`
- Targeted runtime decode test slice: `pass`

## Timings (ms)

- Not collected in this run.

## Kernel / Telemetry Counters

- Not collected in this run.

## Outcome

- Status: `pass`
- Blocking issue(s): `none in this slice`
- Next action: `continue W4 by eliminating host subdivision completion fallback branch inside upsample_coords_sparse canonical flow path`

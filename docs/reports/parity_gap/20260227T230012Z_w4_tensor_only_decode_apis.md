# Parity Gap Run Report

- `run_id`: `20260227T230012Z_w4_tensor_only_decode_apis`
- `date_utc`: `2026-02-27`
- `git_ref`: `dirty_worktree_uncommitted`
- `dirty_worktree`: `true`

## Scope

- Workstream(s): `W3`, `W4` (increment)
- Goal: remove remaining coord-tensor + host-row decode/upsample surfaces from canonical module APIs/call paths, require tensor rows whenever tensor coords are used in cascade, and centralize sparse-structure coord readback through extraction boundary helpers.
- Backend: `runtime-model-wgpu`
- Input(s): `targeted unit/runtime-decode slices`

## Command(s)

```bash
cargo fmt --all
timeout 180s cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run
timeout 180s cargo test -p burn_trellis --features runtime-model-wgpu runtime_decode_tests::sanitize_mesh_geometry_removes_non_finite_vertices_and_reindexes_faces -- --nocapture
timeout 120s cargo test -p burn_trellis --features runtime-model-wgpu cascade_token_budget_accepts_equal_token_count_without_backoff -- --nocapture
./scripts/guard_canonical_runtime.sh
```

## Invariant Summary

- Canonical WGPU fail-fast only: `tightened`
- Pre-extraction host readbacks: `reduced in canonical modules (sparse_structure_decoder no longer contains direct .into_data readback)`
- Decode dispatch presence: `not measured in this run`
- Runtime source identity: `not measured in this run`
- Removed `decode_with_coords_tensor` / `upsample_coords_result_with_coords_tensor` host-row bridge APIs from canonical decoder wrappers/runtime impl: `pass`
- Cascade tensor-coord branch now requires tensor rows and rejects host completion fallback: `pass`
- Canonical guard baseline lock after reduction: `pass`

## Timings (ms)

- Not collected in this run.

## Kernel / Telemetry Counters

- Not collected in this run.

## Outcome

- Status: `pass`
- Blocking issue(s): `none in this slice`
- Next action: `continue W4 by replacing remaining host subdivision completion branch in upsample_coords_sparse with tensor-only canonical flow and tighten caller contracts around subdivision handoff`

# Parity Gap Run Report

- `run_id`: `20260227T231221Z_w4_upsample_host_entry_tensor_delegate`
- `date_utc`: `2026-02-27`
- `git_ref`: `dirty_worktree_uncommitted`
- `dirty_worktree`: `true`

## Scope

- Workstream(s): `W4` (increment)
- Goal: remove remaining host subdivision/decode completion flow from `upsample_coords_sparse` on WGPU builds by delegating host-input entrypoint to tensor-native `upsample_coords_result_with_tensors`.
- Backend: `runtime-model-wgpu`
- Input(s): `targeted runtime decode unit slice`

## Command(s)

```bash
cargo fmt --all
timeout 180s cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run
timeout 180s cargo test -p burn_trellis --features runtime-model-wgpu runtime_decode_tests::sanitize_mesh_geometry_removes_non_finite_vertices_and_reindexes_faces -- --nocapture
./scripts/guard_canonical_runtime.sh
```

## Invariant Summary

- Canonical WGPU fail-fast only: `unchanged`
- Pre-extraction host readbacks: `unchanged in this run`
- Decode dispatch presence: `not measured in this run`
- Runtime source identity: `not measured in this run`
- `upsample_coords_sparse` WGPU build path now forces tensor-native upsample/decode semantics and no longer executes host subdivision completion logic: `pass`
- Canonical guard baseline lock: `pass`

## Timings (ms)

- Not collected in this run.

## Kernel / Telemetry Counters

- Not collected in this run.

## Outcome

- Status: `pass`
- Blocking issue(s): `none in this slice`
- Next action: `continue W4 by tightening call graph to avoid host upsample entry in canonical staged runtime path entirely and then begin W5 neighbor-hash parallelization scaffolding`

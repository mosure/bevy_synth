# Parity Gap Run Report

- `run_id`: `20260227T222711Z_w2_cascade_tensor_upsample`
- `date_utc`: `2026-02-27`
- `git_ref`: `dirty_worktree_uncommitted`
- `dirty_worktree`: `true`

## Scope

- Workstream(s): `W2` (increment)
- Goal: remove host row dependency in canonical WGPU cascade upsample by adding decoder tensor-row upsample entrypoint and wiring shape cascade handoff to use `features_wgpu`.
- Backend: `N/A` (guard + compile + targeted unit test)
- Input(s): `N/A`

## Command(s)

```bash
cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run
cargo check -p burn_synth
cargo test -p burn_trellis --features runtime-model-wgpu runtime_decode_tests::sanitize_mesh_geometry_removes_non_finite_vertices_and_reindexes_faces -- --nocapture
./scripts/guard_canonical_runtime.sh
```

## Invariant Summary

- Canonical WGPU fail-fast only: `compile-verified`
- Pre-extraction host readbacks: `not measured in this run`
- Decode dispatch presence: `not measured in this run`
- Runtime source identity: `not measured in this run`
- Guardrail baseline lock: `pass`
- Decoder upsample now supports full tensor-native input (`upsample_coords_result_with_tensors`): `pass`
- Canonical WGPU cascade upsample now requires/uses device shape row tensor (`shape_lr.features_wgpu`) instead of host rows: `pass`

## Timings (ms)

- Not collected in this run.

## Kernel / Telemetry Counters

- Not collected in this run.

## Outcome

- Status: `pass`
- Blocking issue(s): `none`
- Next action: `continue W2 by removing remaining canonical WGPU host row assumptions in tex concat/shape-cond staging surfaces`

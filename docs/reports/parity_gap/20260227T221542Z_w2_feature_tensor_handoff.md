# Parity Gap Run Report

- `run_id`: `20260227T221542Z_w2_feature_tensor_handoff`
- `date_utc`: `2026-02-27`
- `git_ref`: `dirty_worktree_uncommitted`
- `dirty_worktree`: `true`

## Scope

- Workstream(s): `W2` (increment)
- Goal: carry shape/tex sampled rows as optional device tensors (`features_wgpu`) and consume them first in runtime decode to reduce host-materialized row handoff in canonical WGPU flow.
- Backend: `N/A` (guard + compile + targeted unit test)
- Input(s): `N/A`

## Command(s)

```bash
./scripts/guard_canonical_runtime.sh
cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run
cargo check -p burn_synth
cargo test -p burn_trellis --features runtime-model-wgpu runtime_decode_tests::sanitize_mesh_geometry_removes_non_finite_vertices_and_reindexes_faces -- --nocapture
```

## Invariant Summary

- Canonical WGPU fail-fast only: `compile-verified for tensor decode entrypoints`
- Pre-extraction host readbacks: `not measured in this run`
- Decode dispatch presence: `not measured in this run`
- Runtime source identity: `not measured in this run`
- Guardrail baseline lock: `pass`
- Staged sample structs now preserve optional device row tensors (`features_wgpu`) for shape/tex: `pass`
- Runtime decode prefers `features_wgpu` over host row tensorization on canonical device path: `pass`
- Targeted runtime decode unit test: `pass`

## Timings (ms)

- Not collected in this run.

## Kernel / Telemetry Counters

- Not collected in this run.

## Outcome

- Status: `pass`
- Blocking issue(s): `none`
- Next action: `continue W2 by propagating device-native row traces directly from sparse flow runtime outputs to avoid initial sparse-row host extraction in sampling`

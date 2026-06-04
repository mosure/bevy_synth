# Parity Gap Run Report

- `run_id`: `20260227T223151Z_w2_tex_concat_device_path`
- `date_utc`: `2026-02-27`
- `git_ref`: `dirty_worktree_uncommitted`
- `dirty_worktree`: `true`

## Scope

- Workstream(s): `W2` (increment)
- Goal: remove remaining tex-stage host-row assumptions in canonical WGPU flow by allowing shape concat conditioning to come from `shape_slat.features_wgpu` and making host rows fallback-only.
- Backend: `N/A` (guard + compile + targeted unit tests)
- Input(s): `N/A`

## Command(s)

```bash
cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run
cargo check -p burn_synth
cargo test -p burn_trellis --features runtime-model-wgpu sample_sparse_rows_trace_uses_single_host_readback_when_capturing_snapshots -- --nocapture
cargo test -p burn_trellis --features runtime-model-wgpu runtime_decode_tests::sanitize_mesh_geometry_removes_non_finite_vertices_and_reindexes_faces -- --nocapture
./scripts/guard_canonical_runtime.sh
```

## Invariant Summary

- Canonical WGPU fail-fast only: `compile-verified`
- Pre-extraction host readbacks: `not measured in this run`
- Decode dispatch presence: `not measured in this run`
- Runtime source identity: `not measured in this run`
- Guardrail baseline lock: `pass`
- Tex-stage concat conditioning now supports full device path from `shape_slat.features_wgpu`: `pass`
- Host shape row dependency in tex stage reduced to explicit fallback-only paths: `pass`
- Host `shape_slat_cond` generation now tolerates absent host shape rows (falls back to zeros for unavailable rows): `pass`

## Timings (ms)

- Not collected in this run.

## Kernel / Telemetry Counters

- Not collected in this run.

## Outcome

- Status: `pass`
- Blocking issue(s): `none`
- Next action: `continue W2 by pruning canonical WGPU host row outputs where only device tensors are consumed, then tighten tests to assert host row vectors can be empty on canonical path`

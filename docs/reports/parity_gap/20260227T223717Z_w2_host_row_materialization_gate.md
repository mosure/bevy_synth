# Parity Gap Run Report

- `run_id`: `20260227T223717Z_w2_host_row_materialization_gate`
- `date_utc`: `2026-02-27`
- `git_ref`: `dirty_worktree_uncommitted`
- `dirty_worktree`: `true`

## Scope

- Workstream(s): `W2` (increment)
- Goal: gate sparse-row host materialization with explicit `materialize_host_rows` flow control and use it in staged shape/tex WGPU sampling so canonical non-trace runs can keep host row vectors empty.
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
- Pre-extraction host readbacks: `gated by explicit materialization flag for sparse-row traces`
- Decode dispatch presence: `not measured in this run`
- Runtime source identity: `not measured in this run`
- Guardrail baseline lock: `pass`
- Sparse-flow trace now supports no-host-vector mode while preserving device tensors: `pass`
- Staged shape/tex canonical WGPU paths now set host row materialization off when sampler trace capture is disabled: `pass`
- CPU/test paths still materialize host rows and preserve existing readback-count assertion behavior: `pass`

## Timings (ms)

- Not collected in this run.

## Kernel / Telemetry Counters

- Not collected in this run.

## Outcome

- Status: `pass`
- Blocking issue(s): `none`
- Next action: `continue W2 by adding a targeted WGPU-gated test that asserts shape/tex host row vectors can be empty while decode succeeds via device tensors`

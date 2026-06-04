# Parity Gap Run Report

- `run_id`: `20260227T222345Z_w2_trace_tensor_bridge`
- `date_utc`: `2026-02-27`
- `git_ref`: `dirty_worktree_uncommitted`
- `dirty_worktree`: `true`

## Scope

- Workstream(s): `W2` (increment)
- Goal: propagate sparse-flow row traces with optional WGPU tensors and consume trace tensors in staged sampling feature handoff, avoiding host->device re-upload for canonical decode feature tensors.
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
- Pre-extraction host readbacks: `unchanged/readback-count test preserved`
- Decode dispatch presence: `not measured in this run`
- Runtime source identity: `not measured in this run`
- Guardrail baseline lock: `pass`
- Sparse flow row trace now carries optional device row tensors (`samples_wgpu`, snapshots): `pass`
- Staged shape/tex feature tensor handoff now prefers trace device tensors (denorm+pad on device) over host re-upload fallback: `pass`

## Timings (ms)

- Not collected in this run.

## Kernel / Telemetry Counters

- Not collected in this run.

## Outcome

- Status: `pass`
- Blocking issue(s): `none`
- Next action: `continue W2 by switching canonical WGPU sampling/decode handoff consumers to tolerate empty host row vectors and use device row tensors end-to-end`

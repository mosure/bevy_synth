# Parity Gap Run Report

- `run_id`: `20260227T224658Z_w4_decoder_tensor_input_gate`
- `date_utc`: `2026-02-27`
- `git_ref`: `dirty_worktree_uncommitted`
- `dirty_worktree`: `true`

## Scope

- Workstream(s): `W4` (increment)
- Goal: remove canonical decoder host-completion entry by requiring device-backed coords+rows for canonical WGPU decode path (`decode_with_tensors` only).
- Backend: `runtime-model-wgpu`
- Input(s): `targeted runtime decode unit slice`

## Command(s)

```bash
timeout 240s cargo test -p burn_trellis --features runtime-model-wgpu runtime_decode_tests::sanitize_mesh_geometry_removes_non_finite_vertices_and_reindexes_faces -- --nocapture
./scripts/guard_canonical_runtime.sh
timeout 120s cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run
```

## Invariant Summary

- Canonical WGPU fail-fast only: `tightened`
- Pre-extraction host readbacks: `unchanged in this run`
- Decode dispatch presence: `not measured in this run`
- Runtime source identity: `not measured in this run`
- Canonical decoder now rejects host-backed decode inputs in canonical mode (`coords` or `rows` host surfaces): `pass`
- Existing targeted runtime decode unit slice: `pass`
- Canonical guard baseline lock: `pass`

## Timings (ms)

- Not collected in this run.

## Kernel / Telemetry Counters

- Not collected in this run.

## Outcome

- Status: `pass`
- Blocking issue(s): `none in this slice`
- Next action: `continue W4 by removing remaining canonical host completion branches in stage0/subdivision helper paths and converting callers to explicit tensor APIs only`

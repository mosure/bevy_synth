# Parity Gap Run Report

- `run_id`: `20260227T214400Z_w2_tex_coord_tensor_unit_test`
- `date_utc`: `2026-02-27`
- `git_ref`: `dirty_worktree_uncommitted`
- `dirty_worktree`: `true`

## Scope

- Workstream(s): `W2` (increment verification)
- Goal: validate staged runtime/unit test compilation after introducing `TexSLatSample::coords_wgpu` and decode handoff updates.
- Backend: `N/A` (unit-test compile/run only)
- Input(s): `N/A`

## Command(s)

```bash
timeout 300s cargo test -p burn_trellis --features runtime-model-wgpu sparse_coord_cap_requires_explicit_override -- --nocapture
```

## Invariant Summary

- Canonical WGPU fail-fast only: `not validated in this run`
- Pre-extraction host readbacks: `not measured in this run`
- Decode dispatch presence: `not measured in this run`
- Runtime source identity: `not measured in this run`
- Targeted unit test compile and run: `pass`

## Timings (ms)

- Not collected in this run.

## Kernel / Telemetry Counters

- Not collected in this run.

## Outcome

- Status: `pass`
- Blocking issue(s): `none`
- Next action: `continue W2 migration by reducing host-first feature ownership surfaces`

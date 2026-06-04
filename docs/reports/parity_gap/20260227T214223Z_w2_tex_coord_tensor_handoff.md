# Parity Gap Run Report

- `run_id`: `20260227T214223Z_w2_tex_coord_tensor_handoff`
- `date_utc`: `2026-02-27`
- `git_ref`: `dirty_worktree_uncommitted`
- `dirty_worktree`: `true`

## Scope

- Workstream(s): `W2` (increment)
- Goal: extend device-backed ownership in staged flow by carrying tex coord tensors through sampling and runtime decode.
- Backend: `N/A` (build + guard validation only)
- Input(s): `N/A`

## Command(s)

```bash
./scripts/guard_canonical_runtime.sh
timeout 240s cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run
timeout 240s cargo check -p burn_synth
```

## Invariant Summary

- Canonical WGPU fail-fast only: `not validated in this run`
- Pre-extraction host readbacks: `not measured in this run`
- Decode dispatch presence: `not measured in this run`
- Runtime source identity: `not measured in this run`
- Guardrail baseline lock: `pass`
- Tex coord tensor handoff through decode path: `compile-verified`

## Timings (ms)

- Not collected in this run.

## Kernel / Telemetry Counters

- Not collected in this run.

## Outcome

- Status: `pass`
- Blocking issue(s): `none`
- Next action: `continue W2 by introducing device-backed row-feature ownership (shape/tex slat) and reducing host-first sample surfaces`

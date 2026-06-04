# Parity Gap Run Report

- `run_id`: `20260227T214732Z_w2_runtime_device_bridge_api`
- `date_utc`: `2026-02-27`
- `git_ref`: `dirty_worktree_uncommitted`
- `dirty_worktree`: `true`

## Scope

- Workstream(s): `W2` (increment)
- Goal: add runtime API bridges that return `types::*Device` ownership directly from WGPU tensor assembly calls.
- Backend: `N/A` (guard + compile validation)
- Input(s): `N/A`

## Command(s)

```bash
./scripts/guard_canonical_runtime.sh
timeout 180s cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run
```

## Invariant Summary

- Canonical WGPU fail-fast only: `not validated in this run`
- Pre-extraction host readbacks: `not measured in this run`
- Decode dispatch presence: `not measured in this run`
- Runtime source identity: `not measured in this run`
- Guardrail baseline lock: `pass`
- Runtime device-bridge APIs compile: `pass`

## Timings (ms)

- Not collected in this run.

## Kernel / Telemetry Counters

- Not collected in this run.

## Outcome

- Status: `pass`
- Blocking issue(s): `none`
- Next action: `switch selected staging call-sites from *Owned surfaces to new device-bridge API returns`

# Parity Gap Run Report

- `run_id`: `20260227T213351Z_w2_types_scaffold`
- `date_utc`: `2026-02-27`
- `git_ref`: `dirty_worktree_uncommitted`
- `dirty_worktree`: `true`

## Scope

- Workstream(s): `W2` (scaffold)
- Goal: Introduce `runtime_model/types/*` device-first ownership modules and extraction boundary helpers.
- Backend: `N/A` (build + guard validation only)
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
- Device-first types module scaffold compiles: `pass`

## Timings (ms)

- Not collected in this run.

## Kernel / Telemetry Counters

- Not collected in this run.

## Outcome

- Status: `pass`
- Blocking issue(s): `none`
- Next action: `begin migration of staged sampling/flow ownership from host-first structs to runtime_model::types device-first wrappers`

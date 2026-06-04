# Parity Gap Run Report

- `run_id`: `20260227T214621Z_w2_device_bridge_conversions`
- `date_utc`: `2026-02-27`
- `git_ref`: `dirty_worktree_uncommitted`
- `dirty_worktree`: `true`

## Scope

- Workstream(s): `W2` (increment)
- Goal: add bridge conversions from hybrid sparse ownership types into `runtime_model::types::*Device` and validate behavior.
- Backend: `N/A` (compile + targeted unit tests)
- Input(s): `N/A`

## Command(s)

```bash
./scripts/guard_canonical_runtime.sh
timeout 240s cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run
timeout 300s cargo test -p burn_trellis --features runtime-model-wgpu device_owned_conversion_requires_device -- --nocapture
```

## Invariant Summary

- Canonical WGPU fail-fast only: `not validated in this run`
- Pre-extraction host readbacks: `not measured in this run`
- Decode dispatch presence: `not measured in this run`
- Runtime source identity: `not measured in this run`
- Guardrail baseline lock: `pass`
- Device-bridge conversion methods compile: `pass`
- Host-only conversion refusal tests: `pass` (2 targeted tests)

## Timings (ms)

- Not collected in this run.

## Kernel / Telemetry Counters

- Not collected in this run.

## Outcome

- Status: `pass`
- Blocking issue(s): `none`
- Next action: `continue W2 by routing staged sampling/decode APIs through device bridge types and removing host-first call surfaces`

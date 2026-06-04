# Parity Gap Run Report

- `run_id`: `20260227T220028Z_w2_sampling_device_api_switch`
- `date_utc`: `2026-02-27`
- `git_ref`: `dirty_worktree_uncommitted`
- `dirty_worktree`: `true`

## Scope

- Workstream(s): `W2` (increment)
- Goal: switch staged shape/tex sampling WGPU path to runtime tensor-input API to reduce direct host-first ownership assembly in staging layer.
- Backend: `N/A` (guard + compile + targeted unit test)
- Input(s): `N/A`

## Command(s)

```bash
./scripts/guard_canonical_runtime.sh
timeout 240s cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run
timeout 240s cargo check -p burn_synth
timeout 300s cargo test -p burn_trellis --features runtime-model-wgpu sparse_coord_cap_requires_explicit_override -- --nocapture
```

## Invariant Summary

- Canonical WGPU fail-fast only: `not validated in this run`
- Pre-extraction host readbacks: `not measured in this run`
- Decode dispatch presence: `not measured in this run`
- Runtime source identity: `not measured in this run`
- Guardrail baseline lock: `pass`
- Staged WGPU sampling switched to runtime tensor-input bridge API: `compile-verified`
- Targeted unit test run: `pass`

## Timings (ms)

- Not collected in this run.

## Kernel / Telemetry Counters

- Not collected in this run.

## Outcome

- Status: `pass`
- Blocking issue(s): `none`
- Next action: `continue W2 by migrating decode boundary payload ownership from host vectors toward device-backed wrappers`

# Parity Gap Run Report

- `run_id`: `20260227T213142Z_w1_config_hardening`
- `date_utc`: `2026-02-27`
- `git_ref`: `dirty_worktree_uncommitted`
- `dirty_worktree`: `true`

## Scope

- Workstream(s): `W1`
- Goal: Remove parity-adjacent env toggles from staged/runtime debug behavior and replace with typed run-config controls.
- Backend: `N/A` (build + guard validation only)
- Input(s): `N/A`

## Command(s)

```bash
rg -n "env_flag\\(|TRELLIS2_STAGE_DEBUG|TRELLIS2_DECODER_CONV_TELEMETRY" \
  crates/burn_trellis/src/staged_pipeline_runtime_helpers.rs \
  crates/burn_trellis/src/staged_pipeline.rs \
  crates/burn_trellis/src/pipeline.rs \
  crates/burn_trellis/tool/trellis2_run.rs
./scripts/guard_canonical_runtime.sh
timeout 180s cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run
timeout 240s cargo check -p burn_synth
```

## Invariant Summary

- Canonical WGPU fail-fast only: `not validated in this run`
- Pre-extraction host readbacks: `not measured in this run`
- Decode dispatch presence: `not measured in this run`
- Runtime source identity: `not measured in this run`
- Typed runtime debug controls in staged/runtime path: `pass`
- Guardrail baseline lock: `pass`
- Downstream `TrellisRunOptions` call-sites (`burn_synth`) compile: `pass`

## Timings (ms)

- Not collected in this run.

## Kernel / Telemetry Counters

- Not collected in this run.

## Outcome

- Status: `pass`
- Blocking issue(s): `none`
- Notes:
  - canonical guard scope was expanded to include staged pipeline runtime files.
  - updated baseline counts: `.into_data(` = `22`, `std::env::var(` = `5` in guarded modules.
- Next action: `continue W1 parity-critical env/hook cleanup and proceed to W2 device-first sparse ownership migration`

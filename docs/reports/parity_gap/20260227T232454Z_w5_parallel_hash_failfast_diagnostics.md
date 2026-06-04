# Parity Gap Run Report

- `run_id`: `20260227T232454Z_w5_parallel_hash_failfast_diagnostics`
- `date_utc`: `2026-02-27`
- `git_ref`: `dirty_worktree_uncommitted`
- `dirty_worktree`: `true`

## Scope

- Workstream(s): `W5`
- Goal: add fail-fast overflow diagnostics to the new parallel hash insertion path so probe exhaustion cannot silently corrupt neighbor maps.
- Backend: `wgpu-kernel` + downstream `runtime-model-wgpu`
- Input(s): `compile + hash/scan parity unit slice`

## Command(s)

```bash
cargo fmt --all
cargo check -p burn_flex_gmm --features wgpu-kernel
cargo test -p burn_flex_gmm --features wgpu-kernel neighbor_rows_device_hash_matches_scan -- --nocapture
cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run
./scripts/guard_canonical_runtime.sh
```

## Invariant Summary

- Canonical WGPU fail-fast only: `pass`
- Neighbor-hash insertion overflow now explicitly counted on device (`build_fail_count`): `pass`
- Overflow/probe-exhaustion now returns hard error with table parameters: `pass`
- Hash/scan parity (including duplicate-coord guard): `pass`
- Canonical guard baseline lock: `pass`

## Timings (ms)

- Not collected in this run.

## Kernel / Telemetry Counters

- Not collected in this run.

## Outcome

- Status: `pass`
- Blocking issue(s): `probe-depth distribution telemetry and large-shape performance tuning remain`
- Next action: `add bounded probe-depth counters + stage benchmark harness for 512-equivalent workloads`

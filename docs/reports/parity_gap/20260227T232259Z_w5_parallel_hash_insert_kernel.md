# Parity Gap Run Report

- `run_id`: `20260227T232259Z_w5_parallel_hash_insert_kernel`
- `date_utc`: `2026-02-27`
- `git_ref`: `dirty_worktree_uncommitted`
- `dirty_worktree`: `true`

## Scope

- Workstream(s): `W5`
- Goal: replace serial/chunked neighbor-hash build with true parallel device insertion while preserving scan-equivalent neighbor-row semantics.
- Backend: `wgpu-kernel` + downstream `runtime-model-wgpu`
- Input(s): `compile + hash/scan parity unit slice`

## Command(s)

```bash
cargo check -p burn_flex_gmm --features wgpu-kernel
cargo test -p burn_flex_gmm --features wgpu-kernel neighbor_rows_device_hash_matches_scan -- --nocapture
cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run
./scripts/guard_canonical_runtime.sh
```

## Invariant Summary

- Canonical WGPU fail-fast only: `unchanged`
- Pre-extraction host readbacks: `unchanged`
- Hash build path in canonical device algo: `parallel insertion enabled`
- Hash build deterministic duplicate handling (lowest row retained): `pass`
- Hash/scan parity on test topology: `pass`
- Canonical guard baseline lock: `pass`

## Timings (ms)

- Not collected in this run.

## Kernel / Telemetry Counters

- Not collected in this run.

## Outcome

- Status: `pass`
- Blocking issue(s): `no overflow/probe diagnostics yet; no stage-level 512-equivalent perf evidence yet`
- Next action: `add hash probe/overflow telemetry and run bounded stage benchmarks before promoting W5 to complete`

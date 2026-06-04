# Parity Gap Run Report

- `run_id`: `20260227T234634Z_w5_sorted_hash_parallel_query`
- `date_utc`: `2026-02-27`
- `git_ref`: `dirty_worktree_uncommitted`
- `dirty_worktree`: `true`

## Scope

- Workstream(s): `W5`
- Goal: replace large-workload serial hash-table insertion bottleneck with a non-CAS parallel device path while preserving scan-equivalent neighbor semantics.
- Backend: `wgpu-kernel` + downstream `runtime-model-wgpu`
- Input(s): `large-workload hash-path parity slice + downstream compile`

## Implementation Summary

- Added non-CAS sorted-hash neighbor path for large workloads:
  - `neighbor_coord_hash_kernel` (device hash key generation)
  - GPU `sort_with_indices` over hash keys
  - `neighbor_rows_from_sorted_hash_kernel` (device binary-search query with bounded collision scan)
- Added new algo selector variant `SortedHash` and switched large-workload routing to this path.
- Kept existing hash telemetry counters; updated sorted path to emit hash-stage telemetry values.
- Added direct parity test against scan reference:
  - `neighbor_rows_sorted_hash_matches_scan_reference`

## Command(s)

```bash
cargo fmt --all
cargo check -p burn_flex_gmm --features wgpu-kernel
cargo test -p burn_flex_gmm --features wgpu-kernel neighbor_rows_sorted_hash_matches_scan_reference -- --nocapture
cargo test -p burn_flex_gmm --features wgpu-kernel neighbor_rows_hash_probe_telemetry_records_probe_stats -- --nocapture
cargo test -p burn_flex_gmm --features wgpu-kernel neighbor_rows_device_hash_matches_scan -- --nocapture
cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run
./scripts/guard_canonical_runtime.sh
```

## Invariant Summary

- Canonical WGPU fail-fast only: `pass`
- Hash-path correctness vs scan on large-work topology: `pass`
- Duplicate-coord hash parity guard: `pass`
- Canonical runtime guardrails: `pass`
- CAS insertion blocker status: `unchanged` (still upstream)

## Outcome

- Status: `pass`
- Blocking issue(s): `sorted-hash path needs stage-level tuning evidence at 512-equivalent workloads`
- Next action: `add bounded stage benchmark slice (neighbor-map build-only) and tune match-scan bound / dispatch choices`

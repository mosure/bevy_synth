# Parity Gap Run Report

- `run_id`: `20260227T233812Z_w5_hash_probe_telemetry_and_cas_blocker`
- `date_utc`: `2026-02-27`
- `git_ref`: `dirty_worktree_uncommitted`
- `dirty_worktree`: `true`

## Scope

- Workstream(s): `W5`
- Goal: add hash probe/failure telemetry to neighbor-map build and validate real hash-path execution.
- Backend: `wgpu-kernel` + downstream `runtime-model-wgpu`
- Input(s): `hash-path focused unit slice + downstream compile`

## Command(s)

```bash
cargo fmt --all
cargo check -p burn_flex_gmm --features wgpu-kernel
cargo test -p burn_flex_gmm --features wgpu-kernel neighbor_rows_hash_probe_telemetry_records_probe_stats -- --nocapture
cargo test -p burn_flex_gmm --features wgpu-kernel neighbor_rows_device_hash_matches_scan -- --nocapture
cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run
./scripts/guard_canonical_runtime.sh
```

## Invariant Summary

- Canonical WGPU fail-fast only: `pass`
- Hash build now emits probe diagnostics (`rows`, `probe_total`, `probe_max`, `fail_rows`): `pass`
- Stage telemetry includes hash probe diagnostics: `pass`
- Hash insertion exhaustion still fail-fast: `pass`
- Real parallel CAS insertion feasibility on this stack: `blocked`

## Blocker Details

- Attempting true parallel CAS insertion on this path still panics in `cubecl-spirv` with:
  - `Atomic should have a scope registered`
- To keep canonical runtime stable, hash insertion currently remains deterministic single-lane device execution (`neighbor_hash_build_serial_kernel`) with explicit probe/failure accounting.

## Outcome

- Status: `pass (with blocker)`
- Blocking issue(s): `upstream cubecl-spirv CAS atomic scope support for this kernel path`
- Next action: `implement alternate non-CAS parallel build design (sort/binsearch or multi-pass bucketing) while preserving deterministic semantics`

# Parity Gap Run Report

- `run_id`: `20260227T235824Z_w5_stage_bench_and_threshold_tuning`
- `date_utc`: `2026-02-27`
- `git_ref`: `dirty_worktree_uncommitted`
- `dirty_worktree`: `true`

## Scope

- Workstream(s): `W5`
- Goal: add a bounded neighbor-map stage benchmark harness and tune auto-routing threshold between `scan` and `sorted-hash` using short safe runs.
- Backend: `wgpu-kernel` + downstream `runtime-model-wgpu`
- Input(s): `stage-only neighbor-map build timings`

## Implementation Summary

- Added explicit algorithm-routing API for benchmark/debug without env toggles:
  - `NeighborDeviceAlgoPreference`
  - `neighbor_rows_tensor_from_coords_with_algo`
  - `neighbor_rows_tensor_from_coords_tensor_with_algo`
- Added stage-only benchmark binary:
  - `cargo run -p burn_flex_gmm --features wgpu-kernel --bin neighbor_stage_bench -- ...`
  - file: `crates/burn_flex_gmm/tool/neighbor_stage_bench.rs`
- Tuned auto routing to kernel-aware thresholds:
  - small kernels: `96_000`
  - medium kernels: `240_000`
  - large kernels: `520_000`
- Added routing regression test:
  - `neighbor_algo_auto_uses_kernel_aware_thresholds`

## Commands

```bash
cargo fmt --all
cargo check -p burn_flex_gmm --features wgpu-kernel
cargo test -p burn_flex_gmm --features wgpu-kernel neighbor_rows_sorted_hash_matches_scan_reference -- --nocapture
cargo test -p burn_flex_gmm --features wgpu-kernel neighbor_algo_auto_uses_kernel_aware_thresholds -- --nocapture
cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run
./scripts/guard_canonical_runtime.sh
```

Stage timing slices written under:

- `tmp/runs/20260227T235156Z_neighbor_stage_tune/`

## Bench Snapshot (warmed stage-only)

- `rows=2048, k=3`: `scan=0.465 ms`, `sorted-hash=0.718 ms`
- `rows=4096, k=3`: `scan=0.657 ms`, `sorted-hash=0.592 ms`
- `rows=512, k=9`: `scan=0.801 ms`, `sorted-hash=0.855 ms`
- `rows=1024, k=9`: `scan=2.935 ms`, `sorted-hash=1.460 ms`

Inference from these bounded runs:

- cross-over depends strongly on kernel volume; one global threshold regresses one regime.
- kernel-aware thresholding is required for stable auto behavior.

## Outcome

- Status: `pass`
- Blocking issue(s): `stage-only tuning complete for this slice; full 512-equivalent pipeline perf closure still pending`
- Next action: `W6 sparse conv hotspot closure with kernel-level dispatch/workgroup tuning and stage bench integration`

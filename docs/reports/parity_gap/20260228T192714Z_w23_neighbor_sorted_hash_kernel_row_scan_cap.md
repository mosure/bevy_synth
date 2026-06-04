# W23 Neighbor Sorted-Hash Kernel-Row Scan-Cap Tune

Date: 2026-02-28
Workstream: W23 (post-W22)

## Objective
Reduce decode-path neighbor-query overhead on canonical k3 sparse-conv topology while preserving strict parity/fail-fast behavior.

## Change Summary
- File: `crates/burn_flex_gmm/src/wgpu.rs`
- Updated sorted-hash match-scan cap from one global value to kernel-row-aware caps:
  - `kernel_rows <= 64` -> `8`
  - `kernel_rows <= 256` -> `16`
  - `kernel_rows > 256` -> `32`
- Kept sorted-hash binary-search loop at fixed 32 iterations.
  - Rationale comment added inline: runtime-param loop count regressed sorted-hash parity on CubeCL WGSL.
- Updated telemetry implications through existing `device_hash_probe_max` path (`32 + match_scan`), which now reports lower maxima for canonical k3 decode path.

## Validation

### Compile/tests/guards
- `cargo fmt --all`
- `cargo test -p burn_flex_gmm --features wgpu-kernel neighbor_rows_ -- --nocapture`
- `cargo test -p burn_flex_gmm --features wgpu-kernel neighbor_algo_auto_uses_kernel_aware_thresholds -- --nocapture`
- `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run`
- `cargo test -p burn_trellis --features runtime-model-wgpu canonical_wgpu_no_host_readback_before_extraction -- --nocapture`
- `./scripts/guard_canonical_runtime.sh`

All passed.

### Focused microbench evidence
Command matrix:
- `cargo run -p burn_flex_gmm --features wgpu-kernel --bin neighbor_stage_bench -- --rows <R> --kernel 3 --channels 32 --warmup 1 --iters 3 --algo <auto|sorted>`
- Run artifact: `tmp/runs/20260228T192714Z_neighbor_sorted_scan_tune/01_matrix.log`

Key observations:
- `rows=181381, kernel=3`
  - `sorted-hash mean_ms`: `22.696`
  - prior baseline (W22-era sample): `25.711` (run `tmp/runs/20260228T192107Z_neighbor_algo_matrix/01_matrix.log`)
  - improvement ~`11.7%` on this stage microbench sample.
- `hash_probe_max_max` on k3 cases reduced from `64` to `40` (reflecting `32 + 8` cap).

### Strict runtime sanity
Command:
- `cargo run -p burn_trellis --features runtime-model-wgpu --bin trellis2_run -- --input docs/input_chair.jpg --quality low --backend wgpu --strict-benchmark --require-runtime-model --repeat 2`
- Run artifact: `tmp/runs/20260228T192714Z_trellis2_wgpu_warm_after_neighbor_scan_tune/01_run.log`

Warm run (`run=2`) result:
- `status=ok`
- `timings_ms.total=34603.728315`
- `timings_ms.decode=17532.741464`
- `timings_ms.decode_shape_decoder=7829.612384`
- `timings_ms.decode_tex_decoder=7204.313483`
- `host_readback_count=0`
- `decode_shape_wgpu_dispatches=40`
- `decode_tex_wgpu_dispatches=40`

Reference comparison point (pre-W23 telemetry sample):
- `timings_ms.decode=18145.203712` from `tmp/runs/20260228T191808Z_trellis2_wgpu_warm_profile/01_run.log`.

## Notes
- A candidate dynamic binary-search-step optimization was prototyped and intentionally rolled back in this workstream after parity regression in `neighbor_rows_sorted_hash_matches_scan_reference`.
- Final committed state preserves parity and applies only the kernel-row-aware scan-cap tightening.

# W25 Neighbor Sorted-Hash Mid-Bucket Search-Step Tightening

Date: 2026-02-28
Workstream: W25 (post-W24)
Owner: `crates/burn_flex_gmm/src/wgpu.rs`

## Goal

Reduce sorted-hash query over-iteration on common decode row counts while preserving compile-time static loop bounds and parity behavior.

## Change Summary

Implemented compile-time mid-bucket search-step dispatch for sorted-hash query:

- Added new compile-time kernel variant:
  - `neighbor_rows_from_sorted_hash_kernel_18`
- Extended resolver buckets:
  - `rows <= 2^16 -> 16`
  - `2^16 < rows <= 2^18 -> 18` (new)
  - `2^18 < rows <= 2^24 -> 24`
  - `rows > 2^24 -> 32`
- Updated launch dispatch to route to the 18-step kernel.
- Added regression unit test:
  - `neighbor_sorted_hash_search_step_resolver_uses_mid_bucket`

Rationale comment retained in code:

- Runtime-gated loop bounds previously regressed parity on CubeCL/WGSL.
- Compile-time variants are kept to preserve deterministic parity behavior.

## Validation Commands

1. `cargo fmt --all`
2. `cargo test -p burn_flex_gmm --features wgpu-kernel neighbor_rows_ -- --nocapture`
3. `cargo test -p burn_flex_gmm --features wgpu-kernel neighbor_algo_auto_uses_kernel_aware_thresholds -- --nocapture`
4. `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run`
5. `cargo test -p burn_trellis --features runtime-model-wgpu canonical_wgpu_no_host_readback_before_extraction -- --nocapture`
6. `./scripts/guard_canonical_runtime.sh`

All passed.

## Benchmark Evidence

### A) Isolated neighbor-stage probe reduction (target row shape)

Run artifact:

- `tmp/runs/20260228T203350Z_neighbor_midbucket_probe_bench/01_matrix.log`

Case: `rows=181381`, `kernel=3`, `algo=sorted-hash`

- Previous (W24 reference):
  - `hash_probe_total_mean=117534888`
  - `hash_probe_max_max=32`
  - Source: `tmp/runs/20260228T203600Z_neighbor_sorted_search_steps_compiletime/01_matrix.log`
- W25:
  - `hash_probe_total_mean=88151166`
  - `hash_probe_max_max=26`

Probe work dropped by about 25% for this dominant row case.

### B) Strict runtime sanity with full telemetry

Run artifact:

- `tmp/runs/20260228T203413Z_trellis2_wgpu_w25_neighbor_midbucket_profile/01_run.log`

Warm run (`run=2`) key signals:

- `status=ok`
- `host_readback_count=0`
- `decode_shape_wgpu_dispatches=40`
- `decode_tex_wgpu_dispatches=40`
- `timings_ms.decode=19852.862071`
- `timings_ms.total=37362.694651`

Shape decoder neighbor telemetry (warm pass):

- `device_hash_ms=4018.79`
- `hash_probe_total=117528462`
- `hash_probe_avg=471.28`
- `hash_probe_max=26`

Reference (W24 warm profile):

- `hash_probe_total=146912184`
- `hash_probe_max=32`
- source: `tmp/runs/20260228T204400Z_trellis2_wgpu_warm_profile_sorted_hash_steps_dispatch/01_run.log`

## Interpretation

- W25 achieved the intended probe-count reduction and maintained parity/canonical invariants.
- Runtime decode gains are modest because sorted-hash wall time (`device_hash_ms`) remains dominated by sort/build cost, not binary-search step count.

## Next Step

Prioritize sorted-hash build overhead reduction (not further search-step tuning), likely via custom device hash-build/query kernels that avoid `sort_with_indices` as the dominant cost center.

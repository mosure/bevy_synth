# W24 Neighbor Sorted-Hash Compile-Time Search-Step Dispatch

Date: 2026-02-28
Workstream: W24 (post-W23)

## Objective
Reduce sorted-hash query work in canonical decode neighbor-map builds without reintroducing the known parity regression from runtime-gated binary-search loops.

## Change Summary
- File: `crates/burn_flex_gmm/src/wgpu.rs`
- Replaced single sorted-hash query kernel with compile-time-step variants:
  - `neighbor_rows_from_sorted_hash_kernel_16`
  - `neighbor_rows_from_sorted_hash_kernel_24`
  - `neighbor_rows_from_sorted_hash_kernel_32`
- Added host-side resolver `resolve_neighbor_sorted_hash_search_steps(rows)` to dispatch kernels by row count:
  - `rows <= 2^16` -> 16-step kernel
  - `rows <= 2^24` -> 24-step kernel
  - else -> 32-step kernel
- Kept loop trip-count static per kernel variant to avoid the prior CubeCL/WGSL parity issue with runtime-gated search loops.
- Updated hash-probe telemetry to use resolved search steps (`device_hash_probe_total`, `device_hash_probe_max`).

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
- `cargo run -p burn_flex_gmm --features wgpu-kernel --bin neighbor_stage_bench -- --rows 181381 --kernel <3|27> --channels 32 --warmup 1 --iters 3 --algo <sorted|auto>`
- Run artifact: `tmp/runs/20260228T203600Z_neighbor_sorted_search_steps_compiletime/01_matrix.log`

Key observations:
- `kernel=3, rows=181381`
  - `auto mean_ms`: `22.024`
  - `hash_probe_max_max`: `32`
- Synthetic `kernel=27, rows=181381` matrix entries exceeded available VRAM in this bounded bench harness (`can't allocate buffer of size: 14280488896`), so those cases are excluded from performance claims.

### Strict runtime sanity
Command:
- `cargo run -p burn_trellis --features runtime-model-wgpu --bin trellis2_run -- --input docs/input_chair.jpg --quality low --backend wgpu --strict-benchmark --require-runtime-model --repeat 2`
- Run artifact: `tmp/runs/20260228T203900Z_trellis2_wgpu_warm_after_sorted_hash_steps_dispatch/01_run.log`

Warm run (`run=2`) result:
- `status=ok`
- `timings_ms.total=37154.873541`
- `timings_ms.decode=20013.857931`
- `timings_ms.decode_shape_decoder=9011.237690`
- `timings_ms.decode_tex_decoder=8487.169403`
- `host_readback_count=0`

### Telemetry warm profile
Command:
- `cargo run -p burn_trellis --features runtime-model-wgpu --bin trellis2_run -- --input docs/input_chair.jpg --quality low --backend wgpu --strict-benchmark --require-runtime-model --repeat 2 --runtime-decoder-conv-telemetry`
- Run artifact: `tmp/runs/20260228T204400Z_trellis2_wgpu_warm_profile_sorted_hash_steps_dispatch/01_run.log`

Warm run (`run=2`) result:
- `status=ok`
- `timings_ms.total=33415.091887`
- `timings_ms.decode=18495.477512`
- `timings_ms.decode_shape_decoder=8244.537297`
- `timings_ms.decode_tex_decoder=7799.865237`
- `host_readback_count=0`
- Shape neighbor telemetry:
  - `hash_probe_total=146912184`
  - `hash_probe_avg=589.10`
  - `hash_probe_max=32`

## Notes
- This workstream intentionally preserves static loop trip-counts in WGSL kernels to maintain parity stability.
- Runtime-level gains remain variable across runs; the telemetry sample above is the strongest warm datapoint from this pass.
- Next likely hotspot remains neighbor map build/query wall-time (`device_hash_ms`) for shape decode at large row counts.

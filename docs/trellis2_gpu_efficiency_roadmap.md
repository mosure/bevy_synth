# Trellis2 GPU Efficiency Roadmap

This roadmap targets the current runtime-model WGPU path in `burn_trellis`, with strict numerical parity gates against the existing implementation.

## Baseline Findings

- Pipeline profiling with runtime-model WGPU and reduced sampler steps (`TRELLIS2_SAMPLER_STEPS_OVERRIDE=2`) shows the sparse stage dominating:
  - `stage sparse complete (103761 ms, coords=23728)` on `docs/output_chair_bg_removed.png`.
- Decoder sparse submanifold conv was loop-heavy and CPU-bound. A FlexGMM-style gather+GEMM implementation now exists in `burn_flex_gmm`.
- Micro-bench result for sparse submanifold convolution (`crates/burn_flex_gmm/benches/sparse_subm_conv.rs`):
  - `legacy`: ~`44.1 ms`
  - `flex_gemm`: ~`22.5 ms`
  - Speedup: ~`1.95x`

## Focus Areas (Priority Order)

1. Sparse flow runtime (`runtime_model/sparse_structure_flow.rs`)
- Why: This currently dominates end-to-end wall time.
- Immediate targets:
  - Reduce tensor clone churn in `predict_with_cfg_tensor`.
  - Fuse CFG arithmetic (`w*pos + (1-w)*neg`, optional rescale) into fewer backend ops.
  - Keep all sampler state updates on device and avoid accidental host sync points.

2. Decoder sparse conv path (`runtime_model/sparse_decoder.rs`)
- Why: Core decode kernel path was previously nested-loop scalar math.
- Current status:
  - Added `burn_flex_gmm` crate.
  - Added `TRELLIS2_DECODER_CONV_IMPL=legacy|flex_gmm`.
  - Decoder now tries flex path first and falls back to legacy on error.
- Next targets:
  - Add neighbor-map cache keyed by `(coords_hash, kernel_shape, axis_order/sign)` to avoid repeated map/gather setup.
  - Add grouped weight packing cache to avoid per-call pack rebuild.

3. GPU custom kernels (WGPU/WGSL)
- Why: 100% utilization + low power typically indicates memory-bound dispatch patterns and launch overhead.
- Candidate kernels:
  - Sparse gather + grouped GEMM + bias (single dispatch path per group or tiled multi-group path).
  - Fused per-row layernorm + affine + SiLU for decoder/flow MLP blocks.
  - Optional fused Euler update kernel (`x_t -= dt * v`) over 5D latent tensor.

## Numerical Guardrails

All fused/custom paths must pass parity tests against the existing path:

- Unit parity:
  - `burn_flex_gmm` compares `flex_gemm` vs legacy with tolerance `<= 1e-5`.
  - `burn_trellis` decoder parity test: `sparse_conv_flex_matches_legacy_path`.
- Hook parity:
  - Continue using e2e hook alignment tests for strict stage checks.
- Release gates:
  - `mean_abs`, `rmse`, `max_abs` checks for critical decode/sampler tensors.

## Benchmark Plan

1. Micro benches (already enabled)
- `cargo bench -p burn_flex_gmm --bench sparse_subm_conv -- --sample-size 20`

2. Decoder-only benchmark (next)
- Add bench fixture loading a fixed decode hook snapshot.
- Compare `TRELLIS2_DECODER_CONV_IMPL=legacy` vs `flex_gmm` on identical decode inputs.

3. Stage-level runtime benchmark (next)
- `trellis2_run` with fixed seed + deterministic overrides.
- Capture stage timing and host transfer counters before/after each optimization.

## Implementation Status

- Done:
  - New crate: `crates/burn_flex_gmm`
  - FlexGMM-style sparse conv kernel + legacy reference kernel
  - Unit tests + criterion bench
  - Decoder integration in `burn_trellis` with fallback
  - Runtime decoder WGPU path now supports coord-driven neighbor tensor build
    (`TRELLIS2_DECODER_WGPU_NEIGHBOR_SOURCE=coords|rows`, default `coords`).
  - Decoder WGPU memory guardrails tightened to reduce pathological buffer growth:
    - `TRELLIS2_DECODER_WGPU_MAX_OUTPUT_BYTES` default `256MB`
    - `TRELLIS2_DECODER_WGPU_MAX_INPUT_BYTES` default `1GB`
    - `BURN_FLEX_GMM_WGPU_SPLIT_K_MAX_PARTIAL_BYTES` default `256MB`
  - Stage-runtime cache added in `Trellis2Pipeline` so repeated in-process samples reuse loaded runtimes.
    - Disable with `TRELLIS2_DISABLE_STAGE_RUNTIME_CACHE=1`.
  - Decoder conv telemetry added:
    - JSON timings now include conv call/dispatch/chunk/byte counters per shape/tex decode.
    - Optional per-block logging via `TRELLIS2_DECODER_CONV_TELEMETRY=1`.
- In progress:
  - Stage-level profiling and sparse-flow fusion work
  - Decoder packing/cache improvements
- Pending:
  - WGSL custom kernels for sparse gather+GEMM and fused normalization/activation
  - Full e2e throughput comparison against Python TRELLIS2 baseline

## Latest Bench Snapshot (Medium, WGPU, strict, steps=1, dense_res=16)

- Full (`TRELLIS2_SKIP_PBR=0`, max_children=2): `~757.06s`
- Mesh-only (`TRELLIS2_SKIP_PBR=1`, max_children=2): `~577.29s`
- Implied PBR stage overhead: `~179.77s`

Runtime cache validation (`--repeat 2`, mesh-only, same settings):
- Run 1 `runtime_setup_ms`: `~179737.97`
- Run 2 `runtime_setup_ms`: `~0.025`
- Run 2 total: `~399.43s` (setup amortized; still decode-dominated)

## Post-clean Bench Snapshot (Medium, WGPU, strict, steps=1, dense_res=16)

- Mesh-only (`TRELLIS2_SKIP_PBR=1`, max_children=2): `~610.21s`
- Full (`TRELLIS2_SKIP_PBR=0`, max_children=2): `~796.92s`
- Implied full-pipeline PBR overhead (`full - mesh`): `~186.71s`
- Full decode PBR substage (`decode_pbr_ms`): `~183.18s`

Compared with the prior snapshot above:
- Mesh-only: `+32.92s` (`+5.70%`)
- Full: `+39.86s` (`+5.27%`)

## Phase A Results (Device-Resident ConvNeXt Math)

Implemented in `runtime_model/sparse_decoder.rs`:
- Added WGPU tensor-math path for ConvNeXt decoder blocks (conv + layernorm + MLP + residual) with CPU fallback.
- Added WGPU tensor caches for linear weights/biases and norm vectors.
- Added controls:
  - `TRELLIS2_DECODER_WGPU_DEVICE_MATH` (default on)
  - `TRELLIS2_DECODER_WGPU_DEVICE_MATH_FP16` (default on)

Strict benchmark measurements on `docs/output_chair_bg_removed.png`:

1. Mesh-only, dense_res=16, max_children=2
- Baseline (`device_math=0`): `616.66s` total, `276.56s` decode, `decode_*_conv_calls=40`
- Phase A (`device_math=1`): `418.96s` total, `118.39s` decode, `decode_*_conv_calls=8`
- Delta: total `-197.70s` (`-32.1%`), decode `-158.17s` (`-57.2%`)

2. Full pipeline, dense_res=16, max_children=2
- Phase A (`device_math=1`): `578.58s` total, `272.36s` decode, `170.06s` decode_pbr

3. Full pipeline, strict-safe tuned settings
- Settings: `steps=1`, `dense_res=12`, `max_children=2`, `TRELLIS2_MAX_SPARSE_COORDS=16384`, `device_math=1`
- Cold run: `397.39s` (`6.62 min`)
- Warm run (`--repeat 2`, run 2): `195.97s` (`3.27 min`)
- Result: warm in-process throughput is below `5 min/sample`; cold-start is still above due runtime setup cost (`~181s`)

Notes:
- `max_children=1` is not strict-safe for this sample (`fallback_empty_mesh`), so it is not a valid “best” setting.
- Remaining cold-start gap is now dominated by runtime setup + sparse stage + PBR.

## Phase A.1 Results (Neighbor Cache Keying / Stall Reduction)

Implemented in `crates/burn_flex_gmm/src/wgpu.rs`:
- Neighbor tensor cache key now depends only on topology-relevant fields:
  - `(kernel_d, kernel_h, kernel_w, axis_order, axis_sign, coords_hash, rows, backend, device)`
- Removed channel/group fields from the key (`in/out channels`, `groups`, `channels_per_group`) so
  multiple conv layers with identical sparse topology reuse the same cached neighbor tensor.
- Added regression test:
  - `neighbor_rows_cache_reuses_across_channel_variants_with_same_topology`

Strict benchmark re-run (same tuned settings):
- `steps=1`, `dense_res=12`, `max_children=2`, `TRELLIS2_MAX_SPARSE_COORDS=16384`, `device_math=1`

Repeat 2 run (before cache-key fix):
- Warm run (run 2): `195.97s`
- Cold run (run 1): `390.18s`

Repeat 2 runs (after cache-key fix):
- Warm run (run 2): `186.58s`
- Warm run (run 2, confirm): `186.94s`
- Cold run (run 1): `385.58s`
- Cold run (run 1, confirm): `383.23s`

Observed impact:
- Warm steady-state improvement: ~`-4.6%` to `-4.8%` vs pre-fix warm baseline.
- Cold one-shot improvement: ~`-1.2%` to `-1.8%` vs pre-fix cold baseline.
- Host readbacks remain `3` (`~1.08M` elements) in this strict path; remaining bottlenecks are
  runtime setup + sparse flow + decode/PBR compute.

## Phase A.2 Results (Device Hash Build + Sparse Stall Tuning)

Implemented in `crates/burn_flex_gmm/src/wgpu.rs`:
- Added WGSL device-side hash-table build mode for neighbor mapping:
  - `BURN_FLEX_GMM_WGPU_NEIGHBOR_HASH_BUILD=auto|host|wgsl`
- Added probe-loop control:
  - `BURN_FLEX_GMM_WGPU_NEIGHBOR_HASH_MAX_PROBE=<N>`
- Added WGSL overflow-guard readback: if serial device hash build exceeds `max_probe`,
  return error and fall back to host table build in `Auto`.
- Default probe cap is now `128` (configurable via env) to avoid pathological loop stalls.
- `Auto` backend now prefers host neighbor build for large workloads; explicit `wgsl` override remains available.

Validation:
- `cargo test -p burn_flex_gmm --features wgpu-kernel neighbor_rows_device_hash_matches_scan -- --nocapture`
- `cargo test -p burn_flex_gmm --features wgpu-kernel neighbor_rows_device_backend_matches_host_backend -- --nocapture`
- `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run`

Strict stage-level benchmark split (`steps=1`, `dense_res=12`, `max_children=2`, `max_sparse_coords=16384`,
`device_math=1`, `BURN_FLEX_GMM_WGPU_NEIGHBOR_HASH_MAX_PROBE=128`):

1. Mesh-only (`TRELLIS2_SKIP_PBR=1`)
- WGSL hash build warm run: `125.15s` (`2.09 min`)
  - sparse: `55.81s`
  - decode: `49.26s`
- Host hash build warm run: `128.32s` (`2.14 min`)
  - sparse: `58.59s`
  - decode: `49.21s`
- Observed: WGSL path is slightly faster for mesh-only warm steady-state in this setting.

2. Full pipeline (`TRELLIS2_SKIP_PBR=0`)
- WGSL hash build warm run: `197.05s` (`3.28 min`)
  - sparse: `53.82s`
  - decode: `124.27s`
  - decode_pbr: `74.86s`
- Host hash build warm run: `196.69s` (`3.28 min`)
  - sparse: `52.78s`
  - decode: `124.62s`
  - decode_pbr: `75.53s`
- Observed: full-pipeline warm throughput is effectively parity between host and WGSL hash build;
  PBR + decoder remain the dominant wall-time contributors.

3. Default mode (no `BURN_FLEX_GMM_WGPU_NEIGHBOR_*` env overrides)
- Full warm run: `199.88s` (`3.33 min`)
  - sparse: `53.61s`
  - decode: `127.05s`
  - decode_pbr: `77.54s`
- Mesh-only warm run: `121.52s` (`2.03 min`)
  - sparse: `52.69s`
  - decode: `49.50s`
- Implied PBR overhead (`full - mesh`): `~78.36s` (matches `decode_pbr` scale).

Result:
- Medium quality + `device=wgpu` full strict warm throughput is below target (`<5 min/sample`).
- Cold-start remains dominated by runtime setup (`~215s-234s`) and is still above target.

## Phase A.3 Results (Runtime Setup Load-Time Reduction)

Implemented:
- Added lazy runtime-model loading in `TrellisStageRuntime`:
  - `sparse_flow`, `shape_flow`, `tex_flow`, `shape_decoder`, `tex_decoder` now initialize on first use.
- Added env toggle:
  - `TRELLIS2_RUNTIME_LAZY_MODEL_LOAD=1|0` (default `1`).
- Added stage-runtime cache-key awareness for lazy/eager mode in `Trellis2Pipeline`.

Why:
- `runtime_setup_ms` previously included eager model load/deserialization for all runtime components.
- This dominated cold-start by ~`200s+` on medium WGPU runs.

Strict benchmark comparison (`steps=1`, `dense_res=12`, `max_children=2`, `max_sparse_coords=16384`, `device_math=1`):

1. Full pipeline, lazy load enabled (default)
- Run 1 `runtime_setup_ms`: `~0.46 ms`
- Run 2 `runtime_setup_ms`: `~0.02 ms`
- Run 2 total: `~209.85s` (`3.50 min`)

2. Full pipeline, eager load control (`TRELLIS2_RUNTIME_LAZY_MODEL_LOAD=0`)
- Run 1 `runtime_setup_ms`: `~215969.13 ms` (`~216s`)
- Run 2 `runtime_setup_ms`: `~0.02 ms`
- Run 2 total: `~213.39s` (`3.56 min`)

3. Mesh-only, lazy load enabled
- Run 1 `runtime_setup_ms`: `~0.47 ms`
- Run 2 `runtime_setup_ms`: `~0.02 ms`
- Run 2 total: `~130.90s` (`2.18 min`)

Outcome:
- Runtime setup target (`<30s`) is met by a wide margin under default lazy mode.
- End-to-end warm throughput remains in the same band as eager mode.

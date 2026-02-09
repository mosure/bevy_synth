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
- In progress:
  - Stage-level profiling and sparse-flow fusion work
  - Decoder packing/cache improvements
- Pending:
  - WGSL custom kernels for sparse gather+GEMM and fused normalization/activation
  - Full e2e throughput comparison against Python TRELLIS2 baseline

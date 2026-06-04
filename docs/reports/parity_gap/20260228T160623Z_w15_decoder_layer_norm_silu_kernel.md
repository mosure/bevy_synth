# W15 Decoder fused layer-norm+SiLU custom kernel integration

Run id: `20260228T160623Z_w15_decoder_layer_norm_silu_kernel`
Date (UTC): 2026-02-28

## Scope

Fuse decoder `layer_norm -> SiLU` chains into a single custom WGPU kernel path to reduce dispatch and intermediate tensor traffic in decode upsample blocks.

## Implementation summary

1. Added fused kernel and wrapper in `crates/burn_flex_gmm/src/wgpu.rs`:
   - `layer_norm_affine_silu_kernel`
   - `layer_norm_affine_silu_forward_wgpu(input, weight, bias, eps)`
2. Reused existing per-row stats kernel:
   - `layer_norm_row_stats_kernel`
3. Added parity coverage:
   - `layer_norm_affine_silu_kernel_matches_reference`
4. Integrated into canonical decoder runtime:
   - Added `layer_norm_silu_wgpu(...)` helper in `crates/burn_trellis/src/runtime_model/sparse_decoder_wgpu_ops.rs`
   - Switched two hot upsample chains in `crates/burn_trellis/src/runtime_model/sparse_decoder_runtime_impl.rs` from:
     - `layer_norm_wgpu(...)` + `silu_wgpu(...)`
     to:
     - `layer_norm_silu_wgpu(...)`
5. Import bridge update:
   - `crates/burn_trellis/src/runtime_model/sparse_decoder.rs`

## Correctness validation

Executed and passed:

- `cargo fmt --all`
- `cargo test -p burn_flex_gmm --features wgpu-kernel layer_norm_affine_silu_kernel_matches_reference -- --nocapture`
- `cargo test -p burn_flex_gmm --features wgpu-kernel layer_norm_affine_kernel_matches_reference -- --nocapture`
- `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run`
- `cargo test -p burn_trellis --features runtime-model-wgpu canonical_wgpu_no_host_readback_before_extraction -- --nocapture`
- `./scripts/guard_canonical_runtime.sh`

## Runtime sanity (strict, single run)

Executed:

- `cargo run -p burn_trellis --features runtime-model-wgpu --bin trellis2_run -- --input docs/input_chair.jpg --quality low --backend wgpu --strict-benchmark --require-runtime-model --repeat 1`

Observed:

- status: `ok`
- `timings_ms.total`: `91701.436118`
- `timings_ms.decode_shape_decoder`: `19035.980035`
- `timings_ms.decode_tex_decoder`: `13932.658392`
- `timings_ms.host_readback_count`: `0`
- `timings_ms.host_readback_elements`: `0`
- dispatch invariants present:
  - `wgpu_shape_dispatches=40`
  - `wgpu_tex_dispatches=40`

## Notes

- This work preserves canonical strict fail-fast behavior and keeps decode fully device-resident pre-extraction.
- Single-run timings remain noisy; use bounded repeat runs for robust phase-level trend confirmation.

# W14 Decoder layer-norm affine custom kernel integration

Run id: `20260228T155521Z_w14_decoder_layer_norm_affine_kernel`
Date (UTC): 2026-02-28

## Scope

Replace decoder WGPU layer-norm tensor-op chains with a dedicated fused custom kernel path while preserving canonical strict fail-fast semantics and zero pre-extraction host readbacks.

## Implementation summary

1. Added new custom kernels in `crates/burn_flex_gmm/src/wgpu.rs`:
   - `layer_norm_row_stats_kernel` (row-wise mean/variance)
   - `layer_norm_affine_kernel` (normalize + affine)
2. Added wrapper:
   - `layer_norm_affine_forward_wgpu(input, weight, bias, eps)`
3. Updated decoder runtime integration in `crates/burn_trellis/src/runtime_model/sparse_decoder_wgpu_ops.rs`:
   - `layer_norm_wgpu` now routes to `layer_norm_affine_forward_wgpu`
   - no-affine call sites now pass tensor-native `ones`/`zeros` for weight/bias on-device
   - retains strict fail-fast error propagation
4. Added import bridge in:
   - `crates/burn_trellis/src/runtime_model/sparse_decoder.rs`
5. Added kernel parity coverage:
   - `layer_norm_affine_kernel_matches_reference`

## Correctness validation

Executed and passed:

- `cargo fmt --all`
- `cargo test -p burn_flex_gmm --features wgpu-kernel layer_norm_affine_kernel_matches_reference -- --nocapture`
- `cargo test -p burn_flex_gmm --features wgpu-kernel linear_skinny_kernel_matches_reference -- --nocapture`
- `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run`
- `cargo test -p burn_trellis --features runtime-model-wgpu canonical_wgpu_no_host_readback_before_extraction -- --nocapture`
- `./scripts/guard_canonical_runtime.sh`

## Runtime sanity (strict, single run)

Executed:

- `cargo run -p burn_trellis --features runtime-model-wgpu --bin trellis2_run -- --input docs/input_chair.jpg --quality low --backend wgpu --strict-benchmark --require-runtime-model --repeat 1`

Observed:

- status: `ok`
- `timings_ms.total`: `102965.288899`
- `timings_ms.decode_shape_decoder`: `19412.757648`
- `timings_ms.decode_tex_decoder`: `14479.818934`
- `timings_ms.host_readback_count`: `0`
- `timings_ms.host_readback_elements`: `0`
- decode dispatch invariants present:
  - `wgpu_shape_dispatches=40`
  - `wgpu_tex_dispatches=40`

## Notes

- This workstream keeps canonical runtime fully device-resident through decode boundaries.
- Single-run timing remains workload/cache sensitive; evaluate phase-level trends with bounded repeats for tuning decisions.

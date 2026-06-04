# W10 Sparse-flow RoPE custom kernel kickoff

Run id: `20260228T142200Z_w10_rope_rotate_kernel_integration`
Date (UTC): 2026-02-28

## Scope

Implement and integrate a first custom WGPU kernel in sparse-flow attention hot path:

- kernel: `rope_rotate_pairs_kernel`
- wrapper: `rope_rotate_pairs_wgpu`
- integration point: sparse-flow RoPE `rotate_pairs` in canonical runtime-model WGPU path

## Implementation summary

1. Added a CubeCL/WGPU RoPE pair-rotation kernel in `crates/burn_flex_gmm/src/wgpu.rs`.
2. Added wrapper `rope_rotate_pairs_wgpu(...)` with strict tensor-shape validation and fail-fast launch errors.
3. Integrated kernel into `crates/burn_trellis/src/runtime_model/sparse_structure_flow.rs` via RoPE bridge:
   - canonical raw WGPU runtime backend (`WgpuRuntimeBackend`) uses kernel path
   - kernel launch failure is fail-fast (`panic!`) in canonical WGPU mode
4. Added explicit bridge impl for fusion WGPU backend used by tests (`burn_wgpu::Wgpu<f32, i32, u32>`) that returns `None` (keeps generic test matrix compiling without changing canonical runtime path).
5. Fixed type/ABI issues during integration:
   - kernel args aligned to `Array<f32>` for `as_array_arg` launches
   - explicit const-generic tensor typing for primitive-to-tensor reconstruction
   - propagated trait bounds where sparse-flow model constructors/forward paths require RoPE bridge compatibility

## Validation

Executed and passed:

- `cargo fmt --all`
- `cargo test -p burn_flex_gmm --features wgpu-kernel rope_rotate_pairs_kernel_matches_reference -- --nocapture`
- `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run`
- `cargo test -p burn_trellis --features runtime-model-wgpu canonical_wgpu_no_host_readback_before_extraction -- --nocapture`

## Runtime sanity (strict, single run)

Executed:

- `cargo run -p burn_trellis --features runtime-model-wgpu --bin trellis2_run -- --input docs/input_chair.jpg --quality low --backend wgpu --strict-benchmark --require-runtime-model --repeat 1`

Observed:

- status: `ok`
- `timings_ms.total`: `92115.400612`
- `timings_ms.host_readback_count`: `0`
- `timings_ms.host_readback_elements`: `0`
- decode dispatch invariants present:
  - `wgpu_shape_dispatches=40`
  - `wgpu_tex_dispatches=40`

## Notes

- This is a kickoff kernel focused on RoPE rotation only.
- It does not yet cover the larger sparse-flow attention bottlenecks (QK matmul/softmax/V matmul fusion) or sparse-structure cap/select kernels.

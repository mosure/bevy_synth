# W12 Sparse-flow RoPE direct-coord kernel integration

Run id: `20260228T152922Z_w12_rope_direct_coords_kernel_integration`
Date (UTC): 2026-02-28

## Scope

Add a direct token-coordinate RoPE kernel path that avoids intermediate phase/cos/sin tensor materialization for canonical sparse-flow token-coordinate RoPE.

## Implementation summary

1. Added custom kernel `rope_rotate_pairs_coords_kernel` in `crates/burn_flex_gmm/src/wgpu.rs`.
2. Added wrapper `rope_rotate_pairs_from_coords_wgpu(...)`:
   - input: `x:[B,T,H,D]`, `coords:[T,3]`, `rope_freq:[f32;2]`
   - host-side pair layout precompute (`rope_pair_layout_params`) for axis/frequency mapping
   - single dispatch applies trig + pair rotation
3. Added parity test `rope_rotate_pairs_coords_kernel_matches_reference`.
4. Updated sparse-flow runtime bridge in `crates/burn_trellis/src/runtime_model/sparse_structure_flow.rs`:
   - trait bridge now supports direct coord rotation (`rotate_coords`)
   - canonical token-coordinate path attempts direct-coord kernel first (`maybe_rotate_pairs_coords_wgpu`)
   - retains existing non-WGPU fallback tensor-op path for generic test backends

## Validation

Executed and passed:

- `cargo fmt --all`
- `cargo test -p burn_flex_gmm --features wgpu-kernel rope_rotate_pairs_ -- --nocapture`
- `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run`
- `cargo test -p burn_trellis --features runtime-model-wgpu canonical_wgpu_no_host_readback_before_extraction -- --nocapture`

## Runtime sanity (strict, single run)

Executed:

- `cargo run -p burn_trellis --features runtime-model-wgpu --bin trellis2_run -- --input docs/input_chair.jpg --quality low --backend wgpu --strict-benchmark --require-runtime-model --repeat 1`

Observed:

- status: `ok`
- `timings_ms.total`: `95015.623177`
- `timings_ms.host_readback_count`: `0`
- `timings_ms.host_readback_elements`: `0`
- decode dispatch invariants present:
  - `wgpu_shape_dispatches=40`
  - `wgpu_tex_dispatches=40`

## Notes

- This keeps canonical behavior fail-fast on WGPU kernel errors.
- Single-run timing is still variable run-to-run; attribution should use bounded warm stage benches.

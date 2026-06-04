# W11 Sparse-flow RoPE phase-kernel integration

Run id: `20260228T151953Z_w11_rope_phase_kernel_integration`
Date (UTC): 2026-02-28

## Scope

Add a second custom WGPU RoPE kernel path that consumes phase directly (`[tokens,pairs]`) to remove separate `cos` / `sin` tensor materialization in sparse-flow token-coordinate RoPE.

## Implementation summary

1. Added kernel `rope_rotate_pairs_phase_kernel` in `crates/burn_flex_gmm/src/wgpu.rs`.
2. Added wrapper `rope_rotate_pairs_from_phase_wgpu(...)` with strict shape checks and fail-fast launch error handling.
3. Added kernel parity test `rope_rotate_pairs_phase_kernel_matches_reference`.
4. Refactored sparse-flow RoPE path in `crates/burn_trellis/src/runtime_model/sparse_structure_flow.rs`:
   - added phase helper `rope_phase_from_coord_tensor(...)`
   - added phase bridge method on `RopeRotateWgpuBridge`
   - token-coordinate path now attempts custom phase-kernel first on canonical WGPU, then falls back to tensor-op path for non-WGPU bridge types.

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
- `timings_ms.total`: `107697.445244`
- `timings_ms.host_readback_count`: `0`
- `timings_ms.host_readback_elements`: `0`
- decode dispatch invariants present:
  - `wgpu_shape_dispatches=40`
  - `wgpu_tex_dispatches=40`

## Notes

- This change is a kernel-path simplification for RoPE token-coordinate handling.
- Single strict run timing remained variable vs earlier runs; stage-level benchmarking should use bounded warm runs for attribution.

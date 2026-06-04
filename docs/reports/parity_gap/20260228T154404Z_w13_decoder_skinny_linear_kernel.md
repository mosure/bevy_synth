# W13 Decoder skinny-linear custom kernel integration

Run id: `20260228T154404Z_w13_decoder_skinny_linear_kernel`
Date (UTC): 2026-02-28

## Scope

Replace decoder WGPU large-row tiny-output linear fallback math with a dedicated one-dispatch custom kernel, while preserving strict fail-fast canonical runtime semantics.

## Implementation summary

1. Added custom kernel and wrapper in `crates/burn_flex_gmm/src/wgpu.rs`:
   - `linear_skinny_kernel`
   - `linear_skinny_forward_wgpu(input, weight, bias)`
2. Kernel contract:
   - input shape: `[rows, in_channels]`
   - weight shape: `[out_channels, in_channels]`
   - bias shape: `[out_channels]`
   - output shape: `[rows, out_channels]`
   - computes `output[row, out] = bias[out] + dot(input[row, :], weight[out, :])`
3. Integrated canonical decoder path in `crates/burn_trellis/src/runtime_model/sparse_decoder_wgpu_ops.rs`:
   - Existing skinny branch trigger preserved (`out_channels <= 8 && rows >= 32_768`)
   - Replaced multi-pass per-column reduction tensor ops with `linear_skinny_forward_wgpu`
   - Added inline rationale comment explaining why the dedicated kernel path is required
4. Export wiring:
   - Added import in `crates/burn_trellis/src/runtime_model/sparse_decoder.rs`

## Correctness validation

Executed and passed:

- `cargo fmt --all`
- `cargo test -p burn_flex_gmm --features wgpu-kernel linear_skinny_kernel_matches_reference -- --nocapture`
- `cargo test -p burn_flex_gmm --features wgpu-kernel rope_rotate_pairs_ -- --nocapture`
- `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run`
- `cargo test -p burn_trellis --features runtime-model-wgpu canonical_wgpu_no_host_readback_before_extraction -- --nocapture`
- `./scripts/guard_canonical_runtime.sh`

## Runtime sanity (strict, single run)

Executed:

- `cargo run -p burn_trellis --features runtime-model-wgpu --bin trellis2_run -- --input docs/input_chair.jpg --quality low --backend wgpu --strict-benchmark --require-runtime-model --repeat 1`

Observed:

- status: `ok`
- `timings_ms.total`: `109802.353864`
- `timings_ms.decode_shape_decoder`: `19976.733657`
- `timings_ms.decode_tex_decoder`: `14753.269637`
- `timings_ms.host_readback_count`: `0`
- `timings_ms.host_readback_elements`: `0`
- dispatch invariants present:
  - `wgpu_shape_dispatches=40`
  - `wgpu_tex_dispatches=40`

## Notes

- This workstream is correctness-preserving and residency-preserving; no host-readback surfaces were added.
- Stage/runtime timing remains sensitive to cache/warmup and scene occupancy; compare phase deltas on repeated bounded runs for trend decisions.

# W27 Sparse-Flow MLP Small-Token Unchunked Policy (Non-Fusion WGPU)

Date: 2026-02-28
Workstream: W27 (post-W26)
Owner: `crates/burn_trellis/src/runtime_model/sparse_structure_flow.rs`

## Goal

Reduce sparse-flow stage overhead on common small/mid token counts by removing avoidable MLP chunking overhead in canonical non-fusion WGPU mode.

## Change Summary

Updated sparse-flow backend-aware MLP chunk policy:

- Function: `sparse_flow_mlp_chunk_tokens_for_backend::<B>(tokens)`
- New behavior for non-fusion WGPU backend:
  - if `tokens <= 8192`, return `tokens` (unchunked)
- Existing larger-token policy remains unchanged:
  - `tokens >= 65536` -> 8192
  - `tokens >= 16384` -> 4096

Added test coverage updates in `sparse_flow_backend_chunk_tokens_wgpu_use_wider_chunks`:

- `tokens=4096 -> 4096`
- `tokens=8192 -> 8192`
- existing `tokens=32768 -> 4096` remains asserted

Rationale comment retained in code:

- On native non-fusion WGPU sparse-flow path, small/mid token MLP chunking introduces unnecessary per-chunk matmul/concat overhead; unchunked execution is preferred while keeping large-token safeguards.

## Validation Commands

1. `cargo fmt --all`
2. `cargo test -p burn_trellis --features runtime-model-wgpu sparse_flow_backend_chunk_tokens_wgpu_use_wider_chunks -- --nocapture`
3. `cargo test -p burn_trellis --features runtime-model-wgpu canonical_wgpu_no_host_readback_before_extraction -- --nocapture`
4. `./scripts/guard_canonical_runtime.sh`
5. `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run`

All passed.

## Runtime Evidence

Primary artifact (strict low-quality repeat, canonical comparison mode):

- `tmp/runs/20260228T233900Z_trellis2_w27_sparseflow_mlp_unchunked_small_tokens_rerun/01_run.log`

Command:

- `cargo run -p burn_trellis --features runtime-model-wgpu --bin trellis2_run -- --input docs/input_chair.jpg --quality low --backend wgpu --strict-benchmark --require-runtime-model --repeat 2`

Result:

- `status=ok`
- `host_readback_count=0`
- dispatch invariants: `decode_shape_wgpu_dispatches=40`, `decode_tex_wgpu_dispatches=40`

Warm run (`run=2`) key timings:

- `timings_ms.total=33461.681450`
- `timings_ms.sparse=3744.439305`
- `timings_ms.shape_slat=6941.085066`
- `timings_ms.tex_slat=3537.363554`
- `timings_ms.decode=18954.242802`

## Comparison vs prior low-quality warm baseline (W22)

Reference baseline:

- `docs/reports/parity_gap/20260228T191634Z_w22_decode_end_stage_fence_repeat_stability.md`
- warm run: `total=41122.508622`, `sparse=5276.209279`, `shape_slat=9491.068676`, `tex_slat=4891.551258`, `decode=21155.844158`

Observed deltas (W27 warm - W22 warm):

- `total`: `-7660.83 ms` (improved)
- `sparse`: `-1531.77 ms` (improved)
- `shape_slat`: `-2549.98 ms` (improved)
- `tex_slat`: `-1354.19 ms` (improved)
- `decode`: `-2201.60 ms` (improved)

Interpretation:

- The MLP unchunking policy materially improves sparse-flow and end-to-end warm low-quality latency in canonical strict mode on this run profile.

## Additional run note

- `tmp/runs/20260228T232600Z_trellis2_w27_sparseflow_mlp_unchunked_small_tokens/01_run.log` was captured with `--runtime-decoder-conv-telemetry` enabled and is retained as supplementary evidence only.

## Next Step

Proceed with tex_slat-focused sparse-flow tuning (attention/linear path) under bounded low-quality repeat runs, then validate high-quality strict run impact once low-quality stage behavior stabilizes.

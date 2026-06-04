# W17 Sparse-flow backend-aware MLP/linear chunk tuning (non-fusion WGPU)

Run id: `20260228T162211Z_w17_sparse_flow_chunk_tuning`
Date (UTC): 2026-02-28

## Scope

Reduce sparse-flow stage overhead on canonical non-fusion WGPU by widening chunk sizes for MLP and token-linear paths while preserving strict fail-fast/device-resident behavior.

## Implementation summary

1. Updated `crates/burn_trellis/src/runtime_model/sparse_structure_flow.rs`:
   - Added `attention_uses_non_fusion_module_kernel::<B>()`.
   - Switched non-fusion module-attention checks in self/cross attention chunk planners to this helper.
   - Added backend-aware linear chunk policy:
     - `sparse_flow_linear_chunk_tokens_for_backend::<B>()`
     - non-fusion WGPU @ `tokens>=16384`: `16_384`.
   - Added backend-aware MLP chunk policy:
     - `sparse_flow_mlp_chunk_tokens_for_backend::<B>()`
     - non-fusion WGPU @ `tokens>=16384`: `4_096`
     - non-fusion WGPU @ `tokens>=65536`: `8_192`.
   - Updated MLP sync cadence for non-fusion WGPU:
     - `sparse_flow_mlp_sync_interval_for_backend::<B>()` -> `16`.
   - Wired sparse-flow entry/exit linears and feed-forward path to backend-aware chunk functions.
2. Added targeted policy tests:
   - `sparse_flow_backend_chunk_tokens_cpu_match_defaults`
   - `sparse_flow_backend_chunk_tokens_wgpu_use_wider_chunks`

## Correctness validation

Executed and passed:

- `cargo fmt --all`
- `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run`
- `cargo test -p burn_trellis --features runtime-model-wgpu sparse_flow_backend_chunk_tokens_ -- --nocapture`
- `cargo test -p burn_trellis --features runtime-model-wgpu canonical_wgpu_no_host_readback_before_extraction -- --nocapture`
- `./scripts/guard_canonical_runtime.sh`

## Runtime sanity (strict, repeat=2)

Executed:

- `cargo run -p burn_trellis --features runtime-model-wgpu --bin trellis2_run -- --input docs/input_chair.jpg --quality low --backend wgpu --strict-benchmark --require-runtime-model --repeat 2`

Observed:

- status: `ok`
- pre-extraction readbacks: `host_readback_count=0`, `host_readback_elements=0`
- decode dispatch invariants:
  - `decode_shape_wgpu_dispatches=40`
  - `decode_tex_wgpu_dispatches=40`

Warm pass (`runs[1]`, run=2):

- `timings_ms.total`: `46554.815447`
- `timings_ms.sparse`: `5007.28142`
- `timings_ms.shape_slat`: `8933.690586`
- `timings_ms.tex_slat`: `4398.788883`
- `timings_ms.decode`: `27919.192004`
- `timings_ms.decode_shape_decoder`: `12740.112751`
- `timings_ms.decode_tex_decoder`: `12662.562701`
- mesh stats: `vertices=181381`, `faces=460904`

Sparse-flow operation telemetry (warm pass):

- `sparse_runtime`: `self_attn_ms=828.02`, `cross_attn_ms=717.85`, `mlp_ms=846.43`
- `shape_slat`: `self_attn_ms=1322.51`, `cross_attn_ms=2173.49`, `mlp_ms=3704.63`
- `tex_slat`: `self_attn_ms=1261.92`, `cross_attn_ms=444.25`, `mlp_ms=1778.66`

## Notes

- This pass is heuristic tuning only; no rescue/fallback branches were added.
- For strict before/after comparability, use explicit fixed seeds when comparing to older reports because output occupancy can vary across random initializations.

# W16 Sparse-flow module-attention chunk-plan relaxation (non-fusion WGPU)

Run id: `20260228T161355Z_w16_sparse_flow_chunk_relax`
Date (UTC): 2026-02-28

## Scope

Relax sparse-flow attention chunk planning for canonical non-fusion WGPU (`CubeBackend`) so module-attention stays on flash-kernel-friendly chunk sizes without dense-logits budget downscaling.

## Implementation summary

1. Updated `crates/burn_trellis/src/runtime_model/sparse_structure_flow.rs`:
   - Self-attention streamed path:
     - for non-fusion module attention, set `query_chunk_tokens` and `kv_chunk_tokens` directly from `sparse_flow_module_attention_chunk_cap(tokens)`.
   - Cross-attention chunked path:
     - skip conservative `sparse_flow_module_attention_query_chunk_cap(...)` clamp for non-fusion module attention.
2. Kept conservative logits-budget clamping behavior for fusion backends only.
3. Preserved fail-fast semantics and existing token-cap guards.

## Correctness validation

Executed and passed:

- `cargo fmt --all`
- `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run`
- `cargo test -p burn_trellis --features runtime-model-wgpu canonical_wgpu_no_host_readback_before_extraction -- --nocapture`
- `./scripts/guard_canonical_runtime.sh`

## Runtime sanity (strict, single run)

Executed:

- `cargo run -p burn_trellis --features runtime-model-wgpu --bin trellis2_run -- --input docs/input_chair.jpg --quality low --backend wgpu --strict-benchmark --require-runtime-model --repeat 1`

Observed:

- status: `ok`
- `timings_ms.total`: `86361.579836`
- `timings_ms.shape_slat`: `10181.03252`
- `timings_ms.tex_slat`: `4479.207085`
- `timings_ms.decode_shape_decoder`: `18446.9867`
- `timings_ms.decode_tex_decoder`: `13990.430235`
- `timings_ms.host_readback_count`: `0`
- dispatch invariants present:
  - `wgpu_shape_dispatches=40`
  - `wgpu_tex_dispatches=40`

## Notes

- This pass targeted chunk-fragmentation overhead only; no fallback/rescue behavior was introduced.
- Canonical device-resident decode invariants remain enforced.

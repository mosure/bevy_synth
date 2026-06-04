# W28 Sparse-flow Prepared-Context Dead Path Cleanup

Run ID: `20260228T222118Z_w28_sparse_flow_prepared_path_cleanup`

## Summary

- Removed non-promoted prepared-context cache scaffolding from `crates/burn_trellis/src/runtime_model/sparse_structure_flow.rs`.
- Simplified block/model/runtime call surfaces to keep only active canonical cross-attention flow.
- Deleted dead `*_prepared` variants and `CrossAttentionPreparedContext` type that were not used by runtime.
- Kept canonical behavior unchanged (no fallback additions; fail-fast behavior preserved).

## Files changed

- `crates/burn_trellis/src/runtime_model/sparse_structure_flow.rs`

## Validation

1. `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run`
   - Result: pass
2. `cargo test -p burn_trellis --features runtime-model-wgpu sparse_flow_backend_chunk_tokens_wgpu_use_wider_chunks -- --nocapture`
   - Result: pass
3. `./scripts/guard_canonical_runtime.sh`
   - Result: pass

## Notes

- Direct `rustfmt` on this file failed due edition mismatch (`let chains` parse under non-2024 rustfmt invocation); compile/test validation remained green.

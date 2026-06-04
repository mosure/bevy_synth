# W29 CFG Batch-Pairing Experiment (Rejected)

Run ID: `20260228T223641Z_w29_cfg_batch_pairing_rejected`

## Goal

- Reduce sparse-flow CFG cost by evaluating positive/negative branches in one batched model forward.

## Experiment

- Implemented temporary CFG pairing in `sparse_structure_flow.rs`:
  - Concatenated `x_t` and `cond`/`neg_cond` along batch.
  - Performed single `predict_velocity_*` call.
  - Split paired output into pos/neg and applied standard CFG blend.

## Result

- Rejected due correctness regression in strict runtime sanity:
  - Sparse stage output collapsed from expected occupancy.
  - Decode failed with shape-decoder guard after sparse regression.
  - Evidence log: `tmp/runs/20260228T222118Z_trellis2_w29_cfg_batch_fused_low_strict_repeat2/01_run.log`
  - Observed failure signature:
    - `stage sparse complete (..., coords=4703)` (expected around 8k for this run profile)
    - `wgpu sparse conv tensor output exceeds per-dispatch guard: bytes=681885696 max_bytes=536870912`

## Disposition

- All CFG batch-pairing code reverted.
- Canonical sequential CFG path retained.
- Added inline rationale comments in `sparse_structure_flow.rs` to prevent accidental reintroduction without parity proof.

## Post-revert validation

1. `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run`
   - pass
2. `cargo test -p burn_trellis --features runtime-model-wgpu sparse_flow_backend_chunk_tokens_wgpu_use_wider_chunks -- --nocapture`
   - pass
3. Strict low sanity (post-revert):
   - `tmp/runs/20260228T222118Z_trellis2_w29_postrevert_low_strict_repeat1/01_run.log`
   - status `ok`, canonical occupancy restored (`coords=8338`), host readback invariants preserved (`host_readback_count=0`).

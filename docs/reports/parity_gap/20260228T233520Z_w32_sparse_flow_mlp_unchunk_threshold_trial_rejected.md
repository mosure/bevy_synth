# Parity Gap Run Report

- `run_id`: `20260228T233520Z_w32_sparse_flow_mlp_unchunk_threshold_trial_rejected`
- `date_utc`: `2026-02-28`
- `git_ref`: `uncommitted`
- `dirty_worktree`: `true`

## Scope

- Workstream(s): `W32`
- Goal: evaluate raising non-fusion WGPU sparse-flow MLP unchunk threshold from `8192` to `12288`
- Backend: `wgpu`
- Input(s): `docs/input_chair.jpg` (`quality=low`, strict, repeat=2)

## Command(s)

```bash
cargo fmt --all
cargo test -p burn_trellis --features runtime-model-wgpu sparse_flow_backend_chunk_tokens_wgpu_use_wider_chunks -- --nocapture
cargo test -p burn_trellis --features runtime-model-wgpu canonical_wgpu_no_host_readback_before_extraction -- --nocapture
cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run
cargo run -p burn_trellis --features runtime-model-wgpu --bin trellis2_run -- \
  --input docs/input_chair.jpg \
  --quality low \
  --backend wgpu \
  --strict-benchmark \
  --require-runtime-model \
  --repeat 2 \
  2>&1 | tee tmp/runs/20260228T232444Z_trellis2_w32_sparseflow_mlp_unchunk_12288_low_strict_repeat2/01_run.log
python3 scripts/check_trellis_strict_benchmark_invariants.py \
  tmp/runs/20260228T232444Z_trellis2_w32_sparseflow_mlp_unchunk_12288_low_strict_repeat2/01_run.log \
  --min-shape-dispatches 40 \
  --min-tex-dispatches 40
```

## Invariant Summary

- Canonical WGPU fail-fast only: `pass`
- Pre-extraction host readbacks: `pass` (`host_readback_count=0`)
- Decode dispatch presence: `pass` (`decode_shape_wgpu_dispatches=40`, `decode_tex_wgpu_dispatches=40`)
- Runtime source identity: `pass` (`sparse_source=runtime_model_wgpu`, `decode_source=runtime`)

## Timings (ms)

Trial warm run (`repeat=2`, run 2):

- preprocess_ms: `144.529`
- runtime_setup_ms: `0.002`
- sparse_ms: `4954.097`
- shape_slat_ms: `8304.094`
- tex_slat_ms: `4547.159`
- decode_ms: `19658.970`
- decode_shape_decoder_ms: `9196.619`
- decode_tex_decoder_ms: `8007.261`
- decode_attr_merge_ms: `0.437`
- decode_mesh_ms: `17.043`
- decode_pbr_ms: `2412.628`
- total_ms: `37764.874`

## Kernel / Telemetry Counters

- host_readback_count: `0`
- host_readback_elements: `0`
- decode_shape_wgpu_dispatches: `40`
- decode_tex_wgpu_dispatches: `40`
- neighbor_build_ms: `n/a`
- neighbor_query_ms: `n/a`
- sparse_conv_ms: `n/a`

## Comparison

Baseline:
- `tmp/runs/20260228T233900Z_trellis2_w27_sparseflow_mlp_unchunked_small_tokens_rerun/01_run.log`

Trial:
- `tmp/runs/20260228T232444Z_trellis2_w32_sparseflow_mlp_unchunk_12288_low_strict_repeat2/01_run.log`

Delta vs W27 warm:
- `total`: `33461.681 -> 37764.874` (`+12.86%`)
- `sparse`: `3744.439 -> 4954.097` (`+32.31%`)
- `shape_slat`: `6941.085 -> 8304.094` (`+19.64%`)
- `tex_slat`: `3537.364 -> 4547.159` (`+28.55%`)
- `decode`: `18954.243 -> 19658.970` (`+3.72%`)

Delta vs W31 trial warm:
- `total`: `36671.588 -> 37764.874` (`+2.98%`)

## Outcome

- Status: `fail` (not promotable)
- Blocking issue(s): increased sparse/SLAT stage latency despite passing strict invariants
- Next action: keep canonical `8192` threshold and prioritize alternative decode/sparse-flow optimizations.

# Parity Gap Run Report

- `run_id`: `20260228T233520Z_w33_sparse_flow_chunk_gate_trial_rejected`
- `date_utc`: `2026-02-28`
- `git_ref`: `uncommitted`
- `dirty_worktree`: `true`

## Scope

- Workstream(s): `W33`
- Goal: evaluate widening native module-attention dense-window gate from `8192` to `10240`
- Backend: `wgpu`
- Input(s): `docs/input_chair.jpg` (`quality=low`, strict, repeat=2)

## Command(s)

```bash
cargo fmt --all
cargo test -p burn_trellis --features runtime-model-wgpu sparse_flow_chunked_forward_wgpu_dense_window_covers_8338_tokens -- --nocapture
cargo test -p burn_trellis --features runtime-model-wgpu canonical_wgpu_no_host_readback_before_extraction -- --nocapture
cargo run -p burn_trellis --features runtime-model-wgpu --bin trellis2_run -- \
  --input docs/input_chair.jpg \
  --quality low \
  --backend wgpu \
  --strict-benchmark \
  --require-runtime-model \
  --repeat 2 \
  2>&1 | tee tmp/runs/20260228T233520Z_trellis2_w33_sparseflow_chunked_gate_10240_low_strict_repeat2/01_run.log
python3 scripts/check_trellis_strict_benchmark_invariants.py \
  tmp/runs/20260228T233520Z_trellis2_w33_sparseflow_chunked_gate_10240_low_strict_repeat2/01_run.log \
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

- preprocess_ms: `138.450`
- runtime_setup_ms: `0.002`
- sparse_ms: `4803.105`
- shape_slat_ms: `8717.408`
- tex_slat_ms: `4736.692`
- decode_ms: `20428.838`
- decode_shape_decoder_ms: `9381.392`
- decode_tex_decoder_ms: `8593.913`
- decode_attr_merge_ms: `0.408`
- decode_mesh_ms: `17.485`
- decode_pbr_ms: `2419.928`
- total_ms: `38994.464`

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
- `tmp/runs/20260228T233520Z_trellis2_w33_sparseflow_chunked_gate_10240_low_strict_repeat2/01_run.log`

Delta vs W27 warm:
- `total`: `33461.681 -> 38994.464` (`+16.53%`)
- `sparse`: `3744.439 -> 4803.105` (`+28.27%`)
- `shape_slat`: `6941.085 -> 8717.408` (`+25.59%`)
- `tex_slat`: `3537.364 -> 4736.692` (`+33.90%`)
- `decode`: `18954.243 -> 20428.838` (`+7.78%`)

Delta vs W32 warm:
- `total`: `37764.874 -> 38994.464` (`+3.25%`)

## Outcome

- Status: `fail` (not promotable)
- Blocking issue(s): stage-level regressions in sparse/shape/tex/decode despite invariant pass
- Next action: keep canonical `8192` module-attention dense-window gate and prioritize other hotspots.

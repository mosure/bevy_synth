# Parity Gap Run Report

- `run_id`: `20260228T232444Z_w31_decoder_linear_chunk_cap_trial_rejected`
- `date_utc`: `2026-02-28`
- `git_ref`: `uncommitted`
- `dirty_worktree`: `true`

## Scope

- Workstream(s): `W31`
- Goal: evaluate wider wide-row decode linear chunk cap (`8192 -> 12288`) for canonical WGPU decode path
- Backend: `wgpu`
- Input(s): `docs/input_chair.jpg` (`quality=low`, strict, repeat=2)

## Command(s)

```bash
cargo fmt --all
cargo test -p burn_trellis --features runtime-model-wgpu canonical_wgpu_no_host_readback_before_extraction -- --nocapture
cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run
cargo run -p burn_trellis --features runtime-model-wgpu --bin trellis2_run -- \
  --input docs/input_chair.jpg \
  --quality low \
  --backend wgpu \
  --strict-benchmark \
  --require-runtime-model \
  --repeat 2 \
  2>&1 | tee tmp/runs/20260228T232444Z_trellis2_w31_linear_chunk_cap_12288_low_strict_repeat2_rerun/01_run.log
python3 scripts/check_trellis_strict_benchmark_invariants.py \
  tmp/runs/20260228T232444Z_trellis2_w31_linear_chunk_cap_12288_low_strict_repeat2_rerun/01_run.log \
  --min-shape-dispatches 40 \
  --min-tex-dispatches 40
./scripts/guard_canonical_runtime.sh
```

## Invariant Summary

- Canonical WGPU fail-fast only: `pass`
- Pre-extraction host readbacks: `pass` (`host_readback_count=0`)
- Decode dispatch presence: `pass` (`decode_shape_wgpu_dispatches=40`, `decode_tex_wgpu_dispatches=40`)
- Runtime source identity: `pass` (`sparse_source=runtime_model_wgpu`, `decode_source=runtime`)

## Timings (ms)

Trial warm run (`repeat=2`, run 2):

- preprocess_ms: `139.944`
- runtime_setup_ms: `0.002`
- sparse_ms: `4699.965`
- shape_slat_ms: `7759.006`
- tex_slat_ms: `4196.893`
- decode_ms: `19708.660`
- decode_shape_decoder_ms: `8969.475`
- decode_tex_decoder_ms: `8283.134`
- decode_attr_merge_ms: `0.416`
- decode_mesh_ms: `17.235`
- decode_pbr_ms: `2414.927`
- total_ms: `36671.588`

## Kernel / Telemetry Counters

- host_readback_count: `0`
- host_readback_elements: `0`
- decode_shape_wgpu_dispatches: `40`
- decode_tex_wgpu_dispatches: `40`
- neighbor_build_ms: `n/a`
- neighbor_query_ms: `n/a`
- sparse_conv_ms: `n/a`

## Comparison vs promoted W27 warm baseline

Baseline:
- `tmp/runs/20260228T233900Z_trellis2_w27_sparseflow_mlp_unchunked_small_tokens_rerun/01_run.log`

Delta:
- `total`: `33461.681 -> 36671.588` (`+9.59%`)
- `decode`: `18954.243 -> 19708.660` (`+3.98%`)
- `shape_slat`: `6941.085 -> 7759.006` (`+11.78%`)
- `tex_slat`: `3537.364 -> 4196.893` (`+18.64%`)

## Outcome

- Status: `fail` (not promotable)
- Blocking issue(s): wider decode linear chunk cap regressed warm strict profile
- Next action: keep canonical `8192` cap and prioritize alternative decode/sparse-flow optimizations with bounded A/B runs.

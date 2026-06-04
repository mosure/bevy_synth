# Parity Gap Run Report

- `run_id`: `20260228T231250Z_w30_strict_benchmark_invariant_guard_integration`
- `date_utc`: `2026-02-28`
- `git_ref`: `uncommitted`
- `dirty_worktree`: `true`

## Scope

- Workstream(s): `W30`
- Goal: wire strict benchmark invariant checks into canonical runtime guards and optional CI path
- Backend: `wgpu` (log validation only; no new inference run)
- Input(s): `tmp/runs/20260228T233900Z_trellis2_w27_sparseflow_mlp_unchunked_small_tokens_rerun/01_run.log`

## Command(s)

```bash
./scripts/guard_canonical_runtime.sh
python3 scripts/check_trellis_strict_benchmark_invariants.py \
  tmp/runs/20260228T233900Z_trellis2_w27_sparseflow_mlp_unchunked_small_tokens_rerun/01_run.log \
  --min-shape-dispatches 40 \
  --min-tex-dispatches 40
TRELLIS2_STRICT_BENCH_LOG=tmp/runs/20260228T233900Z_trellis2_w27_sparseflow_mlp_unchunked_small_tokens_rerun/01_run.log \
TRELLIS2_STRICT_BENCH_MIN_SHAPE_DISPATCHES=40 \
TRELLIS2_STRICT_BENCH_MIN_TEX_DISPATCHES=40 \
./scripts/guard_canonical_runtime.sh
bash -n scripts/guard_canonical_runtime.sh
python3 -m py_compile scripts/check_trellis_strict_benchmark_invariants.py
```

## Invariant Summary

- Canonical WGPU fail-fast only: `pass` (strict log has `status=ok`, runtime sources canonical)
- Pre-extraction host readbacks: `pass` (`host_readback_count=0`)
- Decode dispatch presence: `pass` (`decode_shape_wgpu_dispatches=40`, `decode_tex_wgpu_dispatches=40`)
- Runtime source identity: `pass` (`sparse_source=runtime_model_wgpu`, `decode_source=runtime`)

## Timings (ms)

- preprocess_ms: `143.420`
- runtime_setup_ms: `0.002`
- sparse_ms: `3744.439`
- shape_slat_ms: `6941.085`
- tex_slat_ms: `3537.364`
- decode_ms: `18954.243`
- decode_shape_decoder_ms: `8476.262`
- decode_tex_decoder_ms: `7989.943`
- decode_attr_merge_ms: `0.433`
- decode_mesh_ms: `22.183`
- decode_pbr_ms: `2453.404`
- total_ms: `33461.681`

## Kernel / Telemetry Counters

- host_readback_count: `0`
- host_readback_elements: `0`
- decode_shape_wgpu_dispatches: `40`
- decode_tex_wgpu_dispatches: `40`
- neighbor_build_ms: `n/a (log not emitted for this check)`
- neighbor_query_ms: `n/a (log not emitted for this check)`
- sparse_conv_ms: `n/a (log not emitted for this check)`

## Outcome

- Status: `pass`
- Blocking issue(s): `none for W30`
- Next action: use the optional CI strict-benchmark guard step for GPU-enabled strict runs and proceed with next kernel-path workstream.

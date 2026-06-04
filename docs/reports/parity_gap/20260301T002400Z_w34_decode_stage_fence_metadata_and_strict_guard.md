# W34 Decode Stage Fence Metadata + Strict Guard Alignment

## Summary

- Added explicit decode timing mode metadata (`decode_stage_fenced`) from staged runtime decode through pipeline timings and CLI JSON output.
- Kept strict benchmark behavior unchanged (`decode_stage_fenced=true` with stage-boundary fences enabled).
- Preserved non-strict fast path (`decode_stage_fenced=false`), making asynchronous decode substage timings explicitly labeled rather than implicitly interpreted as completion timings.
- Extended strict invariant checker to validate `decode_stage_fenced==true` when the field is present.

## Why

Disabling runtime decode stage fences improves non-strict throughput stability, but decode substage timing fields can become dispatch-submit timings instead of completion timings. Without explicit metadata this is ambiguous and leads to incorrect interpretation.

## Code Changes

- `crates/burn_trellis/src/staged_pipeline_runtime_decode.rs`
  - Compute decode timing mode (`decode_stage_fenced`) from runtime tensor-residency + stage-fence config.
  - Propagate timing mode in `DecodeRuntimeTimings`.
- `crates/burn_trellis/src/staged_pipeline.rs`
  - Added `decode_stage_fenced` to `DecodeRuntimeTimings` and `TrellisStageTimings`.
  - Propagated stage timing mode into stage output timings.
- `crates/burn_trellis/src/pipeline.rs`
  - Added `decode_stage_fenced` to `TrellisPipelineTimings`.
- `crates/burn_trellis/tool/trellis2_run.rs`
  - Emit `timings_ms.decode_stage_fenced` in JSON output.
- `scripts/check_trellis_strict_benchmark_invariants.py`
  - Validate `decode_stage_fenced==true` for strict logs when present.
  - Include decode timing mode in summary output.

## Verification

### Build / tests

- `cargo fmt --all` passed.
- `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run` passed.
- `cargo check -p burn_synth` passed.
- `cargo test -p burn_trellis --features runtime-model-wgpu canonical_wgpu_no_host_readback_before_extraction -- --nocapture` passed.
- `./scripts/guard_canonical_runtime.sh` passed.

### Runtime sanity

1. Strict low repeat-2:
   - Command: `cargo run -p burn_trellis --features runtime-model-wgpu --bin trellis2_run -- --input docs/input_chair.jpg --quality low --backend wgpu --strict-benchmark --require-runtime-model --repeat 2`
   - Log: `tmp/runs/20260301T001900Z_trellis2_w34c_stage_fence_metadata_strict_low_repeat2/01_run.log`
   - Warm run: `total=38554.163 ms`, `decode=20064.258 ms`, `decode_stage_fenced=true`, `host_readback_count=0`, dispatches `40/40`.

2. Non-strict low repeat-2:
   - Command: `cargo run -p burn_trellis --features runtime-model-wgpu --bin trellis2_run -- --input docs/input_chair.jpg --quality low --backend wgpu --require-runtime-model --repeat 2`
   - Log: `tmp/runs/20260301T002200Z_trellis2_w34d_stage_fence_metadata_nonstrict_low_repeat2/01_run.log`
   - Warm run: `total=37239.853 ms`, `decode=20035.366 ms`, `decode_stage_fenced=false`, `host_readback_count=0`, dispatches `40/40`.

3. Strict invariant checker:
   - Command: `python3 scripts/check_trellis_strict_benchmark_invariants.py tmp/runs/20260301T001900Z_trellis2_w34c_stage_fence_metadata_strict_low_repeat2/01_run.log --min-shape-dispatches 40 --min-tex-dispatches 40`
   - Result: passed (`decode_stage_fenced=True`).

## Outcome

- Strict benchmark remains authoritative for per-substage decode timing comparisons.
- Non-strict fast path remains available while explicitly labeled as unfenced timing mode.
- Guard tooling now prevents accidental strict-mode timing de-fencing regressions.

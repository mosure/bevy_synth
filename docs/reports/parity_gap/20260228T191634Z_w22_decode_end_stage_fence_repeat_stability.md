# W22 End-of-Decode WGPU Stage Fence for Repeat Stability

Date: 2026-02-28

## Goal

Stabilize repeat-to-repeat stage timing attribution by preventing queued decode work from spilling into the next run's sparse stage.

## Change Summary

- File: `crates/burn_trellis/src/staged_pipeline.rs`
- Added helper:
  - `runtime_pipeline_stage_boundary_sync(stage, enabled)`
  - Uses `<SparseFlowWgpuBackend as Backend>::sync(&WgpuDevice::default())`
  - Fail-fast on sync error.
- Integrated fence after `decode_latent_to_outputs(...)` returns and before decode stage timing is finalized.
- Fence activation condition:
  - enabled when decode emitted WGPU dispatches (`shape_wgpu_dispatches > 0 || tex_wgpu_dispatches > 0`).

## Why this fix

After W21, strict repeat runs showed unstable warm sparse-stage timing despite canonical invariants holding. Telemetry indicated sparse op totals did not explain the large sparse wall-time, implying queue spill from prior decode into next-repeat sparse measurement.

This fence closes the decode stage fully before returning timings/output, so the next repeat starts from an empty queue.

## Verification

- `cargo fmt --all`
- `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run`
- `cargo test -p burn_trellis --features runtime-model-wgpu canonical_wgpu_no_host_readback_before_extraction -- --nocapture`
- `./scripts/guard_canonical_runtime.sh`

All passed.

## Strict Repeat Sanity (after change)

Command:

- `cargo run -p burn_trellis --features runtime-model-wgpu --bin trellis2_run -- --input docs/input_chair.jpg --quality low --backend wgpu --strict-benchmark --require-runtime-model --repeat 2`

Warm run (`run=2`) results:

- `total=41122.508622 ms`
- `sparse=5276.209279 ms`
- `shape_slat=9491.068676 ms`
- `tex_slat=4891.551258 ms`
- `decode=21155.844158 ms`
- `host_readback_count=0`
- `decode_shape_wgpu_dispatches=40`
- `decode_tex_wgpu_dispatches=40`

## Before/After Signal

From same-session W21 strict run prior to this fix, warm sparse timing showed spill behavior:

- prior warm sparse: `20786.317068 ms`
- now warm sparse: `5276.209279 ms`

This confirms repeat-stage spill was removed. Decode timing now includes all decode queue completion work in the originating run.

## Notes

- This is a timing correctness/stability closure, not a new kernel-speed optimization.
- Canonical invariants remain intact (no host readback before extraction, strict fail-fast semantics).

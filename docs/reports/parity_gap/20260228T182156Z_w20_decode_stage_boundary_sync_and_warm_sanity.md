# W20 Decode Stage-Boundary Sync Attribution + Warm Sanity

Run id: `20260228T182156Z_trellis2_wgpu_decode_telemetry_capture`
Date: 2026-02-28
Scope: canonical runtime-model WGPU decode timing attribution + strict warm sanity evidence.

## Change Summary

1. Added explicit decode stage-boundary synchronization for canonical WGPU decode path:
   - `runtime_decode_stage_boundary_sync("shape_decoder", using_device_decode_inputs)?`
   - `runtime_decode_stage_boundary_sync("tex_decoder", using_device_decode_inputs)?`
2. Added helper in `staged_pipeline_runtime_decode.rs`:
   - `runtime_decode_stage_boundary_sync(stage, enabled)` uses `<SparseFlowWgpuBackend as Backend>::sync(&WgpuDevice::default())`.
3. Kept strict fail-fast semantics:
   - sync failure returns `Err(...)` from decode runtime path (no fallback path introduced).
4. Imported `WgpuDevice` in `staged_pipeline.rs` under `runtime-model-wgpu`.

Rationale:

- WGPU dispatch is asynchronous; stage timers were previously able to under-report shape/tex decode subtimings and over-attribute completion to later decode stages.
- Fencing at stage boundaries fixes attribution while preserving canonical runtime behavior and readback invariants.

## Verification

Commands:

```bash
cargo fmt --all
cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run
cargo test -p burn_trellis --features runtime-model-wgpu canonical_wgpu_no_host_readback_before_extraction -- --nocapture
./scripts/guard_canonical_runtime.sh
```

All commands passed.

## Strict Warm Sanity (repeat=2)

Command:

```bash
cargo run -p burn_trellis --features runtime-model-wgpu --bin trellis2_run -- \
  --input docs/input_chair.jpg \
  --output tmp/runs/20260228T182156Z_trellis2_wgpu_decode_telemetry_capture/trellis2_low_repeat2_telemetry.glb \
  --backend wgpu \
  --quality low \
  --strict-benchmark \
  --require-runtime-model \
  --repeat 2 \
  --runtime-decoder-conv-telemetry
```

Warm run (`run=2`) summary:

- `total_ms`: `42545.553067`
- `decode_ms`: `25214.015539`
- `decode_shape_decoder_ms`: `11654.265966`
- `decode_tex_decoder_ms`: `11035.610260`
- `decode_pbr_ms`: `2481.325421`
- `host_readback_count`: `0`
- `host_readback_elements`: `0`
- `decode_shape_wgpu_dispatches`: `40`
- `decode_tex_wgpu_dispatches`: `40`

Non-telemetry strict warm sanity from the same patch window:

- `total_ms`: `44969.610554`
- `decode_ms`: `25510.128568`
- `decode_shape_decoder_ms`: `11769.902959`
- `decode_tex_decoder_ms`: `11229.630019`
- `host_readback_count`: `0`

## Notes

1. Stage-level decode timings are now physically consistent (shape + tex + pbr aligns with decode total).
2. Per-op decoder telemetry remains async-biased for many operations because those probes do not force per-op queue fences; this report treats stage-level fenced timing as the parity/perf source of truth.

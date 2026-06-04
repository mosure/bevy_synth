# W21 Sparse-conv Auto-Fused Decode-Shape Clamp

Date: 2026-02-28

## Goal

Reduce decode-stage sparse-conv time by tightening auto fused-oc4 selection for high-inner-work decode shapes where baseline kernels are faster.

## Change Summary

- File: `crates/burn_flex_gmm/src/wgpu.rs`
- Added constant:
  - `DEFAULT_SPARSE_WGPU_FUSED_AUTO_MAX_IN_CHANNELS_PER_GROUP: usize = 128`
- Updated auto fused gate in `resolve_sparse_conv_kernel_variant` to require:
  - `config.in_channels_per_group <= DEFAULT_SPARSE_WGPU_FUSED_AUTO_MAX_IN_CHANNELS_PER_GROUP`
- Added regression test:
  - `sparse_conv_auto_schedule_keeps_baseline_for_high_inner_work_decode_shape`
  - Asserts auto resolves to `BaselineSingleGroup` for rows `8338`, in/out `1024`.

Rationale comment in code explains why this is conservative: current WGPU fused path is slower on these decode-like workloads.

## Stage Bench Evidence (bounded)

All runs use `cargo run -p burn_flex_gmm --features wgpu-kernel --bin sparse_conv_stage_bench -- ... --warmup 5 --iters 20`.

### rows=4425, in=512, out=512, k=3

- Baseline split1: `p50=99.626ms`, `mean=100.400ms`
- Fused split1: `p50=107.739ms`, `mean=106.050ms`

### rows=8338, in=1024, out=1024, k=3

- Baseline split1: `p50=699.414ms`, `mean=696.305ms`
- Fused split1: `p50=997.580ms`, `mean=1000.198ms`
- Auto before change resolved fused for this shape and tracked fused numbers.

### rows=9955, in=256, out=256, k=3

- Baseline split1: `p50=59.607ms`, `mean=58.854ms`
- Fused split1: `p50=61.239ms`, `mean=61.171ms`

Conclusion: baseline is consistently faster for the measured decode-shape regimes.

## Verification Commands

- `cargo fmt --all`
- `cargo test -p burn_flex_gmm --features wgpu-kernel sparse_conv_auto_schedule_ -- --nocapture`
- `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run`
- `cargo test -p burn_trellis --features runtime-model-wgpu canonical_wgpu_no_host_readback_before_extraction -- --nocapture`
- `./scripts/guard_canonical_runtime.sh`

All passed.

## Runtime Sanity (strict canonical)

Command:

- `cargo run -p burn_trellis --features runtime-model-wgpu --bin trellis2_run -- --input docs/input_chair.jpg --quality low --backend wgpu --strict-benchmark --require-runtime-model --repeat 2`

Observed invariants after change:

- `host_readback_count=0`
- `decode_shape_wgpu_dispatches=40`
- `decode_tex_wgpu_dispatches=40`

Observed decode reduction in a warm run:

- `decode=18613.711891 ms`
- `decode_shape_decoder=8495.993811 ms`
- `decode_tex_decoder=7693.965088 ms`

Telemetry check (`--runtime-decoder-conv-telemetry`, repeat 1) showed fused usage reduced to:

- `fused_variant_calls=2` (previously observed at 17 in this branch/session)

## Notes / Follow-up

Same-session strict runs showed sparse-stage warm-time variance (`sparse ~20.7s` in some repeats). This change targeted decode sparse-conv only; sparse-flow variance remains an independent issue and should be diagnosed separately with stage-isolated sparse-flow runs.

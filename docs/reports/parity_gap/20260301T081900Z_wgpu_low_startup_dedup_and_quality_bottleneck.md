# WGPU TRELLIS Low-Quality Startup Dedup + Cross-Quality Bottleneck Pass

## Scope
- Keep canonical strict WGPU runtime path and no host-readback fallbacks.
- Reduce avoidable startup/runtime overhead without changing numerics.
- Re-measure low quality (`512_base`) and gather quality-level bottleneck signals.

## Code Changes
- File: `crates/burn_trellis/src/staged_pipeline.rs`
1. Added flow-spec equivalence helper (`flow_specs_load_same_model`) to detect duplicate model loads.
2. Added targeted preload selection helper (`should_preload_shape_flow_variant`) so cascade-only shape variants are preloaded only when needed and non-duplicate.
3. Updated preload orchestration to avoid spawning duplicate shape-flow preload threads.
4. Fixed lock behavior so skipped preloads do not poison lazy-load paths with `None`.
5. Updated `shape_flow_runtime_512`/`shape_flow_runtime_1024` to alias `shape_flow_runtime` when both point to the same model, avoiding duplicate runtime loads.

## Validation Commands
- `~/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/bin/cargo fmt --all`
- `~/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/bin/cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run`
- Strict low (repeat baseline/post-fix):
  - `... cargo run -p burn_trellis --release --features runtime-model-wgpu --bin trellis2_run -- --input docs/output_chair_bg_removed.png --backend wgpu --quality low --strict-benchmark --require-runtime-model --repeat 2`
- Strict low with decoder telemetry:
  - same as above + `--runtime-decoder-conv-telemetry`

## Low-Quality Strict Results (repeat=2)

### Baseline (`03_low_repeat2.log`)
- run1 total: `61124.589 ms`
- run1 runtime_setup: `15250.011 ms`
- run2 total: `22558.611 ms`
- run2 sparse: `4764.867 ms`
- run2 shape_slat: `3697.017 ms`
- run2 tex_slat: `2076.663 ms`
- run2 decode: `11854.403 ms`
- run2 host_readback_count: `0`
- run2 decode dispatches: shape `40`, tex `40`

### After preload dedup (`06_low_repeat2_post_patch.log`)
- run1 total: `56637.649 ms` (`-7.34%` vs baseline run1)
- run1 runtime_setup: `12279.384 ms` (`-19.48%` vs baseline run1)
- run2 total: `21754.285 ms` (`-3.57%` vs baseline run2)
- run2 sparse: `4552.364 ms`
- run2 shape_slat: `3372.089 ms`
- run2 tex_slat: `1968.620 ms`
- run2 decode: `11697.027 ms`
- run2 host_readback_count: `0`
- run2 decode dispatches: shape `40`, tex `40`

### Final verification (`10_low_repeat2_final.log`)
- run1 total: `67166.985 ms`
- run1 runtime_setup: `12622.977 ms`
- run2 total: `21823.368 ms`
- run2 sparse: `4550.992 ms`
- run2 shape_slat: `3572.810 ms`
- run2 tex_slat: `2092.355 ms`
- run2 decode: `11439.154 ms`
- run2 host_readback_count: `0`
- run2 decode dispatches: shape `40`, tex `40`

Notes:
- Warm-path low quality remains ~`21.8-21.9s`, below the requested `~30s` target.
- Startup/runtime_setup reduction is stable (~`12.3-12.6s` vs ~`15.3s` baseline).
- run1 variance persists due one-time GPU/kernel/init effects outside model preload alone.

## Decoder Telemetry Findings (Low)
- Source: `05_low_conv_telemetry.log`
- Sparse conv kernel telemetry shows conv kernels are not decode bottleneck:
  - shape decoder sparse-wgpu conv elapsed: `323.74 ms`
  - tex decoder sparse-wgpu conv elapsed: `250.07 ms`
- Neighbor map build is device-only with no host fallback (`host_builds=0`).
- Major remaining low-quality cost is outside sparse conv kernels (decoder non-conv math + flow stages).

## Cross-Quality Bottleneck Signal
- Medium strict (capped attempt still reproduces cascade-scale expansion):
  - sparse stage: `~48.99s`
  - shape_slat stage: `~66.16s`
  - tex_slat stage: `~28.13s`
  - decode shape substage: `~48.82s`, coords `9,532,921`
- Even with sparse coord cap on initial sparse stage, cascade/decode still grows to multi-million rows; this dominates medium/high runtime.

## Immediate Next Optimization Targets
1. Cascade/decode coord growth control with strict parity-preserving semantics.
2. Decoder non-conv hotspot decomposition (matmul/attention/merge kernels) since sparse conv is no longer dominant.
3. Medium/high-specific profiling harness that captures bounded decode checkpoints without requiring full PBR completion.

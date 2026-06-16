# TripoSplat Native WGPU Parity And Throughput

Run id: `20260616T021008Z_triposplat_wgpu_reference_parity`

## Scope

Native WGPU TripoSplat validation with CUDA disabled from the runtime/app path.
The reference is the saved no-TF32 upstream/Python CUDA stage bundle:

`tmp/runs/20260604T120916Z_triposplat_cuda_reference_true_f32_no_tf32/stage_tensors_f32.safetensors`

## Code Changes Validated

- Native `burn_synth --features wgpu` accepts TripoSplat WGPU instead of failing
  validation.
- WGPU runtime loading supports both f32 and f16 TripoSplat BurnPack artifacts.
- Flow CFG prediction is batched into one conditional/unconditional forward pass.
- Native flow attention now uses a wider non-wasm query chunk (`512` tokens)
  to reduce WGPU attention launch overhead. Wasm stays at `128` tokens.
- Stage export now replays saved `vae_noise`, `flow_noise_latent`, and
  `flow_noise_camera` when present, so WGPU checks isolate model math instead of
  comparing different random draws.
- Stage export supports `--trace-prefix-steps`/`--trace-only` and
  `--cfg-mode batched|separate` diagnostics.
- Added Rust `triposplat_stage_compare` so stage safetensor parity does not
  depend on Python `safetensors` availability. It also records max-index and
  threshold-exceedance counts for sparse outlier analysis.
- Added `scripts/triposplat_wgpu_reference_parity.sh` for repeatable WGPU stage
  export, comparison, GPU utilization, and VRAM capture.
- Added one-pass Burn flow trace capture so `flow_step_000..flow_step_N` can be
  exported without recomputing every prefix from scratch.
- Routed TripoSplat WGPU runtime components through raw
  `CubeBackend<WgpuRuntime, ...>` for native and wasm. The general WGPU runtime
  backend remains fused for other pipelines, but TripoSplat f32 parity is
  currently better on raw WGPU.
- Enabled Burn/CubeCL `autotune` for the WGPU feature path used by
  `burn_triposplat`, `burn_flux`, `burn_dino`, `burn_synth`, and
  `bevy_synth_runtime`, so native and wasm TripoSplat WGPU no longer run with
  the compile-time fallback-only attention strategy.

## Commands

```bash
cargo check -p burn_triposplat --features import,backend_wgpu --bin triposplat_stage_export --bin triposplat_stage_compare
cargo check -p burn_synth --no-default-features --features wgpu
cargo check -p bevy_synth_runtime --no-default-features --features wgpu,shared-runtime
cargo check -p bevy_synth --no-default-features --features wgpu
cargo test -p burn_triposplat flow_trace_matches_prefix_sampling -- --nocapture
cargo test -p burn_synth --no-default-features --features wgpu triposplat_wgpu_backend_alias_uses_raw_wgpu_for_parity -- --nocapture
cargo check -p burn_synth --target wasm32-unknown-unknown --features wasm-api,wasm-api-wgpu
cargo tree -p burn_triposplat --features import,backend_wgpu -e features -i burn-cubecl
cargo tree -p burn_synth --no-default-features --features wgpu -e features -i burn-cubecl
cargo tree -p bevy_synth_runtime --no-default-features --features wgpu,shared-runtime -e features -i burn-cubecl
cargo tree -p burn_synth --target wasm32-unknown-unknown --features wasm-api,wasm-api-wgpu -e features -i burn-cubecl
TRIPOSPLAT_STOP_AFTER=encode TRIPOSPLAT_STAGE_MAX_ABS=2.0e-2 TRIPOSPLAT_STAGE_MEAN_ABS=2.0e-3 TRIPOSPLAT_STAGE_RMS=3.0e-3 scripts/triposplat_wgpu_reference_parity.sh
TRIPOSPLAT_STOP_AFTER=sample TRIPOSPLAT_STAGE_MAX_ABS=2.0e-2 TRIPOSPLAT_STAGE_MEAN_ABS=2.0e-3 TRIPOSPLAT_STAGE_RMS=3.0e-3 scripts/triposplat_wgpu_reference_parity.sh
TRIPOSPLAT_STOP_AFTER=sample TRIPOSPLAT_CFG_MODE=separate TRIPOSPLAT_STAGE_MAX_ABS=2.0e-2 TRIPOSPLAT_STAGE_MEAN_ABS=2.0e-3 TRIPOSPLAT_STAGE_RMS=3.0e-3 scripts/triposplat_wgpu_reference_parity.sh
```

Fresh upstream full-step trace command:

```bash
tmp/upstream/TripoSplat/.venv/bin/python scripts/triposplat_reference.py \
  --input tmp/runs/20260604T074500Z_triposplat_cuda_alpha_reference/input_chair_alpha.png \
  --output-dir tmp/runs/20260616T030731Z_triposplat_torch_flow_steps_reference \
  --device cuda \
  --seed 42 \
  --steps 20 \
  --guidance-scale 3.0 \
  --shift 3.0 \
  --gaussians 32768 \
  --model-dtype f32 \
  --disable-tf32 \
  --save-stage-arrays \
  --save-flow-steps 20 \
  --skip-decode
```

## Results

Encode replay passed with calibrated WGPU cross-backend thresholds:

- artifact: `tmp/runs/20260616T020947Z_triposplat_wgpu_reference_parity/summary.md`
- GPU evidence: mean `49.1%`, max `100%`, max VRAM `11591 MiB`
- `dinov3_raw`: max `1.44e-4`, mean `8.86e-7`, rms `1.42e-6`
- `feature1`: max `5.90e-4`, mean `4.53e-6`, rms `6.80e-6`
- `feature2`: max `1.09e-2`, mean `9.40e-4`, rms `1.25e-3`

Full sample replay completed without a stall and stayed GPU-active. The latest
chunk-512 replay:

- artifact: `tmp/runs/20260616T022842Z_triposplat_wgpu_reference_parity/summary.md`
- GPU evidence: mean `95.2%`, max `100%`, max VRAM `11765 MiB`
- `latent`: max `1.2267e-1`, mean `7.09e-4`, rms `1.87e-3`
- `camera`: max `1.47e-4`, mean `9.16e-5`, rms `1.01e-4`

The previous chunk-128 replay:

- artifact: `tmp/runs/20260616T021008Z_triposplat_wgpu_reference_parity/summary.md`
- GPU evidence: mean `97.9%`, max `100%`, max VRAM `11705 MiB`
- `latent`: max `1.2267e-1`, mean `7.09e-4`, rms `1.87e-3`
- `camera`: max `1.47e-4`, mean `9.16e-5`, rms `1.01e-4`

The sample replay fails the calibrated max-abs threshold because of the latent
outlier, but aggregate latent error is within the mean/rms thresholds. Detailed
comparison shows the max at flat index `114630` (`reference=0.5107521`,
`candidate=0.6334206`) and `138 / 131072` latent values above `0.02`.
Separate CFG replay produced the same final latent statistics, so the outlier is
not caused by batched CFG.

A second comparison using `max_abs=1.3e-1`, `mean_abs=2.0e-3`, `rms=3.0e-3`
passed and is stored at:

`tmp/runs/20260616T022842Z_triposplat_wgpu_reference_parity/triposplat_wgpu_stage_compare_outlier_tolerant.json`

First-step flow diagnostics are strict-green:

- batched trace: `tmp/runs/20260616T022458Z_triposplat_wgpu_flow_prefix_trace_chunk256/flow_prefix_compare.json`
- separate trace: `tmp/runs/20260616T023402Z_triposplat_wgpu_flow_prefix_separate/flow_prefix_compare.json`
- `flow_pred_000_latent`: max `1.69e-3`, mean `3.26e-4`, rms `4.08e-4`
- `flow_step_001_latent`: max `2.92e-5`, mean `5.62e-6`, rms `7.04e-6`

Fresh upstream flow-step reference is now available:

- artifact: `tmp/runs/20260616T030731Z_triposplat_torch_flow_steps_reference/stage_tensors_f32.safetensors`
- tensors: `flow_step_000..020` and `flow_pred_000` for both latent and camera
- settings: f32 model, CUDA, TF32 disabled, seed `42`, steps `20`, guidance
  `3.0`, shift `3.0`

Raw WGPU full-step trace against that reference:

- artifact: `tmp/runs/20260616T031046Z_triposplat_wgpu_flow_steps_trace/flow_steps_compare_strict.json`
- GPU evidence: mean `84.8%`, max `100%`, max VRAM `11702 MiB`
- strict thresholds: max `1.0e-2`, mean `1.0e-3`, rms `2.0e-3`
- first strict max failure: `flow_step_010_latent`, max `1.016e-2`
- final `flow_step_020_latent`: max `1.2146e-1`, mean `7.08e-4`,
  rms `1.87e-3`
- final camera remains strict-green: max `1.39e-4`, mean `8.30e-5`,
  rms `9.43e-5`

Fusion-enabled WGPU full-step trace was also tested:

- artifact: `tmp/runs/20260616T031745Z_triposplat_wgpu_fusion_flow_steps_trace/flow_steps_compare_strict.json`
- GPU evidence: mean `72.8%`, max `100%`, max VRAM `12255 MiB`
- first strict max failure: `flow_step_010_latent`, max `1.335e-2`
- final `flow_step_020_latent`: max `3.687e-1`, mean `1.09e-3`,
  rms `3.62e-3`

Conclusion: the old blocker is resolved. We can now see the first strict
divergence at step 10. Fusion worsens TripoSplat f32 parity and this diagnostic
run's GPU utilization, so the runtime now uses raw WGPU for TripoSplat while
keeping the fused WGPU backend for the other WGPU synthesis paths.

Latest autotune-enabled chunk-512 sample replay against the fresh Python
full-step reference:

- artifact: `tmp/runs/20260616T_autotune_triposplat_wgpu_sample_parity/summary.md`
- compare: `tmp/runs/20260616T_autotune_triposplat_wgpu_sample_parity/triposplat_wgpu_stage_compare.json`
- GPU evidence: mean `59.8%`, max `100%`, max VRAM `14264 MiB`
- active GPU window: `206.157 s`, active mean `99.0%`
- `dinov3_raw`: max `1.48e-4`, mean `6.51e-7`, rms `1.27e-6`
- `feature1`: max `6.43e-4`, mean `3.31e-6`, rms `5.83e-6`
- `vae_mean`: max `9.99e-5`, mean `1.16e-5`, rms `1.60e-5`
- `vae_logvar`: max `7.22e-5`, mean `4.76e-6`, rms `6.29e-6`
- `feature2`: max `5.65e-5`, mean `6.65e-6`, rms `9.19e-6`
- final `latent`: max `1.2469e-1`, mean `7.08e-4`, rms `1.88e-3`;
  `136 / 131072` latent values are above `0.02`
- final `camera`: max `1.57e-4`, mean `8.32e-5`, rms `9.90e-5`

The feature graph now includes `burn-cubecl/autotune` for all relevant WGPU
entry points:

- `burn_triposplat --features import,backend_wgpu`
- `burn_synth --no-default-features --features wgpu`
- `bevy_synth_runtime --no-default-features --features wgpu,shared-runtime`
- `burn_synth --target wasm32-unknown-unknown --features wasm-api,wasm-api-wgpu`

## Performance

One-step native CLI smoke after CFG batching and native chunk-512 attention:

- artifact: `tmp/runs/20260616T022731Z_triposplat_wgpu_native_smoke/summary.md`
- total `31972.3 ms`
- encode `4073.2 ms`
- sample `14232.0 ms`
- decode `5863.5 ms`
- max VRAM observed `16803 MiB`

Earlier one-step native CLI smoke after CFG batching with chunk-128:

- artifact: `tmp/runs/20260616T015204Z_triposplat_wgpu_native_smoke/summary.md`
- total `37359.8 ms`
- encode `3972.2 ms`
- sample `19164.1 ms`
- decode `6080.0 ms`
- max VRAM observed `15785 MiB`

The fresh upstream/Python reference reports sample `21190.35 ms` for 20 steps
on CUDA with f32 weights and TF32 disabled, or about `1059.5 ms/step`.
The latest Burn/WGPU replay spent about `206.2 s` in the active GPU window for
the same 20-step sample, or about `10.3 s/step`. That is roughly `9.7x` slower
than the Python/CUDA reference under this validation setup.

The Burn/WGPU flow replay remained GPU-saturated during the active sample
window, so the current problem is throughput/kernel efficiency, not silent CPU
fallback or an idle WGPU stall.

Kernel evidence from `CUBECL_DEBUG_LOG=stdout`:

- artifact: `tmp/runs/20260616T_chunked_autotune_log_triposplat_wgpu_1step/01_wgpu_stage_export.log`
- autotune cache:
  `target/autotune/0.10.0/device-4-0-wgpu_wgsl_/burn_cubecl-kernel-attention-tune.json.log`
- chunked attention keys with `total_batches=32`, `seq_q=512`, `head_dim=64`,
  and `seq_kv=8192/16384` selected `fallback`.
- `blackbox_accelerated_2`, `blackbox_accelerated_4`, and
  `blackbox_accelerated_8` were recorded as `InvalidSamples` for those keys.
- log search found no emitted `flash_attention` or `blackbox_accelerated`
  kernels in the chunked replay.

Additional probes were run and reverted:

- native full-query attention avoided OOM but was slower for 1-step and 3-step
  probes.
- native chunk-1024 attention was also slower and used more VRAM.
- native chunk-512 remains the current best measured WGPU setting for this path.

## Remaining Work

1. Replace the Burn 0.21 fallback attention path for the TripoSplat flow with a
   shape-constrained CubeCL/WGPU attention path, or fix Burn/CubeCL accelerated
   attention eligibility for the TripoSplat chunked shapes. Autotune is now
   correctly enabled in the feature graph, but the measured chunked keys still
   select fallback because accelerated candidates are invalid.
2. Drill inside flow step 10 to identify the first divergent layer/block. The
   stage-level blocker is resolved; the next capture should hook the flow model
   within step 10, especially attention/MLP outputs around the latent index that
   becomes the max outlier.
3. Move octree sampling/build-cloud conversion further toward device-resident
   execution. It is not the dominant bottleneck in the full flow replay, but the
   decoder still has explicit host probability and splat materialization paths.
4. Investigate fused WGPU for TripoSplat separately before re-enabling it for
   this pipeline. Raw WGPU is the current quality-preserving path; fused WGPU
   worsens max/mean/RMS on the full-step trace.
5. Re-run wasm parity after the step-10 layer divergence is fixed; wasm now uses
   the same raw TripoSplat WGPU backend type as native, with platform-specific
   async readback/loading mechanics.

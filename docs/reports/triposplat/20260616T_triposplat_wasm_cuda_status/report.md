# TripoSplat Wasm, CUDA, and Runtime Status

Run ID: `20260616T_triposplat_wasm_cuda_status`

## Summary

This pass tightened the canonical TripoSplat runtime path and made the wasm failure mode explicit.

- Native `burn_synth` now emits stage-level TripoSplat progress for prepare, encode, sample, and decode rather than one opaque inference span.
- The canonical octree sampler now evaluates parent candidates in chunks. This preserves probabilities because the octree decoder has no query-query coupling, and it bounds WebGPU graphs for the wasm path.
- DINOv3 wasm attention is query-chunked and cleans backend memory after each block; the prior 4.28 GiB encode allocation no longer reproduces.
- TripoSplat wasm no longer attempts the known-broken f32 browser path. It now requires a WebGPU adapter exposing `shader-f16` and keeps fp16 TripoSplat modules in fp16 instead of casting them back to f32 after load.

## Validation Evidence

| Area | Command | Result |
| --- | --- | --- |
| Format | `cargo fmt --check` | Pass |
| Octree sampler | `cargo test -p burn_triposplat octree -- --nocapture` | Pass: 5 tests, including full-vs-chunked parent probability parity |
| Wasm API build | `cargo check -p burn_synth --target wasm32-unknown-unknown --features wasm-api,wasm-api-wgpu` | Pass |
| Wasm UX/API smoke | `BURN_SYNTH_WEB_TRIPOSPLAT_SMOKE=1 ... crates/burn_synth/tests/web_playwright/run.sh` | Pass: 3 passed, 3 skipped |
| Native runtime progress | `cargo test -p burn_synth --features runtime triposplat_native_runtime_emits_stage_level_progress -- --nocapture` | Pass |
| Bevy wrappers | `cargo check -p bevy_synth_runtime && cargo check -p bevy_synth` | Pass |
| CUDA compile | `cargo check -p burn_synth --features cuda` | Pass |
| CUDA reference parity guard | `RUN_ID=20260616T_cuda_toolkit_guard_check3 scripts/triposplat_cuda_reference_parity.sh` | Blocked before model execution: Blackwell compute capability 12.0 with visible CUDA/NVRTC 12.4 |
| CLI splat wiring | `cargo test -p burn_synth --features runtime splat_cli -- --nocapture` | Pass: 5 tests, including dry-run `.splat` output |

## Wasm Result

Current local Chrome/WebGPU reports:

```json
{
  "hasGpu": true,
  "adapter": true,
  "shaderF16": false
}
```

The TripoSplat wasm smoke now records a skip artifact instead of loading model parts and losing the device:

- Result: `tmp/wasm/20260616T_triposplat_wasm_shaderf16_gate/wasm_triposplat_result.json`
- Console: `tmp/wasm/20260616T_triposplat_wasm_shaderf16_gate/wasm_triposplat_console.log`
- Model requests: none
- Page errors: none

Prior failing evidence is preserved at:

- `tmp/wasm/20260616T_triposplat_wasm_octree_chunk_smoke/wasm_triposplat_console.log`

That run completed encode and sample, then failed at octree decode with `VK_ERROR_OUT_OF_DEVICE_MEMORY` and `VK_ERROR_DEVICE_LOST`. The f32 browser path is therefore not treated as a valid fallback.

## CUDA Result

CUDA reference parity did not run on this host. The preflight guard exits before Burn/CubeCL kernel compilation:

```text
GPU compute capability: 12.0
Visible nvcc/NVRTC toolkit: 12.4
This Blackwell GPU requires CUDA/NVRTC 12.9+ for CubeCL/Burn runtime kernel compilation.
```

Install CUDA/NVRTC 12.9+ or point `CUDA_PATH` and `LD_LIBRARY_PATH` at a matching toolkit, then rerun:

```bash
RUN_ID=20260616T_cuda_reference_parity \
scripts/triposplat_cuda_reference_parity.sh
```

## Remaining Gates

1. Run the TripoSplat wasm smoke on a browser/adapter exposing `shader-f16`; expected path is fp16 backend plus f16 `.bpk.parts.json` artifacts.
2. Re-run CUDA stage parity and `.splat` comparison after CUDA/NVRTC 12.9+ is visible.
3. Re-enable native WGPU TripoSplat only after a real no-panic WGPU e2e smoke is recorded.
4. Replace host octree readback/cloud materialization with a GPU-resident decode path where practical; current octree sampling still reads probabilities to host at each level.
5. Refresh performance benchmarks after the CUDA and shader-f16 wasm gates pass; current validation is compile/runtime smoke plus older CUDA sweep evidence.


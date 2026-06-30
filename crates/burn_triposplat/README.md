# burn_triposplat

Canonical TripoSplat integration crate for `burn_synth`.

This crate currently owns:

- upstream TripoSplat runtime defaults and artifact names,
- Gaussian splat data types,
- `.splat` and binary little-endian PLY export compatible with upstream TripoSplat,
- model-root validation and sharded `.bpk.parts.json` artifact recognition,
- DINOv3 conditioning support through `burn_dino`,
- Flux2 VAE encoder conditioning support through `burn_flux`,
- TripoSplat latent flow, octree probability, host systematic octree traversal,
  and elastic Gaussian decoder modules,
- safetensors-to-BurnPack import plus single-file and sharded `.bpk.parts.json`
  load APIs for runtime-required TripoSplat components,
- explicit f32/f16 BurnPack artifact precision as the current quantization
  surface; lower-bit or weight-only quantization is not yet supported,
- a canonical preprocessed-tensor runtime path that encodes DINOv3/Flux2
  conditioning, samples flow latents with seeded noise, and decodes Gaussian
  splats,
- upstream-style multi-density decoding that samples one flow latent once and
  replays the decoder for multiple requested Gaussian counts,
- native `burn_synth::runtime::SynthRuntime::{synthesize_splats,synthesize_asset}`
  wiring that reuses canonical foreground preprocessing, caches
  backend-specific TripoSplat components, and returns Gaussian-splat assets,
- `burn_synth splat --gaussians 32768,65536` multi-output export with
  count-suffixed `.splat` or binary little-endian `.ply` files,
- `bevy_synth_runtime` transport of mesh-or-splat assets plus `bevy_synth`
  headless/output-path export for `.splat` and binary little-endian `.ply`
  Gaussian-splat assets,
- native `bevy_synth` bounded Gaussian-splat mesh preview for in-scene
  inspection while the canonical asset remains `.splat`/PLY,
- `burn_synth` wasm async sharded TripoSplat component loading plus JS-facing
  `.splat` and binary PLY byte outputs, with `bevy_synth_runtime` wasm transport
  of the canonical Gaussian-splat asset.

The direct `TripoSplatPipeline::infer_image` entrypoint and browser/Bevy e2e
paths are still intentionally fail-fast until upstream-weight numerical parity,
GPU-resident octree/splat materialization, and production splat rendering are
completed. Use `TripoSplatPipeline::stage_status()` for the precise readiness
contract.

## Artifact Bootstrap

TripoSplat runtime loads repo-canonical `.bpk` or `.bpk.parts.json` artifacts
under:

```text
crates/burn_triposplat/assets/models/TripoSplat/
```

The upstream safetensors source root must contain the official TripoSplat
checkpoint layout:

```text
background_removal/birefnet.safetensors
diffusion_models/triposplat_fp16.safetensors
clip_vision/dino_v3_vit_h.safetensors
vae/flux2-vae.safetensors
vae/triposplat_vae_decoder_fp16.safetensors
```

Use the workspace bootstrap script to download or reuse those upstream files,
convert runtime-required components to BurnPack, and generate wasm parts:

```bash
scripts/triposplat_bootstrap.sh
```

Useful variants:

```bash
scripts/triposplat_bootstrap.sh --precision both --overwrite-parts
scripts/triposplat_bootstrap.sh --skip-download --source-root /path/to/TripoSplat/ckpts
scripts/triposplat_bootstrap.sh --output-root www/assets/models/TripoSplat --precision f16
```

The script keeps upstream downloads under `tmp/upstream/TripoSplat/` by default
and never writes generated artifacts at the repository root.

## Reference And Quantization Policy

Current TripoSplat artifact precision is intentionally limited to:

- `f32`: correctness-first native default when available.
- `f16`: smaller storage artifact for loading, wasm, and backend-specific
  investigation.

No int8/int4 or weight-only quantization format is currently defined for this
pipeline. Add one only with stage-level numerical parity against upstream
reference outputs.

Generate upstream Python reference outputs with:

```bash
python3 scripts/triposplat_reference.py \
  --input docs/input_chair.jpg \
  --gaussians 32768 \
  --save-stage-arrays
```

The script writes `reference.json`, prepared image output, `.splat`/PLY assets,
and optional `stage_tensors_f32.safetensors` feature/latent arrays under
`tmp/runs/<run_id>/`. If upstream RMBG preprocessing is unavailable on the
current GPU, pass an RGBA input with real alpha; upstream TripoSplat skips RMBG
for alpha-matted inputs and still produces DINOv3/Flux2/flow/decoder reference
evidence.

Compare Rust stage tensors to upstream stage tensors with:

```bash
target/debug/triposplat_stage_export \
  --weights-root crates/burn_triposplat/assets/models/TripoSplat \
  --precision f32 \
  --input-stages tmp/runs/<reference_run>/stage_tensors_f32.safetensors \
  --output tmp/runs/<rust_run>/stage_tensors_f32.safetensors \
  --stop-after encode

python3 scripts/triposplat_compare_stage_tensors.py \
  tmp/runs/<reference_run>/stage_tensors_f32.safetensors \
  tmp/runs/<rust_run>/stage_tensors_f32.safetensors \
  --report tmp/runs/<rust_run>/triposplat_stage_parity.json
```

The same comparison can run without Python package dependencies:

```bash
cargo run -p burn_triposplat --features import --bin triposplat_stage_compare -- \
  tmp/runs/<reference_run>/stage_tensors_f32.safetensors \
  tmp/runs/<rust_run>/stage_tensors_f32.safetensors \
  --report tmp/runs/<rust_run>/triposplat_stage_parity.json
```

For native WGPU stage parity and GPU evidence capture:

```bash
TRIPOSPLAT_STOP_AFTER=encode \
  scripts/triposplat_wgpu_reference_parity.sh
```

Set `TRIPOSPLAT_STOP_AFTER=sample` for full encode+flow-stage parity against
the saved upstream latent. Real-model sample parity is intentionally slower and
should be run with a timeout appropriate for the selected backend.

Compare a Rust `.splat` output to an upstream reference with:

```bash
python3 scripts/triposplat_compare_splat.py \
  tmp/runs/<reference_run>/reference_32768.splat \
  tmp/runs/<rust_run>/output_32768.splat \
  --report tmp/runs/<rust_run>/triposplat_splat_parity.json
```

The gated Rust reference-contract check validates a completed upstream run's
metadata and stage-tensor evidence format:

```bash
TRIPOSPLAT_REFERENCE_JSON=tmp/runs/<reference_run>/reference.json \
  cargo test -p burn_triposplat --features import \
  triposplat_reference_metadata_contract_reference -- --nocapture
```

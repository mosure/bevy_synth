# Vendored and Patched Dependencies

Last reviewed: 2026-06-18

This repository vendors or pins a small set of dependencies when upstream crates do
not yet match the repo's Bevy, Burn, WGPU, wasm, or numerical-correctness
requirements. Vendoring should stay narrow: the canonical goal is to remove local
patches once upstream crates support the same behavior.

## Policy

- Cargo patches are declared in the workspace root `Cargo.toml` under
  `[patch.crates-io]`.
- Vendored crates live under `third_party/` and are not normal workspace
  implementation modules.
- Pinned git dependencies are not vendored crates; they are recorded separately
  because their revisions are part of the reproducible build surface.
- Local source adaptations copied into project crates must be called out here, but
  they should not grow into hidden alternate implementations.
- Before changing a vendored dependency, compare against the upstream crate or git
  revision, keep the delta focused, and run the validation commands for the
  affected pipeline.

## Current Summary

| Dependency | Location | Upstream | Wiring | Why it is patched or pinned |
| --- | --- | --- | --- | --- |
| `burn_dino` | `third_party/burn_dino` | crates.io `burn_dino` 0.7.0 | `[patch.crates-io]` path override | Burn 0.21 API compatibility, DINOv3 support, TripoSplat image-encoder parity, wasm loading, and attention/rope fixes. |
| `cubek-attention` | `third_party/cubek-attention` | crates.io `cubek-attention` 0.2.0 | `[patch.crates-io]` path override | WGPU long-sequence attention safety and dtype selection fixes needed by TripoSplat. |
| `bevy` | pinned git rev `ae2fcc0353d95e887470f0f6fc8a7e434e5549ce` | `github.com/bevyengine/bevy` | direct git dependency in Bevy-facing crates | Keeps Bevy and Burn aligned on the WGPU 29 stack. |
| `bevy_gaussian_splatting` | pinned git rev `e0da1488d7441db54e4f86cc8f5845df308b306f` | `github.com/mosure/bevy_gaussian_splatting` | direct git dependency in `bevy_synth` | Provides the Bevy/WGPU-compatible Gaussian splat renderer used by TripoSplat native and wasm rendering. |

## `third_party/burn_dino`

`burn_dino` is patched through the workspace root:

```toml
[patch.crates-io]
burn_dino = { path = "third_party/burn_dino" }
```

### Reasons for the local fork

- Align the crate with workspace Burn `=0.21.0`.
- Avoid older optional Bevy/import dependency paths that pulled in incompatible
  app-stack crates.
- Support DINOv3 ViT-H/16 behavior used by TripoSplat.
- Preserve numerical parity hooks for image-encoder debugging.
- Improve attention execution by using Burn's module attention path where the
  shape and mask behavior allow it.
- Keep wasm-friendly model loading and attention chunking available for the
  TripoSplat web path.

### Main patch areas

- `Cargo.toml`: pins Burn dependencies to `=0.21.0`, keeps import support aligned
  with `burn-store`, and removes stale optional Bevy argument wiring.
- `src/layers/attention.rs`: routes unmasked attention through Burn's
  `tensor::module::attention` API, while retaining the explicit fallback path for
  masked or quiet-softmax cases.
- `src/layers/rope.rs`: changes RoPE construction to match the DINOv3/HF
  y/x layout used by upstream TripoSplat references.
- `src/layers/layer_norm.rs`: adjusts epsilon and accumulation behavior for
  DINOv3 numerical stability.
- `src/model/dino.rs`: updates Burn 0.21 shape/device/module APIs, interpolation
  options, and RoPE coordinate handling.
- `src/model/dinov3.rs`: adds the DINOv3 model, import/remap support for HF
  safetensors and BurnPack loading, RoPE cache handling, dtype conversion, and
  wasm attention chunking.
- `tool/import.rs`, `tests/correctness.rs`, and examples: update import and
  correctness utilities to the same Burn 0.21 device/type APIs.

### Risk notes

This is the largest vendored delta. It is correctness-critical for TripoSplat and
Trellis image-encoder paths, so changes should be stage-validated rather than
accepted from successful compilation alone. Environment-controlled debug switches
inside the fork are for parity investigation and should not become hidden runtime
configuration surfaces in production paths.

### Suggested validation

Run the smallest command that exercises the touched path first, then broaden:

```bash
cargo check -p burn_triposplat --features import
cargo test -p burn_triposplat --features import dinov3 -- --nocapture
cargo check -p burn_synth --features triposplat
cargo check -p burn_synth --target wasm32-unknown-unknown --features wasm-api,wasm-api-wgpu,triposplat
```

When the change affects numerical behavior, also run the relevant saved-reference
or stage-parity TripoSplat tests before committing.

## `third_party/cubek-attention`

`cubek-attention` is patched through the workspace root:

```toml
[patch.crates-io]
cubek-attention = { path = "third_party/cubek-attention" }
```

### Reasons for the local fork

TripoSplat has long dense attention shapes, including sequence lengths around
8192 and 12294 with head dimension 64. The stock WGPU path can select attention
units that are unsafe or pathological for those native f32 shapes, and dtype
selection can choose an internal tile type that is not appropriate for the
strict f32 correctness path.

### Main patch areas

- `src/definition/spec.rs`: when query/key/value global dtypes match and the
  query dtype is at least 32-bit wide, the attention tile dtype is forced to the
  query dtype. This keeps the f32 path from silently using a lower-precision tile
  type.
- `src/routines/unit.rs`: adds a native WGPU guard for very large f32 unit
  attention shapes. Shapes with large `batch_heads` and `seq_q * seq_kv` are
  rejected with an explicit error instead of launching a known-unsafe path.

### Risk notes

This patch intentionally prevents one bad accelerated path from running. It does
not by itself make every TripoSplat attention shape fast. Benchmark claims should
include whether the run used the padded f16 blackbox path, direct Burn attention,
or the guarded f32 fallback behavior.

### Suggested validation

```bash
cargo test -p burn_triposplat --features import,backend_wgpu attention -- --nocapture
cargo bench -p burn_triposplat --features import,backend_wgpu
```

For performance-sensitive changes, capture stage timings and GPU utilization for
encoder, sampling, and decode separately.

## Pinned Git Dependencies

### Bevy

Bevy-facing crates pin Bevy to:

```toml
git = "https://github.com/bevyengine/bevy.git"
rev = "ae2fcc0353d95e887470f0f6fc8a7e434e5549ce"
```

The pin keeps Bevy on the same WGPU 29 family expected by Burn/CubeCL in this
workspace. Do not bump this independently of Burn, CubeCL, `wgpu`, or
`bevy_gaussian_splatting` validation.

Suggested validation after changing the Bevy revision:

```bash
cargo check -p bevy_synth
cargo check -p bevy_synth_runtime
cargo check -p burn_synth --target wasm32-unknown-unknown --features wasm-api,wasm-api-wgpu
```

### `bevy_gaussian_splatting`

`bevy_synth` pins:

```toml
git = "https://github.com/mosure/bevy_gaussian_splatting.git"
rev = "e0da1488d7441db54e4f86cc8f5845df308b306f"
```

This renderer is used for TripoSplat Gaussian cloud output rather than a mesh
proxy. The pin exists because it is known to match the Bevy/WGPU stack above and
is expected to work for native and wasm rendering.

Suggested validation after changing this revision:

```bash
cargo check -p bevy_synth
cargo test -p bevy_synth_runtime triposplat -- --nocapture
```

Also validate a visual or render-metric comparison for representative
TripoSplat outputs when splat packing, SH/color handling, or transform behavior
changes.

## Local Source Adaptations

These are not patched crates in Cargo, but they are part of the same dependency
maintenance surface.

### Infinite grid

`crates/bevy_synth/src/infinite_grid.rs` is a local adaptation of
`fslabs/bevy_infinite_grid` for the pinned Bevy render API. Keep this file
focused on grid rendering. If upstream becomes compatible with the pinned Bevy
revision, prefer switching back to the external crate.

Validation:

```bash
cargo check -p bevy_synth
```

### Transform gizmos

`crates/bevy_synth_ui/src/bevy_transform_gizmos/` is an internalized copy of the
transform gizmo code while crate publishing or upstream Bevy compatibility is in
flight. This is UI/runtime adaptation, not inference logic.

Validation:

```bash
cargo check -p bevy_synth
cargo check -p bevy_synth_ui
```

### Panorbit camera behavior

The app currently carries local panorbit-style camera controls in
`crates/bevy_synth/src/app.rs` instead of depending directly on
`bevy_panorbit_camera`. The goal is standard orbit/pan/zoom behavior while
remaining compatible with the pinned Bevy revision.

Validation:

```bash
cargo check -p bevy_synth
```

## Update Checklist

1. Identify the exact upstream crate version or git revision being patched.
2. Diff the local copy against that upstream source before editing.
3. Keep the change in the vendored crate only if the canonical upstream crate
   cannot currently support the needed behavior.
4. Run compile checks for the direct crate users.
5. Run stage-level numerical parity tests when math, dtype, layout, loading, or
   tensor shape behavior changes.
6. Run stage-level benchmarks and capture GPU evidence when performance behavior
   changes.
7. Update this document with the reason, files changed, risks, and validation
   commands.
8. Remove the vendor patch once the upstream crate provides the required behavior
   and the workspace validation matrix still passes.

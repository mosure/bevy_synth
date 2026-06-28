# burn_locate_anything 🔥📍

[![crates.io](https://img.shields.io/crates/v/burn_locate_anything.svg)](https://crates.io/crates/burn_locate_anything)
[![GitHub License](https://img.shields.io/github/license/mosure/bevy_synth)](https://raw.githubusercontent.com/mosure/bevy_synth/main/LICENSE-MIT)

Burn-native [LocateAnything](https://github.com/NVlabs/Eagle/tree/main) visual grounding.

`burn_locate_anything` provides model asset inspection, preprocessing, tokenizer/prompt assembly,
MoonViT/projector/Qwen runtime components, and LocateAnything box decoding for Burn backends. The
published crate exposes the Burn-native runtime surface only; Python/Torch comparison tooling stays
in the repository as validation support and is excluded from crates.io packages.

## usage

```rust
use burn_locate_anything::{
    DetectionQuery, LocateAnythingDetector, LocateAnythingRuntime,
    LocateAnythingRuntimeConfig,
};

let image = image::open("assets/images/scene.jpg")?;
let mut runtime = LocateAnythingRuntime::new(LocateAnythingRuntimeConfig {
    model_root: "assets/models/LocateAnything-3B".into(),
    allow_experimental_native_detect: true,
    ..LocateAnythingRuntimeConfig::default()
})?;

let detections = runtime.detect_batch(
    &image,
    &[
        DetectionQuery {
            query: "conference table".to_string(),
            label_hint: None,
        },
        DetectionQuery {
            query: "conference chair".to_string(),
            label_hint: None,
        },
    ],
)?;
```

## features

- [x] model asset inspection for HF safetensor snapshots
- [x] Qwen tokenizer and LocateAnything prompt formatting
- [x] image preprocessing and MoonViT patch planning
- [x] MoonViT/projector/Qwen Burn modules
- [x] parallel box decoding and text-output box parsing
- [x] WGPU-native runtime path
- [x] wasm-compatible crate check without native tokenizer execution
- [x] CDN/cache loader for sharded `.bpk.parts.json` weight artifacts
- [ ] broad multi-scene checkpoint parity gates

## setup

Download the upstream `nvidia/LocateAnything-3B` Hugging Face snapshot and place it under a model
root such as `assets/models/LocateAnything-3B`.

Prepare a CDN-ready metadata + sharded blob-burnpack bundle:

```bash
cargo run -p burn_locate_anything --features import --bin locate_anything_import -- \
  --hf-root assets/models/LocateAnything-3B \
  --output-dir tmp/runs/<run_id>/model/LocateAnything-3B \
  --precision bf16 \
  --shard-size-mib 64
```

The generated upload layout is rooted at `model/LocateAnything-3B`. Runtime CDN loading is opt-in
through `LocateAnythingRuntimeConfig { cdn_base_url, cache_dir, allow_download: true, .. }`; local
`model_root` snapshots are still preferred when present.

> Pre-trained model weights have separate licenses. Keep converted checkpoints and generated
> `.bpk` artifacts out of source control unless a release process explicitly includes them.

## validation

```bash
cargo test -p burn_locate_anything --lib --features backend_wgpu -- --nocapture
cargo check -p burn_locate_anything --target wasm32-unknown-unknown
cargo clippy -p burn_locate_anything --features backend_wgpu -- -D warnings
```

Expensive numerical parity tests are gated by environment variables and require local model weights
plus explicitly supplied fixture paths, for example `LOCATE_ANYTHING_PARITY_IMAGE`.

## references

- [LocateAnything / Eagle](https://github.com/NVlabs/Eagle/tree/main)
- [Burn](https://github.com/tracel-ai/burn)
- [bevy_synth](https://github.com/mosure/bevy_synth)

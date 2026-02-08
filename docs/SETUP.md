# Setup Guide

This project supports multiple model families. Configure only the models you plan to use.

## TripoSG (3D generation, implemented)

### Weights

- Set `TRIPOSG_WEIGHTS_ROOT` to a TripoSG weight directory, or place weights in:
  - `crates/burn_tripo/assets/models/MIDI-3D`
- Optional scribble weights:
  - `TRIPOSG_SCRIBBLE_WEIGHTS_ROOT`
  - or `crates/burn_tripo/assets/models/TripoSG-scribble`

### Burnpack import (native)

```powershell
cargo run -p burn_tripo --bin triposg_import --features import
```

### App selection

- Select TripoSG backend:
  - `--synthesis-models triposg`

## RMBG-1.4 (foreground removal, implemented)

### Weights

- Set `RMBG_WEIGHTS_ROOT` (or `--bg-weights-root`) to RMBG-1.4 root.
- Default fallback location:
  - `crates/burn_foreground/assets/models/RMBG-1.4`
- Canonical foreground model layout reference:
  - `crates/burn_foreground/assets/models/README.md`
- Burnpack import (native):
  - `cargo run -p burn_foreground --features import --bin foreground_import -- --rmbg-model rmbg14 --quantization both`

### App selection

- Select RMBG-1.4:
  - `--rmbg-model rmbg14`

## RMBG-2.0 (foreground removal, implemented)

### Weights

- Set `RMBG2_WEIGHTS_ROOT` (or `--bg-weights-root`) to RMBG-2.0 root.
- Default fallback location:
  - `crates/burn_foreground/assets/models/RMBG-2.0`
- Canonical foreground model layout reference:
  - `crates/burn_foreground/assets/models/README.md`
- Generate RMBG-2.0 burnpacks (f32/f16) with:
  - `cargo run -p burn_foreground --features import --bin foreground_import -- --rmbg-model rmbg2 --quantization both --rmbg2-root crates/burn_foreground/assets/models/RMBG-2.0`

### App selection

- Select RMBG-2.0:
  - `--rmbg-model rmbg2`
- `bevy_synth` defaults to `rmbg2`; you only need the flag to force a model.

Notes:
- On native, if `rmbg2` is selected but unavailable, the runtime falls back to RMBG-1.4 automatically.
- On native, RMBG-2.0 prefers `model_f16.bpk` / `model.bpk` and falls back to ONNX files if burnpacks are missing.
- Precision preference can be set with `RMBG2_BPK_PRECISION` (`f16` default, or `f32`).
- On wasm32, RMBG-2.0 is unavailable and the runtime falls back to RMBG-1.4.
- If folders appear empty in your checkout, weights were not bundled; point env vars to your local model roots or copy weights into the canonical paths above.

Verify canonical RMBG-2.0 burnpacks:

```powershell
Get-ChildItem crates/burn_foreground/assets/models/RMBG-2.0/model*.bpk
```

## Trellis2 (native Rust runtime in progress)

### Weights

- Set `TRELLIS2_WEIGHTS_ROOT` to the TRELLIS.2-4B root containing `pipeline.json`.
- Optional root for TRELLIS-image-large assets:
  - `TRELLIS2_IMAGE_LARGE_ROOT`
- Canonical in-repo locations:
  - `crates/burn_trellis/assets/models/TRELLIS.2-4B`
  - `crates/burn_trellis/assets/models/TRELLIS-image-large`

### Runtime requirements

- Trellis2 runs through the same Rust `burn_trellis` module on native and wasm (no Python bridge runtime path).
- Current implementation includes asset validation, preprocessing parity hooks, and import tooling.
- Full Trellis2 model execution stages are still being implemented.

### Burnpack import (native)

```powershell
cargo run -p burn_trellis --features import --bin trellis2_import -- --quantization both
```

### App selection

- Select Trellis2 only:
  - `--synthesis-models trellis`
- Select both with fallback:
  - `--synthesis-models trellis,triposg`
- Quality presets:
  - `--trellis-quality low|medium|high`

## Combined examples

### TripoSG + RMBG-2.0

```powershell
cargo run -p bevy_synth --release -- --synthesis-models triposg --rmbg-model rmbg2
```

### TripoSG + RMBG-1.4

```powershell
cargo run -p bevy_synth --release -- --synthesis-models triposg --rmbg-model rmbg14
```

### Enable both synthesis backends (Trellis preferred, TripoSG fallback)

```powershell
cargo run -p bevy_synth --release -- --synthesis-models trellis,triposg --rmbg-model rmbg2 --trellis-quality medium
```

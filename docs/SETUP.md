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

## Trellis (planned, not implemented yet)

- Trellis integration is not implemented yet in this repo.
- You can still select it in the app backend list (`--synthesis-models trellis`) to reserve config, but inference will return a clear "not implemented" error.
- No Trellis weights are currently required by the runtime.

## Combined examples

### TripoSG + RMBG-2.0

```powershell
cargo run -p bevy_synth --release -- --synthesis-models triposg --rmbg-model rmbg2
```

### TripoSG + RMBG-1.4

```powershell
cargo run -p bevy_synth --release -- --synthesis-models triposg --rmbg-model rmbg14
```

### Reserve both synthesis backends (TripoSG active, Trellis planned)

```powershell
cargo run -p bevy_synth --release -- --synthesis-models triposg,trellis --rmbg-model rmbg2
```

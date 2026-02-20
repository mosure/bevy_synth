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
- `bevy_synth` defaults to `rmbg14`; use `--rmbg-model rmbg2` to opt into RMBG-2.0.

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

## Trellis2 (native/wasm Rust runtime)

### Weights

- Set `TRELLIS2_WEIGHTS_ROOT` to the TRELLIS.2-4B root containing `pipeline.json`.
- Set `TRELLIS2_IMAGE_LARGE_ROOT` to the TRELLIS-image-large root for decoder assets.
- Canonical in-repo locations:
  - `crates/burn_trellis/assets/models/TRELLIS.2-4B`
  - `crates/burn_trellis/assets/models/TRELLIS-image-large`

### Runtime requirements

- Trellis2 runs through the same Rust `burn_trellis` module on native and wasm (no Python bridge runtime path).
- Runtime prefers burnpack weights and loads `*_f16.bpk` first by default.
- Canonical runtime files under `crates/burn_trellis/assets/models/TRELLIS.2-4B/ckpts`:
  - `*.bpk` (f32)
  - `*_f16.bpk` (f16)
  - `*.json` model configs used by runtime loading

### Burnpack import (native)

```powershell
cargo run -p burn_trellis --features import --bin trellis2_import -- --quantization both
```

- `trellis2_import` now writes:
  - primary assets into `TRELLIS.2-4B`
  - image-large assets into `TRELLIS-image-large`
- Import fails on missing source checkpoints (no silent skip).

### Verify imported files

```powershell
Get-ChildItem crates/burn_trellis/assets/models/TRELLIS.2-4B/ckpts -Filter *.bpk
Get-ChildItem crates/burn_trellis/assets/models/TRELLIS.2-4B/ckpts -Filter *_f16.bpk
Get-ChildItem crates/burn_trellis/assets/models/TRELLIS-image-large/ckpts -Filter *.bpk
Get-ChildItem crates/burn_trellis/assets/models/TRELLIS-image-large/ckpts -Filter *_f16.bpk
```

### Hook parity checks

- Baseline sampled hook alignment (runtime model disabled, schema + finite stats checks):
  - `cargo test -p burn_trellis --test e2e_hook_alignment`
- Strict gate (non-synthetic sparse stage + `1e-3` threshold on matched hooks):
  - `$env:TRELLIS2_E2E_STRICT='1'`
  - `$env:TRELLIS2_E2E_DISABLE_RUNTIME_MODEL='0'`
  - `cargo test -p burn_trellis --features runtime-model --test e2e_hook_alignment -- --nocapture`
- Optional runtime sparse-coordinate cap for large runs:
  - `$env:TRELLIS2_MAX_SPARSE_COORDS='1024'`

### Web asset bundle

- Use `scripts/bundle_web_assets.ps1` to collect runtime web assets into `www/assets/models`.
- The bundle now includes Trellis runtime assets (`TRELLIS.2-4B` and `TRELLIS-image-large`) in addition to TripoSG and RMBG.

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
cargo run -p bevy_synth --release -- --quality balanced --synthesis-models triposg --rmbg-model rmbg2
```

### TripoSG + RMBG-1.4

```powershell
cargo run -p bevy_synth --release -- --quality balanced --synthesis-models triposg --rmbg-model rmbg14
```

### Enable both synthesis backends (Trellis preferred, TripoSG fallback)

```powershell
cargo run -p bevy_synth --release -- --quality balanced --synthesis-models trellis,triposg --rmbg-model rmbg2 --trellis-quality medium
```

## burn_synth CLI Progress Logging

- `burn_synth` now emits structured sampler/stage progress through shared runtime events.
- CLI flags:
  - `--progress off|stages|steps` (`steps` default)
  - `--progress-every <N>` emits every N sampler steps (first/last always emitted)
- Example:

```powershell
cargo run -p burn_synth --features runtime,wgpu -- --quality balanced mesh --input docs/output_chair_bg_removed.png --progress steps --progress-every 5
```

- Log lines include:
  - stage start/finish timing
  - sampler step timing + ETA
  - sampler metadata (step counts, guidance/timestep info)

## MCP Server Mesh Output

Example `tools/call` arguments:

```json
{
  "name": "image_to_mesh",
  "arguments": {
    "input_image_path": "C:/data/input.png",
    "output_mesh_path": "C:/data/output.glb"
  }
}
```

Notes:
- `burn_synth_mcp` writes `.glb` only.
- If `output_mesh_path` has a non-`.glb` extension, the server rewrites it to `.glb`.

## Bevy Scene Control Bridge (Optional)

- `bevy_synth` can optionally ingest external scene commands from a JSON file:
  - `--mcp-scene-control-path <path-to-json>`
- This enables an external agent/process to update the live scene and persisted world cache.

Accepted command formats:

- Root array:
  - `[ { "type": "spawn_cached", ... }, ... ]`
- Envelope object:
  - `{ "commands": [ { "type": "spawn_cached", ... }, ... ] }`

Supported command types:

- `spawn_cached`
  - fields: `cache_key`, optional `translation`, `rotation`, `scale`, `select`
- `delete_by_cache_key`
  - fields: `cache_key`
- `delete_selected`
- `clear_selection`
- `set_camera`
  - fields: `translation`, `rotation`, optional `focus`, `yaw`, `pitch`, `radius`
- `save_cache`

## Runtime Cache Notes

- `bevy_synth` cache stores mesh payload + `.glb` artifact and persisted camera pose/orbit state.
- Native default cache root:
  - `%LOCALAPPDATA%/burn_synth/mesh_cache`

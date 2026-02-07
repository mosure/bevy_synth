# burn_synth 🔥🎛️🎹

[![GitHub License](https://img.shields.io/github/license/mosure/burn_synth)](https://raw.githubusercontent.com/mosure/burn_synth/main/LICENSE-MIT)
[![crates.io](https://img.shields.io/crates/v/burn_synth.svg)](https://crates.io/crates/burn_synth)


3d synthesis implemented in bevy and burn, view the [live demo](https://mosure.github.io/burn_synth/)

## example

| Input | Output |
| --- | --- |
| ![Input chair](docs/output_chair_bg_removed.png) | ![Output chair (rendered mesh)](docs/output_chair_render.png) |


## features

- [x] foreground segmentation
- [x] image to 3d
- [x] text to 3d
- [x] web demo
- [ ] text to image
- [ ] image to composite-3d
- [ ] many-3d to composite-3d
- [ ] image to 4d
- [ ] video to 4d


## setup

- download TripoSG + RMBG weights (or set `TRIPOSG_WEIGHTS_ROOT` / `RMBG_WEIGHTS_ROOT`)
- run the burnpack import tool `cargo run -p burn_3d_synth_tripo --bin triposg_import --features import`
- run the bevy app with the burnpack weights

## web/wasm

Build wasm artifacts into `docs/` (and clear `RUSTFLAGS` for wasm builds):

```powershell
./scripts/build_web.ps1
```

Then serve `docs/` with any static server (example):

```powershell
python -m http.server 8080 --directory docs
```


## model import

```
cargo run -p burn_3d_synth_tripo --bin triposg_import --features import
```


## references

- [assembler](https://assembler3d.github.io/)
- [burn_dino](https://github.com/mosure/burn_dino)
- [midi](https://huanngzh.github.io/MIDI-Page/)
- [trellis](https://github.com/microsoft/TRELLIS.2)
- [triposg](https://yg256li.github.io/TripoSG-Page/)

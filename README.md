# burn_3d_synth

[![GitHub License](https://img.shields.io/github/license/mosure/burn_3d_synth)](https://raw.githubusercontent.com/mosure/burn_3d_synth/main/LICENSE-MIT)
[![crates.io](https://img.shields.io/crates/v/burn_3d_synth.svg)](https://crates.io/crates/burn_3d_synth)


3d synthesis implemented in burn, view the [live demo](https://mosure.github.io/burn_3d_synth/)

## example

| Input | Output |
| --- | --- |
| ![Input chair](docs/output_chair_bg_removed.png) | ![Output chair (rendered mesh)](docs/output_chair_render.png) |


## features

- [x] image to 3d
- [x] text to 3d
- [ ] web demo
- [ ] text to image
- [ ] image to composite-3d
- [ ] many-3d to composite-3d


## setup

- download TripoSG + RMBG weights (or set `TRIPOSG_WEIGHTS_ROOT` / `RMBG_WEIGHTS_ROOT`)
- run the burnpack import tool `cargo run -p burn_3d_synth_tripo --bin triposg_import --features import`
- run the bevy app with the burnpack weights


## model import

```
cargo run -p burn_3d_synth_tripo --bin triposg_import --features import
```


## references

- [assembler](https://assembler3d.github.io/)
- [burn_dino](https://github.com/mosure/burn_dino)
- [midi](https://huanngzh.github.io/MIDI-Page/)
- [triposg](https://yg256li.github.io/TripoSG-Page/)

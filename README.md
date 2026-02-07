# burn_synth 🔥🎛️🎹

[![GitHub License](https://img.shields.io/github/license/mosure/burn_synth)](https://raw.githubusercontent.com/mosure/burn_synth/main/LICENSE-MIT)
[![crates.io](https://img.shields.io/crates/v/burn_synth.svg)](https://crates.io/crates/burn_synth)


3d synthesis implemented in bevy and burn, view the [live demo](https://mosure.github.io/burn_synth/)

## example

| Input | Output |
| --- | --- |
| ![Input chair](docs/output_chair_bg_removed.png) | ![Output chair (rendered mesh)](docs/output_chair_render.png) |


## features

- [x] editor
- [x] wasm demo


### media

| status  | input   | output       |
| ------  | ------- | ------------ |
| ✅      | image   | foreground   |
| ✅      | image   | 3d           |
| ✅      | text    | 3d           |
| ⬜      | text    | image        |
| ⬜      | image   | composite-3d |
| ⬜      | many-3d | composite-3d |
| ⬜      | image   | 4d           |
| ⬜      | video   | 4d           |
| ⬜      | text    | audio        |
| ⬜      | video   | audio        |





## setup

<!-- note: migrate this config section to be feature dependent, e.g. it is likely all models/features will not be used so please modularize the setup instructions/weight download -->

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
- [trellis](https://github.com/microsoft/TRELLIS.2)
- [triposg](https://yg256li.github.io/TripoSG-Page/)

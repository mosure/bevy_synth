# bevy_synth 🕊️🔥🎛️🎹

[![GitHub License](https://img.shields.io/github/license/mosure/lattice)](https://raw.githubusercontent.com/mosure/lattice/main/LICENSE-MIT)
[![crates.io](https://img.shields.io/crates/v/lattice.svg)](https://crates.io/crates/lattice)


asset synthesis implemented in bevy and burn, view the [live demo](https://mosure.github.io/lattice/)


## example

| Input | Output |
| --- | --- |
| ![Input chair](docs/output_chair_bg_removed.png) | ![Output chair (rendered mesh)](docs/output_chair_render.png) |


## features

- [x] editor
- [x] native mcp server
- [x] wasm demo


### media

|         | input   | output       |
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

- follow `docs/SETUP.md` for model-specific setup:
  - RMBG-1.4
  - RMBG-2.0
  - TripoSG
  - Trellis2
- canonical foreground model paths and expected files:
  - `crates/burn_foreground/assets/models/README.md`
- import tooling split by model family:
  - TripoSG: `cargo run -p burn_tripo --features import --bin triposg_import`
  - RMBG: `cargo run -p burn_foreground --features import --bin foreground_import -- --rmbg-model rmbg2 --quantization both`
  - Trellis2: `cargo run -p burn_trellis --features import --bin trellis2_import -- --quantization both`
- app default foreground model:
  - `bevy_synth` defaults to `rmbg2` when available (native fallback to `rmbg14`, wasm fallback to `rmbg14`)


## references

- [assembler](https://assembler3d.github.io/)
- [burn_dino](https://github.com/mosure/burn_dino)
- [lattice](https://arxiv.org/abs/2512.03052)
- [midi](https://huanngzh.github.io/MIDI-Page/)
- [trellis](https://github.com/microsoft/TRELLIS.2)
- [triposg](https://yg256li.github.io/TripoSG-Page/)


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

- see `docs/SETUP.md` for full setup and model import/runtime details
- supported models:
  - foreground: RMBG-1.4, RMBG-2.0
  - synthesis: TripoSG, Trellis2


## references

- [assembler](https://assembler3d.github.io/)
- [burn_dino](https://github.com/mosure/burn_dino)
- [lattice](https://arxiv.org/abs/2512.03052)
- [midi](https://huanngzh.github.io/MIDI-Page/)
- [trellis](https://github.com/microsoft/TRELLIS.2)
- [triposg](https://yg256li.github.io/TripoSG-Page/)


# bevy_synth 🕊️🔥🎛️🎹

[![GitHub License](https://img.shields.io/github/license/mosure/bevy_synth)](https://raw.githubusercontent.com/mosure/bevy_synth/main/LICENSE-MIT)
[![crates.io](https://img.shields.io/crates/v/bevy_synth.svg)](https://crates.io/crates/bevy_synth)


asset synthesis implemented in bevy and burn, view the [live demo](https://mosure.github.io/bevy_synth/)


## example

| Input | Output |
| --- | --- |
| ![Input chair](docs/output_chair_bg_removed.png) | ![Output chair (rendered mesh)](docs/output_chair_render.png) |



## features

- [x] editor
- [x] native mcp server
- [x] wasm demo
- [ ] quantize-aware kernels


### media

|         | input   | output       |
| ------  | ------- | ------------ |
| ✅      | image   | 3d           |
| ✅      | image   | foreground   |
| ✅      | image   | pbr          |
| ✅      | text    | 3d           |
| ⬜      | text    | image        |
| ⬜      | image   | composite-3d |
| ⬜      | many-3d | composite-3d |
| ⬜      | image   | 4d           |
| ⬜      | video   | 4d           |
| ⬜      | text    | audio        |
| ⬜      | video   | audio        |



## setup


### install

```bash
# burn_synth CLI
cargo install burn_synth

# bevy_synth app
cargo install bevy_synth

# burn_synth MCP stdio server
cargo install burn_synth_mcp
```

#### usage:

```bash
# burn_synth: run image -> GLB synthesis
burn_synth mesh \
  --quality balanced \
  --input docs/output_chair_bg_removed.png \
  --output /tmp/chair.glb

# bevy_synth: launch interactive app
bevy_synth --quality balanced

# burn_synth_mcp: start MCP stdio server
burn_synth_mcp
```



### upstream

- see `docs/SETUP.md` for full setup and model import/runtime details
- supported models:
  - foreground: RMBG-1.4, RMBG-2.0
  - synthesis: TripoSG, Trellis2

> note: pre-trained model weights have separate license



### hardware recommendation

- 16GB VRAM


## references

- [assembler](https://assembler3d.github.io/)
- [burn_dino](https://github.com/mosure/burn_dino)
- [lattice](https://arxiv.org/abs/2512.03052)
- [midi](https://huanngzh.github.io/MIDI-Page/)
- [trellis](https://github.com/microsoft/TRELLIS.2)
- [triposg](https://yg256li.github.io/TripoSG-Page/)

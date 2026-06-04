# Sparse Tensor Handoff: No Pre-Extraction Readback

Run ID: `20260228T140510Z_sparse_tensor_handoff_no_readback`

## Scope

- Added tensor-native sparse latent handoff from sparse-flow runtime to sparse-structure decoder on canonical WGPU non-hook path.
- Goal: remove remaining sparse-stage host readback before extraction boundary.

## Code Paths Updated

- `crates/burn_trellis/src/runtime_model/sparse_structure_flow.rs`
  - Added `sample_final_tensor(...)` on runtime impl.
  - Added `sample_final_tensor_wgpu(...)` on runtime enum.
- `crates/burn_trellis/src/runtime_model/sparse_structure_decoder.rs`
  - Added tensor-native latent entry `decode_to_coord_tensor_from_latent_tensor(...)`.
  - Added runtime API `decode_to_sparse_coords_wgpu_latent_tensor(...)`.
- `crates/burn_trellis/src/staged_pipeline_sampling.rs`
  - Canonical WGPU sparse stage now uses tensor-native handoff when hooks are off.

## Validation

### Build and targeted gate

- `cargo fmt --all`
- `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run`
- `cargo test -p burn_trellis --features runtime-model-wgpu canonical_wgpu_no_host_readback_before_extraction -- --nocapture`

All passed.

### Runtime sanity (`512_base`, strict, repeat=2)

Command:

```bash
cargo run -p burn_trellis --features runtime-model-wgpu --bin trellis2_run -- \
  --input tmp/upstream/TRELLIS.2/TRELLIS/assets/example_image/typical_vehicle_cart.png \
  --backend wgpu --quality low --strict-benchmark --require-runtime-model --seed 42 --repeat 2
```

Warm run (`run=2`) key metrics:

- `total_ms`: `62606.770838`
- `sparse_ms`: `3662.325854`
- `shape_slat_ms`: `16064.473721`
- `tex_slat_ms`: `8359.931712`
- `decode_ms`: `34401.966071`
- `host_readback_count`: `0`
- `host_readback_elements`: `0`

## Result

- Canonical WGPU sparse/decode path no longer performs sparse-stage host latent readback before extraction boundary.
- Remaining latency hotspot is still decode + sparse attention/MLP kernels, not host transfer overhead.

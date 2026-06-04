# W18: Sorted-Hash Neighbor Scan Window Tuning (WGPU)

Date: 2026-02-28
Scope: `crates/burn_flex_gmm/src/wgpu.rs`, `crates/burn_trellis/src/runtime_model/sparse_decoder_wgpu_ops.rs`

## Summary

This pass tightened the sorted-hash neighbor query scan window to reduce decode neighbor-map time on canonical WGPU, while preserving strict fail-fast behavior and device-resident decode flow.

Two linked changes are included:

1. ConvNeXt tensor-path neighbor reuse in decoder blocks (avoid rebuilding neighbor tensors per block for unchanged coords/kernel topology).
2. Sorted-hash query scan bound tune:
   - prior: `DEFAULT_NEIGHBOR_SORTED_HASH_MATCH_SCAN = 256`
   - interim: `32`
   - final in this pass: `8`

## Why

Decode hotspot telemetry showed sparse conv math is small while neighbor-map build/query dominates stage time. The sorted-hash query loop cost scales with `rows * kernel_rows * max_match_scan`; reducing `max_match_scan` is a high-leverage bounded tuning knob when collision chains are short in practice.

## Validation

### Build/Test

- `cargo fmt --all`
- `cargo test -p burn_flex_gmm --features wgpu-kernel neighbor_rows_ -- --nocapture`
- `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run`
- `cargo test -p burn_trellis --features runtime-model-wgpu canonical_wgpu_no_host_readback_before_extraction -- --nocapture`

All passed.

### Strict runtime sanity (repeat=2, warm pass = run 2)

Command:

```bash
cargo run -p burn_trellis --features runtime-model-wgpu --bin trellis2_run -- \
  --input docs/input_chair.jpg \
  --quality low \
  --backend wgpu \
  --strict-benchmark \
  --require-runtime-model \
  --seed 7 \
  --repeat 2 \
  --runtime-decoder-conv-telemetry
```

Warm pass (`run=2`) key metrics:

- `timings_ms.total = 52943.243711`
- `timings_ms.decode = 33366.977376`
- `timings_ms.decode_shape_decoder = 15211.768036`
- `timings_ms.decode_tex_decoder = 13092.816033`
- `timings_ms.decode_shape_wgpu_dispatches = 40`
- `timings_ms.decode_tex_wgpu_dispatches = 40`
- `timings_ms.host_readback_count = 0`

Neighbor-map telemetry (run 2):

- shape decoder: `device_hash_ms = 6662.87`, `device_builds = 12`
- tex decoder: `device_hash_ms = 13068.53`, `device_builds = 12`

## Comparison (immediate prior tune point, match_scan=32)

Prior reference run (`20260228`, same command/seed/repeat):

- warm `total = 52771.070705`
- warm `decode = 35046.778279`
- warm `decode_shape_decoder = 15014.438560`
- warm `decode_tex_decoder = 14317.120286`
- tex `device_hash_ms = 14266.83`

Observed delta (warm pass):

- total: `+172.17 ms` (noise-level drift)
- decode: `-1679.80 ms`
- shape decoder: `+197.33 ms`
- tex decoder: `-1224.30 ms`
- tex neighbor hash: `-1198.30 ms`

Interpretation:

- This tuning improved decode hotspot time, especially tex neighbor-map/query.
- End-to-end total remained roughly flat due variance outside decode (sparse/SLAT stage variance across runs).

## Notes

- Canonical path invariants remain satisfied: no host readback before extraction, strict fail-fast semantics retained.
- This is a bounded tuning step, not final closure of neighbor-build bottleneck.

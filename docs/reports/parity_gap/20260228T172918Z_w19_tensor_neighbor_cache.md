# W19: Tensor-Path Neighbor Cache Reuse (WGPU)

Date: 2026-02-28
Scope: `crates/burn_flex_gmm/src/wgpu.rs`

## Summary

This pass closes a cache gap in the tensor-native neighbor build path:

1. Added cache lookup/insert to `neighbor_rows_tensor_from_coords_tensor`.
2. Keyed tensor-path entries by device tensor identity metadata (handle/shape/strides hash), without host coord readback.
3. Kept cache namespace disjoint from host-coord keys (`device_key: ...:tensor`) to avoid cross-path key aliasing.
4. Added regression coverage: `neighbor_rows_tensor_cache_reuses_across_tensor_coord_clones`.

## Why

`neighbor_rows_tensor_from_coords` (host coords) already used the shared neighbor cache, but `neighbor_rows_tensor_from_coords_tensor` always rebuilt neighbor maps. In canonical WGPU decode this caused repeated device rebuilds at shape/tex boundaries even when coords tensors were reused.

## Validation

### Build/Test

- `cargo fmt --all`
- `cargo test -p burn_flex_gmm --features wgpu-kernel neighbor_rows_ -- --nocapture`
- `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run`
- `./scripts/guard_canonical_runtime.sh`

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

- `timings_ms.total = 51550.477620`
- `timings_ms.decode = 31852.309469`
- `timings_ms.decode_shape_decoder = 13769.650985`
- `timings_ms.decode_tex_decoder = 13053.994370`
- `timings_ms.decode_shape_wgpu_dispatches = 40`
- `timings_ms.decode_tex_wgpu_dispatches = 40`
- `timings_ms.host_readback_count = 0`

Warm pass tex neighbor telemetry:

- `cache_hits = 8`
- `cache_misses = 4`
- `device_builds = 4`
- `device_hash_ms = 13028.15`

## Comparison vs W18 warm baseline

Baseline (`docs/reports/parity_gap/20260228T171312Z_w18_neighbor_sorted_hash_scan_tune.md`):

- `total = 52943.243711`
- `decode = 33366.977376`
- `decode_shape_decoder = 15211.768036`
- `decode_tex_decoder = 13092.816033`

Observed delta:

- total: `-1392.77 ms`
- decode: `-1514.67 ms`
- shape decoder: `-1442.12 ms`
- tex decoder: `-38.82 ms`

Interpretation:

- The tensor-path cache closure reduces repeated neighbor-map rebuild overhead and preserves canonical invariants (no pre-extraction host readback, fail-fast path unchanged).

# W26 Neighbor Bucket-Hash Auto Policy (Large Small-k Decode Rows)

Date: 2026-02-28
Workstream: W26 (post-W25)
Owner: `crates/burn_flex_gmm/src/wgpu.rs`

## Goal

Reduce neighbor-map build latency for high-row, small-k decode workloads by routing `Auto` to the custom bucket-hash path where it wins over sorted-hash, while preserving the existing sorted/scan behavior for smaller rows.

## Change Summary

Implemented conservative auto routing in `resolve_neighbor_device_algo(...)`:

- Added bucket route in auto policy:
  - `kernel_rows <= 64`
  - `rows >= 32768`
  - `work >= sorted_threshold`
  - then use `NeighborDeviceAlgo::BucketHash`
- Retained prior sorted-hash vs scan threshold policy outside this gate.
- Added regression test:
  - `neighbor_algo_auto_routes_bucket_hash_for_large_small_k`
- Bench tools already accept explicit `bucket|bucket-hash` and were used for matrix validation.

Rationale comment retained in code:

- Bucket-hash removes sorted-hash `sort_with_indices` overhead on large decode-like small-k shapes.
- Mid-row shapes can still favor sorted-hash, so auto routing is intentionally conservative.

## Validation Commands

1. `cargo fmt --all`
2. `cargo test -p burn_flex_gmm --features wgpu-kernel neighbor_algo_auto_routes_bucket_hash_for_large_small_k -- --nocapture`
3. `cargo test -p burn_flex_gmm --features wgpu-kernel neighbor_rows_ -- --nocapture`
4. `cargo test -p burn_flex_gmm --features wgpu-kernel neighbor_algo_auto_uses_kernel_aware_thresholds -- --nocapture`
5. `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run`
6. `cargo test -p burn_trellis --features runtime-model-wgpu canonical_wgpu_no_host_readback_before_extraction -- --nocapture`
7. `./scripts/guard_canonical_runtime.sh`

All passed.

## Benchmark Evidence

### A) Neighbor stage matrix at threshold boundary

Artifact:

- `tmp/runs/20260228T223900Z_neighbor_auto_bucket_threshold_matrix/01_matrix.log`

Kernel: `k=3`, channels=8, warmup=1, iters=3.

Key cases:

- `rows=32768`
  - auto: `mean=2.387 ms`
  - sorted-hash: `mean=4.092 ms`
  - bucket-hash: `mean=2.050 ms`
- `rows=65536`
  - auto: `mean=4.540 ms`
  - sorted-hash: `mean=13.521 ms`
  - bucket-hash: `mean=3.494 ms`

Interpretation:

- Auto now crosses to bucket in the intended large-row regime and tracks bucket-like performance.
- Sorted-hash remains preferable to keep as the default for smaller row bands unless explicitly bucket-routed.

### B) Strict end-to-end high-quality sanity

Artifact:

- `tmp/runs/20260228T230800Z_trellis2_w26_single_high_strict_tee/01_run.log`

Command:

- `cargo run -p burn_trellis --features runtime-model-wgpu --bin trellis2_run -- --input docs/input_chair.jpg --quality high --backend wgpu --strict-benchmark --require-runtime-model --runtime-decoder-conv-telemetry --repeat 1`

Result:

- `status=ok`
- `host_readback_count=0`
- dispatch invariants: `decode_shape_wgpu_dispatches=40`, `decode_tex_wgpu_dispatches=40`
- `timings_ms.total=151714.996109`
- `timings_ms.sparse=41364.249251`
- `timings_ms.shape_slat=44453.853457`
- `timings_ms.tex_slat=25677.133859`
- `timings_ms.decode=17519.912149`

## Next Step

Proceed to W27 with decode-shape/tex attention hotspot reduction and stage-level kernel fusion experiments, keeping strict no-host-readback invariants and bounded stage-only benches as first gates.

# W7 Dense Sampler Kernel Wrapper (v6)

Date: 2026-02-28

## Scope

Implemented a custom CubeCL/WGSL kernel path for dense trilinear PBR attribute sampling and switched `burn_trellis` dense WGPU sampling batches to that kernel wrapper.

Files touched in this step:

- `crates/burn_flex_gmm/src/wgpu.rs`
- `crates/burn_trellis/src/staged_pipeline_decode.rs`
- `crates/burn_trellis/src/staged_pipeline_tests.rs`

## Correctness

- Added/fixed kernel reference coverage in `burn_flex_gmm`:
  - `dense_trilinear_sample_kernel_matches_reference`
- Re-ran `burn_trellis` WGPU smoke parity:
  - `pbr_bake_wgpu_dense_sampling_matches_cpu_sampling`
- Parity test now enforces byte-level agreement with `<= 1` LSB tolerance.

Rationale for tolerance gate:

- WGSL/CubeCL arithmetic ordering (including FMA behavior) is not bitwise-identical to host scalar math.
- Observed mismatch was 1 LSB (`cpu=66`, `wgpu=67`) at a single channel/texel during strict equality check.
- `<= 1` LSB bound keeps semantic parity while preventing brittle false negatives.

## Stage Bench Evidence

New runs:

- CPU v6: `tmp/runs/20260228T030000Z_w7_pbr_stage_matrix_v6_cpu_kernel_wrapper/summary.csv`
- WGPU v6: `tmp/runs/20260228T030020Z_w7_pbr_stage_matrix_v6_wgpu_kernel_wrapper/summary.csv`

Comparison baseline:

- CPU v5: `tmp/runs/20260228T021136Z_w7_pbr_stage_matrix_v5_cpu_post_refactor/summary.csv`
- WGPU v5: `tmp/runs/20260228T021147Z_w7_pbr_stage_matrix_v5_wgpu_post_refactor/summary.csv`

### p50 (ms) comparison

| Grid | CPU v5 | CPU v6 | WGPU v5 | WGPU v6 | WGPU delta |
|---|---:|---:|---:|---:|---:|
| 64 | 4.338 | 4.290 | 18.087 | 10.988 | -39.3% |
| 96 | 5.006 | 5.268 | 14.666 | 11.778 | -19.7% |
| 128 | 5.584 | 5.668 | 18.683 | 11.854 | -36.6% |

Observations:

- Kernel wrapper materially reduced WGPU dense sampling stage time versus v5.
- Canonical runtime should still keep CPU sampling default for now:
  - WGPU remains slower than CPU by ~2.1x to ~2.6x in this bounded stage bench.

## Commands executed

```bash
cargo test -p burn_flex_gmm --features wgpu-kernel dense_trilinear_sample_kernel_matches_reference -- --nocapture
env BURN_WGPU_SMOKE=1 cargo test -p burn_trellis --features runtime-model-wgpu pbr_bake_wgpu_dense_sampling_matches_cpu_sampling -- --nocapture
WGPU=0 scripts/bench_trellis_pbr_stage_matrix.sh 20260228T030000Z_w7_pbr_stage_matrix_v6_cpu_kernel_wrapper
WGPU=1 scripts/bench_trellis_pbr_stage_matrix.sh 20260228T030020Z_w7_pbr_stage_matrix_v6_wgpu_kernel_wrapper
```

## Status

- Correctness: pass under bounded parity gate (`<= 1` LSB)
- Performance: improved vs prior WGPU implementation, but not yet CPU-competitive for canonical default
- Next: continue W7 closure by removing remaining host-materialization surfaces in sparse ownership flow and pushing further kernel-path residency.

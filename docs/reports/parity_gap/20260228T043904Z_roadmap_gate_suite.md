# Parity Gap Report: Roadmap Gate Suite Closure

- run_id: `20260228T043904Z_roadmap_gate_suite`
- date_utc: `2026-02-28`
- scope: `W2-W8 closure validation`

## Goal

Run the strict roadmap gate suite end-to-end, ensure required alias tests exist and execute, and close remaining non-passing harness items.

## Implementation updates in this pass

1. Added missing roadmap gate test alias `decoder_guide_subdivision_tensor_handoff_parity` in:
   - `crates/burn_trellis/src/runtime_model/sparse_decoder_tests.rs`
2. Added missing roadmap gate test alias `canonical_wgpu_no_host_readback_before_extraction` in:
   - `crates/burn_trellis/src/staged_pipeline_tests.rs`
3. Re-ran full roadmap gate suite and smoke-enabled guide-handoff execution.

## Commands executed

```bash
./scripts/guard_canonical_runtime.sh
cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run
cargo check -p burn_flex_gmm --features wgpu-kernel
cargo test -p burn_trellis --features runtime-model-wgpu sparse_structure_coord_select_token_cap_boundary_parity -- --nocapture
cargo test -p burn_trellis --features runtime-model-wgpu cascade_quantize_token_cap_boundary_parity -- --nocapture
cargo test -p burn_trellis --features runtime-model-wgpu decoder_guide_subdivision_tensor_handoff_parity -- --nocapture
cargo test -p burn_trellis --features runtime-model-wgpu canonical_wgpu_no_host_readback_before_extraction -- --nocapture
cargo test -p burn_trellis --features runtime-model-wgpu decode_pbr_device_path_sparse_hole_failfast -- --nocapture
cargo test -p burn_trellis --features runtime-model-wgpu runtime_decode_tests::runtime_decode_device_gate_allows_host_only_inputs -- --nocapture
BURN_WGPU_SMOKE=1 cargo test -p burn_trellis --features runtime-model-wgpu pbr_bake_wgpu_dense_sampling_matches_cpu_sampling -- --nocapture
cargo test -p burn_flex_gmm --features wgpu-kernel neighbor_hash_parallel_matches_scan_parity -- --nocapture
cargo test -p burn_flex_gmm --features wgpu-kernel neighbor_hash_parallel_collision_stress_bounded -- --nocapture
cargo test -p burn_flex_gmm --features wgpu-kernel sparse_conv_hotspot_kernel_matches_reference_parity -- --nocapture
cargo test -p burn_flex_gmm --features wgpu-kernel neighbor_algo_auto_uses_kernel_aware_thresholds -- --nocapture
BURN_WGPU_SMOKE=1 cargo test -p burn_trellis --features runtime-model-wgpu decoder_guide_subdivision_tensor_handoff_parity -- --nocapture
```

## Results

- `guard_canonical_runtime`: pass
- `burn_trellis` check (`runtime-model-wgpu`): pass
- `burn_flex_gmm` check (`wgpu-kernel`): pass
- all targeted roadmap gate tests: pass
- smoke-enabled guide tensor-handoff test: pass

## Invariant summary

- canonical WGPU fail-fast behavior: pass
- no new canonical `.into_data()` readback entries outside baseline: pass
- roadmap gate symbol coverage includes sparse/cascade/decode/hash/conv and decoder guide handoff: pass

## Notes

- Known warning remains during `burn_trellis` checks/tests:
  - `SparseSubdivisionLogits::from_device_tensors` reported as dead code in non-test builds.
  - This does not block the roadmap gate suite.

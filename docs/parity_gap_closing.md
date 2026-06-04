# TRELLIS.2 Dense/Padding vs Sparse Parity Gap Closure Roadmap

Status: execution roadmap for the current uncommitted branch. All W0-W8 success criteria are currently passing on this branch as of 2026-02-28.

Scope owner: `crates/burn_trellis` canonical runtime-model WGPU path.

Primary objective: close remaining correctness + performance gap to upstream TRELLIS.2 semantics while enforcing a strict device-resident canonical path.

## Execution Ledger

Ledger policy:

- Update this section after each completed workstream gate.
- Record both implementation status and verification evidence.
- Keep roadmap topology unchanged; ledger is additive tracking only.

| Workstream | Status | Completed items | Verification |
|---|---|---|---|
| W0 Baseline lock + guardrails | Completed (2026-02-27) | Added canonical runtime guard script (`scripts/guard_canonical_runtime.sh`), baseline allowlists (`scripts/guards/*.baseline`), CI wiring in `.github/workflows/test.yml`, baseline report template (`docs/reports/parity_gap/TEMPLATE.md`), and initial run report (`docs/reports/parity_gap/20260227T212950Z_w0_guardrails.md`). | `./scripts/guard_canonical_runtime.sh` passed; `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run` passed; `cargo check -p burn_flex_gmm --features wgpu-kernel` passed. |
| W1 Config and feature hardening | Completed (2026-02-27) | Replaced runtime-model debug env reads with typed runtime config (`runtime_model/runtime_config.rs`), replaced staged runtime debug/decoder telemetry env toggles with typed run-config fields, and wired `TrellisRunOptions` -> `TrellisStageRunConfig` -> runtime toggles. Expanded canonical guard scope to include staged runtime files. Added verification report `docs/reports/parity_gap/20260227T213142Z_w1_config_hardening.md`. | `./scripts/guard_canonical_runtime.sh` passed after scope expansion; `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run` passed after wiring changes; `cargo check -p burn_synth` passed for downstream option-surface compatibility. |
| W2 Device-first sparse ownership | Completed (2026-02-28) | Added `runtime_model/types/*` scaffold (`SparseBatchLayoutDevice`, `VarLenTensorDevice`, `SparseTensorDevice`, extraction boundary helpers) and exported via `runtime_model::types`. Extended staged sampling/runtime decode to carry tex coord tensors device-native (`TexSLatSample::coords_wgpu`) instead of relying on shape coord tensor fallback in canonical WGPU flow. Added bridge conversions `VarLenTensorOwned::as_device_owned` and `SparseTensorOwned::as_device_owned` to connect existing hybrid types to new device ownership types, plus runtime API bridge methods returning `types::*Device` directly from WGPU tensor assembly paths. Switched staged shape/tex WGPU sampling to runtime tensor-input API (`sample_sparse_rows_with_trace_wgpu_inputs`) so staging no longer manually assembles owned sparse/varlen surfaces on canonical WGPU path. Wired staged runtime decode to tensor-native decoder entrypoints (shape/tex row tensorization + `decode_*_with_tensors`) so canonical WGPU decode avoids host row completion in `from_latent` and downstream decoder math. Added optional device row ownership on stage samples (`ShapeSLatSample::features_wgpu`, `TexSLatSample::features_wgpu`) and switched runtime decode to prefer those tensors over host row tensorization on canonical device path. Added sparse-flow row trace tensor bridge (`SparseFlowRowTrace::{samples_wgpu,step_*_wgpu}`) and switched staged shape/tex feature tensor handoff to prefer trace device tensors (device-side denorm+pad) over host row re-upload. Added decoder tensor-native cascade upsample entrypoint (`upsample_coords_result_with_tensors`) and switched canonical WGPU cascade upsample to require/use low-res device row tensors (`shape_lr.features_wgpu`) instead of host rows. Removed tex-stage canonical host-row assumption by allowing concat conditioning to come from `shape_slat.features_wgpu` with explicit host fallback-only path; added tensor-native concat build helper and host-tolerant shape-cond host surface population. Added explicit sparse-row trace host materialization control (`materialize_host_rows`) and switched staged canonical WGPU shape/tex sampling to disable host row materialization when trace capture is off. Replaced device-path single-batch sparse layout assumption (`0..rows`) with batch-range derivation from `coords_wgpu` batch column (shape/tex staged sampling), preserving grouped-batch validation semantics for real batched ownership flows. Propagated sparse layout metadata through staged outputs (`SparseStructureSample.layout`, `ShapeSLatSample.layout`, `TexSLatSample.layout`) so canonical shape/tex WGPU handoff consumes owned layout directly instead of re-deriving from coord tensors on normal path. Added targeted unit tests for host-only refusal semantics and batch-layout helper coverage. Tightened sparse-structure sampled layout derivation to prefer `coords_wgpu` batch extraction over `0..rows` fallback assumptions when host coords are absent, and removed tex-slat canonical WGPU host concat rescue by requiring device shape rows for concat tensor assembly. Added verification reports `docs/reports/parity_gap/20260227T213351Z_w2_types_scaffold.md`, `docs/reports/parity_gap/20260227T214223Z_w2_tex_coord_tensor_handoff.md`, `docs/reports/parity_gap/20260227T214400Z_w2_tex_coord_tensor_unit_test.md`, `docs/reports/parity_gap/20260227T214621Z_w2_device_bridge_conversions.md`, `docs/reports/parity_gap/20260227T214732Z_w2_runtime_device_bridge_api.md`, `docs/reports/parity_gap/20260227T220028Z_w2_sampling_device_api_switch.md`, `docs/reports/parity_gap/20260227T221115Z_w2_decode_tensor_entrypoints.md`, `docs/reports/parity_gap/20260227T221542Z_w2_feature_tensor_handoff.md`, `docs/reports/parity_gap/20260227T222345Z_w2_trace_tensor_bridge.md`, `docs/reports/parity_gap/20260227T222711Z_w2_cascade_tensor_upsample.md`, `docs/reports/parity_gap/20260227T223151Z_w2_tex_concat_device_path.md`, `docs/reports/parity_gap/20260227T223717Z_w2_host_row_materialization_gate.md`, `docs/reports/parity_gap/20260228T032438Z_w2_device_layout_derivation.md`, and `docs/reports/parity_gap/20260228T033318Z_w2_layout_propagation_sparse_shape_tex.md`, and `docs/reports/parity_gap/20260228T033755Z_w2_sparse_layout_device_extract_and_tex_concat_gate.md`. | `./scripts/guard_canonical_runtime.sh` passed; `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run` passed; `cargo check -p burn_synth` passed in prior W2 pass; targeted tests `sparse_coord_cap_requires_explicit_override`, `device_owned_conversion_requires_device*`, `sample_sparse_rows_trace_uses_single_host_readback_when_capturing_snapshots`, `sparse_layout_from_batch_ids_tracks_real_batched_ranges`, `sparse_layout_from_batch_ids_rejects_non_grouped_rows`, `sparse_layout_from_coords_tracks_real_batched_ranges`, `runtime_decode_tests::runtime_decode_device_gate_allows_host_only_inputs`, and `runtime_decode_tests::sanitize_mesh_geometry_removes_non_finite_vertices_and_reindexes_faces` passed under `cargo test -p burn_trellis --features runtime-model-wgpu`. |
| W3 Sparse-structure + cascade tensor-native closure | Completed (2026-02-28) | Added tensor-native sparse-structure coord selection helper (`select_positive_coord_tensor`) to centralize threshold/select/sort/cap behavior without host-finalization logic, removed canonical staged-sampling cond-tensor debug readbacks and sparse-structure logits tensor debug dump from hot-path modules, moved sparse-structure + sparse-decoder tensor readbacks through `runtime_model/types/extraction.rs` boundary helpers, and tightened canonical `.into_data()` guard baseline accordingly. Added sparse-structure cap-boundary/empty-mask parity unit tests. Removed canonical staged sparse/cascade coord-layout host readback surfaces by deriving single-batch layout from device row counts in staged sampling and cascade handoff, and deleted the now-unused staged `sparse_layout_from_coords_wgpu` readback helper. Added verification reports `docs/reports/parity_gap/20260227T224433Z_w3_sparse_coord_select_guard_tighten.md`, `docs/reports/parity_gap/20260227T230012Z_w4_tensor_only_decode_apis.md`, and `docs/reports/parity_gap/20260227T230258Z_w3_extraction_boundary_readback_cleanup.md`, and `docs/reports/parity_gap/20260228T042635Z_w3_w5_w7_nonpassing_remediation.md`. | `cargo test -p burn_trellis --features runtime-model-wgpu sparse_structure_coord_select_cap_boundary_parity -- --nocapture` passed; `cargo test -p burn_trellis --features runtime-model-wgpu sparse_structure_coord_select_empty_mask_returns_empty_coords -- --nocapture` passed; `cargo test -p burn_trellis --features runtime-model-wgpu cascade_token_budget_accepts_equal_token_count_without_backoff -- --nocapture` passed; `cargo test -p burn_trellis --features runtime-model-wgpu runtime_decode_tests::sanitize_mesh_geometry_removes_non_finite_vertices_and_reindexes_faces -- --nocapture` passed; `./scripts/guard_canonical_runtime.sh` passed; `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run` passed. |
| W4 Decoder host-completion elimination | Completed (2026-02-28) | Tightened canonical decoder entry contract in `sparse_decoder_runtime_impl.rs` to require device-backed coords+rows (`decode_with_tensors`) for canonical WGPU path, removed coord-tensor+host-row bridge APIs (`decode_with_coords_tensor`, `upsample_coords_result_with_coords_tensor`, and wrapper variants in FDG/Tex runtimes), required tensor rows whenever cascade upsample uses tensor coords, and delegated `upsample_coords_sparse` (WGPU build) to tensor-native upsample path so host subdivision completion logic is no longer executed there. Removed staged runtime decode host-branch selection in canonical WGPU mode (`staged_pipeline_runtime_decode.rs`): shape/tex decode now always use tensor-native decoder calls and fail fast when device shape coords are missing. Tightened canonical staged runtime decode requirements further: decode now fails fast when device shape/tex feature tensors are missing and when tex coord tensor is missing (removed shape-coord fallback), with host row tensorization fallback fully removed. Tightened staged WGPU sampling handoff to require device trace rows (`trace.samples_wgpu`) for shape/tex feature tensors and removed host-row tensor rebuild fallback. Switched staged runtime decode mode selection from compile-time `runtime-model-wgpu` gating to runtime tensor-residency gating: canonical WGPU decode remains strict fail-fast once device tensors are present, while explicit host decode runs (no device tensors) are preserved even when crate is built with `runtime-model-wgpu`. Added regression coverage (`runtime_decode_device_gate_allows_host_only_inputs`) so this gate remains runtime-driven. Added verification reports `docs/reports/parity_gap/20260227T224658Z_w4_decoder_tensor_input_gate.md`, `docs/reports/parity_gap/20260227T230012Z_w4_tensor_only_decode_apis.md`, `docs/reports/parity_gap/20260227T230258Z_w3_extraction_boundary_readback_cleanup.md`, `docs/reports/parity_gap/20260227T231221Z_w4_upsample_host_entry_tensor_delegate.md`, `docs/reports/parity_gap/20260228T005847Z_w4_canonical_wgpu_decode_host_branch_removal.md`, `docs/reports/parity_gap/20260228T023322Z_w4_runtime_decode_device_requirements.md`, and `docs/reports/parity_gap/20260228T025147Z_w4_runtime_decode_backend_mode_selection.md`. | `cargo fmt --all` passed; `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run` passed; `cargo test -p burn_trellis --features runtime-model-wgpu runtime_decode_tests::runtime_decode_device_gate_allows_host_only_inputs -- --nocapture` passed; `cargo test -p burn_trellis --features runtime-model-wgpu runtime_decode_tests::sanitize_mesh_geometry_removes_non_finite_vertices_and_reindexes_faces -- --nocapture` passed; `cargo test -p burn_trellis --features runtime-model-wgpu cascade_token_budget_accepts_equal_token_count_without_backoff -- --nocapture` passed; `./scripts/guard_canonical_runtime.sh` passed; `env BURN_WGPU_SMOKE=1 cargo test -p burn_trellis --features runtime-model-wgpu pbr_bake_wgpu_dense_sampling_matches_cpu_sampling -- --nocapture` passed in prior W4 pass. |
| W5 Neighbor hash parallelization | Completed (2026-02-28) | Added hash-build probe/failure telemetry and fail-fast diagnostics in `burn_flex_gmm/src/wgpu.rs` (`HASH_BUILD_STAT_*`, `device_hash_probe_*`, `device_hash_insert_fail_rows`) and wired telemetry into stage logs. Because true parallel CAS insertion remains blocked by upstream `cubecl-spirv` (`Atomic should have a scope registered`), implemented an alternate non-CAS device path for large workloads: sorted-hash neighbor query (`neighbor_coord_hash_kernel` + GPU `sort_with_indices` + `neighbor_rows_from_sorted_hash_kernel` binary-search query). Added explicit benchmark/debug algo routing API (`NeighborDeviceAlgoPreference` + `neighbor_rows_tensor_from_coords_with_algo`) and kernel-aware auto-threshold tuning from bounded stage runs. Added stage-only benchmark binary (`tool/neighbor_stage_bench.rs`) for safe tuning loops, plus parity/threshold tests (`neighbor_rows_sorted_hash_matches_scan_reference`, `neighbor_algo_auto_uses_kernel_aware_thresholds`). Updated `neighbor_rows_auto_matches_serial_hash_table_backend` expectations to reflect explicit-algo cache-bypass semantics while preserving row-output parity checks. Added verification reports `docs/reports/parity_gap/20260227T231508Z_w5_hash_build_chunking_scaffold.md`, `docs/reports/parity_gap/20260227T232259Z_w5_parallel_hash_insert_kernel.md`, `docs/reports/parity_gap/20260227T232454Z_w5_parallel_hash_failfast_diagnostics.md`, `docs/reports/parity_gap/20260227T233812Z_w5_hash_probe_telemetry_and_cas_blocker.md`, `docs/reports/parity_gap/20260227T234634Z_w5_sorted_hash_parallel_query.md`, and `docs/reports/parity_gap/20260227T235824Z_w5_stage_bench_and_threshold_tuning.md`, and `docs/reports/parity_gap/20260228T042635Z_w3_w5_w7_nonpassing_remediation.md`. | `cargo check -p burn_flex_gmm --features wgpu-kernel` passed; `cargo test -p burn_flex_gmm --features wgpu-kernel neighbor_rows_sorted_hash_matches_scan_reference -- --nocapture` passed; `cargo test -p burn_flex_gmm --features wgpu-kernel neighbor_algo_auto_uses_kernel_aware_thresholds -- --nocapture` passed; `cargo test -p burn_flex_gmm --features wgpu-kernel neighbor_rows_device_hash_matches_scan -- --nocapture` passed; `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run` passed; `./scripts/guard_canonical_runtime.sh` passed. |
| W6 Sparse conv hotspot closure | Completed (2026-02-28) | Added explicit sparse-conv auto-schedule resolver API (`resolve_sparse_wgpu_forward_config`) with resolved variant/split telemetry surface, tuned split-k auto thresholds from bounded stage runs, made auto fused-oc4 selection conservative (baseline-first for common decode shapes), updated stage/criterion benchmark tooling to use explicit typed routing instead of legacy env knobs, removed remaining env-toggle routing from `burn_flex_gmm/src/wgpu.rs` test coverage in favor of explicit typed algo/config selection, and implemented single-group shape-specialized sparse-conv kernels (baseline/fused, split and non-split) with selector wiring + telemetry (`single_group_specialized_calls`). Added hot-shape fused auto route for `rows=4096` single-group low-inner-work profiles and capped auto split-k for very large row counts (`rows>=16384`) to avoid oversized split overhead. Tightened auto policy using phase evidence: fused hot-shape path now caps to low-OC (`out_channels_per_group<=128`), high-OC decode-like workloads (`rows>=8192 && out_channels_per_group>=256`) clamp auto split-k to `<=2`, and borderline fused auto selection is further restricted via raised fused output-work threshold (`DEFAULT_SPARSE_WGPU_FUSED_AUTO_MIN_OUTPUT_WORK=2_300_000`). Added reusable phase-benchmark script `scripts/bench_sparse_conv_phase_matrix.sh` with new `auto_vs_best.csv` output and run artifacts (`tmp/runs/20260228T004120Z_w6_phase_matrix_v1`, `tmp/runs/20260228T004120Z_w6_sparse_conv_phase_matrix_tool_after`, `tmp/runs/20260228T004541Z_w6_phase_matrix_v2_after_tune_rebuild`, `tmp/runs/20260228T010259Z_sparseconv_wgpu_w6_targeted_tune_rerun`, `tmp/runs/20260228T010347Z_sparseconv_wgpu_w6_focus_8192_oc256`, `tmp/runs/20260228T010435Z_sparseconv_wgpu_w6_auto_verify_8192_oc256`, `tmp/runs/20260228T010515Z_w6_sparse_conv_phase_matrix_after_borderline_tune`). Added verification reports `docs/reports/parity_gap/20260228T002508Z_w6_sparse_conv_autoscheduler.md`, `docs/reports/parity_gap/20260228T004120Z_w6_sparse_conv_phase_matrix_and_hotshape_tune.md`, and `docs/reports/parity_gap/20260228T005228Z_w6_scheduler_tighten_and_bench_blocker.md`. | `cargo fmt --all` passed; `cargo check -p burn_flex_gmm --features wgpu-kernel` passed; `cargo check -p burn_flex_gmm --features wgpu-kernel --benches` passed in earlier W6 pass; targeted tests `sparse_conv_auto_schedule_*`, `neighbor_rows_device_hash_matches_scan*`, `wgpu_single_group_specialized_kernel_matches_cpu_flex_path`, `wgpu_fused_oc4_matches_baseline_output`, `wgpu_splitk_matches_default_kernel_output`, `wgpu_fused_splitk_matches_baseline_output` passed; new scheduler tests `sparse_conv_auto_schedule_keeps_baseline_for_rows4096_when_oc_group_is_high`, `sparse_conv_auto_schedule_caps_splitk_for_high_oc_decode_shape`, and `sparse_conv_auto_schedule_keeps_baseline_for_borderline_fused_output_work_shape` passed; stage sanity run `tmp/runs/20260228T003606Z_w6_single_group_specialized_sanity` confirms `single_group_specialized_calls>0`; matrix runs emitted `summary.csv` + `best_by_case.csv` + `auto_vs_best.csv`; previously blocked targeted sweep now reran successfully after VRAM cleanup; `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run` and `./scripts/guard_canonical_runtime.sh` remain green. |
| W7 Decode/PBR GPU kernel closure | Completed (2026-02-28) | Kickoff hotspot cut in `staged_pipeline_decode.rs`: switched PBR voxel-attribute lookup map to `FxHasher`-backed `HashMap` (`rustc-hash`) for sparse lookup-heavy inner loop while preserving strict sparse-hole semantics and no-rescue behavior. Generalized `sample_voxel_attr` map signature to support any hasher so call sites/tests remain compatible. Added bounded PBR stage benchmark harness: benchmark-report test (`pbr_bake_benchmark_report`) plus matrix driver script (`scripts/bench_trellis_pbr_stage_matrix.sh`) with machine-readable `summary.csv`; captured matrix evidence in `tmp/runs/20260228T011803Z_w7_pbr_stage_matrix_v1`. Optimized trilinear sparse sampling inner loop by replacing nested corner loops with fixed 8-corner accumulation in `sample_voxel_attr`; captured post-change matrix evidence in `tmp/runs/20260228T012033Z_w7_pbr_stage_matrix_v2_unrolled_sample` showing directional p50 reductions across grid 64/96/128. Added adaptive dense voxel lookup backend (`VoxelAttrLookup::{Dense,Sparse}` with bounded dense volume threshold) and wired PBR sampling through lookup dispatch; added dense-vs-sparse sampling parity test and captured matrix v3 in `tmp/runs/20260228T013922Z_w7_pbr_stage_matrix_v3_dense_lookup` with further directional p50 reduction over v2. Prototyped a tensor-batched WGPU dense sampler path behind explicit opt-in (`prefer_wgpu_sampling`) plus `TRELLIS2_PBR_BENCH_WGPU` bench switch, and added smoke parity coverage (`pbr_bake_wgpu_dense_sampling_matches_cpu_sampling`); bounded matrix evidence (`tmp/runs/20260228T020112Z_w7_pbr_stage_matrix_v4_cpu_regression`, `tmp/runs/20260228T020121Z_w7_pbr_stage_matrix_v4_wgpu_dense_sampler`) showed WGPU was slower due repeated dense lookup upload overhead. Follow-up v5 refactor introduced persistent per-bake dense lookup tensors (`DenseVoxelWgpuSampler`) to remove repeated uploads and improved WGPU p50 materially (`48.570->18.087`, `32.918->14.666`, `33.586->18.683` for grid 64/96/128) with runs `tmp/runs/20260228T021136Z_w7_pbr_stage_matrix_v5_cpu_post_refactor` and `tmp/runs/20260228T021147Z_w7_pbr_stage_matrix_v5_wgpu_post_refactor`; canonical runtime decode now prefers WGPU dense sampling whenever decode is tensor-native/device mode, while explicit host decode mode keeps CPU sampling. Added a custom CubeCL dense-trilinear kernel wrapper (`dense_trilinear_sample_attrs_wgpu`) and routed batched WGPU dense sampling through it; v6 matrix runs (`tmp/runs/20260228T030000Z_w7_pbr_stage_matrix_v6_cpu_kernel_wrapper`, `tmp/runs/20260228T030020Z_w7_pbr_stage_matrix_v6_wgpu_kernel_wrapper`) reduced WGPU p50 further (`18.087->10.988`, `14.666->11.778`, `18.683->11.854` for grid 64/96/128) while CPU remained ~`4.3-5.7ms`; canonical runtime still routes device decode through WGPU sampling for parity-path residency despite this phase-level perf gap. Added verification reports `docs/reports/parity_gap/20260228T011016Z_w7_pbr_lookup_fast_hasher.md`, `docs/reports/parity_gap/20260228T011803Z_w7_pbr_stage_bench_matrix.md`, `docs/reports/parity_gap/20260228T012033Z_w7_pbr_trilinear_unroll_and_matrix_v2.md`, `docs/reports/parity_gap/20260228T013948Z_w7_dense_lookup_matrix_v3.md`, `docs/reports/parity_gap/20260228T020121Z_w7_wgpu_dense_sampler_probe_and_gate.md`, and `docs/reports/parity_gap/20260228T021147Z_w7_wgpu_dense_sampler_persistent_lookup_v5.md`, and `docs/reports/parity_gap/20260228T030020Z_w7_dense_sampler_kernel_v6.md`, and `docs/reports/parity_gap/20260228T042635Z_w3_w5_w7_nonpassing_remediation.md`. | `cargo fmt --all` passed; `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run` passed; targeted tests `sample_voxel_attr_returns_none_for_sparse_holes`, `sample_voxel_attr_returns_value_when_supported`, `dense_voxel_lookup_sampling_matches_sparse_hash_sampling`, `pbr_bake_produces_textures_and_uvs`, `pbr_quantization_tracks_float_buffers`, `pbr_bake_benchmark_report`, `dense_trilinear_sample_kernel_matches_reference` (under `cargo test -p burn_flex_gmm --features wgpu-kernel`), and `pbr_bake_wgpu_dense_sampling_matches_cpu_sampling` (with `BURN_WGPU_SMOKE=1`, enforcing <=1 LSB byte parity tolerance) passed; `bash -n scripts/bench_trellis_pbr_stage_matrix.sh` passed; matrix runs emitted `summary.csv` for v1/v2/v3 plus v4/v5/v6 CPU/WGPU comparisons; `./scripts/guard_canonical_runtime.sh` passed. |
| W8 Harness closure + release gates | Completed (2026-02-28) | Added explicit roadmap-gate aliases to stabilize harness invariants across crates, including `decoder_guide_subdivision_tensor_handoff_parity` and `canonical_wgpu_no_host_readback_before_extraction` in addition to prior sparse/cascade/decode/hash/conv aliases. Added remediation report `docs/reports/parity_gap/20260228T042635Z_w3_w5_w7_nonpassing_remediation.md` and closure report `docs/reports/parity_gap/20260228T043904Z_roadmap_gate_suite.md`. | `./scripts/guard_canonical_runtime.sh` passed; `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run` passed; `cargo check -p burn_flex_gmm --features wgpu-kernel` passed; roadmap-gate tests for sparse-structure token cap, cascade boundary, decoder guide tensor handoff, canonical no-host-readback guard, decode sparse-hole failfast, neighbor hash parity/collision, and sparse-conv hotspot parity all passed. |
| W9 Sparse latent tensor handoff (post-W8) | Completed (2026-02-28) | Added canonical WGPU tensor-native sparse latent handoff from sparse-flow to sparse-structure decode: runtime sparse-flow now exposes final latent tensor output (`sample_final_tensor_wgpu`), sparse-structure decoder accepts tensor-native latent input (`decode_to_sparse_coords_wgpu_latent_tensor`), and staged sparse sampling switches to tensor handoff when hook capture is off. | `cargo fmt --all` passed; `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run` passed; `cargo test -p burn_trellis --features runtime-model-wgpu canonical_wgpu_no_host_readback_before_extraction -- --nocapture` passed; strict sanity run (`--quality low --repeat 2`) shows warm run `host_readback_count=0`, `host_readback_elements=0`, and `total_ms=62606.770838`. Evidence: `docs/reports/parity_gap/20260228T140510Z_sparse_tensor_handoff_no_readback.md`. |
| W10 Sparse-flow RoPE custom kernel kickoff | Completed (2026-02-28) | Added first custom sparse-flow attention kernel in `burn_flex_gmm`: `rope_rotate_pairs_kernel` + wrapper `rope_rotate_pairs_wgpu` (single-dispatch pair rotation). Integrated canonical runtime-model WGPU path (`WgpuRuntimeBackend`) in `sparse_structure_flow.rs` so RoPE pair rotation uses the custom kernel and fails fast if kernel launch fails. Added explicit bridge behavior for fusion WGPU backend used in tests (no kernel path there) to preserve compile/test coverage while keeping canonical raw WGPU runtime on custom-kernel path. | `cargo fmt --all` passed; `cargo test -p burn_flex_gmm --features wgpu-kernel rope_rotate_pairs_kernel_matches_reference -- --nocapture` passed; `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run` passed; `cargo test -p burn_trellis --features runtime-model-wgpu canonical_wgpu_no_host_readback_before_extraction -- --nocapture` passed; strict single-run sanity (`trellis2_run --input docs/input_chair.jpg --quality low --backend wgpu --strict-benchmark --require-runtime-model --repeat 1`) completed `status=ok` with `timings_ms.total=92115.400612` and `host_readback_count=0`. Evidence: `docs/reports/parity_gap/20260228T142200Z_w10_rope_rotate_kernel_integration.md`. |
| W11 Sparse-flow RoPE phase-kernel integration | Completed (2026-02-28) | Added second custom sparse-flow RoPE kernel path in `burn_flex_gmm`: `rope_rotate_pairs_phase_kernel` + wrapper `rope_rotate_pairs_from_phase_wgpu` that consumes `[tokens,pairs]` phase directly and computes trig in-kernel. Refactored sparse-flow RoPE token-coord path (`sparse_structure_flow.rs`) to build phase once (`rope_phase_from_coord_tensor`) and route canonical WGPU through the new phase-kernel bridge before fallback tensor ops. This removes separate `cos`/`sin` tensor materialization on canonical token-coordinate RoPE path while preserving strict fail-fast semantics. | `cargo fmt --all` passed; `cargo test -p burn_flex_gmm --features wgpu-kernel rope_rotate_pairs_ -- --nocapture` passed (includes new `rope_rotate_pairs_phase_kernel_matches_reference`); `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run` passed; `cargo test -p burn_trellis --features runtime-model-wgpu canonical_wgpu_no_host_readback_before_extraction -- --nocapture` passed; strict single-run sanity (`trellis2_run --input docs/input_chair.jpg --quality low --backend wgpu --strict-benchmark --require-runtime-model --repeat 1`) completed `status=ok` with `timings_ms.total=107697.445244` and `host_readback_count=0`. Evidence: `docs/reports/parity_gap/20260228T151953Z_w11_rope_phase_kernel_integration.md`. |
| W12 Sparse-flow RoPE direct-coord kernel integration | Completed (2026-02-28) | Added third sparse-flow RoPE custom kernel path in `burn_flex_gmm`: `rope_rotate_pairs_coords_kernel` + wrapper `rope_rotate_pairs_from_coords_wgpu` that consumes token coords (`[tokens,3]`) directly, computes per-pair axis/frequency layout once on host (`rope_pair_layout_params`), and performs trig/rotation in one dispatch. Updated runtime-model sparse-flow bridge in `sparse_structure_flow.rs` to route canonical token-coordinate RoPE through direct-coord kernel first (`maybe_rotate_pairs_coords_wgpu`), preserving strict fail-fast semantics and non-WGPU fallback behavior for generic test backends. | `cargo fmt --all` passed; `cargo test -p burn_flex_gmm --features wgpu-kernel rope_rotate_pairs_ -- --nocapture` passed (includes new `rope_rotate_pairs_coords_kernel_matches_reference`); `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run` passed; `cargo test -p burn_trellis --features runtime-model-wgpu canonical_wgpu_no_host_readback_before_extraction -- --nocapture` passed; strict single-run sanity (`trellis2_run --input docs/input_chair.jpg --quality low --backend wgpu --strict-benchmark --require-runtime-model --repeat 1`) completed `status=ok` with `timings_ms.total=95015.623177` and `host_readback_count=0`. Evidence: `docs/reports/parity_gap/20260228T152922Z_w12_rope_direct_coords_kernel_integration.md`. |
| W13 Decoder skinny-linear custom kernel integration | Completed (2026-02-28) | Added `linear_skinny_kernel` + wrapper `linear_skinny_forward_wgpu` in `burn_flex_gmm` for large-row tiny-output linear projections (`output = input * weight^T + bias`) in one dispatch. Wired canonical decoder WGPU skinny-linear branch in `sparse_decoder_wgpu_ops.rs` to use the custom kernel (replacing multi-pass per-column reduction tensor ops), with explicit fail-fast error propagation and rationale comments kept inline. | `cargo fmt --all` passed; `cargo test -p burn_flex_gmm --features wgpu-kernel linear_skinny_kernel_matches_reference -- --nocapture` passed; `cargo test -p burn_flex_gmm --features wgpu-kernel rope_rotate_pairs_ -- --nocapture` passed; `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run` passed; `cargo test -p burn_trellis --features runtime-model-wgpu canonical_wgpu_no_host_readback_before_extraction -- --nocapture` passed; `./scripts/guard_canonical_runtime.sh` passed; strict single-run sanity (`trellis2_run --input docs/input_chair.jpg --quality low --backend wgpu --strict-benchmark --require-runtime-model --repeat 1`) completed `status=ok` with `timings_ms.total=109802.353864`, `timings_ms.decode_shape_decoder=19976.733657`, `timings_ms.decode_tex_decoder=14753.269637`, and `host_readback_count=0`. Evidence: `docs/reports/parity_gap/20260228T154404Z_w13_decoder_skinny_linear_kernel.md`. |
| W14 Decoder layer-norm affine custom kernel integration | Completed (2026-02-28) | Added fused decoder layer-norm kernels in `burn_flex_gmm`: `layer_norm_row_stats_kernel` + `layer_norm_affine_kernel` with wrapper `layer_norm_affine_forward_wgpu` (row stats + affine apply on-device). Routed canonical decoder `layer_norm_wgpu` to this kernel path (including no-affine calls via tensor-native ones/zeros) in `sparse_decoder_wgpu_ops.rs`, replacing multi-op mean/sub/var/sqrt/mul/add tensor chains with strict fail-fast kernel invocation. | `cargo fmt --all` passed; `cargo test -p burn_flex_gmm --features wgpu-kernel layer_norm_affine_kernel_matches_reference -- --nocapture` passed; `cargo test -p burn_flex_gmm --features wgpu-kernel linear_skinny_kernel_matches_reference -- --nocapture` passed; `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run` passed; `cargo test -p burn_trellis --features runtime-model-wgpu canonical_wgpu_no_host_readback_before_extraction -- --nocapture` passed; `./scripts/guard_canonical_runtime.sh` passed; strict single-run sanity (`trellis2_run --input docs/input_chair.jpg --quality low --backend wgpu --strict-benchmark --require-runtime-model --repeat 1`) completed `status=ok` with `timings_ms.total=102965.288899`, `timings_ms.decode_shape_decoder=19412.757648`, `timings_ms.decode_tex_decoder=14479.818934`, and `host_readback_count=0`. Evidence: `docs/reports/parity_gap/20260228T155521Z_w14_decoder_layer_norm_affine_kernel.md`. |
| W15 Decoder fused layer-norm+SiLU custom kernel integration | Completed (2026-02-28) | Added fused decoder kernels in `burn_flex_gmm`: `layer_norm_affine_silu_kernel` + wrapper `layer_norm_affine_silu_forward_wgpu` (row stats + affine + SiLU on-device). Routed canonical decoder upsample chains from separate `layer_norm_wgpu` then `silu_wgpu` into single `layer_norm_silu_wgpu` calls in `sparse_decoder_runtime_impl.rs` for norm1/conv1 and layer_norm/conv2 paths, preserving strict fail-fast semantics and device residency. | `cargo fmt --all` passed; `cargo test -p burn_flex_gmm --features wgpu-kernel layer_norm_affine_silu_kernel_matches_reference -- --nocapture` passed; `cargo test -p burn_flex_gmm --features wgpu-kernel layer_norm_affine_kernel_matches_reference -- --nocapture` passed; `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run` passed; `cargo test -p burn_trellis --features runtime-model-wgpu canonical_wgpu_no_host_readback_before_extraction -- --nocapture` passed; `./scripts/guard_canonical_runtime.sh` passed; strict single-run sanity (`trellis2_run --input docs/input_chair.jpg --quality low --backend wgpu --strict-benchmark --require-runtime-model --repeat 1`) completed `status=ok` with `timings_ms.total=91701.436118`, `timings_ms.decode_shape_decoder=19035.980035`, `timings_ms.decode_tex_decoder=13932.658392`, and `host_readback_count=0`. Evidence: `docs/reports/parity_gap/20260228T160623Z_w15_decoder_layer_norm_silu_kernel.md`. |
| W16 Sparse-flow module-attention chunk-plan relaxation (non-fusion WGPU) | Completed (2026-02-28) | Relaxed sparse-flow chunk planning for non-fusion module-attention kernels (`CubeBackend` flash-attention path) in `sparse_structure_flow.rs`: removed dense-logits budget downscaling from module-kernel chunk sizes on native non-fusion WGPU and kept conservative cap logic for fusion backends only. Applied to both self-attention streamed chunk plan and cross-attention query chunk plan, preserving fail-fast semantics and existing token cap guards. | `cargo fmt --all` passed; `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run` passed; `cargo test -p burn_trellis --features runtime-model-wgpu canonical_wgpu_no_host_readback_before_extraction -- --nocapture` passed; `./scripts/guard_canonical_runtime.sh` passed; strict single-run sanity (`trellis2_run --input docs/input_chair.jpg --quality low --backend wgpu --strict-benchmark --require-runtime-model --repeat 1`) completed `status=ok` with `timings_ms.total=86361.579836`, `timings_ms.shape_slat=10181.03252`, `timings_ms.tex_slat=4479.207085`, `timings_ms.decode_shape_decoder=18446.9867`, `timings_ms.decode_tex_decoder=13990.430235`, and `host_readback_count=0`. Evidence: `docs/reports/parity_gap/20260228T161355Z_w16_sparse_flow_chunk_relax.md`. |
| W17 Sparse-flow backend-aware MLP/linear chunk tuning (non-fusion WGPU) | Completed (2026-02-28) | Tuned sparse-flow chunking heuristics in `sparse_structure_flow.rs` for canonical non-fusion WGPU module-attention path: added `attention_uses_non_fusion_module_kernel` helper, widened MLP chunk sizing (`4_096` / `8_192` tier) and token-linear chunk sizing (`16_384` cap) for native WGPU only, and loosened MLP sync cadence (`16`) to reduce queue-fence overhead while keeping bounded synchronization. Added regression coverage for backend-aware chunk policy (`sparse_flow_backend_chunk_tokens_*`). | `cargo fmt --all` passed; `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run` passed; `cargo test -p burn_trellis --features runtime-model-wgpu sparse_flow_backend_chunk_tokens_ -- --nocapture` passed; `cargo test -p burn_trellis --features runtime-model-wgpu canonical_wgpu_no_host_readback_before_extraction -- --nocapture` passed; `./scripts/guard_canonical_runtime.sh` passed; strict sanity (`trellis2_run --input docs/input_chair.jpg --quality low --backend wgpu --strict-benchmark --require-runtime-model --repeat 2`) completed `status=ok` with warm run `timings_ms.total=46554.815447`, `timings_ms.sparse=5007.28142`, `timings_ms.shape_slat=8933.690586`, `timings_ms.tex_slat=4398.788883`, `timings_ms.decode_shape_decoder=12740.112751`, `timings_ms.decode_tex_decoder=12662.562701`, `host_readback_count=0`, and dispatch invariants `decode_shape_wgpu_dispatches=40` / `decode_tex_wgpu_dispatches=40`. Evidence: `docs/reports/parity_gap/20260228T162211Z_w17_sparse_flow_chunk_tuning.md`. |
| W18 Neighbor sorted-hash query tuning + decoder block neighbor reuse | Completed (2026-02-28) | Added ConvNeXt tensor-path neighbor reuse in decoder block loops (`sparse_decoder_wgpu_ops.rs`) by caching neighbor tensors per kernel topology for stable coord tensors, avoiding per-block rebuild churn. Tuned sorted-hash neighbor query bound in `burn_flex_gmm/src/wgpu.rs` (`DEFAULT_NEIGHBOR_SORTED_HASH_MATCH_SCAN`) from `256` -> `32` -> `8` to cut bounded query work while retaining strict parity checks and fail-fast behavior. | `cargo fmt --all` passed; `cargo test -p burn_flex_gmm --features wgpu-kernel neighbor_rows_ -- --nocapture` passed after each tuning step; `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run` passed; `cargo test -p burn_trellis --features runtime-model-wgpu canonical_wgpu_no_host_readback_before_extraction -- --nocapture` passed; strict sanity (`trellis2_run ... --seed 7 --repeat 2 --runtime-decoder-conv-telemetry`) completed `status=ok` with warm run `timings_ms.total=52943.243711`, `timings_ms.decode=33366.977376`, `timings_ms.decode_tex_decoder=13092.816033`, `timings_ms.host_readback_count=0`, and tex neighbor telemetry `device_hash_ms=13068.53` (`shape device_hash_ms=6662.87`), with dispatch invariants still `40/40`. Evidence: `docs/reports/parity_gap/20260228T171312Z_w18_neighbor_sorted_hash_scan_tune.md`. |
| W19 Tensor-path neighbor cache closure | Completed (2026-02-28) | Closed the uncached tensor-native neighbor path in `burn_flex_gmm/src/wgpu.rs`: `neighbor_rows_tensor_from_coords_tensor` now uses the shared neighbor cache with tensor-identity keys (handle/shape/strides hash, no host coord readback), with explicit host-vs-tensor key namespace separation. Added regression test `neighbor_rows_tensor_cache_reuses_across_tensor_coord_clones` to lock the behavior. | `cargo fmt --all` passed; `cargo test -p burn_flex_gmm --features wgpu-kernel neighbor_rows_ -- --nocapture` passed (includes new tensor-cache test); `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run` passed; `./scripts/guard_canonical_runtime.sh` passed; strict sanity (`trellis2_run ... --seed 7 --repeat 2 --runtime-decoder-conv-telemetry`) completed `status=ok` with warm run `timings_ms.total=51550.477620`, `timings_ms.decode=31852.309469`, `timings_ms.decode_shape_decoder=13769.650985`, `timings_ms.decode_tex_decoder=13053.994370`, `timings_ms.host_readback_count=0`, and tex neighbor telemetry `cache_hits=8`, `cache_misses=4`, `device_builds=4`, `device_hash_ms=13028.15` (dispatch invariants still `40/40`). Evidence: `docs/reports/parity_gap/20260228T172918Z_w19_tensor_neighbor_cache.md`. |
| W20 Decode stage-boundary sync attribution fix + warm sanity | Completed (2026-02-28) | Added explicit decode stage-boundary WGPU synchronization in `staged_pipeline_runtime_decode.rs` (`runtime_decode_stage_boundary_sync`) and wired it immediately after shape/tex decode tensor-runtime completion so `decode_shape_decoder_ms` / `decode_tex_decoder_ms` reflect real GPU completion instead of deferred queue flush in later stages. Kept canonical fail-fast semantics (`sync` failure returns hard error), and preserved host-readback guard behavior. | `cargo fmt --all` passed; `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run` passed; `cargo test -p burn_trellis --features runtime-model-wgpu canonical_wgpu_no_host_readback_before_extraction -- --nocapture` passed; `./scripts/guard_canonical_runtime.sh` passed; strict sanity (`trellis2_run --input docs/input_chair.jpg --quality low --backend wgpu --strict-benchmark --require-runtime-model --repeat 2`) completed `status=ok` with warm run `timings_ms.total=44969.610554`, `timings_ms.decode=25510.128568`, `timings_ms.decode_shape_decoder=11769.902959`, `timings_ms.decode_tex_decoder=11229.630019`, `host_readback_count=0`, and dispatch invariants `decode_shape_wgpu_dispatches=40` / `decode_tex_wgpu_dispatches=40`. Additional telemetry sanity run completed with warm run `timings_ms.total=42545.553067`, `timings_ms.decode=25214.015539`, `timings_ms.decode_shape_decoder=11654.265966`, `timings_ms.decode_tex_decoder=11035.610260`, `host_readback_count=0`. Evidence: `docs/reports/parity_gap/20260228T182156Z_w20_decode_stage_boundary_sync_and_warm_sanity.md`. |
| W21 Sparse-conv auto-fused decode-shape clamp | Completed (2026-02-28) | Tightened sparse-conv auto-kernel selection in `burn_flex_gmm/src/wgpu.rs`: added `DEFAULT_SPARSE_WGPU_FUSED_AUTO_MAX_IN_CHANNELS_PER_GROUP=128` and required it in the auto fused-oc4 gate, keeping high-inner-work decode-like shapes on baseline kernels by default. Added regression test `sparse_conv_auto_schedule_keeps_baseline_for_high_inner_work_decode_shape` (rows=8338, in/out=1024). | `cargo fmt --all` passed; `cargo test -p burn_flex_gmm --features wgpu-kernel sparse_conv_auto_schedule_ -- --nocapture` passed (13 tests); sparse-conv stage bench evidence: rows=8338,in/out=1024 baseline `p50=699.414ms` vs fused `p50=997.580ms`, rows=9955,in/out=256 baseline `p50=59.607ms` vs fused `p50=61.239ms`, rows=4425,in/out=512 baseline `p50=99.626ms` vs fused `p50=107.739ms`; `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run` passed; `cargo test -p burn_trellis --features runtime-model-wgpu canonical_wgpu_no_host_readback_before_extraction -- --nocapture` passed; `./scripts/guard_canonical_runtime.sh` passed; strict runtime runs remained canonical (`host_readback_count=0`, dispatches `40/40`) with decode-stage reduction (warm run observed `decode=18613.711891`, `decode_shape_decoder=8495.993811`, `decode_tex_decoder=7693.965088`), while same-session sparse-stage latency variance remained and is tracked separately. Evidence: `docs/reports/parity_gap/20260228T190803Z_w21_sparse_conv_autofused_clamp.md`. |
| W22 End-of-decode WGPU stage fence for repeat stability | Completed (2026-02-28) | Added explicit end-of-decode pipeline stage fence in `staged_pipeline.rs` (`runtime_pipeline_stage_boundary_sync`) and wired it after `decode_latent_to_outputs(...)` completion when WGPU decode dispatches are present. This prevents trailing decode queue work from spilling into the next repeat's sparse-stage timing and keeps strict fail-fast semantics (`sync` error returns hard failure). | `cargo fmt --all` passed; `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run` passed; `cargo test -p burn_trellis --features runtime-model-wgpu canonical_wgpu_no_host_readback_before_extraction -- --nocapture` passed; `./scripts/guard_canonical_runtime.sh` passed; strict sanity (`trellis2_run --input docs/input_chair.jpg --quality low --backend wgpu --strict-benchmark --require-runtime-model --repeat 2`) completed `status=ok` with warm run `timings_ms.total=41122.508622`, `timings_ms.sparse=5276.209279`, `timings_ms.shape_slat=9491.068676`, `timings_ms.tex_slat=4891.551258`, `timings_ms.decode=21155.844158`, `host_readback_count=0`, and dispatch invariants `decode_shape_wgpu_dispatches=40` / `decode_tex_wgpu_dispatches=40`. Prior same-session W21 warm run had sparse spill (`sparse=20786.317068`), now resolved. Evidence: `docs/reports/parity_gap/20260228T191634Z_w22_decode_end_stage_fence_repeat_stability.md`. |
| W23 Neighbor sorted-hash kernel-row scan-cap tune | Completed (2026-02-28) | Tuned sorted-hash neighbor query cap in `burn_flex_gmm/src/wgpu.rs` from a single global cap to kernel-row-aware caps (`k<=64 -> 8`, `k<=256 -> 16`, else `32`) so canonical decode k3 path reduces bounded query work while larger-kernel parity paths retain conservative coverage. Kept binary-search iterations fixed at 32 with explicit rationale comment after runtime-step loop regression in CubeCL WGSL parity tests. | `cargo fmt --all` passed; `cargo test -p burn_flex_gmm --features wgpu-kernel neighbor_rows_ -- --nocapture` passed; `cargo test -p burn_flex_gmm --features wgpu-kernel neighbor_algo_auto_uses_kernel_aware_thresholds -- --nocapture` passed; `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run` passed; `cargo test -p burn_trellis --features runtime-model-wgpu canonical_wgpu_no_host_readback_before_extraction -- --nocapture` passed; `./scripts/guard_canonical_runtime.sh` passed; neighbor-stage matrix (`tmp/runs/20260228T192714Z_neighbor_sorted_scan_tune/01_matrix.log`) reduced k3 `rows=181381` sorted-hash mean `25.711ms -> 22.696ms` (baseline from `tmp/runs/20260228T192107Z_neighbor_algo_matrix/01_matrix.log`) and lowered probe max `64 -> 40`; strict sanity (`tmp/runs/20260228T192714Z_trellis2_wgpu_warm_after_neighbor_scan_tune/01_run.log`) completed `status=ok` with warm run `timings_ms.total=34603.728315`, `timings_ms.decode=17532.741464`, `host_readback_count=0`, and dispatch invariants `40/40`. Evidence: `docs/reports/parity_gap/20260228T192714Z_w23_neighbor_sorted_hash_kernel_row_scan_cap.md`. |
| W24 Neighbor sorted-hash compile-time search-step dispatch | Completed (2026-02-28) | Replaced single sorted-hash query kernel with compile-time variants (`16/24/32` search steps) and host-side row-bucket dispatch in `burn_flex_gmm/src/wgpu.rs`, preserving static-loop parity behavior while reducing probe work for common decode row counts. Updated hash-probe telemetry to reflect resolved search-step dispatch. | `cargo fmt --all` passed; `cargo test -p burn_flex_gmm --features wgpu-kernel neighbor_rows_ -- --nocapture` passed; `cargo test -p burn_flex_gmm --features wgpu-kernel neighbor_algo_auto_uses_kernel_aware_thresholds -- --nocapture` passed; `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run` passed; `cargo test -p burn_trellis --features runtime-model-wgpu canonical_wgpu_no_host_readback_before_extraction -- --nocapture` passed; `./scripts/guard_canonical_runtime.sh` passed; neighbor stage matrix (`tmp/runs/20260228T203600Z_neighbor_sorted_search_steps_compiletime/01_matrix.log`) kept canonical `k=3` auto path at `mean=22.024ms` with `hash_probe_max=32`; strict sanity (`tmp/runs/20260228T203900Z_trellis2_wgpu_warm_after_sorted_hash_steps_dispatch/01_run.log`) completed `status=ok` with warm run `timings_ms.total=37154.873541`, `timings_ms.decode=20013.857931`, `host_readback_count=0`; telemetry warm profile (`tmp/runs/20260228T204400Z_trellis2_wgpu_warm_profile_sorted_hash_steps_dispatch/01_run.log`) completed `status=ok` with warm run `timings_ms.total=33415.091887`, `timings_ms.decode=18495.477512`, shape neighbor telemetry `hash_probe_avg=589.10` and `hash_probe_max=32`, and dispatch invariants `40/40`. Evidence: `docs/reports/parity_gap/20260228T204400Z_w24_neighbor_sorted_hash_compiletime_search_steps.md`. |
| W25 Neighbor sorted-hash mid-bucket search-step tightening | Completed (2026-02-28) | Added compile-time sorted-hash query kernel variant `neighbor_rows_from_sorted_hash_kernel_18` and tightened row-bucket dispatch in `burn_flex_gmm/src/wgpu.rs` (`<=2^16 -> 16`, `<=2^18 -> 18`, `<=2^24 -> 24`, else `32`) so canonical 512-quality high-row decode queries no longer over-iterate at 24 steps. Added resolver regression coverage (`neighbor_sorted_hash_search_step_resolver_uses_mid_bucket`). | `cargo fmt --all` passed; `cargo test -p burn_flex_gmm --features wgpu-kernel neighbor_rows_ -- --nocapture` passed; `cargo test -p burn_flex_gmm --features wgpu-kernel neighbor_algo_auto_uses_kernel_aware_thresholds -- --nocapture` passed; `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run` passed; `cargo test -p burn_trellis --features runtime-model-wgpu canonical_wgpu_no_host_readback_before_extraction -- --nocapture` passed; `./scripts/guard_canonical_runtime.sh` passed; target-row microbench (`tmp/runs/20260228T203350Z_neighbor_midbucket_probe_bench/01_matrix.log`) reduced `rows=181381,k=3` sorted-hash probe totals `117534888 -> 88151166` and probe max `32 -> 26`; strict telemetry profile (`tmp/runs/20260228T203413Z_trellis2_wgpu_w25_neighbor_midbucket_profile/01_run.log`) remained canonical (`status=ok`, `host_readback_count=0`, dispatch invariants `40/40`) and showed shape-decoder probe totals `146912184 -> 117528462` with probe max `32 -> 26`; evidence and interpretation captured in `docs/reports/parity_gap/20260228T203413Z_w25_neighbor_sorted_hash_mid_bucket_search_steps.md`. |
| W26 Neighbor bucket-hash auto policy for large small-k decode rows | Completed (2026-02-28) | Added bucket-hash auto routing in `burn_flex_gmm/src/wgpu.rs` for large small-k neighbor workloads (`kernel_rows<=64 && rows>=32768`) while keeping sorted-hash routing unchanged below this threshold to avoid mid-row regressions. Added resolver regression test (`neighbor_algo_auto_routes_bucket_hash_for_large_small_k`) and wired tool algo parsing (`bucket|bucket-hash`) for stage-bench validation. | `cargo fmt --all` passed; `cargo test -p burn_flex_gmm --features wgpu-kernel neighbor_algo_auto_routes_bucket_hash_for_large_small_k -- --nocapture` passed; `cargo test -p burn_flex_gmm --features wgpu-kernel neighbor_rows_ -- --nocapture` passed; `cargo test -p burn_flex_gmm --features wgpu-kernel neighbor_algo_auto_uses_kernel_aware_thresholds -- --nocapture` passed; `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run` passed; `cargo test -p burn_trellis --features runtime-model-wgpu canonical_wgpu_no_host_readback_before_extraction -- --nocapture` passed; `./scripts/guard_canonical_runtime.sh` passed; neighbor stage matrix (`tmp/runs/20260228T223900Z_neighbor_auto_bucket_threshold_matrix/01_matrix.log`) shows auto crossing to bucket behavior (`rows=32768`: auto `2.387ms` vs sorted `4.092ms`, bucket `2.050ms`; `rows=65536`: auto `4.540ms` vs sorted `13.521ms`, bucket `3.494ms`); strict high-quality sanity (`tmp/runs/20260228T230800Z_trellis2_w26_single_high_strict_tee/01_run.log`) remained canonical (`status=ok`, `host_readback_count=0`, dispatch invariants `40/40`) with single-run `timings_ms.total=151714.996109`, `timings_ms.decode=17519.912149`, `timings_ms.shape_slat=44453.853457`, `timings_ms.tex_slat=25677.133859`, `timings_ms.sparse=41364.249251`. Evidence: `docs/reports/parity_gap/20260228T230800Z_w26_neighbor_bucket_auto_policy.md`. |
| W27 Sparse-flow MLP small-token unchunked policy (non-fusion WGPU) | Completed (2026-02-28) | Tuned sparse-flow MLP chunk policy in `burn_trellis/src/runtime_model/sparse_structure_flow.rs` so non-fusion WGPU backend keeps small/mid token regimes (`tokens<=8192`) unchunked, reducing per-chunk matmul/concat overhead on the common 4k-8k SLAT path while retaining existing larger-token chunking safeguards. Added regression coverage in `sparse_flow_backend_chunk_tokens_wgpu_use_wider_chunks` for `tokens=4096` and `8192`. | `cargo fmt --all` passed; `cargo test -p burn_trellis --features runtime-model-wgpu sparse_flow_backend_chunk_tokens_wgpu_use_wider_chunks -- --nocapture` passed; `cargo test -p burn_trellis --features runtime-model-wgpu canonical_wgpu_no_host_readback_before_extraction -- --nocapture` passed; `./scripts/guard_canonical_runtime.sh` passed; `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run` passed; strict low-quality repeat sanity (`tmp/runs/20260228T233900Z_trellis2_w27_sparseflow_mlp_unchunked_small_tokens_rerun/01_run.log`) remained canonical (`status=ok`, `host_readback_count=0`, dispatch invariants `40/40`) with warm run `timings_ms.total=33461.681450`, `timings_ms.sparse=3744.439305`, `timings_ms.shape_slat=6941.085066`, `timings_ms.tex_slat=3537.363554`, `timings_ms.decode=18954.242802`; compared to W22 warm baseline (`total=41122.508622`, `sparse=5276.209279`, `shape_slat=9491.068676`, `tex_slat=4891.551258`, `decode=21155.844158`) this pass improves all major stages in the low-quality warm profile. Evidence: `docs/reports/parity_gap/20260228T232600Z_w27_sparse_flow_mlp_small_token_unchunked.md`. |
| W28 Sparse-flow prepared-context dead-path removal | Completed (2026-02-28) | Removed non-promoted prepared-context cache scaffolding from `burn_trellis/src/runtime_model/sparse_structure_flow.rs`: deleted `CrossAttentionPreparedContext`, removed `prepare_context`/`forward_prepared` cross-attention APIs, removed model/runtime `*_prepared` flow variants, and simplified block/model forward signatures to keep only the active canonical cross-attention path. This closes the lingering dead-code surface from failed W31 cache experiments without changing canonical runtime behavior. | `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run` passed; `cargo test -p burn_trellis --features runtime-model-wgpu sparse_flow_backend_chunk_tokens_wgpu_use_wider_chunks -- --nocapture` passed; `./scripts/guard_canonical_runtime.sh` passed. Evidence: `docs/reports/parity_gap/20260228T222118Z_w28_sparse_flow_prepared_path_cleanup.md`. |
| W29 Sparse-flow CFG batch-pairing trial | Rejected (2026-02-28) | Prototyped single-forward CFG pairing (pos/neg batched together) in `sparse_structure_flow.rs`, then rolled it back after strict parity regression. Added explicit inline comments documenting why sequential dual-forward CFG remains canonical today. | Repro failure captured in `tmp/runs/20260228T222118Z_trellis2_w29_cfg_batch_fused_low_strict_repeat2/01_run.log` (`coords=4703`, decode guard failure). Post-revert validation: `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run` passed; `cargo test -p burn_trellis --features runtime-model-wgpu sparse_flow_backend_chunk_tokens_wgpu_use_wider_chunks -- --nocapture` passed; strict low sanity restored in `tmp/runs/20260228T222118Z_trellis2_w29_postrevert_low_strict_repeat1/01_run.log` (`status=ok`, `coords=8338`, `host_readback_count=0`). Evidence: `docs/reports/parity_gap/20260228T223641Z_w29_cfg_batch_pairing_rejected.md`. |
| W30 Strict benchmark invariant guard integration | Completed (2026-02-28) | Wired `scripts/check_trellis_strict_benchmark_invariants.py` into `scripts/guard_canonical_runtime.sh` via explicit optional log-driven configuration (`TRELLIS2_STRICT_BENCH_LOG`, optional baseline+threshold controls), documented usage in `scripts/guards/README.md`, and added optional CI execution in `.github/workflows/test.yml` for strict WGPU runs (when strict env + weights root are enabled). This makes strict benchmark invariants explicit and blocking when a benchmark log is provided while keeping default non-GPU guards lightweight. | `./scripts/guard_canonical_runtime.sh` passed; `python3 scripts/check_trellis_strict_benchmark_invariants.py tmp/runs/20260228T233900Z_trellis2_w27_sparseflow_mlp_unchunked_small_tokens_rerun/01_run.log --min-shape-dispatches 40 --min-tex-dispatches 40` passed; `TRELLIS2_STRICT_BENCH_LOG=tmp/runs/20260228T233900Z_trellis2_w27_sparseflow_mlp_unchunked_small_tokens_rerun/01_run.log TRELLIS2_STRICT_BENCH_MIN_SHAPE_DISPATCHES=40 TRELLIS2_STRICT_BENCH_MIN_TEX_DISPATCHES=40 ./scripts/guard_canonical_runtime.sh` passed; `bash -n scripts/guard_canonical_runtime.sh` passed; `python3 -m py_compile scripts/check_trellis_strict_benchmark_invariants.py` passed. Evidence: `docs/reports/parity_gap/20260228T231250Z_w30_strict_benchmark_invariant_guard_integration.md`. |
| W31 Decoder linear wide-row chunk-cap trial | Rejected (2026-02-28) | Trialed wider force chunk cap for wide decode linear path in `sparse_decoder_wgpu_ops.rs` (`8192 -> 12288`) to reduce dispatch fragmentation, then reverted after strict warm-profile regression vs promoted W27 baseline. Kept canonical cap at `8192` and preserved existing bounded behavior/comment rationale. | Strict run remained canonical in trial (`tmp/runs/20260228T232444Z_trellis2_w31_linear_chunk_cap_12288_low_strict_repeat2_rerun/01_run.log`, `status=ok`, `host_readback_count=0`, dispatch invariants `40/40`) but warm run regressed vs W27 baseline: total `36671.588` vs `33461.681` (+9.59%), decode `19708.660` vs `18954.243` (+3.98%), shape_slat `7759.006` vs `6941.085` (+11.78%). Post-revert validation: `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run` passed; `cargo test -p burn_trellis --features runtime-model-wgpu canonical_wgpu_no_host_readback_before_extraction -- --nocapture` passed; `./scripts/guard_canonical_runtime.sh` passed. Evidence: `docs/reports/parity_gap/20260228T232444Z_w31_decoder_linear_chunk_cap_trial_rejected.md`. |
| W32 Sparse-flow MLP unchunk threshold (12,288) trial | Rejected (2026-02-28) | Trialed raising non-fusion WGPU sparse-flow MLP unchunk threshold (`8192 -> 12288`) in `sparse_structure_flow.rs` to avoid chunk fragmentation around `rows=8338`, then reverted after same-session strict warm-profile regression. Canonical threshold remains `8192` with prior W27 behavior. | Trial run remained canonical (`tmp/runs/20260228T232444Z_trellis2_w32_sparseflow_mlp_unchunk_12288_low_strict_repeat2/01_run.log`, `status=ok`, `host_readback_count=0`, dispatch invariants `40/40`) but warm run regressed vs W27 baseline: total `37764.874` vs `33461.681` (+12.86%), sparse `4954.097` vs `3744.439` (+32.31%), shape_slat `8304.094` vs `6941.085` (+19.64%), tex_slat `4547.159` vs `3537.364` (+28.55%), decode `19658.970` vs `18954.243` (+3.72%). Also regressed vs W31 trial total (`37764.874` vs `36671.588`, +2.98%). Post-revert validation: `cargo test -p burn_trellis --features runtime-model-wgpu sparse_flow_backend_chunk_tokens_wgpu_use_wider_chunks -- --nocapture` passed; `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run` passed. Evidence: `docs/reports/parity_gap/20260228T233520Z_w32_sparse_flow_mlp_unchunk_threshold_trial_rejected.md`. |
| W33 Sparse-flow module-attention dense-window gate (10,240) trial | Rejected (2026-02-28) | Trialed widening native module-attention dense-window gate (`8192 -> 10240`) in `sparse_structure_flow.rs` so `rows=8338` stays on dense module-attention path, then reverted after strict warm-profile regression. Canonical gate remains `8192`. | Trial run remained canonical (`tmp/runs/20260228T233520Z_trellis2_w33_sparseflow_chunked_gate_10240_low_strict_repeat2/01_run.log`, `status=ok`, `host_readback_count=0`, dispatch invariants `40/40`) but warm run regressed vs W27 baseline: total `38994.464` vs `33461.681` (+16.53%), sparse `4803.105` vs `3744.439` (+28.27%), shape_slat `8717.408` vs `6941.085` (+25.59%), tex_slat `4736.692` vs `3537.364` (+33.90%), decode `20428.838` vs `18954.243` (+7.78%). Also regressed vs W32 total (`38994.464` vs `37764.874`, +3.25%). Post-revert validation: `cargo test -p burn_trellis --features runtime-model-wgpu sparse_flow_backend_chunk_tokens_wgpu_use_wider_chunks -- --nocapture` passed; `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run` passed. Evidence: `docs/reports/parity_gap/20260228T233520Z_w33_sparse_flow_chunk_gate_trial_rejected.md`. |
| W34 Decode timing-mode metadata + strict guard alignment | Completed (2026-03-01) | Added explicit decode timing mode metadata (`decode_stage_fenced`) from runtime decode to stage timings to pipeline profile and CLI JSON output, so non-strict runs that disable stage fences are explicitly labeled as unfenced decode substage timing mode. Kept strict mode fenced behavior unchanged and updated strict invariant checker to validate fenced decode timings when the field is present. | `cargo fmt --all` passed; `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run` passed; `cargo check -p burn_synth` passed; `cargo test -p burn_trellis --features runtime-model-wgpu canonical_wgpu_no_host_readback_before_extraction -- --nocapture` passed; `./scripts/guard_canonical_runtime.sh` passed; strict run (`tmp/runs/20260301T001900Z_trellis2_w34c_stage_fence_metadata_strict_low_repeat2/01_run.log`) warm metrics `total=38554.163`, `decode=20064.258`, `decode_stage_fenced=true`, `host_readback_count=0`, dispatches `40/40`; non-strict run (`tmp/runs/20260301T002200Z_trellis2_w34d_stage_fence_metadata_nonstrict_low_repeat2/01_run.log`) warm metrics `total=37239.853`, `decode=20035.366`, `decode_stage_fenced=false`, `host_readback_count=0`, dispatches `40/40`; strict invariant script passed on strict log with decode-stage fence check. Evidence: `docs/reports/parity_gap/20260301T002400Z_w34_decode_stage_fence_metadata_and_strict_guard.md`. |

## 0. Problem Statement

Historical kickoff gaps (addressed across W2-W8 on this branch):

- Sparse structure coord selection is not fully device-owned end-to-end.
- Decoder/runtime boundaries still contain host materialization surfaces.
- Sparse ownership (`VarLenTensorOwned` / `SparseTensorOwned`) is still hybrid host/device.
- Neighbor hash build now uses non-CAS sorted-hash parallel query path for large workloads, but perf tuning and benchmark closure are still pending.
- Core behavior still depends on parity-critical env/hook branches near runtime hot paths.

This roadmap defines a topological implementation plan that closes those gaps without introducing fallback behavior, hidden mode switches, or unstable benchmark practices.

## 1. Hard Invariants (Must Never Regress)

### 1.1 Canonical runtime invariants

I1. Canonical WGPU path is fail-fast only.

I2. No silent CPU reroute in canonical mode.

I3. No parity-critical `std::env::*` branches in canonical runtime behavior.

I4. Host extraction is allowed only at explicit extraction boundaries.

I5. No `.into_data()` calls in canonical sparse/decode flow modules except extraction module allowlist.

### 1.2 Parity invariants against TRELLIS.2 semantics

P1. Cascade decoder upsample semantics use `upsample_times=4`.

P2. Token-cap backoff boundary follows canonical behavior (`<=` boundary semantics handled correctly by tests).

P3. Shape decode subdivisions are re-used by texture decode tensor-natively (no host recompute).

P4. No decode/PBR rescue behavior in canonical mode.

### 1.3 Performance invariants

R1. Stage-level host readback count for canonical sparse/decode flow is zero before extraction boundary.

R2. GPU dispatches are present for decode shape and decode tex in strict benchmark mode.

R3. Workload growth from low to 512-equivalent does not trigger unbounded stage stalls.

## 2. Historical Gap Ledger (Closed)

| Gap ID | Severity | Current Evidence (module) | Why it hurts | Closure target |
|---|---|---|---|---|
| G1 | High | `sparse_structure_decoder.rs` now runs threshold/select/sort/cap tensor-natively, but the logic is still embedded in orchestrator code rather than dedicated reusable device-op wrappers/kernels with strict readback invariants | Harder to enforce/benchmark kernel-path behavior and easier for host sync regressions to re-enter | Full tensor-native select/sort/unique/cap pipeline with explicit no-readback assertions |
| G2 | High | `sparse_decoder_runtime_impl.rs`, `fdg_decoder.rs`, `sparse_unet_vae_decoder.rs` still expose host completion surfaces at decode boundaries | Breaks device residency and adds synchronization stalls | Device-only decode handoff + explicit extraction-only wrappers |
| G3 | High | `sparse_structure_flow.rs` still defines hybrid ownership (`VarLenTensorOwned`, `SparseTensorOwned`) with host vectors/accessors | Forces host-first semantics across sampling and staging | Device-first tensor ownership contract end-to-end |
| G4 | High | `burn_flex_gmm/src/wgpu.rs` now has probe/fail diagnostics and a non-CAS sorted-hash parallel query path, but large-shape tuning/bench closure is incomplete | Large-shape neighbor-map stage may still underperform target without algorithm-specific tuning and stage evidence | Tune sorted-hash scan window/sort overhead and validate against 512-equivalent stage benchmarks; keep CAS path backlog tracked upstream |
| G5 | Medium-High | Env/hook toggles influence parity-critical behavior near runtime hot path | Non-deterministic behavior and branch explosion | Typed runtime config with stable defaults |
| G6 | Medium-High | Legacy rescue/fallback branches still adjacent to canonical path | Silent semantic drift from TRELLIS.2 reference behavior | Remove rescue path from canonical mode, fail-fast |
| G7 | Medium | Harness/tests still contain stale fallback assumptions and weak invariants | Slow-path regressions re-enter unnoticed | Strict invariant assertions in harness + CI guards |

## 3. End-State Architecture

## 3.1 Crate boundaries (target)

### Canonical owners

- `crates/burn_trellis`
  - Pipeline orchestration, model semantics, typed runtime config, extraction boundary orchestration.
  - No low-level WGSL kernel implementation code in runtime orchestrators.

- `crates/burn_flex_gmm`
  - Sparse neighbor + sparse conv kernels and algorithm selectors.
  - Canonical sparse conv backend for TRELLIS decode hotspot.

### New crate (recommended)

- `crates/burn_sparse_kernels_wgpu`
  - Reusable low-level GPU primitives needed by multiple stages:
    - select/compact
    - key generation
    - sort/unique
    - cap/truncate
    - segmented scans/reductions
    - coord expansion kernels
    - grid-sample helpers
  - Exposes typed Rust wrappers accepting/returning tensors only.

Rationale:

- Keeps `burn_trellis` orchestration readable and testable.
- Avoids one giant `runtime_model/*` file becoming kernel plus orchestration mixed together.
- Makes kernel microbenching independent and bounded.

## 3.2 Internal module organization for `burn_trellis`

Proposed module split (file-level design):

- `runtime_model/types/mod.rs`
- `runtime_model/types/sparse_tensor_device.rs`
- `runtime_model/types/varlen_tensor_device.rs`
- `runtime_model/types/sparse_batch_layout_device.rs`
- `runtime_model/types/extraction.rs`

- `runtime_model/flow/mod.rs`
- `runtime_model/flow/sparse_structure_flow_orchestrator.rs`
- `runtime_model/flow/cascade_sampling.rs`

- `runtime_model/decode/mod.rs`
- `runtime_model/decode/sparse_decoder_runtime.rs`
- `runtime_model/decode/subdivision_handoff.rs`

- `runtime_model/config/mod.rs`
- `runtime_model/config/runtime_model_config.rs`
- `runtime_model/config/strict_mode.rs`

- `runtime_model/interop/mod.rs`
- `runtime_model/interop/flex_gmm_bridge.rs`
- `runtime_model/interop/kernel_bridge.rs`

Migration note:

- Existing monolithic files (`sparse_structure_flow.rs`, `sparse_decoder_runtime_impl.rs`, `sparse_structure_decoder.rs`) should be split along this boundary before or during W2/W3 tasks to reduce merge conflicts.

## 3.3 Feature surface simplification

Current usage confusion (`--features runtime-model,runtime-model-wgpu`) should be eliminated.

Target:

- Keep `runtime-model-wgpu` as the canonical public feature for runtime-model WGPU path.
- Keep `runtime-model` internal or developer-only surface.
- Document that `runtime-model-wgpu` already includes `runtime-model`.
- Ensure README and scripts use one feature flag only.

## 3.4 New crate manifest and API design (`burn_sparse_kernels_wgpu`)

Recommended `Cargo.toml` intent:

- `default = []`
- `wgpu-kernel = ["dep:burn", "dep:burn-cubecl", "dep:burn-wgpu"]`
- no feature coupling to model-specific crates

Public API surface:

- `ops::select_compact`
- `ops::key_sort_unique`
- `ops::coord_cap`
- `ops::subdivision_expand`
- `ops::grid_sample3d`

Rules:

- Accept and return tensor wrappers only.
- No host-vector convenience APIs in this crate.
- Expose deterministic behavior contract per op (ordering, tie-breaks).

## 4. Device Ownership Contract (Required for Parity)

## 4.1 Canonical types

Introduce device-first runtime types:

```rust
pub struct VarLenTensorDevice<B: Backend> {
    values: Tensor<B, 2>,
    layout: SparseBatchLayoutDevice<B>,
    channels: usize,
}

pub struct SparseTensorDevice<B: Backend> {
    coords: Tensor<B, 2, Int>,
    values: VarLenTensorDevice<B>,
    sparse_resolution: usize,
}
```

Optional host mirrors are non-canonical and extraction-only:

```rust
pub struct SparseTensorHost {
    pub coords: Vec<[i32; 4]>,
    pub values: Vec<f32>,
    pub layout: SparseBatchLayoutHost,
    pub channels: usize,
    pub sparse_resolution: usize,
}
```

## 4.2 Ownership rules

- Canonical flow/decode APIs accept and return `*Device` types only.
- `into_host*` methods are forbidden in canonical orchestrator modules.
- Host conversion lives only in `runtime_model/types/extraction.rs`.
- Extraction APIs require explicit context labels (`"mesh_export"`, `"debug_dump"`) for auditability.

## 4.3 CI guardrails

- Add deny-list script that fails CI if `.into_data()` appears in canonical modules.
- Allowlist extraction modules and tests only.
- Add deny-list for `std::env::var` in canonical runtime logic modules.

## 5. Kernel Portfolio and Design Specs

All kernels below are required or high-impact for parity closure. Each kernel work package includes correctness, perf, and safety envelopes.

## K1. Sparse structure threshold + compact + cap (device-native)

Goal:

- Replace host finalization of positive coord selection and cap semantics.

Inputs:

- logits tensor (`[N, C]` or reduced `[N]` depending stage)
- threshold value
- optional max token cap

Outputs:

- selected coord tensor `[M, 4]`
- optional selected index tensor `[M]`

Algorithm (device):

1. Threshold mask generation.
2. Prefix-sum compaction or two-pass count + scatter.
3. Canonical key creation (`(b,z,y,x)` packed to `u64`).
4. Key-value sort.
5. Unique-by-key.
6. Deterministic cap/truncate.

Determinism:

- Stable tie-breaking by packed key and source index.
- Exact output ordering across runs for fixed seed/input.

Correctness tests:

- all false mask
- all true mask
- duplicate-heavy coords
- exact boundary case (`count == cap`)
- cap+1 case (`count == cap + 1`)

Microbench envelope:

- sizes: 4k, 16k, 64k, 128k candidate rows
- timeout: 30s per case
- report: kernel_ms, output_count, hash(output)

Acceptance criteria:

- No host readback in canonical sparse structure path.
- Deterministic output hash stability across repeated runs.
- Boundary parity tests (`cap`, `cap+1`) pass.

## K2. Cascade quantize + unique + token-cap backoff

Goal:

- Keep high-res coord down-quantization and token cap semantics fully device-native.

Inputs:

- coord tensor
- `hr_resolution`, `target_sparse_resolution`, `max_num_tokens`

Outputs:

- quantized coord tensor
- selected resolution

Algorithm:

1. Quantize coords for candidate resolution.
2. Sort+unique coords.
3. Evaluate token count.
4. Apply backoff until condition satisfied (`<=` boundary semantics parity test required).

Guardrails:

- bounded iterations (max 4 or config-defined)
- fail-fast if zero coords produced unexpectedly

Correctness tests:

- token cap boundary parity test (`num_tokens == max_num_tokens`)
- deterministic ordering under collisions
- resolution monotonicity

Microbench envelope:

- rows: 8k, 32k, 128k
- collision factors: low, medium, high

Acceptance criteria:

- Token-cap boundary parity test passes for `==` and `+1` cases.
- Backoff loop terminates under bounded iterations.
- No host materialization in cascade handoff.

## K3. Subdivision active-index + child coord expansion

Goal:

- Remove decoder host completion around subdivision logits and coord expansion.

Inputs:

- subdivision logits `[rows, 8]`
- parent coords `[rows, 4]`

Outputs:

- active index pairs `[m, 2]`
- child coords `[m, 4]`
- optional linearized child index `[m]`

Algorithm:

1. Threshold subdivision logits to active mask.
2. Compact active parent-child pairs.
3. Expand child coords from parent + child offset LUT.
4. Optional stable sort by coord key.

Correctness tests:

- parity vs host reference on fixed seeds
- exact match for guide handoff to tex decoder

Microbench envelope:

- rows: 4k, 16k, 64k

Acceptance criteria:

- Decoder guide subdivision handoff parity test passes.
- No host completion branch remains in canonical decode path.

## K4. Parallel neighbor hash build/query (replace serial)

Goal:

- Replace serial `neighbor_hash_build_serial_kernel` bottleneck.

Inputs:

- coords `[rows, 4]`
- kernel offsets `[k, 3]`

Outputs:

- neighbor rows `[rows, k]`

Algorithm:

- Parallel insertion hash-table build with atomic slot claim.
- Bounded probe count + overflow counter.
- Query phase using same hash and offsets.

Design details:

- Use load factor target <= 0.55 by default.
- Use power-of-two table size for cheaper modulo.
- Provide overflow diagnostics in strict benchmark.

Correctness tests:

- matches scan backend on random coords
- adversarial collision test
- stable result ordering across runs

Microbench envelope:

- rows: 8k, 32k, 128k
- compare scan vs hash build/query times separately

Acceptance criteria:

- Hash output parity matches scan reference.
- Collision stress does not deadlock and respects probe bound.
- Canonical path no longer uses serial hash build.

## K5. Sparse conv hotspot kernels (fused + scheduling)

Goal:

- Close largest compute gap in decode sparse conv path.

Scope:

- Identify top-3 shape/group/channel patterns from stage profiling.
- Implement shape-specialized fused kernels where justified.
- Improve split-K scheduling and partial accumulation path.

Correctness tests:

- parity vs baseline sparse conv for each specialized variant
- grouped/channel edge cases

Microbench envelope:

- per shape pattern, bounded 10-iteration benches
- output: rows/s, kernel_ms, effective bandwidth estimate

Acceptance criteria:

- Each specialized kernel passes reference parity for target shapes.
- At least one dominant decode hotspot shows measurable stage-level reduction.
- No hidden fallback to legacy dense-row math in canonical mode.

## K6. GPU decode/PBR kernels

Goal:

- Remove CPU-heavy decode/pbr hot loops in canonical mode.

Targets:

- 3D grid sample attr fetch
- attribute accumulation/raster helpers with deterministic semantics

Correctness tests:

- parity vs reference attrs
- strict sparse-hole fail-fast semantics (no rescue)

Microbench envelope:

- texture sizes 256, 512, 1024
- controlled triangle counts

Acceptance criteria:

- Decode/PBR parity tests pass under strict fail-fast semantics.
- Canonical mode does not enter nearest/rescue attribute branches.
- Stage timing indicates CPU loop dominance removed for targeted substage.

## K7. Optional: Varlen attention/device-resident matmul reductions

Goal:

- Reduce sparse flow attention overhead if profiling shows it still dominates after K1-K6.

Constraint:

- Only start after K1-K5 show clear wins and parity stability.

## 6. Development Topological Order (Execution DAG)

This is the required implementation order. Parallel work is allowed only where noted.

DAG overview:

- W0 -> W1 -> W2 -> W3 -> W4 -> W5 -> W6 -> W7
- W4 -> W5 can overlap with W6 only after W4 correctness gates pass.
- W8 (cleanup + release gates) depends on W3, W5, W6, W7.

## W0. Baseline lock + guardrails (Effort S)

Dependencies: none.

Scope:

- Capture baseline timings and readback counters on current branch.
- Add CI guards for `.into_data()` and `std::env::var` in canonical modules.

Deliverables:

- baseline report template in `docs/reports/parity_gap/<run_id>.md`
- lint/guard script under `scripts/`

Success criteria:

- CI fails on new prohibited host readbacks in canonical runtime.

## W1. Config and feature hardening (Effort S-M)

Dependencies: W0.

Scope:

- Remove parity-critical env toggles from canonical runtime behavior.
- Introduce typed runtime config fields with stable defaults.
- Simplify feature usage guidance to canonical `runtime-model-wgpu` surface.

Deliverables:

- `RuntimeModelConfig` updates
- migration notes in docs

Success criteria:

- canonical runtime behavior reproducible from typed config only.

## W2. Device-first sparse ownership model (Effort L)

Dependencies: W1.

Scope:

- Introduce `SparseTensorDevice` / `VarLenTensorDevice`.
- Migrate staged sampling and sparse flow entrypoints to device-first APIs.
- Remove host-first constructors/accessors from canonical path.

Deliverables:

- new `runtime_model/types/*`
- old host-first surfaces deleted or extraction-only

Success criteria:

- canonical flow compiles without host-owned sparse structures.

## W3. Sparse-structure and cascade tensor-native closure (Effort XL)

Dependencies: W2.

Scope:

- Implement K1 and K2.
- Eliminate host sort/dedup/cap in sparse structure and cascade path.

Deliverables:

- kernel wrappers + orchestrator integration
- token-cap boundary parity tests

Success criteria:

- zero host coord materialization in sparse structure and cascade path.

## W4. Decoder host-completion elimination (Effort L-XL)

Dependencies: W2, W3.

Scope:

- Implement K3.
- Ensure guide subdivisions and active indices are handed off tensor-natively.
- Remove host completion wrappers from decode runtime path.

Deliverables:

- decode orchestration module using device tensors only

Success criteria:

- canonical decode path has no host completion branch before extraction.

## W5. Neighbor hash parallelization (Effort XL)

Dependencies: W3 (can begin after W2 for scaffolding, but integration after W3).

Scope:

- Implement K4 parallel insertion path.
- Keep scan backend only as debug/reference feature (not canonical runtime fallback).

Deliverables:

- hash build/query kernels
- selector policy and overflow telemetry

Success criteria:

- hash path beats serial baseline and does not hang under collision stress.

## W6. Sparse conv hotspot closure (Effort XL)

Dependencies: W5.

Scope:

- Implement K5 specialized/fused kernels and scheduling improvements.

Deliverables:

- top shape specialization kernels
- benchmark evidence of per-shape gains

Success criteria:

- measured decode-stage reduction from sparse conv improvements.

## W7. Decode/PBR GPU kernel closure (Effort XL)

Dependencies: W4, W6.

Scope:

- Implement K6 GPU decode/PBR kernels.
- Remove rescue behavior entirely in canonical mode.

Deliverables:

- tensor-native decode/pbr path
- strict fail-fast semantics for sparse holes

Success criteria:

- decode/PBR no longer dominated by CPU loops in canonical mode.

## W8. Harness closure, cleanup, release gate lock (Effort M)

Dependencies: W3, W5, W6, W7.

Scope:

- Remove stale modules/assumptions.
- Enforce strict invariant assertions in harness.
- Produce final closure report.

Deliverables:

- cleaned harness/docs/tests
- release gate thresholds in CI/docs

Success criteria:

- slow path reintroduction is prevented by test/lint/harness gates.

## 6.1 Topological task matrix (effort + bounded experiment budget)

| Workstream | Depends on | Effort | Max GPU experiments | Safe parallelism | Exit gate |
|---|---|---|---|---|---|
| W0 | none | S | 0 | CPU-only checks can run parallel | guard scripts + baseline report merged |
| W1 | W0 | S-M | 0 | CPU-only tests parallel | typed config replaces env in canonical path |
| W2 | W1 | L | 0-1 (one smoke) | CPU tests parallel | device-first ownership compiles end-to-end |
| W3 | W2 | XL | 3 (2 microbench + 1 smoke) | one GPU job only | sparse structure + cascade host materialization removed |
| W4 | W2,W3 | L-XL | 2 (1 parity stress + 1 smoke) | one GPU job only | decoder handoff tensor-native and host completion removed |
| W5 | W3 | XL | 3 (hash/scan compare + collision + smoke) | one GPU job only | parallel hash replaces serial canonical path |
| W6 | W5 | XL | 3 (shape microbenches + smoke) | one GPU job only | sparse conv hotspot gains validated |
| W7 | W4,W6 | XL | 3 (grid sample + decode/pbr smoke) | one GPU job only | decode/pbr CPU-dominant loops removed |
| W8 | W3,W5,W6,W7 | M | 1 (strict milestone run) | docs/report parallel with GPU run | strict harness invariants enforced in CI |

Interpretation:

- \"Max GPU experiments\" is per workstream completion, not per commit.
- If a gate fails, use diagnostics-only runs and do not exceed the budget unless blocker is understood.

## 7. Agent-Ready Work Packet Topology

Each packet should be executable by one coding agent with minimal overlap.

## Packet A (W0+W1)

Files:

- `crates/burn_trellis/src/runtime_model/config/*`
- `crates/burn_trellis/Cargo.toml`
- `scripts/*` (guard scripts)
- docs updates

Checks:

- `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run`
- targeted config tests

## Packet B (W2)

Files:

- `runtime_model/types/*`
- `runtime_model/flow/*`
- `staged_pipeline_sampling.rs`

Checks:

- sparse ownership unit tests
- smoke compile for runtime-model-wgpu

## Packet C (W3)

Files:

- `sparse_structure_decoder.rs`
- `sparse_structure_flow.rs`
- new kernel wrapper modules

Checks:

- sparse structure parity tests
- cascade token-cap boundary parity test

## Packet D (W4)

Files:

- `sparse_decoder_runtime_impl.rs`
- `fdg_decoder.rs`
- `sparse_unet_vae_decoder.rs`

Checks:

- decoder guide handoff parity tests
- canonical no-host-readback test

## Packet E (W5)

Files:

- `crates/burn_flex_gmm/src/wgpu.rs`
- optional new kernel crate modules

Checks:

- hash vs scan parity tests
- collision stress test with bounded timeout

## Packet F (W6)

Files:

- `crates/burn_flex_gmm/src/wgpu.rs`
- sparse conv microbench fixtures

Checks:

- sparse conv parity tests
- hotspot microbench improvement evidence

## Packet G (W7)

Files:

- decode/pbr modules in `burn_trellis`
- kernel primitives used for grid sample/accumulation

Checks:

- decode/pbr parity tests
- strict fail-fast sparse-hole tests

## Packet H (W8)

Files:

- harness files (`tool/trellis2_run.rs`, benches, tests, docs)

Checks:

- strict benchmark invariant assertions
- final matrix report generation

## 8. Testing and Validation Matrix

## 8.1 Test classes

- `*_correctness`: unit boundaries and deterministic ordering.
- `*_parity`: numeric parity vs reference path.
- `*_reference`: hook/reference artifact alignment.
- `*_smoke`: bounded non-crash runtime checks.

## 8.2 New required tests (minimum)

- `sparse_structure_coord_select_token_cap_boundary_parity`
- `cascade_quantize_token_cap_boundary_parity`
- `decoder_guide_subdivision_tensor_handoff_parity`
- `canonical_wgpu_no_host_readback_before_extraction`
- `neighbor_hash_parallel_matches_scan_parity`
- `neighbor_hash_parallel_collision_stress_bounded`
- `sparse_conv_hotspot_kernel_matches_reference_parity`
- `decode_pbr_device_path_sparse_hole_failfast`

## 8.3 Required commands by phase

W0-W2:

- `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run`
- `cargo check -p burn_flex_gmm --features wgpu-kernel`

W3-W5:

- targeted `cargo test` for sparse structure/decoder/hash modules
- one bounded microbench per touched kernel class

W6-W8:

- strict stage smoke run with bounded settings
- milestone benchmark run after each major phase gate

## 9. Efficient Benchmarking Plan (Safe and Modular)

This section is designed to avoid system lockups and GPU starvation.

## 9.1 Safety protocol

S1. Exactly one GPU-heavy run at a time.

S2. Every GPU run wrapped in `timeout`.

S3. Progression is always small -> medium -> large.

S4. No heavy end-to-end benchmark unless microbench parity for touched kernels passed.

S5. Every run writes machine-readable output under `tmp/runs/<run_id>/`.

S6. Stage watchdog thresholds abort pathological runs early.

## 9.2 Experiment tiers

### E0 Build/test (CPU-dominant)

Purpose:

- compile and unit parity only.

Budget:

- per command <= 120s.

### E1 Kernel microbench (single kernel)

Purpose:

- isolate one kernel hotspot.

Budget:

- per case <= 30s, total <= 120s per kernel family.

### E2 Stage smoke (bounded runtime)

Purpose:

- validate strict runtime path invariants and stage timing sanity.

Budget:

- one run per change-set, <= 180s.

### E3 Milestone benchmark (phase closure)

Purpose:

- produce before/after comparison for one closed phase.

Budget:

- warm+cold pair only, <= 2 runs.

### E4 Final closure benchmark

Purpose:

- final report after all phase gates pass.

Budget:

- small matrix only (no exhaustive sweeps).

## 9.3 Experiment minimization rule per task

- max 1 compile pass
- max 2 targeted parity tests
- max 2 microbench iterations
- max 1 stage smoke run

Escalate only when a failing invariant needs diagnosis.

## 9.4 Example command envelope

Build checks:

- `timeout 120s cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run`
- `timeout 120s cargo check -p burn_flex_gmm --features wgpu-kernel`

Kernel microbench:

- `timeout 120s cargo bench -p burn_flex_gmm --bench sparse_subm_conv -- --sample-size 10`

Strict stage smoke:

- `timeout 180s cargo run -p burn_trellis --features runtime-model-wgpu --bin trellis2_run -- --input <img> --backend wgpu --strict-benchmark --quality low --max-sparse-coords 8192`

## 9.5 GPU safety envelope (anti-stall / anti-lockup)

Run controls that should be implemented before heavy milestone runs:

- Per-stage watchdog in runtime config:
  - sparse stage timeout
  - decode stage timeout
  - decode/pbr stage timeout
- Hard caps for development mode:
  - max sparse coords
  - max children per parent
  - max hash probes
  - max neighbor table load factor
- Single-run GPU lock file under `tmp/runs/` so concurrent benches cannot start accidentally.
- Early-abort conditions in strict mode:
  - zero progress across N decoder iterations
  - hash overflow counter above threshold
  - stage time exceeds watchdog budget

Operational rule:

- Any run that hits watchdog/abort is classified as a blocker run and must not be retried at larger workload until root cause is addressed.

## 10. Observability and Diagnostics Requirements

## 10.1 Required runtime counters

- `host_readback_count`
- `host_readback_elements`
- `decode_shape_wgpu_dispatches`
- `decode_tex_wgpu_dispatches`
- `neighbor_build_ms`
- `neighbor_query_ms`
- `sparse_conv_ms` (shape/tex split)
- `decode_pbr_ms`

## 10.2 Required strict benchmark assertions

A1. If backend requested is WGPU, decode dispatch counts must be non-zero.

A2. Host readback count before extraction boundary must be zero.

A3. Canonical runtime source must be runtime-model path, not legacy path.

A4. Hash overflow counter must remain below configured threshold; otherwise fail-fast.

## 10.3 Run artifact contract

Each run must include:

- command line
- git commit or dirty snapshot note
- JSON timings
- pass/fail invariant summary

Location:

- `tmp/runs/<run_id>/`

Optional committed summaries:

- `docs/reports/parity_gap/<run_id>.md`

## 11. Cleanup Plan to Prevent Slow Paths from Returning

## 11.1 Delete/relocate legacy surfaces

- Remove host-first sparse constructors from canonical flow modules.
- Move host conversion helpers into extraction-only module.
- Remove canonical runtime access to rescue/nearest fallback paths.

## 11.2 Guardrail automation

- CI deny-list for `.into_data()` in canonical modules.
- CI deny-list for `std::env::var` in canonical runtime behavior modules.
- Add code-owner notes for modules that may introduce host sync.

## 11.3 Harness language cleanup

- Remove fallback-oriented language from tests/docs where canonical mode is expected.
- Make strict benchmark invariants explicit and blocking.

## 12. Risk Register and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Kernel correctness drift from host reference | High | mandatory parity tests with fixed seeds and fixtures per kernel |
| GPU hangs from unbounded probes or large workloads | High | bounded probe loops, stage watchdogs, timeout wrappers |
| Merge conflicts from monolithic runtime files | Medium-High | split modules early (W2/W3), use packet ownership boundaries |
| Hidden env toggles reintroduced | Medium | CI deny-list + typed config review gate |
| Bench instability from parallel GPU runs | Medium | enforce single GPU job rule + sequential experiment schedule |
| Overfitting to synthetic benches only | Medium | require stage smoke and milestone runtime validation per phase |

## 13. Success Criteria and Definition of Done

Closure is complete only when all are true:

1. Correctness parity tests pass for sparse structure, cascade, decode handoff, sparse conv, and decode/pbr.
2. Canonical WGPU runtime invariants hold (no hidden fallback, no host completion pre-extraction).
3. Serial neighbor hash path is removed from canonical runtime and replaced with validated parallel path.
4. Sparse ownership is device-backed end-to-end; legacy host-first surfaces are deleted or extraction-only.
5. Harness strict mode asserts device-path invariants and fails on regression.
6. Performance improvements are demonstrated with stage-level evidence, not just one end-to-end number.

## 14. Execution-Ready Topo Sequence (Short Form)

1. W0 guardrails + baseline lock.
2. W1 config/feature cleanup.
3. W2 device ownership migration.
4. W3 sparse structure + cascade tensor-native kernels.
5. W4 decoder host-completion removal.
6. W5 parallel neighbor hash.
7. W6 sparse conv hotspot kernels.
8. W7 decode/pbr GPU kernels.
9. W8 harness cleanup + final closure report.

## 15. Immediate Next Actions (Prepared for implementation)

N1. Create packet branch for W0/W1 and land CI guardrails first.

N2. Define `runtime_model/types` module skeleton and migration shim list for W2.

N3. Draft K1/K2 kernel API signatures in proposed kernel crate (or temporary `burn_flex_gmm::wgpu::ops`) before implementation.

N4. Add boundary parity tests now (token-cap `<=` and guide handoff) so future kernel work is constrained by tests.

N5. Lock benchmark harness stage watchdog defaults before running any more heavy experiments.

This roadmap is intentionally strict: canonical mode remains device-resident and fail-fast, and every performance change is tied to bounded experiments and parity evidence.

# W6 Scheduler Tighten + Bench Blocker (2026-02-28)

## Scope

Workstream: `W6 Sparse conv hotspot closure`.

This pass tightens sparse-conv auto scheduling rules using existing phase evidence and improves phase-matrix reporting surfaces. It also records the current GPU-bench blocker that prevented new sweep execution.

## Implementation

### 1) Tightened hot-shape fused auto route

File:
- `crates/burn_flex_gmm/src/wgpu.rs`

Change:
- Added `DEFAULT_SPARSE_WGPU_FUSED_HOT_MAX_OC_GROUP = 128`.
- Hot-shape fused route now requires `out_channels_per_group <= 128`.

Reason:
- Existing phase matrix showed `rows=4096, ic=64, oc=256` regressing when the broad hot-shape fused route selected fused automatically.

### 2) Added high-OC split-k cap for decode-like workloads

File:
- `crates/burn_flex_gmm/src/wgpu.rs`

Change:
- Added `DEFAULT_SPARSE_WGPU_SPLIT_CAP_HIGH_OC_ROWS = 8192` and
  `DEFAULT_SPARSE_WGPU_SPLIT_CAP_HIGH_OC_GROUP = 256`.
- `resolve_split_k` now clamps to `split_k <= 2` when both thresholds are met.

Reason:
- Existing phase evidence indicated high-OC workloads (`oc>=256`) at `rows>=8192` were frequently over-split by auto (`split=4`) with worse p50/mean.

### 3) Added scheduler tests for new policy

File:
- `crates/burn_flex_gmm/src/wgpu.rs`

New tests:
- `sparse_conv_auto_schedule_keeps_baseline_for_rows4096_when_oc_group_is_high`
- `sparse_conv_auto_schedule_caps_splitk_for_high_oc_decode_shape`

### 4) Improved phase-matrix script output

File:
- `scripts/bench_sparse_conv_phase_matrix.sh`

Change:
- Script now emits `auto_vs_best.csv` with per-case auto vs best p50/mean deltas.

## Bench Evidence (from existing matrix run)

Using existing run artifact:
- `tmp/runs/20260228T004120Z_w6_sparse_conv_phase_matrix_tool_after/summary.csv`

Derived comparison:
- `tmp/runs/20260228T004120Z_w6_sparse_conv_phase_matrix_tool_after/auto_vs_best.csv`

Notable p50 deltas (`auto_p50 - best_p50`, ms):
- `4096,3,64,128`: `+4.282`
- `4096,3,64,256`: `+14.487`
- `8192,3,64,128`: `+2.624`
- `8192,3,64,256`: `+10.400`
- `16384,3,64,128`: `+7.162`

Interpretation:
- Auto scheduling still has measurable gap on several decode-like shapes; this patch addresses the most consistent high-OC and hot-shape regressions first.

## GPU-Bench Blocker (new sweep)

Attempted run:
- `tmp/runs/20260228T005228Z_sparseconv_wgpu_w6_targeted_tune/`

Status:
- aborted on first case due Vulkan device allocation failure:
  - `Failed to create Vulkan device: ERROR_OUT_OF_DEVICE_MEMORY`

Environment evidence (`nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader`):
- `target/release/codec_train` consuming `96410 MiB` GPU memory at failure time.

Impact:
- No new GPU sweep data was collected in this pass; policy tightening was validated via unit tests and prior phase artifacts only.

## Validation

Commands:
- `cargo fmt --all`
- `cargo test -p burn_flex_gmm --features wgpu-kernel sparse_conv_auto_schedule_ -- --nocapture`
- `cargo check -p burn_flex_gmm --features wgpu-kernel`
- `bash -n scripts/bench_sparse_conv_phase_matrix.sh`
- `./scripts/guard_canonical_runtime.sh`
- `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run`

Result:
- all commands passed (existing unrelated warning persists in `burn_trellis`: `from_device_tensors` dead code).

## Follow-up After VRAM Unblock (2026-02-28)

After GPU memory pressure was cleared, additional bounded phase runs were executed.

### 1) Targeted rerun (same matrix that was previously blocked)

Run:
- `tmp/runs/20260228T010259Z_sparseconv_wgpu_w6_targeted_tune_rerun/`

Artifacts:
- `summary.csv`
- `best_by_case_p50.csv`
- `auto_vs_best.csv`

Observed p50 deltas (`auto_p50 - best_p50`, ms):
- `4096,3,64,128`: `0.000`
- `4096,3,64,256`: `+0.121`
- `8192,3,64,128`: `0.000`
- `8192,3,64,256`: `+0.910`

### 2) Focused high-OC confirmation run

Run:
- `tmp/runs/20260228T010347Z_sparseconv_wgpu_w6_focus_8192_oc256/`

Result (`rows=8192,k=3,ic=64,oc=256`, p50 ms):
- `auto(auto)` -> fused split-2: `16.604`
- `baseline split-2`: `15.984`
- `fused split-2`: `15.870`

Interpretation:
- Borderline fused-auto selection around this workload remained unstable and slightly slower on p50.

### 3) Additional scheduler tighten

Code update in `crates/burn_flex_gmm/src/wgpu.rs`:
- Raised `DEFAULT_SPARSE_WGPU_FUSED_AUTO_MIN_OUTPUT_WORK` from `2_000_000` to `2_300_000`.
- Added test:
  - `sparse_conv_auto_schedule_keeps_baseline_for_borderline_fused_output_work_shape`

Validation:
- `cargo test -p burn_flex_gmm --features wgpu-kernel sparse_conv_auto_schedule_ -- --nocapture` passed.
- `cargo check -p burn_flex_gmm --features wgpu-kernel` passed.

### 4) Post-tune single-case verify

Run:
- `tmp/runs/20260228T010435Z_sparseconv_wgpu_w6_auto_verify_8192_oc256/`

Result:
- `auto` now resolves to `baseline` with `split_k=2` for `8192x64->256`.

### 5) Full phase matrix after borderline tune

Run:
- `tmp/runs/20260228T010515Z_w6_sparse_conv_phase_matrix_after_borderline_tune/`

Artifacts:
- `best_by_case.csv`
- `auto_vs_best.csv`

Notable note:
- Per-shape winner identity still shows variance across short bounded runs; use these as directional scheduler evidence, not final claims.

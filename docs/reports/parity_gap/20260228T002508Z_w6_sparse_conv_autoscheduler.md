# W6 Sparse Conv Autoscheduler Tuning (2026-02-28)

## Scope

Workstream: `W6 Sparse conv hotspot closure` (incremental).

This change tunes sparse-conv auto scheduling and removes stale env-driven benchmark control surfaces in favor of explicit typed APIs.

## Implementation

### 1) Added explicit resolved sparse-conv schedule API

File:
- `crates/burn_flex_gmm/src/wgpu.rs`

Additions:
- `SparseWgpuResolvedForwardConfig { kernel_variant, split_k }`
- `resolve_sparse_wgpu_forward_config(config, rows, forward) -> Result<...>`
- Internal unified resolver used by both runtime execution and stats recording.

Purpose:
- Make selected auto variant/split observable and testable.
- Ensure wrapper telemetry reflects the exact schedule the kernel path used.

### 2) Tuned split-k auto thresholds

File:
- `crates/burn_flex_gmm/src/wgpu.rs`

Updated from broad heuristic to tuned thresholds:
- `DEFAULT_SPARSE_WGPU_SPLIT_WORK_THRESHOLD_SPLIT2 = 320_000_000`
- `DEFAULT_SPARSE_WGPU_SPLIT_WORK_THRESHOLD_SPLIT4 = 760_000_000`

Rationale:
- Derived from bounded stage bench runs under `tmp/runs/20260228T002022Z_conv_stage_w6_autosched`.
- Keeps split-k enabled for medium/large workloads while avoiding always-on split-4 for medium cases.

### 3) Made fused auto selection conservative (baseline-first)

File:
- `crates/burn_flex_gmm/src/wgpu.rs`

Auto fused gating now requires:
- `out_channels_per_group >= 256`
- `rows >= 8192`
- `inner_work >= 1024`
- `output_work >= 2_000_000`

Rationale:
- In bounded WGPU stage runs, fused was not consistently better on common decode shapes.
- Preserve explicit fused override support while keeping default auto path stable.

### 4) Removed env-driven benchmark control in criterion bench

File:
- `crates/burn_flex_gmm/benches/sparse_subm_conv.rs`

Changes:
- Replaced legacy env toggles (`BURN_FLEX_GMM_WGPU_*`) with explicit typed APIs:
  - `neighbor_rows_tensor_from_coords_with_algo(...)`
  - `sparse_subm_conv_forward_wgpu_with_config(...)`
- Added explicit benchmark variants:
  - neighbor: scan / sorted-hash / hash-table-serial
  - conv: auto / baseline split1 / baseline split4 / fused split4

### 4b) Removed stale env-toggle test routing in WGPU kernel tests

File:
- `crates/burn_flex_gmm/src/wgpu.rs`

Changes:
- Replaced remaining `std::env::set_var/remove_var` test routing with explicit:
  - `neighbor_rows_tensor_from_coords_with_algo(..., NeighborDeviceAlgoPreference::...)`
  - `sparse_subm_conv_forward_wgpu_with_config(..., SparseWgpuForwardConfig {...})`

Result:
- Tests now validate explicit behavior contracts instead of no-op env state.

### 5) Enhanced stage bench output with resolved schedule fields

File:
- `crates/burn_flex_gmm/tool/sparse_conv_stage_bench.rs`

Output JSON now includes:
- `resolved_variant`
- `resolved_split_k`
- `single_group_specialized_calls`

### 6) Added single-group sparse-conv specialized kernels (shape hotspot path)

Files:
- `crates/burn_flex_gmm/src/wgpu.rs`
- `crates/burn_trellis/src/staged_pipeline_runtime_helpers.rs`

Changes:
- Added single-group specialized WGPU kernels for canonical decode shape pattern:
  - `sparse_subm_conv_single_group_kernel`
  - `sparse_subm_conv_single_group_fused_oc4_kernel`
  - `sparse_subm_conv_splitk_partial_single_group_kernel`
  - `sparse_subm_conv_splitk_partial_single_group_fused_oc4_kernel`
- Extended internal kernel variant routing with:
  - `BaselineSingleGroup`
  - `FusedOc4SingleGroup`
- Selector uses specialization only when layout is true single-group ownership:
  - `groups == 1`
  - `in_channels_per_group == in_channels`
  - `out_channels_per_group == out_channels`
- Added sparse-conv telemetry counter:
  - `single_group_specialized_calls`
- Exposed this counter in stage bench JSON and TRELLIS runtime conv telemetry log line.

## Bench Evidence

### Calibration run (pre-change tuning data)

Run dir:
- `tmp/runs/20260228T002022Z_conv_stage_w6_autosched`

Representative samples used for threshold selection:
- `rows=2048,k=3,in=64,out=128`:
  - baseline split1: `2.483ms`
  - baseline split2: `1.496ms`
  - baseline split4: `1.542ms`
- `rows=4096,k=3,in=64,out=128`:
  - baseline split1: `3.858ms`
  - baseline split2: `3.441ms`
  - baseline split4: `2.737ms`

### Post-change sanity run

Run dir:
- `tmp/runs/20260228T002508Z_w6_sparse_conv_autosched_v2`

Auto resolution confirms baseline-first scheduling:
- `auto_rows2048.json`: `resolved_variant=baseline`, `resolved_split_k=2`
- `auto_rows4096.json`: `resolved_variant=baseline`, `resolved_split_k=4`
- `auto_rows8192.json`: `resolved_variant=baseline`, `resolved_split_k=4`

### Specialized-kernel activation sanity run

Run dir:
- `tmp/runs/20260228T003606Z_w6_single_group_specialized_sanity`

Representative outputs:
- `auto.json`: `single_group_specialized_calls=6`
- `baseline_split1.json`: `single_group_specialized_calls=6`
- `fused_split1.json`: `single_group_specialized_calls=6`

This confirms canonical single-group decode-shape cases are routed through specialized kernels.

## Validation

Commands:
- `cargo fmt --all`
- `cargo check -p burn_flex_gmm --features wgpu-kernel`
- `cargo check -p burn_flex_gmm --features wgpu-kernel --benches`
- `cargo test -p burn_flex_gmm --features wgpu-kernel sparse_conv_auto_schedule_ -- --nocapture`
- `cargo test -p burn_flex_gmm --features wgpu-kernel neighbor_rows_device_hash_matches_scan -- --nocapture`
- `cargo test -p burn_flex_gmm --features wgpu-kernel wgpu_single_group_specialized_kernel_matches_cpu_flex_path -- --nocapture`
- `cargo test -p burn_flex_gmm --features wgpu-kernel wgpu_fused_oc4_matches_baseline_output -- --nocapture`
- `cargo test -p burn_flex_gmm --features wgpu-kernel wgpu_splitk_matches_default_kernel_output -- --nocapture`
- `cargo test -p burn_flex_gmm --features wgpu-kernel wgpu_fused_splitk_matches_baseline_output -- --nocapture`
- `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run`
- `./scripts/guard_canonical_runtime.sh`

Results:
- All commands passed.
- Existing `burn_trellis` warning unchanged: dead code `from_device_tensors` in `sparse_decoder.rs`.

## Status

W6 remains `In progress`.

Completed in this increment:
- scheduler API surface and schedule observability
- auto threshold tuning + conservative fused auto policy
- benchmark harness cleanup (typed controls)

Remaining W6 scope:
- shape-specialized fused kernels that show consistent wins on target decode profiles
- stage-level decode hotspot reduction evidence tied to real TRELLIS runtime traces

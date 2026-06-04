# Parity Gap Run Report

- `run_id`: `20260227T224433Z_w3_sparse_coord_select_guard_tighten`
- `date_utc`: `2026-02-27`
- `git_ref`: `dirty_worktree_uncommitted`
- `dirty_worktree`: `true`

## Scope

- Workstream(s): `W3` (increment)
- Goal: tighten sparse-structure coord selection/cap semantics as tensor-native helper logic, remove remaining staged sparse/debug readbacks in canonical modules, and lock stricter `.into_data()` guard baseline.
- Backend: `N/A` (compile + targeted unit tests + guard script)
- Input(s): `synthetic unit-test logits`

## Command(s)

```bash
cargo fmt --all
timeout 240s cargo test -p burn_trellis --features runtime-model-wgpu sparse_structure_coord_select_cap_boundary_parity -- --nocapture
timeout 120s cargo test -p burn_trellis --features runtime-model-wgpu sparse_structure_coord_select_empty_mask_returns_empty_coords -- --nocapture
./scripts/guard_canonical_runtime.sh
timeout 180s cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run
```

## Invariant Summary

- Canonical WGPU fail-fast only: `unchanged`
- Pre-extraction host readbacks: `reduced in canonical modules (removed staged sampling cond-tensor debug readbacks and sparse-structure logits debug tensor dump)`
- Decode dispatch presence: `not measured in this run`
- Runtime source identity: `not measured in this run`
- Sparse structure cap boundary parity (`count == cap`, `count == cap+1`): `pass`
- Sparse structure all-negative mask handling: `pass`
- Canonical guard baseline lock: `pass (after intentional baseline tighten)`

## Timings (ms)

- Not collected in this run.

## Kernel / Telemetry Counters

- Not collected in this run.

## Outcome

- Status: `pass`
- Blocking issue(s): `none in this slice`
- Next action: `continue W3 by moving sparse-structure/cascade selection primitives into dedicated reusable device op wrappers and tightening no-host-readback assertions for canonical WGPU path`

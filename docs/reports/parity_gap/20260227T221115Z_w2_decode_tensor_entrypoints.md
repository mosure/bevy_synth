# Parity Gap Run Report

- `run_id`: `20260227T221115Z_w2_decode_tensor_entrypoints`
- `date_utc`: `2026-02-27`
- `git_ref`: `dirty_worktree_uncommitted`
- `dirty_worktree`: `true`

## Scope

- Workstream(s): `W2` (increment)
- Goal: wire staged runtime decode to tensor-native decoder entrypoints so canonical WGPU decode avoids host row completion in `from_latent` and downstream decode math.
- Backend: `N/A` (guard + compile validation)
- Input(s): `N/A`

## Command(s)

```bash
./scripts/guard_canonical_runtime.sh
cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run
cargo check -p burn_synth
cargo test -p burn_trellis --features runtime-model-wgpu runtime_decode_tests::sanitize_mesh_geometry_removes_non_finite_vertices_and_reindexes_faces -- --nocapture
```

## Invariant Summary

- Canonical WGPU fail-fast only: `compile-verified for tensor decode entrypoints`
- Pre-extraction host readbacks: `not measured in this run`
- Decode dispatch presence: `not measured in this run`
- Runtime source identity: `not measured in this run`
- Guardrail baseline lock: `pass`
- Staged runtime decode calls decoder tensor entrypoints (`decode_*_with_tensors`) on canonical WGPU coord path: `pass`
- Targeted runtime decode unit test for include-module compilation path: `pass`

## Timings (ms)

- Not collected in this run.

## Kernel / Telemetry Counters

- Not collected in this run.

## Outcome

- Status: `pass`
- Blocking issue(s): `none`
- Next action: `continue W2 by propagating device-backed row ownership from sparse flow outputs into staged sample structs to remove remaining host materialization in canonical WGPU sampling->decode handoff`

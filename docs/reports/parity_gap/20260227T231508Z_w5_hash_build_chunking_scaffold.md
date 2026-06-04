# Parity Gap Run Report

- `run_id`: `20260227T231508Z_w5_hash_build_chunking_scaffold`
- `date_utc`: `2026-02-27`
- `git_ref`: `dirty_worktree_uncommitted`
- `dirty_worktree`: `true`

## Scope

- Workstream(s): `W5` (scaffold)
- Goal: reduce pathological single-dispatch serial hash build behavior by chunking serial hash insertion into bounded row windows per dispatch while preserving deterministic semantics; prepare ground for full parallel insertion replacement.
- Backend: `wgpu-kernel` + downstream `runtime-model-wgpu`
- Input(s): `compile + targeted runtime decode unit slice`

## Command(s)

```bash
cargo fmt --all
timeout 180s cargo check -p burn_flex_gmm --features wgpu-kernel
timeout 180s cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run
timeout 180s cargo test -p burn_trellis --features runtime-model-wgpu runtime_decode_tests::sanitize_mesh_geometry_removes_non_finite_vertices_and_reindexes_faces -- --nocapture
./scripts/guard_canonical_runtime.sh
```

## Invariant Summary

- Canonical WGPU fail-fast only: `unchanged`
- Pre-extraction host readbacks: `unchanged`
- Decode dispatch presence: `not measured in this run`
- Runtime source identity: `not measured in this run`
- Hash build path remains deterministic serial insertion semantics: `pass`
- Hash build launch now chunks row ranges (`row_start..row_end`) to bound per-dispatch loop length: `pass`
- Canonical guard baseline lock: `pass`

## Timings (ms)

- Not collected in this run.

## Kernel / Telemetry Counters

- Not collected in this run.

## Outcome

- Status: `pass`
- Blocking issue(s): `full parallel insertion (atomic/CAS-based path) not yet implemented`
- Next action: `implement parallel insertion build kernel path with bounded probe + collision/overflow diagnostics and switch canonical hash mode to that path`

# W4 Canonical WGPU Decode Host-Branch Removal (2026-02-28)

## Scope

Workstream: `W4 Decoder host-completion elimination`.

This pass removes remaining host decode branch selection in canonical `runtime-model-wgpu` staged runtime decode orchestration.

## Implementation

File:
- `crates/burn_trellis/src/staged_pipeline_runtime_decode.rs`

Changes:
- Canonical `runtime-model-wgpu` decode now requires device shape coord tensors for shape decoder entry.
- Removed host-coord-based branch selection for shape decode in canonical WGPU mode; shape decode always calls tensor-native decoder APIs.
- Canonical `runtime-model-wgpu` tex decode now always uses tensor-native coord/row handoff.
- Tex coords default to tex device coords, with explicit fallback to shape device coords (device-to-device only), never host coords.
- Host row tensorization remains only as a narrow decode-boundary bridge when row tensors are absent, but no host decode-completion path remains in canonical WGPU mode.

Fail-fast behavior:
- If canonical WGPU decode is requested and shape coord tensor is missing, runtime returns an explicit error instead of entering host decode path.

## Validation

Commands:
- `cargo fmt --all`
- `cargo check -p burn_trellis --features runtime-model-wgpu --bin trellis2_run`
- `cargo test -p burn_trellis --features runtime-model-wgpu runtime_decode_tests::sanitize_mesh_geometry_ -- --nocapture`
- `./scripts/guard_canonical_runtime.sh`

Results:
- all commands passed.
- existing unrelated warning persists: `SparseSubdivisionLogits::from_device_tensors` dead code.

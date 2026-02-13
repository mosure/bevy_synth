# AGENTS.md

This file defines the working ethos for contributors and coding agents in this repository.

## Mission

Build numerically correct, GPU-efficient, production-grade 3D synthesis pipelines that are:

1. Canonical to upstream model implementations (TripoSG, TRELLIS.2, FlexGMM/FlexGEMM).
2. Practical in native and wasm targets.
3. Measurable, testable, and reproducible.

## Non-Negotiables

1. Numerical correctness first.
2. Performance work must follow scientific method.
3. Avoid hidden behavior branches and unstable environment-dependent paths.
4. Prefer shared canonical pipeline code over duplicated per-app implementations.

## Numerical Correctness Policy

1. Treat upstream Python/Torch/CUDA as the reference behavior.
2. Validate stage-by-stage with hooks and strict error thresholds where possible.
3. Any optimization must preserve outputs within defined tolerances.
4. If outputs diverge, identify the first failing stage and fix upstream of that point.
5. Mesh quality regressions are correctness regressions, not just "visual differences."

## Performance Policy

1. Optimize the actual bottleneck, not assumed hotspots.
2. Stage-level benchmarking is required (not only end-to-end mega benches).
3. Capture GPU utilization, memory, and stage timestamps during benchmarks.
4. Large GPU idle gaps imply pipeline or transfer bottlenecks and must be investigated.
5. Prefer device-resident execution through decode/PBR; minimize host readbacks/transfers.

## Model Import and Loading

1. `.bpk` artifacts are canonical in this repo; support both `f32` and `f16` variants.
2. Sharded loading/import paths must be first-class for native and wasm.
3. Keep host RAM bounded during load/init (wasm target: under 4 GB).
4. Asset paths should be local/repo-relative defaults, not brittle absolute paths.
5. Bundlers/workflows must include required model variants (including image-large where needed).

## Configuration Principles

1. Defaults should "just work" for intended model runs.
2. Minimize runtime config surface area.
3. Avoid `std::env::*` control branches for core inference behavior.
4. Remove deprecated toggles (for example, legacy `match_python` style branching).

## Architecture Boundaries

1. `burn_synth` should be the canonical inference pipeline surface.
2. `bevy_synth_runtime` should adapt/wrap canonical pipeline behavior, not reimplement it.
3. Avoid duplicate mesh/postprocess logic across CLI/runtime/app layers.
4. Keep modules focused and maintainable; split overly large implementation files.

## Testing and Quality Gates

1. Add/maintain tests that reflect real workload behavior.
2. Avoid synthetic "magic number" tests that do not validate real pipeline properties.
3. Memory tests should observe real process memory behavior during load/init.
4. Keep `clippy` clean for touched targets (`-D warnings` for relevant scope).
5. Bench and test evidence should accompany performance/correctness changes.

## Logging and Observability

1. Pipeline stage logs must include timestamps.
2. Report elapsed timing per stage and end-to-end totals.
3. Keep benchmark outputs machine-readable for comparison over time.

## Practical Decision Rules

1. If portability/integration speed is needed, use CubeCL as baseline.
2. If parity/perf gap remains in hotspots, implement custom kernels in constrained backend.
3. If strict CUDA parity is required, plan explicit CUDA-specific kernel path.
4. Prefer robust, maintainable solutions over one-off benchmark wins.

## Contributor Checklist (Before Merging Significant Changes)

1. Correctness: parity checks/hooks pass for affected stages.
2. Performance: before/after stage timings + utilization captured.
3. Memory: load/init RAM behavior measured on realistic model set.
4. Tooling: relevant tests pass, relevant clippy scope is clean.
5. Integration: CLI + runtime/app paths both exercise the same core implementation.

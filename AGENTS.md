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
5. Prefer fully device-resident execution through decode/PBR; host readbacks/transfers are exceptions, not defaults.
6. On wasm, never rely on synchronous GPU tensor readback; prefer async tensor readback APIs.
7. Preserve GPU backend parity across native/wasm whenever feasible; avoid wasm-only CPU fallbacks unless correctness or platform limits force it.
8. For GPU-capable backends, default inference/extraction paths must stay on GPU and fail fast on GPU errors rather than silently rerouting to CPU.
9. If host transfer is unavoidable, keep it narrow (stage boundary only), explicit in code, and documented with rationale.

## Container + GPU Verification Contract

1. In containerized runs, do not treat backend selection flags as proof of GPU execution.
2. For every claimed `wgpu`/GPU benchmark, capture evidence of actual device usage (adapter/backend logs plus utilization or process-level GPU sample).
3. If in-container GPU evidence tooling is missing, stop and report a blocker with exact environment gap; do not report benchmark/correctness completion.
4. If runtime falls back to CPU (`ndarray`/llvmpipe/software path), treat it as a regression unless explicitly requested for debug.
5. Long-running worker loops must keep publishing progress/blocker signals; never silently stall after first failure.

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
5. `bevy_synth` (UI/headless app) is a runtime shell, not an inference implementation.
6. Native inference in `bevy_synth` paths must go through `burn_synth::runtime` via `bevy_synth_runtime`.
7. Do not keep dormant legacy native inference branches in bevy crates; remove them entirely.
8. If a wrapper path cannot support a requested mode, fail fast with an explicit error (no silent fallback).

## Wrapper Parity Contract

1. For equivalent input/config/backend, `bevy_synth` and `burn_synth` must be numerically aligned within defined tolerance.
2. `bevy_synth_runtime` argument mapping must preserve canonical runtime semantics (seed, mesh mode, backend, quality, target faces).
3. Canonical defaults are owned by `burn_synth` runtime; wrappers may expose flags but must not silently rewrite behavior.
4. Backend failures in wrapper flows must be surfaced as errors; do not auto-switch to CPU or alternate synthesis paths.
5. New inference features should be implemented in `burn_synth` first, then exposed in wrappers.
6. Web/native wrappers must invoke the same canonical GPU extraction mode for the same mesh/backend settings (for example flash mode), with only platform-required readback mechanics differing.

## Deduplication and Modularity Ethos

1. Each inference stage must have one canonical implementation owner:
   - model math and extraction internals in `burn_tripo`
   - pipeline orchestration and defaults in `burn_synth`
   - UI/runtime adaptation only in bevy crates
2. Wrappers must call canonical APIs and must not duplicate stage math (sampling, decode, extraction, mesh conversion) inline.
3. If native and wasm require different mechanics (for example sync vs async readback), keep both paths in the same canonical module behind explicit `cfg` gates, with shared API shape and shared constants.
4. Avoid copy-pasting constants and formulas across crates; define once in the canonical owner and reuse.
5. Platform-specific branches must be narrow and local; do not fork whole pipeline flows when only one stage differs.
6. When duplicate logic is discovered, priority is to extract and centralize it before adding more features on top.
7. New public wrapper features should be thin pass-throughs to canonical pipeline methods, not new parallel implementations.
8. Refactors must reduce total duplicated stage code over time; adding new duplication requires explicit justification in PR notes.

## Packaging and Installability Rules

1. Published binaries must work with plain install commands:
   - `cargo install burn_synth`
   - `cargo install bevy_synth`
   - `cargo install burn_synth_mcp`
2. Default crate features for published binaries must include a fully functional primary runtime path (no required extra feature flags for baseline usage).
3. README examples must include install plus one runnable command for each published binary.
4. Workspace inter-crate versions must be coherent before release (no local semver skew that breaks `cargo check` / `cargo install --path`).
5. Prefer upstream crates over vendored/patch forks before publish unless there is an actively documented blocker.

## Wrapper Boundary Strictness

1. `bevy_synth_runtime` must treat `burn_synth` as the canonical inference/runtime owner.
2. Direct dependencies on lower-level model crates in wrapper/runtime crates are only allowed for narrow adaptation needs (for example mesh type bridging), and must be minimized over time.
3. Path/model-root resolution should be centralized in canonical runtime layers where possible; avoid parallel resolver logic in wrappers.
4. Delete legacy or dormant compatibility branches once canonical paths exist; do not keep dual implementations alive.

## Model Loading Parity Rules

1. Single-file `.bpk` and multi-part `.bpk.parts.json` loading must be numerically equivalent for the same model/precision.
2. Add loader parity tests that compare outputs across load strategies for each supported model component.
3. Loader fallback decisions (precision, source, fallback path) must be explicit in logs with reason.
4. Download/bootstrap logic must tolerate partial files and recover cleanly (resume/retry/replace), never leaving silent corrupted artifacts.
5. Prefer one canonical artifact partitioning scheme per model family; avoid mixed conventions that increase operator confusion.

## Web/Wasm Operational Rules

1. Web wrappers must call the same canonical pipeline and model-loading behavior as `burn_synth` wasm APIs.
2. Wasm startup/model init must expose deterministic user-visible progress (stage and part counts) to avoid "frozen" ambiguity.
3. Run WebGPU-heavy integration tests serially to avoid self-induced GPU contention during validation.
4. Browser-specific workarounds must be explicitly temporary, feature-gated where practical, and easy to remove when upstream fixes land.

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

## Standard Workflow (Agent + Contributor)

1. Reproduce first with an exact command, backend, mesh mode, and seed.
2. Isolate the first failing stage (preprocess, encode, sample, decode, mesh, postprocess) before changing code.
3. Fix in the canonical layer first (`burn_synth` / `burn_tripo`) and adapt in wrappers second.
4. Add the smallest realistic test that fails before the fix and passes after it.
5. Run the required validation matrix for the touched area.
6. Record evidence (commands + key metrics) in a stable location, not ad hoc scratch files.

## Workspace Hygiene and Artifact Layout

1. Never create new ad hoc files at repository root for experiments.
2. Put all temporary run outputs under `tmp/runs/<run_id>/`.
3. Use `tmp/upstream/<project>/<version>/` for upstream snapshots/patch prep.
4. Use `tmp/wasm/<run_id>/` for wasm/browser loop logs and diagnostics.
5. For committed evidence, use `docs/reports/<topic>/<run_id>.md` plus machine-readable sidecars (`.json`/`.csv`).
6. Before merge, ensure no accidental large generated artifacts are included in commits.

## Run ID and Naming Conventions

1. Use run id format: `YYYYMMDDTHHMMSSZ_<pipeline>_<backend>_<goal>`.
2. Name logs by stage when possible: `01_build.log`, `02_bindgen.log`, `03_infer.log`, etc.
3. Name benchmark outputs with the same run id for easy cross-linking.
4. Avoid ambiguous names like `run1`, `new`, `temp`, `debug_final`.

## Test Harness Conventions

1. Use test suffixes consistently: `*_smoke`, `*_correctness`, `*_parity`, `*_reference`.
2. `smoke` tests validate load/shape/non-crash and may be backend-gated (`BURN_WGPU_SMOKE=1`, `BURN_CUDA_SMOKE=1`).
3. `correctness` tests validate numeric tolerances and may be backend-gated (`BURN_WGPU_CORRECTNESS=1`).
4. `reference` tests validate against canonical saved outputs and should use explicit gates (for example `TRIPOSG_FULL_REFERENCE=1`).
5. Expensive runtime integration tests should be `#[ignore = "..."]` with clear run instructions in the ignore reason.
6. Skip messages must explicitly state which env var enables the test.

## Required Validation Matrix (Minimum)

1. Tripo model/pipeline changes:
   - `cargo test -p burn_tripo --features import pipeline::runtime_parity::tests:: -- --nocapture`
   - relevant `*_correctness`/`*_parity` tests for touched stages
2. `bevy_synth_runtime` changes:
   - targeted worker tests
   - `cargo check -p bevy_synth_runtime`
3. `bevy_synth` integration changes:
   - `cargo check -p bevy_synth`
   - one headless parity smoke run against `burn_synth` with matched args (compare mesh stats and/or hash)
4. wasm path changes:
   - `cargo check -p burn_synth --target wasm32-unknown-unknown --features wasm-api,wasm-api-wgpu`
   - relevant wasm smoke/parity tests when available

## Configuration Safety Rails

1. Parity-critical runtime decisions (for example Tripo weight precision) must come from canonical parity helpers, not ad hoc boolean composition.
2. Do not silently rewrite user-requested quality/mesh settings in runtime startup; prefer explicit presets and clear logs.
3. Runtime startup logs should include effective backend, mesh mode, and precision policy for reproducibility.
4. Performance-oriented alternatives must be opt-in and labeled; correctness-first defaults must remain stable.
5. Wrapper/runtime layers must not introduce hidden backend fallbacks that change correctness behavior.

## Script and Automation Conventions

1. Reusable automation belongs in `scripts/`, not `tmp/`.
2. Script names should be domain-first and action-specific (for example `triposg_stage_bench.*`, `wasm_playwright_loop.*`).
3. Keep cross-platform parity for important workflows (`.sh` and `.ps1`) when feasible.
4. One-off investigation scripts should live under `tmp/runs/<run_id>/` and are not part of long-term workflow.

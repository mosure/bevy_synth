# Canonical Runtime Guard Baselines

These baselines lock known exceptions in canonical runtime-model modules so CI can fail on newly introduced host materialization or env-driven behavior.

Files:

- `canonical_runtime_into_data.baseline`: occurrence list for `.into_data(` in canonical runtime modules.
- `canonical_runtime_env_var.baseline`: occurrence list for `std::env::var(` in canonical runtime modules.

Guard command:

```bash
scripts/guard_canonical_runtime.sh
```

Optional strict benchmark invariant guard:

- Set `TRELLIS2_STRICT_BENCH_LOG=/path/to/trellis2_run.log` to enable strict JSON invariant checks.
- Optional baseline regression check:
  - `TRELLIS2_STRICT_BENCH_BASELINE_LOG=/path/to/baseline.log`
  - `TRELLIS2_STRICT_BENCH_MAX_REGRESSION_PCT=20` (default when baseline is set)
- Optional absolute thresholds:
  - `TRELLIS2_STRICT_BENCH_MAX_TOTAL_MS`
  - `TRELLIS2_STRICT_BENCH_MAX_SPARSE_MS`
  - `TRELLIS2_STRICT_BENCH_MAX_SHAPE_SLAT_MS`
  - `TRELLIS2_STRICT_BENCH_MAX_TEX_SLAT_MS`
  - `TRELLIS2_STRICT_BENCH_MAX_DECODE_MS`
- Optional dispatch minimums:
  - `TRELLIS2_STRICT_BENCH_MIN_SHAPE_DISPATCHES` (default `1`)
  - `TRELLIS2_STRICT_BENCH_MIN_TEX_DISPATCHES` (default `1`)

If a baseline update is intentional, run the guard script, inspect the diff, and then update the affected baseline file in the same commit with rationale in PR notes.

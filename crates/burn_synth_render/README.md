# burn_synth_render

Differentiable rendering primitives for scene-scale pose fitting.

This crate currently provides a soft point-surface renderer with gradients through object translation, yaw, and scale. It is intentionally separate from the deterministic rasterized visible-surface fitter used by the scene pipeline: the goal is to validate transform gradients and renderer metrics before replacing production fitting stages.

The current renderer is not a full triangle or PBR differentiable renderer. It is the first proven differentiable stage for silhouette/depth fitting.

## Validation

```bash
cargo test -p burn_synth_render -- --nocapture
cargo clippy -p burn_synth_render -- -D warnings
cargo bench -p burn_synth_render --bench soft_point_renderer -- --sample-size 10
```

Optional WGPU smoke:

```bash
BURN_SYNTH_RENDER_WGPU_SMOKE=1 cargo test -p burn_synth_render --features wgpu wgpu -- --nocapture
```

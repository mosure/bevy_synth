use std::path::PathBuf;

use burn_trellis::TrellisQuality;
use burn_trellis::hook_diff::{
    HookDiffStatus, HookSnapshot, compare_hook_snapshots, compute_stats,
};
use burn_trellis::pipeline::{Trellis2Pipeline, Trellis2PipelineConfig, TrellisRunOptions};

#[test]
fn trellis2_e2e_hook_alignment_against_reference() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let input_image = root.join("assets/hooks/trellis2_preprocess_input.png");
    let reference_hook = root.join("assets/hooks/trellis2_full_reference_alpha_512.safetensors");
    if !input_image.exists() || !reference_hook.exists() {
        eprintln!("Skipping Trellis2 e2e hook alignment: missing input or reference hook capture.");
        return Ok(());
    }

    let mut config = Trellis2PipelineConfig::default();
    config.image_large_root = Some(
        std::env::var("TRELLIS2_IMAGE_LARGE_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(
                    "E:/models/huggingface/hub/models--microsoft--TRELLIS-image-large/snapshots/25e0d31ffbebe4b5a97464dd851910efc3002d96",
                )
            }),
    );
    if let Ok(weights_root) = std::env::var("TRELLIS2_WEIGHTS_ROOT") {
        config.weights_root = PathBuf::from(weights_root);
    }
    if !config.weights_root.exists() {
        let local_default = PathBuf::from(
            "E:/models/huggingface/hub/models--microsoft--TRELLIS.2-4B/snapshots/af44b45f2e35a493886929c6d786e563ec68364d",
        );
        if local_default.exists() {
            config.weights_root = local_default;
        }
    }
    if !config.weights_root.exists() {
        eprintln!(
            "Skipping Trellis2 e2e hook alignment: weights root missing at {}",
            config.weights_root.display()
        );
        return Ok(());
    }

    let out_dir = std::env::temp_dir().join("burn_trellis_e2e_hooks");
    std::fs::create_dir_all(&out_dir)?;
    let actual_hook = out_dir.join("actual_alpha_512.safetensors");

    unsafe {
        std::env::set_var("TRELLIS2_DISABLE_RUNTIME_MODEL", "1");
    }
    let pipeline = Trellis2Pipeline::new(config)?;
    pipeline.validate_runtime()?;
    let _mesh = pipeline.infer_mesh(
        &input_image,
        &TrellisRunOptions {
            quality: TrellisQuality::Low,
            seed: Some(42),
            hook_output: Some(actual_hook.clone()),
            ..TrellisRunOptions::default()
        },
    )?;
    unsafe {
        std::env::remove_var("TRELLIS2_DISABLE_RUNTIME_MODEL");
    }

    let reference = HookSnapshot::from_file(reference_hook)?;
    let actual = HookSnapshot::from_file(&actual_hook)?;
    let report = compare_hook_snapshots(&reference, &actual, None);

    let missing = report
        .entries
        .iter()
        .filter(|entry| entry.status == HookDiffStatus::MissingInActual)
        .count();
    let shape_mismatch = report
        .entries
        .iter()
        .filter(|entry| entry.status == HookDiffStatus::ShapeMismatch)
        .count();
    if missing > 0 || shape_mismatch > 0 || !report.extra_in_actual.is_empty() {
        return Err(format!(
            "hook schema mismatch: missing={missing}, shape_mismatch={shape_mismatch}, extra={}",
            report.extra_in_actual.len()
        )
        .into());
    }

    // Ensure all hooks are numerically comparable and finite.
    for entry in &report.entries {
        let stats = entry
            .stats
            .ok_or_else(|| format!("missing stats for hook '{}'", entry.key))?;
        if !stats.mean_abs.is_finite() || !stats.max_abs.is_finite() || !stats.rmse.is_finite() {
            return Err(format!(
                "non-finite stats for hook '{}': mean_abs={} max_abs={} rmse={}",
                entry.key, stats.mean_abs, stats.max_abs, stats.rmse
            )
            .into());
        }
    }

    // Strict equality for deterministic preprocess + run metadata tensors.
    for key in [
        "preprocess_image.output",
        "run.image",
        "run.final_resolution",
        "run.sparse_structure_resolution",
    ] {
        let actual_tensor = actual
            .tensors
            .get(key)
            .ok_or_else(|| format!("missing key in actual hook: {key}"))?;
        let reference_tensor = reference
            .tensors
            .get(key)
            .ok_or_else(|| format!("missing key in reference hook: {key}"))?;
        let stats = compute_stats(&actual_tensor.data, &reference_tensor.data);
        if stats.max_abs > 0.0 || stats.mean_abs > 0.0 || stats.rmse > 0.0 {
            return Err(format!(
                "{key} mismatch: mean_abs={:.6e} max_abs={:.6e} rmse={:.6e}",
                stats.mean_abs, stats.max_abs, stats.rmse
            )
            .into());
        }
    }

    // Baseline numerical guard for the sparse latent stage.
    let actual_sparse = actual
        .tensors
        .get("sample_sparse_structure.latent")
        .ok_or("missing sample_sparse_structure.latent in actual hook")?;
    let reference_sparse = reference
        .tensors
        .get("sample_sparse_structure.latent")
        .ok_or("missing sample_sparse_structure.latent in reference hook")?;
    let sparse_stats = compute_stats(&actual_sparse.data, &reference_sparse.data);
    if sparse_stats.mean_abs > 0.5 || sparse_stats.max_abs > 4.0 || sparse_stats.rmse > 0.7 {
        return Err(format!(
            "sample_sparse_structure.latent drift exceeded baseline: mean_abs={:.6e} max_abs={:.6e} rmse={:.6e}",
            sparse_stats.mean_abs, sparse_stats.max_abs, sparse_stats.rmse
        )
        .into());
    }

    Ok(())
}

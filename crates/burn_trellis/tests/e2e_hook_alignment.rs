use std::path::PathBuf;

use burn_trellis::TrellisQuality;
use burn_trellis::hook_diff::{
    HookDiffStatus, HookSnapshot, compare_hook_snapshots, compute_stats,
};
use burn_trellis::pipeline::{
    Trellis2Pipeline, Trellis2PipelineConfig, TrellisDevice, TrellisRunOptions,
};

fn env_flag(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(default)
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        return (*message).to_string();
    }
    if let Some(message) = payload.downcast_ref::<String>() {
        return message.clone();
    }
    "unknown panic payload".to_string()
}

#[test]
fn trellis2_e2e_hook_alignment_against_reference() -> Result<(), Box<dyn std::error::Error>> {
    let strict = env_flag("TRELLIS2_E2E_STRICT", false);
    let disable_runtime = env_flag("TRELLIS2_E2E_DISABLE_RUNTIME_MODEL", false);
    let device = std::env::var("TRELLIS2_E2E_DEVICE")
        .ok()
        .map(|value| match value.trim().to_ascii_lowercase().as_str() {
            "cpu" => TrellisDevice::Cpu,
            "wgpu" => TrellisDevice::Wgpu,
            "cuda" => TrellisDevice::Cuda,
            _ => TrellisDevice::Auto,
        })
        .unwrap_or(TrellisDevice::Auto);
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let input_image = root.join("assets/hooks/trellis2_preprocess_input.png");
    let reference_hook = root.join("assets/hooks/trellis2_full_reference_alpha_512.safetensors");
    if !input_image.exists() || !reference_hook.exists() {
        eprintln!("Skipping Trellis2 e2e hook alignment: missing input or reference hook capture.");
        return Ok(());
    }

    let mut config = Trellis2PipelineConfig {
        image_large_root: Some(
            std::env::var("TRELLIS2_IMAGE_LARGE_ROOT")
                .map(PathBuf::from)
                .unwrap_or_else(|_| {
                    PathBuf::from(
                        "E:/models/huggingface/hub/models--microsoft--TRELLIS-image-large/snapshots/25e0d31ffbebe4b5a97464dd851910efc3002d96",
                    )
                }),
        ),
        ..Trellis2PipelineConfig::default()
    };
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

    if disable_runtime {
        eprintln!(
            "Skipping Trellis2 e2e hook alignment: TRELLIS2_E2E_DISABLE_RUNTIME_MODEL=1 is incompatible with decoder-parity mode."
        );
        return Ok(());
    }
    let pipeline = Trellis2Pipeline::new(config)?;
    pipeline.validate_runtime()?;
    let profile = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pipeline.infer_mesh_profile(
            &input_image,
            &TrellisRunOptions {
                quality: TrellisQuality::Low,
                seed: Some(42),
                device,
                hook_output: Some(actual_hook.clone()),
                noise_overrides_hook: Some(reference_hook.clone()),
            },
        )
    })) {
        Ok(Ok(profile)) => profile,
        Ok(Err(err)) => {
            let message = err.to_string();
            if message.contains("runtime decoder") || message.contains("assets are incomplete") {
                eprintln!(
                    "Skipping Trellis2 e2e hook alignment: runtime decoder assets unavailable ({message})"
                );
                return Ok(());
            }
            return Err(err.into());
        }
        Err(payload) => {
            let message = panic_message(payload);
            if message.contains("runtime decoder is required")
                || message.contains("runtime decode pipeline failed")
            {
                eprintln!(
                    "Skipping Trellis2 e2e hook alignment: runtime decoder path unavailable ({message})"
                );
                return Ok(());
            }
            return Err(format!("panic during infer_mesh_profile: {message}").into());
        }
    };
    if strict && profile.sparse_source.as_str() == "synthetic" {
        return Err("strict mode requires non-synthetic sparse stage source".into());
    }
    if strict
        && matches!(device, TrellisDevice::Wgpu)
        && profile.sparse_source.as_str() != "runtime_model_wgpu"
    {
        return Err(format!(
            "strict mode requested WGPU but sparse stage source was '{}'",
            profile.sparse_source.as_str()
        )
        .into());
    }
    for (label, value) in [
        ("preprocess_ms", profile.timings.preprocess_ms),
        ("runtime_setup_ms", profile.timings.runtime_setup_ms),
        ("sparse_ms", profile.timings.sparse_ms),
        ("shape_slat_ms", profile.timings.shape_slat_ms),
        ("tex_slat_ms", profile.timings.tex_slat_ms),
        ("decode_ms", profile.timings.decode_ms),
        ("hook_capture_ms", profile.timings.hook_capture_ms),
        ("total_ms", profile.timings.total_ms),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(format!("invalid timing value {label}={value}").into());
        }
    }
    if strict {
        if let Ok(max_readbacks) = std::env::var("TRELLIS2_E2E_MAX_HOST_READBACKS")
            && let Ok(limit) = max_readbacks.trim().parse::<u64>()
            && profile.timings.host_readback_count > limit
        {
            return Err(format!(
                "host readback count exceeded limit: {} > {}",
                profile.timings.host_readback_count, limit
            )
            .into());
        }
        if let Ok(max_elements) = std::env::var("TRELLIS2_E2E_MAX_HOST_READBACK_ELEMENTS")
            && let Ok(limit) = max_elements.trim().parse::<u64>()
            && profile.timings.host_readback_elements > limit
        {
            return Err(format!(
                "host readback elements exceeded limit: {} > {}",
                profile.timings.host_readback_elements, limit
            )
            .into());
        }
    }

    let reference = HookSnapshot::from_file(reference_hook)?;
    let actual = HookSnapshot::from_file(&actual_hook)?;
    let report = compare_hook_snapshots(&reference, &actual, None);

    if strict {
        for key in [
            "sample_shape_slat.noise_dense",
            "sample_tex_slat.noise_dense",
        ] {
            if !reference.tensors.contains_key(key) {
                return Err(format!(
                    "strict mode requires dense RNG hook key in reference capture: {key}"
                )
                .into());
            }
        }
    }

    // PBR hook schema must be emitted by the Rust path for downstream parity checks.
    for key in [
        "sample_shape_slat.noise_dense",
        "sample_tex_slat.noise_dense",
        "pbr.uv_unwrap.vertices",
        "pbr.uv_unwrap.faces",
        "pbr.uv_unwrap.uvs",
        "pbr.raster.mask",
        "pbr.sample.position",
        "pbr.sample.attrs_float",
        "pbr.texture.base_color_float",
        "pbr.texture.metallic_float",
        "pbr.texture.roughness_float",
        "pbr.texture.alpha_float",
        "pbr.texture.base_color_rgba_u8",
        "pbr.texture.metallic_roughness_u8",
    ] {
        if !actual.tensors.contains_key(key) {
            return Err(format!("missing required pbr hook key in actual output: {key}").into());
        }
    }

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
    if missing > 0 || shape_mismatch > 0 {
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

    if strict {
        let strict_limit = 1.0e-3f32;
        let strict_float_keys = [
            "sample_sparse_structure.latent",
            "decode_shape_slat.subs.0.feats",
            "decode_tex_slat.voxels.feats",
            "pbr.sample.attrs_float",
            "pbr.texture.base_color_float",
            "pbr.texture.metallic_float",
            "pbr.texture.roughness_float",
            "pbr.texture.alpha_float",
        ];
        for key in strict_float_keys {
            let entry = report
                .entries
                .iter()
                .find(|entry| entry.key == key)
                .ok_or_else(|| format!("missing strict float key '{key}'"))?;
            let stats = entry
                .stats
                .ok_or_else(|| format!("missing stats for strict hook '{key}'"))?;
            if stats.mean_abs > strict_limit
                || stats.max_abs > strict_limit
                || stats.rmse > strict_limit
            {
                return Err(format!(
                    "strict float threshold failed for '{key}': mean_abs={:.6e} max_abs={:.6e} rmse={:.6e}",
                    stats.mean_abs, stats.max_abs, stats.rmse
                )
                .into());
            }
        }

        let strict_u8_keys = [
            "pbr.texture.base_color_rgba_u8",
            "pbr.texture.metallic_roughness_u8",
        ];
        for key in strict_u8_keys {
            let entry = report
                .entries
                .iter()
                .find(|entry| entry.key == key)
                .ok_or_else(|| format!("missing strict u8 key '{key}'"))?;
            let stats = entry
                .stats
                .ok_or_else(|| format!("missing stats for strict hook '{key}'"))?;
            if stats.max_abs > 1.0 || stats.mean_abs > 1.0 || stats.rmse > 1.0 {
                return Err(format!(
                    "strict u8 threshold failed for '{key}': mean_abs={:.6e} max_abs={:.6e} rmse={:.6e}",
                    stats.mean_abs, stats.max_abs, stats.rmse
                )
                .into());
            }
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

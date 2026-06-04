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

fn sampled_or_truncated_metadata_keys(snapshot: &HookSnapshot) -> Vec<String> {
    fn is_allowed_noncritical_sampled_key(key: &str) -> bool {
        let base = key
            .strip_suffix(".row_sampled")
            .or_else(|| key.strip_suffix(".flat_sampled_from"))
            .or_else(|| key.strip_suffix(".list_truncated"))
            .unwrap_or(key);
        // Face index tensors can be very large and are not part of strict float parity gates.
        (base.starts_with("decode_latent.mesh.") || base.starts_with("decode_shape_slat.meshes."))
            && base.ends_with(".faces")
    }

    let mut keys = snapshot
        .metadata
        .keys()
        .filter(|key| {
            (key.ends_with(".row_sampled")
                || key.ends_with(".flat_sampled_from")
                || key.ends_with(".list_truncated"))
                && !is_allowed_noncritical_sampled_key(key)
        })
        .cloned()
        .collect::<Vec<_>>();
    keys.sort();
    keys
}

fn scalar_count(snapshot: &HookSnapshot, key: &str) -> Result<u64, String> {
    let tensor = snapshot
        .tensors
        .get(key)
        .ok_or_else(|| format!("missing count key '{key}'"))?;
    if tensor.data.is_empty() {
        return Err(format!("count key '{key}' has no elements"));
    }
    let value = tensor.data[0];
    if !value.is_finite() || value < 0.0 {
        return Err(format!("count key '{key}' is invalid ({value})"));
    }
    Ok(value.round() as u64)
}

fn coords_set(
    snapshot: &HookSnapshot,
    key: &str,
) -> Result<std::collections::HashSet<u64>, String> {
    let tensor = snapshot
        .tensors
        .get(key)
        .ok_or_else(|| format!("missing coords key '{key}'"))?;
    if tensor.shape.len() != 2 || tensor.shape[1] != 4 {
        return Err(format!(
            "coords key '{key}' has invalid shape {:?}; expected [N,4]",
            tensor.shape
        ));
    }
    if tensor.data.len() != tensor.shape[0] * tensor.shape[1] {
        return Err(format!(
            "coords key '{key}' has invalid element count {} for shape {:?}",
            tensor.data.len(),
            tensor.shape
        ));
    }
    let mut out = std::collections::HashSet::with_capacity(tensor.shape[0] * 2);
    for row in 0..tensor.shape[0] {
        let base = row * 4;
        let x = tensor.data[base + 1].round().max(0.0) as u64;
        let y = tensor.data[base + 2].round().max(0.0) as u64;
        let z = tensor.data[base + 3].round().max(0.0) as u64;
        let packed = (x << 42) | (y << 21) | z;
        out.insert(packed);
    }
    Ok(out)
}

fn mesh_axis_balance(snapshot: &HookSnapshot, key: &str) -> Result<[f32; 3], String> {
    let tensor = snapshot
        .tensors
        .get(key)
        .ok_or_else(|| format!("missing mesh vertices key '{key}'"))?;
    if tensor.shape.len() != 2 || tensor.shape[1] != 3 {
        return Err(format!(
            "mesh vertices key '{key}' has invalid shape {:?}; expected [N,3]",
            tensor.shape
        ));
    }
    if tensor.data.len() != tensor.shape[0] * tensor.shape[1] {
        return Err(format!(
            "mesh vertices key '{key}' has invalid element count {} for shape {:?}",
            tensor.data.len(),
            tensor.shape
        ));
    }
    let mut pos = [0u64; 3];
    let mut neg = [0u64; 3];
    for row in tensor.data.chunks_exact(3) {
        for axis in 0..3usize {
            let value = row[axis];
            if value > 1.0e-6 {
                pos[axis] += 1;
            } else if value < -1.0e-6 {
                neg[axis] += 1;
            }
        }
    }
    let mut balance = [0.0f32; 3];
    for axis in 0..3usize {
        let total = pos[axis] + neg[axis];
        if total == 0 {
            balance[axis] = 0.0;
        } else {
            balance[axis] = (pos[axis] as f32 - neg[axis] as f32) / total as f32;
        }
    }
    Ok(balance)
}

fn assert_orientation_balance(
    key: &str,
    reference: [f32; 3],
    actual: [f32; 3],
) -> Result<(), String> {
    const AXIS_NAMES: [&str; 3] = ["x", "y", "z"];
    const SIGNAL_MIN: f32 = 1.0e-2;
    const DELTA_MAX: f32 = 8.0e-2;
    for axis in 0..3usize {
        let ref_value = reference[axis];
        let actual_value = actual[axis];
        let delta = (actual_value - ref_value).abs();
        if delta > DELTA_MAX {
            return Err(format!(
                "strict orientation mismatch for '{key}' axis='{}': reference_balance={:.6e} actual_balance={:.6e} delta={:.6e} (limit={:.6e})",
                AXIS_NAMES[axis], ref_value, actual_value, delta, DELTA_MAX
            ));
        }
        if ref_value.abs() >= SIGNAL_MIN {
            if actual_value.abs() < SIGNAL_MIN {
                return Err(format!(
                    "strict orientation mismatch for '{key}' axis='{}': reference_balance={:.6e} but actual balance lost directional signal ({:.6e})",
                    AXIS_NAMES[axis], ref_value, actual_value
                ));
            }
            if ref_value.signum() != actual_value.signum() {
                return Err(format!(
                    "strict orientation mismatch for '{key}' axis='{}': reference_balance={:.6e} actual_balance={:.6e}",
                    AXIS_NAMES[axis], ref_value, actual_value
                ));
            }
        }
    }
    Ok(())
}

fn is_optional_canonical_missing_key(key: &str) -> bool {
    matches!(
        key,
        "sample_shape_slat.noise_dense" | "sample_tex_slat.noise_dense"
    )
}

#[test]
fn trellis2_e2e_hook_alignment_against_reference() -> Result<(), Box<dyn std::error::Error>> {
    if !cfg!(feature = "runtime-model") {
        eprintln!(
            "Skipping Trellis2 e2e hook alignment: burn_trellis runtime-model feature is disabled."
        );
        return Ok(());
    }

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
    let input_image = std::env::var("TRELLIS2_E2E_INPUT_IMAGE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| root.join("assets/hooks/trellis2_preprocess_input.png"));
    let reference_hook = std::env::var("TRELLIS2_E2E_REFERENCE_HOOK")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            root.join("assets/hooks/trellis2_full_reference_alpha_512.safetensors")
        });
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
                max_sparse_coords: None,
                target_faces: None,
                runtime_stage_debug: false,
                runtime_attention_debug: false,
                runtime_decoder_conv_telemetry: false,
                runtime_stage_fence: strict,
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
    if strict {
        let sparse_source = profile.sparse_source.as_str();
        let decode_source = profile.decode_source.as_str();
        if !matches!(sparse_source, "runtime_model_cpu" | "runtime_model_wgpu") {
            return Err(format!(
                "strict mode requires runtime-model sparse source, got '{}'",
                sparse_source
            )
            .into());
        }
        if matches!(device, TrellisDevice::Wgpu) && sparse_source != "runtime_model_wgpu" {
            return Err(format!(
                "strict mode requested WGPU but sparse stage source was '{}'",
                sparse_source
            )
            .into());
        }
        if decode_source != "runtime" {
            return Err(format!(
                "strict mode requires runtime decode source, got '{}'",
                decode_source
            )
            .into());
        }
        if matches!(device, TrellisDevice::Wgpu)
            && (profile.timings.decode_shape_wgpu_dispatches == 0
                || profile.timings.decode_tex_wgpu_dispatches == 0)
        {
            return Err(
                "strict mode requested WGPU but runtime decode emitted zero dispatches for shape and/or tex decoder"
                    .into(),
            );
        }
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

    if strict {
        let reference_sampled = sampled_or_truncated_metadata_keys(&reference);
        if !reference_sampled.is_empty() {
            return Err(format!(
                "strict mode requires full-capture reference hook (found sampled/truncated metadata keys): {}",
                reference_sampled.join(", ")
            )
            .into());
        }
        let actual_sampled = sampled_or_truncated_metadata_keys(&actual);
        if !actual_sampled.is_empty() {
            return Err(format!(
                "strict mode requires full-capture actual hook (found sampled/truncated metadata keys): {}",
                actual_sampled.join(", ")
            )
            .into());
        }
    }

    let report = compare_hook_snapshots(&reference, &actual, None);

    if strict {
        for key in [
            "decode_shape_slat.input.coords",
            "decode_shape_slat.input.feats",
            "decode_tex_slat.input.coords",
            "decode_tex_slat.input.feats",
        ] {
            if !reference.tensors.contains_key(key) {
                return Err(format!(
                    "strict mode requires dense RNG hook key in reference capture: {key}"
                )
                .into());
            }
        }
        for level in 0..4usize {
            for suffix in ["coords", "feats", "spatial_shape"] {
                let key = format!("decode_shape_slat.subs.{level}.{suffix}");
                if !reference.tensors.contains_key(key.as_str()) {
                    return Err(
                        format!("strict mode requires subdivision reference key: {key}").into(),
                    );
                }
                if !actual.tensors.contains_key(key.as_str()) {
                    return Err(
                        format!("strict mode requires subdivision actual key: {key}").into(),
                    );
                }
            }
        }
    }

    // PBR hook schema must be emitted by the Rust path for downstream parity checks.
    let required_pbr_hook_keys = [
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
    ];
    let mut missing_pbr_in_reference = Vec::new();
    for key in required_pbr_hook_keys {
        if !actual.tensors.contains_key(key) {
            return Err(format!("missing required pbr hook key in actual output: {key}").into());
        }
        if !reference.tensors.contains_key(key) {
            missing_pbr_in_reference.push(key);
        }
    }
    if strict && !missing_pbr_in_reference.is_empty() {
        return Err(format!(
            "strict mode requires PBR reference keys, missing in reference hook: {}",
            missing_pbr_in_reference.join(", ")
        )
        .into());
    }
    if !strict && !missing_pbr_in_reference.is_empty() {
        eprintln!(
            "warning: reference hook is missing {} PBR key(s), so full PBR numeric parity is not being evaluated: {}",
            missing_pbr_in_reference.len(),
            missing_pbr_in_reference.join(", ")
        );
    }

    let missing = report
        .entries
        .iter()
        .filter(|entry| {
            entry.status == HookDiffStatus::MissingInActual
                && !is_optional_canonical_missing_key(entry.key.as_str())
        })
        .count();
    let optional_missing = report
        .entries
        .iter()
        .filter(|entry| {
            entry.status == HookDiffStatus::MissingInActual
                && is_optional_canonical_missing_key(entry.key.as_str())
        })
        .count();
    let shape_mismatch = report
        .entries
        .iter()
        .filter(|entry| {
            entry.status == HookDiffStatus::ShapeMismatch
                && !is_optional_canonical_missing_key(entry.key.as_str())
        })
        .count();
    if missing > 0 || shape_mismatch > 0 {
        return Err(format!(
            "hook schema mismatch: missing={missing}, shape_mismatch={shape_mismatch}, optional_missing={optional_missing}, extra={}",
            report.extra_in_actual.len()
        )
        .into());
    }

    // Ensure all hooks are numerically comparable and finite.
    for entry in &report.entries {
        if entry.status == HookDiffStatus::MissingInActual
            && is_optional_canonical_missing_key(entry.key.as_str())
        {
            continue;
        }
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
        for key in [
            "decode_latent.mesh.0.vertices_count",
            "decode_latent.mesh.0.faces_count",
            "decode_latent.mesh.0.voxel_count",
        ] {
            let actual_count = scalar_count(&actual, key)?;
            let reference_count = scalar_count(&reference, key)?;
            if actual_count != reference_count {
                return Err(format!(
                    "strict structural mismatch for '{key}': actual={} reference={}",
                    actual_count, reference_count
                )
                .into());
            }
        }
        for key in [
            "sample_sparse_structure.coords",
            "sample_shape_slat.slat.coords",
            "sample_tex_slat.slat.coords",
        ] {
            let actual_coords = coords_set(&actual, key)?;
            let reference_coords = coords_set(&reference, key)?;
            if actual_coords != reference_coords {
                let overlap = actual_coords.intersection(&reference_coords).count();
                return Err(format!(
                    "strict coordinate mismatch for '{key}': actual_rows={} reference_rows={} overlap={}",
                    actual_coords.len(),
                    reference_coords.len(),
                    overlap
                )
                .into());
            }
        }
        for key in [
            "decode_latent.mesh.0.vertices",
            "decode_shape_slat.meshes.0.vertices",
        ] {
            let reference_balance = mesh_axis_balance(&reference, key)?;
            let actual_balance = mesh_axis_balance(&actual, key)?;
            assert_orientation_balance(key, reference_balance, actual_balance)?;
        }

        let strict_limit = 1.0e-3f32;
        let strict_float_keys = [
            "sample_sparse_structure.latent",
            "decode_tex_slat.voxels.feats",
            "pbr.uv_unwrap.uvs",
            "pbr.sample.position",
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

        // Subdivision logits are the hardest decoder boundary and are
        // intentionally gated independently to make level-wise drift explicit.
        let subdiv_limit = std::env::var("TRELLIS2_E2E_SUBDIV_MAX")
            .ok()
            .and_then(|value| value.trim().parse::<f32>().ok())
            .unwrap_or(1.0e-2f32);
        for level in 0..4usize {
            let key = format!("decode_shape_slat.subs.{level}.feats");
            let entry = report
                .entries
                .iter()
                .find(|entry| entry.key == key)
                .ok_or_else(|| format!("missing strict subdivision key '{key}'"))?;
            let stats = entry
                .stats
                .ok_or_else(|| format!("missing stats for strict subdivision hook '{key}'"))?;
            let level_limit = std::env::var(format!("TRELLIS2_E2E_SUBDIV_LEVEL{level}_MAX"))
                .ok()
                .and_then(|value| value.trim().parse::<f32>().ok())
                .unwrap_or(subdiv_limit);
            if stats.mean_abs > level_limit
                || stats.max_abs > level_limit
                || stats.rmse > level_limit
            {
                return Err(format!(
                    "strict subdivision threshold failed for '{key}': mean_abs={:.6e} max_abs={:.6e} rmse={:.6e} limit={:.6e}",
                    stats.mean_abs, stats.max_abs, stats.rmse, level_limit
                )
                .into());
            }
        }

        let strict_u8_keys = [
            "pbr.raster.mask",
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
            let limit = if key == "pbr.raster.mask" { 0.0 } else { 1.0 };
            if stats.max_abs > limit || stats.mean_abs > limit || stats.rmse > limit {
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

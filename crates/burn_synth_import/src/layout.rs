use std::path::{Path, PathBuf};

pub const F16_SUFFIX: &str = "_f16";

pub fn precision_label(use_f16: bool) -> &'static str {
    if use_f16 { "f16" } else { "f32" }
}

pub fn with_file_stem_suffix(path: &Path, suffix: &str) -> PathBuf {
    let Some(stem) = path.file_stem() else {
        return path.to_path_buf();
    };
    let stem = stem.to_string_lossy();
    if stem.ends_with(suffix) {
        return path.to_path_buf();
    }

    let ext = path.extension().and_then(|ext| ext.to_str()).unwrap_or("");
    let mut file_name = format!("{stem}{suffix}");
    if !ext.is_empty() {
        file_name.push('.');
        file_name.push_str(ext);
    }
    path.with_file_name(file_name)
}

pub fn burnpack_path(path: &Path, use_f16: bool) -> PathBuf {
    let path = if path
        .extension()
        .map(|ext| ext.eq_ignore_ascii_case("bpk"))
        .unwrap_or(false)
    {
        path.to_path_buf()
    } else {
        path.with_extension("bpk")
    };

    if use_f16 && !path_has_low_precision_marker(&path) {
        with_file_stem_suffix(&path, F16_SUFFIX)
    } else {
        path
    }
}

pub fn candidate_burnpack_names(base_safetensors_path: &str, prefer_f16: bool) -> Vec<String> {
    let base = if base_safetensors_path
        .to_ascii_lowercase()
        .ends_with(".safetensors")
    {
        &base_safetensors_path[..base_safetensors_path.len() - ".safetensors".len()]
    } else if base_safetensors_path.to_ascii_lowercase().ends_with(".bpk") {
        return vec![base_safetensors_path.to_string()];
    } else {
        base_safetensors_path
    };
    let f32 = format!("{base}.bpk");
    let f16 = format!("{base}_f16.bpk");

    if stem_has_low_precision_marker(base) {
        // Low-precision checkpoints are already bf16/fp16/f16 class; prefer canonical
        // `{stem}.bpk` and keep `{stem}_f16.bpk` only as compatibility fallback.
        return vec![f32, f16];
    }

    if prefer_f16 {
        vec![f16, f32]
    } else {
        vec![f32, f16]
    }
}

pub fn stem_has_low_precision_marker(stem: &str) -> bool {
    let normalized = stem
        .replace('\\', "/")
        .rsplit('/')
        .next()
        .unwrap_or(stem)
        .to_ascii_lowercase();

    normalized.contains("_bf16")
        || normalized.contains("-bf16")
        || normalized.ends_with("bf16")
        || normalized.contains("_fp16")
        || normalized.contains("-fp16")
        || normalized.ends_with("fp16")
        || normalized.ends_with("_f16")
}

fn path_has_low_precision_marker(path: &Path) -> bool {
    path.file_stem()
        .and_then(|value| value.to_str())
        .is_some_and(stem_has_low_precision_marker)
}

#[cfg(test)]
mod tests {
    use super::{burnpack_path, candidate_burnpack_names, stem_has_low_precision_marker};
    use std::path::Path;

    #[test]
    fn low_precision_stems_do_not_prefer_artificial_f16_suffix() {
        let candidates = candidate_burnpack_names("ckpts/model_1_3B_1024_bf16.safetensors", true);
        assert_eq!(
            candidates,
            vec![
                "ckpts/model_1_3B_1024_bf16.bpk",
                "ckpts/model_1_3B_1024_bf16_f16.bpk"
            ]
        );
    }

    #[test]
    fn burnpack_path_keeps_base_for_low_precision_sources() {
        let path = burnpack_path(
            Path::new("ckpts/shape_dec_next_dc_f16c32_fp16.safetensors"),
            true,
        );
        assert_eq!(path, Path::new("ckpts/shape_dec_next_dc_f16c32_fp16.bpk"));
    }

    #[test]
    fn marker_detection_covers_trellis_naming_patterns() {
        assert!(stem_has_low_precision_marker(
            "ss_flow_img_dit_1_3B_64_bf16"
        ));
        assert!(stem_has_low_precision_marker(
            "shape_dec_next_dc_f16c32_fp16"
        ));
        assert!(stem_has_low_precision_marker("foo_f16"));
        assert!(!stem_has_low_precision_marker("diffusion_pytorch_model"));
    }
}

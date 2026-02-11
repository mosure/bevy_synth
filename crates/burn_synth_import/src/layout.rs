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

    if use_f16 {
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
    let f16 = format!("{base}_f16.bpk");
    let f32 = format!("{base}.bpk");
    if prefer_f16 {
        vec![f16, f32]
    } else {
        vec![f32, f16]
    }
}

pub fn burnpack_manifest_candidates(candidate_burnpack_path: &str) -> [String; 2] {
    [
        format!("{candidate_burnpack_path}.shards.json"),
        format!("{candidate_burnpack_path}.manifest.json"),
    ]
}

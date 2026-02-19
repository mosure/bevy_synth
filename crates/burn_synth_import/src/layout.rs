use std::path::{Path, PathBuf};

pub const F16_SUFFIX: &str = "_f16";
pub const FP8_SUFFIX: &str = "_fp8";
pub const Q4_SUFFIX: &str = "_q4";

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BurnpackPrecision {
    F32,
    F16,
    Fp8,
    Q4,
}

impl BurnpackPrecision {
    pub const fn label(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Fp8 => "fp8",
            Self::Q4 => "q4",
        }
    }

    pub const fn suffix(self) -> &'static str {
        match self {
            Self::F32 => "",
            Self::F16 => F16_SUFFIX,
            Self::Fp8 => FP8_SUFFIX,
            Self::Q4 => Q4_SUFFIX,
        }
    }
}

pub fn precision_label(use_f16: bool) -> &'static str {
    if use_f16 {
        BurnpackPrecision::F16.label()
    } else {
        BurnpackPrecision::F32.label()
    }
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
    let precision = if use_f16 {
        BurnpackPrecision::F16
    } else {
        BurnpackPrecision::F32
    };
    burnpack_path_with_precision(path, precision)
}

pub fn burnpack_path_with_precision(path: &Path, precision: BurnpackPrecision) -> PathBuf {
    let path = if path
        .extension()
        .map(|ext| ext.eq_ignore_ascii_case("bpk"))
        .unwrap_or(false)
    {
        path.to_path_buf()
    } else {
        path.with_extension("bpk")
    };

    if precision == BurnpackPrecision::F32 {
        path
    } else {
        with_file_stem_suffix(&path, precision.suffix())
    }
}

pub fn candidate_burnpack_names(base_safetensors_path: &str, prefer_f16: bool) -> Vec<String> {
    let order = if prefer_f16 {
        vec![BurnpackPrecision::F16, BurnpackPrecision::F32]
    } else {
        vec![BurnpackPrecision::F32, BurnpackPrecision::F16]
    };
    candidate_burnpack_names_for_order(base_safetensors_path, order.as_slice())
}

pub fn candidate_burnpack_names_for_precision(
    base_safetensors_path: &str,
    preferred: BurnpackPrecision,
    allow_cross_precision_fallback: bool,
) -> Vec<String> {
    let order = if allow_cross_precision_fallback {
        match preferred {
            BurnpackPrecision::F16 => vec![
                BurnpackPrecision::F16,
                BurnpackPrecision::F32,
                BurnpackPrecision::Fp8,
                BurnpackPrecision::Q4,
            ],
            BurnpackPrecision::F32 => vec![
                BurnpackPrecision::F32,
                BurnpackPrecision::F16,
                BurnpackPrecision::Fp8,
                BurnpackPrecision::Q4,
            ],
            BurnpackPrecision::Fp8 => vec![
                BurnpackPrecision::Fp8,
                BurnpackPrecision::F16,
                BurnpackPrecision::F32,
                BurnpackPrecision::Q4,
            ],
            BurnpackPrecision::Q4 => vec![
                BurnpackPrecision::Q4,
                BurnpackPrecision::F16,
                BurnpackPrecision::F32,
                BurnpackPrecision::Fp8,
            ],
        }
    } else {
        vec![preferred]
    };
    candidate_burnpack_names_for_order(base_safetensors_path, order.as_slice())
}

pub fn candidate_burnpack_names_for_order(
    base_safetensors_path: &str,
    order: &[BurnpackPrecision],
) -> Vec<String> {
    if base_safetensors_path.to_ascii_lowercase().ends_with(".bpk") {
        return vec![base_safetensors_path.to_string()];
    }

    let base = if base_safetensors_path
        .to_ascii_lowercase()
        .ends_with(".safetensors")
    {
        &base_safetensors_path[..base_safetensors_path.len() - ".safetensors".len()]
    } else {
        base_safetensors_path
    };
    let base = strip_known_precision_suffix(base);
    let mut out = Vec::new();
    for precision in order {
        let name = if *precision == BurnpackPrecision::F32 {
            format!("{base}.bpk")
        } else {
            format!("{base}{}.bpk", precision.suffix())
        };
        if !out.iter().any(|existing| existing == &name) {
            out.push(name);
        }
    }
    out
}

fn strip_known_precision_suffix(base: &str) -> &str {
    for suffix in [F16_SUFFIX, FP8_SUFFIX, Q4_SUFFIX] {
        if let Some(stripped) = base.strip_suffix(suffix) {
            return stripped;
        }
    }
    base
}

#[cfg(test)]
mod tests {
    use super::{
        BurnpackPrecision, burnpack_path_with_precision, candidate_burnpack_names_for_precision,
    };

    #[test]
    fn burnpack_paths_include_precision_suffixes() {
        let path = std::path::Path::new("model.safetensors");
        assert_eq!(
            burnpack_path_with_precision(path, BurnpackPrecision::F32),
            std::path::PathBuf::from("model.bpk")
        );
        assert_eq!(
            burnpack_path_with_precision(path, BurnpackPrecision::F16),
            std::path::PathBuf::from("model_f16.bpk")
        );
        assert_eq!(
            burnpack_path_with_precision(path, BurnpackPrecision::Fp8),
            std::path::PathBuf::from("model_fp8.bpk")
        );
        assert_eq!(
            burnpack_path_with_precision(path, BurnpackPrecision::Q4),
            std::path::PathBuf::from("model_q4.bpk")
        );
    }

    #[test]
    fn precision_candidates_include_fallbacks_in_order() {
        let fp8 = candidate_burnpack_names_for_precision(
            "weights/model.safetensors",
            BurnpackPrecision::Fp8,
            true,
        );
        assert_eq!(
            fp8,
            vec![
                "weights/model_fp8.bpk",
                "weights/model_f16.bpk",
                "weights/model.bpk",
                "weights/model_q4.bpk"
            ]
        );

        let strict_q4 = candidate_burnpack_names_for_precision(
            "weights/model.safetensors",
            BurnpackPrecision::Q4,
            false,
        );
        assert_eq!(strict_q4, vec!["weights/model_q4.bpk"]);
    }
}

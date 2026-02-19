use std::path::{Path, PathBuf};

const DEFAULT_F16_SUFFIX: &str = "_f16";
const DEFAULT_FP8_SUFFIX: &str = "_fp8";
const DEFAULT_Q4_SUFFIX: &str = "_q4";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BpkPrecision {
    F32,
    F16,
    Fp8,
    Q4,
}

impl BpkPrecision {
    pub const fn suffix(self) -> &'static str {
        match self {
            Self::F32 => "",
            Self::F16 => DEFAULT_F16_SUFFIX,
            Self::Fp8 => DEFAULT_FP8_SUFFIX,
            Self::Q4 => DEFAULT_Q4_SUFFIX,
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Fp8 => "fp8",
            Self::Q4 => "q4",
        }
    }
}

/// Preferred precision order when selecting burnpack weight files.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BpkPrecisionPreference {
    /// Prefer `_f16.bpk` files first, then fallback to `.bpk`.
    PreferF16,
    /// Prefer `.bpk` files first, then fallback to `_f16.bpk`.
    PreferF32,
    /// Prefer `_fp8.bpk` files first.
    PreferFp8,
    /// Prefer `_q4.bpk` files first.
    PreferQ4,
}

impl BpkPrecisionPreference {
    pub const fn preferred(self) -> BpkPrecision {
        match self {
            Self::PreferF16 => BpkPrecision::F16,
            Self::PreferF32 => BpkPrecision::F32,
            Self::PreferFp8 => BpkPrecision::Fp8,
            Self::PreferQ4 => BpkPrecision::Q4,
        }
    }

    pub const fn prefer_f16(self) -> bool {
        matches!(self.preferred(), BpkPrecision::F16)
    }
}

/// Burnpack path selection policy used by model loaders.
///
/// Prefer passing this explicitly from runtime configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BurnpackLoadPolicy {
    pub precision: BpkPrecisionPreference,
    pub allow_cross_precision_fallback: bool,
    pub f16_suffix: &'static str,
    pub fp8_suffix: &'static str,
    pub q4_suffix: &'static str,
}

impl Default for BurnpackLoadPolicy {
    fn default() -> Self {
        Self {
            precision: BpkPrecisionPreference::PreferF32,
            allow_cross_precision_fallback: true,
            f16_suffix: DEFAULT_F16_SUFFIX,
            fp8_suffix: DEFAULT_FP8_SUFFIX,
            q4_suffix: DEFAULT_Q4_SUFFIX,
        }
    }
}

impl BurnpackLoadPolicy {
    pub const fn with_precision(self, precision: BpkPrecisionPreference) -> Self {
        Self { precision, ..self }
    }

    pub const fn with_f16_suffix(self, f16_suffix: &'static str) -> Self {
        Self { f16_suffix, ..self }
    }

    pub const fn with_fp8_suffix(self, fp8_suffix: &'static str) -> Self {
        Self { fp8_suffix, ..self }
    }

    pub const fn with_q4_suffix(self, q4_suffix: &'static str) -> Self {
        Self { q4_suffix, ..self }
    }

    pub const fn with_allow_cross_precision_fallback(self, allow: bool) -> Self {
        Self {
            allow_cross_precision_fallback: allow,
            ..self
        }
    }
}

pub fn candidate_burnpack_paths(path: &Path, policy: BurnpackLoadPolicy) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for precision in candidate_precisions(policy) {
        let candidate = burnpack_path_for_precision(path, precision, policy);
        if !out.iter().any(|existing| existing == &candidate) {
            out.push(candidate);
        }
    }
    out
}

pub fn burnpack_path(path: &Path, use_f16: bool, f16_suffix: &str) -> PathBuf {
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
        with_file_stem_suffix(&path, f16_suffix)
    } else {
        path
    }
}

pub fn burnpack_path_for_precision(
    path: &Path,
    precision: BpkPrecision,
    policy: BurnpackLoadPolicy,
) -> PathBuf {
    let path = if path
        .extension()
        .map(|ext| ext.eq_ignore_ascii_case("bpk"))
        .unwrap_or(false)
    {
        path.to_path_buf()
    } else {
        path.with_extension("bpk")
    };

    let suffix = match precision {
        BpkPrecision::F32 => "",
        BpkPrecision::F16 => policy.f16_suffix,
        BpkPrecision::Fp8 => policy.fp8_suffix,
        BpkPrecision::Q4 => policy.q4_suffix,
    };
    if suffix.is_empty() {
        path
    } else {
        with_file_stem_suffix(&path, suffix)
    }
}

fn candidate_precisions(policy: BurnpackLoadPolicy) -> Vec<BpkPrecision> {
    if !policy.allow_cross_precision_fallback {
        return vec![policy.precision.preferred()];
    }

    match policy.precision {
        BpkPrecisionPreference::PreferF32 => vec![
            BpkPrecision::F32,
            BpkPrecision::F16,
            BpkPrecision::Fp8,
            BpkPrecision::Q4,
        ],
        BpkPrecisionPreference::PreferF16 => vec![
            BpkPrecision::F16,
            BpkPrecision::F32,
            BpkPrecision::Fp8,
            BpkPrecision::Q4,
        ],
        BpkPrecisionPreference::PreferFp8 => vec![
            BpkPrecision::Fp8,
            BpkPrecision::F16,
            BpkPrecision::F32,
            BpkPrecision::Q4,
        ],
        BpkPrecisionPreference::PreferQ4 => vec![
            BpkPrecision::Q4,
            BpkPrecision::F16,
            BpkPrecision::F32,
            BpkPrecision::Fp8,
        ],
    }
}

fn with_file_stem_suffix(path: &Path, suffix: &str) -> PathBuf {
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

#[cfg(test)]
mod tests {
    use super::{
        BpkPrecision, BpkPrecisionPreference, BurnpackLoadPolicy, burnpack_path,
        burnpack_path_for_precision, candidate_burnpack_paths,
    };

    #[test]
    fn default_policy_prefers_f32() {
        let default = BurnpackLoadPolicy::default();
        assert_eq!(default.precision, BpkPrecisionPreference::PreferF32);
    }

    #[test]
    fn path_candidates_follow_precision_preference() {
        let path = std::path::Path::new("model.safetensors");

        let f32_first_default = candidate_burnpack_paths(path, BurnpackLoadPolicy::default());
        assert_eq!(f32_first_default[0], burnpack_path(path, false, "_f16"));
        assert_eq!(f32_first_default[1], burnpack_path(path, true, "_f16"));
        assert_eq!(
            f32_first_default[2],
            burnpack_path_for_precision(path, BpkPrecision::Fp8, BurnpackLoadPolicy::default())
        );
        assert_eq!(
            f32_first_default[3],
            burnpack_path_for_precision(path, BpkPrecision::Q4, BurnpackLoadPolicy::default())
        );

        let f16_first = candidate_burnpack_paths(
            path,
            BurnpackLoadPolicy::default().with_precision(BpkPrecisionPreference::PreferF16),
        );
        assert_eq!(f16_first[0], burnpack_path(path, true, "_f16"));
        assert_eq!(f16_first[1], burnpack_path(path, false, "_f16"));
    }

    #[test]
    fn strict_precision_policy_only_checks_requested_variant() {
        let path = std::path::Path::new("model.safetensors");
        let strict = candidate_burnpack_paths(
            path,
            BurnpackLoadPolicy::default()
                .with_precision(BpkPrecisionPreference::PreferQ4)
                .with_allow_cross_precision_fallback(false),
        );
        assert_eq!(strict.len(), 1);
        assert_eq!(
            strict[0],
            burnpack_path_for_precision(path, BpkPrecision::Q4, BurnpackLoadPolicy::default())
        );
    }
}

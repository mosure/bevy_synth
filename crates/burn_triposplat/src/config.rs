use crate::flow::CfgPredictionMode;

pub const DEFAULT_NUM_STEPS: usize = 20;
pub const DEFAULT_GUIDANCE_SCALE: f32 = 3.0;
pub const DEFAULT_SHIFT: f32 = 3.0;
pub const DEFAULT_SEED: u64 = 42;
pub const DEFAULT_ERODE_RADIUS: usize = 1;
pub const TRIPOSPLAT_CANONICAL_CANVAS_SIZE: usize = 1024;
pub const TRIPOSPLAT_FAST_VAE_TOKEN_LENGTH: usize = 4096;
pub const TRIPOSPLAT_DINOV3_PREFIX_TOKENS: usize = 5;
pub const TRIPOSPLAT_FAST_DINOV3_TOKEN_LENGTH: usize =
    TRIPOSPLAT_FAST_VAE_TOKEN_LENGTH + TRIPOSPLAT_DINOV3_PREFIX_TOKENS;
pub const TRIPOSPLAT_FLOW_LATENT_TOKEN_LENGTH: usize = 8192;
pub const DEFAULT_Q_TOKEN_LENGTH: usize = TRIPOSPLAT_FLOW_LATENT_TOKEN_LENGTH;
pub const TRIPOSPLAT_GAUSSIANS_PER_POINT: usize = 32;
pub const MIN_NUM_GAUSSIANS: usize = 32_768;
pub const MAX_NUM_GAUSSIANS: usize = 262_144;
pub const DEFAULT_NUM_GAUSSIANS: usize = MAX_NUM_GAUSSIANS;
pub const LOW_PROFILE_NUM_STEPS: usize = 5;
pub const HIGH_PROFILE_NUM_STEPS: usize = 50;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TripoSplatOptions {
    pub steps: usize,
    pub guidance_scale: f32,
    pub shift: f32,
    pub seed: u64,
    pub num_gaussians: usize,
    pub erode_radius: usize,
    pub attention_query_chunk_tokens: Option<usize>,
    pub cfg_mode: CfgPredictionMode,
}

impl Default for TripoSplatOptions {
    fn default() -> Self {
        Self {
            steps: DEFAULT_NUM_STEPS,
            guidance_scale: DEFAULT_GUIDANCE_SCALE,
            shift: DEFAULT_SHIFT,
            seed: DEFAULT_SEED,
            num_gaussians: DEFAULT_NUM_GAUSSIANS,
            erode_radius: DEFAULT_ERODE_RADIUS,
            attention_query_chunk_tokens: None,
            cfg_mode: CfgPredictionMode::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TripoSplatProfile {
    Low,
    #[default]
    Balanced,
    High,
    Custom,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TripoSplatProfileSettings {
    pub steps: usize,
    pub guidance_scale: f32,
    pub num_gaussians: usize,
}

impl TripoSplatProfile {
    pub fn settings(self) -> TripoSplatProfileSettings {
        match self {
            Self::Low => TripoSplatProfileSettings {
                steps: LOW_PROFILE_NUM_STEPS,
                guidance_scale: DEFAULT_GUIDANCE_SCALE,
                num_gaussians: MIN_NUM_GAUSSIANS,
            },
            Self::Balanced | Self::Custom => TripoSplatProfileSettings {
                steps: DEFAULT_NUM_STEPS,
                guidance_scale: DEFAULT_GUIDANCE_SCALE,
                num_gaussians: DEFAULT_NUM_GAUSSIANS,
            },
            Self::High => TripoSplatProfileSettings {
                steps: HIGH_PROFILE_NUM_STEPS,
                guidance_scale: DEFAULT_GUIDANCE_SCALE,
                num_gaussians: MAX_NUM_GAUSSIANS,
            },
        }
    }
}

pub fn triposplat_profile_for_settings(
    steps: usize,
    guidance_scale: f32,
    num_gaussians: usize,
) -> TripoSplatProfile {
    for profile in [
        TripoSplatProfile::Low,
        TripoSplatProfile::Balanced,
        TripoSplatProfile::High,
    ] {
        let settings = profile.settings();
        if settings.steps == steps
            && settings.num_gaussians == num_gaussians
            && (settings.guidance_scale - guidance_scale).abs() <= f32::EPSILON
        {
            return profile;
        }
    }
    TripoSplatProfile::Custom
}

pub fn normalize_num_gaussians(num_gaussians: usize) -> Result<usize, String> {
    if !(MIN_NUM_GAUSSIANS..=MAX_NUM_GAUSSIANS).contains(&num_gaussians) {
        return Err(format!(
            "num_gaussians must be in [{MIN_NUM_GAUSSIANS}, {MAX_NUM_GAUSSIANS}], got {num_gaussians}"
        ));
    }
    let rem = num_gaussians % TRIPOSPLAT_GAUSSIANS_PER_POINT;
    if rem == 0 {
        return Ok(num_gaussians);
    }
    let down = num_gaussians - rem;
    let up = (down + TRIPOSPLAT_GAUSSIANS_PER_POINT).min(MAX_NUM_GAUSSIANS);
    if num_gaussians - down < up - num_gaussians {
        Ok(down.max(MIN_NUM_GAUSSIANS))
    } else {
        Ok(up)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_to_gaussians_per_point_multiple() {
        assert_eq!(normalize_num_gaussians(32_769).unwrap(), 32_768);
        assert_eq!(normalize_num_gaussians(32_783).unwrap(), 32_768);
        assert_eq!(normalize_num_gaussians(32_784).unwrap(), 32_800);
    }

    #[test]
    fn rejects_out_of_range_counts() {
        assert!(normalize_num_gaussians(MIN_NUM_GAUSSIANS - 1).is_err());
        assert!(normalize_num_gaussians(MAX_NUM_GAUSSIANS + 1).is_err());
    }

    #[test]
    fn defaults_match_upstream_fast_path_token_contract() {
        assert_eq!(TRIPOSPLAT_CANONICAL_CANVAS_SIZE, 1024);
        assert_eq!(TRIPOSPLAT_FAST_VAE_TOKEN_LENGTH, 4096);
        assert_eq!(TRIPOSPLAT_DINOV3_PREFIX_TOKENS, 5);
        assert_eq!(TRIPOSPLAT_FAST_DINOV3_TOKEN_LENGTH, 4101);
        assert_eq!(TRIPOSPLAT_FLOW_LATENT_TOKEN_LENGTH, 8192);
        assert_eq!(DEFAULT_Q_TOKEN_LENGTH, TRIPOSPLAT_FLOW_LATENT_TOKEN_LENGTH);
    }

    #[test]
    fn profiles_match_upstream_triposplat_ranges() {
        let low = TripoSplatProfile::Low.settings();
        assert_eq!(low.steps, 5);
        assert_eq!(low.guidance_scale, DEFAULT_GUIDANCE_SCALE);
        assert_eq!(low.num_gaussians, MIN_NUM_GAUSSIANS);

        let balanced = TripoSplatProfile::Balanced.settings();
        assert_eq!(balanced.steps, DEFAULT_NUM_STEPS);
        assert_eq!(balanced.guidance_scale, DEFAULT_GUIDANCE_SCALE);
        assert_eq!(balanced.num_gaussians, DEFAULT_NUM_GAUSSIANS);

        let high = TripoSplatProfile::High.settings();
        assert_eq!(high.steps, 50);
        assert_eq!(high.guidance_scale, DEFAULT_GUIDANCE_SCALE);
        assert_eq!(high.num_gaussians, MAX_NUM_GAUSSIANS);
    }

    #[test]
    fn default_options_use_batched_main_cfg_prediction() {
        assert_eq!(
            TripoSplatOptions::default().cfg_mode,
            CfgPredictionMode::BatchedMain
        );
    }
}

use clap::ValueEnum;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TrellisQuality {
    Low,
    #[default]
    Medium,
    High,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrellisQualitySettings {
    pub pipeline_type: &'static str,
    pub max_num_tokens: Option<usize>,
    pub sparse_steps: usize,
    pub shape_steps: usize,
    pub texture_steps: usize,
    pub guidance_strength_sparse: f32,
    pub guidance_strength_shape: f32,
    pub guidance_strength_texture: f32,
}

impl TrellisQuality {
    pub fn settings(self) -> TrellisQualitySettings {
        match self {
            // 512 base path favors speed over detail.
            Self::Low => TrellisQualitySettings {
                pipeline_type: "512_base",
                max_num_tokens: None,
                // Keep canonical TRELLIS.2 sampler budgets even on 512-base.
                sparse_steps: 12,
                shape_steps: 12,
                texture_steps: 12,
                guidance_strength_sparse: 7.5,
                guidance_strength_shape: 7.5,
                guidance_strength_texture: 1.0,
            },
            // Medium tracks TRELLIS.2 canonical 1024 cascade defaults.
            Self::Medium => TrellisQualitySettings {
                pipeline_type: "1024_cascade",
                max_num_tokens: Some(49_152),
                sparse_steps: 12,
                shape_steps: 12,
                texture_steps: 12,
                guidance_strength_sparse: 7.5,
                guidance_strength_shape: 7.5,
                guidance_strength_texture: 1.0,
            },
            // High uses TRELLIS.2 canonical 1024 cascade path with full step budget.
            Self::High => TrellisQualitySettings {
                pipeline_type: "1024_cascade",
                max_num_tokens: Some(49_152),
                sparse_steps: 12,
                shape_steps: 12,
                texture_steps: 12,
                guidance_strength_sparse: 7.5,
                guidance_strength_shape: 7.5,
                guidance_strength_texture: 1.0,
            },
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::TrellisQuality;

    #[test]
    fn quality_pipeline_types_are_stable() {
        assert_eq!(TrellisQuality::Low.settings().pipeline_type, "512_base");
        assert_eq!(
            TrellisQuality::Medium.settings().pipeline_type,
            "1024_cascade"
        );
        assert_eq!(
            TrellisQuality::High.settings().pipeline_type,
            "1024_cascade"
        );
    }

    #[test]
    fn quality_step_budgets_are_ordered() {
        let low = TrellisQuality::Low.settings();
        let medium = TrellisQuality::Medium.settings();
        let high = TrellisQuality::High.settings();
        assert!(low.sparse_steps <= medium.sparse_steps);
        assert!(medium.sparse_steps <= high.sparse_steps);
        assert!(low.shape_steps <= medium.shape_steps);
        assert!(medium.shape_steps <= high.shape_steps);
        assert!(low.texture_steps <= medium.texture_steps);
        assert!(medium.texture_steps <= high.texture_steps);
    }

    #[test]
    fn quality_token_caps_are_stable() {
        let low = TrellisQuality::Low.settings();
        let medium = TrellisQuality::Medium.settings();
        let high = TrellisQuality::High.settings();
        assert_eq!(low.max_num_tokens, None);
        assert_eq!(medium.max_num_tokens, Some(49_152));
        assert_eq!(high.max_num_tokens, Some(49_152));
    }
}

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
                sparse_steps: 1,
                shape_steps: 1,
                texture_steps: 1,
                guidance_strength_sparse: 6.0,
                guidance_strength_shape: 6.0,
                guidance_strength_texture: 1.0,
            },
            // Medium keeps the canonical 1024 single-pass path with fewer sampler steps.
            Self::Medium => TrellisQualitySettings {
                pipeline_type: "1024_single",
                sparse_steps: 6,
                shape_steps: 6,
                texture_steps: 6,
                guidance_strength_sparse: 7.5,
                guidance_strength_shape: 7.5,
                guidance_strength_texture: 1.0,
            },
            // High keeps the same canonical 1024 single-pass pipeline with full step budget.
            Self::High => TrellisQualitySettings {
                pipeline_type: "1024_single",
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
            "1024_single"
        );
        assert_eq!(TrellisQuality::High.settings().pipeline_type, "1024_single");
    }

    #[test]
    fn quality_step_budgets_are_ordered() {
        let low = TrellisQuality::Low.settings();
        let medium = TrellisQuality::Medium.settings();
        let high = TrellisQuality::High.settings();
        assert!(low.sparse_steps < medium.sparse_steps);
        assert!(medium.sparse_steps < high.sparse_steps);
        assert!(low.shape_steps < medium.shape_steps);
        assert!(medium.shape_steps < high.shape_steps);
        assert!(low.texture_steps < medium.texture_steps);
        assert!(medium.texture_steps < high.texture_steps);
    }
}

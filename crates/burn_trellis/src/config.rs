use clap::ValueEnum;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TrellisQuality {
    Low,
    Medium,
    High,
}

impl Default for TrellisQuality {
    fn default() -> Self {
        Self::Medium
    }
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
                sparse_steps: 8,
                shape_steps: 8,
                texture_steps: 8,
                guidance_strength_sparse: 6.0,
                guidance_strength_shape: 6.0,
                guidance_strength_texture: 1.0,
            },
            // 1024 single keeps memory usage lower while preserving quality.
            Self::Medium => TrellisQualitySettings {
                pipeline_type: "1024_single",
                sparse_steps: 12,
                shape_steps: 12,
                texture_steps: 12,
                guidance_strength_sparse: 7.5,
                guidance_strength_shape: 7.5,
                guidance_strength_texture: 1.0,
            },
            // 1024 cascade is the highest quality default path.
            Self::High => TrellisQualitySettings {
                pipeline_type: "1024_cascade",
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
        assert_eq!(
            TrellisQuality::High.settings().pipeline_type,
            "1024_cascade"
        );
    }
}

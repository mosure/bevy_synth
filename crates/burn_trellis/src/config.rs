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

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TrellisComputeProfile {
    #[default]
    ReferenceF32,
    StockBf16Emulated,
    WgpuFastMixedF16,
    WgpuFastSparseSelfF16,
    WgpuFastSparseCrossF16,
    WgpuFastF16Tail1F32,
    WgpuFastF16Tail2F32,
    WgpuFastF16Tail4F32,
    WgpuFastF16Tail6F32,
    WgpuFastF16,
}

impl TrellisComputeProfile {
    pub const FAST_WGPU_BENCHMARK_CANDIDATE: Self = Self::WgpuFastF16Tail1F32;

    pub fn as_str(self) -> &'static str {
        match self {
            Self::ReferenceF32 => "reference-f32",
            Self::StockBf16Emulated => "stock-bf16-emulated",
            Self::WgpuFastMixedF16 => "wgpu-fast-mixed-f16",
            Self::WgpuFastSparseSelfF16 => "wgpu-fast-sparse-self-f16",
            Self::WgpuFastSparseCrossF16 => "wgpu-fast-sparse-cross-f16",
            Self::WgpuFastF16Tail1F32 => "wgpu-fast-f16-tail1-f32",
            Self::WgpuFastF16Tail2F32 => "wgpu-fast-f16-tail2-f32",
            Self::WgpuFastF16Tail4F32 => "wgpu-fast-f16-tail4-f32",
            Self::WgpuFastF16Tail6F32 => "wgpu-fast-f16-tail6-f32",
            Self::WgpuFastF16 => "wgpu-fast-f16",
        }
    }

    pub fn wgpu_module_attention_f16(self) -> bool {
        self.wgpu_sparse_self_attention_f16()
            || self.wgpu_sparse_cross_attention_f16()
            || self.wgpu_slat_module_attention_f16()
    }

    pub fn wgpu_sparse_module_attention_f16(self) -> bool {
        self.wgpu_sparse_self_attention_f16() || self.wgpu_sparse_cross_attention_f16()
    }

    pub fn wgpu_sparse_self_attention_f16(self) -> bool {
        matches!(
            self,
            Self::WgpuFastSparseSelfF16
                | Self::WgpuFastF16Tail1F32
                | Self::WgpuFastF16Tail2F32
                | Self::WgpuFastF16Tail4F32
                | Self::WgpuFastF16Tail6F32
                | Self::WgpuFastF16
        )
    }

    pub fn wgpu_sparse_cross_attention_f16(self) -> bool {
        matches!(
            self,
            Self::WgpuFastSparseCrossF16
                | Self::WgpuFastF16Tail1F32
                | Self::WgpuFastF16Tail2F32
                | Self::WgpuFastF16Tail4F32
                | Self::WgpuFastF16Tail6F32
                | Self::WgpuFastF16
        )
    }

    pub fn wgpu_sparse_final_f32_steps(self) -> usize {
        match self {
            Self::WgpuFastF16Tail1F32 => 1,
            Self::WgpuFastF16Tail2F32 => 2,
            Self::WgpuFastF16Tail4F32 => 4,
            Self::WgpuFastF16Tail6F32 => 6,
            _ => 0,
        }
    }

    pub fn wgpu_slat_module_attention_f16(self) -> bool {
        !matches!(self, Self::ReferenceF32 | Self::StockBf16Emulated)
    }

    pub fn wgpu_linear_f16(self) -> bool {
        // The native WGPU f16 linear bridge currently violates SLat parity on
        // HR flow hooks. Keep it opt-in for diagnostics instead of enabling it
        // through the production fast profile.
        false
    }

    pub fn wgpu_flow_torso_f16(self) -> bool {
        false
    }

    pub fn wgpu_decoder_conv_f16(self) -> bool {
        // The TRELLIS.2 sparse decoders are fp16 models upstream. Keeping the
        // flow/sampler in f32 while running decoder conv activations in fp16 is
        // both faster and closer to Python subdivision/topology hooks.
        !matches!(self, Self::StockBf16Emulated)
    }

    pub fn wgpu_sparse_stock_bf16_emulation(self) -> bool {
        matches!(self, Self::StockBf16Emulated)
    }

    pub fn is_diagnostic(self) -> bool {
        matches!(
            self,
            Self::StockBf16Emulated
                | Self::WgpuFastSparseSelfF16
                | Self::WgpuFastSparseCrossF16
                | Self::WgpuFastF16
        )
    }

    pub fn is_parity_reference(self) -> bool {
        matches!(self, Self::ReferenceF32)
    }

    pub fn is_fast_candidate(self) -> bool {
        matches!(
            self,
            Self::WgpuFastMixedF16
                | Self::WgpuFastF16Tail1F32
                | Self::WgpuFastF16Tail2F32
                | Self::WgpuFastF16Tail4F32
                | Self::WgpuFastF16Tail6F32
        )
    }

    pub fn requires_render_or_hook_parity_gate(self) -> bool {
        !self.is_parity_reference()
    }
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
    use super::{TrellisComputeProfile, TrellisQuality};

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

    #[test]
    fn wgpu_fast_f16_keeps_linear_f16_disabled_until_parity_is_fixed() {
        assert!(TrellisComputeProfile::WgpuFastF16.wgpu_module_attention_f16());
        assert!(TrellisComputeProfile::WgpuFastF16.wgpu_sparse_module_attention_f16());
        assert!(TrellisComputeProfile::WgpuFastF16.wgpu_slat_module_attention_f16());
        assert!(!TrellisComputeProfile::WgpuFastF16.wgpu_linear_f16());
        assert!(!TrellisComputeProfile::WgpuFastF16.wgpu_flow_torso_f16());
    }

    #[test]
    fn wgpu_fast_mixed_f16_keeps_sparse_structure_attention_f32() {
        assert!(TrellisComputeProfile::WgpuFastMixedF16.wgpu_module_attention_f16());
        assert!(!TrellisComputeProfile::WgpuFastMixedF16.wgpu_sparse_module_attention_f16());
        assert!(!TrellisComputeProfile::WgpuFastMixedF16.wgpu_sparse_self_attention_f16());
        assert!(!TrellisComputeProfile::WgpuFastMixedF16.wgpu_sparse_cross_attention_f16());
        assert!(TrellisComputeProfile::WgpuFastMixedF16.wgpu_slat_module_attention_f16());
        assert!(!TrellisComputeProfile::WgpuFastMixedF16.wgpu_linear_f16());
        assert!(!TrellisComputeProfile::WgpuFastMixedF16.wgpu_flow_torso_f16());
    }

    #[test]
    fn stock_bf16_emulated_profile_keeps_fast_f16_paths_disabled() {
        assert!(!TrellisComputeProfile::StockBf16Emulated.wgpu_module_attention_f16());
        assert!(!TrellisComputeProfile::StockBf16Emulated.wgpu_sparse_module_attention_f16());
        assert!(!TrellisComputeProfile::StockBf16Emulated.wgpu_slat_module_attention_f16());
        assert!(!TrellisComputeProfile::StockBf16Emulated.wgpu_linear_f16());
        assert!(!TrellisComputeProfile::StockBf16Emulated.wgpu_flow_torso_f16());
        assert!(!TrellisComputeProfile::StockBf16Emulated.wgpu_decoder_conv_f16());
        assert!(TrellisComputeProfile::StockBf16Emulated.wgpu_sparse_stock_bf16_emulation());
    }

    #[test]
    fn reference_profile_uses_upstream_fp16_decoder() {
        assert!(!TrellisComputeProfile::ReferenceF32.wgpu_module_attention_f16());
        assert!(!TrellisComputeProfile::ReferenceF32.wgpu_linear_f16());
        assert!(TrellisComputeProfile::ReferenceF32.wgpu_decoder_conv_f16());
    }

    #[test]
    fn diagnostic_sparse_f16_profiles_are_split() {
        assert!(TrellisComputeProfile::WgpuFastSparseSelfF16.wgpu_sparse_self_attention_f16());
        assert!(!TrellisComputeProfile::WgpuFastSparseSelfF16.wgpu_sparse_cross_attention_f16());
        assert!(!TrellisComputeProfile::WgpuFastSparseCrossF16.wgpu_sparse_self_attention_f16());
        assert!(TrellisComputeProfile::WgpuFastSparseCrossF16.wgpu_sparse_cross_attention_f16());
    }

    #[test]
    fn diagnostic_f16_tail_profiles_keep_final_steps_f32() {
        assert_eq!(
            TrellisComputeProfile::WgpuFastF16Tail1F32.wgpu_sparse_final_f32_steps(),
            1
        );
        assert_eq!(
            TrellisComputeProfile::WgpuFastF16Tail2F32.wgpu_sparse_final_f32_steps(),
            2
        );
        assert_eq!(
            TrellisComputeProfile::WgpuFastF16Tail4F32.wgpu_sparse_final_f32_steps(),
            4
        );
        assert_eq!(
            TrellisComputeProfile::WgpuFastF16Tail6F32.wgpu_sparse_final_f32_steps(),
            6
        );
        assert!(TrellisComputeProfile::WgpuFastF16Tail4F32.wgpu_sparse_self_attention_f16());
        assert!(TrellisComputeProfile::WgpuFastF16Tail4F32.wgpu_sparse_cross_attention_f16());
    }

    #[test]
    fn compute_profile_classes_do_not_mark_fast_candidates_as_parity_reference() {
        assert!(TrellisComputeProfile::ReferenceF32.is_parity_reference());
        assert!(!TrellisComputeProfile::ReferenceF32.is_fast_candidate());
        assert!(!TrellisComputeProfile::ReferenceF32.requires_render_or_hook_parity_gate());

        assert_eq!(
            TrellisComputeProfile::FAST_WGPU_BENCHMARK_CANDIDATE,
            TrellisComputeProfile::WgpuFastF16Tail1F32
        );
        assert!(TrellisComputeProfile::WgpuFastF16Tail1F32.is_fast_candidate());
        assert!(!TrellisComputeProfile::WgpuFastF16Tail1F32.is_parity_reference());
        assert!(TrellisComputeProfile::WgpuFastF16Tail1F32.requires_render_or_hook_parity_gate());

        assert!(TrellisComputeProfile::WgpuFastF16.is_diagnostic());
        assert!(!TrellisComputeProfile::WgpuFastF16.is_fast_candidate());
        assert!(TrellisComputeProfile::WgpuFastF16.requires_render_or_hook_parity_gate());
    }

    #[test]
    fn default_compute_profile_is_reference_f32() {
        assert_eq!(
            TrellisComputeProfile::default(),
            TrellisComputeProfile::ReferenceF32
        );
    }
}

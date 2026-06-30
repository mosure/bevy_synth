use std::fmt;

use serde::{Deserialize, Serialize};

use crate::SegmentationModelKind;

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SegmentationPrecision {
    F32,
    #[default]
    F16,
    Bf16,
}

impl SegmentationPrecision {
    pub fn label(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
        }
    }
}

impl fmt::Display for SegmentationPrecision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum SegmentationQuantization {
    #[default]
    None,
    Q8,
    Q4,
}

impl SegmentationQuantization {
    pub fn label(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Q8 => "q8",
            Self::Q4 => "q4",
        }
    }

    pub fn file_suffix(self) -> Option<&'static str> {
        match self {
            Self::None => None,
            Self::Q8 => Some("q8"),
            Self::Q4 => Some("q4"),
        }
    }
}

impl fmt::Display for SegmentationQuantization {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum SegmentationModelComponent {
    ImageEncoder,
    PromptEncoder,
    MaskDecoder,
    MemoryEncoder,
    MemoryAttention,
}

impl SegmentationModelComponent {
    pub fn label(self) -> &'static str {
        match self {
            Self::ImageEncoder => "image_encoder",
            Self::PromptEncoder => "prompt_encoder",
            Self::MaskDecoder => "mask_decoder",
            Self::MemoryEncoder => "memory_encoder",
            Self::MemoryAttention => "memory_attention",
        }
    }
}

pub fn required_components(model: SegmentationModelKind) -> &'static [SegmentationModelComponent] {
    match model {
        SegmentationModelKind::BboxPrompt => &[],
        // Scene composition needs image-prompted still-image masks first. SAM2
        // video memory components are tracked as optional artifacts by import
        // reports, not required for the initial image-mask runtime path.
        SegmentationModelKind::Sam2 | SegmentationModelKind::Sam3 => &[
            SegmentationModelComponent::ImageEncoder,
            SegmentationModelComponent::PromptEncoder,
            SegmentationModelComponent::MaskDecoder,
        ],
    }
}

pub fn optional_components(model: SegmentationModelKind) -> &'static [SegmentationModelComponent] {
    match model {
        SegmentationModelKind::BboxPrompt | SegmentationModelKind::Sam3 => &[],
        SegmentationModelKind::Sam2 => &[
            SegmentationModelComponent::MemoryEncoder,
            SegmentationModelComponent::MemoryAttention,
        ],
    }
}

pub fn component_burnpack_file_name(
    component: SegmentationModelComponent,
    precision: SegmentationPrecision,
    quantization: SegmentationQuantization,
) -> String {
    match quantization.file_suffix() {
        Some(quantization) => format!("{}_{}_{}.bpk", component.label(), precision, quantization),
        None => format!("{}_{}.bpk", component.label(), precision),
    }
}

use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use image::{DynamicImage, Rgba, RgbaImage};
use serde::{Deserialize, Serialize};

pub mod assets;
pub mod cdn;
pub mod config;
pub mod image_encoder;
#[cfg(feature = "import")]
pub mod import;
pub mod mask_decoder;
pub mod parity;
pub mod prompt_encoder;
#[cfg(feature = "backend_wgpu")]
mod sam2_wgpu;
pub mod tensor_io;

pub use assets::{
    SegmentationAssetReport, SegmentationBurnpackStatus, SegmentationWeightFileStatus,
    burnpack_parts_manifest_path, inspect_model_assets,
};
pub use cdn::{
    SegmentationFilePartEntry, SegmentationFilePartsManifest, SegmentationFilePartsReport,
    assemble_file_parts, component_safetensors_file_name, component_safetensors_rel_path,
    file_parts_manifest_path, read_file_parts_manifest, resolve_file_part_entry_path,
    segmentation_cdn_root_prefix, segmentation_cdn_root_url, write_file_parts_for_cdn,
};
pub use config::{
    SegmentationModelComponent, SegmentationPrecision, SegmentationQuantization,
    component_burnpack_file_name, optional_components, required_components,
};
pub use image_encoder::{SamImageEncoderConfig, SamImageEncoderVariant, SamImageEncoderWeights};
pub use mask_decoder::{SamMaskDecoderConfig, SamMaskDecoderWeights};
pub use parity::{
    SegmentationMaskParityReport, SegmentationParityConfig, SegmentationParityFixture,
    SegmentationParitySummary, compare_mask, compare_mask_sets, compare_parity_fixture,
    read_parity_fixture, write_parity_summary,
};
pub use prompt_encoder::{SamPromptEncoderConfig, SamPromptEncoderWeights, SamPromptInput};

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum SegmentationModelKind {
    BboxPrompt,
    Sam2,
    Sam3,
}

impl SegmentationModelKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::BboxPrompt => "bbox-prompt",
            Self::Sam2 => "sam2",
            Self::Sam3 => "sam3",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum SegmentationRuntimeBackend {
    BboxPrompt,
    BurnNative,
    #[cfg(feature = "python-reference")]
    PythonReference,
}

impl SegmentationRuntimeBackend {
    pub fn label(self) -> &'static str {
        match self {
            Self::BboxPrompt => "bbox-prompt",
            Self::BurnNative => "burn-native",
            #[cfg(feature = "python-reference")]
            Self::PythonReference => "python-reference",
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SegmentationRuntimeConfig {
    pub model: SegmentationModelKind,
    pub backend: SegmentationRuntimeBackend,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_root: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_dir: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cdn_base_url: Option<String>,
    #[serde(default)]
    pub precision: SegmentationPrecision,
    #[serde(default)]
    pub quantization: SegmentationQuantization,
    #[serde(default)]
    pub allow_download: bool,
    #[serde(default)]
    pub require_gpu: bool,
    #[serde(default)]
    pub profile_stages: bool,
}

impl Default for SegmentationRuntimeConfig {
    fn default() -> Self {
        Self {
            model: SegmentationModelKind::BboxPrompt,
            backend: SegmentationRuntimeBackend::BboxPrompt,
            model_root: None,
            cache_dir: None,
            cdn_base_url: None,
            precision: SegmentationPrecision::default(),
            quantization: SegmentationQuantization::default(),
            allow_download: false,
            require_gpu: true,
            profile_stages: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct SegmentationStageTimings {
    pub preprocess_ms: f64,
    pub encode_ms: f64,
    pub prompt_ms: f64,
    pub decode_ms: f64,
    pub postprocess_ms: f64,
    pub total_ms: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct SegmentationPrompt {
    pub object_id: String,
    pub label: String,
    pub bbox: [f32; 4],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub point: Option<[f32; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_query: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct SegmentationMask {
    pub object_id: String,
    pub label: String,
    pub bbox: [f32; 4],
    pub score: f32,
    pub width: u32,
    pub height: u32,
    pub area_px: u32,
    pub mask_rle: Vec<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mask_png_path: Option<String>,
    pub source_prompt: SegmentationPrompt,
    pub provider: String,
    pub model: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SegmentationError {
    Unsupported(String),
    InvalidPrompt(String),
    Image(String),
    Io(String),
}

impl fmt::Display for SegmentationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported(message) => write!(f, "unsupported segmentation mode: {message}"),
            Self::InvalidPrompt(message) => write!(f, "invalid segmentation prompt: {message}"),
            Self::Image(message) => write!(f, "segmentation image error: {message}"),
            Self::Io(message) => write!(f, "segmentation io error: {message}"),
        }
    }
}

impl std::error::Error for SegmentationError {}

pub type SegmentationResult<T> = Result<T, SegmentationError>;

#[derive(Debug)]
pub struct SegmentationRuntime {
    config: SegmentationRuntimeConfig,
    inner: SegmentationRuntimeInner,
}

#[derive(Debug)]
enum SegmentationRuntimeInner {
    BboxPrompt,
    #[cfg(feature = "backend_wgpu")]
    Sam2Wgpu(Box<sam2_wgpu::Sam2WgpuRuntime>),
}

impl SegmentationRuntime {
    pub fn new(config: SegmentationRuntimeConfig) -> SegmentationResult<Self> {
        match (config.model, config.backend) {
            (SegmentationModelKind::BboxPrompt, SegmentationRuntimeBackend::BboxPrompt) => {
                Ok(Self {
                    config,
                    inner: SegmentationRuntimeInner::BboxPrompt,
                })
            }
            #[cfg(feature = "backend_wgpu")]
            (SegmentationModelKind::Sam2, SegmentationRuntimeBackend::BurnNative) => {
                let runtime = sam2_wgpu::Sam2WgpuRuntime::new(&config)?;
                Ok(Self {
                    config,
                    inner: SegmentationRuntimeInner::Sam2Wgpu(Box::new(runtime)),
                })
            }
            (SegmentationModelKind::Sam2 | SegmentationModelKind::Sam3, _) => {
                let asset_status = config
                    .model_root
                    .as_ref()
                    .and_then(|model_root| {
                        inspect_model_assets(
                            config.model,
                            model_root,
                            config.precision,
                            config.quantization,
                        )
                        .ok()
                    })
                    .map(|report| format!("; missing native artifacts: {:?}", report.missing_files))
                    .unwrap_or_default();
                Err(SegmentationError::Unsupported(format!(
                    "{} with {:?} is not implemented in this build{asset_status}",
                    config.model.label(),
                    config.backend
                )))
            }
            (_, SegmentationRuntimeBackend::BurnNative) => Err(SegmentationError::Unsupported(
                "Burn-native segmentation is not implemented for this model".to_string(),
            )),
            #[cfg(feature = "python-reference")]
            (_, SegmentationRuntimeBackend::PythonReference) => Err(SegmentationError::Unsupported(
                "python-reference segmentation is reserved for fixture capture and not wired yet"
                    .to_string(),
            )),
        }
    }

    pub fn config(&self) -> &SegmentationRuntimeConfig {
        &self.config
    }

    pub fn last_stage_timings(&self) -> Option<SegmentationStageTimings> {
        match &self.inner {
            SegmentationRuntimeInner::BboxPrompt => None,
            #[cfg(feature = "backend_wgpu")]
            SegmentationRuntimeInner::Sam2Wgpu(runtime) => runtime.last_stage_timings(),
        }
    }

    pub fn sam_image_encoder_variant(&self) -> Option<SamImageEncoderVariant> {
        match &self.inner {
            SegmentationRuntimeInner::BboxPrompt => None,
            #[cfg(feature = "backend_wgpu")]
            SegmentationRuntimeInner::Sam2Wgpu(runtime) => Some(runtime.variant()),
        }
    }

    pub fn segment(
        &mut self,
        image: &DynamicImage,
        prompts: &[SegmentationPrompt],
    ) -> SegmentationResult<Vec<SegmentationMask>> {
        match &mut self.inner {
            SegmentationRuntimeInner::BboxPrompt => {
                let width = image.width();
                let height = image.height();
                prompts
                    .iter()
                    .map(|prompt| bbox_prompt_mask(width, height, prompt))
                    .collect()
            }
            #[cfg(feature = "backend_wgpu")]
            SegmentationRuntimeInner::Sam2Wgpu(runtime) => runtime.segment(image, prompts),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BinaryMask {
    width: u32,
    height: u32,
    data: Vec<u8>,
}

impl BinaryMask {
    pub fn new(width: u32, height: u32, data: Vec<u8>) -> SegmentationResult<Self> {
        let expected = width as usize * height as usize;
        if data.len() != expected {
            return Err(SegmentationError::Image(format!(
                "mask data length mismatch: expected {expected}, got {}",
                data.len()
            )));
        }
        Ok(Self {
            width,
            height,
            data: data.into_iter().map(|value| u8::from(value != 0)).collect(),
        })
    }

    pub fn from_normalized_bbox(
        width: u32,
        height: u32,
        bbox: [f32; 4],
    ) -> SegmentationResult<Self> {
        validate_bbox(bbox)?;
        if width == 0 || height == 0 {
            return Err(SegmentationError::Image(
                "cannot create mask for zero-sized image".to_string(),
            ));
        }
        let mut data = vec![0_u8; width as usize * height as usize];
        let x0 = normalized_to_start_px(bbox[0], width);
        let y0 = normalized_to_start_px(bbox[1], height);
        let x1 = normalized_to_end_px(bbox[2], width).max(x0.saturating_add(1));
        let y1 = normalized_to_end_px(bbox[3], height).max(y0.saturating_add(1));
        for y in y0.min(height)..y1.min(height) {
            let row = y as usize * width as usize;
            for x in x0.min(width)..x1.min(width) {
                data[row + x as usize] = 1;
            }
        }
        Self::new(width, height, data)
    }

    pub fn decode_rle(width: u32, height: u32, rle: &[u32]) -> SegmentationResult<Self> {
        let total = width as usize * height as usize;
        let mut data = Vec::with_capacity(total);
        let mut value = 0_u8;
        for &count in rle {
            let new_len = data.len().saturating_add(count as usize);
            if new_len > total {
                return Err(SegmentationError::Image(format!(
                    "mask rle decodes past image size: {new_len} > {total}"
                )));
            }
            data.resize(new_len, value);
            value = u8::from(value == 0);
        }
        if data.len() != total {
            return Err(SegmentationError::Image(format!(
                "mask rle decoded {} pixels, expected {total}",
                data.len()
            )));
        }
        Self::new(width, height, data)
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn data(&self) -> &[u8] {
        &self.data
    }

    pub fn area_px(&self) -> u32 {
        self.data.iter().map(|&value| u32::from(value != 0)).sum()
    }

    pub fn encode_rle(&self) -> Vec<u32> {
        let mut runs = Vec::new();
        let mut expected = 0_u8;
        let mut count = 0_u32;
        for &value in &self.data {
            let value = u8::from(value != 0);
            if value == expected {
                count = count.saturating_add(1);
            } else {
                runs.push(count);
                expected = value;
                count = 1;
            }
        }
        runs.push(count);
        runs
    }

    pub fn bbox_normalized(&self) -> Option<[f32; 4]> {
        let mut min_x = self.width;
        let mut min_y = self.height;
        let mut max_x = 0_u32;
        let mut max_y = 0_u32;
        let mut found = false;
        for y in 0..self.height {
            for x in 0..self.width {
                if self.data[y as usize * self.width as usize + x as usize] != 0 {
                    found = true;
                    min_x = min_x.min(x);
                    min_y = min_y.min(y);
                    max_x = max_x.max(x.saturating_add(1));
                    max_y = max_y.max(y.saturating_add(1));
                }
            }
        }
        found.then_some([
            min_x as f32 / self.width.max(1) as f32,
            min_y as f32 / self.height.max(1) as f32,
            max_x as f32 / self.width.max(1) as f32,
            max_y as f32 / self.height.max(1) as f32,
        ])
    }

    pub fn iou(&self, other: &Self) -> SegmentationResult<f32> {
        if self.width != other.width || self.height != other.height {
            return Err(SegmentationError::Image(format!(
                "mask dimensions differ: {}x{} vs {}x{}",
                self.width, self.height, other.width, other.height
            )));
        }
        let mut intersection = 0_u32;
        let mut union = 0_u32;
        for (&left, &right) in self.data.iter().zip(other.data.iter()) {
            let left = left != 0;
            let right = right != 0;
            intersection = intersection.saturating_add(u32::from(left && right));
            union = union.saturating_add(u32::from(left || right));
        }
        Ok(if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        })
    }
}

pub fn write_mask_png(mask: &BinaryMask, path: &Path) -> SegmentationResult<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| SegmentationError::Io(format!("create {}: {err}", parent.display())))?;
    }
    let mut image = RgbaImage::new(mask.width, mask.height);
    for y in 0..mask.height {
        for x in 0..mask.width {
            let value = mask.data[y as usize * mask.width as usize + x as usize];
            let alpha = if value == 0 { 0 } else { 255 };
            image.put_pixel(x, y, Rgba([255, 255, 255, alpha]));
        }
    }
    image
        .save(path)
        .map_err(|err| SegmentationError::Io(format!("save {}: {err}", path.display())))
}

pub fn write_mask_overlay(
    image: &DynamicImage,
    masks: &[SegmentationMask],
    path: &Path,
) -> SegmentationResult<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| SegmentationError::Io(format!("create {}: {err}", parent.display())))?;
    }
    let mut overlay = image.to_rgba8();
    for (index, mask) in masks.iter().enumerate() {
        let decoded = BinaryMask::decode_rle(mask.width, mask.height, &mask.mask_rle)?;
        let color = overlay_color(index);
        for y in 0..decoded.height {
            for x in 0..decoded.width {
                if decoded.data[y as usize * decoded.width as usize + x as usize] != 0 {
                    let pixel = overlay.get_pixel_mut(x, y);
                    pixel.0[0] = ((u16::from(pixel.0[0]) + u16::from(color[0])) / 2) as u8;
                    pixel.0[1] = ((u16::from(pixel.0[1]) + u16::from(color[1])) / 2) as u8;
                    pixel.0[2] = ((u16::from(pixel.0[2]) + u16::from(color[2])) / 2) as u8;
                    pixel.0[3] = 255;
                }
            }
        }
    }
    overlay
        .save(path)
        .map_err(|err| SegmentationError::Io(format!("save {}: {err}", path.display())))
}

fn bbox_prompt_mask(
    width: u32,
    height: u32,
    prompt: &SegmentationPrompt,
) -> SegmentationResult<SegmentationMask> {
    let binary = BinaryMask::from_normalized_bbox(width, height, prompt.bbox)?;
    let bbox = binary.bbox_normalized().unwrap_or(prompt.bbox);
    Ok(SegmentationMask {
        object_id: prompt.object_id.clone(),
        label: prompt.label.clone(),
        bbox,
        score: 1.0,
        width,
        height,
        area_px: binary.area_px(),
        mask_rle: binary.encode_rle(),
        mask_png_path: None,
        source_prompt: prompt.clone(),
        provider: "bbox-prompt".to_string(),
        model: SegmentationModelKind::BboxPrompt.label().to_string(),
    })
}

fn validate_bbox(bbox: [f32; 4]) -> SegmentationResult<()> {
    if bbox.iter().any(|value| !value.is_finite()) {
        return Err(SegmentationError::InvalidPrompt(format!(
            "bbox contains non-finite value: {bbox:?}"
        )));
    }
    if bbox[0] >= bbox[2] || bbox[1] >= bbox[3] {
        return Err(SegmentationError::InvalidPrompt(format!(
            "bbox must be [x0,y0,x1,y1], got {bbox:?}"
        )));
    }
    Ok(())
}

fn normalized_to_start_px(value: f32, extent: u32) -> u32 {
    (value.clamp(0.0, 1.0) * extent as f32).floor() as u32
}

fn normalized_to_end_px(value: f32, extent: u32) -> u32 {
    ((value.clamp(0.0, 1.0) * extent as f32) - 1.0e-5)
        .ceil()
        .max(0.0) as u32
}

fn overlay_color(index: usize) -> [u8; 3] {
    const COLORS: [[u8; 3]; 8] = [
        [238, 90, 82],
        [80, 175, 120],
        [70, 130, 230],
        [238, 190, 80],
        [185, 90, 210],
        [70, 190, 190],
        [240, 130, 40],
        [145, 145, 145],
    ];
    COLORS[index % COLORS.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bbox_prompt_runtime_returns_mask_for_prompt() {
        let image = DynamicImage::new_rgba8(100, 50);
        let mut runtime = SegmentationRuntime::new(SegmentationRuntimeConfig::default()).unwrap();
        let masks = runtime
            .segment(
                &image,
                &[SegmentationPrompt {
                    object_id: "chair_1".to_string(),
                    label: "chair".to_string(),
                    bbox: [0.10, 0.20, 0.30, 0.60],
                    point: Some([0.20, 0.55]),
                    source_query: Some("chair".to_string()),
                }],
            )
            .unwrap();

        assert_eq!(masks.len(), 1);
        assert_eq!(masks[0].object_id, "chair_1");
        assert_eq!(masks[0].area_px, 20 * 20);
        assert_eq!(masks[0].bbox, [0.10, 0.20, 0.30, 0.60]);
    }

    #[test]
    fn mask_rle_round_trips() {
        let mask = BinaryMask::from_normalized_bbox(10, 10, [0.2, 0.2, 0.5, 0.6]).unwrap();
        let rle = mask.encode_rle();
        let decoded = BinaryMask::decode_rle(mask.width(), mask.height(), &rle).unwrap();
        assert_eq!(mask, decoded);
    }

    #[test]
    fn mask_iou_scores_overlap() {
        let left = BinaryMask::from_normalized_bbox(10, 10, [0.0, 0.0, 0.5, 1.0]).unwrap();
        let right = BinaryMask::from_normalized_bbox(10, 10, [0.25, 0.0, 0.75, 1.0]).unwrap();
        let iou = left.iou(&right).unwrap();
        assert!((iou - (30.0 / 80.0)).abs() < 1.0e-5);
    }

    #[test]
    fn sam_models_fail_fast_until_wired() {
        let err = SegmentationRuntime::new(SegmentationRuntimeConfig {
            model: SegmentationModelKind::Sam2,
            backend: SegmentationRuntimeBackend::BurnNative,
            model_root: Some(PathBuf::from("assets/models/sam2")),
            ..SegmentationRuntimeConfig::default()
        })
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("sam2 with BurnNative is not implemented")
                || err
                    .to_string()
                    .contains("missing SAM2 component safetensors")
        );
    }
}

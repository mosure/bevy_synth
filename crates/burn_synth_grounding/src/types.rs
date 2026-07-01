use std::path::PathBuf;

use burn_depth::{CameraIntrinsics, DepthPipeline, DepthPrecision};
use burn_locate_anything::import::LocateAnythingPrecision;
use burn_locate_anything::{
    DecodeMode, LOCATE_ANYTHING_SAFE_IN_TOKEN_LIMIT, LocateAnythingRuntime,
    LocateAnythingRuntimeBackend, LocateAnythingRuntimeConfig,
};
use burn_segmentation::{
    SegmentationModelKind, SegmentationPrecision, SegmentationQuantization, SegmentationRuntime,
    SegmentationRuntimeBackend, SegmentationRuntimeConfig,
};
use serde::{Deserialize, Serialize};

use crate::locate::default_locate_anything_allowed_categories;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GroundingDepthPrecision {
    F32,
    F16,
}

impl GroundingDepthPrecision {
    pub fn label(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
        }
    }
}

impl From<GroundingDepthPrecision> for DepthPrecision {
    fn from(value: GroundingDepthPrecision) -> Self {
        match value {
            GroundingDepthPrecision::F32 => DepthPrecision::F32,
            GroundingDepthPrecision::F16 => DepthPrecision::F16,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DepthProGroundingConfig {
    pub cache_dir: Option<PathBuf>,
    pub precision: GroundingDepthPrecision,
    pub allow_download: bool,
    pub require_gpu: bool,
}

impl Default for DepthProGroundingConfig {
    fn default() -> Self {
        Self {
            cache_dir: None,
            precision: GroundingDepthPrecision::F16,
            allow_download: true,
            require_gpu: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LocateAnythingGroundingConfig {
    pub model_root: PathBuf,
    pub cache_dir: Option<PathBuf>,
    pub cdn_base_url: Option<String>,
    pub allow_download: bool,
    pub allowed_categories: Vec<String>,
    pub precision: LocateAnythingPrecision,
    pub in_token_limit: usize,
    pub decode_mode: DecodeMode,
    pub max_new_tokens: usize,
    pub repetition_penalty: f32,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
}

impl Default for LocateAnythingGroundingConfig {
    fn default() -> Self {
        let runtime = LocateAnythingRuntimeConfig::default();
        Self {
            model_root: PathBuf::from("assets/models/LocateAnything-3B"),
            cache_dir: None,
            cdn_base_url: None,
            allow_download: false,
            allowed_categories: default_locate_anything_allowed_categories(),
            precision: LocateAnythingPrecision::default(),
            in_token_limit: LOCATE_ANYTHING_SAFE_IN_TOKEN_LIMIT as usize,
            decode_mode: DecodeMode::Hybrid,
            max_new_tokens: runtime.max_new_tokens,
            repetition_penalty: runtime.repetition_penalty,
            top_p: runtime.top_p,
            top_k: runtime.top_k,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SegmentationGroundingConfig {
    pub model: SegmentationModelKind,
    pub backend: SegmentationRuntimeBackend,
    pub model_root: Option<PathBuf>,
    pub cache_dir: Option<PathBuf>,
    pub cdn_base_url: Option<String>,
    pub precision: SegmentationPrecision,
    pub quantization: SegmentationQuantization,
    pub allow_download: bool,
    pub require_gpu: bool,
}

impl Default for SegmentationGroundingConfig {
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
        }
    }
}

impl SegmentationGroundingConfig {
    pub(crate) fn runtime_config(&self) -> SegmentationRuntimeConfig {
        SegmentationRuntimeConfig {
            model: self.model,
            backend: self.backend,
            model_root: self.model_root.clone(),
            cache_dir: self.cache_dir.clone(),
            cdn_base_url: self.cdn_base_url.clone(),
            precision: self.precision,
            quantization: self.quantization,
            allow_download: self.allow_download,
            require_gpu: self.require_gpu,
            profile_stages: true,
        }
    }
}

impl LocateAnythingGroundingConfig {
    pub(crate) fn runtime_config(&self) -> LocateAnythingRuntimeConfig {
        LocateAnythingRuntimeConfig {
            model_root: self.model_root.clone(),
            cache_dir: self.cache_dir.clone(),
            cdn_base_url: self.cdn_base_url.clone(),
            allow_download: self.allow_download,
            precision: self.precision,
            backend: LocateAnythingRuntimeBackend::BurnNative,
            allow_experimental_native_detect: true,
            decode_mode: self.decode_mode,
            max_new_tokens: self.max_new_tokens,
            in_token_limit: self.in_token_limit.max(1) as u32,
            repetition_penalty: self.repetition_penalty,
            top_p: self.top_p,
            top_k: self.top_k,
            ..LocateAnythingRuntimeConfig::default()
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LocateAnythingBurnNativeCacheKey {
    model_root: PathBuf,
    cache_dir: Option<PathBuf>,
    cdn_base_url: Option<String>,
    allow_download: bool,
    precision: LocateAnythingPrecision,
    decode_mode: DecodeMode,
    max_new_tokens: usize,
    in_token_limit: u32,
    repetition_penalty_bits: u32,
    top_p_bits: Option<u32>,
    top_k: Option<usize>,
}

impl LocateAnythingBurnNativeCacheKey {
    pub fn from_config(config: &LocateAnythingRuntimeConfig) -> Self {
        Self {
            model_root: config.model_root.clone(),
            cache_dir: config.cache_dir.clone(),
            cdn_base_url: config.cdn_base_url.clone(),
            allow_download: config.allow_download,
            precision: config.precision,
            decode_mode: config.decode_mode,
            max_new_tokens: config.max_new_tokens,
            in_token_limit: config.in_token_limit,
            repetition_penalty_bits: config.repetition_penalty.to_bits(),
            top_p_bits: config.top_p.map(f32::to_bits),
            top_k: config.top_k,
        }
    }
}

#[derive(Default)]
pub struct SceneGroundingRuntime {
    pub(crate) depth_pro_runtime: Option<CachedDepthProRuntime>,
    pub(crate) locate_anything_burn_native_runtime: Option<CachedLocateAnythingRuntime>,
    pub(crate) segmentation_runtime: Option<CachedSegmentationRuntime>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DepthProRuntimeCacheKey {
    cache_dir: Option<PathBuf>,
    precision: GroundingDepthPrecision,
}

impl DepthProRuntimeCacheKey {
    pub fn from_config(config: &DepthProGroundingConfig) -> Self {
        Self {
            cache_dir: config.cache_dir.clone(),
            precision: config.precision,
        }
    }
}

pub(crate) struct CachedDepthProRuntime {
    pub(crate) key: DepthProRuntimeCacheKey,
    pub(crate) pipeline: DepthPipeline<burn_depth::InferenceBackend>,
}

pub(crate) struct CachedLocateAnythingRuntime {
    pub(crate) key: LocateAnythingBurnNativeCacheKey,
    pub(crate) runtime: LocateAnythingRuntime,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SegmentationRuntimeCacheKey {
    model: SegmentationModelKind,
    backend: SegmentationRuntimeBackend,
    model_root: Option<PathBuf>,
    cache_dir: Option<PathBuf>,
    cdn_base_url: Option<String>,
    precision: SegmentationPrecision,
    quantization: SegmentationQuantization,
}

impl SegmentationRuntimeCacheKey {
    pub fn from_config(config: &SegmentationRuntimeConfig) -> Self {
        Self {
            model: config.model,
            backend: config.backend,
            model_root: config.model_root.clone(),
            cache_dir: config.cache_dir.clone(),
            cdn_base_url: config.cdn_base_url.clone(),
            precision: config.precision,
            quantization: config.quantization,
        }
    }
}

pub(crate) struct CachedSegmentationRuntime {
    pub(crate) key: SegmentationRuntimeCacheKey,
    pub(crate) runtime: SegmentationRuntime,
}

#[derive(Clone, Debug, Serialize)]
pub struct DepthProGroundingReport {
    pub artifact_path: PathBuf,
    pub depth_map_path: PathBuf,
    pub depth_map_metadata_path: PathBuf,
    pub load_ms: f64,
    pub infer_ms: f64,
    pub runtime_cache_hit: bool,
    pub summary: SceneDepthAnnotationSummary,
}

#[derive(Clone, Debug, Serialize)]
pub struct LocateAnythingGroundingReport {
    pub artifact_dir: PathBuf,
    pub detections_path: PathBuf,
    pub overlay_path: PathBuf,
    pub metadata_path: PathBuf,
    pub elapsed_ms: f64,
    pub runtime_cache_hit: bool,
    pub detection_count: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct SegmentationGroundingReport {
    pub artifact_dir: PathBuf,
    pub masks_path: PathBuf,
    pub overlay_path: PathBuf,
    pub elapsed_ms: f64,
    pub runtime_cache_hit: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_variant: Option<String>,
    pub mask_count: usize,
    pub stage_timings: Option<burn_segmentation::SegmentationStageTimings>,
}

#[derive(Clone, Debug)]
pub struct SceneDepthMapEvidence {
    pub depth_m: Vec<f32>,
    pub width: u32,
    pub height: u32,
    pub intrinsics: CameraIntrinsics,
    pub focal_length_px: Option<f32>,
    pub fov_x_degrees: Option<f32>,
    pub vertical_fov_degrees: Option<f32>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct SceneDepthAnnotationSummary {
    pub provider: String,
    pub annotated_objects: usize,
    pub total_objects: usize,
    pub depth_map_size: [u32; 2],
    pub focal_length_px: Option<f32>,
    pub fov_x_degrees: Option<f32>,
    pub vertical_fov_degrees: Option<f32>,
    /// Accepted inlier samples used by the final floor fit.
    pub floor_sample_count: usize,
    pub floor_candidate_sample_count: usize,
    pub floor_inlier_count: usize,
    pub floor_rejected_sample_count: usize,
    pub floor_inlier_ratio: Option<f32>,
    pub floor_estimation_method: Option<String>,
    pub floor_residual_m: Option<f32>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct SceneFarFieldFilterSummary {
    pub enabled: bool,
    pub threshold_m: Option<f32>,
    pub median_object_depth_m: Option<f32>,
    pub lower_quartile_object_depth_m: Option<f32>,
    pub removed_detections: usize,
    pub removed_objects: usize,
    pub removed_detection_labels: Vec<String>,
    pub removed_object_ids: Vec<String>,
}

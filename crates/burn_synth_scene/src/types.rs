use std::path::PathBuf;

use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum SceneQualityProfile {
    Draft,
    Quality,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum SceneScalePolicy {
    #[default]
    AssetPreserving,
    BoundedAnisotropic,
    FreeAnisotropic,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum SceneRotationFitMode {
    Off,
    DepthMaskRansac,
    GptRefine,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum SceneObjectPoseRefinementMode {
    Off,
    Geometry,
    GatedGpt,
    AlwaysGpt,
}

impl SceneObjectPoseRefinementMode {
    pub fn geometry_enabled(self) -> bool {
        !matches!(self, Self::Off)
    }

    pub fn gpt_allowed(self) -> bool {
        matches!(self, Self::GatedGpt | Self::AlwaysGpt)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum SceneFinalYawRefinementMode {
    Off,
    GatedGpt,
    AlwaysGpt,
}

impl SceneFinalYawRefinementMode {
    pub fn enabled(self) -> bool {
        !matches!(self, Self::Off)
    }

    pub fn requires_selection(self) -> bool {
        matches!(self, Self::GatedGpt | Self::AlwaysGpt)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum SceneObjectPoseRefinementSet {
    Tables,
    LargeSeating,
    TablesAndLargeSeating,
    AllFurniture,
}

impl SceneObjectPoseRefinementSet {
    pub fn label(self) -> &'static str {
        match self {
            Self::Tables => "tables",
            Self::LargeSeating => "large-seating",
            Self::TablesAndLargeSeating => "tables-and-large-seating",
            Self::AllFurniture => "all-furniture",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum ScenePoseFitMode {
    ProjectedAabb,
    RenderedSilhouette,
}

impl SceneScalePolicy {
    pub fn max_xz_anisotropy(self) -> Option<f32> {
        match self {
            Self::AssetPreserving => Some(1.0),
            Self::BoundedAnisotropic => Some(1.25),
            Self::FreeAnisotropic => None,
        }
    }

    pub fn allows_axis_feedback(self) -> bool {
        // Screen-space bbox axes do not map reliably to local X/Z after yaw and camera projection.
        // Keep feedback scale uniform; initial layout remains the owner of explicit anisotropy.
        false
    }

    pub fn apply_to_scale(self, scale: [f32; 3]) -> [f32; 3] {
        let Some(max_ratio) = self.max_xz_anisotropy() else {
            return [
                sanitize_scale_component(scale[0]),
                sanitize_scale_component(scale[1]),
                sanitize_scale_component(scale[2]),
            ];
        };
        if max_ratio <= 1.0 + f32::EPSILON {
            let uniform = robust_uniform_scale(scale);
            return [uniform, uniform, uniform];
        }

        let x = sanitize_scale_component(scale[0]);
        let y = sanitize_scale_component(scale[1]);
        let z = sanitize_scale_component(scale[2]);
        let ratio = (x.max(z) / x.min(z).max(1.0e-5)).max(1.0);
        if ratio <= max_ratio {
            return [x, y, z];
        }
        let area_scale = (x * z).sqrt().clamp(0.05, 20.0);
        let root_ratio = max_ratio.sqrt();
        let (next_x, next_z) = if x >= z {
            (area_scale * root_ratio, area_scale / root_ratio)
        } else {
            (area_scale / root_ratio, area_scale * root_ratio)
        };
        [
            next_x.clamp(0.05, 20.0),
            area_scale.clamp(0.05, 20.0),
            next_z.clamp(0.05, 20.0),
        ]
    }
}

fn sanitize_scale_component(value: f32) -> f32 {
    value.abs().clamp(0.05, 20.0)
}

fn robust_uniform_scale(scale: [f32; 3]) -> f32 {
    let mut values = [
        sanitize_scale_component(scale[0]),
        sanitize_scale_component(scale[1]),
        sanitize_scale_component(scale[2]),
    ];
    values.sort_by(f32::total_cmp);
    values[1]
}

pub const DEFAULT_SCENE_RECONSTRUCTION_IMAGE_SCORE: f32 = 0.45;

fn default_instance_count() -> usize {
    1
}

#[derive(Clone, Debug)]
pub struct SceneBuildConfig {
    pub source_scene_path: PathBuf,
    pub object_reference_image_path: PathBuf,
    pub output_dir: PathBuf,
    pub candidate_count: usize,
    pub quality_profile: SceneQualityProfile,
    pub reasoning_model: String,
    pub image_model: String,
    pub allow_catalog_reuse: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct SceneObjectManifest {
    pub source_scene_path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scene_calibration: Option<SceneCalibration>,
    pub objects: Vec<SceneObjectSpec>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Detection {
    pub label: String,
    pub bbox: [f32; 4],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub point: Option<[f32; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    pub source_query: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct SceneGroundingEvidence {
    pub source_image_path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub depth: Option<DepthEvidenceRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub segmentation: Option<SegmentationEvidenceRef>,
    #[serde(default)]
    pub detections: Vec<Detection>,
    pub camera: EstimatedCamera,
    pub floor: EstimatedFloorPlane,
    #[serde(default)]
    pub objects: Vec<ObjectGroundingEvidence>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GroundingVerificationStatus {
    Verified,
    Fallback,
    Invalid,
    Absent,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GptDelegationRole {
    None,
    Hypothesis,
    ImageSynthesis,
    BoundedCandidateSelection,
    DeveloperDiagnosis,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct GroundingContractEntry {
    pub name: String,
    pub producer: String,
    pub pipeline_stage: String,
    pub status: GroundingVerificationStatus,
    pub gpt_role: GptDelegationRole,
    #[serde(default)]
    pub consumers: Vec<String>,
    #[serde(default)]
    pub metrics: Value,
    #[serde(default)]
    pub provenance: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct GroundingContractReport {
    pub schema_version: u32,
    pub source_scene_path: String,
    pub composition_mode: String,
    pub entries: Vec<GroundingContractEntry>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct SceneDecisionLogEntry {
    pub decision: String,
    pub pipeline_stage: String,
    pub source_of_truth: String,
    pub status: GroundingVerificationStatus,
    pub gpt_role: GptDelegationRole,
    #[serde(default)]
    pub metrics: Value,
    #[serde(default)]
    pub alternatives: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct SceneDecisionLog {
    pub schema_version: u32,
    pub source_scene_path: String,
    pub entries: Vec<SceneDecisionLogEntry>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct SegmentationEvidenceRef {
    pub provider: String,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub overlay_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mask_count: Option<usize>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct DepthEvidenceRef {
    pub provider: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub precision: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intrinsics: Option<SceneCameraIntrinsics>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub focal_length_px: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vertical_fov_degrees: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_size: Option<[u32; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub depth_map_size: Option<[u32; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub floor_sample_count: Option<usize>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
pub struct SceneCameraIntrinsics {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub width: u32,
    pub height: u32,
    pub fov_x_degrees: f32,
    pub fov_y_degrees: f32,
}

impl SceneCameraIntrinsics {
    pub fn from_fx_fy(fx: f32, fy: f32, cx: f32, cy: f32, width: u32, height: u32) -> Option<Self> {
        if !fx.is_finite()
            || !fy.is_finite()
            || !cx.is_finite()
            || !cy.is_finite()
            || fx <= 1.0
            || fy <= 1.0
            || width == 0
            || height == 0
        {
            return None;
        }
        let fov_x_degrees = 2.0 * ((width as f32 * 0.5) / fx).atan().to_degrees();
        let fov_y_degrees = 2.0 * ((height as f32 * 0.5) / fy).atan().to_degrees();
        if !fov_x_degrees.is_finite() || !fov_y_degrees.is_finite() {
            return None;
        }
        Some(Self {
            fx,
            fy,
            cx,
            cy,
            width,
            height,
            fov_x_degrees,
            fov_y_degrees,
        })
    }

    pub fn from_focal_length_px(
        focal_length_px: f32,
        width: u32,
        height: u32,
        principal_point: Option<[f32; 2]>,
    ) -> Option<Self> {
        let [cx, cy] = principal_point.unwrap_or([
            (width.saturating_sub(1)) as f32 * 0.5,
            (height.saturating_sub(1)) as f32 * 0.5,
        ]);
        Self::from_fx_fy(focal_length_px, focal_length_px, cx, cy, width, height)
    }

    pub fn from_vertical_fov_degrees(
        vertical_fov_degrees: f32,
        width: u32,
        height: u32,
        principal_point: Option<[f32; 2]>,
    ) -> Option<Self> {
        if !vertical_fov_degrees.is_finite() || vertical_fov_degrees <= 1.0 {
            return None;
        }
        let fy =
            (height as f32 * 0.5) / (vertical_fov_degrees.to_radians() * 0.5).tan().max(1.0e-5);
        Self::from_focal_length_px(fy, width, height, principal_point)
    }

    pub fn is_valid(self) -> bool {
        self.fx.is_finite()
            && self.fy.is_finite()
            && self.cx.is_finite()
            && self.cy.is_finite()
            && self.fov_x_degrees.is_finite()
            && self.fov_y_degrees.is_finite()
            && self.fx > 1.0
            && self.fy > 1.0
            && self.width > 0
            && self.height > 0
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq)]
pub struct EstimatedCamera {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intrinsics: Option<SceneCameraIntrinsics>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub focal_length_px: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub principal_point: Option<[f32; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_size: Option<[u32; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vertical_fov_degrees: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
pub struct EstimatedFloorPlane {
    pub normal: [f32; 3],
    pub distance_m: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub residual_m: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
}

impl Default for EstimatedFloorPlane {
    fn default() -> Self {
        Self {
            normal: [0.0, 1.0, 0.0],
            distance_m: 0.0,
            residual_m: None,
            confidence: None,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ObjectGroundingEvidence {
    pub object_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instance_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reuse_group: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detection: Option<Detection>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mask: Option<ObjectMaskEvidence>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub asset_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contact_pixel: Option<[f32; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub depth_stats: Option<ObjectDepthStats>,
    #[serde(default)]
    pub candidate_floor_contact_rays: Vec<[f32; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metric_contact_point_m: Option<[f32; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_footprint_m: Option<[f32; 2]>,
    #[serde(default)]
    pub provenance: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ObjectMaskEvidence {
    pub provider: String,
    pub model: String,
    pub bbox: [f32; 4],
    pub score: f32,
    pub area_px: u32,
    pub image_size: [u32; 2],
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mask_rle: Vec<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub center_pixel: Option<[f32; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contact_pixel: Option<[f32; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coverage: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mask_png_path: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
pub struct ObjectDepthStats {
    pub median_m: f32,
    pub min_m: f32,
    pub max_m: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contact_m: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample_count: Option<usize>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
pub struct SceneCalibration {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub table_center: Option<[f32; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub table_axis_degrees: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub table_size_m: Option<[f32; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub camera_yaw_degrees: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub camera_pitch_degrees: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub camera_radius_m: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vertical_fov_degrees: Option<f32>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct SceneObjectSpec {
    pub id: String,
    pub label: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    pub bbox: [f32; 4],
    #[serde(default)]
    pub instances: Vec<SceneObjectInstanceSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub representative_instance_id: Option<String>,
    #[serde(default)]
    pub reuse_group: Option<String>,
    #[serde(default = "default_instance_count")]
    pub instance_count: usize,
    pub object_prompt: String,
    #[serde(default)]
    pub camera_hint: Option<String>,
    #[serde(default)]
    pub rotation_hint_degrees: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_footprint_m: Option<[f32; 2]>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct SceneObjectInstanceSpec {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub bbox: [f32; 4],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contact: Option<[f32; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rotation_hint_degrees: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub facing_yaw_degrees: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub side: Option<SceneInstanceSide>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slot_index: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_footprint_m: Option<[f32; 2]>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum SceneInstanceSide {
    Left,
    Right,
    Near,
    Far,
    Head,
    Foot,
    Unknown,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ObjectImageRequest {
    pub object: SceneObjectSpec,
    pub source_scene_path: String,
    pub source_crop_path: String,
    pub object_reference_image_path: String,
    pub prompt: String,
    pub candidate_count: usize,
    pub size: String,
    pub quality: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ObjectImageCandidate {
    pub object_id: String,
    pub candidate_index: usize,
    pub image_path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_image_path: Option<String>,
    pub prompt_hash: String,
    pub score: f32,
    #[serde(default)]
    pub provider_request_id: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
pub struct ObjectImageGenerationPolicy {
    pub min_score: f32,
    pub max_attempts_per_object: usize,
    pub candidates_per_attempt: usize,
}

impl ObjectImageGenerationPolicy {
    pub fn from_total_candidate_budget(candidate_count: usize) -> Self {
        Self {
            min_score: DEFAULT_SCENE_RECONSTRUCTION_IMAGE_SCORE,
            max_attempts_per_object: candidate_count.max(1),
            candidates_per_attempt: 1,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ObjectImageAttemptReport {
    pub object_id: String,
    pub attempt_index: usize,
    pub requested_candidates: usize,
    pub generated_candidates: usize,
    pub best_score_after_attempt: Option<f32>,
    pub accepted: bool,
    pub elapsed_ms: u128,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct SelectedObjectImageCandidate {
    pub object_id: String,
    pub reuse_group: String,
    pub label: String,
    pub image_path: String,
    pub candidate_index: usize,
    pub score: f32,
    pub prompt_hash: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct RejectedObjectImageCandidates {
    pub object_id: String,
    pub best_score: Option<f32>,
    pub min_score: f32,
    pub message: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ObjectImageGenerationReport {
    pub policy: ObjectImageGenerationPolicy,
    pub attempts: Vec<ObjectImageAttemptReport>,
    pub candidates: Vec<ObjectImageCandidate>,
    pub selected_candidates: Vec<SelectedObjectImageCandidate>,
    pub rejected_objects: Vec<RejectedObjectImageCandidates>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct SceneAssetBinding {
    pub asset_id: String,
    pub object_id: String,
    pub label: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub cache_key: Option<String>,
    #[serde(default)]
    pub reusable: bool,
    #[serde(default)]
    pub source_image_path: Option<String>,
    #[serde(default)]
    pub pipeline: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub local_aabb: Option<SceneAssetAabb>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub canonical_frame: Option<SceneAssetFrame>,
    #[serde(default)]
    pub provenance: Option<SceneAssetProvenance>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
pub struct SceneAssetFrame {
    pub yaw_offset_degrees: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub footprint_m: Option<[f32; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub symmetry: Option<SceneAssetSymmetry>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<SceneAssetFrameSource>,
}

impl SceneAssetFrame {
    pub fn heuristic(yaw_offset_degrees: f32, footprint_m: Option<[f32; 2]>) -> Self {
        Self {
            yaw_offset_degrees,
            footprint_m,
            symmetry: None,
            confidence: None,
            source: Some(SceneAssetFrameSource::DescriptorHeuristic),
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SceneAssetSymmetry {
    Asymmetric,
    Bilateral,
    Axis180,
    Axis90,
    Radial,
    Unknown,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SceneAssetFrameSource {
    Explicit,
    AabbHeuristic,
    DescriptorHeuristic,
    PoseFitHeuristic,
    VisualRenderSweep,
    GptVisualSelection,
    AmbiguousFallback,
    Unknown,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct CanonicalPoseEvidence {
    pub asset_id: String,
    pub object_id: String,
    pub label: String,
    pub frame: SceneAssetFrame,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub local_aabb: Option<SceneAssetAabb>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_image_path: Option<String>,
    pub descriptor: String,
    pub method: String,
    pub confidence: f32,
    pub symmetry: SceneAssetSymmetry,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct CanonicalPoseCandidate {
    pub candidate_index: usize,
    pub yaw_offset_degrees: f32,
    pub candidate_yaw_degrees: f32,
    pub score: f32,
    pub source_crop_path: Option<String>,
    pub rendered_image_path: Option<String>,
    #[serde(default)]
    pub metrics: Value,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct CanonicalPoseSelection {
    pub candidate_index: usize,
    pub yaw_offset_degrees: f32,
    pub confidence: f32,
    pub source: SceneAssetFrameSource,
    pub rationale: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct CanonicalPoseCalibrationReport {
    pub schema_version: u32,
    pub mode: String,
    pub asset_id: String,
    pub object_id: String,
    pub label: String,
    pub source_crop_path: Option<String>,
    pub generated_image_path: Option<String>,
    pub selected: CanonicalPoseSelection,
    #[serde(default)]
    pub candidates: Vec<CanonicalPoseCandidate>,
    pub fallback_used: bool,
    #[serde(default)]
    pub warnings: Vec<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
pub struct SceneAssetAabb {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl SceneAssetAabb {
    pub fn size(&self) -> [f32; 3] {
        [
            (self.max[0] - self.min[0]).max(1.0e-5),
            (self.max[1] - self.min[1]).max(1.0e-5),
            (self.max[2] - self.min[2]).max(1.0e-5),
        ]
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct SceneAssetProvenance {
    pub run_id: String,
    pub source_scene_path: String,
    pub source_object_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_crop_path: Option<String>,
    pub generated_by: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ScenePlan {
    pub bsn: String,
    pub placements: Vec<ScenePlacement>,
    #[serde(default)]
    pub camera: Option<SceneCamera>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ScenePlacement {
    pub entity_id: String,
    pub asset_id: String,
    pub translation: [f32; 3],
    pub rotation_y_degrees: f32,
    pub scale: [f32; 3],
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct SceneCamera {
    pub translation: [f32; 3],
    pub focus: [f32; 3],
    #[serde(default)]
    pub yaw: Option<f32>,
    #[serde(default)]
    pub pitch: Option<f32>,
    #[serde(default)]
    pub radius: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vertical_fov_degrees: Option<f32>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ScenePreparation {
    pub run_id: String,
    pub output_dir: String,
    pub source_scene_path: String,
    pub object_reference_image_path: String,
    pub provider: String,
    pub reasoning_model: String,
    pub image_model: String,
    pub object_manifest_schema: Value,
    pub scene_bsn_schema: Value,
    pub object_image_style_prompt: String,
}

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicU64;
use std::time::Duration;

use burn_synth::{ModelSelection, RuntimeConfig, TrellisComputeProfile};
use burn_synth_grounding::{
    GroundingDepthPrecision, LOCATE_ANYTHING_SAFE_IN_TOKEN_LIMIT, LocateAnythingPrecision,
    SegmentationPrecision, SegmentationQuantization,
};
use burn_synth_scene::SceneQualityProfile;
pub use burn_synth_scene::SceneScalePolicy;
use clap::{ArgAction, Args, Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub(crate) const DEFAULT_PROTOCOL_VERSION: &str = "2025-06-18";
pub(crate) const DEFAULT_SCENE_TRELLIS_TARGET_FACES: usize = 80_000;
pub(crate) const DEFAULT_SCENE_TRELLIS_PBR_TEXTURE_SIZE: usize = 512;
pub(crate) const DEFAULT_SCENE_FEEDBACK_ITERS: usize = 8;
pub(crate) const DEFAULT_SCENE_SEGMENTATION_CDN_BASE_URL: &str =
    "https://aberration.technology/model";
pub(crate) const DEFAULT_LOCATE_ANYTHING_CDN_BASE_URL: &str = "https://aberration.technology/model";
pub(crate) static NEXT_SCENE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SceneBuildProgressPhase {
    Started,
    Progress,
    Waiting,
    Completed,
    Failed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SceneBuildExecutionKind {
    Cpu,
    Gpu,
    Network,
    Cache,
    FileIo,
    Viewer,
    Mixed,
    Unknown,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SceneBuildProgressEvent {
    pub run_id: String,
    pub sequence: u64,
    pub stage: String,
    pub phase: SceneBuildProgressPhase,
    pub execution: SceneBuildExecutionKind,
    pub message: String,
    pub elapsed_ms: u64,
    pub item_index: Option<usize>,
    pub item_count: Option<usize>,
    pub artifact_path: Option<String>,
    pub detail: Value,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ForegroundModel {
    Rmbg14,
    Rmbg2,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SynthesisModel {
    Triposg,
    Trellis,
    Triposplat,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InferenceBackend {
    Cpu,
    Wgpu,
    Cuda,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TrellisQuality {
    Low,
    Medium,
    High,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QualityPreset {
    Fast,
    Balanced,
    Full,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FeedbackThresholdProfile {
    Loose,
    Standard,
    Strict,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FeedbackRotationSelector {
    Deterministic,
    RenderedSweep,
    Openai,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SceneRotationFitMode {
    Off,
    DepthMaskRansac,
    GptRefine,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SceneObjectPoseRefinementMode {
    Off,
    Geometry,
    GatedGpt,
    AlwaysGpt,
}

impl SceneObjectPoseRefinementMode {
    pub(crate) fn geometry_enabled(self) -> bool {
        !matches!(self, Self::Off)
    }

    pub(crate) fn gpt_allowed(self) -> bool {
        matches!(self, Self::GatedGpt | Self::AlwaysGpt)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SceneObjectPoseRefinementSet {
    Tables,
    LargeSeating,
    TablesAndLargeSeating,
    AllFurniture,
}

impl SceneObjectPoseRefinementSet {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Tables => "tables",
            Self::LargeSeating => "large-seating",
            Self::TablesAndLargeSeating => "tables-and-large-seating",
            Self::AllFurniture => "all-furniture",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FeedbackRubricScorer {
    Off,
    Openai,
}

#[derive(Clone, Copy, Debug)]
struct QualityDefaults {
    num_steps: usize,
    num_tokens: usize,
    guidance_scale: f32,
    flash_octree_depth: usize,
    flash_min_resolution: usize,
    flash_mini_grid_num: usize,
    flash_num_chunks: usize,
}

impl QualityPreset {
    fn defaults(self) -> QualityDefaults {
        match self {
            Self::Fast => QualityDefaults {
                num_steps: 12,
                num_tokens: 512,
                guidance_scale: 7.0,
                flash_octree_depth: 7,
                flash_min_resolution: 31,
                flash_mini_grid_num: 2,
                flash_num_chunks: 4096,
            },
            Self::Balanced => QualityDefaults {
                num_steps: 20,
                num_tokens: 1024,
                guidance_scale: 7.0,
                flash_octree_depth: 8,
                flash_min_resolution: 31,
                flash_mini_grid_num: 4,
                flash_num_chunks: 8192,
            },
            Self::Full => QualityDefaults {
                num_steps: 50,
                num_tokens: 2048,
                guidance_scale: 7.0,
                flash_octree_depth: 9,
                flash_min_resolution: 63,
                flash_mini_grid_num: 4,
                flash_num_chunks: 10_000,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MeshOutputFormat {
    Obj,
    Gltf,
    Glb,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AssetOutputFormat {
    Auto,
    Glb,
    Splat,
    Ply,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SceneCompositionMode {
    Heuristic,
    CvGrounded,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ScenePoseFitMode {
    ProjectedAabb,
    RenderedSilhouette,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SceneCanonicalPoseMode {
    Off,
    Heuristic,
    RenderSweep,
    Openai,
    Auto,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SceneGroundCalibrationMode {
    DepthHeuristic,
    Gpt,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SceneInstanceGenerationMode {
    CategoryRepresentative,
    FineGrainedTypes,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SceneDepthProvider {
    None,
    DepthPro,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SceneDepthPrecision {
    F32,
    F16,
}

impl From<SceneDepthPrecision> for GroundingDepthPrecision {
    fn from(value: SceneDepthPrecision) -> Self {
        match value {
            SceneDepthPrecision::F32 => GroundingDepthPrecision::F32,
            SceneDepthPrecision::F16 => GroundingDepthPrecision::F16,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SceneLocatorProvider {
    Manifest,
    LocateAnything,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SceneSegmentationProvider {
    None,
    BboxPrompt,
    Sam2,
    Sam3,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SceneSegmentationPrecision {
    F32,
    F16,
    Bf16,
}

impl From<SceneSegmentationPrecision> for SegmentationPrecision {
    fn from(value: SceneSegmentationPrecision) -> Self {
        match value {
            SceneSegmentationPrecision::F32 => SegmentationPrecision::F32,
            SceneSegmentationPrecision::F16 => SegmentationPrecision::F16,
            SceneSegmentationPrecision::Bf16 => SegmentationPrecision::Bf16,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SceneSegmentationQuantization {
    None,
    Q8,
    Q4,
}

impl From<SceneSegmentationQuantization> for SegmentationQuantization {
    fn from(value: SceneSegmentationQuantization) -> Self {
        match value {
            SceneSegmentationQuantization::None => SegmentationQuantization::None,
            SceneSegmentationQuantization::Q8 => SegmentationQuantization::Q8,
            SceneSegmentationQuantization::Q4 => SegmentationQuantization::Q4,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LocateAnythingBackend {
    BurnNative,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SceneLocateAnythingPrecision {
    F32,
    F16,
    Bf16,
}

impl From<SceneLocateAnythingPrecision> for LocateAnythingPrecision {
    fn from(value: SceneLocateAnythingPrecision) -> Self {
        match value {
            SceneLocateAnythingPrecision::F32 => LocateAnythingPrecision::F32,
            SceneLocateAnythingPrecision::F16 => LocateAnythingPrecision::F16,
            SceneLocateAnythingPrecision::Bf16 => LocateAnythingPrecision::Bf16,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CubeClAutotuneLevelSetting {
    Default,
    Minimal,
    Balanced,
    Extensive,
    Full,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CubeClAutotuneCacheSetting {
    Default,
    Local,
    Target,
    Global,
}

#[derive(Parser, Debug, Clone)]
#[command(
    name = "burn_synth_mcp",
    version,
    about = "burn_synth MCP stdio server and scene e2e CLI"
)]
pub struct ServerArgs {
    #[arg(long, value_enum, default_value_t = ForegroundModel::Rmbg2)]
    pub rmbg_model: ForegroundModel,

    #[arg(
        long,
        value_enum,
        value_delimiter = ',',
        default_values_t = [SynthesisModel::Triposg]
    )]
    /// Synthesis backends, ordered by preference (first is preferred).
    pub synthesis_models: Vec<SynthesisModel>,

    #[arg(long, value_enum, default_value_t = InferenceBackend::Wgpu)]
    pub backend: InferenceBackend,

    /// CubeCL autotune effort. `default` respects Burn.toml/cubecl.toml/env configuration.
    #[arg(long, value_enum, default_value_t = CubeClAutotuneLevelSetting::Default)]
    pub cubecl_autotune_level: CubeClAutotuneLevelSetting,

    /// CubeCL autotune cache location. Use `global` to persist across cargo clean/workspace moves.
    #[arg(long, value_enum, default_value_t = CubeClAutotuneCacheSetting::Default)]
    pub cubecl_autotune_cache: CubeClAutotuneCacheSetting,

    #[arg(long)]
    pub weights_root: Option<PathBuf>,

    #[arg(long)]
    pub trellis_weights_root: Option<PathBuf>,

    #[arg(long)]
    pub trellis_image_large_root: Option<PathBuf>,

    #[arg(long, value_enum, default_value_t = TrellisQuality::Low)]
    pub trellis_quality: TrellisQuality,

    /// Enable TRELLIS PBR UV/material texture baking through the Rust/Burn o_voxel export path.
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    pub trellis_pbr: bool,

    /// TRELLIS PBR texture size for Rust/Burn o_voxel GLB export. Uses runtime default when omitted.
    #[arg(long)]
    pub trellis_pbr_texture_size: Option<usize>,

    /// Quality preset (fast, balanced, full). Individual flags override this preset.
    #[arg(long, value_enum, default_value_t = QualityPreset::Balanced)]
    pub quality: QualityPreset,

    #[arg(long)]
    pub bg_weights_root: Option<PathBuf>,

    #[arg(long)]
    pub num_steps: Option<usize>,

    #[arg(long)]
    pub num_tokens: Option<usize>,

    #[arg(long)]
    pub guidance_scale: Option<f32>,

    /// Batch chunk size for image generation tools. Use 0 for auto.
    #[arg(long, default_value_t = 0)]
    pub batch_size: usize,

    /// Explicit VRAM budget in MB for auto batch planning.
    #[arg(long)]
    pub batch_vram_mb: Option<u64>,

    /// Bevy scene command file path for scene_* tools.
    #[arg(long)]
    pub scene_control_path: Option<PathBuf>,

    /// Bevy scene status file path. Defaults to <scene-control-path>.status.json.
    #[arg(long)]
    pub scene_status_path: Option<PathBuf>,

    /// Shared Bevy asset cache root. Defaults to the normal user cache.
    #[arg(long)]
    pub catalog_cache_root: Option<PathBuf>,

    /// Timeout for scene command acknowledgements.
    #[arg(long, default_value_t = 5000)]
    pub scene_timeout_ms: u64,

    /// OpenAI reasoning model used by scene planning tools.
    #[arg(long, default_value = "gpt-5.5")]
    pub openai_reasoning_model: String,

    /// OpenAI image model used by object-image generation tools.
    #[arg(long, default_value = "gpt-image-2")]
    pub openai_image_model: String,

    /// Example/reference isolated-object image for OpenAI object generation.
    #[arg(long, default_value = "docs/input_chair.jpg")]
    pub scene_object_reference_image: PathBuf,

    /// Local cache directory for burn_depth CDN model shards and assembled .bpk artifacts.
    #[arg(long)]
    pub depth_cache_dir: Option<PathBuf>,

    /// Precision used for the burn_depth DepthPro checkpoint.
    #[arg(long, value_enum, default_value_t = SceneDepthPrecision::F16)]
    pub depth_precision: SceneDepthPrecision,

    /// Allow burn_depth to download missing CDN model shards.
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    pub depth_allow_download: bool,

    /// LocateAnything model root used by the Burn-native locator.
    #[arg(long, default_value = "assets/models/LocateAnything-3B")]
    pub locate_anything_model_root: PathBuf,

    /// Local cache directory for LocateAnything CDN model shards and materialized safetensors.
    #[arg(long)]
    pub locate_anything_cache_dir: Option<PathBuf>,

    /// CDN base URL for LocateAnything model metadata and bpk shard manifests.
    #[arg(long, default_value = DEFAULT_LOCATE_ANYTHING_CDN_BASE_URL)]
    pub locate_anything_cdn_base_url: Option<String>,

    /// Allow LocateAnything runtime to download missing CDN artifacts.
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    pub locate_anything_allow_download: bool,

    /// Precision/artifact variant used for LocateAnything CDN bpk shards.
    #[arg(long, value_enum, default_value_t = SceneLocateAnythingPrecision::Bf16)]
    pub locate_anything_precision: SceneLocateAnythingPrecision,

    /// Image token limit for the LocateAnything locator. Defaults to the WGPU-safe limit.
    #[arg(long, default_value_t = LOCATE_ANYTHING_SAFE_IN_TOKEN_LIMIT as usize)]
    pub locate_anything_in_token_limit: usize,

    /// LocateAnything execution backend used by scene-ground when --locator locate-anything.
    #[arg(long, value_enum, default_value_t = LocateAnythingBackend::BurnNative)]
    pub locate_anything_backend: LocateAnythingBackend,

    /// Optional segmentation/mask provider used by CV-grounded scene composition.
    #[arg(long, value_enum, default_value_t = SceneSegmentationProvider::Sam2)]
    pub scene_segmentation_provider: SceneSegmentationProvider,

    /// Segmentation checkpoint precision/artifact variant.
    #[arg(long, value_enum, default_value_t = SceneSegmentationPrecision::F16)]
    pub scene_segmentation_precision: SceneSegmentationPrecision,

    /// Segmentation checkpoint quantization/artifact variant.
    #[arg(long, value_enum, default_value_t = SceneSegmentationQuantization::None)]
    pub scene_segmentation_quantization: SceneSegmentationQuantization,

    /// Local segmentation model root for SAM-family Burn artifacts.
    #[arg(long)]
    pub scene_segmentation_model_root: Option<PathBuf>,

    /// Local segmentation cache directory for CDN-resolved artifacts.
    #[arg(long)]
    pub scene_segmentation_cache_dir: Option<PathBuf>,

    /// CDN base URL for segmentation model manifests/shards.
    #[arg(long, default_value = DEFAULT_SCENE_SEGMENTATION_CDN_BASE_URL)]
    pub scene_segmentation_cdn_base_url: Option<String>,

    /// Allow segmentation runtime to fetch missing CDN artifacts when a loader is available.
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    pub scene_segmentation_allow_download: bool,

    #[command(subcommand)]
    pub(crate) command: Option<ServerCommand>,
}

#[derive(Subcommand, Debug, Clone)]
#[allow(clippy::enum_variant_names)]
pub(crate) enum ServerCommand {
    /// Run the full source-scene image -> object images -> assets -> grounded BSN pipeline once.
    SceneBuild(SceneBuildCliArgs),
    /// Recompute scene composition from saved assets and grounding evidence without regenerating assets.
    SceneGround(SceneGroundCliArgs),
    /// Run visual grounding providers and write an inspection report without scene composition.
    SceneGroundingReport(SceneGroundingReportCliArgs),
    /// Replay render-capture-feedback using existing scene-build artifacts.
    SceneFeedbackReplay(SceneFeedbackReplayCliArgs),
}

#[derive(Args, Debug, Clone)]
pub(crate) struct SceneBuildCliArgs {
    /// Source scene image path.
    #[arg(long, visible_alias = "scene")]
    pub source_scene_path: PathBuf,

    /// Example/reference isolated-object image. Defaults to --scene-object-reference-image.
    #[arg(long, visible_alias = "object-reference")]
    pub object_reference_image_path: Option<PathBuf>,

    /// Output directory for generated images, assets, BSN, metrics, and response JSON.
    #[arg(long)]
    pub output_dir: Option<PathBuf>,

    /// Total guarded generated object-image candidate budget per reusable object.
    #[arg(long, visible_alias = "candidates")]
    pub candidate_count: Option<usize>,

    /// Maximum guarded image-generation attempts per object. Defaults to candidate_count.
    #[arg(long)]
    pub candidate_retry_attempts: Option<usize>,

    /// Image candidates requested per retry attempt. Defaults to 1.
    #[arg(long)]
    pub candidate_batch_size: Option<usize>,

    /// Minimum isolated-object reconstruction score before TRELLIS lifting.
    #[arg(long)]
    pub min_reconstruction_score: Option<f32>,

    /// OpenAI object-image generation profile.
    #[arg(long, value_enum, visible_alias = "profile")]
    pub quality_profile: Option<SceneQualityProfile>,

    /// Allow object planning to consider existing catalog assets.
    #[arg(long, default_value_t = false, action = ArgAction::Set)]
    pub allow_catalog_reuse: bool,

    /// Lift selected generated object images into 3D assets.
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    pub lift_assets: bool,

    /// Image-to-3D asset model list for scene object lifting, ordered by preference.
    #[arg(long, value_enum, value_delimiter = ',')]
    pub synthesis_models: Option<Vec<SynthesisModel>>,

    /// TRELLIS target face count. Use 0 to disable decimation.
    #[arg(long)]
    pub target_faces: Option<usize>,

    /// Batch chunk size for object-image lifting. Use 0 for auto/global default.
    #[arg(long)]
    pub batch_size: Option<usize>,

    /// Explicit VRAM budget in MB for auto batch planning.
    #[arg(long)]
    pub batch_vram_mb: Option<u64>,

    /// Enable TRELLIS PBR UV/material texture baking through the Rust/Burn o_voxel export path.
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    pub trellis_pbr: bool,

    /// TRELLIS PBR texture size for Rust/Burn o_voxel GLB export.
    #[arg(long)]
    pub trellis_pbr_texture_size: Option<usize>,

    /// Add lifted assets to the shared Bevy catalog/cache.
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    pub promote_to_catalog: bool,

    /// Composition mode used after generated object assets are lifted.
    #[arg(long, value_enum, default_value_t = SceneCompositionMode::CvGrounded)]
    pub composition_mode: SceneCompositionMode,

    /// Pose fitting strategy used inside cv-grounded composition.
    #[arg(long, value_enum, default_value_t = ScenePoseFitMode::RenderedSilhouette)]
    pub pose_fit: ScenePoseFitMode,

    /// Canonical asset orientation strategy.
    #[arg(long, value_enum, default_value_t = SceneCanonicalPoseMode::Off)]
    pub canonical_pose: SceneCanonicalPoseMode,

    /// Generated asset scale policy used by layout and feedback.
    #[arg(long, value_enum, default_value_t = SceneScalePolicy::AssetPreserving)]
    pub scale_policy: SceneScalePolicy,

    /// Maximum pose candidates per object for deterministic cv-grounded fitting.
    #[arg(long, default_value_t = 32)]
    pub max_pose_candidates: usize,

    /// Save pose fitting debug sidecars when artifacts are enabled.
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    pub save_pose_debug: bool,

    /// Camera/floor calibration source used after DepthPro evidence is available.
    #[arg(long, value_enum, default_value_t = SceneGroundCalibrationMode::Gpt)]
    pub ground_calibration: SceneGroundCalibrationMode,

    /// Reusable-object instance generation mode. Default generates one asset per category/reuse group.
    #[arg(long, value_enum, default_value_t = SceneInstanceGenerationMode::CategoryRepresentative)]
    pub instance_generation: SceneInstanceGenerationMode,

    /// Depth provider used by CV-grounded scene composition.
    #[arg(long, value_enum, default_value_t = SceneDepthProvider::DepthPro)]
    pub depth_provider: SceneDepthProvider,

    /// Locator provider used by CV-grounded scene composition.
    #[arg(long, value_enum, default_value_t = SceneLocatorProvider::LocateAnything)]
    pub locator: SceneLocatorProvider,

    /// Override the server LocateAnything backend for this scene-build run.
    #[arg(long, value_enum)]
    pub locate_anything_backend: Option<LocateAnythingBackend>,

    /// Override the server scene segmentation provider for this scene-build run.
    #[arg(long, value_enum)]
    pub segmentation_provider: Option<SceneSegmentationProvider>,

    /// Override the server scene segmentation precision for this scene-build run.
    #[arg(long, value_enum)]
    pub segmentation_precision: Option<SceneSegmentationPrecision>,

    /// Override the server scene segmentation quantization for this scene-build run.
    #[arg(long, value_enum)]
    pub segmentation_quantization: Option<SceneSegmentationQuantization>,

    /// Write structured e2e artifacts to the output directory.
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    pub write_artifacts: bool,

    /// Clear the live Bevy scene before applying generated commands.
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    pub clear_existing: bool,

    /// Apply the generated scene to the configured Bevy scene bridge.
    #[arg(long)]
    pub apply: bool,

    /// Run bounded render-capture-feedback layout validation/refinement.
    ///
    /// This is intentionally opt-in. The default scene flow should be a deterministic
    /// geometric solve from LocateAnything boxes, SAM masks, DepthPro depth, and
    /// projected lifted-asset visible surfaces.
    #[arg(long, default_value_t = false, action = ArgAction::Set)]
    pub feedback: bool,

    /// Maximum render-capture-feedback iterations.
    #[arg(long, default_value_t = DEFAULT_SCENE_FEEDBACK_ITERS)]
    pub feedback_iters: usize,

    /// Leave a temporary feedback viewer running after scene-build completes.
    #[arg(long, default_value_t = false, action = ArgAction::Set)]
    pub feedback_keep_viewer: bool,

    /// Optional directory for feedback screenshots, status, metrics, and deltas.
    #[arg(long)]
    pub feedback_capture_dir: Option<PathBuf>,

    /// Geometry-first feedback pass/fail threshold profile.
    #[arg(long, value_enum, default_value_t = FeedbackThresholdProfile::Standard)]
    pub feedback_threshold_profile: FeedbackThresholdProfile,

    /// Rotation candidate selector used during render-capture feedback.
    #[arg(long, value_enum, default_value_t = FeedbackRotationSelector::Deterministic)]
    pub feedback_rotation_selector: FeedbackRotationSelector,

    /// Extra pre-feedback Y-axis rotation fit.
    ///
    /// Keep this off by default: the canonical default pose solve is
    /// --pose-fit=rendered-silhouette, which already scores X/Z/yaw/uniform-scale
    /// candidates against source masks/depth and projected GLB visible surfaces.
    #[arg(long, value_enum, default_value_t = SceneRotationFitMode::Off)]
    pub rotation_fit: SceneRotationFitMode,

    /// Maximum GPT refinement rounds when --rotation-fit=gpt-refine.
    #[arg(long, default_value_t = 0)]
    pub rotation_fit_max_gpt_rounds: usize,

    /// Minimum visible-surface mask IoU required before applying an objective rotation candidate.
    #[arg(long, default_value_t = 0.45)]
    pub rotation_fit_min_mask_iou: f32,

    /// Maximum accepted visible-surface median-depth error in meters.
    #[arg(long, default_value_t = 0.35)]
    pub rotation_fit_max_depth_error_m: f32,

    /// Write rotation-fit candidate overlays, report JSON, and HTML review artifacts.
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    pub rotation_fit_write_artifacts: bool,

    /// Object-set-specific pose refinement after the generic visible-surface fit.
    ///
    /// Defaults to gated-gpt: deterministic mask/depth candidate fitting runs
    /// first, and the report marks only ambiguous/failed object fits for
    /// bounded GPT candidate selection.
    #[arg(long, value_enum, default_value_t = SceneObjectPoseRefinementMode::GatedGpt)]
    pub object_pose_refinement: SceneObjectPoseRefinementMode,

    /// Object set targeted by --object-pose-refinement.
    #[arg(long, value_enum, default_value_t = SceneObjectPoseRefinementSet::TablesAndLargeSeating)]
    pub object_pose_refinement_set: SceneObjectPoseRefinementSet,

    /// Optional source-vs-render scene quality rubric scorer.
    #[arg(long, value_enum, default_value_t = FeedbackRubricScorer::Off)]
    pub feedback_rubric_scorer: FeedbackRubricScorer,
}

#[derive(Args, Debug, Clone)]
pub(crate) struct SceneGroundCliArgs {
    /// Source scene image path.
    #[arg(long, visible_alias = "scene")]
    pub source_scene_path: PathBuf,

    /// Existing object manifest JSON.
    #[arg(long)]
    pub manifest: PathBuf,

    /// Existing asset bindings JSON produced by scene-build/images_to_assets.
    #[arg(long)]
    pub asset_bindings: PathBuf,

    /// Optional grounding evidence JSON. When omitted, manifest bbox/contact evidence is used.
    #[arg(long)]
    pub grounding_evidence: Option<PathBuf>,

    /// Output directory for grounded layout, BSN, commands, and metrics.
    #[arg(long)]
    pub output_dir: Option<PathBuf>,

    /// Composition mode. cv-grounded writes evidence artifacts and uses the evidence adapter.
    #[arg(long, value_enum, default_value_t = SceneCompositionMode::CvGrounded)]
    pub composition_mode: SceneCompositionMode,

    /// Pose fitting strategy used inside cv-grounded composition.
    #[arg(long, value_enum, default_value_t = ScenePoseFitMode::RenderedSilhouette)]
    pub pose_fit: ScenePoseFitMode,

    /// Canonical asset orientation strategy.
    #[arg(long, value_enum, default_value_t = SceneCanonicalPoseMode::Off)]
    pub canonical_pose: SceneCanonicalPoseMode,

    /// Generated asset scale policy used by layout and feedback.
    #[arg(long, value_enum, default_value_t = SceneScalePolicy::AssetPreserving)]
    pub scale_policy: SceneScalePolicy,

    /// Maximum pose candidates per object for deterministic cv-grounded fitting.
    #[arg(long, default_value_t = 32)]
    pub max_pose_candidates: usize,

    /// Save pose fitting debug sidecars when artifacts are enabled.
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    pub save_pose_debug: bool,

    /// Camera/floor calibration source used when grounding evidence is refreshed.
    #[arg(long, value_enum, default_value_t = SceneGroundCalibrationMode::Gpt)]
    pub ground_calibration: SceneGroundCalibrationMode,

    /// Depth provider identifier for run metadata.
    #[arg(long, value_enum, default_value_t = SceneDepthProvider::DepthPro)]
    pub depth_provider: SceneDepthProvider,

    /// Locator provider used to refresh source-image object boxes when evidence is omitted.
    #[arg(long, value_enum, default_value_t = SceneLocatorProvider::LocateAnything)]
    pub locator: SceneLocatorProvider,

    /// Override the server LocateAnything backend for this scene-ground run.
    #[arg(long, value_enum)]
    pub locate_anything_backend: Option<LocateAnythingBackend>,

    /// Override the server scene segmentation provider for this scene-ground run.
    #[arg(long, value_enum)]
    pub segmentation_provider: Option<SceneSegmentationProvider>,

    /// Override the server scene segmentation precision for this scene-ground run.
    #[arg(long, value_enum)]
    pub segmentation_precision: Option<SceneSegmentationPrecision>,

    /// Override the server scene segmentation quantization for this scene-ground run.
    #[arg(long, value_enum)]
    pub segmentation_quantization: Option<SceneSegmentationQuantization>,

    /// Clear the live Bevy scene before applying generated commands.
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    pub clear_existing: bool,

    /// Apply the generated scene to the configured Bevy scene bridge.
    #[arg(long)]
    pub apply: bool,

    /// Run bounded render-capture-feedback layout validation/refinement.
    #[arg(long, default_value_t = false, action = ArgAction::Set)]
    pub feedback: bool,

    /// Maximum render-capture-feedback iterations.
    #[arg(long, default_value_t = DEFAULT_SCENE_FEEDBACK_ITERS)]
    pub feedback_iters: usize,

    /// Leave a temporary feedback viewer running after scene-ground completes.
    #[arg(long, default_value_t = false, action = ArgAction::Set)]
    pub feedback_keep_viewer: bool,

    /// Optional directory for feedback screenshots, status, metrics, and deltas.
    #[arg(long)]
    pub feedback_capture_dir: Option<PathBuf>,

    /// Geometry-first feedback pass/fail threshold profile.
    #[arg(long, value_enum, default_value_t = FeedbackThresholdProfile::Standard)]
    pub feedback_threshold_profile: FeedbackThresholdProfile,

    /// Rotation candidate selector used during render-capture feedback.
    #[arg(long, value_enum, default_value_t = FeedbackRotationSelector::Deterministic)]
    pub feedback_rotation_selector: FeedbackRotationSelector,

    /// Extra pre-feedback Y-axis rotation fit.
    #[arg(long, value_enum, default_value_t = SceneRotationFitMode::Off)]
    pub rotation_fit: SceneRotationFitMode,

    /// Maximum GPT refinement rounds when --rotation-fit=gpt-refine.
    #[arg(long, default_value_t = 0)]
    pub rotation_fit_max_gpt_rounds: usize,

    /// Minimum visible-surface mask IoU required before applying an objective rotation candidate.
    #[arg(long, default_value_t = 0.45)]
    pub rotation_fit_min_mask_iou: f32,

    /// Maximum accepted visible-surface median-depth error in meters.
    #[arg(long, default_value_t = 0.35)]
    pub rotation_fit_max_depth_error_m: f32,

    /// Write rotation-fit candidate overlays, report JSON, and HTML review artifacts.
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    pub rotation_fit_write_artifacts: bool,

    /// Object-set-specific pose refinement after the generic visible-surface fit.
    #[arg(long, value_enum, default_value_t = SceneObjectPoseRefinementMode::GatedGpt)]
    pub object_pose_refinement: SceneObjectPoseRefinementMode,

    /// Object set targeted by --object-pose-refinement.
    #[arg(long, value_enum, default_value_t = SceneObjectPoseRefinementSet::TablesAndLargeSeating)]
    pub object_pose_refinement_set: SceneObjectPoseRefinementSet,

    /// Optional source-vs-render scene quality rubric scorer.
    #[arg(long, value_enum, default_value_t = FeedbackRubricScorer::Off)]
    pub feedback_rubric_scorer: FeedbackRubricScorer,
}

#[derive(Args, Debug, Clone)]
pub(crate) struct SceneGroundingReportCliArgs {
    /// Source scene image path.
    #[arg(long, visible_alias = "scene")]
    pub source_scene_path: PathBuf,

    /// Object query prompt for LocateAnything. Repeat for multiple categories.
    #[arg(long = "query", value_delimiter = ',')]
    pub queries: Vec<String>,

    /// Expected repeated instance count, formatted as query=count. Used only for report grouping.
    #[arg(long = "expected-count")]
    pub expected_counts: Vec<String>,

    /// Output directory for overlays, evidence JSON, quality metrics, and HTML review.
    #[arg(long)]
    pub output_dir: Option<PathBuf>,

    /// Force model loading through the configured CDN/cache rather than any usable local model root.
    #[arg(long, default_value_t = false, action = ArgAction::SetTrue)]
    pub cdn_only: bool,

    /// Number of additional warm passes to run in-process after the first pass.
    #[arg(long, default_value_t = 0)]
    pub warm_runs: usize,

    /// Locator provider used for bbox grounding.
    #[arg(long, value_enum, default_value_t = SceneLocatorProvider::LocateAnything)]
    pub locator: SceneLocatorProvider,

    /// Override the server LocateAnything backend.
    #[arg(long, value_enum)]
    pub locate_anything_backend: Option<LocateAnythingBackend>,

    /// Segmentation provider to run after bbox grounding.
    #[arg(long, value_enum, default_value_t = SceneSegmentationProvider::Sam2)]
    pub segmentation_provider: SceneSegmentationProvider,

    /// Also write bbox-prompt masks as a cheap rectangular baseline.
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    pub bbox_prompt_baseline: bool,

    /// Override the server scene segmentation precision.
    #[arg(long, value_enum)]
    pub segmentation_precision: Option<SceneSegmentationPrecision>,

    /// Override the server scene segmentation quantization.
    #[arg(long, value_enum)]
    pub segmentation_quantization: Option<SceneSegmentationQuantization>,

    /// Optional depth provider for report-side depth/floor annotation.
    #[arg(long, value_enum, default_value_t = SceneDepthProvider::None)]
    pub depth_provider: SceneDepthProvider,

    /// Warn when a detection bbox covers more than this fraction of the image.
    #[arg(long, default_value_t = 0.50)]
    pub max_bbox_area: f32,

    /// Warn when a segmentation mask covers more than this fraction of the image.
    #[arg(long, default_value_t = 0.50)]
    pub max_mask_coverage: f32,
}

#[derive(Args, Debug, Clone)]
pub(crate) struct SceneFeedbackReplayCliArgs {
    /// Existing scene-build output directory with manifest/assets/layout/commands artifacts.
    #[arg(long)]
    pub output_dir: PathBuf,

    /// Manifest JSON path. Defaults to <output-dir>/manifest.json.
    #[arg(long)]
    pub manifest_path: Option<PathBuf>,

    /// Asset bindings JSON path. Defaults to <output-dir>/asset_bindings.json.
    #[arg(long)]
    pub asset_bindings_path: Option<PathBuf>,

    /// Grounded layout JSON path. Defaults to <output-dir>/grounded_layout.json.
    #[arg(long)]
    pub grounded_layout_path: Option<PathBuf>,

    /// Scene command JSON path. Defaults to <output-dir>/commands.json.
    #[arg(long)]
    pub commands_path: Option<PathBuf>,

    /// Rebuild initial commands from grounded_layout.bsn instead of replaying saved commands.json.
    #[arg(long, default_value_t = false, action = ArgAction::SetTrue)]
    pub rebuild_commands_from_grounded_layout: bool,

    /// Maximum render-capture-feedback iterations.
    #[arg(long, default_value_t = DEFAULT_SCENE_FEEDBACK_ITERS)]
    pub feedback_iters: usize,

    /// Leave the temporary feedback viewer running after replay.
    #[arg(long, default_value_t = false, action = ArgAction::Set)]
    pub feedback_keep_viewer: bool,

    /// Optional directory for replay screenshots, status, metrics, and deltas.
    #[arg(long)]
    pub feedback_capture_dir: Option<PathBuf>,

    /// Geometry-first feedback pass/fail threshold profile.
    #[arg(long, value_enum, default_value_t = FeedbackThresholdProfile::Standard)]
    pub feedback_threshold_profile: FeedbackThresholdProfile,

    /// Rotation candidate selector used during render-capture feedback.
    #[arg(long, value_enum, default_value_t = FeedbackRotationSelector::Deterministic)]
    pub feedback_rotation_selector: FeedbackRotationSelector,

    /// Extra pre-feedback Y-axis rotation fit from source SAM masks, DepthPro depth,
    /// and projected GLB visible surface.
    #[arg(long, value_enum, default_value_t = SceneRotationFitMode::Off)]
    pub rotation_fit: SceneRotationFitMode,

    /// Maximum GPT refinement rounds when --rotation-fit=gpt-refine.
    #[arg(long, default_value_t = 0)]
    pub rotation_fit_max_gpt_rounds: usize,

    /// Minimum visible-surface mask IoU required before applying an objective rotation candidate.
    #[arg(long, default_value_t = 0.45)]
    pub rotation_fit_min_mask_iou: f32,

    /// Maximum accepted visible-surface median-depth error in meters.
    #[arg(long, default_value_t = 0.35)]
    pub rotation_fit_max_depth_error_m: f32,

    /// Write rotation-fit candidate overlays, report JSON, and HTML review artifacts.
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    pub rotation_fit_write_artifacts: bool,

    /// Optional source-vs-render scene quality rubric scorer.
    #[arg(long, value_enum, default_value_t = FeedbackRubricScorer::Off)]
    pub feedback_rubric_scorer: FeedbackRubricScorer,
}

#[derive(Clone, Debug)]
pub struct ServerConfig {
    pub default_rmbg_model: ForegroundModel,
    pub default_synthesis_models: Vec<SynthesisModel>,
    pub default_backend: InferenceBackend,
    pub weights_root: Option<PathBuf>,
    pub trellis_weights_root: Option<PathBuf>,
    pub trellis_image_large_root: Option<PathBuf>,
    pub trellis_quality: TrellisQuality,
    pub trellis_pbr_enabled: bool,
    pub trellis_pbr_texture_size: Option<usize>,
    pub quality: QualityPreset,
    pub bg_weights_root: Option<PathBuf>,
    pub num_steps: usize,
    pub num_tokens: usize,
    pub guidance_scale: f32,
    pub flash_octree_depth: usize,
    pub flash_min_resolution: usize,
    pub flash_mini_grid_num: usize,
    pub flash_num_chunks: usize,
    pub batch_size: Option<usize>,
    pub batch_vram_mb: Option<u64>,
    pub scene_control_path: Option<PathBuf>,
    pub scene_status_path: Option<PathBuf>,
    pub catalog_cache_root: Option<PathBuf>,
    pub scene_timeout: Duration,
    pub openai_api_key: Option<String>,
    pub openai_base_url: Option<String>,
    pub openai_project_id: Option<String>,
    pub openai_reasoning_model: String,
    pub openai_image_model: String,
    pub scene_object_reference_image: PathBuf,
    pub depth_cache_dir: Option<PathBuf>,
    pub depth_precision: SceneDepthPrecision,
    pub depth_allow_download: bool,
    pub locate_anything_model_root: PathBuf,
    pub locate_anything_cache_dir: Option<PathBuf>,
    pub locate_anything_cdn_base_url: Option<String>,
    pub locate_anything_allow_download: bool,
    pub locate_anything_precision: SceneLocateAnythingPrecision,
    pub locate_anything_in_token_limit: usize,
    pub locate_anything_backend: LocateAnythingBackend,
    pub scene_segmentation_provider: SceneSegmentationProvider,
    pub scene_segmentation_precision: SceneSegmentationPrecision,
    pub scene_segmentation_quantization: SceneSegmentationQuantization,
    pub scene_segmentation_model_root: Option<PathBuf>,
    pub scene_segmentation_cache_dir: Option<PathBuf>,
    pub scene_segmentation_cdn_base_url: Option<String>,
    pub scene_segmentation_allow_download: bool,
    pub cubecl_autotune_level: CubeClAutotuneLevelSetting,
    pub cubecl_autotune_cache: CubeClAutotuneCacheSetting,
}

impl ServerConfig {
    pub fn from_args(args: ServerArgs) -> Self {
        let quality = args.quality;
        let defaults = quality.defaults();
        Self {
            default_rmbg_model: args.rmbg_model,
            default_synthesis_models: sanitize_synthesis_models(args.synthesis_models),
            default_backend: args.backend,
            weights_root: args.weights_root,
            trellis_weights_root: args.trellis_weights_root,
            trellis_image_large_root: args.trellis_image_large_root,
            trellis_quality: args.trellis_quality,
            trellis_pbr_enabled: args.trellis_pbr,
            trellis_pbr_texture_size: args.trellis_pbr_texture_size,
            quality,
            bg_weights_root: args.bg_weights_root,
            num_steps: args.num_steps.unwrap_or(defaults.num_steps),
            num_tokens: args.num_tokens.unwrap_or(defaults.num_tokens),
            guidance_scale: args.guidance_scale.unwrap_or(defaults.guidance_scale),
            flash_octree_depth: defaults.flash_octree_depth,
            flash_min_resolution: defaults.flash_min_resolution,
            flash_mini_grid_num: defaults.flash_mini_grid_num,
            flash_num_chunks: defaults.flash_num_chunks,
            batch_size: (args.batch_size > 0).then_some(args.batch_size),
            batch_vram_mb: args.batch_vram_mb,
            scene_status_path: args.scene_status_path.or_else(|| {
                args.scene_control_path
                    .as_ref()
                    .map(|path| path.with_extension("status.json"))
            }),
            scene_control_path: args.scene_control_path,
            catalog_cache_root: args.catalog_cache_root,
            scene_timeout: Duration::from_millis(args.scene_timeout_ms.max(1)),
            openai_api_key: env_or_dotenv_var("OPENAI_API_KEY"),
            openai_base_url: env_or_dotenv_var("OPENAI_BASE_URL"),
            openai_project_id: env_or_dotenv_var("OPENAI_PROJECT_ID"),
            openai_reasoning_model: args.openai_reasoning_model,
            openai_image_model: args.openai_image_model,
            scene_object_reference_image: args.scene_object_reference_image,
            depth_cache_dir: args.depth_cache_dir,
            depth_precision: args.depth_precision,
            depth_allow_download: args.depth_allow_download,
            locate_anything_model_root: args.locate_anything_model_root,
            locate_anything_cache_dir: args.locate_anything_cache_dir,
            locate_anything_cdn_base_url: args.locate_anything_cdn_base_url,
            locate_anything_allow_download: args.locate_anything_allow_download,
            locate_anything_precision: args.locate_anything_precision,
            locate_anything_in_token_limit: args.locate_anything_in_token_limit.max(1),
            locate_anything_backend: args.locate_anything_backend,
            scene_segmentation_provider: args.scene_segmentation_provider,
            scene_segmentation_precision: args.scene_segmentation_precision,
            scene_segmentation_quantization: args.scene_segmentation_quantization,
            scene_segmentation_model_root: args.scene_segmentation_model_root,
            scene_segmentation_cache_dir: args.scene_segmentation_cache_dir,
            scene_segmentation_cdn_base_url: args.scene_segmentation_cdn_base_url,
            scene_segmentation_allow_download: args.scene_segmentation_allow_download,
            cubecl_autotune_level: args.cubecl_autotune_level,
            cubecl_autotune_cache: args.cubecl_autotune_cache,
        }
    }

    pub(crate) fn runtime_config(&self) -> RuntimeConfig {
        let mut config = RuntimeConfig {
            model_selection: ModelSelection::new(
                self.default_synthesis_models
                    .iter()
                    .copied()
                    .map(Into::into),
                self.default_rmbg_model.into(),
            ),
            backend: self.default_backend.into(),
            weights_root: self.weights_root.clone(),
            trellis_weights_root: self.trellis_weights_root.clone(),
            trellis_image_large_root: self.trellis_image_large_root.clone(),
            trellis_quality: self.trellis_quality.into(),
            trellis_compute_profile: self.trellis_compute_profile(),
            trellis_pbr_enabled: self.trellis_pbr_enabled,
            trellis_pbr_texture_size: self.trellis_pbr_texture_size,
            bg_weights_root: self.bg_weights_root.clone(),
            num_steps: self.num_steps,
            num_tokens: self.num_tokens,
            guidance_scale: self.guidance_scale,
            ..RuntimeConfig::default()
        };
        config.flash_extract.octree_depth = self.flash_octree_depth;
        config.flash_extract.min_resolution = self.flash_min_resolution;
        config.flash_extract.mini_grid_num = self.flash_mini_grid_num;
        config.flash_extract.num_chunks = self.flash_num_chunks;
        config
    }

    fn trellis_compute_profile(&self) -> TrellisComputeProfile {
        if self.default_backend == InferenceBackend::Wgpu
            && self
                .default_synthesis_models
                .iter()
                .any(|model| matches!(model, SynthesisModel::Trellis))
        {
            TrellisComputeProfile::WgpuFastF16
        } else {
            RuntimeConfig::default().trellis_compute_profile
        }
    }
}

pub(crate) fn sanitize_synthesis_models(models: Vec<SynthesisModel>) -> Vec<SynthesisModel> {
    let mut out = Vec::new();
    for model in models {
        if !out.contains(&model) {
            out.push(model);
        }
    }
    if out.is_empty() {
        out.push(SynthesisModel::Triposg);
    }
    out
}

fn env_or_dotenv_var(key: &str) -> Option<String> {
    env::var(key)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| dotenv_var(Path::new(".env"), key))
}

pub(crate) fn dotenv_var(path: &Path, key: &str) -> Option<String> {
    let contents = fs::read_to_string(path).ok()?;
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line.strip_prefix("export ").unwrap_or(line).trim();
        let Some((name, value)) = line.split_once('=') else {
            continue;
        };
        if name.trim() == key {
            return Some(parse_dotenv_value(value.trim()));
        }
    }
    None
}

fn parse_dotenv_value(value: &str) -> String {
    let value = value.trim();
    if value.len() >= 2 {
        let bytes = value.as_bytes();
        if (bytes[0] == b'"' && bytes[value.len() - 1] == b'"')
            || (bytes[0] == b'\'' && bytes[value.len() - 1] == b'\'')
        {
            return value[1..value.len() - 1].to_string();
        }
    }
    value
        .split_once(" #")
        .map(|(prefix, _)| prefix.trim_end())
        .unwrap_or(value)
        .to_string()
}

impl ForegroundModel {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            ForegroundModel::Rmbg14 => "rmbg14",
            ForegroundModel::Rmbg2 => "rmbg2",
        }
    }
}

impl SynthesisModel {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            SynthesisModel::Triposg => "triposg",
            SynthesisModel::Trellis => "trellis",
            SynthesisModel::Triposplat => "triposplat",
        }
    }
}

impl InferenceBackend {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            InferenceBackend::Cpu => "cpu",
            InferenceBackend::Wgpu => "wgpu",
            InferenceBackend::Cuda => "cuda",
        }
    }
}

impl MeshOutputFormat {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            MeshOutputFormat::Obj => "obj",
            MeshOutputFormat::Gltf => "gltf",
            MeshOutputFormat::Glb => "glb",
        }
    }
}

impl AssetOutputFormat {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            AssetOutputFormat::Auto => "auto",
            AssetOutputFormat::Glb => "glb",
            AssetOutputFormat::Splat => "splat",
            AssetOutputFormat::Ply => "ply",
        }
    }
}

pub(crate) fn runtime_foreground_model_str(value: burn_synth::ForegroundModel) -> &'static str {
    match value {
        burn_synth::ForegroundModel::Rmbg14 => "rmbg14",
        burn_synth::ForegroundModel::Rmbg2 => "rmbg2",
    }
}

pub(crate) fn runtime_synthesis_model_str(value: burn_synth::SynthesisModel) -> &'static str {
    match value {
        burn_synth::SynthesisModel::Triposg => "triposg",
        burn_synth::SynthesisModel::Trellis => "trellis",
        burn_synth::SynthesisModel::Triposplat => "triposplat",
    }
}

pub(crate) fn runtime_backend_str(value: burn_synth::InferenceBackend) -> &'static str {
    match value {
        burn_synth::InferenceBackend::Cpu => "cpu",
        burn_synth::InferenceBackend::Wgpu => "wgpu",
        burn_synth::InferenceBackend::Cuda => "cuda",
    }
}

impl From<ForegroundModel> for burn_synth::ForegroundModel {
    fn from(value: ForegroundModel) -> Self {
        match value {
            ForegroundModel::Rmbg14 => Self::Rmbg14,
            ForegroundModel::Rmbg2 => Self::Rmbg2,
        }
    }
}

impl From<SynthesisModel> for burn_synth::SynthesisModel {
    fn from(value: SynthesisModel) -> Self {
        match value {
            SynthesisModel::Triposg => Self::Triposg,
            SynthesisModel::Trellis => Self::Trellis,
            SynthesisModel::Triposplat => Self::Triposplat,
        }
    }
}

impl From<InferenceBackend> for burn_synth::InferenceBackend {
    fn from(value: InferenceBackend) -> Self {
        match value {
            InferenceBackend::Cpu => Self::Cpu,
            InferenceBackend::Wgpu => Self::Wgpu,
            InferenceBackend::Cuda => Self::Cuda,
        }
    }
}

impl From<TrellisQuality> for burn_synth::TrellisQuality {
    fn from(value: TrellisQuality) -> Self {
        match value {
            TrellisQuality::Low => Self::Low,
            TrellisQuality::Medium => Self::Medium,
            TrellisQuality::High => Self::High,
        }
    }
}

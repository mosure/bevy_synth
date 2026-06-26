#![recursion_limit = "256"]

mod scene_layout;

use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use bevy_synth_runtime::cache::{CachedAssetAabb, CachedMeshMetadata, MeshCache};
use bevy_synth_runtime::{
    SynthMesh as CachedSynthMesh, SynthMeshMaterial as CachedSynthMeshMaterial,
    SynthMeshPbrTextures as CachedSynthMeshPbrTextures, SynthMeshTexture as CachedSynthMeshTexture,
    TripoMesh as CachedTripoMesh,
};
use burn::prelude::{Backend, Tensor};
use burn_depth::{
    CameraIntrinsics, DepthCheckpointSource, DepthLoadConfig, DepthLoadEvent, DepthLoadStage,
    DepthModelKind, DepthPipeline, DepthPrecision, DepthRuntimeConfig, ImageBoundingBox,
    backproject_depth, depth_at_bbox_contact_region, estimate_floor_plane, pixel_to_ray,
};
use burn_locate_anything::{
    DecodeMode, Detection, DetectionQuery, LOCATE_ANYTHING_SAFE_IN_TOKEN_LIMIT,
    LocateAnythingDetector, LocateAnythingRuntime, LocateAnythingRuntimeBackend,
    LocateAnythingRuntimeConfig,
};
use burn_synth::{
    AssetBatchItem, AssetBatchRequest, ForegroundRequest, ImageSource, Mesh, ModelSelection,
    RuntimeBatchPolicy, RuntimeConfig, SynthRuntime, SynthesisAsset, mesh_quality_failures,
    mesh_quality_metrics, write_glb_mesh,
};
use burn_synth_scene::{
    DEFAULT_SCENE_RECONSTRUCTION_IMAGE_SCORE, DepthEvidenceRef, EstimatedFloorPlane,
    GroundedSceneLayout, GroundedScenePlacement, ObjectDepthStats, ObjectGroundingEvidence,
    ObjectImageGenerationPolicy, OpenAiProviderConfig, OpenAiSceneProvider, SceneAiProvider,
    SceneAssetAabb, SceneAssetBinding, SceneAssetFrame, SceneBsnRequest, SceneBuildConfig,
    SceneGroundingEvidence, SceneObjectInstanceSpec, SceneObjectManifest, SceneObjectSpec,
    ScenePipeline, SceneQualityProfile, SceneReasoningRequest, SceneResult,
    grounded_scene_layout_for_manifest, grounded_scene_layout_with_evidence,
    manifest_grounding_evidence, parse_scene_bsn, scene_plan_to_mcp_commands, write_json_file,
};
use clap::{ArgAction, Args, Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Value, json};

use scene_layout::{
    SceneComposeArgs, SceneComposePlan, SceneValidateArgs, compose_scene_layout,
    validate_scene_layout,
};

const DEFAULT_PROTOCOL_VERSION: &str = "2025-06-18";
const DEFAULT_SCENE_TRELLIS_TARGET_FACES: usize = 80_000;
const DEFAULT_SCENE_TRELLIS_PBR_TEXTURE_SIZE: usize = 512;
static NEXT_SCENE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

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

impl From<SceneDepthPrecision> for DepthPrecision {
    fn from(value: SceneDepthPrecision) -> Self {
        match value {
            SceneDepthPrecision::F32 => DepthPrecision::F32,
            SceneDepthPrecision::F16 => DepthPrecision::F16,
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
pub enum LocateAnythingBackend {
    PythonReference,
    BurnNative,
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

    #[arg(long)]
    pub weights_root: Option<PathBuf>,

    #[arg(long)]
    pub trellis_weights_root: Option<PathBuf>,

    #[arg(long)]
    pub trellis_image_large_root: Option<PathBuf>,

    #[arg(long)]
    pub trellis_python_bin: Option<PathBuf>,

    #[arg(long)]
    pub trellis_bridge_script: Option<PathBuf>,

    #[arg(long, value_enum, default_value_t = TrellisQuality::Medium)]
    pub trellis_quality: TrellisQuality,

    /// Enable TRELLIS PBR UV/material texture baking through the Rust/Burn o_voxel export path.
    #[arg(long)]
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

    /// Python executable for the explicit upstream LocateAnything reference locator.
    #[arg(long)]
    pub locate_anything_python_bin: Option<PathBuf>,

    /// Local LocateAnything HF snapshot root used by the explicit reference locator.
    #[arg(long, default_value = "assets/models/LocateAnything-3B")]
    pub locate_anything_model_root: PathBuf,

    /// Reference runner used by --locator locate-anything.
    #[arg(
        long,
        default_value = "crates/burn_locate_anything/python/locate_anything_reference.py"
    )]
    pub locate_anything_reference_script: PathBuf,

    /// Device passed to the explicit upstream LocateAnything reference locator.
    #[arg(long, default_value = "cuda")]
    pub locate_anything_device: String,

    /// Image token limit for the explicit LocateAnything reference locator.
    #[arg(long, default_value_t = LOCATE_ANYTHING_SAFE_IN_TOKEN_LIMIT as usize)]
    pub locate_anything_in_token_limit: usize,

    /// LocateAnything execution backend used by scene-ground when --locator locate-anything.
    #[arg(long, value_enum, default_value_t = LocateAnythingBackend::PythonReference)]
    pub locate_anything_backend: LocateAnythingBackend,

    #[command(subcommand)]
    command: Option<ServerCommand>,
}

#[derive(Subcommand, Debug, Clone)]
#[allow(clippy::enum_variant_names)]
enum ServerCommand {
    /// Run the full source-scene image -> object images -> assets -> grounded BSN pipeline once.
    SceneBuild(SceneBuildCliArgs),
    /// Recompute scene composition from saved assets and grounding evidence without regenerating assets.
    SceneGround(SceneGroundCliArgs),
    /// Replay render-capture-feedback using existing scene-build artifacts.
    SceneFeedbackReplay(SceneFeedbackReplayCliArgs),
}

#[derive(Args, Debug, Clone)]
struct SceneBuildCliArgs {
    /// Source scene image path.
    #[arg(long, visible_alias = "scene")]
    pub source_scene_path: PathBuf,

    /// Example/reference isolated-object image. Defaults to --scene-object-reference-image.
    #[arg(long, visible_alias = "object-reference")]
    pub object_reference_image_path: Option<PathBuf>,

    /// Output directory for generated images, assets, BSN, metrics, and response JSON.
    #[arg(long)]
    pub output_dir: Option<PathBuf>,

    /// Number of generated object-image candidates in the default budget.
    #[arg(long, visible_alias = "candidates")]
    pub candidate_count: Option<usize>,

    /// Maximum guarded image-generation attempts per object.
    #[arg(long)]
    pub candidate_retry_attempts: Option<usize>,

    /// Image candidates requested per retry attempt.
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
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    pub feedback: bool,

    /// Maximum render-capture-feedback iterations.
    #[arg(long, default_value_t = 3)]
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
}

#[derive(Args, Debug, Clone)]
struct SceneGroundCliArgs {
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

    /// Depth provider identifier for run metadata.
    #[arg(long, value_enum, default_value_t = SceneDepthProvider::DepthPro)]
    pub depth_provider: SceneDepthProvider,

    /// Locator provider identifier for run metadata.
    #[arg(long, value_enum, default_value_t = SceneLocatorProvider::Manifest)]
    pub locator: SceneLocatorProvider,

    /// Override the server LocateAnything backend for this scene-ground run.
    #[arg(long, value_enum)]
    pub locate_anything_backend: Option<LocateAnythingBackend>,

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
    #[arg(long, default_value_t = 3)]
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
}

#[derive(Args, Debug, Clone)]
struct SceneFeedbackReplayCliArgs {
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
    #[arg(long, default_value_t = 3)]
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
}

#[derive(Clone, Debug)]
pub struct ServerConfig {
    pub default_rmbg_model: ForegroundModel,
    pub default_synthesis_models: Vec<SynthesisModel>,
    pub default_backend: InferenceBackend,
    pub weights_root: Option<PathBuf>,
    pub trellis_weights_root: Option<PathBuf>,
    pub trellis_image_large_root: Option<PathBuf>,
    pub trellis_python_bin: Option<PathBuf>,
    pub trellis_bridge_script: Option<PathBuf>,
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
    pub locate_anything_python_bin: Option<PathBuf>,
    pub locate_anything_model_root: PathBuf,
    pub locate_anything_reference_script: PathBuf,
    pub locate_anything_device: String,
    pub locate_anything_in_token_limit: usize,
    pub locate_anything_backend: LocateAnythingBackend,
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
            trellis_python_bin: args.trellis_python_bin,
            trellis_bridge_script: args.trellis_bridge_script,
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
            locate_anything_python_bin: args.locate_anything_python_bin,
            locate_anything_model_root: args.locate_anything_model_root,
            locate_anything_reference_script: args.locate_anything_reference_script,
            locate_anything_device: args.locate_anything_device,
            locate_anything_in_token_limit: args.locate_anything_in_token_limit.max(1),
            locate_anything_backend: args.locate_anything_backend,
        }
    }

    fn runtime_config(&self) -> RuntimeConfig {
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
            trellis_python_bin: self.trellis_python_bin.clone(),
            trellis_bridge_script: self.trellis_bridge_script.clone(),
            trellis_quality: self.trellis_quality.into(),
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
}

pub fn run_from_args(args: ServerArgs) -> Result<(), String> {
    let command = args.command.clone();
    let config = ServerConfig::from_args(args);
    match command {
        Some(ServerCommand::SceneBuild(args)) => run_scene_build_command(config, args),
        Some(ServerCommand::SceneGround(args)) => run_scene_ground_command(config, args),
        Some(ServerCommand::SceneFeedbackReplay(args)) => {
            run_scene_feedback_replay_command(config, args)
        }
        None => run_stdio_server(config),
    }
}

fn run_scene_build_command(config: ServerConfig, args: SceneBuildCliArgs) -> Result<(), String> {
    let mut server = McpServer::new(config);
    let response = server.call_scene_build_from_image(SceneBuildFromImageArgs {
        source_scene_path: args.source_scene_path,
        object_reference_image_path: args.object_reference_image_path,
        output_dir: args.output_dir,
        candidate_count: args.candidate_count,
        candidate_retry_attempts: args.candidate_retry_attempts,
        candidate_batch_size: args.candidate_batch_size,
        min_reconstruction_score: args.min_reconstruction_score,
        quality_profile: args.quality_profile,
        allow_catalog_reuse: args.allow_catalog_reuse,
        lift_assets: args.lift_assets,
        target_faces: args.target_faces,
        batch_size: args.batch_size.filter(|value| *value > 0),
        batch_vram_mb: args.batch_vram_mb,
        trellis_pbr: Some(args.trellis_pbr),
        trellis_pbr_texture_size: args.trellis_pbr_texture_size,
        promote_to_catalog: args.promote_to_catalog,
        write_artifacts: args.write_artifacts,
        apply: args.apply,
        clear_existing: args.clear_existing,
        feedback: args.feedback,
        feedback_iters: args.feedback_iters,
        feedback_keep_viewer: args.feedback_keep_viewer,
        feedback_capture_dir: args.feedback_capture_dir,
        feedback_threshold_profile: args.feedback_threshold_profile,
    })?;
    println!(
        "{}",
        serde_json::to_string_pretty(&response)
            .map_err(|err| format!("serialize scene-build response: {err}"))?
    );
    Ok(())
}

fn run_scene_ground_command(config: ServerConfig, args: SceneGroundCliArgs) -> Result<(), String> {
    let mut server = McpServer::new(config);
    let response = server.call_scene_ground(SceneGroundToolArgs {
        source_scene_path: args.source_scene_path,
        manifest: read_json_path(&args.manifest)?,
        asset_bindings: read_json_path(&args.asset_bindings)?,
        grounding_evidence: args
            .grounding_evidence
            .as_ref()
            .map(|path| read_json_path::<SceneGroundingEvidence>(path.as_path()))
            .transpose()?,
        output_dir: args.output_dir,
        composition_mode: args.composition_mode,
        depth_provider: args.depth_provider,
        locator: args.locator,
        locate_anything_backend: args.locate_anything_backend,
        clear_existing: args.clear_existing,
        apply: args.apply,
        feedback: args.feedback,
        feedback_iters: args.feedback_iters,
        feedback_keep_viewer: args.feedback_keep_viewer,
        feedback_capture_dir: args.feedback_capture_dir,
        feedback_threshold_profile: args.feedback_threshold_profile,
    })?;
    println!(
        "{}",
        serde_json::to_string_pretty(&response)
            .map_err(|err| format!("serialize scene-ground response: {err}"))?
    );
    Ok(())
}

fn run_scene_feedback_replay_command(
    config: ServerConfig,
    args: SceneFeedbackReplayCliArgs,
) -> Result<(), String> {
    let mut server = McpServer::new(config);
    let manifest_path = args
        .manifest_path
        .unwrap_or_else(|| args.output_dir.join("manifest.json"));
    let asset_bindings_path = args
        .asset_bindings_path
        .unwrap_or_else(|| args.output_dir.join("asset_bindings.json"));
    let grounded_layout_path = args
        .grounded_layout_path
        .unwrap_or_else(|| args.output_dir.join("grounded_layout.json"));
    let commands_path = args
        .commands_path
        .unwrap_or_else(|| args.output_dir.join("commands.json"));
    let capture_dir = args
        .feedback_capture_dir
        .unwrap_or_else(|| args.output_dir.join("iterations_replay"));
    let manifest = read_json_path::<SceneObjectManifest>(&manifest_path)?;
    let asset_bindings = read_json_path::<Vec<SceneAssetBinding>>(&asset_bindings_path)?;
    let grounded_layout = read_json_path::<GroundedSceneLayout>(&grounded_layout_path)?;
    let commands = if args.rebuild_commands_from_grounded_layout {
        let plan = parse_scene_bsn(&grounded_layout.bsn, &asset_bindings)
            .map_err(|err| err.to_string())?;
        scene_commands_with_cache_reload(
            scene_plan_to_mcp_commands(&plan, &asset_bindings, true)
                .map_err(|err| err.to_string())?,
        )
    } else {
        read_json_path::<Vec<Value>>(&commands_path)?
    };
    let response = server.run_scene_feedback(
        &args.output_dir,
        &manifest,
        &asset_bindings,
        &grounded_layout,
        commands,
        SceneFeedbackOptions {
            max_iters: args.feedback_iters,
            keep_viewer: args.feedback_keep_viewer,
            capture_dir: Some(capture_dir),
            threshold_profile: args.feedback_threshold_profile,
        },
    )?;
    println!(
        "{}",
        serde_json::to_string_pretty(&response)
            .map_err(|err| format!("serialize scene-feedback-replay response: {err}"))?
    );
    Ok(())
}

fn read_json_path<T: DeserializeOwned>(path: &Path) -> Result<T, String> {
    let bytes =
        fs::read(path).map_err(|err| format!("failed to read JSON {}: {err}", path.display()))?;
    serde_json::from_slice(&bytes)
        .map_err(|err| format!("failed to parse JSON {}: {err}", path.display()))
}

pub fn run_stdio_server(config: ServerConfig) -> Result<(), String> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut reader = BufReader::new(stdin.lock());
    let mut writer = BufWriter::new(stdout.lock());
    let mut server = McpServer::new(config);

    while let Some(message) = read_framed_json(&mut reader).map_err(|err| err.to_string())? {
        let response = server.handle_message(message)?;
        if let Some(response) = response {
            write_framed_json(&mut writer, &response).map_err(|err| err.to_string())?;
        }
        if server.should_exit {
            break;
        }
    }

    Ok(())
}

struct McpServer {
    config: ServerConfig,
    runtime: SynthRuntime,
    locate_anything_burn_native_runtime: Option<CachedLocateAnythingRuntime>,
    should_exit: bool,
}

struct CachedLocateAnythingRuntime {
    key: LocateAnythingBurnNativeCacheKey,
    runtime: LocateAnythingRuntime,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct LocateAnythingBurnNativeCacheKey {
    model_root: PathBuf,
    decode_mode: DecodeMode,
    max_new_tokens: usize,
    in_token_limit: u32,
    repetition_penalty_bits: u32,
    top_p_bits: Option<u32>,
    top_k: Option<usize>,
}

impl LocateAnythingBurnNativeCacheKey {
    fn from_config(config: &LocateAnythingRuntimeConfig) -> Self {
        Self {
            model_root: config.model_root.clone(),
            decode_mode: config.decode_mode,
            max_new_tokens: config.max_new_tokens,
            in_token_limit: config.in_token_limit,
            repetition_penalty_bits: config.repetition_penalty.to_bits(),
            top_p_bits: config.top_p.map(f32::to_bits),
            top_k: config.top_k,
        }
    }
}

struct NoopSceneProvider;

impl SceneAiProvider for NoopSceneProvider {
    fn plan_objects(&self, _request: &SceneReasoningRequest) -> SceneResult<SceneObjectManifest> {
        Err(burn_synth_scene::SceneError::Provider(
            "offline schema preparation cannot plan objects".to_string(),
        ))
    }

    fn generate_object_images(
        &self,
        _request: &burn_synth_scene::ObjectImageRequest,
    ) -> SceneResult<Vec<Vec<u8>>> {
        Err(burn_synth_scene::SceneError::Provider(
            "offline schema preparation cannot generate images".to_string(),
        ))
    }

    fn plan_scene_bsn(&self, _request: &SceneBsnRequest) -> SceneResult<String> {
        Err(burn_synth_scene::SceneError::Provider(
            "offline schema preparation cannot plan BSN".to_string(),
        ))
    }
}

impl McpServer {
    fn new(config: ServerConfig) -> Self {
        let runtime = SynthRuntime::new(config.runtime_config());
        Self {
            config,
            runtime,
            locate_anything_burn_native_runtime: None,
            should_exit: false,
        }
    }

    fn handle_message(&mut self, message: Value) -> Result<Option<Value>, String> {
        let request: RpcRequest = serde_json::from_value(message)
            .map_err(|err| format!("invalid JSON-RPC request: {err}"))?;
        self.handle_request(request)
    }

    fn handle_request(&mut self, request: RpcRequest) -> Result<Option<Value>, String> {
        match request.method.as_str() {
            "initialize" => {
                let params: InitializeParams = request
                    .params
                    .map(serde_json::from_value)
                    .transpose()
                    .map_err(|err| format!("invalid initialize params: {err}"))?
                    .unwrap_or_default();
                let protocol_version = params
                    .protocol_version
                    .unwrap_or_else(|| DEFAULT_PROTOCOL_VERSION.to_string());
                let result = json!({
                    "protocolVersion": protocol_version,
                    "serverInfo": {
                        "name": "burn_synth_mcp",
                        "version": env!("CARGO_PKG_VERSION"),
                    },
                    "capabilities": {
                        "tools": {
                            "listChanged": false
                        }
                    }
                });
                Ok(Some(success_response(request.id, result)))
            }
            "notifications/initialized" => Ok(None),
            "tools/list" => Ok(Some(success_response(
                request.id,
                json!({ "tools": tool_defs() }),
            ))),
            "tools/call" => {
                let params: ToolsCallParams = request
                    .params
                    .ok_or_else(|| "missing tools/call params".to_string())
                    .and_then(|value| {
                        serde_json::from_value(value)
                            .map_err(|err| format!("invalid tools/call params: {err}"))
                    })?;
                let result = self.dispatch_tool_call(params);
                Ok(Some(success_response(request.id, result)))
            }
            "shutdown" => Ok(Some(success_response(request.id, Value::Null))),
            "exit" => {
                self.should_exit = true;
                Ok(None)
            }
            _ => {
                if request.id.is_none() {
                    return Ok(None);
                }
                Ok(Some(error_response(
                    request.id,
                    -32601,
                    format!("method '{}' not found", request.method),
                )))
            }
        }
    }

    fn dispatch_tool_call(&mut self, params: ToolsCallParams) -> Value {
        match params.name.as_str() {
            "image_to_foreground" => {
                let args: Result<ForegroundToolArgs, _> = serde_json::from_value(params.arguments);
                match args {
                    Ok(args) => match self.call_image_to_foreground(args) {
                        Ok(value) => success_tool_result(value),
                        Err(err) => error_tool_result(err),
                    },
                    Err(err) => error_tool_result(format!(
                        "invalid arguments for image_to_foreground: {err}"
                    )),
                }
            }
            "image_to_mesh" => {
                let args: Result<MeshToolArgs, _> = serde_json::from_value(params.arguments);
                match args {
                    Ok(args) => match self.call_image_to_mesh(args) {
                        Ok(value) => success_tool_result(value),
                        Err(err) => error_tool_result(err),
                    },
                    Err(err) => {
                        error_tool_result(format!("invalid arguments for image_to_mesh: {err}"))
                    }
                }
            }
            "image_to_splat" => {
                let args: Result<SplatToolArgs, _> = serde_json::from_value(params.arguments);
                match args {
                    Ok(args) => match self.call_image_to_splat(args) {
                        Ok(value) => success_tool_result(value),
                        Err(err) => error_tool_result(err),
                    },
                    Err(err) => {
                        error_tool_result(format!("invalid arguments for image_to_splat: {err}"))
                    }
                }
            }
            "images_to_assets" => {
                let args: Result<ImagesToAssetsToolArgs, _> =
                    serde_json::from_value(params.arguments);
                match args {
                    Ok(args) => match self.call_images_to_assets(args) {
                        Ok(value) => success_tool_result(value),
                        Err(err) => error_tool_result(err),
                    },
                    Err(err) => {
                        error_tool_result(format!("invalid arguments for images_to_assets: {err}"))
                    }
                }
            }
            "scene_prepare_build" => {
                let args: Result<ScenePrepareBuildArgs, _> =
                    serde_json::from_value(params.arguments);
                match args {
                    Ok(args) => match self.call_scene_prepare_build(args) {
                        Ok(value) => success_tool_result(value),
                        Err(err) => error_tool_result(err),
                    },
                    Err(err) => error_tool_result(format!(
                        "invalid arguments for scene_prepare_build: {err}"
                    )),
                }
            }
            "scene_plan_objects" => {
                let args: Result<ScenePrepareBuildArgs, _> =
                    serde_json::from_value(params.arguments);
                match args {
                    Ok(args) => match self.call_scene_plan_objects(args) {
                        Ok(value) => success_tool_result(value),
                        Err(err) => error_tool_result(err),
                    },
                    Err(err) => error_tool_result(format!(
                        "invalid arguments for scene_plan_objects: {err}"
                    )),
                }
            }
            "scene_generate_object_images" => {
                let args: Result<SceneGenerateObjectImagesArgs, _> =
                    serde_json::from_value(params.arguments);
                match args {
                    Ok(args) => match self.call_scene_generate_object_images(args) {
                        Ok(value) => success_tool_result(value),
                        Err(err) => error_tool_result(err),
                    },
                    Err(err) => error_tool_result(format!(
                        "invalid arguments for scene_generate_object_images: {err}"
                    )),
                }
            }
            "scene_build_from_image" => {
                let args: Result<SceneBuildFromImageArgs, _> =
                    serde_json::from_value(params.arguments);
                match args {
                    Ok(args) => match self.call_scene_build_from_image(args) {
                        Ok(value) => success_tool_result(value),
                        Err(err) => error_tool_result(err),
                    },
                    Err(err) => error_tool_result(format!(
                        "invalid arguments for scene_build_from_image: {err}"
                    )),
                }
            }
            "scene_ground" => {
                let args: Result<SceneGroundToolArgs, _> = serde_json::from_value(params.arguments);
                match args {
                    Ok(args) => match self.call_scene_ground(args) {
                        Ok(value) => success_tool_result(value),
                        Err(err) => error_tool_result(err),
                    },
                    Err(err) => {
                        error_tool_result(format!("invalid arguments for scene_ground: {err}"))
                    }
                }
            }
            "scene_plan_bsn" => {
                let args: Result<ScenePlanBsnArgs, _> = serde_json::from_value(params.arguments);
                match args {
                    Ok(args) => match self.call_scene_plan_bsn(args) {
                        Ok(value) => success_tool_result(value),
                        Err(err) => error_tool_result(err),
                    },
                    Err(err) => {
                        error_tool_result(format!("invalid arguments for scene_plan_bsn: {err}"))
                    }
                }
            }
            "scene_apply_bsn" => {
                let args: Result<SceneApplyBsnArgs, _> = serde_json::from_value(params.arguments);
                match args {
                    Ok(args) => match self.call_scene_apply_bsn(args) {
                        Ok(value) => success_tool_result(value),
                        Err(err) => error_tool_result(err),
                    },
                    Err(err) => {
                        error_tool_result(format!("invalid arguments for scene_apply_bsn: {err}"))
                    }
                }
            }
            "scene_status" => match self.call_scene_status() {
                Ok(value) => success_tool_result(value),
                Err(err) => error_tool_result(err),
            },
            "scene_project_status" => match self.call_scene_project_status() {
                Ok(value) => success_tool_result(value),
                Err(err) => error_tool_result(err),
            },
            "scene_list_assets" => match self.call_scene_list_assets() {
                Ok(value) => success_tool_result(value),
                Err(err) => error_tool_result(err),
            },
            "scene_spawn_cached" => {
                let args: Result<SceneSpawnCachedArgs, _> =
                    serde_json::from_value(params.arguments);
                match args {
                    Ok(args) => match self.call_scene_spawn_cached(args) {
                        Ok(value) => success_tool_result(value),
                        Err(err) => error_tool_result(err),
                    },
                    Err(err) => error_tool_result(format!(
                        "invalid arguments for scene_spawn_cached: {err}"
                    )),
                }
            }
            "scene_spawn_path" => {
                let args: Result<SceneSpawnPathArgs, _> = serde_json::from_value(params.arguments);
                match args {
                    Ok(args) => match self.call_scene_spawn_path(args) {
                        Ok(value) => success_tool_result(value),
                        Err(err) => error_tool_result(err),
                    },
                    Err(err) => {
                        error_tool_result(format!("invalid arguments for scene_spawn_path: {err}"))
                    }
                }
            }
            "scene_delete" => {
                let args: Result<SceneDeleteArgs, _> = serde_json::from_value(params.arguments);
                match args {
                    Ok(args) => match self.call_scene_delete(args) {
                        Ok(value) => success_tool_result(value),
                        Err(err) => error_tool_result(err),
                    },
                    Err(err) => {
                        error_tool_result(format!("invalid arguments for scene_delete: {err}"))
                    }
                }
            }
            "scene_clear" => match self.call_scene_clear() {
                Ok(value) => success_tool_result(value),
                Err(err) => error_tool_result(err),
            },
            "scene_set_camera" => {
                let args: Result<SceneSetCameraArgs, _> = serde_json::from_value(params.arguments);
                match args {
                    Ok(args) => match self.call_scene_set_camera(args) {
                        Ok(value) => success_tool_result(value),
                        Err(err) => error_tool_result(err),
                    },
                    Err(err) => {
                        error_tool_result(format!("invalid arguments for scene_set_camera: {err}"))
                    }
                }
            }
            "scene_save" => match self.send_scene_commands(vec![json!({ "type": "save_cache" })]) {
                Ok(value) => success_tool_result(value),
                Err(err) => error_tool_result(err),
            },
            "scene_capture" => {
                let args: Result<SceneCaptureArgs, _> = serde_json::from_value(params.arguments);
                match args {
                    Ok(args) => match self.call_scene_capture(args) {
                        Ok(value) => success_tool_result(value),
                        Err(err) => error_tool_result(err),
                    },
                    Err(err) => {
                        error_tool_result(format!("invalid arguments for scene_capture: {err}"))
                    }
                }
            }
            "scene_compose_assets" => {
                let args: Result<SceneComposeArgs, _> = serde_json::from_value(params.arguments);
                match args {
                    Ok(args) => match self.call_scene_compose_assets(args) {
                        Ok(value) => success_tool_result(value),
                        Err(err) => error_tool_result(err),
                    },
                    Err(err) => error_tool_result(format!(
                        "invalid arguments for scene_compose_assets: {err}"
                    )),
                }
            }
            "scene_validate_layout" => {
                let args: Result<SceneValidateArgs, _> = serde_json::from_value(params.arguments);
                match args {
                    Ok(args) => match self.call_scene_validate_layout(args) {
                        Ok(value) => success_tool_result(value),
                        Err(err) => error_tool_result(err),
                    },
                    Err(err) => error_tool_result(format!(
                        "invalid arguments for scene_validate_layout: {err}"
                    )),
                }
            }
            other => error_tool_result(format!("unknown tool '{other}'")),
        }
    }

    fn call_image_to_foreground(&mut self, args: ForegroundToolArgs) -> Result<Value, String> {
        let input_path = args.input_image_path;
        if !input_path.exists() {
            return Err(format!(
                "input image does not exist: {}",
                input_path.display()
            ));
        }
        let output_path = args
            .output_image_path
            .unwrap_or_else(|| default_output_path(&input_path, "_foreground", "png"));
        ensure_parent_dir(&output_path).map_err(|err| err.to_string())?;

        let selected_model = args.rmbg_model.unwrap_or(self.config.default_rmbg_model);
        let dry_run = args.dry_run;

        let (width, height) = if dry_run {
            let passthrough = image::open(&input_path)
                .map_err(|err| {
                    format!("failed to open input image {}: {err}", input_path.display())
                })?
                .to_rgba8();
            let dims = passthrough.dimensions();
            passthrough.save(&output_path).map_err(|err| {
                format!(
                    "failed to save foreground image {}: {err}",
                    output_path.display()
                )
            })?;
            dims
        } else {
            let output = self
                .runtime
                .extract_foreground(ForegroundRequest {
                    image: ImageSource::from_path(input_path.clone()),
                    model: Some(selected_model.into()),
                })
                .map_err(|err| err.to_string())?;
            let dims = (output.width, output.height);
            output.image.save(&output_path).map_err(|err| {
                format!(
                    "failed to save foreground image {}: {err}",
                    output_path.display()
                )
            })?;
            dims
        };

        Ok(json!({
            "tool": "image_to_foreground",
            "input_image_path": input_path.display().to_string(),
            "output_image_path": output_path.display().to_string(),
            "width": width,
            "height": height,
            "rmbg_model": selected_model.as_str(),
            "dry_run": dry_run,
        }))
    }

    fn call_image_to_mesh(&mut self, args: MeshToolArgs) -> Result<Value, String> {
        let input_path = args.input_image_path;
        if !input_path.exists() {
            return Err(format!(
                "input image does not exist: {}",
                input_path.display()
            ));
        }
        if let Some(output_format) = args.output_format
            && !matches!(output_format, MeshOutputFormat::Glb)
        {
            return Err(format!(
                "only glb output is supported; requested {}",
                output_format.as_str()
            ));
        }
        let assets = self.call_images_to_assets(ImagesToAssetsToolArgs {
            input_image_paths: vec![input_path],
            output_dir: None,
            output_paths: args.output_mesh_path.map(|path| vec![path]),
            output_format: Some(AssetOutputFormat::Glb),
            rmbg_model: args.rmbg_model,
            synthesis_models: args.synthesis_models,
            backend: args.backend,
            target_faces: args.target_faces,
            batch_size: Some(1),
            batch_vram_mb: None,
            trellis_pbr: None,
            trellis_pbr_texture_size: None,
            promote_to_catalog: false,
            dry_run: args.dry_run,
        })?;
        let item = assets["items"]
            .as_array()
            .and_then(|items| items.first())
            .cloned()
            .ok_or_else(|| "image_to_mesh produced no asset item".to_string())?;
        Ok(json!({
            "tool": "image_to_mesh",
            "input_image_path": item["input_image_path"].clone(),
            "output_mesh_path": item["output_path"].clone(),
            "output_format": "glb",
            "vertices": item["vertices"].clone(),
            "faces": item["faces"].clone(),
            "local_aabb": item["local_aabb"].clone(),
            "target_faces": item["target_faces"].clone(),
            "material": item["material"].clone(),
            "rmbg_model": assets["rmbg_model"].clone(),
            "synthesis_models": assets["synthesis_models"].clone(),
            "backend": assets["backend"].clone(),
            "dry_run": assets["dry_run"].clone(),
        }))
    }

    fn call_image_to_splat(&mut self, args: SplatToolArgs) -> Result<Value, String> {
        let assets = self.call_images_to_assets(ImagesToAssetsToolArgs {
            input_image_paths: vec![args.input_image_path],
            output_dir: None,
            output_paths: args.output_splat_path.map(|path| vec![path]),
            output_format: args.output_format.or(Some(AssetOutputFormat::Splat)),
            rmbg_model: args.rmbg_model,
            synthesis_models: Some(vec![SynthesisModel::Triposplat]),
            backend: args.backend,
            target_faces: None,
            batch_size: Some(1),
            batch_vram_mb: None,
            trellis_pbr: None,
            trellis_pbr_texture_size: None,
            promote_to_catalog: false,
            dry_run: args.dry_run,
        })?;
        let item = assets["items"]
            .as_array()
            .and_then(|items| items.first())
            .cloned()
            .ok_or_else(|| "image_to_splat produced no asset item".to_string())?;
        Ok(json!({
            "tool": "image_to_splat",
            "input_image_path": item["input_image_path"].clone(),
            "output_splat_path": item["output_path"].clone(),
            "output_format": item["output_format"].clone(),
            "gaussians": item["gaussians"].clone(),
            "rmbg_model": assets["rmbg_model"].clone(),
            "synthesis_models": assets["synthesis_models"].clone(),
            "backend": assets["backend"].clone(),
            "dry_run": assets["dry_run"].clone(),
        }))
    }

    fn call_images_to_assets(&mut self, args: ImagesToAssetsToolArgs) -> Result<Value, String> {
        if args.input_image_paths.is_empty() {
            return Err("input_image_paths must not be empty".to_string());
        }
        for input in &args.input_image_paths {
            if !input.exists() {
                return Err(format!("input image does not exist: {}", input.display()));
            }
        }
        if let Some(output_paths) = args.output_paths.as_ref()
            && output_paths.len() != args.input_image_paths.len()
        {
            return Err(format!(
                "output_paths length ({}) must match input_image_paths length ({})",
                output_paths.len(),
                args.input_image_paths.len()
            ));
        }

        let selected_rmbg = args.rmbg_model.unwrap_or(self.config.default_rmbg_model);
        let selected_backend = args.backend.unwrap_or(self.config.default_backend);
        let selected_synthesis_models = args
            .synthesis_models
            .map(sanitize_synthesis_models)
            .unwrap_or_else(|| self.config.default_synthesis_models.clone());
        let policy = RuntimeBatchPolicy {
            max_items: args.batch_size.or(self.config.batch_size),
            vram_budget_mb: args.batch_vram_mb.or(self.config.batch_vram_mb),
            ..RuntimeBatchPolicy::default()
        };
        let previous_trellis_pbr_enabled = self.runtime.config().trellis_pbr_enabled;
        let previous_trellis_pbr_texture_size = self.runtime.config().trellis_pbr_texture_size;
        let previous_target_faces = self.runtime.config().target_faces;
        let effective_trellis_pbr_enabled =
            args.trellis_pbr.unwrap_or(previous_trellis_pbr_enabled);
        let effective_trellis_pbr_texture_size = args
            .trellis_pbr_texture_size
            .or(previous_trellis_pbr_texture_size);
        let effective_target_faces = match args.target_faces {
            Some(0) => None,
            Some(value) => Some(value),
            None => previous_target_faces,
        };
        {
            let config = self.runtime.config_mut();
            config.trellis_pbr_enabled = effective_trellis_pbr_enabled;
            config.trellis_pbr_texture_size = effective_trellis_pbr_texture_size;
            config.target_faces = effective_target_faces;
        }

        let batch_result = self.runtime.synthesize_assets_batch(AssetBatchRequest {
            items: args
                .input_image_paths
                .iter()
                .enumerate()
                .map(|(index, input)| {
                    AssetBatchItem::new(
                        format!("asset_{index}"),
                        ImageSource::from_path(input.clone()),
                    )
                })
                .collect(),
            foreground_model: Some(selected_rmbg.into()),
            synthesis_models: Some(
                selected_synthesis_models
                    .iter()
                    .copied()
                    .map(Into::into)
                    .collect(),
            ),
            backend: Some(selected_backend.into()),
            dry_run: args.dry_run,
            policy,
        });
        {
            let config = self.runtime.config_mut();
            config.trellis_pbr_enabled = previous_trellis_pbr_enabled;
            config.trellis_pbr_texture_size = previous_trellis_pbr_texture_size;
            config.target_faces = previous_target_faces;
        }
        let batch = batch_result.map_err(|err| err.to_string())?;

        let mut items = Vec::with_capacity(batch.items.len());
        let mut catalog_cache = if args.promote_to_catalog {
            Some(self.open_catalog_cache()?)
        } else {
            None
        };
        for batch_item in batch.items {
            let input_path = args
                .input_image_paths
                .get(batch_item.item_index)
                .ok_or_else(|| {
                    format!(
                        "asset batch item index {} out of range for {} input images",
                        batch_item.item_index,
                        args.input_image_paths.len()
                    )
                })?;
            let output = batch_item.output.map_err(|err| err.to_string())?;
            let item = write_asset_output(
                input_path,
                args.output_dir.as_deref(),
                args.output_paths
                    .as_ref()
                    .and_then(|paths| paths.get(batch_item.item_index).cloned()),
                args.output_format.unwrap_or(AssetOutputFormat::Auto),
                output.asset,
                effective_target_faces,
                catalog_cache.as_mut(),
            )?;
            items.push(json!({
                "id": batch_item.id,
                "input_image_path": input_path.display().to_string(),
                "chunk_index": batch_item.chunk_index,
                "item_index": batch_item.item_index,
                "elapsed_ms": batch_item.elapsed_ms,
                "foreground_model": runtime_foreground_model_str(output.foreground_model),
                "synthesis_backend": runtime_synthesis_model_str(output.synthesis_backend),
                "backend": runtime_backend_str(output.backend),
                "output_path": item.output_path.display().to_string(),
                "output_format": item.output_format.as_str(),
                "asset_kind": item.asset_kind,
                "vertices": item.vertices,
                "faces": item.faces,
                "gaussians": item.gaussians,
                "local_aabb": item.local_aabb,
                "target_faces": effective_target_faces,
                "material": item.material,
                "mesh_quality": item.mesh_quality,
                "mesh_quality_failures": item.mesh_quality_failures,
                "cache_key": item.catalog_entry.as_ref().map(|entry| entry.cache_key.clone()),
                "catalog_entry": item.catalog_entry,
            }));
        }

        Ok(json!({
            "tool": "images_to_assets",
            "items": items,
            "stats": {
                "total_items": batch.stats.total_items,
                "chunk_size": batch.stats.chunk_size,
                "chunks": batch.stats.chunks,
                "execution_mode": batch.stats.execution_mode.as_str(),
                "vram_budget_mb": batch.stats.vram_budget_mb,
                "estimated_item_mb": batch.stats.estimated_item_mb,
                "elapsed_ms": batch.stats.elapsed_ms,
            },
            "rmbg_model": selected_rmbg.as_str(),
            "synthesis_models": selected_synthesis_models.iter().map(|m| m.as_str()).collect::<Vec<_>>(),
            "backend": selected_backend.as_str(),
            "trellis_pbr_enabled": effective_trellis_pbr_enabled,
            "trellis_pbr_backend": if effective_trellis_pbr_enabled { Some("rust-ovoxel") } else { None },
            "trellis_pbr_texture_size": effective_trellis_pbr_texture_size,
            "target_faces": effective_target_faces,
            "promote_to_catalog": args.promote_to_catalog,
            "dry_run": args.dry_run,
        }))
    }

    fn call_scene_prepare_build(&self, args: ScenePrepareBuildArgs) -> Result<Value, String> {
        let config = self.scene_build_config(args)?;
        let provider = NoopSceneProvider;
        let mut pipeline = ScenePipeline::new(config, provider);
        let preparation = pipeline
            .prepare_openai_inputs()
            .map_err(|err| err.to_string())?;
        serde_json::to_value(preparation).map_err(|err| err.to_string())
    }

    fn call_scene_plan_objects(&self, args: ScenePrepareBuildArgs) -> Result<Value, String> {
        let config = self.scene_build_config(args)?;
        let provider = self.openai_provider()?;
        let pipeline = ScenePipeline::new(config, provider);
        let manifest = pipeline.plan_objects().map_err(|err| err.to_string())?;
        serde_json::to_value(manifest).map_err(|err| err.to_string())
    }

    fn call_scene_generate_object_images(
        &self,
        args: SceneGenerateObjectImagesArgs,
    ) -> Result<Value, String> {
        let prepare_args = ScenePrepareBuildArgs {
            source_scene_path: args.source_scene_path,
            object_reference_image_path: args.object_reference_image_path,
            output_dir: args.output_dir,
            candidate_count: args.candidate_count,
            quality_profile: args.quality_profile,
            allow_catalog_reuse: false,
        };
        let config = self.scene_build_config(prepare_args)?;
        let provider = self.openai_provider()?;
        let pipeline = ScenePipeline::new(config, provider);
        let requests = pipeline
            .prepare_object_image_requests(&args.manifest)
            .map_err(|err| err.to_string())?;
        let candidates = pipeline
            .generate_object_candidates(&requests)
            .map_err(|err| err.to_string())?;
        Ok(json!({
            "tool": "scene_generate_object_images",
            "requests": requests,
            "candidates": candidates,
        }))
    }

    fn call_scene_build_from_image(
        &mut self,
        args: SceneBuildFromImageArgs,
    ) -> Result<Value, String> {
        let e2e_started = Instant::now();
        let mut stage_report = Vec::new();
        let prepare_args = ScenePrepareBuildArgs {
            source_scene_path: args.source_scene_path,
            object_reference_image_path: args.object_reference_image_path,
            output_dir: args.output_dir,
            candidate_count: args.candidate_count,
            quality_profile: args.quality_profile,
            allow_catalog_reuse: args.allow_catalog_reuse,
        };
        let stage_started = Instant::now();
        let config = self.scene_build_config(prepare_args)?;
        let output_dir = config.output_dir.clone();
        let candidate_policy = ObjectImageGenerationPolicy {
            min_score: args
                .min_reconstruction_score
                .unwrap_or(DEFAULT_SCENE_RECONSTRUCTION_IMAGE_SCORE),
            max_attempts_per_object: args
                .candidate_retry_attempts
                .or(args.candidate_count)
                .unwrap_or(config.candidate_count)
                .max(1),
            candidates_per_attempt: args.candidate_batch_size.unwrap_or(1).max(1),
        };
        let provider = self.openai_provider()?;
        let mut pipeline = ScenePipeline::new(config, provider);
        let preparation = pipeline
            .prepare_openai_inputs()
            .map_err(|err| err.to_string())?;
        record_stage(&mut stage_report, "prepare_openai_inputs", stage_started);
        let stage_started = Instant::now();
        let manifest = pipeline.plan_objects().map_err(|err| err.to_string())?;
        record_stage(&mut stage_report, "plan_objects", stage_started);
        let stage_started = Instant::now();
        let requests = pipeline
            .prepare_object_image_requests(&manifest)
            .map_err(|err| err.to_string())?;
        record_stage(
            &mut stage_report,
            "prepare_object_image_requests",
            stage_started,
        );
        let stage_started = Instant::now();
        let candidate_report = pipeline
            .generate_object_candidates_with_policy(&requests, candidate_policy)
            .map_err(|err| err.to_string())?;
        record_stage(
            &mut stage_report,
            "generate_object_candidates",
            stage_started,
        );
        let selected = if candidate_report.rejected_objects.is_empty() {
            candidate_report.selected_candidates.clone()
        } else {
            Vec::new()
        };
        let selected_values = selected_candidates_to_values(&selected);

        let mut response = json!({
            "tool": "scene_build_from_image",
            "preparation": preparation,
            "provider_metadata": pipeline.provider_metadata(),
            "manifest": manifest,
            "object_image_requests": requests,
            "candidate_generation": candidate_report.clone(),
            "candidates": candidate_report.candidates.clone(),
            "selected_candidates": selected_values,
            "lift_assets": args.lift_assets,
        });
        if !candidate_report.rejected_objects.is_empty() {
            response["stage_report"] = json!(stage_report);
            response["e2e_summary"] = scene_build_summary(&response, e2e_started.elapsed());
            if args.write_artifacts {
                write_scene_build_artifacts(&output_dir, &response)?;
            }
            let message = candidate_report
                .rejected_objects
                .first()
                .map(|rejection| rejection.message.clone())
                .unwrap_or_else(|| "scene candidate generation failed guardrails".to_string());
            return Err(message);
        }
        if !args.lift_assets {
            response["stage_report"] = json!(stage_report);
            response["e2e_summary"] = scene_build_summary(&response, e2e_started.elapsed());
            if args.write_artifacts {
                write_scene_build_artifacts(&output_dir, &response)?;
            }
            return Ok(response);
        }

        let input_image_paths = selected
            .iter()
            .map(|candidate| PathBuf::from(&candidate.image_path))
            .collect::<Vec<_>>();
        let stage_started = Instant::now();
        let asset_outputs = self.call_images_to_assets(ImagesToAssetsToolArgs {
            input_image_paths,
            output_dir: Some(output_dir.join("assets")),
            output_paths: None,
            output_format: Some(AssetOutputFormat::Glb),
            rmbg_model: Some(ForegroundModel::Rmbg2),
            synthesis_models: Some(vec![SynthesisModel::Trellis]),
            backend: Some(self.config.default_backend),
            target_faces: args
                .target_faces
                .or(Some(DEFAULT_SCENE_TRELLIS_TARGET_FACES)),
            batch_size: args.batch_size,
            batch_vram_mb: args.batch_vram_mb,
            trellis_pbr: Some(args.trellis_pbr.unwrap_or(true)),
            trellis_pbr_texture_size: args
                .trellis_pbr_texture_size
                .or(Some(DEFAULT_SCENE_TRELLIS_PBR_TEXTURE_SIZE)),
            promote_to_catalog: args.promote_to_catalog,
            dry_run: false,
        })?;
        record_stage(&mut stage_report, "images_to_assets", stage_started);
        let mesh_quality_failures = scene_asset_quality_failures(&asset_outputs);
        if !mesh_quality_failures.is_empty() {
            response["asset_outputs"] = asset_outputs;
            response["mesh_quality_failures"] = json!(mesh_quality_failures);
            response["stage_report"] = json!(stage_report);
            response["e2e_summary"] = scene_build_summary(&response, e2e_started.elapsed());
            if args.write_artifacts {
                write_scene_build_artifacts(&output_dir, &response)?;
            }
            return Err(response["mesh_quality_failures"]
                .as_array()
                .and_then(|failures| failures.first())
                .and_then(Value::as_str)
                .unwrap_or("scene mesh quality gate failed")
                .to_string());
        }
        let stage_started = Instant::now();
        let asset_bindings =
            scene_asset_bindings_from_outputs(&manifest, &selected_values, &asset_outputs)?;
        let grounded_layout = grounded_scene_layout_for_manifest(&manifest, &asset_bindings)
            .map_err(|err| err.to_string())?;
        let mut bsn = grounded_layout.bsn.clone();
        let plan = parse_scene_bsn(&bsn, &asset_bindings).map_err(|err| err.to_string())?;
        let mut commands = scene_commands_with_cache_reload(
            scene_plan_to_mcp_commands(&plan, &asset_bindings, args.clear_existing)
                .map_err(|err| err.to_string())?,
        );
        record_stage(&mut stage_report, "plan_grounded_scene", stage_started);
        if args.feedback && args.lift_assets {
            let stage_started = Instant::now();
            let feedback = self.run_scene_feedback(
                &output_dir,
                &manifest,
                &asset_bindings,
                &grounded_layout,
                commands.clone(),
                SceneFeedbackOptions {
                    max_iters: args.feedback_iters,
                    keep_viewer: args.feedback_keep_viewer,
                    capture_dir: args.feedback_capture_dir.clone(),
                    threshold_profile: args.feedback_threshold_profile,
                },
            )?;
            if let Some(final_commands) = feedback
                .get("final_commands")
                .and_then(Value::as_array)
                .cloned()
            {
                commands = final_commands;
                bsn = feedback_bsn_from_commands(&asset_bindings, &grounded_layout, &commands)?;
            }
            response["feedback"] = feedback;
            record_stage(&mut stage_report, "render_capture_feedback", stage_started);
        }
        response["asset_outputs"] = asset_outputs;
        response["asset_bindings"] =
            serde_json::to_value(&asset_bindings).map_err(|err| err.to_string())?;
        response["bsn"] = json!(bsn);
        response["plan"] = serde_json::to_value(&plan).map_err(|err| err.to_string())?;
        response["grounded_layout"] =
            serde_json::to_value(&grounded_layout).map_err(|err| err.to_string())?;
        response["commands"] = json!(commands);
        response["clear_existing"] = json!(args.clear_existing);
        response["apply"] = json!(args.apply);
        if args.apply && !args.feedback {
            let stage_started = Instant::now();
            match self
                .send_scene_commands(response["commands"].as_array().cloned().unwrap_or_default())
            {
                Ok(acknowledgement) => {
                    response["acknowledgement"] = acknowledgement;
                }
                Err(err) => {
                    record_stage(&mut stage_report, "apply_scene_commands", stage_started);
                    response["apply_error"] = json!(err);
                    response["stage_report"] = json!(stage_report);
                    response["e2e_summary"] = scene_build_summary(&response, e2e_started.elapsed());
                    if args.write_artifacts {
                        write_scene_build_artifacts(&output_dir, &response)?;
                    }
                    return Err(response["apply_error"]
                        .as_str()
                        .unwrap_or("scene apply failed")
                        .to_string());
                }
            }
            record_stage(&mut stage_report, "apply_scene_commands", stage_started);
        }
        response["stage_report"] = json!(stage_report);
        response["e2e_summary"] = scene_build_summary(&response, e2e_started.elapsed());
        if args.write_artifacts {
            write_scene_build_artifacts(&output_dir, &response)?;
        }
        Ok(response)
    }

    fn call_scene_plan_bsn(&self, args: ScenePlanBsnArgs) -> Result<Value, String> {
        let grounded_layout =
            grounded_scene_layout_for_manifest(&args.manifest, &args.asset_bindings)
                .map_err(|err| err.to_string())?;
        let bsn = grounded_layout.bsn.clone();
        let plan = match parse_scene_bsn(&bsn, &args.asset_bindings) {
            Ok(plan) => plan,
            Err(err) => {
                return Ok(json!({
                    "tool": "scene_plan_bsn",
                    "valid": false,
                    "bsn": bsn,
                    "validation_error": err.to_string(),
                    "asset_bindings": args.asset_bindings,
                    "clear_existing": args.clear_existing,
                    "apply": false,
                }));
            }
        };
        let commands = scene_commands_with_cache_reload(
            scene_plan_to_mcp_commands(&plan, &args.asset_bindings, args.clear_existing)
                .map_err(|err| err.to_string())?,
        );
        let mut response = json!({
            "tool": "scene_plan_bsn",
            "valid": true,
            "bsn": bsn,
            "plan": plan,
            "grounded_layout": grounded_layout,
            "commands": commands,
            "asset_bindings": args.asset_bindings,
            "clear_existing": args.clear_existing,
            "apply": args.apply,
        });
        if args.apply {
            response["acknowledgement"] = self.send_scene_commands(
                response["commands"].as_array().cloned().unwrap_or_default(),
            )?;
        }
        Ok(response)
    }

    fn call_scene_ground(&mut self, args: SceneGroundToolArgs) -> Result<Value, String> {
        let started = Instant::now();
        let output_dir = args.output_dir.unwrap_or_else(default_scene_output_dir);
        fs::create_dir_all(&output_dir).map_err(|err| {
            format!(
                "failed to create scene-ground output directory {}: {err}",
                output_dir.display()
            )
        })?;
        let mut stage_report = Vec::new();
        let stage_started = Instant::now();
        let mut manifest = args.manifest;
        manifest.source_scene_path = args.source_scene_path.display().to_string();
        let (grounding_source, mut evidence) = if let Some(evidence) = args.grounding_evidence {
            ("provided", evidence)
        } else if args.locator == SceneLocatorProvider::LocateAnything {
            let backend = args
                .locate_anything_backend
                .unwrap_or(self.config.locate_anything_backend);
            let evidence = self.locate_anything_grounding_evidence(
                backend,
                &manifest,
                &args.source_scene_path,
                &output_dir,
            )?;
            let source = match backend {
                LocateAnythingBackend::PythonReference => "locate_anything_reference",
                LocateAnythingBackend::BurnNative => "locate_anything_burn_native",
            };
            (source, evidence)
        } else {
            ("manifest_fallback", manifest_grounding_evidence(&manifest))
        };
        record_stage(&mut stage_report, "load_grounding_evidence", stage_started);

        if args.depth_provider == SceneDepthProvider::DepthPro && evidence.depth.is_none() {
            let stage_started = Instant::now();
            self.depth_pro_grounding_evidence(&mut evidence, &args.source_scene_path, &output_dir)?;
            record_stage(
                &mut stage_report,
                "depth_pro_grounding_evidence",
                stage_started,
            );
        }

        let stage_started = Instant::now();
        let grounded_layout = match args.composition_mode {
            SceneCompositionMode::Heuristic => {
                grounded_scene_layout_for_manifest(&manifest, &args.asset_bindings)
            }
            SceneCompositionMode::CvGrounded => {
                grounded_scene_layout_with_evidence(&manifest, &args.asset_bindings, &evidence)
            }
        }
        .map_err(|err| err.to_string())?;
        let bsn = grounded_layout.bsn.clone();
        let plan = parse_scene_bsn(&bsn, &args.asset_bindings).map_err(|err| err.to_string())?;
        let mut commands = scene_commands_with_cache_reload(
            scene_plan_to_mcp_commands(&plan, &args.asset_bindings, args.clear_existing)
                .map_err(|err| err.to_string())?,
        );
        record_stage(&mut stage_report, "solve_grounded_scene", stage_started);

        let mut response = json!({
            "tool": "scene_ground",
            "source_scene_path": args.source_scene_path,
            "composition_mode": args.composition_mode,
            "depth_provider": args.depth_provider,
            "locator": args.locator,
            "grounding_source": grounding_source,
            "manifest": manifest,
            "asset_bindings": args.asset_bindings,
            "grounding_evidence": evidence,
            "grounded_layout": grounded_layout,
            "bsn": bsn,
            "plan": plan,
            "commands": commands,
            "clear_existing": args.clear_existing,
            "apply": args.apply,
        });

        if args.feedback {
            let stage_started = Instant::now();
            let manifest_value = response["manifest"].clone();
            let asset_bindings_value = response["asset_bindings"].clone();
            let grounded_layout_value = response["grounded_layout"].clone();
            let manifest: SceneObjectManifest = serde_json::from_value(manifest_value)
                .map_err(|err| format!("decode scene-ground manifest: {err}"))?;
            let asset_bindings: Vec<SceneAssetBinding> =
                serde_json::from_value(asset_bindings_value)
                    .map_err(|err| format!("decode scene-ground asset bindings: {err}"))?;
            let grounded_layout: GroundedSceneLayout =
                serde_json::from_value(grounded_layout_value)
                    .map_err(|err| format!("decode scene-ground layout: {err}"))?;
            let feedback = self.run_scene_feedback(
                &output_dir,
                &manifest,
                &asset_bindings,
                &grounded_layout,
                commands.clone(),
                SceneFeedbackOptions {
                    max_iters: args.feedback_iters,
                    keep_viewer: args.feedback_keep_viewer,
                    capture_dir: args.feedback_capture_dir.clone(),
                    threshold_profile: args.feedback_threshold_profile,
                },
            )?;
            if let Some(final_commands) = feedback
                .get("final_commands")
                .and_then(Value::as_array)
                .cloned()
            {
                commands = final_commands;
                response["commands"] = json!(commands);
            }
            response["feedback"] = feedback;
            record_stage(&mut stage_report, "render_capture_feedback", stage_started);
        }

        if args.apply && !args.feedback {
            let stage_started = Instant::now();
            let acknowledgement = self.send_scene_commands(commands)?;
            response["acknowledgement"] = acknowledgement;
            record_stage(&mut stage_report, "apply_scene_commands", stage_started);
        }

        response["stage_report"] = json!(stage_report);
        response["e2e_summary"] = json!({
            "elapsed_ms": started.elapsed().as_secs_f64() * 1000.0,
            "objects": response["grounded_layout"]["placements"].as_array().map(Vec::len).unwrap_or_default(),
            "composition_mode": response["composition_mode"],
            "grounding_source": response["grounding_source"],
        });
        write_scene_ground_artifacts(&output_dir, &response)?;
        Ok(response)
    }

    fn call_scene_apply_bsn(&self, args: SceneApplyBsnArgs) -> Result<Value, String> {
        let plan =
            parse_scene_bsn(&args.bsn, &args.asset_bindings).map_err(|err| err.to_string())?;
        let commands = scene_commands_with_cache_reload(
            scene_plan_to_mcp_commands(&plan, &args.asset_bindings, args.clear_existing)
                .map_err(|err| err.to_string())?,
        );
        let mut response = json!({
            "tool": "scene_apply_bsn",
            "plan": plan,
            "commands": commands,
            "apply": args.apply,
        });
        if args.apply {
            response["acknowledgement"] = self.send_scene_commands(
                response["commands"].as_array().cloned().unwrap_or_default(),
            )?;
        }
        Ok(response)
    }

    fn scene_build_config(&self, args: ScenePrepareBuildArgs) -> Result<SceneBuildConfig, String> {
        Ok(SceneBuildConfig {
            source_scene_path: args.source_scene_path,
            object_reference_image_path: args
                .object_reference_image_path
                .unwrap_or_else(|| self.config.scene_object_reference_image.clone()),
            output_dir: args.output_dir.unwrap_or_else(default_scene_output_dir),
            candidate_count: args.candidate_count.unwrap_or(3).max(1),
            quality_profile: args.quality_profile.unwrap_or(SceneQualityProfile::Quality),
            reasoning_model: self.config.openai_reasoning_model.clone(),
            image_model: self.config.openai_image_model.clone(),
            allow_catalog_reuse: args.allow_catalog_reuse,
        })
    }

    fn openai_provider(&self) -> Result<OpenAiSceneProvider, String> {
        OpenAiSceneProvider::from_env(OpenAiProviderConfig {
            api_key: self.config.openai_api_key.clone().unwrap_or_default(),
            base_url: self
                .config
                .openai_base_url
                .clone()
                .unwrap_or_else(|| OpenAiProviderConfig::default().base_url),
            project_id: self.config.openai_project_id.clone(),
            reasoning_model: self.config.openai_reasoning_model.clone(),
            image_model: self.config.openai_image_model.clone(),
            ..OpenAiProviderConfig::default()
        })
        .map_err(|err| err.to_string())
    }

    fn open_catalog_cache(&self) -> Result<MeshCache, String> {
        if let Some(root) = self.config.catalog_cache_root.as_ref() {
            MeshCache::load_from_root(root.clone())
        } else {
            MeshCache::load_default()
        }
        .map_err(|err| format!("failed to open shared asset cache: {err}"))
    }

    fn call_scene_status(&self) -> Result<Value, String> {
        let status_path = self
            .config
            .scene_status_path
            .as_ref()
            .ok_or_else(|| "scene_status_path is not configured".to_string())?;
        read_scene_status(status_path)
    }

    fn call_scene_project_status(&self) -> Result<Value, String> {
        let status = self.call_scene_status()?;
        Ok(json!({
            "tool": "scene_project_status",
            "camera": status.get("camera").cloned().unwrap_or(Value::Null),
            "world_items": status.get("world_items").cloned().unwrap_or(Value::Null),
            "projected_items": status.get("projected_items").cloned().unwrap_or(Value::Null),
            "screenshots": status.get("screenshots").cloned().unwrap_or(Value::Null),
            "status": status,
        }))
    }

    fn call_scene_list_assets(&self) -> Result<Value, String> {
        let status = self.call_scene_status()?;
        Ok(json!({
            "tool": "scene_list_assets",
            "cache_entries": status["cache_entries"].clone(),
            "world_items": status["world_items"].clone(),
        }))
    }

    fn call_scene_spawn_cached(&self, args: SceneSpawnCachedArgs) -> Result<Value, String> {
        self.send_scene_commands(vec![json!({
            "type": "spawn_cached",
            "cache_key": args.cache_key,
            "translation": args.translation,
            "rotation": args.rotation,
            "scale": args.scale,
            "select": args.select,
        })])
    }

    fn call_scene_spawn_path(&self, args: SceneSpawnPathArgs) -> Result<Value, String> {
        self.send_scene_commands(vec![json!({
            "type": "spawn_path",
            "path": args.path,
            "translation": args.translation,
            "rotation": args.rotation,
            "scale": args.scale,
            "select": args.select,
        })])
    }

    fn call_scene_delete(&self, args: SceneDeleteArgs) -> Result<Value, String> {
        if let Some(cache_key) = args.cache_key {
            return self.send_scene_commands(vec![json!({
                "type": "delete_by_cache_key",
                "cache_key": cache_key,
            })]);
        }
        if args.selected {
            return self.send_scene_commands(vec![json!({ "type": "delete_selected" })]);
        }
        self.send_scene_commands(vec![json!({ "type": "clear_selection" })])
    }

    fn call_scene_clear(&self) -> Result<Value, String> {
        self.send_scene_commands(vec![json!({ "type": "clear_scene" })])
    }

    fn call_scene_set_camera(&self, args: SceneSetCameraArgs) -> Result<Value, String> {
        self.send_scene_commands(vec![json!({
            "type": "set_camera",
            "translation": args.translation,
            "rotation": args.rotation,
            "focus": args.focus,
            "yaw": args.yaw,
            "pitch": args.pitch,
            "radius": args.radius,
            "vertical_fov": args.vertical_fov,
        })])
    }

    fn call_scene_capture(&self, args: SceneCaptureArgs) -> Result<Value, String> {
        let path = args.output_path;
        let response = self.send_scene_commands(vec![json!({
            "type": "capture_screenshot",
            "path": path.display().to_string(),
        })])?;
        let timeout = self.config.scene_timeout;
        let started = Instant::now();
        while started.elapsed() < timeout {
            if path
                .metadata()
                .map(|metadata| metadata.len() > 0)
                .unwrap_or(false)
            {
                return Ok(json!({
                    "tool": "scene_capture",
                    "output_path": path.display().to_string(),
                    "acknowledgement": response,
                }));
            }
            thread::sleep(Duration::from_millis(25));
        }
        Err(format!(
            "scene_capture timed out waiting for screenshot {}",
            path.display()
        ))
    }

    fn call_scene_compose_assets(&self, args: SceneComposeArgs) -> Result<Value, String> {
        let plan = compose_scene_layout(args)?;
        let mut response =
            serde_json::to_value(&plan).map_err(|err| format!("serialize layout plan: {err}"))?;
        if plan.apply {
            let acknowledgement = self.send_scene_commands(scene_commands_from_plan(&plan)?)?;
            response["acknowledgement"] = acknowledgement;
        }
        Ok(response)
    }

    fn call_scene_validate_layout(&self, mut args: SceneValidateArgs) -> Result<Value, String> {
        if args.scene_status.is_none() {
            args.scene_status = Some(self.call_scene_status()?);
        }
        validate_scene_layout(args)
    }

    fn run_scene_feedback(
        &mut self,
        output_dir: &Path,
        manifest: &SceneObjectManifest,
        asset_bindings: &[SceneAssetBinding],
        grounded_layout: &GroundedSceneLayout,
        initial_commands: Vec<Value>,
        options: SceneFeedbackOptions,
    ) -> Result<Value, String> {
        let original_control_path = self.config.scene_control_path.clone();
        let original_status_path = self.config.scene_status_path.clone();
        let original_timeout = self.config.scene_timeout;
        let capture_root = options
            .capture_dir
            .clone()
            .unwrap_or_else(|| output_dir.join("iterations"));
        fs::create_dir_all(&capture_root).map_err(|err| {
            format!(
                "failed to create feedback capture directory {}: {err}",
                capture_root.display()
            )
        })?;

        let mut spawned_viewer = None;
        if self.config.scene_control_path.is_none() {
            let bridge_dir = output_dir.join("feedback_viewer");
            fs::create_dir_all(&bridge_dir).map_err(|err| {
                format!(
                    "failed to create feedback viewer directory {}: {err}",
                    bridge_dir.display()
                )
            })?;
            let control_path = bridge_dir.join("scene_commands.json");
            let status_path = bridge_dir.join("scene_commands.status.json");
            let log_path = bridge_dir.join("viewer.log");
            spawned_viewer = Some(spawn_feedback_viewer(&control_path, &log_path)?);
            self.config.scene_control_path = Some(control_path);
            self.config.scene_status_path = Some(status_path);
            self.config.scene_timeout = self.config.scene_timeout.max(Duration::from_secs(60));
        }

        let lock_result = self.send_scene_commands(vec![scene_interaction_lock_command(
            true,
            "iterative scene composition",
        )]);
        let feedback_result = match lock_result {
            Ok(lock_ack) => {
                let _ = write_json_file(&capture_root.join("interaction_lock_ack.json"), &lock_ack);
                let mut result =
                    self.run_scene_feedback_iterations(SceneFeedbackIterationContext {
                        capture_root: &capture_root,
                        manifest,
                        asset_bindings,
                        grounded_layout,
                        initial_commands,
                        max_iters: options.max_iters.max(1),
                        threshold_profile: options.threshold_profile,
                    });
                if let Ok(value) = &mut result {
                    value["interaction_lock_ack"] = lock_ack;
                }
                result
            }
            Err(err) => Err(format!("failed to lock scene interaction: {err}")),
        };
        let unlock_result =
            self.send_scene_commands(vec![scene_interaction_lock_command(false, "")]);
        let feedback_result = match (feedback_result, unlock_result) {
            (Ok(mut value), Ok(unlock_ack)) => {
                let _ = write_json_file(
                    &capture_root.join("interaction_unlock_ack.json"),
                    &unlock_ack,
                );
                value["interaction_unlock_ack"] = unlock_ack;
                Ok(value)
            }
            (Ok(_), Err(unlock_err)) => Err(format!(
                "feedback completed but failed to unlock scene interaction: {unlock_err}"
            )),
            (Err(err), Ok(unlock_ack)) => {
                let _ = write_json_file(
                    &capture_root.join("interaction_unlock_ack.json"),
                    &unlock_ack,
                );
                Err(err)
            }
            (Err(err), Err(unlock_err)) => Err(format!(
                "{err}; additionally failed to unlock scene interaction: {unlock_err}"
            )),
        };

        if let Some(mut child) = spawned_viewer
            && !options.keep_viewer
        {
            let _ = child.kill();
            let _ = child.wait();
        }
        self.config.scene_control_path = original_control_path;
        self.config.scene_status_path = original_status_path;
        self.config.scene_timeout = original_timeout;

        feedback_result
    }

    fn run_scene_feedback_iterations(
        &self,
        context: SceneFeedbackIterationContext<'_>,
    ) -> Result<Value, String> {
        let SceneFeedbackIterationContext {
            capture_root,
            manifest,
            asset_bindings,
            grounded_layout,
            initial_commands,
            max_iters,
            threshold_profile,
        } = context;
        let mut commands = initial_commands;
        let thresholds = threshold_profile.thresholds();
        let mut iterations = Vec::new();
        let mut accepted_iteration = None;
        let mut best_iteration = None;
        let mut best_score = f64::NEG_INFINITY;
        let mut best_commands = commands.clone();
        for iteration_index in 0..max_iters {
            let iteration_dir = capture_root.join(format!("iter_{iteration_index:02}"));
            fs::create_dir_all(&iteration_dir).map_err(|err| {
                format!(
                    "failed to create feedback iteration directory {}: {err}",
                    iteration_dir.display()
                )
            })?;
            write_json_file(&iteration_dir.join("commands.json"), &json!(commands))
                .map_err(|err| err.to_string())?;
            let iteration_bsn =
                feedback_bsn_from_commands(asset_bindings, grounded_layout, &commands)?;
            fs::write(iteration_dir.join("scene.bsn"), iteration_bsn).map_err(|err| {
                format!(
                    "failed to write feedback BSN {}: {err}",
                    iteration_dir.join("scene.bsn").display()
                )
            })?;

            let apply_ack = self.send_scene_commands(commands.clone())?;
            write_json_file(&iteration_dir.join("apply_ack.json"), &apply_ack)
                .map_err(|err| err.to_string())?;
            thread::sleep(Duration::from_millis(250));
            let screenshot_path = iteration_dir.join("screenshot.png");
            let capture = self.call_scene_capture(SceneCaptureArgs {
                output_path: screenshot_path.clone(),
            })?;
            write_json_file(&iteration_dir.join("capture_ack.json"), &capture)
                .map_err(|err| err.to_string())?;
            let status = Self::feedback_capture_status(&apply_ack, &capture);
            write_json_file(&iteration_dir.join("status.json"), &status)
                .map_err(|err| err.to_string())?;
            let metrics = scene_feedback_metrics(
                manifest,
                grounded_layout,
                &status,
                &screenshot_path,
                thresholds,
                threshold_profile,
            )?;
            write_json_file(&iteration_dir.join("metrics.json"), &metrics)
                .map_err(|err| err.to_string())?;
            let passed = metrics
                .get("passed")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let selection_score = feedback_selection_score(&metrics);
            if selection_score > best_score {
                best_score = selection_score;
                best_iteration = Some(iteration_index);
                best_commands = commands.clone();
            }
            let deltas = feedback_layout_deltas(&metrics);
            write_json_file(&iteration_dir.join("layout_delta.json"), &deltas)
                .map_err(|err| err.to_string())?;
            iterations.push(json!({
                "iteration": iteration_index,
                "dir": iteration_dir.display().to_string(),
                "screenshot": screenshot_path.display().to_string(),
                "metrics": metrics,
                "layout_delta": deltas,
                "passed": passed,
                "selection_score": selection_score,
            }));
            if passed {
                accepted_iteration = Some(iteration_index);
                break;
            }
            commands = apply_feedback_deltas_to_commands(&commands, &deltas)?;
        }
        if accepted_iteration.is_none() && best_iteration.is_some() {
            commands = best_commands;
        }
        let final_bsn = feedback_bsn_from_commands(asset_bindings, grounded_layout, &commands)?;
        fs::write(capture_root.join("scene.feedback.bsn"), &final_bsn).map_err(|err| {
            format!(
                "failed to write final feedback BSN {}: {err}",
                capture_root.join("scene.feedback.bsn").display()
            )
        })?;
        write_json_file(
            &capture_root.join("commands.feedback.json"),
            &json!(commands),
        )
        .map_err(|err| err.to_string())?;
        let mut final_evidence = Value::Null;
        if accepted_iteration.is_none() && max_iters > 0 {
            let final_dir = capture_root.join("final");
            fs::create_dir_all(&final_dir).map_err(|err| {
                format!(
                    "failed to create final feedback directory {}: {err}",
                    final_dir.display()
                )
            })?;
            write_json_file(&final_dir.join("commands.json"), &json!(commands))
                .map_err(|err| err.to_string())?;
            fs::write(final_dir.join("scene.bsn"), &final_bsn).map_err(|err| {
                format!(
                    "failed to write final feedback BSN {}: {err}",
                    final_dir.join("scene.bsn").display()
                )
            })?;
            let apply_ack = self.send_scene_commands(commands.clone())?;
            write_json_file(&final_dir.join("apply_ack.json"), &apply_ack)
                .map_err(|err| err.to_string())?;
            thread::sleep(Duration::from_millis(250));
            let screenshot_path = final_dir.join("screenshot.png");
            let capture = self.call_scene_capture(SceneCaptureArgs {
                output_path: screenshot_path.clone(),
            })?;
            write_json_file(&final_dir.join("capture_ack.json"), &capture)
                .map_err(|err| err.to_string())?;
            let status = Self::feedback_capture_status(&apply_ack, &capture);
            write_json_file(&final_dir.join("status.json"), &status)
                .map_err(|err| err.to_string())?;
            let metrics = scene_feedback_metrics(
                manifest,
                grounded_layout,
                &status,
                &screenshot_path,
                thresholds,
                threshold_profile,
            )?;
            write_json_file(&final_dir.join("metrics.json"), &metrics)
                .map_err(|err| err.to_string())?;
            let passed = metrics
                .get("passed")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            final_evidence = json!({
                "dir": final_dir.display().to_string(),
                "screenshot": screenshot_path.display().to_string(),
                "metrics": metrics,
                "passed": passed,
            });
        }
        let report = feedback_markdown_report(
            capture_root,
            threshold_profile,
            accepted_iteration,
            &iterations,
        );
        fs::write(capture_root.join("feedback_report.md"), report).map_err(|err| {
            format!(
                "failed to write feedback markdown report {}: {err}",
                capture_root.join("feedback_report.md").display()
            )
        })?;
        Ok(json!({
            "tool": "scene_render_capture_feedback",
            "enabled": true,
            "threshold_profile": threshold_profile,
            "max_iters": max_iters,
            "accepted": accepted_iteration.is_some(),
            "accepted_iteration": accepted_iteration,
            "best_iteration": best_iteration,
            "best_score": if best_score.is_finite() { Value::from(best_score) } else { Value::Null },
            "capture_dir": capture_root.display().to_string(),
            "iterations": iterations,
            "final_evidence": final_evidence,
            "final_bsn_path": capture_root.join("scene.feedback.bsn").display().to_string(),
            "final_commands_path": capture_root.join("commands.feedback.json").display().to_string(),
            "final_commands": commands,
        }))
    }

    fn feedback_capture_status(apply_ack: &Value, capture_ack: &Value) -> Value {
        capture_ack
            .get("acknowledgement")
            .and_then(|ack| ack.get("status"))
            .cloned()
            .or_else(|| apply_ack.get("status").cloned())
            .unwrap_or(Value::Null)
    }

    fn send_scene_commands(&self, commands: Vec<Value>) -> Result<Value, String> {
        if commands.is_empty() {
            return Err("scene command list must not be empty".to_string());
        }
        let control_path = self
            .config
            .scene_control_path
            .as_ref()
            .ok_or_else(|| "scene_control_path is not configured".to_string())?;
        let sequence = next_scene_sequence();
        let session_id = format!("burn_synth_mcp-{}", std::process::id());
        let envelope = json!({
            "session_id": session_id,
            "sequence": sequence,
            "commands": commands,
        });
        atomic_write_json(control_path, &envelope)?;

        let Some(status_path) = self.config.scene_status_path.as_ref() else {
            return Ok(json!({
                "tool": "scene_command",
                "command_path": control_path.display().to_string(),
                "sequence": sequence,
                "acknowledged": false,
            }));
        };
        let status = wait_scene_status(status_path, sequence, self.config.scene_timeout)?;
        Ok(json!({
            "tool": "scene_command",
            "command_path": control_path.display().to_string(),
            "status_path": status_path.display().to_string(),
            "sequence": sequence,
            "acknowledged": true,
            "status": status,
        }))
    }
}

fn scene_commands_from_plan(plan: &SceneComposePlan) -> Result<Vec<Value>, String> {
    let mut commands = Vec::with_capacity(plan.placements.len());
    if plan.clear_existing {
        commands.push(json!({ "type": "clear_scene" }));
    }
    for placement in &plan.placements {
        if let Some(path) = placement.path.as_ref() {
            commands.push(json!({
                "type": "spawn_path",
                "path": path,
                "cache_key": placement.cache_key,
                "translation": placement.translation,
                "rotation": placement.rotation,
                "scale": placement.scale,
                "select": placement.select,
            }));
        } else if let Some(cache_key) = placement.cache_key.as_ref() {
            commands.push(json!({
                "type": "spawn_cached",
                "cache_key": cache_key,
                "translation": placement.translation,
                "rotation": placement.rotation,
                "scale": placement.scale,
                "select": placement.select,
            }));
        } else {
            return Err(format!(
                "placement for '{}' has neither path nor cache_key",
                placement.label
            ));
        }
    }
    Ok(commands)
}

fn scene_interaction_lock_command(locked: bool, reason: &str) -> Value {
    json!({
        "type": "set_interaction_lock",
        "locked": locked,
        "reason": reason,
    })
}

fn scene_commands_with_cache_reload(mut commands: Vec<Value>) -> Vec<Value> {
    let uses_cache = commands.iter().any(|command| {
        command
            .get("type")
            .and_then(Value::as_str)
            .is_some_and(|command_type| command_type == "spawn_cached")
    });
    if !uses_cache {
        return commands;
    }

    let insert_at = commands
        .first()
        .and_then(|command| command.get("type"))
        .and_then(Value::as_str)
        .filter(|command_type| *command_type == "clear_scene")
        .map(|_| 1)
        .unwrap_or(0);
    commands.insert(insert_at, json!({ "type": "reload_cache" }));
    commands
}

fn spawn_feedback_viewer(control_path: &Path, log_path: &Path) -> Result<Child, String> {
    let exe = feedback_viewer_exe()?;
    ensure_parent_dir(control_path).map_err(|err| err.to_string())?;
    ensure_parent_dir(log_path).map_err(|err| err.to_string())?;
    let log = fs::File::create(log_path).map_err(|err| {
        format!(
            "failed to create feedback viewer log {}: {err}",
            log_path.display()
        )
    })?;
    let err_log = log
        .try_clone()
        .map_err(|err| format!("failed to clone feedback viewer log handle: {err}"))?;
    Command::new(&exe)
        .arg("--mcp-scene-control-path")
        .arg(control_path)
        .arg("--ui-visible")
        .arg("false")
        .arg("--read-only")
        .stdin(Stdio::null())
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(err_log))
        .spawn()
        .map_err(|err| format!("failed to spawn feedback viewer {}: {err}", exe.display()))
}

fn feedback_viewer_exe() -> Result<PathBuf, String> {
    let current = std::env::current_exe()
        .map_err(|err| format!("failed to resolve current executable: {err}"))?;
    let direct = current.with_file_name(format!("bevy_synth{}", std::env::consts::EXE_SUFFIX));
    if direct.exists() {
        return Ok(direct);
    }
    if current
        .parent()
        .and_then(|parent| parent.file_name())
        .and_then(|name| name.to_str())
        .is_some_and(|name| name == "deps")
        && let Some(parent) = current.parent().and_then(Path::parent)
    {
        let candidate = parent.join(format!("bevy_synth{}", std::env::consts::EXE_SUFFIX));
        if candidate.exists() {
            return Ok(candidate);
        }
    }
    Ok(direct)
}

fn scene_feedback_metrics(
    manifest: &SceneObjectManifest,
    grounded_layout: &GroundedSceneLayout,
    status: &Value,
    screenshot_path: &Path,
    thresholds: SceneFeedbackThresholds,
    profile: FeedbackThresholdProfile,
) -> Result<Value, String> {
    let projected = status
        .get("projected_items")
        .and_then(Value::as_array)
        .ok_or_else(|| "scene feedback status missing projected_items array".to_string())?;
    let mut object_metrics = Vec::new();
    let mut passed_count = 0usize;
    let mut score_total = 0.0f32;
    let camera_yaw_degrees = status_camera_yaw_degrees(status, grounded_layout.camera.yaw);
    let feedback_camera = FeedbackCamera::from_status(status, screenshot_path);
    let floor_y = grounded_layout.rug_center[1];
    let footprints = feedback_projected_footprints(&grounded_layout.placements, projected);
    let physical = feedback_physical_layout(&grounded_layout.placements, &footprints, thresholds);
    for (index, placement) in grounded_layout.placements.iter().enumerate() {
        let projected_item = projected.get(index).unwrap_or(&Value::Null);
        let observed_bbox = projected_item.get("screen_bbox").and_then(json_array4);
        let observed_contact = projected_item.get("screen_contact").and_then(json_array2);
        let expected_bbox = placement.source_bbox;
        let expected_contact = placement.contact_pixel;
        let expected_anchor = feedback_expected_anchor_pixel(placement);
        let anchor_basis = feedback_anchor_basis(placement);
        let uses_center_anchor = feedback_uses_bbox_center_anchor(placement);
        let (center_error, area_log2_error, aspect_log2_error, contact_error, score, passed) =
            if let Some(observed_bbox) = observed_bbox {
                let expected_center = bbox_center(expected_bbox);
                let observed_center = bbox_center(observed_bbox);
                let center_error = distance2(expected_center, observed_center);
                let expected_area = bbox_area(expected_bbox);
                let observed_area = bbox_area(observed_bbox);
                let area_log2_error = safe_log2_ratio(observed_area, expected_area).abs();
                let aspect_log2_error =
                    safe_log2_ratio(bbox_aspect(observed_bbox), bbox_aspect(expected_bbox)).abs();
                let observed_anchor =
                    feedback_observed_anchor_pixel(placement, observed_bbox, observed_contact);
                let contact_error = distance2(expected_anchor, observed_anchor);
                let center_limit = feedback_center_error_limit(
                    uses_center_anchor,
                    contact_error,
                    thresholds.max_center_error,
                    thresholds.max_contact_error,
                );
                let area_limit = feedback_area_log2_error_limit(
                    uses_center_anchor,
                    thresholds.max_area_log2_error,
                );
                let center_score = (1.0 - center_error / center_limit.max(1.0e-5)).clamp(0.0, 1.0);
                let contact_score = (1.0
                    - contact_error / thresholds.max_contact_error.max(1.0e-5))
                .clamp(0.0, 1.0);
                let area_score = (1.0 - area_log2_error / area_limit.max(1.0e-5)).clamp(0.0, 1.0);
                let score = if uses_center_anchor {
                    center_score * 0.45 + contact_score * 0.25 + area_score * 0.30
                } else {
                    center_score * 0.20 + contact_score * 0.45 + area_score * 0.35
                };
                let passed = center_error <= center_limit
                    && contact_error <= thresholds.max_contact_error
                    && area_log2_error <= area_limit;
                (
                    center_error,
                    area_log2_error,
                    aspect_log2_error,
                    contact_error,
                    score,
                    passed,
                )
            } else {
                (1.0, 8.0, 8.0, 1.0, 0.0, false)
            };
        if passed {
            passed_count += 1;
        }
        score_total += score;
        let observed_bbox = observed_bbox.unwrap_or([0.0, 0.0, 0.0, 0.0]);
        let observed_contact = observed_contact.unwrap_or(bbox_center(observed_bbox));
        let observed_anchor =
            feedback_observed_anchor_pixel(placement, observed_bbox, Some(observed_contact));
        let expected_area = bbox_area(expected_bbox);
        let observed_area = bbox_area(observed_bbox);
        let scale_multiplier = if observed_area > 1.0e-5 {
            (expected_area / observed_area).sqrt().clamp(0.82, 1.22)
        } else {
            1.0
        };
        let contact_delta = vec2_sub(expected_anchor, observed_anchor);
        let center_delta = vec2_sub(bbox_center(expected_bbox), bbox_center(observed_bbox));
        let fallback_delta = [
            (center_delta[0] * 2.0).clamp(-0.35, 0.35),
            0.0,
            (contact_delta[1] * 2.0).clamp(-0.35, 0.35),
        ];
        let target_ground_point =
            feedback_camera.and_then(|camera| camera.ground_point(expected_anchor, floor_y));
        let observed_ground_point = projected_item_ground_point(projected_item);
        let (mut translation_delta, grounding_basis) =
            if let (Some(target), Some(observed)) = (target_ground_point, observed_ground_point) {
                (
                    clamp_xz_delta(
                        [target[0] - observed[0], 0.0, target[2] - observed[2]],
                        0.85,
                    ),
                    "camera-ray-ground-plane",
                )
            } else {
                (fallback_delta, "screen-space-fallback")
            };
        let center_residual_applied = grounding_basis == "camera-ray-ground-plane"
            && !feedback_uses_bbox_center_anchor(placement)
            && contact_error <= 0.04
            && center_error > 0.04;
        if center_residual_applied {
            let residual = [
                (-center_delta[0] * 1.15).clamp(-0.18, 0.18),
                0.0,
                (-center_delta[1] * 1.15).clamp(-0.18, 0.18),
            ];
            translation_delta = clamp_xz_delta(add3(translation_delta, residual), 0.85);
        }
        let contact_residual_applied = grounding_basis == "camera-ray-ground-plane"
            && !feedback_uses_bbox_center_anchor(placement)
            && contact_error > 0.04;
        if contact_residual_applied {
            let residual = [
                (center_delta[0] * 0.65).clamp(-0.14, 0.14),
                0.0,
                (-contact_delta[1] * 1.45).clamp(-0.26, 0.26),
            ];
            translation_delta = clamp_xz_delta(add3(translation_delta, residual), 0.95);
        }
        if feedback_physical_kind(placement) == FeedbackPhysicalKind::Table {
            translation_delta = clamp_xz_delta(translation_delta, 0.15);
        }
        let physical_delta = physical
            .corrections
            .get(&index)
            .copied()
            .unwrap_or([0.0, 0.0, 0.0]);
        if physical_delta[0].abs() + physical_delta[2].abs() > 1.0e-5 {
            translation_delta = clamp_xz_delta(add3(translation_delta, physical_delta), 1.10);
        }
        let predictive_physical_delta = feedback_predictive_physical_delta(
            index,
            placement,
            translation_delta,
            &footprints,
            thresholds,
        );
        if predictive_physical_delta[0].abs() + predictive_physical_delta[2].abs() > 1.0e-5 {
            translation_delta =
                clamp_xz_delta(add3(translation_delta, predictive_physical_delta), 1.10);
        }
        let footprint = footprints.get(index).and_then(|footprint| *footprint);
        let world_footprint = footprint.map(|footprint| {
            json!({
                "min_x": footprint.rect.min_x,
                "min_z": footprint.rect.min_z,
                "max_x": footprint.rect.max_x,
                "max_z": footprint.rect.max_z,
            })
        });
        let projected_cache_key = projected_item.get("cache_key").and_then(Value::as_str);
        let current_yaw_degrees = status_world_item_yaw_degrees(status, index, projected_cache_key)
            .unwrap_or(placement.rotation_y_degrees);
        let yaw_correction =
            feedback_yaw_correction(index, placement, current_yaw_degrees, &physical);
        let canonical_yaw_error =
            normalize_degrees(placement.rotation_y_degrees - current_yaw_degrees);
        let physical_failures = physical
            .object_failures
            .get(&index)
            .cloned()
            .unwrap_or_default();
        let physical_passed = physical_failures.is_empty();
        object_metrics.push(json!({
            "index": index,
            "object_id": placement.object_id,
            "instance_id": placement.instance_id,
            "label": placement.label,
            "cache_key": projected_item.get("cache_key").cloned().unwrap_or(Value::Null),
            "expected_bbox": expected_bbox,
            "observed_bbox": observed_bbox,
            "expected_contact": expected_contact,
            "observed_contact": observed_contact,
            "expected_anchor": expected_anchor,
            "observed_anchor": observed_anchor,
            "anchor_basis": anchor_basis,
            "center_error": center_error,
            "contact_error": contact_error,
            "area_log2_error": area_log2_error,
            "aspect_log2_error": aspect_log2_error,
            "score": score,
            "passed": passed,
            "translation_delta": translation_delta,
            "grounding_basis": grounding_basis,
            "center_residual_applied": center_residual_applied,
            "contact_residual_applied": contact_residual_applied,
            "physical_translation_delta": physical_delta,
            "predictive_physical_translation_delta": predictive_physical_delta,
            "physical_kind": feedback_physical_kind_str(feedback_physical_kind(placement)),
            "world_footprint": world_footprint,
            "physical_passed": physical_passed,
            "physical_failures": physical_failures,
            "target_ground_point": target_ground_point,
            "observed_ground_point": observed_ground_point,
            "scale_multiplier": scale_multiplier,
            "yaw_delta_degrees": yaw_correction.delta_degrees,
            "yaw_basis": yaw_correction.basis,
            "current_yaw_degrees": current_yaw_degrees,
            "canonical_yaw_degrees": placement.rotation_y_degrees,
            "canonical_yaw_error_degrees": canonical_yaw_error,
            "camera_yaw_degrees": camera_yaw_degrees,
            "target_yaw_degrees": normalize_degrees(current_yaw_degrees + yaw_correction.delta_degrees),
        }));
    }
    let object_count = grounded_layout.placements.len().max(1);
    let mean_score = score_total / object_count as f32;
    let projection_passed = passed_count == grounded_layout.placements.len()
        && mean_score >= thresholds.min_overall_score;
    let physical_passed = physical.hard_failure_count == 0;
    let passed = projection_passed && physical_passed;
    Ok(json!({
        "profile": profile,
        "passed": passed,
        "score": mean_score,
        "projection_passed": projection_passed,
        "physical_passed": physical_passed,
        "object_count": grounded_layout.placements.len(),
        "object_pass_count": passed_count,
        "physical_pass_count": grounded_layout.placements.len().saturating_sub(physical.object_failure_count),
        "source_scene_path": manifest.source_scene_path,
        "screenshot_path": screenshot_path.display().to_string(),
        "thresholds": {
            "max_center_error": thresholds.max_center_error,
            "max_contact_error": thresholds.max_contact_error,
            "max_area_log2_error": thresholds.max_area_log2_error,
            "min_overall_score": thresholds.min_overall_score,
            "max_seating_table_overlap_fraction": thresholds.max_seating_table_overlap_fraction,
            "max_seating_table_penetration_m": thresholds.max_seating_table_penetration_m,
            "max_seating_seating_overlap_fraction": thresholds.max_seating_seating_overlap_fraction,
            "max_seating_seating_penetration_m": thresholds.max_seating_seating_penetration_m,
        },
        "physical_layout": {
            "passed": physical_passed,
            "hard_failure_count": physical.hard_failure_count,
            "warning_count": physical.warning_count,
            "object_failure_count": physical.object_failure_count,
            "max_overlap_fraction_smaller": physical.max_overlap_fraction_smaller,
            "min_signed_clearance_m": physical.min_signed_clearance_m,
            "pairs": physical.pairs,
        },
        "objects": object_metrics,
        "camera": status.get("camera").cloned().unwrap_or(Value::Null),
    }))
}

fn feedback_center_error_limit(
    uses_center_anchor: bool,
    contact_error: f32,
    max_center_error: f32,
    max_contact_error: f32,
) -> f32 {
    if uses_center_anchor {
        max_center_error
    } else if contact_error <= max_contact_error {
        max_center_error * 1.60
    } else {
        max_center_error * 1.25
    }
}

fn feedback_area_log2_error_limit(uses_center_anchor: bool, max_area_log2_error: f32) -> f32 {
    if uses_center_anchor {
        max_area_log2_error
    } else {
        max_area_log2_error * 1.25
    }
}

fn feedback_expected_anchor_pixel(placement: &GroundedScenePlacement) -> [f32; 2] {
    if feedback_uses_bbox_center_anchor(placement) {
        bbox_center(placement.source_bbox)
    } else {
        placement.contact_pixel
    }
}

fn feedback_observed_anchor_pixel(
    placement: &GroundedScenePlacement,
    observed_bbox: [f32; 4],
    observed_contact: Option<[f32; 2]>,
) -> [f32; 2] {
    if feedback_uses_bbox_center_anchor(placement) {
        bbox_center(observed_bbox)
    } else {
        observed_contact.unwrap_or_else(|| bbox_center(observed_bbox))
    }
}

fn feedback_anchor_basis(placement: &GroundedScenePlacement) -> &'static str {
    if feedback_uses_bbox_center_anchor(placement) {
        "bbox-center"
    } else {
        "floor-contact"
    }
}

fn feedback_uses_bbox_center_anchor(placement: &GroundedScenePlacement) -> bool {
    let descriptor = format!("{} {}", placement.object_id, placement.label).to_lowercase();
    descriptor.contains("table")
        || descriptor.contains("desk")
        || descriptor.contains("counter")
        || descriptor.contains("bench")
}

fn feedback_layout_deltas(metrics: &Value) -> Value {
    let objects = metrics
        .get("objects")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut scale_groups: HashMap<String, (f64, usize)> = HashMap::new();
    let mut camera_ray_contact_sum = 0.0f64;
    let mut camera_ray_contact_count = 0usize;
    for object in &objects {
        if object
            .get("grounding_basis")
            .and_then(Value::as_str)
            .is_some_and(|basis| basis == "camera-ray-ground-plane")
            && let Some(contact_error) = object
                .get("contact_error")
                .and_then(Value::as_f64)
                .filter(|value| value.is_finite())
        {
            camera_ray_contact_sum += contact_error;
            camera_ray_contact_count += 1;
        }
        let Some(group_key) = feedback_scale_group_key(object) else {
            continue;
        };
        let Some(scale) = object
            .get("scale_multiplier")
            .and_then(Value::as_f64)
            .filter(|value| value.is_finite())
        else {
            continue;
        };
        let scale = damped_feedback_scale_multiplier(object, scale);
        let entry = scale_groups.entry(group_key).or_insert((0.0, 0));
        entry.0 += scale.clamp(0.82, 1.22);
        entry.1 += 1;
    }
    let repeated_scale_by_group = scale_groups
        .into_iter()
        .filter_map(|(key, (sum, count))| {
            if count > 1 {
                Some((key, (sum / count as f64).clamp(0.82, 1.22)))
            } else {
                None
            }
        })
        .collect::<HashMap<_, _>>();
    let mut source_min = [f32::INFINITY, f32::INFINITY];
    let mut source_max = [f32::NEG_INFINITY, f32::NEG_INFINITY];
    let mut observed_min = [f32::INFINITY, f32::INFINITY];
    let mut observed_max = [f32::NEG_INFINITY, f32::NEG_INFINITY];
    for object in &objects {
        if let Some(bbox) = object.get("expected_bbox").and_then(json_array4) {
            expand_bbox_envelope(&mut source_min, &mut source_max, bbox);
        }
        if let Some(bbox) = object.get("observed_bbox").and_then(json_array4) {
            expand_bbox_envelope(&mut observed_min, &mut observed_max, bbox);
        }
    }
    let source_area = envelope_area(source_min, source_max);
    let observed_area = envelope_area(observed_min, observed_max);
    let raw_camera_radius_multiplier = if source_area > 1.0e-5 && observed_area > 1.0e-5 {
        (observed_area / source_area).sqrt().clamp(0.90, 1.10)
    } else {
        1.0
    };
    let camera_radius_multiplier = if camera_ray_contact_count > 0 {
        let mean_contact = camera_ray_contact_sum / camera_ray_contact_count as f64;
        if mean_contact > 0.05 {
            1.0
        } else {
            (1.0 + (raw_camera_radius_multiplier - 1.0) * 0.25).clamp(0.97, 1.03)
        }
    } else {
        raw_camera_radius_multiplier
    };
    let mut object_deltas = objects
        .iter()
        .map(|object| {
            let group_key = feedback_scale_group_key(object);
            let grouped_scale = group_key
                .as_ref()
                .and_then(|key| repeated_scale_by_group.get(key))
                .copied();
            let object_scale = object
                .get("scale_multiplier")
                .and_then(Value::as_f64)
                .filter(|value| value.is_finite())
                .unwrap_or(1.0)
                .clamp(0.82, 1.22);
            let object_scale = damped_feedback_scale_multiplier(object, object_scale);
            let axis_scale = feedback_axis_scale_multiplier(object);
            FeedbackDeltaDraft {
                index: object.get("index").cloned().unwrap_or(Value::Null),
                translation_delta: object
                    .get("translation_delta")
                    .and_then(json_array3)
                    .unwrap_or([0.0, 0.0, 0.0]),
                scale_multiplier: if axis_scale.is_some() {
                    1.0
                } else {
                    grouped_scale.unwrap_or(object_scale)
                },
                scale_multiplier_xyz: axis_scale,
                scale_group_key: group_key,
                scale_source: if grouped_scale.is_some() {
                    "repeated_instance_group"
                } else if axis_scale.is_some() {
                    "axis_projection"
                } else {
                    "object_projection"
                },
                yaw_delta_degrees: object
                    .get("yaw_delta_degrees")
                    .cloned()
                    .unwrap_or(json!(0.0)),
            }
        })
        .collect::<Vec<_>>();
    let thresholds = feedback_thresholds_from_metrics(metrics);
    feedback_project_delta_collisions(&objects, &mut object_deltas, thresholds);
    json!({
        "objects": object_deltas.into_iter().map(|delta| {
            json!({
                "index": delta.index,
                "translation_delta": delta.translation_delta,
                "scale_multiplier": delta.scale_multiplier,
                "scale_multiplier_xyz": delta.scale_multiplier_xyz,
                "scale_group_key": delta.scale_group_key,
                "scale_source": delta.scale_source,
                "yaw_delta_degrees": delta.yaw_delta_degrees,
            })
        }).collect::<Vec<_>>(),
        "camera": {
            "radius_multiplier": camera_radius_multiplier,
        }
    })
}

#[derive(Debug)]
struct FeedbackDeltaDraft {
    index: Value,
    translation_delta: [f32; 3],
    scale_multiplier: f64,
    scale_multiplier_xyz: Option<[f64; 3]>,
    scale_group_key: Option<String>,
    scale_source: &'static str,
    yaw_delta_degrees: Value,
}

fn feedback_axis_scale_multiplier(object: &Value) -> Option<[f64; 3]> {
    if !feedback_json_object_is_table_like(object) {
        return None;
    }
    let expected = object.get("expected_bbox").and_then(json_array4)?;
    let observed = object.get("observed_bbox").and_then(json_array4)?;
    let expected_width = (expected[2] - expected[0]).abs().max(1.0e-5) as f64;
    let expected_height = (expected[3] - expected[1]).abs().max(1.0e-5) as f64;
    let observed_width = (observed[2] - observed[0]).abs().max(1.0e-5) as f64;
    let observed_height = (observed[3] - observed[1]).abs().max(1.0e-5) as f64;
    let width_multiplier =
        damped_axis_scale_ratio(expected_width / observed_width, 0.34, 0.84, 1.22);
    let depth_multiplier =
        damped_axis_scale_ratio(expected_height / observed_height, 0.22, 0.90, 1.12);
    Some([width_multiplier, 1.0, depth_multiplier])
}

fn damped_axis_scale_ratio(ratio: f64, weight: f64, min_value: f64, max_value: f64) -> f64 {
    let ratio = ratio.clamp(0.45, 2.40);
    (1.0 + (ratio - 1.0) * weight).clamp(min_value, max_value)
}

fn feedback_thresholds_from_metrics(metrics: &Value) -> SceneFeedbackThresholds {
    let defaults = FeedbackThresholdProfile::Standard.thresholds();
    let Some(thresholds) = metrics.get("thresholds") else {
        return defaults;
    };
    SceneFeedbackThresholds {
        max_center_error: thresholds
            .get("max_center_error")
            .and_then(Value::as_f64)
            .map(|value| value as f32)
            .unwrap_or(defaults.max_center_error),
        max_contact_error: thresholds
            .get("max_contact_error")
            .and_then(Value::as_f64)
            .map(|value| value as f32)
            .unwrap_or(defaults.max_contact_error),
        max_area_log2_error: thresholds
            .get("max_area_log2_error")
            .and_then(Value::as_f64)
            .map(|value| value as f32)
            .unwrap_or(defaults.max_area_log2_error),
        min_overall_score: thresholds
            .get("min_overall_score")
            .and_then(Value::as_f64)
            .map(|value| value as f32)
            .unwrap_or(defaults.min_overall_score),
        max_seating_table_overlap_fraction: thresholds
            .get("max_seating_table_overlap_fraction")
            .and_then(Value::as_f64)
            .map(|value| value as f32)
            .unwrap_or(defaults.max_seating_table_overlap_fraction),
        max_seating_table_penetration_m: thresholds
            .get("max_seating_table_penetration_m")
            .and_then(Value::as_f64)
            .map(|value| value as f32)
            .unwrap_or(defaults.max_seating_table_penetration_m),
        max_seating_seating_overlap_fraction: thresholds
            .get("max_seating_seating_overlap_fraction")
            .and_then(Value::as_f64)
            .map(|value| value as f32)
            .unwrap_or(defaults.max_seating_seating_overlap_fraction),
        max_seating_seating_penetration_m: thresholds
            .get("max_seating_seating_penetration_m")
            .and_then(Value::as_f64)
            .map(|value| value as f32)
            .unwrap_or(defaults.max_seating_seating_penetration_m),
    }
}

fn feedback_project_delta_collisions(
    objects: &[Value],
    deltas: &mut [FeedbackDeltaDraft],
    thresholds: SceneFeedbackThresholds,
) {
    let mut footprints = objects
        .iter()
        .enumerate()
        .map(|(index, object)| {
            let rect = object
                .get("world_footprint")
                .and_then(json_footprint_rect)?;
            let kind = object
                .get("physical_kind")
                .and_then(Value::as_str)
                .and_then(feedback_physical_kind_from_str)
                .unwrap_or(FeedbackPhysicalKind::Other);
            Some(FeedbackFootprint {
                index,
                kind,
                rect: rect.translated(
                    deltas
                        .get(index)
                        .map(|delta| delta.translation_delta)
                        .unwrap_or([0.0, 0.0, 0.0]),
                ),
            })
        })
        .collect::<Vec<_>>();

    for _ in 0..4 {
        let mut changed = false;
        for left_index in 0..footprints.len() {
            let Some(left) = footprints[left_index] else {
                continue;
            };
            for right_index in (left_index + 1)..footprints.len() {
                let Some(right) = footprints[right_index] else {
                    continue;
                };
                let overlap_area = left.rect.overlap_area(right.rect);
                let signed_clearance_m = left.rect.signed_clearance(right.rect);
                if overlap_area <= 1.0e-5 && signed_clearance_m >= 0.0 {
                    continue;
                }
                let smaller_area = left.rect.area().min(right.rect.area()).max(1.0e-8);
                let overlap_fraction_smaller = (overlap_area / smaller_area).clamp(0.0, 1.0);
                match feedback_pair_relationship(left.kind, right.kind) {
                    "seating_table" => {
                        let (table, seating) = if left.kind == FeedbackPhysicalKind::Table {
                            (left, right)
                        } else {
                            (right, left)
                        };
                        let seating_center_inside_table =
                            table.rect.contains_point(seating.rect.center());
                        if seating_center_inside_table
                            || overlap_fraction_smaller
                                > thresholds.max_seating_table_overlap_fraction
                            || signed_clearance_m < -thresholds.max_seating_table_penetration_m
                        {
                            let source_bbox = objects
                                .get(seating.index)
                                .and_then(|object| object.get("expected_bbox"))
                                .and_then(json_array4)
                                .unwrap_or([0.0, 0.0, 1.0, 1.0]);
                            let delta = seating_table_outward_delta(table, seating, source_bbox);
                            if apply_projected_delta(
                                deltas,
                                &mut footprints,
                                seating.index,
                                delta,
                                1.25,
                            ) {
                                changed = true;
                            }
                        }
                    }
                    "seating_seating"
                        if overlap_fraction_smaller
                            > thresholds.max_seating_seating_overlap_fraction
                            || signed_clearance_m
                                < -thresholds.max_seating_seating_penetration_m =>
                    {
                        let [left_delta, right_delta] =
                            seating_pair_separation_delta(left, right, signed_clearance_m);
                        let left_changed = apply_projected_delta(
                            deltas,
                            &mut footprints,
                            left.index,
                            left_delta,
                            1.25,
                        );
                        let right_changed = apply_projected_delta(
                            deltas,
                            &mut footprints,
                            right.index,
                            right_delta,
                            1.25,
                        );
                        changed |= left_changed || right_changed;
                    }
                    _ => {}
                }
            }
        }
        if !changed {
            break;
        }
    }
}

fn apply_projected_delta(
    deltas: &mut [FeedbackDeltaDraft],
    footprints: &mut [Option<FeedbackFootprint>],
    index: usize,
    correction: [f32; 3],
    max_len: f32,
) -> bool {
    if correction[0].abs() + correction[2].abs() <= 1.0e-5 {
        return false;
    }
    let Some(delta) = deltas.get_mut(index) else {
        return false;
    };
    let next_delta = clamp_xz_delta(add3(delta.translation_delta, correction), max_len);
    let applied = [
        next_delta[0] - delta.translation_delta[0],
        next_delta[1] - delta.translation_delta[1],
        next_delta[2] - delta.translation_delta[2],
    ];
    if applied[0].abs() + applied[2].abs() <= 1.0e-5 {
        return false;
    }
    delta.translation_delta = next_delta;
    if let Some(Some(footprint)) = footprints.get_mut(index) {
        footprint.rect = footprint.rect.translated(applied);
    }
    true
}

fn json_footprint_rect(value: &Value) -> Option<FootprintRect> {
    let rect = FootprintRect {
        min_x: value.get("min_x")?.as_f64()? as f32,
        min_z: value.get("min_z")?.as_f64()? as f32,
        max_x: value.get("max_x")?.as_f64()? as f32,
        max_z: value.get("max_z")?.as_f64()? as f32,
    };
    if rect.min_x.is_finite()
        && rect.min_z.is_finite()
        && rect.max_x.is_finite()
        && rect.max_z.is_finite()
        && rect.width() > 1.0e-4
        && rect.depth() > 1.0e-4
    {
        Some(rect)
    } else {
        None
    }
}

fn damped_feedback_scale_multiplier(object: &Value, raw_scale: f64) -> f64 {
    if feedback_json_object_is_table_like(object) {
        return 1.0;
    }
    let raw_scale = raw_scale.clamp(0.82, 1.22);
    if !object
        .get("grounding_basis")
        .and_then(Value::as_str)
        .is_some_and(|basis| basis == "camera-ray-ground-plane")
    {
        return raw_scale;
    }
    let contact_error = object
        .get("contact_error")
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
        .unwrap_or(1.0);
    let center_error = object
        .get("center_error")
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
        .unwrap_or(1.0);
    let weight = if contact_error <= 0.05 && center_error <= 0.08 {
        1.0
    } else if contact_error <= 0.10 && center_error <= 0.16 {
        0.55
    } else {
        0.25
    };
    (1.0 + (raw_scale - 1.0) * weight).clamp(0.88, 1.12)
}

fn feedback_json_object_is_table_like(object: &Value) -> bool {
    let descriptor = format!(
        "{} {}",
        object
            .get("object_id")
            .and_then(Value::as_str)
            .unwrap_or(""),
        object.get("label").and_then(Value::as_str).unwrap_or("")
    )
    .to_lowercase();
    descriptor.contains("table") || descriptor.contains("desk") || descriptor.contains("counter")
}

fn feedback_scale_group_key(object: &Value) -> Option<String> {
    object
        .get("cache_key")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            object
                .get("object_id")
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty())
        })
        .map(ToString::to_string)
}

fn apply_feedback_deltas_to_commands(
    commands: &[Value],
    deltas: &Value,
) -> Result<Vec<Value>, String> {
    let object_deltas = deltas
        .get("objects")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut out = commands.to_vec();
    let mut spawn_index = 0usize;
    for command in &mut out {
        let command_type = command.get("type").and_then(Value::as_str).unwrap_or("");
        if command_type == "spawn_cached" || command_type == "spawn_path" {
            if let Some(delta) = object_deltas.get(spawn_index) {
                apply_object_delta_to_command(command, delta)?;
            }
            spawn_index += 1;
        } else if command_type == "set_camera" {
            let radius_multiplier = deltas
                .get("camera")
                .and_then(|camera| camera.get("radius_multiplier"))
                .and_then(Value::as_f64)
                .unwrap_or(1.0) as f32;
            if let Some(radius) = command.get("radius").and_then(Value::as_f64) {
                command["radius"] = json!((radius as f32 * radius_multiplier).clamp(1.0, 20.0));
            }
        }
    }
    normalize_reused_command_scales(&mut out);
    Ok(out)
}

fn normalize_reused_command_scales(commands: &mut [Value]) {
    let mut groups: HashMap<String, (f32, usize)> = HashMap::new();
    for command in commands.iter() {
        let command_type = command.get("type").and_then(Value::as_str).unwrap_or("");
        if command_type != "spawn_cached" && command_type != "spawn_path" {
            continue;
        }
        let Some(group_key) = command
            .get("cache_key")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
        else {
            continue;
        };
        let Some(scale) = command.get("scale").and_then(json_array3) else {
            continue;
        };
        let uniform_scale =
            ((scale[0].abs() + scale[1].abs() + scale[2].abs()) / 3.0).clamp(0.05, 20.0);
        let entry = groups.entry(group_key.to_string()).or_insert((0.0, 0));
        entry.0 += uniform_scale;
        entry.1 += 1;
    }
    let repeated_scale = groups
        .into_iter()
        .filter_map(|(key, (sum, count))| {
            if count > 1 {
                Some((key, (sum / count as f32).clamp(0.05, 20.0)))
            } else {
                None
            }
        })
        .collect::<HashMap<_, _>>();
    for command in commands.iter_mut() {
        let command_type = command.get("type").and_then(Value::as_str).unwrap_or("");
        if command_type != "spawn_cached" && command_type != "spawn_path" {
            continue;
        }
        let Some(group_key) = command
            .get("cache_key")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
        else {
            continue;
        };
        let Some(scale) = repeated_scale.get(group_key).copied() else {
            continue;
        };
        command["scale"] = json!([scale, scale, scale]);
    }
}

fn apply_object_delta_to_command(command: &mut Value, delta: &Value) -> Result<(), String> {
    let mut translation = command
        .get("translation")
        .and_then(json_array3)
        .unwrap_or([0.0, 0.0, 0.0]);
    let translation_delta = delta
        .get("translation_delta")
        .and_then(json_array3)
        .unwrap_or([0.0, 0.0, 0.0]);
    translation[0] += translation_delta[0];
    translation[1] += translation_delta[1];
    translation[2] += translation_delta[2];
    command["translation"] = json!(translation);

    let multiplier = delta
        .get("scale_multiplier")
        .and_then(Value::as_f64)
        .unwrap_or(1.0) as f32;
    let axis_multiplier = delta.get("scale_multiplier_xyz").and_then(json_array3);
    let mut scale = command
        .get("scale")
        .and_then(json_array3)
        .unwrap_or([1.0, 1.0, 1.0]);
    if let Some(axis_multiplier) = axis_multiplier {
        for (value, axis_multiplier) in scale.iter_mut().zip(axis_multiplier) {
            *value = (*value * axis_multiplier).clamp(0.05, 20.0);
        }
    } else {
        for value in &mut scale {
            *value = (*value * multiplier).clamp(0.05, 20.0);
        }
    }
    command["scale"] = json!(scale);

    let yaw_delta = delta
        .get("yaw_delta_degrees")
        .and_then(Value::as_f64)
        .unwrap_or(0.0) as f32;
    if yaw_delta.abs() > 1.0e-4 {
        let current_yaw = command
            .get("rotation")
            .and_then(json_array4)
            .map(quat_y_degrees)
            .unwrap_or(0.0);
        let target_yaw = normalize_degrees(current_yaw + yaw_delta.clamp(-30.0, 30.0));
        command["rotation"] = json!(quat_from_y_degrees(target_yaw));
    }
    Ok(())
}

fn feedback_bsn_from_commands(
    asset_bindings: &[SceneAssetBinding],
    grounded_layout: &GroundedSceneLayout,
    commands: &[Value],
) -> Result<String, String> {
    let mut out = String::from("synth_scene_v1 {\n");
    for asset in asset_bindings {
        out.push_str(&format!(
            "asset {} = \"generated:{}\";\n",
            asset.asset_id, asset.asset_id
        ));
    }
    out.push_str(&format!(
        "environment rug translation [{}] scale [{}] color [0.62,0.02,0.26];\n",
        fmt_feedback_vec3(grounded_layout.rug_center),
        fmt_feedback_vec3(grounded_layout.rug_scale)
    ));
    let mut spawn_index = 0usize;
    for command in commands {
        let command_type = command.get("type").and_then(Value::as_str).unwrap_or("");
        if command_type == "spawn_cached" || command_type == "spawn_path" {
            let asset_id = command_asset_id(asset_bindings, command)?;
            let translation = command
                .get("translation")
                .and_then(json_array3)
                .unwrap_or([0.0, 0.0, 0.0]);
            let scale = command
                .get("scale")
                .and_then(json_array3)
                .unwrap_or([1.0, 1.0, 1.0]);
            let rotation_y = command
                .get("rotation")
                .and_then(json_array4)
                .map(quat_y_degrees)
                .unwrap_or(0.0);
            let entity_id = grounded_layout
                .placements
                .get(spawn_index)
                .map(|placement| placement.entity_id.as_str())
                .unwrap_or("feedback_item");
            out.push_str(&format!(
                "spawn {} uses {} translation [{}] rotation_y {} scale [{}];\n",
                entity_id,
                asset_id,
                fmt_feedback_vec3(translation),
                fmt_feedback_num(rotation_y),
                fmt_feedback_vec3(scale)
            ));
            spawn_index += 1;
        } else if command_type == "set_camera" {
            let translation = command
                .get("translation")
                .and_then(json_array3)
                .unwrap_or(grounded_layout.camera.translation);
            let focus = command
                .get("focus")
                .and_then(json_array3)
                .unwrap_or(grounded_layout.camera.focus);
            let yaw = command
                .get("yaw")
                .and_then(Value::as_f64)
                .map(|value| value as f32)
                .or(grounded_layout.camera.yaw)
                .unwrap_or(0.0);
            let pitch = command
                .get("pitch")
                .and_then(Value::as_f64)
                .map(|value| value as f32)
                .or(grounded_layout.camera.pitch)
                .unwrap_or(30.0);
            let radius = command
                .get("radius")
                .and_then(Value::as_f64)
                .map(|value| value as f32)
                .or(grounded_layout.camera.radius)
                .unwrap_or(5.0);
            let vertical_fov = command
                .get("vertical_fov")
                .and_then(Value::as_f64)
                .map(|value| value as f32)
                .or(grounded_layout.camera.vertical_fov_degrees)
                .unwrap_or(72.0);
            out.push_str(&format!(
                "camera translation [{}] focus [{}] yaw {} pitch {} radius {} vertical_fov {};\n",
                fmt_feedback_vec3(translation),
                fmt_feedback_vec3(focus),
                fmt_feedback_num(yaw),
                fmt_feedback_num(pitch),
                fmt_feedback_num(radius),
                fmt_feedback_num(vertical_fov)
            ));
        }
    }
    out.push_str("}\n");
    Ok(out)
}

fn command_asset_id<'a>(
    assets: &'a [SceneAssetBinding],
    command: &Value,
) -> Result<&'a str, String> {
    if let Some(cache_key) = command.get("cache_key").and_then(Value::as_str)
        && let Some(asset) = assets
            .iter()
            .find(|asset| asset.cache_key.as_deref() == Some(cache_key))
    {
        return Ok(asset.asset_id.as_str());
    }
    if let Some(path) = command.get("path").and_then(Value::as_str)
        && let Some(asset) = assets.iter().find(|asset| {
            asset
                .path
                .as_ref()
                .is_some_and(|asset_path| asset_path == path)
        })
    {
        return Ok(asset.asset_id.as_str());
    }
    if let Some(cache_key) = command.get("cache_key").and_then(Value::as_str)
        && let Some(asset) = assets.iter().find(|asset| asset.asset_id == cache_key)
    {
        return Ok(asset.asset_id.as_str());
    }
    Err("feedback command references an unknown asset".to_string())
}

fn feedback_markdown_report(
    capture_root: &Path,
    profile: FeedbackThresholdProfile,
    accepted_iteration: Option<usize>,
    iterations: &[Value],
) -> String {
    let mut out = format!(
        "# Scene Feedback Report\n\nprofile: {:?}\naccepted_iteration: {:?}\n\n",
        profile, accepted_iteration
    );
    for iteration in iterations {
        let index = iteration
            .get("iteration")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let screenshot = iteration
            .get("screenshot")
            .and_then(Value::as_str)
            .unwrap_or("");
        let score = iteration
            .get("metrics")
            .and_then(|metrics| metrics.get("score"))
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        let selection_score = iteration
            .get("selection_score")
            .and_then(Value::as_f64)
            .unwrap_or(score);
        let passed = iteration
            .get("passed")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let physical = iteration
            .get("metrics")
            .and_then(|metrics| metrics.get("physical_layout"))
            .unwrap_or(&Value::Null);
        let physical_passed = physical
            .get("passed")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let hard_failures = physical
            .get("hard_failure_count")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let max_overlap = physical
            .get("max_overlap_fraction_smaller")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        let min_clearance = physical
            .get("min_signed_clearance_m")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        out.push_str(&format!(
            "## Iteration {index}\n\npassed: {passed}\nscore: {score:.4}\nselection_score: {selection_score:.4}\nphysical_passed: {physical_passed}\nhard_overlap_failures: {hard_failures}\nmax_overlap_fraction_smaller: {max_overlap:.4}\nmin_signed_clearance_m: {min_clearance:.4}\n\n![iteration {index}]({})\n\n",
            path_relative_to(capture_root, Path::new(screenshot))
        ));
        if let Some(pairs) = physical.get("pairs").and_then(Value::as_array) {
            let failing_pairs = pairs
                .iter()
                .filter(|pair| {
                    pair.get("hard_failure")
                        .and_then(Value::as_bool)
                        .unwrap_or(false)
                })
                .take(6)
                .collect::<Vec<_>>();
            if !failing_pairs.is_empty() {
                out.push_str(
                    "| Pair | Relationship | Overlap Fraction | Clearance m | Reasons |\n",
                );
                out.push_str("| --- | --- | ---: | ---: | --- |\n");
                for pair in failing_pairs {
                    let left = pair
                        .get("left_instance_id")
                        .and_then(Value::as_str)
                        .or_else(|| pair.get("left_object_id").and_then(Value::as_str))
                        .unwrap_or("left");
                    let right = pair
                        .get("right_instance_id")
                        .and_then(Value::as_str)
                        .or_else(|| pair.get("right_object_id").and_then(Value::as_str))
                        .unwrap_or("right");
                    let relationship = pair
                        .get("relationship")
                        .and_then(Value::as_str)
                        .unwrap_or("object_object");
                    let fraction = pair
                        .get("overlap_fraction_smaller")
                        .and_then(Value::as_f64)
                        .unwrap_or(0.0);
                    let clearance = pair
                        .get("signed_clearance_m")
                        .and_then(Value::as_f64)
                        .unwrap_or(0.0);
                    let reasons = pair
                        .get("failure_reasons")
                        .and_then(Value::as_array)
                        .map(|items| {
                            items
                                .iter()
                                .filter_map(Value::as_str)
                                .collect::<Vec<_>>()
                                .join(", ")
                        })
                        .unwrap_or_default();
                    out.push_str(&format!(
                        "| {left} / {right} | {relationship} | {fraction:.4} | {clearance:.4} | {reasons} |\n"
                    ));
                }
                out.push('\n');
            }
        }
    }
    out
}

fn feedback_selection_score(metrics: &Value) -> f64 {
    let score = metrics
        .get("score")
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
        .unwrap_or(f64::NEG_INFINITY);
    let physical = metrics.get("physical_layout").unwrap_or(&Value::Null);
    let hard_failures = physical
        .get("hard_failure_count")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let max_overlap = physical
        .get("max_overlap_fraction_smaller")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    score - hard_failures * 2.0 - max_overlap
}

fn path_relative_to(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .display()
        .to_string()
}

#[derive(Clone, Copy, Debug)]
struct FeedbackFootprint {
    index: usize,
    kind: FeedbackPhysicalKind,
    rect: FootprintRect,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FeedbackPhysicalKind {
    Table,
    Seating,
    Other,
}

#[derive(Clone, Copy, Debug)]
struct FootprintRect {
    min_x: f32,
    min_z: f32,
    max_x: f32,
    max_z: f32,
}

impl FootprintRect {
    fn from_aabb(min: [f32; 3], max: [f32; 3]) -> Option<Self> {
        let rect = Self {
            min_x: min[0].min(max[0]),
            min_z: min[2].min(max[2]),
            max_x: min[0].max(max[0]),
            max_z: min[2].max(max[2]),
        };
        if rect.min_x.is_finite()
            && rect.min_z.is_finite()
            && rect.max_x.is_finite()
            && rect.max_z.is_finite()
            && rect.width() > 1.0e-4
            && rect.depth() > 1.0e-4
        {
            Some(rect)
        } else {
            None
        }
    }

    fn width(self) -> f32 {
        (self.max_x - self.min_x).max(0.0)
    }

    fn depth(self) -> f32 {
        (self.max_z - self.min_z).max(0.0)
    }

    fn area(self) -> f32 {
        (self.width() * self.depth()).max(1.0e-8)
    }

    fn center(self) -> [f32; 2] {
        [
            (self.min_x + self.max_x) * 0.5,
            (self.min_z + self.max_z) * 0.5,
        ]
    }

    fn half_width(self) -> f32 {
        self.width() * 0.5
    }

    fn half_depth(self) -> f32 {
        self.depth() * 0.5
    }

    fn contains_point(self, point: [f32; 2]) -> bool {
        point[0] >= self.min_x
            && point[0] <= self.max_x
            && point[1] >= self.min_z
            && point[1] <= self.max_z
    }

    fn overlap_extents(self, other: Self) -> [f32; 2] {
        [
            (self.max_x.min(other.max_x) - self.min_x.max(other.min_x)).max(0.0),
            (self.max_z.min(other.max_z) - self.min_z.max(other.min_z)).max(0.0),
        ]
    }

    fn overlap_area(self, other: Self) -> f32 {
        let [x, z] = self.overlap_extents(other);
        x * z
    }

    fn signed_clearance(self, other: Self) -> f32 {
        let [overlap_x, overlap_z] = self.overlap_extents(other);
        if overlap_x > 0.0 && overlap_z > 0.0 {
            return -overlap_x.min(overlap_z);
        }
        let dx = if self.max_x < other.min_x {
            other.min_x - self.max_x
        } else if other.max_x < self.min_x {
            self.min_x - other.max_x
        } else {
            0.0
        };
        let dz = if self.max_z < other.min_z {
            other.min_z - self.max_z
        } else if other.max_z < self.min_z {
            self.min_z - other.max_z
        } else {
            0.0
        };
        (dx * dx + dz * dz).sqrt()
    }

    fn translated(self, delta: [f32; 3]) -> Self {
        Self {
            min_x: self.min_x + delta[0],
            max_x: self.max_x + delta[0],
            min_z: self.min_z + delta[2],
            max_z: self.max_z + delta[2],
        }
    }
}

#[derive(Debug)]
struct FeedbackPhysicalLayout {
    pairs: Vec<Value>,
    corrections: HashMap<usize, [f32; 3]>,
    object_failures: HashMap<usize, Vec<String>>,
    hard_failure_count: usize,
    warning_count: usize,
    object_failure_count: usize,
    max_overlap_fraction_smaller: f32,
    min_signed_clearance_m: f32,
    table_center_xz: Option<[f32; 2]>,
    footprint_centers: HashMap<usize, [f32; 2]>,
}

fn feedback_projected_footprints(
    placements: &[GroundedScenePlacement],
    projected: &[Value],
) -> Vec<Option<FeedbackFootprint>> {
    placements
        .iter()
        .enumerate()
        .map(|(index, placement)| {
            let aabb = projected.get(index)?.get("world_aabb")?;
            let min = aabb.get("min").and_then(json_array3)?;
            let max = aabb.get("max").and_then(json_array3)?;
            Some(FeedbackFootprint {
                index,
                kind: feedback_physical_kind(placement),
                rect: FootprintRect::from_aabb(min, max)?,
            })
        })
        .collect()
}

fn feedback_physical_layout(
    placements: &[GroundedScenePlacement],
    footprints: &[Option<FeedbackFootprint>],
    thresholds: SceneFeedbackThresholds,
) -> FeedbackPhysicalLayout {
    let mut pairs = Vec::new();
    let mut corrections: HashMap<usize, [f32; 3]> = HashMap::new();
    let mut object_failures: HashMap<usize, Vec<String>> = HashMap::new();
    let mut hard_failure_count = 0usize;
    let mut warning_count = 0usize;
    let mut max_overlap_fraction_smaller = 0.0f32;
    let mut min_signed_clearance_m = f32::INFINITY;
    let table_center_xz = footprints
        .iter()
        .flatten()
        .find(|footprint| footprint.kind == FeedbackPhysicalKind::Table)
        .map(|footprint| footprint.rect.center());
    let footprint_centers = footprints
        .iter()
        .flatten()
        .map(|footprint| (footprint.index, footprint.rect.center()))
        .collect::<HashMap<_, _>>();

    for left_index in 0..footprints.len() {
        let Some(left) = footprints[left_index] else {
            continue;
        };
        for right in footprints.iter().skip(left_index + 1) {
            let Some(right) = *right else {
                continue;
            };
            let overlap_area = left.rect.overlap_area(right.rect);
            let signed_clearance_m = left.rect.signed_clearance(right.rect);
            min_signed_clearance_m = min_signed_clearance_m.min(signed_clearance_m);
            if overlap_area <= 1.0e-5 && signed_clearance_m >= 0.0 {
                continue;
            }
            let smaller_area = left.rect.area().min(right.rect.area()).max(1.0e-8);
            let overlap_fraction_smaller = (overlap_area / smaller_area).clamp(0.0, 1.0);
            max_overlap_fraction_smaller =
                max_overlap_fraction_smaller.max(overlap_fraction_smaller);
            let relationship = feedback_pair_relationship(left.kind, right.kind);
            let seating_table = relationship == "seating_table";
            let seating_seating = relationship == "seating_seating";
            let mut reasons = Vec::new();
            if seating_table {
                let (table, seating) = if left.kind == FeedbackPhysicalKind::Table {
                    (left, right)
                } else {
                    (right, left)
                };
                let seating_center_inside_table = table.rect.contains_point(seating.rect.center());
                if seating_center_inside_table {
                    reasons.push("seating_center_inside_table");
                }
                if overlap_fraction_smaller > thresholds.max_seating_table_overlap_fraction {
                    reasons.push("seating_table_overlap_fraction");
                }
                if signed_clearance_m < -thresholds.max_seating_table_penetration_m {
                    reasons.push("seating_table_penetration");
                }
                if !reasons.is_empty() {
                    let delta = seating_table_outward_delta(
                        table,
                        seating,
                        placements[seating.index].source_bbox,
                    );
                    accumulate_feedback_delta(&mut corrections, seating.index, delta, 1.10);
                }
            } else if seating_seating {
                if overlap_fraction_smaller > thresholds.max_seating_seating_overlap_fraction {
                    reasons.push("seating_seating_overlap_fraction");
                }
                if signed_clearance_m < -thresholds.max_seating_seating_penetration_m {
                    reasons.push("seating_seating_penetration");
                }
                if !reasons.is_empty() {
                    let [left_delta, right_delta] =
                        seating_pair_separation_delta(left, right, signed_clearance_m);
                    accumulate_feedback_delta(&mut corrections, left.index, left_delta, 0.55);
                    accumulate_feedback_delta(&mut corrections, right.index, right_delta, 0.55);
                }
            }

            let hard_failure = !reasons.is_empty();
            if hard_failure {
                hard_failure_count += 1;
                let left_message = feedback_physical_failure_message(
                    relationship,
                    &placements[right.index],
                    overlap_fraction_smaller,
                    signed_clearance_m,
                    &reasons,
                );
                let right_message = feedback_physical_failure_message(
                    relationship,
                    &placements[left.index],
                    overlap_fraction_smaller,
                    signed_clearance_m,
                    &reasons,
                );
                object_failures
                    .entry(left.index)
                    .or_default()
                    .push(left_message);
                object_failures
                    .entry(right.index)
                    .or_default()
                    .push(right_message);
            } else {
                warning_count += 1;
            }

            pairs.push(json!({
                "left_index": left.index,
                "right_index": right.index,
                "left_object_id": placements[left.index].object_id,
                "right_object_id": placements[right.index].object_id,
                "left_instance_id": placements[left.index].instance_id,
                "right_instance_id": placements[right.index].instance_id,
                "relationship": relationship,
                "overlap_area": overlap_area,
                "overlap_fraction_smaller": overlap_fraction_smaller,
                "signed_clearance_m": signed_clearance_m,
                "hard_failure": hard_failure,
                "failure_reasons": reasons,
            }));
        }
    }

    FeedbackPhysicalLayout {
        pairs,
        corrections,
        object_failure_count: object_failures.len(),
        object_failures,
        hard_failure_count,
        warning_count,
        max_overlap_fraction_smaller,
        min_signed_clearance_m: if min_signed_clearance_m.is_finite() {
            min_signed_clearance_m
        } else {
            0.0
        },
        table_center_xz,
        footprint_centers,
    }
}

fn feedback_predictive_physical_delta(
    index: usize,
    placement: &GroundedScenePlacement,
    proposed_delta: [f32; 3],
    footprints: &[Option<FeedbackFootprint>],
    thresholds: SceneFeedbackThresholds,
) -> [f32; 3] {
    let Some(current) = footprints.get(index).and_then(|footprint| *footprint) else {
        return [0.0, 0.0, 0.0];
    };
    if current.kind != FeedbackPhysicalKind::Seating {
        return [0.0, 0.0, 0.0];
    }
    let predicted = FeedbackFootprint {
        rect: current.rect.translated(proposed_delta),
        ..current
    };
    let mut correction = [0.0, 0.0, 0.0];
    for other in footprints.iter().flatten().copied() {
        if other.index == index {
            continue;
        }
        let overlap_area = predicted.rect.overlap_area(other.rect);
        let signed_clearance_m = predicted.rect.signed_clearance(other.rect);
        if overlap_area <= 1.0e-5 && signed_clearance_m >= 0.0 {
            continue;
        }
        let smaller_area = predicted.rect.area().min(other.rect.area()).max(1.0e-8);
        let overlap_fraction_smaller = (overlap_area / smaller_area).clamp(0.0, 1.0);
        match feedback_pair_relationship(predicted.kind, other.kind) {
            "seating_table" => {
                let table = other;
                let seating_center_inside_table =
                    table.rect.contains_point(predicted.rect.center());
                if seating_center_inside_table
                    || overlap_fraction_smaller > thresholds.max_seating_table_overlap_fraction
                    || signed_clearance_m < -thresholds.max_seating_table_penetration_m
                {
                    correction = add3(
                        correction,
                        seating_table_outward_delta(table, predicted, placement.source_bbox),
                    );
                }
            }
            "seating_seating"
                if overlap_fraction_smaller > thresholds.max_seating_seating_overlap_fraction
                    || signed_clearance_m < -thresholds.max_seating_seating_penetration_m =>
            {
                let [self_delta, _other_delta] =
                    seating_pair_separation_delta(predicted, other, signed_clearance_m);
                correction = add3(correction, self_delta);
            }
            _ => {}
        }
    }
    clamp_xz_delta(correction, 0.95)
}

fn feedback_physical_kind(placement: &GroundedScenePlacement) -> FeedbackPhysicalKind {
    let descriptor = format!("{} {}", placement.object_id, placement.label).to_lowercase();
    if descriptor.contains("table") || descriptor.contains("desk") || descriptor.contains("counter")
    {
        FeedbackPhysicalKind::Table
    } else if descriptor.contains("chair")
        || descriptor.contains("seat")
        || descriptor.contains("sofa")
        || descriptor.contains("couch")
        || descriptor.contains("stool")
    {
        FeedbackPhysicalKind::Seating
    } else {
        FeedbackPhysicalKind::Other
    }
}

fn feedback_physical_kind_str(kind: FeedbackPhysicalKind) -> &'static str {
    match kind {
        FeedbackPhysicalKind::Table => "table",
        FeedbackPhysicalKind::Seating => "seating",
        FeedbackPhysicalKind::Other => "other",
    }
}

fn feedback_physical_kind_from_str(value: &str) -> Option<FeedbackPhysicalKind> {
    match value {
        "table" => Some(FeedbackPhysicalKind::Table),
        "seating" => Some(FeedbackPhysicalKind::Seating),
        "other" => Some(FeedbackPhysicalKind::Other),
        _ => None,
    }
}

fn feedback_pair_relationship(
    left: FeedbackPhysicalKind,
    right: FeedbackPhysicalKind,
) -> &'static str {
    match (left, right) {
        (FeedbackPhysicalKind::Table, FeedbackPhysicalKind::Seating)
        | (FeedbackPhysicalKind::Seating, FeedbackPhysicalKind::Table) => "seating_table",
        (FeedbackPhysicalKind::Seating, FeedbackPhysicalKind::Seating) => "seating_seating",
        (FeedbackPhysicalKind::Table, _) | (_, FeedbackPhysicalKind::Table) => "table_object",
        _ => "object_object",
    }
}

fn seating_table_outward_delta(
    table: FeedbackFootprint,
    seating: FeedbackFootprint,
    source_bbox: [f32; 4],
) -> [f32; 3] {
    let table_center = table.rect.center();
    let seating_center = seating.rect.center();
    let norm_x = (seating_center[0] - table_center[0]) / table.rect.half_width().max(1.0e-5);
    let norm_z = (seating_center[1] - table_center[1]) / table.rect.half_depth().max(1.0e-5);
    let use_x_axis = if norm_x.abs() + norm_z.abs() <= 1.0e-4 {
        true
    } else {
        norm_x.abs() >= norm_z.abs()
    };
    let clearance = 0.12;
    if use_x_axis {
        let sign = if norm_x.abs() <= 1.0e-4 {
            if bbox_center(source_bbox)[0] < 0.5 {
                -1.0
            } else {
                1.0
            }
        } else {
            norm_x.signum()
        };
        let target = if sign >= 0.0 {
            table.rect.max_x + seating.rect.half_width() + clearance
        } else {
            table.rect.min_x - seating.rect.half_width() - clearance
        };
        [(target - seating_center[0]).clamp(-0.90, 0.90), 0.0, 0.0]
    } else {
        let sign = if norm_z.abs() <= 1.0e-4 {
            if bbox_center(source_bbox)[1] < 0.5 {
                -1.0
            } else {
                1.0
            }
        } else {
            norm_z.signum()
        };
        let target = if sign >= 0.0 {
            table.rect.max_z + seating.rect.half_depth() + clearance
        } else {
            table.rect.min_z - seating.rect.half_depth() - clearance
        };
        [0.0, 0.0, (target - seating_center[1]).clamp(-0.90, 0.90)]
    }
}

fn seating_pair_separation_delta(
    left: FeedbackFootprint,
    right: FeedbackFootprint,
    signed_clearance_m: f32,
) -> [[f32; 3]; 2] {
    let left_center = left.rect.center();
    let right_center = right.rect.center();
    let dx = left_center[0] - right_center[0];
    let dz = left_center[1] - right_center[1];
    let len = (dx * dx + dz * dz).sqrt();
    let direction = if len > 1.0e-5 {
        [dx / len, dz / len]
    } else if left.index <= right.index {
        [-1.0, 0.0]
    } else {
        [1.0, 0.0]
    };
    let step = ((-signed_clearance_m).max(0.0) + 0.08).clamp(0.04, 0.32) * 0.5;
    [
        [direction[0] * step, 0.0, direction[1] * step],
        [-direction[0] * step, 0.0, -direction[1] * step],
    ]
}

fn accumulate_feedback_delta(
    corrections: &mut HashMap<usize, [f32; 3]>,
    index: usize,
    delta: [f32; 3],
    max_len: f32,
) {
    let current = corrections.entry(index).or_insert([0.0, 0.0, 0.0]);
    *current = clamp_xz_delta(add3(*current, delta), max_len);
}

fn feedback_physical_failure_message(
    relationship: &str,
    other: &GroundedScenePlacement,
    overlap_fraction_smaller: f32,
    signed_clearance_m: f32,
    reasons: &[&'static str],
) -> String {
    format!(
        "{relationship} overlap with {} / {:?}: fraction={overlap_fraction_smaller:.3}, clearance_m={signed_clearance_m:.3}, reasons={}",
        other.object_id,
        other.instance_id,
        reasons.join("|")
    )
}

#[derive(Clone, Copy, Debug)]
struct FeedbackCamera {
    translation: [f32; 3],
    rotation: [f32; 4],
    vertical_fov_degrees: f32,
    aspect: f32,
}

impl FeedbackCamera {
    fn from_status(status: &Value, screenshot_path: &Path) -> Option<Self> {
        let camera = status.get("camera")?;
        let translation = camera.get("translation").and_then(json_array3)?;
        let rotation = camera.get("rotation").and_then(json_array4)?;
        let vertical_fov_degrees = camera
            .get("vertical_fov_degrees")
            .and_then(Value::as_f64)
            .map(|value| value as f32)
            .filter(|value| value.is_finite() && *value > 1.0)
            .unwrap_or(70.0);
        let aspect = image::image_dimensions(screenshot_path)
            .ok()
            .map(|(width, height)| width.max(1) as f32 / height.max(1) as f32)
            .filter(|value| value.is_finite() && *value > 0.1)
            .unwrap_or(16.0 / 9.0);
        Some(Self {
            translation,
            rotation,
            vertical_fov_degrees,
            aspect,
        })
    }

    fn ground_point(&self, screen: [f32; 2], floor_y: f32) -> Option<[f32; 3]> {
        let tan_half = (self.vertical_fov_degrees.to_radians() * 0.5).tan();
        let local = normalize3([
            (2.0 * screen[0].clamp(0.0, 1.0) - 1.0) * self.aspect.max(0.1) * tan_half,
            (1.0 - 2.0 * screen[1].clamp(0.0, 1.0)) * tan_half,
            -1.0,
        ])?;
        let direction = quat_rotate_vec3(self.rotation, local);
        if !direction[1].is_finite() || direction[1].abs() <= 1.0e-5 {
            return None;
        }
        let t = (floor_y - self.translation[1]) / direction[1];
        if !t.is_finite() || t <= 0.0 {
            return None;
        }
        Some([
            self.translation[0] + direction[0] * t,
            floor_y,
            self.translation[2] + direction[2] * t,
        ])
    }
}

fn projected_item_ground_point(projected_item: &Value) -> Option<[f32; 3]> {
    let aabb = projected_item.get("world_aabb")?;
    let min = aabb.get("min").and_then(json_array3)?;
    let max = aabb.get("max").and_then(json_array3)?;
    if !min.iter().all(|value| value.is_finite()) || !max.iter().all(|value| value.is_finite()) {
        return None;
    }
    Some([
        (min[0] + max[0]) * 0.5,
        min[1].min(max[1]),
        (min[2] + max[2]) * 0.5,
    ])
}

fn normalize3(value: [f32; 3]) -> Option<[f32; 3]> {
    let len = (value[0] * value[0] + value[1] * value[1] + value[2] * value[2]).sqrt();
    if !len.is_finite() || len <= 1.0e-8 {
        return None;
    }
    Some([value[0] / len, value[1] / len, value[2] / len])
}

fn quat_rotate_vec3(quat: [f32; 4], vector: [f32; 3]) -> [f32; 3] {
    let q = [quat[0], quat[1], quat[2]];
    let t = scale3(cross3(q, vector), 2.0);
    add3(add3(vector, scale3(t, quat[3])), cross3(q, t))
}

fn cross3(lhs: [f32; 3], rhs: [f32; 3]) -> [f32; 3] {
    [
        lhs[1] * rhs[2] - lhs[2] * rhs[1],
        lhs[2] * rhs[0] - lhs[0] * rhs[2],
        lhs[0] * rhs[1] - lhs[1] * rhs[0],
    ]
}

fn add3(lhs: [f32; 3], rhs: [f32; 3]) -> [f32; 3] {
    [lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]]
}

fn scale3(value: [f32; 3], scalar: f32) -> [f32; 3] {
    [value[0] * scalar, value[1] * scalar, value[2] * scalar]
}

fn clamp_xz_delta(delta: [f32; 3], max_len: f32) -> [f32; 3] {
    let len = (delta[0] * delta[0] + delta[2] * delta[2]).sqrt();
    if !len.is_finite() || len <= max_len.max(1.0e-5) {
        return delta;
    }
    let scale = max_len.max(1.0e-5) / len;
    [delta[0] * scale, delta[1], delta[2] * scale]
}

fn json_array2(value: &Value) -> Option<[f32; 2]> {
    let values = value.as_array()?;
    Some([
        values.first()?.as_f64()? as f32,
        values.get(1)?.as_f64()? as f32,
    ])
}

fn json_array3(value: &Value) -> Option<[f32; 3]> {
    let values = value.as_array()?;
    Some([
        values.first()?.as_f64()? as f32,
        values.get(1)?.as_f64()? as f32,
        values.get(2)?.as_f64()? as f32,
    ])
}

fn json_array4(value: &Value) -> Option<[f32; 4]> {
    let values = value.as_array()?;
    Some([
        values.first()?.as_f64()? as f32,
        values.get(1)?.as_f64()? as f32,
        values.get(2)?.as_f64()? as f32,
        values.get(3)?.as_f64()? as f32,
    ])
}

fn bbox_center(bbox: [f32; 4]) -> [f32; 2] {
    [(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5]
}

fn bbox_area(bbox: [f32; 4]) -> f32 {
    ((bbox[2] - bbox[0]).abs() * (bbox[3] - bbox[1]).abs()).max(1.0e-6)
}

fn bbox_aspect(bbox: [f32; 4]) -> f32 {
    ((bbox[2] - bbox[0]).abs() / (bbox[3] - bbox[1]).abs().max(1.0e-6)).max(1.0e-6)
}

fn distance2(lhs: [f32; 2], rhs: [f32; 2]) -> f32 {
    let dx = lhs[0] - rhs[0];
    let dy = lhs[1] - rhs[1];
    (dx * dx + dy * dy).sqrt()
}

fn vec2_sub(lhs: [f32; 2], rhs: [f32; 2]) -> [f32; 2] {
    [lhs[0] - rhs[0], lhs[1] - rhs[1]]
}

fn safe_log2_ratio(lhs: f32, rhs: f32) -> f32 {
    (lhs.max(1.0e-6) / rhs.max(1.0e-6)).log2()
}

fn expand_bbox_envelope(min: &mut [f32; 2], max: &mut [f32; 2], bbox: [f32; 4]) {
    min[0] = min[0].min(bbox[0]);
    min[1] = min[1].min(bbox[1]);
    max[0] = max[0].max(bbox[2]);
    max[1] = max[1].max(bbox[3]);
}

fn envelope_area(min: [f32; 2], max: [f32; 2]) -> f32 {
    if !min[0].is_finite() || !max[0].is_finite() {
        return 0.0;
    }
    ((max[0] - min[0]).abs() * (max[1] - min[1]).abs()).max(1.0e-6)
}

#[derive(Clone, Copy, Debug)]
struct FeedbackYawCorrection {
    delta_degrees: f32,
    basis: &'static str,
}

fn status_camera_yaw_degrees(status: &Value, fallback_degrees: Option<f32>) -> f32 {
    let raw_yaw = status
        .get("camera")
        .and_then(|camera| camera.get("yaw"))
        .and_then(Value::as_f64)
        .map(|value| value as f32)
        .or(fallback_degrees)
        .unwrap_or(180.0);
    let degrees = if raw_yaw.abs() <= std::f32::consts::TAU + 1.0e-3 {
        raw_yaw.to_degrees()
    } else {
        raw_yaw
    };
    normalize_degrees(degrees)
}

fn status_world_item_yaw_degrees(
    status: &Value,
    index: usize,
    cache_key: Option<&str>,
) -> Option<f32> {
    let world_items = status.get("world_items").and_then(Value::as_array)?;
    if let Some(item) = world_items.get(index)
        && cache_key_matches(item, cache_key)
        && let Some(yaw) = world_item_yaw_degrees(item)
    {
        return Some(yaw);
    }
    let cache_key = cache_key?;
    world_items
        .iter()
        .find(|item| cache_key_matches(item, Some(cache_key)))
        .and_then(world_item_yaw_degrees)
}

fn cache_key_matches(item: &Value, cache_key: Option<&str>) -> bool {
    let Some(cache_key) = cache_key else {
        return true;
    };
    item.get("cache_key")
        .and_then(Value::as_str)
        .is_some_and(|value| value == cache_key)
}

fn world_item_yaw_degrees(item: &Value) -> Option<f32> {
    item.get("rotation")
        .and_then(json_array4)
        .map(quat_y_degrees)
}

fn feedback_yaw_correction(
    placement_index: usize,
    placement: &GroundedScenePlacement,
    current_yaw_degrees: f32,
    physical: &FeedbackPhysicalLayout,
) -> FeedbackYawCorrection {
    if feedback_physical_kind(placement) == FeedbackPhysicalKind::Seating
        && let (Some(table_center), Some(object_center)) = (
            physical.table_center_xz,
            physical.footprint_centers.get(&placement_index),
        )
        && let Some(target_yaw) = yaw_toward_xz(*object_center, table_center)
    {
        let semantic_error = normalize_degrees(target_yaw - current_yaw_degrees);
        if semantic_error.abs() > 6.0 {
            let step_degrees = (semantic_error.abs() * 0.65).clamp(4.0, 20.0);
            return FeedbackYawCorrection {
                delta_degrees: semantic_error.clamp(-step_degrees, step_degrees),
                basis: "table-facing-yaw",
            };
        }
    }
    let canonical_error = normalize_degrees(placement.rotation_y_degrees - current_yaw_degrees);
    if canonical_error.abs() > 2.0 {
        let step_degrees = (canonical_error.abs() * 0.70).clamp(3.0, 24.0);
        return FeedbackYawCorrection {
            delta_degrees: canonical_error.clamp(-step_degrees, step_degrees),
            basis: "canonical-bsn-yaw",
        };
    }
    FeedbackYawCorrection {
        delta_degrees: 0.0,
        basis: "canonical-bsn-yaw-within-threshold",
    }
}

fn yaw_toward_xz(from: [f32; 2], target: [f32; 2]) -> Option<f32> {
    let dx = target[0] - from[0];
    let dz = target[1] - from[1];
    if !dx.is_finite() || !dz.is_finite() || dx.abs() + dz.abs() <= 1.0e-5 {
        return None;
    }
    Some(normalize_degrees(dx.atan2(dz).to_degrees()))
}

fn quat_y_degrees(quat: [f32; 4]) -> f32 {
    normalize_degrees((2.0 * quat[1].atan2(quat[3])).to_degrees())
}

fn quat_from_y_degrees(degrees: f32) -> [f32; 4] {
    let half = normalize_degrees(degrees).to_radians() * 0.5;
    [0.0, half.sin(), 0.0, half.cos()]
}

fn normalize_degrees(mut degrees: f32) -> f32 {
    if !degrees.is_finite() {
        return 0.0;
    }
    while degrees <= -180.0 {
        degrees += 360.0;
    }
    while degrees > 180.0 {
        degrees -= 360.0;
    }
    degrees
}

fn fmt_feedback_vec3(value: [f32; 3]) -> String {
    format!(
        "{},{},{}",
        fmt_feedback_num(value[0]),
        fmt_feedback_num(value[1]),
        fmt_feedback_num(value[2])
    )
}

fn fmt_feedback_num(value: f32) -> String {
    let mut text = format!("{value:.4}");
    while text.contains('.') && text.ends_with('0') {
        text.pop();
    }
    if text.ends_with('.') {
        text.push('0');
    }
    if text == "-0.0" {
        "0.0".to_string()
    } else {
        text
    }
}

fn sanitize_synthesis_models(models: Vec<SynthesisModel>) -> Vec<SynthesisModel> {
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

fn default_output_path(input: &Path, suffix: &str, ext: &str) -> PathBuf {
    let parent = input.parent().unwrap_or_else(|| Path::new("."));
    let stem = input
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("output");
    parent.join(format!("{stem}{suffix}.{ext}"))
}

#[derive(Debug)]
struct WrittenAsset {
    output_path: PathBuf,
    output_format: AssetOutputFormat,
    asset_kind: &'static str,
    vertices: Option<usize>,
    faces: Option<usize>,
    gaussians: Option<usize>,
    local_aabb: Option<SceneAssetAabb>,
    material: Option<Value>,
    mesh_quality: Option<Value>,
    mesh_quality_failures: Vec<String>,
    catalog_entry: Option<CachedMeshMetadata>,
}

fn write_asset_output(
    input_path: &Path,
    output_dir: Option<&Path>,
    explicit_output: Option<PathBuf>,
    requested_format: AssetOutputFormat,
    asset: SynthesisAsset,
    target_faces: Option<usize>,
    catalog_cache: Option<&mut MeshCache>,
) -> Result<WrittenAsset, String> {
    match asset {
        SynthesisAsset::Mesh(mesh) => {
            if matches!(
                requested_format,
                AssetOutputFormat::Splat | AssetOutputFormat::Ply
            ) {
                return Err(format!(
                    "mesh synthesis cannot be written as {}",
                    requested_format.as_str()
                ));
            }
            let mesh = apply_mesh_decimation(mesh, target_faces)
                .map_err(|err| format!("mesh decimation failed: {err}"))?;
            let quality = mesh_quality_metrics(&mesh);
            let quality_failures = mesh_quality_failures(&quality);
            let mesh_quality = Some(
                serde_json::to_value(&quality)
                    .map_err(|err| format!("failed to serialize mesh quality metrics: {err}"))?,
            );
            let catalog_mesh = catalog_cache
                .as_ref()
                .map(|_| cached_mesh_from_runtime_mesh(&mesh));
            let output_path =
                resolve_asset_output_path(input_path, output_dir, explicit_output, "_mesh", "glb");
            write_glb_mesh(output_path.as_path(), &mesh)?;
            let material = mesh.material.map(|value| {
                json!({
                    "base_color": value.base_color,
                    "metallic": value.metallic,
                    "roughness": value.roughness,
                    "alpha": value.alpha,
                })
            });
            let catalog_entry = match (catalog_cache, catalog_mesh.as_ref()) {
                (Some(cache), Some(cached_mesh)) => Some(
                    cache
                        .upsert_mesh_for_image(input_path, cached_mesh)
                        .map_err(|err| {
                            format!("failed to promote mesh to shared catalog: {err}")
                        })?,
                ),
                _ => None,
            };
            Ok(WrittenAsset {
                output_path,
                output_format: AssetOutputFormat::Glb,
                asset_kind: "mesh",
                vertices: Some(mesh.vertices.len()),
                faces: Some(mesh.faces.len()),
                gaussians: None,
                local_aabb: mesh_scene_aabb(&mesh),
                material,
                mesh_quality,
                mesh_quality_failures: quality_failures,
                catalog_entry,
            })
        }
        SynthesisAsset::GaussianSplat(splats) => {
            if matches!(requested_format, AssetOutputFormat::Glb) {
                return Err("Gaussian splats cannot be written as glb".to_string());
            }
            let output_format = match requested_format {
                AssetOutputFormat::Ply => AssetOutputFormat::Ply,
                _ => AssetOutputFormat::Splat,
            };
            let output_path = resolve_asset_output_path(
                input_path,
                output_dir,
                explicit_output,
                "_splat",
                output_format.as_str(),
            );
            write_splat_asset(output_path.as_path(), &splats, output_format)?;
            let catalog_entry = match catalog_cache {
                Some(cache) => Some(
                    cache
                        .upsert_gaussian_splat_for_image(input_path, &splats)
                        .map_err(|err| {
                            format!("failed to promote Gaussian splat to shared catalog: {err}")
                        })?,
                ),
                None => None,
            };
            Ok(WrittenAsset {
                output_path,
                output_format,
                asset_kind: "gaussian_splat",
                vertices: None,
                faces: None,
                gaussians: Some(splats.len()),
                local_aabb: None,
                material: None,
                mesh_quality: None,
                mesh_quality_failures: Vec::new(),
                catalog_entry,
            })
        }
    }
}

fn cached_mesh_from_runtime_mesh(mesh: &Mesh) -> CachedSynthMesh {
    CachedSynthMesh {
        mesh: CachedTripoMesh {
            vertices: mesh.vertices.clone(),
            faces: mesh.faces.clone(),
        },
        uvs: mesh.uvs.clone(),
        normals: mesh.normals.clone(),
        material: mesh.material.map(|material| CachedSynthMeshMaterial {
            base_color: material.base_color,
            metallic: material.metallic,
            roughness: material.roughness,
            alpha: material.alpha,
        }),
        pbr_textures: mesh
            .pbr_textures
            .clone()
            .map(|textures| CachedSynthMeshPbrTextures {
                base_color: cached_texture_from_runtime_texture(textures.base_color),
                metallic_roughness: cached_texture_from_runtime_texture(
                    textures.metallic_roughness,
                ),
                normal: textures.normal.map(cached_texture_from_runtime_texture),
                emissive: textures.emissive.map(cached_texture_from_runtime_texture),
                occlusion: textures.occlusion.map(cached_texture_from_runtime_texture),
            }),
    }
}

fn mesh_scene_aabb(mesh: &Mesh) -> Option<SceneAssetAabb> {
    let mut iter = mesh.vertices.iter();
    let first = *iter.next()?;
    let mut min = first;
    let mut max = first;
    for vertex in iter {
        for axis in 0..3 {
            min[axis] = min[axis].min(vertex[axis]);
            max[axis] = max[axis].max(vertex[axis]);
        }
    }
    Some(SceneAssetAabb { min, max })
}

fn cached_aabb_to_scene(value: CachedAssetAabb) -> SceneAssetAabb {
    SceneAssetAabb {
        min: value.min,
        max: value.max,
    }
}

fn inferred_scene_asset_frame(
    label: &str,
    aliases: &[String],
    local_aabb: Option<SceneAssetAabb>,
    target_footprint_m: Option<[f32; 2]>,
) -> SceneAssetFrame {
    let descriptor = format!("{} {}", label, aliases.join(" ")).to_ascii_lowercase();
    let yaw_offset_degrees = if descriptor.contains("table") {
        if local_aabb
            .map(|aabb| aabb.max[0] - aabb.min[0] > (aabb.max[2] - aabb.min[2]) * 1.15)
            .unwrap_or(false)
        {
            90.0
        } else {
            0.0
        }
    } else {
        0.0
    };
    SceneAssetFrame {
        yaw_offset_degrees,
        footprint_m: target_footprint_m,
    }
}

fn cached_texture_from_runtime_texture(texture: burn_synth::MeshTexture) -> CachedSynthMeshTexture {
    CachedSynthMeshTexture {
        width: texture.width,
        height: texture.height,
        rgba8: texture.rgba8,
    }
}

fn resolve_asset_output_path(
    input_path: &Path,
    output_dir: Option<&Path>,
    explicit_output: Option<PathBuf>,
    suffix: &str,
    ext: &str,
) -> PathBuf {
    if let Some(path) = explicit_output {
        if path.extension().is_none() || path.is_dir() {
            let stem = input_path
                .file_stem()
                .and_then(|value| value.to_str())
                .unwrap_or("asset");
            return path.join(format!("{stem}{suffix}.{ext}"));
        }
        return path;
    }
    if let Some(dir) = output_dir {
        let stem = input_path
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("asset");
        return dir.join(format!("{stem}{suffix}.{ext}"));
    }
    default_output_path(input_path, suffix, ext)
}

fn write_splat_asset(
    path: &Path,
    splats: &burn_synth::triposplat::GaussianSplatCloud,
    format: AssetOutputFormat,
) -> Result<(), String> {
    ensure_parent_dir(path).map_err(|err| err.to_string())?;
    match format {
        AssetOutputFormat::Ply => splats.write_ply(path),
        AssetOutputFormat::Splat | AssetOutputFormat::Auto => splats.write_splat(path),
        AssetOutputFormat::Glb => Err("Gaussian splats cannot be written as glb".to_string()),
    }
}

fn apply_mesh_decimation(mesh: Mesh, target_faces: Option<usize>) -> Result<Mesh, String> {
    let target_faces = target_faces.filter(|value| *value > 0);
    let Some(target) = target_faces else {
        return Ok(mesh);
    };
    if mesh.pbr_textures.is_some() {
        return Ok(mesh);
    }
    if mesh.faces.len() <= target {
        return Ok(mesh);
    }
    decimate_mesh(&mesh, target)
}

fn decimate_mesh(mesh: &Mesh, target_faces: usize) -> Result<Mesh, String> {
    if target_faces == 0 || mesh.faces.len() <= target_faces {
        return Ok(mesh.clone());
    }
    if mesh.faces.is_empty() || mesh.vertices.is_empty() {
        return Ok(mesh.clone());
    }

    let mut indices = Vec::with_capacity(mesh.faces.len() * 3);
    for face in &mesh.faces {
        indices.push(face[0]);
        indices.push(face[1]);
        indices.push(face[2]);
    }
    let target_index_count = (target_faces.saturating_mul(3)).min(indices.len());
    if target_index_count < 3 {
        return Err("target face count too small for decimation".to_string());
    }

    let vertices_bytes = meshopt::typed_to_bytes(mesh.vertices.as_slice());
    let adapter =
        meshopt::VertexDataAdapter::new(vertices_bytes, std::mem::size_of::<[f32; 3]>(), 0)
            .map_err(|err| format!("meshopt vertex adapter: {err}"))?;

    let mut result_error = 0.0f32;
    let mut simplified = meshopt::simplify(
        &indices,
        &adapter,
        target_index_count,
        1.0,
        meshopt::SimplifyOptions::None,
        Some(&mut result_error),
    );
    if simplified.len() > target_index_count {
        simplified = meshopt::simplify_sloppy(&indices, &adapter, target_index_count, 1.0, None);
    }
    if simplified.len() < 3 {
        return Err("meshopt simplification produced empty mesh".to_string());
    }

    let (vertex_count, remap) =
        meshopt::generate_vertex_remap(mesh.vertices.as_slice(), Some(&simplified));
    let vertices = meshopt::remap_vertex_buffer(mesh.vertices.as_slice(), vertex_count, &remap);
    let uvs = if mesh.uvs.len() == mesh.vertices.len() && !mesh.uvs.is_empty() {
        meshopt::remap_vertex_buffer(mesh.uvs.as_slice(), vertex_count, &remap)
    } else {
        Vec::new()
    };
    let normals = if mesh.normals.len() == mesh.vertices.len() && !mesh.normals.is_empty() {
        meshopt::remap_vertex_buffer(mesh.normals.as_slice(), vertex_count, &remap)
    } else {
        Vec::new()
    };
    let indices = meshopt::remap_index_buffer(Some(&simplified), vertex_count, &remap);
    if indices.len() < 3 {
        return Err("meshopt remap produced empty mesh".to_string());
    }

    let faces = indices
        .chunks_exact(3)
        .map(|chunk| [chunk[0], chunk[1], chunk[2]])
        .collect::<Vec<[u32; 3]>>();
    Ok(Mesh {
        vertices,
        faces,
        uvs,
        normals,
        material: mesh.material,
        pbr_textures: mesh.pbr_textures.clone(),
    })
}

fn ensure_parent_dir(path: &Path) -> io::Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn next_scene_sequence() -> u64 {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(u128::from(u64::MAX)) as u64;
    let mut current = NEXT_SCENE_SEQUENCE.load(Ordering::Relaxed);
    loop {
        let next = now.max(current.saturating_add(1));
        match NEXT_SCENE_SEQUENCE.compare_exchange_weak(
            current,
            next,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => return next,
            Err(value) => current = value,
        }
    }
}

fn default_scene_output_dir() -> PathBuf {
    PathBuf::from("tmp/runs").join(format!("{}_scene_openai_mcp", next_scene_sequence()))
}

fn default_scene_lift_assets() -> bool {
    true
}

fn default_scene_clear_existing() -> bool {
    true
}

fn default_scene_promote_to_catalog() -> bool {
    true
}

fn default_scene_write_artifacts() -> bool {
    true
}

fn default_scene_feedback() -> bool {
    true
}

fn default_scene_feedback_iters() -> usize {
    3
}

fn default_scene_feedback_threshold_profile() -> FeedbackThresholdProfile {
    FeedbackThresholdProfile::Standard
}

fn default_scene_composition_mode() -> SceneCompositionMode {
    SceneCompositionMode::CvGrounded
}

fn default_scene_depth_provider() -> SceneDepthProvider {
    SceneDepthProvider::DepthPro
}

fn default_scene_locator_provider() -> SceneLocatorProvider {
    SceneLocatorProvider::Manifest
}

fn env_or_dotenv_var(key: &str) -> Option<String> {
    env::var(key)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| dotenv_var(Path::new(".env"), key))
}

fn dotenv_var(path: &Path, key: &str) -> Option<String> {
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

#[cfg(test)]
fn select_scene_candidates(
    manifest: &SceneObjectManifest,
    candidates: &[burn_synth_scene::ObjectImageCandidate],
) -> Result<Vec<Value>, String> {
    let selected = burn_synth_scene::select_object_image_candidates(
        manifest,
        candidates,
        DEFAULT_SCENE_RECONSTRUCTION_IMAGE_SCORE,
    )
    .map_err(|err| err.to_string())?;
    Ok(selected_candidates_to_values(&selected))
}

fn selected_candidates_to_values(
    selected: &[burn_synth_scene::SelectedObjectImageCandidate],
) -> Vec<Value> {
    selected
        .iter()
        .map(|candidate| {
            json!({
                "object_id": candidate.object_id,
                "reuse_group": candidate.reuse_group,
                "label": candidate.label,
                "image_path": candidate.image_path,
                "candidate_index": candidate.candidate_index,
                "score": candidate.score,
                "prompt_hash": candidate.prompt_hash,
            })
        })
        .collect()
}

fn record_stage(stage_report: &mut Vec<Value>, stage: &str, started: Instant) {
    stage_report.push(json!({
        "stage": stage,
        "elapsed_ms": elapsed_ms(started.elapsed()),
    }));
}

fn elapsed_ms(duration: Duration) -> u64 {
    duration.as_millis().min(u128::from(u64::MAX)) as u64
}

fn scene_build_summary(response: &Value, elapsed: Duration) -> Value {
    let rejected = response
        .pointer("/candidate_generation/rejected_objects")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let selected_count = response
        .get("selected_candidates")
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or_default();
    let candidate_count = response
        .get("candidates")
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or_default();
    let assets = response
        .pointer("/asset_outputs/items")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .map(|item| {
                    json!({
                        "input_image_path": item.get("input_image_path").cloned().unwrap_or(Value::Null),
                        "output_path": item.get("output_path").cloned().unwrap_or(Value::Null),
                        "cache_key": item.get("cache_key").cloned().unwrap_or(Value::Null),
                        "vertices": item.get("vertices").cloned().unwrap_or(Value::Null),
                        "faces": item.get("faces").cloned().unwrap_or(Value::Null),
                        "local_aabb": item.get("local_aabb").cloned().unwrap_or(Value::Null),
                        "mesh_quality": item.get("mesh_quality").cloned().unwrap_or(Value::Null),
                        "mesh_quality_failures": item.get("mesh_quality_failures").cloned().unwrap_or(Value::Null),
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let placements = response
        .pointer("/grounded_layout/placements")
        .and_then(Value::as_array)
        .map(|placements| {
            placements
                .iter()
                .map(|placement| {
                    json!({
                        "object_id": placement.get("object_id").cloned().unwrap_or(Value::Null),
                        "asset_id": placement.get("asset_id").cloned().unwrap_or(Value::Null),
                        "translation": placement.get("translation").cloned().unwrap_or(Value::Null),
                        "scale": placement.get("scale").cloned().unwrap_or(Value::Null),
                        "target_footprint_m": placement.get("target_footprint_m").cloned().unwrap_or(Value::Null),
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    json!({
        "ok": rejected.is_empty(),
        "elapsed_ms": elapsed_ms(elapsed),
        "source_scene_path": response.pointer("/manifest/source_scene_path").cloned().unwrap_or(Value::Null),
        "candidate_count": candidate_count,
        "selected_count": selected_count,
        "rejected_objects": rejected,
        "asset_count": assets.len(),
        "assets": assets,
        "placement_count": placements.len(),
        "placements": placements,
        "feedback": response.get("feedback").map(|feedback| json!({
            "enabled": feedback.get("enabled").cloned().unwrap_or(Value::Null),
            "accepted": feedback.get("accepted").cloned().unwrap_or(Value::Null),
            "accepted_iteration": feedback.get("accepted_iteration").cloned().unwrap_or(Value::Null),
            "capture_dir": feedback.get("capture_dir").cloned().unwrap_or(Value::Null),
        })).unwrap_or(Value::Null),
        "stage_report": response.get("stage_report").cloned().unwrap_or(Value::Null),
    })
}

fn scene_asset_quality_failures(asset_outputs: &Value) -> Vec<String> {
    let Some(items) = asset_outputs.get("items").and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut failures = Vec::new();
    for item in items {
        if item.get("asset_kind").and_then(Value::as_str) != Some("mesh") {
            continue;
        }
        if item.get("synthesis_backend").and_then(Value::as_str) != Some("trellis") {
            continue;
        }
        let output_path = item
            .get("output_path")
            .and_then(Value::as_str)
            .unwrap_or("<unknown output>");
        for failure in item
            .get("mesh_quality_failures")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
        {
            failures.push(format!("{output_path}: {failure}"));
        }
    }
    failures
}

fn write_scene_build_artifacts(output_dir: &Path, response: &Value) -> Result<(), String> {
    fs::create_dir_all(output_dir).map_err(|err| {
        format!(
            "create scene build artifact directory {}: {err}",
            output_dir.display()
        )
    })?;
    for (key, file_name) in [
        ("preparation", "preparation.json"),
        ("manifest", "manifest.json"),
        ("object_image_requests", "object_image_requests.json"),
        ("provider_metadata", "provider_metadata.json"),
        ("candidate_generation", "candidate_generation.json"),
        ("candidates", "candidates.json"),
        ("selected_candidates", "selected_candidates.json"),
        ("asset_outputs", "asset_outputs.json"),
        ("asset_bindings", "asset_bindings.json"),
        ("plan", "plan.json"),
        ("grounded_layout", "grounded_layout.json"),
        ("commands", "commands.json"),
        ("feedback", "feedback_report.json"),
        ("stage_report", "stage_report.json"),
        ("e2e_summary", "summary.json"),
    ] {
        if let Some(value) = response.get(key) {
            write_json_file(&output_dir.join(file_name), value).map_err(|err| err.to_string())?;
        }
    }
    if let Some(bsn) = response.get("bsn").and_then(Value::as_str) {
        fs::write(output_dir.join("scene.bsn"), bsn).map_err(|err| {
            format!(
                "write scene BSN artifact {}: {err}",
                output_dir.join("scene.bsn").display()
            )
        })?;
    }
    write_json_file(
        &output_dir.join("scene_build_response_structured.json"),
        response,
    )
    .map_err(|err| err.to_string())?;
    Ok(())
}

#[derive(Clone, Debug)]
struct SceneDepthMapEvidence {
    depth_m: Vec<f32>,
    width: u32,
    height: u32,
    intrinsics: CameraIntrinsics,
    focal_length_px: Option<f32>,
    vertical_fov_degrees: Option<f32>,
}

#[derive(Clone, Debug, Default, Serialize)]
struct SceneDepthAnnotationSummary {
    provider: String,
    annotated_objects: usize,
    total_objects: usize,
    depth_map_size: [u32; 2],
    focal_length_px: Option<f32>,
    vertical_fov_degrees: Option<f32>,
    floor_sample_count: usize,
    floor_residual_m: Option<f32>,
}

impl McpServer {
    fn depth_pro_grounding_evidence(
        &self,
        evidence: &mut SceneGroundingEvidence,
        source_scene_path: &Path,
        output_dir: &Path,
    ) -> Result<(), String> {
        let artifact_dir = output_dir.join("depth_pro");
        fs::create_dir_all(&artifact_dir).map_err(|err| {
            format!(
                "failed to create DepthPro artifact directory {}: {err}",
                artifact_dir.display()
            )
        })?;

        let precision: DepthPrecision = self.config.depth_precision.into();
        let load_config = DepthLoadConfig {
            model: DepthModelKind::DepthPro,
            precision,
            checkpoint: DepthCheckpointSource::default_cdn(DepthModelKind::DepthPro, precision),
            cache_dir: self.config.depth_cache_dir.clone(),
            allow_download: self.config.depth_allow_download,
            require_gpu: true,
        };
        let mut progress_events = Vec::new();
        let device = burn::tensor::Device::<burn_depth::InferenceBackend>::default();
        let load_started = Instant::now();
        let pipeline = DepthPipeline::<burn_depth::InferenceBackend>::load_with_progress(
            &device,
            load_config,
            |event| progress_events.push(depth_load_event_json(event)),
        )
        .map_err(|err| format!("load DepthPro pipeline: {err}"))?;
        let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;

        let image = image::open(source_scene_path).map_err(|err| {
            format!(
                "failed to load depth source image {}: {err}",
                source_scene_path.display()
            )
        })?;
        let infer_started = Instant::now();
        let prediction = pipeline
            .predict(
                image,
                DepthRuntimeConfig {
                    output_size: None,
                    return_gpu_tensors: false,
                },
            )
            .map_err(|err| format!("DepthPro inference failed: {err}"))?;
        let infer_ms = infer_started.elapsed().as_secs_f64() * 1000.0;
        let depth_map = scene_depth_map_from_prediction(prediction)?;
        let mut summary =
            annotate_grounding_evidence_with_depth_map(evidence, &depth_map, "depth_pro");
        summary.provider = "depth-pro".to_string();
        let floor_sample_count = summary.floor_sample_count;

        let summary_path = artifact_dir.join("depth_evidence.json");
        let metadata = json!({
            "provider": "depth-pro",
            "model": "depth-pro",
            "precision": scene_depth_precision_label(self.config.depth_precision),
            "load_ms": load_ms,
            "infer_ms": infer_ms,
            "load_events": progress_events,
            "summary": summary,
        });
        write_json_file(&summary_path, &metadata).map_err(|err| err.to_string())?;

        evidence.depth = Some(DepthEvidenceRef {
            provider: "depth-pro".to_string(),
            model: Some("depth-pro".to_string()),
            precision: Some(scene_depth_precision_label(self.config.depth_precision).to_string()),
            artifact_path: Some(summary_path.display().to_string()),
            focal_length_px: depth_map.focal_length_px,
            vertical_fov_degrees: depth_map.vertical_fov_degrees,
            image_size: Some([depth_map.width, depth_map.height]),
            depth_map_size: Some([depth_map.width, depth_map.height]),
            floor_sample_count: Some(floor_sample_count),
        });
        evidence.camera.focal_length_px = evidence
            .camera
            .focal_length_px
            .or(depth_map.focal_length_px);
        evidence.camera.vertical_fov_degrees = evidence
            .camera
            .vertical_fov_degrees
            .or(depth_map.vertical_fov_degrees);
        evidence.camera.principal_point = evidence
            .camera
            .principal_point
            .or(Some([depth_map.intrinsics.cx, depth_map.intrinsics.cy]));
        evidence.camera.image_size = Some([depth_map.width, depth_map.height]);
        evidence.floor = estimate_scene_floor_plane(&depth_map).unwrap_or_default();

        Ok(())
    }
}

fn scene_depth_precision_label(value: SceneDepthPrecision) -> &'static str {
    match value {
        SceneDepthPrecision::F32 => "f32",
        SceneDepthPrecision::F16 => "f16",
    }
}

fn depth_load_event_json(event: DepthLoadEvent) -> Value {
    json!({
        "stage": depth_load_stage_label(event.stage),
        "message": event.message,
        "current": event.current,
        "total": event.total,
    })
}

fn depth_load_stage_label(stage: DepthLoadStage) -> &'static str {
    match stage {
        DepthLoadStage::Manifest => "manifest",
        DepthLoadStage::CacheHit => "cache_hit",
        DepthLoadStage::CacheMiss => "cache_miss",
        DepthLoadStage::Part => "part",
        DepthLoadStage::Verify => "verify",
        DepthLoadStage::Deserialize => "deserialize",
        DepthLoadStage::ModelReady => "model_ready",
    }
}

fn scene_depth_map_from_prediction<B: Backend>(
    prediction: burn_depth::inference::DepthPrediction<B>,
) -> Result<SceneDepthMapEvidence, String> {
    let dims: [usize; 3] = prediction.depth_m.shape().dims();
    if dims[0] != 1 {
        return Err(format!(
            "scene depth expects batch size 1, got depth tensor shape {:?}",
            dims
        ));
    }
    let height = dims[1] as u32;
    let width = dims[2] as u32;
    let depth_m = tensor_to_vec_f32(prediction.depth_m)?;
    let expected = width as usize * height as usize;
    if depth_m.len() != expected {
        return Err(format!(
            "scene depth tensor data length mismatch: expected {expected}, got {}",
            depth_m.len()
        ));
    }

    let focal_length_px = prediction
        .focallength_px
        .map(tensor_scalar_f32)
        .transpose()?;
    let fovy_rad = prediction.fovy_rad.map(tensor_scalar_f32).transpose()?;
    let vertical_fov_degrees = prediction
        .intrinsics
        .map(|intrinsics| {
            2.0 * ((height as f32 * 0.5) / intrinsics.fy.max(1.0e-5))
                .atan()
                .to_degrees()
        })
        .or_else(|| fovy_rad.map(f32::to_degrees))
        .or_else(|| {
            focal_length_px.map(|focal| {
                2.0 * ((height as f32 * 0.5) / focal.max(1.0e-5))
                    .atan()
                    .to_degrees()
            })
        });
    let intrinsics = prediction.intrinsics.unwrap_or_else(|| {
        let fy = fovy_rad
            .map(|fovy| (height as f32 * 0.5) / (fovy * 0.5).tan().max(1.0e-5))
            .or(focal_length_px)
            .unwrap_or(width.max(height) as f32);
        let fx = focal_length_px.unwrap_or(fy);
        CameraIntrinsics {
            fx,
            fy,
            cx: (width.saturating_sub(1)) as f32 * 0.5,
            cy: (height.saturating_sub(1)) as f32 * 0.5,
            width,
            height,
        }
    });

    Ok(SceneDepthMapEvidence {
        depth_m,
        width,
        height,
        intrinsics,
        focal_length_px,
        vertical_fov_degrees,
    })
}

fn tensor_to_vec_f32<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Result<Vec<f32>, String> {
    tensor
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|err| format!("read tensor data: {err}"))
}

fn tensor_scalar_f32<B: Backend>(tensor: Tensor<B, 1>) -> Result<f32, String> {
    let values = tensor_to_vec_f32(tensor)?;
    values
        .first()
        .copied()
        .filter(|value| value.is_finite())
        .ok_or_else(|| "depth scalar tensor was empty or non-finite".to_string())
}

fn annotate_grounding_evidence_with_depth_map(
    evidence: &mut SceneGroundingEvidence,
    depth_map: &SceneDepthMapEvidence,
    provenance_label: &str,
) -> SceneDepthAnnotationSummary {
    let floor = estimate_scene_floor_plane(depth_map);
    let mut annotated_objects = 0usize;
    for object in &mut evidence.objects {
        let Some(detection) = object.detection.as_ref() else {
            continue;
        };
        let bbox = normalized_bbox_to_image_bbox(detection.bbox, depth_map.width, depth_map.height);
        let bbox_stats =
            depth_stats_for_bbox(&depth_map.depth_m, depth_map.width, depth_map.height, bbox);
        let contact_pixel = object
            .contact_pixel
            .or(detection.point)
            .unwrap_or_else(|| bbox_bottom_center(detection.bbox));
        let contact_depth = depth_at_bbox_contact_region(
            &depth_map.depth_m,
            depth_map.width,
            depth_map.height,
            bbox,
        )
        .or_else(|| sample_depth_at_normalized_pixel(depth_map, contact_pixel));
        let Some(contact_depth) = contact_depth.filter(|value| value.is_finite() && *value > 0.0)
        else {
            continue;
        };

        let pixel = normalized_to_depth_pixel(contact_pixel, depth_map.width, depth_map.height);
        let ray = pixel_to_ray(pixel[0], pixel[1], depth_map.intrinsics);
        let point = backproject_depth(pixel[0], pixel[1], contact_depth, depth_map.intrinsics);
        let target_footprint =
            estimate_depth_target_footprint(detection, bbox, contact_depth, depth_map.intrinsics);
        object.depth_stats =
            bbox_stats.map(|(min_m, median_m, max_m, sample_count)| ObjectDepthStats {
                median_m,
                min_m,
                max_m,
                contact_m: Some(contact_depth),
                sample_count: Some(sample_count),
            });
        if object.depth_stats.is_none() {
            object.depth_stats = Some(ObjectDepthStats {
                median_m: contact_depth,
                min_m: contact_depth,
                max_m: contact_depth,
                contact_m: Some(contact_depth),
                sample_count: Some(1),
            });
        }
        object.contact_pixel = Some(contact_pixel);
        object.candidate_floor_contact_rays.push(ray);
        object.metric_contact_point_m = Some(point);
        object.target_footprint_m = target_footprint;
        if !object
            .provenance
            .iter()
            .any(|entry| entry == provenance_label)
        {
            object.provenance.push(provenance_label.to_string());
        }
        annotated_objects += 1;
    }

    let floor_sample_count = floor_sample_count(depth_map);
    SceneDepthAnnotationSummary {
        provider: provenance_label.to_string(),
        annotated_objects,
        total_objects: evidence.objects.len(),
        depth_map_size: [depth_map.width, depth_map.height],
        focal_length_px: depth_map.focal_length_px,
        vertical_fov_degrees: depth_map.vertical_fov_degrees,
        floor_sample_count,
        floor_residual_m: floor.and_then(|floor| floor.residual_m),
    }
}

fn estimate_scene_floor_plane(depth_map: &SceneDepthMapEvidence) -> Option<EstimatedFloorPlane> {
    let mut points = Vec::new();
    let step_x = (depth_map.width / 64).max(1);
    let step_y = (depth_map.height / 48).max(1);
    let y_start = (depth_map.height as f32 * 0.62).floor() as u32;
    for y in (y_start..depth_map.height).step_by(step_y as usize) {
        for x in (0..depth_map.width).step_by(step_x as usize) {
            let index = y as usize * depth_map.width as usize + x as usize;
            let depth = depth_map.depth_m.get(index).copied().unwrap_or_default();
            if depth.is_finite() && depth > 0.0 {
                points.push(backproject_depth(
                    x as f32 + 0.5,
                    y as f32 + 0.5,
                    depth,
                    depth_map.intrinsics,
                ));
            }
        }
    }
    let plane = estimate_floor_plane(&points)?;
    let residual = if points.is_empty() {
        None
    } else {
        let sum = points
            .iter()
            .map(|point| {
                (plane.normal[0] * point[0]
                    + plane.normal[1] * point[1]
                    + plane.normal[2] * point[2]
                    + plane.d)
                    .abs()
            })
            .sum::<f32>();
        Some(sum / points.len() as f32)
    };
    Some(EstimatedFloorPlane {
        normal: plane.normal,
        distance_m: plane.d,
        residual_m: residual,
        confidence: Some((1.0 / (1.0 + residual.unwrap_or(1.0))).clamp(0.0, 1.0)),
    })
}

fn floor_sample_count(depth_map: &SceneDepthMapEvidence) -> usize {
    let step_x = (depth_map.width / 64).max(1);
    let step_y = (depth_map.height / 48).max(1);
    let y_start = (depth_map.height as f32 * 0.62).floor() as u32;
    let mut count = 0usize;
    for y in (y_start..depth_map.height).step_by(step_y as usize) {
        for x in (0..depth_map.width).step_by(step_x as usize) {
            let index = y as usize * depth_map.width as usize + x as usize;
            let depth = depth_map.depth_m.get(index).copied().unwrap_or_default();
            if depth.is_finite() && depth > 0.0 {
                count += 1;
            }
        }
    }
    count
}

fn normalized_bbox_to_image_bbox(
    bbox: [f32; 4],
    image_width: u32,
    image_height: u32,
) -> ImageBoundingBox {
    let bbox = [
        bbox[0].clamp(0.0, 1.0),
        bbox[1].clamp(0.0, 1.0),
        bbox[2].clamp(0.0, 1.0),
        bbox[3].clamp(0.0, 1.0),
    ];
    let x0 = (bbox[0].min(bbox[2]) * image_width as f32).floor() as u32;
    let x1 = (bbox[0].max(bbox[2]) * image_width as f32).ceil() as u32;
    let y0 = (bbox[1].min(bbox[3]) * image_height as f32).floor() as u32;
    let y1 = (bbox[1].max(bbox[3]) * image_height as f32).ceil() as u32;
    let x0 = x0.min(image_width.saturating_sub(1));
    let y0 = y0.min(image_height.saturating_sub(1));
    let x1 = x1.min(image_width).max(x0 + 1);
    let y1 = y1.min(image_height).max(y0 + 1);
    ImageBoundingBox {
        x: x0,
        y: y0,
        width: x1 - x0,
        height: y1 - y0,
    }
}

fn depth_stats_for_bbox(
    depth_m: &[f32],
    image_width: u32,
    image_height: u32,
    bbox: ImageBoundingBox,
) -> Option<(f32, f32, f32, usize)> {
    if depth_m.len() != image_width as usize * image_height as usize {
        return None;
    }
    let x0 = bbox.x.min(image_width);
    let x1 = bbox.x.saturating_add(bbox.width).min(image_width);
    let y0 = bbox.y.min(image_height);
    let y1 = bbox.y.saturating_add(bbox.height).min(image_height);
    if x0 >= x1 || y0 >= y1 {
        return None;
    }
    let mut values = Vec::new();
    for y in y0..y1 {
        let row = y as usize * image_width as usize;
        for x in x0..x1 {
            let value = depth_m[row + x as usize];
            if value.is_finite() && value > 0.0 {
                values.push(value);
            }
        }
    }
    if values.is_empty() {
        return None;
    }
    values.sort_by(|left, right| left.total_cmp(right));
    Some((
        values[0],
        values[values.len() / 2],
        values[values.len() - 1],
        values.len(),
    ))
}

fn sample_depth_at_normalized_pixel(
    depth_map: &SceneDepthMapEvidence,
    pixel: [f32; 2],
) -> Option<f32> {
    let [x, y] = normalized_to_depth_pixel(pixel, depth_map.width, depth_map.height);
    let x = x
        .round()
        .clamp(0.0, depth_map.width.saturating_sub(1) as f32) as u32;
    let y = y
        .round()
        .clamp(0.0, depth_map.height.saturating_sub(1) as f32) as u32;
    let value = depth_map.depth_m[y as usize * depth_map.width as usize + x as usize];
    (value.is_finite() && value > 0.0).then_some(value)
}

fn normalized_to_depth_pixel(pixel: [f32; 2], width: u32, height: u32) -> [f32; 2] {
    [
        pixel[0].clamp(0.0, 1.0) * width.saturating_sub(1) as f32,
        pixel[1].clamp(0.0, 1.0) * height.saturating_sub(1) as f32,
    ]
}

fn estimate_depth_target_footprint(
    detection: &Detection,
    bbox: ImageBoundingBox,
    contact_depth_m: f32,
    intrinsics: CameraIntrinsics,
) -> Option<[f32; 2]> {
    if !contact_depth_m.is_finite() || contact_depth_m <= 0.0 {
        return None;
    }
    let width_m = bbox.width as f32 * contact_depth_m / intrinsics.fx.max(1.0e-5);
    if !width_m.is_finite() || width_m <= 0.0 {
        return None;
    }
    let descriptor = format!("{} {}", detection.label, detection.source_query).to_ascii_lowercase();
    let footprint = if descriptor.contains("sofa")
        || descriptor.contains("couch")
        || descriptor.contains("sectional")
    {
        [width_m.clamp(1.4, 6.5), (width_m * 0.48).clamp(0.8, 2.8)]
    } else if descriptor.contains("conference") && descriptor.contains("table") {
        [width_m.clamp(1.2, 6.5), (width_m * 0.42).clamp(0.7, 2.8)]
    } else if descriptor.contains("table") {
        [width_m.clamp(0.6, 4.5), (width_m * 0.55).clamp(0.45, 2.5)]
    } else if descriptor.contains("chair") || descriptor.contains("seat") {
        [
            width_m.clamp(0.42, 0.95),
            (width_m * 1.05).clamp(0.42, 1.05),
        ]
    } else {
        [width_m.clamp(0.2, 4.0), width_m.clamp(0.2, 4.0)]
    };
    Some(footprint)
}

#[derive(Debug, Deserialize)]
struct LocateAnythingReferenceResponse {
    #[serde(default)]
    load_ms: Option<f64>,
    #[serde(default)]
    batch_infer_ms: Option<f64>,
    #[serde(default)]
    results: Vec<LocateAnythingReferenceResult>,
}

#[derive(Debug, Deserialize)]
struct LocateAnythingReferenceResult {
    query: String,
    #[serde(default)]
    detections: Vec<Detection>,
    #[serde(default)]
    timings_ms: Option<HashMap<String, f64>>,
}

impl McpServer {
    fn locate_anything_grounding_evidence(
        &mut self,
        backend: LocateAnythingBackend,
        manifest: &SceneObjectManifest,
        source_scene_path: &Path,
        output_dir: &Path,
    ) -> Result<SceneGroundingEvidence, String> {
        match backend {
            LocateAnythingBackend::PythonReference => self
                .locate_anything_reference_grounding_evidence(
                    manifest,
                    source_scene_path,
                    output_dir,
                ),
            LocateAnythingBackend::BurnNative => self
                .locate_anything_burn_native_grounding_evidence(
                    manifest,
                    source_scene_path,
                    output_dir,
                ),
        }
    }

    fn locate_anything_reference_grounding_evidence(
        &self,
        manifest: &SceneObjectManifest,
        source_scene_path: &Path,
        output_dir: &Path,
    ) -> Result<SceneGroundingEvidence, String> {
        let queries = locate_anything_queries(manifest);
        if queries.is_empty() {
            return Err(
                "LocateAnything locator requires at least one non-empty manifest object label"
                    .to_string(),
            );
        }

        let artifact_dir = output_dir.join("locate_anything_reference");
        fs::create_dir_all(&artifact_dir).map_err(|err| {
            format!(
                "failed to create LocateAnything artifact directory {}: {err}",
                artifact_dir.display()
            )
        })?;
        let output_path = artifact_dir.join("reference.json");
        let log_path = artifact_dir.join("reference.log");
        let python_bin = self.locate_anything_python_bin();
        let mut command = Command::new(&python_bin);
        command
            .arg(&self.config.locate_anything_reference_script)
            .arg("--model-root")
            .arg(&self.config.locate_anything_model_root)
            .arg("--image")
            .arg(source_scene_path)
            .arg("--output")
            .arg(&output_path)
            .arg("--device")
            .arg(&self.config.locate_anything_device)
            .arg("--dtype")
            .arg("bf16")
            .arg("--attn")
            .arg("sdpa")
            .arg("--generation-mode")
            .arg("hybrid")
            .arg("--in-token-limit")
            .arg(self.config.locate_anything_in_token_limit.to_string())
            .arg("--max-new-tokens")
            .arg("1024")
            .arg("--temperature")
            .arg("0.0")
            .env("PYTHONUNBUFFERED", "1");
        for query in &queries {
            command.arg("--query").arg(query);
        }

        let started = Instant::now();
        let output = command.output().map_err(|err| {
            format!(
                "failed to launch LocateAnything reference `{}`: {err}",
                python_bin.display()
            )
        })?;
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        let mut log = String::new();
        log.push_str("$ ");
        log.push_str(&format!("{command:?}\n"));
        log.push_str("--- stdout ---\n");
        log.push_str(&String::from_utf8_lossy(&output.stdout));
        log.push_str("\n--- stderr ---\n");
        log.push_str(&String::from_utf8_lossy(&output.stderr));
        log.push_str(&format!("\n--- elapsed_ms: {elapsed_ms:.3} ---\n"));
        fs::write(&log_path, &log).map_err(|err| {
            format!(
                "failed to write LocateAnything reference log {}: {err}",
                log_path.display()
            )
        })?;
        if !output.status.success() {
            return Err(format!(
                "LocateAnything reference failed with status {}; see {}",
                output.status,
                log_path.display()
            ));
        }
        let response: LocateAnythingReferenceResponse = read_json_path(&output_path)?;
        let mut detections = Vec::new();
        let mut timing_by_query = serde_json::Map::new();
        for result in &response.results {
            if let Some(timings) = result.timings_ms.as_ref() {
                timing_by_query.insert(result.query.clone(), json!(timings));
            }
            detections.extend(result.detections.iter().cloned());
        }
        if detections.is_empty() {
            return Err(format!(
                "LocateAnything reference returned no detections for {} queries; see {}",
                queries.len(),
                output_path.display()
            ));
        }

        let mut evidence = locate_anything_evidence_from_detections(
            manifest,
            source_scene_path,
            detections,
            "locate_anything_reference",
        )?;
        let metadata = json!({
            "provider": "locate_anything_reference",
            "python_bin": python_bin,
            "model_root": self.config.locate_anything_model_root,
            "script": self.config.locate_anything_reference_script,
            "device": self.config.locate_anything_device,
            "in_token_limit": self.config.locate_anything_in_token_limit,
            "queries": queries,
            "load_ms": response.load_ms,
            "batch_infer_ms": response.batch_infer_ms,
            "timings_ms": timing_by_query,
            "elapsed_ms": elapsed_ms,
            "reference_json": output_path,
            "reference_log": log_path,
        });
        write_json_file(&artifact_dir.join("metadata.json"), &metadata)
            .map_err(|err| err.to_string())?;
        for object in &mut evidence.objects {
            object
                .provenance
                .push("locate_anything_reference_scene_ground".to_string());
        }
        Ok(evidence)
    }

    fn locate_anything_burn_native_grounding_evidence(
        &mut self,
        manifest: &SceneObjectManifest,
        source_scene_path: &Path,
        output_dir: &Path,
    ) -> Result<SceneGroundingEvidence, String> {
        let queries = locate_anything_queries(manifest);
        if queries.is_empty() {
            return Err(
                "LocateAnything locator requires at least one non-empty manifest object label"
                    .to_string(),
            );
        }

        let artifact_dir = output_dir.join("locate_anything_burn_native");
        fs::create_dir_all(&artifact_dir).map_err(|err| {
            format!(
                "failed to create LocateAnything native artifact directory {}: {err}",
                artifact_dir.display()
            )
        })?;
        let image = image::open(source_scene_path).map_err(|err| {
            format!(
                "failed to load LocateAnything source image {}: {err}",
                source_scene_path.display()
            )
        })?;
        let runtime_config = LocateAnythingRuntimeConfig {
            model_root: self.config.locate_anything_model_root.clone(),
            backend: LocateAnythingRuntimeBackend::BurnNative,
            allow_experimental_native_detect: true,
            decode_mode: DecodeMode::Hybrid,
            max_new_tokens: 1024,
            reference_script: self.config.locate_anything_reference_script.clone(),
            python_bin: self.config.locate_anything_python_bin.clone(),
            reference_device: self.config.locate_anything_device.clone(),
            reference_dtype: "bf16".to_string(),
            reference_attention: "sdpa".to_string(),
            in_token_limit: self.config.locate_anything_in_token_limit as u32,
            run_root: artifact_dir.clone(),
            ..LocateAnythingRuntimeConfig::default()
        };
        let cache_key = LocateAnythingBurnNativeCacheKey::from_config(&runtime_config);
        let cache_hit = self
            .locate_anything_burn_native_runtime
            .as_ref()
            .is_some_and(|cached| cached.key == cache_key);
        if !cache_hit {
            let runtime = LocateAnythingRuntime::new(runtime_config.clone())
                .map_err(|err| format!("initialize Burn-native LocateAnything runtime: {err}"))?;
            self.locate_anything_burn_native_runtime = Some(CachedLocateAnythingRuntime {
                key: cache_key,
                runtime,
            });
        }
        let runtime = &mut self
            .locate_anything_burn_native_runtime
            .as_mut()
            .expect("LocateAnything runtime cache initialized")
            .runtime;
        let detection_queries = queries
            .iter()
            .map(|query| DetectionQuery {
                query: query.clone(),
                label_hint: None,
            })
            .collect::<Vec<_>>();
        let started = Instant::now();
        let batches = runtime
            .detect_batch(&image, &detection_queries)
            .map_err(|err| format!("Burn-native LocateAnything detect failed: {err}"))?;
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        let stage_timings = runtime.last_burn_native_stage_timings().cloned();
        let detections = batches.into_iter().flatten().collect::<Vec<_>>();
        if detections.is_empty() {
            return Err(format!(
                "Burn-native LocateAnything returned no detections for {} queries",
                queries.len()
            ));
        }

        let detections_path = artifact_dir.join("detections.json");
        write_json_file(&detections_path, &detections).map_err(|err| err.to_string())?;
        let mut evidence = locate_anything_evidence_from_detections(
            manifest,
            source_scene_path,
            detections,
            "locate_anything_burn_native",
        )?;
        let metadata = json!({
            "provider": "locate_anything_burn_native",
            "model_root": self.config.locate_anything_model_root,
            "backend": "burn_native",
            "in_token_limit": self.config.locate_anything_in_token_limit,
            "decode_mode": format!("{:?}", runtime_config.decode_mode),
            "max_new_tokens": runtime_config.max_new_tokens,
            "repetition_penalty": runtime_config.repetition_penalty,
            "top_p": runtime_config.top_p,
            "top_k": runtime_config.top_k,
            "queries": queries,
            "elapsed_ms": elapsed_ms,
            "stage_timings": stage_timings,
            "runtime_cache_hit": cache_hit,
            "detections_json": detections_path,
            "compile_feature_hint": "build burn_synth_mcp with --features locate-anything-wgpu for WGPU native execution",
        });
        write_json_file(&artifact_dir.join("metadata.json"), &metadata)
            .map_err(|err| err.to_string())?;
        for object in &mut evidence.objects {
            object
                .provenance
                .push("locate_anything_burn_native_scene_ground".to_string());
        }
        Ok(evidence)
    }

    fn locate_anything_python_bin(&self) -> PathBuf {
        self.config
            .locate_anything_python_bin
            .clone()
            .or_else(|| env::var_os("LOCATE_ANYTHING_PYTHON").map(PathBuf::from))
            .unwrap_or_else(|| {
                let torch_venv = PathBuf::from("/home/mosure/.venvs/torch/bin/python");
                if torch_venv.exists() {
                    torch_venv
                } else {
                    PathBuf::from("python3")
                }
            })
    }
}

fn locate_anything_queries(manifest: &SceneObjectManifest) -> Vec<String> {
    let mut queries = Vec::new();
    for object in &manifest.objects {
        let query = object.label.trim();
        if query.is_empty() {
            continue;
        }
        if !queries.iter().any(|existing: &String| existing == query) {
            queries.push(query.to_string());
        }
    }
    queries
}

fn locate_anything_evidence_from_detections(
    manifest: &SceneObjectManifest,
    source_scene_path: &Path,
    detections: Vec<Detection>,
    provenance_label: &str,
) -> Result<SceneGroundingEvidence, String> {
    let image_size = image::image_dimensions(source_scene_path)
        .ok()
        .map(|(width, height)| [width, height]);
    let mut detections_by_query: HashMap<String, Vec<Detection>> = HashMap::new();
    for detection in &detections {
        detections_by_query
            .entry(normalized_query_key(&detection.source_query))
            .or_default()
            .push(detection.clone());
    }

    let mut objects = Vec::new();
    for object in &manifest.objects {
        let query_key = normalized_query_key(&object.label);
        let matched = detections_by_query
            .get(&query_key)
            .cloned()
            .unwrap_or_default();
        let object_detection = if matched.is_empty() {
            manifest_detection_for_object(object)
        } else {
            Some(union_detection_for_object(object, &matched))
        };
        objects.push(ObjectGroundingEvidence {
            object_id: object.id.clone(),
            instance_id: None,
            reuse_group: object.reuse_group.clone(),
            detection: object_detection,
            asset_id: None,
            contact_pixel: None,
            depth_stats: None,
            candidate_floor_contact_rays: Vec::new(),
            metric_contact_point_m: None,
            target_footprint_m: None,
            provenance: if matched.is_empty() {
                vec!["manifest_fallback_missing_detection".to_string()]
            } else {
                vec![provenance_label.to_string()]
            },
        });

        let instance_evidence =
            locate_anything_instance_evidence(object, &matched, provenance_label);
        objects.extend(instance_evidence);
    }

    Ok(SceneGroundingEvidence {
        source_image_path: source_scene_path.display().to_string(),
        depth: None,
        detections,
        camera: burn_synth_scene::EstimatedCamera {
            image_size,
            ..burn_synth_scene::EstimatedCamera::default()
        },
        floor: burn_synth_scene::EstimatedFloorPlane::default(),
        objects,
    })
}

fn locate_anything_instance_evidence(
    object: &SceneObjectSpec,
    detections: &[Detection],
    provenance_label: &str,
) -> Vec<ObjectGroundingEvidence> {
    let mut out = Vec::new();
    let instances = manifest_instances_for_matching(object);
    let mut used = vec![false; detections.len()];
    for (instance_id, bbox, contact) in instances {
        let detection_index = best_detection_match(&bbox, detections, &used);
        let (detection, provenance) = if let Some(index) = detection_index {
            used[index] = true;
            (
                Some(detections[index].clone()),
                vec![provenance_label.to_string()],
            )
        } else {
            (
                Some(Detection {
                    label: object.label.clone(),
                    bbox,
                    point: contact,
                    confidence: None,
                    source_query: object.label.clone(),
                }),
                vec!["manifest_fallback_missing_detection".to_string()],
            )
        };
        let contact_pixel = detection
            .as_ref()
            .and_then(|detection| detection.point)
            .or_else(|| {
                detection
                    .as_ref()
                    .map(|detection| bbox_bottom_center(detection.bbox))
            })
            .or(contact);
        out.push(ObjectGroundingEvidence {
            object_id: object.id.clone(),
            instance_id,
            reuse_group: object.reuse_group.clone(),
            detection,
            asset_id: None,
            contact_pixel,
            depth_stats: None,
            candidate_floor_contact_rays: Vec::new(),
            metric_contact_point_m: None,
            target_footprint_m: None,
            provenance,
        });
    }
    out
}

type ManifestMatchingInstance = (Option<String>, [f32; 4], Option<[f32; 2]>);

fn manifest_instances_for_matching(object: &SceneObjectSpec) -> Vec<ManifestMatchingInstance> {
    if object.instances.is_empty() {
        return vec![(None, object.bbox, Some(bbox_bottom_center(object.bbox)))];
    }
    object
        .instances
        .iter()
        .map(|instance: &SceneObjectInstanceSpec| {
            (
                instance.id.clone(),
                instance.bbox,
                instance
                    .contact
                    .or_else(|| Some(bbox_bottom_center(instance.bbox))),
            )
        })
        .collect()
}

fn manifest_detection_for_object(object: &SceneObjectSpec) -> Option<Detection> {
    Some(Detection {
        label: object.label.clone(),
        bbox: object.bbox,
        point: Some(bbox_bottom_center(object.bbox)),
        confidence: None,
        source_query: object.label.clone(),
    })
}

fn union_detection_for_object(object: &SceneObjectSpec, detections: &[Detection]) -> Detection {
    let bbox = detections
        .iter()
        .map(|detection| detection.bbox)
        .reduce(union_bbox)
        .unwrap_or(object.bbox);
    Detection {
        label: object.label.clone(),
        bbox,
        point: Some(bbox_bottom_center(bbox)),
        confidence: detections
            .iter()
            .filter_map(|d| d.confidence)
            .reduce(f32::max),
        source_query: object.label.clone(),
    }
}

fn best_detection_match(bbox: &[f32; 4], detections: &[Detection], used: &[bool]) -> Option<usize> {
    detections
        .iter()
        .enumerate()
        .filter(|(index, _)| !used.get(*index).copied().unwrap_or(false))
        .map(|(index, detection)| (index, bbox_iou(*bbox, detection.bbox)))
        .max_by(|left, right| {
            left.1
                .partial_cmp(&right.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(index, _)| index)
}

fn normalized_query_key(value: &str) -> String {
    value.trim().to_ascii_lowercase()
}

fn bbox_bottom_center(bbox: [f32; 4]) -> [f32; 2] {
    [(bbox[0] + bbox[2]) * 0.5, bbox[3]]
}

fn union_bbox(left: [f32; 4], right: [f32; 4]) -> [f32; 4] {
    [
        left[0].min(right[0]),
        left[1].min(right[1]),
        left[2].max(right[2]),
        left[3].max(right[3]),
    ]
}

fn bbox_iou(left: [f32; 4], right: [f32; 4]) -> f32 {
    let ix0 = left[0].max(right[0]);
    let iy0 = left[1].max(right[1]);
    let ix1 = left[2].min(right[2]);
    let iy1 = left[3].min(right[3]);
    let iw = (ix1 - ix0).max(0.0);
    let ih = (iy1 - iy0).max(0.0);
    let intersection = iw * ih;
    let left_area = (left[2] - left[0]).max(0.0) * (left[3] - left[1]).max(0.0);
    let right_area = (right[2] - right[0]).max(0.0) * (right[3] - right[1]).max(0.0);
    let union = left_area + right_area - intersection;
    if union <= f32::EPSILON {
        0.0
    } else {
        intersection / union
    }
}

fn write_scene_ground_artifacts(output_dir: &Path, response: &Value) -> Result<(), String> {
    fs::create_dir_all(output_dir).map_err(|err| {
        format!(
            "failed to create scene-ground artifact directory {}: {err}",
            output_dir.display()
        )
    })?;
    for (key, file_name) in [
        ("manifest", "manifest.json"),
        ("asset_bindings", "asset_bindings.json"),
        ("grounding_evidence", "grounding_evidence.json"),
        ("grounded_layout", "grounded_layout.json"),
        ("commands", "commands.json"),
        ("stage_report", "stage_report.json"),
        ("e2e_summary", "summary.json"),
    ] {
        if let Some(value) = response.get(key) {
            write_json_file(&output_dir.join(file_name), value).map_err(|err| err.to_string())?;
        }
    }
    if let Some(bsn) = response.get("bsn").and_then(Value::as_str) {
        fs::write(output_dir.join("scene.bsn"), bsn).map_err(|err| {
            format!(
                "failed to write scene-ground BSN {}: {err}",
                output_dir.join("scene.bsn").display()
            )
        })?;
    }
    Ok(())
}

fn scene_asset_bindings_from_outputs(
    manifest: &SceneObjectManifest,
    selected_candidates: &[Value],
    asset_outputs: &Value,
) -> Result<Vec<SceneAssetBinding>, String> {
    let items = asset_outputs["items"]
        .as_array()
        .ok_or_else(|| "images_to_assets response missing items array".to_string())?;
    if items.len() != selected_candidates.len() {
        return Err(format!(
            "asset output count ({}) did not match selected candidate count ({})",
            items.len(),
            selected_candidates.len()
        ));
    }
    let objects_by_id = manifest
        .objects
        .iter()
        .map(|object| (object.id.as_str(), object))
        .collect::<HashMap<_, _>>();
    let mut bindings = Vec::with_capacity(manifest.objects.len().max(items.len()));
    let mut bindings_by_reuse_group = HashMap::new();
    for (item, selected) in items.iter().zip(selected_candidates.iter()) {
        let object_id = selected["object_id"]
            .as_str()
            .ok_or_else(|| "selected candidate missing object_id".to_string())?;
        let object = objects_by_id
            .get(object_id)
            .ok_or_else(|| format!("selected candidate references unknown object `{object_id}`"))?;
        let output_path = item["output_path"]
            .as_str()
            .ok_or_else(|| "asset output item missing output_path".to_string())?;
        let cache_key = item["cache_key"].as_str().map(ToOwned::to_owned);
        let local_aabb = item
            .get("local_aabb")
            .filter(|value| !value.is_null())
            .cloned()
            .and_then(|value| serde_json::from_value::<SceneAssetAabb>(value).ok())
            .or_else(|| {
                item.get("catalog_entry")
                    .and_then(|entry| entry.get("local_aabb"))
                    .filter(|value| !value.is_null())
                    .cloned()
                    .and_then(|value| serde_json::from_value::<CachedAssetAabb>(value).ok())
                    .map(cached_aabb_to_scene)
            });
        let binding = SceneAssetBinding {
            asset_id: sanitize_scene_identifier(&format!("{object_id}_asset")),
            object_id: object.id.clone(),
            label: object.label.clone(),
            aliases: object.aliases.clone(),
            path: Some(output_path.to_string()),
            cache_key: cache_key.clone(),
            reusable: cache_key.is_some()
                || object.instance_count > 1
                || object.instances.len() > 1
                || object.reuse_group.is_some(),
            source_image_path: selected["image_path"].as_str().map(ToOwned::to_owned),
            pipeline: item["synthesis_backend"].as_str().map(ToOwned::to_owned),
            local_aabb,
            canonical_frame: Some(inferred_scene_asset_frame(
                &object.label,
                &object.aliases,
                local_aabb,
                object.target_footprint_m,
            )),
            provenance: Some(burn_synth_scene::SceneAssetProvenance {
                run_id: "scene_build_from_image".to_string(),
                source_scene_path: manifest.source_scene_path.clone(),
                source_object_id: object.id.clone(),
                generated_by: "scene_build_from_image".to_string(),
            }),
        };
        let reuse_group = object
            .reuse_group
            .as_deref()
            .filter(|value| !value.is_empty())
            .unwrap_or(object.id.as_str())
            .to_string();
        bindings_by_reuse_group
            .entry(reuse_group)
            .or_insert_with(|| binding.clone());
        bindings.push(binding);
    }

    for object in &manifest.objects {
        if bindings
            .iter()
            .any(|binding| binding.object_id.as_str() == object.id.as_str())
        {
            continue;
        }
        let reuse_group = object
            .reuse_group
            .as_deref()
            .filter(|value| !value.is_empty())
            .unwrap_or(object.id.as_str());
        let Some(source_binding) = bindings_by_reuse_group.get(reuse_group) else {
            return Err(format!(
                "scene object `{}` was not selected and has no reusable asset binding for reuse group `{reuse_group}`",
                object.id
            ));
        };
        let mut binding = source_binding.clone();
        binding.asset_id = sanitize_scene_identifier(&format!("{}_asset", object.id));
        binding.object_id = object.id.clone();
        binding.label = object.label.clone();
        binding.aliases = object.aliases.clone();
        binding.reusable = true;
        if let Some(provenance) = binding.provenance.as_mut() {
            provenance.source_object_id = object.id.clone();
        }
        bindings.push(binding);
    }
    Ok(bindings)
}

fn sanitize_scene_identifier(value: &str) -> String {
    let mut output = String::with_capacity(value.len());
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
            output.push(ch);
        } else {
            output.push('_');
        }
    }
    if output.is_empty() {
        "asset".to_string()
    } else {
        output
    }
}

fn atomic_write_json(path: &Path, value: &Value) -> Result<(), String> {
    ensure_parent_dir(path).map_err(|err| err.to_string())?;
    let tmp = path.with_extension("json.tmp");
    let bytes = serde_json::to_vec_pretty(value)
        .map_err(|err| format!("failed to serialize scene command: {err}"))?;
    fs::write(&tmp, bytes).map_err(|err| format!("failed to write {}: {err}", tmp.display()))?;
    fs::rename(&tmp, path).map_err(|err| {
        format!(
            "failed to atomically replace scene command file {}: {err}",
            path.display()
        )
    })
}

fn read_scene_status(path: &Path) -> Result<Value, String> {
    let content = fs::read_to_string(path)
        .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    serde_json::from_str(&content)
        .map_err(|err| format!("failed to parse scene status {}: {err}", path.display()))
}

fn wait_scene_status(path: &Path, sequence: u64, timeout: Duration) -> Result<Value, String> {
    let started = Instant::now();
    let mut last_error = None;
    while started.elapsed() < timeout {
        match read_scene_status(path) {
            Ok(status) => {
                let acknowledged = status
                    .get("last_sequence")
                    .and_then(Value::as_u64)
                    .map(|last| last >= sequence)
                    .unwrap_or(false);
                if acknowledged {
                    return Ok(status);
                }
            }
            Err(err) => last_error = Some(err),
        }
        thread::sleep(Duration::from_millis(25));
    }
    Err(format!(
        "timed out waiting for scene status {} to acknowledge sequence {sequence}{}",
        path.display(),
        last_error
            .map(|err| format!("; last read error: {err}"))
            .unwrap_or_default()
    ))
}

fn success_response(id: Option<Value>, result: Value) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id.unwrap_or(Value::Null),
        "result": result,
    })
}

fn error_response(id: Option<Value>, code: i32, message: String) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id.unwrap_or(Value::Null),
        "error": {
            "code": code,
            "message": message,
        }
    })
}

fn success_tool_result(payload: Value) -> Value {
    let text = serde_json::to_string_pretty(&payload)
        .unwrap_or_else(|_| "{\"error\":\"failed to render tool payload\"}".to_string());
    json!({
        "content": [
            {
                "type": "text",
                "text": text,
            }
        ],
        "structuredContent": payload,
    })
}

fn error_tool_result(message: String) -> Value {
    json!({
        "isError": true,
        "content": [
            {
                "type": "text",
                "text": message,
            }
        ]
    })
}

fn tool_defs() -> Vec<Value> {
    vec![
        json!({
            "name": "image_to_foreground",
            "description": "Extract foreground alpha from an input image and write a PNG with transparency.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_image_path": { "type": "string", "description": "Path to input image file." },
                    "output_image_path": { "type": "string", "description": "Optional output path (defaults to *_foreground.png)." },
                    "rmbg_model": { "type": "string", "enum": ["rmbg14", "rmbg2"], "description": "Optional RMBG model override." },
                    "dry_run": { "type": "boolean", "description": "Skip model inference and just write a pass-through output image." }
                },
                "required": ["input_image_path"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "image_to_mesh",
            "description": "Run image-to-mesh synthesis and write a GLB mesh output.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_image_path": { "type": "string", "description": "Path to input image file." },
                    "output_mesh_path": { "type": "string", "description": "Optional output GLB path (defaults to *_mesh.glb)." },
                    "rmbg_model": { "type": "string", "enum": ["rmbg14", "rmbg2"], "description": "Optional RMBG model override." },
                    "synthesis_models": { "type": "array", "items": { "type": "string", "enum": ["triposg", "trellis"] }, "description": "Optional mesh synthesis model list override, ordered by preference." },
                    "backend": { "type": "string", "enum": ["cpu", "wgpu", "cuda"], "description": "Optional backend override." },
                    "target_faces": { "type": "integer", "description": "Optional target face count for mesh simplification." },
                    "dry_run": { "type": "boolean", "description": "Skip model inference and emit a canonical cube mesh." }
                },
                "required": ["input_image_path"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "image_to_splat",
            "description": "Run TripoSplat image-to-Gaussian-splat synthesis and write a .splat or .ply output.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_image_path": { "type": "string", "description": "Path to input image file." },
                    "output_splat_path": { "type": "string", "description": "Optional output path (defaults to *_splat.splat)." },
                    "output_format": { "type": "string", "enum": ["splat", "ply"], "description": "Optional splat output format." },
                    "rmbg_model": { "type": "string", "enum": ["rmbg14", "rmbg2"], "description": "Optional RMBG model override." },
                    "backend": { "type": "string", "enum": ["cpu", "wgpu", "cuda"], "description": "Optional backend override." },
                    "dry_run": { "type": "boolean", "description": "Skip model inference and emit a canonical debug splat cloud." }
                },
                "required": ["input_image_path"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "images_to_assets",
            "description": "Run batched image-to-asset synthesis over multiple images with shared model loading and chunk planning.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_image_paths": { "type": "array", "items": { "type": "string" }, "description": "Input image paths to process in one batch request." },
                    "output_dir": { "type": "string", "description": "Optional output directory for per-input output names." },
                    "output_paths": { "type": "array", "items": { "type": "string" }, "description": "Optional explicit output path per input." },
                    "output_format": { "type": "string", "enum": ["auto", "glb", "splat", "ply"], "description": "Optional output format. Auto writes GLB for meshes and .splat for Gaussian splats." },
                    "rmbg_model": { "type": "string", "enum": ["rmbg14", "rmbg2"], "description": "Optional RMBG model override." },
                    "synthesis_models": { "type": "array", "items": { "type": "string", "enum": ["triposg", "trellis", "triposplat"] }, "description": "Optional synthesis model list override, ordered by preference." },
                    "backend": { "type": "string", "enum": ["cpu", "wgpu", "cuda"], "description": "Optional backend override." },
                    "target_faces": { "type": "integer", "description": "Optional target face count for mesh simplification." },
                    "batch_size": { "type": "integer", "description": "Optional explicit chunk size; omit for server default/auto." },
                    "batch_vram_mb": { "type": "integer", "description": "Optional VRAM budget in MB for auto chunking." },
                    "trellis_pbr": { "type": "boolean", "description": "Enable TRELLIS UV/material texture baking through the Rust/Burn o_voxel export path for lifted GLB assets." },
                    "trellis_pbr_texture_size": { "type": "integer", "description": "TRELLIS PBR texture size." },
                    "promote_to_catalog": { "type": "boolean", "description": "Also add generated assets to the shared Bevy catalog/cache for later reuse. Defaults to false for direct batch conversion." },
                    "dry_run": { "type": "boolean", "description": "Skip model inference and emit canonical debug assets." }
                },
                "required": ["input_image_paths"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "scene_prepare_build",
            "description": "Prepare a formal OpenAI scene-builder run offline: validate paths and return strict schemas/prompts without calling OpenAI.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "source_scene_path": { "type": "string", "description": "Source scene image path." },
                    "object_reference_image_path": { "type": "string", "description": "Optional isolated-object style reference image; defaults to docs/input_chair.jpg." },
                    "output_dir": { "type": "string", "description": "Run output directory under tmp/runs." },
                    "candidate_count": { "type": "integer", "description": "Object image candidates per reusable object." },
                    "quality_profile": { "type": "string", "enum": ["draft", "quality"] },
                    "allow_catalog_reuse": { "type": "boolean", "description": "Whether the planner may consider existing catalog assets." }
                },
                "required": ["source_scene_path"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "scene_plan_objects",
            "description": "Use the raw OpenAI API to create a strict object manifest from a source scene image. Requires OPENAI_API_KEY.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "source_scene_path": { "type": "string" },
                    "object_reference_image_path": { "type": "string" },
                    "output_dir": { "type": "string" },
                    "candidate_count": { "type": "integer" },
                    "quality_profile": { "type": "string", "enum": ["draft", "quality"] },
                    "allow_catalog_reuse": { "type": "boolean" }
                },
                "required": ["source_scene_path"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "scene_generate_object_images",
            "description": "Use the raw OpenAI Image API to generate isolated object-image candidates from a scene manifest, source crop, source scene image, and docs/input_chair.jpg-style reference. Requires OPENAI_API_KEY.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "source_scene_path": { "type": "string" },
                    "manifest": { "type": "object", "description": "SceneObjectManifest returned by scene_plan_objects." },
                    "object_reference_image_path": { "type": "string" },
                    "output_dir": { "type": "string" },
                    "candidate_count": { "type": "integer" },
                    "quality_profile": { "type": "string", "enum": ["draft", "quality"] }
                },
                "required": ["source_scene_path", "manifest"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "scene_build_from_image",
            "description": "Quality-first OpenAI scene build: plan objects, generate source-preserving isolated object images, lift selected candidates through RMBG+TRELLIS, generate grounded restricted BSN from image bboxes plus asset AABBs, validate it, and optionally apply to Bevy. Requires OPENAI_API_KEY.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "source_scene_path": { "type": "string" },
                    "object_reference_image_path": { "type": "string" },
                    "output_dir": { "type": "string" },
                    "candidate_count": { "type": "integer" },
                    "candidate_retry_attempts": { "type": "integer", "description": "Maximum guarded image-generation attempts per object. Defaults to candidate_count." },
                    "candidate_batch_size": { "type": "integer", "description": "Generated image candidates requested per retry attempt. Defaults to 1 so weak candidates can be retried without overwriting artifacts." },
                    "min_reconstruction_score": { "type": "number", "description": "Minimum isolated-object reconstruction suitability score before TRELLIS lifting. Defaults to the canonical scene threshold." },
                    "quality_profile": { "type": "string", "enum": ["draft", "quality"] },
                    "allow_catalog_reuse": { "type": "boolean" },
                    "lift_assets": { "type": "boolean", "description": "When false, stop after object image generation." },
                    "target_faces": { "type": "integer" },
                    "batch_size": { "type": "integer" },
                    "batch_vram_mb": { "type": "integer" },
                    "trellis_pbr": { "type": "boolean", "description": "Enable TRELLIS UV/material texture baking through the Rust/Burn o_voxel export path for lifted GLB assets." },
                    "trellis_pbr_texture_size": { "type": "integer", "description": "TRELLIS PBR texture size." },
                    "promote_to_catalog": { "type": "boolean", "description": "Add lifted objects to the shared Bevy catalog/cache for later reuse. Defaults to true; fresh scene mode still does not read existing catalog assets while planning." },
                    "write_artifacts": { "type": "boolean", "description": "Write structured e2e artifacts such as selected candidates, asset outputs, grounded layout, commands, summary, and scene.bsn to output_dir. Defaults to true." },
                    "clear_existing": { "type": "boolean" },
                    "apply": { "type": "boolean" },
                    "feedback": { "type": "boolean", "description": "Run bounded render-capture-feedback placement validation/refinement. Defaults to true for full scene builds." },
                    "feedback_iters": { "type": "integer", "description": "Maximum feedback iterations. Defaults to 3." },
                    "feedback_keep_viewer": { "type": "boolean", "description": "Leave the temporary feedback viewer running after completion." },
                    "feedback_capture_dir": { "type": "string", "description": "Optional feedback artifact directory. Defaults to output_dir/iterations." },
                    "feedback_threshold_profile": { "type": "string", "enum": ["loose", "standard", "strict"] }
                },
                "required": ["source_scene_path"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "scene_ground",
            "description": "Recompute source-scene composition from an existing object manifest, asset bindings, and optional grounding evidence without regenerating object images or TRELLIS assets.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "source_scene_path": { "type": "string" },
                    "manifest": { "type": "object", "description": "SceneObjectManifest from scene_build_from_image or scene_plan_objects." },
                    "asset_bindings": {
                        "type": "array",
                        "items": { "type": "object" },
                        "description": "Generated asset bindings with asset_id plus path/cache_key/local_aabb."
                    },
                    "grounding_evidence": { "type": "object", "description": "Optional SceneGroundingEvidence. When omitted, manifest bbox/contact points are used as an explicit fallback." },
                    "output_dir": { "type": "string" },
                    "composition_mode": { "type": "string", "enum": ["heuristic", "cv-grounded"] },
                    "depth_provider": { "type": "string", "enum": ["none", "depth-pro"] },
                    "locator": { "type": "string", "enum": ["manifest", "locate-anything"] },
                    "locate_anything_backend": { "type": "string", "enum": ["python-reference", "burn-native"], "description": "Optional backend override when locator is locate-anything. Defaults to the server --locate-anything-backend setting." },
                    "clear_existing": { "type": "boolean" },
                    "apply": { "type": "boolean" },
                    "feedback": { "type": "boolean" },
                    "feedback_iters": { "type": "integer" },
                    "feedback_keep_viewer": { "type": "boolean" },
                    "feedback_capture_dir": { "type": "string" },
                    "feedback_threshold_profile": { "type": "string", "enum": ["loose", "standard", "strict"] }
                },
                "required": ["source_scene_path", "manifest", "asset_bindings"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "scene_plan_bsn",
            "description": "Plan a grounded restricted synth_scene_v1 BSN from an existing object manifest and generated asset bindings, using source-image bbox contact points, class scale priors, and asset AABBs; then validate commands before optional Bevy apply.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "manifest": { "type": "object", "description": "SceneObjectManifest returned by scene_plan_objects or scene_build_from_image." },
                    "asset_bindings": {
                        "type": "array",
                        "items": { "type": "object" },
                        "description": "Generated asset bindings with asset_id plus path or cache_key. local_aabb is used for ground-plane bottom-fit and scale when present."
                    },
                    "clear_existing": { "type": "boolean" },
                    "apply": { "type": "boolean", "description": "When true, send commands to Bevy scene bridge." }
                },
                "required": ["manifest", "asset_bindings"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "scene_apply_bsn",
            "description": "Validate restricted synth_scene_v1 BSN against explicit generated asset bindings and optionally apply it to the Bevy scene bridge.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "bsn": { "type": "string", "description": "Restricted synth_scene_v1 text." },
                    "asset_bindings": {
                        "type": "array",
                        "items": { "type": "object" },
                        "description": "Generated asset bindings with asset_id plus path or cache_key."
                    },
                    "clear_existing": { "type": "boolean" },
                    "apply": { "type": "boolean", "description": "When true, send commands to Bevy scene bridge." }
                },
                "required": ["bsn", "asset_bindings"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "scene_status",
            "description": "Read the latest Bevy scene bridge status, including cache entries, world items, camera, and screenshots.",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }
        }),
        json!({
            "name": "scene_project_status",
            "description": "Read camera/world status plus per-object projected screen-space evidence from the Bevy scene bridge.",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }
        }),
        json!({
            "name": "scene_list_assets",
            "description": "List cached assets and spawned world items from the latest Bevy scene bridge status.",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }
        }),
        json!({
            "name": "scene_spawn_cached",
            "description": "Spawn an asset already present in the Bevy mesh/splat cache.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "cache_key": { "type": "string", "description": "Cache key to spawn." },
                    "translation": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 },
                    "rotation": { "type": "array", "items": { "type": "number" }, "minItems": 4, "maxItems": 4 },
                    "scale": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 },
                    "select": { "type": "boolean", "description": "Select the spawned entity." }
                },
                "required": ["cache_key"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "scene_spawn_path",
            "description": "Spawn a GLB mesh asset file directly into the Bevy scene.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "GLB mesh path to spawn." },
                    "translation": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 },
                    "rotation": { "type": "array", "items": { "type": "number" }, "minItems": 4, "maxItems": 4 },
                    "scale": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 },
                    "select": { "type": "boolean", "description": "Select the spawned entity." }
                },
                "required": ["path"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "scene_delete",
            "description": "Delete a spawned cached asset by cache key, delete the selection, or clear selection.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "cache_key": { "type": "string", "description": "Cache key to delete." },
                    "selected": { "type": "boolean", "description": "Delete the current selection when true; clear selection when false and no cache key is provided." }
                },
                "additionalProperties": false
            }
        }),
        json!({
            "name": "scene_clear",
            "description": "Clear all spawned cache-backed scene items from the Bevy scene.",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }
        }),
        json!({
            "name": "scene_set_camera",
            "description": "Set the Bevy scene camera transform and optional orbit state.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "translation": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 },
                    "rotation": { "type": "array", "items": { "type": "number" }, "minItems": 4, "maxItems": 4 },
                    "focus": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 },
                    "yaw": { "type": "number" },
                    "pitch": { "type": "number" },
                    "radius": { "type": "number" },
                    "vertical_fov": { "type": "number" }
                },
                "required": ["translation", "rotation"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "scene_save",
            "description": "Flush the Bevy scene cache/world state.",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }
        }),
        json!({
            "name": "scene_capture",
            "description": "Capture a screenshot from the Bevy primary window and wait for the image file.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_path": { "type": "string", "description": "Screenshot path to write." }
                },
                "required": ["output_path"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "scene_compose_assets",
            "description": "Create deterministic Bevy placements from source-image object boxes and generated asset bindings; optionally apply them to the live scene.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "reference_objects": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": { "type": "string" },
                                "label": { "type": "string" },
                                "aliases": { "type": "array", "items": { "type": "string" } },
                                "bbox": { "type": "array", "items": { "type": "number" }, "minItems": 4, "maxItems": 4, "description": "Normalized source-image box [x_min, y_min, x_max, y_max]." }
                            },
                            "required": ["label", "bbox"],
                            "additionalProperties": false
                        }
                    },
                    "assets": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "reference_id": { "type": "string" },
                                "label": { "type": "string" },
                                "aliases": { "type": "array", "items": { "type": "string" } },
                                "path": { "type": "string" },
                                "cache_key": { "type": "string" },
                                "local_aabb": { "type": "object", "description": "Optional asset local bounds {min:[x,y,z], max:[x,y,z]} for ground-fit scaling." },
                                "select": { "type": "boolean" }
                            },
                            "additionalProperties": false
                        }
                    },
            "apply": { "type": "boolean", "description": "When true, send spawn commands to the configured Bevy scene bridge." },
                    "clear_existing": { "type": "boolean", "description": "When true, clear existing scene instances before placing generated assets." },
                    "layout_width": { "type": "number" },
                    "layout_depth": { "type": "number" },
                    "y": { "type": "number" },
                    "min_scale": { "type": "number" },
                    "scale_multiplier": { "type": "number" }
                },
                "required": ["reference_objects", "assets"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "scene_validate_layout",
            "description": "Validate a composed Bevy scene against source-image object boxes using semantic label matching, object counts, normalized layout, and optional screenshot image similarity.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "reference_objects": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": { "type": "string" },
                                "label": { "type": "string" },
                                "aliases": { "type": "array", "items": { "type": "string" } },
                                "bbox": { "type": "array", "items": { "type": "number" }, "minItems": 4, "maxItems": 4 }
                            },
                            "required": ["label", "bbox"],
                            "additionalProperties": false
                        }
                    },
                    "scene_status": { "type": "object", "description": "Optional scene status JSON. Omit to read the configured scene_status_path." },
                    "source_image_path": { "type": "string" },
                    "rendered_image_path": { "type": "string" },
                    "thresholds": {
                        "type": "object",
                        "properties": {
                            "min_semantic_score": { "type": "number" },
                            "min_layout_score": { "type": "number" },
                            "min_overall_score": { "type": "number" },
                            "max_extra_objects": { "type": "integer" },
                            "min_image_similarity": { "type": "number" }
                        },
                        "additionalProperties": false
                    }
                },
                "required": ["reference_objects"],
                "additionalProperties": false
            }
        }),
    ]
}

fn read_framed_json<R: BufRead + Read>(reader: &mut R) -> io::Result<Option<Value>> {
    let mut content_length = None;
    let mut saw_header = false;

    loop {
        let mut line = String::new();
        let bytes = reader.read_line(&mut line)?;
        if bytes == 0 {
            if !saw_header {
                return Ok(None);
            }
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "unexpected EOF while reading MCP headers",
            ));
        }
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            break;
        }
        saw_header = true;
        if let Some((name, value)) = trimmed.split_once(':')
            && name.eq_ignore_ascii_case("Content-Length")
        {
            let parsed = value.trim().parse::<usize>().map_err(|err| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid Content-Length header: {err}"),
                )
            })?;
            content_length = Some(parsed);
        }
    }

    let content_length = content_length.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "missing Content-Length header in MCP message",
        )
    })?;
    let mut payload = vec![0u8; content_length];
    reader.read_exact(&mut payload)?;
    let value = serde_json::from_slice::<Value>(&payload).map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid MCP JSON payload: {err}"),
        )
    })?;
    Ok(Some(value))
}

fn write_framed_json<W: Write>(writer: &mut W, value: &Value) -> io::Result<()> {
    let payload = serde_json::to_vec(value).map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("failed to serialize MCP JSON payload: {err}"),
        )
    })?;
    write!(writer, "Content-Length: {}\r\n\r\n", payload.len())?;
    writer.write_all(&payload)?;
    writer.flush()
}

#[derive(Debug, Deserialize)]
struct RpcRequest {
    pub id: Option<Value>,
    pub method: String,
    #[serde(default)]
    pub params: Option<Value>,
}

#[derive(Debug, Default, Deserialize)]
struct InitializeParams {
    #[serde(rename = "protocolVersion")]
    pub protocol_version: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ToolsCallParams {
    pub name: String,
    #[serde(default)]
    pub arguments: Value,
}

#[derive(Debug, Deserialize)]
struct ForegroundToolArgs {
    #[serde(alias = "image_path")]
    pub input_image_path: PathBuf,
    #[serde(default, alias = "output_path")]
    pub output_image_path: Option<PathBuf>,
    #[serde(default)]
    pub rmbg_model: Option<ForegroundModel>,
    #[serde(default)]
    pub dry_run: bool,
}

#[derive(Debug, Deserialize)]
struct MeshToolArgs {
    #[serde(alias = "image_path")]
    pub input_image_path: PathBuf,
    #[serde(default, alias = "output_path")]
    pub output_mesh_path: Option<PathBuf>,
    #[serde(default)]
    pub output_format: Option<MeshOutputFormat>,
    #[serde(default)]
    pub rmbg_model: Option<ForegroundModel>,
    #[serde(default)]
    pub synthesis_models: Option<Vec<SynthesisModel>>,
    #[serde(default)]
    pub backend: Option<InferenceBackend>,
    #[serde(default)]
    pub target_faces: Option<usize>,
    #[serde(default)]
    pub dry_run: bool,
}

#[derive(Debug, Deserialize)]
struct SplatToolArgs {
    #[serde(alias = "image_path")]
    pub input_image_path: PathBuf,
    #[serde(default, alias = "output_path")]
    pub output_splat_path: Option<PathBuf>,
    #[serde(default)]
    pub output_format: Option<AssetOutputFormat>,
    #[serde(default)]
    pub rmbg_model: Option<ForegroundModel>,
    #[serde(default)]
    pub backend: Option<InferenceBackend>,
    #[serde(default)]
    pub dry_run: bool,
}

#[derive(Debug, Deserialize)]
struct ImagesToAssetsToolArgs {
    #[serde(default, alias = "image_paths")]
    pub input_image_paths: Vec<PathBuf>,
    #[serde(default)]
    pub output_dir: Option<PathBuf>,
    #[serde(default)]
    pub output_paths: Option<Vec<PathBuf>>,
    #[serde(default)]
    pub output_format: Option<AssetOutputFormat>,
    #[serde(default)]
    pub rmbg_model: Option<ForegroundModel>,
    #[serde(default)]
    pub synthesis_models: Option<Vec<SynthesisModel>>,
    #[serde(default)]
    pub backend: Option<InferenceBackend>,
    #[serde(default)]
    pub target_faces: Option<usize>,
    #[serde(default)]
    pub batch_size: Option<usize>,
    #[serde(default)]
    pub batch_vram_mb: Option<u64>,
    #[serde(default)]
    pub trellis_pbr: Option<bool>,
    #[serde(default)]
    pub trellis_pbr_texture_size: Option<usize>,
    #[serde(default)]
    pub promote_to_catalog: bool,
    #[serde(default)]
    pub dry_run: bool,
}

#[derive(Debug, Deserialize)]
struct ScenePrepareBuildArgs {
    pub source_scene_path: PathBuf,
    #[serde(default)]
    pub object_reference_image_path: Option<PathBuf>,
    #[serde(default)]
    pub output_dir: Option<PathBuf>,
    #[serde(default)]
    pub candidate_count: Option<usize>,
    #[serde(default)]
    pub quality_profile: Option<SceneQualityProfile>,
    #[serde(default)]
    pub allow_catalog_reuse: bool,
}

#[derive(Debug, Deserialize)]
struct SceneGenerateObjectImagesArgs {
    pub source_scene_path: PathBuf,
    pub manifest: SceneObjectManifest,
    #[serde(default)]
    pub object_reference_image_path: Option<PathBuf>,
    #[serde(default)]
    pub output_dir: Option<PathBuf>,
    #[serde(default)]
    pub candidate_count: Option<usize>,
    #[serde(default)]
    pub quality_profile: Option<SceneQualityProfile>,
}

#[derive(Debug, Deserialize)]
struct SceneBuildFromImageArgs {
    pub source_scene_path: PathBuf,
    #[serde(default)]
    pub object_reference_image_path: Option<PathBuf>,
    #[serde(default)]
    pub output_dir: Option<PathBuf>,
    #[serde(default)]
    pub candidate_count: Option<usize>,
    #[serde(default)]
    pub candidate_retry_attempts: Option<usize>,
    #[serde(default)]
    pub candidate_batch_size: Option<usize>,
    #[serde(default)]
    pub min_reconstruction_score: Option<f32>,
    #[serde(default)]
    pub quality_profile: Option<SceneQualityProfile>,
    #[serde(default)]
    pub allow_catalog_reuse: bool,
    #[serde(default = "default_scene_lift_assets")]
    pub lift_assets: bool,
    #[serde(default)]
    pub target_faces: Option<usize>,
    #[serde(default)]
    pub batch_size: Option<usize>,
    #[serde(default)]
    pub batch_vram_mb: Option<u64>,
    #[serde(default)]
    pub trellis_pbr: Option<bool>,
    #[serde(default)]
    pub trellis_pbr_texture_size: Option<usize>,
    #[serde(default = "default_scene_promote_to_catalog")]
    pub promote_to_catalog: bool,
    #[serde(default = "default_scene_write_artifacts")]
    pub write_artifacts: bool,
    #[serde(default)]
    pub apply: bool,
    #[serde(default = "default_scene_clear_existing")]
    pub clear_existing: bool,
    #[serde(default = "default_scene_feedback")]
    pub feedback: bool,
    #[serde(default = "default_scene_feedback_iters")]
    pub feedback_iters: usize,
    #[serde(default)]
    pub feedback_keep_viewer: bool,
    #[serde(default)]
    pub feedback_capture_dir: Option<PathBuf>,
    #[serde(default = "default_scene_feedback_threshold_profile")]
    pub feedback_threshold_profile: FeedbackThresholdProfile,
}

#[derive(Debug, Deserialize)]
struct SceneGroundToolArgs {
    pub source_scene_path: PathBuf,
    pub manifest: SceneObjectManifest,
    pub asset_bindings: Vec<SceneAssetBinding>,
    #[serde(default)]
    pub grounding_evidence: Option<SceneGroundingEvidence>,
    #[serde(default)]
    pub output_dir: Option<PathBuf>,
    #[serde(default = "default_scene_composition_mode")]
    pub composition_mode: SceneCompositionMode,
    #[serde(default = "default_scene_depth_provider")]
    pub depth_provider: SceneDepthProvider,
    #[serde(default = "default_scene_locator_provider")]
    pub locator: SceneLocatorProvider,
    #[serde(default)]
    pub locate_anything_backend: Option<LocateAnythingBackend>,
    #[serde(default = "default_scene_clear_existing")]
    pub clear_existing: bool,
    #[serde(default)]
    pub apply: bool,
    #[serde(default)]
    pub feedback: bool,
    #[serde(default = "default_scene_feedback_iters")]
    pub feedback_iters: usize,
    #[serde(default)]
    pub feedback_keep_viewer: bool,
    #[serde(default)]
    pub feedback_capture_dir: Option<PathBuf>,
    #[serde(default = "default_scene_feedback_threshold_profile")]
    pub feedback_threshold_profile: FeedbackThresholdProfile,
}

#[derive(Clone, Debug)]
struct SceneFeedbackOptions {
    max_iters: usize,
    keep_viewer: bool,
    capture_dir: Option<PathBuf>,
    threshold_profile: FeedbackThresholdProfile,
}

struct SceneFeedbackIterationContext<'a> {
    capture_root: &'a Path,
    manifest: &'a SceneObjectManifest,
    asset_bindings: &'a [SceneAssetBinding],
    grounded_layout: &'a GroundedSceneLayout,
    initial_commands: Vec<Value>,
    max_iters: usize,
    threshold_profile: FeedbackThresholdProfile,
}

#[derive(Clone, Copy, Debug)]
struct SceneFeedbackThresholds {
    max_center_error: f32,
    max_contact_error: f32,
    max_area_log2_error: f32,
    min_overall_score: f32,
    max_seating_table_overlap_fraction: f32,
    max_seating_table_penetration_m: f32,
    max_seating_seating_overlap_fraction: f32,
    max_seating_seating_penetration_m: f32,
}

impl FeedbackThresholdProfile {
    fn thresholds(self) -> SceneFeedbackThresholds {
        match self {
            Self::Loose => SceneFeedbackThresholds {
                max_center_error: 0.18,
                max_contact_error: 0.22,
                max_area_log2_error: 1.20,
                min_overall_score: 0.55,
                max_seating_table_overlap_fraction: 0.45,
                max_seating_table_penetration_m: 0.30,
                max_seating_seating_overlap_fraction: 0.16,
                max_seating_seating_penetration_m: 0.08,
            },
            Self::Standard => SceneFeedbackThresholds {
                max_center_error: 0.10,
                max_contact_error: 0.14,
                max_area_log2_error: 0.65,
                min_overall_score: 0.65,
                max_seating_table_overlap_fraction: 0.35,
                max_seating_table_penetration_m: 0.25,
                max_seating_seating_overlap_fraction: 0.10,
                max_seating_seating_penetration_m: 0.05,
            },
            Self::Strict => SceneFeedbackThresholds {
                max_center_error: 0.06,
                max_contact_error: 0.09,
                max_area_log2_error: 0.35,
                min_overall_score: 0.82,
                max_seating_table_overlap_fraction: 0.25,
                max_seating_table_penetration_m: 0.18,
                max_seating_seating_overlap_fraction: 0.06,
                max_seating_seating_penetration_m: 0.03,
            },
        }
    }
}

#[derive(Debug, Deserialize)]
struct ScenePlanBsnArgs {
    pub manifest: SceneObjectManifest,
    pub asset_bindings: Vec<SceneAssetBinding>,
    #[serde(default)]
    pub apply: bool,
    #[serde(default = "default_scene_clear_existing")]
    pub clear_existing: bool,
}

#[derive(Debug, Deserialize)]
struct SceneApplyBsnArgs {
    pub bsn: String,
    pub asset_bindings: Vec<SceneAssetBinding>,
    #[serde(default)]
    pub clear_existing: bool,
    #[serde(default)]
    pub apply: bool,
}

#[derive(Debug, Deserialize)]
struct SceneSpawnCachedArgs {
    pub cache_key: String,
    #[serde(default)]
    pub translation: Option<[f32; 3]>,
    #[serde(default)]
    pub rotation: Option<[f32; 4]>,
    #[serde(default)]
    pub scale: Option<[f32; 3]>,
    #[serde(default)]
    pub select: bool,
}

#[derive(Debug, Deserialize)]
struct SceneSpawnPathArgs {
    pub path: PathBuf,
    #[serde(default)]
    pub translation: Option<[f32; 3]>,
    #[serde(default)]
    pub rotation: Option<[f32; 4]>,
    #[serde(default)]
    pub scale: Option<[f32; 3]>,
    #[serde(default)]
    pub select: bool,
}

#[derive(Debug, Default, Deserialize)]
struct SceneDeleteArgs {
    #[serde(default)]
    pub cache_key: Option<String>,
    #[serde(default)]
    pub selected: bool,
}

#[derive(Debug, Deserialize)]
struct SceneSetCameraArgs {
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    #[serde(default)]
    pub focus: Option<[f32; 3]>,
    #[serde(default)]
    pub yaw: Option<f32>,
    #[serde(default)]
    pub pitch: Option<f32>,
    #[serde(default)]
    pub radius: Option<f32>,
    #[serde(default)]
    pub vertical_fov: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct SceneCaptureArgs {
    #[serde(alias = "path")]
    pub output_path: PathBuf,
}

impl ForegroundModel {
    fn as_str(self) -> &'static str {
        match self {
            ForegroundModel::Rmbg14 => "rmbg14",
            ForegroundModel::Rmbg2 => "rmbg2",
        }
    }
}

impl SynthesisModel {
    fn as_str(self) -> &'static str {
        match self {
            SynthesisModel::Triposg => "triposg",
            SynthesisModel::Trellis => "trellis",
            SynthesisModel::Triposplat => "triposplat",
        }
    }
}

impl InferenceBackend {
    fn as_str(self) -> &'static str {
        match self {
            InferenceBackend::Cpu => "cpu",
            InferenceBackend::Wgpu => "wgpu",
            InferenceBackend::Cuda => "cuda",
        }
    }
}

impl MeshOutputFormat {
    fn as_str(self) -> &'static str {
        match self {
            MeshOutputFormat::Obj => "obj",
            MeshOutputFormat::Gltf => "gltf",
            MeshOutputFormat::Glb => "glb",
        }
    }
}

impl AssetOutputFormat {
    fn as_str(self) -> &'static str {
        match self {
            AssetOutputFormat::Auto => "auto",
            AssetOutputFormat::Glb => "glb",
            AssetOutputFormat::Splat => "splat",
            AssetOutputFormat::Ply => "ply",
        }
    }
}

fn runtime_foreground_model_str(value: burn_synth::ForegroundModel) -> &'static str {
    match value {
        burn_synth::ForegroundModel::Rmbg14 => "rmbg14",
        burn_synth::ForegroundModel::Rmbg2 => "rmbg2",
    }
}

fn runtime_synthesis_model_str(value: burn_synth::SynthesisModel) -> &'static str {
    match value {
        burn_synth::SynthesisModel::Triposg => "triposg",
        burn_synth::SynthesisModel::Trellis => "trellis",
        burn_synth::SynthesisModel::Triposplat => "triposplat",
    }
}

fn runtime_backend_str(value: burn_synth::InferenceBackend) -> &'static str {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_physical_layout() -> FeedbackPhysicalLayout {
        FeedbackPhysicalLayout {
            pairs: Vec::new(),
            corrections: HashMap::new(),
            object_failures: HashMap::new(),
            hard_failure_count: 0,
            warning_count: 0,
            object_failure_count: 0,
            max_overlap_fraction_smaller: 0.0,
            min_signed_clearance_m: 0.0,
            table_center_xz: None,
            footprint_centers: HashMap::new(),
        }
    }

    fn test_feedback_placement(
        object_id: &str,
        label: &str,
        translation: [f32; 3],
        source_bbox: [f32; 4],
    ) -> GroundedScenePlacement {
        GroundedScenePlacement {
            entity_id: object_id.to_string(),
            asset_id: object_id.to_string(),
            object_id: object_id.to_string(),
            instance_id: None,
            label: label.to_string(),
            source_bbox,
            contact_pixel: bbox_bottom_center_for_test(source_bbox),
            ground_point: translation,
            translation,
            rotation_y_degrees: 0.0,
            scale: [1.0, 1.0, 1.0],
            local_aabb: SceneAssetAabb {
                min: [-0.5, 0.0, -0.5],
                max: [0.5, 1.0, 0.5],
            },
            target_footprint_m: [1.0, 1.0],
        }
    }

    fn bbox_bottom_center_for_test(bbox: [f32; 4]) -> [f32; 2] {
        [(bbox[0] + bbox[2]) * 0.5, bbox[3]]
    }
    use burn_synth_scene::SceneCamera;
    use clap::Parser;

    #[test]
    fn server_args_default_to_balanced_quality_defaults() {
        let args = ServerArgs::parse_from(["burn_synth_mcp"]);
        let config = ServerConfig::from_args(args);
        assert_eq!(config.quality, QualityPreset::Balanced);
        assert_eq!(config.num_steps, 20);
        assert_eq!(config.num_tokens, 1024);
        assert_eq!(config.guidance_scale, 7.0);
        assert_eq!(config.flash_octree_depth, 8);
        assert_eq!(config.flash_min_resolution, 31);
        assert_eq!(config.flash_mini_grid_num, 4);
        assert_eq!(config.flash_num_chunks, 8192);
    }

    #[test]
    fn server_args_accept_scene_ground_command() {
        let args = ServerArgs::parse_from([
            "burn_synth_mcp",
            "scene-ground",
            "--source-scene-path",
            "/tmp/source.jpg",
            "--manifest",
            "/tmp/manifest.json",
            "--asset-bindings",
            "/tmp/assets.json",
            "--composition-mode",
            "cv-grounded",
            "--depth-provider",
            "depth-pro",
            "--locator",
            "locate-anything",
            "--locate-anything-backend",
            "burn-native",
            "--feedback-iters",
            "5",
        ]);
        let Some(ServerCommand::SceneGround(command)) = args.command else {
            panic!("expected scene-ground subcommand");
        };
        assert_eq!(command.composition_mode, SceneCompositionMode::CvGrounded);
        assert_eq!(command.depth_provider, SceneDepthProvider::DepthPro);
        assert_eq!(command.locator, SceneLocatorProvider::LocateAnything);
        assert_eq!(
            command.locate_anything_backend,
            Some(LocateAnythingBackend::BurnNative)
        );
        assert_eq!(command.feedback_iters, 5);
    }

    #[test]
    fn server_args_accept_global_locate_anything_backend() {
        let args =
            ServerArgs::parse_from(["burn_synth_mcp", "--locate-anything-backend", "burn-native"]);
        let config = ServerConfig::from_args(args);
        assert_eq!(
            config.locate_anything_backend,
            LocateAnythingBackend::BurnNative
        );
    }

    #[test]
    fn locate_anything_burn_native_cache_key_ignores_artifact_run_root() {
        let base = LocateAnythingRuntimeConfig {
            model_root: PathBuf::from("assets/models/LocateAnything-3B"),
            backend: LocateAnythingRuntimeBackend::BurnNative,
            allow_experimental_native_detect: true,
            decode_mode: DecodeMode::Hybrid,
            max_new_tokens: 1024,
            in_token_limit: LOCATE_ANYTHING_SAFE_IN_TOKEN_LIMIT,
            run_root: PathBuf::from("tmp/runs/a"),
            ..LocateAnythingRuntimeConfig::default()
        };
        let mut same_runtime = base.clone();
        same_runtime.run_root = PathBuf::from("tmp/runs/b");
        assert_eq!(
            LocateAnythingBurnNativeCacheKey::from_config(&base),
            LocateAnythingBurnNativeCacheKey::from_config(&same_runtime)
        );

        let mut different_tokens = base;
        different_tokens.in_token_limit += 1;
        assert_ne!(
            LocateAnythingBurnNativeCacheKey::from_config(&same_runtime),
            LocateAnythingBurnNativeCacheKey::from_config(&different_tokens)
        );

        let mut different_decode_filter = same_runtime.clone();
        different_decode_filter.top_p = None;
        assert_ne!(
            LocateAnythingBurnNativeCacheKey::from_config(&same_runtime),
            LocateAnythingBurnNativeCacheKey::from_config(&different_decode_filter)
        );
    }

    #[test]
    fn locate_anything_burn_native_scene_ground_reuses_runtime_when_enabled() {
        if std::env::var("LOCATE_ANYTHING_MCP_BURN_NATIVE_CACHE_SMOKE").is_err() {
            eprintln!(
                "skipping: set LOCATE_ANYTHING_MCP_BURN_NATIVE_CACHE_SMOKE=1 to run WGPU LocateAnything MCP cache smoke"
            );
            return;
        }
        let Some(repo_root) = find_repo_root_for_test() else {
            eprintln!("skipping LocateAnything MCP cache smoke; repo root not found");
            return;
        };
        let image_path =
            PathBuf::from("/media/mosure/dolos/demo/Cisco/reconstruction/045-LYS01-3-Galaxy.jpg");
        let model_root = repo_root.join("assets/models/LocateAnything-3B");
        if !image_path.exists() || !model_root.join("config.json").exists() {
            eprintln!(
                "skipping LocateAnything MCP cache smoke; missing {} or {}",
                image_path.display(),
                model_root.display()
            );
            return;
        }

        let mut server = McpServer::new(ServerConfig {
            locate_anything_backend: LocateAnythingBackend::BurnNative,
            locate_anything_model_root: model_root,
            ..ServerConfig::from_args(ServerArgs::parse_from(["burn_synth_mcp"]))
        });
        let manifest = SceneObjectManifest {
            source_scene_path: image_path.display().to_string(),
            scene_calibration: None,
            objects: vec![
                SceneObjectSpec {
                    id: "conference_table".to_string(),
                    label: "conference table".to_string(),
                    aliases: vec!["table".to_string()],
                    bbox: [0.386, 0.519, 0.659, 1.0],
                    instances: Vec::new(),
                    representative_instance_id: None,
                    reuse_group: Some("conference_table".to_string()),
                    instance_count: 1,
                    object_prompt: "conference table".to_string(),
                    camera_hint: None,
                    rotation_hint_degrees: None,
                    target_footprint_m: Some([3.2, 1.2]),
                },
                SceneObjectSpec {
                    id: "conference_chair".to_string(),
                    label: "conference chair".to_string(),
                    aliases: vec!["chair".to_string()],
                    bbox: [0.166, 0.63, 0.36, 1.0],
                    instances: Vec::new(),
                    representative_instance_id: None,
                    reuse_group: Some("conference_chair".to_string()),
                    instance_count: 1,
                    object_prompt: "conference chair".to_string(),
                    camera_hint: None,
                    rotation_hint_degrees: None,
                    target_footprint_m: Some([0.65, 0.65]),
                },
            ],
        };
        let assets = vec![
            SceneAssetBinding {
                asset_id: "conference_table_asset".to_string(),
                object_id: "conference_table".to_string(),
                label: "conference table".to_string(),
                aliases: Vec::new(),
                path: None,
                cache_key: Some("test/conference_table".to_string()),
                reusable: true,
                source_image_path: None,
                pipeline: None,
                local_aabb: Some(SceneAssetAabb {
                    min: [-1.6, 0.0, -0.6],
                    max: [1.6, 0.2, 0.6],
                }),
                canonical_frame: Some(SceneAssetFrame {
                    yaw_offset_degrees: 0.0,
                    footprint_m: Some([3.2, 1.2]),
                }),
                provenance: None,
            },
            SceneAssetBinding {
                asset_id: "conference_chair_asset".to_string(),
                object_id: "conference_chair".to_string(),
                label: "conference chair".to_string(),
                aliases: Vec::new(),
                path: None,
                cache_key: Some("test/conference_chair".to_string()),
                reusable: true,
                source_image_path: None,
                pipeline: None,
                local_aabb: Some(SceneAssetAabb {
                    min: [-0.32, 0.0, -0.32],
                    max: [0.32, 1.1, 0.32],
                }),
                canonical_frame: Some(SceneAssetFrame {
                    yaw_offset_degrees: 0.0,
                    footprint_m: Some([0.64, 0.64]),
                }),
                provenance: None,
            },
        ];
        let root = repo_root.join("tmp/runs").join(format!(
            "{}_locateanything_mcp_burn_native_cache_smoke",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock before unix epoch")
                .as_millis()
        ));
        let first_dir = root.join("first");
        let second_dir = root.join("second");
        let make_args = |output_dir: PathBuf| SceneGroundToolArgs {
            source_scene_path: image_path.clone(),
            manifest: manifest.clone(),
            asset_bindings: assets.clone(),
            grounding_evidence: None,
            output_dir: Some(output_dir),
            composition_mode: SceneCompositionMode::CvGrounded,
            depth_provider: SceneDepthProvider::None,
            locator: SceneLocatorProvider::LocateAnything,
            locate_anything_backend: Some(LocateAnythingBackend::BurnNative),
            clear_existing: true,
            apply: false,
            feedback: false,
            feedback_iters: 0,
            feedback_keep_viewer: false,
            feedback_capture_dir: None,
            feedback_threshold_profile: FeedbackThresholdProfile::Standard,
        };

        server
            .call_scene_ground(make_args(first_dir.clone()))
            .expect("first burn-native scene-ground");
        server
            .call_scene_ground(make_args(second_dir.clone()))
            .expect("second burn-native scene-ground");
        let first_metadata: Value =
            read_json_path(&first_dir.join("locate_anything_burn_native/metadata.json")).unwrap();
        let second_metadata: Value =
            read_json_path(&second_dir.join("locate_anything_burn_native/metadata.json")).unwrap();
        assert_eq!(first_metadata["runtime_cache_hit"], json!(false));
        assert_eq!(second_metadata["runtime_cache_hit"], json!(true));
    }

    #[test]
    fn tool_schema_exposes_scene_ground() {
        let tools = tool_defs();
        let names = tools
            .iter()
            .filter_map(|tool| tool["name"].as_str())
            .collect::<Vec<_>>();
        assert!(names.contains(&"scene_ground"));
        let scene_ground = tools
            .iter()
            .find(|tool| tool["name"] == "scene_ground")
            .expect("scene_ground schema");
        assert_eq!(
            scene_ground["inputSchema"]["properties"]["locate_anything_backend"]["enum"],
            json!(["python-reference", "burn-native"])
        );
    }

    #[test]
    fn locate_anything_evidence_maps_detections_to_instances() {
        let manifest = SceneObjectManifest {
            source_scene_path: "/tmp/source.jpg".to_string(),
            scene_calibration: None,
            objects: vec![SceneObjectSpec {
                id: "chairs".to_string(),
                label: "chair".to_string(),
                aliases: Vec::new(),
                bbox: [0.1, 0.2, 0.8, 0.9],
                instances: vec![
                    SceneObjectInstanceSpec {
                        id: Some("chair_left".to_string()),
                        bbox: [0.10, 0.40, 0.30, 0.90],
                        contact: None,
                        rotation_hint_degrees: None,
                        facing_yaw_degrees: None,
                        side: None,
                        slot_index: None,
                        target_footprint_m: None,
                    },
                    SceneObjectInstanceSpec {
                        id: Some("chair_right".to_string()),
                        bbox: [0.60, 0.40, 0.80, 0.90],
                        contact: None,
                        rotation_hint_degrees: None,
                        facing_yaw_degrees: None,
                        side: None,
                        slot_index: None,
                        target_footprint_m: None,
                    },
                ],
                representative_instance_id: None,
                reuse_group: Some("chair".to_string()),
                instance_count: 2,
                object_prompt: "chair".to_string(),
                camera_hint: None,
                rotation_hint_degrees: None,
                target_footprint_m: None,
            }],
        };
        let detections = vec![
            Detection {
                label: "chair".to_string(),
                bbox: [0.61, 0.41, 0.79, 0.91],
                point: None,
                confidence: Some(0.8),
                source_query: "chair".to_string(),
            },
            Detection {
                label: "chair".to_string(),
                bbox: [0.11, 0.39, 0.29, 0.89],
                point: None,
                confidence: Some(0.9),
                source_query: "chair".to_string(),
            },
        ];
        let evidence = locate_anything_evidence_from_detections(
            &manifest,
            Path::new("/tmp/source.jpg"),
            detections,
            "locate_anything_test",
        )
        .unwrap();
        assert_eq!(evidence.detections.len(), 2);
        let left = evidence
            .objects
            .iter()
            .find(|object| object.instance_id.as_deref() == Some("chair_left"))
            .unwrap();
        let right = evidence
            .objects
            .iter()
            .find(|object| object.instance_id.as_deref() == Some("chair_right"))
            .unwrap();
        assert_eq!(
            left.detection.as_ref().unwrap().bbox,
            [0.11, 0.39, 0.29, 0.89]
        );
        assert_eq!(
            right.detection.as_ref().unwrap().bbox,
            [0.61, 0.41, 0.79, 0.91]
        );
        let object_union = evidence
            .objects
            .iter()
            .find(|object| object.object_id == "chairs" && object.instance_id.is_none())
            .unwrap();
        assert_eq!(
            object_union.detection.as_ref().unwrap().bbox,
            [0.11, 0.39, 0.79, 0.91]
        );
    }

    #[test]
    fn depth_annotation_adds_contact_geometry_and_footprint_hints() {
        let detection = Detection {
            label: "conference chair".to_string(),
            bbox: [0.25, 0.25, 0.75, 0.75],
            point: Some([0.5, 0.75]),
            confidence: Some(0.9),
            source_query: "conference chair".to_string(),
        };
        let mut evidence = SceneGroundingEvidence {
            source_image_path: "/tmp/source.jpg".to_string(),
            depth: None,
            detections: vec![detection.clone()],
            camera: burn_synth_scene::EstimatedCamera::default(),
            floor: EstimatedFloorPlane::default(),
            objects: vec![ObjectGroundingEvidence {
                object_id: "chair".to_string(),
                instance_id: None,
                reuse_group: Some("chair".to_string()),
                detection: Some(detection),
                asset_id: None,
                contact_pixel: None,
                depth_stats: None,
                candidate_floor_contact_rays: Vec::new(),
                metric_contact_point_m: None,
                target_footprint_m: None,
                provenance: Vec::new(),
            }],
        };
        let depth_map = SceneDepthMapEvidence {
            depth_m: vec![
                2.0, 2.0, 2.0, 2.0, //
                2.0, 2.2, 2.2, 2.0, //
                2.0, 2.4, 2.4, 2.0, //
                3.0, 3.0, 3.0, 3.0,
            ],
            width: 4,
            height: 4,
            intrinsics: CameraIntrinsics {
                fx: 4.0,
                fy: 4.0,
                cx: 1.5,
                cy: 1.5,
                width: 4,
                height: 4,
            },
            focal_length_px: Some(4.0),
            vertical_fov_degrees: Some(53.0),
        };

        let summary =
            annotate_grounding_evidence_with_depth_map(&mut evidence, &depth_map, "depth_pro");
        let object = evidence.objects.first().unwrap();

        assert_eq!(summary.annotated_objects, 1);
        assert_eq!(summary.depth_map_size, [4, 4]);
        assert!(summary.floor_sample_count > 0);
        assert!(object.depth_stats.as_ref().unwrap().contact_m.unwrap() > 0.0);
        assert_eq!(object.candidate_floor_contact_rays.len(), 1);
        assert!(object.metric_contact_point_m.unwrap()[2] > 0.0);
        assert!(object.target_footprint_m.unwrap()[0] >= 0.42);
        assert!(object.provenance.contains(&"depth_pro".to_string()));
    }

    #[test]
    fn server_args_quality_and_explicit_overrides_map_to_runtime_config() {
        let args = ServerArgs::parse_from([
            "burn_synth_mcp",
            "--quality",
            "fast",
            "--num-steps",
            "18",
            "--guidance-scale",
            "6.5",
        ]);
        let config = ServerConfig::from_args(args);
        assert_eq!(config.quality, QualityPreset::Fast);
        assert_eq!(config.num_steps, 18);
        assert_eq!(config.num_tokens, 512);
        assert_eq!(config.guidance_scale, 6.5);
        assert_eq!(config.flash_octree_depth, 7);
        assert_eq!(config.flash_min_resolution, 31);
        assert_eq!(config.flash_mini_grid_num, 2);
        assert_eq!(config.flash_num_chunks, 4096);

        let runtime = config.runtime_config();
        assert_eq!(runtime.num_steps, 18);
        assert_eq!(runtime.num_tokens, 512);
        assert_eq!(runtime.guidance_scale, 6.5);
        assert_eq!(runtime.flash_extract.octree_depth, 7);
        assert_eq!(runtime.flash_extract.min_resolution, 31);
        assert_eq!(runtime.flash_extract.mini_grid_num, 2);
        assert_eq!(runtime.flash_extract.num_chunks, 4096);
    }

    #[test]
    fn server_args_accept_scene_build_subcommand() {
        let args = ServerArgs::parse_from([
            "burn_synth_mcp",
            "--backend",
            "wgpu",
            "--trellis-quality",
            "low",
            "scene-build",
            "--source-scene-path",
            "/tmp/scene.jpg",
            "--output-dir",
            "/tmp/scene-run",
            "--candidate-count",
            "2",
            "--candidate-retry-attempts",
            "3",
            "--candidate-batch-size",
            "1",
            "--batch-size",
            "0",
            "--trellis-pbr",
            "false",
            "--apply",
        ]);
        assert_eq!(args.backend, InferenceBackend::Wgpu);
        assert_eq!(args.trellis_quality, TrellisQuality::Low);
        let Some(ServerCommand::SceneBuild(command)) = args.command else {
            panic!("expected scene-build subcommand");
        };
        assert_eq!(command.source_scene_path, PathBuf::from("/tmp/scene.jpg"));
        assert_eq!(command.output_dir, Some(PathBuf::from("/tmp/scene-run")));
        assert_eq!(command.candidate_count, Some(2));
        assert_eq!(command.candidate_retry_attempts, Some(3));
        assert_eq!(command.candidate_batch_size, Some(1));
        assert_eq!(command.batch_size, Some(0));
        assert!(!command.trellis_pbr);
        assert!(command.apply);
        assert!(command.feedback);
        assert_eq!(command.feedback_iters, 3);
        assert_eq!(
            command.feedback_threshold_profile,
            FeedbackThresholdProfile::Standard
        );
    }

    #[test]
    fn server_args_accept_scene_feedback_replay_rebuild_flag() {
        let args = ServerArgs::parse_from([
            "burn_synth_mcp",
            "scene-feedback-replay",
            "--output-dir",
            "/tmp/scene-run",
            "--feedback-iters",
            "4",
            "--rebuild-commands-from-grounded-layout",
        ]);
        let Some(ServerCommand::SceneFeedbackReplay(command)) = args.command else {
            panic!("expected scene-feedback-replay subcommand");
        };
        assert_eq!(command.output_dir, PathBuf::from("/tmp/scene-run"));
        assert_eq!(command.feedback_iters, 4);
        assert!(command.rebuild_commands_from_grounded_layout);
    }

    #[test]
    fn dotenv_var_parses_plain_export_and_quoted_values() {
        let root = unique_test_dir("dotenv");
        fs::create_dir_all(&root).expect("create temp dir");
        let path = root.join(".env");
        fs::write(
            &path,
            "\n# comment\nOPENAI_API_KEY=plain-key # local\nexport OPENAI_PROJECT_ID='proj_123'\nOPENAI_BASE_URL=\"https://example.test\"\n",
        )
        .expect("write .env");

        assert_eq!(
            dotenv_var(&path, "OPENAI_API_KEY").as_deref(),
            Some("plain-key")
        );
        assert_eq!(
            dotenv_var(&path, "OPENAI_PROJECT_ID").as_deref(),
            Some("proj_123")
        );
        assert_eq!(
            dotenv_var(&path, "OPENAI_BASE_URL").as_deref(),
            Some("https://example.test")
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn tool_list_includes_batch_splat_and_scene_tools() {
        let tools = tool_defs();
        let names = tools
            .iter()
            .filter_map(|tool| tool["name"].as_str())
            .collect::<Vec<_>>();
        for expected in [
            "images_to_assets",
            "image_to_splat",
            "scene_status",
            "scene_prepare_build",
            "scene_plan_objects",
            "scene_generate_object_images",
            "scene_build_from_image",
            "scene_plan_bsn",
            "scene_apply_bsn",
            "scene_spawn_cached",
            "scene_spawn_path",
            "scene_clear",
            "scene_capture",
            "scene_project_status",
            "scene_compose_assets",
            "scene_validate_layout",
        ] {
            assert!(names.contains(&expected), "missing tool {expected}");
        }
    }

    #[test]
    fn select_scene_candidates_rejects_low_reconstruction_score() {
        let manifest = SceneObjectManifest {
            source_scene_path: "/tmp/scene.jpg".to_string(),
            scene_calibration: None,
            objects: vec![burn_synth_scene::SceneObjectSpec {
                id: "coffee_table".to_string(),
                label: "coffee table".to_string(),
                aliases: Vec::new(),
                bbox: [0.2, 0.2, 0.8, 0.8],
                instances: Vec::new(),
                representative_instance_id: None,
                reuse_group: None,
                instance_count: 1,
                object_prompt: "white coffee table".to_string(),
                camera_hint: None,
                rotation_hint_degrees: None,
                target_footprint_m: None,
            }],
        };
        let candidates = vec![burn_synth_scene::ObjectImageCandidate {
            object_id: "coffee_table".to_string(),
            candidate_index: 0,
            image_path: "/tmp/table.png".to_string(),
            raw_image_path: None,
            prompt_hash: "hash".to_string(),
            score: DEFAULT_SCENE_RECONSTRUCTION_IMAGE_SCORE - 0.01,
            provider_request_id: None,
        }];
        let err = select_scene_candidates(&manifest, &candidates).unwrap_err();
        assert!(err.contains("not suitable for TRELLIS/RMBG reconstruction"));
    }

    #[test]
    fn scene_build_tool_schema_exposes_retry_and_artifact_controls() {
        let tools = tool_defs();
        let scene_build = tools
            .iter()
            .find(|tool| tool["name"] == "scene_build_from_image")
            .expect("scene_build_from_image tool");
        let properties = &scene_build["inputSchema"]["properties"];
        for key in [
            "candidate_retry_attempts",
            "candidate_batch_size",
            "min_reconstruction_score",
            "batch_size",
            "batch_vram_mb",
            "write_artifacts",
            "feedback",
            "feedback_iters",
            "feedback_keep_viewer",
            "feedback_capture_dir",
            "feedback_threshold_profile",
        ] {
            assert!(
                properties.get(key).is_some(),
                "scene_build_from_image schema missing {key}"
            );
        }
    }

    #[test]
    fn write_scene_build_artifacts_persists_structured_e2e_outputs() {
        let dir = std::env::temp_dir().join(format!(
            "burn_synth_mcp_artifact_test_{}",
            next_scene_sequence()
        ));
        let _ = fs::remove_dir_all(&dir);
        let response = json!({
            "tool": "scene_build_from_image",
            "manifest": {
                "source_scene_path": "/tmp/scene.jpg",
                "objects": []
            },
            "candidates": [],
            "selected_candidates": [],
            "asset_outputs": {
                "items": [
                    {
                        "input_image_path": "/tmp/chair.png",
                        "output_path": "/tmp/chair.glb",
                        "cache_key": "chair-cache",
                        "vertices": 12,
                        "faces": 8,
                        "local_aabb": {
                            "min": [-0.5, 0.0, -0.5],
                            "max": [0.5, 1.0, 0.5]
                        }
                    }
                ]
            },
            "grounded_layout": {
                "placements": [
                    {
                        "object_id": "chair",
                        "asset_id": "chair_asset",
                        "translation": [0.0, 0.0, 1.0],
                        "scale": [1.0, 1.0, 1.0],
                        "target_footprint_m": [0.85, 0.85]
                    }
                ]
            },
            "commands": [{ "type": "clear_scene" }],
            "stage_report": [{ "stage": "generate_object_candidates", "elapsed_ms": 7 }],
            "e2e_summary": {
                "ok": true,
                "elapsed_ms": 12
            },
            "bsn": "synth_scene_v1 {}"
        });

        write_scene_build_artifacts(&dir, &response).unwrap();

        assert!(dir.join("manifest.json").exists());
        assert!(dir.join("asset_outputs.json").exists());
        assert!(dir.join("stage_report.json").exists());
        assert!(dir.join("summary.json").exists());
        assert!(dir.join("scene.bsn").exists());
        assert!(dir.join("scene_build_response_structured.json").exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn scene_asset_bindings_prefer_promoted_catalog_cache_keys() {
        let manifest = SceneObjectManifest {
            source_scene_path: "/tmp/scene.jpg".to_string(),
            scene_calibration: None,
            objects: vec![burn_synth_scene::SceneObjectSpec {
                id: "chair_left".to_string(),
                label: "conference chair".to_string(),
                aliases: vec!["chair".to_string()],
                bbox: [0.1, 0.2, 0.3, 0.7],
                instances: Vec::new(),
                representative_instance_id: None,
                reuse_group: None,
                instance_count: 1,
                object_prompt: "green conference chair".to_string(),
                camera_hint: None,
                rotation_hint_degrees: None,
                target_footprint_m: None,
            }],
        };
        let selected = vec![json!({
            "object_id": "chair_left",
            "reuse_group": "chair_left",
            "label": "conference chair",
            "image_path": "/tmp/chair_candidate.png",
            "candidate_index": 0,
            "score": 0.91,
            "prompt_hash": "abc",
        })];
        let asset_outputs = json!({
            "items": [
                {
                    "output_path": "/tmp/chair_candidate_mesh.glb",
                    "cache_key": "central-chair-cache-key",
                    "synthesis_backend": "trellis",
                    "local_aabb": {
                        "min": [-0.5, 0.0, -0.5],
                        "max": [0.5, 1.0, 0.5]
                    }
                }
            ]
        });

        let bindings =
            scene_asset_bindings_from_outputs(&manifest, &selected, &asset_outputs).unwrap();
        assert_eq!(bindings.len(), 1);
        assert_eq!(
            bindings[0].cache_key.as_deref(),
            Some("central-chair-cache-key")
        );
        assert!(bindings[0].reusable);
        assert_eq!(
            bindings[0].local_aabb.as_ref().map(|aabb| aabb.max[1]),
            Some(1.0)
        );

        let bsn = "synth_scene_v1 {\nasset chair_left_asset = \"generated:chair_left_asset\";\nspawn chair uses chair_left_asset translation [0.0,0.0,0.0] rotation_y 0.0 scale [1.0,1.0,1.0];\n}";
        let plan = parse_scene_bsn(bsn, &bindings).unwrap();
        let commands = scene_plan_to_mcp_commands(&plan, &bindings, true).unwrap();
        assert_eq!(commands[0]["type"], "clear_scene");
        assert_eq!(commands[1]["type"], "spawn_cached");
        assert_eq!(commands[1]["cache_key"], json!("central-chair-cache-key"));
    }

    #[test]
    fn scene_asset_bindings_mark_explicit_instances_reusable() {
        let manifest = SceneObjectManifest {
            source_scene_path: "/tmp/scene.jpg".to_string(),
            scene_calibration: None,
            objects: vec![burn_synth_scene::SceneObjectSpec {
                id: "chair_group".to_string(),
                label: "chair group".to_string(),
                aliases: vec!["chair".to_string()],
                bbox: [0.1, 0.2, 0.8, 0.8],
                instances: vec![
                    burn_synth_scene::SceneObjectInstanceSpec {
                        id: Some("left".to_string()),
                        bbox: [0.1, 0.2, 0.25, 0.7],
                        contact: Some([0.18, 0.7]),
                        rotation_hint_degrees: None,
                        facing_yaw_degrees: None,
                        side: None,
                        slot_index: None,
                        target_footprint_m: None,
                    },
                    burn_synth_scene::SceneObjectInstanceSpec {
                        id: Some("right".to_string()),
                        bbox: [0.65, 0.2, 0.8, 0.7],
                        contact: Some([0.72, 0.7]),
                        rotation_hint_degrees: None,
                        facing_yaw_degrees: None,
                        side: None,
                        slot_index: None,
                        target_footprint_m: None,
                    },
                ],
                representative_instance_id: None,
                reuse_group: None,
                instance_count: 1,
                object_prompt: "one reusable chair".to_string(),
                camera_hint: None,
                rotation_hint_degrees: None,
                target_footprint_m: None,
            }],
        };
        let selected = vec![json!({
            "object_id": "chair_group",
            "reuse_group": "chair_group",
            "label": "chair group",
            "image_path": "/tmp/chair_candidate.png",
            "candidate_index": 0,
            "score": 0.91,
            "prompt_hash": "abc",
        })];
        let asset_outputs = json!({
            "items": [
                {
                    "output_path": "/tmp/chair_candidate_mesh.glb",
                    "synthesis_backend": "trellis"
                }
            ]
        });

        let bindings =
            scene_asset_bindings_from_outputs(&manifest, &selected, &asset_outputs).unwrap();

        assert_eq!(bindings.len(), 1);
        assert!(bindings[0].reusable);
    }

    #[test]
    fn scene_asset_bindings_expand_reused_groups_to_each_scene_object() {
        let manifest = SceneObjectManifest {
            source_scene_path: "/tmp/scene.jpg".to_string(),
            scene_calibration: None,
            objects: vec![
                burn_synth_scene::SceneObjectSpec {
                    id: "whiteboard_left".to_string(),
                    label: "left whiteboard".to_string(),
                    aliases: vec!["whiteboard".to_string()],
                    bbox: [0.05, 0.1, 0.35, 0.6],
                    instances: Vec::new(),
                    representative_instance_id: None,
                    reuse_group: Some("whiteboard".to_string()),
                    instance_count: 1,
                    object_prompt: "whiteboard on a stand".to_string(),
                    camera_hint: None,
                    rotation_hint_degrees: None,
                    target_footprint_m: None,
                },
                burn_synth_scene::SceneObjectSpec {
                    id: "whiteboard_right".to_string(),
                    label: "right whiteboard".to_string(),
                    aliases: vec!["whiteboard".to_string()],
                    bbox: [0.65, 0.1, 0.95, 0.6],
                    instances: Vec::new(),
                    representative_instance_id: None,
                    reuse_group: Some("whiteboard".to_string()),
                    instance_count: 1,
                    object_prompt: "whiteboard on a stand".to_string(),
                    camera_hint: None,
                    rotation_hint_degrees: None,
                    target_footprint_m: None,
                },
            ],
        };
        let selected = vec![json!({
            "object_id": "whiteboard_left",
            "reuse_group": "whiteboard",
            "label": "left whiteboard",
            "image_path": "/tmp/whiteboard_candidate.png",
            "candidate_index": 0,
            "score": 0.95,
            "prompt_hash": "abc",
        })];
        let asset_outputs = json!({
            "items": [
                {
                    "output_path": "/tmp/whiteboard_candidate_mesh.glb",
                    "cache_key": "whiteboard-cache-key",
                    "synthesis_backend": "trellis",
                    "local_aabb": {
                        "min": [-0.5, 0.0, -0.05],
                        "max": [0.5, 1.2, 0.05]
                    }
                }
            ]
        });

        let bindings =
            scene_asset_bindings_from_outputs(&manifest, &selected, &asset_outputs).unwrap();
        assert_eq!(bindings.len(), 2);
        let left = bindings
            .iter()
            .find(|binding| binding.object_id == "whiteboard_left")
            .unwrap();
        let right = bindings
            .iter()
            .find(|binding| binding.object_id == "whiteboard_right")
            .unwrap();
        assert_eq!(left.path, right.path);
        assert_eq!(right.label, "right whiteboard");
        assert!(right.reusable);

        let layout = grounded_scene_layout_for_manifest(&manifest, &bindings).unwrap();
        assert!(layout.bsn.contains("whiteboard_left"));
        assert!(layout.bsn.contains("whiteboard_right"));
    }

    #[test]
    fn scene_asset_quality_failures_gate_trellis_mesh_outputs() {
        let asset_outputs = json!({
            "items": [
                {
                    "asset_kind": "mesh",
                    "synthesis_backend": "trellis",
                    "output_path": "/tmp/chair.glb",
                    "mesh_quality_failures": [
                        "position-welded boundary edge ratio 0.4200 exceeds 0.0500"
                    ]
                },
                {
                    "asset_kind": "mesh",
                    "synthesis_backend": "triposg",
                    "output_path": "/tmp/legacy.glb",
                    "mesh_quality_failures": ["legacy warning"]
                }
            ]
        });

        let failures = scene_asset_quality_failures(&asset_outputs);

        assert_eq!(failures.len(), 1);
        assert!(failures[0].contains("/tmp/chair.glb"));
        assert!(failures[0].contains("boundary edge ratio"));
    }

    #[test]
    fn scene_commands_with_cache_reload_preserves_clear_first() {
        let commands = scene_commands_with_cache_reload(vec![
            json!({ "type": "clear_scene" }),
            json!({ "type": "spawn_cached", "cache_key": "chair" }),
            json!({ "type": "spawn_cached", "cache_key": "table" }),
        ]);

        assert_eq!(commands[0]["type"], "clear_scene");
        assert_eq!(commands[1]["type"], "reload_cache");
        assert_eq!(commands[2]["type"], "spawn_cached");
        assert_eq!(commands.len(), 4);
    }

    #[test]
    fn scene_interaction_lock_command_uses_viewer_control_protocol() {
        let command = scene_interaction_lock_command(true, "iterative scene composition");

        assert_eq!(command["type"], json!("set_interaction_lock"));
        assert_eq!(command["locked"], json!(true));
        assert_eq!(command["reason"], json!("iterative scene composition"));
    }

    #[test]
    fn feedback_deltas_adjust_spawn_and_camera_commands() {
        let commands = vec![
            json!({ "type": "clear_scene" }),
            json!({
                "type": "spawn_cached",
                "cache_key": "chair",
                "translation": [1.0, 0.5, 2.0],
                "scale": [1.0, 1.0, 1.0],
                "rotation": [0.0, 0.0, 0.0, 1.0]
            }),
            json!({
                "type": "set_camera",
                "translation": [0.0, 2.0, 5.0],
                "rotation": [0.0, 0.0, 0.0, 1.0],
                "focus": [0.0, 0.0, 0.0],
                "yaw": 180.0,
                "pitch": 25.0,
                "radius": 5.0,
                "vertical_fov": 72.0
            }),
        ];
        let deltas = json!({
            "objects": [{
                "index": 0,
                "translation_delta": [0.25, 0.0, -0.5],
                "scale_multiplier": 1.2,
                "yaw_delta_degrees": 12.0
            }],
            "camera": {
                "radius_multiplier": 0.9
            }
        });

        let adjusted = apply_feedback_deltas_to_commands(&commands, &deltas).unwrap();

        assert_eq!(adjusted[1]["translation"], json!([1.25, 0.5, 1.5]));
        let adjusted_scale = adjusted[1]["scale"]
            .as_array()
            .expect("spawn command keeps scale array");
        for component in adjusted_scale {
            let component = component.as_f64().expect("scale component is numeric");
            assert!((component - 1.2).abs() <= 1.0e-5);
        }
        let adjusted_yaw = quat_y_degrees(json_array4(&adjusted[1]["rotation"]).unwrap());
        assert!((adjusted_yaw - 12.0).abs() <= 1.0e-4);
        assert_eq!(adjusted[2]["radius"], json!(4.5));
    }

    #[test]
    fn feedback_deltas_apply_axis_scale_for_table_projection() {
        let commands = vec![json!({
            "type": "spawn_cached",
            "cache_key": "table",
            "translation": [0.0, 0.0, 0.0],
            "scale": [2.0, 0.5, 4.0],
        })];
        let deltas = json!({
            "objects": [{
                "translation_delta": [0.0, 0.0, 0.0],
                "scale_multiplier": 1.0,
                "scale_multiplier_xyz": [1.2, 1.0, 0.9]
            }]
        });

        let adjusted = apply_feedback_deltas_to_commands(&commands, &deltas).unwrap();

        let scale = json_array3(&adjusted[0]["scale"]).unwrap();
        assert!((scale[0] - 2.4).abs() <= 1.0e-5);
        assert!((scale[1] - 0.5).abs() <= 1.0e-5);
        assert!((scale[2] - 3.6).abs() <= 1.0e-5);
    }

    #[test]
    fn feedback_deltas_emit_axis_scale_for_skinny_table_projection() {
        let metrics = json!({
            "objects": [{
                "index": 0,
                "object_id": "conference_table",
                "label": "white rectangular conference table",
                "cache_key": "table-cache",
                "expected_bbox": [0.30, 0.48, 0.65, 0.96],
                "observed_bbox": [0.40, 0.44, 0.58, 0.95],
                "translation_delta": [0.0, 0.0, 0.0],
                "scale_multiplier": 1.22,
                "yaw_delta_degrees": 0.0
            }]
        });

        let deltas = feedback_layout_deltas(&metrics);
        let object = &deltas["objects"][0];
        let axis_scale = json_array3(&object["scale_multiplier_xyz"]).unwrap();

        assert_eq!(object["scale_source"], json!("axis_projection"));
        assert!(axis_scale[0] > 1.1);
        assert!((axis_scale[1] - 1.0).abs() <= 1.0e-6);
        assert!(axis_scale[2] < 1.02);
    }

    #[test]
    fn feedback_deltas_normalize_existing_reused_command_scales() {
        let commands = vec![
            json!({
                "type": "spawn_cached",
                "cache_key": "chair-cache",
                "translation": [0.0, 0.0, 0.0],
                "scale": [1.0, 1.0, 1.0],
            }),
            json!({
                "type": "spawn_cached",
                "cache_key": "chair-cache",
                "translation": [1.0, 0.0, 0.0],
                "scale": [2.0, 2.0, 2.0],
            }),
            json!({
                "type": "spawn_cached",
                "cache_key": "table-cache",
                "translation": [0.0, 0.0, 1.0],
                "scale": [0.75, 0.75, 0.75],
            }),
        ];
        let deltas = json!({
            "objects": [
                { "translation_delta": [0.0, 0.0, 0.0], "scale_multiplier": 1.0 },
                { "translation_delta": [0.0, 0.0, 0.0], "scale_multiplier": 1.0 },
                { "translation_delta": [0.0, 0.0, 0.0], "scale_multiplier": 1.0 }
            ]
        });

        let adjusted = apply_feedback_deltas_to_commands(&commands, &deltas).unwrap();

        assert_eq!(adjusted[0]["scale"], json!([1.5, 1.5, 1.5]));
        assert_eq!(adjusted[1]["scale"], json!([1.5, 1.5, 1.5]));
        assert_eq!(adjusted[2]["scale"], json!([0.75, 0.75, 0.75]));
    }

    #[test]
    fn feedback_deltas_share_scale_for_reused_instances() {
        let metrics = json!({
            "objects": [
                {
                    "index": 0,
                    "object_id": "chair",
                    "cache_key": "chair-cache",
                    "expected_bbox": [0.1, 0.1, 0.2, 0.3],
                    "observed_bbox": [0.1, 0.1, 0.3, 0.5],
                    "translation_delta": [0.0, 0.0, 0.0],
                    "scale_multiplier": 0.82,
                    "yaw_delta_degrees": 0.0
                },
                {
                    "index": 1,
                    "object_id": "chair",
                    "cache_key": "chair-cache",
                    "expected_bbox": [0.5, 0.1, 0.6, 0.3],
                    "observed_bbox": [0.5, 0.1, 0.55, 0.2],
                    "translation_delta": [0.0, 0.0, 0.0],
                    "scale_multiplier": 1.22,
                    "yaw_delta_degrees": 0.0
                },
                {
                    "index": 2,
                    "object_id": "table",
                    "cache_key": "table-cache",
                    "expected_bbox": [0.2, 0.4, 0.8, 0.6],
                    "observed_bbox": [0.2, 0.4, 0.8, 0.6],
                    "translation_delta": [0.0, 0.0, 0.0],
                    "scale_multiplier": 0.95,
                    "yaw_delta_degrees": 0.0
                }
            ]
        });

        let deltas = feedback_layout_deltas(&metrics);
        let objects = deltas["objects"].as_array().unwrap();
        let chair_scale_a = objects[0]["scale_multiplier"].as_f64().unwrap();
        let chair_scale_b = objects[1]["scale_multiplier"].as_f64().unwrap();
        let table_scale = objects[2]["scale_multiplier"].as_f64().unwrap();

        assert!((chair_scale_a - 1.02).abs() <= 1.0e-6);
        assert!((chair_scale_b - 1.02).abs() <= 1.0e-6);
        assert!((table_scale - 1.0).abs() <= 1.0e-6);
        assert_eq!(objects[0]["scale_group_key"], json!("chair-cache"));
        assert_eq!(objects[1]["scale_source"], json!("repeated_instance_group"));
        assert_eq!(objects[2]["scale_source"], json!("axis_projection"));
    }

    #[test]
    fn feedback_deltas_project_simultaneous_chair_moves_out_of_overlap() {
        let left_rect = FootprintRect {
            min_x: -1.0,
            min_z: -0.2,
            max_x: -0.6,
            max_z: 0.2,
        };
        let right_rect = FootprintRect {
            min_x: 0.6,
            min_z: -0.2,
            max_x: 1.0,
            max_z: 0.2,
        };
        let metrics = json!({
            "thresholds": {
                "max_seating_seating_overlap_fraction": 0.10,
                "max_seating_seating_penetration_m": 0.05
            },
            "objects": [
                {
                    "index": 0,
                    "object_id": "chair_left",
                    "cache_key": "chair-cache",
                    "expected_bbox": [0.2, 0.6, 0.35, 0.95],
                    "observed_bbox": [0.2, 0.6, 0.35, 0.95],
                    "translation_delta": [0.8, 0.0, 0.0],
                    "scale_multiplier": 1.0,
                    "yaw_delta_degrees": 0.0,
                    "physical_kind": "seating",
                    "world_footprint": {
                        "min_x": left_rect.min_x,
                        "min_z": left_rect.min_z,
                        "max_x": left_rect.max_x,
                        "max_z": left_rect.max_z
                    }
                },
                {
                    "index": 1,
                    "object_id": "chair_right",
                    "cache_key": "chair-cache",
                    "expected_bbox": [0.65, 0.6, 0.8, 0.95],
                    "observed_bbox": [0.65, 0.6, 0.8, 0.95],
                    "translation_delta": [-0.8, 0.0, 0.0],
                    "scale_multiplier": 1.0,
                    "yaw_delta_degrees": 0.0,
                    "physical_kind": "seating",
                    "world_footprint": {
                        "min_x": right_rect.min_x,
                        "min_z": right_rect.min_z,
                        "max_x": right_rect.max_x,
                        "max_z": right_rect.max_z
                    }
                }
            ]
        });

        let deltas = feedback_layout_deltas(&metrics);
        let objects = deltas["objects"].as_array().unwrap();
        let left_delta = json_array3(&objects[0]["translation_delta"]).unwrap();
        let right_delta = json_array3(&objects[1]["translation_delta"]).unwrap();
        let projected_left = left_rect.translated(left_delta);
        let projected_right = right_rect.translated(right_delta);

        assert!(left_delta[0] < 0.8);
        assert!(right_delta[0] > -0.8);
        assert!(
            projected_left.signed_clearance(projected_right)
                >= -FeedbackThresholdProfile::Standard
                    .thresholds()
                    .max_seating_seating_penetration_m
                    - 1.0e-4
        );
    }

    #[test]
    fn feedback_metrics_fail_chair_contained_inside_table_footprint() {
        let manifest = SceneObjectManifest {
            source_scene_path: "/tmp/source.jpg".to_string(),
            scene_calibration: None,
            objects: Vec::new(),
        };
        let table = test_feedback_placement(
            "conference_table",
            "conference table",
            [0.0, 0.0, 0.0],
            [0.30, 0.40, 0.70, 0.70],
        );
        let chair = test_feedback_placement(
            "conference_chair",
            "conference chair",
            [0.0, 0.0, 0.0],
            [0.45, 0.55, 0.55, 0.85],
        );
        let layout = GroundedSceneLayout {
            bsn: "scene {}".to_string(),
            placements: vec![table, chair],
            camera: SceneCamera {
                translation: [0.0, 2.0, 5.0],
                focus: [0.0, 0.0, 0.0],
                yaw: Some(180.0),
                pitch: Some(25.0),
                radius: Some(5.0),
                vertical_fov_degrees: Some(72.0),
            },
            rug_center: [0.0, 0.0, 0.0],
            rug_scale: [1.0, 1.0, 1.0],
        };
        let status = json!({
            "projected_items": [
                {
                    "cache_key": "table",
                    "screen_bbox": [0.30, 0.40, 0.70, 0.70],
                    "screen_contact": [0.50, 0.55],
                    "world_aabb": {
                        "min": [-1.5, 0.0, -0.6],
                        "max": [1.5, 0.4, 0.6]
                    }
                },
                {
                    "cache_key": "chair",
                    "screen_bbox": [0.45, 0.55, 0.55, 0.85],
                    "screen_contact": [0.50, 0.85],
                    "world_aabb": {
                        "min": [-0.25, 0.0, -0.25],
                        "max": [0.25, 1.0, 0.25]
                    }
                }
            ],
            "camera": { "radius": 5.0 }
        });

        let metrics = scene_feedback_metrics(
            &manifest,
            &layout,
            &status,
            Path::new("/tmp/iter.png"),
            FeedbackThresholdProfile::Standard.thresholds(),
            FeedbackThresholdProfile::Standard,
        )
        .unwrap();

        assert!(!metrics["passed"].as_bool().unwrap());
        assert_eq!(metrics["projection_passed"], json!(true));
        assert_eq!(metrics["physical_passed"], json!(false));
        assert_eq!(metrics["physical_layout"]["hard_failure_count"], json!(1));
        assert_eq!(
            metrics["physical_layout"]["pairs"][0]["failure_reasons"][0],
            json!("seating_center_inside_table")
        );
        assert_eq!(metrics["objects"][1]["physical_passed"], json!(false));
        let deltas = feedback_layout_deltas(&metrics);
        let translation_delta = json_array3(&deltas["objects"][1]["translation_delta"]).unwrap();
        assert!(translation_delta[0].abs() + translation_delta[2].abs() > 0.1);
    }

    #[test]
    fn feedback_selection_score_penalizes_hard_overlap_failures() {
        let clean = json!({
            "score": 0.60,
            "physical_layout": {
                "hard_failure_count": 0,
                "max_overlap_fraction_smaller": 0.0
            }
        });
        let overlapped = json!({
            "score": 0.95,
            "physical_layout": {
                "hard_failure_count": 1,
                "max_overlap_fraction_smaller": 1.0
            }
        });

        assert!(feedback_selection_score(&clean) > feedback_selection_score(&overlapped));
    }

    #[test]
    fn feedback_predictive_delta_prevents_move_into_table() {
        let placement = test_feedback_placement(
            "conference_chair",
            "conference chair",
            [0.0, 0.0, -1.0],
            [0.45, 0.7, 0.55, 0.95],
        );
        let footprints = vec![
            Some(FeedbackFootprint {
                index: 0,
                kind: FeedbackPhysicalKind::Table,
                rect: FootprintRect {
                    min_x: -1.0,
                    min_z: -0.5,
                    max_x: 1.0,
                    max_z: 0.5,
                },
            }),
            Some(FeedbackFootprint {
                index: 1,
                kind: FeedbackPhysicalKind::Seating,
                rect: FootprintRect {
                    min_x: -0.25,
                    min_z: -1.2,
                    max_x: 0.25,
                    max_z: -0.8,
                },
            }),
        ];

        let correction = feedback_predictive_physical_delta(
            1,
            &placement,
            [0.0, 0.0, 0.7],
            &footprints,
            FeedbackThresholdProfile::Standard.thresholds(),
        );

        assert!(correction[2] < -0.2);
    }

    #[test]
    fn feedback_yaw_uses_table_facing_target_when_available() {
        let mut physical = empty_physical_layout();
        physical.table_center_xz = Some([0.0, 0.0]);
        physical.footprint_centers.insert(0, [1.0, 0.0]);
        let placement = test_feedback_placement(
            "conference_chair",
            "conference chair",
            [1.0, 0.0, 0.0],
            [0.6, 0.5, 0.8, 0.9],
        );

        let correction = feedback_yaw_correction(0, &placement, 0.0, &physical);

        assert_eq!(correction.basis, "table-facing-yaw");
        assert!(correction.delta_degrees < -3.0);
    }

    #[test]
    fn feedback_deltas_damp_camera_ray_scale_until_contact_converges() {
        let metrics = json!({
            "objects": [
                {
                    "index": 0,
                    "object_id": "table",
                    "cache_key": "table-cache",
                    "expected_bbox": [0.2, 0.5, 0.8, 1.0],
                    "observed_bbox": [0.1, 0.2, 0.9, 1.2],
                    "translation_delta": [0.0, 0.0, -0.5],
                    "grounding_basis": "camera-ray-ground-plane",
                    "center_error": 0.24,
                    "contact_error": 0.31,
                    "scale_multiplier": 0.82,
                    "yaw_delta_degrees": 0.0
                }
            ]
        });

        let deltas = feedback_layout_deltas(&metrics);
        let scale = deltas["objects"][0]["scale_multiplier"].as_f64().unwrap();
        let camera_radius = deltas["camera"]["radius_multiplier"].as_f64().unwrap();

        assert!((scale - 1.0).abs() <= 1.0e-6);
        assert!((camera_radius - 1.0).abs() <= 1.0e-6);
    }

    #[test]
    fn feedback_yaw_uses_live_world_item_against_canonical_bsn_yaw() {
        let placement = GroundedScenePlacement {
            entity_id: "chair_1".to_string(),
            asset_id: "chair".to_string(),
            object_id: "chair".to_string(),
            instance_id: Some("chair_1".to_string()),
            label: "chair".to_string(),
            source_bbox: [0.1, 0.2, 0.3, 0.7],
            contact_pixel: [0.2, 0.7],
            ground_point: [0.0, 0.0, 0.0],
            translation: [0.0, 0.0, 0.0],
            rotation_y_degrees: 90.0,
            scale: [1.0, 1.0, 1.0],
            local_aabb: SceneAssetAabb {
                min: [-0.3, 0.0, -0.4],
                max: [0.3, 1.0, 0.4],
            },
            target_footprint_m: [0.6, 0.8],
        };

        let correction = feedback_yaw_correction(0, &placement, 20.0, &empty_physical_layout());

        assert_eq!(correction.basis, "canonical-bsn-yaw");
        assert!(correction.delta_degrees > 20.0);
    }

    #[test]
    fn status_world_item_yaw_matches_cache_key_when_order_differs() {
        let status = json!({
            "world_items": [
                {
                    "cache_key": "table-cache",
                    "rotation": quat_from_y_degrees(0.0),
                },
                {
                    "cache_key": "chair-cache",
                    "rotation": quat_from_y_degrees(-90.0),
                }
            ]
        });

        let yaw = status_world_item_yaw_degrees(&status, 0, Some("chair-cache")).unwrap();

        assert!((yaw + 90.0).abs() <= 1.0e-5);
    }

    #[test]
    fn feedback_status_prefers_capture_acknowledgement_for_screenshot_metrics() {
        let apply_ack = json!({
            "status": {
                "sequence": 1,
                "projected_items": [{
                    "screen_bbox": [0.1, 0.1, 0.2, 0.2]
                }]
            }
        });
        let capture_ack = json!({
            "acknowledgement": {
                "status": {
                    "sequence": 2,
                    "projected_items": [{
                        "screen_bbox": [0.3, 0.3, 0.4, 0.4]
                    }]
                }
            }
        });

        let status = McpServer::feedback_capture_status(&apply_ack, &capture_ack);

        assert_eq!(status["sequence"], json!(2));
        assert_eq!(
            status["projected_items"][0]["screen_bbox"],
            json!([0.3, 0.3, 0.4, 0.4])
        );
    }

    #[test]
    fn feedback_metrics_use_camera_ray_grounding_when_status_has_world_aabb() {
        let manifest = SceneObjectManifest {
            source_scene_path: "/tmp/source.jpg".to_string(),
            scene_calibration: None,
            objects: Vec::new(),
        };
        let layout = GroundedSceneLayout {
            bsn: "scene {}".to_string(),
            placements: vec![GroundedScenePlacement {
                entity_id: "chair_1".to_string(),
                asset_id: "chair".to_string(),
                object_id: "chair".to_string(),
                instance_id: Some("chair_1".to_string()),
                label: "chair".to_string(),
                source_bbox: [0.16, 0.63, 0.37, 1.0],
                contact_pixel: [0.347, 0.985],
                ground_point: [0.0, 0.0, 0.0],
                translation: [0.0, 0.0, 0.0],
                rotation_y_degrees: 0.0,
                scale: [1.0, 1.0, 1.0],
                local_aabb: SceneAssetAabb {
                    min: [-0.5, 0.0, -0.5],
                    max: [0.5, 1.0, 0.5],
                },
                target_footprint_m: [0.8, 0.8],
            }],
            camera: SceneCamera {
                translation: [0.0, 2.0, -3.0],
                focus: [0.0, 0.7, 0.0],
                yaw: Some(180.0),
                pitch: Some(25.0),
                radius: Some(4.0),
                vertical_fov_degrees: Some(70.0),
            },
            rug_center: [0.0, 0.0, 0.0],
            rug_scale: [1.0, 1.0, 1.0],
        };
        let status = json!({
            "projected_items": [{
                "cache_key": "chair",
                "screen_bbox": [0.60, 0.42, 0.82, 0.74],
                "screen_contact": [0.70, 0.66],
                "world_aabb": {
                    "min": [-0.5, 0.0, -0.5],
                    "max": [0.5, 1.0, 0.5]
                },
                "projected_corners": 8,
                "total_corners": 8
            }],
            "camera": {
                "translation": [-0.00000027, 1.9615524, -3.07795],
                "rotation": [0.0000000083, 0.9816272, 0.19080901, -0.0000000429],
                "yaw": std::f32::consts::PI,
                "pitch": 0.38397244,
                "radius": 3.3142834,
                "vertical_fov_degrees": 70.0
            }
        });

        let metrics = scene_feedback_metrics(
            &manifest,
            &layout,
            &status,
            Path::new("/tmp/no_screenshot.png"),
            FeedbackThresholdProfile::Standard.thresholds(),
            FeedbackThresholdProfile::Standard,
        )
        .unwrap();
        let object = &metrics["objects"][0];
        let translation_delta = json_array3(&object["translation_delta"]).unwrap();

        assert_eq!(object["grounding_basis"], json!("camera-ray-ground-plane"));
        assert!(translation_delta[0] > 0.05);
        assert!(translation_delta[2] < -0.5);
        assert_eq!(object["contact_residual_applied"], json!(true));
        assert!(object["target_ground_point"].is_array());
        assert!(object["observed_ground_point"].is_array());
    }

    #[test]
    fn feedback_metrics_use_bbox_center_anchor_for_tabletops() {
        let manifest = SceneObjectManifest {
            source_scene_path: "/tmp/source.jpg".to_string(),
            scene_calibration: None,
            objects: Vec::new(),
        };
        let layout = GroundedSceneLayout {
            bsn: "scene {}".to_string(),
            placements: vec![GroundedScenePlacement {
                entity_id: "table_1".to_string(),
                asset_id: "table".to_string(),
                object_id: "conference_table".to_string(),
                instance_id: None,
                label: "conference table".to_string(),
                source_bbox: [0.4, 0.5, 0.6, 1.0],
                contact_pixel: [0.5, 1.0],
                ground_point: [0.0, 0.0, 0.0],
                translation: [0.0, 0.0, 0.0],
                rotation_y_degrees: 0.0,
                scale: [1.0, 1.0, 1.0],
                local_aabb: SceneAssetAabb {
                    min: [-1.0, 0.0, -0.4],
                    max: [1.0, 0.2, 0.4],
                },
                target_footprint_m: [2.0, 0.8],
            }],
            camera: SceneCamera {
                translation: [0.0, 2.0, -3.0],
                focus: [0.0, 0.7, 0.0],
                yaw: Some(180.0),
                pitch: Some(25.0),
                radius: Some(4.0),
                vertical_fov_degrees: Some(70.0),
            },
            rug_center: [0.0, 0.0, 0.0],
            rug_scale: [1.0, 1.0, 1.0],
        };
        let status = json!({
            "projected_items": [{
                "cache_key": "table",
                "screen_bbox": [0.4, 0.3, 0.6, 0.7],
                "screen_contact": [0.5, 0.45],
                "world_aabb": {
                    "min": [-1.0, 0.0, -0.4],
                    "max": [1.0, 0.2, 0.4]
                },
                "projected_corners": 8,
                "total_corners": 8
            }],
            "camera": {
                "translation": [0.0, 2.0, -3.0],
                "rotation": [0.0, 0.9816272, 0.19080901, 0.0],
                "yaw": std::f32::consts::PI,
                "pitch": 0.38397244,
                "radius": 3.3142834,
                "vertical_fov_degrees": 70.0
            }
        });

        let metrics = scene_feedback_metrics(
            &manifest,
            &layout,
            &status,
            Path::new("/tmp/no_screenshot.png"),
            FeedbackThresholdProfile::Standard.thresholds(),
            FeedbackThresholdProfile::Standard,
        )
        .unwrap();
        let object = &metrics["objects"][0];
        let translation_delta = json_array3(&object["translation_delta"]).unwrap();

        assert_eq!(object["anchor_basis"], json!("bbox-center"));
        assert_eq!(object["expected_anchor"], json!([0.5, 0.75]));
        assert_eq!(object["observed_anchor"], json!([0.5, 0.5]));
        assert!(translation_delta[2].abs() <= 0.850001);
    }

    #[test]
    fn feedback_metrics_emit_bounded_corrections_for_projection_mismatch() {
        let manifest = SceneObjectManifest {
            source_scene_path: "/tmp/source.jpg".to_string(),
            scene_calibration: None,
            objects: Vec::new(),
        };
        let layout = GroundedSceneLayout {
            bsn: "scene {}".to_string(),
            placements: vec![GroundedScenePlacement {
                entity_id: "chair_1".to_string(),
                asset_id: "chair".to_string(),
                object_id: "chair".to_string(),
                instance_id: Some("chair_1".to_string()),
                label: "chair".to_string(),
                source_bbox: [0.4, 0.4, 0.6, 0.7],
                contact_pixel: [0.5, 0.7],
                ground_point: [0.0, 0.0, 0.0],
                translation: [0.0, 0.0, 0.0],
                rotation_y_degrees: 0.0,
                scale: [1.0, 1.0, 1.0],
                local_aabb: SceneAssetAabb {
                    min: [-0.5, 0.0, -0.5],
                    max: [0.5, 1.0, 0.5],
                },
                target_footprint_m: [0.8, 0.8],
            }],
            camera: SceneCamera {
                translation: [0.0, 2.0, 5.0],
                focus: [0.0, 0.0, 0.0],
                yaw: Some(180.0),
                pitch: Some(25.0),
                radius: Some(5.0),
                vertical_fov_degrees: Some(72.0),
            },
            rug_center: [0.0, 0.0, 0.0],
            rug_scale: [1.0, 1.0, 1.0],
        };
        let status = json!({
            "projected_items": [{
                "cache_key": "chair",
                "screen_bbox": [0.45, 0.5, 0.55, 0.6],
                "screen_contact": [0.5, 0.6],
                "projected_corners": 8,
                "total_corners": 8
            }],
            "camera": {
                "radius": 5.0
            }
        });

        let metrics = scene_feedback_metrics(
            &manifest,
            &layout,
            &status,
            Path::new("/tmp/iter.png"),
            FeedbackThresholdProfile::Standard.thresholds(),
            FeedbackThresholdProfile::Standard,
        )
        .unwrap();
        let deltas = feedback_layout_deltas(&metrics);

        assert!(!metrics["passed"].as_bool().unwrap());
        assert_eq!(metrics["object_count"], json!(1));
        assert_eq!(metrics["object_pass_count"], json!(0));
        let object_delta = &deltas["objects"][0];
        let translation_delta = object_delta["translation_delta"].as_array().unwrap();
        assert!(translation_delta[0].as_f64().unwrap().abs() <= 1.0e-6);
        assert!(translation_delta[1].as_f64().unwrap().abs() <= 1.0e-6);
        assert!((translation_delta[2].as_f64().unwrap() - 0.2).abs() <= 1.0e-5);
        let scale = object_delta["scale_multiplier"].as_f64().unwrap();
        assert!((scale - 1.22).abs() <= 1.0e-5);
        let yaw_delta = object_delta["yaw_delta_degrees"].as_f64().unwrap();
        assert!(yaw_delta.abs() <= 1.0e-6);
        assert_eq!(
            metrics["objects"][0]["yaw_basis"],
            json!("canonical-bsn-yaw-within-threshold")
        );
    }

    #[test]
    fn apply_mesh_decimation_preserves_pbr_baked_meshes() {
        let mesh = Mesh {
            vertices: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            faces: vec![[0, 1, 2], [1, 3, 2]],
            uvs: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            normals: Vec::new(),
            material: None,
            pbr_textures: Some(burn_synth::MeshPbrTextures {
                base_color: burn_synth::MeshTexture {
                    width: 1,
                    height: 1,
                    rgba8: vec![255, 255, 255, 255],
                },
                metallic_roughness: burn_synth::MeshTexture {
                    width: 1,
                    height: 1,
                    rgba8: vec![0, 255, 0, 255],
                },
                normal: None,
                emissive: None,
                occlusion: None,
            }),
        };

        let output = apply_mesh_decimation(mesh.clone(), Some(1)).expect("decimation");

        assert_eq!(output.faces.len(), mesh.faces.len());
        assert!(output.pbr_textures.is_some());
    }

    #[test]
    fn scene_compose_plan_generates_spawn_commands_with_validation_keys() {
        let plan = compose_scene_layout(SceneComposeArgs {
            reference_objects: vec![scene_layout::SceneReferenceObject {
                id: Some("chair_1".to_string()),
                label: "chair".to_string(),
                aliases: Vec::new(),
                bbox: [0.1, 0.2, 0.3, 0.6],
            }],
            assets: vec![scene_layout::SceneAssetBinding {
                reference_id: Some("chair_1".to_string()),
                label: Some("chair".to_string()),
                aliases: Vec::new(),
                path: Some(PathBuf::from("/tmp/chair.glb")),
                cache_key: None,
                local_aabb: None,
                select: true,
            }],
            apply: false,
            clear_existing: true,
            layout_width: 6.0,
            layout_depth: 4.0,
            y: 0.0,
            min_scale: 0.35,
            scale_multiplier: 1.0,
        })
        .expect("compose plan");
        let commands = scene_commands_from_plan(&plan).expect("scene commands");
        assert_eq!(commands.len(), 2);
        assert_eq!(commands[0]["type"], "clear_scene");
        assert_eq!(commands[1]["type"], "spawn_path");
        assert_eq!(commands[1]["cache_key"], "path:/tmp/chair.glb");
        assert_eq!(commands[1]["select"], true);
    }

    #[test]
    fn scene_sequence_is_strictly_monotonic() {
        let first = next_scene_sequence();
        let second = next_scene_sequence();
        assert!(second > first);
    }

    #[test]
    fn scene_command_waits_for_matching_status_sequence() {
        let root = unique_test_dir("scene_bridge");
        fs::create_dir_all(&root).expect("create temp dir");
        let command_path = root.join("scene_commands.json");
        let status_path = command_path.with_extension("status.json");
        let config = ServerConfig {
            scene_control_path: Some(command_path.clone()),
            scene_status_path: Some(status_path.clone()),
            scene_timeout: Duration::from_secs(1),
            ..ServerConfig::from_args(ServerArgs::parse_from(["burn_synth_mcp"]))
        };
        let server = McpServer::new(config);
        let status_path_for_thread = status_path.clone();
        let command_path_for_thread = command_path.clone();
        let handle = thread::spawn(move || {
            let started = Instant::now();
            loop {
                if command_path_for_thread.exists() {
                    let command = read_scene_status(&command_path_for_thread)
                        .expect("command JSON should parse");
                    let sequence = command["sequence"].as_u64().expect("sequence");
                    atomic_write_json(
                        &status_path_for_thread,
                        &json!({
                            "last_sequence": sequence,
                            "ok": true,
                            "cache_entries": [],
                            "world_items": [],
                            "camera": null,
                            "screenshots": [],
                        }),
                    )
                    .expect("write status");
                    return;
                }
                assert!(started.elapsed() < Duration::from_secs(1));
                thread::sleep(Duration::from_millis(10));
            }
        });

        let response = server
            .send_scene_commands(vec![json!({ "type": "clear_selection" })])
            .expect("scene command should be acknowledged");
        handle.join().expect("status writer thread");
        assert_eq!(response["acknowledged"], true);
        assert!(response["status"]["last_sequence"].as_u64().is_some());
        fs::remove_dir_all(root).expect("remove temp dir");
    }

    fn unique_test_dir(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "burn_synth_mcp_{label}_{}_{}",
            std::process::id(),
            nanos
        ))
    }

    fn find_repo_root_for_test() -> Option<PathBuf> {
        let mut dir = std::env::current_dir().ok()?;
        loop {
            if dir.join("Cargo.toml").exists() && dir.join("crates/burn_synth_mcp").exists() {
                return Some(dir);
            }
            if !dir.pop() {
                return None;
            }
        }
    }
}

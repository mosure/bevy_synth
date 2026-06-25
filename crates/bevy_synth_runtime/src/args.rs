use std::path::PathBuf;

use bevy::prelude::Resource;
use burn_synth::quality::{
    DEFAULT_CHUNK_SIZE as CORE_DEFAULT_CHUNK_SIZE, DEFAULT_SEED as CORE_DEFAULT_SEED,
    DEFAULT_TRELLIS_TARGET_FACES as CORE_DEFAULT_TRELLIS_TARGET_FACES,
    DEFAULT_TRIPOSG_TARGET_FACES as CORE_DEFAULT_TRIPOSG_TARGET_FACES, RuntimeQualityPreset,
};
use burn_triposplat::{
    MAX_NUM_GAUSSIANS, MIN_NUM_GAUSSIANS, TRIPOSPLAT_GAUSSIANS_PER_POINT,
    TripoSplatProfile as CoreTripoSplatProfile, TripoSplatProfileSettings, normalize_num_gaussians,
    triposplat_profile_for_settings,
};
use clap::{ArgAction, Parser, ValueEnum};

#[derive(Parser, Debug)]
#[command(about = "bevy_synth", version)]
pub struct Args {
    /// Path to an input image for mesh inference.
    #[arg(long)]
    pub image: Option<PathBuf>,

    /// Optional text prompt (scribble model). Requires --text-embeds.
    #[arg(long)]
    pub prompt: Option<String>,

    /// Path to a safetensors file containing text embeddings for the scribble model.
    #[arg(long)]
    pub text_embeds: Option<PathBuf>,

    /// Tensor key in the text embedding safetensors file.
    #[arg(long, default_value = "input.text_embeds")]
    pub text_embeds_key: String,

    /// Optional weights root for TripoSG (image-only) pipeline.
    #[arg(long)]
    pub weights_root: Option<PathBuf>,

    /// Optional weights root for Trellis2 pipeline.
    #[arg(long)]
    pub trellis_weights_root: Option<PathBuf>,

    /// Optional weights root for TripoSplat pipeline.
    #[arg(long)]
    pub triposplat_weights_root: Option<PathBuf>,

    /// Optional weights root for TRELLIS-image-large assets.
    #[arg(long)]
    pub trellis_image_large_root: Option<PathBuf>,

    /// Legacy option kept for backward CLI compatibility; ignored by Trellis2 Rust runtime.
    #[arg(long)]
    pub trellis_python_bin: Option<PathBuf>,

    /// Legacy option kept for backward CLI compatibility; ignored by Trellis2 Rust runtime.
    #[arg(long)]
    pub trellis_bridge_script: Option<PathBuf>,

    /// Trellis quality preset (low, medium, high).
    #[arg(long, value_enum, default_value_t = TrellisQuality::Low)]
    pub trellis_quality: TrellisQuality,

    /// Enable native Trellis PBR texture baking. Pass `--trellis-pbr true` for textured output.
    #[arg(long, default_value_t = false, action = ArgAction::Set)]
    pub trellis_pbr: bool,

    /// Native Trellis PBR texture size. Use 0 for the runtime default.
    #[arg(long, default_value_t = DEFAULT_TRELLIS_PBR_TEXTURE_SIZE)]
    pub trellis_pbr_texture_size: usize,

    /// Trellis target face count. Use 0 to disable Trellis decimation.
    #[arg(long)]
    pub trellis_faces: Option<usize>,

    /// Optional sparse-coordinate cap for Trellis decode. Use 0 to disable the explicit cap.
    #[arg(long)]
    pub trellis_max_sparse_coords: Option<usize>,

    /// Optional weights root for TripoSG-scribble pipeline.
    #[arg(long)]
    pub scribble_weights_root: Option<PathBuf>,

    /// Quality preset (fast, balanced, full). Individual flags override this preset.
    #[arg(long, value_enum, default_value_t = QualityPreset::Balanced)]
    pub quality: QualityPreset,

    /// TripoSplat profile (low, balanced, high). Individual TripoSplat flags override this preset.
    #[arg(long, value_enum, default_value_t = TripoSplatProfile::Balanced)]
    pub triposplat_profile: TripoSplatProfile,

    /// Number of diffusion steps (overrides --quality).
    #[arg(long)]
    pub num_steps: Option<usize>,

    /// Number of latents tokens (overrides --quality).
    #[arg(long)]
    pub num_tokens: Option<usize>,

    /// Guidance scale (overrides --quality).
    #[arg(long)]
    pub guidance_scale: Option<f32>,

    /// Flow timestep schedule shift for TripoSplat.
    #[arg(long)]
    pub triposplat_shift: Option<f32>,

    /// Target Gaussian count for TripoSplat.
    #[arg(long, value_parser = parse_triposplat_gaussians)]
    pub gaussians: Option<usize>,

    /// Alpha matte erosion radius for TripoSplat preprocessing.
    #[arg(long)]
    pub triposplat_erode_radius: Option<usize>,

    /// Optional RNG seed for deterministic sampling.
    #[arg(long)]
    pub seed: Option<u64>,

    /// Grid resolution used for mesh extraction (overrides --quality).
    #[arg(long)]
    pub resolution: Option<usize>,

    /// Chunk size for VAE grid decoding (overrides --quality).
    #[arg(long)]
    pub chunk_size: Option<usize>,

    /// Bounds for grid decoding (6 floats: minX minY minZ maxX maxY maxZ).
    #[arg(
        long,
        num_args = 6,
        value_delimiter = ' ',
        default_value = "-1.005 -1.005 -1.005 1.005 1.005 1.005"
    )]
    pub bounds: Vec<f32>,

    /// Mesh extraction mode.
    #[arg(long, value_enum, default_value_t = MeshMode::Flash)]
    pub mesh_mode: MeshMode,

    /// Dense octree depth used for hierarchical extraction (overrides --quality).
    #[arg(long)]
    pub dense_octree_depth: Option<usize>,

    /// Hierarchical octree depth used for hierarchical extraction (overrides --quality).
    #[arg(long)]
    pub hierarchical_octree_depth: Option<usize>,

    /// Band threshold used to expand near-surface regions in hierarchical extraction
    /// (overrides --quality).
    #[arg(long)]
    pub band_threshold: Option<f32>,

    /// Flash octree depth used for flash extraction (overrides --quality).
    #[arg(long)]
    pub flash_octree_depth: Option<usize>,

    /// Flash minimum resolution (overrides --quality).
    #[arg(long)]
    pub flash_min_resolution: Option<usize>,

    /// Flash mini grid count per axis (overrides --quality).
    #[arg(long)]
    pub flash_mini_grid_num: Option<usize>,

    /// Flash chunk size for VAE queries (overrides --quality).
    #[arg(long)]
    pub flash_num_chunks: Option<usize>,

    /// Flash iso offset (overrides --quality).
    #[arg(long)]
    pub flash_mc_level: Option<f32>,

    /// Target face count for mesh decimation (post-process). Use 0 to disable.
    /// Defaults to 10,000 for TripoSG.
    #[arg(long)]
    pub faces: Option<usize>,

    /// Path to write a GLB file for the inferred mesh.
    #[arg(long)]
    pub output: Option<PathBuf>,

    /// Path to an existing mesh file to display (glb/obj/gltf).
    #[arg(long)]
    pub mesh: Option<PathBuf>,

    /// Optional weights root for RMBG background removal.
    #[arg(long)]
    pub bg_weights_root: Option<PathBuf>,

    /// Synthesis backend models to enable (comma-delimited, ordered by preference).
    #[arg(long, value_enum, value_delimiter = ',')]
    pub synthesis_models: Vec<SynthesisModel>,

    /// Foreground model variant.
    #[arg(long, value_enum, default_value_t = RmbgModel::Rmbg14)]
    pub rmbg_model: RmbgModel,

    /// Background removal backend (auto, cpu, gpu).
    #[arg(long, value_enum, default_value_t = RmbgBackend::Auto)]
    pub rmbg_backend: RmbgBackend,

    /// DINO image encoder backend (auto, cpu, gpu).
    #[arg(long, value_enum, default_value_t = DinoBackend::Auto)]
    pub dino_backend: DinoBackend,

    /// TripoSG/TripoSplat burnpack precision preference on web/native (auto, f16, f32).
    /// Defaults to `f16` to reduce weight download/storage footprint.
    #[arg(long, value_enum, default_value_t = WeightPrecision::F16)]
    pub weights_precision: WeightPrecision,

    /// RMBG burnpack precision preference (auto, f16, f32).
    /// On wasm, `auto` favors f16 to reduce startup memory pressure.
    #[arg(long, value_enum, default_value_t = WeightPrecision::Auto)]
    pub rmbg_weights_precision: WeightPrecision,

    /// Inference backend (cpu, wgpu, cuda).
    #[arg(long, value_enum, default_value_t = BackendKind::Wgpu)]
    pub backend: BackendKind,

    /// Pause Bevy window updates/renders while inference is running.
    /// This is a temporary workaround for upstream Linux swapchain instability.
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    pub pause_render_during_inference: bool,

    /// Show Bevy UI overlays. Press F1 in the app to toggle this at runtime.
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    pub ui_visible: bool,

    /// Viewer read-only mode: allow camera navigation but disable scene edits and cache persistence.
    #[arg(long, action = ArgAction::SetTrue)]
    pub read_only: bool,

    /// Maximum number of queued images to batch per inference dispatch.
    #[arg(long, default_value_t = 1)]
    pub max_batch_size: usize,

    /// Optional JSON command file path for external MCP/agent scene control.
    #[arg(long)]
    pub mcp_scene_control_path: Option<PathBuf>,

    /// Restricted synth_scene_v1 BSN scene file to load into the viewer at startup.
    #[arg(long)]
    pub scene_bsn: Option<PathBuf>,

    /// Asset binding JSON for --scene-bsn.
    #[arg(long)]
    pub scene_assets_json: Option<PathBuf>,

    /// Clear the current scene before applying --scene-bsn.
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    pub scene_bsn_clear_existing: bool,
}

#[derive(ValueEnum, Clone, Copy, Debug)]
pub enum RmbgBackend {
    Auto,
    Cpu,
    Gpu,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum RmbgModel {
    Rmbg14,
    Rmbg2,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum SynthesisModel {
    Triposg,
    Trellis,
    Triposplat,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrellisQuality {
    Low,
    Medium,
    High,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum DinoBackend {
    Auto,
    Cpu,
    Gpu,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightPrecision {
    Auto,
    F16,
    F32,
}

#[derive(ValueEnum, Clone, Debug)]
pub enum BackendKind {
    Cpu,
    Wgpu,
    Cuda,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum QualityPreset {
    Fast,
    Balanced,
    Full,
}

#[derive(ValueEnum, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TripoSplatProfile {
    Low,
    #[default]
    Balanced,
    High,
    Custom,
}

impl TripoSplatProfile {
    pub fn settings(self) -> TripoSplatProfileSettings {
        CoreTripoSplatProfile::from(self).settings()
    }
}

impl From<TripoSplatProfile> for CoreTripoSplatProfile {
    fn from(value: TripoSplatProfile) -> Self {
        match value {
            TripoSplatProfile::Low => Self::Low,
            TripoSplatProfile::Balanced => Self::Balanced,
            TripoSplatProfile::High => Self::High,
            TripoSplatProfile::Custom => Self::Custom,
        }
    }
}

impl From<CoreTripoSplatProfile> for TripoSplatProfile {
    fn from(value: CoreTripoSplatProfile) -> Self {
        match value {
            CoreTripoSplatProfile::Low => Self::Low,
            CoreTripoSplatProfile::Balanced => Self::Balanced,
            CoreTripoSplatProfile::High => Self::High,
            CoreTripoSplatProfile::Custom => Self::Custom,
        }
    }
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeshMode {
    Dense,
    Hierarchical,
    Flash,
}

pub const DEFAULT_CHUNK_SIZE: usize = CORE_DEFAULT_CHUNK_SIZE;
pub const DEFAULT_SEED: u64 = CORE_DEFAULT_SEED;
pub const DEFAULT_TRIPOSG_TARGET_FACES: usize = CORE_DEFAULT_TRIPOSG_TARGET_FACES;
pub const DEFAULT_TRELLIS_TARGET_FACES: usize = CORE_DEFAULT_TRELLIS_TARGET_FACES;
pub const DEFAULT_TRELLIS_PBR_TEXTURE_SIZE: usize =
    burn_synth::runtime::DEFAULT_TRELLIS_PBR_TEXTURE_SIZE;
pub const TRIPOSPLAT_MIN_NUM_GAUSSIANS: usize = MIN_NUM_GAUSSIANS;
pub const TRIPOSPLAT_MAX_NUM_GAUSSIANS: usize = MAX_NUM_GAUSSIANS;
pub const TRIPOSPLAT_GAUSSIAN_STEP: usize = TRIPOSPLAT_GAUSSIANS_PER_POINT * 1024;

impl QualityPreset {
    pub fn defaults(self) -> burn_synth::quality::RuntimeQualityDefaults {
        RuntimeQualityPreset::from(self).defaults()
    }
}

impl From<QualityPreset> for RuntimeQualityPreset {
    fn from(value: QualityPreset) -> Self {
        match value {
            QualityPreset::Fast => Self::Fast,
            QualityPreset::Balanced => Self::Balanced,
            QualityPreset::Full => Self::Full,
        }
    }
}

#[derive(Resource, Debug, Clone)]
pub struct AppArgs {
    pub image: Option<PathBuf>,
    pub prompt: Option<String>,
    pub text_embeds: Option<PathBuf>,
    pub text_embeds_key: String,
    pub weights_root: Option<PathBuf>,
    pub trellis_weights_root: Option<PathBuf>,
    pub triposplat_weights_root: Option<PathBuf>,
    pub trellis_image_large_root: Option<PathBuf>,
    pub trellis_python_bin: Option<PathBuf>,
    pub trellis_bridge_script: Option<PathBuf>,
    pub trellis_quality: TrellisQuality,
    pub trellis_pbr_enabled: bool,
    pub trellis_pbr_texture_size: Option<usize>,
    pub trellis_target_faces: Option<usize>,
    pub trellis_max_sparse_coords: Option<usize>,
    pub scribble_weights_root: Option<PathBuf>,
    pub quality: QualityPreset,
    pub triposplat_profile: TripoSplatProfile,
    pub num_steps: usize,
    pub num_tokens: usize,
    pub guidance_scale: f32,
    pub triposplat_shift: f32,
    pub triposplat_num_gaussians: usize,
    pub triposplat_erode_radius: usize,
    pub seed: Option<u64>,
    pub resolution: usize,
    pub chunk_size: usize,
    pub bounds: Vec<f32>,
    pub mesh_mode: MeshMode,
    pub dense_octree_depth: usize,
    pub hierarchical_octree_depth: usize,
    pub band_threshold: f32,
    pub flash_octree_depth: usize,
    pub flash_min_resolution: usize,
    pub flash_mini_grid_num: usize,
    pub flash_num_chunks: usize,
    pub flash_mc_level: f32,
    pub target_faces: Option<usize>,
    pub output: Option<PathBuf>,
    pub mesh: Option<PathBuf>,
    pub bg_weights_root: Option<PathBuf>,
    pub synthesis_models: Vec<SynthesisModel>,
    pub available_synthesis_models: Vec<SynthesisModel>,
    pub rmbg_model: RmbgModel,
    pub backend: BackendKind,
    pub rmbg_backend: RmbgBackend,
    pub dino_backend: DinoBackend,
    pub weights_precision: WeightPrecision,
    pub rmbg_weights_precision: WeightPrecision,
    pub pause_render_during_inference: bool,
    pub ui_visible: bool,
    pub read_only: bool,
    pub max_batch_size: usize,
    pub mcp_scene_control_path: Option<PathBuf>,
    pub scene_bsn: Option<PathBuf>,
    pub scene_assets_json: Option<PathBuf>,
    pub scene_bsn_clear_existing: bool,
}

pub fn build_app_args(args: Args) -> AppArgs {
    let quality = args.quality;
    let defaults = quality.defaults();
    let explicit_synthesis_models = !args.synthesis_models.is_empty();
    let synthesis_models = sanitize_synthesis_models(args.synthesis_models);
    let available_synthesis_models = if explicit_synthesis_models {
        synthesis_models.clone()
    } else {
        default_available_synthesis_models()
    };
    let triposplat_selected = synthesis_models
        .first()
        .is_some_and(|model| matches!(model, SynthesisModel::Triposplat));
    let triposplat_profile = args.triposplat_profile;
    let triposplat_defaults = triposplat_profile.settings();
    let seed = args.seed.or(Some(DEFAULT_SEED));
    let target_faces = match args.faces {
        Some(0) => None,
        Some(value) => Some(value),
        None => Some(DEFAULT_TRIPOSG_TARGET_FACES),
    };
    let trellis_target_faces = match args.trellis_faces.or(args.faces) {
        Some(0) => None,
        Some(value) => Some(value),
        None => Some(DEFAULT_TRELLIS_TARGET_FACES),
    };
    let trellis_max_sparse_coords = match args.trellis_max_sparse_coords {
        Some(0) | None => None,
        Some(value) => Some(value),
    };
    AppArgs {
        image: args.image,
        prompt: args.prompt,
        text_embeds: args.text_embeds,
        text_embeds_key: args.text_embeds_key,
        weights_root: args.weights_root,
        trellis_weights_root: args.trellis_weights_root,
        triposplat_weights_root: args.triposplat_weights_root,
        trellis_image_large_root: args.trellis_image_large_root,
        trellis_python_bin: args.trellis_python_bin,
        trellis_bridge_script: args.trellis_bridge_script,
        trellis_quality: args.trellis_quality,
        trellis_pbr_enabled: args.trellis_pbr,
        trellis_pbr_texture_size: (args.trellis_pbr_texture_size > 0)
            .then_some(args.trellis_pbr_texture_size),
        trellis_target_faces,
        trellis_max_sparse_coords,
        scribble_weights_root: args.scribble_weights_root,
        quality,
        triposplat_profile,
        num_steps: args.num_steps.unwrap_or(if triposplat_selected {
            triposplat_defaults.steps
        } else {
            defaults.num_steps
        }),
        num_tokens: args.num_tokens.unwrap_or(defaults.num_tokens),
        guidance_scale: args.guidance_scale.unwrap_or(if triposplat_selected {
            triposplat_defaults.guidance_scale
        } else {
            defaults.guidance_scale
        }),
        triposplat_shift: args.triposplat_shift.unwrap_or(3.0),
        triposplat_num_gaussians: args.gaussians.unwrap_or(triposplat_defaults.num_gaussians),
        triposplat_erode_radius: args.triposplat_erode_radius.unwrap_or(1),
        seed,
        resolution: args.resolution.unwrap_or(defaults.resolution),
        chunk_size: args.chunk_size.unwrap_or(defaults.chunk_size),
        bounds: args.bounds,
        mesh_mode: args.mesh_mode,
        dense_octree_depth: args
            .dense_octree_depth
            .unwrap_or(defaults.dense_octree_depth),
        hierarchical_octree_depth: args
            .hierarchical_octree_depth
            .unwrap_or(defaults.hierarchical_octree_depth),
        band_threshold: args.band_threshold.unwrap_or(defaults.band_threshold),
        flash_octree_depth: args
            .flash_octree_depth
            .unwrap_or(defaults.flash_octree_depth),
        flash_min_resolution: args
            .flash_min_resolution
            .unwrap_or(defaults.flash_min_resolution),
        flash_mini_grid_num: args
            .flash_mini_grid_num
            .unwrap_or(defaults.flash_mini_grid_num),
        flash_num_chunks: args.flash_num_chunks.unwrap_or(defaults.flash_num_chunks),
        flash_mc_level: args.flash_mc_level.unwrap_or(defaults.flash_mc_level),
        target_faces,
        output: args.output,
        mesh: args.mesh,
        bg_weights_root: args.bg_weights_root,
        synthesis_models,
        available_synthesis_models,
        rmbg_model: args.rmbg_model,
        backend: args.backend,
        rmbg_backend: args.rmbg_backend,
        dino_backend: args.dino_backend,
        weights_precision: args.weights_precision,
        rmbg_weights_precision: args.rmbg_weights_precision,
        pause_render_during_inference: args.pause_render_during_inference,
        ui_visible: args.ui_visible,
        read_only: args.read_only,
        max_batch_size: args.max_batch_size.max(1),
        mcp_scene_control_path: args.mcp_scene_control_path,
        scene_bsn: args.scene_bsn,
        scene_assets_json: args.scene_assets_json,
        scene_bsn_clear_existing: args.scene_bsn_clear_existing,
    }
}

impl Default for AppArgs {
    fn default() -> Self {
        build_app_args(Args::parse_from(["bevy_synth"]))
    }
}

impl AppArgs {
    pub fn apply_triposplat_profile(&mut self, profile: TripoSplatProfile) {
        self.triposplat_profile = profile;
        if profile == TripoSplatProfile::Custom {
            return;
        }
        let settings = profile.settings();
        self.num_steps = settings.steps;
        self.guidance_scale = settings.guidance_scale;
        self.triposplat_num_gaussians = settings.num_gaussians;
    }

    pub fn refresh_triposplat_profile_from_current_settings(&mut self) {
        self.triposplat_profile = triposplat_profile_for_settings(
            self.num_steps,
            self.guidance_scale,
            self.triposplat_num_gaussians,
        )
        .into();
    }
}

fn parse_triposplat_gaussians(value: &str) -> Result<usize, String> {
    let raw = value
        .parse::<usize>()
        .map_err(|err| format!("invalid TripoSplat gaussian count `{value}`: {err}"))?;
    normalize_num_gaussians(raw)
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

fn default_available_synthesis_models() -> Vec<SynthesisModel> {
    vec![
        SynthesisModel::Triposg,
        SynthesisModel::Trellis,
        SynthesisModel::Triposplat,
    ]
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::{
        Args, DEFAULT_CHUNK_SIZE, DEFAULT_SEED, DEFAULT_TRELLIS_PBR_TEXTURE_SIZE,
        DEFAULT_TRELLIS_TARGET_FACES, DEFAULT_TRIPOSG_TARGET_FACES, QualityPreset, RmbgModel,
        SynthesisModel, TrellisQuality, TripoSplatProfile, WeightPrecision, build_app_args,
    };

    #[test]
    fn defaults_use_rmbg14_and_batch_one() {
        let args = Args::parse_from(["bevy_synth"]);
        let app_args = build_app_args(args);
        assert!(matches!(app_args.rmbg_model, RmbgModel::Rmbg14));
        assert_eq!(app_args.synthesis_models, vec![SynthesisModel::Triposg]);
        assert_eq!(
            app_args.available_synthesis_models,
            vec![
                SynthesisModel::Triposg,
                SynthesisModel::Trellis,
                SynthesisModel::Triposplat
            ]
        );
        assert_eq!(app_args.max_batch_size, 1);
    }

    #[test]
    fn cli_can_select_rmbg14_and_custom_batch_size() {
        let args = Args::parse_from([
            "bevy_synth",
            "--synthesis-models",
            "triposg,trellis",
            "--rmbg-model",
            "rmbg14",
            "--max-batch-size",
            "4",
        ]);
        let app_args = build_app_args(args);
        assert!(matches!(app_args.rmbg_model, RmbgModel::Rmbg14));
        assert_eq!(
            app_args.synthesis_models,
            vec![SynthesisModel::Triposg, SynthesisModel::Trellis]
        );
        assert_eq!(
            app_args.available_synthesis_models,
            vec![SynthesisModel::Triposg, SynthesisModel::Trellis]
        );
        assert_eq!(app_args.max_batch_size, 4);
    }

    #[test]
    fn max_batch_size_is_clamped_to_one() {
        let args = Args::parse_from(["bevy_synth", "--max-batch-size", "0"]);
        let app_args = build_app_args(args);
        assert_eq!(app_args.max_batch_size, 1);
    }

    #[test]
    fn bsn_scene_viewer_args_are_preserved() {
        let args = Args::parse_from([
            "bevy_synth",
            "--scene-bsn",
            "tmp/runs/demo/scene.bsn",
            "--scene-assets-json",
            "tmp/runs/demo/assets.json",
            "--scene-bsn-clear-existing",
            "false",
            "--ui-visible",
            "false",
            "--read-only",
        ]);
        let app_args = build_app_args(args);
        assert_eq!(
            app_args.scene_bsn.as_deref(),
            Some(std::path::Path::new("tmp/runs/demo/scene.bsn"))
        );
        assert_eq!(
            app_args.scene_assets_json.as_deref(),
            Some(std::path::Path::new("tmp/runs/demo/assets.json"))
        );
        assert!(!app_args.scene_bsn_clear_existing);
        assert!(!app_args.ui_visible);
        assert!(app_args.read_only);
    }

    #[test]
    fn synthesis_models_are_deduplicated() {
        let args = Args::parse_from([
            "bevy_synth",
            "--synthesis-models",
            "triposg,triposg,trellis,trellis",
        ]);
        let app_args = build_app_args(args);
        assert_eq!(
            app_args.synthesis_models,
            vec![SynthesisModel::Triposg, SynthesisModel::Trellis]
        );
    }

    #[test]
    fn fast_and_balanced_presets_use_backend_tuned_chunk_default() {
        let fast = build_app_args(Args::parse_from(["bevy_synth", "--quality", "fast"]));
        let balanced = build_app_args(Args::parse_from(["bevy_synth", "--quality", "balanced"]));
        assert_eq!(fast.chunk_size, DEFAULT_CHUNK_SIZE);
        assert_eq!(balanced.chunk_size, DEFAULT_CHUNK_SIZE);
    }

    #[test]
    fn quality_defaults_to_balanced() {
        let defaults = build_app_args(Args::parse_from(["bevy_synth"]));
        assert_eq!(defaults.quality, QualityPreset::Balanced);
        assert_eq!(defaults.num_steps, 20);
        assert_eq!(defaults.num_tokens, 1024);
        assert_eq!(defaults.flash_octree_depth, 8);
        assert_eq!(defaults.flash_min_resolution, 31);
    }

    #[test]
    fn trellis_settings_have_pipeline_specific_defaults_and_overrides() {
        let defaults = build_app_args(Args::parse_from(["bevy_synth"]));
        assert_eq!(defaults.trellis_quality, TrellisQuality::Low);
        assert!(!defaults.trellis_pbr_enabled);
        assert_eq!(
            defaults.trellis_pbr_texture_size,
            Some(DEFAULT_TRELLIS_PBR_TEXTURE_SIZE)
        );
        assert_eq!(defaults.target_faces, Some(DEFAULT_TRIPOSG_TARGET_FACES));
        assert_eq!(
            defaults.trellis_target_faces,
            Some(DEFAULT_TRELLIS_TARGET_FACES)
        );
        assert_eq!(defaults.trellis_max_sparse_coords, None);

        let custom = build_app_args(Args::parse_from([
            "bevy_synth",
            "--synthesis-models",
            "trellis",
            "--trellis-quality",
            "high",
            "--trellis-pbr",
            "true",
            "--trellis-pbr-texture-size",
            "2048",
            "--trellis-faces",
            "500000",
            "--trellis-max-sparse-coords",
            "4096",
        ]));
        assert_eq!(custom.trellis_quality, TrellisQuality::High);
        assert!(custom.trellis_pbr_enabled);
        assert_eq!(custom.trellis_pbr_texture_size, Some(2048));
        assert_eq!(custom.trellis_target_faces, Some(500_000));
        assert_eq!(custom.trellis_max_sparse_coords, Some(4096));

        let face_budget_only = build_app_args(Args::parse_from([
            "bevy_synth",
            "--synthesis-models",
            "trellis",
            "--trellis-faces",
            "250000",
        ]));
        assert_eq!(face_budget_only.trellis_target_faces, Some(250_000));
        assert_eq!(
            face_budget_only.trellis_max_sparse_coords, None,
            "Trellis face budget must not be reused as a sparse-coordinate cap"
        );

        let disabled_faces = build_app_args(Args::parse_from([
            "bevy_synth",
            "--synthesis-models",
            "trellis",
            "--trellis-faces",
            "0",
            "--trellis-max-sparse-coords",
            "0",
            "--trellis-pbr-texture-size",
            "0",
        ]));
        assert_eq!(disabled_faces.trellis_target_faces, None);
        assert_eq!(disabled_faces.trellis_max_sparse_coords, None);
        assert_eq!(disabled_faces.trellis_pbr_texture_size, None);
    }

    #[test]
    fn triposplat_profile_controls_default_steps_guidance_and_gaussians() {
        let low = build_app_args(Args::parse_from([
            "bevy_synth",
            "--synthesis-models",
            "triposplat",
            "--triposplat-profile",
            "low",
        ]));
        assert_eq!(low.triposplat_profile, TripoSplatProfile::Low);
        assert_eq!(low.num_steps, 5);
        assert_eq!(low.guidance_scale, 3.0);
        assert_eq!(low.triposplat_num_gaussians, 32_768);

        let high = build_app_args(Args::parse_from([
            "bevy_synth",
            "--synthesis-models",
            "triposplat",
            "--triposplat-profile",
            "high",
        ]));
        assert_eq!(high.num_steps, 50);
        assert_eq!(high.guidance_scale, 3.0);
        assert_eq!(high.triposplat_num_gaussians, 262_144);
    }

    #[test]
    fn explicit_triposplat_flags_override_profile() {
        let args = build_app_args(Args::parse_from([
            "bevy_synth",
            "--synthesis-models",
            "triposplat",
            "--triposplat-profile",
            "low",
            "--num-steps",
            "18",
            "--guidance-scale",
            "4.5",
            "--gaussians",
            "65537",
        ]));
        assert_eq!(args.triposplat_profile, TripoSplatProfile::Low);
        assert_eq!(args.num_steps, 18);
        assert_eq!(args.guidance_scale, 4.5);
        assert_eq!(args.triposplat_num_gaussians, 65_536);
    }

    #[test]
    fn pause_render_during_inference_defaults_on_and_can_be_disabled() {
        let default_args = build_app_args(Args::parse_from(["bevy_synth"]));
        assert!(default_args.pause_render_during_inference);
        assert!(default_args.ui_visible);
        assert!(!default_args.read_only);

        let disabled = build_app_args(Args::parse_from([
            "bevy_synth",
            "--pause-render-during-inference",
            "false",
        ]));
        assert!(!disabled.pause_render_during_inference);
    }

    #[test]
    fn weight_precision_defaults_to_f16_with_rmbg_auto() {
        let app_args = build_app_args(Args::parse_from(["bevy_synth"]));
        assert_eq!(app_args.weights_precision, WeightPrecision::F16);
        assert_eq!(app_args.rmbg_weights_precision, WeightPrecision::Auto);
    }

    #[test]
    fn seed_defaults_to_canonical_value_unless_overridden() {
        let default_args = build_app_args(Args::parse_from(["bevy_synth"]));
        assert_eq!(default_args.seed, Some(DEFAULT_SEED));

        let custom = build_app_args(Args::parse_from(["bevy_synth", "--seed", "7"]));
        assert_eq!(custom.seed, Some(7));
    }
}

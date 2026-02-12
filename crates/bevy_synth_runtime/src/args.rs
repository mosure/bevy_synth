use std::path::PathBuf;

use bevy::prelude::Resource;
use clap::{Parser, ValueEnum};

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
    #[arg(long, value_enum, default_value_t = TrellisQuality::Medium)]
    pub trellis_quality: TrellisQuality,

    /// Optional weights root for TripoSG-scribble pipeline.
    #[arg(long)]
    pub scribble_weights_root: Option<PathBuf>,

    /// Quality preset (fast, balanced, full). Individual flags override this preset.
    #[arg(long, value_enum, default_value_t = QualityPreset::Full)]
    pub quality: QualityPreset,

    /// Match Python TripoSG defaults for parity (full quality + fixed seed + decimation target).
    /// Enabled by default; pass --no-match-python to disable.
    #[arg(long, default_value_t = true)]
    pub match_python: bool,

    /// Disable Python parity defaults (use CLI-provided settings).
    #[arg(long, default_value_t = false)]
    pub no_match_python: bool,

    /// Number of diffusion steps (overrides --quality).
    #[arg(long)]
    pub num_steps: Option<usize>,

    /// Number of latents tokens (overrides --quality).
    #[arg(long)]
    pub num_tokens: Option<usize>,

    /// Guidance scale (overrides --quality).
    #[arg(long)]
    pub guidance_scale: Option<f32>,

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
    /// Defaults to 10,000 when --match-python is enabled.
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
    #[arg(
        long,
        value_enum,
        value_delimiter = ',',
        default_values_t = [SynthesisModel::Triposg]
    )]
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

    /// Inference backend (cpu, wgpu, cuda).
    #[arg(long, value_enum, default_value_t = BackendKind::Wgpu)]
    pub backend: BackendKind,

    /// Maximum number of queued images to batch per inference dispatch.
    #[arg(long, default_value_t = 1)]
    pub max_batch_size: usize,

    /// Optional JSON command file path for external MCP/agent scene control.
    #[arg(long)]
    pub mcp_scene_control_path: Option<PathBuf>,
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

#[derive(ValueEnum, Clone, Debug)]
pub enum BackendKind {
    Cpu,
    Wgpu,
    Cuda,
}

#[derive(ValueEnum, Clone, Debug)]
pub enum QualityPreset {
    Fast,
    Balanced,
    Full,
}

#[derive(ValueEnum, Clone, Debug)]
pub enum MeshMode {
    Dense,
    Hierarchical,
    Flash,
}

pub const DEFAULT_CHUNK_SIZE: usize = 10_000;

#[derive(Clone, Copy, Debug)]
pub struct QualityDefaults {
    pub num_steps: usize,
    pub num_tokens: usize,
    pub guidance_scale: f32,
    pub resolution: usize,
    pub chunk_size: usize,
    pub dense_octree_depth: usize,
    pub hierarchical_octree_depth: usize,
    pub band_threshold: f32,
    pub flash_octree_depth: usize,
    pub flash_min_resolution: usize,
    pub flash_mini_grid_num: usize,
    pub flash_num_chunks: usize,
    pub flash_mc_level: f32,
}

impl QualityPreset {
    pub fn defaults(self) -> QualityDefaults {
        match self {
            QualityPreset::Fast => QualityDefaults {
                num_steps: 12,
                num_tokens: 512,
                guidance_scale: 7.0,
                resolution: 128,
                chunk_size: 8192,
                dense_octree_depth: 6,
                hierarchical_octree_depth: 7,
                band_threshold: 1.0,
                flash_octree_depth: 7,
                flash_min_resolution: 31,
                flash_mini_grid_num: 2,
                flash_num_chunks: 4096,
                flash_mc_level: 0.0,
            },
            QualityPreset::Balanced => QualityDefaults {
                num_steps: 20,
                num_tokens: 1024,
                guidance_scale: 7.0,
                resolution: 192,
                chunk_size: 8192,
                dense_octree_depth: 7,
                hierarchical_octree_depth: 8,
                band_threshold: 1.0,
                flash_octree_depth: 8,
                flash_min_resolution: 31,
                flash_mini_grid_num: 4,
                flash_num_chunks: 8192,
                flash_mc_level: 0.0,
            },
            QualityPreset::Full => QualityDefaults {
                num_steps: 50,
                num_tokens: 2048,
                guidance_scale: 7.0,
                resolution: 256,
                chunk_size: DEFAULT_CHUNK_SIZE,
                dense_octree_depth: 8,
                hierarchical_octree_depth: 9,
                band_threshold: 1.0,
                flash_octree_depth: 9,
                flash_min_resolution: 63,
                flash_mini_grid_num: 4,
                flash_num_chunks: DEFAULT_CHUNK_SIZE,
                flash_mc_level: 0.0,
            },
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
    pub trellis_image_large_root: Option<PathBuf>,
    pub trellis_python_bin: Option<PathBuf>,
    pub trellis_bridge_script: Option<PathBuf>,
    pub trellis_quality: TrellisQuality,
    pub scribble_weights_root: Option<PathBuf>,
    pub num_steps: usize,
    pub num_tokens: usize,
    pub guidance_scale: f32,
    pub seed: Option<u64>,
    pub match_python: bool,
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
    pub rmbg_model: RmbgModel,
    pub backend: BackendKind,
    pub rmbg_backend: RmbgBackend,
    pub dino_backend: DinoBackend,
    pub max_batch_size: usize,
    pub mcp_scene_control_path: Option<PathBuf>,
}

pub fn build_app_args(args: Args) -> AppArgs {
    let match_python = if args.no_match_python {
        false
    } else {
        args.match_python
    };
    let defaults = if match_python {
        QualityPreset::Full.defaults()
    } else {
        args.quality.defaults()
    };
    let seed = args.seed.or(if match_python { Some(42) } else { None });
    let target_faces = match args.faces {
        Some(0) => None,
        Some(value) => Some(value),
        None => {
            if match_python {
                Some(10_000)
            } else {
                None
            }
        }
    };
    AppArgs {
        image: args.image,
        prompt: args.prompt,
        text_embeds: args.text_embeds,
        text_embeds_key: args.text_embeds_key,
        weights_root: args.weights_root,
        trellis_weights_root: args.trellis_weights_root,
        trellis_image_large_root: args.trellis_image_large_root,
        trellis_python_bin: args.trellis_python_bin,
        trellis_bridge_script: args.trellis_bridge_script,
        trellis_quality: args.trellis_quality,
        scribble_weights_root: args.scribble_weights_root,
        num_steps: args.num_steps.unwrap_or(defaults.num_steps),
        num_tokens: args.num_tokens.unwrap_or(defaults.num_tokens),
        guidance_scale: args.guidance_scale.unwrap_or(defaults.guidance_scale),
        seed,
        match_python,
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
        synthesis_models: sanitize_synthesis_models(args.synthesis_models),
        rmbg_model: args.rmbg_model,
        backend: args.backend,
        rmbg_backend: args.rmbg_backend,
        dino_backend: args.dino_backend,
        max_batch_size: args.max_batch_size.max(1),
        mcp_scene_control_path: args.mcp_scene_control_path,
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

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::{Args, RmbgModel, SynthesisModel, build_app_args};

    #[test]
    fn defaults_use_rmbg14_and_batch_one() {
        let args = Args::parse_from(["bevy_synth"]);
        let app_args = build_app_args(args);
        assert!(matches!(app_args.rmbg_model, RmbgModel::Rmbg14));
        assert_eq!(app_args.synthesis_models, vec![SynthesisModel::Triposg]);
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
        assert_eq!(app_args.max_batch_size, 4);
    }

    #[test]
    fn max_batch_size_is_clamped_to_one() {
        let args = Args::parse_from(["bevy_synth", "--max-batch-size", "0"]);
        let app_args = build_app_args(args);
        assert_eq!(app_args.max_batch_size, 1);
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
}

use std::path::PathBuf;

use bevy::prelude::Resource;
use clap::{Parser, ValueEnum};

#[derive(Parser, Debug)]
#[command(about = "TripoSG Bevy viewer", version)]
pub(crate) struct Args {
    /// Path to an input image for TripoSG inference.
    #[arg(long)]
    pub(crate) image: Option<PathBuf>,

    /// Optional text prompt (scribble model). Requires --text-embeds.
    #[arg(long)]
    pub(crate) prompt: Option<String>,

    /// Path to a safetensors file containing text embeddings for the scribble model.
    #[arg(long)]
    pub(crate) text_embeds: Option<PathBuf>,

    /// Tensor key in the text embedding safetensors file.
    #[arg(long, default_value = "input.text_embeds")]
    pub(crate) text_embeds_key: String,

    /// Optional weights root for TripoSG (image-only) pipeline.
    #[arg(long)]
    pub(crate) weights_root: Option<PathBuf>,

    /// Optional weights root for TripoSG-scribble pipeline.
    #[arg(long)]
    pub(crate) scribble_weights_root: Option<PathBuf>,

    /// Quality preset (fast, balanced, full). Individual flags override this preset.
    #[arg(long, value_enum, default_value_t = QualityPreset::Full)]
    pub(crate) quality: QualityPreset,

    /// Match Python TripoSG defaults for parity (full quality + fixed seed + decimation target).
    /// Enabled by default; pass --no-match-python to disable.
    #[arg(long, default_value_t = true)]
    pub(crate) match_python: bool,

    /// Disable Python parity defaults (use CLI-provided settings).
    #[arg(long, default_value_t = false)]
    pub(crate) no_match_python: bool,

    /// Number of diffusion steps (overrides --quality).
    #[arg(long)]
    pub(crate) num_steps: Option<usize>,

    /// Number of latents tokens (overrides --quality).
    #[arg(long)]
    pub(crate) num_tokens: Option<usize>,

    /// Guidance scale (overrides --quality).
    #[arg(long)]
    pub(crate) guidance_scale: Option<f32>,

    /// Optional RNG seed for deterministic sampling.
    #[arg(long)]
    pub(crate) seed: Option<u64>,

    /// Grid resolution used for mesh extraction (overrides --quality).
    #[arg(long)]
    pub(crate) resolution: Option<usize>,

    /// Chunk size for VAE grid decoding (overrides --quality).
    #[arg(long)]
    pub(crate) chunk_size: Option<usize>,

    /// Bounds for grid decoding (6 floats: minX minY minZ maxX maxY maxZ).
    #[arg(
        long,
        num_args = 6,
        value_delimiter = ' ',
        default_value = "-1.005 -1.005 -1.005 1.005 1.005 1.005"
    )]
    pub(crate) bounds: Vec<f32>,

    /// Mesh extraction mode.
    #[arg(long, value_enum, default_value_t = MeshMode::Flash)]
    pub(crate) mesh_mode: MeshMode,

    /// Dense octree depth used for hierarchical extraction (overrides --quality).
    #[arg(long)]
    pub(crate) dense_octree_depth: Option<usize>,

    /// Hierarchical octree depth used for hierarchical extraction (overrides --quality).
    #[arg(long)]
    pub(crate) hierarchical_octree_depth: Option<usize>,

    /// Band threshold used to expand near-surface regions in hierarchical extraction
    /// (overrides --quality).
    #[arg(long)]
    pub(crate) band_threshold: Option<f32>,

    /// Flash octree depth used for flash extraction (overrides --quality).
    #[arg(long)]
    pub(crate) flash_octree_depth: Option<usize>,

    /// Flash minimum resolution (overrides --quality).
    #[arg(long)]
    pub(crate) flash_min_resolution: Option<usize>,

    /// Flash mini grid count per axis (overrides --quality).
    #[arg(long)]
    pub(crate) flash_mini_grid_num: Option<usize>,

    /// Flash chunk size for VAE queries (overrides --quality).
    #[arg(long)]
    pub(crate) flash_num_chunks: Option<usize>,

    /// Flash iso offset (overrides --quality).
    #[arg(long)]
    pub(crate) flash_mc_level: Option<f32>,

    /// Target face count for mesh decimation (post-process). Use 0 to disable.
    /// Defaults to 10,000 when --match-python is enabled.
    #[arg(long)]
    pub(crate) faces: Option<usize>,

    /// Path to write an OBJ file for the inferred mesh.
    #[arg(long)]
    pub(crate) output: Option<PathBuf>,

    /// Path to an existing mesh file to display (glb/obj/gltf).
    #[arg(long)]
    pub(crate) mesh: Option<PathBuf>,

    /// Optional weights root for RMBG background removal.
    #[arg(long)]
    pub(crate) bg_weights_root: Option<PathBuf>,

    /// Background removal backend (auto, cpu, gpu).
    #[arg(long, value_enum, default_value_t = RmbgBackend::Auto)]
    pub(crate) rmbg_backend: RmbgBackend,

    /// DINO image encoder backend (auto, cpu, gpu).
    #[arg(long, value_enum, default_value_t = DinoBackend::Auto)]
    pub(crate) dino_backend: DinoBackend,

    /// Inference backend (cpu, wgpu, cuda).
    #[arg(long, value_enum, default_value_t = BackendKind::Wgpu)]
    pub(crate) backend: BackendKind,
}

#[derive(ValueEnum, Clone, Copy, Debug)]
pub(crate) enum RmbgBackend {
    Auto,
    Cpu,
    Gpu,
}

#[derive(ValueEnum, Clone, Copy, Debug)]
pub(crate) enum DinoBackend {
    Auto,
    Cpu,
    Gpu,
}

#[derive(ValueEnum, Clone, Debug)]
pub(crate) enum BackendKind {
    Cpu,
    Wgpu,
    Cuda,
}

#[derive(ValueEnum, Clone, Debug)]
pub(crate) enum QualityPreset {
    Fast,
    Balanced,
    Full,
}

#[derive(ValueEnum, Clone, Debug)]
pub(crate) enum MeshMode {
    Dense,
    Hierarchical,
    Flash,
}

pub(crate) const DEFAULT_CHUNK_SIZE: usize = 10_000;

#[derive(Clone, Copy, Debug)]
pub(crate) struct QualityDefaults {
    pub(crate) num_steps: usize,
    pub(crate) num_tokens: usize,
    pub(crate) guidance_scale: f32,
    pub(crate) resolution: usize,
    pub(crate) chunk_size: usize,
    pub(crate) dense_octree_depth: usize,
    pub(crate) hierarchical_octree_depth: usize,
    pub(crate) band_threshold: f32,
    pub(crate) flash_octree_depth: usize,
    pub(crate) flash_min_resolution: usize,
    pub(crate) flash_mini_grid_num: usize,
    pub(crate) flash_num_chunks: usize,
    pub(crate) flash_mc_level: f32,
}

impl QualityPreset {
    pub(crate) fn defaults(self) -> QualityDefaults {
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
pub(crate) struct AppArgs {
    pub(crate) image: Option<PathBuf>,
    pub(crate) prompt: Option<String>,
    pub(crate) text_embeds: Option<PathBuf>,
    pub(crate) text_embeds_key: String,
    pub(crate) weights_root: Option<PathBuf>,
    pub(crate) scribble_weights_root: Option<PathBuf>,
    pub(crate) num_steps: usize,
    pub(crate) num_tokens: usize,
    pub(crate) guidance_scale: f32,
    pub(crate) seed: Option<u64>,
    pub(crate) match_python: bool,
    pub(crate) resolution: usize,
    pub(crate) chunk_size: usize,
    pub(crate) bounds: Vec<f32>,
    pub(crate) mesh_mode: MeshMode,
    pub(crate) dense_octree_depth: usize,
    pub(crate) hierarchical_octree_depth: usize,
    pub(crate) band_threshold: f32,
    pub(crate) flash_octree_depth: usize,
    pub(crate) flash_min_resolution: usize,
    pub(crate) flash_mini_grid_num: usize,
    pub(crate) flash_num_chunks: usize,
    pub(crate) flash_mc_level: f32,
    pub(crate) target_faces: Option<usize>,
    pub(crate) output: Option<PathBuf>,
    pub(crate) mesh: Option<PathBuf>,
    pub(crate) bg_weights_root: Option<PathBuf>,
    pub(crate) backend: BackendKind,
    pub(crate) rmbg_backend: RmbgBackend,
    pub(crate) dino_backend: DinoBackend,
}

pub(crate) fn build_app_args(args: Args) -> AppArgs {
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
        backend: args.backend,
        rmbg_backend: args.rmbg_backend,
        dino_backend: args.dino_backend,
    }
}

#![recursion_limit = "256"]

use std::path::{Path, PathBuf};
use std::time::Instant;

use burn::prelude::*;
use burn_foreground::pipeline::{PrepareImageConfig, RmbgPipeline, prepare_image_tensor};
use burn_tripo::pipeline::geometry::{HierarchicalExtractConfig, hierarchical_extract_geometry};
use burn_tripo::pipeline::mesh::grid_to_mesh;
use burn_tripo::pipeline::triposg::TripoSGPipeline;

fn main() {
    let handle = std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(run_profile)
        .expect("failed to spawn profile thread");
    handle.join().expect("profile thread panicked");
}

fn run_profile() {
    let backend = std::env::var("TRIPOSG_PROFILE_BACKEND").unwrap_or_else(|_| "wgpu".to_string());
    match backend.as_str() {
        "cpu" => profile_with_backend::<burn::backend::NdArray<f32>>(),
        "wgpu" => profile_with_backend::<burn_wgpu::Wgpu>(),
        "cuda" => {
            #[cfg(feature = "cuda")]
            profile_with_backend::<burn_cuda::Cuda>();
            #[cfg(not(feature = "cuda"))]
            eprintln!("TRIPOSG_PROFILE_BACKEND=cuda requires the `cuda` feature");
        }
        other => eprintln!("Unknown TRIPOSG_PROFILE_BACKEND={other}; use cpu|wgpu|cuda"),
    }
}

fn profile_with_backend<B: Backend>() {
    let weights_root = resolve_weights_root(
        "TRIPOSG_WEIGHTS_ROOT",
        r"E:\\repos\\TripoSG\\pretrained_weights\\TripoSG",
    )
    .or_else(|| {
        resolve_weights_root(
            "TRIPOSG_WEIGHTS_ROOT",
            "./crates/burn_tripo/assets/models/MIDI-3D",
        )
    });
    let rmbg_root = resolve_weights_root(
        "RMBG_WEIGHTS_ROOT",
        r"E:\\repos\\TripoSG\\pretrained_weights\\RMBG-1.4",
    )
    .or_else(|| {
        resolve_weights_root(
            "RMBG_WEIGHTS_ROOT",
            "./crates/burn_foreground/assets/models/RMBG-1.4",
        )
    });

    let image_path = std::env::var("TRIPOSG_PROFILE_IMAGE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("docs/input_chair.jpg"));

    let Some(weights_root) = weights_root else {
        eprintln!("TripoSG weights not found; set TRIPOSG_WEIGHTS_ROOT to run profiling.");
        return;
    };
    let Some(rmbg_root) = rmbg_root else {
        eprintln!("RMBG weights not found; set RMBG_WEIGHTS_ROOT to run profiling.");
        return;
    };
    if !image_path.exists() {
        eprintln!("Profile image not found at {:?}", image_path);
        return;
    }

    let device = B::Device::default();

    let t0 = Instant::now();
    let mut pipeline = TripoSGPipeline::from_pretrained(weights_root, &device)
        .expect("failed to load TripoSG weights");
    let rmbg_pipeline =
        RmbgPipeline::from_pretrained(rmbg_root, &device).expect("failed to load RMBG weights");
    B::sync(&device);
    let load_ms = elapsed_ms(t0);

    let steps = env_usize("TRIPOSG_PROFILE_STEPS", 4);
    let tokens = env_usize("TRIPOSG_PROFILE_TOKENS", 256);
    let resolution = env_usize("TRIPOSG_PROFILE_RES", 64);
    let chunk_size = env_usize("TRIPOSG_PROFILE_CHUNK", 65_536);
    let dense_depth = env_usize("TRIPOSG_PROFILE_DENSE_DEPTH", 5);
    let hierarchical_depth = env_usize("TRIPOSG_PROFILE_HIER_DEPTH", 6);
    let guidance = env_f32("TRIPOSG_PROFILE_GUIDANCE", 7.0);

    let t0 = Instant::now();
    let image_tensor = prepare_image_tensor::<B>(
        &image_path,
        Some(&rmbg_pipeline),
        &device,
        &PrepareImageConfig::default(),
    )
    .expect("prepare image");
    B::sync(&device);
    let prepare_ms = elapsed_ms(t0);

    let t0 = Instant::now();
    let processed = pipeline.image_processor.preprocess(image_tensor.clone());
    B::sync(&device);
    let preprocess_ms = elapsed_ms(t0);

    let t0 = Instant::now();
    let embeds = pipeline
        .image_encoder
        .as_ref()
        .expect("TripoSG image encoder unavailable")
        .forward(processed);
    B::sync(&device);
    let encode_ms = elapsed_ms(t0);

    let t0 = Instant::now();
    let output =
        pipeline.sample_from_embeds(embeds.clone(), 1, steps, tokens, guidance, None, None);
    B::sync(&device);
    let diffusion_ms = elapsed_ms(t0);

    let t0 = Instant::now();
    let grid = pipeline
        .decode_grid(
            &output.latents,
            [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
            resolution,
            chunk_size,
        )
        .expect("dense grid");
    B::sync(&device);
    let dense_ms = elapsed_ms(t0);

    let config = HierarchicalExtractConfig {
        bounds: [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
        dense_octree_depth: dense_depth,
        hierarchical_octree_depth: hierarchical_depth,
        chunk_size,
        band_threshold: 1.0,
    };
    let t0 = Instant::now();
    let _hier_grid =
        hierarchical_extract_geometry(&output.latents, &pipeline.vae, &config).expect("hier grid");
    B::sync(&device);
    let hier_ms = elapsed_ms(t0);

    let t0 = Instant::now();
    let _mesh = grid_to_mesh(&grid, 0.0);
    let mesh_ms = elapsed_ms(t0);

    let t0 = Instant::now();
    let output_mesh = pipeline
        .sample_mesh_hierarchical(image_tensor, steps, tokens, guidance, &config, None)
        .expect("e2e mesh");
    B::sync(&device);
    let e2e_ms = elapsed_ms(t0);

    println!("=== TripoSG profile ({}) ===", backend_name::<B>());
    println!("weights_load_ms: {load_ms:.2}");
    println!("prepare_image_ms: {prepare_ms:.2}");
    println!("preprocess_ms: {preprocess_ms:.2}");
    println!("image_encode_ms: {encode_ms:.2}");
    println!("diffusion_ms: {diffusion_ms:.2}");
    println!("vae_decode_dense_ms: {dense_ms:.2}");
    println!("vae_decode_hier_ms: {hier_ms:.2}");
    println!(
        "mesh_ms: {mesh_ms:.2} (faces: {})",
        output_mesh
            .mesh
            .as_ref()
            .map(|m| m.faces.len())
            .unwrap_or(0)
    );
    println!("e2e_mesh_hier_ms: {e2e_ms:.2}");
}

fn backend_name<B: Backend>() -> &'static str {
    let name = std::any::type_name::<B>();
    if name.contains("Wgpu") {
        "wgpu"
    } else if name.contains("Cuda") {
        "cuda"
    } else {
        "cpu"
    }
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(default)
}

fn resolve_weights_root(env_var: &str, fallback: &str) -> Option<PathBuf> {
    if let Ok(value) = std::env::var(env_var) {
        let path = PathBuf::from(value);
        if let Some(root) = normalize_weights_root(&path) {
            return Some(root);
        }
    }
    let fallback = PathBuf::from(fallback);
    normalize_weights_root(&fallback)
}

fn normalize_weights_root(path: &Path) -> Option<PathBuf> {
    if path.is_dir() {
        return Some(path.to_path_buf());
    }
    if path.is_file() {
        return path.parent().map(|parent| parent.to_path_buf());
    }
    None
}

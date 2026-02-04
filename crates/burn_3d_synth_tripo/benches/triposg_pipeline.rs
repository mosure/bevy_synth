use std::path::{Path, PathBuf};
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use burn::prelude::*;
use burn_3d_synth_bg_removal::pipeline::{prepare_image_tensor, PrepareImageConfig, RmbgPipeline};
use burn_3d_synth_tripo::pipeline::geometry::{hierarchical_extract_geometry, HierarchicalExtractConfig};
use burn_3d_synth_tripo::pipeline::mesh::grid_to_mesh;
use burn_3d_synth_tripo::pipeline::triposg::TripoSGPipeline;

#[cfg(feature = "import")]
fn bench_triposg_steps(c: &mut Criterion) {
    let backend = std::env::var("TRIPOSG_BENCH_BACKEND").unwrap_or_else(|_| "wgpu".to_string());
    match backend.as_str() {
        "cpu" => bench_with_backend::<burn::backend::NdArray<f32>>(c),
        "wgpu" => bench_with_backend::<burn_wgpu::Wgpu>(c),
        "cuda" => {
            #[cfg(feature = "cuda")]
            bench_with_backend::<burn_cuda::Cuda>(c);
            #[cfg(not(feature = "cuda"))]
            eprintln!("TRIPOSG_BENCH_BACKEND=cuda requires the `cuda` feature");
        }
        other => eprintln!("Unknown TRIPOSG_BENCH_BACKEND={other}; use cpu|wgpu|cuda"),
    }
}

#[cfg(not(feature = "import"))]
fn bench_triposg_steps(_c: &mut Criterion) {
    eprintln!("Enable the `import` feature to run TripoSG benchmarks.");
}

#[cfg(feature = "import")]
fn bench_with_backend<B: Backend>(c: &mut Criterion) {
    let weights_root = resolve_weights_root("TRIPOSG_WEIGHTS_ROOT", r"E:\repos\TripoSG\pretrained_weights\TripoSG")
        .or_else(|| resolve_weights_root("TRIPOSG_WEIGHTS_ROOT", "./crates/burn_3d_synth_tripo/assets/models/MIDI-3D"));
    let rmbg_root = resolve_weights_root("RMBG_WEIGHTS_ROOT", r"E:\repos\TripoSG\pretrained_weights\RMBG-1.4")
        .or_else(|| resolve_weights_root("RMBG_WEIGHTS_ROOT", "./crates/burn_3d_synth_bg_removal/assets/models/RMBG-1.4"));

    let image_path = std::env::var("TRIPOSG_BENCH_IMAGE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("docs/input_chair.jpg"));

    if weights_root.is_none() {
        eprintln!("TripoSG weights not found; set TRIPOSG_WEIGHTS_ROOT to run benches.");
        return;
    }
    if rmbg_root.is_none() {
        eprintln!("RMBG weights not found; set RMBG_WEIGHTS_ROOT to run benches.");
        return;
    }
    if !image_path.exists() {
        eprintln!("Benchmark image not found at {:?}", image_path);
        return;
    }

    let device = B::Device::default();
    let mut pipeline = TripoSGPipeline::from_pretrained(weights_root.unwrap(), &device)
        .expect("failed to load TripoSG weights");
    let rmbg_pipeline = RmbgPipeline::from_pretrained(rmbg_root.unwrap(), &device)
        .expect("failed to load RMBG weights");

    let bench_steps = env_usize("TRIPOSG_BENCH_STEPS", 8);
    let bench_tokens = env_usize("TRIPOSG_BENCH_TOKENS", 512);
    let bench_resolution = env_usize("TRIPOSG_BENCH_RES", 128);
    let bench_chunk = env_usize("TRIPOSG_BENCH_CHUNK", 65_536);
    let dense_depth = env_usize("TRIPOSG_BENCH_DENSE_DEPTH", 6);
    let hierarchical_depth = env_usize("TRIPOSG_BENCH_HIER_DEPTH", 7);
    let guidance_scale = env_f32("TRIPOSG_BENCH_GUIDANCE", 7.0);

    let image_tensor = prepare_image_tensor::<B>(
        &image_path,
        Some(&rmbg_pipeline),
        &device,
        &PrepareImageConfig::default(),
    )
    .expect("failed to prepare image");

    let mut group = c.benchmark_group("triposg_pipeline");

    group.bench_function(BenchmarkId::new("prepare_image_tensor", backend_name::<B>()), |b| {
        b.iter(|| {
            let prepared = prepare_image_tensor::<B>(
                &image_path,
                Some(&rmbg_pipeline),
                &device,
                &PrepareImageConfig::default(),
            )
            .expect("failed to prepare image");
            std::hint::black_box(prepared)
        })
    });

    group.bench_function(BenchmarkId::new("image_preprocess", backend_name::<B>()), |b| {
        b.iter(|| {
            let processed = pipeline.image_processor.preprocess(image_tensor.clone());
            std::hint::black_box(processed)
        })
    });

    let processed = pipeline.image_processor.preprocess(image_tensor.clone());
    group.bench_function(BenchmarkId::new("image_encode", backend_name::<B>()), |b| {
        b.iter(|| {
            let embeds = pipeline.image_encoder.forward(processed.clone());
            std::hint::black_box(embeds)
        })
    });

    let image_embeds = pipeline.image_encoder.forward(processed.clone());
    group.bench_function(BenchmarkId::new("diffusion", backend_name::<B>()), |b| {
        b.iter(|| {
            let output = pipeline.sample_from_embeds(
                image_embeds.clone(),
                1,
                bench_steps,
                bench_tokens,
                guidance_scale,
                None,
                None,
            );
            std::hint::black_box(output.latents)
        })
    });

    let latents = pipeline.prepare_latents(
        1,
        bench_tokens,
        pipeline.transformer.config().in_channels,
        &device,
        None,
    );
    group.bench_function(BenchmarkId::new("vae_decode_dense", backend_name::<B>()), |b| {
        b.iter(|| {
            let grid = pipeline
                .decode_grid(
                    latents.clone(),
                    [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
                    bench_resolution,
                    bench_chunk,
                )
                .expect("decode grid");
            std::hint::black_box(grid)
        })
    });

    let config = HierarchicalExtractConfig {
        bounds: [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
        dense_octree_depth: dense_depth,
        hierarchical_octree_depth: hierarchical_depth,
        chunk_size: bench_chunk,
        band_threshold: 1.0,
    };

    group.bench_function(BenchmarkId::new("vae_decode_hier", backend_name::<B>()), |b| {
        b.iter(|| {
            let grid = hierarchical_extract_geometry(latents.clone(), &pipeline.vae, &config)
                .expect("hierarchical grid");
            std::hint::black_box(grid)
        })
    });

    let grid = pipeline
        .decode_grid(
            latents.clone(),
            [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
            64,
            bench_chunk,
        )
        .expect("decode grid");
    group.bench_function(BenchmarkId::new("mesh", backend_name::<B>()), |b| {
        b.iter(|| {
            let mesh = grid_to_mesh(&grid, 0.0);
            std::hint::black_box(mesh)
        })
    });

    group.bench_function(BenchmarkId::new("e2e_mesh_hier", backend_name::<B>()), |b| {
        b.iter(|| {
            let output = pipeline
                .sample_mesh_hierarchical(
                    image_tensor.clone(),
                    bench_steps,
                    bench_tokens,
                    guidance_scale,
                    &config,
                    None,
                )
                .expect("sample mesh");
            std::hint::black_box(output.mesh)
        })
    });

    group.finish();
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

fn criterion_config() -> Criterion {
    let sample_size = env_usize("TRIPOSG_BENCH_SAMPLE_SIZE", 10);
    let warmup = env_f32("TRIPOSG_BENCH_WARMUP_SECS", 3.0);
    let measure = env_f32("TRIPOSG_BENCH_MEASURE_SECS", 15.0);
    Criterion::default()
        .sample_size(sample_size.max(1))
        .warm_up_time(Duration::from_secs_f32(warmup.max(0.1)))
        .measurement_time(Duration::from_secs_f32(measure.max(0.5)))
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = bench_triposg_steps
}
criterion_main!(benches);

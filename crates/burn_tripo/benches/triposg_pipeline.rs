#![recursion_limit = "256"]

use std::path::{Path, PathBuf};
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use burn::prelude::*;
use burn_foreground::pipeline::{PrepareImageConfig, RmbgPipeline, prepare_image_tensor};
use burn_tripo::pipeline::geometry::{
    FlashExtractConfig, HierarchicalExtractConfig, flash_extract_geometry,
    hierarchical_extract_geometry,
};
use burn_tripo::pipeline::mesh::{grid_to_mesh, sdf_to_mesh_surface_nets};
use burn_tripo::pipeline::triposg::TripoSGPipeline;

#[derive(Clone, Copy, Debug)]
enum BenchPreset {
    Fast,
    Balanced,
    Full,
}

#[derive(Clone, Copy, Debug)]
struct BenchDefaults {
    steps: usize,
    tokens: usize,
    resolution: usize,
    chunk: usize,
    dense_depth: usize,
    hier_depth: usize,
    guidance: f32,
    flash_depth: usize,
    flash_min_res: usize,
    flash_mini: usize,
    flash_chunk: usize,
    flash_mc_level: f32,
}

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
    let weights_root = resolve_weights_root(
        "TRIPOSG_WEIGHTS_ROOT",
        r"E:\repos\TripoSG\pretrained_weights\TripoSG",
    )
    .or_else(|| {
        resolve_weights_root(
            "TRIPOSG_WEIGHTS_ROOT",
            "./crates/burn_tripo/assets/models/MIDI-3D",
        )
    });
    let rmbg_root = resolve_weights_root(
        "RMBG_WEIGHTS_ROOT",
        r"E:\repos\TripoSG\pretrained_weights\RMBG-1.4",
    )
    .or_else(|| {
        resolve_weights_root(
            "RMBG_WEIGHTS_ROOT",
            "./crates/burn_foreground/assets/models/RMBG-1.4",
        )
    });

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

    let preset = bench_preset();
    let defaults = bench_defaults(preset);
    let bench_steps = env_usize("TRIPOSG_BENCH_STEPS", defaults.steps);
    let bench_tokens = env_usize("TRIPOSG_BENCH_TOKENS", defaults.tokens);
    let bench_resolution = env_usize("TRIPOSG_BENCH_RES", defaults.resolution);
    let bench_chunk = env_usize("TRIPOSG_BENCH_CHUNK", defaults.chunk);
    let dense_depth = env_usize("TRIPOSG_BENCH_DENSE_DEPTH", defaults.dense_depth);
    let hierarchical_depth = env_usize("TRIPOSG_BENCH_HIER_DEPTH", defaults.hier_depth);
    let guidance_scale = env_f32("TRIPOSG_BENCH_GUIDANCE", defaults.guidance);
    let flash_octree_depth = env_usize("TRIPOSG_BENCH_FLASH_DEPTH", defaults.flash_depth);
    let flash_min_resolution = env_usize("TRIPOSG_BENCH_FLASH_MIN_RES", defaults.flash_min_res);
    let flash_mini_grid = env_usize("TRIPOSG_BENCH_FLASH_MINI", defaults.flash_mini);
    let flash_num_chunks = env_usize("TRIPOSG_BENCH_FLASH_CHUNK", defaults.flash_chunk);
    let flash_mc_level = env_f32("TRIPOSG_BENCH_FLASH_MC_LEVEL", defaults.flash_mc_level);

    let run_diffusion = std::env::var("TRIPOSG_BENCH_DIFFUSION").is_ok();
    let run_dense = std::env::var("TRIPOSG_BENCH_DENSE").is_ok();
    let run_hier = std::env::var("TRIPOSG_BENCH_HIER").is_ok();
    let run_mesh = std::env::var("TRIPOSG_BENCH_MESH").is_ok();
    let run_flash = std::env::var("TRIPOSG_BENCH_FLASH").is_ok();
    let run_e2e = std::env::var("TRIPOSG_BENCH_E2E").is_ok();

    let image_tensor = prepare_image_tensor::<B>(
        &image_path,
        Some(&rmbg_pipeline),
        &device,
        &PrepareImageConfig::default(),
    )
    .expect("failed to prepare image");

    let mut group = c.benchmark_group("triposg_pipeline");

    group.bench_function(
        BenchmarkId::new("prepare_image_tensor", backend_name::<B>()),
        |b| {
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
        },
    );

    group.bench_function(
        BenchmarkId::new("image_preprocess", backend_name::<B>()),
        |b| {
            b.iter(|| {
                let processed = pipeline.image_processor.preprocess(image_tensor.clone());
                std::hint::black_box(processed)
            })
        },
    );

    let processed = pipeline.image_processor.preprocess(image_tensor.clone());
    group.bench_function(BenchmarkId::new("image_encode", backend_name::<B>()), |b| {
        b.iter(|| {
            let embeds = pipeline.image_encoder.forward(processed.clone());
            std::hint::black_box(embeds)
        })
    });

    let image_embeds = pipeline.image_encoder.forward(processed.clone());
    if run_diffusion {
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
    }

    let latents = pipeline.prepare_latents(
        1,
        bench_tokens,
        pipeline.transformer.config().in_channels,
        &device,
        None,
    );
    if run_dense {
        group.bench_function(
            BenchmarkId::new("vae_decode_dense", backend_name::<B>()),
            |b| {
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
            },
        );
    }

    let config = HierarchicalExtractConfig {
        bounds: [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
        dense_octree_depth: dense_depth,
        hierarchical_octree_depth: hierarchical_depth,
        chunk_size: bench_chunk,
        band_threshold: 1.0,
    };
    let flash_config = FlashExtractConfig {
        bounds: [-1.005, -1.005, -1.005, 1.005, 1.005, 1.005],
        octree_depth: flash_octree_depth,
        num_chunks: flash_num_chunks,
        mc_level: flash_mc_level,
        min_resolution: flash_min_resolution,
        mini_grid_num: flash_mini_grid,
    };

    if run_hier {
        group.bench_function(
            BenchmarkId::new("vae_decode_hier", backend_name::<B>()),
            |b| {
                b.iter(|| {
                    let grid =
                        hierarchical_extract_geometry(latents.clone(), &pipeline.vae, &config)
                            .expect("hierarchical grid");
                    std::hint::black_box(grid)
                })
            },
        );
    }

    if run_mesh {
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
    }

    if run_flash {
        group.bench_function(
            BenchmarkId::new("flash_extract", backend_name::<B>()),
            |b| {
                b.iter(|| {
                    let grid =
                        flash_extract_geometry(latents.clone(), &pipeline.vae, &flash_config)
                            .expect("flash grid");
                    std::hint::black_box(grid)
                })
            },
        );

        let flash_grid = flash_extract_geometry(latents.clone(), &pipeline.vae, &flash_config)
            .expect("flash grid");
        group.bench_function(BenchmarkId::new("flash_mesh", backend_name::<B>()), |b| {
            b.iter(|| {
                let mesh = sdf_to_mesh_surface_nets(&flash_grid);
                std::hint::black_box(mesh)
            })
        });
    }

    if run_hier {
        group.bench_function(
            BenchmarkId::new("e2e_mesh_hier", backend_name::<B>()),
            |b| {
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
            },
        );
    }

    if run_flash && run_e2e {
        group.bench_function(
            BenchmarkId::new("e2e_mesh_flash", backend_name::<B>()),
            |b| {
                b.iter(|| {
                    let output = pipeline
                        .sample_mesh_flash(
                            image_tensor.clone(),
                            bench_steps,
                            bench_tokens,
                            guidance_scale,
                            &flash_config,
                            None,
                        )
                        .expect("sample mesh");
                    std::hint::black_box(output.mesh)
                })
            },
        );
    }

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

fn bench_preset() -> BenchPreset {
    if std::env::var("TRIPOSG_BENCH_FULL").is_ok() {
        return BenchPreset::Full;
    }
    if std::env::var("TRIPOSG_BENCH_BALANCED").is_ok() {
        return BenchPreset::Balanced;
    }
    if let Ok(value) = std::env::var("TRIPOSG_BENCH_PRESET") {
        match value.to_ascii_lowercase().as_str() {
            "full" => return BenchPreset::Full,
            "balanced" | "balance" => return BenchPreset::Balanced,
            "fast" => return BenchPreset::Fast,
            _ => {}
        }
    }
    BenchPreset::Fast
}

fn bench_defaults(preset: BenchPreset) -> BenchDefaults {
    match preset {
        BenchPreset::Fast => BenchDefaults {
            steps: 6,
            tokens: 256,
            resolution: 96,
            chunk: 4096,
            dense_depth: 4,
            hier_depth: 5,
            guidance: 7.0,
            flash_depth: 6,
            flash_min_res: 15,
            flash_mini: 1,
            flash_chunk: 2048,
            flash_mc_level: 0.0,
        },
        BenchPreset::Balanced => BenchDefaults {
            steps: 12,
            tokens: 512,
            resolution: 128,
            chunk: 4096,
            dense_depth: 5,
            hier_depth: 6,
            guidance: 7.0,
            flash_depth: 7,
            flash_min_res: 31,
            flash_mini: 2,
            flash_chunk: 4096,
            flash_mc_level: 0.0,
        },
        BenchPreset::Full => BenchDefaults {
            steps: 50,
            tokens: 2048,
            resolution: 512,
            chunk: 10_000,
            dense_depth: 8,
            hier_depth: 9,
            guidance: 7.0,
            flash_depth: 9,
            flash_min_res: 63,
            flash_mini: 4,
            flash_chunk: 10_000,
            flash_mc_level: 0.0,
        },
    }
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
    let preset = bench_preset();
    let (default_sample, default_warmup, default_measure) = match preset {
        BenchPreset::Fast => (10, 0.2, 0.8),
        BenchPreset::Balanced => (10, 0.5, 1.5),
        BenchPreset::Full => (10, 3.0, 15.0),
    };
    let sample_size = env_usize("TRIPOSG_BENCH_SAMPLE_SIZE", default_sample).max(10);
    let warmup = env_f32("TRIPOSG_BENCH_WARMUP_SECS", default_warmup);
    let measure = env_f32("TRIPOSG_BENCH_MEASURE_SECS", default_measure);
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

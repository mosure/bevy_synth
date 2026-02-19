#![cfg(feature = "import")]

use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Instant;

use burn::backend::NdArray;
use criterion::{criterion_group, criterion_main, Criterion};

use burn_foreground::pipeline::{prepare_image_data, PrepareImageConfig, RmbgPipeline};
use burn_foreground::rmbg2::Rmbg2Pipeline;

const FALLBACK_INPUT_IMAGE: &str = r"F:\repos\TRELLIS\assets\nano_banana\chair\chair_0.jpg";
const FALLBACK_RMBG14_ROOT: &str = r"E:\repos\TripoSG\pretrained_weights\RMBG-1.4";
const FALLBACK_RMBG2_ROOT: &str = r"F:\repos\burn_3d_synth\tmp_rmbg2_mirror2";

fn resolve_input_image() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("RMBG_TEST_IMAGE") {
        let path = PathBuf::from(path);
        if path.exists() {
            return Some(path);
        }
    }
    let path = PathBuf::from(FALLBACK_INPUT_IMAGE);
    path.exists().then_some(path)
}

fn resolve_rmbg14_root() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("RMBG14_WEIGHTS_ROOT") {
        let path = PathBuf::from(path);
        if path.exists() {
            return Some(path);
        }
    }
    if let Ok(path) = std::env::var("RMBG_WEIGHTS_ROOT") {
        let path = PathBuf::from(path);
        if path.exists() {
            return Some(path);
        }
    }
    let path = PathBuf::from(FALLBACK_RMBG14_ROOT);
    path.exists().then_some(path)
}

fn resolve_rmbg2_root() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("RMBG2_WEIGHTS_ROOT") {
        let path = PathBuf::from(path);
        if path.exists() {
            return Some(path);
        }
    }
    let path = PathBuf::from(FALLBACK_RMBG2_ROOT);
    path.exists().then_some(path)
}

fn bench_rmbg_variants(c: &mut Criterion) {
    let Some(input_image) = resolve_input_image() else {
        eprintln!("Skipping foreground benches: no input image found.");
        return;
    };

    let mut group = c.benchmark_group("foreground_prepare");
    let config = PrepareImageConfig::default();

    if let Some(root) = resolve_rmbg14_root() {
        match manual_bench_rmbg14(&input_image, &root, &config, 3) {
            Ok(mean_ms) => {
                eprintln!("Manual bench rmbg14 mean: {:.2} ms", mean_ms);
            }
            Err(err) => {
                eprintln!("Skipping rmbg14 bench: {err}");
            }
        }
    } else {
        eprintln!("Skipping rmbg14 bench: RMBG-1.4 weights root not found.");
    }

    if let Some(root) = resolve_rmbg2_root() {
        let pipeline = match Rmbg2Pipeline::from_pretrained(&root) {
            Ok(pipeline) => pipeline,
            Err(err) => {
                eprintln!("Skipping rmbg2 bench: failed to load burn pipeline: {err}");
                group.finish();
                return;
            }
        };
        let pipeline = Box::new(pipeline);
        group.bench_function("rmbg2", |b| {
            b.iter(|| {
                let prepared = pipeline
                    .prepare_image_data(&input_image, &config)
                    .expect("RMBG-2.0 prepare failed");
                black_box(prepared.data.len());
            })
        });
    } else {
        eprintln!("Skipping rmbg2 bench: RMBG-2.0 weights root not found.");
    }

    group.finish();
}

fn manual_bench_rmbg14(
    input_image: &Path,
    root: &Path,
    config: &PrepareImageConfig,
    runs: usize,
) -> Result<f64, String> {
    let input_image = input_image.to_path_buf();
    let root = root.to_path_buf();
    let config = config.clone();
    let runs = runs.max(1);

    let handle = thread::Builder::new()
        .name("rmbg14-manual-bench".to_string())
        .stack_size(64 * 1024 * 1024)
        .spawn(move || {
            let device = <NdArray<f32> as burn::tensor::backend::Backend>::Device::default();
            let pipeline = RmbgPipeline::from_pretrained(&root, &device)
                .map_err(|err| format!("failed to load RMBG-1.4 pipeline: {err}"))?;

            let mut total_ms = 0.0f64;
            for _ in 0..runs {
                let start = Instant::now();
                let prepared =
                    prepare_image_data::<NdArray<f32>>(&input_image, Some(&pipeline), &config)
                        .map_err(|err| format!("RMBG-1.4 prepare failed: {err}"))?;
                black_box(prepared.data.len());
                total_ms += start.elapsed().as_secs_f64() * 1000.0;
            }
            Ok::<f64, String>(total_ms / runs as f64)
        })
        .map_err(|err| format!("failed to spawn RMBG-1.4 bench thread: {err}"))?;

    handle
        .join()
        .map_err(|_| "RMBG-1.4 bench thread panicked".to_string())?
}

criterion_group!(benches, bench_rmbg_variants);
criterion_main!(benches);

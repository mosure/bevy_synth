use std::env;
use std::path::PathBuf;
use std::time::Instant;

use burn_segmentation::{
    SegmentationMask, SegmentationModelKind, SegmentationPrompt, SegmentationRuntime,
    SegmentationRuntimeBackend, SegmentationRuntimeConfig, SegmentationStageTimings,
};
use serde::Serialize;

#[derive(Debug, Serialize)]
struct BenchReport {
    model: String,
    variant: Option<String>,
    backend: String,
    model_root: String,
    image: String,
    image_size: [u32; 2],
    prompt_count: usize,
    warmup_runs: usize,
    measured_runs: usize,
    load_ms: f64,
    samples: Vec<BenchSample>,
    average: SegmentationStageTimings,
    masks: Vec<MaskSummary>,
}

#[derive(Debug, Serialize)]
struct BenchSample {
    run: usize,
    wall_ms: f64,
    stages: SegmentationStageTimings,
}

#[derive(Debug, Serialize)]
struct MaskSummary {
    object_id: String,
    score: f32,
    area_px: u32,
    bbox: [f32; 4],
}

fn main() {
    if let Err(err) = run() {
        eprintln!("segmentation_sam2_bench error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = Args::parse()?;
    let image = image::open(&args.image)
        .map_err(|err| format!("open image {}: {err}", args.image.display()))?;
    let prompt = SegmentationPrompt {
        object_id: "prompt_0".to_string(),
        label: "object".to_string(),
        bbox: args.bbox,
        point: None,
        source_query: Some("object".to_string()),
    };
    let load_start = Instant::now();
    let mut runtime = SegmentationRuntime::new(SegmentationRuntimeConfig {
        model: SegmentationModelKind::Sam2,
        backend: SegmentationRuntimeBackend::BurnNative,
        model_root: Some(args.model_root.clone()),
        profile_stages: true,
        ..SegmentationRuntimeConfig::default()
    })
    .map_err(|err| err.to_string())?;
    let load_ms = elapsed_ms(load_start);

    for _ in 0..args.warmup_runs {
        runtime
            .segment(&image, std::slice::from_ref(&prompt))
            .map_err(|err| err.to_string())?;
    }

    let mut samples = Vec::new();
    let mut masks = Vec::new();
    for run in 0..args.runs {
        let start = Instant::now();
        let result = runtime
            .segment(&image, std::slice::from_ref(&prompt))
            .map_err(|err| err.to_string())?;
        let wall_ms = elapsed_ms(start);
        let stages = runtime
            .last_stage_timings()
            .ok_or_else(|| "runtime did not record stage timings".to_string())?;
        if run + 1 == args.runs {
            masks = result.into_iter().map(mask_summary).collect();
        }
        samples.push(BenchSample {
            run,
            wall_ms,
            stages,
        });
    }
    let average = average_timings(&samples);
    let report = BenchReport {
        model: SegmentationModelKind::Sam2.label().to_string(),
        variant: runtime
            .sam_image_encoder_variant()
            .map(|variant| variant.label().to_string()),
        backend: SegmentationRuntimeBackend::BurnNative.label().to_string(),
        model_root: args.model_root.display().to_string(),
        image: args.image.display().to_string(),
        image_size: [image.width(), image.height()],
        prompt_count: 1,
        warmup_runs: args.warmup_runs,
        measured_runs: args.runs,
        load_ms,
        samples,
        average,
        masks,
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&report)
            .map_err(|err| format!("serialize bench report: {err}"))?
    );
    Ok(())
}

#[derive(Debug)]
struct Args {
    model_root: PathBuf,
    image: PathBuf,
    bbox: [f32; 4],
    warmup_runs: usize,
    runs: usize,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut args = env::args().skip(1);
        let mut model_root = None;
        let mut image = None;
        let mut bbox = None;
        let mut warmup_runs = 1usize;
        let mut runs = 3usize;
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--model-root" => model_root = args.next().map(PathBuf::from),
                "--image" => image = args.next().map(PathBuf::from),
                "--bbox" => bbox = args.next().map(parse_bbox).transpose()?,
                "--warmup-runs" => {
                    warmup_runs = args
                        .next()
                        .ok_or_else(|| "--warmup-runs requires a value".to_string())?
                        .parse()
                        .map_err(|err| format!("parse --warmup-runs: {err}"))?;
                }
                "--runs" => {
                    runs = args
                        .next()
                        .ok_or_else(|| "--runs requires a value".to_string())?
                        .parse()
                        .map_err(|err| format!("parse --runs: {err}"))?;
                }
                "--help" | "-h" => return Err(usage()),
                other => return Err(format!("unknown argument `{other}`\n{}", usage())),
            }
        }
        Ok(Self {
            model_root: model_root.ok_or_else(|| "--model-root is required".to_string())?,
            image: image.ok_or_else(|| "--image is required".to_string())?,
            bbox: bbox.unwrap_or([0.15, 0.10, 0.85, 0.90]),
            warmup_runs,
            runs,
        })
    }
}

fn parse_bbox(value: String) -> Result<[f32; 4], String> {
    let parts = value
        .split(',')
        .map(|part| {
            part.trim()
                .parse::<f32>()
                .map_err(|err| format!("parse bbox component `{part}`: {err}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if parts.len() != 4 {
        return Err(format!(
            "bbox requires four comma-separated values, got {value}"
        ));
    }
    Ok([parts[0], parts[1], parts[2], parts[3]])
}

fn usage() -> String {
    "usage: segmentation_sam2_bench --model-root DIR --image IMAGE [--bbox x0,y0,x1,y1] [--warmup-runs N] [--runs N]".to_string()
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn average_timings(samples: &[BenchSample]) -> SegmentationStageTimings {
    let n = samples.len().max(1) as f64;
    let mut out = SegmentationStageTimings::default();
    for sample in samples {
        out.preprocess_ms += sample.stages.preprocess_ms / n;
        out.encode_ms += sample.stages.encode_ms / n;
        out.prompt_ms += sample.stages.prompt_ms / n;
        out.decode_ms += sample.stages.decode_ms / n;
        out.postprocess_ms += sample.stages.postprocess_ms / n;
        out.total_ms += sample.stages.total_ms / n;
    }
    out
}

fn mask_summary(mask: SegmentationMask) -> MaskSummary {
    MaskSummary {
        object_id: mask.object_id,
        score: mask.score,
        area_px: mask.area_px,
        bbox: mask.bbox,
    }
}

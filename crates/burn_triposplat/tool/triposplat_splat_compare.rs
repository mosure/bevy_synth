use std::{fs, path::PathBuf};

use clap::Parser;
use serde::Serialize;

const SPLAT_RECORD_BYTES: usize = 32;

#[derive(Debug, Parser)]
#[command(about = "Compare TripoSplat .splat files record-by-record.")]
struct Args {
    reference: PathBuf,
    candidate: PathBuf,

    #[arg(long)]
    report: Option<PathBuf>,

    #[arg(long, default_value_t = 1.0e-4)]
    position_max_abs: f64,

    #[arg(long, default_value_t = 1.0e-5)]
    position_rms: f64,

    #[arg(long, default_value_t = 1.0e-5)]
    scale_max_abs: f64,

    #[arg(long, default_value_t = 1.0e-6)]
    scale_rms: f64,

    #[arg(long, default_value_t = 1)]
    rgba_max_abs: u8,

    #[arg(long, default_value_t = 1)]
    rotation_max_abs: u8,
}

#[derive(Debug, Serialize)]
struct Report {
    reference: String,
    candidate: String,
    reference_bytes: usize,
    candidate_bytes: usize,
    reference_records: usize,
    candidate_records: usize,
    thresholds: Thresholds,
    position: Option<FloatDiffSummary>,
    scale: Option<FloatDiffSummary>,
    rgba: Option<ByteDiffSummary>,
    rotation: Option<ByteDiffSummary>,
    failures: Vec<String>,
    passed: bool,
}

#[derive(Debug, Serialize)]
struct Thresholds {
    position_max_abs: f64,
    position_rms: f64,
    scale_max_abs: f64,
    scale_rms: f64,
    rgba_max_abs: u8,
    rotation_max_abs: u8,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct FloatDiffSummary {
    elements: usize,
    max_abs: f64,
    mean_abs: f64,
    rms: f64,
    max_abs_record: usize,
    max_abs_component: usize,
    reference_at_max_abs: f32,
    candidate_at_max_abs: f32,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct ByteDiffSummary {
    elements: usize,
    max_abs: u8,
    mean_abs: f64,
    rms: f64,
    max_abs_record: usize,
    max_abs_component: usize,
    reference_at_max_abs: u8,
    candidate_at_max_abs: u8,
}

#[derive(Clone, Copy, Debug)]
struct SplatRecord {
    position: [f32; 3],
    scale: [f32; 3],
    rgba: [u8; 4],
    rotation: [u8; 4],
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let reference_bytes = fs::read(&args.reference)?;
    let candidate_bytes = fs::read(&args.candidate)?;
    let reference_records = parse_splat_records(&reference_bytes);
    let candidate_records = parse_splat_records(&candidate_bytes);
    let mut failures = Vec::new();

    if reference_bytes.len() % SPLAT_RECORD_BYTES != 0 {
        failures.push(format!(
            "reference byte length {} is not divisible by {SPLAT_RECORD_BYTES}",
            reference_bytes.len()
        ));
    }
    if candidate_bytes.len() % SPLAT_RECORD_BYTES != 0 {
        failures.push(format!(
            "candidate byte length {} is not divisible by {SPLAT_RECORD_BYTES}",
            candidate_bytes.len()
        ));
    }
    if reference_records.len() != candidate_records.len() {
        failures.push(format!(
            "record count mismatch: reference={} candidate={}",
            reference_records.len(),
            candidate_records.len()
        ));
    }

    let comparable = reference_records.len().min(candidate_records.len());
    let (position, scale, rgba, rotation) = if comparable == 0 {
        (None, None, None, None)
    } else {
        let position = float_diff(
            &reference_records,
            &candidate_records,
            comparable,
            |record| record.position,
        );
        let scale = float_diff(
            &reference_records,
            &candidate_records,
            comparable,
            |record| record.scale,
        );
        let rgba = byte_diff(
            &reference_records,
            &candidate_records,
            comparable,
            |record| record.rgba,
        );
        let rotation = byte_diff(
            &reference_records,
            &candidate_records,
            comparable,
            |record| record.rotation,
        );

        if position.max_abs > args.position_max_abs {
            failures.push(format!(
                "position max_abs {:.6e} > {:.6e}",
                position.max_abs, args.position_max_abs
            ));
        }
        if position.rms > args.position_rms {
            failures.push(format!(
                "position rms {:.6e} > {:.6e}",
                position.rms, args.position_rms
            ));
        }
        if scale.max_abs > args.scale_max_abs {
            failures.push(format!(
                "scale max_abs {:.6e} > {:.6e}",
                scale.max_abs, args.scale_max_abs
            ));
        }
        if scale.rms > args.scale_rms {
            failures.push(format!(
                "scale rms {:.6e} > {:.6e}",
                scale.rms, args.scale_rms
            ));
        }
        if rgba.max_abs > args.rgba_max_abs {
            failures.push(format!(
                "rgba max_abs {} > {}",
                rgba.max_abs, args.rgba_max_abs
            ));
        }
        if rotation.max_abs > args.rotation_max_abs {
            failures.push(format!(
                "rotation max_abs {} > {}",
                rotation.max_abs, args.rotation_max_abs
            ));
        }

        (Some(position), Some(scale), Some(rgba), Some(rotation))
    };

    let report = Report {
        reference: args.reference.display().to_string(),
        candidate: args.candidate.display().to_string(),
        reference_bytes: reference_bytes.len(),
        candidate_bytes: candidate_bytes.len(),
        reference_records: reference_records.len(),
        candidate_records: candidate_records.len(),
        thresholds: Thresholds {
            position_max_abs: args.position_max_abs,
            position_rms: args.position_rms,
            scale_max_abs: args.scale_max_abs,
            scale_rms: args.scale_rms,
            rgba_max_abs: args.rgba_max_abs,
            rotation_max_abs: args.rotation_max_abs,
        },
        position,
        scale,
        rgba,
        rotation,
        passed: failures.is_empty(),
        failures,
    };

    let text = serde_json::to_string_pretty(&report)? + "\n";
    if let Some(path) = &args.report {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, &text)?;
    }
    print!("{text}");

    if report.passed {
        Ok(())
    } else {
        std::process::exit(1);
    }
}

fn parse_splat_records(bytes: &[u8]) -> Vec<SplatRecord> {
    bytes
        .chunks_exact(SPLAT_RECORD_BYTES)
        .map(|chunk| SplatRecord {
            position: [
                read_f32_le(chunk, 0),
                read_f32_le(chunk, 4),
                read_f32_le(chunk, 8),
            ],
            scale: [
                read_f32_le(chunk, 12),
                read_f32_le(chunk, 16),
                read_f32_le(chunk, 20),
            ],
            rgba: [chunk[24], chunk[25], chunk[26], chunk[27]],
            rotation: [chunk[28], chunk[29], chunk[30], chunk[31]],
        })
        .collect()
}

fn read_f32_le(bytes: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

fn float_diff(
    reference: &[SplatRecord],
    candidate: &[SplatRecord],
    count: usize,
    field: impl Fn(&SplatRecord) -> [f32; 3],
) -> FloatDiffSummary {
    let mut max_abs = 0.0f64;
    let mut max_abs_record = 0usize;
    let mut max_abs_component = 0usize;
    let mut reference_at_max_abs = 0.0f32;
    let mut candidate_at_max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut elements = 0usize;

    for record in 0..count {
        let expected = field(&reference[record]);
        let actual = field(&candidate[record]);
        for component in 0..3 {
            let diff = (expected[component] as f64 - actual[component] as f64).abs();
            if diff > max_abs {
                max_abs = diff;
                max_abs_record = record;
                max_abs_component = component;
                reference_at_max_abs = expected[component];
                candidate_at_max_abs = actual[component];
            }
            sum_abs += diff;
            sum_sq += diff * diff;
            elements += 1;
        }
    }

    FloatDiffSummary {
        elements,
        max_abs,
        mean_abs: sum_abs / elements as f64,
        rms: (sum_sq / elements as f64).sqrt(),
        max_abs_record,
        max_abs_component,
        reference_at_max_abs,
        candidate_at_max_abs,
    }
}

fn byte_diff(
    reference: &[SplatRecord],
    candidate: &[SplatRecord],
    count: usize,
    field: impl Fn(&SplatRecord) -> [u8; 4],
) -> ByteDiffSummary {
    let mut max_abs = 0u8;
    let mut max_abs_record = 0usize;
    let mut max_abs_component = 0usize;
    let mut reference_at_max_abs = 0u8;
    let mut candidate_at_max_abs = 0u8;
    let mut sum_abs = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut elements = 0usize;

    for record in 0..count {
        let expected = field(&reference[record]);
        let actual = field(&candidate[record]);
        for component in 0..4 {
            let diff = expected[component].abs_diff(actual[component]);
            if diff > max_abs {
                max_abs = diff;
                max_abs_record = record;
                max_abs_component = component;
                reference_at_max_abs = expected[component];
                candidate_at_max_abs = actual[component];
            }
            let diff = f64::from(diff);
            sum_abs += diff;
            sum_sq += diff * diff;
            elements += 1;
        }
    }

    ByteDiffSummary {
        elements,
        max_abs,
        mean_abs: sum_abs / elements as f64,
        rms: (sum_sq / elements as f64).sqrt(),
        max_abs_record,
        max_abs_component,
        reference_at_max_abs,
        candidate_at_max_abs,
    }
}

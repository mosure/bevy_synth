use std::{fs, path::PathBuf};

use burn_trellis::hook_diff::{HookDiffStatus, HookSnapshot, compare_hook_snapshots};
use clap::Parser;
use serde::Serialize;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Compare Trellis safetensor hook captures and report per-hook numerical deviations"
)]
struct Args {
    /// Reference hook safetensors file (typically Python trace).
    #[arg(long)]
    reference: PathBuf,

    /// Burn/runtime hook safetensors file to compare against reference.
    #[arg(long)]
    actual: PathBuf,

    /// Optional key prefix filter (e.g. "sample_shape_slat").
    #[arg(long)]
    prefix: Option<String>,

    /// Fail if any matching hook max_abs exceeds this threshold.
    #[arg(long)]
    fail_max_abs: Option<f32>,

    /// Fail if any matching hook mean_abs exceeds this threshold.
    #[arg(long)]
    fail_mean_abs: Option<f32>,

    /// Fail if any matching hook rmse exceeds this threshold.
    #[arg(long)]
    fail_rmse: Option<f32>,

    /// Allow keys that exist in reference but are missing in actual.
    #[arg(long, default_value_t = false)]
    allow_missing: bool,

    /// Allow shape mismatches.
    #[arg(long, default_value_t = false)]
    allow_shape_mismatch: bool,

    /// Allow keys that exist in actual but not in reference.
    #[arg(long, default_value_t = false)]
    allow_extra: bool,

    /// Optional path for a machine-readable JSON report.
    #[arg(long)]
    json_out: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct JsonReport {
    reference: String,
    actual: String,
    prefix: Option<String>,
    passed: bool,
    summary: JsonSummary,
    entries: Vec<JsonEntry>,
    extra_in_actual: Vec<String>,
}

#[derive(Debug, Serialize)]
struct JsonSummary {
    matched: usize,
    missing: usize,
    shape_mismatch: usize,
    extra: usize,
    worst_mean_abs: f32,
    worst_max_abs: f32,
    worst_rmse: f32,
    threshold_failures: usize,
}

#[derive(Debug, Serialize)]
struct JsonEntry {
    key: String,
    status: &'static str,
    reference_shape: Vec<usize>,
    actual_shape: Option<Vec<usize>>,
    stats: Option<JsonStats>,
    threshold_failed: bool,
}

#[derive(Debug, Serialize)]
struct JsonStats {
    mean_abs: f32,
    max_abs: f32,
    rmse: f32,
    non_finite_count: usize,
}

fn main() {
    let args = Args::parse();
    if let Err(err) = run(args) {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn run(args: Args) -> Result<(), String> {
    let reference = HookSnapshot::from_file(&args.reference)
        .map_err(|err| format!("failed to load reference hooks: {err}"))?;
    let actual = HookSnapshot::from_file(&args.actual)
        .map_err(|err| format!("failed to load actual hooks: {err}"))?;

    let report = compare_hook_snapshots(&reference, &actual, args.prefix.as_deref());
    let mut matched = 0usize;
    let mut missing = 0usize;
    let mut shape_mismatch = 0usize;
    let mut threshold_failures = 0usize;
    let mut worst_max_abs = 0.0f32;
    let mut worst_mean_abs = 0.0f32;
    let mut worst_rmse = 0.0f32;
    let mut json_entries = Vec::with_capacity(report.entries.len());

    println!(
        "{:<48} {:<16} {:>12} {:>12} {:>12} {:>12}",
        "hook", "status", "mean_abs", "max_abs", "rmse", "non_finite"
    );
    println!("{}", "-".repeat(121));

    for entry in &report.entries {
        match entry.status {
            HookDiffStatus::Match => {
                matched += 1;
                let stats = entry
                    .stats
                    .as_ref()
                    .ok_or_else(|| format!("missing stats for matched hook '{}'", entry.key))?;
                worst_max_abs = worst_max_abs.max(stats.max_abs);
                worst_mean_abs = worst_mean_abs.max(stats.mean_abs);
                worst_rmse = worst_rmse.max(stats.rmse);

                let mut failed = false;
                if let Some(limit) = args.fail_max_abs
                    && stats.max_abs > limit
                {
                    failed = true;
                }
                if let Some(limit) = args.fail_mean_abs
                    && stats.mean_abs > limit
                {
                    failed = true;
                }
                if let Some(limit) = args.fail_rmse
                    && stats.rmse > limit
                {
                    failed = true;
                }
                if stats.non_finite_count > 0 {
                    failed = true;
                }
                if failed {
                    threshold_failures += 1;
                }

                json_entries.push(JsonEntry {
                    key: entry.key.clone(),
                    status: "match",
                    reference_shape: entry.reference_shape.clone(),
                    actual_shape: entry.actual_shape.clone(),
                    stats: Some(JsonStats {
                        mean_abs: stats.mean_abs,
                        max_abs: stats.max_abs,
                        rmse: stats.rmse,
                        non_finite_count: stats.non_finite_count,
                    }),
                    threshold_failed: failed,
                });

                println!(
                    "{:<48} {:<16} {:>12.6e} {:>12.6e} {:>12.6e} {:>12}",
                    entry.key,
                    if failed {
                        "match(thresh-fail)"
                    } else {
                        "match"
                    },
                    stats.mean_abs,
                    stats.max_abs,
                    stats.rmse,
                    stats.non_finite_count
                );
            }
            HookDiffStatus::MissingInActual => {
                missing += 1;
                json_entries.push(JsonEntry {
                    key: entry.key.clone(),
                    status: "missing",
                    reference_shape: entry.reference_shape.clone(),
                    actual_shape: None,
                    stats: None,
                    threshold_failed: false,
                });
                println!(
                    "{:<48} {:<16} {:>12} {:>12} {:>12} {:>12}",
                    entry.key, "missing", "-", "-", "-", "-"
                );
            }
            HookDiffStatus::ShapeMismatch => {
                shape_mismatch += 1;
                json_entries.push(JsonEntry {
                    key: entry.key.clone(),
                    status: "shape_mismatch",
                    reference_shape: entry.reference_shape.clone(),
                    actual_shape: entry.actual_shape.clone(),
                    stats: None,
                    threshold_failed: false,
                });
                println!(
                    "{:<48} {:<16} {:>12} {:>12} {:>12} {:>12}",
                    entry.key, "shape-mismatch", "-", "-", "-", "-"
                );
                if let Some(actual_shape) = entry.actual_shape.as_ref() {
                    println!(
                        "  ref_shape={:?} actual_shape={:?}",
                        entry.reference_shape, actual_shape
                    );
                }
            }
        }
    }

    if !report.extra_in_actual.is_empty() {
        println!("\nextra hooks in actual (not in reference):");
        for key in &report.extra_in_actual {
            println!("  {key}");
        }
    }

    println!("\nsummary:");
    println!("  matched={matched}");
    println!("  missing={missing}");
    println!("  shape_mismatch={shape_mismatch}");
    println!("  extra={}", report.extra_in_actual.len());
    println!(
        "  worst(mean_abs={:.6e}, max_abs={:.6e}, rmse={:.6e})",
        worst_mean_abs, worst_max_abs, worst_rmse
    );
    println!("  threshold_failures={threshold_failures}");

    let failed = (missing > 0 && !args.allow_missing)
        || (shape_mismatch > 0 && !args.allow_shape_mismatch)
        || (!report.extra_in_actual.is_empty() && !args.allow_extra)
        || threshold_failures > 0;

    if let Some(json_out) = args.json_out.as_ref() {
        let json_report = JsonReport {
            reference: args.reference.display().to_string(),
            actual: args.actual.display().to_string(),
            prefix: args.prefix.clone(),
            passed: !failed,
            summary: JsonSummary {
                matched,
                missing,
                shape_mismatch,
                extra: report.extra_in_actual.len(),
                worst_mean_abs,
                worst_max_abs,
                worst_rmse,
                threshold_failures,
            },
            entries: json_entries,
            extra_in_actual: report.extra_in_actual.clone(),
        };
        if let Some(parent) = json_out.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)
                .map_err(|err| format!("failed to create json output directory: {err}"))?;
        }
        let json = serde_json::to_string_pretty(&json_report)
            .map_err(|err| format!("failed to encode json report: {err}"))?;
        fs::write(json_out, json).map_err(|err| format!("failed to write json report: {err}"))?;
    }

    if missing > 0 && !args.allow_missing {
        return Err("hook diff failed: missing hooks in actual".to_string());
    }
    if shape_mismatch > 0 && !args.allow_shape_mismatch {
        return Err("hook diff failed: shape mismatch detected".to_string());
    }
    if !report.extra_in_actual.is_empty() && !args.allow_extra {
        return Err("hook diff failed: extra hooks in actual".to_string());
    }
    if threshold_failures > 0 {
        return Err("hook diff failed: numeric threshold exceeded".to_string());
    }

    Ok(())
}

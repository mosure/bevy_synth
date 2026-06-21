use std::path::PathBuf;

use burn_trellis::hook_diff::{HookDiffStatus, HookSnapshot, compare_hook_snapshots};
use clap::Parser;

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
                println!(
                    "{:<48} {:<16} {:>12} {:>12} {:>12} {:>12}",
                    entry.key, "missing", "-", "-", "-", "-"
                );
            }
            HookDiffStatus::ShapeMismatch => {
                shape_mismatch += 1;
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

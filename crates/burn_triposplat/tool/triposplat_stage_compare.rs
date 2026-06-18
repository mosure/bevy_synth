use std::{fs, path::PathBuf};

use clap::Parser;
use safetensors::{
    SafeTensors,
    tensor::{Dtype, TensorView},
};
use serde::Serialize;

const DEFAULT_TENSORS: &[&str] = &["image_rgb_0_1", "feature1", "feature2", "latent", "camera"];

#[derive(Debug, Parser)]
#[command(about = "Compare TripoSplat stage safetensors with explicit numeric thresholds.")]
struct Args {
    reference: PathBuf,
    candidate: PathBuf,

    #[arg(long)]
    report: Option<PathBuf>,

    #[arg(long = "tensor")]
    tensors: Vec<String>,

    #[arg(long, default_value_t = 1.0e-2)]
    max_abs: f64,

    #[arg(long, default_value_t = 1.0e-3)]
    mean_abs: f64,

    #[arg(long, default_value_t = 2.0e-3)]
    rms: f64,
}

#[derive(Debug, Serialize)]
struct Report {
    reference: String,
    candidate: String,
    thresholds: Thresholds,
    tensors: Vec<TensorReport>,
    failures: Vec<String>,
    passed: bool,
}

#[derive(Debug, Serialize)]
struct Thresholds {
    max_abs: f64,
    mean_abs: f64,
    rms: f64,
}

#[derive(Debug, Serialize)]
struct TensorReport {
    name: String,
    reference_shape: Option<Vec<usize>>,
    candidate_shape: Option<Vec<usize>>,
    reference_dtype: Option<String>,
    candidate_dtype: Option<String>,
    diff: Option<DiffSummary>,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct DiffSummary {
    max_abs: f64,
    mean_abs: f64,
    rms: f64,
    max_abs_flat_index: Option<usize>,
    reference_at_max_abs: Option<f32>,
    candidate_at_max_abs: Option<f32>,
    count_abs_gt_threshold: usize,
    fraction_abs_gt_threshold: f64,
    reference_nonfinite_count: usize,
    candidate_nonfinite_count: usize,
    nonfinite_pair_count: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let reference_bytes = fs::read(&args.reference)?;
    let candidate_bytes = fs::read(&args.candidate)?;
    let reference = SafeTensors::deserialize(&reference_bytes)?;
    let candidate = SafeTensors::deserialize(&candidate_bytes)?;
    let selected = selected_tensors(&args, &reference);

    let mut report = Report {
        reference: args.reference.display().to_string(),
        candidate: args.candidate.display().to_string(),
        thresholds: Thresholds {
            max_abs: args.max_abs,
            mean_abs: args.mean_abs,
            rms: args.rms,
        },
        tensors: Vec::new(),
        failures: Vec::new(),
        passed: false,
    };

    if selected.is_empty() {
        report
            .failures
            .push("no tensors selected for comparison".to_string());
    }

    for name in selected {
        compare_tensor(&reference, &candidate, &name, &args, &mut report);
    }

    report.passed = report.failures.is_empty();
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

fn selected_tensors(args: &Args, reference: &SafeTensors<'_>) -> Vec<String> {
    if !args.tensors.is_empty() {
        return args.tensors.clone();
    }

    DEFAULT_TENSORS
        .iter()
        .copied()
        .filter(|name| reference.tensor(name).is_ok())
        .map(str::to_string)
        .collect()
}

fn compare_tensor(
    reference: &SafeTensors<'_>,
    candidate: &SafeTensors<'_>,
    name: &str,
    args: &Args,
    report: &mut Report,
) {
    let reference_view = match reference.tensor(name) {
        Ok(view) => view,
        Err(_) => {
            report
                .failures
                .push(format!("missing reference tensor {name}"));
            report.tensors.push(TensorReport::missing(name));
            return;
        }
    };
    let candidate_view = match candidate.tensor(name) {
        Ok(view) => view,
        Err(_) => {
            report
                .failures
                .push(format!("missing candidate tensor {name}"));
            report
                .tensors
                .push(TensorReport::reference_only(name, &reference_view));
            return;
        }
    };

    let mut tensor_report = TensorReport {
        name: name.to_string(),
        reference_shape: Some(reference_view.shape().to_vec()),
        candidate_shape: Some(candidate_view.shape().to_vec()),
        reference_dtype: Some(format!("{:?}", reference_view.dtype())),
        candidate_dtype: Some(format!("{:?}", candidate_view.dtype())),
        diff: None,
    };

    if reference_view.dtype() != Dtype::F32 {
        report.failures.push(format!(
            "{name} reference dtype {:?} is not F32",
            reference_view.dtype()
        ));
        report.tensors.push(tensor_report);
        return;
    }
    if candidate_view.dtype() != Dtype::F32 {
        report.failures.push(format!(
            "{name} candidate dtype {:?} is not F32",
            candidate_view.dtype()
        ));
        report.tensors.push(tensor_report);
        return;
    }
    if reference_view.shape() != candidate_view.shape() {
        report.failures.push(format!(
            "{name} shape mismatch: reference={:?} candidate={:?}",
            reference_view.shape(),
            candidate_view.shape()
        ));
        report.tensors.push(tensor_report);
        return;
    }

    let reference_values = match f32_values(&reference_view) {
        Ok(values) => values,
        Err(err) => {
            report
                .failures
                .push(format!("{name} failed to read reference values: {err}"));
            report.tensors.push(tensor_report);
            return;
        }
    };
    let candidate_values = match f32_values(&candidate_view) {
        Ok(values) => values,
        Err(err) => {
            report
                .failures
                .push(format!("{name} failed to read candidate values: {err}"));
            report.tensors.push(tensor_report);
            return;
        }
    };

    let diff = diff_summary(&reference_values, &candidate_values, args.max_abs);
    tensor_report.diff = Some(diff);
    if diff.max_abs > args.max_abs {
        report.failures.push(format!(
            "{name} max_abs {:.6e} > {:.6e}",
            diff.max_abs, args.max_abs
        ));
    }
    if diff.mean_abs > args.mean_abs {
        report.failures.push(format!(
            "{name} mean_abs {:.6e} > {:.6e}",
            diff.mean_abs, args.mean_abs
        ));
    }
    if diff.rms > args.rms {
        report
            .failures
            .push(format!("{name} rms {:.6e} > {:.6e}", diff.rms, args.rms));
    }
    if diff.reference_nonfinite_count > 0 || diff.candidate_nonfinite_count > 0 {
        report.failures.push(format!(
            "{name} contains non-finite values: reference={} candidate={} pairs={}",
            diff.reference_nonfinite_count,
            diff.candidate_nonfinite_count,
            diff.nonfinite_pair_count
        ));
    }
    report.tensors.push(tensor_report);
}

impl TensorReport {
    fn missing(name: &str) -> Self {
        Self {
            name: name.to_string(),
            reference_shape: None,
            candidate_shape: None,
            reference_dtype: None,
            candidate_dtype: None,
            diff: None,
        }
    }

    fn reference_only(name: &str, reference: &TensorView<'_>) -> Self {
        Self {
            name: name.to_string(),
            reference_shape: Some(reference.shape().to_vec()),
            candidate_shape: None,
            reference_dtype: Some(format!("{:?}", reference.dtype())),
            candidate_dtype: None,
            diff: None,
        }
    }
}

fn f32_values(view: &TensorView<'_>) -> Result<Vec<f32>, String> {
    let chunks = view.data().chunks_exact(4);
    if !chunks.remainder().is_empty() {
        return Err("F32 tensor byte length is not divisible by 4".to_string());
    }
    Ok(chunks
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn diff_summary(reference: &[f32], candidate: &[f32], max_abs_threshold: f64) -> DiffSummary {
    assert_eq!(
        reference.len(),
        candidate.len(),
        "shape validation should guarantee equal lengths"
    );
    if reference.is_empty() {
        return DiffSummary {
            max_abs: 0.0,
            mean_abs: 0.0,
            rms: 0.0,
            max_abs_flat_index: None,
            reference_at_max_abs: None,
            candidate_at_max_abs: None,
            count_abs_gt_threshold: 0,
            fraction_abs_gt_threshold: 0.0,
            reference_nonfinite_count: 0,
            candidate_nonfinite_count: 0,
            nonfinite_pair_count: 0,
        };
    }

    const NONFINITE_SENTINEL: f64 = 1.0e30;
    let mut max_abs = 0.0f64;
    let mut max_abs_flat_index = 0usize;
    let mut sum_abs = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut count_abs_gt_threshold = 0usize;
    let mut reference_nonfinite_count = 0usize;
    let mut candidate_nonfinite_count = 0usize;
    let mut nonfinite_pair_count = 0usize;
    for (index, (reference, candidate)) in reference.iter().zip(candidate.iter()).enumerate() {
        let reference_finite = reference.is_finite();
        let candidate_finite = candidate.is_finite();
        if !reference_finite {
            reference_nonfinite_count += 1;
        }
        if !candidate_finite {
            candidate_nonfinite_count += 1;
        }
        if !(reference_finite && candidate_finite) {
            nonfinite_pair_count += 1;
            count_abs_gt_threshold += 1;
            if max_abs < NONFINITE_SENTINEL {
                max_abs = NONFINITE_SENTINEL;
                max_abs_flat_index = index;
            }
            continue;
        }
        let diff = *candidate as f64 - *reference as f64;
        let abs = diff.abs();
        if abs > max_abs {
            max_abs = abs;
            max_abs_flat_index = index;
        }
        if abs > max_abs_threshold {
            count_abs_gt_threshold += 1;
        }
        sum_abs += abs;
        sum_sq += diff * diff;
    }
    let count = reference.len() as f64;
    let (mean_abs, rms) = if nonfinite_pair_count > 0 {
        (NONFINITE_SENTINEL, NONFINITE_SENTINEL)
    } else {
        (sum_abs / count, (sum_sq / count).sqrt())
    };
    DiffSummary {
        max_abs,
        mean_abs,
        rms,
        max_abs_flat_index: Some(max_abs_flat_index),
        reference_at_max_abs: Some(reference[max_abs_flat_index]),
        candidate_at_max_abs: Some(candidate[max_abs_flat_index]),
        count_abs_gt_threshold,
        fraction_abs_gt_threshold: count_abs_gt_threshold as f64 / count,
        reference_nonfinite_count,
        candidate_nonfinite_count,
        nonfinite_pair_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diff_summary_counts_candidate_nan_as_failure_magnitude() {
        let diff = diff_summary(&[1.0, 2.0, 3.0], &[1.0, f32::NAN, 3.0], 0.1);

        assert_eq!(diff.reference_nonfinite_count, 0);
        assert_eq!(diff.candidate_nonfinite_count, 1);
        assert_eq!(diff.nonfinite_pair_count, 1);
        assert_eq!(diff.count_abs_gt_threshold, 1);
        assert!(diff.max_abs > 1.0e20);
        assert!(diff.mean_abs > 1.0e20);
        assert!(diff.rms > 1.0e20);
    }
}

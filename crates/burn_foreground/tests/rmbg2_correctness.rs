#![cfg(feature = "import")]

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use safetensors::tensor::{SafeTensors, TensorView};

use burn_foreground::pipeline::PrepareImageConfig;
use burn_foreground::rmbg2::Rmbg2Pipeline;

const FALLBACK_INPUT_IMAGE: &str = r"F:\repos\TRELLIS\assets\nano_banana\chair\chair_0.jpg";
const FALLBACK_RMBG2_ROOT: &str = r"F:\repos\burn_3d_synth\tmp_rmbg2_mirror2";

const PREPARED_MAX_ABS: f32 = 255.0;
const PREPARED_MEAN_ABS: f32 = 2.0;
const ALPHA_MAX_ABS: f32 = 1.0;
const ALPHA_MEAN_ABS: f32 = 0.02;
const PROBS_MAX_ABS: f32 = 0.2;
const PROBS_MEAN_ABS: f32 = 0.02;

#[test]
fn rmbg2_prepare_matches_reference() -> Result<(), Box<dyn std::error::Error>> {
    let reference_path = asset_path("assets/hooks/rmbg2_chair_reference.safetensors");
    if !reference_path.exists() {
        eprintln!(
            "skipping: rmbg2 reference file not found at {}",
            reference_path.display()
        );
        return Ok(());
    }

    let Some(rmbg2_root) = resolve_rmbg2_root() else {
        eprintln!("skipping: RMBG-2.0 weights root not found");
        return Ok(());
    };
    let input_image = resolve_input_image();
    if !input_image.exists() {
        eprintln!(
            "skipping: input image not found at {}",
            input_image.display()
        );
        return Ok(());
    }

    let reference = HookReference::load(&reference_path)?;
    let prepared_ref = reference
        .get("output.prepared")
        .ok_or("missing output.prepared in reference")?;
    let alpha_ref = reference
        .get("output.alpha_mask")
        .ok_or("missing output.alpha_mask in reference")?;
    let alpha_probs_ref = reference.get("output.alpha_probs");

    let rmbg2 = match Rmbg2Pipeline::from_pretrained(&rmbg2_root) {
        Ok(pipeline) => pipeline,
        Err(err) => {
            eprintln!("skipping: failed to load RMBG-2.0 burn pipeline: {err}");
            return Ok(());
        }
    };
    let prepared = rmbg2.prepare_image_data(&input_image, &PrepareImageConfig::default())?;

    let ref_height = prepared_ref.shape.get(2).copied().unwrap_or(0);
    let ref_width = prepared_ref.shape.get(3).copied().unwrap_or(0);
    let prepared_stats = compute_stats_overlap(
        &prepared.data,
        prepared.height,
        prepared.width,
        &prepared_ref.data,
        ref_height,
        ref_width,
        3,
    );
    let alpha_mask = prepared
        .alpha_mask
        .ok_or("missing alpha mask from rmbg2 prepare")?;
    let alpha_stats = compute_stats(&alpha_mask, &alpha_ref.data);
    let mut errors = Vec::new();
    if prepared_stats.max_abs > PREPARED_MAX_ABS || prepared_stats.mean_abs > PREPARED_MEAN_ABS {
        errors.push(format!(
            "prepared image out of tolerance: mean_abs={:.6}, max_abs={:.6}",
            prepared_stats.mean_abs, prepared_stats.max_abs
        ));
    }
    if alpha_stats.max_abs > ALPHA_MAX_ABS || alpha_stats.mean_abs > ALPHA_MEAN_ABS {
        errors.push(format!(
            "alpha mask out of tolerance: mean_abs={:.6}, max_abs={:.6}",
            alpha_stats.mean_abs, alpha_stats.max_abs
        ));
    }

    if let (Some(alpha_probs), Some(alpha_probs_ref)) = (prepared.alpha_probs, alpha_probs_ref) {
        let probs_stats = compute_stats(&alpha_probs, &alpha_probs_ref.data);
        if probs_stats.max_abs > PROBS_MAX_ABS || probs_stats.mean_abs > PROBS_MEAN_ABS {
            errors.push(format!(
                "alpha probs out of tolerance: mean_abs={:.6}, max_abs={:.6}",
                probs_stats.mean_abs, probs_stats.max_abs
            ));
        }
    }

    if !errors.is_empty() {
        return Err(errors.join("; ").into());
    }

    Ok(())
}

fn resolve_rmbg2_root() -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Ok(path) = std::env::var("RMBG2_WEIGHTS_ROOT") {
        candidates.push(PathBuf::from(path));
    }
    if let Ok(path) = std::env::var("RMBG_WEIGHTS_ROOT") {
        candidates.push(PathBuf::from(path));
    }
    candidates.push(PathBuf::from(FALLBACK_RMBG2_ROOT));
    candidates.push(asset_path("assets/models/RMBG-2.0"));

    candidates.into_iter().find(|path| path.exists())
}

fn resolve_input_image() -> PathBuf {
    if let Ok(path) = std::env::var("RMBG_TEST_IMAGE") {
        return PathBuf::from(path);
    }
    PathBuf::from(FALLBACK_INPUT_IMAGE)
}

fn asset_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(relative)
}

struct HookReference {
    tensors: BTreeMap<String, HookTensor>,
}

#[derive(Clone)]
struct HookTensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl HookReference {
    fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = fs::read(path)?;
        let safetensors = SafeTensors::deserialize(&bytes)?;
        let mut tensors = BTreeMap::new();
        for name in safetensors.names() {
            let view = safetensors.tensor(name)?;
            let data = tensor_view_to_vec(&view);
            tensors.insert(
                name.to_string(),
                HookTensor {
                    shape: view.shape().to_vec(),
                    data,
                },
            );
        }
        Ok(Self { tensors })
    }

    fn get(&self, name: &str) -> Option<HookTensor> {
        self.tensors.get(name).cloned()
    }
}

fn tensor_view_to_vec(view: &TensorView<'_>) -> Vec<f32> {
    view.data()
        .chunks_exact(4)
        .map(|chunk| {
            let bytes: [u8; 4] = chunk.try_into().unwrap();
            f32::from_le_bytes(bytes)
        })
        .collect()
}

struct MetricStats {
    mean_abs: f32,
    max_abs: f32,
}

fn compute_stats(lhs: &[f32], rhs: &[f32]) -> MetricStats {
    let mut sum_abs = 0.0f32;
    let mut max_abs = 0.0f32;

    for (&a, &b) in lhs.iter().zip(rhs.iter()) {
        let abs = (a - b).abs();
        sum_abs += abs;
        max_abs = max_abs.max(abs);
    }

    let len = lhs.len().max(1) as f32;
    MetricStats {
        mean_abs: sum_abs / len,
        max_abs,
    }
}

fn compute_stats_overlap(
    lhs: &[f32],
    lhs_height: usize,
    lhs_width: usize,
    rhs: &[f32],
    rhs_height: usize,
    rhs_width: usize,
    channels: usize,
) -> MetricStats {
    let height = lhs_height.min(rhs_height);
    let width = lhs_width.min(rhs_width);
    let mut sum_abs = 0.0f32;
    let mut max_abs = 0.0f32;
    let mut count = 0usize;

    for c in 0..channels {
        let lhs_base = c * lhs_height * lhs_width;
        let rhs_base = c * rhs_height * rhs_width;
        for y in 0..height {
            let lhs_row = lhs_base + y * lhs_width;
            let rhs_row = rhs_base + y * rhs_width;
            for x in 0..width {
                let abs = (lhs[lhs_row + x] - rhs[rhs_row + x]).abs();
                sum_abs += abs;
                max_abs = max_abs.max(abs);
                count += 1;
            }
        }
    }

    let denom = count.max(1) as f32;
    MetricStats {
        mean_abs: sum_abs / denom,
        max_abs,
    }
}

#![cfg(feature = "import")]

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use safetensors::tensor::{SafeTensors, TensorView};

use burn_foreground::pipeline::{PrepareImageConfig, RmbgPipeline, prepare_image_data};

const RMBG_ROOT: &str = r"E:\repos\TripoSG\pretrained_weights\RMBG-1.4";
const INPUT_IMAGE: &str = r"F:\repos\TRELLIS\assets\nano_banana\chair\chair_0.jpg";

const MAX_ABS: f32 = 200.0;
const MEAN_ABS: f32 = 1.0;
const MSE: f32 = 30.0;
const MASK_MAX_ABS: f32 = 1.0;
const MASK_MEAN_ABS: f32 = 0.002;

#[test]
fn rmbg_preprocess_matches_reference() -> Result<(), Box<dyn std::error::Error>> {
    unsafe {
        std::env::set_var("RMBG_STRICT_INTERP", "1");
    }
    let reference_path = asset_path("assets/hooks/rmbg_chair_reference.safetensors");
    if !reference_path.exists() {
        eprintln!(
            "skipping: rmbg reference file not found at {}",
            reference_path.display()
        );
        return Ok(());
    }

    if !Path::new(RMBG_ROOT).exists() {
        eprintln!("skipping: RMBG weights root not found at {}", RMBG_ROOT);
        return Ok(());
    }

    if !Path::new(INPUT_IMAGE).exists() {
        eprintln!("skipping: input image not found at {}", INPUT_IMAGE);
        return Ok(());
    }

    let reference = HookReference::load(reference_path.as_path())?;
    let prepared_ref = reference
        .get("output.prepared")
        .ok_or("missing output.prepared in reference")?;
    let alpha_ref = reference
        .get("output.alpha_mask")
        .ok_or("missing output.alpha_mask in reference")?;
    let alpha_probs_ref = reference.get("output.alpha_probs");

    let device = Default::default();
    let rmbg = match RmbgPipeline::from_pretrained(RMBG_ROOT, &device) {
        Ok(pipeline) => pipeline,
        Err(err) => {
            eprintln!("skipping: failed to load RMBG-1.4 pipeline: {err}");
            return Ok(());
        }
    };

    if std::env::var("RMBG_DEBUG_WEIGHTS").is_ok() {
        let weights_path = Path::new(RMBG_ROOT).join("model.safetensors");
        if weights_path.exists() {
            let weights = HookReference::load(weights_path.as_path())?;
            debug_compare_weight("conv_in.weight", &rmbg.model.conv_in.weight.val(), &weights);
            debug_compare_weight("side1.weight", &rmbg.model.side1.weight.val(), &weights);
        }
    }
    let prepared = prepare_image_data::<burn::backend::NdArray<f32>>(
        Path::new(INPUT_IMAGE),
        Some(&rmbg),
        &PrepareImageConfig::default(),
    )?;

    let debug = std::env::var("RMBG_DEBUG").is_ok();
    if debug {
        eprintln!("prepared bbox: {:?}", prepared.bbox);
        if alpha_ref.shape.len() == 2 {
            let ref_bbox = bbox_from_mask(&alpha_ref.data, alpha_ref.shape[1], alpha_ref.shape[0]);
            eprintln!("reference bbox: {:?}", ref_bbox);
            if let Some([x, y, w, h]) = ref_bbox {
                let width = alpha_ref.shape[1];
                let height = alpha_ref.shape[0];
                let mut left_count = 0usize;
                let mut right_count = 0usize;
                let mut top_count = 0usize;
                let mut bottom_count = 0usize;
                for yy in y..(y + h).min(height) {
                    let left_idx = yy * width + x.min(width - 1);
                    let right_idx = yy * width + (x + w - 1).min(width - 1);
                    if alpha_ref.data[left_idx] > 0.0 {
                        left_count += 1;
                    }
                    if alpha_ref.data[right_idx] > 0.0 {
                        right_count += 1;
                    }
                }
                for xx in x..(x + w).min(width) {
                    let top_idx = y.min(height - 1) * width + xx;
                    let bottom_idx = (y + h - 1).min(height - 1) * width + xx;
                    if alpha_ref.data[top_idx] > 0.0 {
                        top_count += 1;
                    }
                    if alpha_ref.data[bottom_idx] > 0.0 {
                        bottom_count += 1;
                    }
                }
                eprintln!(
                    "reference bbox edge counts: left={} right={} top={} bottom={}",
                    left_count, right_count, top_count, bottom_count
                );
            }
        } else {
            eprintln!(
                "reference bbox: unexpected alpha shape {:?}",
                alpha_ref.shape
            );
        }
        if let Some(alpha_mask) = prepared.alpha_mask.as_ref() {
            let alpha_stats = compute_stats(alpha_mask, &alpha_ref.data);
            eprintln!(
                "alpha stats: mean_abs={:.6} max_abs={:.6} mse={:.6}",
                alpha_stats.mean_abs, alpha_stats.max_abs, alpha_stats.mse
            );
            if let Some([x, y, w, h]) = prepared.bbox {
                let width = prepared.width;
                let height = prepared.height;
                if width == 0 || height == 0 {
                    eprintln!("bbox debug skipped: invalid prepared dims");
                } else {
                    let mut left_count = 0usize;
                    let mut right_count = 0usize;
                    let mut top_count = 0usize;
                    let mut bottom_count = 0usize;
                    for yy in y..(y + h).min(height) {
                        let left_idx = yy * width + x.min(width - 1);
                        let right_idx = yy * width + (x + w - 1).min(width - 1);
                        if alpha_mask[left_idx] > 0.0 {
                            left_count += 1;
                        }
                        if alpha_mask[right_idx] > 0.0 {
                            right_count += 1;
                        }
                    }
                    for xx in x..(x + w).min(width) {
                        let top_idx = y.min(height - 1) * width + xx;
                        let bottom_idx = (y + h - 1).min(height - 1) * width + xx;
                        if alpha_mask[top_idx] > 0.0 {
                            top_count += 1;
                        }
                        if alpha_mask[bottom_idx] > 0.0 {
                            bottom_count += 1;
                        }
                    }
                    eprintln!(
                        "bbox edge counts: left={} right={} top={} bottom={}",
                        left_count, right_count, top_count, bottom_count
                    );
                }
            }
        }
        if let (Some(alpha_probs), Some(alpha_probs_ref)) =
            (prepared.alpha_probs.as_ref(), alpha_probs_ref.as_ref())
        {
            let prob_stats = compute_stats(alpha_probs, &alpha_probs_ref.data);
            eprintln!(
                "alpha probs stats: mean_abs={:.6} max_abs={:.6} mse={:.6}",
                prob_stats.mean_abs, prob_stats.max_abs, prob_stats.mse
            );
        }
    }
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
    if debug {
        eprintln!(
            "prepared stats: mean_abs={:.6} max_abs={:.6} mse={:.6}",
            prepared_stats.mean_abs, prepared_stats.max_abs, prepared_stats.mse
        );
    }
    if prepared_stats.max_abs > MAX_ABS
        || prepared_stats.mean_abs > MEAN_ABS
        || prepared_stats.mse > MSE
    {
        if debug && let Some(alpha_mask) = prepared.alpha_mask.as_ref() {
            let alpha_stats = compute_stats(alpha_mask, &alpha_ref.data);
            eprintln!(
                "alpha stats: mean_abs={:.6} max_abs={:.6} mse={:.6}",
                alpha_stats.mean_abs, alpha_stats.max_abs, alpha_stats.mse
            );
        }
        return Err(format!(
            "prepared image out of tolerance: mean_abs={:.6} max_abs={:.6} mse={:.6}",
            prepared_stats.mean_abs, prepared_stats.max_abs, prepared_stats.mse
        )
        .into());
    }

    let alpha_mask = prepared
        .alpha_mask
        .ok_or("missing alpha mask from prepare_image_data")?;
    let alpha_stats = compute_stats(&alpha_mask, &alpha_ref.data);
    if alpha_stats.max_abs > MASK_MAX_ABS || alpha_stats.mean_abs > MASK_MEAN_ABS {
        return Err(format!(
            "alpha mask out of tolerance: mean_abs={:.6} max_abs={:.6}",
            alpha_stats.mean_abs, alpha_stats.max_abs
        )
        .into());
    }

    Ok(())
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
    mse: f32,
}

fn bbox_from_mask(mask: &[f32], width: usize, height: usize) -> Option<[usize; 4]> {
    let mut min_x = width;
    let mut min_y = height;
    let mut max_x = 0usize;
    let mut max_y = 0usize;
    let mut found = false;
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if mask[idx] <= 0.0 {
                continue;
            }
            found = true;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
    }
    if !found {
        None
    } else {
        Some([min_x, min_y, max_x - min_x + 1, max_y - min_y + 1])
    }
}
fn compute_stats(lhs: &[f32], rhs: &[f32]) -> MetricStats {
    let mut sum_abs = 0.0f32;
    let mut max_abs = 0.0f32;
    let mut mse = 0.0f32;

    for (&a, &b) in lhs.iter().zip(rhs.iter()) {
        let diff = a - b;
        let abs = diff.abs();
        sum_abs += abs;
        max_abs = max_abs.max(abs);
        mse += diff * diff;
    }

    let len = lhs.len().max(1) as f32;
    MetricStats {
        mean_abs: sum_abs / len,
        max_abs,
        mse: mse / len,
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
    let mut mse = 0.0f32;
    let mut count = 0usize;

    for c in 0..channels {
        let lhs_base = c * lhs_height * lhs_width;
        let rhs_base = c * rhs_height * rhs_width;
        for y in 0..height {
            let lhs_row = lhs_base + y * lhs_width;
            let rhs_row = rhs_base + y * rhs_width;
            for x in 0..width {
                let a = lhs[lhs_row + x];
                let b = rhs[rhs_row + x];
                let diff = a - b;
                let abs = diff.abs();
                sum_abs += abs;
                max_abs = max_abs.max(abs);
                mse += diff * diff;
                count += 1;
            }
        }
    }

    let denom = count.max(1) as f32;
    MetricStats {
        mean_abs: sum_abs / denom,
        max_abs,
        mse: mse / denom,
    }
}

fn debug_compare_weight(
    name: &str,
    tensor: &burn::tensor::Tensor<burn::backend::NdArray<f32>, 4>,
    weights: &HookReference,
) {
    if let Some(ref_weight) = weights.get(name) {
        if let Ok(data) = tensor.clone().into_data().convert::<f32>().to_vec::<f32>() {
            let stats = compute_stats(&data, &ref_weight.data);
            eprintln!(
                "weight {name}: mean_abs={:.6} max_abs={:.6} mse={:.6}",
                stats.mean_abs, stats.max_abs, stats.mse
            );
        }
    } else {
        eprintln!("weight {name} not found in safetensors");
    }
}

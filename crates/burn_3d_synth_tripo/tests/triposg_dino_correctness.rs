#![cfg(feature = "import")]

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use burn::prelude::*;
use safetensors::tensor::{SafeTensors, TensorView};

use burn_3d_synth_tripo::model::triposg::{
    hooks::HookRecorder,
    image_encoder::import::{load_dinov2_processor, load_triposg_dinov2},
};

const FALLBACK_WEIGHTS_PATH: &str = "assets/models/MIDI-3D/image_encoder_dinov2/model.safetensors";
const TRIPOSG_ROOT: &str = r"E:\repos\TripoSG\pretrained_weights\TripoSG";

const PATCH_MAX_ABS: f32 = 1.5;
const PATCH_MEAN_ABS: f32 = 0.12;
const PATCH_MSE: f32 = 0.05;
const CLS_MAX_ABS: f32 = 1.0;
const PREP_MAX_ABS: f32 = 0.15;
const PREP_MEAN_ABS: f32 = 1e-2;
const PREP_MSE: f32 = 1e-3;

#[test]
fn triposg_dino_hooks_match_reference() -> Result<(), Box<dyn std::error::Error>> {
    unsafe {
        std::env::set_var("DINO_STRICT_PREPROCESS", "1");
    }
    let reference_path = asset_path("assets/hooks/triposg_dino_reference.safetensors");
    if !reference_path.exists() {
        eprintln!(
            "skipping: hook reference file not found at {}",
            reference_path.display()
        );
        return Ok(());
    }

    let weights_path = resolve_weights_path("image_encoder_dinov2/model.safetensors");
    if !weights_path.exists() {
        eprintln!(
            "skipping: DINOv2 weights file not found at {}",
            weights_path.display()
        );
        return Ok(());
    }
    let weights_root = resolve_weights_root_from_path(&weights_path);

    let reference = HookReference::load(reference_path.as_path())?;
    let device = Default::default();
    let image_encoder = load_triposg_dinov2::<burn::backend::NdArray<f32>>(&device, &weights_path)?;
    let image_processor = load_dinov2_processor(weights_root)?;

    let input = reference
        .get_input("input.image")
        .ok_or("missing input.image in reference")?;
    let image = tensor_from_data::<burn::backend::NdArray<f32>>(&input, &device)?;

    let processed = image_processor.preprocess(image);
    let output = image_encoder.dino.forward(processed.clone(), None);
    let cls = output.x_norm_clstoken.unsqueeze_dim(1);
    let patch = output.x_norm_patchtokens;
    let image_embeds = Tensor::cat(vec![cls.clone(), patch.clone()], 1);

    let mut hooks = HookRecorder::new();
    hooks.record_tensor("image.preprocessed", &processed);
    hooks.record_tensor("output.image_embeds", &image_embeds);
    hooks.record_tensor("output.cls_token", &cls);
    hooks.record_tensor("output.patch_tokens", &patch);

    for (name, reference_tensor) in reference.tensors.iter() {
        if name.starts_with("input.") {
            continue;
        }
        let Some(burn_tensor) = hooks.tensors().get(name) else {
            return Err(format!("missing hook tensor `{name}` in burn output").into());
        };
        if burn_tensor.shape != reference_tensor.shape {
            return Err(format!(
                "shape mismatch for `{name}`: burn {:?} vs ref {:?}",
                burn_tensor.shape, reference_tensor.shape
            )
            .into());
        }

        let stats = compute_stats(&burn_tensor.data, &reference_tensor.data);
        let (max_abs_limit, mean_abs_limit, mse_limit) = if name == "image.preprocessed" {
            (PREP_MAX_ABS, PREP_MEAN_ABS, PREP_MSE)
        } else if name.contains("cls_token") {
            (CLS_MAX_ABS, PATCH_MEAN_ABS, PATCH_MSE)
        } else {
            (PATCH_MAX_ABS, PATCH_MEAN_ABS, PATCH_MSE)
        };
        if stats.max_abs > max_abs_limit || stats.mean_abs > mean_abs_limit || stats.mse > mse_limit
        {
            return Err(format!(
                "hook `{name}` out of tolerance: mean_abs={:.6} max_abs={:.6} mse={:.6}",
                stats.mean_abs, stats.max_abs, stats.mse
            )
            .into());
        }
    }

    Ok(())
}

fn resolve_weights_path(leaf: &str) -> std::path::PathBuf {
    if let Ok(root) = std::env::var("TRIPOSG_WEIGHTS_ROOT") {
        let candidate = Path::new(&root).join(leaf);
        if candidate.exists() {
            return candidate;
        }
    }
    let candidate = Path::new(TRIPOSG_ROOT).join(leaf);
    if candidate.exists() {
        return candidate;
    }
    asset_path(FALLBACK_WEIGHTS_PATH)
}

fn asset_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(relative)
}

fn resolve_weights_root_from_path(weights_path: &Path) -> std::path::PathBuf {
    if weights_path.is_dir() {
        return weights_path.to_path_buf();
    }
    let parent = weights_path.parent().unwrap_or_else(|| Path::new("."));
    if parent
        .file_name()
        .map(|name| name == "image_encoder_dinov2")
        .unwrap_or(false)
        && let Some(root) = parent.parent()
    {
        return root.to_path_buf();
    }
    parent.to_path_buf()
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

    fn get_input(&self, name: &str) -> Option<HookTensor> {
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

fn tensor_from_data<B: Backend>(
    tensor: &HookTensor,
    device: &B::Device,
) -> Result<Tensor<B, 4>, Box<dyn std::error::Error>> {
    let shape: [usize; 4] = tensor
        .shape
        .clone()
        .try_into()
        .map_err(|_| "unexpected input rank")?;
    let data = Tensor::<B, 1>::from_floats(tensor.data.as_slice(), device);
    Ok(data.reshape([
        shape[0] as i32,
        shape[1] as i32,
        shape[2] as i32,
        shape[3] as i32,
    ]))
}

struct MetricStats {
    mean_abs: f32,
    max_abs: f32,
    mse: f32,
}

fn compute_stats(burn: &[f32], reference: &[f32]) -> MetricStats {
    let mut sum_abs = 0.0f32;
    let mut max_abs = 0.0f32;
    let mut mse = 0.0f32;

    for (&lhs, &rhs) in burn.iter().zip(reference.iter()) {
        let diff = lhs - rhs;
        let abs = diff.abs();
        sum_abs += abs;
        max_abs = max_abs.max(abs);
        mse += diff * diff;
    }

    let len = burn.len().max(1) as f32;
    MetricStats {
        mean_abs: sum_abs / len,
        max_abs,
        mse: mse / len,
    }
}

#![cfg(feature = "import")]

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use burn::prelude::*;
use safetensors::tensor::{SafeTensors, TensorView};

use burn_tripo::model::triposg::image_encoder::import::{
    load_triposg_dinov2, load_triposg_dinov2_from_safetensors,
};

const TRIPOSG_ROOT: &str = r"E:\repos\TripoSG\pretrained_weights\TripoSG";

#[test]
fn dino_burnpack_and_safetensors_loader_outputs_match_reference()
-> Result<(), Box<dyn std::error::Error>> {
    let reference_path = std::env::var("TRIPOSG_DINO_PARITY_REFERENCE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| asset_path("assets/hooks/triposg_pipeline_reference.safetensors"));
    if !reference_path.exists() {
        eprintln!(
            "skipping: pipeline reference file not found at {}",
            reference_path.display()
        );
        return Ok(());
    }

    let weights_root = resolve_weights_root();
    if !weights_root.exists() {
        eprintln!(
            "skipping: TripoSG weights root not found at {}",
            weights_root.display()
        );
        return Ok(());
    }

    let reference = HookReference::load(reference_path.as_path())?;
    let pixel_values = reference
        .get_input("input.pixel_values")
        .ok_or("missing input.pixel_values in reference")?;
    let image_embeds_ref = reference
        .get_input("input.image_embeds")
        .ok_or("missing input.image_embeds in reference")?;

    let dino_weights = weights_root.join("image_encoder_dinov2/model.safetensors");
    if !dino_weights.exists() {
        eprintln!(
            "skipping: DINO weights file not found at {}",
            dino_weights.display()
        );
        return Ok(());
    }

    let device = Default::default();
    let pixel_tensor = tensor_from_data_4d::<burn::backend::NdArray<f32>>(&pixel_values, &device)?;

    let encoder_bpk = load_triposg_dinov2::<burn::backend::NdArray<f32>>(&device, &dino_weights)?;
    let encoder_safe = load_triposg_dinov2_from_safetensors::<burn::backend::NdArray<f32>>(
        &device,
        &dino_weights,
    )?;

    let embeds_bpk = encoder_bpk.forward(pixel_tensor.clone());
    let embeds_safe = encoder_safe.forward(pixel_tensor);

    let bpk_vs_ref = compute_stats_from_tensor(&embeds_bpk, &image_embeds_ref)?;
    let safe_vs_ref = compute_stats_from_tensor(&embeds_safe, &image_embeds_ref)?;
    let bpk_vs_safe = compute_stats_tensors(&embeds_bpk, &embeds_safe)?;
    let bpk_data = tensor_to_vec3(&embeds_bpk)?;
    let ref_data = &image_embeds_ref.data;
    let [batch, tokens, channels] = embeds_bpk.shape().dims();
    let token_stride = tokens * channels;
    let patch_count = tokens.saturating_sub(1);
    let cls_stats = compute_stats(&bpk_data[0..channels], &ref_data[0..channels]);
    let patch_stats = compute_stats(
        &bpk_data[channels..token_stride],
        &ref_data[channels..token_stride],
    );
    let cls_swapped_stats = if patch_count > 0 {
        compute_stats(&bpk_data[0..channels], &ref_data[channels..(channels * 2)])
    } else {
        MetricStats {
            mean_abs: 0.0,
            max_abs: 0.0,
            mse: 0.0,
        }
    };

    println!(
        "dino_loader_parity: bpk_vs_ref mean_abs={:.6} max_abs={:.6} mse={:.6}",
        bpk_vs_ref.mean_abs, bpk_vs_ref.max_abs, bpk_vs_ref.mse
    );
    println!(
        "dino_loader_parity: safe_vs_ref mean_abs={:.6} max_abs={:.6} mse={:.6}",
        safe_vs_ref.mean_abs, safe_vs_ref.max_abs, safe_vs_ref.mse
    );
    println!(
        "dino_loader_parity: bpk_vs_safe mean_abs={:.6} max_abs={:.6} mse={:.6}",
        bpk_vs_safe.mean_abs, bpk_vs_safe.max_abs, bpk_vs_safe.mse
    );
    println!(
        "dino_loader_parity: cls_vs_ref mean_abs={:.6} max_abs={:.6} mse={:.6}",
        cls_stats.mean_abs, cls_stats.max_abs, cls_stats.mse
    );
    println!(
        "dino_loader_parity: patch_vs_ref mean_abs={:.6} max_abs={:.6} mse={:.6}",
        patch_stats.mean_abs, patch_stats.max_abs, patch_stats.mse
    );
    println!(
        "dino_loader_parity: cls_vs_first_patch_ref mean_abs={:.6} max_abs={:.6} mse={:.6}",
        cls_swapped_stats.mean_abs, cls_swapped_stats.max_abs, cls_swapped_stats.mse
    );
    println!(
        "dino_loader_parity: shape batch={} tokens={} channels={}",
        batch, tokens, channels
    );

    assert!(
        bpk_vs_safe.mean_abs <= 1e-4 && bpk_vs_safe.max_abs <= 5e-3,
        "burnpack and safetensors loaders diverged too much: mean_abs={:.6} max_abs={:.6}",
        bpk_vs_safe.mean_abs,
        bpk_vs_safe.max_abs
    );

    Ok(())
}

fn asset_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(relative)
}

fn resolve_weights_root() -> PathBuf {
    if let Ok(root) = std::env::var("TRIPOSG_WEIGHTS_ROOT") {
        let path = PathBuf::from(root);
        if path.exists() {
            return path;
        }
    }
    let path = PathBuf::from(TRIPOSG_ROOT);
    if path.exists() {
        return path;
    }
    asset_path("assets/models/MIDI-3D")
}

#[derive(Clone)]
struct HookTensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

struct HookReference {
    tensors: BTreeMap<String, HookTensor>,
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

fn tensor_from_data_4d<B: Backend>(
    tensor: &HookTensor,
    device: &B::Device,
) -> Result<Tensor<B, 4>, Box<dyn std::error::Error>> {
    let shape: [usize; 4] = tensor
        .shape
        .clone()
        .try_into()
        .map_err(|_| "unexpected tensor rank")?;
    let data = Tensor::<B, 1>::from_floats(tensor.data.as_slice(), device);
    Ok(data.reshape([
        shape[0] as i32,
        shape[1] as i32,
        shape[2] as i32,
        shape[3] as i32,
    ]))
}

#[derive(Debug)]
struct MetricStats {
    mean_abs: f32,
    max_abs: f32,
    mse: f32,
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

fn compute_stats_from_tensor<B: Backend>(
    tensor: &Tensor<B, 3>,
    reference: &HookTensor,
) -> Result<MetricStats, Box<dyn std::error::Error>> {
    let data = tensor
        .clone()
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|_| "failed to convert tensor")?;
    Ok(compute_stats(&data, &reference.data))
}

fn compute_stats_tensors<B: Backend>(
    lhs: &Tensor<B, 3>,
    rhs: &Tensor<B, 3>,
) -> Result<MetricStats, Box<dyn std::error::Error>> {
    let lhs_data = tensor_to_vec3(lhs)?;
    let rhs_data = tensor_to_vec3(rhs)?;
    Ok(compute_stats(&lhs_data, &rhs_data))
}

fn tensor_to_vec3<B: Backend>(
    tensor: &Tensor<B, 3>,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    tensor
        .clone()
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|_| "failed to convert tensor".into())
}

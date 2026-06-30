#![cfg(feature = "import")]

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use burn::prelude::*;
use safetensors::tensor::{SafeTensors, TensorView};

use burn_tripo::pipeline::triposg::TripoSGPipeline;

const TRIPOSG_ROOT: &str = r"E:\repos\TripoSG\pretrained_weights\TripoSG";

const EMBED_MAX_ABS: f32 = 1e-3;
const EMBED_MEAN_ABS: f32 = 1e-4;
const EMBED_MSE: f32 = 1e-6;

const LATENT_MAX_ABS: f32 = 1e-3;
const LATENT_MEAN_ABS: f32 = 1e-4;
const LATENT_MSE: f32 = 1e-6;

const GRID_MAX_ABS: f32 = 1e-3;
const GRID_MEAN_ABS: f32 = 1e-4;
const GRID_MSE: f32 = 1e-6;

#[test]
fn triposg_pipeline_from_reference_embeds_matches_reference_strictly()
-> Result<(), Box<dyn std::error::Error>> {
    if std::env::var("TRIPOSG_FULL_REFERENCE").is_err() {
        eprintln!(
            "skipping: set TRIPOSG_FULL_REFERENCE=1 to run full TripoSG reference embed test"
        );
        return Ok(());
    }

    let reference_path = asset_path("assets/hooks/triposg_pipeline_reference.safetensors");
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
    let device = Default::default();

    let pixel_values = reference
        .get_input("input.pixel_values")
        .ok_or("missing input.pixel_values in reference")?;
    let image_embeds_ref = reference
        .get_input("input.image_embeds")
        .ok_or("missing input.image_embeds in reference")?;
    let input_latents = reference
        .get_input("input.latents")
        .ok_or("missing input.latents in reference")?;
    let output_latents = reference
        .get_input("output.latents")
        .ok_or("missing output.latents in reference")?;
    let output_grid = reference
        .get_input("output.grid_logits")
        .ok_or("missing output.grid_logits in reference")?;

    let num_steps = reference
        .get_scalar("meta.num_steps")
        .ok_or("missing meta.num_steps in reference")? as usize;
    let num_tokens = reference
        .get_scalar("meta.num_tokens")
        .ok_or("missing meta.num_tokens in reference")? as usize;
    let guidance_scale = reference
        .get_scalar("meta.guidance_scale")
        .ok_or("missing meta.guidance_scale in reference")?;
    let resolution = reference
        .get_scalar("meta.resolution")
        .ok_or("missing meta.resolution in reference")? as usize;
    let chunk_size = reference
        .get_scalar("meta.chunk_size")
        .ok_or("missing meta.chunk_size in reference")? as usize;
    let bounds = reference
        .get_vector("meta.bounds")
        .ok_or("missing meta.bounds in reference")?;
    let bounds = [
        bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5],
    ];

    let pixel_tensor = tensor_from_data_4d::<burn::backend::NdArray<f32>>(&pixel_values, &device)?;
    let image_embeds_tensor =
        tensor_from_data_3d::<burn::backend::NdArray<f32>>(&image_embeds_ref, &device)?;
    let input_latents_tensor =
        tensor_from_data_3d::<burn::backend::NdArray<f32>>(&input_latents, &device)?;

    let mut pipeline = TripoSGPipeline::from_pretrained(weights_root, &device)?;

    let embeds_from_pixels = pipeline
        .image_encoder
        .as_ref()
        .expect("TripoSG image encoder unavailable")
        .forward(pixel_tensor);
    let stats = compute_stats_from_tensor(&embeds_from_pixels, &image_embeds_ref)?;
    assert_within(
        "image_embeds_from_pixels",
        &stats,
        EMBED_MAX_ABS,
        EMBED_MEAN_ABS,
        EMBED_MSE,
    )?;

    let output = pipeline.sample_from_embeds(
        image_embeds_tensor,
        1,
        num_steps,
        num_tokens,
        guidance_scale,
        None,
        Some(input_latents_tensor),
    );
    let stats = compute_stats_from_tensor(&output.latents, &output_latents)?;
    assert_within(
        "pipeline.latents.from_reference_embeds",
        &stats,
        LATENT_MAX_ABS,
        LATENT_MEAN_ABS,
        LATENT_MSE,
    )?;

    let grid = pipeline.decode_grid(&output.latents, bounds, resolution, chunk_size)?;
    let stats = compute_stats(&grid.values, &output_grid.data);
    println!(
        "decoder.grid_logits.from_reference_embeds mean_abs={:.6} max_abs={:.6} mse={:.6}",
        stats.mean_abs, stats.max_abs, stats.mse
    );

    let output_latents_tensor =
        tensor_from_data_3d::<burn::backend::NdArray<f32>>(&output_latents, &device)?;
    let grid_from_reference_latents =
        pipeline.decode_grid(&output_latents_tensor, bounds, resolution, chunk_size)?;
    let stats_from_reference_latents =
        compute_stats(&grid_from_reference_latents.values, &output_grid.data);
    println!(
        "decoder.grid_logits.from_reference_latents mean_abs={:.6} max_abs={:.6} mse={:.6}",
        stats_from_reference_latents.mean_abs,
        stats_from_reference_latents.max_abs,
        stats_from_reference_latents.mse
    );

    assert_within(
        "decoder.grid_logits.from_reference_embeds",
        &stats,
        GRID_MAX_ABS,
        GRID_MEAN_ABS,
        GRID_MSE,
    )?;

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

    fn get_scalar(&self, name: &str) -> Option<f32> {
        self.tensors
            .get(name)
            .and_then(|tensor| tensor.data.first().copied())
    }

    fn get_vector(&self, name: &str) -> Option<Vec<f32>> {
        self.tensors.get(name).map(|tensor| tensor.data.clone())
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

fn tensor_from_data_3d<B: Backend>(
    tensor: &HookTensor,
    device: &B::Device,
) -> Result<Tensor<B, 3>, Box<dyn std::error::Error>> {
    let shape: [usize; 3] = tensor
        .shape
        .clone()
        .try_into()
        .map_err(|_| "unexpected tensor rank")?;
    let data = Tensor::<B, 1>::from_floats(tensor.data.as_slice(), device);
    Ok(data.reshape([shape[0] as i32, shape[1] as i32, shape[2] as i32]))
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

fn assert_within(
    label: &str,
    stats: &MetricStats,
    max_abs_limit: f32,
    mean_abs_limit: f32,
    mse_limit: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    if stats.max_abs > max_abs_limit || stats.mean_abs > mean_abs_limit || stats.mse > mse_limit {
        return Err(format!(
            "{label} out of tolerance: mean_abs={:.6} max_abs={:.6} mse={:.6}",
            stats.mean_abs, stats.max_abs, stats.mse
        )
        .into());
    }
    Ok(())
}

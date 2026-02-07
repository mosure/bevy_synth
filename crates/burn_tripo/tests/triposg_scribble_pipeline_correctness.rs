#![cfg(feature = "import")]

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use burn::prelude::*;
use safetensors::tensor::{SafeTensors, TensorView};

use burn_foreground::pipeline::{PrepareImageConfig, RmbgPipeline, prepare_image_data};
use burn_tripo::pipeline::{
    mesh::{DenseGrid, Mesh as TripoMesh, grid_to_mesh},
    triposg_scribble::TripoSGScribblePipeline,
};

const TRIPOSG_SCRIBBLE_ROOT: &str = r"E:\repos\TripoSG\pretrained_weights\TripoSG-scribble";
const RMBG_ROOT: &str = r"E:\repos\TripoSG\pretrained_weights\RMBG-1.4";
const INPUT_IMAGE: &str = r"F:\repos\TRELLIS\assets\nano_banana\chair\chair_0.jpg";

const MAX_ABS: f32 = 5e-2;
const MEAN_ABS: f32 = 5e-3;
const MSE: f32 = 5e-3;
const GRID_MAX_ABS: f32 = 1e-1;

#[test]
fn triposg_scribble_pipeline_matches_reference() -> Result<(), Box<dyn std::error::Error>> {
    let reference_path = asset_path("assets/hooks/triposg_scribble_pipeline_reference.safetensors");
    if !reference_path.exists() {
        eprintln!(
            "skipping: scribble pipeline reference file not found at {}",
            reference_path.display()
        );
        return Ok(());
    }

    let weights_root = resolve_weights_root();
    if !weights_root.exists() {
        eprintln!(
            "skipping: TripoSG-scribble weights root not found at {}",
            weights_root.display()
        );
        return Ok(());
    }

    let reference = HookReference::load(reference_path.as_path())?;
    let device = Default::default();

    let input_image = reference
        .get_input("input.image")
        .ok_or("missing input.image in reference")?;
    let input_latents = reference
        .get_input("input.latents")
        .ok_or("missing input.latents in reference")?;
    let input_text_embeds = reference
        .get_input("input.text_embeds")
        .ok_or("missing input.text_embeds in reference")?;
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

    let input_image = tensor_from_data_4d::<burn::backend::NdArray<f32>>(&input_image, &device)?;
    let input_latents =
        tensor_from_data_3d::<burn::backend::NdArray<f32>>(&input_latents, &device)?;
    let text_embeds =
        tensor_from_data_3d::<burn::backend::NdArray<f32>>(&input_text_embeds, &device)?;

    if Path::new(RMBG_ROOT).exists() && Path::new(INPUT_IMAGE).exists() {
        let rmbg = RmbgPipeline::from_pretrained(RMBG_ROOT, &device)?;
        let prepared = prepare_image_data::<burn::backend::NdArray<f32>>(
            Path::new(INPUT_IMAGE),
            Some(&rmbg),
            &PrepareImageConfig::default(),
        )?;
        let input_data = input_image
            .clone()
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .map_err(|_| "failed to read input.image")?;
        if prepared.data.len() != input_data.len() {
            return Err(format!(
                "prepared image shape mismatch: {} values vs {}",
                prepared.data.len(),
                input_data.len()
            )
            .into());
        }
        let stats = compute_stats(&prepared.data, &input_data);
        if stats.max_abs > MAX_ABS || stats.mean_abs > MEAN_ABS || stats.mse > MSE {
            return Err(format!(
                "prepared image out of tolerance: mean_abs={:.6} max_abs={:.6} mse={:.6}",
                stats.mean_abs, stats.max_abs, stats.mse
            )
            .into());
        }
    }

    let mut pipeline = TripoSGScribblePipeline::from_pretrained(weights_root, &device)?;
    let image_embeds = pipeline.encode_image(input_image.clone());

    if let Some(reference_embeds) = reference.get_input("input.image_embeds") {
        let stats = compute_stats_from_tensor(&image_embeds, &reference_embeds)?;
        if stats.max_abs > MAX_ABS || stats.mean_abs > MEAN_ABS || stats.mse > MSE {
            return Err(format!(
                "image embeds out of tolerance: mean_abs={:.6} max_abs={:.6} mse={:.6}",
                stats.mean_abs, stats.max_abs, stats.mse
            )
            .into());
        }
    }

    if let (Some(step0_noise_ref), Some(step0_latents_ref)) = (
        reference.get_input("output.noise_pred.step0"),
        reference.get_input("output.latents.step0"),
    ) {
        let do_guidance = guidance_scale > 1.0;
        let guided_text = if do_guidance {
            let zeros =
                Tensor::<burn::backend::NdArray<f32>, 3>::zeros(text_embeds.shape(), &device);
            Tensor::cat(vec![zeros, text_embeds.clone()], 0)
        } else {
            text_embeds.clone()
        };
        let guided_image = if do_guidance {
            let zeros =
                Tensor::<burn::backend::NdArray<f32>, 3>::zeros(image_embeds.shape(), &device);
            Tensor::cat(vec![zeros, image_embeds.clone()], 0)
        } else {
            image_embeds.clone()
        };

        pipeline
            .scheduler
            .set_timesteps(num_steps, None, None, None)
            .map_err(|err| format!("failed to set timesteps: {err}"))?;

        let mut latents = input_latents.clone();
        let timesteps = pipeline.scheduler.timesteps().to_vec();
        if let Some(&t) = timesteps.first() {
            let latent_model_input = if do_guidance {
                Tensor::cat(vec![latents.clone(), latents.clone()], 0)
            } else {
                latents.clone()
            };
            let model_batch = latent_model_input.shape().dims::<3>()[0];
            let timestep = Tensor::<burn::backend::NdArray<f32>, 1>::from_floats(
                vec![t; model_batch].as_slice(),
                &device,
            );

            let mut noise_pred = pipeline.transformer.forward(
                latent_model_input,
                timestep,
                guided_text,
                Some(guided_image),
                None,
            );

            if do_guidance {
                let half = model_batch / 2;
                let channels = pipeline.transformer.config().in_channels;
                let noise_uncond = noise_pred
                    .clone()
                    .slice([0..half, 0..num_tokens, 0..channels]);
                let noise_cond = noise_pred.slice([half..(half * 2), 0..num_tokens, 0..channels]);
                noise_pred =
                    noise_uncond.clone() + (noise_cond - noise_uncond).mul_scalar(guidance_scale);
            }

            let stats = compute_stats_from_tensor(&noise_pred, &step0_noise_ref)?;
            if stats.max_abs > MAX_ABS || stats.mean_abs > MEAN_ABS || stats.mse > MSE {
                return Err(format!(
                    "step0 noise_pred out of tolerance: mean_abs={:.6} max_abs={:.6} mse={:.6}",
                    stats.mean_abs, stats.max_abs, stats.mse
                )
                .into());
            }

            latents = pipeline.scheduler.step(noise_pred, t, latents);
            let stats = compute_stats_from_tensor(&latents, &step0_latents_ref)?;
            if stats.max_abs > MAX_ABS || stats.mean_abs > MEAN_ABS || stats.mse > MSE {
                return Err(format!(
                    "step0 latents out of tolerance: mean_abs={:.6} max_abs={:.6} mse={:.6}",
                    stats.mean_abs, stats.max_abs, stats.mse
                )
                .into());
            }
        }
    }

    let output = pipeline.sample_with_embeddings(
        text_embeds,
        image_embeds,
        num_steps,
        num_tokens,
        guidance_scale,
        None,
        Some(input_latents),
    );

    let stats = compute_stats_from_tensor(&output.latents, &output_latents)?;
    if stats.max_abs > MAX_ABS || stats.mean_abs > MEAN_ABS || stats.mse > MSE {
        return Err(format!(
            "latent output out of tolerance: mean_abs={:.6} max_abs={:.6} mse={:.6}",
            stats.mean_abs, stats.max_abs, stats.mse
        )
        .into());
    }

    let grid = pipeline.decode_grid(output.latents, bounds, resolution, chunk_size)?;

    let stats = compute_stats(&grid.values, &output_grid.data);
    if stats.max_abs > GRID_MAX_ABS || stats.mean_abs > MEAN_ABS || stats.mse > MSE {
        return Err(format!(
            "grid logits out of tolerance: mean_abs={:.6} max_abs={:.6} mse={:.6}",
            stats.mean_abs, stats.max_abs, stats.mse
        )
        .into());
    }

    let reference_grid = DenseGrid {
        values: output_grid.data.clone(),
        size: [resolution, resolution, resolution],
        bounds,
    };
    compare_meshes(
        &grid_to_mesh(&grid, 0.0),
        &grid_to_mesh(&reference_grid, 0.0),
    )?;

    Ok(())
}

fn asset_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(relative)
}

fn resolve_weights_root() -> std::path::PathBuf {
    if let Ok(root) = std::env::var("TRIPOSG_SCRIBBLE_WEIGHTS_ROOT") {
        let path = std::path::PathBuf::from(root);
        if path.exists() {
            return path;
        }
    }
    let candidate = std::path::PathBuf::from(TRIPOSG_SCRIBBLE_ROOT);
    if candidate.exists() {
        return candidate;
    }
    asset_path("assets/models/TripoSG-scribble")
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
        .map_err(|_| "unexpected input rank")?;
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

fn compute_stats_from_tensor<B: Backend>(
    burn_tensor: &Tensor<B, 3>,
    reference: &HookTensor,
) -> Result<MetricStats, Box<dyn std::error::Error>> {
    let data = burn_tensor
        .clone()
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|err| format!("failed to convert burn tensor: {err:?}"))?;
    Ok(compute_stats(&data, &reference.data))
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

fn compare_meshes(
    mesh: &Option<TripoMesh>,
    reference: &Option<TripoMesh>,
) -> Result<(), Box<dyn std::error::Error>> {
    match (mesh, reference) {
        (None, None) => Ok(()),
        (Some(_), None) | (None, Some(_)) => {
            Err("mesh extraction mismatch: reference/actual presence differs".into())
        }
        (Some(mesh), Some(reference)) => {
            if mesh.vertices.len() != reference.vertices.len() {
                return Err(format!(
                    "mesh vertex count mismatch: {} vs {}",
                    mesh.vertices.len(),
                    reference.vertices.len()
                )
                .into());
            }
            if mesh.faces.len() != reference.faces.len() {
                return Err(format!(
                    "mesh face count mismatch: {} vs {}",
                    mesh.faces.len(),
                    reference.faces.len()
                )
                .into());
            }

            let (min_a, max_a) = mesh_bounds(mesh);
            let (min_b, max_b) = mesh_bounds(reference);
            let tol = 1e-2;
            for i in 0..3 {
                if (min_a[i] - min_b[i]).abs() > tol || (max_a[i] - max_b[i]).abs() > tol {
                    return Err(format!(
                        "mesh bounds mismatch on axis {}: [{}, {}] vs [{}, {}]",
                        i, min_a[i], max_a[i], min_b[i], max_b[i]
                    )
                    .into());
                }
            }
            Ok(())
        }
    }
}

fn mesh_bounds(mesh: &TripoMesh) -> ([f32; 3], [f32; 3]) {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for v in &mesh.vertices {
        for i in 0..3 {
            min[i] = min[i].min(v[i]);
            max[i] = max[i].max(v[i]);
        }
    }
    (min, max)
}

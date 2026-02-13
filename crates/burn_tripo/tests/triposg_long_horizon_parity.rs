#![cfg(feature = "import")]

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use burn::prelude::*;
use burn_tripo::model::triposg::{
    dit::{TripoSGDiT, TripoSGDiTConfig, import::load_triposg_dit_from_safetensors},
    load_policy::BurnpackLoadPolicy,
    scheduler::RectifiedFlowSchedulerConfig,
};
use burn_tripo::model::triposg::dit::import::load_triposg_dit_with_policy;
use safetensors::tensor::{SafeTensors, TensorView};

const TRIPOSG_ROOT: &str = r"E:\repos\TripoSG\pretrained_weights\TripoSG";
const DEFAULT_REF: &str = r"F:\repos\burn_3d_synth\tmp\triposg_chair_ref_50_all.safetensors";

#[test]
fn triposg_long_horizon_reports_bpk_vs_safetensors() -> Result<(), Box<dyn std::error::Error>> {
    let reference_path = resolve_reference_path();
    if !reference_path.exists() {
        eprintln!(
            "skipping: long-horizon reference not found at {}",
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
    let image_embeds_ref = reference
        .get_input("input.image_embeds")
        .ok_or("missing input.image_embeds in reference")?;
    let input_latents_ref = reference
        .get_input("input.latents")
        .ok_or("missing input.latents in reference")?;
    let step0_noise_ref = reference
        .get_input("output.noise_pred.step0")
        .ok_or("missing output.noise_pred.step0 in reference")?;
    let step0_latents_ref = reference
        .get_input("output.latents.step0")
        .ok_or("missing output.latents.step0 in reference")?;
    let step1_noise_ref = reference
        .get_input("output.noise_pred.step1")
        .ok_or("missing output.noise_pred.step1 in reference")?;
    let step1_latents_ref = reference
        .get_input("output.latents.step1")
        .ok_or("missing output.latents.step1 in reference")?;
    let num_steps = reference
        .get_scalar("meta.num_steps")
        .ok_or("missing meta.num_steps in reference")? as usize;
    let num_tokens = reference
        .get_scalar("meta.num_tokens")
        .ok_or("missing meta.num_tokens in reference")? as usize;
    let guidance_scale = reference
        .get_scalar("meta.guidance_scale")
        .ok_or("missing meta.guidance_scale in reference")?;

    let device = Default::default();
    let image_embeds =
        tensor_from_data_3d::<burn::backend::NdArray<f32>>(&image_embeds_ref, &device)?;
    let input_latents =
        tensor_from_data_3d::<burn::backend::NdArray<f32>>(&input_latents_ref, &device)?;

    let run_mode = std::env::var("TRIPOSG_LONG_MODE").unwrap_or_else(|_| "bpk".to_string());
    let use_safetensors = run_mode.eq_ignore_ascii_case("safe")
        || run_mode.eq_ignore_ascii_case("safetensors");
    let run_label = if use_safetensors { "safe" } else { "bpk" };
    let out = run_denoise_only(
        &weights_root,
        &device,
        use_safetensors,
        image_embeds,
        input_latents,
        num_steps,
        num_tokens,
        guidance_scale,
    )?;

    let step0_noise_stats = compute_stats_from_tensor(&out.step0_noise, &step0_noise_ref)?;
    let step0_latents_stats = compute_stats_from_tensor(&out.step0_latents, &step0_latents_ref)?;
    let step1_noise_stats = compute_stats_from_tensor(&out.step1_noise, &step1_noise_ref)?;
    let step1_latents_stats = compute_stats_from_tensor(&out.step1_latents, &step1_latents_ref)?;

    println!(
        "long_horizon.{run_label}.step0_noise_vs_ref mean_abs={:.6} max_abs={:.6} mse={:.6}",
        step0_noise_stats.mean_abs, step0_noise_stats.max_abs, step0_noise_stats.mse
    );
    println!(
        "long_horizon.{run_label}.step0_latents_vs_ref mean_abs={:.6} max_abs={:.6} mse={:.6}",
        step0_latents_stats.mean_abs, step0_latents_stats.max_abs, step0_latents_stats.mse
    );
    println!(
        "long_horizon.{run_label}.step1_noise_vs_ref mean_abs={:.6} max_abs={:.6} mse={:.6}",
        step1_noise_stats.mean_abs, step1_noise_stats.max_abs, step1_noise_stats.mse
    );
    println!(
        "long_horizon.{run_label}.step1_latents_vs_ref mean_abs={:.6} max_abs={:.6} mse={:.6}",
        step1_latents_stats.mean_abs, step1_latents_stats.max_abs, step1_latents_stats.mse
    );

    Ok(())
}

struct DenoiseRun<B: Backend> {
    step0_noise: Tensor<B, 3>,
    step0_latents: Tensor<B, 3>,
    step1_noise: Tensor<B, 3>,
    step1_latents: Tensor<B, 3>,
}

#[allow(clippy::too_many_arguments)]
fn run_denoise_only<B: Backend>(
    weights_root: &Path,
    device: &B::Device,
    use_safetensors: bool,
    image_embeds: Tensor<B, 3>,
    input_latents: Tensor<B, 3>,
    num_steps: usize,
    num_tokens: usize,
    guidance_scale: f32,
) -> Result<DenoiseRun<B>, Box<dyn std::error::Error>> {
    let dit_cfg = TripoSGDiTConfig::from_config_file(weights_root.join("transformer/config.json"))
        .unwrap_or_else(|_| TripoSGDiTConfig::triposg_pretrained());
    let dit_path = weights_root.join("transformer/diffusion_pytorch_model.safetensors");
    let transformer = if use_safetensors {
        load_triposg_dit_from_safetensors::<B>(&dit_cfg, device, &dit_path)?
    } else {
        load_triposg_dit_with_policy::<B>(
            &dit_cfg,
            device,
            &dit_path,
            BurnpackLoadPolicy::default(),
        )?
    };

    run_denoise_loop(
        transformer,
        weights_root,
        image_embeds,
        input_latents,
        num_steps,
        num_tokens,
        guidance_scale,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_denoise_loop<B: Backend>(
    transformer: TripoSGDiT<B>,
    weights_root: &Path,
    image_embeds: Tensor<B, 3>,
    input_latents: Tensor<B, 3>,
    num_steps: usize,
    num_tokens: usize,
    guidance_scale: f32,
) -> Result<DenoiseRun<B>, Box<dyn std::error::Error>> {
    let scheduler_cfg = RectifiedFlowSchedulerConfig::from_config_file(
        weights_root.join("scheduler/scheduler_config.json"),
    )
    .unwrap_or_else(|_| RectifiedFlowSchedulerConfig::midi_3d());
    let mut scheduler = scheduler_cfg.init();
    scheduler
        .set_timesteps(num_steps, None, None, None)
        .map_err(|err| format!("failed to set timesteps: {err}"))?;

    let device = input_latents.device();
    let do_guidance = guidance_scale > 1.0;
    let conditioned_embeds = if do_guidance {
        let zeros = Tensor::<B, 3>::zeros(image_embeds.shape(), &device);
        Tensor::cat(vec![zeros, image_embeds], 0)
    } else {
        image_embeds
    };

    let timesteps = scheduler.timesteps().to_vec();
    if timesteps.len() < 2 {
        return Err("need at least 2 timesteps for this parity test".into());
    }
    let mut latents = input_latents;
    let mut step0_noise = None;
    let mut step0_latents = None;
    let mut step1_noise = None;
    let mut step1_latents = None;
    let channels = transformer.config().in_channels;
    for (step_idx, &t) in timesteps.iter().take(2).enumerate() {
        let latent_model_input = if do_guidance {
            Tensor::cat(vec![latents.clone(), latents.clone()], 0)
        } else {
            latents.clone()
        };
        let model_batch = latent_model_input.shape().dims::<3>()[0];
        let timestep = Tensor::<B, 1>::full([model_batch], t, &device);
        let mut noise_pred = transformer.forward(
            latent_model_input,
            timestep,
            conditioned_embeds.clone(),
            None,
            None,
        );

        if do_guidance {
            let half = model_batch / 2;
            let noise_uncond = noise_pred
                .clone()
                .slice([0..half, 0..num_tokens, 0..channels]);
            let noise_cond = noise_pred.slice([half..(half * 2), 0..num_tokens, 0..channels]);
            noise_pred =
                noise_uncond.clone() + (noise_cond - noise_uncond).mul_scalar(guidance_scale);
        }

        if step_idx == 0 {
            step0_noise = Some(noise_pred.clone());
        } else if step_idx == 1 {
            step1_noise = Some(noise_pred.clone());
        }
        latents = scheduler.step(noise_pred, t, latents);
        if step_idx == 0 {
            step0_latents = Some(latents.clone());
        } else if step_idx == 1 {
            step1_latents = Some(latents.clone());
        }
    }

    let step0_noise = step0_noise.ok_or("no denoise step executed")?;
    let step0_latents = step0_latents.ok_or("missing step0 latents")?;
    let step1_noise = step1_noise.ok_or("missing step1 noise")?;
    let step1_latents = step1_latents.ok_or("missing step1 latents")?;
    Ok(DenoiseRun {
        step0_noise,
        step0_latents,
        step1_noise,
        step1_latents,
    })
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
}

fn tensor_view_to_vec(view: &TensorView<'_>) -> Vec<f32> {
    view.data()
        .chunks_exact(4)
        .map(|chunk| {
            let bytes: [u8; 4] = chunk.try_into().expect("valid f32 bytes");
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

#[derive(Debug, Clone, Copy)]
struct MetricStats {
    mean_abs: f32,
    max_abs: f32,
    mse: f32,
}

fn compute_stats(lhs: &[f32], rhs: &[f32]) -> MetricStats {
    let mut sum_abs = 0.0f32;
    let mut max_abs = 0.0f32;
    let mut sum_sq = 0.0f32;
    for (&a, &b) in lhs.iter().zip(rhs.iter()) {
        let d = a - b;
        let abs = d.abs();
        sum_abs += abs;
        max_abs = max_abs.max(abs);
        sum_sq += d * d;
    }
    let n = lhs.len().max(1) as f32;
    MetricStats {
        mean_abs: sum_abs / n,
        max_abs,
        mse: sum_sq / n,
    }
}

fn compute_stats_from_tensor<B: Backend>(
    tensor: &Tensor<B, 3>,
    reference: &HookTensor,
) -> Result<MetricStats, Box<dyn std::error::Error>> {
    let values = tensor
        .clone()
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|_| "failed to convert tensor to vec")?;
    Ok(compute_stats(&values, &reference.data))
}

fn resolve_reference_path() -> PathBuf {
    if let Ok(path) = std::env::var("TRIPOSG_LONG_REF") {
        return PathBuf::from(path);
    }
    PathBuf::from(DEFAULT_REF)
}

fn resolve_weights_root() -> PathBuf {
    if let Ok(path) = std::env::var("TRIPOSG_WEIGHTS_ROOT") {
        let path = PathBuf::from(path);
        if path.exists() {
            return path;
        }
    }
    let path = PathBuf::from(TRIPOSG_ROOT);
    if path.exists() {
        return path;
    }
    Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/models/MIDI-3D")
}

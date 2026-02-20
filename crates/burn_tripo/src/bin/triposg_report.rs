#![recursion_limit = "256"]

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
    triposg::TripoSGPipeline,
};

type CpuBackend = burn::backend::NdArray<f32>;
type WgpuBackend = burn_wgpu::Wgpu<f32, i32, u32>;

const TRIPOSG_ROOT: &str = r"E:\repos\TripoSG\pretrained_weights\TripoSG";
const RMBG_ROOT: &str = r"E:\repos\TripoSG\pretrained_weights\RMBG-1.4";
const INPUT_IMAGE: &str = r"E:\repos\TripoSG\assets\example_data\hjswed.png";

fn main() {
    let handle = std::thread::Builder::new()
        .stack_size(512 * 1024 * 1024)
        .spawn(|| {
            if let Err(err) = run() {
                eprintln!("{err}");
                std::process::exit(1);
            }
        })
        .expect("failed to spawn report thread");
    match handle.join() {
        Ok(()) => {}
        Err(_) => {
            eprintln!("report thread panicked");
            std::process::exit(1);
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    unsafe {
        if std::env::var("RMBG_STRICT_INTERP").is_err() {
            std::env::set_var("RMBG_STRICT_INTERP", "1");
        }
        if std::env::var("DINO_STRICT_PREPROCESS").is_err() {
            std::env::set_var("DINO_STRICT_PREPROCESS", "1");
        }
    }

    let reference_path = std::env::var("TRIPOSG_REPORT_REFERENCE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| asset_path("assets/hooks/triposg_pipeline_reference.safetensors"));
    if !reference_path.exists() {
        return Err(format!("reference file not found at {}", reference_path.display()).into());
    }

    let weights_root = resolve_weights_root();
    if !weights_root.exists() {
        return Err(format!(
            "TripoSG weights root not found at {}",
            weights_root.display()
        )
        .into());
    }

    let reference = HookReference::load(reference_path.as_path())?;

    let backend = std::env::var("TRIPOSG_REPORT_BACKEND").unwrap_or_else(|_| "wgpu".to_string());
    match backend.as_str() {
        "cpu" => run_with_backend::<CpuBackend>(&reference, &weights_root, Default::default()),
        "wgpu" => {
            let result = std::panic::catch_unwind(|| {
                let device = burn_wgpu::WgpuDevice::default();
                run_with_backend::<WgpuBackend>(&reference, &weights_root, device)
            });
            match result {
                Ok(outcome) => outcome,
                Err(_) => {
                    eprintln!("triposg_report: wgpu backend unavailable, falling back to cpu");
                    run_with_backend::<CpuBackend>(&reference, &weights_root, Default::default())
                }
            }
        }
        other => Err(format!("unknown TRIPOSG_REPORT_BACKEND={other}; use cpu|wgpu").into()),
    }
}

fn run_with_backend<B: Backend>(
    reference: &HookReference,
    weights_root: &Path,
    device: B::Device,
) -> Result<(), Box<dyn std::error::Error>> {
    let input_image_hook = reference
        .get_input("input.image")
        .ok_or("missing input.image in reference")?;
    let input_latents_hook = reference
        .get_input("input.latents")
        .ok_or("missing input.latents in reference")?;
    let output_latents = reference
        .get_input("output.latents")
        .ok_or("missing output.latents in reference")?;
    let output_grid = reference.get_input("output.grid_logits");
    let skip_decode = std::env::var("TRIPOSG_REPORT_SKIP_DECODE").is_ok();
    if !skip_decode && output_grid.is_none() {
        return Err("missing output.grid_logits in reference".into());
    }

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

    let input_image = tensor_from_data_4d::<B>(&input_image_hook, &device)?;
    let input_latents = tensor_from_data_3d::<B>(&input_latents_hook, &device)?;

    println!("TripoSG E2E numerical report");
    println!("- input image: {INPUT_IMAGE}");
    println!("- num_steps: {num_steps}");
    println!("- num_tokens: {num_tokens}");
    println!("- guidance_scale: {guidance_scale}");
    println!("- resolution: {resolution}");
    println!("- chunk_size: {chunk_size}");
    println!(
        "- backend: {}",
        std::any::type_name::<B>()
            .split("::")
            .last()
            .unwrap_or("backend")
    );

    if Path::new(RMBG_ROOT).exists() && Path::new(INPUT_IMAGE).exists() {
        let rmbg = RmbgPipeline::from_pretrained(RMBG_ROOT, &device)?;
        let prepared = prepare_image_data::<B>(
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
        let ref_dims = input_image.shape().dims::<4>();
        let ref_height = ref_dims[2] as usize;
        let ref_width = ref_dims[3] as usize;
        let stats = compute_stats_overlap(
            &prepared.data,
            prepared.height,
            prepared.width,
            &input_data,
            ref_height,
            ref_width,
            3,
        );
        print_stats("prep.image", &stats);
    } else {
        println!("prep.image: skipped (missing RMBG or input image)");
    }

    let mut pipeline = TripoSGPipeline::from_pretrained(weights_root, &device)?;

    let processed = pipeline.image_processor.preprocess(input_image.clone());
    let mut reference_pixel_values = None;
    if let Some(reference_pixels) = reference.get_input("input.pixel_values") {
        let stats = compute_stats_from_tensor_4d(&processed, &reference_pixels)?;
        print_stats("encoder.pixel_values", &stats);
        reference_pixel_values = Some(reference_pixels);
    }
    let reference_tokens_pre_pos = reference.get_input("encoder.tokens_pre_pos");
    let reference_pos_interp = reference.get_input("encoder.pos_interp");
    let reference_hidden0 = reference.get_input("encoder.hidden.0");
    if reference_tokens_pre_pos.is_some()
        || reference_pos_interp.is_some()
        || reference_hidden0.is_some()
    {
        println!(
            "encoder.debug_hooks: skipped (requires burn_dino.patch forward_from_tokens/debug_embeddings)"
        );
    }

    let image_embeds = pipeline.encode_image(input_image.clone());
    let reference_embeds = reference.get_input("input.image_embeds");
    if let Some(reference_embeds) = reference_embeds.as_ref() {
        let stats = compute_stats_from_tensor(&image_embeds, reference_embeds)?;
        print_stats("encoder.image_embeds", &stats);

        if let Some(reference_pixels) = reference_pixel_values {
            let pixel_tensor = tensor_from_data_4d::<B>(&reference_pixels, &device)?;
            let embeds_from_pixels = pipeline
                .image_encoder
                .as_ref()
                .expect("TripoSG image encoder unavailable")
                .forward(pixel_tensor);
            let stats = compute_stats_from_tensor(&embeds_from_pixels, reference_embeds)?;
            print_stats("encoder.image_embeds_from_pixels", &stats);
        }
    }

    if std::env::var("TRIPOSG_REPORT_ENCODER_ONLY").is_ok() {
        return Ok(());
    }

    let do_guidance = guidance_scale > 1.0;
    let guided_embeds = if do_guidance {
        let zeros = Tensor::<B, 3>::zeros(image_embeds.shape(), &device);
        Tensor::cat(vec![zeros, image_embeds.clone()], 0)
    } else {
        image_embeds.clone()
    };

    pipeline
        .scheduler
        .set_timesteps(num_steps, None, None, None)
        .map_err(|err| format!("failed to set timesteps: {err}"))?;
    let timesteps = pipeline.scheduler.timesteps().to_vec();
    let report_all_steps = std::env::var("TRIPOSG_REPORT_ALL_STEPS").is_ok();

    if report_all_steps {
        let _ = report_denoise_steps(
            &mut pipeline,
            input_latents.clone(),
            guided_embeds.clone(),
            do_guidance,
            guidance_scale,
            num_tokens,
            timesteps.as_slice(),
            reference,
            "steps.pipeline_embeds",
        )?;

        if let Some(reference_embeds) = reference_embeds.as_ref() {
            let embeds_ref = tensor_from_data_3d::<B>(reference_embeds, &device)?;
            let guided_embeds_ref = if do_guidance {
                let zeros = Tensor::<B, 3>::zeros(embeds_ref.shape(), &device);
                Tensor::cat(vec![zeros, embeds_ref], 0)
            } else {
                embeds_ref
            };
            let _ = report_denoise_steps(
                &mut pipeline,
                input_latents.clone(),
                guided_embeds_ref,
                do_guidance,
                guidance_scale,
                num_tokens,
                timesteps.as_slice(),
                reference,
                "steps.reference_embeds",
            )?;
        }
    }

    if let (Some(step0_noise_ref), Some(step0_latents_ref)) = (
        reference.get_input("output.noise_pred.step0"),
        reference.get_input("output.latents.step0"),
    ) && let Some(&t0) = timesteps.first()
    {
        let latent_model_input = if do_guidance {
            Tensor::cat(vec![input_latents.clone(), input_latents.clone()], 0)
        } else {
            input_latents.clone()
        };
        let model_batch = latent_model_input.shape().dims::<3>()[0];
        let timestep = Tensor::<B, 1>::from_floats(vec![t0; model_batch].as_slice(), &device);
        let mut noise_pred = pipeline.transformer.forward(
            latent_model_input,
            timestep,
            guided_embeds.clone(),
            None,
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
        print_stats("transformer.noise_pred.step0", &stats);

        let latents_step0 = pipeline
            .scheduler
            .step(noise_pred, t0, input_latents.clone());
        let stats = compute_stats_from_tensor(&latents_step0, &step0_latents_ref)?;
        print_stats("scheduler.latents.step0", &stats);

        if let (Some(step1_noise_ref), Some(step1_latents_ref)) = (
            reference.get_input("output.noise_pred.step1"),
            reference.get_input("output.latents.step1"),
        ) && timesteps.len() > 1
        {
            let t1 = timesteps[1];
            let latent_model_input = if do_guidance {
                Tensor::cat(vec![latents_step0.clone(), latents_step0.clone()], 0)
            } else {
                latents_step0.clone()
            };
            let model_batch = latent_model_input.shape().dims::<3>()[0];
            let timestep = Tensor::<B, 1>::from_floats(vec![t1; model_batch].as_slice(), &device);
            let mut noise_pred = pipeline.transformer.forward(
                latent_model_input,
                timestep,
                guided_embeds,
                None,
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

            let stats = compute_stats_from_tensor(&noise_pred, &step1_noise_ref)?;
            print_stats("transformer.noise_pred.step1", &stats);

            let latents_step1 = pipeline.scheduler.step(noise_pred, t1, latents_step0);
            let stats = compute_stats_from_tensor(&latents_step1, &step1_latents_ref)?;
            print_stats("scheduler.latents.step1", &stats);
        }
    }

    if let (Some(reference_embeds), Some(step0_noise_ref), Some(step0_latents_ref)) = (
        reference_embeds.as_ref(),
        reference.get_input("output.noise_pred.step0"),
        reference.get_input("output.latents.step0"),
    ) && let Some(&t0) = timesteps.first()
    {
        pipeline
            .scheduler
            .set_timesteps(num_steps, None, None, None)
            .map_err(|err| format!("failed to reset timesteps: {err}"))?;
        let embeds_ref = tensor_from_data_3d::<B>(reference_embeds, &device)?;
        let guided_embeds_ref = if do_guidance {
            let zeros = Tensor::<B, 3>::zeros(embeds_ref.shape(), &device);
            Tensor::cat(vec![zeros, embeds_ref], 0)
        } else {
            embeds_ref
        };

        let latent_model_input = if do_guidance {
            Tensor::cat(vec![input_latents.clone(), input_latents.clone()], 0)
        } else {
            input_latents.clone()
        };
        let model_batch = latent_model_input.shape().dims::<3>()[0];
        let timestep = Tensor::<B, 1>::from_floats(vec![t0; model_batch].as_slice(), &device);
        let mut noise_pred = pipeline.transformer.forward(
            latent_model_input,
            timestep,
            guided_embeds_ref,
            None,
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
        print_stats("transformer.noise_pred.step0.from_reference_embeds", &stats);

        let latents_step0 = pipeline
            .scheduler
            .step(noise_pred, t0, input_latents.clone());
        let stats = compute_stats_from_tensor(&latents_step0, &step0_latents_ref)?;
        print_stats("scheduler.latents.step0.from_reference_embeds", &stats);
    }

    if std::env::var("TRIPOSG_REPORT_STEPS_ONLY").is_ok() {
        return Ok(());
    }

    if std::env::var("TRIPOSG_REPORT_REF_EMBEDS").is_ok()
        && let Some(reference_embeds) = reference_embeds.as_ref()
    {
        let embeds_tensor = tensor_from_data_3d::<B>(reference_embeds, &device)?;
        let output_ref = pipeline.sample_from_embeds(
            embeds_tensor,
            1,
            num_steps,
            num_tokens,
            guidance_scale,
            None,
            Some(input_latents.clone()),
        );

        let stats = compute_stats_from_tensor(&output_ref.latents, &output_latents)?;
        print_stats("pipeline.latents.from_reference_embeds", &stats);
        if !skip_decode {
            let output_grid = output_grid
                .as_ref()
                .expect("output.grid_logits checked above");
            let grid = pipeline.decode_grid(&output_ref.latents, bounds, resolution, chunk_size)?;
            let stats = compute_stats(&grid.values, &output_grid.data);
            print_stats("decoder.grid_logits.from_reference_embeds", &stats);
        }

        if std::env::var("TRIPOSG_REPORT_REF_EMBEDS_ONLY").is_ok() {
            return Ok(());
        }
    }

    if std::env::var("TRIPOSG_REPORT_CPU_DINO").is_ok()
        && std::any::type_name::<B>()
            .to_ascii_lowercase()
            .contains("wgpu")
    {
        let cpu_device = <CpuBackend as Backend>::Device::default();
        let cpu_pipeline =
            TripoSGPipeline::<CpuBackend>::from_pretrained(weights_root, &cpu_device)?;
        let cpu_image = tensor_from_data_4d::<CpuBackend>(&input_image_hook, &cpu_device)?;
        let cpu_embeds = cpu_pipeline.encode_image(cpu_image);
        let cpu_dims = cpu_embeds.shape().dims::<3>();
        let cpu_data = cpu_embeds
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .map_err(|_| "failed to read cpu image embeds")?;
        let embeds_tensor = Tensor::<B, 1>::from_floats(cpu_data.as_slice(), &device).reshape([
            cpu_dims[0] as i32,
            cpu_dims[1] as i32,
            cpu_dims[2] as i32,
        ]);

        let output_cpu = pipeline.sample_from_embeds(
            embeds_tensor,
            1,
            num_steps,
            num_tokens,
            guidance_scale,
            None,
            Some(input_latents.clone()),
        );

        let stats = compute_stats_from_tensor(&output_cpu.latents, &output_latents)?;
        print_stats("pipeline.latents.from_cpu_dino", &stats);
        if !skip_decode {
            let output_grid = output_grid
                .as_ref()
                .expect("output.grid_logits checked above");
            let grid = pipeline.decode_grid(&output_cpu.latents, bounds, resolution, chunk_size)?;
            let stats = compute_stats(&grid.values, &output_grid.data);
            print_stats("decoder.grid_logits.from_cpu_dino", &stats);
        }
    }

    let output = pipeline.sample(
        input_image.clone(),
        num_steps,
        num_tokens,
        guidance_scale,
        None,
        Some(input_latents),
    );

    let stats = compute_stats_from_tensor(&output.latents, &output_latents)?;
    print_stats("pipeline.latents", &stats);

    if skip_decode {
        return Ok(());
    }

    let output_grid = output_grid
        .as_ref()
        .expect("output.grid_logits checked above");
    let grid = pipeline.decode_grid(&output.latents, bounds, resolution, chunk_size)?;
    let stats = compute_stats(&grid.values, &output_grid.data);
    print_stats("decoder.grid_logits", &stats);

    let reference_grid = DenseGrid {
        values: output_grid.data.clone(),
        size: [resolution, resolution, resolution],
        bounds,
    };
    report_mesh(
        "mesh",
        &grid_to_mesh(&grid, 0.0),
        &grid_to_mesh(&reference_grid, 0.0),
    );

    Ok(())
}

fn print_stats(label: &str, stats: &MetricStats) {
    println!(
        "{label}: mean_abs={:.6} max_abs={:.6} mse={:.6}",
        stats.mean_abs, stats.max_abs, stats.mse
    );
}

#[allow(clippy::too_many_arguments)]
fn report_denoise_steps<B: Backend>(
    pipeline: &mut TripoSGPipeline<B>,
    latents_start: Tensor<B, 3>,
    conditioned_embeds: Tensor<B, 3>,
    do_guidance: bool,
    guidance_scale: f32,
    num_tokens: usize,
    timesteps: &[f32],
    reference: &HookReference,
    label_prefix: &str,
) -> Result<Tensor<B, 3>, Box<dyn std::error::Error>> {
    pipeline
        .scheduler
        .set_timesteps(timesteps.len(), None, None, None)
        .map_err(|err| format!("failed to set timesteps: {err}"))?;

    let device = latents_start.device();
    let mut latents = latents_start;
    let channels = pipeline.transformer.config().in_channels;

    for (step, &t) in timesteps.iter().enumerate() {
        let noise_key = format!("output.noise_pred.step{step}");
        let Some(noise_ref) = reference.get_input(&noise_key) else {
            break;
        };
        let latents_key = format!("output.latents.step{step}");
        let Some(latents_ref) = reference.get_input(&latents_key) else {
            break;
        };

        let latent_model_input = if do_guidance {
            Tensor::cat(vec![latents.clone(), latents.clone()], 0)
        } else {
            latents.clone()
        };
        let model_batch = latent_model_input.shape().dims::<3>()[0];
        let timestep = Tensor::<B, 1>::from_floats(vec![t; model_batch].as_slice(), &device);

        let mut noise_pred = pipeline.transformer.forward(
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

        let stats = compute_stats_from_tensor(&noise_pred, &noise_ref)?;
        print_stats(&format!("{label_prefix}.noise_pred.step{step}"), &stats);

        latents = pipeline.scheduler.step(noise_pred, t, latents);
        let stats = compute_stats_from_tensor(&latents, &latents_ref)?;
        print_stats(&format!("{label_prefix}.latents.step{step}"), &stats);
    }

    Ok(latents)
}

fn report_mesh(label: &str, mesh: &Option<TripoMesh>, reference: &Option<TripoMesh>) {
    match (mesh, reference) {
        (None, None) => println!("{label}: none (both missing)"),
        (Some(_), None) => println!("{label}: mismatch (reference missing, output present)"),
        (None, Some(_)) => println!("{label}: mismatch (reference present, output missing)"),
        (Some(mesh), Some(reference)) => {
            let (min_a, max_a) = mesh_bounds(mesh);
            let (min_b, max_b) = mesh_bounds(reference);
            println!(
                "{label}: vertices={} faces={} bounds_diff_min={:?} bounds_diff_max={:?}",
                mesh.vertices.len(),
                mesh.faces.len(),
                [
                    min_a[0] - min_b[0],
                    min_a[1] - min_b[1],
                    min_a[2] - min_b[2]
                ],
                [
                    max_a[0] - max_b[0],
                    max_a[1] - max_b[1],
                    max_a[2] - max_b[2]
                ]
            );
        }
    }
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

fn compute_stats_from_tensor_4d<B: Backend>(
    tensor: &Tensor<B, 4>,
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
                let diff = lhs[lhs_row + x] - rhs[rhs_row + x];
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

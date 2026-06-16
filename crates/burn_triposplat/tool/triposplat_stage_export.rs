use std::{borrow::Cow, fs, path::PathBuf};

use burn::{backend::NdArray, prelude::*, tensor::backend::BackendTypes};
use burn_triposplat::{
    CfgPredictionMode, ElasticGaussianFixedlenDecoderConfig,
    OctreeProbabilityFixedlenDecoderConfig, OctreeSample, TripoSplatBurnpackPrecision,
    TripoSplatOptions, TripoSplatPipeline, TripoSplatRuntimeComponents, normalize_num_gaussians,
};
use clap::{Parser, ValueEnum};
use safetensors::{
    SafeTensors,
    tensor::{Dtype, TensorView, View, serialize_to_file},
};

type ExportBackend = NdArray<f32>;
#[cfg(feature = "backend_wgpu")]
type WgpuExportBackend = burn::backend::Wgpu<f32, i32, u32>;

#[derive(Debug, Parser)]
#[command(about = "Export Rust TripoSplat stage tensors for upstream parity comparison.")]
struct Args {
    #[arg(long)]
    weights_root: PathBuf,

    #[arg(long, default_value = "f32")]
    precision: PrecisionArg,

    #[arg(long, value_enum, default_value_t = BackendArg::Ndarray)]
    backend: BackendArg,

    #[arg(long)]
    input_stages: PathBuf,

    #[arg(long)]
    output: PathBuf,

    #[arg(long)]
    splat_output: Option<PathBuf>,

    #[arg(long)]
    ply_output: Option<PathBuf>,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    #[arg(long, default_value_t = 20)]
    steps: usize,

    #[arg(long, default_value_t = 3.0)]
    guidance_scale: f32,

    #[arg(long, default_value_t = 3.0)]
    shift: f32,

    #[arg(long, default_value_t = 32768)]
    gaussians: usize,

    #[arg(long, value_enum, default_value_t = StopAfter::Encode)]
    stop_after: StopAfter,

    #[arg(long, default_value_t = false)]
    decode_reference_latent: bool,

    #[arg(long, default_value_t = false)]
    decode_reference_sample: bool,

    #[arg(long, default_value_t = false)]
    use_reference_decoder_features: bool,

    #[arg(long, default_value_t = 0)]
    trace_prefix_steps: usize,

    #[arg(long, default_value_t = false)]
    trace_only: bool,

    #[arg(long, value_enum, default_value_t = CfgModeArg::Batched)]
    cfg_mode: CfgModeArg,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum PrecisionArg {
    F32,
    F16,
}

impl From<PrecisionArg> for TripoSplatBurnpackPrecision {
    fn from(value: PrecisionArg) -> Self {
        match value {
            PrecisionArg::F32 => Self::F32,
            PrecisionArg::F16 => Self::F16,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum StopAfter {
    Encode,
    Sample,
    Decode,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum BackendArg {
    Ndarray,
    Wgpu,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CfgModeArg {
    Batched,
    Separate,
}

impl From<CfgModeArg> for CfgPredictionMode {
    fn from(value: CfgModeArg) -> Self {
        match value {
            CfgModeArg::Batched => Self::Batched,
            CfgModeArg::Separate => Self::Separate,
        }
    }
}

#[derive(Clone)]
struct OwnedTensor {
    shape: Vec<usize>,
    data: Vec<u8>,
    dtype: Dtype,
}

impl View for OwnedTensor {
    fn dtype(&self) -> Dtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.data)
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    match args.backend {
        BackendArg::Ndarray => {
            if matches!(args.precision, PrecisionArg::F16) {
                return Err(
                    "triposplat_stage_export ndarray backend does not support F16 tensors; use --precision f32 or --backend wgpu"
                        .into(),
                );
            }
            run_export::<ExportBackend>(&args, "ndarray", Default::default())
        }
        BackendArg::Wgpu => {
            #[cfg(feature = "backend_wgpu")]
            {
                run_export::<WgpuExportBackend>(&args, "wgpu", Default::default())
            }
            #[cfg(not(feature = "backend_wgpu"))]
            {
                Err("triposplat_stage_export --backend wgpu requires feature backend_wgpu".into())
            }
        }
    }
}

fn run_export<B: Backend>(
    args: &Args,
    backend_label: &'static str,
    device: B::Device,
) -> Result<(), Box<dyn std::error::Error>> {
    if args.decode_reference_latent && args.stop_after >= StopAfter::Decode {
        return run_reference_latent_decode_export::<B>(args, backend_label, device);
    }

    let pipeline = TripoSplatPipeline::from_pretrained(
        Some(args.weights_root.clone()),
        args.precision.into(),
    )?;
    let components = pipeline.load_runtime_components::<B>(&device)?;
    let image = read_image_tensor(&args.input_stages, &device)?;
    let tensors = export_stages(&components, image, &args)?;

    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)?;
    }
    serialize_to_file(
        tensors,
        Some(stage_metadata(args, backend_label)),
        &args.output,
    )?;
    eprintln!(
        "[triposplat_stage_export] wrote {} backend={} stop_after={:?}",
        args.output.display(),
        backend_label,
        args.stop_after
    );
    Ok(())
}

fn run_reference_latent_decode_export<B: Backend>(
    args: &Args,
    backend_label: &'static str,
    device: B::Device,
) -> Result<(), Box<dyn std::error::Error>> {
    let precision: TripoSplatBurnpackPrecision = args.precision.into();
    let decoder_path = args
        .weights_root
        .join("vae")
        .join(format!("triposplat_vae_decoder{}.bpk", precision.suffix()));
    let decoder = burn_triposplat::import::load_triposplat_decoder_from_burnpack_file::<B>(
        &device,
        &decoder_path,
        &OctreeProbabilityFixedlenDecoderConfig::triposplat(),
        &ElasticGaussianFixedlenDecoderConfig::triposplat(),
    )?;
    let options = TripoSplatOptions {
        steps: args.steps,
        guidance_scale: args.guidance_scale,
        shift: args.shift,
        seed: args.seed,
        num_gaussians: args.gaussians,
        ..Default::default()
    };
    let latent = read_required_f32_tensor_3d(&args.input_stages, "latent", &device)?;
    let camera = read_optional_f32_tensor_3d::<B>(&args.input_stages, "camera", &device)?;
    let mut tensors = vec![tensor_entry("latent", latent.clone())?];
    if let Some(camera) = camera {
        tensors.push(tensor_entry("camera", camera)?);
    }

    let splats = if args.decode_reference_sample {
        let sample = OctreeSample {
            points: read_required_f32_tensor_3d(&args.input_stages, "decoder_points", &device)?,
            log_probs: read_required_f32_tensor_2d(
                &args.input_stages,
                "decoder_log_probs",
                &device,
            )?,
        };
        tensors.push(tensor_entry("decoder_points", sample.points.clone())?);
        tensors.push(tensor_entry("decoder_log_probs", sample.log_probs.clone())?);
        let features = if args.use_reference_decoder_features {
            read_required_f32_tensor_3d(&args.input_stages, "decoder_features", &device)?
        } else {
            decoder.gs.forward(&sample, latent.clone())
        };
        tensors.push(tensor_entry("decoder_features", features.clone())?);
        decoder.gs.build_cloud(&sample, features)?
    } else {
        let num_gaussians = normalize_num_gaussians(options.num_gaussians)?;
        decoder.decode_to_cloud_with_seed(latent.clone(), num_gaussians, options.seed)?
    };
    eprintln!(
        "[triposplat_stage_export] decoded_reference_latent_splats={}",
        splats.len()
    );
    if let Some(path) = &args.splat_output {
        splats.write_splat(path)?;
        eprintln!(
            "[triposplat_stage_export] wrote_splat={} bytes={}",
            path.display(),
            splats.stats().splat_bytes
        );
    }
    if let Some(path) = &args.ply_output {
        splats.write_ply(path)?;
        eprintln!("[triposplat_stage_export] wrote_ply={}", path.display());
    }

    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)?;
    }
    serialize_to_file(
        tensors,
        Some(stage_metadata(args, backend_label)),
        &args.output,
    )?;
    eprintln!(
        "[triposplat_stage_export] wrote {} backend={} stop_after={:?}",
        args.output.display(),
        backend_label,
        args.stop_after
    );
    Ok(())
}

fn export_stages<B: Backend>(
    components: &TripoSplatRuntimeComponents<B>,
    image: Tensor<B, 4>,
    args: &Args,
) -> Result<Vec<(String, OwnedTensor)>, Box<dyn std::error::Error>> {
    let device = image.device();
    let mut out = vec![tensor_entry("image_rgb_0_1", image.clone())?];
    let condition =
        match read_optional_f32_tensor_4d::<B>(&args.input_stages, "vae_noise", &device)? {
            Some(vae_noise) => {
                eprintln!("[triposplat_stage_export] replaying upstream vae_noise");
                let diagnostics =
                    components.conditioning_diagnostics_with_vae_noise(image, vae_noise);
                out.push(tensor_entry("dinov3_raw", diagnostics.dinov3_raw.clone())?);
                out.push(tensor_entry("feature1", diagnostics.feature1.clone())?);
                out.push(tensor_entry("vae_mean", diagnostics.vae_mean.clone())?);
                out.push(tensor_entry("vae_logvar", diagnostics.vae_logvar.clone())?);
                let condition = diagnostics.into_condition();
                out.push(tensor_entry(
                    "feature2",
                    condition
                        .feature2
                        .clone()
                        .expect("TripoSplat VAE diagnostics should produce feature2"),
                )?);
                condition
            }
            None => {
                let condition = components.encode_preprocessed_image(image, args.seed);
                out.push(tensor_entry("feature1", condition.feature1.clone())?);
                if let Some(feature2) = condition.feature2.clone() {
                    out.push(tensor_entry("feature2", feature2)?);
                }
                condition
            }
        };

    if args.stop_after >= StopAfter::Sample {
        let options = TripoSplatOptions {
            steps: args.steps,
            guidance_scale: args.guidance_scale,
            shift: args.shift,
            seed: args.seed,
            num_gaussians: args.gaussians,
            ..Default::default()
        };
        let sampled = if args.decode_reference_latent {
            eprintln!("[triposplat_stage_export] decoding upstream reference latent");
            burn_triposplat::FlowState {
                latent: read_required_f32_tensor_3d(&args.input_stages, "latent", &device)?,
                camera: read_optional_f32_tensor_3d(&args.input_stages, "camera", &device)?,
            }
        } else {
            let reference_condition = read_reference_condition(&args.input_stages, &device)?
                .unwrap_or_else(|| condition.clone());
            let reference_flow_noise = read_reference_flow_noise(&args.input_stages, &device)?;
            if let Some(noise) = reference_flow_noise.clone()
                && args.trace_prefix_steps > 0
            {
                export_flow_prefix_trace(
                    components,
                    &mut out,
                    reference_condition.clone(),
                    noise,
                    options,
                    args.trace_prefix_steps,
                    args.cfg_mode.into(),
                )?;
                if args.trace_only {
                    return Ok(out);
                }
            }

            if let Some(noise) = reference_flow_noise {
                eprintln!(
                    "[triposplat_stage_export] replaying upstream flow_noise_latent for sample"
                );
                components.sample_latent_from_noise_with_cfg_mode(
                    reference_condition,
                    noise,
                    options,
                    args.cfg_mode.into(),
                )
            } else {
                components.sample_latent(condition, options)
            }
        };
        out.push(tensor_entry("latent", sampled.latent.clone())?);
        if let Some(camera) = sampled.camera.clone() {
            out.push(tensor_entry("camera", camera)?);
        }

        if args.stop_after >= StopAfter::Decode {
            let decoded = components.decode_latent(sampled.latent, options)?;
            eprintln!(
                "[triposplat_stage_export] decoded_splats={}",
                decoded.splats.len()
            );
            if let Some(path) = &args.splat_output {
                decoded.splats.write_splat(path)?;
                eprintln!(
                    "[triposplat_stage_export] wrote_splat={} bytes={}",
                    path.display(),
                    decoded.splats.stats().splat_bytes
                );
            }
            if let Some(path) = &args.ply_output {
                decoded.splats.write_ply(path)?;
                eprintln!("[triposplat_stage_export] wrote_ply={}", path.display());
            }
        }
    }
    Ok(out)
}

fn export_flow_prefix_trace<B: Backend>(
    components: &TripoSplatRuntimeComponents<B>,
    out: &mut Vec<(String, OwnedTensor)>,
    condition: burn_triposplat::TripoSplatCondition<B>,
    noise: burn_triposplat::FlowState<B>,
    options: TripoSplatOptions,
    prefix_steps: usize,
    cfg_mode: CfgPredictionMode,
) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!(
        "[triposplat_stage_export] exporting flow prefix trace steps={prefix_steps} cfg_mode={cfg_mode:?}"
    );
    let trace = components.sample_latent_trace_from_noise_with_cfg_mode(
        condition,
        noise,
        options,
        prefix_steps,
        cfg_mode,
    );
    if let Some(pred) = trace.pred0 {
        out.push(tensor_entry("flow_pred_000_latent", pred.latent)?);
        if let Some(camera) = pred.camera {
            out.push(tensor_entry("flow_pred_000_camera", camera)?);
        }
    }
    for (step, state) in trace.steps.into_iter().enumerate() {
        out.push(tensor_entry(
            &format!("flow_step_{step:03}_latent"),
            state.latent,
        )?);
        if let Some(camera) = state.camera {
            out.push(tensor_entry(
                &format!("flow_step_{step:03}_camera"),
                camera,
            )?);
        }
    }
    Ok(())
}

fn read_image_tensor<B: Backend>(
    path: &PathBuf,
    device: &<B as BackendTypes>::Device,
) -> Result<Tensor<B, 4>, Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let view = tensors.tensor("image_rgb_0_1")?;
    if view.dtype() != Dtype::F32 {
        return Err(format!("image_rgb_0_1 must be F32, got {:?}", view.dtype()).into());
    }
    let shape = view.shape();
    if shape.len() != 4 {
        return Err(format!("image_rgb_0_1 must be rank 4, got shape {:?}", shape).into());
    }
    let values = f32_values(&view)?;
    Ok(Tensor::<B, 1>::from_floats(values.as_slice(), device)
        .reshape([shape[0], shape[1], shape[2], shape[3]]))
}

fn read_reference_condition<B: Backend>(
    path: &PathBuf,
    device: &<B as BackendTypes>::Device,
) -> Result<Option<burn_triposplat::TripoSplatCondition<B>>, Box<dyn std::error::Error>> {
    let Some(feature1) = read_optional_f32_tensor_3d(path, "feature1", device)? else {
        return Ok(None);
    };
    let feature2 = read_optional_f32_tensor_3d(path, "feature2", device)?;
    Ok(Some(burn_triposplat::TripoSplatCondition {
        feature1,
        feature2,
        rng_normals_consumed: 0,
    }))
}

fn read_reference_flow_noise<B: Backend>(
    path: &PathBuf,
    device: &<B as BackendTypes>::Device,
) -> Result<Option<burn_triposplat::FlowState<B>>, Box<dyn std::error::Error>> {
    let Some(latent) = read_optional_f32_tensor_3d(path, "flow_noise_latent", device)? else {
        return Ok(None);
    };
    Ok(Some(burn_triposplat::FlowState {
        latent,
        camera: read_optional_f32_tensor_3d(path, "flow_noise_camera", device)?,
    }))
}

fn read_optional_f32_tensor_3d<B: Backend>(
    path: &PathBuf,
    name: &str,
    device: &<B as BackendTypes>::Device,
) -> Result<Option<Tensor<B, 3>>, Box<dyn std::error::Error>> {
    let Some((shape, values)) = read_optional_f32_tensor(path, name)? else {
        return Ok(None);
    };
    if shape.len() != 3 {
        return Err(format!("{name} must be rank 3, got shape {shape:?}").into());
    }
    Ok(Some(
        Tensor::<B, 1>::from_floats(values.as_slice(), device)
            .reshape([shape[0], shape[1], shape[2]]),
    ))
}

fn read_required_f32_tensor_3d<B: Backend>(
    path: &PathBuf,
    name: &str,
    device: &<B as BackendTypes>::Device,
) -> Result<Tensor<B, 3>, Box<dyn std::error::Error>> {
    read_optional_f32_tensor_3d(path, name, device)?
        .ok_or_else(|| format!("missing required tensor {name} in {}", path.display()).into())
}

fn read_required_f32_tensor_2d<B: Backend>(
    path: &PathBuf,
    name: &str,
    device: &<B as BackendTypes>::Device,
) -> Result<Tensor<B, 2>, Box<dyn std::error::Error>> {
    let Some((shape, values)) = read_optional_f32_tensor(path, name)? else {
        return Err(format!("missing required tensor {name} in {}", path.display()).into());
    };
    if shape.len() != 2 {
        return Err(format!("{name} must be rank 2, got shape {shape:?}").into());
    }
    Ok(Tensor::<B, 1>::from_floats(values.as_slice(), device).reshape([shape[0], shape[1]]))
}

fn read_optional_f32_tensor_4d<B: Backend>(
    path: &PathBuf,
    name: &str,
    device: &<B as BackendTypes>::Device,
) -> Result<Option<Tensor<B, 4>>, Box<dyn std::error::Error>> {
    let Some((shape, values)) = read_optional_f32_tensor(path, name)? else {
        return Ok(None);
    };
    if shape.len() != 4 {
        return Err(format!("{name} must be rank 4, got shape {shape:?}").into());
    }
    Ok(Some(
        Tensor::<B, 1>::from_floats(values.as_slice(), device)
            .reshape([shape[0], shape[1], shape[2], shape[3]]),
    ))
}

fn read_optional_f32_tensor(
    path: &PathBuf,
    name: &str,
) -> Result<Option<(Vec<usize>, Vec<f32>)>, Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let Ok(view) = tensors.tensor(name) else {
        return Ok(None);
    };
    if view.dtype() != Dtype::F32 {
        return Err(format!("{name} must be F32, got {:?}", view.dtype()).into());
    }
    Ok(Some((view.shape().to_vec(), f32_values(&view)?)))
}

fn tensor_entry<B: Backend, const D: usize>(
    name: &str,
    tensor: Tensor<B, D>,
) -> Result<(String, OwnedTensor), Box<dyn std::error::Error>> {
    let shape = tensor.dims().to_vec();
    let values = tensor
        .to_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|err| format!("failed to read tensor {name}: {err:?}"))?;
    Ok((name.to_string(), owned_f32_tensor(shape, &values)))
}

fn f32_values(view: &TensorView<'_>) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let chunks = view.data().chunks_exact(4);
    if !chunks.remainder().is_empty() {
        return Err("F32 tensor byte length is not divisible by 4".into());
    }
    Ok(chunks
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn owned_f32_tensor(shape: Vec<usize>, values: &[f32]) -> OwnedTensor {
    let mut data = Vec::with_capacity(values.len() * 4);
    for value in values {
        data.extend_from_slice(&value.to_le_bytes());
    }
    OwnedTensor {
        shape,
        data,
        dtype: Dtype::F32,
    }
}

fn stage_metadata(
    args: &Args,
    backend_label: &'static str,
) -> std::collections::HashMap<String, String> {
    [
        ("format", "triposplat_rust_stage_tensors_v1".to_string()),
        ("backend", backend_label.to_string()),
        ("precision", format!("{:?}", args.precision).to_lowercase()),
        ("seed", args.seed.to_string()),
        ("steps", args.steps.to_string()),
        ("guidance_scale", args.guidance_scale.to_string()),
        ("shift", args.shift.to_string()),
        ("num_gaussians", args.gaussians.to_string()),
        ("cfg_mode", format!("{:?}", args.cfg_mode).to_lowercase()),
        (
            "decode_reference_latent",
            args.decode_reference_latent.to_string(),
        ),
        (
            "decode_reference_sample",
            args.decode_reference_sample.to_string(),
        ),
        (
            "use_reference_decoder_features",
            args.use_reference_decoder_features.to_string(),
        ),
        (
            "splat_output",
            args.splat_output
                .as_ref()
                .map(|path| path.display().to_string())
                .unwrap_or_default(),
        ),
        (
            "ply_output",
            args.ply_output
                .as_ref()
                .map(|path| path.display().to_string())
                .unwrap_or_default(),
        ),
        (
            "stop_after",
            format!("{:?}", args.stop_after).to_lowercase(),
        ),
    ]
    .into_iter()
    .map(|(key, value)| (key.to_string(), value))
    .collect()
}

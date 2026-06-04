use std::{borrow::Cow, fs, path::PathBuf};

use burn::{backend::NdArray, prelude::*, tensor::backend::BackendTypes};
use burn_triposplat::{
    TripoSplatBurnpackPrecision, TripoSplatOptions, TripoSplatPipeline, TripoSplatRuntimeComponents,
};
use clap::{Parser, ValueEnum};
use safetensors::{
    SafeTensors,
    tensor::{Dtype, TensorView, View, serialize_to_file},
};

type ExportBackend = NdArray<f32>;

#[derive(Debug, Parser)]
#[command(about = "Export Rust TripoSplat stage tensors for upstream parity comparison.")]
struct Args {
    #[arg(long)]
    weights_root: PathBuf,

    #[arg(long, default_value = "f32")]
    precision: PrecisionArg,

    #[arg(long)]
    input_stages: PathBuf,

    #[arg(long)]
    output: PathBuf,

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
    if matches!(args.precision, PrecisionArg::F16) {
        return Err(
            "triposplat_stage_export uses the NdArray backend, which does not support F16 tensors; use --precision f32"
                .into(),
        );
    }
    let device = Default::default();
    let pipeline = TripoSplatPipeline::from_pretrained(
        Some(args.weights_root.clone()),
        args.precision.into(),
    )?;
    let components = pipeline.load_runtime_components::<ExportBackend>(&device)?;
    let image = read_image_tensor(&args.input_stages, &device)?;
    let tensors = export_stages(&components, image, &args)?;

    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)?;
    }
    serialize_to_file(tensors, Some(stage_metadata(&args)), &args.output)?;
    eprintln!(
        "[triposplat_stage_export] wrote {} stop_after={:?}",
        args.output.display(),
        args.stop_after
    );
    Ok(())
}

fn export_stages(
    components: &TripoSplatRuntimeComponents<ExportBackend>,
    image: Tensor<ExportBackend, 4>,
    args: &Args,
) -> Result<Vec<(String, OwnedTensor)>, Box<dyn std::error::Error>> {
    let mut out = vec![tensor_entry("image_rgb_0_1", image.clone())?];
    let condition = components.encode_preprocessed_image(image, args.seed);
    out.push(tensor_entry("feature1", condition.feature1.clone())?);
    if let Some(feature2) = condition.feature2.clone() {
        out.push(tensor_entry("feature2", feature2)?);
    }

    if args.stop_after >= StopAfter::Sample {
        let options = TripoSplatOptions {
            steps: args.steps,
            guidance_scale: args.guidance_scale,
            shift: args.shift,
            seed: args.seed,
            num_gaussians: args.gaussians,
            ..Default::default()
        };
        let sampled = components.sample_latent(condition, options);
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
        }
    }
    Ok(out)
}

fn read_image_tensor(
    path: &PathBuf,
    device: &<ExportBackend as BackendTypes>::Device,
) -> Result<Tensor<ExportBackend, 4>, Box<dyn std::error::Error>> {
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
    Ok(
        Tensor::<ExportBackend, 1>::from_floats(values.as_slice(), device)
            .reshape([shape[0], shape[1], shape[2], shape[3]]),
    )
}

fn tensor_entry<const D: usize>(
    name: &str,
    tensor: Tensor<ExportBackend, D>,
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

fn stage_metadata(args: &Args) -> std::collections::HashMap<String, String> {
    [
        ("format", "triposplat_rust_stage_tensors_v1".to_string()),
        ("backend", "ndarray".to_string()),
        ("precision", format!("{:?}", args.precision).to_lowercase()),
        ("seed", args.seed.to_string()),
        ("steps", args.steps.to_string()),
        ("guidance_scale", args.guidance_scale.to_string()),
        ("shift", args.shift.to_string()),
        ("num_gaussians", args.gaussians.to_string()),
        (
            "stop_after",
            format!("{:?}", args.stop_after).to_lowercase(),
        ),
    ]
    .into_iter()
    .map(|(key, value)| (key.to_string(), value))
    .collect()
}

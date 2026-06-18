#![allow(
    clippy::ptr_arg,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::vec_init_then_push
)]

use std::{borrow::Cow, fs, path::PathBuf, time::Instant};

use burn::{
    backend::NdArray,
    prelude::*,
    tensor::{DType, FloatDType, TensorData, backend::BackendTypes},
};
use burn_triposplat::{
    CfgPredictionMode, ElasticGaussianFixedlenDecoderConfig, GaussianSplatCloud,
    OctreeGaussianDecoder, OctreeProbabilityFixedlenDecoderConfig, OctreeSample,
    TripoSplatBurnpackPrecision, TripoSplatOptions, TripoSplatPipeline,
    TripoSplatRuntimeComponents, normalize_num_gaussians,
    runtime::{TripoSplatConditioningDiagnostics, TripoSplatEncodeTiming},
};
use clap::{Parser, ValueEnum};
use safetensors::{
    SafeTensors,
    tensor::{Dtype, TensorView, View, serialize_to_file},
};
use serde::Serialize;

type ExportBackend = NdArray<f32>;
#[cfg(feature = "backend_cuda")]
type CudaExportBackend = burn::backend::Cuda<f32, i32>;
#[cfg(feature = "backend_wgpu")]
type WgpuExportBackend = burn::backend::Wgpu<f32, i32, u32>;
#[cfg(feature = "backend_wgpu")]
type WgpuF16ExportBackend = burn::backend::Wgpu<burn::tensor::f16, i32, u32>;
#[cfg(feature = "backend_wgpu")]
type WgpuFlex32ExportBackend = burn::backend::Wgpu<burn::backend::wgpu::flex32, i32, u32>;

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

    #[arg(long, default_value_t = false)]
    use_reference_condition: bool,

    #[arg(long, default_value_t = 0)]
    trace_prefix_steps: usize,

    #[arg(long, default_value_t = false)]
    trace_only: bool,

    #[arg(long, value_enum)]
    cfg_mode: Option<CfgModeArg>,

    #[arg(long, value_enum, default_value_t = RuntimeProfileArg::Default)]
    runtime_profile: RuntimeProfileArg,

    #[arg(long, value_enum)]
    flow_compute_dtype: Option<ComputeDtypeArg>,

    #[arg(long, value_enum)]
    encode_dino_compute_dtype: Option<ComputeDtypeArg>,

    #[arg(long, value_enum)]
    encode_vae_compute_dtype: Option<ComputeDtypeArg>,

    #[arg(long)]
    profile_flow_output: Option<PathBuf>,

    #[arg(long, default_value_t = false)]
    profile_flow_only: bool,

    #[arg(long)]
    attention_qkv_output: Option<PathBuf>,

    #[arg(
        long,
        default_value = "cfg.separate.cond_forward.noise_refiner_00.block.attn"
    )]
    attention_qkv_label: String,

    #[arg(long)]
    profile_query_chunk_tokens: Option<usize>,

    #[arg(long)]
    flow_query_chunk_tokens: Option<usize>,

    #[arg(long, default_value_t = 0)]
    flow_warmup_steps: usize,

    #[arg(long, default_value_t = 1)]
    flow_timing_repeats: usize,

    #[arg(long)]
    flow_timing_output: Option<PathBuf>,

    #[arg(long, default_value_t = 0)]
    decode_warmup_steps: usize,

    #[arg(long, default_value_t = 1)]
    decode_timing_repeats: usize,

    #[arg(long)]
    decode_timing_output: Option<PathBuf>,

    #[arg(long, default_value_t = 0)]
    encode_warmup_steps: usize,

    #[arg(long, default_value_t = 1)]
    encode_timing_repeats: usize,

    #[arg(long)]
    encode_timing_output: Option<PathBuf>,

    #[arg(long)]
    vae_finite_report_output: Option<PathBuf>,

    #[arg(long)]
    forward_trace_output: Option<PathBuf>,

    #[arg(long, default_value_t = 0)]
    forward_trace_step: usize,

    #[arg(long, default_value_t = 512)]
    forward_trace_tokens: usize,
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
    Cuda,
    Wgpu,
    WgpuF16,
    WgpuBf16,
    WgpuFlex32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum CfgModeArg {
    Batched,
    BatchedMain,
    Separate,
}

impl From<CfgModeArg> for CfgPredictionMode {
    fn from(value: CfgModeArg) -> Self {
        match value {
            CfgModeArg::Batched => Self::Batched,
            CfgModeArg::BatchedMain => Self::BatchedMain,
            CfgModeArg::Separate => Self::Separate,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum RuntimeProfileArg {
    Default,
    Fast,
}

impl RuntimeProfileArg {
    fn as_str(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Fast => "fast",
        }
    }
}

fn effective_cfg_mode(args: &Args) -> CfgModeArg {
    args.cfg_mode.unwrap_or(match args.runtime_profile {
        RuntimeProfileArg::Default => CfgModeArg::Separate,
        RuntimeProfileArg::Fast => CfgModeArg::BatchedMain,
    })
}

fn effective_cfg_prediction_mode(args: &Args) -> CfgPredictionMode {
    effective_cfg_mode(args).into()
}

fn validate_runtime_profile(
    args: &Args,
    backend_label: &'static str,
) -> Result<(), Box<dyn std::error::Error>> {
    if args.runtime_profile != RuntimeProfileArg::Fast {
        return Ok(());
    }
    if backend_label != "wgpu-f16" || !matches!(args.precision, PrecisionArg::F16) {
        return Err(
            "--runtime-profile fast requires --backend wgpu-f16 --precision f16 so every TripoSplat stage uses the fp16 WGPU fast path"
                .into(),
        );
    }
    if let Some(cfg_mode) = args.cfg_mode
        && cfg_mode != CfgModeArg::BatchedMain
    {
        return Err(
            "--runtime-profile fast requires --cfg-mode batched-main or an omitted --cfg-mode"
                .into(),
        );
    }
    if args.flow_query_chunk_tokens.is_some() {
        return Err(
            "--runtime-profile fast owns the flow attention chunk policy; omit --flow-query-chunk-tokens"
                .into(),
        );
    }
    for (label, dtype, expected) in [
        (
            "--flow-compute-dtype",
            args.flow_compute_dtype,
            ComputeDtypeArg::F16,
        ),
        (
            "--encode-dino-compute-dtype",
            args.encode_dino_compute_dtype,
            ComputeDtypeArg::F32,
        ),
        (
            "--encode-vae-compute-dtype",
            args.encode_vae_compute_dtype,
            ComputeDtypeArg::F16,
        ),
    ] {
        if let Some(dtype) = dtype
            && dtype != expected
        {
            return Err(format!(
                "--runtime-profile fast requires {label}={expected:?} or an omitted {label}"
            )
            .into());
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum ComputeDtypeArg {
    F32,
    F16,
    Bf16,
}

impl From<ComputeDtypeArg> for FloatDType {
    fn from(value: ComputeDtypeArg) -> Self {
        match value {
            ComputeDtypeArg::F32 => Self::F32,
            ComputeDtypeArg::F16 => Self::F16,
            ComputeDtypeArg::Bf16 => Self::BF16,
        }
    }
}

#[derive(Serialize)]
struct FlowProfileReport {
    profile_schema_version: u32,
    backend_type: String,
    backend_precision_policy: String,
    runtime_profile: String,
    strict_reference_parity_supported: bool,
    strict_reference_parity_note: String,
    cfg_mode: String,
    step: usize,
    total_steps: usize,
    guidance_scale: f32,
    shift: f32,
    query_chunk_tokens: Option<usize>,
    attention_records: usize,
    attention_dense_calls: usize,
    attention_query_chunks: usize,
    output_latent_shape: Vec<usize>,
    output_camera_shape: Option<Vec<usize>>,
    records: Vec<burn_triposplat::components::TripoSplatProfileRecord>,
}

#[derive(Serialize)]
struct FlowTimingReport {
    timing_schema_version: u32,
    backend_type: String,
    backend_name: String,
    backend_precision_policy: String,
    runtime_profile: String,
    strict_reference_parity_supported: bool,
    strict_reference_parity_note: String,
    cfg_mode: String,
    precision: String,
    steps: usize,
    guidance_scale: f32,
    shift: f32,
    attention_query_chunk_tokens: Option<usize>,
    warmup_steps: usize,
    repeats: usize,
    sample_ms: Vec<f64>,
    sample_ms_avg: f64,
    sample_ms_min: f64,
    sample_ms_max: f64,
    condition_feature1_shape: Vec<usize>,
    condition_feature2_shape: Option<Vec<usize>>,
    condition_feature1_tokens: usize,
    condition_feature2_tokens: Option<usize>,
    output_latent_shape: Vec<usize>,
    output_camera_shape: Option<Vec<usize>>,
}

#[derive(Serialize)]
struct EncodeTimingReport {
    timing_schema_version: u32,
    backend_type: String,
    backend_name: String,
    backend_precision_policy: String,
    runtime_profile: String,
    strict_reference_parity_supported: bool,
    strict_reference_parity_note: String,
    precision: String,
    reference_vae_noise: bool,
    warmup_steps: usize,
    repeats: usize,
    total_ms: Vec<f64>,
    total_ms_avg: f64,
    total_ms_min: f64,
    total_ms_max: f64,
    stage_avg: TripoSplatEncodeTiming,
    repeats_detail: Vec<TripoSplatEncodeTiming>,
}

#[derive(Clone, Serialize)]
struct DecodeTimingSample {
    total_ms: f64,
    octree_sample_ms: Option<f64>,
    gaussian_forward_ms: Option<f64>,
    build_cloud_ms: Option<f64>,
    output_splats: usize,
}

#[derive(Serialize)]
struct DecodeTimingReport {
    timing_schema_version: u32,
    backend_type: String,
    backend_name: String,
    backend_precision_policy: String,
    runtime_profile: String,
    strict_reference_parity_supported: bool,
    strict_reference_parity_note: String,
    precision: String,
    reference_latent: bool,
    reference_sample: bool,
    reference_decoder_features: bool,
    gaussians: usize,
    decoder_tokens: usize,
    warmup_steps: usize,
    repeats: usize,
    total_ms_avg: f64,
    total_ms_min: f64,
    total_ms_max: f64,
    gaussian_forward_ms_avg: Option<f64>,
    build_cloud_ms_avg: Option<f64>,
    octree_sample_ms_avg: Option<f64>,
    repeats_detail: Vec<DecodeTimingSample>,
}

#[derive(Serialize)]
struct TensorFiniteStats {
    name: String,
    shape: Vec<usize>,
    dtype: String,
    finite_count: usize,
    nan_count: usize,
    infinite_count: usize,
    nonfinite_count: usize,
    min_finite: Option<f32>,
    max_finite: Option<f32>,
    mean_finite: Option<f64>,
}

#[derive(Serialize)]
struct VaeFiniteReport {
    report_schema_version: u32,
    backend_type: String,
    backend_name: String,
    precision: String,
    encode_vae_compute_dtype: String,
    records: Vec<TensorFiniteStats>,
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
                    "triposplat_stage_export ndarray backend does not support F16 tensors; use --precision f32 or a GPU f16 backend"
                        .into(),
                );
            }
            run_export::<ExportBackend>(&args, "ndarray", Default::default())
        }
        BackendArg::Cuda => {
            #[cfg(feature = "backend_cuda")]
            {
                if !matches!(args.precision, PrecisionArg::F32) {
                    return Err(
                        "triposplat_stage_export --backend cuda currently requires --precision f32"
                            .into(),
                    );
                }
                run_export::<CudaExportBackend>(&args, "cuda", Default::default())
            }
            #[cfg(not(feature = "backend_cuda"))]
            {
                Err("triposplat_stage_export --backend cuda requires feature backend_cuda".into())
            }
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
        BackendArg::WgpuF16 => {
            #[cfg(feature = "backend_wgpu")]
            {
                if !matches!(args.precision, PrecisionArg::F16) {
                    return Err(
                        "triposplat_stage_export --backend wgpu-f16 requires --precision f16"
                            .into(),
                    );
                }
                run_export::<WgpuF16ExportBackend>(&args, "wgpu-f16", Default::default())
            }
            #[cfg(not(feature = "backend_wgpu"))]
            {
                Err(
                    "triposplat_stage_export --backend wgpu-f16 requires feature backend_wgpu"
                        .into(),
                )
            }
        }
        BackendArg::WgpuBf16 => {
            Err(
                "triposplat_stage_export --backend wgpu-bf16 is disabled: Burn/CubeCL WGPU BF16 encode currently fails with DTypeMismatch and elemwise_fuse WGPU validation errors"
                    .into(),
            )
        }
        BackendArg::WgpuFlex32 => {
            #[cfg(feature = "backend_wgpu")]
            {
                if !matches!(args.precision, PrecisionArg::F32) {
                    return Err(
                        "triposplat_stage_export --backend wgpu-flex32 requires --precision f32"
                            .into(),
                    );
                }
                run_export::<WgpuFlex32ExportBackend>(&args, "wgpu-flex32", Default::default())
            }
            #[cfg(not(feature = "backend_wgpu"))]
            {
                Err(
                    "triposplat_stage_export --backend wgpu-flex32 requires feature backend_wgpu"
                        .into(),
                )
            }
        }
    }
}

fn run_export<B: Backend>(
    args: &Args,
    backend_label: &'static str,
    device: B::Device,
) -> Result<(), Box<dyn std::error::Error>> {
    validate_runtime_profile(args, backend_label)?;
    if args.decode_reference_latent && args.stop_after >= StopAfter::Decode {
        return run_reference_latent_decode_export::<B>(args, backend_label, device);
    }

    let precision: TripoSplatBurnpackPrecision = args.precision.into();
    let pipeline = TripoSplatPipeline::from_pretrained(Some(args.weights_root.clone()), precision)?;
    let mut compute_dtypes =
        burn_triposplat::import::default_runtime_compute_dtypes_for_backend::<B>(precision);
    if args.runtime_profile == RuntimeProfileArg::Fast {
        compute_dtypes = burn_triposplat::import::wgpu_f16_fast_runtime_compute_dtypes();
        eprintln!(
            "[triposplat_stage_export] runtime_profile=fast backend=wgpu-f16 precision=f16 cfg_mode={:?} dino=f32 flux_vae=f16 flow=f16 decoder=f16 attention=blackbox-padded",
            effective_cfg_mode(args)
        );
    }
    if backend_label == "cuda" && matches!(args.precision, PrecisionArg::F32) {
        eprintln!(
            "[triposplat_stage_export] warning: CUDA f32 uses Burn/CubeCL accelerated matmul, which may promote f32 tile operands to TF32; treat no-TF32 Python parity as unsupported unless a strict CUDA precision policy is added upstream"
        );
    }
    if let Some(dtype) = args.flow_compute_dtype {
        let dtype: FloatDType = dtype.into();
        if backend_label == "wgpu" && matches!(dtype, FloatDType::F16 | FloatDType::BF16) {
            return Err(
                "triposplat_stage_export --backend wgpu --flow-compute-dtype f16/bf16 is disabled: profiling shows this path is CPU-bound with low GPU utilization; use --backend wgpu-f16 for f16 experiments or the default f32 WGPU path for validated runs"
                    .into(),
            );
        }
        eprintln!("[triposplat_stage_export] casting flow to {dtype:?}");
        compute_dtypes.flow = Some(dtype);
    }
    if let Some(dtype) = args.encode_dino_compute_dtype {
        let dtype: FloatDType = dtype.into();
        if backend_label == "wgpu" && matches!(dtype, FloatDType::BF16) {
            return Err(
                "triposplat_stage_export --backend wgpu --encode-dino-compute-dtype bf16 is disabled until Burn/CubeCL WGPU BF16 encode is validated; use f16 for DINO encode experiments"
                    .into(),
            );
        }
        eprintln!("[triposplat_stage_export] loading DINOv3 encoder with {dtype:?} compute");
        compute_dtypes.dinov3 = Some(dtype);
    }
    if let Some(dtype) = args.encode_vae_compute_dtype {
        let dtype: FloatDType = dtype.into();
        if backend_label == "wgpu" && matches!(dtype, FloatDType::BF16) {
            return Err(
                "triposplat_stage_export --backend wgpu --encode-vae-compute-dtype bf16 is disabled: Burn/CubeCL WGPU BF16 Flux VAE encode segfaulted locally (rc=139); use f16 for the validated fast VAE experiment"
                    .into(),
            );
        }
        eprintln!("[triposplat_stage_export] loading Flux VAE encoder with {dtype:?} compute");
        compute_dtypes.flux2_vae_encoder = Some(dtype);
    }
    let components =
        pipeline.load_runtime_components_with_compute_dtypes::<B>(&device, compute_dtypes)?;
    let image = read_image_tensor(&args.input_stages, &device)?;
    let tensors = export_stages(&components, image, args, backend_label)?;

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
        attention_query_chunk_tokens: args.flow_query_chunk_tokens,
        cfg_mode: effective_cfg_prediction_mode(args),
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
        if decode_timing_requested(args) {
            if args.use_reference_decoder_features {
                return Err(
                    "decode timing with --use-reference-decoder-features is disabled because it skips Burn decoder forward; omit that flag to benchmark Burn decode"
                        .into(),
                );
            }
            let (splats, features, timing) = time_reference_sample_decode::<B>(
                args,
                backend_label,
                &device,
                &decoder,
                &sample,
                latent.clone(),
            )?;
            tensors.push(tensor_entry("decoder_features", features)?);
            if let Some(path) = &args.decode_timing_output {
                write_decode_timing_report::<B>(path, args, backend_label, &device, timing)?;
            }
            splats
        } else {
            let features = if args.use_reference_decoder_features {
                read_required_f32_tensor_3d(&args.input_stages, "decoder_features", &device)?
            } else {
                decoder.gs.forward(&sample, latent.clone())
            };
            tensors.push(tensor_entry("decoder_features", features.clone())?);
            decoder.gs.build_cloud(&sample, features)?
        }
    } else {
        let num_gaussians = normalize_num_gaussians(options.num_gaussians)?;
        if decode_timing_requested(args) {
            let (splats, timing) = time_full_decode::<B>(
                args,
                backend_label,
                &device,
                &decoder,
                latent.clone(),
                num_gaussians,
            )?;
            if let Some(path) = &args.decode_timing_output {
                write_decode_timing_report::<B>(path, args, backend_label, &device, timing)?;
            }
            splats
        } else {
            decoder.decode_to_cloud_with_seed(latent.clone(), num_gaussians, options.seed)?
        }
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

fn decode_timing_requested(args: &Args) -> bool {
    args.decode_warmup_steps > 0
        || args.decode_timing_repeats > 1
        || args.decode_timing_output.is_some()
}

fn time_reference_sample_decode<B: Backend>(
    args: &Args,
    _backend_label: &'static str,
    device: &B::Device,
    decoder: &OctreeGaussianDecoder<B>,
    sample: &OctreeSample<B>,
    latent: Tensor<B, 3>,
) -> Result<(GaussianSplatCloud, Tensor<B, 3>, Vec<DecodeTimingSample>), Box<dyn std::error::Error>>
{
    let mut last = None;
    for _ in 0..args.decode_warmup_steps {
        let _ = timed_reference_sample_decode_once(device, decoder, sample, latent.clone())?;
    }
    let repeats = args.decode_timing_repeats.max(1);
    let mut timings = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let (cloud, features, timing) =
            timed_reference_sample_decode_once(device, decoder, sample, latent.clone())?;
        timings.push(timing);
        last = Some((cloud, features));
    }
    let (cloud, features) = last.expect("at least one decode timing repeat should run");
    Ok((cloud, features, timings))
}

fn timed_reference_sample_decode_once<B: Backend>(
    device: &B::Device,
    decoder: &OctreeGaussianDecoder<B>,
    sample: &OctreeSample<B>,
    latent: Tensor<B, 3>,
) -> Result<(GaussianSplatCloud, Tensor<B, 3>, DecodeTimingSample), Box<dyn std::error::Error>> {
    sync_decode::<B>(device, "pre reference-sample decode timing")?;
    let total_start = Instant::now();

    let stage_start = Instant::now();
    let features = decoder.gs.forward(sample, latent);
    sync_decode::<B>(device, "Gaussian decoder forward")?;
    let gaussian_forward_ms = stage_start.elapsed().as_secs_f64() * 1000.0;

    let features_for_cloud = features.clone();
    let stage_start = Instant::now();
    let cloud = decoder.gs.build_cloud(sample, features_for_cloud)?;
    sync_decode::<B>(device, "Gaussian cloud build")?;
    let build_cloud_ms = stage_start.elapsed().as_secs_f64() * 1000.0;
    let output_splats = cloud.len();

    Ok((
        cloud,
        features,
        DecodeTimingSample {
            total_ms: total_start.elapsed().as_secs_f64() * 1000.0,
            octree_sample_ms: None,
            gaussian_forward_ms: Some(gaussian_forward_ms),
            build_cloud_ms: Some(build_cloud_ms),
            output_splats,
        },
    ))
}

fn time_full_decode<B: Backend>(
    args: &Args,
    _backend_label: &'static str,
    device: &B::Device,
    decoder: &OctreeGaussianDecoder<B>,
    latent: Tensor<B, 3>,
    num_gaussians: usize,
) -> Result<(GaussianSplatCloud, Vec<DecodeTimingSample>), Box<dyn std::error::Error>> {
    let mut last = None;
    for _ in 0..args.decode_warmup_steps {
        let _ = timed_full_decode_once(device, decoder, latent.clone(), num_gaussians, args.seed)?;
    }
    let repeats = args.decode_timing_repeats.max(1);
    let mut timings = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let (cloud, timing) =
            timed_full_decode_once(device, decoder, latent.clone(), num_gaussians, args.seed)?;
        timings.push(timing);
        last = Some(cloud);
    }
    Ok((
        last.expect("at least one decode timing repeat should run"),
        timings,
    ))
}

fn timed_full_decode_once<B: Backend>(
    device: &B::Device,
    decoder: &OctreeGaussianDecoder<B>,
    latent: Tensor<B, 3>,
    num_gaussians: usize,
    seed: u64,
) -> Result<(GaussianSplatCloud, DecodeTimingSample), Box<dyn std::error::Error>> {
    sync_decode::<B>(device, "pre full decode timing")?;
    let start = Instant::now();
    let cloud = decoder.decode_to_cloud_with_seed(latent, num_gaussians, seed)?;
    sync_decode::<B>(device, "full decode")?;
    let output_splats = cloud.len();
    Ok((
        cloud,
        DecodeTimingSample {
            total_ms: start.elapsed().as_secs_f64() * 1000.0,
            octree_sample_ms: None,
            gaussian_forward_ms: None,
            build_cloud_ms: None,
            output_splats,
        },
    ))
}

fn sync_decode<B: Backend>(
    device: &B::Device,
    label: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Err(err) = B::sync(device) {
        return Err(format!("decode timing sync failed after {label}: {err:?}").into());
    }
    Ok(())
}

fn write_decode_timing_report<B: Backend>(
    path: &PathBuf,
    args: &Args,
    backend_label: &'static str,
    device: &B::Device,
    timings: Vec<DecodeTimingSample>,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let decoder_tokens =
        normalize_num_gaussians(args.gaussians)? / burn_triposplat::TRIPOSPLAT_GAUSSIANS_PER_POINT;
    let report = DecodeTimingReport {
        timing_schema_version: 1,
        backend_type: std::any::type_name::<B>().to_string(),
        backend_name: B::name(device),
        backend_precision_policy: backend_precision_policy(args, std::any::type_name::<B>()),
        runtime_profile: args.runtime_profile.as_str().to_string(),
        strict_reference_parity_supported: strict_reference_parity_supported(
            args,
            std::any::type_name::<B>(),
        ),
        strict_reference_parity_note: strict_reference_parity_note(
            args,
            std::any::type_name::<B>(),
        ),
        precision: format!("{:?}", args.precision).to_lowercase(),
        reference_latent: args.decode_reference_latent,
        reference_sample: args.decode_reference_sample,
        reference_decoder_features: args.use_reference_decoder_features,
        gaussians: args.gaussians,
        decoder_tokens,
        warmup_steps: args.decode_warmup_steps,
        repeats: timings.len(),
        total_ms_avg: avg_decode_field(&timings, |timing| Some(timing.total_ms)).unwrap_or(0.0),
        total_ms_min: min_decode_field(&timings, |timing| Some(timing.total_ms)).unwrap_or(0.0),
        total_ms_max: max_decode_field(&timings, |timing| Some(timing.total_ms)).unwrap_or(0.0),
        gaussian_forward_ms_avg: avg_decode_field(&timings, |timing| timing.gaussian_forward_ms),
        build_cloud_ms_avg: avg_decode_field(&timings, |timing| timing.build_cloud_ms),
        octree_sample_ms_avg: avg_decode_field(&timings, |timing| timing.octree_sample_ms),
        repeats_detail: timings,
    };
    fs::write(path, serde_json::to_string_pretty(&report)? + "\n")?;
    eprintln!(
        "[triposplat_stage_export] wrote_decode_timing={}",
        path.display()
    );
    eprintln!(
        "[triposplat_stage_export] decode_timing_avg_ms={:.3} backend={} repeats={}",
        report.total_ms_avg, backend_label, report.repeats
    );
    Ok(())
}

fn avg_decode_field(
    timings: &[DecodeTimingSample],
    field: impl Fn(&DecodeTimingSample) -> Option<f64>,
) -> Option<f64> {
    let values = timings.iter().filter_map(field).collect::<Vec<_>>();
    if values.is_empty() {
        None
    } else {
        Some(values.iter().copied().sum::<f64>() / values.len() as f64)
    }
}

fn min_decode_field(
    timings: &[DecodeTimingSample],
    field: impl Fn(&DecodeTimingSample) -> Option<f64>,
) -> Option<f64> {
    timings.iter().filter_map(field).reduce(f64::min)
}

fn max_decode_field(
    timings: &[DecodeTimingSample],
    field: impl Fn(&DecodeTimingSample) -> Option<f64>,
) -> Option<f64> {
    timings.iter().filter_map(field).reduce(f64::max)
}

fn export_stages<B: Backend>(
    components: &TripoSplatRuntimeComponents<B>,
    image: Tensor<B, 4>,
    args: &Args,
    backend_label: &'static str,
) -> Result<Vec<(String, OwnedTensor)>, Box<dyn std::error::Error>> {
    let device = image.device();
    let mut out = vec![tensor_entry("image_rgb_0_1", image.clone())?];
    let condition = if args.use_reference_condition {
        let condition = read_reference_condition(&args.input_stages, &device)?
            .ok_or("--use-reference-condition requires feature1 in --input-stages")?;
        eprintln!("[triposplat_stage_export] using upstream reference condition; skipping encode");
        out.push(tensor_entry("feature1", condition.feature1.clone())?);
        if let Some(feature2) = condition.feature2.clone() {
            out.push(tensor_entry("feature2", feature2)?);
        }
        condition
    } else {
        match read_optional_f32_tensor_4d::<B>(&args.input_stages, "vae_noise", &device)? {
            Some(vae_noise) => {
                eprintln!("[triposplat_stage_export] replaying upstream vae_noise");
                if let Some(path) = &args.vae_finite_report_output {
                    write_vae_finite_report(
                        components,
                        args,
                        image.clone(),
                        vae_noise.clone(),
                        path,
                    )?;
                }
                let diagnostics = conditioning_diagnostics_with_optional_timing(
                    components, args, image, vae_noise,
                );
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
                let condition = encode_with_optional_timing(components, args, image);
                out.push(tensor_entry("feature1", condition.feature1.clone())?);
                if let Some(feature2) = condition.feature2.clone() {
                    out.push(tensor_entry("feature2", feature2)?);
                }
                condition
            }
        }
    };

    if args.stop_after >= StopAfter::Sample {
        let options = TripoSplatOptions {
            steps: args.steps,
            guidance_scale: args.guidance_scale,
            shift: args.shift,
            seed: args.seed,
            num_gaussians: args.gaussians,
            attention_query_chunk_tokens: args.flow_query_chunk_tokens,
            cfg_mode: effective_cfg_prediction_mode(args),
            ..Default::default()
        };
        let sampled = if args.decode_reference_latent {
            eprintln!("[triposplat_stage_export] decoding upstream reference latent");
            burn_triposplat::FlowState {
                latent: read_required_f32_tensor_3d(&args.input_stages, "latent", &device)?,
                camera: read_optional_f32_tensor_3d(&args.input_stages, "camera", &device)?,
            }
        } else {
            let reference_condition = if args.use_reference_condition {
                condition.clone()
            } else {
                read_reference_condition(&args.input_stages, &device)?
                    .unwrap_or_else(|| condition.clone())
            };
            let reference_flow_noise = read_reference_flow_noise(&args.input_stages, &device)?;
            if let Some(noise) = reference_flow_noise.clone()
                && (args.profile_flow_output.is_some() || args.attention_qkv_output.is_some())
            {
                warmup_flow_profile_if_requested(
                    components,
                    args,
                    reference_condition.clone(),
                    noise.clone(),
                    options,
                    effective_cfg_prediction_mode(args),
                );
                eprintln!(
                    "[triposplat_stage_export] profiling flow prediction step=0 cfg_mode={:?}",
                    effective_cfg_mode(args)
                );
                let mut captured_qkv = None;
                let (output_latent_shape, output_camera_shape, records) =
                    if args.attention_qkv_output.is_some() {
                        let profile =
                            if let Some(query_chunk_tokens) = args.profile_query_chunk_tokens {
                                components
                                    .flow
                                    .profile_euler_cfg_prediction_at_step_with_qkv_capture(
                                        noise,
                                        reference_condition.clone(),
                                        args.steps,
                                        0,
                                        args.guidance_scale,
                                        args.shift,
                                        effective_cfg_prediction_mode(args),
                                        query_chunk_tokens,
                                        args.attention_qkv_label.clone(),
                                    )
                            } else {
                                components
                                    .flow
                                    .profile_euler_cfg_prediction_at_step_with_mode_and_qkv_capture(
                                        noise,
                                        reference_condition.clone(),
                                        args.steps,
                                        0,
                                        args.guidance_scale,
                                        args.shift,
                                        effective_cfg_prediction_mode(args),
                                        args.attention_qkv_label.clone(),
                                    )
                            };
                        captured_qkv = profile.qkv_capture;
                        (
                            profile.output.latent.dims().to_vec(),
                            profile
                                .output
                                .camera
                                .as_ref()
                                .map(|camera| camera.dims().to_vec()),
                            profile.records,
                        )
                    } else if let Some(query_chunk_tokens) = args.profile_query_chunk_tokens {
                        let profile = components
                            .flow
                            .profile_euler_cfg_prediction_at_step_with_query_chunk_tokens(
                                noise,
                                reference_condition.clone(),
                                args.steps,
                                0,
                                args.guidance_scale,
                                args.shift,
                                effective_cfg_prediction_mode(args),
                                query_chunk_tokens,
                            );
                        (
                            profile.output.latent.dims().to_vec(),
                            profile
                                .output
                                .camera
                                .as_ref()
                                .map(|camera| camera.dims().to_vec()),
                            profile.records,
                        )
                    } else {
                        let profile = components
                            .flow
                            .profile_euler_cfg_prediction_at_step_with_mode(
                                noise,
                                reference_condition.clone(),
                                args.steps,
                                0,
                                args.guidance_scale,
                                args.shift,
                                effective_cfg_prediction_mode(args),
                            );
                        (
                            profile.output.latent.dims().to_vec(),
                            profile
                                .output
                                .camera
                                .as_ref()
                                .map(|camera| camera.dims().to_vec()),
                            profile.records,
                        )
                    };
                if let Some(path) = &args.profile_flow_output {
                    write_flow_profile_report::<B>(
                        path,
                        args,
                        output_latent_shape,
                        output_camera_shape,
                        records,
                    )?;
                } else {
                    drop(records);
                }
                if let Some(path) = &args.attention_qkv_output {
                    let capture = captured_qkv.ok_or_else(|| {
                        format!(
                            "attention QKV label '{}' was not captured during flow profile",
                            args.attention_qkv_label
                        )
                    })?;
                    write_attention_qkv_capture(path, args, capture)?;
                }
                if args.profile_flow_only {
                    return Ok(out);
                }
            }
            if let (Some(path), Some(noise)) =
                (&args.forward_trace_output, reference_flow_noise.clone())
            {
                if args.forward_trace_step >= args.steps {
                    return Err("--forward-trace-step must be less than --steps".into());
                }
                eprintln!(
                    "[triposplat_stage_export] exporting forward trace step={} cfg_mode={:?} tokens={}",
                    args.forward_trace_step,
                    effective_cfg_mode(args),
                    args.forward_trace_tokens
                );
                let trace_sample = if args.forward_trace_step == 0 {
                    noise
                } else {
                    components.flow.sample_euler_cfg_prefix_with_mode(
                        noise,
                        reference_condition.clone(),
                        args.steps,
                        args.forward_trace_step,
                        args.guidance_scale,
                        args.shift,
                        effective_cfg_prediction_mode(args),
                    )
                };
                let trace = components
                    .flow
                    .trace_euler_cfg_prediction_at_step_with_mode(
                        trace_sample,
                        reference_condition.clone(),
                        args.steps,
                        args.forward_trace_step,
                        args.guidance_scale,
                        args.shift,
                        effective_cfg_prediction_mode(args),
                        args.forward_trace_tokens,
                    );
                write_forward_trace(path, args, trace)?;
                if args.trace_only {
                    return Ok(out);
                }
            }
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
                    effective_cfg_prediction_mode(args),
                )?;
                if args.trace_only {
                    return Ok(out);
                }
            }

            if let Some(noise) = reference_flow_noise {
                eprintln!(
                    "[triposplat_stage_export] replaying upstream flow_noise_latent for sample"
                );
                sample_latent_from_noise_with_optional_timing(
                    components,
                    args,
                    reference_condition,
                    noise,
                    options,
                    effective_cfg_prediction_mode(args),
                )
            } else {
                sample_latent_with_optional_timing(components, args, condition, options)
            }
        };
        out.push(tensor_entry("latent", sampled.latent.clone())?);
        if let Some(camera) = sampled.camera.clone() {
            out.push(tensor_entry("camera", camera)?);
        }

        if args.stop_after >= StopAfter::Decode {
            let splats = if decode_timing_requested(args) {
                let num_gaussians = normalize_num_gaussians(options.num_gaussians)?;
                let (splats, timing) = time_full_decode::<B>(
                    args,
                    backend_label,
                    &device,
                    &components.decoder,
                    sampled.latent,
                    num_gaussians,
                )?;
                if let Some(path) = &args.decode_timing_output {
                    write_decode_timing_report::<B>(path, args, backend_label, &device, timing)?;
                }
                splats
            } else {
                components.decode_latent(sampled.latent, options)?.splats
            };
            eprintln!("[triposplat_stage_export] decoded_splats={}", splats.len());
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
        }
    }
    Ok(out)
}

fn encode_timing_requested(args: &Args) -> bool {
    args.encode_warmup_steps > 0
        || args.encode_timing_repeats > 1
        || args.encode_timing_output.is_some()
}

fn encode_with_optional_timing<B: Backend>(
    components: &TripoSplatRuntimeComponents<B>,
    args: &Args,
    image: Tensor<B, 4>,
) -> burn_triposplat::TripoSplatCondition<B> {
    if !encode_timing_requested(args) {
        return components.encode_preprocessed_image(image, args.seed);
    }

    let device = image.device();
    for _ in 0..args.encode_warmup_steps {
        let _ = components.encode_preprocessed_image(image.clone(), args.seed);
        B::sync(&device).expect("encode warmup sync failed");
    }

    let repeats = args.encode_timing_repeats.max(1);
    let mut timings = Vec::with_capacity(repeats);
    let mut last = None;
    for _ in 0..repeats {
        B::sync(&device).expect("encode pre-timing sync failed");
        let (condition, timing) =
            components.encode_preprocessed_image_with_timing(image.clone(), args.seed);
        timings.push(timing);
        last = Some(condition);
    }

    if let Some(path) = &args.encode_timing_output {
        write_encode_timing_report::<B>(path, args, &device, false, timings)
            .expect("failed to write encode timing report");
    }
    last.expect("at least one encode timing repeat should run")
}

fn conditioning_diagnostics_with_optional_timing<B: Backend>(
    components: &TripoSplatRuntimeComponents<B>,
    args: &Args,
    image: Tensor<B, 4>,
    vae_noise: Tensor<B, 4>,
) -> TripoSplatConditioningDiagnostics<B> {
    if !encode_timing_requested(args) {
        return components.conditioning_diagnostics_with_vae_noise(image, vae_noise);
    }

    let device = image.device();
    for _ in 0..args.encode_warmup_steps {
        let _ =
            components.conditioning_diagnostics_with_vae_noise(image.clone(), vae_noise.clone());
        B::sync(&device).expect("encode warmup sync failed");
    }

    let repeats = args.encode_timing_repeats.max(1);
    let mut timings = Vec::with_capacity(repeats);
    let mut last = None;
    for _ in 0..repeats {
        B::sync(&device).expect("encode pre-timing sync failed");
        let (diagnostics, timing) = components
            .conditioning_diagnostics_with_vae_noise_timed(image.clone(), vae_noise.clone());
        timings.push(timing);
        last = Some(diagnostics);
    }

    if let Some(path) = &args.encode_timing_output {
        write_encode_timing_report::<B>(path, args, &device, true, timings)
            .expect("failed to write encode timing report");
    }
    last.expect("at least one encode timing repeat should run")
}

fn write_encode_timing_report<B: Backend>(
    path: &PathBuf,
    args: &Args,
    device: &B::Device,
    reference_vae_noise: bool,
    timings: Vec<TripoSplatEncodeTiming>,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let total_ms = timings
        .iter()
        .map(|timing| timing.total_ms)
        .collect::<Vec<_>>();
    let total_ms_avg = total_ms.iter().copied().sum::<f64>() / total_ms.len().max(1) as f64;
    let total_ms_min = total_ms.iter().copied().fold(f64::INFINITY, f64::min);
    let total_ms_max = total_ms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let report = EncodeTimingReport {
        timing_schema_version: 1,
        backend_type: std::any::type_name::<B>().to_string(),
        backend_name: B::name(device),
        backend_precision_policy: backend_precision_policy(args, std::any::type_name::<B>()),
        runtime_profile: args.runtime_profile.as_str().to_string(),
        strict_reference_parity_supported: strict_reference_parity_supported(
            args,
            std::any::type_name::<B>(),
        ),
        strict_reference_parity_note: strict_reference_parity_note(
            args,
            std::any::type_name::<B>(),
        ),
        precision: format!("{:?}", args.precision).to_lowercase(),
        reference_vae_noise,
        warmup_steps: args.encode_warmup_steps,
        repeats: timings.len(),
        total_ms,
        total_ms_avg,
        total_ms_min,
        total_ms_max,
        stage_avg: average_encode_timing(&timings),
        repeats_detail: timings,
    };
    fs::write(path, serde_json::to_string_pretty(&report)? + "\n")?;
    eprintln!(
        "[triposplat_stage_export] wrote_encode_timing={}",
        path.display()
    );
    Ok(())
}

fn average_encode_timing(timings: &[TripoSplatEncodeTiming]) -> TripoSplatEncodeTiming {
    let count = timings.len().max(1) as f64;
    let avg =
        |value: fn(&TripoSplatEncodeTiming) -> f64| timings.iter().map(value).sum::<f64>() / count;
    let last = timings.last().cloned().unwrap_or_default();
    TripoSplatEncodeTiming {
        input_cast_ms: avg(|timing| timing.input_cast_ms),
        dinov3_normalize_ms: avg(|timing| timing.dinov3_normalize_ms),
        dinov3_forward_ms: avg(|timing| timing.dinov3_forward_ms),
        dinov3_layer_norm_ms: avg(|timing| timing.dinov3_layer_norm_ms),
        flux_image_ms: avg(|timing| timing.flux_image_ms),
        vae_encode_ms: avg(|timing| timing.vae_encode_ms),
        condition_pack_ms: avg(|timing| timing.condition_pack_ms),
        total_ms: avg(|timing| timing.total_ms),
        feature1_shape: last.feature1_shape,
        feature2_shape: last.feature2_shape,
    }
}

fn write_vae_finite_report<B: Backend>(
    components: &TripoSplatRuntimeComponents<B>,
    args: &Args,
    image_rgb_0_1: Tensor<B, 4>,
    vae_noise: Tensor<B, 4>,
    path: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let vae_dtype = components.flux2_vae_encoder.float_dtype();
    let flux_image = cast_tensor_dtype(image_rgb_0_1, vae_dtype) * 2.0 - 1.0;
    let vae_noise = cast_tensor_dtype(vae_noise, vae_dtype);
    let trace = components
        .flux2_vae_encoder
        .encode_with_noise_trace(flux_image, vae_noise);

    let mut records = Vec::new();
    records.push(tensor_finite_stats(
        "encoder.conv_in",
        trace.encoder.conv_in,
    )?);
    records.push(tensor_finite_stats(
        "encoder.down_0_resnet_0",
        trace.encoder.down_0_resnet_0,
    )?);
    records.push(tensor_finite_stats(
        "encoder.down_0_resnet_1",
        trace.encoder.down_0_resnet_1,
    )?);
    records.push(tensor_finite_stats(
        "encoder.down_0_sampler",
        trace.encoder.down_0_sampler,
    )?);
    records.push(tensor_finite_stats(
        "encoder.down_1_resnet_0",
        trace.encoder.down_1_resnet_0,
    )?);
    records.push(tensor_finite_stats(
        "encoder.down_1_resnet_1",
        trace.encoder.down_1_resnet_1,
    )?);
    records.push(tensor_finite_stats(
        "encoder.down_1_sampler",
        trace.encoder.down_1_sampler,
    )?);
    records.push(tensor_finite_stats(
        "encoder.down_2_resnet_0",
        trace.encoder.down_2_resnet_0,
    )?);
    records.push(tensor_finite_stats(
        "encoder.down_2_resnet_1",
        trace.encoder.down_2_resnet_1,
    )?);
    records.push(tensor_finite_stats(
        "encoder.down_2_sampler",
        trace.encoder.down_2_sampler,
    )?);
    records.push(tensor_finite_stats(
        "encoder.down_3_resnet_0",
        trace.encoder.down_3_resnet_0,
    )?);
    records.push(tensor_finite_stats(
        "encoder.down_3_resnet_1",
        trace.encoder.down_3_resnet_1,
    )?);
    records.push(tensor_finite_stats(
        "encoder.mid_resnet_0",
        trace.encoder.mid_resnet_0,
    )?);
    records.push(tensor_finite_stats(
        "encoder.mid_attn",
        trace.encoder.mid_attn,
    )?);
    records.push(tensor_finite_stats(
        "encoder.mid_resnet_1",
        trace.encoder.mid_resnet_1,
    )?);
    records.push(tensor_finite_stats(
        "encoder.encoder_out",
        trace.encoder.encoder_out,
    )?);
    records.push(tensor_finite_stats("moments", trace.moments)?);
    records.push(tensor_finite_stats("mean", trace.mean)?);
    records.push(tensor_finite_stats("logvar", trace.logvar)?);
    records.push(tensor_finite_stats("latents", trace.latents)?);
    records.push(tensor_finite_stats("unshuffled", trace.unshuffled)?);
    records.push(tensor_finite_stats("normalized", trace.normalized)?);
    records.push(tensor_finite_stats("tokens", trace.tokens)?);

    let report = VaeFiniteReport {
        report_schema_version: 1,
        backend_type: std::any::type_name::<B>().to_string(),
        backend_name: B::name(&components.dinov3.patch_embed.proj.weight.val().device()),
        precision: format!("{:?}", args.precision).to_lowercase(),
        encode_vae_compute_dtype: args
            .encode_vae_compute_dtype
            .map(|dtype| format!("{dtype:?}").to_lowercase())
            .unwrap_or_else(|| "default".to_string()),
        records,
    };
    fs::write(path, serde_json::to_string_pretty(&report)? + "\n")?;
    eprintln!(
        "[triposplat_stage_export] wrote_vae_finite_report={}",
        path.display()
    );
    Ok(())
}

fn tensor_finite_stats<B: Backend, const D: usize>(
    name: &str,
    tensor: Tensor<B, D>,
) -> Result<TensorFiniteStats, Box<dyn std::error::Error>> {
    let shape = tensor.dims().to_vec();
    let dtype = format!("{:?}", tensor.dtype());
    let values = tensor
        .to_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|err| format!("failed to read tensor {name}: {err:?}"))?;
    let mut finite_count = 0usize;
    let mut nan_count = 0usize;
    let mut infinite_count = 0usize;
    let mut min_finite = f32::INFINITY;
    let mut max_finite = f32::NEG_INFINITY;
    let mut sum_finite = 0.0_f64;
    for value in values {
        if value.is_nan() {
            nan_count += 1;
        } else if value.is_infinite() {
            infinite_count += 1;
        } else {
            finite_count += 1;
            min_finite = min_finite.min(value);
            max_finite = max_finite.max(value);
            sum_finite += value as f64;
        }
    }
    let nonfinite_count = nan_count + infinite_count;
    Ok(TensorFiniteStats {
        name: name.to_string(),
        shape,
        dtype,
        finite_count,
        nan_count,
        infinite_count,
        nonfinite_count,
        min_finite: (finite_count > 0).then_some(min_finite),
        max_finite: (finite_count > 0).then_some(max_finite),
        mean_finite: (finite_count > 0).then_some(sum_finite / finite_count as f64),
    })
}

fn cast_tensor_dtype<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    dtype: FloatDType,
) -> Tensor<B, D> {
    let current: FloatDType = tensor.dtype().into();
    if current == dtype {
        tensor
    } else {
        tensor.cast(dtype)
    }
}

fn warmup_flow_profile_if_requested<B: Backend>(
    components: &TripoSplatRuntimeComponents<B>,
    args: &Args,
    condition: burn_triposplat::TripoSplatCondition<B>,
    noise: burn_triposplat::FlowState<B>,
    options: TripoSplatOptions,
    cfg_mode: CfgPredictionMode,
) {
    if args.flow_warmup_steps == 0 {
        return;
    }

    eprintln!(
        "[triposplat_stage_export] warming flow profile runs={}",
        args.flow_warmup_steps
    );
    let device = noise.latent.device();
    for _ in 0..args.flow_warmup_steps {
        let _ = components.sample_latent_from_noise_with_cfg_mode(
            condition.clone(),
            noise.clone(),
            options,
            cfg_mode,
        );
        B::sync(&device).expect("flow profile warmup sync failed");
    }
}

fn write_flow_profile_report<B: Backend>(
    path: &PathBuf,
    args: &Args,
    output_latent_shape: Vec<usize>,
    output_camera_shape: Option<Vec<usize>>,
    records: Vec<burn_triposplat::components::TripoSplatProfileRecord>,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let report = FlowProfileReport {
        profile_schema_version: 2,
        backend_type: std::any::type_name::<B>().to_string(),
        backend_precision_policy: backend_precision_policy(args, std::any::type_name::<B>()),
        runtime_profile: args.runtime_profile.as_str().to_string(),
        strict_reference_parity_supported: strict_reference_parity_supported(
            args,
            std::any::type_name::<B>(),
        ),
        strict_reference_parity_note: strict_reference_parity_note(
            args,
            std::any::type_name::<B>(),
        ),
        cfg_mode: format!("{:?}", effective_cfg_mode(args)),
        step: 0,
        total_steps: args.steps,
        guidance_scale: args.guidance_scale,
        shift: args.shift,
        query_chunk_tokens: args.profile_query_chunk_tokens,
        attention_records: records
            .iter()
            .filter(|record| record.attention_path.is_some())
            .count(),
        attention_dense_calls: records.iter().filter_map(|record| record.dense_calls).sum(),
        attention_query_chunks: records
            .iter()
            .filter_map(|record| record.query_chunks)
            .sum(),
        output_latent_shape,
        output_camera_shape,
        records,
    };
    fs::write(path, serde_json::to_string_pretty(&report)? + "\n")?;
    eprintln!(
        "[triposplat_stage_export] wrote_flow_profile={}",
        path.display()
    );
    Ok(())
}

fn sample_latent_with_optional_timing<B: Backend>(
    components: &TripoSplatRuntimeComponents<B>,
    args: &Args,
    condition: burn_triposplat::TripoSplatCondition<B>,
    options: TripoSplatOptions,
) -> burn_triposplat::FlowState<B> {
    if args.flow_warmup_steps == 0
        && args.flow_timing_repeats <= 1
        && args.flow_timing_output.is_none()
    {
        return components.sample_latent(condition, options);
    }

    let device = condition.feature1.device();
    for _ in 0..args.flow_warmup_steps {
        let _ = components.sample_latent(condition.clone(), options);
        B::sync(&device).expect("flow warmup sync failed");
    }

    let repeats = args.flow_timing_repeats.max(1);
    let mut sample_ms = Vec::with_capacity(repeats);
    let mut last = None;
    for _ in 0..repeats {
        B::sync(&device).expect("flow pre-timing sync failed");
        let start = Instant::now();
        let sample = components.sample_latent(condition.clone(), options);
        B::sync(&device).expect("flow post-timing sync failed");
        sample_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        last = Some(sample);
    }
    let sample = last.expect("at least one flow timing repeat should run");
    if let Some(path) = &args.flow_timing_output {
        write_flow_timing_report::<B>(
            path,
            args,
            &device,
            &condition,
            &sample,
            sample_ms,
            args.flow_warmup_steps,
            repeats,
        )
        .expect("failed to write flow timing report");
    }
    sample
}

fn sample_latent_from_noise_with_optional_timing<B: Backend>(
    components: &TripoSplatRuntimeComponents<B>,
    args: &Args,
    condition: burn_triposplat::TripoSplatCondition<B>,
    noise: burn_triposplat::FlowState<B>,
    options: TripoSplatOptions,
    cfg_mode: CfgPredictionMode,
) -> burn_triposplat::FlowState<B> {
    if args.flow_warmup_steps == 0
        && args.flow_timing_repeats <= 1
        && args.flow_timing_output.is_none()
    {
        return components
            .sample_latent_from_noise_with_cfg_mode(condition, noise, options, cfg_mode);
    }

    let device = noise.latent.device();
    for _ in 0..args.flow_warmup_steps {
        let _ = components.sample_latent_from_noise_with_cfg_mode(
            condition.clone(),
            noise.clone(),
            options,
            cfg_mode,
        );
        B::sync(&device).expect("flow warmup sync failed");
    }

    let repeats = args.flow_timing_repeats.max(1);
    let mut sample_ms = Vec::with_capacity(repeats);
    let mut last = None;
    for _ in 0..repeats {
        B::sync(&device).expect("flow pre-timing sync failed");
        let start = Instant::now();
        let sample = components.sample_latent_from_noise_with_cfg_mode(
            condition.clone(),
            noise.clone(),
            options,
            cfg_mode,
        );
        B::sync(&device).expect("flow post-timing sync failed");
        sample_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        last = Some(sample);
    }
    let sample = last.expect("at least one flow timing repeat should run");
    if let Some(path) = &args.flow_timing_output {
        write_flow_timing_report::<B>(
            path,
            args,
            &device,
            &condition,
            &sample,
            sample_ms,
            args.flow_warmup_steps,
            repeats,
        )
        .expect("failed to write flow timing report");
    }
    sample
}

fn write_flow_timing_report<B: Backend>(
    path: &PathBuf,
    args: &Args,
    device: &B::Device,
    condition: &burn_triposplat::TripoSplatCondition<B>,
    sample: &burn_triposplat::FlowState<B>,
    sample_ms: Vec<f64>,
    warmup_steps: usize,
    repeats: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let sample_ms_avg = sample_ms.iter().copied().sum::<f64>() / sample_ms.len().max(1) as f64;
    let sample_ms_min = sample_ms.iter().copied().fold(f64::INFINITY, f64::min);
    let sample_ms_max = sample_ms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let condition_feature1_shape = condition.feature1.dims().to_vec();
    let condition_feature2_shape = condition
        .feature2
        .as_ref()
        .map(|feature| feature.dims().to_vec());
    let condition_feature1_tokens = condition_feature1_shape.get(1).copied().unwrap_or(0);
    let condition_feature2_tokens = condition_feature2_shape
        .as_ref()
        .and_then(|shape| shape.get(1).copied());
    let report = FlowTimingReport {
        timing_schema_version: 2,
        backend_type: std::any::type_name::<B>().to_string(),
        backend_name: B::name(device),
        backend_precision_policy: backend_precision_policy(args, std::any::type_name::<B>()),
        runtime_profile: args.runtime_profile.as_str().to_string(),
        strict_reference_parity_supported: strict_reference_parity_supported(
            args,
            std::any::type_name::<B>(),
        ),
        strict_reference_parity_note: strict_reference_parity_note(
            args,
            std::any::type_name::<B>(),
        ),
        cfg_mode: format!("{:?}", effective_cfg_mode(args)),
        precision: format!("{:?}", args.precision).to_lowercase(),
        steps: args.steps,
        guidance_scale: args.guidance_scale,
        shift: args.shift,
        attention_query_chunk_tokens: args.flow_query_chunk_tokens,
        warmup_steps,
        repeats,
        sample_ms,
        sample_ms_avg,
        sample_ms_min,
        sample_ms_max,
        condition_feature1_shape,
        condition_feature2_shape,
        condition_feature1_tokens,
        condition_feature2_tokens,
        output_latent_shape: sample.latent.dims().to_vec(),
        output_camera_shape: sample.camera.as_ref().map(|camera| camera.dims().to_vec()),
    };
    fs::write(path, serde_json::to_string_pretty(&report)? + "\n")?;
    eprintln!(
        "[triposplat_stage_export] wrote_flow_timing={}",
        path.display()
    );
    Ok(())
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
    if trace.preds.is_empty() {
        if let Some(pred) = trace.pred0 {
            out.push(tensor_entry("flow_pred_000_latent", pred.latent)?);
            if let Some(camera) = pred.camera {
                out.push(tensor_entry("flow_pred_000_camera", camera)?);
            }
        }
    } else {
        for (step, pred) in trace.preds.into_iter().enumerate() {
            out.push(tensor_entry(
                &format!("flow_pred_{step:03}_latent"),
                pred.latent,
            )?);
            if let Some(camera) = pred.camera {
                out.push(tensor_entry(
                    &format!("flow_pred_{step:03}_camera"),
                    camera,
                )?);
            }
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

fn write_forward_trace<B: Backend>(
    path: &PathBuf,
    args: &Args,
    trace: Vec<(String, Tensor<B, 3>)>,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tensors = trace
        .into_iter()
        .map(|(name, tensor)| tensor_entry(&name, tensor))
        .collect::<Result<Vec<_>, _>>()?;
    serialize_to_file(tensors, Some(forward_trace_metadata(args)), path)?;
    eprintln!(
        "[triposplat_stage_export] wrote_forward_trace={}",
        path.display()
    );
    Ok(())
}

fn write_attention_qkv_capture<B: Backend>(
    path: &PathBuf,
    args: &Args,
    capture: burn_triposplat::components::AttentionQkvCapture<B>,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let label = capture.label.clone();
    let q_shape = capture.q.dims().to_vec();
    let k_shape = capture.k.dims().to_vec();
    let v_shape = capture.v.dims().to_vec();
    let tensors = vec![
        tensor_entry("q", capture.q)?,
        tensor_entry("k", capture.k)?,
        tensor_entry("v", capture.v)?,
    ];
    serialize_to_file(
        tensors,
        Some(attention_qkv_metadata(
            args, label, q_shape, k_shape, v_shape,
        )),
        path,
    )?;
    eprintln!(
        "[triposplat_stage_export] wrote_attention_qkv_capture={}",
        path.display()
    );
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
    Ok(f32_tensor_1d(values, device).reshape([shape[0], shape[1], shape[2], shape[3]]))
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
        f32_tensor_1d(values, device).reshape([shape[0], shape[1], shape[2]]),
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
    Ok(f32_tensor_1d(values, device).reshape([shape[0], shape[1]]))
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
        f32_tensor_1d(values, device).reshape([shape[0], shape[1], shape[2], shape[3]]),
    ))
}

fn f32_tensor_1d<B: Backend>(
    values: Vec<f32>,
    device: &<B as BackendTypes>::Device,
) -> Tensor<B, 1> {
    let len = values.len();
    Tensor::<B, 1>::from_data(TensorData::new(values, [len]), (device, DType::F32))
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
        (
            "backend_precision_policy",
            backend_precision_policy(args, backend_label),
        ),
        (
            "strict_reference_parity_supported",
            strict_reference_parity_supported(args, backend_label).to_string(),
        ),
        (
            "strict_reference_parity_note",
            strict_reference_parity_note(args, backend_label),
        ),
        ("runtime_profile", args.runtime_profile.as_str().to_string()),
        ("precision", format!("{:?}", args.precision).to_lowercase()),
        ("seed", args.seed.to_string()),
        ("steps", args.steps.to_string()),
        ("guidance_scale", args.guidance_scale.to_string()),
        ("shift", args.shift.to_string()),
        ("num_gaussians", args.gaussians.to_string()),
        (
            "cfg_mode",
            format!("{:?}", effective_cfg_mode(args)).to_lowercase(),
        ),
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
        (
            "flow_query_chunk_tokens",
            args.flow_query_chunk_tokens
                .map(|value| value.to_string())
                .unwrap_or_else(|| "default".to_string()),
        ),
        (
            "flow_compute_dtype",
            args.flow_compute_dtype
                .map(|dtype| format!("{dtype:?}").to_lowercase())
                .unwrap_or_else(|| "default".to_string()),
        ),
        (
            "encode_dino_compute_dtype",
            args.encode_dino_compute_dtype
                .map(|dtype| format!("{dtype:?}").to_lowercase())
                .unwrap_or_else(|| "default".to_string()),
        ),
        (
            "encode_vae_compute_dtype",
            args.encode_vae_compute_dtype
                .map(|dtype| format!("{dtype:?}").to_lowercase())
                .unwrap_or_else(|| "default".to_string()),
        ),
        ("flow_warmup_steps", args.flow_warmup_steps.to_string()),
        ("flow_timing_repeats", args.flow_timing_repeats.to_string()),
    ]
    .into_iter()
    .map(|(key, value)| (key.to_string(), value))
    .collect()
}

fn backend_precision_policy(args: &Args, backend: &str) -> String {
    let backend = backend.to_ascii_lowercase();
    let precision = format!("{:?}", args.precision).to_ascii_lowercase();
    let flow_compute_dtype = args.flow_compute_dtype.map(|dtype| format!("{dtype:?}"));
    let encode_dino_compute_dtype = args
        .encode_dino_compute_dtype
        .map(|dtype| format!("{dtype:?}"));
    let encode_vae_compute_dtype = args
        .encode_vae_compute_dtype
        .map(|dtype| format!("{dtype:?}"));
    if backend.contains("cuda") && matches!(args.precision, PrecisionArg::F32) {
        return "cuda-f32-cubecl-accelerated-matmul-may-use-tf32-stage-operands".to_string();
    }
    if backend.contains("wgpu") && matches!(args.precision, PrecisionArg::F32) {
        if matches!(flow_compute_dtype.as_deref(), Some("F16") | Some("BF16")) {
            return "wgpu-f32-weights-with-non-f32-flow-compute-dtype".to_string();
        }
        if matches!(
            encode_dino_compute_dtype.as_deref(),
            Some("F16") | Some("BF16")
        ) {
            return "wgpu-f32-flow-with-non-f32-encoder-compute-dtype".to_string();
        }
        if matches!(
            encode_vae_compute_dtype.as_deref(),
            Some("F16") | Some("BF16")
        ) {
            return "wgpu-f32-dino-flow-with-non-f32-flux-vae-compute-dtype".to_string();
        }
        return "wgpu-f32-strict-no-tf32-public-burn-primitives".to_string();
    }
    if backend.contains("wgpu") && args.runtime_profile == RuntimeProfileArg::Fast {
        return format!(
            "wgpu-{precision}-fast-f32-dino-f16-flux-vae-flow-decoder-blackbox-padded-attention"
        );
    }
    if backend.contains("wgpu") {
        return format!("wgpu-{precision}-non-f32-reference-mode");
    }
    if backend.contains("ndarray") {
        return format!("ndarray-{precision}-cpu-reference-mode");
    }
    format!("{backend}-{precision}-unspecified-precision-policy")
}

fn strict_reference_parity_supported(args: &Args, backend: &str) -> bool {
    let backend = backend.to_ascii_lowercase();
    if backend.contains("cuda") {
        return false;
    }
    if backend.contains("wgpu") {
        return matches!(args.precision, PrecisionArg::F32)
            && !matches!(
                args.flow_compute_dtype,
                Some(ComputeDtypeArg::F16 | ComputeDtypeArg::Bf16)
            )
            && !matches!(
                args.encode_dino_compute_dtype,
                Some(ComputeDtypeArg::F16 | ComputeDtypeArg::Bf16)
            )
            && !matches!(
                args.encode_vae_compute_dtype,
                Some(ComputeDtypeArg::F16 | ComputeDtypeArg::Bf16)
            );
    }
    backend.contains("ndarray") && matches!(args.precision, PrecisionArg::F32)
}

fn strict_reference_parity_note(args: &Args, backend: &str) -> String {
    let backend = backend.to_ascii_lowercase();
    if backend.contains("cuda") {
        return "Burn/CubeCL CUDA accelerated f32 matmul may select TF32 tile operands, while the upstream TripoSplat reference is captured with torch TF32 disabled.".to_string();
    }
    if backend.contains("wgpu") && strict_reference_parity_supported(args, &backend) {
        return "WGPU f32 is the current strict TripoSplat Burn parity path; attention uses public Burn primitives for the long dense f32 TripoSplat shape.".to_string();
    }
    if backend.contains("wgpu") {
        return "Non-f32 WGPU modes are performance/quality experiments and are not strict no-TF32 Python parity references.".to_string();
    }
    if backend.contains("ndarray") && strict_reference_parity_supported(args, &backend) {
        return "NdArray f32 is CPU-reference-capable but not a GPU performance path.".to_string();
    }
    "Strict TripoSplat reference parity has not been established for this backend/precision combination.".to_string()
}

fn forward_trace_metadata(args: &Args) -> std::collections::HashMap<String, String> {
    [
        ("format", "triposplat_rust_forward_trace_v1".to_string()),
        ("precision", format!("{:?}", args.precision).to_lowercase()),
        ("seed", args.seed.to_string()),
        ("steps", args.steps.to_string()),
        ("guidance_scale", args.guidance_scale.to_string()),
        ("shift", args.shift.to_string()),
        (
            "cfg_mode",
            format!("{:?}", effective_cfg_mode(args)).to_lowercase(),
        ),
        ("forward_trace_step", args.forward_trace_step.to_string()),
        (
            "forward_trace_tokens",
            args.forward_trace_tokens.to_string(),
        ),
    ]
    .into_iter()
    .map(|(key, value)| (key.to_string(), value))
    .collect()
}

fn attention_qkv_metadata(
    args: &Args,
    label: String,
    q_shape: Vec<usize>,
    k_shape: Vec<usize>,
    v_shape: Vec<usize>,
) -> std::collections::HashMap<String, String> {
    [
        ("format", "triposplat_attention_qkv_v1".to_string()),
        ("precision", format!("{:?}", args.precision).to_lowercase()),
        ("backend", format!("{:?}", args.backend).to_lowercase()),
        ("seed", args.seed.to_string()),
        ("steps", args.steps.to_string()),
        ("guidance_scale", args.guidance_scale.to_string()),
        ("shift", args.shift.to_string()),
        (
            "cfg_mode",
            format!("{:?}", effective_cfg_mode(args)).to_lowercase(),
        ),
        ("attention_label", label),
        ("attention_label_filter", args.attention_qkv_label.clone()),
        ("layout", "B,T,H,D".to_string()),
        ("q_shape", format!("{q_shape:?}")),
        ("k_shape", format!("{k_shape:?}")),
        ("v_shape", format!("{v_shape:?}")),
        (
            "query_chunk_tokens",
            args.profile_query_chunk_tokens
                .map(|value| value.to_string())
                .unwrap_or_else(|| "default".to_string()),
        ),
    ]
    .into_iter()
    .map(|(key, value)| (key.to_string(), value))
    .collect()
}

use std::any::TypeId;
use std::sync::Mutex;
#[cfg(target_arch = "wasm32")]
use std::sync::mpsc::TryRecvError;
use std::sync::mpsc::{self, Receiver, Sender};
#[cfg(not(target_arch = "wasm32"))]
use std::thread;
#[cfg(target_arch = "wasm32")]
use std::time::Duration;
use std::time::Instant;

use bevy::prelude::*;
use burn::prelude::*;
use burn::tensor::module::interpolate;
use burn::tensor::ops::{InterpolateMode, InterpolateOptions};

use burn_foreground::pipeline::{
    PrepareImageConfig, PreparedImageData, RmbgPipeline, prepare_image_data,
    prepare_image_data_from_bytes, prepare_image_tensor, prepare_image_tensor_from_bytes,
};
use burn_foreground::rmbg2::Rmbg2Pipeline;
use burn_foreground::rmbg14::import::{
    load_rmbg_config_from_json_bytes, load_rmbg_from_burnpack_bytes,
    load_rmbg_processor_from_json_bytes,
};
use burn_trellis::config::TrellisQuality as TrellisRuntimeQuality;
use burn_trellis::pipeline::{
    Trellis2Pipeline, Trellis2PipelineConfig, TrellisDevice, TrellisRunOptions,
};
use burn_tripo::model::triposg::dit::TripoSGDiTConfig;
use burn_tripo::model::triposg::dit::import::load_triposg_dit_from_burnpack_bytes;
use burn_tripo::model::triposg::image_encoder::import::{
    default_dinov2_config, load_dinov2_processor_from_json_bytes,
    load_triposg_dinov2_from_burnpack_bytes,
};
use burn_tripo::model::triposg::image_encoder::{DinoImageProcessor, TripoSGImageEncoder};
use burn_tripo::model::triposg::scheduler::RectifiedFlowSchedulerConfig;
use burn_tripo::model::triposg::vae::TripoSGVaeConfig;
use burn_tripo::model::triposg::vae::import::load_triposg_vae_from_burnpack_bytes;
use burn_tripo::pipeline::{
    geometry::{
        FlashExtractConfig, HierarchicalExtractConfig, flash_extract_geometry,
        hierarchical_extract_geometry,
    },
    mesh::{DenseGrid, Mesh as TripoMesh, grid_to_mesh, sdf_to_mesh_diff_dmc},
    runtime_parity::{
        DinoBackendChoice as SharedDinoBackendChoice, configure_triposg_parity_env,
        decimate_tripo_mesh, resolve_dino_backend,
    },
    triposg::TripoSGPipeline,
    triposg_scribble::TripoSGScribblePipeline,
};

use crate::SynthMesh;
use crate::args::TrellisQuality;
use crate::args::{
    AppArgs, BackendKind, DEFAULT_CHUNK_SIZE, DinoBackend, MeshMode, RmbgBackend, RmbgModel,
    SynthesisModel,
};
#[cfg(not(target_arch = "wasm32"))]
use crate::io::load_text_embeds;
#[cfg(target_arch = "wasm32")]
use crate::model_loader::{
    burnpack_manifest_candidates, candidate_burnpack_names, parse_shard_manifest_bytes,
    prefer_f16_burnpack, resolve_manifest_entry_uri,
};
#[cfg(not(target_arch = "wasm32"))]
use crate::model_loader::{
    load_burnpack_asset_from_root, load_optional_text_candidates_from_root,
    load_optional_text_from_root,
};
#[cfg(not(target_arch = "wasm32"))]
use crate::paths::resolve_scribble_root;
use crate::paths::{resolve_rmbg_root, resolve_triposg_root};
use crate::state::{InferenceRequest, InferenceWorker, WorkerCommand, WorkerEvent};
#[cfg(target_arch = "wasm32")]
use js_sys::Uint8Array;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::{JsFuture, spawn_local};
#[cfg(target_arch = "wasm32")]
use web_sys::{ReadableStreamDefaultReader, Response};

#[path = "worker_unavailable.rs"]
mod worker_unavailable;
#[cfg(any(target_arch = "wasm32", not(feature = "wgpu"), not(feature = "cuda")))]
use worker_unavailable::worker_loop_backend_unavailable;

const WGPU_CHUNK_SIZE_CAP: usize = 8_192;
const CUDA_CHUNK_SIZE_CAP: usize = 32_768;
const DINO_CONFIG_RELPATHS: [&str; 3] = [
    "image_encoder_dinov2/config.json",
    "image_encoder_2/config.json",
    "image_encoder_1/config.json",
];
const DINO_PREPROCESSOR_RELPATHS: [&str; 3] = [
    "feature_extractor_dinov2/preprocessor_config.json",
    "feature_extractor_2/preprocessor_config.json",
    "feature_extractor_1/preprocessor_config.json",
];
#[cfg(any(target_arch = "wasm32", test))]
const ONE_GIB: u64 = 1024 * 1024 * 1024;
#[cfg(any(target_arch = "wasm32", test))]
const DEFAULT_WEB_MAX_HOST_RAM_BYTES: u64 = 4 * ONE_GIB;

struct Rmbg14Artifacts {
    burnpack: Vec<u8>,
    config_json: Option<String>,
    processor_json: Option<String>,
}

#[cfg(not(target_arch = "wasm32"))]
struct TriposgArtifacts {
    vae_burnpack: Vec<u8>,
    dit_burnpack: Vec<u8>,
    dino_burnpack: Vec<u8>,
    vae_config_json: Option<String>,
    dit_config_json: Option<String>,
    scheduler_config_json: Option<String>,
    dino_config_json: Option<String>,
    dino_preproc_json: Option<String>,
}

#[cfg(any(target_arch = "wasm32", test))]
#[derive(Debug, Clone)]
struct WasmHostMemoryBudget {
    limit_bytes: u64,
    retained_bytes: u64,
    peak_bytes: u64,
}

#[cfg(any(target_arch = "wasm32", test))]
impl WasmHostMemoryBudget {
    fn new(limit_bytes: u64) -> Self {
        Self {
            limit_bytes: limit_bytes.max(1),
            retained_bytes: 0,
            peak_bytes: 0,
        }
    }

    fn reserve_retained(&mut self, bytes: u64, context: &str) -> Result<(), String> {
        self.retained_bytes = self.retained_bytes.saturating_add(bytes);
        self.observe_total(self.retained_bytes, context)
    }

    fn release_retained(&mut self, bytes: u64) {
        self.retained_bytes = self.retained_bytes.saturating_sub(bytes);
    }

    fn observe_temporary(&mut self, temporary_bytes: u64, context: &str) -> Result<(), String> {
        let total = self.retained_bytes.saturating_add(temporary_bytes);
        self.observe_total(total, context)
    }

    fn observe_total(&mut self, total_bytes: u64, context: &str) -> Result<(), String> {
        self.peak_bytes = self.peak_bytes.max(total_bytes);
        if total_bytes > self.limit_bytes {
            return Err(format!(
                "wasm host RAM budget exceeded while {context}: {} used (limit {}). \
set build-time BURN_SYNTH_WEB_MAX_HOST_RAM_BYTES to raise",
                format_mebibytes(total_bytes),
                format_mebibytes(self.limit_bytes)
            ));
        }
        Ok(())
    }

    fn peak_bytes(&self) -> u64 {
        self.peak_bytes
    }

    fn limit_bytes(&self) -> u64 {
        self.limit_bytes
    }
}

fn synthesis_attempt_order(models: &[SynthesisModel]) -> Result<Vec<SynthesisModel>, String> {
    let mut order = Vec::with_capacity(models.len());
    for model in models.iter().copied() {
        if !order.contains(&model) {
            order.push(model);
        }
    }
    if order.is_empty() {
        Err("No synthesis backend selected.".to_string())
    } else {
        Ok(order)
    }
}

fn synthesis_model_name(model: SynthesisModel) -> &'static str {
    match model {
        SynthesisModel::Triposg => "TripoSG",
        SynthesisModel::Trellis => "Trellis",
    }
}

fn map_dino_backend_choice(value: DinoBackend) -> SharedDinoBackendChoice {
    match value {
        DinoBackend::Auto => SharedDinoBackendChoice::Auto,
        DinoBackend::Cpu => SharedDinoBackendChoice::Cpu,
        DinoBackend::Gpu => SharedDinoBackendChoice::Gpu,
    }
}

fn map_dino_backend_choice_back(value: SharedDinoBackendChoice) -> DinoBackend {
    match value {
        SharedDinoBackendChoice::Auto => DinoBackend::Auto,
        SharedDinoBackendChoice::Cpu => DinoBackend::Cpu,
        SharedDinoBackendChoice::Gpu => DinoBackend::Gpu,
    }
}

fn synthesis_unavailable_message(
    triposg_load_error: Option<&str>,
    trellis_load_error: Option<&str>,
) -> String {
    let triposg = match triposg_load_error {
        Some(err) => format!("TripoSG: {err}"),
        None => "TripoSG: disabled".to_string(),
    };
    let trellis = match trellis_load_error {
        Some(err) => format!("Trellis: {err}"),
        None => "Trellis: disabled".to_string(),
    };
    format!("No synthesis backend is available. {triposg}. {trellis}.")
}

#[cfg(not(target_arch = "wasm32"))]
pub fn start_worker(args: &AppArgs) -> InferenceWorker {
    let (command_tx, command_rx) = mpsc::channel();
    let (event_tx, event_rx) = mpsc::channel();
    let args = args.clone();
    let _ = thread::Builder::new()
        .name("synth-worker".to_string())
        .stack_size(64 * 1024 * 1024)
        .spawn(move || worker_loop(args, command_rx, event_tx))
        .expect("failed to spawn synth worker thread");
    InferenceWorker {
        sender: command_tx,
        receiver: Mutex::new(event_rx),
    }
}

#[cfg(target_arch = "wasm32")]
pub fn start_worker(args: &AppArgs) -> InferenceWorker {
    let (command_tx, command_rx) = mpsc::channel();
    let (event_tx, event_rx) = mpsc::channel();
    let args = args.clone();
    spawn_local(async move {
        worker_loop_wasm(args, command_rx, event_tx).await;
    });
    InferenceWorker {
        sender: command_tx,
        receiver: Mutex::new(event_rx),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn worker_loop(args: AppArgs, command_rx: Receiver<WorkerCommand>, event_tx: Sender<WorkerEvent>) {
    match args.backend {
        BackendKind::Cpu => {
            worker_loop_backend::<burn::backend::NdArray<f32>>(args, command_rx, event_tx)
        }
        BackendKind::Wgpu => {
            #[cfg(feature = "wgpu")]
            {
                worker_loop_backend::<burn_wgpu::Wgpu>(args, command_rx, event_tx);
            }
            #[cfg(not(feature = "wgpu"))]
            {
                worker_loop_backend_unavailable(
                    command_rx,
                    event_tx,
                    "wgpu backend not enabled (enable the `wgpu` feature)",
                );
            }
        }
        BackendKind::Cuda => {
            #[cfg(feature = "cuda")]
            {
                worker_loop_backend::<burn_cuda::Cuda>(args, command_rx, event_tx);
            }
            #[cfg(not(feature = "cuda"))]
            {
                worker_loop_backend_unavailable(
                    command_rx,
                    event_tx,
                    "cuda backend not enabled (enable the `cuda` feature)",
                );
            }
        }
    }
}

#[cfg(target_arch = "wasm32")]
async fn worker_loop_wasm(
    args: AppArgs,
    command_rx: Receiver<WorkerCommand>,
    event_tx: Sender<WorkerEvent>,
) {
    match args.backend {
        BackendKind::Cpu => {
            worker_loop_backend_wasm::<burn::backend::NdArray<f32>>(args, command_rx, event_tx)
                .await;
        }
        BackendKind::Wgpu => {
            #[cfg(feature = "wgpu")]
            {
                worker_loop_backend_wasm::<burn_wgpu::Wgpu>(args, command_rx, event_tx).await;
            }
            #[cfg(not(feature = "wgpu"))]
            {
                worker_loop_backend_unavailable(
                    command_rx,
                    event_tx,
                    "wgpu backend not enabled (enable the `wgpu` feature)",
                );
            }
        }
        BackendKind::Cuda => {
            worker_loop_backend_unavailable(
                command_rx,
                event_tx,
                "cuda backend is unavailable on wasm32",
            );
        }
    }
}

struct PipelineState<B: Backend> {
    device: B::Device,
    rmbg14_cpu: Option<RmbgPipeline<burn::backend::NdArray<f32>>>,
    rmbg14_device: Option<RmbgPipeline<B>>,
    rmbg2: Option<Rmbg2Pipeline>,
    rmbg_model: RmbgModel,
    rmbg_backend: RmbgBackend,
    dino_backend: DinoBackend,
    dino_cpu: Option<DinoCpuState>,
    triposg: Option<TripoSGPipeline<B>>,
    scribble: Option<TripoSGScribblePipeline<B>>,
    trellis: Option<Trellis2Pipeline>,
    trellis_load_error: Option<String>,
    synthesis_models: Vec<SynthesisModel>,
    triposg_load_error: Option<String>,
    text_embeds: Option<Tensor<B, 3>>,
    bounds: [f32; 6],
    hierarchical: HierarchicalExtractConfig,
    chunk_size: usize,
    flash: FlashExtractConfig,
}

struct DinoCpuState {
    device: <burn::backend::NdArray<f32> as Backend>::Device,
    encoder: TripoSGImageEncoder<burn::backend::NdArray<f32>>,
    processor: DinoImageProcessor,
}

#[cfg(not(target_arch = "wasm32"))]
type Rmbg14Pipelines<B> = (
    Option<RmbgPipeline<burn::backend::NdArray<f32>>>,
    Option<RmbgPipeline<B>>,
);

#[cfg(not(target_arch = "wasm32"))]
fn load_rmbg14_pipelines<B: Backend>(
    rmbg_root: &std::path::Path,
    rmbg_backend: RmbgBackend,
    device: &B::Device,
) -> Result<Rmbg14Pipelines<B>, String> {
    let artifacts = load_rmbg14_artifacts_from_root(rmbg_root)?;
    let rmbg_config = if let Some(json) = artifacts.config_json.as_ref() {
        load_rmbg_config_from_json_bytes(json.as_bytes())
            .map_err(|err| format!("failed to parse RMBG config: {err}"))?
    } else {
        burn_foreground::rmbg14::RmbgConfig::rmbg_1_4()
    };
    let rmbg_processor = if let Some(json) = artifacts.processor_json.as_ref() {
        load_rmbg_processor_from_json_bytes(json.as_bytes())
            .map_err(|err| format!("failed to parse RMBG preprocessor config: {err}"))?
    } else {
        burn_foreground::preprocess::RmbgImageProcessor::default()
    };
    let rmbg_burnpack = artifacts.burnpack;

    match rmbg_backend {
        RmbgBackend::Cpu => {
            let cpu_device = <burn::backend::NdArray<f32> as Backend>::Device::default();
            let model = load_rmbg_from_burnpack_bytes(&cpu_device, rmbg_burnpack, &rmbg_config)
                .map_err(|err| format!("failed to load RMBG burnpack on CPU: {err}"))?;
            let rmbg = RmbgPipeline::new(model, rmbg_processor);
            Ok((Some(rmbg), None))
        }
        RmbgBackend::Gpu | RmbgBackend::Auto => {
            let model = load_rmbg_from_burnpack_bytes(device, rmbg_burnpack, &rmbg_config)
                .map_err(|err| format!("failed to load RMBG burnpack: {err}"))?;
            let mut rmbg = RmbgPipeline::new(model, rmbg_processor);
            cap_rmbg14_processor_for_backend::<B>(&mut rmbg, rmbg_backend);
            Ok((None, Some(rmbg)))
        }
    }
}

fn load_trellis_pipeline(args: &AppArgs) -> Result<Trellis2Pipeline, String> {
    let mut config = Trellis2PipelineConfig::default();
    if let Some(path) = args.trellis_weights_root.as_ref() {
        config.weights_root = path.clone();
    }
    if let Some(path) = args.trellis_image_large_root.as_ref() {
        config.image_large_root = Some(path.clone());
    }

    let pipeline = Trellis2Pipeline::new(config)
        .map_err(|err| format!("failed to initialize Trellis2: {err}"))?;
    pipeline
        .validate_runtime()
        .map_err(|err| format!("Trellis2 runtime unavailable: {err}"))?;
    Ok(pipeline)
}

#[cfg(not(target_arch = "wasm32"))]
fn load_triposg_pipeline<B: Backend>(
    args: &AppArgs,
    device: &B::Device,
    dino_backend: DinoBackend,
) -> Result<(Option<DinoCpuState>, TripoSGPipeline<B>), String> {
    let weights_root = resolve_triposg_root(args.weights_root.as_ref());
    let artifacts = load_triposg_artifacts_from_root(&weights_root)?;
    build_triposg_pipeline_from_artifacts::<B>(device, dino_backend, artifacts)
}

#[cfg(not(target_arch = "wasm32"))]
fn load_rmbg14_artifacts_from_root(root: &std::path::Path) -> Result<Rmbg14Artifacts, String> {
    let burnpack = load_burnpack_asset_from_root(root, "model.safetensors", "RMBG_BPK_PRECISION")?;
    let config_json = load_optional_text_from_root(root, "config.json")?;
    let processor_json = load_optional_text_from_root(root, "preprocessor_config.json")?;
    Ok(Rmbg14Artifacts {
        burnpack,
        config_json,
        processor_json,
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn load_triposg_artifacts_from_root(root: &std::path::Path) -> Result<TriposgArtifacts, String> {
    let vae_burnpack = load_burnpack_asset_from_root(
        root,
        "vae/diffusion_pytorch_model.safetensors",
        "TRIPOSG_BPK_PRECISION",
    )?;
    let dit_burnpack = load_burnpack_asset_from_root(
        root,
        "transformer/diffusion_pytorch_model.safetensors",
        "TRIPOSG_BPK_PRECISION",
    )?;
    let dino_burnpack = load_burnpack_asset_from_root(
        root,
        "image_encoder_dinov2/model.safetensors",
        "TRIPOSG_BPK_PRECISION",
    )?;
    let vae_config_json = load_optional_text_from_root(root, "vae/config.json")?;
    let dit_config_json = load_optional_text_from_root(root, "transformer/config.json")?;
    let scheduler_config_json =
        load_optional_text_from_root(root, "scheduler/scheduler_config.json")?;
    let dino_config_json = load_optional_text_candidates_from_root(root, &DINO_CONFIG_RELPATHS)?;
    let dino_preproc_json =
        load_optional_text_candidates_from_root(root, &DINO_PREPROCESSOR_RELPATHS)?;

    Ok(TriposgArtifacts {
        vae_burnpack,
        dit_burnpack,
        dino_burnpack,
        vae_config_json,
        dit_config_json,
        scheduler_config_json,
        dino_config_json,
        dino_preproc_json,
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn build_triposg_pipeline_from_artifacts<B: Backend>(
    device: &B::Device,
    dino_backend: DinoBackend,
    artifacts: TriposgArtifacts,
) -> Result<(Option<DinoCpuState>, TripoSGPipeline<B>), String> {
    let vae_config = if let Some(json) = artifacts.vae_config_json.as_ref() {
        TripoSGVaeConfig::from_config_bytes(json.as_bytes())
            .map_err(|err| format!("failed to parse TripoSG VAE config: {err}"))?
    } else {
        TripoSGVaeConfig::midi_3d()
    };
    let dit_config = if let Some(json) = artifacts.dit_config_json.as_ref() {
        TripoSGDiTConfig::from_config_bytes(json.as_bytes())
            .map_err(|err| format!("failed to parse TripoSG DiT config: {err}"))?
    } else {
        TripoSGDiTConfig::triposg_pretrained()
    };
    let scheduler_config = if let Some(json) = artifacts.scheduler_config_json.as_ref() {
        RectifiedFlowSchedulerConfig::from_config_bytes(json.as_bytes())
            .map_err(|err| format!("failed to parse scheduler config: {err}"))?
    } else {
        RectifiedFlowSchedulerConfig::midi_3d()
    };

    let parsed_dino_config = artifacts.dino_config_json.as_ref().and_then(|json| {
        burn_tripo::model::triposg::image_encoder::import::load_dinov2_config_from_json_bytes(
            json.as_bytes(),
        )
    });
    let dino_fallback_size = parsed_dino_config.as_ref().map(|cfg| cfg.image_size);
    let mut dino_config = parsed_dino_config
        .clone()
        .unwrap_or_else(default_dinov2_config);
    if let Some(size) = artifacts.dino_preproc_json.as_ref().and_then(|json| {
        burn_tripo::model::triposg::image_encoder::import::load_dinov2_preprocess_size_from_json_bytes(
            json.as_bytes(),
        )
    }) {
        let patch = dino_config.patch_size.max(1);
        let grid = size / patch;
        if grid > 0 {
            dino_config.positional_encoding_interpolate.output_size = Some([grid, grid]);
        }
    }
    let dino_processor = if let Some(json) = artifacts.dino_preproc_json.as_ref() {
        load_dinov2_processor_from_json_bytes(json.as_bytes(), dino_fallback_size)
            .map_err(|err| format!("failed to parse DINO preprocessor config: {err}"))?
    } else {
        DinoImageProcessor::default()
    };

    let dino_burnpack_for_cpu = if matches!(dino_backend, DinoBackend::Cpu) {
        Some(artifacts.dino_burnpack.clone())
    } else {
        None
    };

    let vae = load_triposg_vae_from_burnpack_bytes(&vae_config, device, artifacts.vae_burnpack)
        .map_err(|err| format!("failed to load TripoSG VAE burnpack: {err}"))?;
    let dit = load_triposg_dit_from_burnpack_bytes(&dit_config, device, artifacts.dit_burnpack)
        .map_err(|err| format!("failed to load TripoSG DiT burnpack: {err}"))?;
    let image_encoder =
        load_triposg_dinov2_from_burnpack_bytes(device, dino_config, artifacts.dino_burnpack)
            .map_err(|err| format!("failed to load DINOv2 burnpack: {err}"))?;
    let scheduler = scheduler_config.init();
    let triposg = TripoSGPipeline::new(vae, dit, scheduler, image_encoder, dino_processor.clone());

    let dino_cpu = if matches!(dino_backend, DinoBackend::Cpu) {
        let cpu_device = <burn::backend::NdArray<f32> as Backend>::Device::default();
        let cpu_config = parsed_dino_config.unwrap_or_else(default_dinov2_config);
        let encoder = load_triposg_dinov2_from_burnpack_bytes(
            &cpu_device,
            cpu_config,
            dino_burnpack_for_cpu.expect("cpu dino burnpack must exist"),
        )
        .map_err(|err| format!("failed to load DINOv2 burnpack on CPU: {err}"))?;
        Some(DinoCpuState {
            device: cpu_device,
            encoder,
            processor: dino_processor,
        })
    } else {
        None
    };

    Ok((dino_cpu, triposg))
}

fn trellis_quality_to_runtime(quality: TrellisQuality) -> TrellisRuntimeQuality {
    match quality {
        TrellisQuality::Low => TrellisRuntimeQuality::Low,
        TrellisQuality::Medium => TrellisRuntimeQuality::Medium,
        TrellisQuality::High => TrellisRuntimeQuality::High,
    }
}

fn trellis_device_for_backend(backend: BackendKind) -> TrellisDevice {
    match backend {
        BackendKind::Cpu => TrellisDevice::Cpu,
        BackendKind::Wgpu => TrellisDevice::Wgpu,
        BackendKind::Cuda => TrellisDevice::Cuda,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn worker_loop_backend<B: Backend>(
    args: AppArgs,
    command_rx: Receiver<WorkerCommand>,
    event_tx: Sender<WorkerEvent>,
) {
    let mut state = build_pipeline_state::<B>(&args);
    if let Err(err) = state.as_ref() {
        warn!("Inference worker failed to initialize: {err}");
    }
    for command in command_rx {
        match command {
            WorkerCommand::Infer(requests) => {
                let start = Instant::now();
                let results = match state.as_mut() {
                    Ok(state) => run_inference_with_state(state, &args, &requests),
                    Err(err) => vec![Err(err.clone()); requests.len()],
                };
                let _ = event_tx.send(WorkerEvent {
                    requests,
                    results,
                    elapsed: start.elapsed(),
                    status_message: None,
                });
            }
            WorkerCommand::Shutdown => break,
        }
    }
}

#[cfg(target_arch = "wasm32")]
async fn worker_loop_backend_wasm<B: Backend>(
    args: AppArgs,
    command_rx: Receiver<WorkerCommand>,
    event_tx: Sender<WorkerEvent>,
) {
    let mut state = build_pipeline_state_wasm::<B>(&args, &event_tx).await;
    if let Err(err) = state.as_ref() {
        warn!("Inference worker failed to initialize: {err}");
        let _ = event_tx.send(WorkerEvent {
            requests: Vec::new(),
            results: Vec::new(),
            elapsed: Duration::ZERO,
            status_message: Some(format!("Model load failed: {err}")),
        });
    } else {
        let _ = event_tx.send(WorkerEvent {
            requests: Vec::new(),
            results: Vec::new(),
            elapsed: Duration::ZERO,
            status_message: Some("Model weights ready.".to_string()),
        });
    }

    loop {
        match command_rx.try_recv() {
            Ok(WorkerCommand::Infer(requests)) => {
                let start = Instant::now();
                let results = match state.as_mut() {
                    Ok(state) => run_inference_with_state(state, &args, &requests),
                    Err(err) => vec![Err(err.clone()); requests.len()],
                };
                let _ = event_tx.send(WorkerEvent {
                    requests,
                    results,
                    elapsed: start.elapsed(),
                    status_message: None,
                });
            }
            Ok(WorkerCommand::Shutdown) => break,
            Err(TryRecvError::Empty) => {
                gloo_timers::future::TimeoutFuture::new(8).await;
            }
            Err(TryRecvError::Disconnected) => break,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn build_pipeline_state<B: Backend>(args: &AppArgs) -> Result<PipelineState<B>, String> {
    configure_cubecl_autotune::<B>();
    let bounds = parse_bounds(&args.bounds).map_err(|err| err.to_string())?;
    let device = B::Device::default();
    if let Some(seed) = args.seed {
        B::seed(&device, seed);
    }

    configure_triposg_parity_env(args.match_python, None);

    let synthesis_models = args.synthesis_models.clone();
    let synthesis_order = synthesis_attempt_order(&synthesis_models)?;
    let wants_triposg = synthesis_order
        .iter()
        .any(|model| matches!(model, SynthesisModel::Triposg));

    let dino_backend = map_dino_backend_choice_back(resolve_dino_backend::<B>(
        map_dino_backend_choice(args.dino_backend),
        args.match_python,
    ));

    let chunk_size = tuned_chunk_size::<B>(args.chunk_size);
    let hierarchical = HierarchicalExtractConfig {
        bounds,
        dense_octree_depth: args.dense_octree_depth,
        hierarchical_octree_depth: args.hierarchical_octree_depth,
        chunk_size,
        band_threshold: args.band_threshold,
    };
    let flash = FlashExtractConfig {
        bounds,
        octree_depth: args.flash_octree_depth,
        num_chunks: args.flash_num_chunks,
        mc_level: args.flash_mc_level,
        min_resolution: args.flash_min_resolution,
        mini_grid_num: args.flash_mini_grid_num,
    };
    let requested_rmbg_model = args.rmbg_model;
    let rmbg_backend = match args.rmbg_backend {
        RmbgBackend::Auto => RmbgBackend::Gpu,
        other => other,
    };

    let mut active_rmbg_model = requested_rmbg_model;
    let (rmbg14_cpu, rmbg14_device, rmbg2) = match requested_rmbg_model {
        RmbgModel::Rmbg14 => {
            let rmbg14_root = resolve_rmbg_root(args.bg_weights_root.as_ref(), RmbgModel::Rmbg14);
            let (cpu, device_pipeline) =
                load_rmbg14_pipelines::<B>(&rmbg14_root, rmbg_backend, &device)?;
            (cpu, device_pipeline, None)
        }
        RmbgModel::Rmbg2 => {
            let rmbg2_root = resolve_rmbg_root(args.bg_weights_root.as_ref(), RmbgModel::Rmbg2);
            match Rmbg2Pipeline::from_pretrained(&rmbg2_root) {
                Ok(rmbg2) => (None, None, Some(rmbg2)),
                Err(rmbg2_err) => {
                    let rmbg14_root =
                        resolve_rmbg_root(args.bg_weights_root.as_ref(), RmbgModel::Rmbg14);
                    warn!(
                        "RMBG-2.0 unavailable at '{}': {}. Falling back to RMBG-1.4 at '{}'.",
                        rmbg2_root.display(),
                        rmbg2_err,
                        rmbg14_root.display()
                    );
                    let (cpu, device_pipeline) =
                        load_rmbg14_pipelines::<B>(&rmbg14_root, rmbg_backend, &device).map_err(
                            |rmbg14_err| {
                                format!(
                                    "failed to load RMBG-2.0 at {}: {}; fallback RMBG-1.4 at {} also failed: {}",
                                    rmbg2_root.display(),
                                    rmbg2_err,
                                    rmbg14_root.display(),
                                    rmbg14_err
                                )
                            },
                        )?;
                    active_rmbg_model = RmbgModel::Rmbg14;
                    (cpu, device_pipeline, None)
                }
            }
        }
    };

    let wants_text = args.text_embeds.is_some() || args.prompt.is_some();
    if wants_text {
        if !wants_triposg {
            return Err(
                "text/scribble mode currently requires TripoSG; include `triposg` in --synthesis-models"
                    .to_string(),
            );
        }

        let text_path = args
            .text_embeds
            .as_ref()
            .ok_or_else(|| "text prompt provided without --text-embeds".to_string())?;
        let text_embeds = load_text_embeds::<B>(text_path, &args.text_embeds_key, &device)
            .map_err(|err| format!("failed to load text embeddings: {err}"))?;

        let weights_root = resolve_scribble_root(
            args.scribble_weights_root
                .as_ref()
                .or(args.weights_root.as_ref()),
        );
        let scribble = TripoSGScribblePipeline::from_pretrained(weights_root, &device)
            .map_err(|err| format!("failed to load TripoSG-scribble weights: {err}"))?;

        return Ok(PipelineState {
            device,
            rmbg14_cpu,
            rmbg14_device,
            rmbg2,
            rmbg_model: active_rmbg_model,
            rmbg_backend,
            dino_backend,
            dino_cpu: None,
            triposg: None,
            scribble: Some(scribble),
            trellis: None,
            trellis_load_error: None,
            synthesis_models: synthesis_models.clone(),
            triposg_load_error: None,
            text_embeds: Some(text_embeds),
            bounds,
            hierarchical,
            chunk_size,
            flash,
        });
    }

    let mut trellis = None;
    let mut trellis_load_error = None;
    let mut dino_cpu = None;
    let mut triposg = None;
    let mut triposg_load_error = None;
    for (index, model) in synthesis_order.iter().copied().enumerate() {
        let fallback = synthesis_order.get(index + 1).copied();
        match model {
            SynthesisModel::Triposg => {
                match load_triposg_pipeline::<B>(args, &device, dino_backend) {
                    Ok((dino_state, triposg_pipeline)) => {
                        dino_cpu = dino_state;
                        triposg = Some(triposg_pipeline);
                        break;
                    }
                    Err(err) => {
                        triposg_load_error = Some(err.clone());
                        if let Some(next_model) = fallback {
                            warn!(
                                "Requested {} backend unavailable ({}); trying {} fallback.",
                                synthesis_model_name(model),
                                err,
                                synthesis_model_name(next_model)
                            );
                        }
                    }
                }
            }
            SynthesisModel::Trellis => match load_trellis_pipeline(args) {
                Ok(pipeline) => {
                    trellis = Some(pipeline);
                    break;
                }
                Err(err) => {
                    trellis_load_error = Some(err.clone());
                    if let Some(next_model) = fallback {
                        warn!(
                            "Requested {} backend unavailable ({}); trying {} fallback.",
                            synthesis_model_name(model),
                            err,
                            synthesis_model_name(next_model)
                        );
                    }
                }
            },
        }
    }

    if triposg.is_none() && trellis.is_none() {
        return Err(synthesis_unavailable_message(
            triposg_load_error.as_deref(),
            trellis_load_error.as_deref(),
        ));
    }

    Ok(PipelineState {
        device,
        rmbg14_cpu,
        rmbg14_device,
        rmbg2,
        rmbg_model: active_rmbg_model,
        rmbg_backend,
        dino_backend,
        dino_cpu,
        triposg,
        scribble: None,
        trellis,
        trellis_load_error,
        synthesis_models,
        triposg_load_error,
        text_embeds: None,
        bounds,
        hierarchical,
        chunk_size,
        flash,
    })
}

#[cfg(target_arch = "wasm32")]
#[derive(Default)]
struct DownloadTotals {
    known_total: u64,
    known_downloaded: u64,
    unknown_downloaded: u64,
}

#[cfg(target_arch = "wasm32")]
async fn build_pipeline_state_wasm<B: Backend>(
    args: &AppArgs,
    event_tx: &Sender<WorkerEvent>,
) -> Result<PipelineState<B>, String> {
    configure_triposg_parity_env(args.match_python, None);
    configure_cubecl_autotune::<B>();
    let bounds = parse_bounds(&args.bounds).map_err(|err| err.to_string())?;
    let device = B::Device::default();
    if let Some(seed) = args.seed {
        B::seed(&device, seed);
    }

    let synthesis_models = args.synthesis_models.clone();
    let synthesis_order = synthesis_attempt_order(&synthesis_models)?;

    if args.text_embeds.is_some() || args.prompt.is_some() {
        return Err("text/scribble mode is not supported on wasm yet".to_string());
    }

    let dino_backend = map_dino_backend_choice_back(resolve_dino_backend::<B>(
        map_dino_backend_choice(args.dino_backend),
        args.match_python,
    ));
    let chunk_size = tuned_chunk_size::<B>(args.chunk_size);
    let hierarchical = HierarchicalExtractConfig {
        bounds,
        dense_octree_depth: args.dense_octree_depth,
        hierarchical_octree_depth: args.hierarchical_octree_depth,
        chunk_size,
        band_threshold: args.band_threshold,
    };
    let flash = FlashExtractConfig {
        bounds,
        octree_depth: args.flash_octree_depth,
        num_chunks: args.flash_num_chunks,
        mc_level: args.flash_mc_level,
        min_resolution: args.flash_min_resolution,
        mini_grid_num: args.flash_mini_grid_num,
    };
    let rmbg_backend = match args.rmbg_backend {
        RmbgBackend::Auto => RmbgBackend::Gpu,
        other => other,
    };

    let mut totals = DownloadTotals::default();
    let mut host_ram_budget = WasmHostMemoryBudget::new(web_max_host_ram_bytes());
    send_worker_status(event_tx, "Loading model weights...");
    send_worker_status(
        event_tx,
        format!(
            "WASM host RAM budget for model loading: {}",
            format_mebibytes(host_ram_budget.limit_bytes())
        ),
    );

    let mut rmbg_model = args.rmbg_model;
    if matches!(rmbg_model, RmbgModel::Rmbg2) {
        send_worker_status(
            event_tx,
            "RMBG-2.0 ONNX is not available on wasm32, falling back to RMBG-1.4.",
        );
        rmbg_model = RmbgModel::Rmbg14;
    }

    let rmbg_root = resolve_rmbg_root(args.bg_weights_root.as_ref(), rmbg_model);
    let rmbg_root_url = normalize_web_path(&rmbg_root);

    let rmbg_artifacts =
        load_rmbg14_artifacts_wasm(&rmbg_root_url, event_tx, &mut totals, &mut host_ram_budget)
            .await?;
    let rmbg_config = if let Some(json) = rmbg_artifacts.config_json.as_ref() {
        load_rmbg_config_from_json_bytes(json.as_bytes())
            .map_err(|err| format!("failed to parse RMBG config: {err}"))?
    } else {
        burn_foreground::rmbg14::RmbgConfig::rmbg_1_4()
    };
    let rmbg_processor = if let Some(json) = rmbg_artifacts.processor_json.as_ref() {
        load_rmbg_processor_from_json_bytes(json.as_bytes())
            .map_err(|err| format!("failed to parse RMBG preprocessor config: {err}"))?
    } else {
        burn_foreground::preprocess::RmbgImageProcessor::default()
    };
    let rmbg_burnpack = rmbg_artifacts.burnpack;
    let rmbg_burnpack_bytes = rmbg_burnpack.len() as u64;
    host_ram_budget.reserve_retained(rmbg_burnpack_bytes, "retaining RMBG burnpack bytes")?;

    let (rmbg14_cpu, rmbg14_device) = match rmbg_backend {
        RmbgBackend::Cpu => {
            let cpu_device = <burn::backend::NdArray<f32> as Backend>::Device::default();
            let model =
                match load_rmbg_from_burnpack_bytes(&cpu_device, rmbg_burnpack, &rmbg_config) {
                    Ok(model) => model,
                    Err(err) => {
                        host_ram_budget.release_retained(rmbg_burnpack_bytes);
                        return Err(format!("failed to load RMBG burnpack on CPU: {err}"));
                    }
                };
            host_ram_budget.release_retained(rmbg_burnpack_bytes);
            (Some(RmbgPipeline::new(model, rmbg_processor.clone())), None)
        }
        RmbgBackend::Gpu | RmbgBackend::Auto => {
            let model = match load_rmbg_from_burnpack_bytes(&device, rmbg_burnpack, &rmbg_config) {
                Ok(model) => model,
                Err(err) => {
                    host_ram_budget.release_retained(rmbg_burnpack_bytes);
                    return Err(format!("failed to load RMBG burnpack: {err}"));
                }
            };
            host_ram_budget.release_retained(rmbg_burnpack_bytes);
            let mut pipeline = RmbgPipeline::new(model, rmbg_processor);
            cap_rmbg14_processor_for_backend::<B>(&mut pipeline, rmbg_backend);
            (None, Some(pipeline))
        }
    };

    let mut triposg = None;
    let mut trellis = None;
    let mut dino_cpu = None;
    let mut triposg_load_error = None;
    let mut trellis_load_error = None;
    for (index, model) in synthesis_order.iter().copied().enumerate() {
        let fallback = synthesis_order.get(index + 1).copied();
        match model {
            SynthesisModel::Triposg => {
                match load_triposg_pipeline_wasm::<B>(
                    args,
                    &device,
                    dino_backend,
                    event_tx,
                    &mut totals,
                    &mut host_ram_budget,
                )
                .await
                {
                    Ok((dino_state, pipeline)) => {
                        dino_cpu = dino_state;
                        triposg = Some(pipeline);
                        break;
                    }
                    Err(err) => {
                        triposg_load_error = Some(err.clone());
                        if let Some(next_model) = fallback {
                            let log_message = format!(
                                "Requested {} backend unavailable ({}); trying {} fallback.",
                                synthesis_model_name(model),
                                err,
                                synthesis_model_name(next_model)
                            );
                            warn!("{log_message}");
                            send_worker_status(event_tx, log_message);
                        }
                    }
                }
            }
            SynthesisModel::Trellis => match load_trellis_pipeline(args) {
                Ok(pipeline) => {
                    trellis = Some(pipeline);
                    break;
                }
                Err(err) => {
                    trellis_load_error = Some(err.clone());
                    if let Some(next_model) = fallback {
                        let log_message = format!(
                            "Requested {} backend unavailable ({}); trying {} fallback.",
                            synthesis_model_name(model),
                            err,
                            synthesis_model_name(next_model)
                        );
                        warn!("{log_message}");
                        send_worker_status(event_tx, log_message);
                    }
                }
            },
        }
    }
    if triposg.is_none() && trellis.is_none() {
        return Err(synthesis_unavailable_message(
            triposg_load_error.as_deref(),
            trellis_load_error.as_deref(),
        ));
    }

    send_worker_status(
        event_tx,
        format!(
            "Model weights loaded. peak host RAM during load: {}",
            format_mebibytes(host_ram_budget.peak_bytes())
        ),
    );

    Ok(PipelineState {
        device,
        rmbg14_cpu,
        rmbg14_device,
        rmbg2: None,
        rmbg_model,
        rmbg_backend,
        dino_backend,
        dino_cpu,
        triposg,
        scribble: None,
        trellis,
        trellis_load_error,
        synthesis_models,
        triposg_load_error,
        text_embeds: None,
        bounds,
        hierarchical,
        chunk_size,
        flash,
    })
}

#[cfg(target_arch = "wasm32")]
async fn load_triposg_pipeline_wasm<B: Backend>(
    args: &AppArgs,
    device: &B::Device,
    dino_backend: DinoBackend,
    event_tx: &Sender<WorkerEvent>,
    totals: &mut DownloadTotals,
    host_ram_budget: &mut WasmHostMemoryBudget,
) -> Result<(Option<DinoCpuState>, TripoSGPipeline<B>), String> {
    let triposg_root = resolve_triposg_root(args.weights_root.as_ref());
    let triposg_root_url = normalize_web_path(&triposg_root);

    let vae_config_json =
        fetch_optional_text(&join_web_path(&triposg_root_url, "vae/config.json")).await?;
    let dit_config_json =
        fetch_optional_text(&join_web_path(&triposg_root_url, "transformer/config.json")).await?;
    let scheduler_config_json = fetch_optional_text(&join_web_path(
        &triposg_root_url,
        "scheduler/scheduler_config.json",
    ))
    .await?;
    let dino_config_candidates = DINO_CONFIG_RELPATHS
        .iter()
        .map(|rel| join_web_path(&triposg_root_url, rel))
        .collect::<Vec<_>>();
    let dino_config_json = fetch_optional_text_candidates(&dino_config_candidates).await?;
    let dino_preproc_candidates = DINO_PREPROCESSOR_RELPATHS
        .iter()
        .map(|rel| join_web_path(&triposg_root_url, rel))
        .collect::<Vec<_>>();
    let dino_preproc_json = fetch_optional_text_candidates(&dino_preproc_candidates).await?;

    let vae_config = if let Some(json) = vae_config_json.as_ref() {
        TripoSGVaeConfig::from_config_bytes(json.as_bytes())
            .map_err(|err| format!("failed to parse TripoSG VAE config: {err}"))?
    } else {
        TripoSGVaeConfig::midi_3d()
    };
    let dit_config = if let Some(json) = dit_config_json.as_ref() {
        TripoSGDiTConfig::from_config_bytes(json.as_bytes())
            .map_err(|err| format!("failed to parse TripoSG DiT config: {err}"))?
    } else {
        TripoSGDiTConfig::triposg_pretrained()
    };
    let scheduler_config = if let Some(json) = scheduler_config_json.as_ref() {
        RectifiedFlowSchedulerConfig::from_config_bytes(json.as_bytes())
            .map_err(|err| format!("failed to parse scheduler config: {err}"))?
    } else {
        RectifiedFlowSchedulerConfig::midi_3d()
    };
    let parsed_dino_config = dino_config_json.as_ref().and_then(|json| {
        burn_tripo::model::triposg::image_encoder::import::load_dinov2_config_from_json_bytes(
            json.as_bytes(),
        )
    });
    let dino_fallback_size = parsed_dino_config.as_ref().map(|cfg| cfg.image_size);
    let mut dino_config = parsed_dino_config
        .clone()
        .unwrap_or_else(default_dinov2_config);
    if let Some(size) = dino_preproc_json.as_ref().and_then(|json| {
        burn_tripo::model::triposg::image_encoder::import::load_dinov2_preprocess_size_from_json_bytes(
            json.as_bytes(),
        )
    }) {
        let patch = dino_config.patch_size.max(1);
        let grid = size / patch;
        if grid > 0 {
            dino_config.positional_encoding_interpolate.output_size = Some([grid, grid]);
        }
    }
    let dino_processor = if let Some(json) = dino_preproc_json.as_ref() {
        load_dinov2_processor_from_json_bytes(json.as_bytes(), dino_fallback_size)
            .map_err(|err| format!("failed to parse DINO preprocessor config: {err}"))?
    } else {
        DinoImageProcessor::default()
    };

    let vae_burnpack = download_burnpack_asset(
        &join_web_path(&triposg_root_url, "vae/diffusion_pytorch_model.safetensors"),
        "TripoSG VAE",
        "TRIPOSG_BPK_PRECISION",
        event_tx,
        totals,
        host_ram_budget,
    )
    .await?;
    let vae_burnpack_bytes = vae_burnpack.len() as u64;
    host_ram_budget.reserve_retained(vae_burnpack_bytes, "retaining TripoSG VAE burnpack bytes")?;
    let vae = match load_triposg_vae_from_burnpack_bytes(&vae_config, device, vae_burnpack) {
        Ok(model) => model,
        Err(err) => {
            host_ram_budget.release_retained(vae_burnpack_bytes);
            return Err(format!("failed to load TripoSG VAE burnpack: {err}"));
        }
    };
    host_ram_budget.release_retained(vae_burnpack_bytes);

    let dit_burnpack = download_burnpack_asset(
        &join_web_path(
            &triposg_root_url,
            "transformer/diffusion_pytorch_model.safetensors",
        ),
        "TripoSG DiT",
        "TRIPOSG_BPK_PRECISION",
        event_tx,
        totals,
        host_ram_budget,
    )
    .await?;
    let dit_burnpack_bytes = dit_burnpack.len() as u64;
    host_ram_budget.reserve_retained(dit_burnpack_bytes, "retaining TripoSG DiT burnpack bytes")?;
    let dit = match load_triposg_dit_from_burnpack_bytes(&dit_config, device, dit_burnpack) {
        Ok(model) => model,
        Err(err) => {
            host_ram_budget.release_retained(dit_burnpack_bytes);
            return Err(format!("failed to load TripoSG DiT burnpack: {err}"));
        }
    };
    host_ram_budget.release_retained(dit_burnpack_bytes);

    let dino_gpu_burnpack = download_burnpack_asset(
        &join_web_path(&triposg_root_url, "image_encoder_dinov2/model.safetensors"),
        "DINOv2",
        "TRIPOSG_BPK_PRECISION",
        event_tx,
        totals,
        host_ram_budget,
    )
    .await?;
    let dino_gpu_burnpack_bytes = dino_gpu_burnpack.len() as u64;
    host_ram_budget.reserve_retained(
        dino_gpu_burnpack_bytes,
        "retaining DINOv2 GPU burnpack bytes",
    )?;
    let image_encoder =
        match load_triposg_dinov2_from_burnpack_bytes(device, dino_config, dino_gpu_burnpack) {
            Ok(encoder) => encoder,
            Err(err) => {
                host_ram_budget.release_retained(dino_gpu_burnpack_bytes);
                return Err(format!("failed to load DINOv2 burnpack: {err}"));
            }
        };
    host_ram_budget.release_retained(dino_gpu_burnpack_bytes);

    let scheduler = scheduler_config.init();
    let triposg = TripoSGPipeline::new(vae, dit, scheduler, image_encoder, dino_processor.clone());

    let dino_cpu = if matches!(dino_backend, DinoBackend::Cpu) {
        let cpu_device = <burn::backend::NdArray<f32> as Backend>::Device::default();
        let cpu_config = parsed_dino_config.unwrap_or_else(default_dinov2_config);

        // Load CPU DINO from a separate download to avoid retaining two giant copies of the
        // same burnpack simultaneously during wasm startup.
        let dino_cpu_burnpack = download_burnpack_asset(
            &join_web_path(&triposg_root_url, "image_encoder_dinov2/model.safetensors"),
            "DINOv2 (CPU)",
            "TRIPOSG_BPK_PRECISION",
            event_tx,
            totals,
            host_ram_budget,
        )
        .await?;
        let dino_cpu_burnpack_bytes = dino_cpu_burnpack.len() as u64;
        host_ram_budget.reserve_retained(
            dino_cpu_burnpack_bytes,
            "retaining DINOv2 CPU burnpack bytes",
        )?;
        let encoder = match load_triposg_dinov2_from_burnpack_bytes(
            &cpu_device,
            cpu_config,
            dino_cpu_burnpack,
        ) {
            Ok(encoder) => encoder,
            Err(err) => {
                host_ram_budget.release_retained(dino_cpu_burnpack_bytes);
                return Err(format!("failed to load DINOv2 burnpack on CPU: {err}"));
            }
        };
        host_ram_budget.release_retained(dino_cpu_burnpack_bytes);
        Some(DinoCpuState {
            device: cpu_device,
            encoder,
            processor: dino_processor,
        })
    } else {
        None
    };

    Ok((dino_cpu, triposg))
}

#[cfg(target_arch = "wasm32")]
async fn load_rmbg14_artifacts_wasm(
    rmbg_root_url: &str,
    event_tx: &Sender<WorkerEvent>,
    totals: &mut DownloadTotals,
    host_ram_budget: &mut WasmHostMemoryBudget,
) -> Result<Rmbg14Artifacts, String> {
    let burnpack = download_burnpack_asset(
        &join_web_path(rmbg_root_url, "model.safetensors"),
        "RMBG",
        "RMBG_BPK_PRECISION",
        event_tx,
        totals,
        host_ram_budget,
    )
    .await?;
    let config_json = fetch_optional_text(&join_web_path(rmbg_root_url, "config.json")).await?;
    let processor_json =
        fetch_optional_text(&join_web_path(rmbg_root_url, "preprocessor_config.json")).await?;
    Ok(Rmbg14Artifacts {
        burnpack,
        config_json,
        processor_json,
    })
}

#[cfg(target_arch = "wasm32")]
fn send_worker_status(event_tx: &Sender<WorkerEvent>, message: impl Into<String>) {
    let _ = event_tx.send(WorkerEvent {
        requests: Vec::new(),
        results: Vec::new(),
        elapsed: Duration::ZERO,
        status_message: Some(message.into()),
    });
}

#[cfg(target_arch = "wasm32")]
fn normalize_web_path(path: &std::path::Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

#[cfg(target_arch = "wasm32")]
fn join_web_path(root: &str, rel: &str) -> String {
    let mut out = root.trim_end_matches('/').to_string();
    out.push('/');
    out.push_str(rel.trim_start_matches('/'));
    out
}

#[cfg(any(target_arch = "wasm32", test))]
fn format_mebibytes(bytes: u64) -> String {
    format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0))
}

#[cfg(any(target_arch = "wasm32", test))]
fn web_max_host_ram_bytes() -> u64 {
    option_env!("BURN_SYNTH_WEB_MAX_HOST_RAM_BYTES")
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_WEB_MAX_HOST_RAM_BYTES)
}

#[cfg(target_arch = "wasm32")]
fn web_max_burnpack_bytes() -> u64 {
    const DEFAULT_MAX_BPK_BYTES: u64 = ONE_GIB;
    option_env!("BURN_SYNTH_WEB_MAX_BPK_BYTES")
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_MAX_BPK_BYTES)
}

#[cfg(target_arch = "wasm32")]
fn burnpack_too_large_error(url: &str, bytes: u64, limit_bytes: u64) -> String {
    format!(
        "burnpack at {url} is {} which exceeds browser limit {} (set build-time BURN_SYNTH_WEB_MAX_BPK_BYTES to raise)",
        format_mebibytes(bytes),
        format_mebibytes(limit_bytes),
    )
}

#[cfg(target_arch = "wasm32")]
async fn download_burnpack_asset(
    base_safetensors_url: &str,
    label: &str,
    precision_env: &str,
    event_tx: &Sender<WorkerEvent>,
    totals: &mut DownloadTotals,
    host_ram_budget: &mut WasmHostMemoryBudget,
) -> Result<Vec<u8>, String> {
    let candidates =
        candidate_burnpack_names(base_safetensors_url, prefer_f16_burnpack(precision_env));
    let max_bytes = web_max_burnpack_bytes();
    let mut last_error = "no matching burnpack candidate".to_string();
    for candidate in candidates {
        match download_binary_with_status(
            &candidate,
            label,
            max_bytes,
            event_tx,
            totals,
            host_ram_budget,
        )
        .await
        {
            Ok(bytes) => return Ok(bytes),
            Err(err) => {
                last_error = err;
                if !last_error.contains("HTTP 404") {
                    break;
                }
            }
        }

        for manifest_url in burnpack_manifest_candidates(&candidate) {
            match download_burnpack_from_manifest(
                &manifest_url,
                label,
                max_bytes,
                event_tx,
                totals,
                host_ram_budget,
            )
            .await
            {
                Ok(bytes) => return Ok(bytes),
                Err(err) => {
                    last_error = err;
                    if !last_error.contains("HTTP 404") {
                        break;
                    }
                }
            }
        }

        if !last_error.contains("HTTP 404") {
            break;
        }
    }
    Err(format!(
        "failed to download burnpack for {label} from {base_safetensors_url}: {last_error}"
    ))
}

#[cfg(target_arch = "wasm32")]
async fn download_burnpack_from_manifest(
    manifest_url: &str,
    label: &str,
    max_bytes: u64,
    event_tx: &Sender<WorkerEvent>,
    totals: &mut DownloadTotals,
    host_ram_budget: &mut WasmHostMemoryBudget,
) -> Result<Vec<u8>, String> {
    let manifest_text = fetch_text(manifest_url).await?;
    let manifest = parse_shard_manifest_bytes(manifest_text.as_bytes(), manifest_url)?;
    let entries = manifest.shard_entries();
    if entries.is_empty() {
        return Err(format!(
            "shard manifest {manifest_url} contains no shard entries"
        ));
    }

    host_ram_budget.observe_temporary(
        manifest.total_bytes,
        &format!("allocating shard reconstruction buffer for {label}"),
    )?;

    let mut output = Vec::with_capacity(manifest.total_bytes as usize);
    for (index, entry) in entries.iter().enumerate() {
        let shard_url = resolve_manifest_entry_uri(manifest_url, entry.path());
        let shard_label = format!("{label} shard {}/{}", index + 1, entries.len());
        let bytes = download_binary_with_status(
            &shard_url,
            &shard_label,
            max_bytes,
            event_tx,
            totals,
            host_ram_budget,
        )
        .await?;
        host_ram_budget.observe_temporary(
            output.len() as u64 + bytes.len() as u64,
            &format!("reconstructing {label} from shard {}", index + 1),
        )?;
        output.extend_from_slice(&bytes);
    }

    if manifest.total_bytes > 0 && output.len() as u64 != manifest.total_bytes {
        return Err(format!(
            "manifest {manifest_url} expected {} bytes but reconstructed {} bytes",
            manifest.total_bytes,
            output.len()
        ));
    }
    Ok(output)
}

#[cfg(target_arch = "wasm32")]
async fn download_binary_with_status(
    url: &str,
    label: &str,
    max_bytes: u64,
    event_tx: &Sender<WorkerEvent>,
    totals: &mut DownloadTotals,
    host_ram_budget: &mut WasmHostMemoryBudget,
) -> Result<Vec<u8>, String> {
    let mut registered_total = false;
    let mut prev = 0u64;
    let bytes = fetch_binary_with_progress(url, max_bytes, |loaded, total| {
        if let Some(total_bytes) = total
            && !registered_total
        {
            totals.known_total = totals.known_total.saturating_add(total_bytes);
            registered_total = true;
        }
        if registered_total {
            let delta = loaded.saturating_sub(prev);
            totals.known_downloaded = totals.known_downloaded.saturating_add(delta);
        } else {
            let delta = loaded.saturating_sub(prev);
            totals.unknown_downloaded = totals.unknown_downloaded.saturating_add(delta);
        }
        prev = loaded;

        let message = if totals.known_total > 0 {
            let percent = (totals.known_downloaded as f64 / totals.known_total as f64) * 100.0;
            format!(
                "Loading {label}... {percent:.1}% ({}/{})",
                format_mebibytes(totals.known_downloaded),
                format_mebibytes(totals.known_total)
            )
        } else {
            format!(
                "Loading {label}... {} downloaded",
                format_mebibytes(totals.unknown_downloaded)
            )
        };
        send_worker_status(event_tx, message);
    })
    .await?;
    host_ram_budget.observe_temporary(bytes.len() as u64, &format!("downloading {label}"))?;
    Ok(bytes)
}

#[cfg(target_arch = "wasm32")]
async fn fetch_optional_text(url: &str) -> Result<Option<String>, String> {
    match fetch_text(url).await {
        Ok(text) => Ok(Some(text)),
        Err(err) if err.contains("HTTP 404") => Ok(None),
        Err(err) => Err(err),
    }
}

#[cfg(target_arch = "wasm32")]
async fn fetch_optional_text_candidates(urls: &[String]) -> Result<Option<String>, String> {
    for url in urls {
        match fetch_text(url).await {
            Ok(text) => return Ok(Some(text)),
            Err(err) if err.contains("HTTP 404") => continue,
            Err(err) => return Err(err),
        }
    }
    Ok(None)
}

#[cfg(target_arch = "wasm32")]
async fn fetch_text(url: &str) -> Result<String, String> {
    let window = web_sys::window().ok_or_else(|| "window is unavailable".to_string())?;
    let response_value = JsFuture::from(window.fetch_with_str(url))
        .await
        .map_err(|err| format!("fetch failed for {url}: {err:?}"))?;
    let response: Response = response_value
        .dyn_into()
        .map_err(|_| format!("invalid response object for {url}"))?;
    if !response.ok() {
        return Err(format!("HTTP {} for {}", response.status(), url));
    }
    let text_promise = response
        .text()
        .map_err(|err| format!("failed to read text for {url}: {err:?}"))?;
    let text_value = JsFuture::from(text_promise)
        .await
        .map_err(|err| format!("failed to await text for {url}: {err:?}"))?;
    Ok(text_value.as_string().unwrap_or_default())
}

#[cfg(target_arch = "wasm32")]
async fn fetch_binary_with_progress<F>(
    url: &str,
    max_bytes: u64,
    mut on_progress: F,
) -> Result<Vec<u8>, String>
where
    F: FnMut(u64, Option<u64>),
{
    let window = web_sys::window().ok_or_else(|| "window is unavailable".to_string())?;
    let response_value = JsFuture::from(window.fetch_with_str(url))
        .await
        .map_err(|err| format!("fetch failed for {url}: {err:?}"))?;
    let response: Response = response_value
        .dyn_into()
        .map_err(|_| format!("invalid response object for {url}"))?;
    if !response.ok() {
        return Err(format!("HTTP {} for {}", response.status(), url));
    }

    let total = response
        .headers()
        .get("content-length")
        .ok()
        .flatten()
        .and_then(|value| value.parse::<u64>().ok());
    if let Some(total_bytes) = total
        && total_bytes > max_bytes
    {
        return Err(burnpack_too_large_error(url, total_bytes, max_bytes));
    }

    if let Some(body) = response.body() {
        let reader: ReadableStreamDefaultReader = body
            .get_reader()
            .dyn_into()
            .map_err(|_| format!("failed to create stream reader for {url}"))?;

        let mut output = Vec::new();
        let mut loaded = 0u64;
        loop {
            let chunk = JsFuture::from(reader.read())
                .await
                .map_err(|err| format!("stream read failed for {url}: {err:?}"))?;
            let done = js_sys::Reflect::get(&chunk, &"done".into())
                .ok()
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            if done {
                break;
            }
            let value = js_sys::Reflect::get(&chunk, &"value".into())
                .map_err(|_| format!("missing stream chunk value for {url}"))?;
            let chunk_bytes = Uint8Array::new(&value);
            let len = chunk_bytes.length() as usize;
            if loaded.saturating_add(len as u64) > max_bytes {
                return Err(burnpack_too_large_error(
                    url,
                    loaded.saturating_add(len as u64),
                    max_bytes,
                ));
            }
            let old_len = output.len();
            output.resize(old_len + len, 0);
            chunk_bytes.copy_to(&mut output[old_len..]);
            loaded = loaded.saturating_add(len as u64);
            on_progress(loaded, total);
        }
        on_progress(loaded, total);
        return Ok(output);
    }

    let buffer_promise = response
        .array_buffer()
        .map_err(|err| format!("failed to start array_buffer for {url}: {err:?}"))?;
    let buffer = JsFuture::from(buffer_promise)
        .await
        .map_err(|err| format!("failed to await array_buffer for {url}: {err:?}"))?;
    let bytes = Uint8Array::new(&buffer);
    if (bytes.length() as u64) > max_bytes {
        return Err(burnpack_too_large_error(
            url,
            bytes.length() as u64,
            max_bytes,
        ));
    }
    let mut output = vec![0u8; bytes.length() as usize];
    bytes.copy_to(&mut output);
    on_progress(output.len() as u64, total);
    Ok(output)
}

fn run_inference_with_state<B: Backend>(
    state: &mut PipelineState<B>,
    args: &AppArgs,
    requests: &[InferenceRequest],
) -> Vec<Result<Option<SynthMesh>, String>> {
    if requests.is_empty() {
        return Vec::new();
    }

    let synthesis_backend = match select_synthesis_backend(state) {
        Ok(backend) => backend,
        Err(err) => return vec![Err(err); requests.len()],
    };
    if matches!(synthesis_backend, ActiveSynthesisBackend::Trellis) {
        return run_trellis_batch(state, args, requests).unwrap_or_else(|err| {
            vec![Err(format!("Trellis batch execution failed: {err}")); requests.len()]
        });
    }

    let device = state.device.clone();
    let prepare_config =
        prepare_image_config_for_backend::<B>(state.rmbg_model, state.rmbg_backend);
    let mut results: Vec<Option<Result<Option<SynthMesh>, String>>> = vec![None; requests.len()];
    let mut batch_indices = Vec::new();
    let mut batch_images = Vec::new();
    let mut batch_prepared = Vec::new();

    for (idx, request) in requests.iter().enumerate() {
        match prepare_request(state, &device, request, &prepare_config) {
            Ok((image, prepared_cpu)) => {
                batch_indices.push(idx);
                batch_images.push(image);
                batch_prepared.push(prepared_cpu);
            }
            Err(err) => {
                results[idx] = Some(Err(err));
            }
        }
    }

    if !batch_images.is_empty() {
        let batch_results = if state.scribble.is_some() {
            run_scribble_batch(state, args, &batch_images)
        } else {
            run_triposg_batch(state, args, &batch_images, &batch_prepared)
        };
        let batch_results = match batch_results {
            Ok(results) => results,
            Err(err) => vec![Err(err); batch_images.len()],
        };

        for (slot, idx) in batch_indices.iter().enumerate() {
            let value = batch_results
                .get(slot)
                .cloned()
                .unwrap_or_else(|| Err("missing batch result".to_string()));
            results[*idx] = Some(value);
        }
    }

    results
        .into_iter()
        .map(|result| result.unwrap_or_else(|| Err("missing inference result".to_string())))
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ActiveSynthesisBackend {
    Triposg,
    Trellis,
}

fn select_synthesis_backend<B: Backend>(
    state: &PipelineState<B>,
) -> Result<ActiveSynthesisBackend, String> {
    let triposg_loaded = state.triposg.is_some() || state.scribble.is_some();
    let trellis_loaded = state.trellis.is_some();
    for model in state.synthesis_models.iter() {
        match model {
            SynthesisModel::Triposg => {
                if triposg_loaded {
                    return Ok(ActiveSynthesisBackend::Triposg);
                }
            }
            SynthesisModel::Trellis => {
                if trellis_loaded {
                    return Ok(ActiveSynthesisBackend::Trellis);
                }
            }
        }
    }

    if triposg_loaded {
        return Ok(ActiveSynthesisBackend::Triposg);
    }

    if let Some(err) = state.triposg_load_error.as_ref() {
        return Err(format!("TripoSG backend is unavailable: {err}"));
    }

    if let Some(err) = state.trellis_load_error.as_ref() {
        return Err(format!("Trellis backend is unavailable: {err}"));
    }

    Err("No synthesis backend is available.".to_string())
}

fn run_trellis_batch<B: Backend>(
    state: &mut PipelineState<B>,
    args: &AppArgs,
    requests: &[InferenceRequest],
) -> Result<Vec<Result<Option<SynthMesh>, String>>, String> {
    let pipeline = state.trellis.as_ref().ok_or_else(|| {
        if let Some(err) = state.trellis_load_error.as_ref() {
            format!("Trellis backend is unavailable: {err}")
        } else {
            "Trellis backend is not loaded".to_string()
        }
    })?;
    let quality = trellis_quality_to_runtime(args.trellis_quality);
    let device = trellis_device_for_backend(args.backend.clone());
    let mut results = Vec::with_capacity(requests.len());
    for request in requests {
        if request.image_contents.is_some() {
            results.push(Err(format!(
                "Trellis2 inference from uploaded bytes is unsupported for '{}'.",
                request.image_path.display()
            )));
            continue;
        }
        let options = TrellisRunOptions {
            quality,
            device,
            seed: args.seed,
            hook_output: None,
            noise_overrides_hook: None,
        };
        match pipeline.infer_mesh(&request.image_path, &options) {
            Ok(mesh) => {
                let mesh = SynthMesh::from(mesh);
                results.push(Ok(apply_mesh_decimation(Some(mesh), args.target_faces)));
            }
            Err(err) => {
                results.push(Err(format!(
                    "Trellis2 inference failed for '{}': {err}",
                    request.image_path.display()
                )));
            }
        }
    }
    Ok(results)
}

fn prepare_request<B: Backend>(
    state: &PipelineState<B>,
    device: &B::Device,
    request: &InferenceRequest,
    config: &PrepareImageConfig,
) -> Result<(Tensor<B, 4>, Option<PreparedImageData>), String> {
    if let Some(bytes) = request.image_contents.as_deref() {
        return match state.rmbg_model {
            RmbgModel::Rmbg2 => Err(format!(
                "RMBG-2.0 preprocessing from uploaded bytes is unsupported for '{}'.",
                request.image_path.display()
            )),
            RmbgModel::Rmbg14 => match state.rmbg_backend {
                RmbgBackend::Cpu => {
                    let rmbg = state
                        .rmbg14_cpu
                        .as_ref()
                        .ok_or_else(|| "RMBG-1.4 CPU pipeline not loaded".to_string())?;
                    let prepared: PreparedImageData =
                        prepare_image_data_from_bytes(bytes, Some(rmbg), config)
                            .map_err(|err| format!("failed to prepare image bytes: {err}"))?;
                    let image = prepared.to_tensor::<B>(device);
                    Ok((image, Some(prepared)))
                }
                RmbgBackend::Gpu | RmbgBackend::Auto => {
                    let image = prepare_image_tensor_from_bytes::<B>(
                        bytes,
                        state.rmbg14_device.as_ref(),
                        device,
                        config,
                    )
                    .map_err(|err| format!("failed to prepare image bytes: {err}"))?;
                    Ok((image, None))
                }
            },
        };
    }

    match state.rmbg_model {
        RmbgModel::Rmbg2 => {
            let rmbg2 = state
                .rmbg2
                .as_ref()
                .ok_or_else(|| "RMBG-2.0 pipeline not loaded".to_string())?;
            let prepared = rmbg2
                .prepare_image_data(&request.image_path, config)
                .map_err(|err| format!("failed to prepare image with RMBG-2.0: {err}"))?;
            let image = prepared.to_tensor::<B>(device);
            Ok((image, Some(prepared)))
        }
        RmbgModel::Rmbg14 => match state.rmbg_backend {
            RmbgBackend::Cpu => {
                let rmbg = state
                    .rmbg14_cpu
                    .as_ref()
                    .ok_or_else(|| "RMBG-1.4 CPU pipeline not loaded".to_string())?;
                let prepared: PreparedImageData =
                    prepare_image_data(&request.image_path, Some(rmbg), config)
                        .map_err(|err| format!("failed to prepare image: {err}"))?;
                let image = prepared.to_tensor::<B>(device);
                Ok((image, Some(prepared)))
            }
            RmbgBackend::Gpu | RmbgBackend::Auto => {
                let image = prepare_image_tensor::<B>(
                    &request.image_path,
                    state.rmbg14_device.as_ref(),
                    device,
                    config,
                )
                .map_err(|err| format!("failed to prepare image: {err}"))?;
                Ok((image, None))
            }
        },
    }
}

fn run_triposg_batch<B: Backend>(
    state: &mut PipelineState<B>,
    args: &AppArgs,
    images: &[Tensor<B, 4>],
    prepared_cpu: &[Option<PreparedImageData>],
) -> Result<Vec<Result<Option<SynthMesh>, String>>, String> {
    if images.len() != prepared_cpu.len() {
        return Err("prepared image mismatch".to_string());
    }

    let bounds = state.bounds;
    let chunk_size = state.chunk_size;
    let hierarchical = state.hierarchical.clone();
    let flash = state.flash.clone();
    let decode_config = DecodeConfig {
        mesh_mode: args.mesh_mode.clone(),
        bounds,
        resolution: args.resolution,
        chunk_size,
        hierarchical: &hierarchical,
        flash: &flash,
    };
    let pipeline = state
        .triposg
        .as_mut()
        .ok_or_else(|| "TripoSG pipeline not loaded".to_string())?;

    let image_embeds = if matches!(state.dino_backend, DinoBackend::Cpu) {
        let dino = state
            .dino_cpu
            .as_ref()
            .ok_or_else(|| "DINO CPU encoder not loaded".to_string())?;
        let mut cpu_images = Vec::with_capacity(images.len());
        for (image, prepared) in images.iter().zip(prepared_cpu.iter()) {
            let cpu_image = if let Some(prepared) = prepared.as_ref() {
                prepared.to_tensor::<burn::backend::NdArray<f32>>(&dino.device)
            } else {
                tensor_to_cpu(image, &dino.device)?
            };
            cpu_images.push(cpu_image);
        }
        let processed = preprocess_and_stack_images(&dino.processor, &cpu_images)?;
        let cpu_embeds = dino.encoder.forward(processed);
        convert_embeds_to_device(&cpu_embeds, &state.device)?
    } else {
        let processed = preprocess_and_stack_images(&pipeline.image_processor, images)?;
        pipeline.image_encoder.forward(processed)
    };

    let [batch_size, _, _] = image_embeds.shape().dims::<3>();
    let output = pipeline.sample_from_embeds(
        image_embeds,
        batch_size,
        args.num_steps,
        args.num_tokens,
        args.guidance_scale,
        None,
        None,
    );

    let label = if matches!(state.dino_backend, DinoBackend::Cpu) {
        "TripoSG (CPU DINO)"
    } else {
        "TripoSG"
    };
    let mut results = Vec::with_capacity(batch_size);
    let [_, latent_tokens, latent_channels] = output.latents.shape().dims::<3>();
    for idx in 0..batch_size {
        let sample =
            output
                .latents
                .clone()
                .slice([idx..(idx + 1), 0..latent_tokens, 0..latent_channels]);
        let (mesh, grid) = decode_triposg_mesh(pipeline, sample, &decode_config)?;
        if mesh.is_none() {
            log_empty_mesh_stats(label, &grid);
        }
        results.push(Ok(apply_mesh_decimation(
            mesh.map(SynthMesh::from),
            args.target_faces,
        )));
    }

    Ok(results)
}

fn run_scribble_batch<B: Backend>(
    state: &mut PipelineState<B>,
    args: &AppArgs,
    images: &[Tensor<B, 4>],
) -> Result<Vec<Result<Option<SynthMesh>, String>>, String> {
    let bounds = state.bounds;
    let chunk_size = state.chunk_size;
    let hierarchical = state.hierarchical.clone();
    let flash = state.flash.clone();
    let decode_config = DecodeConfig {
        mesh_mode: args.mesh_mode.clone(),
        bounds,
        resolution: args.resolution,
        chunk_size,
        hierarchical: &hierarchical,
        flash: &flash,
    };
    let pipeline = state
        .scribble
        .as_mut()
        .ok_or_else(|| "scribble pipeline not loaded".to_string())?;
    let text_embeds = state
        .text_embeds
        .as_ref()
        .ok_or_else(|| "text embeddings not loaded".to_string())?;
    let text_batch = expand_text_embeds(text_embeds, images.len())?;

    let processed = preprocess_and_stack_images(&pipeline.image_processor, images)?;
    let image_embeds = pipeline.image_encoder.forward(processed);
    let [batch_size, _, _] = image_embeds.shape().dims::<3>();

    let output = pipeline.sample_with_embeddings(
        text_batch,
        image_embeds,
        args.num_steps,
        args.num_tokens,
        args.guidance_scale,
        None,
        None,
    );

    let mut results = Vec::with_capacity(batch_size);
    let [_, latent_tokens, latent_channels] = output.latents.shape().dims::<3>();
    for idx in 0..batch_size {
        let sample =
            output
                .latents
                .clone()
                .slice([idx..(idx + 1), 0..latent_tokens, 0..latent_channels]);
        let (mesh, grid) = decode_scribble_mesh(pipeline, sample, &decode_config)?;
        if mesh.is_none() {
            log_empty_mesh_stats("TripoSG-scribble", &grid);
        }
        results.push(Ok(apply_mesh_decimation(
            mesh.map(SynthMesh::from),
            args.target_faces,
        )));
    }

    Ok(results)
}

fn preprocess_and_stack_images<B: Backend>(
    processor: &DinoImageProcessor,
    images: &[Tensor<B, 4>],
) -> Result<Tensor<B, 4>, String> {
    if images.is_empty() {
        return Err("no images provided for preprocessing".to_string());
    }
    let mut processed = Vec::with_capacity(images.len());
    for image in images.iter() {
        processed.push(processor.preprocess(image.clone()));
    }
    stack_images(processed, processor.resize_mode.clone())
}

fn stack_images<B: Backend>(
    images: Vec<Tensor<B, 4>>,
    resize_mode: InterpolateMode,
) -> Result<Tensor<B, 4>, String> {
    if images.is_empty() {
        return Err("no images to stack".to_string());
    }
    let [_, _, target_height, target_width] = images[0].shape().dims::<4>();
    let mut stacked = Vec::with_capacity(images.len());
    for image in images {
        let [_, _, height, width] = image.shape().dims::<4>();
        if height != target_height || width != target_width {
            let options = InterpolateOptions {
                mode: resize_mode.clone(),
            };
            stacked.push(interpolate(image, [target_height, target_width], options));
        } else {
            stacked.push(image);
        }
    }
    Ok(Tensor::cat(stacked, 0))
}

fn expand_text_embeds<B: Backend>(
    text_embeds: &Tensor<B, 3>,
    batch_size: usize,
) -> Result<Tensor<B, 3>, String> {
    let [text_batch, _, _] = text_embeds.shape().dims::<3>();
    if text_batch == batch_size {
        return Ok(text_embeds.clone());
    }
    if text_batch != 1 {
        return Err(format!(
            "text embeddings batch {} does not match image batch {}",
            text_batch, batch_size
        ));
    }
    let mut copies = Vec::with_capacity(batch_size);
    for _ in 0..batch_size {
        copies.push(text_embeds.clone());
    }
    Ok(Tensor::cat(copies, 0))
}

#[cfg(not(target_arch = "wasm32"))]
fn tensor_to_cpu<B: Backend>(
    image: &Tensor<B, 4>,
    device: &<burn::backend::NdArray<f32> as Backend>::Device,
) -> Result<Tensor<burn::backend::NdArray<f32>, 4>, String> {
    let data = image
        .clone()
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|_| "failed to read image tensor")?;
    let dims = image.shape().dims::<4>();
    let flat = Tensor::<burn::backend::NdArray<f32>, 1>::from_floats(data.as_slice(), device);
    Ok(flat.reshape([
        dims[0] as i32,
        dims[1] as i32,
        dims[2] as i32,
        dims[3] as i32,
    ]))
}

#[cfg(target_arch = "wasm32")]
fn tensor_to_cpu<B: Backend>(
    _image: &Tensor<B, 4>,
    _device: &<burn::backend::NdArray<f32> as Backend>::Device,
) -> Result<Tensor<burn::backend::NdArray<f32>, 4>, String> {
    Err("tensor readback requires async handling on wasm32".to_string())
}

#[cfg(not(target_arch = "wasm32"))]
fn convert_embeds_to_device<B: Backend>(
    embeds: &Tensor<burn::backend::NdArray<f32>, 3>,
    device: &B::Device,
) -> Result<Tensor<B, 3>, String> {
    let embed_dims = embeds.shape().dims::<3>();
    let embed_data = embeds
        .clone()
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|_| "failed to read DINO embeddings")?;
    Ok(
        Tensor::<B, 1>::from_floats(embed_data.as_slice(), device).reshape([
            embed_dims[0] as i32,
            embed_dims[1] as i32,
            embed_dims[2] as i32,
        ]),
    )
}

#[cfg(target_arch = "wasm32")]
fn convert_embeds_to_device<B: Backend>(
    _embeds: &Tensor<burn::backend::NdArray<f32>, 3>,
    _device: &B::Device,
) -> Result<Tensor<B, 3>, String> {
    Err("tensor readback requires async handling on wasm32".to_string())
}

struct DecodeConfig<'a> {
    mesh_mode: MeshMode,
    bounds: [f32; 6],
    resolution: usize,
    chunk_size: usize,
    hierarchical: &'a HierarchicalExtractConfig,
    flash: &'a FlashExtractConfig,
}

fn decode_triposg_mesh<B: Backend>(
    pipeline: &TripoSGPipeline<B>,
    latents: Tensor<B, 3>,
    config: &DecodeConfig<'_>,
) -> Result<(Option<TripoMesh>, DenseGrid), String> {
    let grid = match config.mesh_mode {
        MeshMode::Dense => pipeline
            .decode_grid(
                latents.clone(),
                config.bounds,
                config.resolution,
                config.chunk_size,
            )
            .map_err(|err| err.to_string())?,
        MeshMode::Hierarchical => {
            hierarchical_extract_geometry(latents.clone(), &pipeline.vae, config.hierarchical)
                .map_err(|err| err.to_string())?
        }
        MeshMode::Flash => flash_extract_geometry(latents.clone(), &pipeline.vae, config.flash)
            .map_err(|err| err.to_string())?,
    };
    let mesh = match config.mesh_mode {
        MeshMode::Dense | MeshMode::Hierarchical => grid_to_mesh(&grid, 0.0),
        MeshMode::Flash => sdf_to_mesh_diff_dmc(&grid),
    };
    Ok((mesh, grid))
}

fn decode_scribble_mesh<B: Backend>(
    pipeline: &TripoSGScribblePipeline<B>,
    latents: Tensor<B, 3>,
    config: &DecodeConfig<'_>,
) -> Result<(Option<TripoMesh>, DenseGrid), String> {
    let grid = match config.mesh_mode {
        MeshMode::Dense => pipeline
            .decode_grid(
                latents.clone(),
                config.bounds,
                config.resolution,
                config.chunk_size,
            )
            .map_err(|err| err.to_string())?,
        MeshMode::Hierarchical => {
            hierarchical_extract_geometry(latents.clone(), &pipeline.vae, config.hierarchical)
                .map_err(|err| err.to_string())?
        }
        MeshMode::Flash => flash_extract_geometry(latents.clone(), &pipeline.vae, config.flash)
            .map_err(|err| err.to_string())?,
    };
    let mesh = match config.mesh_mode {
        MeshMode::Dense | MeshMode::Hierarchical => grid_to_mesh(&grid, 0.0),
        MeshMode::Flash => sdf_to_mesh_diff_dmc(&grid),
    };
    Ok((mesh, grid))
}

fn tuned_chunk_size<B: Backend>(requested: usize) -> usize {
    let requested = requested.max(1);
    let mut chunk_size = if requested == DEFAULT_CHUNK_SIZE && is_gpu_backend::<B>() {
        if is_wgpu_backend::<B>() {
            WGPU_CHUNK_SIZE_CAP
        } else if is_cuda_backend::<B>() {
            CUDA_CHUNK_SIZE_CAP
        } else {
            requested
        }
    } else {
        requested
    };

    if is_wgpu_backend::<B>() && chunk_size > WGPU_CHUNK_SIZE_CAP {
        warn!(
            "Capping chunk size from {} to {} for WGPU backend to avoid oversized buffers.",
            chunk_size, WGPU_CHUNK_SIZE_CAP
        );
        chunk_size = WGPU_CHUNK_SIZE_CAP;
    }
    if is_cuda_backend::<B>() && chunk_size > CUDA_CHUNK_SIZE_CAP {
        warn!(
            "Capping chunk size from {} to {} for CUDA backend to avoid oversized buffers.",
            chunk_size, CUDA_CHUNK_SIZE_CAP
        );
        chunk_size = CUDA_CHUNK_SIZE_CAP;
    }

    chunk_size
}

fn is_gpu_backend<B: Backend>() -> bool {
    is_wgpu_backend::<B>() || is_cuda_backend::<B>()
}

fn is_wgpu_backend<B: Backend>() -> bool {
    #[cfg(feature = "wgpu")]
    {
        TypeId::of::<B>() == TypeId::of::<burn_wgpu::Wgpu>()
    }
    #[cfg(not(feature = "wgpu"))]
    {
        let _ = TypeId::of::<B>();
        false
    }
}

fn is_cuda_backend<B: Backend>() -> bool {
    #[cfg(feature = "cuda")]
    {
        TypeId::of::<B>() == TypeId::of::<burn_cuda::Cuda>()
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = TypeId::of::<B>();
        false
    }
}

fn configure_cubecl_autotune<B: Backend>() {
    #[cfg(not(target_arch = "wasm32"))]
    if is_wgpu_backend::<B>() && std::env::var("CUBECL_AUTOTUNE_LEVEL").is_err() {
        unsafe {
            std::env::set_var("CUBECL_AUTOTUNE_LEVEL", "minimal");
        }
    }
}

fn prepare_image_config_for_backend<B: Backend>(
    rmbg_model: RmbgModel,
    rmbg_backend: RmbgBackend,
) -> PrepareImageConfig {
    let mut config = PrepareImageConfig::default();
    if let Some(value) = std::env::var("TRIPOSG_MAX_IMAGE_DIM")
        .ok()
        .and_then(|val| val.parse::<usize>().ok())
        && value > 0
    {
        config.max_dimension = value;
    }
    if is_wgpu_backend::<B>() {
        config.max_dimension = config.max_dimension.min(1024);
        if matches!(rmbg_model, RmbgModel::Rmbg14) && matches!(rmbg_backend, RmbgBackend::Gpu) {
            // Avoid large im2col allocations in wgpu conv autotune for RMBG-1.4.
            config.max_dimension = config.max_dimension.min(384);
        }
    }
    config
}

fn cap_rmbg14_processor_for_backend<B: Backend>(
    pipeline: &mut RmbgPipeline<B>,
    rmbg_backend: RmbgBackend,
) {
    if !is_wgpu_backend::<B>() || !matches!(rmbg_backend, RmbgBackend::Gpu) {
        return;
    }
    let cap = rmbg_processor_size_cap();
    if cap == 0 {
        return;
    }
    let previous = pipeline.processor.size;
    let next = match previous {
        Some([height, width]) => [height.min(cap), width.min(cap)],
        None => [cap, cap],
    };
    if previous != Some(next) {
        warn!(
            "Capping RMBG-1.4 processor size from {:?} to {:?} for WGPU stability.",
            previous, next
        );
        pipeline.processor.size = Some(next);
    }
}

fn rmbg_processor_size_cap() -> usize {
    if let Ok(value) = std::env::var("RMBG_MAX_DIM")
        && let Ok(parsed) = value.parse::<usize>()
    {
        return parsed;
    }
    384
}

fn log_empty_mesh_stats(label: &str, grid: &burn_tripo::pipeline::mesh::DenseGrid) {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut nan_count = 0usize;
    for &value in &grid.values {
        if value.is_nan() {
            nan_count += 1;
            continue;
        }
        min = min.min(value);
        max = max.max(value);
    }
    if min == f32::INFINITY {
        warn!("{label} grid contained only NaNs.");
    } else {
        let total = grid.values.len();
        let nan_ratio = nan_count as f32 / total.max(1) as f32 * 100.0;
        warn!(
            "{label} mesh empty; grid min={min:.4}, max={max:.4}, NaN={nan_count} ({nan_ratio:.2}%)."
        );
    }
}

fn apply_mesh_decimation(
    mesh: Option<SynthMesh>,
    target_faces: Option<usize>,
) -> Option<SynthMesh> {
    let target_faces = target_faces.filter(|value| *value > 0);
    let mut mesh = mesh?;
    if !mesh.uvs.is_empty() || mesh.material.is_some() || mesh.pbr_textures.is_some() {
        return Some(mesh);
    }
    if let Some(target) = target_faces
        && mesh.mesh.faces.len() > target
    {
        match decimate_tripo_mesh(&mesh.mesh, target) {
            Ok(decimated) => mesh.mesh = decimated,
            Err(err) => warn!("mesh decimation failed ({err}); using full mesh."),
        }
    }
    Some(mesh)
}

fn parse_bounds(bounds: &[f32]) -> Result<[f32; 6], Box<dyn std::error::Error>> {
    if bounds.len() != 6 {
        return Err("bounds must contain exactly 6 floats".into());
    }
    Ok([
        bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    use clap::Parser;
    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    use std::sync::Arc;
    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    use std::thread;
    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    use std::time::Duration;
    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System, get_current_pid};

    fn sample_mesh_grid(n: usize) -> TripoMesh {
        let mut vertices = Vec::new();
        let mut faces = Vec::new();
        for y in 0..=n {
            for x in 0..=n {
                vertices.push([x as f32, y as f32, 0.0]);
            }
        }
        for y in 0..n {
            for x in 0..n {
                let i0 = (y * (n + 1) + x) as u32;
                let i1 = i0 + 1;
                let i2 = i0 + (n + 1) as u32;
                let i3 = i2 + 1;
                faces.push([i0, i1, i3]);
                faces.push([i0, i3, i2]);
            }
        }
        TripoMesh { vertices, faces }
    }

    #[test]
    fn dino_backend_choice_mapping_round_trips() {
        for value in [DinoBackend::Auto, DinoBackend::Cpu, DinoBackend::Gpu] {
            assert_eq!(
                map_dino_backend_choice_back(map_dino_backend_choice(value)),
                value
            );
        }
    }

    #[test]
    fn apply_mesh_decimation_reduces_untextured_mesh() {
        let mesh = SynthMesh::from(sample_mesh_grid(20));
        let original_faces = mesh.mesh.faces.len();
        let output =
            apply_mesh_decimation(Some(mesh), Some(300)).expect("decimation should return mesh");
        assert!(output.mesh.faces.len() <= 300);
        assert!(output.mesh.faces.len() > 0);
        assert!(output.mesh.faces.len() < original_faces);
    }

    #[test]
    fn apply_mesh_decimation_skips_textured_meshes() {
        let mut mesh = SynthMesh::from(sample_mesh_grid(20));
        mesh.uvs = vec![[0.0, 0.0]; mesh.mesh.vertices.len()];
        let original_faces = mesh.mesh.faces.clone();
        let output =
            apply_mesh_decimation(Some(mesh), Some(300)).expect("decimation should return mesh");
        assert_eq!(output.mesh.faces, original_faces);
    }

    #[test]
    fn wasm_host_ram_budget_defaults_to_4gib() {
        assert_eq!(web_max_host_ram_bytes(), 4 * ONE_GIB);
    }

    #[test]
    fn wasm_host_ram_budget_rejects_over_limit_observation() {
        let mut budget = WasmHostMemoryBudget::new(4 * ONE_GIB);
        budget
            .reserve_retained(3 * ONE_GIB, "retain model A")
            .expect("reserve should succeed");
        let err = budget
            .observe_temporary((ONE_GIB * 2) - 1, "download model B shard")
            .expect_err("should exceed 4 GiB budget");
        assert!(
            err.contains("exceeded"),
            "unexpected budget error message: {err}"
        );
    }

    #[test]
    fn wasm_host_ram_budget_tracks_peak_observation() {
        let mut budget = WasmHostMemoryBudget::new(4 * ONE_GIB);
        budget
            .reserve_retained(512 * 1024 * 1024, "retain model")
            .expect("retain should fit");
        budget
            .observe_temporary(256 * 1024 * 1024, "download shard")
            .expect("temporary should fit");
        budget
            .observe_temporary(128 * 1024 * 1024, "smaller temporary")
            .expect("temporary should fit");
        budget.release_retained(512 * 1024 * 1024);
        assert!(
            budget.peak_bytes() >= 768 * 1024 * 1024,
            "peak should track retained + temporary observations"
        );
        assert!(
            budget.peak_bytes() < budget.limit_bytes(),
            "peak {} should remain below limit {}",
            budget.peak_bytes(),
            budget.limit_bytes()
        );
    }

    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    fn refresh_process_rss_bytes(system: &mut System, pid: Pid) -> Option<u64> {
        system.refresh_processes_specifics(
            ProcessesToUpdate::Some(&[pid]),
            true,
            ProcessRefreshKind::nothing().with_memory(),
        );
        system.process(pid).map(|process| process.memory())
    }

    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    struct ProcessMemoryMonitor {
        done: Arc<AtomicBool>,
        peak_bytes: Arc<AtomicU64>,
        join: Option<thread::JoinHandle<()>>,
    }

    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    impl ProcessMemoryMonitor {
        fn start(pid: Pid, baseline_bytes: u64) -> Self {
            let done = Arc::new(AtomicBool::new(false));
            let peak_bytes = Arc::new(AtomicU64::new(baseline_bytes));
            let done_thread = Arc::clone(&done);
            let peak_thread = Arc::clone(&peak_bytes);
            let join = thread::Builder::new()
                .name("worker_rss_monitor".to_string())
                .spawn(move || {
                    let mut system = System::new();
                    while !done_thread.load(Ordering::Relaxed) {
                        if let Some(bytes) = refresh_process_rss_bytes(&mut system, pid) {
                            peak_thread.fetch_max(bytes, Ordering::Relaxed);
                        }
                        thread::sleep(Duration::from_millis(2));
                    }
                    if let Some(bytes) = refresh_process_rss_bytes(&mut system, pid) {
                        peak_thread.fetch_max(bytes, Ordering::Relaxed);
                    }
                })
                .expect("failed to spawn RSS monitor");
            Self {
                done,
                peak_bytes,
                join: Some(join),
            }
        }

        fn stop(mut self) -> u64 {
            self.done.store(true, Ordering::Relaxed);
            if let Some(join) = self.join.take() {
                let _ = join.join();
            }
            self.peak_bytes.load(Ordering::Relaxed)
        }
    }

    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    #[test]
    fn native_wgpu_triposg_rmbg_load_peak_ram_under_budget_when_configured() {
        let Some(weights_root) = std::env::var_os("TRIPOSG_WEIGHTS_ROOT") else {
            eprintln!("skipping: TRIPOSG_WEIGHTS_ROOT not set");
            return;
        };
        let Some(rmbg_root) = std::env::var_os("RMBG_WEIGHTS_ROOT")
            .or_else(|| std::env::var_os("RMBG14_WEIGHTS_ROOT"))
        else {
            eprintln!("skipping: RMBG_WEIGHTS_ROOT or RMBG14_WEIGHTS_ROOT not set");
            return;
        };
        let max_ram_bytes = std::env::var("BURN_SYNTH_NATIVE_LOAD_MAX_RAM_BYTES")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(DEFAULT_WEB_MAX_HOST_RAM_BYTES);

        let cli = vec![
            "bevy_synth".to_string(),
            "--backend".to_string(),
            "wgpu".to_string(),
            "--synthesis-models".to_string(),
            "triposg".to_string(),
            "--rmbg-model".to_string(),
            "rmbg14".to_string(),
            "--rmbg-backend".to_string(),
            "gpu".to_string(),
            "--dino-backend".to_string(),
            "gpu".to_string(),
            "--weights-root".to_string(),
            weights_root.to_string_lossy().into_owned(),
            "--bg-weights-root".to_string(),
            rmbg_root.to_string_lossy().into_owned(),
        ];
        let args = crate::args::build_app_args(crate::args::Args::parse_from(cli));
        let handle = thread::Builder::new()
            .name("native-load-ram-check".to_string())
            .stack_size(256 * 1024 * 1024)
            .spawn(move || -> Result<u64, String> {
                let pid = get_current_pid().map_err(|err| format!("failed to get PID: {err}"))?;
                let mut system = System::new();
                let baseline_bytes = refresh_process_rss_bytes(&mut system, pid).unwrap_or(0);
                let monitor = ProcessMemoryMonitor::start(pid, baseline_bytes);
                let _state = build_pipeline_state::<burn_wgpu::Wgpu>(&args)?;
                thread::sleep(Duration::from_millis(20));
                let peak_bytes = monitor.stop();
                Ok(peak_bytes.saturating_sub(baseline_bytes))
            })
            .expect("failed to spawn native load RAM check thread");
        let peak_delta = handle
            .join()
            .expect("native load RAM check thread panicked");
        let peak_delta = peak_delta.expect(
            "failed to load configured model roots for RAM check; fix model roots/import before asserting RAM budget",
        );

        assert!(
            peak_delta <= max_ram_bytes,
            "native load peak host RAM delta exceeded budget: {} MiB > {} MiB",
            peak_delta as f64 / (1024.0 * 1024.0),
            max_ram_bytes as f64 / (1024.0 * 1024.0)
        );
    }
}

use std::any::TypeId;
use std::sync::Mutex;
#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
use std::sync::Once;
#[cfg(target_arch = "wasm32")]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(target_arch = "wasm32")]
use std::sync::mpsc::TryRecvError;
use std::sync::mpsc::{self, Receiver, Sender};
#[cfg(not(target_arch = "wasm32"))]
use std::thread;
#[cfg(target_arch = "wasm32")]
use std::time::Duration;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

use bevy::prelude::*;
use burn::prelude::*;
use burn::tensor::module::interpolate;
use burn::tensor::ops::{InterpolateMode, InterpolateOptions};

use burn_foreground::pipeline::{
    PrepareImageConfig, PreparedImageData, RmbgPipeline, prepare_image_data,
    prepare_image_data_from_bytes, prepare_image_tensor, prepare_image_tensor_from_bytes,
};
#[cfg(target_arch = "wasm32")]
use burn_foreground::pipeline::{
    prepare_image_tensor_async, prepare_image_tensor_from_bytes_async,
};
use burn_foreground::rmbg2::Rmbg2Pipeline;
use burn_foreground::rmbg14::import::load_rmbg_config_from_json_bytes;
#[cfg(target_arch = "wasm32")]
use burn_foreground::rmbg14::import::load_rmbg_from_burnpack_bytes;
#[cfg(not(target_arch = "wasm32"))]
use burn_foreground::rmbg14::import::load_rmbg_from_burnpack_file;
use burn_foreground::rmbg14::import::load_rmbg_processor_from_json_bytes;
use burn_foreground::rmbg14::set_rmbg_strict_interp_override;
use burn_trellis::config::TrellisQuality as TrellisRuntimeQuality;
use burn_trellis::pipeline::{
    Trellis2Pipeline, Trellis2PipelineConfig, TrellisDevice, TrellisRunOptions,
};
use burn_tripo::model::triposg::dit::TripoSGDiTConfig;
#[cfg(target_arch = "wasm32")]
use burn_tripo::model::triposg::dit::import::load_triposg_dit_from_burnpack_bytes;
#[cfg(not(target_arch = "wasm32"))]
use burn_tripo::model::triposg::dit::import::load_triposg_dit_from_burnpack_file;
use burn_tripo::model::triposg::image_encoder::import::default_dinov2_config;
#[cfg(target_arch = "wasm32")]
use burn_tripo::model::triposg::image_encoder::import::load_dinov2_processor_from_json_bytes;
#[cfg(target_arch = "wasm32")]
use burn_tripo::model::triposg::image_encoder::import::load_triposg_dinov2_from_burnpack_bytes;
#[cfg(not(target_arch = "wasm32"))]
use burn_tripo::model::triposg::image_encoder::import::{
    load_dinov2_config_from_json_bytes, load_dinov2_processor,
    load_triposg_dinov2_from_burnpack_file,
};
use burn_tripo::model::triposg::image_encoder::{DinoImageProcessor, TripoSGImageEncoder};
#[cfg(not(target_arch = "wasm32"))]
use burn_tripo::model::triposg::load_policy::BurnpackLoadPolicy;
use burn_tripo::model::triposg::scheduler::RectifiedFlowSchedulerConfig;
use burn_tripo::model::triposg::vae::TripoSGVaeConfig;
#[cfg(target_arch = "wasm32")]
use burn_tripo::model::triposg::vae::import::load_triposg_vae_decoder_from_burnpack_bytes;
#[cfg(not(target_arch = "wasm32"))]
use burn_tripo::model::triposg::vae::import::load_triposg_vae_decoder_from_burnpack_file;
#[cfg(all(test, not(target_arch = "wasm32")))]
use burn_tripo::model::triposg::vae::import::load_triposg_vae_from_burnpack_file;
use burn_tripo::pipeline::{
    geometry::{
        FlashExtractConfig, HierarchicalExtractConfig, flash_extract_geometry,
        hierarchical_extract_geometry,
    },
    mesh::{DenseGrid, Mesh as TripoMesh, grid_to_mesh, sdf_to_mesh_diff_dmc},
    runtime_parity::{
        DinoBackendChoice as SharedDinoBackendChoice, TripoSGRuntimeParityProfile,
        decimate_tripo_mesh, resolve_dino_backend, triposg_runtime_profile,
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
use crate::io::mesh_to_glb_bytes;
#[cfg(target_arch = "wasm32")]
use crate::model_loader::prefer_f16_burnpack;
#[cfg(not(target_arch = "wasm32"))]
use crate::model_loader::{
    load_optional_text_candidates_from_root, load_optional_text_from_root,
    resolve_burnpack_asset_path_from_root, resolve_burnpack_asset_path_from_root_with_preference,
};
#[cfg(not(target_arch = "wasm32"))]
use crate::paths::resolve_scribble_root;
use crate::paths::{resolve_rmbg_root, resolve_triposg_root};
use crate::state::{InferenceRequest, InferenceWorker, WorkerCommand, WorkerEvent};
#[cfg(target_arch = "wasm32")]
use burn_synth::wasm_loader::{
    DownloadTotals, WasmHostMemoryBudget, download_burnpack_asset, fetch_optional_text,
    fetch_optional_text_candidates, format_mebibytes, join_web_path, normalize_web_path,
    web_max_host_ram_bytes,
};
#[cfg(target_arch = "wasm32")]
use gloo_timers::future::TimeoutFuture;
#[cfg(target_arch = "wasm32")]
use js_sys::{Function, Promise, Reflect};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::{JsFuture, spawn_local};
#[cfg(any(target_arch = "wasm32", not(feature = "wgpu"), not(feature = "cuda")))]
#[path = "worker_unavailable.rs"]
mod worker_unavailable;
#[cfg(any(target_arch = "wasm32", not(feature = "wgpu"), not(feature = "cuda")))]
use worker_unavailable::worker_loop_backend_unavailable;
#[cfg(not(target_arch = "wasm32"))]
#[path = "worker_runtime_bridge.rs"]
mod worker_runtime_bridge;

#[cfg(feature = "wgpu")]
type WgpuRuntimeBackend = burn_wgpu::Wgpu<burn::tensor::f16, i32, u32>;

const WGPU_CHUNK_SIZE_TARGET: usize = 32_768;
const WGPU_CHUNK_SIZE_CAP: usize = 65_536;
const CUDA_CHUNK_SIZE_CAP: usize = 32_768;
// Keep runtime DINO config resolution aligned with burn_tripo canonical loader.
// Legacy image_encoder_{1,2} configs can have incompatible channel layouts (e.g. 7-channel
// patch embed) for standard TripoSG image-only weights and cause hard load failures.
const DINO_CONFIG_RELPATHS: [&str; 1] = ["image_encoder_dinov2/config.json"];
// Preprocessor config remains compatible as a fallback when dedicated DINOv2 preprocessor
// metadata is absent in older weight layouts.
#[cfg(target_arch = "wasm32")]
const DINO_PREPROCESSOR_RELPATHS: [&str; 3] = [
    "feature_extractor_dinov2/preprocessor_config.json",
    "feature_extractor_2/preprocessor_config.json",
    "feature_extractor_1/preprocessor_config.json",
];

#[cfg(target_arch = "wasm32")]
fn wasm_console_log(message: &str) {
    web_sys::console::log_1(&wasm_bindgen::JsValue::from_str(message));
}

#[cfg(target_arch = "wasm32")]
fn wasm_now_ms() -> f64 {
    js_sys::Date::now()
}

#[cfg(target_arch = "wasm32")]
fn wasm_elapsed_since(start_ms: f64) -> Duration {
    let elapsed_ms = (js_sys::Date::now() - start_ms).max(0.0);
    Duration::from_secs_f64(elapsed_ms / 1000.0)
}

#[cfg(target_arch = "wasm32")]
pub async fn wasm_webgpu_available() -> bool {
    let Some(window) = web_sys::window() else {
        return false;
    };
    let navigator = window.navigator();
    let navigator_js: wasm_bindgen::JsValue = navigator.into();
    let gpu = match Reflect::get(&navigator_js, &wasm_bindgen::JsValue::from_str("gpu")) {
        Ok(value) if !value.is_undefined() && !value.is_null() => value,
        _ => return false,
    };
    let request_adapter =
        match Reflect::get(&gpu, &wasm_bindgen::JsValue::from_str("requestAdapter")) {
            Ok(value) => value,
            Err(_) => return false,
        };
    let request_adapter = match request_adapter.dyn_into::<Function>() {
        Ok(func) => func,
        Err(_) => return false,
    };
    let promise = match request_adapter.call0(&gpu) {
        Ok(value) => value,
        Err(_) => return false,
    };
    let promise = match promise.dyn_into::<Promise>() {
        Ok(promise) => promise,
        Err(_) => return false,
    };
    match JsFuture::from(promise).await {
        Ok(adapter) => !adapter.is_null() && !adapter.is_undefined(),
        Err(_) => false,
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wgpu"))]
async fn initialize_wgpu_runtime_for_wasm() -> Result<(), String> {
    static INIT_DONE: AtomicBool = AtomicBool::new(false);
    if INIT_DONE.load(Ordering::Acquire) {
        return Ok(());
    }
    let device = burn_wgpu::WgpuDevice::default();
    let options = burn_wgpu::RuntimeOptions {
        tasks_max: 32,
        memory_config: burn_wgpu::MemoryConfiguration::ExclusivePages,
    };
    burn_wgpu::init_setup_async::<burn_wgpu::graphics::WebGpu>(&device, options).await;
    INIT_DONE.store(true, Ordering::Release);
    Ok(())
}

struct Rmbg14Artifacts {
    #[cfg(not(target_arch = "wasm32"))]
    burnpack_path: std::path::PathBuf,
    #[cfg(target_arch = "wasm32")]
    burnpack: Vec<u8>,
    config_json: Option<String>,
    processor_json: Option<String>,
}

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Copy)]
struct WasmTripoLoadOptions {
    dino_backend: DinoBackend,
    parity: TripoSGRuntimeParityProfile,
    prefer_f16_weights: bool,
}

#[cfg(target_arch = "wasm32")]
struct WasmTripoLoadContext<'a> {
    args: &'a AppArgs,
    event_tx: &'a Sender<WorkerEvent>,
    totals: &'a mut DownloadTotals,
    host_ram_budget: &'a mut WasmHostMemoryBudget,
}

#[cfg(not(target_arch = "wasm32"))]
struct TriposgArtifacts {
    vae_burnpack_path: std::path::PathBuf,
    dit_burnpack_path: std::path::PathBuf,
    dino_burnpack_path: std::path::PathBuf,
    vae_config_json: Option<String>,
    dit_config_json: Option<String>,
    scheduler_config_json: Option<String>,
    dino_config_json: Option<String>,
    dino_processor: DinoImageProcessor,
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

fn triposg_parity_profile() -> TripoSGRuntimeParityProfile {
    triposg_runtime_profile(None)
}

fn dino_processor_target_size(
    processor: &DinoImageProcessor,
    fallback_size: Option<usize>,
) -> Option<usize> {
    processor
        .crop_size
        .map(|[height, width]| height.min(width))
        .or(processor.size_shortest_edge)
        .or(fallback_size)
        .filter(|size| *size > 0)
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

#[cfg(target_arch = "wasm32")]
pub async fn infer_glb_from_image_bytes_wasm(
    args: &AppArgs,
    image_bytes: Vec<u8>,
    file_name: &str,
    timeout: Duration,
) -> Result<Vec<u8>, String> {
    if image_bytes.is_empty() {
        return Err("image bytes are empty".to_string());
    }

    let worker = start_worker(args);
    let request_id = 1_u32;
    let request = InferenceRequest {
        id: request_id,
        image_path: wasm_virtual_upload_path(file_name, request_id),
        image_contents: Some(image_bytes),
        output_path: None,
    };

    worker
        .sender
        .send(WorkerCommand::Infer(vec![request.clone()]))
        .map_err(|err| format!("failed to queue wasm inference request: {err}"))?;

    let result = wait_for_wasm_request_result(&worker, request_id, timeout).await;
    let _ = worker.sender.send(WorkerCommand::Shutdown);
    let result = result?;

    match result {
        Ok(Some(mesh)) => {
            let glb = mesh_to_glb_bytes(&mesh)
                .map_err(|err| format!("failed to serialize wasm inference mesh to GLB: {err}"))?;
            Ok(glb)
        }
        Ok(None) => Err(format!(
            "synthesis produced an empty mesh for {}",
            request.image_path.display()
        )),
        Err(err) => Err(format!(
            "synthesis inference failed for {}: {err}",
            request.image_path.display()
        )),
    }
}

#[cfg(target_arch = "wasm32")]
async fn wait_for_wasm_request_result(
    worker: &InferenceWorker,
    request_id: u32,
    timeout: Duration,
) -> Result<Result<Option<SynthMesh>, String>, String> {
    let started_ms = wasm_now_ms();
    loop {
        let next_event = {
            let receiver = worker
                .receiver
                .lock()
                .map_err(|_| "failed to lock wasm worker receiver".to_string())?;
            receiver.try_recv()
        };

        match next_event {
            Ok(event) => {
                if let Some(position) = event.requests.iter().position(|req| req.id == request_id) {
                    return event.results.get(position).cloned().ok_or_else(|| {
                        "worker returned mismatched request/result counts".to_string()
                    });
                }
            }
            Err(TryRecvError::Empty) => {
                if wasm_elapsed_since(started_ms) > timeout {
                    return Err(format!(
                        "timed out waiting for wasm inference result after {:.1}s",
                        timeout.as_secs_f32()
                    ));
                }
                TimeoutFuture::new(10).await;
            }
            Err(TryRecvError::Disconnected) => {
                return Err("wasm inference worker disconnected".to_string());
            }
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn wasm_virtual_upload_path(file_name: &str, request_id: u32) -> std::path::PathBuf {
    let mut sanitized = String::new();
    for ch in file_name.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-') {
            sanitized.push(ch);
        } else {
            sanitized.push('_');
        }
    }
    if sanitized.is_empty() {
        sanitized.push_str("upload_image");
    }
    std::path::PathBuf::from(format!("uploaded/{request_id:08}_{sanitized}"))
}

#[cfg(not(target_arch = "wasm32"))]
fn worker_loop(args: AppArgs, command_rx: Receiver<WorkerCommand>, event_tx: Sender<WorkerEvent>) {
    if worker_runtime_bridge::supports_shared_runtime(&args) {
        info!("Using shared burn_synth runtime worker path.");
        worker_runtime_bridge::worker_loop_shared_runtime(args, command_rx, event_tx);
        return;
    }
    info!("Using legacy bevy_synth_runtime worker path.");
    match args.backend {
        BackendKind::Cpu => {
            worker_loop_backend::<burn::backend::NdArray<f32>>(args, command_rx, event_tx)
        }
        BackendKind::Wgpu => {
            #[cfg(feature = "wgpu")]
            {
                worker_loop_backend::<WgpuRuntimeBackend>(args, command_rx, event_tx);
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
            if !wasm_webgpu_available().await {
                warn!(
                    "webgpu adapter unavailable on wasm32; falling back to CPU backend for wasm inference"
                );
                worker_loop_backend_wasm::<burn::backend::NdArray<f32>>(args, command_rx, event_tx)
                    .await;
                return;
            }
            #[cfg(feature = "wgpu")]
            {
                if let Err(err) = initialize_wgpu_runtime_for_wasm().await {
                    warn!("failed to initialize wasm webgpu runtime: {}", err);
                    worker_loop_backend_wasm::<burn::backend::NdArray<f32>>(
                        args, command_rx, event_tx,
                    )
                    .await;
                    return;
                }
                worker_loop_backend_wasm::<WgpuRuntimeBackend>(args, command_rx, event_tx).await;
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

    match rmbg_backend {
        RmbgBackend::Cpu => {
            let cpu_device = <burn::backend::NdArray<f32> as Backend>::Device::default();
            let model =
                load_rmbg_from_burnpack_file(&cpu_device, &artifacts.burnpack_path, &rmbg_config)
                    .map_err(|err| format!("failed to load RMBG burnpack on CPU: {err}"))?;
            <burn::backend::NdArray<f32> as Backend>::sync(&cpu_device);
            <burn::backend::NdArray<f32> as Backend>::memory_cleanup(&cpu_device);
            let rmbg = RmbgPipeline::new(model, rmbg_processor);
            Ok((Some(rmbg), None))
        }
        RmbgBackend::Gpu | RmbgBackend::Auto => {
            let model =
                load_rmbg_from_burnpack_file(device, &artifacts.burnpack_path, &rmbg_config)
                    .map_err(|err| format!("failed to load RMBG burnpack: {err}"))?;
            B::sync(device);
            B::memory_cleanup(device);
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
    parity: TripoSGRuntimeParityProfile,
) -> Result<(Option<DinoCpuState>, TripoSGPipeline<B>), String> {
    let weights_root = resolve_triposg_root(args.weights_root.as_ref());
    let artifacts = load_triposg_artifacts_from_root(&weights_root, parity.burnpack_policy)?;
    build_triposg_pipeline_from_artifacts::<B>(device, dino_backend, artifacts, parity)
}

#[cfg(not(target_arch = "wasm32"))]
fn load_rmbg14_artifacts_from_root(root: &std::path::Path) -> Result<Rmbg14Artifacts, String> {
    let burnpack_path = resolve_burnpack_asset_path_from_root(root, "model.safetensors")?;
    let config_json = load_optional_text_from_root(root, "config.json")?;
    let processor_json = load_optional_text_from_root(root, "preprocessor_config.json")?;
    Ok(Rmbg14Artifacts {
        burnpack_path,
        config_json,
        processor_json,
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn load_triposg_artifacts_from_root(
    root: &std::path::Path,
    burnpack_policy: BurnpackLoadPolicy,
) -> Result<TriposgArtifacts, String> {
    let prefer_f16 = burnpack_policy.precision.prefer_f16();
    let vae_burnpack_path = resolve_burnpack_asset_path_from_root_with_preference(
        root,
        "vae/diffusion_pytorch_model.safetensors",
        prefer_f16,
    )?;
    let dit_burnpack_path = resolve_burnpack_asset_path_from_root_with_preference(
        root,
        "transformer/diffusion_pytorch_model.safetensors",
        prefer_f16,
    )?;
    let dino_burnpack_path = resolve_burnpack_asset_path_from_root_with_preference(
        root,
        "image_encoder_dinov2/model.safetensors",
        prefer_f16,
    )?;
    let vae_config_json = load_optional_text_from_root(root, "vae/config.json")?;
    let dit_config_json = load_optional_text_from_root(root, "transformer/config.json")?;
    let scheduler_config_json =
        load_optional_text_from_root(root, "scheduler/scheduler_config.json")?;
    let dino_config_json = load_optional_text_candidates_from_root(root, &DINO_CONFIG_RELPATHS)?;
    let dino_processor = load_dinov2_processor(root)
        .map_err(|err| format!("failed to load DINO preprocessor config: {err}"))?;

    Ok(TriposgArtifacts {
        vae_burnpack_path,
        dit_burnpack_path,
        dino_burnpack_path,
        vae_config_json,
        dit_config_json,
        scheduler_config_json,
        dino_config_json,
        dino_processor,
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn build_triposg_pipeline_from_artifacts<B: Backend>(
    device: &B::Device,
    dino_backend: DinoBackend,
    artifacts: TriposgArtifacts,
    parity: TripoSGRuntimeParityProfile,
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

    let parsed_dino_config = artifacts
        .dino_config_json
        .as_ref()
        .and_then(|json| load_dinov2_config_from_json_bytes(json.as_bytes()));
    let dino_fallback_size = parsed_dino_config.as_ref().map(|cfg| cfg.image_size);
    let mut dino_config = parsed_dino_config
        .clone()
        .unwrap_or_else(default_dinov2_config);
    let dino_processor = artifacts
        .dino_processor
        .with_strict_preprocess(parity.strict_dino_preprocess);
    if let Some(target_size) = dino_processor_target_size(&dino_processor, dino_fallback_size) {
        let patch = dino_config.patch_size.max(1);
        let grid = target_size / patch;
        if grid > 0 {
            dino_config.positional_encoding_interpolate.output_size = Some([grid, grid]);
            info!(
                "Configured DINO positional interpolation grid to {}x{} (target_size={}, patch_size={}).",
                grid, grid, target_size, patch
            );
        }
    }

    let vae = load_triposg_vae_decoder_from_burnpack_file(
        &vae_config,
        device,
        &artifacts.vae_burnpack_path,
    )
    .map_err(|err| format!("failed to load TripoSG VAE burnpack: {err}"))?;
    B::sync(device);
    B::memory_cleanup(device);
    let dit =
        load_triposg_dit_from_burnpack_file(&dit_config, device, &artifacts.dit_burnpack_path)
            .map_err(|err| format!("failed to load TripoSG DiT burnpack: {err}"))?;
    B::sync(device);
    B::memory_cleanup(device);
    let scheduler = scheduler_config.init();
    let (image_encoder, dino_cpu) = if matches!(dino_backend, DinoBackend::Cpu) {
        let cpu_device = <burn::backend::NdArray<f32> as Backend>::Device::default();
        let encoder = load_triposg_dinov2_from_burnpack_file(
            &cpu_device,
            dino_config.clone(),
            &artifacts.dino_burnpack_path,
        )
        .map_err(|err| format!("failed to load DINOv2 burnpack on CPU: {err}"))?;
        <burn::backend::NdArray<f32> as Backend>::sync(&cpu_device);
        <burn::backend::NdArray<f32> as Backend>::memory_cleanup(&cpu_device);
        (
            None,
            Some(DinoCpuState {
                device: cpu_device,
                encoder,
                processor: dino_processor.clone(),
            }),
        )
    } else {
        let image_encoder = load_triposg_dinov2_from_burnpack_file(
            device,
            dino_config.clone(),
            &artifacts.dino_burnpack_path,
        )
        .map_err(|err| format!("failed to load DINOv2 burnpack: {err}"))?;
        B::sync(device);
        B::memory_cleanup(device);
        (Some(image_encoder), None)
    };
    let triposg = TripoSGPipeline::new_with_optional_image_encoder(
        vae,
        dit,
        scheduler,
        image_encoder,
        dino_processor.clone(),
    );

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
                wasm_console_log(&format!(
                    "worker_loop_wasm received infer requests={}",
                    requests.len()
                ));
                let start_ms = wasm_now_ms();
                let results = match state.as_mut() {
                    Ok(state) => run_inference_with_state_wasm(state, &args, &requests).await,
                    Err(err) => vec![Err(err.clone()); requests.len()],
                };
                let _ = event_tx.send(WorkerEvent {
                    requests,
                    results,
                    elapsed: wasm_elapsed_since(start_ms),
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
    let triposg_parity = triposg_parity_profile();
    set_rmbg_strict_interp_override(Some(triposg_parity.strict_rmbg_interp));

    let synthesis_models = args.synthesis_models.clone();
    let synthesis_order = synthesis_attempt_order(&synthesis_models)?;
    let wants_triposg = synthesis_order
        .iter()
        .any(|model| matches!(model, SynthesisModel::Triposg));

    let dino_backend = map_dino_backend_choice_back(resolve_dino_backend::<B>(
        map_dino_backend_choice(args.dino_backend),
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
    let requested_rmbg_backend = match args.rmbg_backend {
        RmbgBackend::Auto => RmbgBackend::Gpu,
        other => other,
    };
    let rmbg_backend = requested_rmbg_backend;

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
                match load_triposg_pipeline::<B>(args, &device, dino_backend, triposg_parity) {
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
async fn build_pipeline_state_wasm<B: Backend>(
    args: &AppArgs,
    event_tx: &Sender<WorkerEvent>,
) -> Result<PipelineState<B>, String> {
    wasm_console_log("build_pipeline_state_wasm start");
    configure_cubecl_autotune::<B>();
    let bounds = parse_bounds(&args.bounds).map_err(|err| err.to_string())?;
    let device = B::Device::default();
    if let Some(seed) = args.seed {
        B::seed(&device, seed);
    }
    let triposg_parity = triposg_parity_profile();
    set_rmbg_strict_interp_override(Some(triposg_parity.strict_rmbg_interp));
    let prefer_f16_weights = is_wgpu_backend::<B>()
        && (triposg_parity.burnpack_policy.precision.prefer_f16() || prefer_f16_burnpack());

    let synthesis_models = args.synthesis_models.clone();
    let synthesis_order = synthesis_attempt_order(&synthesis_models)?;

    if args.text_embeds.is_some() || args.prompt.is_some() {
        return Err("text/scribble mode is not supported on wasm yet".to_string());
    }

    let dino_backend = map_dino_backend_choice_back(resolve_dino_backend::<B>(
        map_dino_backend_choice(args.dino_backend),
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
    let mut rmbg_backend = match args.rmbg_backend {
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

    let rmbg_artifacts = load_rmbg14_artifacts_wasm(
        &rmbg_root_url,
        prefer_f16_weights,
        event_tx,
        &mut totals,
        &mut host_ram_budget,
    )
    .await?;
    wasm_console_log("build_pipeline_state_wasm downloaded RMBG artifacts");
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
            wasm_console_log("build_pipeline_state_wasm loading RMBG on CPU");
            let cpu_device = <burn::backend::NdArray<f32> as Backend>::Device::default();
            let model =
                match load_rmbg_from_burnpack_bytes(&cpu_device, rmbg_burnpack, &rmbg_config) {
                    Ok(model) => model,
                    Err(err) => {
                        host_ram_budget.release_retained(rmbg_burnpack_bytes);
                        return Err(format!("failed to load RMBG burnpack on CPU: {err}"));
                    }
                };
            <burn::backend::NdArray<f32> as Backend>::sync(&cpu_device);
            <burn::backend::NdArray<f32> as Backend>::memory_cleanup(&cpu_device);
            host_ram_budget.release_retained(rmbg_burnpack_bytes);
            (Some(RmbgPipeline::new(model, rmbg_processor.clone())), None)
        }
        RmbgBackend::Gpu | RmbgBackend::Auto => {
            wasm_console_log("build_pipeline_state_wasm loading RMBG on GPU");
            let cpu_fallback_bytes = rmbg_burnpack.clone();
            let model = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                load_rmbg_from_burnpack_bytes(&device, rmbg_burnpack, &rmbg_config)
            }));
            match model {
                Ok(Ok(model)) => {
                    // Avoid blocking sync/readback semantics on wasm GPU backends.
                    host_ram_budget.release_retained(rmbg_burnpack_bytes);
                    let mut pipeline = RmbgPipeline::new(model, rmbg_processor);
                    cap_rmbg14_processor_for_backend::<B>(&mut pipeline, rmbg_backend);
                    wasm_console_log("build_pipeline_state_wasm RMBG GPU load complete");
                    (None, Some(pipeline))
                }
                Ok(Err(err)) => {
                    warn!(
                        "RMBG GPU load failed on wasm ({}); falling back to CPU RMBG pipeline.",
                        err
                    );
                    send_worker_status(
                        event_tx,
                        format!(
                            "RMBG GPU load failed on wasm ({}); falling back to CPU RMBG pipeline.",
                            err
                        ),
                    );
                    rmbg_backend = RmbgBackend::Cpu;
                    let cpu_device = <burn::backend::NdArray<f32> as Backend>::Device::default();
                    let model = load_rmbg_from_burnpack_bytes(
                        &cpu_device,
                        cpu_fallback_bytes,
                        &rmbg_config,
                    )
                    .map_err(|cpu_err| {
                        format!("failed to load RMBG burnpack on CPU after GPU failure: {cpu_err}")
                    })?;
                    <burn::backend::NdArray<f32> as Backend>::sync(&cpu_device);
                    <burn::backend::NdArray<f32> as Backend>::memory_cleanup(&cpu_device);
                    host_ram_budget.release_retained(rmbg_burnpack_bytes);
                    (Some(RmbgPipeline::new(model, rmbg_processor.clone())), None)
                }
                Err(_) => {
                    warn!("RMBG GPU load panicked on wasm; falling back to CPU RMBG pipeline.");
                    send_worker_status(
                        event_tx,
                        "RMBG GPU load panicked on wasm; falling back to CPU RMBG pipeline.",
                    );
                    rmbg_backend = RmbgBackend::Cpu;
                    let cpu_device = <burn::backend::NdArray<f32> as Backend>::Device::default();
                    let model = load_rmbg_from_burnpack_bytes(
                        &cpu_device,
                        cpu_fallback_bytes,
                        &rmbg_config,
                    )
                    .map_err(|cpu_err| {
                        format!("failed to load RMBG burnpack on CPU after GPU panic: {cpu_err}")
                    })?;
                    <burn::backend::NdArray<f32> as Backend>::sync(&cpu_device);
                    <burn::backend::NdArray<f32> as Backend>::memory_cleanup(&cpu_device);
                    host_ram_budget.release_retained(rmbg_burnpack_bytes);
                    (Some(RmbgPipeline::new(model, rmbg_processor.clone())), None)
                }
            }
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
                let options = WasmTripoLoadOptions {
                    dino_backend,
                    parity: triposg_parity,
                    prefer_f16_weights,
                };
                let mut context = WasmTripoLoadContext {
                    args,
                    event_tx,
                    totals: &mut totals,
                    host_ram_budget: &mut host_ram_budget,
                };
                match load_triposg_pipeline_wasm::<B>(&device, options, &mut context).await {
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
    device: &B::Device,
    options: WasmTripoLoadOptions,
    context: &mut WasmTripoLoadContext<'_>,
) -> Result<(Option<DinoCpuState>, TripoSGPipeline<B>), String> {
    let triposg_root = resolve_triposg_root(context.args.weights_root.as_ref());
    let triposg_root_url = normalize_web_path(&triposg_root);
    let mut emit_status = |message: String| send_worker_status(context.event_tx, message);

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
    let dino_processor = if let Some(json) = dino_preproc_json.as_ref() {
        load_dinov2_processor_from_json_bytes(json.as_bytes(), dino_fallback_size)
            .map_err(|err| format!("failed to parse DINO preprocessor config: {err}"))?
    } else {
        DinoImageProcessor::default()
    }
    .with_strict_preprocess(options.parity.strict_dino_preprocess);
    if let Some(target_size) = dino_processor_target_size(&dino_processor, dino_fallback_size) {
        let patch = dino_config.patch_size.max(1);
        let grid = target_size / patch;
        if grid > 0 {
            dino_config.positional_encoding_interpolate.output_size = Some([grid, grid]);
            info!(
                "Configured DINO positional interpolation grid to {}x{} (target_size={}, patch_size={}).",
                grid, grid, target_size, patch
            );
        }
    }

    let vae_burnpack = download_burnpack_asset(
        &join_web_path(&triposg_root_url, "vae/diffusion_pytorch_model.safetensors"),
        "TripoSG VAE",
        options.prefer_f16_weights,
        context.totals,
        context.host_ram_budget,
        &mut emit_status,
    )
    .await?;
    let vae_burnpack_bytes = vae_burnpack.len() as u64;
    context
        .host_ram_budget
        .reserve_retained(vae_burnpack_bytes, "retaining TripoSG VAE burnpack bytes")?;
    let vae = match load_triposg_vae_decoder_from_burnpack_bytes(&vae_config, device, vae_burnpack)
    {
        Ok(model) => model,
        Err(err) => {
            context.host_ram_budget.release_retained(vae_burnpack_bytes);
            return Err(format!("failed to load TripoSG VAE burnpack: {err}"));
        }
    };
    // Avoid blocking sync/readback semantics on wasm GPU backends.
    context.host_ram_budget.release_retained(vae_burnpack_bytes);

    let dit_burnpack = download_burnpack_asset(
        &join_web_path(
            &triposg_root_url,
            "transformer/diffusion_pytorch_model.safetensors",
        ),
        "TripoSG DiT",
        options.prefer_f16_weights,
        context.totals,
        context.host_ram_budget,
        &mut emit_status,
    )
    .await?;
    let dit_burnpack_bytes = dit_burnpack.len() as u64;
    context
        .host_ram_budget
        .reserve_retained(dit_burnpack_bytes, "retaining TripoSG DiT burnpack bytes")?;
    let dit = match load_triposg_dit_from_burnpack_bytes(&dit_config, device, dit_burnpack) {
        Ok(model) => model,
        Err(err) => {
            context.host_ram_budget.release_retained(dit_burnpack_bytes);
            return Err(format!("failed to load TripoSG DiT burnpack: {err}"));
        }
    };
    // Avoid blocking sync/readback semantics on wasm GPU backends.
    context.host_ram_budget.release_retained(dit_burnpack_bytes);

    let (image_encoder, dino_cpu) = if matches!(options.dino_backend, DinoBackend::Cpu) {
        let cpu_device = <burn::backend::NdArray<f32> as Backend>::Device::default();
        let dino_cpu_burnpack = download_burnpack_asset(
            &join_web_path(&triposg_root_url, "image_encoder_dinov2/model.safetensors"),
            "DINOv2 (CPU)",
            options.prefer_f16_weights,
            context.totals,
            context.host_ram_budget,
            &mut emit_status,
        )
        .await?;
        let dino_cpu_burnpack_bytes = dino_cpu_burnpack.len() as u64;
        context.host_ram_budget.reserve_retained(
            dino_cpu_burnpack_bytes,
            "retaining DINOv2 CPU burnpack bytes",
        )?;
        let encoder = match load_triposg_dinov2_from_burnpack_bytes(
            &cpu_device,
            dino_config.clone(),
            dino_cpu_burnpack,
        ) {
            Ok(encoder) => encoder,
            Err(err) => {
                context
                    .host_ram_budget
                    .release_retained(dino_cpu_burnpack_bytes);
                return Err(format!("failed to load DINOv2 burnpack on CPU: {err}"));
            }
        };
        <burn::backend::NdArray<f32> as Backend>::sync(&cpu_device);
        <burn::backend::NdArray<f32> as Backend>::memory_cleanup(&cpu_device);
        context
            .host_ram_budget
            .release_retained(dino_cpu_burnpack_bytes);
        (
            None,
            Some(DinoCpuState {
                device: cpu_device,
                encoder,
                processor: dino_processor.clone(),
            }),
        )
    } else {
        let dino_gpu_burnpack = download_burnpack_asset(
            &join_web_path(&triposg_root_url, "image_encoder_dinov2/model.safetensors"),
            "DINOv2",
            options.prefer_f16_weights,
            context.totals,
            context.host_ram_budget,
            &mut emit_status,
        )
        .await?;
        let dino_gpu_burnpack_bytes = dino_gpu_burnpack.len() as u64;
        context.host_ram_budget.reserve_retained(
            dino_gpu_burnpack_bytes,
            "retaining DINOv2 GPU burnpack bytes",
        )?;
        let image_encoder = match load_triposg_dinov2_from_burnpack_bytes(
            device,
            dino_config.clone(),
            dino_gpu_burnpack,
        ) {
            Ok(encoder) => encoder,
            Err(err) => {
                context
                    .host_ram_budget
                    .release_retained(dino_gpu_burnpack_bytes);
                return Err(format!("failed to load DINOv2 burnpack: {err}"));
            }
        };
        // Avoid blocking sync/readback semantics on wasm GPU backends.
        context
            .host_ram_budget
            .release_retained(dino_gpu_burnpack_bytes);
        (Some(image_encoder), None)
    };

    let scheduler = scheduler_config.init();
    let triposg = TripoSGPipeline::new_with_optional_image_encoder(
        vae,
        dit,
        scheduler,
        image_encoder,
        dino_processor.clone(),
    );

    Ok((dino_cpu, triposg))
}

#[cfg(target_arch = "wasm32")]
async fn load_rmbg14_artifacts_wasm(
    rmbg_root_url: &str,
    prefer_f16_weights: bool,
    event_tx: &Sender<WorkerEvent>,
    totals: &mut DownloadTotals,
    host_ram_budget: &mut WasmHostMemoryBudget,
) -> Result<Rmbg14Artifacts, String> {
    let mut emit_status = |message: String| send_worker_status(event_tx, message);
    let burnpack = download_burnpack_asset(
        &join_web_path(rmbg_root_url, "model.safetensors"),
        "RMBG",
        prefer_f16_weights,
        totals,
        host_ram_budget,
        &mut emit_status,
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

#[cfg(not(target_arch = "wasm32"))]
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

#[cfg(target_arch = "wasm32")]
async fn run_inference_with_state_wasm<B: Backend>(
    state: &mut PipelineState<B>,
    args: &AppArgs,
    requests: &[InferenceRequest],
) -> Vec<Result<Option<SynthMesh>, String>> {
    wasm_console_log(&format!(
        "run_inference_with_state_wasm enter requests={}",
        requests.len()
    ));
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
        info!(
            "wasm inference preparing request idx={} has_bytes={} path={}",
            idx,
            request.image_contents.is_some(),
            request.image_path.display()
        );
        match prepare_request_wasm(state, &device, request, &prepare_config).await {
            Ok((image, prepared_cpu)) => {
                info!("wasm inference prepared request idx={} successfully", idx);
                batch_indices.push(idx);
                batch_images.push(image);
                batch_prepared.push(prepared_cpu);
            }
            Err(err) => {
                warn!(
                    "wasm inference failed to prepare request idx={}: {}",
                    idx, err
                );
                results[idx] = Some(Err(err));
            }
        }
    }

    if !batch_images.is_empty() {
        let batch_results = if state.scribble.is_some() {
            run_scribble_batch(state, args, &batch_images)
        } else {
            run_triposg_batch_wasm(state, args, &batch_images, &batch_prepared).await
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

#[cfg(target_arch = "wasm32")]
async fn prepare_request_wasm<B: Backend>(
    state: &PipelineState<B>,
    device: &B::Device,
    request: &InferenceRequest,
    config: &PrepareImageConfig,
) -> Result<(Tensor<B, 4>, Option<PreparedImageData>), String> {
    wasm_console_log(&format!(
        "prepare_request_wasm enter model={:?} backend={:?} has_bytes={} rmbg14_device_loaded={}",
        state.rmbg_model,
        state.rmbg_backend,
        request.image_contents.is_some(),
        state.rmbg14_device.is_some()
    ));
    if matches!(state.rmbg_model, RmbgModel::Rmbg14)
        && matches!(state.rmbg_backend, RmbgBackend::Gpu | RmbgBackend::Auto)
    {
        info!(
            "wasm prepare_request async RMBG path: model={:?} backend={:?} has_bytes={} rmbg14_device_loaded={}",
            state.rmbg_model,
            state.rmbg_backend,
            request.image_contents.is_some(),
            state.rmbg14_device.is_some()
        );
        if let Some(bytes) = request.image_contents.as_deref() {
            let image = prepare_image_tensor_from_bytes_async::<B>(
                bytes,
                state.rmbg14_device.as_ref(),
                device,
                config,
            )
            .await
            .map_err(|err| format!("failed to prepare image bytes: {err}"))?;
            return Ok((image, None));
        }

        let image = prepare_image_tensor_async::<B>(
            &request.image_path,
            state.rmbg14_device.as_ref(),
            device,
            config,
        )
        .await
        .map_err(|err| format!("failed to prepare image: {err}"))?;
        return Ok((image, None));
    }

    warn!(
        "wasm prepare_request sync fallback path: model={:?} backend={:?} has_bytes={} rmbg14_device_loaded={}",
        state.rmbg_model,
        state.rmbg_backend,
        request.image_contents.is_some(),
        state.rmbg14_device.is_some()
    );
    prepare_request(state, device, request, config)
}

#[cfg(not(target_arch = "wasm32"))]
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
        pipeline
            .image_encoder
            .as_ref()
            .ok_or_else(|| "DINO GPU encoder not loaded".to_string())?
            .forward(processed)
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

#[cfg(target_arch = "wasm32")]
async fn run_triposg_batch_wasm<B: Backend>(
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
        pipeline
            .image_encoder
            .as_ref()
            .ok_or_else(|| "DINO GPU encoder not loaded".to_string())?
            .forward(processed)
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
        let (mesh, grid) = decode_triposg_mesh_wasm(pipeline, sample, &decode_config).await?;
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

#[cfg(not(target_arch = "wasm32"))]
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

#[cfg(target_arch = "wasm32")]
async fn decode_triposg_mesh_wasm<B: Backend>(
    pipeline: &TripoSGPipeline<B>,
    latents: Tensor<B, 3>,
    config: &DecodeConfig<'_>,
) -> Result<(Option<TripoMesh>, DenseGrid), String> {
    if !matches!(config.mesh_mode, MeshMode::Dense) {
        warn!(
            "WASM decode forcing dense mesh mode from {:?} to avoid sync-only readback ops.",
            config.mesh_mode
        );
    }
    let values = decode_grid_values_chunked_async(
        &latents,
        &pipeline.vae,
        config.bounds,
        config.resolution,
        config.chunk_size,
    )
    .await?;
    let grid = DenseGrid {
        values,
        size: [config.resolution, config.resolution, config.resolution],
        bounds: config.bounds,
    };
    let mesh = grid_to_mesh(&grid, 0.0);
    Ok((mesh, grid))
}

#[cfg(target_arch = "wasm32")]
async fn decode_grid_values_chunked_async<B: Backend>(
    latents: &Tensor<B, 3>,
    vae: &burn_tripo::model::triposg::vae::TripoSGVae<B>,
    bounds: [f32; 6],
    resolution: usize,
    chunk_size: usize,
) -> Result<Vec<f32>, String> {
    let total = resolution * resolution * resolution;
    let device = latents.device();
    let mut values = vec![0.0f32; total];
    let step_x = dense_grid_step(bounds[0], bounds[3], resolution);
    let step_y = dense_grid_step(bounds[1], bounds[4], resolution);
    let step_z = dense_grid_step(bounds[2], bounds[5], resolution);

    let mut coords = Vec::with_capacity(chunk_size.saturating_mul(3));
    let mut chunk_start = 0usize;
    for idx in 0..total {
        let (x, y, z) = dense_grid_index_to_xyz(idx, resolution);
        coords.push(bounds[0] + step_x * x as f32);
        coords.push(bounds[1] + step_y * y as f32);
        coords.push(bounds[2] + step_z * z as f32);
        let count = coords.len() / 3;
        if count < chunk_size {
            continue;
        }
        let end = chunk_start + count;
        write_decoded_chunk_contiguous_async(
            latents,
            vae,
            &coords,
            &device,
            &mut values[chunk_start..end],
        )
        .await?;
        coords.clear();
        chunk_start = end;
    }

    if !coords.is_empty() {
        let count = coords.len() / 3;
        let end = chunk_start + count;
        write_decoded_chunk_contiguous_async(
            latents,
            vae,
            &coords,
            &device,
            &mut values[chunk_start..end],
        )
        .await?;
    }

    Ok(values)
}

#[cfg(target_arch = "wasm32")]
async fn write_decoded_chunk_contiguous_async<B: Backend>(
    latents: &Tensor<B, 3>,
    vae: &burn_tripo::model::triposg::vae::TripoSGVae<B>,
    coords: &[f32],
    device: &B::Device,
    output_slice: &mut [f32],
) -> Result<(), String> {
    let count = coords.len() / 3;
    if count == 0 {
        return Ok(());
    }
    let coords_tensor = Tensor::<B, 1>::from_floats(coords, device)
        .reshape([count as i32, 3])
        .unsqueeze_dim(0);
    let decoded = vae.decode(coords_tensor, latents.clone(), None);
    let data = decoded
        .into_data_async()
        .await
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|err| format!("failed to convert decoded grid: {err:?}"))?;
    output_slice.copy_from_slice(&data[..output_slice.len()]);
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn dense_grid_step(min: f32, max: f32, resolution: usize) -> f32 {
    if resolution <= 1 {
        return 0.0;
    }
    (max - min) / (resolution as f32 - 1.0)
}

#[cfg(target_arch = "wasm32")]
fn dense_grid_index_to_xyz(idx: usize, resolution: usize) -> (usize, usize, usize) {
    let area = resolution * resolution;
    let z = idx / area;
    let rem = idx - z * area;
    let y = rem / resolution;
    let x = rem - y * resolution;
    (x, y, z)
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
            WGPU_CHUNK_SIZE_TARGET
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
    let _ = TypeId::of::<B>();
    std::any::type_name::<B>()
        .to_ascii_lowercase()
        .contains("wgpu")
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
    let _ = TypeId::of::<B>();
    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    if is_wgpu_backend::<B>() {
        configure_wgpu_runtime_memory_profile();
    }
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
fn configure_wgpu_runtime_memory_profile() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let device = burn_wgpu::WgpuDevice::default();
        let options = burn_wgpu::RuntimeOptions {
            tasks_max: 32,
            memory_config: burn_wgpu::MemoryConfiguration::ExclusivePages,
        };
        let _setup =
            burn_wgpu::init_setup::<burn_wgpu::graphics::AutoGraphicsApi>(&device, options);
        info!("Configured WGPU runtime memory profile: ExclusivePages.");
    });
}

fn prepare_image_config_for_backend<B: Backend>(
    rmbg_model: RmbgModel,
    rmbg_backend: RmbgBackend,
) -> PrepareImageConfig {
    let mut config = PrepareImageConfig::default();
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
    type WgpuLoadTestBackend = burn_wgpu::Wgpu<burn::tensor::f16, i32, u32>;
    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    use std::fs;
    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    use std::path::Path;
    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    use std::path::PathBuf;
    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    use std::sync::Arc;
    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    use std::thread;
    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    use std::time::Duration;
    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    use std::time::{SystemTime, UNIX_EPOCH};
    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System, get_current_pid};
    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    const ONE_GIB: u64 = 1024 * 1024 * 1024;

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

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn native_triposg_loader_canonicalizes_legacy_dino_preprocessor() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "bevy_synth_runtime_dino_loader_test_{}_{}",
            std::process::id(),
            nonce
        ));

        std::fs::create_dir_all(root.join("vae")).expect("create vae dir");
        std::fs::create_dir_all(root.join("transformer")).expect("create transformer dir");
        std::fs::create_dir_all(root.join("image_encoder_dinov2")).expect("create dino dir");
        std::fs::create_dir_all(root.join("feature_extractor_2"))
            .expect("create legacy preprocessor dir");

        // Minimal placeholders are enough here: this test verifies loader/path policy and
        // preprocessor canonicalization, not burnpack deserialization.
        std::fs::write(root.join("vae/diffusion_pytorch_model_f16.bpk"), b"stub")
            .expect("write vae burnpack");
        std::fs::write(
            root.join("transformer/diffusion_pytorch_model_f16.bpk"),
            b"stub",
        )
        .expect("write transformer burnpack");
        std::fs::write(root.join("image_encoder_dinov2/model_f16.bpk"), b"stub")
            .expect("write dino burnpack");
        std::fs::write(
            root.join("feature_extractor_2/preprocessor_config.json"),
            r#"{
                "image_processor_type": "BitImageProcessor",
                "crop_size": { "height": 512, "width": 512 },
                "do_center_crop": true,
                "do_resize": true,
                "size": { "shortest_edge": 512 }
            }"#,
        )
        .expect("write legacy dino preprocessor");

        let artifacts = load_triposg_artifacts_from_root(&root, BurnpackLoadPolicy::default())
            .expect("load triposg artifacts with canonicalized dino processor");
        assert_eq!(artifacts.dino_processor.size_shortest_edge, Some(256));
        assert_eq!(artifacts.dino_processor.crop_size, Some([224, 224]));

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn apply_mesh_decimation_reduces_untextured_mesh() {
        let mesh = SynthMesh::from(sample_mesh_grid(20));
        let original_faces = mesh.mesh.faces.len();
        let output =
            apply_mesh_decimation(Some(mesh), Some(300)).expect("decimation should return mesh");
        assert!(output.mesh.faces.len() <= 300);
        assert!(!output.mesh.faces.is_empty());
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
    fn has_burnpack_or_manifest(root: &Path, base_safetensors_rel: &str, prefer_f16: bool) -> bool {
        let candidates =
            crate::model_loader::candidate_burnpack_names(base_safetensors_rel, prefer_f16);
        for candidate in candidates {
            let candidate_path = root.join(&candidate);
            if candidate_path.exists() {
                return true;
            }
            for manifest in crate::model_loader::burnpack_manifest_candidates(&candidate) {
                if root.join(manifest).exists() {
                    return true;
                }
            }
        }
        false
    }

    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    struct TempDirGuard {
        path: PathBuf,
    }

    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    impl TempDirGuard {
        fn new(path: PathBuf) -> Self {
            Self { path }
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    impl Drop for TempDirGuard {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    fn unique_temp_weights_root() -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after unix epoch")
            .as_nanos();
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/native_load_test_weights")
            .join(format!("triposg_{nanos}"))
    }

    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    fn hard_link_or_copy(source: &Path, destination: &Path) -> Result<(), String> {
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)
                .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
        }
        match fs::hard_link(source, destination) {
            Ok(()) => Ok(()),
            Err(_) => fs::copy(source, destination).map(|_| ()).map_err(|err| {
                format!(
                    "failed to link/copy {} -> {}: {err}",
                    source.display(),
                    destination.display()
                )
            }),
        }
    }

    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    fn link_optional_file(
        source_root: &Path,
        dest_root: &Path,
        rel_path: &str,
    ) -> Result<(), String> {
        let source = source_root.join(rel_path);
        if !source.exists() {
            return Ok(());
        }
        hard_link_or_copy(&source, &dest_root.join(rel_path))
    }

    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    fn link_burnpack_asset_tree(
        source_root: &Path,
        dest_root: &Path,
        base_safetensors_rel: &str,
        prefer_f16: bool,
    ) -> Result<(), String> {
        let candidates =
            crate::model_loader::candidate_burnpack_names(base_safetensors_rel, prefer_f16);
        for candidate in candidates {
            let source_candidate = source_root.join(&candidate);
            if source_candidate.exists() {
                return hard_link_or_copy(&source_candidate, &dest_root.join(&candidate));
            }
            for manifest_rel in crate::model_loader::burnpack_manifest_candidates(&candidate) {
                let source_manifest = source_root.join(&manifest_rel);
                if !source_manifest.exists() {
                    continue;
                }

                let manifest_bytes = fs::read(&source_manifest).map_err(|err| {
                    format!(
                        "failed to read shard manifest {}: {err}",
                        source_manifest.display()
                    )
                })?;
                let manifest = crate::model_loader::parse_shard_manifest_bytes(
                    &manifest_bytes,
                    &source_manifest.display().to_string(),
                )?;
                hard_link_or_copy(&source_manifest, &dest_root.join(&manifest_rel))?;
                for shard in manifest.shard_entries() {
                    let source_shard = source_root.join(shard.path());
                    if !source_shard.exists() {
                        return Err(format!(
                            "missing shard {} referenced by {}",
                            source_shard.display(),
                            source_manifest.display()
                        ));
                    }
                    hard_link_or_copy(&source_shard, &dest_root.join(shard.path()))?;
                }
                return Ok(());
            }
        }
        Err(format!(
            "missing burnpack/manifest for {base_safetensors_rel} under {}",
            source_root.display()
        ))
    }

    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    fn prepare_isolated_triposg_weights_root(source_root: &Path) -> Result<TempDirGuard, String> {
        let isolated_root = unique_temp_weights_root();
        fs::create_dir_all(&isolated_root).map_err(|err| {
            format!(
                "failed to create isolated TripoSG test root {}: {err}",
                isolated_root.display()
            )
        })?;

        link_burnpack_asset_tree(
            source_root,
            &isolated_root,
            "vae/diffusion_pytorch_model.safetensors",
            true,
        )?;
        link_burnpack_asset_tree(
            source_root,
            &isolated_root,
            "transformer/diffusion_pytorch_model.safetensors",
            true,
        )?;
        link_burnpack_asset_tree(
            source_root,
            &isolated_root,
            "image_encoder_dinov2/model.safetensors",
            true,
        )?;

        // Keep only canonical configs needed for VAE/DiT/scheduler. DINO config is omitted
        // intentionally so runtime uses `default_dinov2_config` and avoids legacy mismatch.
        link_optional_file(source_root, &isolated_root, "vae/config.json")?;
        link_optional_file(source_root, &isolated_root, "transformer/config.json")?;
        link_optional_file(
            source_root,
            &isolated_root,
            "scheduler/scheduler_config.json",
        )?;

        Ok(TempDirGuard::new(isolated_root))
    }

    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    fn build_native_wgpu_load_args() -> (crate::args::AppArgs, TempDirGuard) {
        let source_weights_root = resolve_triposg_root(None);
        let rmbg_root = resolve_rmbg_root(None, RmbgModel::Rmbg14);

        assert!(
            source_weights_root.is_dir(),
            "TripoSG weights root missing: {}",
            source_weights_root.display()
        );
        assert!(
            rmbg_root.is_dir(),
            "RMBG-1.4 weights root missing: {}",
            rmbg_root.display()
        );
        let isolated_root = prepare_isolated_triposg_weights_root(&source_weights_root).expect(
            "failed to prepare isolated TripoSG test root with canonical burnpacks/configs",
        );

        assert!(
            has_burnpack_or_manifest(&rmbg_root, "model.safetensors", true),
            "missing RMBG burnpack/manifest under {}",
            rmbg_root.display()
        );

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
            isolated_root.path().to_string_lossy().into_owned(),
            "--bg-weights-root".to_string(),
            rmbg_root.to_string_lossy().into_owned(),
        ];
        (
            crate::args::build_app_args(crate::args::Args::parse_from(cli)),
            isolated_root,
        )
    }

    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    fn measure_pipeline_load_peak_delta_bytes(args: crate::args::AppArgs) -> Result<u64, String> {
        let handle = thread::Builder::new()
            .name("native-load-ram-check".to_string())
            .stack_size(256 * 1024 * 1024)
            .spawn(move || -> Result<u64, String> {
                let pid = get_current_pid().map_err(|err| format!("failed to get PID: {err}"))?;
                let mut system = System::new();
                let baseline_bytes = refresh_process_rss_bytes(&mut system, pid).unwrap_or(0);
                let monitor = ProcessMemoryMonitor::start(pid, baseline_bytes);
                let _state = build_pipeline_state::<WgpuLoadTestBackend>(&args)?;
                thread::sleep(Duration::from_millis(20));
                let peak_bytes = monitor.stop();
                Ok(peak_bytes.saturating_sub(baseline_bytes))
            })
            .map_err(|err| format!("failed to spawn native load RAM check thread: {err}"))?;
        handle
            .join()
            .map_err(|_| "native load RAM check thread panicked".to_string())?
    }

    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    fn measure_peak_and_final_rss_delta_bytes<F>(action: F) -> Result<(u64, u64), String>
    where
        F: FnOnce() -> Result<(), String>,
    {
        let pid = get_current_pid().map_err(|err| format!("failed to get PID: {err}"))?;
        let mut system = System::new();
        let baseline = refresh_process_rss_bytes(&mut system, pid).unwrap_or(0);
        let monitor = ProcessMemoryMonitor::start(pid, baseline);
        action()?;
        thread::sleep(Duration::from_millis(20));
        let final_bytes = refresh_process_rss_bytes(&mut system, pid).unwrap_or(baseline);
        let peak_bytes = monitor.stop();
        Ok((
            peak_bytes.saturating_sub(baseline),
            final_bytes.saturating_sub(baseline),
        ))
    }

    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    fn format_mib(bytes: u64) -> f64 {
        bytes as f64 / (1024.0 * 1024.0)
    }

    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    #[test]
    #[ignore = "component-level TripoSG + RMBG WGPU load RAM telemetry"]
    fn native_wgpu_component_load_peak_ram_report() {
        let (args, _isolated_root_guard) = build_native_wgpu_load_args();
        let weights_root = resolve_triposg_root(args.weights_root.as_ref());
        let rmbg_root = resolve_rmbg_root(args.bg_weights_root.as_ref(), RmbgModel::Rmbg14);
        let parity = triposg_parity_profile();
        let artifacts = load_triposg_artifacts_from_root(&weights_root, parity.burnpack_policy)
            .expect("resolve TripoSG burnpack artifact paths");
        let device = <WgpuLoadTestBackend as Backend>::Device::default();

        let vae_config = if let Some(json) = artifacts.vae_config_json.as_ref() {
            TripoSGVaeConfig::from_config_bytes(json.as_bytes()).expect("parse TripoSG VAE config")
        } else {
            TripoSGVaeConfig::midi_3d()
        };
        let dit_config = if let Some(json) = artifacts.dit_config_json.as_ref() {
            TripoSGDiTConfig::from_config_bytes(json.as_bytes()).expect("parse TripoSG DiT config")
        } else {
            TripoSGDiTConfig::triposg_pretrained()
        };
        let parsed_dino_config = artifacts
            .dino_config_json
            .as_ref()
            .and_then(|json| load_dinov2_config_from_json_bytes(json.as_bytes()));
        let dino_fallback_size = parsed_dino_config.as_ref().map(|cfg| cfg.image_size);
        let mut dino_config = parsed_dino_config
            .clone()
            .unwrap_or_else(default_dinov2_config);
        let dino_processor = artifacts
            .dino_processor
            .with_strict_preprocess(parity.strict_dino_preprocess);
        if let Some(target_size) = dino_processor_target_size(&dino_processor, dino_fallback_size) {
            let patch = dino_config.patch_size.max(1);
            let grid = target_size / patch;
            if grid > 0 {
                dino_config.positional_encoding_interpolate.output_size = Some([grid, grid]);
            }
        }
        let rmbg_config = load_optional_text_from_root(&rmbg_root, "config.json")
            .expect("read RMBG config")
            .map(|json| {
                load_rmbg_config_from_json_bytes(json.as_bytes()).expect("parse RMBG config json")
            })
            .unwrap_or_else(burn_foreground::rmbg14::RmbgConfig::rmbg_1_4);
        let rmbg_burnpack_path = crate::model_loader::resolve_burnpack_asset_path_from_root(
            &rmbg_root,
            "model.safetensors",
        )
        .expect("resolve RMBG burnpack path");

        let (vae_peak, vae_final) = measure_peak_and_final_rss_delta_bytes(|| {
            let model = load_triposg_vae_from_burnpack_file::<WgpuLoadTestBackend>(
                &vae_config,
                &device,
                &artifacts.vae_burnpack_path,
            )
            .map_err(|err| format!("load VAE: {err}"))?;
            <WgpuLoadTestBackend as Backend>::sync(&device);
            drop(model);
            <WgpuLoadTestBackend as Backend>::sync(&device);
            <WgpuLoadTestBackend as Backend>::memory_cleanup(&device);
            Ok(())
        })
        .expect("measure VAE load RAM");

        let (dit_peak, dit_final) = measure_peak_and_final_rss_delta_bytes(|| {
            let model = load_triposg_dit_from_burnpack_file::<WgpuLoadTestBackend>(
                &dit_config,
                &device,
                &artifacts.dit_burnpack_path,
            )
            .map_err(|err| format!("load DiT: {err}"))?;
            <WgpuLoadTestBackend as Backend>::sync(&device);
            drop(model);
            <WgpuLoadTestBackend as Backend>::sync(&device);
            <WgpuLoadTestBackend as Backend>::memory_cleanup(&device);
            Ok(())
        })
        .expect("measure DiT load RAM");

        let (dino_peak, dino_final) = measure_peak_and_final_rss_delta_bytes(|| {
            let model = load_triposg_dinov2_from_burnpack_file::<WgpuLoadTestBackend>(
                &device,
                dino_config.clone(),
                &artifacts.dino_burnpack_path,
            )
            .map_err(|err| format!("load DINOv2: {err}"))?;
            <WgpuLoadTestBackend as Backend>::sync(&device);
            drop(model);
            <WgpuLoadTestBackend as Backend>::sync(&device);
            <WgpuLoadTestBackend as Backend>::memory_cleanup(&device);
            Ok(())
        })
        .expect("measure DINOv2 load RAM");

        let (rmbg_peak, rmbg_final) = measure_peak_and_final_rss_delta_bytes(|| {
            let model = load_rmbg_from_burnpack_file::<WgpuLoadTestBackend>(
                &device,
                &rmbg_burnpack_path,
                &rmbg_config,
            )
            .map_err(|err| format!("load RMBG: {err}"))?;
            <WgpuLoadTestBackend as Backend>::sync(&device);
            drop(model);
            <WgpuLoadTestBackend as Backend>::sync(&device);
            <WgpuLoadTestBackend as Backend>::memory_cleanup(&device);
            Ok(())
        })
        .expect("measure RMBG load RAM");

        println!(
            "[RAM] VAE peak {:.1} MiB final {:.1} MiB",
            format_mib(vae_peak),
            format_mib(vae_final)
        );
        println!(
            "[RAM] DiT peak {:.1} MiB final {:.1} MiB",
            format_mib(dit_peak),
            format_mib(dit_final)
        );
        println!(
            "[RAM] DINO peak {:.1} MiB final {:.1} MiB",
            format_mib(dino_peak),
            format_mib(dino_final)
        );
        println!(
            "[RAM] RMBG peak {:.1} MiB final {:.1} MiB",
            format_mib(rmbg_peak),
            format_mib(rmbg_final)
        );
    }

    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    #[test]
    #[ignore = "full TripoSG+RMBG WGPU worker init smoke; run explicitly"]
    fn native_wgpu_triposg_worker_init_smoke() {
        let handle = std::thread::Builder::new()
            .name("worker_wgpu_init_smoke".to_string())
            .stack_size(256 * 1024 * 1024)
            .spawn(move || {
                let (args, _isolated_root_guard) = build_native_wgpu_load_args();
                let state = build_pipeline_state::<WgpuLoadTestBackend>(&args)
                    .expect("build WGPU pipeline state without DINO mismatch");
                assert!(
                    state.triposg.is_some() || state.scribble.is_some(),
                    "expected at least one Tripo backend to load"
                );
                assert!(
                    state.triposg_load_error.is_none(),
                    "unexpected Tripo load error: {:?}",
                    state.triposg_load_error
                );
            })
            .expect("spawn worker WGPU init smoke thread");
        handle
            .join()
            .expect("worker WGPU init smoke thread panicked");
    }

    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    #[test]
    #[ignore = "full TripoSG+RMBG WGPU worker inference smoke"]
    fn native_wgpu_triposg_worker_inference_smoke() {
        let handle = std::thread::Builder::new()
            .name("worker_wgpu_inference_smoke".to_string())
            .stack_size(256 * 1024 * 1024)
            .spawn(move || {
                let (mut args, _isolated_root_guard) = build_native_wgpu_load_args();
                args.num_steps = 1;
                args.num_tokens = 64;
                args.resolution = 64;
                args.chunk_size = 4096;
                args.target_faces = Some(2_000);

                let mut state = build_pipeline_state::<WgpuLoadTestBackend>(&args)
                    .expect("build WGPU pipeline state");
                let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .parent()
                    .and_then(|path| path.parent())
                    .expect("resolve workspace root from CARGO_MANIFEST_DIR");
                let request = InferenceRequest {
                    id: 1,
                    image_path: repo_root.join("docs/input_chair.jpg"),
                    image_contents: None,
                    output_path: None,
                };
                let results = run_inference_with_state(&mut state, &args, &[request]);
                assert_eq!(results.len(), 1);
                assert!(
                    results[0].is_ok(),
                    "worker inference should succeed, got {:?}",
                    results[0]
                );
            })
            .expect("spawn worker WGPU inference smoke thread");
        handle
            .join()
            .expect("worker WGPU inference smoke thread panicked");
    }

    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    #[test]
    #[ignore = "full TripoSG+RMBG load/init RAM cap check; run explicitly"]
    fn native_wgpu_triposg_rmbg_load_peak_ram_under_4gib() {
        let (args, _isolated_root_guard) = build_native_wgpu_load_args();
        let peak_delta = measure_pipeline_load_peak_delta_bytes(args).expect(
            "failed to load TripoSG + RMBG weights; fix local import/bundling before asserting RAM budget",
        );
        let max_ram_bytes = 4 * ONE_GIB;

        assert!(
            peak_delta <= max_ram_bytes,
            "native load peak host RAM delta exceeded budget: {} MiB > {} MiB",
            peak_delta as f64 / (1024.0 * 1024.0),
            max_ram_bytes as f64 / (1024.0 * 1024.0)
        );
    }
}

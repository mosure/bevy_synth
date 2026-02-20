#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;
use std::sync::Mutex;
#[cfg(target_arch = "wasm32")]
use std::sync::mpsc::TryRecvError;
use std::sync::mpsc::{self, Receiver, Sender};
#[cfg(not(target_arch = "wasm32"))]
use std::thread;
#[cfg(target_arch = "wasm32")]
use std::time::Duration;

use bevy::prelude::*;
#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
use bevy::render::renderer::{
    RenderAdapter, RenderAdapterInfo, RenderDevice, RenderInstance, RenderQueue, WgpuWrapper,
};

#[cfg(target_arch = "wasm32")]
use crate::SynthMesh;
use crate::args::AppArgs;
#[cfg(target_arch = "wasm32")]
use crate::args::{BackendKind, DinoBackend, QualityPreset, RmbgBackend, WeightPrecision};
#[cfg(target_arch = "wasm32")]
use crate::io::{mesh_from_glb_bytes, mesh_to_glb_bytes};
#[cfg(target_arch = "wasm32")]
use crate::state::InferenceRequest;
#[cfg(target_arch = "wasm32")]
use crate::state::{
    InferenceWorker, WASM_STATUS_LOADING_MODELS, WASM_STATUS_MODEL_LOAD_FAILED_PREFIX,
    WASM_STATUS_MODEL_READY, WorkerCommand, WorkerEvent,
};
#[cfg(not(target_arch = "wasm32"))]
use crate::state::{InferenceWorker, WorkerCommand, WorkerEvent};
#[cfg(target_arch = "wasm32")]
use burn_synth::wasm::WasmInferencePreset;
#[cfg(target_arch = "wasm32")]
use burn_synth::wasm_api::{
    infer_glb_from_image_bytes_with_preset_cached, warmup_pipeline_for_preset_with_status,
};
#[cfg(target_arch = "wasm32")]
use gloo_timers::future::TimeoutFuture;
#[cfg(target_arch = "wasm32")]
use js_sys::{Function, Promise, Reflect};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::{JsFuture, spawn_local};
#[cfg(all(not(target_arch = "wasm32"), feature = "shared-runtime"))]
#[path = "worker_runtime_bridge.rs"]
mod worker_runtime_bridge;
#[cfg(all(not(target_arch = "wasm32"), not(feature = "shared-runtime")))]
compile_error!(
    "bevy_synth_runtime native builds require the `shared-runtime` feature; legacy worker path has been removed"
);

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
pub type SharedWgpuDevice = burn_wgpu::WgpuDevice;
#[cfg(not(all(feature = "wgpu", not(target_arch = "wasm32"))))]
pub type SharedWgpuDevice = ();
#[cfg(not(target_arch = "wasm32"))]
pub type WorkerWakeCallback = Arc<dyn Fn() + Send + Sync + 'static>;

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
fn unwrap_wgpu_wrapper<T: Clone>(wrapper: &WgpuWrapper<T>) -> T {
    <WgpuWrapper<T> as Clone>::clone(wrapper).into_inner()
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
fn native_wgpu_runtime_options() -> burn_wgpu::RuntimeOptions {
    burn_wgpu::RuntimeOptions::default()
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
pub fn init_shared_wgpu_device_from_bevy_render(
    render_adapter: &RenderAdapter,
    render_adapter_info: &RenderAdapterInfo,
    render_device: &RenderDevice,
    render_instance: &RenderInstance,
    render_queue: &RenderQueue,
) -> SharedWgpuDevice {
    let setup = burn_wgpu::WgpuSetup {
        adapter: unwrap_wgpu_wrapper(&render_adapter.0),
        device: render_device.wgpu_device().clone(),
        instance: unwrap_wgpu_wrapper(&render_instance.0),
        queue: unwrap_wgpu_wrapper(&render_queue.0),
        backend: render_adapter_info.backend,
    };
    let options = native_wgpu_runtime_options();
    burn_wgpu::init_device(setup, options)
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
    wasm_webgpu_request_adapter().await.is_some()
}

#[cfg(target_arch = "wasm32")]
pub async fn wasm_webgpu_shader_f16_supported() -> bool {
    let Some(adapter) = wasm_webgpu_request_adapter().await else {
        return false;
    };
    let features = match Reflect::get(&adapter, &wasm_bindgen::JsValue::from_str("features")) {
        Ok(value) if !value.is_undefined() && !value.is_null() => value,
        _ => return false,
    };
    let has_method = match Reflect::get(&features, &wasm_bindgen::JsValue::from_str("has")) {
        Ok(value) => value,
        Err(_) => return false,
    };
    let has_method = match has_method.dyn_into::<Function>() {
        Ok(func) => func,
        Err(_) => return false,
    };
    match has_method.call1(&features, &wasm_bindgen::JsValue::from_str("shader-f16")) {
        Ok(value) => value.as_bool().unwrap_or(false),
        Err(_) => false,
    }
}

#[cfg(target_arch = "wasm32")]
async fn wasm_webgpu_request_adapter() -> Option<wasm_bindgen::JsValue> {
    let window = web_sys::window()?;
    let navigator = window.navigator();
    let navigator_js: wasm_bindgen::JsValue = navigator.into();
    let gpu = match Reflect::get(&navigator_js, &wasm_bindgen::JsValue::from_str("gpu")) {
        Ok(value) if !value.is_undefined() && !value.is_null() => value,
        _ => return None,
    };
    let request_adapter =
        match Reflect::get(&gpu, &wasm_bindgen::JsValue::from_str("requestAdapter")) {
            Ok(value) => value,
            Err(_) => return None,
        };
    let request_adapter = match request_adapter.dyn_into::<Function>() {
        Ok(func) => func,
        Err(_) => return None,
    };
    let promise = match request_adapter.call0(&gpu) {
        Ok(value) => value,
        Err(_) => return None,
    };
    let promise = match promise.dyn_into::<Promise>() {
        Ok(promise) => promise,
        Err(_) => return None,
    };
    match JsFuture::from(promise).await {
        Ok(adapter) if !adapter.is_null() && !adapter.is_undefined() => Some(adapter),
        _ => None,
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn start_worker(args: &AppArgs) -> InferenceWorker {
    start_worker_with_wake(args, None)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn start_worker_with_wake(
    args: &AppArgs,
    wake_callback: Option<WorkerWakeCallback>,
) -> InferenceWorker {
    start_worker_with_shared_wgpu_device_and_wake(args, None, wake_callback)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn start_worker_with_shared_wgpu_device(
    args: &AppArgs,
    shared_wgpu_device: Option<SharedWgpuDevice>,
) -> InferenceWorker {
    start_worker_with_shared_wgpu_device_and_wake(args, shared_wgpu_device, None)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn start_worker_with_shared_wgpu_device_and_wake(
    args: &AppArgs,
    shared_wgpu_device: Option<SharedWgpuDevice>,
    wake_callback: Option<WorkerWakeCallback>,
) -> InferenceWorker {
    let (command_tx, command_rx) = mpsc::channel();
    let (event_tx, event_rx) = mpsc::channel();
    let args = args.clone();
    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    let shared_wgpu_device = shared_wgpu_device.clone();
    let wake_callback = wake_callback.clone();
    let _ = thread::Builder::new()
        .name("synth-worker".to_string())
        .stack_size(64 * 1024 * 1024)
        .spawn(move || {
            worker_loop(
                args,
                command_rx,
                event_tx,
                shared_wgpu_device,
                wake_callback,
            )
        })
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
    std::path::PathBuf::from(format!("uploaded/{request_id:03}_{sanitized}"))
}

#[cfg(not(target_arch = "wasm32"))]
fn worker_loop(
    args: AppArgs,
    command_rx: Receiver<WorkerCommand>,
    event_tx: Sender<WorkerEvent>,
    shared_wgpu_device: Option<SharedWgpuDevice>,
    wake_callback: Option<WorkerWakeCallback>,
) {
    let _ = &shared_wgpu_device;
    info!("Using canonical burn_synth runtime worker path.");
    worker_runtime_bridge::worker_loop_shared_runtime(args, command_rx, event_tx, wake_callback);
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
fn app_args_to_wasm_preset(args: &AppArgs) -> WasmInferencePreset {
    let quality = match args.quality {
        QualityPreset::Fast => "fast",
        QualityPreset::Balanced => "balanced",
        QualityPreset::Full => "full",
    };
    let backend = match args.backend {
        BackendKind::Cpu => "cpu",
        BackendKind::Wgpu => "wgpu",
        BackendKind::Cuda => "cuda",
    };
    let rmbg_backend = match args.rmbg_backend {
        RmbgBackend::Auto => "auto",
        RmbgBackend::Cpu => "cpu",
        RmbgBackend::Gpu => "gpu",
    };
    let dino_backend = match args.dino_backend {
        DinoBackend::Auto => "auto",
        DinoBackend::Cpu => "cpu",
        DinoBackend::Gpu => "gpu",
    };
    let weights_precision = match args.weights_precision {
        WeightPrecision::Auto => "auto",
        WeightPrecision::F16 => "f16",
        WeightPrecision::F32 => "f32",
    };
    let rmbg_weights_precision = match args.rmbg_weights_precision {
        WeightPrecision::Auto => "auto",
        WeightPrecision::F16 => "f16",
        WeightPrecision::F32 => "f32",
    };
    WasmInferencePreset {
        quality,
        num_steps: args.num_steps,
        num_tokens: args.num_tokens,
        resolution: args.flash_min_resolution.max(2),
        faces: args.target_faces.unwrap_or(0),
        flash_octree_depth: args.flash_octree_depth.max(1),
        flash_num_chunks: args.flash_num_chunks.max(1),
        flash_mini_grid_num: args.flash_mini_grid_num.max(1),
        seed: args.seed.unwrap_or(42),
        backend,
        rmbg_backend,
        dino_backend,
        weights_precision,
        rmbg_weights_precision,
    }
}

#[cfg(target_arch = "wasm32")]
async fn ensure_wasm_pipeline_state_via_burn_synth(
    warmup: &mut Option<Result<(), String>>,
    preset: &WasmInferencePreset,
    event_tx: &Sender<WorkerEvent>,
) {
    if warmup.is_none() {
        send_worker_status(event_tx, WASM_STATUS_LOADING_MODELS);
        let mut last_status: Option<String> = None;
        let loaded = warmup_pipeline_for_preset_with_status(preset, |message| {
            if last_status.as_ref().is_some_and(|prev| prev == &message) {
                return;
            }
            last_status = Some(message.clone());
            send_worker_status(event_tx, message);
        })
        .await;
        if let Err(err) = loaded.as_ref() {
            warn!("Inference worker failed to initialize: {err}");
            send_worker_status(
                event_tx,
                format!("{WASM_STATUS_MODEL_LOAD_FAILED_PREFIX} {err}"),
            );
        } else {
            send_worker_status(event_tx, WASM_STATUS_MODEL_READY);
        }
        *warmup = Some(loaded);
    }
}

#[cfg(target_arch = "wasm32")]
async fn infer_request_via_burn_synth(
    request: &InferenceRequest,
    preset: &WasmInferencePreset,
) -> Result<Option<SynthMesh>, String> {
    let image_bytes = request.image_contents.as_ref().ok_or_else(|| {
        format!(
            "wasm inference requires uploaded image bytes for '{}'",
            request.image_path.display()
        )
    })?;
    let glb_bytes = infer_glb_from_image_bytes_with_preset_cached(image_bytes.as_slice(), preset)
        .await
        .map_err(|err| {
            format!(
                "wasm inference failed for '{}': {err}",
                request.image_path.display()
            )
        })?;
    let mesh = mesh_from_glb_bytes(glb_bytes.as_slice())
        .map_err(|err| format!("failed to decode wasm GLB output: {err}"))?;
    Ok(Some(mesh))
}

#[cfg(target_arch = "wasm32")]
async fn worker_loop_wasm(
    args: AppArgs,
    command_rx: Receiver<WorkerCommand>,
    event_tx: Sender<WorkerEvent>,
) {
    // Keep Bevy wasm runtime behavior canonical by delegating all model load + inference
    // semantics to burn_synth's wasm API/cache layer.
    let preset = app_args_to_wasm_preset(&args);
    let mut warmup: Option<Result<(), String>> = None;

    loop {
        match command_rx.try_recv() {
            Ok(WorkerCommand::Warmup) => {
                ensure_wasm_pipeline_state_via_burn_synth(&mut warmup, &preset, &event_tx).await;
            }
            Ok(WorkerCommand::Infer(requests)) => {
                ensure_wasm_pipeline_state_via_burn_synth(&mut warmup, &preset, &event_tx).await;
                let start_ms = wasm_now_ms();
                let results = match warmup.as_ref() {
                    Some(Ok(())) => {
                        let mut out = Vec::with_capacity(requests.len());
                        for request in &requests {
                            out.push(infer_request_via_burn_synth(request, &preset).await);
                        }
                        out
                    }
                    Some(Err(err)) => vec![Err(err.clone()); requests.len()],
                    None => vec![Err("wasm warmup state unavailable".to_string()); requests.len()],
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
                TimeoutFuture::new(8).await;
            }
            Err(TryRecvError::Disconnected) => break,
        }
    }
}

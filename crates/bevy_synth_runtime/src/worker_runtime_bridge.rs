#![cfg(not(target_arch = "wasm32"))]

use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
use std::sync::Once;
use std::sync::mpsc::{Receiver, Sender};
use std::time::Instant;

use burn_synth::pipeline::ModelSelection;
use burn_synth::progress::{
    ProgressCallback, ProgressVerbosity, RuntimeProgressEvent, RuntimeProgressObserver,
    log_progress_event,
};
use burn_synth::runtime::{
    AssetBatchItem, AssetBatchRequest, DinoBackend as RuntimeDinoBackend, InferenceBackend,
    RuntimeBatchPolicy, RuntimeConfig, SynthRuntime, TrellisQuality as RuntimeTrellisQuality,
};
use burn_synth::set_bootstrap_status_callback;
use burn_synth::{
    ForegroundModel, ImageSource, SynthesisAsset as RuntimeSynthesisAsset, SynthesisModel,
};
use burn_triposplat::TripoSplatBurnpackPrecision;

use crate::args::{
    AppArgs, BackendKind, DinoBackend, MeshMode, RmbgModel, TrellisQuality, WeightPrecision,
};
use crate::state::{
    InferenceRequest, WASM_STATUS_LOADING_MODELS, WASM_STATUS_MODEL_LOAD_FAILED_PREFIX,
    WASM_STATUS_MODEL_READY, WorkerCommand, WorkerEvent,
};
use crate::worker::{SharedWgpuDevice, WorkerWakeCallback};
use crate::{
    SynthAsset, SynthMesh, SynthMeshMaterial, SynthMeshPbrTextures, SynthMeshTexture, TripoMesh,
};

const DEFAULT_BOUNDS: [f32; 6] = [-1.005, -1.005, -1.005, 1.005, 1.005, 1.005];
const WORKER_PANIC_PREFIX: &str = "synthesis worker panicked";

fn send_worker_status(event_tx: &Sender<WorkerEvent>, message: impl Into<String>) {
    let _ = event_tx.send(WorkerEvent {
        requests: Vec::new(),
        results: Vec::new(),
        elapsed: Default::default(),
        status_message: Some(message.into()),
    });
}

fn ui_progress_callback(event_tx: Sender<WorkerEvent>) -> ProgressCallback {
    Arc::new(move |event| {
        log_progress_event(event);
        if let Some(message) = format_runtime_progress_for_ui(event) {
            send_worker_status(&event_tx, message);
        }
    })
}

fn format_runtime_progress_for_ui(event: &RuntimeProgressEvent) -> Option<String> {
    match event {
        RuntimeProgressEvent::RunStarted { run, detail } => {
            let mut message = format!("Starting {}", run_label(run));
            if let Some(detail) = detail {
                append_detail(&mut message, ": ", detail);
            }
            Some(message)
        }
        RuntimeProgressEvent::StageStarted { stage, detail, .. } => {
            let mut message = stage_label(stage);
            if let Some(detail) = detail {
                let detail = format_detail(detail);
                if !detail.is_empty() {
                    message.push_str(" (");
                    message.push_str(&detail);
                    message.push(')');
                }
            }
            Some(message)
        }
        RuntimeProgressEvent::Step {
            stage,
            step,
            total_steps,
            eta_ms,
            detail,
            ..
        } => {
            let percent = if *total_steps > 0 {
                (*step as f64 / *total_steps as f64) * 100.0
            } else {
                0.0
            };
            let mut message = format!(
                "{}: step {}/{} ({percent:.1}%)",
                stage_label(stage),
                step,
                total_steps
            );
            if let Some(eta_ms) = eta_ms {
                message.push_str(&format!(", ETA {}", format_duration_ms(*eta_ms)));
            }
            if let Some(detail) = detail {
                append_detail(&mut message, " - ", detail);
            }
            Some(message)
        }
        RuntimeProgressEvent::StageCompleted {
            stage,
            elapsed_ms,
            detail,
            ..
        } => {
            let mut message = format!(
                "{} complete ({})",
                stage_label(stage),
                format_duration_ms(*elapsed_ms)
            );
            if let Some(detail) = detail {
                append_detail(&mut message, ": ", detail);
            }
            Some(message)
        }
        RuntimeProgressEvent::Warning { message, .. } => Some(format!("Warning: {message}")),
        RuntimeProgressEvent::RunCompleted {
            run,
            elapsed_ms,
            detail,
        } => {
            let mut message = format!(
                "{} complete ({})",
                run_label(run),
                format_duration_ms(*elapsed_ms)
            );
            if let Some(detail) = detail {
                append_detail(&mut message, ": ", detail);
            }
            Some(message)
        }
    }
}

fn run_label(run: &str) -> &'static str {
    match run {
        "asset" => "synthesis",
        "mesh" => "mesh synthesis",
        "splat" => "Gaussian synthesis",
        "foreground" => "foreground extraction",
        _ => "synthesis",
    }
}

fn stage_label(stage: &str) -> String {
    match stage {
        "foreground.materialize_input" => "Preparing input image".to_string(),
        "foreground.load_image" => "Loading input image".to_string(),
        "foreground.run" => "Removing background".to_string(),
        "triposplat.cuda_preflight" => "Checking CUDA backend".to_string(),
        "triposplat.preprocess_foreground" => "Preparing foreground".to_string(),
        "triposplat.load_backend" => "Loading TripoSplat weights".to_string(),
        "triposplat.prepare_tensor" => "Preparing TripoSplat tensor".to_string(),
        "triposplat.encode" => "Encoding image".to_string(),
        "triposplat.sample" => "Sampling TripoSplat flow".to_string(),
        "triposplat.decode" => "Decoding Gaussian splats".to_string(),
        "triposplat.dry_run" => "Preparing TripoSplat dry run".to_string(),
        "triposg.preprocess_foreground" => "Preparing foreground".to_string(),
        "triposg.load_backend" => "Loading TripoSG weights".to_string(),
        "triposg.prepare_tensor" => "Preparing TripoSG tensor".to_string(),
        "triposg.encode" => "Encoding image".to_string(),
        "triposg.sample" => "Sampling TripoSG".to_string(),
        "triposg.decode" => "Decoding mesh".to_string(),
        "triposg.mesh_extract" => "Extracting mesh".to_string(),
        "trellis.preprocess_foreground" => "Preparing foreground".to_string(),
        "trellis.load_backend" => "Loading Trellis2 weights".to_string(),
        "trellis.prepare_tensor" => "Preparing Trellis2 tensor".to_string(),
        "trellis.encode" => "Encoding image".to_string(),
        "trellis.sample" => "Sampling Trellis2".to_string(),
        "trellis.decode" => "Decoding mesh".to_string(),
        _ => stage.replace('.', " "),
    }
}

fn format_detail(detail: &str) -> String {
    detail
        .split_whitespace()
        .filter(|part| {
            !part.starts_with("weights_root=")
                && !part.starts_with("image=")
                && !part.starts_with("path=")
        })
        .map(|part| part.replace('_', " "))
        .collect::<Vec<_>>()
        .join(" ")
}

fn append_detail(message: &mut String, separator: &str, detail: &str) {
    let detail = format_detail(detail);
    if !detail.is_empty() {
        message.push_str(separator);
        message.push_str(&detail);
    }
}

fn format_duration_ms(ms: f64) -> String {
    if ms < 1000.0 {
        format!("{ms:.0} ms")
    } else if ms < 60_000.0 {
        format!("{:.1}s", ms / 1000.0)
    } else {
        let total_secs = (ms / 1000.0).round() as u64;
        format!("{}m {:02}s", total_secs / 60, total_secs % 60)
    }
}

pub(crate) fn validate_canonical_runtime_args(args: &AppArgs) -> Result<(), String> {
    if args.prompt.is_some() || args.text_embeds.is_some() || args.scribble_weights_root.is_some() {
        return Err(
            "prompt/scribble inference is not supported in canonical burn_synth runtime mode"
                .to_string(),
        );
    }
    if !matches!(args.mesh_mode, MeshMode::Flash) {
        return Err(format!(
            "mesh mode {:?} is unsupported in canonical burn_synth runtime mode; use flash",
            args.mesh_mode
        ));
    }
    Ok(())
}

pub(crate) fn worker_loop_shared_runtime(
    args: AppArgs,
    command_rx: Receiver<WorkerCommand>,
    event_tx: Sender<WorkerEvent>,
    shared_wgpu_device: Option<SharedWgpuDevice>,
    wake_callback: Option<WorkerWakeCallback>,
) {
    send_worker_status(&event_tx, WASM_STATUS_LOADING_MODELS);
    let bootstrap_event_tx = event_tx.clone();
    set_bootstrap_status_callback(Some(Arc::new(move |message| {
        send_worker_status(&bootstrap_event_tx, message);
    })));
    let runtime_result = catch_worker_unwind("while initializing canonical runtime", || {
        build_runtime(&args, shared_wgpu_device, event_tx.clone())
    });

    let mut runtime = match runtime_result {
        Ok(runtime) => runtime,
        Err(err) => {
            set_bootstrap_status_callback(None);
            send_worker_status(
                &event_tx,
                format!("{WASM_STATUS_MODEL_LOAD_FAILED_PREFIX} {err}"),
            );
            for command in command_rx {
                match command {
                    WorkerCommand::Warmup => {}
                    WorkerCommand::Infer(requests) => {
                        let sent = event_tx.send(WorkerEvent {
                            results: vec![Err(err.clone()); requests.len()],
                            requests,
                            elapsed: Default::default(),
                            status_message: None,
                        });
                        if sent.is_ok()
                            && let Some(wake) = wake_callback.as_ref()
                        {
                            wake();
                        }
                    }
                    WorkerCommand::Shutdown => break,
                }
            }
            return;
        }
    };
    send_worker_status(&event_tx, WASM_STATUS_MODEL_READY);

    let mut terminal_worker_error: Option<String> = None;
    for command in command_rx {
        match command {
            WorkerCommand::Warmup => {}
            WorkerCommand::Infer(requests) => {
                let started = Instant::now();
                let results = if let Some(err) = terminal_worker_error.as_ref() {
                    vec![
                        Err(format!(
                            "synthesis worker stopped after prior panic; restart the app before retrying: {err}"
                        ));
                        requests.len()
                    ]
                } else {
                    let context = format!(
                        "while processing native batch of {} request(s)",
                        requests.len()
                    );
                    match catch_worker_unwind(&context, || {
                        infer_request_batch(&mut runtime, &args, &requests)
                    }) {
                        Ok(results) => results,
                        Err(err) => {
                            if err.starts_with(WORKER_PANIC_PREFIX) {
                                terminal_worker_error = Some(err.clone());
                            }
                            vec![Err(err); requests.len()]
                        }
                    }
                };
                let sent = event_tx.send(WorkerEvent {
                    requests,
                    results,
                    elapsed: started.elapsed(),
                    status_message: None,
                });
                if sent.is_ok()
                    && let Some(wake) = wake_callback.as_ref()
                {
                    wake();
                }
            }
            WorkerCommand::Shutdown => break,
        }
    }
    set_bootstrap_status_callback(None);
}

fn catch_worker_unwind<T, F>(context: &str, operation: F) -> Result<T, String>
where
    F: FnOnce() -> Result<T, String>,
{
    match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(result) => result,
        Err(payload) => Err(format!(
            "{WORKER_PANIC_PREFIX} {context}: {}",
            panic_payload_message(payload.as_ref())
        )),
    }
}

fn panic_payload_message(payload: &(dyn Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<String>() {
        return message.clone();
    }
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        return (*message).to_string();
    }
    "panic payload was not a string".to_string()
}

fn build_runtime(
    args: &AppArgs,
    shared_wgpu_device: Option<SharedWgpuDevice>,
    event_tx: Sender<WorkerEvent>,
) -> Result<SynthRuntime, String> {
    validate_canonical_runtime_args(args)?;
    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    if matches!(args.backend, BackendKind::Wgpu) && shared_wgpu_device.is_none() {
        configure_wgpu_runtime_memory_profile();
    }
    let config = runtime_config_from_args(args, shared_wgpu_device, event_tx)?;
    Ok(SynthRuntime::new(config))
}

fn runtime_config_from_args(
    args: &AppArgs,
    shared_wgpu_device: Option<SharedWgpuDevice>,
    event_tx: Sender<WorkerEvent>,
) -> Result<RuntimeConfig, String> {
    let model_selection = ModelSelection::new(
        args.synthesis_models
            .iter()
            .copied()
            .map(map_synthesis_model),
        map_foreground_model(args.rmbg_model),
    );
    let primary_model = args.synthesis_models.first().copied();
    let target_faces = if matches!(primary_model, Some(crate::args::SynthesisModel::Trellis)) {
        args.trellis_target_faces
    } else {
        args.target_faces
    };
    let mut config = RuntimeConfig {
        model_selection,
        backend: map_backend(&args.backend),
        weights_root: args.weights_root.clone(),
        trellis_weights_root: args.trellis_weights_root.clone(),
        triposplat_weights_root: args.triposplat_weights_root.clone(),
        trellis_image_large_root: args.trellis_image_large_root.clone(),
        triposplat_weights_precision: if args
            .synthesis_models
            .iter()
            .any(|model| matches!(model, crate::args::SynthesisModel::Triposplat))
        {
            map_triposplat_weights_precision(args.weights_precision)
        } else {
            RuntimeConfig::default().triposplat_weights_precision
        },
        trellis_quality: map_trellis_quality(args.trellis_quality),
        trellis_max_sparse_coords: args.trellis_max_sparse_coords,
        trellis_pbr_enabled: args.trellis_pbr_enabled,
        trellis_pbr_texture_size: args.trellis_pbr_texture_size,
        bg_weights_root: args.bg_weights_root.clone(),
        num_steps: args.num_steps,
        num_tokens: args.num_tokens,
        guidance_scale: args.guidance_scale,
        triposplat_num_steps: args.num_steps,
        triposplat_guidance_scale: args.guidance_scale,
        triposplat_shift: args.triposplat_shift,
        triposplat_num_gaussians: args.triposplat_num_gaussians,
        triposplat_erode_radius: args.triposplat_erode_radius,
        seed: args.seed.or(RuntimeConfig::default().seed),
        dino_backend: map_dino_backend(args.dino_backend),
        target_faces,
        ..RuntimeConfig::default()
    };
    config.progress = RuntimeProgressObserver::with_callback(
        ProgressVerbosity::Steps,
        1,
        ui_progress_callback(event_tx),
    );

    let bounds = parse_bounds(args.bounds.as_slice())?;
    config.flash_extract.bounds = bounds;
    config.flash_extract.octree_depth = args.flash_octree_depth;
    config.flash_extract.min_resolution = args.flash_min_resolution;
    config.flash_extract.mini_grid_num = args.flash_mini_grid_num;
    config.flash_extract.num_chunks = args.flash_num_chunks;
    config.flash_extract.mc_level = args.flash_mc_level;
    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    {
        config.wgpu_device = shared_wgpu_device;
    }
    #[cfg(not(all(feature = "wgpu", not(target_arch = "wasm32"))))]
    let _ = shared_wgpu_device;

    Ok(config)
}

fn infer_request_batch(
    runtime: &mut SynthRuntime,
    args: &AppArgs,
    requests: &[InferenceRequest],
) -> Result<Vec<Result<Option<SynthAsset>, String>>, String> {
    let mut results = Vec::with_capacity(requests.len());
    let mut start = 0;
    while start < requests.len() {
        let mut end = start + 1;
        while end < requests.len()
            && requests_compatible_for_batch(args, &requests[start], &requests[end])
        {
            end += 1;
        }
        results.extend(infer_homogeneous_request_batch(
            runtime,
            args,
            &requests[start..end],
        )?);
        start = end;
    }
    Ok(results)
}

fn requests_compatible_for_batch(
    args: &AppArgs,
    lhs: &InferenceRequest,
    rhs: &InferenceRequest,
) -> bool {
    lhs.settings == rhs.settings
        && effective_synthesis_models(args, lhs) == effective_synthesis_models(args, rhs)
}

fn infer_homogeneous_request_batch(
    runtime: &mut SynthRuntime,
    args: &AppArgs,
    requests: &[InferenceRequest],
) -> Result<Vec<Result<Option<SynthAsset>, String>>, String> {
    let Some(first) = requests.first() else {
        return Ok(Vec::new());
    };
    apply_request_settings(runtime, args, first);
    let synthesis_models = effective_synthesis_models(args, first);
    let batch = runtime
        .synthesize_assets_batch(AssetBatchRequest {
            items: requests
                .iter()
                .map(|request| {
                    let image = request
                        .image_contents
                        .as_ref()
                        .map(|bytes| ImageSource::from_bytes(bytes.clone()))
                        .unwrap_or_else(|| ImageSource::from_path(request.image_path.clone()));
                    AssetBatchItem::new(request.id.to_string(), image)
                })
                .collect(),
            foreground_model: Some(map_foreground_model(args.rmbg_model)),
            synthesis_models: Some(
                synthesis_models
                    .iter()
                    .copied()
                    .map(map_synthesis_model)
                    .collect(),
            ),
            backend: Some(map_backend(&args.backend)),
            dry_run: false,
            policy: RuntimeBatchPolicy {
                max_items: Some(requests.len().max(1)),
                ..RuntimeBatchPolicy::default()
            },
        })
        .map_err(|err| format!("synthesis batch inference failed: {err}"))?;
    Ok(batch
        .items
        .into_iter()
        .map(|item| {
            item.output
                .map(|output| Some(runtime_asset_to_synth_asset(output.asset)))
                .map_err(|err| format!("synthesis inference failed: {err}"))
        })
        .collect())
}

fn apply_request_settings(runtime: &mut SynthRuntime, args: &AppArgs, request: &InferenceRequest) {
    {
        let request_model = request
            .synthesis_models
            .first()
            .copied()
            .or_else(|| args.synthesis_models.first().copied());
        let trellis_request = matches!(request_model, Some(crate::args::SynthesisModel::Trellis));
        let config = runtime.config_mut();
        config.num_steps = request.settings.num_steps;
        config.num_tokens = request.settings.num_tokens;
        config.guidance_scale = request.settings.guidance_scale;
        config.target_faces = if trellis_request {
            request.settings.trellis_target_faces
        } else {
            request.settings.target_faces
        };
        config.triposplat_num_steps = request.settings.num_steps;
        config.triposplat_guidance_scale = request.settings.guidance_scale;
        config.triposplat_num_gaussians = request.settings.triposplat_num_gaussians;
        config.trellis_quality = map_trellis_quality(request.settings.trellis_quality);
        config.trellis_max_sparse_coords = request.settings.trellis_max_sparse_coords;
        config.trellis_pbr_enabled = request.settings.trellis_pbr_enabled;
        config.trellis_pbr_texture_size = request.settings.trellis_pbr_texture_size;
    }
}

fn effective_synthesis_models(
    args: &AppArgs,
    request: &InferenceRequest,
) -> Vec<crate::args::SynthesisModel> {
    if request.synthesis_models.is_empty() {
        args.synthesis_models.clone()
    } else {
        request.synthesis_models.clone()
    }
}

fn runtime_asset_to_synth_asset(asset: RuntimeSynthesisAsset) -> SynthAsset {
    match asset {
        RuntimeSynthesisAsset::Mesh(mesh) => SynthAsset::Mesh(runtime_mesh_to_synth_mesh(mesh)),
        RuntimeSynthesisAsset::GaussianSplat(splats) => SynthAsset::GaussianSplat(splats),
    }
}

fn runtime_mesh_to_synth_mesh(mesh: burn_synth::mesh::Mesh) -> SynthMesh {
    SynthMesh {
        mesh: TripoMesh {
            vertices: mesh.vertices,
            faces: mesh.faces,
        },
        uvs: mesh.uvs,
        normals: mesh.normals,
        material: mesh.material.map(|material| SynthMeshMaterial {
            base_color: material.base_color,
            metallic: material.metallic,
            roughness: material.roughness,
            alpha: material.alpha,
        }),
        pbr_textures: mesh.pbr_textures.map(|textures| SynthMeshPbrTextures {
            base_color: map_texture(textures.base_color),
            metallic_roughness: map_texture(textures.metallic_roughness),
            normal: textures.normal.map(map_texture),
            emissive: textures.emissive.map(map_texture),
            occlusion: textures.occlusion.map(map_texture),
        }),
    }
}

fn map_texture(texture: burn_synth::mesh::MeshTexture) -> SynthMeshTexture {
    SynthMeshTexture {
        width: texture.width,
        height: texture.height,
        rgba8: texture.rgba8,
    }
}

fn map_foreground_model(value: RmbgModel) -> ForegroundModel {
    match value {
        RmbgModel::Rmbg14 => ForegroundModel::Rmbg14,
        RmbgModel::Rmbg2 => ForegroundModel::Rmbg2,
    }
}

fn map_synthesis_model(value: crate::args::SynthesisModel) -> SynthesisModel {
    match value {
        crate::args::SynthesisModel::Triposg => SynthesisModel::Triposg,
        crate::args::SynthesisModel::Trellis => SynthesisModel::Trellis,
        crate::args::SynthesisModel::Triposplat => SynthesisModel::Triposplat,
    }
}

fn map_backend(value: &BackendKind) -> InferenceBackend {
    match value {
        BackendKind::Cpu => InferenceBackend::Cpu,
        BackendKind::Wgpu => InferenceBackend::Wgpu,
        BackendKind::Cuda => InferenceBackend::Cuda,
    }
}

fn map_dino_backend(value: DinoBackend) -> RuntimeDinoBackend {
    match value {
        DinoBackend::Auto => RuntimeDinoBackend::Auto,
        DinoBackend::Cpu => RuntimeDinoBackend::Cpu,
        DinoBackend::Gpu => RuntimeDinoBackend::Gpu,
    }
}

fn map_trellis_quality(value: TrellisQuality) -> RuntimeTrellisQuality {
    match value {
        TrellisQuality::Low => RuntimeTrellisQuality::Low,
        TrellisQuality::Medium => RuntimeTrellisQuality::Medium,
        TrellisQuality::High => RuntimeTrellisQuality::High,
    }
}

fn map_triposplat_weights_precision(value: WeightPrecision) -> Option<TripoSplatBurnpackPrecision> {
    match value {
        WeightPrecision::Auto => None,
        WeightPrecision::F16 => Some(TripoSplatBurnpackPrecision::F16),
        WeightPrecision::F32 => Some(TripoSplatBurnpackPrecision::F32),
    }
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
fn configure_wgpu_runtime_memory_profile() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let device = burn_wgpu::WgpuDevice::default();
        let options = burn_wgpu::RuntimeOptions::default();
        let _setup =
            burn_wgpu::init_setup::<burn_wgpu::graphics::AutoGraphicsApi>(&device, options);
    });
}

fn parse_bounds(raw: &[f32]) -> Result<[f32; 6], String> {
    if raw.len() != 6 {
        return Err(format!(
            "expected 6 bounds values (minX minY minZ maxX maxY maxZ), got {}",
            raw.len()
        ));
    }
    let mut out = DEFAULT_BOUNDS;
    out.copy_from_slice(&raw[..6]);
    if out
        .iter()
        .any(|value| !value.is_finite() || value.is_nan() || value.is_infinite())
    {
        return Err("bounds must contain finite numbers".to_string());
    }
    if out[0] >= out[3] || out[1] >= out[4] || out[2] >= out[5] {
        return Err("bounds min values must be strictly less than max values".to_string());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;

    use burn_synth::progress::RuntimeProgressEvent;
    use clap::Parser;

    use super::{
        RuntimeTrellisQuality, WORKER_PANIC_PREFIX, catch_worker_unwind,
        format_runtime_progress_for_ui, parse_bounds, runtime_config_from_args,
        validate_canonical_runtime_args,
    };
    use crate::args::{Args, BackendKind, MeshMode, SynthesisModel, build_app_args};
    use burn_triposplat::TripoSplatBurnpackPrecision;

    #[test]
    fn validates_canonical_runtime_for_trellis_flash_pipeline() {
        let args = build_app_args(Args::parse_from([
            "bevy_synth",
            "--synthesis-models",
            "trellis",
        ]));
        validate_canonical_runtime_args(&args).expect("trellis flash pipeline should be valid");
    }

    #[test]
    fn canonical_runtime_rejects_prompt_and_non_flash_mesh_mode() {
        let args = build_app_args(Args::parse_from(["bevy_synth"]));
        assert!(
            args.synthesis_models
                .iter()
                .any(|model| matches!(model, SynthesisModel::Triposg))
        );
        validate_canonical_runtime_args(&args).expect("default triposg flash should be valid");

        let mut args = build_app_args(Args::parse_from(["bevy_synth"]));
        args.prompt = Some("chair".to_string());
        let err = validate_canonical_runtime_args(&args).expect_err("prompt should be rejected");
        assert!(err.contains("prompt/scribble inference"));

        let mut args = build_app_args(Args::parse_from(["bevy_synth"]));
        args.mesh_mode = MeshMode::Dense;
        let err =
            validate_canonical_runtime_args(&args).expect_err("non-flash mesh mode should fail");
        assert!(err.contains("unsupported"));
    }

    #[test]
    fn canonical_runtime_accepts_triposplat_asset_requests() {
        let args = build_app_args(Args::parse_from([
            "bevy_synth",
            "--synthesis-models",
            "triposplat",
        ]));
        validate_canonical_runtime_args(&args)
            .expect("triposplat splat output should pass through the worker asset surface");
    }

    #[test]
    fn runtime_config_maps_triposplat_precision_from_app_args() {
        let (event_tx, _event_rx) = mpsc::channel();
        let f16_args = build_app_args(Args::parse_from([
            "bevy_synth",
            "--synthesis-models",
            "triposplat",
            "--weights-precision",
            "f16",
        ]));
        let f16_config = runtime_config_from_args(&f16_args, None, event_tx.clone())
            .expect("runtime config should preserve triposplat precision");
        assert_eq!(
            f16_config.triposplat_weights_precision,
            Some(TripoSplatBurnpackPrecision::F16)
        );

        let f32_args = build_app_args(Args::parse_from([
            "bevy_synth",
            "--synthesis-models",
            "triposplat",
            "--weights-precision",
            "f32",
        ]));
        let f32_config = runtime_config_from_args(&f32_args, None, event_tx)
            .expect("runtime config should preserve triposplat precision");
        assert_eq!(
            f32_config.triposplat_weights_precision,
            Some(TripoSplatBurnpackPrecision::F32)
        );
    }

    #[test]
    fn runtime_config_maps_trellis_settings_from_app_args() {
        let (event_tx, _event_rx) = mpsc::channel();
        let args = build_app_args(Args::parse_from([
            "bevy_synth",
            "--synthesis-models",
            "trellis",
            "--trellis-quality",
            "high",
            "--trellis-pbr",
            "false",
            "--trellis-pbr-texture-size",
            "2048",
            "--trellis-faces",
            "500000",
        ]));
        let config = runtime_config_from_args(&args, None, event_tx)
            .expect("runtime config should preserve trellis settings");
        assert_eq!(config.trellis_quality, RuntimeTrellisQuality::High);
        assert!(!config.trellis_pbr_enabled);
        assert_eq!(config.trellis_pbr_texture_size, Some(2048));
        assert_eq!(config.target_faces, Some(500_000));
        assert_eq!(config.trellis_max_sparse_coords, None);
    }

    #[test]
    fn runtime_config_does_not_cap_trellis_sparse_coords_by_default() {
        let (event_tx, _event_rx) = mpsc::channel();
        let args = build_app_args(Args::parse_from([
            "bevy_synth",
            "--synthesis-models",
            "trellis",
        ]));
        let config = runtime_config_from_args(&args, None, event_tx)
            .expect("runtime config should preserve trellis defaults");

        assert_eq!(
            config.trellis_max_sparse_coords, None,
            "UI/runtime defaults must not destructively cap TRELLIS sparse coordinates"
        );
    }

    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    #[test]
    fn runtime_config_preserves_shared_wgpu_device() {
        let (event_tx, _event_rx) = mpsc::channel();
        let mut args = build_app_args(Args::parse_from(["bevy_synth", "--backend", "wgpu"]));
        args.backend = BackendKind::Wgpu;
        let shared_device = burn_wgpu::WgpuDevice::Existing(7);
        let config = runtime_config_from_args(&args, Some(shared_device.clone()), event_tx)
            .expect("runtime config should build without loading weights");

        assert_eq!(config.wgpu_device, Some(shared_device));
    }

    #[test]
    fn ui_progress_formatter_distinguishes_stage_and_sampling_step() {
        let stage = RuntimeProgressEvent::StageStarted {
            run: "asset",
            stage: "triposplat.encode",
            total_steps: None,
            detail: Some("seed=42".to_string()),
        };
        assert_eq!(
            format_runtime_progress_for_ui(&stage).as_deref(),
            Some("Encoding image (seed=42)")
        );

        let step = RuntimeProgressEvent::Step {
            run: "asset",
            stage: "triposplat.sample",
            step: 3,
            total_steps: 20,
            step_ms: 250.0,
            elapsed_ms: 750.0,
            eta_ms: Some(4250.0),
            detail: Some("guidance_scale=3.000".to_string()),
        };
        assert_eq!(
            format_runtime_progress_for_ui(&step).as_deref(),
            Some("Sampling TripoSplat flow: step 3/20 (15.0%), ETA 4.2s - guidance scale=3.000")
        );

        let load_done = RuntimeProgressEvent::StageCompleted {
            run: "asset",
            stage: "triposplat.load_backend",
            total_steps: None,
            elapsed_ms: 261_013.9,
            detail: Some(
                "weights_root=/home/mosure/.burn_synth/models/TripoSplat precision=f16 compute=f16"
                    .to_string(),
            ),
        };
        assert_eq!(
            format_runtime_progress_for_ui(&load_done).as_deref(),
            Some("Loading TripoSplat weights complete (4m 21s): precision=f16 compute=f16")
        );
    }

    #[test]
    fn parse_bounds_rejects_invalid_ranges() {
        let err = parse_bounds(&[1.0, 0.0, 0.0, 0.5, 1.0, 1.0]).expect_err("invalid bounds");
        assert!(err.contains("strictly less"));
    }

    #[test]
    fn worker_unwind_guard_reports_panic_as_error() {
        let err = catch_worker_unwind::<(), _>("while testing panic conversion", || {
            panic!("backend compiler failed")
        })
        .expect_err("panic should be converted into a worker error");

        assert!(
            err.starts_with(WORKER_PANIC_PREFIX),
            "unexpected error: {err}"
        );
        assert!(
            err.contains("backend compiler failed"),
            "unexpected error: {err}"
        );
    }
}

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
    ProgressVerbosity, RuntimeProgressObserver, default_log_progress_callback,
};
use burn_synth::runtime::{
    AssetRequest, DinoBackend as RuntimeDinoBackend, InferenceBackend, RuntimeConfig, SynthRuntime,
    TrellisQuality as RuntimeTrellisQuality,
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
        build_runtime(&args, shared_wgpu_device)
    });
    set_bootstrap_status_callback(None);

    let mut runtime = match runtime_result {
        Ok(runtime) => runtime,
        Err(err) => {
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
                let mut results = Vec::with_capacity(requests.len());
                for request in requests.iter() {
                    if let Some(err) = terminal_worker_error.as_ref() {
                        results.push(Err(format!(
                            "synthesis worker stopped after prior panic; restart the app before retrying: {err}"
                        )));
                        continue;
                    }
                    let context = format!("while processing {}", request.image_path.display());
                    let result = catch_worker_unwind(&context, || {
                        infer_one_request(&mut runtime, &args, request)
                    });
                    if let Err(err) = &result
                        && err.starts_with(WORKER_PANIC_PREFIX)
                    {
                        terminal_worker_error = Some(err.clone());
                    }
                    let result = result.map(Some);
                    results.push(result);
                }
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
) -> Result<SynthRuntime, String> {
    validate_canonical_runtime_args(args)?;
    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    if matches!(args.backend, BackendKind::Wgpu) && shared_wgpu_device.is_none() {
        configure_wgpu_runtime_memory_profile();
    }
    let config = runtime_config_from_args(args, shared_wgpu_device)?;
    Ok(SynthRuntime::new(config))
}

fn runtime_config_from_args(
    args: &AppArgs,
    shared_wgpu_device: Option<SharedWgpuDevice>,
) -> Result<RuntimeConfig, String> {
    let model_selection = ModelSelection::new(
        args.synthesis_models
            .iter()
            .copied()
            .map(map_synthesis_model),
        map_foreground_model(args.rmbg_model),
    );
    let mut config = RuntimeConfig {
        model_selection,
        backend: map_backend(&args.backend),
        weights_root: args.weights_root.clone(),
        trellis_weights_root: args.trellis_weights_root.clone(),
        triposplat_weights_root: args.triposplat_weights_root.clone(),
        trellis_image_large_root: args.trellis_image_large_root.clone(),
        trellis_python_bin: args.trellis_python_bin.clone(),
        trellis_bridge_script: args.trellis_bridge_script.clone(),
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
        target_faces: args.target_faces,
        ..RuntimeConfig::default()
    };
    config.progress = RuntimeProgressObserver::with_callback(
        ProgressVerbosity::Stages,
        1,
        default_log_progress_callback(),
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

fn infer_one_request(
    runtime: &mut SynthRuntime,
    args: &AppArgs,
    request: &InferenceRequest,
) -> Result<SynthAsset, String> {
    {
        let config = runtime.config_mut();
        config.num_steps = request.settings.num_steps;
        config.guidance_scale = request.settings.guidance_scale;
        config.triposplat_num_steps = request.settings.num_steps;
        config.triposplat_guidance_scale = request.settings.guidance_scale;
        config.triposplat_num_gaussians = request.settings.triposplat_num_gaussians;
    }

    let image = request
        .image_contents
        .as_ref()
        .map(|bytes| ImageSource::from_bytes(bytes.clone()))
        .unwrap_or_else(|| ImageSource::from_path(request.image_path.clone()));

    let synthesis_models = if request.synthesis_models.is_empty() {
        args.synthesis_models.clone()
    } else {
        request.synthesis_models.clone()
    };

    let request = AssetRequest {
        image,
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
    };

    let output = runtime
        .synthesize_asset(request)
        .map_err(|err| format!("synthesis inference failed: {err}"))?;
    Ok(runtime_asset_to_synth_asset(output.asset))
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
    use clap::Parser;

    use super::{
        WORKER_PANIC_PREFIX, catch_worker_unwind, parse_bounds, runtime_config_from_args,
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
        let f16_args = build_app_args(Args::parse_from([
            "bevy_synth",
            "--synthesis-models",
            "triposplat",
            "--weights-precision",
            "f16",
        ]));
        let f16_config = runtime_config_from_args(&f16_args, None)
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
        let f32_config = runtime_config_from_args(&f32_args, None)
            .expect("runtime config should preserve triposplat precision");
        assert_eq!(
            f32_config.triposplat_weights_precision,
            Some(TripoSplatBurnpackPrecision::F32)
        );
    }

    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    #[test]
    fn runtime_config_preserves_shared_wgpu_device() {
        let mut args = build_app_args(Args::parse_from(["bevy_synth", "--backend", "wgpu"]));
        args.backend = BackendKind::Wgpu;
        let shared_device = burn_wgpu::WgpuDevice::Existing(7);
        let config = runtime_config_from_args(&args, Some(shared_device.clone()))
            .expect("runtime config should build without loading weights");

        assert_eq!(config.wgpu_device, Some(shared_device));
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

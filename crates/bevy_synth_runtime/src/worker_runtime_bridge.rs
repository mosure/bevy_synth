#![cfg(not(target_arch = "wasm32"))]

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
use std::sync::Once;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;
use std::time::Instant;

use burn_synth::pipeline::ModelSelection;
use burn_synth::progress::{
    ProgressVerbosity, RuntimeProgressObserver, default_log_progress_callback,
};
use burn_synth::runtime::{
    DinoBackend as RuntimeDinoBackend, InferenceBackend, MeshRequest, RuntimeConfig, SynthRuntime,
    TrellisQuality as RuntimeTrellisQuality,
};
use burn_synth::set_bootstrap_status_callback;
use burn_synth::{ForegroundModel, ImageSource, SynthesisModel};

use crate::args::{AppArgs, BackendKind, DinoBackend, MeshMode, RmbgModel, TrellisQuality};
use crate::state::{
    InferenceRequest, WASM_STATUS_LOADING_MODELS, WASM_STATUS_MODEL_LOAD_FAILED_PREFIX,
    WASM_STATUS_MODEL_READY, WorkerCommand, WorkerEvent,
};
use crate::worker::WorkerWakeCallback;
use crate::{SynthMesh, SynthMeshMaterial, SynthMeshPbrTextures, SynthMeshTexture, TripoMesh};

const DEFAULT_BOUNDS: [f32; 6] = [-1.005, -1.005, -1.005, 1.005, 1.005, 1.005];

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
    wake_callback: Option<WorkerWakeCallback>,
) {
    send_worker_status(&event_tx, WASM_STATUS_LOADING_MODELS);
    let bootstrap_event_tx = event_tx.clone();
    set_bootstrap_status_callback(Some(Arc::new(move |message| {
        send_worker_status(&bootstrap_event_tx, message);
    })));
    let runtime_result = build_runtime(&args);
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

    for command in command_rx {
        match command {
            WorkerCommand::Warmup => {}
            WorkerCommand::Infer(requests) => {
                let started = Instant::now();
                let mut results = Vec::with_capacity(requests.len());
                for request in requests.iter() {
                    let result = infer_one_request(&mut runtime, &args, request).map(Some);
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

fn build_runtime(args: &AppArgs) -> Result<SynthRuntime, String> {
    validate_canonical_runtime_args(args)?;
    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    if matches!(args.backend, BackendKind::Wgpu) {
        configure_wgpu_runtime_memory_profile();
    }
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
        trellis_image_large_root: args.trellis_image_large_root.clone(),
        trellis_python_bin: args.trellis_python_bin.clone(),
        trellis_bridge_script: args.trellis_bridge_script.clone(),
        trellis_quality: map_trellis_quality(args.trellis_quality),
        bg_weights_root: args.bg_weights_root.clone(),
        num_steps: args.num_steps,
        num_tokens: args.num_tokens,
        guidance_scale: args.guidance_scale,
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

    Ok(SynthRuntime::new(config))
}

fn infer_one_request(
    runtime: &mut SynthRuntime,
    args: &AppArgs,
    request: &InferenceRequest,
) -> Result<SynthMesh, String> {
    let image = request
        .image_contents
        .as_ref()
        .map(|bytes| ImageSource::from_bytes(bytes.clone()))
        .unwrap_or_else(|| ImageSource::from_path(request.image_path.clone()));

    let request = MeshRequest {
        image,
        foreground_model: Some(map_foreground_model(args.rmbg_model)),
        synthesis_models: Some(
            args.synthesis_models
                .iter()
                .copied()
                .map(map_synthesis_model)
                .collect(),
        ),
        backend: Some(map_backend(&args.backend)),
        dry_run: false,
    };

    let output = runtime
        .synthesize_mesh(request)
        .map_err(|err| format!("synthesis inference failed: {err}"))?;
    Ok(runtime_mesh_to_synth_mesh(output.mesh))
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

    use super::{parse_bounds, validate_canonical_runtime_args};
    use crate::args::{Args, MeshMode, SynthesisModel, build_app_args};

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
    fn parse_bounds_rejects_invalid_ranges() {
        let err = parse_bounds(&[1.0, 0.0, 0.0, 0.5, 1.0, 1.0]).expect_err("invalid bounds");
        assert!(err.contains("strictly less"));
    }
}

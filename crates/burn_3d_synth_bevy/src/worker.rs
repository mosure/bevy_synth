use std::any::TypeId;
use std::sync::Mutex;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::{Duration, Instant};

use bevy::prelude::*;
use burn::prelude::*;

use burn_3d_synth_bg_removal::pipeline::{
    PrepareImageConfig, PreparedImageData, RmbgPipeline, prepare_image_data,
    prepare_image_tensor,
};
use burn_3d_synth_tripo::model::triposg::image_encoder::{
    DinoImageProcessor, TripoSGImageEncoder,
};
use burn_3d_synth_tripo::model::triposg::image_encoder::import::{
    load_dinov2_processor, load_triposg_dinov2,
};
use burn_3d_synth_tripo::pipeline::{
    geometry::{FlashExtractConfig, HierarchicalExtractConfig},
    mesh::Mesh as TripoMesh,
    triposg::TripoSGPipeline,
    triposg_scribble::TripoSGScribblePipeline,
};

use crate::args::{AppArgs, BackendKind, DinoBackend, MeshMode, RmbgBackend, DEFAULT_CHUNK_SIZE};
use crate::io::load_text_embeds;
use crate::paths::{resolve_rmbg_root, resolve_scribble_root, resolve_triposg_root};
use crate::state::{InferenceRequest, InferenceWorker, WorkerCommand, WorkerEvent};

const WGPU_CHUNK_SIZE_CAP: usize = 8_192;
const CUDA_CHUNK_SIZE_CAP: usize = 32_768;

pub(crate) fn start_worker(args: &AppArgs) -> InferenceWorker {
    let (command_tx, command_rx) = mpsc::channel();
    let (event_tx, event_rx) = mpsc::channel();
    let args = args.clone();
    let _ = thread::Builder::new()
        .name("triposg-worker".to_string())
        .stack_size(64 * 1024 * 1024)
        .spawn(move || worker_loop(args, command_rx, event_tx))
        .expect("failed to spawn TripoSG worker thread");
    InferenceWorker {
        sender: command_tx,
        receiver: Mutex::new(event_rx),
    }
}

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

fn worker_loop_backend_unavailable(
    command_rx: Receiver<WorkerCommand>,
    event_tx: Sender<WorkerEvent>,
    message: &'static str,
) {
    for command in command_rx {
        match command {
            WorkerCommand::Infer(request) => {
                let _ = event_tx.send(WorkerEvent {
                    request,
                    result: Err(message.to_string()),
                    elapsed: Duration::ZERO,
                });
            }
            WorkerCommand::Shutdown => break,
        }
    }
}

struct PipelineState<B: Backend> {
    device: B::Device,
    rmbg_cpu: Option<RmbgPipeline<burn::backend::NdArray<f32>>>,
    rmbg_device: Option<RmbgPipeline<B>>,
    rmbg_backend: RmbgBackend,
    dino_backend: DinoBackend,
    dino_cpu: Option<DinoCpuState>,
    triposg: Option<TripoSGPipeline<B>>,
    scribble: Option<TripoSGScribblePipeline<B>>,
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
            WorkerCommand::Infer(request) => {
                let start = Instant::now();
                let result = match state.as_mut() {
                    Ok(state) => run_inference_with_state(state, &args, &request),
                    Err(err) => Err(err.clone()),
                };
                let _ = event_tx.send(WorkerEvent {
                    request,
                    result,
                    elapsed: start.elapsed(),
                });
            }
            WorkerCommand::Shutdown => break,
        }
    }
}

fn build_pipeline_state<B: Backend>(args: &AppArgs) -> Result<PipelineState<B>, String> {
    configure_cubecl_autotune::<B>();
    let bounds = parse_bounds(&args.bounds).map_err(|err| err.to_string())?;
    let device = B::Device::default();
    if let Some(seed) = args.seed {
        B::seed(&device, seed);
    }

    if std::env::var("DINO_STRICT_PREPROCESS").is_err() {
        unsafe {
            std::env::set_var("DINO_STRICT_PREPROCESS", "1");
        }
    }
    if std::env::var("RMBG_STRICT_INTERP").is_err() {
        unsafe {
            std::env::set_var("RMBG_STRICT_INTERP", "1");
        }
    }
    if args.match_python && std::env::var("TRIPOSG_MAX_IMAGE_DIM").is_err() {
        unsafe {
            std::env::set_var("TRIPOSG_MAX_IMAGE_DIM", "2000");
        }
    }

    let rmbg_root = resolve_rmbg_root(args.bg_weights_root.as_ref());
    let rmbg_backend = match args.rmbg_backend {
        RmbgBackend::Auto => {
            if is_wgpu_backend::<B>() {
                RmbgBackend::Cpu
            } else {
                RmbgBackend::Gpu
            }
        }
        other => other,
    };
    let (rmbg_cpu, rmbg_device) = match rmbg_backend {
        RmbgBackend::Cpu => {
            let cpu_device = <burn::backend::NdArray<f32> as Backend>::Device::default();
            let rmbg =
                RmbgPipeline::from_pretrained(&rmbg_root, &cpu_device).map_err(|err| {
                    format!("failed to load RMBG weights on CPU: {err}")
                })?;
            (Some(rmbg), None)
        }
        RmbgBackend::Gpu | RmbgBackend::Auto => {
            let rmbg = RmbgPipeline::from_pretrained(&rmbg_root, &device)
                .map_err(|err| format!("failed to load RMBG weights: {err}"))?;
            (None, Some(rmbg))
        }
    };

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

    let dino_backend = match args.dino_backend {
        DinoBackend::Auto => {
            if is_gpu_backend::<B>() {
                if args.match_python && is_wgpu_backend::<B>() {
                    DinoBackend::Cpu
                } else {
                    DinoBackend::Gpu
                }
            } else {
                DinoBackend::Cpu
            }
        }
        other => other,
    };

    let wants_text = args.text_embeds.is_some() || args.prompt.is_some();
    if wants_text {
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
            rmbg_cpu,
            rmbg_device,
            rmbg_backend,
            dino_backend,
            dino_cpu: None,
            triposg: None,
            scribble: Some(scribble),
            text_embeds: Some(text_embeds),
            bounds,
            hierarchical,
            chunk_size,
            flash,
        });
    }

    let weights_root = resolve_triposg_root(args.weights_root.as_ref());
    let dino_cpu = if matches!(dino_backend, DinoBackend::Cpu) {
        let cpu_device = <burn::backend::NdArray<f32> as Backend>::Device::default();
        let weights_path = weights_root.join("image_encoder_dinov2/model.safetensors");
        let encoder = load_triposg_dinov2(&cpu_device, &weights_path)
            .map_err(|err| format!("failed to load DINOv2 weights on CPU: {err}"))?;
        let processor = load_dinov2_processor(&weights_root)
            .map_err(|err| format!("failed to load DINOv2 processor: {err}"))?;
        Some(DinoCpuState {
            device: cpu_device,
            encoder,
            processor,
        })
    } else {
        None
    };

    let triposg = TripoSGPipeline::from_pretrained(weights_root, &device)
        .map_err(|err| format!("failed to load TripoSG weights: {err}"))?;

    Ok(PipelineState {
        device,
        rmbg_cpu,
        rmbg_device,
        rmbg_backend,
        dino_backend,
        dino_cpu,
        triposg: Some(triposg),
        scribble: None,
        text_embeds: None,
        bounds,
        hierarchical,
        chunk_size,
        flash,
    })
}

fn run_inference_with_state<B: Backend>(
    state: &mut PipelineState<B>,
    args: &AppArgs,
    request: &InferenceRequest,
) -> Result<Option<TripoMesh>, String> {
    let device = state.device.clone();
    let prepare_config = prepare_image_config_for_backend::<B>(state.rmbg_backend);
    let (image, prepared_cpu) = match state.rmbg_backend {
        RmbgBackend::Cpu => {
            let rmbg = state
                .rmbg_cpu
                .as_ref()
                .ok_or_else(|| "RMBG CPU pipeline not loaded".to_string())?;
            let prepared: PreparedImageData = prepare_image_data(
                &request.image_path,
                Some(rmbg),
                &prepare_config,
            )
            .map_err(|err| format!("failed to prepare image: {err}"))?;
            let image = prepared.to_tensor::<B>(&device);
            (image, Some(prepared))
        }
        RmbgBackend::Gpu | RmbgBackend::Auto => {
            let image = prepare_image_tensor::<B>(
                &request.image_path,
                state.rmbg_device.as_ref(),
                &device,
                &prepare_config,
            )
            .map_err(|err| format!("failed to prepare image: {err}"))?;
            (image, None)
        }
    };

    if state.scribble.is_some() {
        let text_embeds = state
            .text_embeds
            .as_ref()
            .ok_or_else(|| "text embeddings not loaded".to_string())?
            .clone();
        let pipeline = state
            .scribble
            .as_mut()
            .ok_or_else(|| "scribble pipeline not loaded".to_string())?;
        let output = match args.mesh_mode {
            MeshMode::Dense => pipeline
                .sample_mesh(
                    image,
                    text_embeds,
                    args.num_steps,
                    args.num_tokens,
                    args.guidance_scale,
                    state.bounds,
                    args.resolution,
                    state.chunk_size,
                    None,
                )
                .map_err(|err| err.to_string())?,
            MeshMode::Hierarchical => pipeline
                .sample_mesh_hierarchical(
                    image,
                    text_embeds,
                    args.num_steps,
                    args.num_tokens,
                    args.guidance_scale,
                    &state.hierarchical,
                    None,
                )
                .map_err(|err| err.to_string())?,
            MeshMode::Flash => pipeline
                .sample_mesh_flash(
                    image,
                    text_embeds,
                    args.num_steps,
                    args.num_tokens,
                    args.guidance_scale,
                    &state.flash,
                    None,
                )
                .map_err(|err| err.to_string())?,
        };
        if output.mesh.is_none() {
            log_empty_mesh_stats("TripoSG-scribble", &output.grid);
        }
        return Ok(apply_mesh_decimation(output.mesh, args.target_faces));
    }

    let pipeline = state
        .triposg
        .as_mut()
        .ok_or_else(|| "TripoSG pipeline not loaded".to_string())?;

    if matches!(state.dino_backend, DinoBackend::Cpu) {
        let dino = state
            .dino_cpu
            .as_ref()
            .ok_or_else(|| "DINO CPU encoder not loaded".to_string())?;
        let cpu_image = if let Some(prepared) = prepared_cpu.as_ref() {
            prepared.to_tensor::<burn::backend::NdArray<f32>>(&dino.device)
        } else {
            let data = image
                .clone()
                .into_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .map_err(|_| "failed to read image tensor")?;
            let dims = image.shape().dims::<4>();
            let flat = Tensor::<burn::backend::NdArray<f32>, 1>::from_floats(
                data.as_slice(),
                &dino.device,
            );
            flat.reshape([
                dims[0] as i32,
                dims[1] as i32,
                dims[2] as i32,
                dims[3] as i32,
            ])
        };
        let processed = dino.processor.preprocess(cpu_image);
        let cpu_embeds = dino.encoder.forward(processed);
        let embed_dims = cpu_embeds.shape().dims::<3>();
        let embed_data = cpu_embeds
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .map_err(|_| "failed to read DINO embeddings")?;
        let embeds = Tensor::<B, 1>::from_floats(embed_data.as_slice(), &device).reshape([
            embed_dims[0] as i32,
            embed_dims[1] as i32,
            embed_dims[2] as i32,
        ]);

        let output = match args.mesh_mode {
            MeshMode::Dense => pipeline
                .sample_mesh_from_embeds(
                    embeds,
                    args.num_steps,
                    args.num_tokens,
                    args.guidance_scale,
                    state.bounds,
                    args.resolution,
                    state.chunk_size,
                    None,
                )
                .map_err(|err| err.to_string())?,
            MeshMode::Hierarchical => pipeline
                .sample_mesh_hierarchical_from_embeds(
                    embeds,
                    args.num_steps,
                    args.num_tokens,
                    args.guidance_scale,
                    &state.hierarchical,
                    None,
                )
                .map_err(|err| err.to_string())?,
            MeshMode::Flash => pipeline
                .sample_mesh_flash_from_embeds(
                    embeds,
                    args.num_steps,
                    args.num_tokens,
                    args.guidance_scale,
                    &state.flash,
                    None,
                )
                .map_err(|err| err.to_string())?,
        };
        if output.mesh.is_none() {
            log_empty_mesh_stats("TripoSG (CPU DINO)", &output.grid);
        }
        return Ok(apply_mesh_decimation(output.mesh, args.target_faces));
    }

    let output = match args.mesh_mode {
        MeshMode::Dense => pipeline
            .sample_mesh(
                image,
                args.num_steps,
                args.num_tokens,
                args.guidance_scale,
                state.bounds,
                args.resolution,
                state.chunk_size,
                None,
            )
            .map_err(|err| err.to_string())?,
        MeshMode::Hierarchical => pipeline
            .sample_mesh_hierarchical(
                image,
                args.num_steps,
                args.num_tokens,
                args.guidance_scale,
                &state.hierarchical,
                None,
            )
            .map_err(|err| err.to_string())?,
        MeshMode::Flash => pipeline
            .sample_mesh_flash(
                image,
                args.num_steps,
                args.num_tokens,
                args.guidance_scale,
                &state.flash,
                None,
            )
            .map_err(|err| err.to_string())?,
    };
    if output.mesh.is_none() {
        log_empty_mesh_stats("TripoSG", &output.grid);
    }
    Ok(apply_mesh_decimation(output.mesh, args.target_faces))
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
    if is_wgpu_backend::<B>() && std::env::var("CUBECL_AUTOTUNE_LEVEL").is_err() {
        unsafe {
            std::env::set_var("CUBECL_AUTOTUNE_LEVEL", "minimal");
        }
    }
}

fn prepare_image_config_for_backend<B: Backend>(rmbg_backend: RmbgBackend) -> PrepareImageConfig {
    let mut config = PrepareImageConfig::default();
    if let Some(value) = std::env::var("TRIPOSG_MAX_IMAGE_DIM")
        .ok()
        .and_then(|val| val.parse::<usize>().ok())
        && value > 0
    {
        config.max_dimension = value;
        return config;
    }
    if is_wgpu_backend::<B>() {
        config.max_dimension = config.max_dimension.min(1024);
        if matches!(rmbg_backend, RmbgBackend::Gpu) {
            // Avoid large im2col allocations in wgpu conv autotune for RMBG.
            config.max_dimension = config.max_dimension.min(384);
        }
    }
    config
}

fn log_empty_mesh_stats(
    label: &str,
    grid: &burn_3d_synth_tripo::pipeline::mesh::DenseGrid,
) {
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
    mesh: Option<TripoMesh>,
    target_faces: Option<usize>,
) -> Option<TripoMesh> {
    let target_faces = target_faces.filter(|value| *value > 0);
    let mut mesh = mesh?;
    if let Some(target) = target_faces
        && mesh.faces.len() > target
    {
        match decimate_mesh(&mesh, target) {
            Ok(decimated) => mesh = decimated,
            Err(err) => warn!("mesh decimation failed ({err}); using full mesh."),
        }
    }
    Some(mesh)
}

fn decimate_mesh(mesh: &TripoMesh, target_faces: usize) -> Result<TripoMesh, String> {
    if target_faces == 0 || mesh.faces.len() <= target_faces {
        return Ok(mesh.clone());
    }
    if mesh.faces.is_empty() || mesh.vertices.is_empty() {
        return Ok(mesh.clone());
    }

    let mut indices = Vec::with_capacity(mesh.faces.len() * 3);
    for face in &mesh.faces {
        indices.push(face[0]);
        indices.push(face[1]);
        indices.push(face[2]);
    }
    let target_index_count = (target_faces.saturating_mul(3)).min(indices.len());
    if target_index_count < 3 {
        return Err("target face count too small for decimation".to_string());
    }

    let vertices_bytes = meshopt::typed_to_bytes(mesh.vertices.as_slice());
    let adapter = meshopt::VertexDataAdapter::new(
        vertices_bytes,
        std::mem::size_of::<[f32; 3]>(),
        0,
    )
    .map_err(|err| format!("meshopt vertex adapter: {err}"))?;

    let mut result_error = 0.0f32;
    let mut simplified = meshopt::simplify(
        &indices,
        &adapter,
        target_index_count,
        1.0,
        meshopt::SimplifyOptions::None,
        Some(&mut result_error),
    );
    if simplified.len() > target_index_count {
        simplified = meshopt::simplify_sloppy(
            &indices,
            &adapter,
            target_index_count,
            1.0,
            None,
        );
    }
    if simplified.len() < 3 {
        return Err("meshopt simplification produced empty mesh".to_string());
    }

    let (vertex_count, remap) =
        meshopt::generate_vertex_remap(mesh.vertices.as_slice(), Some(&simplified));
    let vertices = meshopt::remap_vertex_buffer(mesh.vertices.as_slice(), vertex_count, &remap);
    let indices = meshopt::remap_index_buffer(Some(&simplified), vertex_count, &remap);
    if indices.len() < 3 {
        return Err("meshopt remap produced empty mesh".to_string());
    }

    let faces = indices
        .chunks_exact(3)
        .map(|chunk| [chunk[0], chunk[1], chunk[2]])
        .collect::<Vec<[u32; 3]>>();
    Ok(TripoMesh { vertices, faces })
}

fn parse_bounds(bounds: &[f32]) -> Result<[f32; 6], Box<dyn std::error::Error>> {
    if bounds.len() != 6 {
        return Err("bounds must contain exactly 6 floats".into());
    }
    Ok([
        bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5],
    ])
}

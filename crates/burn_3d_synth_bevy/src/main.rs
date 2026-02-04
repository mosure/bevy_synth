use std::any::TypeId;
use std::collections::VecDeque;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use bevy::app::AppExit;
use bevy::camera::primitives::{Aabb, MeshAabb};
use bevy::ecs::message::{MessageReader, MessageWriter};
use bevy::light::PointLight;
use bevy::math::primitives::Sphere;
use bevy::math::Ray3d;
use bevy::pbr::{MeshMaterial3d, StandardMaterial};
use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task};
use bevy::window::{FileDragAndDrop, PrimaryWindow, WindowCloseRequested};
use bevy_asset::RenderAssetUsages;
use bevy_mesh::{Indices, Mesh as BevyMesh, Mesh3d, PrimitiveTopology};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use burn::prelude::*;
use clap::{Parser, ValueEnum};
use futures_lite::future;
use safetensors::tensor::{SafeTensors, TensorView};

use burn_3d_synth_bg_removal::pipeline::{
    prepare_image_tensor,
    PrepareImageConfig,
    RmbgPipeline,
};
use burn_3d_synth_tripo::pipeline::{
    geometry::HierarchicalExtractConfig,
    mesh::Mesh as TripoMesh,
    triposg::TripoSGPipeline,
    triposg_scribble::TripoSGScribblePipeline,
};

#[derive(Parser, Debug)]
#[command(about = "TripoSG Bevy viewer", version)]
struct Args {
    /// Path to an input image for TripoSG inference.
    #[arg(long)]
    image: Option<PathBuf>,

    /// Optional text prompt (scribble model). Requires --text-embeds.
    #[arg(long)]
    prompt: Option<String>,

    /// Path to a safetensors file containing text embeddings for the scribble model.
    #[arg(long)]
    text_embeds: Option<PathBuf>,

    /// Tensor key in the text embedding safetensors file.
    #[arg(long, default_value = "input.text_embeds")]
    text_embeds_key: String,

    /// Optional weights root for TripoSG (image-only) pipeline.
    #[arg(long)]
    weights_root: Option<PathBuf>,

    /// Optional weights root for TripoSG-scribble pipeline.
    #[arg(long)]
    scribble_weights_root: Option<PathBuf>,

    /// Number of diffusion steps.
    #[arg(long, default_value_t = 50)]
    num_steps: usize,

    /// Number of latents tokens.
    #[arg(long, default_value_t = 2048)]
    num_tokens: usize,

    /// Guidance scale.
    #[arg(long, default_value_t = 7.0)]
    guidance_scale: f32,

    /// Optional RNG seed for deterministic sampling.
    #[arg(long)]
    seed: Option<u64>,

    /// Grid resolution used for mesh extraction.
    #[arg(long, default_value_t = 256)]
    resolution: usize,

    /// Chunk size for VAE grid decoding.
    #[arg(long, default_value_t = DEFAULT_CHUNK_SIZE)]
    chunk_size: usize,

    /// Bounds for grid decoding (6 floats: minX minY minZ maxX maxY maxZ).
    #[arg(
        long,
        num_args = 6,
        value_delimiter = ' ',
        default_value = "-1.005 -1.005 -1.005 1.005 1.005 1.005"
    )]
    bounds: Vec<f32>,

    /// Mesh extraction mode.
    #[arg(long, value_enum, default_value_t = MeshMode::Hierarchical)]
    mesh_mode: MeshMode,

    /// Dense octree depth used for hierarchical extraction.
    #[arg(long, default_value_t = 8)]
    dense_octree_depth: usize,

    /// Hierarchical octree depth used for hierarchical extraction.
    #[arg(long, default_value_t = 9)]
    hierarchical_octree_depth: usize,

    /// Band threshold used to expand near-surface regions in hierarchical extraction.
    #[arg(long, default_value_t = 1.0)]
    band_threshold: f32,

    /// Path to write an OBJ file for the inferred mesh.
    #[arg(long)]
    output: Option<PathBuf>,

    /// Path to an existing mesh file to display (glb/obj/gltf).
    #[arg(long)]
    mesh: Option<PathBuf>,

    /// Optional weights root for RMBG background removal.
    #[arg(long)]
    bg_weights_root: Option<PathBuf>,

    /// Inference backend (cpu, wgpu, cuda).
    #[arg(long, value_enum, default_value_t = BackendKind::Wgpu)]
    backend: BackendKind,
}

#[derive(ValueEnum, Clone, Debug)]
enum BackendKind {
    Cpu,
    Wgpu,
    Cuda,
}

#[derive(ValueEnum, Clone, Debug)]
enum MeshMode {
    Dense,
    Hierarchical,
}

const DEFAULT_CHUNK_SIZE: usize = 10_000;
const WGPU_CHUNK_SIZE_CAP: usize = 8_192;
const CUDA_CHUNK_SIZE_CAP: usize = 32_768;

#[derive(Resource, Debug, Clone)]
struct AppArgs {
    image: Option<PathBuf>,
    prompt: Option<String>,
    text_embeds: Option<PathBuf>,
    text_embeds_key: String,
    weights_root: Option<PathBuf>,
    scribble_weights_root: Option<PathBuf>,
    num_steps: usize,
    num_tokens: usize,
    guidance_scale: f32,
    seed: Option<u64>,
    resolution: usize,
    chunk_size: usize,
    bounds: Vec<f32>,
    mesh_mode: MeshMode,
    dense_octree_depth: usize,
    hierarchical_octree_depth: usize,
    band_threshold: f32,
    output: Option<PathBuf>,
    mesh: Option<PathBuf>,
    bg_weights_root: Option<PathBuf>,
    backend: BackendKind,
}

#[derive(Resource, Default)]
struct UiStatus {
    message: String,
    processing: bool,
}

#[derive(Resource)]
struct TitlePulse {
    timer: Timer,
    phase: usize,
}

impl Default for TitlePulse {
    fn default() -> Self {
        Self {
            timer: Timer::from_seconds(0.5, TimerMode::Repeating),
            phase: 0,
        }
    }
}

#[derive(Resource, Default)]
struct DragState {
    active: Option<DragSelection>,
}

#[derive(Resource, Default)]
struct ExitState {
    requested: bool,
}

struct DragSelection {
    entity: Entity,
    plane_y: f32,
    offset: Vec3,
}

#[derive(Resource, Default)]
struct InferenceQueue {
    active: Option<InferenceTask>,
    pending: VecDeque<InferenceRequest>,
    counter: u32,
    completed: usize,
}

#[derive(Clone, Debug)]
struct InferenceRequest {
    image_path: PathBuf,
    output_path: Option<PathBuf>,
}

struct InferenceTask {
    request: InferenceRequest,
    task: Task<Result<Option<TripoMesh>, String>>,
}

#[derive(Component)]
struct DraggableMesh {
    local_min: Vec3,
    local_max: Vec3,
}

#[derive(Component)]
struct PendingAabb {
    handle: Handle<BevyMesh>,
}

#[derive(Component)]
struct Spinner;

fn main() {
    let args = Args::parse();

    let status_message = if args.image.is_none() && args.mesh.is_none() {
        "Drag & drop an image (.png/.jpg) or mesh (.glb/.gltf/.obj) to begin.".to_string()
    } else {
        "Initializing TripoSG viewer…".to_string()
    };

    App::new()
        .insert_resource(AppArgs {
            image: args.image,
            prompt: args.prompt,
            text_embeds: args.text_embeds,
            text_embeds_key: args.text_embeds_key,
            weights_root: args.weights_root,
            scribble_weights_root: args.scribble_weights_root,
            num_steps: args.num_steps,
            num_tokens: args.num_tokens,
            guidance_scale: args.guidance_scale,
            seed: args.seed,
            resolution: args.resolution,
            chunk_size: args.chunk_size,
            bounds: args.bounds,
            mesh_mode: args.mesh_mode,
            dense_octree_depth: args.dense_octree_depth,
            hierarchical_octree_depth: args.hierarchical_octree_depth,
            band_threshold: args.band_threshold,
            output: args.output,
            mesh: args.mesh,
            bg_weights_root: args.bg_weights_root,
            backend: args.backend,
        })
        .insert_resource(InferenceQueue::default())
        .insert_resource(DragState::default())
        .insert_resource(ExitState::default())
        .insert_resource(TitlePulse::default())
        .insert_resource(UiStatus {
            message: status_message,
            processing: false,
        })
        .add_plugins(DefaultPlugins)
        .add_plugins(PanOrbitCameraPlugin)
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                handle_exit_requests,
                handle_file_drop,
                drive_inference,
                update_pending_aabb,
                (drag_start, drag_update, drag_end).chain(),
                sync_panorbit_enabled,
                update_spinner,
                rotate_spinner,
                update_window_title,
            ),
        )
        .run();
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<BevyMesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    args: Res<AppArgs>,
    mut queue: ResMut<InferenceQueue>,
    mut status: ResMut<UiStatus>,
) {
    info!("burn_3d_synth_bevy args: {:?}", *args);

    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(Vec3::new(0.0, 1.5, 5.0)),
        PanOrbitCamera {
            allow_upside_down: true,
            orbit_smoothness: 0.1,
            pan_smoothness: 0.1,
            zoom_smoothness: 0.1,
            ..default()
        },
    ));
    commands.spawn((
        PointLight::default(),
        Transform::from_xyz(2.0, 3.0, 2.0),
    ));

    let spinner_mesh = meshes.add(BevyMesh::from(Sphere::new(0.12)));
    let spinner_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.95, 0.75, 0.35),
        emissive: Color::srgb(0.2, 0.1, 0.05).into(),
        ..default()
    });
    commands.spawn((
        Spinner,
        Mesh3d(spinner_mesh),
        MeshMaterial3d(spinner_material),
        Transform::from_xyz(0.0, 0.75, 0.0),
        Visibility::Hidden,
    ));

    if let Some(mesh_path) = args.mesh.as_ref() {
        if mesh_path.exists() {
            spawn_mesh_asset(
                &mut commands,
                &asset_server,
                &mut materials,
                mesh_path.clone(),
            );
        } else {
            warn!("Mesh path {:?} does not exist; skipping", mesh_path);
        }
    }

    if let Some(image_path) = args.image.as_ref() {
        enqueue_inference(image_path.clone(), &args, &mut queue);
    }

    update_status_message(&args, &queue, &mut status);
}

#[allow(clippy::too_many_arguments)]
fn handle_file_drop(
    mut events: MessageReader<FileDragAndDrop>,
    mut queue: ResMut<InferenceQueue>,
    args: Res<AppArgs>,
    asset_server: Res<AssetServer>,
    mut commands: Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut status: ResMut<UiStatus>,
    exit_state: Res<ExitState>,
) {
    if exit_state.requested {
        return;
    }
    for event in events.read() {
        let FileDragAndDrop::DroppedFile { path_buf, .. } = event else {
            continue;
        };

        if is_image_file(path_buf) {
            enqueue_inference(path_buf.clone(), &args, &mut queue);
            info!("Queued inference for {}", path_buf.display());
            continue;
        }

        if is_mesh_file(path_buf) {
            spawn_mesh_asset(
                &mut commands,
                &asset_server,
                &mut materials,
                path_buf.clone(),
            );
            info!("Loaded mesh asset {}", path_buf.display());
            continue;
        }

        warn!(
            "Dropped file {} is not a supported image or mesh",
            path_buf.display()
        );
    }

    update_status_message(&args, &queue, &mut status);
}

fn drive_inference(
    mut commands: Commands,
    mut queue: ResMut<InferenceQueue>,
    args: Res<AppArgs>,
    mut meshes: ResMut<Assets<BevyMesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut status: ResMut<UiStatus>,
    exit_state: Res<ExitState>,
) {
    if exit_state.requested {
        return;
    }
    if let Some(active) = queue.active.as_mut()
        && let Some(result) = future::block_on(future::poll_once(&mut active.task))
    {
        let request = active.request.clone();
        queue.active = None;
        queue.completed += 1;
        handle_inference_result(
            &mut commands,
            &mut meshes,
            &mut materials,
            request,
            result,
        );
    }

    if queue.active.is_none()
        && let Some(request) = queue.pending.pop_front()
    {
        let args = args.clone();
        let task_request = request.clone();
        let task = AsyncComputeTaskPool::get().spawn(async move {
            let run = std::panic::AssertUnwindSafe(|| {
                run_inference(&task_request.image_path, &args)
            });
            std::panic::catch_unwind(run)
                .map_err(|_| "panic during inference".to_string())?
                .map_err(|err| err.to_string())
        });
        queue.active = Some(InferenceTask { request, task });
    }

    update_status_message(&args, &queue, &mut status);
}

fn update_pending_aabb(
    mut commands: Commands,
    meshes: Res<Assets<BevyMesh>>,
    pending: Query<(Entity, &PendingAabb)>,
) {
    for (entity, pending) in pending.iter() {
        let Some(mesh) = meshes.get(&pending.handle) else {
            continue;
        };
        let Some(aabb) = mesh.compute_aabb() else {
            continue;
        };
        let (min, max) = aabb_min_max(&aabb);
        commands
            .entity(entity)
            .insert(DraggableMesh {
                local_min: min,
                local_max: max,
            })
            .remove::<PendingAabb>();
    }
}

fn drag_start(
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    draggables: Query<(Entity, &DraggableMesh, &GlobalTransform)>,
    mut drag_state: ResMut<DragState>,
) {
    if !buttons.just_pressed(MouseButton::Left) {
        return;
    }

    let Ok(window) = windows.single() else {
        return;
    };
    let Some(cursor) = window.cursor_position() else {
        return;
    };
    let Ok((camera, camera_transform)) = cameras.single() else {
        return;
    };
    let Ok(ray) = camera.viewport_to_world(camera_transform, cursor) else {
        return;
    };

    let mut best: Option<(Entity, f32, Vec3, Vec3)> = None;
    for (entity, bounds, transform) in draggables.iter() {
        let (min, max) = world_aabb(bounds, transform);
        let Some(dist) = ray_aabb_intersection(ray, min, max) else {
            continue;
        };
        let hit = ray.get_point(dist);
        let translation = transform.translation();
        if best
            .as_ref()
            .map(|(_, best_dist, _, _)| dist < *best_dist)
            .unwrap_or(true)
        {
            best = Some((entity, dist, hit, translation));
        }
    }

    if let Some((entity, _dist, hit, translation)) = best {
        let offset = translation - hit;
        drag_state.active = Some(DragSelection {
            entity,
            plane_y: translation.y,
            offset,
        });
    }
}

fn drag_update(
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    mut drag_state: ResMut<DragState>,
    mut transforms: Query<&mut Transform>,
) {
    let Some(selection) = drag_state.active.as_mut() else {
        return;
    };
    if !buttons.pressed(MouseButton::Left) {
        return;
    }

    let Ok(window) = windows.single() else {
        return;
    };
    let Some(cursor) = window.cursor_position() else {
        return;
    };
    let Ok((camera, camera_transform)) = cameras.single() else {
        return;
    };
    let Ok(ray) = camera.viewport_to_world(camera_transform, cursor) else {
        return;
    };
    let Some(hit) = ray_plane_intersection(ray, selection.plane_y) else {
        return;
    };

    if let Ok(mut transform) = transforms.get_mut(selection.entity) {
        transform.translation = hit + selection.offset;
    }
}

fn drag_end(buttons: Res<ButtonInput<MouseButton>>, mut drag_state: ResMut<DragState>) {
    if buttons.just_released(MouseButton::Left) {
        drag_state.active = None;
    }
}

fn sync_panorbit_enabled(
    drag_state: Res<DragState>,
    mut cameras: Query<&mut PanOrbitCamera>,
) {
    let enabled = drag_state.active.is_none();
    for mut camera in cameras.iter_mut() {
        camera.enabled = enabled;
    }
}

fn update_spinner(
    queue: Res<InferenceQueue>,
    mut query: Query<&mut Visibility, With<Spinner>>,
) {
    let visible = queue.active.is_some();
    for mut visibility in query.iter_mut() {
        *visibility = if visible {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
    }
}

fn rotate_spinner(
    time: Res<Time>,
    queue: Res<InferenceQueue>,
    mut query: Query<&mut Transform, With<Spinner>>,
) {
    if queue.active.is_none() {
        return;
    }
    for mut transform in query.iter_mut() {
        transform.rotate_y(time.delta_secs() * 1.5);
        transform.rotate_x(time.delta_secs() * 0.8);
    }
}

fn update_window_title(
    time: Res<Time>,
    queue: Res<InferenceQueue>,
    status: Res<UiStatus>,
    mut pulse: ResMut<TitlePulse>,
    mut windows: Query<&mut Window, With<PrimaryWindow>>,
) {
    let processing = queue.active.is_some();
    let mut should_update = status.is_changed();

    if processing {
        pulse.timer.tick(time.delta());
        if pulse.timer.just_finished() {
            pulse.phase = (pulse.phase + 1) % 4;
            should_update = true;
        }
    } else if pulse.phase != 0 {
        pulse.phase = 0;
        should_update = true;
    }

    if !should_update {
        return;
    }

    let title = if let Some(active) = queue.active.as_ref() {
        let dots = ".".repeat(pulse.phase);
        let name = active
            .request
            .image_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("image");
        format!(
            "TripoSG Viewer — Processing: {name} (Queued: {}){dots}",
            queue.pending.len()
        )
    } else {
        format!("TripoSG Viewer — {}", status.message)
    };

    if let Ok(mut window) = windows.single_mut() {
        window.title = title;
    }
}

fn handle_exit_requests(
    keys: Res<ButtonInput<KeyCode>>,
    mut close_events: MessageReader<WindowCloseRequested>,
    mut exit: MessageWriter<AppExit>,
    mut exit_state: ResMut<ExitState>,
    mut queue: ResMut<InferenceQueue>,
    mut status: ResMut<UiStatus>,
) {
    if exit_state.requested {
        return;
    }
    let mut requested = keys.just_pressed(KeyCode::Escape);
    if close_events.read().next().is_some() {
        requested = true;
    }
    if !requested {
        return;
    }

    exit_state.requested = true;
    queue.active = None;
    queue.pending.clear();
    status.processing = false;
    status.message = "Shutting down…".to_string();
    info!("{}", status.message);
    exit.write(AppExit::Success);
}

fn handle_inference_result(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<BevyMesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    request: InferenceRequest,
    result: Result<Option<TripoMesh>, String>,
) {
    match result {
        Ok(Some(mesh)) => {
            if let Some(output) = request.output_path.as_ref()
                && let Err(err) = write_obj(output, &mesh)
            {
                warn!("failed to write mesh to {}: {err}", output.display());
            }

            let (local_min, local_max) = mesh_bounds(&mesh);
            let bevy_mesh = to_bevy_mesh(&mesh);
            let mesh_handle = meshes.add(bevy_mesh);
            let material = materials.add(StandardMaterial {
                base_color: Color::srgb(0.78, 0.84, 0.92),
                ..default()
            });

            commands.spawn((
                DraggableMesh {
                    local_min,
                    local_max,
                },
                Mesh3d(mesh_handle),
                MeshMaterial3d(material),
            ));
        }
        Ok(None) => {
            warn!("TripoSG inference produced an empty mesh for {}", request.image_path.display());
        }
        Err(err) => {
            warn!(
                "TripoSG inference failed for {}: {}",
                request.image_path.display(),
                err
            );
        }
    }
}

fn enqueue_inference(image_path: PathBuf, args: &AppArgs, queue: &mut InferenceQueue) {
    let output_path = resolve_output_path(args.output.as_ref(), &image_path, queue.counter);
    let request = InferenceRequest {
        image_path,
        output_path,
    };
    queue.counter = queue.counter.wrapping_add(1);
    queue.pending.push_back(request);
}

fn update_status_message(args: &AppArgs, queue: &InferenceQueue, status: &mut UiStatus) {
    let message = if let Some(active) = queue.active.as_ref() {
        format!(
            "Processing {} ({} queued)",
            active.request.image_path.display(),
            queue.pending.len()
        )
    } else if !queue.pending.is_empty() {
        format!("Queued {} inference job(s)…", queue.pending.len())
    } else if args.image.is_none() && args.mesh.is_none() && queue.completed == 0 {
        "Drag & drop an image (.png/.jpg) or mesh (.glb/.gltf/.obj) to begin.".to_string()
    } else {
        "Ready. Drag & drop another image or mesh to add more.".to_string()
    };

    status.processing = queue.active.is_some();
    if status.message != message {
        status.message = message;
        info!("{}", status.message);
    }
}

fn spawn_mesh_asset(
    commands: &mut Commands,
    asset_server: &Res<AssetServer>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    mesh_path: PathBuf,
) {
    let mesh_handle: Handle<BevyMesh> = asset_server.load(mesh_path);
    let material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.82, 0.82, 0.9),
        ..default()
    });
    commands.spawn((
        PendingAabb {
            handle: mesh_handle.clone(),
        },
        Mesh3d(mesh_handle),
        MeshMaterial3d(material),
    ));
}

fn run_inference(
    image_path: &Path,
    args: &AppArgs,
) -> Result<Option<TripoMesh>, Box<dyn std::error::Error>> {
    match args.backend {
        BackendKind::Cpu => {
            run_inference_with_backend::<burn::backend::NdArray<f32>>(image_path, args)
        }
        BackendKind::Wgpu => {
            #[cfg(feature = "wgpu")]
            {
                run_inference_with_backend::<burn_wgpu::Wgpu>(image_path, args)
            }
            #[cfg(not(feature = "wgpu"))]
            {
                Err("wgpu backend not enabled (enable the `wgpu` feature)".into())
            }
        }
        BackendKind::Cuda => {
            #[cfg(feature = "cuda")]
            {
                run_inference_with_backend::<burn_cuda::Cuda>(image_path, args)
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err("cuda backend not enabled (enable the `cuda` feature)".into())
            }
        }
    }
}

fn run_inference_with_backend<B: Backend>(
    image_path: &Path,
    args: &AppArgs,
) -> Result<Option<TripoMesh>, Box<dyn std::error::Error>> {
    configure_cubecl_autotune::<B>();
    let bounds = parse_bounds(&args.bounds)?;
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

    let rmbg_root = resolve_rmbg_root(args.bg_weights_root.as_ref());
    let rmbg_pipeline = RmbgPipeline::from_pretrained(rmbg_root, &device)?;
    let image = prepare_image_tensor::<B>(
        image_path,
        Some(&rmbg_pipeline),
        &device,
        &PrepareImageConfig::default(),
    )?;

    let tuned_chunk_size = tuned_chunk_size::<B>(args.chunk_size);
    let hierarchical_config = HierarchicalExtractConfig {
        bounds,
        dense_octree_depth: args.dense_octree_depth,
        hierarchical_octree_depth: args.hierarchical_octree_depth,
        chunk_size: tuned_chunk_size,
        band_threshold: args.band_threshold,
    };

    let wants_text = args.text_embeds.is_some() || args.prompt.is_some();
    if wants_text {
        let text_path = args
            .text_embeds
            .as_ref()
            .ok_or("text prompt provided without --text-embeds")?;
        let text_embeds = load_text_embeds::<B>(text_path, &args.text_embeds_key, &device)?;

        let weights_root = resolve_scribble_root(
            args.scribble_weights_root
                .as_ref()
                .or(args.weights_root.as_ref()),
        );

        let mut pipeline = TripoSGScribblePipeline::from_pretrained(weights_root, &device)?;
        let output = match args.mesh_mode {
            MeshMode::Dense => pipeline.sample_mesh(
                image,
                text_embeds,
                args.num_steps,
                args.num_tokens,
                args.guidance_scale,
                bounds,
                args.resolution,
                tuned_chunk_size,
                None,
            )?,
            MeshMode::Hierarchical => pipeline.sample_mesh_hierarchical(
                image,
                text_embeds,
                args.num_steps,
                args.num_tokens,
                args.guidance_scale,
                &hierarchical_config,
                None,
            )?,
        };
        return Ok(output.mesh);
    }

    let weights_root = resolve_triposg_root(args.weights_root.as_ref());
    let mut pipeline = TripoSGPipeline::from_pretrained(weights_root, &device)?;
    let output = match args.mesh_mode {
        MeshMode::Dense => pipeline.sample_mesh(
            image,
            args.num_steps,
            args.num_tokens,
            args.guidance_scale,
            bounds,
            args.resolution,
            tuned_chunk_size,
            None,
        )?,
        MeshMode::Hierarchical => pipeline.sample_mesh_hierarchical(
            image,
            args.num_steps,
            args.num_tokens,
            args.guidance_scale,
            &hierarchical_config,
            None,
        )?,
    };
    Ok(output.mesh)
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
        return TypeId::of::<B>() == TypeId::of::<burn_wgpu::Wgpu>();
    }
    #[cfg(not(feature = "wgpu"))]
    {
        false
    }
}

fn is_cuda_backend<B: Backend>() -> bool {
    #[cfg(feature = "cuda")]
    {
        return TypeId::of::<B>() == TypeId::of::<burn_cuda::Cuda>();
    }
    #[cfg(not(feature = "cuda"))]
    {
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

fn load_text_embeds<B: Backend>(
    path: &Path,
    key: &str,
    device: &B::Device,
) -> Result<Tensor<B, 3>, Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    let safetensors = SafeTensors::deserialize(&bytes)?;
    let view = match safetensors.tensor(key) {
        Ok(tensor) => tensor,
        Err(_) => {
            let names = safetensors.names();
            if names.len() == 1 {
                safetensors.tensor(names[0])?
            } else {
                let available = names.join(", ");
                return Err(format!(
                    "text embeddings key '{key}' not found; available tensors: {available}"
                )
                .into());
            }
        }
    };

    let data = tensor_view_to_f32(&view)?;
    let shape = view.shape();
    let (batch, seq, dim) = match shape.len() {
        2 => (1, shape[0], shape[1]),
        3 => (shape[0], shape[1], shape[2]),
        _ => {
            return Err(format!(
                "expected text embeddings with rank 2 or 3, got shape {:?}",
                shape
            )
            .into())
        }
    };

    let tensor = Tensor::<B, 1>::from_floats(data.as_slice(), device)
        .reshape([batch as i32, seq as i32, dim as i32]);
    Ok(tensor)
}

fn tensor_view_to_f32(view: &TensorView<'_>) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    use safetensors::Dtype;
    match view.dtype() {
        Dtype::F32 => {
            let data = bytemuck::cast_slice::<u8, f32>(view.data());
            Ok(data.to_vec())
        }
        Dtype::F16 => {
            let data = bytemuck::cast_slice::<u8, half::f16>(view.data());
            Ok(data.iter().map(|value| f32::from(*value)).collect())
        }
        Dtype::BF16 => {
            let data = bytemuck::cast_slice::<u8, half::bf16>(view.data());
            Ok(data.iter().map(|value| f32::from(*value)).collect())
        }
        other => Err(format!("unsupported text embedding dtype {other:?}").into()),
    }
}

fn to_bevy_mesh(mesh: &TripoMesh) -> BevyMesh {
    let mut bevy_mesh =
        BevyMesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());

    let normals = compute_normals(mesh);
    let uvs = vec![[0.0, 0.0]; mesh.vertices.len()];
    let indices: Vec<u32> = mesh
        .faces
        .iter()
        .flat_map(|face| face.iter().copied())
        .collect();

    bevy_mesh.insert_attribute(BevyMesh::ATTRIBUTE_POSITION, mesh.vertices.clone());
    bevy_mesh.insert_attribute(BevyMesh::ATTRIBUTE_NORMAL, normals);
    bevy_mesh.insert_attribute(BevyMesh::ATTRIBUTE_UV_0, uvs);
    bevy_mesh.insert_indices(Indices::U32(indices));
    bevy_mesh
}

fn compute_normals(mesh: &TripoMesh) -> Vec<[f32; 3]> {
    let mut normals = vec![[0.0f32; 3]; mesh.vertices.len()];
    for face in &mesh.faces {
        let [i0, i1, i2] = *face;
        let v0 = mesh.vertices[i0 as usize];
        let v1 = mesh.vertices[i1 as usize];
        let v2 = mesh.vertices[i2 as usize];
        let e1 = [
            v1[0] - v0[0],
            v1[1] - v0[1],
            v1[2] - v0[2],
        ];
        let e2 = [
            v2[0] - v0[0],
            v2[1] - v0[1],
            v2[2] - v0[2],
        ];
        let n = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        for &idx in &[i0, i1, i2] {
            let entry = &mut normals[idx as usize];
            entry[0] += n[0];
            entry[1] += n[1];
            entry[2] += n[2];
        }
    }

    for normal in &mut normals {
        let length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2])
            .sqrt();
        if length > 1e-6 {
            normal[0] /= length;
            normal[1] /= length;
            normal[2] /= length;
        } else {
            *normal = [0.0, 1.0, 0.0];
        }
    }

    normals
}

fn mesh_bounds(mesh: &TripoMesh) -> (Vec3, Vec3) {
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for v in &mesh.vertices {
        min = min.min(Vec3::new(v[0], v[1], v[2]));
        max = max.max(Vec3::new(v[0], v[1], v[2]));
    }
    (min, max)
}

fn parse_bounds(bounds: &[f32]) -> Result<[f32; 6], Box<dyn std::error::Error>> {
    if bounds.len() != 6 {
        return Err("bounds must contain exactly 6 floats".into());
    }
    Ok([
        bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5],
    ])
}

fn resolve_triposg_root(explicit: Option<&PathBuf>) -> PathBuf {
    if let Some(path) = explicit
        && let Some(root) = normalize_weights_root(path)
    {
        return root;
    }
    if let Ok(root) = std::env::var("TRIPOSG_WEIGHTS_ROOT") {
        let path = PathBuf::from(root);
        if let Some(root) = normalize_weights_root(&path) {
            return root;
        }
    }
    let fallback = PathBuf::from(r"E:\repos\TripoSG\pretrained_weights\TripoSG");
    if let Some(root) = normalize_weights_root(&fallback) {
        return root;
    }
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let local = manifest_dir.join("../burn_3d_synth_tripo/assets/models/MIDI-3D");
    normalize_weights_root(&local).unwrap_or(local)
}

fn resolve_rmbg_root(explicit: Option<&PathBuf>) -> PathBuf {
    if let Some(path) = explicit
        && let Some(root) = normalize_rmbg_root(path)
    {
        return root;
    }
    if let Ok(root) = std::env::var("RMBG_WEIGHTS_ROOT") {
        let path = PathBuf::from(root);
        if let Some(root) = normalize_rmbg_root(&path) {
            return root;
        }
    }
    let fallback = PathBuf::from(r"E:\repos\TripoSG\pretrained_weights\RMBG-1.4");
    if let Some(root) = normalize_rmbg_root(&fallback) {
        return root;
    }
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let local = manifest_dir.join("../burn_3d_synth_bg_removal/assets/models/RMBG-1.4");
    normalize_rmbg_root(&local).unwrap_or(local)
}

fn resolve_scribble_root(explicit: Option<&PathBuf>) -> PathBuf {
    if let Some(path) = explicit
        && let Some(root) = normalize_weights_root(path)
    {
        return root;
    }
    if let Ok(root) = std::env::var("TRIPOSG_SCRIBBLE_WEIGHTS_ROOT") {
        let path = PathBuf::from(root);
        if let Some(root) = normalize_weights_root(&path) {
            return root;
        }
    }
    let fallback = PathBuf::from(r"E:\repos\TripoSG\pretrained_weights\TripoSG-scribble");
    if let Some(root) = normalize_weights_root(&fallback) {
        return root;
    }
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let local = manifest_dir.join("../burn_3d_synth_tripo/assets/models/TripoSG-scribble");
    normalize_weights_root(&local).unwrap_or(local)
}

fn normalize_weights_root(path: &Path) -> Option<PathBuf> {
    if path.is_dir() {
        return Some(path.to_path_buf());
    }
    if path.is_file()
        && let Some(parent) = path.parent().and_then(|p| p.parent())
    {
        return Some(parent.to_path_buf());
    }
    None
}

fn normalize_rmbg_root(path: &Path) -> Option<PathBuf> {
    if path.is_dir() {
        return Some(path.to_path_buf());
    }
    if path.is_file()
        && path.file_name().and_then(|n| n.to_str()) == Some("model.safetensors")
    {
        return path.parent().map(|p| p.to_path_buf());
    }
    None
}

fn write_obj(path: &Path, mesh: &TripoMesh) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let handle = fs::File::create(path)?;
    let mut writer = BufWriter::new(handle);
    for v in &mesh.vertices {
        writeln!(writer, "v {} {} {}", v[0], v[1], v[2])?;
    }
    for face in &mesh.faces {
        writeln!(
            writer,
            "f {} {} {}",
            face[0] + 1,
            face[1] + 1,
            face[2] + 1
        )?;
    }
    writer.flush()?;
    Ok(())
}

fn resolve_output_path(output: Option<&PathBuf>, image_path: &Path, index: u32) -> Option<PathBuf> {
    let output = output?;
    if output.extension().is_none() || output.is_dir() {
        let dir = output.to_path_buf();
        let stem = image_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("mesh");
        let suffix = if index == 0 {
            String::new()
        } else {
            format!("_{index}")
        };
        return Some(dir.join(format!("{stem}{suffix}.obj")));
    }

    if index == 0 {
        return Some(output.clone());
    }
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    let stem = output
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("mesh");
    let ext = output.extension().and_then(|s| s.to_str()).unwrap_or("obj");
    Some(parent.join(format!("{stem}_{index}.{ext}")))
}

fn is_image_file(path: &Path) -> bool {
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
    matches!(
        ext.to_ascii_lowercase().as_str(),
        "png" | "jpg" | "jpeg" | "bmp" | "gif" | "webp" | "tga" | "tif" | "tiff"
    )
}

fn is_mesh_file(path: &Path) -> bool {
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
    matches!(
        ext.to_ascii_lowercase().as_str(),
        "glb" | "gltf" | "obj" | "fbx"
    )
}

fn aabb_min_max(aabb: &Aabb) -> (Vec3, Vec3) {
    let min = Vec3::from(aabb.min());
    let max = Vec3::from(aabb.max());
    (min, max)
}

fn world_aabb(bounds: &DraggableMesh, transform: &GlobalTransform) -> (Vec3, Vec3) {
    let min = bounds.local_min;
    let max = bounds.local_max;
    let corners = [
        Vec3::new(min.x, min.y, min.z),
        Vec3::new(min.x, min.y, max.z),
        Vec3::new(min.x, max.y, min.z),
        Vec3::new(min.x, max.y, max.z),
        Vec3::new(max.x, min.y, min.z),
        Vec3::new(max.x, min.y, max.z),
        Vec3::new(max.x, max.y, min.z),
        Vec3::new(max.x, max.y, max.z),
    ];

    let mut world_min = Vec3::splat(f32::INFINITY);
    let mut world_max = Vec3::splat(f32::NEG_INFINITY);
    for corner in corners {
        let world = transform.transform_point(corner);
        world_min = world_min.min(world);
        world_max = world_max.max(world);
    }
    (world_min, world_max)
}

fn ray_aabb_intersection(ray: Ray3d, min: Vec3, max: Vec3) -> Option<f32> {
    let mut tmin = 0.0f32;
    let mut tmax = f32::INFINITY;
    let origin = ray.origin;
    let dir = ray.direction.as_vec3();

    for i in 0..3 {
        let origin_axis = origin[i];
        let dir_axis = dir[i];
        let min_axis = min[i];
        let max_axis = max[i];

        if dir_axis.abs() < 1e-6 {
            if origin_axis < min_axis || origin_axis > max_axis {
                return None;
            }
        } else {
            let inv = 1.0 / dir_axis;
            let mut t1 = (min_axis - origin_axis) * inv;
            let mut t2 = (max_axis - origin_axis) * inv;
            if t1 > t2 {
                std::mem::swap(&mut t1, &mut t2);
            }
            tmin = tmin.max(t1);
            tmax = tmax.min(t2);
            if tmax < tmin {
                return None;
            }
        }
    }

    if tmax >= 0.0 {
        Some(tmin.max(0.0))
    } else {
        None
    }
}

fn ray_plane_intersection(ray: Ray3d, plane_y: f32) -> Option<Vec3> {
    let dir = ray.direction.as_vec3();
    let denom = dir.y;
    if denom.abs() < 1e-6 {
        return None;
    }
    let t = (plane_y - ray.origin.y) / denom;
    if t < 0.0 {
        return None;
    }
    Some(ray.origin + dir * t)
}

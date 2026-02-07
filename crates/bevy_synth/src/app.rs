use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::mpsc::TryRecvError;

use bevy::app::AppExit;
#[cfg(target_arch = "wasm32")]
use bevy::asset::io::web::WebAssetPlugin;
#[cfg(target_arch = "wasm32")]
use bevy::asset::{AssetMetaCheck, AssetMode, AssetPlugin, UnapprovedPathMode};
use bevy::camera::ClearColorConfig;
use bevy::camera::primitives::MeshAabb;
use bevy::camera::visibility::RenderLayers;
use bevy::ecs::message::{MessageReader, MessageWriter};
use bevy::ecs::system::SystemParam;
use bevy::light::PointLight;
use bevy::pbr::{MeshMaterial3d, StandardMaterial};
use bevy::prelude::*;
use bevy::ui::IsDefaultUiCamera;
use bevy::window::{FileDragAndDrop, PrimaryWindow, WindowCloseRequested};
use bevy_editor_core::selection::{
    EditorSelection, Selectable, remove_entity_from_selection_if_despawned,
};
#[cfg(not(target_arch = "wasm32"))]
use bevy_file_dialog::prelude::{DialogFilePicked, FileDialogExt, FileDialogPlugin};
use bevy_infinite_grid::{InfiniteGridBundle, InfiniteGridPlugin};
use bevy_mesh::{Mesh as BevyMesh, Mesh3d};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin, PanOrbitCameraSystemSet};
use bevy_picking::DefaultPickingPlugins;
use bevy_picking::hover::PickingInteraction;
use bevy_picking::input::PointerInputPlugin;
use bevy_picking::prelude::{MeshPickingCamera, MeshPickingSettings, Pickable};
use bevy_transform_gizmos::prelude::{GizmoCamera, TransformGizmoPlugin};
use bevy_transform_gizmos::{GizmoTransformable, TransformGizmo, TransformGizmoSystems};
use clap::Parser;

use bevy_synth_runtime::TripoMesh;
use bevy_synth_runtime::args::{AppArgs, Args, build_app_args};
use bevy_synth_runtime::cache::{CachedWorldItem, MeshCache};
use bevy_synth_runtime::io::{is_image_file, is_mesh_file, resolve_output_path, write_obj};
use bevy_synth_runtime::mesh::to_bevy_mesh;
use bevy_synth_runtime::state::{
    ExitState, InferenceQueue, InferenceRequest, InferenceWorker, Spinner, TitlePulse, UiStatus,
    WorkerCommand,
};
use bevy_synth_runtime::worker::start_worker;
#[cfg(not(target_arch = "wasm32"))]
use bevy_synth_ui::ImagePickDialog;
use bevy_synth_ui::{
    BurnSynthUiPlugin, CatalogSpawnRequest, CatalogState, CatalogStatus, CatalogUiState, DragState,
    MainCamera, preview_light_layers,
};

#[derive(Component, Clone, Debug)]
pub(crate) struct CachedMeshInstance {
    pub(crate) cache_key: String,
}

#[derive(Resource)]
pub(crate) struct MeshCacheResource {
    pub(crate) cache: MeshCache,
}

impl MeshCacheResource {
    fn load_or_empty() -> Self {
        match MeshCache::load_default() {
            Ok(cache) => Self { cache },
            Err(err) => {
                warn!("Mesh cache unavailable; continuing without persisted cache: {err}");
                Self {
                    cache: MeshCache::empty_default(),
                }
            }
        }
    }
}

#[derive(Resource)]
struct WorldCachePersistence {
    dirty: bool,
    timer: Timer,
}

impl Default for WorldCachePersistence {
    fn default() -> Self {
        Self {
            dirty: false,
            timer: Timer::from_seconds(0.35, TimerMode::Once),
        }
    }
}

#[derive(SystemParam)]
struct FileDropContext<'w, 's> {
    events: MessageReader<'w, 's, FileDragAndDrop>,
    queue: ResMut<'w, InferenceQueue>,
    args: Res<'w, AppArgs>,
    asset_server: Res<'w, AssetServer>,
    commands: Commands<'w, 's>,
    materials: ResMut<'w, Assets<StandardMaterial>>,
    status: ResMut<'w, UiStatus>,
    catalog: ResMut<'w, CatalogState>,
    exit_state: Res<'w, ExitState>,
}

#[derive(SystemParam)]
pub(crate) struct InferenceContext<'w, 's> {
    commands: Commands<'w, 's>,
    queue: ResMut<'w, InferenceQueue>,
    worker: Res<'w, InferenceWorker>,
    args: Res<'w, AppArgs>,
    meshes: ResMut<'w, Assets<BevyMesh>>,
    materials: ResMut<'w, Assets<StandardMaterial>>,
    cache: ResMut<'w, MeshCacheResource>,
    status: ResMut<'w, UiStatus>,
    catalog: ResMut<'w, CatalogState>,
    exit_state: Res<'w, ExitState>,
}

pub(crate) fn run() {
    if std::env::var("RUST_MIN_STACK").is_err() {
        unsafe {
            std::env::set_var("RUST_MIN_STACK", "67108864");
        }
    }
    let args = Args::parse();
    let app_args = build_app_args(args);

    let status_message = if app_args.image.is_none() && app_args.mesh.is_none() {
        "Drag & drop an image (.png/.jpg) or mesh (.glb/.gltf/.obj) to begin.".to_string()
    } else {
        "Initializing synth viewer…".to_string()
    };

    let mut app = App::new();
    app.insert_resource(app_args)
        .insert_resource(InferenceQueue::default())
        .insert_resource(ExitState::default())
        .insert_resource(TitlePulse::default())
        .init_resource::<EditorSelection>()
        .insert_resource(MeshCacheResource::load_or_empty())
        .insert_resource(WorldCachePersistence::default())
        .insert_resource(UiStatus {
            message: status_message,
            processing: false,
            worker_message: None,
        });
    add_default_plugins(&mut app);
    if !app.is_plugin_added::<PointerInputPlugin>() {
        app.add_plugins(DefaultPickingPlugins);
    }
    app.add_plugins(TransformGizmoPlugin)
        .add_plugins(PanOrbitCameraPlugin)
        .add_plugins(InfiniteGridPlugin)
        .add_plugins(BurnSynthUiPlugin)
        .add_systems(Startup, (setup, configure_mesh_picking).chain())
        .add_systems(
            Update,
            (
                handle_exit_requests,
                handle_file_drop,
                drive_inference,
                handle_catalog_spawn_requests,
                delete_selected_meshes,
                (mark_world_cache_dirty, persist_world_cache).chain(),
                update_spinner,
                rotate_spinner,
                update_window_title,
            ),
        )
        .add_systems(
            PostUpdate,
            (
                remove_entity_from_selection_if_despawned,
                update_selection_from_primary_click.after(TransformGizmoSystems::Main),
                sync_panorbit_bindings,
                sync_panorbit_enabled,
            )
                .before(PanOrbitCameraSystemSet),
        );

    #[cfg(not(target_arch = "wasm32"))]
    {
        app.add_plugins(FileDialogPlugin::new().with_pick_file::<ImagePickDialog>())
            .add_systems(Update, (handle_open_file_dialog, handle_file_dialog_picks));
    }

    app.run();
}

fn configure_mesh_picking(mut settings: ResMut<MeshPickingSettings>) {
    // Avoid ambiguous multi-camera rays by limiting picking to explicitly marked cameras/targets.
    settings.require_markers = true;
}

#[cfg(not(target_arch = "wasm32"))]
fn add_default_plugins(app: &mut App) {
    app.add_plugins(DefaultPlugins);
}

#[cfg(target_arch = "wasm32")]
fn add_default_plugins(app: &mut App) {
    let asset_root = web_asset_root();
    let asset_plugin = AssetPlugin {
        file_path: asset_root,
        mode: AssetMode::Unprocessed,
        watch_for_changes_override: Some(false),
        meta_check: AssetMetaCheck::Never,
        unapproved_path_mode: UnapprovedPathMode::Allow,
        ..default()
    };
    app.add_plugins(WebAssetPlugin {
        silence_startup_warning: true,
    });
    app.add_plugins(DefaultPlugins.set(asset_plugin));
}

#[cfg(target_arch = "wasm32")]
fn web_asset_root() -> String {
    if let Some(value) = option_env!("BURN_SYNTH_WEB_ASSET_ROOT") {
        let trimmed = value.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    "assets".to_string()
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<BevyMesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    args: Res<AppArgs>,
    mut queue: ResMut<InferenceQueue>,
    mut status: ResMut<UiStatus>,
    mut catalog: ResMut<CatalogState>,
    mut cache: ResMut<MeshCacheResource>,
    mut world_cache: ResMut<WorldCachePersistence>,
) {
    info!("bevy_synth args: {:?}", *args);

    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(Vec3::new(0.0, 1.5, 5.0)),
        PanOrbitCamera {
            allow_upside_down: true,
            orbit_smoothness: 0.1,
            pan_smoothness: 0.1,
            zoom_smoothness: 0.1,
            button_orbit: MouseButton::Right,
            button_pan: MouseButton::Middle,
            ..default()
        },
        GizmoCamera,
        MeshPickingCamera,
        RenderLayers::layer(0).with(12),
        MainCamera,
    ));
    commands.spawn((
        PointLight::default(),
        Transform::from_xyz(2.0, 3.0, 2.0),
        preview_light_layers(),
    ));

    commands.spawn((
        InfiniteGridBundle::default(),
        Pickable::IGNORE,
        RenderLayers::layer(0),
    ));

    commands.spawn((
        Camera2d,
        IsDefaultUiCamera,
        Camera {
            order: 10,
            clear_color: ClearColorConfig::None,
            ..default()
        },
    ));

    let worker = start_worker(args.as_ref());
    commands.insert_resource(worker);

    hydrate_from_cache(
        &mut commands,
        &mut meshes,
        &mut materials,
        &mut queue,
        &mut catalog,
        &mut cache,
    );
    world_cache.dirty = false;
    world_cache.timer.reset();

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
        let request = enqueue_inference(image_path.clone(), &args, &mut queue);
        catalog.add_pending(&request);
    }

    update_status_message(&args, &queue, &mut status);
}

fn hydrate_from_cache(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<BevyMesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    queue: &mut ResMut<InferenceQueue>,
    catalog: &mut ResMut<CatalogState>,
    cache: &mut ResMut<MeshCacheResource>,
) {
    let mesh_entries = cache.cache.mesh_entries().to_vec();
    let world_items = cache.cache.world_items().to_vec();
    if mesh_entries.is_empty() && world_items.is_empty() {
        return;
    }

    let mut loaded_meshes = 0usize;
    let mut loaded_world_items = 0usize;
    let mut handles_by_key: HashMap<String, (Handle<BevyMesh>, Handle<StandardMaterial>)> =
        HashMap::new();

    for metadata in mesh_entries {
        let mesh = match cache.cache.load_mesh(&metadata.cache_key) {
            Ok(Some(mesh)) => mesh,
            Ok(None) => {
                warn!(
                    "Cache metadata exists for key {} but mesh payload is missing.",
                    metadata.cache_key
                );
                continue;
            }
            Err(err) => {
                warn!(
                    "Failed to load cached mesh for key {}: {err}",
                    metadata.cache_key
                );
                continue;
            }
        };

        let mesh_handle = meshes.add(to_bevy_mesh(&mesh));
        let material = materials.add(StandardMaterial {
            base_color: Color::srgb(0.78, 0.84, 0.92),
            ..default()
        });
        handles_by_key.insert(
            metadata.cache_key.clone(),
            (mesh_handle.clone(), material.clone()),
        );

        let entry_id = queue.counter;
        queue.counter = queue.counter.wrapping_add(1);
        catalog.add_ready(
            entry_id,
            metadata.label.clone(),
            mesh_handle,
            material,
            Some(metadata.source_image_path.clone()),
            Some(metadata.cache_key.clone()),
        );
        loaded_meshes += 1;
    }

    for item in world_items {
        let Some((mesh, material)) = handles_by_key.get(&item.cache_key) else {
            warn!(
                "Skipping cached world item for unknown cache key {}",
                item.cache_key
            );
            continue;
        };
        let Some(transform) = transform_from_cached_world_item(&item) else {
            warn!(
                "Skipping cached world item with invalid transform for key {}",
                item.cache_key
            );
            continue;
        };
        spawn_mesh_instance(
            commands,
            mesh.clone(),
            material.clone(),
            transform,
            Some(item.cache_key.clone()),
        );
        loaded_world_items += 1;
    }

    if loaded_meshes > 0 {
        queue.completed = queue.completed.max(loaded_meshes);
    }
    if loaded_meshes > 0 || loaded_world_items > 0 {
        info!(
            "Loaded {loaded_meshes} cached catalog mesh(es) and {loaded_world_items} cached world item(s)."
        );
    }
}

fn transform_from_cached_world_item(item: &CachedWorldItem) -> Option<Transform> {
    let translation = Vec3::from_array(item.translation);
    let raw_scale = Vec3::from_array(item.scale);
    let raw_rotation = Quat::from_xyzw(
        item.rotation[0],
        item.rotation[1],
        item.rotation[2],
        item.rotation[3],
    );

    if !translation.is_finite() || !raw_scale.is_finite() || !raw_rotation.is_finite() {
        return None;
    }

    let scale = if raw_scale.length_squared() > 0.0 {
        raw_scale
    } else {
        Vec3::ONE
    };

    let rotation = if raw_rotation.length_squared() > 0.0 {
        raw_rotation.normalize()
    } else {
        Quat::IDENTITY
    };

    Some(Transform {
        translation,
        rotation,
        scale,
    })
}

fn handle_file_drop(mut ctx: FileDropContext) {
    if ctx.exit_state.requested {
        return;
    }
    for event in ctx.events.read() {
        let FileDragAndDrop::DroppedFile { path_buf, .. } = event else {
            continue;
        };

        if is_image_file(path_buf) {
            let request = enqueue_inference(path_buf.clone(), &ctx.args, &mut ctx.queue);
            ctx.catalog.add_pending(&request);
            info!("Queued inference for {}", path_buf.display());
            continue;
        }

        if is_mesh_file(path_buf) {
            spawn_mesh_asset(
                &mut ctx.commands,
                &ctx.asset_server,
                &mut ctx.materials,
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

    update_status_message(&ctx.args, &ctx.queue, &mut ctx.status);
}

#[cfg(not(target_arch = "wasm32"))]
fn handle_open_file_dialog(
    keys: Res<ButtonInput<KeyCode>>,
    mut commands: Commands,
    exit_state: Res<ExitState>,
) {
    if exit_state.requested {
        return;
    }
    let ctrl = keys.any_pressed([KeyCode::ControlLeft, KeyCode::ControlRight]);
    let os = keys.any_pressed([KeyCode::SuperLeft, KeyCode::SuperRight]);
    if (ctrl || os) && keys.just_pressed(KeyCode::KeyO) {
        commands
            .dialog()
            .set_title("Open Image")
            .add_filter(
                "Images",
                &[
                    "png", "jpg", "jpeg", "bmp", "gif", "webp", "tga", "tif", "tiff",
                ],
            )
            .pick_multiple_file_paths::<ImagePickDialog>();
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn handle_file_dialog_picks(
    mut events: MessageReader<DialogFilePicked<ImagePickDialog>>,
    mut queue: ResMut<InferenceQueue>,
    args: Res<AppArgs>,
    mut status: ResMut<UiStatus>,
    mut catalog: ResMut<CatalogState>,
    exit_state: Res<ExitState>,
) {
    if exit_state.requested {
        return;
    }
    let mut queued = 0usize;
    for event in events.read() {
        if is_image_file(&event.path) {
            let request = enqueue_inference(event.path.clone(), &args, &mut queue);
            catalog.add_pending(&request);
            queued += 1;
        } else {
            warn!(
                "Selected file {} is not a supported image",
                event.path.display()
            );
        }
    }

    if queued > 0 {
        update_status_message(&args, &queue, &mut status);
    }
}

pub(crate) fn drive_inference(mut ctx: InferenceContext) {
    if ctx.exit_state.requested {
        return;
    }
    let receiver = match ctx.worker.receiver.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    loop {
        match receiver.try_recv() {
            Ok(event) => {
                if let Some(message) = event.status_message {
                    ctx.status.worker_message = Some(message.clone());
                    ctx.status.message = message;
                }
                if event.requests.is_empty() && event.results.is_empty() {
                    continue;
                }
                ctx.status.worker_message = None;
                ctx.queue.active = None;
                if event.requests.len() != event.results.len() {
                    warn!(
                        "Inference batch mismatch: {} requests, {} results",
                        event.requests.len(),
                        event.results.len()
                    );
                }
                ctx.queue.completed += event.requests.len();
                if event.requests.len() == 1 {
                    if let Some(request) = event.requests.first() {
                        info!(
                            "Inference completed in {:.2}s for {}",
                            event.elapsed.as_secs_f32(),
                            request.image_path.display()
                        );
                    }
                } else {
                    info!(
                        "Inference completed in {:.2}s for {} images",
                        event.elapsed.as_secs_f32(),
                        event.requests.len()
                    );
                }
                for (request, result) in event.requests.into_iter().zip(event.results.into_iter()) {
                    handle_inference_result(
                        &mut ctx.commands,
                        &mut ctx.meshes,
                        &mut ctx.materials,
                        &mut ctx.cache,
                        &mut ctx.catalog,
                        request,
                        result,
                    );
                }
            }
            Err(TryRecvError::Empty) => break,
            Err(TryRecvError::Disconnected) => {
                warn!("Inference worker disconnected; pending requests will be skipped.");
                ctx.queue.active = None;
                ctx.queue.pending.clear();
                break;
            }
        }
    }

    if ctx.queue.active.is_none() && !ctx.queue.pending.is_empty() {
        if ctx.status.worker_message.is_some() {
            ctx.status.worker_message = None;
        }
        let max_batch = ctx.args.max_batch_size.max(1);
        let mut batch = Vec::with_capacity(max_batch);
        while batch.len() < max_batch {
            if let Some(request) = ctx.queue.pending.pop_front() {
                batch.push(request);
            } else {
                break;
            }
        }
        if batch.is_empty() {
            update_status_message(&ctx.args, &ctx.queue, &mut ctx.status);
            return;
        }
        if let Err(err) = ctx.worker.sender.send(WorkerCommand::Infer(batch.clone())) {
            warn!("Failed to send inference request: {err}");
        } else {
            ctx.queue.active = Some(batch);
        }
    }

    update_status_message(&ctx.args, &ctx.queue, &mut ctx.status);
}

fn handle_catalog_spawn_requests(
    mut requests: MessageReader<CatalogSpawnRequest>,
    mut commands: Commands,
    mut selection: Option<ResMut<EditorSelection>>,
) {
    for request in requests.read() {
        let entity = spawn_mesh_instance(
            &mut commands,
            request.mesh.clone(),
            request.material.clone(),
            request.transform,
            request.cache_key.clone(),
        );
        if request.select_spawned
            && let Some(selection) = selection.as_mut()
        {
            selection.set(entity);
        }
    }
}

fn sync_panorbit_bindings(mut cameras: Query<&mut PanOrbitCamera>) {
    for mut camera in cameras.iter_mut() {
        if camera.button_orbit != MouseButton::Right {
            camera.button_orbit = MouseButton::Right;
        }
        if camera.button_pan != MouseButton::Middle {
            camera.button_pan = MouseButton::Middle;
        }
    }
}

fn sync_panorbit_enabled(
    gizmos: Query<&TransformGizmo>,
    gizmo_handles_hover: Query<&PickingInteraction, With<bevy_transform_gizmos::InteractionKind>>,
    drag: Res<DragState>,
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    ui_state: Res<CatalogUiState>,
    mut cameras: Query<&mut PanOrbitCamera>,
) {
    let gizmo_active = gizmos.iter().any(|gizmo| gizmo.interaction().is_some());
    let gizmo_handle_pressed = buttons.pressed(MouseButton::Left)
        && gizmo_handles_hover
            .iter()
            .any(|interaction| *interaction != PickingInteraction::None);
    let ui_block = windows
        .single()
        .ok()
        .map(|window| ui_state.cursor_over_ui(window) && buttons.pressed(MouseButton::Left))
        .unwrap_or(false);
    let enabled = !gizmo_active && !gizmo_handle_pressed && !drag.is_dragging() && !ui_block;
    for mut camera in cameras.iter_mut() {
        if !enabled {
            camera.target_focus = camera.focus;
            if let Some(yaw) = camera.yaw {
                camera.target_yaw = yaw;
            }
            if let Some(pitch) = camera.pitch {
                camera.target_pitch = pitch;
            }
            if let Some(radius) = camera.radius {
                camera.target_radius = radius;
            }
        }
        camera.enabled = enabled;
    }
}

fn update_spinner(queue: Res<InferenceQueue>, mut query: Query<&mut Visibility, With<Spinner>>) {
    let visible = queue
        .active
        .as_ref()
        .map(|batch| !batch.is_empty())
        .unwrap_or(false);
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
    if queue
        .active
        .as_ref()
        .map(|batch| batch.is_empty())
        .unwrap_or(true)
    {
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
    let processing = queue
        .active
        .as_ref()
        .map(|batch| !batch.is_empty())
        .unwrap_or(false);
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
        let name = if active.len() == 1 {
            active[0]
                .image_path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("image")
                .to_string()
        } else {
            format!("{} images", active.len())
        };
        format!(
            "burn_synth — processing: {name} (queued: {}){dots}",
            queue.pending.len()
        )
    } else {
        format!("burn_synth — {}", status.message)
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
    worker: Res<InferenceWorker>,
    mut cache: ResMut<MeshCacheResource>,
    cached_instances: Query<(&CachedMeshInstance, &Transform)>,
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
    if let Err(err) = cache
        .cache
        .set_world_items(collect_cached_world_items(&cached_instances))
    {
        warn!("Failed to flush world cache during shutdown: {err}");
    }
    queue.active = None;
    queue.pending.clear();
    status.processing = false;
    status.worker_message = None;
    status.message = "Shutting down…".to_string();
    info!("{}", status.message);
    let _ = worker.sender.send(WorkerCommand::Shutdown);
    exit.write(AppExit::Success);
}

fn handle_inference_result(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<BevyMesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    cache: &mut ResMut<MeshCacheResource>,
    catalog: &mut ResMut<CatalogState>,
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

            let cached_metadata = match cache
                .cache
                .upsert_mesh_for_image(&request.image_path, &mesh)
            {
                Ok(metadata) => Some(metadata),
                Err(err) => {
                    warn!(
                        "Failed to cache mesh output for {}: {err}",
                        request.image_path.display()
                    );
                    None
                }
            };
            let cache_key = cached_metadata
                .as_ref()
                .map(|metadata| metadata.cache_key.clone());

            let bevy_mesh = to_bevy_mesh(&mesh);
            let mesh_handle = meshes.add(bevy_mesh);
            let material = materials.add(StandardMaterial {
                base_color: Color::srgb(0.78, 0.84, 0.92),
                ..default()
            });
            spawn_mesh_instance(
                commands,
                mesh_handle.clone(),
                material.clone(),
                Transform::default(),
                cache_key.clone(),
            );
            if let Some(entry) = catalog.entry_mut(request.id) {
                entry.status = CatalogStatus::Ready;
                entry.mesh = Some(mesh_handle);
                entry.material = Some(material);
                entry.source_image_path = Some(request.image_path.display().to_string());
                entry.cache_key = cache_key;
                if let Some(metadata) = cached_metadata {
                    entry.label = metadata.label;
                    entry.source_image_path = Some(metadata.source_image_path);
                }
                catalog.bump_revision();
            }
        }
        Ok(None) => {
            warn!(
                "Synthesis inference produced an empty mesh for {}",
                request.image_path.display()
            );
            if let Some(entry) = catalog.entry_mut(request.id) {
                entry.status = CatalogStatus::Failed("empty mesh".to_string());
                catalog.bump_revision();
            }
        }
        Err(err) => {
            warn!(
                "Synthesis inference failed for {}: {}",
                request.image_path.display(),
                err
            );
            if let Some(entry) = catalog.entry_mut(request.id) {
                entry.status = CatalogStatus::Failed(err);
                catalog.bump_revision();
            }
        }
    }
}

pub(crate) fn enqueue_inference(
    image_path: PathBuf,
    args: &AppArgs,
    queue: &mut InferenceQueue,
) -> InferenceRequest {
    let output_path = resolve_output_path(args.output.as_ref(), &image_path, queue.counter);
    let request = InferenceRequest {
        id: queue.counter,
        image_path,
        output_path,
    };
    queue.counter = queue.counter.wrapping_add(1);
    queue.pending.push_back(request.clone());
    request
}

fn update_status_message(args: &AppArgs, queue: &InferenceQueue, status: &mut UiStatus) {
    if let Some(message) = status.worker_message.clone() {
        status.processing = true;
        if status.message != message {
            status.message = message;
            info!("{}", status.message);
        }
        return;
    }

    let message = if let Some(active) = queue.active.as_ref() {
        let label = if active.len() == 1 {
            active[0].image_path.display().to_string()
        } else {
            format!("{} images", active.len())
        };
        format!("processing {label} ({} queued)", queue.pending.len())
    } else if !queue.pending.is_empty() {
        format!("Queued {} inference job(s)…", queue.pending.len())
    } else if args.image.is_none() && args.mesh.is_none() && queue.completed == 0 {
        "Drag & drop an image (.png/.jpg) or mesh (.glb/.gltf/.obj) to begin.".to_string()
    } else {
        "Ready. Drag & drop another image or mesh to add more.".to_string()
    };

    status.processing = queue
        .active
        .as_ref()
        .map(|batch| !batch.is_empty())
        .unwrap_or(false);
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
    spawn_mesh_instance(commands, mesh_handle, material, Transform::default(), None);
}

pub(crate) fn spawn_mesh_instance(
    commands: &mut Commands,
    mesh_handle: Handle<BevyMesh>,
    material: Handle<StandardMaterial>,
    transform: Transform,
    cache_key: Option<String>,
) -> Entity {
    let mut entity_commands = commands.spawn((
        GizmoTransformable,
        Selectable,
        Pickable {
            should_block_lower: false,
            is_hoverable: true,
        },
        Mesh3d(mesh_handle),
        MeshMaterial3d(material),
        transform,
        RenderLayers::layer(0),
    ));
    if let Some(cache_key) = cache_key {
        entity_commands.insert(CachedMeshInstance { cache_key });
    }
    entity_commands.id()
}

fn mark_world_cache_dirty(
    mut persistence: ResMut<WorldCachePersistence>,
    changed: Query<
        (),
        (
            With<CachedMeshInstance>,
            Or<(Added<CachedMeshInstance>, Changed<Transform>)>,
        ),
    >,
    mut removed: RemovedComponents<CachedMeshInstance>,
) {
    if changed.is_empty() && removed.read().next().is_none() {
        return;
    }
    persistence.dirty = true;
    persistence.timer.reset();
}

fn persist_world_cache(
    time: Res<Time>,
    mut persistence: ResMut<WorldCachePersistence>,
    mut cache: ResMut<MeshCacheResource>,
    query: Query<(&CachedMeshInstance, &Transform)>,
) {
    if !persistence.dirty {
        return;
    }
    persistence.timer.tick(time.delta());
    if !persistence.timer.is_finished() {
        return;
    }

    match cache
        .cache
        .set_world_items(collect_cached_world_items(&query))
    {
        Ok(()) => {
            persistence.dirty = false;
        }
        Err(err) => {
            warn!("Failed to persist world cache items: {err}");
            persistence.timer.reset();
        }
    }
}

fn collect_cached_world_items(
    query: &Query<(&CachedMeshInstance, &Transform)>,
) -> Vec<CachedWorldItem> {
    let mut world_items = Vec::new();
    for (cached, transform) in query.iter() {
        let rotation = if transform.rotation.length_squared() > 0.0 {
            transform.rotation.normalize()
        } else {
            Quat::IDENTITY
        };
        world_items.push(CachedWorldItem {
            cache_key: cached.cache_key.clone(),
            translation: transform.translation.to_array(),
            rotation: rotation.to_array(),
            scale: transform.scale.to_array(),
        });
    }
    world_items
}

fn delete_selected_meshes(
    keys: Res<ButtonInput<KeyCode>>,
    mut selection: ResMut<EditorSelection>,
    transformables: Query<(), With<GizmoTransformable>>,
    mut commands: Commands,
) {
    if !keys.just_pressed(KeyCode::Delete) && !keys.just_pressed(KeyCode::Backspace) {
        return;
    }

    let to_despawn: Vec<Entity> = selection
        .iter()
        .filter(|entity| transformables.contains(*entity))
        .collect();
    if to_despawn.is_empty() {
        return;
    }

    for entity in to_despawn {
        commands.entity(entity).despawn();
    }
    selection.clear();
}

fn update_selection_from_primary_click(
    buttons: Res<ButtonInput<MouseButton>>,
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut selection: ResMut<EditorSelection>,
    windows: Query<&Window, With<PrimaryWindow>>,
    ui_state: Res<CatalogUiState>,
    gizmo_handles_hover: Query<&PickingInteraction, With<bevy_transform_gizmos::InteractionKind>>,
    gizmos: Query<&TransformGizmo>,
    cameras: Query<(&Camera, &GlobalTransform), With<MainCamera>>,
    gizmo_handles_meshes: Query<
        (&Mesh3d, &GlobalTransform),
        With<bevy_transform_gizmos::InteractionKind>,
    >,
    transformables: Query<(Entity, &Mesh3d, &GlobalTransform), With<GizmoTransformable>>,
    meshes: Res<Assets<BevyMesh>>,
) {
    if !buttons.just_pressed(MouseButton::Left) {
        return;
    }
    let Ok(window) = windows.single() else {
        return;
    };
    if ui_state.cursor_over_ui(window) {
        return;
    }
    if gizmos.iter().any(|gizmo| gizmo.interaction().is_some()) {
        return;
    }
    if gizmo_handles_hover
        .iter()
        .any(|interaction| *interaction != PickingInteraction::None)
    {
        return;
    }

    let Some(cursor) = window.cursor_position() else {
        return;
    };
    let Ok((camera, camera_transform)) = cameras.single() else {
        return;
    };
    let Ok(ray) = camera.viewport_to_world(camera_transform, cursor) else {
        return;
    };
    // Never alter world selection when clicking gizmo handles.
    let on_gizmo_handle = gizmo_handles_meshes.iter().any(|(mesh3d, transform)| {
        let Some(mesh) = meshes.get(&mesh3d.0) else {
            return false;
        };
        let Some(aabb) = mesh.compute_aabb() else {
            return false;
        };
        let (world_min, world_max) =
            world_aabb(aabb.center.into(), aabb.half_extents.into(), transform);
        ray_aabb_intersection(ray.origin, ray.direction.as_vec3(), world_min, world_max).is_some()
    });
    if on_gizmo_handle {
        return;
    }

    let mut best_hit: Option<(Entity, f32)> = None;
    for (entity, mesh3d, transform) in transformables.iter() {
        let Some(mesh) = meshes.get(&mesh3d.0) else {
            continue;
        };
        let Some(aabb) = mesh.compute_aabb() else {
            continue;
        };
        let (world_min, world_max) =
            world_aabb(aabb.center.into(), aabb.half_extents.into(), transform);
        let Some(distance) =
            ray_aabb_intersection(ray.origin, ray.direction.as_vec3(), world_min, world_max)
        else {
            continue;
        };
        if best_hit
            .as_ref()
            .map(|(_, best_distance)| distance < *best_distance)
            .unwrap_or(true)
        {
            best_hit = Some((entity, distance));
        }
    }

    if let Some((entity, _)) = best_hit {
        if keyboard_input.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight]) {
            selection.toggle(entity);
            return;
        }
        if selection.primary() == Some(entity) && selection.iter().count() == 1 {
            return;
        }
        selection.set(entity);
    } else if !keyboard_input.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight]) {
        selection.clear();
    }
}

fn world_aabb(
    local_center: Vec3,
    local_half_extents: Vec3,
    transform: &GlobalTransform,
) -> (Vec3, Vec3) {
    let local_min = local_center - local_half_extents;
    let local_max = local_center + local_half_extents;
    let mut world_min = Vec3::splat(f32::INFINITY);
    let mut world_max = Vec3::splat(f32::NEG_INFINITY);
    let world_from_local = transform.to_matrix();
    for &x in &[local_min.x, local_max.x] {
        for &y in &[local_min.y, local_max.y] {
            for &z in &[local_min.z, local_max.z] {
                let point = world_from_local.transform_point3(Vec3::new(x, y, z));
                world_min = world_min.min(point);
                world_max = world_max.max(point);
            }
        }
    }
    (world_min, world_max)
}

fn ray_aabb_intersection(origin: Vec3, direction: Vec3, min: Vec3, max: Vec3) -> Option<f32> {
    let mut t_min: f32 = 0.0;
    let mut t_max: f32 = f32::INFINITY;
    for (origin_axis, direction_axis, min_axis, max_axis) in [
        (origin.x, direction.x, min.x, max.x),
        (origin.y, direction.y, min.y, max.y),
        (origin.z, direction.z, min.z, max.z),
    ] {
        if direction_axis.abs() < f32::EPSILON {
            if origin_axis < min_axis || origin_axis > max_axis {
                return None;
            }
            continue;
        }
        let inv_direction = 1.0 / direction_axis;
        let mut t1 = (min_axis - origin_axis) * inv_direction;
        let mut t2 = (max_axis - origin_axis) * inv_direction;
        if t1 > t2 {
            std::mem::swap(&mut t1, &mut t2);
        }
        t_min = t_min.max(t1);
        t_max = t_max.min(t2);
        if t_max < t_min {
            return None;
        }
    }
    Some(t_min.max(0.0))
}

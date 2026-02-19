use std::collections::HashMap;
use std::path::{Path, PathBuf};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc::RecvTimeoutError;
use std::sync::mpsc::TryRecvError;
#[cfg(not(target_arch = "wasm32"))]
use std::time::SystemTime;
#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};
#[cfg(not(target_arch = "wasm32"))]
use std::{fs, io};

use bevy::app::AppExit;
use bevy::asset::RenderAssetUsages;
#[cfg(target_arch = "wasm32")]
use bevy::asset::io::web::WebAssetPlugin;
#[cfg(target_arch = "wasm32")]
use bevy::asset::{AssetMetaCheck, AssetMode, AssetPlugin, UnapprovedPathMode};
#[cfg(not(target_arch = "wasm32"))]
use bevy::camera::primitives::MeshAabb;
use bevy::camera::visibility::RenderLayers;
use bevy::ecs::message::{MessageReader, MessageWriter};
use bevy::ecs::system::SystemParam;
use bevy::light::{
    AmbientLight, CascadeShadowConfigBuilder, DirectionalLight, DirectionalLightShadowMap,
    light_consts::lux,
};
use bevy::math::primitives::Cuboid;
use bevy::pbr::{MeshMaterial3d, StandardMaterial};
use bevy::prelude::*;
#[cfg(all(not(target_arch = "wasm32"), feature = "wgpu"))]
use bevy::render::RenderApp;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
#[cfg(all(not(target_arch = "wasm32"), feature = "wgpu"))]
use bevy::render::renderer::{
    RenderAdapter, RenderAdapterInfo, RenderDevice, RenderInstance, RenderQueue,
};
use bevy::window::{PrimaryWindow, WindowCloseRequested};
#[cfg(not(target_arch = "wasm32"))]
use bevy::winit::{EventLoopProxy, EventLoopProxyWrapper, UpdateMode, WakeUp, WinitSettings};
use bevy_editor_core::selection::{
    EditorSelection, Selectable, remove_entity_from_selection_if_despawned,
};
use bevy_file_dialog::prelude::{
    DialogFileDropped, DialogFileLoaded, FileDialogExt, FileDialogPlugin,
};
use bevy_infinite_grid::{InfiniteGridBundle, InfiniteGridPlugin, InfiniteGridSettings};
use bevy_mesh::{Mesh as BevyMesh, Mesh3d};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin, PanOrbitCameraSystemSet};
#[cfg(not(target_arch = "wasm32"))]
use bevy_picking::DefaultPickingPlugins;
use bevy_picking::hover::PickingInteraction;
#[cfg(not(target_arch = "wasm32"))]
use bevy_picking::input::PointerInputPlugin;
#[cfg(not(target_arch = "wasm32"))]
use bevy_picking::prelude::MeshPickingSettings;
use bevy_picking::prelude::{MeshPickingCamera, Pickable};
#[cfg(not(target_arch = "wasm32"))]
use bevy_transform_gizmos::TransformGizmoSystems;
use bevy_transform_gizmos::prelude::GizmoCamera;
#[cfg(not(target_arch = "wasm32"))]
use bevy_transform_gizmos::prelude::TransformGizmoPlugin;
use bevy_transform_gizmos::{GizmoTransformable, TransformGizmo};
use clap::Parser;
#[cfg(not(target_arch = "wasm32"))]
use serde::Deserialize;

#[cfg(not(target_arch = "wasm32"))]
use bevy_synth_runtime::args::BackendKind;
#[cfg(all(not(target_arch = "wasm32"), feature = "wgpu"))]
use bevy_synth_runtime::args::MeshMode;
#[cfg(target_arch = "wasm32")]
use bevy_synth_runtime::args::WeightPrecision;
use bevy_synth_runtime::args::{AppArgs, Args, build_app_args};
use bevy_synth_runtime::cache::{CachedCameraState, CachedWorldItem, MeshCache};
use bevy_synth_runtime::io::{is_image_file, is_mesh_file, resolve_output_path, write_glb};
use bevy_synth_runtime::mesh::to_bevy_mesh_synth;
use bevy_synth_runtime::state::{
    ExitState, InferenceQueue, InferenceRequest, InferenceWorker, Spinner, TitlePulse, UiStatus,
    WorkerCommand,
};
#[cfg(target_arch = "wasm32")]
use bevy_synth_runtime::state::{
    WASM_STATUS_LOADING_MODELS, WASM_STATUS_MODEL_LOAD_FAILED_PREFIX, WASM_STATUS_MODEL_READY,
};
use bevy_synth_runtime::worker::start_worker;
#[cfg(all(not(target_arch = "wasm32"), feature = "wgpu"))]
use bevy_synth_runtime::worker::{
    SharedWgpuDevice, WorkerWakeCallback, init_shared_wgpu_device_from_bevy_render,
    start_worker_with_shared_wgpu_device_and_wake,
};
#[cfg(all(not(target_arch = "wasm32"), not(feature = "wgpu")))]
use bevy_synth_runtime::worker::{WorkerWakeCallback, start_worker_with_wake};
use bevy_synth_runtime::{SynthMesh, SynthMeshTexture};
use bevy_synth_ui::ImagePickDialog;
use bevy_synth_ui::{
    BurnSynthUiPlugin, CatalogDeleteRequest, CatalogSpawnRequest, CatalogState, CatalogStatus,
    CatalogUiState, DragState, MainCamera, preview_light_layers,
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
                warn!("mesh cache unavailable; continuing without persisted cache: {err}");
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

#[cfg(not(target_arch = "wasm32"))]
#[derive(Resource, Default)]
struct McpSceneControl {
    path: Option<PathBuf>,
    last_modified: Option<SystemTime>,
}

#[cfg(target_arch = "wasm32")]
#[derive(Resource, Default)]
struct WasmStartupGate {
    model_ready: bool,
    model_failed: bool,
    scene_initialized: bool,
}

#[cfg(target_arch = "wasm32")]
#[derive(Resource)]
struct WasmWarmupKickoff {
    sent: bool,
    timer: Timer,
}

#[cfg(target_arch = "wasm32")]
impl Default for WasmWarmupKickoff {
    fn default() -> Self {
        Self {
            sent: false,
            timer: Timer::from_seconds(1.5, TimerMode::Once),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl McpSceneControl {
    fn from_args(args: &AppArgs) -> Self {
        Self {
            path: args.mcp_scene_control_path.clone(),
            last_modified: None,
        }
    }
}

#[cfg(all(not(target_arch = "wasm32"), feature = "wgpu"))]
#[derive(Resource, Default, Clone)]
struct SharedWgpuInferenceDevice {
    device: Option<SharedWgpuDevice>,
}

#[cfg(all(not(target_arch = "wasm32"), feature = "wgpu"))]
struct SharedWgpuInferenceDevicePlugin;

#[cfg(all(not(target_arch = "wasm32"), feature = "wgpu"))]
impl Plugin for SharedWgpuInferenceDevicePlugin {
    fn build(&self, _app: &mut App) {}

    fn finish(&self, app: &mut App) {
        let wants_shared_wgpu = app
            .world()
            .get_resource::<AppArgs>()
            .map(should_share_wgpu_inference_device)
            .unwrap_or(false);
        if !wants_shared_wgpu {
            return;
        }

        let shared_device = {
            let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
                warn!(
                    "RenderApp unavailable; Burn WGPU inference will use an isolated runtime device."
                );
                return;
            };
            let world = render_app.world();
            Some(init_shared_wgpu_device_from_bevy_render(
                world.resource::<RenderAdapter>(),
                world.resource::<RenderAdapterInfo>(),
                world.resource::<RenderDevice>(),
                world.resource::<RenderInstance>(),
                world.resource::<RenderQueue>(),
            ))
        };

        if shared_device.is_some() {
            info!("Initialized shared Burn WGPU device from Bevy render context.");
        }
        app.insert_resource(SharedWgpuInferenceDevice {
            device: shared_device,
        });
    }
}

#[cfg(not(target_arch = "wasm32"))]
const INFERENCE_PAUSE_WAIT: Duration = Duration::from_secs(60 * 60);

#[cfg(not(target_arch = "wasm32"))]
#[derive(Component)]
struct InferencePauseOverlay;

#[cfg(not(target_arch = "wasm32"))]
#[derive(Resource, Debug, Default, Clone)]
pub(crate) struct InferenceRenderPauseState {
    pub(crate) applied: bool,
    pub(crate) pending_apply: bool,
    pub(crate) overlay_visible: bool,
    pub(crate) saved_settings: Option<WinitSettings>,
}

#[cfg(not(target_arch = "wasm32"))]
fn paused_winit_settings() -> WinitSettings {
    let mode = UpdateMode::Reactive {
        wait: INFERENCE_PAUSE_WAIT,
        react_to_device_events: false,
        react_to_user_events: true,
        react_to_window_events: false,
    };
    WinitSettings {
        focused_mode: mode,
        unfocused_mode: mode,
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn should_pause_render_during_inference(
    args: &AppArgs,
    queue: &InferenceQueue,
    exit_requested: bool,
) -> bool {
    if !args.pause_render_during_inference || exit_requested {
        return false;
    }
    let active = queue
        .active
        .as_ref()
        .map(|batch| !batch.is_empty())
        .unwrap_or(false);
    active || !queue.pending.is_empty()
}

#[cfg(not(target_arch = "wasm32"))]
fn sync_inference_render_pause(
    args: Res<AppArgs>,
    queue: Res<InferenceQueue>,
    exit_state: Res<ExitState>,
    mut pause_state: ResMut<InferenceRenderPauseState>,
    winit_settings: Option<ResMut<WinitSettings>>,
    event_loop_proxy: Option<Res<EventLoopProxyWrapper<WakeUp>>>,
    mut overlays: Query<&mut Visibility, With<InferencePauseOverlay>>,
) {
    let mut request_wakeup = false;
    let should_pause = should_pause_render_during_inference(&args, &queue, exit_state.requested);
    let Some(mut winit_settings) = winit_settings else {
        pause_state.applied = false;
        pause_state.pending_apply = false;
        pause_state.overlay_visible = false;
        pause_state.saved_settings = None;
        set_inference_pause_overlay_visibility(false, &mut overlays);
        return;
    };

    if should_pause {
        if !pause_state.overlay_visible {
            pause_state.overlay_visible = true;
            set_inference_pause_overlay_visibility(true, &mut overlays);
        }

        if !pause_state.applied {
            if !pause_state.pending_apply {
                // Show the overlay first, then freeze on the next update.
                pause_state.pending_apply = true;
                request_wakeup = true;
            } else {
                pause_state.saved_settings = Some(winit_settings.clone());
                *winit_settings = paused_winit_settings();
                pause_state.applied = true;
                pause_state.pending_apply = false;
            }
        } else {
            pause_state.pending_apply = false;
        }
    } else {
        pause_state.pending_apply = false;
        if let Some(saved) = pause_state.saved_settings.take() {
            *winit_settings = saved;
            request_wakeup = true;
        }
        pause_state.applied = false;
        if pause_state.overlay_visible {
            pause_state.overlay_visible = false;
            set_inference_pause_overlay_visibility(false, &mut overlays);
        }
    }

    if request_wakeup && let Some(proxy) = event_loop_proxy.as_ref() {
        let _ = proxy.send_event(WakeUp);
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn set_inference_pause_overlay_visibility(
    visible: bool,
    overlays: &mut Query<&mut Visibility, With<InferencePauseOverlay>>,
) {
    let next = if visible {
        Visibility::Visible
    } else {
        Visibility::Hidden
    };
    for mut visibility in overlays.iter_mut() {
        *visibility = next;
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn sync_inference_render_pause_before(
    args: Res<AppArgs>,
    queue: Res<InferenceQueue>,
    exit_state: Res<ExitState>,
    pause_state: ResMut<InferenceRenderPauseState>,
    winit_settings: Option<ResMut<WinitSettings>>,
    event_loop_proxy: Option<Res<EventLoopProxyWrapper<WakeUp>>>,
    overlays: Query<&mut Visibility, With<InferencePauseOverlay>>,
) {
    sync_inference_render_pause(
        args,
        queue,
        exit_state,
        pause_state,
        winit_settings,
        event_loop_proxy,
        overlays,
    );
}

#[cfg(not(target_arch = "wasm32"))]
fn sync_inference_render_pause_after(
    args: Res<AppArgs>,
    queue: Res<InferenceQueue>,
    exit_state: Res<ExitState>,
    pause_state: ResMut<InferenceRenderPauseState>,
    winit_settings: Option<ResMut<WinitSettings>>,
    event_loop_proxy: Option<Res<EventLoopProxyWrapper<WakeUp>>>,
    overlays: Query<&mut Visibility, With<InferencePauseOverlay>>,
) {
    sync_inference_render_pause(
        args,
        queue,
        exit_state,
        pause_state,
        winit_settings,
        event_loop_proxy,
        overlays,
    );
}

#[cfg(not(target_arch = "wasm32"))]
fn make_worker_wake_callback(
    args: &AppArgs,
    event_loop_proxy: Option<&Res<EventLoopProxyWrapper<WakeUp>>>,
) -> Option<WorkerWakeCallback> {
    if !args.pause_render_during_inference {
        return None;
    }
    let proxy = event_loop_proxy.map(|proxy| EventLoopProxy::clone(&**proxy))?;
    Some(std::sync::Arc::new(move || {
        let _ = proxy.send_event(WakeUp);
    }))
}

#[cfg(all(not(target_arch = "wasm32"), feature = "wgpu"))]
pub(crate) fn should_share_wgpu_inference_device(args: &AppArgs) -> bool {
    should_share_wgpu_inference_device_for_platform(args, cfg!(target_os = "linux"))
}

#[cfg(all(not(target_arch = "wasm32"), feature = "wgpu"))]
pub(crate) fn should_share_wgpu_inference_device_for_platform(
    args: &AppArgs,
    is_linux: bool,
) -> bool {
    if !matches!(args.backend, BackendKind::Wgpu) {
        return false;
    }

    let full_flash_workload = matches!(args.mesh_mode, MeshMode::Flash)
        && args.flash_octree_depth >= 9
        && args.flash_min_resolution >= 63;
    if is_linux && full_flash_workload {
        info!(
            "Using isolated Burn WGPU device for Linux full+flash workload \
             (octree_depth={}, min_resolution={}) to avoid render swapchain instability.",
            args.flash_octree_depth, args.flash_min_resolution
        );
        return false;
    }

    true
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Deserialize)]
struct McpSceneCommandEnvelope {
    commands: Vec<McpSceneCommand>,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum McpSceneCommand {
    SpawnCached {
        cache_key: String,
        #[serde(default)]
        translation: Option<[f32; 3]>,
        #[serde(default)]
        rotation: Option<[f32; 4]>,
        #[serde(default)]
        scale: Option<[f32; 3]>,
        #[serde(default)]
        select: bool,
    },
    DeleteByCacheKey {
        cache_key: String,
    },
    DeleteSelected,
    ClearSelection,
    SetCamera {
        translation: [f32; 3],
        rotation: [f32; 4],
        #[serde(default)]
        focus: Option<[f32; 3]>,
        #[serde(default)]
        yaw: Option<f32>,
        #[serde(default)]
        pitch: Option<f32>,
        #[serde(default)]
        radius: Option<f32>,
    },
    SaveCache,
}

#[derive(SystemParam)]
pub(crate) struct InferenceContext<'w, 's> {
    commands: Commands<'w, 's>,
    queue: ResMut<'w, InferenceQueue>,
    worker: Res<'w, InferenceWorker>,
    args: Res<'w, AppArgs>,
    meshes: ResMut<'w, Assets<BevyMesh>>,
    images: ResMut<'w, Assets<Image>>,
    materials: ResMut<'w, Assets<StandardMaterial>>,
    cache: ResMut<'w, MeshCacheResource>,
    status: ResMut<'w, UiStatus>,
    catalog: ResMut<'w, CatalogState>,
    exit_state: Res<'w, ExitState>,
    #[cfg(target_arch = "wasm32")]
    wasm_startup: ResMut<'w, WasmStartupGate>,
}

pub(crate) fn run() {
    let args = Args::parse();
    #[cfg(target_arch = "wasm32")]
    let mut app_args = build_app_args(args);
    #[cfg(not(target_arch = "wasm32"))]
    let app_args = build_app_args(args);
    #[cfg(target_arch = "wasm32")]
    apply_wasm_url_overrides(&mut app_args);
    #[cfg(not(target_arch = "wasm32"))]
    if should_run_headless_once(&app_args) {
        if let Err(err) = run_headless_once(&app_args) {
            eprintln!("headless synthesis failed: {err}");
            std::process::exit(1);
        }
        return;
    }
    #[cfg(not(target_arch = "wasm32"))]
    let mcp_scene_control = McpSceneControl::from_args(&app_args);

    let status_message = if app_args.image.is_none() && app_args.mesh.is_none() {
        "upload an image (.png/.jpg) to begin.".to_string()
    } else {
        "initializing viewer…".to_string()
    };

    let mut app = App::new();
    app.insert_resource(app_args)
        .insert_resource(InferenceQueue::default())
        .insert_resource(ExitState::default())
        .insert_resource(TitlePulse::default())
        .insert_resource(DirectionalLightShadowMap { size: 4096 })
        .init_resource::<EditorSelection>()
        .insert_resource(MeshCacheResource::load_or_empty())
        .insert_resource(WorldCachePersistence::default())
        .insert_resource(UiStatus {
            message: status_message,
            processing: false,
            worker_message: None,
        });
    #[cfg(target_arch = "wasm32")]
    app.insert_resource(WasmStartupGate::default());
    #[cfg(target_arch = "wasm32")]
    app.insert_resource(WasmWarmupKickoff::default());
    #[cfg(not(target_arch = "wasm32"))]
    app.init_resource::<InferenceRenderPauseState>();
    #[cfg(not(target_arch = "wasm32"))]
    app.insert_resource(mcp_scene_control);
    add_default_plugins(&mut app);
    #[cfg(all(not(target_arch = "wasm32"), feature = "wgpu"))]
    app.add_plugins(SharedWgpuInferenceDevicePlugin);
    #[cfg(not(target_arch = "wasm32"))]
    if !app.is_plugin_added::<PointerInputPlugin>() {
        app.add_plugins(DefaultPickingPlugins);
    }
    #[cfg(not(target_arch = "wasm32"))]
    app.add_plugins(TransformGizmoPlugin);
    app.add_plugins(PanOrbitCameraPlugin);
    app.add_plugins(InfiniteGridPlugin);
    app.add_plugins(BurnSynthUiPlugin);
    app.add_systems(
        Update,
        (
            handle_exit_requests,
            handle_open_file_dialog,
            handle_file_dialog_loads,
            handle_dropped_files,
            #[cfg(target_arch = "wasm32")]
            kickoff_wasm_warmup.before(finish_wasm_startup_when_models_ready),
            #[cfg(target_arch = "wasm32")]
            finish_wasm_startup_when_models_ready.before(drive_inference),
            #[cfg(not(target_arch = "wasm32"))]
            sync_inference_render_pause_before.before(drive_inference),
            drive_inference,
            #[cfg(not(target_arch = "wasm32"))]
            sync_inference_render_pause_after.after(drive_inference),
            handle_catalog_spawn_requests,
            handle_catalog_delete_requests,
            delete_selected_meshes,
            (mark_world_cache_dirty, persist_world_cache).chain(),
            update_spinner,
            rotate_spinner,
            update_window_title,
        ),
    );
    app.add_systems(
        PostUpdate,
        (sync_panorbit_bindings, sync_panorbit_enabled).before(PanOrbitCameraSystemSet),
    );
    app.add_plugins(
        FileDialogPlugin::new()
            .with_load_file::<ImagePickDialog>()
            .with_drop_file::<ImagePickDialog>(),
    );

    #[cfg(not(target_arch = "wasm32"))]
    {
        app.add_systems(Startup, (setup, configure_mesh_picking).chain());
        app.add_systems(
            PostUpdate,
            (
                remove_entity_from_selection_if_despawned,
                update_selection_from_primary_click.after(TransformGizmoSystems::Main),
            ),
        );
    }

    #[cfg(target_arch = "wasm32")]
    {
        app.add_systems(Startup, setup);
        app.add_systems(PostUpdate, remove_entity_from_selection_if_despawned);
    }

    #[cfg(not(target_arch = "wasm32"))]
    app.add_systems(
        Update,
        poll_mcp_scene_control.before(mark_world_cache_dirty),
    );

    app.run();
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn should_run_headless_once(args: &AppArgs) -> bool {
    if args.image.is_none() || args.output.is_none() || args.mesh.is_some() {
        return false;
    }
    true
}

#[cfg(not(target_arch = "wasm32"))]
fn run_headless_once(args: &AppArgs) -> Result<(), String> {
    let image_path = args
        .image
        .as_ref()
        .cloned()
        .ok_or_else(|| "headless mode requires --image".to_string())?;
    if !image_path.exists() {
        return Err(format!("input image not found: {}", image_path.display()));
    }
    let output_path = resolve_output_path(args.output.as_ref(), &image_path, 0)
        .ok_or_else(|| "headless mode requires --output".to_string())?;

    println!(
        "running headless bevy_synth inference: input={} output={}",
        image_path.display(),
        output_path.display()
    );

    let start = Instant::now();
    let mesh = run_headless_once_inference(args, &image_path)?;

    write_glb(&output_path, &mesh)
        .map_err(|err| format!("failed to write {}: {err}", output_path.display()))?;
    println!(
        "headless synthesis completed in {:.2}s -> {}",
        start.elapsed().as_secs_f32(),
        output_path.display()
    );
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn run_headless_once_inference(args: &AppArgs, image_path: &Path) -> Result<SynthMesh, String> {
    let worker = start_worker(args);
    let request = InferenceRequest {
        id: 0,
        image_path: image_path.to_path_buf(),
        image_contents: None,
        output_path: None,
    };
    worker
        .sender
        .send(WorkerCommand::Infer(vec![request]))
        .map_err(|err| format!("failed to queue inference request: {err}"))?;

    let mut result = None;
    let start = Instant::now();
    while result.is_none() {
        let next = {
            let receiver = worker
                .receiver
                .lock()
                .map_err(|_| "failed to lock worker receiver".to_string())?;
            receiver.recv_timeout(Duration::from_millis(500))
        };

        match next {
            Ok(event) => {
                if let Some(message) = event.status_message {
                    println!("{message}");
                }
                if event.requests.is_empty() && event.results.is_empty() {
                    continue;
                }
                let mut results = event.results.into_iter();
                result = Some(
                    results
                        .next()
                        .ok_or_else(|| "worker returned empty inference result".to_string())?,
                );
            }
            Err(RecvTimeoutError::Timeout) => {
                if start.elapsed() > Duration::from_secs(60 * 30) {
                    let _ = worker.sender.send(WorkerCommand::Shutdown);
                    return Err("timed out waiting for worker inference result".to_string());
                }
            }
            Err(RecvTimeoutError::Disconnected) => {
                let _ = worker.sender.send(WorkerCommand::Shutdown);
                return Err("inference worker disconnected before returning a result".to_string());
            }
        }
    }

    let _ = worker.sender.send(WorkerCommand::Shutdown);
    match result.expect("result must be set before loop exits") {
        Ok(Some(mesh)) => Ok(mesh),
        Ok(None) => Err(format!(
            "synthesis produced an empty mesh for {}",
            image_path.display()
        )),
        Err(err) => Err(format!(
            "synthesis inference failed for {}: {err}",
            image_path.display()
        )),
    }
}

#[cfg(not(target_arch = "wasm32"))]
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
    app.add_plugins(
        DefaultPlugins
            .set(WebAssetPlugin {
                silence_startup_warning: true,
            })
            .set(asset_plugin),
    );
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

#[cfg(target_arch = "wasm32")]
fn parse_weight_precision_override(value: &str) -> Option<WeightPrecision> {
    match value.trim().to_ascii_lowercase().as_str() {
        "f16" | "fp16" => Some(WeightPrecision::F16),
        "f32" | "fp32" => Some(WeightPrecision::F32),
        "auto" => Some(WeightPrecision::Auto),
        _ => None,
    }
}

#[cfg(target_arch = "wasm32")]
fn apply_wasm_url_overrides(args: &mut AppArgs) {
    let Some(window) = web_sys::window() else {
        return;
    };
    let Ok(search) = window.location().search() else {
        return;
    };
    if search.is_empty() {
        return;
    }

    for pair in search.trim_start_matches('?').split('&') {
        if pair.is_empty() {
            continue;
        }
        let mut parts = pair.splitn(2, '=');
        let Some(key) = parts.next() else {
            continue;
        };
        let value = parts.next().unwrap_or_default();
        let key = key.trim().to_ascii_lowercase();

        if matches!(
            key.as_str(),
            "weights_precision" | "triposg_weights_precision"
        ) {
            if let Some(precision) = parse_weight_precision_override(value) {
                args.weights_precision = precision;
            }
            continue;
        }

        if key == "rmbg_weights_precision"
            && let Some(precision) = parse_weight_precision_override(value)
        {
            args.rmbg_weights_precision = precision;
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn wasm_console_log(message: &str) {
    web_sys::console::log_1(&wasm_bindgen::JsValue::from_str(message));
}

#[cfg(target_arch = "wasm32")]
fn wasm_set_warmup_state(state: &str) {
    if let Some(window) = web_sys::window() {
        let window_js: wasm_bindgen::JsValue = window.into();
        let _ = js_sys::Reflect::set(
            &window_js,
            &wasm_bindgen::JsValue::from_str("__bevySynthWarmupState"),
            &wasm_bindgen::JsValue::from_str(state),
        );
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn spawn_inference_pause_overlay(commands: &mut Commands) {
    commands
        .spawn((
            InferencePauseOverlay,
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(0.0),
                right: Val::Px(0.0),
                top: Val::Px(0.0),
                bottom: Val::Px(0.0),
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..default()
            },
            GlobalZIndex(10_000),
            BackgroundColor(Color::srgba(0.03, 0.04, 0.06, 0.9)),
            Visibility::Hidden,
        ))
        .with_children(|root| {
            root.spawn((
                Node {
                    width: Val::Percent(72.0),
                    max_width: Val::Px(880.0),
                    padding: UiRect::all(Val::Px(24.0)),
                    border: UiRect::all(Val::Px(1.0)),
                    justify_content: JustifyContent::Center,
                    align_items: AlignItems::Center,
                    ..default()
                },
                BorderColor::all(Color::srgba(0.82, 0.88, 0.98, 0.75)),
                BackgroundColor(Color::srgba(0.09, 0.11, 0.16, 0.96)),
            ))
            .with_children(|panel| {
                panel.spawn((
                    Text::new(
                        "rendering paused during inference.\ntemporary workaround for an open wgpu bug.\nthe viewport will resume after inference completes.",
                    ),
                    TextFont::from_font_size(22.0),
                    TextColor(Color::srgb(0.94, 0.96, 1.0)),
                    TextLayout::new_with_justify(Justify::Center),
                ));
            });
        });
}

#[allow(clippy::too_many_arguments)]
fn initialize_interactive_scene(
    commands: &mut Commands,
    asset_server: &AssetServer,
    meshes: &mut ResMut<Assets<BevyMesh>>,
    images: &mut ResMut<Assets<Image>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    ambient_light: &mut ResMut<AmbientLight>,
    args: &AppArgs,
    queue: &mut ResMut<InferenceQueue>,
    status: &mut ResMut<UiStatus>,
    catalog: &mut ResMut<CatalogState>,
    cache: &mut ResMut<MeshCacheResource>,
    world_cache: &mut ResMut<WorldCachePersistence>,
) {
    let mut camera_transform =
        Transform::from_translation(Vec3::new(0.0, 1.5, 5.0)).looking_at(Vec3::ZERO, Vec3::Y);
    let mut camera_orbit = PanOrbitCamera {
        allow_upside_down: true,
        orbit_smoothness: 0.1,
        pan_smoothness: 0.1,
        zoom_smoothness: 0.1,
        button_orbit: MouseButton::Right,
        button_pan: MouseButton::Middle,
        ..default()
    };
    if let Some(cached_camera) = cache.cache.camera_state()
        && !apply_cached_camera_state(cached_camera, &mut camera_transform, &mut camera_orbit)
    {
        warn!("ignoring cached camera state due to invalid values.");
    }

    commands.spawn((
        Camera3d::default(),
        camera_transform,
        camera_orbit,
        GizmoCamera,
        MeshPickingCamera,
        RenderLayers::layer(0).with(12),
        MainCamera,
    ));
    spawn_default_lighting(commands, ambient_light);

    commands.spawn((
        InfiniteGridBundle {
            settings: InfiniteGridSettings {
                fadeout_distance: 200.0,
                ..default()
            },
            ..default()
        },
        Pickable::IGNORE,
        RenderLayers::layer(0),
    ));

    hydrate_from_cache(commands, meshes, images, materials, queue, catalog, cache);
    seed_default_catalog_cube(meshes, materials, queue, catalog);
    world_cache.dirty = false;
    world_cache.timer.reset();

    if let Some(mesh_path) = args.mesh.as_ref() {
        if mesh_path.exists() {
            spawn_mesh_asset(commands, asset_server, materials, mesh_path.clone());
        } else {
            warn!("mesh path {:?} does not exist; skipping", mesh_path);
        }
    }

    if let Some(image_path) = args.image.as_ref() {
        let request = enqueue_inference(image_path.clone(), args, queue);
        catalog.add_pending(&request);
    }

    update_status_message(args, queue, status);
}

#[allow(clippy::too_many_arguments)]
#[cfg(not(target_arch = "wasm32"))]
fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<BevyMesh>>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut ambient_light: ResMut<AmbientLight>,
    args: Res<AppArgs>,
    #[cfg(all(not(target_arch = "wasm32"), feature = "wgpu"))] shared_wgpu_device: Option<
        Res<SharedWgpuInferenceDevice>,
    >,
    #[cfg(not(target_arch = "wasm32"))] event_loop_proxy: Option<
        Res<EventLoopProxyWrapper<WakeUp>>,
    >,
    mut queue: ResMut<InferenceQueue>,
    mut status: ResMut<UiStatus>,
    mut catalog: ResMut<CatalogState>,
    mut cache: ResMut<MeshCacheResource>,
    mut world_cache: ResMut<WorldCachePersistence>,
) {
    info!("bevy_synth args: {:?}", *args);

    commands.spawn((
        Camera2d,
        IsDefaultUiCamera,
        Camera {
            order: 10,
            clear_color: ClearColorConfig::None,
            ..default()
        },
    ));
    #[cfg(not(target_arch = "wasm32"))]
    spawn_inference_pause_overlay(&mut commands);

    #[cfg(not(target_arch = "wasm32"))]
    let wake_callback = make_worker_wake_callback(args.as_ref(), event_loop_proxy.as_ref());
    #[cfg(all(not(target_arch = "wasm32"), feature = "wgpu"))]
    let worker = start_worker_with_shared_wgpu_device_and_wake(
        args.as_ref(),
        shared_wgpu_device
            .as_ref()
            .and_then(|shared| shared.device.clone()),
        wake_callback,
    );
    #[cfg(all(not(target_arch = "wasm32"), not(feature = "wgpu")))]
    let worker = start_worker_with_wake(args.as_ref(), wake_callback);
    commands.insert_resource(worker);

    initialize_interactive_scene(
        &mut commands,
        &asset_server,
        &mut meshes,
        &mut images,
        &mut materials,
        &mut ambient_light,
        args.as_ref(),
        &mut queue,
        &mut status,
        &mut catalog,
        &mut cache,
        &mut world_cache,
    );
}

#[cfg(target_arch = "wasm32")]
fn setup(mut commands: Commands, args: Res<AppArgs>, mut status: ResMut<UiStatus>) {
    info!("bevy_synth args: {:?}", *args);
    wasm_set_warmup_state("idle");

    let worker = start_worker(args.as_ref());
    commands.insert_resource(worker);

    status.worker_message = Some("Initializing wasm runtime...".to_string());
    status.message = "Initializing wasm runtime...".to_string();
}

#[cfg(target_arch = "wasm32")]
fn kickoff_wasm_warmup(
    time: Res<Time>,
    mut warmup: ResMut<WasmWarmupKickoff>,
    worker: Res<InferenceWorker>,
    mut status: ResMut<UiStatus>,
    exit_state: Res<ExitState>,
) {
    if warmup.sent || exit_state.requested {
        return;
    }
    warmup.timer.tick(time.delta());
    if !warmup.timer.is_finished() {
        return;
    }

    match worker.sender.send(WorkerCommand::Warmup) {
        Ok(()) => {
            warmup.sent = true;
            status.worker_message = Some(WASM_STATUS_LOADING_MODELS.to_string());
            status.message = WASM_STATUS_LOADING_MODELS.to_string();
            wasm_set_warmup_state("requested");
            wasm_console_log("bevy_synth wasm warmup: requested");
        }
        Err(err) => {
            let message = format!("failed to request wasm model warmup: {err}");
            status.worker_message = Some(message.clone());
            status.message = message;
            wasm_set_warmup_state("failed");
            warn!("{}", status.message);
        }
    }
}

#[cfg(target_arch = "wasm32")]
#[allow(clippy::too_many_arguments)]
fn finish_wasm_startup_when_models_ready(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<BevyMesh>>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut ambient_light: ResMut<AmbientLight>,
    args: Res<AppArgs>,
    mut queue: ResMut<InferenceQueue>,
    mut status: ResMut<UiStatus>,
    mut catalog: ResMut<CatalogState>,
    mut cache: ResMut<MeshCacheResource>,
    mut world_cache: ResMut<WorldCachePersistence>,
    mut startup: ResMut<WasmStartupGate>,
) {
    if startup.scene_initialized || (!startup.model_ready && !startup.model_failed) {
        return;
    }
    initialize_interactive_scene(
        &mut commands,
        &asset_server,
        &mut meshes,
        &mut images,
        &mut materials,
        &mut ambient_light,
        args.as_ref(),
        &mut queue,
        &mut status,
        &mut catalog,
        &mut cache,
        &mut world_cache,
    );
    startup.scene_initialized = true;
    if startup.model_ready {
        status.worker_message = None;
        update_status_message(args.as_ref(), &queue, &mut status);
    }
}

fn spawn_default_lighting(commands: &mut Commands, ambient_light: &mut AmbientLight) {
    ambient_light.color = Color::srgb(0.84, 0.88, 0.95);
    ambient_light.brightness = 220.0;

    commands.spawn((
        DirectionalLight {
            color: Color::srgb(1.0, 0.97, 0.93),
            illuminance: lux::AMBIENT_DAYLIGHT,
            shadows_enabled: true,
            shadow_depth_bias: 0.12,
            shadow_normal_bias: 0.8,
            ..default()
        },
        Transform::from_xyz(9.0, 12.0, 7.0).looking_at(Vec3::ZERO, Vec3::Y),
        CascadeShadowConfigBuilder {
            first_cascade_far_bound: 12.0,
            maximum_distance: 72.0,
            ..default()
        }
        .build(),
        preview_light_layers(),
    ));

    commands.spawn((
        DirectionalLight {
            color: Color::srgb(0.72, 0.82, 1.0),
            illuminance: 14_000.0,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_xyz(-10.0, 6.0, -8.0).looking_at(Vec3::ZERO, Vec3::Y),
        preview_light_layers(),
    ));

    commands.spawn((
        DirectionalLight {
            color: Color::srgb(1.0, 0.92, 0.84),
            illuminance: 8_000.0,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_xyz(0.0, 9.5, -12.0).looking_at(Vec3::ZERO, Vec3::Y),
        preview_light_layers(),
    ));
}

fn hydrate_from_cache(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<BevyMesh>>,
    images: &mut ResMut<Assets<Image>>,
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
                    "cache metadata exists for key {} but mesh payload is missing.",
                    metadata.cache_key
                );
                continue;
            }
            Err(err) => {
                warn!(
                    "failed to load cached mesh for key {}: {err}",
                    metadata.cache_key
                );
                continue;
            }
        };

        let mesh_handle = meshes.add(to_bevy_mesh_synth(&mesh));
        let material = materials.add(standard_material_for_inference(&mesh, images.as_mut()));
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
                "skipping cached world item for unknown cache key {}",
                item.cache_key
            );
            continue;
        };
        let Some(transform) = transform_from_cached_world_item(&item) else {
            warn!(
                "skipping cached world item with invalid transform for key {}",
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
            "loaded {loaded_meshes} cached catalog mesh(es) and {loaded_world_items} cached world item(s)."
        );
    }
}

fn seed_default_catalog_cube(
    meshes: &mut ResMut<Assets<BevyMesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    queue: &mut ResMut<InferenceQueue>,
    catalog: &mut ResMut<CatalogState>,
) {
    if catalog.has_ready_cube_entry() {
        return;
    }

    let mesh = meshes.add(BevyMesh::from(Cuboid::from_size(Vec3::ONE)));
    let material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.8, 0.84, 0.9),
        perceptual_roughness: 0.58,
        cull_mode: None,
        ..default()
    });
    let id = queue.counter;
    queue.counter = queue.counter.wrapping_add(1);
    catalog.add_ready(id, "cube".to_string(), mesh, material, None, None);
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

fn camera_state_from_components(
    transform: &Transform,
    orbit: &PanOrbitCamera,
) -> Option<CachedCameraState> {
    let translation = transform.translation;
    let rotation = if transform.rotation.length_squared() > 0.0 {
        transform.rotation.normalize()
    } else {
        Quat::IDENTITY
    };
    let focus = orbit.focus;
    let yaw = orbit.yaw.unwrap_or(orbit.target_yaw);
    let pitch = orbit.pitch.unwrap_or(orbit.target_pitch);
    let radius = orbit.radius.unwrap_or(orbit.target_radius);
    if !translation.is_finite()
        || !rotation.is_finite()
        || !focus.is_finite()
        || !yaw.is_finite()
        || !pitch.is_finite()
        || !radius.is_finite()
        || radius <= 0.0
    {
        return None;
    }

    Some(CachedCameraState {
        translation: translation.to_array(),
        rotation: rotation.to_array(),
        focus: focus.to_array(),
        yaw,
        pitch,
        radius,
    })
}

fn apply_cached_camera_state(
    state: &CachedCameraState,
    transform: &mut Transform,
    orbit: &mut PanOrbitCamera,
) -> bool {
    let translation = Vec3::from_array(state.translation);
    let raw_rotation = Quat::from_xyzw(
        state.rotation[0],
        state.rotation[1],
        state.rotation[2],
        state.rotation[3],
    );
    let focus = Vec3::from_array(state.focus);
    if !translation.is_finite()
        || !raw_rotation.is_finite()
        || !focus.is_finite()
        || !state.yaw.is_finite()
        || !state.pitch.is_finite()
        || !state.radius.is_finite()
        || state.radius <= 0.0
    {
        return false;
    }

    transform.translation = translation;
    let to_focus = focus - translation;
    if to_focus.length_squared() > 0.000_001 {
        transform.look_at(focus, Vec3::Y);
    } else {
        transform.rotation = if raw_rotation.length_squared() > 0.0 {
            raw_rotation.normalize()
        } else {
            Quat::IDENTITY
        };
    }
    orbit.focus = focus;
    orbit.target_focus = focus;
    orbit.yaw = Some(state.yaw);
    orbit.target_yaw = state.yaw;
    orbit.pitch = Some(state.pitch);
    orbit.target_pitch = state.pitch;
    orbit.radius = Some(state.radius);
    orbit.target_radius = state.radius;
    true
}

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
            .set_title("open image")
            .add_filter(
                "images",
                &[
                    "png", "jpg", "jpeg", "bmp", "gif", "webp", "tga", "tif", "tiff",
                ],
            )
            .load_multiple_files::<ImagePickDialog>();
    }
}

#[allow(clippy::too_many_arguments)]
fn handle_file_dialog_loads(
    mut events: MessageReader<DialogFileLoaded<ImagePickDialog>>,
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut materials: ResMut<Assets<StandardMaterial>>,
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
        queued += ingest_candidate_file(
            event.path(),
            &event.file_name,
            event.contents.as_slice(),
            &args,
            &mut queue,
            &mut catalog,
            &mut commands,
            &asset_server,
            &mut materials,
            "selected file",
        );
    }

    if queued > 0 {
        update_status_message(&args, &queue, &mut status);
    }
}

#[allow(clippy::too_many_arguments)]
fn handle_dropped_files(
    mut events: MessageReader<DialogFileDropped<ImagePickDialog>>,
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut materials: ResMut<Assets<StandardMaterial>>,
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
        queued += ingest_candidate_file(
            event.path(),
            &event.file_name,
            event.contents.as_slice(),
            &args,
            &mut queue,
            &mut catalog,
            &mut commands,
            &asset_server,
            &mut materials,
            "dropped file",
        );
    }
    if queued > 0 {
        update_status_message(&args, &queue, &mut status);
    }
}

#[allow(clippy::too_many_arguments)]
fn ingest_candidate_file(
    path: Option<&Path>,
    file_name: &str,
    contents: &[u8],
    args: &AppArgs,
    queue: &mut InferenceQueue,
    catalog: &mut CatalogState,
    commands: &mut Commands,
    asset_server: &AssetServer,
    materials: &mut Assets<StandardMaterial>,
    source_label: &str,
) -> usize {
    if let Some(path) = path {
        if is_image_file(path) {
            let request = enqueue_inference(path.to_path_buf(), args, queue);
            catalog.add_pending(&request);
            info!("queued inference for {}", path.display());
            return 1;
        }
        if is_mesh_file(path) {
            spawn_mesh_asset(commands, asset_server, materials, path.to_path_buf());
            info!("loaded mesh asset {}", path.display());
            return 0;
        }
        warn!(
            "{source_label} {} is not a supported image or mesh",
            path.display()
        );
        return 0;
    }

    let inferred_path = Path::new(file_name);
    if is_image_file(inferred_path) {
        let request = enqueue_inference_with_contents(
            virtual_upload_path(file_name, queue.counter),
            Some(contents.to_vec()),
            args,
            queue,
        );
        catalog.add_pending(&request);
        info!("queued uploaded inference for {file_name}");
        return 1;
    }
    if is_mesh_file(inferred_path) {
        warn!(
            "{source_label} {file_name} is a mesh, but web uploads for mesh assets require URL-backed asset loading."
        );
        return 0;
    }
    warn!("{source_label} {file_name} is not a supported image or mesh");
    0
}

fn virtual_upload_path(file_name: &str, request_id: u32) -> PathBuf {
    let mut sanitized = String::with_capacity(file_name.len());
    for ch in file_name.chars() {
        let allowed = ch.is_ascii_alphanumeric() || matches!(ch, '.' | '-' | '_');
        sanitized.push(if allowed { ch } else { '_' });
    }
    if sanitized.is_empty() {
        sanitized.push_str("upload_image");
    }
    PathBuf::from(format!("uploaded/{request_id:08}_{sanitized}"))
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn poll_mcp_scene_control(
    mut control: ResMut<McpSceneControl>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<BevyMesh>>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut cache: ResMut<MeshCacheResource>,
    mut selection: ResMut<EditorSelection>,
    transformables: Query<(), With<GizmoTransformable>>,
    cached_instances: Query<(Entity, &CachedMeshInstance)>,
    mut query_set: ParamSet<(
        Query<(&mut Transform, &mut PanOrbitCamera), With<MainCamera>>,
        Query<(&CachedMeshInstance, &Transform)>,
    )>,
    mut world_cache: ResMut<WorldCachePersistence>,
) {
    let Some(path) = control.path.clone() else {
        return;
    };
    if !path.exists() {
        return;
    }

    let metadata = match fs::metadata(&path) {
        Ok(metadata) => metadata,
        Err(err) => {
            warn!(
                "Failed to inspect MCP scene control file {}: {err}",
                path.display()
            );
            return;
        }
    };
    let modified = metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH);
    if control
        .last_modified
        .map(|last| modified <= last)
        .unwrap_or(false)
    {
        return;
    }
    control.last_modified = Some(modified);

    let commands_to_apply = match read_mcp_scene_commands(&path) {
        Ok(commands_to_apply) => commands_to_apply,
        Err(err) => {
            warn!(
                "Failed to parse MCP scene control file {}: {err}",
                path.display()
            );
            return;
        }
    };
    if commands_to_apply.is_empty() {
        return;
    }

    let mut scene_changed = false;
    let mut force_cache_flush = false;
    for command in commands_to_apply {
        match command {
            McpSceneCommand::SpawnCached {
                cache_key,
                translation,
                rotation,
                scale,
                select,
            } => {
                let mesh = match cache.cache.load_mesh(&cache_key) {
                    Ok(Some(mesh)) => mesh,
                    Ok(None) => {
                        warn!("MCP spawn_cached skipped: cache key {cache_key} not found");
                        continue;
                    }
                    Err(err) => {
                        warn!("MCP spawn_cached failed for {cache_key}: {err}");
                        continue;
                    }
                };
                let Some(transform) = transform_from_optional_parts(translation, rotation, scale)
                else {
                    warn!("MCP spawn_cached skipped due to invalid transform values");
                    continue;
                };
                let mesh_handle = meshes.add(to_bevy_mesh_synth(&mesh));
                let material =
                    materials.add(standard_material_for_inference(&mesh, images.as_mut()));
                let entity = spawn_mesh_instance(
                    &mut commands,
                    mesh_handle,
                    material,
                    transform,
                    Some(cache_key),
                );
                if select {
                    selection.set(entity);
                }
                scene_changed = true;
            }
            McpSceneCommand::DeleteByCacheKey { cache_key } => {
                let to_despawn: Vec<Entity> = cached_instances
                    .iter()
                    .filter_map(|(entity, cached)| {
                        if cached.cache_key == cache_key {
                            Some(entity)
                        } else {
                            None
                        }
                    })
                    .collect();
                for entity in to_despawn {
                    commands.entity(entity).despawn();
                    scene_changed = true;
                }
            }
            McpSceneCommand::DeleteSelected => {
                let to_despawn: Vec<Entity> = selection
                    .iter()
                    .filter(|entity| transformables.contains(*entity))
                    .collect();
                for entity in to_despawn {
                    commands.entity(entity).despawn();
                    scene_changed = true;
                }
                selection.clear();
            }
            McpSceneCommand::ClearSelection => {
                selection.clear();
            }
            McpSceneCommand::SetCamera {
                translation,
                rotation,
                focus,
                yaw,
                pitch,
                radius,
            } => {
                if let Ok((mut transform, mut orbit)) = query_set.p0().single_mut() {
                    let target_translation = Vec3::from_array(translation);
                    let target_rotation =
                        Quat::from_xyzw(rotation[0], rotation[1], rotation[2], rotation[3]);
                    if target_translation.is_finite() && target_rotation.is_finite() {
                        transform.translation = target_translation;
                        transform.rotation = if target_rotation.length_squared() > 0.0 {
                            target_rotation.normalize()
                        } else {
                            Quat::IDENTITY
                        };
                        if let Some(focus) = focus {
                            let focus = Vec3::from_array(focus);
                            if focus.is_finite() {
                                orbit.focus = focus;
                                orbit.target_focus = focus;
                            }
                        }
                        if let Some(yaw) = yaw
                            && yaw.is_finite()
                        {
                            orbit.yaw = Some(yaw);
                            orbit.target_yaw = yaw;
                        }
                        if let Some(pitch) = pitch
                            && pitch.is_finite()
                        {
                            orbit.pitch = Some(pitch);
                            orbit.target_pitch = pitch;
                        }
                        if let Some(radius) = radius
                            && radius.is_finite()
                            && radius > 0.0
                        {
                            orbit.radius = Some(radius);
                            orbit.target_radius = radius;
                        }
                        scene_changed = true;
                    }
                }
            }
            McpSceneCommand::SaveCache => {
                force_cache_flush = true;
            }
        }
    }

    if force_cache_flush {
        let camera_state = {
            let main_camera = query_set.p0();
            main_camera
                .single()
                .ok()
                .and_then(|(transform, orbit)| camera_state_from_components(transform, orbit))
        };
        let cached_query = query_set.p1();
        if let Err(err) = flush_world_cache_now(&mut cache, &cached_query, camera_state) {
            warn!("MCP save_cache failed: {err}");
        } else {
            world_cache.dirty = false;
            world_cache.timer.reset();
        }
    } else if scene_changed {
        world_cache.dirty = true;
        world_cache.timer.reset();
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
                    #[cfg(target_arch = "wasm32")]
                    {
                        if message == WASM_STATUS_MODEL_READY {
                            ctx.wasm_startup.model_ready = true;
                            wasm_set_warmup_state("ready");
                            wasm_console_log("bevy_synth wasm warmup: ready");
                        } else if message == WASM_STATUS_LOADING_MODELS {
                            wasm_set_warmup_state("loading");
                            wasm_console_log("bevy_synth wasm warmup: loading");
                        } else if message.starts_with(WASM_STATUS_MODEL_LOAD_FAILED_PREFIX) {
                            ctx.wasm_startup.model_failed = true;
                            wasm_set_warmup_state("failed");
                            wasm_console_log("bevy_synth wasm warmup: failed");
                        }
                    }
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
                        &mut ctx.images,
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

    #[cfg(target_arch = "wasm32")]
    if !ctx.wasm_startup.scene_initialized || !ctx.wasm_startup.model_ready {
        update_status_message(&ctx.args, &ctx.queue, &mut ctx.status);
        return;
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

fn handle_catalog_delete_requests(
    mut requests: MessageReader<CatalogDeleteRequest>,
    mut cache: ResMut<MeshCacheResource>,
    cached_instances: Query<(Entity, &CachedMeshInstance)>,
    mut commands: Commands,
) {
    for request in requests.read() {
        let Some(cache_key) = request.cache_key.as_ref() else {
            continue;
        };

        if let Err(err) = cache.cache.remove_mesh_entry(cache_key) {
            warn!("Failed to remove cached mesh entry {cache_key}: {err}");
        }

        let to_despawn: Vec<Entity> = cached_instances
            .iter()
            .filter_map(|(entity, cached)| {
                if &cached.cache_key == cache_key {
                    Some(entity)
                } else {
                    None
                }
            })
            .collect();
        for entity in to_despawn {
            commands.entity(entity).despawn();
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
            "bevy_synth — processing: {name} (queued: {}){dots}",
            queue.pending.len()
        )
    } else {
        format!("bevy_synth — {}", status.message)
    };

    if let Ok(mut window) = windows.single_mut() {
        window.title = title;
    }
}

#[allow(clippy::too_many_arguments)]
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
    main_camera: Query<(&Transform, &PanOrbitCamera), With<MainCamera>>,
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
    let camera_state = main_camera
        .single()
        .ok()
        .and_then(|(transform, orbit)| camera_state_from_components(transform, orbit));
    if let Err(err) = flush_world_cache_now(&mut cache, &cached_instances, camera_state) {
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

#[allow(clippy::too_many_arguments)]
fn handle_inference_result(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<BevyMesh>>,
    images: &mut ResMut<Assets<Image>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    cache: &mut ResMut<MeshCacheResource>,
    catalog: &mut ResMut<CatalogState>,
    request: InferenceRequest,
    result: Result<Option<SynthMesh>, String>,
) {
    match result {
        Ok(Some(mesh)) => {
            if let Some(output) = request.output_path.as_ref()
                && let Err(err) = write_glb(output, &mesh)
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

            let bevy_mesh = to_bevy_mesh_synth(&mesh);
            let mesh_handle = meshes.add(bevy_mesh);
            let material = materials.add(standard_material_for_inference(&mesh, images.as_mut()));
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

fn standard_material_for_inference(
    mesh: &SynthMesh,
    images: &mut Assets<Image>,
) -> StandardMaterial {
    let mut out = if let Some(material) = mesh.material {
        let base = material.base_color;
        let alpha = material.alpha.clamp(0.0, 1.0);
        StandardMaterial {
            base_color: Color::srgba(base[0], base[1], base[2], alpha),
            metallic: material.metallic.clamp(0.0, 1.0),
            perceptual_roughness: material.roughness.clamp(0.045, 1.0),
            cull_mode: None,
            alpha_mode: if alpha < 0.995 {
                AlphaMode::Blend
            } else {
                AlphaMode::Opaque
            },
            ..default()
        }
    } else {
        StandardMaterial {
            base_color: Color::srgb(0.78, 0.84, 0.92),
            cull_mode: None,
            ..default()
        }
    };

    if let Some(pbr) = mesh.pbr_textures.as_ref() {
        out.base_color_texture = Some(images.add(synth_texture_to_image(
            &pbr.base_color,
            TextureFormat::Rgba8UnormSrgb,
        )));
        out.metallic_roughness_texture = Some(images.add(synth_texture_to_image(
            &pbr.metallic_roughness,
            TextureFormat::Rgba8Unorm,
        )));
        out.metallic = 1.0;
        out.perceptual_roughness = 1.0;

        if let Some(texture) = pbr.normal.as_ref() {
            out.normal_map_texture =
                Some(images.add(synth_texture_to_image(texture, TextureFormat::Rgba8Unorm)));
        }
        if let Some(texture) = pbr.emissive.as_ref() {
            out.emissive_texture = Some(images.add(synth_texture_to_image(
                texture,
                TextureFormat::Rgba8UnormSrgb,
            )));
            out.emissive = LinearRgba::WHITE;
        }
        if let Some(texture) = pbr.occlusion.as_ref() {
            out.occlusion_texture =
                Some(images.add(synth_texture_to_image(texture, TextureFormat::Rgba8Unorm)));
        }
    }

    out
}

fn synth_texture_to_image(texture: &SynthMeshTexture, format: TextureFormat) -> Image {
    let expected = texture.width as usize * texture.height as usize * 4;
    let bytes = if texture.rgba8.len() == expected {
        texture.rgba8.clone()
    } else {
        vec![255u8; expected]
    };
    Image::new(
        Extent3d {
            width: texture.width.max(1),
            height: texture.height.max(1),
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        bytes,
        format,
        RenderAssetUsages::default(),
    )
}

pub(crate) fn enqueue_inference(
    image_path: PathBuf,
    args: &AppArgs,
    queue: &mut InferenceQueue,
) -> InferenceRequest {
    enqueue_inference_with_contents(image_path, None, args, queue)
}

pub(crate) fn enqueue_inference_with_contents(
    image_path: PathBuf,
    image_contents: Option<Vec<u8>>,
    args: &AppArgs,
    queue: &mut InferenceQueue,
) -> InferenceRequest {
    let output_path = resolve_output_path(args.output.as_ref(), &image_path, queue.counter);
    let request = InferenceRequest {
        id: queue.counter,
        image_path,
        image_contents,
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
        "upload an image (.png/.jpg) to begin.".to_string()
    } else {
        "ready. upload another image.".to_string()
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
    asset_server: &AssetServer,
    materials: &mut Assets<StandardMaterial>,
    mesh_path: PathBuf,
) {
    let mesh_handle: Handle<BevyMesh> = asset_server.load(mesh_path);
    let material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.82, 0.82, 0.9),
        cull_mode: None,
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

#[allow(clippy::type_complexity)]
fn mark_world_cache_dirty(
    mut persistence: ResMut<WorldCachePersistence>,
    changed: Query<
        (),
        (
            With<CachedMeshInstance>,
            Or<(Added<CachedMeshInstance>, Changed<Transform>)>,
        ),
    >,
    changed_camera: Query<
        (),
        (
            With<MainCamera>,
            Or<(Changed<Transform>, Changed<PanOrbitCamera>)>,
        ),
    >,
    mut removed: RemovedComponents<CachedMeshInstance>,
) {
    if changed.is_empty() && changed_camera.is_empty() && removed.read().next().is_none() {
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
    main_camera: Query<(&Transform, &PanOrbitCamera), With<MainCamera>>,
) {
    if !persistence.dirty {
        return;
    }
    persistence.timer.tick(time.delta());
    if !persistence.timer.is_finished() {
        return;
    }

    let camera_state = main_camera
        .single()
        .ok()
        .and_then(|(transform, orbit)| camera_state_from_components(transform, orbit));
    match flush_world_cache_now(&mut cache, &query, camera_state) {
        Ok(()) => {
            persistence.dirty = false;
        }
        Err(err) => {
            warn!("failed to persist world cache state: {err}");
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

fn flush_world_cache_now(
    cache: &mut ResMut<MeshCacheResource>,
    query: &Query<(&CachedMeshInstance, &Transform)>,
    camera_state: Option<CachedCameraState>,
) -> Result<(), String> {
    cache
        .cache
        .set_world_items(collect_cached_world_items(query))
        .map_err(|err| err.to_string())?;
    cache
        .cache
        .set_camera_state(camera_state)
        .map_err(|err| err.to_string())?;
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn read_mcp_scene_commands(path: &std::path::Path) -> Result<Vec<McpSceneCommand>, io::Error> {
    let content = fs::read_to_string(path)?;
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }
    if trimmed.starts_with('[') {
        return serde_json::from_str::<Vec<McpSceneCommand>>(trimmed).map_err(|err| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid command array JSON: {err}"),
            )
        });
    }

    serde_json::from_str::<McpSceneCommandEnvelope>(trimmed)
        .map(|envelope| envelope.commands)
        .map_err(|err| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid command envelope JSON: {err}"),
            )
        })
}

#[cfg(not(target_arch = "wasm32"))]
fn transform_from_optional_parts(
    translation: Option<[f32; 3]>,
    rotation: Option<[f32; 4]>,
    scale: Option<[f32; 3]>,
) -> Option<Transform> {
    let translation = translation.map(Vec3::from_array).unwrap_or(Vec3::ZERO);
    let rotation = rotation
        .map(|q| Quat::from_xyzw(q[0], q[1], q[2], q[3]))
        .unwrap_or(Quat::IDENTITY);
    let scale = scale.map(Vec3::from_array).unwrap_or(Vec3::ONE);
    if !translation.is_finite() || !rotation.is_finite() || !scale.is_finite() {
        return None;
    }
    let rotation = if rotation.length_squared() > 0.0 {
        rotation.normalize()
    } else {
        Quat::IDENTITY
    };
    Some(Transform {
        translation,
        rotation,
        scale: if scale.length_squared() > 0.0 {
            scale
        } else {
            Vec3::ONE
        },
    })
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

#[allow(clippy::too_many_arguments)]
#[cfg(not(target_arch = "wasm32"))]
fn update_selection_from_primary_click(
    buttons: Res<ButtonInput<MouseButton>>,
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut selection: ResMut<EditorSelection>,
    windows: Query<&Window, With<PrimaryWindow>>,
    ui_state: Res<CatalogUiState>,
    gizmos: Query<(&TransformGizmo, &InheritedVisibility)>,
    cameras: Query<(&Camera, &GlobalTransform), With<MainCamera>>,
    gizmo_handles_meshes: Query<
        (&Mesh3d, &GlobalTransform, Option<&InheritedVisibility>),
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
    let (gizmo_interacting, gizmo_visible) = gizmos
        .single()
        .map(|(gizmo, inherited_visibility)| {
            (gizmo.interaction().is_some(), inherited_visibility.get())
        })
        .unwrap_or((false, false));
    if gizmo_interacting {
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
    let on_gizmo_handle = gizmo_visible
        && gizmo_handles_meshes
            .iter()
            .any(|(mesh3d, transform, inherited_visibility)| {
                if let Some(inherited_visibility) = inherited_visibility
                    && !inherited_visibility.get()
                {
                    return false;
                }
                let Some(mesh) = meshes.get(&mesh3d.0) else {
                    return false;
                };
                let Some(aabb) = mesh.compute_aabb() else {
                    return false;
                };
                let (world_min, world_max) =
                    world_aabb(aabb.center.into(), aabb.half_extents.into(), transform);
                ray_aabb_intersection(ray.origin, ray.direction.as_vec3(), world_min, world_max)
                    .is_some()
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

#[cfg(not(target_arch = "wasm32"))]
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

#[cfg(not(target_arch = "wasm32"))]
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

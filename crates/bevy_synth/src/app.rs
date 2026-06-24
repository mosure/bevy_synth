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
#[cfg(target_arch = "wasm32")]
use bevy::asset::io::web::WebAssetPlugin;
#[cfg(target_arch = "wasm32")]
use bevy::asset::{AssetMetaCheck, AssetMode};
use bevy::asset::{AssetPlugin, RenderAssetUsages, UnapprovedPathMode};
use bevy::camera::primitives::MeshAabb;
use bevy::camera::visibility::RenderLayers;
use bevy::ecs::message::{MessageReader, MessageWriter};
use bevy::ecs::system::SystemParam;
use bevy::input::mouse::{MouseMotion, MouseScrollUnit, MouseWheel};
use bevy::light::{
    CascadeShadowConfigBuilder, DirectionalLight, DirectionalLightShadowMap, GlobalAmbientLight,
    PointLight,
};
use bevy::math::primitives::Cuboid;
use bevy::pbr::{MeshMaterial3d, StandardMaterial};
use bevy::prelude::*;
#[cfg(all(not(target_arch = "wasm32"), feature = "wgpu"))]
use bevy::render::RenderApp;
#[cfg(target_arch = "wasm32")]
use bevy::render::RenderPlugin;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
#[cfg(all(not(target_arch = "wasm32"), feature = "wgpu"))]
use bevy::render::renderer::{
    RenderAdapter, RenderAdapterInfo, RenderDevice, RenderInstance, RenderQueue,
};
#[cfg(target_arch = "wasm32")]
use bevy::render::settings::{Backends, WgpuSettings};
#[cfg(not(target_arch = "wasm32"))]
use bevy::render::view::screenshot::{Screenshot, save_to_disk};
use bevy::window::{PrimaryWindow, WindowCloseRequested};
#[cfg(target_arch = "wasm32")]
use bevy::window::{Window, WindowPlugin};
#[cfg(not(target_arch = "wasm32"))]
use bevy::winit::{
    EventLoopProxy, EventLoopProxyWrapper, UpdateMode, WinitSettings, WinitUserEvent,
};
use bevy_gaussian_splatting::gaussian::settings::GaussianColorSpace;
use bevy_gaussian_splatting::sort::SortMode;
use bevy_gaussian_splatting::{
    CloudSettings, Gaussian3d, GaussianCamera, GaussianSplattingPlugin, PlanarGaussian3d,
    PlanarGaussian3dHandle, SphericalHarmonicCoefficients,
};
use bevy_mesh::{Mesh as BevyMesh, Mesh3d};
use bevy_picking::DefaultPickingPlugins;
use bevy_picking::hover::PickingInteraction;
use bevy_picking::input::PointerInputPlugin;
use bevy_picking::prelude::MeshPickingSettings;
use bevy_picking::prelude::{MeshPickingCamera, Pickable};
use bevy_synth_ui::bevy_editor_core::selection::{
    EditorSelection, Selectable, remove_entity_from_selection_if_despawned,
};
use bevy_synth_ui::bevy_file_dialog::prelude::{
    DialogFileDropped, DialogFileLoaded, DialogFileSaveCanceled, DialogFileSaved, FileDialogExt,
    FileDialogPlugin,
};
use bevy_synth_ui::bevy_transform_gizmos;
use bevy_synth_ui::bevy_transform_gizmos::TransformGizmoSystems;
use bevy_synth_ui::bevy_transform_gizmos::prelude::GizmoCamera;
use bevy_synth_ui::bevy_transform_gizmos::prelude::TransformGizmoPlugin;
use bevy_synth_ui::bevy_transform_gizmos::{
    GizmoTransformable, TransformGizmo, TransformGizmoOffset,
};
use clap::Parser;
#[cfg(not(target_arch = "wasm32"))]
use serde::{Deserialize, Serialize};

#[cfg(not(target_arch = "wasm32"))]
use burn_synth_scene::scene_bsn_file_to_mcp_command_envelope;
#[cfg(not(target_arch = "wasm32"))]
use burn_synth_scene::{
    SceneAssetAabb, SceneAssetBinding, SceneAssetFrame, SceneCamera, parse_scene_bsn,
};

#[cfg(all(not(target_arch = "wasm32"), feature = "wgpu"))]
use bevy_synth_runtime::args::BackendKind;
use bevy_synth_runtime::args::SynthesisModel;
use bevy_synth_runtime::args::{AppArgs, Args, build_app_args};
#[cfg(target_arch = "wasm32")]
use bevy_synth_runtime::args::{QualityPreset, RmbgModel, TripoSplatProfile, WeightPrecision};
#[cfg(not(target_arch = "wasm32"))]
use bevy_synth_runtime::cache::CachedAssetKind;
use bevy_synth_runtime::cache::{
    CachedAssetAabb, CachedAssetFrame, CachedCameraState, CachedMeshMetadata, CachedWorldItem,
    MeshCache,
};
#[cfg(not(target_arch = "wasm32"))]
use bevy_synth_runtime::io::mesh_from_glb_bytes;
use bevy_synth_runtime::io::{
    SceneGlbMeshInstance, image_bytes_to_bevy_image, is_image_file, is_mesh_file,
    resolve_output_path, scene_meshes_to_glb_bytes, write_glb,
};
use bevy_synth_runtime::mesh::to_bevy_mesh_synth;
use bevy_synth_runtime::state::{
    ExitState, InferenceQueue, InferenceRequest, InferenceSettings, InferenceWorker, Spinner,
    TitlePulse, UiStatus, WorkerCommand,
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
use bevy_synth_runtime::{GaussianSplatCloud, SynthAsset, SynthMesh, SynthMeshTexture, TripoMesh};
use bevy_synth_ui::ImagePickDialog;
use bevy_synth_ui::{
    BurnSynthUiPlugin, BurnSynthUiSystemSet, CatalogDeleteRequest, CatalogSpawnAsset,
    CatalogSpawnRequest, CatalogState, CatalogStatus, CatalogUiState, DragState, MainCamera,
    SceneSaveKind, SceneSaveRequest, preview_light_layers,
};

use crate::infinite_grid::{InfiniteGridBundle, InfiniteGridPlugin, InfiniteGridSettings};

const BUILTIN_CUBE_SOURCE_IMAGE: &str = "builtin/cube";
const PANORBIT_MIN_RADIUS: f32 = 0.05;
const PANORBIT_MAX_RADIUS: f32 = 500.0;
const PANORBIT_ORBIT_SMOOTHNESS: f32 = 0.1;
const PANORBIT_PAN_SMOOTHNESS: f32 = 0.02;
const PANORBIT_ZOOM_SMOOTHNESS: f32 = 0.1;
const PANORBIT_SNAP_EPSILON: f32 = 0.001;

#[derive(Component, Clone, Debug)]
struct PanOrbitCamera {
    button_orbit: MouseButton,
    button_pan: MouseButton,
    enabled: bool,
    initialized: bool,
    allow_upside_down: bool,
    is_upside_down: bool,
    focus: Vec3,
    target_focus: Vec3,
    yaw: Option<f32>,
    target_yaw: f32,
    pitch: Option<f32>,
    target_pitch: f32,
    radius: Option<f32>,
    target_radius: f32,
}

impl Default for PanOrbitCamera {
    fn default() -> Self {
        Self {
            button_orbit: MouseButton::Left,
            button_pan: MouseButton::Right,
            enabled: true,
            initialized: false,
            allow_upside_down: false,
            is_upside_down: false,
            focus: Vec3::ZERO,
            target_focus: Vec3::ZERO,
            yaw: None,
            target_yaw: 0.0,
            pitch: None,
            target_pitch: 0.0,
            radius: None,
            target_radius: 1.0,
        }
    }
}

#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PanOrbitCameraSystemSet;

struct PanOrbitCameraPlugin;

impl Plugin for PanOrbitCameraPlugin {
    fn build(&self, app: &mut App) {
        app.configure_sets(PostUpdate, PanOrbitCameraSystemSet)
            .add_systems(
                PostUpdate,
                update_panorbit_camera.in_set(PanOrbitCameraSystemSet),
            );
    }
}

#[derive(Component, Clone, Debug)]
pub(crate) struct CachedMeshInstance {
    pub(crate) cache_key: String,
}

#[derive(Clone, Debug)]
struct SceneBsnSaveDialog;

#[derive(Clone, Debug)]
struct SceneGlbSaveDialog;

#[derive(Resource, Default)]
struct PendingSceneBsnSave {
    assets_json: Option<Vec<u8>>,
}

#[derive(Resource, Clone, Debug, Default)]
pub(crate) struct SceneInteractionLock {
    pub(crate) locked: bool,
    pub(crate) reason: Option<String>,
}

impl SceneInteractionLock {
    pub(crate) fn set(&mut self, locked: bool, reason: Option<String>) {
        self.locked = locked;
        self.reason = if locked {
            reason.filter(|value| !value.trim().is_empty())
        } else {
            None
        };
    }
}

#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub(crate) struct GaussianSplatPickBounds {
    pub(crate) center: Vec3,
    pub(crate) half_extents: Vec3,
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
        let flush_delay_seconds = if cfg!(target_arch = "wasm32") {
            // On web, users often refresh immediately after scene edits.
            // Persist world placements on the same frame to avoid losing them.
            0.0
        } else {
            0.35
        };
        Self {
            dirty: false,
            timer: Timer::from_seconds(flush_delay_seconds, TimerMode::Once),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Resource, Default)]
struct McpSceneControl {
    path: Option<PathBuf>,
    status_path: Option<PathBuf>,
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
            // The wasm worker warms model pipelines on demand before the first
            // matching inference request. Startup should show the app first so
            // model/profile selection happens inside the Bevy UI.
            sent: true,
            timer: Timer::from_seconds(1.5, TimerMode::Once),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl McpSceneControl {
    fn from_args(args: &AppArgs) -> Self {
        let status_path = args
            .mcp_scene_control_path
            .as_ref()
            .map(|path| path.with_extension("status.json"));
        Self {
            path: args.mcp_scene_control_path.clone(),
            status_path,
            last_modified: None,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn prepare_startup_bsn_scene(args: &mut AppArgs) -> Result<(), String> {
    let Some(bsn_path) = args.scene_bsn.as_ref() else {
        if args.scene_assets_json.is_some() {
            return Err("--scene-assets-json requires --scene-bsn".to_string());
        }
        return Ok(());
    };
    let assets_path = args
        .scene_assets_json
        .as_ref()
        .ok_or_else(|| "--scene-bsn requires --scene-assets-json".to_string())?;
    let command_path = args.mcp_scene_control_path.clone().unwrap_or_else(|| {
        std::env::temp_dir().join(format!(
            "bevy_synth_bsn_scene_{}_commands.json",
            std::process::id()
        ))
    });
    let envelope = scene_bsn_file_to_mcp_command_envelope(
        bsn_path,
        assets_path,
        args.scene_bsn_clear_existing,
        Some("bevy_synth-startup-bsn"),
        Some(1),
    )
    .map_err(|err| err.to_string())?;
    if let Some(parent) = command_path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            format!(
                "failed to create BSN scene command directory {}: {err}",
                parent.display()
            )
        })?;
    }
    let bytes = serde_json::to_vec_pretty(&envelope)
        .map_err(|err| format!("failed to serialize BSN scene command envelope: {err}"))?;
    fs::write(&command_path, bytes).map_err(|err| {
        format!(
            "failed to write BSN scene command envelope {}: {err}",
            command_path.display()
        )
    })?;
    args.mcp_scene_control_path = Some(command_path);
    Ok(())
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

        info!("Initialized shared Burn WGPU device from Bevy render context.");
        app.insert_resource(SharedWgpuInferenceDevice {
            device: shared_device,
        });
    }
}

#[cfg(not(target_arch = "wasm32"))]
const INFERENCE_PAUSE_WAIT: Duration = Duration::from_secs(60 * 60);
#[cfg(not(target_arch = "wasm32"))]
const INFERENCE_DISPATCH_VISIBLE_FRAMES: u8 = 3;

#[cfg(not(target_arch = "wasm32"))]
#[derive(Component)]
struct InferencePauseOverlay;

#[cfg(not(target_arch = "wasm32"))]
#[derive(Resource, Debug, Clone)]
pub(crate) struct InferenceDispatchGate {
    visible_frames_remaining: u8,
}

#[cfg(not(target_arch = "wasm32"))]
impl Default for InferenceDispatchGate {
    fn default() -> Self {
        Self {
            visible_frames_remaining: INFERENCE_DISPATCH_VISIBLE_FRAMES,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl InferenceDispatchGate {
    #[cfg(test)]
    pub(crate) fn ready_for_dispatch() -> Self {
        Self {
            visible_frames_remaining: 0,
        }
    }
}

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
    queue
        .active
        .as_ref()
        .map(|batch| !batch.is_empty())
        .unwrap_or(false)
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn should_wait_before_inference_dispatch(
    gate: &mut InferenceDispatchGate,
    queue: &InferenceQueue,
) -> bool {
    if queue.active.is_some() || queue.pending.is_empty() {
        gate.visible_frames_remaining = INFERENCE_DISPATCH_VISIBLE_FRAMES;
        return false;
    }
    if gate.visible_frames_remaining == 0 {
        return false;
    }
    gate.visible_frames_remaining -= 1;
    true
}

#[cfg(not(target_arch = "wasm32"))]
fn reset_inference_dispatch_gate(gate: &mut InferenceDispatchGate) {
    gate.visible_frames_remaining = INFERENCE_DISPATCH_VISIBLE_FRAMES;
}

#[cfg(not(target_arch = "wasm32"))]
fn sync_inference_render_pause(
    args: Res<AppArgs>,
    queue: Res<InferenceQueue>,
    exit_state: Res<ExitState>,
    mut pause_state: ResMut<InferenceRenderPauseState>,
    winit_settings: Option<ResMut<WinitSettings>>,
    event_loop_proxy: Option<Res<EventLoopProxyWrapper>>,
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
        let _ = proxy.send_event(WinitUserEvent::WakeUp);
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
    event_loop_proxy: Option<Res<EventLoopProxyWrapper>>,
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
    event_loop_proxy: Option<Res<EventLoopProxyWrapper>>,
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
    event_loop_proxy: Option<&Res<EventLoopProxyWrapper>>,
) -> Option<WorkerWakeCallback> {
    if !args.pause_render_during_inference {
        return None;
    }
    let proxy = event_loop_proxy.map(|proxy| EventLoopProxy::clone(&**proxy))?;
    Some(std::sync::Arc::new(move || {
        let _ = proxy.send_event(WinitUserEvent::WakeUp);
    }))
}

#[cfg(all(not(target_arch = "wasm32"), feature = "wgpu"))]
pub(crate) fn should_share_wgpu_inference_device(args: &AppArgs) -> bool {
    should_share_wgpu_inference_device_for_platform(args, cfg!(target_os = "linux"))
}

#[cfg(all(not(target_arch = "wasm32"), feature = "wgpu"))]
pub(crate) fn should_share_wgpu_inference_device_for_platform(
    args: &AppArgs,
    _is_linux: bool,
) -> bool {
    matches!(args.backend, BackendKind::Wgpu)
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Deserialize)]
struct McpSceneCommandEnvelope {
    #[serde(default)]
    session_id: Option<String>,
    #[serde(default)]
    sequence: Option<u64>,
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
    SpawnPath {
        path: PathBuf,
        #[serde(default)]
        cache_key: Option<String>,
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
    ClearScene,
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
        #[serde(default)]
        vertical_fov: Option<f32>,
    },
    CaptureScreenshot {
        path: PathBuf,
    },
    SetInteractionLock {
        locked: bool,
        #[serde(default)]
        reason: Option<String>,
    },
    ReloadCache,
    SaveCache,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Serialize)]
struct McpSceneStatus {
    session_id: Option<String>,
    last_sequence: Option<u64>,
    ok: bool,
    message: String,
    requested_commands: usize,
    applied_commands: usize,
    command_results: Vec<McpSceneCommandResult>,
    cache_entries: Vec<CachedMeshMetadata>,
    world_items: Vec<CachedWorldItem>,
    projected_items: Vec<McpProjectedWorldItem>,
    camera: Option<CachedCameraState>,
    screenshots: Vec<String>,
    interaction_locked: bool,
    interaction_lock_reason: Option<String>,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Serialize)]
struct McpProjectedWorldItem {
    cache_key: String,
    world_aabb: Option<CachedAssetAabb>,
    screen_bbox: Option<[f32; 4]>,
    screen_contact: Option<[f32; 2]>,
    projected_corners: usize,
    total_corners: usize,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Serialize)]
struct McpSceneCommandResult {
    index: usize,
    command_type: &'static str,
    applied: bool,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    path: Option<String>,
}

#[derive(SystemParam)]
pub(crate) struct InferenceContext<'w, 's> {
    commands: Commands<'w, 's>,
    queue: ResMut<'w, InferenceQueue>,
    worker: Res<'w, InferenceWorker>,
    args: Res<'w, AppArgs>,
    #[cfg(not(target_arch = "wasm32"))]
    dispatch_gate: ResMut<'w, InferenceDispatchGate>,
    meshes: ResMut<'w, Assets<BevyMesh>>,
    images: ResMut<'w, Assets<Image>>,
    materials: ResMut<'w, Assets<StandardMaterial>>,
    gaussian_clouds: ResMut<'w, Assets<PlanarGaussian3d>>,
    cache: ResMut<'w, MeshCacheResource>,
    status: ResMut<'w, UiStatus>,
    catalog: ResMut<'w, CatalogState>,
    exit_state: Res<'w, ExitState>,
    #[cfg(target_arch = "wasm32")]
    wasm_startup: ResMut<'w, WasmStartupGate>,
}

pub(crate) fn run() {
    let args = Args::parse();
    let mut app_args = build_app_args(args);
    #[cfg(target_arch = "wasm32")]
    apply_wasm_url_overrides(&mut app_args);
    #[cfg(not(target_arch = "wasm32"))]
    if let Err(err) = prepare_startup_bsn_scene(&mut app_args) {
        eprintln!("failed to prepare BSN scene viewer: {err}");
        std::process::exit(1);
    }
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
        .insert_resource(PendingSceneBsnSave::default())
        .insert_resource(SceneInteractionLock::default())
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
    app.init_resource::<InferenceDispatchGate>();
    #[cfg(not(target_arch = "wasm32"))]
    app.insert_resource(mcp_scene_control);
    add_default_plugins(&mut app);
    #[cfg(all(not(target_arch = "wasm32"), feature = "wgpu"))]
    app.add_plugins(SharedWgpuInferenceDevicePlugin);
    if !app.is_plugin_added::<PointerInputPlugin>() {
        app.add_plugins(DefaultPickingPlugins);
    }
    app.add_plugins(TransformGizmoPlugin);
    app.add_plugins(PanOrbitCameraPlugin);
    app.add_plugins(InfiniteGridPlugin);
    app.add_plugins(GaussianSplattingPlugin);
    app.add_plugins(BurnSynthUiPlugin);
    app.add_systems(
        Update,
        (
            handle_exit_requests,
            handle_open_file_dialog,
            handle_file_dialog_loads,
            handle_dropped_files,
            handle_scene_save_requests,
            handle_scene_bsn_save_results,
            handle_scene_glb_save_results,
            #[cfg(target_arch = "wasm32")]
            kickoff_wasm_warmup.before(finish_wasm_startup_when_models_ready),
            #[cfg(target_arch = "wasm32")]
            finish_wasm_startup_when_models_ready.before(drive_inference),
            #[cfg(not(target_arch = "wasm32"))]
            sync_inference_render_pause_before.before(drive_inference),
            drive_inference,
            #[cfg(not(target_arch = "wasm32"))]
            sync_inference_render_pause_after.after(drive_inference),
            handle_catalog_spawn_requests.after(BurnSynthUiSystemSet::CatalogRequests),
            handle_catalog_delete_requests.after(BurnSynthUiSystemSet::CatalogRequests),
            delete_selected_meshes,
            (mark_world_cache_dirty, persist_world_cache).chain(),
            update_spinner,
            rotate_spinner,
            update_window_title,
        ),
    );
    app.add_systems(
        PostUpdate,
        enforce_scene_interaction_lock.before(TransformGizmoSystems::Main),
    );
    app.add_systems(
        PostUpdate,
        (sync_panorbit_bindings, sync_panorbit_enabled).before(PanOrbitCameraSystemSet),
    );
    app.add_plugins(
        FileDialogPlugin::new()
            .with_load_file::<ImagePickDialog>()
            .with_drop_file::<ImagePickDialog>()
            .with_save_file::<SceneBsnSaveDialog>()
            .with_save_file::<SceneGlbSaveDialog>(),
    );

    #[cfg(not(target_arch = "wasm32"))]
    {
        app.add_systems(Startup, (setup, configure_mesh_picking).chain());
        app.add_systems(
            PostUpdate,
            (
                sync_gaussian_splat_pick_bounds,
                remove_entity_from_selection_if_despawned,
                update_selection_from_primary_click
                    .after(TransformGizmoSystems::Main)
                    .after(sync_gaussian_splat_pick_bounds),
            ),
        );
    }

    #[cfg(target_arch = "wasm32")]
    {
        app.add_systems(Startup, (setup, configure_mesh_picking).chain());
        app.add_systems(
            PostUpdate,
            (
                sync_gaussian_splat_pick_bounds,
                remove_entity_from_selection_if_despawned,
                update_selection_from_primary_click
                    .after(TransformGizmoSystems::Main)
                    .after(sync_gaussian_splat_pick_bounds),
            ),
        );
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
    let output_path = resolve_asset_output_path(args, &image_path, 0)
        .ok_or_else(|| "headless mode requires --output".to_string())?;

    println!(
        "running headless bevy_synth inference: input={} output={}",
        image_path.display(),
        output_path.display()
    );

    let start = Instant::now();
    let asset = run_headless_once_inference(args, &image_path)?;
    let asset_kind = synth_asset_kind(&asset);

    write_synthesis_asset(&output_path, asset)
        .map_err(|err| format!("failed to write {}: {err}", output_path.display()))?;
    println!(
        "headless synthesis completed: asset_kind={} total_ms={:.1} -> {}",
        asset_kind,
        start.elapsed().as_secs_f64() * 1000.0,
        output_path.display()
    );
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn run_headless_once_inference(args: &AppArgs, image_path: &Path) -> Result<SynthAsset, String> {
    let worker = start_worker(args);
    let request = InferenceRequest {
        id: 0,
        image_path: image_path.to_path_buf(),
        image_contents: None,
        output_path: None,
        synthesis_models: args.synthesis_models.clone(),
        settings: InferenceSettings::from_args(args),
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
        Ok(Some(asset)) => Ok(asset),
        Ok(None) => Err(format!(
            "synthesis produced an empty asset for {}",
            image_path.display()
        )),
        Err(err) => Err(format!(
            "synthesis inference failed for {}: {err}",
            image_path.display()
        )),
    }
}

fn primary_synthesis_model_is_triposplat(args: &AppArgs) -> bool {
    args.synthesis_models
        .first()
        .is_some_and(|model| matches!(model, SynthesisModel::Triposplat))
}

fn resolve_asset_output_path(args: &AppArgs, image_path: &Path, index: u32) -> Option<PathBuf> {
    if primary_synthesis_model_is_triposplat(args) {
        resolve_gaussian_splat_output_path(args.output.as_ref(), image_path, index)
    } else {
        resolve_output_path(args.output.as_ref(), image_path, index)
    }
}

fn resolve_gaussian_splat_output_path(
    output: Option<&PathBuf>,
    image_path: &Path,
    index: u32,
) -> Option<PathBuf> {
    let output = output?;
    if output.extension().is_none() || output.is_dir() {
        let dir = output.to_path_buf();
        let stem = image_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("splats");
        let suffix = if index == 0 {
            String::new()
        } else {
            format!("_{index}")
        };
        return Some(dir.join(format!("{stem}{suffix}.splat")));
    }

    let output = if is_gaussian_splat_output_path(output) {
        output.clone()
    } else {
        output.with_extension("splat")
    };

    if index == 0 {
        return Some(output);
    }
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    let stem = output
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("splats");
    let ext = output
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("splat");
    Some(parent.join(format!("{stem}_{index}.{ext}")))
}

fn is_gaussian_splat_output_path(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|s| s.to_str())
            .map(|ext| ext.to_ascii_lowercase()),
        Some(ext) if matches!(ext.as_str(), "splat" | "ply")
    )
}

#[cfg(not(target_arch = "wasm32"))]
fn write_synthesis_asset(path: &Path, asset: SynthAsset) -> Result<(), String> {
    match asset {
        SynthAsset::Mesh(mesh) => write_glb(path, &mesh).map_err(|err| err.to_string()),
        SynthAsset::GaussianSplat(splats) => write_gaussian_splat_output(path, &splats),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn synth_asset_kind(asset: &SynthAsset) -> &'static str {
    match asset {
        SynthAsset::Mesh(_) => "mesh",
        SynthAsset::GaussianSplat(_) => "gaussian_splat",
    }
}

fn write_gaussian_splat_output(path: &Path, splats: &GaussianSplatCloud) -> Result<(), String> {
    match path
        .extension()
        .and_then(|s| s.to_str())
        .map(|ext| ext.to_ascii_lowercase())
        .as_deref()
    {
        Some("splat") => splats.write_splat(path),
        Some("ply") => splats.write_ply(path),
        Some(ext) => Err(format!(
            "Gaussian splat output requires .splat or .ply extension, got .{ext}"
        )),
        None => Err("Gaussian splat output requires .splat or .ply extension".to_string()),
    }
}

fn configure_mesh_picking(mut settings: ResMut<MeshPickingSettings>) {
    // Avoid ambiguous multi-camera rays by limiting picking to explicitly marked cameras/targets.
    settings.require_markers = true;
}

#[cfg(not(target_arch = "wasm32"))]
fn add_default_plugins(app: &mut App) {
    app.add_plugins(DefaultPlugins.set(AssetPlugin {
        // Native users and MCP agents spawn generated assets from tmp/run dirs,
        // desktop file pickers, and cache paths outside Bevy's asset root.
        unapproved_path_mode: UnapprovedPathMode::Allow,
        ..default()
    }));
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
    let webgpu_render_plugin = RenderPlugin {
        // Force browser WebGPU backend for Bevy rendering on wasm.
        // This prevents silent fallback to WebGL2 and keeps renderer/inference backend parity.
        render_creation: WgpuSettings {
            backends: Some(Backends::BROWSER_WEBGPU),
            ..default()
        }
        .into(),
        ..default()
    };
    app.add_plugins(
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: Some(Window {
                    fit_canvas_to_parent: true,
                    ..default()
                }),
                ..default()
            })
            .set(WebAssetPlugin {
                silence_startup_warning: true,
            })
            .set(webgpu_render_plugin)
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
fn parse_quality_override(value: &str) -> Option<QualityPreset> {
    match value.trim().to_ascii_lowercase().as_str() {
        "fast" => Some(QualityPreset::Fast),
        "balanced" | "balance" => Some(QualityPreset::Balanced),
        "full" => Some(QualityPreset::Full),
        _ => None,
    }
}

#[cfg(target_arch = "wasm32")]
fn parse_triposplat_profile_override(value: &str) -> Option<TripoSplatProfile> {
    match value.trim().to_ascii_lowercase().as_str() {
        "low" | "fast" => Some(TripoSplatProfile::Low),
        "balanced" | "balance" | "default" => Some(TripoSplatProfile::Balanced),
        "high" | "full" => Some(TripoSplatProfile::High),
        "custom" => Some(TripoSplatProfile::Custom),
        _ => None,
    }
}

#[cfg(target_arch = "wasm32")]
fn parse_synthesis_model_override(value: &str) -> Option<SynthesisModel> {
    match value.trim().to_ascii_lowercase().as_str() {
        "triposg" | "tripo" => Some(SynthesisModel::Triposg),
        "triposplat" | "tripo-splat" | "splat" => Some(SynthesisModel::Triposplat),
        _ => None,
    }
}

#[cfg(target_arch = "wasm32")]
fn parse_synthesis_models_override(value: &str) -> Option<Vec<SynthesisModel>> {
    let mut models = Vec::new();
    for item in value.split(',') {
        if let Some(model) = parse_synthesis_model_override(item)
            && !models.contains(&model)
        {
            models.push(model);
        }
    }
    (!models.is_empty()).then_some(models)
}

#[cfg(target_arch = "wasm32")]
fn parse_rmbg_model_override(value: &str) -> Option<RmbgModel> {
    match value.trim().to_ascii_lowercase().as_str() {
        "rmbg14" | "rmbg-1.4" | "bria-rmbg-1.4" => Some(RmbgModel::Rmbg14),
        _ => None,
    }
}

#[cfg(target_arch = "wasm32")]
fn apply_quality_override(args: &mut AppArgs, quality: QualityPreset) {
    let defaults = quality.defaults();
    args.quality = quality;
    args.num_steps = defaults.num_steps;
    args.num_tokens = defaults.num_tokens;
    args.guidance_scale = defaults.guidance_scale;
    args.resolution = defaults.resolution;
    args.chunk_size = defaults.chunk_size;
    args.dense_octree_depth = defaults.dense_octree_depth;
    args.hierarchical_octree_depth = defaults.hierarchical_octree_depth;
    args.band_threshold = defaults.band_threshold;
    args.flash_octree_depth = defaults.flash_octree_depth;
    args.flash_min_resolution = defaults.flash_min_resolution;
    args.flash_mini_grid_num = defaults.flash_mini_grid_num;
    args.flash_num_chunks = defaults.flash_num_chunks;
    args.flash_mc_level = defaults.flash_mc_level;
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

    let mut quality_override: Option<QualityPreset> = None;
    let mut triposplat_profile_override: Option<TripoSplatProfile> = None;
    let mut synthesis_models_override: Option<Vec<SynthesisModel>> = None;
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

        if key == "quality" {
            if let Some(quality) = parse_quality_override(value) {
                quality_override = Some(quality);
            }
            continue;
        }

        if matches!(key.as_str(), "triposplat_profile" | "splat_profile") {
            if let Some(profile) = parse_triposplat_profile_override(value) {
                triposplat_profile_override = Some(profile);
            }
            continue;
        }

        if matches!(key.as_str(), "synthesis_model" | "synthesis")
            && let Some(model) = parse_synthesis_model_override(value)
        {
            synthesis_models_override = Some(vec![model]);
            continue;
        }

        if key == "synthesis_models"
            && let Some(models) = parse_synthesis_models_override(value)
        {
            synthesis_models_override = Some(models);
            continue;
        }

        if matches!(key.as_str(), "rmbg_model" | "foreground_model")
            && let Some(model) = parse_rmbg_model_override(value)
        {
            args.rmbg_model = model;
            continue;
        }

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

    if let Some(quality) = quality_override {
        apply_quality_override(args, quality);
    }
    if let Some(profile) = triposplat_profile_override {
        args.apply_triposplat_profile(profile);
    }
    if let Some(models) = synthesis_models_override {
        args.synthesis_models = models.clone();
        args.available_synthesis_models = models;
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

#[cfg(target_arch = "wasm32")]
fn wasm_set_warmup_message(message: &str) {
    if let Some(window) = web_sys::window() {
        let window_js: wasm_bindgen::JsValue = window.into();
        let _ = js_sys::Reflect::set(
            &window_js,
            &wasm_bindgen::JsValue::from_str("__bevySynthWarmupMessage"),
            &wasm_bindgen::JsValue::from_str(message),
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
            BackgroundColor(Color::srgba(0.03, 0.04, 0.06, 0.58)),
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
                    TextLayout::justify(Justify::Center),
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
    gaussian_clouds: &mut ResMut<Assets<PlanarGaussian3d>>,
    ambient_light: &mut ResMut<GlobalAmbientLight>,
    args: &AppArgs,
    queue: &mut ResMut<InferenceQueue>,
    status: &mut ResMut<UiStatus>,
    catalog: &mut ResMut<CatalogState>,
    cache: &mut ResMut<MeshCacheResource>,
    world_cache: &mut ResMut<WorldCachePersistence>,
) {
    let mut camera_transform =
        Transform::from_translation(Vec3::new(0.0, 1.5, 5.0)).looking_at(Vec3::ZERO, Vec3::Y);
    let mut camera_orbit = PanOrbitCamera::default();
    if let Some(cached_camera) = cache.cache.camera_state()
        && !apply_cached_camera_state(cached_camera, &mut camera_transform, &mut camera_orbit)
    {
        warn!("ignoring cached camera state due to invalid values.");
    }

    commands.spawn((
        Camera3d::default(),
        GaussianCamera::default(),
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
                scale: 1.0,
                fadeout_distance: 90.0,
                dot_fadeout_strength: 0.22,
                minor_line_color: Color::srgba(0.32, 0.36, 0.44, 0.24),
                major_line_color: Color::srgba(0.60, 0.66, 0.76, 0.42),
                x_axis_color: Color::srgb(0.92, 0.24, 0.22),
                z_axis_color: Color::srgb(0.20, 0.42, 0.95),
            },
            ..default()
        },
        Pickable::IGNORE,
        RenderLayers::layer(0),
    ));

    hydrate_from_cache(
        commands,
        meshes,
        images,
        materials,
        gaussian_clouds,
        queue,
        catalog,
        cache,
    );
    seed_default_catalog_cube(meshes, materials, queue, catalog, cache);
    world_cache.dirty = false;
    world_cache.timer.reset();

    if let Some(mesh_path) = args.mesh.as_ref() {
        if mesh_path.exists() {
            if let Err(err) = spawn_mesh_asset(
                commands,
                asset_server,
                meshes,
                images,
                materials,
                mesh_path.clone(),
            ) {
                warn!("failed to load mesh path {}: {err}", mesh_path.display());
            }
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
    mut gaussian_clouds: ResMut<Assets<PlanarGaussian3d>>,
    mut ambient_light: ResMut<GlobalAmbientLight>,
    args: Res<AppArgs>,
    #[cfg(all(not(target_arch = "wasm32"), feature = "wgpu"))] shared_wgpu_device: Option<
        Res<SharedWgpuInferenceDevice>,
    >,
    #[cfg(not(target_arch = "wasm32"))] event_loop_proxy: Option<Res<EventLoopProxyWrapper>>,
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
        &mut gaussian_clouds,
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
fn setup(
    mut commands: Commands,
    args: Res<AppArgs>,
    mut status: ResMut<UiStatus>,
    mut startup: ResMut<WasmStartupGate>,
) {
    info!("bevy_synth args: {:?}", *args);
    wasm_set_warmup_state("ready");
    wasm_set_warmup_message("App ready. Open or drop an image to run inference.");

    let worker = start_worker(args.as_ref());
    commands.insert_resource(worker);

    startup.model_ready = true;
    status.worker_message = None;
    status.message = "upload an image (.png/.jpg) to begin.".to_string();
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
            wasm_set_warmup_message(WASM_STATUS_LOADING_MODELS);
            wasm_console_log("bevy_synth wasm warmup: requested");
        }
        Err(err) => {
            let message = format!("failed to request wasm model warmup: {err}");
            status.worker_message = Some(message.clone());
            status.message = message;
            wasm_set_warmup_state("failed");
            wasm_set_warmup_message(&status.message);
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
    mut gaussian_clouds: ResMut<Assets<PlanarGaussian3d>>,
    mut ambient_light: ResMut<GlobalAmbientLight>,
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
        &mut gaussian_clouds,
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

fn spawn_default_lighting(commands: &mut Commands, ambient_light: &mut GlobalAmbientLight) {
    // Keep a modest ambient base and drive shape with a single sun + soft point fills.
    // Web targets commonly support only one directional light in forward mode.
    ambient_light.color = Color::srgb(0.86, 0.9, 0.96);
    ambient_light.brightness = 260.0;

    commands.spawn((
        DirectionalLight {
            color: Color::srgb(1.0, 0.98, 0.95),
            illuminance: 24_000.0,
            shadow_maps_enabled: true,
            // Slightly higher bias to reduce self-shadow acne on simple hard-edge meshes (e.g. cube).
            shadow_depth_bias: 0.24,
            shadow_normal_bias: 1.8,
            ..default()
        },
        Transform::from_xyz(7.5, 11.0, 8.5).looking_at(Vec3::new(0.0, 0.4, 0.0), Vec3::Y),
        CascadeShadowConfigBuilder {
            first_cascade_far_bound: 12.0,
            maximum_distance: 56.0,
            ..default()
        }
        .build(),
        preview_light_layers(),
    ));

    commands.spawn((
        PointLight {
            color: Color::srgb(0.76, 0.86, 1.0),
            intensity: 90_000.0,
            range: 34.0,
            radius: 0.45,
            ..default()
        },
        Transform::from_xyz(-9.0, 5.5, -7.0),
        preview_light_layers(),
    ));

    commands.spawn((
        PointLight {
            color: Color::srgb(1.0, 0.94, 0.87),
            intensity: 65_000.0,
            range: 30.0,
            radius: 0.4,
            ..default()
        },
        Transform::from_xyz(8.5, 4.5, -6.5),
        preview_light_layers(),
    ));

    commands.spawn((
        PointLight {
            color: Color::srgb(0.94, 0.97, 1.0),
            intensity: 38_000.0,
            range: 22.0,
            radius: 0.35,
            ..default()
        },
        Transform::from_xyz(0.0, 9.0, 0.0),
        preview_light_layers(),
    ));
}

#[derive(Clone)]
enum CachedAssetHandles {
    Mesh {
        mesh: Handle<BevyMesh>,
        material: Handle<StandardMaterial>,
    },
    GaussianSplat {
        cloud: Handle<PlanarGaussian3d>,
    },
}

#[allow(clippy::too_many_arguments)]
fn hydrate_from_cache(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<BevyMesh>>,
    images: &mut ResMut<Assets<Image>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    gaussian_clouds: &mut ResMut<Assets<PlanarGaussian3d>>,
    queue: &mut ResMut<InferenceQueue>,
    catalog: &mut ResMut<CatalogState>,
    cache: &mut ResMut<MeshCacheResource>,
) {
    // Ensure older caches with smooth-shaded built-in cube topology are migrated to flat-shaded cube topology.
    if let Err(err) = cache.cache.upsert_mesh_for_image(
        Path::new(BUILTIN_CUBE_SOURCE_IMAGE),
        &default_cube_synth_mesh(),
    ) {
        warn!("failed to refresh built-in cube cache entry: {err}");
    }

    let asset_entries = cache.cache.asset_entries().to_vec();
    let world_items = cache.cache.world_items().to_vec();
    if asset_entries.is_empty() && world_items.is_empty() {
        return;
    }

    let mut loaded_assets = 0usize;
    let mut loaded_world_items = 0usize;
    let mut handles_by_key: HashMap<String, CachedAssetHandles> = HashMap::new();

    for metadata in asset_entries {
        let asset = match cache.cache.load_asset(&metadata.cache_key) {
            Ok(Some(asset)) => asset,
            Ok(None) => {
                warn!(
                    "cache metadata exists for key {} but asset payload is missing.",
                    metadata.cache_key
                );
                continue;
            }
            Err(err) => {
                warn!(
                    "failed to load cached asset for key {}: {err}",
                    metadata.cache_key
                );
                continue;
            }
        };

        let Some(handles) =
            cached_asset_handles(asset, &metadata, meshes, images, materials, gaussian_clouds)
        else {
            continue;
        };
        handles_by_key.insert(metadata.cache_key.clone(), handles.clone());

        let entry_id = queue.counter;
        queue.counter = queue.counter.wrapping_add(1);
        match handles {
            CachedAssetHandles::Mesh { mesh, material } => {
                catalog.add_ready(
                    entry_id,
                    metadata.label.clone(),
                    mesh,
                    material,
                    Some(metadata.source_image_path.clone()),
                    Some(metadata.cache_key.clone()),
                );
            }
            CachedAssetHandles::GaussianSplat { cloud } => {
                catalog.add_ready_gaussian_splat(
                    entry_id,
                    metadata.label.clone(),
                    cloud,
                    Some(metadata.source_image_path.clone()),
                    Some(metadata.cache_key.clone()),
                );
            }
        }
        let source_image = cached_source_image_handle(&cache.cache, &metadata, images.as_mut());
        catalog.set_source_image(entry_id, source_image);
        loaded_assets += 1;
    }

    for item in world_items {
        let Some(handles) = handles_by_key.get(&item.cache_key) else {
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
        spawn_cached_asset_instance(commands, handles, transform, Some(item.cache_key.clone()));
        loaded_world_items += 1;
    }

    if loaded_assets > 0 {
        queue.completed = queue.completed.max(loaded_assets);
    }
    if loaded_assets > 0 || loaded_world_items > 0 {
        info!(
            "loaded {loaded_assets} cached catalog asset(s) and {loaded_world_items} cached world item(s)."
        );
    }
}

fn cached_asset_handles(
    asset: SynthAsset,
    metadata: &CachedMeshMetadata,
    meshes: &mut ResMut<Assets<BevyMesh>>,
    images: &mut ResMut<Assets<Image>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    gaussian_clouds: &mut ResMut<Assets<PlanarGaussian3d>>,
) -> Option<CachedAssetHandles> {
    match asset {
        SynthAsset::Mesh(mesh) => {
            let mesh_handle = meshes.add(to_bevy_mesh_synth(&mesh));
            let material = if metadata.source_image_path == BUILTIN_CUBE_SOURCE_IMAGE {
                materials.add(default_cube_material())
            } else {
                materials.add(standard_material_for_inference(&mesh, images.as_mut()))
            };
            Some(CachedAssetHandles::Mesh {
                mesh: mesh_handle,
                material,
            })
        }
        SynthAsset::GaussianSplat(splats) => {
            match gaussian_splat_cloud_handle(&splats, gaussian_clouds) {
                Ok(cloud) => Some(CachedAssetHandles::GaussianSplat { cloud }),
                Err(err) => {
                    warn!(
                        "failed to build cached TripoSplat Gaussian cloud for key {}: {err}",
                        metadata.cache_key
                    );
                    None
                }
            }
        }
    }
}

fn cached_source_image_handle(
    cache: &MeshCache,
    metadata: &CachedMeshMetadata,
    images: &mut Assets<Image>,
) -> Option<Handle<Image>> {
    match cache.load_source_image(metadata) {
        Ok(Some(source)) => match image_bytes_to_bevy_image(source.bytes.as_slice()) {
            Ok(image) => Some(images.add(image)),
            Err(err) => {
                warn!(
                    "failed to decode cached source image for key {}: {err}",
                    metadata.cache_key
                );
                None
            }
        },
        Ok(None) => None,
        Err(err) => {
            warn!(
                "failed to load cached source image for key {}: {err}",
                metadata.cache_key
            );
            None
        }
    }
}

fn seed_default_catalog_cube(
    meshes: &mut ResMut<Assets<BevyMesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    queue: &mut ResMut<InferenceQueue>,
    catalog: &mut ResMut<CatalogState>,
    cache: &mut ResMut<MeshCacheResource>,
) {
    if catalog.has_ready_cube_entry() {
        return;
    }

    let cached_metadata = match cache.cache.upsert_mesh_for_image(
        Path::new(BUILTIN_CUBE_SOURCE_IMAGE),
        &default_cube_synth_mesh(),
    ) {
        Ok(metadata) => Some(metadata),
        Err(err) => {
            warn!("Failed to cache built-in cube mesh: {err}");
            None
        }
    };

    let mesh = meshes.add(BevyMesh::from(Cuboid::from_size(Vec3::ONE)));
    let material = materials.add(default_cube_material());
    let id = queue.counter;
    queue.counter = queue.counter.wrapping_add(1);
    catalog.add_ready(
        id,
        "cube".to_string(),
        mesh,
        material,
        cached_metadata
            .as_ref()
            .map(|metadata| metadata.source_image_path.clone()),
        cached_metadata
            .as_ref()
            .map(|metadata| metadata.cache_key.clone()),
    );
}

fn default_cube_synth_mesh() -> SynthMesh {
    SynthMesh::from(TripoMesh {
        vertices: vec![
            // -Z face
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            // +Z face
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
            // -X face
            [-0.5, -0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, 0.5, 0.5],
            [-0.5, -0.5, 0.5],
            // +X face
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5],
            // -Y face
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, -0.5, 0.5],
            [-0.5, -0.5, 0.5],
            // +Y face
            [-0.5, 0.5, -0.5],
            [0.5, 0.5, -0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ],
        faces: vec![
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [8, 10, 9],
            [8, 11, 10],
            [12, 13, 14],
            [12, 14, 15],
            [16, 17, 18],
            [16, 18, 19],
            [20, 22, 21],
            [20, 23, 22],
        ],
    })
}

fn default_cube_material() -> StandardMaterial {
    StandardMaterial {
        base_color: Color::srgb(0.8, 0.84, 0.9),
        perceptual_roughness: 0.58,
        ..default()
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

fn camera_state_from_components(
    transform: &Transform,
    orbit: &PanOrbitCamera,
    projection: Option<&Projection>,
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
        vertical_fov_degrees: projection.and_then(|projection| match projection {
            Projection::Perspective(perspective) => Some(perspective.fov.to_degrees()),
            _ => None,
        }),
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
    orbit.initialized = true;
    true
}

fn update_panorbit_camera(
    time: Res<Time>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut motion_events: MessageReader<MouseMotion>,
    mut wheel_events: MessageReader<MouseWheel>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut cameras: Query<
        (
            &Camera,
            &mut Projection,
            &mut Transform,
            &mut PanOrbitCamera,
        ),
        With<MainCamera>,
    >,
) {
    let motion = motion_events
        .read()
        .fold(Vec2::ZERO, |acc, event| acc + event.delta);
    let (scroll_line, scroll_pixel) =
        wheel_events
            .read()
            .fold((0.0f32, 0.0f32), |acc, event| match event.unit {
                MouseScrollUnit::Line => (acc.0 + event.y, acc.1),
                MouseScrollUnit::Pixel => (acc.0, acc.1 + event.y * 0.005),
            });
    let has_motion = motion.length_squared() > 0.0;
    let has_scroll = (scroll_line + scroll_pixel).abs() > 0.0;
    let dt = time.delta_secs();
    let window_size = windows
        .single()
        .ok()
        .map(|window| Vec2::new(window.width(), window.height()));

    for (camera, mut projection, mut transform, mut orbit) in cameras.iter_mut() {
        if !orbit.initialized {
            initialize_panorbit_from_transform(&mut transform, &mut projection, &mut orbit);
        }

        let mut has_moved = false;

        if orbit.enabled {
            if has_motion
                && buttons.pressed(orbit.button_orbit)
                && let Some(window_size) = window_size.filter(|size| size.x > 0.0 && size.y > 0.0)
            {
                let delta_x = motion.x / window_size.x * std::f32::consts::TAU;
                let delta_x = if orbit.is_upside_down {
                    -delta_x
                } else {
                    delta_x
                };
                let delta_y = motion.y / window_size.y * std::f32::consts::PI;
                orbit.target_yaw -= delta_x;
                orbit.target_pitch += delta_y;
                has_moved = true;
            }

            if has_motion && buttons.pressed(orbit.button_pan) {
                let viewport_size = camera
                    .logical_viewport_size()
                    .or(window_size)
                    .filter(|size| size.x > 0.0 && size.y > 0.0);
                if let Some(viewport_size) = viewport_size {
                    let mut pan = motion;
                    let mut multiplier = 1.0;
                    match projection.as_ref() {
                        Projection::Perspective(perspective) => {
                            pan *= Vec2::new(
                                perspective.fov * perspective.aspect_ratio,
                                perspective.fov,
                            ) / viewport_size;
                            multiplier = orbit.target_radius.max(PANORBIT_MIN_RADIUS);
                        }
                        Projection::Orthographic(orthographic) => {
                            pan *= Vec2::new(orthographic.area.width(), orthographic.area.height())
                                / viewport_size;
                        }
                        Projection::Custom(_) => {
                            pan *= Vec2::splat(1.0 / viewport_size.y);
                            multiplier = orbit.target_radius.max(PANORBIT_MIN_RADIUS);
                        }
                    }
                    let right = transform.rotation * Vec3::X * -pan.x;
                    let up = transform.rotation * Vec3::Y * pan.y;
                    orbit.target_focus += (right + up) * multiplier;
                    has_moved = true;
                }
            }

            if has_scroll {
                let line_delta = -scroll_line * orbit.target_radius * 0.2;
                let pixel_delta = -scroll_pixel * orbit.target_radius * 0.2;
                orbit.target_radius += line_delta + pixel_delta;
                if let Some(radius) = orbit.radius.as_mut() {
                    *radius =
                        (*radius + pixel_delta).clamp(PANORBIT_MIN_RADIUS, PANORBIT_MAX_RADIUS);
                }
                has_moved = true;
            }
        }

        if buttons.just_pressed(orbit.button_orbit) || buttons.just_released(orbit.button_orbit) {
            orbit.is_upside_down = transform.up().dot(Vec3::Y) < 0.0;
        }

        orbit.target_radius = orbit
            .target_radius
            .clamp(PANORBIT_MIN_RADIUS, PANORBIT_MAX_RADIUS);
        if !orbit.allow_upside_down {
            orbit.target_pitch = orbit
                .target_pitch
                .clamp(-std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2);
        }

        let (Some(yaw), Some(pitch), Some(radius)) = (orbit.yaw, orbit.pitch, orbit.radius) else {
            continue;
        };
        if !has_moved
            && orbit.target_yaw == yaw
            && orbit.target_pitch == pitch
            && orbit.target_radius == radius
            && orbit.target_focus == orbit.focus
        {
            continue;
        }

        let new_yaw =
            panorbit_lerp_and_snap_f32(yaw, orbit.target_yaw, PANORBIT_ORBIT_SMOOTHNESS, dt);
        let new_pitch =
            panorbit_lerp_and_snap_f32(pitch, orbit.target_pitch, PANORBIT_ORBIT_SMOOTHNESS, dt);
        let new_radius =
            panorbit_lerp_and_snap_f32(radius, orbit.target_radius, PANORBIT_ZOOM_SMOOTHNESS, dt);
        let new_focus = panorbit_lerp_and_snap_vec3(
            orbit.focus,
            orbit.target_focus,
            PANORBIT_PAN_SMOOTHNESS,
            dt,
        );

        update_panorbit_transform(
            new_yaw,
            new_pitch,
            new_radius,
            new_focus,
            &mut transform,
            &mut projection,
        );
        orbit.yaw = Some(new_yaw);
        orbit.pitch = Some(new_pitch);
        orbit.radius = Some(new_radius);
        orbit.focus = new_focus;
    }
}

fn initialize_panorbit_from_transform(
    transform: &mut Transform,
    projection: &mut Projection,
    orbit: &mut PanOrbitCamera,
) {
    let focus = orbit.focus;
    let offset = transform.translation - focus;
    let radius = offset
        .length()
        .clamp(PANORBIT_MIN_RADIUS, PANORBIT_MAX_RADIUS);
    let direction = if radius > PANORBIT_MIN_RADIUS {
        offset / radius
    } else {
        Vec3::Z
    };
    let yaw = direction.x.atan2(direction.z);
    let pitch = direction.y.clamp(-1.0, 1.0).asin();
    orbit.yaw = Some(yaw);
    orbit.target_yaw = yaw;
    orbit.pitch = Some(pitch);
    orbit.target_pitch = pitch;
    orbit.radius = Some(radius);
    orbit.target_radius = radius;
    orbit.focus = focus;
    orbit.target_focus = focus;
    update_panorbit_transform(yaw, pitch, radius, focus, transform, projection);
    orbit.initialized = true;
}

fn update_panorbit_transform(
    yaw: f32,
    pitch: f32,
    mut radius: f32,
    focus: Vec3,
    transform: &mut Transform,
    projection: &mut Projection,
) {
    if let Projection::Orthographic(orthographic) = projection {
        orthographic.scale = radius;
        radius = (orthographic.near + orthographic.far) / 2.0;
    }
    let yaw_rot = Quat::from_axis_angle(Vec3::Y, yaw);
    let pitch_rot = Quat::from_axis_angle(Vec3::X, -pitch);
    transform.rotation = yaw_rot * pitch_rot;
    transform.translation = focus + transform.rotation * Vec3::new(0.0, 0.0, radius);
}

fn panorbit_lerp_and_snap_f32(from: f32, to: f32, smoothness: f32, dt: f32) -> f32 {
    let t = smoothness.powi(7);
    let mut value = from.lerp(to, 1.0 - t.powf(dt));
    if smoothness < 1.0 && (value - to).abs() < PANORBIT_SNAP_EPSILON {
        value = to;
    }
    value
}

fn panorbit_lerp_and_snap_vec3(from: Vec3, to: Vec3, smoothness: f32, dt: f32) -> Vec3 {
    let t = smoothness.powi(7);
    let mut value = from.lerp(to, 1.0 - t.powf(dt));
    if smoothness < 1.0 && (value - to).length() < PANORBIT_SNAP_EPSILON {
        value = to;
    }
    value
}

fn handle_open_file_dialog(
    keys: Res<ButtonInput<KeyCode>>,
    mut commands: Commands,
    exit_state: Res<ExitState>,
    interaction_lock: Res<SceneInteractionLock>,
) {
    if exit_state.requested || interaction_lock.locked {
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
    mut meshes: ResMut<Assets<BevyMesh>>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut queue: ResMut<InferenceQueue>,
    args: Res<AppArgs>,
    mut status: ResMut<UiStatus>,
    mut catalog: ResMut<CatalogState>,
    exit_state: Res<ExitState>,
    interaction_lock: Res<SceneInteractionLock>,
) {
    if exit_state.requested {
        return;
    }
    if interaction_lock.locked {
        for _ in events.read() {}
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
            &mut meshes,
            &mut images,
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
    mut meshes: ResMut<Assets<BevyMesh>>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut queue: ResMut<InferenceQueue>,
    args: Res<AppArgs>,
    mut status: ResMut<UiStatus>,
    mut catalog: ResMut<CatalogState>,
    exit_state: Res<ExitState>,
    interaction_lock: Res<SceneInteractionLock>,
) {
    if exit_state.requested {
        return;
    }
    if interaction_lock.locked {
        for _ in events.read() {}
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
            &mut meshes,
            &mut images,
            &mut materials,
            "dropped file",
        );
    }
    if queued > 0 {
        update_status_message(&args, &queue, &mut status);
    }
}

fn handle_scene_save_requests(
    mut requests: MessageReader<SceneSaveRequest>,
    mut commands: Commands,
    cache: Res<MeshCacheResource>,
    cached_query: Query<(&CachedMeshInstance, &Transform)>,
    camera_query: Query<(&Transform, &PanOrbitCamera), With<MainCamera>>,
    mut pending_bsn: ResMut<PendingSceneBsnSave>,
    mut status: ResMut<UiStatus>,
    interaction_lock: Res<SceneInteractionLock>,
) {
    let mut last_request = None;
    for request in requests.read() {
        last_request = Some(request.kind);
    }
    if interaction_lock.locked {
        return;
    }
    let Some(kind) = last_request else {
        return;
    };

    let world_items = collect_cached_world_items(&cached_query);
    let camera_state = camera_query
        .single()
        .ok()
        .and_then(|(transform, orbit)| camera_state_from_components(transform, orbit, None));

    match kind {
        SceneSaveKind::Bsn => {
            #[cfg(not(target_arch = "wasm32"))]
            match scene_bsn_export_for_world(&cache.cache, &world_items, camera_state) {
                Ok((bsn, assets_json)) => {
                    pending_bsn.assets_json = Some(assets_json);
                    commands
                        .dialog()
                        .set_title("save scene")
                        .set_file_name("scene.bsn")
                        .add_filter("BSN scene", &["bsn"])
                        .save_file::<SceneBsnSaveDialog>(bsn.into_bytes());
                }
                Err(err) => {
                    pending_bsn.assets_json = None;
                    status.message = format!("scene save failed: {err}");
                    warn!("{}", status.message);
                }
            }
            #[cfg(target_arch = "wasm32")]
            {
                let _ = camera_state;
                pending_bsn.assets_json = None;
                status.message =
                    "BSN scene save requires native sidecar file support; export GLB instead."
                        .to_string();
                warn!("{}", status.message);
            }
        }
        SceneSaveKind::Glb => match scene_glb_export_for_world(&cache.cache, &world_items) {
            Ok(glb) => {
                commands
                    .dialog()
                    .set_title("export scene")
                    .set_file_name("scene.glb")
                    .add_filter("GLB scene", &["glb"])
                    .save_file::<SceneGlbSaveDialog>(glb);
            }
            Err(err) => {
                status.message = format!("GLB export failed: {err}");
                warn!("{}", status.message);
            }
        },
    }
}

fn handle_scene_bsn_save_results(
    mut saved: MessageReader<DialogFileSaved<SceneBsnSaveDialog>>,
    mut canceled: MessageReader<DialogFileSaveCanceled<SceneBsnSaveDialog>>,
    mut pending: ResMut<PendingSceneBsnSave>,
    mut status: ResMut<UiStatus>,
) {
    let mut canceled_any = false;
    for _ in canceled.read() {
        canceled_any = true;
    }
    if canceled_any {
        pending.assets_json = None;
        status.message = "scene save canceled.".to_string();
    }

    for event in saved.read() {
        match &event.result {
            Ok(()) => {
                #[cfg(not(target_arch = "wasm32"))]
                if let Some(path) = event.path()
                    && let Some(assets_json) = pending.assets_json.take()
                {
                    let sidecar = scene_assets_sidecar_path(path);
                    if let Err(err) = fs::write(&sidecar, assets_json) {
                        status.message = format!(
                            "saved {}, but failed to write {}: {err}",
                            event.file_name,
                            sidecar.display()
                        );
                        warn!("{}", status.message);
                        continue;
                    }
                    status.message = format!(
                        "saved {} and {}.",
                        event.file_name,
                        sidecar
                            .file_name()
                            .and_then(|name| name.to_str())
                            .unwrap_or("scene.assets.json")
                    );
                    continue;
                }
                pending.assets_json = None;
                status.message = format!("saved {}.", event.file_name);
            }
            Err(err) => {
                pending.assets_json = None;
                status.message = format!("scene save failed: {err}");
                warn!("{}", status.message);
            }
        }
    }
}

fn handle_scene_glb_save_results(
    mut saved: MessageReader<DialogFileSaved<SceneGlbSaveDialog>>,
    mut canceled: MessageReader<DialogFileSaveCanceled<SceneGlbSaveDialog>>,
    mut status: ResMut<UiStatus>,
) {
    for _ in canceled.read() {
        status.message = "GLB export canceled.".to_string();
    }
    for event in saved.read() {
        match &event.result {
            Ok(()) => {
                status.message = format!("exported {}.", event.file_name);
            }
            Err(err) => {
                status.message = format!("GLB export failed: {err}");
                warn!("{}", status.message);
            }
        }
    }
}

pub(crate) fn scene_glb_export_for_world(
    cache: &MeshCache,
    world_items: &[CachedWorldItem],
) -> Result<Vec<u8>, String> {
    if world_items.is_empty() {
        return Err("scene is empty".to_string());
    }

    let mut instances = Vec::with_capacity(world_items.len());
    for (index, item) in world_items.iter().enumerate() {
        let asset = cache
            .load_asset(&item.cache_key)
            .map_err(|err| err.to_string())?
            .ok_or_else(|| format!("cache entry {} is missing its payload", item.cache_key))?;
        match asset {
            SynthAsset::Mesh(mesh) => {
                instances.push(SceneGlbMeshInstance {
                    name: scene_entity_name(index, &item.cache_key),
                    mesh,
                    translation: item.translation,
                    rotation: item.rotation,
                    scale: item.scale,
                });
            }
            SynthAsset::GaussianSplat(_) => {
                return Err(
                    "GLB export currently supports mesh scenes only; use BSN for splats."
                        .to_string(),
                );
            }
        }
    }

    scene_meshes_to_glb_bytes(&instances).map_err(|err| err.to_string())
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn scene_bsn_export_for_world(
    cache: &MeshCache,
    world_items: &[CachedWorldItem],
    camera_state: Option<CachedCameraState>,
) -> Result<(String, Vec<u8>), String> {
    if world_items.is_empty() {
        return Err("scene is empty".to_string());
    }

    let (assets, asset_ids) = scene_asset_bindings_for_world(cache, world_items)?;
    let mut bsn = String::from("synth_scene_v1 {\n");
    for asset in &assets {
        bsn.push_str(&format!(
            "asset {} = \"cache:{}\";\n",
            asset.asset_id,
            asset
                .cache_key
                .as_deref()
                .unwrap_or(asset.asset_id.as_str())
        ));
    }
    for (index, item) in world_items.iter().enumerate() {
        let asset_id = asset_ids
            .get(&item.cache_key)
            .ok_or_else(|| format!("missing scene asset id for {}", item.cache_key))?;
        bsn.push_str(&format!(
            "spawn {} uses {} translation [{}] rotation_y {} scale [{}];\n",
            scene_entity_name(index, &item.cache_key),
            asset_id,
            fmt_scene_vec3(item.translation),
            fmt_scene_f32(rotation_y_degrees_from_quat(item.rotation)),
            fmt_scene_vec3(item.scale)
        ));
    }
    if let Some(camera) = camera_state.map(scene_camera_from_cached_state) {
        bsn.push_str(&format!(
            "camera translation [{}] focus [{}] yaw {} pitch {} radius {};\n",
            fmt_scene_vec3(camera.translation),
            fmt_scene_vec3(camera.focus),
            fmt_scene_f32(camera.yaw.unwrap_or(0.0)),
            fmt_scene_f32(camera.pitch.unwrap_or(0.0)),
            fmt_scene_f32(camera.radius.unwrap_or(0.0))
        ));
    }
    bsn.push_str("}\n");

    parse_scene_bsn(&bsn, &assets).map_err(|err| err.to_string())?;
    let assets_json = serde_json::to_vec_pretty(&assets).map_err(|err| err.to_string())?;
    Ok((bsn, assets_json))
}

#[cfg(not(target_arch = "wasm32"))]
fn scene_asset_bindings_for_world(
    cache: &MeshCache,
    world_items: &[CachedWorldItem],
) -> Result<(Vec<SceneAssetBinding>, HashMap<String, String>), String> {
    let mut assets = Vec::new();
    let mut asset_ids = HashMap::new();
    for item in world_items {
        if asset_ids.contains_key(&item.cache_key) {
            continue;
        }
        let metadata = cache
            .asset_entries()
            .iter()
            .find(|entry| entry.cache_key == item.cache_key)
            .ok_or_else(|| format!("cache entry {} is missing metadata", item.cache_key))?;
        let mut asset_id = format!("asset_{}", sanitize_scene_identifier(&item.cache_key));
        if asset_ids.values().any(|existing| existing == &asset_id) {
            asset_id = format!("{asset_id}_{}", assets.len() + 1);
        }
        asset_ids.insert(item.cache_key.clone(), asset_id.clone());
        assets.push(SceneAssetBinding {
            asset_id,
            object_id: sanitize_scene_identifier(&metadata.label),
            label: metadata.label.clone(),
            aliases: Vec::new(),
            path: None,
            cache_key: Some(metadata.cache_key.clone()),
            reusable: true,
            source_image_path: Some(metadata.source_image_path.clone()),
            pipeline: Some(scene_pipeline_label(metadata.asset_kind).to_string()),
            local_aabb: metadata.local_aabb.map(scene_aabb_from_cached),
            canonical_frame: metadata.canonical_frame.map(scene_frame_from_cached),
            provenance: None,
        });
    }
    Ok((assets, asset_ids))
}

#[cfg(not(target_arch = "wasm32"))]
fn scene_pipeline_label(kind: CachedAssetKind) -> &'static str {
    match kind {
        CachedAssetKind::Mesh => "mesh",
        CachedAssetKind::GaussianSplat => "gaussian_splat",
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn scene_aabb_from_cached(aabb: bevy_synth_runtime::cache::CachedAssetAabb) -> SceneAssetAabb {
    SceneAssetAabb {
        min: aabb.min,
        max: aabb.max,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn scene_frame_from_cached(frame: CachedAssetFrame) -> SceneAssetFrame {
    SceneAssetFrame {
        yaw_offset_degrees: frame.yaw_offset_degrees,
        footprint_m: frame.footprint_m,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn scene_camera_from_cached_state(camera: CachedCameraState) -> SceneCamera {
    SceneCamera {
        translation: camera.translation,
        focus: camera.focus,
        yaw: Some(camera.yaw.to_degrees()),
        pitch: Some(camera.pitch.to_degrees()),
        radius: Some(camera.radius),
        vertical_fov_degrees: camera.vertical_fov_degrees,
    }
}

fn scene_entity_name(index: usize, cache_key: &str) -> String {
    format!(
        "item_{:03}_{}",
        index + 1,
        sanitize_scene_identifier(cache_key)
    )
}

fn sanitize_scene_identifier(value: &str) -> String {
    let mut out = String::with_capacity(value.len().max(1));
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
        } else if !out.ends_with('_') {
            out.push('_');
        }
    }
    while out.ends_with('_') {
        out.pop();
    }
    if out.is_empty() {
        out.push_str("asset");
    }
    if out
        .chars()
        .next()
        .map(|ch| ch.is_ascii_digit())
        .unwrap_or(false)
    {
        out.insert_str(0, "id_");
    }
    out
}

#[cfg(not(target_arch = "wasm32"))]
fn rotation_y_degrees_from_quat(rotation: [f32; 4]) -> f32 {
    if rotation.iter().any(|value| !value.is_finite()) {
        return 0.0;
    }
    let quat = Quat::from_xyzw(rotation[0], rotation[1], rotation[2], rotation[3]);
    let quat = if quat.length_squared() > 0.0 {
        quat.normalize()
    } else {
        Quat::IDENTITY
    };
    let (yaw, _, _) = quat.to_euler(EulerRot::YXZ);
    yaw.to_degrees()
}

#[cfg(not(target_arch = "wasm32"))]
fn fmt_scene_vec3(value: [f32; 3]) -> String {
    value.map(fmt_scene_f32).join(",")
}

#[cfg(not(target_arch = "wasm32"))]
fn fmt_scene_f32(value: f32) -> String {
    let value = if value.abs() < 1.0e-6 { 0.0 } else { value };
    let mut out = format!("{value:.6}");
    while out.contains('.') && out.ends_with('0') {
        out.pop();
    }
    if out.ends_with('.') {
        out.pop();
    }
    if out == "-0" { "0".to_string() } else { out }
}

#[cfg(not(target_arch = "wasm32"))]
fn scene_assets_sidecar_path(path: &Path) -> PathBuf {
    path.with_extension("assets.json")
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
    meshes: &mut Assets<BevyMesh>,
    images: &mut Assets<Image>,
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
            match spawn_mesh_asset(
                commands,
                asset_server,
                meshes,
                images,
                materials,
                path.to_path_buf(),
            ) {
                Ok(_) => info!("loaded mesh asset {}", path.display()),
                Err(err) => warn!("failed to load mesh asset {}: {err}", path.display()),
            }
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
    PathBuf::from(format!("uploaded/{request_id:03}_{sanitized}"))
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn poll_mcp_scene_control(
    mut control: ResMut<McpSceneControl>,
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<BevyMesh>>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut gaussian_clouds: ResMut<Assets<PlanarGaussian3d>>,
    mut cache: ResMut<MeshCacheResource>,
    mut selection: ResMut<EditorSelection>,
    mut interaction_lock: ResMut<SceneInteractionLock>,
    transformables: Query<(), With<GizmoTransformable>>,
    cached_instances: Query<(Entity, &CachedMeshInstance)>,
    mut query_set: ParamSet<(
        Query<(&mut Transform, &mut PanOrbitCamera, &mut Projection), With<MainCamera>>,
        Query<(&CachedMeshInstance, &Transform)>,
        Query<(&Camera, &GlobalTransform), With<MainCamera>>,
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

    let envelope = match read_mcp_scene_commands(&path) {
        Ok(envelope) => envelope,
        Err(err) => {
            warn!(
                "Failed to parse MCP scene control file {}: {err}",
                path.display()
            );
            return;
        }
    };
    let session_id = envelope.session_id.clone();
    let sequence = envelope.sequence;
    let commands_to_apply = envelope.commands;
    if commands_to_apply.is_empty() {
        return;
    }
    let requested_commands = commands_to_apply.len();

    let mut scene_changed = false;
    let mut force_cache_flush = false;
    let mut screenshots = Vec::new();
    let mut predicted_world_items = Vec::new();
    let mut deleted_cache_keys = Vec::new();
    let mut cleared_scene = false;
    let mut command_results = Vec::with_capacity(requested_commands);
    for (command_index, command) in commands_to_apply.into_iter().enumerate() {
        match command {
            McpSceneCommand::SpawnCached {
                cache_key,
                translation,
                rotation,
                scale,
                select,
            } => {
                let asset = match cache.cache.load_asset(&cache_key) {
                    Ok(Some(asset)) => asset,
                    Ok(None) => {
                        warn!("MCP spawn_cached skipped: cache key {cache_key} not found");
                        command_results.push(mcp_scene_command_result(
                            command_index,
                            "spawn_cached",
                            false,
                            format!("cache key {cache_key} not found"),
                            Some(cache_key),
                            None,
                        ));
                        continue;
                    }
                    Err(err) => {
                        warn!("MCP spawn_cached failed for {cache_key}: {err}");
                        command_results.push(mcp_scene_command_result(
                            command_index,
                            "spawn_cached",
                            false,
                            format!("cache load failed: {err}"),
                            Some(cache_key),
                            None,
                        ));
                        continue;
                    }
                };
                let Some(transform) = transform_from_optional_parts(translation, rotation, scale)
                else {
                    warn!("MCP spawn_cached skipped due to invalid transform values");
                    command_results.push(mcp_scene_command_result(
                        command_index,
                        "spawn_cached",
                        false,
                        "invalid transform values",
                        Some(cache_key),
                        None,
                    ));
                    continue;
                };
                let metadata = cache
                    .cache
                    .asset_entries()
                    .iter()
                    .find(|entry| entry.cache_key == cache_key)
                    .cloned()
                    .unwrap_or_else(|| CachedMeshMetadata {
                        cache_key: cache_key.clone(),
                        source_image_path: cache_key.clone(),
                        label: cache_key.clone(),
                        source_image_payload_id: None,
                        source_image_name: None,
                        source_image_mime: None,
                        asset_kind: Default::default(),
                        mesh_payload_id: String::new(),
                        gltf_output_id: None,
                        glb_output_id: String::new(),
                        splat_payload_id: None,
                        local_aabb: None,
                        canonical_frame: None,
                        updated_at_unix_ms: 0,
                    });
                let Some(handles) = cached_asset_handles(
                    asset,
                    &metadata,
                    &mut meshes,
                    &mut images,
                    &mut materials,
                    &mut gaussian_clouds,
                ) else {
                    command_results.push(mcp_scene_command_result(
                        command_index,
                        "spawn_cached",
                        false,
                        "failed to prepare cached asset handles",
                        Some(cache_key),
                        None,
                    ));
                    continue;
                };
                let predicted_item =
                    cached_world_item_from_transform(cache_key.clone(), &transform);
                let entity = spawn_cached_asset_instance(
                    &mut commands,
                    &handles,
                    transform,
                    Some(cache_key.clone()),
                );
                if select {
                    selection.set(entity);
                }
                predicted_world_items.push(predicted_item);
                command_results.push(mcp_scene_command_result(
                    command_index,
                    "spawn_cached",
                    true,
                    "spawned cached asset",
                    Some(cache_key),
                    None,
                ));
                scene_changed = true;
            }
            McpSceneCommand::SpawnPath {
                path,
                cache_key,
                translation,
                rotation,
                scale,
                select,
            } => {
                if !path.exists() {
                    warn!("MCP spawn_path skipped: path {} not found", path.display());
                    command_results.push(mcp_scene_command_result(
                        command_index,
                        "spawn_path",
                        false,
                        "path not found",
                        cache_key,
                        Some(path.display().to_string()),
                    ));
                    continue;
                }
                if !is_mesh_file(path.as_path()) {
                    warn!(
                        "MCP spawn_path skipped: {} is not a supported mesh file",
                        path.display()
                    );
                    command_results.push(mcp_scene_command_result(
                        command_index,
                        "spawn_path",
                        false,
                        "unsupported mesh file",
                        cache_key,
                        Some(path.display().to_string()),
                    ));
                    continue;
                }
                let Some(transform) = transform_from_optional_parts(translation, rotation, scale)
                else {
                    warn!("MCP spawn_path skipped due to invalid transform values");
                    command_results.push(mcp_scene_command_result(
                        command_index,
                        "spawn_path",
                        false,
                        "invalid transform values",
                        cache_key,
                        Some(path.display().to_string()),
                    ));
                    continue;
                };
                let cache_key = cache_key.unwrap_or_else(|| format!("path:{}", path.display()));
                let path_label = path.display().to_string();
                let predicted_item =
                    cached_world_item_from_transform(cache_key.clone(), &transform);
                let entity = match spawn_mesh_asset_with_transform(
                    &mut commands,
                    &asset_server,
                    &mut meshes,
                    &mut images,
                    &mut materials,
                    path.clone(),
                    transform,
                    Some(cache_key.clone()),
                ) {
                    Ok(entity) => entity,
                    Err(err) => {
                        warn!("MCP spawn_path failed for {}: {err}", path.display());
                        command_results.push(mcp_scene_command_result(
                            command_index,
                            "spawn_path",
                            false,
                            format!("failed to load mesh path: {err}"),
                            Some(cache_key),
                            Some(path.display().to_string()),
                        ));
                        continue;
                    }
                };
                if select {
                    selection.set(entity);
                }
                predicted_world_items.push(predicted_item);
                command_results.push(mcp_scene_command_result(
                    command_index,
                    "spawn_path",
                    true,
                    "spawned mesh path",
                    Some(cache_key),
                    Some(path_label),
                ));
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
                let deleted = !to_despawn.is_empty();
                for entity in to_despawn {
                    commands.entity(entity).despawn();
                    scene_changed = true;
                }
                deleted_cache_keys.push(cache_key);
                command_results.push(mcp_scene_command_result(
                    command_index,
                    "delete_by_cache_key",
                    deleted,
                    if deleted {
                        "deleted cached instances"
                    } else {
                        "no cached instances matched"
                    },
                    deleted_cache_keys.last().cloned(),
                    None,
                ));
            }
            McpSceneCommand::ClearScene => {
                let to_despawn = cached_instances
                    .iter()
                    .map(|(entity, _)| entity)
                    .collect::<Vec<_>>();
                let deleted = to_despawn.len();
                for entity in to_despawn {
                    commands.entity(entity).despawn();
                }
                selection.clear();
                cleared_scene = true;
                scene_changed = true;
                command_results.push(mcp_scene_command_result(
                    command_index,
                    "clear_scene",
                    true,
                    format!("cleared {deleted} cached scene item(s)"),
                    None,
                    None,
                ));
            }
            McpSceneCommand::DeleteSelected => {
                let to_despawn: Vec<Entity> = selection
                    .iter()
                    .filter(|entity| transformables.contains(*entity))
                    .collect();
                let deleted = !to_despawn.is_empty();
                for entity in to_despawn {
                    commands.entity(entity).despawn();
                    scene_changed = true;
                }
                selection.clear();
                command_results.push(mcp_scene_command_result(
                    command_index,
                    "delete_selected",
                    true,
                    if deleted {
                        "deleted selected transformables"
                    } else {
                        "selection cleared; no transformables selected"
                    },
                    None,
                    None,
                ));
            }
            McpSceneCommand::ClearSelection => {
                selection.clear();
                command_results.push(mcp_scene_command_result(
                    command_index,
                    "clear_selection",
                    true,
                    "selection cleared",
                    None,
                    None,
                ));
            }
            McpSceneCommand::SetCamera {
                translation,
                rotation,
                focus,
                yaw,
                pitch,
                radius,
                vertical_fov,
            } => {
                if let Ok((mut transform, mut orbit, mut projection)) = query_set.p0().single_mut()
                {
                    let mut target_translation = Vec3::from_array(translation);
                    let target_rotation =
                        Quat::from_xyzw(rotation[0], rotation[1], rotation[2], rotation[3]);
                    let target_focus = focus
                        .map(Vec3::from_array)
                        .filter(|value| value.is_finite())
                        .unwrap_or(orbit.focus);
                    let target_yaw = yaw
                        .filter(|value| value.is_finite())
                        .map(|value| value.to_radians());
                    let target_pitch = pitch
                        .filter(|value| value.is_finite())
                        .map(|value| value.abs().to_radians());
                    let target_radius = radius.filter(|value| value.is_finite() && *value > 0.0);
                    let target_vertical_fov =
                        vertical_fov.filter(|value| value.is_finite() && *value > 0.0);
                    if let (Some(yaw), Some(pitch), Some(radius)) =
                        (target_yaw, target_pitch, target_radius)
                    {
                        let yaw_rot = Quat::from_axis_angle(Vec3::Y, yaw);
                        let pitch_rot = Quat::from_axis_angle(Vec3::X, -pitch);
                        let orbit_rotation = yaw_rot * pitch_rot;
                        target_translation =
                            target_focus + orbit_rotation * Vec3::new(0.0, 0.0, radius);
                    }
                    if target_translation.is_finite() && target_rotation.is_finite() {
                        transform.translation = target_translation;
                        if let (Some(yaw), Some(pitch), Some(radius)) =
                            (target_yaw, target_pitch, target_radius)
                        {
                            let yaw_rot = Quat::from_axis_angle(Vec3::Y, yaw);
                            let pitch_rot = Quat::from_axis_angle(Vec3::X, -pitch);
                            transform.rotation = yaw_rot * pitch_rot;
                            orbit.yaw = Some(yaw);
                            orbit.target_yaw = yaw;
                            orbit.pitch = Some(pitch);
                            orbit.target_pitch = pitch;
                            orbit.radius = Some(radius);
                            orbit.target_radius = radius;
                        } else if target_focus.distance_squared(target_translation) > 0.000_001 {
                            transform.look_at(target_focus, Vec3::Y);
                        } else {
                            transform.rotation = if target_rotation.length_squared() > 0.0 {
                                target_rotation.normalize()
                            } else {
                                Quat::IDENTITY
                            };
                        }
                        orbit.focus = target_focus;
                        orbit.target_focus = target_focus;
                        orbit.initialized = true;
                        if let (Some(vertical_fov), Projection::Perspective(perspective)) =
                            (target_vertical_fov, projection.as_mut())
                        {
                            perspective.fov = vertical_fov.to_radians();
                        }
                        command_results.push(mcp_scene_command_result(
                            command_index,
                            "set_camera",
                            true,
                            "camera updated",
                            None,
                            None,
                        ));
                        scene_changed = true;
                    } else {
                        command_results.push(mcp_scene_command_result(
                            command_index,
                            "set_camera",
                            false,
                            "camera transform was not finite",
                            None,
                            None,
                        ));
                    }
                } else {
                    command_results.push(mcp_scene_command_result(
                        command_index,
                        "set_camera",
                        false,
                        "main camera not found",
                        None,
                        None,
                    ));
                }
            }
            McpSceneCommand::CaptureScreenshot { path } => {
                if let Some(parent) = path.parent()
                    && !parent.as_os_str().is_empty()
                    && let Err(err) = fs::create_dir_all(parent)
                {
                    warn!(
                        "MCP capture_screenshot could not create {}: {err}",
                        parent.display()
                    );
                    command_results.push(mcp_scene_command_result(
                        command_index,
                        "capture_screenshot",
                        false,
                        format!("could not create parent directory: {err}"),
                        None,
                        Some(path.display().to_string()),
                    ));
                    continue;
                }
                screenshots.push(path.display().to_string());
                commands
                    .spawn(Screenshot::primary_window())
                    .observe(save_to_disk(path));
                command_results.push(mcp_scene_command_result(
                    command_index,
                    "capture_screenshot",
                    true,
                    "screenshot requested",
                    None,
                    screenshots.last().cloned(),
                ));
            }
            McpSceneCommand::SetInteractionLock { locked, reason } => {
                interaction_lock.set(locked, reason.clone());
                if locked {
                    selection.clear();
                }
                command_results.push(mcp_scene_command_result(
                    command_index,
                    "set_interaction_lock",
                    true,
                    if locked {
                        reason
                            .as_deref()
                            .filter(|value| !value.trim().is_empty())
                            .unwrap_or("scene interaction locked")
                            .to_string()
                    } else {
                        "scene interaction unlocked".to_string()
                    },
                    None,
                    None,
                ));
            }
            McpSceneCommand::ReloadCache => match MeshCache::load_default() {
                Ok(reloaded_cache) => {
                    let entries = reloaded_cache.asset_entries().len();
                    cache.cache = reloaded_cache;
                    command_results.push(mcp_scene_command_result(
                        command_index,
                        "reload_cache",
                        true,
                        format!("reloaded {entries} cached catalog asset(s)"),
                        None,
                        None,
                    ));
                }
                Err(err) => {
                    warn!("MCP reload_cache failed: {err}");
                    command_results.push(mcp_scene_command_result(
                        command_index,
                        "reload_cache",
                        false,
                        format!("cache reload failed: {err}"),
                        None,
                        None,
                    ));
                }
            },
            McpSceneCommand::SaveCache => {
                force_cache_flush = true;
                command_results.push(mcp_scene_command_result(
                    command_index,
                    "save_cache",
                    true,
                    "cache flush requested",
                    None,
                    None,
                ));
            }
        }
    }

    if force_cache_flush {
        let camera_state = {
            let main_camera = query_set.p0();
            main_camera
                .single()
                .ok()
                .and_then(|(transform, orbit, projection)| {
                    camera_state_from_components(transform, orbit, Some(projection))
                })
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

    if let Some(status_path) = control.status_path.as_deref() {
        let camera_state = {
            let main_camera = query_set.p0();
            main_camera
                .single()
                .ok()
                .and_then(|(transform, orbit, projection)| {
                    camera_state_from_components(transform, orbit, Some(projection))
                })
        };
        let mut world_items = {
            let cached_query = query_set.p1();
            if cleared_scene {
                Vec::new()
            } else {
                collect_cached_world_items(&cached_query)
            }
        };
        if !deleted_cache_keys.is_empty() {
            world_items.retain(|item| !deleted_cache_keys.contains(&item.cache_key));
        }
        world_items.extend(predicted_world_items);
        let projected_items = {
            let camera_query = query_set.p2();
            collect_projected_world_items(&cache.cache, &world_items, &camera_query)
        };
        let status = McpSceneStatus {
            session_id,
            last_sequence: sequence,
            ok: command_results.iter().all(|result| result.applied),
            message: if command_results.iter().all(|result| result.applied) {
                "applied".to_string()
            } else {
                "partially_applied".to_string()
            },
            requested_commands,
            applied_commands: command_results
                .iter()
                .filter(|result| result.applied)
                .count(),
            command_results,
            cache_entries: cache.cache.asset_entries().to_vec(),
            world_items,
            projected_items,
            camera: camera_state,
            screenshots,
            interaction_locked: interaction_lock.locked,
            interaction_lock_reason: interaction_lock.reason.clone(),
        };
        if let Err(err) = write_mcp_scene_status(status_path, &status) {
            warn!(
                "Failed to write MCP scene status {}: {err}",
                status_path.display()
            );
        }
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
                        } else {
                            wasm_set_warmup_state("loading");
                        }
                        wasm_set_warmup_message(&message);
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
                for (request, result) in event.requests.into_iter().zip(event.results) {
                    handle_inference_result(
                        &mut ctx.commands,
                        &mut ctx.meshes,
                        &mut ctx.images,
                        &mut ctx.materials,
                        &mut ctx.gaussian_clouds,
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
        #[cfg(not(target_arch = "wasm32"))]
        if should_wait_before_inference_dispatch(&mut ctx.dispatch_gate, &ctx.queue) {
            update_status_message(&ctx.args, &ctx.queue, &mut ctx.status);
            return;
        }

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

    #[cfg(not(target_arch = "wasm32"))]
    if ctx.queue.active.is_none() && ctx.queue.pending.is_empty() {
        reset_inference_dispatch_gate(&mut ctx.dispatch_gate);
    }

    update_status_message(&ctx.args, &ctx.queue, &mut ctx.status);
}

fn handle_catalog_spawn_requests(
    mut requests: MessageReader<CatalogSpawnRequest>,
    mut commands: Commands,
    mut selection: Option<ResMut<EditorSelection>>,
    interaction_lock: Res<SceneInteractionLock>,
) {
    if interaction_lock.locked {
        for _ in requests.read() {}
        return;
    }
    for request in requests.read() {
        let entity = match &request.asset {
            CatalogSpawnAsset::Mesh { mesh, material } => spawn_mesh_instance(
                &mut commands,
                mesh.clone(),
                material.clone(),
                request.transform,
                request.cache_key.clone(),
            ),
            CatalogSpawnAsset::GaussianSplat { cloud } => spawn_gaussian_splat_instance(
                &mut commands,
                cloud.clone(),
                request.transform,
                request.cache_key.clone(),
            ),
        };
        if request.select_spawned
            && let Some(selection) = selection.as_mut()
        {
            selection.set(entity);
        }
    }
}

pub(crate) fn handle_catalog_delete_requests(
    mut requests: MessageReader<CatalogDeleteRequest>,
    mut cache: ResMut<MeshCacheResource>,
    cached_instances: Query<(Entity, &CachedMeshInstance)>,
    mut commands: Commands,
    interaction_lock: Res<SceneInteractionLock>,
) {
    if interaction_lock.locked {
        for _ in requests.read() {}
        return;
    }
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

fn enforce_scene_interaction_lock(
    interaction_lock: Res<SceneInteractionLock>,
    mut selection: ResMut<EditorSelection>,
) {
    if interaction_lock.locked {
        selection.clear();
    }
}

fn sync_panorbit_bindings(mut cameras: Query<&mut PanOrbitCamera>) {
    for mut camera in cameras.iter_mut() {
        if camera.button_orbit != MouseButton::Left {
            camera.button_orbit = MouseButton::Left;
        }
        if camera.button_pan != MouseButton::Right {
            camera.button_pan = MouseButton::Right;
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
    interaction_lock: Res<SceneInteractionLock>,
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
        .map(|window| ui_state.cursor_over_ui(window))
        .unwrap_or(false);
    let enabled = !interaction_lock.locked
        && !gizmo_active
        && !gizmo_handle_pressed
        && !drag.is_dragging()
        && !ui_block;
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
            pulse.phase = (pulse.phase + 1) % title_rattler_frame_count();
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
        processing_window_title(&name, queue.pending.len(), pulse.phase)
    } else {
        format!("bevy_synth — {}", status.message)
    };

    if let Ok(mut window) = windows.single_mut() {
        window.title = title;
    }
}

pub(crate) fn processing_window_title(name: &str, queued: usize, phase: usize) -> String {
    format!(
        "bevy_synth [{}] processing: {name} (queued: {queued})",
        title_rattler_frame(phase)
    )
}

pub(crate) fn title_rattler_frame(phase: usize) -> &'static str {
    let rattler = rattles::presets::ascii::simple_dots();
    let len = rattler.len();
    if len == 0 {
        return "   ";
    }
    rattler.frame(phase % len)
}

fn title_rattler_frame_count() -> usize {
    rattles::presets::ascii::simple_dots().len().max(1)
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
        .and_then(|(transform, orbit)| camera_state_from_components(transform, orbit, None));
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
    gaussian_clouds: &mut ResMut<Assets<PlanarGaussian3d>>,
    cache: &mut ResMut<MeshCacheResource>,
    catalog: &mut ResMut<CatalogState>,
    request: InferenceRequest,
    result: Result<Option<SynthAsset>, String>,
) {
    match result {
        Ok(Some(SynthAsset::Mesh(mesh))) => {
            if let Some(output) = request.output_path.as_ref()
                && let Err(err) = write_glb(output, &mesh)
            {
                warn!("failed to write mesh to {}: {err}", output.display());
            }

            let cached_metadata = match cache.cache.upsert_mesh_for_image_with_source_bytes(
                &request.image_path,
                request.image_contents.as_deref(),
                &mesh,
            ) {
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
                entry.gaussian = None;
                entry.source_image_path = Some(request.image_path.display().to_string());
                entry.cache_key = cache_key;
                entry.source_image = cached_metadata.as_ref().and_then(|metadata| {
                    cached_source_image_handle(&cache.cache, metadata, images.as_mut())
                });
                if let Some(metadata) = cached_metadata.as_ref() {
                    entry.label = metadata.label.clone();
                    entry.source_image_path = Some(metadata.source_image_path.clone());
                }
                catalog.bump_revision();
            }
        }
        Ok(Some(SynthAsset::GaussianSplat(splats))) => {
            let count = splats.len();
            if let Some(output) = request.output_path.as_ref() {
                match write_gaussian_splat_output(output, &splats) {
                    Ok(()) => info!(
                        "Wrote Gaussian splat asset with {count} splats to {}",
                        output.display()
                    ),
                    Err(err) => warn!(
                        "Failed to write Gaussian splat asset to {}: {err}",
                        output.display()
                    ),
                }
            }

            let cached_metadata = match cache
                .cache
                .upsert_gaussian_splat_for_image_with_source_bytes(
                    &request.image_path,
                    request.image_contents.as_deref(),
                    &splats,
                ) {
                Ok(metadata) => Some(metadata),
                Err(err) => {
                    warn!(
                        "Failed to cache Gaussian splat output for {}: {err}",
                        request.image_path.display()
                    );
                    None
                }
            };
            let cache_key = cached_metadata
                .as_ref()
                .map(|metadata| metadata.cache_key.clone());

            match gaussian_splat_cloud_handle(&splats, gaussian_clouds) {
                Ok(cloud_handle) => {
                    info!(
                        "Showing Bevy Gaussian cloud with {count} TripoSplat splats from {}",
                        request.image_path.display()
                    );
                    spawn_gaussian_splat_instance(
                        commands,
                        cloud_handle.clone(),
                        Transform::default(),
                        cache_key.clone(),
                    );
                    if let Some(entry) = catalog.entry_mut(request.id) {
                        entry.status = CatalogStatus::Ready;
                        entry.mesh = None;
                        entry.material = None;
                        entry.gaussian = Some(cloud_handle);
                        entry.source_image_path = Some(request.image_path.display().to_string());
                        entry.cache_key = cache_key;
                        entry.source_image = cached_metadata.as_ref().and_then(|metadata| {
                            cached_source_image_handle(&cache.cache, metadata, images.as_mut())
                        });
                        if let Some(metadata) = cached_metadata.as_ref() {
                            entry.label = metadata.label.clone();
                            entry.source_image_path = Some(metadata.source_image_path.clone());
                        }
                        catalog.bump_revision();
                    }
                }
                Err(err) => {
                    warn!(
                        "Failed to build Gaussian splat cloud for {}: {err}",
                        request.image_path.display()
                    );
                    if let Some(entry) = catalog.entry_mut(request.id) {
                        entry.status =
                            CatalogStatus::Failed(format!("Gaussian splat cloud failed: {err}"));
                        catalog.bump_revision();
                    }
                }
            }
        }
        Ok(None) => {
            warn!(
                "Synthesis inference produced an empty asset for {}",
                request.image_path.display()
            );
            if let Some(entry) = catalog.entry_mut(request.id) {
                entry.status = CatalogStatus::Failed("empty asset".to_string());
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

    let has_valid_uvs = mesh.uvs.len() == mesh.mesh.vertices.len() && !mesh.uvs.is_empty();
    if let Some(pbr) = mesh.pbr_textures.as_ref() {
        // TRELLIS outputs ship PBR textures + UVs and should render through
        // texture-backed StandardMaterial. TripoSG outputs have no PBR textures
        // and stay on the default non-textured material path.
        if !has_valid_uvs {
            warn!(
                "Skipping PBR texture assignment because mesh UVs are missing or invalid (vertices={}, uvs={}).",
                mesh.mesh.vertices.len(),
                mesh.uvs.len()
            );
            return out;
        }
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

fn gaussian_splat_cloud_handle(
    splats: &GaussianSplatCloud,
    gaussian_clouds: &mut Assets<PlanarGaussian3d>,
) -> Result<Handle<PlanarGaussian3d>, String> {
    Ok(gaussian_clouds.add(gaussian_splat_cloud_to_planar_gaussian_3d(splats)?))
}

pub(crate) fn gaussian_splat_cloud_to_planar_gaussian_3d(
    splats: &GaussianSplatCloud,
) -> Result<PlanarGaussian3d, String> {
    let transformed = splats.transformed_splats_for_bevy_display()?;
    let gaussians = transformed
        .iter()
        .enumerate()
        .map(|(index, splat)| {
            if !splat.opacity.is_finite() {
                return Err(format!(
                    "Gaussian splat {index} contains non-finite opacity"
                ));
            }
            let mut spherical_harmonic = SphericalHarmonicCoefficients::default();
            for channel in 0..3 {
                spherical_harmonic.set(channel, splat.features_dc[channel]);
            }
            Ok(Gaussian3d {
                position_visibility: [splat.position[0], splat.position[1], splat.position[2], 1.0]
                    .into(),
                spherical_harmonic,
                rotation: splat.rotation.into(),
                scale_opacity: [
                    splat.scale[0],
                    splat.scale[1],
                    splat.scale[2],
                    splat.opacity.clamp(0.0, 1.0),
                ]
                .into(),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(PlanarGaussian3d::from(gaussians))
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
    let output_path = resolve_asset_output_path(args, &image_path, queue.counter);
    let request = InferenceRequest {
        id: queue.counter,
        image_path,
        image_contents,
        output_path,
        synthesis_models: args.synthesis_models.clone(),
        settings: InferenceSettings::from_args(args),
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
    meshes: &mut Assets<BevyMesh>,
    images: &mut Assets<Image>,
    materials: &mut Assets<StandardMaterial>,
    mesh_path: PathBuf,
) -> Result<Entity, String> {
    spawn_mesh_asset_with_transform(
        commands,
        asset_server,
        meshes,
        images,
        materials,
        mesh_path,
        Transform::default(),
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn spawn_mesh_asset_with_transform(
    commands: &mut Commands,
    asset_server: &AssetServer,
    meshes: &mut Assets<BevyMesh>,
    images: &mut Assets<Image>,
    materials: &mut Assets<StandardMaterial>,
    mesh_path: PathBuf,
    transform: Transform,
    cache_key: Option<String>,
) -> Result<Entity, String> {
    #[cfg(target_arch = "wasm32")]
    let _ = (&mut *meshes, &mut *images);

    #[cfg(not(target_arch = "wasm32"))]
    if is_glb_file(&mesh_path) {
        let (mesh_handle, material) =
            load_generated_glb_mesh_asset(&mesh_path, meshes, images, materials)?;
        return Ok(spawn_mesh_instance(
            commands,
            mesh_handle,
            material,
            transform,
            cache_key,
        ));
    }

    if is_gltf_file(&mesh_path) {
        let mesh_handle: Handle<BevyMesh> = asset_server.load(
            GltfAssetLabel::Primitive {
                mesh: 0,
                primitive: 0,
            }
            .from_asset(mesh_path),
        );
        let material = materials.add(StandardMaterial {
            base_color: Color::srgb(0.82, 0.82, 0.9),
            cull_mode: None,
            ..default()
        });
        return Ok(spawn_mesh_instance(
            commands,
            mesh_handle,
            material,
            transform,
            cache_key,
        ));
    }

    let mesh_handle: Handle<BevyMesh> = asset_server.load(mesh_path);
    let material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.82, 0.82, 0.9),
        cull_mode: None,
        ..default()
    });
    Ok(spawn_mesh_instance(
        commands,
        mesh_handle,
        material,
        transform,
        cache_key,
    ))
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn load_generated_glb_mesh_asset(
    path: &Path,
    meshes: &mut Assets<BevyMesh>,
    images: &mut Assets<Image>,
    materials: &mut Assets<StandardMaterial>,
) -> Result<(Handle<BevyMesh>, Handle<StandardMaterial>), String> {
    let bytes =
        std::fs::read(path).map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    let mesh = mesh_from_glb_bytes(bytes.as_slice())
        .map_err(|err| format!("failed to parse {}: {err}", path.display()))?;
    let mesh_handle = meshes.add(to_bevy_mesh_synth(&mesh));
    let material = materials.add(standard_material_for_inference(&mesh, images));
    Ok((mesh_handle, material))
}

#[cfg(not(target_arch = "wasm32"))]
fn is_glb_file(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|value| value.to_str())
            .map(|value| value.to_ascii_lowercase())
            .as_deref(),
        Some("glb")
    )
}

fn is_gltf_file(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|value| value.to_str())
            .map(|value| value.to_ascii_lowercase())
            .as_deref(),
        Some("glb" | "gltf")
    )
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

fn spawn_cached_asset_instance(
    commands: &mut Commands,
    handles: &CachedAssetHandles,
    transform: Transform,
    cache_key: Option<String>,
) -> Entity {
    match handles {
        CachedAssetHandles::Mesh { mesh, material } => spawn_mesh_instance(
            commands,
            mesh.clone(),
            material.clone(),
            transform,
            cache_key,
        ),
        CachedAssetHandles::GaussianSplat { cloud } => {
            spawn_gaussian_splat_instance(commands, cloud.clone(), transform, cache_key)
        }
    }
}

pub(crate) fn spawn_gaussian_splat_instance(
    commands: &mut Commands,
    cloud_handle: Handle<PlanarGaussian3d>,
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
        PlanarGaussian3dHandle(cloud_handle),
        triposplat_cloud_settings(),
        transform,
        RenderLayers::layer(0),
        Name::new("triposplat_gaussian_cloud"),
    ));
    if let Some(cache_key) = cache_key {
        entity_commands.insert(CachedMeshInstance { cache_key });
    }
    entity_commands.id()
}

pub(crate) fn triposplat_cloud_settings() -> CloudSettings {
    CloudSettings {
        sort_mode: SortMode::Std,
        color_space: GaussianColorSpace::SrgbRec709Display,
        ..default()
    }
}

#[allow(clippy::type_complexity)]
#[cfg(not(target_arch = "wasm32"))]
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

#[allow(clippy::type_complexity)]
#[cfg(target_arch = "wasm32")]
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
        .and_then(|(transform, orbit)| camera_state_from_components(transform, orbit, None));
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
        world_items.push(cached_world_item_from_transform(
            cached.cache_key.clone(),
            transform,
        ));
    }
    world_items
}

#[cfg(not(target_arch = "wasm32"))]
fn collect_projected_world_items(
    cache: &MeshCache,
    world_items: &[CachedWorldItem],
    camera_query: &Query<(&Camera, &GlobalTransform), With<MainCamera>>,
) -> Vec<McpProjectedWorldItem> {
    let metadata_by_key = cache
        .asset_entries()
        .iter()
        .map(|entry| (entry.cache_key.as_str(), entry))
        .collect::<HashMap<_, _>>();
    let camera = camera_query.single().ok();
    world_items
        .iter()
        .map(|item| {
            let world_aabb = metadata_by_key
                .get(item.cache_key.as_str())
                .and_then(|metadata| metadata.local_aabb)
                .and_then(|aabb| {
                    transform_from_cached_world_item(item)
                        .map(|transform| transformed_world_aabb(&transform, aabb))
                });
            let (screen_bbox, screen_contact, projected_corners, total_corners) = if let (
                Some(world_aabb),
                Some((camera, camera_transform)),
            ) =
                (world_aabb, camera)
            {
                project_world_aabb(camera, camera_transform, world_aabb)
            } else {
                (None, None, 0, 0)
            };
            McpProjectedWorldItem {
                cache_key: item.cache_key.clone(),
                world_aabb,
                screen_bbox,
                screen_contact,
                projected_corners,
                total_corners,
            }
        })
        .collect()
}

#[cfg(not(target_arch = "wasm32"))]
fn transformed_world_aabb(transform: &Transform, local_aabb: CachedAssetAabb) -> CachedAssetAabb {
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for corner in aabb_corners(local_aabb) {
        let world = transform.transform_point(corner);
        min = min.min(world);
        max = max.max(world);
    }
    CachedAssetAabb {
        min: min.to_array(),
        max: max.to_array(),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn aabb_corners(aabb: CachedAssetAabb) -> [Vec3; 8] {
    let min = Vec3::from_array(aabb.min);
    let max = Vec3::from_array(aabb.max);
    [
        Vec3::new(min.x, min.y, min.z),
        Vec3::new(max.x, min.y, min.z),
        Vec3::new(min.x, max.y, min.z),
        Vec3::new(max.x, max.y, min.z),
        Vec3::new(min.x, min.y, max.z),
        Vec3::new(max.x, min.y, max.z),
        Vec3::new(min.x, max.y, max.z),
        Vec3::new(max.x, max.y, max.z),
    ]
}

#[cfg(not(target_arch = "wasm32"))]
fn project_world_aabb(
    camera: &Camera,
    camera_transform: &GlobalTransform,
    world_aabb: CachedAssetAabb,
) -> (Option<[f32; 4]>, Option<[f32; 2]>, usize, usize) {
    let Some(viewport) = camera
        .logical_viewport_size()
        .filter(|size| size.x > 0.0 && size.y > 0.0)
    else {
        return (None, None, 0, 8);
    };
    let mut projected = Vec::with_capacity(8);
    for corner in aabb_corners(world_aabb) {
        if let Ok(pixel) = camera.world_to_viewport(camera_transform, corner) {
            let normalized = Vec2::new(pixel.x / viewport.x, pixel.y / viewport.y);
            if normalized.is_finite() {
                projected.push(normalized);
            }
        }
    }
    let screen_bbox = if projected.is_empty() {
        None
    } else {
        let mut min = Vec2::splat(f32::INFINITY);
        let mut max = Vec2::splat(f32::NEG_INFINITY);
        for point in &projected {
            min = min.min(*point);
            max = max.max(*point);
        }
        Some([min.x, min.y, max.x, max.y])
    };
    let contact_world = Vec3::new(
        (world_aabb.min[0] + world_aabb.max[0]) * 0.5,
        world_aabb.min[1],
        (world_aabb.min[2] + world_aabb.max[2]) * 0.5,
    );
    let screen_contact = camera
        .world_to_viewport(camera_transform, contact_world)
        .ok()
        .map(|pixel| [pixel.x / viewport.x, pixel.y / viewport.y])
        .filter(|point| point[0].is_finite() && point[1].is_finite());
    (screen_bbox, screen_contact, projected.len(), 8)
}

fn cached_world_item_from_transform(cache_key: String, transform: &Transform) -> CachedWorldItem {
    let rotation = if transform.rotation.length_squared() > 0.0 {
        transform.rotation.normalize()
    } else {
        Quat::IDENTITY
    };
    CachedWorldItem {
        cache_key,
        translation: transform.translation.to_array(),
        rotation: rotation.to_array(),
        scale: transform.scale.to_array(),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn mcp_scene_command_result(
    index: usize,
    command_type: &'static str,
    applied: bool,
    message: impl Into<String>,
    cache_key: Option<String>,
    path: Option<String>,
) -> McpSceneCommandResult {
    McpSceneCommandResult {
        index,
        command_type,
        applied,
        message: message.into(),
        cache_key,
        path,
    }
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
fn read_mcp_scene_commands(path: &std::path::Path) -> Result<McpSceneCommandEnvelope, io::Error> {
    let content = fs::read_to_string(path)?;
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return Ok(McpSceneCommandEnvelope {
            session_id: None,
            sequence: None,
            commands: Vec::new(),
        });
    }
    if trimmed.starts_with('[') {
        let commands = serde_json::from_str::<Vec<McpSceneCommand>>(trimmed).map_err(|err| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid command array JSON: {err}"),
            )
        })?;
        return Ok(McpSceneCommandEnvelope {
            session_id: None,
            sequence: None,
            commands,
        });
    }

    serde_json::from_str::<McpSceneCommandEnvelope>(trimmed).map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid command envelope JSON: {err}"),
        )
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn write_mcp_scene_status(path: &std::path::Path, status: &McpSceneStatus) -> io::Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("status.json.tmp");
    let bytes = serde_json::to_vec_pretty(status).map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("failed to serialize MCP scene status: {err}"),
        )
    })?;
    fs::write(&tmp, bytes)?;
    fs::rename(tmp, path)
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

#[allow(clippy::type_complexity)]
pub(crate) fn sync_gaussian_splat_pick_bounds(
    mut commands: Commands,
    gaussian_clouds: Res<Assets<PlanarGaussian3d>>,
    clouds_without_bounds: Query<
        (Entity, &PlanarGaussian3dHandle),
        (With<GizmoTransformable>, Without<GaussianSplatPickBounds>),
    >,
) {
    for (entity, cloud_handle) in clouds_without_bounds.iter() {
        let Some(cloud) = gaussian_clouds.get(&cloud_handle.0) else {
            continue;
        };
        if let Some(bounds) = gaussian_splat_pick_bounds(cloud) {
            commands
                .entity(entity)
                .insert((bounds, TransformGizmoOffset(bounds.center)));
        }
    }
}

pub(crate) fn gaussian_splat_pick_bounds(
    cloud: &PlanarGaussian3d,
) -> Option<GaussianSplatPickBounds> {
    if cloud.position_visibility.is_empty() {
        return None;
    }

    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for (position_visibility, scale_opacity) in cloud
        .position_visibility
        .iter()
        .zip(cloud.scale_opacity.iter())
    {
        let position = Vec3::from_array(position_visibility.position);
        let scale = Vec3::from_array(scale_opacity.scale);
        if !position.is_finite() || !scale.is_finite() {
            return None;
        }
        let radius = scale.abs() * 3.0 + Vec3::splat(0.01);
        min = min.min(position - radius);
        max = max.max(position + radius);
    }

    if !min.is_finite() || !max.is_finite() {
        return None;
    }
    let center = (min + max) * 0.5;
    let half_extents = ((max - min) * 0.5).max(Vec3::splat(0.05));
    Some(GaussianSplatPickBounds {
        center,
        half_extents,
    })
}

fn delete_selected_meshes(
    keys: Res<ButtonInput<KeyCode>>,
    mut selection: ResMut<EditorSelection>,
    transformables: Query<(), With<GizmoTransformable>>,
    mut commands: Commands,
    interaction_lock: Res<SceneInteractionLock>,
) {
    if interaction_lock.locked {
        return;
    }
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
    gaussian_transformables: Query<
        (Entity, &GaussianSplatPickBounds, &GlobalTransform),
        With<GizmoTransformable>,
    >,
    meshes: Res<Assets<BevyMesh>>,
    interaction_lock: Res<SceneInteractionLock>,
) {
    if interaction_lock.locked {
        return;
    }
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
    for (entity, bounds, transform) in gaussian_transformables.iter() {
        let (world_min, world_max) = world_aabb(bounds.center, bounds.half_extents, transform);
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

pub(crate) fn world_aabb(
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

pub(crate) fn ray_aabb_intersection(
    origin: Vec3,
    direction: Vec3,
    min: Vec3,
    max: Vec3,
) -> Option<f32> {
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

#[cfg(not(target_arch = "wasm32"))]
use std::cmp::Reverse;
use std::collections::VecDeque;
#[cfg(not(target_arch = "wasm32"))]
use std::fs;
#[cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};
#[cfg(not(target_arch = "wasm32"))]
use std::time::UNIX_EPOCH;
use std::time::{Duration, Instant};

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::RenderLayers;
use bevy::camera::{CameraOutputMode, RenderTarget};
use bevy::input::mouse::MouseScrollUnit;
use bevy::light::{DirectionalLight, PointLight};
use bevy::pbr::MeshMaterial3d;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bevy::ui::widget::Button;
use bevy::ui::{ComputedNode, OverflowAxis, ScrollPosition};
use bevy::window::PrimaryWindow;
use bevy_gaussian_splatting::gaussian::settings::GaussianColorSpace;
use bevy_gaussian_splatting::sort::SortMode;
use bevy_gaussian_splatting::{
    CloudSettings, GaussianCamera, PlanarGaussian3d, PlanarGaussian3dHandle,
};
use bevy_mesh::{Mesh as BevyMesh, Mesh3d};
use bevy_picking::Pickable;
use bevy_picking::events::{Pointer, Scroll as PointerScroll};
use log::info;

/// Internalized editor-core module.
///
/// TODO: switch this module to the upstream published `bevy_editor_core` crate
/// once the external crate is finalized and versioned for this repository.
pub mod bevy_editor_core;
/// Internalized file-dialog module.
///
/// TODO: switch this module to the upstream published `bevy_file_dialog` crate
/// once the external crate is finalized and versioned for this repository.
pub mod bevy_file_dialog;
/// Internalized transform-gizmos module.
///
/// TODO: switch this module to the upstream published `bevy_transform_gizmos` crate
/// once the external crate is finalized and versioned for this repository.
pub mod bevy_transform_gizmos;

use crate::bevy_file_dialog::prelude::FileDialogExt;

use bevy_synth_runtime::args::{
    AppArgs, BackendKind, DEFAULT_TRELLIS_PBR_TEXTURE_SIZE, SynthesisModel,
    TRIPOSPLAT_GAUSSIAN_STEP, TRIPOSPLAT_MAX_NUM_GAUSSIANS, TRIPOSPLAT_MIN_NUM_GAUSSIANS,
    TrellisQuality, TripoSplatProfile,
};
use bevy_synth_runtime::cache::{CachedSceneCategoryMetric, CachedSceneMetrics};
use bevy_synth_runtime::state::{InferenceQueue, InferenceRequest, UiStatus};

const PANEL_WIDTH: f32 = 336.0;
const MENU_HEIGHT: f32 = 44.0;
const THUMB_SIZE: f32 = 84.0;
const ENTRY_GAP: f32 = 10.0;
const CATALOG_PAGE_SIZE: usize = 6;
const CATALOG_MODE_SELECTOR_WIDTH: f32 = 92.0;
const CATALOG_NAV_BUTTON_WIDTH: f32 = 24.0;
const CATALOG_PAGE_LABEL_WIDTH: f32 = 30.0;
const CATALOG_DELETE_BUTTON_WIDTH: f32 = 52.0;
const CATALOG_TOGGLE_BUTTON_WIDTH: f32 = 44.0;
const PIPELINE_SELECTOR_WIDTH: f32 = 176.0;
const PIPELINE_SELECTOR_HEIGHT: f32 = 30.0;
const STATUS_BADGE_WIDTH: f32 = 260.0;
const SETTINGS_MODAL_WIDTH: f32 = 560.0;
const SETTINGS_MODAL_MAX_HEIGHT_VH: f32 = 88.0;
const SETTINGS_TAB_BODY_MAX_HEIGHT_VH: f32 = 64.0;
const PREVIEW_SIZE: u32 = 128;
const PREVIEW_MAX_LAYER: usize = 30;
const GIZMO_LAYER: usize = 12;
const PREVIEW_CAMERA_FOV: f32 = std::f32::consts::FRAC_PI_4;
const PREVIEW_CAMERA_MARGIN: f32 = 0.2;
const PREVIEW_TARGET_RADIUS: f32 = 0.72;
const PREVIEW_FALLBACK_RADIUS: f32 = 0.72;
const CATALOG_LABEL_MAX_CHARS: usize = 34;
const CATALOG_STATUS_MAX_CHARS: usize = 36;
const CATALOG_DOUBLE_CLICK_SECONDS: f64 = 0.35;
const DRAG_GHOST_ALPHA: f32 = 0.35;
const MENU_BG: Color = Color::srgb(0.08, 0.09, 0.11);
const PANEL_BG: Color = Color::srgb(0.06, 0.07, 0.09);
const PANEL_BORDER: Color = Color::srgb(0.13, 0.14, 0.18);
const BUTTON_BG: Color = Color::srgb(0.1, 0.11, 0.14);
const BUTTON_BG_HOVER: Color = Color::srgb(0.13, 0.14, 0.18);
const BUTTON_BG_PRESSED: Color = Color::srgb(0.17, 0.19, 0.24);
const BUTTON_BORDER: Color = Color::srgb(0.28, 0.3, 0.35);
const BUTTON_BORDER_HOVER: Color = Color::srgb(0.36, 0.4, 0.5);
const BUTTON_BORDER_PRESSED: Color = Color::srgb(0.46, 0.52, 0.64);
const BUTTON_BG_DISABLED: Color = Color::srgb(0.08, 0.09, 0.11);
const BUTTON_BORDER_DISABLED: Color = Color::srgb(0.2, 0.22, 0.26);
const BUTTON_TEXT: Color = Color::srgb(0.86, 0.88, 0.94);
const BUTTON_TEXT_DISABLED: Color = Color::srgb(0.45, 0.48, 0.56);
const BUTTON_OPEN_BG: Color = Color::srgb(0.14, 0.18, 0.3);
const BUTTON_OPEN_BG_HOVER: Color = Color::srgb(0.18, 0.24, 0.39);
const BUTTON_OPEN_BG_PRESSED: Color = Color::srgb(0.22, 0.29, 0.47);
const BUTTON_OPEN_BORDER: Color = Color::srgb(0.32, 0.4, 0.62);
const BUTTON_OPEN_BORDER_HOVER: Color = Color::srgb(0.43, 0.53, 0.8);
const BUTTON_OPEN_BORDER_PRESSED: Color = Color::srgb(0.55, 0.66, 0.95);
const BUTTON_ACTIVE_BG: Color = Color::srgb(0.18, 0.23, 0.34);
const BUTTON_ACTIVE_BORDER: Color = Color::srgb(0.48, 0.58, 0.78);
const ENTRY_BG: Color = Color::srgb(0.1, 0.11, 0.14);
const ENTRY_BG_HOVER: Color = Color::srgb(0.13, 0.15, 0.2);
const ENTRY_BG_PRESSED: Color = Color::srgb(0.17, 0.2, 0.28);
const ENTRY_BORDER: Color = Color::srgb(0.2, 0.22, 0.26);
const ENTRY_BORDER_HOVER: Color = Color::srgb(0.3, 0.35, 0.44);
const ENTRY_BORDER_PRESSED: Color = Color::srgb(0.44, 0.51, 0.64);
const STATUS_BADGE_BG: Color = Color::srgb(0.1, 0.11, 0.14);
const STATUS_BADGE_BORDER: Color = Color::srgb(0.24, 0.27, 0.33);
const STATUS_IDLE: Color = Color::srgb(0.52, 0.56, 0.64);
const STATUS_PENDING: Color = Color::srgb(0.26, 0.62, 0.88);
const STATUS_PROCESSING: Color = Color::srgb(0.93, 0.66, 0.2);
const MODAL_SCRIM: Color = Color::srgba(0.0, 0.0, 0.0, 0.45);
const MODAL_BG: Color = Color::srgb(0.07, 0.08, 0.1);
const MODAL_BORDER: Color = Color::srgb(0.24, 0.27, 0.34);
const TRIPOSPLAT_MIN_STEPS: usize = 1;
const TRIPOSPLAT_MAX_STEPS: usize = 50;
const TRIPOSPLAT_MIN_GUIDANCE: f32 = 1.0;
const TRIPOSPLAT_MAX_GUIDANCE: f32 = 10.0;
const TRIPOSPLAT_GUIDANCE_STEP: f32 = 0.5;
const TRIPOSG_MIN_STEPS: usize = 1;
const TRIPOSG_MAX_STEPS: usize = 50;
const TRIPOSG_MIN_TOKENS: usize = 128;
const TRIPOSG_MAX_TOKENS: usize = 4096;
const TRIPOSG_TOKEN_STEP: usize = 128;
const TRIPOSG_MIN_GUIDANCE: f32 = 1.0;
const TRIPOSG_MAX_GUIDANCE: f32 = 12.0;
const TRIPOSG_GUIDANCE_STEP: f32 = 0.5;
const TRIPOSG_FACE_STEP: usize = 1000;
const TRIPOSG_MAX_FACES: usize = 100_000;
const TRELLIS_PBR_TEXTURE_MIN: usize = 1024;
const TRELLIS_PBR_TEXTURE_MAX: usize = 4096;
const TRELLIS_PBR_TEXTURE_STEP: usize = 1024;
const TRELLIS_FACE_STEP: usize = 100_000;
const TRELLIS_MAX_FACES: usize = 2_000_000;
const TRELLIS_SPARSE_COORD_STEP: usize = 512;
const TRELLIS_MAX_SPARSE_COORDS: usize = 49_152;
pub const UNSAVED_SCENE_ENTRY_ID: u32 = u32::MAX;
const DEFAULT_PIPELINE_OPTIONS: [SynthesisModel; 3] = [
    SynthesisModel::Triposg,
    SynthesisModel::Trellis,
    SynthesisModel::Triposplat,
];
const VIEWER_GROUND_Y_STEP: f32 = 0.05;
const VIEWER_CONTACT_TOLERANCE_STEP: f32 = 0.01;
const DEFAULT_VIEWER_FRUSTUM_LENGTH: f32 = 0.75;
const VIEWER_FRUSTUM_LENGTH_STEP: f32 = 0.05;
const VIEWER_CONTACT_TOLERANCE_MIN: f32 = 0.0;
const VIEWER_CONTACT_TOLERANCE_MAX: f32 = 0.25;
const VIEWER_GROUND_Y_MIN: f32 = -2.0;
const VIEWER_GROUND_Y_MAX: f32 = 2.0;
const VIEWER_FRUSTUM_LENGTH_MIN: f32 = 0.05;
const VIEWER_FRUSTUM_LENGTH_MAX: f32 = 3.0;
const VIEWER_DEPTH_CLOUD_MIN_GAUSSIANS: usize = 8_192;
const VIEWER_DEPTH_CLOUD_DEFAULT_GAUSSIANS: usize = 262_144;
const VIEWER_DEPTH_CLOUD_MAX_GAUSSIANS: usize = 1280 * 720;
const VIEWER_DEPTH_CLOUD_GAUSSIAN_STEP: usize = 32_768;
const PROCESSING_EVENT_LIMIT: usize = 16;
const PROCESSING_ARTIFACT_LIMIT: usize = 8;
const DEVELOPER_EVENT_ROWS: usize = 12;
const DEVELOPER_ARTIFACT_ROWS: usize = 10;
const DEVELOPER_VISUAL_ROWS: usize = 3;
const DEVELOPER_VISUAL_THUMB_WIDTH: f32 = 144.0;
const DEVELOPER_VISUAL_THUMB_HEIGHT: f32 = 82.0;

#[derive(Component)]
pub struct MainCamera;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ViewerAabbOverlayMode {
    Off,
    #[default]
    Selected,
    All,
}

impl ViewerAabbOverlayMode {
    fn label(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Selected => "selected",
            Self::All => "all",
        }
    }
}

#[derive(Clone, Debug, Resource, PartialEq)]
pub struct ViewerDebugSettings {
    pub aabb_overlay: ViewerAabbOverlayMode,
    pub draw_ground_contact: bool,
    pub draw_scene_camera_frustum: bool,
    pub depth_cloud_overlay: bool,
    pub depth_cloud_max_gaussians: usize,
    pub ground_y: f32,
    pub contact_tolerance: f32,
    pub scene_camera_frustum_length: f32,
}

impl Default for ViewerDebugSettings {
    fn default() -> Self {
        Self {
            aabb_overlay: ViewerAabbOverlayMode::Selected,
            draw_ground_contact: true,
            draw_scene_camera_frustum: true,
            depth_cloud_overlay: false,
            depth_cloud_max_gaussians: VIEWER_DEPTH_CLOUD_DEFAULT_GAUSSIANS,
            ground_y: 0.0,
            contact_tolerance: 0.02,
            scene_camera_frustum_length: DEFAULT_VIEWER_FRUSTUM_LENGTH,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CatalogMode {
    #[default]
    Object,
    Scene,
}

impl CatalogMode {
    fn label(self) -> &'static str {
        match self {
            Self::Object => "object",
            Self::Scene => "scene",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ScenePipelineKind {
    #[default]
    Explicit,
}

impl ScenePipelineKind {
    fn label(self) -> &'static str {
        match self {
            Self::Explicit => "explicit",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SceneTablePoseRefinementSetting {
    Off,
    Geometry,
    #[default]
    GatedGpt,
    AlwaysGpt,
}

impl SceneTablePoseRefinementSetting {
    fn label(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Geometry => "geometry",
            Self::GatedGpt => "gated-gpt",
            Self::AlwaysGpt => "always-gpt",
        }
    }

    fn cycle(self, delta: isize) -> Self {
        const OPTIONS: [SceneTablePoseRefinementSetting; 4] = [
            SceneTablePoseRefinementSetting::Off,
            SceneTablePoseRefinementSetting::Geometry,
            SceneTablePoseRefinementSetting::GatedGpt,
            SceneTablePoseRefinementSetting::AlwaysGpt,
        ];
        let index = OPTIONS
            .iter()
            .position(|candidate| *candidate == self)
            .unwrap_or(2);
        let next = (index as isize + delta).rem_euclid(OPTIONS.len() as isize) as usize;
        OPTIONS[next]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CatalogPipelineChoice {
    Object(SynthesisModel),
    Scene(ScenePipelineKind),
}

impl CatalogPipelineChoice {
    fn label(self) -> &'static str {
        match self {
            Self::Object(model) => pipeline_label(model),
            Self::Scene(pipeline) => pipeline.label(),
        }
    }
}

#[derive(Clone, Debug, Resource)]
pub struct ScenePipelineUiSettings {
    pub pipeline: ScenePipelineKind,
    pub image_to_3d_model: SynthesisModel,
    pub quality_profile: SceneQualityProfileSetting,
    pub ground_calibration: SceneGroundCalibrationSetting,
    pub instance_generation: SceneInstanceGenerationSetting,
    pub table_pose_refinement: SceneTablePoseRefinementSetting,
    pub candidate_count: usize,
    pub feedback_iterations: usize,
    pub pbr_enabled: bool,
    pub pbr_texture_size: usize,
    pub target_faces: usize,
    pub allow_catalog_reuse: bool,
    pub lift_assets: bool,
    pub locate_anything_enabled: bool,
    pub depth_enabled: bool,
    pub segmentation_enabled: bool,
    pub pose_fit_enabled: bool,
    pub feedback_enabled: bool,
    pub write_artifacts: bool,
    pub promote_to_catalog: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SceneProcessingEvent {
    pub stage: String,
    pub phase: String,
    pub execution: String,
    pub message: String,
    pub elapsed_ms: u64,
    pub item_index: Option<usize>,
    pub item_count: Option<usize>,
    pub artifact_path: Option<String>,
    pub token_usage: Option<String>,
    pub is_failure: bool,
}

#[derive(Clone, Debug, Resource)]
pub struct SceneProcessingState {
    active: bool,
    run_id: Option<String>,
    source_label: Option<String>,
    wall_started_at: Option<Instant>,
    last_event_at: Option<Instant>,
    current_stage: String,
    current_phase: String,
    current_execution: String,
    current_message: String,
    elapsed_ms: u64,
    recent_events: VecDeque<SceneProcessingEvent>,
    recent_artifacts: VecDeque<String>,
    token_usage_summary: Option<String>,
    last_error: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum ProcessingArtifactVisualKind {
    Locate,
    Segmentation,
    Depth,
    IsolatedRender,
    Crop,
    Generated,
    Canonical,
    Projection,
    Feedback,
    Source,
    Other,
}

impl ProcessingArtifactVisualKind {
    fn label(&self) -> &'static str {
        match self {
            Self::Locate => "locate",
            Self::Segmentation => "mask",
            Self::Depth => "depth",
            Self::IsolatedRender => "isolated",
            Self::Crop => "crop",
            Self::Generated => "object",
            Self::Canonical => "canonical",
            Self::Projection => "projection",
            Self::Feedback => "feedback",
            Self::Source => "source",
            Self::Other => "image",
        }
    }

    fn priority(&self) -> usize {
        match self {
            Self::Locate => 0,
            Self::Segmentation => 1,
            Self::Depth => 2,
            Self::IsolatedRender => 3,
            Self::Crop => 4,
            Self::Generated => 5,
            Self::Canonical => 6,
            Self::Projection => 7,
            Self::Feedback => 8,
            Self::Source => 9,
            Self::Other => 10,
        }
    }
}

#[derive(Clone)]
struct ProcessingArtifactPreview {
    path: String,
    kind: ProcessingArtifactVisualKind,
    image: Handle<Image>,
}

#[derive(Resource, Default)]
struct ProcessingArtifactPreviewCache {
    signature: String,
    previews: Vec<ProcessingArtifactPreview>,
    total_count: usize,
    page: usize,
    page_count: usize,
}

impl Default for SceneProcessingState {
    fn default() -> Self {
        Self {
            active: false,
            run_id: None,
            source_label: None,
            wall_started_at: None,
            last_event_at: None,
            current_stage: "idle".to_string(),
            current_phase: "idle".to_string(),
            current_execution: "unknown".to_string(),
            current_message: "idle".to_string(),
            elapsed_ms: 0,
            recent_events: VecDeque::new(),
            recent_artifacts: VecDeque::new(),
            token_usage_summary: None,
            last_error: None,
        }
    }
}

impl SceneProcessingState {
    pub fn begin(&mut self, source_label: impl Into<String>) {
        let now = Instant::now();
        self.active = true;
        self.run_id = None;
        self.source_label = Some(source_label.into());
        self.wall_started_at = Some(now);
        self.last_event_at = Some(now);
        self.current_stage = "scene_build".to_string();
        self.current_phase = "started".to_string();
        self.current_execution = "mixed".to_string();
        self.current_message = "scene build queued".to_string();
        self.elapsed_ms = 0;
        self.recent_events.clear();
        self.recent_artifacts.clear();
        self.token_usage_summary = None;
        self.last_error = None;
    }

    pub fn push_event(&mut self, run_id: String, event: SceneProcessingEvent) {
        let now = Instant::now();
        if self.wall_started_at.is_none() {
            self.wall_started_at = now.checked_sub(Duration::from_millis(event.elapsed_ms));
        }
        self.last_event_at = Some(now);
        self.active = !event.phase.eq_ignore_ascii_case("completed")
            || !event.stage.eq_ignore_ascii_case("scene_build");
        self.run_id = Some(run_id);
        self.current_stage = event.stage.clone();
        self.current_phase = event.phase.clone();
        self.current_execution = event.execution.clone();
        self.current_message = event.message.clone();
        self.elapsed_ms = self.elapsed_ms.max(event.elapsed_ms);
        if event.is_failure {
            self.last_error = Some(event.message.clone());
            self.active = false;
        }
        if let Some(token_usage) = event.token_usage.as_ref() {
            self.token_usage_summary = Some(token_usage.clone());
        }
        if let Some(path) = event.artifact_path.as_ref()
            && !self
                .recent_artifacts
                .iter()
                .any(|existing| existing == path)
        {
            self.recent_artifacts.push_front(path.clone());
            while self.recent_artifacts.len() > PROCESSING_ARTIFACT_LIMIT {
                self.recent_artifacts.pop_back();
            }
        }
        self.recent_events.push_front(event);
        while self.recent_events.len() > PROCESSING_EVENT_LIMIT {
            self.recent_events.pop_back();
        }
    }

    pub fn finish_success(&mut self, message: impl Into<String>) {
        self.tick();
        self.active = false;
        self.current_stage = "scene_build".to_string();
        self.current_phase = "completed".to_string();
        self.current_execution = "mixed".to_string();
        self.current_message = message.into();
    }

    pub fn finish_failure(&mut self, message: impl Into<String>) {
        self.tick();
        self.active = false;
        let message = message.into();
        self.current_stage = "scene_build".to_string();
        self.current_phase = "failed".to_string();
        self.current_execution = "mixed".to_string();
        self.current_message = message.clone();
        self.last_error = Some(message);
    }

    pub fn is_visible(&self) -> bool {
        self.active || self.last_error.is_some() || !self.recent_events.is_empty()
    }

    pub fn token_usage_summary(&self) -> Option<&str> {
        self.token_usage_summary.as_deref()
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn artifact_roots(&self) -> Vec<PathBuf> {
        let mut roots = Vec::new();
        if let Some(run_id) = self.run_id.as_deref()
            && !run_id.trim().is_empty()
        {
            roots.push(PathBuf::from("tmp").join("runs").join(run_id));
        }
        for path in &self.recent_artifacts {
            let path = PathBuf::from(path);
            if path.is_file() {
                roots.push(path.clone());
                if let Some(parent) = path.parent() {
                    roots.push(parent.to_path_buf());
                    if let Some(run_root) = parent.parent() {
                        roots.push(run_root.to_path_buf());
                    }
                }
            } else {
                roots.push(path);
            }
        }
        roots
    }

    pub fn tick(&mut self) {
        if !self.active {
            return;
        }
        if let Some(started_at) = self.wall_started_at {
            self.elapsed_ms = self.elapsed_ms.max(saturating_elapsed_ms(started_at));
        }
    }

    fn last_event_age_ms(&self) -> Option<u64> {
        self.last_event_at.map(saturating_elapsed_ms)
    }
}

fn saturating_elapsed_ms(instant: Instant) -> u64 {
    instant.elapsed().as_millis().min(u128::from(u64::MAX)) as u64
}

impl Default for ScenePipelineUiSettings {
    fn default() -> Self {
        Self {
            pipeline: ScenePipelineKind::Explicit,
            image_to_3d_model: SynthesisModel::Trellis,
            quality_profile: SceneQualityProfileSetting::Fast,
            ground_calibration: SceneGroundCalibrationSetting::Gpt,
            instance_generation: SceneInstanceGenerationSetting::CategoryRepresentative,
            table_pose_refinement: SceneTablePoseRefinementSetting::GatedGpt,
            candidate_count: 1,
            feedback_iterations: 0,
            pbr_enabled: true,
            pbr_texture_size: DEFAULT_TRELLIS_PBR_TEXTURE_SIZE,
            target_faces: 80_000,
            allow_catalog_reuse: false,
            lift_assets: true,
            locate_anything_enabled: true,
            depth_enabled: true,
            segmentation_enabled: true,
            pose_fit_enabled: true,
            feedback_enabled: false,
            write_artifacts: true,
            promote_to_catalog: true,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SceneQualityProfileSetting {
    Fast,
    #[default]
    Balanced,
    Full,
}

impl SceneQualityProfileSetting {
    fn label(self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Balanced => "balanced",
            Self::Full => "full",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SceneGroundCalibrationSetting {
    DepthHeuristic,
    #[default]
    Gpt,
}

impl SceneGroundCalibrationSetting {
    fn label(self) -> &'static str {
        match self {
            Self::DepthHeuristic => "depth heuristic",
            Self::Gpt => "gpt",
        }
    }

    fn cycle(self, delta: isize) -> Self {
        match (self, delta >= 0) {
            (Self::DepthHeuristic, true) | (Self::Gpt, false) => Self::Gpt,
            (Self::Gpt, true) | (Self::DepthHeuristic, false) => Self::DepthHeuristic,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SceneInstanceGenerationSetting {
    #[default]
    CategoryRepresentative,
    FineGrainedTypes,
}

impl SceneInstanceGenerationSetting {
    fn label(self) -> &'static str {
        match self {
            Self::CategoryRepresentative => "per category",
            Self::FineGrainedTypes => "fine types",
        }
    }

    fn cycle(self, delta: isize) -> Self {
        match (self, delta >= 0) {
            (Self::CategoryRepresentative, true) | (Self::FineGrainedTypes, false) => {
                Self::FineGrainedTypes
            }
            (Self::FineGrainedTypes, true) | (Self::CategoryRepresentative, false) => {
                Self::CategoryRepresentative
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct ImagePickDialog;

#[derive(Clone, Debug)]
pub enum CatalogSpawnAsset {
    Mesh {
        mesh: Handle<BevyMesh>,
        material: Handle<StandardMaterial>,
    },
    GaussianSplat {
        cloud: Handle<PlanarGaussian3d>,
    },
}

#[derive(Clone, Debug)]
pub struct CatalogScenePreviewItem {
    pub asset: CatalogSpawnAsset,
    pub transform: Transform,
}

#[derive(Message, Clone, Debug)]
pub struct CatalogSpawnRequest {
    pub asset: CatalogSpawnAsset,
    pub transform: Transform,
    pub cache_key: Option<String>,
    pub select_spawned: bool,
}

#[derive(Message, Clone, Debug)]
pub struct CatalogDeleteRequest {
    pub cache_key: Option<String>,
}

#[derive(Message, Clone, Debug)]
pub struct SceneLoadRequest {
    pub scene_key: String,
}

#[derive(Message, Clone, Debug)]
pub struct SceneDeleteRequest {
    pub scene_key: String,
}

#[derive(Message, Clone, Debug)]
pub struct SceneSaveToCatalogRequest {
    pub label: Option<String>,
}

#[derive(Message, Clone, Debug)]
pub struct SceneRenameRequest {
    pub scene_key: String,
    pub label: String,
}

#[derive(Message, Clone, Debug)]
pub struct SceneBuildRequest {
    pub source_path: Option<std::path::PathBuf>,
    pub file_name: String,
    pub contents: Option<Vec<u8>>,
    pub settings: ScenePipelineUiSettings,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SceneSaveKind {
    Catalog,
    Bsn,
    Glb,
}

#[derive(Message, Clone, Debug)]
pub struct SceneSaveRequest {
    pub kind: SceneSaveKind,
}

pub struct BurnSynthUiPlugin;

#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BurnSynthUiSystemSet {
    CatalogRequests,
}

impl Plugin for BurnSynthUiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CatalogState>()
            .init_resource::<DragState>()
            .init_resource::<CatalogSelectionState>()
            .init_resource::<ScenePipelineUiSettings>()
            .init_resource::<ViewerDebugSettings>()
            .init_resource::<SceneProcessingState>()
            .init_resource::<ProcessingArtifactPreviewCache>()
            .init_resource::<DeveloperPanelState>()
            .init_resource::<CatalogModeDropdownState>()
            .init_resource::<SettingsModalState>()
            .init_resource::<PipelineDropdownState>()
            .init_resource::<SaveSceneMenuState>()
            .init_resource::<CatalogSourceImageModalState>()
            .add_message::<CatalogSpawnRequest>()
            .add_message::<CatalogDeleteRequest>()
            .add_message::<SceneLoadRequest>()
            .add_message::<SceneDeleteRequest>()
            .add_message::<SceneSaveToCatalogRequest>()
            .add_message::<SceneRenameRequest>()
            .add_message::<SceneBuildRequest>()
            .add_message::<SceneSaveRequest>()
            .add_observer(handle_settings_scroll)
            .add_systems(Startup, setup_ui)
            .add_systems(
                Update,
                (
                    update_queue_text,
                    handle_catalog_toggle,
                    handle_page_buttons,
                    handle_catalog_delete_button.in_set(BurnSynthUiSystemSet::CatalogRequests),
                    handle_catalog_delete_shortcut.in_set(BurnSynthUiSystemSet::CatalogRequests),
                    (
                        handle_catalog_mode_selector_button,
                        handle_catalog_mode_option_button,
                        sync_catalog_mode_dropdown,
                        update_catalog_mode_value_label,
                    )
                        .chain(),
                    (
                        handle_pipeline_selector_button,
                        handle_pipeline_option_button,
                        sync_pipeline_dropdown,
                        update_pipeline_value_label,
                    )
                        .chain(),
                ),
            )
            .add_systems(
                Update,
                (
                    (
                        handle_save_scene_button,
                        handle_save_scene_option_button,
                        sync_save_scene_menu,
                    )
                        .chain(),
                    handle_settings_button,
                    handle_settings_close_button,
                    handle_triposplat_profile_button,
                    handle_triposplat_setting_step_button,
                    handle_triposg_setting_step_button,
                    handle_trellis_quality_button,
                    handle_trellis_pbr_toggle_button,
                    handle_trellis_setting_step_button,
                    handle_scene_quality_button,
                    handle_scene_setting_step_button,
                    handle_scene_setting_toggle_button,
                    handle_settings_tab_button,
                    handle_developer_panel_tab_button,
                    handle_viewer_aabb_mode_button,
                    handle_viewer_debug_toggle_button,
                    handle_viewer_debug_step_button,
                    (
                        tick_processing_elapsed,
                        handle_developer_visual_page_button,
                        sync_processing_artifact_previews,
                        sync_settings_modal,
                        sync_settings_tab_visuals,
                        sync_developer_panel_tab_visuals,
                        sync_developer_visual_page_controls,
                        sync_settings_developer_panel,
                        sync_settings_developer_visual_grid,
                        update_settings_labels,
                        update_viewer_debug_labels,
                    )
                        .chain(),
                ),
            )
            .add_systems(
                Update,
                (
                    handle_source_image_modal_close_button,
                    handle_source_image_modal_tab_button,
                    handle_source_image_modal_escape,
                    sync_source_image_modal,
                    sync_source_image_modal_tab_visuals,
                    (sync_catalog_previews, rebuild_catalog_list).chain(),
                    update_button_visuals,
                    sync_processing_panel,
                    (
                        handle_catalog_entry_interaction,
                        handle_catalog_scene_load_button,
                        update_drag_ghost,
                        handle_drag_release.in_set(BurnSynthUiSystemSet::CatalogRequests),
                        cleanup_drag_ghosts,
                    )
                        .chain(),
                    spin_thumbnails,
                ),
            );

        app.add_systems(Update, handle_open_button);
    }
}

#[derive(Resource)]
pub struct CatalogState {
    entries: Vec<CatalogEntry>,
    active_mode: CatalogMode,
    expanded: bool,
    object_page: usize,
    scene_page: usize,
    available_layers: Vec<usize>,
    revision: u64,
}

impl Default for CatalogState {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            active_mode: CatalogMode::Object,
            expanded: true,
            object_page: 0,
            scene_page: 0,
            available_layers: (1..=PREVIEW_MAX_LAYER)
                .rev()
                .filter(|layer| *layer != GIZMO_LAYER)
                .collect(),
            revision: 0,
        }
    }
}

impl CatalogState {
    pub fn is_empty(&self) -> bool {
        self.active_entries_len() == 0
    }

    pub fn active_mode(&self) -> CatalogMode {
        self.active_mode
    }

    pub fn set_active_mode(&mut self, mode: CatalogMode) {
        if self.active_mode != mode {
            self.active_mode = mode;
            self.clamp_page();
            self.bump_revision();
        }
    }

    pub fn add_pending(&mut self, request: &InferenceRequest) {
        let label = request
            .image_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("image")
            .to_string();
        self.entries.push(CatalogEntry {
            id: request.id,
            kind: CatalogEntryKind::Object,
            label,
            status: CatalogStatus::Pending,
            mesh: None,
            material: None,
            gaussian: None,
            scene_key: None,
            scene_items: Vec::new(),
            scene_pipeline: None,
            scene_metrics: None,
            scene_artifact_dir: None,
            source_image_path: Some(request.image_path.display().to_string()),
            source_image: None,
            cache_key: None,
            preview: None,
        });
        self.clamp_page();
        self.bump_revision();
    }

    pub fn add_ready(
        &mut self,
        id: u32,
        label: String,
        mesh: Handle<BevyMesh>,
        material: Handle<StandardMaterial>,
        source_image_path: Option<String>,
        cache_key: Option<String>,
    ) {
        self.entries.push(CatalogEntry {
            id,
            kind: CatalogEntryKind::Object,
            label,
            status: CatalogStatus::Ready,
            mesh: Some(mesh),
            material: Some(material),
            gaussian: None,
            scene_key: None,
            scene_items: Vec::new(),
            scene_pipeline: None,
            scene_metrics: None,
            scene_artifact_dir: None,
            source_image_path,
            source_image: None,
            cache_key,
            preview: None,
        });
        self.clamp_page();
        self.bump_revision();
    }

    pub fn add_ready_without_preview(
        &mut self,
        id: u32,
        label: String,
        source_image_path: Option<String>,
        cache_key: Option<String>,
    ) {
        self.entries.push(CatalogEntry {
            id,
            kind: CatalogEntryKind::Object,
            label,
            status: CatalogStatus::Ready,
            mesh: None,
            material: None,
            gaussian: None,
            scene_key: None,
            scene_items: Vec::new(),
            scene_pipeline: None,
            scene_metrics: None,
            scene_artifact_dir: None,
            source_image_path,
            source_image: None,
            cache_key,
            preview: None,
        });
        self.clamp_page();
        self.bump_revision();
    }

    pub fn add_ready_gaussian_splat(
        &mut self,
        id: u32,
        label: String,
        gaussian: Handle<PlanarGaussian3d>,
        source_image_path: Option<String>,
        cache_key: Option<String>,
    ) {
        self.entries.push(CatalogEntry {
            id,
            kind: CatalogEntryKind::Object,
            label,
            status: CatalogStatus::Ready,
            mesh: None,
            material: None,
            gaussian: Some(gaussian),
            scene_key: None,
            scene_items: Vec::new(),
            scene_pipeline: None,
            scene_metrics: None,
            scene_artifact_dir: None,
            source_image_path,
            source_image: None,
            cache_key,
            preview: None,
        });
        self.clamp_page();
        self.bump_revision();
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_ready_scene(
        &mut self,
        id: u32,
        label: String,
        scene_key: Option<String>,
        scene_items: Vec<CatalogScenePreviewItem>,
        source_image_path: Option<String>,
        source_image: Option<Handle<Image>>,
        pipeline: Option<String>,
        metrics: Option<CachedSceneMetrics>,
        artifact_dir: Option<String>,
    ) {
        self.remove_entry(id);
        self.entries.push(CatalogEntry {
            id,
            kind: CatalogEntryKind::Scene,
            label,
            status: CatalogStatus::Ready,
            mesh: None,
            material: None,
            gaussian: None,
            scene_key,
            scene_items,
            scene_pipeline: pipeline,
            scene_metrics: metrics,
            scene_artifact_dir: artifact_dir,
            source_image_path,
            source_image,
            cache_key: None,
            preview: None,
        });
        self.clamp_page();
        self.bump_revision();
    }

    pub fn upsert_unsaved_scene(
        &mut self,
        scene_items: Vec<CatalogScenePreviewItem>,
        source_image: Option<Handle<Image>>,
    ) {
        self.add_ready_scene(
            UNSAVED_SCENE_ENTRY_ID,
            "unsaved scene".to_string(),
            None,
            scene_items,
            None,
            source_image,
            Some("current".to_string()),
            None,
            None,
        );
    }

    pub fn entry(&self, id: u32) -> Option<&CatalogEntry> {
        self.entries.iter().find(|entry| entry.id == id)
    }

    pub fn entry_mut(&mut self, id: u32) -> Option<&mut CatalogEntry> {
        self.entries.iter_mut().find(|entry| entry.id == id)
    }

    pub fn set_source_image(&mut self, id: u32, image: Option<Handle<Image>>) {
        if let Some(entry) = self.entry_mut(id) {
            entry.source_image = image;
            self.bump_revision();
        }
    }

    pub fn remove_entry(&mut self, id: u32) -> Option<CatalogEntry> {
        let index = self.entries.iter().position(|entry| entry.id == id)?;
        let removed = self.entries.remove(index);
        self.clamp_page();
        self.bump_revision();
        Some(removed)
    }

    pub fn bump_revision(&mut self) {
        self.revision = self.revision.wrapping_add(1);
    }

    pub fn alloc_preview_layer(&mut self) -> Option<usize> {
        self.available_layers.pop()
    }

    pub fn release_preview_layer(&mut self, layer: usize) {
        if layer == 0 || layer == GIZMO_LAYER || layer > PREVIEW_MAX_LAYER {
            return;
        }
        if !self.available_layers.contains(&layer) {
            self.available_layers.push(layer);
        }
    }

    pub fn page_count(&self) -> usize {
        let total = self.active_entries_len();
        if total == 0 {
            1
        } else {
            total.div_ceil(CATALOG_PAGE_SIZE)
        }
    }

    pub fn clamp_page(&mut self) {
        let max_page = self.page_count().saturating_sub(1);
        let page = self.active_page_mut();
        if *page > max_page {
            *page = max_page;
        }
    }

    pub fn set_page(&mut self, page: usize) {
        *self.active_page_mut() = page;
        self.clamp_page();
    }

    pub fn page(&self) -> usize {
        match self.active_mode {
            CatalogMode::Object => self.object_page,
            CatalogMode::Scene => self.scene_page,
        }
    }

    pub fn visible_indices(&self) -> Vec<usize> {
        let active = self.active_indices();
        let total = active.len();
        if total == 0 {
            return Vec::new();
        }
        let page = self.page();
        let start = total.saturating_sub(CATALOG_PAGE_SIZE * (page + 1));
        let end = total.saturating_sub(CATALOG_PAGE_SIZE * page);
        (start..end).rev().map(|index| active[index]).collect()
    }

    pub fn has_ready_cube_entry(&self) -> bool {
        self.entries.iter().any(|entry| {
            entry.kind == CatalogEntryKind::Object
                && matches!(entry.status, CatalogStatus::Ready)
                && entry.label.eq_ignore_ascii_case("cube")
        })
    }

    pub fn has_object_cache_key(&self, cache_key: &str) -> bool {
        self.entries.iter().any(|entry| {
            entry.kind == CatalogEntryKind::Object
                && entry
                    .cache_key
                    .as_deref()
                    .is_some_and(|key| key == cache_key)
        })
    }

    fn active_entries_len(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.mode() == self.active_mode)
            .count()
    }

    fn active_indices(&self) -> Vec<usize> {
        self.entries
            .iter()
            .enumerate()
            .filter_map(|(index, entry)| (entry.mode() == self.active_mode).then_some(index))
            .collect()
    }

    fn active_page_mut(&mut self) -> &mut usize {
        match self.active_mode {
            CatalogMode::Object => &mut self.object_page,
            CatalogMode::Scene => &mut self.scene_page,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CatalogEntryKind {
    Object,
    Scene,
}

pub struct CatalogEntry {
    pub id: u32,
    pub kind: CatalogEntryKind,
    pub label: String,
    pub status: CatalogStatus,
    pub mesh: Option<Handle<BevyMesh>>,
    pub material: Option<Handle<StandardMaterial>>,
    pub gaussian: Option<Handle<PlanarGaussian3d>>,
    pub scene_key: Option<String>,
    pub scene_items: Vec<CatalogScenePreviewItem>,
    pub scene_pipeline: Option<String>,
    pub scene_metrics: Option<CachedSceneMetrics>,
    pub scene_artifact_dir: Option<String>,
    pub source_image_path: Option<String>,
    pub source_image: Option<Handle<Image>>,
    pub cache_key: Option<String>,
    pub preview: Option<PreviewScene>,
}

impl CatalogEntry {
    fn mode(&self) -> CatalogMode {
        match self.kind {
            CatalogEntryKind::Object => CatalogMode::Object,
            CatalogEntryKind::Scene => CatalogMode::Scene,
        }
    }

    fn is_unsaved_scene(&self) -> bool {
        self.kind == CatalogEntryKind::Scene && self.id == UNSAVED_SCENE_ENTRY_ID
    }
}

#[derive(Clone, Debug)]
pub enum CatalogStatus {
    Pending,
    Ready,
    Failed(String),
}

pub struct PreviewScene {
    pub image: Handle<Image>,
    pub asset_entities: Vec<Entity>,
    pub camera_entity: Entity,
    pub light_entities: Vec<Entity>,
    pub layer_index: usize,
}

#[derive(Clone, Copy, Debug)]
struct PreviewFit {
    mesh_translation: Vec3,
    mesh_scale: f32,
    radius: f32,
}

#[derive(Clone)]
enum PreviewAsset {
    Mesh {
        mesh: Handle<BevyMesh>,
        material: Handle<StandardMaterial>,
    },
    GaussianSplat {
        cloud: Handle<PlanarGaussian3d>,
    },
    Scene {
        items: Vec<CatalogScenePreviewItem>,
    },
}

impl PreviewFit {
    fn fallback() -> Self {
        Self {
            mesh_translation: Vec3::ZERO,
            mesh_scale: 1.0,
            radius: PREVIEW_FALLBACK_RADIUS,
        }
    }
}

#[derive(Resource)]
pub struct CatalogUiState {
    list_entity: Entity,
    last_revision: u64,
    last_expanded: bool,
    panel_width: f32,
    catalog_mode_menu_open: bool,
    settings_modal_open: bool,
    source_modal_open: bool,
    pipeline_menu_open: bool,
    save_menu_open: bool,
}

impl CatalogUiState {
    pub fn cursor_over_ui(&self, window: &Window) -> bool {
        if self.catalog_mode_menu_open
            || self.settings_modal_open
            || self.source_modal_open
            || self.pipeline_menu_open
            || self.save_menu_open
        {
            return true;
        }
        window
            .cursor_position()
            .map(|cursor| {
                let from_top = (window.height() - cursor.y).max(0.0);
                cursor.x <= self.panel_width || from_top <= MENU_HEIGHT
            })
            .unwrap_or(false)
    }
}

#[derive(Resource, Default)]
pub struct DragState {
    active: Option<u32>,
    ghost: Option<Entity>,
    ghost_entry: Option<u32>,
}

impl DragState {
    pub fn is_dragging(&self) -> bool {
        self.active.is_some()
    }
}

#[derive(Resource, Default)]
struct CatalogSelectionState {
    selected: Option<u32>,
    last_pressed: Option<(u32, f64)>,
}

#[derive(Component)]
struct QueueText;

#[derive(Component)]
struct QueueStatusBadge;

#[derive(Component)]
struct QueueStatusDot;

#[derive(Component)]
struct ProcessingPanelRoot;

#[derive(Component)]
struct ProcessingCurrentText;

#[derive(Component)]
struct ProcessingTimelineText;

#[derive(Component)]
struct ProcessingArtifactText;

#[derive(Component)]
struct ProcessingErrorText;

#[derive(Component, Default)]
struct SettingsDeveloperCurrentText;

#[derive(Component, Default)]
struct SettingsDeveloperTokenText;

#[derive(Component, Default)]
struct SettingsDeveloperEventsText;

#[derive(Component, Default)]
struct SettingsDeveloperArtifactText;

#[derive(Component, Default)]
struct SettingsDeveloperVisualText;

#[derive(Component)]
struct SettingsDeveloperTabButton {
    tab: DeveloperPanelTab,
}

#[derive(Component)]
struct SettingsDeveloperVisualPageButton {
    direction: DeveloperVisualPageDirection,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DeveloperVisualPageDirection {
    Previous,
    Next,
}

#[derive(Component, Default)]
struct SettingsDeveloperVisualPagerText;

#[derive(Component)]
struct SettingsDeveloperTabPanel {
    tab: DeveloperPanelTab,
}

#[derive(Component, Default)]
struct SettingsDeveloperVisualGrid {
    signature: String,
}

#[derive(Component)]
struct CatalogList;

#[derive(Component)]
struct CatalogToggleButton;

#[derive(Component)]
struct ToggleLabel;

#[derive(Component)]
struct CatalogModeDropdownHost;

#[derive(Component)]
struct CatalogModeSelectorButton;

#[derive(Component)]
struct CatalogModeOptionButton {
    mode: CatalogMode,
}

#[derive(Component)]
struct CatalogModeDropdownRoot;

#[derive(Component)]
struct CatalogModeValueLabel;

#[derive(Component)]
struct CatalogEntryButton {
    id: u32,
}

#[derive(Component)]
struct CatalogSceneLoadButton {
    id: u32,
}

#[derive(Component)]
struct CatalogPrevButton;

#[derive(Component)]
struct CatalogNextButton;

#[derive(Component)]
struct PageLabel;

#[derive(Component)]
struct CatalogDeleteButton;

#[derive(Component)]
struct OpenImageButton;

#[derive(Component)]
struct SaveSceneButton;

#[derive(Component)]
struct SaveSceneOptionButton {
    kind: SceneSaveKind,
}

#[derive(Component)]
struct SaveSceneMenuRoot;

#[derive(Component)]
struct PipelineDropdownHost;

#[derive(Component)]
struct PipelineSelectorButton;

#[derive(Component)]
struct PipelineOptionButton {
    choice: CatalogPipelineChoice,
}

#[derive(Component)]
struct PipelineDropdownRoot;

#[derive(Component)]
struct PipelineValueLabel;

#[derive(Component)]
struct SettingsButton;

#[derive(Component)]
struct SettingsCloseButton;

#[derive(Component)]
struct CatalogSourceImageModalRoot;

#[derive(Component)]
struct CatalogSourceImageCloseButton;

#[derive(Component)]
struct CatalogSourceImageTabButton {
    tab: CatalogSourceImageTab,
}

#[derive(Component)]
struct CatalogSourceImageTabPanel {
    tab: CatalogSourceImageTab,
}

#[derive(Component)]
struct SettingsModalRoot;

#[derive(Component)]
struct SettingsTabButton {
    tab: SettingsModalTab,
}

#[derive(Component)]
struct SettingsTabPanel {
    tab: SettingsModalTab,
}

#[derive(Component)]
struct SettingsScrollArea;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum SettingsModalTab {
    #[default]
    Pipeline,
    Generation,
    Grounding,
    Debug,
    General,
    Physics,
    Developer,
}

impl SettingsModalTab {
    fn label(self) -> &'static str {
        match self {
            Self::Pipeline => "pipeline",
            Self::Generation => "generation",
            Self::Grounding => "grounding",
            Self::Debug => "debug",
            Self::General => "general",
            Self::Physics => "physics",
            Self::Developer => "developer",
        }
    }
}

#[derive(Component)]
struct TripoSplatProfileButton {
    profile: TripoSplatProfile,
}

#[derive(Component)]
struct TripoSplatSettingStepButton {
    setting: TripoSplatSetting,
    delta: TripoSplatSettingDelta,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TripoSplatSetting {
    Steps,
    Guidance,
    Gaussians,
}

#[derive(Clone, Copy, Debug)]
enum TripoSplatSettingDelta {
    Integer(isize),
    Float(f32),
}

#[derive(Component)]
struct TripoSplatSettingValueLabel {
    setting: TripoSplatSetting,
}

#[derive(Component)]
struct TripoSplatProfileValueLabel;

#[derive(Component)]
struct TripoSgSettingStepButton {
    setting: TripoSgSetting,
    delta: TripoSgSettingDelta,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TripoSgSetting {
    Steps,
    Tokens,
    Guidance,
    TargetFaces,
}

#[derive(Clone, Copy, Debug)]
enum TripoSgSettingDelta {
    Integer(isize),
    Float(f32),
}

#[derive(Component)]
struct TripoSgSettingValueLabel {
    setting: TripoSgSetting,
}

#[derive(Component)]
struct TrellisQualityButton {
    quality: TrellisQuality,
}

#[derive(Component)]
struct TrellisPbrToggleButton;

#[derive(Component)]
struct TrellisSettingStepButton {
    setting: TrellisSetting,
    delta: TrellisSettingDelta,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TrellisSetting {
    Resolution,
    Pbr,
    PbrTextureSize,
    TargetFaces,
    MaxSparseCoords,
}

#[derive(Clone, Copy, Debug)]
enum TrellisSettingDelta {
    Integer(isize),
}

#[derive(Component)]
struct SceneQualityButton {
    quality: SceneQualityProfileSetting,
}

#[derive(Component)]
struct SceneSettingStepButton {
    setting: SceneSetting,
    delta: SceneSettingDelta,
}

#[derive(Component)]
struct SceneSettingToggleButton {
    setting: SceneToggleSetting,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SceneSetting {
    GroundCalibration,
    InstanceGeneration,
    TablePoseRefinement,
    CandidateCount,
    FeedbackIterations,
    PbrTextureSize,
    TargetFaces,
}

#[derive(Clone, Copy, Debug)]
enum SceneSettingDelta {
    Integer(isize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SceneToggleSetting {
    Pbr,
    CatalogReuse,
    LiftAssets,
    LocateAnything,
    Depth,
    Segmentation,
    PoseFit,
    Feedback,
    WriteArtifacts,
    PromoteToCatalog,
}

#[derive(Component)]
struct SceneSettingValueLabel {
    setting: SceneSetting,
}

#[derive(Component)]
struct SceneToggleValueLabel {
    setting: SceneToggleSetting,
}

#[derive(Component)]
struct SceneQualityValueLabel;

#[derive(Component)]
struct SceneImageTo3dModelValueLabel;

#[derive(Component)]
struct ViewerAabbModeButton {
    mode: ViewerAabbOverlayMode,
}

#[derive(Component)]
struct ViewerAabbModeValueLabel;

#[derive(Component)]
struct ViewerDebugToggleButton {
    setting: ViewerDebugToggleSetting,
}

#[derive(Component)]
struct ViewerDebugStepButton {
    setting: ViewerDebugNumericSetting,
    delta: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ViewerDebugToggleSetting {
    GroundContact,
    SceneCameraFrustum,
    DepthCloud,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ViewerDebugNumericSetting {
    GroundY,
    ContactTolerance,
    SceneCameraFrustumLength,
    DepthCloudMaxGaussians,
}

#[derive(Component)]
struct ViewerDebugToggleValueLabel {
    setting: ViewerDebugToggleSetting,
}

#[derive(Component)]
struct ViewerDebugNumericValueLabel {
    setting: ViewerDebugNumericSetting,
}

#[derive(Component)]
struct TrellisSettingValueLabel {
    setting: TrellisSetting,
}

#[derive(Component)]
struct TrellisQualityValueLabel;

#[derive(Component)]
struct ThumbnailSpin;

#[derive(Component)]
struct DragGhost;

#[derive(Component, Clone, Copy)]
struct ControlButton(ControlButtonKind);

#[derive(Clone, Copy)]
enum ControlButtonKind {
    Primary,
    Secondary,
    Nav,
}

#[derive(Component)]
struct ButtonLabel;

#[derive(Component)]
pub struct UiRootNode;

#[derive(Resource, Default)]
struct CatalogModeDropdownState {
    open: bool,
    entity: Option<Entity>,
}

#[derive(Resource, Default)]
struct SettingsModalState {
    open: bool,
    entity: Option<Entity>,
    pipeline: Option<CatalogPipelineChoice>,
    tab: SettingsModalTab,
}

#[derive(Resource, Default)]
struct DeveloperPanelState {
    tab: DeveloperPanelTab,
    visual_page: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum DeveloperPanelTab {
    #[default]
    Status,
    Events,
    Artifacts,
    Visuals,
}

impl DeveloperPanelTab {
    fn label(self) -> &'static str {
        match self {
            Self::Status => "status",
            Self::Events => "events",
            Self::Artifacts => "artifacts",
            Self::Visuals => "visuals",
        }
    }
}

#[derive(Resource, Default)]
struct PipelineDropdownState {
    open: bool,
    entity: Option<Entity>,
}

#[derive(Resource, Default)]
struct SaveSceneMenuState {
    open: bool,
    entity: Option<Entity>,
}

#[derive(Resource, Default)]
struct CatalogSourceImageModalState {
    entry_id: Option<u32>,
    rendered_entry_id: Option<u32>,
    entity: Option<Entity>,
    tab: CatalogSourceImageTab,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum CatalogSourceImageTab {
    #[default]
    Image,
    Stats,
}

impl CatalogSourceImageTab {
    fn label(self) -> &'static str {
        match self {
            Self::Image => "image",
            Self::Stats => "stats",
        }
    }
}

#[derive(Resource, Clone, Debug, PartialEq, Eq)]
struct AvailablePipelines {
    object_models: Vec<SynthesisModel>,
    scene_pipelines: Vec<ScenePipelineKind>,
}

fn pipeline_label(model: SynthesisModel) -> &'static str {
    match model {
        SynthesisModel::Triposg => "TripoSG",
        SynthesisModel::Trellis => "Trellis.2",
        SynthesisModel::Triposplat => "TripoSplat",
    }
}

fn pipeline_selector_value_text(choice: CatalogPipelineChoice, option_count: usize) -> String {
    if option_count <= 1 {
        format!("{} only", choice.label())
    } else {
        choice.label().to_string()
    }
}

fn available_pipeline_models(args: Option<&AppArgs>) -> Vec<SynthesisModel> {
    let models: Box<dyn Iterator<Item = SynthesisModel> + '_> = match args {
        Some(args) => Box::new(args.available_synthesis_models.iter().copied()),
        None => Box::new(DEFAULT_PIPELINE_OPTIONS.into_iter()),
    };
    let mut out = Vec::new();
    for model in models {
        if !out.contains(&model) {
            out.push(model);
        }
    }
    if out.is_empty() {
        out.push(SynthesisModel::Triposg);
    }
    out
}

fn active_pipeline_choices(
    mode: CatalogMode,
    available: Option<&AvailablePipelines>,
) -> Vec<CatalogPipelineChoice> {
    match mode {
        CatalogMode::Object => available
            .map(|available| {
                available
                    .object_models
                    .iter()
                    .copied()
                    .map(CatalogPipelineChoice::Object)
                    .collect()
            })
            .unwrap_or_else(|| {
                DEFAULT_PIPELINE_OPTIONS
                    .into_iter()
                    .map(CatalogPipelineChoice::Object)
                    .collect()
            }),
        CatalogMode::Scene => available
            .map(|available| {
                available
                    .scene_pipelines
                    .iter()
                    .copied()
                    .map(CatalogPipelineChoice::Scene)
                    .collect()
            })
            .unwrap_or_else(|| vec![CatalogPipelineChoice::Scene(ScenePipelineKind::Explicit)]),
    }
}

fn pipeline_available(
    available: Option<&AvailablePipelines>,
    choice: CatalogPipelineChoice,
) -> bool {
    match choice {
        CatalogPipelineChoice::Object(model) => available
            .map(|available| available.object_models.contains(&model))
            .unwrap_or(true),
        CatalogPipelineChoice::Scene(pipeline) => available
            .map(|available| available.scene_pipelines.contains(&pipeline))
            .unwrap_or(true),
    }
}

fn pipeline_supported(args: Option<&AppArgs>, choice: CatalogPipelineChoice) -> bool {
    let Some(args) = args else {
        return true;
    };
    match choice {
        CatalogPipelineChoice::Scene(_) => true,
        CatalogPipelineChoice::Object(model) => match model {
            SynthesisModel::Triposplat => triposplat_supported_for_backend(args.backend.clone()),
            SynthesisModel::Trellis => trellis_supported_for_backend(args.backend.clone()),
            SynthesisModel::Triposg => true,
        },
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn triposplat_supported_for_backend(backend: BackendKind) -> bool {
    matches!(backend, BackendKind::Wgpu | BackendKind::Cuda)
}

#[cfg(target_arch = "wasm32")]
fn triposplat_supported_for_backend(backend: BackendKind) -> bool {
    matches!(backend, BackendKind::Wgpu)
}

#[cfg(not(target_arch = "wasm32"))]
fn trellis_supported_for_backend(backend: BackendKind) -> bool {
    matches!(backend, BackendKind::Wgpu | BackendKind::Cuda)
}

#[cfg(target_arch = "wasm32")]
fn trellis_supported_for_backend(backend: BackendKind) -> bool {
    matches!(backend, BackendKind::Wgpu)
}

fn setup_ui(mut commands: Commands, args: Option<Res<AppArgs>>) {
    let mut list_entity = Entity::PLACEHOLDER;
    let pipeline_models = available_pipeline_models(args.as_deref());
    commands.insert_resource(AvailablePipelines {
        object_models: pipeline_models,
        scene_pipelines: vec![ScenePipelineKind::Explicit],
    });

    let root = commands
        .spawn((
            UiRootNode,
            // Keep world-space picking active in empty viewport regions.
            Pickable::IGNORE,
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(0.0),
                left: Val::Px(0.0),
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                flex_direction: FlexDirection::Column,
                ..default()
            },
        ))
        .id();

    commands.entity(root).with_children(|parent| {
        parent
            .spawn((
                Node {
                    width: Val::Percent(100.0),
                    height: Val::Px(MENU_HEIGHT),
                    padding: UiRect::horizontal(Val::Px(14.0)),
                    justify_content: JustifyContent::SpaceBetween,
                    align_items: AlignItems::Center,
                    ..default()
                },
                BackgroundColor(MENU_BG),
            ))
            .with_children(|menu| {
                menu.spawn(Node {
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Center,
                    column_gap: Val::Px(12.0),
                    ..default()
                })
                .with_children(|left| {
                    left.spawn((
                        Text::new("bevy_synth"),
                        TextFont::from_font_size(16.0),
                        TextColor(Color::srgb(0.92, 0.94, 0.98)),
                    ));

                    left.spawn((
                        Button,
                        OpenImageButton,
                        ControlButton(ControlButtonKind::Primary),
                        Node {
                            padding: UiRect::axes(Val::Px(10.0), Val::Px(6.0)),
                            border: UiRect::all(Val::Px(1.0)),
                            ..default()
                        },
                        BorderColor::all(BUTTON_OPEN_BORDER),
                        BackgroundColor(BUTTON_OPEN_BG),
                    ))
                    .with_children(|button| {
                        button.spawn((
                            Text::new("open image"),
                            TextFont::from_font_size(13.0),
                            TextColor(BUTTON_TEXT),
                            ButtonLabel,
                        ));
                    });
                });

                menu.spawn(Node {
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Center,
                    column_gap: Val::Px(8.0),
                    ..default()
                })
                .with_children(|right| {
                    right
                        .spawn((
                            Button,
                            SaveSceneButton,
                            ControlButton(ControlButtonKind::Secondary),
                            Node {
                                position_type: PositionType::Relative,
                                padding: UiRect::axes(Val::Px(10.0), Val::Px(5.0)),
                                border: UiRect::all(Val::Px(1.0)),
                                overflow: Overflow::visible(),
                                ..default()
                            },
                            BorderColor::all(BUTTON_BORDER),
                            BackgroundColor(BUTTON_BG),
                        ))
                        .with_children(|button| {
                            button.spawn((
                                Text::new("save scene"),
                                TextFont::from_font_size(13.0),
                                TextColor(BUTTON_TEXT),
                                ButtonLabel,
                            ));
                        });

                    right
                        .spawn((
                            Button,
                            SettingsButton,
                            ControlButton(ControlButtonKind::Secondary),
                            Node {
                                padding: UiRect::axes(Val::Px(10.0), Val::Px(5.0)),
                                border: UiRect::all(Val::Px(1.0)),
                                ..default()
                            },
                            BorderColor::all(BUTTON_BORDER),
                            BackgroundColor(BUTTON_BG),
                        ))
                        .with_children(|button| {
                            button.spawn((
                                Text::new("settings"),
                                TextFont::from_font_size(13.0),
                                TextColor(BUTTON_TEXT),
                                ButtonLabel,
                            ));
                        });

                    right
                        .spawn((
                            Node {
                                width: Val::Px(STATUS_BADGE_WIDTH),
                                flex_direction: FlexDirection::Row,
                                align_items: AlignItems::Center,
                                justify_content: JustifyContent::FlexStart,
                                column_gap: Val::Px(7.0),
                                padding: UiRect::axes(Val::Px(10.0), Val::Px(5.0)),
                                border: UiRect::all(Val::Px(1.0)),
                                overflow: Overflow::clip_x(),
                                ..default()
                            },
                            BackgroundColor(STATUS_BADGE_BG),
                            BorderColor::all(STATUS_BADGE_BORDER),
                            QueueStatusBadge,
                        ))
                        .with_children(|badge| {
                            badge.spawn((
                                Node {
                                    width: Val::Px(8.0),
                                    height: Val::Px(8.0),
                                    ..default()
                                },
                                BackgroundColor(STATUS_IDLE),
                                QueueStatusDot,
                            ));
                            badge.spawn((
                                Text::new("idle"),
                                TextFont::from_font_size(13.0),
                                TextColor(Color::srgb(0.72, 0.76, 0.84)),
                                QueueText,
                            ));
                        });
                });
            });

        parent
            .spawn((
                ProcessingPanelRoot,
                Node {
                    position_type: PositionType::Absolute,
                    top: Val::Px(MENU_HEIGHT + 12.0),
                    right: Val::Px(12.0),
                    width: Val::Px(330.0),
                    max_height: Val::Px(220.0),
                    flex_direction: FlexDirection::Column,
                    row_gap: Val::Px(8.0),
                    padding: UiRect::all(Val::Px(12.0)),
                    border: UiRect::all(Val::Px(1.0)),
                    overflow: Overflow::clip_y(),
                    ..default()
                },
                Visibility::Hidden,
                BackgroundColor(Color::srgba(0.045, 0.05, 0.065, 0.94)),
                BorderColor::all(Color::srgb(0.24, 0.28, 0.36)),
            ))
            .with_children(|panel| {
                panel.spawn((
                    Text::new("processing"),
                    TextFont::from_font_size(13.0),
                    TextColor(Color::srgb(0.88, 0.91, 0.96)),
                    ProcessingCurrentText,
                ));
                panel.spawn((
                    Text::new(""),
                    TextFont::from_font_size(11.0),
                    TextColor(Color::srgb(0.72, 0.78, 0.88)),
                    ProcessingTimelineText,
                ));
                panel.spawn((
                    Text::new(""),
                    TextFont::from_font_size(10.0),
                    TextColor(Color::srgb(0.58, 0.66, 0.76)),
                    ProcessingArtifactText,
                ));
                panel.spawn((
                    Text::new(""),
                    TextFont::from_font_size(11.0),
                    TextColor(Color::srgb(0.96, 0.62, 0.62)),
                    ProcessingErrorText,
                ));
            });

        parent
            .spawn((
                Node {
                    width: Val::Px(PANEL_WIDTH),
                    flex_direction: FlexDirection::Column,
                    align_items: AlignItems::Stretch,
                    row_gap: Val::Px(12.0),
                    padding: UiRect::all(Val::Px(14.0)),
                    border: UiRect::right(Val::Px(1.0)),
                    ..default()
                },
                BackgroundColor(PANEL_BG),
                BorderColor::all(PANEL_BORDER),
            ))
            .with_children(|panel| {
                panel
                    .spawn(Node {
                        width: Val::Percent(100.0),
                        height: Val::Px(30.0),
                        justify_content: JustifyContent::FlexStart,
                        align_items: AlignItems::Center,
                        column_gap: Val::Px(8.0),
                        ..default()
                    })
                    .with_children(|header| {
                        header
                            .spawn((
                                CatalogModeDropdownHost,
                                Node {
                                    position_type: PositionType::Relative,
                                    width: Val::Px(CATALOG_MODE_SELECTOR_WIDTH),
                                    height: Val::Px(28.0),
                                    overflow: Overflow::visible(),
                                    ..default()
                                },
                            ))
                            .with_children(|host| {
                                host.spawn((
                                    Button,
                                    CatalogModeSelectorButton,
                                    ControlButton(ControlButtonKind::Secondary),
                                    Node {
                                        width: Val::Percent(100.0),
                                        height: Val::Percent(100.0),
                                        padding: UiRect::axes(Val::Px(8.0), Val::Px(4.0)),
                                        border: UiRect::all(Val::Px(1.0)),
                                        align_items: AlignItems::Center,
                                        justify_content: JustifyContent::Center,
                                        ..default()
                                    },
                                    BorderColor::all(BUTTON_ACTIVE_BORDER),
                                    BackgroundColor(BUTTON_ACTIVE_BG),
                                ))
                                .with_children(|button| {
                                    button.spawn((
                                        Text::new(CatalogMode::Object.label()),
                                        TextFont::from_font_size(13.0),
                                        TextColor(BUTTON_TEXT),
                                        CatalogModeValueLabel,
                                        ButtonLabel,
                                    ));
                                });
                            });
                        header
                            .spawn(Node {
                                flex_direction: FlexDirection::Row,
                                align_items: AlignItems::Center,
                                column_gap: Val::Px(6.0),
                                ..default()
                            })
                            .with_children(|controls| {
                                controls
                                    .spawn((
                                        Button,
                                        CatalogPrevButton,
                                        ControlButton(ControlButtonKind::Nav),
                                        Node {
                                            width: Val::Px(CATALOG_NAV_BUTTON_WIDTH),
                                            height: Val::Px(24.0),
                                            border: UiRect::all(Val::Px(1.0)),
                                            align_items: AlignItems::Center,
                                            justify_content: JustifyContent::Center,
                                            ..default()
                                        },
                                        BorderColor::all(BUTTON_BORDER),
                                        BackgroundColor(BUTTON_BG),
                                    ))
                                    .with_children(|button| {
                                        button.spawn((
                                            Text::new("<"),
                                            TextFont::from_font_size(12.0),
                                            TextColor(BUTTON_TEXT),
                                            ButtonLabel,
                                        ));
                                    });
                                controls.spawn((
                                    Text::new("1/1"),
                                    TextFont::from_font_size(12.0),
                                    TextColor(Color::srgb(0.78, 0.82, 0.9)),
                                    Node {
                                        width: Val::Px(CATALOG_PAGE_LABEL_WIDTH),
                                        justify_content: JustifyContent::Center,
                                        ..default()
                                    },
                                    PageLabel,
                                ));
                                controls
                                    .spawn((
                                        Button,
                                        CatalogNextButton,
                                        ControlButton(ControlButtonKind::Nav),
                                        Node {
                                            width: Val::Px(CATALOG_NAV_BUTTON_WIDTH),
                                            height: Val::Px(24.0),
                                            border: UiRect::all(Val::Px(1.0)),
                                            align_items: AlignItems::Center,
                                            justify_content: JustifyContent::Center,
                                            ..default()
                                        },
                                        BorderColor::all(BUTTON_BORDER),
                                        BackgroundColor(BUTTON_BG),
                                    ))
                                    .with_children(|button| {
                                        button.spawn((
                                            Text::new(">"),
                                            TextFont::from_font_size(12.0),
                                            TextColor(BUTTON_TEXT),
                                            ButtonLabel,
                                        ));
                                    });
                                controls
                                    .spawn((
                                        Button,
                                        CatalogDeleteButton,
                                        ControlButton(ControlButtonKind::Secondary),
                                        Node {
                                            width: Val::Px(CATALOG_DELETE_BUTTON_WIDTH),
                                            height: Val::Px(24.0),
                                            border: UiRect::all(Val::Px(1.0)),
                                            align_items: AlignItems::Center,
                                            justify_content: JustifyContent::Center,
                                            ..default()
                                        },
                                        BorderColor::all(BUTTON_BORDER),
                                        BackgroundColor(BUTTON_BG),
                                    ))
                                    .with_children(|button| {
                                        button.spawn((
                                            Text::new("delete"),
                                            TextFont::from_font_size(12.0),
                                            TextColor(BUTTON_TEXT),
                                            ButtonLabel,
                                        ));
                                    });
                                controls
                                    .spawn((
                                        Button,
                                        CatalogToggleButton,
                                        ControlButton(ControlButtonKind::Secondary),
                                        Node {
                                            width: Val::Px(CATALOG_TOGGLE_BUTTON_WIDTH),
                                            height: Val::Px(24.0),
                                            border: UiRect::all(Val::Px(1.0)),
                                            align_items: AlignItems::Center,
                                            justify_content: JustifyContent::Center,
                                            ..default()
                                        },
                                        BorderColor::all(BUTTON_BORDER),
                                        BackgroundColor(BUTTON_BG),
                                    ))
                                    .with_children(|button| {
                                        button.spawn((
                                            Text::new("hide"),
                                            TextFont::from_font_size(12.0),
                                            TextColor(BUTTON_TEXT),
                                            ToggleLabel,
                                            ButtonLabel,
                                        ));
                                    });
                            });
                    });

                let list = panel
                    .spawn((
                        Node {
                            width: Val::Percent(100.0),
                            flex_direction: FlexDirection::Column,
                            row_gap: Val::Px(ENTRY_GAP),
                            flex_grow: 1.0,
                            overflow: Overflow::clip_y(),
                            ..default()
                        },
                        CatalogList,
                    ))
                    .id();
                list_entity = list;
            });
    });

    commands.insert_resource(CatalogUiState {
        list_entity,
        last_revision: 0,
        last_expanded: true,
        panel_width: PANEL_WIDTH,
        catalog_mode_menu_open: false,
        settings_modal_open: false,
        source_modal_open: false,
        pipeline_menu_open: false,
        save_menu_open: false,
    });
}

#[allow(clippy::type_complexity)]
fn update_queue_text(
    queue: Res<InferenceQueue>,
    status: Option<Res<UiStatus>>,
    mut query: Query<(&mut Text, &mut TextColor), With<QueueText>>,
    mut dots: Query<&mut BackgroundColor, (With<QueueStatusDot>, Without<QueueStatusBadge>)>,
    mut badges: Query<
        (&mut BackgroundColor, &mut BorderColor),
        (With<QueueStatusBadge>, Without<QueueStatusDot>),
    >,
) {
    let (text, text_color, dot_color, badge_bg, badge_border) = if let Some(worker_message) = status
        .as_ref()
        .and_then(|state| state.worker_message.as_ref())
    {
        let is_failure = worker_message.to_ascii_lowercase().contains("failed");
        (
            compact_worker_status_text(worker_message),
            if is_failure {
                Color::srgb(0.96, 0.72, 0.72)
            } else {
                Color::srgb(0.76, 0.86, 0.98)
            },
            if is_failure {
                Color::srgb(0.86, 0.28, 0.28)
            } else {
                STATUS_PENDING
            },
            if is_failure {
                Color::srgb(0.22, 0.1, 0.1)
            } else {
                Color::srgb(0.08, 0.15, 0.2)
            },
            if is_failure {
                Color::srgb(0.58, 0.23, 0.23)
            } else {
                Color::srgb(0.2, 0.4, 0.55)
            },
        )
    } else if let Some(active) = queue.active.as_ref() {
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
        (
            format!("processing: {name} | queued: {}", queue.pending.len()),
            Color::srgb(0.95, 0.89, 0.74),
            STATUS_PROCESSING,
            Color::srgb(0.21, 0.15, 0.08),
            Color::srgb(0.58, 0.41, 0.18),
        )
    } else if !queue.pending.is_empty() {
        (
            format!("queued: {}", queue.pending.len()),
            Color::srgb(0.76, 0.86, 0.98),
            STATUS_PENDING,
            Color::srgb(0.08, 0.15, 0.2),
            Color::srgb(0.2, 0.4, 0.55),
        )
    } else {
        (
            "idle".to_string(),
            Color::srgb(0.72, 0.76, 0.84),
            STATUS_IDLE,
            STATUS_BADGE_BG,
            STATUS_BADGE_BORDER,
        )
    };

    for (mut node, mut color) in query.iter_mut() {
        if node.0 != text {
            node.0 = text.clone();
        }
        if color.0 != text_color {
            color.0 = text_color;
        }
    }
    for mut dot in dots.iter_mut() {
        if dot.0 != dot_color {
            dot.0 = dot_color;
        }
    }
    for (mut bg, mut border) in badges.iter_mut() {
        if bg.0 != badge_bg {
            bg.0 = badge_bg;
        }
        *border = BorderColor::all(badge_border);
    }
}

#[allow(clippy::type_complexity)]
fn tick_processing_elapsed(mut state: ResMut<SceneProcessingState>) {
    state.tick();
}

#[allow(clippy::type_complexity)]
fn sync_processing_panel(
    state: Res<SceneProcessingState>,
    mut roots: Query<&mut Visibility, With<ProcessingPanelRoot>>,
    mut text_queries: ParamSet<(
        Query<&mut Text, With<ProcessingCurrentText>>,
        Query<&mut Text, With<ProcessingTimelineText>>,
        Query<&mut Text, With<ProcessingArtifactText>>,
        Query<&mut Text, With<ProcessingErrorText>>,
    )>,
) {
    let visible = state.is_visible();
    for mut visibility in &mut roots {
        *visibility = if visible {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
    }
    if !visible {
        return;
    }

    let status = if state.active {
        "running"
    } else {
        &state.current_phase
    };
    let source = state
        .source_label
        .as_deref()
        .map(ellipsize_processing_text)
        .unwrap_or_else(|| "scene".to_string());
    let elapsed = format_elapsed_ms(state.elapsed_ms);
    let mut current_rows = vec![
        format!(
            "{status} | {elapsed} | {}",
            ellipsize_text(&state.current_stage, 32)
        ),
        source,
        format!(
            "{} | {}",
            state.current_phase,
            ellipsize_text(&state.current_execution, 16)
        ),
        ellipsize_text(&state.current_message, 64),
    ];
    if let Some(token_usage) = state.token_usage_summary.as_ref() {
        current_rows.push(ellipsize_text(token_usage, 64));
    }
    let current_text = current_rows.join("\n");
    for mut text in &mut text_queries.p0() {
        text.0 = current_text.clone();
    }

    let rows = state
        .recent_events
        .iter()
        .take(2)
        .map(format_processing_event)
        .collect::<Vec<_>>();
    let timeline_text = if rows.is_empty() {
        String::new()
    } else {
        rows.join("\n")
    };
    for mut text in &mut text_queries.p1() {
        text.0 = timeline_text.clone();
    }

    let artifact_text = state
        .recent_artifacts
        .iter()
        .take(1)
        .map(|path| format!("artifact: {}", ellipsize_text(path, 48)))
        .collect::<Vec<_>>()
        .join("\n");
    for mut text in &mut text_queries.p2() {
        text.0 = artifact_text.clone();
    }

    let error_text = state
        .last_error
        .as_deref()
        .map(|error| format!("error: {}", ellipsize_text(error, 96)))
        .unwrap_or_default();
    for mut text in &mut text_queries.p3() {
        text.0 = error_text.clone();
    }
}

#[allow(clippy::type_complexity)]
fn sync_settings_developer_panel(
    state: Res<SceneProcessingState>,
    artifact_previews: Res<ProcessingArtifactPreviewCache>,
    mut text_queries: ParamSet<(
        Query<&mut Text, With<SettingsDeveloperCurrentText>>,
        Query<&mut Text, With<SettingsDeveloperTokenText>>,
        Query<&mut Text, With<SettingsDeveloperEventsText>>,
        Query<&mut Text, With<SettingsDeveloperArtifactText>>,
        Query<&mut Text, With<SettingsDeveloperVisualText>>,
    )>,
) {
    let current_text = format_developer_current_block(&state);
    for mut text in &mut text_queries.p0() {
        text.0 = current_text.clone();
    }

    let token_text = format_developer_token_block(&state);
    for mut text in &mut text_queries.p1() {
        text.0 = token_text.clone();
    }

    let event_text = format_developer_event_block(&state);
    for mut text in &mut text_queries.p2() {
        text.0 = event_text.clone();
    }

    let artifact_text = format_developer_artifact_block(&state);
    for mut text in &mut text_queries.p3() {
        text.0 = artifact_text.clone();
    }

    let visual_text = format_developer_visual_block(&state, &artifact_previews);
    for mut text in &mut text_queries.p4() {
        text.0 = visual_text.clone();
    }
}

fn format_developer_current_block(state: &SceneProcessingState) -> String {
    let active = if state.active { "active" } else { "idle" };
    let last_event = state
        .last_event_age_ms()
        .map(format_elapsed_ms)
        .unwrap_or_else(|| "none".to_string());
    let error = state
        .last_error
        .as_deref()
        .map(|value| format!("\nerror: {}", ellipsize_text(value, 92)))
        .unwrap_or_default();
    format!(
        "state: {active}\nrun: {}\nsource: {}\nstage: {} / {} / {}\nelapsed: {} | last event: {last_event}\nmessage: {}{}",
        state.run_id.as_deref().unwrap_or("none"),
        ellipsize_text(state.source_label.as_deref().unwrap_or("none"), 74),
        ellipsize_text(&state.current_stage, 40),
        state.current_phase,
        state.current_execution,
        format_elapsed_ms(state.elapsed_ms),
        ellipsize_text(&state.current_message, 92),
        error
    )
}

fn format_developer_token_block(state: &SceneProcessingState) -> String {
    state
        .token_usage_summary
        .as_deref()
        .map(|summary| ellipsize_text(summary, 104))
        .unwrap_or_else(|| {
            if state.active {
                "waiting for provider token usage; local GPU stages do not emit token counts"
                    .to_string()
            } else {
                "no token usage reported yet".to_string()
            }
        })
}

fn format_developer_event_block(state: &SceneProcessingState) -> String {
    let rows = state
        .recent_events
        .iter()
        .take(DEVELOPER_EVENT_ROWS)
        .map(format_developer_event)
        .collect::<Vec<_>>();
    if rows.is_empty() {
        return if state.active {
            "worker is active; waiting for first progress event".to_string()
        } else {
            "no scene build events yet".to_string()
        };
    }
    rows.join("\n")
}

fn format_developer_artifact_block(state: &SceneProcessingState) -> String {
    let rows = state
        .recent_artifacts
        .iter()
        .take(DEVELOPER_ARTIFACT_ROWS)
        .map(|path| format!("{} {}", artifact_kind_label(path), ellipsize_text(path, 92)))
        .collect::<Vec<_>>();
    if rows.is_empty() {
        return if state.active {
            "waiting for first artifact path; generated files appear under tmp/runs/<run_id>"
                .to_string()
        } else {
            "no artifacts yet".to_string()
        };
    }
    rows.join("\n")
}

fn format_developer_visual_block(
    state: &SceneProcessingState,
    artifact_previews: &ProcessingArtifactPreviewCache,
) -> String {
    if artifact_previews.total_count == 0 {
        if state.active {
            "waiting for locate/depth/crop/canonical/feedback images".to_string()
        } else {
            "no visual artifacts yet".to_string()
        }
    } else {
        format!(
            "{} visual artifact(s) | latest first | page {}/{}",
            artifact_previews.total_count,
            artifact_previews.page + 1,
            artifact_previews.page_count.max(1)
        )
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn sync_processing_artifact_previews(
    state: Res<SceneProcessingState>,
    mut developer: ResMut<DeveloperPanelState>,
    mut cache: ResMut<ProcessingArtifactPreviewCache>,
    mut images: ResMut<Assets<Image>>,
) {
    let discovered = discover_processing_visual_artifacts(&state);
    let total_count = discovered.len();
    let page_count = total_count.div_ceil(DEVELOPER_VISUAL_ROWS);
    let max_page = page_count.saturating_sub(1);
    if developer.visual_page > max_page {
        developer.visual_page = max_page;
    }
    let page = developer.visual_page;
    let page_start = page.saturating_mul(DEVELOPER_VISUAL_ROWS);
    let signature = discovered
        .iter()
        .map(|(path, kind)| format!("{}:{}", kind.label(), path.display()))
        .collect::<Vec<_>>()
        .join("|");
    let signature = format!("page={page};total={total_count};{signature}");
    if cache.signature == signature {
        return;
    }

    cache.signature = signature;
    cache.total_count = total_count;
    cache.page = page;
    cache.page_count = page_count;
    cache.previews.clear();
    for (path, kind) in discovered
        .into_iter()
        .skip(page_start)
        .take(DEVELOPER_VISUAL_ROWS)
    {
        if let Some(image) = load_processing_artifact_preview(&path, &mut images) {
            cache.previews.push(ProcessingArtifactPreview {
                path: path.display().to_string(),
                kind,
                image,
            });
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn sync_processing_artifact_previews(mut cache: ResMut<ProcessingArtifactPreviewCache>) {
    if !cache.signature.is_empty() || !cache.previews.is_empty() || cache.total_count != 0 {
        cache.signature.clear();
        cache.previews.clear();
        cache.total_count = 0;
        cache.page = 0;
        cache.page_count = 0;
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn discover_processing_visual_artifacts(
    state: &SceneProcessingState,
) -> Vec<(PathBuf, ProcessingArtifactVisualKind)> {
    let mut roots = Vec::new();
    if let Some(run_id) = state.run_id.as_deref()
        && !run_id.trim().is_empty()
    {
        roots.push(PathBuf::from("tmp").join("runs").join(run_id));
    }
    for path in &state.recent_artifacts {
        roots.push(PathBuf::from(path));
    }

    let mut discovered = Vec::new();
    for root in roots {
        collect_visual_artifacts(&root, 0, &mut discovered);
        if discovered.len() >= 96 {
            break;
        }
    }

    sort_visual_artifacts_for_display(&mut discovered);
    discovered.dedup_by(|(left_path, _), (right_path, _)| left_path == right_path);
    discovered
}

#[cfg(not(target_arch = "wasm32"))]
fn visual_artifact_modified_ms(path: &Path) -> u128 {
    fs::metadata(path)
        .and_then(|metadata| metadata.modified())
        .ok()
        .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_millis())
        .unwrap_or_default()
}

#[cfg(not(target_arch = "wasm32"))]
fn visual_artifact_generation_order(path: &Path) -> Vec<u64> {
    let mut values = Vec::new();
    let mut current = String::new();
    for ch in path.to_string_lossy().chars() {
        if ch.is_ascii_digit() {
            current.push(ch);
        } else if !current.is_empty() {
            values.push(current.parse::<u64>().unwrap_or(u64::MAX));
            current.clear();
        }
    }
    if !current.is_empty() {
        values.push(current.parse::<u64>().unwrap_or(u64::MAX));
    }
    values
}

#[cfg(not(target_arch = "wasm32"))]
fn sort_visual_artifacts_for_display(discovered: &mut [(PathBuf, ProcessingArtifactVisualKind)]) {
    discovered.sort_by_cached_key(|(path, kind)| {
        (
            Reverse(visual_artifact_modified_ms(path)),
            Reverse(visual_artifact_generation_order(path)),
            kind.priority(),
            visual_artifact_score(path, kind),
            Reverse(path.clone()),
        )
    });
}

#[cfg(not(target_arch = "wasm32"))]
fn collect_visual_artifacts(
    path: &Path,
    depth: usize,
    out: &mut Vec<(PathBuf, ProcessingArtifactVisualKind)>,
) {
    if out.len() >= 128 || depth > 6 {
        return;
    }
    if path.is_file() {
        if let Some(kind) = visual_artifact_kind(path) {
            out.push((path.to_path_buf(), kind));
        }
        return;
    }
    if !path.is_dir() {
        return;
    }
    let Ok(entries) = fs::read_dir(path) else {
        return;
    };
    let mut entries = entries.filter_map(Result::ok).collect::<Vec<_>>();
    entries.sort_by_key(|entry| entry.path());
    for entry in entries {
        collect_visual_artifacts(&entry.path(), depth + 1, out);
        if out.len() >= 128 {
            break;
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn visual_artifact_kind(path: &Path) -> Option<ProcessingArtifactVisualKind> {
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if !matches!(extension.as_str(), "png" | "jpg" | "jpeg" | "webp") {
        return None;
    }

    let lower = path.to_string_lossy().to_ascii_lowercase();
    if lower.contains("detections_overlay") || lower.contains("locate") {
        Some(ProcessingArtifactVisualKind::Locate)
    } else if lower.contains("masks_overlay")
        || lower.contains("segmentation")
        || lower.contains("/mask")
        || lower.contains("\\mask")
    {
        Some(ProcessingArtifactVisualKind::Segmentation)
    } else if lower.contains("depth") || lower.contains("floor") {
        Some(ProcessingArtifactVisualKind::Depth)
    } else if lower.contains("current_isolated_full_frame")
        || lower.contains("isolated_render_full_frame")
        || (lower.contains("rotation_candidates") && lower.ends_with("_screenshot.png"))
    {
        Some(ProcessingArtifactVisualKind::IsolatedRender)
    } else if lower.contains("/crops/") || lower.contains("\\crops\\") || lower.contains("_crop") {
        Some(ProcessingArtifactVisualKind::Crop)
    } else if lower.contains("/generated/")
        || lower.contains("\\generated\\")
        || lower.contains("candidate")
    {
        Some(ProcessingArtifactVisualKind::Generated)
    } else if lower.contains("canonical") || lower.contains("yaw") {
        Some(ProcessingArtifactVisualKind::Canonical)
    } else if lower.contains("projection_fit")
        || lower.contains("visible_surface")
        || lower.contains("silhouette")
    {
        Some(ProcessingArtifactVisualKind::Projection)
    } else if lower.contains("/iterations")
        || lower.contains("\\iterations")
        || lower.contains("feedback")
        || lower.ends_with("screenshot.png")
    {
        Some(ProcessingArtifactVisualKind::Feedback)
    } else if lower.contains("source") || lower.contains("input") {
        Some(ProcessingArtifactVisualKind::Source)
    } else {
        Some(ProcessingArtifactVisualKind::Other)
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn visual_artifact_score(path: &Path, kind: &ProcessingArtifactVisualKind) -> usize {
    let lower = path.to_string_lossy().to_ascii_lowercase();
    match kind {
        ProcessingArtifactVisualKind::Locate if lower.contains("detections_overlay") => 0,
        ProcessingArtifactVisualKind::Segmentation if lower.contains("masks_overlay") => 0,
        ProcessingArtifactVisualKind::Depth if lower.contains("depth_overlay") => 0,
        ProcessingArtifactVisualKind::IsolatedRender
            if lower.contains("current_isolated_full_frame") =>
        {
            0
        }
        ProcessingArtifactVisualKind::IsolatedRender => 1,
        ProcessingArtifactVisualKind::Projection if lower.contains("projection_fit_overlay") => 0,
        ProcessingArtifactVisualKind::Feedback if lower.ends_with("screenshot.png") => 0,
        ProcessingArtifactVisualKind::Canonical if lower.contains("selection") => 0,
        ProcessingArtifactVisualKind::Crop => 1,
        ProcessingArtifactVisualKind::Generated => 1,
        _ => 2,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn load_processing_artifact_preview(
    path: &Path,
    images: &mut Assets<Image>,
) -> Option<Handle<Image>> {
    let bytes = fs::read(path).ok()?;
    let decoded = image::load_from_memory(&bytes).ok()?.to_rgba8();
    let (width, height) = decoded.dimensions();
    if width == 0 || height == 0 {
        return None;
    }
    let image = Image::new(
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        decoded.into_raw(),
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    Some(images.add(image))
}

fn format_processing_event(event: &SceneProcessingEvent) -> String {
    format_event_row(event, 72, true)
}

fn format_developer_event(event: &SceneProcessingEvent) -> String {
    format_event_row(event, 96, false)
}

fn format_event_row(
    event: &SceneProcessingEvent,
    max_message_chars: usize,
    compact: bool,
) -> String {
    let item = match (event.item_index, event.item_count) {
        (Some(index), Some(total)) => format!(" [{index}/{total}]"),
        (None, Some(total)) => format!(" [{total}]"),
        _ => String::new(),
    };
    let marker = if event.is_failure { "!" } else { "-" };
    let artifact = event
        .artifact_path
        .as_deref()
        .map(|path| format!(" -> {}", ellipsize_text(path, 40)))
        .unwrap_or_default();
    if compact {
        format!(
            "{marker} {} {} {}{}: {}",
            format_elapsed_ms(event.elapsed_ms),
            event.phase,
            event.stage,
            item,
            ellipsize_text(&event.message, max_message_chars)
        )
    } else {
        format!(
            "{marker} {} [{}] {} / {}{}: {}{}",
            format_elapsed_ms(event.elapsed_ms),
            event.execution,
            event.phase,
            event.stage,
            item,
            ellipsize_text(&event.message, max_message_chars),
            artifact
        )
    }
}

fn artifact_kind_label(path: &str) -> &'static str {
    let lower = path.to_ascii_lowercase();
    if lower.ends_with(".glb") || lower.ends_with(".gltf") || lower.ends_with(".splat") {
        "asset"
    } else if lower.ends_with(".png")
        || lower.ends_with(".jpg")
        || lower.ends_with(".jpeg")
        || lower.ends_with(".webp")
    {
        "image"
    } else if lower.ends_with(".bsn") {
        "bsn  "
    } else if lower.ends_with(".json") || lower.ends_with(".jsonl") {
        "json "
    } else if lower.contains("/assets") || lower.ends_with("assets") {
        "dir  "
    } else {
        "file "
    }
}

fn compact_worker_status_text(message: &str) -> String {
    let normalized = message.trim();
    if let Some(rest) = normalized.strip_prefix("scene ") {
        let (phase, stage_and_message) = rest.split_once(": ").unwrap_or((rest, ""));
        let (stage, _) = stage_and_message
            .split_once(" - ")
            .unwrap_or((stage_and_message, ""));
        if !stage.is_empty() {
            let label = format!("scene {phase}: {stage}");
            return ellipsize_text(&label, 34);
        }
    }
    ellipsize_text(normalized, 34)
}

fn format_elapsed_ms(elapsed_ms: u64) -> String {
    let seconds = elapsed_ms as f64 / 1000.0;
    if seconds < 60.0 {
        format!("{seconds:.1}s")
    } else {
        let minutes = (seconds / 60.0).floor() as u64;
        let seconds = (seconds as u64) % 60;
        format!("{minutes}:{seconds:02}")
    }
}

fn ellipsize_processing_text(text: &str) -> String {
    ellipsize_text(text, 64)
}

fn handle_catalog_toggle(
    mut interactions: Query<&Interaction, (Changed<Interaction>, With<CatalogToggleButton>)>,
    mut catalog: ResMut<CatalogState>,
) {
    for interaction in interactions.iter_mut() {
        if *interaction == Interaction::Pressed {
            catalog.expanded = !catalog.expanded;
            catalog.bump_revision();
        }
    }
}

fn handle_catalog_mode_selector_button(
    mut interactions: Query<&Interaction, (Changed<Interaction>, With<CatalogModeSelectorButton>)>,
    mut dropdown: ResMut<CatalogModeDropdownState>,
) {
    for interaction in interactions.iter_mut() {
        if *interaction == Interaction::Pressed {
            dropdown.open = !dropdown.open;
        }
    }
}

fn handle_catalog_mode_option_button(
    mut interactions: Query<(&Interaction, &CatalogModeOptionButton), Changed<Interaction>>,
    mut dropdown: ResMut<CatalogModeDropdownState>,
    mut pipeline_dropdown: ResMut<PipelineDropdownState>,
    mut settings: ResMut<SettingsModalState>,
    mut selection: ResMut<CatalogSelectionState>,
    mut drag: ResMut<DragState>,
    mut catalog: ResMut<CatalogState>,
) {
    for (interaction, button) in interactions.iter_mut() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        catalog.set_active_mode(button.mode);
        selection.selected = None;
        selection.last_pressed = None;
        drag.active = None;
        drag.ghost_entry = None;
        dropdown.open = false;
        pipeline_dropdown.open = false;
        settings.open = false;
    }
}

fn sync_catalog_mode_dropdown(
    mut commands: Commands,
    catalog: Res<CatalogState>,
    mut dropdown: ResMut<CatalogModeDropdownState>,
    mut ui: ResMut<CatalogUiState>,
    hosts: Query<Entity, With<CatalogModeDropdownHost>>,
    children: Query<&Children>,
) {
    ui.catalog_mode_menu_open = dropdown.open;
    match (dropdown.open, dropdown.entity) {
        (true, None) => {
            let Ok(host) = hosts.single() else {
                dropdown.open = false;
                ui.catalog_mode_menu_open = false;
                return;
            };
            dropdown.entity = Some(spawn_catalog_mode_dropdown(
                &mut commands,
                host,
                catalog.active_mode(),
            ));
        }
        (false, Some(entity)) => {
            despawn_children_recursive(entity, &mut commands, &children);
            commands.entity(entity).despawn();
            dropdown.entity = None;
        }
        _ => {}
    }
}

fn update_catalog_mode_value_label(
    catalog: Res<CatalogState>,
    mut labels: Query<&mut Text, With<CatalogModeValueLabel>>,
) {
    let next = catalog.active_mode().label().to_string();
    for mut label in labels.iter_mut() {
        if label.0 != next {
            label.0 = next.clone();
        }
    }
}

fn spawn_catalog_mode_dropdown(
    commands: &mut Commands,
    host: Entity,
    active: CatalogMode,
) -> Entity {
    let mut menu_entity = Entity::PLACEHOLDER;
    commands.entity(host).with_children(|host| {
        menu_entity = host
            .spawn((
                CatalogModeDropdownRoot,
                Node {
                    position_type: PositionType::Absolute,
                    top: Val::Px(32.0),
                    left: Val::Px(0.0),
                    width: Val::Px(104.0),
                    padding: UiRect::all(Val::Px(4.0)),
                    border: UiRect::all(Val::Px(1.0)),
                    flex_direction: FlexDirection::Column,
                    row_gap: Val::Px(3.0),
                    ..default()
                },
                ZIndex(100),
                GlobalZIndex(20_000),
                BorderColor::all(PANEL_BORDER),
                BackgroundColor(PANEL_BG),
            ))
            .with_children(|menu| {
                for mode in [CatalogMode::Object, CatalogMode::Scene] {
                    menu.spawn((
                        Button,
                        CatalogModeOptionButton { mode },
                        ControlButton(ControlButtonKind::Secondary),
                        Node {
                            width: Val::Percent(100.0),
                            height: Val::Px(26.0),
                            padding: UiRect::axes(Val::Px(8.0), Val::Px(4.0)),
                            border: UiRect::all(Val::Px(1.0)),
                            align_items: AlignItems::Center,
                            ..default()
                        },
                        BorderColor::all(if mode == active {
                            BUTTON_ACTIVE_BORDER
                        } else {
                            BUTTON_BORDER
                        }),
                        BackgroundColor(if mode == active {
                            BUTTON_ACTIVE_BG
                        } else {
                            BUTTON_BG
                        }),
                    ))
                    .with_children(|button| {
                        button.spawn((
                            Text::new(mode.label()),
                            TextFont::from_font_size(12.0),
                            TextColor(BUTTON_TEXT),
                            ButtonLabel,
                        ));
                    });
                }
            })
            .id();
    });
    menu_entity
}

fn rebuild_catalog_list(
    mut commands: Commands,
    catalog: Res<CatalogState>,
    mut ui: ResMut<CatalogUiState>,
    children: Query<&Children>,
    mut toggle_query: Query<&mut Text, (With<ToggleLabel>, Without<PageLabel>)>,
    mut page_labels: Query<&mut Text, (With<PageLabel>, Without<ToggleLabel>)>,
) {
    if catalog.revision == ui.last_revision && catalog.expanded == ui.last_expanded {
        return;
    }

    ui.last_revision = catalog.revision;
    ui.last_expanded = catalog.expanded;

    for mut label in toggle_query.iter_mut() {
        label.0 = if catalog.expanded {
            "hide".to_string()
        } else {
            "show".to_string()
        };
    }
    let page_count = catalog.page_count();
    let page_index = catalog.page().saturating_add(1);
    for mut label in page_labels.iter_mut() {
        label.0 = format!("{}/{}", page_index, page_count);
    }

    despawn_children_recursive(ui.list_entity, &mut commands, &children);
    if !catalog.expanded {
        return;
    }

    let indices = catalog.visible_indices();
    commands.entity(ui.list_entity).with_children(|parent| {
        if indices.is_empty() {
            let (empty_title, empty_hint) = match catalog.active_mode() {
                CatalogMode::Object => (
                    "No object catalog items yet",
                    "Drop an image, or click open image to queue one.",
                ),
                CatalogMode::Scene => (
                    "No saved scenes yet",
                    "Open an image in scene mode to run the explicit scene pipeline.",
                ),
            };
            parent
                .spawn((
                    Node {
                        width: Val::Percent(100.0),
                        padding: UiRect::all(Val::Px(12.0)),
                        border: UiRect::all(Val::Px(1.0)),
                        flex_direction: FlexDirection::Column,
                        row_gap: Val::Px(4.0),
                        ..default()
                    },
                    BackgroundColor(Color::srgb(0.08, 0.09, 0.12)),
                    BorderColor::all(Color::srgb(0.2, 0.22, 0.28)),
                ))
                .with_children(|empty| {
                    empty.spawn((
                        Text::new(empty_title),
                        TextFont::from_font_size(13.0),
                        TextColor(Color::srgb(0.88, 0.9, 0.95)),
                    ));
                    empty.spawn((
                        Text::new(empty_hint),
                        TextFont::from_font_size(12.0),
                        TextColor(Color::srgb(0.66, 0.7, 0.78)),
                    ));
                });
            return;
        }

        for &index in indices.iter() {
            let entry = &catalog.entries[index];
            let (status_label, status_color) = match &entry.status {
                CatalogStatus::Pending => ("pending".to_string(), Color::srgb(0.9, 0.7, 0.2)),
                CatalogStatus::Ready if entry.kind == CatalogEntryKind::Scene => {
                    let mut label = scene_entry_status_text(entry);
                    label = ellipsize_text(&label, CATALOG_STATUS_MAX_CHARS);
                    (label, Color::srgb(0.4, 0.85, 0.55))
                }
                CatalogStatus::Ready => ("ready".to_string(), Color::srgb(0.4, 0.85, 0.55)),
                CatalogStatus::Failed(err) => {
                    let mut label = if err.is_empty() {
                        "failed".to_string()
                    } else {
                        format!("failed: {err}")
                    };
                    label = ellipsize_text(&label, CATALOG_STATUS_MAX_CHARS);
                    (label, Color::srgb(0.9, 0.3, 0.3))
                }
            };
            let display_label = ellipsize_text(&entry.label, CATALOG_LABEL_MAX_CHARS);
            parent
                .spawn((
                    Button,
                    CatalogEntryButton { id: entry.id },
                    Node {
                        width: Val::Percent(100.0),
                        height: Val::Px(THUMB_SIZE + 16.0),
                        padding: UiRect::all(Val::Px(8.0)),
                        border: UiRect::all(Val::Px(1.0)),
                        column_gap: Val::Px(10.0),
                        align_items: AlignItems::Center,
                        ..default()
                    },
                    BackgroundColor(ENTRY_BG),
                    BorderColor::all(ENTRY_BORDER),
                ))
                .with_children(|row| {
                    if let Some(preview) = entry.preview.as_ref() {
                        row.spawn((
                            Node {
                                width: Val::Px(THUMB_SIZE),
                                height: Val::Px(THUMB_SIZE),
                                ..default()
                            },
                            ImageNode::new(preview.image.clone()),
                        ));
                    } else {
                        row.spawn((
                            Node {
                                width: Val::Px(THUMB_SIZE),
                                height: Val::Px(THUMB_SIZE),
                                border: UiRect::all(Val::Px(1.0)),
                                ..default()
                            },
                            BackgroundColor(Color::srgb(0.08, 0.09, 0.12)),
                            BorderColor::all(Color::srgb(0.22, 0.24, 0.3)),
                        ))
                        .with_children(|pending| {
                            pending.spawn((
                                Text::new("pending"),
                                TextFont::from_font_size(12.0),
                                TextColor(Color::srgb(0.7, 0.73, 0.8)),
                            ));
                        });
                    }

                    row.spawn(Node {
                        width: Val::Px(PANEL_WIDTH - THUMB_SIZE - 96.0),
                        min_width: Val::Px(0.0),
                        flex_direction: FlexDirection::Column,
                        justify_content: JustifyContent::Center,
                        padding: UiRect::left(Val::Px(4.0)),
                        row_gap: Val::Px(4.0),
                        overflow: Overflow::clip(),
                        ..default()
                    })
                    .with_children(|text_col| {
                        text_col.spawn((
                            Text::new(display_label.clone()),
                            TextFont::from_font_size(13.0),
                            TextColor(Color::srgb(0.9, 0.92, 0.97)),
                        ));
                        text_col.spawn((
                            Text::new(status_label.clone()),
                            TextFont::from_font_size(12.0),
                            TextColor(status_color),
                        ));
                    });

                    if entry.kind == CatalogEntryKind::Scene && !entry.is_unsaved_scene() {
                        row.spawn((
                            Button,
                            CatalogSceneLoadButton { id: entry.id },
                            ControlButton(ControlButtonKind::Secondary),
                            Node {
                                width: Val::Px(46.0),
                                height: Val::Px(26.0),
                                margin: UiRect::left(Val::Auto),
                                justify_content: JustifyContent::Center,
                                align_items: AlignItems::Center,
                                border: UiRect::all(Val::Px(1.0)),
                                ..default()
                            },
                            BorderColor::all(BUTTON_BORDER),
                            BackgroundColor(BUTTON_BG),
                        ))
                        .with_children(|button| {
                            button.spawn((
                                Text::new("load"),
                                TextFont::from_font_size(12.0),
                                TextColor(BUTTON_TEXT),
                                ButtonLabel,
                            ));
                        });
                    } else {
                        row.spawn((
                            Node {
                                width: Val::Px(10.0),
                                height: Val::Px(10.0),
                                margin: UiRect::left(Val::Auto),
                                ..default()
                            },
                            BackgroundColor(status_color),
                        ));
                    }
                });
        }
    });
}

fn scene_entry_status_text(entry: &CatalogEntry) -> String {
    if entry.is_unsaved_scene() {
        return "current world".to_string();
    }
    let Some(metrics) = entry.scene_metrics.as_ref() else {
        return entry
            .scene_pipeline
            .as_deref()
            .unwrap_or("scene")
            .to_string();
    };
    let mut parts = Vec::new();
    if let Some(count) = metrics.object_count.or(metrics.placement_count) {
        parts.push(format!("{count} objects"));
    }
    if let Some(elapsed_ms) = metrics.elapsed_ms {
        parts.push(format!("{:.1}s", elapsed_ms as f32 / 1000.0));
    }
    if metrics.ok == Some(false) {
        parts.push("needs review".to_string());
    }
    if parts.is_empty() {
        entry
            .scene_pipeline
            .as_deref()
            .unwrap_or("scene")
            .to_string()
    } else {
        parts.join(" | ")
    }
}

fn despawn_children_recursive(
    entity: Entity,
    commands: &mut Commands,
    children_query: &Query<&Children>,
) {
    let Ok(children) = children_query.get(entity) else {
        return;
    };
    for child in children.iter() {
        despawn_children_recursive(child, commands, children_query);
        commands.entity(child).despawn();
    }
}

fn handle_catalog_entry_interaction(
    mut interactions: Query<(&Interaction, &CatalogEntryButton), Changed<Interaction>>,
    time: Res<Time>,
    catalog: Res<CatalogState>,
    mut drag: ResMut<DragState>,
    mut selection: ResMut<CatalogSelectionState>,
    mut source_modal: ResMut<CatalogSourceImageModalState>,
) {
    for (interaction, entry) in interactions.iter_mut() {
        if *interaction == Interaction::Pressed {
            let now = time.elapsed_secs_f64();
            let double_click = selection.last_pressed.is_some_and(|(id, last)| {
                id == entry.id && now - last <= CATALOG_DOUBLE_CLICK_SECONDS
            });
            selection.last_pressed = Some((entry.id, now));
            if double_click {
                drag.active = None;
                drag.ghost_entry = None;
                source_modal.entry_id = Some(entry.id);
                source_modal.tab = CatalogSourceImageTab::Image;
                return;
            } else {
                selection.selected = Some(entry.id);
                let is_object = catalog
                    .entry(entry.id)
                    .is_some_and(|entry| entry.kind == CatalogEntryKind::Object);
                drag.active = is_object.then_some(entry.id);
                drag.ghost_entry = None;
            }
        }
    }
}

fn handle_catalog_scene_load_button(
    mut interactions: Query<(&Interaction, &CatalogSceneLoadButton), Changed<Interaction>>,
    catalog: Res<CatalogState>,
    mut requests: MessageWriter<SceneLoadRequest>,
) {
    for (interaction, button) in interactions.iter_mut() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        let Some(entry) = catalog.entry(button.id) else {
            continue;
        };
        let Some(scene_key) = entry.scene_key.clone() else {
            continue;
        };
        requests.write(SceneLoadRequest { scene_key });
    }
}

fn handle_source_image_modal_close_button(
    mut interactions: Query<
        &Interaction,
        (Changed<Interaction>, With<CatalogSourceImageCloseButton>),
    >,
    mut modal: ResMut<CatalogSourceImageModalState>,
) {
    for interaction in interactions.iter_mut() {
        if *interaction == Interaction::Pressed {
            modal.entry_id = None;
        }
    }
}

fn handle_source_image_modal_tab_button(
    mut modal: ResMut<CatalogSourceImageModalState>,
    mut interactions: Query<(&Interaction, &CatalogSourceImageTabButton), Changed<Interaction>>,
) {
    for (interaction, button) in interactions.iter_mut() {
        if *interaction == Interaction::Pressed {
            modal.tab = button.tab;
        }
    }
}

fn handle_source_image_modal_escape(
    keys: Res<ButtonInput<KeyCode>>,
    mut modal: ResMut<CatalogSourceImageModalState>,
) {
    if modal.entry_id.is_some() && keys.just_pressed(KeyCode::Escape) {
        modal.entry_id = None;
    }
}

fn sync_source_image_modal(
    mut commands: Commands,
    catalog: Res<CatalogState>,
    mut modal: ResMut<CatalogSourceImageModalState>,
    mut ui: ResMut<CatalogUiState>,
    children: Query<&Children>,
) {
    if modal.entry_id.and_then(|id| catalog.entry(id)).is_none() {
        modal.entry_id = None;
    }
    ui.source_modal_open = modal.entry_id.is_some();

    if modal.entity.is_some() && modal.rendered_entry_id != modal.entry_id {
        if let Some(entity) = modal.entity.take() {
            despawn_children_recursive(entity, &mut commands, &children);
            commands.entity(entity).despawn();
        }
        modal.rendered_entry_id = None;
    }

    match (modal.entry_id, modal.entity) {
        (Some(id), None) => {
            if let Some(entry) = catalog.entry(id) {
                modal.entity = Some(spawn_source_image_modal(&mut commands, entry));
                modal.rendered_entry_id = Some(id);
            }
        }
        (None, Some(entity)) => {
            despawn_children_recursive(entity, &mut commands, &children);
            commands.entity(entity).despawn();
            modal.entity = None;
            modal.rendered_entry_id = None;
            modal.tab = CatalogSourceImageTab::Image;
        }
        (Some(_), Some(_)) => {}
        _ => {}
    }
}

fn sync_source_image_modal_tab_visuals(
    modal: Res<CatalogSourceImageModalState>,
    mut panels: Query<(&CatalogSourceImageTabPanel, &mut Node, &mut Visibility)>,
    mut tabs: Query<
        (
            &CatalogSourceImageTabButton,
            &Interaction,
            &Children,
            &mut BackgroundColor,
            &mut BorderColor,
        ),
        With<Button>,
    >,
    mut labels: Query<&mut TextColor, With<ButtonLabel>>,
) {
    for (panel, mut node, mut visibility) in panels.iter_mut() {
        let (next_visibility, next_display) = if panel.tab == modal.tab {
            (Visibility::Visible, Display::Flex)
        } else {
            (Visibility::Hidden, Display::None)
        };
        if *visibility != next_visibility {
            *visibility = next_visibility;
        }
        if node.display != next_display {
            node.display = next_display;
        }
    }

    for (button, interaction, children, mut background, mut border) in tabs.iter_mut() {
        let active = button.tab == modal.tab;
        let (bg, br) = if active {
            (BUTTON_ACTIVE_BG, BUTTON_ACTIVE_BORDER)
        } else {
            match *interaction {
                Interaction::Pressed => (BUTTON_BG_PRESSED, BUTTON_BORDER_PRESSED),
                Interaction::Hovered => (BUTTON_BG_HOVER, BUTTON_BORDER_HOVER),
                Interaction::None => (BUTTON_BG, BUTTON_BORDER),
            }
        };
        *background = BackgroundColor(bg);
        *border = BorderColor::all(br);
        for child in children.iter() {
            if let Ok(mut label) = labels.get_mut(child) {
                label.0 = if active { Color::WHITE } else { BUTTON_TEXT };
            }
        }
    }
}

fn delete_catalog_entry(
    id: u32,
    catalog: &mut CatalogState,
    selection: &mut CatalogSelectionState,
    drag: &mut DragState,
    commands: &mut Commands,
    delete_requests: &mut MessageWriter<CatalogDeleteRequest>,
    scene_delete_requests: &mut MessageWriter<SceneDeleteRequest>,
) {
    let Some(entry) = catalog.remove_entry(id) else {
        if selection.selected == Some(id) {
            selection.selected = None;
        }
        if drag.active == Some(id) {
            drag.active = None;
        }
        if drag.ghost_entry == Some(id) {
            clear_drag_ghost(drag, commands);
        }
        return;
    };
    if entry.is_unsaved_scene() {
        catalog.entries.push(entry);
        catalog.clamp_page();
        catalog.bump_revision();
        return;
    }

    if let Some(preview) = entry.preview {
        for entity in preview.asset_entities {
            commands.entity(entity).despawn();
        }
        commands.entity(preview.camera_entity).despawn();
        for light in preview.light_entities {
            commands.entity(light).despawn();
        }
        catalog.release_preview_layer(preview.layer_index);
    }
    if selection.selected == Some(id) {
        selection.selected = None;
    }
    if drag.active == Some(id) {
        drag.active = None;
    }
    if drag.ghost_entry == Some(id) {
        clear_drag_ghost(drag, commands);
    }

    match entry.kind {
        CatalogEntryKind::Object => {
            delete_requests.write(CatalogDeleteRequest {
                cache_key: entry.cache_key,
            });
        }
        CatalogEntryKind::Scene => {
            if let Some(scene_key) = entry.scene_key {
                scene_delete_requests.write(SceneDeleteRequest { scene_key });
            }
        }
    }
}

fn handle_catalog_delete_button(
    mut interactions: Query<&Interaction, (Changed<Interaction>, With<CatalogDeleteButton>)>,
    mut catalog: ResMut<CatalogState>,
    mut selection: ResMut<CatalogSelectionState>,
    mut drag: ResMut<DragState>,
    mut commands: Commands,
    mut delete_requests: MessageWriter<CatalogDeleteRequest>,
    mut scene_delete_requests: MessageWriter<SceneDeleteRequest>,
) {
    for interaction in interactions.iter_mut() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        let Some(id) = selection.selected else {
            continue;
        };
        delete_catalog_entry(
            id,
            &mut catalog,
            &mut selection,
            &mut drag,
            &mut commands,
            &mut delete_requests,
            &mut scene_delete_requests,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn handle_catalog_delete_shortcut(
    keys: Res<ButtonInput<KeyCode>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    ui_state: Res<CatalogUiState>,
    mut catalog: ResMut<CatalogState>,
    mut selection: ResMut<CatalogSelectionState>,
    mut drag: ResMut<DragState>,
    mut commands: Commands,
    mut delete_requests: MessageWriter<CatalogDeleteRequest>,
    mut scene_delete_requests: MessageWriter<SceneDeleteRequest>,
) {
    if !keys.just_pressed(KeyCode::Delete) && !keys.just_pressed(KeyCode::Backspace) {
        return;
    }
    let Ok(window) = windows.single() else {
        return;
    };
    if !ui_state.cursor_over_ui(window) {
        return;
    }
    let Some(id) = selection.selected else {
        return;
    };
    delete_catalog_entry(
        id,
        &mut catalog,
        &mut selection,
        &mut drag,
        &mut commands,
        &mut delete_requests,
        &mut scene_delete_requests,
    );
}

fn handle_page_buttons(
    mut prev: Query<&Interaction, (Changed<Interaction>, With<CatalogPrevButton>)>,
    mut next: Query<&Interaction, (Changed<Interaction>, With<CatalogNextButton>)>,
    mut catalog: ResMut<CatalogState>,
) {
    let mut changed = false;
    for interaction in prev.iter_mut() {
        if *interaction == Interaction::Pressed {
            let page = catalog.page();
            if page > 0 {
                catalog.set_page(page - 1);
                changed = true;
            }
        }
    }
    for interaction in next.iter_mut() {
        if *interaction == Interaction::Pressed {
            let page = catalog.page();
            if page + 1 < catalog.page_count() {
                catalog.set_page(page + 1);
                changed = true;
            }
        }
    }
    if changed {
        catalog.bump_revision();
    }
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn update_button_visuals(
    catalog: Res<CatalogState>,
    drag: Res<DragState>,
    args: Option<Res<AppArgs>>,
    available: Option<Res<AvailablePipelines>>,
    modal: Res<SettingsModalState>,
    viewer_debug: Res<ViewerDebugSettings>,
    mut selection: ResMut<CatalogSelectionState>,
    mut controls: Query<
        (
            &Interaction,
            &ControlButton,
            Option<&CatalogPrevButton>,
            Option<&CatalogNextButton>,
            Option<&CatalogDeleteButton>,
            Option<&PipelineSelectorButton>,
            Option<&PipelineOptionButton>,
            Option<&SettingsButton>,
            Option<&TripoSplatProfileButton>,
            Option<&TrellisQualityButton>,
            Option<&TrellisPbrToggleButton>,
            Option<&ViewerAabbModeButton>,
            &Children,
            &mut BackgroundColor,
            &mut BorderColor,
        ),
        With<Button>,
    >,
    mut labels: Query<&mut TextColor, With<ButtonLabel>>,
    mut entries: Query<
        (
            &Interaction,
            &CatalogEntryButton,
            &mut BackgroundColor,
            &mut BorderColor,
        ),
        (With<Button>, Without<ControlButton>),
    >,
) {
    if let Some(selected) = selection.selected
        && catalog.entry(selected).is_none()
    {
        selection.selected = None;
    }

    let args_ref = args.as_deref();
    let available_ref = available.as_deref();
    let selected_pipeline = active_pipeline_choice(&catalog, args_ref, None);
    let settings_enabled = pipeline_settings_enabled(&catalog, args_ref);

    for (
        interaction,
        button,
        prev,
        next,
        delete,
        pipeline_selector,
        pipeline_option,
        settings_button,
        profile,
        trellis_quality,
        trellis_pbr,
        viewer_aabb,
        children,
        mut bg,
        mut border,
    ) in controls.iter_mut()
    {
        let disabled = if prev.is_some() {
            catalog.page() == 0
        } else if next.is_some() {
            catalog.page() + 1 >= catalog.page_count()
        } else if delete.is_some() {
            selection.selected.is_none()
        } else if let Some(pipeline_option) = pipeline_option {
            !pipeline_available(available_ref, pipeline_option.choice)
                || !pipeline_supported(args_ref, pipeline_option.choice)
        } else if settings_button.is_some() {
            !settings_enabled
        } else {
            false
        };
        let active = pipeline_selector.is_some()
            || pipeline_option.is_some_and(|pipeline| Some(pipeline.choice) == selected_pipeline)
            || settings_button.is_some_and(|_| settings_enabled && modal.open)
            || profile
                .zip(args_ref)
                .is_some_and(|(profile, args)| profile.profile == args.triposplat_profile)
            || trellis_quality
                .zip(args_ref)
                .is_some_and(|(button, args)| button.quality == args.trellis_quality)
            || trellis_pbr
                .zip(args_ref)
                .is_some_and(|(_, args)| args.trellis_pbr_enabled)
            || viewer_aabb.is_some_and(|button| button.mode == viewer_debug.aabb_overlay);
        let (button_bg, button_border, text_color) =
            control_button_palette(button.0, *interaction, disabled, active);
        if bg.0 != button_bg {
            bg.0 = button_bg;
        }
        *border = BorderColor::all(button_border);
        for child in children.iter() {
            if let Ok(mut label) = labels.get_mut(child)
                && label.0 != text_color
            {
                label.0 = text_color;
            }
        }
    }

    for (interaction, entry, mut bg, mut border) in entries.iter_mut() {
        let dragging_this_entry = drag.active == Some(entry.id);
        let selected = selection.selected == Some(entry.id);
        let (entry_bg, entry_border) = if dragging_this_entry {
            (BUTTON_OPEN_BG_HOVER, BUTTON_OPEN_BORDER_HOVER)
        } else if selected {
            (ENTRY_BG_HOVER, ENTRY_BORDER_HOVER)
        } else {
            match *interaction {
                Interaction::Pressed => (ENTRY_BG_PRESSED, ENTRY_BORDER_PRESSED),
                Interaction::Hovered => (ENTRY_BG_HOVER, ENTRY_BORDER_HOVER),
                Interaction::None => (ENTRY_BG, ENTRY_BORDER),
            }
        };
        if bg.0 != entry_bg {
            bg.0 = entry_bg;
        }
        *border = BorderColor::all(entry_border);
    }
}

fn control_button_palette(
    kind: ControlButtonKind,
    interaction: Interaction,
    disabled: bool,
    active: bool,
) -> (Color, Color, Color) {
    if disabled {
        return (
            BUTTON_BG_DISABLED,
            BUTTON_BORDER_DISABLED,
            BUTTON_TEXT_DISABLED,
        );
    }
    if active {
        return match interaction {
            Interaction::Pressed => (
                BUTTON_OPEN_BG_PRESSED,
                BUTTON_OPEN_BORDER_PRESSED,
                BUTTON_TEXT,
            ),
            Interaction::Hovered => (BUTTON_OPEN_BG_HOVER, BUTTON_OPEN_BORDER_HOVER, BUTTON_TEXT),
            Interaction::None => (BUTTON_ACTIVE_BG, BUTTON_ACTIVE_BORDER, BUTTON_TEXT),
        };
    }

    match kind {
        ControlButtonKind::Primary => match interaction {
            Interaction::Pressed => (
                BUTTON_OPEN_BG_PRESSED,
                BUTTON_OPEN_BORDER_PRESSED,
                BUTTON_TEXT,
            ),
            Interaction::Hovered => (BUTTON_OPEN_BG_HOVER, BUTTON_OPEN_BORDER_HOVER, BUTTON_TEXT),
            Interaction::None => (BUTTON_OPEN_BG, BUTTON_OPEN_BORDER, BUTTON_TEXT),
        },
        ControlButtonKind::Secondary | ControlButtonKind::Nav => match interaction {
            Interaction::Pressed => (BUTTON_BG_PRESSED, BUTTON_BORDER_PRESSED, BUTTON_TEXT),
            Interaction::Hovered => (BUTTON_BG_HOVER, BUTTON_BORDER_HOVER, BUTTON_TEXT),
            Interaction::None => (BUTTON_BG, BUTTON_BORDER, BUTTON_TEXT),
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn handle_drag_release(
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform), With<MainCamera>>,
    catalog: Res<CatalogState>,
    mut drag: ResMut<DragState>,
    mut selection: ResMut<CatalogSelectionState>,
    ui_state: Res<CatalogUiState>,
    mut commands: Commands,
    mut spawn_requests: MessageWriter<CatalogSpawnRequest>,
) {
    if !buttons.just_released(MouseButton::Left) {
        return;
    }
    clear_drag_ghost(&mut drag, &mut commands);
    let Some(id) = drag.active.take() else {
        return;
    };
    let Ok(window) = windows.single() else {
        return;
    };
    let Ok((camera, camera_transform)) = cameras.single() else {
        return;
    };
    let Some(position) = cursor_spawn_position(window, &ui_state, camera, camera_transform) else {
        return;
    };
    let Some(entry) = catalog.entry(id) else {
        return;
    };
    let asset = if let (Some(mesh), Some(material)) = (entry.mesh.clone(), entry.material.clone()) {
        CatalogSpawnAsset::Mesh { mesh, material }
    } else if let Some(cloud) = entry.gaussian.clone() {
        CatalogSpawnAsset::GaussianSplat { cloud }
    } else {
        return;
    };
    spawn_requests.write(CatalogSpawnRequest {
        asset,
        transform: Transform::from_translation(position),
        cache_key: entry.cache_key.clone(),
        select_spawned: true,
    });
    if selection.selected == Some(id) {
        selection.selected = None;
    }
}

#[allow(clippy::too_many_arguments)]
fn update_drag_ghost(
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform), With<MainCamera>>,
    catalog: Res<CatalogState>,
    mut drag: ResMut<DragState>,
    ui_state: Res<CatalogUiState>,
    mut commands: Commands,
    mut materials: Option<ResMut<Assets<StandardMaterial>>>,
) {
    let Some(id) = drag.active else {
        clear_drag_ghost(&mut drag, &mut commands);
        return;
    };
    if !buttons.pressed(MouseButton::Left) {
        clear_drag_ghost(&mut drag, &mut commands);
        return;
    }

    let Ok(window) = windows.single() else {
        clear_drag_ghost(&mut drag, &mut commands);
        return;
    };
    let Ok((camera, camera_transform)) = cameras.single() else {
        clear_drag_ghost(&mut drag, &mut commands);
        return;
    };
    let Some(position) = cursor_spawn_position(window, &ui_state, camera, camera_transform) else {
        clear_drag_ghost(&mut drag, &mut commands);
        return;
    };

    let Some(entry) = catalog.entry(id) else {
        clear_drag_ghost(&mut drag, &mut commands);
        return;
    };
    let Some(mesh_handle) = entry.mesh.clone() else {
        clear_drag_ghost(&mut drag, &mut commands);
        return;
    };
    let Some(materials) = materials.as_mut() else {
        clear_drag_ghost(&mut drag, &mut commands);
        return;
    };

    if drag.ghost.is_none() || drag.ghost_entry != Some(id) {
        clear_drag_ghost(&mut drag, &mut commands);
        let material = materials.add(StandardMaterial {
            base_color: Color::srgba(0.78, 0.84, 0.92, DRAG_GHOST_ALPHA),
            alpha_mode: AlphaMode::Blend,
            unlit: true,
            ..default()
        });
        let ghost = commands
            .spawn((
                DragGhost,
                Pickable::IGNORE,
                Mesh3d(mesh_handle),
                MeshMaterial3d(material),
                Transform::from_translation(position),
                RenderLayers::layer(0),
            ))
            .id();
        drag.ghost = Some(ghost);
        drag.ghost_entry = Some(id);
        return;
    }

    if let Some(ghost) = drag.ghost {
        commands
            .entity(ghost)
            .insert(Transform::from_translation(position));
    }
}

fn clear_drag_ghost(drag: &mut DragState, commands: &mut Commands) {
    if let Some(entity) = drag.ghost.take() {
        commands.entity(entity).despawn();
    }
    drag.ghost_entry = None;
}

fn cleanup_drag_ghosts(
    buttons: Res<ButtonInput<MouseButton>>,
    mut drag: ResMut<DragState>,
    ghosts: Query<Entity, With<DragGhost>>,
    mut commands: Commands,
) {
    if drag.active.is_some() && buttons.pressed(MouseButton::Left) {
        return;
    }
    drag.ghost = None;
    drag.ghost_entry = None;
    for entity in ghosts.iter() {
        commands.entity(entity).despawn();
    }
}

fn cursor_spawn_position(
    window: &Window,
    ui_state: &CatalogUiState,
    camera: &Camera,
    camera_transform: &GlobalTransform,
) -> Option<Vec3> {
    if ui_state.cursor_over_ui(window) {
        return None;
    }
    let cursor = window.cursor_position()?;
    let ray = camera.viewport_to_world(camera_transform, cursor).ok()?;
    let hit = if ray.direction.y.abs() > 0.0001 {
        let t = -ray.origin.y / ray.direction.y;
        if t.is_finite() && t > 0.0 {
            Some(ray.origin + ray.direction * t)
        } else {
            None
        }
    } else {
        None
    };
    Some(hit.unwrap_or(ray.origin + ray.direction * 4.0))
}

fn sync_catalog_previews(
    mut catalog: ResMut<CatalogState>,
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    meshes: Res<Assets<BevyMesh>>,
    gaussian_clouds: Res<Assets<PlanarGaussian3d>>,
) {
    enum PreviewAction {
        Create {
            index: usize,
            asset: PreviewAsset,
            fit: PreviewFit,
        },
        Remove {
            index: usize,
        },
    }

    let visible_ids: Vec<u32> = catalog
        .visible_indices()
        .into_iter()
        .filter_map(|index| catalog.entries.get(index).map(|entry| entry.id))
        .collect();

    let mut actions = Vec::new();
    for (index, entry) in catalog.entries.iter().enumerate() {
        let should_show = visible_ids.contains(&entry.id)
            && matches!(entry.status, CatalogStatus::Ready)
            && ((entry.mesh.is_some() && entry.material.is_some())
                || entry.gaussian.is_some()
                || (entry.kind == CatalogEntryKind::Scene && !entry.scene_items.is_empty()));

        match (should_show, entry.preview.is_some()) {
            (true, false) => {
                if let (Some(mesh), Some(material)) = (entry.mesh.clone(), entry.material.clone()) {
                    let fit = meshes
                        .get(&mesh)
                        .map(preview_fit_for_mesh)
                        .unwrap_or_else(PreviewFit::fallback);
                    actions.push(PreviewAction::Create {
                        index,
                        asset: PreviewAsset::Mesh { mesh, material },
                        fit,
                    });
                } else if let Some(cloud) = entry.gaussian.clone() {
                    let fit = gaussian_clouds
                        .get(&cloud)
                        .map(preview_fit_for_gaussian_cloud)
                        .unwrap_or_else(PreviewFit::fallback);
                    actions.push(PreviewAction::Create {
                        index,
                        asset: PreviewAsset::GaussianSplat { cloud },
                        fit,
                    });
                } else if entry.kind == CatalogEntryKind::Scene && !entry.scene_items.is_empty() {
                    let fit =
                        preview_fit_for_scene_items(&entry.scene_items, &meshes, &gaussian_clouds);
                    actions.push(PreviewAction::Create {
                        index,
                        asset: PreviewAsset::Scene {
                            items: entry.scene_items.clone(),
                        },
                        fit,
                    });
                }
            }
            (false, true) => actions.push(PreviewAction::Remove { index }),
            _ => {}
        }
    }

    let mut changed = false;
    for action in actions {
        match action {
            PreviewAction::Create { index, asset, fit } => {
                if let Some(layer_index) = catalog.alloc_preview_layer() {
                    let preview =
                        spawn_preview_scene(&mut commands, &mut images, asset, layer_index, fit);
                    if let Some(entry) = catalog.entries.get_mut(index) {
                        entry.preview = Some(preview);
                    }
                    changed = true;
                }
            }
            PreviewAction::Remove { index } => {
                let preview = catalog
                    .entries
                    .get_mut(index)
                    .and_then(|entry| entry.preview.take());
                if let Some(preview) = preview {
                    for entity in preview.asset_entities {
                        commands.entity(entity).despawn();
                    }
                    commands.entity(preview.camera_entity).despawn();
                    for light in preview.light_entities {
                        commands.entity(light).despawn();
                    }
                    catalog.release_preview_layer(preview.layer_index);
                    changed = true;
                }
            }
        }
    }

    if changed {
        catalog.bump_revision();
    }
}

fn spin_thumbnails(time: Res<Time>, mut query: Query<&mut Transform, With<ThumbnailSpin>>) {
    for mut transform in query.iter_mut() {
        transform.rotate_y(time.delta_secs() * 0.8);
        transform.rotate_x(time.delta_secs() * 0.3);
    }
}

fn handle_open_button(
    mut interactions: Query<&Interaction, (Changed<Interaction>, With<OpenImageButton>)>,
    mut commands: Commands,
) {
    for interaction in interactions.iter_mut() {
        if *interaction == Interaction::Pressed {
            commands
                .dialog()
                .set_title("Open Image")
                .add_filter(
                    "Images",
                    &[
                        "png", "jpg", "jpeg", "bmp", "gif", "webp", "tga", "tif", "tiff",
                    ],
                )
                .load_multiple_files::<ImagePickDialog>();
        }
    }
}

fn handle_pipeline_selector_button(
    catalog: Res<CatalogState>,
    available: Option<Res<AvailablePipelines>>,
    mut dropdown: ResMut<PipelineDropdownState>,
    mut interactions: Query<&Interaction, (Changed<Interaction>, With<PipelineSelectorButton>)>,
) {
    let option_count = active_pipeline_choices(catalog.active_mode(), available.as_deref()).len();
    for interaction in interactions.iter_mut() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        dropdown.open = option_count > 1 && !dropdown.open;
    }
}

fn handle_pipeline_option_button(
    mut args: Option<ResMut<AppArgs>>,
    catalog: Res<CatalogState>,
    mut scene_settings: ResMut<ScenePipelineUiSettings>,
    available: Option<Res<AvailablePipelines>>,
    mut dropdown: ResMut<PipelineDropdownState>,
    mut modal: ResMut<SettingsModalState>,
    mut interactions: Query<(&Interaction, &PipelineOptionButton), Changed<Interaction>>,
) {
    let available_ref = available.as_deref();
    for (interaction, button) in interactions.iter_mut() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        if !pipeline_available(available_ref, button.choice) {
            info!(
                "synthesis pipeline {} is not enabled for this app launch",
                button.choice.label()
            );
            continue;
        }
        if matches!(catalog.active_mode(), CatalogMode::Scene)
            && let CatalogPipelineChoice::Object(model) = button.choice
        {
            if !pipeline_supported(args.as_deref(), button.choice) {
                if let Some(args) = args.as_deref() {
                    info!(
                        "scene image-to-3d model {} is unavailable for backend {:?}",
                        pipeline_label(model),
                        args.backend
                    );
                }
                continue;
            }
            scene_settings.image_to_3d_model = model;
            dropdown.open = false;
            info!(
                "selected scene image-to-3d model: {}",
                pipeline_label(model)
            );
            continue;
        }
        match button.choice {
            CatalogPipelineChoice::Object(model) => {
                let Some(args) = args.as_deref_mut() else {
                    return;
                };
                if args
                    .synthesis_models
                    .first()
                    .is_some_and(|current| *current == model)
                {
                    dropdown.open = false;
                    continue;
                }
                if !pipeline_supported(Some(&*args), button.choice) {
                    info!(
                        "synthesis pipeline {} is unavailable for backend {:?}",
                        pipeline_label(model),
                        args.backend
                    );
                    continue;
                }
                args.synthesis_models = vec![model];
                if !pipeline_has_settings(model) {
                    modal.open = false;
                }
                if matches!(model, SynthesisModel::Triposplat)
                    && args.triposplat_profile != TripoSplatProfile::Custom
                {
                    let profile = args.triposplat_profile;
                    args.apply_triposplat_profile(profile);
                }
            }
            CatalogPipelineChoice::Scene(pipeline) => {
                scene_settings.pipeline = pipeline;
            }
        }
        dropdown.open = false;
        info!("selected synthesis pipeline: {}", button.choice.label());
    }
}

#[allow(clippy::too_many_arguments)]
fn sync_pipeline_dropdown(
    mut commands: Commands,
    catalog: Res<CatalogState>,
    args: Option<Res<AppArgs>>,
    scene_settings: Res<ScenePipelineUiSettings>,
    available: Option<Res<AvailablePipelines>>,
    mut dropdown: ResMut<PipelineDropdownState>,
    mut ui: ResMut<CatalogUiState>,
    hosts: Query<Entity, With<PipelineDropdownHost>>,
    children: Query<&Children>,
) {
    ui.pipeline_menu_open = dropdown.open;
    let Some(available) = available else {
        dropdown.open = false;
        ui.pipeline_menu_open = false;
        if let Some(entity) = dropdown.entity.take() {
            despawn_children_recursive(entity, &mut commands, &children);
            commands.entity(entity).despawn();
        }
        return;
    };
    let choices = active_pipeline_choices(catalog.active_mode(), Some(&available));
    if choices.len() <= 1 {
        dropdown.open = false;
        ui.pipeline_menu_open = false;
    }

    match (dropdown.open, dropdown.entity) {
        (true, None) => {
            let Ok(host) = hosts.single() else {
                dropdown.open = false;
                return;
            };
            dropdown.entity = Some(spawn_pipeline_dropdown(
                &mut commands,
                host,
                catalog.active_mode(),
                args.as_deref(),
                &scene_settings,
                &available,
            ));
        }
        (false, Some(entity)) => {
            despawn_children_recursive(entity, &mut commands, &children);
            commands.entity(entity).despawn();
            dropdown.entity = None;
        }
        _ => {}
    }
}

fn handle_save_scene_button(
    mut interactions: Query<&Interaction, (Changed<Interaction>, With<SaveSceneButton>)>,
    mut menu: ResMut<SaveSceneMenuState>,
) {
    for interaction in interactions.iter_mut() {
        if *interaction == Interaction::Pressed {
            menu.open = !menu.open;
        }
    }
}

fn handle_save_scene_option_button(
    mut interactions: Query<(&Interaction, &SaveSceneOptionButton), Changed<Interaction>>,
    mut menu: ResMut<SaveSceneMenuState>,
    mut requests: MessageWriter<SceneSaveRequest>,
) {
    for (interaction, button) in interactions.iter_mut() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        menu.open = false;
        requests.write(SceneSaveRequest { kind: button.kind });
    }
}

fn sync_save_scene_menu(
    mut commands: Commands,
    mut menu: ResMut<SaveSceneMenuState>,
    mut ui: ResMut<CatalogUiState>,
    hosts: Query<Entity, With<SaveSceneButton>>,
    children: Query<&Children>,
) {
    ui.save_menu_open = menu.open;
    match (menu.open, menu.entity) {
        (true, None) => {
            let Ok(host) = hosts.single() else {
                menu.open = false;
                ui.save_menu_open = false;
                return;
            };
            menu.entity = Some(spawn_save_scene_menu(&mut commands, host));
        }
        (false, Some(entity)) => {
            despawn_children_recursive(entity, &mut commands, &children);
            commands.entity(entity).despawn();
            menu.entity = None;
        }
        _ => {}
    }
}

fn spawn_save_scene_menu(commands: &mut Commands, host: Entity) -> Entity {
    let mut menu_entity = Entity::PLACEHOLDER;
    commands.entity(host).with_children(|host| {
        menu_entity = host
            .spawn((
                SaveSceneMenuRoot,
                Node {
                    position_type: PositionType::Absolute,
                    top: Val::Px(32.0),
                    right: Val::Px(0.0),
                    width: Val::Px(154.0),
                    padding: UiRect::all(Val::Px(4.0)),
                    border: UiRect::all(Val::Px(1.0)),
                    flex_direction: FlexDirection::Column,
                    row_gap: Val::Px(3.0),
                    ..default()
                },
                ZIndex(100),
                GlobalZIndex(20_000),
                BorderColor::all(PANEL_BORDER),
                BackgroundColor(PANEL_BG),
            ))
            .with_children(|menu| {
                spawn_save_scene_option(menu, SceneSaveKind::Catalog, "save to catalog");
                spawn_save_scene_option(menu, SceneSaveKind::Bsn, "save BSN");
                spawn_save_scene_option(menu, SceneSaveKind::Glb, "export GLB");
            })
            .id();
    });
    menu_entity
}

fn spawn_save_scene_option(
    parent: &mut ChildSpawnerCommands<'_>,
    kind: SceneSaveKind,
    label: &str,
) {
    parent
        .spawn((
            Button,
            SaveSceneOptionButton { kind },
            ControlButton(ControlButtonKind::Secondary),
            Node {
                width: Val::Percent(100.0),
                height: Val::Px(26.0),
                padding: UiRect::axes(Val::Px(8.0), Val::Px(4.0)),
                border: UiRect::all(Val::Px(1.0)),
                align_items: AlignItems::Center,
                ..default()
            },
            BorderColor::all(BUTTON_BORDER),
            BackgroundColor(BUTTON_BG),
        ))
        .with_children(|button| {
            button.spawn((
                Text::new(label),
                TextFont::from_font_size(12.0),
                TextColor(BUTTON_TEXT),
                ButtonLabel,
            ));
        });
}

fn update_pipeline_value_label(
    catalog: Res<CatalogState>,
    args: Option<Res<AppArgs>>,
    scene_settings: Res<ScenePipelineUiSettings>,
    available: Option<Res<AvailablePipelines>>,
    mut labels: Query<&mut Text, With<PipelineValueLabel>>,
) {
    let selected = active_pipeline_choice(&catalog, args.as_deref(), Some(&scene_settings))
        .unwrap_or(CatalogPipelineChoice::Object(SynthesisModel::Triposg));
    let option_count = active_pipeline_choices(catalog.active_mode(), available.as_deref()).len();
    let next = pipeline_selector_value_text(selected, option_count);
    for mut label in labels.iter_mut() {
        if label.0 != next {
            label.0 = next.clone();
        }
    }
}

fn spawn_pipeline_dropdown(
    commands: &mut Commands,
    host: Entity,
    mode: CatalogMode,
    args: Option<&AppArgs>,
    scene_settings: &ScenePipelineUiSettings,
    available: &AvailablePipelines,
) -> Entity {
    let selected_pipeline = active_pipeline_choice_for_mode(mode, args, Some(scene_settings));
    let choices = active_pipeline_choices(mode, Some(available));
    let mut dropdown_entity = Entity::PLACEHOLDER;
    commands.entity(host).with_children(|host| {
        dropdown_entity = host
            .spawn((
                PipelineDropdownRoot,
                Node {
                    position_type: PositionType::Absolute,
                    top: Val::Px(PIPELINE_SELECTOR_HEIGHT + 4.0),
                    left: Val::Px(0.0),
                    width: Val::Px(PIPELINE_SELECTOR_WIDTH),
                    padding: UiRect::all(Val::Px(4.0)),
                    border: UiRect::all(Val::Px(1.0)),
                    flex_direction: FlexDirection::Column,
                    row_gap: Val::Px(3.0),
                    ..default()
                },
                ZIndex(100),
                GlobalZIndex(20_000),
                BorderColor::all(PANEL_BORDER),
                BackgroundColor(PANEL_BG),
            ))
            .with_children(|menu| {
                for choice in choices {
                    menu.spawn((
                        Button,
                        PipelineOptionButton { choice },
                        ControlButton(ControlButtonKind::Secondary),
                        Node {
                            width: Val::Percent(100.0),
                            height: Val::Px(26.0),
                            padding: UiRect::axes(Val::Px(8.0), Val::Px(4.0)),
                            border: UiRect::all(Val::Px(1.0)),
                            align_items: AlignItems::Center,
                            justify_content: JustifyContent::SpaceBetween,
                            ..default()
                        },
                        BorderColor::all(if Some(choice) == selected_pipeline {
                            BUTTON_ACTIVE_BORDER
                        } else {
                            BUTTON_BORDER
                        }),
                        BackgroundColor(if Some(choice) == selected_pipeline {
                            BUTTON_ACTIVE_BG
                        } else {
                            BUTTON_BG
                        }),
                    ))
                    .with_children(|button| {
                        button.spawn((
                            Text::new(choice.label()),
                            TextFont::from_font_size(12.0),
                            TextColor(BUTTON_TEXT),
                            ButtonLabel,
                        ));
                    });
                }
            })
            .id();
    });
    dropdown_entity
}

fn handle_settings_button(
    mut interactions: Query<&Interaction, (Changed<Interaction>, With<SettingsButton>)>,
    catalog: Res<CatalogState>,
    args: Option<Res<AppArgs>>,
    mut modal: ResMut<SettingsModalState>,
) {
    for interaction in interactions.iter_mut() {
        if *interaction == Interaction::Pressed {
            if pipeline_settings_enabled(&catalog, args.as_deref()) {
                modal.open = !modal.open;
            } else {
                modal.open = false;
            }
        }
    }
}

fn handle_settings_close_button(
    mut interactions: Query<&Interaction, (Changed<Interaction>, With<SettingsCloseButton>)>,
    mut modal: ResMut<SettingsModalState>,
) {
    for interaction in interactions.iter_mut() {
        if *interaction == Interaction::Pressed {
            modal.open = false;
        }
    }
}

fn handle_triposplat_profile_button(
    args: Option<ResMut<AppArgs>>,
    mut interactions: Query<(&Interaction, &TripoSplatProfileButton), Changed<Interaction>>,
) {
    let Some(mut args) = args else {
        return;
    };
    if active_settings_pipeline(Some(&*args)) != Some(SynthesisModel::Triposplat) {
        return;
    }
    for (interaction, button) in interactions.iter_mut() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        args.apply_triposplat_profile(button.profile);
        info!(
            "selected TripoSplat profile: {}",
            triposplat_profile_label(button.profile)
        );
    }
}

fn handle_triposplat_setting_step_button(
    args: Option<ResMut<AppArgs>>,
    mut interactions: Query<(&Interaction, &TripoSplatSettingStepButton), Changed<Interaction>>,
) {
    let Some(mut args) = args else {
        return;
    };
    if active_settings_pipeline(Some(&*args)) != Some(SynthesisModel::Triposplat) {
        return;
    }
    for (interaction, button) in interactions.iter_mut() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        adjust_triposplat_setting(&mut args, button.setting, button.delta);
    }
}

fn handle_triposg_setting_step_button(
    args: Option<ResMut<AppArgs>>,
    mut interactions: Query<(&Interaction, &TripoSgSettingStepButton), Changed<Interaction>>,
) {
    let Some(mut args) = args else {
        return;
    };
    if active_settings_pipeline(Some(&*args)) != Some(SynthesisModel::Triposg) {
        return;
    }
    for (interaction, button) in interactions.iter_mut() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        adjust_triposg_setting(&mut args, button.setting, button.delta);
    }
}

fn handle_trellis_quality_button(
    args: Option<ResMut<AppArgs>>,
    mut interactions: Query<(&Interaction, &TrellisQualityButton), Changed<Interaction>>,
) {
    let Some(mut args) = args else {
        return;
    };
    if active_settings_pipeline(Some(&*args)) != Some(SynthesisModel::Trellis) {
        return;
    }
    for (interaction, button) in interactions.iter_mut() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        args.trellis_quality = button.quality;
        info!(
            "Trellis.2 settings: quality={} resolution={}",
            trellis_quality_label(button.quality),
            trellis_resolution_text(button.quality)
        );
    }
}

fn handle_trellis_pbr_toggle_button(
    args: Option<ResMut<AppArgs>>,
    mut interactions: Query<&Interaction, (Changed<Interaction>, With<TrellisPbrToggleButton>)>,
) {
    let Some(mut args) = args else {
        return;
    };
    if active_settings_pipeline(Some(&*args)) != Some(SynthesisModel::Trellis) {
        return;
    }
    for interaction in interactions.iter_mut() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        args.trellis_pbr_enabled = !args.trellis_pbr_enabled;
        info!(
            "Trellis.2 settings: pbr={}",
            if args.trellis_pbr_enabled {
                "on"
            } else {
                "off"
            }
        );
    }
}

fn handle_trellis_setting_step_button(
    args: Option<ResMut<AppArgs>>,
    mut interactions: Query<(&Interaction, &TrellisSettingStepButton), Changed<Interaction>>,
) {
    let Some(mut args) = args else {
        return;
    };
    if active_settings_pipeline(Some(&*args)) != Some(SynthesisModel::Trellis) {
        return;
    }
    for (interaction, button) in interactions.iter_mut() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        adjust_trellis_setting(&mut args, button.setting, button.delta);
    }
}

fn handle_scene_quality_button(
    mut scene_settings: ResMut<ScenePipelineUiSettings>,
    mut interactions: Query<(&Interaction, &SceneQualityButton), Changed<Interaction>>,
) {
    for (interaction, button) in interactions.iter_mut() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        scene_settings.quality_profile = button.quality;
    }
}

fn handle_scene_setting_step_button(
    mut scene_settings: ResMut<ScenePipelineUiSettings>,
    mut interactions: Query<(&Interaction, &SceneSettingStepButton), Changed<Interaction>>,
) {
    for (interaction, button) in interactions.iter_mut() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        adjust_scene_setting(&mut scene_settings, button.setting, button.delta);
    }
}

fn handle_scene_setting_toggle_button(
    mut scene_settings: ResMut<ScenePipelineUiSettings>,
    mut interactions: Query<(&Interaction, &SceneSettingToggleButton), Changed<Interaction>>,
) {
    for (interaction, button) in interactions.iter_mut() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        match button.setting {
            SceneToggleSetting::Pbr => scene_settings.pbr_enabled = !scene_settings.pbr_enabled,
            SceneToggleSetting::CatalogReuse => {
                scene_settings.allow_catalog_reuse = !scene_settings.allow_catalog_reuse
            }
            SceneToggleSetting::LiftAssets => {
                scene_settings.lift_assets = !scene_settings.lift_assets
            }
            SceneToggleSetting::LocateAnything => {
                scene_settings.locate_anything_enabled = !scene_settings.locate_anything_enabled
            }
            SceneToggleSetting::Depth => {
                scene_settings.depth_enabled = !scene_settings.depth_enabled
            }
            SceneToggleSetting::Segmentation => {
                scene_settings.segmentation_enabled = !scene_settings.segmentation_enabled
            }
            SceneToggleSetting::PoseFit => {
                scene_settings.pose_fit_enabled = !scene_settings.pose_fit_enabled
            }
            SceneToggleSetting::Feedback => {
                scene_settings.feedback_enabled = !scene_settings.feedback_enabled
            }
            SceneToggleSetting::WriteArtifacts => {
                scene_settings.write_artifacts = !scene_settings.write_artifacts
            }
            SceneToggleSetting::PromoteToCatalog => {
                scene_settings.promote_to_catalog = !scene_settings.promote_to_catalog
            }
        }
    }
}

fn handle_settings_tab_button(
    mut modal: ResMut<SettingsModalState>,
    mut interactions: Query<(&Interaction, &SettingsTabButton), Changed<Interaction>>,
) {
    for (interaction, button) in interactions.iter_mut() {
        if *interaction == Interaction::Pressed && modal.tab != button.tab {
            modal.tab = button.tab;
        }
    }
}

fn handle_settings_scroll(
    mut scroll: On<Pointer<PointerScroll>>,
    mut query: Query<(&Node, &ComputedNode, &mut ScrollPosition), With<SettingsScrollArea>>,
) {
    let Ok((node, computed, mut scroll_position)) = query.get_mut(scroll.entity) else {
        return;
    };
    if node.overflow.y != OverflowAxis::Scroll || scroll.y == 0.0 {
        return;
    }

    scroll.propagate(false);
    let visible_size = computed.size() * computed.inverse_scale_factor();
    let content_size = computed.content_size() * computed.inverse_scale_factor();
    let max_offset = (content_size - visible_size).max(Vec2::ZERO);
    let unit = match scroll.unit {
        MouseScrollUnit::Line => MouseScrollUnit::SCROLL_UNIT_CONVERSION_FACTOR,
        MouseScrollUnit::Pixel => 1.0,
    };
    scroll_position.y = (scroll_position.y - scroll.y * unit).clamp(0.0, max_offset.y);
}

fn handle_developer_panel_tab_button(
    mut state: ResMut<DeveloperPanelState>,
    mut interactions: Query<(&Interaction, &SettingsDeveloperTabButton), Changed<Interaction>>,
) {
    for (interaction, button) in interactions.iter_mut() {
        if *interaction == Interaction::Pressed && state.tab != button.tab {
            state.tab = button.tab;
        }
    }
}

fn handle_developer_visual_page_button(
    mut state: ResMut<DeveloperPanelState>,
    cache: Res<ProcessingArtifactPreviewCache>,
    mut interactions: Query<
        (&Interaction, &SettingsDeveloperVisualPageButton),
        Changed<Interaction>,
    >,
) {
    let max_page = cache.page_count.saturating_sub(1);
    for (interaction, button) in interactions.iter_mut() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        match button.direction {
            DeveloperVisualPageDirection::Previous => {
                state.visual_page = state.visual_page.saturating_sub(1);
            }
            DeveloperVisualPageDirection::Next => {
                state.visual_page = state.visual_page.saturating_add(1).min(max_page);
            }
        }
    }
}

fn handle_viewer_aabb_mode_button(
    mut settings: ResMut<ViewerDebugSettings>,
    mut interactions: Query<(&Interaction, &ViewerAabbModeButton), Changed<Interaction>>,
) {
    for (interaction, button) in interactions.iter_mut() {
        if *interaction == Interaction::Pressed {
            settings.aabb_overlay = button.mode;
        }
    }
}

fn handle_viewer_debug_toggle_button(
    mut settings: ResMut<ViewerDebugSettings>,
    mut interactions: Query<(&Interaction, &ViewerDebugToggleButton), Changed<Interaction>>,
) {
    for (interaction, button) in interactions.iter_mut() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        match button.setting {
            ViewerDebugToggleSetting::GroundContact => {
                settings.draw_ground_contact = !settings.draw_ground_contact;
            }
            ViewerDebugToggleSetting::SceneCameraFrustum => {
                settings.draw_scene_camera_frustum = !settings.draw_scene_camera_frustum;
            }
            ViewerDebugToggleSetting::DepthCloud => {
                settings.depth_cloud_overlay = !settings.depth_cloud_overlay;
            }
        }
    }
}

fn handle_viewer_debug_step_button(
    mut settings: ResMut<ViewerDebugSettings>,
    mut interactions: Query<(&Interaction, &ViewerDebugStepButton), Changed<Interaction>>,
) {
    for (interaction, button) in interactions.iter_mut() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        match button.setting {
            ViewerDebugNumericSetting::GroundY => {
                settings.ground_y = (settings.ground_y + button.delta)
                    .clamp(VIEWER_GROUND_Y_MIN, VIEWER_GROUND_Y_MAX);
            }
            ViewerDebugNumericSetting::ContactTolerance => {
                settings.contact_tolerance = (settings.contact_tolerance + button.delta)
                    .clamp(VIEWER_CONTACT_TOLERANCE_MIN, VIEWER_CONTACT_TOLERANCE_MAX);
            }
            ViewerDebugNumericSetting::SceneCameraFrustumLength => {
                settings.scene_camera_frustum_length = (settings.scene_camera_frustum_length
                    + button.delta)
                    .clamp(VIEWER_FRUSTUM_LENGTH_MIN, VIEWER_FRUSTUM_LENGTH_MAX);
            }
            ViewerDebugNumericSetting::DepthCloudMaxGaussians => {
                let next = settings.depth_cloud_max_gaussians as f32 + button.delta;
                let stepped = (next / VIEWER_DEPTH_CLOUD_GAUSSIAN_STEP as f32).round()
                    * VIEWER_DEPTH_CLOUD_GAUSSIAN_STEP as f32;
                settings.depth_cloud_max_gaussians = stepped
                    .clamp(
                        VIEWER_DEPTH_CLOUD_MIN_GAUSSIANS as f32,
                        VIEWER_DEPTH_CLOUD_MAX_GAUSSIANS as f32,
                    )
                    .round() as usize;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn sync_settings_modal(
    mut commands: Commands,
    catalog: Res<CatalogState>,
    args: Option<Res<AppArgs>>,
    scene_settings: Res<ScenePipelineUiSettings>,
    available: Option<Res<AvailablePipelines>>,
    mut modal: ResMut<SettingsModalState>,
    mut ui: ResMut<CatalogUiState>,
    children: Query<&Children>,
) {
    let active_pipeline = active_pipeline_choice(&catalog, args.as_deref(), Some(&scene_settings));
    if active_pipeline.is_none() || !pipeline_settings_enabled(&catalog, args.as_deref()) {
        modal.open = false;
    }
    if modal.entity.is_some() && modal.pipeline != active_pipeline {
        if let Some(entity) = modal.entity.take() {
            despawn_children_recursive(entity, &mut commands, &children);
            commands.entity(entity).despawn();
        }
        modal.pipeline = None;
        modal.tab = SettingsModalTab::Pipeline;
    }
    ui.settings_modal_open = modal.open;
    match (modal.open, modal.entity) {
        (true, None) => {
            if let Some(pipeline) = active_pipeline {
                modal.entity = Some(spawn_settings_modal(
                    &mut commands,
                    pipeline,
                    modal.tab,
                    available.as_deref(),
                ));
                modal.pipeline = Some(pipeline);
            }
        }
        (false, Some(entity)) => {
            despawn_children_recursive(entity, &mut commands, &children);
            commands.entity(entity).despawn();
            modal.entity = None;
            modal.pipeline = None;
        }
        _ => {}
    }
}

fn sync_settings_tab_visuals(
    modal: Res<SettingsModalState>,
    mut panels: Query<(&SettingsTabPanel, &mut Node, &mut Visibility)>,
    mut tabs: Query<
        (
            &SettingsTabButton,
            &Interaction,
            &Children,
            &mut BackgroundColor,
            &mut BorderColor,
        ),
        With<Button>,
    >,
    mut labels: Query<&mut TextColor, With<ButtonLabel>>,
) {
    for (panel, mut node, mut visibility) in panels.iter_mut() {
        let (next_visibility, next_display) = if panel.tab == modal.tab {
            (Visibility::Visible, Display::Flex)
        } else {
            (Visibility::Hidden, Display::None)
        };
        if *visibility != next_visibility {
            *visibility = next_visibility;
        }
        if node.display != next_display {
            node.display = next_display;
        }
    }
    for (tab, interaction, children, mut bg, mut border) in tabs.iter_mut() {
        let active = tab.tab == modal.tab;
        let (button_bg, button_border, text_color) =
            control_button_palette(ControlButtonKind::Secondary, *interaction, false, active);
        if bg.0 != button_bg {
            bg.0 = button_bg;
        }
        *border = BorderColor::all(button_border);
        for child in children.iter() {
            if let Ok(mut label) = labels.get_mut(child)
                && label.0 != text_color
            {
                label.0 = text_color;
            }
        }
    }
}

fn sync_developer_panel_tab_visuals(
    state: Res<DeveloperPanelState>,
    mut panels: Query<(&SettingsDeveloperTabPanel, &mut Node, &mut Visibility)>,
    mut tabs: Query<
        (
            &SettingsDeveloperTabButton,
            &Interaction,
            &Children,
            &mut BackgroundColor,
            &mut BorderColor,
        ),
        With<Button>,
    >,
    mut labels: Query<&mut TextColor, With<ButtonLabel>>,
) {
    for (panel, mut node, mut visibility) in panels.iter_mut() {
        let (next_visibility, next_display) = if panel.tab == state.tab {
            (Visibility::Visible, Display::Flex)
        } else {
            (Visibility::Hidden, Display::None)
        };
        if *visibility != next_visibility {
            *visibility = next_visibility;
        }
        if node.display != next_display {
            node.display = next_display;
        }
    }
    for (tab, interaction, children, mut bg, mut border) in tabs.iter_mut() {
        let active = tab.tab == state.tab;
        let (button_bg, button_border, text_color) =
            control_button_palette(ControlButtonKind::Secondary, *interaction, false, active);
        if bg.0 != button_bg {
            bg.0 = button_bg;
        }
        *border = BorderColor::all(button_border);
        for child in children.iter() {
            if let Ok(mut label) = labels.get_mut(child)
                && label.0 != text_color
            {
                label.0 = text_color;
            }
        }
    }
}

fn sync_developer_visual_page_controls(
    artifact_previews: Res<ProcessingArtifactPreviewCache>,
    mut buttons: Query<
        (
            &SettingsDeveloperVisualPageButton,
            &Interaction,
            &Children,
            &mut BackgroundColor,
            &mut BorderColor,
        ),
        With<Button>,
    >,
    mut labels: Query<&mut TextColor, With<ButtonLabel>>,
    mut pager_texts: Query<&mut Text, With<SettingsDeveloperVisualPagerText>>,
) {
    let page_text = if artifact_previews.total_count == 0 {
        "page 0/0 | 0 images".to_string()
    } else {
        format!(
            "page {}/{} | {} images | latest first",
            artifact_previews.page + 1,
            artifact_previews.page_count.max(1),
            artifact_previews.total_count
        )
    };
    for mut text in &mut pager_texts {
        text.0 = page_text.clone();
    }

    for (button, interaction, children, mut bg, mut border) in &mut buttons {
        let disabled = artifact_previews.total_count == 0
            || match button.direction {
                DeveloperVisualPageDirection::Previous => artifact_previews.page == 0,
                DeveloperVisualPageDirection::Next => {
                    artifact_previews.page + 1 >= artifact_previews.page_count.max(1)
                }
            };
        let (button_bg, button_border, text_color) =
            control_button_palette(ControlButtonKind::Nav, *interaction, disabled, false);
        if bg.0 != button_bg {
            bg.0 = button_bg;
        }
        *border = BorderColor::all(button_border);
        for child in children.iter() {
            if let Ok(mut label) = labels.get_mut(child)
                && label.0 != text_color
            {
                label.0 = text_color;
            }
        }
    }
}

fn sync_settings_developer_visual_grid(
    mut commands: Commands,
    children: Query<&Children>,
    artifact_previews: Res<ProcessingArtifactPreviewCache>,
    mut grids: Query<(Entity, &mut SettingsDeveloperVisualGrid)>,
) {
    for (entity, mut grid) in &mut grids {
        if grid.signature == artifact_previews.signature {
            continue;
        }
        despawn_children_recursive(entity, &mut commands, &children);
        grid.signature = artifact_previews.signature.clone();
        commands.entity(entity).with_children(|parent| {
            if artifact_previews.previews.is_empty() {
                parent.spawn((
                    Text::new("no image artifacts discovered for the active run"),
                    TextFont::from_font_size(11.0),
                    TextColor(Color::srgb(0.62, 0.66, 0.74)),
                ));
                return;
            }
            for preview in artifact_previews.previews.iter() {
                spawn_developer_visual_preview_row(parent, preview);
            }
        });
    }
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn update_settings_labels(
    args: Option<Res<AppArgs>>,
    scene_settings: Res<ScenePipelineUiSettings>,
    mut profile_labels: Query<
        &mut Text,
        (
            With<TripoSplatProfileValueLabel>,
            Without<TripoSplatSettingValueLabel>,
            Without<TripoSgSettingValueLabel>,
            Without<TrellisQualityValueLabel>,
            Without<TrellisSettingValueLabel>,
            Without<SceneImageTo3dModelValueLabel>,
        ),
    >,
    mut value_labels: Query<
        (&TripoSplatSettingValueLabel, &mut Text),
        (
            Without<TripoSplatProfileValueLabel>,
            Without<TripoSgSettingValueLabel>,
            Without<TrellisQualityValueLabel>,
            Without<TrellisSettingValueLabel>,
            Without<SceneImageTo3dModelValueLabel>,
        ),
    >,
    mut triposg_value_labels: Query<
        (&TripoSgSettingValueLabel, &mut Text),
        (
            Without<TripoSplatProfileValueLabel>,
            Without<TripoSplatSettingValueLabel>,
            Without<TrellisQualityValueLabel>,
            Without<TrellisSettingValueLabel>,
            Without<SceneImageTo3dModelValueLabel>,
        ),
    >,
    mut trellis_quality_labels: Query<
        &mut Text,
        (
            With<TrellisQualityValueLabel>,
            Without<TripoSplatProfileValueLabel>,
            Without<TripoSplatSettingValueLabel>,
            Without<TripoSgSettingValueLabel>,
            Without<TrellisSettingValueLabel>,
            Without<SceneImageTo3dModelValueLabel>,
        ),
    >,
    mut trellis_value_labels: Query<
        (&TrellisSettingValueLabel, &mut Text),
        (
            Without<TripoSplatProfileValueLabel>,
            Without<TripoSplatSettingValueLabel>,
            Without<TripoSgSettingValueLabel>,
            Without<TrellisQualityValueLabel>,
            Without<SceneImageTo3dModelValueLabel>,
        ),
    >,
    mut scene_quality_labels: Query<
        &mut Text,
        (
            With<SceneQualityValueLabel>,
            Without<TripoSplatProfileValueLabel>,
            Without<TripoSplatSettingValueLabel>,
            Without<TripoSgSettingValueLabel>,
            Without<TrellisQualityValueLabel>,
            Without<TrellisSettingValueLabel>,
            Without<SceneSettingValueLabel>,
            Without<SceneToggleValueLabel>,
            Without<SceneImageTo3dModelValueLabel>,
        ),
    >,
    mut scene_value_labels: Query<
        (&SceneSettingValueLabel, &mut Text),
        (
            Without<TripoSplatProfileValueLabel>,
            Without<TripoSplatSettingValueLabel>,
            Without<TripoSgSettingValueLabel>,
            Without<TrellisQualityValueLabel>,
            Without<TrellisSettingValueLabel>,
            Without<SceneQualityValueLabel>,
            Without<SceneToggleValueLabel>,
            Without<SceneImageTo3dModelValueLabel>,
        ),
    >,
    mut scene_toggle_labels: Query<
        (&SceneToggleValueLabel, &mut Text),
        (
            Without<TripoSplatProfileValueLabel>,
            Without<TripoSplatSettingValueLabel>,
            Without<TripoSgSettingValueLabel>,
            Without<TrellisQualityValueLabel>,
            Without<TrellisSettingValueLabel>,
            Without<SceneQualityValueLabel>,
            Without<SceneSettingValueLabel>,
            Without<SceneImageTo3dModelValueLabel>,
        ),
    >,
    mut scene_image_model_labels: Query<
        &mut Text,
        (
            With<SceneImageTo3dModelValueLabel>,
            Without<TripoSplatProfileValueLabel>,
            Without<TripoSplatSettingValueLabel>,
            Without<TripoSgSettingValueLabel>,
            Without<TrellisQualityValueLabel>,
            Without<TrellisSettingValueLabel>,
            Without<SceneQualityValueLabel>,
            Without<SceneSettingValueLabel>,
            Without<SceneToggleValueLabel>,
        ),
    >,
) {
    if let Some(args) = args {
        for mut label in profile_labels.iter_mut() {
            let next = triposplat_profile_label(args.triposplat_profile).to_string();
            if label.0 != next {
                label.0 = next;
            }
        }
        for (value, mut label) in value_labels.iter_mut() {
            let next = triposplat_setting_value_text(&args, value.setting);
            if label.0 != next {
                label.0 = next;
            }
        }
        for (value, mut label) in triposg_value_labels.iter_mut() {
            let next = triposg_setting_value_text(&args, value.setting);
            if label.0 != next {
                label.0 = next;
            }
        }
        for mut label in trellis_quality_labels.iter_mut() {
            let next = trellis_quality_value_text(args.trellis_quality);
            if label.0 != next {
                label.0 = next;
            }
        }
        for (value, mut label) in trellis_value_labels.iter_mut() {
            let next = trellis_setting_value_text(&args, value.setting);
            if label.0 != next {
                label.0 = next;
            }
        }
    }
    for mut label in scene_quality_labels.iter_mut() {
        let next = scene_settings.quality_profile.label().to_string();
        if label.0 != next {
            label.0 = next;
        }
    }
    for (value, mut label) in scene_value_labels.iter_mut() {
        let next = scene_setting_value_text(&scene_settings, value.setting);
        if label.0 != next {
            label.0 = next;
        }
    }
    for (value, mut label) in scene_toggle_labels.iter_mut() {
        let next = scene_toggle_value_text(&scene_settings, value.setting);
        if label.0 != next {
            label.0 = next;
        }
    }
    for mut label in scene_image_model_labels.iter_mut() {
        let next = pipeline_label(scene_settings.image_to_3d_model).to_string();
        if label.0 != next {
            label.0 = next;
        }
    }
}

#[allow(clippy::type_complexity)]
fn update_viewer_debug_labels(
    viewer_debug: Res<ViewerDebugSettings>,
    mut aabb_labels: Query<
        &mut Text,
        (
            With<ViewerAabbModeValueLabel>,
            Without<ViewerDebugToggleValueLabel>,
            Without<ViewerDebugNumericValueLabel>,
        ),
    >,
    mut toggle_labels: Query<
        (&ViewerDebugToggleValueLabel, &mut Text),
        (
            Without<ViewerAabbModeValueLabel>,
            Without<ViewerDebugNumericValueLabel>,
        ),
    >,
    mut numeric_labels: Query<
        (&ViewerDebugNumericValueLabel, &mut Text),
        (
            Without<ViewerAabbModeValueLabel>,
            Without<ViewerDebugToggleValueLabel>,
        ),
    >,
) {
    for mut label in aabb_labels.iter_mut() {
        let next = viewer_debug.aabb_overlay.label().to_string();
        if label.0 != next {
            label.0 = next;
        }
    }
    for (value, mut label) in toggle_labels.iter_mut() {
        let next = match value.setting {
            ViewerDebugToggleSetting::GroundContact => {
                if viewer_debug.draw_ground_contact {
                    "on"
                } else {
                    "off"
                }
            }
            ViewerDebugToggleSetting::SceneCameraFrustum => {
                if viewer_debug.draw_scene_camera_frustum {
                    "on"
                } else {
                    "off"
                }
            }
            ViewerDebugToggleSetting::DepthCloud => {
                if viewer_debug.depth_cloud_overlay {
                    "on"
                } else {
                    "off"
                }
            }
        }
        .to_string();
        if label.0 != next {
            label.0 = next;
        }
    }
    for (value, mut label) in numeric_labels.iter_mut() {
        let next = match value.setting {
            ViewerDebugNumericSetting::GroundY => format!("{:.2}", viewer_debug.ground_y),
            ViewerDebugNumericSetting::ContactTolerance => {
                format!("{:.2}", viewer_debug.contact_tolerance)
            }
            ViewerDebugNumericSetting::SceneCameraFrustumLength => {
                format!("{:.2}", viewer_debug.scene_camera_frustum_length)
            }
            ViewerDebugNumericSetting::DepthCloudMaxGaussians => {
                format!("{}", viewer_debug.depth_cloud_max_gaussians)
            }
        };
        if label.0 != next {
            label.0 = next;
        }
    }
}

fn spawn_source_image_modal(commands: &mut Commands, entry: &CatalogEntry) -> Entity {
    let title = ellipsize_text(&entry.label, 56);
    let source_text = entry
        .source_image_path
        .as_deref()
        .map(|path| ellipsize_text(path, 72))
        .unwrap_or_else(|| "source image unknown".to_string());
    commands
        .spawn((
            CatalogSourceImageModalRoot,
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(0.0),
                left: Val::Px(0.0),
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(MODAL_SCRIM),
            GlobalZIndex(30_000),
        ))
        .with_children(|root| {
            root.spawn((
                Node {
                    width: Val::Px(560.0),
                    flex_direction: FlexDirection::Column,
                    row_gap: Val::Px(12.0),
                    padding: UiRect::all(Val::Px(16.0)),
                    border: UiRect::all(Val::Px(1.0)),
                    ..default()
                },
                BackgroundColor(MODAL_BG),
                BorderColor::all(MODAL_BORDER),
            ))
            .with_children(|panel| {
                panel
                    .spawn(Node {
                        width: Val::Percent(100.0),
                        justify_content: JustifyContent::SpaceBetween,
                        align_items: AlignItems::Center,
                        column_gap: Val::Px(12.0),
                        ..default()
                    })
                    .with_children(|header| {
                        header.spawn((
                            Text::new(title.clone()),
                            TextFont::from_font_size(16.0),
                            TextColor(Color::srgb(0.92, 0.94, 0.98)),
                        ));
                        header
                            .spawn((
                                Button,
                                CatalogSourceImageCloseButton,
                                ControlButton(ControlButtonKind::Secondary),
                                Node {
                                    padding: UiRect::axes(Val::Px(9.0), Val::Px(4.0)),
                                    border: UiRect::all(Val::Px(1.0)),
                                    ..default()
                                },
                                BorderColor::all(BUTTON_BORDER),
                                BackgroundColor(BUTTON_BG),
                            ))
                            .with_children(|button| {
                                button.spawn((
                                    Text::new("close"),
                                    TextFont::from_font_size(12.0),
                                    TextColor(BUTTON_TEXT),
                                    ButtonLabel,
                                ));
                            });
                    });

                panel.spawn((
                    Text::new(source_text),
                    TextFont::from_font_size(12.0),
                    TextColor(Color::srgb(0.7, 0.74, 0.82)),
                ));

                if entry.kind == CatalogEntryKind::Scene {
                    spawn_source_image_tabs(panel);
                    spawn_source_image_tab_panel(
                        panel,
                        CatalogSourceImageTab::Image,
                        true,
                        |panel| spawn_source_image_body(panel, entry),
                    );
                    spawn_source_image_tab_panel(
                        panel,
                        CatalogSourceImageTab::Stats,
                        false,
                        |panel| spawn_scene_details_stats(panel, entry),
                    );
                } else {
                    spawn_source_image_body(panel, entry);
                }
            });
        })
        .id()
}

fn spawn_source_image_tabs(panel: &mut ChildSpawnerCommands) {
    panel
        .spawn(Node {
            flex_direction: FlexDirection::Row,
            column_gap: Val::Px(6.0),
            ..default()
        })
        .with_children(|row| {
            for tab in [CatalogSourceImageTab::Image, CatalogSourceImageTab::Stats] {
                row.spawn((
                    Button,
                    CatalogSourceImageTabButton { tab },
                    Node {
                        height: Val::Px(28.0),
                        padding: UiRect::axes(Val::Px(10.0), Val::Px(4.0)),
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        border: UiRect::all(Val::Px(1.0)),
                        ..default()
                    },
                    BorderColor::all(BUTTON_BORDER),
                    BackgroundColor(BUTTON_BG),
                ))
                .with_children(|button| {
                    button.spawn((
                        Text::new(tab.label()),
                        TextFont::from_font_size(12.0),
                        TextColor(BUTTON_TEXT),
                        ButtonLabel,
                    ));
                });
            }
        });
}

fn spawn_source_image_tab_panel(
    parent: &mut ChildSpawnerCommands,
    tab: CatalogSourceImageTab,
    visible: bool,
    spawn_content: impl FnOnce(&mut ChildSpawnerCommands),
) {
    parent
        .spawn((
            CatalogSourceImageTabPanel { tab },
            Node {
                display: if visible {
                    Display::Flex
                } else {
                    Display::None
                },
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(10.0),
                ..default()
            },
            if visible {
                Visibility::Visible
            } else {
                Visibility::Hidden
            },
        ))
        .with_children(spawn_content);
}

fn spawn_source_image_body(parent: &mut ChildSpawnerCommands, entry: &CatalogEntry) {
    if let Some(image) = entry.source_image.as_ref() {
        parent.spawn((
            Node {
                width: Val::Px(512.0),
                height: Val::Px(512.0),
                border: UiRect::all(Val::Px(1.0)),
                align_self: AlignSelf::Center,
                ..default()
            },
            BorderColor::all(Color::srgb(0.24, 0.27, 0.34)),
            ImageNode::new(image.clone()),
        ));
    } else {
        parent
            .spawn((
                Node {
                    width: Val::Px(512.0),
                    height: Val::Px(220.0),
                    border: UiRect::all(Val::Px(1.0)),
                    align_self: AlignSelf::Center,
                    justify_content: JustifyContent::Center,
                    align_items: AlignItems::Center,
                    ..default()
                },
                BorderColor::all(Color::srgb(0.24, 0.27, 0.34)),
                BackgroundColor(Color::srgb(0.05, 0.06, 0.08)),
            ))
            .with_children(|missing| {
                missing.spawn((
                    Text::new("source image unavailable"),
                    TextFont::from_font_size(13.0),
                    TextColor(Color::srgb(0.8, 0.84, 0.9)),
                ));
            });
    }
}

fn spawn_scene_details_stats(parent: &mut ChildSpawnerCommands, entry: &CatalogEntry) {
    parent
        .spawn(Node {
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(12.0),
            ..default()
        })
        .with_children(|stats| {
            let pipeline = entry.scene_pipeline.as_deref().unwrap_or("scene");
            spawn_scene_stats_section(stats, "summary", |section| {
                spawn_scene_stat_row(section, "pipeline", pipeline.to_string());
                if let Some(scene_key) = entry.scene_key.as_deref() {
                    spawn_scene_stat_row(section, "scene key", ellipsize_text(scene_key, 52));
                }
                if let Some(metrics) = entry.scene_metrics.as_ref() {
                    spawn_scene_stat_row(section, "status", scene_metric_status(metrics));
                    if let Some(elapsed_ms) = metrics.elapsed_ms {
                        spawn_scene_stat_row(
                            section,
                            "runtime",
                            format!("{:.1}s", elapsed_ms as f32 / 1000.0),
                        );
                    }
                    spawn_scene_stat_row(section, "counts", scene_metric_counts_text(metrics));
                } else {
                    spawn_scene_stat_row(section, "status", "no cached metrics".to_string());
                }
            });

            if let Some(metrics) = entry.scene_metrics.as_ref() {
                spawn_scene_stats_section(stats, "categories", |section| {
                    if metrics.category_breakdown.is_empty() {
                        spawn_scene_stat_row(section, "breakdown", "unavailable".to_string());
                    } else {
                        for category in metrics.category_breakdown.iter().take(10) {
                            spawn_scene_category_row(section, category);
                        }
                    }
                });

                spawn_scene_stats_section(stats, "quality", |section| {
                    spawn_scene_stat_row(section, "feedback", scene_feedback_text(metrics));
                    if let Some(stage) = metrics.failed_stage.as_deref() {
                        spawn_scene_stat_row(section, "failed stage", stage.to_string());
                    }
                });
            }

            spawn_scene_stats_section(stats, "artifacts", |section| {
                if let Some(path) = entry.source_image_path.as_deref() {
                    spawn_scene_stat_row(section, "source", ellipsize_text(path, 58));
                }
                if let Some(dir) = entry.scene_artifact_dir.as_deref() {
                    spawn_scene_stat_row(section, "run dir", ellipsize_text(dir, 58));
                }
            });
        });
}

fn spawn_scene_stats_section(
    parent: &mut ChildSpawnerCommands,
    title: &'static str,
    spawn_content: impl FnOnce(&mut ChildSpawnerCommands),
) {
    parent
        .spawn(Node {
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(5.0),
            ..default()
        })
        .with_children(|section| {
            section.spawn((
                Text::new(title),
                TextFont::from_font_size(11.0),
                TextColor(Color::srgb(0.58, 0.64, 0.74)),
            ));
            spawn_content(section);
        });
}

fn spawn_scene_stat_row(parent: &mut ChildSpawnerCommands, label: &'static str, value: String) {
    parent
        .spawn(Node {
            width: Val::Percent(100.0),
            justify_content: JustifyContent::SpaceBetween,
            align_items: AlignItems::FlexStart,
            column_gap: Val::Px(16.0),
            ..default()
        })
        .with_children(|row| {
            row.spawn((
                Text::new(label),
                TextFont::from_font_size(12.0),
                TextColor(Color::srgb(0.66, 0.7, 0.78)),
            ));
            row.spawn((
                Text::new(value),
                TextFont::from_font_size(12.0),
                TextColor(Color::srgb(0.82, 0.86, 0.94)),
            ));
        });
}

fn spawn_scene_category_row(
    parent: &mut ChildSpawnerCommands,
    category: &CachedSceneCategoryMetric,
) {
    spawn_scene_stat_row(
        parent,
        "category",
        format!(
            "{} | {}",
            category.label,
            scene_category_counts_text(category)
        ),
    );
}

fn scene_metric_status(metrics: &CachedSceneMetrics) -> String {
    match metrics.ok {
        Some(true) => "ok".to_string(),
        Some(false) => "needs review".to_string(),
        None => "unknown".to_string(),
    }
}

fn scene_metric_counts_text(metrics: &CachedSceneMetrics) -> String {
    let mut parts = Vec::new();
    if let Some(count) = metrics.object_count {
        parts.push(format!("{count} objects"));
    }
    if let Some(count) = metrics.asset_count {
        parts.push(format!("{count} assets"));
    }
    if let Some(count) = metrics.placement_count {
        parts.push(format!("{count} placements"));
    }
    if parts.is_empty() {
        "unavailable".to_string()
    } else {
        parts.join(" | ")
    }
}

fn scene_feedback_text(metrics: &CachedSceneMetrics) -> String {
    match (metrics.feedback_accepted, metrics.feedback_iteration) {
        (Some(true), Some(iteration)) => format!("accepted at iter {iteration}"),
        (Some(true), None) => "accepted".to_string(),
        (Some(false), Some(iteration)) => format!("failed after iter {iteration}"),
        (Some(false), None) => "failed".to_string(),
        (None, _) => "not recorded".to_string(),
    }
}

fn scene_category_counts_text(category: &CachedSceneCategoryMetric) -> String {
    let mut parts = Vec::new();
    if let Some(count) = category.object_count {
        parts.push(format!("{count} planned"));
    }
    if let Some(count) = category.detection_count {
        parts.push(format!("{count} detected"));
    }
    if let Some(count) = category.asset_count {
        parts.push(format!("{count} assets"));
    }
    if let Some(count) = category.placement_count {
        parts.push(format!("{count} placed"));
    }
    if parts.is_empty() {
        "no counts".to_string()
    } else {
        parts.join(" / ")
    }
}

fn spawn_settings_modal(
    commands: &mut Commands,
    pipeline: CatalogPipelineChoice,
    active_tab: SettingsModalTab,
    available: Option<&AvailablePipelines>,
) -> Entity {
    commands
        .spawn((
            SettingsModalRoot,
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(0.0),
                left: Val::Px(0.0),
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(MODAL_SCRIM),
        ))
        .with_children(|root| {
            root.spawn((
                Node {
                    width: Val::Px(SETTINGS_MODAL_WIDTH),
                    max_height: Val::Vh(SETTINGS_MODAL_MAX_HEIGHT_VH),
                    flex_direction: FlexDirection::Column,
                    row_gap: Val::Px(14.0),
                    padding: UiRect::all(Val::Px(16.0)),
                    border: UiRect::all(Val::Px(1.0)),
                    overflow: Overflow::clip_y(),
                    ..default()
                },
                BackgroundColor(MODAL_BG),
                BorderColor::all(MODAL_BORDER),
            ))
            .with_children(|panel| {
                panel
                    .spawn(Node {
                        width: Val::Percent(100.0),
                        justify_content: JustifyContent::SpaceBetween,
                        align_items: AlignItems::Center,
                        ..default()
                    })
                    .with_children(|header| {
                        header.spawn((
                            Text::new(settings_modal_title(pipeline)),
                            TextFont::from_font_size(16.0),
                            TextColor(Color::srgb(0.92, 0.94, 0.98)),
                        ));
                        header
                            .spawn((
                                Button,
                                SettingsCloseButton,
                                ControlButton(ControlButtonKind::Secondary),
                                Node {
                                    padding: UiRect::axes(Val::Px(9.0), Val::Px(4.0)),
                                    border: UiRect::all(Val::Px(1.0)),
                                    ..default()
                                },
                                BorderColor::all(BUTTON_BORDER),
                                BackgroundColor(BUTTON_BG),
                            ))
                            .with_children(|button| {
                                button.spawn((
                                    Text::new("close"),
                                    TextFont::from_font_size(12.0),
                                    TextColor(BUTTON_TEXT),
                                    ButtonLabel,
                                ));
                            });
                    });

                spawn_settings_tabs(panel, pipeline);
                for tab in settings_tabs_for_pipeline(pipeline) {
                    spawn_settings_tab_panel(panel, tab, active_tab == tab, |panel| {
                        spawn_settings_tab_content(panel, pipeline, tab, available);
                    });
                }
            });
        })
        .id()
}

fn settings_modal_title(pipeline: CatalogPipelineChoice) -> &'static str {
    match pipeline {
        CatalogPipelineChoice::Object(SynthesisModel::Triposg) => "TripoSG settings",
        CatalogPipelineChoice::Object(SynthesisModel::Triposplat) => "TripoSplat settings",
        CatalogPipelineChoice::Object(SynthesisModel::Trellis) => "Trellis.2 settings",
        CatalogPipelineChoice::Scene(ScenePipelineKind::Explicit) => "explicit scene settings",
    }
}

fn settings_tabs_for_pipeline(pipeline: CatalogPipelineChoice) -> Vec<SettingsModalTab> {
    match pipeline {
        CatalogPipelineChoice::Scene(ScenePipelineKind::Explicit) => vec![
            SettingsModalTab::Pipeline,
            SettingsModalTab::Generation,
            SettingsModalTab::Grounding,
            SettingsModalTab::Debug,
            SettingsModalTab::General,
            SettingsModalTab::Physics,
            SettingsModalTab::Developer,
        ],
        CatalogPipelineChoice::Object(_) => vec![
            SettingsModalTab::Pipeline,
            SettingsModalTab::General,
            SettingsModalTab::Physics,
            SettingsModalTab::Developer,
        ],
    }
}

fn spawn_settings_tabs(panel: &mut ChildSpawnerCommands, pipeline: CatalogPipelineChoice) {
    panel
        .spawn(Node {
            width: Val::Percent(100.0),
            flex_direction: FlexDirection::Row,
            column_gap: Val::Px(6.0),
            row_gap: Val::Px(6.0),
            flex_wrap: FlexWrap::Wrap,
            ..default()
        })
        .with_children(|row| {
            for tab in settings_tabs_for_pipeline(pipeline) {
                row.spawn((
                    Button,
                    SettingsTabButton { tab },
                    Node {
                        height: Val::Px(28.0),
                        padding: UiRect::axes(Val::Px(10.0), Val::Px(4.0)),
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        border: UiRect::all(Val::Px(1.0)),
                        ..default()
                    },
                    BorderColor::all(BUTTON_BORDER),
                    BackgroundColor(BUTTON_BG),
                ))
                .with_children(|button| {
                    button.spawn((
                        Text::new(tab.label()),
                        TextFont::from_font_size(12.0),
                        TextColor(BUTTON_TEXT),
                        ButtonLabel,
                    ));
                });
            }
        });
}

fn spawn_settings_tab_content(
    panel: &mut ChildSpawnerCommands,
    pipeline: CatalogPipelineChoice,
    tab: SettingsModalTab,
    available: Option<&AvailablePipelines>,
) {
    match (pipeline, tab) {
        (CatalogPipelineChoice::Object(SynthesisModel::Triposg), SettingsModalTab::Pipeline) => {
            spawn_object_pipeline_selector(panel, available, pipeline);
            spawn_triposg_settings(panel);
        }
        (CatalogPipelineChoice::Object(SynthesisModel::Triposplat), SettingsModalTab::Pipeline) => {
            spawn_object_pipeline_selector(panel, available, pipeline);
            spawn_triposplat_settings(panel);
        }
        (CatalogPipelineChoice::Object(SynthesisModel::Trellis), SettingsModalTab::Pipeline) => {
            spawn_object_pipeline_selector(panel, available, pipeline);
            spawn_trellis_settings(panel);
        }
        (CatalogPipelineChoice::Scene(ScenePipelineKind::Explicit), SettingsModalTab::Pipeline) => {
            spawn_scene_pipeline_settings(panel, available);
        }
        (
            CatalogPipelineChoice::Scene(ScenePipelineKind::Explicit),
            SettingsModalTab::Generation,
        ) => {
            spawn_scene_generation_settings(panel);
        }
        (
            CatalogPipelineChoice::Scene(ScenePipelineKind::Explicit),
            SettingsModalTab::Grounding,
        ) => {
            spawn_scene_grounding_settings(panel);
        }
        (CatalogPipelineChoice::Scene(ScenePipelineKind::Explicit), SettingsModalTab::Debug) => {
            spawn_scene_debug_settings(panel);
        }
        (_, SettingsModalTab::General) => spawn_general_settings(panel),
        (_, SettingsModalTab::Physics) => spawn_physics_settings(panel),
        (_, SettingsModalTab::Developer) => spawn_developer_settings(panel),
        _ => {}
    }
}

fn spawn_settings_tab_panel(
    parent: &mut ChildSpawnerCommands,
    tab: SettingsModalTab,
    visible: bool,
    spawn_content: impl FnOnce(&mut ChildSpawnerCommands),
) {
    parent
        .spawn((
            SettingsTabPanel { tab },
            SettingsScrollArea,
            Node {
                width: Val::Percent(100.0),
                max_height: Val::Vh(SETTINGS_TAB_BODY_MAX_HEIGHT_VH),
                display: if visible {
                    Display::Flex
                } else {
                    Display::None
                },
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(10.0),
                overflow: Overflow::scroll_y(),
                padding: UiRect::right(Val::Px(4.0)),
                ..default()
            },
            ScrollPosition::default(),
            if visible {
                Visibility::Visible
            } else {
                Visibility::Hidden
            },
        ))
        .with_children(spawn_content);
}

fn spawn_object_pipeline_selector(
    panel: &mut ChildSpawnerCommands,
    available: Option<&AvailablePipelines>,
    selected: CatalogPipelineChoice,
) {
    let choices = active_pipeline_choices(CatalogMode::Object, available);
    panel
        .spawn(Node {
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(7.0),
            ..default()
        })
        .with_children(|column| {
            column
                .spawn(Node {
                    width: Val::Percent(100.0),
                    justify_content: JustifyContent::SpaceBetween,
                    align_items: AlignItems::Center,
                    column_gap: Val::Px(10.0),
                    ..default()
                })
                .with_children(|row| {
                    row.spawn((
                        Text::new("pipeline"),
                        TextFont::from_font_size(13.0),
                        TextColor(Color::srgb(0.82, 0.86, 0.94)),
                    ));
                    row.spawn((
                        Text::new(selected.label()),
                        TextFont::from_font_size(13.0),
                        TextColor(Color::srgb(0.92, 0.94, 0.98)),
                        PipelineValueLabel,
                    ));
                });
            spawn_pipeline_button_row(column, choices);
        });
}

fn spawn_scene_pipeline_selector(
    panel: &mut ChildSpawnerCommands,
    available: Option<&AvailablePipelines>,
) {
    panel
        .spawn(Node {
            width: Val::Percent(100.0),
            justify_content: JustifyContent::SpaceBetween,
            align_items: AlignItems::Center,
            column_gap: Val::Px(10.0),
            ..default()
        })
        .with_children(|row| {
            row.spawn((
                Text::new("scene pipeline"),
                TextFont::from_font_size(13.0),
                TextColor(Color::srgb(0.82, 0.86, 0.94)),
            ));
            row.spawn((
                Text::new(ScenePipelineKind::Explicit.label()),
                TextFont::from_font_size(13.0),
                TextColor(Color::srgb(0.92, 0.94, 0.98)),
            ));
        });

    let choices = scene_image_to_3d_pipeline_choices(available);
    panel
        .spawn(Node {
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(7.0),
            ..default()
        })
        .with_children(|column| {
            column
                .spawn(Node {
                    width: Val::Percent(100.0),
                    justify_content: JustifyContent::SpaceBetween,
                    align_items: AlignItems::Center,
                    column_gap: Val::Px(10.0),
                    ..default()
                })
                .with_children(|row| {
                    row.spawn((
                        Text::new("image to 3d"),
                        TextFont::from_font_size(13.0),
                        TextColor(Color::srgb(0.82, 0.86, 0.94)),
                    ));
                    row.spawn((
                        Text::new(pipeline_label(SynthesisModel::Trellis)),
                        TextFont::from_font_size(13.0),
                        TextColor(Color::srgb(0.92, 0.94, 0.98)),
                        SceneImageTo3dModelValueLabel,
                    ));
                });
            spawn_pipeline_button_row(column, choices);
        });
}

fn scene_image_to_3d_pipeline_choices(
    available: Option<&AvailablePipelines>,
) -> Vec<CatalogPipelineChoice> {
    active_pipeline_choices(CatalogMode::Object, available)
        .into_iter()
        .filter(|choice| {
            !matches!(
                choice,
                CatalogPipelineChoice::Object(SynthesisModel::Triposplat)
            )
        })
        .collect()
}

fn spawn_pipeline_button_row(
    parent: &mut ChildSpawnerCommands,
    choices: Vec<CatalogPipelineChoice>,
) {
    parent
        .spawn(Node {
            flex_direction: FlexDirection::Row,
            column_gap: Val::Px(6.0),
            align_items: AlignItems::Center,
            flex_wrap: FlexWrap::Wrap,
            ..default()
        })
        .with_children(|row| {
            for choice in choices {
                row.spawn((
                    Button,
                    PipelineOptionButton { choice },
                    ControlButton(ControlButtonKind::Secondary),
                    Node {
                        height: Val::Px(28.0),
                        padding: UiRect::axes(Val::Px(10.0), Val::Px(4.0)),
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        border: UiRect::all(Val::Px(1.0)),
                        ..default()
                    },
                    BorderColor::all(BUTTON_BORDER),
                    BackgroundColor(BUTTON_BG),
                ))
                .with_children(|button| {
                    button.spawn((
                        Text::new(choice.label()),
                        TextFont::from_font_size(12.0),
                        TextColor(BUTTON_TEXT),
                        ButtonLabel,
                    ));
                });
            }
        });
}

fn spawn_triposplat_settings(panel: &mut ChildSpawnerCommands) {
    panel
        .spawn(Node {
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(8.0),
            ..default()
        })
        .with_children(|profiles| {
            profiles.spawn((
                Text::new("profile"),
                TextFont::from_font_size(12.0),
                TextColor(Color::srgb(0.66, 0.7, 0.78)),
            ));
            profiles
                .spawn(Node {
                    flex_direction: FlexDirection::Row,
                    column_gap: Val::Px(6.0),
                    align_items: AlignItems::Center,
                    ..default()
                })
                .with_children(|row| {
                    for profile in [
                        TripoSplatProfile::Low,
                        TripoSplatProfile::Balanced,
                        TripoSplatProfile::High,
                    ] {
                        row.spawn((
                            Button,
                            TripoSplatProfileButton { profile },
                            ControlButton(ControlButtonKind::Secondary),
                            Node {
                                padding: UiRect::axes(Val::Px(10.0), Val::Px(5.0)),
                                border: UiRect::all(Val::Px(1.0)),
                                ..default()
                            },
                            BorderColor::all(BUTTON_BORDER),
                            BackgroundColor(BUTTON_BG),
                        ))
                        .with_children(|button| {
                            button.spawn((
                                Text::new(triposplat_profile_label(profile)),
                                TextFont::from_font_size(12.0),
                                TextColor(BUTTON_TEXT),
                                ButtonLabel,
                            ));
                        });
                    }
                    row.spawn((
                        Text::new("balanced"),
                        TextFont::from_font_size(12.0),
                        TextColor(Color::srgb(0.72, 0.76, 0.84)),
                        TripoSplatProfileValueLabel,
                    ));
                });
        });

    spawn_triposplat_setting_row(panel, "steps", TripoSplatSetting::Steps);
    spawn_triposplat_setting_row(panel, "cfg guidance", TripoSplatSetting::Guidance);
    spawn_triposplat_setting_row(panel, "gaussian count", TripoSplatSetting::Gaussians);
}

fn spawn_triposg_settings(panel: &mut ChildSpawnerCommands) {
    spawn_triposg_setting_row(panel, "steps", TripoSgSetting::Steps);
    spawn_triposg_setting_row(panel, "tokens", TripoSgSetting::Tokens);
    spawn_triposg_setting_row(panel, "cfg guidance", TripoSgSetting::Guidance);
    spawn_triposg_setting_row(panel, "target faces", TripoSgSetting::TargetFaces);
}

fn spawn_trellis_settings(panel: &mut ChildSpawnerCommands) {
    panel
        .spawn(Node {
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(8.0),
            ..default()
        })
        .with_children(|profiles| {
            profiles.spawn((
                Text::new("quality"),
                TextFont::from_font_size(12.0),
                TextColor(Color::srgb(0.66, 0.7, 0.78)),
            ));
            profiles
                .spawn(Node {
                    flex_direction: FlexDirection::Row,
                    column_gap: Val::Px(6.0),
                    align_items: AlignItems::Center,
                    ..default()
                })
                .with_children(|row| {
                    for quality in [
                        TrellisQuality::Low,
                        TrellisQuality::Medium,
                        TrellisQuality::High,
                    ] {
                        row.spawn((
                            Button,
                            TrellisQualityButton { quality },
                            ControlButton(ControlButtonKind::Secondary),
                            Node {
                                padding: UiRect::axes(Val::Px(10.0), Val::Px(5.0)),
                                border: UiRect::all(Val::Px(1.0)),
                                ..default()
                            },
                            BorderColor::all(BUTTON_BORDER),
                            BackgroundColor(BUTTON_BG),
                        ))
                        .with_children(|button| {
                            button.spawn((
                                Text::new(trellis_quality_label(quality)),
                                TextFont::from_font_size(12.0),
                                TextColor(BUTTON_TEXT),
                                ButtonLabel,
                            ));
                        });
                    }
                    row.spawn((
                        Text::new(trellis_quality_value_text(TrellisQuality::Low)),
                        TextFont::from_font_size(12.0),
                        TextColor(Color::srgb(0.72, 0.76, 0.84)),
                        TrellisQualityValueLabel,
                    ));
                });
        });

    spawn_trellis_value_row(panel, "resolution", TrellisSetting::Resolution);
    spawn_trellis_toggle_row(panel, "pbr textures");
    spawn_trellis_setting_row(panel, "pbr texture size", TrellisSetting::PbrTextureSize);
    spawn_trellis_setting_row(panel, "target faces", TrellisSetting::TargetFaces);
    spawn_trellis_setting_row(panel, "sparse cap", TrellisSetting::MaxSparseCoords);
}

fn spawn_scene_pipeline_settings(
    panel: &mut ChildSpawnerCommands,
    available: Option<&AvailablePipelines>,
) {
    spawn_scene_pipeline_selector(panel, available);
    spawn_scene_quality_settings(panel);
}

fn spawn_scene_quality_settings(panel: &mut ChildSpawnerCommands) {
    panel
        .spawn(Node {
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(8.0),
            ..default()
        })
        .with_children(|profiles| {
            profiles.spawn((
                Text::new("quality"),
                TextFont::from_font_size(12.0),
                TextColor(Color::srgb(0.66, 0.7, 0.78)),
            ));
            profiles
                .spawn(Node {
                    flex_direction: FlexDirection::Row,
                    column_gap: Val::Px(6.0),
                    align_items: AlignItems::Center,
                    ..default()
                })
                .with_children(|row| {
                    for quality in [
                        SceneQualityProfileSetting::Fast,
                        SceneQualityProfileSetting::Balanced,
                        SceneQualityProfileSetting::Full,
                    ] {
                        row.spawn((
                            Button,
                            SceneQualityButton { quality },
                            ControlButton(ControlButtonKind::Secondary),
                            Node {
                                padding: UiRect::axes(Val::Px(10.0), Val::Px(5.0)),
                                border: UiRect::all(Val::Px(1.0)),
                                ..default()
                            },
                            BorderColor::all(BUTTON_BORDER),
                            BackgroundColor(BUTTON_BG),
                        ))
                        .with_children(|button| {
                            button.spawn((
                                Text::new(quality.label()),
                                TextFont::from_font_size(12.0),
                                TextColor(BUTTON_TEXT),
                                ButtonLabel,
                            ));
                        });
                    }
                    row.spawn((
                        Text::new(SceneQualityProfileSetting::Fast.label()),
                        TextFont::from_font_size(12.0),
                        TextColor(Color::srgb(0.72, 0.76, 0.84)),
                        SceneQualityValueLabel,
                    ));
                });
        });
}

fn spawn_scene_generation_settings(panel: &mut ChildSpawnerCommands) {
    spawn_scene_settings_section_label(panel, "asset generation");
    spawn_scene_setting_row(panel, "instances", SceneSetting::InstanceGeneration, 1);
    spawn_scene_toggle_row(panel, "lift assets", SceneToggleSetting::LiftAssets);
    spawn_scene_toggle_row(panel, "catalog reuse", SceneToggleSetting::CatalogReuse);
    spawn_scene_toggle_row(
        panel,
        "promote catalog",
        SceneToggleSetting::PromoteToCatalog,
    );
    spawn_scene_settings_section_label(panel, "mesh output");
    spawn_scene_toggle_row(panel, "pbr textures", SceneToggleSetting::Pbr);
    spawn_scene_setting_row(
        panel,
        "pbr texture size",
        SceneSetting::PbrTextureSize,
        TRELLIS_PBR_TEXTURE_STEP as isize,
    );
    spawn_scene_setting_row(
        panel,
        "target faces",
        SceneSetting::TargetFaces,
        TRELLIS_FACE_STEP as isize,
    );
}

fn spawn_scene_grounding_settings(panel: &mut ChildSpawnerCommands) {
    spawn_scene_settings_section_label(panel, "layout search");
    spawn_scene_setting_row(panel, "candidates", SceneSetting::CandidateCount, 1);
    spawn_scene_setting_row(panel, "ground cal", SceneSetting::GroundCalibration, 1);
    spawn_scene_setting_row(panel, "table refine", SceneSetting::TablePoseRefinement, 1);
    spawn_scene_settings_section_label(panel, "evidence");
    spawn_scene_toggle_row(panel, "locate bboxes", SceneToggleSetting::LocateAnything);
    spawn_scene_toggle_row(panel, "depth/floor", SceneToggleSetting::Depth);
    spawn_scene_toggle_row(panel, "sam masks", SceneToggleSetting::Segmentation);
    spawn_scene_toggle_row(panel, "visible fit", SceneToggleSetting::PoseFit);
}

fn spawn_scene_debug_settings(panel: &mut ChildSpawnerCommands) {
    spawn_scene_settings_section_label(panel, "feedback");
    spawn_scene_setting_row(panel, "feedback iters", SceneSetting::FeedbackIterations, 1);
    spawn_scene_toggle_row(panel, "feedback loop", SceneToggleSetting::Feedback);
    spawn_scene_settings_section_label(panel, "artifacts");
    spawn_scene_toggle_row(panel, "write artifacts", SceneToggleSetting::WriteArtifacts);
}

fn spawn_general_settings(panel: &mut ChildSpawnerCommands) {
    spawn_viewer_aabb_row(panel);
}

fn spawn_physics_settings(panel: &mut ChildSpawnerCommands) {
    spawn_viewer_debug_toggle_row(
        panel,
        "ground/contact",
        ViewerDebugToggleSetting::GroundContact,
    );
    spawn_viewer_debug_numeric_row(
        panel,
        "ground y",
        ViewerDebugNumericSetting::GroundY,
        VIEWER_GROUND_Y_STEP,
    );
    spawn_viewer_debug_numeric_row(
        panel,
        "contact tolerance",
        ViewerDebugNumericSetting::ContactTolerance,
        VIEWER_CONTACT_TOLERANCE_STEP,
    );
}

fn spawn_developer_settings(panel: &mut ChildSpawnerCommands) {
    spawn_developer_tabs(panel);
    spawn_developer_tab_panel(panel, DeveloperPanelTab::Status, true, |panel| {
        spawn_developer_text_block::<SettingsDeveloperCurrentText>(panel, "current", "idle");
        spawn_developer_text_block::<SettingsDeveloperTokenText>(panel, "tokens", "no token usage");
        spawn_developer_section_label(panel, "debug overlays");
        spawn_viewer_debug_toggle_row(
            panel,
            "scene camera frustum",
            ViewerDebugToggleSetting::SceneCameraFrustum,
        );
        spawn_viewer_debug_numeric_row(
            panel,
            "frustum length",
            ViewerDebugNumericSetting::SceneCameraFrustumLength,
            VIEWER_FRUSTUM_LENGTH_STEP,
        );
        spawn_viewer_debug_toggle_row(
            panel,
            "depth rgb splats",
            ViewerDebugToggleSetting::DepthCloud,
        );
        spawn_viewer_debug_numeric_row(
            panel,
            "depth splat cap",
            ViewerDebugNumericSetting::DepthCloudMaxGaussians,
            VIEWER_DEPTH_CLOUD_GAUSSIAN_STEP as f32,
        );
    });
    spawn_developer_tab_panel(panel, DeveloperPanelTab::Events, false, |panel| {
        spawn_developer_text_block::<SettingsDeveloperEventsText>(
            panel,
            "events",
            "no scene build events yet",
        );
    });
    spawn_developer_tab_panel(panel, DeveloperPanelTab::Artifacts, false, |panel| {
        spawn_developer_text_block::<SettingsDeveloperArtifactText>(
            panel,
            "artifacts",
            "no artifacts yet",
        );
    });
    spawn_developer_tab_panel(panel, DeveloperPanelTab::Visuals, false, |panel| {
        spawn_developer_text_block::<SettingsDeveloperVisualText>(
            panel,
            "visual artifacts",
            "no visual artifacts yet",
        );
        spawn_developer_visual_pager(panel);
        panel.spawn((
            SettingsDeveloperVisualGrid::default(),
            Node {
                width: Val::Percent(100.0),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(8.0),
                ..default()
            },
        ));
    });
}

fn spawn_developer_visual_pager(panel: &mut ChildSpawnerCommands) {
    panel
        .spawn(Node {
            width: Val::Percent(100.0),
            height: Val::Px(28.0),
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Center,
            column_gap: Val::Px(8.0),
            ..default()
        })
        .with_children(|row| {
            spawn_developer_visual_page_button(row, DeveloperVisualPageDirection::Previous, "<");
            row.spawn((
                Text::new("page 0/0 | 0 images"),
                TextFont::from_font_size(10.5),
                TextColor(Color::srgb(0.72, 0.77, 0.86)),
                SettingsDeveloperVisualPagerText,
                Node {
                    flex_grow: 1.0,
                    justify_content: JustifyContent::Center,
                    ..default()
                },
            ));
            spawn_developer_visual_page_button(row, DeveloperVisualPageDirection::Next, ">");
        });
}

fn spawn_developer_visual_page_button(
    row: &mut ChildSpawnerCommands,
    direction: DeveloperVisualPageDirection,
    label: &'static str,
) {
    row.spawn((
        Button,
        SettingsDeveloperVisualPageButton { direction },
        Node {
            width: Val::Px(30.0),
            height: Val::Px(24.0),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            border: UiRect::all(Val::Px(1.0)),
            ..default()
        },
        BorderColor::all(BUTTON_BORDER_DISABLED),
        BackgroundColor(BUTTON_BG_DISABLED),
    ))
    .with_children(|button| {
        button.spawn((
            Text::new(label),
            TextFont::from_font_size(12.0),
            TextColor(BUTTON_TEXT_DISABLED),
            ButtonLabel,
        ));
    });
}

fn spawn_developer_tabs(panel: &mut ChildSpawnerCommands) {
    panel
        .spawn(Node {
            flex_direction: FlexDirection::Row,
            column_gap: Val::Px(6.0),
            ..default()
        })
        .with_children(|row| {
            for tab in [
                DeveloperPanelTab::Status,
                DeveloperPanelTab::Events,
                DeveloperPanelTab::Artifacts,
                DeveloperPanelTab::Visuals,
            ] {
                row.spawn((
                    Button,
                    SettingsDeveloperTabButton { tab },
                    Node {
                        height: Val::Px(26.0),
                        padding: UiRect::axes(Val::Px(9.0), Val::Px(4.0)),
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        border: UiRect::all(Val::Px(1.0)),
                        ..default()
                    },
                    BorderColor::all(BUTTON_BORDER),
                    BackgroundColor(BUTTON_BG),
                ))
                .with_children(|button| {
                    button.spawn((
                        Text::new(tab.label()),
                        TextFont::from_font_size(11.0),
                        TextColor(BUTTON_TEXT),
                        ButtonLabel,
                    ));
                });
            }
        });
}

fn spawn_developer_tab_panel(
    parent: &mut ChildSpawnerCommands,
    tab: DeveloperPanelTab,
    visible: bool,
    spawn_content: impl FnOnce(&mut ChildSpawnerCommands),
) {
    parent
        .spawn((
            SettingsDeveloperTabPanel { tab },
            Node {
                width: Val::Percent(100.0),
                display: if visible {
                    Display::Flex
                } else {
                    Display::None
                },
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(10.0),
                ..default()
            },
            if visible {
                Visibility::Visible
            } else {
                Visibility::Hidden
            },
        ))
        .with_children(spawn_content);
}

fn spawn_developer_text_block<T: Component + Default>(
    panel: &mut ChildSpawnerCommands,
    label: &'static str,
    value: &'static str,
) {
    panel
        .spawn(Node {
            width: Val::Percent(100.0),
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(4.0),
            ..default()
        })
        .with_children(|block| {
            block.spawn((
                Text::new(label),
                TextFont::from_font_size(11.0),
                TextColor(Color::srgb(0.58, 0.64, 0.74)),
            ));
            block.spawn((
                Text::new(value),
                TextFont::from_font_size(10.5),
                TextColor(Color::srgb(0.76, 0.81, 0.9)),
                T::default(),
            ));
        });
}

fn spawn_developer_visual_preview_row(
    parent: &mut ChildSpawnerCommands,
    preview: &ProcessingArtifactPreview,
) {
    let filename = preview
        .path
        .rsplit(['/', '\\'])
        .next()
        .filter(|value| !value.is_empty())
        .unwrap_or(preview.path.as_str());
    parent
        .spawn(Node {
            width: Val::Percent(100.0),
            flex_direction: FlexDirection::Row,
            column_gap: Val::Px(8.0),
            align_items: AlignItems::Center,
            ..default()
        })
        .with_children(|row| {
            row.spawn((
                Node {
                    width: Val::Px(DEVELOPER_VISUAL_THUMB_WIDTH),
                    height: Val::Px(DEVELOPER_VISUAL_THUMB_HEIGHT),
                    border: UiRect::all(Val::Px(1.0)),
                    ..default()
                },
                BackgroundColor(Color::srgb(0.04, 0.045, 0.055)),
                BorderColor::all(Color::srgb(0.22, 0.26, 0.34)),
                ImageNode::new(preview.image.clone()),
            ));
            row.spawn(Node {
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(3.0),
                flex_grow: 1.0,
                ..default()
            })
            .with_children(|labels| {
                labels.spawn((
                    Text::new(preview.kind.label()),
                    TextFont::from_font_size(11.0),
                    TextColor(Color::srgb(0.62, 0.7, 0.86)),
                ));
                labels.spawn((
                    Text::new(ellipsize_text(filename, 42)),
                    TextFont::from_font_size(10.5),
                    TextColor(Color::srgb(0.78, 0.82, 0.9)),
                ));
            });
        });
}

fn spawn_viewer_aabb_row(parent: &mut ChildSpawnerCommands) {
    parent
        .spawn(Node {
            width: Val::Percent(100.0),
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(7.0),
            ..default()
        })
        .with_children(|column| {
            column
                .spawn(Node {
                    width: Val::Percent(100.0),
                    justify_content: JustifyContent::SpaceBetween,
                    align_items: AlignItems::Center,
                    column_gap: Val::Px(10.0),
                    ..default()
                })
                .with_children(|row| {
                    row.spawn((
                        Text::new("object AABB"),
                        TextFont::from_font_size(13.0),
                        TextColor(Color::srgb(0.82, 0.86, 0.94)),
                    ));
                    row.spawn((
                        Text::new(ViewerAabbOverlayMode::Selected.label()),
                        TextFont::from_font_size(13.0),
                        TextColor(Color::srgb(0.92, 0.94, 0.98)),
                        ViewerAabbModeValueLabel,
                    ));
                });
            column
                .spawn(Node {
                    flex_direction: FlexDirection::Row,
                    column_gap: Val::Px(6.0),
                    align_items: AlignItems::Center,
                    ..default()
                })
                .with_children(|row| {
                    for mode in [
                        ViewerAabbOverlayMode::Off,
                        ViewerAabbOverlayMode::Selected,
                        ViewerAabbOverlayMode::All,
                    ] {
                        row.spawn((
                            Button,
                            ViewerAabbModeButton { mode },
                            ControlButton(ControlButtonKind::Secondary),
                            Node {
                                height: Val::Px(28.0),
                                padding: UiRect::axes(Val::Px(10.0), Val::Px(4.0)),
                                justify_content: JustifyContent::Center,
                                align_items: AlignItems::Center,
                                border: UiRect::all(Val::Px(1.0)),
                                ..default()
                            },
                            BorderColor::all(BUTTON_BORDER),
                            BackgroundColor(BUTTON_BG),
                        ))
                        .with_children(|button| {
                            button.spawn((
                                Text::new(mode.label()),
                                TextFont::from_font_size(12.0),
                                TextColor(BUTTON_TEXT),
                                ButtonLabel,
                            ));
                        });
                    }
                });
        });
}

fn spawn_viewer_debug_toggle_row(
    parent: &mut ChildSpawnerCommands,
    label: &'static str,
    setting: ViewerDebugToggleSetting,
) {
    parent
        .spawn(Node {
            width: Val::Percent(100.0),
            justify_content: JustifyContent::SpaceBetween,
            align_items: AlignItems::Center,
            column_gap: Val::Px(10.0),
            ..default()
        })
        .with_children(|row| {
            row.spawn((
                Text::new(label),
                TextFont::from_font_size(13.0),
                TextColor(Color::srgb(0.82, 0.86, 0.94)),
            ));
            row.spawn((
                Button,
                ViewerDebugToggleButton { setting },
                ControlButton(ControlButtonKind::Secondary),
                Node {
                    width: Val::Px(72.0),
                    height: Val::Px(26.0),
                    justify_content: JustifyContent::Center,
                    align_items: AlignItems::Center,
                    border: UiRect::all(Val::Px(1.0)),
                    ..default()
                },
                BorderColor::all(BUTTON_BORDER),
                BackgroundColor(BUTTON_BG),
            ))
            .with_children(|button| {
                button.spawn((
                    Text::new("on"),
                    TextFont::from_font_size(13.0),
                    TextColor(BUTTON_TEXT),
                    ButtonLabel,
                    ViewerDebugToggleValueLabel { setting },
                ));
            });
        });
}

fn spawn_viewer_debug_numeric_row(
    parent: &mut ChildSpawnerCommands,
    label: &'static str,
    setting: ViewerDebugNumericSetting,
    step: f32,
) {
    parent
        .spawn(Node {
            width: Val::Percent(100.0),
            justify_content: JustifyContent::SpaceBetween,
            align_items: AlignItems::Center,
            column_gap: Val::Px(10.0),
            ..default()
        })
        .with_children(|row| {
            row.spawn((
                Text::new(label),
                TextFont::from_font_size(13.0),
                TextColor(Color::srgb(0.82, 0.86, 0.94)),
            ));
            row.spawn(Node {
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Center,
                column_gap: Val::Px(7.0),
                ..default()
            })
            .with_children(|control| {
                spawn_viewer_debug_step_button(control, setting, -step);
                control.spawn((
                    Text::new("0.00"),
                    TextFont::from_font_size(13.0),
                    TextColor(Color::srgb(0.92, 0.94, 0.98)),
                    ViewerDebugNumericValueLabel { setting },
                ));
                spawn_viewer_debug_step_button(control, setting, step);
            });
        });
}

fn spawn_viewer_debug_step_button(
    parent: &mut ChildSpawnerCommands,
    setting: ViewerDebugNumericSetting,
    delta: f32,
) {
    parent
        .spawn((
            Button,
            ViewerDebugStepButton { setting, delta },
            ControlButton(ControlButtonKind::Nav),
            Node {
                width: Val::Px(28.0),
                height: Val::Px(24.0),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                border: UiRect::all(Val::Px(1.0)),
                ..default()
            },
            BorderColor::all(BUTTON_BORDER),
            BackgroundColor(BUTTON_BG),
        ))
        .with_children(|button| {
            button.spawn((
                Text::new(if delta > 0.0 { "+" } else { "-" }),
                TextFont::from_font_size(14.0),
                TextColor(BUTTON_TEXT),
                ButtonLabel,
            ));
        });
}

fn spawn_triposg_setting_row(
    parent: &mut ChildSpawnerCommands,
    label: &'static str,
    setting: TripoSgSetting,
) {
    parent
        .spawn(Node {
            width: Val::Percent(100.0),
            justify_content: JustifyContent::SpaceBetween,
            align_items: AlignItems::Center,
            column_gap: Val::Px(10.0),
            ..default()
        })
        .with_children(|row| {
            row.spawn((
                Text::new(label),
                TextFont::from_font_size(13.0),
                TextColor(Color::srgb(0.82, 0.86, 0.94)),
            ));
            row.spawn(Node {
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Center,
                column_gap: Val::Px(7.0),
                ..default()
            })
            .with_children(|control| {
                spawn_triposg_setting_step_button(control, setting, false);
                control.spawn((
                    Text::new("0"),
                    TextFont::from_font_size(13.0),
                    TextColor(Color::srgb(0.92, 0.94, 0.98)),
                    TripoSgSettingValueLabel { setting },
                ));
                spawn_triposg_setting_step_button(control, setting, true);
            });
        });
}

fn spawn_trellis_value_row(
    parent: &mut ChildSpawnerCommands,
    label: &'static str,
    setting: TrellisSetting,
) {
    parent
        .spawn(Node {
            width: Val::Percent(100.0),
            justify_content: JustifyContent::SpaceBetween,
            align_items: AlignItems::Center,
            column_gap: Val::Px(10.0),
            ..default()
        })
        .with_children(|row| {
            row.spawn((
                Text::new(label),
                TextFont::from_font_size(13.0),
                TextColor(Color::srgb(0.82, 0.86, 0.94)),
            ));
            row.spawn((
                Text::new("0"),
                TextFont::from_font_size(13.0),
                TextColor(Color::srgb(0.92, 0.94, 0.98)),
                TrellisSettingValueLabel { setting },
            ));
        });
}

fn spawn_trellis_toggle_row(parent: &mut ChildSpawnerCommands, label: &'static str) {
    parent
        .spawn(Node {
            width: Val::Percent(100.0),
            justify_content: JustifyContent::SpaceBetween,
            align_items: AlignItems::Center,
            column_gap: Val::Px(10.0),
            ..default()
        })
        .with_children(|row| {
            row.spawn((
                Text::new(label),
                TextFont::from_font_size(13.0),
                TextColor(Color::srgb(0.82, 0.86, 0.94)),
            ));
            row.spawn((
                Button,
                TrellisPbrToggleButton,
                ControlButton(ControlButtonKind::Secondary),
                Node {
                    width: Val::Px(72.0),
                    height: Val::Px(26.0),
                    justify_content: JustifyContent::Center,
                    align_items: AlignItems::Center,
                    border: UiRect::all(Val::Px(1.0)),
                    ..default()
                },
                BorderColor::all(BUTTON_BORDER),
                BackgroundColor(BUTTON_BG),
            ))
            .with_children(|button| {
                button.spawn((
                    Text::new("on"),
                    TextFont::from_font_size(13.0),
                    TextColor(BUTTON_TEXT),
                    ButtonLabel,
                    TrellisSettingValueLabel {
                        setting: TrellisSetting::Pbr,
                    },
                ));
            });
        });
}

fn spawn_trellis_setting_row(
    parent: &mut ChildSpawnerCommands,
    label: &'static str,
    setting: TrellisSetting,
) {
    parent
        .spawn(Node {
            width: Val::Percent(100.0),
            justify_content: JustifyContent::SpaceBetween,
            align_items: AlignItems::Center,
            column_gap: Val::Px(10.0),
            ..default()
        })
        .with_children(|row| {
            row.spawn((
                Text::new(label),
                TextFont::from_font_size(13.0),
                TextColor(Color::srgb(0.82, 0.86, 0.94)),
            ));
            row.spawn(Node {
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Center,
                column_gap: Val::Px(7.0),
                ..default()
            })
            .with_children(|control| {
                spawn_trellis_setting_step_button(control, setting, false);
                control.spawn((
                    Text::new("0"),
                    TextFont::from_font_size(13.0),
                    TextColor(Color::srgb(0.92, 0.94, 0.98)),
                    TrellisSettingValueLabel { setting },
                ));
                spawn_trellis_setting_step_button(control, setting, true);
            });
        });
}

fn spawn_scene_setting_row(
    parent: &mut ChildSpawnerCommands,
    label: &'static str,
    setting: SceneSetting,
    step: isize,
) {
    parent
        .spawn(Node {
            width: Val::Percent(100.0),
            justify_content: JustifyContent::SpaceBetween,
            align_items: AlignItems::Center,
            column_gap: Val::Px(10.0),
            ..default()
        })
        .with_children(|row| {
            row.spawn((
                Text::new(label),
                TextFont::from_font_size(13.0),
                TextColor(Color::srgb(0.82, 0.86, 0.94)),
            ));
            row.spawn(Node {
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Center,
                column_gap: Val::Px(7.0),
                ..default()
            })
            .with_children(|control| {
                spawn_scene_setting_step_button(control, setting, -step);
                control.spawn((
                    Text::new("0"),
                    TextFont::from_font_size(13.0),
                    TextColor(Color::srgb(0.92, 0.94, 0.98)),
                    SceneSettingValueLabel { setting },
                ));
                spawn_scene_setting_step_button(control, setting, step);
            });
        });
}

fn spawn_scene_settings_section_label(parent: &mut ChildSpawnerCommands, label: &'static str) {
    parent.spawn((
        Text::new(label),
        TextFont::from_font_size(11.0),
        TextColor(Color::srgb(0.58, 0.64, 0.74)),
    ));
}

fn spawn_developer_section_label(parent: &mut ChildSpawnerCommands, label: &'static str) {
    spawn_scene_settings_section_label(parent, label);
}

fn spawn_scene_toggle_row(
    parent: &mut ChildSpawnerCommands,
    label: &'static str,
    setting: SceneToggleSetting,
) {
    parent
        .spawn(Node {
            width: Val::Percent(100.0),
            justify_content: JustifyContent::SpaceBetween,
            align_items: AlignItems::Center,
            column_gap: Val::Px(10.0),
            ..default()
        })
        .with_children(|row| {
            row.spawn((
                Text::new(label),
                TextFont::from_font_size(13.0),
                TextColor(Color::srgb(0.82, 0.86, 0.94)),
            ));
            row.spawn((
                Button,
                SceneSettingToggleButton { setting },
                ControlButton(ControlButtonKind::Secondary),
                Node {
                    width: Val::Px(72.0),
                    height: Val::Px(26.0),
                    justify_content: JustifyContent::Center,
                    align_items: AlignItems::Center,
                    border: UiRect::all(Val::Px(1.0)),
                    ..default()
                },
                BorderColor::all(BUTTON_BORDER),
                BackgroundColor(BUTTON_BG),
            ))
            .with_children(|button| {
                button.spawn((
                    Text::new("on"),
                    TextFont::from_font_size(13.0),
                    TextColor(BUTTON_TEXT),
                    ButtonLabel,
                    SceneToggleValueLabel { setting },
                ));
            });
        });
}

fn spawn_scene_setting_step_button(
    parent: &mut ChildSpawnerCommands,
    setting: SceneSetting,
    delta: isize,
) {
    parent
        .spawn((
            Button,
            SceneSettingStepButton {
                setting,
                delta: SceneSettingDelta::Integer(delta),
            },
            ControlButton(ControlButtonKind::Nav),
            Node {
                width: Val::Px(28.0),
                height: Val::Px(24.0),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                border: UiRect::all(Val::Px(1.0)),
                ..default()
            },
            BorderColor::all(BUTTON_BORDER),
            BackgroundColor(BUTTON_BG),
        ))
        .with_children(|button| {
            button.spawn((
                Text::new(if delta > 0 { "+" } else { "-" }),
                TextFont::from_font_size(14.0),
                TextColor(BUTTON_TEXT),
                ButtonLabel,
            ));
        });
}

fn spawn_triposplat_setting_row(
    parent: &mut ChildSpawnerCommands,
    label: &'static str,
    setting: TripoSplatSetting,
) {
    parent
        .spawn(Node {
            width: Val::Percent(100.0),
            justify_content: JustifyContent::SpaceBetween,
            align_items: AlignItems::Center,
            column_gap: Val::Px(10.0),
            ..default()
        })
        .with_children(|row| {
            row.spawn((
                Text::new(label),
                TextFont::from_font_size(13.0),
                TextColor(Color::srgb(0.82, 0.86, 0.94)),
            ));
            row.spawn(Node {
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Center,
                column_gap: Val::Px(7.0),
                ..default()
            })
            .with_children(|control| {
                spawn_setting_step_button(control, setting, false);
                control.spawn((
                    Text::new("0"),
                    TextFont::from_font_size(13.0),
                    TextColor(Color::srgb(0.92, 0.94, 0.98)),
                    TripoSplatSettingValueLabel { setting },
                ));
                spawn_setting_step_button(control, setting, true);
            });
        });
}

fn spawn_triposg_setting_step_button(
    parent: &mut ChildSpawnerCommands,
    setting: TripoSgSetting,
    positive: bool,
) {
    let delta = match setting {
        TripoSgSetting::Steps => TripoSgSettingDelta::Integer(if positive { 1 } else { -1 }),
        TripoSgSetting::Tokens => TripoSgSettingDelta::Integer(if positive {
            TRIPOSG_TOKEN_STEP as isize
        } else {
            -(TRIPOSG_TOKEN_STEP as isize)
        }),
        TripoSgSetting::Guidance => TripoSgSettingDelta::Float(if positive {
            TRIPOSG_GUIDANCE_STEP
        } else {
            -TRIPOSG_GUIDANCE_STEP
        }),
        TripoSgSetting::TargetFaces => TripoSgSettingDelta::Integer(if positive {
            TRIPOSG_FACE_STEP as isize
        } else {
            -(TRIPOSG_FACE_STEP as isize)
        }),
    };
    parent
        .spawn((
            Button,
            TripoSgSettingStepButton { setting, delta },
            ControlButton(ControlButtonKind::Nav),
            Node {
                width: Val::Px(28.0),
                height: Val::Px(24.0),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                border: UiRect::all(Val::Px(1.0)),
                ..default()
            },
            BorderColor::all(BUTTON_BORDER),
            BackgroundColor(BUTTON_BG),
        ))
        .with_children(|button| {
            button.spawn((
                Text::new(if positive { "+" } else { "-" }),
                TextFont::from_font_size(14.0),
                TextColor(BUTTON_TEXT),
                ButtonLabel,
            ));
        });
}

fn spawn_trellis_setting_step_button(
    parent: &mut ChildSpawnerCommands,
    setting: TrellisSetting,
    positive: bool,
) {
    let delta = match setting {
        TrellisSetting::PbrTextureSize => TrellisSettingDelta::Integer(if positive {
            TRELLIS_PBR_TEXTURE_STEP as isize
        } else {
            -(TRELLIS_PBR_TEXTURE_STEP as isize)
        }),
        TrellisSetting::TargetFaces => TrellisSettingDelta::Integer(if positive {
            TRELLIS_FACE_STEP as isize
        } else {
            -(TRELLIS_FACE_STEP as isize)
        }),
        TrellisSetting::MaxSparseCoords => TrellisSettingDelta::Integer(if positive {
            TRELLIS_SPARSE_COORD_STEP as isize
        } else {
            -(TRELLIS_SPARSE_COORD_STEP as isize)
        }),
        TrellisSetting::Resolution | TrellisSetting::Pbr => TrellisSettingDelta::Integer(0),
    };
    parent
        .spawn((
            Button,
            TrellisSettingStepButton { setting, delta },
            ControlButton(ControlButtonKind::Nav),
            Node {
                width: Val::Px(28.0),
                height: Val::Px(24.0),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                border: UiRect::all(Val::Px(1.0)),
                ..default()
            },
            BorderColor::all(BUTTON_BORDER),
            BackgroundColor(BUTTON_BG),
        ))
        .with_children(|button| {
            button.spawn((
                Text::new(if positive { "+" } else { "-" }),
                TextFont::from_font_size(14.0),
                TextColor(BUTTON_TEXT),
                ButtonLabel,
            ));
        });
}

fn spawn_setting_step_button(
    parent: &mut ChildSpawnerCommands,
    setting: TripoSplatSetting,
    positive: bool,
) {
    let delta = match setting {
        TripoSplatSetting::Steps => TripoSplatSettingDelta::Integer(if positive { 1 } else { -1 }),
        TripoSplatSetting::Guidance => TripoSplatSettingDelta::Float(if positive {
            TRIPOSPLAT_GUIDANCE_STEP
        } else {
            -TRIPOSPLAT_GUIDANCE_STEP
        }),
        TripoSplatSetting::Gaussians => TripoSplatSettingDelta::Integer(if positive {
            TRIPOSPLAT_GAUSSIAN_STEP as isize
        } else {
            -(TRIPOSPLAT_GAUSSIAN_STEP as isize)
        }),
    };
    parent
        .spawn((
            Button,
            TripoSplatSettingStepButton { setting, delta },
            ControlButton(ControlButtonKind::Nav),
            Node {
                width: Val::Px(28.0),
                height: Val::Px(24.0),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                border: UiRect::all(Val::Px(1.0)),
                ..default()
            },
            BorderColor::all(BUTTON_BORDER),
            BackgroundColor(BUTTON_BG),
        ))
        .with_children(|button| {
            button.spawn((
                Text::new(if positive { "+" } else { "-" }),
                TextFont::from_font_size(14.0),
                TextColor(BUTTON_TEXT),
                ButtonLabel,
            ));
        });
}

fn adjust_triposplat_setting(
    args: &mut AppArgs,
    setting: TripoSplatSetting,
    delta: TripoSplatSettingDelta,
) {
    match (setting, delta) {
        (TripoSplatSetting::Steps, TripoSplatSettingDelta::Integer(delta)) => {
            args.num_steps = apply_integer_delta(
                args.num_steps,
                delta,
                TRIPOSPLAT_MIN_STEPS,
                TRIPOSPLAT_MAX_STEPS,
            );
        }
        (TripoSplatSetting::Guidance, TripoSplatSettingDelta::Float(delta)) => {
            args.guidance_scale = (args.guidance_scale + delta)
                .clamp(TRIPOSPLAT_MIN_GUIDANCE, TRIPOSPLAT_MAX_GUIDANCE);
        }
        (TripoSplatSetting::Gaussians, TripoSplatSettingDelta::Integer(delta)) => {
            args.triposplat_num_gaussians = apply_integer_delta(
                args.triposplat_num_gaussians,
                delta,
                TRIPOSPLAT_MIN_NUM_GAUSSIANS,
                TRIPOSPLAT_MAX_NUM_GAUSSIANS,
            );
        }
        _ => {}
    }
    args.refresh_triposplat_profile_from_current_settings();
    info!(
        "TripoSplat settings: profile={} steps={} guidance_scale={:.3} gaussians={}",
        triposplat_profile_label(args.triposplat_profile),
        args.num_steps,
        args.guidance_scale,
        args.triposplat_num_gaussians
    );
}

fn adjust_triposg_setting(args: &mut AppArgs, setting: TripoSgSetting, delta: TripoSgSettingDelta) {
    match (setting, delta) {
        (TripoSgSetting::Steps, TripoSgSettingDelta::Integer(delta)) => {
            args.num_steps =
                apply_integer_delta(args.num_steps, delta, TRIPOSG_MIN_STEPS, TRIPOSG_MAX_STEPS);
        }
        (TripoSgSetting::Tokens, TripoSgSettingDelta::Integer(delta)) => {
            args.num_tokens = apply_integer_delta(
                args.num_tokens,
                delta,
                TRIPOSG_MIN_TOKENS,
                TRIPOSG_MAX_TOKENS,
            );
        }
        (TripoSgSetting::Guidance, TripoSgSettingDelta::Float(delta)) => {
            args.guidance_scale =
                (args.guidance_scale + delta).clamp(TRIPOSG_MIN_GUIDANCE, TRIPOSG_MAX_GUIDANCE);
        }
        (TripoSgSetting::TargetFaces, TripoSgSettingDelta::Integer(delta)) => {
            let current = args.target_faces.unwrap_or(0);
            let next = apply_integer_delta(current, delta, 0, TRIPOSG_MAX_FACES);
            args.target_faces = (next > 0).then_some(next);
        }
        _ => {}
    }
    info!(
        "TripoSG settings: steps={} tokens={} guidance_scale={:.3} target_faces={}",
        args.num_steps,
        args.num_tokens,
        args.guidance_scale,
        args.target_faces
            .map(format_grouped_usize)
            .unwrap_or_else(|| "disabled".to_string())
    );
}

fn adjust_trellis_setting(args: &mut AppArgs, setting: TrellisSetting, delta: TrellisSettingDelta) {
    match (setting, delta) {
        (TrellisSetting::PbrTextureSize, TrellisSettingDelta::Integer(delta)) => {
            let current = args
                .trellis_pbr_texture_size
                .unwrap_or(DEFAULT_TRELLIS_PBR_TEXTURE_SIZE);
            args.trellis_pbr_texture_size = Some(apply_integer_delta(
                current,
                delta,
                TRELLIS_PBR_TEXTURE_MIN,
                TRELLIS_PBR_TEXTURE_MAX,
            ));
        }
        (TrellisSetting::TargetFaces, TrellisSettingDelta::Integer(delta)) => {
            let current = args.trellis_target_faces.unwrap_or(0);
            let next = apply_integer_delta(current, delta, 0, TRELLIS_MAX_FACES);
            args.trellis_target_faces = (next > 0).then_some(next);
        }
        (TrellisSetting::MaxSparseCoords, TrellisSettingDelta::Integer(delta)) => {
            let current = args.trellis_max_sparse_coords.unwrap_or(0);
            let next = apply_integer_delta(current, delta, 0, TRELLIS_MAX_SPARSE_COORDS);
            args.trellis_max_sparse_coords = (next > 0).then_some(next);
        }
        _ => {}
    }
    info!(
        "Trellis.2 settings: quality={} pbr={} texture_size={} target_faces={} max_sparse_coords={}",
        trellis_quality_label(args.trellis_quality),
        if args.trellis_pbr_enabled {
            "on"
        } else {
            "off"
        },
        args.trellis_pbr_texture_size
            .map(format_grouped_usize)
            .unwrap_or_else(|| "runtime".to_string()),
        args.trellis_target_faces
            .map(format_grouped_usize)
            .unwrap_or_else(|| "disabled".to_string()),
        args.trellis_max_sparse_coords
            .map(format_grouped_usize)
            .unwrap_or_else(|| "uncapped".to_string())
    );
}

fn adjust_scene_setting(
    settings: &mut ScenePipelineUiSettings,
    setting: SceneSetting,
    delta: SceneSettingDelta,
) {
    match (setting, delta) {
        (SceneSetting::GroundCalibration, SceneSettingDelta::Integer(delta)) => {
            settings.ground_calibration = settings.ground_calibration.cycle(delta);
        }
        (SceneSetting::InstanceGeneration, SceneSettingDelta::Integer(delta)) => {
            settings.instance_generation = settings.instance_generation.cycle(delta);
        }
        (SceneSetting::TablePoseRefinement, SceneSettingDelta::Integer(delta)) => {
            settings.table_pose_refinement = settings.table_pose_refinement.cycle(delta);
        }
        (SceneSetting::CandidateCount, SceneSettingDelta::Integer(delta)) => {
            settings.candidate_count = apply_integer_delta(settings.candidate_count, delta, 1, 6);
        }
        (SceneSetting::FeedbackIterations, SceneSettingDelta::Integer(delta)) => {
            settings.feedback_iterations =
                apply_integer_delta(settings.feedback_iterations, delta, 0, 24);
        }
        (SceneSetting::PbrTextureSize, SceneSettingDelta::Integer(delta)) => {
            settings.pbr_texture_size = apply_integer_delta(
                settings.pbr_texture_size,
                delta,
                TRELLIS_PBR_TEXTURE_MIN,
                TRELLIS_PBR_TEXTURE_MAX,
            );
        }
        (SceneSetting::TargetFaces, SceneSettingDelta::Integer(delta)) => {
            settings.target_faces = apply_integer_delta(
                settings.target_faces,
                delta,
                TRELLIS_FACE_STEP,
                TRELLIS_MAX_FACES,
            );
        }
    }
    info!(
        "explicit scene settings: image_to_3d={} quality={} ground_calibration={} instances={} table_refine={} candidates={} feedback_iters={} pbr={} texture_size={} target_faces={} catalog_reuse={} lift_assets={} locate={} depth={} segmentation={} pose_fit={} feedback={} artifacts={} promote={}",
        pipeline_label(settings.image_to_3d_model),
        settings.quality_profile.label(),
        settings.ground_calibration.label(),
        settings.instance_generation.label(),
        settings.table_pose_refinement.label(),
        settings.candidate_count,
        settings.feedback_iterations,
        if settings.pbr_enabled { "on" } else { "off" },
        format_grouped_usize(settings.pbr_texture_size),
        format_grouped_usize(settings.target_faces),
        if settings.allow_catalog_reuse {
            "on"
        } else {
            "off"
        },
        if settings.lift_assets { "on" } else { "off" },
        if settings.locate_anything_enabled {
            "on"
        } else {
            "off"
        },
        if settings.depth_enabled { "on" } else { "off" },
        if settings.segmentation_enabled {
            "on"
        } else {
            "off"
        },
        if settings.pose_fit_enabled {
            "on"
        } else {
            "off"
        },
        if settings.feedback_enabled {
            "on"
        } else {
            "off"
        },
        if settings.write_artifacts {
            "on"
        } else {
            "off"
        },
        if settings.promote_to_catalog {
            "on"
        } else {
            "off"
        },
    );
}

fn apply_integer_delta(value: usize, delta: isize, min: usize, max: usize) -> usize {
    value.saturating_add_signed(delta).clamp(min, max)
}

fn triposplat_profile_label(profile: TripoSplatProfile) -> &'static str {
    match profile {
        TripoSplatProfile::Low => "low",
        TripoSplatProfile::Balanced => "balanced",
        TripoSplatProfile::High => "high",
        TripoSplatProfile::Custom => "custom",
    }
}

fn trellis_quality_label(quality: TrellisQuality) -> &'static str {
    match quality {
        TrellisQuality::Low => "low",
        TrellisQuality::Medium => "medium",
        TrellisQuality::High => "high",
    }
}

fn trellis_resolution_text(quality: TrellisQuality) -> &'static str {
    match quality {
        TrellisQuality::Low => "512",
        TrellisQuality::Medium | TrellisQuality::High => "1024",
    }
}

fn trellis_quality_value_text(quality: TrellisQuality) -> String {
    format!(
        "{} / {}",
        trellis_quality_label(quality),
        trellis_resolution_text(quality)
    )
}

fn triposplat_setting_value_text(args: &AppArgs, setting: TripoSplatSetting) -> String {
    match setting {
        TripoSplatSetting::Steps => args.num_steps.to_string(),
        TripoSplatSetting::Guidance => format!("{:.1}", args.guidance_scale),
        TripoSplatSetting::Gaussians => format_grouped_usize(args.triposplat_num_gaussians),
    }
}

fn triposg_setting_value_text(args: &AppArgs, setting: TripoSgSetting) -> String {
    match setting {
        TripoSgSetting::Steps => args.num_steps.to_string(),
        TripoSgSetting::Tokens => format_grouped_usize(args.num_tokens),
        TripoSgSetting::Guidance => format!("{:.1}", args.guidance_scale),
        TripoSgSetting::TargetFaces => args
            .target_faces
            .map(format_grouped_usize)
            .unwrap_or_else(|| "disabled".to_string()),
    }
}

fn trellis_setting_value_text(args: &AppArgs, setting: TrellisSetting) -> String {
    match setting {
        TrellisSetting::Resolution => trellis_resolution_text(args.trellis_quality).to_string(),
        TrellisSetting::Pbr => {
            if args.trellis_pbr_enabled {
                "on".to_string()
            } else {
                "off".to_string()
            }
        }
        TrellisSetting::PbrTextureSize => {
            if args.trellis_pbr_enabled {
                args.trellis_pbr_texture_size
                    .map(format_grouped_usize)
                    .unwrap_or_else(|| "runtime".to_string())
            } else {
                "disabled".to_string()
            }
        }
        TrellisSetting::TargetFaces => args
            .trellis_target_faces
            .map(format_grouped_usize)
            .unwrap_or_else(|| "disabled".to_string()),
        TrellisSetting::MaxSparseCoords => args
            .trellis_max_sparse_coords
            .map(format_grouped_usize)
            .unwrap_or_else(|| "uncapped".to_string()),
    }
}

fn scene_setting_value_text(settings: &ScenePipelineUiSettings, setting: SceneSetting) -> String {
    match setting {
        SceneSetting::GroundCalibration => settings.ground_calibration.label().to_string(),
        SceneSetting::InstanceGeneration => settings.instance_generation.label().to_string(),
        SceneSetting::TablePoseRefinement => settings.table_pose_refinement.label().to_string(),
        SceneSetting::CandidateCount => settings.candidate_count.to_string(),
        SceneSetting::FeedbackIterations => settings.feedback_iterations.to_string(),
        SceneSetting::PbrTextureSize => {
            if settings.pbr_enabled {
                format_grouped_usize(settings.pbr_texture_size)
            } else {
                "disabled".to_string()
            }
        }
        SceneSetting::TargetFaces => format_grouped_usize(settings.target_faces),
    }
}

fn scene_toggle_value_text(
    settings: &ScenePipelineUiSettings,
    setting: SceneToggleSetting,
) -> String {
    let enabled = match setting {
        SceneToggleSetting::Pbr => settings.pbr_enabled,
        SceneToggleSetting::CatalogReuse => settings.allow_catalog_reuse,
        SceneToggleSetting::LiftAssets => settings.lift_assets,
        SceneToggleSetting::LocateAnything => settings.locate_anything_enabled,
        SceneToggleSetting::Depth => settings.depth_enabled,
        SceneToggleSetting::Segmentation => settings.segmentation_enabled,
        SceneToggleSetting::PoseFit => settings.pose_fit_enabled,
        SceneToggleSetting::Feedback => settings.feedback_enabled,
        SceneToggleSetting::WriteArtifacts => settings.write_artifacts,
        SceneToggleSetting::PromoteToCatalog => settings.promote_to_catalog,
    };
    if enabled {
        "on".to_string()
    } else {
        "off".to_string()
    }
}

fn pipeline_has_settings(model: SynthesisModel) -> bool {
    matches!(
        model,
        SynthesisModel::Triposg | SynthesisModel::Trellis | SynthesisModel::Triposplat
    )
}

fn active_settings_pipeline(args: Option<&AppArgs>) -> Option<SynthesisModel> {
    args.and_then(|args| args.synthesis_models.first().copied())
        .filter(|model| pipeline_has_settings(*model))
}

fn active_pipeline_choice(
    catalog: &CatalogState,
    args: Option<&AppArgs>,
    scene_settings: Option<&ScenePipelineUiSettings>,
) -> Option<CatalogPipelineChoice> {
    active_pipeline_choice_for_mode(catalog.active_mode(), args, scene_settings)
}

fn active_pipeline_choice_for_mode(
    mode: CatalogMode,
    args: Option<&AppArgs>,
    scene_settings: Option<&ScenePipelineUiSettings>,
) -> Option<CatalogPipelineChoice> {
    match mode {
        CatalogMode::Object => args
            .and_then(|args| args.synthesis_models.first().copied())
            .map(CatalogPipelineChoice::Object),
        CatalogMode::Scene => Some(CatalogPipelineChoice::Scene(
            scene_settings
                .map(|settings| settings.pipeline)
                .unwrap_or(ScenePipelineKind::Explicit),
        )),
    }
}

fn pipeline_settings_enabled(catalog: &CatalogState, args: Option<&AppArgs>) -> bool {
    match catalog.active_mode() {
        CatalogMode::Object => active_settings_pipeline(args).is_some(),
        CatalogMode::Scene => true,
    }
}

fn ellipsize_text(value: &str, max_chars: usize) -> String {
    let char_count = value.chars().count();
    if char_count <= max_chars {
        return value.to_string();
    }
    if max_chars <= 3 {
        return "...".chars().take(max_chars).collect();
    }
    let keep = max_chars - 3;
    let mut out: String = value.chars().take(keep).collect();
    out.push_str("...");
    out
}

fn format_grouped_usize(value: usize) -> String {
    let raw = value.to_string();
    let mut out = String::with_capacity(raw.len() + raw.len().saturating_sub(1) / 3);
    let first_group_len = raw.len() % 3;
    for (index, ch) in raw.chars().enumerate() {
        if index > 0
            && (index == first_group_len
                || (index > first_group_len && (index - first_group_len).is_multiple_of(3)))
        {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

fn spawn_preview_scene(
    commands: &mut Commands,
    images: &mut Assets<Image>,
    asset: PreviewAsset,
    layer_index: usize,
    fit: PreviewFit,
) -> PreviewScene {
    let layer = RenderLayers::layer(layer_index);
    let size = Extent3d {
        width: PREVIEW_SIZE,
        height: PREVIEW_SIZE,
        depth_or_array_layers: 1,
    };
    let mut image = Image::new_fill(
        size,
        TextureDimension::D2,
        &[14, 16, 20, 255],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    image.texture_descriptor.usage |= TextureUsages::RENDER_ATTACHMENT;
    let image_handle = images.add(image);

    let half_fov = PREVIEW_CAMERA_FOV * 0.5;
    let camera_distance =
        (fit.radius / half_fov.tan()).max(fit.radius + 0.35) + PREVIEW_CAMERA_MARGIN;

    let mut asset_entities = Vec::new();
    match asset {
        PreviewAsset::Mesh { mesh, material } => {
            asset_entities.push(
                commands
                    .spawn((
                        Pickable::IGNORE,
                        Mesh3d(mesh),
                        MeshMaterial3d(material),
                        Transform {
                            translation: fit.mesh_translation,
                            scale: Vec3::splat(fit.mesh_scale),
                            ..default()
                        },
                        layer.clone(),
                        ThumbnailSpin,
                    ))
                    .id(),
            );
        }
        PreviewAsset::GaussianSplat { cloud } => {
            asset_entities.push(
                commands
                    .spawn((
                        Pickable::IGNORE,
                        PlanarGaussian3dHandle(cloud),
                        triposplat_preview_cloud_settings(),
                        Transform {
                            translation: fit.mesh_translation,
                            scale: Vec3::splat(fit.mesh_scale),
                            ..default()
                        },
                        layer.clone(),
                        ThumbnailSpin,
                    ))
                    .id(),
            );
        }
        PreviewAsset::Scene { items } => {
            for item in items {
                let mut transform = item.transform;
                transform.translation =
                    fit.mesh_translation + transform.translation * fit.mesh_scale;
                transform.scale *= fit.mesh_scale;
                match item.asset {
                    CatalogSpawnAsset::Mesh { mesh, material } => {
                        asset_entities.push(
                            commands
                                .spawn((
                                    Pickable::IGNORE,
                                    Mesh3d(mesh),
                                    MeshMaterial3d(material),
                                    transform,
                                    layer.clone(),
                                ))
                                .id(),
                        );
                    }
                    CatalogSpawnAsset::GaussianSplat { cloud } => {
                        asset_entities.push(
                            commands
                                .spawn((
                                    Pickable::IGNORE,
                                    PlanarGaussian3dHandle(cloud),
                                    triposplat_preview_cloud_settings(),
                                    transform,
                                    layer.clone(),
                                ))
                                .id(),
                        );
                    }
                }
            }
        }
    };
    let light_entities = vec![
        commands
            .spawn((
                DirectionalLight {
                    color: Color::srgb(1.0, 0.98, 0.95),
                    illuminance: 18_000.0,
                    shadow_maps_enabled: false,
                    ..default()
                },
                Transform::from_xyz(2.8, 4.0, 3.4).looking_at(Vec3::ZERO, Vec3::Y),
                layer.clone(),
            ))
            .id(),
        commands
            .spawn((
                PointLight {
                    color: Color::srgb(0.76, 0.86, 1.0),
                    intensity: 12_000.0,
                    range: 7.0,
                    radius: 0.35,
                    shadow_maps_enabled: false,
                    ..default()
                },
                Transform::from_xyz(-2.5, 2.6, 2.2),
                layer.clone(),
            ))
            .id(),
    ];

    let camera_entity = commands
        .spawn((
            Camera3d::default(),
            Camera {
                order: 2,
                output_mode: CameraOutputMode::Write {
                    blend_state: None,
                    clear_color: ClearColorConfig::Default,
                },
                ..default()
            },
            RenderTarget::Image(image_handle.clone().into()),
            Projection::Perspective(PerspectiveProjection {
                fov: PREVIEW_CAMERA_FOV,
                near: 0.01,
                ..default()
            }),
            Transform::from_translation(Vec3::new(0.0, fit.radius * 0.35, camera_distance))
                .looking_at(Vec3::ZERO, Vec3::Y),
            GaussianCamera::default(),
            layer.clone(),
        ))
        .id();

    PreviewScene {
        image: image_handle,
        asset_entities,
        camera_entity,
        light_entities,
        layer_index,
    }
}

fn triposplat_preview_cloud_settings() -> CloudSettings {
    CloudSettings {
        sort_mode: SortMode::Std,
        color_space: GaussianColorSpace::SrgbRec709Display,
        ..default()
    }
}

fn preview_fit_for_scene_items(
    items: &[CatalogScenePreviewItem],
    meshes: &Assets<BevyMesh>,
    gaussian_clouds: &Assets<PlanarGaussian3d>,
) -> PreviewFit {
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    let mut any = false;
    for item in items {
        match &item.asset {
            CatalogSpawnAsset::Mesh { mesh, .. } => {
                if let Some(mesh) = meshes.get(mesh) {
                    accumulate_mesh_bounds(mesh, item.transform, &mut min, &mut max, &mut any);
                }
            }
            CatalogSpawnAsset::GaussianSplat { cloud } => {
                if let Some(cloud) = gaussian_clouds.get(cloud) {
                    accumulate_gaussian_bounds(cloud, item.transform, &mut min, &mut max, &mut any);
                }
            }
        }
    }
    if !any || !min.is_finite() || !max.is_finite() {
        return PreviewFit::fallback();
    }
    preview_fit_from_bounds(min, max)
}

fn preview_fit_for_mesh(mesh: &BevyMesh) -> PreviewFit {
    let Some(positions) = mesh
        .attribute(BevyMesh::ATTRIBUTE_POSITION)
        .and_then(|attribute| attribute.as_float3())
    else {
        return PreviewFit::fallback();
    };
    if positions.is_empty() {
        return PreviewFit::fallback();
    }

    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for position in positions {
        let point = Vec3::new(position[0], position[1], position[2]);
        min = min.min(point);
        max = max.max(point);
    }

    preview_fit_from_bounds(min, max)
}

fn preview_fit_from_bounds(min: Vec3, max: Vec3) -> PreviewFit {
    if !min.is_finite() || !max.is_finite() {
        return PreviewFit::fallback();
    }

    let center = (min + max) * 0.5;
    let half_extents = (max - min) * 0.5;
    let raw_radius = half_extents.length();
    if !raw_radius.is_finite() || raw_radius <= 0.000_1 {
        return PreviewFit::fallback();
    }

    let mesh_scale = PREVIEW_TARGET_RADIUS / raw_radius;
    if !mesh_scale.is_finite() || mesh_scale <= 0.000_1 {
        return PreviewFit::fallback();
    }

    PreviewFit {
        mesh_translation: -center * mesh_scale,
        mesh_scale,
        radius: (raw_radius * mesh_scale).max(0.05),
    }
}

fn accumulate_mesh_bounds(
    mesh: &BevyMesh,
    transform: Transform,
    min: &mut Vec3,
    max: &mut Vec3,
    any: &mut bool,
) {
    let Some(positions) = mesh
        .attribute(BevyMesh::ATTRIBUTE_POSITION)
        .and_then(|attribute| attribute.as_float3())
    else {
        return;
    };
    let matrix = transform.to_matrix();
    for position in positions {
        let point = matrix.transform_point3(Vec3::new(position[0], position[1], position[2]));
        if point.is_finite() {
            *min = min.min(point);
            *max = max.max(point);
            *any = true;
        }
    }
}

fn accumulate_gaussian_bounds(
    cloud: &PlanarGaussian3d,
    transform: Transform,
    min: &mut Vec3,
    max: &mut Vec3,
    any: &mut bool,
) {
    let matrix = transform.to_matrix();
    for position_visibility in cloud.position_visibility.iter() {
        let point = matrix.transform_point3(Vec3::new(
            position_visibility.position[0],
            position_visibility.position[1],
            position_visibility.position[2],
        ));
        if point.is_finite() {
            *min = min.min(point);
            *max = max.max(point);
            *any = true;
        }
    }
}

fn preview_fit_for_gaussian_cloud(cloud: &PlanarGaussian3d) -> PreviewFit {
    if cloud.position_visibility.is_empty() {
        return PreviewFit::fallback();
    }

    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for position_visibility in cloud.position_visibility.iter() {
        let point = Vec3::new(
            position_visibility.position[0],
            position_visibility.position[1],
            position_visibility.position[2],
        );
        if !point.is_finite() {
            return PreviewFit::fallback();
        }
        min = min.min(point);
        max = max.max(point);
    }

    if !min.is_finite() || !max.is_finite() {
        return PreviewFit::fallback();
    }

    let center = (min + max) * 0.5;
    let half_extents = (max - min) * 0.5;
    let raw_radius = half_extents.length();
    if !raw_radius.is_finite() || raw_radius <= 0.000_1 {
        return PreviewFit::fallback();
    }

    let mesh_scale = PREVIEW_TARGET_RADIUS / raw_radius;
    if !mesh_scale.is_finite() || mesh_scale <= 0.000_1 {
        return PreviewFit::fallback();
    }

    PreviewFit {
        mesh_translation: -center * mesh_scale,
        mesh_scale,
        radius: (raw_radius * mesh_scale).max(0.05),
    }
}

pub fn preview_light_layers() -> RenderLayers {
    RenderLayers::layer(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_gaussian_splatting::{Gaussian3d, SphericalHarmonicCoefficients};
    use bevy_synth_runtime::state::InferenceQueue;

    fn ui_test_app(args: Option<AppArgs>) -> App {
        let mut app = App::new();
        if let Some(args) = args {
            app.insert_resource(args);
        }
        app.insert_resource(InferenceQueue::default());
        app.insert_resource(Assets::<Image>::default());
        app.insert_resource(Assets::<BevyMesh>::default());
        app.insert_resource(Assets::<PlanarGaussian3d>::default());
        app.insert_resource(ButtonInput::<MouseButton>::default());
        app.insert_resource(ButtonInput::<KeyCode>::default());
        app.insert_resource(Time::<()>::default());
        app.add_plugins(BurnSynthUiPlugin);
        app
    }

    #[test]
    fn ui_root_is_pass_through_for_world_picking() {
        let mut app = ui_test_app(None);

        app.update();

        let world = app.world_mut();
        let mut query = world.query::<(&UiRootNode, &Pickable)>();
        let pickables: Vec<Pickable> = query
            .iter(world)
            .map(|(_, pickable)| pickable.clone())
            .collect();
        assert_eq!(pickables.len(), 1, "expected exactly one UI root node");
        assert_eq!(pickables[0], Pickable::IGNORE);
    }

    #[test]
    fn pipeline_selector_is_owned_by_settings_modal_for_single_launch_model() {
        let mut args = AppArgs::default();
        args.backend = BackendKind::Wgpu;
        args.synthesis_models = vec![SynthesisModel::Triposplat];
        args.available_synthesis_models = vec![SynthesisModel::Triposplat];
        let mut app = ui_test_app(Some(args));

        app.update();

        let world = app.world_mut();
        assert_eq!(
            world.query::<&PipelineSelectorButton>().iter(world).count(),
            0
        );
        assert_eq!(
            world.query::<&PipelineOptionButton>().iter(world).count(),
            0
        );
        assert_eq!(
            world.resource::<AvailablePipelines>().object_models,
            vec![SynthesisModel::Triposplat]
        );

        app.world_mut().resource_mut::<SettingsModalState>().open = true;
        app.update();
        app.update();

        let world = app.world_mut();
        assert_eq!(
            world.query::<&PipelineOptionButton>().iter(world).count(),
            1
        );
    }

    #[test]
    fn scene_source_modal_has_image_and_stats_tabs() {
        let mut app = ui_test_app(None);
        let metrics = CachedSceneMetrics {
            ok: Some(true),
            elapsed_ms: Some(42_000),
            object_count: Some(3),
            asset_count: Some(2),
            placement_count: Some(3),
            feedback_accepted: Some(true),
            feedback_iteration: Some(2),
            failed_stage: None,
            category_breakdown: vec![
                CachedSceneCategoryMetric {
                    label: "chair".to_string(),
                    object_count: Some(2),
                    detection_count: Some(4),
                    asset_count: Some(1),
                    placement_count: Some(2),
                },
                CachedSceneCategoryMetric {
                    label: "table".to_string(),
                    object_count: Some(1),
                    detection_count: Some(1),
                    asset_count: Some(1),
                    placement_count: Some(1),
                },
            ],
        };
        app.world_mut()
            .resource_mut::<CatalogState>()
            .add_ready_scene(
                42,
                "meeting scene".to_string(),
                Some("scene_cache_key".to_string()),
                Vec::new(),
                Some("/tmp/source_scene.jpg".to_string()),
                None,
                Some("explicit".to_string()),
                Some(metrics),
                Some("tmp/runs/demo_scene".to_string()),
            );
        app.world_mut()
            .resource_mut::<CatalogSourceImageModalState>()
            .entry_id = Some(42);

        app.update();
        app.update();

        {
            let world = app.world_mut();
            assert_eq!(
                world
                    .query::<&CatalogSourceImageTabButton>()
                    .iter(world)
                    .count(),
                2
            );
            assert_eq!(
                world
                    .query::<&CatalogSourceImageTabPanel>()
                    .iter(world)
                    .count(),
                2
            );
            let mut texts = world.query::<&Text>();
            let values: Vec<_> = texts.iter(world).map(|text| text.0.clone()).collect();
            assert!(values.iter().any(|text| text == "categories"));
            assert!(values.iter().any(|text| {
                text.contains("chair | 2 planned / 4 detected / 1 assets / 2 placed")
            }));
        }

        let stats_button = {
            let world = app.world_mut();
            let mut query = world.query::<(Entity, &CatalogSourceImageTabButton)>();
            query
                .iter(world)
                .find_map(|(entity, button)| {
                    (button.tab == CatalogSourceImageTab::Stats).then_some(entity)
                })
                .expect("stats tab button")
        };
        app.world_mut()
            .entity_mut(stats_button)
            .insert(Interaction::Pressed);
        app.update();
        app.update();

        let world = app.world_mut();
        let mut panels = world.query::<(&CatalogSourceImageTabPanel, &Node)>();
        let visible_tabs: Vec<_> = panels
            .iter(world)
            .filter_map(|(panel, node)| (node.display != Display::None).then_some(panel.tab))
            .collect();
        assert_eq!(visible_tabs, vec![CatalogSourceImageTab::Stats]);
    }

    #[test]
    fn settings_modal_spawns_enabled_model_options() {
        let mut args = AppArgs::default();
        args.backend = BackendKind::Wgpu;
        args.synthesis_models = vec![SynthesisModel::Triposg];
        args.available_synthesis_models = vec![
            SynthesisModel::Triposg,
            SynthesisModel::Trellis,
            SynthesisModel::Triposplat,
        ];
        let mut app = ui_test_app(Some(args));

        app.update();
        app.world_mut().resource_mut::<SettingsModalState>().open = true;
        app.update();
        app.update();

        let world = app.world_mut();
        let mut query = world.query::<&PipelineOptionButton>();
        let models: Vec<_> = query
            .iter(world)
            .filter_map(|button| match button.choice {
                CatalogPipelineChoice::Object(model) => Some(model),
                CatalogPipelineChoice::Scene(_) => None,
            })
            .collect();
        assert_eq!(
            models,
            vec![
                SynthesisModel::Triposg,
                SynthesisModel::Trellis,
                SynthesisModel::Triposplat
            ]
        );
    }

    #[test]
    fn scene_settings_model_buttons_update_scene_mesh_asset_model() {
        let mut args = AppArgs::default();
        args.backend = BackendKind::Wgpu;
        args.synthesis_models = vec![SynthesisModel::Triposg];
        args.available_synthesis_models = vec![
            SynthesisModel::Triposg,
            SynthesisModel::Trellis,
            SynthesisModel::Triposplat,
        ];
        let mut app = ui_test_app(Some(args));
        app.world_mut()
            .resource_mut::<CatalogState>()
            .set_active_mode(CatalogMode::Scene);
        app.world_mut().resource_mut::<SettingsModalState>().open = true;
        app.update();
        app.update();

        {
            let world = app.world_mut();
            let mut query = world.query::<&PipelineOptionButton>();
            let models: Vec<_> = query
                .iter(world)
                .filter_map(|button| match button.choice {
                    CatalogPipelineChoice::Object(model) => Some(model),
                    CatalogPipelineChoice::Scene(_) => None,
                })
                .collect();
            assert_eq!(
                models,
                vec![SynthesisModel::Triposg, SynthesisModel::Trellis]
            );
        }

        let triposg_button = {
            let world = app.world_mut();
            let mut query = world.query::<(Entity, &PipelineOptionButton)>();
            query
                .iter(world)
                .find_map(|(entity, button)| {
                    matches!(
                        button.choice,
                        CatalogPipelineChoice::Object(SynthesisModel::Triposg)
                    )
                    .then_some(entity)
                })
                .expect("TripoSG scene image-to-3d button")
        };
        app.world_mut()
            .entity_mut(triposg_button)
            .insert(Interaction::Pressed);
        app.update();

        assert_eq!(
            app.world()
                .resource::<ScenePipelineUiSettings>()
                .image_to_3d_model,
            SynthesisModel::Triposg
        );
        assert_eq!(
            app.world().resource::<AppArgs>().synthesis_models,
            vec![SynthesisModel::Triposg],
            "scene image-to-3d selection must not mutate the object catalog pipeline"
        );
    }

    #[test]
    fn scene_optional_stage_toggle_labels_track_settings() {
        let mut settings = ScenePipelineUiSettings::default();
        assert_eq!(
            scene_toggle_value_text(&settings, SceneToggleSetting::LocateAnything),
            "on"
        );
        assert_eq!(
            scene_toggle_value_text(&settings, SceneToggleSetting::Depth),
            "on"
        );
        assert_eq!(
            scene_toggle_value_text(&settings, SceneToggleSetting::Segmentation),
            "on"
        );
        assert_eq!(
            scene_toggle_value_text(&settings, SceneToggleSetting::PoseFit),
            "on"
        );

        settings.locate_anything_enabled = false;
        settings.depth_enabled = false;
        settings.segmentation_enabled = false;
        settings.pose_fit_enabled = false;

        assert_eq!(
            scene_toggle_value_text(&settings, SceneToggleSetting::LocateAnything),
            "off"
        );
        assert_eq!(
            scene_toggle_value_text(&settings, SceneToggleSetting::Depth),
            "off"
        );
        assert_eq!(
            scene_toggle_value_text(&settings, SceneToggleSetting::Segmentation),
            "off"
        );
        assert_eq!(
            scene_toggle_value_text(&settings, SceneToggleSetting::PoseFit),
            "off"
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn processing_artifact_classifier_prioritizes_scene_intermediates() {
        assert_eq!(
            visual_artifact_kind(std::path::Path::new("tmp/runs/demo/detections_overlay.png")),
            Some(ProcessingArtifactVisualKind::Locate)
        );
        assert_eq!(
            visual_artifact_kind(std::path::Path::new("tmp/runs/demo/depth_map.png")),
            Some(ProcessingArtifactVisualKind::Depth)
        );
        assert_eq!(
            visual_artifact_kind(std::path::Path::new(
                "tmp/runs/demo/objects/crops/chair_0.jpg"
            )),
            Some(ProcessingArtifactVisualKind::Crop)
        );
        assert_eq!(
            visual_artifact_kind(std::path::Path::new(
                "tmp/runs/demo/canonical_pose/chair_selection.png"
            )),
            Some(ProcessingArtifactVisualKind::Canonical)
        );
        assert_eq!(
            visual_artifact_kind(std::path::Path::new(
                "tmp/runs/demo/iterations/iter_03/screenshot.png"
            )),
            Some(ProcessingArtifactVisualKind::Feedback)
        );
        assert_eq!(
            visual_artifact_kind(std::path::Path::new(
                "tmp/runs/demo/iterations/iter_03/rotation_candidates/chair/current_isolated_full_frame.png"
            )),
            Some(ProcessingArtifactVisualKind::IsolatedRender)
        );
        assert_eq!(
            visual_artifact_kind(std::path::Path::new(
                "tmp/runs/demo/iterations/iter_03/rotation_candidates/chair/candidate_00_yaw_pos0_screenshot.png"
            )),
            Some(ProcessingArtifactVisualKind::IsolatedRender)
        );
        assert_eq!(
            artifact_kind_label("tmp/runs/demo/iterations/iter_03/scene.bsn"),
            "bsn  "
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn processing_artifacts_sort_latest_generation_first() {
        let root = temp_visual_artifact_dir("latest_sort");
        let older = root.join("iterations/iter_00/screenshot.png");
        let newer = root.join("iterations/iter_03/screenshot.png");
        std::fs::create_dir_all(older.parent().expect("older parent")).expect("older dir");
        std::fs::create_dir_all(newer.parent().expect("newer parent")).expect("newer dir");
        std::fs::write(&older, &[] as &[u8]).expect("older artifact");
        std::thread::sleep(std::time::Duration::from_millis(10));
        std::fs::write(&newer, &[] as &[u8]).expect("newer artifact");

        let mut state = SceneProcessingState::default();
        state
            .recent_artifacts
            .push_front(root.display().to_string());
        let discovered = discover_processing_visual_artifacts(&state);

        assert_eq!(
            discovered.first().map(|(path, _)| path),
            Some(&newer),
            "latest pipeline artifact should be shown first"
        );
        std::fs::remove_dir_all(root).expect("remove temp artifacts");
    }

    #[test]
    fn developer_visual_tab_renders_artifact_previews() {
        let mut args = AppArgs::default();
        args.synthesis_models = vec![SynthesisModel::Triposg];
        let mut app = ui_test_app(Some(args));
        app.world_mut().resource_mut::<SettingsModalState>().open = true;
        app.world_mut().resource_mut::<SettingsModalState>().tab = SettingsModalTab::Developer;
        app.world_mut().resource_mut::<DeveloperPanelState>().tab = DeveloperPanelTab::Visuals;
        let artifact_path = std::env::temp_dir().join(format!(
            "bevy_synth_ui_{}_detections_overlay.png",
            std::process::id()
        ));
        image::save_buffer_with_format(
            &artifact_path,
            &[
                255, 255, 255, 255, 64, 64, 64, 255, 64, 64, 64, 255, 255, 255, 255, 255,
            ],
            2,
            2,
            image::ColorType::Rgba8,
            image::ImageFormat::Png,
        )
        .expect("write preview artifact");
        {
            let mut state = app.world_mut().resource_mut::<SceneProcessingState>();
            state
                .recent_artifacts
                .push_front(artifact_path.display().to_string());
        }

        app.update();
        app.update();

        let world = app.world_mut();
        let mut grids = world.query::<(&SettingsDeveloperVisualGrid, &Children)>();
        let child_counts = grids
            .iter(world)
            .map(|(_, children)| children.len())
            .collect::<Vec<_>>();
        assert_eq!(child_counts, vec![1]);
        let mut texts = world.query::<&Text>();
        let values = texts
            .iter(world)
            .map(|text| text.0.clone())
            .collect::<Vec<_>>();
        assert!(values.iter().any(|text| text == "locate"));
        assert!(
            values
                .iter()
                .any(|text| text.contains("detections_overlay.png"))
        );
        std::fs::remove_file(artifact_path).expect("remove preview artifact");
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn developer_visual_tab_paginates_artifact_previews() {
        let mut args = AppArgs::default();
        args.synthesis_models = vec![SynthesisModel::Triposg];
        let mut app = ui_test_app(Some(args));
        app.world_mut().resource_mut::<SettingsModalState>().open = true;
        app.world_mut().resource_mut::<SettingsModalState>().tab = SettingsModalTab::Developer;
        app.world_mut().resource_mut::<DeveloperPanelState>().tab = DeveloperPanelTab::Visuals;
        let root = temp_visual_artifact_dir("pagination");
        std::fs::create_dir_all(&root).expect("preview dir");
        for index in 0..(DEVELOPER_VISUAL_ROWS + 1) {
            write_preview_png(&root.join(format!("iter_{index:02}_detections_overlay.png")));
        }
        {
            let mut state = app.world_mut().resource_mut::<SceneProcessingState>();
            state
                .recent_artifacts
                .push_front(root.display().to_string());
        }

        app.update();
        app.update();

        {
            let cache = app.world().resource::<ProcessingArtifactPreviewCache>();
            assert_eq!(cache.total_count, DEVELOPER_VISUAL_ROWS + 1);
            assert_eq!(cache.page, 0);
            assert_eq!(cache.page_count, 2);
            assert_eq!(cache.previews.len(), DEVELOPER_VISUAL_ROWS);
            assert!(
                cache.previews[0]
                    .path
                    .contains(&format!("iter_{:02}", DEVELOPER_VISUAL_ROWS)),
                "first page should be latest-first"
            );
        }
        assert_visual_grid_child_count(&mut app, DEVELOPER_VISUAL_ROWS);

        let next_button = {
            let world = app.world_mut();
            let mut query = world.query::<(Entity, &SettingsDeveloperVisualPageButton)>();
            query
                .iter(world)
                .find_map(|(entity, button)| {
                    (button.direction == DeveloperVisualPageDirection::Next).then_some(entity)
                })
                .expect("next visual page button")
        };
        app.world_mut()
            .entity_mut(next_button)
            .insert(Interaction::Pressed);
        app.update();
        app.update();

        {
            let cache = app.world().resource::<ProcessingArtifactPreviewCache>();
            assert_eq!(cache.page, 1);
            assert_eq!(cache.previews.len(), 1);
        }
        assert_visual_grid_child_count(&mut app, 1);
        let world = app.world_mut();
        let mut pager_texts =
            world.query_filtered::<&Text, With<SettingsDeveloperVisualPagerText>>();
        assert!(
            pager_texts
                .iter(world)
                .any(|text| text.0.contains("page 2/2")),
            "pager text should report the active page"
        );
        std::fs::remove_dir_all(root).expect("remove preview artifacts");
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn temp_visual_artifact_dir(label: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "bevy_synth_ui_{label}_{}_{}",
            std::process::id(),
            nanos
        ))
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn write_preview_png(path: &std::path::Path) {
        image::save_buffer_with_format(
            path,
            &[
                255, 255, 255, 255, 64, 64, 64, 255, 64, 64, 64, 255, 255, 255, 255, 255,
            ],
            2,
            2,
            image::ColorType::Rgba8,
            image::ImageFormat::Png,
        )
        .expect("write preview artifact");
    }

    fn assert_visual_grid_child_count(app: &mut App, expected: usize) {
        let world = app.world_mut();
        let mut grids = world.query::<(&SettingsDeveloperVisualGrid, &Children)>();
        let child_counts = grids
            .iter(world)
            .map(|(_, children)| children.len())
            .collect::<Vec<_>>();
        assert_eq!(child_counts, vec![expected]);
    }

    #[test]
    fn unavailable_launch_models_are_not_selectable() {
        let available = AvailablePipelines {
            object_models: vec![SynthesisModel::Triposplat],
            scene_pipelines: vec![ScenePipelineKind::Explicit],
        };
        assert!(pipeline_available(
            Some(&available),
            CatalogPipelineChoice::Object(SynthesisModel::Triposplat)
        ));
        assert!(!pipeline_available(
            Some(&available),
            CatalogPipelineChoice::Object(SynthesisModel::Triposg)
        ));
        assert!(pipeline_available(
            Some(&available),
            CatalogPipelineChoice::Scene(ScenePipelineKind::Explicit)
        ));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn triposplat_pipeline_is_supported_on_native_wgpu() {
        let mut args = AppArgs::default();
        args.backend = BackendKind::Wgpu;
        assert!(pipeline_supported(
            Some(&args),
            CatalogPipelineChoice::Object(SynthesisModel::Triposplat)
        ));
        assert!(pipeline_supported(
            Some(&args),
            CatalogPipelineChoice::Object(SynthesisModel::Trellis)
        ));
    }

    #[test]
    fn ready_cube_detection_allows_cached_cube_entries() {
        let mut catalog = CatalogState::default();
        assert!(!catalog.has_ready_cube_entry());

        catalog.add_ready(
            1,
            "cube".to_string(),
            Handle::default(),
            Handle::default(),
            Some("builtin/cube".to_string()),
            Some("builtin-cube-cache-key".to_string()),
        );
        assert!(catalog.has_ready_cube_entry());
    }

    #[test]
    fn catalog_labels_are_ellipsized_to_fixed_width() {
        assert_eq!(ellipsize_text("short", 12), "short");
        assert_eq!(
            ellipsize_text("very-long-catalog-entry-name", 13),
            "very-long-..."
        );
        assert_eq!(ellipsize_text("abcdef", 2), "..");
    }

    #[test]
    fn splat_catalog_entry_creates_gaussian_preview_scene() {
        let mut app = App::new();
        app.insert_resource(CatalogState::default());
        app.insert_resource(Assets::<Image>::default());
        app.insert_resource(Assets::<BevyMesh>::default());
        app.insert_resource(Assets::<PlanarGaussian3d>::default());
        app.add_systems(Update, sync_catalog_previews);

        let cloud = PlanarGaussian3d::from(vec![
            Gaussian3d {
                position_visibility: [-0.25, 0.0, 0.0, 1.0].into(),
                spherical_harmonic: SphericalHarmonicCoefficients::default(),
                rotation: [1.0, 0.0, 0.0, 0.0].into(),
                scale_opacity: [0.04, 0.04, 0.04, 0.8].into(),
            },
            Gaussian3d {
                position_visibility: [0.25, 0.2, 0.0, 1.0].into(),
                spherical_harmonic: SphericalHarmonicCoefficients::default(),
                rotation: [1.0, 0.0, 0.0, 0.0].into(),
                scale_opacity: [0.04, 0.04, 0.04, 0.8].into(),
            },
        ]);
        let cloud_handle = app
            .world_mut()
            .resource_mut::<Assets<PlanarGaussian3d>>()
            .add(cloud);
        app.world_mut()
            .resource_mut::<CatalogState>()
            .add_ready_gaussian_splat(
                7,
                "splat".to_string(),
                cloud_handle,
                Some("input.png".to_string()),
                Some("cache-key".to_string()),
            );

        app.update();

        let world = app.world_mut();
        let (has_preview, has_gaussian, has_mesh, has_material) = {
            let catalog = world.resource::<CatalogState>();
            let entry = catalog.entry(7).expect("splat catalog entry");
            (
                entry.preview.is_some(),
                entry.gaussian.is_some(),
                entry.mesh.is_some(),
                entry.material.is_some(),
            )
        };
        assert!(has_preview, "splat entry should get a preview");
        assert_eq!(
            world
                .resource::<CatalogState>()
                .entry(7)
                .and_then(|entry| entry.preview.as_ref())
                .map(|preview| preview.light_entities.len()),
            Some(2),
            "preview scene should own isolated light entities"
        );
        assert!(has_gaussian);
        assert!(!has_mesh);
        assert!(!has_material);
        assert_eq!(
            world.query::<&PlanarGaussian3dHandle>().iter(world).count(),
            1
        );
        let mut settings = world.query::<&CloudSettings>();
        let settings = settings
            .single(world)
            .expect("one Gaussian preview settings");
        assert_eq!(settings.sort_mode, SortMode::Std);
        assert_eq!(settings.color_space, GaussianColorSpace::SrgbRec709Display);
        assert_eq!(world.query::<&GaussianCamera>().iter(world).count(), 1);
        assert_eq!(world.query::<&DirectionalLight>().iter(world).count(), 1);
        assert_eq!(world.query::<&PointLight>().iter(world).count(), 1);
    }

    #[test]
    fn settings_modal_opens_for_all_pipeline_settings() {
        let mut triposg_args = AppArgs::default();
        triposg_args.synthesis_models = vec![
            SynthesisModel::Triposg,
            SynthesisModel::Trellis,
            SynthesisModel::Triposplat,
        ];
        let mut app = ui_test_app(Some(triposg_args));

        app.world_mut().resource_mut::<SettingsModalState>().open = true;
        app.update();
        app.update();

        {
            let world = app.world_mut();
            assert!(world.resource::<SettingsModalState>().open);
            assert_eq!(
                world.query::<&SettingsModalRoot>().iter(world).count(),
                1,
                "TripoSG settings should open when TripoSG is active"
            );
            assert_eq!(
                world
                    .query::<&TripoSgSettingValueLabel>()
                    .iter(world)
                    .count(),
                4
            );
            assert_eq!(
                world
                    .query::<&TripoSplatProfileButton>()
                    .iter(world)
                    .count(),
                0
            );
        }

        app.world_mut().resource_mut::<AppArgs>().synthesis_models =
            vec![SynthesisModel::Triposplat, SynthesisModel::Triposg];
        app.world_mut().resource_mut::<SettingsModalState>().open = true;
        app.update();
        app.update();

        {
            let world = app.world_mut();
            assert!(world.resource::<SettingsModalState>().open);
            assert_eq!(
                world.query::<&SettingsModalRoot>().iter(world).count(),
                1,
                "TripoSplat settings should open when TripoSplat is active"
            );
            assert_eq!(
                world
                    .query::<&TripoSplatSettingValueLabel>()
                    .iter(world)
                    .count(),
                3
            );
            assert_eq!(
                world
                    .query::<&TripoSplatProfileButton>()
                    .iter(world)
                    .count(),
                3
            );
            assert_eq!(
                world
                    .query::<&TripoSgSettingValueLabel>()
                    .iter(world)
                    .count(),
                0
            );
        }

        app.world_mut().resource_mut::<AppArgs>().synthesis_models =
            vec![SynthesisModel::Trellis, SynthesisModel::Triposg];
        app.world_mut().resource_mut::<SettingsModalState>().open = true;
        app.update();
        app.update();

        {
            let world = app.world_mut();
            assert!(world.resource::<SettingsModalState>().open);
            assert_eq!(
                world.query::<&SettingsModalRoot>().iter(world).count(),
                1,
                "Trellis.2 settings should open when Trellis.2 is active"
            );
            assert_eq!(
                world.query::<&TrellisQualityButton>().iter(world).count(),
                3
            );
            assert_eq!(
                world
                    .query::<&TrellisSettingValueLabel>()
                    .iter(world)
                    .count(),
                5
            );
        }
    }

    #[test]
    fn settings_modal_rebuilds_or_closes_when_pipeline_changes() {
        let mut args = AppArgs::default();
        args.synthesis_models = vec![SynthesisModel::Triposplat, SynthesisModel::Triposg];
        let mut app = ui_test_app(Some(args));

        app.world_mut().resource_mut::<SettingsModalState>().open = true;
        app.update();
        app.update();
        {
            let world = app.world_mut();
            assert_eq!(world.query::<&SettingsModalRoot>().iter(world).count(), 1);
            assert_eq!(
                world
                    .query::<&TripoSplatSettingValueLabel>()
                    .iter(world)
                    .count(),
                3
            );
        }

        app.world_mut().resource_mut::<AppArgs>().synthesis_models =
            vec![SynthesisModel::Triposg, SynthesisModel::Triposplat];
        app.update();
        app.update();

        {
            let world = app.world_mut();
            assert!(world.resource::<SettingsModalState>().open);
            assert_eq!(world.query::<&SettingsModalRoot>().iter(world).count(), 1);
            assert_eq!(
                world
                    .query::<&TripoSgSettingValueLabel>()
                    .iter(world)
                    .count(),
                4
            );
            assert_eq!(
                world
                    .query::<&TripoSplatSettingValueLabel>()
                    .iter(world)
                    .count(),
                0
            );
        }

        app.world_mut().resource_mut::<AppArgs>().synthesis_models = vec![SynthesisModel::Trellis];
        app.update();
        app.update();

        let world = app.world_mut();
        assert!(world.resource::<SettingsModalState>().open);
        assert_eq!(world.query::<&SettingsModalRoot>().iter(world).count(), 1);
        assert_eq!(
            world
                .query::<&TrellisSettingValueLabel>()
                .iter(world)
                .count(),
            5
        );
    }

    #[test]
    fn settings_modal_uses_tabs_for_pipeline_general_physics_and_developer() {
        let mut args = AppArgs::default();
        args.synthesis_models = vec![SynthesisModel::Triposg];
        let mut app = ui_test_app(Some(args));

        app.world_mut().resource_mut::<SettingsModalState>().open = true;
        app.update();
        app.update();

        {
            let world = app.world_mut();
            assert_eq!(world.query::<&SettingsTabButton>().iter(world).count(), 4);
            assert_eq!(world.query::<&SettingsTabPanel>().iter(world).count(), 4);
            let mut panels = world.query::<(&SettingsTabPanel, &Node)>();
            let visible_tabs: Vec<_> = panels
                .iter(world)
                .filter_map(|(panel, node)| (node.display != Display::None).then_some(panel.tab))
                .collect();
            assert_eq!(visible_tabs, vec![SettingsModalTab::Pipeline]);
        }

        app.world_mut().resource_mut::<SettingsModalState>().tab = SettingsModalTab::General;
        app.update();

        let world = app.world_mut();
        let mut panels = world.query::<(&SettingsTabPanel, &Node)>();
        let visible_tabs: Vec<_> = panels
            .iter(world)
            .filter_map(|(panel, node)| (node.display != Display::None).then_some(panel.tab))
            .collect();
        assert_eq!(visible_tabs, vec![SettingsModalTab::General]);
        assert_eq!(
            world.query::<&ViewerAabbModeButton>().iter(world).count(),
            3
        );

        app.world_mut().resource_mut::<SettingsModalState>().tab = SettingsModalTab::Developer;
        app.update();

        let world = app.world_mut();
        let mut panels = world.query::<(&SettingsTabPanel, &Node)>();
        let visible_tabs: Vec<_> = panels
            .iter(world)
            .filter_map(|(panel, node)| (node.display != Display::None).then_some(panel.tab))
            .collect();
        assert_eq!(visible_tabs, vec![SettingsModalTab::Developer]);
        assert_eq!(
            world
                .query::<&SettingsDeveloperEventsText>()
                .iter(world)
                .count(),
            1
        );
    }

    #[test]
    fn scene_settings_modal_splits_long_pipeline_controls() {
        let mut args = AppArgs::default();
        args.synthesis_models = vec![SynthesisModel::Trellis, SynthesisModel::Triposg];
        let mut app = ui_test_app(Some(args));
        app.world_mut()
            .resource_mut::<CatalogState>()
            .set_active_mode(CatalogMode::Scene);
        app.world_mut().resource_mut::<SettingsModalState>().open = true;
        app.update();
        app.update();

        let world = app.world_mut();
        assert_eq!(world.query::<&SettingsTabButton>().iter(world).count(), 7);
        assert_eq!(world.query::<&SettingsTabPanel>().iter(world).count(), 7);
        assert_eq!(world.query::<&SettingsScrollArea>().iter(world).count(), 7);
        assert_eq!(
            settings_tabs_for_pipeline(CatalogPipelineChoice::Scene(ScenePipelineKind::Explicit)),
            vec![
                SettingsModalTab::Pipeline,
                SettingsModalTab::Generation,
                SettingsModalTab::Grounding,
                SettingsModalTab::Debug,
                SettingsModalTab::General,
                SettingsModalTab::Physics,
                SettingsModalTab::Developer,
            ]
        );

        let mut scroll_panels = world.query::<(&SettingsScrollArea, &Node)>();
        for (_, node) in scroll_panels.iter(world) {
            assert_eq!(node.max_height, Val::Vh(SETTINGS_TAB_BODY_MAX_HEIGHT_VH));
            assert_eq!(node.overflow.y, OverflowAxis::Scroll);
        }
    }

    #[test]
    fn worker_status_text_is_compacted_for_menu_bar() {
        let text = compact_worker_status_text(
            "scene progress: images_to_assets - running TRELLIS batch for 2 image(s)",
        );
        assert_eq!(text, "scene progress: images_to_assets");
        assert!(text.len() <= 34);
    }

    #[test]
    fn scene_processing_heartbeat_advances_elapsed_without_new_events() {
        let mut state = SceneProcessingState::default();
        state.begin("source.jpg");
        state.wall_started_at = Some(Instant::now() - Duration::from_millis(2500));
        state.tick();

        assert!(
            state.elapsed_ms >= 2400,
            "active processing elapsed time should advance from wall clock even without worker events"
        );
        assert!(format_developer_current_block(&state).contains("last event:"));
    }

    #[test]
    fn developer_processing_blocks_include_artifacts_and_recent_events() {
        let mut state = SceneProcessingState::default();
        state.begin("scene.png");
        state.push_event(
            "run_001".to_string(),
            SceneProcessingEvent {
                stage: "images_to_assets".to_string(),
                phase: "progress".to_string(),
                execution: "gpu".to_string(),
                message: "running TRELLIS batch for 2 image(s)".to_string(),
                elapsed_ms: 42_000,
                item_index: Some(1),
                item_count: Some(2),
                artifact_path: Some("tmp/runs/run_001/assets".to_string()),
                token_usage: None,
                is_failure: false,
            },
        );

        let event_text = format_developer_event_block(&state);
        assert!(event_text.contains("[gpu] progress / images_to_assets"));
        assert!(event_text.contains("[1/2]"));
        let artifact_text = format_developer_artifact_block(&state);
        assert!(artifact_text.contains("dir"));
        assert!(artifact_text.contains("tmp/runs/run_001/assets"));
    }

    #[test]
    fn viewer_debug_buttons_update_shared_settings() {
        let mut args = AppArgs::default();
        args.synthesis_models = vec![SynthesisModel::Triposg];
        let mut app = ui_test_app(Some(args));

        app.world_mut().resource_mut::<SettingsModalState>().open = true;
        app.update();
        app.update();

        let all_button = {
            let world = app.world_mut();
            let mut query = world.query::<(Entity, &ViewerAabbModeButton)>();
            query
                .iter(world)
                .find_map(|(entity, button)| {
                    (button.mode == ViewerAabbOverlayMode::All).then_some(entity)
                })
                .expect("all AABB button")
        };
        app.world_mut()
            .entity_mut(all_button)
            .insert(Interaction::Pressed);
        app.update();
        assert_eq!(
            app.world().resource::<ViewerDebugSettings>().aabb_overlay,
            ViewerAabbOverlayMode::All
        );

        let tolerance_step = {
            let world = app.world_mut();
            let mut query = world.query::<(Entity, &ViewerDebugStepButton)>();
            query
                .iter(world)
                .find_map(|(entity, button)| {
                    (button.setting == ViewerDebugNumericSetting::ContactTolerance
                        && button.delta > 0.0)
                        .then_some(entity)
                })
                .expect("contact tolerance step button")
        };
        app.world_mut()
            .entity_mut(tolerance_step)
            .insert(Interaction::Pressed);
        app.update();
        assert!(
            app.world()
                .resource::<ViewerDebugSettings>()
                .contact_tolerance
                > 0.02
        );

        let frustum_length_step = {
            let world = app.world_mut();
            let mut query = world.query::<(Entity, &ViewerDebugStepButton)>();
            query
                .iter(world)
                .find_map(|(entity, button)| {
                    (button.setting == ViewerDebugNumericSetting::SceneCameraFrustumLength
                        && button.delta > 0.0)
                        .then_some(entity)
                })
                .expect("frustum length step button")
        };
        app.world_mut()
            .entity_mut(frustum_length_step)
            .insert(Interaction::Pressed);
        app.update();
        assert_eq!(
            app.world()
                .resource::<ViewerDebugSettings>()
                .scene_camera_frustum_length,
            DEFAULT_VIEWER_FRUSTUM_LENGTH + VIEWER_FRUSTUM_LENGTH_STEP
        );

        let depth_cloud_toggle = {
            let world = app.world_mut();
            let mut query = world.query::<(Entity, &ViewerDebugToggleButton)>();
            query
                .iter(world)
                .find_map(|(entity, button)| {
                    (button.setting == ViewerDebugToggleSetting::DepthCloud).then_some(entity)
                })
                .expect("depth cloud debug toggle")
        };
        app.world_mut()
            .entity_mut(depth_cloud_toggle)
            .insert(Interaction::Pressed);
        app.update();
        assert!(
            app.world()
                .resource::<ViewerDebugSettings>()
                .depth_cloud_overlay
        );

        let depth_cap_step = {
            let world = app.world_mut();
            let mut query = world.query::<(Entity, &ViewerDebugStepButton)>();
            query
                .iter(world)
                .find_map(|(entity, button)| {
                    (button.setting == ViewerDebugNumericSetting::DepthCloudMaxGaussians
                        && button.delta > 0.0)
                        .then_some(entity)
                })
                .expect("depth cloud cap step button")
        };
        app.world_mut()
            .entity_mut(depth_cap_step)
            .insert(Interaction::Pressed);
        app.update();
        assert_eq!(
            app.world()
                .resource::<ViewerDebugSettings>()
                .depth_cloud_max_gaussians,
            VIEWER_DEPTH_CLOUD_DEFAULT_GAUSSIANS + VIEWER_DEPTH_CLOUD_GAUSSIAN_STEP
        );
    }

    #[test]
    fn triposplat_profile_buttons_apply_canonical_settings() {
        let mut args = AppArgs::default();
        args.apply_triposplat_profile(TripoSplatProfile::Low);
        assert_eq!(args.num_steps, 5);
        assert_eq!(args.guidance_scale, 3.0);
        assert_eq!(args.triposplat_num_gaussians, TRIPOSPLAT_MIN_NUM_GAUSSIANS);

        args.apply_triposplat_profile(TripoSplatProfile::High);
        assert_eq!(args.num_steps, 50);
        assert_eq!(args.guidance_scale, 3.0);
        assert_eq!(args.triposplat_num_gaussians, TRIPOSPLAT_MAX_NUM_GAUSSIANS);
    }

    #[test]
    fn triposplat_manual_steps_mark_profile_custom_and_clamp() {
        let mut args = AppArgs::default();
        args.apply_triposplat_profile(TripoSplatProfile::Low);
        adjust_triposplat_setting(
            &mut args,
            TripoSplatSetting::Steps,
            TripoSplatSettingDelta::Integer(-100),
        );
        assert_eq!(args.num_steps, TRIPOSPLAT_MIN_STEPS);
        assert_eq!(args.triposplat_profile, TripoSplatProfile::Custom);

        adjust_triposplat_setting(
            &mut args,
            TripoSplatSetting::Steps,
            TripoSplatSettingDelta::Integer(100),
        );
        assert_eq!(args.num_steps, TRIPOSPLAT_MAX_STEPS);
    }

    #[test]
    fn triposplat_manual_gaussian_count_stays_in_supported_range() {
        let mut args = AppArgs::default();
        args.apply_triposplat_profile(TripoSplatProfile::High);
        adjust_triposplat_setting(
            &mut args,
            TripoSplatSetting::Gaussians,
            TripoSplatSettingDelta::Integer(TRIPOSPLAT_GAUSSIAN_STEP as isize),
        );
        assert_eq!(args.triposplat_num_gaussians, TRIPOSPLAT_MAX_NUM_GAUSSIANS);

        args.apply_triposplat_profile(TripoSplatProfile::Low);
        adjust_triposplat_setting(
            &mut args,
            TripoSplatSetting::Gaussians,
            TripoSplatSettingDelta::Integer(-(TRIPOSPLAT_GAUSSIAN_STEP as isize)),
        );
        assert_eq!(args.triposplat_num_gaussians, TRIPOSPLAT_MIN_NUM_GAUSSIANS);
    }

    #[test]
    fn triposplat_gaussian_value_text_uses_exact_grouped_count() {
        let mut args = AppArgs::default();
        args.triposplat_num_gaussians = TRIPOSPLAT_MAX_NUM_GAUSSIANS;
        assert_eq!(
            triposplat_setting_value_text(&args, TripoSplatSetting::Gaussians),
            "262,144"
        );
    }

    #[test]
    fn triposg_manual_settings_clamp_and_format_values() {
        let mut args = AppArgs::default();
        args.num_steps = 1;
        adjust_triposg_setting(
            &mut args,
            TripoSgSetting::Steps,
            TripoSgSettingDelta::Integer(-100),
        );
        assert_eq!(args.num_steps, TRIPOSG_MIN_STEPS);

        adjust_triposg_setting(
            &mut args,
            TripoSgSetting::Steps,
            TripoSgSettingDelta::Integer(100),
        );
        assert_eq!(args.num_steps, TRIPOSG_MAX_STEPS);

        args.num_tokens = 1024;
        adjust_triposg_setting(
            &mut args,
            TripoSgSetting::Tokens,
            TripoSgSettingDelta::Integer(TRIPOSG_TOKEN_STEP as isize),
        );
        assert_eq!(args.num_tokens, 1152);
        assert_eq!(
            triposg_setting_value_text(&args, TripoSgSetting::Tokens),
            "1,152"
        );

        args.target_faces = None;
        adjust_triposg_setting(
            &mut args,
            TripoSgSetting::TargetFaces,
            TripoSgSettingDelta::Integer(TRIPOSG_FACE_STEP as isize),
        );
        assert_eq!(args.target_faces, Some(TRIPOSG_FACE_STEP));
        adjust_triposg_setting(
            &mut args,
            TripoSgSetting::TargetFaces,
            TripoSgSettingDelta::Integer(-(TRIPOSG_FACE_STEP as isize)),
        );
        assert_eq!(args.target_faces, None);
        assert_eq!(
            triposg_setting_value_text(&args, TripoSgSetting::TargetFaces),
            "disabled"
        );
    }

    #[test]
    fn trellis_settings_clamp_and_format_values() {
        let mut args = AppArgs::default();
        args.trellis_quality = TrellisQuality::Low;
        assert_eq!(
            trellis_setting_value_text(&args, TrellisSetting::Resolution),
            "512"
        );
        args.trellis_quality = TrellisQuality::High;
        assert_eq!(
            trellis_quality_value_text(args.trellis_quality),
            "high / 1024"
        );

        args.trellis_pbr_enabled = false;
        assert_eq!(
            trellis_setting_value_text(&args, TrellisSetting::PbrTextureSize),
            "disabled"
        );
        args.trellis_pbr_enabled = true;
        args.trellis_pbr_texture_size = Some(DEFAULT_TRELLIS_PBR_TEXTURE_SIZE);
        adjust_trellis_setting(
            &mut args,
            TrellisSetting::PbrTextureSize,
            TrellisSettingDelta::Integer(10_000),
        );
        assert_eq!(args.trellis_pbr_texture_size, Some(TRELLIS_PBR_TEXTURE_MAX));

        args.trellis_target_faces = None;
        adjust_trellis_setting(
            &mut args,
            TrellisSetting::TargetFaces,
            TrellisSettingDelta::Integer(TRELLIS_FACE_STEP as isize),
        );
        assert_eq!(args.trellis_target_faces, Some(TRELLIS_FACE_STEP));
        adjust_trellis_setting(
            &mut args,
            TrellisSetting::TargetFaces,
            TrellisSettingDelta::Integer(-(TRELLIS_FACE_STEP as isize)),
        );
        assert_eq!(args.trellis_target_faces, None);
        assert_eq!(
            trellis_setting_value_text(&args, TrellisSetting::TargetFaces),
            "disabled"
        );

        args.trellis_max_sparse_coords = None;
        adjust_trellis_setting(
            &mut args,
            TrellisSetting::MaxSparseCoords,
            TrellisSettingDelta::Integer(TRELLIS_SPARSE_COORD_STEP as isize),
        );
        assert_eq!(
            args.trellis_max_sparse_coords,
            Some(TRELLIS_SPARSE_COORD_STEP)
        );
        adjust_trellis_setting(
            &mut args,
            TrellisSetting::MaxSparseCoords,
            TrellisSettingDelta::Integer(-(TRELLIS_SPARSE_COORD_STEP as isize)),
        );
        assert_eq!(args.trellis_max_sparse_coords, None);
        assert_eq!(
            trellis_setting_value_text(&args, TrellisSetting::MaxSparseCoords),
            "uncapped"
        );
    }

    #[test]
    fn pipeline_setting_gate_tracks_active_pipeline() {
        let mut catalog = CatalogState::default();
        let mut args = AppArgs::default();
        args.synthesis_models = vec![SynthesisModel::Triposg, SynthesisModel::Triposplat];
        assert_eq!(
            active_settings_pipeline(Some(&args)),
            Some(SynthesisModel::Triposg)
        );
        assert!(pipeline_settings_enabled(&catalog, Some(&args)));

        args.synthesis_models = vec![SynthesisModel::Triposplat, SynthesisModel::Triposg];
        assert_eq!(
            active_settings_pipeline(Some(&args)),
            Some(SynthesisModel::Triposplat)
        );
        assert!(pipeline_settings_enabled(&catalog, Some(&args)));

        args.synthesis_models = vec![SynthesisModel::Trellis];
        assert_eq!(
            active_settings_pipeline(Some(&args)),
            Some(SynthesisModel::Trellis)
        );
        assert!(pipeline_settings_enabled(&catalog, Some(&args)));

        catalog.set_active_mode(CatalogMode::Scene);
        assert!(pipeline_settings_enabled(&catalog, Some(&args)));
    }
}

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

use crate::bevy_file_dialog::prelude::FileDialogExt;
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
pub enum SceneObjectPoseRefinementSetting {
    Off,
    Geometry,
    #[default]
    GatedGpt,
    AlwaysGpt,
}

impl SceneObjectPoseRefinementSetting {
    fn label(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Geometry => "geometry",
            Self::GatedGpt => "gated-gpt",
            Self::AlwaysGpt => "always-gpt",
        }
    }

    fn cycle(self, delta: isize) -> Self {
        const OPTIONS: [SceneObjectPoseRefinementSetting; 4] = [
            SceneObjectPoseRefinementSetting::Off,
            SceneObjectPoseRefinementSetting::Geometry,
            SceneObjectPoseRefinementSetting::GatedGpt,
            SceneObjectPoseRefinementSetting::AlwaysGpt,
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
    pub object_pose_refinement: SceneObjectPoseRefinementSetting,
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
            object_pose_refinement: SceneObjectPoseRefinementSetting::GatedGpt,
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
    ObjectPoseRefinement,
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

mod catalog;
mod menus;
mod preview;
mod processing;
mod settings_handlers;
mod settings_modal;
mod setup;
mod source_modal;
#[cfg(test)]
mod tests;

use catalog::*;
use menus::*;
pub use preview::preview_light_layers;
use preview::*;
use processing::*;
use settings_handlers::*;
use settings_modal::*;
use setup::*;
use source_modal::*;

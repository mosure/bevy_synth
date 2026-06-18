use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::RenderLayers;
use bevy::camera::{CameraOutputMode, RenderTarget};
use bevy::pbr::MeshMaterial3d;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bevy::ui::widget::Button;
use bevy::window::PrimaryWindow;
use bevy_gaussian_splatting::gaussian::settings::GaussianColorSpace;
use bevy_gaussian_splatting::sort::SortMode;
use bevy_gaussian_splatting::{
    CloudSettings, GaussianCamera, PlanarGaussian3d, PlanarGaussian3dHandle,
};
use bevy_mesh::{Mesh as BevyMesh, Mesh3d};
use bevy_picking::Pickable;
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
    AppArgs, BackendKind, SynthesisModel, TRIPOSPLAT_GAUSSIAN_STEP, TRIPOSPLAT_MAX_NUM_GAUSSIANS,
    TRIPOSPLAT_MIN_NUM_GAUSSIANS, TripoSplatProfile,
};
use bevy_synth_runtime::state::{InferenceQueue, InferenceRequest, UiStatus};

const PANEL_WIDTH: f32 = 336.0;
const MENU_HEIGHT: f32 = 44.0;
const THUMB_SIZE: f32 = 84.0;
const ENTRY_GAP: f32 = 10.0;
const CATALOG_PAGE_SIZE: usize = 6;
const PIPELINE_SELECTOR_WIDTH: f32 = 176.0;
const PIPELINE_SELECTOR_HEIGHT: f32 = 30.0;
const PREVIEW_SIZE: u32 = 128;
const PREVIEW_MAX_LAYER: usize = 30;
const GIZMO_LAYER: usize = 12;
const PREVIEW_CAMERA_FOV: f32 = std::f32::consts::FRAC_PI_4;
const PREVIEW_CAMERA_MARGIN: f32 = 0.2;
const PREVIEW_TARGET_RADIUS: f32 = 0.72;
const PREVIEW_FALLBACK_RADIUS: f32 = 0.72;
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
const DEFAULT_PIPELINE_OPTIONS: [SynthesisModel; 2] =
    [SynthesisModel::Triposg, SynthesisModel::Triposplat];

#[derive(Component)]
pub struct MainCamera;

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

pub struct BurnSynthUiPlugin;

impl Plugin for BurnSynthUiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CatalogState>()
            .init_resource::<DragState>()
            .init_resource::<CatalogSelectionState>()
            .init_resource::<SettingsModalState>()
            .init_resource::<PipelineDropdownState>()
            .add_message::<CatalogSpawnRequest>()
            .add_message::<CatalogDeleteRequest>()
            .add_systems(Startup, setup_ui)
            .add_systems(
                Update,
                (
                    update_queue_text,
                    handle_catalog_toggle,
                    handle_page_buttons,
                    handle_catalog_delete_button,
                    handle_catalog_delete_shortcut,
                    (
                        handle_pipeline_selector_button,
                        handle_pipeline_option_button,
                        sync_pipeline_dropdown,
                        update_pipeline_value_label,
                    )
                        .chain(),
                    handle_settings_button,
                    handle_settings_close_button,
                    handle_triposplat_profile_button,
                    handle_triposplat_setting_step_button,
                    sync_settings_modal,
                    update_settings_labels,
                    (sync_catalog_previews, rebuild_catalog_list).chain(),
                    update_button_visuals,
                    (
                        handle_catalog_entry_interaction,
                        update_drag_ghost,
                        handle_drag_release,
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
    expanded: bool,
    page: usize,
    available_layers: Vec<usize>,
    revision: u64,
}

impl Default for CatalogState {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            expanded: true,
            page: 0,
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
        self.entries.is_empty()
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
            label,
            status: CatalogStatus::Pending,
            mesh: None,
            material: None,
            gaussian: None,
            source_image_path: Some(request.image_path.display().to_string()),
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
            label,
            status: CatalogStatus::Ready,
            mesh: Some(mesh),
            material: Some(material),
            gaussian: None,
            source_image_path,
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
            label,
            status: CatalogStatus::Ready,
            mesh: None,
            material: None,
            gaussian: None,
            source_image_path,
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
            label,
            status: CatalogStatus::Ready,
            mesh: None,
            material: None,
            gaussian: Some(gaussian),
            source_image_path,
            cache_key,
            preview: None,
        });
        self.clamp_page();
        self.bump_revision();
    }

    pub fn entry(&self, id: u32) -> Option<&CatalogEntry> {
        self.entries.iter().find(|entry| entry.id == id)
    }

    pub fn entry_mut(&mut self, id: u32) -> Option<&mut CatalogEntry> {
        self.entries.iter_mut().find(|entry| entry.id == id)
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
        let total = self.entries.len();
        if total == 0 {
            1
        } else {
            total.div_ceil(CATALOG_PAGE_SIZE)
        }
    }

    pub fn clamp_page(&mut self) {
        let max_page = self.page_count().saturating_sub(1);
        if self.page > max_page {
            self.page = max_page;
        }
    }

    pub fn set_page(&mut self, page: usize) {
        self.page = page;
        self.clamp_page();
    }

    pub fn page(&self) -> usize {
        self.page
    }

    pub fn visible_indices(&self) -> Vec<usize> {
        let total = self.entries.len();
        if total == 0 {
            return Vec::new();
        }
        let start = total.saturating_sub(CATALOG_PAGE_SIZE * (self.page + 1));
        let end = total.saturating_sub(CATALOG_PAGE_SIZE * self.page);
        (start..end).rev().collect()
    }

    pub fn has_ready_cube_entry(&self) -> bool {
        self.entries.iter().any(|entry| {
            matches!(entry.status, CatalogStatus::Ready) && entry.label.eq_ignore_ascii_case("cube")
        })
    }
}

pub struct CatalogEntry {
    pub id: u32,
    pub label: String,
    pub status: CatalogStatus,
    pub mesh: Option<Handle<BevyMesh>>,
    pub material: Option<Handle<StandardMaterial>>,
    pub gaussian: Option<Handle<PlanarGaussian3d>>,
    pub source_image_path: Option<String>,
    pub cache_key: Option<String>,
    pub preview: Option<PreviewScene>,
}

#[derive(Clone, Debug)]
pub enum CatalogStatus {
    Pending,
    Ready,
    Failed(String),
}

pub struct PreviewScene {
    pub image: Handle<Image>,
    pub asset_entity: Entity,
    pub camera_entity: Entity,
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
    settings_modal_open: bool,
}

impl CatalogUiState {
    pub fn cursor_over_ui(&self, window: &Window) -> bool {
        if self.settings_modal_open {
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
}

#[derive(Component)]
struct QueueText;

#[derive(Component)]
struct QueueStatusBadge;

#[derive(Component)]
struct QueueStatusDot;

#[derive(Component)]
struct CatalogList;

#[derive(Component)]
struct CatalogToggleButton;

#[derive(Component)]
struct ToggleLabel;

#[derive(Component)]
struct CatalogEntryButton {
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
struct PipelineDropdownHost;

#[derive(Component)]
struct PipelineSelectorButton;

#[derive(Component)]
struct PipelineOptionButton {
    model: SynthesisModel,
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
struct SettingsModalRoot;

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
struct UiRootNode;

#[derive(Resource, Default)]
struct SettingsModalState {
    open: bool,
    entity: Option<Entity>,
}

#[derive(Resource, Default)]
struct PipelineDropdownState {
    open: bool,
    entity: Option<Entity>,
}

#[derive(Resource, Clone, Debug, PartialEq, Eq)]
struct AvailablePipelines {
    models: Vec<SynthesisModel>,
}

fn pipeline_label(model: SynthesisModel) -> &'static str {
    match model {
        SynthesisModel::Triposg => "TripoSG",
        SynthesisModel::Trellis => "Trellis",
        SynthesisModel::Triposplat => "TripoSplat",
    }
}

fn pipeline_selector_value_text(model: SynthesisModel, option_count: usize) -> String {
    if option_count <= 1 {
        format!("{} only", pipeline_label(model))
    } else {
        pipeline_label(model).to_string()
    }
}

fn available_pipeline_models(args: Option<&AppArgs>) -> Vec<SynthesisModel> {
    let models: Box<dyn Iterator<Item = SynthesisModel> + '_> = match args {
        Some(args) => Box::new(args.synthesis_models.iter().copied()),
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

fn pipeline_available(available: Option<&AvailablePipelines>, model: SynthesisModel) -> bool {
    available
        .map(|available| available.models.contains(&model))
        .unwrap_or(true)
}

fn pipeline_supported(args: Option<&AppArgs>, model: SynthesisModel) -> bool {
    let Some(args) = args else {
        return true;
    };
    match model {
        SynthesisModel::Triposplat => triposplat_supported_for_backend(args.backend.clone()),
        SynthesisModel::Triposg | SynthesisModel::Trellis => true,
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

fn setup_ui(mut commands: Commands, args: Option<Res<AppArgs>>) {
    let mut list_entity = Entity::PLACEHOLDER;
    let pipeline_models = available_pipeline_models(args.as_deref());
    let selected_pipeline = args
        .as_deref()
        .and_then(|args| args.synthesis_models.first().copied())
        .or_else(|| pipeline_models.first().copied())
        .unwrap_or(SynthesisModel::Triposg);
    let multiple_pipelines = pipeline_models.len() > 1;
    commands.insert_resource(AvailablePipelines {
        models: pipeline_models.clone(),
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

                    left.spawn((
                        PipelineDropdownHost,
                        Node {
                            position_type: PositionType::Relative,
                            width: Val::Px(PIPELINE_SELECTOR_WIDTH),
                            height: Val::Px(PIPELINE_SELECTOR_HEIGHT),
                            overflow: Overflow::visible(),
                            ..default()
                        },
                    ))
                    .with_children(|selector| {
                        if multiple_pipelines {
                            selector
                                .spawn((
                                    Button,
                                    PipelineSelectorButton,
                                    ControlButton(ControlButtonKind::Secondary),
                                    Node {
                                        width: Val::Px(PIPELINE_SELECTOR_WIDTH),
                                        height: Val::Px(PIPELINE_SELECTOR_HEIGHT),
                                        padding: UiRect::axes(Val::Px(10.0), Val::Px(4.0)),
                                        border: UiRect::all(Val::Px(1.0)),
                                        column_gap: Val::Px(8.0),
                                        align_items: AlignItems::Center,
                                        justify_content: JustifyContent::SpaceBetween,
                                        ..default()
                                    },
                                    BorderColor::all(BUTTON_ACTIVE_BORDER),
                                    BackgroundColor(BUTTON_ACTIVE_BG),
                                ))
                                .with_children(|button| {
                                    button.spawn((
                                        Text::new(pipeline_selector_value_text(
                                            selected_pipeline,
                                            pipeline_models.len(),
                                        )),
                                        TextFont::from_font_size(12.0),
                                        TextColor(BUTTON_TEXT),
                                        PipelineValueLabel,
                                        ButtonLabel,
                                    ));
                                    button.spawn((
                                        Text::new("v"),
                                        TextFont::from_font_size(12.0),
                                        TextColor(BUTTON_TEXT),
                                        ButtonLabel,
                                    ));
                                });
                        } else {
                            selector
                                .spawn((
                                    Node {
                                        width: Val::Px(PIPELINE_SELECTOR_WIDTH),
                                        height: Val::Px(PIPELINE_SELECTOR_HEIGHT),
                                        padding: UiRect::axes(Val::Px(10.0), Val::Px(4.0)),
                                        border: UiRect::all(Val::Px(1.0)),
                                        column_gap: Val::Px(8.0),
                                        align_items: AlignItems::Center,
                                        justify_content: JustifyContent::FlexStart,
                                        ..default()
                                    },
                                    BorderColor::all(BUTTON_ACTIVE_BORDER),
                                    BackgroundColor(BUTTON_ACTIVE_BG),
                                ))
                                .with_children(|selector| {
                                    selector.spawn((
                                        Text::new(pipeline_selector_value_text(
                                            selected_pipeline,
                                            pipeline_models.len(),
                                        )),
                                        TextFont::from_font_size(12.0),
                                        TextColor(BUTTON_TEXT),
                                        PipelineValueLabel,
                                    ));
                                });
                        }
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
                                flex_direction: FlexDirection::Row,
                                align_items: AlignItems::Center,
                                column_gap: Val::Px(7.0),
                                padding: UiRect::axes(Val::Px(10.0), Val::Px(5.0)),
                                border: UiRect::all(Val::Px(1.0)),
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
                        justify_content: JustifyContent::SpaceBetween,
                        align_items: AlignItems::Center,
                        ..default()
                    })
                    .with_children(|header| {
                        header.spawn((
                            Text::new("catalog"),
                            TextFont::from_font_size(14.0),
                            TextColor(Color::srgb(0.82, 0.86, 0.94)),
                        ));
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
                                            padding: UiRect::axes(Val::Px(6.0), Val::Px(2.0)),
                                            border: UiRect::all(Val::Px(1.0)),
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
                                    PageLabel,
                                ));
                                controls
                                    .spawn((
                                        Button,
                                        CatalogNextButton,
                                        ControlButton(ControlButtonKind::Nav),
                                        Node {
                                            padding: UiRect::axes(Val::Px(6.0), Val::Px(2.0)),
                                            border: UiRect::all(Val::Px(1.0)),
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
                                            padding: UiRect::axes(Val::Px(8.0), Val::Px(4.0)),
                                            border: UiRect::all(Val::Px(1.0)),
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
                                            padding: UiRect::axes(Val::Px(8.0), Val::Px(4.0)),
                                            border: UiRect::all(Val::Px(1.0)),
                                            ..default()
                                        },
                                        BorderColor::all(BUTTON_BORDER),
                                        BackgroundColor(BUTTON_BG),
                                    ))
                                    .with_children(|button| {
                                        button.spawn((
                                            Text::new("collapse"),
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
        settings_modal_open: false,
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
            worker_message.clone(),
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
            "collapse".to_string()
        } else {
            "expand".to_string()
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
                        Text::new("No catalog items yet"),
                        TextFont::from_font_size(13.0),
                        TextColor(Color::srgb(0.88, 0.9, 0.95)),
                    ));
                    empty.spawn((
                        Text::new("Drop an image, or click open image to queue one."),
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
                CatalogStatus::Ready => ("ready".to_string(), Color::srgb(0.4, 0.85, 0.55)),
                CatalogStatus::Failed(err) => {
                    let mut label = if err.is_empty() {
                        "failed".to_string()
                    } else {
                        format!("failed: {err}")
                    };
                    if label.len() > 48 {
                        label.truncate(48);
                        label.push_str("...");
                    }
                    (label, Color::srgb(0.9, 0.3, 0.3))
                }
            };
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
                        flex_direction: FlexDirection::Column,
                        justify_content: JustifyContent::Center,
                        padding: UiRect::left(Val::Px(4.0)),
                        row_gap: Val::Px(4.0),
                        ..default()
                    })
                    .with_children(|text_col| {
                        text_col.spawn((
                            Text::new(entry.label.clone()),
                            TextFont::from_font_size(13.0),
                            TextColor(Color::srgb(0.9, 0.92, 0.97)),
                        ));
                        text_col.spawn((
                            Text::new(status_label.clone()),
                            TextFont::from_font_size(12.0),
                            TextColor(status_color),
                        ));
                    });

                    row.spawn((
                        Node {
                            width: Val::Px(10.0),
                            height: Val::Px(10.0),
                            margin: UiRect::left(Val::Auto),
                            ..default()
                        },
                        BackgroundColor(status_color),
                    ));
                });
        }
    });
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
    mut drag: ResMut<DragState>,
    mut selection: ResMut<CatalogSelectionState>,
) {
    for (interaction, entry) in interactions.iter_mut() {
        if *interaction == Interaction::Pressed {
            selection.selected = Some(entry.id);
            drag.active = Some(entry.id);
            drag.ghost_entry = None;
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

    if let Some(preview) = entry.preview {
        commands.entity(preview.asset_entity).despawn();
        commands.entity(preview.camera_entity).despawn();
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

    delete_requests.write(CatalogDeleteRequest {
        cache_key: entry.cache_key,
    });
}

fn handle_catalog_delete_button(
    mut interactions: Query<&Interaction, (Changed<Interaction>, With<CatalogDeleteButton>)>,
    mut catalog: ResMut<CatalogState>,
    mut selection: ResMut<CatalogSelectionState>,
    mut drag: ResMut<DragState>,
    mut commands: Commands,
    mut delete_requests: MessageWriter<CatalogDeleteRequest>,
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
    let selected_pipeline = args_ref.and_then(|args| args.synthesis_models.first().copied());
    let triposplat_settings_enabled = triposplat_settings_enabled(args_ref);

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
            !pipeline_available(available_ref, pipeline_option.model)
                || !pipeline_supported(args_ref, pipeline_option.model)
        } else if settings_button.is_some() {
            !triposplat_settings_enabled
        } else {
            false
        };
        let active = pipeline_selector.is_some()
            || pipeline_option.is_some_and(|pipeline| Some(pipeline.model) == selected_pipeline)
            || settings_button.is_some_and(|_| triposplat_settings_enabled && modal.open)
            || profile
                .zip(args_ref)
                .is_some_and(|(profile, args)| profile.profile == args.triposplat_profile);
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
            && ((entry.mesh.is_some() && entry.material.is_some()) || entry.gaussian.is_some());

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
                    commands.entity(preview.asset_entity).despawn();
                    commands.entity(preview.camera_entity).despawn();
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
    available: Option<Res<AvailablePipelines>>,
    mut dropdown: ResMut<PipelineDropdownState>,
    mut interactions: Query<&Interaction, (Changed<Interaction>, With<PipelineSelectorButton>)>,
) {
    let option_count = available
        .as_deref()
        .map(|available| available.models.len())
        .unwrap_or(DEFAULT_PIPELINE_OPTIONS.len());
    for interaction in interactions.iter_mut() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        dropdown.open = option_count > 1 && !dropdown.open;
    }
}

fn handle_pipeline_option_button(
    args: Option<ResMut<AppArgs>>,
    available: Option<Res<AvailablePipelines>>,
    mut dropdown: ResMut<PipelineDropdownState>,
    mut modal: ResMut<SettingsModalState>,
    mut interactions: Query<(&Interaction, &PipelineOptionButton), Changed<Interaction>>,
) {
    let Some(mut args) = args else {
        return;
    };
    let available_ref = available.as_deref();
    for (interaction, button) in interactions.iter_mut() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        if !pipeline_available(available_ref, button.model) {
            info!(
                "synthesis pipeline {} is not enabled for this app launch",
                pipeline_label(button.model)
            );
            continue;
        }
        if args
            .synthesis_models
            .first()
            .is_some_and(|current| *current == button.model)
        {
            dropdown.open = false;
            continue;
        }
        if !pipeline_supported(Some(&args), button.model) {
            info!(
                "synthesis pipeline {} is unavailable for backend {:?}",
                pipeline_label(button.model),
                &args.backend
            );
            continue;
        }
        args.synthesis_models = vec![button.model];
        dropdown.open = false;
        if !matches!(button.model, SynthesisModel::Triposplat) {
            modal.open = false;
        }
        if matches!(button.model, SynthesisModel::Triposplat)
            && args.triposplat_profile != TripoSplatProfile::Custom
        {
            let profile = args.triposplat_profile;
            args.apply_triposplat_profile(profile);
        }
        info!(
            "selected synthesis pipeline: {}",
            pipeline_label(button.model)
        );
    }
}

fn sync_pipeline_dropdown(
    mut commands: Commands,
    args: Option<Res<AppArgs>>,
    available: Option<Res<AvailablePipelines>>,
    mut dropdown: ResMut<PipelineDropdownState>,
    hosts: Query<Entity, With<PipelineDropdownHost>>,
    children: Query<&Children>,
) {
    let Some(available) = available else {
        dropdown.open = false;
        if let Some(entity) = dropdown.entity.take() {
            despawn_children_recursive(entity, &mut commands, &children);
            commands.entity(entity).despawn();
        }
        return;
    };
    if available.models.len() <= 1 {
        dropdown.open = false;
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
                args.as_deref(),
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

fn update_pipeline_value_label(
    args: Option<Res<AppArgs>>,
    available: Option<Res<AvailablePipelines>>,
    mut labels: Query<&mut Text, With<PipelineValueLabel>>,
) {
    let selected = args
        .as_deref()
        .and_then(|args| args.synthesis_models.first().copied())
        .or_else(|| {
            available
                .as_deref()
                .and_then(|available| available.models.first().copied())
        })
        .unwrap_or(SynthesisModel::Triposg);
    let option_count = available
        .as_deref()
        .map(|available| available.models.len())
        .unwrap_or(DEFAULT_PIPELINE_OPTIONS.len());
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
    args: Option<&AppArgs>,
    available: &AvailablePipelines,
) -> Entity {
    let selected_pipeline = args.and_then(|args| args.synthesis_models.first().copied());
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
                for model in available.models.iter().copied() {
                    menu.spawn((
                        Button,
                        PipelineOptionButton { model },
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
                        BorderColor::all(if Some(model) == selected_pipeline {
                            BUTTON_ACTIVE_BORDER
                        } else {
                            BUTTON_BORDER
                        }),
                        BackgroundColor(if Some(model) == selected_pipeline {
                            BUTTON_ACTIVE_BG
                        } else {
                            BUTTON_BG
                        }),
                    ))
                    .with_children(|button| {
                        button.spawn((
                            Text::new(pipeline_label(model)),
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
    args: Option<Res<AppArgs>>,
    mut modal: ResMut<SettingsModalState>,
) {
    for interaction in interactions.iter_mut() {
        if *interaction == Interaction::Pressed {
            if triposplat_settings_enabled(args.as_deref()) {
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
    if !triposplat_settings_enabled(Some(&*args)) {
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
    if !triposplat_settings_enabled(Some(&*args)) {
        return;
    }
    for (interaction, button) in interactions.iter_mut() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        adjust_triposplat_setting(&mut args, button.setting, button.delta);
    }
}

fn sync_settings_modal(
    mut commands: Commands,
    args: Option<Res<AppArgs>>,
    mut modal: ResMut<SettingsModalState>,
    mut ui: ResMut<CatalogUiState>,
    children: Query<&Children>,
) {
    if !triposplat_settings_enabled(args.as_deref()) {
        modal.open = false;
    }
    ui.settings_modal_open = modal.open;
    match (modal.open, modal.entity) {
        (true, None) => {
            modal.entity = Some(spawn_settings_modal(&mut commands));
        }
        (false, Some(entity)) => {
            despawn_children_recursive(entity, &mut commands, &children);
            commands.entity(entity).despawn();
            modal.entity = None;
        }
        _ => {}
    }
}

fn update_settings_labels(
    args: Option<Res<AppArgs>>,
    mut profile_labels: Query<
        &mut Text,
        (
            With<TripoSplatProfileValueLabel>,
            Without<TripoSplatSettingValueLabel>,
        ),
    >,
    mut value_labels: Query<
        (&TripoSplatSettingValueLabel, &mut Text),
        Without<TripoSplatProfileValueLabel>,
    >,
) {
    let Some(args) = args else {
        return;
    };
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
}

fn spawn_settings_modal(commands: &mut Commands) -> Entity {
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
                    width: Val::Px(390.0),
                    flex_direction: FlexDirection::Column,
                    row_gap: Val::Px(14.0),
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
                        ..default()
                    })
                    .with_children(|header| {
                        header.spawn((
                            Text::new("TripoSplat settings"),
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
            });
        })
        .id()
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

fn triposplat_setting_value_text(args: &AppArgs, setting: TripoSplatSetting) -> String {
    match setting {
        TripoSplatSetting::Steps => args.num_steps.to_string(),
        TripoSplatSetting::Guidance => format!("{:.1}", args.guidance_scale),
        TripoSplatSetting::Gaussians => format_grouped_usize(args.triposplat_num_gaussians),
    }
}

fn triposplat_settings_enabled(args: Option<&AppArgs>) -> bool {
    args.and_then(|args| args.synthesis_models.first().copied())
        .is_some_and(|model| matches!(model, SynthesisModel::Triposplat))
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

    let asset_entity = match asset {
        PreviewAsset::Mesh { mesh, material } => commands
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
        PreviewAsset::GaussianSplat { cloud } => commands
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
    };

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
        asset_entity,
        camera_entity,
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
    let mut layers = RenderLayers::layer(0);
    for layer in 1..=PREVIEW_MAX_LAYER {
        layers = layers.with(layer);
    }
    layers
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
    fn pipeline_selector_collapses_to_single_launch_model() {
        let mut args = AppArgs::default();
        args.backend = BackendKind::Wgpu;
        args.synthesis_models = vec![SynthesisModel::Triposplat];
        let mut app = ui_test_app(Some(args));

        app.update();

        let world = app.world_mut();
        let mut value_query = world.query_filtered::<&Text, With<PipelineValueLabel>>();
        let values: Vec<_> = value_query
            .iter(world)
            .map(|label| label.0.clone())
            .collect();
        assert_eq!(values, vec!["TripoSplat only".to_string()]);
        assert_eq!(
            world.query::<&PipelineSelectorButton>().iter(world).count(),
            0
        );
        assert_eq!(
            world.query::<&PipelineOptionButton>().iter(world).count(),
            0
        );
        assert_eq!(
            world.resource::<AvailablePipelines>().models,
            vec![SynthesisModel::Triposplat]
        );
    }

    #[test]
    fn pipeline_dropdown_spawns_enabled_model_options() {
        let mut args = AppArgs::default();
        args.backend = BackendKind::Wgpu;
        args.synthesis_models = vec![SynthesisModel::Triposg, SynthesisModel::Triposplat];
        let mut app = ui_test_app(Some(args));

        app.update();
        {
            let world = app.world_mut();
            assert_eq!(
                world.query::<&PipelineSelectorButton>().iter(world).count(),
                1
            );
            assert_eq!(
                world.query::<&PipelineOptionButton>().iter(world).count(),
                0
            );
        }

        app.world_mut().resource_mut::<PipelineDropdownState>().open = true;
        app.update();

        let world = app.world_mut();
        let mut query = world.query::<&PipelineOptionButton>();
        let models: Vec<_> = query.iter(world).map(|button| button.model).collect();
        assert_eq!(
            models,
            vec![SynthesisModel::Triposg, SynthesisModel::Triposplat]
        );
    }

    #[test]
    fn unavailable_launch_models_are_not_selectable() {
        let available = AvailablePipelines {
            models: vec![SynthesisModel::Triposplat],
        };
        assert!(pipeline_available(
            Some(&available),
            SynthesisModel::Triposplat
        ));
        assert!(!pipeline_available(
            Some(&available),
            SynthesisModel::Triposg
        ));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn triposplat_pipeline_is_supported_on_native_wgpu() {
        let mut args = AppArgs::default();
        args.backend = BackendKind::Wgpu;
        assert!(pipeline_supported(Some(&args), SynthesisModel::Triposplat));
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
    }

    #[test]
    fn triposplat_settings_modal_only_opens_for_triposplat_pipeline() {
        let mut triposg_args = AppArgs::default();
        triposg_args.synthesis_models = vec![SynthesisModel::Triposg, SynthesisModel::Triposplat];
        let mut app = ui_test_app(Some(triposg_args));

        app.world_mut().resource_mut::<SettingsModalState>().open = true;
        app.update();

        {
            let world = app.world_mut();
            assert!(!world.resource::<SettingsModalState>().open);
            assert_eq!(
                world.query::<&SettingsModalRoot>().iter(world).count(),
                0,
                "TripoSplat settings should not be available for TripoSG"
            );
        }

        app.world_mut().resource_mut::<AppArgs>().synthesis_models =
            vec![SynthesisModel::Triposplat, SynthesisModel::Triposg];
        app.world_mut().resource_mut::<SettingsModalState>().open = true;
        app.update();

        let world = app.world_mut();
        assert!(world.resource::<SettingsModalState>().open);
        assert_eq!(
            world.query::<&SettingsModalRoot>().iter(world).count(),
            1,
            "TripoSplat settings should open when TripoSplat is active"
        );
    }

    #[test]
    fn triposplat_settings_modal_closes_when_pipeline_changes() {
        let mut args = AppArgs::default();
        args.synthesis_models = vec![SynthesisModel::Triposplat, SynthesisModel::Triposg];
        let mut app = ui_test_app(Some(args));

        app.world_mut().resource_mut::<SettingsModalState>().open = true;
        app.update();
        {
            let world = app.world_mut();
            assert_eq!(world.query::<&SettingsModalRoot>().iter(world).count(), 1);
        }

        app.world_mut().resource_mut::<AppArgs>().synthesis_models =
            vec![SynthesisModel::Triposg, SynthesisModel::Triposplat];
        app.update();

        let world = app.world_mut();
        assert!(!world.resource::<SettingsModalState>().open);
        assert_eq!(world.query::<&SettingsModalRoot>().iter(world).count(), 0);
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
    fn triposplat_setting_gate_tracks_active_pipeline() {
        let mut args = AppArgs::default();
        args.synthesis_models = vec![SynthesisModel::Triposg, SynthesisModel::Triposplat];
        assert!(!triposplat_settings_enabled(Some(&args)));

        args.synthesis_models = vec![SynthesisModel::Triposplat, SynthesisModel::Triposg];
        assert!(triposplat_settings_enabled(Some(&args)));
    }
}

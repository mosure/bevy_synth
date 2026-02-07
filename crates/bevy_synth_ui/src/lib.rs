use bevy::asset::RenderAssetUsages;
use bevy::camera::RenderTarget;
use bevy::camera::visibility::RenderLayers;
use bevy::pbr::MeshMaterial3d;
use bevy::prelude::*;
use bevy::render::alpha::AlphaMode;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bevy::window::PrimaryWindow;
use bevy_mesh::{Mesh as BevyMesh, Mesh3d};
use bevy_picking::Pickable;

use bevy_synth_runtime::state::{InferenceQueue, InferenceRequest};

#[cfg(not(target_arch = "wasm32"))]
use bevy_file_dialog::prelude::FileDialogExt;

const PANEL_WIDTH: f32 = 336.0;
const MENU_HEIGHT: f32 = 44.0;
const THUMB_SIZE: f32 = 84.0;
const ENTRY_GAP: f32 = 10.0;
const CATALOG_PAGE_SIZE: usize = 8;
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

#[derive(Component)]
pub struct MainCamera;

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Debug)]
pub struct ImagePickDialog;

#[derive(Message, Clone, Debug)]
pub struct CatalogSpawnRequest {
    pub mesh: Handle<BevyMesh>,
    pub material: Handle<StandardMaterial>,
    pub transform: Transform,
    pub cache_key: Option<String>,
    pub select_spawned: bool,
}

pub struct BurnSynthUiPlugin;

impl Plugin for BurnSynthUiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CatalogState>()
            .init_resource::<DragState>()
            .add_message::<CatalogSpawnRequest>()
            .add_systems(Startup, setup_ui)
            .add_systems(
                Update,
                (
                    update_queue_text,
                    handle_catalog_toggle,
                    handle_page_buttons,
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

        #[cfg(not(target_arch = "wasm32"))]
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
}

pub struct CatalogEntry {
    pub id: u32,
    pub label: String,
    pub status: CatalogStatus,
    pub mesh: Option<Handle<BevyMesh>>,
    pub material: Option<Handle<StandardMaterial>>,
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
    pub mesh_entity: Entity,
    pub camera_entity: Entity,
    pub layer_index: usize,
}

#[derive(Clone, Copy, Debug)]
struct PreviewFit {
    mesh_translation: Vec3,
    mesh_scale: f32,
    radius: f32,
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
}

impl CatalogUiState {
    pub fn cursor_over_ui(&self, window: &Window) -> bool {
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

#[cfg(not(target_arch = "wasm32"))]
#[derive(Component)]
struct OpenImageButton;

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

fn setup_ui(mut commands: Commands) {
    let mut list_entity = Entity::PLACEHOLDER;

    let root = commands
        .spawn(Node {
            position_type: PositionType::Absolute,
            top: Val::Px(0.0),
            left: Val::Px(0.0),
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            flex_direction: FlexDirection::Column,
            ..default()
        })
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
                        Text::new("burn_synth"),
                        TextFont::from_font_size(16.0),
                        TextColor(Color::srgb(0.92, 0.94, 0.98)),
                    ));

                    #[cfg(not(target_arch = "wasm32"))]
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
                        BorderRadius::all(Val::Px(7.0)),
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
                            BorderRadius::all(Val::Px(999.0)),
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
                                BorderRadius::all(Val::Px(999.0)),
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
                                        BorderRadius::all(Val::Px(6.0)),
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
                                        BorderRadius::all(Val::Px(6.0)),
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
                                        CatalogToggleButton,
                                        ControlButton(ControlButtonKind::Secondary),
                                        Node {
                                            padding: UiRect::axes(Val::Px(8.0), Val::Px(4.0)),
                                            border: UiRect::all(Val::Px(1.0)),
                                            ..default()
                                        },
                                        BorderColor::all(BUTTON_BORDER),
                                        BackgroundColor(BUTTON_BG),
                                        BorderRadius::all(Val::Px(6.0)),
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
    });
}

fn update_queue_text(
    queue: Res<InferenceQueue>,
    mut query: Query<(&mut Text, &mut TextColor), With<QueueText>>,
    mut dots: Query<&mut BackgroundColor, (With<QueueStatusDot>, Without<QueueStatusBadge>)>,
    mut badges: Query<
        (&mut BackgroundColor, &mut BorderColor),
        (With<QueueStatusBadge>, Without<QueueStatusDot>),
    >,
) {
    let (text, text_color, dot_color, badge_bg, badge_border) =
        if let Some(active) = queue.active.as_ref() {
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
                    BorderRadius::all(Val::Px(8.0)),
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
                    BorderRadius::all(Val::Px(8.0)),
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
                            BorderRadius::all(Val::Px(6.0)),
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
                        BorderRadius::all(Val::Px(6.0)),
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
) {
    for (interaction, entry) in interactions.iter_mut() {
        if *interaction == Interaction::Pressed {
            drag.active = Some(entry.id);
            drag.ghost_entry = None;
        }
    }
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

fn update_button_visuals(
    catalog: Res<CatalogState>,
    drag: Res<DragState>,
    mut controls: Query<
        (
            &Interaction,
            &ControlButton,
            Option<&CatalogPrevButton>,
            Option<&CatalogNextButton>,
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
    for (interaction, button, prev, next, children, mut bg, mut border) in controls.iter_mut() {
        let disabled = if prev.is_some() {
            catalog.page() == 0
        } else if next.is_some() {
            catalog.page() + 1 >= catalog.page_count()
        } else {
            false
        };
        let (button_bg, button_border, text_color) =
            control_button_palette(button.0, *interaction, disabled);
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
        let (entry_bg, entry_border) = if dragging_this_entry {
            (BUTTON_OPEN_BG_HOVER, BUTTON_OPEN_BORDER_HOVER)
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
) -> (Color, Color, Color) {
    if disabled {
        return (
            BUTTON_BG_DISABLED,
            BUTTON_BORDER_DISABLED,
            BUTTON_TEXT_DISABLED,
        );
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
    let (Some(mesh_handle), Some(material)) = (entry.mesh.clone(), entry.material.clone()) else {
        return;
    };
    spawn_requests.write(CatalogSpawnRequest {
        mesh: mesh_handle,
        material,
        transform: Transform::from_translation(position),
        cache_key: entry.cache_key.clone(),
        select_spawned: true,
    });
}

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
) {
    enum PreviewAction {
        Create {
            index: usize,
            mesh: Handle<BevyMesh>,
            material: Handle<StandardMaterial>,
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
            && entry.mesh.is_some()
            && entry.material.is_some();

        match (should_show, entry.preview.is_some()) {
            (true, false) => {
                if let (Some(mesh), Some(material)) = (entry.mesh.clone(), entry.material.clone()) {
                    let fit = meshes
                        .get(&mesh)
                        .map(preview_fit_for_mesh)
                        .unwrap_or_else(PreviewFit::fallback);
                    actions.push(PreviewAction::Create {
                        index,
                        mesh,
                        material,
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
            PreviewAction::Create {
                index,
                mesh,
                material,
                fit,
            } => {
                if let Some(layer_index) = catalog.alloc_preview_layer() {
                    let preview = spawn_preview_scene(
                        &mut commands,
                        &mut images,
                        mesh,
                        material,
                        layer_index,
                        fit,
                    );
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
                    commands.entity(preview.mesh_entity).despawn();
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

#[cfg(not(target_arch = "wasm32"))]
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
                .pick_multiple_file_paths::<ImagePickDialog>();
        }
    }
}

fn spawn_preview_scene(
    commands: &mut Commands,
    images: &mut Assets<Image>,
    mesh_handle: Handle<BevyMesh>,
    material: Handle<StandardMaterial>,
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

    let mesh_entity = commands
        .spawn((
            Pickable::IGNORE,
            Mesh3d(mesh_handle),
            MeshMaterial3d(material),
            Transform {
                translation: fit.mesh_translation,
                scale: Vec3::splat(fit.mesh_scale),
                ..default()
            },
            layer.clone(),
            ThumbnailSpin,
        ))
        .id();

    let camera_entity = commands
        .spawn((
            Camera3d::default(),
            Camera {
                order: 2,
                target: RenderTarget::Image(image_handle.clone().into()),
                ..default()
            },
            Projection::Perspective(PerspectiveProjection {
                fov: PREVIEW_CAMERA_FOV,
                near: 0.01,
                ..default()
            }),
            Transform::from_translation(Vec3::new(0.0, fit.radius * 0.35, camera_distance))
                .looking_at(Vec3::ZERO, Vec3::Y),
            layer.clone(),
        ))
        .id();

    PreviewScene {
        image: image_handle,
        mesh_entity,
        camera_entity,
        layer_index,
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

pub fn preview_light_layers() -> RenderLayers {
    let mut layers = RenderLayers::layer(0);
    for layer in 1..=PREVIEW_MAX_LAYER {
        layers = layers.with(layer);
    }
    layers
}

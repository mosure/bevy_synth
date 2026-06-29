use bevy::ecs::message::MessageReader;
use bevy::input::mouse::{MouseMotion, MouseScrollUnit, MouseWheel};
use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use bevy_picking::hover::PickingInteraction;
use bevy_synth_ui::bevy_transform_gizmos;
use bevy_synth_ui::bevy_transform_gizmos::TransformGizmo;
use bevy_synth_ui::{CatalogUiState, DragState, MainCamera};

const PANORBIT_MIN_RADIUS: f32 = 0.05;
const PANORBIT_MAX_RADIUS: f32 = 500.0;
const PANORBIT_ORBIT_SMOOTHNESS: f32 = 0.1;
const PANORBIT_PAN_SMOOTHNESS: f32 = 0.02;
const PANORBIT_ZOOM_SMOOTHNESS: f32 = 0.1;
const PANORBIT_SNAP_EPSILON: f32 = 0.001;

#[derive(Component, Clone, Debug)]
pub(crate) struct PanOrbitCamera {
    pub(crate) button_orbit: MouseButton,
    pub(crate) button_pan: MouseButton,
    pub(crate) enabled: bool,
    pub(crate) initialized: bool,
    pub(crate) allow_upside_down: bool,
    pub(crate) is_upside_down: bool,
    pub(crate) focus: Vec3,
    pub(crate) target_focus: Vec3,
    pub(crate) yaw: Option<f32>,
    pub(crate) target_yaw: f32,
    pub(crate) pitch: Option<f32>,
    pub(crate) target_pitch: f32,
    pub(crate) radius: Option<f32>,
    pub(crate) target_radius: f32,
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
pub(crate) struct PanOrbitCameraSystemSet;

pub(crate) struct PanOrbitCameraPlugin;

impl Plugin for PanOrbitCameraPlugin {
    fn build(&self, app: &mut App) {
        app.configure_sets(PostUpdate, PanOrbitCameraSystemSet)
            .add_systems(
                PostUpdate,
                update_panorbit_camera.in_set(PanOrbitCameraSystemSet),
            );
    }
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

pub(crate) fn sync_panorbit_bindings(mut cameras: Query<&mut PanOrbitCamera>) {
    for mut camera in cameras.iter_mut() {
        if camera.button_orbit != MouseButton::Left {
            camera.button_orbit = MouseButton::Left;
        }
        if camera.button_pan != MouseButton::Right {
            camera.button_pan = MouseButton::Right;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn sync_panorbit_enabled(
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
        .map(|window| ui_state.cursor_over_ui(window))
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

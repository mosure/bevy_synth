use std::path::{Path, PathBuf};
use std::time::SystemTime;
use std::{fs, mem};

use bevy::camera::visibility::RenderLayers;
use bevy::prelude::*;
use bevy_gaussian_splatting::gaussian::settings::GaussianColorSpace;
use bevy_gaussian_splatting::sort::SortMode;
use bevy_gaussian_splatting::{
    CloudSettings, Gaussian3d, PlanarGaussian3d, PlanarGaussian3dHandle,
    SphericalHarmonicCoefficients,
};
use bevy_picking::prelude::Pickable;
use bevy_synth_ui::{SceneProcessingState, ViewerDebugSettings};
use serde_json::Value;

use crate::app::SceneRenderCamera;

const DEPTH_DEBUG_MAX_GAUSSIANS: usize = 1280 * 720;

#[derive(Component)]
pub(crate) struct DepthDebugCloud;

#[derive(Clone, Debug)]
pub(super) struct SceneDebugCamera {
    pub(super) transform: Transform,
    pub(super) vertical_fov_degrees: f32,
    pub(super) aspect: f32,
    source_image_path: PathBuf,
    depth_summary_path: PathBuf,
    depth_raw_path: PathBuf,
    intrinsics: DepthDebugIntrinsics,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct DepthDebugIntrinsics {
    pub(crate) fx: f32,
    pub(crate) fy: f32,
    pub(crate) cx: f32,
    pub(crate) cy: f32,
    pub(crate) width: usize,
    pub(crate) height: usize,
}

#[derive(Resource, Default)]
pub(crate) struct SceneDepthDebugState {
    pub(super) evidence_signature: Option<String>,
    pub(super) camera: Option<SceneDebugCamera>,
    pub(super) cloud_signature: Option<String>,
    pub(super) cloud_entity: Option<Entity>,
    pub(super) last_error: Option<String>,
}

pub(crate) fn sync_scene_depth_debug_artifacts(
    settings: Res<ViewerDebugSettings>,
    processing: Res<SceneProcessingState>,
    mut state: ResMut<SceneDepthDebugState>,
    mut scene_cameras: Query<(&mut Transform, &mut Projection), With<SceneRenderCamera>>,
) {
    if !settings.draw_scene_camera_frustum && !settings.depth_cloud_overlay {
        return;
    }
    if !settings.is_changed() && !processing.is_changed() && state.camera.is_some() {
        return;
    }

    let roots = processing.artifact_roots();
    let Some(evidence_path) = find_scene_grounding_evidence_path(&roots) else {
        return;
    };
    let signature = file_signature(&evidence_path);
    if state.evidence_signature.as_deref() != Some(signature.as_str()) {
        match load_scene_debug_camera(&evidence_path) {
            Ok(camera) => {
                state.camera = Some(camera);
                state.evidence_signature = Some(signature);
                state.cloud_signature = None;
                state.last_error = None;
            }
            Err(err) => {
                if state.last_error.as_deref() != Some(err.as_str()) {
                    warn!("Depth debug overlay could not load scene camera evidence: {err}");
                }
                state.last_error = Some(err);
                return;
            }
        }
    }

    let Some(camera) = state.camera.as_ref() else {
        return;
    };
    for (mut transform, mut projection) in scene_cameras.iter_mut() {
        *transform = camera.transform;
        if let Projection::Perspective(perspective) = projection.as_mut() {
            perspective.fov = camera.vertical_fov_degrees.to_radians();
            perspective.aspect_ratio = camera.aspect.max(0.1);
        }
    }
}

pub(crate) fn sync_depth_debug_cloud(
    mut commands: Commands,
    settings: Res<ViewerDebugSettings>,
    mut state: ResMut<SceneDepthDebugState>,
    mut gaussian_clouds: ResMut<Assets<PlanarGaussian3d>>,
    existing_clouds: Query<Entity, With<DepthDebugCloud>>,
) {
    if !settings.depth_cloud_overlay {
        if state.cloud_entity.is_some() || !existing_clouds.is_empty() {
            for entity in existing_clouds.iter() {
                commands.entity(entity).despawn();
            }
            state.cloud_entity = None;
            state.cloud_signature = None;
        }
        return;
    }

    let Some(camera) = state.camera.clone() else {
        return;
    };
    let signature = format!(
        "{}|{}|{}|{}|{}",
        state.evidence_signature.as_deref().unwrap_or_default(),
        camera.source_image_path.display(),
        camera.depth_summary_path.display(),
        camera.depth_raw_path.display(),
        settings
            .depth_cloud_max_gaussians
            .min(DEPTH_DEBUG_MAX_GAUSSIANS)
    );
    if state.cloud_signature.as_deref() == Some(signature.as_str())
        && state.cloud_entity.is_some()
        && !settings.is_changed()
    {
        return;
    }

    match load_depth_debug_cloud(&camera, settings.depth_cloud_max_gaussians) {
        Ok(cloud) => {
            for entity in existing_clouds.iter() {
                commands.entity(entity).despawn();
            }
            let handle = gaussian_clouds.add(cloud);
            let entity = commands
                .spawn((
                    PlanarGaussian3dHandle(handle),
                    depth_debug_cloud_settings(),
                    Transform::IDENTITY,
                    RenderLayers::layer(0),
                    Pickable {
                        should_block_lower: false,
                        is_hoverable: false,
                    },
                    DepthDebugCloud,
                    Name::new("depth_debug_rgb_gaussian_cloud"),
                ))
                .id();
            state.cloud_entity = Some(entity);
            state.cloud_signature = Some(signature);
            state.last_error = None;
        }
        Err(err) => {
            if state.last_error.as_deref() != Some(err.as_str()) {
                warn!("Depth debug overlay could not build RGB Gaussian cloud: {err}");
            }
            state.last_error = Some(err);
        }
    }
}

fn find_scene_grounding_evidence_path(roots: &[PathBuf]) -> Option<PathBuf> {
    for root in roots {
        for candidate in scene_grounding_evidence_candidates(root) {
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

fn scene_grounding_evidence_candidates(root: &Path) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if root.is_file() {
        candidates.push(root.to_path_buf());
        if let Some(parent) = root.parent() {
            candidates.push(parent.join("grounding_evidence.json"));
            candidates.push(parent.join("pre_generation_grounding_evidence.json"));
            if let Some(run_root) = parent.parent() {
                candidates.push(run_root.join("grounding_evidence.json"));
                candidates.push(run_root.join("pre_generation_grounding_evidence.json"));
            }
        }
    } else {
        candidates.push(root.join("grounding_evidence.json"));
        candidates.push(root.join("pre_generation_grounding_evidence.json"));
    }
    candidates
}

fn load_scene_debug_camera(evidence_path: &Path) -> Result<SceneDebugCamera, String> {
    let evidence = read_json_value(evidence_path)?;
    let source_image_path = evidence
        .get("source_image_path")
        .and_then(Value::as_str)
        .map(|value| resolve_artifact_path(value, evidence_path.parent()))
        .ok_or_else(|| format!("{} is missing source_image_path", evidence_path.display()))?;
    let depth_summary_path = evidence
        .get("depth")
        .and_then(|depth| depth.get("artifact_path"))
        .and_then(Value::as_str)
        .map(|value| resolve_artifact_path(value, evidence_path.parent()))
        .ok_or_else(|| format!("{} is missing depth.artifact_path", evidence_path.display()))?;
    let depth_summary = read_json_value(&depth_summary_path)?;
    let sidecar = depth_summary.get("depth_map_sidecar").ok_or_else(|| {
        format!(
            "{} is missing depth_map_sidecar",
            depth_summary_path.display()
        )
    })?;
    let depth_raw_path = sidecar
        .get("raw_path")
        .and_then(Value::as_str)
        .map(|value| resolve_artifact_path(value, depth_summary_path.parent()))
        .or_else(|| {
            sidecar
                .get("relative_raw_path")
                .and_then(Value::as_str)
                .map(|value| resolve_artifact_path(value, depth_summary_path.parent()))
        })
        .ok_or_else(|| {
            format!(
                "{} depth_map_sidecar is missing raw_path",
                depth_summary_path.display()
            )
        })?;
    let intrinsics = parse_depth_debug_intrinsics(sidecar)?;
    let vertical_fov_degrees = sidecar
        .get("vertical_fov_degrees")
        .and_then(Value::as_f64)
        .map(|value| value as f32)
        .or_else(|| {
            evidence
                .get("camera")
                .and_then(|camera| camera.get("vertical_fov_degrees"))
                .and_then(Value::as_f64)
                .map(|value| value as f32)
        })
        .filter(|value| value.is_finite() && *value > 1.0)
        .unwrap_or(60.0);
    let origin = source_origin_xz_from_evidence_value(&evidence).unwrap_or([0.0, 0.0]);
    let camera_height = source_camera_height_from_evidence_value(&evidence).unwrap_or(0.0);
    let transform = scene_debug_camera_transform(origin, camera_height, 0.0);
    Ok(SceneDebugCamera {
        transform,
        vertical_fov_degrees,
        aspect: intrinsics.width as f32 / intrinsics.height.max(1) as f32,
        source_image_path,
        depth_summary_path,
        depth_raw_path,
        intrinsics,
    })
}

fn parse_depth_debug_intrinsics(sidecar: &Value) -> Result<DepthDebugIntrinsics, String> {
    let width = sidecar
        .get("width")
        .and_then(Value::as_u64)
        .ok_or_else(|| "depth sidecar is missing width".to_string())? as usize;
    let height = sidecar
        .get("height")
        .and_then(Value::as_u64)
        .ok_or_else(|| "depth sidecar is missing height".to_string())? as usize;
    let intrinsics = sidecar
        .get("intrinsics")
        .ok_or_else(|| "depth sidecar is missing intrinsics".to_string())?;
    let fx = json_field_f32(intrinsics, "fx").ok_or_else(|| "intrinsics.fx missing".to_string())?;
    let fy = json_field_f32(intrinsics, "fy").ok_or_else(|| "intrinsics.fy missing".to_string())?;
    let cx = json_field_f32(intrinsics, "cx").ok_or_else(|| "intrinsics.cx missing".to_string())?;
    let cy = json_field_f32(intrinsics, "cy").ok_or_else(|| "intrinsics.cy missing".to_string())?;
    if width == 0
        || height == 0
        || ![fx, fy, cx, cy].iter().all(|value| value.is_finite())
        || fx <= 0.0
        || fy <= 0.0
    {
        return Err("depth sidecar intrinsics are invalid".to_string());
    }
    Ok(DepthDebugIntrinsics {
        fx,
        fy,
        cx,
        cy,
        width,
        height,
    })
}

fn depth_debug_cloud_settings() -> CloudSettings {
    CloudSettings {
        sort_mode: SortMode::None,
        color_space: GaussianColorSpace::SrgbRec709Display,
        ..default()
    }
}

fn load_depth_debug_cloud(
    camera: &SceneDebugCamera,
    max_gaussians: usize,
) -> Result<PlanarGaussian3d, String> {
    let raw = fs::read(&camera.depth_raw_path)
        .map_err(|err| format!("read {} failed: {err}", camera.depth_raw_path.display()))?;
    let expected_len = camera.intrinsics.width * camera.intrinsics.height * mem::size_of::<f32>();
    if raw.len() < expected_len {
        return Err(format!(
            "{} is too short for {}x{} f32 depth: {} bytes < {expected_len}",
            camera.depth_raw_path.display(),
            camera.intrinsics.width,
            camera.intrinsics.height,
            raw.len()
        ));
    }
    let source = image::ImageReader::open(&camera.source_image_path)
        .map_err(|err| format!("open {} failed: {err}", camera.source_image_path.display()))?
        .decode()
        .map_err(|err| {
            format!(
                "decode {} failed: {err}",
                camera.source_image_path.display()
            )
        })?
        .to_rgba8();
    let image_width = source.width().max(1);
    let image_height = source.height().max(1);
    let max_gaussians = max_gaussians.clamp(1, DEPTH_DEBUG_MAX_GAUSSIANS);
    let stride = depth_debug_sample_stride(
        camera.intrinsics.width,
        camera.intrinsics.height,
        max_gaussians,
    );
    let mut gaussians = Vec::with_capacity(
        ((camera.intrinsics.width / stride).max(1) * (camera.intrinsics.height / stride).max(1))
            .min(max_gaussians),
    );

    'rows: for y in (0..camera.intrinsics.height).step_by(stride) {
        for x in (0..camera.intrinsics.width).step_by(stride) {
            let offset = (y * camera.intrinsics.width + x) * mem::size_of::<f32>();
            let depth = f32::from_le_bytes([
                raw[offset],
                raw[offset + 1],
                raw[offset + 2],
                raw[offset + 3],
            ]);
            if !depth.is_finite() || depth <= 0.0 {
                continue;
            }
            let world = depth_debug_world_point(x, y, depth, camera.intrinsics, camera.transform);
            if !world.is_finite() {
                continue;
            }
            let image_x = (((x as f32 + 0.5) * image_width as f32
                / camera.intrinsics.width.max(1) as f32)
                .floor() as u32)
                .min(image_width - 1);
            let image_y = (((y as f32 + 0.5) * image_height as f32
                / camera.intrinsics.height.max(1) as f32)
                .floor() as u32)
                .min(image_height - 1);
            let rgba = source.get_pixel(image_x, image_y).0;
            let mut spherical_harmonic = SphericalHarmonicCoefficients::default();
            spherical_harmonic.set(0, rgba[0] as f32 / 255.0);
            spherical_harmonic.set(1, rgba[1] as f32 / 255.0);
            spherical_harmonic.set(2, rgba[2] as f32 / 255.0);
            let radius = depth_debug_gaussian_radius(depth, camera.intrinsics, stride);
            gaussians.push(Gaussian3d {
                position_visibility: [world.x, world.y, world.z, 1.0].into(),
                spherical_harmonic,
                rotation: [1.0, 0.0, 0.0, 0.0].into(),
                scale_opacity: [radius, radius, radius, 0.7].into(),
            });
            if gaussians.len() >= max_gaussians {
                break 'rows;
            }
        }
    }
    if gaussians.is_empty() {
        return Err(format!(
            "no finite positive depth samples in {}",
            camera.depth_raw_path.display()
        ));
    }
    Ok(PlanarGaussian3d::from(gaussians))
}

pub(crate) fn depth_debug_sample_stride(width: usize, height: usize, max_count: usize) -> usize {
    if max_count == 0 {
        return width.max(height).max(1);
    }
    let count = width.saturating_mul(height);
    if count <= max_count {
        return 1;
    }
    ((count as f64 / max_count as f64).sqrt().ceil() as usize).max(1)
}

pub(crate) fn depth_debug_world_point(
    x: usize,
    y: usize,
    depth_m: f32,
    intrinsics: DepthDebugIntrinsics,
    camera_transform: Transform,
) -> Vec3 {
    let camera_x = ((x as f32 + 0.5) - intrinsics.cx) * depth_m / intrinsics.fx;
    let camera_y_down = ((y as f32 + 0.5) - intrinsics.cy) * depth_m / intrinsics.fy;
    camera_transform.transform_point(Vec3::new(camera_x, -camera_y_down, -depth_m))
}

fn depth_debug_gaussian_radius(
    depth_m: f32,
    intrinsics: DepthDebugIntrinsics,
    stride: usize,
) -> f32 {
    let focal = intrinsics.fx.min(intrinsics.fy).max(1.0);
    (depth_m / focal * stride as f32).clamp(0.0025, 0.075)
}

fn scene_debug_camera_transform(origin_xz: [f32; 2], height_m: f32, floor_y: f32) -> Transform {
    let translation = Vec3::new(-origin_xz[0], floor_y + height_m, origin_xz[1]);
    Transform::from_translation(translation).looking_at(translation + Vec3::NEG_Z, Vec3::Y)
}

fn source_origin_xz_from_evidence_value(evidence: &Value) -> Option<[f32; 2]> {
    let objects = evidence.get("objects")?.as_array()?;
    let table_origin = objects
        .iter()
        .find(|object| object_grounding_text(object).contains("table"))
        .and_then(metric_contact_xz_from_object);
    if table_origin.is_some() {
        return table_origin;
    }
    let mut sum = [0.0f32; 2];
    let mut count = 0usize;
    for object in objects {
        if let Some(point) = metric_contact_xz_from_object(object) {
            sum[0] += point[0];
            sum[1] += point[1];
            count += 1;
        }
    }
    (count > 0).then_some([sum[0] / count as f32, sum[1] / count as f32])
}

fn metric_contact_xz_from_object(object: &Value) -> Option<[f32; 2]> {
    let point = object.get("metric_contact_point_m")?.as_array()?;
    if point.len() < 3 {
        return None;
    }
    let x = point[0].as_f64()? as f32;
    let z = point[2].as_f64()? as f32;
    (x.is_finite() && z.is_finite()).then_some([x, z])
}

fn object_grounding_text(object: &Value) -> String {
    let mut text = String::new();
    for key in ["object_id", "instance_id", "reuse_group", "asset_id"] {
        if let Some(value) = object.get(key).and_then(Value::as_str) {
            text.push_str(value);
            text.push(' ');
        }
    }
    if let Some(detection) = object.get("detection") {
        for key in ["label", "source_query"] {
            if let Some(value) = detection.get(key).and_then(Value::as_str) {
                text.push_str(value);
                text.push(' ');
            }
        }
    }
    text.to_ascii_lowercase()
}

fn source_camera_height_from_evidence_value(evidence: &Value) -> Option<f32> {
    let floor = evidence.get("floor")?;
    let normal = floor.get("normal")?.as_array()?;
    if normal.len() < 2 {
        return None;
    }
    let normal_y = normal[1].as_f64()? as f32;
    let distance_m = floor.get("distance_m")?.as_f64()? as f32;
    let residual_ok = floor
        .get("residual_m")
        .and_then(Value::as_f64)
        .map(|value| value <= 0.10)
        .unwrap_or(false);
    let confidence_ok = floor
        .get("confidence")
        .and_then(Value::as_f64)
        .map(|value| value >= 0.72)
        .unwrap_or(true);
    if !normal_y.is_finite() || normal_y.abs() <= 1.0e-5 || !residual_ok || !confidence_ok {
        return None;
    }
    let y_down_at_camera_origin = -distance_m / normal_y;
    (y_down_at_camera_origin.is_finite() && (1.10..=3.80).contains(&y_down_at_camera_origin))
        .then_some(y_down_at_camera_origin)
}

fn read_json_value(path: &Path) -> Result<Value, String> {
    let bytes = fs::read(path).map_err(|err| format!("read {} failed: {err}", path.display()))?;
    serde_json::from_slice(&bytes).map_err(|err| format!("parse {} failed: {err}", path.display()))
}

fn json_field_f32(value: &Value, key: &str) -> Option<f32> {
    value
        .get(key)
        .and_then(Value::as_f64)
        .map(|value| value as f32)
        .filter(|value| value.is_finite())
}

fn resolve_artifact_path(path: &str, base: Option<&Path>) -> PathBuf {
    let raw = PathBuf::from(path);
    if raw.is_absolute() || raw.exists() {
        raw
    } else if let Some(base) = base {
        base.join(raw)
    } else {
        raw
    }
}

fn file_signature(path: &Path) -> String {
    let modified = fs::metadata(path)
        .and_then(|metadata| metadata.modified())
        .ok()
        .and_then(|modified| modified.duration_since(SystemTime::UNIX_EPOCH).ok())
        .map(|duration| duration.as_millis())
        .unwrap_or_default();
    format!("{}#{modified}", path.display())
}

pub(crate) fn draw_scene_camera_frustum(
    gizmos: &mut Gizmos,
    camera: &SceneDebugCamera,
    frustum_length: f32,
) {
    let fov = camera.vertical_fov_degrees.to_radians();
    if !fov.is_finite() || fov <= 0.0 {
        return;
    }
    let aspect = camera.aspect.max(0.1);
    let length = scene_camera_frustum_length(frustum_length);
    let half_height = (fov * 0.5).tan() * length;
    let half_width = half_height * aspect;
    let origin = camera.transform.translation;
    let forward = camera.transform.rotation * -Vec3::Z;
    let right = camera.transform.rotation * Vec3::X;
    let up = camera.transform.rotation * Vec3::Y;
    let center = origin + forward * length;
    let far_corners = [
        center + up * half_height - right * half_width,
        center + up * half_height + right * half_width,
        center - up * half_height + right * half_width,
        center - up * half_height - right * half_width,
    ];
    let color = Color::srgb(1.0, 0.72, 0.18);
    for (a, b) in [(0, 1), (1, 2), (2, 3), (3, 0)] {
        gizmos.line(far_corners[a], far_corners[b], color);
        gizmos.line(origin, far_corners[a], color);
    }
    draw_debug_cross(
        gizmos,
        origin,
        scene_camera_frustum_cross_size(length),
        color,
    );
}

pub(crate) fn scene_camera_frustum_length(frustum_length: f32) -> f32 {
    frustum_length.clamp(0.05, 3.0)
}

pub(crate) fn scene_camera_frustum_cross_size(frustum_length: f32) -> f32 {
    (scene_camera_frustum_length(frustum_length) * 0.025).clamp(0.02, 0.08)
}

fn draw_debug_cross(gizmos: &mut Gizmos, center: Vec3, radius: f32, color: Color) {
    gizmos.line(center - Vec3::X * radius, center + Vec3::X * radius, color);
    gizmos.line(center - Vec3::Y * radius, center + Vec3::Y * radius, color);
    gizmos.line(center - Vec3::Z * radius, center + Vec3::Z * radius, color);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_depth_debug_camera(label: &str, transform: Transform) -> SceneDebugCamera {
        let root = std::env::temp_dir().join(format!(
            "bevy_synth_depth_debug_{label}_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("create temp depth debug dir");

        let source_image_path = root.join("source.png");
        let image = image::RgbaImage::from_pixel(4, 4, image::Rgba([32, 96, 160, 255]));
        image.save(&source_image_path).expect("write source image");

        let depth_raw_path = root.join("depth.raw");
        let mut raw = Vec::new();
        for _ in 0..16 {
            raw.extend_from_slice(&2.0f32.to_le_bytes());
        }
        fs::write(&depth_raw_path, raw).expect("write raw depth");

        SceneDebugCamera {
            transform,
            vertical_fov_degrees: 60.0,
            aspect: 1.0,
            source_image_path,
            depth_summary_path: root.join("depth_summary.json"),
            depth_raw_path,
            intrinsics: DepthDebugIntrinsics {
                fx: 100.0,
                fy: 100.0,
                cx: 1.5,
                cy: 1.5,
                width: 4,
                height: 4,
            },
        }
    }

    #[test]
    fn depth_debug_cloud_uses_no_sort_and_respects_cap() {
        let camera = temp_depth_debug_camera(
            "cap",
            Transform::from_translation(Vec3::new(0.0, 1.0, 0.0))
                .looking_at(Vec3::new(0.0, 1.0, -1.0), Vec3::Y),
        );

        let cloud = load_depth_debug_cloud(&camera, 3).expect("load depth debug cloud");
        assert_eq!(cloud.position_visibility.len(), 3);

        let settings = depth_debug_cloud_settings();
        assert_eq!(settings.sort_mode, SortMode::None);
        assert_eq!(settings.color_space, GaussianColorSpace::SrgbRec709Display);
    }

    #[test]
    fn depth_debug_cloud_is_not_rebuilt_when_editor_camera_moves() {
        let camera = temp_depth_debug_camera(
            "editor_motion",
            Transform::from_translation(Vec3::new(0.0, 1.0, 0.0))
                .looking_at(Vec3::new(0.0, 1.0, -1.0), Vec3::Y),
        );
        let mut app = App::new();
        app.insert_resource(ViewerDebugSettings {
            depth_cloud_overlay: true,
            depth_cloud_max_gaussians: 4,
            ..default()
        });
        app.insert_resource(SceneDepthDebugState {
            evidence_signature: Some("sig".to_string()),
            camera: Some(camera),
            cloud_signature: None,
            cloud_entity: None,
            last_error: None,
        });
        app.insert_resource(Assets::<PlanarGaussian3d>::default());
        let moving_entity = app
            .world_mut()
            .spawn((
                Transform::from_xyz(0.0, 1.0, 5.0),
                GlobalTransform::default(),
            ))
            .id();
        app.add_systems(Update, sync_depth_debug_cloud);

        app.update();
        let first_entity = app
            .world()
            .resource::<SceneDepthDebugState>()
            .cloud_entity
            .expect("depth cloud entity");
        let mut settings_query = app.world_mut().query::<&CloudSettings>();
        let settings = settings_query
            .iter(app.world())
            .next()
            .expect("cloud settings");
        assert_eq!(settings.sort_mode, SortMode::None);

        app.world_mut()
            .entity_mut(moving_entity)
            .insert(Transform::from_xyz(1.0, 2.0, 6.0));
        app.update();

        assert_eq!(
            app.world().resource::<SceneDepthDebugState>().cloud_entity,
            Some(first_entity),
            "moving unrelated camera/editor transforms must not rebuild the scene-depth debug cloud"
        );
    }
}

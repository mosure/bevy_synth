use std::collections::{HashMap, HashSet};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::bsn::normalize_bbox;
use crate::ground_calibration::estimated_floor_from_camera_pitch;
use crate::object_images::image_dimensions_aspect;
use crate::projection_fit::{ProjectionFitReport, fit_grounded_scene_projection};
use crate::*;

#[derive(Clone, Copy, Debug)]
pub struct GroundedSceneLayoutConfig {
    pub camera_height_m: f32,
    pub camera_pitch_down_degrees: f32,
    pub vertical_fov_degrees: f32,
    pub image_aspect: f32,
    pub floor_y: f32,
    pub seating_clearance_m: f32,
    pub scale_policy: SceneScalePolicy,
}

impl Default for GroundedSceneLayoutConfig {
    fn default() -> Self {
        Self {
            camera_height_m: 3.2,
            camera_pitch_down_degrees: 58.0,
            vertical_fov_degrees: 72.0,
            image_aspect: 2.0,
            floor_y: 0.0,
            seating_clearance_m: 0.18,
            scale_policy: SceneScalePolicy::default(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct GroundedSceneLayout {
    pub bsn: String,
    pub placements: Vec<GroundedScenePlacement>,
    pub camera: SceneCamera,
    pub rug_center: [f32; 3],
    pub rug_scale: [f32; 3],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projection_fit: Option<ProjectionFitReport>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct GroundedScenePlacement {
    pub entity_id: String,
    pub asset_id: String,
    pub object_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instance_id: Option<String>,
    pub label: String,
    pub source_bbox: [f32; 4],
    pub contact_pixel: [f32; 2],
    pub ground_point: [f32; 3],
    pub translation: [f32; 3],
    pub rotation_y_degrees: f32,
    #[serde(default)]
    pub asset_yaw_offset_degrees: f32,
    pub scale: [f32; 3],
    pub local_aabb: SceneAssetAabb,
    pub target_footprint_m: [f32; 2],
}

impl GroundedScenePlacement {
    pub fn ground_anchor_max_drift_m(&self) -> f32 {
        let footprint = self.target_footprint_m[0]
            .abs()
            .max(self.target_footprint_m[1].abs())
            .max(0.1);
        let descriptor = format!("{} {}", self.object_id, self.label).to_ascii_lowercase();
        if descriptor.contains("table")
            || descriptor.contains("desk")
            || descriptor.contains("counter")
        {
            (footprint * 0.16).clamp(0.20, 0.45)
        } else if descriptor.contains("chair")
            || descriptor.contains("seat")
            || descriptor.contains("sofa")
            || descriptor.contains("couch")
        {
            (footprint * 0.55).clamp(0.24, 0.56)
        } else {
            (footprint * 0.45).clamp(0.28, 0.75)
        }
    }

    pub fn local_ground_anchor_xz(&self) -> [f32; 2] {
        [
            (self.local_aabb.min[0] + self.local_aabb.max[0]) * 0.5,
            (self.local_aabb.min[2] + self.local_aabb.max[2]) * 0.5,
        ]
    }

    pub fn ground_anchor_world_offset_xz(&self) -> [f32; 2] {
        let anchor = self.local_ground_anchor_xz();
        let scaled_x = anchor[0] * self.scale[0];
        let scaled_z = anchor[1] * self.scale[2];
        let yaw = self.rotation_y_degrees.to_radians();
        let cos = yaw.cos();
        let sin = yaw.sin();
        [
            scaled_x * cos + scaled_z * sin,
            -scaled_x * sin + scaled_z * cos,
        ]
    }

    pub fn translation_for_ground_anchor(&self, floor_y: f32) -> [f32; 3] {
        let offset = self.ground_anchor_world_offset_xz();
        [
            self.ground_point[0] - offset[0],
            floor_y - self.local_aabb.min[1] * self.scale[1],
            self.ground_point[2] - offset[1],
        ]
    }

    pub fn sync_translation_to_ground_anchor(&mut self, floor_y: f32) {
        self.ground_point[1] = floor_y;
        self.translation = self.translation_for_ground_anchor(floor_y);
    }

    pub fn sync_translation_to_current_ground_anchor(&mut self) {
        self.sync_translation_to_ground_anchor(self.ground_point[1]);
    }

    pub fn sync_ground_anchor_from_translation(&mut self, floor_y: f32) {
        let offset = self.ground_anchor_world_offset_xz();
        self.ground_point = [
            self.translation[0] + offset[0],
            floor_y,
            self.translation[2] + offset[1],
        ];
    }

    pub fn sync_ground_anchor_from_current_translation(&mut self) {
        self.sync_ground_anchor_from_translation(self.ground_point[1]);
    }

    pub fn world_ground_anchor(&self) -> [f32; 3] {
        let offset = self.ground_anchor_world_offset_xz();
        [
            self.translation[0] + offset[0],
            self.ground_point[1],
            self.translation[2] + offset[1],
        ]
    }
}

#[derive(Clone, Debug, PartialEq)]
struct ResolvedSceneObjectInstance {
    id: Option<String>,
    bbox: [f32; 4],
    contact_pixel: [f32; 2],
    rotation_hint_degrees: Option<f32>,
    facing_yaw_degrees: Option<f32>,
    side: Option<SceneInstanceSide>,
    slot_index: Option<usize>,
    target_footprint_m: Option<[f32; 2]>,
}

pub fn grounded_scene_bsn(
    manifest: &SceneObjectManifest,
    assets: &[SceneAssetBinding],
) -> SceneResult<String> {
    Ok(grounded_scene_layout_for_manifest(manifest, assets)?.bsn)
}

pub fn grounded_scene_layout_for_manifest(
    manifest: &SceneObjectManifest,
    assets: &[SceneAssetBinding],
) -> SceneResult<GroundedSceneLayout> {
    let mut config = GroundedSceneLayoutConfig::default();
    if let Ok(aspect) = image_dimensions_aspect(Path::new(&manifest.source_scene_path)) {
        config.image_aspect = aspect;
    }
    grounded_scene_layout(manifest, assets, config)
}

pub fn grounded_scene_layout_with_evidence(
    manifest: &SceneObjectManifest,
    assets: &[SceneAssetBinding],
    evidence: &SceneGroundingEvidence,
) -> SceneResult<GroundedSceneLayout> {
    grounded_scene_layout_with_evidence_config(
        manifest,
        assets,
        evidence,
        GroundedSceneLayoutConfig::default(),
    )
}

pub fn grounded_scene_layout_with_evidence_config(
    manifest: &SceneObjectManifest,
    assets: &[SceneAssetBinding],
    evidence: &SceneGroundingEvidence,
    mut config: GroundedSceneLayoutConfig,
) -> SceneResult<GroundedSceneLayout> {
    let manifest = manifest_with_grounding_evidence(manifest, evidence);
    if let Some([width, height]) = evidence
        .camera
        .image_size
        .or_else(|| evidence.depth.as_ref().and_then(|depth| depth.image_size))
    {
        config.image_aspect = width.max(1) as f32 / height.max(1) as f32;
    } else if let Ok(aspect) = image_dimensions_aspect(Path::new(&manifest.source_scene_path)) {
        config.image_aspect = aspect;
    }
    if let Some(vertical_fov) = evidence.camera.vertical_fov_degrees
        && vertical_fov.is_finite()
        && vertical_fov > 1.0
    {
        config.vertical_fov_degrees = vertical_fov.clamp(20.0, 120.0);
    }
    let effective_evidence =
        evidence_with_scene_floor_calibration(evidence, manifest.scene_calibration);
    let grounding_geometry = GroundingGeometry::from_evidence(&effective_evidence, config.floor_y);
    grounded_scene_layout_internal(
        &manifest,
        assets,
        config,
        Some((&grounding_geometry, &effective_evidence)),
    )
}

fn evidence_with_scene_floor_calibration(
    evidence: &SceneGroundingEvidence,
    calibration: Option<SceneCalibration>,
) -> SceneGroundingEvidence {
    let Some(calibration) = calibration else {
        return evidence.clone();
    };
    let Some(camera_pitch_degrees) = calibration
        .camera_pitch_degrees
        .filter(|value| value.is_finite() && value.abs() > 1.0)
    else {
        return evidence.clone();
    };
    let camera_height_m = source_camera_height_from_floor(&evidence.floor)
        .or_else(|| {
            evidence
                .floor
                .distance_m
                .is_finite()
                .then_some(-evidence.floor.distance_m)
        })
        .filter(|value| value.is_finite() && (0.8..=4.5).contains(value))
        .unwrap_or(1.65);
    let confidence = evidence.floor.confidence.unwrap_or(0.72);
    let residual_m = evidence.floor.residual_m.unwrap_or(0.12);
    let mut out = evidence.clone();
    out.floor = estimated_floor_from_camera_pitch(
        camera_height_m,
        Some(camera_pitch_degrees),
        confidence,
        residual_m,
    );
    out
}

pub fn manifest_with_grounding_evidence(
    manifest: &SceneObjectManifest,
    evidence: &SceneGroundingEvidence,
) -> SceneObjectManifest {
    let mut out = manifest.clone();
    for object in &mut out.objects {
        if let Some(object_evidence) = best_object_evidence(evidence, &object.id, None) {
            apply_object_grounding_evidence(object, object_evidence);
        }
        for instance in &mut object.instances {
            let instance_id = instance.id.as_deref();
            if let Some(object_evidence) = best_object_evidence(evidence, &object.id, instance_id) {
                if let Some(bbox) = object_evidence_bbox(object_evidence) {
                    instance.bbox = normalize_bbox(bbox);
                }
                if let Some(contact) = object_evidence
                    .contact_pixel
                    .or_else(|| object_evidence.detection.as_ref().and_then(|d| d.point))
                {
                    instance.contact = Some(normalize_contact_pixel(contact));
                }
                if let Some(footprint) = object_evidence.target_footprint_m {
                    instance.target_footprint_m = Some(sane_footprint(footprint));
                }
            }
        }
        dedupe_object_instances_by_bbox(&mut object.instances);
        let mut existing_instance_ids = object
            .instances
            .iter()
            .filter_map(|instance| instance.id.clone())
            .collect::<HashSet<_>>();
        let mut existing_instance_bboxes = object
            .instances
            .iter()
            .map(|instance| normalize_bbox(instance.bbox))
            .collect::<Vec<_>>();
        for object_evidence in &evidence.objects {
            if object_evidence.object_id != object.id {
                continue;
            }
            let Some(instance_id) = object_evidence.instance_id.as_deref() else {
                continue;
            };
            if existing_instance_ids.contains(instance_id) {
                continue;
            }
            let Some(detection) = object_evidence.detection.as_ref() else {
                continue;
            };
            let bbox =
                normalize_bbox(object_evidence_bbox(object_evidence).unwrap_or(detection.bbox));
            if duplicate_instance_bbox(bbox, &existing_instance_bboxes) {
                continue;
            }
            existing_instance_ids.insert(instance_id.to_string());
            existing_instance_bboxes.push(bbox);
            object.instances.push(SceneObjectInstanceSpec {
                id: object_evidence.instance_id.clone(),
                bbox,
                contact: object_evidence
                    .contact_pixel
                    .or(detection.point)
                    .map(normalize_contact_pixel)
                    .or_else(|| Some(normalize_contact_pixel(bbox_bottom_center(detection.bbox)))),
                rotation_hint_degrees: None,
                facing_yaw_degrees: None,
                side: Some(SceneInstanceSide::Unknown),
                slot_index: None,
                target_footprint_m: object_evidence
                    .target_footprint_m
                    .or(object.target_footprint_m)
                    .map(sane_footprint),
            });
        }
        if object.instances.is_empty() {
            object.instance_count = object.instance_count.max(1);
        } else {
            object.instance_count = object.instances.len();
        }
    }
    out
}

fn dedupe_object_instances_by_bbox(instances: &mut Vec<SceneObjectInstanceSpec>) {
    let mut existing = Vec::new();
    instances.retain(|instance| {
        let bbox = normalize_bbox(instance.bbox);
        if duplicate_instance_bbox(bbox, &existing) {
            false
        } else {
            existing.push(bbox);
            true
        }
    });
}

pub fn manifest_grounding_evidence(manifest: &SceneObjectManifest) -> SceneGroundingEvidence {
    let image_size = image::image_dimensions(&manifest.source_scene_path)
        .ok()
        .map(|(width, height)| [width, height]);
    let mut detections = Vec::new();
    let mut objects = Vec::new();
    for object in &manifest.objects {
        for instance in resolved_object_instances(object) {
            let source_query = object.label.clone();
            let detection = Detection {
                label: object.label.clone(),
                bbox: instance.bbox,
                point: Some(instance.contact_pixel),
                confidence: None,
                source_query,
            };
            detections.push(detection.clone());
            objects.push(ObjectGroundingEvidence {
                object_id: object.id.clone(),
                instance_id: instance.id,
                reuse_group: object.reuse_group.clone(),
                detection: Some(detection),
                mask: None,
                asset_id: None,
                contact_pixel: Some(instance.contact_pixel),
                depth_stats: None,
                candidate_floor_contact_rays: Vec::new(),
                metric_contact_point_m: None,
                target_footprint_m: None,
                provenance: vec!["manifest_fallback".to_string()],
            });
        }
    }
    SceneGroundingEvidence {
        source_image_path: manifest.source_scene_path.clone(),
        depth: None,
        segmentation: None,
        detections,
        camera: EstimatedCamera {
            image_size,
            ..EstimatedCamera::default()
        },
        floor: EstimatedFloorPlane::default(),
        objects,
    }
}

fn best_object_evidence<'a>(
    evidence: &'a SceneGroundingEvidence,
    object_id: &str,
    instance_id: Option<&str>,
) -> Option<&'a ObjectGroundingEvidence> {
    evidence.objects.iter().find(|object| {
        object.object_id == object_id
            && match (instance_id, object.instance_id.as_deref()) {
                (Some(expected), Some(actual)) => expected == actual,
                (None, None) => true,
                (None, Some(_)) => false,
                (Some(_), None) => false,
            }
    })
}

fn apply_object_grounding_evidence(
    object: &mut SceneObjectSpec,
    evidence: &ObjectGroundingEvidence,
) {
    if let Some(bbox) = object_evidence_bbox(evidence) {
        object.bbox = normalize_bbox(bbox);
    }
    if let Some(detection) = evidence.detection.as_ref()
        && object.label.trim().is_empty()
    {
        object.label = detection.label.clone();
    }
    if let Some(reuse_group) = evidence.reuse_group.as_ref()
        && object.reuse_group.as_ref() != Some(reuse_group)
    {
        object.reuse_group = Some(reuse_group.clone());
    }
    if let Some(footprint) = evidence.target_footprint_m {
        object.target_footprint_m = Some(sane_footprint(footprint));
    }
}

fn object_evidence_bbox(evidence: &ObjectGroundingEvidence) -> Option<[f32; 4]> {
    evidence
        .mask
        .as_ref()
        .map(|mask| mask.bbox)
        .or_else(|| evidence.detection.as_ref().map(|detection| detection.bbox))
}

fn duplicate_instance_bbox(candidate: [f32; 4], existing: &[[f32; 4]]) -> bool {
    existing.iter().any(|bbox| {
        bbox_iou(candidate, *bbox) >= 0.86 || {
            let center_delta = distance2(bbox_center(candidate), bbox_center(*bbox));
            let area_delta = safe_log2_ratio(bbox_area(candidate), bbox_area(*bbox)).abs();
            center_delta <= 0.035 && area_delta <= 0.35
        }
    })
}

fn bbox_center(bbox: [f32; 4]) -> [f32; 2] {
    [(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5]
}

fn bbox_area(bbox: [f32; 4]) -> f32 {
    ((bbox[2] - bbox[0]).abs() * (bbox[3] - bbox[1]).abs()).max(1.0e-6)
}

fn bbox_iou(left: [f32; 4], right: [f32; 4]) -> f32 {
    let x = (left[2].min(right[2]) - left[0].max(right[0])).max(0.0);
    let y = (left[3].min(right[3]) - left[1].max(right[1])).max(0.0);
    let intersection = x * y;
    let union = bbox_area(left) + bbox_area(right) - intersection;
    if union <= 1.0e-8 {
        0.0
    } else {
        (intersection / union).clamp(0.0, 1.0)
    }
}

fn safe_log2_ratio(observed: f32, expected: f32) -> f32 {
    (observed.max(1.0e-8) / expected.max(1.0e-8)).log2()
}

fn distance2(left: [f32; 2], right: [f32; 2]) -> f32 {
    let dx = left[0] - right[0];
    let dy = left[1] - right[1];
    (dx * dx + dy * dy).sqrt()
}

#[derive(Clone, Debug, Default)]
struct GroundingGeometry {
    contact_points_by_instance: HashMap<(String, Option<String>), [f32; 3]>,
    source_origin_xz: Option<[f32; 2]>,
    source_camera_height_m: Option<f32>,
    source_vertical_fov_degrees: Option<f32>,
}

impl GroundingGeometry {
    fn from_evidence(evidence: &SceneGroundingEvidence, floor_y: f32) -> Self {
        let mut out = Self::default();
        for object in &evidence.objects {
            let Some(point) = table_center_point_from_evidence(object, evidence)
                .or_else(|| floor_contact_point_from_evidence(object, &evidence.floor))
                .or(object.metric_contact_point_m)
            else {
                continue;
            };
            if !point.iter().all(|value| value.is_finite()) {
                continue;
            }
            out.contact_points_by_instance.insert(
                (object.object_id.clone(), object.instance_id.clone()),
                [point[0], floor_y, point[2]],
            );
        }
        out.source_origin_xz = source_origin_xz_from_evidence(evidence, &out);
        out.source_camera_height_m = source_camera_height_from_floor(&evidence.floor);
        out.source_vertical_fov_degrees = evidence
            .camera
            .vertical_fov_degrees
            .or_else(|| {
                evidence
                    .depth
                    .as_ref()
                    .and_then(|depth| depth.vertical_fov_degrees)
            })
            .filter(|value| value.is_finite() && *value > 1.0);
        out
    }

    fn contact_point(
        &self,
        object: &SceneObjectSpec,
        instance: &ResolvedSceneObjectInstance,
    ) -> Option<[f32; 3]> {
        let instance_key = (object.id.clone(), instance.id.clone());
        self.contact_points_by_instance
            .get(&instance_key)
            .copied()
            .or_else(|| {
                self.contact_points_by_instance
                    .get(&(object.id.clone(), None))
                    .copied()
            })
    }
}

fn source_origin_xz_from_evidence(
    evidence: &SceneGroundingEvidence,
    geometry: &GroundingGeometry,
) -> Option<[f32; 2]> {
    evidence
        .objects
        .iter()
        .filter(|object| object_grounding_is_table_like(object))
        .find_map(|object| {
            table_center_point_from_evidence(object, evidence)
                .or_else(|| floor_contact_point_from_evidence(object, &evidence.floor))
                .or(object.metric_contact_point_m)
        })
        .map(|point| [point[0], point[2]])
        .or_else(|| {
            let mut sum = [0.0f32; 2];
            let mut count = 0usize;
            for point in geometry.contact_points_by_instance.values() {
                if point[0].is_finite() && point[2].is_finite() {
                    sum[0] += point[0];
                    sum[1] += point[2];
                    count += 1;
                }
            }
            (count > 0).then_some([sum[0] / count as f32, sum[1] / count as f32])
        })
}

fn source_camera_height_from_floor(floor: &EstimatedFloorPlane) -> Option<f32> {
    if !estimated_floor_plane_is_valid(floor) || floor.normal[1].abs() <= 1.0e-5 {
        return None;
    }
    let residual_ok = floor
        .residual_m
        .filter(|value| value.is_finite())
        .is_some_and(|value| value <= 0.10);
    let confidence_ok = floor
        .confidence
        .filter(|value| value.is_finite())
        .is_none_or(|value| value >= 0.72);
    if !residual_ok || !confidence_ok {
        return None;
    }
    let normal_len = floor
        .normal
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
        .max(1.0e-5);
    let camera_height = -floor.distance_m / normal_len;
    (camera_height.is_finite() && (1.10..=3.80).contains(&camera_height)).then_some(camera_height)
}

fn source_camera_from_grounding_geometry(
    geometry: &GroundingGeometry,
    placements: &[GroundedScenePlacement],
    config: GroundedSceneLayoutConfig,
    metric_frame: Option<MetricSceneFrame>,
) -> Option<SceneCamera> {
    let camera_height = geometry.source_camera_height_m?;
    let (center, size) = placement_bounds_for_camera(placements)?;
    let y = config.floor_y + camera_height;
    let requested_pitch_degrees = metric_frame
        .and_then(|frame| frame.camera_pitch_degrees)
        .map(f32::abs)
        .filter(|value| value.is_finite() && *value > 1.0)
        .unwrap_or(config.camera_pitch_down_degrees)
        .clamp(5.0, 80.0);
    let yaw_degrees = metric_frame
        .and_then(|frame| frame.camera_yaw_degrees)
        .filter(|value| value.is_finite())
        .unwrap_or(0.0);
    let vertical_fov_degrees = geometry
        .source_vertical_fov_degrees
        .unwrap_or(config.vertical_fov_degrees)
        .clamp(20.0, 120.0);
    let horizontal = source_camera_horizontal_distance_for_bounds(
        size,
        vertical_fov_degrees,
        config.image_aspect,
        camera_height,
        requested_pitch_degrees,
    );
    let yaw = yaw_degrees.to_radians();
    let focus = [center[0], config.floor_y, center[1]];
    let translation = [
        focus[0] + yaw.sin() * horizontal,
        y,
        focus[2] + yaw.cos() * horizontal,
    ];
    let vertical = (translation[1] - focus[1]).abs().max(0.05);
    let radius = (horizontal * horizontal + vertical * vertical).sqrt();
    let pitch_degrees = vertical
        .atan2(horizontal.max(0.05))
        .to_degrees()
        .clamp(5.0, 80.0);
    let pitch_degrees = pitch_degrees.min(requested_pitch_degrees.max(5.0));
    Some(SceneCamera {
        translation,
        focus,
        yaw: Some(yaw_degrees),
        pitch: Some(pitch_degrees),
        radius: Some(radius),
        vertical_fov_degrees: Some(vertical_fov_degrees),
    })
}

fn placement_bounds_for_camera(
    placements: &[GroundedScenePlacement],
) -> Option<([f32; 2], [f32; 2])> {
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_z = f32::INFINITY;
    let mut max_z = f32::NEG_INFINITY;
    for placement in placements {
        let half_x = placement.target_footprint_m[0].abs().max(0.25) * 0.5;
        let half_z = placement.target_footprint_m[1].abs().max(0.25) * 0.5;
        min_x = min_x.min(placement.ground_point[0] - half_x);
        max_x = max_x.max(placement.ground_point[0] + half_x);
        min_z = min_z.min(placement.ground_point[2] - half_z);
        max_z = max_z.max(placement.ground_point[2] + half_z);
    }
    if !min_x.is_finite() {
        return None;
    }
    let center = [(min_x + max_x) * 0.5, (min_z + max_z) * 0.5];
    let size = [(max_x - min_x).max(0.5), (max_z - min_z).max(0.5)];
    Some((center, size))
}

fn source_camera_horizontal_distance_for_bounds(
    size: [f32; 2],
    vertical_fov_degrees: f32,
    aspect: f32,
    camera_height: f32,
    requested_pitch_degrees: f32,
) -> f32 {
    let fov_y = vertical_fov_degrees.clamp(20.0, 120.0).to_radians();
    let fov_x = 2.0 * ((fov_y * 0.5).tan() * aspect.max(0.5)).atan();
    let fit_width = size[0] * 0.55 / (fov_x * 0.5).tan().max(0.1);
    let fit_depth = size[1] * 0.85;
    let pitch_distance = camera_height / requested_pitch_degrees.to_radians().tan().max(0.1);
    fit_width
        .max(fit_depth)
        .max(pitch_distance)
        .max(2.5)
        .clamp(0.5, 14.0)
}

fn table_center_point_from_evidence(
    object: &ObjectGroundingEvidence,
    evidence: &SceneGroundingEvidence,
) -> Option<[f32; 3]> {
    if !object_grounding_is_table_like(object) {
        return None;
    }
    let detection = object.detection.as_ref()?;
    let depth = object
        .depth_stats
        .map(|stats| stats.median_m)
        .filter(|value| value.is_finite() && *value > 0.0)?;
    let intrinsics = source_camera_intrinsics_from_evidence(evidence)?;
    let width = intrinsics.width.max(1) as f32;
    let height = intrinsics.height.max(1) as f32;
    let center = bbox_center(normalize_bbox(
        object_evidence_bbox(object).unwrap_or(detection.bbox),
    ));
    let pixel_x = center[0] * (width - 1.0).max(1.0);
    let pixel_y = center[1] * (height - 1.0).max(1.0);
    Some([
        (pixel_x - intrinsics.cx) * depth / intrinsics.fx.max(1.0e-5),
        (pixel_y - intrinsics.cy) * depth / intrinsics.fy.max(1.0e-5),
        depth,
    ])
}

fn object_grounding_is_table_like(object: &ObjectGroundingEvidence) -> bool {
    let descriptor = format!(
        "{} {}",
        object.object_id,
        object
            .detection
            .as_ref()
            .map(|detection| detection.label.as_str())
            .unwrap_or_default()
    )
    .to_ascii_lowercase();
    descriptor.contains("table") || descriptor.contains("desk") || descriptor.contains("counter")
}

pub(crate) fn floor_contact_point_from_evidence(
    object: &ObjectGroundingEvidence,
    floor: &EstimatedFloorPlane,
) -> Option<[f32; 3]> {
    if !estimated_floor_plane_is_valid(floor) {
        return None;
    }
    object
        .candidate_floor_contact_rays
        .iter()
        .find_map(|ray| ray_floor_intersection(*ray, floor))
}

fn estimated_floor_plane_is_valid(floor: &EstimatedFloorPlane) -> bool {
    let normal_len_sq = floor.normal.iter().map(|value| value * value).sum::<f32>();
    let residual_ok = floor
        .residual_m
        .filter(|value| value.is_finite())
        .is_some_and(|value| value <= 0.18);
    normal_len_sq.is_finite() && normal_len_sq > 0.25 && floor.distance_m.is_finite() && residual_ok
}

fn ray_floor_intersection(ray: [f32; 3], floor: &EstimatedFloorPlane) -> Option<[f32; 3]> {
    if !ray.iter().all(|value| value.is_finite()) {
        return None;
    }
    let denom = floor.normal[0] * ray[0] + floor.normal[1] * ray[1] + floor.normal[2] * ray[2];
    if !denom.is_finite() || denom.abs() < 1.0e-5 {
        return None;
    }
    let t = -floor.distance_m / denom;
    if !t.is_finite() || t <= 0.0 {
        return None;
    }
    Some([ray[0] * t, ray[1] * t, ray[2] * t])
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct MetricSceneFrame {
    pub(crate) table_axis_degrees: f32,
    pub(crate) table_size_m: [f32; 2],
    pub(crate) seating_clearance_m: f32,
    pub(crate) camera_yaw_degrees: Option<f32>,
    pub(crate) camera_pitch_degrees: Option<f32>,
    pub(crate) camera_radius_m: Option<f32>,
    pub(crate) vertical_fov_degrees: Option<f32>,
}

impl MetricSceneFrame {
    fn from_manifest(
        manifest: &SceneObjectManifest,
        config: GroundedSceneLayoutConfig,
    ) -> Option<Self> {
        let calibration = manifest.scene_calibration?;
        Some(Self {
            table_axis_degrees: finite_or(calibration.table_axis_degrees, 0.0),
            table_size_m: sane_footprint(calibration.table_size_m.unwrap_or([3.2, 1.2])),
            seating_clearance_m: config.seating_clearance_m.clamp(0.04, 0.60),
            camera_yaw_degrees: calibration
                .camera_yaw_degrees
                .filter(|value| value.is_finite()),
            camera_pitch_degrees: calibration
                .camera_pitch_degrees
                .filter(|value| value.is_finite()),
            camera_radius_m: calibration
                .camera_radius_m
                .filter(|value| value.is_finite() && *value > 0.0),
            vertical_fov_degrees: calibration
                .vertical_fov_degrees
                .filter(|value| value.is_finite() && *value > 0.0),
        })
    }

    fn table_point(&self) -> [f32; 3] {
        [0.0, 0.0, 0.0]
    }

    pub(crate) fn side_point(
        &self,
        side: SceneInstanceSide,
        slot_index: usize,
        slot_count: usize,
        target_footprint: [f32; 2],
    ) -> [f32; 3] {
        let table_width = self.table_size_m[0].max(0.5);
        let table_length = self.table_size_m[1].max(0.5);
        let object_depth = target_footprint[1].max(target_footprint[0] * 0.75).max(0.2);
        let clearance = self
            .seating_clearance_m
            .max((object_depth * 0.20).clamp(0.08, 0.28));
        let image_side_sign = self.camera_yaw_degrees.map(|yaw| {
            if yaw.to_radians().cos() >= 0.0 {
                1.0
            } else {
                -1.0
            }
        });
        let slot_count = slot_count.max(1);
        let slot_t = if slot_count <= 1 {
            0.0
        } else {
            (slot_index.min(slot_count - 1) as f32 + 0.5) / slot_count as f32 - 0.5
        };
        let local = match side {
            SceneInstanceSide::Left => [
                (-table_width * 0.5 - object_depth * 0.5 - clearance)
                    * image_side_sign.unwrap_or(1.0),
                slot_t * table_length * 0.88,
            ],
            SceneInstanceSide::Right => [
                (table_width * 0.5 + object_depth * 0.5 + clearance)
                    * image_side_sign.unwrap_or(1.0),
                slot_t * table_length * 0.88,
            ],
            SceneInstanceSide::Near | SceneInstanceSide::Foot => [
                slot_t * table_width * 0.80,
                if let Some(sign) = image_side_sign {
                    (table_length * 0.5 + object_depth * 0.5 + clearance) * sign
                } else {
                    -table_length * 0.5 - object_depth * 0.5 - clearance
                },
            ],
            SceneInstanceSide::Far | SceneInstanceSide::Head => [
                slot_t * table_width * 0.80,
                if let Some(sign) = image_side_sign {
                    (-table_length * 0.5 - object_depth * 0.5 - clearance) * sign
                } else {
                    table_length * 0.5 + object_depth * 0.5 + clearance
                },
            ],
            SceneInstanceSide::Unknown => [0.0, 0.0],
        };
        rotate_table_frame_point(local, self.table_axis_degrees)
    }
}

pub fn grounded_scene_layout(
    manifest: &SceneObjectManifest,
    assets: &[SceneAssetBinding],
    config: GroundedSceneLayoutConfig,
) -> SceneResult<GroundedSceneLayout> {
    grounded_scene_layout_internal(manifest, assets, config, None)
}

fn grounded_scene_layout_internal(
    manifest: &SceneObjectManifest,
    assets: &[SceneAssetBinding],
    config: GroundedSceneLayoutConfig,
    grounding_geometry: Option<(&GroundingGeometry, &SceneGroundingEvidence)>,
) -> SceneResult<GroundedSceneLayout> {
    if manifest.objects.is_empty() {
        return Err(SceneError::Validation(
            "grounded scene layout requires at least one object".to_string(),
        ));
    }
    if assets.is_empty() {
        return Err(SceneError::Validation(
            "grounded scene layout requires at least one asset binding".to_string(),
        ));
    }

    let asset_by_object = assets
        .iter()
        .map(|asset| (asset.object_id.as_str(), asset))
        .collect::<HashMap<_, _>>();
    let metric_frame = MetricSceneFrame::from_manifest(manifest, config);
    let table_contact = manifest
        .objects
        .iter()
        .find(|object| is_table_like(object))
        .and_then(|object| {
            resolved_object_instances(object)
                .into_iter()
                .next()
                .map(|instance| {
                    object_contact_point_with_geometry(
                        object,
                        &instance,
                        config,
                        grounding_geometry.map(|(geometry, _)| geometry),
                    )
                })
        })
        .transpose()?;

    let mut raw = Vec::new();
    for object in &manifest.objects {
        let asset = asset_by_object.get(object.id.as_str()).ok_or_else(|| {
            SceneError::Validation(format!(
                "missing asset binding for scene object `{}`",
                object.id
            ))
        })?;
        for instance in resolved_object_instances(object) {
            let contact = object_contact_point_with_geometry(
                object,
                &instance,
                config,
                grounding_geometry.map(|(geometry, _)| geometry),
            )?;
            raw.push((object, *asset, instance, contact));
        }
    }
    let side_slots = metric_side_slots(&raw);

    let center_x = table_contact.map(|point| point[0]).unwrap_or_else(|| {
        raw.iter().map(|(_, _, _, point)| point[0]).sum::<f32>() / raw.len() as f32
    });
    let center_z = table_contact.map(|point| point[2]).unwrap_or_else(|| {
        raw.iter().map(|(_, _, _, point)| point[2]).sum::<f32>() / raw.len() as f32
    });
    let table_centered = Some([0.0, config.floor_y, 0.0]);

    let mut placements = Vec::with_capacity(raw.len());
    for (raw_index, (object, asset, instance, contact)) in raw.into_iter().enumerate() {
        let local_aabb = asset.local_aabb.unwrap_or(SceneAssetAabb {
            min: [-0.5, 0.0, -0.5],
            max: [0.5, 1.0, 0.5],
        });
        let asset_frame = scene_asset_frame(asset, object, local_aabb);
        let target_footprint = target_footprint_m(object, &instance, asset_frame, metric_frame);
        let scale = asset_scale_for_footprint(
            object,
            local_aabb,
            target_footprint,
            asset_frame,
            config.scale_policy,
        );
        let depth_ground_point = grounding_geometry
            .and_then(|(geometry, _)| geometry.contact_point(object, &instance))
            .map(|point| {
                source_metric_delta_to_layout(
                    [point[0] - center_x, point[2] - center_z],
                    config.floor_y,
                )
            });
        let ground_point = depth_ground_point
            .or_else(|| {
                metric_ground_point(
                    object,
                    &instance,
                    target_footprint,
                    metric_frame,
                    side_slots.get(&raw_index).copied(),
                )
            })
            .unwrap_or([contact[0] - center_x, config.floor_y, contact[2] - center_z]);
        let instance_yaw_degrees = instance
            .rotation_hint_degrees
            .or(instance.facing_yaw_degrees)
            .or(object.rotation_hint_degrees)
            .unwrap_or_else(|| {
                grounded_yaw_degrees(
                    object,
                    &instance,
                    ground_point,
                    table_centered,
                    metric_frame,
                )
            });
        let rotation_y_degrees =
            canonical_spawn_yaw_degrees(instance_yaw_degrees, asset_frame.yaw_offset_degrees, 0.0);
        let entity_id = if let Some(instance_id) = instance.id.as_deref() {
            sanitize_bsn_identifier(&format!("{}_{}", object.id, instance_id))
        } else {
            sanitize_bsn_identifier(&object.id)
        };
        let mut placement = GroundedScenePlacement {
            entity_id,
            asset_id: asset.asset_id.clone(),
            object_id: object.id.clone(),
            instance_id: instance.id.clone(),
            label: object.label.clone(),
            source_bbox: instance.bbox,
            contact_pixel: instance.contact_pixel,
            ground_point,
            translation: [0.0; 3],
            rotation_y_degrees,
            asset_yaw_offset_degrees: asset_frame.yaw_offset_degrees,
            scale,
            local_aabb,
            target_footprint_m: target_footprint,
        };
        placement.sync_translation_to_ground_anchor(config.floor_y);
        placements.push(placement);
    }
    enforce_scale_policy(&mut placements, config.floor_y, config.scale_policy);
    normalize_repeated_asset_scales(&mut placements, config.floor_y);

    let camera = grounding_geometry
        .and_then(|(geometry, _)| {
            source_camera_from_grounding_geometry(geometry, &placements, config, metric_frame)
        })
        .unwrap_or_else(|| grounded_camera_from_placements(&placements, config, metric_frame));
    let projection_fit = grounding_geometry.and_then(|(_, evidence)| {
        fit_grounded_scene_projection(&mut placements, &camera, evidence, config.floor_y)
    });
    enforce_scale_policy(&mut placements, config.floor_y, config.scale_policy);
    normalize_repeated_asset_scales(&mut placements, config.floor_y);
    let (rug_center, rug_scale) = rug_from_placements(&placements, config.floor_y);
    let bsn = grounded_bsn_text(assets, &placements, rug_center, rug_scale, &camera);
    let parsed = parse_scene_bsn(&bsn, assets)?;
    if parsed.placements.len() != placements.len() {
        return Err(SceneError::Validation(
            "grounded BSN placement count changed during parse".to_string(),
        ));
    }
    Ok(GroundedSceneLayout {
        bsn,
        placements,
        camera,
        rug_center,
        rug_scale,
        projection_fit,
    })
}

fn source_metric_delta_to_layout(delta_xz: [f32; 2], floor_y: f32) -> [f32; 3] {
    [delta_xz[0], floor_y, -delta_xz[1]]
}

fn object_contact_point_with_geometry(
    object: &SceneObjectSpec,
    instance: &ResolvedSceneObjectInstance,
    config: GroundedSceneLayoutConfig,
    grounding_geometry: Option<&GroundingGeometry>,
) -> SceneResult<[f32; 3]> {
    if let Some(point) =
        grounding_geometry.and_then(|geometry| geometry.contact_point(object, instance))
    {
        return Ok(point);
    }
    object_contact_point(object, instance, config)
}

fn object_contact_point(
    object: &SceneObjectSpec,
    instance: &ResolvedSceneObjectInstance,
    config: GroundedSceneLayoutConfig,
) -> SceneResult<[f32; 3]> {
    let [u, v] = instance.contact_pixel;
    let u = u.clamp(0.0, 1.0);
    let v = v.clamp(0.02, 0.995);
    floor_intersection_from_normalized_pixel(u, v, config).ok_or_else(|| {
        SceneError::Validation(format!(
            "object `{}` bottom point did not intersect the estimated ground plane",
            object.id
        ))
    })
}

fn resolved_object_instances(object: &SceneObjectSpec) -> Vec<ResolvedSceneObjectInstance> {
    if !object.instances.is_empty() {
        return object
            .instances
            .iter()
            .enumerate()
            .map(|(index, instance)| {
                let bbox = normalize_bbox(instance.bbox);
                ResolvedSceneObjectInstance {
                    id: instance
                        .id
                        .clone()
                        .filter(|value| !value.trim().is_empty())
                        .or_else(|| Some(format!("{:03}", index + 1))),
                    bbox,
                    contact_pixel: instance
                        .contact
                        .map(normalize_contact_pixel)
                        .unwrap_or_else(|| bbox_bottom_center(bbox)),
                    rotation_hint_degrees: instance.rotation_hint_degrees,
                    facing_yaw_degrees: instance.facing_yaw_degrees,
                    side: instance.side,
                    slot_index: instance.slot_index,
                    target_footprint_m: instance.target_footprint_m,
                }
            })
            .collect();
    }

    let instance_count = object.instance_count.max(1);
    (0..instance_count)
        .map(|index| {
            let bbox = instance_bbox(object, index, instance_count);
            ResolvedSceneObjectInstance {
                id: if instance_count == 1 {
                    None
                } else {
                    Some(format!("{:03}", index + 1))
                },
                bbox,
                contact_pixel: bbox_bottom_center(bbox),
                rotation_hint_degrees: None,
                facing_yaw_degrees: None,
                side: None,
                slot_index: None,
                target_footprint_m: None,
            }
        })
        .collect()
}

pub(crate) fn bbox_bottom_center(bbox: [f32; 4]) -> [f32; 2] {
    [
        ((bbox[0] + bbox[2]) * 0.5).clamp(0.0, 1.0),
        bbox[3].clamp(0.0, 1.0),
    ]
}

fn normalize_contact_pixel(value: [f32; 2]) -> [f32; 2] {
    [value[0].clamp(0.0, 1.0), value[1].clamp(0.0, 1.0)]
}

fn floor_intersection_from_normalized_pixel(
    u: f32,
    v: f32,
    config: GroundedSceneLayoutConfig,
) -> Option<[f32; 3]> {
    let tan_half = (config.vertical_fov_degrees.to_radians() * 0.5).tan();
    let x = (2.0 * u - 1.0) * config.image_aspect.max(0.1) * tan_half;
    let y = (1.0 - 2.0 * v) * tan_half;
    let z = 1.0;
    let pitch = config.camera_pitch_down_degrees.to_radians();
    let cos = pitch.cos();
    let sin = pitch.sin();
    let ray = [x, y * cos - z * sin, y * sin + z * cos];
    if ray[1] >= -1.0e-5 {
        return None;
    }
    let t = (config.floor_y - config.camera_height_m) / ray[1];
    Some([ray[0] * t, config.floor_y, ray[2] * t])
}

fn instance_bbox(
    object: &SceneObjectSpec,
    instance_index: usize,
    instance_count: usize,
) -> [f32; 4] {
    let bbox = normalize_bbox(object.bbox);
    if instance_count <= 1 {
        return bbox;
    }
    let width = (bbox[2] - bbox[0]).max(1.0e-5);
    let step = width / instance_count as f32;
    let x0 = bbox[0] + step * instance_index as f32;
    let x1 = if instance_index + 1 == instance_count {
        bbox[2]
    } else {
        x0 + step
    };
    [x0, bbox[1], x1, bbox[3]]
}

fn target_footprint_m(
    object: &SceneObjectSpec,
    instance: &ResolvedSceneObjectInstance,
    asset_frame: SceneAssetFrame,
    metric_frame: Option<MetricSceneFrame>,
) -> [f32; 2] {
    if let Some(footprint) = instance.target_footprint_m {
        return sane_footprint(footprint);
    }
    if let Some(footprint) = object.target_footprint_m {
        return sane_footprint(footprint);
    }
    if is_table_like(object)
        && let Some(frame) = metric_frame
    {
        return sane_footprint(frame.table_size_m);
    }
    if let Some(footprint) = asset_frame.footprint_m {
        return sane_footprint(footprint);
    }
    let descriptor = object_descriptor(object);
    let bbox = normalize_bbox(object.bbox);
    let width = (bbox[2] - bbox[0]).max(0.01);
    if descriptor.contains("sofa")
        || descriptor.contains("couch")
        || descriptor.contains("sectional")
    {
        let length = if width > 0.8 { 4.8 } else { 3.4 };
        [length, 2.2]
    } else if descriptor.contains("conference") && descriptor.contains("table") {
        [3.2, 1.15]
    } else if descriptor.contains("table") {
        [1.8, 0.95]
    } else if descriptor.contains("chair") {
        [0.58, 0.62]
    } else {
        [1.0, 1.0]
    }
}

pub(crate) fn sane_footprint(value: [f32; 2]) -> [f32; 2] {
    [value[0].clamp(0.1, 12.0), value[1].clamp(0.1, 12.0)]
}

fn uniform_asset_scale(local_aabb: SceneAssetAabb, target_footprint: [f32; 2]) -> f32 {
    let size = local_aabb.size();
    let local_footprint = size[0].max(size[2]).max(1.0e-5);
    let target = target_footprint[0].max(target_footprint[1]).max(0.1);
    (target / local_footprint).clamp(0.05, 20.0)
}

fn asset_scale_for_footprint(
    object: &SceneObjectSpec,
    local_aabb: SceneAssetAabb,
    target_footprint: [f32; 2],
    asset_frame: SceneAssetFrame,
    scale_policy: SceneScalePolicy,
) -> [f32; 3] {
    let uniform = uniform_asset_scale(local_aabb, target_footprint);
    if !is_table_like(object) || scale_policy == SceneScalePolicy::AssetPreserving {
        return [uniform, uniform, uniform];
    }

    let size = local_aabb.size();
    if size[0] <= 1.0e-5 || size[2] <= 1.0e-5 {
        return [uniform, uniform, uniform];
    }

    let yaw_offset = normalize_degrees(asset_frame.yaw_offset_degrees).abs();
    let local_targets = if yaw_offset > 45.0 && yaw_offset < 135.0 {
        [target_footprint[1], target_footprint[0]]
    } else {
        target_footprint
    };
    let scale_x = (local_targets[0].max(0.1) / size[0]).clamp(0.05, 20.0);
    let scale_z = (local_targets[1].max(0.1) / size[2]).clamp(0.05, 20.0);
    let scale_y = (scale_x * scale_z).sqrt().clamp(0.05, 20.0);
    scale_policy.apply_to_scale([scale_x, scale_y, scale_z])
}

fn enforce_scale_policy(
    placements: &mut [GroundedScenePlacement],
    floor_y: f32,
    scale_policy: SceneScalePolicy,
) {
    for placement in placements {
        let next = scale_policy.apply_to_scale(placement.scale);
        if next != placement.scale {
            placement.scale = next;
            placement.sync_translation_to_ground_anchor(floor_y);
        }
    }
}

fn normalize_repeated_asset_scales(placements: &mut [GroundedScenePlacement], floor_y: f32) {
    let mut grouped: HashMap<String, ([f32; 3], usize)> = HashMap::new();
    for placement in placements.iter() {
        let entry = grouped
            .entry(placement.asset_id.clone())
            .or_insert(([0.0; 3], 0));
        for axis in 0..3 {
            entry.0[axis] += placement.scale[axis].abs();
        }
        entry.1 += 1;
    }
    let repeated_scale = grouped
        .into_iter()
        .filter_map(|(asset_id, (sum, count))| {
            if count > 1 {
                Some((
                    asset_id,
                    [
                        (sum[0] / count as f32).clamp(0.05, 20.0),
                        (sum[1] / count as f32).clamp(0.05, 20.0),
                        (sum[2] / count as f32).clamp(0.05, 20.0),
                    ],
                ))
            } else {
                None
            }
        })
        .collect::<HashMap<_, _>>();
    for placement in placements.iter_mut() {
        let Some(scale) = repeated_scale.get(&placement.asset_id).copied() else {
            continue;
        };
        placement.scale = scale;
        placement.sync_translation_to_ground_anchor(floor_y);
    }
}

fn metric_side_slots(
    raw: &[(
        &SceneObjectSpec,
        &SceneAssetBinding,
        ResolvedSceneObjectInstance,
        [f32; 3],
    )],
) -> HashMap<usize, (usize, usize)> {
    let mut by_side: HashMap<SceneInstanceSide, Vec<(usize, f32)>> = HashMap::new();
    for (index, (_, _, instance, _)) in raw.iter().enumerate() {
        let Some(side) = instance
            .side
            .filter(|side| *side != SceneInstanceSide::Unknown)
        else {
            continue;
        };
        by_side
            .entry(side)
            .or_default()
            .push((index, side_contact_axis(side, instance.contact_pixel)));
    }

    let mut slots = HashMap::new();
    for (_side, mut entries) in by_side {
        let count = entries.len().max(1);
        let explicit_valid = entries.iter().all(|(index, _)| {
            raw[*index]
                .2
                .slot_index
                .is_some_and(|slot_index| slot_index < count)
        });
        if explicit_valid {
            entries.sort_by_key(|(index, _)| raw[*index].2.slot_index.unwrap_or(0));
        } else {
            entries.sort_by(|(left_index, left_axis), (right_index, right_axis)| {
                left_axis
                    .partial_cmp(right_axis)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| left_index.cmp(right_index))
            });
        }
        for (slot_index, (raw_index, _)) in entries.into_iter().enumerate() {
            slots.insert(raw_index, (slot_index, count));
        }
    }
    slots
}

fn side_contact_axis(side: SceneInstanceSide, contact_pixel: [f32; 2]) -> f32 {
    match side {
        SceneInstanceSide::Left | SceneInstanceSide::Right => contact_pixel[1],
        SceneInstanceSide::Near
        | SceneInstanceSide::Far
        | SceneInstanceSide::Head
        | SceneInstanceSide::Foot => contact_pixel[0],
        SceneInstanceSide::Unknown => 0.5,
    }
}

fn metric_ground_point(
    object: &SceneObjectSpec,
    instance: &ResolvedSceneObjectInstance,
    target_footprint: [f32; 2],
    metric_frame: Option<MetricSceneFrame>,
    side_slot: Option<(usize, usize)>,
) -> Option<[f32; 3]> {
    let frame = metric_frame?;
    if is_table_like(object) {
        return Some(frame.table_point());
    }
    let side = instance.side?;
    if side == SceneInstanceSide::Unknown {
        return None;
    }
    let (slot_index, slot_count) = side_slot.unwrap_or_else(|| {
        let slot_count = 1;
        let slot_index = instance
            .slot_index
            .or_else(|| instance.id.as_deref().and_then(last_numeric_suffix))
            .unwrap_or_else(|| side_slot_from_contact(side, instance.contact_pixel, slot_count));
        (slot_index, slot_count)
    });
    Some(frame.side_point(side, slot_index, slot_count, target_footprint))
}

fn side_slot_from_contact(
    side: SceneInstanceSide,
    contact_pixel: [f32; 2],
    slot_count: usize,
) -> usize {
    if slot_count <= 1 {
        return 0;
    }
    let axis = match side {
        SceneInstanceSide::Left | SceneInstanceSide::Right => contact_pixel[1],
        SceneInstanceSide::Near
        | SceneInstanceSide::Far
        | SceneInstanceSide::Head
        | SceneInstanceSide::Foot => contact_pixel[0],
        SceneInstanceSide::Unknown => 0.5,
    };
    ((axis.clamp(0.0, 0.999) * slot_count as f32).floor() as usize).min(slot_count - 1)
}

fn last_numeric_suffix(value: &str) -> Option<usize> {
    let suffix = value
        .rsplit(|ch: char| !ch.is_ascii_digit())
        .find(|part| !part.is_empty())?;
    suffix
        .parse::<usize>()
        .ok()
        .map(|value| value.saturating_sub(1))
}

fn grounded_yaw_degrees(
    object: &SceneObjectSpec,
    instance: &ResolvedSceneObjectInstance,
    from: [f32; 3],
    table: Option<[f32; 3]>,
    metric_frame: Option<MetricSceneFrame>,
) -> f32 {
    if is_table_like(object) {
        return metric_frame
            .map(|frame| frame.table_axis_degrees)
            .unwrap_or(0.0);
    }
    if let (Some(frame), Some(side)) = (metric_frame, instance.side)
        && side != SceneInstanceSide::Unknown
        && let Some(yaw) = bsn_yaw_toward_point_degrees(from, frame.table_point())
    {
        return yaw;
    }
    let Some(target) = table else {
        return 0.0;
    };
    bsn_yaw_toward_point_degrees(from, target).unwrap_or(0.0)
}

pub(crate) fn bsn_yaw_toward_point_degrees(from: [f32; 3], target: [f32; 3]) -> Option<f32> {
    let dx = target[0] - from[0];
    let dz = target[2] - from[2];
    if !dx.is_finite() || !dz.is_finite() || dx.abs() + dz.abs() <= 1.0e-5 {
        return None;
    }
    Some(normalize_degrees(dx.atan2(dz).to_degrees()))
}

fn scene_asset_frame(
    asset: &SceneAssetBinding,
    object: &SceneObjectSpec,
    local_aabb: SceneAssetAabb,
) -> SceneAssetFrame {
    if let Some(frame) = asset.canonical_frame {
        return frame;
    }
    let size = local_aabb.size();
    let descriptor = object_descriptor(object);
    let footprint_m = object.target_footprint_m;
    if descriptor.contains("table") {
        let yaw_offset = if size[0] > size[2] * 1.15 { 90.0 } else { 0.0 };
        let mut frame = SceneAssetFrame::heuristic(yaw_offset, footprint_m);
        frame.symmetry = Some(SceneAssetSymmetry::Axis180);
        frame.confidence = Some(0.70);
        frame
    } else {
        let mut frame = SceneAssetFrame::heuristic(0.0, footprint_m);
        frame.symmetry = Some(
            if descriptor.contains("chair")
                || descriptor.contains("seat")
                || descriptor.contains("sofa")
                || descriptor.contains("couch")
            {
                SceneAssetSymmetry::Bilateral
            } else {
                SceneAssetSymmetry::Unknown
            },
        );
        frame.confidence = Some(0.55);
        frame
    }
}

fn rotate_table_frame_point(local_xz: [f32; 2], yaw_degrees: f32) -> [f32; 3] {
    let yaw = yaw_degrees.to_radians();
    let cos = yaw.cos();
    let sin = yaw.sin();
    [
        local_xz[0] * cos + local_xz[1] * sin,
        0.0,
        -local_xz[0] * sin + local_xz[1] * cos,
    ]
}

pub(crate) fn normalize_degrees(mut degrees: f32) -> f32 {
    if !degrees.is_finite() {
        return 0.0;
    }
    while degrees <= -180.0 {
        degrees += 360.0;
    }
    while degrees > 180.0 {
        degrees -= 360.0;
    }
    degrees
}

fn finite_or(value: Option<f32>, fallback: f32) -> f32 {
    value.filter(|value| value.is_finite()).unwrap_or(fallback)
}

fn rug_from_placements(
    placements: &[GroundedScenePlacement],
    floor_y: f32,
) -> ([f32; 3], [f32; 3]) {
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_z = f32::INFINITY;
    let mut max_z = f32::NEG_INFINITY;
    for placement in placements {
        let half = placement.target_footprint_m[0].max(placement.target_footprint_m[1]) * 0.5;
        min_x = min_x.min(placement.ground_point[0] - half);
        max_x = max_x.max(placement.ground_point[0] + half);
        min_z = min_z.min(placement.ground_point[2] - half);
        max_z = max_z.max(placement.ground_point[2] + half);
    }
    if !min_x.is_finite() {
        return ([0.0, floor_y, 0.0], [4.0, 1.0, 3.0]);
    }
    let center = [(min_x + max_x) * 0.5, floor_y, (min_z + max_z) * 0.5];
    let scale = [
        (max_x - min_x + 0.75).max(2.0),
        1.0,
        (max_z - min_z + 0.75).max(2.0),
    ];
    (center, scale)
}

fn grounded_camera_from_placements(
    placements: &[GroundedScenePlacement],
    config: GroundedSceneLayoutConfig,
    metric_frame: Option<MetricSceneFrame>,
) -> SceneCamera {
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_z = f32::INFINITY;
    let mut max_z = f32::NEG_INFINITY;
    let mut max_extent = 4.0f32;
    for placement in placements {
        min_x = min_x.min(placement.ground_point[0]);
        max_x = max_x.max(placement.ground_point[0]);
        min_z = min_z.min(placement.ground_point[2]);
        max_z = max_z.max(placement.ground_point[2]);
        max_extent = max_extent.max(placement.ground_point[0].abs() * 2.0);
        max_extent = max_extent.max(placement.ground_point[2].abs() * 2.0);
    }
    let focus = if min_x.is_finite() {
        [
            (min_x + max_x) * 0.5,
            config.floor_y + 0.72,
            (min_z + max_z) * 0.5,
        ]
    } else {
        [0.0, config.floor_y + 0.72, 0.0]
    };
    let radius = metric_frame
        .and_then(|frame| frame.camera_radius_m)
        .unwrap_or_else(|| (max_extent * 0.95).max(4.5));
    let yaw = metric_frame
        .and_then(|frame| frame.camera_yaw_degrees)
        .unwrap_or(180.0);
    let pitch = metric_frame
        .and_then(|frame| frame.camera_pitch_degrees)
        .map(f32::abs)
        .unwrap_or(30.0)
        .clamp(8.0, 80.0);
    let translation = camera_translation_from_orbit(focus, yaw, pitch, radius);
    SceneCamera {
        translation,
        focus,
        yaw: Some(yaw),
        pitch: Some(pitch),
        radius: Some(radius),
        vertical_fov_degrees: metric_frame
            .and_then(|frame| frame.vertical_fov_degrees)
            .or(Some(config.vertical_fov_degrees)),
    }
}

fn camera_translation_from_orbit(
    focus: [f32; 3],
    yaw_degrees: f32,
    pitch_degrees: f32,
    radius: f32,
) -> [f32; 3] {
    let yaw = yaw_degrees.to_radians();
    let pitch = pitch_degrees.to_radians();
    let horizontal = radius * pitch.cos().abs();
    [
        focus[0] + horizontal * yaw.sin(),
        (focus[1] + radius * pitch.sin()).max(0.25),
        focus[2] + horizontal * yaw.cos(),
    ]
}

fn grounded_bsn_text(
    assets: &[SceneAssetBinding],
    placements: &[GroundedScenePlacement],
    rug_center: [f32; 3],
    rug_scale: [f32; 3],
    camera: &SceneCamera,
) -> String {
    let mut out = String::from("synth_scene_v1 {\n");
    let mut declared = HashSet::new();
    for placement in placements {
        if declared.insert(placement.asset_id.as_str())
            && let Some(asset) = assets
                .iter()
                .find(|asset| asset.asset_id == placement.asset_id)
        {
            out.push_str(&scene_asset_declaration_for_bsn(asset));
            out.push('\n');
        }
    }
    out.push_str(&format!(
        "environment rug translation [{}] scale [{}] color [0.62,0.02,0.26];\n",
        fmt_vec3(rug_center),
        fmt_vec3(rug_scale)
    ));
    for placement in placements {
        out.push_str(&format!(
            "spawn {} uses {} translation [{}] rotation_y {} scale [{}];\n",
            placement.entity_id,
            placement.asset_id,
            fmt_vec3(placement.translation),
            fmt_num(placement.rotation_y_degrees),
            fmt_vec3(placement.scale)
        ));
    }
    out.push_str(&format!(
        "camera translation [{}] focus [{}]",
        fmt_vec3(camera.translation),
        fmt_vec3(camera.focus),
    ));
    if let Some(yaw) = camera.yaw {
        out.push_str(&format!(" yaw {}", fmt_num(yaw)));
    }
    if let Some(pitch) = camera.pitch {
        out.push_str(&format!(" pitch {}", fmt_num(pitch)));
    }
    if let Some(radius) = camera.radius {
        out.push_str(&format!(" radius {}", fmt_num(radius)));
    }
    if let Some(vertical_fov) = camera.vertical_fov_degrees {
        out.push_str(&format!(" vertical_fov {}", fmt_num(vertical_fov)));
    }
    out.push_str(";\n");
    out.push_str("}\n");
    out
}

fn fmt_vec3(value: [f32; 3]) -> String {
    format!(
        "{},{},{}",
        fmt_num(value[0]),
        fmt_num(value[1]),
        fmt_num(value[2])
    )
}

fn fmt_num(value: f32) -> String {
    let mut text = format!("{value:.4}");
    while text.contains('.') && text.ends_with('0') {
        text.pop();
    }
    if text.ends_with('.') {
        text.push('0');
    }
    if text == "-0.0" {
        "0.0".to_string()
    } else {
        text
    }
}

fn is_table_like(object: &SceneObjectSpec) -> bool {
    object_descriptor(object).contains("table")
}

pub(crate) fn object_descriptor(object: &SceneObjectSpec) -> String {
    format!(
        "{} {} {}",
        object.id,
        object.label,
        object.aliases.join(" ")
    )
    .to_ascii_lowercase()
}

fn sanitize_bsn_identifier(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "entity".to_string()
    } else {
        out
    }
}

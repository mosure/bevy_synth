use serde::{Deserialize, Serialize};

use crate::layout::{floor_contact_point_from_evidence, normalize_degrees};
use crate::{
    EstimatedFloorPlane, GroundedScenePlacement, ObjectDepthStats, ObjectGroundingEvidence,
    SceneAssetAabb, SceneCamera, SceneGroundingEvidence,
};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ProjectionFitReport {
    pub applied: bool,
    pub fit_mode: String,
    pub iteration_count: usize,
    pub candidate_count: usize,
    pub initial_loss: f32,
    pub final_loss: f32,
    pub initial_loss_without_ground_anchor: f32,
    pub final_loss_without_ground_anchor: f32,
    pub initial_ground_anchor_loss: f32,
    pub final_ground_anchor_loss: f32,
    pub initial_score: f32,
    pub final_score: f32,
    pub max_ground_anchor_error_m: f32,
    pub mean_ground_anchor_error_m: f32,
    pub camera: ProjectionFitCameraReport,
    pub initial_objects: Vec<ProjectionFitObjectReport>,
    pub objects: Vec<ProjectionFitObjectReport>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub candidates: Vec<ProjectionFitCandidateReport>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ProjectionFitCameraReport {
    pub basis: String,
    pub translation: [f32; 3],
    pub focus: [f32; 3],
    pub vertical_fov_degrees: f32,
    pub aspect: f32,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ProjectionFitObjectReport {
    pub index: usize,
    pub object_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instance_id: Option<String>,
    pub label: String,
    pub source_bbox: [f32; 4],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projected_bbox: Option<[f32; 4]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projected_contact: Option<[f32; 2]>,
    pub center_error: f32,
    pub contact_error: f32,
    pub area_log2_error: f32,
    pub aspect_log2_error: f32,
    pub bbox_iou: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub depth_log2_error: Option<f32>,
    pub yaw_prior_error_degrees: f32,
    pub ground_anchor_basis: String,
    pub target_ground_point: [f32; 3],
    pub observed_ground_point: [f32; 3],
    pub ground_anchor_error_m: f32,
    pub ground_anchor_max_drift_m: f32,
    pub ground_anchor_loss: f32,
    pub loss_without_ground_anchor: f32,
    pub loss: f32,
    pub score: f32,
    pub translation: [f32; 3],
    pub rotation_y_degrees: f32,
    pub scale: [f32; 3],
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ProjectionFitCandidateReport {
    pub index: usize,
    pub object_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instance_id: Option<String>,
    pub label: String,
    pub stage: String,
    pub yaw_degrees: f32,
    pub total_loss: f32,
    pub score: f32,
    pub accepted: bool,
}

#[derive(Clone, Debug)]
struct ProjectionTarget {
    source_bbox: [f32; 4],
    contact_pixel: [f32; 2],
    depth_stats: Option<ObjectDepthStats>,
    yaw_prior_degrees: f32,
    initial_scale: f32,
    ground_anchor: [f32; 3],
    ground_anchor_basis: GroundAnchorBasis,
    source_camera_anchor: Option<[f32; 3]>,
    source_camera_origin_xz: Option<[f32; 2]>,
    ground_anchor_max_drift_m: f32,
    ground_anchor_weight: f32,
    kind: ProjectionObjectKind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProjectionObjectKind {
    Table,
    Seating,
    Other,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GroundAnchorBasis {
    CameraRayGroundPlane,
    MetricDepthContact,
    MetricDepthCenter,
    LayoutContactPixel,
}

impl GroundAnchorBasis {
    fn as_str(self) -> &'static str {
        match self {
            Self::CameraRayGroundPlane => "camera-ray-ground-plane",
            Self::MetricDepthContact => "metric-depth-contact",
            Self::MetricDepthCenter => "metric-depth-center",
            Self::LayoutContactPixel => "layout-contact-pixel",
        }
    }

    fn loss_weight(self) -> f32 {
        match self {
            Self::CameraRayGroundPlane => 1.60,
            Self::MetricDepthContact => 1.20,
            Self::MetricDepthCenter => 1.20,
            Self::LayoutContactPixel => 0.70,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum ProjectionCamera {
    Layout(LayoutProjectionCamera),
    Source(SourceProjectionCamera),
}

#[derive(Clone, Copy, Debug)]
struct LayoutProjectionCamera {
    translation: [f32; 3],
    forward: [f32; 3],
    right: [f32; 3],
    up: [f32; 3],
    vertical_fov_degrees: f32,
    aspect: f32,
}

#[derive(Clone, Copy, Debug)]
struct SourceProjectionCamera {
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    width: f32,
    height: f32,
    vertical_fov_degrees: f32,
    aspect: f32,
    floor: Option<EstimatedFloorPlane>,
}

#[derive(Clone, Debug)]
struct Evaluation {
    total_loss: f32,
    total_loss_without_ground_anchor: f32,
    total_ground_anchor_loss: f32,
    reports: Vec<ProjectionFitObjectReport>,
}

pub(crate) fn fit_grounded_scene_projection(
    placements: &mut [GroundedScenePlacement],
    camera: &SceneCamera,
    evidence: &SceneGroundingEvidence,
    floor_y: f32,
) -> Option<ProjectionFitReport> {
    if placements.is_empty() {
        return None;
    }
    let depth = evidence.depth.as_ref()?;
    let aspect = evidence
        .camera
        .image_size
        .or(depth.image_size)
        .map(|[width, height]| width.max(1) as f32 / height.max(1) as f32)
        .filter(|value| value.is_finite() && *value > 0.1)
        .unwrap_or(16.0 / 9.0);
    let targets = placements
        .iter()
        .map(|placement| ProjectionTarget::from_placement(placement, evidence))
        .collect::<Vec<_>>();
    if targets.iter().all(Option::is_none) {
        return None;
    }
    let camera = ProjectionCamera::from_scene_camera(camera, aspect)
        .or_else(|| ProjectionCamera::from_evidence(evidence, &targets))?;

    let initial_eval = evaluate_scene(placements, &targets, camera, floor_y);
    let mut best_loss = initial_eval.total_loss;
    let mut iteration_count = 0usize;
    let mut candidates = Vec::new();
    yaw_sweep(
        placements,
        &targets,
        camera,
        floor_y,
        &mut best_loss,
        &mut iteration_count,
        &mut candidates,
    );
    coordinate_search(
        placements,
        &targets,
        camera,
        floor_y,
        &mut best_loss,
        &mut iteration_count,
    );
    enforce_repeated_asset_scale(placements, floor_y);
    let final_eval = evaluate_scene(placements, &targets, camera, floor_y);
    let (max_ground_anchor_error_m, mean_ground_anchor_error_m) =
        anchor_error_summary(&final_eval.reports);

    Some(ProjectionFitReport {
        applied: final_eval.total_loss + 1.0e-5 < initial_eval.total_loss,
        fit_mode: "projected_aabb_canonical_pose".to_string(),
        iteration_count,
        candidate_count: candidates.len(),
        initial_loss: initial_eval.total_loss,
        final_loss: final_eval.total_loss,
        initial_loss_without_ground_anchor: initial_eval.total_loss_without_ground_anchor,
        final_loss_without_ground_anchor: final_eval.total_loss_without_ground_anchor,
        initial_ground_anchor_loss: initial_eval.total_ground_anchor_loss,
        final_ground_anchor_loss: final_eval.total_ground_anchor_loss,
        initial_score: projection_score(initial_eval.total_loss),
        final_score: projection_score(final_eval.total_loss),
        max_ground_anchor_error_m,
        mean_ground_anchor_error_m,
        camera: ProjectionFitCameraReport {
            basis: camera.basis().to_string(),
            translation: camera.report_translation(),
            focus: camera.report_focus(),
            vertical_fov_degrees: camera.vertical_fov_degrees(),
            aspect: camera.aspect(),
        },
        initial_objects: initial_eval.reports,
        objects: final_eval.reports,
        candidates,
    })
}

impl ProjectionTarget {
    fn from_placement(
        placement: &GroundedScenePlacement,
        evidence: &SceneGroundingEvidence,
    ) -> Option<Self> {
        let object_evidence = best_evidence_for_placement(placement, evidence);
        let source_bbox = object_evidence
            .and_then(|evidence| {
                evidence
                    .mask
                    .as_ref()
                    .map(|mask| mask.bbox)
                    .or_else(|| evidence.detection.as_ref().map(|detection| detection.bbox))
            })
            .unwrap_or(placement.source_bbox);
        let contact_pixel = object_evidence
            .and_then(|evidence| {
                evidence.contact_pixel.or_else(|| {
                    evidence
                        .detection
                        .as_ref()
                        .and_then(|detection| detection.point)
                })
            })
            .unwrap_or(placement.contact_pixel);
        if !source_bbox.iter().all(|value| value.is_finite()) {
            return None;
        }
        let kind = placement_kind(placement);
        let (ground_anchor_basis, source_camera_anchor) =
            ground_anchor_source(object_evidence, &evidence.floor, evidence, kind);
        Some(Self {
            source_bbox: normalize_bbox(source_bbox),
            contact_pixel: normalize_point(contact_pixel),
            depth_stats: object_evidence.and_then(|evidence| evidence.depth_stats),
            yaw_prior_degrees: placement.rotation_y_degrees,
            initial_scale: representative_scale(placement.scale).max(1.0e-5),
            ground_anchor: placement.ground_point,
            ground_anchor_basis,
            source_camera_anchor,
            source_camera_origin_xz: source_camera_anchor.map(|point| {
                [
                    point[0] - placement.ground_point[0],
                    point[2] - placement.ground_point[2],
                ]
            }),
            ground_anchor_max_drift_m: placement.ground_anchor_max_drift_m(),
            ground_anchor_weight: ground_anchor_basis.loss_weight(),
            kind,
        })
    }
}

fn best_evidence_for_placement<'a>(
    placement: &GroundedScenePlacement,
    evidence: &'a SceneGroundingEvidence,
) -> Option<&'a ObjectGroundingEvidence> {
    evidence
        .objects
        .iter()
        .find(|object| {
            object.object_id == placement.object_id
                && object.instance_id.as_deref() == placement.instance_id.as_deref()
        })
        .or_else(|| {
            evidence.objects.iter().find(|object| {
                object.object_id == placement.object_id && object.instance_id.is_none()
            })
        })
}

fn ground_anchor_source(
    object_evidence: Option<&ObjectGroundingEvidence>,
    floor: &EstimatedFloorPlane,
    evidence: &SceneGroundingEvidence,
    kind: ProjectionObjectKind,
) -> (GroundAnchorBasis, Option<[f32; 3]>) {
    let Some(object_evidence) = object_evidence else {
        return (GroundAnchorBasis::LayoutContactPixel, None);
    };
    if kind == ProjectionObjectKind::Table
        && let Some(point) = source_bbox_center_metric_point(object_evidence, evidence)
    {
        return (GroundAnchorBasis::MetricDepthCenter, Some(point));
    }
    if let Some(point) = floor_contact_point_from_evidence(object_evidence, floor) {
        (GroundAnchorBasis::CameraRayGroundPlane, Some(point))
    } else if object_evidence.metric_contact_point_m.is_some() {
        (
            GroundAnchorBasis::MetricDepthContact,
            object_evidence.metric_contact_point_m,
        )
    } else {
        (GroundAnchorBasis::LayoutContactPixel, None)
    }
}

fn source_bbox_center_metric_point(
    object_evidence: &ObjectGroundingEvidence,
    evidence: &SceneGroundingEvidence,
) -> Option<[f32; 3]> {
    let detection = object_evidence.detection.as_ref()?;
    let depth = object_evidence
        .depth_stats
        .map(|stats| stats.median_m)
        .filter(|value| value.is_finite() && *value > 0.0)?;
    let (fx, fy, cx, cy, width, height) = source_intrinsics(evidence)?;
    let center = bbox_center(normalize_bbox(detection.bbox));
    let pixel_x = center[0] * (width - 1.0).max(1.0);
    let pixel_y = center[1] * (height - 1.0).max(1.0);
    Some([
        (pixel_x - cx) * depth / fx.max(1.0e-5),
        (pixel_y - cy) * depth / fy.max(1.0e-5),
        depth,
    ])
}

fn source_intrinsics(evidence: &SceneGroundingEvidence) -> Option<(f32, f32, f32, f32, f32, f32)> {
    let [width, height] = evidence
        .camera
        .image_size
        .or_else(|| evidence.depth.as_ref().and_then(|depth| depth.image_size))?;
    let width = width.max(1) as f32;
    let height = height.max(1) as f32;
    let vertical_fov_degrees = evidence
        .camera
        .vertical_fov_degrees
        .or_else(|| {
            evidence
                .depth
                .as_ref()
                .and_then(|depth| depth.vertical_fov_degrees)
        })
        .filter(|value| value.is_finite() && *value > 1.0)
        .unwrap_or(72.0);
    let fy = evidence
        .camera
        .focal_length_px
        .or_else(|| {
            evidence
                .depth
                .as_ref()
                .and_then(|depth| depth.focal_length_px)
        })
        .filter(|value| value.is_finite() && *value > 1.0)
        .unwrap_or_else(|| {
            (height * 0.5) / (vertical_fov_degrees.to_radians() * 0.5).tan().max(1.0e-5)
        });
    let fx = fy;
    let principal = evidence
        .camera
        .principal_point
        .unwrap_or([(width - 1.0) * 0.5, (height - 1.0) * 0.5]);
    Some((fx, fy, principal[0], principal[1], width, height))
}

impl ProjectionCamera {
    fn from_evidence(
        evidence: &SceneGroundingEvidence,
        targets: &[Option<ProjectionTarget>],
    ) -> Option<Self> {
        if targets
            .iter()
            .filter_map(Option::as_ref)
            .all(|target| target.source_camera_anchor.is_none())
        {
            return None;
        }
        let [width, height] = evidence
            .camera
            .image_size
            .or_else(|| evidence.depth.as_ref().and_then(|depth| depth.image_size))?;
        let width = width.max(1) as f32;
        let height = height.max(1) as f32;
        let vertical_fov_degrees = evidence
            .camera
            .vertical_fov_degrees
            .or_else(|| {
                evidence
                    .depth
                    .as_ref()
                    .and_then(|depth| depth.vertical_fov_degrees)
            })
            .filter(|value| value.is_finite() && *value > 1.0)
            .unwrap_or(72.0);
        let fy = evidence
            .camera
            .focal_length_px
            .or_else(|| {
                evidence
                    .depth
                    .as_ref()
                    .and_then(|depth| depth.focal_length_px)
            })
            .filter(|value| value.is_finite() && *value > 1.0)
            .unwrap_or_else(|| {
                (height * 0.5) / (vertical_fov_degrees.to_radians() * 0.5).tan().max(1.0e-5)
            });
        let fx = fy;
        let principal = evidence
            .camera
            .principal_point
            .unwrap_or([(width - 1.0) * 0.5, (height - 1.0) * 0.5]);
        Some(Self::Source(SourceProjectionCamera {
            fx,
            fy,
            cx: principal[0],
            cy: principal[1],
            width,
            height,
            vertical_fov_degrees,
            aspect: width / height.max(1.0),
            floor: source_projection_floor(&evidence.floor),
        }))
    }

    fn from_scene_camera(camera: &SceneCamera, aspect: f32) -> Option<Self> {
        let forward = normalize3(sub3(camera.focus, camera.translation))?;
        let mut right = normalize3(cross3(forward, [0.0, 1.0, 0.0]))
            .or_else(|| normalize3(cross3(forward, [0.0, 0.0, 1.0])))?;
        if !right.iter().all(|value| value.is_finite()) {
            right = [1.0, 0.0, 0.0];
        }
        let up = normalize3(cross3(right, forward))?;
        Some(Self::Layout(LayoutProjectionCamera {
            translation: camera.translation,
            forward,
            right,
            up,
            vertical_fov_degrees: camera
                .vertical_fov_degrees
                .filter(|value| value.is_finite() && *value > 1.0)
                .unwrap_or(72.0),
            aspect: aspect.max(0.1),
        }))
    }

    fn basis(self) -> &'static str {
        match self {
            Self::Layout(_) => "layout-camera",
            Self::Source(_) => "source-depth-intrinsics",
        }
    }

    fn vertical_fov_degrees(self) -> f32 {
        match self {
            Self::Layout(camera) => camera.vertical_fov_degrees,
            Self::Source(camera) => camera.vertical_fov_degrees,
        }
    }

    fn aspect(self) -> f32 {
        match self {
            Self::Layout(camera) => camera.aspect,
            Self::Source(camera) => camera.aspect,
        }
    }

    fn report_translation(self) -> [f32; 3] {
        match self {
            Self::Layout(camera) => camera.translation,
            Self::Source(_) => [0.0, 0.0, 0.0],
        }
    }

    fn report_focus(self) -> [f32; 3] {
        match self {
            Self::Layout(camera) => [
                camera.translation[0] + camera.forward[0],
                camera.translation[1] + camera.forward[1],
                camera.translation[2] + camera.forward[2],
            ],
            Self::Source(_) => [0.0, 0.0, 1.0],
        }
    }

    fn project_point(
        self,
        point: [f32; 3],
        target: &ProjectionTarget,
        floor_y: f32,
    ) -> Option<([f32; 2], f32)> {
        match self {
            Self::Layout(camera) => camera.project_point(point),
            Self::Source(camera) => camera.project_point(point, target, floor_y),
        }
    }
}

impl LayoutProjectionCamera {
    fn project_point(self, point: [f32; 3]) -> Option<([f32; 2], f32)> {
        let rel = sub3(point, self.translation);
        let z = dot3(rel, self.forward);
        if !z.is_finite() || z <= 1.0e-4 {
            return None;
        }
        let tan_half = (self.vertical_fov_degrees.to_radians() * 0.5).tan();
        let x = dot3(rel, self.right) / (z * tan_half * self.aspect);
        let y = dot3(rel, self.up) / (z * tan_half);
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        Some(([(x + 1.0) * 0.5, (1.0 - y) * 0.5], z))
    }
}

impl SourceProjectionCamera {
    fn project_point(
        self,
        point: [f32; 3],
        target: &ProjectionTarget,
        floor_y: f32,
    ) -> Option<([f32; 2], f32)> {
        let source = self.source_camera_point(point, target, floor_y)?;
        let z = source[2];
        if !z.is_finite() || z <= 1.0e-4 {
            return None;
        }
        let u = (self.fx * source[0] / z + self.cx) / (self.width - 1.0).max(1.0);
        let v = (self.fy * source[1] / z + self.cy) / (self.height - 1.0).max(1.0);
        if !u.is_finite() || !v.is_finite() {
            return None;
        }
        Some(([u, v], z))
    }

    fn source_camera_point(
        self,
        point: [f32; 3],
        target: &ProjectionTarget,
        floor_y: f32,
    ) -> Option<[f32; 3]> {
        let anchor = target.source_camera_anchor?;
        let origin = target.source_camera_origin_xz?;
        let x = point[0] + origin[0];
        let z = point[2] + origin[1];
        let height_above_floor = point[1] - floor_y;
        let floor_y_camera =
            if target.ground_anchor_basis == GroundAnchorBasis::CameraRayGroundPlane {
                self.floor
                    .and_then(|floor| source_floor_y_at(floor, x, z))
                    .unwrap_or(anchor[1])
            } else {
                anchor[1]
            };
        Some([x, floor_y_camera - height_above_floor, z])
    }
}

fn source_projection_floor(floor: &EstimatedFloorPlane) -> Option<EstimatedFloorPlane> {
    let normal_len_sq = floor.normal.iter().map(|value| value * value).sum::<f32>();
    let residual_ok = floor
        .residual_m
        .filter(|value| value.is_finite())
        .is_none_or(|value| value <= 0.35);
    (normal_len_sq.is_finite()
        && normal_len_sq > 0.25
        && floor.normal[1].abs() > 1.0e-5
        && floor.distance_m.is_finite()
        && residual_ok)
        .then_some(*floor)
}

fn source_floor_y_at(floor: EstimatedFloorPlane, x: f32, z: f32) -> Option<f32> {
    let y = -(floor.normal[0] * x + floor.normal[2] * z + floor.distance_m) / floor.normal[1];
    (y.is_finite()).then_some(y)
}

fn yaw_sweep(
    placements: &mut [GroundedScenePlacement],
    targets: &[Option<ProjectionTarget>],
    camera: ProjectionCamera,
    floor_y: f32,
    best_loss: &mut f32,
    iteration_count: &mut usize,
    candidates: &mut Vec<ProjectionFitCandidateReport>,
) {
    for index in 0..placements.len() {
        let Some(target) = targets.get(index).and_then(Option::as_ref) else {
            continue;
        };
        let initial_yaw = placements[index].rotation_y_degrees;
        for yaw in yaw_candidates(initial_yaw, target) {
            let mut trial = placements.to_vec();
            trial[index].rotation_y_degrees = normalize_degrees(yaw);
            enforce_candidate_bounds(
                &mut trial[index],
                &placements[index],
                &targets[index],
                floor_y,
            );
            enforce_repeated_asset_scale(&mut trial, floor_y);
            let loss = evaluate_scene_loss(&trial, targets, camera, floor_y);
            *iteration_count += 1;
            let accepted = loss + 1.0e-5 < *best_loss;
            candidates.push(ProjectionFitCandidateReport {
                index,
                object_id: placements[index].object_id.clone(),
                instance_id: placements[index].instance_id.clone(),
                label: placements[index].label.clone(),
                stage: "yaw_sweep".to_string(),
                yaw_degrees: normalize_degrees(yaw),
                total_loss: loss,
                score: projection_score(loss),
                accepted,
            });
            if accepted {
                placements.clone_from_slice(&trial);
                *best_loss = loss;
            }
        }
    }
}

fn yaw_candidates(initial_yaw: f32, target: &ProjectionTarget) -> Vec<f32> {
    let mut values = Vec::new();
    for offset in [
        0.0, -8.0, 8.0, -15.0, 15.0, -25.0, 25.0, -40.0, 40.0, -60.0, 60.0, -90.0, 90.0, -120.0,
        120.0, 180.0,
    ] {
        push_unique_yaw(&mut values, initial_yaw + offset);
    }
    for absolute in [0.0, 90.0, -90.0, 180.0] {
        push_unique_yaw(&mut values, absolute);
    }
    if target.kind == ProjectionObjectKind::Table {
        let aspect = bbox_aspect(target.source_bbox);
        if aspect > 1.25 {
            push_unique_yaw(&mut values, 90.0);
            push_unique_yaw(&mut values, -90.0);
        } else if aspect < 0.80 {
            push_unique_yaw(&mut values, 0.0);
            push_unique_yaw(&mut values, 180.0);
        }
    }
    values
}

fn push_unique_yaw(values: &mut Vec<f32>, yaw: f32) {
    let yaw = normalize_degrees(yaw);
    if values
        .iter()
        .all(|existing| angular_error_degrees(*existing, yaw) > 1.0)
    {
        values.push(yaw);
    }
}

fn coordinate_search(
    placements: &mut [GroundedScenePlacement],
    targets: &[Option<ProjectionTarget>],
    camera: ProjectionCamera,
    floor_y: f32,
    best_loss: &mut f32,
    iteration_count: &mut usize,
) {
    let steps = [
        (0.55, 0.18, 24.0),
        (0.30, 0.12, 14.0),
        (0.16, 0.07, 8.0),
        (0.08, 0.04, 4.0),
        (0.04, 0.02, 2.0),
    ];
    for (move_step, scale_step, yaw_step) in steps {
        let mut improved = true;
        let mut pass_count = 0usize;
        while improved && pass_count < 3 {
            improved = false;
            pass_count += 1;
            for index in 0..placements.len() {
                if targets.get(index).and_then(Option::as_ref).is_none() {
                    continue;
                }
                let candidates = [
                    CandidateDelta::Translate([move_step, 0.0, 0.0]),
                    CandidateDelta::Translate([-move_step, 0.0, 0.0]),
                    CandidateDelta::Translate([0.0, 0.0, move_step]),
                    CandidateDelta::Translate([0.0, 0.0, -move_step]),
                    CandidateDelta::Scale(1.0 + scale_step),
                    CandidateDelta::Scale(1.0 - scale_step),
                    CandidateDelta::Yaw(yaw_step),
                    CandidateDelta::Yaw(-yaw_step),
                ];
                for candidate in candidates {
                    let mut trial = placements.to_vec();
                    apply_candidate_delta(&mut trial[index], candidate, floor_y);
                    enforce_candidate_bounds(
                        &mut trial[index],
                        &placements[index],
                        &targets[index],
                        floor_y,
                    );
                    enforce_repeated_asset_scale(&mut trial, floor_y);
                    let loss = evaluate_scene_loss(&trial, targets, camera, floor_y);
                    *iteration_count += 1;
                    if loss + 1.0e-5 < *best_loss {
                        placements.clone_from_slice(&trial);
                        *best_loss = loss;
                        improved = true;
                    }
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
enum CandidateDelta {
    Translate([f32; 3]),
    Scale(f32),
    Yaw(f32),
}

fn apply_candidate_delta(
    placement: &mut GroundedScenePlacement,
    delta: CandidateDelta,
    floor_y: f32,
) {
    match delta {
        CandidateDelta::Translate(delta) => {
            placement.translation[0] += delta[0];
            placement.translation[2] += delta[2];
            placement.ground_point[0] += delta[0];
            placement.ground_point[2] += delta[2];
        }
        CandidateDelta::Scale(multiplier) => {
            for axis in &mut placement.scale {
                *axis = (*axis * multiplier).clamp(0.05, 20.0);
            }
            placement.translation[1] = floor_y - placement.local_aabb.min[1] * placement.scale[1];
        }
        CandidateDelta::Yaw(delta) => {
            placement.rotation_y_degrees = normalize_degrees(placement.rotation_y_degrees + delta);
        }
    }
}

fn enforce_candidate_bounds(
    placement: &mut GroundedScenePlacement,
    baseline: &GroundedScenePlacement,
    target: &Option<ProjectionTarget>,
    floor_y: f32,
) {
    let kind = target
        .as_ref()
        .map(|target| target.kind)
        .unwrap_or_else(|| placement_kind(placement));
    let anchor = target
        .as_ref()
        .map(|target| target.ground_anchor)
        .unwrap_or(baseline.ground_point);
    let max_move = target
        .as_ref()
        .map(|target| target.ground_anchor_max_drift_m)
        .unwrap_or_else(|| baseline.ground_anchor_max_drift_m());
    clamp_ground_anchor_drift(placement, anchor, max_move);

    let initial_scale = target
        .as_ref()
        .map(|target| target.initial_scale)
        .unwrap_or_else(|| representative_scale(baseline.scale).max(1.0e-5));
    let (min_scale, max_scale) = match kind {
        ProjectionObjectKind::Table => (initial_scale * 0.70, initial_scale * 1.36),
        ProjectionObjectKind::Seating => (initial_scale * 0.66, initial_scale * 1.50),
        ProjectionObjectKind::Other => (initial_scale * 0.65, initial_scale * 1.55),
    };
    let current_scale = representative_scale(placement.scale).max(1.0e-5);
    let next_scale = current_scale.clamp(min_scale.max(0.05), max_scale.min(20.0));
    let multiplier = next_scale / current_scale;
    for axis in &mut placement.scale {
        *axis = (*axis * multiplier).clamp(0.05, 20.0);
    }
    placement.translation[1] = floor_y - placement.local_aabb.min[1] * placement.scale[1];
}

fn clamp_ground_anchor_drift(
    placement: &mut GroundedScenePlacement,
    anchor: [f32; 3],
    max_drift_m: f32,
) {
    let max_drift_m = max_drift_m.max(1.0e-4);
    let dx = placement.ground_point[0] - anchor[0];
    let dz = placement.ground_point[2] - anchor[2];
    let distance = (dx * dx + dz * dz).sqrt();
    let (next_x, next_z) = if distance.is_finite() && distance > max_drift_m {
        let scale = max_drift_m / distance;
        (anchor[0] + dx * scale, anchor[2] + dz * scale)
    } else {
        (placement.ground_point[0], placement.ground_point[2])
    };
    placement.ground_point[0] = next_x;
    placement.ground_point[2] = next_z;
    placement.translation[0] = next_x;
    placement.translation[2] = next_z;
}

fn enforce_repeated_asset_scale(placements: &mut [GroundedScenePlacement], floor_y: f32) {
    let mut grouped = std::collections::HashMap::<String, ([f32; 3], usize)>::new();
    for placement in placements.iter() {
        let entry = grouped
            .entry(placement.asset_id.clone())
            .or_insert(([0.0; 3], 0));
        for axis in 0..3 {
            entry.0[axis] += placement.scale[axis].abs();
        }
        entry.1 += 1;
    }
    let repeated = grouped
        .into_iter()
        .filter_map(|(asset_id, (sum, count))| {
            (count > 1).then_some((
                asset_id,
                [
                    (sum[0] / count as f32).clamp(0.05, 20.0),
                    (sum[1] / count as f32).clamp(0.05, 20.0),
                    (sum[2] / count as f32).clamp(0.05, 20.0),
                ],
            ))
        })
        .collect::<std::collections::HashMap<_, _>>();
    for placement in placements {
        let Some(scale) = repeated.get(&placement.asset_id).copied() else {
            continue;
        };
        placement.scale = scale;
        placement.translation[1] = floor_y - placement.local_aabb.min[1] * scale[1];
    }
}

fn representative_scale(scale: [f32; 3]) -> f32 {
    ((scale[0].abs() + scale[1].abs() + scale[2].abs()) / 3.0).clamp(0.05, 20.0)
}

fn evaluate_scene_loss(
    placements: &[GroundedScenePlacement],
    targets: &[Option<ProjectionTarget>],
    camera: ProjectionCamera,
    floor_y: f32,
) -> f32 {
    evaluate_scene(placements, targets, camera, floor_y).total_loss
}

fn evaluate_scene(
    placements: &[GroundedScenePlacement],
    targets: &[Option<ProjectionTarget>],
    camera: ProjectionCamera,
    floor_y: f32,
) -> Evaluation {
    let mut reports = Vec::with_capacity(placements.len());
    let mut total_loss = 0.0f32;
    let mut total_ground_anchor_loss = 0.0f32;
    for (index, placement) in placements.iter().enumerate() {
        let Some(target) = targets.get(index).and_then(Option::as_ref) else {
            continue;
        };
        let report = evaluate_object(index, placement, target, camera, floor_y);
        total_loss += report.loss;
        total_ground_anchor_loss += report.ground_anchor_loss;
        reports.push(report);
    }
    total_loss += overlap_penalty(placements, targets);
    Evaluation {
        total_loss,
        total_loss_without_ground_anchor: total_loss - total_ground_anchor_loss,
        total_ground_anchor_loss,
        reports,
    }
}

fn evaluate_object(
    index: usize,
    placement: &GroundedScenePlacement,
    target: &ProjectionTarget,
    camera: ProjectionCamera,
    floor_y: f32,
) -> ProjectionFitObjectReport {
    let projected_bbox = projected_aabb_bbox(placement, target, camera, floor_y);
    let projected_contact = camera
        .project_point(placement.ground_point, target, floor_y)
        .map(|(point, _)| point);
    let mut center_error = 1.0;
    let mut contact_error = 1.0;
    let mut area_log2_error = 8.0;
    let mut aspect_log2_error = 8.0;
    let mut iou = 0.0;
    let mut depth_log2_error = None;
    let ground_anchor_error_m = distance_xz(placement.ground_point, target.ground_anchor);
    let ground_anchor_loss = ((ground_anchor_error_m
        / target.ground_anchor_max_drift_m.max(1.0e-4))
    .powi(2)
    .min(4.0))
        * target.ground_anchor_weight;
    let mut loss_without_ground_anchor = 16.0;
    let mut loss = 16.0 + ground_anchor_loss;
    if let Some(projected_bbox) = projected_bbox {
        let target_center = bbox_center(target.source_bbox);
        let projected_center = bbox_center(projected_bbox);
        center_error = distance2(target_center, projected_center);
        let target_anchor = if target.kind == ProjectionObjectKind::Table {
            target_center
        } else {
            target.contact_pixel
        };
        let projected_anchor =
            projected_contact.unwrap_or_else(|| bbox_bottom_center(projected_bbox));
        contact_error = distance2(target_anchor, projected_anchor);
        area_log2_error =
            safe_log2_ratio(bbox_area(projected_bbox), bbox_area(target.source_bbox)).abs();
        aspect_log2_error =
            safe_log2_ratio(bbox_aspect(projected_bbox), bbox_aspect(target.source_bbox)).abs();
        iou = bbox_iou(projected_bbox, target.source_bbox);
        depth_log2_error = target.depth_stats.and_then(|stats| {
            let center = transformed_aabb_center(placement);
            let (_, z) = camera.project_point(center, target, floor_y)?;
            let depth = stats.contact_m.unwrap_or(stats.median_m).max(1.0e-4);
            Some(safe_log2_ratio(z, depth).abs().min(4.0))
        });
        let yaw_prior_error_degrees =
            angular_error_degrees(placement.rotation_y_degrees, target.yaw_prior_degrees);
        let (area_weight, aspect_weight, iou_weight) = match target.kind {
            ProjectionObjectKind::Table => (0.55, 0.46, 1.05),
            ProjectionObjectKind::Seating => (0.52, 0.40, 0.95),
            ProjectionObjectKind::Other => (0.45, 0.24, 0.85),
        };
        loss_without_ground_anchor = center_error * 2.20
            + contact_error
                * if target.kind == ProjectionObjectKind::Table {
                    0.75
                } else {
                    1.60
                }
            + area_log2_error * area_weight
            + aspect_log2_error * aspect_weight
            + (1.0 - iou) * iou_weight
            + depth_log2_error.unwrap_or(0.0) * 0.12
            + (yaw_prior_error_degrees / 180.0) * 0.08;
        loss = loss_without_ground_anchor + ground_anchor_loss;
    }
    let yaw_prior_error_degrees =
        angular_error_degrees(placement.rotation_y_degrees, target.yaw_prior_degrees);
    ProjectionFitObjectReport {
        index,
        object_id: placement.object_id.clone(),
        instance_id: placement.instance_id.clone(),
        label: placement.label.clone(),
        source_bbox: target.source_bbox,
        projected_bbox,
        projected_contact,
        center_error,
        contact_error,
        area_log2_error,
        aspect_log2_error,
        bbox_iou: iou,
        depth_log2_error,
        yaw_prior_error_degrees,
        ground_anchor_basis: target.ground_anchor_basis.as_str().to_string(),
        target_ground_point: target.ground_anchor,
        observed_ground_point: placement.ground_point,
        ground_anchor_error_m,
        ground_anchor_max_drift_m: target.ground_anchor_max_drift_m,
        ground_anchor_loss,
        loss_without_ground_anchor,
        loss,
        score: projection_score(loss),
        translation: placement.translation,
        rotation_y_degrees: placement.rotation_y_degrees,
        scale: placement.scale,
    }
}

fn anchor_error_summary(reports: &[ProjectionFitObjectReport]) -> (f32, f32) {
    if reports.is_empty() {
        return (0.0, 0.0);
    }
    let mut max_error = 0.0f32;
    let mut sum = 0.0f32;
    for report in reports {
        max_error = max_error.max(report.ground_anchor_error_m);
        sum += report.ground_anchor_error_m;
    }
    (max_error, sum / reports.len() as f32)
}

fn projected_aabb_bbox(
    placement: &GroundedScenePlacement,
    target: &ProjectionTarget,
    camera: ProjectionCamera,
    floor_y: f32,
) -> Option<[f32; 4]> {
    let corners = transformed_aabb_corners(placement);
    let mut min = [f32::INFINITY, f32::INFINITY];
    let mut max = [f32::NEG_INFINITY, f32::NEG_INFINITY];
    let mut count = 0usize;
    for corner in corners {
        let Some((point, _)) = camera.project_point(corner, target, floor_y) else {
            continue;
        };
        min[0] = min[0].min(point[0]);
        min[1] = min[1].min(point[1]);
        max[0] = max[0].max(point[0]);
        max[1] = max[1].max(point[1]);
        count += 1;
    }
    if count < 2 {
        return None;
    }
    Some(normalize_bbox([min[0], min[1], max[0], max[1]]))
}

fn transformed_aabb_corners(placement: &GroundedScenePlacement) -> [[f32; 3]; 8] {
    let SceneAssetAabb { min, max } = placement.local_aabb;
    let mut out = [[0.0; 3]; 8];
    let mut index = 0usize;
    for x in [min[0], max[0]] {
        for y in [min[1], max[1]] {
            for z in [min[2], max[2]] {
                out[index] = transform_local_point(placement, [x, y, z]);
                index += 1;
            }
        }
    }
    out
}

fn transformed_aabb_center(placement: &GroundedScenePlacement) -> [f32; 3] {
    let min = placement.local_aabb.min;
    let max = placement.local_aabb.max;
    transform_local_point(
        placement,
        [
            (min[0] + max[0]) * 0.5,
            (min[1] + max[1]) * 0.5,
            (min[2] + max[2]) * 0.5,
        ],
    )
}

fn transform_local_point(placement: &GroundedScenePlacement, local: [f32; 3]) -> [f32; 3] {
    let scaled = [
        local[0] * placement.scale[0],
        local[1] * placement.scale[1],
        local[2] * placement.scale[2],
    ];
    let yaw = placement.rotation_y_degrees.to_radians();
    let cos = yaw.cos();
    let sin = yaw.sin();
    [
        placement.translation[0] + scaled[0] * cos + scaled[2] * sin,
        placement.translation[1] + scaled[1],
        placement.translation[2] - scaled[0] * sin + scaled[2] * cos,
    ]
}

fn overlap_penalty(
    placements: &[GroundedScenePlacement],
    targets: &[Option<ProjectionTarget>],
) -> f32 {
    let mut penalty = 0.0f32;
    for (left, left_placement) in placements.iter().enumerate() {
        let Some(left_rect) = footprint_rect(left_placement) else {
            continue;
        };
        let left_kind = targets
            .get(left)
            .and_then(Option::as_ref)
            .map(|target| target.kind)
            .unwrap_or_else(|| placement_kind(left_placement));
        for (right, right_placement) in placements.iter().enumerate().skip(left + 1) {
            let Some(right_rect) = footprint_rect(right_placement) else {
                continue;
            };
            let right_kind = targets
                .get(right)
                .and_then(Option::as_ref)
                .map(|target| target.kind)
                .unwrap_or_else(|| placement_kind(right_placement));
            let overlap = rect_overlap_area(left_rect, right_rect);
            if overlap <= 1.0e-5 {
                continue;
            }
            let smaller = rect_area(left_rect).min(rect_area(right_rect)).max(1.0e-6);
            let fraction = (overlap / smaller).clamp(0.0, 1.0);
            let weight = if (left_kind == ProjectionObjectKind::Table
                && right_kind == ProjectionObjectKind::Seating)
                || (left_kind == ProjectionObjectKind::Seating
                    && right_kind == ProjectionObjectKind::Table)
            {
                3.5
            } else if left_kind == ProjectionObjectKind::Seating
                && right_kind == ProjectionObjectKind::Seating
            {
                1.2
            } else {
                0.6
            };
            penalty += fraction * weight;
        }
    }
    penalty
}

fn footprint_rect(placement: &GroundedScenePlacement) -> Option<[f32; 4]> {
    let corners = transformed_aabb_corners(placement);
    let mut min_x = f32::INFINITY;
    let mut min_z = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_z = f32::NEG_INFINITY;
    for corner in corners {
        min_x = min_x.min(corner[0]);
        min_z = min_z.min(corner[2]);
        max_x = max_x.max(corner[0]);
        max_z = max_z.max(corner[2]);
    }
    if min_x.is_finite() && min_z.is_finite() && max_x > min_x && max_z > min_z {
        Some([min_x, min_z, max_x, max_z])
    } else {
        None
    }
}

fn rect_overlap_area(left: [f32; 4], right: [f32; 4]) -> f32 {
    let x = (left[2].min(right[2]) - left[0].max(right[0])).max(0.0);
    let z = (left[3].min(right[3]) - left[1].max(right[1])).max(0.0);
    x * z
}

fn rect_area(rect: [f32; 4]) -> f32 {
    ((rect[2] - rect[0]).max(0.0) * (rect[3] - rect[1]).max(0.0)).max(1.0e-8)
}

fn placement_kind(placement: &GroundedScenePlacement) -> ProjectionObjectKind {
    let descriptor = format!("{} {}", placement.object_id, placement.label).to_ascii_lowercase();
    if descriptor.contains("table") || descriptor.contains("desk") || descriptor.contains("counter")
    {
        ProjectionObjectKind::Table
    } else if descriptor.contains("chair")
        || descriptor.contains("seat")
        || descriptor.contains("sofa")
        || descriptor.contains("couch")
    {
        ProjectionObjectKind::Seating
    } else {
        ProjectionObjectKind::Other
    }
}

fn normalize_bbox(mut bbox: [f32; 4]) -> [f32; 4] {
    bbox[0] = bbox[0].clamp(0.0, 1.0);
    bbox[1] = bbox[1].clamp(0.0, 1.0);
    bbox[2] = bbox[2].clamp(0.0, 1.0);
    bbox[3] = bbox[3].clamp(0.0, 1.0);
    if bbox[0] > bbox[2] {
        bbox.swap(0, 2);
    }
    if bbox[1] > bbox[3] {
        bbox.swap(1, 3);
    }
    bbox
}

fn normalize_point(point: [f32; 2]) -> [f32; 2] {
    [point[0].clamp(0.0, 1.0), point[1].clamp(0.0, 1.0)]
}

fn bbox_center(bbox: [f32; 4]) -> [f32; 2] {
    [(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5]
}

fn bbox_bottom_center(bbox: [f32; 4]) -> [f32; 2] {
    [(bbox[0] + bbox[2]) * 0.5, bbox[3]]
}

fn bbox_area(bbox: [f32; 4]) -> f32 {
    ((bbox[2] - bbox[0]).abs() * (bbox[3] - bbox[1]).abs()).max(1.0e-6)
}

fn bbox_aspect(bbox: [f32; 4]) -> f32 {
    ((bbox[2] - bbox[0]).abs() / (bbox[3] - bbox[1]).abs().max(1.0e-6)).max(1.0e-6)
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

fn distance_xz(left: [f32; 3], right: [f32; 3]) -> f32 {
    let dx = left[0] - right[0];
    let dz = left[2] - right[2];
    (dx * dx + dz * dz).sqrt()
}

fn angular_error_degrees(left: f32, right: f32) -> f32 {
    normalize_degrees(left - right).abs()
}

fn projection_score(loss: f32) -> f32 {
    (1.0 / (1.0 + loss.max(0.0))).clamp(0.0, 1.0)
}

fn sub3(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn dot3(left: [f32; 3], right: [f32; 3]) -> f32 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn cross3(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn normalize3(value: [f32; 3]) -> Option<[f32; 3]> {
    let len = dot3(value, value).sqrt();
    if !len.is_finite() || len <= 1.0e-8 {
        None
    } else {
        Some([value[0] / len, value[1] / len, value[2] / len])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DepthEvidenceRef, EstimatedCamera, EstimatedFloorPlane};

    fn test_camera() -> SceneCamera {
        SceneCamera {
            translation: [0.0, 2.0, 5.0],
            focus: [0.0, 0.5, 0.0],
            yaw: Some(180.0),
            pitch: Some(18.0),
            radius: Some(5.0),
            vertical_fov_degrees: Some(70.0),
        }
    }

    fn test_placement(entity_id: &str, x: f32) -> GroundedScenePlacement {
        GroundedScenePlacement {
            entity_id: entity_id.to_string(),
            asset_id: "chair_asset".to_string(),
            object_id: "chair_group".to_string(),
            instance_id: Some(entity_id.to_string()),
            label: "chair".to_string(),
            source_bbox: [0.42, 0.45, 0.58, 0.85],
            contact_pixel: [0.50, 0.85],
            ground_point: [x, 0.0, 0.0],
            translation: [x, 0.0, 0.0],
            rotation_y_degrees: 0.0,
            scale: [1.0, 1.0, 1.0],
            local_aabb: SceneAssetAabb {
                min: [-0.25, 0.0, -0.25],
                max: [0.25, 1.0, 0.25],
            },
            target_footprint_m: [0.5, 0.5],
        }
    }

    fn test_target(placement: &GroundedScenePlacement) -> ProjectionTarget {
        ProjectionTarget {
            source_bbox: placement.source_bbox,
            contact_pixel: placement.contact_pixel,
            depth_stats: None,
            yaw_prior_degrees: placement.rotation_y_degrees,
            initial_scale: placement.scale[0],
            ground_anchor: placement.ground_point,
            ground_anchor_basis: GroundAnchorBasis::LayoutContactPixel,
            source_camera_anchor: None,
            source_camera_origin_xz: None,
            ground_anchor_max_drift_m: placement.ground_anchor_max_drift_m(),
            ground_anchor_weight: GroundAnchorBasis::LayoutContactPixel.loss_weight(),
            kind: placement_kind(placement),
        }
    }

    #[test]
    fn projected_bbox_moves_with_translation() {
        let camera = ProjectionCamera::from_scene_camera(&test_camera(), 16.0 / 9.0).unwrap();
        let left_placement = test_placement("left", -0.5);
        let right_placement = test_placement("right", 0.5);
        let left = projected_aabb_bbox(&left_placement, &test_target(&left_placement), camera, 0.0)
            .unwrap();
        let right = projected_aabb_bbox(
            &right_placement,
            &test_target(&right_placement),
            camera,
            0.0,
        )
        .unwrap();
        assert!(bbox_center(left)[0] < bbox_center(right)[0]);
    }

    #[test]
    fn projection_fit_reports_candidates_and_does_not_worsen_loss() {
        let mut placements = vec![test_placement("chair_001", -1.0)];
        let evidence = SceneGroundingEvidence {
            source_image_path: "synthetic.png".to_string(),
            depth: Some(DepthEvidenceRef {
                provider: "synthetic".to_string(),
                model: None,
                precision: None,
                artifact_path: None,
                focal_length_px: None,
                vertical_fov_degrees: Some(70.0),
                image_size: Some([1600, 900]),
                depth_map_size: Some([1600, 900]),
                floor_sample_count: Some(64),
            }),
            segmentation: None,
            detections: Vec::new(),
            camera: EstimatedCamera {
                image_size: Some([1600, 900]),
                vertical_fov_degrees: Some(70.0),
                ..EstimatedCamera::default()
            },
            floor: EstimatedFloorPlane::default(),
            objects: Vec::new(),
        };
        let report =
            fit_grounded_scene_projection(&mut placements, &test_camera(), &evidence, 0.0).unwrap();
        assert!(report.final_loss <= report.initial_loss);
        assert_eq!(report.fit_mode, "projected_aabb_canonical_pose");
        assert_eq!(report.candidate_count, report.candidates.len());
        assert!(!report.candidates.is_empty());
        assert!(
            report.max_ground_anchor_error_m
                <= report.objects[0].ground_anchor_max_drift_m + 1.0e-4
        );
    }

    #[test]
    fn projection_fit_keeps_camera_ray_ground_anchor_fixed() {
        let mut placement = test_placement("chair_001", 0.0);
        placement.source_bbox = [0.76, 0.45, 0.94, 0.88];
        placement.contact_pixel = [0.86, 0.88];
        let initial_anchor = placement.ground_point;
        let max_drift = placement.ground_anchor_max_drift_m();
        let mut placements = vec![placement.clone()];
        let evidence = SceneGroundingEvidence {
            source_image_path: "synthetic.png".to_string(),
            depth: Some(DepthEvidenceRef {
                provider: "synthetic".to_string(),
                model: None,
                precision: None,
                artifact_path: None,
                focal_length_px: None,
                vertical_fov_degrees: Some(70.0),
                image_size: Some([1600, 900]),
                depth_map_size: Some([1600, 900]),
                floor_sample_count: Some(64),
            }),
            segmentation: None,
            detections: Vec::new(),
            camera: EstimatedCamera {
                image_size: Some([1600, 900]),
                vertical_fov_degrees: Some(70.0),
                ..EstimatedCamera::default()
            },
            floor: EstimatedFloorPlane {
                normal: [0.0, 1.0, 0.0],
                distance_m: 2.0,
                residual_m: Some(0.01),
                confidence: Some(0.99),
            },
            objects: vec![ObjectGroundingEvidence {
                object_id: placement.object_id.clone(),
                instance_id: placement.instance_id.clone(),
                reuse_group: None,
                detection: Some(crate::Detection {
                    label: placement.label.clone(),
                    bbox: placement.source_bbox,
                    point: Some(placement.contact_pixel),
                    confidence: Some(0.95),
                    source_query: "chair".to_string(),
                }),
                mask: None,
                asset_id: None,
                contact_pixel: Some(placement.contact_pixel),
                depth_stats: None,
                candidate_floor_contact_rays: vec![[0.0, -1.0, 1.0]],
                metric_contact_point_m: Some([0.0, -2.0, 2.0]),
                target_footprint_m: Some(placement.target_footprint_m),
                provenance: vec!["synthetic_camera_ray".to_string()],
            }],
        };

        let report =
            fit_grounded_scene_projection(&mut placements, &test_camera(), &evidence, 0.0).unwrap();
        let object = &report.objects[0];

        assert_eq!(report.camera.basis, "layout-camera");
        assert_eq!(object.ground_anchor_basis, "camera-ray-ground-plane");
        assert_eq!(object.target_ground_point, initial_anchor);
        assert!(
            object.ground_anchor_error_m <= max_drift + 1.0e-4,
            "ground anchor drift {} exceeded {}",
            object.ground_anchor_error_m,
            max_drift
        );
        assert_eq!(
            report.max_ground_anchor_error_m,
            object.ground_anchor_error_m
        );
    }

    #[test]
    fn projection_fit_reports_metric_anchor_when_floor_ray_is_rejected() {
        let placement = test_placement("chair_001", 0.0);
        let mut placements = vec![placement.clone()];
        let evidence = SceneGroundingEvidence {
            source_image_path: "synthetic.png".to_string(),
            depth: Some(DepthEvidenceRef {
                provider: "synthetic".to_string(),
                model: None,
                precision: None,
                artifact_path: None,
                focal_length_px: Some(800.0),
                vertical_fov_degrees: Some(60.0),
                image_size: Some([1600, 900]),
                depth_map_size: Some([1600, 900]),
                floor_sample_count: Some(64),
            }),
            segmentation: None,
            detections: Vec::new(),
            camera: EstimatedCamera {
                focal_length_px: Some(800.0),
                principal_point: Some([799.5, 449.5]),
                image_size: Some([1600, 900]),
                vertical_fov_degrees: Some(60.0),
                ..EstimatedCamera::default()
            },
            floor: EstimatedFloorPlane {
                normal: [0.0, 1.0, 0.0],
                distance_m: 2.0,
                residual_m: Some(0.50),
                confidence: Some(0.50),
            },
            objects: vec![ObjectGroundingEvidence {
                object_id: placement.object_id.clone(),
                instance_id: placement.instance_id.clone(),
                reuse_group: None,
                detection: Some(crate::Detection {
                    label: placement.label.clone(),
                    bbox: placement.source_bbox,
                    point: Some(placement.contact_pixel),
                    confidence: Some(0.95),
                    source_query: "chair".to_string(),
                }),
                mask: None,
                asset_id: None,
                contact_pixel: Some(placement.contact_pixel),
                depth_stats: None,
                candidate_floor_contact_rays: vec![[0.0, -1.0, 1.0]],
                metric_contact_point_m: Some([0.0, -2.0, 2.0]),
                target_footprint_m: Some(placement.target_footprint_m),
                provenance: vec!["synthetic_metric_depth".to_string()],
            }],
        };

        let report =
            fit_grounded_scene_projection(&mut placements, &test_camera(), &evidence, 0.0).unwrap();

        assert_eq!(report.camera.basis, "layout-camera");
        assert_eq!(
            report.objects[0].ground_anchor_basis,
            "metric-depth-contact"
        );
    }

    #[test]
    fn metric_depth_anchor_projection_ignores_unrelated_floor_plane() {
        let placement = test_placement("chair_001", 0.0);
        let width = 1600.0f32;
        let height = 900.0f32;
        let fx = 800.0f32;
        let fy = 800.0f32;
        let cx = 799.5f32;
        let cy = 449.5f32;
        let depth = 2.0f32;
        let pixel_x = placement.contact_pixel[0] * (width - 1.0);
        let pixel_y = placement.contact_pixel[1] * (height - 1.0);
        let metric_contact = [
            (pixel_x - cx) * depth / fx,
            (pixel_y - cy) * depth / fy,
            depth,
        ];
        let evidence = SceneGroundingEvidence {
            source_image_path: "synthetic.png".to_string(),
            depth: Some(DepthEvidenceRef {
                provider: "synthetic".to_string(),
                model: None,
                precision: None,
                artifact_path: None,
                focal_length_px: Some(fy),
                vertical_fov_degrees: Some(60.0),
                image_size: Some([width as u32, height as u32]),
                depth_map_size: Some([width as u32, height as u32]),
                floor_sample_count: Some(64),
            }),
            segmentation: None,
            detections: Vec::new(),
            camera: EstimatedCamera {
                focal_length_px: Some(fy),
                principal_point: Some([cx, cy]),
                image_size: Some([width as u32, height as u32]),
                vertical_fov_degrees: Some(60.0),
                ..EstimatedCamera::default()
            },
            floor: EstimatedFloorPlane {
                normal: [0.0, 1.0, 0.0],
                distance_m: -1.2,
                residual_m: Some(0.01),
                confidence: Some(0.99),
            },
            objects: vec![ObjectGroundingEvidence {
                object_id: placement.object_id.clone(),
                instance_id: placement.instance_id.clone(),
                reuse_group: None,
                detection: Some(crate::Detection {
                    label: placement.label.clone(),
                    bbox: placement.source_bbox,
                    point: Some(placement.contact_pixel),
                    confidence: Some(0.95),
                    source_query: "chair".to_string(),
                }),
                mask: None,
                asset_id: None,
                contact_pixel: Some(placement.contact_pixel),
                depth_stats: None,
                candidate_floor_contact_rays: Vec::new(),
                metric_contact_point_m: Some(metric_contact),
                target_footprint_m: Some(placement.target_footprint_m),
                provenance: vec!["synthetic_metric_depth".to_string()],
            }],
        };
        let target = ProjectionTarget::from_placement(&placement, &evidence).unwrap();
        let camera = ProjectionCamera::from_evidence(&evidence, &[Some(target.clone())]).unwrap();
        let (projected, _) = camera
            .project_point(placement.ground_point, &target, 0.0)
            .unwrap();

        assert_eq!(
            target.ground_anchor_basis,
            GroundAnchorBasis::MetricDepthContact
        );
        assert!((projected[0] - placement.contact_pixel[0]).abs() < 1.0e-5);
        assert!((projected[1] - placement.contact_pixel[1]).abs() < 1.0e-5);
    }

    #[test]
    fn repeated_asset_scale_is_shared() {
        let mut placements = vec![
            test_placement("chair_001", -0.5),
            test_placement("chair_002", 0.5),
        ];
        placements[0].scale = [1.4, 1.4, 1.4];
        placements[1].scale = [0.6, 0.6, 0.6];
        enforce_repeated_asset_scale(&mut placements, 0.0);
        assert!((placements[0].scale[0] - placements[1].scale[0]).abs() < 1.0e-6);
    }
}

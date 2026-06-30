use super::soft_refinement::dense_soft_surface_refine_pose;
use super::visible_surface::{
    SurfaceDepthSummary, binary_mask_iou, mask_point_moments, mask_point_surface_loss,
    project_mesh_visible_surface_mask, surface_depth_loss, surface_depth_summary_report,
    write_rotation_candidate_mask_png,
};
use super::*;

#[derive(Clone)]
pub(super) struct RotationFitCandidate {
    pub(super) candidate_index: usize,
    pub(super) stage: &'static str,
    pub(super) yaw_degrees: f32,
    pub(super) yaw_delta_degrees: f32,
    pub(super) mask_iou: f32,
    pub(super) bbox_iou: f32,
    pub(super) depth_error_m: Option<f32>,
    pub(super) depth_passed: bool,
    pub(super) loss: f32,
    pub(super) passed: bool,
    pub(super) projected_bbox: Option<[f32; 4]>,
    pub(super) front_facing_face_count: usize,
    pub(super) covered_px: usize,
    pub(super) fallback_points: bool,
    pub(super) artifact_path: Option<PathBuf>,
}

#[derive(Clone)]
pub(super) struct VisibleSurfacePoseCandidate {
    pub(super) candidate_index: usize,
    pub(super) stage: &'static str,
    pub(super) placement: GroundedScenePlacement,
    pub(super) mask_iou: f32,
    pub(super) bbox_iou: f32,
    pub(super) center_error: f32,
    pub(super) area_log2_error: f32,
    pub(super) aspect_log2_error: f32,
    pub(super) depth_error_m: Option<f32>,
    pub(super) surface_depth_loss: Option<f32>,
    pub(super) dense_depth_loss: Option<f32>,
    pub(super) dense_depth_mae_m: Option<f32>,
    pub(super) dense_depth_sample_count: usize,
    pub(super) point_surface_loss: f32,
    pub(super) semantic_yaw_prior_degrees: Option<f32>,
    pub(super) semantic_yaw_error_degrees: Option<f32>,
    pub(super) semantic_yaw_loss: f32,
    pub(super) candidate_depth_summary: Option<SurfaceDepthSummary>,
    pub(super) target_depth_summary: Option<SurfaceDepthSummary>,
    pub(super) surface_depth_passed: bool,
    pub(super) depth_passed: bool,
    pub(super) loss: f32,
    pub(super) passed: bool,
    pub(super) projected_bbox: Option<[f32; 4]>,
    pub(super) front_facing_face_count: usize,
    pub(super) covered_px: usize,
    pub(super) fallback_points: bool,
    pub(super) artifact_path: Option<PathBuf>,
    pub(super) dense_optimization: Option<Value>,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn evaluate_visible_surface_pose_candidates(
    mesh: &RenderMesh,
    baseline: &GroundedScenePlacement,
    fit_object: &crate::ProjectionFitObjectReport,
    projection_camera: Option<&crate::ProjectionFitCameraReport>,
    evidence: &SceneGroundingEvidence,
    intrinsics: RotationFitIntrinsics,
    target: &RotationFitTarget,
    target_mask: &[u8],
    scene_placements: &[GroundedScenePlacement],
    config: &SceneVisibleSurfacePoseFitConfig<'_>,
    candidates_dir: &Path,
) -> Vec<VisibleSurfacePoseCandidate> {
    let mut candidates = Vec::new();
    push_visible_surface_pose_candidate(
        &mut candidates,
        mesh,
        baseline,
        baseline,
        fit_object,
        projection_camera,
        evidence,
        intrinsics,
        target,
        target_mask,
        scene_placements,
        "baseline",
        config,
        candidates_dir,
    );

    let mut best_placement = candidates
        .first()
        .map(|candidate| candidate.placement.clone())
        .unwrap_or_else(|| baseline.clone());
    let mut best_loss = candidates
        .first()
        .map(|candidate| candidate.loss)
        .unwrap_or(f32::INFINITY);
    let mut coarse = Vec::new();
    let semantic_prior_yaw = semantic_yaw_prior_for_placement(baseline, scene_placements);
    for (yaw, stage) in rotation_fit_candidate_yaws(baseline.rotation_y_degrees, None) {
        let mut trial = baseline.clone();
        trial.rotation_y_degrees = yaw;
        clamp_visible_surface_pose_candidate(&mut trial, baseline, config.scale_policy);
        if let Some(candidate) = visible_surface_pose_candidate(
            mesh,
            &trial,
            fit_object,
            projection_camera,
            evidence,
            intrinsics,
            target,
            target_mask,
            scene_placements,
            candidates.len() + coarse.len(),
            stage,
            config,
            candidates_dir,
        ) {
            if candidate.loss < best_loss {
                best_loss = candidate.loss;
                best_placement = candidate.placement.clone();
            }
            coarse.push(candidate);
        }
    }
    if let Some(prior_yaw) = semantic_prior_yaw {
        for delta in [-30.0, -15.0, -8.0, 0.0, 8.0, 15.0, 30.0] {
            let yaw = normalize_degrees(prior_yaw + delta);
            if coarse.iter().any(|candidate| {
                angular_distance_degrees(candidate.placement.rotation_y_degrees, yaw) <= 0.5
            }) {
                continue;
            }
            let mut trial = baseline.clone();
            trial.rotation_y_degrees = yaw;
            clamp_visible_surface_pose_candidate(&mut trial, baseline, config.scale_policy);
            if let Some(candidate) = visible_surface_pose_candidate(
                mesh,
                &trial,
                fit_object,
                projection_camera,
                evidence,
                intrinsics,
                target,
                target_mask,
                scene_placements,
                candidates.len() + coarse.len(),
                "semantic_yaw_prior",
                config,
                candidates_dir,
            ) {
                if candidate.loss < best_loss {
                    best_loss = candidate.loss;
                    best_placement = candidate.placement.clone();
                }
                coarse.push(candidate);
            }
        }
    }
    candidates.append(&mut coarse);

    if best_loss.is_finite() {
        for (yaw, stage) in rotation_fit_candidate_yaws(
            baseline.rotation_y_degrees,
            Some(best_placement.rotation_y_degrees),
        ) {
            let mut trial = best_placement.clone();
            trial.rotation_y_degrees = yaw;
            clamp_visible_surface_pose_candidate(&mut trial, baseline, config.scale_policy);
            if let Some(candidate) = visible_surface_pose_candidate(
                mesh,
                &trial,
                fit_object,
                projection_camera,
                evidence,
                intrinsics,
                target,
                target_mask,
                scene_placements,
                candidates.len(),
                stage,
                config,
                candidates_dir,
            ) {
                if candidate.loss < best_loss {
                    best_loss = candidate.loss;
                    best_placement = candidate.placement.clone();
                }
                candidates.push(candidate);
            }
        }
    }

    let steps = [
        (0.28, 0.13, 18.0),
        (0.16, 0.08, 10.0),
        (0.08, 0.045, 5.0),
        (0.04, 0.025, 2.5),
    ];
    for (move_step, scale_step, yaw_step) in steps {
        let mut improved = true;
        let mut pass_count = 0usize;
        while improved && pass_count < 3 {
            improved = false;
            pass_count += 1;
            let deltas = [
                VisibleSurfacePoseDelta::Translate(move_step, 0.0),
                VisibleSurfacePoseDelta::Translate(-move_step, 0.0),
                VisibleSurfacePoseDelta::Translate(0.0, move_step),
                VisibleSurfacePoseDelta::Translate(0.0, -move_step),
                VisibleSurfacePoseDelta::Scale(1.0 + scale_step),
                VisibleSurfacePoseDelta::Scale(1.0 - scale_step),
                VisibleSurfacePoseDelta::Yaw(yaw_step),
                VisibleSurfacePoseDelta::Yaw(-yaw_step),
            ];
            for delta in deltas {
                let mut trial = best_placement.clone();
                apply_visible_surface_pose_delta(&mut trial, delta, config.scale_policy);
                clamp_visible_surface_pose_candidate(&mut trial, baseline, config.scale_policy);
                if let Some(candidate) = visible_surface_pose_candidate(
                    mesh,
                    &trial,
                    fit_object,
                    projection_camera,
                    evidence,
                    intrinsics,
                    target,
                    target_mask,
                    scene_placements,
                    candidates.len(),
                    delta.stage_name(),
                    config,
                    candidates_dir,
                ) {
                    if candidate.loss + 1.0e-5 < best_loss {
                        best_loss = candidate.loss;
                        best_placement = candidate.placement.clone();
                        improved = true;
                    }
                    candidates.push(candidate);
                }
            }
        }
    }
    if best_loss.is_finite()
        && let Some((dense_placement, dense_report)) = dense_soft_surface_refine_pose(
            mesh,
            &best_placement,
            baseline,
            fit_object,
            evidence,
            intrinsics,
            target,
            target_mask,
            config,
        )
    {
        let mut trial = dense_placement;
        clamp_visible_surface_pose_candidate(&mut trial, baseline, config.scale_policy);
        if let Some(mut candidate) = visible_surface_pose_candidate(
            mesh,
            &trial,
            fit_object,
            projection_camera,
            evidence,
            intrinsics,
            target,
            target_mask,
            scene_placements,
            candidates.len(),
            "dense_soft_surface",
            config,
            candidates_dir,
        ) {
            candidate.dense_optimization = Some(dense_report);
            candidates.push(candidate);
        }
    }
    candidates
}

#[allow(clippy::too_many_arguments)]
fn push_visible_surface_pose_candidate(
    candidates: &mut Vec<VisibleSurfacePoseCandidate>,
    mesh: &RenderMesh,
    baseline: &GroundedScenePlacement,
    placement: &GroundedScenePlacement,
    fit_object: &crate::ProjectionFitObjectReport,
    projection_camera: Option<&crate::ProjectionFitCameraReport>,
    evidence: &SceneGroundingEvidence,
    intrinsics: RotationFitIntrinsics,
    target: &RotationFitTarget,
    target_mask: &[u8],
    scene_placements: &[GroundedScenePlacement],
    stage: &'static str,
    config: &SceneVisibleSurfacePoseFitConfig<'_>,
    candidates_dir: &Path,
) {
    let mut trial = placement.clone();
    clamp_visible_surface_pose_candidate(&mut trial, baseline, config.scale_policy);
    if let Some(candidate) = visible_surface_pose_candidate(
        mesh,
        &trial,
        fit_object,
        projection_camera,
        evidence,
        intrinsics,
        target,
        target_mask,
        scene_placements,
        candidates.len(),
        stage,
        config,
        candidates_dir,
    ) {
        candidates.push(candidate);
    }
}

#[allow(clippy::too_many_arguments)]
fn visible_surface_pose_candidate(
    mesh: &RenderMesh,
    placement: &GroundedScenePlacement,
    fit_object: &crate::ProjectionFitObjectReport,
    projection_camera: Option<&crate::ProjectionFitCameraReport>,
    evidence: &SceneGroundingEvidence,
    intrinsics: RotationFitIntrinsics,
    target: &RotationFitTarget,
    target_mask: &[u8],
    scene_placements: &[GroundedScenePlacement],
    candidate_index: usize,
    stage: &'static str,
    config: &SceneVisibleSurfacePoseFitConfig<'_>,
    candidates_dir: &Path,
) -> Option<VisibleSurfacePoseCandidate> {
    let surface = project_mesh_visible_surface_mask(
        mesh,
        placement,
        fit_object,
        projection_camera,
        evidence,
        intrinsics,
        target.crop_bbox,
    )?;
    let mask_iou = binary_mask_iou(&surface.mask, target_mask);
    let bbox_iou = surface
        .bbox
        .map(|bbox| normalized_bbox_iou(bbox, target.bbox))
        .unwrap_or(0.0);
    let (center_error, area_log2_error, aspect_log2_error) = surface
        .bbox
        .map(|bbox| {
            (
                distance2(bbox_center(bbox), bbox_center(target.bbox)),
                safe_log2_ratio(bbox_area(bbox), bbox_area(target.bbox)).abs(),
                safe_log2_ratio(bbox_aspect(bbox), bbox_aspect(target.bbox)).abs(),
            )
        })
        .unwrap_or((1.0, 6.0, 6.0));
    let depth_error_m = target.depth_median_m.and_then(|target_depth| {
        surface
            .median_depth_m
            .map(|observed| (observed - target_depth).abs())
    });
    let depth_passed = depth_error_m.is_none_or(|value| value <= config.max_depth_error_m);
    let depth_loss = depth_error_m
        .map(|value| (value / config.max_depth_error_m.max(0.05)).clamp(0.0, 4.0))
        .unwrap_or(0.20);
    let target_depth_summary = target
        .depth_stats
        .map(SurfaceDepthSummary::from_object_stats);
    let surface_depth_loss = surface.depth_summary.and_then(|observed| {
        target_depth_summary
            .map(|target| surface_depth_loss(target, observed, config.max_depth_error_m))
    });
    let dense_depth_comparison = target.dense_depth_crop.as_ref().and_then(|depth_crop| {
        dense_depth_comparison(
            depth_crop,
            target_mask,
            &surface.mask,
            &surface.depth_buffer,
            config.max_depth_error_m,
        )
    });
    let depth_distribution_loss = surface_depth_loss.unwrap_or(depth_loss);
    let target_moments = mask_point_moments(target_mask);
    let point_surface_loss = mask_point_surface_loss(target_moments, surface.mask_moments);
    let scale = representative_pose_scale(placement.scale);
    let target_scale = representative_pose_scale(fit_object.scale);
    let scale_loss = safe_log2_ratio(scale, target_scale).abs().min(3.0) * 0.08;
    let semantic_yaw_prior_degrees = semantic_yaw_prior_for_placement(placement, scene_placements);
    let semantic_yaw_error_degrees = semantic_yaw_prior_degrees
        .map(|prior| angular_distance_degrees(placement.rotation_y_degrees, prior));
    let semantic_yaw_loss = semantic_yaw_error_degrees
        .map(semantic_yaw_loss_for_error)
        .unwrap_or(0.0);
    let loss = (1.0 - mask_iou) * 2.35
        + (1.0 - bbox_iou) * 0.95
        + center_error * 1.25
        + area_log2_error * 0.28
        + aspect_log2_error * 0.18
        + depth_distribution_loss * 0.58
        + dense_depth_comparison
            .map(|comparison| comparison.normalized_loss * 0.52)
            .unwrap_or(0.0)
        + point_surface_loss * 0.46
        + scale_loss
        + semantic_yaw_loss;
    let surface_depth_passed = surface_depth_loss.is_none_or(|value| value <= 1.15 || depth_passed);
    let passed = depth_passed
        && surface_depth_passed
        && if target.mask_kind == "sam_rle" {
            mask_iou >= config.min_mask_iou
        } else {
            bbox_iou >= (config.min_mask_iou * 0.55).clamp(0.15, 0.45)
        };
    let mut candidate = VisibleSurfacePoseCandidate {
        candidate_index,
        stage,
        placement: placement.clone(),
        mask_iou,
        bbox_iou,
        center_error,
        area_log2_error,
        aspect_log2_error,
        depth_error_m,
        surface_depth_loss,
        dense_depth_loss: dense_depth_comparison.map(|comparison| comparison.normalized_loss),
        dense_depth_mae_m: dense_depth_comparison.map(|comparison| comparison.mae_m),
        dense_depth_sample_count: dense_depth_comparison
            .map(|comparison| comparison.sample_count)
            .unwrap_or(0),
        point_surface_loss,
        semantic_yaw_prior_degrees,
        semantic_yaw_error_degrees,
        semantic_yaw_loss,
        candidate_depth_summary: surface.depth_summary,
        target_depth_summary,
        surface_depth_passed,
        depth_passed,
        loss,
        passed,
        projected_bbox: surface.bbox,
        front_facing_face_count: surface.front_facing_face_count,
        covered_px: surface.covered_px,
        fallback_points: surface.fallback_points,
        artifact_path: None,
        dense_optimization: None,
    };
    if config.write_artifacts {
        let path = candidates_dir.join(format!(
            "candidate_{candidate_index:03}_{}_yaw_{:+04.0}.png",
            stage, placement.rotation_y_degrees
        ));
        if write_rotation_candidate_mask_png(&path, &surface.mask, target_mask).is_ok() {
            candidate.artifact_path = Some(path);
        }
    }
    Some(candidate)
}

pub(super) fn visible_surface_pose_candidate_passes_target(
    best: &VisibleSurfacePoseCandidate,
    baseline: &VisibleSurfacePoseCandidate,
    target: &RotationFitTarget,
    config: SceneVisibleSurfacePoseFitConfig<'_>,
) -> bool {
    if best.loss + VISIBLE_SURFACE_POSE_FIT_MIN_APPLY_IMPROVEMENT >= baseline.loss {
        return false;
    }
    if config.object_filter.is_refinement()
        && placement_is_table_like(&best.placement)
        && best.bbox_iou >= 0.72
        && best.bbox_iou > baseline.bbox_iou + 0.18
        && best.center_error <= 0.055
        && best.mask_iou > baseline.mask_iou + 0.035
    {
        return true;
    }
    if !best.depth_passed {
        return false;
    }
    if target.mask_kind == "sam_rle" {
        best.mask_iou >= config.min_mask_iou
            || best.mask_iou + 0.04 >= baseline.mask_iou && best.bbox_iou > baseline.bbox_iou
    } else {
        best.bbox_iou > baseline.bbox_iou + 0.03 || best.passed
    }
}

pub(super) fn visible_surface_pose_candidate_report(
    candidate: &VisibleSurfacePoseCandidate,
) -> Value {
    json!({
        "candidate_index": candidate.candidate_index,
        "stage": candidate.stage,
        "translation": candidate.placement.translation,
        "ground_point": candidate.placement.ground_point,
        "scale": candidate.placement.scale,
        "yaw_degrees": candidate.placement.rotation_y_degrees,
        "mask_iou": candidate.mask_iou,
        "bbox_iou": candidate.bbox_iou,
        "center_error": candidate.center_error,
        "area_log2_error": candidate.area_log2_error,
        "aspect_log2_error": candidate.aspect_log2_error,
        "depth_error_m": candidate.depth_error_m,
        "surface_depth_loss": candidate.surface_depth_loss,
        "dense_depth_loss": candidate.dense_depth_loss,
        "dense_depth_mae_m": candidate.dense_depth_mae_m,
        "dense_depth_sample_count": candidate.dense_depth_sample_count,
        "point_surface_loss": candidate.point_surface_loss,
        "semantic_yaw_prior_degrees": candidate.semantic_yaw_prior_degrees,
        "semantic_yaw_error_degrees": candidate.semantic_yaw_error_degrees,
        "semantic_yaw_loss": candidate.semantic_yaw_loss,
        "candidate_depth_summary": candidate
            .candidate_depth_summary
            .map(surface_depth_summary_report),
        "target_depth_summary": candidate
            .target_depth_summary
            .map(surface_depth_summary_report),
        "surface_depth_passed": candidate.surface_depth_passed,
        "depth_passed": candidate.depth_passed,
        "loss": candidate.loss,
        "passed": candidate.passed,
        "projected_bbox": candidate.projected_bbox,
        "front_facing_face_count": candidate.front_facing_face_count,
        "covered_px": candidate.covered_px,
        "fallback_points": candidate.fallback_points,
        "artifact_path": candidate.artifact_path.as_ref().map(|path| path.display().to_string()),
        "dense_optimization": candidate.dense_optimization.clone(),
    })
}

pub(super) fn semantic_yaw_prior_for_placement(
    placement: &GroundedScenePlacement,
    scene_placements: &[GroundedScenePlacement],
) -> Option<f32> {
    if !placement_is_chair_like(placement) {
        return None;
    }
    let from = placement.ground_point;
    let nearest_table = scene_placements
        .iter()
        .filter(|candidate| candidate.entity_id != placement.entity_id)
        .filter(|candidate| placement_is_table_like(candidate))
        .min_by(|left, right| {
            ground_distance2(from, left.ground_point)
                .total_cmp(&ground_distance2(from, right.ground_point))
        })?;
    yaw_toward_point_degrees(from, nearest_table.ground_point)
}

pub(super) fn semantic_yaw_loss_for_error(error_degrees: f32) -> f32 {
    let normalized = (error_degrees.abs() / 180.0).clamp(0.0, 1.0);
    normalized * normalized * 0.16
}

fn yaw_toward_point_degrees(from: [f32; 3], target: [f32; 3]) -> Option<f32> {
    let dx = target[0] - from[0];
    let dz = target[2] - from[2];
    if !dx.is_finite() || !dz.is_finite() || dx.abs() + dz.abs() <= 1.0e-5 {
        return None;
    }
    Some(normalize_degrees(dx.atan2(dz).to_degrees()))
}

fn angular_distance_degrees(left: f32, right: f32) -> f32 {
    normalize_degrees(left - right).abs()
}

fn ground_distance2(left: [f32; 3], right: [f32; 3]) -> f32 {
    let dx = left[0] - right[0];
    let dz = left[2] - right[2];
    dx * dx + dz * dz
}

fn placement_is_chair_like(placement: &GroundedScenePlacement) -> bool {
    let descriptor = placement_descriptor(placement);
    descriptor.contains("chair")
        || descriptor.contains("seat")
        || descriptor.contains("stool")
        || descriptor.contains("armchair")
}

fn placement_is_table_like(placement: &GroundedScenePlacement) -> bool {
    let descriptor = placement_descriptor(placement);
    descriptor.contains("table")
        || descriptor.contains("desk")
        || descriptor.contains("counter")
        || descriptor.contains("workstation")
}

fn placement_descriptor(placement: &GroundedScenePlacement) -> String {
    format!(
        "{} {} {} {}",
        placement.entity_id, placement.asset_id, placement.object_id, placement.label
    )
    .to_ascii_lowercase()
}

#[derive(Clone, Copy)]
enum VisibleSurfacePoseDelta {
    Translate(f32, f32),
    Scale(f32),
    Yaw(f32),
}

impl VisibleSurfacePoseDelta {
    fn stage_name(self) -> &'static str {
        match self {
            Self::Translate(_, _) => "coordinate_translate",
            Self::Scale(_) => "coordinate_scale",
            Self::Yaw(_) => "coordinate_yaw",
        }
    }
}

pub(super) fn command_placement_for_pose_fit(
    placement: &GroundedScenePlacement,
    command: Option<&Value>,
    config: SceneVisibleSurfacePoseFitConfig<'_>,
) -> GroundedScenePlacement {
    let mut out = placement.clone();
    if let Some(command) = command {
        if let Some(translation) = command.get("translation").and_then(json_array3) {
            out.translation = translation;
            out.ground_point[0] = translation[0];
            out.ground_point[2] = translation[2];
        }
        if let Some(scale) = command.get("scale").and_then(json_array3) {
            out.scale = config.scale_policy.apply_to_scale(scale);
        }
        if let Some(yaw) = command
            .get("rotation")
            .and_then(json_array4)
            .map(quat_y_degrees)
        {
            out.rotation_y_degrees = normalize_degrees(yaw);
        }
    }
    out.translation[1] = -out.local_aabb.min[1] * out.scale[1];
    out
}

pub(super) fn apply_pose_fit_candidate_to_command(
    command: &mut Value,
    placement: &GroundedScenePlacement,
    scale_policy: SceneScalePolicy,
) {
    command["translation"] = json!(placement.translation);
    command["rotation"] = json!(quat_from_y_degrees(placement.rotation_y_degrees));
    command["scale"] = json!(scale_policy.apply_to_scale(placement.scale));
}

fn apply_visible_surface_pose_delta(
    placement: &mut GroundedScenePlacement,
    delta: VisibleSurfacePoseDelta,
    scale_policy: SceneScalePolicy,
) {
    match delta {
        VisibleSurfacePoseDelta::Translate(dx, dz) => {
            placement.translation[0] += dx;
            placement.translation[2] += dz;
            placement.ground_point[0] += dx;
            placement.ground_point[2] += dz;
        }
        VisibleSurfacePoseDelta::Scale(multiplier) => {
            placement.scale = scale_policy.apply_to_scale([
                placement.scale[0] * multiplier,
                placement.scale[1] * multiplier,
                placement.scale[2] * multiplier,
            ]);
            placement.translation[1] = -placement.local_aabb.min[1] * placement.scale[1];
        }
        VisibleSurfacePoseDelta::Yaw(delta) => {
            placement.rotation_y_degrees = normalize_degrees(placement.rotation_y_degrees + delta);
        }
    }
}

pub(super) fn clamp_visible_surface_pose_candidate(
    placement: &mut GroundedScenePlacement,
    baseline: &GroundedScenePlacement,
    scale_policy: SceneScalePolicy,
) {
    let max_drift = (baseline.ground_anchor_max_drift_m() * 1.15).clamp(0.12, 0.75);
    let dx = placement.ground_point[0] - baseline.ground_point[0];
    let dz = placement.ground_point[2] - baseline.ground_point[2];
    let drift = (dx * dx + dz * dz).sqrt();
    if drift.is_finite() && drift > max_drift {
        let scale = max_drift / drift;
        placement.ground_point[0] = baseline.ground_point[0] + dx * scale;
        placement.ground_point[2] = baseline.ground_point[2] + dz * scale;
        placement.translation[0] = placement.ground_point[0];
        placement.translation[2] = placement.ground_point[2];
    }

    let baseline_scale = representative_pose_scale(baseline.scale);
    let (min_scale_multiplier, max_scale_multiplier) =
        visible_surface_scale_bounds_for_placement(baseline);
    let scale = representative_pose_scale(placement.scale).clamp(
        (baseline_scale * min_scale_multiplier).max(0.05),
        (baseline_scale * max_scale_multiplier).min(20.0),
    );
    let current = representative_pose_scale(placement.scale).max(1.0e-5);
    let multiplier = scale / current;
    placement.scale = scale_policy.apply_to_scale([
        placement.scale[0] * multiplier,
        placement.scale[1] * multiplier,
        placement.scale[2] * multiplier,
    ]);
    placement.translation[1] = -placement.local_aabb.min[1] * placement.scale[1];
}

fn visible_surface_scale_bounds_for_placement(placement: &GroundedScenePlacement) -> (f32, f32) {
    if placement_is_table_like(placement) {
        (0.88, 1.14)
    } else {
        (0.72, 1.38)
    }
}

pub(super) fn representative_pose_scale(scale: [f32; 3]) -> f32 {
    let mut values = [
        scale[0].abs().clamp(0.05, 20.0),
        scale[1].abs().clamp(0.05, 20.0),
        scale[2].abs().clamp(0.05, 20.0),
    ];
    values.sort_by(f32::total_cmp);
    values[1]
}

pub(super) fn normalize_reused_layout_scales(
    placements: &mut [GroundedScenePlacement],
    scale_policy: SceneScalePolicy,
) {
    let mut groups: HashMap<String, ([f32; 3], usize)> = HashMap::new();
    for placement in placements.iter() {
        let entry = groups
            .entry(placement.asset_id.clone())
            .or_insert(([0.0; 3], 0));
        for axis in 0..3 {
            entry.0[axis] += placement.scale[axis].abs().clamp(0.05, 20.0);
        }
        entry.1 += 1;
    }
    let repeated = groups
        .into_iter()
        .filter_map(|(asset_id, (sum, count))| {
            (count > 1).then_some((
                asset_id,
                scale_policy.apply_to_scale([
                    sum[0] / count as f32,
                    sum[1] / count as f32,
                    sum[2] / count as f32,
                ]),
            ))
        })
        .collect::<HashMap<_, _>>();
    for placement in placements {
        let Some(scale) = repeated.get(&placement.asset_id).copied() else {
            continue;
        };
        placement.scale = scale;
        placement.translation[1] = -placement.local_aabb.min[1] * scale[1];
    }
}

pub(super) fn sync_layout_placements_from_commands(
    layout: &mut GroundedSceneLayout,
    commands: &[Value],
) {
    let mut placement_index = 0usize;
    for command in commands {
        let command_type = command.get("type").and_then(Value::as_str).unwrap_or("");
        if command_type != "spawn_cached" && command_type != "spawn_path" {
            continue;
        }
        let Some(placement) = layout.placements.get_mut(placement_index) else {
            break;
        };
        if let Some(translation) = command.get("translation").and_then(json_array3) {
            placement.translation = translation;
            placement.ground_point[0] = translation[0];
            placement.ground_point[2] = translation[2];
        }
        if let Some(scale) = command.get("scale").and_then(json_array3) {
            placement.scale = scale;
            placement.translation[1] = -placement.local_aabb.min[1] * scale[1];
        }
        if let Some(yaw) = command
            .get("rotation")
            .and_then(json_array4)
            .map(quat_y_degrees)
        {
            placement.rotation_y_degrees = normalize_degrees(yaw);
        }
        placement_index += 1;
    }
}

pub(super) fn sync_commands_from_layout_placements(
    commands: &mut [Value],
    placements: &[GroundedScenePlacement],
) {
    let mut placement_index = 0usize;
    for command in commands {
        let command_type = command.get("type").and_then(Value::as_str).unwrap_or("");
        if command_type != "spawn_cached" && command_type != "spawn_path" {
            continue;
        }
        let Some(placement) = placements.get(placement_index) else {
            break;
        };
        command["translation"] = json!(placement.translation);
        command["rotation"] = json!(quat_from_y_degrees(placement.rotation_y_degrees));
        command["scale"] = json!(placement.scale);
        placement_index += 1;
    }
}

pub(super) fn rotation_fit_candidate_yaws(
    current_yaw: f32,
    refine_around: Option<f32>,
) -> Vec<(f32, &'static str)> {
    let mut yaws = Vec::new();
    let mut push = |yaw: f32, stage: &'static str| {
        let yaw = normalize_degrees(yaw);
        if yaws.iter().any(|(existing, _): &(f32, &'static str)| {
            (normalize_degrees(*existing - yaw)).abs() < 0.25
        }) {
            return;
        }
        yaws.push((yaw, stage));
    };
    if let Some(center) = refine_around {
        for delta in [-20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0] {
            push(center + delta, "fine");
        }
    } else {
        for delta in [
            -180.0, -150.0, -120.0, -90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0,
        ] {
            push(current_yaw + delta, "coarse");
        }
    }
    yaws
}

pub(super) fn rotation_fit_candidate_report(candidate: &RotationFitCandidate) -> Value {
    json!({
        "candidate_index": candidate.candidate_index,
        "stage": candidate.stage,
        "yaw_degrees": candidate.yaw_degrees,
        "yaw_delta_degrees": candidate.yaw_delta_degrees,
        "mask_iou": candidate.mask_iou,
        "bbox_iou": candidate.bbox_iou,
        "depth_error_m": candidate.depth_error_m,
        "depth_passed": candidate.depth_passed,
        "loss": candidate.loss,
        "passed": candidate.passed,
        "projected_bbox": candidate.projected_bbox,
        "front_facing_face_count": candidate.front_facing_face_count,
        "covered_px": candidate.covered_px,
        "fallback_points": candidate.fallback_points,
        "artifact_path": candidate.artifact_path.as_ref().map(|path| path.display().to_string()),
    })
}

pub(super) fn rotation_fit_target_for_placement(
    placement: &GroundedScenePlacement,
    evidence: &SceneGroundingEvidence,
    depth_sidecar: Option<&LoadedSceneDepthMap>,
) -> Option<RotationFitTarget> {
    let object = evidence
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
        })?;
    let mask = object.mask.as_ref();
    let (binary, bbox, mask_kind) = if let Some(mask) = mask {
        if !mask.mask_rle.is_empty() {
            (
                BinaryMask::decode_rle(mask.image_size[0], mask.image_size[1], &mask.mask_rle)
                    .ok()?,
                mask.bbox,
                "sam_rle",
            )
        } else {
            (
                BinaryMask::from_normalized_bbox(mask.image_size[0], mask.image_size[1], mask.bbox)
                    .ok()?,
                mask.bbox,
                "mask_bbox_fallback",
            )
        }
    } else {
        let [width, height] = evidence
            .camera
            .image_size
            .or_else(|| evidence.depth.as_ref().and_then(|depth| depth.image_size))
            .unwrap_or([1024, 512]);
        (
            BinaryMask::from_normalized_bbox(width, height, placement.source_bbox).ok()?,
            placement.source_bbox,
            "placement_bbox_fallback",
        )
    };
    let crop_bbox = padded_bbox(bbox, 0.14);
    let dense_depth_crop = depth_sidecar.and_then(|depth_map| {
        dense_depth_crop_from_sidecar(
            depth_map,
            crop_bbox,
            ROTATION_FIT_CROP_RESOLUTION as usize,
            object.depth_stats.map(|stats| stats.median_m),
        )
    });
    Some(RotationFitTarget {
        mask: binary,
        bbox,
        crop_bbox,
        depth_median_m: object.depth_stats.map(|stats| stats.median_m),
        depth_stats: object.depth_stats,
        dense_depth_crop,
        mask_kind,
    })
}

pub(super) fn rotation_fit_target_crop_mask(target: &RotationFitTarget) -> Vec<u8> {
    let res = ROTATION_FIT_CROP_RESOLUTION as usize;
    let mut out = vec![0_u8; res * res];
    let width = target.mask.width().max(1);
    let height = target.mask.height().max(1);
    let data = target.mask.data();
    for y in 0..res {
        for x in 0..res {
            let u = target.crop_bbox[0]
                + ((x as f32 + 0.5) / res as f32) * (target.crop_bbox[2] - target.crop_bbox[0]);
            let v = target.crop_bbox[1]
                + ((y as f32 + 0.5) / res as f32) * (target.crop_bbox[3] - target.crop_bbox[1]);
            let px = (u.clamp(0.0, 1.0) * (width - 1) as f32).round() as u32;
            let py = (v.clamp(0.0, 1.0) * (height - 1) as f32).round() as u32;
            let index = py as usize * width as usize + px as usize;
            out[y * res + x] = data.get(index).copied().unwrap_or(0);
        }
    }
    out
}

pub(super) fn rotation_fit_intrinsics(
    evidence: &SceneGroundingEvidence,
) -> Option<RotationFitIntrinsics> {
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
    let principal = evidence
        .camera
        .principal_point
        .unwrap_or([(width - 1.0) * 0.5, (height - 1.0) * 0.5]);
    Some(RotationFitIntrinsics {
        fx: fy,
        fy,
        cx: principal[0],
        cy: principal[1],
        width,
        height,
    })
}

pub(super) fn rotation_fit_asset_path(
    placement: &GroundedScenePlacement,
    command: Option<&Value>,
    asset_bindings: &[SceneAssetBinding],
    output_dir: &Path,
) -> Option<PathBuf> {
    command
        .and_then(|command| command.get("path").and_then(Value::as_str))
        .map(PathBuf::from)
        .or_else(|| {
            asset_bindings
                .iter()
                .find(|binding| binding.asset_id == placement.asset_id)
                .or_else(|| {
                    asset_bindings
                        .iter()
                        .find(|binding| binding.object_id == placement.object_id)
                })
                .and_then(|binding| binding.path.as_deref())
                .map(PathBuf::from)
        })
        .map(|path| {
            if path.exists() || path.is_absolute() {
                path
            } else {
                output_dir.join(path)
            }
        })
}

pub(super) fn load_rotation_fit_mesh(path: &Path) -> Result<RenderMesh, String> {
    burn_synth_render::mesh::load_glb_mesh(path)
}

pub(super) fn placement_with_yaw(
    placement: &GroundedScenePlacement,
    current_yaw: f32,
) -> GroundedScenePlacement {
    let mut out = placement.clone();
    out.rotation_y_degrees = current_yaw;
    out
}

pub(super) fn projection_fit_object_matches_placement(
    object: &crate::ProjectionFitObjectReport,
    placement: &GroundedScenePlacement,
) -> bool {
    object.object_id == placement.object_id
        && object.instance_id.as_deref() == placement.instance_id.as_deref()
}

pub(super) fn rotation_fit_spawn_command_indices(commands: &[Value]) -> Vec<usize> {
    commands
        .iter()
        .enumerate()
        .filter_map(|(index, command)| {
            let command_type = command.get("type").and_then(Value::as_str)?;
            matches!(command_type, "spawn_cached" | "spawn_path").then_some(index)
        })
        .collect()
}

pub(super) fn rotation_fit_object_dir_name(
    index: usize,
    placement: &GroundedScenePlacement,
) -> String {
    let label = format!(
        "{}_{}",
        placement.object_id,
        placement.instance_id.as_deref().unwrap_or("instance")
    );
    let slug = label
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>();
    format!("{index:02}_{slug}")
}

#[allow(clippy::too_many_arguments)]
pub(super) fn evaluate_rotation_fit_candidates(
    mesh: &RenderMesh,
    placement: &GroundedScenePlacement,
    fit_object: &crate::ProjectionFitObjectReport,
    evidence: &SceneGroundingEvidence,
    intrinsics: RotationFitIntrinsics,
    target: &RotationFitTarget,
    target_mask: &[u8],
    current_yaw: f32,
    config: &SceneRotationFitConfig<'_>,
    candidates_dir: &Path,
) -> Vec<RotationFitCandidate> {
    let mut candidates = Vec::new();
    let mut coarse_candidates = Vec::new();
    for (yaw, stage) in rotation_fit_candidate_yaws(current_yaw, None) {
        if let Some(candidate) = evaluate_rotation_fit_candidate(
            mesh,
            placement,
            fit_object,
            evidence,
            intrinsics,
            target,
            target_mask,
            current_yaw,
            yaw,
            stage,
            coarse_candidates.len(),
            config,
            candidates_dir,
        ) {
            coarse_candidates.push(candidate);
        }
    }
    let best_coarse_yaw = coarse_candidates
        .iter()
        .min_by(|left, right| left.loss.total_cmp(&right.loss))
        .map(|candidate| candidate.yaw_degrees);
    candidates.append(&mut coarse_candidates);
    if let Some(best_yaw) = best_coarse_yaw {
        for (yaw, stage) in rotation_fit_candidate_yaws(current_yaw, Some(best_yaw)) {
            if candidates
                .iter()
                .any(|candidate| (candidate.yaw_degrees - yaw).abs() <= 0.25)
            {
                continue;
            }
            if let Some(candidate) = evaluate_rotation_fit_candidate(
                mesh,
                placement,
                fit_object,
                evidence,
                intrinsics,
                target,
                target_mask,
                current_yaw,
                yaw,
                stage,
                candidates.len(),
                config,
                candidates_dir,
            ) {
                candidates.push(candidate);
            }
        }
    }
    candidates
}

#[allow(clippy::too_many_arguments)]
fn evaluate_rotation_fit_candidate(
    mesh: &RenderMesh,
    placement: &GroundedScenePlacement,
    fit_object: &crate::ProjectionFitObjectReport,
    evidence: &SceneGroundingEvidence,
    intrinsics: RotationFitIntrinsics,
    target: &RotationFitTarget,
    target_mask: &[u8],
    current_yaw: f32,
    yaw: f32,
    stage: &'static str,
    candidate_index: usize,
    config: &SceneRotationFitConfig<'_>,
    candidates_dir: &Path,
) -> Option<RotationFitCandidate> {
    let mut candidate_placement = placement.clone();
    candidate_placement.rotation_y_degrees = yaw;
    let surface = project_mesh_visible_surface_mask(
        mesh,
        &candidate_placement,
        fit_object,
        None,
        evidence,
        intrinsics,
        target.crop_bbox,
    )?;
    let mask_iou = binary_mask_iou(&surface.mask, target_mask);
    let bbox_iou = surface
        .bbox
        .map(|bbox| normalized_bbox_iou(bbox, target.bbox))
        .unwrap_or(0.0);
    let depth_error_m = target.depth_median_m.and_then(|target_depth| {
        surface
            .median_depth_m
            .map(|observed| (observed - target_depth).abs())
    });
    let depth_passed = depth_error_m.is_none_or(|value| value <= config.max_depth_error_m);
    let depth_loss = depth_error_m
        .map(|value| (value / config.max_depth_error_m.max(0.05)).clamp(0.0, 4.0))
        .unwrap_or(0.15);
    let yaw_delta = normalize_degrees(yaw - current_yaw);
    let yaw_prior_loss = (yaw_delta.abs() / 180.0).min(1.0) * 0.08;
    let loss = (1.0 - mask_iou) * 2.00 + (1.0 - bbox_iou) * 0.70 + depth_loss + yaw_prior_loss;
    let passed = mask_iou >= config.min_mask_iou && depth_passed;
    let mut candidate = RotationFitCandidate {
        candidate_index,
        stage,
        yaw_degrees: yaw,
        yaw_delta_degrees: yaw_delta,
        mask_iou,
        bbox_iou,
        depth_error_m,
        depth_passed,
        loss,
        passed,
        projected_bbox: surface.bbox,
        front_facing_face_count: surface.front_facing_face_count,
        covered_px: surface.covered_px,
        fallback_points: surface.fallback_points,
        artifact_path: None,
    };
    if config.write_artifacts {
        let path = candidates_dir.join(format!(
            "candidate_{candidate_index:03}_yaw_{:+04.0}.png",
            yaw
        ));
        if write_rotation_candidate_mask_png(&path, &surface.mask, target_mask).is_ok() {
            candidate.artifact_path = Some(path);
        }
    }
    Some(candidate)
}

fn padded_bbox(bbox: [f32; 4], pad_fraction: f32) -> [f32; 4] {
    let w = (bbox[2] - bbox[0]).abs();
    let h = (bbox[3] - bbox[1]).abs();
    let pad = w.max(h).max(0.01) * pad_fraction;
    [
        (bbox[0] - pad).clamp(0.0, 1.0),
        (bbox[1] - pad).clamp(0.0, 1.0),
        (bbox[2] + pad).clamp(0.0, 1.0),
        (bbox[3] + pad).clamp(0.0, 1.0),
    ]
}

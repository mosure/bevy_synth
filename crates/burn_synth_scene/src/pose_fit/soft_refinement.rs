use super::pose_search::{clamp_visible_surface_pose_candidate, representative_pose_scale};
use super::visible_surface::{rotation_fit_source_floor_y_at, square_mask_resolution};
use super::*;
use burn_synth_render::{
    CameraIntrinsics as SoftCameraIntrinsics, ObjectTransformValues as SoftObjectTransformValues,
    SoftPoseOptimizationConfig, SoftPoseOptimizationResult, SoftRenderConfig,
    optimize_soft_pose_ndarray,
};

#[allow(clippy::too_many_arguments)]
pub(super) fn dense_soft_surface_refine_pose(
    mesh: &RenderMesh,
    placement: &GroundedScenePlacement,
    baseline: &GroundedScenePlacement,
    fit_object: &crate::ProjectionFitObjectReport,
    evidence: &SceneGroundingEvidence,
    intrinsics: RotationFitIntrinsics,
    target: &RotationFitTarget,
    target_mask: &[u8],
    config: &SceneVisibleSurfacePoseFitConfig<'_>,
) -> Option<(GroundedScenePlacement, Value)> {
    let points = dense_soft_surface_points(mesh, DENSE_SOFT_FIT_MAX_POINTS);
    if points.len() < 4 {
        return None;
    }
    let transform_context =
        DenseSoftSurfaceTransformContext::from_placement(placement, fit_object, evidence)?;
    let initial_transform = transform_context.to_soft_transform(placement);
    let dense_depth_target = target
        .dense_depth_crop
        .as_ref()
        .and_then(|crop| {
            resample_dense_depth_crop(crop, DENSE_SOFT_FIT_RESOLUTION, DENSE_SOFT_FIT_RESOLUTION)
        })
        .filter(|(_, _, valid_count)| *valid_count >= 8);
    let render_config = SoftRenderConfig {
        width: DENSE_SOFT_FIT_RESOLUTION,
        height: DENSE_SOFT_FIT_RESOLUTION,
        sigma_px: 1.75,
        depth_sigma_m: config.max_depth_error_m.max(0.08),
        mask_weight: 1.0,
        depth_weight: if dense_depth_target.is_some() {
            0.34
        } else if target.depth_median_m.is_some() {
            0.16
        } else {
            0.0
        },
    };
    let mut target_mask_f32 = resample_square_mask_to_f32(
        target_mask,
        DENSE_SOFT_FIT_RESOLUTION,
        DENSE_SOFT_FIT_RESOLUTION,
    )?;
    if target_mask_f32
        .iter()
        .filter(|value| **value >= 0.5)
        .count()
        < 8
    {
        return None;
    }
    let (target_depth, dense_depth_valid_count, depth_source) =
        if let Some((depth, valid, valid_count)) = dense_depth_target {
            for (mask, valid) in target_mask_f32.iter_mut().zip(valid) {
                *mask *= valid;
            }
            (depth, valid_count, "depth_sidecar_crop")
        } else {
            (
                vec![
                    target
                        .depth_median_m
                        .unwrap_or(initial_transform.tz)
                        .max(1.0e-4);
                    DENSE_SOFT_FIT_RESOLUTION * DENSE_SOFT_FIT_RESOLUTION
                ],
                0,
                "depth_median_constant",
            )
        };
    if target_mask_f32
        .iter()
        .filter(|value| **value >= 0.5)
        .count()
        < 8
    {
        return None;
    }
    let soft_intrinsics =
        soft_camera_intrinsics_for_crop(intrinsics, target.crop_bbox, DENSE_SOFT_FIT_RESOLUTION)?;
    let baseline_scale = representative_pose_scale(baseline.scale);
    let started = Instant::now();
    let result = optimize_soft_pose_ndarray(
        &points,
        initial_transform,
        soft_intrinsics,
        render_config,
        &target_mask_f32,
        &target_depth,
        SoftPoseOptimizationConfig {
            iterations: DENSE_SOFT_FIT_ITERATIONS,
            learning_rate_translation: 0.006,
            learning_rate_yaw: 0.003,
            learning_rate_scale: 0.003,
            min_scale: (baseline_scale * 0.72).max(0.05),
            max_scale: (baseline_scale * 1.38).min(20.0),
            max_translation_step: (baseline.ground_anchor_max_drift_m() * 0.20).clamp(0.01, 0.05),
            optimize_ty: false,
        },
    );
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    if !result.final_loss.is_finite() || result.final_loss >= result.initial_loss {
        return Some((
            placement.clone(),
            dense_soft_surface_report(
                &result,
                elapsed_ms,
                points.len(),
                target.crop_bbox,
                dense_depth_valid_count,
                depth_source,
                false,
                "loss_not_improved",
            ),
        ));
    }
    let mut refined = transform_context.placement_from_soft_transform(placement, result.transform);
    refined.scale = config.scale_policy.apply_to_scale(refined.scale);
    refined.translation[1] = -refined.local_aabb.min[1] * refined.scale[1];
    refined.ground_point[0] = refined.translation[0];
    refined.ground_point[2] = refined.translation[2];
    clamp_visible_surface_pose_candidate(&mut refined, baseline, config.scale_policy);
    Some((
        refined,
        dense_soft_surface_report(
            &result,
            elapsed_ms,
            points.len(),
            target.crop_bbox,
            dense_depth_valid_count,
            depth_source,
            true,
            "optimized",
        ),
    ))
}

#[derive(Clone, Copy, Debug)]
pub(super) struct DenseSoftSurfaceTransformContext {
    origin_xz: [f32; 2],
    floor_y_camera: f32,
}

impl DenseSoftSurfaceTransformContext {
    pub(super) fn from_placement(
        placement: &GroundedScenePlacement,
        fit_object: &crate::ProjectionFitObjectReport,
        evidence: &SceneGroundingEvidence,
    ) -> Option<Self> {
        let origin_xz = fit_object.source_camera_origin_xz?;
        let anchor = fit_object.source_camera_anchor?;
        let camera_x = placement.translation[0] + origin_xz[0];
        let camera_z = origin_xz[1] - placement.translation[2];
        let ground_anchor_basis_camera_ray =
            fit_object.ground_anchor_basis == "camera-ray-ground-plane";
        let floor_y_camera = if ground_anchor_basis_camera_ray {
            rotation_fit_source_floor_y_at(evidence, camera_x, camera_z).unwrap_or(anchor[1])
        } else {
            anchor[1]
        };
        Some(Self {
            origin_xz,
            floor_y_camera,
        })
    }

    pub(super) fn to_soft_transform(
        self,
        placement: &GroundedScenePlacement,
    ) -> SoftObjectTransformValues {
        SoftObjectTransformValues {
            tx: self.origin_xz[0] + placement.translation[0],
            ty: self.floor_y_camera - placement.translation[1],
            tz: self.origin_xz[1] - placement.translation[2],
            yaw: -placement.rotation_y_degrees.to_radians(),
            scale: representative_pose_scale(placement.scale),
        }
    }

    pub(super) fn placement_from_soft_transform(
        self,
        placement: &GroundedScenePlacement,
        transform: SoftObjectTransformValues,
    ) -> GroundedScenePlacement {
        let mut out = placement.clone();
        out.translation[0] = transform.tx - self.origin_xz[0];
        out.translation[2] = self.origin_xz[1] - transform.tz;
        out.rotation_y_degrees = normalize_degrees(-transform.yaw.to_degrees());
        out.scale = [transform.scale, transform.scale, transform.scale];
        out.translation[1] = -out.local_aabb.min[1] * out.scale[1];
        out.ground_point[0] = out.translation[0];
        out.ground_point[2] = out.translation[2];
        out
    }
}

pub(super) fn dense_soft_surface_points(mesh: &RenderMesh, max_points: usize) -> Vec<[f32; 3]> {
    let mut points = Vec::new();
    for vertex in &mesh.vertices {
        points.push([vertex[0], -vertex[1], -vertex[2]]);
    }
    for face in &mesh.faces {
        let indices = [face[0] as usize, face[1] as usize, face[2] as usize];
        if indices.iter().any(|index| *index >= mesh.vertices.len()) {
            continue;
        }
        let a = mesh.vertices[indices[0]];
        let b = mesh.vertices[indices[1]];
        let c = mesh.vertices[indices[2]];
        let centroid = [
            (a[0] + b[0] + c[0]) / 3.0,
            (a[1] + b[1] + c[1]) / 3.0,
            (a[2] + b[2] + c[2]) / 3.0,
        ];
        points.push([centroid[0], -centroid[1], -centroid[2]]);
    }
    if points.len() <= max_points {
        return points;
    }
    let stride = ((points.len() as f32) / max_points.max(1) as f32).ceil() as usize;
    points
        .into_iter()
        .enumerate()
        .filter_map(|(index, point)| (index % stride == 0).then_some(point))
        .take(max_points)
        .collect()
}

pub(super) fn soft_camera_intrinsics_for_crop(
    intrinsics: RotationFitIntrinsics,
    crop_bbox: [f32; 4],
    resolution: usize,
) -> Option<SoftCameraIntrinsics> {
    let crop_w = (crop_bbox[2] - crop_bbox[0]).abs();
    let crop_h = (crop_bbox[3] - crop_bbox[1]).abs();
    if crop_w <= 1.0e-5 || crop_h <= 1.0e-5 {
        return None;
    }
    let full_w = (intrinsics.width - 1.0).max(1.0);
    let full_h = (intrinsics.height - 1.0).max(1.0);
    let scale = resolution as f32;
    Some(SoftCameraIntrinsics {
        fx: intrinsics.fx / full_w / crop_w * scale,
        fy: intrinsics.fy / full_h / crop_h * scale,
        cx: ((intrinsics.cx / full_w) - crop_bbox[0]) / crop_w * scale,
        cy: ((intrinsics.cy / full_h) - crop_bbox[1]) / crop_h * scale,
        width: resolution,
        height: resolution,
    })
}

fn resample_square_mask_to_f32(mask: &[u8], width: usize, height: usize) -> Option<Vec<f32>> {
    let source_res = square_mask_resolution(mask)?;
    let mut out = vec![0.0; width * height];
    for y in 0..height {
        for x in 0..width {
            let sx = ((x as f32 + 0.5) / width as f32 * source_res as f32)
                .floor()
                .clamp(0.0, (source_res - 1) as f32) as usize;
            let sy = ((y as f32 + 0.5) / height as f32 * source_res as f32)
                .floor()
                .clamp(0.0, (source_res - 1) as f32) as usize;
            out[y * width + x] = if mask[sy * source_res + sx] != 0 {
                1.0
            } else {
                0.0
            };
        }
    }
    Some(out)
}

#[allow(clippy::too_many_arguments)]
fn dense_soft_surface_report(
    result: &SoftPoseOptimizationResult,
    elapsed_ms: f64,
    point_count: usize,
    crop_bbox: [f32; 4],
    dense_depth_valid_count: usize,
    depth_source: &str,
    applied_to_candidate: bool,
    status: &str,
) -> Value {
    json!({
        "status": status,
        "renderer": "burn_synth_render::soft_point_surface",
        "method": "autodiff-soft-crop-silhouette-depth",
        "optimized_fields": ["translation_x_camera", "translation_z_camera", "yaw_y", "uniform_scale"],
        "fixed_fields": ["translation_y_floor_contact"],
        "depth_source": depth_source,
        "dense_depth_valid_count": dense_depth_valid_count,
        "crop_bbox": crop_bbox,
        "render_resolution": [DENSE_SOFT_FIT_RESOLUTION, DENSE_SOFT_FIT_RESOLUTION],
        "point_count": point_count,
        "iteration_count": result.steps.len(),
        "elapsed_ms": elapsed_ms,
        "initial_soft_loss": result.initial_loss,
        "final_soft_loss": result.final_loss,
        "loss_delta": result.final_loss - result.initial_loss,
        "applied_to_candidate": applied_to_candidate,
        "initial_transform": result.steps.first().map(|step| step.transform),
        "final_transform": result.transform,
        "steps": result.steps,
    })
}

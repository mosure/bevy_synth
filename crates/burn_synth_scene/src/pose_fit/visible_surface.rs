use super::*;

#[derive(Clone)]
pub(super) struct ProjectedCandidateSurface {
    pub(super) mask: Vec<u8>,
    pub(super) depth_buffer: Vec<f32>,
    pub(super) bbox: Option<[f32; 4]>,
    pub(super) median_depth_m: Option<f32>,
    pub(super) depth_summary: Option<SurfaceDepthSummary>,
    pub(super) mask_moments: MaskPointMoments,
    pub(super) front_facing_face_count: usize,
    pub(super) covered_px: usize,
    pub(super) fallback_points: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct SurfaceDepthSummary {
    pub(super) min_m: f32,
    pub(super) p10_m: f32,
    pub(super) median_m: f32,
    pub(super) p90_m: f32,
    pub(super) max_m: f32,
    pub(super) contact_m: Option<f32>,
    pub(super) sample_count: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct MaskPointMoments {
    pub(super) centroid: Option<[f32; 2]>,
    pub(super) variance: Option<[f32; 2]>,
    pub(super) coverage: f32,
}

pub(super) fn project_mesh_visible_surface_mask(
    mesh: &RenderMesh,
    placement: &GroundedScenePlacement,
    fit_object: &crate::ProjectionFitObjectReport,
    projection_camera: Option<&crate::ProjectionFitCameraReport>,
    evidence: &SceneGroundingEvidence,
    intrinsics: RotationFitIntrinsics,
    crop_bbox: [f32; 4],
) -> Option<ProjectedCandidateSurface> {
    let projection =
        RotationFitProjection::from_fit_object(fit_object, projection_camera, intrinsics)?;
    let res = ROTATION_FIT_CROP_RESOLUTION as usize;
    let mut mask = vec![0_u8; res * res];
    let mut depth_buffer = vec![f32::INFINITY; res * res];
    let mut projected_points = Vec::new();
    let mut depths = Vec::new();
    let mut front_facing_face_count = 0usize;

    for face in &mesh.faces {
        let indices = [face[0] as usize, face[1] as usize, face[2] as usize];
        if indices.iter().any(|index| *index >= mesh.vertices.len()) {
            continue;
        }
        let mut camera_points = [[0.0; 3]; 3];
        let mut projected = [[0.0; 2]; 3];
        let mut valid = true;
        for (slot, index) in indices.iter().copied().enumerate() {
            let world = rotation_fit_transform_local_point(placement, mesh.vertices[index]);
            let Some((camera_point, projected_point)) = projection.project(world, evidence) else {
                valid = false;
                break;
            };
            camera_points[slot] = camera_point;
            projected[slot] = projected_point;
        }
        if !valid {
            continue;
        }
        let normal = cross3(
            sub3(camera_points[1], camera_points[0]),
            sub3(camera_points[2], camera_points[0]),
        );
        let centroid = [
            (camera_points[0][0] + camera_points[1][0] + camera_points[2][0]) / 3.0,
            (camera_points[0][1] + camera_points[1][1] + camera_points[2][1]) / 3.0,
            (camera_points[0][2] + camera_points[1][2] + camera_points[2][2]) / 3.0,
        ];
        if dot3(normal, [-centroid[0], -centroid[1], -centroid[2]]) <= 0.0 {
            continue;
        }
        front_facing_face_count += 1;
        for (point, camera_point) in projected.iter().copied().zip(camera_points) {
            projected_points.push(point);
            depths.push(camera_point[2]);
        }
        rasterize_projected_triangle(
            projected,
            [
                camera_points[0][2],
                camera_points[1][2],
                camera_points[2][2],
            ],
            crop_bbox,
            &mut mask,
            &mut depth_buffer,
        );
    }

    let mut fallback_points = false;
    if mask.iter().all(|value| *value == 0) {
        fallback_points = true;
        for vertex in &mesh.vertices {
            let world = rotation_fit_transform_local_point(placement, *vertex);
            let Some((camera_point, projected_point)) = projection.project(world, evidence) else {
                continue;
            };
            projected_points.push(projected_point);
            depths.push(camera_point[2]);
            splat_projected_point(
                projected_point,
                camera_point[2],
                crop_bbox,
                &mut mask,
                &mut depth_buffer,
            );
        }
    }
    let covered_px = mask.iter().filter(|value| **value != 0).count();
    if covered_px == 0 {
        return None;
    }
    let bbox = normalized_points_bbox(&projected_points);
    let mut covered_depths = depth_buffer
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if covered_depths.is_empty() {
        covered_depths = depths;
    }
    let depth_summary =
        SurfaceDepthSummary::from_depths_with_mask(&covered_depths, &mask, &depth_buffer);
    let median_depth_m = depth_summary.map(|summary| summary.median_m);
    let mask_moments = mask_point_moments(&mask);
    Some(ProjectedCandidateSurface {
        mask,
        depth_buffer,
        bbox,
        median_depth_m,
        depth_summary,
        mask_moments,
        front_facing_face_count,
        covered_px,
        fallback_points,
    })
}

#[derive(Clone, Copy)]
enum RotationFitProjection {
    Source {
        origin: [f32; 2],
        anchor: [f32; 3],
        ground_anchor_basis: &'static str,
        intrinsics: RotationFitIntrinsics,
    },
    Layout {
        translation: [f32; 3],
        forward: [f32; 3],
        right: [f32; 3],
        up: [f32; 3],
        vertical_fov_degrees: f32,
        aspect: f32,
    },
}

impl RotationFitProjection {
    fn from_fit_object(
        fit_object: &crate::ProjectionFitObjectReport,
        projection_camera: Option<&crate::ProjectionFitCameraReport>,
        intrinsics: RotationFitIntrinsics,
    ) -> Option<Self> {
        if let (Some(origin), Some(anchor)) = (
            fit_object.source_camera_origin_xz,
            fit_object.source_camera_anchor,
        ) {
            let ground_anchor_basis = if fit_object.ground_anchor_basis == "camera-ray-ground-plane"
            {
                "camera-ray-ground-plane"
            } else {
                "layout-contact-pixel"
            };
            return Some(Self::Source {
                origin,
                anchor,
                ground_anchor_basis,
                intrinsics,
            });
        }
        let camera = projection_camera?;
        if camera.basis != "layout-camera" {
            return None;
        }
        let forward = normalize3(sub3(camera.focus, camera.translation))?;
        let right = normalize3(cross3(forward, [0.0, 1.0, 0.0]))
            .or_else(|| normalize3(cross3(forward, [0.0, 0.0, 1.0])))?;
        let up = normalize3(cross3(right, forward))?;
        Some(Self::Layout {
            translation: camera.translation,
            forward,
            right,
            up,
            vertical_fov_degrees: camera.vertical_fov_degrees,
            aspect: camera.aspect.max(0.1),
        })
    }

    fn project(
        self,
        world: [f32; 3],
        evidence: &SceneGroundingEvidence,
    ) -> Option<([f32; 3], [f32; 2])> {
        match self {
            Self::Source {
                origin,
                anchor,
                ground_anchor_basis,
                intrinsics,
            } => {
                let camera_point = rotation_fit_source_camera_point(
                    world,
                    origin,
                    anchor,
                    ground_anchor_basis,
                    evidence,
                )?;
                let projected_point =
                    rotation_fit_project_source_camera_point(camera_point, intrinsics)?;
                Some((camera_point, projected_point))
            }
            Self::Layout {
                translation,
                forward,
                right,
                up,
                vertical_fov_degrees,
                aspect,
            } => {
                let rel = sub3(world, translation);
                let z = dot3(rel, forward);
                if !z.is_finite() || z <= 1.0e-4 {
                    return None;
                }
                let x = dot3(rel, right);
                let y = dot3(rel, up);
                let tan_half = (vertical_fov_degrees.to_radians() * 0.5).tan();
                if !tan_half.is_finite() || tan_half <= 1.0e-5 {
                    return None;
                }
                let u = (x / (z * tan_half * aspect) + 1.0) * 0.5;
                let v = (1.0 - y / (z * tan_half)) * 0.5;
                (u.is_finite() && v.is_finite()).then_some(([x, y, z], [u, v]))
            }
        }
    }
}

fn rotation_fit_transform_local_point(
    placement: &GroundedScenePlacement,
    local: [f32; 3],
) -> [f32; 3] {
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

pub(super) fn rotation_fit_source_camera_point(
    point: [f32; 3],
    origin: [f32; 2],
    anchor: [f32; 3],
    ground_anchor_basis: &str,
    evidence: &SceneGroundingEvidence,
) -> Option<[f32; 3]> {
    let x = point[0] + origin[0];
    let z = origin[1] - point[2];
    let height_above_floor = point[1];
    let floor_y_camera = if ground_anchor_basis == "camera-ray-ground-plane" {
        rotation_fit_source_floor_y_at(evidence, x, z).unwrap_or(anchor[1])
    } else {
        anchor[1]
    };
    Some([x, floor_y_camera - height_above_floor, z])
}

pub(super) fn rotation_fit_source_floor_y_at(
    evidence: &SceneGroundingEvidence,
    x: f32,
    z: f32,
) -> Option<f32> {
    let floor = evidence.floor;
    let normal_len_sq = floor.normal.iter().map(|value| value * value).sum::<f32>();
    let residual_ok = floor
        .residual_m
        .filter(|value| value.is_finite())
        .is_none_or(|value| value <= 0.18);
    if !normal_len_sq.is_finite()
        || normal_len_sq <= 0.25
        || floor.normal[1].abs() <= 1.0e-5
        || !floor.distance_m.is_finite()
        || !residual_ok
    {
        return None;
    }
    let y = -(floor.normal[0] * x + floor.normal[2] * z + floor.distance_m) / floor.normal[1];
    y.is_finite().then_some(y)
}

pub(super) fn rotation_fit_project_source_camera_point(
    point: [f32; 3],
    intrinsics: RotationFitIntrinsics,
) -> Option<[f32; 2]> {
    let z = point[2];
    if !z.is_finite() || z <= 1.0e-4 {
        return None;
    }
    let u = (intrinsics.fx * point[0] / z + intrinsics.cx) / (intrinsics.width - 1.0).max(1.0);
    let v = (intrinsics.fy * point[1] / z + intrinsics.cy) / (intrinsics.height - 1.0).max(1.0);
    (u.is_finite() && v.is_finite()).then_some([u, v])
}

pub(super) fn rasterize_projected_triangle(
    points: [[f32; 2]; 3],
    depths: [f32; 3],
    crop_bbox: [f32; 4],
    mask: &mut [u8],
    depth_buffer: &mut [f32],
) {
    let res = ROTATION_FIT_CROP_RESOLUTION as i32;
    let crop_w = (crop_bbox[2] - crop_bbox[0]).abs().max(1.0e-5);
    let crop_h = (crop_bbox[3] - crop_bbox[1]).abs().max(1.0e-5);
    let to_px = |point: [f32; 2]| -> [f32; 2] {
        [
            (point[0] - crop_bbox[0]) / crop_w * (res - 1) as f32,
            (point[1] - crop_bbox[1]) / crop_h * (res - 1) as f32,
        ]
    };
    let p = [to_px(points[0]), to_px(points[1]), to_px(points[2])];
    let min_x = p
        .iter()
        .map(|point| point[0].floor() as i32)
        .min()
        .unwrap_or(0)
        .clamp(0, res - 1);
    let max_x = p
        .iter()
        .map(|point| point[0].ceil() as i32)
        .max()
        .unwrap_or(0)
        .clamp(0, res - 1);
    let min_y = p
        .iter()
        .map(|point| point[1].floor() as i32)
        .min()
        .unwrap_or(0)
        .clamp(0, res - 1);
    let max_y = p
        .iter()
        .map(|point| point[1].ceil() as i32)
        .max()
        .unwrap_or(0)
        .clamp(0, res - 1);
    let area = edge2(p[0], p[1], p[2]);
    if !area.is_finite() || area.abs() <= 1.0e-5 {
        return;
    }
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let point = [x as f32 + 0.5, y as f32 + 0.5];
            let w0 = edge2(p[1], p[2], point) / area;
            let w1 = edge2(p[2], p[0], point) / area;
            let w2 = edge2(p[0], p[1], point) / area;
            if w0 < -1.0e-4 || w1 < -1.0e-4 || w2 < -1.0e-4 {
                continue;
            }
            let depth = w0 * depths[0] + w1 * depths[1] + w2 * depths[2];
            if !depth.is_finite() {
                continue;
            }
            let index = y as usize * res as usize + x as usize;
            if depth < depth_buffer[index] {
                depth_buffer[index] = depth;
                mask[index] = 1;
            }
        }
    }
}

fn splat_projected_point(
    point: [f32; 2],
    depth: f32,
    crop_bbox: [f32; 4],
    mask: &mut [u8],
    depth_buffer: &mut [f32],
) {
    let res = ROTATION_FIT_CROP_RESOLUTION as i32;
    let crop_w = (crop_bbox[2] - crop_bbox[0]).abs().max(1.0e-5);
    let crop_h = (crop_bbox[3] - crop_bbox[1]).abs().max(1.0e-5);
    let x = ((point[0] - crop_bbox[0]) / crop_w * (res - 1) as f32).round() as i32;
    let y = ((point[1] - crop_bbox[1]) / crop_h * (res - 1) as f32).round() as i32;
    for dy in -1..=1 {
        for dx in -1..=1 {
            let px = (x + dx).clamp(0, res - 1);
            let py = (y + dy).clamp(0, res - 1);
            let index = py as usize * res as usize + px as usize;
            if depth < depth_buffer[index] {
                depth_buffer[index] = depth;
                mask[index] = 1;
            }
        }
    }
}

pub(super) fn write_rotation_candidate_mask_png(
    path: &Path,
    predicted: &[u8],
    target: &[u8],
) -> Result<(), String> {
    let res = ROTATION_FIT_CROP_RESOLUTION;
    let mut image = image::RgbaImage::new(res, res);
    for y in 0..res {
        for x in 0..res {
            let index = y as usize * res as usize + x as usize;
            let pred = predicted.get(index).copied().unwrap_or(0) != 0;
            let truth = target.get(index).copied().unwrap_or(0) != 0;
            let color = match (truth, pred) {
                (true, true) => image::Rgba([245, 245, 245, 255]),
                (true, false) => image::Rgba([34, 210, 91, 220]),
                (false, true) => image::Rgba([62, 145, 255, 220]),
                (false, false) => image::Rgba([12, 14, 18, 255]),
            };
            image.put_pixel(x, y, color);
        }
    }
    ensure_parent_dir(path).map_err(|err| err.to_string())?;
    image.save(path).map_err(|err| {
        format!(
            "failed to save rotation candidate {}: {err}",
            path.display()
        )
    })
}

pub(super) fn binary_mask_iou(left: &[u8], right: &[u8]) -> f32 {
    let len = left.len().min(right.len());
    if len == 0 {
        return 0.0;
    }
    let mut intersection = 0usize;
    let mut union = 0usize;
    for index in 0..len {
        let l = left[index] != 0;
        let r = right[index] != 0;
        if l && r {
            intersection += 1;
        }
        if l || r {
            union += 1;
        }
    }
    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

impl SurfaceDepthSummary {
    pub(super) fn from_object_stats(stats: crate::ObjectDepthStats) -> Self {
        let median = stats.median_m.max(1.0e-4);
        let min = stats.min_m.min(median).max(1.0e-4);
        let max = stats.max_m.max(median).max(min + 1.0e-4);
        Self {
            min_m: min,
            p10_m: min,
            median_m: median,
            p90_m: max,
            max_m: max,
            contact_m: stats
                .contact_m
                .filter(|value| value.is_finite() && *value > 0.0),
            sample_count: stats.sample_count.unwrap_or(0),
        }
    }

    pub(super) fn from_depths_with_mask(
        depths: &[f32],
        mask: &[u8],
        depth_buffer: &[f32],
    ) -> Option<Self> {
        let mut sorted = depths
            .iter()
            .copied()
            .filter(|value| value.is_finite() && *value > 0.0)
            .collect::<Vec<_>>();
        if sorted.is_empty() {
            return None;
        }
        sorted.sort_by(f32::total_cmp);
        let sample_count = sorted.len();
        Some(Self {
            min_m: sorted[0],
            p10_m: sorted_quantile(&sorted, 0.10),
            median_m: sorted_quantile(&sorted, 0.50),
            p90_m: sorted_quantile(&sorted, 0.90),
            max_m: *sorted.last().unwrap_or(&sorted[0]),
            contact_m: surface_contact_depth_m(mask, depth_buffer),
            sample_count,
        })
    }
}

fn sorted_quantile(sorted: &[f32], q: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let q = q.clamp(0.0, 1.0);
    let index = ((sorted.len().saturating_sub(1)) as f32 * q).round() as usize;
    sorted[index.min(sorted.len() - 1)]
}

fn surface_contact_depth_m(mask: &[u8], depth_buffer: &[f32]) -> Option<f32> {
    let res = square_mask_resolution(mask)?;
    let mut bottom_row = None;
    for y in (0..res).rev() {
        let row = y * res;
        if (0..res).any(|x| mask[row + x] != 0) {
            bottom_row = Some(y);
            break;
        }
    }
    let bottom = bottom_row?;
    let start = bottom.saturating_sub((res / 8).max(1));
    let mut depths = Vec::new();
    for y in start..=bottom {
        let row = y * res;
        for x in 0..res {
            let index = row + x;
            if mask[index] == 0 {
                continue;
            }
            let depth = depth_buffer.get(index).copied().unwrap_or(f32::INFINITY);
            if depth.is_finite() && depth > 0.0 {
                depths.push(depth);
            }
        }
    }
    median_f32(&mut depths)
}

pub(super) fn surface_depth_loss(
    target: SurfaceDepthSummary,
    observed: SurfaceDepthSummary,
    max_depth_error_m: f32,
) -> f32 {
    let scale = max_depth_error_m.max(0.05);
    let median = ((observed.median_m - target.median_m).abs() / scale).clamp(0.0, 4.0);
    let near = ((observed.p10_m - target.p10_m).abs() / (scale * 1.5)).clamp(0.0, 4.0);
    let far = ((observed.p90_m - target.p90_m).abs() / (scale * 2.0)).clamp(0.0, 4.0);
    let target_span = (target.p90_m - target.p10_m).abs().max(0.05);
    let observed_span = (observed.p90_m - observed.p10_m).abs().max(0.05);
    let span = safe_log2_ratio(observed_span, target_span).abs().min(3.0);
    let contact = match (target.contact_m, observed.contact_m) {
        (Some(target), Some(observed)) => ((observed - target).abs() / scale).clamp(0.0, 4.0),
        _ => median,
    };
    median * 0.44 + contact * 0.24 + near * 0.12 + far * 0.08 + span * 0.12
}

pub(super) fn surface_depth_summary_report(summary: SurfaceDepthSummary) -> Value {
    json!({
        "min_m": summary.min_m,
        "p10_m": summary.p10_m,
        "median_m": summary.median_m,
        "p90_m": summary.p90_m,
        "max_m": summary.max_m,
        "contact_m": summary.contact_m,
        "sample_count": summary.sample_count,
    })
}

pub(super) fn mask_point_moments(mask: &[u8]) -> MaskPointMoments {
    let Some(res) = square_mask_resolution(mask) else {
        return MaskPointMoments::default();
    };
    let mut count = 0.0_f32;
    let mut sum = [0.0_f32; 2];
    for y in 0..res {
        for x in 0..res {
            if mask[y * res + x] == 0 {
                continue;
            }
            let point = [(x as f32 + 0.5) / res as f32, (y as f32 + 0.5) / res as f32];
            count += 1.0;
            sum[0] += point[0];
            sum[1] += point[1];
        }
    }
    if count <= 0.0 {
        return MaskPointMoments::default();
    }
    let centroid = [sum[0] / count, sum[1] / count];
    let mut variance = [0.0_f32; 2];
    for y in 0..res {
        for x in 0..res {
            if mask[y * res + x] == 0 {
                continue;
            }
            let point = [(x as f32 + 0.5) / res as f32, (y as f32 + 0.5) / res as f32];
            variance[0] += (point[0] - centroid[0]).powi(2);
            variance[1] += (point[1] - centroid[1]).powi(2);
        }
    }
    variance[0] /= count;
    variance[1] /= count;
    MaskPointMoments {
        centroid: Some(centroid),
        variance: Some(variance),
        coverage: count / mask.len().max(1) as f32,
    }
}

pub(super) fn mask_point_surface_loss(target: MaskPointMoments, observed: MaskPointMoments) -> f32 {
    let centroid = match (target.centroid, observed.centroid) {
        (Some(target), Some(observed)) => distance2(target, observed).sqrt().min(1.0),
        _ => 0.35,
    };
    let variance = match (target.variance, observed.variance) {
        (Some(target), Some(observed)) => {
            let x = safe_log2_ratio(observed[0].max(1.0e-5).sqrt(), target[0].max(1.0e-5).sqrt())
                .abs()
                .min(3.0);
            let y = safe_log2_ratio(observed[1].max(1.0e-5).sqrt(), target[1].max(1.0e-5).sqrt())
                .abs()
                .min(3.0);
            (x + y) * 0.5
        }
        _ => 0.50,
    };
    let coverage = safe_log2_ratio(observed.coverage.max(1.0e-5), target.coverage.max(1.0e-5))
        .abs()
        .min(3.0);
    centroid * 0.52 + variance * 0.30 + coverage * 0.18
}

pub(super) fn square_mask_resolution(mask: &[u8]) -> Option<usize> {
    if mask.is_empty() {
        return None;
    }
    let res = (mask.len() as f64).sqrt().round() as usize;
    (res > 0 && res * res == mask.len()).then_some(res)
}

fn normalized_points_bbox(points: &[[f32; 2]]) -> Option<[f32; 4]> {
    let mut min = [f32::INFINITY, f32::INFINITY];
    let mut max = [f32::NEG_INFINITY, f32::NEG_INFINITY];
    for point in points {
        min[0] = min[0].min(point[0]);
        min[1] = min[1].min(point[1]);
        max[0] = max[0].max(point[0]);
        max[1] = max[1].max(point[1]);
    }
    if min[0].is_finite() && min[1].is_finite() && max[0].is_finite() && max[1].is_finite() {
        Some([
            min[0].clamp(0.0, 1.0),
            min[1].clamp(0.0, 1.0),
            max[0].clamp(0.0, 1.0),
            max[1].clamp(0.0, 1.0),
        ])
    } else {
        None
    }
}

fn median_f32(values: &mut Vec<f32>) -> Option<f32> {
    values.retain(|value| value.is_finite());
    if values.is_empty() {
        return None;
    }
    values.sort_by(|left, right| left.total_cmp(right));
    Some(values[values.len() / 2])
}

fn edge2(a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> f32 {
    (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])
}

fn sub3(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn cross3(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn dot3(left: [f32; 3], right: [f32; 3]) -> f32 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

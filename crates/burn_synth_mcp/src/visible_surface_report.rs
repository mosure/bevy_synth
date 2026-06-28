use crate::prelude::*;

pub(crate) fn write_visible_surface_fit_artifacts(
    output_dir: &Path,
    response: &Value,
    projection_fit: &Value,
) -> Result<(), String> {
    let Some(layout) = response.get("grounded_layout") else {
        return Ok(());
    };
    let Ok(placements) = serde_json::from_value::<Vec<GroundedScenePlacement>>(
        layout
            .get("placements")
            .cloned()
            .unwrap_or_else(|| Value::Array(Vec::new())),
    ) else {
        return Ok(());
    };
    let Ok(asset_bindings) = serde_json::from_value::<Vec<SceneAssetBinding>>(
        response
            .get("asset_bindings")
            .cloned()
            .unwrap_or_else(|| Value::Array(Vec::new())),
    ) else {
        return Ok(());
    };
    let Ok(evidence) = serde_json::from_value::<SceneGroundingEvidence>(
        response
            .get("grounding_evidence")
            .cloned()
            .unwrap_or_else(|| Value::Null),
    ) else {
        return Ok(());
    };
    let fit_objects = projection_fit
        .get("objects")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let source_path = visible_surface_source_path(response);
    let intrinsics = visible_source_intrinsics(&evidence);
    let mut objects = Vec::new();
    let mut projected_mesh_count = 0usize;
    let mut warning_count = 0usize;

    for placement in &placements {
        let fit_object = fit_objects
            .iter()
            .find(|object| fit_object_matches_placement(object, placement));
        let binding = asset_bindings
            .iter()
            .find(|binding| binding.asset_id == placement.asset_id)
            .or_else(|| {
                asset_bindings
                    .iter()
                    .find(|binding| binding.object_id == placement.object_id)
            });
        let source_object = best_grounding_object_for_placement(placement, &evidence);
        let target_bbox = fit_object
            .and_then(|object| {
                object
                    .pointer("/visible_surface/source_mask_bbox")
                    .and_then(json_array4)
            })
            .or_else(|| source_object.and_then(|object| object.mask.as_ref().map(|mask| mask.bbox)))
            .or_else(|| {
                fit_object.and_then(|object| object.get("source_bbox").and_then(json_array4))
            })
            .unwrap_or(placement.source_bbox);
        let source_depth_median_m = source_object
            .and_then(|object| object.depth_stats)
            .map(|stats| stats.median_m)
            .or_else(|| {
                fit_object.and_then(|object| {
                    object
                        .pointer("/visible_surface/source_depth_median_m")
                        .and_then(Value::as_f64)
                        .map(|value| value as f32)
                })
            });
        let mut object_report = json!({
            "object_id": placement.object_id,
            "instance_id": placement.instance_id,
            "label": placement.label,
            "asset_id": placement.asset_id,
            "source_mask_bbox": target_bbox,
            "source_depth_median_m": source_depth_median_m,
            "projected_aabb_bbox": fit_object
                .and_then(|object| object.get("projected_bbox").and_then(json_array4)),
            "projected_aabb_bbox_iou": fit_object
                .and_then(|object| object.get("bbox_iou").and_then(Value::as_f64)),
            "aabb_depth_log2_error": fit_object
                .and_then(|object| object.get("depth_log2_error").and_then(Value::as_f64)),
            "mesh_surface_projection_available": false,
            "rasterized_visible_surface": false,
        });
        let Some(path) = binding.and_then(|binding| binding.path.as_deref()) else {
            warning_count += 1;
            object_report["warning"] = json!("asset binding has no GLB path");
            objects.push(object_report);
            continue;
        };
        let path = resolve_report_asset_path(path, output_dir);
        let Some(fit_object) = fit_object else {
            warning_count += 1;
            object_report["asset_path"] = json!(path.display().to_string());
            object_report["warning"] = json!("projection fit object report missing");
            objects.push(object_report);
            continue;
        };
        let Some(intrinsics) = intrinsics else {
            warning_count += 1;
            object_report["asset_path"] = json!(path.display().to_string());
            object_report["warning"] = json!("source camera intrinsics unavailable");
            objects.push(object_report);
            continue;
        };
        match project_glb_mesh_visible_surface(&path, placement, fit_object, &evidence, intrinsics)
        {
            Ok(surface) => {
                projected_mesh_count += 1;
                let mesh_iou = normalized_bbox_iou(surface.bbox, target_bbox);
                let mesh_depth_log2_error = source_depth_median_m
                    .map(|depth| safe_log2_ratio_f32(surface.median_depth_m, depth).abs());
                object_report["asset_path"] = json!(path.display().to_string());
                object_report["mesh_surface_projection_available"] = json!(true);
                object_report["mesh_projection_kind"] = json!(surface.projection_kind);
                object_report["mesh_projected_bbox"] = json!(surface.bbox);
                object_report["mesh_bbox_iou"] = json!(mesh_iou);
                object_report["mesh_depth_median_m"] = json!(surface.median_depth_m);
                object_report["mesh_depth_log2_error"] = json!(mesh_depth_log2_error);
                object_report["mesh_vertex_count"] = json!(surface.vertex_count);
                object_report["mesh_face_count"] = json!(surface.face_count);
                object_report["front_facing_face_count"] = json!(surface.front_facing_face_count);
                object_report["backface_culling_fallback"] =
                    json!(surface.backface_culling_fallback);
                object_report["geometry_match_ready"] = json!(source_depth_median_m.is_some());
            }
            Err(err) => {
                warning_count += 1;
                object_report["asset_path"] = json!(path.display().to_string());
                object_report["warning"] = json!(err);
            }
        }
        objects.push(object_report);
    }

    let report = json!({
        "schema_version": 1,
        "stage": "visible_surface_fit",
        "source_scene_path": source_path
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_default(),
        "target": "source_sam_mask_depth_visible_surface",
        "prediction": "projected_glb_mesh_front_face_or_vertex_surface",
        "optimizer_loss_uses_full_mesh_visible_surface": false,
        "rasterized_visible_surface_depth_buffer": false,
        "note": "This report projects the generated GLB mesh after canonical-pose and scene-placement fitting. It is stronger than AABB-only fit evidence, but it is still not a per-pixel renderer/object-ID/depth-buffer loss.",
        "object_count": placements.len(),
        "projected_mesh_count": projected_mesh_count,
        "warning_count": warning_count,
        "objects": objects,
    });
    write_json_file(&output_dir.join("visible_surface_fit_report.json"), &report)
        .map_err(|err| err.to_string())?;

    if let Some(source_path) = source_path
        && source_path.exists()
    {
        let overlay_path = output_dir.join("visible_surface_fit_overlay.png");
        if let Err(err) = write_visible_surface_fit_overlay(&source_path, &report, &overlay_path) {
            let warning = json!({
                "visible_surface_fit_overlay": overlay_path.display().to_string(),
                "warning": err,
            });
            write_json_file(
                &output_dir.join("visible_surface_fit_overlay_error.json"),
                &warning,
            )
            .map_err(|err| err.to_string())?;
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct VisibleSourceIntrinsics {
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    width: f32,
    height: f32,
}

#[derive(Clone)]
struct MeshSurfaceProjection {
    bbox: [f32; 4],
    median_depth_m: f32,
    vertex_count: usize,
    face_count: usize,
    front_facing_face_count: usize,
    backface_culling_fallback: bool,
    projection_kind: &'static str,
}

fn visible_source_intrinsics(evidence: &SceneGroundingEvidence) -> Option<VisibleSourceIntrinsics> {
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
    Some(VisibleSourceIntrinsics {
        fx: fy,
        fy,
        cx: principal[0],
        cy: principal[1],
        width,
        height,
    })
}

fn project_glb_mesh_visible_surface(
    path: &Path,
    placement: &GroundedScenePlacement,
    fit_object: &Value,
    evidence: &SceneGroundingEvidence,
    intrinsics: VisibleSourceIntrinsics,
) -> Result<MeshSurfaceProjection, String> {
    let bytes = fs::read(path)
        .map_err(|err| format!("failed to read GLB mesh {}: {err}", path.display()))?;
    let mesh = bevy_synth_runtime::io::mesh_from_glb_bytes(&bytes)
        .map_err(|err| format!("failed to parse GLB mesh {}: {err}", path.display()))?;
    if mesh.mesh.vertices.is_empty() {
        return Err("GLB mesh has no vertices".to_string());
    }
    let origin = fit_object
        .get("source_camera_origin_xz")
        .and_then(json_array2)
        .ok_or_else(|| {
            "projection fit report missing source_camera_origin_xz; rerun with current scene fit"
                .to_string()
        })?;
    let anchor = fit_object
        .get("source_camera_anchor")
        .and_then(json_array3)
        .ok_or_else(|| {
            "projection fit report missing source_camera_anchor; rerun with current scene fit"
                .to_string()
        })?;
    let ground_anchor_basis = fit_object
        .get("ground_anchor_basis")
        .and_then(Value::as_str)
        .unwrap_or_default();

    let mut projected = Vec::new();
    let mut depths = Vec::new();
    let mut front_facing_face_count = 0usize;
    for face in &mesh.mesh.faces {
        let indices = [face[0] as usize, face[1] as usize, face[2] as usize];
        if indices
            .iter()
            .any(|index| *index >= mesh.mesh.vertices.len())
        {
            continue;
        }
        let mut camera_points = [[0.0; 3]; 3];
        let mut projected_points = [[0.0; 2]; 3];
        let mut valid = true;
        for (slot, index) in indices.iter().copied().enumerate() {
            let world = transform_report_local_point(placement, mesh.mesh.vertices[index]);
            let Some(camera_point) =
                report_source_camera_point(world, origin, anchor, ground_anchor_basis, evidence)
            else {
                valid = false;
                break;
            };
            let Some(projected_point) = project_source_camera_point(camera_point, intrinsics)
            else {
                valid = false;
                break;
            };
            camera_points[slot] = camera_point;
            projected_points[slot] = projected_point;
        }
        if !valid {
            continue;
        }
        let normal = cross3_report(
            sub3_report(camera_points[1], camera_points[0]),
            sub3_report(camera_points[2], camera_points[0]),
        );
        let centroid = [
            (camera_points[0][0] + camera_points[1][0] + camera_points[2][0]) / 3.0,
            (camera_points[0][1] + camera_points[1][1] + camera_points[2][1]) / 3.0,
            (camera_points[0][2] + camera_points[1][2] + camera_points[2][2]) / 3.0,
        ];
        let front_facing = dot3_report(normal, [-centroid[0], -centroid[1], -centroid[2]]) > 0.0;
        if front_facing {
            front_facing_face_count += 1;
            for (point, camera_point) in projected_points.into_iter().zip(camera_points) {
                projected.push(point);
                depths.push(camera_point[2]);
            }
        }
    }

    let mut backface_culling_fallback = false;
    if projected.len() < 3 {
        backface_culling_fallback = true;
        projected.clear();
        depths.clear();
        for vertex in &mesh.mesh.vertices {
            let world = transform_report_local_point(placement, *vertex);
            let Some(camera_point) =
                report_source_camera_point(world, origin, anchor, ground_anchor_basis, evidence)
            else {
                continue;
            };
            let Some(projected_point) = project_source_camera_point(camera_point, intrinsics)
            else {
                continue;
            };
            projected.push(projected_point);
            depths.push(camera_point[2]);
        }
    }
    if projected.len() < 2 || depths.is_empty() {
        return Err("mesh projection produced too few visible points".to_string());
    }
    let bbox = normalized_points_bbox(&projected)
        .ok_or_else(|| "mesh projection bbox is invalid".to_string())?;
    let median_depth_m = median_f32(&mut depths)
        .ok_or_else(|| "mesh projection depth median is invalid".to_string())?;
    Ok(MeshSurfaceProjection {
        bbox,
        median_depth_m,
        vertex_count: mesh.mesh.vertices.len(),
        face_count: mesh.mesh.faces.len(),
        front_facing_face_count,
        backface_culling_fallback,
        projection_kind: if backface_culling_fallback {
            "all_projected_vertices"
        } else {
            "front_facing_face_vertices"
        },
    })
}

fn fit_object_matches_placement(object: &Value, placement: &GroundedScenePlacement) -> bool {
    object
        .get("object_id")
        .and_then(Value::as_str)
        .is_some_and(|id| id == placement.object_id)
        && object.get("instance_id").and_then(Value::as_str) == placement.instance_id.as_deref()
}

fn best_grounding_object_for_placement<'a>(
    placement: &GroundedScenePlacement,
    evidence: &'a SceneGroundingEvidence,
) -> Option<&'a burn_synth_scene::ObjectGroundingEvidence> {
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

fn resolve_report_asset_path(path: &str, output_dir: &Path) -> PathBuf {
    let path = PathBuf::from(path);
    if path.exists() || path.is_absolute() {
        path
    } else {
        output_dir.join(path)
    }
}

fn report_source_camera_point(
    point: [f32; 3],
    origin: [f32; 2],
    anchor: [f32; 3],
    ground_anchor_basis: &str,
    evidence: &SceneGroundingEvidence,
) -> Option<[f32; 3]> {
    let x = point[0] + origin[0];
    let z = point[2] + origin[1];
    let height_above_floor = point[1];
    let floor_y_camera = if ground_anchor_basis == "camera-ray-ground-plane" {
        report_source_floor_y_at(evidence, x, z).unwrap_or(anchor[1])
    } else {
        anchor[1]
    };
    Some([x, floor_y_camera - height_above_floor, z])
}

fn report_source_floor_y_at(evidence: &SceneGroundingEvidence, x: f32, z: f32) -> Option<f32> {
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

fn project_source_camera_point(
    point: [f32; 3],
    intrinsics: VisibleSourceIntrinsics,
) -> Option<[f32; 2]> {
    let z = point[2];
    if !z.is_finite() || z <= 1.0e-4 {
        return None;
    }
    let u = (intrinsics.fx * point[0] / z + intrinsics.cx) / (intrinsics.width - 1.0).max(1.0);
    let v = (intrinsics.fy * point[1] / z + intrinsics.cy) / (intrinsics.height - 1.0).max(1.0);
    (u.is_finite() && v.is_finite()).then_some([u, v])
}

fn transform_report_local_point(placement: &GroundedScenePlacement, local: [f32; 3]) -> [f32; 3] {
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

fn safe_log2_ratio_f32(observed: f32, expected: f32) -> f32 {
    (observed.max(1.0e-8) / expected.max(1.0e-8)).log2()
}

fn json_array2(value: &Value) -> Option<[f32; 2]> {
    let values = value.as_array()?;
    if values.len() != 2 {
        return None;
    }
    Some([values[0].as_f64()? as f32, values[1].as_f64()? as f32])
}

fn json_array3(value: &Value) -> Option<[f32; 3]> {
    let values = value.as_array()?;
    if values.len() != 3 {
        return None;
    }
    Some([
        values[0].as_f64()? as f32,
        values[1].as_f64()? as f32,
        values[2].as_f64()? as f32,
    ])
}

fn sub3_report(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn cross3_report(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn dot3_report(left: [f32; 3], right: [f32; 3]) -> f32 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn write_visible_surface_fit_overlay(
    source_path: &Path,
    report: &Value,
    output_path: &Path,
) -> Result<(), String> {
    let image = image::open(source_path)
        .map_err(|err| {
            format!(
                "failed to open visible-surface source {}: {err}",
                source_path.display()
            )
        })?
        .resize(1800, 1800, image::imageops::FilterType::Triangle)
        .to_rgba8();
    let mut image = image;
    let objects = report
        .get("objects")
        .and_then(Value::as_array)
        .ok_or_else(|| "visible surface report missing objects array".to_string())?;
    for object in objects {
        if let Some(source_bbox) = object.get("source_mask_bbox").and_then(json_array4) {
            draw_normalized_rect(&mut image, source_bbox, image::Rgba([36, 220, 74, 255]));
        }
        if let Some(projected_bbox) = object.get("projected_aabb_bbox").and_then(json_array4) {
            draw_normalized_rect(&mut image, projected_bbox, image::Rgba([255, 76, 216, 255]));
        }
        if let Some(mesh_bbox) = object.get("mesh_projected_bbox").and_then(json_array4) {
            draw_normalized_rect(&mut image, mesh_bbox, image::Rgba([68, 158, 255, 255]));
        }
    }
    image.save(output_path).map_err(|err| {
        format!(
            "failed to write visible-surface overlay {}: {err}",
            output_path.display()
        )
    })
}

fn visible_surface_source_path(response: &Value) -> Option<PathBuf> {
    response
        .get("source_scene_path")
        .and_then(Value::as_str)
        .or_else(|| {
            response
                .get("grounding_evidence")
                .and_then(|evidence| evidence.get("source_image_path"))
                .and_then(Value::as_str)
        })
        .or_else(|| {
            response
                .get("manifest")
                .and_then(|manifest| manifest.get("source_scene_path"))
                .and_then(Value::as_str)
        })
        .map(PathBuf::from)
}

fn draw_normalized_rect(image: &mut image::RgbaImage, bbox: [f32; 4], color: image::Rgba<u8>) {
    let width = image.width().saturating_sub(1).max(1);
    let height = image.height().saturating_sub(1).max(1);
    let x0 = (bbox[0].clamp(0.0, 1.0) * width as f32).round() as u32;
    let y0 = (bbox[1].clamp(0.0, 1.0) * height as f32).round() as u32;
    let x1 = (bbox[2].clamp(0.0, 1.0) * width as f32).round() as u32;
    let y1 = (bbox[3].clamp(0.0, 1.0) * height as f32).round() as u32;
    for thickness in 0..3 {
        let x0 = x0.saturating_sub(thickness);
        let y0 = y0.saturating_sub(thickness);
        let x1 = (x1 + thickness).min(width);
        let y1 = (y1 + thickness).min(height);
        for x in x0..=x1 {
            image.put_pixel(x, y0, color);
            image.put_pixel(x, y1, color);
        }
        for y in y0..=y1 {
            image.put_pixel(x0, y, color);
            image.put_pixel(x1, y, color);
        }
    }
}

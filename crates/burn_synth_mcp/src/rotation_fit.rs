use std::fmt::Write as _;

use crate::prelude::*;

const ROTATION_FIT_CROP_RESOLUTION: u32 = 128;
const ROTATION_FIT_MIN_APPLY_IMPROVEMENT: f32 = 0.04;

#[derive(Clone, Copy, Debug)]
pub(crate) struct SceneRotationFitConfig<'a> {
    pub(crate) mode: SceneRotationFitMode,
    pub(crate) max_gpt_rounds: usize,
    pub(crate) min_mask_iou: f32,
    pub(crate) max_depth_error_m: f32,
    pub(crate) write_artifacts: bool,
    pub(crate) output_dir: &'a Path,
}

#[derive(Clone, Debug)]
pub(crate) struct SceneRotationFitOutcome {
    pub(crate) commands: Vec<Value>,
    pub(crate) grounded_layout: GroundedSceneLayout,
    pub(crate) report: Value,
}

#[derive(Clone)]
struct RotationFitTarget {
    mask: BinaryMask,
    bbox: [f32; 4],
    crop_bbox: [f32; 4],
    depth_median_m: Option<f32>,
    mask_kind: &'static str,
}

#[derive(Clone, Copy)]
struct RotationFitIntrinsics {
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    width: f32,
    height: f32,
}

#[derive(Clone)]
struct ProjectedCandidateSurface {
    mask: Vec<u8>,
    bbox: Option<[f32; 4]>,
    median_depth_m: Option<f32>,
    front_facing_face_count: usize,
    covered_px: usize,
    fallback_points: bool,
}

#[derive(Clone)]
struct RotationFitCandidate {
    candidate_index: usize,
    stage: &'static str,
    yaw_degrees: f32,
    yaw_delta_degrees: f32,
    mask_iou: f32,
    bbox_iou: f32,
    depth_error_m: Option<f32>,
    depth_passed: bool,
    loss: f32,
    passed: bool,
    projected_bbox: Option<[f32; 4]>,
    front_facing_face_count: usize,
    covered_px: usize,
    fallback_points: bool,
    artifact_path: Option<PathBuf>,
}

pub(crate) fn apply_scene_rotation_fit(
    config: SceneRotationFitConfig<'_>,
    manifest: &SceneObjectManifest,
    asset_bindings: &[SceneAssetBinding],
    evidence: Option<&SceneGroundingEvidence>,
    grounded_layout: &GroundedSceneLayout,
    commands: &[Value],
) -> Result<SceneRotationFitOutcome, String> {
    let mut out_commands = commands.to_vec();
    let mut out_layout = grounded_layout.clone();
    let mut object_reports = Vec::new();
    let mut applied_count = 0usize;
    let mut skipped_count = 0usize;
    let mut warning_count = 0usize;

    if config.mode == SceneRotationFitMode::Off {
        let report = rotation_fit_report(
            config.mode,
            config,
            manifest,
            0,
            0,
            0,
            Vec::new(),
            "disabled",
        );
        write_rotation_fit_artifacts_if_requested(config, manifest, &report)?;
        return Ok(SceneRotationFitOutcome {
            commands: out_commands,
            grounded_layout: out_layout,
            report,
        });
    }

    let Some(evidence) = evidence else {
        let report = rotation_fit_report(
            config.mode,
            config,
            manifest,
            0,
            grounded_layout.placements.len(),
            1,
            Vec::new(),
            "missing_grounding_evidence",
        );
        write_rotation_fit_artifacts_if_requested(config, manifest, &report)?;
        return Ok(SceneRotationFitOutcome {
            commands: out_commands,
            grounded_layout: out_layout,
            report,
        });
    };
    let Some(projection_fit) = grounded_layout.projection_fit.as_ref() else {
        let report = rotation_fit_report(
            config.mode,
            config,
            manifest,
            0,
            grounded_layout.placements.len(),
            1,
            Vec::new(),
            "missing_projection_fit_report",
        );
        write_rotation_fit_artifacts_if_requested(config, manifest, &report)?;
        return Ok(SceneRotationFitOutcome {
            commands: out_commands,
            grounded_layout: out_layout,
            report,
        });
    };
    let Some(intrinsics) = rotation_fit_intrinsics(evidence) else {
        let report = rotation_fit_report(
            config.mode,
            config,
            manifest,
            0,
            grounded_layout.placements.len(),
            1,
            Vec::new(),
            "missing_source_camera_intrinsics",
        );
        write_rotation_fit_artifacts_if_requested(config, manifest, &report)?;
        return Ok(SceneRotationFitOutcome {
            commands: out_commands,
            grounded_layout: out_layout,
            report,
        });
    };

    let spawn_command_indices = rotation_fit_spawn_command_indices(&out_commands);
    let candidate_root = config.output_dir.join("rotation_fit_candidates");
    if config.write_artifacts {
        fs::create_dir_all(&candidate_root).map_err(|err| {
            format!(
                "failed to create rotation-fit candidate dir {}: {err}",
                candidate_root.display()
            )
        })?;
    }

    for placement_index in 0..grounded_layout.placements.len() {
        let placement = &grounded_layout.placements[placement_index];
        let mut report = json!({
            "index": placement_index,
            "object_id": placement.object_id,
            "instance_id": placement.instance_id,
            "label": placement.label,
            "asset_id": placement.asset_id,
            "applied": false,
        });
        let Some(command_index) = spawn_command_indices.get(placement_index).copied() else {
            skipped_count += 1;
            report["skip_reason"] = json!("missing_spawn_command");
            object_reports.push(report);
            continue;
        };
        let Some(fit_object) = projection_fit
            .objects
            .iter()
            .find(|object| projection_fit_object_matches_placement(object, placement))
        else {
            skipped_count += 1;
            report["skip_reason"] = json!("missing_projection_fit_object");
            object_reports.push(report);
            continue;
        };
        let Some(target) = rotation_fit_target_for_placement(placement, evidence) else {
            skipped_count += 1;
            report["skip_reason"] = json!("missing_mask_or_bbox_target");
            object_reports.push(report);
            continue;
        };
        let Some(asset_path) = rotation_fit_asset_path(
            placement,
            out_commands.get(command_index),
            asset_bindings,
            config.output_dir,
        ) else {
            skipped_count += 1;
            report["skip_reason"] = json!("missing_asset_path");
            object_reports.push(report);
            continue;
        };
        let mesh = match load_rotation_fit_mesh(&asset_path) {
            Ok(mesh) => mesh,
            Err(err) => {
                warning_count += 1;
                skipped_count += 1;
                report["skip_reason"] = json!("mesh_load_failed");
                report["error"] = json!(err);
                object_reports.push(report);
                continue;
            }
        };
        let current_yaw = out_commands
            .get(command_index)
            .and_then(|command| command.get("rotation"))
            .and_then(json_array4)
            .map(quat_y_degrees)
            .unwrap_or(placement.rotation_y_degrees);
        let command_placement = placement_with_yaw(placement, current_yaw);
        let target_mask = rotation_fit_target_crop_mask(&target);
        let candidates_dir =
            candidate_root.join(rotation_fit_object_dir_name(placement_index, placement));
        if config.write_artifacts {
            fs::create_dir_all(&candidates_dir).map_err(|err| {
                format!(
                    "failed to create rotation-fit object candidate dir {}: {err}",
                    candidates_dir.display()
                )
            })?;
        }
        let candidates = evaluate_rotation_fit_candidates(
            &mesh,
            &command_placement,
            fit_object,
            evidence,
            intrinsics,
            &target,
            &target_mask,
            current_yaw,
            &config,
            &candidates_dir,
        );
        if candidates.is_empty() {
            warning_count += 1;
            skipped_count += 1;
            report["skip_reason"] = json!("no_projected_candidates");
            object_reports.push(report);
            continue;
        }
        let baseline = candidates
            .iter()
            .find(|candidate| candidate.yaw_delta_degrees.abs() <= 0.1)
            .or_else(|| {
                candidates.iter().min_by(|left, right| {
                    left.yaw_delta_degrees
                        .abs()
                        .total_cmp(&right.yaw_delta_degrees.abs())
                })
            });
        let best = candidates
            .iter()
            .min_by(|left, right| left.loss.total_cmp(&right.loss));
        let selected = best.filter(|best| {
            best.passed
                && baseline.is_none_or(|baseline| {
                    best.loss + ROTATION_FIT_MIN_APPLY_IMPROVEMENT < baseline.loss
                        || baseline.yaw_delta_degrees.abs() <= 0.1 && !baseline.passed
                })
        });
        let candidate_reports = candidates
            .iter()
            .map(rotation_fit_candidate_report)
            .collect::<Vec<_>>();
        report["target"] = json!({
            "mask_kind": target.mask_kind,
            "bbox": target.bbox,
            "crop_bbox": target.crop_bbox,
            "depth_median_m": target.depth_median_m,
        });
        report["baseline"] = baseline
            .map(rotation_fit_candidate_report)
            .unwrap_or(Value::Null);
        report["best"] = best
            .map(rotation_fit_candidate_report)
            .unwrap_or(Value::Null);
        report["candidate_count"] = json!(candidate_reports.len());
        report["candidates"] = json!(candidate_reports);
        report["asset_path"] = json!(asset_path.display().to_string());
        if let Some(selected) = selected {
            applied_count += 1;
            let selected_yaw = normalize_degrees(selected.yaw_degrees);
            out_commands[command_index]["rotation"] = json!(quat_from_y_degrees(selected_yaw));
            out_layout.placements[placement_index].rotation_y_degrees = selected_yaw;
            report["applied"] = json!(true);
            report["selected"] = rotation_fit_candidate_report(selected);
            report["yaw_before_degrees"] = json!(current_yaw);
            report["yaw_after_degrees"] = json!(selected_yaw);
            report["yaw_delta_degrees"] = json!(normalize_degrees(selected_yaw - current_yaw));
        } else {
            skipped_count += 1;
            report["skip_reason"] = json!("no_candidate_passed_or_improved_gate");
        }
        object_reports.push(report);
    }

    let status = if applied_count > 0 {
        "applied"
    } else {
        "no_applicable_candidates"
    };
    normalize_reused_command_scales(&mut out_commands);
    let report = rotation_fit_report(
        config.mode,
        config,
        manifest,
        applied_count,
        skipped_count,
        warning_count,
        object_reports,
        status,
    );
    write_rotation_fit_artifacts_if_requested(config, manifest, &report)?;
    Ok(SceneRotationFitOutcome {
        commands: out_commands,
        grounded_layout: out_layout,
        report,
    })
}

#[allow(clippy::too_many_arguments)]
fn rotation_fit_report(
    mode: SceneRotationFitMode,
    config: SceneRotationFitConfig<'_>,
    manifest: &SceneObjectManifest,
    applied_count: usize,
    skipped_count: usize,
    warning_count: usize,
    objects: Vec<Value>,
    status: &str,
) -> Value {
    json!({
        "schema_version": 1,
        "stage": "rotation_fit",
        "status": status,
        "mode": mode,
        "algorithm": "deterministic-coarse-to-fine-visible-surface-yaw-fit",
        "source_scene_path": manifest.source_scene_path,
        "artifact_dir": config.output_dir.display().to_string(),
        "max_gpt_rounds": config.max_gpt_rounds,
        "min_mask_iou": config.min_mask_iou,
        "max_depth_error_m": config.max_depth_error_m,
        "write_artifacts": config.write_artifacts,
        "applied_count": applied_count,
        "skipped_count": skipped_count,
        "warning_count": warning_count,
        "object_count": objects.len(),
        "objects": objects,
        "gpt_refine_note": if mode == SceneRotationFitMode::GptRefine {
            "objective depth/mask fitting runs first; bounded GPT crop/candidate selection may be applied by the feedback selector when explicitly enabled"
        } else {
            "GPT is not used in depth-mask-ransac mode"
        },
    })
}

fn write_rotation_fit_artifacts_if_requested(
    config: SceneRotationFitConfig<'_>,
    manifest: &SceneObjectManifest,
    report: &Value,
) -> Result<(), String> {
    if !config.write_artifacts {
        return Ok(());
    }
    fs::create_dir_all(config.output_dir).map_err(|err| {
        format!(
            "failed to create rotation-fit output dir {}: {err}",
            config.output_dir.display()
        )
    })?;
    write_json_file(&config.output_dir.join("rotation_fit_report.json"), report)
        .map_err(|err| err.to_string())?;
    let overlay_path = config.output_dir.join("rotation_fit_overlay.png");
    if let Err(err) = write_rotation_fit_overlay(manifest, report, &overlay_path) {
        write_json_file(
            &config.output_dir.join("rotation_fit_overlay_error.json"),
            &json!({ "path": overlay_path, "error": err }),
        )
        .map_err(|err| err.to_string())?;
    }
    let html = rotation_fit_review_html(report);
    fs::write(config.output_dir.join("rotation_fit_review.html"), html).map_err(|err| {
        format!(
            "failed to write rotation-fit review html {}: {err}",
            config.output_dir.join("rotation_fit_review.html").display()
        )
    })?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn evaluate_rotation_fit_candidates(
    mesh: &CachedSynthMesh,
    placement: &GroundedScenePlacement,
    fit_object: &burn_synth_scene::ProjectionFitObjectReport,
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
    mesh: &CachedSynthMesh,
    placement: &GroundedScenePlacement,
    fit_object: &burn_synth_scene::ProjectionFitObjectReport,
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

fn project_mesh_visible_surface_mask(
    mesh: &CachedSynthMesh,
    placement: &GroundedScenePlacement,
    fit_object: &burn_synth_scene::ProjectionFitObjectReport,
    evidence: &SceneGroundingEvidence,
    intrinsics: RotationFitIntrinsics,
    crop_bbox: [f32; 4],
) -> Option<ProjectedCandidateSurface> {
    let origin = fit_object.source_camera_origin_xz?;
    let anchor = fit_object.source_camera_anchor?;
    let ground_anchor_basis = fit_object.ground_anchor_basis.as_str();
    let res = ROTATION_FIT_CROP_RESOLUTION as usize;
    let mut mask = vec![0_u8; res * res];
    let mut depth_buffer = vec![f32::INFINITY; res * res];
    let mut projected_points = Vec::new();
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
        let mut projected = [[0.0; 2]; 3];
        let mut valid = true;
        for (slot, index) in indices.iter().copied().enumerate() {
            let world = rotation_fit_transform_local_point(placement, mesh.mesh.vertices[index]);
            let Some(camera_point) = rotation_fit_source_camera_point(
                world,
                origin,
                anchor,
                ground_anchor_basis,
                evidence,
            ) else {
                valid = false;
                break;
            };
            let Some(projected_point) =
                rotation_fit_project_source_camera_point(camera_point, intrinsics)
            else {
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
        for vertex in &mesh.mesh.vertices {
            let world = rotation_fit_transform_local_point(placement, *vertex);
            let Some(camera_point) = rotation_fit_source_camera_point(
                world,
                origin,
                anchor,
                ground_anchor_basis,
                evidence,
            ) else {
                continue;
            };
            let Some(projected_point) =
                rotation_fit_project_source_camera_point(camera_point, intrinsics)
            else {
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
    let median_depth_m = median_f32(&mut covered_depths);
    Some(ProjectedCandidateSurface {
        mask,
        bbox,
        median_depth_m,
        front_facing_face_count,
        covered_px,
        fallback_points,
    })
}

fn rotation_fit_candidate_yaws(
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

fn rotation_fit_candidate_report(candidate: &RotationFitCandidate) -> Value {
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

fn rotation_fit_target_for_placement(
    placement: &GroundedScenePlacement,
    evidence: &SceneGroundingEvidence,
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
    Some(RotationFitTarget {
        mask: binary,
        bbox,
        crop_bbox,
        depth_median_m: object.depth_stats.map(|stats| stats.median_m),
        mask_kind,
    })
}

fn rotation_fit_target_crop_mask(target: &RotationFitTarget) -> Vec<u8> {
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

fn rotation_fit_intrinsics(evidence: &SceneGroundingEvidence) -> Option<RotationFitIntrinsics> {
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

fn rotation_fit_asset_path(
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

fn load_rotation_fit_mesh(path: &Path) -> Result<CachedSynthMesh, String> {
    let bytes =
        fs::read(path).map_err(|err| format!("failed to read GLB {}: {err}", path.display()))?;
    bevy_synth_runtime::io::mesh_from_glb_bytes(&bytes)
        .map_err(|err| format!("failed to parse GLB {}: {err}", path.display()))
}

fn placement_with_yaw(
    placement: &GroundedScenePlacement,
    current_yaw: f32,
) -> GroundedScenePlacement {
    let mut out = placement.clone();
    out.rotation_y_degrees = current_yaw;
    out
}

fn projection_fit_object_matches_placement(
    object: &burn_synth_scene::ProjectionFitObjectReport,
    placement: &GroundedScenePlacement,
) -> bool {
    object.object_id == placement.object_id
        && object.instance_id.as_deref() == placement.instance_id.as_deref()
}

fn rotation_fit_spawn_command_indices(commands: &[Value]) -> Vec<usize> {
    commands
        .iter()
        .enumerate()
        .filter_map(|(index, command)| {
            let command_type = command.get("type").and_then(Value::as_str)?;
            matches!(command_type, "spawn_cached" | "spawn_path").then_some(index)
        })
        .collect()
}

fn rotation_fit_object_dir_name(index: usize, placement: &GroundedScenePlacement) -> String {
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

fn rotation_fit_source_camera_point(
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
        rotation_fit_source_floor_y_at(evidence, x, z).unwrap_or(anchor[1])
    } else {
        anchor[1]
    };
    Some([x, floor_y_camera - height_above_floor, z])
}

fn rotation_fit_source_floor_y_at(
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

fn rotation_fit_project_source_camera_point(
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

fn rasterize_projected_triangle(
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

fn write_rotation_candidate_mask_png(
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

fn write_rotation_fit_overlay(
    manifest: &SceneObjectManifest,
    report: &Value,
    output_path: &Path,
) -> Result<(), String> {
    let source_path = Path::new(&manifest.source_scene_path);
    if !source_path.exists() {
        return Err(format!(
            "source scene image does not exist: {}",
            source_path.display()
        ));
    }
    let mut image = image::open(source_path)
        .map_err(|err| {
            format!(
                "failed to open source scene {}: {err}",
                source_path.display()
            )
        })?
        .resize(1800, 1800, image::imageops::FilterType::Triangle)
        .to_rgba8();
    for object in report
        .get("objects")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        if let Some(bbox) = object.pointer("/target/bbox").and_then(json_array4) {
            draw_rotation_fit_rect(&mut image, bbox, image::Rgba([46, 220, 88, 255]));
        }
        if let Some(bbox) = object
            .pointer("/selected/projected_bbox")
            .and_then(json_array4)
        {
            draw_rotation_fit_rect(&mut image, bbox, image::Rgba([64, 155, 255, 255]));
        } else if let Some(bbox) = object.pointer("/best/projected_bbox").and_then(json_array4) {
            draw_rotation_fit_rect(&mut image, bbox, image::Rgba([255, 196, 44, 255]));
        }
    }
    ensure_parent_dir(output_path).map_err(|err| err.to_string())?;
    image.save(output_path).map_err(|err| {
        format!(
            "failed to write rotation-fit overlay {}: {err}",
            output_path.display()
        )
    })
}

fn rotation_fit_review_html(report: &Value) -> String {
    let mut html = String::new();
    html.push_str(
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>rotation fit</title><style>",
    );
    html.push_str("body{background:#101216;color:#e8eaed;font-family:system-ui,sans-serif;margin:24px}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #2a2d34;padding:8px;text-align:left;vertical-align:top}.ok{color:#75e39a}.bad{color:#ff8b8b}img{width:96px;height:96px;image-rendering:pixelated;border:1px solid #333}.candidates{display:flex;gap:6px;flex-wrap:wrap}.meta{color:#9aa0aa}</style></head><body>");
    let _ = write!(
        html,
        "<h1>Rotation Fit</h1><p class=\"meta\">mode: {:?}, status: {}, applied: {}, skipped: {}</p>",
        report.get("mode").unwrap_or(&Value::Null),
        report
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("unknown"),
        report
            .get("applied_count")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        report
            .get("skipped_count")
            .and_then(Value::as_u64)
            .unwrap_or(0),
    );
    html.push_str("<p class=\"meta\">candidate colors: green=source SAM/bbox target, blue=projected mesh surface, white=overlap.</p>");
    html.push_str("<table><thead><tr><th>object</th><th>decision</th><th>target</th><th>best</th><th>candidates</th></tr></thead><tbody>");
    for object in report
        .get("objects")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let applied = object
            .get("applied")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let class = if applied { "ok" } else { "bad" };
        let decision = object
            .get("skip_reason")
            .and_then(Value::as_str)
            .unwrap_or(if applied { "applied" } else { "not_applied" });
        let object_id = object
            .get("object_id")
            .and_then(Value::as_str)
            .unwrap_or("");
        let label = object.get("label").and_then(Value::as_str).unwrap_or("");
        let best = object.get("best").unwrap_or(&Value::Null);
        let _ = write!(
            html,
            "<tr><td><strong>{}</strong><br><span class=\"meta\">{}</span></td><td class=\"{}\">{}</td><td><pre>{}</pre></td><td><pre>{}</pre></td><td><div class=\"candidates\">",
            html_escape(object_id),
            html_escape(label),
            class,
            html_escape(decision),
            html_escape(
                &serde_json::to_string_pretty(object.get("target").unwrap_or(&Value::Null))
                    .unwrap_or_default()
            ),
            html_escape(&serde_json::to_string_pretty(best).unwrap_or_default()),
        );
        for candidate in object
            .get("candidates")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let Some(path) = candidate.get("artifact_path").and_then(Value::as_str) else {
                continue;
            };
            let title = format!(
                "idx {} yaw {} mask_iou {:.3} depth {:?}",
                candidate
                    .get("candidate_index")
                    .and_then(Value::as_u64)
                    .unwrap_or(0),
                candidate
                    .get("yaw_degrees")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0),
                candidate
                    .get("mask_iou")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0),
                candidate.get("depth_error_m"),
            );
            let _ = write!(
                html,
                "<a href=\"{}\" title=\"{}\"><img src=\"{}\"></a>",
                html_escape(path),
                html_escape(&title),
                html_escape(path)
            );
        }
        html.push_str("</div></td></tr>");
    }
    html.push_str("</tbody></table></body></html>");
    html
}

fn draw_rotation_fit_rect(image: &mut image::RgbaImage, bbox: [f32; 4], color: image::Rgba<u8>) {
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

pub(crate) fn binary_mask_iou(left: &[u8], right: &[u8]) -> f32 {
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

fn html_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_fit_mask_iou_matches_overlap_fraction() {
        let left = [1, 1, 0, 0, 1, 0];
        let right = [1, 0, 1, 0, 1, 0];
        let iou = binary_mask_iou(&left, &right);
        assert!((iou - 0.5).abs() < 1.0e-6);
    }

    #[test]
    fn rotation_fit_candidate_yaws_include_180_flip_and_fine_refinement() {
        let coarse = rotation_fit_candidate_yaws(10.0, None);
        assert!(coarse.iter().any(|(yaw, _)| (*yaw - -170.0).abs() < 0.25));
        assert!(coarse.iter().any(|(yaw, _)| (*yaw - 10.0).abs() < 0.25));
        let fine = rotation_fit_candidate_yaws(10.0, Some(-170.0));
        assert!(fine.iter().any(|(yaw, _)| (*yaw - -175.0).abs() < 0.25));
        assert!(fine.iter().any(|(yaw, _)| (*yaw - -150.0).abs() < 0.25));
    }

    #[test]
    fn rotation_fit_rasterizes_triangle_into_crop_mask() {
        let mut mask = vec![
            0_u8;
            ROTATION_FIT_CROP_RESOLUTION as usize
                * ROTATION_FIT_CROP_RESOLUTION as usize
        ];
        let mut depth = vec![f32::INFINITY; mask.len()];
        rasterize_projected_triangle(
            [[0.2, 0.2], [0.8, 0.2], [0.5, 0.8]],
            [2.0, 2.0, 2.0],
            [0.0, 0.0, 1.0, 1.0],
            &mut mask,
            &mut depth,
        );
        assert!(mask.iter().filter(|value| **value != 0).count() > 1000);
        assert!(depth.iter().any(|value| value.is_finite()));
    }
}

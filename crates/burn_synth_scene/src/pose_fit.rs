use std::fmt::Write as _;

use crate::pose_fit_prelude::*;

mod pose_search;
mod soft_refinement;
mod visible_surface;

use pose_search::{
    apply_pose_fit_candidate_to_command, command_placement_for_pose_fit,
    evaluate_rotation_fit_candidates, evaluate_visible_surface_pose_candidates,
    load_rotation_fit_mesh, normalize_reused_layout_scales, placement_with_yaw,
    projection_fit_object_matches_placement, rotation_fit_asset_path,
    rotation_fit_candidate_report, rotation_fit_intrinsics, rotation_fit_object_dir_name,
    rotation_fit_spawn_command_indices, rotation_fit_target_crop_mask,
    rotation_fit_target_for_placement, sync_commands_from_layout_placements,
    sync_layout_placements_from_commands, visible_surface_pose_candidate_passes_target,
    visible_surface_pose_candidate_report,
};
use visible_surface::{SurfaceDepthSummary, surface_depth_summary_report};

const ROTATION_FIT_CROP_RESOLUTION: u32 = 128;
const DENSE_SOFT_FIT_RESOLUTION: usize = 40;
const DENSE_SOFT_FIT_MAX_POINTS: usize = 96;
const DENSE_SOFT_FIT_ITERATIONS: usize = 10;
const ROTATION_FIT_MIN_APPLY_IMPROVEMENT: f32 = 0.04;
const VISIBLE_SURFACE_POSE_FIT_MIN_APPLY_IMPROVEMENT: f32 = 0.025;

#[derive(Clone, Copy, Debug)]
pub struct SceneRotationFitConfig<'a> {
    pub mode: SceneRotationFitMode,
    pub max_gpt_rounds: usize,
    pub min_mask_iou: f32,
    pub max_depth_error_m: f32,
    pub write_artifacts: bool,
    pub output_dir: &'a Path,
}

#[derive(Clone, Debug)]
pub struct SceneRotationFitOutcome {
    pub commands: Vec<Value>,
    pub grounded_layout: GroundedSceneLayout,
    pub report: Value,
}

pub struct SceneObjectPoseRefinementConfig<'a> {
    pub mode: SceneObjectPoseRefinementMode,
    pub object_set: SceneObjectPoseRefinementSet,
    pub pose_fit: SceneVisibleSurfacePoseFitConfig<'a>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScenePoseFitObjectFilter {
    All,
    RefinementSet(SceneObjectPoseRefinementSet),
}

impl ScenePoseFitObjectFilter {
    fn includes(self, placement: &GroundedScenePlacement) -> bool {
        match self {
            Self::All => true,
            Self::RefinementSet(object_set) => {
                object_pose_refinement_set_includes(object_set, placement)
            }
        }
    }

    fn stage_name(self) -> &'static str {
        match self {
            Self::All => "visible_surface_pose_fit",
            Self::RefinementSet(_) => "object_pose_refinement",
        }
    }

    fn report_file_stem(self) -> &'static str {
        match self {
            Self::All => "visible_surface_pose_fit",
            Self::RefinementSet(_) => "object_pose_refinement",
        }
    }

    fn output_dir_name(self) -> &'static str {
        match self {
            Self::All => "visible_surface_pose_candidates",
            Self::RefinementSet(_) => "object_pose_candidates",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::RefinementSet(object_set) => object_set.label(),
        }
    }

    fn is_refinement(self) -> bool {
        matches!(self, Self::RefinementSet(_))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SceneVisibleSurfacePoseFitConfig<'a> {
    pub mode: ScenePoseFitMode,
    pub min_mask_iou: f32,
    pub max_depth_error_m: f32,
    pub write_artifacts: bool,
    pub output_dir: &'a Path,
    pub scale_policy: SceneScalePolicy,
    pub object_filter: ScenePoseFitObjectFilter,
}

#[derive(Clone)]
struct RotationFitTarget {
    mask: BinaryMask,
    bbox: [f32; 4],
    crop_bbox: [f32; 4],
    depth_median_m: Option<f32>,
    depth_stats: Option<crate::ObjectDepthStats>,
    dense_depth_crop: Option<DenseDepthCrop>,
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

pub fn apply_scene_rotation_fit(
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
        let Some(target) = rotation_fit_target_for_placement(placement, evidence, None) else {
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
            "depth_stats": target.depth_stats,
            "depth_summary": target
                .depth_stats
                .map(SurfaceDepthSummary::from_object_stats)
                .map(surface_depth_summary_report),
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
    normalize_reused_command_scales(&mut out_commands, SceneScalePolicy::AssetPreserving);
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

pub fn apply_scene_visible_surface_pose_fit(
    config: SceneVisibleSurfacePoseFitConfig<'_>,
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

    if config.mode != ScenePoseFitMode::RenderedSilhouette {
        let report = visible_surface_pose_fit_report(
            config,
            manifest,
            0,
            grounded_layout.placements.len(),
            0,
            Vec::new(),
            "disabled",
        );
        write_visible_surface_pose_fit_artifacts_if_requested(config, manifest, &report)?;
        return Ok(SceneRotationFitOutcome {
            commands: out_commands,
            grounded_layout: out_layout,
            report,
        });
    }

    let Some(evidence) = evidence else {
        let report = visible_surface_pose_fit_report(
            config,
            manifest,
            0,
            grounded_layout.placements.len(),
            1,
            Vec::new(),
            "missing_grounding_evidence",
        );
        write_visible_surface_pose_fit_artifacts_if_requested(config, manifest, &report)?;
        return Ok(SceneRotationFitOutcome {
            commands: out_commands,
            grounded_layout: out_layout,
            report,
        });
    };
    let Some(projection_fit) = grounded_layout.projection_fit.as_ref() else {
        let report = visible_surface_pose_fit_report(
            config,
            manifest,
            0,
            grounded_layout.placements.len(),
            1,
            Vec::new(),
            "missing_projection_fit_report",
        );
        write_visible_surface_pose_fit_artifacts_if_requested(config, manifest, &report)?;
        return Ok(SceneRotationFitOutcome {
            commands: out_commands,
            grounded_layout: out_layout,
            report,
        });
    };
    let Some(intrinsics) = rotation_fit_intrinsics(evidence) else {
        let report = visible_surface_pose_fit_report(
            config,
            manifest,
            0,
            grounded_layout.placements.len(),
            1,
            Vec::new(),
            "missing_source_camera_intrinsics",
        );
        write_visible_surface_pose_fit_artifacts_if_requested(config, manifest, &report)?;
        return Ok(SceneRotationFitOutcome {
            commands: out_commands,
            grounded_layout: out_layout,
            report,
        });
    };
    let (depth_sidecar, depth_sidecar_report) = match load_scene_depth_map_sidecar(evidence) {
        Ok(depth_sidecar) => {
            let report = depth_sidecar
                .as_ref()
                .map(loaded_depth_map_report)
                .unwrap_or_else(|| json!({ "status": "absent" }));
            (depth_sidecar, report)
        }
        Err(err) => {
            warning_count += 1;
            (
                None,
                json!({
                    "status": "error",
                    "error": err,
                }),
            )
        }
    };

    let spawn_command_indices = rotation_fit_spawn_command_indices(&out_commands);
    let candidate_root = config
        .output_dir
        .join(config.object_filter.output_dir_name());
    if config.write_artifacts {
        fs::create_dir_all(&candidate_root).map_err(|err| {
            format!(
                "failed to create visible-surface pose candidate dir {}: {err}",
                candidate_root.display()
            )
        })?;
    }

    for placement_index in 0..grounded_layout.placements.len() {
        let placement = &grounded_layout.placements[placement_index];
        if !config.object_filter.includes(placement) {
            continue;
        }
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
        let Some(target) =
            rotation_fit_target_for_placement(placement, evidence, depth_sidecar.as_ref())
        else {
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

        let baseline_placement =
            command_placement_for_pose_fit(placement, out_commands.get(command_index), config);
        let target_mask = rotation_fit_target_crop_mask(&target);
        let candidates_dir =
            candidate_root.join(rotation_fit_object_dir_name(placement_index, placement));
        if config.write_artifacts {
            fs::create_dir_all(&candidates_dir).map_err(|err| {
                format!(
                    "failed to create visible-surface pose object candidate dir {}: {err}",
                    candidates_dir.display()
                )
            })?;
        }
        let candidates = evaluate_visible_surface_pose_candidates(
            &mesh,
            &baseline_placement,
            fit_object,
            out_layout.projection_fit.as_ref().map(|fit| &fit.camera),
            evidence,
            intrinsics,
            &target,
            &target_mask,
            &out_layout.placements,
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
            .find(|candidate| candidate.stage == "baseline")
            .or_else(|| candidates.first());
        let best = candidates
            .iter()
            .min_by(|left, right| left.loss.total_cmp(&right.loss));
        let selected = best.filter(|best| {
            let Some(baseline) = baseline else {
                return false;
            };
            visible_surface_pose_candidate_passes_target(best, baseline, &target, config)
        });
        let candidate_reports = candidates
            .iter()
            .map(visible_surface_pose_candidate_report)
            .collect::<Vec<_>>();
        report["target"] = json!({
            "mask_kind": target.mask_kind,
            "bbox": target.bbox,
            "crop_bbox": target.crop_bbox,
            "depth_median_m": target.depth_median_m,
            "depth_stats": target.depth_stats,
            "depth_summary": target
                .depth_stats
                .map(SurfaceDepthSummary::from_object_stats)
                .map(surface_depth_summary_report),
            "dense_depth_crop": target
                .dense_depth_crop
                .as_ref()
                .map(dense_depth_crop_report),
        });
        report["baseline"] = baseline
            .map(visible_surface_pose_candidate_report)
            .unwrap_or(Value::Null);
        report["best"] = best
            .map(visible_surface_pose_candidate_report)
            .unwrap_or(Value::Null);
        report["candidate_count"] = json!(candidate_reports.len());
        report["candidates"] = json!(candidate_reports);
        report["asset_path"] = json!(asset_path.display().to_string());

        if let Some(selected) = selected {
            applied_count += 1;
            apply_pose_fit_candidate_to_command(
                &mut out_commands[command_index],
                &selected.placement,
                config.scale_policy,
            );
            out_layout.placements[placement_index] = selected.placement.clone();
            report["applied"] = json!(true);
            report["selected"] = visible_surface_pose_candidate_report(selected);
            report["translation_before"] = json!(baseline_placement.translation);
            report["translation_after"] = json!(selected.placement.translation);
            report["scale_before"] = json!(baseline_placement.scale);
            report["scale_after"] = json!(selected.placement.scale);
            report["yaw_before_degrees"] = json!(baseline_placement.rotation_y_degrees);
            report["yaw_after_degrees"] = json!(selected.placement.rotation_y_degrees);
            report["yaw_delta_degrees"] = json!(normalize_degrees(
                selected.placement.rotation_y_degrees - baseline_placement.rotation_y_degrees
            ));
        } else {
            skipped_count += 1;
            report["skip_reason"] = json!("no_candidate_passed_or_improved_gate");
        }
        object_reports.push(report);
    }

    normalize_reused_command_scales(&mut out_commands, config.scale_policy);
    sync_layout_placements_from_commands(&mut out_layout, &out_commands);
    normalize_reused_layout_scales(&mut out_layout.placements, config.scale_policy);
    sync_commands_from_layout_placements(&mut out_commands, &out_layout.placements);
    let status = if applied_count > 0 {
        "applied"
    } else {
        "no_applicable_candidates"
    };
    let mut report = visible_surface_pose_fit_report(
        config,
        manifest,
        applied_count,
        skipped_count,
        warning_count,
        object_reports,
        status,
    );
    report["depth_map_sidecar"] = depth_sidecar_report;
    write_visible_surface_pose_fit_artifacts_if_requested(config, manifest, &report)?;
    Ok(SceneRotationFitOutcome {
        commands: out_commands,
        grounded_layout: out_layout,
        report,
    })
}

pub fn apply_scene_object_pose_refinement(
    config: SceneObjectPoseRefinementConfig<'_>,
    manifest: &SceneObjectManifest,
    asset_bindings: &[SceneAssetBinding],
    evidence: Option<&SceneGroundingEvidence>,
    grounded_layout: &GroundedSceneLayout,
    commands: &[Value],
) -> Result<SceneRotationFitOutcome, String> {
    let mode = config.mode;
    let object_set = config.object_set;
    let mut pose_config = config.pose_fit;
    pose_config.object_filter = ScenePoseFitObjectFilter::RefinementSet(object_set);
    if mode == SceneObjectPoseRefinementMode::Off {
        let mut report =
            visible_surface_pose_fit_report(pose_config, manifest, 0, 0, 0, Vec::new(), "disabled");
        report["object_pose_refinement_mode"] = json!(mode);
        report["object_pose_refinement_set"] = json!(object_set);
        report["gpt_gate"] = pose_refinement_gpt_gate(mode, &report, pose_config.object_filter);
        return Ok(SceneRotationFitOutcome {
            commands: commands.to_vec(),
            grounded_layout: grounded_layout.clone(),
            report,
        });
    }
    let mut outcome = apply_scene_visible_surface_pose_fit(
        pose_config,
        manifest,
        asset_bindings,
        evidence,
        grounded_layout,
        commands,
    )?;
    outcome.report["object_pose_refinement_mode"] = json!(mode);
    outcome.report["object_pose_refinement_set"] = json!(object_set);
    outcome.report["gpt_gate"] =
        pose_refinement_gpt_gate(mode, &outcome.report, pose_config.object_filter);
    if pose_config.write_artifacts {
        write_visible_surface_pose_fit_artifacts_if_requested(
            pose_config,
            manifest,
            &outcome.report,
        )?;
    }
    Ok(outcome)
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
    let object_count = objects.len();
    let quality_gate = pose_fit_quality_gate(
        status,
        applied_count,
        skipped_count,
        warning_count,
        object_count,
    );
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
        "object_count": object_count,
        "quality_gate": quality_gate,
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
fn visible_surface_pose_fit_report(
    config: SceneVisibleSurfacePoseFitConfig<'_>,
    manifest: &SceneObjectManifest,
    applied_count: usize,
    skipped_count: usize,
    warning_count: usize,
    objects: Vec<Value>,
    status: &str,
) -> Value {
    let object_count = objects.len();
    let quality_gate = pose_fit_quality_gate(
        status,
        applied_count,
        skipped_count,
        warning_count,
        object_count,
    );
    json!({
        "schema_version": 1,
        "stage": config.object_filter.stage_name(),
        "status": status,
        "mode": config.mode,
        "object_filter": config.object_filter.label(),
        "algorithm": "bounded-visible-surface-mask-dense-depth-coordinate-search-plus-soft-refinement",
        "source_scene_path": manifest.source_scene_path,
        "artifact_dir": config.output_dir.display().to_string(),
        "min_mask_iou": config.min_mask_iou,
        "max_depth_error_m": config.max_depth_error_m,
        "scale_policy": config.scale_policy,
        "write_artifacts": config.write_artifacts,
        "applied_count": applied_count,
        "skipped_count": skipped_count,
        "warning_count": warning_count,
        "object_count": object_count,
        "quality_gate": quality_gate,
        "variables": ["translation_x", "translation_z", "uniform_scale", "yaw_y"],
        "constraints": {
            "floor_y": "bottom_to_floor_from_layout",
            "translation": "clamped_to_camera_ray_ground_anchor",
            "scale": "scene_scale_policy_and_reused_asset_shared_scale",
            "rotation": "y_axis_only"
        },
        "objects": objects,
        "differentiable_renderer": "soft_point_surface_autodiff_refinement",
        "note": "This is the implemented deterministic depth+mask visible-surface optimizer. It rasterizes generated GLB visible faces into the source-camera crop and searches bounded yaw/XZ/scale candidates. Candidate visible-surface depth quantiles/contact depth, dense per-pixel DepthPro sidecar crops, and mask point moments are matched against projected generated GLB surfaces. A Burn autodiff soft point-surface crop refinement can add a dense candidate, which is still re-scored by the deterministic visible-surface gate before being applied."
    })
}

fn pose_fit_quality_gate(
    status: &str,
    applied_count: usize,
    skipped_count: usize,
    warning_count: usize,
    object_count: usize,
) -> Value {
    let disabled = status == "disabled";
    let no_work = object_count == 0 && skipped_count == 0;
    let no_applicable_targets = status == "no_applicable_candidates" && object_count == 0;
    let disabled_or_no_work = disabled || no_work || no_applicable_targets;
    let degraded = !disabled_or_no_work
        && (warning_count > 0 || skipped_count > 0 || (object_count > 0 && applied_count == 0));
    let reason = if degraded && warning_count > 0 {
        "warnings_present"
    } else if degraded && applied_count > 0 && skipped_count > 0 {
        "partial_candidates_applied"
    } else if degraded {
        "no_candidates_applied"
    } else if disabled_or_no_work {
        "not_applicable"
    } else {
        "passed"
    };
    json!({
        "passed": !degraded,
        "degraded": degraded,
        "hard_failure": false,
        "reason": reason,
    })
}

#[cfg(test)]
mod pose_fit_tests {
    use super::*;

    #[test]
    fn pose_fit_quality_gate_degrades_partial_large_object_failure() {
        let gate = pose_fit_quality_gate("applied", 1, 1, 0, 2);

        assert_eq!(gate["passed"], json!(false));
        assert_eq!(gate["degraded"], json!(true));
        assert_eq!(gate["reason"], json!("partial_candidates_applied"));
    }

    #[test]
    fn pose_fit_quality_gate_keeps_empty_no_work_non_degraded() {
        let gate = pose_fit_quality_gate("no_applicable_candidates", 0, 0, 0, 0);

        assert_eq!(gate["passed"], json!(true));
        assert_eq!(gate["degraded"], json!(false));
        assert_eq!(gate["reason"], json!("not_applicable"));
    }
}

fn write_visible_surface_pose_fit_artifacts_if_requested(
    config: SceneVisibleSurfacePoseFitConfig<'_>,
    manifest: &SceneObjectManifest,
    report: &Value,
) -> Result<(), String> {
    if !config.write_artifacts {
        return Ok(());
    }
    fs::create_dir_all(config.output_dir).map_err(|err| {
        format!(
            "failed to create visible-surface pose-fit output dir {}: {err}",
            config.output_dir.display()
        )
    })?;
    write_json_file(
        &config.output_dir.join(format!(
            "{}_report.json",
            config.object_filter.report_file_stem()
        )),
        report,
    )
    .map_err(|err| err.to_string())?;
    let overlay_path = config.output_dir.join(format!(
        "{}_overlay.png",
        config.object_filter.report_file_stem()
    ));
    if let Err(err) = write_rotation_fit_overlay(manifest, report, &overlay_path) {
        write_json_file(
            &config.output_dir.join(format!(
                "{}_overlay_error.json",
                config.object_filter.report_file_stem()
            )),
            &json!({ "path": overlay_path, "error": err }),
        )
        .map_err(|err| err.to_string())?;
    }
    let html = visible_surface_pose_fit_review_html(report);
    fs::write(
        config.output_dir.join(format!(
            "{}_review.html",
            config.object_filter.report_file_stem()
        )),
        html,
    )
    .map_err(|err| {
        format!(
            "failed to write visible-surface pose-fit review html {}: {err}",
            config
                .output_dir
                .join(format!(
                    "{}_review.html",
                    config.object_filter.report_file_stem()
                ))
                .display()
        )
    })?;
    Ok(())
}

pub(super) fn placement_is_table_like(placement: &GroundedScenePlacement) -> bool {
    let descriptor = format!(
        "{} {} {}",
        placement.object_id, placement.label, placement.entity_id
    )
    .to_ascii_lowercase();
    descriptor.contains("table")
        || descriptor.contains("desk")
        || descriptor.contains("coffee")
        || descriptor.contains("counter")
}

pub(super) fn placement_is_sofa_like(placement: &GroundedScenePlacement) -> bool {
    let descriptor = format!(
        "{} {} {}",
        placement.object_id, placement.label, placement.entity_id
    )
    .to_ascii_lowercase();
    descriptor.contains("sofa")
        || descriptor.contains("couch")
        || descriptor.contains("sectional")
        || descriptor.contains("loveseat")
        || descriptor.contains("settee")
}

pub(super) fn placement_is_large_seating_like(placement: &GroundedScenePlacement) -> bool {
    placement_is_sofa_like(placement)
}

pub(super) fn placement_is_furniture_like(placement: &GroundedScenePlacement) -> bool {
    placement_is_table_like(placement)
        || placement_is_large_seating_like(placement)
        || placement_descriptor(placement).contains("chair")
        || placement_descriptor(placement).contains("seat")
        || placement_descriptor(placement).contains("bench")
}

pub(super) fn object_pose_refinement_set_includes(
    object_set: SceneObjectPoseRefinementSet,
    placement: &GroundedScenePlacement,
) -> bool {
    match object_set {
        SceneObjectPoseRefinementSet::Tables => placement_is_table_like(placement),
        SceneObjectPoseRefinementSet::LargeSeating => placement_is_large_seating_like(placement),
        SceneObjectPoseRefinementSet::TablesAndLargeSeating => {
            placement_is_table_like(placement) || placement_is_large_seating_like(placement)
        }
        SceneObjectPoseRefinementSet::AllFurniture => placement_is_furniture_like(placement),
    }
}

fn placement_descriptor(placement: &GroundedScenePlacement) -> String {
    format!(
        "{} {} {}",
        placement.object_id, placement.label, placement.entity_id
    )
    .to_ascii_lowercase()
}

fn pose_refinement_gpt_gate(
    mode: SceneObjectPoseRefinementMode,
    report: &Value,
    object_filter: ScenePoseFitObjectFilter,
) -> Value {
    let mut required_objects = Vec::new();
    if !mode.gpt_allowed() {
        return json!({
            "enabled": false,
            "required": false,
            "reason": "mode_does_not_allow_gpt",
            "objects": required_objects,
        });
    }
    for object in report
        .get("objects")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        if mode == SceneObjectPoseRefinementMode::AlwaysGpt {
            required_objects.push(pose_refinement_gpt_gate_object_report(
                object,
                "mode_always_gpt",
            ));
            continue;
        }
        if !object
            .get("applied")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            required_objects.push(pose_refinement_gpt_gate_object_report(
                object,
                "no_geometry_candidate_applied",
            ));
            continue;
        }
        let selected = object.get("selected").unwrap_or(&Value::Null);
        let baseline = object.get("baseline").unwrap_or(&Value::Null);
        let mask_iou = selected
            .get("mask_iou")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        let bbox_iou = selected
            .get("bbox_iou")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        let center_error = selected
            .get("center_error")
            .and_then(Value::as_f64)
            .unwrap_or(1.0);
        let selected_loss = selected
            .get("loss")
            .and_then(Value::as_f64)
            .unwrap_or(f64::INFINITY);
        let baseline_loss = baseline
            .get("loss")
            .and_then(Value::as_f64)
            .unwrap_or(f64::INFINITY);
        let yaw_delta = object
            .get("yaw_delta_degrees")
            .and_then(Value::as_f64)
            .unwrap_or(0.0)
            .abs();
        let low_overlap = mask_iou < 0.38 && bbox_iou < 0.52;
        let off_center = center_error > 0.10;
        let ambiguous_loss = baseline_loss.is_finite()
            && selected_loss.is_finite()
            && baseline_loss - selected_loss < 0.075
            && yaw_delta > 20.0;
        if low_overlap {
            required_objects.push(pose_refinement_gpt_gate_object_report(
                object,
                "low_mask_and_bbox_overlap",
            ));
        } else if off_center {
            required_objects.push(pose_refinement_gpt_gate_object_report(
                object,
                "projected_center_error",
            ));
        } else if ambiguous_loss {
            required_objects.push(pose_refinement_gpt_gate_object_report(
                object,
                "ambiguous_geometry_margin",
            ));
        }
    }
    json!({
        "enabled": true,
        "required": !required_objects.is_empty(),
        "mode": mode,
        "object_filter": object_filter.label(),
        "objects": required_objects,
        "note": format!(
            "GPT is only a bounded candidate selector for {} fits flagged by objective geometry. It is not a source of absolute transform truth.",
            object_filter.label()
        ),
    })
}

fn pose_refinement_gpt_gate_object_report(object: &Value, reason: &'static str) -> Value {
    json!({
        "object_id": object.get("object_id").cloned().unwrap_or(Value::Null),
        "instance_id": object.get("instance_id").cloned().unwrap_or(Value::Null),
        "label": object.get("label").cloned().unwrap_or(Value::Null),
        "reason": reason,
        "baseline": object.get("baseline").cloned().unwrap_or(Value::Null),
        "best": object.get("best").cloned().unwrap_or(Value::Null),
        "selected": object.get("selected").cloned().unwrap_or(Value::Null),
    })
}

fn visible_surface_pose_fit_review_html(report: &Value) -> String {
    let mut html = String::new();
    html.push_str("<!doctype html><html><head><meta charset=\"utf-8\"><title>visible surface pose fit</title><style>");
    html.push_str("body{background:#101216;color:#e8eaed;font-family:system-ui,sans-serif;margin:24px}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #2a2d34;padding:8px;text-align:left;vertical-align:top}.ok{color:#75e39a}.bad{color:#ff8b8b}img{width:96px;height:96px;image-rendering:pixelated;border:1px solid #333}.candidates{display:flex;gap:6px;flex-wrap:wrap}.meta{color:#9aa0aa}</style></head><body>");
    let _ = write!(
        html,
        "<h1>Visible Surface Pose Fit</h1><p class=\"meta\">status: {}, applied: {}, skipped: {}</p>",
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
    html.push_str("<p class=\"meta\">candidate colors: green=source SAM/bbox target, blue=projected mesh surface, white=overlap. The selected candidate is applied before render feedback.</p>");
    html.push_str("<table><thead><tr><th>object</th><th>decision</th><th>best</th><th>selected</th><th>candidates</th></tr></thead><tbody>");
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
        let _ = write!(
            html,
            "<tr><td><strong>{}</strong><br><span class=\"meta\">{}</span></td><td class=\"{}\">{}</td><td><pre>{}</pre></td><td><pre>{}</pre></td><td><div class=\"candidates\">",
            html_escape(object_id),
            html_escape(label),
            class,
            html_escape(decision),
            html_escape(
                &serde_json::to_string_pretty(object.get("best").unwrap_or(&Value::Null))
                    .unwrap_or_default()
            ),
            html_escape(
                &serde_json::to_string_pretty(object.get("selected").unwrap_or(&Value::Null))
                    .unwrap_or_default()
            ),
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
                "idx {} stage {} yaw {} loss {:.3} mask {:.3} bbox {:.3}",
                candidate
                    .get("candidate_index")
                    .and_then(Value::as_u64)
                    .unwrap_or(0),
                candidate.get("stage").and_then(Value::as_str).unwrap_or(""),
                candidate
                    .get("yaw_degrees")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0),
                candidate.get("loss").and_then(Value::as_f64).unwrap_or(0.0),
                candidate
                    .get("mask_iou")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0),
                candidate
                    .get("bbox_iou")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0),
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
    fn pose_fit_quality_gate_distinguishes_degraded_from_no_work() {
        let passed = pose_fit_quality_gate("complete", 2, 0, 0, 2);
        assert_eq!(passed["passed"], json!(true));
        assert_eq!(passed["reason"], json!("passed"));

        let no_work = pose_fit_quality_gate("no_applicable_candidates", 0, 3, 0, 0);
        assert_eq!(no_work["passed"], json!(true));
        assert_eq!(no_work["reason"], json!("not_applicable"));

        let degraded = pose_fit_quality_gate("complete", 0, 3, 0, 3);
        assert_eq!(degraded["passed"], json!(false));
        assert_eq!(degraded["reason"], json!("no_candidates_applied"));

        let warning = pose_fit_quality_gate("complete", 2, 1, 1, 3);
        assert_eq!(warning["passed"], json!(false));
        assert_eq!(warning["reason"], json!("warnings_present"));
    }
}

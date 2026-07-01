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
const VISIBLE_SURFACE_SEMANTIC_YAW_REPAIR_MAX_LOSS_REGRESSION: f32 = 0.35;
const FINAL_YAW_LOSS_REGRESSION_EPSILON: f32 = 0.25;
const FINAL_YAW_BBOX_REGRESSION_EPSILON: f32 = 0.08;
const FINAL_YAW_MASK_REGRESSION_EPSILON: f32 = 0.04;
const FINAL_YAW_DEPTH_REGRESSION_EPSILON_M: f32 = 0.35;
const FINAL_YAW_POSE_TRANSLATION_MAX_DELTA_M: f32 = 0.65;
const FINAL_YAW_POSE_SCALE_RATIO_MIN: f32 = 0.68;
const FINAL_YAW_POSE_SCALE_RATIO_MAX: f32 = 1.48;

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

#[derive(Clone, Copy, Debug)]
pub struct SceneFinalYawRefinementConfig<'a> {
    pub mode: SceneFinalYawRefinementMode,
    pub object_set: SceneObjectPoseRefinementSet,
    pub confidence_threshold: f32,
    pub max_candidates: usize,
    pub write_artifacts: bool,
    pub output_dir: &'a Path,
    pub grounding_evidence: Option<&'a SceneGroundingEvidence>,
    pub rendered_selection_task: Option<&'a Value>,
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
        let selected = baseline.and_then(|baseline| {
            candidates
                .iter()
                .filter(|candidate| {
                    visible_surface_pose_candidate_passes_target(
                        candidate, baseline, &target, config,
                    )
                })
                .min_by(|left, right| left.loss.total_cmp(&right.loss))
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

pub fn apply_scene_final_yaw_refinement(
    config: SceneFinalYawRefinementConfig<'_>,
    manifest: &SceneObjectManifest,
    asset_bindings: &[SceneAssetBinding],
    grounded_layout: &GroundedSceneLayout,
    commands: &[Value],
    prior_pose_report: Option<&Value>,
    response: Option<&SceneRotationSelectionResponse>,
) -> Result<SceneRotationFitOutcome, String> {
    let mut out_commands = commands.to_vec();
    let mut out_layout = grounded_layout.clone();
    if !config.mode.enabled() {
        let report = final_yaw_refinement_report(
            config,
            manifest,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            "disabled",
        );
        write_final_yaw_refinement_artifacts_if_requested(config, &report)?;
        return Ok(SceneRotationFitOutcome {
            commands: out_commands,
            grounded_layout: out_layout,
            report,
        });
    }

    let spawn_command_indices = rotation_fit_spawn_command_indices(commands);
    let mut objects = final_yaw_refinement_objects(
        config,
        asset_bindings,
        grounded_layout,
        commands,
        prior_pose_report,
        &spawn_command_indices,
    );
    let mut applied = Vec::new();
    let mut ignored = Vec::new();
    if let Some(response) = response {
        for selection in &response.objects {
            let Some(object_position) = objects.iter().position(|object| {
                object.get("index").and_then(Value::as_u64) == Some(selection.index as u64)
            }) else {
                ignored.push(json!({
                    "index": selection.index,
                    "candidate_index": selection.candidate_index,
                    "reason": "object_index_not_available",
                }));
                continue;
            };
            let object = &mut objects[object_position];
            let Some(gpt_candidate) =
                final_yaw_refinement_candidate_for_object(object, selection.candidate_index)
            else {
                ignored.push(json!({
                    "index": selection.index,
                    "candidate_index": selection.candidate_index,
                    "reason": "candidate_index_not_available",
                }));
                continue;
            };
            let (candidate, selection_policy, override_reason) = final_yaw_metric_guarded_candidate(
                config.mode,
                object,
                gpt_candidate,
                selection.confidence,
                config.confidence_threshold,
            );
            let accepted_candidate_index = candidate
                .get("candidate_index")
                .and_then(Value::as_u64)
                .map(|value| value as usize)
                .unwrap_or(selection.candidate_index);
            if selection.confidence < config.confidence_threshold {
                ignored.push(json!({
                    "index": selection.index,
                    "candidate_index": selection.candidate_index,
                    "confidence": selection.confidence,
                    "threshold": config.confidence_threshold,
                    "reason": "confidence_below_threshold",
                }));
                continue;
            }
            if let Err(reason) =
                final_yaw_refinement_candidate_passes_gate(config.mode, object, &candidate)
            {
                ignored.push(json!({
                    "index": selection.index,
                    "candidate_index": selection.candidate_index,
                    "accepted_candidate_index": accepted_candidate_index,
                    "confidence": selection.confidence,
                    "reason": reason,
                    "selection_policy": selection_policy,
                    "override_reason": override_reason,
                }));
                continue;
            }
            let Some(command_index) = object
                .get("command_index")
                .and_then(Value::as_u64)
                .map(|value| value as usize)
            else {
                ignored.push(json!({
                    "index": selection.index,
                    "candidate_index": selection.candidate_index,
                    "reason": "missing_command_index",
                }));
                continue;
            };
            let Some(placement_index) = object
                .get("placement_index")
                .and_then(Value::as_u64)
                .map(|value| value as usize)
            else {
                ignored.push(json!({
                    "index": selection.index,
                    "candidate_index": selection.candidate_index,
                    "reason": "missing_placement_index",
                }));
                continue;
            };
            let Some(yaw) = candidate
                .get("candidate_yaw_degrees")
                .and_then(Value::as_f64)
                .map(|value| normalize_degrees(value as f32))
            else {
                ignored.push(json!({
                    "index": selection.index,
                    "candidate_index": selection.candidate_index,
                    "reason": "candidate_missing_yaw",
                }));
                continue;
            };
            let pose_apply = final_yaw_pose_apply_for_candidate(object, &candidate);
            if let Some(command) = out_commands.get_mut(command_index) {
                if let Some(pose_apply) = &pose_apply {
                    command["translation"] = json!(pose_apply.translation);
                    command["scale"] = json!(pose_apply.scale);
                }
                command["rotation"] = json!(quat_from_y_degrees(yaw));
            }
            if let Some(placement) = out_layout.placements.get_mut(placement_index) {
                if let Some(pose_apply) = &pose_apply {
                    placement.translation = pose_apply.translation;
                    placement.scale = pose_apply.scale;
                    placement.rotation_y_degrees = yaw;
                    placement.sync_ground_anchor_from_current_translation();
                } else {
                    placement.rotation_y_degrees = yaw;
                    placement.sync_translation_to_current_ground_anchor();
                }
            }
            object["applied"] = json!(true);
            object["selected_candidate_index"] = json!(accepted_candidate_index);
            object["selected_yaw_degrees"] = json!(yaw);
            if let Some(pose_apply) = &pose_apply {
                object["selected_pose_application"] = json!({
                    "applied": true,
                    "translation": pose_apply.translation,
                    "scale": pose_apply.scale,
                    "reason": pose_apply.reason,
                });
            }
            object["selector_result"] = json!({
                "gpt_candidate_index": selection.candidate_index,
                "accepted_candidate_index": accepted_candidate_index,
                "confidence": selection.confidence,
                "rationale": selection.rationale,
                "selection_policy": selection_policy,
                "override_reason": override_reason,
            });
            if let Some(candidates) = object
                .pointer_mut("/rotation_selection/candidates")
                .and_then(Value::as_array_mut)
            {
                for item in candidates {
                    let candidate_index = item
                        .get("candidate_index")
                        .and_then(Value::as_u64)
                        .map(|value| value as usize);
                    item["selected"] = json!(candidate_index == Some(accepted_candidate_index));
                }
            }
            applied.push(json!({
                "index": selection.index,
                "candidate_index": accepted_candidate_index,
                "accepted_candidate_index": accepted_candidate_index,
                "gpt_candidate_index": selection.candidate_index,
                "yaw_degrees": yaw,
                "pose_application": pose_apply
                    .as_ref()
                    .map(|pose| {
                        json!({
                            "translation": pose.translation,
                            "scale": pose.scale,
                            "reason": pose.reason,
                        })
                    })
                    .unwrap_or(Value::Null),
                "confidence": selection.confidence,
                "selection_policy": selection_policy,
                "override_reason": override_reason,
            }));
        }
    }

    sync_layout_placements_from_commands(&mut out_layout, &out_commands);
    let status = if !applied.is_empty() {
        "applied"
    } else if response.is_some() {
        "selection_rejected_or_noop"
    } else if objects.is_empty() {
        "no_targets"
    } else {
        "awaiting_candidate_selection"
    };
    let report = final_yaw_refinement_report(config, manifest, objects, applied, ignored, status);
    write_final_yaw_refinement_artifacts_if_requested(config, &report)?;
    Ok(SceneRotationFitOutcome {
        commands: out_commands,
        grounded_layout: out_layout,
        report,
    })
}

fn final_yaw_refinement_objects(
    config: SceneFinalYawRefinementConfig<'_>,
    asset_bindings: &[SceneAssetBinding],
    grounded_layout: &GroundedSceneLayout,
    commands: &[Value],
    prior_pose_report: Option<&Value>,
    spawn_command_indices: &[usize],
) -> Vec<Value> {
    let mut objects = Vec::new();
    for placement_index in 0..grounded_layout.placements.len() {
        let placement = &grounded_layout.placements[placement_index];
        if !object_pose_refinement_set_includes(config.object_set, placement) {
            continue;
        }
        let Some(command_index) = spawn_command_indices.get(placement_index).copied() else {
            continue;
        };
        let prior = final_yaw_prior_object_report(prior_pose_report, placement);
        if config.mode == SceneFinalYawRefinementMode::GatedGpt
            && !final_yaw_refinement_object_requires_selection(placement, prior)
        {
            continue;
        }
        let command = commands.get(command_index);
        let current_yaw = command
            .and_then(|command| command.get("rotation"))
            .and_then(json_array4)
            .map(quat_y_degrees)
            .unwrap_or(placement.rotation_y_degrees);
        let mut candidates = Vec::new();
        let current_metrics = prior.and_then(|prior| final_yaw_current_metrics(prior, current_yaw));
        final_yaw_push_candidate(
            &mut candidates,
            current_yaw,
            current_yaw,
            "current",
            current_metrics.as_ref().map(|(_, metrics)| metrics.clone()),
            current_metrics.map(|(source, _)| source),
            config.max_candidates,
        );
        if let Some(prior) = prior {
            for key in ["selected", "best", "baseline"] {
                if let Some(candidate) = prior.get(key)
                    && !candidate.is_null()
                    && let Some(yaw) = candidate.get("yaw_degrees").and_then(Value::as_f64)
                {
                    final_yaw_push_candidate(
                        &mut candidates,
                        current_yaw,
                        yaw as f32,
                        key,
                        Some(candidate.clone()),
                        Some(key),
                        config.max_candidates,
                    );
                }
            }
            let mut geometry_candidates = prior
                .get("candidates")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|candidate| {
                    let yaw = candidate.get("yaw_degrees").and_then(Value::as_f64)? as f32;
                    let loss = candidate
                        .get("loss")
                        .and_then(Value::as_f64)
                        .unwrap_or(f64::INFINITY);
                    Some((loss, yaw, candidate.clone()))
                })
                .collect::<Vec<_>>();
            geometry_candidates.sort_by(|left, right| left.0.total_cmp(&right.0));
            for (_, yaw, candidate) in geometry_candidates.into_iter().take(6) {
                final_yaw_push_candidate(
                    &mut candidates,
                    current_yaw,
                    yaw,
                    "geometry_top_loss",
                    Some(candidate),
                    Some("geometry_top_loss"),
                    config.max_candidates,
                );
            }
        }
        for delta in [180.0, -180.0, 90.0, -90.0, 45.0, -45.0, 30.0, -30.0] {
            final_yaw_push_candidate(
                &mut candidates,
                current_yaw,
                current_yaw + delta,
                "cardinal_probe",
                None,
                None,
                config.max_candidates,
            );
        }
        let measured_best = final_yaw_best_measured_candidate_from_candidates(&candidates);
        let source_crop = final_yaw_source_crop_for_placement(placement, asset_bindings);
        let source_evidence =
            final_yaw_grounding_evidence_for_placement(config.grounding_evidence, placement);
        let source_mask = source_evidence
            .and_then(|evidence| evidence.mask.as_ref())
            .map(final_yaw_source_mask_report)
            .unwrap_or(Value::Null);
        let source_mask_path = source_evidence
            .and_then(|evidence| evidence.mask.as_ref())
            .and_then(|mask| mask.mask_png_path.clone())
            .map(Value::String)
            .unwrap_or(Value::Null);
        let source_depth_stats = source_evidence
            .and_then(|evidence| evidence.depth_stats)
            .map(|depth_stats| json!(depth_stats))
            .unwrap_or(Value::Null);
        let baseline = prior
            .and_then(|report| report.get("baseline").cloned())
            .unwrap_or(Value::Null);
        let best = prior
            .and_then(|report| report.get("best").cloned())
            .unwrap_or(Value::Null);
        let mut object = json!({
            "index": placement_index,
            "placement_index": placement_index,
            "command_index": command_index,
            "object_id": placement.object_id,
            "instance_id": placement.instance_id,
            "label": placement.label,
            "asset_id": placement.asset_id,
            "source_crop": source_crop,
            "source_mask": source_mask,
            "source_mask_path": source_mask_path,
            "source_depth_stats": source_depth_stats,
            "source_bbox": placement.source_bbox,
            "current_yaw_degrees": current_yaw,
            "asset_yaw_offset_degrees": placement.asset_yaw_offset_degrees,
            "translation": placement.translation,
            "scale": placement.scale,
            "baseline": baseline,
            "best": best,
            "measured_best_candidate_index": measured_best
                .as_ref()
                .and_then(|candidate| candidate.get("candidate_index"))
                .cloned()
                .unwrap_or(Value::Null),
            "measured_best_yaw_degrees": measured_best
                .as_ref()
                .and_then(|candidate| candidate.get("candidate_yaw_degrees"))
                .cloned()
                .unwrap_or(Value::Null),
            "applied": false,
            "requires_selection": true,
            "rotation_selection": {
                "selection_source": "final_contextual_measured_pose_candidates",
                "instruction": "Choose candidate_index only. This final stage may apply the selected measured candidate pose for the target object; do not invent transforms outside the listed candidates.",
                "current_yaw_degrees": current_yaw,
                "selected_candidate_index": 0,
                "selected_yaw_degrees": current_yaw,
                "candidates": candidates,
            },
        });
        final_yaw_attach_rendered_selection_task_evidence(
            &mut object,
            config.rendered_selection_task,
            placement_index,
        );
        objects.push(object);
    }
    objects
}

fn final_yaw_attach_rendered_selection_task_evidence(
    object: &mut Value,
    rendered_selection_task: Option<&Value>,
    placement_index: usize,
) {
    let Some(rendered_object) =
        final_yaw_rendered_selection_task_object(rendered_selection_task, placement_index)
    else {
        return;
    };
    for key in ["current_full_scene_render"] {
        if let Some(value) = rendered_object.get(key) {
            object[key] = value.clone();
        }
    }
    let Some(rendered_candidates) = rendered_object
        .pointer("/rotation_selection/candidates")
        .and_then(Value::as_array)
    else {
        return;
    };
    let Some(candidates) = object
        .pointer_mut("/rotation_selection/candidates")
        .and_then(Value::as_array_mut)
    else {
        return;
    };
    for candidate in candidates.iter_mut() {
        let candidate_index = candidate
            .get("candidate_index")
            .and_then(Value::as_u64)
            .map(|value| value as usize);
        let Some(rendered_candidate) = rendered_candidates.iter().find(|rendered_candidate| {
            rendered_candidate
                .get("candidate_index")
                .and_then(Value::as_u64)
                .map(|value| value as usize)
                == candidate_index
        }) else {
            continue;
        };
        for key in [
            "rendered_candidate_full_scene",
            "rendered_candidate_full_frame",
            "rendered_candidate_capture",
            "rendered_candidate_bbox",
            "rendered_candidate_crop",
            "rendered_candidate_object_only_full_frame",
            "rendered_candidate_object_only_capture",
            "final_yaw_visual_fit",
        ] {
            if let Some(value) = rendered_candidate.get(key) {
                candidate[key] = value.clone();
            }
        }
    }
    for rendered_candidate in rendered_candidates {
        let candidate_index = rendered_candidate
            .get("candidate_index")
            .and_then(Value::as_u64)
            .map(|value| value as usize);
        let exists = candidates.iter().any(|candidate| {
            candidate
                .get("candidate_index")
                .and_then(Value::as_u64)
                .map(|value| value as usize)
                == candidate_index
        });
        if !exists {
            candidates.push(rendered_candidate.clone());
        }
    }
}

fn final_yaw_rendered_selection_task_object(
    rendered_selection_task: Option<&Value>,
    placement_index: usize,
) -> Option<&Value> {
    rendered_selection_task?
        .get("objects")
        .and_then(Value::as_array)?
        .iter()
        .find(|object| {
            object
                .get("placement_index")
                .and_then(Value::as_u64)
                .map(|value| value as usize)
                == Some(placement_index)
                || object
                    .get("index")
                    .and_then(Value::as_u64)
                    .map(|value| value as usize)
                    == Some(placement_index)
        })
}

fn final_yaw_current_metrics(prior: &Value, current_yaw: f32) -> Option<(&'static str, Value)> {
    for key in ["selected", "baseline", "best"] {
        let Some(candidate) = prior.get(key) else {
            continue;
        };
        if final_yaw_candidate_matches_yaw(candidate, current_yaw) {
            return Some((key, candidate.clone()));
        }
    }
    prior
        .get("candidates")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .find(|candidate| final_yaw_candidate_matches_yaw(candidate, current_yaw))
        .cloned()
        .map(|candidate| ("candidates", candidate))
}

fn final_yaw_candidate_matches_yaw(candidate: &Value, yaw: f32) -> bool {
    candidate
        .get("yaw_degrees")
        .or_else(|| candidate.get("candidate_yaw_degrees"))
        .and_then(Value::as_f64)
        .map(|candidate_yaw| normalize_degrees(candidate_yaw as f32 - yaw).abs() < 0.5)
        .unwrap_or(false)
}

fn final_yaw_prior_object_report<'a>(
    prior_pose_report: Option<&'a Value>,
    placement: &GroundedScenePlacement,
) -> Option<&'a Value> {
    prior_pose_report
        .and_then(|report| report.get("objects"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .find(|object| {
            object.get("object_id").and_then(Value::as_str) == Some(placement.object_id.as_str())
                && object
                    .get("instance_id")
                    .and_then(Value::as_str)
                    .map(Some)
                    .unwrap_or(None)
                    == placement.instance_id.as_deref()
        })
        .or_else(|| {
            prior_pose_report
                .and_then(|report| report.get("objects"))
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .find(|object| {
                    object.get("object_id").and_then(Value::as_str)
                        == Some(placement.object_id.as_str())
                        && object.get("instance_id").is_none_or(Value::is_null)
                })
        })
}

fn final_yaw_refinement_object_requires_selection(
    placement: &GroundedScenePlacement,
    prior: Option<&Value>,
) -> bool {
    if placement_is_large_seating_like(placement) {
        return true;
    }
    let Some(prior) = prior else {
        return true;
    };
    if !prior
        .get("applied")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return true;
    }
    let selected = prior.get("selected").unwrap_or(&Value::Null);
    let mask_iou = selected
        .get("mask_iou")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let bbox_iou = selected
        .get("bbox_iou")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    mask_iou < 0.35 || bbox_iou < 0.50 || placement_is_table_like(placement)
}

fn final_yaw_push_candidate(
    candidates: &mut Vec<Value>,
    current_yaw: f32,
    yaw: f32,
    source: &'static str,
    metrics: Option<Value>,
    metrics_source: Option<&str>,
    max_candidates: usize,
) {
    if candidates.len() >= max_candidates.max(1) {
        return;
    }
    let yaw = normalize_degrees(yaw);
    if candidates.iter().any(|candidate| {
        candidate
            .get("candidate_yaw_degrees")
            .and_then(Value::as_f64)
            .map(|existing| normalize_degrees(existing as f32 - yaw).abs() < 0.5)
            .unwrap_or(false)
    }) {
        return;
    }
    let candidate_index = candidates.len();
    let mut candidate = json!({
        "candidate_index": candidate_index,
        "candidate_yaw_degrees": yaw,
        "yaw_delta_degrees": normalize_degrees(yaw - current_yaw),
        "source": source,
        "selected": candidate_index == 0,
    });
    if let Some(metrics) = metrics {
        candidate["geometry_metrics"] = metrics.clone();
        if let Some(metrics_source) = metrics_source {
            candidate["metrics_source"] = json!(metrics_source);
        }
        for key in [
            "mask_iou",
            "bbox_iou",
            "center_error",
            "depth_error_m",
            "surface_depth_loss",
            "dense_depth_loss",
            "loss",
            "passed",
            "projected_bbox",
            "artifact_path",
        ] {
            if let Some(value) = metrics.get(key) {
                candidate[key] = value.clone();
            }
        }
    }
    candidates.push(candidate);
}

fn final_yaw_source_crop_for_placement(
    placement: &GroundedScenePlacement,
    asset_bindings: &[SceneAssetBinding],
) -> Value {
    asset_bindings
        .iter()
        .find(|binding| binding.asset_id == placement.asset_id)
        .or_else(|| {
            asset_bindings
                .iter()
                .find(|binding| binding.object_id == placement.object_id)
        })
        .and_then(|binding| {
            binding
                .provenance
                .as_ref()
                .and_then(|provenance| provenance.source_crop_path.clone())
                .or_else(|| binding.source_image_path.clone())
        })
        .map(Value::String)
        .unwrap_or(Value::Null)
}

fn final_yaw_grounding_evidence_for_placement<'a>(
    evidence: Option<&'a SceneGroundingEvidence>,
    placement: &GroundedScenePlacement,
) -> Option<&'a ObjectGroundingEvidence> {
    let objects = &evidence?.objects;
    objects
        .iter()
        .find(|object| {
            object.object_id == placement.object_id && object.instance_id == placement.instance_id
        })
        .or_else(|| {
            objects.iter().find(|object| {
                object.object_id == placement.object_id && object.instance_id.is_none()
            })
        })
        .or_else(|| {
            placement.instance_id.as_ref().and_then(|instance_id| {
                objects.iter().find(|object| {
                    object.instance_id.as_ref() == Some(instance_id)
                        && object
                            .reuse_group
                            .as_ref()
                            .is_none_or(|reuse_group| reuse_group == &placement.object_id)
                })
            })
        })
}

fn final_yaw_source_mask_report(mask: &ObjectMaskEvidence) -> Value {
    json!({
        "provider": mask.provider,
        "model": mask.model,
        "bbox": mask.bbox,
        "score": mask.score,
        "area_px": mask.area_px,
        "image_size": mask.image_size,
        "center_pixel": mask.center_pixel,
        "contact_pixel": mask.contact_pixel,
        "coverage": mask.coverage,
        "artifact_path": mask.artifact_path,
        "mask_png_path": mask.mask_png_path,
    })
}

fn final_yaw_refinement_candidate_for_object(
    object: &Value,
    candidate_index: usize,
) -> Option<Value> {
    object
        .pointer("/rotation_selection/candidates")
        .and_then(Value::as_array)?
        .iter()
        .find(|candidate| {
            candidate
                .get("candidate_index")
                .and_then(Value::as_u64)
                .map(|value| value as usize)
                == Some(candidate_index)
        })
        .cloned()
}

fn final_yaw_metric_guarded_candidate(
    mode: SceneFinalYawRefinementMode,
    object: &Value,
    candidate: Value,
    confidence: f32,
    confidence_threshold: f32,
) -> (Value, &'static str, Option<&'static str>) {
    if mode != SceneFinalYawRefinementMode::GatedGpt {
        return (candidate, "gpt_selection", None);
    }
    let Some(best) = final_yaw_best_measured_candidate(object) else {
        return (candidate, "gpt_selection_no_measured_best", None);
    };
    let selected_index = candidate
        .get("candidate_index")
        .and_then(Value::as_u64)
        .map(|value| value as usize);
    let best_index = best
        .get("candidate_index")
        .and_then(Value::as_u64)
        .map(|value| value as usize);
    if selected_index == best_index {
        return (candidate, "gpt_selected_metric_best", None);
    }
    if let Some(reason) = final_yaw_metric_regression_reason(&candidate, &best) {
        if final_yaw_rendered_context_selection_may_override_metric_guard(
            object,
            &candidate,
            confidence,
            confidence_threshold,
        ) {
            return (candidate, "gpt_rendered_context_selection", Some(reason));
        }
        return (best, "gpt_overridden_by_metric_gate", Some(reason));
    }
    (candidate, "gpt_selection_metric_compatible", None)
}

fn final_yaw_best_measured_candidate(object: &Value) -> Option<Value> {
    let candidates = object
        .pointer("/rotation_selection/candidates")
        .and_then(Value::as_array)?;
    final_yaw_best_measured_candidate_from_candidates(candidates)
}

fn final_yaw_best_measured_candidate_from_candidates(candidates: &[Value]) -> Option<Value> {
    candidates
        .iter()
        .filter(|candidate| candidate.get("geometry_metrics").is_some())
        .filter_map(|candidate| {
            final_yaw_candidate_quality_score(candidate).map(|score| (score, candidate))
        })
        .min_by(|left, right| left.0.total_cmp(&right.0))
        .map(|(_, candidate)| candidate.clone())
}

fn final_yaw_candidate_quality_score(candidate: &Value) -> Option<f32> {
    final_yaw_metric(candidate, "loss").or_else(|| {
        let bbox = final_yaw_metric(candidate, "bbox_iou");
        let mask = final_yaw_metric(candidate, "mask_iou");
        let depth = final_yaw_metric(candidate, "depth_error_m");
        let surface = final_yaw_metric(candidate, "surface_depth_loss");
        if bbox.is_none() && mask.is_none() && depth.is_none() && surface.is_none() {
            return None;
        }
        Some(
            (1.0 - bbox.unwrap_or(0.0))
                + (1.0 - mask.unwrap_or(0.0))
                + depth.unwrap_or(0.0) * 0.25
                + surface.unwrap_or(0.0) * 0.10,
        )
    })
}

fn final_yaw_metric_regression_reason(
    candidate: &Value,
    measured_best: &Value,
) -> Option<&'static str> {
    if let (Some(candidate_loss), Some(best_loss)) = (
        final_yaw_metric(candidate, "loss"),
        final_yaw_metric(measured_best, "loss"),
    ) && candidate_loss > best_loss + FINAL_YAW_LOSS_REGRESSION_EPSILON
    {
        return Some("loss_regression_vs_measured_best");
    }
    if let (Some(candidate_bbox), Some(best_bbox), Some(candidate_mask), Some(best_mask)) = (
        final_yaw_metric(candidate, "bbox_iou"),
        final_yaw_metric(measured_best, "bbox_iou"),
        final_yaw_metric(candidate, "mask_iou"),
        final_yaw_metric(measured_best, "mask_iou"),
    ) && candidate_bbox + FINAL_YAW_BBOX_REGRESSION_EPSILON < best_bbox
        && candidate_mask + FINAL_YAW_MASK_REGRESSION_EPSILON < best_mask
    {
        return Some("bbox_mask_regression_vs_measured_best");
    }
    if let (Some(candidate_depth), Some(best_depth), Some(candidate_bbox), Some(best_bbox)) = (
        final_yaw_metric(candidate, "depth_error_m"),
        final_yaw_metric(measured_best, "depth_error_m"),
        final_yaw_metric(candidate, "bbox_iou"),
        final_yaw_metric(measured_best, "bbox_iou"),
    ) && candidate_depth > best_depth + FINAL_YAW_DEPTH_REGRESSION_EPSILON_M
        && candidate_bbox + 0.03 < best_bbox
    {
        return Some("depth_bbox_regression_vs_measured_best");
    }
    None
}

fn final_yaw_rendered_context_selection_may_override_metric_guard(
    object: &Value,
    candidate: &Value,
    confidence: f32,
    confidence_threshold: f32,
) -> bool {
    if confidence < confidence_threshold {
        return false;
    }
    if !final_yaw_candidate_has_render_evidence(candidate) {
        return false;
    }
    final_yaw_object_is_contextual_metric_weak(object)
}

fn final_yaw_object_is_contextual_metric_weak(object: &Value) -> bool {
    let descriptor = ["object_id", "label", "asset_id", "instance_id"]
        .iter()
        .filter_map(|key| object.get(*key).and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase();
    descriptor.contains("sofa")
        || descriptor.contains("couch")
        || descriptor.contains("sectional")
        || descriptor.contains("loveseat")
        || descriptor.contains("settee")
        || descriptor.contains("table")
        || descriptor.contains("desk")
        || descriptor.contains("counter")
}

fn final_yaw_refinement_candidate_passes_gate(
    mode: SceneFinalYawRefinementMode,
    object: &Value,
    candidate: &Value,
) -> Result<(), &'static str> {
    let current_yaw = object
        .get("current_yaw_degrees")
        .and_then(Value::as_f64)
        .unwrap_or(0.0) as f32;
    let candidate_yaw = candidate
        .get("candidate_yaw_degrees")
        .and_then(Value::as_f64)
        .unwrap_or(current_yaw as f64) as f32;
    if normalize_degrees(candidate_yaw - current_yaw).abs() < 0.5 {
        return Ok(());
    }
    let baseline = object.get("baseline").unwrap_or(&Value::Null);
    if candidate.get("geometry_metrics").is_none()
        && mode != SceneFinalYawRefinementMode::AlwaysGpt
        && !final_yaw_candidate_has_render_evidence(candidate)
    {
        return Err("candidate_missing_geometry_metrics");
    }
    let baseline_bbox = final_yaw_metric(baseline, "bbox_iou").unwrap_or(0.0);
    let candidate_bbox = final_yaw_metric(candidate, "bbox_iou").unwrap_or(baseline_bbox);
    if candidate_bbox + 0.03 < baseline_bbox {
        return Err("bbox_iou_regression");
    }
    let baseline_mask = final_yaw_metric(baseline, "mask_iou").unwrap_or(0.0);
    let candidate_mask = final_yaw_metric(candidate, "mask_iou").unwrap_or(baseline_mask);
    if candidate_mask + 0.05 < baseline_mask {
        return Err("mask_iou_regression");
    }
    let baseline_depth = final_yaw_metric(baseline, "depth_error_m");
    let candidate_depth = final_yaw_metric(candidate, "depth_error_m");
    let edge_crop = object
        .get("source_bbox")
        .and_then(json_array4)
        .map(final_yaw_bbox_touches_image_edge)
        .unwrap_or(false);
    let large_seating = object
        .get("label")
        .and_then(Value::as_str)
        .map(|label| {
            let label = label.to_ascii_lowercase();
            label.contains("sofa") || label.contains("couch") || label.contains("sectional")
        })
        .unwrap_or(false);
    let improved_2d =
        candidate_bbox > baseline_bbox + 0.04 || candidate_mask > baseline_mask + 0.04;
    if let (Some(baseline_depth), Some(candidate_depth)) = (baseline_depth, candidate_depth)
        && candidate_depth > baseline_depth + 0.35
        && !(edge_crop && large_seating && improved_2d)
    {
        return Err("depth_error_regression");
    }
    Ok(())
}

fn final_yaw_candidate_has_render_evidence(candidate: &Value) -> bool {
    [
        "rendered_candidate_full_scene",
        "rendered_candidate_full_frame",
        "rendered_candidate_object_only_full_frame",
    ]
    .iter()
    .any(|key| {
        candidate
            .get(*key)
            .and_then(Value::as_str)
            .is_some_and(|path| !path.trim().is_empty())
    })
}

#[derive(Clone, Debug)]
struct FinalYawPoseApplication {
    translation: [f32; 3],
    scale: [f32; 3],
    reason: &'static str,
}

fn final_yaw_pose_apply_for_candidate(
    object: &Value,
    candidate: &Value,
) -> Option<FinalYawPoseApplication> {
    let metrics = candidate
        .get("geometry_metrics")
        .filter(|value| value.is_object())
        .unwrap_or(candidate);
    let translation = metrics.get("translation").and_then(json_array3)?;
    let scale = metrics.get("scale").and_then(json_array3)?;
    let current_translation = object.get("translation").and_then(json_array3)?;
    let current_scale = object.get("scale").and_then(json_array3)?;
    if !final_yaw_pose_translation_delta_is_bounded(current_translation, translation) {
        return None;
    }
    if !final_yaw_pose_scale_ratio_is_bounded(current_scale, scale) {
        return None;
    }
    Some(FinalYawPoseApplication {
        translation,
        scale,
        reason: "selected_candidate_pose_metrics",
    })
}

fn final_yaw_pose_translation_delta_is_bounded(current: [f32; 3], candidate: [f32; 3]) -> bool {
    let dx = candidate[0] - current[0];
    let dz = candidate[2] - current[2];
    (dx * dx + dz * dz).sqrt() <= FINAL_YAW_POSE_TRANSLATION_MAX_DELTA_M
}

fn final_yaw_pose_scale_ratio_is_bounded(current: [f32; 3], candidate: [f32; 3]) -> bool {
    current
        .iter()
        .zip(candidate.iter())
        .all(|(current, candidate)| {
            let current = current.abs().max(1.0e-5);
            let ratio = candidate.abs() / current;
            (FINAL_YAW_POSE_SCALE_RATIO_MIN..=FINAL_YAW_POSE_SCALE_RATIO_MAX).contains(&ratio)
        })
}

fn final_yaw_metric(value: &Value, key: &str) -> Option<f32> {
    value
        .get(key)
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
        .map(|value| value as f32)
        .or_else(|| {
            value
                .get("geometry_metrics")
                .and_then(|metrics| metrics.get(key))
                .and_then(Value::as_f64)
                .filter(|value| value.is_finite())
                .map(|value| value as f32)
        })
}

fn final_yaw_bbox_touches_image_edge(bbox: [f32; 4]) -> bool {
    bbox[0] <= 0.03 || bbox[1] <= 0.03 || bbox[2] >= 0.97 || bbox[3] >= 0.97
}

fn final_yaw_refinement_report(
    config: SceneFinalYawRefinementConfig<'_>,
    manifest: &SceneObjectManifest,
    objects: Vec<Value>,
    applied: Vec<Value>,
    ignored: Vec<Value>,
    status: &str,
) -> Value {
    let selection_task = final_yaw_refinement_selection_task(&manifest.source_scene_path, &objects);
    json!({
        "schema_version": 1,
        "stage": "final_context_yaw_refinement",
        "status": status,
        "mode": config.mode,
        "object_set": config.object_set,
        "confidence_threshold": config.confidence_threshold,
        "max_candidates": config.max_candidates,
        "source_scene_path": manifest.source_scene_path,
        "algorithm": "bounded-full-scene-context-measured-pose-candidate-selection",
        "constraint": "select_one_measured_candidate_no_freeform_transform",
        "object_count": objects.len(),
        "applied_count": applied.len(),
        "ignored_count": ignored.len(),
        "objects": objects,
        "applied": applied,
        "ignored": ignored,
        "selection_task": selection_task,
    })
}

fn final_yaw_refinement_selection_task(source_scene_path: &str, objects: &[Value]) -> Value {
    let objects = objects
        .iter()
        .filter_map(|object| {
            let rotation_selection = object.get("rotation_selection")?;
            Some(json!({
                "index": object.get("index").cloned().unwrap_or(Value::Null),
                "placement_index": object
                    .get("placement_index")
                    .cloned()
                    .unwrap_or(Value::Null),
                "command_index": object.get("command_index").cloned().unwrap_or(Value::Null),
                "object_id": object.get("object_id").cloned().unwrap_or(Value::Null),
                "instance_id": object.get("instance_id").cloned().unwrap_or(Value::Null),
                "label": object.get("label").cloned().unwrap_or(Value::Null),
                "source_crop": object.get("source_crop").cloned().unwrap_or(Value::Null),
                "source_mask": object.get("source_mask").cloned().unwrap_or(Value::Null),
                "source_mask_path": object
                    .get("source_mask_path")
                    .cloned()
                    .unwrap_or(Value::Null),
                "source_depth_stats": object
                    .get("source_depth_stats")
                    .cloned()
                    .unwrap_or(Value::Null),
                "source_bbox": object.get("source_bbox").cloned().unwrap_or(Value::Null),
                "current_yaw_degrees": object
                    .get("current_yaw_degrees")
                    .cloned()
                    .unwrap_or(Value::Null),
                "rotation_selection": rotation_selection.clone(),
                "instruction": "Select candidate_index only. Compare source crop and full-scene candidate renders when attached; do not invent transforms outside the listed candidates.",
            }))
        })
        .collect::<Vec<_>>();
    if objects.is_empty() {
        return Value::Null;
    }
    json!({
        "purpose": "final-context-object-measured-pose-selection",
        "source_scene_path": source_scene_path,
        "instruction": "Choose one listed pose candidate per object. Use the original source scene, source-reference overlay, object source mask, and mask-tight crop as primary contextual evidence, then compare full-scene and target-only full-frame rendered candidates. Rendered cardinal probes without geometry metrics are valid choices when they visibly align object orientation better than the measured metric-best candidate. For large sofas/sectionals, prioritize visible curve direction, open side, and table relation over bbox-only coverage when those disagree. Return candidate_index values only; do not invent positions, scale, or yaw values.",
        "schema": {
            "type": "object",
            "properties": {
                "objects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": { "type": "integer" },
                            "candidate_index": { "type": "integer" },
                            "confidence": { "type": "number" },
                            "rationale": { "type": "string" }
                        },
                        "required": ["index", "candidate_index", "confidence", "rationale"],
                        "additionalProperties": false
                    }
                }
            },
            "required": ["objects"],
            "additionalProperties": false
        },
        "objects": objects,
    })
}

fn write_final_yaw_refinement_artifacts_if_requested(
    config: SceneFinalYawRefinementConfig<'_>,
    report: &Value,
) -> Result<(), String> {
    if !config.write_artifacts {
        return Ok(());
    }
    fs::create_dir_all(config.output_dir).map_err(|err| {
        format!(
            "failed to create final yaw refinement output dir {}: {err}",
            config.output_dir.display()
        )
    })?;
    write_json_file(
        &config.output_dir.join("final_yaw_refinement_report.json"),
        report,
    )
    .map_err(|err| err.to_string())?;
    if let Some(task) = report.get("selection_task")
        && !task.is_null()
    {
        write_json_file(&config.output_dir.join("selection_task.json"), task)
            .map_err(|err| err.to_string())?;
    }
    Ok(())
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

    fn final_yaw_test_manifest() -> SceneObjectManifest {
        SceneObjectManifest {
            source_scene_path: "scene.jpg".to_string(),
            scene_calibration: None,
            objects: Vec::new(),
        }
    }

    fn final_yaw_test_layout() -> GroundedSceneLayout {
        let aabb = crate::SceneAssetAabb {
            min: [-1.0, 0.0, -0.5],
            max: [1.0, 1.0, 0.5],
        };
        GroundedSceneLayout {
            bsn: String::new(),
            placements: vec![GroundedScenePlacement {
                entity_id: "sofa_entity".to_string(),
                asset_id: "sofa_asset".to_string(),
                object_id: "sofa".to_string(),
                instance_id: None,
                label: "tan couch".to_string(),
                source_bbox: [0.02, 0.25, 0.92, 0.78],
                contact_pixel: [0.45, 0.78],
                ground_point: [1.0, 0.0, 2.0],
                translation: [1.0, 0.0, 2.0],
                rotation_y_degrees: 180.0,
                asset_yaw_offset_degrees: 0.0,
                scale: [2.0, 2.0, 2.0],
                local_aabb: aabb,
                target_footprint_m: [3.0, 1.5],
            }],
            camera: crate::SceneCamera {
                translation: [0.0, 2.0, -4.0],
                focus: [0.0, 0.0, 0.0],
                yaw: None,
                pitch: None,
                radius: None,
                vertical_fov_degrees: Some(55.0),
            },
            rug_center: [0.0, 0.0, 0.0],
            rug_scale: [1.0, 1.0, 1.0],
            projection_fit: None,
        }
    }

    fn final_yaw_test_binding() -> Vec<SceneAssetBinding> {
        vec![SceneAssetBinding {
            asset_id: "sofa_asset".to_string(),
            object_id: "sofa".to_string(),
            label: "tan couch".to_string(),
            aliases: Vec::new(),
            path: Some("sofa.glb".to_string()),
            cache_key: Some("sofa_cache".to_string()),
            reusable: true,
            source_image_path: None,
            pipeline: None,
            local_aabb: None,
            canonical_frame: None,
            provenance: Some(crate::SceneAssetProvenance {
                run_id: "test".to_string(),
                source_scene_path: "scene.jpg".to_string(),
                source_object_id: "sofa".to_string(),
                source_crop_path: Some("sofa_crop.png".to_string()),
                generated_by: "test".to_string(),
            }),
        }]
    }

    fn final_yaw_test_commands() -> Vec<Value> {
        vec![
            json!({ "type": "clear_scene" }),
            json!({
                "type": "spawn_cached",
                "cache_key": "sofa_cache",
                "translation": [1.0, 0.0, 2.0],
                "rotation": quat_from_y_degrees(180.0),
                "scale": [2.0, 2.0, 2.0],
            }),
            json!({ "type": "set_camera" }),
        ]
    }

    fn final_yaw_prior_report() -> Value {
        json!({
            "objects": [{
                "object_id": "sofa",
                "instance_id": null,
                "applied": false,
                "baseline": {
                    "yaw_degrees": 180.0,
                    "mask_iou": 0.14,
                    "bbox_iou": 0.29,
                    "depth_error_m": 1.23,
                    "loss": 7.7
                },
                "best": {
                    "yaw_degrees": -6.0,
                    "mask_iou": 0.38,
                    "bbox_iou": 0.42,
                    "depth_error_m": 1.30,
                    "loss": 6.2
                },
                "candidates": [{
                    "candidate_index": 0,
                    "yaw_degrees": 180.0,
                    "mask_iou": 0.14,
                    "bbox_iou": 0.29,
                    "depth_error_m": 1.23,
                    "loss": 7.7
                }, {
                    "candidate_index": 1,
                    "yaw_degrees": -6.0,
                    "mask_iou": 0.38,
                    "bbox_iou": 0.42,
                    "depth_error_m": 1.30,
                    "loss": 6.2
                }]
            }]
        })
    }

    fn final_yaw_config(output_dir: &Path) -> SceneFinalYawRefinementConfig<'_> {
        SceneFinalYawRefinementConfig {
            mode: SceneFinalYawRefinementMode::GatedGpt,
            object_set: SceneObjectPoseRefinementSet::TablesAndLargeSeating,
            confidence_threshold: 0.70,
            max_candidates: 12,
            write_artifacts: false,
            output_dir,
            grounding_evidence: None,
            rendered_selection_task: None,
        }
    }

    fn final_yaw_config_with_mode(
        output_dir: &Path,
        mode: SceneFinalYawRefinementMode,
    ) -> SceneFinalYawRefinementConfig<'_> {
        SceneFinalYawRefinementConfig {
            mode,
            ..final_yaw_config(output_dir)
        }
    }

    fn final_yaw_test_grounding_evidence() -> SceneGroundingEvidence {
        SceneGroundingEvidence {
            source_image_path: "scene.jpg".to_string(),
            depth: None,
            segmentation: None,
            detections: Vec::new(),
            camera: crate::EstimatedCamera::default(),
            floor: crate::EstimatedFloorPlane::default(),
            objects: vec![ObjectGroundingEvidence {
                object_id: "sofa".to_string(),
                instance_id: None,
                reuse_group: None,
                detection: None,
                mask: Some(ObjectMaskEvidence {
                    provider: "sam2".to_string(),
                    model: "sam2.1-large".to_string(),
                    bbox: [0.04, 0.24, 0.90, 0.80],
                    score: 0.94,
                    area_px: 42_000,
                    image_size: [1024, 576],
                    mask_rle: vec![1, 2, 3, 4],
                    center_pixel: Some([0.46, 0.54]),
                    contact_pixel: Some([0.45, 0.78]),
                    coverage: Some(0.31),
                    artifact_path: Some("masks.json".to_string()),
                    mask_png_path: Some("sofa_mask.png".to_string()),
                }),
                asset_id: Some("sofa_asset".to_string()),
                contact_pixel: Some([0.45, 0.78]),
                depth_stats: Some(crate::ObjectDepthStats {
                    median_m: 2.4,
                    min_m: 1.5,
                    max_m: 3.1,
                    contact_m: Some(2.7),
                    sample_count: Some(3200),
                }),
                candidate_floor_contact_rays: Vec::new(),
                metric_contact_point_m: None,
                target_footprint_m: Some([3.0, 1.5]),
                provenance: vec!["test".to_string()],
            }],
        }
    }

    fn final_yaw_config_with_grounding<'a>(
        output_dir: &'a Path,
        grounding_evidence: &'a SceneGroundingEvidence,
    ) -> SceneFinalYawRefinementConfig<'a> {
        SceneFinalYawRefinementConfig {
            grounding_evidence: Some(grounding_evidence),
            ..final_yaw_config(output_dir)
        }
    }

    #[test]
    fn final_yaw_refinement_applies_yaw_only_from_bounded_selection() {
        let manifest = final_yaw_test_manifest();
        let bindings = final_yaw_test_binding();
        let layout = final_yaw_test_layout();
        let commands = final_yaw_test_commands();
        let prior = final_yaw_prior_report();
        let prepared = apply_scene_final_yaw_refinement(
            final_yaw_config(Path::new("tmp")),
            &manifest,
            &bindings,
            &layout,
            &commands,
            Some(&prior),
            None,
        )
        .expect("prepare final yaw");
        let target_candidate = prepared.report["objects"][0]["rotation_selection"]["candidates"]
            .as_array()
            .unwrap()
            .iter()
            .find(|candidate| {
                candidate["candidate_yaw_degrees"]
                    .as_f64()
                    .is_some_and(|yaw| (yaw + 6.0).abs() < 0.5)
            })
            .and_then(|candidate| candidate["candidate_index"].as_u64())
            .expect("best yaw candidate") as usize;
        let response = SceneRotationSelectionResponse {
            objects: vec![crate::SceneRotationSelection {
                index: 0,
                candidate_index: target_candidate,
                confidence: 0.92,
                rationale: "full-scene context aligns the couch better".to_string(),
            }],
        };
        let applied = apply_scene_final_yaw_refinement(
            final_yaw_config(Path::new("tmp")),
            &manifest,
            &bindings,
            &layout,
            &commands,
            Some(&prior),
            Some(&response),
        )
        .expect("apply final yaw");

        assert_eq!(applied.report["status"], json!("applied"));
        assert_eq!(
            applied.commands[1]["translation"],
            commands[1]["translation"]
        );
        assert_eq!(applied.commands[1]["scale"], commands[1]["scale"]);
        assert!((applied.grounded_layout.placements[0].rotation_y_degrees + 6.0).abs() < 0.5);
    }

    #[test]
    fn final_yaw_refinement_applies_measured_pose_candidate_atomically() {
        let manifest = final_yaw_test_manifest();
        let bindings = final_yaw_test_binding();
        let layout = final_yaw_test_layout();
        let commands = final_yaw_test_commands();
        let mut prior = final_yaw_prior_report();
        prior["objects"][0]["best"]["translation"] = json!([1.18, 0.25, 1.62]);
        prior["objects"][0]["best"]["scale"] = json!([2.7, 2.7, 2.7]);
        prior["objects"][0]["candidates"][1]["translation"] = json!([1.18, 0.25, 1.62]);
        prior["objects"][0]["candidates"][1]["scale"] = json!([2.7, 2.7, 2.7]);
        let prepared = apply_scene_final_yaw_refinement(
            final_yaw_config(Path::new("tmp")),
            &manifest,
            &bindings,
            &layout,
            &commands,
            Some(&prior),
            None,
        )
        .expect("prepare final pose");
        let target_candidate = prepared.report["objects"][0]["rotation_selection"]["candidates"]
            .as_array()
            .unwrap()
            .iter()
            .find(|candidate| {
                candidate["candidate_yaw_degrees"]
                    .as_f64()
                    .is_some_and(|yaw| (yaw + 6.0).abs() < 0.5)
            })
            .and_then(|candidate| candidate["candidate_index"].as_u64())
            .expect("measured pose candidate") as usize;
        let response = SceneRotationSelectionResponse {
            objects: vec![crate::SceneRotationSelection {
                index: 0,
                candidate_index: target_candidate,
                confidence: 0.92,
                rationale: "measured pose candidate best matches source".to_string(),
            }],
        };
        let applied = apply_scene_final_yaw_refinement(
            final_yaw_config(Path::new("tmp")),
            &manifest,
            &bindings,
            &layout,
            &commands,
            Some(&prior),
            Some(&response),
        )
        .expect("apply final pose");

        assert_eq!(applied.report["status"], json!("applied"));
        let command_translation = json_array3(&applied.commands[1]["translation"]).unwrap();
        let command_scale = json_array3(&applied.commands[1]["scale"]).unwrap();
        assert!(
            command_translation
                .iter()
                .zip([1.18, 0.25, 1.62])
                .all(|(left, right)| (*left - right).abs() < 1.0e-5)
        );
        assert!(
            command_scale
                .iter()
                .zip([2.7, 2.7, 2.7])
                .all(|(left, right)| (*left - right).abs() < 1.0e-5)
        );
        assert_eq!(
            applied.report["objects"][0]["selected_pose_application"]["applied"],
            json!(true)
        );
        assert_eq!(
            applied.report["applied"][0]["pose_application"]["reason"],
            json!("selected_candidate_pose_metrics")
        );
        assert!((applied.grounded_layout.placements[0].rotation_y_degrees + 6.0).abs() < 0.5);
        assert_eq!(
            applied.grounded_layout.placements[0].translation,
            [1.18, 0.25, 1.62]
        );
        assert_eq!(applied.grounded_layout.placements[0].scale, [2.7, 2.7, 2.7]);
    }

    #[test]
    fn final_yaw_current_candidate_carries_matching_baseline_metrics() {
        let manifest = final_yaw_test_manifest();
        let bindings = final_yaw_test_binding();
        let layout = final_yaw_test_layout();
        let commands = final_yaw_test_commands();
        let prior = final_yaw_prior_report();
        let prepared = apply_scene_final_yaw_refinement(
            final_yaw_config(Path::new("tmp")),
            &manifest,
            &bindings,
            &layout,
            &commands,
            Some(&prior),
            None,
        )
        .expect("prepare final yaw");

        let current = &prepared.report["objects"][0]["rotation_selection"]["candidates"][0];
        assert_eq!(current["source"], json!("current"));
        assert_eq!(current["metrics_source"], json!("baseline"));
        assert_eq!(current["bbox_iou"], json!(0.29));
        assert_eq!(current["loss"], json!(7.7));
    }

    #[test]
    fn final_yaw_selection_task_includes_source_mask_evidence() {
        let manifest = final_yaw_test_manifest();
        let bindings = final_yaw_test_binding();
        let layout = final_yaw_test_layout();
        let commands = final_yaw_test_commands();
        let prior = final_yaw_prior_report();
        let grounding = final_yaw_test_grounding_evidence();
        let prepared = apply_scene_final_yaw_refinement(
            final_yaw_config_with_grounding(Path::new("tmp"), &grounding),
            &manifest,
            &bindings,
            &layout,
            &commands,
            Some(&prior),
            None,
        )
        .expect("prepare final yaw with grounding");

        let object = &prepared.report["objects"][0];
        assert_eq!(object["source_mask_path"], json!("sofa_mask.png"));
        let mask_bbox = json_array4(&object["source_mask"]["bbox"]).unwrap();
        assert!((mask_bbox[0] - 0.04).abs() < 1.0e-5);
        assert!((mask_bbox[3] - 0.80).abs() < 1.0e-5);
        assert!(object["source_mask"]["mask_rle"].is_null());
        assert!((object["source_depth_stats"]["median_m"].as_f64().unwrap() - 2.4).abs() < 1.0e-5);

        let task_object = &prepared.report["selection_task"]["objects"][0];
        assert_eq!(task_object["source_mask_path"], json!("sofa_mask.png"));
        assert_eq!(
            task_object["source_mask"]["mask_png_path"],
            json!("sofa_mask.png")
        );
        assert!(task_object["source_mask"]["mask_rle"].is_null());
    }

    #[test]
    fn final_yaw_gated_mode_overrides_worse_gpt_current_selection() {
        let manifest = final_yaw_test_manifest();
        let bindings = final_yaw_test_binding();
        let layout = final_yaw_test_layout();
        let commands = final_yaw_test_commands();
        let prior = final_yaw_prior_report();
        let response = SceneRotationSelectionResponse {
            objects: vec![crate::SceneRotationSelection {
                index: 0,
                candidate_index: 0,
                confidence: 0.99,
                rationale: "semantic guess incorrectly keeps current".to_string(),
            }],
        };
        let applied = apply_scene_final_yaw_refinement(
            final_yaw_config(Path::new("tmp")),
            &manifest,
            &bindings,
            &layout,
            &commands,
            Some(&prior),
            Some(&response),
        )
        .expect("apply final yaw");

        assert_eq!(applied.report["status"], json!("applied"));
        assert!((applied.grounded_layout.placements[0].rotation_y_degrees + 6.0).abs() < 0.5);
        assert_eq!(
            applied.report["applied"][0]["selection_policy"],
            json!("gpt_overridden_by_metric_gate")
        );
        assert_eq!(
            applied.report["applied"][0]["gpt_candidate_index"],
            json!(0)
        );
        assert_ne!(
            applied.report["applied"][0]["accepted_candidate_index"],
            json!(0)
        );
    }

    #[test]
    fn final_yaw_gated_mode_keeps_high_confidence_rendered_context_selection() {
        let manifest = final_yaw_test_manifest();
        let bindings = final_yaw_test_binding();
        let layout = final_yaw_test_layout();
        let commands = final_yaw_test_commands();
        let prior = final_yaw_prior_report();
        let prepared = apply_scene_final_yaw_refinement(
            final_yaw_config(Path::new("tmp")),
            &manifest,
            &bindings,
            &layout,
            &commands,
            Some(&prior),
            None,
        )
        .expect("prepare final yaw");
        let mut rendered_task = prepared.report["selection_task"].clone();
        let current_candidate = rendered_task["objects"][0]["rotation_selection"]["candidates"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|candidate| candidate["candidate_index"].as_u64() == Some(0))
            .unwrap();
        current_candidate["rendered_candidate_full_scene"] =
            json!("/tmp/sofa_candidate_current_full_scene.png");
        current_candidate["rendered_candidate_object_only_full_frame"] =
            json!("/tmp/sofa_candidate_current_object_only.png");
        let response = SceneRotationSelectionResponse {
            objects: vec![crate::SceneRotationSelection {
                index: 0,
                candidate_index: 0,
                confidence: 0.74,
                rationale: "rendered context shows the sofa curve is semantically correct"
                    .to_string(),
            }],
        };
        let applied = apply_scene_final_yaw_refinement(
            SceneFinalYawRefinementConfig {
                rendered_selection_task: Some(&rendered_task),
                ..final_yaw_config(Path::new("tmp"))
            },
            &manifest,
            &bindings,
            &layout,
            &commands,
            Some(&prior),
            Some(&response),
        )
        .expect("apply final yaw");

        assert_eq!(applied.report["status"], json!("applied"));
        assert!((applied.grounded_layout.placements[0].rotation_y_degrees - 180.0).abs() < 0.5);
        assert_eq!(
            applied.report["applied"][0]["selection_policy"],
            json!("gpt_rendered_context_selection")
        );
        assert_eq!(
            applied.report["applied"][0]["accepted_candidate_index"],
            json!(0)
        );
    }

    #[test]
    fn final_yaw_always_gpt_keeps_explicit_current_selection() {
        let manifest = final_yaw_test_manifest();
        let bindings = final_yaw_test_binding();
        let layout = final_yaw_test_layout();
        let commands = final_yaw_test_commands();
        let prior = final_yaw_prior_report();
        let response = SceneRotationSelectionResponse {
            objects: vec![crate::SceneRotationSelection {
                index: 0,
                candidate_index: 0,
                confidence: 0.99,
                rationale: "always-gpt intentionally keeps current".to_string(),
            }],
        };
        let applied = apply_scene_final_yaw_refinement(
            final_yaw_config_with_mode(Path::new("tmp"), SceneFinalYawRefinementMode::AlwaysGpt),
            &manifest,
            &bindings,
            &layout,
            &commands,
            Some(&prior),
            Some(&response),
        )
        .expect("apply final yaw");

        assert_eq!(applied.report["status"], json!("applied"));
        assert!((applied.grounded_layout.placements[0].rotation_y_degrees - 180.0).abs() < 0.5);
        assert_eq!(
            applied.report["applied"][0]["selection_policy"],
            json!("gpt_selection")
        );
    }

    #[test]
    fn final_yaw_gated_mode_accepts_rendered_unmeasured_candidate() {
        let manifest = final_yaw_test_manifest();
        let bindings = final_yaw_test_binding();
        let layout = final_yaw_test_layout();
        let commands = final_yaw_test_commands();
        let prior = final_yaw_prior_report();
        let prepared = apply_scene_final_yaw_refinement(
            final_yaw_config(Path::new("tmp")),
            &manifest,
            &bindings,
            &layout,
            &commands,
            Some(&prior),
            None,
        )
        .expect("prepare final yaw");
        let unmeasured_candidate = prepared.report["objects"][0]["rotation_selection"]["candidates"]
            .as_array()
            .unwrap()
            .iter()
            .find(|candidate| {
                candidate.get("geometry_metrics").is_none()
                    && candidate["yaw_delta_degrees"]
                        .as_f64()
                        .is_some_and(|delta| (delta - 45.0).abs() < 1.0e-5)
            })
            .and_then(|candidate| candidate["candidate_index"].as_u64())
            .expect("rendered cardinal probe candidate")
            as usize;
        let mut rendered_task = prepared.report["selection_task"].clone();
        let candidates = rendered_task["objects"][0]["rotation_selection"]["candidates"]
            .as_array_mut()
            .unwrap();
        let rendered_candidate = candidates
            .iter_mut()
            .find(|candidate| {
                candidate["candidate_index"].as_u64() == Some(unmeasured_candidate as u64)
            })
            .unwrap();
        rendered_candidate["rendered_candidate_full_scene"] =
            json!("/tmp/sofa_candidate_45_full_scene.png");
        rendered_candidate["rendered_candidate_object_only_full_frame"] =
            json!("/tmp/sofa_candidate_45_object_only.png");
        let response = SceneRotationSelectionResponse {
            objects: vec![crate::SceneRotationSelection {
                index: 0,
                candidate_index: unmeasured_candidate,
                confidence: 0.91,
                rationale: "rendered cardinal probe matches couch curve direction".to_string(),
            }],
        };
        let applied = apply_scene_final_yaw_refinement(
            SceneFinalYawRefinementConfig {
                rendered_selection_task: Some(&rendered_task),
                ..final_yaw_config(Path::new("tmp"))
            },
            &manifest,
            &bindings,
            &layout,
            &commands,
            Some(&prior),
            Some(&response),
        )
        .expect("apply rendered unmeasured final yaw");

        assert_eq!(applied.report["status"], json!("applied"));
        assert_eq!(
            applied.report["applied"][0]["selection_policy"],
            json!("gpt_selection_metric_compatible")
        );
        assert_eq!(
            applied.report["applied"][0]["accepted_candidate_index"],
            json!(unmeasured_candidate)
        );
        assert!(applied.report["objects"][0]["selected_pose_application"].is_null());
        assert!((applied.grounded_layout.placements[0].rotation_y_degrees - -135.0).abs() < 0.5);
    }

    #[test]
    fn final_yaw_gated_mode_accepts_rendered_fine_candidate_appended_to_task() {
        let manifest = final_yaw_test_manifest();
        let bindings = final_yaw_test_binding();
        let layout = final_yaw_test_layout();
        let commands = final_yaw_test_commands();
        let prior = final_yaw_prior_report();
        let prepared = apply_scene_final_yaw_refinement(
            final_yaw_config(Path::new("tmp")),
            &manifest,
            &bindings,
            &layout,
            &commands,
            Some(&prior),
            None,
        )
        .expect("prepare final yaw");
        let mut rendered_task = prepared.report["selection_task"].clone();
        rendered_task["objects"][0]["rotation_selection"]["candidates"]
            .as_array_mut()
            .unwrap()
            .push(json!({
                "candidate_index": 99,
                "candidate_yaw_degrees": -70.0,
                "parent_candidate_index": 2,
                "source": "fine_probe",
                "rendered_candidate_full_scene": "/tmp/sofa_candidate_fine_full_scene.png",
                "rendered_candidate_object_only_full_frame": "/tmp/sofa_candidate_fine_object_only.png",
                "final_yaw_visual_fit": {
                    "score": 0.91,
                    "target_silhouette_score": 0.87
                }
            }));
        let response = SceneRotationSelectionResponse {
            objects: vec![crate::SceneRotationSelection {
                index: 0,
                candidate_index: 99,
                confidence: 0.88,
                rationale: "fine probe best matches source sofa silhouette".to_string(),
            }],
        };
        let applied = apply_scene_final_yaw_refinement(
            SceneFinalYawRefinementConfig {
                rendered_selection_task: Some(&rendered_task),
                ..final_yaw_config(Path::new("tmp"))
            },
            &manifest,
            &bindings,
            &layout,
            &commands,
            Some(&prior),
            Some(&response),
        )
        .expect("apply rendered fine candidate final yaw");

        assert_eq!(applied.report["status"], json!("applied"));
        assert_eq!(
            applied.report["applied"][0]["accepted_candidate_index"],
            json!(99)
        );
        assert!((applied.grounded_layout.placements[0].rotation_y_degrees + 70.0).abs() < 0.5);
        assert!(
            applied.report["objects"][0]["rotation_selection"]["candidates"]
                .as_array()
                .unwrap()
                .iter()
                .any(|candidate| candidate["candidate_index"] == json!(99)
                    && candidate["selected"] == json!(true))
        );
    }

    #[test]
    fn final_yaw_refinement_rejects_unmeasured_gated_candidate() {
        let manifest = final_yaw_test_manifest();
        let bindings = final_yaw_test_binding();
        let layout = final_yaw_test_layout();
        let commands = final_yaw_test_commands();
        let prior = final_yaw_prior_report();
        let prepared = apply_scene_final_yaw_refinement(
            final_yaw_config(Path::new("tmp")),
            &manifest,
            &bindings,
            &layout,
            &commands,
            Some(&prior),
            None,
        )
        .expect("prepare final yaw");
        let unmeasured_candidate = prepared.report["objects"][0]["rotation_selection"]["candidates"]
            .as_array()
            .unwrap()
            .iter()
            .find(|candidate| {
                candidate.get("geometry_metrics").is_none()
                    && candidate["yaw_delta_degrees"]
                        .as_f64()
                        .is_some_and(|delta| delta.abs() > 1.0)
            })
            .and_then(|candidate| candidate["candidate_index"].as_u64())
            .expect("unmeasured candidate") as usize;
        let response = SceneRotationSelectionResponse {
            objects: vec![crate::SceneRotationSelection {
                index: 0,
                candidate_index: unmeasured_candidate,
                confidence: 0.99,
                rationale: "bad free candidate".to_string(),
            }],
        };
        let applied = apply_scene_final_yaw_refinement(
            final_yaw_config(Path::new("tmp")),
            &manifest,
            &bindings,
            &layout,
            &commands,
            Some(&prior),
            Some(&response),
        )
        .expect("reject final yaw");

        assert_eq!(
            applied.report["status"],
            json!("selection_rejected_or_noop")
        );
        assert_eq!(
            applied.report["ignored"][0]["reason"],
            json!("candidate_missing_geometry_metrics")
        );
        assert_eq!(applied.commands[1]["rotation"], commands[1]["rotation"]);
    }

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

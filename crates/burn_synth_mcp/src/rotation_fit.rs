use std::fmt::Write as _;

use crate::prelude::*;

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ScenePoseFitObjectFilter {
    All,
    TablesOnly,
}

impl ScenePoseFitObjectFilter {
    fn includes(self, placement: &GroundedScenePlacement) -> bool {
        match self {
            Self::All => true,
            Self::TablesOnly => placement_is_table_like(placement),
        }
    }

    fn stage_name(self) -> &'static str {
        match self {
            Self::All => "visible_surface_pose_fit",
            Self::TablesOnly => "table_pose_refinement",
        }
    }

    fn report_file_stem(self) -> &'static str {
        match self {
            Self::All => "visible_surface_pose_fit",
            Self::TablesOnly => "table_pose_refinement",
        }
    }

    fn output_dir_name(self) -> &'static str {
        match self {
            Self::All => "visible_surface_pose_candidates",
            Self::TablesOnly => "table_pose_candidates",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::TablesOnly => "tables-only",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SceneVisibleSurfacePoseFitConfig<'a> {
    pub(crate) mode: ScenePoseFitMode,
    pub(crate) min_mask_iou: f32,
    pub(crate) max_depth_error_m: f32,
    pub(crate) write_artifacts: bool,
    pub(crate) output_dir: &'a Path,
    pub(crate) scale_policy: SceneScalePolicy,
    pub(crate) object_filter: ScenePoseFitObjectFilter,
}

#[derive(Clone)]
struct RotationFitTarget {
    mask: BinaryMask,
    bbox: [f32; 4],
    crop_bbox: [f32; 4],
    depth_median_m: Option<f32>,
    depth_stats: Option<burn_synth_scene::ObjectDepthStats>,
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

pub(crate) fn apply_scene_visible_surface_pose_fit(
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

pub(crate) fn apply_scene_table_pose_refinement(
    mode: SceneTablePoseRefinementMode,
    mut config: SceneVisibleSurfacePoseFitConfig<'_>,
    manifest: &SceneObjectManifest,
    asset_bindings: &[SceneAssetBinding],
    evidence: Option<&SceneGroundingEvidence>,
    grounded_layout: &GroundedSceneLayout,
    commands: &[Value],
) -> Result<SceneRotationFitOutcome, String> {
    config.object_filter = ScenePoseFitObjectFilter::TablesOnly;
    if mode == SceneTablePoseRefinementMode::Off {
        let report =
            visible_surface_pose_fit_report(config, manifest, 0, 0, 0, Vec::new(), "disabled");
        return Ok(SceneRotationFitOutcome {
            commands: commands.to_vec(),
            grounded_layout: grounded_layout.clone(),
            report,
        });
    }
    let mut outcome = apply_scene_visible_surface_pose_fit(
        config,
        manifest,
        asset_bindings,
        evidence,
        grounded_layout,
        commands,
    )?;
    outcome.report["table_pose_refinement_mode"] = json!(mode);
    outcome.report["gpt_gate"] = table_pose_refinement_gpt_gate(mode, &outcome.report);
    if config.write_artifacts {
        write_visible_surface_pose_fit_artifacts_if_requested(config, manifest, &outcome.report)?;
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
fn visible_surface_pose_fit_report(
    config: SceneVisibleSurfacePoseFitConfig<'_>,
    manifest: &SceneObjectManifest,
    applied_count: usize,
    skipped_count: usize,
    warning_count: usize,
    objects: Vec<Value>,
    status: &str,
) -> Value {
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
        "object_count": objects.len(),
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

pub(crate) fn table_pose_refinement_gpt_gate(
    mode: SceneTablePoseRefinementMode,
    report: &Value,
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
        if mode == SceneTablePoseRefinementMode::AlwaysGpt {
            required_objects.push(table_gpt_gate_object_report(object, "mode_always_gpt"));
            continue;
        }
        if object
            .get("applied")
            .and_then(Value::as_bool)
            .unwrap_or(false)
            == false
        {
            required_objects.push(table_gpt_gate_object_report(
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
            required_objects.push(table_gpt_gate_object_report(
                object,
                "low_mask_and_bbox_overlap",
            ));
        } else if off_center {
            required_objects.push(table_gpt_gate_object_report(
                object,
                "projected_center_error",
            ));
        } else if ambiguous_loss {
            required_objects.push(table_gpt_gate_object_report(
                object,
                "ambiguous_geometry_margin",
            ));
        }
    }
    json!({
        "enabled": true,
        "required": !required_objects.is_empty(),
        "mode": mode,
        "objects": required_objects,
        "note": "GPT is only a bounded candidate selector for table fits flagged by objective geometry. It is not a source of absolute transform truth.",
    })
}

fn table_gpt_gate_object_report(object: &Value, reason: &'static str) -> Value {
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
    use super::pose_search::{
        VisibleSurfacePoseCandidate, clamp_visible_surface_pose_candidate,
        rotation_fit_candidate_yaws, semantic_yaw_loss_for_error, semantic_yaw_prior_for_placement,
        visible_surface_pose_candidate_passes_target,
    };
    use super::soft_refinement::{
        DenseSoftSurfaceTransformContext, dense_soft_surface_points,
        dense_soft_surface_refine_pose, soft_camera_intrinsics_for_crop,
    };
    use super::visible_surface::{
        SurfaceDepthSummary, binary_mask_iou, mask_point_moments, mask_point_surface_loss,
        rasterize_projected_triangle, rotation_fit_source_camera_point, surface_depth_loss,
    };
    use super::*;
    use burn_synth_render::SoftRenderConfig;

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

    #[test]
    fn rotation_fit_source_camera_point_matches_projection_fit_z_convention() {
        let evidence = SceneGroundingEvidence {
            source_image_path: "/tmp/source.jpg".to_string(),
            depth: None,
            segmentation: None,
            detections: Vec::new(),
            camera: burn_synth_scene::EstimatedCamera {
                focal_length_px: Some(1000.0),
                principal_point: Some([500.0, 500.0]),
                image_size: Some([1000, 1000]),
                vertical_fov_degrees: Some(60.0),
                confidence: Some(1.0),
            },
            floor: burn_synth_scene::EstimatedFloorPlane {
                normal: [0.0, 1.0, 0.0],
                distance_m: 0.0,
                residual_m: Some(0.01),
                confidence: Some(1.0),
            },
            objects: Vec::new(),
        };
        let point = rotation_fit_source_camera_point(
            [0.25, 0.5, -0.75],
            [1.0, 4.0],
            [1.0, 0.0, 4.0],
            "metric-depth-contact",
            &evidence,
        )
        .expect("camera point");
        assert!((point[0] - 1.25).abs() < 1.0e-6);
        assert!((point[1] + 0.5).abs() < 1.0e-6);
        assert!((point[2] - 4.75).abs() < 1.0e-6);
    }

    #[test]
    fn visible_surface_pose_gate_accepts_improved_bbox_fallback_without_sam_mask() {
        let baseline = VisibleSurfacePoseCandidate {
            candidate_index: 0,
            stage: "baseline",
            placement: test_pose_fit_placement(0.0),
            mask_iou: 0.15,
            bbox_iou: 0.20,
            center_error: 0.2,
            area_log2_error: 1.0,
            aspect_log2_error: 0.5,
            depth_error_m: Some(0.10),
            surface_depth_loss: Some(0.20),
            dense_depth_loss: None,
            dense_depth_mae_m: None,
            dense_depth_sample_count: 0,
            point_surface_loss: 0.10,
            semantic_yaw_prior_degrees: None,
            semantic_yaw_error_degrees: None,
            semantic_yaw_loss: 0.0,
            candidate_depth_summary: Some(test_surface_depth_summary(2.1)),
            target_depth_summary: Some(test_surface_depth_summary(2.0)),
            surface_depth_passed: true,
            depth_passed: true,
            loss: 2.0,
            passed: false,
            projected_bbox: Some([0.1, 0.1, 0.4, 0.5]),
            front_facing_face_count: 10,
            covered_px: 100,
            fallback_points: false,
            artifact_path: None,
            dense_optimization: None,
        };
        let mut best = baseline.clone();
        best.bbox_iou = 0.42;
        best.mask_iou = 0.22;
        best.loss = 1.2;
        let target = RotationFitTarget {
            mask: BinaryMask::from_normalized_bbox(100, 100, [0.1, 0.1, 0.5, 0.5])
                .expect("bbox mask"),
            bbox: [0.1, 0.1, 0.5, 0.5],
            crop_bbox: [0.0, 0.0, 0.6, 0.6],
            depth_median_m: Some(2.0),
            depth_stats: Some(burn_synth_scene::ObjectDepthStats {
                median_m: 2.0,
                min_m: 1.8,
                max_m: 2.4,
                contact_m: Some(2.1),
                sample_count: Some(64),
            }),
            dense_depth_crop: None,
            mask_kind: "mask_bbox_fallback",
        };
        let config = SceneVisibleSurfacePoseFitConfig {
            mode: ScenePoseFitMode::RenderedSilhouette,
            min_mask_iou: 0.45,
            max_depth_error_m: 0.35,
            write_artifacts: false,
            output_dir: Path::new("/tmp"),
            scale_policy: SceneScalePolicy::AssetPreserving,
            object_filter: ScenePoseFitObjectFilter::All,
        };
        assert!(visible_surface_pose_candidate_passes_target(
            &best, &baseline, &target, config
        ));
    }

    #[test]
    fn visible_surface_pose_gate_accepts_strong_table_projection_despite_depth_tail() {
        let mut table = test_pose_fit_placement(-90.0);
        table.entity_id = "conference_table_01".to_string();
        table.asset_id = "conference_table_01_asset".to_string();
        table.object_id = "conference_table_01".to_string();
        table.label = "white rounded rectangular conference table".to_string();
        table.translation = [0.0, 0.172, 0.08];
        table.ground_point = [0.0, 0.0, 0.08];
        table.scale = [1.212, 1.212, 1.212];

        let baseline = VisibleSurfacePoseCandidate {
            candidate_index: 0,
            stage: "baseline",
            placement: table.clone(),
            mask_iou: 0.275,
            bbox_iou: 0.582,
            center_error: 0.034,
            area_log2_error: 0.20,
            aspect_log2_error: 0.35,
            depth_error_m: Some(0.57),
            surface_depth_loss: Some(1.0),
            dense_depth_loss: None,
            dense_depth_mae_m: None,
            dense_depth_sample_count: 0,
            point_surface_loss: 0.18,
            semantic_yaw_prior_degrees: None,
            semantic_yaw_error_degrees: None,
            semantic_yaw_loss: 0.0,
            candidate_depth_summary: Some(test_surface_depth_summary(2.6)),
            target_depth_summary: Some(test_surface_depth_summary(2.0)),
            surface_depth_passed: false,
            depth_passed: false,
            loss: 4.70,
            passed: false,
            projected_bbox: Some([0.31, 0.30, 0.78, 0.72]),
            front_facing_face_count: 100,
            covered_px: 1000,
            fallback_points: false,
            artifact_path: None,
            dense_optimization: None,
        };

        let mut best_table = table;
        best_table.rotation_y_degrees = -62.5;
        best_table.translation = [-0.073, 0.157, 0.298];
        best_table.ground_point = [-0.073, 0.0, 0.298];
        best_table.scale = [1.106, 1.106, 1.106];
        let mut best = baseline.clone();
        best.candidate_index = 7;
        best.stage = "fine_table";
        best.placement = best_table;
        best.mask_iou = 0.333;
        best.bbox_iou = 0.856;
        best.center_error = 0.022;
        best.depth_error_m = Some(0.63);
        best.surface_depth_loss = Some(1.2);
        best.loss = 4.25;
        best.passed = false;
        best.projected_bbox = Some([0.30, 0.29, 0.80, 0.70]);

        let target = RotationFitTarget {
            mask: BinaryMask::from_normalized_bbox(100, 100, [0.30, 0.29, 0.80, 0.70])
                .expect("bbox mask"),
            bbox: [0.30, 0.29, 0.80, 0.70],
            crop_bbox: [0.20, 0.20, 0.90, 0.80],
            depth_median_m: Some(2.0),
            depth_stats: Some(burn_synth_scene::ObjectDepthStats {
                median_m: 2.0,
                min_m: 1.6,
                max_m: 3.0,
                contact_m: Some(2.0),
                sample_count: Some(256),
            }),
            dense_depth_crop: None,
            mask_kind: "sam_rle",
        };
        let mut config = SceneVisibleSurfacePoseFitConfig {
            mode: ScenePoseFitMode::RenderedSilhouette,
            min_mask_iou: 0.45,
            max_depth_error_m: 0.35,
            write_artifacts: false,
            output_dir: Path::new("/tmp"),
            scale_policy: SceneScalePolicy::AssetPreserving,
            object_filter: ScenePoseFitObjectFilter::All,
        };
        assert!(
            !visible_surface_pose_candidate_passes_target(&best, &baseline, &target, config),
            "generic fit should still reject a depth-failing candidate"
        );

        config.object_filter = ScenePoseFitObjectFilter::TablesOnly;
        assert!(
            visible_surface_pose_candidate_passes_target(&best, &baseline, &target, config),
            "table refinement should accept the strong projected table improvement"
        );
    }

    #[test]
    fn visible_surface_semantic_yaw_prior_faces_nearest_table_for_chairs() {
        let mut chair = test_pose_fit_placement(0.0);
        chair.ground_point = [0.0, 0.0, 0.0];
        chair.translation = [0.0, 0.0, 0.0];
        let mut table = test_pose_fit_placement(0.0);
        table.entity_id = "coffee_table".to_string();
        table.asset_id = "coffee_table_asset".to_string();
        table.object_id = "coffee_table".to_string();
        table.label = "coffee table".to_string();
        table.ground_point = [1.0, 0.0, 0.0];
        table.translation = [1.0, 0.0, 0.0];

        let prior = semantic_yaw_prior_for_placement(&chair, &[chair.clone(), table])
            .expect("chair should get a table-facing semantic prior");
        assert!((prior - 90.0).abs() <= 1.0e-5);
        assert!(semantic_yaw_loss_for_error(180.0) > semantic_yaw_loss_for_error(20.0));
    }

    #[test]
    fn visible_surface_semantic_yaw_prior_does_not_apply_to_tables() {
        let mut table = test_pose_fit_placement(0.0);
        table.entity_id = "coffee_table".to_string();
        table.asset_id = "coffee_table_asset".to_string();
        table.object_id = "coffee_table".to_string();
        table.label = "coffee table".to_string();
        let mut chair = test_pose_fit_placement(0.0);
        chair.ground_point = [1.0, 0.0, 0.0];

        assert!(semantic_yaw_prior_for_placement(&table, &[table.clone(), chair]).is_none());
    }

    #[test]
    fn visible_surface_depth_loss_prefers_matching_quantiles_and_contact() {
        let target = SurfaceDepthSummary {
            min_m: 1.8,
            p10_m: 1.9,
            median_m: 2.1,
            p90_m: 2.5,
            max_m: 2.7,
            contact_m: Some(2.0),
            sample_count: 100,
        };
        let close = SurfaceDepthSummary {
            min_m: 1.82,
            p10_m: 1.92,
            median_m: 2.08,
            p90_m: 2.52,
            max_m: 2.8,
            contact_m: Some(2.03),
            sample_count: 100,
        };
        let far = SurfaceDepthSummary {
            min_m: 3.0,
            p10_m: 3.2,
            median_m: 3.6,
            p90_m: 4.1,
            max_m: 4.4,
            contact_m: Some(3.5),
            sample_count: 100,
        };
        assert!(surface_depth_loss(target, close, 0.35) < 0.2);
        assert!(surface_depth_loss(target, far, 0.35) > 2.0);
    }

    #[test]
    fn mask_point_surface_loss_prefers_aligned_point_cloud_proxy() {
        let mut target = vec![0_u8; 16 * 16];
        let mut aligned = vec![0_u8; 16 * 16];
        let mut shifted = vec![0_u8; 16 * 16];
        for y in 4..12 {
            for x in 5..11 {
                target[y * 16 + x] = 1;
                aligned[y * 16 + x] = 1;
            }
        }
        for y in 1..9 {
            for x in 1..7 {
                shifted[y * 16 + x] = 1;
            }
        }
        let target = mask_point_moments(&target);
        let aligned_loss = mask_point_surface_loss(target, mask_point_moments(&aligned));
        let shifted_loss = mask_point_surface_loss(target, mask_point_moments(&shifted));
        assert!(aligned_loss < 1.0e-6);
        assert!(shifted_loss > aligned_loss + 0.20);
    }

    #[test]
    fn soft_camera_intrinsics_crop_maps_full_frame_center() {
        let intrinsics = RotationFitIntrinsics {
            fx: 1000.0,
            fy: 1000.0,
            cx: 500.0,
            cy: 250.0,
            width: 1001.0,
            height: 501.0,
        };
        let crop = [0.25, 0.25, 0.75, 0.75];
        let soft = soft_camera_intrinsics_for_crop(intrinsics, crop, 40).expect("crop intrinsics");
        assert!((soft.cx - 20.0).abs() < 1.0e-5);
        assert!((soft.cy - 20.0).abs() < 1.0e-5);
        assert!((soft.fx - 80.0).abs() < 1.0e-5);
        assert!((soft.fy - 160.0).abs() < 1.0e-5);
    }

    #[test]
    fn dense_soft_surface_pose_fit_optimizes_source_crop_transform() {
        let mesh = test_dense_fit_mesh();
        let evidence = test_dense_fit_evidence();
        let intrinsics = rotation_fit_intrinsics(&evidence).expect("intrinsics");
        let fit_object = test_projection_fit_object_report();
        let mut baseline = test_pose_fit_placement(0.0);
        baseline.translation = [0.10, 0.0, 0.12];
        baseline.ground_point = [0.10, 0.0, 0.12];
        baseline.scale = [0.92, 0.92, 0.92];
        let mut target_placement = baseline.clone();
        target_placement.translation = [0.0, 0.0, 0.0];
        target_placement.ground_point = [0.0, 0.0, 0.0];
        target_placement.rotation_y_degrees = 18.0;
        target_placement.scale = [1.04, 1.04, 1.04];

        let crop_bbox = [0.0, 0.0, 1.0, 1.0];
        let points = dense_soft_surface_points(&mesh, DENSE_SOFT_FIT_MAX_POINTS);
        let context = DenseSoftSurfaceTransformContext::from_placement(
            &target_placement,
            &fit_object,
            &evidence,
        )
        .expect("context");
        let soft_intrinsics = soft_camera_intrinsics_for_crop(
            intrinsics,
            crop_bbox,
            ROTATION_FIT_CROP_RESOLUTION as usize,
        )
        .expect("soft intrinsics");
        let target_transform = context.to_soft_transform(&target_placement);
        let (target_mask_f32, target_depth_f32) = burn_synth_render::cpu_render_soft_surface(
            &points,
            target_transform,
            soft_intrinsics,
            SoftRenderConfig {
                width: ROTATION_FIT_CROP_RESOLUTION as usize,
                height: ROTATION_FIT_CROP_RESOLUTION as usize,
                sigma_px: 1.75,
                depth_sigma_m: 0.12,
                mask_weight: 1.0,
                depth_weight: 0.0,
            },
        );
        let target_mask = target_mask_f32
            .iter()
            .map(|value| u8::from(*value >= 0.18))
            .collect::<Vec<_>>();
        let target = RotationFitTarget {
            mask: BinaryMask::from_normalized_bbox(100, 100, [0.25, 0.25, 0.75, 0.75])
                .expect("mask"),
            bbox: [0.25, 0.25, 0.75, 0.75],
            crop_bbox,
            depth_median_m: Some(target_transform.tz),
            depth_stats: Some(burn_synth_scene::ObjectDepthStats {
                median_m: target_transform.tz,
                min_m: target_transform.tz - 0.1,
                max_m: target_transform.tz + 0.1,
                contact_m: Some(target_transform.tz),
                sample_count: Some(256),
            }),
            dense_depth_crop: Some(DenseDepthCrop {
                depth_m: target_depth_f32,
                valid_mask: vec![1; (ROTATION_FIT_CROP_RESOLUTION as usize).pow(2)],
                width: ROTATION_FIT_CROP_RESOLUTION as usize,
                height: ROTATION_FIT_CROP_RESOLUTION as usize,
                valid_count: (ROTATION_FIT_CROP_RESOLUTION as usize).pow(2),
                source_path: PathBuf::from("synthetic_depth"),
            }),
            mask_kind: "sam_rle",
        };
        let config = SceneVisibleSurfacePoseFitConfig {
            mode: ScenePoseFitMode::RenderedSilhouette,
            min_mask_iou: 0.20,
            max_depth_error_m: 0.35,
            write_artifacts: false,
            output_dir: Path::new("/tmp"),
            scale_policy: SceneScalePolicy::AssetPreserving,
            object_filter: ScenePoseFitObjectFilter::All,
        };

        let (refined, report) = dense_soft_surface_refine_pose(
            &mesh,
            &baseline,
            &baseline,
            &fit_object,
            &evidence,
            intrinsics,
            &target,
            &target_mask,
            &config,
        )
        .expect("dense refinement");

        assert_eq!(report["status"], json!("optimized"));
        assert!(
            report["final_soft_loss"].as_f64().unwrap()
                < report["initial_soft_loss"].as_f64().unwrap(),
            "dense optimizer should reduce crop loss: {report:#}"
        );
        assert!(
            (refined.translation[0] - target_placement.translation[0]).abs()
                < (baseline.translation[0] - target_placement.translation[0]).abs(),
            "x translation should move toward target: refined={refined:?}"
        );
        assert!(
            (refined.translation[2] - target_placement.translation[2]).abs()
                < (baseline.translation[2] - target_placement.translation[2]).abs(),
            "z translation should move toward target: refined={refined:?}"
        );
    }

    #[test]
    fn table_pose_refinement_gated_gpt_flags_ambiguous_or_bad_table_fit() {
        let report = json!({
            "objects": [
                {
                    "object_id": "conference_table",
                    "instance_id": null,
                    "label": "conference table",
                    "applied": true,
                    "yaw_delta_degrees": 34.0,
                    "baseline": { "loss": 1.00 },
                    "best": { "loss": 0.96 },
                    "selected": {
                        "loss": 0.96,
                        "mask_iou": 0.34,
                        "bbox_iou": 0.49,
                        "center_error": 0.08
                    }
                }
            ]
        });
        let geometry_gate =
            table_pose_refinement_gpt_gate(SceneTablePoseRefinementMode::Geometry, &report);
        assert_eq!(geometry_gate["required"], json!(false));

        let gated = table_pose_refinement_gpt_gate(SceneTablePoseRefinementMode::GatedGpt, &report);
        assert_eq!(gated["required"], json!(true));
        assert_eq!(
            gated["objects"][0]["reason"],
            json!("low_mask_and_bbox_overlap")
        );
    }

    #[test]
    fn table_pose_refinement_clamps_table_scale_more_tightly_than_chair_scale() {
        let mut table = test_pose_fit_placement(0.0);
        table.entity_id = "table_0".to_string();
        table.object_id = "conference_table".to_string();
        table.label = "conference table".to_string();
        let mut table_trial = table.clone();
        table_trial.scale = [2.0, 2.0, 2.0];
        clamp_visible_surface_pose_candidate(
            &mut table_trial,
            &table,
            SceneScalePolicy::AssetPreserving,
        );
        assert!(
            table_trial.scale[0] <= 1.141,
            "table scale should stay near the baseline: {:?}",
            table_trial.scale
        );

        let chair = test_pose_fit_placement(0.0);
        let mut chair_trial = chair.clone();
        chair_trial.scale = [2.0, 2.0, 2.0];
        clamp_visible_surface_pose_candidate(
            &mut chair_trial,
            &chair,
            SceneScalePolicy::AssetPreserving,
        );
        assert!(
            chair_trial.scale[0] > table_trial.scale[0],
            "non-table objects should keep the wider generic fit window"
        );
    }

    #[test]
    fn table_pose_refinement_filter_does_not_match_conference_chairs() {
        let mut chair = test_pose_fit_placement(0.0);
        chair.entity_id = "chair_group_01".to_string();
        chair.object_id = "chair_group_01".to_string();
        chair.label = "reusable tufted swivel conference chair group".to_string();
        assert!(!placement_is_table_like(&chair));

        let mut table = test_pose_fit_placement(0.0);
        table.entity_id = "conference_table_01".to_string();
        table.object_id = "conference_table_01".to_string();
        table.label = "white rounded rectangular conference table".to_string();
        assert!(placement_is_table_like(&table));
    }

    fn test_pose_fit_placement(yaw: f32) -> GroundedScenePlacement {
        GroundedScenePlacement {
            entity_id: "chair_0".to_string(),
            asset_id: "chair_asset".to_string(),
            object_id: "chair".to_string(),
            instance_id: None,
            label: "chair".to_string(),
            source_bbox: [0.1, 0.1, 0.5, 0.5],
            contact_pixel: [0.3, 0.5],
            ground_point: [0.0, 0.0, 0.0],
            translation: [0.0, 0.0, 0.0],
            rotation_y_degrees: yaw,
            scale: [1.0, 1.0, 1.0],
            local_aabb: SceneAssetAabb {
                min: [-0.5, 0.0, -0.5],
                max: [0.5, 1.0, 0.5],
            },
            target_footprint_m: [1.0, 1.0],
        }
    }

    fn test_dense_fit_mesh() -> CachedSynthMesh {
        CachedSynthMesh {
            mesh: CachedTripoMesh {
                vertices: vec![
                    [-0.30, 0.00, -0.20],
                    [0.30, 0.00, -0.20],
                    [-0.30, 0.55, -0.20],
                    [0.30, 0.55, -0.20],
                    [-0.20, 0.00, 0.20],
                    [0.20, 0.00, 0.20],
                    [-0.20, 0.45, 0.20],
                    [0.20, 0.45, 0.20],
                ],
                faces: vec![
                    [0, 1, 2],
                    [1, 3, 2],
                    [4, 6, 5],
                    [5, 6, 7],
                    [0, 2, 4],
                    [2, 6, 4],
                    [1, 5, 3],
                    [3, 5, 7],
                ],
            },
            uvs: Vec::new(),
            normals: Vec::new(),
            material: None,
            pbr_textures: None,
        }
    }

    fn test_dense_fit_evidence() -> SceneGroundingEvidence {
        SceneGroundingEvidence {
            source_image_path: "/tmp/source.jpg".to_string(),
            depth: Some(burn_synth_scene::DepthEvidenceRef {
                provider: "synthetic".to_string(),
                model: None,
                precision: None,
                artifact_path: None,
                focal_length_px: Some(220.0),
                vertical_fov_degrees: Some(70.0),
                image_size: Some([256, 256]),
                depth_map_size: Some([256, 256]),
                floor_sample_count: Some(128),
            }),
            segmentation: None,
            detections: Vec::new(),
            camera: burn_synth_scene::EstimatedCamera {
                focal_length_px: Some(220.0),
                principal_point: Some([128.0, 128.0]),
                image_size: Some([256, 256]),
                vertical_fov_degrees: Some(70.0),
                confidence: Some(1.0),
            },
            floor: burn_synth_scene::EstimatedFloorPlane {
                normal: [0.0, 1.0, 0.0],
                distance_m: 0.0,
                residual_m: Some(0.01),
                confidence: Some(1.0),
            },
            objects: Vec::new(),
        }
    }

    fn test_projection_fit_object_report() -> burn_synth_scene::ProjectionFitObjectReport {
        burn_synth_scene::ProjectionFitObjectReport {
            index: 0,
            object_id: "chair".to_string(),
            instance_id: None,
            label: "chair".to_string(),
            source_bbox: [0.25, 0.25, 0.75, 0.75],
            projected_bbox: Some([0.30, 0.30, 0.70, 0.70]),
            projected_contact: Some([0.5, 0.75]),
            center_error: 0.0,
            contact_error: 0.0,
            area_log2_error: 0.0,
            aspect_log2_error: 0.0,
            bbox_iou: 1.0,
            depth_log2_error: None,
            yaw_prior_error_degrees: 0.0,
            ground_anchor_basis: "metric-depth-contact".to_string(),
            target_ground_point: [0.0, 0.0, 0.0],
            observed_ground_point: [0.0, 0.0, 0.0],
            source_camera_anchor: Some([0.0, 0.0, 3.0]),
            source_camera_origin_xz: Some([0.0, 3.0]),
            ground_anchor_error_m: 0.0,
            ground_anchor_max_drift_m: 0.6,
            ground_anchor_loss: 0.0,
            loss_without_ground_anchor: 0.0,
            loss: 0.0,
            score: 1.0,
            translation: [0.0, 0.0, 0.0],
            rotation_y_degrees: 0.0,
            scale: [1.0, 1.0, 1.0],
            visible_surface: None,
        }
    }

    fn test_surface_depth_summary(median_m: f32) -> SurfaceDepthSummary {
        SurfaceDepthSummary {
            min_m: median_m - 0.1,
            p10_m: median_m - 0.05,
            median_m,
            p90_m: median_m + 0.05,
            max_m: median_m + 0.1,
            contact_m: Some(median_m),
            sample_count: 8,
        }
    }
}

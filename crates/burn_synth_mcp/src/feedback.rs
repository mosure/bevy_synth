use crate::prelude::*;

pub(crate) fn scene_commands_from_plan(plan: &SceneComposePlan) -> Result<Vec<Value>, String> {
    let mut commands = Vec::with_capacity(plan.placements.len());
    if plan.clear_existing {
        commands.push(json!({ "type": "clear_scene" }));
    }
    for placement in &plan.placements {
        if let Some(path) = placement.path.as_ref() {
            commands.push(json!({
                "type": "spawn_path",
                "path": path,
                "cache_key": placement.cache_key,
                "translation": placement.translation,
                "rotation": placement.rotation,
                "scale": placement.scale,
                "select": placement.select,
            }));
        } else if let Some(cache_key) = placement.cache_key.as_ref() {
            commands.push(json!({
                "type": "spawn_cached",
                "cache_key": cache_key,
                "translation": placement.translation,
                "rotation": placement.rotation,
                "scale": placement.scale,
                "select": placement.select,
            }));
        } else {
            return Err(format!(
                "placement for '{}' has neither path nor cache_key",
                placement.label
            ));
        }
    }
    Ok(commands)
}

pub(crate) fn scene_interaction_lock_command(locked: bool, reason: &str) -> Value {
    json!({
        "type": "set_interaction_lock",
        "locked": locked,
        "reason": reason,
    })
}

pub(crate) fn scene_commands_with_cache_reload(mut commands: Vec<Value>) -> Vec<Value> {
    let uses_cache = commands.iter().any(|command| {
        command
            .get("type")
            .and_then(Value::as_str)
            .is_some_and(|command_type| command_type == "spawn_cached")
    });
    if !uses_cache {
        return commands;
    }

    let insert_at = commands
        .first()
        .and_then(|command| command.get("type"))
        .and_then(Value::as_str)
        .filter(|command_type| *command_type == "clear_scene")
        .map(|_| 1)
        .unwrap_or(0);
    commands.insert(insert_at, json!({ "type": "reload_cache" }));
    commands
}

pub(crate) fn scene_commands_with_asset_local_aabbs(
    mut commands: Vec<Value>,
    asset_bindings: &[SceneAssetBinding],
) -> Vec<Value> {
    for command in &mut commands {
        let Some(command_type) = command.get("type").and_then(Value::as_str) else {
            continue;
        };
        if command_type != "spawn_cached" && command_type != "spawn_path" {
            continue;
        }
        if command
            .get("local_aabb")
            .is_some_and(|value| !value.is_null())
        {
            continue;
        }
        let cache_key = command.get("cache_key").and_then(Value::as_str);
        let path = command.get("path").and_then(Value::as_str);
        let Some(asset) = asset_bindings.iter().find(|asset| {
            cache_key
                .is_some_and(|key| key == asset.asset_id || asset.cache_key.as_deref() == Some(key))
                || path.is_some_and(|path| asset.path.as_deref() == Some(path))
        }) else {
            continue;
        };
        if let Some(local_aabb) = asset.local_aabb {
            command["local_aabb"] = json!(local_aabb);
        }
    }
    commands
}

pub(crate) fn scene_composition_candidates(
    requested_mode: SceneCompositionMode,
    compare_feedback_candidates: bool,
    manifest: &SceneObjectManifest,
    asset_bindings: &[SceneAssetBinding],
    grounding_evidence: &SceneGroundingEvidence,
    clear_existing: bool,
    scale_policy: SceneScalePolicy,
) -> Result<Vec<SceneCompositionCandidate>, String> {
    scene_composition_candidate_modes(requested_mode, compare_feedback_candidates)
        .into_iter()
        .map(|mode| {
            let mut config = GroundedSceneLayoutConfig::default();
            config.scale_policy = scale_policy;
            let layout = match mode {
                SceneCompositionMode::Heuristic => {
                    grounded_scene_layout(manifest, asset_bindings, config)
                }
                SceneCompositionMode::CvGrounded => grounded_scene_layout_with_evidence_config(
                    manifest,
                    asset_bindings,
                    grounding_evidence,
                    config,
                ),
            }
            .map_err(|err| err.to_string())?;
            let plan =
                parse_scene_bsn(&layout.bsn, asset_bindings).map_err(|err| err.to_string())?;
            let commands = scene_commands_with_cache_reload(
                scene_plan_to_mcp_commands(&plan, asset_bindings, clear_existing)
                    .map_err(|err| err.to_string())?,
            );
            Ok(SceneCompositionCandidate {
                mode,
                layout,
                plan,
                commands,
            })
        })
        .collect()
}

pub(crate) fn scene_composition_candidate_modes(
    requested_mode: SceneCompositionMode,
    compare_feedback_candidates: bool,
) -> Vec<SceneCompositionMode> {
    if compare_feedback_candidates && requested_mode == SceneCompositionMode::CvGrounded {
        vec![
            SceneCompositionMode::CvGrounded,
            SceneCompositionMode::Heuristic,
        ]
    } else {
        vec![requested_mode]
    }
}

pub(crate) fn scene_composition_mode_label(mode: SceneCompositionMode) -> &'static str {
    match mode {
        SceneCompositionMode::Heuristic => "heuristic",
        SceneCompositionMode::CvGrounded => "cv-grounded",
    }
}

pub(crate) fn feedback_result_selection_score(feedback: &Value) -> f64 {
    let best_score = feedback
        .get("best_score")
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
        .unwrap_or(f64::NEG_INFINITY);
    let final_score = feedback
        .get("final_evidence")
        .and_then(|evidence| evidence.get("metrics"))
        .map(feedback_selection_score)
        .filter(|value| value.is_finite())
        .unwrap_or(f64::NEG_INFINITY);
    let score = best_score.max(final_score);
    let accepted_bonus = if feedback
        .get("accepted")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        1000.0
    } else {
        0.0
    };
    score + accepted_bonus
}

pub(crate) fn feedback_report_path_from_result(feedback: &Value) -> Option<String> {
    let capture_dir = feedback.get("capture_dir").and_then(Value::as_str)?;
    Some(
        Path::new(capture_dir)
            .join("feedback_report.md")
            .display()
            .to_string(),
    )
}

pub(crate) fn finite_json_f64(value: f64) -> Option<Value> {
    value.is_finite().then_some(Value::from(value))
}

pub(crate) fn spawn_feedback_viewer(control_path: &Path, log_path: &Path) -> Result<Child, String> {
    let exe = feedback_viewer_exe()?;
    ensure_parent_dir(control_path).map_err(|err| err.to_string())?;
    ensure_parent_dir(log_path).map_err(|err| err.to_string())?;
    let log = fs::File::create(log_path).map_err(|err| {
        format!(
            "failed to create feedback viewer log {}: {err}",
            log_path.display()
        )
    })?;
    let err_log = log
        .try_clone()
        .map_err(|err| format!("failed to clone feedback viewer log handle: {err}"))?;
    Command::new(&exe)
        .arg("--mcp-scene-control-path")
        .arg(control_path)
        .arg("--ui-visible")
        .arg("false")
        .arg("--read-only")
        .stdin(Stdio::null())
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(err_log))
        .spawn()
        .map_err(|err| format!("failed to spawn feedback viewer {}: {err}", exe.display()))
}

pub(crate) fn feedback_viewer_exe() -> Result<PathBuf, String> {
    let current = std::env::current_exe()
        .map_err(|err| format!("failed to resolve current executable: {err}"))?;
    let direct = current.with_file_name(format!("bevy_synth{}", std::env::consts::EXE_SUFFIX));
    if direct.exists() {
        return Ok(direct);
    }
    if current
        .parent()
        .and_then(|parent| parent.file_name())
        .and_then(|name| name.to_str())
        .is_some_and(|name| name == "deps")
        && let Some(parent) = current.parent().and_then(Path::parent)
    {
        let candidate = parent.join(format!("bevy_synth{}", std::env::consts::EXE_SUFFIX));
        if candidate.exists() {
            return Ok(candidate);
        }
    }
    Ok(direct)
}

pub(crate) fn scene_feedback_metrics(
    manifest: &SceneObjectManifest,
    grounded_layout: &GroundedSceneLayout,
    status: &Value,
    screenshot_path: &Path,
    thresholds: SceneFeedbackThresholds,
    profile: FeedbackThresholdProfile,
) -> Result<Value, String> {
    let projected = status
        .get("projected_items")
        .and_then(Value::as_array)
        .ok_or_else(|| "scene feedback status missing projected_items array".to_string())?;
    let mut object_metrics = Vec::new();
    let mut passed_count = 0usize;
    let mut score_total = 0.0f32;
    let camera_yaw_degrees = status_camera_yaw_degrees(status, grounded_layout.camera.yaw);
    let feedback_camera = FeedbackCamera::from_status(status, screenshot_path);
    let floor_y = grounded_layout.rug_center[1];
    let footprints = feedback_projected_footprints(&grounded_layout.placements, projected);
    let physical = feedback_physical_layout(&grounded_layout.placements, &footprints, thresholds);
    let mut yaw_passed_count = 0usize;
    for (index, placement) in grounded_layout.placements.iter().enumerate() {
        let projected_item = projected.get(index).unwrap_or(&Value::Null);
        let observed_bbox = projected_item.get("screen_bbox").and_then(json_array4);
        let observed_contact = projected_item.get("screen_contact").and_then(json_array2);
        let expected_bbox = placement.source_bbox;
        let expected_contact = placement.contact_pixel;
        let expected_anchor = feedback_expected_anchor_pixel(placement);
        let anchor_basis = feedback_anchor_basis(placement);
        let uses_center_anchor = feedback_uses_bbox_center_anchor(placement);
        let source_edge_cropped = feedback_source_bbox_edge_cropped(placement);
        let projected_cache_key = projected_item.get("cache_key").and_then(Value::as_str);
        let current_yaw_degrees = status_world_item_yaw_degrees(status, index, projected_cache_key)
            .unwrap_or(placement.rotation_y_degrees);
        let yaw_correction =
            feedback_yaw_correction(index, placement, current_yaw_degrees, &physical);
        let rotation_selection = feedback_rotation_selection(
            current_yaw_degrees,
            yaw_correction.delta_degrees,
            yaw_correction.basis,
        );
        let selected_yaw_delta = rotation_selection
            .get("selected_yaw_delta_degrees")
            .and_then(Value::as_f64)
            .map(|value| value as f32)
            .unwrap_or(yaw_correction.delta_degrees);
        let yaw_delta_abs = selected_yaw_delta.abs();
        let max_yaw_delta = feedback_max_yaw_delta_for_pass(profile, placement);
        let (
            center_error,
            area_log2_error,
            aspect_log2_error,
            contact_error,
            score,
            passed,
            bbox_overscan,
            max_bbox_overscan,
        ) = if let Some(observed_bbox) = observed_bbox {
            let edge_crop_relax = source_edge_cropped
                && (feedback_edge_crop_can_relax_projection(placement) || uses_center_anchor);
            let visible_bbox_scoring =
                source_edge_cropped || feedback_bbox_outside_frame(observed_bbox);
            let bbox_overscan = feedback_bbox_overscan(observed_bbox);
            let max_bbox_overscan = feedback_max_bbox_overscan(placement);
            let use_full_bbox_for_area = bbox_overscan > max_bbox_overscan;
            let expected_score_bbox = if visible_bbox_scoring {
                feedback_visible_bbox(expected_bbox).unwrap_or(expected_bbox)
            } else {
                expected_bbox
            };
            let observed_score_bbox = if visible_bbox_scoring && !use_full_bbox_for_area {
                feedback_visible_bbox(observed_bbox).unwrap_or(observed_bbox)
            } else {
                observed_bbox
            };
            let expected_center = bbox_center(expected_score_bbox);
            let observed_center = bbox_center(observed_score_bbox);
            let center_error = distance2(expected_center, observed_center);
            let expected_area = bbox_area(expected_score_bbox);
            let observed_area = bbox_area(observed_score_bbox);
            let area_log2_error = safe_log2_ratio(observed_area, expected_area).abs();
            let aspect_log2_error = safe_log2_ratio(
                bbox_aspect(observed_score_bbox),
                bbox_aspect(expected_score_bbox),
            )
            .abs();
            let observed_anchor = if uses_center_anchor {
                observed_center
            } else {
                feedback_observed_anchor_pixel(placement, observed_bbox, observed_contact)
            };
            let contact_error = distance2(expected_anchor, observed_anchor);
            let mut center_limit = feedback_center_error_limit(
                uses_center_anchor,
                contact_error,
                thresholds.max_center_error,
                thresholds.max_contact_error,
            );
            if edge_crop_relax
                && !uses_center_anchor
                && contact_error <= thresholds.max_contact_error
            {
                center_limit = center_limit.max(1.0);
            }
            let area_limit = feedback_area_log2_error_limit(
                uses_center_anchor,
                feedback_repeated_object_like(placement),
                edge_crop_relax,
                center_error,
                contact_error,
                thresholds.max_area_log2_error,
                thresholds.max_center_error,
                thresholds.max_contact_error,
            );
            let center_score = (1.0 - center_error / center_limit.max(1.0e-5)).clamp(0.0, 1.0);
            let contact_score =
                (1.0 - contact_error / thresholds.max_contact_error.max(1.0e-5)).clamp(0.0, 1.0);
            let area_score = (1.0 - area_log2_error / area_limit.max(1.0e-5)).clamp(0.0, 1.0);
            let score = if uses_center_anchor {
                center_score * 0.45 + contact_score * 0.25 + area_score * 0.30
            } else {
                center_score * 0.20 + contact_score * 0.45 + area_score * 0.35
            };
            let overscan_passed = bbox_overscan <= max_bbox_overscan;
            let overscan_score =
                (1.0 - bbox_overscan / (max_bbox_overscan * 1.75).max(1.0e-5)).clamp(0.0, 1.0);
            let score = score * (0.65 + 0.35 * overscan_score);
            let passed = center_error <= center_limit
                && contact_error <= thresholds.max_contact_error
                && area_log2_error <= area_limit
                && overscan_passed;
            (
                center_error,
                area_log2_error,
                aspect_log2_error,
                contact_error,
                score,
                passed,
                bbox_overscan,
                max_bbox_overscan,
            )
        } else {
            (1.0, 8.0, 8.0, 1.0, 0.0, false, 1.0, 0.0)
        };
        if passed {
            passed_count += 1;
        }
        if yaw_delta_abs <= max_yaw_delta {
            yaw_passed_count += 1;
        }
        score_total += score;
        let observed_bbox = observed_bbox.unwrap_or([0.0, 0.0, 0.0, 0.0]);
        let observed_contact = observed_contact.unwrap_or(bbox_center(observed_bbox));
        let visible_bbox_scoring =
            source_edge_cropped || feedback_bbox_outside_frame(observed_bbox);
        let expected_delta_bbox = if visible_bbox_scoring {
            feedback_visible_bbox(expected_bbox).unwrap_or(expected_bbox)
        } else {
            expected_bbox
        };
        let observed_delta_bbox = if visible_bbox_scoring {
            feedback_visible_bbox(observed_bbox).unwrap_or(observed_bbox)
        } else {
            observed_bbox
        };
        let observed_anchor = if feedback_uses_bbox_center_anchor(placement) {
            bbox_center(observed_delta_bbox)
        } else {
            feedback_observed_anchor_pixel(placement, observed_bbox, Some(observed_contact))
        };
        let expected_area = bbox_area(expected_delta_bbox);
        let scale_observed_bbox = if bbox_overscan > max_bbox_overscan {
            observed_bbox
        } else {
            observed_delta_bbox
        };
        let observed_area = bbox_area(scale_observed_bbox);
        let scale_multiplier = if observed_area > 1.0e-5 {
            (expected_area / observed_area).sqrt().clamp(0.82, 1.22)
        } else {
            1.0
        };
        let contact_delta = vec2_sub(expected_anchor, observed_anchor);
        let center_delta = vec2_sub(
            bbox_center(expected_delta_bbox),
            bbox_center(observed_delta_bbox),
        );
        let fallback_delta = [
            (center_delta[0] * 2.0).clamp(-0.35, 0.35),
            0.0,
            (contact_delta[1] * 2.0).clamp(-0.35, 0.35),
        ];
        let target_ground_point =
            feedback_camera.and_then(|camera| camera.ground_point(expected_anchor, floor_y));
        let observed_ground_point = projected_item_ground_point(projected_item);
        let (mut translation_delta, grounding_basis) =
            if let (Some(target), Some(observed)) = (target_ground_point, observed_ground_point) {
                (
                    clamp_xz_delta(
                        [target[0] - observed[0], 0.0, target[2] - observed[2]],
                        0.85,
                    ),
                    "camera-ray-ground-plane",
                )
            } else {
                (fallback_delta, "screen-space-fallback")
            };
        let center_residual_applied = grounding_basis == "camera-ray-ground-plane"
            && !feedback_uses_bbox_center_anchor(placement)
            && contact_error <= 0.04
            && center_error > 0.04;
        if center_residual_applied {
            let residual = [
                (-center_delta[0] * 1.15).clamp(-0.18, 0.18),
                0.0,
                (-center_delta[1] * 1.15).clamp(-0.18, 0.18),
            ];
            translation_delta = clamp_xz_delta(add3(translation_delta, residual), 0.85);
        }
        let contact_residual_applied = grounding_basis == "camera-ray-ground-plane"
            && !feedback_uses_bbox_center_anchor(placement)
            && contact_error > 0.04;
        if contact_residual_applied {
            let residual = [
                (center_delta[0] * 0.65).clamp(-0.14, 0.14),
                0.0,
                (-contact_delta[1] * 1.45).clamp(-0.26, 0.26),
            ];
            translation_delta = clamp_xz_delta(add3(translation_delta, residual), 0.95);
        }
        if feedback_physical_kind(placement) == FeedbackPhysicalKind::Table {
            translation_delta = clamp_xz_delta(translation_delta, 0.15);
        }
        let physical_delta = physical
            .corrections
            .get(&index)
            .copied()
            .unwrap_or([0.0, 0.0, 0.0]);
        if physical_delta[0].abs() + physical_delta[2].abs() > 1.0e-5 {
            translation_delta = clamp_xz_delta(add3(translation_delta, physical_delta), 1.10);
        }
        let predictive_physical_delta = feedback_predictive_physical_delta(
            index,
            placement,
            translation_delta,
            &footprints,
            thresholds,
        );
        if predictive_physical_delta[0].abs() + predictive_physical_delta[2].abs() > 1.0e-5 {
            translation_delta =
                clamp_xz_delta(add3(translation_delta, predictive_physical_delta), 1.10);
        }
        let footprint = footprints.get(index).and_then(|footprint| *footprint);
        let world_footprint = footprint.map(|footprint| {
            json!({
                "min_x": footprint.rect.min_x,
                "min_z": footprint.rect.min_z,
                "max_x": footprint.rect.max_x,
                "max_z": footprint.rect.max_z,
            })
        });
        let canonical_yaw_error =
            normalize_degrees(placement.rotation_y_degrees - current_yaw_degrees);
        let physical_failures = physical
            .object_failures
            .get(&index)
            .cloned()
            .unwrap_or_default();
        let physical_passed = physical_failures.is_empty();
        object_metrics.push(json!({
            "index": index,
            "object_id": placement.object_id,
            "instance_id": placement.instance_id,
            "label": placement.label,
            "cache_key": projected_item.get("cache_key").cloned().unwrap_or(Value::Null),
            "expected_bbox": expected_bbox,
            "observed_bbox": observed_bbox,
            "visible_expected_bbox": feedback_visible_bbox(expected_bbox),
            "visible_observed_bbox": feedback_visible_bbox(observed_bbox),
            "visible_bbox_scoring": visible_bbox_scoring,
            "expected_contact": expected_contact,
            "observed_contact": observed_contact,
            "expected_anchor": expected_anchor,
            "observed_anchor": observed_anchor,
            "anchor_basis": anchor_basis,
            "source_edge_cropped": source_edge_cropped,
            "center_error": center_error,
            "contact_error": contact_error,
            "area_log2_error": area_log2_error,
            "aspect_log2_error": aspect_log2_error,
            "bbox_overscan": bbox_overscan,
            "max_bbox_overscan": max_bbox_overscan,
            "score": score,
            "passed": passed,
            "translation_delta": translation_delta,
            "grounding_basis": grounding_basis,
            "center_residual_applied": center_residual_applied,
            "contact_residual_applied": contact_residual_applied,
            "physical_translation_delta": physical_delta,
            "predictive_physical_translation_delta": predictive_physical_delta,
            "physical_kind": feedback_physical_kind_str(feedback_physical_kind(placement)),
            "world_footprint": world_footprint,
            "physical_passed": physical_passed,
            "physical_failures": physical_failures,
            "target_ground_point": target_ground_point,
            "observed_ground_point": observed_ground_point,
            "ground_anchor_point": placement.ground_point,
            "ground_anchor_max_drift_m": placement.ground_anchor_max_drift_m(),
            "scale_multiplier": scale_multiplier,
            "yaw_delta_degrees": selected_yaw_delta,
            "yaw_delta_abs_degrees": yaw_delta_abs,
            "max_yaw_delta_degrees": max_yaw_delta,
            "yaw_passed": yaw_delta_abs <= max_yaw_delta,
            "yaw_basis": yaw_correction.basis,
            "rotation_selection": rotation_selection,
            "current_yaw_degrees": current_yaw_degrees,
            "canonical_yaw_degrees": placement.rotation_y_degrees,
            "canonical_yaw_error_degrees": canonical_yaw_error,
            "camera_yaw_degrees": camera_yaw_degrees,
            "target_yaw_degrees": normalize_degrees(current_yaw_degrees + selected_yaw_delta),
        }));
    }
    let object_count = grounded_layout.placements.len().max(1);
    let mean_score = score_total / object_count as f32;
    let projection_passed = passed_count == grounded_layout.placements.len()
        && mean_score >= thresholds.min_overall_score;
    let physical_passed = physical.hard_failure_count == 0;
    let rotation_passed = yaw_passed_count == grounded_layout.placements.len();
    let passed = projection_passed && physical_passed && rotation_passed;
    Ok(json!({
        "profile": profile,
        "passed": passed,
        "score": mean_score,
        "projection_passed": projection_passed,
        "physical_passed": physical_passed,
        "rotation_passed": rotation_passed,
        "object_count": grounded_layout.placements.len(),
        "object_pass_count": passed_count,
        "rotation_pass_count": yaw_passed_count,
        "physical_pass_count": grounded_layout.placements.len().saturating_sub(physical.object_failure_count),
        "source_scene_path": manifest.source_scene_path,
        "screenshot_path": screenshot_path.display().to_string(),
        "thresholds": {
            "max_center_error": thresholds.max_center_error,
            "max_contact_error": thresholds.max_contact_error,
            "max_area_log2_error": thresholds.max_area_log2_error,
            "min_overall_score": thresholds.min_overall_score,
            "max_seating_table_overlap_fraction": thresholds.max_seating_table_overlap_fraction,
            "max_seating_table_penetration_m": thresholds.max_seating_table_penetration_m,
            "max_seating_seating_overlap_fraction": thresholds.max_seating_seating_overlap_fraction,
            "max_seating_seating_penetration_m": thresholds.max_seating_seating_penetration_m,
        },
        "physical_layout": {
            "passed": physical_passed,
            "hard_failure_count": physical.hard_failure_count,
            "warning_count": physical.warning_count,
            "object_failure_count": physical.object_failure_count,
            "max_overlap_fraction_smaller": physical.max_overlap_fraction_smaller,
            "min_signed_clearance_m": physical.min_signed_clearance_m,
            "pairs": physical.pairs,
        },
        "objects": object_metrics,
        "camera": status.get("camera").cloned().unwrap_or(Value::Null),
    }))
}

pub(crate) fn feedback_center_error_limit(
    uses_center_anchor: bool,
    contact_error: f32,
    max_center_error: f32,
    max_contact_error: f32,
) -> f32 {
    if uses_center_anchor {
        max_center_error
    } else if contact_error <= max_contact_error {
        max_center_error * 1.60
    } else {
        max_center_error * 1.25
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn feedback_area_log2_error_limit(
    uses_center_anchor: bool,
    repeated_object_like: bool,
    edge_cropped: bool,
    center_error: f32,
    contact_error: f32,
    max_area_log2_error: f32,
    max_center_error: f32,
    max_contact_error: f32,
) -> f32 {
    if uses_center_anchor
        && edge_cropped
        && center_error <= max_center_error
        && contact_error <= max_contact_error
    {
        max_area_log2_error.max(2.05)
    } else if uses_center_anchor {
        max_area_log2_error
    } else if edge_cropped && contact_error <= max_contact_error {
        max_area_log2_error.max(1.35)
    } else if repeated_object_like
        && contact_error <= max_contact_error * 0.35
        && center_error <= max_center_error
    {
        max_area_log2_error.max(0.98)
    } else {
        max_area_log2_error * 1.25
    }
}

pub(crate) fn feedback_expected_anchor_pixel(placement: &GroundedScenePlacement) -> [f32; 2] {
    if feedback_uses_bbox_center_anchor(placement) {
        bbox_center(placement.source_bbox)
    } else {
        placement.contact_pixel
    }
}

pub(crate) fn feedback_observed_anchor_pixel(
    placement: &GroundedScenePlacement,
    observed_bbox: [f32; 4],
    observed_contact: Option<[f32; 2]>,
) -> [f32; 2] {
    if feedback_uses_bbox_center_anchor(placement) {
        bbox_center(observed_bbox)
    } else {
        observed_contact.unwrap_or_else(|| bbox_center(observed_bbox))
    }
}

pub(crate) fn feedback_anchor_basis(placement: &GroundedScenePlacement) -> &'static str {
    if feedback_uses_bbox_center_anchor(placement) {
        "bbox-center"
    } else {
        "floor-contact"
    }
}

pub(crate) fn feedback_uses_bbox_center_anchor(placement: &GroundedScenePlacement) -> bool {
    let descriptor = format!("{} {}", placement.object_id, placement.label).to_lowercase();
    descriptor.contains("table")
        || descriptor.contains("desk")
        || descriptor.contains("counter")
        || descriptor.contains("bench")
        || (feedback_source_bbox_edge_cropped(placement)
            && (descriptor.contains("sofa")
                || descriptor.contains("couch")
                || descriptor.contains("sectional")
                || descriptor.contains("loveseat")
                || descriptor.contains("banquette")))
}

pub(crate) fn feedback_source_bbox_edge_cropped(placement: &GroundedScenePlacement) -> bool {
    let [x0, y0, x1, y1] = placement.source_bbox;
    x0 <= 0.035 || y0 <= 0.035 || x1 >= 0.965 || y1 >= 0.965
}

pub(crate) fn feedback_max_yaw_delta_for_pass(
    profile: FeedbackThresholdProfile,
    placement: &GroundedScenePlacement,
) -> f32 {
    let base = match profile {
        FeedbackThresholdProfile::Loose => 16.0,
        FeedbackThresholdProfile::Standard => 8.0,
        FeedbackThresholdProfile::Strict => 4.0,
    };
    if feedback_uses_bbox_center_anchor(placement) {
        (base * 0.75_f32).max(3.0)
    } else {
        base
    }
}

pub(crate) fn feedback_repeated_object_like(placement: &GroundedScenePlacement) -> bool {
    placement.instance_id.is_some()
        || placement.object_id.to_ascii_lowercase().contains("group")
        || placement.asset_id.to_ascii_lowercase().contains("group")
}

pub(crate) fn feedback_edge_crop_can_relax_projection(placement: &GroundedScenePlacement) -> bool {
    if feedback_repeated_object_like(placement) {
        return true;
    }
    let descriptor = format!("{} {}", placement.object_id, placement.label).to_ascii_lowercase();
    descriptor.contains("chair") || descriptor.contains("stool")
}

pub(crate) fn feedback_bbox_outside_frame(bbox: [f32; 4]) -> bool {
    bbox[0] < -1.0e-3 || bbox[1] < -1.0e-3 || bbox[2] > 1.0 + 1.0e-3 || bbox[3] > 1.0 + 1.0e-3
}

pub(crate) fn feedback_bbox_overscan(bbox: [f32; 4]) -> f32 {
    [
        (-bbox[0]).max(0.0),
        (-bbox[1]).max(0.0),
        (bbox[2] - 1.0).max(0.0),
        (bbox[3] - 1.0).max(0.0),
    ]
    .into_iter()
    .fold(0.0f32, f32::max)
}

pub(crate) fn feedback_max_bbox_overscan(placement: &GroundedScenePlacement) -> f32 {
    let descriptor = format!("{} {}", placement.object_id, placement.label).to_ascii_lowercase();
    if !feedback_source_bbox_edge_cropped(placement) {
        return 0.08;
    }
    if feedback_repeated_object_like(placement)
        || descriptor.contains("chair")
        || descriptor.contains("stool")
    {
        0.85
    } else if descriptor.contains("sofa")
        || descriptor.contains("couch")
        || descriptor.contains("sectional")
        || descriptor.contains("loveseat")
    {
        0.38
    } else {
        0.25
    }
}

pub(crate) fn feedback_visible_bbox(bbox: [f32; 4]) -> Option<[f32; 4]> {
    let clipped = [
        bbox[0].clamp(0.0, 1.0),
        bbox[1].clamp(0.0, 1.0),
        bbox[2].clamp(0.0, 1.0),
        bbox[3].clamp(0.0, 1.0),
    ];
    (bbox_area(clipped) > 1.0e-5).then_some(clipped)
}

#[cfg(test)]
pub(crate) fn feedback_layout_deltas(metrics: &Value) -> Value {
    feedback_layout_deltas_with_policy(metrics, SceneScalePolicy::AssetPreserving)
}

pub(crate) fn feedback_layout_deltas_with_policy(
    metrics: &Value,
    scale_policy: SceneScalePolicy,
) -> Value {
    let objects = metrics
        .get("objects")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut scale_groups: HashMap<String, (f64, usize)> = HashMap::new();
    let mut camera_ray_contact_sum = 0.0f64;
    let mut camera_ray_contact_count = 0usize;
    for object in &objects {
        if object
            .get("grounding_basis")
            .and_then(Value::as_str)
            .is_some_and(|basis| basis == "camera-ray-ground-plane")
            && let Some(contact_error) = object
                .get("contact_error")
                .and_then(Value::as_f64)
                .filter(|value| value.is_finite())
        {
            camera_ray_contact_sum += contact_error;
            camera_ray_contact_count += 1;
        }
        let Some(group_key) = feedback_scale_group_key(object) else {
            continue;
        };
        let Some(scale) = object
            .get("scale_multiplier")
            .and_then(Value::as_f64)
            .filter(|value| value.is_finite())
        else {
            continue;
        };
        let scale = damped_feedback_scale_multiplier(object, scale);
        let entry = scale_groups.entry(group_key).or_insert((0.0, 0));
        entry.0 += scale.clamp(0.82, 1.22);
        entry.1 += 1;
    }
    let repeated_scale_by_group = scale_groups
        .into_iter()
        .filter_map(|(key, (sum, count))| {
            if count > 1 {
                Some((key, (sum / count as f64).clamp(0.82, 1.22)))
            } else {
                None
            }
        })
        .collect::<HashMap<_, _>>();
    let mut source_min = [f32::INFINITY, f32::INFINITY];
    let mut source_max = [f32::NEG_INFINITY, f32::NEG_INFINITY];
    let mut observed_min = [f32::INFINITY, f32::INFINITY];
    let mut observed_max = [f32::NEG_INFINITY, f32::NEG_INFINITY];
    for object in &objects {
        if let Some(bbox) = object.get("expected_bbox").and_then(json_array4) {
            expand_bbox_envelope(&mut source_min, &mut source_max, bbox);
        }
        if let Some(bbox) = object.get("observed_bbox").and_then(json_array4) {
            expand_bbox_envelope(&mut observed_min, &mut observed_max, bbox);
        }
    }
    let source_area = envelope_area(source_min, source_max);
    let observed_area = envelope_area(observed_min, observed_max);
    let raw_camera_radius_multiplier = if source_area > 1.0e-5 && observed_area > 1.0e-5 {
        (observed_area / source_area).sqrt().clamp(0.90, 1.10)
    } else {
        1.0
    };
    let camera_radius_multiplier = if camera_ray_contact_count > 0 {
        let mean_contact = camera_ray_contact_sum / camera_ray_contact_count as f64;
        if mean_contact > 0.05 {
            1.0
        } else {
            (1.0 + (raw_camera_radius_multiplier - 1.0) * 0.25).clamp(0.97, 1.03)
        }
    } else {
        raw_camera_radius_multiplier
    };
    let mut object_deltas = objects
        .iter()
        .map(|object| {
            let group_key = feedback_scale_group_key(object);
            let grouped_scale = group_key
                .as_ref()
                .and_then(|key| repeated_scale_by_group.get(key))
                .copied();
            let object_scale = object
                .get("scale_multiplier")
                .and_then(Value::as_f64)
                .filter(|value| value.is_finite())
                .unwrap_or(1.0)
                .clamp(0.82, 1.22);
            let object_scale = damped_feedback_scale_multiplier(object, object_scale);
            let axis_scale = feedback_axis_scale_multiplier(object, scale_policy);
            FeedbackDeltaDraft {
                index: object.get("index").cloned().unwrap_or(Value::Null),
                translation_delta: object
                    .get("translation_delta")
                    .and_then(json_array3)
                    .unwrap_or([0.0, 0.0, 0.0]),
                scale_multiplier: if axis_scale.is_some() {
                    1.0
                } else {
                    grouped_scale.unwrap_or(object_scale)
                },
                scale_multiplier_xyz: axis_scale,
                scale_group_key: group_key,
                scale_source: if grouped_scale.is_some() {
                    "repeated_instance_group"
                } else if axis_scale.is_some() {
                    "axis_projection"
                } else {
                    "object_projection"
                },
                yaw_delta_degrees: object
                    .get("yaw_delta_degrees")
                    .cloned()
                    .unwrap_or(json!(0.0)),
                ground_anchor_point: object
                    .get("target_ground_point")
                    .and_then(json_array3)
                    .or_else(|| object.get("ground_anchor_point").and_then(json_array3)),
                ground_anchor_max_drift_m: object
                    .get("ground_anchor_max_drift_m")
                    .and_then(Value::as_f64)
                    .filter(|value| value.is_finite() && *value > 0.0)
                    .map(|value| value as f32),
            }
        })
        .collect::<Vec<_>>();
    let thresholds = feedback_thresholds_from_metrics(metrics);
    feedback_project_delta_collisions(&objects, &mut object_deltas, thresholds);
    json!({
        "objects": object_deltas.into_iter().map(|delta| {
            json!({
                "index": delta.index,
                "translation_delta": delta.translation_delta,
                "scale_multiplier": delta.scale_multiplier,
                "scale_multiplier_xyz": delta.scale_multiplier_xyz,
                "scale_group_key": delta.scale_group_key,
                "scale_source": delta.scale_source,
                "yaw_delta_degrees": delta.yaw_delta_degrees,
                "ground_anchor_point": delta.ground_anchor_point,
                "ground_anchor_max_drift_m": delta.ground_anchor_max_drift_m,
            })
        }).collect::<Vec<_>>(),
        "camera": {
            "radius_multiplier": camera_radius_multiplier,
        }
    })
}

#[derive(Debug)]
pub(crate) struct FeedbackDeltaDraft {
    pub(crate) index: Value,
    pub(crate) translation_delta: [f32; 3],
    pub(crate) scale_multiplier: f64,
    pub(crate) scale_multiplier_xyz: Option<[f64; 3]>,
    pub(crate) scale_group_key: Option<String>,
    pub(crate) scale_source: &'static str,
    pub(crate) yaw_delta_degrees: Value,
    pub(crate) ground_anchor_point: Option<[f32; 3]>,
    pub(crate) ground_anchor_max_drift_m: Option<f32>,
}

pub(crate) fn feedback_axis_scale_multiplier(
    object: &Value,
    scale_policy: SceneScalePolicy,
) -> Option<[f64; 3]> {
    if !scale_policy.allows_axis_feedback() {
        return None;
    }
    if !feedback_json_object_is_table_like(object) {
        return None;
    }
    let source_edge_cropped = object
        .get("source_edge_cropped")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let visible_bbox_scoring = object
        .get("visible_bbox_scoring")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if source_edge_cropped || visible_bbox_scoring {
        return None;
    }
    let expected = object.get("expected_bbox").and_then(json_array4)?;
    let observed = object.get("observed_bbox").and_then(json_array4)?;
    let expected_width = (expected[2] - expected[0]).abs().max(1.0e-5) as f64;
    let expected_height = (expected[3] - expected[1]).abs().max(1.0e-5) as f64;
    let observed_width = (observed[2] - observed[0]).abs().max(1.0e-5) as f64;
    let observed_height = (observed[3] - observed[1]).abs().max(1.0e-5) as f64;
    let width_multiplier =
        damped_axis_scale_ratio(expected_width / observed_width, 0.34, 0.84, 1.22);
    let depth_multiplier =
        damped_axis_scale_ratio(expected_height / observed_height, 0.22, 0.90, 1.12);
    Some([width_multiplier, 1.0, depth_multiplier])
}

pub(crate) fn damped_axis_scale_ratio(
    ratio: f64,
    weight: f64,
    min_value: f64,
    max_value: f64,
) -> f64 {
    let ratio = ratio.clamp(0.45, 2.40);
    (1.0 + (ratio - 1.0) * weight).clamp(min_value, max_value)
}

pub(crate) fn feedback_thresholds_from_metrics(metrics: &Value) -> SceneFeedbackThresholds {
    let defaults = FeedbackThresholdProfile::Standard.thresholds();
    let Some(thresholds) = metrics.get("thresholds") else {
        return defaults;
    };
    SceneFeedbackThresholds {
        max_center_error: thresholds
            .get("max_center_error")
            .and_then(Value::as_f64)
            .map(|value| value as f32)
            .unwrap_or(defaults.max_center_error),
        max_contact_error: thresholds
            .get("max_contact_error")
            .and_then(Value::as_f64)
            .map(|value| value as f32)
            .unwrap_or(defaults.max_contact_error),
        max_area_log2_error: thresholds
            .get("max_area_log2_error")
            .and_then(Value::as_f64)
            .map(|value| value as f32)
            .unwrap_or(defaults.max_area_log2_error),
        min_overall_score: thresholds
            .get("min_overall_score")
            .and_then(Value::as_f64)
            .map(|value| value as f32)
            .unwrap_or(defaults.min_overall_score),
        max_seating_table_overlap_fraction: thresholds
            .get("max_seating_table_overlap_fraction")
            .and_then(Value::as_f64)
            .map(|value| value as f32)
            .unwrap_or(defaults.max_seating_table_overlap_fraction),
        max_seating_table_penetration_m: thresholds
            .get("max_seating_table_penetration_m")
            .and_then(Value::as_f64)
            .map(|value| value as f32)
            .unwrap_or(defaults.max_seating_table_penetration_m),
        max_seating_seating_overlap_fraction: thresholds
            .get("max_seating_seating_overlap_fraction")
            .and_then(Value::as_f64)
            .map(|value| value as f32)
            .unwrap_or(defaults.max_seating_seating_overlap_fraction),
        max_seating_seating_penetration_m: thresholds
            .get("max_seating_seating_penetration_m")
            .and_then(Value::as_f64)
            .map(|value| value as f32)
            .unwrap_or(defaults.max_seating_seating_penetration_m),
    }
}

pub(crate) fn feedback_project_delta_collisions(
    objects: &[Value],
    deltas: &mut [FeedbackDeltaDraft],
    thresholds: SceneFeedbackThresholds,
) {
    let mut footprints = objects
        .iter()
        .enumerate()
        .map(|(index, object)| {
            let rect = object
                .get("world_footprint")
                .and_then(json_footprint_rect)?;
            let kind = object
                .get("physical_kind")
                .and_then(Value::as_str)
                .and_then(feedback_physical_kind_from_str)
                .unwrap_or(FeedbackPhysicalKind::Other);
            Some(FeedbackFootprint {
                index,
                kind,
                rect: rect.translated(
                    deltas
                        .get(index)
                        .map(|delta| delta.translation_delta)
                        .unwrap_or([0.0, 0.0, 0.0]),
                ),
            })
        })
        .collect::<Vec<_>>();

    for _ in 0..4 {
        let mut changed = false;
        for left_index in 0..footprints.len() {
            let Some(left) = footprints[left_index] else {
                continue;
            };
            for right_index in (left_index + 1)..footprints.len() {
                let Some(right) = footprints[right_index] else {
                    continue;
                };
                let overlap_area = left.rect.overlap_area(right.rect);
                let signed_clearance_m = left.rect.signed_clearance(right.rect);
                if overlap_area <= 1.0e-5 && signed_clearance_m >= 0.0 {
                    continue;
                }
                let smaller_area = left.rect.area().min(right.rect.area()).max(1.0e-8);
                let overlap_fraction_smaller = (overlap_area / smaller_area).clamp(0.0, 1.0);
                match feedback_pair_relationship(left.kind, right.kind) {
                    "seating_table" => {
                        let (table, seating) = if left.kind == FeedbackPhysicalKind::Table {
                            (left, right)
                        } else {
                            (right, left)
                        };
                        if objects
                            .get(seating.index)
                            .is_some_and(feedback_json_object_is_open_sectional_seating)
                        {
                            continue;
                        }
                        let seating_center_inside_table =
                            table.rect.contains_point(seating.rect.center());
                        if seating_center_inside_table
                            || overlap_fraction_smaller
                                > thresholds.max_seating_table_overlap_fraction
                            || signed_clearance_m < -thresholds.max_seating_table_penetration_m
                        {
                            let source_bbox = objects
                                .get(seating.index)
                                .and_then(|object| object.get("expected_bbox"))
                                .and_then(json_array4)
                                .unwrap_or([0.0, 0.0, 1.0, 1.0]);
                            let delta = seating_table_outward_delta(table, seating, source_bbox);
                            if apply_projected_delta(
                                deltas,
                                &mut footprints,
                                seating.index,
                                delta,
                                1.25,
                            ) {
                                changed = true;
                            }
                        }
                    }
                    "seating_seating"
                        if overlap_fraction_smaller
                            > thresholds.max_seating_seating_overlap_fraction
                            || signed_clearance_m
                                < -thresholds.max_seating_seating_penetration_m =>
                    {
                        let [left_delta, right_delta] =
                            seating_pair_separation_delta(left, right, signed_clearance_m);
                        let left_changed = apply_projected_delta(
                            deltas,
                            &mut footprints,
                            left.index,
                            left_delta,
                            1.25,
                        );
                        let right_changed = apply_projected_delta(
                            deltas,
                            &mut footprints,
                            right.index,
                            right_delta,
                            1.25,
                        );
                        changed |= left_changed || right_changed;
                    }
                    _ => {}
                }
            }
        }
        if !changed {
            break;
        }
    }
}

pub(crate) fn apply_projected_delta(
    deltas: &mut [FeedbackDeltaDraft],
    footprints: &mut [Option<FeedbackFootprint>],
    index: usize,
    correction: [f32; 3],
    max_len: f32,
) -> bool {
    if correction[0].abs() + correction[2].abs() <= 1.0e-5 {
        return false;
    }
    let Some(delta) = deltas.get_mut(index) else {
        return false;
    };
    let next_delta = clamp_xz_delta(add3(delta.translation_delta, correction), max_len);
    let applied = [
        next_delta[0] - delta.translation_delta[0],
        next_delta[1] - delta.translation_delta[1],
        next_delta[2] - delta.translation_delta[2],
    ];
    if applied[0].abs() + applied[2].abs() <= 1.0e-5 {
        return false;
    }
    delta.translation_delta = next_delta;
    delta.ground_anchor_point = None;
    delta.ground_anchor_max_drift_m = None;
    if let Some(Some(footprint)) = footprints.get_mut(index) {
        footprint.rect = footprint.rect.translated(applied);
    }
    true
}

pub(crate) fn json_footprint_rect(value: &Value) -> Option<FootprintRect> {
    let rect = FootprintRect {
        min_x: value.get("min_x")?.as_f64()? as f32,
        min_z: value.get("min_z")?.as_f64()? as f32,
        max_x: value.get("max_x")?.as_f64()? as f32,
        max_z: value.get("max_z")?.as_f64()? as f32,
    };
    if rect.min_x.is_finite()
        && rect.min_z.is_finite()
        && rect.max_x.is_finite()
        && rect.max_z.is_finite()
        && rect.width() > 1.0e-4
        && rect.depth() > 1.0e-4
    {
        Some(rect)
    } else {
        None
    }
}

pub(crate) fn damped_feedback_scale_multiplier(object: &Value, raw_scale: f64) -> f64 {
    if feedback_json_object_is_table_like(object) {
        if object
            .get("source_edge_cropped")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            return 1.0;
        }
        return raw_scale.clamp(0.88, 1.18);
    }
    let raw_scale = raw_scale.clamp(0.82, 1.22);
    if !object
        .get("grounding_basis")
        .and_then(Value::as_str)
        .is_some_and(|basis| basis == "camera-ray-ground-plane")
    {
        return raw_scale;
    }
    let contact_error = object
        .get("contact_error")
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
        .unwrap_or(1.0);
    let center_error = object
        .get("center_error")
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
        .unwrap_or(1.0);
    let area_error = object
        .get("area_log2_error")
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
        .unwrap_or(0.0);
    if feedback_json_object_is_large_edge_cropped_furniture(object)
        && contact_error <= 0.05
        && (center_error > 0.16 || area_error > 1.0)
    {
        return raw_scale;
    }
    let weight = if contact_error <= 0.05 && center_error <= 0.08 {
        1.0
    } else if contact_error <= 0.10 && center_error <= 0.16 {
        0.55
    } else {
        0.25
    };
    (1.0 + (raw_scale - 1.0) * weight).clamp(0.88, 1.12)
}

pub(crate) fn feedback_json_object_is_table_like(object: &Value) -> bool {
    let descriptor = format!(
        "{} {}",
        object
            .get("object_id")
            .and_then(Value::as_str)
            .unwrap_or(""),
        object.get("label").and_then(Value::as_str).unwrap_or("")
    )
    .to_lowercase();
    descriptor.contains("table") || descriptor.contains("desk") || descriptor.contains("counter")
}

pub(crate) fn feedback_json_object_is_large_edge_cropped_furniture(object: &Value) -> bool {
    if !object
        .get("source_edge_cropped")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return false;
    }
    if object.get("instance_id").and_then(Value::as_str).is_some()
        || object
            .get("object_id")
            .and_then(Value::as_str)
            .is_some_and(|object_id| object_id.to_ascii_lowercase().contains("group"))
    {
        return false;
    }
    let descriptor = format!(
        "{} {}",
        object
            .get("object_id")
            .and_then(Value::as_str)
            .unwrap_or(""),
        object.get("label").and_then(Value::as_str).unwrap_or("")
    )
    .to_lowercase();
    descriptor.contains("sofa")
        || descriptor.contains("couch")
        || descriptor.contains("sectional")
        || descriptor.contains("loveseat")
        || descriptor.contains("banquette")
}

pub(crate) fn feedback_json_object_is_open_sectional_seating(object: &Value) -> bool {
    let descriptor = format!(
        "{} {}",
        object
            .get("object_id")
            .and_then(Value::as_str)
            .unwrap_or(""),
        object.get("label").and_then(Value::as_str).unwrap_or("")
    )
    .to_lowercase();
    feedback_descriptor_is_open_sectional_seating(&descriptor)
}

pub(crate) fn feedback_scale_group_key(object: &Value) -> Option<String> {
    object
        .get("cache_key")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            object
                .get("object_id")
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty())
        })
        .map(ToString::to_string)
}

#[cfg(test)]
pub(crate) fn apply_feedback_deltas_to_commands(
    commands: &[Value],
    deltas: &Value,
) -> Result<Vec<Value>, String> {
    apply_feedback_deltas_to_commands_with_policy(
        commands,
        deltas,
        SceneScalePolicy::AssetPreserving,
    )
}

pub(crate) fn apply_feedback_deltas_to_commands_with_policy(
    commands: &[Value],
    deltas: &Value,
    scale_policy: SceneScalePolicy,
) -> Result<Vec<Value>, String> {
    let object_deltas = deltas
        .get("objects")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut out = commands.to_vec();
    let mut spawn_index = 0usize;
    for command in &mut out {
        let command_type = command.get("type").and_then(Value::as_str).unwrap_or("");
        if command_type == "spawn_cached" || command_type == "spawn_path" {
            if let Some(delta) = object_deltas.get(spawn_index) {
                apply_object_delta_to_command(command, delta, scale_policy)?;
            }
            spawn_index += 1;
        } else if command_type == "set_camera" {
            let radius_multiplier = deltas
                .get("camera")
                .and_then(|camera| camera.get("radius_multiplier"))
                .and_then(Value::as_f64)
                .unwrap_or(1.0) as f32;
            if let Some(radius) = command.get("radius").and_then(Value::as_f64) {
                command["radius"] = json!((radius as f32 * radius_multiplier).clamp(1.0, 20.0));
            }
        }
    }
    normalize_reused_command_scales(&mut out);
    Ok(out)
}

pub(crate) fn normalize_reused_command_scales(commands: &mut [Value]) {
    let mut groups: HashMap<String, ([f32; 3], usize)> = HashMap::new();
    for command in commands.iter() {
        let command_type = command.get("type").and_then(Value::as_str).unwrap_or("");
        if command_type != "spawn_cached" && command_type != "spawn_path" {
            continue;
        }
        let Some(group_key) = command
            .get("cache_key")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
        else {
            continue;
        };
        let Some(scale) = command.get("scale").and_then(json_array3) else {
            continue;
        };
        let entry = groups.entry(group_key.to_string()).or_insert(([0.0; 3], 0));
        for (axis, value) in scale.iter().enumerate() {
            entry.0[axis] += value.abs().clamp(0.05, 20.0);
        }
        entry.1 += 1;
    }
    let repeated_scale = groups
        .into_iter()
        .filter_map(|(key, (sum, count))| {
            if count > 1 {
                Some((
                    key,
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
    for command in commands.iter_mut() {
        let command_type = command.get("type").and_then(Value::as_str).unwrap_or("");
        if command_type != "spawn_cached" && command_type != "spawn_path" {
            continue;
        }
        let Some(group_key) = command
            .get("cache_key")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
        else {
            continue;
        };
        let Some(scale) = repeated_scale.get(group_key).copied() else {
            continue;
        };
        command["scale"] = json!(scale);
    }
}

pub(crate) fn apply_object_delta_to_command(
    command: &mut Value,
    delta: &Value,
    scale_policy: SceneScalePolicy,
) -> Result<(), String> {
    let mut translation = command
        .get("translation")
        .and_then(json_array3)
        .unwrap_or([0.0, 0.0, 0.0]);
    let translation_delta = delta
        .get("translation_delta")
        .and_then(json_array3)
        .unwrap_or([0.0, 0.0, 0.0]);
    translation[0] += translation_delta[0];
    translation[1] += translation_delta[1];
    translation[2] += translation_delta[2];
    if let (Some(anchor), Some(max_drift_m)) = (
        delta.get("ground_anchor_point").and_then(json_array3),
        delta
            .get("ground_anchor_max_drift_m")
            .and_then(Value::as_f64)
            .filter(|value| value.is_finite() && *value > 0.0)
            .map(|value| value as f32),
    ) {
        clamp_translation_to_ground_anchor(&mut translation, anchor, max_drift_m);
    }
    command["translation"] = json!(translation);

    let multiplier = delta
        .get("scale_multiplier")
        .and_then(Value::as_f64)
        .unwrap_or(1.0) as f32;
    let axis_multiplier = delta.get("scale_multiplier_xyz").and_then(json_array3);
    let mut scale = command
        .get("scale")
        .and_then(json_array3)
        .unwrap_or([1.0, 1.0, 1.0]);
    if let Some(axis_multiplier) = axis_multiplier {
        for (value, axis_multiplier) in scale.iter_mut().zip(axis_multiplier) {
            *value = (*value * axis_multiplier).clamp(0.05, 20.0);
        }
    } else {
        for value in &mut scale {
            *value = (*value * multiplier).clamp(0.05, 20.0);
        }
    }
    scale = apply_feedback_scale_policy(scale, scale_policy);
    command["scale"] = json!(scale);

    let yaw_delta = delta
        .get("yaw_delta_degrees")
        .and_then(Value::as_f64)
        .unwrap_or(0.0) as f32;
    if yaw_delta.abs() > 1.0e-4 {
        let current_yaw = command
            .get("rotation")
            .and_then(json_array4)
            .map(quat_y_degrees)
            .unwrap_or(0.0);
        let target_yaw = normalize_degrees(current_yaw + yaw_delta.clamp(-30.0, 30.0));
        command["rotation"] = json!(quat_from_y_degrees(target_yaw));
    }
    Ok(())
}

fn apply_feedback_scale_policy(scale: [f32; 3], scale_policy: SceneScalePolicy) -> [f32; 3] {
    let Some(max_ratio) = scale_policy.max_xz_anisotropy() else {
        return scale;
    };
    if max_ratio <= 1.0 + f32::EPSILON {
        let uniform = ((scale[0].abs() + scale[1].abs() + scale[2].abs()) / 3.0).clamp(0.05, 20.0);
        return [uniform, uniform, uniform];
    }
    let x = scale[0].abs().clamp(0.05, 20.0);
    let z = scale[2].abs().clamp(0.05, 20.0);
    let ratio = (x.max(z) / x.min(z).max(1.0e-5)).max(1.0);
    if ratio <= max_ratio {
        return scale;
    }
    let area_scale = (x * z).sqrt().clamp(0.05, 20.0);
    let root_ratio = max_ratio.sqrt();
    let (next_x, next_z) = if x >= z {
        (area_scale * root_ratio, area_scale / root_ratio)
    } else {
        (area_scale / root_ratio, area_scale * root_ratio)
    };
    [
        next_x.clamp(0.05, 20.0).copysign(scale[0]),
        area_scale.clamp(0.05, 20.0).copysign(scale[1]),
        next_z.clamp(0.05, 20.0).copysign(scale[2]),
    ]
}

pub(crate) fn clamp_translation_to_ground_anchor(
    translation: &mut [f32; 3],
    anchor: [f32; 3],
    max_drift_m: f32,
) {
    let max_drift_m = max_drift_m.max(1.0e-4);
    let dx = translation[0] - anchor[0];
    let dz = translation[2] - anchor[2];
    let distance = (dx * dx + dz * dz).sqrt();
    if !distance.is_finite() || distance <= max_drift_m {
        return;
    }
    let scale = max_drift_m / distance;
    translation[0] = anchor[0] + dx * scale;
    translation[2] = anchor[2] + dz * scale;
}

pub(crate) fn feedback_bsn_from_commands(
    asset_bindings: &[SceneAssetBinding],
    grounded_layout: &GroundedSceneLayout,
    commands: &[Value],
) -> Result<String, String> {
    let mut out = String::from("synth_scene_v1 {\n");
    for asset in asset_bindings {
        out.push_str(&format!(
            "asset {} = \"generated:{}\";\n",
            asset.asset_id, asset.asset_id
        ));
    }
    out.push_str(&format!(
        "environment rug translation [{}] scale [{}] color [0.62,0.02,0.26];\n",
        fmt_feedback_vec3(grounded_layout.rug_center),
        fmt_feedback_vec3(grounded_layout.rug_scale)
    ));
    let mut spawn_index = 0usize;
    for command in commands {
        let command_type = command.get("type").and_then(Value::as_str).unwrap_or("");
        if command_type == "spawn_cached" || command_type == "spawn_path" {
            let asset_id = command_asset_id(asset_bindings, command)?;
            let translation = command
                .get("translation")
                .and_then(json_array3)
                .unwrap_or([0.0, 0.0, 0.0]);
            let scale = command
                .get("scale")
                .and_then(json_array3)
                .unwrap_or([1.0, 1.0, 1.0]);
            let rotation_y = command
                .get("rotation")
                .and_then(json_array4)
                .map(quat_y_degrees)
                .unwrap_or(0.0);
            let entity_id = grounded_layout
                .placements
                .get(spawn_index)
                .map(|placement| placement.entity_id.as_str())
                .unwrap_or("feedback_item");
            out.push_str(&format!(
                "spawn {} uses {} translation [{}] rotation_y {} scale [{}];\n",
                entity_id,
                asset_id,
                fmt_feedback_vec3(translation),
                fmt_feedback_num(rotation_y),
                fmt_feedback_vec3(scale)
            ));
            spawn_index += 1;
        } else if command_type == "set_camera" {
            let translation = command
                .get("translation")
                .and_then(json_array3)
                .unwrap_or(grounded_layout.camera.translation);
            let focus = command
                .get("focus")
                .and_then(json_array3)
                .unwrap_or(grounded_layout.camera.focus);
            let yaw = command
                .get("yaw")
                .and_then(Value::as_f64)
                .map(|value| value as f32)
                .or(grounded_layout.camera.yaw);
            let pitch = command
                .get("pitch")
                .and_then(Value::as_f64)
                .map(|value| value as f32)
                .or(grounded_layout.camera.pitch);
            let radius = command
                .get("radius")
                .and_then(Value::as_f64)
                .map(|value| value as f32)
                .or(grounded_layout.camera.radius);
            let vertical_fov = command
                .get("vertical_fov")
                .and_then(Value::as_f64)
                .map(|value| value as f32)
                .or(grounded_layout.camera.vertical_fov_degrees)
                .unwrap_or(72.0);
            out.push_str(&format!(
                "camera translation [{}] focus [{}]",
                fmt_feedback_vec3(translation),
                fmt_feedback_vec3(focus)
            ));
            if let Some(yaw) = yaw {
                out.push_str(&format!(" yaw {}", fmt_feedback_num(yaw)));
            }
            if let Some(pitch) = pitch {
                out.push_str(&format!(" pitch {}", fmt_feedback_num(pitch)));
            }
            if let Some(radius) = radius {
                out.push_str(&format!(" radius {}", fmt_feedback_num(radius)));
            }
            out.push_str(&format!(
                " vertical_fov {};\n",
                fmt_feedback_num(vertical_fov)
            ));
        }
    }
    out.push_str("}\n");
    Ok(out)
}

pub(crate) fn command_asset_id<'a>(
    assets: &'a [SceneAssetBinding],
    command: &Value,
) -> Result<&'a str, String> {
    if let Some(cache_key) = command.get("cache_key").and_then(Value::as_str)
        && let Some(asset) = assets
            .iter()
            .find(|asset| asset.cache_key.as_deref() == Some(cache_key))
    {
        return Ok(asset.asset_id.as_str());
    }
    if let Some(path) = command.get("path").and_then(Value::as_str)
        && let Some(asset) = assets.iter().find(|asset| {
            asset
                .path
                .as_ref()
                .is_some_and(|asset_path| asset_path == path)
        })
    {
        return Ok(asset.asset_id.as_str());
    }
    if let Some(cache_key) = command.get("cache_key").and_then(Value::as_str)
        && let Some(asset) = assets.iter().find(|asset| asset.asset_id == cache_key)
    {
        return Ok(asset.asset_id.as_str());
    }
    Err("feedback command references an unknown asset".to_string())
}

pub(crate) fn feedback_markdown_report(
    capture_root: &Path,
    profile: FeedbackThresholdProfile,
    accepted_iteration: Option<usize>,
    iterations: &[Value],
) -> String {
    let mut out = format!(
        "# Scene Feedback Report\n\nprofile: {:?}\naccepted_iteration: {:?}\n\n",
        profile, accepted_iteration
    );
    for iteration in iterations {
        let index = iteration
            .get("iteration")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let screenshot = iteration
            .get("screenshot")
            .and_then(Value::as_str)
            .unwrap_or("");
        let score = iteration
            .get("metrics")
            .and_then(|metrics| metrics.get("score"))
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        let selection_score = iteration
            .get("selection_score")
            .and_then(Value::as_f64)
            .unwrap_or(score);
        let passed = iteration
            .get("passed")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let physical = iteration
            .get("metrics")
            .and_then(|metrics| metrics.get("physical_layout"))
            .unwrap_or(&Value::Null);
        let physical_passed = physical
            .get("passed")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let projection_passed = iteration
            .get("metrics")
            .and_then(|metrics| metrics.get("projection_passed"))
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let rotation_passed = iteration
            .get("metrics")
            .and_then(|metrics| metrics.get("rotation_passed"))
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let object_pass_count = iteration
            .get("metrics")
            .and_then(|metrics| metrics.get("object_pass_count"))
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let object_count = iteration
            .get("metrics")
            .and_then(|metrics| metrics.get("object_count"))
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let rotation_pass_count = iteration
            .get("metrics")
            .and_then(|metrics| metrics.get("rotation_pass_count"))
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let hard_failures = physical
            .get("hard_failure_count")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let max_overlap = physical
            .get("max_overlap_fraction_smaller")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        let min_clearance = physical
            .get("min_signed_clearance_m")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        out.push_str(&format!(
            "## Iteration {index}\n\npassed: {passed}\nscore: {score:.4}\nselection_score: {selection_score:.4}\nprojection_passed: {projection_passed} ({object_pass_count}/{object_count})\nrotation_passed: {rotation_passed} ({rotation_pass_count}/{object_count})\nphysical_passed: {physical_passed}\nhard_overlap_failures: {hard_failures}\nmax_overlap_fraction_smaller: {max_overlap:.4}\nmin_signed_clearance_m: {min_clearance:.4}\n\n![iteration {index}]({})\n\n",
            path_relative_to(capture_root, Path::new(screenshot))
        ));
        if let Some(pairs) = physical.get("pairs").and_then(Value::as_array) {
            let failing_pairs = pairs
                .iter()
                .filter(|pair| {
                    pair.get("hard_failure")
                        .and_then(Value::as_bool)
                        .unwrap_or(false)
                })
                .take(6)
                .collect::<Vec<_>>();
            if !failing_pairs.is_empty() {
                out.push_str(
                    "| Pair | Relationship | Overlap Fraction | Clearance m | Reasons |\n",
                );
                out.push_str("| --- | --- | ---: | ---: | --- |\n");
                for pair in failing_pairs {
                    let left = pair
                        .get("left_instance_id")
                        .and_then(Value::as_str)
                        .or_else(|| pair.get("left_object_id").and_then(Value::as_str))
                        .unwrap_or("left");
                    let right = pair
                        .get("right_instance_id")
                        .and_then(Value::as_str)
                        .or_else(|| pair.get("right_object_id").and_then(Value::as_str))
                        .unwrap_or("right");
                    let relationship = pair
                        .get("relationship")
                        .and_then(Value::as_str)
                        .unwrap_or("object_object");
                    let fraction = pair
                        .get("overlap_fraction_smaller")
                        .and_then(Value::as_f64)
                        .unwrap_or(0.0);
                    let clearance = pair
                        .get("signed_clearance_m")
                        .and_then(Value::as_f64)
                        .unwrap_or(0.0);
                    let reasons = pair
                        .get("failure_reasons")
                        .and_then(Value::as_array)
                        .map(|items| {
                            items
                                .iter()
                                .filter_map(Value::as_str)
                                .collect::<Vec<_>>()
                                .join(", ")
                        })
                        .unwrap_or_default();
                    out.push_str(&format!(
                        "| {left} / {right} | {relationship} | {fraction:.4} | {clearance:.4} | {reasons} |\n"
                    ));
                }
                out.push('\n');
            }
        }
    }
    out
}

pub(crate) fn feedback_selection_score(metrics: &Value) -> f64 {
    let score = metrics
        .get("score")
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
        .unwrap_or(f64::NEG_INFINITY);
    let object_count = metrics
        .get("object_count")
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(1.0);
    let object_pass_fraction = metrics
        .get("object_pass_count")
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
        .unwrap_or(0.0)
        / object_count;
    let rotation_pass_fraction = metrics
        .get("rotation_pass_count")
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
        .unwrap_or(0.0)
        / object_count;
    let projection_bonus = metrics
        .get("projection_passed")
        .and_then(Value::as_bool)
        .map(|passed| if passed { 1.00 } else { 0.0 })
        .unwrap_or(0.0);
    let rotation_bonus = metrics
        .get("rotation_passed")
        .and_then(Value::as_bool)
        .map(|passed| if passed { 0.15 } else { 0.0 })
        .unwrap_or(0.0);
    let physical = metrics.get("physical_layout").unwrap_or(&Value::Null);
    let hard_failures = physical
        .get("hard_failure_count")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let max_overlap = physical
        .get("max_overlap_fraction_smaller")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    score
        + object_pass_fraction * 1.60
        + rotation_pass_fraction * 0.25
        + projection_bonus
        + rotation_bonus
        - hard_failures * 3.0
        - max_overlap * 1.5
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn feedback_iteration_context(
    iteration_index: usize,
    previous_iteration: Option<&Value>,
    previous_commands: Option<&[Value]>,
    current_commands: &[Value],
    screenshot_path: &Path,
    metrics: &Value,
    layout_delta: &Value,
    object_crops: &Value,
) -> Value {
    let previous = previous_iteration
        .map(feedback_iteration_snapshot_for_context)
        .unwrap_or(Value::Null);
    let rotation_selection_task = feedback_rotation_selection_task(metrics, object_crops);
    json!({
        "purpose": "scene-composition-iteration-context",
        "description": "Use previous/current screenshots plus transform deltas to correlate rendered changes with command-space changes.",
        "iteration": iteration_index,
        "previous_iteration": previous,
        "current_iteration": {
            "screenshot": screenshot_path.display().to_string(),
            "metrics_summary": feedback_metrics_summary(metrics),
            "transform_delta_to_next": layout_delta,
            "object_crops": object_crops,
            "rotation_selection_task": rotation_selection_task,
        },
        "command_transform_delta_from_previous": feedback_command_delta_summary(previous_commands, current_commands),
    })
}

pub(crate) fn feedback_iteration_snapshot_for_context(iteration: &Value) -> Value {
    json!({
        "iteration": iteration.get("iteration").cloned().unwrap_or(Value::Null),
        "screenshot": iteration.get("screenshot").cloned().unwrap_or(Value::Null),
        "metrics_summary": iteration
            .get("metrics")
            .map(feedback_metrics_summary)
            .unwrap_or(Value::Null),
        "transform_delta_to_next": iteration.get("layout_delta").cloned().unwrap_or(Value::Null),
        "object_crops": iteration
            .get("object_crops")
            .cloned()
            .unwrap_or(Value::Null),
        "rotation_selection_task": iteration
            .pointer("/iteration_context/current_iteration/rotation_selection_task")
            .cloned()
            .unwrap_or(Value::Null),
    })
}

pub(crate) fn feedback_rotation_selection_task(metrics: &Value, object_crops: &Value) -> Value {
    let Some(objects) = metrics.get("objects").and_then(Value::as_array) else {
        return Value::Null;
    };
    let crop_objects = object_crops
        .get("objects")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut tasks = Vec::new();
    for object in objects {
        let Some(rotation_selection) = object.get("rotation_selection") else {
            continue;
        };
        let index = object.get("index").and_then(Value::as_u64).unwrap_or(0);
        let crop = crop_objects
            .iter()
            .find(|crop| crop.get("index").and_then(Value::as_u64) == Some(index))
            .cloned()
            .unwrap_or(Value::Null);
        tasks.push(json!({
            "index": index,
            "object_id": object.get("object_id").cloned().unwrap_or(Value::Null),
            "instance_id": object.get("instance_id").cloned().unwrap_or(Value::Null),
            "label": object.get("label").cloned().unwrap_or(Value::Null),
            "source_crop": crop.get("source_crop").cloned().unwrap_or(Value::Null),
            "rendered_crop": crop.get("rendered_crop").cloned().unwrap_or(Value::Null),
            "source_bbox": object.get("expected_bbox").cloned().unwrap_or(Value::Null),
            "rendered_bbox": object.get("observed_bbox").cloned().unwrap_or(Value::Null),
            "current_yaw_degrees": object.get("current_yaw_degrees").cloned().unwrap_or(Value::Null),
            "canonical_yaw_degrees": object.get("canonical_yaw_degrees").cloned().unwrap_or(Value::Null),
            "rotation_selection": rotation_selection,
        }));
    }
    if tasks.is_empty() {
        return Value::Null;
    }
    json!({
        "purpose": "bounded-object-rotation-selection",
        "instruction": "Choose one candidate per object by comparing the source crop to the rendered crop. Return candidate_index values only; do not invent absolute yaw or transform values.",
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
        "objects": tasks,
    })
}

pub(crate) fn feedback_rotation_selection_prompt(task: &Value) -> String {
    let task_json = serde_json::to_string_pretty(task)
        .unwrap_or_else(|_| serde_json::to_string(task).unwrap_or_default());
    format!(
        "You are selecting bounded object yaw corrections for a 3D scene composition feedback loop.\n\
         Compare each source crop with its rendered crop. For each object, choose exactly one \
         candidate_index from the provided candidates. Do not invent absolute yaw, transforms, \
         positions, scales, or new candidate values. If the crop evidence is ambiguous, choose \
         the candidate closest to the rendered object's apparent orientation and lower confidence.\n\
         Source/render crop images are attached in source, rendered pairs in the same object order \
         as the JSON task.\n\nJSON task:\n{task_json}"
    )
}

pub(crate) fn feedback_rotation_selection_image_paths(task: &Value) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    let mut seen = HashSet::new();
    for object in task
        .get("objects")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        for key in ["source_crop", "rendered_crop"] {
            let Some(path) = object.get(key).and_then(Value::as_str) else {
                continue;
            };
            if path.trim().is_empty() || !seen.insert(path.to_string()) {
                continue;
            }
            paths.push(PathBuf::from(path));
        }
    }
    paths
}

pub(crate) fn apply_feedback_rotation_selection_response(
    metrics: &mut Value,
    response: &SceneRotationSelectionResponse,
) -> Value {
    let Some(objects) = metrics.get_mut("objects").and_then(Value::as_array_mut) else {
        return json!({
            "applied_count": 0,
            "ignored": ["metrics_missing_objects"],
            "objects": [],
        });
    };
    let mut response_by_index = HashMap::new();
    for selection in &response.objects {
        response_by_index.insert(selection.index, selection);
    }
    let mut applied = Vec::new();
    let mut ignored = Vec::new();
    for object in objects {
        let Some(index) = object
            .get("index")
            .and_then(Value::as_u64)
            .map(|value| value as usize)
        else {
            ignored.push(json!({
                "reason": "object_missing_index",
            }));
            continue;
        };
        let Some(selection) = response_by_index.get(&index) else {
            continue;
        };
        let Some(candidate) =
            feedback_rotation_candidate_for_object(object, selection.candidate_index)
        else {
            ignored.push(json!({
                "index": index,
                "candidate_index": selection.candidate_index,
                "reason": "candidate_index_not_available",
            }));
            continue;
        };
        let yaw_delta = candidate
            .get("yaw_delta_degrees")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        let target_yaw = candidate
            .get("candidate_yaw_degrees")
            .and_then(Value::as_f64)
            .unwrap_or_else(|| {
                let current_yaw = object
                    .get("current_yaw_degrees")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0) as f32;
                normalize_degrees(current_yaw + yaw_delta as f32) as f64
            });
        object["yaw_delta_degrees"] = json!(yaw_delta);
        object["target_yaw_degrees"] = json!(target_yaw);
        if let Some(rotation_selection) = object.get_mut("rotation_selection") {
            rotation_selection["selection_source"] = json!("openai_candidate_selector");
            rotation_selection["selected_candidate_index"] = json!(selection.candidate_index);
            rotation_selection["selected_yaw_delta_degrees"] = json!(yaw_delta);
            rotation_selection["selected_yaw_degrees"] = json!(target_yaw);
            rotation_selection["selector_result"] = json!({
                "confidence": selection.confidence,
                "rationale": selection.rationale,
            });
            if let Some(candidates) = rotation_selection
                .get_mut("candidates")
                .and_then(Value::as_array_mut)
            {
                for candidate in candidates {
                    let candidate_index = candidate
                        .get("candidate_index")
                        .and_then(Value::as_u64)
                        .map(|value| value as usize);
                    candidate["selected"] =
                        json!(candidate_index == Some(selection.candidate_index));
                }
            }
        }
        applied.push(json!({
            "index": index,
            "candidate_index": selection.candidate_index,
            "yaw_delta_degrees": yaw_delta,
            "target_yaw_degrees": target_yaw,
            "confidence": selection.confidence,
        }));
    }
    json!({
        "applied_count": applied.len(),
        "ignored": ignored,
        "objects": applied,
    })
}

fn feedback_rotation_candidate_for_object(object: &Value, candidate_index: usize) -> Option<Value> {
    object
        .get("rotation_selection")
        .and_then(|selection| selection.get("candidates"))
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

pub(crate) fn feedback_rotation_selection(
    current_yaw_degrees: f32,
    target_delta_degrees: f32,
    basis: &'static str,
) -> Value {
    let current_yaw_degrees = normalize_degrees(current_yaw_degrees);
    let target_delta_degrees = target_delta_degrees.clamp(-30.0, 30.0);
    let candidates = feedback_rotation_candidate_deltas(target_delta_degrees);
    let selected_index = candidates
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| {
            let left_error = (left.0 - target_delta_degrees).abs();
            let right_error = (right.0 - target_delta_degrees).abs();
            left_error
                .partial_cmp(&right_error)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(index, _)| index)
        .unwrap_or(0);
    let selected_delta = candidates
        .get(selected_index)
        .map(|(delta, _)| *delta)
        .unwrap_or(0.0);
    let candidate_values = candidates
        .iter()
        .enumerate()
        .map(|(index, (delta, search_level))| {
            json!({
                "candidate_index": index,
                "yaw_delta_degrees": *delta,
                "candidate_yaw_degrees": normalize_degrees(current_yaw_degrees + *delta),
                "selected": index == selected_index,
                "search_level": search_level,
            })
        })
        .collect::<Vec<_>>();
    json!({
        "selection_source": "deterministic_closest_to_feedback_target",
        "selection_basis": basis,
        "search_strategy": "bounded-coarse-to-fine-relative-yaw",
        "instruction": "Compare source/rendered crops and choose candidate_index only; do not provide absolute yaw or transform values.",
        "current_yaw_degrees": current_yaw_degrees,
        "target_delta_degrees": target_delta_degrees,
        "selected_candidate_index": selected_index,
        "selected_yaw_delta_degrees": selected_delta,
        "selected_yaw_degrees": normalize_degrees(current_yaw_degrees + selected_delta),
        "candidates": candidate_values,
    })
}

fn feedback_rotation_candidate_deltas(target_delta_degrees: f32) -> Vec<(f32, &'static str)> {
    let target_delta_degrees = target_delta_degrees.clamp(-30.0, 30.0);
    let magnitude = target_delta_degrees.abs();
    let span = if magnitude > 1.0e-3 {
        (magnitude * 1.5).clamp(8.0, 30.0)
    } else {
        12.0
    };
    let mut deltas = Vec::new();
    push_unique_rotation_delta(&mut deltas, -span, "coarse");
    push_unique_rotation_delta(&mut deltas, -span * 0.5, "mid");
    push_unique_rotation_delta(&mut deltas, 0.0, "coarse");
    push_unique_rotation_delta(&mut deltas, span * 0.5, "mid");
    push_unique_rotation_delta(&mut deltas, span, "coarse");
    let fine_span = (span * 0.25).clamp(2.0, 8.0);
    push_unique_rotation_delta(&mut deltas, target_delta_degrees - fine_span, "fine-left");
    push_unique_rotation_delta(
        &mut deltas,
        target_delta_degrees - fine_span * 0.5,
        "fine-mid-left",
    );
    push_unique_rotation_delta(&mut deltas, target_delta_degrees, "feedback-target");
    push_unique_rotation_delta(
        &mut deltas,
        target_delta_degrees + fine_span * 0.5,
        "fine-mid-right",
    );
    push_unique_rotation_delta(&mut deltas, target_delta_degrees + fine_span, "fine-right");
    deltas.sort_by(|left, right| {
        left.0
            .partial_cmp(&right.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    deltas
}

fn push_unique_rotation_delta(
    deltas: &mut Vec<(f32, &'static str)>,
    delta_degrees: f32,
    search_level: &'static str,
) {
    if !delta_degrees.is_finite() {
        return;
    }
    let delta_degrees = delta_degrees.clamp(-30.0, 30.0);
    if deltas
        .iter()
        .any(|(existing, _)| (*existing - delta_degrees).abs() <= 1.0)
    {
        return;
    }
    deltas.push((delta_degrees, search_level));
}

pub(crate) fn feedback_object_crops(
    iteration_dir: &Path,
    source_scene_path: &Path,
    screenshot_path: &Path,
    metrics: &Value,
) -> Value {
    let objects = metrics
        .get("objects")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if objects.is_empty() {
        return Value::Null;
    }
    let crop_dir = iteration_dir.join("objects");
    if let Err(err) = fs::create_dir_all(&crop_dir) {
        return json!({ "error": format!("create object crop directory {}: {err}", crop_dir.display()) });
    }
    let source_image = match image::open(source_scene_path) {
        Ok(image) => Some(image),
        Err(err) => {
            return json!({
                "error": format!("open source image {}: {err}", source_scene_path.display()),
            });
        }
    };
    let rendered_image = match image::open(screenshot_path) {
        Ok(image) => Some(image),
        Err(err) => {
            return json!({
                "error": format!("open rendered screenshot {}: {err}", screenshot_path.display()),
            });
        }
    };
    let source_image = source_image.unwrap();
    let rendered_image = rendered_image.unwrap();
    let mut crop_reports = Vec::new();
    for object in objects {
        let index = object
            .get("index")
            .and_then(Value::as_u64)
            .unwrap_or(crop_reports.len() as u64) as usize;
        let object_id = object
            .get("object_id")
            .and_then(Value::as_str)
            .unwrap_or("object");
        let label = object.get("label").and_then(Value::as_str).unwrap_or("");
        let name = sanitize_feedback_crop_name(index, object_id);
        let source_bbox = object.get("expected_bbox").and_then(json_array4);
        let rendered_bbox = object.get("observed_bbox").and_then(json_array4);
        let source_path = crop_dir.join(format!("{name}_source.png"));
        let rendered_path = crop_dir.join(format!("{name}_render.png"));
        let mut report = json!({
            "index": index,
            "object_id": object_id,
            "label": label,
            "source_bbox": source_bbox,
            "rendered_bbox": rendered_bbox,
        });
        if let Some(bbox) = source_bbox {
            match write_feedback_crop(&source_image, bbox, &source_path) {
                Ok(()) => report["source_crop"] = json!(source_path.display().to_string()),
                Err(err) => report["source_crop_error"] = json!(err),
            }
        }
        if let Some(bbox) = rendered_bbox.filter(|bbox| bbox_area(*bbox) > 1.0e-5) {
            match write_feedback_crop(&rendered_image, bbox, &rendered_path) {
                Ok(()) => report["rendered_crop"] = json!(rendered_path.display().to_string()),
                Err(err) => report["rendered_crop_error"] = json!(err),
            }
        }
        crop_reports.push(report);
    }
    json!({
        "dir": crop_dir.display().to_string(),
        "objects": crop_reports,
    })
}

fn write_feedback_crop(
    image: &image::DynamicImage,
    bbox: [f32; 4],
    output_path: &Path,
) -> Result<(), String> {
    let width = image.width().max(1);
    let height = image.height().max(1);
    let pad_x = ((bbox[2] - bbox[0]).abs() * 0.12).max(0.02);
    let pad_y = ((bbox[3] - bbox[1]).abs() * 0.12).max(0.02);
    let x0 = ((bbox[0] - pad_x).clamp(0.0, 1.0) * width as f32).floor() as u32;
    let y0 = ((bbox[1] - pad_y).clamp(0.0, 1.0) * height as f32).floor() as u32;
    let x1 = ((bbox[2] + pad_x).clamp(0.0, 1.0) * width as f32).ceil() as u32;
    let y1 = ((bbox[3] + pad_y).clamp(0.0, 1.0) * height as f32).ceil() as u32;
    let crop_width = x1
        .saturating_sub(x0)
        .max(1)
        .min(width.saturating_sub(x0).max(1));
    let crop_height = y1
        .saturating_sub(y0)
        .max(1)
        .min(height.saturating_sub(y0).max(1));
    let crop = image.crop_imm(x0, y0, crop_width, crop_height);
    crop.save(output_path)
        .map_err(|err| format!("write feedback crop {}: {err}", output_path.display()))
}

fn sanitize_feedback_crop_name(index: usize, value: &str) -> String {
    let mut out = format!("{index:02}_");
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    out
}

pub(crate) fn feedback_metrics_summary(metrics: &Value) -> Value {
    json!({
        "passed": metrics.get("passed").cloned().unwrap_or(Value::Null),
        "score": metrics.get("score").cloned().unwrap_or(Value::Null),
        "projection_passed": metrics.get("projection_passed").cloned().unwrap_or(Value::Null),
        "physical_passed": metrics.get("physical_passed").cloned().unwrap_or(Value::Null),
        "rotation_passed": metrics.get("rotation_passed").cloned().unwrap_or(Value::Null),
        "object_count": metrics.get("object_count").cloned().unwrap_or(Value::Null),
        "object_pass_count": metrics.get("object_pass_count").cloned().unwrap_or(Value::Null),
        "rotation_pass_count": metrics.get("rotation_pass_count").cloned().unwrap_or(Value::Null),
        "physical_layout": metrics.get("physical_layout").cloned().unwrap_or(Value::Null),
    })
}

pub(crate) fn feedback_command_delta_summary(
    previous_commands: Option<&[Value]>,
    current_commands: &[Value],
) -> Value {
    let Some(previous_commands) = previous_commands else {
        return Value::Null;
    };
    let previous_spawns = feedback_spawn_command_transforms(previous_commands);
    let current_spawns = feedback_spawn_command_transforms(current_commands);
    let object_count = previous_spawns.len().min(current_spawns.len());
    let objects = (0..object_count)
        .map(|index| {
            let previous = &previous_spawns[index];
            let current = &current_spawns[index];
            json!({
                "index": index,
                "identity": current.identity,
                "previous_translation": previous.translation,
                "current_translation": current.translation,
                "translation_delta": sub3(current.translation, previous.translation),
                "previous_scale": previous.scale,
                "current_scale": current.scale,
                "scale_ratio": safe_div3(current.scale, previous.scale),
                "previous_yaw_degrees": previous.yaw_degrees,
                "current_yaw_degrees": current.yaw_degrees,
                "yaw_delta_degrees": normalize_degrees(current.yaw_degrees - previous.yaw_degrees),
            })
        })
        .collect::<Vec<_>>();
    json!({
        "object_count": object_count,
        "objects": objects,
        "camera": feedback_camera_command_delta(previous_commands, current_commands),
    })
}

#[derive(Clone, Debug)]
pub(crate) struct FeedbackCommandTransform {
    identity: String,
    translation: [f32; 3],
    scale: [f32; 3],
    yaw_degrees: f32,
}

pub(crate) fn feedback_spawn_command_transforms(
    commands: &[Value],
) -> Vec<FeedbackCommandTransform> {
    commands
        .iter()
        .filter_map(|command| {
            let command_type = command.get("type").and_then(Value::as_str)?;
            if command_type != "spawn_cached" && command_type != "spawn_path" {
                return None;
            }
            let identity = command
                .get("cache_key")
                .and_then(Value::as_str)
                .or_else(|| command.get("path").and_then(Value::as_str))
                .unwrap_or("<unknown>")
                .to_string();
            Some(FeedbackCommandTransform {
                identity,
                translation: command
                    .get("translation")
                    .and_then(json_array3)
                    .unwrap_or([0.0, 0.0, 0.0]),
                scale: command
                    .get("scale")
                    .and_then(json_array3)
                    .unwrap_or([1.0, 1.0, 1.0]),
                yaw_degrees: command
                    .get("rotation")
                    .and_then(json_array4)
                    .map(quat_y_degrees)
                    .unwrap_or(0.0),
            })
        })
        .collect()
}

pub(crate) fn feedback_camera_command_delta(
    previous_commands: &[Value],
    current_commands: &[Value],
) -> Value {
    let previous = previous_commands
        .iter()
        .find(|command| command.get("type").and_then(Value::as_str) == Some("set_camera"));
    let current = current_commands
        .iter()
        .find(|command| command.get("type").and_then(Value::as_str) == Some("set_camera"));
    let (Some(previous), Some(current)) = (previous, current) else {
        return Value::Null;
    };
    let previous_radius = previous
        .get("radius")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let current_radius = current.get("radius").and_then(Value::as_f64).unwrap_or(0.0);
    json!({
        "previous_radius": previous_radius,
        "current_radius": current_radius,
        "radius_delta": current_radius - previous_radius,
        "previous_focus": previous.get("focus").cloned().unwrap_or(Value::Null),
        "current_focus": current.get("focus").cloned().unwrap_or(Value::Null),
        "previous_translation": previous.get("translation").cloned().unwrap_or(Value::Null),
        "current_translation": current.get("translation").cloned().unwrap_or(Value::Null),
    })
}

pub(crate) fn safe_div3(numerator: [f32; 3], denominator: [f32; 3]) -> [f32; 3] {
    [
        safe_div(numerator[0], denominator[0]),
        safe_div(numerator[1], denominator[1]),
        safe_div(numerator[2], denominator[2]),
    ]
}

pub(crate) fn safe_div(numerator: f32, denominator: f32) -> f32 {
    if denominator.abs() > 1.0e-6 {
        numerator / denominator
    } else {
        1.0
    }
}

pub(crate) fn path_relative_to(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .display()
        .to_string()
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct FeedbackFootprint {
    pub(crate) index: usize,
    pub(crate) kind: FeedbackPhysicalKind,
    pub(crate) rect: FootprintRect,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FeedbackPhysicalKind {
    Table,
    Seating,
    Other,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct FootprintRect {
    pub(crate) min_x: f32,
    pub(crate) min_z: f32,
    pub(crate) max_x: f32,
    pub(crate) max_z: f32,
}

impl FootprintRect {
    fn from_aabb(min: [f32; 3], max: [f32; 3]) -> Option<Self> {
        let rect = Self {
            min_x: min[0].min(max[0]),
            min_z: min[2].min(max[2]),
            max_x: min[0].max(max[0]),
            max_z: min[2].max(max[2]),
        };
        if rect.min_x.is_finite()
            && rect.min_z.is_finite()
            && rect.max_x.is_finite()
            && rect.max_z.is_finite()
            && rect.width() > 1.0e-4
            && rect.depth() > 1.0e-4
        {
            Some(rect)
        } else {
            None
        }
    }

    fn width(self) -> f32 {
        (self.max_x - self.min_x).max(0.0)
    }

    fn depth(self) -> f32 {
        (self.max_z - self.min_z).max(0.0)
    }

    fn area(self) -> f32 {
        (self.width() * self.depth()).max(1.0e-8)
    }

    fn center(self) -> [f32; 2] {
        [
            (self.min_x + self.max_x) * 0.5,
            (self.min_z + self.max_z) * 0.5,
        ]
    }

    fn half_width(self) -> f32 {
        self.width() * 0.5
    }

    fn half_depth(self) -> f32 {
        self.depth() * 0.5
    }

    fn contains_point(self, point: [f32; 2]) -> bool {
        point[0] >= self.min_x
            && point[0] <= self.max_x
            && point[1] >= self.min_z
            && point[1] <= self.max_z
    }

    fn overlap_extents(self, other: Self) -> [f32; 2] {
        [
            (self.max_x.min(other.max_x) - self.min_x.max(other.min_x)).max(0.0),
            (self.max_z.min(other.max_z) - self.min_z.max(other.min_z)).max(0.0),
        ]
    }

    fn overlap_area(self, other: Self) -> f32 {
        let [x, z] = self.overlap_extents(other);
        x * z
    }

    pub(crate) fn signed_clearance(self, other: Self) -> f32 {
        let [overlap_x, overlap_z] = self.overlap_extents(other);
        if overlap_x > 0.0 && overlap_z > 0.0 {
            return -overlap_x.min(overlap_z);
        }
        let dx = if self.max_x < other.min_x {
            other.min_x - self.max_x
        } else if other.max_x < self.min_x {
            self.min_x - other.max_x
        } else {
            0.0
        };
        let dz = if self.max_z < other.min_z {
            other.min_z - self.max_z
        } else if other.max_z < self.min_z {
            self.min_z - other.max_z
        } else {
            0.0
        };
        (dx * dx + dz * dz).sqrt()
    }

    pub(crate) fn translated(self, delta: [f32; 3]) -> Self {
        Self {
            min_x: self.min_x + delta[0],
            max_x: self.max_x + delta[0],
            min_z: self.min_z + delta[2],
            max_z: self.max_z + delta[2],
        }
    }
}

#[derive(Debug)]
pub(crate) struct FeedbackPhysicalLayout {
    pub(crate) pairs: Vec<Value>,
    pub(crate) corrections: HashMap<usize, [f32; 3]>,
    pub(crate) object_failures: HashMap<usize, Vec<String>>,
    pub(crate) hard_failure_count: usize,
    pub(crate) warning_count: usize,
    pub(crate) object_failure_count: usize,
    pub(crate) max_overlap_fraction_smaller: f32,
    pub(crate) min_signed_clearance_m: f32,
}

pub(crate) fn feedback_projected_footprints(
    placements: &[GroundedScenePlacement],
    projected: &[Value],
) -> Vec<Option<FeedbackFootprint>> {
    placements
        .iter()
        .enumerate()
        .map(|(index, placement)| {
            let aabb = projected.get(index)?.get("world_aabb")?;
            let min = aabb.get("min").and_then(json_array3)?;
            let max = aabb.get("max").and_then(json_array3)?;
            Some(FeedbackFootprint {
                index,
                kind: feedback_physical_kind(placement),
                rect: FootprintRect::from_aabb(min, max)?,
            })
        })
        .collect()
}

pub(crate) fn feedback_physical_layout(
    placements: &[GroundedScenePlacement],
    footprints: &[Option<FeedbackFootprint>],
    thresholds: SceneFeedbackThresholds,
) -> FeedbackPhysicalLayout {
    let mut pairs = Vec::new();
    let mut corrections: HashMap<usize, [f32; 3]> = HashMap::new();
    let mut object_failures: HashMap<usize, Vec<String>> = HashMap::new();
    let mut hard_failure_count = 0usize;
    let mut warning_count = 0usize;
    let mut max_overlap_fraction_smaller = 0.0f32;
    let mut min_signed_clearance_m = f32::INFINITY;
    for left_index in 0..footprints.len() {
        let Some(left) = footprints[left_index] else {
            continue;
        };
        for right in footprints.iter().skip(left_index + 1) {
            let Some(right) = *right else {
                continue;
            };
            let overlap_area = left.rect.overlap_area(right.rect);
            let signed_clearance_m = left.rect.signed_clearance(right.rect);
            min_signed_clearance_m = min_signed_clearance_m.min(signed_clearance_m);
            if overlap_area <= 1.0e-5 && signed_clearance_m >= 0.0 {
                continue;
            }
            let smaller_area = left.rect.area().min(right.rect.area()).max(1.0e-8);
            let overlap_fraction_smaller = (overlap_area / smaller_area).clamp(0.0, 1.0);
            max_overlap_fraction_smaller =
                max_overlap_fraction_smaller.max(overlap_fraction_smaller);
            let relationship = feedback_pair_relationship(left.kind, right.kind);
            let seating_table = relationship == "seating_table";
            let seating_seating = relationship == "seating_seating";
            let mut reasons = Vec::new();
            if seating_table {
                let (table, seating) = if left.kind == FeedbackPhysicalKind::Table {
                    (left, right)
                } else {
                    (right, left)
                };
                let seating_center_inside_table = table.rect.contains_point(seating.rect.center());
                if seating_center_inside_table {
                    reasons.push("seating_center_inside_table");
                }
                if overlap_fraction_smaller > thresholds.max_seating_table_overlap_fraction {
                    reasons.push("seating_table_overlap_fraction");
                }
                if signed_clearance_m < -thresholds.max_seating_table_penetration_m {
                    reasons.push("seating_table_penetration");
                }
                let open_sectional_table_pair =
                    feedback_open_sectional_table_pair(placements, left, right);
                if !reasons.is_empty() {
                    if open_sectional_table_pair {
                        reasons.clear();
                    } else {
                        let delta = seating_table_outward_delta(
                            table,
                            seating,
                            placements[seating.index].source_bbox,
                        );
                        accumulate_feedback_delta(&mut corrections, seating.index, delta, 1.10);
                    }
                }
            } else if seating_seating {
                if overlap_fraction_smaller > thresholds.max_seating_seating_overlap_fraction {
                    reasons.push("seating_seating_overlap_fraction");
                }
                if signed_clearance_m < -thresholds.max_seating_seating_penetration_m {
                    reasons.push("seating_seating_penetration");
                }
                if !reasons.is_empty() {
                    let [left_delta, right_delta] =
                        seating_pair_separation_delta(left, right, signed_clearance_m);
                    accumulate_feedback_delta(&mut corrections, left.index, left_delta, 0.55);
                    accumulate_feedback_delta(&mut corrections, right.index, right_delta, 0.55);
                }
            }

            let hard_failure = !reasons.is_empty();
            if hard_failure {
                hard_failure_count += 1;
                let left_message = feedback_physical_failure_message(
                    relationship,
                    &placements[right.index],
                    overlap_fraction_smaller,
                    signed_clearance_m,
                    &reasons,
                );
                let right_message = feedback_physical_failure_message(
                    relationship,
                    &placements[left.index],
                    overlap_fraction_smaller,
                    signed_clearance_m,
                    &reasons,
                );
                object_failures
                    .entry(left.index)
                    .or_default()
                    .push(left_message);
                object_failures
                    .entry(right.index)
                    .or_default()
                    .push(right_message);
            } else {
                warning_count += 1;
            }

            pairs.push(json!({
                "left_index": left.index,
                "right_index": right.index,
                "left_object_id": placements[left.index].object_id,
                "right_object_id": placements[right.index].object_id,
                "left_instance_id": placements[left.index].instance_id,
                "right_instance_id": placements[right.index].instance_id,
                "relationship": relationship,
                "overlap_area": overlap_area,
                "overlap_fraction_smaller": overlap_fraction_smaller,
                "signed_clearance_m": signed_clearance_m,
                "hard_failure": hard_failure,
                "failure_reasons": reasons,
            }));
        }
    }

    FeedbackPhysicalLayout {
        pairs,
        corrections,
        object_failure_count: object_failures.len(),
        object_failures,
        hard_failure_count,
        warning_count,
        max_overlap_fraction_smaller,
        min_signed_clearance_m: if min_signed_clearance_m.is_finite() {
            min_signed_clearance_m
        } else {
            0.0
        },
    }
}

pub(crate) fn feedback_predictive_physical_delta(
    index: usize,
    placement: &GroundedScenePlacement,
    proposed_delta: [f32; 3],
    footprints: &[Option<FeedbackFootprint>],
    thresholds: SceneFeedbackThresholds,
) -> [f32; 3] {
    let Some(current) = footprints.get(index).and_then(|footprint| *footprint) else {
        return [0.0, 0.0, 0.0];
    };
    if current.kind != FeedbackPhysicalKind::Seating {
        return [0.0, 0.0, 0.0];
    }
    let predicted = FeedbackFootprint {
        rect: current.rect.translated(proposed_delta),
        ..current
    };
    let mut correction = [0.0, 0.0, 0.0];
    for other in footprints.iter().flatten().copied() {
        if other.index == index {
            continue;
        }
        let overlap_area = predicted.rect.overlap_area(other.rect);
        let signed_clearance_m = predicted.rect.signed_clearance(other.rect);
        if overlap_area <= 1.0e-5 && signed_clearance_m >= 0.0 {
            continue;
        }
        let smaller_area = predicted.rect.area().min(other.rect.area()).max(1.0e-8);
        let overlap_fraction_smaller = (overlap_area / smaller_area).clamp(0.0, 1.0);
        match feedback_pair_relationship(predicted.kind, other.kind) {
            "seating_table" => {
                let table = other;
                if feedback_open_sectional_descriptor(placement) {
                    continue;
                }
                let seating_center_inside_table =
                    table.rect.contains_point(predicted.rect.center());
                if seating_center_inside_table
                    || overlap_fraction_smaller > thresholds.max_seating_table_overlap_fraction
                    || signed_clearance_m < -thresholds.max_seating_table_penetration_m
                {
                    correction = add3(
                        correction,
                        seating_table_outward_delta(table, predicted, placement.source_bbox),
                    );
                }
            }
            "seating_seating"
                if overlap_fraction_smaller > thresholds.max_seating_seating_overlap_fraction
                    || signed_clearance_m < -thresholds.max_seating_seating_penetration_m =>
            {
                let [self_delta, _other_delta] =
                    seating_pair_separation_delta(predicted, other, signed_clearance_m);
                correction = add3(correction, self_delta);
            }
            _ => {}
        }
    }
    clamp_xz_delta(correction, 0.95)
}

pub(crate) fn feedback_physical_kind(placement: &GroundedScenePlacement) -> FeedbackPhysicalKind {
    let descriptor = format!("{} {}", placement.object_id, placement.label).to_lowercase();
    if descriptor.contains("table") || descriptor.contains("desk") || descriptor.contains("counter")
    {
        FeedbackPhysicalKind::Table
    } else if descriptor.contains("chair")
        || descriptor.contains("seat")
        || descriptor.contains("sofa")
        || descriptor.contains("couch")
        || descriptor.contains("stool")
    {
        FeedbackPhysicalKind::Seating
    } else {
        FeedbackPhysicalKind::Other
    }
}

pub(crate) fn feedback_open_sectional_table_pair(
    placements: &[GroundedScenePlacement],
    left: FeedbackFootprint,
    right: FeedbackFootprint,
) -> bool {
    let seating_index = match (left.kind, right.kind) {
        (FeedbackPhysicalKind::Table, FeedbackPhysicalKind::Seating) => right.index,
        (FeedbackPhysicalKind::Seating, FeedbackPhysicalKind::Table) => left.index,
        _ => return false,
    };
    placements
        .get(seating_index)
        .is_some_and(feedback_open_sectional_descriptor)
}

pub(crate) fn feedback_open_sectional_descriptor(placement: &GroundedScenePlacement) -> bool {
    let descriptor = format!("{} {}", placement.object_id, placement.label).to_lowercase();
    feedback_descriptor_is_open_sectional_seating(&descriptor)
}

pub(crate) fn feedback_descriptor_is_open_sectional_seating(descriptor: &str) -> bool {
    let sofa_like = descriptor.contains("sofa")
        || descriptor.contains("couch")
        || descriptor.contains("sectional")
        || descriptor.contains("loveseat");
    let open_like = descriptor.contains("open")
        || descriptor.contains("crescent")
        || descriptor.contains("curved")
        || descriptor.contains("sectional")
        || descriptor.contains("u-shape")
        || descriptor.contains("u shaped")
        || descriptor.contains("u-shaped");
    sofa_like && open_like
}

pub(crate) fn feedback_physical_kind_str(kind: FeedbackPhysicalKind) -> &'static str {
    match kind {
        FeedbackPhysicalKind::Table => "table",
        FeedbackPhysicalKind::Seating => "seating",
        FeedbackPhysicalKind::Other => "other",
    }
}

pub(crate) fn feedback_physical_kind_from_str(value: &str) -> Option<FeedbackPhysicalKind> {
    match value {
        "table" => Some(FeedbackPhysicalKind::Table),
        "seating" => Some(FeedbackPhysicalKind::Seating),
        "other" => Some(FeedbackPhysicalKind::Other),
        _ => None,
    }
}

pub(crate) fn feedback_pair_relationship(
    left: FeedbackPhysicalKind,
    right: FeedbackPhysicalKind,
) -> &'static str {
    match (left, right) {
        (FeedbackPhysicalKind::Table, FeedbackPhysicalKind::Seating)
        | (FeedbackPhysicalKind::Seating, FeedbackPhysicalKind::Table) => "seating_table",
        (FeedbackPhysicalKind::Seating, FeedbackPhysicalKind::Seating) => "seating_seating",
        (FeedbackPhysicalKind::Table, _) | (_, FeedbackPhysicalKind::Table) => "table_object",
        _ => "object_object",
    }
}

pub(crate) fn seating_table_outward_delta(
    table: FeedbackFootprint,
    seating: FeedbackFootprint,
    source_bbox: [f32; 4],
) -> [f32; 3] {
    let table_center = table.rect.center();
    let seating_center = seating.rect.center();
    let norm_x = (seating_center[0] - table_center[0]) / table.rect.half_width().max(1.0e-5);
    let norm_z = (seating_center[1] - table_center[1]) / table.rect.half_depth().max(1.0e-5);
    let use_x_axis = if norm_x.abs() + norm_z.abs() <= 1.0e-4 {
        true
    } else {
        norm_x.abs() >= norm_z.abs()
    };
    let clearance = 0.12;
    if use_x_axis {
        let sign = if norm_x.abs() <= 1.0e-4 {
            if bbox_center(source_bbox)[0] < 0.5 {
                -1.0
            } else {
                1.0
            }
        } else {
            norm_x.signum()
        };
        let target = if sign >= 0.0 {
            table.rect.max_x + seating.rect.half_width() + clearance
        } else {
            table.rect.min_x - seating.rect.half_width() - clearance
        };
        [(target - seating_center[0]).clamp(-0.90, 0.90), 0.0, 0.0]
    } else {
        let sign = if norm_z.abs() <= 1.0e-4 {
            if bbox_center(source_bbox)[1] < 0.5 {
                -1.0
            } else {
                1.0
            }
        } else {
            norm_z.signum()
        };
        let target = if sign >= 0.0 {
            table.rect.max_z + seating.rect.half_depth() + clearance
        } else {
            table.rect.min_z - seating.rect.half_depth() - clearance
        };
        [0.0, 0.0, (target - seating_center[1]).clamp(-0.90, 0.90)]
    }
}

pub(crate) fn seating_pair_separation_delta(
    left: FeedbackFootprint,
    right: FeedbackFootprint,
    signed_clearance_m: f32,
) -> [[f32; 3]; 2] {
    let left_center = left.rect.center();
    let right_center = right.rect.center();
    let dx = left_center[0] - right_center[0];
    let dz = left_center[1] - right_center[1];
    let len = (dx * dx + dz * dz).sqrt();
    let direction = if len > 1.0e-5 {
        [dx / len, dz / len]
    } else if left.index <= right.index {
        [-1.0, 0.0]
    } else {
        [1.0, 0.0]
    };
    let step = ((-signed_clearance_m).max(0.0) + 0.08).clamp(0.04, 0.32) * 0.5;
    [
        [direction[0] * step, 0.0, direction[1] * step],
        [-direction[0] * step, 0.0, -direction[1] * step],
    ]
}

pub(crate) fn accumulate_feedback_delta(
    corrections: &mut HashMap<usize, [f32; 3]>,
    index: usize,
    delta: [f32; 3],
    max_len: f32,
) {
    let current = corrections.entry(index).or_insert([0.0, 0.0, 0.0]);
    *current = clamp_xz_delta(add3(*current, delta), max_len);
}

pub(crate) fn feedback_physical_failure_message(
    relationship: &str,
    other: &GroundedScenePlacement,
    overlap_fraction_smaller: f32,
    signed_clearance_m: f32,
    reasons: &[&'static str],
) -> String {
    format!(
        "{relationship} overlap with {} / {:?}: fraction={overlap_fraction_smaller:.3}, clearance_m={signed_clearance_m:.3}, reasons={}",
        other.object_id,
        other.instance_id,
        reasons.join("|")
    )
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct FeedbackCamera {
    translation: [f32; 3],
    rotation: [f32; 4],
    vertical_fov_degrees: f32,
    aspect: f32,
}

impl FeedbackCamera {
    fn from_status(status: &Value, screenshot_path: &Path) -> Option<Self> {
        let camera = status.get("camera")?;
        let translation = camera.get("translation").and_then(json_array3)?;
        let rotation = camera.get("rotation").and_then(json_array4)?;
        let vertical_fov_degrees = camera
            .get("vertical_fov_degrees")
            .and_then(Value::as_f64)
            .map(|value| value as f32)
            .filter(|value| value.is_finite() && *value > 1.0)
            .unwrap_or(70.0);
        let aspect = image::image_dimensions(screenshot_path)
            .ok()
            .map(|(width, height)| width.max(1) as f32 / height.max(1) as f32)
            .filter(|value| value.is_finite() && *value > 0.1)
            .unwrap_or(16.0 / 9.0);
        Some(Self {
            translation,
            rotation,
            vertical_fov_degrees,
            aspect,
        })
    }

    fn ground_point(&self, screen: [f32; 2], floor_y: f32) -> Option<[f32; 3]> {
        let tan_half = (self.vertical_fov_degrees.to_radians() * 0.5).tan();
        let local = normalize3([
            (2.0 * screen[0].clamp(0.0, 1.0) - 1.0) * self.aspect.max(0.1) * tan_half,
            (1.0 - 2.0 * screen[1].clamp(0.0, 1.0)) * tan_half,
            -1.0,
        ])?;
        let direction = quat_rotate_vec3(self.rotation, local);
        if !direction[1].is_finite() || direction[1].abs() <= 1.0e-5 {
            return None;
        }
        let t = (floor_y - self.translation[1]) / direction[1];
        if !t.is_finite() || t <= 0.0 {
            return None;
        }
        Some([
            self.translation[0] + direction[0] * t,
            floor_y,
            self.translation[2] + direction[2] * t,
        ])
    }
}

pub(crate) fn projected_item_ground_point(projected_item: &Value) -> Option<[f32; 3]> {
    let aabb = projected_item.get("world_aabb")?;
    let min = aabb.get("min").and_then(json_array3)?;
    let max = aabb.get("max").and_then(json_array3)?;
    if !min.iter().all(|value| value.is_finite()) || !max.iter().all(|value| value.is_finite()) {
        return None;
    }
    Some([
        (min[0] + max[0]) * 0.5,
        min[1].min(max[1]),
        (min[2] + max[2]) * 0.5,
    ])
}

pub(crate) fn normalize3(value: [f32; 3]) -> Option<[f32; 3]> {
    let len = (value[0] * value[0] + value[1] * value[1] + value[2] * value[2]).sqrt();
    if !len.is_finite() || len <= 1.0e-8 {
        return None;
    }
    Some([value[0] / len, value[1] / len, value[2] / len])
}

pub(crate) fn quat_rotate_vec3(quat: [f32; 4], vector: [f32; 3]) -> [f32; 3] {
    let q = [quat[0], quat[1], quat[2]];
    let t = scale3(cross3(q, vector), 2.0);
    add3(add3(vector, scale3(t, quat[3])), cross3(q, t))
}

pub(crate) fn cross3(lhs: [f32; 3], rhs: [f32; 3]) -> [f32; 3] {
    [
        lhs[1] * rhs[2] - lhs[2] * rhs[1],
        lhs[2] * rhs[0] - lhs[0] * rhs[2],
        lhs[0] * rhs[1] - lhs[1] * rhs[0],
    ]
}

pub(crate) fn add3(lhs: [f32; 3], rhs: [f32; 3]) -> [f32; 3] {
    [lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]]
}

pub(crate) fn sub3(lhs: [f32; 3], rhs: [f32; 3]) -> [f32; 3] {
    [lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]]
}

pub(crate) fn scale3(value: [f32; 3], scalar: f32) -> [f32; 3] {
    [value[0] * scalar, value[1] * scalar, value[2] * scalar]
}

pub(crate) fn clamp_xz_delta(delta: [f32; 3], max_len: f32) -> [f32; 3] {
    let len = (delta[0] * delta[0] + delta[2] * delta[2]).sqrt();
    if !len.is_finite() || len <= max_len.max(1.0e-5) {
        return delta;
    }
    let scale = max_len.max(1.0e-5) / len;
    [delta[0] * scale, delta[1], delta[2] * scale]
}

pub(crate) fn json_array2(value: &Value) -> Option<[f32; 2]> {
    let values = value.as_array()?;
    Some([
        values.first()?.as_f64()? as f32,
        values.get(1)?.as_f64()? as f32,
    ])
}

pub(crate) fn json_array3(value: &Value) -> Option<[f32; 3]> {
    let values = value.as_array()?;
    Some([
        values.first()?.as_f64()? as f32,
        values.get(1)?.as_f64()? as f32,
        values.get(2)?.as_f64()? as f32,
    ])
}

pub(crate) fn json_array4(value: &Value) -> Option<[f32; 4]> {
    let values = value.as_array()?;
    Some([
        values.first()?.as_f64()? as f32,
        values.get(1)?.as_f64()? as f32,
        values.get(2)?.as_f64()? as f32,
        values.get(3)?.as_f64()? as f32,
    ])
}

pub(crate) fn bbox_center(bbox: [f32; 4]) -> [f32; 2] {
    [(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5]
}

pub(crate) fn bbox_area(bbox: [f32; 4]) -> f32 {
    ((bbox[2] - bbox[0]).abs() * (bbox[3] - bbox[1]).abs()).max(1.0e-6)
}

pub(crate) fn bbox_aspect(bbox: [f32; 4]) -> f32 {
    ((bbox[2] - bbox[0]).abs() / (bbox[3] - bbox[1]).abs().max(1.0e-6)).max(1.0e-6)
}

pub(crate) fn distance2(lhs: [f32; 2], rhs: [f32; 2]) -> f32 {
    let dx = lhs[0] - rhs[0];
    let dy = lhs[1] - rhs[1];
    (dx * dx + dy * dy).sqrt()
}

pub(crate) fn vec2_sub(lhs: [f32; 2], rhs: [f32; 2]) -> [f32; 2] {
    [lhs[0] - rhs[0], lhs[1] - rhs[1]]
}

pub(crate) fn safe_log2_ratio(lhs: f32, rhs: f32) -> f32 {
    (lhs.max(1.0e-6) / rhs.max(1.0e-6)).log2()
}

pub(crate) fn expand_bbox_envelope(min: &mut [f32; 2], max: &mut [f32; 2], bbox: [f32; 4]) {
    min[0] = min[0].min(bbox[0]);
    min[1] = min[1].min(bbox[1]);
    max[0] = max[0].max(bbox[2]);
    max[1] = max[1].max(bbox[3]);
}

pub(crate) fn envelope_area(min: [f32; 2], max: [f32; 2]) -> f32 {
    if !min[0].is_finite() || !max[0].is_finite() {
        return 0.0;
    }
    ((max[0] - min[0]).abs() * (max[1] - min[1]).abs()).max(1.0e-6)
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct FeedbackYawCorrection {
    pub(crate) delta_degrees: f32,
    pub(crate) basis: &'static str,
}

pub(crate) fn status_camera_yaw_degrees(status: &Value, fallback_degrees: Option<f32>) -> f32 {
    let raw_yaw = status
        .get("camera")
        .and_then(|camera| camera.get("yaw"))
        .and_then(Value::as_f64)
        .map(|value| value as f32)
        .or(fallback_degrees)
        .unwrap_or(180.0);
    let degrees = if raw_yaw.abs() <= std::f32::consts::TAU + 1.0e-3 {
        raw_yaw.to_degrees()
    } else {
        raw_yaw
    };
    normalize_degrees(degrees)
}

pub(crate) fn status_world_item_yaw_degrees(
    status: &Value,
    index: usize,
    cache_key: Option<&str>,
) -> Option<f32> {
    let world_items = status.get("world_items").and_then(Value::as_array)?;
    if let Some(item) = world_items.get(index)
        && cache_key_matches(item, cache_key)
        && let Some(yaw) = world_item_yaw_degrees(item)
    {
        return Some(yaw);
    }
    let cache_key = cache_key?;
    world_items
        .iter()
        .find(|item| cache_key_matches(item, Some(cache_key)))
        .and_then(world_item_yaw_degrees)
}

pub(crate) fn cache_key_matches(item: &Value, cache_key: Option<&str>) -> bool {
    let Some(cache_key) = cache_key else {
        return true;
    };
    item.get("cache_key")
        .and_then(Value::as_str)
        .is_some_and(|value| value == cache_key)
}

pub(crate) fn world_item_yaw_degrees(item: &Value) -> Option<f32> {
    item.get("rotation")
        .and_then(json_array4)
        .map(quat_y_degrees)
}

pub(crate) fn feedback_yaw_correction(
    _placement_index: usize,
    placement: &GroundedScenePlacement,
    current_yaw_degrees: f32,
    _physical: &FeedbackPhysicalLayout,
) -> FeedbackYawCorrection {
    let canonical_error = normalize_degrees(placement.rotation_y_degrees - current_yaw_degrees);
    if canonical_error.abs() > 2.0 {
        let step_degrees = (canonical_error.abs() * 0.70).clamp(3.0, 24.0);
        return FeedbackYawCorrection {
            delta_degrees: canonical_error.clamp(-step_degrees, step_degrees),
            basis: "canonical-bsn-yaw",
        };
    }
    FeedbackYawCorrection {
        delta_degrees: 0.0,
        basis: "canonical-bsn-yaw-within-threshold",
    }
}

pub(crate) fn quat_y_degrees(quat: [f32; 4]) -> f32 {
    normalize_degrees((2.0 * quat[1].atan2(quat[3])).to_degrees())
}

pub(crate) fn quat_from_y_degrees(degrees: f32) -> [f32; 4] {
    let half = normalize_degrees(degrees).to_radians() * 0.5;
    [0.0, half.sin(), 0.0, half.cos()]
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

pub(crate) fn fmt_feedback_vec3(value: [f32; 3]) -> String {
    format!(
        "{},{},{}",
        fmt_feedback_num(value[0]),
        fmt_feedback_num(value[1]),
        fmt_feedback_num(value[2])
    )
}

pub(crate) fn fmt_feedback_num(value: f32) -> String {
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

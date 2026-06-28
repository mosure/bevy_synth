use crate::prelude::*;

#[derive(Clone, Debug)]
pub(crate) struct CanonicalPoseCalibrationRun {
    pub(crate) asset_bindings: Vec<SceneAssetBinding>,
    pub(crate) reports: Vec<CanonicalPoseCalibrationReport>,
    pub(crate) selection_task: Value,
    pub(crate) image_paths: Vec<PathBuf>,
    pub(crate) selection_report: Value,
}

struct CanonicalPoseReportContext<'a> {
    mode: SceneCanonicalPoseMode,
    max_candidates: usize,
    manifest: &'a SceneObjectManifest,
    selected_candidates: &'a [Value],
    object_image_requests: &'a [ObjectImageRequest],
    evidence: &'a SceneGroundingEvidence,
}

pub(crate) fn build_canonical_pose_calibration(
    mode: SceneCanonicalPoseMode,
    max_candidates: usize,
    manifest: &SceneObjectManifest,
    asset_bindings: &[SceneAssetBinding],
    selected_candidates: &[Value],
    object_image_requests: &[ObjectImageRequest],
    evidence: &SceneGroundingEvidence,
) -> CanonicalPoseCalibrationRun {
    let context = CanonicalPoseReportContext {
        mode,
        max_candidates,
        manifest,
        selected_candidates,
        object_image_requests,
        evidence,
    };
    let mut reports = asset_bindings
        .iter()
        .enumerate()
        .map(|(index, asset)| canonical_pose_report_for_asset(index, asset, &context))
        .collect::<Vec<_>>();
    let asset_bindings = asset_bindings_with_calibrated_frames(asset_bindings, &reports);
    let selection_task = canonical_pose_selection_task(&reports);
    let image_paths = canonical_pose_selection_image_paths(&selection_task);
    for report in &mut reports {
        if mode == SceneCanonicalPoseMode::Off {
            report.fallback_used = false;
        }
    }
    CanonicalPoseCalibrationRun {
        asset_bindings,
        reports,
        selection_task,
        image_paths,
        selection_report: Value::Null,
    }
}

pub(crate) fn apply_canonical_pose_openai_selection(
    run: &mut CanonicalPoseCalibrationRun,
    response: &SceneRotationSelectionResponse,
) -> Value {
    let mut applied = Vec::new();
    let mut ignored = Vec::new();
    for selection in &response.objects {
        let Some(report) = run.reports.get_mut(selection.index) else {
            ignored.push(json!({
                "index": selection.index,
                "candidate_index": selection.candidate_index,
                "reason": "unknown asset index",
            }));
            continue;
        };
        let Some(candidate) = report
            .candidates
            .iter()
            .find(|candidate| candidate.candidate_index == selection.candidate_index)
            .cloned()
        else {
            ignored.push(json!({
                "index": selection.index,
                "asset_id": report.asset_id,
                "candidate_index": selection.candidate_index,
                "reason": "candidate index not available for asset",
            }));
            continue;
        };
        if selection.confidence < 0.55 {
            ignored.push(json!({
                "index": selection.index,
                "asset_id": report.asset_id,
                "candidate_index": selection.candidate_index,
                "confidence": selection.confidence,
                "reason": "confidence below 0.55 threshold",
            }));
            continue;
        }
        report.selected = CanonicalPoseSelection {
            candidate_index: candidate.candidate_index,
            yaw_offset_degrees: candidate.yaw_offset_degrees,
            confidence: selection.confidence.clamp(0.0, 1.0),
            source: SceneAssetFrameSource::GptVisualSelection,
            rationale: selection.rationale.clone(),
        };
        report.fallback_used = false;
        applied.push(json!({
            "index": selection.index,
            "asset_id": report.asset_id,
            "candidate_index": selection.candidate_index,
            "yaw_offset_degrees": candidate.yaw_offset_degrees,
            "confidence": selection.confidence,
        }));
    }
    run.asset_bindings = asset_bindings_with_calibrated_frames(&run.asset_bindings, &run.reports);
    run.selection_report = json!({
        "selector": "openai",
        "applied_count": applied.len(),
        "ignored_count": ignored.len(),
        "applied": applied,
        "ignored": ignored,
    });
    run.selection_report.clone()
}

pub(crate) fn apply_canonical_pose_rendered_selection(
    run: &mut CanonicalPoseCalibrationRun,
) -> Value {
    let mut applied = Vec::new();
    let mut skipped = Vec::new();
    for (report_index, report) in run.reports.iter_mut().enumerate() {
        let prior_selection = report.selected.clone();
        let source_metrics = report
            .source_crop_path
            .as_deref()
            .and_then(|path| reference_pose_image_metrics(path).ok().flatten());
        let generated_metrics = report
            .generated_image_path
            .as_deref()
            .and_then(|path| reference_pose_image_metrics(path).ok().flatten());
        if source_metrics.is_none() && generated_metrics.is_none() {
            skipped.push(json!({
                "index": report_index,
                "asset_id": report.asset_id,
                "reason": "missing source crop and generated object image metrics",
            }));
            continue;
        }

        let mut best = None::<(usize, f32, f32, f32, RenderedPoseImageMetrics)>;
        for (candidate_index, candidate) in report.candidates.iter_mut().enumerate() {
            let Some(render_path) = candidate.rendered_image_path.as_deref() else {
                continue;
            };
            let Ok(pixel_metrics) = rendered_pose_image_metrics(render_path) else {
                continue;
            };
            if let Err(err) = validate_canonical_pose_thumbnail_metrics(&pixel_metrics) {
                candidate.metrics["render_similarity_error"] = json!(err);
                continue;
            }
            let projected_bbox = projected_bbox_for_pose_candidate(candidate);
            let rendered_metrics = projected_bbox
                .map(|bbox| pixel_metrics.clone().with_foreground_bbox(bbox))
                .unwrap_or_else(|| pixel_metrics.clone());
            let generated_score = generated_metrics
                .as_ref()
                .map(|metrics| rendered_pose_similarity(&rendered_metrics, metrics))
                .unwrap_or(0.0);
            let source_score = source_metrics
                .as_ref()
                .map(|metrics| rendered_pose_similarity(&rendered_metrics, metrics))
                .unwrap_or(0.0);
            let evidence_weight = if generated_metrics.is_some() && source_metrics.is_some() {
                0.82
            } else {
                0.70
            };
            let evidence_score = if generated_metrics.is_some() && source_metrics.is_some() {
                generated_score * 0.64 + source_score * 0.36
            } else if generated_metrics.is_some() {
                generated_score
            } else {
                source_score
            };
            let measured_score = (evidence_score * evidence_weight
                + candidate.score * (1.0 - evidence_weight))
                .clamp(0.0, 1.0);
            candidate.score = measured_score;
            candidate.metrics["render_similarity"] = json!({
                "measured_score": measured_score,
                "generated_score": generated_score,
                "source_score": source_score,
                "rendered": rendered_metrics,
                "pixel_rendered": pixel_metrics,
                "projected_bbox_used": projected_bbox,
                "generated_available": generated_metrics.is_some(),
                "source_available": source_metrics.is_some(),
            });
            if best
                .as_ref()
                .map(|(_, score, _, _, _)| measured_score > *score)
                .unwrap_or(true)
            {
                best = Some((
                    candidate_index,
                    measured_score,
                    generated_score,
                    source_score,
                    rendered_metrics,
                ));
            }
        }

        let Some((candidate_vec_index, measured_score, generated_score, source_score, metrics)) =
            best
        else {
            skipped.push(json!({
                "index": report_index,
                "asset_id": report.asset_id,
                "reason": "no rendered candidate thumbnails were available",
            }));
            continue;
        };
        let candidate = report.candidates[candidate_vec_index].clone();
        let fallback_used = measured_score < 0.38;
        if fallback_used {
            report
                .warnings
                .push("rendered thumbnail similarity below confidence threshold".to_string());
            report.selected = CanonicalPoseSelection {
                candidate_index: prior_selection.candidate_index,
                yaw_offset_degrees: prior_selection.yaw_offset_degrees,
                confidence: measured_score.clamp(0.0, 1.0),
                source: SceneAssetFrameSource::AmbiguousFallback,
                rationale: "rendered thumbnail evidence was below confidence threshold; retained the prior deterministic canonical frame".to_string(),
            };
        } else {
            report.selected = CanonicalPoseSelection {
                candidate_index: candidate.candidate_index,
                yaw_offset_degrees: candidate.yaw_offset_degrees,
                confidence: measured_score.clamp(0.0, 1.0),
                source: SceneAssetFrameSource::VisualRenderSweep,
                rationale: "selected from rendered asset thumbnail similarity against source/generated object evidence".to_string(),
            };
        }
        report.fallback_used = fallback_used;
        applied.push(json!({
            "index": report_index,
            "asset_id": report.asset_id,
            "candidate_index": report.selected.candidate_index,
            "yaw_offset_degrees": report.selected.yaw_offset_degrees,
            "confidence": report.selected.confidence,
            "best_measured_candidate_index": candidate.candidate_index,
            "best_measured_yaw_offset_degrees": candidate.yaw_offset_degrees,
            "generated_score": generated_score,
            "source_score": source_score,
            "rendered_metrics": metrics,
            "fallback": report.fallback_used,
        }));
    }
    run.asset_bindings = asset_bindings_with_calibrated_frames(&run.asset_bindings, &run.reports);
    refresh_canonical_pose_selection_inputs(run);
    run.selection_report = json!({
        "selector": "rendered-thumbnail-sweep",
        "applied_count": applied.len(),
        "skipped_count": skipped.len(),
        "applied": applied,
        "skipped": skipped,
    });
    run.selection_report.clone()
}

pub(crate) fn canonical_pose_verification_report(
    mode: SceneCanonicalPoseMode,
    run: &CanonicalPoseCalibrationRun,
) -> Value {
    let visual_evidence_required = matches!(
        mode,
        SceneCanonicalPoseMode::Auto
            | SceneCanonicalPoseMode::RenderSweep
            | SceneCanonicalPoseMode::Openai
    );
    let mut candidate_count = 0usize;
    let mut rendered_candidate_count = 0usize;
    let mut visual_selected_count = 0usize;
    let mut fallback_assets = Vec::new();
    let mut low_confidence_assets = Vec::new();
    let mut missing_rendered_assets = Vec::new();
    let mut source_evidence_missing_assets = Vec::new();
    let mut generated_evidence_missing_assets = Vec::new();

    for report in &run.reports {
        candidate_count += report.candidates.len();
        let rendered_count = report
            .candidates
            .iter()
            .filter(|candidate| {
                candidate
                    .rendered_image_path
                    .as_deref()
                    .is_some_and(|path| !path.is_empty())
            })
            .count();
        rendered_candidate_count += rendered_count;
        if visual_evidence_required && rendered_count == 0 {
            missing_rendered_assets.push(json!({
                "asset_id": report.asset_id,
                "object_id": report.object_id,
                "label": report.label,
                "reason": "no rendered canonical pose candidate thumbnails were available",
            }));
        }
        if report.source_crop_path.is_none() {
            source_evidence_missing_assets.push(json!({
                "asset_id": report.asset_id,
                "object_id": report.object_id,
                "label": report.label,
                "reason": "missing source crop evidence",
            }));
        }
        if report.generated_image_path.is_none() {
            generated_evidence_missing_assets.push(json!({
                "asset_id": report.asset_id,
                "object_id": report.object_id,
                "label": report.label,
                "reason": "missing generated object image evidence",
            }));
        }
        if matches!(
            report.selected.source,
            SceneAssetFrameSource::VisualRenderSweep | SceneAssetFrameSource::GptVisualSelection
        ) && !report.fallback_used
        {
            visual_selected_count += 1;
        }
        if report.fallback_used
            || matches!(
                report.selected.source,
                SceneAssetFrameSource::AmbiguousFallback
            )
        {
            fallback_assets.push(json!({
                "asset_id": report.asset_id,
                "object_id": report.object_id,
                "label": report.label,
                "selected_source": report.selected.source,
                "confidence": report.selected.confidence,
                "warnings": report.warnings,
            }));
        } else if report.selected.confidence < 0.55 {
            low_confidence_assets.push(json!({
                "asset_id": report.asset_id,
                "object_id": report.object_id,
                "label": report.label,
                "selected_source": report.selected.source,
                "confidence": report.selected.confidence,
            }));
        }
    }

    let status = if mode == SceneCanonicalPoseMode::Off {
        "disabled"
    } else if mode == SceneCanonicalPoseMode::Heuristic {
        "heuristic"
    } else if run.reports.is_empty() {
        "no_assets"
    } else if !missing_rendered_assets.is_empty() {
        "invalid"
    } else if !fallback_assets.is_empty() {
        "fallback"
    } else if !low_confidence_assets.is_empty() {
        "low_confidence"
    } else if visual_selected_count == run.reports.len() {
        "verified"
    } else {
        "partial"
    };
    let requires_attention = visual_evidence_required
        && matches!(
            status,
            "invalid" | "fallback" | "low_confidence" | "partial"
        )
        && !run.reports.is_empty();

    json!({
        "status": status,
        "visual_verified": status == "verified",
        "requires_attention": requires_attention,
        "mode": canonical_pose_mode_label(mode),
        "visual_evidence_required": visual_evidence_required,
        "asset_count": run.reports.len(),
        "candidate_count": candidate_count,
        "rendered_candidate_count": rendered_candidate_count,
        "visual_selected_count": visual_selected_count,
        "fallback_count": fallback_assets.len(),
        "missing_rendered_asset_count": missing_rendered_assets.len(),
        "source_evidence_missing_count": source_evidence_missing_assets.len(),
        "generated_evidence_missing_count": generated_evidence_missing_assets.len(),
        "low_confidence_count": low_confidence_assets.len(),
        "selector": run
            .selection_report
            .get("selector")
            .cloned()
            .unwrap_or(Value::Null),
        "render_report": run
            .selection_report
            .get("render_report")
            .cloned()
            .unwrap_or(Value::Null),
        "missing_rendered_assets": missing_rendered_assets,
        "fallback_assets": fallback_assets,
        "low_confidence_assets": low_confidence_assets,
        "source_evidence_missing_assets": source_evidence_missing_assets,
        "generated_evidence_missing_assets": generated_evidence_missing_assets,
    })
}

pub(crate) fn canonical_pose_selection_prompt(task: &Value) -> String {
    let task_json = serde_json::to_string_pretty(task)
        .unwrap_or_else(|_| serde_json::to_string(task).unwrap_or_default());
    format!(
        "You are selecting bounded canonical yaw corrections for generated 3D assets before scene placement.\n\
         Compare each source crop and generated object image with the candidate yaw descriptions. \
         Pick exactly one candidate_index per asset. Do not invent absolute yaw, positions, scales, \
         transforms, or new candidate values. If evidence is ambiguous, choose the closest deterministic \
         candidate and lower confidence.\n\nJSON task:\n{task_json}"
    )
}

#[derive(Clone, Debug, serde::Serialize)]
struct RenderedPoseImageMetrics {
    width: u32,
    height: u32,
    foreground_bbox: [f32; 4],
    foreground_area_ratio: f32,
    foreground_center: [f32; 2],
    foreground_aspect: f32,
    mean_luma: f32,
}

impl RenderedPoseImageMetrics {
    fn with_foreground_bbox(mut self, bbox: [f32; 4]) -> Self {
        let bbox = [
            bbox[0].clamp(0.0, 1.0),
            bbox[1].clamp(0.0, 1.0),
            bbox[2].clamp(0.0, 1.0),
            bbox[3].clamp(0.0, 1.0),
        ];
        let width = (bbox[2] - bbox[0]).max(1.0 / self.width.max(1) as f32);
        let height = (bbox[3] - bbox[1]).max(1.0 / self.height.max(1) as f32);
        self.foreground_bbox = bbox;
        self.foreground_center = [(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5];
        self.foreground_aspect = width / height;
        self.foreground_area_ratio = width * height;
        self
    }

    fn foreground_bbox_area_ratio(&self) -> f32 {
        (self.foreground_bbox[2] - self.foreground_bbox[0]).max(0.0)
            * (self.foreground_bbox[3] - self.foreground_bbox[1]).max(0.0)
    }

    fn is_informative_reference(&self) -> bool {
        if self.foreground_area_ratio <= 0.002 {
            return false;
        }
        let bbox_area = self.foreground_bbox_area_ratio();
        if self.foreground_area_ratio >= 0.92 && bbox_area >= 0.92 {
            return false;
        }
        true
    }
}

fn reference_pose_image_metrics(path: &str) -> Result<Option<RenderedPoseImageMetrics>, String> {
    if is_mask_artifact_path(path) {
        return Ok(None);
    }
    let metrics = rendered_pose_image_metrics(path)?;
    Ok(metrics.is_informative_reference().then_some(metrics))
}

fn is_mask_artifact_path(path: &str) -> bool {
    let normalized = path.replace('\\', "/").to_ascii_lowercase();
    normalized.contains("/masks/") || normalized.ends_with("_mask.png")
}

fn validate_canonical_pose_thumbnail_metrics(
    metrics: &RenderedPoseImageMetrics,
) -> Result<(), String> {
    const MIN_FOREGROUND_AREA_RATIO: f32 = 0.008;
    let bbox_area = metrics.foreground_bbox_area_ratio();
    if metrics.foreground_area_ratio < MIN_FOREGROUND_AREA_RATIO {
        return Err(format!(
            "canonical pose thumbnail foreground area {:.5} below {:.5}; capture is likely blank or background-only",
            metrics.foreground_area_ratio, MIN_FOREGROUND_AREA_RATIO
        ));
    }
    if metrics.foreground_area_ratio >= 0.92 && bbox_area >= 0.92 {
        return Err(
            "canonical pose thumbnail foreground covers the full frame; capture is likely blank or a non-isolated mask"
                .to_string(),
        );
    }
    Ok(())
}

pub(crate) fn canonical_pose_thumbnail_pixel_metrics(path: &Path) -> Result<Value, String> {
    let path = path.display().to_string();
    let metrics = rendered_pose_image_metrics(&path)?;
    validate_canonical_pose_thumbnail_metrics(&metrics)?;
    serde_json::to_value(metrics).map_err(|err| err.to_string())
}

fn rendered_pose_image_metrics(path: &str) -> Result<RenderedPoseImageMetrics, String> {
    let image = image::open(path)
        .map_err(|err| format!("open canonical pose image {path}: {err}"))?
        .to_rgba8();
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return Err(format!("canonical pose image {path} is empty"));
    }

    let mut border = [0.0f32; 4];
    let mut border_count = 0.0f32;
    for y in 0..height {
        for x in 0..width {
            if x != 0 && y != 0 && x + 1 != width && y + 1 != height {
                continue;
            }
            let pixel = image.get_pixel(x, y).0;
            border[0] += pixel[0] as f32;
            border[1] += pixel[1] as f32;
            border[2] += pixel[2] as f32;
            border[3] += pixel[3] as f32;
            border_count += 1.0;
        }
    }
    if border_count > 0.0 {
        border[0] /= border_count;
        border[1] /= border_count;
        border[2] /= border_count;
        border[3] /= border_count;
    }

    let mut min_x = width;
    let mut min_y = height;
    let mut max_x = 0u32;
    let mut max_y = 0u32;
    let mut foreground = 0u32;
    let mut luma_sum = 0.0f32;
    for y in 0..height {
        for x in 0..width {
            let pixel = image.get_pixel(x, y).0;
            let alpha_foreground = (pixel[3] as f32 - border[3]).abs() > 32.0;
            let diff = ((pixel[0] as f32 - border[0]).abs()
                + (pixel[1] as f32 - border[1]).abs()
                + (pixel[2] as f32 - border[2]).abs())
                / 3.0;
            if alpha_foreground || diff > 22.0 {
                foreground += 1;
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
                luma_sum +=
                    pixel[0] as f32 * 0.2126 + pixel[1] as f32 * 0.7152 + pixel[2] as f32 * 0.0722;
            }
        }
    }

    if foreground < (width * height / 250).max(8) {
        min_x = 0;
        min_y = 0;
        max_x = width.saturating_sub(1);
        max_y = height.saturating_sub(1);
        foreground = width * height;
        luma_sum = image
            .pixels()
            .map(|pixel| {
                let pixel = pixel.0;
                pixel[0] as f32 * 0.2126 + pixel[1] as f32 * 0.7152 + pixel[2] as f32 * 0.0722
            })
            .sum();
    }

    let bbox_width = (max_x + 1).saturating_sub(min_x).max(1);
    let bbox_height = (max_y + 1).saturating_sub(min_y).max(1);
    let bbox = [
        min_x as f32 / width as f32,
        min_y as f32 / height as f32,
        (max_x + 1) as f32 / width as f32,
        (max_y + 1) as f32 / height as f32,
    ];
    Ok(RenderedPoseImageMetrics {
        width,
        height,
        foreground_bbox: bbox,
        foreground_area_ratio: foreground as f32 / (width * height) as f32,
        foreground_center: [(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5],
        foreground_aspect: bbox_width as f32 / bbox_height as f32,
        mean_luma: (luma_sum / foreground.max(1) as f32) / 255.0,
    })
}

fn projected_bbox_for_pose_candidate(candidate: &CanonicalPoseCandidate) -> Option<[f32; 4]> {
    let bbox = candidate
        .metrics
        .pointer("/render/status/projected_items/0/screen_bbox")?
        .as_array()?;
    if bbox.len() != 4 {
        return None;
    }
    let mut output = [0.0f32; 4];
    for (index, value) in bbox.iter().enumerate() {
        output[index] = value.as_f64()? as f32;
    }
    if output[2] <= output[0] || output[3] <= output[1] {
        return None;
    }
    Some(output)
}

fn rendered_pose_similarity(
    rendered: &RenderedPoseImageMetrics,
    reference: &RenderedPoseImageMetrics,
) -> f32 {
    let aspect_error =
        safe_log_ratio(rendered.foreground_aspect, reference.foreground_aspect).abs();
    let area_error = safe_log_ratio(
        rendered.foreground_area_ratio,
        reference.foreground_area_ratio,
    )
    .abs();
    let center_error = ((rendered.foreground_center[0] - reference.foreground_center[0]).powi(2)
        + (rendered.foreground_center[1] - reference.foreground_center[1]).powi(2))
    .sqrt();
    let bbox_iou = bbox_iou_2d(rendered.foreground_bbox, reference.foreground_bbox);
    let luma_error = (rendered.mean_luma - reference.mean_luma).abs();
    (1.0 - (aspect_error * 0.22
        + area_error * 0.18
        + center_error * 0.28
        + (1.0 - bbox_iou) * 0.24
        + luma_error * 0.08))
        .clamp(0.0, 1.0)
}

fn safe_log_ratio(left: f32, right: f32) -> f32 {
    (left.max(1.0e-4) / right.max(1.0e-4)).log2()
}

fn bbox_iou_2d(left: [f32; 4], right: [f32; 4]) -> f32 {
    let ix0 = left[0].max(right[0]);
    let iy0 = left[1].max(right[1]);
    let ix1 = left[2].min(right[2]);
    let iy1 = left[3].min(right[3]);
    let iw = (ix1 - ix0).max(0.0);
    let ih = (iy1 - iy0).max(0.0);
    let intersection = iw * ih;
    let left_area = (left[2] - left[0]).max(0.0) * (left[3] - left[1]).max(0.0);
    let right_area = (right[2] - right[0]).max(0.0) * (right[3] - right[1]).max(0.0);
    let union = left_area + right_area - intersection;
    if union <= f32::EPSILON {
        0.0
    } else {
        intersection / union
    }
}

pub(crate) fn refresh_canonical_pose_selection_inputs(run: &mut CanonicalPoseCalibrationRun) {
    run.selection_task = canonical_pose_selection_task(&run.reports);
    run.image_paths = canonical_pose_selection_image_paths(&run.selection_task);
}

fn canonical_pose_report_for_asset(
    index: usize,
    asset: &SceneAssetBinding,
    context: &CanonicalPoseReportContext<'_>,
) -> CanonicalPoseCalibrationReport {
    let object = context
        .manifest
        .objects
        .iter()
        .find(|object| object.id == asset.object_id);
    let descriptor = format!("{} {}", asset.label, asset.aliases.join(" ")).to_ascii_lowercase();
    let heuristic_frame = asset.canonical_frame.unwrap_or_else(|| {
        inferred_scene_asset_frame(
            &asset.label,
            &asset.aliases,
            asset.local_aabb,
            object.and_then(|object| object.target_footprint_m),
        )
    });
    let source_crop_path =
        source_crop_for_asset(asset, context.object_image_requests, context.evidence);
    let generated_image_path = generated_image_for_asset(asset, context.selected_candidates);
    let mut warnings = Vec::new();
    if source_crop_path.is_none() {
        warnings.push("missing source crop; calibration is not visually grounded".to_string());
    }
    if generated_image_path.is_none() {
        warnings.push(
            "missing generated object image; calibration cannot compare reconstruction input"
                .to_string(),
        );
    }
    let candidates = canonical_pose_candidates_for_asset(
        &descriptor,
        heuristic_frame,
        source_crop_path.as_deref(),
        context.max_candidates,
        context.mode,
    );
    let selected = match context.mode {
        SceneCanonicalPoseMode::Off => CanonicalPoseSelection {
            candidate_index: 0,
            yaw_offset_degrees: 0.0,
            confidence: 1.0,
            source: SceneAssetFrameSource::Explicit,
            rationale: "canonical pose correction disabled".to_string(),
        },
        SceneCanonicalPoseMode::Heuristic => CanonicalPoseSelection {
            candidate_index: 0,
            yaw_offset_degrees: heuristic_frame.yaw_offset_degrees,
            confidence: heuristic_frame.confidence.unwrap_or(0.50),
            source: heuristic_frame
                .source
                .unwrap_or(SceneAssetFrameSource::DescriptorHeuristic),
            rationale: "using descriptor/AABB heuristic canonical frame".to_string(),
        },
        SceneCanonicalPoseMode::RenderSweep
        | SceneCanonicalPoseMode::Auto
        | SceneCanonicalPoseMode::Openai => candidates
            .iter()
            .max_by(|left, right| {
                left.score
                    .partial_cmp(&right.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| right.candidate_index.cmp(&left.candidate_index))
            })
            .map(|candidate| CanonicalPoseSelection {
                candidate_index: candidate.candidate_index,
                yaw_offset_degrees: candidate.yaw_offset_degrees,
                confidence: candidate.score.clamp(0.0, 1.0),
                source: SceneAssetFrameSource::VisualRenderSweep,
                rationale: "deterministic candidate selected from canonical yaw sweep".to_string(),
            })
            .unwrap_or_else(|| CanonicalPoseSelection {
                candidate_index: 0,
                yaw_offset_degrees: heuristic_frame.yaw_offset_degrees,
                confidence: heuristic_frame.confidence.unwrap_or(0.35),
                source: SceneAssetFrameSource::AmbiguousFallback,
                rationale: "no canonical pose candidates were available".to_string(),
            }),
    };
    let fallback_used = matches!(selected.source, SceneAssetFrameSource::AmbiguousFallback)
        || (matches!(
            context.mode,
            SceneCanonicalPoseMode::Auto | SceneCanonicalPoseMode::RenderSweep
        ) && selected.confidence < 0.55);
    let mut report = CanonicalPoseCalibrationReport {
        schema_version: 1,
        mode: canonical_pose_mode_label(context.mode).to_string(),
        asset_id: asset.asset_id.clone(),
        object_id: asset.object_id.clone(),
        label: asset.label.clone(),
        source_crop_path,
        generated_image_path,
        selected,
        candidates,
        fallback_used,
        warnings,
    };
    if report.fallback_used {
        report.selected.source = SceneAssetFrameSource::AmbiguousFallback;
        report
            .warnings
            .push("selected canonical pose confidence below visual threshold".to_string());
    }
    report
        .candidates
        .iter_mut()
        .for_each(|candidate| candidate.metrics["asset_index"] = json!(index));
    report
}

fn canonical_pose_candidates_for_asset(
    descriptor: &str,
    heuristic_frame: SceneAssetFrame,
    source_crop_path: Option<&str>,
    max_candidates: usize,
    mode: SceneCanonicalPoseMode,
) -> Vec<CanonicalPoseCandidate> {
    if mode == SceneCanonicalPoseMode::Off {
        return vec![CanonicalPoseCandidate {
            candidate_index: 0,
            yaw_offset_degrees: 0.0,
            candidate_yaw_degrees: 0.0,
            score: 1.0,
            source_crop_path: source_crop_path.map(str::to_string),
            rendered_image_path: None,
            metrics: json!({ "basis": "disabled" }),
        }];
    }
    let mut offsets = Vec::new();
    push_unique_yaw(&mut offsets, heuristic_frame.yaw_offset_degrees);
    let symmetry = heuristic_frame
        .symmetry
        .unwrap_or_else(|| symmetry_for_descriptor(descriptor));
    match symmetry {
        SceneAssetSymmetry::Radial => {}
        SceneAssetSymmetry::Axis90 => {
            for yaw in [0.0, 90.0, 180.0, -90.0] {
                push_unique_yaw(&mut offsets, yaw);
            }
        }
        SceneAssetSymmetry::Axis180 | SceneAssetSymmetry::Bilateral => {
            push_unique_yaw(&mut offsets, heuristic_frame.yaw_offset_degrees + 180.0);
            push_unique_yaw(&mut offsets, heuristic_frame.yaw_offset_degrees + 90.0);
            push_unique_yaw(&mut offsets, heuristic_frame.yaw_offset_degrees - 90.0);
        }
        SceneAssetSymmetry::Asymmetric | SceneAssetSymmetry::Unknown => {
            for yaw in [0.0, 90.0, 180.0, -90.0] {
                push_unique_yaw(&mut offsets, yaw);
            }
        }
    }
    let max_candidates = max_candidates.max(1);
    offsets
        .into_iter()
        .take(max_candidates)
        .enumerate()
        .map(|(candidate_index, yaw)| {
            let heuristic_delta = angle_distance_degrees(yaw, heuristic_frame.yaw_offset_degrees);
            let evidence_bonus = if source_crop_path.is_some() {
                0.04
            } else {
                0.0
            };
            let score = (heuristic_frame.confidence.unwrap_or(0.45) + evidence_bonus
                - heuristic_delta / 360.0)
                .clamp(0.0, 1.0);
            CanonicalPoseCandidate {
                candidate_index,
                yaw_offset_degrees: yaw,
                candidate_yaw_degrees: yaw,
                score,
                source_crop_path: source_crop_path.map(str::to_string),
                rendered_image_path: None,
                metrics: json!({
                    "basis": "deterministic_yaw_sweep",
                    "heuristic_yaw_offset_degrees": heuristic_frame.yaw_offset_degrees,
                    "heuristic_delta_degrees": heuristic_delta,
                    "symmetry": symmetry,
                    "rendered_asset_thumbnail": false,
                    "renderer": "pending",
                }),
            }
        })
        .collect()
}

fn asset_bindings_with_calibrated_frames(
    asset_bindings: &[SceneAssetBinding],
    reports: &[CanonicalPoseCalibrationReport],
) -> Vec<SceneAssetBinding> {
    let reports_by_asset = reports
        .iter()
        .map(|report| (report.asset_id.as_str(), report))
        .collect::<HashMap<_, _>>();
    asset_bindings
        .iter()
        .map(|asset| {
            let mut asset = asset.clone();
            if let Some(report) = reports_by_asset.get(asset.asset_id.as_str()) {
                let previous = asset.canonical_frame;
                asset.canonical_frame = Some(SceneAssetFrame {
                    yaw_offset_degrees: report.selected.yaw_offset_degrees,
                    footprint_m: previous.and_then(|frame| frame.footprint_m),
                    symmetry: previous.and_then(|frame| frame.symmetry).or_else(|| {
                        Some(symmetry_for_descriptor(&format!(
                            "{} {}",
                            asset.label,
                            asset.aliases.join(" ")
                        )))
                    }),
                    confidence: Some(report.selected.confidence),
                    source: Some(report.selected.source),
                });
            }
            asset
        })
        .collect()
}

fn canonical_pose_selection_task(reports: &[CanonicalPoseCalibrationReport]) -> Value {
    let objects = reports
        .iter()
        .enumerate()
        .map(|(index, report)| {
            json!({
                "index": index,
                "asset_id": report.asset_id,
                "object_id": report.object_id,
                "label": report.label,
                "source_crop_path": report.source_crop_path,
                "generated_image_path": report.generated_image_path,
                "selected_candidate_index": report.selected.candidate_index,
                "selected_yaw_offset_degrees": report.selected.yaw_offset_degrees,
                "candidates": report.candidates,
            })
        })
        .collect::<Vec<_>>();
    json!({
        "purpose": "bounded-canonical-asset-rotation-selection",
        "instruction": "Choose candidate_index values only. Candidate yaw offsets are local asset canonical corrections, not world transforms.",
        "objects": objects,
    })
}

fn canonical_pose_selection_image_paths(task: &Value) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    let mut seen = HashSet::new();
    for object in task
        .get("objects")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        for key in ["source_crop_path", "generated_image_path"] {
            let Some(path) = object.get(key).and_then(Value::as_str) else {
                continue;
            };
            if seen.insert(path.to_string()) {
                paths.push(PathBuf::from(path));
            }
        }
        for candidate in object
            .get("candidates")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let Some(path) = candidate
                .get("rendered_image_path")
                .and_then(Value::as_str)
                .filter(|path| !path.trim().is_empty())
            else {
                continue;
            };
            if seen.insert(path.to_string()) {
                paths.push(PathBuf::from(path));
            }
        }
    }
    paths
}

fn source_crop_for_asset(
    asset: &SceneAssetBinding,
    object_image_requests: &[ObjectImageRequest],
    evidence: &SceneGroundingEvidence,
) -> Option<String> {
    object_image_requests
        .iter()
        .find(|request| request.object.id == asset.object_id)
        .map(|request| request.source_crop_path.clone())
        .or_else(|| {
            evidence.objects.iter().find_map(|object| {
                (object.object_id == asset.object_id).then(|| {
                    object
                        .mask
                        .as_ref()
                        .and_then(|mask| mask.mask_png_path.clone())
                        .or_else(|| {
                            object
                                .mask
                                .as_ref()
                                .and_then(|mask| mask.artifact_path.clone())
                        })
                })?
            })
        })
}

fn generated_image_for_asset(
    asset: &SceneAssetBinding,
    selected_candidates: &[Value],
) -> Option<String> {
    selected_candidates
        .iter()
        .find(|selected| {
            selected.get("object_id").and_then(Value::as_str) == Some(&asset.object_id)
        })
        .and_then(|selected| selected.get("image_path").and_then(Value::as_str))
        .map(str::to_string)
        .or_else(|| asset.source_image_path.clone())
}

fn push_unique_yaw(values: &mut Vec<f32>, yaw: f32) {
    let yaw = normalize_degrees(yaw);
    if values
        .iter()
        .all(|existing| angle_distance_degrees(*existing, yaw) > 1.0e-3)
    {
        values.push(yaw);
    }
}

fn angle_distance_degrees(left: f32, right: f32) -> f32 {
    normalize_degrees(left - right).abs()
}

fn normalize_degrees(mut degrees: f32) -> f32 {
    while degrees > 180.0 {
        degrees -= 360.0;
    }
    while degrees <= -180.0 {
        degrees += 360.0;
    }
    degrees
}

pub(crate) fn canonical_pose_mode_label(mode: SceneCanonicalPoseMode) -> &'static str {
    match mode {
        SceneCanonicalPoseMode::Off => "off",
        SceneCanonicalPoseMode::Heuristic => "heuristic",
        SceneCanonicalPoseMode::RenderSweep => "render-sweep",
        SceneCanonicalPoseMode::Openai => "openai",
        SceneCanonicalPoseMode::Auto => "auto",
    }
}

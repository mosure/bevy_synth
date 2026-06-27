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

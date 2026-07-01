use crate::prelude::*;
use crate::server::McpServer;
use burn_synth_scene::SceneObjectSpec;
use std::fmt::Write as FmtWrite;

impl McpServer {
    pub(crate) fn call_scene_grounding_report(
        &mut self,
        args: SceneGroundingReportCliArgs,
    ) -> Result<Value, String> {
        let started = Instant::now();
        let output_dir = args.output_dir.unwrap_or_else(|| {
            default_scene_output_dir()
                .with_file_name(format!("{}_scene_grounding_report", next_scene_sequence()))
        });
        fs::create_dir_all(&output_dir).map_err(|err| {
            format!(
                "failed to create scene grounding report directory {}: {err}",
                output_dir.display()
            )
        })?;

        if args.cdn_only {
            let cache_root = output_dir.join("model_cache");
            self.config.locate_anything_model_root = output_dir
                .join("cdn_only_model_roots")
                .join("LocateAnything-3B");
            self.config.locate_anything_cache_dir = Some(cache_root.join("locate_anything"));
            self.config.scene_segmentation_model_root = None;
            self.config.scene_segmentation_cache_dir = Some(cache_root.join("segmentation"));
        }

        let manifest = scene_grounding_report_manifest(
            &args.source_scene_path,
            &args.queries,
            &args.expected_counts,
        )?;
        write_json_file(&output_dir.join("manifest.json"), &manifest)
            .map_err(|err| err.to_string())?;
        let source_copy = output_dir.join("source.jpg");
        if !source_copy.exists() {
            fs::copy(&args.source_scene_path, &source_copy).map_err(|err| {
                format!(
                    "failed to copy source image {} to {}: {err}",
                    args.source_scene_path.display(),
                    source_copy.display()
                )
            })?;
        }

        let mut pass_reports = Vec::new();
        for pass_index in 0..=args.warm_runs {
            let pass_label = if pass_index == 0 {
                "cold".to_string()
            } else {
                format!("warm_{pass_index:02}")
            };
            let pass_dir = output_dir.join(format!("pass_{pass_index:02}_{pass_label}"));
            let pass_started = Instant::now();
            fs::create_dir_all(&pass_dir).map_err(|err| {
                format!(
                    "failed to create grounding report pass directory {}: {err}",
                    pass_dir.display()
                )
            })?;

            let mut stage_report = Vec::new();
            let stage_started = Instant::now();
            let category_filter = self.config.scene_category_filter.clone();
            let (mut evidence, locate_report) =
                if args.locator == SceneLocatorProvider::LocateAnything {
                    self.locate_anything_grounding_evidence_with_report(
                        args.locate_anything_backend
                            .unwrap_or(self.config.locate_anything_backend),
                        &manifest,
                        &args.source_scene_path,
                        &pass_dir,
                        &category_filter,
                    )?
                } else {
                    (
                        manifest_grounding_evidence(&manifest),
                        LocateAnythingGroundingReport {
                            artifact_dir: pass_dir.join("manifest_fallback"),
                            detections_path: pass_dir.join("manifest_fallback/detections.json"),
                            overlay_path: pass_dir.join("manifest_fallback/detections_overlay.png"),
                            metadata_path: pass_dir.join("manifest_fallback/metadata.json"),
                            elapsed_ms: 0.0,
                            runtime_cache_hit: true,
                            detection_count: manifest.objects.len(),
                        },
                    )
                };
            record_stage(
                &mut stage_report,
                "locate_anything_grounding",
                stage_started,
            );
            write_json_file(
                &pass_dir.join("grounding_evidence_locate_anything.json"),
                &evidence,
            )
            .map_err(|err| err.to_string())?;

            let mut sam_report = None;
            let mut sam_evidence = None;
            if args.segmentation_provider != SceneSegmentationProvider::None {
                let stage_started = Instant::now();
                let mut segmented = evidence.clone();
                let report = self.segmentation_grounding_evidence(
                    args.segmentation_provider,
                    args.segmentation_precision,
                    args.segmentation_quantization,
                    &mut segmented,
                    &args.source_scene_path,
                    &pass_dir,
                )?;
                record_stage(
                    &mut stage_report,
                    "segmentation_grounding_evidence",
                    stage_started,
                );
                if let Some(report) = report {
                    write_json_file(&pass_dir.join("grounding_evidence_sam2.json"), &segmented)
                        .map_err(|err| err.to_string())?;
                    sam_report = Some(report);
                    sam_evidence = Some(segmented);
                }
            }

            let mut bbox_prompt_report = None;
            if args.bbox_prompt_baseline {
                let stage_started = Instant::now();
                let mut bbox_evidence = evidence.clone();
                let bbox_config = SegmentationGroundingConfig {
                    model: SegmentationModelKind::BboxPrompt,
                    backend: SegmentationRuntimeBackend::BboxPrompt,
                    model_root: None,
                    cache_dir: None,
                    cdn_base_url: None,
                    precision: args
                        .segmentation_precision
                        .unwrap_or(self.config.scene_segmentation_precision)
                        .into(),
                    quantization: args
                        .segmentation_quantization
                        .unwrap_or(self.config.scene_segmentation_quantization)
                        .into(),
                    allow_download: false,
                    require_gpu: false,
                };
                let mut bbox_runtime = SceneGroundingRuntime::default();
                let report = bbox_runtime.segmentation_grounding_evidence(
                    &mut bbox_evidence,
                    &args.source_scene_path,
                    &pass_dir,
                    bbox_config,
                )?;
                record_stage(
                    &mut stage_report,
                    "bbox_prompt_segmentation_baseline",
                    stage_started,
                );
                write_json_file(
                    &pass_dir.join("grounding_evidence_bbox_prompt_masks.json"),
                    &bbox_evidence,
                )
                .map_err(|err| err.to_string())?;
                bbox_prompt_report = Some(report);
            }

            let mut depth_report = None;
            if args.depth_provider == SceneDepthProvider::DepthPro {
                let stage_started = Instant::now();
                let depth = self.depth_pro_grounding_evidence(
                    sam_evidence.as_mut().unwrap_or(&mut evidence),
                    &args.source_scene_path,
                    &pass_dir,
                )?;
                record_stage(
                    &mut stage_report,
                    "depth_pro_grounding_evidence",
                    stage_started,
                );
                depth_report = Some(depth);
            }

            let quality_evidence = sam_evidence.as_ref().unwrap_or(&evidence);
            let quality_report = scene_grounding_quality_report(
                &manifest,
                quality_evidence,
                args.max_bbox_area,
                args.max_mask_coverage,
            );
            write_json_file(&pass_dir.join("quality_report.json"), &quality_report)
                .map_err(|err| err.to_string())?;
            write_json_file(&pass_dir.join("stage_report.json"), &stage_report)
                .map_err(|err| err.to_string())?;

            pass_reports.push(json!({
                "pass_index": pass_index,
                "label": pass_label,
                "output_dir": pass_dir,
                "elapsed_ms": elapsed_ms(pass_started.elapsed()),
                "locate_anything": locate_report,
                "segmentation": sam_report,
                "bbox_prompt_baseline": bbox_prompt_report,
                "depth": depth_report,
                "quality": quality_report,
                "stage_report": stage_report,
            }));
        }

        let response = json!({
            "tool": "scene_grounding_report",
            "source_scene_path": args.source_scene_path,
            "output_dir": output_dir,
            "cdn_only": args.cdn_only,
            "queries": manifest.objects.iter().map(|object| object.label.clone()).collect::<Vec<_>>(),
            "warm_runs": args.warm_runs,
            "locator": args.locator,
            "segmentation_provider": args.segmentation_provider,
            "bbox_prompt_baseline": args.bbox_prompt_baseline,
            "depth_provider": args.depth_provider,
            "elapsed_ms": elapsed_ms(started.elapsed()),
            "passes": pass_reports,
        });
        write_json_file(
            &response["output_dir"]
                .as_str()
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("tmp/runs/scene_grounding_report"))
                .join("summary.json"),
            &response,
        )
        .map_err(|err| err.to_string())?;
        let report_path = write_scene_grounding_review_html(&response)?;
        let mut response = response;
        response["review_html"] = json!(report_path);
        write_json_file(
            &response["output_dir"]
                .as_str()
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("tmp/runs/scene_grounding_report"))
                .join("summary.json"),
            &response,
        )
        .map_err(|err| err.to_string())?;
        Ok(response)
    }
}

pub(crate) fn scene_grounding_report_manifest(
    source_scene_path: &Path,
    queries: &[String],
    expected_counts: &[String],
) -> Result<SceneObjectManifest, String> {
    let mut labels = queries
        .iter()
        .map(|query| query.trim())
        .filter(|query| !query.is_empty())
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    if labels.is_empty() {
        labels = vec!["chair".to_string(), "table".to_string(), "sofa".to_string()];
    }
    labels.sort();
    labels.dedup();

    let expected_counts = parse_expected_counts(expected_counts)?;
    let objects = labels
        .into_iter()
        .map(|label| {
            let id = sanitize_scene_identifier(&label.to_ascii_lowercase());
            let instance_count = expected_counts
                .get(&normalized_report_key(&label))
                .copied()
                .unwrap_or(1)
                .max(1);
            SceneObjectSpec {
                id: id.clone(),
                label: label.clone(),
                aliases: vec![label.clone()],
                bbox: [0.0, 0.0, 1.0, 1.0],
                instances: Vec::new(),
                representative_instance_id: None,
                reuse_group: Some(id),
                instance_count,
                object_prompt: label,
                camera_hint: None,
                rotation_hint_degrees: None,
                target_footprint_m: None,
            }
        })
        .collect();
    Ok(SceneObjectManifest {
        source_scene_path: source_scene_path.display().to_string(),
        scene_calibration: None,
        objects,
    })
}

fn parse_expected_counts(entries: &[String]) -> Result<HashMap<String, usize>, String> {
    let mut counts = HashMap::new();
    for entry in entries {
        let Some((label, count)) = entry.split_once('=') else {
            return Err(format!(
                "expected count `{entry}` must be formatted as query=count"
            ));
        };
        let label = normalized_report_key(label);
        if label.is_empty() {
            return Err(format!("expected count `{entry}` has an empty query"));
        }
        let count = count
            .trim()
            .parse::<usize>()
            .map_err(|err| format!("expected count `{entry}` has invalid count: {err}"))?
            .max(1);
        counts.insert(label, count);
    }
    Ok(counts)
}

fn normalized_report_key(value: &str) -> String {
    value
        .trim()
        .to_ascii_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

pub(crate) fn scene_grounding_quality_report(
    manifest: &SceneObjectManifest,
    evidence: &SceneGroundingEvidence,
    max_bbox_area: f32,
    max_mask_coverage: f32,
) -> Value {
    let mut objects = Vec::new();
    let mut group_warnings = Vec::new();
    let mut warning_count = 0usize;
    let mut detected_count = 0usize;
    let mut masked_count = 0usize;
    for object in &evidence.objects {
        let mut warnings = Vec::new();
        let detection_bbox = object.detection.as_ref().map(|detection| detection.bbox);
        let bbox_area = detection_bbox.map(normalized_bbox_area);
        if object.detection.is_some() {
            detected_count += 1;
        } else {
            warnings.push("missing_detection".to_string());
        }
        if let Some(area) = bbox_area {
            if area > max_bbox_area {
                warnings.push(format!("bbox_area_too_large:{area:.3}>{max_bbox_area:.3}"));
            }
            if area < 0.008 {
                warnings.push(format!(
                    "bbox_area_tiny_requires_depth_filter:{area:.3}<0.008"
                ));
            }
            if area <= 0.0 {
                warnings.push("invalid_bbox_area".to_string());
            }
        }
        if let Some(bbox) = detection_bbox
            && (bbox[0] <= 0.005 || bbox[2] >= 0.995)
        {
            warnings.push("bbox_touches_horizontal_image_edge".to_string());
        }

        let mask_coverage = object.mask.as_ref().and_then(|mask| mask.coverage);
        let mask_area_ratio_to_bbox =
            object
                .mask
                .as_ref()
                .and_then(|mask| match (detection_bbox, mask.coverage) {
                    (Some(bbox), Some(coverage)) => {
                        let bbox_area = normalized_bbox_area(bbox);
                        (bbox_area > 0.0).then_some(coverage / bbox_area)
                    }
                    _ => None,
                });
        if object.mask.is_some() {
            masked_count += 1;
        } else {
            warnings.push("missing_mask".to_string());
        }
        if let Some(coverage) = mask_coverage {
            if coverage > max_mask_coverage {
                warnings.push(format!(
                    "mask_coverage_too_large:{coverage:.3}>{max_mask_coverage:.3}"
                ));
            }
            if coverage <= 0.0 {
                warnings.push("empty_mask".to_string());
            }
        }
        if let Some(ratio) = mask_area_ratio_to_bbox {
            if ratio < 0.05 {
                warnings.push(format!("mask_bbox_fill_too_low:{ratio:.3}<0.050"));
            }
            if ratio > 1.05 {
                warnings.push(format!("mask_exceeds_bbox_area:{ratio:.3}>1.050"));
            }
        }
        warning_count += warnings.len();
        objects.push(json!({
            "object_id": object.object_id,
            "instance_id": object.instance_id,
            "reuse_group": object.reuse_group,
            "detection_label": object.detection.as_ref().map(|detection| detection.label.clone()),
            "source_query": object.detection.as_ref().map(|detection| detection.source_query.clone()),
            "detection_bbox": detection_bbox,
            "bbox_area": bbox_area,
            "mask_bbox": object.mask.as_ref().map(|mask| mask.bbox),
            "mask_area_px": object.mask.as_ref().map(|mask| mask.area_px),
            "mask_coverage": mask_coverage,
            "mask_area_ratio_to_bbox": mask_area_ratio_to_bbox,
            "warnings": warnings,
        }));
    }
    let mut expected_by_object = HashMap::new();
    for object in &manifest.objects {
        expected_by_object.insert(object.id.clone(), object.instance_count.max(1));
    }
    let mut instances_by_object: HashMap<String, Vec<(String, [f32; 4])>> = HashMap::new();
    for object in &evidence.objects {
        let (Some(instance_id), Some(detection)) =
            (object.instance_id.as_ref(), object.detection.as_ref())
        else {
            continue;
        };
        instances_by_object
            .entry(object.object_id.clone())
            .or_default()
            .push((instance_id.clone(), detection.bbox));
    }
    for (object_id, instances) in &instances_by_object {
        if let Some(expected) = expected_by_object.get(object_id)
            && instances.len() != *expected
        {
            group_warnings.push(json!({
                "object_id": object_id,
                "kind": "expected_instance_count_mismatch",
                "expected": expected,
                "actual": instances.len(),
            }));
        }
        for left_index in 0..instances.len() {
            for right_index in left_index + 1..instances.len() {
                let iou = normalized_bbox_iou(instances[left_index].1, instances[right_index].1);
                if iou > 0.50 {
                    group_warnings.push(json!({
                        "object_id": object_id,
                        "kind": "high_same_category_bbox_overlap",
                        "left_instance_id": instances[left_index].0,
                        "right_instance_id": instances[right_index].0,
                        "iou": iou,
                    }));
                }
            }
        }
    }
    warning_count += group_warnings.len();
    json!({
        "status": if warning_count == 0 { "pass" } else { "warn" },
        "object_count": evidence.objects.len(),
        "detected_count": detected_count,
        "masked_count": masked_count,
        "warning_count": warning_count,
        "max_bbox_area": max_bbox_area,
        "max_mask_coverage": max_mask_coverage,
        "group_warnings": group_warnings,
        "objects": objects,
    })
}

pub(crate) fn normalized_bbox_area(bbox: [f32; 4]) -> f32 {
    let width = (bbox[2] - bbox[0]).max(0.0);
    let height = (bbox[3] - bbox[1]).max(0.0);
    width * height
}

pub(crate) fn normalized_bbox_iou(left: [f32; 4], right: [f32; 4]) -> f32 {
    let x0 = left[0].max(right[0]);
    let y0 = left[1].max(right[1]);
    let x1 = left[2].min(right[2]);
    let y1 = left[3].min(right[3]);
    let intersection = normalized_bbox_area([x0, y0, x1, y1]);
    let union = normalized_bbox_area(left) + normalized_bbox_area(right) - intersection;
    if union > 0.0 {
        intersection / union
    } else {
        0.0
    }
}

fn write_scene_grounding_review_html(response: &Value) -> Result<PathBuf, String> {
    let output_dir = response
        .get("output_dir")
        .and_then(Value::as_str)
        .map(PathBuf::from)
        .ok_or_else(|| "scene grounding report response missing output_dir".to_string())?;
    let report_path = output_dir.join("review.html");
    let mut html = String::new();
    let source = "source.jpg";
    writeln!(
        &mut html,
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>Grounding Report</title><style>{}</style></head><body>",
        grounding_report_css()
    )
    .map_err(|err| err.to_string())?;
    writeln!(
        &mut html,
        "<h1>Scene Grounding Report</h1><p><code>{}</code></p>",
        html_escape(
            response
                .get("source_scene_path")
                .and_then(Value::as_str)
                .unwrap_or("")
        )
    )
    .map_err(|err| err.to_string())?;
    writeln!(
        &mut html,
        "<section class=\"summary\"><div><b>CDN only</b><span>{}</span></div><div><b>Elapsed</b><span>{} ms</span></div><div><b>Queries</b><span>{}</span></div></section>",
        response.get("cdn_only").and_then(Value::as_bool).unwrap_or(false),
        response.get("elapsed_ms").and_then(Value::as_u64).unwrap_or_default(),
        html_escape(&response.get("queries").cloned().unwrap_or(Value::Null).to_string())
    )
    .map_err(|err| err.to_string())?;
    writeln!(
        &mut html,
        "<section><h2>Source</h2><img class=\"wide\" src=\"{}\"></section>",
        source
    )
    .map_err(|err| err.to_string())?;

    if let Some(passes) = response.get("passes").and_then(Value::as_array) {
        for pass in passes {
            let pass_dir = pass
                .get("output_dir")
                .and_then(Value::as_str)
                .map(PathBuf::from)
                .unwrap_or_else(|| output_dir.clone());
            let pass_rel = path_relative_to(&output_dir, &pass_dir);
            writeln!(
                &mut html,
                "<section><h2>{}</h2><div class=\"cards\">",
                html_escape(pass.get("label").and_then(Value::as_str).unwrap_or("pass"))
            )
            .map_err(|err| err.to_string())?;
            image_card(
                &mut html,
                "LocateAnything detections",
                &format!("{pass_rel}/locate_anything_burn_native/detections_overlay.png"),
            )?;
            image_card(
                &mut html,
                "SAM2 masks",
                &format!("{pass_rel}/segmentation_sam2/masks_overlay.png"),
            )?;
            image_card(
                &mut html,
                "Bbox prompt masks",
                &format!("{pass_rel}/segmentation_bbox-prompt/masks_overlay.png"),
            )?;
            writeln!(&mut html, "</div>").map_err(|err| err.to_string())?;
            write_pass_metrics_html(&mut html, pass)?;
            writeln!(&mut html, "</section>").map_err(|err| err.to_string())?;
        }
    }
    writeln!(&mut html, "</body></html>").map_err(|err| err.to_string())?;
    fs::write(&report_path, html).map_err(|err| {
        format!(
            "failed to write grounding report HTML {}: {err}",
            report_path.display()
        )
    })?;
    Ok(report_path)
}

fn write_pass_metrics_html(html: &mut String, pass: &Value) -> Result<(), String> {
    let locate = pass.get("locate_anything").unwrap_or(&Value::Null);
    let segmentation = pass.get("segmentation").unwrap_or(&Value::Null);
    let quality = pass.get("quality").unwrap_or(&Value::Null);
    writeln!(
        html,
        "<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>"
    )
    .map_err(|err| err.to_string())?;
    metric_row(
        html,
        "Locate elapsed ms",
        locate.get("elapsed_ms").cloned().unwrap_or(Value::Null),
    )?;
    metric_row(
        html,
        "Locate runtime cache hit",
        locate
            .get("runtime_cache_hit")
            .cloned()
            .unwrap_or(Value::Null),
    )?;
    metric_row(
        html,
        "Detections",
        locate
            .get("detection_count")
            .cloned()
            .unwrap_or(Value::Null),
    )?;
    metric_row(
        html,
        "SAM elapsed ms",
        segmentation
            .get("elapsed_ms")
            .cloned()
            .unwrap_or(Value::Null),
    )?;
    metric_row(
        html,
        "SAM runtime cache hit",
        segmentation
            .get("runtime_cache_hit")
            .cloned()
            .unwrap_or(Value::Null),
    )?;
    metric_row(
        html,
        "Masks",
        segmentation
            .get("mask_count")
            .cloned()
            .unwrap_or(Value::Null),
    )?;
    metric_row(
        html,
        "Quality status",
        quality.get("status").cloned().unwrap_or(Value::Null),
    )?;
    metric_row(
        html,
        "Quality warnings",
        quality.get("warning_count").cloned().unwrap_or(Value::Null),
    )?;
    writeln!(html, "</tbody></table>").map_err(|err| err.to_string())?;
    writeln!(
        html,
        "<details><summary>Object quality</summary><pre>{}</pre></details>",
        html_escape(
            &serde_json::to_string_pretty(quality.get("objects").unwrap_or(&Value::Null))
                .unwrap_or_else(|_| "null".to_string())
        )
    )
    .map_err(|err| err.to_string())
}

fn metric_row(html: &mut String, label: &str, value: Value) -> Result<(), String> {
    writeln!(
        html,
        "<tr><td>{}</td><td><code>{}</code></td></tr>",
        html_escape(label),
        html_escape(&value.to_string())
    )
    .map_err(|err| err.to_string())
}

fn image_card(html: &mut String, title: &str, src: &str) -> Result<(), String> {
    writeln!(
        html,
        "<figure><figcaption>{}</figcaption><img src=\"{}\"></figure>",
        html_escape(title),
        html_escape(src)
    )
    .map_err(|err| err.to_string())
}

fn path_relative_to(base: &Path, path: &Path) -> String {
    path.strip_prefix(base)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

fn html_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn grounding_report_css() -> &'static str {
    r#"
body{margin:0;padding:28px;background:#0f1115;color:#e6e7eb;font:14px/1.45 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
h1,h2{margin:0 0 14px}
section{margin:0 0 28px;padding:18px;background:#171a21;border:1px solid #2a2f3a;border-radius:8px}
code,pre{background:#0b0d12;color:#d8dee9;border-radius:4px}
code{padding:2px 5px}
pre{padding:12px;overflow:auto}
.summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}
.summary div{display:flex;flex-direction:column;gap:4px;padding:12px;background:#10131a;border:1px solid #282d38;border-radius:6px}
.summary span{color:#b9c0cf}
.wide{max-width:100%;border-radius:6px;border:1px solid #2a2f3a}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px}
figure{margin:0}
figcaption{margin:0 0 8px;color:#cbd3e3}
figure img{width:100%;height:auto;border-radius:6px;border:1px solid #2a2f3a;background:#05070a}
table{width:100%;border-collapse:collapse;margin-top:14px}
th,td{padding:8px 10px;border-bottom:1px solid #2a2f3a;text-align:left;vertical-align:top}
details{margin-top:12px}
summary{cursor:pointer;color:#cbd3e3}
"#
}

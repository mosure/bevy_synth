use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use burn::prelude::{Backend, Tensor};
use burn_depth::{
    CameraIntrinsics, DepthCheckpointSource, DepthLoadConfig, DepthLoadEvent, DepthLoadStage,
    DepthModelKind, DepthPipeline, DepthPrecision, DepthRuntimeConfig, ImageBoundingBox,
    backproject_depth, depth_at_bbox_contact_region, pixel_to_ray,
};
use burn_segmentation::BinaryMask;
use burn_synth_scene::{
    DepthEvidenceRef, Detection, EstimatedFloorPlane, ObjectDepthStats, ObjectGroundingEvidence,
    SceneGroundingEvidence, write_json_file,
};
use image::{Rgba, RgbaImage};
use serde_json::{Value, json};

use crate::image_util::{
    bbox_area_normalized, bbox_bottom_center, draw_normalized_bbox, draw_normalized_cross,
    draw_normalized_square, normalized_bbox_to_image_bbox, overlay_color, write_detection_overlay,
};
use crate::types::*;

impl SceneGroundingRuntime {
    pub fn depth_pro_grounding_evidence(
        &mut self,
        evidence: &mut SceneGroundingEvidence,
        source_scene_path: &Path,
        output_dir: &Path,
        config: DepthProGroundingConfig,
    ) -> Result<DepthProGroundingReport, String> {
        let artifact_dir = output_dir.join("depth_pro");
        fs::create_dir_all(&artifact_dir).map_err(|err| {
            format!(
                "failed to create DepthPro artifact directory {}: {err}",
                artifact_dir.display()
            )
        })?;

        let cache_key = DepthProRuntimeCacheKey::from_config(&config);
        let cache_hit = self
            .depth_pro_runtime
            .as_ref()
            .is_some_and(|cached| cached.key == cache_key);
        let mut progress_events = Vec::new();
        let load_started = Instant::now();
        if !cache_hit {
            let precision: DepthPrecision = config.precision.into();
            let load_config = DepthLoadConfig {
                model: DepthModelKind::DepthPro,
                precision,
                checkpoint: DepthCheckpointSource::default_cdn(DepthModelKind::DepthPro, precision),
                cache_dir: config.cache_dir.clone(),
                allow_download: config.allow_download,
                require_gpu: config.require_gpu,
            };
            let device = burn::tensor::Device::<burn_depth::InferenceBackend>::default();
            let pipeline = DepthPipeline::<burn_depth::InferenceBackend>::load_with_progress(
                &device,
                load_config,
                |event| progress_events.push(depth_load_event_json(event)),
            )
            .map_err(|err| format!("load DepthPro pipeline: {err}"))?;
            self.depth_pro_runtime = Some(CachedDepthProRuntime {
                key: cache_key,
                pipeline,
            });
        }
        let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;
        let pipeline = &self
            .depth_pro_runtime
            .as_ref()
            .expect("DepthPro runtime cache initialized")
            .pipeline;

        let image = image::open(source_scene_path).map_err(|err| {
            format!(
                "failed to load depth source image {}: {err}",
                source_scene_path.display()
            )
        })?;
        let infer_started = Instant::now();
        let prediction = pipeline
            .predict(
                image,
                DepthRuntimeConfig {
                    output_size: None,
                    return_gpu_tensors: false,
                },
            )
            .map_err(|err| format!("DepthPro inference failed: {err}"))?;
        let infer_ms = infer_started.elapsed().as_secs_f64() * 1000.0;
        let depth_map = scene_depth_map_from_prediction(prediction)?;
        let mut summary =
            annotate_grounding_evidence_with_depth_map(evidence, &depth_map, "depth_pro");
        summary.provider = "depth-pro".to_string();
        let far_field_filter = filter_far_field_grounding_evidence(evidence, &depth_map);
        summary.annotated_objects = evidence
            .objects
            .iter()
            .filter(|object| object.metric_contact_point_m.is_some())
            .count();
        summary.total_objects = evidence.objects.len();
        let floor_sample_count = summary.floor_sample_count;
        let depth_map_sidecar = write_depth_map_sidecar(&artifact_dir, &depth_map)?;
        let visualizations =
            write_depth_visualizations(source_scene_path, &artifact_dir, &depth_map, evidence)?;

        let summary_path = artifact_dir.join("depth_evidence.json");
        let metadata = json!({
            "provider": "depth-pro",
            "model": "depth-pro",
            "precision": config.precision.label(),
            "load_ms": load_ms,
            "infer_ms": infer_ms,
            "runtime_cache_hit": cache_hit,
            "load_events": progress_events,
            "summary": summary,
            "far_field_filter": far_field_filter,
            "depth_map_sidecar": depth_map_sidecar.metadata,
            "visualizations": visualizations,
        });
        write_json_file(&summary_path, &metadata).map_err(|err| err.to_string())?;

        evidence.depth = Some(DepthEvidenceRef {
            provider: "depth-pro".to_string(),
            model: Some("depth-pro".to_string()),
            precision: Some(config.precision.label().to_string()),
            artifact_path: Some(summary_path.display().to_string()),
            focal_length_px: depth_map.focal_length_px,
            vertical_fov_degrees: depth_map.vertical_fov_degrees,
            image_size: Some([depth_map.width, depth_map.height]),
            depth_map_size: Some([depth_map.width, depth_map.height]),
            floor_sample_count: Some(floor_sample_count),
        });
        evidence.camera.focal_length_px = evidence
            .camera
            .focal_length_px
            .or(depth_map.focal_length_px);
        evidence.camera.vertical_fov_degrees = evidence
            .camera
            .vertical_fov_degrees
            .or(depth_map.vertical_fov_degrees);
        evidence.camera.principal_point = evidence
            .camera
            .principal_point
            .or(Some([depth_map.intrinsics.cx, depth_map.intrinsics.cy]));
        evidence.camera.image_size = Some([depth_map.width, depth_map.height]);

        Ok(DepthProGroundingReport {
            artifact_path: summary_path,
            depth_map_path: depth_map_sidecar.raw_path,
            depth_map_metadata_path: depth_map_sidecar.metadata_path,
            load_ms,
            infer_ms,
            runtime_cache_hit: cache_hit,
            summary,
        })
    }
}

fn depth_load_event_json(event: DepthLoadEvent) -> Value {
    json!({
        "stage": depth_load_stage_label(event.stage),
        "message": event.message,
        "current": event.current,
        "total": event.total,
    })
}

fn depth_load_stage_label(stage: DepthLoadStage) -> &'static str {
    match stage {
        DepthLoadStage::Manifest => "manifest",
        DepthLoadStage::CacheHit => "cache_hit",
        DepthLoadStage::CacheMiss => "cache_miss",
        DepthLoadStage::Part => "part",
        DepthLoadStage::Verify => "verify",
        DepthLoadStage::Deserialize => "deserialize",
        DepthLoadStage::ModelReady => "model_ready",
    }
}

fn scene_depth_map_from_prediction<B: Backend>(
    prediction: burn_depth::inference::DepthPrediction<B>,
) -> Result<SceneDepthMapEvidence, String> {
    let dims: [usize; 3] = prediction.depth_m.shape().dims();
    if dims[0] != 1 {
        return Err(format!(
            "scene depth expects batch size 1, got depth tensor shape {:?}",
            dims
        ));
    }
    let height = dims[1] as u32;
    let width = dims[2] as u32;
    let depth_m = tensor_to_vec_f32(prediction.depth_m)?;
    let expected = width as usize * height as usize;
    if depth_m.len() != expected {
        return Err(format!(
            "scene depth tensor data length mismatch: expected {expected}, got {}",
            depth_m.len()
        ));
    }

    let focal_length_px = prediction
        .focallength_px
        .map(tensor_scalar_f32)
        .transpose()?;
    let fovy_rad = prediction.fovy_rad.map(tensor_scalar_f32).transpose()?;
    let vertical_fov_degrees = prediction
        .intrinsics
        .map(|intrinsics| {
            2.0 * ((height as f32 * 0.5) / intrinsics.fy.max(1.0e-5))
                .atan()
                .to_degrees()
        })
        .or_else(|| fovy_rad.map(f32::to_degrees))
        .or_else(|| {
            focal_length_px.map(|focal| {
                2.0 * ((height as f32 * 0.5) / focal.max(1.0e-5))
                    .atan()
                    .to_degrees()
            })
        });
    let intrinsics = prediction.intrinsics.unwrap_or_else(|| {
        let fy = fovy_rad
            .map(|fovy| (height as f32 * 0.5) / (fovy * 0.5).tan().max(1.0e-5))
            .or(focal_length_px)
            .unwrap_or(width.max(height) as f32);
        let fx = focal_length_px.unwrap_or(fy);
        CameraIntrinsics {
            fx,
            fy,
            cx: (width.saturating_sub(1)) as f32 * 0.5,
            cy: (height.saturating_sub(1)) as f32 * 0.5,
            width,
            height,
        }
    });

    Ok(SceneDepthMapEvidence {
        depth_m,
        width,
        height,
        intrinsics,
        focal_length_px,
        vertical_fov_degrees,
    })
}

fn tensor_to_vec_f32<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Result<Vec<f32>, String> {
    tensor
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|err| format!("read tensor data: {err}"))
}

fn tensor_scalar_f32<B: Backend>(tensor: Tensor<B, 1>) -> Result<f32, String> {
    let values = tensor_to_vec_f32(tensor)?;
    values
        .first()
        .copied()
        .filter(|value| value.is_finite())
        .ok_or_else(|| "depth scalar tensor was empty or non-finite".to_string())
}

pub fn annotate_grounding_evidence_with_depth_map(
    evidence: &mut SceneGroundingEvidence,
    depth_map: &SceneDepthMapEvidence,
    provenance_label: &str,
) -> SceneDepthAnnotationSummary {
    let floor_exclusions = floor_sample_exclusion_bboxes(evidence);
    let floor_report =
        estimate_scene_floor_plane_report_with_exclusions(depth_map, &floor_exclusions)
            .or_else(|| estimate_scene_floor_plane_report_with_exclusions(depth_map, &[]));
    let floor_residual_m = floor_report
        .as_ref()
        .and_then(|report| report.floor.residual_m);
    evidence.floor = floor_report
        .as_ref()
        .map(|report| report.floor)
        .unwrap_or_default();
    let mut annotated_objects = 0usize;
    for object in &mut evidence.objects {
        let Some(detection) = object.detection.clone() else {
            continue;
        };
        let source_bbox = object
            .mask
            .as_ref()
            .map(|mask| mask.bbox)
            .unwrap_or(detection.bbox);
        let bbox = normalized_bbox_to_image_bbox(source_bbox, depth_map.width, depth_map.height);
        let mask_binary = object.mask.as_ref().and_then(|mask| {
            (!mask.mask_rle.is_empty())
                .then(|| {
                    BinaryMask::decode_rle(mask.image_size[0], mask.image_size[1], &mask.mask_rle)
                        .ok()
                })
                .flatten()
        });
        let bbox_stats = mask_binary
            .as_ref()
            .and_then(|mask| depth_stats_for_mask(depth_map, mask))
            .or_else(|| {
                depth_stats_for_bbox(&depth_map.depth_m, depth_map.width, depth_map.height, bbox)
            });
        let contact_pixel = object
            .mask
            .as_ref()
            .and_then(|mask| mask.contact_pixel)
            .or(object.contact_pixel)
            .or(detection.point)
            .unwrap_or_else(|| bbox_bottom_center(source_bbox));
        let contact_depth = mask_binary
            .as_ref()
            .and_then(|mask| depth_at_mask_contact_region(depth_map, mask))
            .or_else(|| {
                depth_at_bbox_contact_region(
                    &depth_map.depth_m,
                    depth_map.width,
                    depth_map.height,
                    bbox,
                )
            })
            .or_else(|| sample_depth_at_normalized_pixel(depth_map, contact_pixel));
        let Some(contact_depth) = contact_depth.filter(|value| value.is_finite() && *value > 0.0)
        else {
            continue;
        };

        let pixel = normalized_to_depth_pixel(contact_pixel, depth_map.width, depth_map.height);
        let ray = pixel_to_ray(pixel[0], pixel[1], depth_map.intrinsics);
        let point = backproject_depth(pixel[0], pixel[1], contact_depth, depth_map.intrinsics);
        let target_footprint =
            estimate_depth_target_footprint(&detection, bbox, contact_depth, depth_map.intrinsics);
        object.depth_stats =
            bbox_stats.map(|(min_m, median_m, max_m, sample_count)| ObjectDepthStats {
                median_m,
                min_m,
                max_m,
                contact_m: Some(contact_depth),
                sample_count: Some(sample_count),
            });
        if object.depth_stats.is_none() {
            object.depth_stats = Some(ObjectDepthStats {
                median_m: contact_depth,
                min_m: contact_depth,
                max_m: contact_depth,
                contact_m: Some(contact_depth),
                sample_count: Some(1),
            });
        }
        object.contact_pixel = Some(contact_pixel);
        object.candidate_floor_contact_rays.push(ray);
        object.metric_contact_point_m = Some(point);
        if object.target_footprint_m.is_none() {
            object.target_footprint_m = target_footprint;
        }
        if !object
            .provenance
            .iter()
            .any(|entry| entry == provenance_label)
        {
            object.provenance.push(provenance_label.to_string());
        }
        annotated_objects += 1;
    }

    SceneDepthAnnotationSummary {
        provider: provenance_label.to_string(),
        annotated_objects,
        total_objects: evidence.objects.len(),
        depth_map_size: [depth_map.width, depth_map.height],
        focal_length_px: depth_map.focal_length_px,
        vertical_fov_degrees: depth_map.vertical_fov_degrees,
        floor_sample_count: floor_report
            .as_ref()
            .map(|report| report.inlier_count)
            .unwrap_or(0),
        floor_candidate_sample_count: floor_report
            .as_ref()
            .map(|report| report.candidate_count)
            .unwrap_or_else(|| collect_floor_depth_samples(depth_map, &floor_exclusions).len()),
        floor_inlier_count: floor_report
            .as_ref()
            .map(|report| report.inlier_count)
            .unwrap_or(0),
        floor_rejected_sample_count: floor_report
            .as_ref()
            .map(|report| report.candidate_count.saturating_sub(report.inlier_count))
            .unwrap_or(0),
        floor_inlier_ratio: floor_report.as_ref().map(|report| report.inlier_ratio),
        floor_estimation_method: floor_report
            .as_ref()
            .map(|report| report.method.to_string()),
        floor_residual_m,
    }
}

pub fn filter_far_field_grounding_evidence(
    evidence: &mut SceneGroundingEvidence,
    depth_map: &SceneDepthMapEvidence,
) -> SceneFarFieldFilterSummary {
    let mut object_depths = evidence
        .objects
        .iter()
        .filter_map(|object| {
            object
                .depth_stats
                .as_ref()
                .and_then(|stats| stats.contact_m.or(Some(stats.median_m)))
                .filter(|value| value.is_finite() && *value > 0.0)
        })
        .collect::<Vec<_>>();
    if object_depths.len() < 2 {
        return SceneFarFieldFilterSummary {
            enabled: false,
            ..SceneFarFieldFilterSummary::default()
        };
    }
    object_depths.sort_by(f32::total_cmp);
    let median = object_depths[object_depths.len() / 2];
    let lower_quartile = object_depths[object_depths.len() / 4];
    let median_threshold = (median * 2.35).clamp(4.5, 8.0);
    let lower_quartile_threshold = (lower_quartile * 2.8).clamp(4.25, 7.0);
    let threshold = median_threshold.min(lower_quartile_threshold);

    let original_detection_count = evidence.detections.len();
    let mut removed_detection_labels = Vec::new();
    evidence.detections.retain(|detection| {
        let remove = detection_is_far_field(detection, depth_map, threshold);
        if remove {
            removed_detection_labels.push(format!(
                "{}@{:.3},{:.3},{:.3},{:.3}",
                detection.label,
                detection.bbox[0],
                detection.bbox[1],
                detection.bbox[2],
                detection.bbox[3]
            ));
        }
        !remove
    });

    let original_object_count = evidence.objects.len();
    let mut removed_object_ids = Vec::new();
    evidence.objects.retain(|object| {
        let remove = object_is_far_field(object, threshold);
        if remove {
            removed_object_ids.push(format!(
                "{}{}",
                object.object_id,
                object
                    .instance_id
                    .as_ref()
                    .map(|id| format!(":{id}"))
                    .unwrap_or_default()
            ));
        }
        !remove
    });

    SceneFarFieldFilterSummary {
        enabled: true,
        threshold_m: Some(threshold),
        median_object_depth_m: Some(median),
        lower_quartile_object_depth_m: Some(lower_quartile),
        removed_detections: original_detection_count.saturating_sub(evidence.detections.len()),
        removed_objects: original_object_count.saturating_sub(evidence.objects.len()),
        removed_detection_labels,
        removed_object_ids,
    }
}

fn detection_is_far_field(
    detection: &Detection,
    depth_map: &SceneDepthMapEvidence,
    threshold_m: f32,
) -> bool {
    let bbox = normalized_bbox_to_image_bbox(detection.bbox, depth_map.width, depth_map.height);
    let area = bbox_area_normalized(detection.bbox);
    let depth =
        depth_at_bbox_contact_region(&depth_map.depth_m, depth_map.width, depth_map.height, bbox)
            .or_else(|| {
                depth_stats_for_bbox(&depth_map.depth_m, depth_map.width, depth_map.height, bbox)
                    .map(|(_, median, _, _)| median)
            });
    let Some(depth) = depth.filter(|value| value.is_finite() && *value > 0.0) else {
        return false;
    };
    depth > threshold_m && area < 0.08
}

fn object_is_far_field(object: &ObjectGroundingEvidence, threshold_m: f32) -> bool {
    let Some(stats) = object.depth_stats.as_ref() else {
        return false;
    };
    let depth = stats.contact_m.unwrap_or(stats.median_m);
    if !depth.is_finite() || depth <= threshold_m {
        return false;
    }
    let area = object
        .detection
        .as_ref()
        .map(|detection| bbox_area_normalized(detection.bbox))
        .unwrap_or(0.0);
    area < 0.08
}

pub fn estimate_scene_floor_plane(
    depth_map: &SceneDepthMapEvidence,
) -> Option<EstimatedFloorPlane> {
    estimate_scene_floor_plane_with_exclusions(depth_map, &[]).map(|(floor, _)| floor)
}

pub(crate) fn estimate_scene_floor_plane_with_exclusions(
    depth_map: &SceneDepthMapEvidence,
    exclusion_bboxes: &[[f32; 4]],
) -> Option<(EstimatedFloorPlane, usize)> {
    estimate_scene_floor_plane_report_with_exclusions(depth_map, exclusion_bboxes)
        .map(|report| (report.floor, report.inlier_count))
}

#[derive(Clone, Copy, Debug)]
struct FloorDepthSample {
    normalized: [f32; 2],
    point: [f32; 3],
}

#[derive(Clone, Debug)]
struct FloorPlaneFitReport {
    floor: EstimatedFloorPlane,
    candidate_count: usize,
    inlier_count: usize,
    inlier_ratio: f32,
    inlier_indices: Vec<usize>,
    method: &'static str,
}

#[derive(Clone, Copy, Debug)]
struct CandidatePlane {
    normal: [f32; 3],
    d: f32,
}

fn estimate_scene_floor_plane_report_with_exclusions(
    depth_map: &SceneDepthMapEvidence,
    exclusion_bboxes: &[[f32; 4]],
) -> Option<FloorPlaneFitReport> {
    let samples = collect_floor_depth_samples(depth_map, exclusion_bboxes);
    fit_floor_plane_from_samples(&samples)
}

fn collect_floor_depth_samples(
    depth_map: &SceneDepthMapEvidence,
    exclusion_bboxes: &[[f32; 4]],
) -> Vec<FloorDepthSample> {
    let mut points = Vec::new();
    let step_x = (depth_map.width / 64).max(1);
    let step_y = (depth_map.height / 48).max(1);
    let y_start = (depth_map.height as f32 * 0.58).floor() as u32;
    for y in (y_start..depth_map.height).step_by(step_y as usize) {
        for x in (0..depth_map.width).step_by(step_x as usize) {
            let normalized = [
                x as f32 / depth_map.width.saturating_sub(1).max(1) as f32,
                y as f32 / depth_map.height.saturating_sub(1).max(1) as f32,
            ];
            if floor_sample_excluded(normalized, exclusion_bboxes) {
                continue;
            }
            let index = y as usize * depth_map.width as usize + x as usize;
            let depth = depth_map.depth_m.get(index).copied().unwrap_or_default();
            if depth.is_finite() && depth > 0.0 {
                if !floor_sample_depth_is_locally_smooth(depth_map, x, y, depth) {
                    continue;
                }
                points.push(FloorDepthSample {
                    normalized,
                    point: backproject_depth(
                        x as f32 + 0.5,
                        y as f32 + 0.5,
                        depth,
                        depth_map.intrinsics,
                    ),
                });
            }
        }
    }
    points
}

fn fit_floor_plane_from_samples(samples: &[FloorDepthSample]) -> Option<FloorPlaneFitReport> {
    let min_samples = if samples.len() < 32 { 6 } else { 32 };
    if samples.len() < min_samples {
        return None;
    }

    let mut best: Option<(CandidatePlane, Vec<usize>, f32, f32)> = None;
    for plane in deterministic_floor_plane_candidates(samples) {
        if plane.normal[1] < 0.48 {
            continue;
        }
        let (inlier_indices, residual) = floor_plane_inliers(samples, plane);
        let inlier_ratio = inlier_indices.len() as f32 / samples.len() as f32;
        if inlier_indices.len() < min_samples || inlier_ratio < 0.16 {
            continue;
        }
        let score =
            inlier_indices.len() as f32 * 8.0 + plane.normal[1] * 48.0 + inlier_ratio * 40.0
                - residual * 180.0;
        let should_replace = best
            .as_ref()
            .is_none_or(|(_, _, best_residual, best_score)| {
                score > *best_score + 1.0
                    || ((score - *best_score).abs() <= 1.0 && residual < *best_residual)
            });
        if should_replace {
            best = Some((plane, inlier_indices, residual, score));
        }
    }

    let (mut plane, mut inlier_indices, _, _) = best?;
    plane = floor_plane_with_median_offset(samples, plane, &inlier_indices)?;
    let (refined_inliers, _) = floor_plane_inliers(samples, plane);
    if refined_inliers.len() >= min_samples {
        inlier_indices = refined_inliers;
    };
    let inlier_ratio = inlier_indices.len() as f32 / samples.len() as f32;
    let residual = floor_plane_mean_residual(samples, plane, &inlier_indices)?;
    let confidence = floor_plane_confidence(plane.normal[1], residual, inlier_ratio);
    let floor = EstimatedFloorPlane {
        normal: plane.normal,
        distance_m: plane.d,
        residual_m: Some(residual),
        confidence: Some(confidence),
    };
    Some(FloorPlaneFitReport {
        floor,
        candidate_count: samples.len(),
        inlier_count: inlier_indices.len(),
        inlier_ratio,
        inlier_indices,
        method: "deterministic_object_excluded_depth_ransac",
    })
}

fn deterministic_floor_plane_candidates(samples: &[FloorDepthSample]) -> Vec<CandidatePlane> {
    let mut candidates = Vec::new();
    let n = samples.len();
    let push_candidate = |candidates: &mut Vec<CandidatePlane>, a: usize, b: usize, c: usize| {
        if a == b || b == c || a == c {
            return;
        }
        if let Some(plane) = plane_from_points(samples[a].point, samples[b].point, samples[c].point)
        {
            candidates.push(plane);
        }
    };

    let iteration_count = if n < 64 { 96 } else { 256 };
    for i in 0..iteration_count {
        let a = (i * 37 + 11) % n;
        let b = (i * 73 + 19) % n;
        let c = (i * 109 + 23) % n;
        push_candidate(&mut candidates, a, b, c);
    }
    for &(a_q, b_q, c_q) in &[
        (0.05, 0.50, 0.95),
        (0.10, 0.45, 0.90),
        (0.15, 0.55, 0.85),
        (0.20, 0.60, 0.98),
        (0.02, 0.35, 0.70),
    ] {
        let index = |q: f32| ((n.saturating_sub(1)) as f32 * q).round() as usize;
        push_candidate(&mut candidates, index(a_q), index(b_q), index(c_q));
    }
    if let Some(horizontal) = horizontal_floor_plane_candidate(samples) {
        candidates.push(horizontal);
    }
    candidates
}

fn horizontal_floor_plane_candidate(samples: &[FloorDepthSample]) -> Option<CandidatePlane> {
    let mut offsets = samples
        .iter()
        .map(|sample| -sample.point[1])
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if offsets.is_empty() {
        return None;
    }
    offsets.sort_by(f32::total_cmp);
    Some(CandidatePlane {
        normal: [0.0, 1.0, 0.0],
        d: offsets[offsets.len() / 2],
    })
}

fn plane_from_points(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> Option<CandidatePlane> {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let normal = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ];
    let len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    if !len.is_finite() || len <= 1.0e-6 {
        return None;
    }
    let mut normal = [normal[0] / len, normal[1] / len, normal[2] / len];
    if normal[1] < 0.0 {
        normal = [-normal[0], -normal[1], -normal[2]];
    }
    let d = -(normal[0] * a[0] + normal[1] * a[1] + normal[2] * a[2]);
    (normal.iter().all(|value| value.is_finite()) && d.is_finite())
        .then_some(CandidatePlane { normal, d })
}

fn floor_plane_inliers(samples: &[FloorDepthSample], plane: CandidatePlane) -> (Vec<usize>, f32) {
    let mut inliers = Vec::new();
    let threshold = floor_inlier_threshold(samples);
    let mut residual_sum = 0.0f32;
    for (index, sample) in samples.iter().enumerate() {
        let distance = point_plane_distance(sample.point, plane);
        if distance <= threshold {
            inliers.push(index);
            residual_sum += distance;
        }
    }
    let residual = if inliers.is_empty() {
        f32::INFINITY
    } else {
        residual_sum / inliers.len() as f32
    };
    (inliers, residual)
}

fn floor_plane_with_median_offset(
    samples: &[FloorDepthSample],
    plane: CandidatePlane,
    inlier_indices: &[usize],
) -> Option<CandidatePlane> {
    if inlier_indices.is_empty() {
        return None;
    }
    let mut offsets = inlier_indices
        .iter()
        .filter_map(|index| samples.get(*index))
        .map(|sample| {
            -plane.normal[0] * sample.point[0]
                - plane.normal[1] * sample.point[1]
                - plane.normal[2] * sample.point[2]
        })
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if offsets.is_empty() {
        return None;
    }
    offsets.sort_by(f32::total_cmp);
    Some(CandidatePlane {
        normal: plane.normal,
        d: offsets[offsets.len() / 2],
    })
}

fn floor_plane_mean_residual(
    samples: &[FloorDepthSample],
    plane: CandidatePlane,
    inlier_indices: &[usize],
) -> Option<f32> {
    if inlier_indices.is_empty() {
        return None;
    }
    let sum = inlier_indices
        .iter()
        .filter_map(|index| samples.get(*index))
        .map(|sample| point_plane_distance(sample.point, plane))
        .sum::<f32>();
    Some(sum / inlier_indices.len() as f32)
}

fn point_plane_distance(point: [f32; 3], plane: CandidatePlane) -> f32 {
    (plane.normal[0] * point[0] + plane.normal[1] * point[1] + plane.normal[2] * point[2] + plane.d)
        .abs()
}

fn floor_inlier_threshold(samples: &[FloorDepthSample]) -> f32 {
    let mut depths = samples
        .iter()
        .map(|sample| sample.point[2])
        .filter(|depth| depth.is_finite() && *depth > 0.0)
        .collect::<Vec<_>>();
    if depths.is_empty() {
        return 0.12;
    }
    depths.sort_by(f32::total_cmp);
    let median_depth = depths[depths.len() / 2];
    (0.055 + median_depth * 0.025).clamp(0.08, 0.22)
}

fn floor_plane_confidence(normal_y: f32, residual: f32, inlier_ratio: f32) -> f32 {
    let residual_score = (1.0 - residual / 0.18).clamp(0.0, 1.0);
    let inlier_score = (inlier_ratio / 0.45).clamp(0.0, 1.0);
    let normal_score = ((normal_y - 0.45) / 0.50).clamp(0.0, 1.0);
    (0.48 * residual_score + 0.34 * inlier_score + 0.18 * normal_score).clamp(0.0, 1.0)
}

fn floor_sample_depth_is_locally_smooth(
    depth_map: &SceneDepthMapEvidence,
    x: u32,
    y: u32,
    center_depth: f32,
) -> bool {
    let radius = 2u32;
    let mut values = Vec::new();
    let y0 = y.saturating_sub(radius);
    let y1 = y
        .saturating_add(radius)
        .min(depth_map.height.saturating_sub(1));
    let x0 = x.saturating_sub(radius);
    let x1 = x
        .saturating_add(radius)
        .min(depth_map.width.saturating_sub(1));
    for yy in y0..=y1 {
        let row = yy as usize * depth_map.width as usize;
        for xx in x0..=x1 {
            let value = depth_map.depth_m[row + xx as usize];
            if value.is_finite() && value > 0.0 {
                values.push(value);
            }
        }
    }
    if values.len() < 5 {
        return true;
    }
    values.sort_by(f32::total_cmp);
    let median = values[values.len() / 2];
    let threshold = (0.10 + center_depth * 0.18).clamp(0.20, 1.50);
    (center_depth - median).abs() <= threshold
}

pub(crate) fn floor_sample_exclusion_bboxes(evidence: &SceneGroundingEvidence) -> Vec<[f32; 4]> {
    let mut bboxes = evidence
        .detections
        .iter()
        .map(|detection| detection.bbox)
        .collect::<Vec<_>>();
    for object in &evidence.objects {
        if let Some(mask) = object.mask.as_ref()
            && !bboxes.iter().any(|bbox| bbox == &mask.bbox)
        {
            bboxes.push(mask.bbox);
        }
        if let Some(detection) = object.detection.as_ref()
            && !bboxes.iter().any(|bbox| bbox == &detection.bbox)
        {
            bboxes.push(detection.bbox);
        }
    }
    bboxes
}

fn floor_sample_excluded(pixel: [f32; 2], exclusion_bboxes: &[[f32; 4]]) -> bool {
    const MARGIN: f32 = 0.04;
    exclusion_bboxes.iter().any(|bbox| {
        let x0 = bbox[0].min(bbox[2]).clamp(0.0, 1.0);
        let x1 = bbox[0].max(bbox[2]).clamp(0.0, 1.0);
        let y0 = bbox[1].min(bbox[3]).clamp(0.0, 1.0);
        let y1 = bbox[1].max(bbox[3]).clamp(0.0, 1.0);
        pixel[0] >= (x0 - MARGIN).max(0.0)
            && pixel[0] <= (x1 + MARGIN).min(1.0)
            && pixel[1] >= (y0 - MARGIN).max(0.0)
            && pixel[1] <= (y1 + MARGIN).min(1.0)
    })
}

fn write_depth_visualizations(
    source_scene_path: &Path,
    artifact_dir: &Path,
    depth_map: &SceneDepthMapEvidence,
    evidence: &SceneGroundingEvidence,
) -> Result<Value, String> {
    let depth_path = artifact_dir.join("depth_meters_visual.png");
    write_depth_map_visualization(depth_map, &depth_path)?;

    let contacts_path = artifact_dir.join("depth_contacts_overlay.png");
    write_depth_contacts_overlay(source_scene_path, evidence, &contacts_path)?;

    let floor_path = artifact_dir.join("floor_samples_overlay.png");
    write_floor_samples_overlay(source_scene_path, depth_map, evidence, &floor_path)?;

    let filtered_detections_path = artifact_dir.join("filtered_detections_overlay.png");
    write_detection_overlay(
        source_scene_path,
        &evidence.detections,
        &filtered_detections_path,
    )?;

    Ok(json!({
        "depth_meters_visual": depth_path,
        "depth_contacts_overlay": contacts_path,
        "floor_samples_overlay": floor_path,
        "filtered_detections_overlay": filtered_detections_path,
    }))
}

#[derive(Clone, Debug)]
pub(crate) struct DepthMapSidecarArtifacts {
    pub(crate) raw_path: PathBuf,
    pub(crate) metadata_path: PathBuf,
    pub(crate) metadata: Value,
}

pub(crate) fn write_depth_map_sidecar(
    artifact_dir: &Path,
    depth_map: &SceneDepthMapEvidence,
) -> Result<DepthMapSidecarArtifacts, String> {
    fs::create_dir_all(artifact_dir).map_err(|err| {
        format!(
            "failed to create depth sidecar directory {}: {err}",
            artifact_dir.display()
        )
    })?;
    let raw_path = artifact_dir.join("depth_meters_f32le.bin");
    let metadata_path = artifact_dir.join("depth_meters_f32le.json");
    let expected = depth_map.width as usize * depth_map.height as usize;
    if depth_map.depth_m.len() != expected {
        return Err(format!(
            "depth sidecar shape mismatch: expected {expected} values for {}x{}, got {}",
            depth_map.width,
            depth_map.height,
            depth_map.depth_m.len()
        ));
    }

    let mut writer = fs::File::create(&raw_path).map_err(|err| {
        format!(
            "failed to create depth sidecar {}: {err}",
            raw_path.display()
        )
    })?;
    for depth in &depth_map.depth_m {
        writer.write_all(&depth.to_le_bytes()).map_err(|err| {
            format!(
                "failed to write depth sidecar {}: {err}",
                raw_path.display()
            )
        })?;
    }
    writer.flush().map_err(|err| {
        format!(
            "failed to flush depth sidecar {}: {err}",
            raw_path.display()
        )
    })?;

    let finite = depth_map
        .depth_m
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0)
        .collect::<Vec<_>>();
    let (min_m, max_m) = finite
        .iter()
        .copied()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), value| {
            (min.min(value), max.max(value))
        });
    let finite_count = finite.len();
    let metadata = json!({
        "schema_version": 1,
        "encoding": "f32le",
        "unit": "meters",
        "raw_path": raw_path,
        "metadata_path": metadata_path,
        "relative_raw_path": raw_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("depth_meters_f32le.bin"),
        "width": depth_map.width,
        "height": depth_map.height,
        "value_count": depth_map.depth_m.len(),
        "byte_length": depth_map.depth_m.len() * std::mem::size_of::<f32>(),
        "finite_positive_count": finite_count,
        "min_m": (finite_count > 0).then_some(min_m),
        "max_m": (finite_count > 0).then_some(max_m),
        "intrinsics": {
            "fx": depth_map.intrinsics.fx,
            "fy": depth_map.intrinsics.fy,
            "cx": depth_map.intrinsics.cx,
            "cy": depth_map.intrinsics.cy,
            "width": depth_map.intrinsics.width,
            "height": depth_map.intrinsics.height,
        },
        "focal_length_px": depth_map.focal_length_px,
        "vertical_fov_degrees": depth_map.vertical_fov_degrees,
    });
    write_json_file(&metadata_path, &metadata).map_err(|err| err.to_string())?;
    Ok(DepthMapSidecarArtifacts {
        raw_path,
        metadata_path,
        metadata,
    })
}

fn write_depth_map_visualization(
    depth_map: &SceneDepthMapEvidence,
    output_path: &Path,
) -> Result<(), String> {
    let (lo, hi) = depth_visualization_range(&depth_map.depth_m);
    let scale = (hi - lo).max(1.0e-5);
    let mut image = RgbaImage::new(depth_map.width, depth_map.height);
    for y in 0..depth_map.height {
        for x in 0..depth_map.width {
            let index = y as usize * depth_map.width as usize + x as usize;
            let depth = depth_map.depth_m.get(index).copied().unwrap_or_default();
            let pixel = if depth.is_finite() && depth > 0.0 {
                let t = ((depth - lo) / scale).clamp(0.0, 1.0);
                let value = ((1.0 - t) * 255.0).round() as u8;
                Rgba([value, value, value, 255])
            } else {
                Rgba([0, 0, 0, 255])
            };
            image.put_pixel(x, y, pixel);
        }
    }
    image.save(output_path).map_err(|err| {
        format!(
            "failed to write depth visualization {}: {err}",
            output_path.display()
        )
    })
}

fn write_depth_contacts_overlay(
    source_scene_path: &Path,
    evidence: &SceneGroundingEvidence,
    output_path: &Path,
) -> Result<(), String> {
    let mut image = image::open(source_scene_path)
        .map_err(|err| {
            format!(
                "failed to load source image for depth contact overlay {}: {err}",
                source_scene_path.display()
            )
        })?
        .to_rgba8();
    for (index, object) in evidence.objects.iter().enumerate() {
        let color = overlay_color(index);
        if let Some(detection) = object.detection.as_ref() {
            draw_normalized_bbox(&mut image, detection.bbox, color, 3);
        }
        if let Some(contact_pixel) = object.contact_pixel {
            draw_normalized_cross(&mut image, contact_pixel, Rgba([255, 128, 0, 255]), 10);
        }
    }
    image.save(output_path).map_err(|err| {
        format!(
            "failed to write depth contact overlay {}: {err}",
            output_path.display()
        )
    })
}

fn write_floor_samples_overlay(
    source_scene_path: &Path,
    depth_map: &SceneDepthMapEvidence,
    evidence: &SceneGroundingEvidence,
    output_path: &Path,
) -> Result<(), String> {
    let mut image = image::open(source_scene_path)
        .map_err(|err| {
            format!(
                "failed to load source image for floor overlay {}: {err}",
                source_scene_path.display()
            )
        })?
        .to_rgba8();
    let exclusion_bboxes = floor_sample_exclusion_bboxes(evidence);
    let samples = collect_floor_depth_samples(depth_map, &exclusion_bboxes);
    let fit = fit_floor_plane_from_samples(&samples);
    let mut inlier_flags = vec![false; samples.len()];
    if let Some(fit) = fit.as_ref() {
        for index in &fit.inlier_indices {
            if let Some(flag) = inlier_flags.get_mut(*index) {
                *flag = true;
            }
        }
    }
    for (index, sample) in samples.iter().enumerate() {
        let color = if inlier_flags.get(index).copied().unwrap_or(false) {
            Rgba([0, 235, 120, 255])
        } else {
            Rgba([255, 164, 0, 210])
        };
        draw_normalized_square(&mut image, sample.normalized, color, 2);
    }
    image.save(output_path).map_err(|err| {
        format!(
            "failed to write floor sample overlay {}: {err}",
            output_path.display()
        )
    })
}

fn depth_visualization_range(depth_m: &[f32]) -> (f32, f32) {
    let step = (depth_m.len() / 65_536).max(1);
    let mut values = depth_m
        .iter()
        .step_by(step)
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0)
        .collect::<Vec<_>>();
    if values.is_empty() {
        return (0.0, 1.0);
    }
    values.sort_by(f32::total_cmp);
    let lo_index = ((values.len().saturating_sub(1)) as f32 * 0.02).round() as usize;
    let hi_index = ((values.len().saturating_sub(1)) as f32 * 0.98).round() as usize;
    let lo = values[lo_index.min(values.len() - 1)];
    let hi = values[hi_index.min(values.len() - 1)];
    if hi > lo { (lo, hi) } else { (lo, lo + 1.0) }
}

fn depth_stats_for_bbox(
    depth_m: &[f32],
    image_width: u32,
    image_height: u32,
    bbox: ImageBoundingBox,
) -> Option<(f32, f32, f32, usize)> {
    if depth_m.len() != image_width as usize * image_height as usize {
        return None;
    }
    let x0 = bbox.x.min(image_width);
    let x1 = bbox.x.saturating_add(bbox.width).min(image_width);
    let y0 = bbox.y.min(image_height);
    let y1 = bbox.y.saturating_add(bbox.height).min(image_height);
    if x0 >= x1 || y0 >= y1 {
        return None;
    }
    let mut values = Vec::new();
    for y in y0..y1 {
        let row = y as usize * image_width as usize;
        for x in x0..x1 {
            let value = depth_m[row + x as usize];
            if value.is_finite() && value > 0.0 {
                values.push(value);
            }
        }
    }
    if values.is_empty() {
        return None;
    }
    values.sort_by(|left, right| left.total_cmp(right));
    Some((
        values[0],
        values[values.len() / 2],
        values[values.len() - 1],
        values.len(),
    ))
}

fn depth_stats_for_mask(
    depth_map: &SceneDepthMapEvidence,
    mask: &BinaryMask,
) -> Option<(f32, f32, f32, usize)> {
    if depth_map.depth_m.len() != depth_map.width as usize * depth_map.height as usize {
        return None;
    }
    let stride = mask_sampling_stride(mask.area_px() as usize, 65_536);
    let mut values = Vec::new();
    for y in (0..mask.height()).step_by(stride) {
        let row = y as usize * mask.width() as usize;
        for x in (0..mask.width()).step_by(stride) {
            if mask.data()[row + x as usize] == 0 {
                continue;
            }
            if let Some(value) = sample_depth_at_mask_pixel(depth_map, mask, x, y) {
                values.push(value);
            }
        }
    }
    if values.is_empty() && stride > 1 {
        for y in 0..mask.height() {
            let row = y as usize * mask.width() as usize;
            for x in 0..mask.width() {
                if mask.data()[row + x as usize] == 0 {
                    continue;
                }
                if let Some(value) = sample_depth_at_mask_pixel(depth_map, mask, x, y) {
                    values.push(value);
                }
            }
        }
    }
    depth_value_stats(values)
}

fn depth_at_mask_contact_region(
    depth_map: &SceneDepthMapEvidence,
    mask: &BinaryMask,
) -> Option<f32> {
    let mut bottom_y = None::<u32>;
    for y in 0..mask.height() {
        let row = y as usize * mask.width() as usize;
        if (0..mask.width()).any(|x| mask.data()[row + x as usize] != 0) {
            bottom_y = Some(y);
        }
    }
    let bottom_y = bottom_y?;
    let band = ((mask.height() as f32) * 0.08).ceil().max(3.0) as u32;
    let y0 = bottom_y.saturating_sub(band);
    let stride = mask_sampling_stride(mask.area_px() as usize, 16_384);
    let mut values = Vec::new();
    for y in (y0..=bottom_y).step_by(stride) {
        let row = y as usize * mask.width() as usize;
        for x in (0..mask.width()).step_by(stride) {
            if mask.data()[row + x as usize] == 0 {
                continue;
            }
            if let Some(value) = sample_depth_at_mask_pixel(depth_map, mask, x, y) {
                values.push(value);
            }
        }
    }
    depth_value_stats(values).map(|(_, median, _, _)| median)
}

fn sample_depth_at_mask_pixel(
    depth_map: &SceneDepthMapEvidence,
    mask: &BinaryMask,
    x: u32,
    y: u32,
) -> Option<f32> {
    let u = (x as f32 + 0.5) / mask.width().max(1) as f32;
    let v = (y as f32 + 0.5) / mask.height().max(1) as f32;
    sample_depth_at_normalized_pixel(depth_map, [u, v])
}

fn depth_value_stats(mut values: Vec<f32>) -> Option<(f32, f32, f32, usize)> {
    values.retain(|value| value.is_finite() && *value > 0.0);
    if values.is_empty() {
        return None;
    }
    values.sort_by(f32::total_cmp);
    Some((
        values[0],
        values[values.len() / 2],
        values[values.len() - 1],
        values.len(),
    ))
}

fn mask_sampling_stride(area_px: usize, target_samples: usize) -> usize {
    if area_px <= target_samples.max(1) {
        return 1;
    }
    ((area_px as f64 / target_samples.max(1) as f64)
        .sqrt()
        .ceil() as usize)
        .max(1)
}

fn sample_depth_at_normalized_pixel(
    depth_map: &SceneDepthMapEvidence,
    pixel: [f32; 2],
) -> Option<f32> {
    let [x, y] = normalized_to_depth_pixel(pixel, depth_map.width, depth_map.height);
    let x = x
        .round()
        .clamp(0.0, depth_map.width.saturating_sub(1) as f32) as u32;
    let y = y
        .round()
        .clamp(0.0, depth_map.height.saturating_sub(1) as f32) as u32;
    let value = depth_map.depth_m[y as usize * depth_map.width as usize + x as usize];
    (value.is_finite() && value > 0.0).then_some(value)
}

fn normalized_to_depth_pixel(pixel: [f32; 2], width: u32, height: u32) -> [f32; 2] {
    [
        pixel[0].clamp(0.0, 1.0) * width.saturating_sub(1) as f32,
        pixel[1].clamp(0.0, 1.0) * height.saturating_sub(1) as f32,
    ]
}

fn estimate_depth_target_footprint(
    detection: &Detection,
    bbox: ImageBoundingBox,
    contact_depth_m: f32,
    intrinsics: CameraIntrinsics,
) -> Option<[f32; 2]> {
    if !contact_depth_m.is_finite() || contact_depth_m <= 0.0 {
        return None;
    }
    let width_m = bbox.width as f32 * contact_depth_m / intrinsics.fx.max(1.0e-5);
    if !width_m.is_finite() || width_m <= 0.0 {
        return None;
    }
    let descriptor = format!("{} {}", detection.label, detection.source_query).to_ascii_lowercase();
    let footprint = if descriptor.contains("sofa")
        || descriptor.contains("couch")
        || descriptor.contains("sectional")
    {
        [width_m.clamp(1.4, 6.5), (width_m * 0.48).clamp(0.8, 2.8)]
    } else if descriptor.contains("conference") && descriptor.contains("table") {
        [width_m.clamp(1.2, 6.5), (width_m * 0.42).clamp(0.7, 2.8)]
    } else if descriptor.contains("table") {
        [width_m.clamp(0.6, 4.5), (width_m * 0.55).clamp(0.45, 2.5)]
    } else if descriptor.contains("chair") || descriptor.contains("seat") {
        [
            width_m.clamp(0.42, 0.95),
            (width_m * 1.05).clamp(0.42, 1.05),
        ]
    } else {
        [width_m.clamp(0.2, 4.0), width_m.clamp(0.2, 4.0)]
    };
    Some(footprint)
}

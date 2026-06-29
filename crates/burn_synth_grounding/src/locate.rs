use std::fs;
use std::path::Path;
use std::time::Instant;

use burn_locate_anything::LocateAnythingDetector;
use burn_synth_scene::{
    Detection, EstimatedCamera, EstimatedFloorPlane, ObjectGroundingEvidence,
    SceneGroundingEvidence, SceneObjectInstanceSpec, SceneObjectManifest, SceneObjectSpec,
    write_json_file,
};
use serde_json::json;

use crate::image_util::{
    bbox_area_normalized, bbox_bottom_center, bbox_center, bbox_iou, union_bbox,
    write_detection_overlay,
};
use crate::types::*;
use crate::{DetectionQuery, LocateAnythingDetection, LocateAnythingRuntime};

impl SceneGroundingRuntime {
    pub fn locate_anything_burn_native_grounding_evidence(
        &mut self,
        manifest: &SceneObjectManifest,
        source_scene_path: &Path,
        output_dir: &Path,
        config: LocateAnythingGroundingConfig,
    ) -> Result<(SceneGroundingEvidence, LocateAnythingGroundingReport), String> {
        let queries = if config.allowed_categories.is_empty() {
            locate_anything_queries(manifest)
        } else {
            locate_anything_queries_for_allowed_categories(&config.allowed_categories)
        };
        if queries.is_empty() {
            return Err(
                "LocateAnything locator requires at least one non-empty allowed category"
                    .to_string(),
            );
        }

        let artifact_dir = output_dir.join("locate_anything_burn_native");
        fs::create_dir_all(&artifact_dir).map_err(|err| {
            format!(
                "failed to create LocateAnything native artifact directory {}: {err}",
                artifact_dir.display()
            )
        })?;
        let image = image::open(source_scene_path).map_err(|err| {
            format!(
                "failed to load LocateAnything source image {}: {err}",
                source_scene_path.display()
            )
        })?;
        let runtime_config = config.runtime_config();
        let cache_key = LocateAnythingBurnNativeCacheKey::from_config(&runtime_config);
        let cache_hit = self
            .locate_anything_burn_native_runtime
            .as_ref()
            .is_some_and(|cached| cached.key == cache_key);
        if !cache_hit {
            let runtime = LocateAnythingRuntime::new(runtime_config.clone())
                .map_err(|err| format!("initialize Burn-native LocateAnything runtime: {err}"))?;
            self.locate_anything_burn_native_runtime = Some(CachedLocateAnythingRuntime {
                key: cache_key,
                runtime,
            });
        }
        let runtime = &mut self
            .locate_anything_burn_native_runtime
            .as_mut()
            .expect("LocateAnything runtime cache initialized")
            .runtime;
        let detection_queries = queries
            .iter()
            .map(|query| DetectionQuery {
                query: query.clone(),
                label_hint: None,
            })
            .collect::<Vec<_>>();
        let started = Instant::now();
        let batches = runtime
            .detect_batch(&image, &detection_queries)
            .map_err(|err| format!("Burn-native LocateAnything detect failed: {err}"))?;
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        let stage_timings = runtime.last_burn_native_stage_timings().cloned();
        let raw_detections = batches
            .into_iter()
            .flatten()
            .map(scene_detection_from_locate_anything)
            .collect::<Vec<_>>();
        let LocateAnythingDetectionFilter {
            detections,
            dropped,
            deduped,
        } = filter_locate_anything_detections(raw_detections.clone(), &queries);
        if detections.is_empty() {
            return Err(format!(
                "Burn-native LocateAnything returned no usable detections for {} queries after category filtering",
                queries.len(),
            ));
        }

        let raw_detections_path = artifact_dir.join("raw_detections.json");
        write_json_file(&raw_detections_path, &raw_detections).map_err(|err| err.to_string())?;
        let raw_overlay_path = artifact_dir.join("raw_detections_overlay.png");
        write_detection_overlay(source_scene_path, &raw_detections, &raw_overlay_path)?;
        let detections_path = artifact_dir.join("detections.json");
        write_json_file(&detections_path, &detections).map_err(|err| err.to_string())?;
        let overlay_path = artifact_dir.join("detections_overlay.png");
        write_detection_overlay(source_scene_path, &detections, &overlay_path)?;
        let filtered_detection_count = detections.len();
        let mut evidence = locate_anything_evidence_from_detections(
            manifest,
            source_scene_path,
            detections,
            "locate_anything_burn_native",
        )?;
        let metadata_path = artifact_dir.join("metadata.json");
        let metadata = json!({
            "provider": "locate_anything_burn_native",
            "model_root": config.model_root,
            "backend": "burn_native",
            "in_token_limit": config.in_token_limit,
            "decode_mode": format!("{:?}", runtime_config.decode_mode),
            "max_new_tokens": runtime_config.max_new_tokens,
            "repetition_penalty": runtime_config.repetition_penalty,
            "top_p": runtime_config.top_p,
            "top_k": runtime_config.top_k,
            "queries": queries,
            "allowed_categories": config.allowed_categories,
            "raw_detection_count": raw_detections.len(),
            "filtered_detection_count": filtered_detection_count,
            "dropped_detections": dropped,
            "deduped_detections": deduped,
            "elapsed_ms": elapsed_ms,
            "stage_timings": stage_timings,
            "runtime_cache_hit": cache_hit,
            "raw_detections_json": raw_detections_path,
            "raw_detections_overlay": raw_overlay_path,
            "detections_json": detections_path,
            "detections_overlay": overlay_path,
            "compile_feature_hint": "build caller with burn_synth_grounding/locate-anything-wgpu for WGPU native execution",
        });
        write_json_file(&metadata_path, &metadata).map_err(|err| err.to_string())?;
        for object in &mut evidence.objects {
            object
                .provenance
                .push("locate_anything_burn_native_scene_ground".to_string());
        }
        let report = LocateAnythingGroundingReport {
            artifact_dir,
            detections_path,
            overlay_path,
            metadata_path,
            elapsed_ms,
            runtime_cache_hit: cache_hit,
            detection_count: evidence.detections.len(),
        };
        Ok((evidence, report))
    }
}

fn scene_detection_from_locate_anything(detection: LocateAnythingDetection) -> Detection {
    Detection {
        label: detection.label,
        bbox: detection.bbox,
        point: detection.point,
        confidence: detection.confidence,
        source_query: detection.source_query,
    }
}

pub fn locate_anything_queries(manifest: &SceneObjectManifest) -> Vec<String> {
    locate_anything_categories(manifest)
}

pub fn default_locate_anything_allowed_categories() -> Vec<String> {
    ["table", "chair", "plant", "sofa", "couch"]
        .into_iter()
        .map(str::to_string)
        .collect()
}

pub fn locate_anything_queries_for_allowed_categories(categories: &[String]) -> Vec<String> {
    let mut queries = Vec::new();
    for category in categories {
        let query = normalized_query_key(category);
        if query.is_empty() || canonical_locate_anything_category(&query).is_none() {
            continue;
        }
        if !queries.iter().any(|existing: &String| existing == &query) {
            queries.push(query);
        }
    }
    queries
}

pub fn locate_anything_categories(manifest: &SceneObjectManifest) -> Vec<String> {
    let mut categories = Vec::new();
    for object in &manifest.objects {
        let Some(query) = locate_anything_category_for_object(object) else {
            continue;
        };
        if !categories
            .iter()
            .any(|existing: &String| existing == &query)
        {
            categories.push(query);
        }
    }
    categories
}

fn locate_anything_category_for_object(object: &SceneObjectSpec) -> Option<String> {
    object
        .aliases
        .iter()
        .chain(std::iter::once(&object.label))
        .map(|candidate| candidate.trim())
        .filter(|candidate| !candidate.is_empty())
        .min_by(|left, right| {
            let left_words = left.split_whitespace().count();
            let right_words = right.split_whitespace().count();
            left_words
                .cmp(&right_words)
                .then_with(|| left.len().cmp(&right.len()))
        })
        .map(str::to_string)
}

#[derive(Debug)]
pub(crate) struct LocateAnythingDetectionFilter {
    pub(crate) detections: Vec<Detection>,
    pub(crate) dropped: Vec<Detection>,
    pub(crate) deduped: Vec<Detection>,
}

pub(crate) fn filter_locate_anything_detections(
    detections: Vec<Detection>,
    queries: &[String],
) -> LocateAnythingDetectionFilter {
    let allowed = queries
        .iter()
        .filter_map(|query| canonical_locate_anything_category(query))
        .collect::<Vec<_>>();
    let mut kept = Vec::new();
    let mut dropped = Vec::new();
    for detection in detections {
        if locate_anything_detection_allowed(&detection, &allowed) {
            kept.push(detection);
        } else {
            dropped.push(detection);
        }
    }

    kept.sort_by(|left, right| {
        canonical_detection_category(left)
            .cmp(&canonical_detection_category(right))
            .then_with(|| {
                right
                    .confidence
                    .unwrap_or(0.0)
                    .total_cmp(&left.confidence.unwrap_or(0.0))
            })
            .then_with(|| {
                bbox_area_normalized(right.bbox).total_cmp(&bbox_area_normalized(left.bbox))
            })
    });

    let mut filtered = Vec::new();
    let mut deduped = Vec::new();
    'candidate: for detection in kept {
        let category = canonical_detection_category(&detection);
        for existing in &filtered {
            if category.is_some()
                && category == canonical_detection_category(existing)
                && bbox_iou(detection.bbox, existing.bbox) >= 0.75
            {
                deduped.push(detection);
                continue 'candidate;
            }
        }
        filtered.push(detection);
    }

    LocateAnythingDetectionFilter {
        detections: filtered,
        dropped,
        deduped,
    }
}

fn locate_anything_detection_allowed(detection: &Detection, allowed: &[&'static str]) -> bool {
    let Some(source_category) = canonical_locate_anything_category(&detection.source_query) else {
        return false;
    };
    if !allowed.iter().any(|category| *category == source_category) {
        return false;
    }
    let label_key = normalized_query_key(&detection.label);
    if locate_anything_generic_label(&label_key) {
        return true;
    }
    canonical_locate_anything_category(&label_key)
        .is_some_and(|label_category| label_category == source_category)
}

fn canonical_detection_category(detection: &Detection) -> Option<&'static str> {
    canonical_locate_anything_category(&detection.label)
        .or_else(|| canonical_locate_anything_category(&detection.source_query))
}

fn locate_anything_generic_label(label: &str) -> bool {
    label.is_empty()
        || matches!(
            label,
            "object" | "objects" | "item" | "items" | "thing" | "target" | "region"
        )
}

fn canonical_locate_anything_category(value: &str) -> Option<&'static str> {
    let key = normalized_query_key(value);
    if key.contains("sofa")
        || key.contains("couch")
        || key.contains("sectional")
        || key.contains("settee")
    {
        return Some("sofa");
    }
    if key.contains("chair")
        || key.contains("seat")
        || key.contains("stool")
        || key.contains("armchair")
    {
        return Some("chair");
    }
    if key.contains("table") || key.contains("desk") {
        return Some("table");
    }
    if key.contains("plant") || key.contains("potted") || key.contains("tree") {
        return Some("plant");
    }
    None
}

pub fn locate_anything_evidence_from_detections(
    manifest: &SceneObjectManifest,
    source_scene_path: &Path,
    detections: Vec<Detection>,
    provenance_label: &str,
) -> Result<SceneGroundingEvidence, String> {
    let image_size = image::image_dimensions(source_scene_path)
        .ok()
        .map(|(width, height)| [width, height]);

    let mut objects = Vec::new();
    for object in &manifest.objects {
        let query_key = normalized_query_key(&object.label);
        let matched = detections
            .iter()
            .filter(|detection| detection_matches_object(object, detection, &query_key))
            .cloned()
            .collect::<Vec<_>>();
        let object_detection = object_detection_from_matches(object, &matched);
        objects.push(ObjectGroundingEvidence {
            object_id: object.id.clone(),
            instance_id: None,
            reuse_group: object.reuse_group.clone(),
            detection: object_detection,
            mask: None,
            asset_id: None,
            contact_pixel: None,
            depth_stats: None,
            candidate_floor_contact_rays: Vec::new(),
            metric_contact_point_m: None,
            target_footprint_m: object.target_footprint_m,
            provenance: if matched.is_empty() {
                vec!["manifest_fallback_missing_detection".to_string()]
            } else {
                vec![provenance_label.to_string()]
            },
        });

        let instance_evidence =
            locate_anything_instance_evidence(object, &matched, provenance_label);
        objects.extend(instance_evidence);
    }

    Ok(SceneGroundingEvidence {
        source_image_path: source_scene_path.display().to_string(),
        depth: None,
        segmentation: None,
        detections,
        camera: EstimatedCamera {
            image_size,
            ..EstimatedCamera::default()
        },
        floor: EstimatedFloorPlane::default(),
        objects,
    })
}

fn locate_anything_instance_evidence(
    object: &SceneObjectSpec,
    detections: &[Detection],
    provenance_label: &str,
) -> Vec<ObjectGroundingEvidence> {
    let mut out = Vec::new();
    let instances = manifest_instances_for_matching(object);
    let mut used_instance_ids = object
        .instances
        .iter()
        .filter_map(|instance| instance.id.clone())
        .collect::<Vec<_>>();

    if !detections.is_empty() && (object.instance_count > 1 || !object.instances.is_empty()) {
        let mut sorted_detections = detections.iter().collect::<Vec<_>>();
        sorted_detections.sort_by(|left, right| {
            bbox_center(left.bbox)[0]
                .total_cmp(&bbox_center(right.bbox)[0])
                .then_with(|| bbox_center(left.bbox)[1].total_cmp(&bbox_center(right.bbox)[1]))
        });
        let mut sorted_instances = instances;
        sorted_instances.sort_by(|left, right| {
            let left_slot = object
                .instances
                .iter()
                .find(|instance| instance.id == left.0)
                .and_then(|instance| instance.slot_index)
                .unwrap_or(usize::MAX);
            let right_slot = object
                .instances
                .iter()
                .find(|instance| instance.id == right.0)
                .and_then(|instance| instance.slot_index)
                .unwrap_or(usize::MAX);
            left_slot
                .cmp(&right_slot)
                .then_with(|| bbox_center(left.1)[0].total_cmp(&bbox_center(right.1)[0]))
        });
        for (index, detection) in sorted_detections.into_iter().enumerate() {
            let instance = sorted_instances.get(index);
            let instance_id = instance
                .and_then(|(id, _, _, _)| id.clone())
                .unwrap_or_else(|| generated_locate_instance_id(&mut used_instance_ids, index + 1));
            let target_footprint_m = instance
                .and_then(|(_, _, _, footprint)| *footprint)
                .or(object.target_footprint_m);
            out.push(ObjectGroundingEvidence {
                object_id: object.id.clone(),
                instance_id: Some(instance_id),
                reuse_group: object.reuse_group.clone(),
                detection: Some(detection.clone()),
                mask: None,
                asset_id: None,
                contact_pixel: detection
                    .point
                    .or_else(|| Some(bbox_bottom_center(detection.bbox))),
                depth_stats: None,
                candidate_floor_contact_rays: Vec::new(),
                metric_contact_point_m: None,
                target_footprint_m,
                provenance: vec![format!("{provenance_label}_instance_detection")],
            });
        }
        return out;
    }

    for (instance_id, bbox, contact, target_footprint_m) in instances {
        let (detection, provenance) = (
            Some(Detection {
                label: object.label.clone(),
                bbox,
                point: contact,
                confidence: None,
                source_query: object.label.clone(),
            }),
            vec!["manifest_fallback_missing_detection".to_string()],
        );
        let contact_pixel = detection
            .as_ref()
            .and_then(|detection| detection.point)
            .or_else(|| {
                detection
                    .as_ref()
                    .map(|detection| bbox_bottom_center(detection.bbox))
            })
            .or(contact);
        out.push(ObjectGroundingEvidence {
            object_id: object.id.clone(),
            instance_id,
            reuse_group: object.reuse_group.clone(),
            detection,
            mask: None,
            asset_id: None,
            contact_pixel,
            depth_stats: None,
            candidate_floor_contact_rays: Vec::new(),
            metric_contact_point_m: None,
            target_footprint_m,
            provenance,
        });
    }
    out
}

fn generated_locate_instance_id(used_instance_ids: &mut Vec<String>, extra_index: usize) -> String {
    let mut index = extra_index.max(1);
    loop {
        let id = format!("locate_{index:02}");
        if !used_instance_ids.iter().any(|existing| existing == &id) {
            used_instance_ids.push(id.clone());
            return id;
        }
        index += 1;
    }
}

fn detection_matches_object(
    object: &SceneObjectSpec,
    detection: &Detection,
    query_key: &str,
) -> bool {
    let detection_label = normalized_query_key(&detection.label);
    let source_query = normalized_query_key(&detection.source_query);
    let detection_category = canonical_detection_category(detection);
    object_label_keys(object).iter().any(|key| {
        let object_category = canonical_locate_anything_category(key);
        detection_label == *key
            || detection_label.contains(key)
            || key.contains(&detection_label)
            || (detection_category.is_some() && detection_category == object_category)
            || (source_query == query_key && detection_label.is_empty())
    })
}

fn object_label_keys(object: &SceneObjectSpec) -> Vec<String> {
    let mut keys = Vec::new();
    let label = normalized_query_key(&object.label);
    if !label.is_empty() {
        keys.push(label);
    }
    for alias in &object.aliases {
        let key = normalized_query_key(alias);
        if !key.is_empty() && !keys.iter().any(|existing| existing == &key) {
            keys.push(key);
        }
    }
    keys
}

type ManifestMatchingInstance = (Option<String>, [f32; 4], Option<[f32; 2]>, Option<[f32; 2]>);

fn manifest_instances_for_matching(object: &SceneObjectSpec) -> Vec<ManifestMatchingInstance> {
    if object.instances.is_empty() {
        return Vec::new();
    }
    object
        .instances
        .iter()
        .map(|instance: &SceneObjectInstanceSpec| {
            (
                instance.id.clone(),
                instance.bbox,
                instance
                    .contact
                    .or_else(|| Some(bbox_bottom_center(instance.bbox))),
                instance.target_footprint_m.or(object.target_footprint_m),
            )
        })
        .collect()
}

fn manifest_detection_for_object(object: &SceneObjectSpec) -> Option<Detection> {
    Some(Detection {
        label: object.label.clone(),
        bbox: object.bbox,
        point: Some(bbox_bottom_center(object.bbox)),
        confidence: None,
        source_query: object.label.clone(),
    })
}

fn union_detection_for_object(object: &SceneObjectSpec, detections: &[Detection]) -> Detection {
    let bbox = detections
        .iter()
        .map(|detection| detection.bbox)
        .reduce(union_bbox)
        .unwrap_or(object.bbox);
    Detection {
        label: object.label.clone(),
        bbox,
        point: Some(bbox_bottom_center(bbox)),
        confidence: detections
            .iter()
            .filter_map(|d| d.confidence)
            .reduce(f32::max),
        source_query: object.label.clone(),
    }
}

fn object_detection_from_matches(
    object: &SceneObjectSpec,
    detections: &[Detection],
) -> Option<Detection> {
    if detections.is_empty() {
        return manifest_detection_for_object(object);
    }
    if object.instance_count <= 1 && object.instances.is_empty() {
        return Some(best_singleton_detection_for_object(object, detections));
    }
    Some(union_detection_for_object(object, detections))
}

fn best_singleton_detection_for_object(
    object: &SceneObjectSpec,
    detections: &[Detection],
) -> Detection {
    detections
        .iter()
        .max_by(|left, right| {
            detection_preference_score(object, left)
                .total_cmp(&detection_preference_score(object, right))
                .then_with(|| {
                    bbox_area_normalized(left.bbox).total_cmp(&bbox_area_normalized(right.bbox))
                })
        })
        .cloned()
        .unwrap_or_else(|| {
            manifest_detection_for_object(object).expect("manifest detection exists")
        })
}

fn detection_preference_score(object: &SceneObjectSpec, detection: &Detection) -> f32 {
    let confidence = detection.confidence.unwrap_or(0.50).clamp(0.0, 1.0);
    let source_key = normalized_query_key(&detection.source_query);
    let label_key = normalized_query_key(&detection.label);
    let label_bonus = if object_label_keys(object)
        .iter()
        .any(|key| source_key == *key || label_key == *key)
    {
        0.10
    } else {
        0.0
    };
    let area = bbox_area_normalized(detection.bbox).clamp(0.0, 1.0);
    confidence + label_bonus + area.min(0.30) * 0.20
}

fn normalized_query_key(value: &str) -> String {
    value.trim().to_ascii_lowercase()
}

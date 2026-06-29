use std::collections::{HashMap, HashSet};
use std::path::Path;

use serde_json::{Value, json};

use crate::*;

pub fn chair_type_grouping_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["groups"],
        "properties": {
            "groups": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["group_id", "label", "description", "member_indices", "confidence"],
                    "properties": {
                        "group_id": { "type": "string" },
                        "label": { "type": "string" },
                        "description": { "type": "string" },
                        "member_indices": {
                            "type": "array",
                            "items": { "type": "integer" }
                        },
                        "confidence": { "type": "number" }
                    }
                }
            }
        }
    })
}

pub fn prepare_chair_type_grouping_request(
    source_scene_path: &Path,
    output_dir: &Path,
    _manifest: &SceneObjectManifest,
    evidence: &SceneGroundingEvidence,
) -> SceneResult<Option<SceneChairTypeGroupingRequest>> {
    let detections = chair_detections(evidence);
    if detections.len() <= 1 {
        return Ok(None);
    }

    let crop_dir = output_dir.join("chair_type_grouping").join("crops");
    let mut items = Vec::with_capacity(detections.len());
    let mut crop_image_paths = Vec::with_capacity(detections.len());
    for (index, detection) in detections.into_iter().enumerate() {
        let instance_id = format!("chair_{:02}", index + 1);
        let object = SceneObjectSpec {
            id: instance_id.clone(),
            label: detection.label.clone(),
            aliases: vec!["chair".to_string()],
            bbox: normalize_bbox(detection.bbox),
            instances: Vec::new(),
            representative_instance_id: None,
            reuse_group: Some("chair".to_string()),
            instance_count: 1,
            object_prompt: "source-observed chair instance".to_string(),
            camera_hint: None,
            rotation_hint_degrees: None,
            target_footprint_m: None,
        };
        let crop_path = crop_scene_object(
            source_scene_path,
            &object,
            &crop_dir.join(format!("{instance_id}_crop_1024.jpg")),
        )?;
        crop_image_paths.push(crop_path.clone());
        items.push(SceneChairTypeCrop {
            index,
            instance_id,
            bbox: normalize_bbox(detection.bbox),
            point: detection.point,
            confidence: detection.confidence,
            crop_path: crop_path.display().to_string(),
            label: detection.label.clone(),
            source_query: detection.source_query.clone(),
        });
    }

    let item_summary = items
        .iter()
        .map(|item| {
            format!(
                "{}: label='{}', query='{}', bbox=[{:.3},{:.3},{:.3},{:.3}]",
                item.index,
                item.label,
                item.source_query,
                item.bbox[0],
                item.bbox[1],
                item.bbox[2],
                item.bbox[3]
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let prompt = format!(
        "Classify source-image chair detections into reusable 3D asset types.\n\
Use image 1 as the full scene context. Images 2..N are per-chair crops in the same order as the list below.\n\
Group chairs together only when they are the same visible chair model/material/frame type; different perspective, occlusion, or left/right side of the same chair type should remain in one group.\n\
Split visually distinct chair kinds, for example black lounge chairs vs light mesh meeting chairs.\n\
Every index must appear in exactly one group. Do not include tables, plants, sofas, lights, windows, or walls.\n\
Return concise labels suitable for generated object prompts.\n\nChair detections:\n{item_summary}",
    );
    Ok(Some(SceneChairTypeGroupingRequest {
        prompt,
        source_scene_path: source_scene_path.to_path_buf(),
        crop_image_paths,
        items,
    }))
}

pub fn apply_chair_type_groups(
    manifest: &SceneObjectManifest,
    evidence: &SceneGroundingEvidence,
    request: &SceneChairTypeGroupingRequest,
    response: &SceneChairTypeGroupingResponse,
) -> SceneResult<(SceneObjectManifest, SceneGroundingEvidence)> {
    let assignment = validated_chair_type_assignment(request, response);
    if assignment.groups.len() <= 1 {
        return Ok((manifest.clone(), evidence.clone()));
    }

    let chair_object_ids = manifest
        .objects
        .iter()
        .filter(|object| scene_object_is_chair(object))
        .map(|object| object.id.clone())
        .collect::<HashSet<_>>();
    let base_object = manifest
        .objects
        .iter()
        .find(|object| scene_object_is_chair(object))
        .cloned()
        .unwrap_or_else(default_chair_object);

    let mut next_manifest = manifest.clone();
    next_manifest
        .objects
        .retain(|object| !scene_object_is_chair(object));

    let mut used_ids = next_manifest
        .objects
        .iter()
        .map(|object| object.id.clone())
        .collect::<HashSet<_>>();
    let mut new_evidence_objects = evidence
        .objects
        .iter()
        .filter(|object| !chair_object_ids.contains(&object.object_id))
        .cloned()
        .collect::<Vec<_>>();

    for (group_index, group) in assignment.groups.iter().enumerate() {
        let id = unique_object_id(&mut used_ids, &format!("chair_{}", group.label));
        let instances = group
            .items
            .iter()
            .enumerate()
            .map(|(slot_index, item)| SceneObjectInstanceSpec {
                id: Some(item.instance_id.clone()),
                bbox: normalize_bbox(item.bbox),
                contact: item.point.or_else(|| Some(bbox_bottom_center(item.bbox))),
                rotation_hint_degrees: None,
                facing_yaw_degrees: None,
                side: Some(SceneInstanceSide::Unknown),
                slot_index: Some(slot_index),
                target_footprint_m: base_object.target_footprint_m,
            })
            .collect::<Vec<_>>();
        let bbox = group
            .items
            .iter()
            .map(|item| normalize_bbox(item.bbox))
            .reduce(union_bbox)
            .unwrap_or(base_object.bbox);
        let representative_instance_id = instances.first().and_then(|instance| instance.id.clone());
        let label = if group.label.trim().is_empty() {
            format!("chair type {}", group_index + 1)
        } else {
            group.label.clone()
        };
        let description = if group.description.trim().is_empty() {
            format!("source-observed {label}")
        } else {
            group.description.clone()
        };
        next_manifest.objects.push(SceneObjectSpec {
            id: id.clone(),
            label: label.clone(),
            aliases: vec!["chair".to_string(), label.clone()],
            bbox,
            instances: instances.clone(),
            representative_instance_id,
            reuse_group: Some(id.clone()),
            instance_count: instances.len().max(1),
            object_prompt: format!(
                "{}. Reusable chair asset for {} visible instance(s). Preserve this exact source-observed chair type; do not mix it with other chair styles.",
                description,
                instances.len().max(1),
            ),
            camera_hint: base_object.camera_hint.clone(),
            rotation_hint_degrees: base_object.rotation_hint_degrees,
            target_footprint_m: base_object.target_footprint_m,
        });

        let group_detection = Detection {
            label: label.clone(),
            bbox,
            point: Some(bbox_bottom_center(bbox)),
            confidence: group
                .items
                .iter()
                .filter_map(|item| item.confidence)
                .max_by(f32::total_cmp),
            source_query: "chair".to_string(),
        };
        new_evidence_objects.push(ObjectGroundingEvidence {
            object_id: id.clone(),
            instance_id: None,
            reuse_group: Some(id.clone()),
            detection: Some(group_detection),
            mask: None,
            asset_id: None,
            contact_pixel: Some(bbox_bottom_center(bbox)),
            depth_stats: None,
            candidate_floor_contact_rays: Vec::new(),
            metric_contact_point_m: None,
            target_footprint_m: base_object.target_footprint_m,
            provenance: vec![
                "locate_anything_chair_detection".to_string(),
                "gpt_chair_type_grouping".to_string(),
            ],
        });
        for item in &group.items {
            let detection = Detection {
                label: item.label.clone(),
                bbox: normalize_bbox(item.bbox),
                point: item.point.or_else(|| Some(bbox_bottom_center(item.bbox))),
                confidence: item.confidence,
                source_query: item.source_query.clone(),
            };
            new_evidence_objects.push(ObjectGroundingEvidence {
                object_id: id.clone(),
                instance_id: Some(item.instance_id.clone()),
                reuse_group: Some(id.clone()),
                detection: Some(detection.clone()),
                mask: None,
                asset_id: None,
                contact_pixel: detection.point,
                depth_stats: None,
                candidate_floor_contact_rays: Vec::new(),
                metric_contact_point_m: None,
                target_footprint_m: base_object.target_footprint_m,
                provenance: vec![
                    "locate_anything_chair_detection".to_string(),
                    "gpt_chair_type_grouping_instance".to_string(),
                ],
            });
        }
    }

    let mut next_evidence = evidence.clone();
    next_evidence.objects = new_evidence_objects;
    Ok((next_manifest, next_evidence))
}

pub fn chair_type_grouping_report(
    request: &SceneChairTypeGroupingRequest,
    response: &SceneChairTypeGroupingResponse,
) -> Value {
    let assignment = validated_chair_type_assignment(request, response);
    json!({
        "requested_chair_count": request.items.len(),
        "group_count": assignment.groups.len(),
        "groups": assignment.groups.iter().map(|group| {
            json!({
                "group_id": group.group_id,
                "label": group.label,
                "description": group.description,
                "confidence": group.confidence,
                "members": group.items.iter().map(|item| {
                    json!({
                        "index": item.index,
                        "instance_id": item.instance_id,
                        "bbox": item.bbox,
                        "crop_path": item.crop_path,
                        "label": item.label,
                        "source_query": item.source_query,
                    })
                }).collect::<Vec<_>>()
            })
        }).collect::<Vec<_>>()
    })
}

#[derive(Clone, Debug)]
struct ValidatedChairTypeAssignment {
    groups: Vec<ValidatedChairTypeGroup>,
}

#[derive(Clone, Debug)]
struct ValidatedChairTypeGroup {
    group_id: String,
    label: String,
    description: String,
    confidence: f32,
    items: Vec<SceneChairTypeCrop>,
}

fn validated_chair_type_assignment(
    request: &SceneChairTypeGroupingRequest,
    response: &SceneChairTypeGroupingResponse,
) -> ValidatedChairTypeAssignment {
    let item_by_index = request
        .items
        .iter()
        .map(|item| (item.index, item))
        .collect::<HashMap<_, _>>();
    let mut used = HashSet::new();
    let mut groups = Vec::new();
    for (group_index, group) in response.groups.iter().enumerate() {
        let mut items = Vec::new();
        for index in &group.member_indices {
            if used.contains(index) {
                continue;
            }
            let Some(item) = item_by_index.get(index) else {
                continue;
            };
            used.insert(*index);
            items.push((*item).clone());
        }
        if items.is_empty() {
            continue;
        }
        groups.push(ValidatedChairTypeGroup {
            group_id: nonempty_or_default(
                &group.group_id,
                &format!("chair_type_{}", group_index + 1),
            ),
            label: nonempty_or_default(&group.label, &format!("chair type {}", group_index + 1)),
            description: group.description.trim().to_string(),
            confidence: group.confidence.clamp(0.0, 1.0),
            items,
        });
    }

    for item in &request.items {
        if used.contains(&item.index) {
            continue;
        }
        groups.push(ValidatedChairTypeGroup {
            group_id: format!("chair_type_unassigned_{}", item.index),
            label: "chair".to_string(),
            description: "source-observed chair instance not assigned by grouping response"
                .to_string(),
            confidence: 0.0,
            items: vec![item.clone()],
        });
    }

    ValidatedChairTypeAssignment { groups }
}

fn chair_detections(evidence: &SceneGroundingEvidence) -> Vec<Detection> {
    evidence
        .detections
        .iter()
        .filter(|detection| detection_is_chair(detection))
        .cloned()
        .collect()
}

fn detection_is_chair(detection: &Detection) -> bool {
    let label = detection.label.to_ascii_lowercase();
    let query = detection.source_query.to_ascii_lowercase();
    text_is_chair(&label) || text_is_chair(&query)
}

fn scene_object_is_chair(object: &SceneObjectSpec) -> bool {
    text_is_chair(&object.label)
        || object.aliases.iter().any(|alias| text_is_chair(alias))
        || text_is_chair(&object.object_prompt)
}

fn text_is_chair(value: &str) -> bool {
    let value = value.to_ascii_lowercase();
    value.contains("chair")
        || value.contains("seat")
        || value.contains("stool")
        || value.contains("armchair")
}

fn default_chair_object() -> SceneObjectSpec {
    SceneObjectSpec {
        id: "chair".to_string(),
        label: "chair".to_string(),
        aliases: vec!["chair".to_string()],
        bbox: [0.4, 0.4, 0.6, 0.8],
        instances: Vec::new(),
        representative_instance_id: None,
        reuse_group: Some("chair".to_string()),
        instance_count: 1,
        object_prompt: "source-observed chair".to_string(),
        camera_hint: None,
        rotation_hint_degrees: None,
        target_footprint_m: None,
    }
}

fn unique_object_id(used_ids: &mut HashSet<String>, value: &str) -> String {
    let base = slug(value);
    let mut id = if base.is_empty() {
        "chair_type".to_string()
    } else {
        base
    };
    if used_ids.insert(id.clone()) {
        return id;
    }
    for index in 2.. {
        id = format!(
            "{}_{}",
            id.trim_end_matches(|c: char| c.is_ascii_digit()),
            index
        );
        if used_ids.insert(id.clone()) {
            return id;
        }
    }
    unreachable!()
}

fn slug(value: &str) -> String {
    let mut out = String::new();
    let mut previous_underscore = false;
    for ch in value.chars().flat_map(char::to_lowercase) {
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
            previous_underscore = false;
        } else if !previous_underscore {
            out.push('_');
            previous_underscore = true;
        }
    }
    out.trim_matches('_').to_string()
}

fn nonempty_or_default(value: &str, fallback: &str) -> String {
    let value = value.trim();
    if value.is_empty() {
        fallback.to_string()
    } else {
        value.to_string()
    }
}

fn normalize_bbox(bbox: [f32; 4]) -> [f32; 4] {
    [
        bbox[0].min(bbox[2]).clamp(0.0, 1.0),
        bbox[1].min(bbox[3]).clamp(0.0, 1.0),
        bbox[0].max(bbox[2]).clamp(0.0, 1.0),
        bbox[1].max(bbox[3]).clamp(0.0, 1.0),
    ]
}

fn bbox_bottom_center(bbox: [f32; 4]) -> [f32; 2] {
    [(bbox[0] + bbox[2]) * 0.5, bbox[3]]
}

fn union_bbox(left: [f32; 4], right: [f32; 4]) -> [f32; 4] {
    [
        left[0].min(right[0]),
        left[1].min(right[1]),
        left[2].max(right[2]),
        left[3].max(right[3]),
    ]
}

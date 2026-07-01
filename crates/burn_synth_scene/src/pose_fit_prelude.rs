pub(crate) use std::collections::HashMap;
#[cfg(test)]
pub(crate) use std::env;
pub(crate) use std::fs;
pub(crate) use std::path::{Path, PathBuf};
pub(crate) use std::time::Instant;
#[cfg(test)]
pub(crate) use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) use burn_segmentation::BinaryMask;
pub(crate) use burn_synth_render::mesh::RenderMesh;
pub(crate) use serde_json::{Value, json};

pub(crate) use crate::depth_sidecar::{
    DenseDepthCrop, LoadedSceneDepthMap, dense_depth_comparison, dense_depth_crop_from_sidecar,
    dense_depth_crop_report, load_scene_depth_map_sidecar, loaded_depth_map_report,
    resample_dense_depth_crop,
};
pub(crate) use crate::{
    GroundedSceneLayout, GroundedScenePlacement, ObjectGroundingEvidence, ObjectMaskEvidence,
    SceneAssetBinding, SceneFinalYawRefinementMode, SceneGroundingEvidence, SceneObjectManifest,
    SceneObjectPoseRefinementMode, SceneObjectPoseRefinementSet, ScenePoseFitMode,
    SceneRotationFitMode, SceneRotationSelectionResponse, SceneScalePolicy, write_json_file,
};

pub(crate) fn json_array3(value: &Value) -> Option<[f32; 3]> {
    let array = value.as_array()?;
    if array.len() != 3 {
        return None;
    }
    Some([
        array[0].as_f64()? as f32,
        array[1].as_f64()? as f32,
        array[2].as_f64()? as f32,
    ])
    .filter(|items| items.iter().all(|item| item.is_finite()))
}

pub(crate) fn json_array4(value: &Value) -> Option<[f32; 4]> {
    let array = value.as_array()?;
    if array.len() != 4 {
        return None;
    }
    Some([
        array[0].as_f64()? as f32,
        array[1].as_f64()? as f32,
        array[2].as_f64()? as f32,
        array[3].as_f64()? as f32,
    ])
    .filter(|items| items.iter().all(|item| item.is_finite()))
}

pub(crate) fn quat_from_y_degrees(degrees: f32) -> [f32; 4] {
    let half = degrees.to_radians() * 0.5;
    [0.0, half.sin(), 0.0, half.cos()]
}

pub(crate) fn quat_y_degrees(quat: [f32; 4]) -> f32 {
    let [_, y, _, w] = quat;
    (2.0 * y.atan2(w)).to_degrees()
}

pub(crate) fn normalize_reused_command_scales(
    commands: &mut [Value],
    scale_policy: SceneScalePolicy,
) {
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
            (count > 1).then_some((
                key,
                scale_policy.apply_to_scale([
                    (sum[0] / count as f32).clamp(0.05, 20.0),
                    (sum[1] / count as f32).clamp(0.05, 20.0),
                    (sum[2] / count as f32).clamp(0.05, 20.0),
                ]),
            ))
        })
        .collect::<HashMap<_, _>>();
    for command in commands {
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
        if let Some(scale) = repeated_scale.get(group_key).copied() {
            command["scale"] = json!(scale);
        }
    }
}

pub(crate) fn ensure_parent_dir(path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

pub(crate) fn normalize_degrees(mut degrees: f32) -> f32 {
    while degrees > 180.0 {
        degrees -= 360.0;
    }
    while degrees <= -180.0 {
        degrees += 360.0;
    }
    degrees
}

pub(crate) fn bbox_center(bbox: [f32; 4]) -> [f32; 2] {
    [(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5]
}

pub(crate) fn bbox_area(bbox: [f32; 4]) -> f32 {
    (bbox[2] - bbox[0]).max(0.0) * (bbox[3] - bbox[1]).max(0.0)
}

pub(crate) fn bbox_aspect(bbox: [f32; 4]) -> f32 {
    let width = (bbox[2] - bbox[0]).abs().max(1.0e-5);
    let height = (bbox[3] - bbox[1]).abs().max(1.0e-5);
    width / height
}

pub(crate) fn distance2(left: [f32; 2], right: [f32; 2]) -> f32 {
    let dx = left[0] - right[0];
    let dy = left[1] - right[1];
    dx * dx + dy * dy
}

pub(crate) fn safe_log2_ratio(observed: f32, expected: f32) -> f32 {
    if observed > 0.0 && expected > 0.0 {
        (observed / expected).log2()
    } else {
        0.0
    }
}

pub(crate) fn normalized_bbox_iou(left: [f32; 4], right: [f32; 4]) -> f32 {
    let x0 = left[0].max(right[0]);
    let y0 = left[1].max(right[1]);
    let x1 = left[2].min(right[2]);
    let y1 = left[3].min(right[3]);
    let intersection = (x1 - x0).max(0.0) * (y1 - y0).max(0.0);
    let union = bbox_area(left) + bbox_area(right) - intersection;
    if union <= 1.0e-8 {
        0.0
    } else {
        (intersection / union).clamp(0.0, 1.0)
    }
}

pub(crate) fn normalize3(value: [f32; 3]) -> Option<[f32; 3]> {
    let len = (value[0] * value[0] + value[1] * value[1] + value[2] * value[2]).sqrt();
    (len.is_finite() && len > 1.0e-6).then_some([value[0] / len, value[1] / len, value[2] / len])
}

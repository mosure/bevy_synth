use std::collections::{HashMap, HashSet};
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use image::GenericImageView;
use image::codecs::jpeg::JpegEncoder;
use image::imageops::FilterType;
use serde::Serialize;
use serde_json::{Value, json};

use crate::*;

static METRIC_WRITE_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

pub fn object_manifest_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["source_scene_path", "scene_calibration", "objects"],
        "properties": {
            "source_scene_path": { "type": "string" },
            "scene_calibration": {
                "type": ["object", "null"],
                "additionalProperties": false,
                "required": ["table_center", "table_axis_degrees", "table_size_m", "camera_yaw_degrees", "camera_pitch_degrees", "camera_radius_m", "vertical_fov_degrees"],
                "properties": {
                    "table_center": { "type": ["array", "null"], "items": { "type": "number" }, "minItems": 2, "maxItems": 2 },
                    "table_axis_degrees": { "type": ["number", "null"] },
                    "table_size_m": { "type": ["array", "null"], "items": { "type": "number" }, "minItems": 2, "maxItems": 2 },
                    "camera_yaw_degrees": { "type": ["number", "null"] },
                    "camera_pitch_degrees": { "type": ["number", "null"] },
                    "camera_radius_m": { "type": ["number", "null"] },
                    "vertical_fov_degrees": { "type": ["number", "null"] }
                }
            },
            "objects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["id", "label", "aliases", "bbox", "instances", "representative_instance_id", "reuse_group", "instance_count", "object_prompt", "camera_hint", "rotation_hint_degrees", "target_footprint_m"],
                    "properties": {
                        "id": { "type": "string" },
                        "label": { "type": "string" },
                        "aliases": { "type": "array", "items": { "type": "string" } },
                        "bbox": { "type": "array", "items": { "type": "number" }, "minItems": 4, "maxItems": 4 },
                        "instances": {
                            "type": "array",
                            "description": "Per-visible-instance placement evidence for repeated reusable objects. Empty for single objects.",
                            "items": {
                                "type": "object",
                                "additionalProperties": false,
                                "required": ["id", "bbox", "contact", "rotation_hint_degrees", "facing_yaw_degrees", "side", "slot_index", "target_footprint_m"],
                                "properties": {
                                    "id": { "type": ["string", "null"] },
                                    "bbox": { "type": "array", "items": { "type": "number" }, "minItems": 4, "maxItems": 4 },
                                    "contact": { "type": ["array", "null"], "items": { "type": "number" }, "minItems": 2, "maxItems": 2 },
                                    "rotation_hint_degrees": { "type": ["number", "null"] },
                                    "facing_yaw_degrees": { "type": ["number", "null"] },
                                    "side": { "type": ["string", "null"], "enum": ["left", "right", "near", "far", "head", "foot", "unknown", null] },
                                    "slot_index": { "type": ["integer", "null"] },
                                    "target_footprint_m": { "type": ["array", "null"], "items": { "type": "number" }, "minItems": 2, "maxItems": 2 }
                                }
                            }
                        },
                        "representative_instance_id": { "type": ["string", "null"] },
                        "reuse_group": { "type": ["string", "null"] },
                        "instance_count": { "type": "integer" },
                        "object_prompt": { "type": "string" },
                        "camera_hint": { "type": ["string", "null"] },
                        "rotation_hint_degrees": { "type": ["number", "null"] },
                        "target_footprint_m": { "type": ["array", "null"], "items": { "type": "number" }, "minItems": 2, "maxItems": 2 }
                    }
                }
            }
        }
    })
}

pub fn scene_bsn_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["bsn"],
        "properties": {
            "bsn": {
                "type": "string",
                "description": "Restricted synth_scene_v1 scene text. Contains asset declarations, spawn lines, optional environment lines, and camera line."
            }
        }
    })
}

pub fn rotation_selection_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["objects"],
        "properties": {
            "objects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["index", "candidate_index", "confidence", "rationale"],
                    "properties": {
                        "index": { "type": "integer" },
                        "candidate_index": { "type": "integer" },
                        "confidence": { "type": "number" },
                        "rationale": { "type": "string" }
                    }
                }
            }
        }
    })
}

pub fn parse_scene_bsn(bsn: &str, assets: &[SceneAssetBinding]) -> SceneResult<ScenePlan> {
    let known_assets = assets
        .iter()
        .map(|asset| asset.asset_id.as_str())
        .collect::<HashSet<_>>();
    let mut declared_assets = HashSet::new();
    let mut placements = Vec::new();
    let mut camera = None;
    let mut entity_ids = HashSet::new();
    let mut saw_header = false;

    for raw_line in bsn.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with("//") || line == "}" {
            continue;
        }
        if line.starts_with("synth_scene_v1") {
            saw_header = true;
            continue;
        }
        if let Some(asset_id) = parse_asset_line(line)? {
            if !known_assets.contains(asset_id.as_str()) {
                return Err(SceneError::Validation(format!(
                    "BSN declares unknown asset id `{asset_id}`"
                )));
            }
            declared_assets.insert(asset_id);
            continue;
        }
        if line.starts_with("spawn ") {
            let placement = parse_spawn_line(line)?;
            if !declared_assets.contains(&placement.asset_id) {
                return Err(SceneError::Validation(format!(
                    "spawn `{}` references undeclared asset `{}`",
                    placement.entity_id, placement.asset_id
                )));
            }
            if !entity_ids.insert(placement.entity_id.clone()) {
                return Err(SceneError::Validation(format!(
                    "duplicate entity id `{}`",
                    placement.entity_id
                )));
            }
            reject_proxy_furniture(&placement)?;
            placements.push(placement);
            continue;
        }
        if line.starts_with("camera ") {
            camera = Some(parse_camera_line(line)?);
            continue;
        }
        if line.starts_with("environment ") {
            validate_environment_line(line)?;
            continue;
        }
        return Err(SceneError::Parse(format!("unsupported BSN line: {line}")));
    }

    if !saw_header {
        return Err(SceneError::Parse(
            "BSN must start with synth_scene_v1 {".to_string(),
        ));
    }
    if placements.is_empty() {
        return Err(SceneError::Validation(
            "BSN must contain at least one spawn line".to_string(),
        ));
    }
    Ok(ScenePlan {
        bsn: bsn.to_string(),
        placements,
        camera,
    })
}

pub fn scene_plan_to_mcp_commands(
    plan: &ScenePlan,
    assets: &[SceneAssetBinding],
    clear_existing: bool,
) -> SceneResult<Vec<Value>> {
    let asset_map = assets
        .iter()
        .map(|asset| (asset.asset_id.as_str(), asset))
        .collect::<HashMap<_, _>>();
    let mut commands = Vec::new();
    if clear_existing {
        commands.push(json!({ "type": "clear_scene" }));
    }
    for placement in &plan.placements {
        let asset = asset_map.get(placement.asset_id.as_str()).ok_or_else(|| {
            SceneError::Validation(format!(
                "placement references missing asset `{}`",
                placement.asset_id
            ))
        })?;
        let rotation = quat_from_y_degrees(placement.rotation_y_degrees);
        if let Some(cache_key) = asset.cache_key.as_ref() {
            commands.push(json!({
                "type": "spawn_cached",
                "cache_key": cache_key,
                "local_aabb": asset.local_aabb,
                "translation": placement.translation,
                "rotation": rotation,
                "scale": placement.scale,
                "select": false,
            }));
        } else if let Some(path) = asset.path.as_ref() {
            commands.push(json!({
                "type": "spawn_path",
                "path": path,
                "cache_key": asset.asset_id,
                "local_aabb": asset.local_aabb,
                "translation": placement.translation,
                "rotation": rotation,
                "scale": placement.scale,
                "select": false,
            }));
        } else {
            return Err(SceneError::Validation(format!(
                "asset `{}` has neither cache_key nor path",
                asset.asset_id
            )));
        }
    }
    if let Some(camera) = plan.camera.as_ref() {
        commands.push(json!({
            "type": "set_camera",
            "translation": camera.translation,
            "rotation": [0.0, 0.0, 0.0, 1.0],
            "focus": camera.focus,
            "yaw": camera.yaw,
            "pitch": camera.pitch,
            "radius": camera.radius,
            "vertical_fov": camera.vertical_fov_degrees,
        }));
    }
    Ok(commands)
}

pub fn load_scene_asset_bindings(path: &Path) -> SceneResult<Vec<SceneAssetBinding>> {
    serde_json::from_slice(&fs::read(path)?)
        .map_err(|err| SceneError::Parse(format!("asset binding JSON: {err}")))
}

pub fn scene_bsn_to_mcp_command_envelope(
    bsn: &str,
    assets: &[SceneAssetBinding],
    clear_existing: bool,
    session_id: Option<&str>,
    sequence: Option<u64>,
) -> SceneResult<Value> {
    let plan = parse_scene_bsn(bsn, assets)?;
    let commands = scene_plan_to_mcp_commands(&plan, assets, clear_existing)?;
    Ok(json!({
        "session_id": session_id,
        "sequence": sequence,
        "commands": commands,
    }))
}

pub fn scene_bsn_file_to_mcp_command_envelope(
    bsn_path: &Path,
    assets_json_path: &Path,
    clear_existing: bool,
    session_id: Option<&str>,
    sequence: Option<u64>,
) -> SceneResult<Value> {
    let bsn = fs::read_to_string(bsn_path)?;
    let assets = load_scene_asset_bindings(assets_json_path)?;
    scene_bsn_to_mcp_command_envelope(&bsn, &assets, clear_existing, session_id, sequence)
}

pub fn write_metric(output_dir: &Path, stage: &str, value: Value) -> SceneResult<()> {
    let _guard = METRIC_WRITE_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .map_err(|_| SceneError::Io("metric write lock poisoned".into()))?;
    fs::create_dir_all(output_dir)?;
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(output_dir.join("metrics.jsonl"))?;
    let event = json!({
        "timestamp_unix_ms": unix_ms(),
        "stage": stage,
        "value": value,
    });
    writeln!(file, "{event}")?;
    Ok(())
}

pub fn write_json_file<T: Serialize + ?Sized>(path: &Path, value: &T) -> SceneResult<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let bytes = serde_json::to_vec_pretty(value)
        .map_err(|err| SceneError::Io(format!("serialize {}: {err}", path.display())))?;
    fs::write(path, bytes)?;
    Ok(())
}

pub fn crop_scene_object(
    source_scene_path: &Path,
    object: &SceneObjectSpec,
    output_path: &Path,
) -> SceneResult<PathBuf> {
    let image = image::open(source_scene_path)?;
    let (width, height) = image.dimensions();
    let bbox = normalize_bbox(object.bbox);
    let pad_x = ((bbox[2] - bbox[0]) * 0.10).max(0.02);
    let pad_y = ((bbox[3] - bbox[1]) * 0.10).max(0.02);
    let x0 = ((bbox[0] - pad_x).clamp(0.0, 1.0) * width as f32).floor() as u32;
    let y0 = ((bbox[1] - pad_y).clamp(0.0, 1.0) * height as f32).floor() as u32;
    let x1 = ((bbox[2] + pad_x).clamp(0.0, 1.0) * width as f32).ceil() as u32;
    let y1 = ((bbox[3] + pad_y).clamp(0.0, 1.0) * height as f32).ceil() as u32;
    let crop_width = x1.saturating_sub(x0).max(1);
    let crop_height = y1.saturating_sub(y0).max(1);
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let crop = image.crop_imm(x0, y0, crop_width, crop_height);
    write_resized_jpeg(&crop, output_path, 1024, 90)?;
    Ok(output_path.to_path_buf())
}

pub(crate) fn representative_crop_bbox(object: &SceneObjectSpec) -> [f32; 4] {
    let Some(instance) = representative_instance(object) else {
        return normalize_bbox(object.bbox);
    };
    normalize_bbox(instance.bbox)
}

fn representative_instance(object: &SceneObjectSpec) -> Option<&SceneObjectInstanceSpec> {
    if object.instances.is_empty() {
        return None;
    }
    if let Some(id) = object
        .representative_instance_id
        .as_deref()
        .filter(|id| !id.trim().is_empty())
        && let Some(instance) = object
            .instances
            .iter()
            .find(|instance| instance.id.as_deref() == Some(id))
    {
        return Some(instance);
    }
    object.instances.iter().max_by(|left, right| {
        bbox_area(normalize_bbox(left.bbox))
            .partial_cmp(&bbox_area(normalize_bbox(right.bbox)))
            .unwrap_or(std::cmp::Ordering::Equal)
    })
}

fn bbox_area(bbox: [f32; 4]) -> f32 {
    (bbox[2] - bbox[0]).max(0.0) * (bbox[3] - bbox[1]).max(0.0)
}

pub fn resize_image_for_api(input_path: &Path, output_path: &Path) -> SceneResult<PathBuf> {
    let image = image::open(input_path)?;
    write_resized_jpeg(&image, output_path, 1024, 90)?;
    Ok(output_path.to_path_buf())
}

fn write_resized_jpeg(
    image: &image::DynamicImage,
    output_path: &Path,
    max_edge: u32,
    quality: u8,
) -> SceneResult<()> {
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let resized = if image.width().max(image.height()) > max_edge {
        image.resize(max_edge, max_edge, FilterType::Lanczos3)
    } else {
        image.clone()
    };
    let rgb = resized.to_rgb8();
    let mut file = fs::File::create(output_path)?;
    let mut encoder = JpegEncoder::new_with_quality(&mut file, quality);
    encoder
        .encode_image(&rgb)
        .map_err(|err| SceneError::Image(err.to_string()))
}

pub(crate) fn validate_build_config(config: &SceneBuildConfig) -> SceneResult<()> {
    if !config.source_scene_path.exists() {
        return Err(SceneError::Config(format!(
            "source scene image does not exist: {}",
            config.source_scene_path.display()
        )));
    }
    if !config.object_reference_image_path.exists() {
        return Err(SceneError::Config(format!(
            "object reference image does not exist: {}",
            config.object_reference_image_path.display()
        )));
    }
    if config.candidate_count == 0 {
        return Err(SceneError::Config(
            "candidate_count must be greater than zero".to_string(),
        ));
    }
    Ok(())
}

fn parse_asset_line(line: &str) -> SceneResult<Option<String>> {
    if !line.starts_with("asset ") {
        return Ok(None);
    }
    let without_prefix = line
        .strip_prefix("asset ")
        .unwrap()
        .trim_end_matches(';')
        .trim();
    let (asset_id, _) = without_prefix
        .split_once('=')
        .ok_or_else(|| SceneError::Parse(format!("invalid asset line: {line}")))?;
    let asset_id = asset_id.trim();
    if asset_id.is_empty() || asset_id.contains(char::is_whitespace) {
        return Err(SceneError::Parse(format!(
            "invalid asset id in line: {line}"
        )));
    }
    Ok(Some(asset_id.to_string()))
}

fn parse_spawn_line(line: &str) -> SceneResult<ScenePlacement> {
    let tokens = split_bsn_tokens(line.trim_end_matches(';'));
    if tokens.len() < 10 {
        return Err(SceneError::Parse(format!("invalid spawn line: {line}")));
    }
    if tokens.first().map(String::as_str) != Some("spawn") {
        return Err(SceneError::Parse(format!("invalid spawn line: {line}")));
    }
    let entity_id = tokens[1].clone();
    expect_token(&tokens, 2, "uses", line)?;
    let asset_id = tokens[3].clone();
    expect_token(&tokens, 4, "translation", line)?;
    let translation = parse_vec3_token(&tokens[5], line)?;
    expect_token(&tokens, 6, "rotation_y", line)?;
    let rotation_y_degrees = tokens[7]
        .parse::<f32>()
        .map_err(|_| SceneError::Parse(format!("invalid rotation_y in line: {line}")))?;
    expect_token(&tokens, 8, "scale", line)?;
    let scale = parse_vec3_token(&tokens[9], line)?;
    for value in translation
        .into_iter()
        .chain([rotation_y_degrees])
        .chain(scale)
    {
        if !value.is_finite() {
            return Err(SceneError::Validation(format!(
                "non-finite transform in line: {line}"
            )));
        }
    }
    Ok(ScenePlacement {
        entity_id,
        asset_id,
        translation,
        rotation_y_degrees,
        scale,
    })
}

fn parse_camera_line(line: &str) -> SceneResult<SceneCamera> {
    let tokens = split_bsn_tokens(line.trim_end_matches(';'));
    if tokens.len() < 5 {
        return Err(SceneError::Parse(format!("invalid camera line: {line}")));
    }
    expect_token(&tokens, 1, "translation", line)?;
    let translation = parse_vec3_token(&tokens[2], line)?;
    expect_token(&tokens, 3, "focus", line)?;
    let focus = parse_vec3_token(&tokens[4], line)?;
    let mut yaw = None;
    let mut pitch = None;
    let mut radius = None;
    let mut vertical_fov_degrees = None;
    let mut index = 5;
    while index + 1 < tokens.len() {
        match tokens[index].as_str() {
            "yaw" => yaw = Some(parse_f32_token(&tokens[index + 1], line)?),
            "pitch" => pitch = Some(parse_f32_token(&tokens[index + 1], line)?),
            "radius" => radius = Some(parse_f32_token(&tokens[index + 1], line)?),
            "vertical_fov" => {
                vertical_fov_degrees = Some(parse_f32_token(&tokens[index + 1], line)?)
            }
            other => return Err(SceneError::Parse(format!("unknown camera key `{other}`"))),
        }
        index += 2;
    }
    Ok(SceneCamera {
        translation,
        focus,
        yaw,
        pitch,
        radius,
        vertical_fov_degrees,
    })
}

fn validate_environment_line(line: &str) -> SceneResult<()> {
    let tokens = split_bsn_tokens(line.trim_end_matches(';'));
    let kind = tokens.get(1).map(String::as_str).unwrap_or_default();
    match kind {
        "rug" | "floor" | "wall" | "reference_plane" => Ok(()),
        _ => Err(SceneError::Validation(format!(
            "unsupported environment primitive in line: {line}"
        ))),
    }
}

fn reject_proxy_furniture(placement: &ScenePlacement) -> SceneResult<()> {
    let id = placement.entity_id.to_ascii_lowercase();
    if id.contains("cube") || id.contains("proxy") || id.contains("debug") {
        return Err(SceneError::Validation(format!(
            "furniture placement `{}` looks like a proxy/debug asset",
            placement.entity_id
        )));
    }
    Ok(())
}

fn split_bsn_tokens(line: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut bracket_depth = 0usize;
    for ch in line.chars() {
        match ch {
            '[' => {
                bracket_depth += 1;
                current.push(ch);
            }
            ']' => {
                bracket_depth = bracket_depth.saturating_sub(1);
                current.push(ch);
            }
            ' ' | '\t' if bracket_depth == 0 => {
                if !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
            }
            _ => current.push(ch),
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

fn parse_vec3_token(token: &str, line: &str) -> SceneResult<[f32; 3]> {
    let body = token
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
        .ok_or_else(|| {
            SceneError::Parse(format!("invalid vec3 token `{token}` in line: {line}"))
        })?;
    let parts = body
        .split(',')
        .map(|part| parse_f32_token(part.trim(), line))
        .collect::<SceneResult<Vec<_>>>()?;
    if parts.len() != 3 {
        return Err(SceneError::Parse(format!(
            "vec3 token `{token}` must have three values"
        )));
    }
    Ok([parts[0], parts[1], parts[2]])
}

fn parse_f32_token(token: &str, line: &str) -> SceneResult<f32> {
    let value = token
        .parse::<f32>()
        .map_err(|_| SceneError::Parse(format!("invalid number `{token}` in line: {line}")))?;
    if !value.is_finite() {
        return Err(SceneError::Validation(format!(
            "non-finite number `{token}` in line: {line}"
        )));
    }
    Ok(value)
}

fn expect_token(tokens: &[String], index: usize, expected: &str, line: &str) -> SceneResult<()> {
    if tokens.get(index).map(String::as_str) == Some(expected) {
        Ok(())
    } else {
        Err(SceneError::Parse(format!(
            "expected token `{expected}` at position {index} in line: {line}"
        )))
    }
}

fn quat_from_y_degrees(degrees: f32) -> [f32; 4] {
    let half = degrees.to_radians() * 0.5;
    [0.0, half.sin(), 0.0, half.cos()]
}

pub(crate) fn normalize_bbox(mut bbox: [f32; 4]) -> [f32; 4] {
    bbox[0] = bbox[0].clamp(0.0, 1.0);
    bbox[1] = bbox[1].clamp(0.0, 1.0);
    bbox[2] = bbox[2].clamp(0.0, 1.0);
    bbox[3] = bbox[3].clamp(0.0, 1.0);
    if bbox[0] > bbox[2] {
        bbox.swap(0, 2);
    }
    if bbox[1] > bbox[3] {
        bbox.swap(1, 3);
    }
    bbox
}

pub(crate) fn extract_structured_output(value: Value) -> SceneResult<Value> {
    if let Some(parsed) = value.get("output_parsed") {
        return Ok(parsed.clone());
    }
    if let Some(text) = value.get("output_text").and_then(Value::as_str) {
        return serde_json::from_str(text)
            .map_err(|err| SceneError::Provider(format!("parse output_text JSON: {err}")));
    }
    let output = value
        .get("output")
        .and_then(Value::as_array)
        .ok_or_else(|| SceneError::Provider("responses output missing".to_string()))?;
    for item in output {
        if let Some(content) = item.get("content").and_then(Value::as_array) {
            for content_item in content {
                if let Some(text) = content_item.get("text").and_then(Value::as_str)
                    && let Ok(parsed) = serde_json::from_str::<Value>(text)
                {
                    return Ok(parsed);
                }
            }
        }
    }
    Err(SceneError::Provider(
        "could not locate structured output JSON".to_string(),
    ))
}

pub(crate) fn redact_openai_value(value: &Value) -> String {
    let mut value = value.clone();
    if let Some(object) = value.as_object_mut() {
        object.remove("prompt");
        object.remove("b64_json");
    }
    value.to_string()
}

pub(crate) fn image_data_url(path: &Path) -> SceneResult<String> {
    let mime = image_mime_type(path);
    let bytes = fs::read(path)?;
    Ok(format!(
        "data:{mime};base64,{}",
        BASE64_STANDARD.encode(bytes)
    ))
}

pub(crate) fn image_mime_type(path: &Path) -> &'static str {
    match path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        _ => "image/png",
    }
}

pub(crate) fn default_run_id(label: &str) -> String {
    format!("{}_{}", unix_compact(), label)
}

fn unix_compact() -> String {
    unix_ms().to_string()
}

pub(crate) fn unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

pub(crate) fn stable_hash_hex(value: &str) -> String {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

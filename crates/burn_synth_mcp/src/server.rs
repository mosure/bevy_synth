use crate::prelude::*;
use image::GenericImageView;
use std::cell::RefCell;
use std::rc::Rc;

const FEEDBACK_RENDER_SETTLE_MS: u64 = 750;
const FEEDBACK_ROTATION_RENDER_MAX_CANDIDATES: usize = 18;
const FINAL_YAW_RENDER_MIN_FOREGROUND_RATIO: f32 = 0.04;
const FINAL_YAW_TARGET_ONLY_RENDER_MIN_FOREGROUND_RATIO: f32 = 0.008;
const FINAL_YAW_RENDER_SAMPLE_STRIDE: u32 = 4;
const FINAL_YAW_VISUAL_SCORE_IMPROVEMENT: f64 = 0.045;
const FINAL_YAW_EDGE_SEATING_FINE_PARENT_SCORE_IMPROVEMENT: f64 = 0.04;
const FINAL_YAW_EDGE_SEATING_CONFIDENT_SELECTION_MAX_VISUAL_REGRESSION: f64 = 0.08;
const FINAL_YAW_FINE_DELTAS_DEGREES: [f32; 8] = [-20.0, -15.0, -10.0, -5.0, 5.0, 10.0, 15.0, 20.0];
const FINAL_YAW_MASK_DESCRIPTOR_BINS: usize = 8;
const FINAL_YAW_MASK_DESCRIPTOR_MIN_PIXELS: usize = 16;
const FINAL_YAW_CONTACT_SHEET_TILE_WIDTH: u32 = 360;
const FINAL_YAW_CONTACT_SHEET_TILE_HEIGHT: u32 = 203;
const FINAL_YAW_CONTACT_SHEET_BORDER: u32 = 8;
const FINAL_YAW_CONTACT_SHEET_GAP: u32 = 14;
const FINAL_YAW_CONTACT_SHEET_COLUMNS: usize = 3;
const SCENE_CAPTURE_WINDOW_LONG_EDGE: u32 = 1600;
const SCENE_CAPTURE_WINDOW_MIN_SHORT_EDGE: u32 = 360;

fn scene_capture_window_size_for_source(source_scene_path: &str) -> Option<(u32, u32)> {
    let (width, height) = image::image_dimensions(source_scene_path).ok()?;
    if width == 0 || height == 0 {
        return None;
    }
    let aspect = width as f32 / height as f32;
    if aspect >= 1.0 {
        let target_width = SCENE_CAPTURE_WINDOW_LONG_EDGE;
        let target_height = ((target_width as f32 / aspect).round() as u32)
            .max(SCENE_CAPTURE_WINDOW_MIN_SHORT_EDGE);
        Some((target_width, target_height))
    } else {
        let target_height = SCENE_CAPTURE_WINDOW_LONG_EDGE;
        let target_width = ((target_height as f32 * aspect).round() as u32)
            .max(SCENE_CAPTURE_WINDOW_MIN_SHORT_EDGE);
        Some((target_width, target_height))
    }
}

pub fn run_stdio_server(config: ServerConfig) -> Result<(), String> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut reader = BufReader::new(stdin.lock());
    let mut writer = BufWriter::new(stdout.lock());
    let mut server = McpServer::new(config);

    while let Some(message) = read_framed_json(&mut reader).map_err(|err| err.to_string())? {
        let response = server.handle_message(message)?;
        if let Some(response) = response {
            write_framed_json(&mut writer, &response).map_err(|err| err.to_string())?;
        }
        if server.should_exit {
            break;
        }
    }

    Ok(())
}

pub(crate) struct McpServer {
    pub(crate) config: ServerConfig,
    pub(crate) runtime: SynthRuntime,
    pub(crate) grounding: SceneGroundingRuntime,
    pub(crate) should_exit: bool,
}

struct NoopSceneProvider;

struct SceneAssetFrameCalibrationRequest<'a> {
    output_dir: &'a Path,
    write_artifacts: bool,
    mode: SceneCanonicalPoseMode,
    max_candidates: usize,
    manifest: &'a SceneObjectManifest,
    asset_bindings: &'a [SceneAssetBinding],
    selected_candidates: &'a [Value],
    object_image_requests: &'a [ObjectImageRequest],
    evidence: &'a SceneGroundingEvidence,
}

#[derive(Clone, Copy)]
enum CanonicalPoseThumbnailSpawnBasis<'a> {
    Cache(&'a str),
    Path(&'a str),
}

impl<'a> CanonicalPoseThumbnailSpawnBasis<'a> {
    fn from_asset(asset: &'a SceneAssetBinding) -> Option<Self> {
        asset
            .cache_key
            .as_deref()
            .map(Self::Cache)
            .or_else(|| asset.path.as_deref().map(Self::Path))
    }

    fn label(self) -> &'static str {
        match self {
            Self::Cache(_) => "cache",
            Self::Path(_) => "path",
        }
    }
}

fn feedback_selector_uses_rendered_candidates(selector: FeedbackRotationSelector) -> bool {
    matches!(
        selector,
        FeedbackRotationSelector::RenderedSweep | FeedbackRotationSelector::Openai
    )
}

fn feedback_rotation_spawn_commands(commands: &[Value]) -> Vec<(usize, Value)> {
    commands
        .iter()
        .enumerate()
        .filter_map(|(command_index, command)| {
            let command_type = command.get("type").and_then(Value::as_str)?;
            matches!(command_type, "spawn_cached" | "spawn_path")
                .then(|| (command_index, command.clone()))
        })
        .collect()
}

fn feedback_rotation_camera_command(commands: &[Value]) -> Option<Value> {
    commands
        .iter()
        .rev()
        .find(|command| command.get("type").and_then(Value::as_str) == Some("set_camera"))
        .cloned()
}

fn feedback_rotation_candidate_commands(
    base_spawn: &Value,
    candidate_yaw_degrees: f32,
    camera_command: &Value,
) -> Vec<Value> {
    let mut spawn = base_spawn.clone();
    spawn["rotation"] = json!(quat_from_y_degrees(candidate_yaw_degrees));
    spawn["select"] = json!(false);
    vec![
        json!({ "type": "clear_scene" }),
        json!({ "type": "reload_cache" }),
        spawn,
        camera_command.clone(),
    ]
}

fn final_yaw_full_scene_candidate_commands(
    commands: &[Value],
    target_command_index: usize,
    candidate: &Value,
) -> Vec<Value> {
    let candidate_yaw_degrees = candidate
        .get("candidate_yaw_degrees")
        .and_then(Value::as_f64)
        .map(|value| value as f32)
        .unwrap_or(0.0);
    let candidate_pose = final_yaw_candidate_pose(candidate);
    let mut out = vec![
        json!({ "type": "clear_scene" }),
        json!({ "type": "reload_cache" }),
    ];
    let mut camera_commands = Vec::new();
    for (command_index, command) in commands.iter().enumerate() {
        let command_type = command.get("type").and_then(Value::as_str).unwrap_or("");
        match command_type {
            "spawn_cached" | "spawn_path" => {
                let mut spawn = command.clone();
                spawn["select"] = json!(false);
                if command_index == target_command_index {
                    if let Some((translation, scale)) = candidate_pose {
                        spawn["translation"] = json!(translation);
                        spawn["scale"] = json!(scale);
                    }
                    spawn["rotation"] = json!(quat_from_y_degrees(candidate_yaw_degrees));
                }
                out.push(spawn);
            }
            "set_camera" => camera_commands.push(command.clone()),
            _ => {}
        }
    }
    out.extend(camera_commands);
    out
}

fn final_yaw_target_only_candidate_commands(
    commands: &[Value],
    target_command_index: usize,
    candidate: &Value,
) -> Option<Vec<Value>> {
    let candidate_yaw_degrees = candidate
        .get("candidate_yaw_degrees")
        .and_then(Value::as_f64)
        .map(|value| value as f32)
        .unwrap_or(0.0);
    let candidate_pose = final_yaw_candidate_pose(candidate);
    let mut out = vec![
        json!({ "type": "clear_scene" }),
        json!({ "type": "reload_cache" }),
    ];
    let mut camera_commands = Vec::new();
    let mut target_spawn = None;
    for (command_index, command) in commands.iter().enumerate() {
        let command_type = command.get("type").and_then(Value::as_str).unwrap_or("");
        match command_type {
            "spawn_cached" | "spawn_path" if command_index == target_command_index => {
                let mut spawn = command.clone();
                spawn["select"] = json!(false);
                if let Some((translation, scale)) = candidate_pose {
                    spawn["translation"] = json!(translation);
                    spawn["scale"] = json!(scale);
                }
                spawn["rotation"] = json!(quat_from_y_degrees(candidate_yaw_degrees));
                target_spawn = Some(spawn);
            }
            "set_camera" => camera_commands.push(command.clone()),
            _ => {}
        }
    }
    out.push(target_spawn?);
    out.extend(camera_commands);
    Some(out)
}

fn final_yaw_candidate_pose(candidate: &Value) -> Option<([f32; 3], [f32; 3])> {
    let metrics = candidate
        .get("geometry_metrics")
        .filter(|value| value.is_object())
        .unwrap_or(candidate);
    let translation = metrics.get("translation").and_then(final_yaw_json_array3)?;
    let scale = metrics.get("scale").and_then(final_yaw_json_array3)?;
    Some((translation, scale))
}

fn final_yaw_render_foreground_report(path: &Path, min_foreground_ratio: f32) -> Value {
    let image = match image::open(path) {
        Ok(image) => image.to_rgba8(),
        Err(err) => {
            return json!({
                "ok": false,
                "min_foreground_ratio": min_foreground_ratio,
                "error": err.to_string(),
            });
        }
    };
    let stride = FINAL_YAW_RENDER_SAMPLE_STRIDE.max(1) as usize;
    let mut samples = 0usize;
    let mut foreground = 0usize;
    for y in (0..image.height()).step_by(stride) {
        for x in (0..image.width()).step_by(stride) {
            let pixel = image.get_pixel(x, y).0;
            let r = pixel[0] as f32 / 255.0;
            let g = pixel[1] as f32 / 255.0;
            let b = pixel[2] as f32 / 255.0;
            let max_channel = r.max(g).max(b);
            let min_channel = r.min(g).min(b);
            let saturation = if max_channel > 1.0e-5 {
                (max_channel - min_channel) / max_channel
            } else {
                0.0
            };
            let luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            if (saturation > 0.16 && luminance > 0.12) || luminance > 0.56 {
                foreground = foreground.saturating_add(1);
            }
            samples = samples.saturating_add(1);
        }
    }
    let foreground_ratio = if samples > 0 {
        foreground as f32 / samples as f32
    } else {
        0.0
    };
    json!({
        "ok": foreground_ratio >= min_foreground_ratio,
        "foreground_ratio": foreground_ratio,
        "foreground_samples": foreground,
        "sample_count": samples,
        "min_foreground_ratio": min_foreground_ratio,
        "sample_stride": FINAL_YAW_RENDER_SAMPLE_STRIDE,
    })
}

fn final_yaw_render_path_has_foreground(path: &str) -> bool {
    final_yaw_render_foreground_report(Path::new(path), FINAL_YAW_RENDER_MIN_FOREGROUND_RATIO)
        .get("ok")
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

fn final_yaw_spawn_ordinal(commands: &[Value], target_command_index: usize) -> Option<usize> {
    let mut ordinal = 0usize;
    for (command_index, command) in commands.iter().enumerate() {
        let command_type = command.get("type").and_then(Value::as_str).unwrap_or("");
        if !matches!(command_type, "spawn_cached" | "spawn_path") {
            continue;
        }
        if command_index == target_command_index {
            return Some(ordinal);
        }
        ordinal = ordinal.saturating_add(1);
    }
    None
}

fn final_yaw_target_projected_bbox(
    status: &Value,
    commands: &[Value],
    target_command_index: usize,
) -> Option<[f32; 4]> {
    let ordinal = final_yaw_spawn_ordinal(commands, target_command_index)?;
    status
        .get("projected_items")
        .and_then(Value::as_array)?
        .get(ordinal)
        .and_then(|item| item.get("screen_bbox"))
        .and_then(json_array4)
        .and_then(feedback_visible_bbox)
}

fn final_yaw_bbox_area(bbox: [f32; 4]) -> f32 {
    (bbox[2] - bbox[0]).max(0.0) * (bbox[3] - bbox[1]).max(0.0)
}

fn final_yaw_bbox_iou(left: [f32; 4], right: [f32; 4]) -> f32 {
    let intersection = final_yaw_bbox_area([
        left[0].max(right[0]),
        left[1].max(right[1]),
        left[2].min(right[2]),
        left[3].min(right[3]),
    ]);
    let union = final_yaw_bbox_area(left) + final_yaw_bbox_area(right) - intersection;
    if union > 1.0e-6 {
        (intersection / union).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn final_yaw_bbox_center_error(left: [f32; 4], right: [f32; 4]) -> f32 {
    let left_center = [(left[0] + left[2]) * 0.5, (left[1] + left[3]) * 0.5];
    let right_center = [(right[0] + right[2]) * 0.5, (right[1] + right[3]) * 0.5];
    let dx = left_center[0] - right_center[0];
    let dy = left_center[1] - right_center[1];
    (dx * dx + dy * dy).sqrt()
}

fn final_yaw_bbox_area_log2_error(left: [f32; 4], right: [f32; 4]) -> f32 {
    let left_area = final_yaw_bbox_area(left).max(1.0e-6);
    let right_area = final_yaw_bbox_area(right).max(1.0e-6);
    (left_area / right_area).log2().abs()
}

#[derive(Clone)]
struct FinalYawMaskDescriptor {
    bbox: [f32; 4],
    centroid: [f32; 2],
    area_ratio: f32,
    full_histogram: Vec<f64>,
    local_histogram: Vec<f64>,
    local_row_profile: Vec<f64>,
    local_column_profile: Vec<f64>,
    foreground_samples: usize,
    sample_count: usize,
}

fn final_yaw_mask_descriptor_from_image<F>(
    image: &image::DynamicImage,
    stride: u32,
    is_foreground: F,
) -> Option<FinalYawMaskDescriptor>
where
    F: Fn(image::Rgba<u8>) -> bool,
{
    let width = image.width().max(1);
    let height = image.height().max(1);
    let stride = stride.max(1) as usize;
    let width_denom = width.saturating_sub(1).max(1) as f32;
    let height_denom = height.saturating_sub(1).max(1) as f32;
    let mut points = Vec::new();
    let mut sample_count = 0usize;
    for y in (0..height).step_by(stride) {
        for x in (0..width).step_by(stride) {
            sample_count = sample_count.saturating_add(1);
            if is_foreground(image.get_pixel(x, y)) {
                points.push([x as f32 / width_denom, y as f32 / height_denom]);
            }
        }
    }
    if points.len() < FINAL_YAW_MASK_DESCRIPTOR_MIN_PIXELS {
        return None;
    }

    let mut min_x = 1.0f32;
    let mut min_y = 1.0f32;
    let mut max_x = 0.0f32;
    let mut max_y = 0.0f32;
    let mut sum_x = 0.0f32;
    let mut sum_y = 0.0f32;
    for [x, y] in &points {
        min_x = min_x.min(*x);
        min_y = min_y.min(*y);
        max_x = max_x.max(*x);
        max_y = max_y.max(*y);
        sum_x += *x;
        sum_y += *y;
    }
    let foreground_samples = points.len();
    let centroid = [
        sum_x / foreground_samples as f32,
        sum_y / foreground_samples as f32,
    ];
    let bbox = [min_x, min_y, max_x, max_y];
    let mut full_histogram =
        vec![0.0f64; FINAL_YAW_MASK_DESCRIPTOR_BINS * FINAL_YAW_MASK_DESCRIPTOR_BINS];
    let mut local_histogram =
        vec![0.0f64; FINAL_YAW_MASK_DESCRIPTOR_BINS * FINAL_YAW_MASK_DESCRIPTOR_BINS];
    let bbox_width = (bbox[2] - bbox[0]).max(1.0e-6);
    let bbox_height = (bbox[3] - bbox[1]).max(1.0e-6);
    for [x, y] in &points {
        let full_x = ((*x * FINAL_YAW_MASK_DESCRIPTOR_BINS as f32).floor() as usize)
            .min(FINAL_YAW_MASK_DESCRIPTOR_BINS - 1);
        let full_y = ((*y * FINAL_YAW_MASK_DESCRIPTOR_BINS as f32).floor() as usize)
            .min(FINAL_YAW_MASK_DESCRIPTOR_BINS - 1);
        full_histogram[full_y * FINAL_YAW_MASK_DESCRIPTOR_BINS + full_x] += 1.0;

        let local_x_norm = ((*x - bbox[0]) / bbox_width).clamp(0.0, 0.999_999);
        let local_y_norm = ((*y - bbox[1]) / bbox_height).clamp(0.0, 0.999_999);
        let local_x = ((local_x_norm * FINAL_YAW_MASK_DESCRIPTOR_BINS as f32).floor() as usize)
            .min(FINAL_YAW_MASK_DESCRIPTOR_BINS - 1);
        let local_y = ((local_y_norm * FINAL_YAW_MASK_DESCRIPTOR_BINS as f32).floor() as usize)
            .min(FINAL_YAW_MASK_DESCRIPTOR_BINS - 1);
        local_histogram[local_y * FINAL_YAW_MASK_DESCRIPTOR_BINS + local_x] += 1.0;
    }
    let inv_count = 1.0 / foreground_samples as f64;
    for value in &mut full_histogram {
        *value *= inv_count;
    }
    for value in &mut local_histogram {
        *value *= inv_count;
    }

    let mut local_row_profile = vec![0.0f64; FINAL_YAW_MASK_DESCRIPTOR_BINS];
    let mut local_column_profile = vec![0.0f64; FINAL_YAW_MASK_DESCRIPTOR_BINS];
    for row in 0..FINAL_YAW_MASK_DESCRIPTOR_BINS {
        for column in 0..FINAL_YAW_MASK_DESCRIPTOR_BINS {
            let value = local_histogram[row * FINAL_YAW_MASK_DESCRIPTOR_BINS + column];
            local_row_profile[row] += value;
            local_column_profile[column] += value;
        }
    }

    Some(FinalYawMaskDescriptor {
        bbox,
        centroid,
        area_ratio: foreground_samples as f32 / sample_count.max(1) as f32,
        full_histogram,
        local_histogram,
        local_row_profile,
        local_column_profile,
        foreground_samples,
        sample_count,
    })
}

fn final_yaw_source_mask_descriptor(path: &Path) -> Result<FinalYawMaskDescriptor, String> {
    let image = image::open(path).map_err(|err| {
        format!(
            "failed to open final-yaw source mask {}: {err}",
            path.display()
        )
    })?;
    final_yaw_mask_descriptor_from_image(&image, FINAL_YAW_RENDER_SAMPLE_STRIDE, |pixel| {
        pixel.0[3] > 127
    })
    .ok_or_else(|| {
        format!(
            "final-yaw source mask {} has too few alpha foreground samples",
            path.display()
        )
    })
}

fn final_yaw_target_render_mask_descriptor(path: &Path) -> Result<FinalYawMaskDescriptor, String> {
    let image = image::open(path).map_err(|err| {
        format!(
            "failed to open final-yaw target-only render {}: {err}",
            path.display()
        )
    })?;
    final_yaw_mask_descriptor_from_image(&image, FINAL_YAW_RENDER_SAMPLE_STRIDE, |pixel| {
        let r = pixel.0[0] as f32 / 255.0;
        let g = pixel.0[1] as f32 / 255.0;
        let b = pixel.0[2] as f32 / 255.0;
        let max_channel = r.max(g).max(b);
        let min_channel = r.min(g).min(b);
        let saturation = if max_channel > 1.0e-5 {
            (max_channel - min_channel) / max_channel
        } else {
            0.0
        };
        let luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        (saturation > 0.16 && luminance > 0.12) || luminance > 0.56
    })
    .ok_or_else(|| {
        format!(
            "final-yaw target-only render {} has too few foreground samples",
            path.display()
        )
    })
}

fn final_yaw_histogram_similarity(left: &[f64], right: &[f64]) -> f64 {
    if left.len() != right.len() || left.is_empty() {
        return 0.0;
    }
    let l1 = left
        .iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .sum::<f64>();
    (1.0 - 0.5 * l1).clamp(0.0, 1.0)
}

fn final_yaw_mask_descriptor_fit_report(
    source: &FinalYawMaskDescriptor,
    target: &FinalYawMaskDescriptor,
) -> Value {
    let bbox_iou = final_yaw_bbox_iou(source.bbox, target.bbox) as f64;
    let center_error = final_yaw_bbox_center_error(source.bbox, target.bbox) as f64;
    let centroid_error = {
        let dx = source.centroid[0] - target.centroid[0];
        let dy = source.centroid[1] - target.centroid[1];
        (dx * dx + dy * dy).sqrt() as f64
    };
    let center_component = (1.0 - centroid_error * 3.0).clamp(0.0, 1.0);
    let area_log2_error = (target.area_ratio.max(1.0e-6) / source.area_ratio.max(1.0e-6))
        .log2()
        .abs() as f64;
    let area_component = (-area_log2_error * 0.75).exp().clamp(0.0, 1.0);
    let full_histogram_similarity =
        final_yaw_histogram_similarity(&source.full_histogram, &target.full_histogram);
    let local_histogram_similarity =
        final_yaw_histogram_similarity(&source.local_histogram, &target.local_histogram);
    let row_profile_similarity =
        final_yaw_histogram_similarity(&source.local_row_profile, &target.local_row_profile);
    let column_profile_similarity =
        final_yaw_histogram_similarity(&source.local_column_profile, &target.local_column_profile);
    let score = (0.24 * full_histogram_similarity)
        + (0.28 * local_histogram_similarity)
        + (0.14 * bbox_iou)
        + (0.14 * center_component)
        + (0.10 * area_component)
        + (0.05 * row_profile_similarity)
        + (0.05 * column_profile_similarity);
    json!({
        "score": score.clamp(0.0, 1.0),
        "source_bbox": source.bbox,
        "target_bbox": target.bbox,
        "source_centroid": source.centroid,
        "target_centroid": target.centroid,
        "source_area_ratio": source.area_ratio,
        "target_area_ratio": target.area_ratio,
        "source_foreground_samples": source.foreground_samples,
        "target_foreground_samples": target.foreground_samples,
        "source_sample_count": source.sample_count,
        "target_sample_count": target.sample_count,
        "bbox_iou": bbox_iou,
        "bbox_center_error": center_error,
        "centroid_error": centroid_error,
        "center_component": center_component,
        "area_log2_error": area_log2_error,
        "area_component": area_component,
        "full_histogram_similarity": full_histogram_similarity,
        "local_histogram_similarity": local_histogram_similarity,
        "row_profile_similarity": row_profile_similarity,
        "column_profile_similarity": column_profile_similarity,
        "score_policy": "0.24*full_hist + 0.28*local_hist + 0.14*bbox_iou + 0.14*centroid + 0.10*area + 0.05*rows + 0.05*cols",
    })
}

fn final_yaw_render_mask_fit_report(object: &Value, target_only_path: &Path) -> Value {
    let Some(source_mask_path) = object.get("source_mask_path").and_then(Value::as_str) else {
        return json!({
            "enabled": false,
            "reason": "missing_source_mask_path",
        });
    };
    let source_mask_path = Path::new(source_mask_path);
    match (
        final_yaw_source_mask_descriptor(source_mask_path),
        final_yaw_target_render_mask_descriptor(target_only_path),
    ) {
        (Ok(source), Ok(target)) => {
            let mut report = final_yaw_mask_descriptor_fit_report(&source, &target);
            report["enabled"] = json!(true);
            report["source_mask_path"] = json!(source_mask_path.display().to_string());
            report["target_only_render_path"] = json!(target_only_path.display().to_string());
            report["descriptor"] = json!("alpha_source_vs_target_only_foreground_8x8_histogram");
            report
        }
        (source_result, target_result) => json!({
            "enabled": false,
            "source_mask_path": source_mask_path.display().to_string(),
            "target_only_render_path": target_only_path.display().to_string(),
            "source_error": source_result.err(),
            "target_error": target_result.err(),
        }),
    }
}

fn final_yaw_render_visual_fit_report(
    object: &Value,
    screenshot_path: &Path,
    target_bbox: [f32; 4],
    crop_path: &Path,
    foreground: &Value,
    target_only_screenshot_path: Option<&Path>,
) -> Value {
    let source_bbox = object
        .get("source_bbox")
        .and_then(final_yaw_json_array4)
        .unwrap_or(target_bbox);
    let bbox_iou = final_yaw_bbox_iou(target_bbox, source_bbox);
    let center_error = final_yaw_bbox_center_error(target_bbox, source_bbox);
    let area_log2_error = final_yaw_bbox_area_log2_error(target_bbox, source_bbox);
    let mut crop_error = Value::Null;
    let mut visual_similarity = Value::Null;
    let mut visual_score = None;
    match image::open(screenshot_path) {
        Ok(image) => match write_feedback_crop(&image, target_bbox, crop_path) {
            Ok(()) => {
                if let Some(source_crop) = object.get("source_crop").and_then(Value::as_str) {
                    match feedback_crop_visual_similarity(Path::new(source_crop), crop_path) {
                        Ok(value) => {
                            visual_score = value.get("score").and_then(Value::as_f64);
                            visual_similarity = value;
                        }
                        Err(err) => crop_error = json!(err),
                    }
                }
            }
            Err(err) => crop_error = json!(err),
        },
        Err(err) => crop_error = json!(err.to_string()),
    }
    let center_component = (1.0 - center_error as f64 * 3.0).clamp(0.0, 1.0);
    let area_component = (-(area_log2_error as f64) * 0.75).exp().clamp(0.0, 1.0);
    let crop_component = visual_score.unwrap_or(0.5).clamp(0.0, 1.0);
    let target_silhouette_fit = target_only_screenshot_path
        .map(|path| final_yaw_render_mask_fit_report(object, path))
        .unwrap_or_else(|| {
            json!({
                "enabled": false,
                "reason": "missing_target_only_render",
            })
        });
    let target_silhouette_score = target_silhouette_fit
        .get("score")
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite());
    let foreground_ratio = foreground
        .get("foreground_ratio")
        .and_then(Value::as_f64)
        .unwrap_or(0.0)
        .clamp(0.0, 1.0);
    let foreground_component = (foreground_ratio / 0.18).clamp(0.0, 1.0);
    let (score, score_policy) = if let Some(mask_component) = target_silhouette_score {
        (
            (0.44 * mask_component)
                + (0.22 * bbox_iou as f64)
                + (0.14 * center_component)
                + (0.08 * area_component)
                + (0.08 * crop_component)
                + (0.04 * foreground_component),
            "0.44*source_mask_target_silhouette + 0.22*bbox_iou + 0.14*center + 0.08*area + 0.08*crop + 0.04*foreground",
        )
    } else {
        (
            (0.42 * bbox_iou as f64)
                + (0.28 * center_component)
                + (0.18 * area_component)
                + (0.08 * crop_component)
                + (0.04 * foreground_component),
            "0.42*bbox_iou + 0.28*center + 0.18*area + 0.08*crop + 0.04*foreground",
        )
    };
    json!({
        "score": score.clamp(0.0, 1.0),
        "bbox_iou": bbox_iou,
        "center_error": center_error,
        "area_log2_error": area_log2_error,
        "center_component": center_component,
        "area_component": area_component,
        "crop_visual_score": visual_score,
        "foreground_ratio": foreground_ratio,
        "foreground_component": foreground_component,
        "target_silhouette_score": target_silhouette_score,
        "target_silhouette_fit": target_silhouette_fit,
        "source_bbox": source_bbox,
        "projected_bbox": target_bbox,
        "rendered_crop": crop_path.display().to_string(),
        "visual_similarity": visual_similarity,
        "crop_error": crop_error,
        "score_policy": score_policy,
    })
}

fn final_yaw_candidate_visual_score(candidate: &Value) -> Option<f64> {
    candidate
        .pointer("/final_yaw_visual_fit/score")
        .and_then(Value::as_f64)
        .or_else(|| {
            candidate
                .pointer("/visual_fit/score")
                .and_then(Value::as_f64)
        })
        .filter(|value| value.is_finite())
}

pub(crate) fn final_yaw_visual_best_response(
    task: &Value,
    response: &SceneRotationSelectionResponse,
) -> SceneRotationSelectionResponse {
    let mut objects = Vec::with_capacity(response.objects.len());
    for selection in &response.objects {
        let Some(object) = task
            .get("objects")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .find(|object| {
                object.get("index").and_then(Value::as_u64) == Some(selection.index as u64)
            })
        else {
            objects.push(selection.clone());
            continue;
        };
        if !final_yaw_object_allows_fine_refinement(object) {
            objects.push(selection.clone());
            continue;
        }
        let Some(candidates) = object
            .pointer("/rotation_selection/candidates")
            .and_then(Value::as_array)
        else {
            objects.push(selection.clone());
            continue;
        };
        let selected_score = candidates
            .iter()
            .find(|candidate| {
                candidate.get("candidate_index").and_then(Value::as_u64)
                    == Some(selection.candidate_index as u64)
            })
            .and_then(final_yaw_candidate_visual_score)
            .unwrap_or(f64::NEG_INFINITY);
        let best = candidates
            .iter()
            .filter_map(|candidate| {
                let index = candidate.get("candidate_index").and_then(Value::as_u64)? as usize;
                let score = final_yaw_candidate_visual_score(candidate)?;
                Some((score, index))
            })
            .max_by(|left, right| left.0.total_cmp(&right.0));
        if final_yaw_object_is_edge_cropped_large_seating(object)
            && selection.confidence >= 0.72
            && candidates
                .iter()
                .find(|candidate| {
                    candidate.get("candidate_index").and_then(Value::as_u64)
                        == Some(selection.candidate_index as u64)
                })
                .is_some_and(final_yaw_candidate_has_rendered_context)
            && best.is_none_or(|(best_score, _)| {
                selected_score.is_finite()
                    && best_score
                        <= selected_score
                            + FINAL_YAW_EDGE_SEATING_CONFIDENT_SELECTION_MAX_VISUAL_REGRESSION
            })
        {
            objects.push(selection.clone());
            continue;
        }
        if let Some(parent_guard) = final_yaw_fine_parent_hysteresis_selection(
            object,
            selection,
            candidates,
            selected_score,
        ) {
            objects.push(parent_guard);
            continue;
        }
        let Some((best_score, best_index)) = best else {
            objects.push(selection.clone());
            continue;
        };
        if best_index != selection.candidate_index
            && best_score > selected_score + FINAL_YAW_VISUAL_SCORE_IMPROVEMENT
        {
            let mut next = selection.clone();
            next.candidate_index = best_index;
            next.confidence = next.confidence.max(0.86);
            next.rationale = format!(
                "visual fit gate selected candidate {best_index}: score {best_score:.3} beats selected candidate {} score {:.3} by more than {:.3}. Previous rationale: {}",
                selection.candidate_index,
                selected_score,
                FINAL_YAW_VISUAL_SCORE_IMPROVEMENT,
                selection.rationale
            );
            objects.push(next);
        } else {
            objects.push(selection.clone());
        }
    }
    SceneRotationSelectionResponse { objects }
}

fn final_yaw_fine_parent_hysteresis_selection(
    object: &Value,
    selection: &burn_synth_scene::SceneRotationSelection,
    candidates: &[Value],
    selected_score: f64,
) -> Option<burn_synth_scene::SceneRotationSelection> {
    if !selected_score.is_finite() || !final_yaw_object_is_edge_cropped_large_seating(object) {
        return None;
    }
    let selected = candidates.iter().find(|candidate| {
        candidate.get("candidate_index").and_then(Value::as_u64)
            == Some(selection.candidate_index as u64)
    })?;
    let parent_index = selected
        .pointer("/fine_refinement/parent_candidate_index")
        .and_then(Value::as_u64)? as usize;
    let parent = candidates.iter().find(|candidate| {
        candidate.get("candidate_index").and_then(Value::as_u64) == Some(parent_index as u64)
    })?;
    let parent_score = final_yaw_candidate_visual_score(parent)?;
    if selected_score >= parent_score + FINAL_YAW_EDGE_SEATING_FINE_PARENT_SCORE_IMPROVEMENT {
        return None;
    }
    let mut next = selection.clone();
    next.candidate_index = parent_index;
    next.confidence = next.confidence.clamp(0.72, 0.84);
    next.rationale = format!(
        "fine-parent hysteresis kept edge-cropped large seating candidate {parent_index}: fine candidate {} score {:.3} did not improve parent score {:.3} by at least {:.3}. Previous rationale: {}",
        selection.candidate_index,
        selected_score,
        parent_score,
        FINAL_YAW_EDGE_SEATING_FINE_PARENT_SCORE_IMPROVEMENT,
        selection.rationale
    );
    Some(next)
}

pub(crate) fn final_yaw_add_fine_refinement_candidates(
    task: &mut Value,
    response: &SceneRotationSelectionResponse,
) -> usize {
    let Some(objects) = task.get_mut("objects").and_then(Value::as_array_mut) else {
        return 0;
    };
    let mut added = 0usize;
    for selection in &response.objects {
        let Some(object) = objects.iter_mut().find(|object| {
            object.get("index").and_then(Value::as_u64) == Some(selection.index as u64)
        }) else {
            continue;
        };
        if !final_yaw_object_allows_fine_refinement(object) {
            continue;
        }
        let current_yaw = object
            .get("current_yaw_degrees")
            .and_then(Value::as_f64)
            .map(|value| value as f32);
        let Some(candidates) = object
            .pointer_mut("/rotation_selection/candidates")
            .and_then(Value::as_array_mut)
        else {
            continue;
        };
        let Some(parent) = candidates
            .iter()
            .find(|candidate| {
                candidate.get("candidate_index").and_then(Value::as_u64)
                    == Some(selection.candidate_index as u64)
            })
            .cloned()
        else {
            continue;
        };
        let Some(parent_yaw) = parent
            .get("candidate_yaw_degrees")
            .and_then(Value::as_f64)
            .filter(|value| value.is_finite())
            .map(|value| value as f32)
        else {
            continue;
        };
        let current_yaw = current_yaw.unwrap_or(parent_yaw);
        let mut next_index = candidates
            .iter()
            .filter_map(|candidate| candidate.get("candidate_index").and_then(Value::as_u64))
            .max()
            .map(|value| value as usize + 1)
            .unwrap_or(candidates.len());
        for delta in FINAL_YAW_FINE_DELTAS_DEGREES {
            let yaw = normalize_degrees(parent_yaw + delta);
            if candidates.iter().any(|candidate| {
                candidate
                    .get("candidate_yaw_degrees")
                    .and_then(Value::as_f64)
                    .map(|candidate_yaw| normalize_degrees(candidate_yaw as f32 - yaw).abs() < 1.0)
                    .unwrap_or(false)
            }) {
                continue;
            }
            candidates.push(json!({
                "candidate_index": next_index,
                "candidate_yaw_degrees": yaw,
                "yaw_degrees": yaw,
                "yaw_delta_degrees": normalize_degrees(yaw - current_yaw),
                "source": "fine_probe",
                "fine_refinement": {
                    "parent_candidate_index": selection.candidate_index,
                    "parent_yaw_degrees": parent_yaw,
                    "delta_degrees": delta,
                },
                "selected": false,
            }));
            next_index += 1;
            added += 1;
        }
    }
    added
}

fn final_yaw_object_allows_fine_refinement(object: &Value) -> bool {
    let descriptor = final_yaw_object_descriptor(object);
    descriptor.contains("sofa")
        || descriptor.contains("couch")
        || descriptor.contains("sectional")
        || descriptor.contains("loveseat")
        || descriptor.contains("settee")
        || descriptor.contains("table")
        || descriptor.contains("desk")
        || descriptor.contains("counter")
}

fn final_yaw_object_is_edge_cropped_large_seating(object: &Value) -> bool {
    final_yaw_object_is_large_seating(object)
        && object
            .get("source_bbox")
            .and_then(final_yaw_json_array4)
            .is_some_and(final_yaw_bbox_touches_image_edge)
}

fn final_yaw_object_is_large_seating(object: &Value) -> bool {
    let descriptor = final_yaw_object_descriptor(object);
    descriptor.contains("sofa")
        || descriptor.contains("couch")
        || descriptor.contains("sectional")
        || descriptor.contains("loveseat")
        || descriptor.contains("settee")
}

fn final_yaw_object_descriptor(object: &Value) -> String {
    ["object_id", "label", "asset_id", "instance_id"]
        .iter()
        .filter_map(|key| object.get(*key).and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase()
}

fn final_yaw_bbox_touches_image_edge(bbox: [f32; 4]) -> bool {
    bbox[0] <= 0.03 || bbox[1] <= 0.03 || bbox[2] >= 0.97 || bbox[3] >= 0.97
}

pub(crate) fn final_yaw_metric_best_response(
    task: &Value,
) -> Option<SceneRotationSelectionResponse> {
    let objects = task
        .get("objects")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|object| {
            let index = object.get("index").and_then(Value::as_u64)? as usize;
            let candidates = object
                .pointer("/rotation_selection/candidates")
                .and_then(Value::as_array)?;
            let candidate = candidates
                .iter()
                .filter(|candidate| candidate.get("geometry_metrics").is_some())
                .filter(|candidate| final_yaw_candidate_safe_without_renderer(object, candidate))
                .filter_map(|candidate| {
                    let score = final_yaw_candidate_metric_score(candidate)?;
                    Some((score, candidate))
                })
                .min_by(|left, right| left.0.total_cmp(&right.0))
                .map(|(_, candidate)| candidate)?;
            let candidate_index =
                candidate.get("candidate_index").and_then(Value::as_u64)? as usize;
            Some(burn_synth_scene::SceneRotationSelection {
                index,
                candidate_index,
                confidence: 1.0,
                rationale: "renderer unavailable; selected lowest safe measured final-pose loss"
                    .to_string(),
            })
        })
        .collect::<Vec<_>>();
    (!objects.is_empty()).then_some(SceneRotationSelectionResponse { objects })
}

fn final_yaw_candidate_safe_without_renderer(object: &Value, candidate: &Value) -> bool {
    if final_yaw_candidate_passed_geometry_gate(candidate) {
        return true;
    }
    !final_yaw_object_requires_rendered_metric_fallback(object)
        && final_yaw_candidate_has_relaxed_metric_evidence(candidate)
}

fn final_yaw_candidate_passed_geometry_gate(candidate: &Value) -> bool {
    candidate
        .get("passed")
        .or_else(|| {
            candidate
                .get("geometry_metrics")
                .and_then(|metrics| metrics.get("passed"))
        })
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

fn final_yaw_object_requires_rendered_metric_fallback(object: &Value) -> bool {
    let descriptor = [
        object.get("label").and_then(Value::as_str).unwrap_or(""),
        object
            .get("object_id")
            .and_then(Value::as_str)
            .unwrap_or(""),
    ]
    .join(" ")
    .to_ascii_lowercase();
    descriptor.contains("sofa")
        || descriptor.contains("couch")
        || descriptor.contains("sectional")
        || descriptor.contains("table")
        || descriptor.contains("desk")
}

fn final_yaw_candidate_has_relaxed_metric_evidence(candidate: &Value) -> bool {
    let bbox = final_yaw_candidate_metric(candidate, "bbox_iou").unwrap_or(0.0);
    let mask = final_yaw_candidate_metric(candidate, "mask_iou").unwrap_or(0.0);
    let depth = final_yaw_candidate_metric(candidate, "depth_error_m").unwrap_or(f32::INFINITY);
    bbox >= 0.50 && mask >= 0.40 && depth <= 0.75
}

fn final_yaw_candidate_has_rendered_context(candidate: &Value) -> bool {
    candidate.get("rendered_candidate_full_scene").is_some()
        || candidate.get("rendered_candidate_full_frame").is_some()
}

fn final_yaw_candidate_metric_score(candidate: &Value) -> Option<f32> {
    final_yaw_candidate_metric(candidate, "loss").or_else(|| {
        let bbox = final_yaw_candidate_metric(candidate, "bbox_iou");
        let mask = final_yaw_candidate_metric(candidate, "mask_iou");
        let depth = final_yaw_candidate_metric(candidate, "depth_error_m");
        let surface = final_yaw_candidate_metric(candidate, "surface_depth_loss");
        if bbox.is_none() && mask.is_none() && depth.is_none() && surface.is_none() {
            return None;
        }
        Some(
            (1.0 - bbox.unwrap_or(0.0))
                + (1.0 - mask.unwrap_or(0.0))
                + depth.unwrap_or(0.0) * 0.25
                + surface.unwrap_or(0.0) * 0.10,
        )
    })
}

fn final_yaw_candidate_metric(candidate: &Value, key: &str) -> Option<f32> {
    candidate
        .get(key)
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
        .map(|value| value as f32)
        .or_else(|| {
            candidate
                .get("geometry_metrics")
                .and_then(|metrics| metrics.get(key))
                .and_then(Value::as_f64)
                .filter(|value| value.is_finite())
                .map(|value| value as f32)
        })
}

pub(crate) fn final_yaw_refinement_prompt(task: &Value) -> String {
    let task_json = serde_json::to_string_pretty(task)
        .unwrap_or_else(|_| serde_json::to_string(task).unwrap_or_default());
    format!(
        "You are selecting final bounded pose candidates for a 3D scene composition.\n\
         The goal is not to choose an object-canonical front view. The goal is to choose the \
         candidate whose yaw makes the generated 3D object semantically match the original source \
         image from the source camera perspective. For each object, use the original full source \
         scene, source-reference overlay, object source mask, and mask-tight source crop as the \
         primary evidence, then compare candidate renders. Full-scene candidate renders keep all \
         other scene objects visible and apply only the selected object's measured candidate pose. \
         Target-only full-frame renders preserve the same source-camera projection and screen-space \
         position but hide other objects so object orientation is easier to inspect. \
         Rendered cardinal probes without geometry metrics are valid choices when they visibly \
         match the source object orientation better than the measured metric-best candidate. \
         Compare candidates in screen space against the source overlay and crop: the same object \
         face should occupy the same side of the source bbox, visible seat cushions should face the \
         camera on the same side, backs/outer shells should not be flipped toward the camera when \
         the source shows seats, and curved/return sections should bend in the same image direction. \
         Use the source mask/bbox to reason about what part of the object is actually visible; do \
         not reward a candidate solely because its AABB covers more pixels when its front/back or \
         left/right semantic surfaces are flipped relative to the source image. \
         For large sofas/sectionals, prioritize visible cushion/back orientation, curve direction, \
         open side, foreground-vs-background ordering, and relation to the table over bbox-only \
         coverage when those disagree. Sectionals are not symmetric under 180-degree flips; do not \
         keep a candidate just because it is labeled current if a cardinal probe better matches the \
         source silhouette and visible seating direction. \
         Fine-probe candidates are local yaw refinements around a previously plausible rendered \
         candidate. Prefer a fine probe only when it provides a clear source-perspective semantic \
         improvement over its parent candidate; if the source object is clipped by an image edge, \
         occluded, or the fine probe only slightly improves bbox/crop metrics while rotating the \
         object farther from the parent semantic orientation, keep the parent/cardinal candidate \
         and report lower confidence. \
         When contact_sheet_path is present, use the contact sheet as the primary visual \
         comparison artifact: source evidence is shown first, and each candidate is identified by \
         the border color recorded in rotation_selection.candidates[].contact_sheet_border_color. \
         Choose exactly one candidate_index from the provided list. Do not invent yaw values, \
         transforms, positions, scales, objects, or explanations that require changing anything \
         except the selected candidate_index. Respect measured metrics in the JSON when comparing \
         candidates with comparable visual orientation. Treat final_yaw_visual_fit.score as a \
         measured source-perspective render match; if you choose a materially lower-scoring \
         candidate, your rationale must identify the exact semantic surface flip or source-camera \
         mismatch that justifies it. Do not let a failed measured candidate win solely on bbox/mask \
         coverage if a rendered candidate is semantically better aligned. \
         If the evidence is ambiguous, keep the current-yaw candidate and use lower confidence. \
         Respond with JSON using exactly this shape: \
         {{\"objects\":[{{\"index\":0,\"candidate_index\":0,\"confidence\":0.0,\"rationale\":\"...\"}}]}}. \
         Include one entry in objects for every input object index.\n\n\
         JSON task:\n{task_json}"
    )
}

pub(crate) fn final_yaw_refinement_image_paths(task: &Value) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    let mut seen = HashSet::new();
    for key in ["source_scene_path", "source_reference_image"] {
        let Some(path) = task.get(key).and_then(Value::as_str) else {
            continue;
        };
        if !path.trim().is_empty() && seen.insert(path.to_string()) {
            paths.push(PathBuf::from(path));
        }
    }
    for object in task
        .get("objects")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        for key in [
            "contact_sheet_path",
            "source_crop",
            "source_mask_path",
            "current_full_scene_render",
        ] {
            let Some(path) = object.get(key).and_then(Value::as_str) else {
                continue;
            };
            if !path.trim().is_empty() && seen.insert(path.to_string()) {
                paths.push(PathBuf::from(path));
            }
        }
        if let Some(path) = object
            .get("source_mask")
            .and_then(|mask| mask.get("mask_png_path"))
            .and_then(Value::as_str)
            && !path.trim().is_empty()
            && seen.insert(path.to_string())
        {
            paths.push(PathBuf::from(path));
        }
        for candidate in object
            .pointer("/rotation_selection/candidates")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            for key in [
                "rendered_candidate_full_scene",
                "rendered_candidate_full_frame",
                "rendered_candidate_object_only_full_frame",
            ] {
                let Some(path) = candidate.get(key).and_then(Value::as_str) else {
                    continue;
                };
                if !path.trim().is_empty() && seen.insert(path.to_string()) {
                    paths.push(PathBuf::from(path));
                }
            }
        }
    }
    paths
}

pub(crate) fn write_final_yaw_candidate_contact_sheets(
    task: &mut Value,
    output_dir: &Path,
) -> Result<Vec<PathBuf>, String> {
    let objects = task
        .get_mut("objects")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| "final-yaw task missing objects array".to_string())?;
    let contact_dir = output_dir.join("contact_sheets");
    fs::create_dir_all(&contact_dir).map_err(|err| {
        format!(
            "failed to create final-yaw contact sheet dir {}: {err}",
            contact_dir.display()
        )
    })?;
    let mut paths = Vec::new();
    for object in objects {
        let index = object.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
        let label = object
            .get("label")
            .and_then(Value::as_str)
            .unwrap_or("object");
        let slug = final_yaw_contact_sheet_slug(label);
        let output_path = contact_dir.join(format!("{index:02}_{slug}_contact_sheet.png"));
        if write_final_yaw_candidate_contact_sheet(object, &output_path)? {
            object["contact_sheet_path"] = json!(output_path.display().to_string());
            paths.push(output_path);
        }
    }
    Ok(paths)
}

fn write_final_yaw_candidate_contact_sheet(
    object: &mut Value,
    output_path: &Path,
) -> Result<bool, String> {
    let mut tiles = Vec::new();
    for key in [
        "source_crop",
        "source_mask_path",
        "current_full_scene_render",
    ] {
        if let Some(path) = object.get(key).and_then(Value::as_str) {
            tiles.push(FinalYawContactSheetTile {
                path: PathBuf::from(path),
                color: image::Rgba([245, 245, 245, 255]),
            });
        }
    }
    let Some(candidates) = object
        .pointer_mut("/rotation_selection/candidates")
        .and_then(Value::as_array_mut)
    else {
        return Ok(false);
    };
    for candidate in candidates.iter_mut() {
        let candidate_index = candidate
            .get("candidate_index")
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize;
        let (color_name, color) = final_yaw_candidate_contact_color(candidate_index);
        candidate["contact_sheet_border_color"] = json!({
            "name": color_name,
            "rgba": [color[0], color[1], color[2], color[3]],
        });
        let Some(path) = candidate
            .get("rendered_candidate_full_scene")
            .or_else(|| candidate.get("rendered_candidate_full_frame"))
            .and_then(Value::as_str)
        else {
            continue;
        };
        tiles.push(FinalYawContactSheetTile {
            path: PathBuf::from(path),
            color,
        });
    }
    if tiles.len() <= 1 {
        return Ok(false);
    }
    let columns = FINAL_YAW_CONTACT_SHEET_COLUMNS.min(tiles.len()).max(1);
    let rows = tiles.len().div_ceil(columns);
    let tile_w = FINAL_YAW_CONTACT_SHEET_TILE_WIDTH + FINAL_YAW_CONTACT_SHEET_BORDER * 2;
    let tile_h = FINAL_YAW_CONTACT_SHEET_TILE_HEIGHT + FINAL_YAW_CONTACT_SHEET_BORDER * 2;
    let width = (columns as u32 * tile_w)
        + ((columns.saturating_sub(1)) as u32 * FINAL_YAW_CONTACT_SHEET_GAP);
    let height =
        (rows as u32 * tile_h) + ((rows.saturating_sub(1)) as u32 * FINAL_YAW_CONTACT_SHEET_GAP);
    let mut sheet = image::RgbaImage::from_pixel(width, height, image::Rgba([12, 14, 18, 255]));
    for (tile_index, tile) in tiles.iter().enumerate() {
        let Some(tile_image) = load_final_yaw_contact_tile(&tile.path)? else {
            continue;
        };
        let col = tile_index % columns;
        let row = tile_index / columns;
        let x = col as u32 * (tile_w + FINAL_YAW_CONTACT_SHEET_GAP);
        let y = row as u32 * (tile_h + FINAL_YAW_CONTACT_SHEET_GAP);
        draw_final_yaw_contact_border(&mut sheet, x, y, tile_w, tile_h, tile.color);
        image::imageops::overlay(
            &mut sheet,
            &tile_image,
            (x + FINAL_YAW_CONTACT_SHEET_BORDER) as i64,
            (y + FINAL_YAW_CONTACT_SHEET_BORDER) as i64,
        );
    }
    sheet.save(output_path).map_err(|err| {
        format!(
            "failed to write final-yaw contact sheet {}: {err}",
            output_path.display()
        )
    })?;
    Ok(true)
}

fn final_yaw_contact_sheet_slug(value: &str) -> String {
    let slug = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect::<String>()
        .split('_')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("_");
    if slug.is_empty() {
        "object".to_string()
    } else {
        slug
    }
}

struct FinalYawContactSheetTile {
    path: PathBuf,
    color: image::Rgba<u8>,
}

fn final_yaw_candidate_contact_color(index: usize) -> (&'static str, image::Rgba<u8>) {
    const COLORS: [(&str, [u8; 4]); 12] = [
        ("red", [238, 75, 75, 255]),
        ("green", [80, 210, 116, 255]),
        ("blue", [84, 150, 255, 255]),
        ("magenta", [238, 84, 214, 255]),
        ("cyan", [60, 210, 218, 255]),
        ("orange", [255, 160, 56, 255]),
        ("purple", [168, 116, 255, 255]),
        ("yellow", [245, 220, 70, 255]),
        ("pink", [255, 125, 170, 255]),
        ("lime", [168, 230, 72, 255]),
        ("teal", [62, 184, 160, 255]),
        ("white", [244, 244, 244, 255]),
    ];
    let (name, rgba) = COLORS[index % COLORS.len()];
    (name, image::Rgba(rgba))
}

fn load_final_yaw_contact_tile(path: &Path) -> Result<Option<image::RgbaImage>, String> {
    if path.as_os_str().is_empty() || !path.exists() {
        return Ok(None);
    }
    let image = image::open(path).map_err(|err| {
        format!(
            "failed to open final-yaw contact tile {}: {err}",
            path.display()
        )
    })?;
    Ok(Some(
        image
            .resize_to_fill(
                FINAL_YAW_CONTACT_SHEET_TILE_WIDTH,
                FINAL_YAW_CONTACT_SHEET_TILE_HEIGHT,
                image::imageops::FilterType::Triangle,
            )
            .to_rgba8(),
    ))
}

fn draw_final_yaw_contact_border(
    sheet: &mut image::RgbaImage,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    color: image::Rgba<u8>,
) {
    let max_x = (x + width).min(sheet.width());
    let max_y = (y + height).min(sheet.height());
    for yy in y..max_y {
        for xx in x..max_x {
            let on_border = xx < x + FINAL_YAW_CONTACT_SHEET_BORDER
                || yy < y + FINAL_YAW_CONTACT_SHEET_BORDER
                || xx >= max_x.saturating_sub(FINAL_YAW_CONTACT_SHEET_BORDER)
                || yy >= max_y.saturating_sub(FINAL_YAW_CONTACT_SHEET_BORDER);
            if on_border {
                sheet.put_pixel(xx, yy, color);
            }
        }
    }
}

fn write_final_yaw_source_reference_image(
    source_scene_path: &str,
    task: &Value,
    output_dir: &Path,
) -> Result<Option<PathBuf>, String> {
    if source_scene_path.trim().is_empty() {
        return Ok(None);
    }
    let source_path = Path::new(source_scene_path);
    let mut image = image::open(source_path)
        .map_err(|err| {
            format!(
                "failed to open source scene {}: {err}",
                source_path.display()
            )
        })?
        .to_rgba8();
    let colors = [
        image::Rgba([36, 220, 74, 255]),
        image::Rgba([255, 196, 44, 255]),
        image::Rgba([68, 158, 255, 255]),
        image::Rgba([255, 76, 216, 255]),
    ];
    for (index, object) in task
        .get("objects")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
    {
        if let Some(bbox) = object.get("source_bbox").and_then(final_yaw_json_array4) {
            draw_final_yaw_reference_rect(&mut image, bbox, colors[index % colors.len()]);
        }
    }
    if let Err(err) = fs::create_dir_all(output_dir) {
        return Err(format!(
            "failed to create final-yaw output dir {}: {err}",
            output_dir.display()
        ));
    }
    let output_path = output_dir.join("source_scene_final_yaw_reference.png");
    image.save(&output_path).map_err(|err| {
        format!(
            "failed to write final-yaw source reference {}: {err}",
            output_path.display()
        )
    })?;
    Ok(Some(output_path))
}

fn final_yaw_json_array4(value: &Value) -> Option<[f32; 4]> {
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
}

fn final_yaw_json_array3(value: &Value) -> Option<[f32; 3]> {
    let array = value.as_array()?;
    if array.len() != 3 {
        return None;
    }
    Some([
        array[0].as_f64()? as f32,
        array[1].as_f64()? as f32,
        array[2].as_f64()? as f32,
    ])
}

fn draw_final_yaw_reference_rect(
    image: &mut image::RgbaImage,
    bbox: [f32; 4],
    color: image::Rgba<u8>,
) {
    let width = image.width().max(1);
    let height = image.height().max(1);
    let x0 = (bbox[0].clamp(0.0, 1.0) * (width - 1) as f32).round() as u32;
    let y0 = (bbox[1].clamp(0.0, 1.0) * (height - 1) as f32).round() as u32;
    let x1 = (bbox[2].clamp(0.0, 1.0) * (width - 1) as f32).round() as u32;
    let y1 = (bbox[3].clamp(0.0, 1.0) * (height - 1) as f32).round() as u32;
    let (min_x, max_x) = (x0.min(x1), x0.max(x1));
    let (min_y, max_y) = (y0.min(y1), y0.max(y1));
    for thickness in 0..3 {
        let left = min_x.saturating_sub(thickness);
        let right = (max_x + thickness).min(width - 1);
        let top = min_y.saturating_sub(thickness);
        let bottom = (max_y + thickness).min(height - 1);
        for x in left..=right {
            image.put_pixel(x, top, color);
            image.put_pixel(x, bottom, color);
        }
        for y in top..=bottom {
            image.put_pixel(left, y, color);
            image.put_pixel(right, y, color);
        }
    }
}

fn apply_gpt_ground_calibration<P: SceneAiProvider>(
    pipeline: &ScenePipeline<P>,
    manifest: &mut SceneObjectManifest,
    evidence: &mut SceneGroundingEvidence,
    output_dir: &Path,
    write_artifacts: bool,
    extra_image_paths: &[PathBuf],
) -> Result<SceneGroundCalibrationReport, String> {
    let request =
        pipeline.prepare_ground_calibration_request(manifest, evidence, extra_image_paths);
    let response = pipeline
        .calibrate_ground(&request)
        .map_err(|err| err.to_string())?;
    let (next_manifest, next_evidence, report) =
        apply_ground_calibration_response(manifest, evidence, &response)
            .map_err(|err| err.to_string())?;
    if write_artifacts {
        write_json_file(
            &output_dir.join("ground_calibration_gpt_request.json"),
            &request,
        )
        .map_err(|err| err.to_string())?;
        write_json_file(
            &output_dir.join("ground_calibration_gpt_response.json"),
            &response,
        )
        .map_err(|err| err.to_string())?;
        write_json_file(
            &output_dir.join("ground_calibration_gpt_report.json"),
            &report,
        )
        .map_err(|err| err.to_string())?;
    }
    *manifest = next_manifest;
    *evidence = next_evidence;
    Ok(report)
}

fn deterministic_ground_calibration_report(
    manifest: &SceneObjectManifest,
    evidence: &SceneGroundingEvidence,
    mode: SceneGroundCalibrationMode,
) -> SceneGroundCalibrationReport {
    let vertical_fov_degrees = evidence
        .camera
        .vertical_fov_degrees
        .or_else(|| {
            evidence
                .depth
                .as_ref()
                .and_then(|depth| depth.vertical_fov_degrees)
        })
        .or_else(|| {
            manifest
                .scene_calibration
                .and_then(|calibration| calibration.vertical_fov_degrees)
        })
        .unwrap_or(60.0);
    let floor_confidence = evidence.floor.confidence.unwrap_or(0.0);
    let floor_residual_m = evidence.floor.residual_m.unwrap_or(0.0);
    let camera_height_m = (-evidence.floor.distance_m).max(0.0);
    let rationale = match mode {
        SceneGroundCalibrationMode::DepthFirst => {
            "depth-first deterministic calibration: retained DepthPro intrinsics/floor evidence without GPT"
        }
        SceneGroundCalibrationMode::DepthHeuristic => {
            "depth-heuristic calibration: retained existing depth/floor heuristic evidence without GPT"
        }
        SceneGroundCalibrationMode::Gpt => "GPT calibration disabled for deterministic report path",
    };
    SceneGroundCalibrationReport {
        schema_version: 1,
        source_scene_path: evidence.source_image_path.clone(),
        previous_camera: evidence.camera,
        previous_floor: evidence.floor,
        response: burn_synth_scene::SceneGroundCalibrationResponse {
            camera_height_m,
            vertical_fov_degrees,
            floor_confidence,
            floor_residual_m,
            scene_calibration: manifest.scene_calibration,
            rationale: rationale.to_string(),
        },
        applied_camera: evidence.camera,
        applied_floor: evidence.floor,
    }
}

fn mark_canonical_pose_verification_failure(response: &mut Value, verification: &Value) {
    let requires_attention = verification
        .get("requires_attention")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if !requires_attention {
        return;
    }
    if !response
        .get("failed_stage")
        .map(Value::is_null)
        .unwrap_or(true)
    {
        return;
    }
    response["failed_stage"] = json!("canonical_pose_calibration.visual_verification");
    response["next_action"] = json!({
        "reason": "Canonical pose render-sweep was requested but visual thumbnail evidence was missing, partial, or selected through fallback.",
        "status": verification.get("status").cloned().unwrap_or(Value::Null),
        "inspect": {
            "verification": "canonical_pose_verification.json",
            "selection": "canonical_pose_selection.json",
            "calibration": "canonical_pose_calibration_report.json",
            "render_root": verification
                .get("render_report")
                .and_then(|report| report.get("root"))
                .cloned()
                .unwrap_or(Value::Null),
            "viewer_log": verification
                .get("render_report")
                .and_then(|report| report.get("viewer_log"))
                .cloned()
                .unwrap_or(Value::Null),
        },
    });
}

fn canonical_pose_thumbnail_spawn_command(
    spawn_basis: CanonicalPoseThumbnailSpawnBasis<'_>,
    asset: &SceneAssetBinding,
    translation: [f32; 3],
    rotation: [f32; 4],
) -> Value {
    let mut command = match spawn_basis {
        CanonicalPoseThumbnailSpawnBasis::Cache(cache_key) => json!({
            "type": "spawn_cached",
            "cache_key": cache_key,
        }),
        CanonicalPoseThumbnailSpawnBasis::Path(path) => json!({
            "type": "spawn_path",
            "path": path,
            "cache_key": asset.cache_key.clone().unwrap_or_else(|| asset.asset_id.clone()),
        }),
    };
    command["translation"] = json!(translation);
    command["rotation"] = json!(rotation);
    command["scale"] = json!([1.0, 1.0, 1.0]);
    command["select"] = json!(false);
    if let Some(local_aabb) = asset.local_aabb {
        command["local_aabb"] = json!(local_aabb);
    }
    command
}

fn canonical_pose_yaw_file_component(yaw_degrees: f32) -> String {
    let rounded = yaw_degrees.round() as i32;
    if rounded < 0 {
        format!("neg{}", rounded.abs())
    } else {
        format!("pos{rounded}")
    }
}

fn append_canonical_pose_render_warning(run: &mut CanonicalPoseCalibrationRun, warning: &str) {
    for report in &mut run.reports {
        report.warnings.push(warning.to_string());
    }
}

fn canonical_pose_thumbnail_status_summary(status: &Value) -> Value {
    json!({
        "ok": status.get("ok").cloned().unwrap_or(Value::Null),
        "message": status.get("message").cloned().unwrap_or(Value::Null),
        "applied_commands": status
            .get("applied_commands")
            .cloned()
            .unwrap_or(Value::Null),
        "projected_items": status
            .get("projected_items")
            .cloned()
            .unwrap_or(Value::Null),
        "camera": status.get("camera").cloned().unwrap_or(Value::Null),
        "screenshots": status.get("screenshots").cloned().unwrap_or(Value::Null),
    })
}

fn canonical_pose_thumbnail_capture_summary(capture_ack: &Value) -> Value {
    json!({
        "output_path": capture_ack
            .get("output_path")
            .cloned()
            .unwrap_or(Value::Null),
        "acknowledged": capture_ack
            .pointer("/acknowledgement/acknowledged")
            .cloned()
            .unwrap_or(Value::Null),
        "sequence": capture_ack
            .pointer("/acknowledgement/sequence")
            .cloned()
            .unwrap_or(Value::Null),
    })
}

impl SceneAiProvider for NoopSceneProvider {
    fn plan_objects(&self, _request: &SceneReasoningRequest) -> SceneResult<SceneObjectManifest> {
        Err(burn_synth_scene::SceneError::Provider(
            "offline schema preparation cannot plan objects".to_string(),
        ))
    }

    fn generate_object_images(
        &self,
        _request: &burn_synth_scene::ObjectImageRequest,
    ) -> SceneResult<Vec<Vec<u8>>> {
        Err(burn_synth_scene::SceneError::Provider(
            "offline schema preparation cannot generate images".to_string(),
        ))
    }

    fn plan_scene_bsn(&self, _request: &SceneBsnRequest) -> SceneResult<String> {
        Err(burn_synth_scene::SceneError::Provider(
            "offline schema preparation cannot plan BSN".to_string(),
        ))
    }
}

fn validate_scene_pose_fit_mode(mode: ScenePoseFitMode) -> Result<(), String> {
    match mode {
        ScenePoseFitMode::ProjectedAabb => Ok(()),
        ScenePoseFitMode::RenderedSilhouette => Ok(()),
    }
}

#[derive(Clone, Debug)]
pub(crate) struct SceneAssetLiftPolicy {
    pub(crate) synthesis_models: Vec<SynthesisModel>,
    pub(crate) output_format: AssetOutputFormat,
    pub(crate) requires_mesh_assets: bool,
    pub(crate) warnings: Vec<String>,
}

impl SceneAssetLiftPolicy {
    fn to_json(&self) -> Value {
        json!({
            "synthesis_models": self
                .synthesis_models
                .iter()
                .map(|model| model.as_str())
                .collect::<Vec<_>>(),
            "output_format": self.output_format.as_str(),
            "requires_mesh_assets": self.requires_mesh_assets,
            "warnings": self.warnings,
        })
    }
}

pub(crate) fn scene_asset_lift_policy(
    args: &SceneBuildFromImageArgs,
) -> Result<SceneAssetLiftPolicy, String> {
    let requires_mesh_assets = scene_build_requires_mesh_assets(args);
    let synthesis_models = args
        .synthesis_models
        .clone()
        .map(sanitize_synthesis_models)
        .unwrap_or_else(|| vec![SynthesisModel::Trellis]);
    let mut warnings = Vec::new();

    if requires_mesh_assets {
        let invalid = synthesis_models
            .iter()
            .copied()
            .filter(|model| !scene_synthesis_model_outputs_mesh(*model))
            .map(|model| model.as_str())
            .collect::<Vec<_>>();
        if !invalid.is_empty() {
            return Err(format!(
                "explicit scene composition with {:?} pose fitting requires GLB mesh assets, but selected image-to-3d model(s) [{}] emit non-mesh outputs. Use trellis or triposg, or disable rendered-silhouette/table pose fitting for splat-only experiments.",
                args.pose_fit,
                invalid.join(", ")
            ));
        }
        if !synthesis_models.contains(&SynthesisModel::Trellis) {
            warnings.push(
                "Trellis.2 is the preferred scene asset model for PBR/mesh quality; non-Trellis mesh models are treated as an explicit quality ablation."
                    .to_string(),
            );
        }
    }

    Ok(SceneAssetLiftPolicy {
        synthesis_models,
        output_format: if requires_mesh_assets {
            AssetOutputFormat::Glb
        } else {
            AssetOutputFormat::Auto
        },
        requires_mesh_assets,
        warnings,
    })
}

pub(crate) fn scene_build_requires_mesh_assets(args: &SceneBuildFromImageArgs) -> bool {
    args.lift_assets
        && args.composition_mode == SceneCompositionMode::CvGrounded
        && (args.pose_fit == ScenePoseFitMode::RenderedSilhouette
            || args.rotation_fit != SceneRotationFitMode::Off
            || args.object_pose_refinement.geometry_enabled())
}

fn scene_instance_generation_type_aware(mode: SceneInstanceGenerationMode) -> bool {
    matches!(
        mode,
        SceneInstanceGenerationMode::TypeAwareReuse | SceneInstanceGenerationMode::FineGrainedTypes
    )
}

fn scene_type_aware_categories(
    mode: SceneInstanceGenerationMode,
    requested: Option<&[String]>,
) -> Vec<SceneObjectCategory> {
    if !scene_instance_generation_type_aware(mode) {
        return Vec::new();
    }
    let values = requested
        .map(|values| values.to_vec())
        .unwrap_or_else(|| vec!["chair".to_string()]);
    values
        .iter()
        .filter_map(|value| parse_scene_category(value))
        .filter(|category| {
            matches!(category, SceneObjectCategory::Chair)
                || mode == SceneInstanceGenerationMode::FineGrainedTypes
        })
        .fold(Vec::new(), |mut out, category| {
            if !out.contains(&category) {
                out.push(category);
            }
            out
        })
}

fn scene_synthesis_model_outputs_mesh(model: SynthesisModel) -> bool {
    matches!(model, SynthesisModel::Triposg | SynthesisModel::Trellis)
}

pub(crate) fn validate_scene_asset_outputs_for_policy(
    asset_outputs: &Value,
    policy: &SceneAssetLiftPolicy,
) -> Result<(), String> {
    if !policy.requires_mesh_assets {
        return Ok(());
    }
    let Some(items) = asset_outputs.get("items").and_then(Value::as_array) else {
        return Err(
            "scene asset contract failed: images_to_assets response missing items".to_string(),
        );
    };
    let mut failures = Vec::new();
    for (index, item) in items.iter().enumerate() {
        let object = item
            .get("input_image_path")
            .and_then(Value::as_str)
            .or_else(|| item.get("id").and_then(Value::as_str))
            .unwrap_or("<unknown>");
        let asset_kind = item
            .get("asset_kind")
            .and_then(Value::as_str)
            .unwrap_or("<missing>");
        let output_format = item
            .get("output_format")
            .and_then(Value::as_str)
            .unwrap_or("<missing>");
        if asset_kind != "mesh" {
            failures.push(format!(
                "item {index} ({object}) produced asset_kind={asset_kind}, expected mesh"
            ));
        }
        if output_format != "glb" {
            failures.push(format!(
                "item {index} ({object}) produced output_format={output_format}, expected glb"
            ));
        }
        if item.get("local_aabb").is_none_or(Value::is_null) {
            failures.push(format!(
                "item {index} ({object}) has no local_aabb for scene placement"
            ));
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "scene asset contract failed: {}",
            failures.join("; ")
        ))
    }
}

fn scene_pose_fit_note(mode: ScenePoseFitMode) -> &'static str {
    match mode {
        ScenePoseFitMode::ProjectedAabb => {
            "projected AABB/contact/depth fitting; segmentation masks refine source bboxes when available"
        }
        ScenePoseFitMode::RenderedSilhouette => {
            "bounded source-camera visible-surface fitting; SAM/bbox masks, DepthPro dense depth sidecar crops, depth distributions, and mask point moments are matched against projected generated GLB surfaces before feedback"
        }
    }
}

pub(crate) fn scene_object_image_generation_policy(
    args: &SceneBuildFromImageArgs,
    default_candidate_count: usize,
) -> ObjectImageGenerationPolicy {
    let requested_candidate_count = args
        .candidate_count
        .unwrap_or(default_candidate_count)
        .max(1);
    let quality_profile = args.quality_profile.unwrap_or(SceneQualityProfile::Quality);
    let default_candidates_per_attempt =
        if quality_profile == SceneQualityProfile::Quality && requested_candidate_count > 1 {
            requested_candidate_count.min(2)
        } else {
            1
        };
    let candidates_per_attempt = args
        .candidate_batch_size
        .unwrap_or(default_candidates_per_attempt)
        .max(1);
    let default_max_attempts = requested_candidate_count.saturating_add(candidates_per_attempt - 1)
        / candidates_per_attempt;
    ObjectImageGenerationPolicy {
        min_score: args
            .min_reconstruction_score
            .unwrap_or(DEFAULT_SCENE_RECONSTRUCTION_IMAGE_SCORE),
        max_attempts_per_object: args
            .candidate_retry_attempts
            .unwrap_or(default_max_attempts)
            .max(1),
        candidates_per_attempt,
    }
}

fn scene_object_image_request_parallelism(request_count: usize) -> usize {
    request_count.clamp(1, 3)
}

struct SceneBuildProgressReporter<'a, F>
where
    F: FnMut(SceneBuildProgressEvent),
{
    emit: &'a mut F,
    run_id: String,
    started: Instant,
    sequence: u64,
    output_dir: Option<PathBuf>,
    write_artifacts: bool,
}

impl<'a, F> SceneBuildProgressReporter<'a, F>
where
    F: FnMut(SceneBuildProgressEvent),
{
    fn new(emit: &'a mut F, write_artifacts: bool) -> Self {
        Self {
            emit,
            run_id: format!("scene_build_{}", next_scene_sequence()),
            started: Instant::now(),
            sequence: 0,
            output_dir: None,
            write_artifacts,
        }
    }

    fn set_output_dir(&mut self, output_dir: PathBuf) {
        if let Some(name) = output_dir.file_name().and_then(|value| value.to_str())
            && !name.trim().is_empty()
        {
            self.run_id = name.to_string();
        }
        self.output_dir = Some(output_dir);
    }

    fn emit(
        &mut self,
        stage: impl Into<String>,
        phase: SceneBuildProgressPhase,
        execution: SceneBuildExecutionKind,
        message: impl Into<String>,
        detail: Value,
    ) {
        self.emit_with_items(stage, phase, execution, message, None, None, None, detail);
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_with_items(
        &mut self,
        stage: impl Into<String>,
        phase: SceneBuildProgressPhase,
        execution: SceneBuildExecutionKind,
        message: impl Into<String>,
        item_index: Option<usize>,
        item_count: Option<usize>,
        artifact_path: Option<PathBuf>,
        detail: Value,
    ) {
        self.sequence = self.sequence.saturating_add(1);
        let event = SceneBuildProgressEvent {
            run_id: self.run_id.clone(),
            sequence: self.sequence,
            stage: stage.into(),
            phase,
            execution,
            message: message.into(),
            elapsed_ms: elapsed_ms(self.started.elapsed()),
            item_index,
            item_count,
            artifact_path: artifact_path
                .as_ref()
                .map(|path| path.display().to_string()),
            detail,
        };
        if self.write_artifacts
            && let Some(output_dir) = self.output_dir.as_ref()
            && let Err(err) = append_scene_progress_event(output_dir, &event)
        {
            eprintln!("burn_synth_mcp: failed to write scene progress event: {err}");
        }
        (self.emit)(event);
    }
}

pub(crate) struct SceneFeedbackProgressEmission {
    stage: &'static str,
    phase: SceneBuildProgressPhase,
    execution: SceneBuildExecutionKind,
    message: String,
    item_index: Option<usize>,
    item_count: Option<usize>,
    artifact_path: Option<PathBuf>,
    detail: Value,
}

pub(crate) type SceneFeedbackProgressSink<'a> =
    Rc<RefCell<Box<dyn FnMut(SceneFeedbackProgressEmission) + 'a>>>;

#[allow(clippy::too_many_arguments)]
fn emit_scene_feedback_progress(
    progress: &Option<SceneFeedbackProgressSink<'_>>,
    stage: &'static str,
    phase: SceneBuildProgressPhase,
    execution: SceneBuildExecutionKind,
    message: impl Into<String>,
    item_index: Option<usize>,
    item_count: Option<usize>,
    artifact_path: Option<PathBuf>,
    detail: Value,
) {
    if let Some(progress) = progress {
        (progress.borrow_mut())(SceneFeedbackProgressEmission {
            stage,
            phase,
            execution,
            message: message.into(),
            item_index,
            item_count,
            artifact_path,
            detail,
        });
    }
}

fn runtime_progress_event_json(event: &RuntimeProgressEvent) -> Value {
    match event {
        RuntimeProgressEvent::RunStarted { run, detail } => json!({
            "kind": "run_started",
            "run": run,
            "detail": detail,
            "message": runtime_progress_event_message(event),
        }),
        RuntimeProgressEvent::StageStarted {
            run,
            stage,
            total_steps,
            detail,
        } => json!({
            "kind": "stage_started",
            "run": run,
            "stage": stage,
            "total_steps": total_steps,
            "detail": detail,
            "message": runtime_progress_event_message(event),
        }),
        RuntimeProgressEvent::Step {
            run,
            stage,
            step,
            total_steps,
            step_ms,
            elapsed_ms,
            eta_ms,
            detail,
        } => json!({
            "kind": "step",
            "run": run,
            "stage": stage,
            "step": step,
            "total_steps": total_steps,
            "step_ms": step_ms,
            "elapsed_ms": elapsed_ms,
            "eta_ms": eta_ms,
            "detail": detail,
            "message": runtime_progress_event_message(event),
        }),
        RuntimeProgressEvent::StageCompleted {
            run,
            stage,
            total_steps,
            elapsed_ms,
            detail,
        } => json!({
            "kind": "stage_completed",
            "run": run,
            "stage": stage,
            "total_steps": total_steps,
            "elapsed_ms": elapsed_ms,
            "detail": detail,
            "message": runtime_progress_event_message(event),
        }),
        RuntimeProgressEvent::Warning { run, message } => json!({
            "kind": "warning",
            "run": run,
            "message": message,
        }),
        RuntimeProgressEvent::RunCompleted {
            run,
            elapsed_ms,
            detail,
        } => json!({
            "kind": "run_completed",
            "run": run,
            "elapsed_ms": elapsed_ms,
            "detail": detail,
            "message": runtime_progress_event_message(event),
        }),
    }
}

fn runtime_progress_event_message(event: &RuntimeProgressEvent) -> String {
    match event {
        RuntimeProgressEvent::RunStarted { run, detail } => match detail {
            Some(detail) => format!("{run} started: {detail}"),
            None => format!("{run} started"),
        },
        RuntimeProgressEvent::StageStarted { stage, detail, .. } => match detail {
            Some(detail) => format!("{stage} started: {detail}"),
            None => format!("{stage} started"),
        },
        RuntimeProgressEvent::Step {
            stage,
            step,
            total_steps,
            eta_ms,
            ..
        } => {
            let eta = eta_ms
                .map(|value| format!(", eta_ms={value:.1}"))
                .unwrap_or_default();
            format!("{stage} step {step}/{total_steps}{eta}")
        }
        RuntimeProgressEvent::StageCompleted {
            stage, elapsed_ms, ..
        } => format!("{stage} complete ({elapsed_ms:.1} ms)"),
        RuntimeProgressEvent::Warning { message, .. } => format!("warning: {message}"),
        RuntimeProgressEvent::RunCompleted {
            run, elapsed_ms, ..
        } => format!("{run} complete ({elapsed_ms:.1} ms)"),
    }
}

fn runtime_progress_scene_phase(kind: &str) -> SceneBuildProgressPhase {
    match kind {
        "run_started" | "stage_started" => SceneBuildProgressPhase::Started,
        "run_completed" | "stage_completed" => SceneBuildProgressPhase::Completed,
        "warning" => SceneBuildProgressPhase::Failed,
        _ => SceneBuildProgressPhase::Progress,
    }
}

fn runtime_progress_scene_execution(event: &Value) -> SceneBuildExecutionKind {
    let stage_or_run = event
        .get("stage")
        .or_else(|| event.get("run"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    if stage_or_run.contains("load") || stage_or_run.contains("download") {
        SceneBuildExecutionKind::FileIo
    } else if stage_or_run.contains("rmbg")
        || stage_or_run.contains("foreground")
        || stage_or_run.contains("tripo")
        || stage_or_run.contains("trellis")
        || stage_or_run.contains("dino")
        || stage_or_run.contains("vae")
    {
        SceneBuildExecutionKind::Gpu
    } else {
        SceneBuildExecutionKind::Mixed
    }
}

fn emit_asset_runtime_progress_events<F>(
    progress: &mut SceneBuildProgressReporter<'_, F>,
    events: &[Value],
    chunk_index: usize,
    item_index: Option<usize>,
    item_count: Option<usize>,
    artifact_path: PathBuf,
) where
    F: FnMut(SceneBuildProgressEvent),
{
    for event in events {
        let kind = event
            .get("kind")
            .and_then(Value::as_str)
            .unwrap_or("progress");
        let stage = event
            .get("stage")
            .or_else(|| event.get("run"))
            .and_then(Value::as_str)
            .unwrap_or("runtime");
        let message = event
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or(stage)
            .to_string();
        progress.emit_with_items(
            format!("images_to_assets.{stage}"),
            runtime_progress_scene_phase(kind),
            runtime_progress_scene_execution(event),
            message,
            item_index,
            item_count,
            Some(artifact_path.clone()),
            json!({
                "chunk_index": chunk_index,
                "runtime_progress": event,
            }),
        );
    }
}

fn write_source_depth_normal_evidence_for_asset(
    asset: &SceneAssetBinding,
    evidence: &SceneGroundingEvidence,
    depth_map: Option<&LoadedSceneDepthMap>,
    output_path: &Path,
) -> Result<Option<Value>, String> {
    let Some(depth_map) = depth_map else {
        return Ok(None);
    };
    let Some(object) = evidence.objects.iter().find(|object| {
        object.object_id == asset.object_id
            || object.asset_id.as_deref() == Some(asset.asset_id.as_str())
    }) else {
        return Ok(None);
    };
    let Some(mask) = source_normal_mask(object, evidence) else {
        return Ok(None);
    };
    let bbox = object
        .mask
        .as_ref()
        .map(|mask| mask.bbox)
        .or_else(|| object.detection.as_ref().map(|detection| detection.bbox))
        .unwrap_or([0.0, 0.0, 1.0, 1.0]);
    let bbox = pad_normalized_bbox_for_render(bbox, 0.08);
    let intrinsics = source_depth_intrinsics(evidence, depth_map);
    let depth_sidecar_label = depth_map.raw_path.display().to_string();
    let mask_kind = if object
        .mask
        .as_ref()
        .is_some_and(|mask| !mask.mask_rle.is_empty())
    {
        "sam_rle"
    } else {
        "bbox"
    };
    let input = SourceDepthNormalInput {
        depth_map: DepthMapView {
            depth_m: &depth_map.depth_m,
            width: depth_map.width,
            height: depth_map.height,
        },
        mask: BinaryMaskView {
            data: mask.data(),
            width: mask.width(),
            height: mask.height(),
        },
        bbox,
        intrinsics,
        depth_sidecar_label: Some(depth_sidecar_label.as_str()),
        mask_kind,
    };
    write_render_source_depth_normal_evidence(input, output_path).map(Some)
}

fn source_normal_mask(
    object: &ObjectGroundingEvidence,
    evidence: &SceneGroundingEvidence,
) -> Option<BinaryMask> {
    if let Some(mask) = object.mask.as_ref() {
        if !mask.mask_rle.is_empty() {
            return BinaryMask::decode_rle(mask.image_size[0], mask.image_size[1], &mask.mask_rle)
                .ok();
        }
        return BinaryMask::from_normalized_bbox(mask.image_size[0], mask.image_size[1], mask.bbox)
            .ok();
    }
    let bbox = object
        .detection
        .as_ref()
        .map(|detection| detection.bbox)
        .unwrap_or([0.0, 0.0, 1.0, 1.0]);
    let size = evidence
        .camera
        .image_size
        .or_else(|| evidence.depth.as_ref().and_then(|depth| depth.image_size))
        .unwrap_or([1024, 1024]);
    BinaryMask::from_normalized_bbox(size[0], size[1], bbox).ok()
}

fn source_depth_intrinsics(
    evidence: &SceneGroundingEvidence,
    depth_map: &LoadedSceneDepthMap,
) -> DepthNormalIntrinsics {
    let focal = evidence
        .depth
        .as_ref()
        .and_then(|depth| depth.focal_length_px)
        .or(evidence.camera.focal_length_px)
        .unwrap_or(depth_map.width.max(depth_map.height) as f32);
    let principal = evidence.camera.principal_point.unwrap_or([
        (depth_map.width.saturating_sub(1)) as f32 * 0.5,
        (depth_map.height.saturating_sub(1)) as f32 * 0.5,
    ]);
    DepthNormalIntrinsics {
        fx: focal.max(1.0e-5),
        fy: focal.max(1.0e-5),
        cx: principal[0],
        cy: principal[1],
        width: depth_map.width,
        height: depth_map.height,
    }
}

fn cached_mesh_normal_input(mesh: &CachedSynthMesh) -> MeshNormalInput<'_> {
    MeshNormalInput {
        vertices: &mesh.mesh.vertices,
        faces: &mesh.mesh.faces,
        normals: Some(&mesh.normals),
    }
}

fn scene_aabb_to_render(aabb: SceneAssetAabb) -> RenderAabb {
    RenderAabb {
        min: aabb.min,
        max: aabb.max,
    }
}

fn pad_normalized_bbox_for_render(bbox: [f32; 4], padding: f32) -> [f32; 4] {
    let width = (bbox[2] - bbox[0]).max(0.0);
    let height = (bbox[3] - bbox[1]).max(0.0);
    [
        (bbox[0] - width * padding).clamp(0.0, 1.0),
        (bbox[1] - height * padding).clamp(0.0, 1.0),
        (bbox[2] + width * padding).clamp(0.0, 1.0),
        (bbox[3] + height * padding).clamp(0.0, 1.0),
    ]
}

impl McpServer {
    pub(crate) fn new(config: ServerConfig) -> Self {
        let runtime = SynthRuntime::new(config.runtime_config());
        Self {
            config,
            runtime,
            grounding: SceneGroundingRuntime::default(),
            should_exit: false,
        }
    }

    fn handle_message(&mut self, message: Value) -> Result<Option<Value>, String> {
        let request: RpcRequest = serde_json::from_value(message)
            .map_err(|err| format!("invalid JSON-RPC request: {err}"))?;
        self.handle_request(request)
    }

    fn handle_request(&mut self, request: RpcRequest) -> Result<Option<Value>, String> {
        match request.method.as_str() {
            "initialize" => {
                let params: InitializeParams = request
                    .params
                    .map(serde_json::from_value)
                    .transpose()
                    .map_err(|err| format!("invalid initialize params: {err}"))?
                    .unwrap_or_default();
                let protocol_version = params
                    .protocol_version
                    .unwrap_or_else(|| DEFAULT_PROTOCOL_VERSION.to_string());
                let result = json!({
                    "protocolVersion": protocol_version,
                    "serverInfo": {
                        "name": "burn_synth_mcp",
                        "version": env!("CARGO_PKG_VERSION"),
                    },
                    "capabilities": {
                        "tools": {
                            "listChanged": false
                        }
                    }
                });
                Ok(Some(success_response(request.id, result)))
            }
            "notifications/initialized" => Ok(None),
            "tools/list" => Ok(Some(success_response(
                request.id,
                json!({ "tools": tool_defs() }),
            ))),
            "tools/call" => {
                let params: ToolsCallParams = request
                    .params
                    .ok_or_else(|| "missing tools/call params".to_string())
                    .and_then(|value| {
                        serde_json::from_value(value)
                            .map_err(|err| format!("invalid tools/call params: {err}"))
                    })?;
                let result = self.dispatch_tool_call(params);
                Ok(Some(success_response(request.id, result)))
            }
            "shutdown" => Ok(Some(success_response(request.id, Value::Null))),
            "exit" => {
                self.should_exit = true;
                Ok(None)
            }
            _ => {
                if request.id.is_none() {
                    return Ok(None);
                }
                Ok(Some(error_response(
                    request.id,
                    -32601,
                    format!("method '{}' not found", request.method),
                )))
            }
        }
    }

    fn dispatch_tool_call(&mut self, params: ToolsCallParams) -> Value {
        match params.name.as_str() {
            "image_to_foreground" => {
                let args: Result<ForegroundToolArgs, _> = serde_json::from_value(params.arguments);
                match args {
                    Ok(args) => match self.call_image_to_foreground(args) {
                        Ok(value) => success_tool_result(value),
                        Err(err) => error_tool_result(err),
                    },
                    Err(err) => error_tool_result(format!(
                        "invalid arguments for image_to_foreground: {err}"
                    )),
                }
            }
            "image_to_mesh" => {
                let args: Result<MeshToolArgs, _> = serde_json::from_value(params.arguments);
                match args {
                    Ok(args) => match self.call_image_to_mesh(args) {
                        Ok(value) => success_tool_result(value),
                        Err(err) => error_tool_result(err),
                    },
                    Err(err) => {
                        error_tool_result(format!("invalid arguments for image_to_mesh: {err}"))
                    }
                }
            }
            "image_to_splat" => {
                let args: Result<SplatToolArgs, _> = serde_json::from_value(params.arguments);
                match args {
                    Ok(args) => match self.call_image_to_splat(args) {
                        Ok(value) => success_tool_result(value),
                        Err(err) => error_tool_result(err),
                    },
                    Err(err) => {
                        error_tool_result(format!("invalid arguments for image_to_splat: {err}"))
                    }
                }
            }
            "images_to_assets" => {
                let args: Result<ImagesToAssetsToolArgs, _> =
                    serde_json::from_value(params.arguments);
                match args {
                    Ok(args) => match self.call_images_to_assets(args) {
                        Ok(value) => success_tool_result(value),
                        Err(err) => error_tool_result(err),
                    },
                    Err(err) => {
                        error_tool_result(format!("invalid arguments for images_to_assets: {err}"))
                    }
                }
            }
            "scene_prepare_build" => {
                let args: Result<ScenePrepareBuildArgs, _> =
                    serde_json::from_value(params.arguments);
                match args {
                    Ok(args) => match self.call_scene_prepare_build(args) {
                        Ok(value) => success_tool_result(value),
                        Err(err) => error_tool_result(err),
                    },
                    Err(err) => error_tool_result(format!(
                        "invalid arguments for scene_prepare_build: {err}"
                    )),
                }
            }
            "scene_plan_objects" => {
                let args: Result<ScenePrepareBuildArgs, _> =
                    serde_json::from_value(params.arguments);
                match args {
                    Ok(args) => match self.call_scene_plan_objects(args) {
                        Ok(value) => success_tool_result(value),
                        Err(err) => error_tool_result(err),
                    },
                    Err(err) => error_tool_result(format!(
                        "invalid arguments for scene_plan_objects: {err}"
                    )),
                }
            }
            "scene_generate_object_images" => {
                let args: Result<SceneGenerateObjectImagesArgs, _> =
                    serde_json::from_value(params.arguments);
                match args {
                    Ok(args) => match self.call_scene_generate_object_images(args) {
                        Ok(value) => success_tool_result(value),
                        Err(err) => error_tool_result(err),
                    },
                    Err(err) => error_tool_result(format!(
                        "invalid arguments for scene_generate_object_images: {err}"
                    )),
                }
            }
            "scene_build_from_image" => {
                let args: Result<SceneBuildFromImageArgs, _> =
                    serde_json::from_value(params.arguments);
                match args {
                    Ok(args) => match self.call_scene_build_from_image(args) {
                        Ok(value) => success_tool_result(value),
                        Err(err) => error_tool_result(err),
                    },
                    Err(err) => error_tool_result(format!(
                        "invalid arguments for scene_build_from_image: {err}"
                    )),
                }
            }
            "scene_ground" => {
                let args: Result<SceneGroundToolArgs, _> = serde_json::from_value(params.arguments);
                match args {
                    Ok(args) => match self.call_scene_ground(args) {
                        Ok(value) => success_tool_result(value),
                        Err(err) => error_tool_result(err),
                    },
                    Err(err) => {
                        error_tool_result(format!("invalid arguments for scene_ground: {err}"))
                    }
                }
            }
            "scene_plan_bsn" => {
                let args: Result<ScenePlanBsnArgs, _> = serde_json::from_value(params.arguments);
                match args {
                    Ok(args) => match self.call_scene_plan_bsn(args) {
                        Ok(value) => success_tool_result(value),
                        Err(err) => error_tool_result(err),
                    },
                    Err(err) => {
                        error_tool_result(format!("invalid arguments for scene_plan_bsn: {err}"))
                    }
                }
            }
            "scene_apply_bsn" => {
                let args: Result<SceneApplyBsnArgs, _> = serde_json::from_value(params.arguments);
                match args {
                    Ok(args) => match self.call_scene_apply_bsn(args) {
                        Ok(value) => success_tool_result(value),
                        Err(err) => error_tool_result(err),
                    },
                    Err(err) => {
                        error_tool_result(format!("invalid arguments for scene_apply_bsn: {err}"))
                    }
                }
            }
            "scene_status" => match self.call_scene_status() {
                Ok(value) => success_tool_result(value),
                Err(err) => error_tool_result(err),
            },
            "scene_project_status" => match self.call_scene_project_status() {
                Ok(value) => success_tool_result(value),
                Err(err) => error_tool_result(err),
            },
            "scene_list_assets" => match self.call_scene_list_assets() {
                Ok(value) => success_tool_result(value),
                Err(err) => error_tool_result(err),
            },
            "scene_spawn_cached" => {
                let args: Result<SceneSpawnCachedArgs, _> =
                    serde_json::from_value(params.arguments);
                match args {
                    Ok(args) => match self.call_scene_spawn_cached(args) {
                        Ok(value) => success_tool_result(value),
                        Err(err) => error_tool_result(err),
                    },
                    Err(err) => error_tool_result(format!(
                        "invalid arguments for scene_spawn_cached: {err}"
                    )),
                }
            }
            "scene_spawn_path" => {
                let args: Result<SceneSpawnPathArgs, _> = serde_json::from_value(params.arguments);
                match args {
                    Ok(args) => match self.call_scene_spawn_path(args) {
                        Ok(value) => success_tool_result(value),
                        Err(err) => error_tool_result(err),
                    },
                    Err(err) => {
                        error_tool_result(format!("invalid arguments for scene_spawn_path: {err}"))
                    }
                }
            }
            "scene_delete" => {
                let args: Result<SceneDeleteArgs, _> = serde_json::from_value(params.arguments);
                match args {
                    Ok(args) => match self.call_scene_delete(args) {
                        Ok(value) => success_tool_result(value),
                        Err(err) => error_tool_result(err),
                    },
                    Err(err) => {
                        error_tool_result(format!("invalid arguments for scene_delete: {err}"))
                    }
                }
            }
            "scene_clear" => match self.call_scene_clear() {
                Ok(value) => success_tool_result(value),
                Err(err) => error_tool_result(err),
            },
            "scene_set_camera" => {
                let args: Result<SceneSetCameraArgs, _> = serde_json::from_value(params.arguments);
                match args {
                    Ok(args) => match self.call_scene_set_camera(args) {
                        Ok(value) => success_tool_result(value),
                        Err(err) => error_tool_result(err),
                    },
                    Err(err) => {
                        error_tool_result(format!("invalid arguments for scene_set_camera: {err}"))
                    }
                }
            }
            "scene_save" => match self.send_scene_commands(vec![json!({ "type": "save_cache" })]) {
                Ok(value) => success_tool_result(value),
                Err(err) => error_tool_result(err),
            },
            "scene_capture" => {
                let args: Result<SceneCaptureArgs, _> = serde_json::from_value(params.arguments);
                match args {
                    Ok(args) => match self.call_scene_capture(args) {
                        Ok(value) => success_tool_result(value),
                        Err(err) => error_tool_result(err),
                    },
                    Err(err) => {
                        error_tool_result(format!("invalid arguments for scene_capture: {err}"))
                    }
                }
            }
            "scene_compose_assets" => {
                let args: Result<SceneComposeArgs, _> = serde_json::from_value(params.arguments);
                match args {
                    Ok(args) => match self.call_scene_compose_assets(args) {
                        Ok(value) => success_tool_result(value),
                        Err(err) => error_tool_result(err),
                    },
                    Err(err) => error_tool_result(format!(
                        "invalid arguments for scene_compose_assets: {err}"
                    )),
                }
            }
            "scene_validate_layout" => {
                let args: Result<SceneValidateArgs, _> = serde_json::from_value(params.arguments);
                match args {
                    Ok(args) => match self.call_scene_validate_layout(args) {
                        Ok(value) => success_tool_result(value),
                        Err(err) => error_tool_result(err),
                    },
                    Err(err) => error_tool_result(format!(
                        "invalid arguments for scene_validate_layout: {err}"
                    )),
                }
            }
            other => error_tool_result(format!("unknown tool '{other}'")),
        }
    }

    fn call_image_to_foreground(&mut self, args: ForegroundToolArgs) -> Result<Value, String> {
        let input_path = args.input_image_path;
        if !input_path.exists() {
            return Err(format!(
                "input image does not exist: {}",
                input_path.display()
            ));
        }
        let output_path = args
            .output_image_path
            .unwrap_or_else(|| default_output_path(&input_path, "_foreground", "png"));
        ensure_parent_dir(&output_path).map_err(|err| err.to_string())?;

        let selected_model = args.rmbg_model.unwrap_or(self.config.default_rmbg_model);
        let dry_run = args.dry_run;

        let (width, height) = if dry_run {
            let passthrough = image::open(&input_path)
                .map_err(|err| {
                    format!("failed to open input image {}: {err}", input_path.display())
                })?
                .to_rgba8();
            let dims = passthrough.dimensions();
            passthrough.save(&output_path).map_err(|err| {
                format!(
                    "failed to save foreground image {}: {err}",
                    output_path.display()
                )
            })?;
            dims
        } else {
            let output = self
                .runtime
                .extract_foreground(ForegroundRequest {
                    image: ImageSource::from_path(input_path.clone()),
                    model: Some(selected_model.into()),
                })
                .map_err(|err| err.to_string())?;
            let dims = (output.width, output.height);
            output.image.save(&output_path).map_err(|err| {
                format!(
                    "failed to save foreground image {}: {err}",
                    output_path.display()
                )
            })?;
            dims
        };

        Ok(json!({
            "tool": "image_to_foreground",
            "input_image_path": input_path.display().to_string(),
            "output_image_path": output_path.display().to_string(),
            "width": width,
            "height": height,
            "rmbg_model": selected_model.as_str(),
            "dry_run": dry_run,
        }))
    }

    fn call_image_to_mesh(&mut self, args: MeshToolArgs) -> Result<Value, String> {
        let input_path = args.input_image_path;
        if !input_path.exists() {
            return Err(format!(
                "input image does not exist: {}",
                input_path.display()
            ));
        }
        if let Some(output_format) = args.output_format
            && !matches!(output_format, MeshOutputFormat::Glb)
        {
            return Err(format!(
                "only glb output is supported; requested {}",
                output_format.as_str()
            ));
        }
        let assets = self.call_images_to_assets(ImagesToAssetsToolArgs {
            input_image_paths: vec![input_path],
            output_dir: None,
            output_paths: args.output_mesh_path.map(|path| vec![path]),
            output_format: Some(AssetOutputFormat::Glb),
            rmbg_model: args.rmbg_model,
            synthesis_models: args.synthesis_models,
            backend: args.backend,
            target_faces: args.target_faces,
            batch_size: Some(1),
            batch_vram_mb: None,
            trellis_pbr: None,
            trellis_pbr_texture_size: None,
            promote_to_catalog: false,
            dry_run: args.dry_run,
        })?;
        let item = assets["items"]
            .as_array()
            .and_then(|items| items.first())
            .cloned()
            .ok_or_else(|| "image_to_mesh produced no asset item".to_string())?;
        Ok(json!({
            "tool": "image_to_mesh",
            "input_image_path": item["input_image_path"].clone(),
            "output_mesh_path": item["output_path"].clone(),
            "output_format": "glb",
            "vertices": item["vertices"].clone(),
            "faces": item["faces"].clone(),
            "local_aabb": item["local_aabb"].clone(),
            "target_faces": item["target_faces"].clone(),
            "material": item["material"].clone(),
            "rmbg_model": assets["rmbg_model"].clone(),
            "synthesis_models": assets["synthesis_models"].clone(),
            "backend": assets["backend"].clone(),
            "dry_run": assets["dry_run"].clone(),
        }))
    }

    fn call_image_to_splat(&mut self, args: SplatToolArgs) -> Result<Value, String> {
        let assets = self.call_images_to_assets(ImagesToAssetsToolArgs {
            input_image_paths: vec![args.input_image_path],
            output_dir: None,
            output_paths: args.output_splat_path.map(|path| vec![path]),
            output_format: args.output_format.or(Some(AssetOutputFormat::Splat)),
            rmbg_model: args.rmbg_model,
            synthesis_models: Some(vec![SynthesisModel::Triposplat]),
            backend: args.backend,
            target_faces: None,
            batch_size: Some(1),
            batch_vram_mb: None,
            trellis_pbr: None,
            trellis_pbr_texture_size: None,
            promote_to_catalog: false,
            dry_run: args.dry_run,
        })?;
        let item = assets["items"]
            .as_array()
            .and_then(|items| items.first())
            .cloned()
            .ok_or_else(|| "image_to_splat produced no asset item".to_string())?;
        Ok(json!({
            "tool": "image_to_splat",
            "input_image_path": item["input_image_path"].clone(),
            "output_splat_path": item["output_path"].clone(),
            "output_format": item["output_format"].clone(),
            "gaussians": item["gaussians"].clone(),
            "rmbg_model": assets["rmbg_model"].clone(),
            "synthesis_models": assets["synthesis_models"].clone(),
            "backend": assets["backend"].clone(),
            "dry_run": assets["dry_run"].clone(),
        }))
    }

    pub(crate) fn call_images_to_assets(
        &mut self,
        args: ImagesToAssetsToolArgs,
    ) -> Result<Value, String> {
        if args.input_image_paths.is_empty() {
            return Err("input_image_paths must not be empty".to_string());
        }
        for input in &args.input_image_paths {
            if !input.exists() {
                return Err(format!("input image does not exist: {}", input.display()));
            }
        }
        if let Some(output_paths) = args.output_paths.as_ref()
            && output_paths.len() != args.input_image_paths.len()
        {
            return Err(format!(
                "output_paths length ({}) must match input_image_paths length ({})",
                output_paths.len(),
                args.input_image_paths.len()
            ));
        }

        let selected_rmbg = args.rmbg_model.unwrap_or(self.config.default_rmbg_model);
        let selected_backend = args.backend.unwrap_or(self.config.default_backend);
        let selected_synthesis_models = args
            .synthesis_models
            .map(sanitize_synthesis_models)
            .unwrap_or_else(|| self.config.default_synthesis_models.clone());
        let policy = RuntimeBatchPolicy {
            max_items: args.batch_size.or(self.config.batch_size),
            vram_budget_mb: args.batch_vram_mb.or(self.config.batch_vram_mb),
            ..RuntimeBatchPolicy::default()
        };
        let previous_trellis_pbr_enabled = self.runtime.config().trellis_pbr_enabled;
        let previous_trellis_pbr_texture_size = self.runtime.config().trellis_pbr_texture_size;
        let previous_target_faces = self.runtime.config().target_faces;
        let previous_progress = self.runtime.config().progress.clone();
        let previous_progress_callback = previous_progress.callback().cloned();
        let runtime_progress_events = Arc::new(Mutex::new(Vec::<RuntimeProgressEvent>::new()));
        let runtime_progress_events_for_callback = Arc::clone(&runtime_progress_events);
        let effective_trellis_pbr_enabled =
            args.trellis_pbr.unwrap_or(previous_trellis_pbr_enabled);
        let effective_trellis_pbr_texture_size = args
            .trellis_pbr_texture_size
            .or(previous_trellis_pbr_texture_size);
        let effective_target_faces = match args.target_faces {
            Some(0) => None,
            Some(value) => Some(value),
            None => previous_target_faces,
        };
        {
            let config = self.runtime.config_mut();
            config.trellis_pbr_enabled = effective_trellis_pbr_enabled;
            config.trellis_pbr_texture_size = effective_trellis_pbr_texture_size;
            config.target_faces = effective_target_faces;
            config.progress = RuntimeProgressObserver::with_callback(
                match previous_progress.verbosity {
                    ProgressVerbosity::Steps => ProgressVerbosity::Steps,
                    ProgressVerbosity::Stages | ProgressVerbosity::Off => ProgressVerbosity::Stages,
                },
                previous_progress.step_interval,
                Arc::new(move |event| {
                    if let Some(callback) = previous_progress_callback.as_ref() {
                        callback(event);
                    }
                    if let Ok(mut events) = runtime_progress_events_for_callback.lock() {
                        events.push(event.clone());
                    }
                }),
            );
        }

        let batch_result = self.runtime.synthesize_assets_batch(AssetBatchRequest {
            items: args
                .input_image_paths
                .iter()
                .enumerate()
                .map(|(index, input)| {
                    AssetBatchItem::new(
                        format!("asset_{index}"),
                        ImageSource::from_path(input.clone()),
                    )
                })
                .collect(),
            foreground_model: Some(selected_rmbg.into()),
            synthesis_models: Some(
                selected_synthesis_models
                    .iter()
                    .copied()
                    .map(Into::into)
                    .collect(),
            ),
            backend: Some(selected_backend.into()),
            dry_run: args.dry_run,
            policy,
        });
        let captured_runtime_progress_events = runtime_progress_events
            .lock()
            .map(|events| events.clone())
            .unwrap_or_default();
        {
            let config = self.runtime.config_mut();
            config.trellis_pbr_enabled = previous_trellis_pbr_enabled;
            config.trellis_pbr_texture_size = previous_trellis_pbr_texture_size;
            config.target_faces = previous_target_faces;
            config.progress = previous_progress;
        }
        let batch = batch_result.map_err(|err| err.to_string())?;
        let runtime_progress_events_json = captured_runtime_progress_events
            .iter()
            .map(runtime_progress_event_json)
            .collect::<Vec<_>>();

        let mut items = Vec::with_capacity(batch.items.len());
        let mut catalog_cache = if args.promote_to_catalog {
            Some(self.open_catalog_cache()?)
        } else {
            None
        };
        for batch_item in batch.items {
            let input_path = args
                .input_image_paths
                .get(batch_item.item_index)
                .ok_or_else(|| {
                    format!(
                        "asset batch item index {} out of range for {} input images",
                        batch_item.item_index,
                        args.input_image_paths.len()
                    )
                })?;
            let output = batch_item.output.map_err(|err| err.to_string())?;
            let item = write_asset_output(
                input_path,
                args.output_dir.as_deref(),
                args.output_paths
                    .as_ref()
                    .and_then(|paths| paths.get(batch_item.item_index).cloned()),
                args.output_format.unwrap_or(AssetOutputFormat::Auto),
                output.asset,
                effective_target_faces,
                catalog_cache.as_mut(),
            )?;
            items.push(json!({
                "id": batch_item.id,
                "input_image_path": input_path.display().to_string(),
                "chunk_index": batch_item.chunk_index,
                "item_index": batch_item.item_index,
                "elapsed_ms": batch_item.elapsed_ms,
                "foreground_model": runtime_foreground_model_str(output.foreground_model),
                "synthesis_backend": runtime_synthesis_model_str(output.synthesis_backend),
                "backend": runtime_backend_str(output.backend),
                "output_path": item.output_path.display().to_string(),
                "output_format": item.output_format.as_str(),
                "asset_kind": item.asset_kind,
                "vertices": item.vertices,
                "faces": item.faces,
                "gaussians": item.gaussians,
                "local_aabb": item.local_aabb,
                "target_faces": effective_target_faces,
                "material": item.material,
                "mesh_quality": item.mesh_quality,
                "mesh_quality_failures": item.mesh_quality_failures,
                "cache_key": item.catalog_entry.as_ref().map(|entry| entry.cache_key.clone()),
                "catalog_entry": item.catalog_entry,
            }));
        }

        Ok(json!({
            "tool": "images_to_assets",
            "items": items,
            "stats": {
                "total_items": batch.stats.total_items,
                "chunk_size": batch.stats.chunk_size,
                "chunks": batch.stats.chunks,
                "execution_mode": batch.stats.execution_mode.as_str(),
                "vram_budget_mb": batch.stats.vram_budget_mb,
                "estimated_item_mb": batch.stats.estimated_item_mb,
                "elapsed_ms": batch.stats.elapsed_ms,
            },
            "rmbg_model": selected_rmbg.as_str(),
            "synthesis_models": selected_synthesis_models.iter().map(|m| m.as_str()).collect::<Vec<_>>(),
            "backend": selected_backend.as_str(),
            "trellis_pbr_enabled": effective_trellis_pbr_enabled,
            "trellis_pbr_backend": if effective_trellis_pbr_enabled { Some("rust-ovoxel") } else { None },
            "trellis_pbr_texture_size": effective_trellis_pbr_texture_size,
            "target_faces": effective_target_faces,
            "runtime_progress_events": runtime_progress_events_json,
            "promote_to_catalog": args.promote_to_catalog,
            "dry_run": args.dry_run,
        }))
    }

    fn call_scene_prepare_build(&self, args: ScenePrepareBuildArgs) -> Result<Value, String> {
        let config = self.scene_build_config(args)?;
        let provider = NoopSceneProvider;
        let mut pipeline = ScenePipeline::new(config, provider);
        let preparation = pipeline
            .prepare_openai_inputs()
            .map_err(|err| err.to_string())?;
        serde_json::to_value(preparation).map_err(|err| err.to_string())
    }

    fn call_scene_plan_objects(&self, args: ScenePrepareBuildArgs) -> Result<Value, String> {
        let config = self.scene_build_config(args)?;
        let provider = self.openai_provider()?;
        let pipeline = ScenePipeline::new(config, provider);
        let manifest = pipeline.plan_objects().map_err(|err| err.to_string())?;
        serde_json::to_value(manifest).map_err(|err| err.to_string())
    }

    fn call_scene_generate_object_images(
        &self,
        args: SceneGenerateObjectImagesArgs,
    ) -> Result<Value, String> {
        let prepare_args = ScenePrepareBuildArgs {
            source_scene_path: args.source_scene_path,
            object_reference_image_path: args.object_reference_image_path,
            output_dir: args.output_dir,
            candidate_count: args.candidate_count,
            quality_profile: args.quality_profile,
            allow_catalog_reuse: false,
            allowed_categories: None,
            denied_categories: None,
        };
        let config = self.scene_build_config(prepare_args)?;
        let provider = self.openai_provider()?;
        let pipeline = ScenePipeline::new(config, provider);
        let requests = pipeline
            .prepare_object_image_requests(&args.manifest)
            .map_err(|err| err.to_string())?;
        let candidates = pipeline
            .generate_object_candidates(&requests)
            .map_err(|err| err.to_string())?;
        Ok(json!({
            "tool": "scene_generate_object_images",
            "requests": requests,
            "candidates": candidates,
        }))
    }

    pub(crate) fn call_scene_build_from_image(
        &mut self,
        args: SceneBuildFromImageArgs,
    ) -> Result<Value, String> {
        let mut noop = |_| {};
        self.call_scene_build_from_image_with_progress(args, &mut noop)
    }

    pub(crate) fn call_scene_build_from_image_with_progress<F>(
        &mut self,
        args: SceneBuildFromImageArgs,
        progress: &mut F,
    ) -> Result<Value, String>
    where
        F: FnMut(SceneBuildProgressEvent),
    {
        let mut reporter = SceneBuildProgressReporter::new(progress, args.write_artifacts);
        let result = self.call_scene_build_from_image_inner(args, &mut reporter);
        if let Err(err) = result.as_ref() {
            reporter.emit(
                "scene_build",
                SceneBuildProgressPhase::Failed,
                SceneBuildExecutionKind::Mixed,
                format!("scene build failed: {err}"),
                json!({ "error": err }),
            );
        }
        result
    }

    fn call_scene_build_from_image_inner<F>(
        &mut self,
        args: SceneBuildFromImageArgs,
        progress: &mut SceneBuildProgressReporter<'_, F>,
    ) -> Result<Value, String>
    where
        F: FnMut(SceneBuildProgressEvent),
    {
        validate_scene_pose_fit_mode(args.pose_fit)?;
        let asset_lift_policy = scene_asset_lift_policy(&args)?;
        let e2e_started = Instant::now();
        let mut stage_report = Vec::new();
        let prepare_args = ScenePrepareBuildArgs {
            source_scene_path: args.source_scene_path.clone(),
            object_reference_image_path: args.object_reference_image_path.clone(),
            output_dir: args.output_dir.clone(),
            candidate_count: args.candidate_count,
            quality_profile: args.quality_profile,
            allow_catalog_reuse: args.allow_catalog_reuse,
            allowed_categories: args.allowed_categories.clone(),
            denied_categories: args.denied_categories.clone(),
        };
        let stage_started = Instant::now();
        progress.emit(
            "prepare_openai_inputs",
            SceneBuildProgressPhase::Started,
            SceneBuildExecutionKind::FileIo,
            format!(
                "preparing scene build for {}",
                args.source_scene_path.display()
            ),
            json!({
                "source_scene_path": args.source_scene_path.display().to_string(),
                "lift_assets": args.lift_assets,
                "feedback": args.feedback,
            }),
        );
        let config = self.scene_build_config(prepare_args)?;
        let category_filter = config.category_filter.clone();
        let output_dir = config.output_dir.clone();
        progress.set_output_dir(output_dir.clone());
        let candidate_policy = scene_object_image_generation_policy(&args, config.candidate_count);
        let provider = self.openai_provider()?;
        let mut pipeline = ScenePipeline::new(config, provider);
        let preparation = pipeline
            .prepare_openai_inputs()
            .map_err(|err| err.to_string())?;
        record_stage(&mut stage_report, "prepare_openai_inputs", stage_started);
        progress.emit_with_items(
            "prepare_openai_inputs",
            SceneBuildProgressPhase::Completed,
            SceneBuildExecutionKind::FileIo,
            "scene build inputs prepared",
            None,
            None,
            Some(output_dir.clone()),
            json!({ "output_dir": output_dir.display().to_string() }),
        );
        let stage_started = Instant::now();
        progress.emit(
            "plan_objects",
            SceneBuildProgressPhase::Waiting,
            SceneBuildExecutionKind::Network,
            "requesting object plan from OpenAI",
            json!({
                "reasoning_model": self.config.openai_reasoning_model,
                "execution": "network"
            }),
        );
        let manifest_initial = pipeline.plan_objects().map_err(|err| err.to_string())?;
        let (filtered_manifest, _, category_filter_report) =
            apply_scene_category_filter(&manifest_initial, None, &category_filter);
        if filtered_manifest.objects.is_empty() {
            return Err(format!(
                "scene category filter removed all planned objects; allowed={:?} denied={:?}",
                category_filter.allow, category_filter.deny
            ));
        }
        let mut category_filter_reports = vec![category_filter_report.clone()];
        let mut manifest = filtered_manifest;
        record_stage(&mut stage_report, "plan_objects", stage_started);
        progress.emit_with_items(
            "plan_objects",
            SceneBuildProgressPhase::Completed,
            SceneBuildExecutionKind::Network,
            format!("planned {} object(s)", manifest.objects.len()),
            None,
            Some(manifest.objects.len()),
            Some(output_dir.join("manifest_initial.json")),
            json!({ "objects": manifest.objects.len() }),
        );
        if args.write_artifacts {
            write_json_file(&output_dir.join("manifest_initial.json"), &manifest_initial)
                .map_err(|err| err.to_string())?;
            write_json_file(
                &output_dir.join("category_filter_report.json"),
                &category_filter_report,
            )
            .map_err(|err| err.to_string())?;
            write_json_file(&output_dir.join("manifest.json"), &manifest)
                .map_err(|err| err.to_string())?;
        }
        let segmentation_provider = args
            .segmentation_provider
            .unwrap_or(self.config.scene_segmentation_provider);
        let segmentation_precision = args
            .segmentation_precision
            .unwrap_or(self.config.scene_segmentation_precision);
        let segmentation_quantization = args
            .segmentation_quantization
            .unwrap_or(self.config.scene_segmentation_quantization);
        let mut pre_generation_grounding_source = "not_requested".to_string();
        let mut pre_generation_grounding_evidence: Option<SceneGroundingEvidence> = None;
        let mut pre_generation_locate_anything_report: Option<LocateAnythingGroundingReport> = None;
        let mut pre_generation_segmentation_report: Option<SegmentationGroundingReport> = None;
        let mut pre_generation_depth_report: Option<DepthProGroundingReport> = None;
        let mut pre_generation_ground_calibration_report: Option<SceneGroundCalibrationReport> =
            None;
        if args.composition_mode == SceneCompositionMode::CvGrounded {
            let (mut evidence, mut source) = if args.locator == SceneLocatorProvider::LocateAnything
            {
                let stage_started = Instant::now();
                let backend = args
                    .locate_anything_backend
                    .unwrap_or(self.config.locate_anything_backend);
                progress.emit(
                    "pre_generation_locate_anything_grounding",
                    SceneBuildProgressPhase::Started,
                    SceneBuildExecutionKind::Gpu,
                    "running LocateAnything before object crops",
                    json!({
                        "locator": args.locator,
                        "backend": backend,
                        "purpose": "object cardinality and crop bboxes before gpt-image-2",
                    }),
                );
                let (evidence, report) = self.locate_anything_grounding_evidence_with_report(
                    backend,
                    &manifest,
                    &args.source_scene_path,
                    &output_dir,
                    &category_filter,
                )?;
                let source = "locate_anything_burn_native_pre_generation".to_string();
                pre_generation_locate_anything_report = Some(report);
                record_stage(
                    &mut stage_report,
                    "pre_generation_locate_anything_grounding",
                    stage_started,
                );
                progress.emit_with_items(
                    "pre_generation_locate_anything_grounding",
                    SceneBuildProgressPhase::Completed,
                    SceneBuildExecutionKind::Gpu,
                    "LocateAnything crop grounding complete",
                    None,
                    pre_generation_grounding_evidence
                        .as_ref()
                        .map(|evidence| evidence.objects.len()),
                    pre_generation_locate_anything_report
                        .as_ref()
                        .map(|report| report.overlay_path.clone()),
                    json!({
                        "grounding_source": source,
                        "detections": evidence.detections.len(),
                        "objects": evidence.objects.len(),
                        "manifest_artifact": output_dir.join("manifest_grounded_for_crops.json"),
                    }),
                );
                (evidence, source)
            } else {
                (
                    manifest_grounding_evidence(&manifest),
                    "manifest_fallback_pre_generation".to_string(),
                )
            };

            let type_aware_categories = scene_type_aware_categories(
                args.instance_generation,
                args.type_aware_categories.as_deref(),
            );
            if args.locator == SceneLocatorProvider::LocateAnything
                && scene_instance_generation_type_aware(args.instance_generation)
                && type_aware_categories.contains(&SceneObjectCategory::Chair)
            {
                match pipeline.prepare_chair_type_grouping_request(&manifest, &evidence) {
                    Ok(Some(request)) => {
                        let stage_started = Instant::now();
                        let request_path = output_dir.join("chair_type_grouping_request.json");
                        let response_path = output_dir.join("chair_type_grouping_response.json");
                        let report_path = output_dir.join("chair_type_grouping_report.json");
                        if args.write_artifacts {
                            write_json_file(&request_path, &request)
                                .map_err(|err| err.to_string())?;
                        }
                        progress.emit_with_items(
                            "chair_type_grouping",
                            SceneBuildProgressPhase::Started,
                            SceneBuildExecutionKind::Network,
                            "classifying repeated chair detections into reusable asset types",
                            None,
                            Some(request.items.len()),
                            Some(request_path.clone()),
                            json!({
                                "chair_detection_count": request.items.len(),
                                "crop_count": request.crop_image_paths.len(),
                                "purpose": "split visually distinct chair types before shared image generation and 3d lift",
                            }),
                        );
                        match pipeline.classify_chair_types(&request) {
                            Ok(response) => {
                                let report = chair_type_grouping_report(&request, &response);
                                let (next_manifest, next_evidence) = apply_chair_type_groups(
                                    &manifest, &evidence, &request, &response,
                                )
                                .map_err(|err| err.to_string())?;
                                manifest = next_manifest;
                                evidence = next_evidence;
                                source = format!("{source}+chair_type_grouping");
                                record_stage(
                                    &mut stage_report,
                                    "chair_type_grouping",
                                    stage_started,
                                );
                                if args.write_artifacts {
                                    write_json_file(&response_path, &response)
                                        .map_err(|err| err.to_string())?;
                                    write_json_file(&report_path, &report)
                                        .map_err(|err| err.to_string())?;
                                    write_json_file(
                                        &output_dir.join("manifest_after_chair_type_grouping.json"),
                                        &manifest,
                                    )
                                    .map_err(|err| err.to_string())?;
                                    write_json_file(
                                        &output_dir
                                            .join("grounding_after_chair_type_grouping.json"),
                                        &evidence,
                                    )
                                    .map_err(|err| err.to_string())?;
                                }
                                progress.emit_with_items(
                                    "chair_type_grouping",
                                    SceneBuildProgressPhase::Completed,
                                    SceneBuildExecutionKind::Network,
                                    "chair type grouping complete",
                                    None,
                                    report
                                        .get("group_count")
                                        .and_then(Value::as_u64)
                                        .map(|value| value as usize),
                                    Some(report_path),
                                    json!({
                                        "group_count": report
                                            .get("group_count")
                                            .and_then(Value::as_u64),
                                        "manifest_objects": manifest.objects.len(),
                                        "source": source.clone(),
                                    }),
                                );
                            }
                            Err(err) => {
                                record_stage(
                                    &mut stage_report,
                                    "chair_type_grouping",
                                    stage_started,
                                );
                                let error_path = output_dir.join("chair_type_grouping_error.json");
                                if args.write_artifacts {
                                    write_json_file(
                                        &error_path,
                                        &json!({
                                            "stage": "chair_type_grouping",
                                            "error": err.to_string(),
                                            "message": "fine-grained instance generation failed",
                                        }),
                                    )
                                    .map_err(|write_err| write_err.to_string())?;
                                }
                                progress.emit_with_items(
                                    "chair_type_grouping",
                                    SceneBuildProgressPhase::Failed,
                                    SceneBuildExecutionKind::Network,
                                    "chair type grouping failed",
                                    None,
                                    Some(request.items.len()),
                                    Some(error_path),
                                    json!({
                                        "error": err.to_string(),
                                        "fallback": Value::Null,
                                    }),
                                );
                                return Err(format!("chair type grouping failed: {err}"));
                            }
                        }
                    }
                    Ok(None) => {
                        progress.emit(
                            "chair_type_grouping",
                            SceneBuildProgressPhase::Completed,
                            SceneBuildExecutionKind::Cpu,
                            "chair type grouping not needed",
                            json!({
                                "reason": "zero_or_one_chair_detection",
                            }),
                        );
                    }
                    Err(err) => {
                        return Err(err.to_string());
                    }
                }
            } else if args.locator == SceneLocatorProvider::LocateAnything {
                progress.emit(
                    "chair_type_grouping",
                    SceneBuildProgressPhase::Completed,
                    SceneBuildExecutionKind::Cpu,
                    "same-category instances will share representative assets",
                    json!({
                        "instance_generation": args.instance_generation,
                        "type_aware_categories": type_aware_categories
                            .iter()
                            .map(|category| category.as_str())
                            .collect::<Vec<_>>(),
                        "reason": "chair_type_grouping_not_selected",
                    }),
                );
            }

            let (filtered_manifest, filtered_evidence, post_grounding_filter_report) =
                apply_scene_category_filter(&manifest, Some(&evidence), &category_filter);
            if filtered_manifest.objects.is_empty() {
                return Err(format!(
                    "scene category filter removed all objects after grounding; allowed={:?} denied={:?}",
                    category_filter.allow, category_filter.deny
                ));
            }
            manifest = filtered_manifest;
            evidence = filtered_evidence.unwrap_or(evidence);
            if args.write_artifacts {
                write_json_file(
                    &output_dir.join("category_filter_after_grounding_report.json"),
                    &post_grounding_filter_report,
                )
                .map_err(|err| err.to_string())?;
            }
            category_filter_reports.push(post_grounding_filter_report);

            if segmentation_provider != SceneSegmentationProvider::None
                && evidence.segmentation.is_none()
            {
                let stage_started = Instant::now();
                progress.emit(
                    "pre_generation_segmentation_grounding",
                    SceneBuildProgressPhase::Started,
                    if segmentation_provider == SceneSegmentationProvider::BboxPrompt {
                        SceneBuildExecutionKind::Cpu
                    } else {
                        SceneBuildExecutionKind::Gpu
                    },
                    "running SAM/object-query masks before object crops",
                    json!({
                        "segmentation_provider": segmentation_provider,
                        "segmentation_precision": segmentation_precision,
                        "segmentation_quantization": segmentation_quantization,
                        "objects": evidence.objects.len(),
                        "purpose": "mask-refined crop boxes and visible-pixel depth selection",
                    }),
                );
                pre_generation_segmentation_report = self.segmentation_grounding_evidence(
                    segmentation_provider,
                    Some(segmentation_precision),
                    Some(segmentation_quantization),
                    &mut evidence,
                    &args.source_scene_path,
                    &output_dir,
                )?;
                record_stage(
                    &mut stage_report,
                    "pre_generation_segmentation_grounding",
                    stage_started,
                );
                progress.emit_with_items(
                    "pre_generation_segmentation_grounding",
                    SceneBuildProgressPhase::Completed,
                    if segmentation_provider == SceneSegmentationProvider::BboxPrompt {
                        SceneBuildExecutionKind::Cpu
                    } else {
                        SceneBuildExecutionKind::Gpu
                    },
                    "pre-generation mask grounding complete",
                    None,
                    pre_generation_segmentation_report
                        .as_ref()
                        .map(|report| report.mask_count),
                    pre_generation_segmentation_report
                        .as_ref()
                        .map(|report| report.overlay_path.clone()),
                    json!({
                        "mask_count": pre_generation_segmentation_report
                            .as_ref()
                            .map(|report| report.mask_count),
                        "runtime_cache_hit": pre_generation_segmentation_report
                            .as_ref()
                            .map(|report| report.runtime_cache_hit),
                    }),
                );
            }

            if args.depth_provider == SceneDepthProvider::DepthPro && evidence.depth.is_none() {
                let stage_started = Instant::now();
                progress.emit(
                    "pre_generation_depth_pro_grounding",
                    SceneBuildProgressPhase::Started,
                    SceneBuildExecutionKind::Gpu,
                    "running DepthPro before object crops",
                    json!({
                        "depth_provider": args.depth_provider,
                        "cache_dir": self.config.depth_cache_dir.clone(),
                        "precision": self.config.depth_precision,
                        "mask_count": evidence
                            .segmentation
                            .as_ref()
                            .and_then(|segmentation| segmentation.mask_count),
                        "purpose": "camera/floor calibration and visible-surface depth stats before composition",
                    }),
                );
                let depth_report = self.depth_pro_grounding_evidence(
                    &mut evidence,
                    &args.source_scene_path,
                    &output_dir,
                )?;
                record_stage(
                    &mut stage_report,
                    "pre_generation_depth_pro_grounding",
                    stage_started,
                );
                progress.emit_with_items(
                    "pre_generation_depth_pro_grounding",
                    SceneBuildProgressPhase::Completed,
                    SceneBuildExecutionKind::Gpu,
                    "pre-generation DepthPro grounding complete",
                    None,
                    Some(evidence.objects.len()),
                    Some(output_dir.join("depth_pro").join("depth_evidence.json")),
                    json!({
                        "has_depth": evidence.depth.is_some(),
                        "runtime_cache_hit": depth_report.runtime_cache_hit,
                        "load_ms": depth_report.load_ms,
                        "infer_ms": depth_report.infer_ms,
                        "depth_map_path": depth_report.depth_map_path.display().to_string(),
                        "depth_map_metadata_path": depth_report
                            .depth_map_metadata_path
                            .display()
                            .to_string(),
                        "floor_sample_count": evidence
                            .depth
                            .as_ref()
                            .and_then(|depth| depth.floor_sample_count),
                    }),
                );
                pre_generation_depth_report = Some(depth_report);
            }

            if args.ground_calibration == SceneGroundCalibrationMode::Gpt {
                let stage_started = Instant::now();
                progress.emit(
                    "ground_calibration_gpt",
                    SceneBuildProgressPhase::Started,
                    SceneBuildExecutionKind::Network,
                    "calibrating source camera/floor with GPT",
                    json!({
                        "ground_calibration": args.ground_calibration,
                        "depth_provider": args.depth_provider,
                        "has_depth": evidence.depth.is_some(),
                        "purpose": "replace noisy depth-floor heuristic with typed camera/floor evidence before object crops and composition",
                    }),
                );
                let report = apply_gpt_ground_calibration(
                    &pipeline,
                    &mut manifest,
                    &mut evidence,
                    &output_dir,
                    args.write_artifacts,
                    &[],
                )?;
                source = format!("{source}+gpt_ground_calibration");
                record_stage(&mut stage_report, "ground_calibration_gpt", stage_started);
                progress.emit(
                    "ground_calibration_gpt",
                    SceneBuildProgressPhase::Completed,
                    SceneBuildExecutionKind::Network,
                    "GPT camera/floor calibration complete",
                    json!({
                        "camera_height_m": -report.applied_floor.distance_m,
                        "vertical_fov_degrees": report.applied_camera.vertical_fov_degrees,
                        "floor_confidence": report.applied_floor.confidence,
                        "floor_residual_m": report.applied_floor.residual_m,
                    }),
                );
                pre_generation_ground_calibration_report = Some(report);
            } else {
                let stage_started = Instant::now();
                let report = deterministic_ground_calibration_report(
                    &manifest,
                    &evidence,
                    args.ground_calibration,
                );
                let stage_name = match args.ground_calibration {
                    SceneGroundCalibrationMode::DepthFirst => "ground_calibration_depth_first",
                    SceneGroundCalibrationMode::DepthHeuristic => {
                        "ground_calibration_depth_heuristic"
                    }
                    SceneGroundCalibrationMode::Gpt => "ground_calibration_gpt",
                };
                if args.write_artifacts {
                    write_json_file(
                        &output_dir.join(format!("{stage_name}_report.json")),
                        &report,
                    )
                    .map_err(|err| err.to_string())?;
                }
                record_stage(&mut stage_report, stage_name, stage_started);
                progress.emit(
                    "ground_calibration",
                    SceneBuildProgressPhase::Completed,
                    SceneBuildExecutionKind::Cpu,
                    "using deterministic depth floor/camera calibration",
                    json!({
                        "ground_calibration": args.ground_calibration,
                        "has_depth": evidence.depth.is_some(),
                        "camera_height_m": -report.applied_floor.distance_m,
                        "vertical_fov_degrees": report.applied_camera.vertical_fov_degrees,
                        "floor_confidence": report.applied_floor.confidence,
                        "floor_residual_m": report.applied_floor.residual_m,
                    }),
                );
                pre_generation_ground_calibration_report = Some(report);
            }

            manifest = manifest_with_grounding_evidence(&manifest, &evidence);
            pre_generation_grounding_source = source;
            pre_generation_grounding_evidence = Some(evidence);
            if args.write_artifacts {
                write_json_file(&output_dir.join("manifest.json"), &manifest)
                    .map_err(|err| err.to_string())?;
                write_json_file(
                    &output_dir.join("manifest_grounded_for_crops.json"),
                    &manifest,
                )
                .map_err(|err| err.to_string())?;
                if let Some(evidence) = pre_generation_grounding_evidence.as_ref() {
                    write_json_file(
                        &output_dir.join("pre_generation_grounding_evidence.json"),
                        evidence,
                    )
                    .map_err(|err| err.to_string())?;
                }
                if let Some(report) = pre_generation_locate_anything_report.as_ref() {
                    write_json_file(
                        &output_dir.join("pre_generation_locate_anything_report.json"),
                        report,
                    )
                    .map_err(|err| err.to_string())?;
                }
                if let Some(report) = pre_generation_segmentation_report.as_ref() {
                    write_json_file(
                        &output_dir.join("pre_generation_segmentation_report.json"),
                        report,
                    )
                    .map_err(|err| err.to_string())?;
                }
                if let Some(report) = pre_generation_depth_report.as_ref() {
                    write_json_file(&output_dir.join("pre_generation_depth_report.json"), report)
                        .map_err(|err| err.to_string())?;
                }
                if let Some(report) = pre_generation_ground_calibration_report.as_ref() {
                    write_json_file(
                        &output_dir.join("pre_generation_ground_calibration_report.json"),
                        report,
                    )
                    .map_err(|err| err.to_string())?;
                }
            }
        }
        let stage_started = Instant::now();
        progress.emit(
            "prepare_object_image_requests",
            SceneBuildProgressPhase::Started,
            SceneBuildExecutionKind::Cpu,
            "preparing isolated object image prompts and crops",
            json!({ "objects": manifest.objects.len() }),
        );
        let requests = pipeline
            .prepare_object_image_requests(&manifest)
            .map_err(|err| err.to_string())?;
        record_stage(
            &mut stage_report,
            "prepare_object_image_requests",
            stage_started,
        );
        progress.emit_with_items(
            "prepare_object_image_requests",
            SceneBuildProgressPhase::Completed,
            SceneBuildExecutionKind::Cpu,
            format!("prepared {} object image request(s)", requests.len()),
            None,
            Some(requests.len()),
            Some(output_dir.join("object_image_requests.json")),
            json!({ "requests": requests.len() }),
        );
        if args.write_artifacts {
            write_json_file(&output_dir.join("object_image_requests.json"), &requests)
                .map_err(|err| err.to_string())?;
        }
        let stage_started = Instant::now();
        progress.emit_with_items(
            "generate_object_candidates",
            SceneBuildProgressPhase::Waiting,
            SceneBuildExecutionKind::Network,
            format!(
                "generating object images: {} request(s), {} candidate(s) per attempt",
                requests.len(),
                candidate_policy.candidates_per_attempt
            ),
            None,
            Some(requests.len()),
            None,
            json!({
                "requests": requests.len(),
                "max_attempts_per_object": candidate_policy.max_attempts_per_object,
                "candidates_per_attempt": candidate_policy.candidates_per_attempt,
                "parallel_requests": scene_object_image_request_parallelism(requests.len()),
                "min_score": candidate_policy.min_score,
                "execution": "network"
            }),
        );
        let candidate_report = pipeline
            .generate_object_candidates_with_policy_parallel(
                &requests,
                candidate_policy,
                scene_object_image_request_parallelism(requests.len()),
            )
            .map_err(|err| err.to_string())?;
        record_stage(
            &mut stage_report,
            "generate_object_candidates",
            stage_started,
        );
        progress.emit_with_items(
            "generate_object_candidates",
            SceneBuildProgressPhase::Completed,
            SceneBuildExecutionKind::Network,
            format!(
                "generated {} candidate image(s), selected {}",
                candidate_report.candidates.len(),
                candidate_report.selected_candidates.len()
            ),
            None,
            Some(candidate_report.candidates.len()),
            Some(output_dir.join("candidate_generation.json")),
            json!({
                "candidates": candidate_report.candidates.len(),
                "selected": candidate_report.selected_candidates.len(),
                "rejected": candidate_report.rejected_objects.len(),
            }),
        );
        let mut selected = if candidate_report.rejected_objects.is_empty() {
            candidate_report.selected_candidates.clone()
        } else {
            Vec::new()
        };
        let mut selected_values = selected_candidates_to_values_with_requests(&selected, &requests);
        let scene_placement_pipeline =
            scene_placement_pipeline_plan(ScenePlacementPipelineSelection {
                entry_point: ScenePlacementEntryPoint::SceneBuild,
                lift_assets: args.lift_assets,
                composition_mode: args.composition_mode,
                pose_fit: args.pose_fit,
                canonical_pose: args.canonical_pose,
                scale_policy: args.scale_policy,
                ground_calibration: args.ground_calibration,
                instance_generation: args.instance_generation,
                depth_provider: args.depth_provider,
                locator: args.locator,
                segmentation_provider,
                feedback: args.feedback,
                feedback_iters: args.feedback_iters,
                feedback_rotation_selector: args.feedback_rotation_selector,
                feedback_rubric_scorer: args.feedback_rubric_scorer,
                rotation_fit: args.rotation_fit,
                object_pose_refinement: args.object_pose_refinement,
                object_pose_refinement_set: args.object_pose_refinement_set,
                final_yaw_refinement: args.final_yaw_refinement,
                final_yaw_refinement_set: args.final_yaw_refinement_set,
                max_pose_candidates: args.max_pose_candidates,
            });

        let mut response = json!({
            "tool": "scene_build_from_image",
            "preparation": preparation,
            "provider_metadata": pipeline.provider_metadata(),
            "manifest_initial": manifest_initial.clone(),
            "category_filter": category_filter.clone(),
            "category_filter_reports": category_filter_reports.clone(),
            "manifest_grounded_for_crops": manifest.clone(),
            "pre_generation_grounding_source": pre_generation_grounding_source.clone(),
            "pre_generation_grounding_evidence": pre_generation_grounding_evidence.clone(),
            "pre_generation_locate_anything_report": pre_generation_locate_anything_report.clone(),
            "pre_generation_segmentation_report": pre_generation_segmentation_report.clone(),
            "pre_generation_depth_report": pre_generation_depth_report.clone(),
            "pre_generation_ground_calibration_report": pre_generation_ground_calibration_report.clone(),
            "manifest": manifest.clone(),
            "object_image_requests": requests,
            "candidate_generation": candidate_report.clone(),
            "candidates": candidate_report.candidates.clone(),
            "selected_candidates": selected_values.clone(),
            "lift_assets": args.lift_assets,
            "scene_asset_lift_policy": asset_lift_policy.to_json(),
            "ground_calibration": args.ground_calibration,
            "instance_generation": args.instance_generation,
            "scene_placement_pipeline": scene_placement_pipeline.to_json(),
        });
        if args.write_artifacts {
            write_scene_build_artifacts(&output_dir, &response)?;
        }
        if !candidate_report.rejected_objects.is_empty() {
            response["stage_report"] = json!(stage_report);
            attach_scene_token_usage(&mut response);
            response["e2e_summary"] = scene_build_summary(&response, e2e_started.elapsed());
            if args.write_artifacts {
                write_scene_build_artifacts(&output_dir, &response)?;
            }
            let message = candidate_report
                .rejected_objects
                .first()
                .map(|rejection| rejection.message.clone())
                .unwrap_or_else(|| "scene candidate generation failed guardrails".to_string());
            return Err(message);
        }
        if !args.lift_assets {
            response["stage_report"] = json!(stage_report);
            attach_scene_token_usage(&mut response);
            response["e2e_summary"] = scene_build_summary(&response, e2e_started.elapsed());
            if args.write_artifacts {
                write_scene_build_artifacts(&output_dir, &response)?;
            }
            return Ok(response);
        }

        let selected_synthesis_models = asset_lift_policy.synthesis_models.clone();
        let selected_asset_model = selected_synthesis_models
            .first()
            .copied()
            .unwrap_or(SynthesisModel::Trellis);
        response["asset_synthesis_models"] = json!(
            selected_synthesis_models
                .iter()
                .map(|model| model.as_str())
                .collect::<Vec<_>>()
        );
        let stage_started = Instant::now();
        progress.emit_with_items(
            "images_to_assets",
            SceneBuildProgressPhase::Started,
            SceneBuildExecutionKind::Gpu,
            format!(
                "lifting {} selected image(s) into {} assets",
                selected_values.len(),
                selected_asset_model.as_str()
            ),
            None,
            Some(selected_values.len()),
            None,
            json!({
                "selected": selected_values.len(),
                "synthesis_models": selected_synthesis_models.iter().map(|model| model.as_str()).collect::<Vec<_>>(),
                "backend": self.config.default_backend,
                "batch_size": args.batch_size,
                "batch_vram_mb": args.batch_vram_mb,
                "trellis_pbr": args.trellis_pbr.unwrap_or(true),
            }),
        );
        let mut excluded_asset_candidates = HashSet::new();
        let mut cached_asset_outputs = HashMap::<(String, usize), Value>::new();
        let mut asset_attempts = Vec::new();
        let asset_outputs = loop {
            let missing_selected = selected_values
                .iter()
                .filter(|selected| {
                    scene_selected_candidate_key(selected)
                        .is_none_or(|key| !cached_asset_outputs.contains_key(&key))
                })
                .cloned()
                .collect::<Vec<_>>();
            if !missing_selected.is_empty() {
                let missing_input_image_paths = missing_selected
                    .iter()
                    .map(|candidate| {
                        candidate
                            .get("image_path")
                            .and_then(Value::as_str)
                            .map(PathBuf::from)
                            .ok_or_else(|| "selected candidate missing image_path".to_string())
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let chunk_size =
                    scene_asset_lift_chunk_size(args.batch_size, missing_selected.len());
                for (chunk_index, (candidate_chunk, input_chunk)) in missing_selected
                    .chunks(chunk_size)
                    .zip(missing_input_image_paths.chunks(chunk_size))
                    .enumerate()
                {
                    progress.emit_with_items(
                        "images_to_assets",
                        SceneBuildProgressPhase::Progress,
                        SceneBuildExecutionKind::Gpu,
                        format!(
                            "running {} asset lift chunk {}/{} for {} image(s)",
                            selected_asset_model.as_str(),
                            chunk_index + 1,
                            missing_selected.len().div_ceil(chunk_size),
                            input_chunk.len()
                        ),
                        Some(cached_asset_outputs.len()),
                        Some(selected_values.len()),
                        Some(output_dir.join("assets")),
                        json!({
                            "attempt_index": asset_attempts.len(),
                            "chunk_index": chunk_index,
                            "chunk_count": missing_selected.len().div_ceil(chunk_size),
                            "chunk_size": chunk_size,
                            "batch_size_source": if args.batch_size.unwrap_or(0) > 0 { "explicit" } else { "scene_build_review_default" },
                            "synthesis_models": selected_synthesis_models.iter().map(|model| model.as_str()).collect::<Vec<_>>(),
                            "inputs": input_chunk.iter().map(|path| path.display().to_string()).collect::<Vec<_>>(),
                            "objects": candidate_chunk.iter().map(|candidate| {
                                json!({
                                    "object_id": candidate.get("object_id").and_then(Value::as_str),
                                    "candidate_index": candidate.get("candidate_index").and_then(Value::as_u64),
                                })
                            }).collect::<Vec<_>>(),
                        }),
                    );
                    let new_outputs = self.call_images_to_assets(ImagesToAssetsToolArgs {
                        input_image_paths: input_chunk.to_vec(),
                        output_dir: Some(output_dir.join("assets")),
                        output_paths: None,
                        output_format: Some(asset_lift_policy.output_format),
                        rmbg_model: Some(ForegroundModel::Rmbg2),
                        synthesis_models: Some(selected_synthesis_models.clone()),
                        backend: Some(self.config.default_backend),
                        target_faces: args
                            .target_faces
                            .or(Some(DEFAULT_SCENE_TRELLIS_TARGET_FACES)),
                        batch_size: Some(input_chunk.len()),
                        batch_vram_mb: args.batch_vram_mb,
                        trellis_pbr: Some(args.trellis_pbr.unwrap_or(true)),
                        trellis_pbr_texture_size: args
                            .trellis_pbr_texture_size
                            .or(Some(DEFAULT_SCENE_TRELLIS_PBR_TEXTURE_SIZE)),
                        promote_to_catalog: args.promote_to_catalog,
                        dry_run: false,
                    })?;
                    let runtime_progress_events = new_outputs
                        .get("runtime_progress_events")
                        .and_then(Value::as_array)
                        .cloned()
                        .unwrap_or_default();
                    emit_asset_runtime_progress_events(
                        progress,
                        &runtime_progress_events,
                        chunk_index,
                        Some(cached_asset_outputs.len()),
                        Some(selected_values.len()),
                        output_dir.join("assets"),
                    );
                    cache_scene_asset_outputs(
                        &mut cached_asset_outputs,
                        candidate_chunk,
                        &new_outputs,
                    )?;
                    progress.emit_with_items(
                        "images_to_assets",
                        SceneBuildProgressPhase::Progress,
                        SceneBuildExecutionKind::Gpu,
                        format!(
                            "{} asset lift chunk {}/{} complete; checking asset quality",
                            selected_asset_model.as_str(),
                            chunk_index + 1,
                            missing_selected.len().div_ceil(chunk_size)
                        ),
                        Some(cached_asset_outputs.len()),
                        Some(selected_values.len()),
                        Some(output_dir.join("assets")),
                        json!({
                            "chunk_index": chunk_index,
                            "stats": new_outputs.get("stats").cloned().unwrap_or(Value::Null),
                            "items": new_outputs.get("items").and_then(Value::as_array).map(Vec::len).unwrap_or_default(),
                            "runtime_progress_events": runtime_progress_events,
                        }),
                    );
                }
            }
            let outputs =
                scene_cached_asset_outputs_for_selected(&selected_values, &cached_asset_outputs)?;
            let attempt_failures =
                scene_asset_quality_failures_with_selected(&outputs, &selected_values);
            asset_attempts.push(json!({
                "attempt_index": asset_attempts.len(),
                "selected_candidates": selected_values.clone(),
                "lifted_candidates": missing_selected,
                "mesh_quality_failures": attempt_failures.iter().map(SceneAssetQualityFailure::message).collect::<Vec<_>>(),
            }));
            if attempt_failures.is_empty() {
                break outputs;
            }

            for failure in &attempt_failures {
                excluded_asset_candidates
                    .insert((failure.object_id.clone(), failure.candidate_index));
            }
            let next_selected = select_object_image_candidates_with_exclusions(
                &manifest,
                &candidate_report.candidates,
                candidate_policy.min_score,
                &excluded_asset_candidates,
            );
            let Ok(next_selected) = next_selected else {
                break outputs;
            };
            if next_selected == selected {
                break outputs;
            }
            selected = next_selected;
            selected_values = selected_candidates_to_values_with_requests(&selected, &requests);
        };
        record_stage(&mut stage_report, "images_to_assets", stage_started);
        progress.emit_with_items(
            "images_to_assets",
            SceneBuildProgressPhase::Completed,
            SceneBuildExecutionKind::Gpu,
            "asset lifting and mesh quality gates complete",
            None,
            Some(selected_values.len()),
            Some(output_dir.join("asset_outputs.json")),
            json!({
                "attempts": asset_attempts.len(),
                "items": asset_outputs.get("items").and_then(Value::as_array).map(Vec::len).unwrap_or_default(),
            }),
        );
        let mesh_quality_failures =
            scene_asset_quality_failures_with_selected(&asset_outputs, &selected_values)
                .into_iter()
                .map(|failure| failure.message())
                .collect::<Vec<_>>();
        response["selected_candidates"] = json!(selected_values.clone());
        response["asset_lift_attempts"] = json!(asset_attempts);
        response["asset_outputs"] = asset_outputs.clone();
        if let Err(err) =
            validate_scene_asset_outputs_for_policy(&asset_outputs, &asset_lift_policy)
        {
            response["failed_stage"] = json!("images_to_assets.asset_contract");
            response["asset_contract_error"] = json!(err);
            response["next_action"] = json!({
                "kind": "rerun_scene_with_mesh_assets",
                "recommendation": "Use Trellis.2 or another GLB mesh-producing model for explicit scene composition. TripoSplat Gaussian splats need a separate splat-aware placement solver.",
                "reason": "The active scene pose optimizer requires mesh local AABBs and GLB visible-surface fitting.",
            });
            response["stage_report"] = json!(stage_report);
            attach_scene_token_usage(&mut response);
            response["e2e_summary"] = scene_build_summary(&response, e2e_started.elapsed());
            if args.write_artifacts {
                write_scene_build_artifacts(&output_dir, &response)?;
            }
            return Err(response["asset_contract_error"]
                .as_str()
                .unwrap_or("scene asset contract failed")
                .to_string());
        }
        if !mesh_quality_failures.is_empty() {
            response["mesh_quality_failures"] = json!(mesh_quality_failures);
            response["failed_stage"] = json!("images_to_assets.mesh_quality_gate");
            response["next_action"] = json!({
                "kind": "regenerate_failed_assets",
                "recommendation": "Generate another isolated object-image candidate for each failed asset and rerun TRELLIS lifting before scene grounding/composition.",
                "reason": "Bad mesh topology should not be allowed to propagate into scene placement, feedback, or catalog reuse.",
            });
            response["stage_report"] = json!(stage_report);
            attach_scene_token_usage(&mut response);
            response["e2e_summary"] = scene_build_summary(&response, e2e_started.elapsed());
            if args.write_artifacts {
                write_scene_build_artifacts(&output_dir, &response)?;
            }
            return Err(response["mesh_quality_failures"]
                .as_array()
                .and_then(|failures| failures.first())
                .and_then(Value::as_str)
                .unwrap_or("scene mesh quality gate failed")
                .to_string());
        }
        let mut asset_bindings =
            scene_asset_bindings_from_outputs(&manifest, &selected_values, &asset_outputs)?;
        response["asset_bindings_initial"] =
            serde_json::to_value(&asset_bindings).map_err(|err| err.to_string())?;
        response["asset_bindings"] =
            serde_json::to_value(&asset_bindings).map_err(|err| err.to_string())?;
        response["asset_bindings_calibrated"] =
            serde_json::to_value(&asset_bindings).map_err(|err| err.to_string())?;
        if args.write_artifacts {
            write_scene_build_artifacts(&output_dir, &response)?;
        }
        let stage_started = Instant::now();
        progress.emit(
            "load_grounding_evidence",
            SceneBuildProgressPhase::Started,
            if args.locator == SceneLocatorProvider::LocateAnything {
                SceneBuildExecutionKind::Gpu
            } else {
                SceneBuildExecutionKind::Cpu
            },
            if args.locator == SceneLocatorProvider::LocateAnything {
                "running LocateAnything grounding"
            } else {
                "using manifest grounding fallback"
            },
            json!({
                "composition_mode": args.composition_mode,
                "locator": args.locator,
            }),
        );
        let (grounding_source, mut grounding_evidence): (String, SceneGroundingEvidence) =
            if let Some(evidence) = pre_generation_grounding_evidence.clone() {
                (pre_generation_grounding_source.clone(), evidence)
            } else if args.composition_mode == SceneCompositionMode::CvGrounded {
                if args.locator == SceneLocatorProvider::LocateAnything {
                    let backend = args
                        .locate_anything_backend
                        .unwrap_or(self.config.locate_anything_backend);
                    let evidence = self.locate_anything_grounding_evidence(
                        backend,
                        &manifest,
                        &args.source_scene_path,
                        &output_dir,
                        &category_filter,
                    )?;
                    let LocateAnythingBackend::BurnNative = backend;
                    ("locate_anything_burn_native".to_string(), evidence)
                } else {
                    (
                        "manifest_fallback".to_string(),
                        manifest_grounding_evidence(&manifest),
                    )
                }
            } else {
                (
                    "disabled".to_string(),
                    manifest_grounding_evidence(&manifest),
                )
            };
        record_stage(&mut stage_report, "load_grounding_evidence", stage_started);
        let mut segmentation_report = pre_generation_segmentation_report.clone();
        progress.emit_with_items(
            "load_grounding_evidence",
            SceneBuildProgressPhase::Completed,
            if args.locator == SceneLocatorProvider::LocateAnything {
                SceneBuildExecutionKind::Gpu
            } else {
                SceneBuildExecutionKind::Cpu
            },
            format!(
                "grounding evidence loaded from {grounding_source}; {} object(s)",
                grounding_evidence.objects.len()
            ),
            None,
            Some(grounding_evidence.objects.len()),
            Some(output_dir.join("grounding_evidence.json")),
            json!({
                "grounding_source": grounding_source,
                "objects": grounding_evidence.objects.len(),
            }),
        );
        if args.composition_mode == SceneCompositionMode::CvGrounded
            && segmentation_provider != SceneSegmentationProvider::None
            && grounding_evidence.segmentation.is_none()
        {
            let stage_started = Instant::now();
            progress.emit(
                "segmentation_grounding_evidence",
                SceneBuildProgressPhase::Started,
                if segmentation_provider == SceneSegmentationProvider::BboxPrompt {
                    SceneBuildExecutionKind::Cpu
                } else {
                    SceneBuildExecutionKind::Gpu
                },
                "running segmentation mask grounding",
                json!({
                    "segmentation_provider": segmentation_provider,
                    "segmentation_precision": segmentation_precision,
                    "segmentation_quantization": segmentation_quantization,
                    "objects": grounding_evidence.objects.len(),
                }),
            );
            segmentation_report = self.segmentation_grounding_evidence(
                segmentation_provider,
                Some(segmentation_precision),
                Some(segmentation_quantization),
                &mut grounding_evidence,
                &args.source_scene_path,
                &output_dir,
            )?;
            record_stage(
                &mut stage_report,
                "segmentation_grounding_evidence",
                stage_started,
            );
            progress.emit_with_items(
                "segmentation_grounding_evidence",
                SceneBuildProgressPhase::Completed,
                if segmentation_provider == SceneSegmentationProvider::BboxPrompt {
                    SceneBuildExecutionKind::Cpu
                } else {
                    SceneBuildExecutionKind::Gpu
                },
                "segmentation mask grounding complete",
                None,
                segmentation_report.as_ref().map(|report| report.mask_count),
                segmentation_report
                    .as_ref()
                    .map(|report| report.overlay_path.clone()),
                json!({
                    "segmentation_provider": segmentation_provider,
                    "segmentation_precision": segmentation_precision,
                    "segmentation_quantization": segmentation_quantization,
                    "mask_count": segmentation_report.as_ref().map(|report| report.mask_count),
                    "runtime_cache_hit": segmentation_report.as_ref().map(|report| report.runtime_cache_hit),
                }),
            );
        }
        if args.composition_mode == SceneCompositionMode::CvGrounded
            && args.depth_provider == SceneDepthProvider::DepthPro
            && grounding_evidence.depth.is_none()
        {
            let stage_started = Instant::now();
            progress.emit(
                "depth_pro_grounding_evidence",
                SceneBuildProgressPhase::Started,
                SceneBuildExecutionKind::Gpu,
                "running DepthPro camera/depth/floor grounding",
                json!({
                    "depth_provider": args.depth_provider,
                    "cache_dir": self.config.depth_cache_dir.clone(),
                    "precision": self.config.depth_precision,
                    "mask_count": grounding_evidence
                        .segmentation
                        .as_ref()
                        .and_then(|segmentation| segmentation.mask_count),
                }),
            );
            let depth_report = self.depth_pro_grounding_evidence(
                &mut grounding_evidence,
                &args.source_scene_path,
                &output_dir,
            )?;
            record_stage(
                &mut stage_report,
                "depth_pro_grounding_evidence",
                stage_started,
            );
            progress.emit_with_items(
                "depth_pro_grounding_evidence",
                SceneBuildProgressPhase::Completed,
                SceneBuildExecutionKind::Gpu,
                "DepthPro grounding complete",
                None,
                Some(grounding_evidence.objects.len()),
                Some(output_dir.join("depth_pro").join("depth_evidence.json")),
                json!({
                    "has_depth": grounding_evidence.depth.is_some(),
                    "runtime_cache_hit": depth_report.runtime_cache_hit,
                    "load_ms": depth_report.load_ms,
                    "infer_ms": depth_report.infer_ms,
                    "depth_map_path": depth_report.depth_map_path.display().to_string(),
                    "depth_map_metadata_path": depth_report
                        .depth_map_metadata_path
                        .display()
                        .to_string(),
                    "floor_sample_count": grounding_evidence
                        .depth
                        .as_ref()
                        .and_then(|depth| depth.floor_sample_count),
                }),
            );
        }
        let initial_asset_bindings = asset_bindings.clone();
        let stage_started = Instant::now();
        progress.emit(
            "canonical_pose_calibration",
            SceneBuildProgressPhase::Started,
            if args.canonical_pose == SceneCanonicalPoseMode::Openai {
                SceneBuildExecutionKind::Network
            } else {
                SceneBuildExecutionKind::Cpu
            },
            "calibrating generated asset-local canonical yaw frames",
            json!({
                "canonical_pose": args.canonical_pose,
                "asset_bindings": asset_bindings.len(),
                "max_pose_candidates": args.max_pose_candidates,
            }),
        );
        let canonical_pose_calibration =
            self.calibrate_scene_asset_frames(SceneAssetFrameCalibrationRequest {
                output_dir: &output_dir,
                write_artifacts: args.write_artifacts,
                mode: args.canonical_pose,
                max_candidates: args.max_pose_candidates,
                manifest: &manifest,
                asset_bindings: &asset_bindings,
                selected_candidates: &selected_values,
                object_image_requests: &requests,
                evidence: &grounding_evidence,
            })?;
        asset_bindings = canonical_pose_calibration.asset_bindings.clone();
        response["asset_bindings_initial"] =
            serde_json::to_value(&initial_asset_bindings).map_err(|err| err.to_string())?;
        response["asset_bindings"] =
            serde_json::to_value(&asset_bindings).map_err(|err| err.to_string())?;
        response["asset_bindings_calibrated"] =
            serde_json::to_value(&asset_bindings).map_err(|err| err.to_string())?;
        response["canonical_pose_calibration"] =
            serde_json::to_value(&canonical_pose_calibration.reports)
                .map_err(|err| err.to_string())?;
        response["canonical_pose_selection"] = canonical_pose_calibration.selection_report.clone();
        response["canonical_pose_selection_task"] =
            canonical_pose_calibration.selection_task.clone();
        let canonical_pose_verification =
            canonical_pose_verification_report(args.canonical_pose, &canonical_pose_calibration);
        response["canonical_pose_verification"] = canonical_pose_verification.clone();
        mark_canonical_pose_verification_failure(&mut response, &canonical_pose_verification);
        if args.write_artifacts {
            write_scene_build_artifacts(&output_dir, &response)?;
        }
        record_stage(
            &mut stage_report,
            "canonical_pose_calibration",
            stage_started,
        );
        progress.emit_with_items(
            "canonical_pose_calibration",
            SceneBuildProgressPhase::Completed,
            if args.canonical_pose == SceneCanonicalPoseMode::Openai {
                SceneBuildExecutionKind::Network
            } else {
                SceneBuildExecutionKind::Cpu
            },
            "canonical pose calibration complete",
            None,
            Some(canonical_pose_calibration.reports.len()),
            Some(output_dir.join("canonical_pose_calibration_report.json")),
            json!({
                "selector": canonical_pose_calibration
                    .selection_report
                    .get("selector")
                    .cloned()
                    .unwrap_or(Value::Null),
                "fallback_count": canonical_pose_calibration
                    .reports
                    .iter()
                    .filter(|report| report.fallback_used)
                    .count(),
                "verification_status": canonical_pose_verification
                    .get("status")
                    .cloned()
                    .unwrap_or(Value::Null),
                "visual_verified": canonical_pose_verification
                    .get("visual_verified")
                    .cloned()
                    .unwrap_or(Value::Null),
            }),
        );
        let stage_started = Instant::now();
        progress.emit(
            "plan_grounded_scene",
            SceneBuildProgressPhase::Started,
            SceneBuildExecutionKind::Cpu,
            "solving grounded scene layout and projection fit",
            json!({
                "asset_bindings": asset_bindings.len(),
                "objects": grounding_evidence.objects.len(),
                "composition_mode": args.composition_mode,
                "requested_pose_fit": args.pose_fit,
                "pose_fit": args.pose_fit,
            }),
        );
        let composition_candidates = scene_composition_candidates(
            args.composition_mode,
            args.feedback && args.lift_assets,
            &manifest,
            &asset_bindings,
            &grounding_evidence,
            args.clear_existing,
            args.scale_policy,
        )?;
        let mut selected_composition = composition_candidates
            .first()
            .cloned()
            .ok_or_else(|| "scene composition produced no candidates".to_string())?;
        let mut commands = selected_composition.commands.clone();
        let mut feedback_candidate_reports = Vec::new();
        record_stage(&mut stage_report, "plan_grounded_scene", stage_started);
        progress.emit_with_items(
            "plan_grounded_scene",
            SceneBuildProgressPhase::Completed,
            SceneBuildExecutionKind::Cpu,
            format!(
                "planned {} composition candidate(s)",
                composition_candidates.len()
            ),
            None,
            Some(composition_candidates.len()),
            Some(output_dir.join("grounded_layout.json")),
            json!({
                "candidate_count": composition_candidates.len(),
                "initial_mode": selected_composition.mode,
                "commands": commands.len(),
            }),
        );
        if args.pose_fit == ScenePoseFitMode::RenderedSilhouette && args.lift_assets {
            let stage_started = Instant::now();
            progress.emit(
                "visible_surface_pose_fit",
                SceneBuildProgressPhase::Started,
                SceneBuildExecutionKind::Cpu,
                "fitting object transforms from source masks/depth and projected GLB visible surfaces",
                json!({
                    "pose_fit": args.pose_fit,
                    "objects": selected_composition.layout.placements.len(),
                    "segmentation_provider": segmentation_provider,
                    "depth_provider": args.depth_provider,
                    "scale_policy": args.scale_policy,
                }),
            );
            let fit = apply_scene_visible_surface_pose_fit(
                SceneVisibleSurfacePoseFitConfig {
                    mode: args.pose_fit,
                    min_mask_iou: args.rotation_fit_min_mask_iou,
                    max_depth_error_m: args.rotation_fit_max_depth_error_m,
                    write_artifacts: args.write_artifacts,
                    output_dir: &output_dir.join("visible_surface_pose_fit"),
                    scale_policy: args.scale_policy,
                    object_filter: ScenePoseFitObjectFilter::All,
                },
                &manifest,
                &asset_bindings,
                Some(&grounding_evidence),
                &selected_composition.layout,
                &commands,
            )?;
            commands = scene_commands_with_asset_local_aabbs(fit.commands, &asset_bindings);
            selected_composition.layout = fit.grounded_layout;
            let pose_bsn = feedback_bsn_from_commands(
                &asset_bindings,
                &selected_composition.layout,
                &commands,
            )?;
            selected_composition.plan =
                parse_scene_bsn(&pose_bsn, &asset_bindings).map_err(|err| err.to_string())?;
            selected_composition.layout.bsn = pose_bsn;
            selected_composition.commands = commands.clone();
            response["visible_surface_pose_fit"] = fit.report;
            record_stage(&mut stage_report, "visible_surface_pose_fit", stage_started);
            progress.emit_with_items(
                "visible_surface_pose_fit",
                SceneBuildProgressPhase::Completed,
                SceneBuildExecutionKind::Cpu,
                "visible-surface pose fit complete",
                None,
                Some(selected_composition.layout.placements.len()),
                Some(
                    output_dir
                        .join("visible_surface_pose_fit")
                        .join("visible_surface_pose_fit_report.json"),
                ),
                json!({
                    "applied_count": response
                        .pointer("/visible_surface_pose_fit/applied_count")
                        .cloned()
                        .unwrap_or(Value::Null),
                    "skipped_count": response
                        .pointer("/visible_surface_pose_fit/skipped_count")
                        .cloned()
                        .unwrap_or(Value::Null),
                }),
            );
        }
        if args.pose_fit == ScenePoseFitMode::RenderedSilhouette
            && args.lift_assets
            && args.object_pose_refinement.geometry_enabled()
        {
            let stage_started = Instant::now();
            progress.emit(
                "object_pose_refinement",
                SceneBuildProgressPhase::Started,
                SceneBuildExecutionKind::Cpu,
                "refining selected object placement, rotation, and uniform scale with mask/depth constraints",
                json!({
                    "mode": args.object_pose_refinement,
                    "object_set": args.object_pose_refinement_set,
                    "scale_policy": args.scale_policy,
                }),
            );
            let fit = apply_scene_object_pose_refinement(
                SceneObjectPoseRefinementConfig {
                    mode: args.object_pose_refinement,
                    object_set: args.object_pose_refinement_set,
                    pose_fit: SceneVisibleSurfacePoseFitConfig {
                        mode: args.pose_fit,
                        min_mask_iou: args.rotation_fit_min_mask_iou,
                        max_depth_error_m: args.rotation_fit_max_depth_error_m,
                        write_artifacts: args.write_artifacts,
                        output_dir: &output_dir.join("object_pose_refinement"),
                        scale_policy: args.scale_policy,
                        object_filter: ScenePoseFitObjectFilter::RefinementSet(
                            args.object_pose_refinement_set,
                        ),
                    },
                },
                &manifest,
                &asset_bindings,
                Some(&grounding_evidence),
                &selected_composition.layout,
                &commands,
            )?;
            commands = scene_commands_with_asset_local_aabbs(fit.commands, &asset_bindings);
            selected_composition.layout = fit.grounded_layout;
            let pose_bsn = feedback_bsn_from_commands(
                &asset_bindings,
                &selected_composition.layout,
                &commands,
            )?;
            selected_composition.plan =
                parse_scene_bsn(&pose_bsn, &asset_bindings).map_err(|err| err.to_string())?;
            selected_composition.layout.bsn = pose_bsn;
            selected_composition.commands = commands.clone();
            response["object_pose_refinement"] = fit.report;
            record_stage(&mut stage_report, "object_pose_refinement", stage_started);
            progress.emit_with_items(
                "object_pose_refinement",
                SceneBuildProgressPhase::Completed,
                SceneBuildExecutionKind::Cpu,
                "object pose refinement complete",
                None,
                Some(selected_composition.layout.placements.len()),
                Some(
                    output_dir
                        .join("object_pose_refinement")
                        .join("object_pose_refinement_report.json"),
                ),
                json!({
                    "applied_count": response
                        .pointer("/object_pose_refinement/applied_count")
                        .cloned()
                        .unwrap_or(Value::Null),
                    "gpt_required": response
                        .pointer("/object_pose_refinement/gpt_gate/required")
                        .cloned()
                        .unwrap_or(Value::Null),
                }),
            );
        }
        if args.pose_fit == ScenePoseFitMode::RenderedSilhouette
            && args.lift_assets
            && args.final_yaw_refinement.enabled()
        {
            let stage_started = Instant::now();
            progress.emit(
                "final_context_yaw_refinement",
                SceneBuildProgressPhase::Started,
                SceneBuildExecutionKind::Viewer,
                "selecting final full-scene contextual yaw candidates for high-risk objects",
                json!({
                    "mode": args.final_yaw_refinement,
                    "object_set": args.final_yaw_refinement_set,
                    "confidence_threshold": args.final_yaw_confidence_threshold,
                    "max_candidates": args.final_yaw_max_candidates,
                }),
            );
            let prior_pose_report = response
                .get("object_pose_refinement")
                .or_else(|| response.get("visible_surface_pose_fit"))
                .cloned();
            let fit = self.run_final_context_yaw_refinement(
                &output_dir.join("final_yaw_refinement"),
                &manifest,
                &asset_bindings,
                &selected_composition.layout,
                &commands,
                Some(&grounding_evidence),
                prior_pose_report.as_ref(),
                args.final_yaw_refinement,
                args.final_yaw_refinement_set,
                args.final_yaw_confidence_threshold,
                args.final_yaw_max_candidates,
                args.write_artifacts,
            )?;
            commands = scene_commands_with_asset_local_aabbs(fit.commands, &asset_bindings);
            selected_composition.layout = fit.grounded_layout;
            let pose_bsn = feedback_bsn_from_commands(
                &asset_bindings,
                &selected_composition.layout,
                &commands,
            )?;
            selected_composition.plan =
                parse_scene_bsn(&pose_bsn, &asset_bindings).map_err(|err| err.to_string())?;
            selected_composition.layout.bsn = pose_bsn;
            selected_composition.commands = commands.clone();
            response["final_yaw_refinement"] = fit.report;
            record_stage(
                &mut stage_report,
                "final_context_yaw_refinement",
                stage_started,
            );
            progress.emit_with_items(
                "final_context_yaw_refinement",
                SceneBuildProgressPhase::Completed,
                SceneBuildExecutionKind::Viewer,
                "final contextual yaw refinement complete",
                None,
                Some(selected_composition.layout.placements.len()),
                Some(
                    output_dir
                        .join("final_yaw_refinement")
                        .join("final_yaw_refinement_report.json"),
                ),
                json!({
                    "status": response
                        .pointer("/final_yaw_refinement/status")
                        .cloned()
                        .unwrap_or(Value::Null),
                    "applied_count": response
                        .pointer("/final_yaw_refinement/applied_count")
                        .cloned()
                        .unwrap_or(Value::Null),
                }),
            );
        }
        if args.feedback && args.lift_assets {
            let stage_started = Instant::now();
            progress.emit_with_items(
                "render_capture_feedback",
                SceneBuildProgressPhase::Started,
                SceneBuildExecutionKind::Viewer,
                format!(
                    "running render-capture-feedback for up to {} iteration(s)",
                    args.feedback_iters
                ),
                None,
                Some(args.feedback_iters),
                args.feedback_capture_dir.clone(),
                json!({
                    "feedback_iters": args.feedback_iters,
                    "threshold_profile": args.feedback_threshold_profile,
                    "rotation_selector": args.feedback_rotation_selector,
                    "rotation_fit": args.rotation_fit,
                    "rubric_scorer": args.feedback_rubric_scorer,
                }),
            );
            let feedback_progress: SceneFeedbackProgressSink<'_> = Rc::new(RefCell::new(Box::new(
                |emission: SceneFeedbackProgressEmission| {
                    progress.emit_with_items(
                        emission.stage,
                        emission.phase,
                        emission.execution,
                        emission.message,
                        emission.item_index,
                        emission.item_count,
                        emission.artifact_path,
                        emission.detail,
                    );
                },
            )));
            let selection = self.run_scene_composition_feedback_selection(
                &output_dir,
                &manifest,
                &asset_bindings,
                composition_candidates.clone(),
                SceneFeedbackOptions {
                    max_iters: args.feedback_iters,
                    keep_viewer: args.feedback_keep_viewer,
                    capture_dir: args.feedback_capture_dir.clone(),
                    threshold_profile: args.feedback_threshold_profile,
                    rotation_selector: args.feedback_rotation_selector,
                    rotation_fit: args.rotation_fit,
                    rotation_fit_max_gpt_rounds: args.rotation_fit_max_gpt_rounds,
                    rotation_fit_min_mask_iou: args.rotation_fit_min_mask_iou,
                    rotation_fit_max_depth_error_m: args.rotation_fit_max_depth_error_m,
                    rotation_fit_write_artifacts: args.rotation_fit_write_artifacts,
                    rubric_scorer: args.feedback_rubric_scorer,
                    scale_policy: args.scale_policy,
                    grounding_evidence: Some(grounding_evidence.clone()),
                },
                Some(Rc::clone(&feedback_progress)),
            )?;
            drop(feedback_progress);
            selected_composition = selection.candidate;
            commands = selection.commands;
            feedback_candidate_reports = selection.candidate_reports;
            let feedback = selection.feedback;
            if feedback
                .get("accepted")
                .and_then(Value::as_bool)
                .is_some_and(|accepted| !accepted)
                && response.get("failed_stage").is_none()
            {
                response["failed_stage"] = json!("render_capture_feedback.quality_gate");
                response["next_action"] = json!({
                    "reason": "Render-capture-feedback did not find a composition that satisfied projection and physical layout thresholds.",
                    "inspect": feedback_report_path_from_result(&feedback)
                        .unwrap_or_else(|| output_dir.join("iterations/feedback_report.md").display().to_string()),
                });
            }
            response["feedback"] = feedback;
            record_stage(&mut stage_report, "render_capture_feedback", stage_started);
            progress.emit_with_items(
                "render_capture_feedback",
                SceneBuildProgressPhase::Completed,
                SceneBuildExecutionKind::Viewer,
                "render-capture-feedback complete",
                None,
                Some(args.feedback_iters),
                feedback_report_path_from_result(response.get("feedback").unwrap_or(&Value::Null))
                    .map(PathBuf::from)
                    .or_else(|| Some(output_dir.join("feedback_report.json"))),
                json!({
                    "candidate_reports": feedback_candidate_reports.len(),
                    "accepted": response
                        .get("feedback")
                        .and_then(|feedback| feedback.get("accepted"))
                        .cloned()
                        .unwrap_or(Value::Null),
                }),
            );
        }
        let grounded_layout = selected_composition.layout;
        let plan = selected_composition.plan;
        let bsn = feedback_bsn_from_commands(&asset_bindings, &grounded_layout, &commands)?;
        response["asset_outputs"] = asset_outputs;
        response["asset_bindings_initial"] =
            serde_json::to_value(&initial_asset_bindings).map_err(|err| err.to_string())?;
        response["asset_bindings"] =
            serde_json::to_value(&asset_bindings).map_err(|err| err.to_string())?;
        response["asset_bindings_calibrated"] =
            serde_json::to_value(&asset_bindings).map_err(|err| err.to_string())?;
        response["canonical_pose_calibration"] =
            serde_json::to_value(&canonical_pose_calibration.reports)
                .map_err(|err| err.to_string())?;
        response["canonical_pose_selection"] = canonical_pose_calibration.selection_report;
        response["canonical_pose_selection_task"] = canonical_pose_calibration.selection_task;
        response["canonical_pose_verification"] = canonical_pose_verification;
        response["requested_composition_mode"] = json!(args.composition_mode);
        response["composition_mode"] = json!(selected_composition.mode);
        response["requested_pose_fit"] = json!(args.pose_fit);
        response["pose_fit"] = json!(args.pose_fit);
        response["pose_fit_note"] = json!(scene_pose_fit_note(args.pose_fit));
        response["canonical_pose"] = json!(args.canonical_pose);
        response["scale_policy"] = json!(args.scale_policy);
        response["max_pose_candidates"] = json!(args.max_pose_candidates);
        response["save_pose_debug"] = json!(args.save_pose_debug);
        if !feedback_candidate_reports.is_empty() {
            response["composition_candidate_reports"] = json!(feedback_candidate_reports);
        }
        response["depth_provider"] = json!(args.depth_provider);
        response["locator"] = json!(args.locator);
        response["segmentation_provider"] = json!(segmentation_provider);
        response["segmentation_precision"] = json!(segmentation_precision);
        response["segmentation_quantization"] = json!(segmentation_quantization);
        if let Some(report) = segmentation_report {
            response["segmentation_grounding"] = json!(report);
        }
        response["grounding_source"] = json!(grounding_source);
        response["grounding_evidence"] =
            serde_json::to_value(&grounding_evidence).map_err(|err| err.to_string())?;
        response["bsn"] = json!(bsn);
        response["plan"] = serde_json::to_value(&plan).map_err(|err| err.to_string())?;
        response["grounded_layout"] =
            serde_json::to_value(&grounded_layout).map_err(|err| err.to_string())?;
        let command_count = commands.len();
        response["commands"] = json!(commands);
        response["clear_existing"] = json!(args.clear_existing);
        response["apply"] = json!(args.apply);
        if args.apply && !args.feedback {
            let stage_started = Instant::now();
            progress.emit(
                "apply_scene_commands",
                SceneBuildProgressPhase::Started,
                SceneBuildExecutionKind::Viewer,
                format!("applying {command_count} scene command(s) to Bevy bridge"),
                json!({ "commands": command_count }),
            );
            match self
                .send_scene_commands(response["commands"].as_array().cloned().unwrap_or_default())
            {
                Ok(acknowledgement) => {
                    response["acknowledgement"] = acknowledgement;
                }
                Err(err) => {
                    record_stage(&mut stage_report, "apply_scene_commands", stage_started);
                    response["apply_error"] = json!(err);
                    response["stage_report"] = json!(stage_report);
                    attach_scene_token_usage(&mut response);
                    response["e2e_summary"] = scene_build_summary(&response, e2e_started.elapsed());
                    if args.write_artifacts {
                        write_scene_build_artifacts(&output_dir, &response)?;
                    }
                    return Err(response["apply_error"]
                        .as_str()
                        .unwrap_or("scene apply failed")
                        .to_string());
                }
            }
            record_stage(&mut stage_report, "apply_scene_commands", stage_started);
            progress.emit(
                "apply_scene_commands",
                SceneBuildProgressPhase::Completed,
                SceneBuildExecutionKind::Viewer,
                "scene commands applied",
                json!({ "commands": command_count }),
            );
        }
        response["stage_report"] = json!(stage_report);
        let token_usage = attach_scene_token_usage(&mut response);
        response["e2e_summary"] = scene_build_summary(&response, e2e_started.elapsed());
        attach_scene_grounding_contracts(
            &mut response,
            &args,
            &grounding_source,
            &grounding_evidence,
            segmentation_provider,
        )?;
        if args.promote_to_catalog {
            let stage_started = Instant::now();
            progress.emit(
                "promote_scene_to_catalog",
                SceneBuildProgressPhase::Started,
                SceneBuildExecutionKind::Cache,
                "promoting accepted scene snapshot to shared catalog",
                json!({ "catalog_cache_root": self.config.catalog_cache_root.clone() }),
            );
            let mut catalog_cache = self.open_catalog_cache()?;
            let scene_catalog_entry = promote_scene_build_scene_to_catalog(
                &mut catalog_cache,
                &args.source_scene_path,
                &output_dir,
                response["bsn"].as_str().unwrap_or_default(),
                &asset_bindings,
                &grounded_layout,
                &response,
            )?;
            record_stage(&mut stage_report, "promote_scene_to_catalog", stage_started);
            response["scene_catalog_entry"] = scene_catalog_entry;
            response["stage_report"] = json!(stage_report);
            response["e2e_summary"] = scene_build_summary(&response, e2e_started.elapsed());
            attach_scene_grounding_contracts(
                &mut response,
                &args,
                &grounding_source,
                &grounding_evidence,
                segmentation_provider,
            )?;
            progress.emit(
                "promote_scene_to_catalog",
                SceneBuildProgressPhase::Completed,
                SceneBuildExecutionKind::Cache,
                "scene catalog promotion complete",
                json!({
                    "scene_catalog_entry": response.get("scene_catalog_entry").cloned().unwrap_or(Value::Null),
                }),
            );
        }
        if args.write_artifacts {
            progress.emit(
                "write_scene_build_artifacts",
                SceneBuildProgressPhase::Started,
                SceneBuildExecutionKind::FileIo,
                "writing scene build artifacts",
                json!({ "output_dir": output_dir.display().to_string() }),
            );
            write_scene_build_artifacts(&output_dir, &response)?;
        }
        progress.emit_with_items(
            "scene_build",
            SceneBuildProgressPhase::Completed,
            SceneBuildExecutionKind::Mixed,
            "scene build complete",
            None,
            Some(command_count),
            Some(output_dir.join("scene_build_response_structured.json")),
            json!({
                "commands": command_count,
                "elapsed_ms": elapsed_ms(e2e_started.elapsed()),
                "failed_stage": response.get("failed_stage").cloned().unwrap_or(Value::Null),
                "token_usage": token_usage,
            }),
        );
        Ok(response)
    }

    fn call_scene_plan_bsn(&self, args: ScenePlanBsnArgs) -> Result<Value, String> {
        let grounded_layout =
            grounded_scene_layout_for_manifest(&args.manifest, &args.asset_bindings)
                .map_err(|err| err.to_string())?;
        let bsn = grounded_layout.bsn.clone();
        let plan = match parse_scene_bsn(&bsn, &args.asset_bindings) {
            Ok(plan) => plan,
            Err(err) => {
                return Ok(json!({
                    "tool": "scene_plan_bsn",
                    "valid": false,
                    "bsn": bsn,
                    "validation_error": err.to_string(),
                    "asset_bindings": args.asset_bindings,
                    "clear_existing": args.clear_existing,
                    "apply": false,
                }));
            }
        };
        let commands = scene_commands_with_cache_reload(
            scene_plan_to_mcp_commands(&plan, &args.asset_bindings, args.clear_existing)
                .map_err(|err| err.to_string())?,
        );
        let mut response = json!({
            "tool": "scene_plan_bsn",
            "valid": true,
            "bsn": bsn,
            "plan": plan,
            "grounded_layout": grounded_layout,
            "commands": commands,
            "asset_bindings": args.asset_bindings,
            "clear_existing": args.clear_existing,
            "apply": args.apply,
        });
        if args.apply {
            response["acknowledgement"] = self.send_scene_commands(
                response["commands"].as_array().cloned().unwrap_or_default(),
            )?;
        }
        Ok(response)
    }

    pub(crate) fn call_scene_ground(&mut self, args: SceneGroundToolArgs) -> Result<Value, String> {
        validate_scene_pose_fit_mode(args.pose_fit)?;
        let started = Instant::now();
        let output_dir = args.output_dir.unwrap_or_else(default_scene_output_dir);
        fs::create_dir_all(&output_dir).map_err(|err| {
            format!(
                "failed to create scene-ground output directory {}: {err}",
                output_dir.display()
            )
        })?;
        let mut stage_report = Vec::new();
        let stage_started = Instant::now();
        let category_filter = self.scene_category_filter(
            args.allowed_categories.clone(),
            args.denied_categories.clone(),
        );
        let mut manifest = args.manifest;
        manifest.source_scene_path = args.source_scene_path.display().to_string();
        let mut asset_bindings = args.asset_bindings;
        let (grounding_source, mut evidence) = if let Some(evidence) = args.grounding_evidence {
            ("provided", evidence)
        } else if args.locator == SceneLocatorProvider::LocateAnything {
            let backend = args
                .locate_anything_backend
                .unwrap_or(self.config.locate_anything_backend);
            let evidence = self.locate_anything_grounding_evidence(
                backend,
                &manifest,
                &args.source_scene_path,
                &output_dir,
                &category_filter,
            )?;
            let LocateAnythingBackend::BurnNative = backend;
            let source = "locate_anything_burn_native";
            (source, evidence)
        } else {
            ("manifest_fallback", manifest_grounding_evidence(&manifest))
        };
        let (filtered_manifest, filtered_evidence, category_filter_report) =
            apply_scene_category_filter(&manifest, Some(&evidence), &category_filter);
        if filtered_manifest.objects.is_empty() {
            return Err(format!(
                "scene category filter removed all scene-ground objects; allowed={:?} denied={:?}",
                category_filter.allow, category_filter.deny
            ));
        }
        manifest = filtered_manifest;
        evidence = filtered_evidence.unwrap_or(evidence);
        let allowed_object_ids = manifest
            .objects
            .iter()
            .map(|object| object.id.as_str())
            .collect::<HashSet<_>>();
        asset_bindings.retain(|binding| allowed_object_ids.contains(binding.object_id.as_str()));
        write_json_file(
            &output_dir.join("category_filter_report.json"),
            &category_filter_report,
        )
        .map_err(|err| err.to_string())?;
        record_stage(&mut stage_report, "load_grounding_evidence", stage_started);
        let segmentation_provider = args
            .segmentation_provider
            .unwrap_or(self.config.scene_segmentation_provider);
        let segmentation_precision = args
            .segmentation_precision
            .unwrap_or(self.config.scene_segmentation_precision);
        let segmentation_quantization = args
            .segmentation_quantization
            .unwrap_or(self.config.scene_segmentation_quantization);
        let mut segmentation_report = None;
        let mut depth_report = None;
        let mut ground_calibration_report = None;

        if args.composition_mode == SceneCompositionMode::CvGrounded
            && segmentation_provider != SceneSegmentationProvider::None
            && evidence.segmentation.is_none()
        {
            let stage_started = Instant::now();
            segmentation_report = self.segmentation_grounding_evidence(
                segmentation_provider,
                Some(segmentation_precision),
                Some(segmentation_quantization),
                &mut evidence,
                &args.source_scene_path,
                &output_dir,
            )?;
            record_stage(
                &mut stage_report,
                "segmentation_grounding_evidence",
                stage_started,
            );
        }
        if args.depth_provider == SceneDepthProvider::DepthPro && evidence.depth.is_none() {
            let stage_started = Instant::now();
            depth_report = Some(self.depth_pro_grounding_evidence(
                &mut evidence,
                &args.source_scene_path,
                &output_dir,
            )?);
            record_stage(
                &mut stage_report,
                "depth_pro_grounding_evidence",
                stage_started,
            );
        }
        if args.composition_mode == SceneCompositionMode::CvGrounded
            && args.ground_calibration == SceneGroundCalibrationMode::Gpt
        {
            let stage_started = Instant::now();
            let config = SceneBuildConfig {
                source_scene_path: args.source_scene_path.clone(),
                object_reference_image_path: self.config.scene_object_reference_image.clone(),
                output_dir: output_dir.clone(),
                candidate_count: 1,
                quality_profile: SceneQualityProfile::Draft,
                reasoning_model: self.config.openai_reasoning_model.clone(),
                image_model: self.config.openai_image_model.clone(),
                allow_catalog_reuse: false,
                category_filter: self.config.scene_category_filter.clone(),
            };
            let provider = self.openai_provider()?;
            let pipeline = ScenePipeline::new(config, provider);
            ground_calibration_report = Some(apply_gpt_ground_calibration(
                &pipeline,
                &mut manifest,
                &mut evidence,
                &output_dir,
                true,
                &[],
            )?);
            record_stage(&mut stage_report, "ground_calibration_gpt", stage_started);
        } else if args.composition_mode == SceneCompositionMode::CvGrounded {
            let stage_started = Instant::now();
            let report = deterministic_ground_calibration_report(
                &manifest,
                &evidence,
                args.ground_calibration,
            );
            let stage_name = match args.ground_calibration {
                SceneGroundCalibrationMode::DepthFirst => "ground_calibration_depth_first",
                SceneGroundCalibrationMode::DepthHeuristic => "ground_calibration_depth_heuristic",
                SceneGroundCalibrationMode::Gpt => "ground_calibration_gpt",
            };
            write_json_file(
                &output_dir.join(format!("{stage_name}_report.json")),
                &report,
            )
            .map_err(|err| err.to_string())?;
            record_stage(&mut stage_report, stage_name, stage_started);
            ground_calibration_report = Some(report);
        }

        let initial_asset_bindings = asset_bindings.clone();
        let stage_started = Instant::now();
        let canonical_pose_calibration =
            self.calibrate_scene_asset_frames(SceneAssetFrameCalibrationRequest {
                output_dir: &output_dir,
                write_artifacts: true,
                mode: args.canonical_pose,
                max_candidates: args.max_pose_candidates,
                manifest: &manifest,
                asset_bindings: &asset_bindings,
                selected_candidates: &[],
                object_image_requests: &[],
                evidence: &evidence,
            })?;
        asset_bindings = canonical_pose_calibration.asset_bindings.clone();
        record_stage(
            &mut stage_report,
            "canonical_pose_calibration",
            stage_started,
        );
        let canonical_pose_verification =
            canonical_pose_verification_report(args.canonical_pose, &canonical_pose_calibration);

        let stage_started = Instant::now();
        let composition_candidates = scene_composition_candidates(
            args.composition_mode,
            args.feedback,
            &manifest,
            &asset_bindings,
            &evidence,
            args.clear_existing,
            args.scale_policy,
        )?;
        let mut selected_composition = composition_candidates
            .first()
            .cloned()
            .ok_or_else(|| "scene composition produced no candidates".to_string())?;
        let mut commands = selected_composition.commands.clone();
        let mut feedback_candidate_reports = Vec::new();
        record_stage(&mut stage_report, "solve_grounded_scene", stage_started);
        let scene_placement_pipeline =
            scene_placement_pipeline_plan(ScenePlacementPipelineSelection {
                entry_point: ScenePlacementEntryPoint::SceneGround,
                lift_assets: true,
                composition_mode: args.composition_mode,
                pose_fit: args.pose_fit,
                canonical_pose: args.canonical_pose,
                scale_policy: args.scale_policy,
                ground_calibration: args.ground_calibration,
                instance_generation: SceneInstanceGenerationMode::CategoryRepresentative,
                depth_provider: args.depth_provider,
                locator: args.locator,
                segmentation_provider,
                feedback: args.feedback,
                feedback_iters: args.feedback_iters,
                feedback_rotation_selector: args.feedback_rotation_selector,
                feedback_rubric_scorer: args.feedback_rubric_scorer,
                rotation_fit: args.rotation_fit,
                object_pose_refinement: args.object_pose_refinement,
                object_pose_refinement_set: args.object_pose_refinement_set,
                final_yaw_refinement: args.final_yaw_refinement,
                final_yaw_refinement_set: args.final_yaw_refinement_set,
                max_pose_candidates: args.max_pose_candidates,
            });

        let mut response = json!({
            "tool": "scene_ground",
            "source_scene_path": args.source_scene_path,
            "requested_composition_mode": args.composition_mode,
            "requested_pose_fit": args.pose_fit,
            "pose_fit": args.pose_fit,
            "pose_fit_note": scene_pose_fit_note(args.pose_fit),
            "canonical_pose": args.canonical_pose,
            "scale_policy": args.scale_policy,
            "max_pose_candidates": args.max_pose_candidates,
            "save_pose_debug": args.save_pose_debug,
            "ground_calibration": args.ground_calibration,
            "depth_provider": args.depth_provider,
            "locator": args.locator,
            "category_filter": category_filter,
            "category_filter_report": category_filter_report,
            "segmentation_provider": segmentation_provider,
            "segmentation_precision": segmentation_precision,
            "segmentation_quantization": segmentation_quantization,
            "scene_placement_pipeline": scene_placement_pipeline.to_json(),
            "grounding_source": grounding_source,
            "manifest": manifest.clone(),
            "asset_bindings_initial": initial_asset_bindings,
            "asset_bindings": asset_bindings.clone(),
            "asset_bindings_calibrated": asset_bindings.clone(),
            "canonical_pose_calibration": canonical_pose_calibration.reports,
            "canonical_pose_selection": canonical_pose_calibration.selection_report,
            "canonical_pose_selection_task": canonical_pose_calibration.selection_task,
            "canonical_pose_verification": canonical_pose_verification.clone(),
            "grounding_evidence": evidence.clone(),
            "clear_existing": args.clear_existing,
            "apply": args.apply,
            "promote_to_catalog": args.promote_to_catalog,
        });
        mark_canonical_pose_verification_failure(&mut response, &canonical_pose_verification);
        if let Some(report) = segmentation_report {
            response["segmentation_grounding"] = json!(report);
        }
        if let Some(report) = depth_report {
            response["depth_grounding"] = json!(report);
        }
        if let Some(report) = ground_calibration_report {
            response["ground_calibration_report"] = json!(report);
        }

        if args.pose_fit == ScenePoseFitMode::RenderedSilhouette {
            let stage_started = Instant::now();
            let fit = apply_scene_visible_surface_pose_fit(
                SceneVisibleSurfacePoseFitConfig {
                    mode: args.pose_fit,
                    min_mask_iou: args.rotation_fit_min_mask_iou,
                    max_depth_error_m: args.rotation_fit_max_depth_error_m,
                    write_artifacts: true,
                    output_dir: &output_dir.join("visible_surface_pose_fit"),
                    scale_policy: args.scale_policy,
                    object_filter: ScenePoseFitObjectFilter::All,
                },
                &manifest,
                &asset_bindings,
                Some(&evidence),
                &selected_composition.layout,
                &commands,
            )?;
            commands = scene_commands_with_asset_local_aabbs(fit.commands, &asset_bindings);
            selected_composition.layout = fit.grounded_layout;
            let pose_bsn = feedback_bsn_from_commands(
                &asset_bindings,
                &selected_composition.layout,
                &commands,
            )?;
            selected_composition.plan =
                parse_scene_bsn(&pose_bsn, &asset_bindings).map_err(|err| err.to_string())?;
            selected_composition.layout.bsn = pose_bsn;
            selected_composition.commands = commands.clone();
            response["visible_surface_pose_fit"] = fit.report;
            record_stage(&mut stage_report, "visible_surface_pose_fit", stage_started);
        }
        if args.pose_fit == ScenePoseFitMode::RenderedSilhouette
            && args.object_pose_refinement.geometry_enabled()
        {
            let stage_started = Instant::now();
            let fit = apply_scene_object_pose_refinement(
                SceneObjectPoseRefinementConfig {
                    mode: args.object_pose_refinement,
                    object_set: args.object_pose_refinement_set,
                    pose_fit: SceneVisibleSurfacePoseFitConfig {
                        mode: args.pose_fit,
                        min_mask_iou: args.rotation_fit_min_mask_iou,
                        max_depth_error_m: args.rotation_fit_max_depth_error_m,
                        write_artifacts: true,
                        output_dir: &output_dir.join("object_pose_refinement"),
                        scale_policy: args.scale_policy,
                        object_filter: ScenePoseFitObjectFilter::RefinementSet(
                            args.object_pose_refinement_set,
                        ),
                    },
                },
                &manifest,
                &asset_bindings,
                Some(&evidence),
                &selected_composition.layout,
                &commands,
            )?;
            commands = scene_commands_with_asset_local_aabbs(fit.commands, &asset_bindings);
            selected_composition.layout = fit.grounded_layout;
            let pose_bsn = feedback_bsn_from_commands(
                &asset_bindings,
                &selected_composition.layout,
                &commands,
            )?;
            selected_composition.plan =
                parse_scene_bsn(&pose_bsn, &asset_bindings).map_err(|err| err.to_string())?;
            selected_composition.layout.bsn = pose_bsn;
            selected_composition.commands = commands.clone();
            response["object_pose_refinement"] = fit.report;
            record_stage(&mut stage_report, "object_pose_refinement", stage_started);
        }

        if args.pose_fit == ScenePoseFitMode::RenderedSilhouette
            && args.final_yaw_refinement.enabled()
        {
            let stage_started = Instant::now();
            let prior_pose_report = response
                .get("object_pose_refinement")
                .or_else(|| response.get("visible_surface_pose_fit"))
                .cloned();
            let fit = self.run_final_context_yaw_refinement(
                &output_dir.join("final_yaw_refinement"),
                &manifest,
                &asset_bindings,
                &selected_composition.layout,
                &commands,
                Some(&evidence),
                prior_pose_report.as_ref(),
                args.final_yaw_refinement,
                args.final_yaw_refinement_set,
                args.final_yaw_confidence_threshold,
                args.final_yaw_max_candidates,
                true,
            )?;
            commands = scene_commands_with_asset_local_aabbs(fit.commands, &asset_bindings);
            selected_composition.layout = fit.grounded_layout;
            let pose_bsn = feedback_bsn_from_commands(
                &asset_bindings,
                &selected_composition.layout,
                &commands,
            )?;
            selected_composition.plan =
                parse_scene_bsn(&pose_bsn, &asset_bindings).map_err(|err| err.to_string())?;
            selected_composition.layout.bsn = pose_bsn;
            selected_composition.commands = commands.clone();
            response["final_yaw_refinement"] = fit.report;
            record_stage(
                &mut stage_report,
                "final_context_yaw_refinement",
                stage_started,
            );
        }

        if args.feedback {
            let stage_started = Instant::now();
            let selection = self.run_scene_composition_feedback_selection(
                &output_dir,
                &manifest,
                &asset_bindings,
                composition_candidates.clone(),
                SceneFeedbackOptions {
                    max_iters: args.feedback_iters,
                    keep_viewer: args.feedback_keep_viewer,
                    capture_dir: args.feedback_capture_dir.clone(),
                    threshold_profile: args.feedback_threshold_profile,
                    rotation_selector: args.feedback_rotation_selector,
                    rotation_fit: args.rotation_fit,
                    rotation_fit_max_gpt_rounds: args.rotation_fit_max_gpt_rounds,
                    rotation_fit_min_mask_iou: args.rotation_fit_min_mask_iou,
                    rotation_fit_max_depth_error_m: args.rotation_fit_max_depth_error_m,
                    rotation_fit_write_artifacts: args.rotation_fit_write_artifacts,
                    rubric_scorer: args.feedback_rubric_scorer,
                    scale_policy: args.scale_policy,
                    grounding_evidence: Some(evidence.clone()),
                },
                None,
            )?;
            selected_composition = selection.candidate;
            commands = selection.commands;
            feedback_candidate_reports = selection.candidate_reports;
            let feedback = selection.feedback;
            if feedback
                .get("accepted")
                .and_then(Value::as_bool)
                .is_some_and(|accepted| !accepted)
            {
                response["failed_stage"] = json!("render_capture_feedback.quality_gate");
                response["next_action"] = json!({
                    "reason": "Render-capture-feedback did not find a composition that satisfied projection and physical layout thresholds.",
                    "inspect": feedback_report_path_from_result(&feedback)
                        .unwrap_or_else(|| output_dir.join("iterations/feedback_report.md").display().to_string()),
                });
            }
            response["feedback"] = feedback;
            record_stage(&mut stage_report, "render_capture_feedback", stage_started);
        }

        let grounded_layout = selected_composition.layout;
        let plan = selected_composition.plan;
        let bsn = feedback_bsn_from_commands(&asset_bindings, &grounded_layout, &commands)?;
        response["composition_mode"] = json!(selected_composition.mode);
        response["requested_pose_fit"] = json!(args.pose_fit);
        response["pose_fit"] = json!(args.pose_fit);
        if !feedback_candidate_reports.is_empty() {
            response["composition_candidate_reports"] = json!(feedback_candidate_reports);
        }
        response["grounded_layout"] =
            serde_json::to_value(&grounded_layout).map_err(|err| err.to_string())?;
        response["bsn"] = json!(bsn);
        response["plan"] = serde_json::to_value(&plan).map_err(|err| err.to_string())?;
        response["commands"] = json!(commands.clone());

        if args.apply && !args.feedback {
            let stage_started = Instant::now();
            let acknowledgement = self.send_scene_commands(commands)?;
            response["acknowledgement"] = acknowledgement;
            record_stage(&mut stage_report, "apply_scene_commands", stage_started);
        }

        response["stage_report"] = json!(stage_report);
        response["e2e_summary"] = json!({
            "elapsed_ms": started.elapsed().as_secs_f64() * 1000.0,
            "objects": response["grounded_layout"]["placements"].as_array().map(Vec::len).unwrap_or_default(),
            "composition_mode": response["composition_mode"],
            "grounding_source": response["grounding_source"],
        });
        if args.promote_to_catalog {
            let stage_started = Instant::now();
            let mut catalog_cache = self.open_catalog_cache()?;
            let scene_catalog_entry = promote_scene_build_scene_to_catalog(
                &mut catalog_cache,
                &args.source_scene_path,
                &output_dir,
                response["bsn"].as_str().unwrap_or_default(),
                &asset_bindings,
                &grounded_layout,
                &response,
            )?;
            record_stage(&mut stage_report, "promote_scene_to_catalog", stage_started);
            response["scene_catalog_entry"] = scene_catalog_entry;
            response["stage_report"] = json!(stage_report);
            response["e2e_summary"] = json!({
                "elapsed_ms": started.elapsed().as_secs_f64() * 1000.0,
                "objects": response["grounded_layout"]["placements"].as_array().map(Vec::len).unwrap_or_default(),
                "composition_mode": response["composition_mode"],
                "grounding_source": response["grounding_source"],
            });
        }
        write_scene_ground_artifacts(&output_dir, &response)?;
        Ok(response)
    }

    fn call_scene_apply_bsn(&self, args: SceneApplyBsnArgs) -> Result<Value, String> {
        let plan =
            parse_scene_bsn(&args.bsn, &args.asset_bindings).map_err(|err| err.to_string())?;
        let commands = scene_commands_with_cache_reload(
            scene_plan_to_mcp_commands(&plan, &args.asset_bindings, args.clear_existing)
                .map_err(|err| err.to_string())?,
        );
        let mut response = json!({
            "tool": "scene_apply_bsn",
            "plan": plan,
            "commands": commands,
            "apply": args.apply,
        });
        if args.apply {
            response["acknowledgement"] = self.send_scene_commands(
                response["commands"].as_array().cloned().unwrap_or_default(),
            )?;
        }
        Ok(response)
    }

    fn scene_build_config(&self, args: ScenePrepareBuildArgs) -> Result<SceneBuildConfig, String> {
        let category_filter =
            self.scene_category_filter(args.allowed_categories, args.denied_categories);
        Ok(SceneBuildConfig {
            source_scene_path: args.source_scene_path,
            object_reference_image_path: args
                .object_reference_image_path
                .unwrap_or_else(|| self.config.scene_object_reference_image.clone()),
            output_dir: args.output_dir.unwrap_or_else(default_scene_output_dir),
            candidate_count: args.candidate_count.unwrap_or(3).max(1),
            quality_profile: args.quality_profile.unwrap_or(SceneQualityProfile::Quality),
            reasoning_model: self.config.openai_reasoning_model.clone(),
            image_model: self.config.openai_image_model.clone(),
            allow_catalog_reuse: args.allow_catalog_reuse,
            category_filter,
        })
    }

    fn scene_category_filter(
        &self,
        allowed_categories: Option<Vec<String>>,
        denied_categories: Option<Vec<String>>,
    ) -> SceneCategoryFilterConfig {
        SceneCategoryFilterConfig {
            allow: allowed_categories
                .unwrap_or_else(|| self.config.scene_category_filter.allow.clone()),
            deny: denied_categories
                .unwrap_or_else(|| self.config.scene_category_filter.deny.clone()),
        }
    }

    fn openai_provider(&self) -> Result<OpenAiSceneProvider, String> {
        OpenAiSceneProvider::from_env(OpenAiProviderConfig {
            api_key: self.config.openai_api_key.clone().unwrap_or_default(),
            base_url: self
                .config
                .openai_base_url
                .clone()
                .unwrap_or_else(|| OpenAiProviderConfig::default().base_url),
            project_id: self.config.openai_project_id.clone(),
            reasoning_model: self.config.openai_reasoning_model.clone(),
            image_model: self.config.openai_image_model.clone(),
            ..OpenAiProviderConfig::default()
        })
        .map_err(|err| err.to_string())
    }

    fn calibrate_scene_asset_frames(
        &mut self,
        request: SceneAssetFrameCalibrationRequest<'_>,
    ) -> Result<CanonicalPoseCalibrationRun, String> {
        let mut run = build_canonical_pose_calibration(
            request.mode,
            request.max_candidates,
            request.manifest,
            request.asset_bindings,
            request.selected_candidates,
            request.object_image_requests,
            request.evidence,
        );
        let render_report = self.render_canonical_pose_candidate_thumbnails(&request, &mut run);
        let rendered_selection_report = if matches!(
            request.mode,
            SceneCanonicalPoseMode::Auto
                | SceneCanonicalPoseMode::RenderSweep
                | SceneCanonicalPoseMode::Openai
        ) {
            let mut report = apply_canonical_pose_rendered_selection(&mut run);
            report["render_report"] = render_report.clone();
            report
        } else {
            Value::Null
        };
        run.selection_report = match request.mode {
            SceneCanonicalPoseMode::Off => json!({
                "selector": "disabled",
                "applied_count": 0,
                "reason": "canonical pose calibration disabled",
                "render_report": render_report,
            }),
            SceneCanonicalPoseMode::Openai => {
                if run.reports.is_empty() || run.image_paths.is_empty() {
                    let mut report = if rendered_selection_report.is_null() {
                        json!({
                            "selector": "openai",
                            "fallback": true,
                            "applied_count": 0,
                        })
                    } else {
                        rendered_selection_report.clone()
                    };
                    report["openai_fallback"] = json!(true);
                    report["reason"] = json!(
                        "missing canonical pose image evidence; kept rendered/deterministic candidates"
                    );
                    report["render_report"] = render_report;
                    report
                } else {
                    let prompt = canonical_pose_selection_prompt(&run.selection_task);
                    if request.write_artifacts {
                        write_json_file(
                            &request
                                .output_dir
                                .join("canonical_pose_selection_request.json"),
                            &json!({
                                "prompt": prompt,
                                "task": run.selection_task.clone(),
                                "image_paths": run
                                    .image_paths
                                    .iter()
                                    .map(|path| path.display().to_string())
                                    .collect::<Vec<_>>(),
                            }),
                        )
                        .map_err(|err| err.to_string())?;
                    }
                    match self.openai_provider().and_then(|provider| {
                        provider
                            .select_rotation_candidates(&SceneRotationSelectionRequest {
                                prompt,
                                task: run.selection_task.clone(),
                                image_paths: run.image_paths.clone(),
                            })
                            .map_err(|err| err.to_string())
                    }) {
                        Ok(response) => {
                            if request.write_artifacts {
                                write_json_file(
                                    &request
                                        .output_dir
                                        .join("canonical_pose_selection_response.json"),
                                    &serde_json::to_value(&response)
                                        .map_err(|err| err.to_string())?,
                                )
                                .map_err(|err| err.to_string())?;
                            }
                            let mut report =
                                apply_canonical_pose_openai_selection(&mut run, &response);
                            report["render_report"] = render_report;
                            report
                        }
                        Err(err) => {
                            let mut report = if rendered_selection_report.is_null() {
                                json!({
                                    "selector": "openai",
                                    "fallback": true,
                                    "applied_count": 0,
                                })
                            } else {
                                rendered_selection_report.clone()
                            };
                            report["openai_fallback"] = json!(true);
                            report["openai_error"] = json!(err);
                            report["reason"] =
                                json!("kept rendered/deterministic canonical pose candidates");
                            report["render_report"] = render_report;
                            report
                        }
                    }
                }
            }
            SceneCanonicalPoseMode::RenderSweep | SceneCanonicalPoseMode::Auto => {
                if rendered_selection_report.is_null() {
                    json!({
                        "selector": canonical_pose_mode_label(request.mode),
                        "applied_count": run
                            .reports
                            .iter()
                            .filter(|report| !report.fallback_used)
                            .count(),
                        "fallback_count": run
                            .reports
                            .iter()
                            .filter(|report| report.fallback_used)
                            .count(),
                        "reason": "deterministic canonical pose candidate selection",
                        "render_report": render_report,
                    })
                } else {
                    rendered_selection_report
                }
            }
            SceneCanonicalPoseMode::Heuristic => json!({
                "selector": canonical_pose_mode_label(request.mode),
                "applied_count": run
                    .reports
                    .iter()
                    .filter(|report| !report.fallback_used)
                    .count(),
                "fallback_count": run
                    .reports
                    .iter()
                    .filter(|report| report.fallback_used)
                    .count(),
                "reason": "deterministic canonical pose candidate selection",
                "render_report": render_report,
            }),
        };
        Ok(run)
    }

    fn render_canonical_pose_candidate_thumbnails(
        &mut self,
        request: &SceneAssetFrameCalibrationRequest<'_>,
        run: &mut CanonicalPoseCalibrationRun,
    ) -> Value {
        if !matches!(
            request.mode,
            SceneCanonicalPoseMode::Auto
                | SceneCanonicalPoseMode::RenderSweep
                | SceneCanonicalPoseMode::Openai
        ) {
            return json!({
                "enabled": false,
                "reason": "thumbnail rendering is only enabled for canonical_pose=auto, render-sweep, or openai",
            });
        }
        if run.reports.is_empty() {
            return json!({
                "enabled": true,
                "rendered": 0,
                "attempted": 0,
                "reason": "no asset pose reports to render",
            });
        }

        let root = request.output_dir.join("canonical_pose_renders");
        let bridge_dir = root.join("_viewer");
        let control_path = bridge_dir.join("scene_commands.json");
        let status_path = control_path.with_extension("status.json");
        let log_path = bridge_dir.join("viewer.log");
        if let Err(err) = fs::create_dir_all(&root) {
            let message = format!(
                "failed to create canonical pose render directory {}: {err}",
                root.display()
            );
            append_canonical_pose_render_warning(run, &message);
            return json!({
                "enabled": true,
                "attempted": 0,
                "rendered": 0,
                "error": message,
            });
        }

        let original_control_path = self.config.scene_control_path.clone();
        let original_status_path = self.config.scene_status_path.clone();
        let original_timeout = self.config.scene_timeout;
        let mut spawned_viewer = match spawn_feedback_viewer(&control_path, &log_path, None) {
            Ok(child) => child,
            Err(err) => {
                let message = format!("failed to spawn canonical pose thumbnail viewer: {err}");
                append_canonical_pose_render_warning(run, &message);
                return json!({
                    "enabled": true,
                    "attempted": 0,
                    "rendered": 0,
                    "root": root.display().to_string(),
                    "viewer_log": log_path.display().to_string(),
                    "error": message,
                });
            }
        };
        self.config.scene_control_path = Some(control_path);
        self.config.scene_status_path = Some(status_path);
        self.config.scene_timeout = self.config.scene_timeout.max(Duration::from_secs(60));

        let depth_map = match load_scene_depth_map_sidecar(request.evidence) {
            Ok(depth_map) => depth_map,
            Err(err) => {
                append_canonical_pose_render_warning(
                    run,
                    &format!("failed to load source depth normal sidecar: {err}"),
                );
                None
            }
        };
        let mut attempted = 0usize;
        let mut rendered = 0usize;
        let mut normal_rendered = 0usize;
        let mut source_normals = 0usize;
        let mut errors = Vec::new();
        for report_index in 0..run.reports.len() {
            let asset_id = run.reports[report_index].asset_id.clone();
            let Some(asset) = request
                .asset_bindings
                .iter()
                .find(|asset| asset.asset_id == asset_id)
            else {
                let message = format!("asset binding missing for canonical pose asset {asset_id}");
                run.reports[report_index].warnings.push(message.clone());
                errors.push(json!({
                    "asset_id": asset_id,
                    "error": message,
                }));
                continue;
            };
            let Some(spawn_basis) = CanonicalPoseThumbnailSpawnBasis::from_asset(asset) else {
                let message = format!(
                    "asset {} has no cache_key or path; cannot render canonical pose thumbnails",
                    asset.asset_id
                );
                run.reports[report_index].warnings.push(message.clone());
                errors.push(json!({
                    "asset_id": asset.asset_id,
                    "error": message,
                }));
                continue;
            };
            let asset_dir = root.join(sanitize_scene_identifier(&asset.asset_id));
            if let Err(err) = fs::create_dir_all(&asset_dir) {
                let message = format!(
                    "failed to create canonical pose render asset directory {}: {err}",
                    asset_dir.display()
                );
                run.reports[report_index].warnings.push(message.clone());
                errors.push(json!({
                    "asset_id": asset.asset_id,
                    "error": message,
                }));
                continue;
            }
            let source_normal_path = asset_dir.join("source_depth_normal.png");
            let source_normal_evidence = match write_source_depth_normal_evidence_for_asset(
                asset,
                request.evidence,
                depth_map.as_ref(),
                &source_normal_path,
            ) {
                Ok(evidence) => {
                    if evidence.is_some() {
                        source_normals += 1;
                    }
                    evidence
                }
                Err(err) => {
                    let message = format!(
                        "failed to render source depth normal evidence for {}: {err}",
                        asset.asset_id
                    );
                    run.reports[report_index].warnings.push(message.clone());
                    errors.push(json!({
                        "asset_id": asset.asset_id,
                        "stage": "source_depth_normal",
                        "error": message,
                    }));
                    None
                }
            };
            let normal_mesh = match self.load_canonical_pose_normal_mesh(asset, spawn_basis) {
                Ok(mesh) => mesh,
                Err(err) => {
                    let message = format!(
                        "failed to load mesh for canonical pose normal pass for {}: {err}",
                        asset.asset_id
                    );
                    run.reports[report_index].warnings.push(message.clone());
                    errors.push(json!({
                        "asset_id": asset.asset_id,
                        "stage": "candidate_mesh_normal",
                        "error": message,
                    }));
                    None
                }
            };
            for candidate_index in 0..run.reports[report_index].candidates.len() {
                attempted += 1;
                let candidate_id =
                    run.reports[report_index].candidates[candidate_index].candidate_index;
                let candidate_yaw =
                    run.reports[report_index].candidates[candidate_index].yaw_offset_degrees;
                let output_path = asset_dir.join(format!(
                    "candidate_{:02}_yaw_{}.png",
                    candidate_id,
                    canonical_pose_yaw_file_component(candidate_yaw)
                ));
                let normal_path = asset_dir.join(format!(
                    "candidate_{:02}_yaw_{}_normal.png",
                    candidate_id,
                    canonical_pose_yaw_file_component(candidate_yaw)
                ));
                match self.render_canonical_pose_candidate_thumbnail(
                    asset,
                    spawn_basis,
                    candidate_yaw,
                    &output_path,
                ) {
                    Ok(render_info) => {
                        rendered += 1;
                        let candidate = &mut run.reports[report_index].candidates[candidate_index];
                        candidate.rendered_image_path = Some(output_path.display().to_string());
                        candidate.metrics["rendered_asset_thumbnail"] = json!(true);
                        candidate.metrics["renderer"] = json!("bevy_synth_private_viewer");
                        candidate.metrics["render"] = render_info;
                        if let Some(mesh) = normal_mesh.as_ref() {
                            match write_render_candidate_mesh_normal(
                                cached_mesh_normal_input(mesh),
                                scene_aabb_to_render(asset.local_aabb.unwrap_or(SceneAssetAabb {
                                    min: [-0.5, 0.0, -0.5],
                                    max: [0.5, 1.0, 0.5],
                                })),
                                candidate_yaw,
                                &normal_path,
                            ) {
                                Ok(normal_render) => {
                                    normal_rendered += 1;
                                    let similarity = source_normal_evidence
                                        .as_ref()
                                        .and_then(|evidence| {
                                            evidence
                                                .get("path")
                                                .and_then(Value::as_str)
                                                .map(PathBuf::from)
                                        })
                                        .and_then(|source_path| {
                                            normal_map_similarity(&source_path, &normal_path).ok()
                                        });
                                    candidate.metrics["normal_evidence"] = json!({
                                        "source": source_normal_evidence,
                                        "candidate": normal_render,
                                        "similarity": similarity,
                                        "descriptor": "source_depth_normal_vs_rendered_mesh_normal",
                                    });
                                }
                                Err(err) => {
                                    candidate.metrics["normal_evidence_error"] = json!(err);
                                }
                            }
                        }
                    }
                    Err(err) => {
                        let message = format!(
                            "failed to render canonical pose thumbnail for {} candidate {}: {err}",
                            asset.asset_id, candidate_id
                        );
                        run.reports[report_index].warnings.push(message.clone());
                        errors.push(json!({
                            "asset_id": asset.asset_id,
                            "candidate_index": candidate_id,
                            "error": message,
                        }));
                    }
                }
            }
        }

        let _ = self.send_scene_commands(vec![json!({ "type": "clear_scene" })]);
        let _ = spawned_viewer.kill();
        let _ = spawned_viewer.wait();
        self.config.scene_control_path = original_control_path;
        self.config.scene_status_path = original_status_path;
        self.config.scene_timeout = original_timeout;
        refresh_canonical_pose_selection_inputs(run);

        json!({
            "enabled": true,
            "renderer": "bevy_synth_private_viewer",
            "root": root.display().to_string(),
            "viewer_log": log_path.display().to_string(),
            "attempted": attempted,
            "rendered": rendered,
            "normal_rendered": normal_rendered,
            "source_normals": source_normals,
            "error_count": errors.len(),
            "errors": errors,
        })
    }

    fn load_canonical_pose_normal_mesh(
        &self,
        asset: &SceneAssetBinding,
        spawn_basis: CanonicalPoseThumbnailSpawnBasis<'_>,
    ) -> Result<Option<CachedSynthMesh>, String> {
        if let CanonicalPoseThumbnailSpawnBasis::Cache(cache_key) = spawn_basis {
            let cache = self.open_catalog_cache()?;
            if let Some(mesh) = cache
                .load_mesh(cache_key)
                .map_err(|err| format!("load cached mesh {cache_key}: {err}"))?
            {
                return Ok(Some(mesh));
            }
        }
        if let Some(path) = asset.path.as_deref() {
            let bytes = fs::read(path).map_err(|err| format!("read mesh glb {path}: {err}"))?;
            let mesh = bevy_synth_runtime::io::mesh_from_glb_bytes(&bytes)
                .map_err(|err| format!("parse mesh glb {path}: {err}"))?;
            return Ok(Some(mesh));
        }
        Ok(None)
    }

    fn render_canonical_pose_candidate_thumbnail(
        &self,
        asset: &SceneAssetBinding,
        spawn_basis: CanonicalPoseThumbnailSpawnBasis<'_>,
        yaw_degrees: f32,
        output_path: &Path,
    ) -> Result<Value, String> {
        let local_aabb = asset.local_aabb.unwrap_or(SceneAssetAabb {
            min: [-0.5, 0.0, -0.5],
            max: [0.5, 1.0, 0.5],
        });
        let size = local_aabb.size();
        let center = [
            (local_aabb.min[0] + local_aabb.max[0]) * 0.5,
            (local_aabb.min[1] + local_aabb.max[1]) * 0.5,
            (local_aabb.min[2] + local_aabb.max[2]) * 0.5,
        ];
        let radius = size[0].max(size[1]).max(size[2]).max(0.75) * 2.45;
        let focus = [0.0, (size[1] * 0.48).max(0.15), 0.0];
        let translation = [-center[0], -local_aabb.min[1], -center[2]];
        let mut commands = vec![
            json!({ "type": "clear_scene" }),
            canonical_pose_thumbnail_spawn_command(
                spawn_basis,
                asset,
                translation,
                quat_from_y_degrees(yaw_degrees),
            ),
            json!({
                "type": "set_camera",
                "translation": [0.0, focus[1] + 0.25, radius],
                "rotation": [0.0, 0.0, 0.0, 1.0],
                "focus": focus,
                "yaw": 0.0,
                "pitch": 20.0,
                "radius": radius,
                "vertical_fov": 42.0,
            }),
        ];
        commands = scene_commands_with_cache_reload(commands);
        let apply_ack = self.send_scene_commands(commands)?;
        let mut last_error = None::<String>;
        let mut capture_ack = Value::Null;
        let mut status = Value::Null;
        let mut thumbnail_metrics = Value::Null;
        let mut file_size = 0u64;
        for attempt in 0..3 {
            let (attempt_capture_ack, attempt_status) =
                self.capture_feedback_when_projected_ready(&apply_ack, output_path, 1)?;
            capture_ack = attempt_capture_ack;
            status = attempt_status;
            file_size = output_path
                .metadata()
                .map(|metadata| metadata.len())
                .unwrap_or(0);
            if file_size == 0 {
                last_error = Some(format!(
                    "thumbnail file was not written: {}",
                    output_path.display()
                ));
            } else {
                match canonical_pose_thumbnail_pixel_metrics(output_path) {
                    Ok(metrics) => {
                        thumbnail_metrics = metrics;
                        last_error = None;
                        break;
                    }
                    Err(err) => {
                        last_error = Some(format!(
                            "thumbnail validation failed on attempt {}: {err}",
                            attempt + 1
                        ));
                    }
                }
            }
            thread::sleep(Duration::from_millis(FEEDBACK_RENDER_SETTLE_MS));
        }
        if let Some(err) = last_error {
            return Err(err);
        }
        Ok(json!({
            "output_path": output_path.display().to_string(),
            "yaw_degrees": yaw_degrees,
            "file_size": file_size,
            "thumbnail_metrics": thumbnail_metrics,
            "spawn_basis": spawn_basis.label(),
            "local_aabb": local_aabb,
            "camera": {
                "focus": focus,
                "radius": radius,
                "pitch": 20.0,
                "yaw": 0.0,
                "vertical_fov": 42.0,
            },
            "status": canonical_pose_thumbnail_status_summary(&status),
            "capture": canonical_pose_thumbnail_capture_summary(&capture_ack),
        }))
    }

    fn open_catalog_cache(&self) -> Result<MeshCache, String> {
        if let Some(root) = self.config.catalog_cache_root.as_ref() {
            MeshCache::load_from_root(root.clone())
        } else {
            MeshCache::load_default()
        }
        .map_err(|err| format!("failed to open shared asset cache: {err}"))
    }

    fn call_scene_status(&self) -> Result<Value, String> {
        let status_path = self
            .config
            .scene_status_path
            .as_ref()
            .ok_or_else(|| "scene_status_path is not configured".to_string())?;
        read_scene_status(status_path)
    }

    fn call_scene_project_status(&self) -> Result<Value, String> {
        let status = self.call_scene_status()?;
        Ok(json!({
            "tool": "scene_project_status",
            "camera": status.get("camera").cloned().unwrap_or(Value::Null),
            "world_items": status.get("world_items").cloned().unwrap_or(Value::Null),
            "projected_items": status.get("projected_items").cloned().unwrap_or(Value::Null),
            "screenshots": status.get("screenshots").cloned().unwrap_or(Value::Null),
            "status": status,
        }))
    }

    fn call_scene_list_assets(&self) -> Result<Value, String> {
        let status = self.call_scene_status()?;
        Ok(json!({
            "tool": "scene_list_assets",
            "cache_entries": status["cache_entries"].clone(),
            "world_items": status["world_items"].clone(),
        }))
    }

    fn call_scene_spawn_cached(&self, args: SceneSpawnCachedArgs) -> Result<Value, String> {
        self.send_scene_commands(vec![json!({
            "type": "spawn_cached",
            "cache_key": args.cache_key,
            "translation": args.translation,
            "rotation": args.rotation,
            "scale": args.scale,
            "select": args.select,
        })])
    }

    fn call_scene_spawn_path(&self, args: SceneSpawnPathArgs) -> Result<Value, String> {
        self.send_scene_commands(vec![json!({
            "type": "spawn_path",
            "path": args.path,
            "translation": args.translation,
            "rotation": args.rotation,
            "scale": args.scale,
            "select": args.select,
        })])
    }

    fn call_scene_delete(&self, args: SceneDeleteArgs) -> Result<Value, String> {
        if let Some(cache_key) = args.cache_key {
            return self.send_scene_commands(vec![json!({
                "type": "delete_by_cache_key",
                "cache_key": cache_key,
            })]);
        }
        if args.selected {
            return self.send_scene_commands(vec![json!({ "type": "delete_selected" })]);
        }
        self.send_scene_commands(vec![json!({ "type": "clear_selection" })])
    }

    fn call_scene_clear(&self) -> Result<Value, String> {
        self.send_scene_commands(vec![json!({ "type": "clear_scene" })])
    }

    fn call_scene_set_camera(&self, args: SceneSetCameraArgs) -> Result<Value, String> {
        self.send_scene_commands(vec![json!({
            "type": "set_camera",
            "translation": args.translation,
            "rotation": args.rotation,
            "focus": args.focus,
            "yaw": args.yaw,
            "pitch": args.pitch,
            "radius": args.radius,
            "vertical_fov": args.vertical_fov,
        })])
    }

    fn call_scene_capture(&self, args: SceneCaptureArgs) -> Result<Value, String> {
        let path = args.output_path;
        if path
            .metadata()
            .map(|metadata| metadata.len() > 0)
            .unwrap_or(false)
            && let Err(err) = fs::remove_file(&path)
        {
            return Err(format!(
                "failed to remove stale screenshot {} before capture: {err}",
                path.display()
            ));
        }
        let response = self.send_scene_commands(vec![json!({
            "type": "capture_screenshot",
            "path": path.display().to_string(),
        })])?;
        let timeout = self.config.scene_timeout;
        let started = Instant::now();
        while started.elapsed() < timeout {
            if path
                .metadata()
                .map(|metadata| metadata.len() > 0)
                .unwrap_or(false)
            {
                return Ok(json!({
                    "tool": "scene_capture",
                    "output_path": path.display().to_string(),
                    "acknowledgement": response,
                }));
            }
            thread::sleep(Duration::from_millis(25));
        }
        Err(format!(
            "scene_capture timed out waiting for screenshot {}",
            path.display()
        ))
    }

    fn call_scene_compose_assets(&self, args: SceneComposeArgs) -> Result<Value, String> {
        let plan = compose_scene_layout(args)?;
        let mut response =
            serde_json::to_value(&plan).map_err(|err| format!("serialize layout plan: {err}"))?;
        if plan.apply {
            let acknowledgement = self.send_scene_commands(scene_commands_from_plan(&plan)?)?;
            response["acknowledgement"] = acknowledgement;
        }
        Ok(response)
    }

    fn call_scene_validate_layout(&self, mut args: SceneValidateArgs) -> Result<Value, String> {
        if args.scene_status.is_none() {
            args.scene_status = Some(self.call_scene_status()?);
        }
        validate_scene_layout(args)
    }

    fn run_scene_composition_feedback_selection(
        &mut self,
        output_dir: &Path,
        manifest: &SceneObjectManifest,
        asset_bindings: &[SceneAssetBinding],
        candidates: Vec<SceneCompositionCandidate>,
        options: SceneFeedbackOptions,
        progress: Option<SceneFeedbackProgressSink<'_>>,
    ) -> Result<SceneCompositionFeedbackSelection, String> {
        if candidates.is_empty() {
            return Err("composition feedback selection requires candidates".to_string());
        }
        let compare_candidates = candidates.len() > 1;
        let mut reports = Vec::new();
        let mut best: Option<(f64, SceneCompositionCandidate, Vec<Value>, Value)> = None;
        for candidate in candidates {
            let capture_dir = if compare_candidates {
                let base = options
                    .capture_dir
                    .clone()
                    .unwrap_or_else(|| output_dir.join("iterations"));
                Some(base.join(scene_composition_mode_label(candidate.mode)))
            } else {
                options.capture_dir.clone()
            };
            let feedback = self.run_scene_feedback(
                output_dir,
                manifest,
                asset_bindings,
                &candidate.layout,
                candidate.commands.clone(),
                SceneFeedbackOptions {
                    max_iters: options.max_iters,
                    keep_viewer: options.keep_viewer,
                    capture_dir,
                    threshold_profile: options.threshold_profile,
                    rotation_selector: options.rotation_selector,
                    rotation_fit: options.rotation_fit,
                    rotation_fit_max_gpt_rounds: options.rotation_fit_max_gpt_rounds,
                    rotation_fit_min_mask_iou: options.rotation_fit_min_mask_iou,
                    rotation_fit_max_depth_error_m: options.rotation_fit_max_depth_error_m,
                    rotation_fit_write_artifacts: options.rotation_fit_write_artifacts,
                    rubric_scorer: options.rubric_scorer,
                    scale_policy: options.scale_policy,
                    grounding_evidence: options.grounding_evidence.clone(),
                },
                progress.clone(),
            );
            match feedback {
                Ok(mut feedback) => {
                    let selection_score = feedback_result_selection_score(&feedback);
                    feedback["composition_mode"] = json!(candidate.mode);
                    feedback["selection_score"] =
                        finite_json_f64(selection_score).unwrap_or(Value::Null);
                    let final_commands = feedback
                        .get("final_commands")
                        .and_then(Value::as_array)
                        .cloned()
                        .unwrap_or_else(|| candidate.commands.clone());
                    reports.push(json!({
                        "mode": candidate.mode,
                        "accepted": feedback.get("accepted").and_then(Value::as_bool).unwrap_or(false),
                        "best_iteration": feedback.get("best_iteration").cloned().unwrap_or(Value::Null),
                        "best_score": feedback.get("best_score").cloned().unwrap_or(Value::Null),
                        "selection_score": finite_json_f64(selection_score).unwrap_or(Value::Null),
                        "capture_dir": feedback.get("capture_dir").cloned().unwrap_or(Value::Null),
                        "feedback_report": feedback_report_path_from_result(&feedback).unwrap_or_default(),
                    }));
                    if best
                        .as_ref()
                        .is_none_or(|(best_score, _, _, _)| selection_score > *best_score)
                    {
                        best = Some((selection_score, candidate, final_commands, feedback));
                    }
                }
                Err(err) => {
                    reports.push(json!({
                        "mode": candidate.mode,
                        "accepted": false,
                        "selection_score": Value::Null,
                        "error": err,
                    }));
                }
            }
        }
        let Some((selection_score, candidate, commands, mut feedback)) = best else {
            return Err(format!(
                "all composition feedback candidates failed: {}",
                serde_json::to_string(&reports).unwrap_or_else(|_| "<unserializable>".to_string())
            ));
        };
        feedback["candidate_selection"] = json!({
            "selected_mode": candidate.mode,
            "selected_score": finite_json_f64(selection_score).unwrap_or(Value::Null),
            "compared_candidates": compare_candidates,
            "candidates": reports.clone(),
        });
        Ok(SceneCompositionFeedbackSelection {
            candidate,
            commands,
            feedback,
            candidate_reports: reports,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_scene_feedback(
        &mut self,
        output_dir: &Path,
        manifest: &SceneObjectManifest,
        asset_bindings: &[SceneAssetBinding],
        grounded_layout: &GroundedSceneLayout,
        initial_commands: Vec<Value>,
        options: SceneFeedbackOptions,
        progress: Option<SceneFeedbackProgressSink<'_>>,
    ) -> Result<Value, String> {
        let original_control_path = self.config.scene_control_path.clone();
        let original_status_path = self.config.scene_status_path.clone();
        let original_timeout = self.config.scene_timeout;
        let capture_root = options
            .capture_dir
            .clone()
            .unwrap_or_else(|| output_dir.join("iterations"));
        fs::create_dir_all(&capture_root).map_err(|err| {
            format!(
                "failed to create feedback capture directory {}: {err}",
                capture_root.display()
            )
        })?;

        let mut spawned_viewer = None;
        if self.config.scene_control_path.is_none() {
            let bridge_dir = output_dir.join("feedback_viewer");
            fs::create_dir_all(&bridge_dir).map_err(|err| {
                format!(
                    "failed to create feedback viewer directory {}: {err}",
                    bridge_dir.display()
                )
            })?;
            let control_path = bridge_dir.join("scene_commands.json");
            let status_path = bridge_dir.join("scene_commands.status.json");
            let log_path = bridge_dir.join("viewer.log");
            spawned_viewer = Some(spawn_feedback_viewer(
                &control_path,
                &log_path,
                scene_capture_window_size_for_source(&manifest.source_scene_path),
            )?);
            self.config.scene_control_path = Some(control_path);
            self.config.scene_status_path = Some(status_path);
            self.config.scene_timeout = self.config.scene_timeout.max(Duration::from_secs(60));
        }

        let lock_result = self.send_scene_commands(vec![scene_interaction_lock_command(
            true,
            "iterative scene composition",
        )]);
        let feedback_result = match lock_result {
            Ok(lock_ack) => {
                let _ = write_json_file(&capture_root.join("interaction_lock_ack.json"), &lock_ack);
                let mut result = self.run_scene_feedback_iterations(
                    SceneFeedbackIterationContext {
                        capture_root: &capture_root,
                        manifest,
                        asset_bindings,
                        grounded_layout,
                        initial_commands,
                        max_iters: options.max_iters.max(1),
                        threshold_profile: options.threshold_profile,
                        rotation_selector: options.rotation_selector,
                        rotation_fit: options.rotation_fit,
                        rotation_fit_max_gpt_rounds: options.rotation_fit_max_gpt_rounds,
                        rotation_fit_min_mask_iou: options.rotation_fit_min_mask_iou,
                        rotation_fit_max_depth_error_m: options.rotation_fit_max_depth_error_m,
                        rotation_fit_write_artifacts: options.rotation_fit_write_artifacts,
                        rubric_scorer: options.rubric_scorer,
                        scale_policy: options.scale_policy,
                        grounding_evidence: options.grounding_evidence.as_ref(),
                    },
                    progress.clone(),
                );
                if let Ok(value) = &mut result {
                    value["interaction_lock_ack"] = lock_ack;
                }
                result
            }
            Err(err) => Err(format!("failed to lock scene interaction: {err}")),
        };
        let unlock_result =
            self.send_scene_commands(vec![scene_interaction_lock_command(false, "")]);
        let feedback_result = match (feedback_result, unlock_result) {
            (Ok(mut value), Ok(unlock_ack)) => {
                let _ = write_json_file(
                    &capture_root.join("interaction_unlock_ack.json"),
                    &unlock_ack,
                );
                value["interaction_unlock_ack"] = unlock_ack;
                Ok(value)
            }
            (Ok(_), Err(unlock_err)) => Err(format!(
                "feedback completed but failed to unlock scene interaction: {unlock_err}"
            )),
            (Err(err), Ok(unlock_ack)) => {
                let _ = write_json_file(
                    &capture_root.join("interaction_unlock_ack.json"),
                    &unlock_ack,
                );
                Err(err)
            }
            (Err(err), Err(unlock_err)) => Err(format!(
                "{err}; additionally failed to unlock scene interaction: {unlock_err}"
            )),
        };

        if let Some(mut child) = spawned_viewer
            && !options.keep_viewer
        {
            let _ = child.kill();
            let _ = child.wait();
        }
        self.config.scene_control_path = original_control_path;
        self.config.scene_status_path = original_status_path;
        self.config.scene_timeout = original_timeout;

        feedback_result
    }

    fn run_scene_feedback_iterations(
        &self,
        context: SceneFeedbackIterationContext<'_>,
        progress: Option<SceneFeedbackProgressSink<'_>>,
    ) -> Result<Value, String> {
        let SceneFeedbackIterationContext {
            capture_root,
            manifest,
            asset_bindings,
            grounded_layout,
            initial_commands,
            max_iters,
            threshold_profile,
            rotation_selector,
            rotation_fit,
            rotation_fit_max_gpt_rounds,
            rotation_fit_min_mask_iou,
            rotation_fit_max_depth_error_m,
            rotation_fit_write_artifacts,
            rubric_scorer,
            scale_policy,
            grounding_evidence,
        } = context;
        let mut commands = scene_commands_with_asset_local_aabbs(initial_commands, asset_bindings);
        let mut feedback_layout = grounded_layout.clone();
        let mut rotation_fit_report = Value::Null;
        if rotation_fit != SceneRotationFitMode::Off {
            let rotation_fit_dir = capture_root.join("rotation_fit");
            let fit = apply_scene_rotation_fit(
                SceneRotationFitConfig {
                    mode: rotation_fit,
                    max_gpt_rounds: rotation_fit_max_gpt_rounds,
                    min_mask_iou: rotation_fit_min_mask_iou,
                    max_depth_error_m: rotation_fit_max_depth_error_m,
                    write_artifacts: rotation_fit_write_artifacts,
                    output_dir: &rotation_fit_dir,
                },
                manifest,
                asset_bindings,
                grounding_evidence,
                &feedback_layout,
                &commands,
            )?;
            commands = scene_commands_with_asset_local_aabbs(fit.commands, asset_bindings);
            feedback_layout = fit.grounded_layout;
            rotation_fit_report = fit.report;
            write_json_file(
                &capture_root.join("commands.rotation_fit.json"),
                &json!(commands),
            )
            .map_err(|err| err.to_string())?;
            write_json_file(
                &capture_root.join("grounded_layout.rotation_fit.json"),
                &json!(feedback_layout),
            )
            .map_err(|err| err.to_string())?;
        }
        let grounded_layout = &feedback_layout;
        let thresholds = threshold_profile.thresholds();
        let mut iterations = Vec::new();
        let mut accepted_iteration = None;
        let mut best_iteration = None;
        let mut best_score = f64::NEG_INFINITY;
        let mut best_commands = commands.clone();
        let mut previous_iteration_snapshot: Option<Value> = None;
        let mut previous_commands: Option<Vec<Value>> = None;
        for iteration_index in 0..max_iters {
            let iteration_rotation_selector = if rotation_fit == SceneRotationFitMode::GptRefine
                && iteration_index < rotation_fit_max_gpt_rounds
            {
                FeedbackRotationSelector::Openai
            } else {
                rotation_selector
            };
            let iteration_dir = capture_root.join(format!("iter_{iteration_index:02}"));
            fs::create_dir_all(&iteration_dir).map_err(|err| {
                format!(
                    "failed to create feedback iteration directory {}: {err}",
                    iteration_dir.display()
                )
            })?;
            write_json_file(&iteration_dir.join("commands.json"), &json!(commands))
                .map_err(|err| err.to_string())?;
            let iteration_bsn =
                feedback_bsn_from_commands(asset_bindings, grounded_layout, &commands)?;
            let iteration_bsn_path = iteration_dir.join("scene.bsn");
            fs::write(&iteration_bsn_path, iteration_bsn).map_err(|err| {
                format!(
                    "failed to write feedback BSN {}: {err}",
                    iteration_bsn_path.display()
                )
            })?;
            emit_scene_feedback_progress(
                &progress,
                "render_capture_feedback.bsn",
                SceneBuildProgressPhase::Completed,
                SceneBuildExecutionKind::FileIo,
                format!("wrote iteration {iteration_index} BSN"),
                Some(iteration_index + 1),
                Some(max_iters),
                Some(iteration_bsn_path.clone()),
                json!({
                    "iteration": iteration_index,
                    "scene_bsn": iteration_bsn_path,
                }),
            );

            let apply_ack = self.send_scene_commands(commands.clone())?;
            write_json_file(&iteration_dir.join("apply_ack.json"), &apply_ack)
                .map_err(|err| err.to_string())?;
            let screenshot_path = iteration_dir.join("screenshot.png");
            let (capture, status) = self.capture_feedback_when_projected_ready(
                &apply_ack,
                &screenshot_path,
                grounded_layout.placements.len(),
            )?;
            write_json_file(&iteration_dir.join("capture_ack.json"), &capture)
                .map_err(|err| err.to_string())?;
            write_json_file(&iteration_dir.join("status.json"), &status)
                .map_err(|err| err.to_string())?;
            emit_scene_feedback_progress(
                &progress,
                "render_capture_feedback.screenshot",
                SceneBuildProgressPhase::Completed,
                SceneBuildExecutionKind::Viewer,
                format!("captured iteration {iteration_index} full-scene render"),
                Some(iteration_index + 1),
                Some(max_iters),
                Some(screenshot_path.clone()),
                json!({
                    "iteration": iteration_index,
                    "screenshot": screenshot_path,
                    "projected_items": status
                        .get("projected_items")
                        .and_then(Value::as_array)
                        .map(Vec::len)
                        .unwrap_or(0),
                }),
            );
            let mut metrics = scene_feedback_metrics(
                manifest,
                grounded_layout,
                &status,
                &screenshot_path,
                thresholds,
                threshold_profile,
            )?;
            let mut object_crops = feedback_object_crops(
                &iteration_dir,
                Path::new(&manifest.source_scene_path),
                &screenshot_path,
                &metrics,
            );
            let rotation_render_report =
                if feedback_selector_uses_rendered_candidates(iteration_rotation_selector) {
                    self.render_feedback_rotation_candidate_crops(
                        &iteration_dir,
                        &commands,
                        &mut metrics,
                        &mut object_crops,
                        &progress,
                    )?
                } else {
                    Value::Null
                };
            if !rotation_render_report.is_null() {
                metrics["rotation_candidate_render"] = rotation_render_report.clone();
                write_json_file(
                    &iteration_dir.join("rotation_candidate_render.json"),
                    &rotation_render_report,
                )
                .map_err(|err| err.to_string())?;
            }
            if !object_crops.is_null() {
                metrics["object_crops"] = object_crops.clone();
                write_json_file(&iteration_dir.join("object_crops.json"), &object_crops)
                    .map_err(|err| err.to_string())?;
                let crop_artifact = object_crops
                    .get("dir")
                    .and_then(Value::as_str)
                    .map(PathBuf::from)
                    .unwrap_or_else(|| iteration_dir.join("object_crops.json"));
                emit_scene_feedback_progress(
                    &progress,
                    "render_capture_feedback.object_crops",
                    SceneBuildProgressPhase::Completed,
                    SceneBuildExecutionKind::FileIo,
                    format!("wrote iteration {iteration_index} per-object crops"),
                    Some(iteration_index + 1),
                    Some(max_iters),
                    Some(crop_artifact),
                    json!({
                        "iteration": iteration_index,
                        "object_count": object_crops
                            .get("objects")
                            .and_then(Value::as_array)
                            .map(Vec::len)
                            .unwrap_or(0),
                    }),
                );
            }
            let rotation_selection_task = feedback_rotation_selection_task(&metrics, &object_crops);
            if !rotation_selection_task.is_null() {
                write_json_file(
                    &iteration_dir.join("rotation_selection_task.json"),
                    &rotation_selection_task,
                )
                .map_err(|err| err.to_string())?;
            }
            let rotation_selection_report = self.apply_feedback_rotation_selector(
                iteration_rotation_selector,
                &iteration_dir,
                &rotation_selection_task,
                &mut metrics,
            )?;
            if !rotation_selection_report.is_null() {
                metrics["rotation_selector"] = rotation_selection_report.clone();
                write_json_file(
                    &iteration_dir.join("rotation_selection_report.json"),
                    &rotation_selection_report,
                )
                .map_err(|err| err.to_string())?;
            }
            let rubric_report = self.apply_feedback_rubric_scorer(
                rubric_scorer,
                &iteration_dir,
                Path::new(&manifest.source_scene_path),
                &screenshot_path,
                manifest,
                grounded_layout,
                &metrics,
                Some(iteration_index),
            )?;
            if !rubric_report.is_null() {
                metrics["scene_quality_rubric"] = rubric_report.clone();
                apply_feedback_scene_quality_rubric_gate(&mut metrics);
                write_json_file(
                    &iteration_dir.join("scene_quality_rubric.json"),
                    &rubric_report,
                )
                .map_err(|err| err.to_string())?;
            }
            write_json_file(&iteration_dir.join("metrics.json"), &metrics)
                .map_err(|err| err.to_string())?;
            let passed = metrics
                .get("passed")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let selection_score = feedback_selection_score(&metrics);
            if selection_score > best_score {
                best_score = selection_score;
                best_iteration = Some(iteration_index);
                best_commands = commands.clone();
            }
            let deltas = feedback_layout_deltas_with_policy(&metrics, scale_policy);
            write_json_file(&iteration_dir.join("layout_delta.json"), &deltas)
                .map_err(|err| err.to_string())?;
            let iteration_context = feedback_iteration_context(
                iteration_index,
                previous_iteration_snapshot.as_ref(),
                previous_commands.as_deref(),
                &commands,
                &screenshot_path,
                &metrics,
                &deltas,
                &object_crops,
            );
            write_json_file(
                &iteration_dir.join("iteration_context.json"),
                &iteration_context,
            )
            .map_err(|err| err.to_string())?;
            iterations.push(json!({
                "iteration": iteration_index,
                "dir": iteration_dir.display().to_string(),
                "screenshot": screenshot_path.display().to_string(),
                "metrics": metrics.clone(),
                "layout_delta": deltas.clone(),
                "object_crops": object_crops.clone(),
                "iteration_context": iteration_context.clone(),
                "passed": passed,
                "selection_score": selection_score,
            }));
            if passed {
                accepted_iteration = Some(iteration_index);
                break;
            }
            previous_commands = Some(commands.clone());
            previous_iteration_snapshot = iterations.last().cloned();
            commands =
                apply_feedback_deltas_to_commands_with_policy(&commands, &deltas, scale_policy)?;
        }
        let mut final_evidence = Value::Null;
        let mut final_accepted = false;
        if accepted_iteration.is_none() && max_iters > 0 {
            let final_dir = capture_root.join("final");
            fs::create_dir_all(&final_dir).map_err(|err| {
                format!(
                    "failed to create final feedback directory {}: {err}",
                    final_dir.display()
                )
            })?;
            write_json_file(&final_dir.join("commands.json"), &json!(commands))
                .map_err(|err| err.to_string())?;
            let final_bsn = feedback_bsn_from_commands(asset_bindings, grounded_layout, &commands)?;
            fs::write(final_dir.join("scene.bsn"), &final_bsn).map_err(|err| {
                format!(
                    "failed to write final feedback BSN {}: {err}",
                    final_dir.join("scene.bsn").display()
                )
            })?;
            let apply_ack = self.send_scene_commands(commands.clone())?;
            write_json_file(&final_dir.join("apply_ack.json"), &apply_ack)
                .map_err(|err| err.to_string())?;
            thread::sleep(Duration::from_millis(FEEDBACK_RENDER_SETTLE_MS));
            let screenshot_path = final_dir.join("screenshot.png");
            let capture = self.call_scene_capture(SceneCaptureArgs {
                output_path: screenshot_path.clone(),
            })?;
            write_json_file(&final_dir.join("capture_ack.json"), &capture)
                .map_err(|err| err.to_string())?;
            let status = Self::feedback_capture_status(&apply_ack, &capture);
            write_json_file(&final_dir.join("status.json"), &status)
                .map_err(|err| err.to_string())?;
            let mut metrics = scene_feedback_metrics(
                manifest,
                grounded_layout,
                &status,
                &screenshot_path,
                thresholds,
                threshold_profile,
            )?;
            let rubric_report = self.apply_feedback_rubric_scorer(
                rubric_scorer,
                &final_dir,
                Path::new(&manifest.source_scene_path),
                &screenshot_path,
                manifest,
                grounded_layout,
                &metrics,
                None,
            )?;
            if !rubric_report.is_null() {
                metrics["scene_quality_rubric"] = rubric_report.clone();
                apply_feedback_scene_quality_rubric_gate(&mut metrics);
                write_json_file(&final_dir.join("scene_quality_rubric.json"), &rubric_report)
                    .map_err(|err| err.to_string())?;
            }
            write_json_file(&final_dir.join("metrics.json"), &metrics)
                .map_err(|err| err.to_string())?;
            let passed = metrics
                .get("passed")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let selection_score = feedback_selection_score(&metrics);
            let selected = passed || selection_score >= best_score;
            final_evidence = json!({
                "dir": final_dir.display().to_string(),
                "screenshot": screenshot_path.display().to_string(),
                "metrics": metrics,
                "passed": passed,
                "selection_score": selection_score,
                "selected": selected,
            });
            if selected {
                final_accepted = passed;
                if selection_score > best_score {
                    best_score = selection_score;
                }
            } else if best_iteration.is_some() {
                commands = best_commands;
            }
        } else if accepted_iteration.is_none() && best_iteration.is_some() {
            commands = best_commands;
        }
        let final_bsn = feedback_bsn_from_commands(asset_bindings, grounded_layout, &commands)?;
        let final_bsn_path = capture_root.join("scene.feedback.bsn");
        fs::write(&final_bsn_path, &final_bsn).map_err(|err| {
            format!(
                "failed to write final feedback BSN {}: {err}",
                final_bsn_path.display()
            )
        })?;
        emit_scene_feedback_progress(
            &progress,
            "render_capture_feedback.final_bsn",
            SceneBuildProgressPhase::Completed,
            SceneBuildExecutionKind::FileIo,
            "wrote selected feedback BSN",
            None,
            Some(max_iters),
            Some(final_bsn_path.clone()),
            json!({
                "accepted_iteration": accepted_iteration,
                "best_iteration": best_iteration,
                "final_bsn": final_bsn_path,
            }),
        );
        write_json_file(
            &capture_root.join("commands.feedback.json"),
            &json!(commands),
        )
        .map_err(|err| err.to_string())?;
        let report = feedback_markdown_report(
            capture_root,
            threshold_profile,
            accepted_iteration,
            &iterations,
        );
        fs::write(capture_root.join("feedback_report.md"), report).map_err(|err| {
            format!(
                "failed to write feedback markdown report {}: {err}",
                capture_root.join("feedback_report.md").display()
            )
        })?;
        let rotation_report = feedback_rotation_html_report(
            capture_root,
            threshold_profile,
            accepted_iteration,
            &iterations,
        );
        fs::write(
            capture_root.join("feedback_rotation_report.html"),
            rotation_report,
        )
        .map_err(|err| {
            format!(
                "failed to write feedback rotation html report {}: {err}",
                capture_root.join("feedback_rotation_report.html").display()
            )
        })?;
        Ok(json!({
            "tool": "scene_render_capture_feedback",
            "enabled": true,
            "threshold_profile": threshold_profile,
            "rotation_selector": rotation_selector,
            "rotation_fit": rotation_fit,
            "rotation_fit_report": rotation_fit_report,
            "rubric_scorer": rubric_scorer,
            "max_iters": max_iters,
            "accepted": accepted_iteration.is_some() || final_accepted,
            "accepted_iteration": accepted_iteration,
            "accepted_final": final_accepted,
            "best_iteration": best_iteration,
            "best_score": if best_score.is_finite() { Value::from(best_score) } else { Value::Null },
            "capture_dir": capture_root.display().to_string(),
            "rotation_report_path": capture_root.join("feedback_rotation_report.html").display().to_string(),
            "iterations": iterations,
            "final_evidence": final_evidence,
            "final_bsn_path": capture_root.join("scene.feedback.bsn").display().to_string(),
            "final_commands_path": capture_root.join("commands.feedback.json").display().to_string(),
            "final_commands": commands,
        }))
    }

    fn capture_feedback_when_projected_ready(
        &self,
        apply_ack: &Value,
        screenshot_path: &Path,
        expected_items: usize,
    ) -> Result<(Value, Value), String> {
        let timeout = self.config.scene_timeout.max(Duration::from_secs(10));
        let started = Instant::now();
        let mut attempt = 0usize;
        thread::sleep(Duration::from_millis(FEEDBACK_RENDER_SETTLE_MS));
        loop {
            let capture = self.call_scene_capture(SceneCaptureArgs {
                output_path: screenshot_path.to_path_buf(),
            })?;
            let status = Self::feedback_capture_status(apply_ack, &capture);
            if Self::feedback_status_projected_items_ready(&status, expected_items) {
                return Ok((capture, status));
            }
            if started.elapsed() >= timeout {
                return Ok((capture, status));
            }
            attempt = attempt.saturating_add(1);
            let sleep_ms = if attempt < 4 { 250 } else { 500 };
            thread::sleep(Duration::from_millis(sleep_ms));
        }
    }

    fn capture_feedback_when_projected_ready_with_foreground(
        &self,
        apply_ack: &Value,
        screenshot_path: &Path,
        expected_items: usize,
        min_foreground_ratio: f32,
    ) -> Result<(Value, Value, Value), String> {
        let timeout = self.config.scene_timeout.max(Duration::from_secs(10));
        let started = Instant::now();
        let mut attempt = 0usize;
        thread::sleep(Duration::from_millis(FEEDBACK_RENDER_SETTLE_MS));
        loop {
            let capture = self.call_scene_capture(SceneCaptureArgs {
                output_path: screenshot_path.to_path_buf(),
            })?;
            let status = Self::feedback_capture_status(apply_ack, &capture);
            let foreground =
                final_yaw_render_foreground_report(screenshot_path, min_foreground_ratio);
            let ready = Self::feedback_status_projected_items_ready(&status, expected_items)
                && foreground
                    .get("ok")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
            if ready || started.elapsed() >= timeout {
                return Ok((capture, status, foreground));
            }
            attempt = attempt.saturating_add(1);
            let sleep_ms = if attempt < 4 { 250 } else { 500 };
            thread::sleep(Duration::from_millis(sleep_ms));
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_final_context_yaw_refinement(
        &mut self,
        output_dir: &Path,
        manifest: &SceneObjectManifest,
        asset_bindings: &[SceneAssetBinding],
        grounded_layout: &GroundedSceneLayout,
        commands: &[Value],
        grounding_evidence: Option<&SceneGroundingEvidence>,
        prior_pose_report: Option<&Value>,
        mode: SceneFinalYawRefinementMode,
        object_set: SceneObjectPoseRefinementSet,
        confidence_threshold: f32,
        max_candidates: usize,
        write_artifacts: bool,
    ) -> Result<SceneRotationFitOutcome, String> {
        let config = SceneFinalYawRefinementConfig {
            mode,
            object_set,
            confidence_threshold,
            max_candidates,
            write_artifacts,
            output_dir,
            grounding_evidence,
            rendered_selection_task: None,
        };
        let mut prepared = apply_scene_final_yaw_refinement(
            config,
            manifest,
            asset_bindings,
            grounded_layout,
            commands,
            prior_pose_report,
            None,
        )?;
        let task = prepared
            .report
            .get("selection_task")
            .cloned()
            .unwrap_or(Value::Null);
        if task.is_null() {
            return Ok(prepared);
        }
        if mode == SceneFinalYawRefinementMode::MetricBest {
            if let Some(response) = final_yaw_metric_best_response(&task) {
                let mut applied = apply_scene_final_yaw_refinement(
                    SceneFinalYawRefinementConfig {
                        rendered_selection_task: None,
                        ..config
                    },
                    manifest,
                    asset_bindings,
                    grounded_layout,
                    commands,
                    prior_pose_report,
                    Some(&response),
                )?;
                applied.report["selection_task"] = task;
                applied.report["selection_status"] = json!("metric_best_applied_without_rendering");
                applied.report["selection_response"] =
                    serde_json::to_value(response).map_err(|err| err.to_string())?;
                if write_artifacts {
                    write_json_file(
                        &output_dir.join("selection_response.metric_best.json"),
                        &applied.report["selection_response"],
                    )
                    .map_err(|err| err.to_string())?;
                    write_json_file(
                        &output_dir.join("final_yaw_refinement_report.json"),
                        &applied.report,
                    )
                    .map_err(|err| err.to_string())?;
                }
                return Ok(applied);
            }
            prepared.report["selection_status"] =
                json!("metric_best_unavailable_or_not_safe_without_rendering");
            if write_artifacts {
                write_json_file(
                    &output_dir.join("final_yaw_refinement_report.json"),
                    &prepared.report,
                )
                .map_err(|err| err.to_string())?;
            }
            return Ok(prepared);
        }
        if !mode.requires_selection() {
            return Ok(prepared);
        }
        let original_control_path = self.config.scene_control_path.clone();
        let original_status_path = self.config.scene_status_path.clone();
        let original_timeout = self.config.scene_timeout;
        let mut spawned_viewer = None;
        if self.config.scene_control_path.is_none() {
            let bridge_dir = output_dir.join("_viewer");
            fs::create_dir_all(&bridge_dir).map_err(|err| {
                format!(
                    "failed to create final-yaw viewer directory {}: {err}",
                    bridge_dir.display()
                )
            })?;
            let control_path = bridge_dir.join("scene_commands.json");
            let status_path = control_path.with_extension("status.json");
            let log_path = bridge_dir.join("viewer.log");
            let window_size = scene_capture_window_size_for_source(&manifest.source_scene_path);
            match spawn_feedback_viewer(&control_path, &log_path, window_size) {
                Ok(child) => {
                    prepared.report["private_viewer"] = json!({
                        "enabled": true,
                        "control_path": control_path.display().to_string(),
                        "status_path": status_path.display().to_string(),
                        "log_path": log_path.display().to_string(),
                        "window_size": window_size,
                    });
                    spawned_viewer = Some(child);
                    self.config.scene_control_path = Some(control_path);
                    self.config.scene_status_path = Some(status_path);
                    self.config.scene_timeout =
                        self.config.scene_timeout.max(Duration::from_secs(60));
                }
                Err(err) => {
                    prepared.report["selection_status"] = json!("private_viewer_spawn_failed");
                    prepared.report["selection_error"] = json!(err);
                    if write_artifacts {
                        write_json_file(
                            &output_dir.join("final_yaw_refinement_report.json"),
                            &prepared.report,
                        )
                        .map_err(|err| err.to_string())?;
                    }
                    return Ok(prepared);
                }
            }
        }

        let result = self.run_final_context_yaw_refinement_with_scene_bridge(
            output_dir,
            manifest,
            asset_bindings,
            grounded_layout,
            commands,
            prior_pose_report,
            config,
            prepared,
            task,
        );

        if let Some(mut child) = spawned_viewer {
            let _ = self.send_scene_commands_unacknowledged(vec![json!({ "type": "clear_scene" })]);
            let _ = child.kill();
            let _ = child.wait();
        }
        self.config.scene_control_path = original_control_path;
        self.config.scene_status_path = original_status_path;
        self.config.scene_timeout = original_timeout;

        result
    }

    #[allow(clippy::too_many_arguments)]
    fn run_final_context_yaw_refinement_with_scene_bridge(
        &self,
        output_dir: &Path,
        manifest: &SceneObjectManifest,
        asset_bindings: &[SceneAssetBinding],
        grounded_layout: &GroundedSceneLayout,
        commands: &[Value],
        prior_pose_report: Option<&Value>,
        config: SceneFinalYawRefinementConfig<'_>,
        mut prepared: SceneRotationFitOutcome,
        mut task: Value,
    ) -> Result<SceneRotationFitOutcome, String> {
        let write_artifacts = config.write_artifacts;
        let render_report =
            self.render_final_context_yaw_candidates(output_dir, commands, &mut task)?;
        prepared.report["candidate_render_report"] = render_report.clone();
        if write_artifacts {
            match write_final_yaw_source_reference_image(
                &manifest.source_scene_path,
                &task,
                output_dir,
            ) {
                Ok(Some(path)) => {
                    let path = path.display().to_string();
                    task["source_reference_image"] = json!(path);
                }
                Ok(None) => {}
                Err(err) => {
                    prepared.report["source_reference_warning"] = json!(err);
                }
            }
            match write_final_yaw_candidate_contact_sheets(&mut task, output_dir) {
                Ok(paths) => {
                    prepared.report["contact_sheets"] = json!(
                        paths
                            .iter()
                            .map(|path| path.display().to_string())
                            .collect::<Vec<_>>()
                    );
                }
                Err(err) => {
                    prepared.report["contact_sheet_warning"] = json!(err);
                }
            }
        }
        prepared.report["selection_task"] = task.clone();
        if write_artifacts {
            write_json_file(&output_dir.join("selection_task.rendered.json"), &task)
                .map_err(|err| err.to_string())?;
        }
        if render_report
            .get("rendered")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            == 0
        {
            prepared.report["selection_status"] = json!("rendering_unavailable");
            if let Some(response) = final_yaw_metric_best_response(&task) {
                let rendered_config = SceneFinalYawRefinementConfig {
                    rendered_selection_task: Some(&task),
                    ..config
                };
                let mut applied = apply_scene_final_yaw_refinement(
                    rendered_config,
                    manifest,
                    asset_bindings,
                    grounded_layout,
                    commands,
                    prior_pose_report,
                    Some(&response),
                )?;
                applied.report["candidate_render_report"] = render_report;
                applied.report["selection_task"] = task;
                applied.report["selection_status"] =
                    json!("rendering_unavailable_safe_metric_best_applied");
                applied.report["selection_response"] =
                    serde_json::to_value(response).map_err(|err| err.to_string())?;
                if write_artifacts {
                    write_json_file(
                        &output_dir.join("selection_response.metric_best.json"),
                        &applied.report["selection_response"],
                    )
                    .map_err(|err| err.to_string())?;
                    write_json_file(
                        &output_dir.join("final_yaw_refinement_report.json"),
                        &applied.report,
                    )
                    .map_err(|err| err.to_string())?;
                }
                return Ok(applied);
            }
            prepared.report["selection_status"] =
                json!("rendering_unavailable_no_safe_metric_fallback");
            prepared.report["selection_blocker"] = json!(
                "final contextual renders are required for failed or ambiguous large-object candidates"
            );
            if write_artifacts {
                write_json_file(
                    &output_dir.join("final_yaw_refinement_report.json"),
                    &prepared.report,
                )
                .map_err(|err| err.to_string())?;
            }
            return Ok(prepared);
        }

        let prompt = final_yaw_refinement_prompt(&task);
        let image_paths = final_yaw_refinement_image_paths(&task);
        if write_artifacts {
            write_json_file(
                &output_dir.join("selection_request.json"),
                &json!({
                    "prompt": prompt,
                    "task": task,
                    "image_paths": image_paths
                        .iter()
                        .map(|path| path.display().to_string())
                        .collect::<Vec<_>>(),
                }),
            )
            .map_err(|err| err.to_string())?;
        }
        let mut response = match self.openai_provider().and_then(|provider| {
            provider
                .select_rotation_candidates(&SceneRotationSelectionRequest {
                    prompt: final_yaw_refinement_prompt(&task),
                    task: task.clone(),
                    image_paths,
                })
                .map_err(|err| err.to_string())
        }) {
            Ok(response) => response,
            Err(err) => {
                prepared.report["selection_status"] = json!("openai_unavailable");
                prepared.report["selection_error"] = json!(err);
                return Ok(prepared);
            }
        };
        if write_artifacts {
            write_json_file(
                &output_dir.join("selection_response.json"),
                &serde_json::to_value(&response).map_err(|err| err.to_string())?,
            )
            .map_err(|err| err.to_string())?;
        }
        let coarse_visual_response = final_yaw_visual_best_response(&task, &response);
        let coarse_visual_response_value =
            serde_json::to_value(&coarse_visual_response).map_err(|err| err.to_string())?;
        let coarse_response_value =
            serde_json::to_value(&response).map_err(|err| err.to_string())?;
        let visual_gate_changed = coarse_visual_response_value != coarse_response_value;
        if visual_gate_changed {
            prepared.report["visual_gate_response"] = coarse_visual_response_value.clone();
            prepared.report["visual_gate_previous_response"] = coarse_response_value;
            prepared.report["visual_gate_stage"] = json!("coarse");
            response = coarse_visual_response;
            if write_artifacts {
                write_json_file(
                    &output_dir.join("selection_response.visual_gate.json"),
                    &coarse_visual_response_value,
                )
                .map_err(|err| err.to_string())?;
            }
        }
        let mut fine_render_report = Value::Null;
        let fine_candidate_count = final_yaw_add_fine_refinement_candidates(&mut task, &response);
        if fine_candidate_count > 0 {
            fine_render_report =
                self.render_final_context_yaw_candidates(output_dir, commands, &mut task)?;
            if write_artifacts {
                match write_final_yaw_candidate_contact_sheets(&mut task, output_dir) {
                    Ok(paths) => {
                        prepared.report["fine_contact_sheets"] = json!(
                            paths
                                .iter()
                                .map(|path| path.display().to_string())
                                .collect::<Vec<_>>()
                        );
                    }
                    Err(err) => {
                        prepared.report["fine_contact_sheet_warning"] = json!(err);
                    }
                }
            }
            let fine_prompt = final_yaw_refinement_prompt(&task);
            let fine_image_paths = final_yaw_refinement_image_paths(&task);
            if write_artifacts {
                write_json_file(&output_dir.join("selection_task.fine.rendered.json"), &task)
                    .map_err(|err| err.to_string())?;
                write_json_file(
                    &output_dir.join("selection_request.fine.json"),
                    &json!({
                        "prompt": fine_prompt,
                        "task": task,
                        "image_paths": fine_image_paths
                            .iter()
                            .map(|path| path.display().to_string())
                            .collect::<Vec<_>>(),
                        "fine_candidate_count": fine_candidate_count,
                    }),
                )
                .map_err(|err| err.to_string())?;
            }
            match self.openai_provider().and_then(|provider| {
                provider
                    .select_rotation_candidates(&SceneRotationSelectionRequest {
                        prompt: final_yaw_refinement_prompt(&task),
                        task: task.clone(),
                        image_paths: fine_image_paths,
                    })
                    .map_err(|err| err.to_string())
            }) {
                Ok(fine_response) => {
                    response = fine_response;
                    if write_artifacts {
                        write_json_file(
                            &output_dir.join("selection_response.fine.json"),
                            &serde_json::to_value(&response).map_err(|err| err.to_string())?,
                        )
                        .map_err(|err| err.to_string())?;
                    }
                }
                Err(err) => {
                    prepared.report["fine_selection_warning"] = json!(err);
                }
            }
            let visual_response = final_yaw_visual_best_response(&task, &response);
            let visual_response_value =
                serde_json::to_value(&visual_response).map_err(|err| err.to_string())?;
            let response_value = serde_json::to_value(&response).map_err(|err| err.to_string())?;
            if visual_response_value != response_value {
                prepared.report["visual_gate_response"] = visual_response_value.clone();
                prepared.report["visual_gate_previous_response"] = response_value;
                prepared.report["visual_gate_stage"] = json!("fine");
                response = visual_response;
                if write_artifacts {
                    write_json_file(
                        &output_dir.join("selection_response.visual_gate.json"),
                        &visual_response_value,
                    )
                    .map_err(|err| err.to_string())?;
                }
            }
        }
        let rendered_config = SceneFinalYawRefinementConfig {
            rendered_selection_task: Some(&task),
            ..config
        };
        let mut applied = apply_scene_final_yaw_refinement(
            rendered_config,
            manifest,
            asset_bindings,
            grounded_layout,
            commands,
            prior_pose_report,
            Some(&response),
        )?;
        applied.report["candidate_render_report"] = json!({
            "coarse": render_report,
            "fine": fine_render_report,
            "fine_candidate_count": fine_candidate_count,
        });
        applied.report["selection_task"] = task;
        applied.report["selection_response"] =
            serde_json::to_value(response).map_err(|err| err.to_string())?;
        if write_artifacts {
            write_json_file(
                &output_dir.join("final_yaw_refinement_report.json"),
                &applied.report,
            )
            .map_err(|err| err.to_string())?;
        }
        Ok(applied)
    }

    fn render_final_context_yaw_candidates(
        &self,
        output_dir: &Path,
        commands: &[Value],
        task: &mut Value,
    ) -> Result<Value, String> {
        let root = output_dir.join("full_scene_candidates");
        fs::create_dir_all(&root).map_err(|err| {
            format!(
                "failed to create final yaw candidate directory {}: {err}",
                root.display()
            )
        })?;
        let expected_items = feedback_rotation_spawn_commands(commands).len().max(1);
        let mut attempted = 0usize;
        let mut rendered = 0usize;
        let mut target_only_attempted = 0usize;
        let mut target_only_rendered = 0usize;
        let mut reused = 0usize;
        let mut target_only_reused = 0usize;
        let mut errors = Vec::new();
        let Some(objects) = task.get_mut("objects").and_then(Value::as_array_mut) else {
            return Ok(json!({
                "enabled": true,
                "attempted": 0,
                "rendered": 0,
                "reason": "selection_task_missing_objects",
            }));
        };
        for object in objects {
            let index = object
                .get("index")
                .and_then(Value::as_u64)
                .map(|value| value as usize)
                .unwrap_or(usize::MAX);
            let command_index = object
                .get("command_index")
                .and_then(Value::as_u64)
                .map(|value| value as usize)
                .unwrap_or(usize::MAX);
            if index == usize::MAX || command_index == usize::MAX {
                errors.push(json!({
                    "index": index,
                    "reason": "missing_index_or_command_index",
                }));
                continue;
            }
            let object_id = object
                .get("object_id")
                .and_then(Value::as_str)
                .unwrap_or("object");
            let object_dir = root.join(sanitize_feedback_crop_name(index, object_id));
            fs::create_dir_all(&object_dir).map_err(|err| {
                format!(
                    "failed to create final yaw object candidate directory {}: {err}",
                    object_dir.display()
                )
            })?;
            let object_context = object.clone();
            let Some(candidates) = object
                .pointer_mut("/rotation_selection/candidates")
                .and_then(Value::as_array_mut)
            else {
                continue;
            };
            let mut current_full_scene_render = None;
            for candidate in candidates {
                let candidate_index = candidate
                    .get("candidate_index")
                    .and_then(Value::as_u64)
                    .unwrap_or(0) as usize;
                let Some(candidate_yaw) = candidate
                    .get("candidate_yaw_degrees")
                    .and_then(Value::as_f64)
                    .filter(|value| value.is_finite())
                    .map(|value| value as f32)
                else {
                    errors.push(json!({
                        "index": index,
                        "candidate_index": candidate_index,
                        "reason": "candidate_missing_yaw",
                    }));
                    continue;
                };
                let has_full_render = candidate
                    .get("rendered_candidate_full_scene")
                    .and_then(Value::as_str)
                    .is_some_and(|path| {
                        Path::new(path).exists() && final_yaw_render_path_has_foreground(path)
                    });
                let has_target_only_render = candidate
                    .get("rendered_candidate_object_only_full_frame")
                    .and_then(Value::as_str)
                    .is_some_and(|path| {
                        Path::new(path).exists() && final_yaw_render_path_has_foreground(path)
                    });
                if has_full_render && has_target_only_render {
                    let full_path = candidate
                        .get("rendered_candidate_full_scene")
                        .and_then(Value::as_str)
                        .map(PathBuf::from);
                    let target_only_path = candidate
                        .get("rendered_candidate_object_only_full_frame")
                        .and_then(Value::as_str)
                        .map(PathBuf::from);
                    let target_bbox = candidate
                        .get("rendered_candidate_bbox")
                        .and_then(final_yaw_json_array4)
                        .or_else(|| {
                            candidate
                                .pointer("/final_yaw_visual_fit/projected_bbox")
                                .and_then(final_yaw_json_array4)
                        });
                    if let (Some(full_path), Some(target_only_path), Some(target_bbox)) =
                        (full_path, target_only_path, target_bbox)
                    {
                        let yaw_label = candidate
                            .get("candidate_yaw_degrees")
                            .and_then(Value::as_f64)
                            .map(|value| canonical_pose_yaw_file_component(value as f32))
                            .unwrap_or_else(|| format!("candidate_{candidate_index:02}"));
                        let crop_path = candidate
                            .get("rendered_candidate_crop")
                            .and_then(Value::as_str)
                            .map(PathBuf::from)
                            .unwrap_or_else(|| {
                                object_dir.join(format!(
                                    "candidate_{candidate_index:02}_yaw_{yaw_label}_full_scene_target_crop.png"
                                ))
                            });
                        let foreground = candidate
                            .pointer("/rendered_candidate_capture/foreground")
                            .cloned()
                            .unwrap_or_else(|| {
                                final_yaw_render_foreground_report(
                                    &full_path,
                                    FINAL_YAW_RENDER_MIN_FOREGROUND_RATIO,
                                )
                            });
                        let visual_fit = final_yaw_render_visual_fit_report(
                            &object_context,
                            &full_path,
                            target_bbox,
                            &crop_path,
                            &foreground,
                            Some(&target_only_path),
                        );
                        candidate["rendered_candidate_bbox"] = json!(target_bbox);
                        candidate["rendered_candidate_crop"] =
                            json!(crop_path.display().to_string());
                        candidate["final_yaw_visual_fit"] = visual_fit;
                    }
                    attempted += 1;
                    rendered += 1;
                    target_only_attempted += 1;
                    target_only_rendered += 1;
                    reused += 1;
                    target_only_reused += 1;
                    continue;
                }
                attempted += 1;
                let yaw_label = canonical_pose_yaw_file_component(candidate_yaw);
                let screenshot_path = object_dir.join(format!(
                    "candidate_{candidate_index:02}_yaw_{yaw_label}_full_scene.png"
                ));
                let apply_ack = match self.send_scene_commands(
                    final_yaw_full_scene_candidate_commands(commands, command_index, candidate),
                ) {
                    Ok(value) => value,
                    Err(err) => {
                        errors.push(json!({
                            "index": index,
                            "candidate_index": candidate_index,
                            "error": err,
                        }));
                        continue;
                    }
                };
                let (capture, status, foreground) = match self
                    .capture_feedback_when_projected_ready_with_foreground(
                        &apply_ack,
                        &screenshot_path,
                        expected_items,
                        FINAL_YAW_RENDER_MIN_FOREGROUND_RATIO,
                    ) {
                    Ok(value) => value,
                    Err(err) => {
                        errors.push(json!({
                            "index": index,
                            "candidate_index": candidate_index,
                            "error": err,
                        }));
                        continue;
                    }
                };
                if !foreground
                    .get("ok")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                {
                    candidate["rendered_candidate_full_scene_validation"] = foreground.clone();
                    errors.push(json!({
                        "index": index,
                        "candidate_index": candidate_index,
                        "reason": "full_scene_candidate_render_blank_or_not_ready",
                        "foreground": foreground,
                    }));
                    continue;
                }
                let target_bbox = final_yaw_target_projected_bbox(&status, commands, command_index);
                if let Some(target_bbox) = target_bbox {
                    let crop_path = object_dir.join(format!(
                        "candidate_{candidate_index:02}_yaw_{yaw_label}_full_scene_target_crop.png"
                    ));
                    let visual_fit = final_yaw_render_visual_fit_report(
                        &object_context,
                        &screenshot_path,
                        target_bbox,
                        &crop_path,
                        &foreground,
                        None,
                    );
                    candidate["rendered_candidate_bbox"] = json!(target_bbox);
                    candidate["rendered_candidate_crop"] = json!(crop_path.display().to_string());
                    candidate["final_yaw_visual_fit"] = visual_fit;
                } else {
                    candidate["final_yaw_visual_fit"] = json!({
                        "error": "missing_target_projected_bbox",
                    });
                }
                candidate["rendered_candidate_full_scene"] =
                    json!(screenshot_path.display().to_string());
                candidate["rendered_candidate_full_frame"] =
                    json!(screenshot_path.display().to_string());
                candidate["rendered_candidate_capture"] = json!({
                    "ok": capture.get("ok").cloned().unwrap_or(Value::Null),
                    "output_path": screenshot_path.display().to_string(),
                    "expected_items": expected_items,
                    "projected_item_count": status
                        .get("projected_items")
                        .and_then(Value::as_array)
                        .map(Vec::len)
                        .unwrap_or(0),
                    "foreground": foreground,
                });
                if candidate_index == 0 {
                    current_full_scene_render = Some(screenshot_path.display().to_string());
                }
                rendered += 1;

                let target_only_path = object_dir.join(format!(
                    "candidate_{candidate_index:02}_yaw_{yaw_label}_target_only_full_frame.png"
                ));
                let Some(target_only_commands) =
                    final_yaw_target_only_candidate_commands(commands, command_index, candidate)
                else {
                    errors.push(json!({
                        "index": index,
                        "candidate_index": candidate_index,
                        "reason": "target_only_candidate_missing_spawn_command",
                    }));
                    continue;
                };
                target_only_attempted += 1;
                let apply_ack = match self.send_scene_commands(target_only_commands) {
                    Ok(value) => value,
                    Err(err) => {
                        errors.push(json!({
                            "index": index,
                            "candidate_index": candidate_index,
                            "target_only": true,
                            "error": err,
                        }));
                        continue;
                    }
                };
                let (capture, status, foreground) = match self
                    .capture_feedback_when_projected_ready_with_foreground(
                        &apply_ack,
                        &target_only_path,
                        1,
                        FINAL_YAW_TARGET_ONLY_RENDER_MIN_FOREGROUND_RATIO,
                    ) {
                    Ok(value) => value,
                    Err(err) => {
                        errors.push(json!({
                            "index": index,
                            "candidate_index": candidate_index,
                            "target_only": true,
                            "error": err,
                        }));
                        continue;
                    }
                };
                if !foreground
                    .get("ok")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                {
                    candidate["rendered_candidate_object_only_validation"] = foreground.clone();
                    errors.push(json!({
                        "index": index,
                        "candidate_index": candidate_index,
                        "target_only": true,
                        "reason": "target_only_candidate_render_blank_or_not_ready",
                        "foreground": foreground,
                    }));
                    continue;
                }
                candidate["rendered_candidate_object_only_full_frame"] =
                    json!(target_only_path.display().to_string());
                candidate["rendered_candidate_object_only_capture"] = json!({
                    "ok": capture.get("ok").cloned().unwrap_or(Value::Null),
                    "output_path": target_only_path.display().to_string(),
                    "expected_items": 1,
                    "projected_item_count": status
                        .get("projected_items")
                        .and_then(Value::as_array)
                        .map(Vec::len)
                        .unwrap_or(0),
                    "foreground": foreground,
                });
                if let Some(target_bbox) = candidate
                    .get("rendered_candidate_bbox")
                    .and_then(final_yaw_json_array4)
                {
                    let crop_path = candidate
                        .get("rendered_candidate_crop")
                        .and_then(Value::as_str)
                        .map(PathBuf::from)
                        .unwrap_or_else(|| {
                            object_dir.join(format!(
                                "candidate_{candidate_index:02}_yaw_{yaw_label}_full_scene_target_crop.png"
                            ))
                        });
                    let full_foreground = candidate
                        .pointer("/rendered_candidate_capture/foreground")
                        .cloned()
                        .unwrap_or_else(|| {
                            final_yaw_render_foreground_report(
                                &screenshot_path,
                                FINAL_YAW_RENDER_MIN_FOREGROUND_RATIO,
                            )
                        });
                    let visual_fit = final_yaw_render_visual_fit_report(
                        &object_context,
                        &screenshot_path,
                        target_bbox,
                        &crop_path,
                        &full_foreground,
                        Some(&target_only_path),
                    );
                    candidate["rendered_candidate_crop"] = json!(crop_path.display().to_string());
                    candidate["final_yaw_visual_fit"] = visual_fit;
                }
                target_only_rendered += 1;
            }
            if let Some(path) = current_full_scene_render {
                object["current_full_scene_render"] = json!(path);
            }
        }
        let _ = self.send_scene_commands_unacknowledged(commands.to_vec());
        Ok(json!({
            "enabled": true,
            "selector_support": "full_scene_context_single_object_measured_pose_candidates",
            "root": root.display().to_string(),
            "attempted": attempted,
            "rendered": rendered,
            "reused": reused,
            "target_only_attempted": target_only_attempted,
            "target_only_rendered": target_only_rendered,
            "target_only_reused": target_only_reused,
            "error_count": errors.len(),
            "errors": errors,
        }))
    }

    fn render_feedback_rotation_candidate_crops(
        &self,
        iteration_dir: &Path,
        commands: &[Value],
        metrics: &mut Value,
        object_crops: &mut Value,
        progress: &Option<SceneFeedbackProgressSink<'_>>,
    ) -> Result<Value, String> {
        let object_count = metrics
            .get("objects")
            .and_then(Value::as_array)
            .map(Vec::len)
            .unwrap_or(0);
        if object_count == 0 || object_crops.is_null() {
            return Ok(json!({
                "enabled": true,
                "rendered": 0,
                "attempted": 0,
                "reason": "no feedback objects with crops",
            }));
        }
        let spawn_commands = feedback_rotation_spawn_commands(commands);
        let Some(camera_command) = feedback_rotation_camera_command(commands) else {
            return Ok(json!({
                "enabled": true,
                "rendered": 0,
                "attempted": 0,
                "error": "feedback commands did not include a set_camera command",
            }));
        };
        let candidate_root = iteration_dir.join("rotation_candidates");
        fs::create_dir_all(&candidate_root).map_err(|err| {
            format!(
                "failed to create rotation candidate directory {}: {err}",
                candidate_root.display()
            )
        })?;

        let mut crop_objects = object_crops
            .get_mut("objects")
            .and_then(Value::as_array_mut);
        let Some(metrics_objects) = metrics.get_mut("objects").and_then(Value::as_array_mut) else {
            return Ok(json!({
                "enabled": true,
                "rendered": 0,
                "attempted": 0,
                "error": "metrics missing object array",
            }));
        };

        let mut attempted = 0usize;
        let mut rendered = 0usize;
        let mut scored = 0usize;
        let mut errors = Vec::new();
        for object in metrics_objects {
            let index = object
                .get("index")
                .and_then(Value::as_u64)
                .map(|value| value as usize)
                .unwrap_or(usize::MAX);
            if index == usize::MAX {
                errors.push(json!({
                    "reason": "object_missing_index",
                }));
                continue;
            }
            let Some((_, base_spawn)) = spawn_commands.get(index) else {
                errors.push(json!({
                    "index": index,
                    "reason": "spawn_command_missing_for_object_index",
                }));
                continue;
            };
            let object_id = object
                .get("object_id")
                .and_then(Value::as_str)
                .unwrap_or("object")
                .to_string();
            let name = sanitize_feedback_crop_name(index, &object_id);
            let source_crop = crop_objects
                .as_deref()
                .and_then(|objects| {
                    objects.iter().find(|crop| {
                        crop.get("index").and_then(Value::as_u64) == Some(index as u64)
                    })
                })
                .and_then(|crop| crop.get("source_crop").and_then(Value::as_str))
                .map(PathBuf::from);
            if object.get("rotation_selection").is_none() {
                continue;
            }
            let object_dir = candidate_root.join(&name);
            fs::create_dir_all(&object_dir).map_err(|err| {
                format!(
                    "failed to create rotation candidate object directory {}: {err}",
                    object_dir.display()
                )
            })?;

            let current_yaw = base_spawn
                .get("rotation")
                .and_then(json_array4)
                .map(quat_y_degrees)
                .or_else(|| {
                    object
                        .get("current_yaw_degrees")
                        .and_then(Value::as_f64)
                        .map(|value| value as f32)
                })
                .unwrap_or(0.0);
            let current_screenshot_path = object_dir.join("current_isolated_full_frame.png");
            let current_commands =
                feedback_rotation_candidate_commands(base_spawn, current_yaw, &camera_command);
            match self
                .send_scene_commands(current_commands)
                .and_then(|apply_ack| {
                    self.capture_feedback_when_projected_ready(
                        &apply_ack,
                        &current_screenshot_path,
                        1,
                    )
                }) {
                Ok((capture, status)) => {
                    let projected_bbox = status
                        .get("projected_items")
                        .and_then(Value::as_array)
                        .and_then(|items| items.first())
                        .and_then(|item| item.get("screen_bbox"))
                        .and_then(json_array4)
                        .and_then(feedback_visible_bbox);
                    let isolated_report = json!({
                        "yaw_degrees": current_yaw,
                        "full_frame": current_screenshot_path.display().to_string(),
                        "projected_bbox": projected_bbox,
                        "capture": {
                            "ok": capture.get("ok").cloned().unwrap_or(Value::Null),
                            "output_path": current_screenshot_path.display().to_string(),
                            "projected_item_count": status
                                .get("projected_items")
                                .and_then(Value::as_array)
                                .map(Vec::len)
                                .unwrap_or(0),
                        },
                    });
                    object["isolated_render_full_frame"] =
                        json!(current_screenshot_path.display().to_string());
                    object["isolated_render_bbox"] = projected_bbox
                        .map(|bbox| json!(bbox))
                        .unwrap_or(Value::Null);
                    emit_scene_feedback_progress(
                        progress,
                        "render_capture_feedback.object_isolated",
                        SceneBuildProgressPhase::Completed,
                        SceneBuildExecutionKind::Viewer,
                        format!("captured isolated full-frame render for {object_id}"),
                        Some(index.saturating_add(1)),
                        Some(object_count),
                        Some(current_screenshot_path.clone()),
                        json!({
                            "index": index,
                            "object_id": object_id.clone(),
                            "path": current_screenshot_path.display().to_string(),
                            "projected_bbox": projected_bbox,
                        }),
                    );
                    if let Some(crop_objects) = crop_objects.as_deref_mut()
                        && let Some(crop_object) = crop_objects.iter_mut().find(|crop| {
                            crop.get("index").and_then(Value::as_u64) == Some(index as u64)
                        })
                    {
                        crop_object["isolated_render_full_frame"] =
                            json!(current_screenshot_path.display().to_string());
                        crop_object["isolated_render_bbox"] = projected_bbox
                            .map(|bbox| json!(bbox))
                            .unwrap_or(Value::Null);
                        crop_object["isolated_render"] = isolated_report;
                    }
                }
                Err(err) => {
                    errors.push(json!({
                        "index": index,
                        "reason": "current_isolated_full_frame_failed",
                        "error": err,
                    }));
                }
            }

            let Some(rotation_selection) = object.get_mut("rotation_selection") else {
                continue;
            };
            let Some(candidates) = rotation_selection
                .get_mut("candidates")
                .and_then(Value::as_array_mut)
            else {
                continue;
            };
            let mut rendered_candidates = Vec::new();
            for candidate in candidates
                .iter_mut()
                .take(FEEDBACK_ROTATION_RENDER_MAX_CANDIDATES)
            {
                let candidate_index = candidate
                    .get("candidate_index")
                    .and_then(Value::as_u64)
                    .unwrap_or(rendered_candidates.len() as u64)
                    as usize;
                let Some(candidate_yaw) = candidate
                    .get("candidate_yaw_degrees")
                    .and_then(Value::as_f64)
                    .filter(|value| value.is_finite())
                    .map(|value| value as f32)
                else {
                    errors.push(json!({
                        "index": index,
                        "candidate_index": candidate_index,
                        "reason": "candidate_missing_yaw",
                    }));
                    continue;
                };
                attempted += 1;
                let yaw_label = canonical_pose_yaw_file_component(candidate_yaw);
                let screenshot_path = object_dir.join(format!(
                    "candidate_{candidate_index:02}_yaw_{yaw_label}_screenshot.png"
                ));
                let crop_path = object_dir.join(format!(
                    "candidate_{candidate_index:02}_yaw_{yaw_label}_crop.png"
                ));
                let candidate_commands = feedback_rotation_candidate_commands(
                    base_spawn,
                    candidate_yaw,
                    &camera_command,
                );
                let apply_ack = match self.send_scene_commands(candidate_commands) {
                    Ok(value) => value,
                    Err(err) => {
                        errors.push(json!({
                            "index": index,
                            "candidate_index": candidate_index,
                            "error": err,
                        }));
                        continue;
                    }
                };
                let (capture, status) = match self.capture_feedback_when_projected_ready(
                    &apply_ack,
                    &screenshot_path,
                    1,
                ) {
                    Ok(value) => value,
                    Err(err) => {
                        errors.push(json!({
                            "index": index,
                            "candidate_index": candidate_index,
                            "error": err,
                        }));
                        continue;
                    }
                };
                candidate["rendered_candidate_screenshot"] =
                    json!(screenshot_path.display().to_string());
                candidate["rendered_candidate_full_frame"] =
                    json!(screenshot_path.display().to_string());
                candidate["rendered_candidate_capture"] = json!({
                    "ok": capture.get("ok").cloned().unwrap_or(Value::Null),
                    "output_path": screenshot_path.display().to_string(),
                    "projected_item_count": status
                        .get("projected_items")
                        .and_then(Value::as_array)
                        .map(Vec::len)
                        .unwrap_or(0),
                });
                if let Some(bbox) = status
                    .get("projected_items")
                    .and_then(Value::as_array)
                    .and_then(|items| items.first())
                    .and_then(|item| item.get("screen_bbox"))
                    .and_then(json_array4)
                    .and_then(feedback_visible_bbox)
                    && let Ok(image) = image::open(&screenshot_path)
                {
                    match write_feedback_crop(&image, bbox, &crop_path) {
                        Ok(()) => {
                            candidate["rendered_candidate_crop"] =
                                json!(crop_path.display().to_string());
                            candidate["rendered_candidate_bbox"] = json!(bbox);
                            if let Some(source_crop) = &source_crop {
                                match feedback_crop_visual_similarity(source_crop, &crop_path) {
                                    Ok(similarity) => {
                                        let score = similarity
                                            .get("score")
                                            .and_then(Value::as_f64)
                                            .unwrap_or(0.0);
                                        candidate["visual_similarity"] = similarity;
                                        candidate["visual_score"] = json!(score);
                                        scored += 1;
                                    }
                                    Err(err) => {
                                        candidate["visual_similarity_error"] = json!(err);
                                    }
                                }
                            }
                            rendered += 1;
                            rendered_candidates.push(json!({
                                "candidate_index": candidate_index,
                                "yaw_degrees": candidate_yaw,
                                "full_frame": screenshot_path.display().to_string(),
                                "crop": crop_path.display().to_string(),
                                "screenshot": screenshot_path.display().to_string(),
                                "visual_score": candidate.get("visual_score").cloned().unwrap_or(Value::Null),
                            }));
                        }
                        Err(err) => {
                            candidate["rendered_candidate_crop_error"] = json!(err);
                            errors.push(json!({
                                "index": index,
                                "candidate_index": candidate_index,
                                "error": err,
                            }));
                        }
                    }
                }
            }
            let rendered_candidate_count = rendered_candidates.len();
            if let Some(crop_objects) = crop_objects.as_deref_mut()
                && let Some(crop_object) = crop_objects
                    .iter_mut()
                    .find(|crop| crop.get("index").and_then(Value::as_u64) == Some(index as u64))
            {
                crop_object["rotation_candidates"] = json!(rendered_candidates);
            }
            emit_scene_feedback_progress(
                progress,
                "render_capture_feedback.rotation_candidates",
                SceneBuildProgressPhase::Completed,
                SceneBuildExecutionKind::Viewer,
                format!("captured isolated yaw candidates for {object_id}"),
                Some(index.saturating_add(1)),
                Some(object_count),
                Some(object_dir.clone()),
                json!({
                    "index": index,
                    "object_id": object_id.clone(),
                    "candidate_dir": object_dir.display().to_string(),
                    "candidate_count": candidates.len().min(FEEDBACK_ROTATION_RENDER_MAX_CANDIDATES),
                    "rendered_count": rendered_candidate_count,
                }),
            );
        }
        let _ = self.send_scene_commands_unacknowledged(commands.to_vec());
        Ok(json!({
            "enabled": true,
            "selector_support": "isolated_per_object_scene_camera_yaw_candidates",
            "root": candidate_root.display().to_string(),
            "max_candidates_per_object": FEEDBACK_ROTATION_RENDER_MAX_CANDIDATES,
            "attempted": attempted,
            "rendered": rendered,
            "scored": scored,
            "error_count": errors.len(),
            "errors": errors,
        }))
    }

    fn apply_feedback_rotation_selector(
        &self,
        selector: FeedbackRotationSelector,
        iteration_dir: &Path,
        task: &Value,
        metrics: &mut Value,
    ) -> Result<Value, String> {
        if task.is_null() {
            return Ok(Value::Null);
        }
        match selector {
            FeedbackRotationSelector::Deterministic => Ok(json!({
                "selector": "deterministic",
                "applied_count": 0,
                "reason": "using deterministic geometry-selected rotation candidates",
            })),
            FeedbackRotationSelector::RenderedSweep => {
                let applied = apply_feedback_rendered_rotation_selection(metrics);
                Ok(json!({
                    "selector": "rendered-sweep",
                    "applied": applied,
                }))
            }
            FeedbackRotationSelector::Openai => {
                let image_paths = feedback_rotation_selection_image_paths(task);
                let prompt = feedback_rotation_selection_prompt(task);
                write_json_file(
                    &iteration_dir.join("rotation_selection_request.json"),
                    &json!({
                        "prompt": prompt,
                        "task": task,
                        "image_paths": image_paths
                            .iter()
                            .map(|path| path.display().to_string())
                            .collect::<Vec<_>>(),
                    }),
                )
                .map_err(|err| err.to_string())?;
                match self.openai_provider().and_then(|provider| {
                    provider
                        .select_rotation_candidates(&SceneRotationSelectionRequest {
                            prompt: feedback_rotation_selection_prompt(task),
                            task: task.clone(),
                            image_paths,
                        })
                        .map_err(|err| err.to_string())
                }) {
                    Ok(response) => {
                        write_json_file(
                            &iteration_dir.join("rotation_selection_response.json"),
                            &serde_json::to_value(&response).map_err(|err| err.to_string())?,
                        )
                        .map_err(|err| err.to_string())?;
                        let applied =
                            apply_feedback_rotation_selection_response(metrics, &response);
                        Ok(json!({
                            "selector": "openai",
                            "fallback": false,
                            "applied": applied,
                        }))
                    }
                    Err(err) => Ok(json!({
                        "selector": "openai",
                        "fallback": true,
                        "error": err,
                        "reason": "keeping deterministic geometry-selected rotation candidates",
                    })),
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_feedback_rubric_scorer(
        &self,
        scorer: FeedbackRubricScorer,
        iteration_dir: &Path,
        source_scene_path: &Path,
        screenshot_path: &Path,
        manifest: &SceneObjectManifest,
        grounded_layout: &GroundedSceneLayout,
        metrics: &Value,
        iteration_index: Option<usize>,
    ) -> Result<Value, String> {
        match scorer {
            FeedbackRubricScorer::Off => Ok(Value::Null),
            FeedbackRubricScorer::Openai => {
                let prompt = feedback_scene_quality_rubric_prompt();
                let context = feedback_scene_quality_context(
                    manifest,
                    grounded_layout,
                    metrics,
                    iteration_index,
                );
                write_json_file(
                    &iteration_dir.join("scene_quality_rubric_request.json"),
                    &json!({
                        "scorer": scorer,
                        "prompt": prompt,
                        "source_scene_path": source_scene_path.display().to_string(),
                        "rendered_scene_path": screenshot_path.display().to_string(),
                        "context": context,
                    }),
                )
                .map_err(|err| err.to_string())?;
                match self.openai_provider().and_then(|provider| {
                    provider
                        .score_scene_quality(&SceneQualityRubricRequest {
                            prompt,
                            source_scene_path: source_scene_path.to_path_buf(),
                            rendered_scene_path: screenshot_path.to_path_buf(),
                            context,
                        })
                        .map_err(|err| err.to_string())
                }) {
                    Ok(response) => serde_json::to_value(SceneQualityRubricResponse {
                        overall_score: response.overall_score.clamp(0.0, 1.0),
                        object_count_score: response.object_count_score.clamp(0.0, 1.0),
                        placement_score: response.placement_score.clamp(0.0, 1.0),
                        scale_score: response.scale_score.clamp(0.0, 1.0),
                        rotation_score: response.rotation_score.clamp(0.0, 1.0),
                        camera_score: response.camera_score.clamp(0.0, 1.0),
                        physical_plausibility_score: response
                            .physical_plausibility_score
                            .clamp(0.0, 1.0),
                        summary: response.summary,
                        blocking_issue_count: response.blocking_issue_count,
                        issues: response.issues,
                    })
                    .map_err(|err| err.to_string()),
                    Err(err) => Ok(json!({
                        "scorer": "openai",
                        "fallback": true,
                        "error": err,
                        "reason": "scene quality rubric scoring failed; deterministic geometry metrics remain authoritative",
                    })),
                }
            }
        }
    }

    pub(crate) fn feedback_capture_status(apply_ack: &Value, capture_ack: &Value) -> Value {
        let apply_status = apply_ack.get("status").cloned().unwrap_or(Value::Null);
        let capture_status = capture_ack
            .get("acknowledgement")
            .and_then(|ack| ack.get("status"))
            .cloned();
        if let Some(capture_status) = capture_status
            && Self::feedback_status_has_projected_aabbs(&capture_status)
        {
            return capture_status;
        }
        if Self::feedback_status_has_projected_aabbs(&apply_status) {
            return apply_status;
        }
        capture_ack
            .get("acknowledgement")
            .and_then(|ack| ack.get("status"))
            .cloned()
            .or_else(|| (!apply_status.is_null()).then_some(apply_status))
            .unwrap_or(Value::Null)
    }

    fn feedback_status_has_projected_aabbs(status: &Value) -> bool {
        status
            .get("projected_items")
            .and_then(Value::as_array)
            .is_some_and(|items| {
                !items.is_empty()
                    && items.iter().all(|item| {
                        item.get("screen_bbox")
                            .is_some_and(|value| !value.is_null())
                            && item.get("world_aabb").is_some_and(|value| !value.is_null())
                    })
            })
    }

    pub(crate) fn feedback_status_projected_items_ready(
        status: &Value,
        expected_items: usize,
    ) -> bool {
        if expected_items == 0 {
            return true;
        }
        let Some(projected_items) = status.get("projected_items").and_then(Value::as_array) else {
            return false;
        };
        if projected_items.len() < expected_items {
            return false;
        }
        projected_items.iter().take(expected_items).all(|item| {
            item.get("screen_bbox").and_then(Value::as_array).is_some()
                && item
                    .get("projected_corners")
                    .and_then(Value::as_u64)
                    .is_some_and(|corners| corners > 0)
                && item.get("world_aabb").is_some_and(|value| !value.is_null())
        })
    }

    fn scene_command_envelope(
        &self,
        commands: Vec<Value>,
    ) -> Result<(PathBuf, u64, Value), String> {
        if commands.is_empty() {
            return Err("scene command list must not be empty".to_string());
        }
        let control_path = self
            .config
            .scene_control_path
            .as_ref()
            .ok_or_else(|| "scene_control_path is not configured".to_string())?;
        let sequence = next_scene_sequence();
        let session_id = format!("burn_synth_mcp-{}", std::process::id());
        let envelope = json!({
            "session_id": session_id,
            "sequence": sequence,
            "commands": commands,
        });
        Ok((control_path.clone(), sequence, envelope))
    }

    fn send_scene_commands_unacknowledged(&self, commands: Vec<Value>) -> Result<Value, String> {
        let (control_path, sequence, envelope) = self.scene_command_envelope(commands)?;
        atomic_write_json(&control_path, &envelope)?;
        Ok(json!({
            "tool": "scene_command",
            "command_path": control_path.display().to_string(),
            "sequence": sequence,
            "acknowledged": false,
        }))
    }

    pub(crate) fn send_scene_commands(&self, commands: Vec<Value>) -> Result<Value, String> {
        let (control_path, sequence, envelope) = self.scene_command_envelope(commands)?;
        atomic_write_json(&control_path, &envelope)?;
        let Some(status_path) = self.config.scene_status_path.as_ref() else {
            return Ok(json!({
                "tool": "scene_command",
                "command_path": control_path.display().to_string(),
                "sequence": sequence,
                "acknowledged": false,
            }));
        };
        let status = wait_scene_status(status_path, sequence, self.config.scene_timeout)?;
        Ok(json!({
            "tool": "scene_command",
            "command_path": control_path.display().to_string(),
            "status_path": status_path.display().to_string(),
            "sequence": sequence,
            "acknowledged": true,
            "status": status,
        }))
    }
}

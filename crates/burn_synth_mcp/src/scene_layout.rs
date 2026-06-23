use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use image::imageops::FilterType;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct SceneReferenceObject {
    #[serde(default)]
    pub id: Option<String>,
    pub label: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    /// Normalized source-image box: [x_min, y_min, x_max, y_max].
    pub bbox: [f32; 4],
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct SceneAssetBinding {
    #[serde(default)]
    pub reference_id: Option<String>,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default)]
    pub path: Option<PathBuf>,
    #[serde(default)]
    pub cache_key: Option<String>,
    #[serde(default)]
    pub select: bool,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct SceneComposeArgs {
    pub reference_objects: Vec<SceneReferenceObject>,
    pub assets: Vec<SceneAssetBinding>,
    #[serde(default)]
    pub apply: bool,
    #[serde(default)]
    pub clear_existing: bool,
    #[serde(default = "default_layout_width")]
    pub layout_width: f32,
    #[serde(default = "default_layout_depth")]
    pub layout_depth: f32,
    #[serde(default)]
    pub y: f32,
    #[serde(default = "default_min_scale")]
    pub min_scale: f32,
    #[serde(default = "default_scale_multiplier")]
    pub scale_multiplier: f32,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct SceneValidateArgs {
    pub reference_objects: Vec<SceneReferenceObject>,
    #[serde(default)]
    pub scene_status: Option<Value>,
    #[serde(default)]
    pub source_image_path: Option<PathBuf>,
    #[serde(default)]
    pub rendered_image_path: Option<PathBuf>,
    #[serde(default)]
    pub thresholds: SceneValidationThresholds,
}

#[derive(Clone, Copy, Debug, Deserialize)]
pub(crate) struct SceneValidationThresholds {
    #[serde(default = "default_min_semantic_score")]
    pub min_semantic_score: f32,
    #[serde(default = "default_min_layout_score")]
    pub min_layout_score: f32,
    #[serde(default = "default_min_overall_score")]
    pub min_overall_score: f32,
    #[serde(default)]
    pub max_extra_objects: usize,
    #[serde(default)]
    pub min_image_similarity: Option<f32>,
}

impl Default for SceneValidationThresholds {
    fn default() -> Self {
        Self {
            min_semantic_score: default_min_semantic_score(),
            min_layout_score: default_min_layout_score(),
            min_overall_score: default_min_overall_score(),
            max_extra_objects: 0,
            min_image_similarity: None,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ScenePlacement {
    pub reference_id: Option<String>,
    pub label: String,
    pub asset_index: usize,
    pub path: Option<String>,
    pub cache_key: Option<String>,
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
    pub source_bbox: [f32; 4],
    pub select: bool,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct SceneComposePlan {
    pub tool: &'static str,
    pub placements: Vec<ScenePlacement>,
    pub unmatched_reference_objects: Vec<String>,
    pub unused_assets: Vec<usize>,
    pub layout_width: f32,
    pub layout_depth: f32,
    pub apply: bool,
    pub clear_existing: bool,
}

#[derive(Clone, Debug, Serialize)]
struct SceneValidationMatch {
    reference_id: Option<String>,
    expected_label: String,
    observed_cache_key: String,
    observed_label: String,
    semantic_score: f32,
    position_score: f32,
    scale_score: f32,
    layout_score: f32,
    overall_score: f32,
    expected_center: [f32; 2],
    observed_center: [f32; 2],
}

#[derive(Clone, Debug, Serialize)]
struct ImageSimilarity {
    source_image_path: String,
    rendered_image_path: String,
    score: f32,
    normalized_cross_correlation: f32,
    psnr_db: f32,
    mean_abs_luma_delta: f32,
}

pub(crate) fn compose_scene_layout(args: SceneComposeArgs) -> Result<SceneComposePlan, String> {
    if args.reference_objects.is_empty() {
        return Err("reference_objects must not be empty".to_string());
    }
    if args.assets.is_empty() {
        return Err("assets must not be empty".to_string());
    }
    if !args.layout_width.is_finite() || args.layout_width <= 0.0 {
        return Err("layout_width must be a positive finite number".to_string());
    }
    if !args.layout_depth.is_finite() || args.layout_depth <= 0.0 {
        return Err("layout_depth must be a positive finite number".to_string());
    }

    let mut used_assets = HashSet::new();
    let mut placements = Vec::new();
    let mut unmatched_reference_objects = Vec::new();

    for reference in &args.reference_objects {
        let Some((asset_index, asset)) =
            best_asset_for_reference(reference, &args.assets, &used_assets)
        else {
            unmatched_reference_objects.push(reference_name(reference));
            continue;
        };
        used_assets.insert(asset_index);
        let bbox = normalize_bbox(reference.bbox);
        let center_x = (bbox[0] + bbox[2]) * 0.5;
        let center_y = (bbox[1] + bbox[3]) * 0.5;
        let width = (bbox[2] - bbox[0]).max(0.01);
        let height = (bbox[3] - bbox[1]).max(0.01);
        let footprint_x = (width * args.layout_width * args.scale_multiplier).max(args.min_scale);
        let footprint_z = (height * args.layout_depth * args.scale_multiplier).max(args.min_scale);
        let uniform_scale = footprint_x.max(footprint_z);
        let cache_key = asset
            .cache_key
            .clone()
            .or_else(|| asset.path.as_ref().map(|path| path_cache_key(path)));
        placements.push(ScenePlacement {
            reference_id: reference.id.clone(),
            label: reference.label.clone(),
            asset_index,
            path: asset.path.as_ref().map(|path| path.display().to_string()),
            cache_key,
            translation: [
                (center_x - 0.5) * args.layout_width,
                args.y,
                (center_y - 0.5) * args.layout_depth,
            ],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [uniform_scale, uniform_scale, uniform_scale],
            source_bbox: bbox,
            select: asset.select,
        });
    }

    let unused_assets = (0..args.assets.len())
        .filter(|index| !used_assets.contains(index))
        .collect::<Vec<_>>();

    Ok(SceneComposePlan {
        tool: "scene_compose_assets",
        placements,
        unmatched_reference_objects,
        unused_assets,
        layout_width: args.layout_width,
        layout_depth: args.layout_depth,
        apply: args.apply,
        clear_existing: args.clear_existing,
    })
}

pub(crate) fn validate_scene_layout(mut args: SceneValidateArgs) -> Result<Value, String> {
    if args.reference_objects.is_empty() {
        return Err("reference_objects must not be empty".to_string());
    }
    let status = args
        .scene_status
        .take()
        .ok_or_else(|| "scene_status is required for layout validation".to_string())?;
    let observed = observed_objects_from_status(&status)?;
    let matches = assign_observed_objects(&args.reference_objects, &observed);
    let matched_observed = matches
        .iter()
        .filter_map(|m| m.as_ref().map(|m| m.observed_cache_key.clone()))
        .collect::<HashSet<_>>();
    let missing = args
        .reference_objects
        .iter()
        .zip(matches.iter())
        .filter(|(_, matched)| matched.is_none())
        .map(|(reference, _)| reference_name(reference))
        .collect::<Vec<_>>();
    let extra = observed
        .iter()
        .filter(|item| !matched_observed.contains(&item.cache_key))
        .map(|item| item.cache_key.clone())
        .collect::<Vec<_>>();

    let match_values = matches
        .iter()
        .filter_map(|matched| matched.as_ref())
        .cloned()
        .collect::<Vec<_>>();
    let semantic_score = mean_score(match_values.iter().map(|m| m.semantic_score));
    let layout_score = mean_score(match_values.iter().map(|m| m.layout_score));
    let overall_score = mean_score(match_values.iter().map(|m| m.overall_score));

    let image_similarity = match (
        args.source_image_path.as_ref(),
        args.rendered_image_path.as_ref(),
    ) {
        (Some(source), Some(rendered)) => Some(compare_images(source, rendered)?),
        _ => None,
    };
    let image_score_ok = image_similarity
        .as_ref()
        .zip(args.thresholds.min_image_similarity)
        .map(|(similarity, threshold)| similarity.score >= threshold)
        .unwrap_or(true);
    let passed = missing.is_empty()
        && extra.len() <= args.thresholds.max_extra_objects
        && semantic_score >= args.thresholds.min_semantic_score
        && layout_score >= args.thresholds.min_layout_score
        && overall_score >= args.thresholds.min_overall_score
        && image_score_ok;

    Ok(json!({
        "tool": "scene_validate_layout",
        "passed": passed,
        "scores": {
            "semantic": semantic_score,
            "layout": layout_score,
            "overall": overall_score,
        },
        "thresholds": {
            "min_semantic_score": args.thresholds.min_semantic_score,
            "min_layout_score": args.thresholds.min_layout_score,
            "min_overall_score": args.thresholds.min_overall_score,
            "max_extra_objects": args.thresholds.max_extra_objects,
            "min_image_similarity": args.thresholds.min_image_similarity,
        },
        "matches": match_values,
        "missing_reference_objects": missing,
        "extra_observed_objects": extra,
        "observed_count": observed.len(),
        "reference_count": args.reference_objects.len(),
        "image_similarity": image_similarity,
    }))
}

pub(crate) fn path_cache_key(path: &Path) -> String {
    format!("path:{}", path.display())
}

fn best_asset_for_reference<'a>(
    reference: &SceneReferenceObject,
    assets: &'a [SceneAssetBinding],
    used_assets: &HashSet<usize>,
) -> Option<(usize, &'a SceneAssetBinding)> {
    let mut best = None;
    let mut best_score = f32::NEG_INFINITY;
    for (index, asset) in assets.iter().enumerate() {
        if used_assets.contains(&index) {
            continue;
        }
        let score = asset_reference_score(reference, asset);
        if score > best_score {
            best = Some((index, asset));
            best_score = score;
        }
    }
    best
}

fn asset_reference_score(reference: &SceneReferenceObject, asset: &SceneAssetBinding) -> f32 {
    if let (Some(reference_id), Some(asset_reference_id)) =
        (reference.id.as_ref(), asset.reference_id.as_ref())
        && normalized_label(reference_id) == normalized_label(asset_reference_id)
    {
        return 2.0;
    }
    let mut labels = Vec::new();
    if let Some(label) = asset.label.as_ref() {
        labels.push(label.as_str());
    }
    labels.extend(asset.aliases.iter().map(String::as_str));
    if let Some(path) = asset.path.as_ref()
        && let Some(stem) = path.file_stem().and_then(|value| value.to_str())
    {
        labels.push(stem);
    }
    if let Some(cache_key) = asset.cache_key.as_ref() {
        labels.push(cache_key);
    }
    labels
        .iter()
        .map(|label| reference_label_score(reference, label))
        .fold(0.0, f32::max)
}

#[derive(Clone, Debug)]
struct ObservedObject {
    cache_key: String,
    label: String,
    center: [f32; 2],
    area: f32,
}

#[derive(Deserialize)]
struct StatusCacheEntry {
    cache_key: String,
    #[serde(default)]
    label: String,
    #[serde(default)]
    source_image_path: String,
}

#[derive(Clone, Deserialize)]
struct StatusWorldItem {
    cache_key: String,
    translation: [f32; 3],
    #[serde(default = "unit_scale")]
    scale: [f32; 3],
}

#[derive(Deserialize)]
struct StatusSnapshot {
    #[serde(default)]
    cache_entries: Vec<StatusCacheEntry>,
    #[serde(default)]
    world_items: Vec<StatusWorldItem>,
}

fn observed_objects_from_status(status: &Value) -> Result<Vec<ObservedObject>, String> {
    let snapshot: StatusSnapshot = serde_json::from_value(status.clone())
        .map_err(|err| format!("invalid scene status for layout validation: {err}"))?;
    let labels = snapshot
        .cache_entries
        .iter()
        .map(|entry| {
            let label = if !entry.label.trim().is_empty() {
                entry.label.clone()
            } else {
                label_from_path_or_key(&entry.source_image_path)
            };
            (entry.cache_key.clone(), label)
        })
        .collect::<HashMap<_, _>>();
    Ok(normalize_world_items(&snapshot.world_items)
        .into_iter()
        .map(|(item, center, area)| {
            let label = labels
                .get(&item.cache_key)
                .cloned()
                .unwrap_or_else(|| label_from_path_or_key(&item.cache_key));
            ObservedObject {
                cache_key: item.cache_key,
                label,
                center,
                area,
            }
        })
        .collect())
}

fn normalize_world_items(items: &[StatusWorldItem]) -> Vec<(StatusWorldItem, [f32; 2], f32)> {
    if items.is_empty() {
        return Vec::new();
    }
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_z = f32::INFINITY;
    let mut max_z = f32::NEG_INFINITY;
    for item in items {
        let half_x = item.scale[0].abs() * 0.5;
        let half_z = item.scale[2].abs() * 0.5;
        min_x = min_x.min(item.translation[0] - half_x);
        max_x = max_x.max(item.translation[0] + half_x);
        min_z = min_z.min(item.translation[2] - half_z);
        max_z = max_z.max(item.translation[2] + half_z);
    }
    let width = (max_x - min_x).max(1.0e-5);
    let depth = (max_z - min_z).max(1.0e-5);
    items
        .iter()
        .cloned()
        .map(|item| {
            let center = [
                ((item.translation[0] - min_x) / width).clamp(0.0, 1.0),
                ((item.translation[2] - min_z) / depth).clamp(0.0, 1.0),
            ];
            let area =
                ((item.scale[0].abs() / width) * (item.scale[2].abs() / depth)).clamp(1.0e-5, 1.0);
            (item, center, area)
        })
        .collect()
}

fn assign_observed_objects(
    references: &[SceneReferenceObject],
    observed: &[ObservedObject],
) -> Vec<Option<SceneValidationMatch>> {
    let mut pairs = Vec::new();
    for (reference_index, reference) in references.iter().enumerate() {
        let reference_bbox = normalize_bbox(reference.bbox);
        let expected_center = [
            (reference_bbox[0] + reference_bbox[2]) * 0.5,
            (reference_bbox[1] + reference_bbox[3]) * 0.5,
        ];
        let expected_area = ((reference_bbox[2] - reference_bbox[0])
            * (reference_bbox[3] - reference_bbox[1]))
            .max(1.0e-5);
        for (observed_index, observed) in observed.iter().enumerate() {
            let semantic_score = reference_label_score(reference, &observed.label)
                .max(reference_label_score(reference, &observed.cache_key));
            let dx = expected_center[0] - observed.center[0];
            let dy = expected_center[1] - observed.center[1];
            let position_score =
                (1.0 - ((dx * dx + dy * dy).sqrt() / 2.0_f32.sqrt())).clamp(0.0, 1.0);
            let scale_score = scale_similarity(expected_area, observed.area);
            let layout_score = (position_score * 0.8) + (scale_score * 0.2);
            let overall_score =
                (semantic_score * 0.55) + (layout_score * 0.35) + (scale_score * 0.10);
            pairs.push((
                overall_score,
                reference_index,
                observed_index,
                SceneValidationMatch {
                    reference_id: reference.id.clone(),
                    expected_label: reference.label.clone(),
                    observed_cache_key: observed.cache_key.clone(),
                    observed_label: observed.label.clone(),
                    semantic_score,
                    position_score,
                    scale_score,
                    layout_score,
                    overall_score,
                    expected_center,
                    observed_center: observed.center,
                },
            ));
        }
    }
    pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut assigned_references = HashSet::new();
    let mut assigned_observed = HashSet::new();
    let mut out = vec![None; references.len()];
    for (_, reference_index, observed_index, matched) in pairs {
        if assigned_references.contains(&reference_index)
            || assigned_observed.contains(&observed_index)
        {
            continue;
        }
        assigned_references.insert(reference_index);
        assigned_observed.insert(observed_index);
        out[reference_index] = Some(matched);
    }
    out
}

fn compare_images(source: &Path, rendered: &Path) -> Result<ImageSimilarity, String> {
    let source_image = image::open(source)
        .map_err(|err| format!("failed to open source image {}: {err}", source.display()))?
        .resize_exact(64, 64, FilterType::Triangle)
        .to_luma8();
    let rendered_image = image::open(rendered)
        .map_err(|err| {
            format!(
                "failed to open rendered image {}: {err}",
                rendered.display()
            )
        })?
        .resize_exact(64, 64, FilterType::Triangle)
        .to_luma8();

    let mut source_values = Vec::with_capacity(64 * 64);
    let mut rendered_values = Vec::with_capacity(64 * 64);
    for (source_px, rendered_px) in source_image.pixels().zip(rendered_image.pixels()) {
        source_values.push(source_px[0] as f32);
        rendered_values.push(rendered_px[0] as f32);
    }
    let ncc = normalized_cross_correlation(&source_values, &rendered_values);
    let mut mse = 0.0f32;
    let mut mae = 0.0f32;
    for (source, rendered) in source_values.iter().zip(rendered_values.iter()) {
        let delta = source - rendered;
        mse += delta * delta;
        mae += delta.abs();
    }
    mse /= source_values.len() as f32;
    mae /= source_values.len() as f32;
    let psnr_db = if mse <= f32::EPSILON {
        99.0
    } else {
        10.0 * ((255.0 * 255.0) / mse).log10()
    };
    let score = ((((ncc + 1.0) * 0.5) * 0.75) + ((1.0 - (mae / 255.0)) * 0.25)).clamp(0.0, 1.0);
    Ok(ImageSimilarity {
        source_image_path: source.display().to_string(),
        rendered_image_path: rendered.display().to_string(),
        score,
        normalized_cross_correlation: ncc,
        psnr_db,
        mean_abs_luma_delta: mae,
    })
}

fn normalized_cross_correlation(lhs: &[f32], rhs: &[f32]) -> f32 {
    if lhs.len() != rhs.len() || lhs.is_empty() {
        return 0.0;
    }
    let mean_lhs = lhs.iter().sum::<f32>() / lhs.len() as f32;
    let mean_rhs = rhs.iter().sum::<f32>() / rhs.len() as f32;
    let mut numerator = 0.0;
    let mut lhs_var = 0.0;
    let mut rhs_var = 0.0;
    for (lhs, rhs) in lhs.iter().zip(rhs.iter()) {
        let dl = lhs - mean_lhs;
        let dr = rhs - mean_rhs;
        numerator += dl * dr;
        lhs_var += dl * dl;
        rhs_var += dr * dr;
    }
    let denom = (lhs_var * rhs_var).sqrt();
    if denom <= f32::EPSILON {
        if (mean_lhs - mean_rhs).abs() <= f32::EPSILON {
            1.0
        } else {
            0.0
        }
    } else {
        (numerator / denom).clamp(-1.0, 1.0)
    }
}

fn reference_label_score(reference: &SceneReferenceObject, observed: &str) -> f32 {
    let mut best = label_score(&reference.label, observed);
    if let Some(id) = reference.id.as_ref() {
        best = best.max(label_score(id, observed));
    }
    for alias in &reference.aliases {
        best = best.max(label_score(alias, observed));
    }
    best
}

fn label_score(expected: &str, observed: &str) -> f32 {
    let expected_norm = normalized_label(expected);
    let observed_norm = normalized_label(observed);
    if expected_norm.is_empty() || observed_norm.is_empty() {
        return 0.0;
    }
    if expected_norm == observed_norm
        || observed_norm.contains(&expected_norm)
        || expected_norm.contains(&observed_norm)
    {
        return 1.0;
    }
    let expected_tokens = label_tokens(&expected_norm);
    let observed_tokens = label_tokens(&observed_norm);
    if expected_tokens.is_empty() || observed_tokens.is_empty() {
        return 0.0;
    }
    let overlap = expected_tokens.intersection(&observed_tokens).count() as f32;
    if overlap == 0.0 {
        return 0.0;
    }
    let precision = overlap / observed_tokens.len() as f32;
    let recall = overlap / expected_tokens.len() as f32;
    (2.0 * precision * recall) / (precision + recall)
}

fn label_tokens(value: &str) -> HashSet<String> {
    value
        .split_whitespace()
        .filter(|token| {
            !token.is_empty()
                && !matches!(
                    *token,
                    "mesh" | "splat" | "asset" | "object" | "path" | "glb" | "png" | "jpg" | "jpeg"
                )
        })
        .map(ToOwned::to_owned)
        .collect()
}

fn normalized_label(value: &str) -> String {
    let path_tail = value
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(value)
        .trim_start_matches("path:");
    let stem = path_tail
        .rsplit_once('.')
        .map(|(stem, _)| stem)
        .unwrap_or(path_tail);
    let mut out = String::with_capacity(stem.len());
    let mut last_was_space = true;
    for ch in stem.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            last_was_space = false;
        } else if !last_was_space {
            out.push(' ');
            last_was_space = true;
        }
    }
    out.trim().to_string()
}

fn label_from_path_or_key(value: &str) -> String {
    let normalized = normalized_label(value);
    if normalized.is_empty() {
        value.to_string()
    } else {
        normalized
    }
}

fn normalize_bbox(mut bbox: [f32; 4]) -> [f32; 4] {
    for value in &mut bbox {
        if !value.is_finite() {
            *value = 0.0;
        }
        *value = value.clamp(0.0, 1.0);
    }
    if bbox[2] < bbox[0] {
        bbox.swap(0, 2);
    }
    if bbox[3] < bbox[1] {
        bbox.swap(1, 3);
    }
    if (bbox[2] - bbox[0]).abs() <= f32::EPSILON {
        bbox[2] = (bbox[0] + 0.01).min(1.0);
    }
    if (bbox[3] - bbox[1]).abs() <= f32::EPSILON {
        bbox[3] = (bbox[1] + 0.01).min(1.0);
    }
    bbox
}

fn reference_name(reference: &SceneReferenceObject) -> String {
    reference
        .id
        .clone()
        .unwrap_or_else(|| reference.label.clone())
}

fn scale_similarity(expected_area: f32, observed_area: f32) -> f32 {
    if expected_area <= 0.0 || observed_area <= 0.0 {
        return 0.0;
    }
    let ratio = (observed_area / expected_area).max(1.0e-5);
    (1.0 - (ratio.log2().abs() / 2.0)).clamp(0.0, 1.0)
}

fn mean_score(scores: impl Iterator<Item = f32>) -> f32 {
    let mut total = 0.0;
    let mut count = 0usize;
    for score in scores {
        total += score;
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        total / count as f32
    }
}

fn unit_scale() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}

fn default_layout_width() -> f32 {
    6.0
}

fn default_layout_depth() -> f32 {
    4.0
}

fn default_min_scale() -> f32 {
    0.35
}

fn default_scale_multiplier() -> f32 {
    1.0
}

fn default_min_semantic_score() -> f32 {
    0.72
}

fn default_min_layout_score() -> f32 {
    0.70
}

fn default_min_overall_score() -> f32 {
    0.70
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn reference_objects() -> Vec<SceneReferenceObject> {
        vec![
            SceneReferenceObject {
                id: Some("chair_left".to_string()),
                label: "wood chair".to_string(),
                aliases: vec!["chair".to_string()],
                bbox: [0.10, 0.35, 0.35, 0.90],
            },
            SceneReferenceObject {
                id: Some("table".to_string()),
                label: "dining table".to_string(),
                aliases: vec!["table".to_string()],
                bbox: [0.45, 0.40, 0.80, 0.82],
            },
        ]
    }

    #[test]
    fn compose_scene_layout_preserves_relative_positions() {
        let plan = compose_scene_layout(SceneComposeArgs {
            reference_objects: reference_objects(),
            assets: vec![
                SceneAssetBinding {
                    reference_id: Some("chair_left".to_string()),
                    label: Some("chair".to_string()),
                    aliases: Vec::new(),
                    path: Some(PathBuf::from("/tmp/chair.glb")),
                    cache_key: None,
                    select: false,
                },
                SceneAssetBinding {
                    reference_id: Some("table".to_string()),
                    label: Some("table".to_string()),
                    aliases: Vec::new(),
                    path: Some(PathBuf::from("/tmp/table.glb")),
                    cache_key: None,
                    select: false,
                },
            ],
            apply: false,
            clear_existing: false,
            layout_width: 6.0,
            layout_depth: 4.0,
            y: 0.0,
            min_scale: 0.35,
            scale_multiplier: 1.0,
        })
        .expect("layout plan");

        assert_eq!(plan.placements.len(), 2);
        assert!(plan.placements[0].translation[0] < plan.placements[1].translation[0]);
        assert!(
            plan.placements[0]
                .cache_key
                .as_deref()
                .is_some_and(|key| key.starts_with("path:"))
        );
    }

    #[test]
    fn validate_scene_layout_checks_semantic_and_layout_match() {
        let status = json!({
            "cache_entries": [
                { "cache_key": "chair_cache", "label": "wood chair", "source_image_path": "chair.png" },
                { "cache_key": "table_cache", "label": "dining table", "source_image_path": "table.png" }
            ],
            "world_items": [
                { "cache_key": "chair_cache", "translation": [-2.0, 0.0, 0.9], "scale": [1.5, 1.5, 1.5] },
                { "cache_key": "table_cache", "translation": [1.5, 0.0, 0.6], "scale": [2.0, 2.0, 2.0] }
            ]
        });
        let result = validate_scene_layout(SceneValidateArgs {
            reference_objects: reference_objects(),
            scene_status: Some(status),
            source_image_path: None,
            rendered_image_path: None,
            thresholds: SceneValidationThresholds::default(),
        })
        .expect("layout validation");
        assert_eq!(result["passed"], true);
        assert!(result["scores"]["semantic"].as_f64().unwrap() > 0.9);
    }

    #[test]
    fn validate_scene_layout_rejects_swapped_semantics() {
        let status = json!({
            "cache_entries": [
                { "cache_key": "chair_cache", "label": "wood chair", "source_image_path": "chair.png" },
                { "cache_key": "table_cache", "label": "dining table", "source_image_path": "table.png" }
            ],
            "world_items": [
                { "cache_key": "chair_cache", "translation": [1.5, 0.0, 0.6], "scale": [1.5, 1.5, 1.5] },
                { "cache_key": "table_cache", "translation": [-2.0, 0.0, 0.9], "scale": [2.0, 2.0, 2.0] }
            ]
        });
        let result = validate_scene_layout(SceneValidateArgs {
            reference_objects: reference_objects(),
            scene_status: Some(status),
            source_image_path: None,
            rendered_image_path: None,
            thresholds: SceneValidationThresholds::default(),
        })
        .expect("layout validation");
        assert_eq!(result["passed"], false);
        assert!(result["scores"]["layout"].as_f64().unwrap() < 0.70);
    }

    #[test]
    fn validate_scene_layout_can_gate_render_image_similarity() {
        let dir = std::env::temp_dir().join(format!(
            "burn_synth_mcp_scene_similarity_{}_{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock before unix epoch")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let source = dir.join("source.png");
        let rendered = dir.join("rendered.png");
        let mut image = image::RgbaImage::new(8, 8);
        for y in 0..8 {
            for x in 0..8 {
                let value = if x < 4 { 40 } else { 220 };
                image.put_pixel(x, y, image::Rgba([value, value, value, 255]));
            }
        }
        image.save(&source).expect("save source");
        image.save(&rendered).expect("save rendered");

        let status = json!({
            "cache_entries": [
                { "cache_key": "chair_cache", "label": "wood chair", "source_image_path": "chair.png" }
            ],
            "world_items": [
                { "cache_key": "chair_cache", "translation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0] }
            ]
        });
        let result = validate_scene_layout(SceneValidateArgs {
            reference_objects: vec![SceneReferenceObject {
                id: Some("chair".to_string()),
                label: "wood chair".to_string(),
                aliases: vec!["chair".to_string()],
                bbox: [0.25, 0.25, 0.75, 0.75],
            }],
            scene_status: Some(status),
            source_image_path: Some(source),
            rendered_image_path: Some(rendered),
            thresholds: SceneValidationThresholds {
                min_image_similarity: Some(0.99),
                max_extra_objects: 0,
                ..SceneValidationThresholds::default()
            },
        })
        .expect("layout validation");
        assert_eq!(result["passed"], true, "{result:#}");
        assert!(
            result["image_similarity"]["score"].as_f64().unwrap() >= 0.99,
            "{result:#}"
        );
        std::fs::remove_dir_all(dir).expect("remove temp dir");
    }
}

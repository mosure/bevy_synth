use std::path::Path;

use serde::Serialize;

use crate::bsn::normalize_bbox;
use crate::layout::object_descriptor;
use crate::*;

pub fn object_manifest_prompt(
    scene_path: &Path,
    reference_path: &Path,
    allow_catalog_reuse: bool,
    category_filter: &SceneCategoryFilterConfig,
) -> String {
    format!(
        "Analyze the source scene image at `{}` and produce a strict object manifest for 3D reconstruction. \
Use the reference image `{}` as the expected clean isolated object-image style: single object, centered, full visible silhouette, neutral background, 3/4 camera. \
{} \
For the furniture demo prefer reusable object groups: one tan open sectional sofa, one coffee table, and reusable chair groups split by visually distinct chair type; repeated instances of the same chair type should share one group, but black lounge chairs and light mesh meeting chairs should not. Do not generate cube/proxy furniture. \
Include scene_calibration when a dominant table or seating arrangement is visible: table_center in normalized image coordinates, table_axis_degrees where 0 means table length points away from the camera in the source image, table_size_m in real meters, and camera yaw/radius plus positive orbit camera pitch degrees above the floor for a source-like viewer camera. Use Bevy/PanOrbit yaw convention: 180 degrees places the camera on the near/source side looking toward positive table depth, 0 degrees places it on the far side. \
Do not annotate, estimate, or invent object bboxes. LocateAnything and segmentation provide the authoritative boxes/masks after planning. Because the manifest schema still requires bbox fields, set object and instance bbox values to [0.0,0.0,1.0,1.0] placeholders only; they are ignored whenever locator evidence is available. \
For repeated reusable objects, set instance_count to the number of visible instances and fill instances only with semantic identity fields such as id/rotation_hint_degrees/facing_yaw_degrees/side/slot_index/target_footprint_m. Do not rely on GPT boxes or contacts to split a group later. Use side=left/right/near/far/head/foot relative to the dominant table in source-image perspective. \
Set representative_instance_id to the clearest single reusable instance identity when known, but do not choose it from a GPT bbox. \
Contact points are supplied by LocateAnything/segmentation/depth when available. For schema-required contacts, use null unless a clear floor contact is already encoded by the detector. \
Object prompts must preserve observed scale relationships and plan shape; do not describe a more symmetric, more closed, more conventional, or more complete product than the source actually shows. \
For the Curry Up Now style sofa specifically, describe the visible source object as a large low tan crescent or semicircular banquette-like sectional with a continuous curved outer back, open center, tufted cushions, and source-cropped foreground extent. It is not a conventional straight L-sectional product sofa and not a closed circle/ring. allow_catalog_reuse={}.",
        scene_path.display(),
        reference_path.display(),
        scene_category_filter_prompt(category_filter),
        allow_catalog_reuse
    )
}

pub fn object_image_prompt(reference_path: &Path, object: &SceneObjectSpec) -> String {
    let camera_hint = object
        .camera_hint
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or("3/4 product camera from slightly above, matching the source crop perspective");
    let rotation_hint = object
        .rotation_hint_degrees
        .map(|degrees| format!("Target yaw/rotation hint: {degrees:.1} degrees."))
        .unwrap_or_else(|| {
            "Use the source crop orientation; do not invent a different canonical view.".to_string()
        });
    let background = reconstruction_background_guidance(object);
    let geometry = object_geometry_guardrails(object);
    let crop_edges = source_crop_edge_guidance(object);
    format!(
        "{}\n{}\n{}\nReference style image: `{}`.\nObject id: {}.\nObject label: {}.\nSource crop bbox: [{:.4},{:.4},{:.4},{:.4}].\nInput image priority: image 1 is the source object crop and is the hard geometry/camera/crop anchor; image 2 is whole-scene context only; image 3 is style reference only. Do not let image 2 or image 3 override the object shape from image 1.\nCamera/orientation: {} {}\nSource-preserving edit requirement: use the source crop as the geometry anchor. Isolate and clean up the same observed object instead of inventing a new product render, new plan shape, or canonical showroom pose. Keep the object's visible perspective, footprint, proportions, curvature, and contact points consistent with the crop. Generate a clean isolated object image for 3D reconstruction. Preserve the source object geometry, material, color, scale proportions, and camera angle. If the observed object is cut off by the source image border, preserve that partial visible source shape instead of completing hidden ends into a new closed product. Do not include the room, rug, table clutter, extra chairs, people, walls, text, shadows cast by the original scene, or background furniture. Do not replace the object with a proxy, cube, simplified block, alternate furniture type, or stylized approximation. Full object visible when possible, but do not hallucinate unobserved shape or wraparound structure to make it look complete. {}\n{}",
        object.object_prompt,
        geometry,
        crop_edges,
        reference_path.display(),
        object.id,
        object.label,
        object.bbox[0],
        object.bbox[1],
        object.bbox[2],
        object.bbox[3],
        camera_hint,
        rotation_hint,
        background,
        "Keep edges crisp and leave clear separation between every thin leg/arm/frame member and the background; avoid contact shadows that merge into the object silhouette."
    )
}

fn source_crop_edge_guidance(object: &SceneObjectSpec) -> String {
    let bbox = normalize_bbox(object.bbox);
    let mut edges = Vec::new();
    if bbox[0] <= 0.035 {
        edges.push("left");
    }
    if bbox[1] <= 0.035 {
        edges.push("top");
    }
    if bbox[2] >= 0.965 {
        edges.push("right");
    }
    if bbox[3] >= 0.965 {
        edges.push("bottom");
    }
    if edges.is_empty() {
        return "Source crop edge constraint: no source image border crop was detected, so keep a complete visible silhouette without adding scene context.".to_string();
    }

    let descriptor = object_descriptor(object);
    let edges = edges.join(", ");
    if descriptor.contains("sofa")
        || descriptor.contains("couch")
        || descriptor.contains("sectional")
        || descriptor.contains("banquette")
    {
        format!(
            "Source crop edge constraint: the observed sofa bbox touches the source image {edges} edge(s). The generated foreground silhouette must continue to the same {edges} edge(s) of the image with no blue/background margin on those cropped sides. Treat those sides as intentional open cut lines from the source photo. Do not center the sofa with padding on cropped sides, do not complete hidden left/right/bottom ends, and do not turn the crop into a finished showroom product sofa."
        )
    } else {
        format!(
            "Source crop edge constraint: the source bbox touches the image {edges} edge(s). Preserve the same cropped extent on those generated image edge(s), with no background padding that changes the visible source silhouette."
        )
    }
}

pub fn object_image_prompt_template() -> String {
    "Input: source scene image + source object crop + docs/input_chair.jpg style reference. Output: source-preserving isolated object image suitable for RMBG and TRELLIS, on a flat high-contrast matte background with crisp object/background separation. Preserve object geometry/material/camera/footprint; remove scene context without inventing a new object.".to_string()
}

fn reconstruction_background_guidance(object: &SceneObjectSpec) -> &'static str {
    let descriptor = format!(
        "{} {} {}",
        object.label,
        object.aliases.join(" "),
        object.object_prompt
    )
    .to_ascii_lowercase();
    let background = if descriptor.contains("green")
        || descriptor.contains("blue")
        || descriptor.contains("teal")
    {
        "solid matte warm coral-orange background (#d95f3f)"
    } else if descriptor.contains("white")
        || descriptor.contains("cream")
        || descriptor.contains("tan")
        || descriptor.contains("beige")
        || descriptor.contains("mustard")
        || descriptor.contains("yellow")
        || descriptor.contains("metal")
        || descriptor.contains("silver")
    {
        "solid matte cobalt-blue background (#1f5fd6)"
    } else {
        "solid matte magenta-purple background (#9b2fd6)"
    };
    match background {
        "solid matte warm coral-orange background (#d95f3f)" => {
            "Use a solid matte warm coral-orange background (#d95f3f), not gray/white/cream, not gradient, not transparent, and no floor plane."
        }
        "solid matte cobalt-blue background (#1f5fd6)" => {
            "Use a solid matte cobalt-blue background (#1f5fd6), not gray/white/cream, not gradient, not transparent, and no floor plane."
        }
        _ => {
            "Use a solid matte magenta-purple background (#9b2fd6), not gray/white/cream, not gradient, not transparent, and no floor plane."
        }
    }
}

fn object_geometry_guardrails(object: &SceneObjectSpec) -> &'static str {
    let descriptor = object_descriptor(object);
    if descriptor.contains("sofa")
        || descriptor.contains("couch")
        || descriptor.contains("sectional")
    {
        "Geometry constraints: preserve the observed source sofa crop as a large low tan curved crescent or semicircular banquette-like sectional, not a new product concept. It should read as one continuous curved sofa with a broad arc, open center, tufted cushions, source-facing perspective, and source-cropped foreground extent. Do not convert it into a conventional straight L-sectional, a blocky modular showroom product, a closed circle/ring, or a fully symmetric horseshoe. Do not output a complete isolated product-render sectional when the source only shows a cropped partial sofa. If the sofa touches a source image edge, keep that visible source extent open/cropped instead of inventing hidden ends. Keep the silhouette wide and low, preserve the curved outer back and open inner void, keep seat thickness uniform, back panels vertical, and legs small/dark where visible."
    } else if descriptor.contains("chair") {
        "Geometry constraints: preserve one complete source-observed chair, not a generic showroom chair. Match the observed support structure from the source crop: keep four separate thin legs or sled/loop frame when visible, and do not invent a central pedestal, swivel column, caster wheels, or five-star base unless that exact base is visible in the source crop. Preserve the observed arm style, back height, seat shape, fabric/mesh material, and contact points. Do not generate multiple chairs in one image."
    } else if descriptor.contains("table") {
        "Geometry constraints: preserve a flat rectangular tabletop with real thickness, four straight vertical legs and/or a slim rectangular metal frame. Do not merge the tabletop into the background. Do not omit thin legs, rails, or feet. Keep all frame lines straight and parallel."
    } else {
        "Geometry constraints: preserve the exact observed object silhouette and proportions from the source crop; do not add extra objects or simplify fine structural members."
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct ObjectImageSuitability {
    pub(crate) background_rgb: [u8; 3],
    pub(crate) contrast_ratio_gt15: f32,
    pub(crate) contrast_ratio_gt25: f32,
    pub(crate) contrast_ratio_gt40: f32,
    pub(crate) score: f32,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ObjectImageMatteStats {
    pub(crate) alpha_coverage: f32,
    pub(crate) alpha_bbox: Option<[u32; 4]>,
    pub(crate) image_size: [u32; 2],
}

pub(crate) fn decode_generated_object_rgb(bytes: &[u8]) -> SceneResult<image::RgbImage> {
    Ok(image::load_from_memory(bytes)
        .map_err(|err| SceneError::Image(format!("decode generated image: {err}")))?
        .to_rgb8())
}

pub(crate) fn score_generated_object_rgb(image: &image::RgbImage) -> ObjectImageSuitability {
    let (width, height) = image.dimensions();
    let short_edge = width.min(height).max(1);
    let corner = (short_edge / 16).clamp(4, 64).min(short_edge);
    let mut sums = [0u64; 3];
    let mut samples = 0u64;
    for y in 0..height {
        for x in 0..width {
            let in_corner = (x < corner || x >= width.saturating_sub(corner))
                && (y < corner || y >= height.saturating_sub(corner));
            if !in_corner {
                continue;
            }
            let pixel = image.get_pixel(x, y).0;
            sums[0] += pixel[0] as u64;
            sums[1] += pixel[1] as u64;
            sums[2] += pixel[2] as u64;
            samples += 1;
        }
    }
    let samples = samples.max(1);
    let background = [
        (sums[0] / samples) as u8,
        (sums[1] / samples) as u8,
        (sums[2] / samples) as u8,
    ];

    let mut gt15 = 0usize;
    let mut gt25 = 0usize;
    let mut gt40 = 0usize;
    let total = width.saturating_mul(height).max(1) as usize;
    for pixel in image.pixels() {
        let rgb = pixel.0;
        let dr = rgb[0] as f32 - background[0] as f32;
        let dg = rgb[1] as f32 - background[1] as f32;
        let db = rgb[2] as f32 - background[2] as f32;
        let distance = (dr * dr + dg * dg + db * db).sqrt();
        if distance > 15.0 {
            gt15 += 1;
        }
        if distance > 25.0 {
            gt25 += 1;
        }
        if distance > 40.0 {
            gt40 += 1;
        }
    }
    let ratio15 = gt15 as f32 / total as f32;
    let ratio25 = gt25 as f32 / total as f32;
    let ratio40 = gt40 as f32 / total as f32;

    let occupancy_score = if ratio25 < 0.03 {
        0.0
    } else if ratio25 < 0.08 {
        (ratio25 - 0.03) / 0.05
    } else if ratio25 <= 0.72 {
        1.0
    } else if ratio25 < 0.90 {
        (0.90 - ratio25) / 0.18
    } else {
        0.0
    };
    let contrast_score = (ratio40 / 0.08).clamp(0.0, 1.0);
    let edge_score = (ratio15 / 0.10).clamp(0.0, 1.0);
    let score = (0.70 * occupancy_score * contrast_score + 0.30 * edge_score).clamp(0.0, 1.0);

    ObjectImageSuitability {
        background_rgb: background,
        contrast_ratio_gt15: ratio15,
        contrast_ratio_gt25: ratio25,
        contrast_ratio_gt40: ratio40,
        score,
    }
}

pub(crate) fn matte_generated_object_rgb(
    image: &image::RgbImage,
    suitability: ObjectImageSuitability,
) -> (image::RgbaImage, ObjectImageMatteStats) {
    let (width, height) = image.dimensions();
    let mut output = image::RgbaImage::new(width, height);
    let bg = suitability.background_rgb;
    let low = 18.0f32;
    let high = 45.0f32;
    let mut foreground = 0usize;
    let mut min_x = width;
    let mut min_y = height;
    let mut max_x = 0u32;
    let mut max_y = 0u32;

    for y in 0..height {
        for x in 0..width {
            let rgb = image.get_pixel(x, y).0;
            let dr = rgb[0] as f32 - bg[0] as f32;
            let dg = rgb[1] as f32 - bg[1] as f32;
            let db = rgb[2] as f32 - bg[2] as f32;
            let distance = (dr * dr + dg * dg + db * db).sqrt();
            let alpha = if distance <= low {
                0
            } else if distance >= high {
                255
            } else {
                (((distance - low) / (high - low)) * 255.0)
                    .round()
                    .clamp(0.0, 255.0) as u8
            };
            if alpha > 127 {
                foreground += 1;
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
            }
            output.put_pixel(x, y, image::Rgba([rgb[0], rgb[1], rgb[2], alpha]));
        }
    }

    let total = width.saturating_mul(height).max(1) as f32;
    let alpha_bbox = if foreground > 0 {
        Some([min_x, min_y, max_x + 1, max_y + 1])
    } else {
        None
    };
    (
        output,
        ObjectImageMatteStats {
            alpha_coverage: foreground as f32 / total,
            alpha_bbox,
            image_size: [width, height],
        },
    )
}

pub fn scene_bsn_prompt(manifest: &SceneObjectManifest, assets: &[SceneAssetBinding]) -> String {
    format!(
        "Create a restricted synth_scene_v1 BSN scene using only these generated asset ids: {}. \
The source manifest has {} object specs from {}. \
Use repeated chair instances from the same reusable chair asset where appropriate. \
Furniture must be spawned with generated assets only. Rug/floor may be environment primitives. \
Every statement must be on exactly one line. Do not split asset, spawn, camera, or environment statements across lines. \
Use only this grammar:\n\
synth_scene_v1 {{\n\
asset <asset_id> = \"generated:<asset_id>\";\n\
spawn <entity_id> uses <asset_id> translation [x,y,z] rotation_y <degrees> scale [x,y,z];\n\
environment rug translation [x,y,z] scale [x,y,z] color [r,g,b];\n\
camera translation [x,y,z] focus [x,y,z] yaw <degrees> pitch <degrees> radius <value>;\n\
}}\n\
Emit only valid synth_scene_v1 text.",
        assets
            .iter()
            .map(|asset| asset.asset_id.as_str())
            .collect::<Vec<_>>()
            .join(", "),
        manifest.objects.len(),
        manifest.source_scene_path
    )
}

pub(crate) fn generated_shape_consistency_score(
    object: &SceneObjectSpec,
    matte: &ObjectImageMatteStats,
    source_image_aspect: f32,
) -> f32 {
    if object.instance_count > 1 || !object.instances.is_empty() {
        return 1.0;
    }
    let descriptor = object_descriptor(object);
    let is_sofa = descriptor.contains("sofa")
        || descriptor.contains("couch")
        || descriptor.contains("sectional");
    let strict_ratio = if is_sofa {
        if descriptor.contains("crescent")
            || descriptor.contains("semicircular")
            || descriptor.contains("semicircle")
            || descriptor.contains("banquette")
            || descriptor.contains("curved")
        {
            Some(0.20)
        } else {
            let bbox = normalize_bbox(object.bbox);
            let touches_horizontal_edge = bbox[0] <= 0.02 || bbox[2] >= 0.98;
            let spans_scene_width = (bbox[2] - bbox[0]) >= 0.85;
            Some(if touches_horizontal_edge && spans_scene_width {
                0.90
            } else {
                0.86
            })
        }
    } else if descriptor.contains("table") {
        Some(0.45)
    } else {
        None
    };
    let Some(min_ratio) = strict_ratio else {
        return 1.0;
    };
    let Some(alpha_bbox) = matte.alpha_bbox else {
        return 0.0;
    };
    let bbox = normalize_bbox(object.bbox);
    let source_w = (bbox[2] - bbox[0]).max(1.0e-5) * source_image_aspect.max(0.1);
    let source_h = (bbox[3] - bbox[1]).max(1.0e-5);
    let source_aspect = source_w / source_h;
    let alpha_w = alpha_bbox[2].saturating_sub(alpha_bbox[0]).max(1) as f32;
    let alpha_h = alpha_bbox[3].saturating_sub(alpha_bbox[1]).max(1) as f32;
    let generated_aspect = alpha_w / alpha_h;
    let ratio =
        (source_aspect.min(generated_aspect) / source_aspect.max(generated_aspect)).clamp(0.0, 1.0);
    if ratio < min_ratio { 0.0 } else { ratio }
}

pub(crate) fn generated_source_crop_edge_mismatch(
    object: &SceneObjectSpec,
    matte: &ObjectImageMatteStats,
) -> bool {
    let Some(alpha_bbox) = matte.alpha_bbox else {
        return true;
    };
    generated_open_sofa_lost_source_crop_edge(
        &object_descriptor(object),
        normalize_bbox(object.bbox),
        matte,
        alpha_bbox,
    )
}

pub(crate) fn object_reconstruction_min_score(
    object: &SceneObjectSpec,
    base_min_score: f32,
) -> f32 {
    let base_min_score = base_min_score.clamp(0.0, 1.0);
    if object.instance_count > 1 || !object.instances.is_empty() {
        return base_min_score;
    }
    let descriptor = object_descriptor(object);
    if descriptor.contains("table") || descriptor.contains("desk") || descriptor.contains("counter")
    {
        base_min_score.max(0.60)
    } else if descriptor.contains("sofa")
        || descriptor.contains("couch")
        || descriptor.contains("sectional")
        || descriptor.contains("banquette")
    {
        base_min_score.max(0.48)
    } else {
        base_min_score
    }
}

fn generated_open_sofa_lost_source_crop_edge(
    descriptor: &str,
    source_bbox: [f32; 4],
    matte: &ObjectImageMatteStats,
    alpha_bbox: [u32; 4],
) -> bool {
    let sofa_like = descriptor.contains("sofa")
        || descriptor.contains("couch")
        || descriptor.contains("sectional")
        || descriptor.contains("banquette");
    let open_like = descriptor.contains("open")
        || descriptor.contains("crescent")
        || descriptor.contains("curved")
        || descriptor.contains("semicircular")
        || descriptor.contains("semicircle")
        || descriptor.contains("sectional");
    if !(sofa_like && open_like) {
        return false;
    }

    let [width, height] = matte.image_size;
    let width = width.max(1) as f32;
    let height = height.max(1) as f32;
    let generated = [
        alpha_bbox[0] as f32 / width,
        alpha_bbox[1] as f32 / height,
        alpha_bbox[2] as f32 / width,
        alpha_bbox[3] as f32 / height,
    ];
    let source_edge = 0.035;
    let generated_edge = 0.042;
    let generated_far_edge = 0.958;
    (source_bbox[0] <= source_edge && generated[0] > generated_edge)
        || (source_bbox[1] <= source_edge && generated[1] > generated_edge)
        || (source_bbox[2] >= 1.0 - source_edge && generated[2] < generated_far_edge)
        || (source_bbox[3] >= 1.0 - source_edge && generated[3] < generated_far_edge)
}

pub(crate) fn image_dimensions_aspect(path: &Path) -> SceneResult<f32> {
    let (width, height) = image::image_dimensions(path)?;
    Ok(width.max(1) as f32 / height.max(1) as f32)
}

use std::fs;
use std::path::Path;
use std::time::Instant;

use burn_segmentation::{
    BinaryMask, SegmentationPrompt, SegmentationRuntime, write_mask_overlay, write_mask_png,
};
use burn_synth_scene::{
    ObjectGroundingEvidence, ObjectMaskEvidence, SceneGroundingEvidence, SegmentationEvidenceRef,
    write_json_file,
};

use crate::image_util::{bbox_bottom_center, sanitize_artifact_stem};
use crate::types::*;

impl SceneGroundingRuntime {
    pub fn segmentation_grounding_evidence(
        &mut self,
        evidence: &mut SceneGroundingEvidence,
        source_scene_path: &Path,
        output_dir: &Path,
        config: SegmentationGroundingConfig,
    ) -> Result<SegmentationGroundingReport, String> {
        let artifact_dir = output_dir.join(format!("segmentation_{}", config.model.label()));
        let masks_dir = artifact_dir.join("masks");
        fs::create_dir_all(&masks_dir).map_err(|err| {
            format!(
                "failed to create segmentation artifact directory {}: {err}",
                masks_dir.display()
            )
        })?;
        let image = image::open(source_scene_path).map_err(|err| {
            format!(
                "failed to load segmentation source image {}: {err}",
                source_scene_path.display()
            )
        })?;
        let (object_indices, prompts) = segmentation_prompts_from_evidence(evidence);
        if prompts.is_empty() {
            return Err(
                "segmentation grounding requires at least one object with bbox evidence"
                    .to_string(),
            );
        }

        let runtime_config = config.runtime_config();
        let cache_key = SegmentationRuntimeCacheKey::from_config(&runtime_config);
        let cache_hit = self
            .segmentation_runtime
            .as_ref()
            .is_some_and(|cached| cached.key == cache_key);
        if !cache_hit {
            let runtime = SegmentationRuntime::new(runtime_config.clone())
                .map_err(|err| format!("initialize segmentation runtime: {err}"))?;
            self.segmentation_runtime = Some(CachedSegmentationRuntime {
                key: cache_key,
                runtime,
            });
        }
        let runtime = &mut self
            .segmentation_runtime
            .as_mut()
            .expect("segmentation runtime cache initialized")
            .runtime;

        let started = Instant::now();
        let mut masks = runtime
            .segment(&image, &prompts)
            .map_err(|err| format!("segmentation inference failed: {err}"))?;
        for (index, mask) in masks.iter_mut().enumerate() {
            let png_path = masks_dir.join(format!(
                "{index:03}_{}.png",
                sanitize_artifact_stem(&mask.object_id)
            ));
            let binary = BinaryMask::decode_rle(mask.width, mask.height, &mask.mask_rle)
                .map_err(|err| format!("decode segmentation mask {}: {err}", mask.object_id))?;
            write_mask_png(&binary, &png_path)
                .map_err(|err| format!("write segmentation mask {}: {err}", mask.object_id))?;
            mask.mask_png_path = Some(png_path.display().to_string());
        }
        let overlay_path = artifact_dir.join("masks_overlay.png");
        write_mask_overlay(&image, &masks, &overlay_path)
            .map_err(|err| format!("write segmentation overlay: {err}"))?;
        let masks_path = artifact_dir.join("masks.json");
        write_json_file(&masks_path, &masks).map_err(|err| err.to_string())?;
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;

        evidence.segmentation = Some(SegmentationEvidenceRef {
            provider: config.backend.label().to_string(),
            model: config.model.label().to_string(),
            artifact_path: Some(masks_path.display().to_string()),
            overlay_path: Some(overlay_path.display().to_string()),
            mask_count: Some(masks.len()),
        });
        for (mask, object_index) in masks.iter().zip(object_indices.iter().copied()) {
            if let Some(object) = evidence.objects.get_mut(object_index) {
                object.mask = Some(ObjectMaskEvidence {
                    provider: mask.provider.clone(),
                    model: mask.model.clone(),
                    bbox: mask.bbox,
                    score: mask.score,
                    area_px: mask.area_px,
                    image_size: [mask.width, mask.height],
                    mask_rle: mask.mask_rle.clone(),
                    center_pixel: mask_center_pixel(mask),
                    contact_pixel: mask_contact_pixel(mask),
                    coverage: mask_coverage(mask),
                    artifact_path: Some(masks_path.display().to_string()),
                    mask_png_path: mask.mask_png_path.clone(),
                });
                if let Some(contact) = object.mask.as_ref().and_then(|mask| mask.contact_pixel) {
                    object.contact_pixel = Some(contact);
                }
                let provenance = format!("segmentation_{}", config.model.label());
                if !object.provenance.iter().any(|entry| entry == &provenance) {
                    object.provenance.push(provenance);
                }
            }
        }

        Ok(SegmentationGroundingReport {
            artifact_dir,
            masks_path,
            overlay_path,
            elapsed_ms,
            runtime_cache_hit: cache_hit,
            model_variant: runtime
                .sam_image_encoder_variant()
                .map(|variant| variant.label().to_string()),
            mask_count: masks.len(),
            stage_timings: runtime.last_stage_timings(),
        })
    }
}

fn mask_center_pixel(mask: &burn_segmentation::SegmentationMask) -> Option<[f32; 2]> {
    let binary = BinaryMask::decode_rle(mask.width, mask.height, &mask.mask_rle).ok()?;
    normalized_mask_center(&binary)
}

fn mask_contact_pixel(mask: &burn_segmentation::SegmentationMask) -> Option<[f32; 2]> {
    let binary = BinaryMask::decode_rle(mask.width, mask.height, &mask.mask_rle).ok()?;
    normalized_mask_bottom_center(&binary)
        .or_else(|| binary.bbox_normalized().map(bbox_bottom_center))
}

fn mask_coverage(mask: &burn_segmentation::SegmentationMask) -> Option<f32> {
    let total = mask.width.checked_mul(mask.height)?;
    (total > 0).then_some(mask.area_px as f32 / total as f32)
}

fn normalized_mask_center(mask: &BinaryMask) -> Option<[f32; 2]> {
    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;
    let mut count = 0.0f64;
    for y in 0..mask.height() {
        for x in 0..mask.width() {
            if mask.data()[y as usize * mask.width() as usize + x as usize] == 0 {
                continue;
            }
            sum_x += x as f64 + 0.5;
            sum_y += y as f64 + 0.5;
            count += 1.0;
        }
    }
    (count > 0.0).then_some([
        (sum_x / count / mask.width().max(1) as f64) as f32,
        (sum_y / count / mask.height().max(1) as f64) as f32,
    ])
}

fn normalized_mask_bottom_center(mask: &BinaryMask) -> Option<[f32; 2]> {
    let mut bottom_y = None::<u32>;
    for y in 0..mask.height() {
        let row = y as usize * mask.width() as usize;
        if (0..mask.width()).any(|x| mask.data()[row + x as usize] != 0) {
            bottom_y = Some(y);
        }
    }
    let y = bottom_y?;
    let row = y as usize * mask.width() as usize;
    let mut sum_x = 0.0f64;
    let mut count = 0.0f64;
    for x in 0..mask.width() {
        if mask.data()[row + x as usize] != 0 {
            sum_x += x as f64 + 0.5;
            count += 1.0;
        }
    }
    (count > 0.0).then_some([
        (sum_x / count / mask.width().max(1) as f64) as f32,
        ((y as f64 + 0.5) / mask.height().max(1) as f64) as f32,
    ])
}

fn segmentation_prompts_from_evidence(
    evidence: &SceneGroundingEvidence,
) -> (Vec<usize>, Vec<SegmentationPrompt>) {
    let mut object_indices = Vec::new();
    let mut prompts = Vec::new();
    for (index, object) in evidence.objects.iter().enumerate() {
        let Some(detection) = object.detection.as_ref() else {
            continue;
        };
        let label = if detection.label.trim().is_empty() {
            object.object_id.clone()
        } else {
            detection.label.clone()
        };
        object_indices.push(index);
        prompts.push(SegmentationPrompt {
            object_id: segmentation_prompt_object_id(index, object),
            label,
            bbox: detection.bbox,
            point: detection.point.or(object.contact_pixel),
            source_query: Some(detection.source_query.clone()),
        });
    }
    (object_indices, prompts)
}

fn segmentation_prompt_object_id(index: usize, object: &ObjectGroundingEvidence) -> String {
    match object.instance_id.as_deref() {
        Some(instance_id) => format!("{:03}_{}_{}", index, object.object_id, instance_id),
        None => format!("{:03}_{}", index, object.object_id),
    }
}

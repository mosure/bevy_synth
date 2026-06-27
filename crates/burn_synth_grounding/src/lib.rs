use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use burn::prelude::{Backend, Tensor};
use burn_depth::{
    CameraIntrinsics, DepthCheckpointSource, DepthLoadConfig, DepthLoadEvent, DepthLoadStage,
    DepthModelKind, DepthPipeline, DepthPrecision, DepthRuntimeConfig, ImageBoundingBox,
    backproject_depth, depth_at_bbox_contact_region, estimate_floor_plane, pixel_to_ray,
};
use burn_locate_anything::LocateAnythingDetector;
pub use burn_locate_anything::{
    DecodeMode, Detection as LocateAnythingDetection, DetectionQuery,
    LOCATE_ANYTHING_SAFE_IN_TOKEN_LIMIT, LocateAnythingRuntime, LocateAnythingRuntimeBackend,
    LocateAnythingRuntimeConfig,
};
use burn_segmentation::{
    BinaryMask, SegmentationPrompt, SegmentationRuntime, SegmentationRuntimeConfig,
    write_mask_overlay, write_mask_png,
};
pub use burn_segmentation::{
    SegmentationModelKind, SegmentationPrecision, SegmentationQuantization,
    SegmentationRuntimeBackend,
};
use burn_synth_scene::{
    DepthEvidenceRef, Detection, EstimatedCamera, EstimatedFloorPlane, ObjectDepthStats,
    ObjectGroundingEvidence, ObjectMaskEvidence, SceneGroundingEvidence, SceneObjectInstanceSpec,
    SceneObjectManifest, SceneObjectSpec, SegmentationEvidenceRef, write_json_file,
};
use image::{Rgba, RgbaImage};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GroundingDepthPrecision {
    F32,
    F16,
}

impl GroundingDepthPrecision {
    pub fn label(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
        }
    }
}

impl From<GroundingDepthPrecision> for DepthPrecision {
    fn from(value: GroundingDepthPrecision) -> Self {
        match value {
            GroundingDepthPrecision::F32 => DepthPrecision::F32,
            GroundingDepthPrecision::F16 => DepthPrecision::F16,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DepthProGroundingConfig {
    pub cache_dir: Option<PathBuf>,
    pub precision: GroundingDepthPrecision,
    pub allow_download: bool,
    pub require_gpu: bool,
}

impl Default for DepthProGroundingConfig {
    fn default() -> Self {
        Self {
            cache_dir: None,
            precision: GroundingDepthPrecision::F16,
            allow_download: true,
            require_gpu: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LocateAnythingGroundingConfig {
    pub model_root: PathBuf,
    pub in_token_limit: usize,
    pub decode_mode: DecodeMode,
    pub max_new_tokens: usize,
    pub repetition_penalty: f32,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
}

impl Default for LocateAnythingGroundingConfig {
    fn default() -> Self {
        let runtime = LocateAnythingRuntimeConfig::default();
        Self {
            model_root: PathBuf::from("assets/models/LocateAnything-3B"),
            in_token_limit: LOCATE_ANYTHING_SAFE_IN_TOKEN_LIMIT as usize,
            decode_mode: DecodeMode::Hybrid,
            max_new_tokens: runtime.max_new_tokens,
            repetition_penalty: runtime.repetition_penalty,
            top_p: runtime.top_p,
            top_k: runtime.top_k,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SegmentationGroundingConfig {
    pub model: SegmentationModelKind,
    pub backend: SegmentationRuntimeBackend,
    pub model_root: Option<PathBuf>,
    pub cache_dir: Option<PathBuf>,
    pub cdn_base_url: Option<String>,
    pub precision: SegmentationPrecision,
    pub quantization: SegmentationQuantization,
    pub allow_download: bool,
    pub require_gpu: bool,
}

impl Default for SegmentationGroundingConfig {
    fn default() -> Self {
        Self {
            model: SegmentationModelKind::BboxPrompt,
            backend: SegmentationRuntimeBackend::BboxPrompt,
            model_root: None,
            cache_dir: None,
            cdn_base_url: None,
            precision: SegmentationPrecision::default(),
            quantization: SegmentationQuantization::default(),
            allow_download: false,
            require_gpu: true,
        }
    }
}

impl SegmentationGroundingConfig {
    fn runtime_config(&self) -> SegmentationRuntimeConfig {
        SegmentationRuntimeConfig {
            model: self.model,
            backend: self.backend,
            model_root: self.model_root.clone(),
            cache_dir: self.cache_dir.clone(),
            cdn_base_url: self.cdn_base_url.clone(),
            precision: self.precision,
            quantization: self.quantization,
            allow_download: self.allow_download,
            require_gpu: self.require_gpu,
            profile_stages: false,
        }
    }
}

impl LocateAnythingGroundingConfig {
    fn runtime_config(&self) -> LocateAnythingRuntimeConfig {
        LocateAnythingRuntimeConfig {
            model_root: self.model_root.clone(),
            backend: LocateAnythingRuntimeBackend::BurnNative,
            allow_experimental_native_detect: true,
            decode_mode: self.decode_mode,
            max_new_tokens: self.max_new_tokens,
            in_token_limit: self.in_token_limit.max(1) as u32,
            repetition_penalty: self.repetition_penalty,
            top_p: self.top_p,
            top_k: self.top_k,
            ..LocateAnythingRuntimeConfig::default()
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LocateAnythingBurnNativeCacheKey {
    model_root: PathBuf,
    decode_mode: DecodeMode,
    max_new_tokens: usize,
    in_token_limit: u32,
    repetition_penalty_bits: u32,
    top_p_bits: Option<u32>,
    top_k: Option<usize>,
}

impl LocateAnythingBurnNativeCacheKey {
    pub fn from_config(config: &LocateAnythingRuntimeConfig) -> Self {
        Self {
            model_root: config.model_root.clone(),
            decode_mode: config.decode_mode,
            max_new_tokens: config.max_new_tokens,
            in_token_limit: config.in_token_limit,
            repetition_penalty_bits: config.repetition_penalty.to_bits(),
            top_p_bits: config.top_p.map(f32::to_bits),
            top_k: config.top_k,
        }
    }
}

#[derive(Default)]
pub struct SceneGroundingRuntime {
    depth_pro_runtime: Option<CachedDepthProRuntime>,
    locate_anything_burn_native_runtime: Option<CachedLocateAnythingRuntime>,
    segmentation_runtime: Option<CachedSegmentationRuntime>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DepthProRuntimeCacheKey {
    cache_dir: Option<PathBuf>,
    precision: GroundingDepthPrecision,
}

impl DepthProRuntimeCacheKey {
    pub fn from_config(config: &DepthProGroundingConfig) -> Self {
        Self {
            cache_dir: config.cache_dir.clone(),
            precision: config.precision,
        }
    }
}

struct CachedDepthProRuntime {
    key: DepthProRuntimeCacheKey,
    pipeline: DepthPipeline<burn_depth::InferenceBackend>,
}

struct CachedLocateAnythingRuntime {
    key: LocateAnythingBurnNativeCacheKey,
    runtime: LocateAnythingRuntime,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SegmentationRuntimeCacheKey {
    model: SegmentationModelKind,
    backend: SegmentationRuntimeBackend,
    model_root: Option<PathBuf>,
    cache_dir: Option<PathBuf>,
    cdn_base_url: Option<String>,
    precision: SegmentationPrecision,
    quantization: SegmentationQuantization,
}

impl SegmentationRuntimeCacheKey {
    pub fn from_config(config: &SegmentationRuntimeConfig) -> Self {
        Self {
            model: config.model,
            backend: config.backend,
            model_root: config.model_root.clone(),
            cache_dir: config.cache_dir.clone(),
            cdn_base_url: config.cdn_base_url.clone(),
            precision: config.precision,
            quantization: config.quantization,
        }
    }
}

struct CachedSegmentationRuntime {
    key: SegmentationRuntimeCacheKey,
    runtime: SegmentationRuntime,
}

#[derive(Clone, Debug, Serialize)]
pub struct DepthProGroundingReport {
    pub artifact_path: PathBuf,
    pub load_ms: f64,
    pub infer_ms: f64,
    pub runtime_cache_hit: bool,
    pub summary: SceneDepthAnnotationSummary,
}

#[derive(Clone, Debug, Serialize)]
pub struct LocateAnythingGroundingReport {
    pub artifact_dir: PathBuf,
    pub detections_path: PathBuf,
    pub metadata_path: PathBuf,
    pub elapsed_ms: f64,
    pub runtime_cache_hit: bool,
    pub detection_count: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct SegmentationGroundingReport {
    pub artifact_dir: PathBuf,
    pub masks_path: PathBuf,
    pub overlay_path: PathBuf,
    pub elapsed_ms: f64,
    pub runtime_cache_hit: bool,
    pub mask_count: usize,
}

#[derive(Clone, Debug)]
pub struct SceneDepthMapEvidence {
    pub depth_m: Vec<f32>,
    pub width: u32,
    pub height: u32,
    pub intrinsics: CameraIntrinsics,
    pub focal_length_px: Option<f32>,
    pub vertical_fov_degrees: Option<f32>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct SceneDepthAnnotationSummary {
    pub provider: String,
    pub annotated_objects: usize,
    pub total_objects: usize,
    pub depth_map_size: [u32; 2],
    pub focal_length_px: Option<f32>,
    pub vertical_fov_degrees: Option<f32>,
    pub floor_sample_count: usize,
    pub floor_residual_m: Option<f32>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct SceneFarFieldFilterSummary {
    pub enabled: bool,
    pub threshold_m: Option<f32>,
    pub median_object_depth_m: Option<f32>,
    pub lower_quartile_object_depth_m: Option<f32>,
    pub removed_detections: usize,
    pub removed_objects: usize,
    pub removed_detection_labels: Vec<String>,
    pub removed_object_ids: Vec<String>,
}

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
        evidence.floor = estimate_scene_floor_plane(&depth_map).unwrap_or_default();

        Ok(DepthProGroundingReport {
            artifact_path: summary_path,
            load_ms,
            infer_ms,
            runtime_cache_hit: cache_hit,
            summary,
        })
    }

    pub fn locate_anything_burn_native_grounding_evidence(
        &mut self,
        manifest: &SceneObjectManifest,
        source_scene_path: &Path,
        output_dir: &Path,
        config: LocateAnythingGroundingConfig,
    ) -> Result<(SceneGroundingEvidence, LocateAnythingGroundingReport), String> {
        let queries = locate_anything_queries(manifest);
        if queries.is_empty() {
            return Err(
                "LocateAnything locator requires at least one non-empty manifest object label"
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
        let detections = batches
            .into_iter()
            .flatten()
            .map(scene_detection_from_locate_anything)
            .collect::<Vec<_>>();
        if detections.is_empty() {
            return Err(format!(
                "Burn-native LocateAnything returned no detections for {} queries",
                queries.len()
            ));
        }

        let detections_path = artifact_dir.join("detections.json");
        write_json_file(&detections_path, &detections).map_err(|err| err.to_string())?;
        let overlay_path = artifact_dir.join("detections_overlay.png");
        write_detection_overlay(source_scene_path, &detections, &overlay_path)?;
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
            "elapsed_ms": elapsed_ms,
            "stage_timings": stage_timings,
            "runtime_cache_hit": cache_hit,
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
            metadata_path,
            elapsed_ms,
            runtime_cache_hit: cache_hit,
            detection_count: evidence.detections.len(),
        };
        Ok((evidence, report))
    }

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
                    artifact_path: Some(masks_path.display().to_string()),
                    mask_png_path: mask.mask_png_path.clone(),
                });
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
            mask_count: masks.len(),
        })
    }
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

fn sanitize_artifact_stem(value: &str) -> String {
    let mut out = String::with_capacity(value.len().max(1));
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() { "_".to_string() } else { out }
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
    let (floor, floor_sample_count) =
        estimate_scene_floor_plane_with_exclusions(depth_map, &floor_exclusions)
            .map(|(floor, count)| (Some(floor), count))
            .unwrap_or_else(|| {
                let floor = estimate_scene_floor_plane(depth_map);
                (floor, floor_sample_count_with_exclusions(depth_map, &[]))
            });
    let mut annotated_objects = 0usize;
    for object in &mut evidence.objects {
        let Some(detection) = object.detection.as_ref() else {
            continue;
        };
        let bbox = normalized_bbox_to_image_bbox(detection.bbox, depth_map.width, depth_map.height);
        let bbox_stats =
            depth_stats_for_bbox(&depth_map.depth_m, depth_map.width, depth_map.height, bbox);
        let contact_pixel = object
            .contact_pixel
            .or(detection.point)
            .unwrap_or_else(|| bbox_bottom_center(detection.bbox));
        let contact_depth = depth_at_bbox_contact_region(
            &depth_map.depth_m,
            depth_map.width,
            depth_map.height,
            bbox,
        )
        .or_else(|| sample_depth_at_normalized_pixel(depth_map, contact_pixel));
        let Some(contact_depth) = contact_depth.filter(|value| value.is_finite() && *value > 0.0)
        else {
            continue;
        };

        let pixel = normalized_to_depth_pixel(contact_pixel, depth_map.width, depth_map.height);
        let ray = pixel_to_ray(pixel[0], pixel[1], depth_map.intrinsics);
        let point = backproject_depth(pixel[0], pixel[1], contact_depth, depth_map.intrinsics);
        let target_footprint =
            estimate_depth_target_footprint(detection, bbox, contact_depth, depth_map.intrinsics);
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
        floor_sample_count,
        floor_residual_m: floor.and_then(|floor| floor.residual_m),
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

fn bbox_area_normalized(bbox: [f32; 4]) -> f32 {
    (bbox[2] - bbox[0]).abs().clamp(0.0, 1.0) * (bbox[3] - bbox[1]).abs().clamp(0.0, 1.0)
}

pub fn estimate_scene_floor_plane(
    depth_map: &SceneDepthMapEvidence,
) -> Option<EstimatedFloorPlane> {
    estimate_scene_floor_plane_with_exclusions(depth_map, &[]).map(|(floor, _)| floor)
}

fn estimate_scene_floor_plane_with_exclusions(
    depth_map: &SceneDepthMapEvidence,
    exclusion_bboxes: &[[f32; 4]],
) -> Option<(EstimatedFloorPlane, usize)> {
    let mut points = Vec::new();
    let step_x = (depth_map.width / 64).max(1);
    let step_y = (depth_map.height / 48).max(1);
    let y_start = (depth_map.height as f32 * 0.62).floor() as u32;
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
                points.push(backproject_depth(
                    x as f32 + 0.5,
                    y as f32 + 0.5,
                    depth,
                    depth_map.intrinsics,
                ));
            }
        }
    }
    if points.len() < 32 {
        return None;
    }
    let plane = estimate_floor_plane(&points)?;
    let residual = if points.is_empty() {
        None
    } else {
        let sum = points
            .iter()
            .map(|point| {
                (plane.normal[0] * point[0]
                    + plane.normal[1] * point[1]
                    + plane.normal[2] * point[2]
                    + plane.d)
                    .abs()
            })
            .sum::<f32>();
        Some(sum / points.len() as f32)
    };
    let floor = EstimatedFloorPlane {
        normal: plane.normal,
        distance_m: plane.d,
        residual_m: residual,
        confidence: Some((1.0 / (1.0 + residual.unwrap_or(1.0))).clamp(0.0, 1.0)),
    };
    Some((floor, points.len()))
}

fn floor_sample_count_with_exclusions(
    depth_map: &SceneDepthMapEvidence,
    exclusion_bboxes: &[[f32; 4]],
) -> usize {
    let step_x = (depth_map.width / 64).max(1);
    let step_y = (depth_map.height / 48).max(1);
    let y_start = (depth_map.height as f32 * 0.62).floor() as u32;
    let mut count = 0usize;
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
                count += 1;
            }
        }
    }
    count
}

fn floor_sample_exclusion_bboxes(evidence: &SceneGroundingEvidence) -> Vec<[f32; 4]> {
    let mut bboxes = evidence
        .detections
        .iter()
        .map(|detection| detection.bbox)
        .collect::<Vec<_>>();
    for object in &evidence.objects {
        if let Some(detection) = object.detection.as_ref()
            && !bboxes.iter().any(|bbox| bbox == &detection.bbox)
        {
            bboxes.push(detection.bbox);
        }
    }
    bboxes
}

fn floor_sample_excluded(pixel: [f32; 2], exclusion_bboxes: &[[f32; 4]]) -> bool {
    const MARGIN: f32 = 0.015;
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

fn write_detection_overlay(
    source_scene_path: &Path,
    detections: &[Detection],
    output_path: &Path,
) -> Result<(), String> {
    let mut image = image::open(source_scene_path)
        .map_err(|err| {
            format!(
                "failed to load source image for detection overlay {}: {err}",
                source_scene_path.display()
            )
        })?
        .to_rgba8();
    for (index, detection) in detections.iter().enumerate() {
        let color = overlay_color(index);
        draw_normalized_bbox(&mut image, detection.bbox, color, 4);
        draw_normalized_cross(
            &mut image,
            detection
                .point
                .unwrap_or_else(|| bbox_bottom_center(detection.bbox)),
            color,
            8,
        );
    }
    image.save(output_path).map_err(|err| {
        format!(
            "failed to write detection overlay {}: {err}",
            output_path.display()
        )
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
    let step_x = (depth_map.width / 64).max(1);
    let step_y = (depth_map.height / 48).max(1);
    let y_start = (depth_map.height as f32 * 0.62).floor() as u32;
    for y in (y_start..depth_map.height).step_by(step_y as usize) {
        for x in (0..depth_map.width).step_by(step_x as usize) {
            let normalized = [
                x as f32 / depth_map.width.saturating_sub(1).max(1) as f32,
                y as f32 / depth_map.height.saturating_sub(1).max(1) as f32,
            ];
            if floor_sample_excluded(normalized, &exclusion_bboxes) {
                continue;
            }
            let index = y as usize * depth_map.width as usize + x as usize;
            let depth = depth_map.depth_m.get(index).copied().unwrap_or_default();
            if depth.is_finite() && depth > 0.0 {
                draw_normalized_square(&mut image, normalized, Rgba([0, 220, 255, 255]), 2);
            }
        }
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

fn overlay_color(index: usize) -> Rgba<u8> {
    const COLORS: [[u8; 4]; 8] = [
        [230, 57, 70, 255],
        [42, 157, 143, 255],
        [69, 123, 157, 255],
        [233, 196, 106, 255],
        [131, 56, 236, 255],
        [255, 128, 0, 255],
        [0, 180, 216, 255],
        [255, 0, 110, 255],
    ];
    Rgba(COLORS[index % COLORS.len()])
}

fn draw_normalized_bbox(image: &mut RgbaImage, bbox: [f32; 4], color: Rgba<u8>, thickness: u32) {
    let width = image.width();
    let height = image.height();
    if width == 0 || height == 0 {
        return;
    }
    let x0 = (bbox[0].min(bbox[2]).clamp(0.0, 1.0) * width.saturating_sub(1) as f32).round() as i32;
    let x1 = (bbox[0].max(bbox[2]).clamp(0.0, 1.0) * width.saturating_sub(1) as f32).round() as i32;
    let y0 =
        (bbox[1].min(bbox[3]).clamp(0.0, 1.0) * height.saturating_sub(1) as f32).round() as i32;
    let y1 =
        (bbox[1].max(bbox[3]).clamp(0.0, 1.0) * height.saturating_sub(1) as f32).round() as i32;
    let thickness = thickness.max(1) as i32;
    for offset in 0..thickness {
        draw_line(image, x0, y0 + offset, x1, y0 + offset, color);
        draw_line(image, x0, y1 - offset, x1, y1 - offset, color);
        draw_line(image, x0 + offset, y0, x0 + offset, y1, color);
        draw_line(image, x1 - offset, y0, x1 - offset, y1, color);
    }
}

fn draw_normalized_cross(image: &mut RgbaImage, pixel: [f32; 2], color: Rgba<u8>, radius: i32) {
    let width = image.width();
    let height = image.height();
    if width == 0 || height == 0 {
        return;
    }
    let x = (pixel[0].clamp(0.0, 1.0) * width.saturating_sub(1) as f32).round() as i32;
    let y = (pixel[1].clamp(0.0, 1.0) * height.saturating_sub(1) as f32).round() as i32;
    draw_line(image, x - radius, y, x + radius, y, color);
    draw_line(image, x, y - radius, x, y + radius, color);
}

fn draw_normalized_square(image: &mut RgbaImage, pixel: [f32; 2], color: Rgba<u8>, radius: i32) {
    let width = image.width();
    let height = image.height();
    if width == 0 || height == 0 {
        return;
    }
    let x = (pixel[0].clamp(0.0, 1.0) * width.saturating_sub(1) as f32).round() as i32;
    let y = (pixel[1].clamp(0.0, 1.0) * height.saturating_sub(1) as f32).round() as i32;
    for yy in (y - radius)..=(y + radius) {
        for xx in (x - radius)..=(x + radius) {
            put_pixel_checked(image, xx, yy, color);
        }
    }
}

fn draw_line(image: &mut RgbaImage, x0: i32, y0: i32, x1: i32, y1: i32, color: Rgba<u8>) {
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let mut x = x0;
    let mut y = y0;
    loop {
        put_pixel_checked(image, x, y, color);
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
}

fn put_pixel_checked(image: &mut RgbaImage, x: i32, y: i32, color: Rgba<u8>) {
    if x >= 0 && y >= 0 && x < image.width() as i32 && y < image.height() as i32 {
        image.put_pixel(x as u32, y as u32, color);
    }
}

fn normalized_bbox_to_image_bbox(
    bbox: [f32; 4],
    image_width: u32,
    image_height: u32,
) -> ImageBoundingBox {
    let bbox = [
        bbox[0].clamp(0.0, 1.0),
        bbox[1].clamp(0.0, 1.0),
        bbox[2].clamp(0.0, 1.0),
        bbox[3].clamp(0.0, 1.0),
    ];
    let x0 = (bbox[0].min(bbox[2]) * image_width as f32).floor() as u32;
    let x1 = (bbox[0].max(bbox[2]) * image_width as f32).ceil() as u32;
    let y0 = (bbox[1].min(bbox[3]) * image_height as f32).floor() as u32;
    let y1 = (bbox[1].max(bbox[3]) * image_height as f32).ceil() as u32;
    let x0 = x0.min(image_width.saturating_sub(1));
    let y0 = y0.min(image_height.saturating_sub(1));
    let x1 = x1.min(image_width).max(x0 + 1);
    let y1 = y1.min(image_height).max(y0 + 1);
    ImageBoundingBox {
        x: x0,
        y: y0,
        width: x1 - x0,
        height: y1 - y0,
    }
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

pub fn locate_anything_queries(manifest: &SceneObjectManifest) -> Vec<String> {
    let categories = locate_anything_categories(manifest);
    if categories.is_empty() {
        Vec::new()
    } else {
        vec![categories.join("</c>")]
    }
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
    let mut used = vec![false; detections.len()];
    let mut used_instance_ids = object
        .instances
        .iter()
        .filter_map(|instance| instance.id.clone())
        .collect::<Vec<_>>();
    for (instance_id, bbox, contact, target_footprint_m) in instances {
        let detection_index = best_detection_match(&bbox, detections, &used);
        let (detection, provenance) = if let Some(index) = detection_index {
            used[index] = true;
            (
                Some(detections[index].clone()),
                vec![provenance_label.to_string()],
            )
        } else {
            (
                Some(Detection {
                    label: object.label.clone(),
                    bbox,
                    point: contact,
                    confidence: None,
                    source_query: object.label.clone(),
                }),
                vec!["manifest_fallback_missing_detection".to_string()],
            )
        };
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
    if detections.len() > used.iter().filter(|used| **used).count()
        && (object.instance_count > 1 || !object.instances.is_empty())
    {
        let mut extra_index = 0usize;
        for (detection_index, detection) in detections.iter().enumerate() {
            if used.get(detection_index).copied().unwrap_or(false) {
                continue;
            }
            extra_index += 1;
            let instance_id = generated_locate_instance_id(&mut used_instance_ids, extra_index);
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
                target_footprint_m: object.target_footprint_m,
                provenance: vec![format!("{provenance_label}_extra_instance")],
            });
        }
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
    object_label_keys(object).iter().any(|key| {
        detection_label == *key
            || detection_label.contains(key)
            || key.contains(&detection_label)
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
            let left_score = bbox_iou(object.bbox, left.bbox);
            let right_score = bbox_iou(object.bbox, right.bbox);
            left_score.total_cmp(&right_score).then_with(|| {
                bbox_area_normalized(left.bbox).total_cmp(&bbox_area_normalized(right.bbox))
            })
        })
        .cloned()
        .unwrap_or_else(|| {
            manifest_detection_for_object(object).expect("manifest detection exists")
        })
}

fn best_detection_match(bbox: &[f32; 4], detections: &[Detection], used: &[bool]) -> Option<usize> {
    detections
        .iter()
        .enumerate()
        .filter(|(index, _)| !used.get(*index).copied().unwrap_or(false))
        .map(|(index, detection)| (index, bbox_iou(*bbox, detection.bbox)))
        .max_by(|left, right| {
            left.1
                .partial_cmp(&right.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(index, _)| index)
}

fn normalized_query_key(value: &str) -> String {
    value.trim().to_ascii_lowercase()
}

pub fn bbox_bottom_center(bbox: [f32; 4]) -> [f32; 2] {
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

pub fn bbox_iou(left: [f32; 4], right: [f32; 4]) -> f32 {
    let ix0 = left[0].max(right[0]);
    let iy0 = left[1].max(right[1]);
    let ix1 = left[2].min(right[2]);
    let iy1 = left[3].min(right[3]);
    let iw = (ix1 - ix0).max(0.0);
    let ih = (iy1 - iy0).max(0.0);
    let intersection = iw * ih;
    let left_area = (left[2] - left[0]).max(0.0) * (left[3] - left[1]).max(0.0);
    let right_area = (right[2] - right[0]).max(0.0) * (right[3] - right[1]).max(0.0);
    let union = left_area + right_area - intersection;
    if union <= f32::EPSILON {
        0.0
    } else {
        intersection / union
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_synth_scene::{
        SceneInstanceSide, SceneObjectInstanceSpec, SceneObjectManifest, SceneObjectSpec,
    };

    #[test]
    fn segmentation_grounding_attaches_bbox_masks_and_artifacts() {
        let run_id = format!(
            "burn_synth_grounding_segmentation_test_{}_{}",
            std::process::id(),
            Instant::now().elapsed().as_nanos()
        );
        let root = std::env::temp_dir().join(run_id);
        fs::create_dir_all(&root).unwrap();
        let image_path = root.join("source.png");
        RgbaImage::from_pixel(20, 10, Rgba([24, 48, 72, 255]))
            .save(&image_path)
            .unwrap();
        let detection = Detection {
            label: "chair".to_string(),
            bbox: [0.10, 0.20, 0.40, 0.80],
            point: Some([0.25, 0.80]),
            confidence: Some(0.9),
            source_query: "chair".to_string(),
        };
        let mut evidence = SceneGroundingEvidence {
            source_image_path: image_path.display().to_string(),
            depth: None,
            segmentation: None,
            detections: vec![detection.clone()],
            camera: EstimatedCamera::default(),
            floor: EstimatedFloorPlane::default(),
            objects: vec![ObjectGroundingEvidence {
                object_id: "chair".to_string(),
                instance_id: Some("chair_01".to_string()),
                reuse_group: Some("chair".to_string()),
                detection: Some(detection),
                mask: None,
                asset_id: None,
                contact_pixel: Some([0.25, 0.80]),
                depth_stats: None,
                candidate_floor_contact_rays: Vec::new(),
                metric_contact_point_m: None,
                target_footprint_m: None,
                provenance: Vec::new(),
            }],
        };
        let mut runtime = SceneGroundingRuntime::default();
        let report = runtime
            .segmentation_grounding_evidence(
                &mut evidence,
                &image_path,
                &root,
                SegmentationGroundingConfig::default(),
            )
            .unwrap();

        assert_eq!(report.mask_count, 1);
        assert!(report.masks_path.exists());
        assert!(report.overlay_path.exists());
        assert_eq!(
            evidence
                .segmentation
                .as_ref()
                .and_then(|segmentation| segmentation.mask_count),
            Some(1)
        );
        let object_mask = evidence.objects[0].mask.as_ref().unwrap();
        assert_eq!(object_mask.image_size, [20, 10]);
        assert_eq!(object_mask.area_px, 6 * 6);
        assert!(
            object_mask
                .mask_png_path
                .as_ref()
                .is_some_and(|path| Path::new(path).exists())
        );

        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn locate_anything_cache_key_ignores_non_execution_flags() {
        let base = LocateAnythingRuntimeConfig {
            model_root: PathBuf::from("assets/models/LocateAnything-3B"),
            backend: LocateAnythingRuntimeBackend::BurnNative,
            allow_experimental_native_detect: true,
            decode_mode: DecodeMode::Hybrid,
            max_new_tokens: 1024,
            in_token_limit: LOCATE_ANYTHING_SAFE_IN_TOKEN_LIMIT,
            ..LocateAnythingRuntimeConfig::default()
        };
        let mut same_runtime = base.clone();
        same_runtime.require_gpu = false;
        assert_eq!(
            LocateAnythingBurnNativeCacheKey::from_config(&base),
            LocateAnythingBurnNativeCacheKey::from_config(&same_runtime)
        );

        let mut different_tokens = base;
        different_tokens.in_token_limit += 1;
        assert_ne!(
            LocateAnythingBurnNativeCacheKey::from_config(&same_runtime),
            LocateAnythingBurnNativeCacheKey::from_config(&different_tokens)
        );

        let mut different_decode_filter = same_runtime.clone();
        different_decode_filter.top_p = None;
        assert_ne!(
            LocateAnythingBurnNativeCacheKey::from_config(&same_runtime),
            LocateAnythingBurnNativeCacheKey::from_config(&different_decode_filter)
        );
    }

    #[test]
    fn depth_pro_cache_key_ignores_non_execution_policy_flags() {
        let base = DepthProGroundingConfig {
            cache_dir: Some(PathBuf::from("/tmp/depth-cache")),
            precision: GroundingDepthPrecision::F16,
            allow_download: true,
            require_gpu: true,
        };
        let mut same = base.clone();
        assert_eq!(
            DepthProRuntimeCacheKey::from_config(&base),
            DepthProRuntimeCacheKey::from_config(&same)
        );

        same.precision = GroundingDepthPrecision::F32;
        assert_ne!(
            DepthProRuntimeCacheKey::from_config(&base),
            DepthProRuntimeCacheKey::from_config(&same)
        );

        same = base.clone();
        same.allow_download = false;
        assert_eq!(
            DepthProRuntimeCacheKey::from_config(&base),
            DepthProRuntimeCacheKey::from_config(&same)
        );

        same = base.clone();
        same.require_gpu = false;
        assert_eq!(
            DepthProRuntimeCacheKey::from_config(&base),
            DepthProRuntimeCacheKey::from_config(&same)
        );

        same = base.clone();
        same.cache_dir = Some(PathBuf::from("/tmp/other-depth-cache"));
        assert_ne!(
            DepthProRuntimeCacheKey::from_config(&base),
            DepthProRuntimeCacheKey::from_config(&same)
        );
    }

    #[test]
    fn locate_anything_evidence_maps_detections_to_instances() {
        let manifest = SceneObjectManifest {
            source_scene_path: "/tmp/source.jpg".to_string(),
            scene_calibration: None,
            objects: vec![SceneObjectSpec {
                id: "chairs".to_string(),
                label: "chair".to_string(),
                aliases: Vec::new(),
                bbox: [0.10, 0.40, 0.80, 0.90],
                instances: vec![
                    SceneObjectInstanceSpec {
                        id: Some("chair_left".to_string()),
                        bbox: [0.10, 0.40, 0.30, 0.90],
                        contact: None,
                        rotation_hint_degrees: None,
                        facing_yaw_degrees: None,
                        side: Some(SceneInstanceSide::Left),
                        slot_index: None,
                        target_footprint_m: None,
                    },
                    SceneObjectInstanceSpec {
                        id: Some("chair_right".to_string()),
                        bbox: [0.60, 0.40, 0.80, 0.90],
                        contact: None,
                        rotation_hint_degrees: None,
                        facing_yaw_degrees: None,
                        side: Some(SceneInstanceSide::Right),
                        slot_index: None,
                        target_footprint_m: None,
                    },
                ],
                representative_instance_id: None,
                reuse_group: Some("chair".to_string()),
                instance_count: 2,
                object_prompt: "chair".to_string(),
                camera_hint: None,
                rotation_hint_degrees: None,
                target_footprint_m: None,
            }],
        };
        let detections = vec![
            Detection {
                label: "chair".to_string(),
                bbox: [0.61, 0.41, 0.79, 0.91],
                point: None,
                confidence: Some(0.8),
                source_query: "chair".to_string(),
            },
            Detection {
                label: "chair".to_string(),
                bbox: [0.11, 0.39, 0.29, 0.89],
                point: None,
                confidence: Some(0.9),
                source_query: "chair".to_string(),
            },
        ];
        let evidence = locate_anything_evidence_from_detections(
            &manifest,
            Path::new("/tmp/source.jpg"),
            detections,
            "locate_anything_test",
        )
        .unwrap();
        assert_eq!(evidence.detections.len(), 2);
        let left = evidence
            .objects
            .iter()
            .find(|object| object.instance_id.as_deref() == Some("chair_left"))
            .unwrap();
        let right = evidence
            .objects
            .iter()
            .find(|object| object.instance_id.as_deref() == Some("chair_right"))
            .unwrap();
        assert_eq!(
            left.detection.as_ref().unwrap().bbox,
            [0.11, 0.39, 0.29, 0.89]
        );
        assert_eq!(
            right.detection.as_ref().unwrap().bbox,
            [0.61, 0.41, 0.79, 0.91]
        );
        let object_union = evidence
            .objects
            .iter()
            .find(|object| object.object_id == "chairs" && object.instance_id.is_none())
            .unwrap();
        assert_eq!(
            object_union.detection.as_ref().unwrap().bbox,
            [0.11, 0.39, 0.79, 0.91]
        );
    }

    #[test]
    fn locate_anything_queries_use_hf_style_combined_categories() {
        let manifest = SceneObjectManifest {
            source_scene_path: "/tmp/source.jpg".to_string(),
            scene_calibration: None,
            objects: vec![
                SceneObjectSpec {
                    id: "table".to_string(),
                    label: "conference table".to_string(),
                    aliases: vec!["table".to_string()],
                    bbox: [0.30, 0.40, 0.70, 0.90],
                    instances: Vec::new(),
                    representative_instance_id: None,
                    reuse_group: Some("table".to_string()),
                    instance_count: 1,
                    object_prompt: "table".to_string(),
                    camera_hint: None,
                    rotation_hint_degrees: None,
                    target_footprint_m: None,
                },
                SceneObjectSpec {
                    id: "chair".to_string(),
                    label: "conference chair".to_string(),
                    aliases: vec!["chair".to_string()],
                    bbox: [0.10, 0.40, 0.80, 0.90],
                    instances: Vec::new(),
                    representative_instance_id: None,
                    reuse_group: Some("chair".to_string()),
                    instance_count: 1,
                    object_prompt: "chair".to_string(),
                    camera_hint: None,
                    rotation_hint_degrees: None,
                    target_footprint_m: None,
                },
            ],
        };
        assert_eq!(
            locate_anything_queries(&manifest),
            vec!["table</c>chair".to_string()]
        );
    }

    #[test]
    fn combined_locate_anything_labels_map_back_to_manifest_objects() {
        let manifest = SceneObjectManifest {
            source_scene_path: "/tmp/source.jpg".to_string(),
            scene_calibration: None,
            objects: vec![
                SceneObjectSpec {
                    id: "table".to_string(),
                    label: "conference table".to_string(),
                    aliases: vec!["table".to_string()],
                    bbox: [0.30, 0.40, 0.70, 0.90],
                    instances: Vec::new(),
                    representative_instance_id: None,
                    reuse_group: Some("table".to_string()),
                    instance_count: 1,
                    object_prompt: "table".to_string(),
                    camera_hint: None,
                    rotation_hint_degrees: None,
                    target_footprint_m: Some([3.2, 1.2]),
                },
                SceneObjectSpec {
                    id: "chair".to_string(),
                    label: "conference chair".to_string(),
                    aliases: vec!["chair".to_string()],
                    bbox: [0.10, 0.40, 0.80, 0.90],
                    instances: Vec::new(),
                    representative_instance_id: None,
                    reuse_group: Some("chair".to_string()),
                    instance_count: 1,
                    object_prompt: "chair".to_string(),
                    camera_hint: None,
                    rotation_hint_degrees: None,
                    target_footprint_m: Some([0.65, 0.65]),
                },
            ],
        };
        let combined = "conference table</c>conference chair".to_string();
        let evidence = locate_anything_evidence_from_detections(
            &manifest,
            Path::new("/tmp/source.jpg"),
            vec![
                Detection {
                    label: "conference chair".to_string(),
                    bbox: [0.11, 0.39, 0.29, 0.89],
                    point: None,
                    confidence: None,
                    source_query: combined.clone(),
                },
                Detection {
                    label: "conference table".to_string(),
                    bbox: [0.32, 0.42, 0.68, 0.88],
                    point: None,
                    confidence: None,
                    source_query: combined,
                },
            ],
            "locate_anything_test",
        )
        .unwrap();

        let table = evidence
            .objects
            .iter()
            .find(|object| object.object_id == "table")
            .unwrap();
        let chair = evidence
            .objects
            .iter()
            .find(|object| object.object_id == "chair")
            .unwrap();
        assert_eq!(
            table.detection.as_ref().unwrap().bbox,
            [0.32, 0.42, 0.68, 0.88]
        );
        assert_eq!(
            chair.detection.as_ref().unwrap().bbox,
            [0.11, 0.39, 0.29, 0.89]
        );
        assert_eq!(table.target_footprint_m, Some([3.2, 1.2]));
        assert_eq!(chair.target_footprint_m, Some([0.65, 0.65]));
    }

    #[test]
    fn locate_anything_evidence_does_not_duplicate_object_without_instances() {
        let manifest = SceneObjectManifest {
            source_scene_path: "/tmp/source.jpg".to_string(),
            scene_calibration: None,
            objects: vec![SceneObjectSpec {
                id: "table".to_string(),
                label: "conference table".to_string(),
                aliases: Vec::new(),
                bbox: [0.30, 0.40, 0.70, 0.90],
                instances: Vec::new(),
                representative_instance_id: None,
                reuse_group: Some("table".to_string()),
                instance_count: 1,
                object_prompt: "conference table".to_string(),
                camera_hint: None,
                rotation_hint_degrees: None,
                target_footprint_m: Some([3.2, 1.2]),
            }],
        };
        let evidence = locate_anything_evidence_from_detections(
            &manifest,
            Path::new("/tmp/source.jpg"),
            vec![Detection {
                label: "conference table".to_string(),
                bbox: [0.32, 0.42, 0.68, 0.88],
                point: None,
                confidence: Some(0.8),
                source_query: "conference table".to_string(),
            }],
            "locate_anything_test",
        )
        .unwrap();
        assert_eq!(evidence.objects.len(), 1);
        assert_eq!(evidence.objects[0].object_id, "table");
        assert!(evidence.objects[0].instance_id.is_none());
        assert_eq!(evidence.objects[0].target_footprint_m, Some([3.2, 1.2]));
    }

    #[test]
    fn singleton_object_uses_best_detection_instead_of_label_union() {
        let manifest = SceneObjectManifest {
            source_scene_path: "/tmp/source.jpg".to_string(),
            scene_calibration: None,
            objects: vec![SceneObjectSpec {
                id: "table".to_string(),
                label: "conference table".to_string(),
                aliases: vec!["table".to_string()],
                bbox: [0.30, 0.40, 0.70, 0.90],
                instances: Vec::new(),
                representative_instance_id: None,
                reuse_group: Some("table".to_string()),
                instance_count: 1,
                object_prompt: "conference table".to_string(),
                camera_hint: None,
                rotation_hint_degrees: None,
                target_footprint_m: Some([3.2, 1.2]),
            }],
        };
        let evidence = locate_anything_evidence_from_detections(
            &manifest,
            Path::new("/tmp/source.jpg"),
            vec![
                Detection {
                    label: "table".to_string(),
                    bbox: [0.386, 0.519, 0.659, 1.0],
                    point: None,
                    confidence: None,
                    source_query: "table</c>chair".to_string(),
                },
                Detection {
                    label: "table".to_string(),
                    bbox: [0.778, 0.401, 0.832, 0.481],
                    point: None,
                    confidence: None,
                    source_query: "table</c>chair".to_string(),
                },
            ],
            "locate_anything_test",
        )
        .unwrap();

        assert_eq!(evidence.objects.len(), 1);
        let table = evidence.objects.first().unwrap();
        assert_eq!(
            table.detection.as_ref().unwrap().bbox,
            [0.386, 0.519, 0.659, 1.0]
        );
    }

    #[test]
    fn depth_annotation_adds_contact_geometry_and_footprint_hints() {
        let detection = Detection {
            label: "conference chair".to_string(),
            bbox: [0.25, 0.25, 0.75, 0.75],
            point: Some([0.5, 0.75]),
            confidence: Some(0.9),
            source_query: "conference chair".to_string(),
        };
        let mut evidence = SceneGroundingEvidence {
            source_image_path: "/tmp/source.jpg".to_string(),
            depth: None,
            segmentation: None,
            detections: vec![detection.clone()],
            camera: EstimatedCamera::default(),
            floor: EstimatedFloorPlane::default(),
            objects: vec![ObjectGroundingEvidence {
                object_id: "chair".to_string(),
                instance_id: None,
                reuse_group: Some("chair".to_string()),
                detection: Some(detection),
                mask: None,
                asset_id: None,
                contact_pixel: None,
                depth_stats: None,
                candidate_floor_contact_rays: Vec::new(),
                metric_contact_point_m: None,
                target_footprint_m: Some([0.7, 0.8]),
                provenance: Vec::new(),
            }],
        };
        let depth_map = SceneDepthMapEvidence {
            depth_m: vec![
                2.0, 2.0, 2.0, 2.0, //
                2.0, 2.2, 2.2, 2.0, //
                2.0, 2.4, 2.4, 2.0, //
                3.0, 3.0, 3.0, 3.0,
            ],
            width: 4,
            height: 4,
            intrinsics: CameraIntrinsics {
                fx: 4.0,
                fy: 4.0,
                cx: 1.5,
                cy: 1.5,
                width: 4,
                height: 4,
            },
            focal_length_px: Some(4.0),
            vertical_fov_degrees: Some(53.0),
        };

        let summary =
            annotate_grounding_evidence_with_depth_map(&mut evidence, &depth_map, "depth_pro");
        let object = evidence.objects.first().unwrap();

        assert_eq!(summary.annotated_objects, 1);
        assert_eq!(summary.depth_map_size, [4, 4]);
        assert!(summary.floor_sample_count > 0);
        assert!(object.depth_stats.as_ref().unwrap().contact_m.unwrap() > 0.0);
        assert_eq!(object.candidate_floor_contact_rays.len(), 1);
        assert!(object.metric_contact_point_m.unwrap()[2] > 0.0);
        assert_eq!(object.target_footprint_m, Some([0.7, 0.8]));
        assert!(object.provenance.contains(&"depth_pro".to_string()));
    }

    #[test]
    fn far_field_filter_removes_small_background_detections() {
        let near_detection = Detection {
            label: "chair".to_string(),
            bbox: [0.10, 0.50, 0.30, 0.90],
            point: Some([0.20, 0.90]),
            confidence: None,
            source_query: "chair".to_string(),
        };
        let far_detection = Detection {
            label: "chair".to_string(),
            bbox: [0.80, 0.35, 0.84, 0.50],
            point: Some([0.82, 0.50]),
            confidence: None,
            source_query: "chair".to_string(),
        };
        let mut evidence = SceneGroundingEvidence {
            source_image_path: "/tmp/source.jpg".to_string(),
            depth: None,
            segmentation: None,
            detections: vec![near_detection.clone(), far_detection.clone()],
            camera: EstimatedCamera::default(),
            floor: EstimatedFloorPlane::default(),
            objects: vec![
                ObjectGroundingEvidence {
                    object_id: "near_chair".to_string(),
                    instance_id: None,
                    reuse_group: Some("chair".to_string()),
                    detection: Some(near_detection),
                    mask: None,
                    asset_id: None,
                    contact_pixel: Some([0.20, 0.90]),
                    depth_stats: Some(ObjectDepthStats {
                        median_m: 2.0,
                        min_m: 1.8,
                        max_m: 2.2,
                        contact_m: Some(2.0),
                        sample_count: Some(16),
                    }),
                    candidate_floor_contact_rays: Vec::new(),
                    metric_contact_point_m: Some([0.0, 0.0, 2.0]),
                    target_footprint_m: None,
                    provenance: vec!["depth_pro".to_string()],
                },
                ObjectGroundingEvidence {
                    object_id: "far_chair".to_string(),
                    instance_id: None,
                    reuse_group: Some("chair".to_string()),
                    detection: Some(far_detection),
                    mask: None,
                    asset_id: None,
                    contact_pixel: Some([0.82, 0.50]),
                    depth_stats: Some(ObjectDepthStats {
                        median_m: 12.0,
                        min_m: 10.0,
                        max_m: 12.5,
                        contact_m: Some(12.0),
                        sample_count: Some(16),
                    }),
                    candidate_floor_contact_rays: Vec::new(),
                    metric_contact_point_m: Some([2.0, 0.0, 12.0]),
                    target_footprint_m: None,
                    provenance: vec!["depth_pro".to_string()],
                },
            ],
        };
        let mut depth = vec![2.0; 100];
        for y in 3..5 {
            for x in 8..9 {
                depth[y * 10 + x] = 12.0;
            }
        }
        let depth_map = SceneDepthMapEvidence {
            depth_m: depth,
            width: 10,
            height: 10,
            intrinsics: CameraIntrinsics {
                fx: 10.0,
                fy: 10.0,
                cx: 4.5,
                cy: 4.5,
                width: 10,
                height: 10,
            },
            focal_length_px: Some(10.0),
            vertical_fov_degrees: Some(50.0),
        };

        let summary = filter_far_field_grounding_evidence(&mut evidence, &depth_map);
        assert!(summary.enabled);
        assert_eq!(summary.removed_detections, 1);
        assert_eq!(summary.removed_objects, 1);
        assert_eq!(evidence.detections.len(), 1);
        assert_eq!(evidence.objects.len(), 1);
        assert_eq!(evidence.objects[0].object_id, "near_chair");
    }
}

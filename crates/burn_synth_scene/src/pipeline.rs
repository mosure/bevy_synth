use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::thread;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::bsn::{
    default_run_id, representative_crop_bbox, stable_hash_hex, unix_ms, validate_build_config,
};
use crate::object_images::{
    decode_generated_object_rgb, generated_shape_consistency_score,
    generated_source_crop_edge_mismatch, image_dimensions_aspect, matte_generated_object_rgb,
    object_reconstruction_min_score, score_generated_object_rgb,
};
use crate::*;

pub trait SceneAiProvider {
    fn plan_objects(&self, request: &SceneReasoningRequest) -> SceneResult<SceneObjectManifest>;
    fn generate_object_images(&self, request: &ObjectImageRequest) -> SceneResult<Vec<Vec<u8>>>;
    fn plan_scene_bsn(&self, request: &SceneBsnRequest) -> SceneResult<String>;
    fn select_rotation_candidates(
        &self,
        _request: &SceneRotationSelectionRequest,
    ) -> SceneResult<SceneRotationSelectionResponse> {
        Err(SceneError::Provider(
            "rotation candidate selection is not supported by this provider".to_string(),
        ))
    }
    fn provider_metadata(&self) -> Value {
        Value::Null
    }
}

#[derive(Clone, Debug)]
pub struct SceneReasoningRequest {
    pub source_scene_path: PathBuf,
    pub object_reference_image_path: PathBuf,
    pub prompt: String,
}

#[derive(Clone, Debug)]
pub struct SceneBsnRequest {
    pub source_scene_path: PathBuf,
    pub object_manifest: SceneObjectManifest,
    pub asset_bindings: Vec<SceneAssetBinding>,
    pub prompt: String,
}

#[derive(Clone, Debug)]
pub struct SceneRotationSelectionRequest {
    pub prompt: String,
    pub task: Value,
    pub image_paths: Vec<PathBuf>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct SceneRotationSelectionResponse {
    pub objects: Vec<SceneRotationSelection>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct SceneRotationSelection {
    pub index: usize,
    pub candidate_index: usize,
    pub confidence: f32,
    pub rationale: String,
}

pub struct ScenePipeline<P> {
    config: SceneBuildConfig,
    provider: P,
    run_id: String,
}

struct ObjectImageRequestGenerationResult {
    attempts: Vec<ObjectImageAttemptReport>,
    candidates: Vec<ObjectImageCandidate>,
    accepted: bool,
}

fn normalize_object_image_generation_policy(
    policy: ObjectImageGenerationPolicy,
) -> ObjectImageGenerationPolicy {
    ObjectImageGenerationPolicy {
        min_score: policy.min_score.clamp(0.0, 1.0),
        max_attempts_per_object: policy.max_attempts_per_object.max(1),
        candidates_per_attempt: policy.candidates_per_attempt.max(1),
    }
}

impl<P: SceneAiProvider> ScenePipeline<P> {
    pub fn new(config: SceneBuildConfig, provider: P) -> Self {
        Self {
            run_id: default_run_id("scene_openai"),
            config,
            provider,
        }
    }

    pub fn prepare_openai_inputs(&mut self) -> SceneResult<ScenePreparation> {
        validate_build_config(&self.config)?;
        fs::create_dir_all(&self.config.output_dir)?;
        Ok(ScenePreparation {
            run_id: self.run_id.clone(),
            output_dir: self.config.output_dir.display().to_string(),
            source_scene_path: self.config.source_scene_path.display().to_string(),
            object_reference_image_path: self
                .config
                .object_reference_image_path
                .display()
                .to_string(),
            provider: "openai".to_string(),
            reasoning_model: self.config.reasoning_model.clone(),
            image_model: self.config.image_model.clone(),
            object_manifest_schema: object_manifest_schema(),
            scene_bsn_schema: scene_bsn_schema(),
            object_image_style_prompt: object_image_prompt_template(),
        })
    }

    pub fn plan_objects(&self) -> SceneResult<SceneObjectManifest> {
        validate_build_config(&self.config)?;
        self.provider.plan_objects(&SceneReasoningRequest {
            source_scene_path: self.config.source_scene_path.clone(),
            object_reference_image_path: self.config.object_reference_image_path.clone(),
            prompt: object_manifest_prompt(
                &self.config.source_scene_path,
                &self.config.object_reference_image_path,
                self.config.allow_catalog_reuse,
            ),
        })
    }

    pub fn prepare_object_image_requests(
        &self,
        manifest: &SceneObjectManifest,
    ) -> SceneResult<Vec<ObjectImageRequest>> {
        let crop_dir = self.config.output_dir.join("objects").join("crops");
        let api_input_dir = self.config.output_dir.join("objects").join("api_inputs");
        fs::create_dir_all(&crop_dir)?;
        fs::create_dir_all(&api_input_dir)?;
        let api_scene_path = resize_image_for_api(
            &self.config.source_scene_path,
            &api_input_dir.join("source_scene_1024.jpg"),
        )?;
        let api_reference_path = resize_image_for_api(
            &self.config.object_reference_image_path,
            &api_input_dir.join("object_reference_1024.jpg"),
        )?;
        let mut requests = Vec::new();
        for object in &manifest.objects {
            let mut request_object = object.clone();
            request_object.bbox = representative_crop_bbox(object);
            let crop_path = crop_scene_object(
                &self.config.source_scene_path,
                &request_object,
                &crop_dir.join(format!("{}_crop_1024.jpg", object.id)),
            )?;
            requests.push(ObjectImageRequest {
                object: request_object.clone(),
                source_scene_path: api_scene_path.display().to_string(),
                source_crop_path: crop_path.display().to_string(),
                object_reference_image_path: api_reference_path.display().to_string(),
                prompt: object_image_prompt(
                    &self.config.object_reference_image_path,
                    &request_object,
                ),
                candidate_count: self.config.candidate_count.max(1),
                size: "1024x1024".to_string(),
                quality: match self.config.quality_profile {
                    SceneQualityProfile::Draft => "medium",
                    SceneQualityProfile::Quality => "high",
                }
                .to_string(),
            });
        }
        Ok(requests)
    }

    pub fn generate_object_candidates(
        &self,
        requests: &[ObjectImageRequest],
    ) -> SceneResult<Vec<ObjectImageCandidate>> {
        let output_dir = self.config.output_dir.join("objects").join("generated");
        fs::create_dir_all(&output_dir)?;
        let mut candidates = Vec::new();
        for request in requests {
            let (mut generated, _) =
                self.generate_candidates_for_request(request, &output_dir, 0, 0)?;
            candidates.append(&mut generated);
        }
        Ok(candidates)
    }

    pub fn generate_object_candidates_with_policy(
        &self,
        requests: &[ObjectImageRequest],
        policy: ObjectImageGenerationPolicy,
    ) -> SceneResult<ObjectImageGenerationReport> {
        let output_dir = self.config.output_dir.join("objects").join("generated");
        fs::create_dir_all(&output_dir)?;
        let policy = normalize_object_image_generation_policy(policy);
        let mut attempts = Vec::new();
        let mut candidates = Vec::new();
        let mut processed_request_count = 0usize;
        for request in requests {
            let result =
                self.generate_candidates_for_request_with_policy(request, &output_dir, policy)?;
            attempts.extend(result.attempts);
            candidates.extend(result.candidates);
            processed_request_count += 1;
            if !result.accepted {
                write_metric(
                    &self.config.output_dir,
                    "openai.object_image.guardrail_abort",
                    json!({
                        "object_id": request.object.id,
                        "processed_objects": processed_request_count,
                        "total_objects": requests.len(),
                        "reason": "required object exhausted image candidate attempts",
                    }),
                )?;
                break;
            }
        }
        let processed_requests = &requests[..processed_request_count.min(requests.len())];
        let rejected_objects =
            object_image_candidate_rejections(processed_requests, &candidates, policy.min_score);
        let selected_candidates = if rejected_objects.is_empty() {
            let manifest = SceneObjectManifest {
                source_scene_path: self.config.source_scene_path.display().to_string(),
                scene_calibration: None,
                objects: processed_requests
                    .iter()
                    .map(|request| request.object.clone())
                    .collect(),
            };
            select_object_image_candidates(&manifest, &candidates, policy.min_score)?
        } else {
            Vec::new()
        };
        Ok(ObjectImageGenerationReport {
            policy,
            attempts,
            candidates,
            selected_candidates,
            rejected_objects,
        })
    }

    pub fn generate_object_candidates_with_policy_parallel(
        &self,
        requests: &[ObjectImageRequest],
        policy: ObjectImageGenerationPolicy,
        max_parallel_requests: usize,
    ) -> SceneResult<ObjectImageGenerationReport>
    where
        P: Sync,
    {
        let output_dir = self.config.output_dir.join("objects").join("generated");
        fs::create_dir_all(&output_dir)?;
        let policy = normalize_object_image_generation_policy(policy);
        let max_parallel_requests = max_parallel_requests.max(1);
        if max_parallel_requests == 1 || requests.len() <= 1 {
            return self.generate_object_candidates_with_policy(requests, policy);
        }

        let mut per_request_results = Vec::new();
        for (chunk_index, chunk) in requests.chunks(max_parallel_requests).enumerate() {
            let chunk_start = chunk_index * max_parallel_requests;
            let mut chunk_results = thread::scope(|scope| -> SceneResult<_> {
                let mut handles = Vec::new();
                for (offset, request) in chunk.iter().enumerate() {
                    let request_index = chunk_start + offset;
                    let output_dir = &output_dir;
                    handles.push(scope.spawn(move || {
                        self.generate_candidates_for_request_with_policy(
                            request, output_dir, policy,
                        )
                        .map(|result| (request_index, result))
                    }));
                }

                let mut results = Vec::with_capacity(handles.len());
                for handle in handles {
                    results.push(handle.join().map_err(|_| {
                        SceneError::Provider("object image generation worker panicked".to_string())
                    })??);
                }
                Ok(results)
            })?;
            chunk_results.sort_by_key(|(request_index, _)| *request_index);
            let rejected = chunk_results.iter().any(|(_, result)| !result.accepted);
            per_request_results.extend(chunk_results);
            if rejected {
                break;
            }
        }

        per_request_results.sort_by_key(|(request_index, _)| *request_index);
        let processed_request_count = per_request_results.len();
        let mut attempts = Vec::new();
        let mut candidates = Vec::new();
        let mut first_rejected_object = None;
        for (request_index, result) in per_request_results {
            if !result.accepted && first_rejected_object.is_none() {
                first_rejected_object = Some(request_index);
            }
            attempts.extend(result.attempts);
            candidates.extend(result.candidates);
        }
        if let Some(request_index) = first_rejected_object {
            let request = &requests[request_index];
            write_metric(
                &self.config.output_dir,
                "openai.object_image.guardrail_abort",
                json!({
                    "object_id": request.object.id,
                    "processed_objects": processed_request_count,
                    "total_objects": requests.len(),
                    "reason": "required object exhausted image candidate attempts",
                    "parallel_requests": max_parallel_requests,
                }),
            )?;
        }

        let processed_requests = &requests[..processed_request_count.min(requests.len())];
        let rejected_objects =
            object_image_candidate_rejections(processed_requests, &candidates, policy.min_score);
        let selected_candidates = if rejected_objects.is_empty() {
            let manifest = SceneObjectManifest {
                source_scene_path: self.config.source_scene_path.display().to_string(),
                scene_calibration: None,
                objects: processed_requests
                    .iter()
                    .map(|request| request.object.clone())
                    .collect(),
            };
            select_object_image_candidates(&manifest, &candidates, policy.min_score)?
        } else {
            Vec::new()
        };
        Ok(ObjectImageGenerationReport {
            policy,
            attempts,
            candidates,
            selected_candidates,
            rejected_objects,
        })
    }

    fn generate_candidates_for_request_with_policy(
        &self,
        request: &ObjectImageRequest,
        output_dir: &Path,
        policy: ObjectImageGenerationPolicy,
    ) -> SceneResult<ObjectImageRequestGenerationResult> {
        let mut attempts = Vec::new();
        let mut candidates = Vec::new();
        let mut accepted = false;
        let mut next_candidate_index = 0usize;
        let min_score = object_reconstruction_min_score(&request.object, policy.min_score);
        for attempt_index in 0..policy.max_attempts_per_object {
            let mut attempt_request = request.clone();
            attempt_request.candidate_count = policy.candidates_per_attempt;
            let (mut generated, elapsed_ms) = self.generate_candidates_for_request(
                &attempt_request,
                output_dir,
                next_candidate_index,
                attempt_index,
            )?;
            next_candidate_index += generated.len();
            candidates.append(&mut generated);
            let best_score = best_candidate_for_object(&candidates, &request.object.id)
                .map(|candidate| candidate.score);
            accepted = best_score.is_some_and(|score| score >= min_score);
            attempts.push(ObjectImageAttemptReport {
                object_id: request.object.id.clone(),
                attempt_index,
                requested_candidates: attempt_request.candidate_count,
                generated_candidates: next_candidate_index,
                best_score_after_attempt: best_score,
                accepted,
                elapsed_ms,
            });
            write_metric(
                &self.config.output_dir,
                "openai.object_image.guardrail_attempt",
                json!({
                    "object_id": request.object.id,
                    "attempt_index": attempt_index,
                    "requested_candidates": attempt_request.candidate_count,
                    "generated_candidates": next_candidate_index,
                    "best_score_after_attempt": best_score,
                    "base_min_score": policy.min_score,
                    "min_score": min_score,
                    "accepted": accepted,
                }),
            )?;
            if accepted {
                break;
            }
        }
        if !accepted {
            write_metric(
                &self.config.output_dir,
                "openai.object_image.guardrail_rejected",
                json!({
                    "object_id": request.object.id,
                    "best_score": best_candidate_for_object(&candidates, &request.object.id).map(|candidate| candidate.score),
                    "base_min_score": policy.min_score,
                    "min_score": min_score,
                }),
            )?;
        }
        Ok(ObjectImageRequestGenerationResult {
            attempts,
            candidates,
            accepted,
        })
    }

    fn generate_candidates_for_request(
        &self,
        request: &ObjectImageRequest,
        output_dir: &Path,
        candidate_index_offset: usize,
        attempt_index: usize,
    ) -> SceneResult<(Vec<ObjectImageCandidate>, u128)> {
        let started = unix_ms();
        write_metric(
            &self.config.output_dir,
            "openai.object_image.start",
            json!({
                "object_id": request.object.id,
                "attempt_index": attempt_index,
                "candidate_count": request.candidate_count,
                "quality": request.quality,
                "size": request.size,
                "source_scene_path": request.source_scene_path,
                "source_crop_path": request.source_crop_path,
                "object_reference_image_path": request.object_reference_image_path,
            }),
        )?;
        eprintln!(
            "burn_synth_scene: generating {} object image candidate(s) for {} (attempt {})",
            request.candidate_count,
            request.object.id,
            attempt_index + 1
        );
        let images = self.provider.generate_object_images(request)?;
        let elapsed_ms = unix_ms().saturating_sub(started);
        let source_image_aspect = image_dimensions_aspect(Path::new(&request.source_scene_path))?;
        write_metric(
            &self.config.output_dir,
            "openai.object_image",
            json!({
                "object_id": request.object.id,
                "attempt_index": attempt_index,
                "candidate_count": images.len(),
                "elapsed_ms": elapsed_ms,
                "quality": request.quality,
                "size": request.size,
            }),
        )?;
        let mut candidates = Vec::with_capacity(images.len());
        for (provider_candidate_index, bytes) in images.into_iter().enumerate() {
            let candidate_index = candidate_index_offset + provider_candidate_index;
            let image_path = output_dir.join(format!(
                "{}_candidate_{}.png",
                request.object.id, candidate_index
            ));
            let raw_image_path = output_dir.join(format!(
                "{}_candidate_{}_raw.png",
                request.object.id, candidate_index
            ));
            let rgb = decode_generated_object_rgb(&bytes)?;
            let suitability = score_generated_object_rgb(&rgb);
            let (matted, matte_stats) = matte_generated_object_rgb(&rgb, suitability);
            let shape_score = generated_shape_consistency_score(
                &request.object,
                &matte_stats,
                source_image_aspect,
            );
            let source_crop_edge_mismatch =
                generated_source_crop_edge_mismatch(&request.object, &matte_stats);
            let score = (suitability.score * shape_score).clamp(0.0, 1.0);
            fs::write(&raw_image_path, &bytes)?;
            matted.save(&image_path)?;
            write_metric(
                &self.config.output_dir,
                "openai.object_image.candidate",
                json!({
                    "object_id": request.object.id,
                    "attempt_index": attempt_index,
                    "provider_candidate_index": provider_candidate_index,
                    "candidate_index": candidate_index,
                    "image_path": image_path.display().to_string(),
                    "raw_image_path": raw_image_path.display().to_string(),
                    "score": suitability.score,
                    "background_rgb": suitability.background_rgb,
                    "contrast_ratio_gt15": suitability.contrast_ratio_gt15,
                    "contrast_ratio_gt25": suitability.contrast_ratio_gt25,
                    "contrast_ratio_gt40": suitability.contrast_ratio_gt40,
                    "alpha_coverage": matte_stats.alpha_coverage,
                    "alpha_bbox": matte_stats.alpha_bbox,
                    "shape_score": shape_score,
                    "source_crop_edge_mismatch": source_crop_edge_mismatch,
                    "score_after_shape": score,
                }),
            )?;
            candidates.push(ObjectImageCandidate {
                object_id: request.object.id.clone(),
                candidate_index,
                image_path: image_path.display().to_string(),
                raw_image_path: Some(raw_image_path.display().to_string()),
                prompt_hash: stable_hash_hex(&request.prompt),
                score,
                provider_request_id: None,
            });
        }
        Ok((candidates, elapsed_ms))
    }

    pub fn plan_scene_bsn(
        &self,
        manifest: SceneObjectManifest,
        asset_bindings: Vec<SceneAssetBinding>,
    ) -> SceneResult<String> {
        grounded_scene_bsn(&manifest, &asset_bindings)
    }

    pub fn provider_metadata(&self) -> Value {
        self.provider.provider_metadata()
    }
}

pub fn select_object_image_candidates(
    manifest: &SceneObjectManifest,
    candidates: &[ObjectImageCandidate],
    min_score: f32,
) -> SceneResult<Vec<SelectedObjectImageCandidate>> {
    select_object_image_candidates_with_exclusions(manifest, candidates, min_score, &HashSet::new())
}

pub fn select_object_image_candidates_with_exclusions(
    manifest: &SceneObjectManifest,
    candidates: &[ObjectImageCandidate],
    min_score: f32,
    excluded_candidates: &HashSet<(String, usize)>,
) -> SceneResult<Vec<SelectedObjectImageCandidate>> {
    let mut by_object = candidates
        .iter()
        .filter(|candidate| {
            !excluded_candidates.contains(&(candidate.object_id.clone(), candidate.candidate_index))
        })
        .map(|candidate| (candidate.object_id.as_str(), candidate))
        .collect::<Vec<_>>();
    by_object.sort_by(|(_, left), (_, right)| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.candidate_index.cmp(&right.candidate_index))
    });
    let mut selected = Vec::new();
    let mut seen_groups = HashSet::new();
    for object in &manifest.objects {
        let group = object
            .reuse_group
            .as_deref()
            .filter(|value| !value.is_empty())
            .unwrap_or(object.id.as_str())
            .to_string();
        if !seen_groups.insert(group.clone()) {
            continue;
        }
        let candidate = by_object
            .iter()
            .find(|(object_id, _)| *object_id == object.id)
            .map(|(_, candidate)| *candidate)
            .ok_or_else(|| {
                SceneError::Validation(format!(
                    "no generated image candidate for object `{}`",
                    object.id
                ))
            })?;
        let object_min_score = object_reconstruction_min_score(object, min_score);
        if candidate.score < object_min_score {
            return Err(SceneError::Validation(candidate_rejection_message(
                &object.id,
                Some(candidate.score),
                object_min_score,
            )));
        }
        selected.push(SelectedObjectImageCandidate {
            object_id: object.id.clone(),
            reuse_group: group,
            label: object.label.clone(),
            image_path: candidate.image_path.clone(),
            candidate_index: candidate.candidate_index,
            score: candidate.score,
            prompt_hash: candidate.prompt_hash.clone(),
        });
    }
    Ok(selected)
}

pub fn object_image_candidate_rejections(
    requests: &[ObjectImageRequest],
    candidates: &[ObjectImageCandidate],
    min_score: f32,
) -> Vec<RejectedObjectImageCandidates> {
    requests
        .iter()
        .filter_map(|request| {
            let best_score = best_candidate_for_object(candidates, &request.object.id)
                .map(|candidate| candidate.score);
            let object_min_score = object_reconstruction_min_score(&request.object, min_score);
            if best_score.is_some_and(|score| score >= object_min_score) {
                None
            } else {
                Some(RejectedObjectImageCandidates {
                    object_id: request.object.id.clone(),
                    best_score,
                    min_score: object_min_score,
                    message: candidate_rejection_message(
                        &request.object.id,
                        best_score,
                        object_min_score,
                    ),
                })
            }
        })
        .collect()
}

fn best_candidate_for_object<'a>(
    candidates: &'a [ObjectImageCandidate],
    object_id: &str,
) -> Option<&'a ObjectImageCandidate> {
    candidates
        .iter()
        .filter(|candidate| candidate.object_id == object_id)
        .max_by(|left, right| {
            left.score
                .partial_cmp(&right.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| right.candidate_index.cmp(&left.candidate_index))
        })
}

fn candidate_rejection_message(object_id: &str, best_score: Option<f32>, min_score: f32) -> String {
    match best_score {
        Some(score) => format!(
            "generated image candidate for object `{object_id}` is not suitable for TRELLIS/RMBG reconstruction (score={score:.3}, min={min_score:.3}); regenerate with more candidates or improve the isolated-object prompt/background"
        ),
        None => format!(
            "no generated image candidate for object `{object_id}`; regenerate the object image stage"
        ),
    }
}

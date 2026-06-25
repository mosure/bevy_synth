use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use clap::{ArgAction, Parser, Subcommand, ValueEnum};
use image::GenericImageView;
use image::codecs::jpeg::JpegEncoder;
use image::imageops::FilterType;
use reqwest::blocking::multipart::{Form, Part};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

pub type SceneResult<T> = Result<T, SceneError>;

#[derive(Debug)]
pub enum SceneError {
    Config(String),
    Io(String),
    Image(String),
    Http(String),
    Provider(String),
    Parse(String),
    Validation(String),
}

impl std::fmt::Display for SceneError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Config(err) => write!(f, "configuration error: {err}"),
            Self::Io(err) => write!(f, "I/O error: {err}"),
            Self::Image(err) => write!(f, "image error: {err}"),
            Self::Http(err) => write!(f, "OpenAI HTTP error: {err}"),
            Self::Provider(err) => write!(f, "provider error: {err}"),
            Self::Parse(err) => write!(f, "parse error: {err}"),
            Self::Validation(err) => write!(f, "validation error: {err}"),
        }
    }
}

impl std::error::Error for SceneError {}

impl From<std::io::Error> for SceneError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value.to_string())
    }
}

impl From<image::ImageError> for SceneError {
    fn from(value: image::ImageError) -> Self {
        Self::Image(value.to_string())
    }
}

#[derive(Parser, Debug, Clone)]
#[command(
    name = "burn_synth_scene",
    version,
    about = "Scene-image to object-asset composition pipeline"
)]
pub struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug, Clone)]
enum Command {
    /// Plan and generate object images through the OpenAI provider.
    Build(BuildArgs),
    /// Validate a restricted BSN scene file against asset bindings.
    ValidateBsn(ValidateBsnArgs),
    /// Write a Bevy/MCP scene command envelope from a restricted BSN scene.
    WriteCommands(WriteCommandsArgs),
}

#[derive(Parser, Debug, Clone)]
struct BuildArgs {
    #[arg(long)]
    scene: PathBuf,
    #[arg(long, default_value = "docs/input_chair.jpg")]
    object_reference: PathBuf,
    #[arg(long)]
    output_dir: Option<PathBuf>,
    #[arg(long, default_value_t = 3)]
    candidates: usize,
    #[arg(long, value_enum, default_value_t = SceneQualityProfile::Quality)]
    profile: SceneQualityProfile,
    #[arg(long, default_value = "gpt-5.5")]
    reasoning_model: String,
    #[arg(long, default_value = "gpt-image-2")]
    image_model: String,
}

#[derive(Parser, Debug, Clone)]
struct ValidateBsnArgs {
    #[arg(long)]
    bsn: PathBuf,
    #[arg(long)]
    assets_json: PathBuf,
}

#[derive(Parser, Debug, Clone)]
struct WriteCommandsArgs {
    #[arg(long)]
    bsn: PathBuf,
    #[arg(long)]
    assets_json: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    clear_existing: bool,
    #[arg(long)]
    session_id: Option<String>,
    #[arg(long)]
    sequence: Option<u64>,
}

pub fn run_cli(cli: Cli) -> SceneResult<()> {
    match cli.command {
        Command::Build(args) => {
            let output_dir = args.output_dir.unwrap_or_else(|| {
                PathBuf::from("tmp/runs").join(default_run_id("scene_openai_build"))
            });
            let config = SceneBuildConfig {
                source_scene_path: args.scene,
                object_reference_image_path: args.object_reference,
                output_dir,
                candidate_count: args.candidates,
                quality_profile: args.profile,
                reasoning_model: args.reasoning_model,
                image_model: args.image_model,
                allow_catalog_reuse: false,
            };
            let provider = OpenAiSceneProvider::from_env(OpenAiProviderConfig {
                reasoning_model: config.reasoning_model.clone(),
                image_model: config.image_model.clone(),
                ..OpenAiProviderConfig::default()
            })?;
            let mut pipeline = ScenePipeline::new(config, provider);
            let preparation = pipeline.prepare_openai_inputs()?;
            let output_dir = PathBuf::from(&preparation.output_dir);
            write_json_file(&output_dir.join("preparation.json"), &preparation)?;
            let manifest = pipeline.plan_objects()?;
            write_json_file(&output_dir.join("manifest.json"), &manifest)?;
            let requests = pipeline.prepare_object_image_requests(&manifest)?;
            write_json_file(&output_dir.join("object_image_requests.json"), &requests)?;
            let candidates = pipeline.generate_object_candidates(&requests)?;
            write_json_file(&output_dir.join("object_candidates.json"), &candidates)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "run_id": preparation.run_id,
                    "output_dir": preparation.output_dir,
                    "objects": manifest.objects.len(),
                    "candidates": candidates.len(),
                    "next_stage": "Use burn_synth_mcp images_to_assets with selected object candidate images, then scene_apply_bsn."
                }))
                .unwrap()
            );
            Ok(())
        }
        Command::ValidateBsn(args) => {
            let bsn = fs::read_to_string(args.bsn)?;
            let assets = load_scene_asset_bindings(&args.assets_json)?;
            let parsed = parse_scene_bsn(&bsn, &assets)?;
            println!("{}", serde_json::to_string_pretty(&parsed).unwrap());
            Ok(())
        }
        Command::WriteCommands(args) => {
            let envelope = scene_bsn_file_to_mcp_command_envelope(
                &args.bsn,
                &args.assets_json,
                args.clear_existing,
                args.session_id.as_deref(),
                args.sequence,
            )?;
            write_json_file(&args.output, &envelope)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "output": args.output,
                    "commands": envelope
                        .get("commands")
                        .and_then(Value::as_array)
                        .map(Vec::len)
                        .unwrap_or_default(),
                    "session_id": envelope.get("session_id"),
                    "sequence": envelope.get("sequence"),
                }))
                .unwrap()
            );
            Ok(())
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum SceneQualityProfile {
    Draft,
    Quality,
}

pub const DEFAULT_SCENE_RECONSTRUCTION_IMAGE_SCORE: f32 = 0.45;

#[derive(Clone, Debug)]
pub struct SceneBuildConfig {
    pub source_scene_path: PathBuf,
    pub object_reference_image_path: PathBuf,
    pub output_dir: PathBuf,
    pub candidate_count: usize,
    pub quality_profile: SceneQualityProfile,
    pub reasoning_model: String,
    pub image_model: String,
    pub allow_catalog_reuse: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct SceneObjectManifest {
    pub source_scene_path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scene_calibration: Option<SceneCalibration>,
    pub objects: Vec<SceneObjectSpec>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
pub struct SceneCalibration {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub table_center: Option<[f32; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub table_axis_degrees: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub table_size_m: Option<[f32; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub camera_yaw_degrees: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub camera_pitch_degrees: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub camera_radius_m: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vertical_fov_degrees: Option<f32>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct SceneObjectSpec {
    pub id: String,
    pub label: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    pub bbox: [f32; 4],
    #[serde(default)]
    pub instances: Vec<SceneObjectInstanceSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub representative_instance_id: Option<String>,
    #[serde(default)]
    pub reuse_group: Option<String>,
    #[serde(default = "default_instance_count")]
    pub instance_count: usize,
    pub object_prompt: String,
    #[serde(default)]
    pub camera_hint: Option<String>,
    #[serde(default)]
    pub rotation_hint_degrees: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_footprint_m: Option<[f32; 2]>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct SceneObjectInstanceSpec {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub bbox: [f32; 4],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contact: Option<[f32; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rotation_hint_degrees: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub facing_yaw_degrees: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub side: Option<SceneInstanceSide>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slot_index: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_footprint_m: Option<[f32; 2]>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum SceneInstanceSide {
    Left,
    Right,
    Near,
    Far,
    Head,
    Foot,
    Unknown,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ObjectImageRequest {
    pub object: SceneObjectSpec,
    pub source_scene_path: String,
    pub source_crop_path: String,
    pub object_reference_image_path: String,
    pub prompt: String,
    pub candidate_count: usize,
    pub size: String,
    pub quality: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ObjectImageCandidate {
    pub object_id: String,
    pub candidate_index: usize,
    pub image_path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_image_path: Option<String>,
    pub prompt_hash: String,
    pub score: f32,
    #[serde(default)]
    pub provider_request_id: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
pub struct ObjectImageGenerationPolicy {
    pub min_score: f32,
    pub max_attempts_per_object: usize,
    pub candidates_per_attempt: usize,
}

impl ObjectImageGenerationPolicy {
    pub fn from_total_candidate_budget(candidate_count: usize) -> Self {
        Self {
            min_score: DEFAULT_SCENE_RECONSTRUCTION_IMAGE_SCORE,
            max_attempts_per_object: candidate_count.max(1),
            candidates_per_attempt: 1,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ObjectImageAttemptReport {
    pub object_id: String,
    pub attempt_index: usize,
    pub requested_candidates: usize,
    pub generated_candidates: usize,
    pub best_score_after_attempt: Option<f32>,
    pub accepted: bool,
    pub elapsed_ms: u128,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct SelectedObjectImageCandidate {
    pub object_id: String,
    pub reuse_group: String,
    pub label: String,
    pub image_path: String,
    pub candidate_index: usize,
    pub score: f32,
    pub prompt_hash: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct RejectedObjectImageCandidates {
    pub object_id: String,
    pub best_score: Option<f32>,
    pub min_score: f32,
    pub message: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ObjectImageGenerationReport {
    pub policy: ObjectImageGenerationPolicy,
    pub attempts: Vec<ObjectImageAttemptReport>,
    pub candidates: Vec<ObjectImageCandidate>,
    pub selected_candidates: Vec<SelectedObjectImageCandidate>,
    pub rejected_objects: Vec<RejectedObjectImageCandidates>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct SceneAssetBinding {
    pub asset_id: String,
    pub object_id: String,
    pub label: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub cache_key: Option<String>,
    #[serde(default)]
    pub reusable: bool,
    #[serde(default)]
    pub source_image_path: Option<String>,
    #[serde(default)]
    pub pipeline: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub local_aabb: Option<SceneAssetAabb>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub canonical_frame: Option<SceneAssetFrame>,
    #[serde(default)]
    pub provenance: Option<SceneAssetProvenance>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
pub struct SceneAssetFrame {
    pub yaw_offset_degrees: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub footprint_m: Option<[f32; 2]>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
pub struct SceneAssetAabb {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl SceneAssetAabb {
    pub fn size(&self) -> [f32; 3] {
        [
            (self.max[0] - self.min[0]).max(1.0e-5),
            (self.max[1] - self.min[1]).max(1.0e-5),
            (self.max[2] - self.min[2]).max(1.0e-5),
        ]
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct SceneAssetProvenance {
    pub run_id: String,
    pub source_scene_path: String,
    pub source_object_id: String,
    pub generated_by: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ScenePlan {
    pub bsn: String,
    pub placements: Vec<ScenePlacement>,
    #[serde(default)]
    pub camera: Option<SceneCamera>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ScenePlacement {
    pub entity_id: String,
    pub asset_id: String,
    pub translation: [f32; 3],
    pub rotation_y_degrees: f32,
    pub scale: [f32; 3],
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct SceneCamera {
    pub translation: [f32; 3],
    pub focus: [f32; 3],
    #[serde(default)]
    pub yaw: Option<f32>,
    #[serde(default)]
    pub pitch: Option<f32>,
    #[serde(default)]
    pub radius: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vertical_fov_degrees: Option<f32>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ScenePreparation {
    pub run_id: String,
    pub output_dir: String,
    pub source_scene_path: String,
    pub object_reference_image_path: String,
    pub provider: String,
    pub reasoning_model: String,
    pub image_model: String,
    pub object_manifest_schema: Value,
    pub scene_bsn_schema: Value,
    pub object_image_style_prompt: String,
}

pub trait SceneAiProvider {
    fn plan_objects(&self, request: &SceneReasoningRequest) -> SceneResult<SceneObjectManifest>;
    fn generate_object_images(&self, request: &ObjectImageRequest) -> SceneResult<Vec<Vec<u8>>>;
    fn plan_scene_bsn(&self, request: &SceneBsnRequest) -> SceneResult<String>;
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

pub struct ScenePipeline<P> {
    config: SceneBuildConfig,
    provider: P,
    run_id: String,
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
        let policy = ObjectImageGenerationPolicy {
            min_score: policy.min_score.clamp(0.0, 1.0),
            max_attempts_per_object: policy.max_attempts_per_object.max(1),
            candidates_per_attempt: policy.candidates_per_attempt.max(1),
        };
        let mut attempts = Vec::new();
        let mut candidates = Vec::new();
        for request in requests {
            let mut accepted = false;
            let mut next_candidate_index = 0usize;
            for attempt_index in 0..policy.max_attempts_per_object {
                let mut attempt_request = request.clone();
                attempt_request.candidate_count = policy.candidates_per_attempt;
                let (mut generated, elapsed_ms) = self.generate_candidates_for_request(
                    &attempt_request,
                    &output_dir,
                    next_candidate_index,
                    attempt_index,
                )?;
                next_candidate_index += generated.len();
                candidates.append(&mut generated);
                let best_score = best_candidate_for_object(&candidates, &request.object.id)
                    .map(|candidate| candidate.score);
                accepted = best_score.is_some_and(|score| score >= policy.min_score);
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
                        "min_score": policy.min_score,
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
                        "min_score": policy.min_score,
                    }),
                )?;
            }
        }
        let rejected_objects =
            object_image_candidate_rejections(requests, &candidates, policy.min_score);
        let selected_candidates = if rejected_objects.is_empty() {
            let manifest = SceneObjectManifest {
                source_scene_path: self.config.source_scene_path.display().to_string(),
                scene_calibration: None,
                objects: requests
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
    let mut by_object = candidates
        .iter()
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
        if candidate.score < min_score {
            return Err(SceneError::Validation(candidate_rejection_message(
                &object.id,
                Some(candidate.score),
                min_score,
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
            if best_score.is_some_and(|score| score >= min_score) {
                None
            } else {
                Some(RejectedObjectImageCandidates {
                    object_id: request.object.id.clone(),
                    best_score,
                    min_score,
                    message: candidate_rejection_message(&request.object.id, best_score, min_score),
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

#[derive(Clone, Debug)]
pub struct OpenAiProviderConfig {
    pub api_key: String,
    pub base_url: String,
    pub project_id: Option<String>,
    pub reasoning_model: String,
    pub image_model: String,
    pub timeout: Duration,
}

impl Default for OpenAiProviderConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: "https://api.openai.com".to_string(),
            project_id: None,
            reasoning_model: "gpt-5.5".to_string(),
            image_model: "gpt-image-2".to_string(),
            timeout: Duration::from_secs(180),
        }
    }
}

pub struct OpenAiSceneProvider {
    config: OpenAiProviderConfig,
    client: reqwest::blocking::Client,
    request_log: RefCell<Vec<Value>>,
}

impl OpenAiSceneProvider {
    pub fn from_env(mut config: OpenAiProviderConfig) -> SceneResult<Self> {
        config.api_key = env::var("OPENAI_API_KEY")
            .ok()
            .or_else(|| (!config.api_key.is_empty()).then(|| config.api_key.clone()))
            .ok_or_else(|| {
                SceneError::Config(
                    "OPENAI_API_KEY is required for live OpenAI scene generation".to_string(),
                )
            })?;
        if let Ok(base_url) = env::var("OPENAI_BASE_URL") {
            config.base_url = base_url;
        }
        config.project_id = env::var("OPENAI_PROJECT_ID")
            .ok()
            .or(config.project_id.take());
        let client = reqwest::blocking::Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|err| SceneError::Http(err.to_string()))?;
        Ok(Self {
            config,
            client,
            request_log: RefCell::new(Vec::new()),
        })
    }

    fn auth(
        &self,
        request: reqwest::blocking::RequestBuilder,
    ) -> reqwest::blocking::RequestBuilder {
        let request = request.bearer_auth(&self.config.api_key);
        if let Some(project_id) = self.config.project_id.as_ref() {
            request.header("OpenAI-Project", project_id)
        } else {
            request
        }
    }

    fn responses_schema_request(
        &self,
        prompt: &str,
        schema: Value,
        image_paths: &[PathBuf],
    ) -> SceneResult<Value> {
        let mut content = vec![json!({ "type": "input_text", "text": prompt })];
        for image_path in image_paths {
            content.push(json!({
                "type": "input_image",
                "image_url": image_data_url(image_path)?,
            }));
        }
        Ok(json!({
            "model": self.config.reasoning_model,
            "input": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "burn_synth_scene",
                    "strict": true,
                    "schema": schema
                }
            }
        }))
    }

    fn post_responses_schema(
        &self,
        operation: &'static str,
        prompt: &str,
        schema: Value,
        image_paths: &[PathBuf],
    ) -> SceneResult<Value> {
        let url = format!(
            "{}/v1/responses",
            self.config.base_url.trim_end_matches('/')
        );
        let response = self
            .auth(self.client.post(url).json(&self.responses_schema_request(
                prompt,
                schema,
                image_paths,
            )?))
            .send()
            .map_err(|err| SceneError::Http(err.to_string()))?;
        let status = response.status();
        let value: Value = response
            .json()
            .map_err(|err| SceneError::Http(format!("decode response body: {err}")))?;
        if !status.is_success() {
            return Err(SceneError::Http(format!(
                "status {status}: {}",
                redact_openai_value(&value)
            )));
        }
        let response_model = value.get("model").and_then(Value::as_str);
        self.warn_model_mismatch(operation, &self.config.reasoning_model, response_model);
        self.record_request(json!({
            "operation": operation,
            "api": "responses",
            "requested_model": self.config.reasoning_model,
            "response_id": value.get("id").and_then(Value::as_str),
            "response_model": response_model,
            "image_count": image_paths.len(),
        }));
        extract_structured_output(value)
    }

    fn record_request(&self, value: Value) {
        self.request_log.borrow_mut().push(value);
    }

    fn warn_model_mismatch(&self, operation: &str, requested: &str, response_model: Option<&str>) {
        if let Some(response_model) = response_model
            && response_model != requested
        {
            eprintln!(
                "burn_synth_scene: OpenAI {operation} returned model `{response_model}` while `{requested}` was requested"
            );
        }
    }
}

impl SceneAiProvider for OpenAiSceneProvider {
    fn plan_objects(&self, request: &SceneReasoningRequest) -> SceneResult<SceneObjectManifest> {
        let mut value = self.post_responses_schema(
            "plan_objects",
            &request.prompt,
            object_manifest_schema(),
            &[
                request.source_scene_path.clone(),
                request.object_reference_image_path.clone(),
            ],
        )?;
        value["source_scene_path"] = json!(request.source_scene_path.display().to_string());
        serde_json::from_value(value).map_err(|err| SceneError::Provider(err.to_string()))
    }

    fn generate_object_images(&self, request: &ObjectImageRequest) -> SceneResult<Vec<Vec<u8>>> {
        let url = format!(
            "{}/v1/images/edits",
            self.config.base_url.trim_end_matches('/')
        );
        let mut form = Form::new()
            .text("model", self.config.image_model.clone())
            .text("prompt", request.prompt.clone())
            .text("n", request.candidate_count.to_string())
            .text("size", request.size.clone())
            .text("quality", request.quality.clone())
            .text("background", "opaque");
        for image_path in [
            request.source_scene_path.as_str(),
            request.source_crop_path.as_str(),
            request.object_reference_image_path.as_str(),
        ] {
            let bytes = fs::read(image_path)?;
            let file_name = Path::new(image_path)
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or("input.png")
                .to_string();
            form = form.part(
                "image[]",
                Part::bytes(bytes)
                    .file_name(file_name)
                    .mime_str(image_mime_type(Path::new(image_path)))
                    .map_err(|err| SceneError::Http(err.to_string()))?,
            );
        }
        let response = self
            .auth(self.client.post(url).multipart(form))
            .send()
            .map_err(|err| SceneError::Http(err.to_string()))?;
        let status = response.status();
        let value: Value = response
            .json()
            .map_err(|err| SceneError::Http(format!("decode image response body: {err}")))?;
        if !status.is_success() {
            return Err(SceneError::Http(format!(
                "status {status}: {}",
                redact_openai_value(&value)
            )));
        }
        let response_model = value.get("model").and_then(Value::as_str);
        self.warn_model_mismatch(
            "generate_object_images",
            &self.config.image_model,
            response_model,
        );
        self.record_request(json!({
            "operation": "generate_object_images",
            "api": "images.edits",
            "object_id": request.object.id,
            "requested_model": self.config.image_model,
            "response_id": value.get("id").and_then(Value::as_str),
            "response_model": response_model,
            "requested_candidates": request.candidate_count,
            "quality": request.quality,
            "size": request.size,
        }));
        let data = value
            .get("data")
            .and_then(Value::as_array)
            .ok_or_else(|| SceneError::Provider("image response missing data array".to_string()))?;
        let mut images = Vec::with_capacity(data.len());
        for item in data {
            let b64 = item
                .get("b64_json")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    SceneError::Provider("image response item missing b64_json".to_string())
                })?;
            images.push(
                BASE64_STANDARD
                    .decode(b64)
                    .map_err(|err| SceneError::Provider(format!("decode image base64: {err}")))?,
            );
        }
        Ok(images)
    }

    fn plan_scene_bsn(&self, request: &SceneBsnRequest) -> SceneResult<String> {
        let value = self.post_responses_schema(
            "plan_scene_bsn",
            &request.prompt,
            scene_bsn_schema(),
            &[request.source_scene_path.clone()],
        )?;
        let bsn = value
            .get("bsn")
            .and_then(Value::as_str)
            .ok_or_else(|| SceneError::Provider("scene plan missing bsn field".to_string()))?;
        let _ = parse_scene_bsn(bsn, &request.asset_bindings)?;
        Ok(bsn.to_string())
    }

    fn provider_metadata(&self) -> Value {
        json!({
            "provider": "openai",
            "base_url": self.config.base_url,
            "project_id_set": self.config.project_id.is_some(),
            "requested_reasoning_model": self.config.reasoning_model,
            "requested_image_model": self.config.image_model,
            "requests": self.request_log.borrow().clone(),
        })
    }
}

pub fn object_manifest_prompt(
    scene_path: &Path,
    reference_path: &Path,
    allow_catalog_reuse: bool,
) -> String {
    format!(
        "Analyze the source scene image at `{}` and produce a strict object manifest for 3D reconstruction. \
Use the reference image `{}` as the expected clean isolated object-image style: single object, centered, full visible silhouette, neutral background, 3/4 camera. \
For the furniture demo prefer reusable object groups: one tan open sectional sofa, one coffee table, one reusable chair group for repeated chairs, and no generated cube/proxy furniture. \
Include scene_calibration when a dominant table or seating arrangement is visible: table_center in normalized image coordinates, table_axis_degrees where 0 means table length points away from the camera in the source image, table_size_m in real meters, and camera yaw/radius plus positive orbit camera pitch degrees above the floor for a source-like viewer camera. Use Bevy/PanOrbit yaw convention: 180 degrees places the camera on the near/source side looking toward positive table depth, 0 degrees places it on the far side. \
For repeated reusable objects, set instance_count to the number of visible instances and fill instances with one bbox/contact/rotation_hint_degrees/facing_yaw_degrees/side/slot_index/target_footprint_m entry per observed object. Do not rely on a single group bbox being split later. Use side=left/right/near/far/head/foot relative to the dominant table in source-image perspective. \
Set representative_instance_id to the clearest single reusable instance crop; never use a bbox that contains the table plus all chairs as the representative crop. \
Normalized bboxes must be [x_min,y_min,x_max,y_max]. Contact points are normalized [x,y] pixels at the visible floor/object contact point, usually near the bottom center of that instance. For large furniture, the bbox should tightly contain only the object, not the whole room context. \
Object prompts must preserve observed scale relationships and plan shape; do not describe a more curved, more symmetric, more wrapped, or more complete product than the source actually shows. \
For the sofa specifically, describe the visible source object as a wide low tan open sectional with a mostly straight long run and a gentle rounded right-side bend; never ask for a complete C-shaped, horseshoe, U-shaped, circular, ring-like, or wraparound couch. allow_catalog_reuse={}.",
        scene_path.display(),
        reference_path.display(),
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
    format!(
        "{}\n{}\nReference style image: `{}`.\nObject id: {}.\nObject label: {}.\nSource crop bbox: [{:.4},{:.4},{:.4},{:.4}].\nCamera/orientation: {} {}\nSource-preserving edit requirement: use the source crop as the geometry anchor. Isolate and clean up the same observed object instead of inventing a new product render, new plan shape, or canonical showroom pose. Keep the object's visible perspective, footprint, proportions, curvature, and contact points consistent with the crop. Generate a clean isolated object image for 3D reconstruction. Preserve the source object geometry, material, color, scale proportions, and camera angle. Do not include the room, rug, table clutter, extra chairs, people, walls, text, shadows cast by the original scene, or background furniture. Do not replace the object with a proxy, cube, simplified block, alternate furniture type, or stylized approximation. Full object visible when possible, but do not hallucinate unobserved shape or wraparound structure to make it look complete. {}\n{}",
        object.object_prompt,
        geometry,
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
        "Geometry constraints: preserve the observed source sofa crop as a wide low tan open sectional, not a new product concept. It should read as a long mostly straight sofa run with a gentle rounded right-side bend and an open end, viewed from the source perspective. Do not generate a complete C-shaped, U-shaped, horseshoe, circular, ring-like, wraparound, or showroom conversation-pit sofa. Do not add an extra return segment to close the shape. Keep the silhouette wide and low, keep the visible straight run mostly straight, preserve the open end and source-facing perspective, keep seat thickness uniform, back panels vertical, and legs small/dark where visible."
    } else if descriptor.contains("chair") {
        "Geometry constraints: preserve one complete high-back segmented chair with stacked horizontal back cushions, padded seat, two metal loop arms, central pedestal, and five-star base. Do not generate multiple chairs in one image."
    } else if descriptor.contains("table") {
        "Geometry constraints: preserve a flat rectangular tabletop with real thickness, four straight vertical legs and/or a slim rectangular metal frame. Do not merge the tabletop into the background. Do not omit thin legs, rails, or feet. Keep all frame lines straight and parallel."
    } else {
        "Geometry constraints: preserve the exact observed object silhouette and proportions from the source crop; do not add extra objects or simplify fine structural members."
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
struct ObjectImageSuitability {
    background_rgb: [u8; 3],
    contrast_ratio_gt15: f32,
    contrast_ratio_gt25: f32,
    contrast_ratio_gt40: f32,
    score: f32,
}

#[derive(Clone, Debug, Serialize)]
struct ObjectImageMatteStats {
    alpha_coverage: f32,
    alpha_bbox: Option<[u32; 4]>,
}

fn decode_generated_object_rgb(bytes: &[u8]) -> SceneResult<image::RgbImage> {
    Ok(image::load_from_memory(bytes)
        .map_err(|err| SceneError::Image(format!("decode generated image: {err}")))?
        .to_rgb8())
}

fn score_generated_object_rgb(image: &image::RgbImage) -> ObjectImageSuitability {
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

fn matte_generated_object_rgb(
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

fn generated_shape_consistency_score(
    object: &SceneObjectSpec,
    matte: &ObjectImageMatteStats,
    source_image_aspect: f32,
) -> f32 {
    if object.instance_count > 1 || !object.instances.is_empty() {
        return 1.0;
    }
    let descriptor = object_descriptor(object);
    let strict_ratio = if descriptor.contains("sofa")
        || descriptor.contains("couch")
        || descriptor.contains("sectional")
    {
        Some(0.62)
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

fn image_dimensions_aspect(path: &Path) -> SceneResult<f32> {
    let (width, height) = image::image_dimensions(path)?;
    Ok(width.max(1) as f32 / height.max(1) as f32)
}

#[derive(Clone, Copy, Debug)]
pub struct GroundedSceneLayoutConfig {
    pub camera_height_m: f32,
    pub camera_pitch_down_degrees: f32,
    pub vertical_fov_degrees: f32,
    pub image_aspect: f32,
    pub floor_y: f32,
    pub seating_clearance_m: f32,
}

impl Default for GroundedSceneLayoutConfig {
    fn default() -> Self {
        Self {
            camera_height_m: 3.2,
            camera_pitch_down_degrees: 58.0,
            vertical_fov_degrees: 72.0,
            image_aspect: 2.0,
            floor_y: 0.0,
            seating_clearance_m: 0.18,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct GroundedSceneLayout {
    pub bsn: String,
    pub placements: Vec<GroundedScenePlacement>,
    pub camera: SceneCamera,
    pub rug_center: [f32; 3],
    pub rug_scale: [f32; 3],
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct GroundedScenePlacement {
    pub entity_id: String,
    pub asset_id: String,
    pub object_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instance_id: Option<String>,
    pub label: String,
    pub source_bbox: [f32; 4],
    pub contact_pixel: [f32; 2],
    pub ground_point: [f32; 3],
    pub translation: [f32; 3],
    pub rotation_y_degrees: f32,
    pub scale: [f32; 3],
    pub local_aabb: SceneAssetAabb,
    pub target_footprint_m: [f32; 2],
}

#[derive(Clone, Debug, PartialEq)]
struct ResolvedSceneObjectInstance {
    id: Option<String>,
    bbox: [f32; 4],
    contact_pixel: [f32; 2],
    rotation_hint_degrees: Option<f32>,
    facing_yaw_degrees: Option<f32>,
    side: Option<SceneInstanceSide>,
    slot_index: Option<usize>,
    target_footprint_m: Option<[f32; 2]>,
}

pub fn grounded_scene_bsn(
    manifest: &SceneObjectManifest,
    assets: &[SceneAssetBinding],
) -> SceneResult<String> {
    Ok(grounded_scene_layout_for_manifest(manifest, assets)?.bsn)
}

pub fn grounded_scene_layout_for_manifest(
    manifest: &SceneObjectManifest,
    assets: &[SceneAssetBinding],
) -> SceneResult<GroundedSceneLayout> {
    let mut config = GroundedSceneLayoutConfig::default();
    if let Ok(aspect) = image_dimensions_aspect(Path::new(&manifest.source_scene_path)) {
        config.image_aspect = aspect;
    }
    grounded_scene_layout(manifest, assets, config)
}

#[derive(Clone, Copy, Debug)]
struct MetricSceneFrame {
    table_axis_degrees: f32,
    table_size_m: [f32; 2],
    seating_clearance_m: f32,
    camera_yaw_degrees: Option<f32>,
    camera_pitch_degrees: Option<f32>,
    camera_radius_m: Option<f32>,
    vertical_fov_degrees: Option<f32>,
}

impl MetricSceneFrame {
    fn from_manifest(
        manifest: &SceneObjectManifest,
        config: GroundedSceneLayoutConfig,
    ) -> Option<Self> {
        let calibration = manifest.scene_calibration?;
        Some(Self {
            table_axis_degrees: finite_or(calibration.table_axis_degrees, 0.0),
            table_size_m: sane_footprint(calibration.table_size_m.unwrap_or([3.2, 1.2])),
            seating_clearance_m: config.seating_clearance_m.clamp(0.04, 0.60),
            camera_yaw_degrees: calibration
                .camera_yaw_degrees
                .filter(|value| value.is_finite()),
            camera_pitch_degrees: calibration
                .camera_pitch_degrees
                .filter(|value| value.is_finite()),
            camera_radius_m: calibration
                .camera_radius_m
                .filter(|value| value.is_finite() && *value > 0.0),
            vertical_fov_degrees: calibration
                .vertical_fov_degrees
                .filter(|value| value.is_finite() && *value > 0.0),
        })
    }

    fn table_point(&self) -> [f32; 3] {
        [0.0, 0.0, 0.0]
    }

    fn side_point(
        &self,
        side: SceneInstanceSide,
        slot_index: usize,
        slot_count: usize,
        target_footprint: [f32; 2],
    ) -> [f32; 3] {
        let table_width = self.table_size_m[0].max(0.5);
        let table_length = self.table_size_m[1].max(0.5);
        let object_depth = target_footprint[1].max(target_footprint[0] * 0.75).max(0.2);
        let clearance = self
            .seating_clearance_m
            .max((object_depth * 0.20).clamp(0.08, 0.28));
        let image_side_sign = self.camera_yaw_degrees.map(|yaw| {
            if yaw.to_radians().cos() >= 0.0 {
                1.0
            } else {
                -1.0
            }
        });
        let slot_count = slot_count.max(1);
        let slot_t = if slot_count <= 1 {
            0.0
        } else {
            (slot_index.min(slot_count - 1) as f32 + 0.5) / slot_count as f32 - 0.5
        };
        let local = match side {
            SceneInstanceSide::Left => [
                (-table_width * 0.5 - object_depth * 0.5 - clearance)
                    * image_side_sign.unwrap_or(1.0),
                slot_t * table_length * 0.88,
            ],
            SceneInstanceSide::Right => [
                (table_width * 0.5 + object_depth * 0.5 + clearance)
                    * image_side_sign.unwrap_or(1.0),
                slot_t * table_length * 0.88,
            ],
            SceneInstanceSide::Near | SceneInstanceSide::Foot => [
                slot_t * table_width * 0.80,
                if let Some(sign) = image_side_sign {
                    (table_length * 0.5 + object_depth * 0.5 + clearance) * sign
                } else {
                    -table_length * 0.5 - object_depth * 0.5 - clearance
                },
            ],
            SceneInstanceSide::Far | SceneInstanceSide::Head => [
                slot_t * table_width * 0.80,
                if let Some(sign) = image_side_sign {
                    (-table_length * 0.5 - object_depth * 0.5 - clearance) * sign
                } else {
                    table_length * 0.5 + object_depth * 0.5 + clearance
                },
            ],
            SceneInstanceSide::Unknown => [0.0, 0.0],
        };
        rotate_table_frame_point(local, self.table_axis_degrees)
    }
}

pub fn grounded_scene_layout(
    manifest: &SceneObjectManifest,
    assets: &[SceneAssetBinding],
    config: GroundedSceneLayoutConfig,
) -> SceneResult<GroundedSceneLayout> {
    if manifest.objects.is_empty() {
        return Err(SceneError::Validation(
            "grounded scene layout requires at least one object".to_string(),
        ));
    }
    if assets.is_empty() {
        return Err(SceneError::Validation(
            "grounded scene layout requires at least one asset binding".to_string(),
        ));
    }

    let asset_by_object = assets
        .iter()
        .map(|asset| (asset.object_id.as_str(), asset))
        .collect::<HashMap<_, _>>();
    let metric_frame = MetricSceneFrame::from_manifest(manifest, config);
    let table_contact = manifest
        .objects
        .iter()
        .find(|object| is_table_like(object))
        .and_then(|object| {
            resolved_object_instances(object)
                .into_iter()
                .next()
                .map(|instance| object_contact_point(object, &instance, config))
        })
        .transpose()?;

    let mut raw = Vec::new();
    for object in &manifest.objects {
        let asset = asset_by_object.get(object.id.as_str()).ok_or_else(|| {
            SceneError::Validation(format!(
                "missing asset binding for scene object `{}`",
                object.id
            ))
        })?;
        for instance in resolved_object_instances(object) {
            let contact = object_contact_point(object, &instance, config)?;
            raw.push((object, *asset, instance, contact));
        }
    }
    let side_slots = metric_side_slots(&raw);

    let center_x = table_contact.map(|point| point[0]).unwrap_or_else(|| {
        raw.iter().map(|(_, _, _, point)| point[0]).sum::<f32>() / raw.len() as f32
    });
    let center_z = table_contact.map(|point| point[2]).unwrap_or_else(|| {
        raw.iter().map(|(_, _, _, point)| point[2]).sum::<f32>() / raw.len() as f32
    });
    let table_centered = Some([0.0, config.floor_y, 0.0]);

    let mut placements = Vec::with_capacity(raw.len());
    for (raw_index, (object, asset, instance, contact)) in raw.into_iter().enumerate() {
        let local_aabb = asset.local_aabb.unwrap_or(SceneAssetAabb {
            min: [-0.5, 0.0, -0.5],
            max: [0.5, 1.0, 0.5],
        });
        let asset_frame = scene_asset_frame(asset, object, local_aabb);
        let target_footprint = target_footprint_m(object, &instance, asset_frame, metric_frame);
        let scale_value = uniform_asset_scale(local_aabb, target_footprint);
        let ground_point = metric_ground_point(
            object,
            &instance,
            target_footprint,
            metric_frame,
            side_slots.get(&raw_index).copied(),
        )
        .unwrap_or([contact[0] - center_x, config.floor_y, contact[2] - center_z]);
        let translation = [
            ground_point[0],
            config.floor_y - local_aabb.min[1] * scale_value,
            ground_point[2],
        ];
        let rotation_y_degrees = normalize_degrees(
            instance
                .rotation_hint_degrees
                .or(instance.facing_yaw_degrees)
                .or(object.rotation_hint_degrees)
                .unwrap_or_else(|| {
                    grounded_yaw_degrees(
                        object,
                        &instance,
                        ground_point,
                        table_centered,
                        metric_frame,
                    )
                })
                - asset_frame.yaw_offset_degrees,
        );
        let entity_id = if let Some(instance_id) = instance.id.as_deref() {
            sanitize_bsn_identifier(&format!("{}_{}", object.id, instance_id))
        } else {
            sanitize_bsn_identifier(&object.id)
        };
        placements.push(GroundedScenePlacement {
            entity_id,
            asset_id: asset.asset_id.clone(),
            object_id: object.id.clone(),
            instance_id: instance.id.clone(),
            label: object.label.clone(),
            source_bbox: instance.bbox,
            contact_pixel: instance.contact_pixel,
            ground_point,
            translation,
            rotation_y_degrees,
            scale: [scale_value, scale_value, scale_value],
            local_aabb,
            target_footprint_m: target_footprint,
        });
    }
    normalize_repeated_asset_scales(&mut placements, config.floor_y);

    let (rug_center, rug_scale) = rug_from_placements(&placements, config.floor_y);
    let camera = grounded_camera_from_placements(&placements, config, metric_frame);
    let bsn = grounded_bsn_text(assets, &placements, rug_center, rug_scale, &camera);
    let parsed = parse_scene_bsn(&bsn, assets)?;
    if parsed.placements.len() != placements.len() {
        return Err(SceneError::Validation(
            "grounded BSN placement count changed during parse".to_string(),
        ));
    }
    Ok(GroundedSceneLayout {
        bsn,
        placements,
        camera,
        rug_center,
        rug_scale,
    })
}

fn object_contact_point(
    object: &SceneObjectSpec,
    instance: &ResolvedSceneObjectInstance,
    config: GroundedSceneLayoutConfig,
) -> SceneResult<[f32; 3]> {
    let [u, v] = instance.contact_pixel;
    let u = u.clamp(0.0, 1.0);
    let v = v.clamp(0.02, 0.995);
    floor_intersection_from_normalized_pixel(u, v, config).ok_or_else(|| {
        SceneError::Validation(format!(
            "object `{}` bottom point did not intersect the estimated ground plane",
            object.id
        ))
    })
}

fn resolved_object_instances(object: &SceneObjectSpec) -> Vec<ResolvedSceneObjectInstance> {
    if !object.instances.is_empty() {
        return object
            .instances
            .iter()
            .enumerate()
            .map(|(index, instance)| {
                let bbox = normalize_bbox(instance.bbox);
                ResolvedSceneObjectInstance {
                    id: instance
                        .id
                        .clone()
                        .filter(|value| !value.trim().is_empty())
                        .or_else(|| Some(format!("{:03}", index + 1))),
                    bbox,
                    contact_pixel: instance
                        .contact
                        .map(normalize_contact_pixel)
                        .unwrap_or_else(|| bbox_bottom_center(bbox)),
                    rotation_hint_degrees: instance.rotation_hint_degrees,
                    facing_yaw_degrees: instance.facing_yaw_degrees,
                    side: instance.side,
                    slot_index: instance.slot_index,
                    target_footprint_m: instance.target_footprint_m,
                }
            })
            .collect();
    }

    let instance_count = object.instance_count.max(1);
    (0..instance_count)
        .map(|index| {
            let bbox = instance_bbox(object, index, instance_count);
            ResolvedSceneObjectInstance {
                id: if instance_count == 1 {
                    None
                } else {
                    Some(format!("{:03}", index + 1))
                },
                bbox,
                contact_pixel: bbox_bottom_center(bbox),
                rotation_hint_degrees: None,
                facing_yaw_degrees: None,
                side: None,
                slot_index: None,
                target_footprint_m: None,
            }
        })
        .collect()
}

fn bbox_bottom_center(bbox: [f32; 4]) -> [f32; 2] {
    [
        ((bbox[0] + bbox[2]) * 0.5).clamp(0.0, 1.0),
        bbox[3].clamp(0.0, 1.0),
    ]
}

fn normalize_contact_pixel(value: [f32; 2]) -> [f32; 2] {
    [value[0].clamp(0.0, 1.0), value[1].clamp(0.0, 1.0)]
}

fn floor_intersection_from_normalized_pixel(
    u: f32,
    v: f32,
    config: GroundedSceneLayoutConfig,
) -> Option<[f32; 3]> {
    let tan_half = (config.vertical_fov_degrees.to_radians() * 0.5).tan();
    let x = (2.0 * u - 1.0) * config.image_aspect.max(0.1) * tan_half;
    let y = (1.0 - 2.0 * v) * tan_half;
    let z = 1.0;
    let pitch = config.camera_pitch_down_degrees.to_radians();
    let cos = pitch.cos();
    let sin = pitch.sin();
    let ray = [x, y * cos - z * sin, y * sin + z * cos];
    if ray[1] >= -1.0e-5 {
        return None;
    }
    let t = (config.floor_y - config.camera_height_m) / ray[1];
    Some([ray[0] * t, config.floor_y, ray[2] * t])
}

fn instance_bbox(
    object: &SceneObjectSpec,
    instance_index: usize,
    instance_count: usize,
) -> [f32; 4] {
    let bbox = normalize_bbox(object.bbox);
    if instance_count <= 1 {
        return bbox;
    }
    let width = (bbox[2] - bbox[0]).max(1.0e-5);
    let step = width / instance_count as f32;
    let x0 = bbox[0] + step * instance_index as f32;
    let x1 = if instance_index + 1 == instance_count {
        bbox[2]
    } else {
        x0 + step
    };
    [x0, bbox[1], x1, bbox[3]]
}

fn target_footprint_m(
    object: &SceneObjectSpec,
    instance: &ResolvedSceneObjectInstance,
    asset_frame: SceneAssetFrame,
    metric_frame: Option<MetricSceneFrame>,
) -> [f32; 2] {
    if let Some(footprint) = instance.target_footprint_m {
        return sane_footprint(footprint);
    }
    if let Some(footprint) = object.target_footprint_m {
        return sane_footprint(footprint);
    }
    if is_table_like(object)
        && let Some(frame) = metric_frame
    {
        return sane_footprint(frame.table_size_m);
    }
    if let Some(footprint) = asset_frame.footprint_m {
        return sane_footprint(footprint);
    }
    let descriptor = object_descriptor(object);
    let bbox = normalize_bbox(object.bbox);
    let width = (bbox[2] - bbox[0]).max(0.01);
    if descriptor.contains("sofa")
        || descriptor.contains("couch")
        || descriptor.contains("sectional")
    {
        let length = if width > 0.8 { 4.8 } else { 3.4 };
        [length, 2.2]
    } else if descriptor.contains("conference") && descriptor.contains("table") {
        [3.2, 1.15]
    } else if descriptor.contains("table") {
        [1.8, 0.95]
    } else if descriptor.contains("chair") {
        [0.58, 0.62]
    } else {
        [1.0, 1.0]
    }
}

fn sane_footprint(value: [f32; 2]) -> [f32; 2] {
    [value[0].clamp(0.1, 12.0), value[1].clamp(0.1, 12.0)]
}

fn uniform_asset_scale(local_aabb: SceneAssetAabb, target_footprint: [f32; 2]) -> f32 {
    let size = local_aabb.size();
    let local_footprint = size[0].max(size[2]).max(1.0e-5);
    let target = target_footprint[0].max(target_footprint[1]).max(0.1);
    (target / local_footprint).clamp(0.05, 20.0)
}

fn normalize_repeated_asset_scales(placements: &mut [GroundedScenePlacement], floor_y: f32) {
    let mut grouped: HashMap<String, (f32, usize)> = HashMap::new();
    for placement in placements.iter() {
        let entry = grouped
            .entry(placement.asset_id.clone())
            .or_insert((0.0, 0));
        entry.0 += placement.scale[0].abs();
        entry.1 += 1;
    }
    let repeated_scale = grouped
        .into_iter()
        .filter_map(|(asset_id, (sum, count))| {
            if count > 1 {
                Some((asset_id, (sum / count as f32).clamp(0.05, 20.0)))
            } else {
                None
            }
        })
        .collect::<HashMap<_, _>>();
    for placement in placements.iter_mut() {
        let Some(scale) = repeated_scale.get(&placement.asset_id).copied() else {
            continue;
        };
        placement.scale = [scale, scale, scale];
        placement.translation[1] = floor_y - placement.local_aabb.min[1] * scale;
    }
}

fn metric_side_slots(
    raw: &[(
        &SceneObjectSpec,
        &SceneAssetBinding,
        ResolvedSceneObjectInstance,
        [f32; 3],
    )],
) -> HashMap<usize, (usize, usize)> {
    let mut by_side: HashMap<SceneInstanceSide, Vec<(usize, f32)>> = HashMap::new();
    for (index, (_, _, instance, _)) in raw.iter().enumerate() {
        let Some(side) = instance
            .side
            .filter(|side| *side != SceneInstanceSide::Unknown)
        else {
            continue;
        };
        by_side
            .entry(side)
            .or_default()
            .push((index, side_contact_axis(side, instance.contact_pixel)));
    }

    let mut slots = HashMap::new();
    for (_side, mut entries) in by_side {
        let count = entries.len().max(1);
        let explicit_valid = entries.iter().all(|(index, _)| {
            raw[*index]
                .2
                .slot_index
                .is_some_and(|slot_index| slot_index < count)
        });
        if explicit_valid {
            entries.sort_by_key(|(index, _)| raw[*index].2.slot_index.unwrap_or(0));
        } else {
            entries.sort_by(|(left_index, left_axis), (right_index, right_axis)| {
                left_axis
                    .partial_cmp(right_axis)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| left_index.cmp(right_index))
            });
        }
        for (slot_index, (raw_index, _)) in entries.into_iter().enumerate() {
            slots.insert(raw_index, (slot_index, count));
        }
    }
    slots
}

fn side_contact_axis(side: SceneInstanceSide, contact_pixel: [f32; 2]) -> f32 {
    match side {
        SceneInstanceSide::Left | SceneInstanceSide::Right => contact_pixel[1],
        SceneInstanceSide::Near
        | SceneInstanceSide::Far
        | SceneInstanceSide::Head
        | SceneInstanceSide::Foot => contact_pixel[0],
        SceneInstanceSide::Unknown => 0.5,
    }
}

fn metric_ground_point(
    object: &SceneObjectSpec,
    instance: &ResolvedSceneObjectInstance,
    target_footprint: [f32; 2],
    metric_frame: Option<MetricSceneFrame>,
    side_slot: Option<(usize, usize)>,
) -> Option<[f32; 3]> {
    let frame = metric_frame?;
    if is_table_like(object) {
        return Some(frame.table_point());
    }
    let side = instance.side?;
    if side == SceneInstanceSide::Unknown {
        return None;
    }
    let (slot_index, slot_count) = side_slot.unwrap_or_else(|| {
        let slot_count = 1;
        let slot_index = instance
            .slot_index
            .or_else(|| instance.id.as_deref().and_then(last_numeric_suffix))
            .unwrap_or_else(|| side_slot_from_contact(side, instance.contact_pixel, slot_count));
        (slot_index, slot_count)
    });
    Some(frame.side_point(side, slot_index, slot_count, target_footprint))
}

fn side_slot_from_contact(
    side: SceneInstanceSide,
    contact_pixel: [f32; 2],
    slot_count: usize,
) -> usize {
    if slot_count <= 1 {
        return 0;
    }
    let axis = match side {
        SceneInstanceSide::Left | SceneInstanceSide::Right => contact_pixel[1],
        SceneInstanceSide::Near
        | SceneInstanceSide::Far
        | SceneInstanceSide::Head
        | SceneInstanceSide::Foot => contact_pixel[0],
        SceneInstanceSide::Unknown => 0.5,
    };
    ((axis.clamp(0.0, 0.999) * slot_count as f32).floor() as usize).min(slot_count - 1)
}

fn last_numeric_suffix(value: &str) -> Option<usize> {
    let suffix = value
        .rsplit(|ch: char| !ch.is_ascii_digit())
        .find(|part| !part.is_empty())?;
    suffix
        .parse::<usize>()
        .ok()
        .map(|value| value.saturating_sub(1))
}

fn grounded_yaw_degrees(
    object: &SceneObjectSpec,
    instance: &ResolvedSceneObjectInstance,
    from: [f32; 3],
    table: Option<[f32; 3]>,
    metric_frame: Option<MetricSceneFrame>,
) -> f32 {
    if is_table_like(object) {
        return metric_frame
            .map(|frame| frame.table_axis_degrees)
            .unwrap_or(0.0);
    }
    if let (Some(frame), Some(side)) = (metric_frame, instance.side)
        && side != SceneInstanceSide::Unknown
    {
        if let Some(yaw) = bsn_yaw_toward_point_degrees(from, frame.table_point()) {
            return yaw;
        }
    }
    let Some(target) = table else {
        return 0.0;
    };
    bsn_yaw_toward_point_degrees(from, target).unwrap_or(0.0)
}

fn bsn_yaw_toward_point_degrees(from: [f32; 3], target: [f32; 3]) -> Option<f32> {
    let dx = target[0] - from[0];
    let dz = target[2] - from[2];
    if !dx.is_finite() || !dz.is_finite() || dx.abs() + dz.abs() <= 1.0e-5 {
        return None;
    }
    Some(normalize_degrees(dx.atan2(dz).to_degrees()))
}

fn scene_asset_frame(
    asset: &SceneAssetBinding,
    object: &SceneObjectSpec,
    local_aabb: SceneAssetAabb,
) -> SceneAssetFrame {
    if let Some(frame) = asset.canonical_frame {
        return frame;
    }
    let size = local_aabb.size();
    let descriptor = object_descriptor(object);
    let footprint_m = object.target_footprint_m;
    if descriptor.contains("table") {
        let yaw_offset = if size[0] > size[2] * 1.15 { 90.0 } else { 0.0 };
        SceneAssetFrame {
            yaw_offset_degrees: yaw_offset,
            footprint_m,
        }
    } else if descriptor.contains("chair") {
        SceneAssetFrame {
            yaw_offset_degrees: 0.0,
            footprint_m,
        }
    } else {
        SceneAssetFrame {
            yaw_offset_degrees: 0.0,
            footprint_m,
        }
    }
}

fn rotate_table_frame_point(local_xz: [f32; 2], yaw_degrees: f32) -> [f32; 3] {
    let yaw = yaw_degrees.to_radians();
    let cos = yaw.cos();
    let sin = yaw.sin();
    [
        local_xz[0] * cos + local_xz[1] * sin,
        0.0,
        -local_xz[0] * sin + local_xz[1] * cos,
    ]
}

fn normalize_degrees(mut degrees: f32) -> f32 {
    if !degrees.is_finite() {
        return 0.0;
    }
    while degrees <= -180.0 {
        degrees += 360.0;
    }
    while degrees > 180.0 {
        degrees -= 360.0;
    }
    degrees
}

fn finite_or(value: Option<f32>, fallback: f32) -> f32 {
    value.filter(|value| value.is_finite()).unwrap_or(fallback)
}

fn rug_from_placements(
    placements: &[GroundedScenePlacement],
    floor_y: f32,
) -> ([f32; 3], [f32; 3]) {
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_z = f32::INFINITY;
    let mut max_z = f32::NEG_INFINITY;
    for placement in placements {
        let half = placement.target_footprint_m[0].max(placement.target_footprint_m[1]) * 0.5;
        min_x = min_x.min(placement.ground_point[0] - half);
        max_x = max_x.max(placement.ground_point[0] + half);
        min_z = min_z.min(placement.ground_point[2] - half);
        max_z = max_z.max(placement.ground_point[2] + half);
    }
    if !min_x.is_finite() {
        return ([0.0, floor_y, 0.0], [4.0, 1.0, 3.0]);
    }
    let center = [(min_x + max_x) * 0.5, floor_y, (min_z + max_z) * 0.5];
    let scale = [
        (max_x - min_x + 0.75).max(2.0),
        1.0,
        (max_z - min_z + 0.75).max(2.0),
    ];
    (center, scale)
}

fn grounded_camera_from_placements(
    placements: &[GroundedScenePlacement],
    config: GroundedSceneLayoutConfig,
    metric_frame: Option<MetricSceneFrame>,
) -> SceneCamera {
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_z = f32::INFINITY;
    let mut max_z = f32::NEG_INFINITY;
    let mut max_extent = 4.0f32;
    for placement in placements {
        min_x = min_x.min(placement.ground_point[0]);
        max_x = max_x.max(placement.ground_point[0]);
        min_z = min_z.min(placement.ground_point[2]);
        max_z = max_z.max(placement.ground_point[2]);
        max_extent = max_extent.max(placement.ground_point[0].abs() * 2.0);
        max_extent = max_extent.max(placement.ground_point[2].abs() * 2.0);
    }
    let focus = if min_x.is_finite() {
        [
            (min_x + max_x) * 0.5,
            config.floor_y + 0.72,
            (min_z + max_z) * 0.5,
        ]
    } else {
        [0.0, config.floor_y + 0.72, 0.0]
    };
    let radius = metric_frame
        .and_then(|frame| frame.camera_radius_m)
        .unwrap_or_else(|| (max_extent * 0.95).max(4.5));
    let yaw = metric_frame
        .and_then(|frame| frame.camera_yaw_degrees)
        .unwrap_or(180.0);
    let pitch = metric_frame
        .and_then(|frame| frame.camera_pitch_degrees)
        .map(f32::abs)
        .unwrap_or(30.0)
        .clamp(8.0, 80.0);
    let translation = camera_translation_from_orbit(focus, yaw, pitch, radius);
    SceneCamera {
        translation,
        focus,
        yaw: Some(yaw),
        pitch: Some(pitch),
        radius: Some(radius),
        vertical_fov_degrees: metric_frame
            .and_then(|frame| frame.vertical_fov_degrees)
            .or(Some(config.vertical_fov_degrees)),
    }
}

fn camera_translation_from_orbit(
    focus: [f32; 3],
    yaw_degrees: f32,
    pitch_degrees: f32,
    radius: f32,
) -> [f32; 3] {
    let yaw = yaw_degrees.to_radians();
    let pitch = pitch_degrees.to_radians();
    let horizontal = radius * pitch.cos().abs();
    [
        focus[0] + horizontal * yaw.sin(),
        (focus[1] + radius * pitch.sin()).max(0.25),
        focus[2] + horizontal * yaw.cos(),
    ]
}

fn grounded_bsn_text(
    assets: &[SceneAssetBinding],
    placements: &[GroundedScenePlacement],
    rug_center: [f32; 3],
    rug_scale: [f32; 3],
    camera: &SceneCamera,
) -> String {
    let mut out = String::from("synth_scene_v1 {\n");
    let mut declared = HashSet::new();
    for placement in placements {
        if declared.insert(placement.asset_id.as_str())
            && assets
                .iter()
                .any(|asset| asset.asset_id == placement.asset_id)
        {
            out.push_str(&format!(
                "asset {} = \"generated:{}\";\n",
                placement.asset_id, placement.asset_id
            ));
        }
    }
    out.push_str(&format!(
        "environment rug translation [{}] scale [{}] color [0.62,0.02,0.26];\n",
        fmt_vec3(rug_center),
        fmt_vec3(rug_scale)
    ));
    for placement in placements {
        out.push_str(&format!(
            "spawn {} uses {} translation [{}] rotation_y {} scale [{}];\n",
            placement.entity_id,
            placement.asset_id,
            fmt_vec3(placement.translation),
            fmt_num(placement.rotation_y_degrees),
            fmt_vec3(placement.scale)
        ));
    }
    out.push_str(&format!(
        "camera translation [{}] focus [{}] yaw {} pitch {} radius {} vertical_fov {};\n",
        fmt_vec3(camera.translation),
        fmt_vec3(camera.focus),
        fmt_num(camera.yaw.unwrap_or(0.0)),
        fmt_num(camera.pitch.unwrap_or(30.0)),
        fmt_num(camera.radius.unwrap_or(5.0)),
        fmt_num(camera.vertical_fov_degrees.unwrap_or(72.0))
    ));
    out.push_str("}\n");
    out
}

fn fmt_vec3(value: [f32; 3]) -> String {
    format!(
        "{},{},{}",
        fmt_num(value[0]),
        fmt_num(value[1]),
        fmt_num(value[2])
    )
}

fn fmt_num(value: f32) -> String {
    let mut text = format!("{value:.4}");
    while text.contains('.') && text.ends_with('0') {
        text.pop();
    }
    if text.ends_with('.') {
        text.push('0');
    }
    if text == "-0.0" {
        "0.0".to_string()
    } else {
        text
    }
}

fn is_table_like(object: &SceneObjectSpec) -> bool {
    object_descriptor(object).contains("table")
}

fn object_descriptor(object: &SceneObjectSpec) -> String {
    format!(
        "{} {} {}",
        object.id,
        object.label,
        object.aliases.join(" ")
    )
    .to_ascii_lowercase()
}

fn sanitize_bsn_identifier(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "entity".to_string()
    } else {
        out
    }
}

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

fn representative_crop_bbox(object: &SceneObjectSpec) -> [f32; 4] {
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

fn validate_build_config(config: &SceneBuildConfig) -> SceneResult<()> {
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
        .chain(scale.into_iter())
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

fn normalize_bbox(mut bbox: [f32; 4]) -> [f32; 4] {
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

fn extract_structured_output(value: Value) -> SceneResult<Value> {
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

fn redact_openai_value(value: &Value) -> String {
    let mut value = value.clone();
    if let Some(object) = value.as_object_mut() {
        object.remove("prompt");
        object.remove("b64_json");
    }
    value.to_string()
}

fn image_data_url(path: &Path) -> SceneResult<String> {
    let mime = image_mime_type(path);
    let bytes = fs::read(path)?;
    Ok(format!(
        "data:{mime};base64,{}",
        BASE64_STANDARD.encode(bytes)
    ))
}

fn image_mime_type(path: &Path) -> &'static str {
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

fn default_run_id(label: &str) -> String {
    format!("{}_{}", unix_compact(), label)
}

fn unix_compact() -> String {
    unix_ms().to_string()
}

fn unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn stable_hash_hex(value: &str) -> String {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

fn default_instance_count() -> usize {
    1
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::collections::VecDeque;
    use std::io::Cursor;

    struct RetryImageProvider {
        images: RefCell<VecDeque<Vec<u8>>>,
    }

    impl RetryImageProvider {
        fn new(images: Vec<Vec<u8>>) -> Self {
            Self {
                images: RefCell::new(images.into()),
            }
        }
    }

    impl SceneAiProvider for RetryImageProvider {
        fn plan_objects(
            &self,
            _request: &SceneReasoningRequest,
        ) -> SceneResult<SceneObjectManifest> {
            Err(SceneError::Provider(
                "plan_objects is not used by retry image tests".to_string(),
            ))
        }

        fn generate_object_images(
            &self,
            request: &ObjectImageRequest,
        ) -> SceneResult<Vec<Vec<u8>>> {
            let mut images = self.images.borrow_mut();
            let mut output = Vec::new();
            for _ in 0..request.candidate_count {
                output.push(images.pop_front().ok_or_else(|| {
                    SceneError::Provider("test image provider exhausted".to_string())
                })?);
            }
            Ok(output)
        }

        fn plan_scene_bsn(&self, _request: &SceneBsnRequest) -> SceneResult<String> {
            Err(SceneError::Provider(
                "plan_scene_bsn is not used by retry image tests".to_string(),
            ))
        }
    }

    fn png_bytes(image: image::RgbImage) -> Vec<u8> {
        let mut cursor = Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(image)
            .write_to(&mut cursor, image::ImageFormat::Png)
            .unwrap();
        cursor.into_inner()
    }

    #[test]
    fn preparation_records_configured_openai_models() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("scene.jpg");
        let reference = dir.path().join("input_chair.jpg");
        fs::write(&source, b"source").unwrap();
        fs::write(&reference, b"reference").unwrap();
        let config = SceneBuildConfig {
            source_scene_path: source,
            object_reference_image_path: reference,
            output_dir: dir.path().join("out"),
            candidate_count: 1,
            quality_profile: SceneQualityProfile::Quality,
            reasoning_model: "gpt-5.5".to_string(),
            image_model: "gpt-image-2".to_string(),
            allow_catalog_reuse: false,
        };
        let provider = RetryImageProvider::new(Vec::new());
        let mut pipeline = ScenePipeline::new(config, provider);

        let preparation = pipeline.prepare_openai_inputs().unwrap();

        assert_eq!(preparation.provider, "openai");
        assert_eq!(preparation.reasoning_model, "gpt-5.5");
        assert_eq!(preparation.image_model, "gpt-image-2");
    }

    fn low_contrast_candidate_png() -> Vec<u8> {
        let mut image = image::RgbImage::from_pixel(128, 128, image::Rgb([222, 222, 222]));
        for y in 50..72 {
            for x in 20..108 {
                image.put_pixel(x, y, image::Rgb([234, 234, 232]));
            }
        }
        png_bytes(image)
    }

    fn high_contrast_candidate_png() -> Vec<u8> {
        let mut image = image::RgbImage::from_pixel(128, 128, image::Rgb([31, 95, 214]));
        for y in 32..96 {
            for x in 24..104 {
                image.put_pixel(x, y, image::Rgb([238, 238, 232]));
            }
        }
        png_bytes(image)
    }

    fn chair_asset() -> SceneAssetBinding {
        SceneAssetBinding {
            asset_id: "chair_asset".to_string(),
            object_id: "chair_group".to_string(),
            label: "chair".to_string(),
            aliases: vec!["conference chair".to_string()],
            path: Some("/tmp/chair.glb".to_string()),
            cache_key: None,
            reusable: true,
            source_image_path: Some("/tmp/chair.png".to_string()),
            pipeline: Some("trellis".to_string()),
            local_aabb: Some(SceneAssetAabb {
                min: [-0.5, 0.0, -0.5],
                max: [0.5, 1.0, 0.5],
            }),
            canonical_frame: None,
            provenance: None,
        }
    }

    #[test]
    fn bsn_parser_accepts_restricted_scene_and_emits_commands() {
        let bsn = r#"
synth_scene_v1 {
asset chair_asset = "generated:chair_asset";
spawn chair_left uses chair_asset translation [-1.0,0.0,2.0] rotation_y 25.0 scale [1.0,1.0,1.0];
spawn chair_right uses chair_asset translation [1.0,0.0,2.0] rotation_y -25.0 scale [1.0,1.0,1.0];
environment rug translation [0.0,0.0,0.0] scale [4.0,1.0,3.0];
camera translation [0.0,4.0,6.0] focus [0.0,0.0,0.0] yaw 0.0 pitch -0.5 radius 6.0;
}
"#;
        let plan = parse_scene_bsn(bsn, &[chair_asset()]).expect("valid bsn");
        assert_eq!(plan.placements.len(), 2);
        assert!(plan.camera.is_some());
        let commands = scene_plan_to_mcp_commands(&plan, &[chair_asset()], true).unwrap();
        assert_eq!(commands[0]["type"], "clear_scene");
        assert_eq!(commands[1]["type"], "spawn_path");
        assert_eq!(commands[1]["cache_key"], "chair_asset");
    }

    #[test]
    fn bsn_to_mcp_envelope_preserves_commands_and_sequence() {
        let bsn = r#"
synth_scene_v1 {
asset chair_asset = "generated:chair_asset";
spawn chair_left uses chair_asset translation [-1.0,0.0,2.0] rotation_y 25.0 scale [1.0,1.0,1.0];
}
"#;
        let envelope =
            scene_bsn_to_mcp_command_envelope(bsn, &[chair_asset()], true, Some("viewer"), Some(7))
                .expect("valid envelope");
        assert_eq!(envelope["session_id"], json!("viewer"));
        assert_eq!(envelope["sequence"], json!(7));
        let commands = envelope["commands"].as_array().unwrap();
        assert_eq!(commands[0]["type"], "clear_scene");
        assert_eq!(commands[1]["type"], "spawn_path");
    }

    #[test]
    fn bsn_parser_rejects_proxy_furniture() {
        let bsn = r#"
synth_scene_v1 {
asset chair_asset = "generated:chair_asset";
spawn debug_cube_chair uses chair_asset translation [0.0,0.0,0.0] rotation_y 0.0 scale [1.0,1.0,1.0];
}
"#;
        let err = parse_scene_bsn(bsn, &[chair_asset()]).unwrap_err();
        assert!(err.to_string().contains("proxy/debug"));
    }

    #[test]
    fn scene_bsn_prompt_requires_single_line_restricted_grammar() {
        let manifest = SceneObjectManifest {
            source_scene_path: "/tmp/scene.jpg".to_string(),
            scene_calibration: None,
            objects: vec![],
        };
        let prompt = scene_bsn_prompt(&manifest, &[chair_asset()]);
        assert!(prompt.contains("Every statement must be on exactly one line"));
        assert!(prompt.contains("asset <asset_id> = \"generated:<asset_id>\";"));
        assert!(prompt.contains("spawn <entity_id> uses <asset_id>"));
    }

    #[test]
    fn grounded_scene_layout_uses_asset_aabb_scale_and_bottom_fit() {
        let manifest = SceneObjectManifest {
            source_scene_path: "/tmp/scene.jpg".to_string(),
            scene_calibration: None,
            objects: vec![
                SceneObjectSpec {
                    id: "curved_sofa_001".to_string(),
                    label: "curved sofa".to_string(),
                    aliases: vec!["sectional".to_string()],
                    bbox: [0.0, 0.2, 1.0, 0.95],
                    instances: Vec::new(),
                    representative_instance_id: None,
                    reuse_group: None,
                    instance_count: 1,
                    object_prompt: "tan sectional sofa".to_string(),
                    camera_hint: None,
                    rotation_hint_degrees: None,
                    target_footprint_m: None,
                },
                SceneObjectSpec {
                    id: "coffee_table_001".to_string(),
                    label: "coffee table".to_string(),
                    aliases: Vec::new(),
                    bbox: [0.4, 0.4, 0.65, 0.65],
                    instances: Vec::new(),
                    representative_instance_id: None,
                    reuse_group: None,
                    instance_count: 1,
                    object_prompt: "white coffee table scaled below the sofa".to_string(),
                    camera_hint: None,
                    rotation_hint_degrees: None,
                    target_footprint_m: None,
                },
            ],
        };
        let assets = vec![
            SceneAssetBinding {
                asset_id: "curved_sofa_001_asset".to_string(),
                object_id: "curved_sofa_001".to_string(),
                label: "curved sofa".to_string(),
                aliases: Vec::new(),
                path: None,
                cache_key: Some("sofa".to_string()),
                reusable: true,
                source_image_path: None,
                pipeline: Some("trellis".to_string()),
                local_aabb: Some(SceneAssetAabb {
                    min: [-0.5, -0.25, -0.5],
                    max: [0.5, 0.75, 0.5],
                }),
                canonical_frame: None,
                provenance: None,
            },
            SceneAssetBinding {
                asset_id: "coffee_table_001_asset".to_string(),
                object_id: "coffee_table_001".to_string(),
                label: "coffee table".to_string(),
                aliases: Vec::new(),
                path: None,
                cache_key: Some("table".to_string()),
                reusable: true,
                source_image_path: None,
                pipeline: Some("trellis".to_string()),
                local_aabb: Some(SceneAssetAabb {
                    min: [-0.5, -0.1, -0.5],
                    max: [0.5, 0.4, 0.5],
                }),
                canonical_frame: None,
                provenance: None,
            },
        ];

        let layout =
            grounded_scene_layout(&manifest, &assets, GroundedSceneLayoutConfig::default())
                .expect("grounded layout");

        let sofa = layout
            .placements
            .iter()
            .find(|placement| placement.object_id == "curved_sofa_001")
            .unwrap();
        let table = layout
            .placements
            .iter()
            .find(|placement| placement.object_id == "coffee_table_001")
            .unwrap();
        assert!(sofa.scale[0] > table.scale[0]);
        assert_eq!(table.target_footprint_m, [1.8, 0.95]);
        let sofa_bottom = sofa.translation[1] + sofa.local_aabb.min[1] * sofa.scale[1];
        assert!(sofa_bottom.abs() < 1.0e-4);
        assert!(
            layout
                .bsn
                .contains("spawn curved_sofa_001 uses curved_sofa_001_asset")
        );
        parse_scene_bsn(&layout.bsn, &assets).expect("grounded BSN parses");
    }

    #[test]
    fn grounded_scene_layout_uses_explicit_repeated_instances() {
        let manifest = SceneObjectManifest {
            source_scene_path: "/tmp/scene.jpg".to_string(),
            scene_calibration: None,
            objects: vec![
                SceneObjectSpec {
                    id: "conference_table".to_string(),
                    label: "conference table".to_string(),
                    aliases: vec!["table".to_string()],
                    bbox: [0.35, 0.35, 0.65, 0.7],
                    instances: Vec::new(),
                    representative_instance_id: None,
                    reuse_group: None,
                    instance_count: 1,
                    object_prompt: "rectangular table".to_string(),
                    camera_hint: None,
                    rotation_hint_degrees: None,
                    target_footprint_m: None,
                },
                SceneObjectSpec {
                    id: "black_mesh_chair_group".to_string(),
                    label: "black mesh conference chair group".to_string(),
                    aliases: vec!["chair".to_string()],
                    bbox: [0.1, 0.2, 0.9, 0.85],
                    instances: vec![
                        SceneObjectInstanceSpec {
                            id: Some("left_back".to_string()),
                            bbox: [0.1, 0.2, 0.22, 0.62],
                            contact: Some([0.16, 0.62]),
                            rotation_hint_degrees: Some(-35.0),
                            facing_yaw_degrees: None,
                            side: None,
                            slot_index: None,
                            target_footprint_m: None,
                        },
                        SceneObjectInstanceSpec {
                            id: Some("right_front".to_string()),
                            bbox: [0.72, 0.46, 0.9, 0.85],
                            contact: Some([0.81, 0.85]),
                            rotation_hint_degrees: Some(42.0),
                            facing_yaw_degrees: None,
                            side: None,
                            slot_index: None,
                            target_footprint_m: None,
                        },
                    ],
                    representative_instance_id: None,
                    reuse_group: Some("black_mesh_conference_chair".to_string()),
                    instance_count: 2,
                    object_prompt: "one reusable black mesh conference chair".to_string(),
                    camera_hint: None,
                    rotation_hint_degrees: None,
                    target_footprint_m: None,
                },
            ],
        };
        let assets = vec![
            SceneAssetBinding {
                asset_id: "conference_table_asset".to_string(),
                object_id: "conference_table".to_string(),
                label: "conference table".to_string(),
                aliases: Vec::new(),
                path: None,
                cache_key: Some("table".to_string()),
                reusable: true,
                source_image_path: None,
                pipeline: Some("trellis".to_string()),
                local_aabb: Some(SceneAssetAabb {
                    min: [-0.5, 0.0, -0.5],
                    max: [0.5, 0.4, 0.5],
                }),
                canonical_frame: None,
                provenance: None,
            },
            SceneAssetBinding {
                asset_id: "black_mesh_chair_group_asset".to_string(),
                object_id: "black_mesh_chair_group".to_string(),
                label: "black mesh conference chair group".to_string(),
                aliases: Vec::new(),
                path: None,
                cache_key: Some("chair".to_string()),
                reusable: true,
                source_image_path: None,
                pipeline: Some("trellis".to_string()),
                local_aabb: Some(SceneAssetAabb {
                    min: [-0.4, 0.0, -0.4],
                    max: [0.4, 1.1, 0.4],
                }),
                canonical_frame: None,
                provenance: None,
            },
        ];

        let layout =
            grounded_scene_layout(&manifest, &assets, GroundedSceneLayoutConfig::default())
                .expect("grounded layout");

        let chairs = layout
            .placements
            .iter()
            .filter(|placement| placement.object_id == "black_mesh_chair_group")
            .collect::<Vec<_>>();
        assert_eq!(chairs.len(), 2);
        assert_eq!(chairs[0].instance_id.as_deref(), Some("left_back"));
        assert_eq!(chairs[0].source_bbox, [0.1, 0.2, 0.22, 0.62]);
        assert_eq!(chairs[0].contact_pixel, [0.16, 0.62]);
        assert_eq!(chairs[0].rotation_y_degrees, -35.0);
        assert_eq!(chairs[1].instance_id.as_deref(), Some("right_front"));
        assert_eq!(chairs[1].source_bbox, [0.72, 0.46, 0.9, 0.85]);
        assert_eq!(chairs[1].contact_pixel, [0.81, 0.85]);
        assert_eq!(chairs[1].rotation_y_degrees, 42.0);
        assert!(
            layout
                .bsn
                .contains("spawn black_mesh_chair_group_left_back")
        );
        assert!(
            layout
                .bsn
                .contains("spawn black_mesh_chair_group_right_front")
        );
    }

    #[test]
    fn grounded_scene_layout_keeps_reused_asset_instances_same_scale() {
        let manifest = SceneObjectManifest {
            source_scene_path: "/tmp/scene.jpg".to_string(),
            scene_calibration: None,
            objects: vec![SceneObjectSpec {
                id: "chair_group".to_string(),
                label: "chair group".to_string(),
                aliases: vec!["chair".to_string()],
                bbox: [0.1, 0.2, 0.9, 0.9],
                instances: vec![
                    SceneObjectInstanceSpec {
                        id: Some("near_large".to_string()),
                        bbox: [0.1, 0.55, 0.3, 0.95],
                        contact: Some([0.2, 0.95]),
                        rotation_hint_degrees: None,
                        facing_yaw_degrees: None,
                        side: None,
                        slot_index: None,
                        target_footprint_m: Some([0.9, 0.9]),
                    },
                    SceneObjectInstanceSpec {
                        id: Some("far_small".to_string()),
                        bbox: [0.6, 0.25, 0.75, 0.55],
                        contact: Some([0.675, 0.55]),
                        rotation_hint_degrees: None,
                        facing_yaw_degrees: None,
                        side: None,
                        slot_index: None,
                        target_footprint_m: Some([0.45, 0.45]),
                    },
                ],
                representative_instance_id: Some("near_large".to_string()),
                reuse_group: Some("chair".to_string()),
                instance_count: 2,
                object_prompt: "one reusable chair".to_string(),
                camera_hint: None,
                rotation_hint_degrees: None,
                target_footprint_m: None,
            }],
        };
        let assets = vec![SceneAssetBinding {
            asset_id: "chair_asset".to_string(),
            object_id: "chair_group".to_string(),
            label: "chair".to_string(),
            aliases: Vec::new(),
            path: None,
            cache_key: Some("chair".to_string()),
            reusable: true,
            source_image_path: None,
            pipeline: Some("trellis".to_string()),
            local_aabb: Some(SceneAssetAabb {
                min: [-0.5, 0.0, -0.5],
                max: [0.5, 1.0, 0.5],
            }),
            canonical_frame: None,
            provenance: None,
        }];

        let layout =
            grounded_scene_layout(&manifest, &assets, GroundedSceneLayoutConfig::default())
                .expect("grounded layout");
        let chairs = layout
            .placements
            .iter()
            .filter(|placement| placement.asset_id == "chair_asset")
            .collect::<Vec<_>>();

        assert_eq!(chairs.len(), 2);
        assert!((chairs[0].scale[0] - chairs[1].scale[0]).abs() <= 1.0e-6);
        assert!((chairs[0].translation[1] - chairs[1].translation[1]).abs() <= 1.0e-6);
    }

    #[test]
    fn grounded_scene_layout_uses_calibrated_table_slots_and_source_camera() {
        let manifest = SceneObjectManifest {
            source_scene_path: "/tmp/galaxy.jpg".to_string(),
            scene_calibration: Some(SceneCalibration {
                table_center: Some([0.50, 0.60]),
                table_axis_degrees: Some(0.0),
                table_size_m: Some([1.2, 3.4]),
                camera_yaw_degrees: Some(0.0),
                camera_pitch_degrees: Some(-30.0),
                camera_radius_m: Some(5.2),
                vertical_fov_degrees: Some(78.0),
            }),
            objects: vec![
                SceneObjectSpec {
                    id: "conference_table".to_string(),
                    label: "conference table".to_string(),
                    aliases: vec!["table".to_string()],
                    bbox: [0.35, 0.35, 0.65, 0.85],
                    instances: Vec::new(),
                    representative_instance_id: None,
                    reuse_group: None,
                    instance_count: 1,
                    object_prompt: "large rectangular conference table".to_string(),
                    camera_hint: None,
                    rotation_hint_degrees: None,
                    target_footprint_m: None,
                },
                SceneObjectSpec {
                    id: "gray_chair_group".to_string(),
                    label: "gray conference chair group".to_string(),
                    aliases: vec!["chair".to_string()],
                    bbox: [0.12, 0.30, 0.92, 0.95],
                    instances: vec![
                        SceneObjectInstanceSpec {
                            id: Some("left_01".to_string()),
                            bbox: [0.20, 0.38, 0.30, 0.72],
                            contact: Some([0.25, 0.72]),
                            rotation_hint_degrees: None,
                            facing_yaw_degrees: None,
                            side: Some(SceneInstanceSide::Left),
                            slot_index: Some(0),
                            target_footprint_m: Some([0.58, 0.62]),
                        },
                        SceneObjectInstanceSpec {
                            id: Some("right_01".to_string()),
                            bbox: [0.74, 0.42, 0.86, 0.78],
                            contact: Some([0.80, 0.78]),
                            rotation_hint_degrees: None,
                            facing_yaw_degrees: None,
                            side: Some(SceneInstanceSide::Right),
                            slot_index: Some(0),
                            target_footprint_m: Some([0.58, 0.62]),
                        },
                        SceneObjectInstanceSpec {
                            id: Some("near_01".to_string()),
                            bbox: [0.10, 0.66, 0.26, 0.98],
                            contact: Some([0.18, 0.98]),
                            rotation_hint_degrees: None,
                            facing_yaw_degrees: None,
                            side: Some(SceneInstanceSide::Near),
                            slot_index: Some(0),
                            target_footprint_m: Some([0.58, 0.62]),
                        },
                        SceneObjectInstanceSpec {
                            id: Some("far_01".to_string()),
                            bbox: [0.48, 0.22, 0.57, 0.48],
                            contact: Some([0.52, 0.48]),
                            rotation_hint_degrees: None,
                            facing_yaw_degrees: None,
                            side: Some(SceneInstanceSide::Far),
                            slot_index: Some(0),
                            target_footprint_m: Some([0.58, 0.62]),
                        },
                    ],
                    representative_instance_id: Some("right_01".to_string()),
                    reuse_group: Some("gray_conference_chair".to_string()),
                    instance_count: 4,
                    object_prompt: "one gray conference chair with mesh back".to_string(),
                    camera_hint: None,
                    rotation_hint_degrees: None,
                    target_footprint_m: Some([0.58, 0.62]),
                },
            ],
        };
        let assets = vec![
            SceneAssetBinding {
                asset_id: "conference_table_asset".to_string(),
                object_id: "conference_table".to_string(),
                label: "conference table".to_string(),
                aliases: Vec::new(),
                path: None,
                cache_key: Some("table".to_string()),
                reusable: true,
                source_image_path: None,
                pipeline: Some("trellis".to_string()),
                local_aabb: Some(SceneAssetAabb {
                    min: [-0.35, 0.0, -1.0],
                    max: [0.35, 0.32, 1.0],
                }),
                canonical_frame: Some(SceneAssetFrame {
                    yaw_offset_degrees: 0.0,
                    footprint_m: Some([1.2, 3.4]),
                }),
                provenance: None,
            },
            SceneAssetBinding {
                asset_id: "gray_chair_group_asset".to_string(),
                object_id: "gray_chair_group".to_string(),
                label: "gray conference chair".to_string(),
                aliases: vec!["chair".to_string()],
                path: None,
                cache_key: Some("chair".to_string()),
                reusable: true,
                source_image_path: None,
                pipeline: Some("trellis".to_string()),
                local_aabb: Some(SceneAssetAabb {
                    min: [-0.3, 0.0, -0.3],
                    max: [0.3, 1.0, 0.3],
                }),
                canonical_frame: Some(SceneAssetFrame {
                    yaw_offset_degrees: 0.0,
                    footprint_m: Some([0.58, 0.62]),
                }),
                provenance: None,
            },
        ];

        let layout =
            grounded_scene_layout(&manifest, &assets, GroundedSceneLayoutConfig::default())
                .expect("grounded layout");
        let table = layout
            .placements
            .iter()
            .find(|placement| placement.object_id == "conference_table")
            .unwrap();
        let chairs = layout
            .placements
            .iter()
            .filter(|placement| placement.object_id == "gray_chair_group")
            .collect::<Vec<_>>();
        assert_eq!(chairs.len(), 4);
        assert_eq!(table.ground_point, [0.0, 0.0, 0.0]);
        assert_eq!(table.target_footprint_m, [1.2, 3.4]);
        assert!(table.scale[0] > chairs[0].scale[0]);
        let left = chairs
            .iter()
            .find(|placement| placement.instance_id.as_deref() == Some("left_01"))
            .unwrap();
        let right = chairs
            .iter()
            .find(|placement| placement.instance_id.as_deref() == Some("right_01"))
            .unwrap();
        let near = chairs
            .iter()
            .find(|placement| placement.instance_id.as_deref() == Some("near_01"))
            .unwrap();
        let far = chairs
            .iter()
            .find(|placement| placement.instance_id.as_deref() == Some("far_01"))
            .unwrap();
        assert!(left.ground_point[0] < -0.9);
        assert!(right.ground_point[0] > 0.9);
        assert!(near.ground_point[2] > 1.9);
        assert!(far.ground_point[2] < -1.9);
        assert!((left.rotation_y_degrees - 90.0).abs() < 1.0e-3);
        assert!((right.rotation_y_degrees + 90.0).abs() < 1.0e-3);
        assert!(near.rotation_y_degrees.abs() >= 179.0);
        assert!(far.rotation_y_degrees.abs() < 1.0e-3);
        assert!(layout.camera.translation[1] > 2.0);
        assert_eq!(layout.camera.pitch, Some(30.0));
        assert_eq!(layout.camera.vertical_fov_degrees, Some(78.0));
        assert!(layout.bsn.contains("vertical_fov 78.0"));
        parse_scene_bsn(&layout.bsn, &assets).expect("calibrated BSN parses");
    }

    #[test]
    fn metric_frame_maps_source_sides_through_camera_yaw() {
        let frame = MetricSceneFrame {
            table_axis_degrees: 0.0,
            table_size_m: [1.2, 3.4],
            seating_clearance_m: 0.18,
            camera_yaw_degrees: Some(180.0),
            camera_pitch_degrees: Some(24.0),
            camera_radius_m: Some(4.2),
            vertical_fov_degrees: Some(74.0),
        };

        let left = frame.side_point(SceneInstanceSide::Left, 0, 1, [0.58, 0.62]);
        let right = frame.side_point(SceneInstanceSide::Right, 0, 1, [0.58, 0.62]);
        let near = frame.side_point(SceneInstanceSide::Near, 0, 1, [0.58, 0.62]);
        let far = frame.side_point(SceneInstanceSide::Far, 0, 1, [0.58, 0.62]);

        assert!(left[0] > 0.9);
        assert!(right[0] < -0.9);
        assert!(near[2] < -1.9);
        assert!(far[2] > 1.9);
    }

    #[test]
    fn bsn_yaw_convention_faces_plus_z_at_zero_degrees() {
        let from = [0.0, 0.0, 0.0];

        assert!(
            (bsn_yaw_toward_point_degrees(from, [0.0, 0.0, 1.0]).unwrap() - 0.0).abs() < 1.0e-6
        );
        assert!(
            (bsn_yaw_toward_point_degrees(from, [1.0, 0.0, 0.0]).unwrap() - 90.0).abs() < 1.0e-6
        );
        assert!(
            (bsn_yaw_toward_point_degrees(from, [-1.0, 0.0, 0.0]).unwrap() + 90.0).abs() < 1.0e-6
        );
        assert!(
            bsn_yaw_toward_point_degrees(from, [0.0, 0.0, -1.0])
                .unwrap()
                .abs()
                >= 179.999
        );
        assert!(bsn_yaw_toward_point_degrees(from, from).is_none());
    }

    #[test]
    fn representative_crop_bbox_prefers_requested_single_instance() {
        let object = SceneObjectSpec {
            id: "chair_group".to_string(),
            label: "chair group".to_string(),
            aliases: vec!["chair".to_string()],
            bbox: [0.1, 0.2, 0.9, 0.9],
            instances: vec![
                SceneObjectInstanceSpec {
                    id: Some("left".to_string()),
                    bbox: [0.1, 0.3, 0.25, 0.75],
                    contact: Some([0.18, 0.75]),
                    rotation_hint_degrees: None,
                    facing_yaw_degrees: None,
                    side: Some(SceneInstanceSide::Left),
                    slot_index: Some(0),
                    target_footprint_m: None,
                },
                SceneObjectInstanceSpec {
                    id: Some("right".to_string()),
                    bbox: [0.72, 0.42, 0.9, 0.86],
                    contact: Some([0.81, 0.86]),
                    rotation_hint_degrees: None,
                    facing_yaw_degrees: None,
                    side: Some(SceneInstanceSide::Right),
                    slot_index: Some(0),
                    target_footprint_m: None,
                },
            ],
            representative_instance_id: Some("right".to_string()),
            reuse_group: Some("chair".to_string()),
            instance_count: 2,
            object_prompt: "one chair".to_string(),
            camera_hint: None,
            rotation_hint_degrees: None,
            target_footprint_m: None,
        };

        assert_eq!(representative_crop_bbox(&object), [0.72, 0.42, 0.9, 0.86]);
    }

    #[test]
    fn grounded_scene_layout_re_ranks_global_slots_per_table_side() {
        let manifest = SceneObjectManifest {
            source_scene_path: "/tmp/scene.jpg".to_string(),
            scene_calibration: Some(SceneCalibration {
                table_center: Some([0.5, 0.6]),
                table_axis_degrees: Some(0.0),
                table_size_m: Some([3.2, 1.25]),
                camera_yaw_degrees: Some(180.0),
                camera_pitch_degrees: Some(24.0),
                camera_radius_m: Some(4.2),
                vertical_fov_degrees: Some(74.0),
            }),
            objects: vec![
                SceneObjectSpec {
                    id: "table".to_string(),
                    label: "conference table".to_string(),
                    aliases: vec!["table".to_string()],
                    bbox: [0.3, 0.4, 0.7, 0.9],
                    instances: Vec::new(),
                    representative_instance_id: None,
                    reuse_group: None,
                    instance_count: 1,
                    object_prompt: "conference table".to_string(),
                    camera_hint: None,
                    rotation_hint_degrees: None,
                    target_footprint_m: Some([3.2, 1.25]),
                },
                SceneObjectSpec {
                    id: "chair_group".to_string(),
                    label: "chair group".to_string(),
                    aliases: vec!["chair".to_string()],
                    bbox: [0.2, 0.2, 0.8, 0.8],
                    instances: vec![
                        SceneObjectInstanceSpec {
                            id: Some("far_left_global_02".to_string()),
                            bbox: [0.38, 0.45, 0.46, 0.7],
                            contact: Some([0.42, 0.7]),
                            rotation_hint_degrees: Some(135.0),
                            facing_yaw_degrees: Some(135.0),
                            side: Some(SceneInstanceSide::Far),
                            slot_index: Some(2),
                            target_footprint_m: Some([0.58, 0.62]),
                        },
                        SceneObjectInstanceSpec {
                            id: Some("far_right_global_04".to_string()),
                            bbox: [0.58, 0.45, 0.66, 0.7],
                            contact: Some([0.62, 0.7]),
                            rotation_hint_degrees: Some(-135.0),
                            facing_yaw_degrees: Some(-135.0),
                            side: Some(SceneInstanceSide::Far),
                            slot_index: Some(4),
                            target_footprint_m: Some([0.58, 0.62]),
                        },
                    ],
                    representative_instance_id: Some("far_right_global_04".to_string()),
                    reuse_group: Some("chair".to_string()),
                    instance_count: 2,
                    object_prompt: "one conference chair".to_string(),
                    camera_hint: None,
                    rotation_hint_degrees: None,
                    target_footprint_m: Some([0.58, 0.62]),
                },
            ],
        };
        let assets = vec![
            SceneAssetBinding {
                asset_id: "table_asset".to_string(),
                object_id: "table".to_string(),
                label: "conference table".to_string(),
                aliases: Vec::new(),
                path: None,
                cache_key: Some("table".to_string()),
                reusable: true,
                source_image_path: None,
                pipeline: Some("trellis".to_string()),
                local_aabb: Some(SceneAssetAabb {
                    min: [-0.5, 0.0, -0.2],
                    max: [0.5, 0.2, 0.2],
                }),
                canonical_frame: Some(SceneAssetFrame {
                    yaw_offset_degrees: 90.0,
                    footprint_m: Some([3.2, 1.25]),
                }),
                provenance: None,
            },
            SceneAssetBinding {
                asset_id: "chair_asset".to_string(),
                object_id: "chair_group".to_string(),
                label: "chair".to_string(),
                aliases: Vec::new(),
                path: None,
                cache_key: Some("chair".to_string()),
                reusable: true,
                source_image_path: None,
                pipeline: Some("trellis".to_string()),
                local_aabb: Some(SceneAssetAabb {
                    min: [-0.35, 0.0, -0.35],
                    max: [0.35, 1.0, 0.35],
                }),
                canonical_frame: Some(SceneAssetFrame {
                    yaw_offset_degrees: 0.0,
                    footprint_m: Some([0.58, 0.62]),
                }),
                provenance: None,
            },
        ];

        let layout =
            grounded_scene_layout(&manifest, &assets, GroundedSceneLayoutConfig::default())
                .expect("grounded layout");
        let chairs = layout
            .placements
            .iter()
            .filter(|placement| placement.object_id == "chair_group")
            .collect::<Vec<_>>();

        assert_eq!(chairs.len(), 2);
        assert!(
            (chairs[0].ground_point[0] - chairs[1].ground_point[0]).abs() > 0.5,
            "global slot indices should be converted to unique side-local positions: {chairs:?}"
        );
        assert!((chairs[0].ground_point[2] - chairs[1].ground_point[2]).abs() < 1.0e-4);
    }

    #[test]
    fn sofa_shape_score_rejects_source_aspect_drift() {
        let object = SceneObjectSpec {
            id: "curved_sofa_001".to_string(),
            label: "curved sofa".to_string(),
            aliases: Vec::new(),
            bbox: [0.0, 0.155, 1.0, 1.0],
            instances: Vec::new(),
            representative_instance_id: None,
            reuse_group: None,
            instance_count: 1,
            object_prompt: "tan sectional sofa".to_string(),
            camera_hint: None,
            rotation_hint_degrees: None,
            target_footprint_m: None,
        };
        let matte = ObjectImageMatteStats {
            alpha_coverage: 0.36,
            alpha_bbox: Some([65, 187, 971, 897]),
        };

        let score = generated_shape_consistency_score(&object, &matte, 2.0);

        assert_eq!(score, 0.0);
    }

    #[test]
    fn sofa_shape_score_keeps_wide_open_sectional_selectable() {
        let object = SceneObjectSpec {
            id: "tan_open_sectional_sofa_001".to_string(),
            label: "tan open sectional sofa".to_string(),
            aliases: Vec::new(),
            bbox: [0.0, 0.155, 1.0, 1.0],
            instances: Vec::new(),
            representative_instance_id: None,
            reuse_group: None,
            instance_count: 1,
            object_prompt: "wide low tan open sectional with gentle right bend".to_string(),
            camera_hint: None,
            rotation_hint_degrees: None,
            target_footprint_m: None,
        };
        let matte = ObjectImageMatteStats {
            alpha_coverage: 0.29,
            alpha_bbox: Some([16, 173, 1005, 863]),
        };

        let score = generated_shape_consistency_score(&object, &matte, 1.779);

        assert!(score >= 0.45, "score {score}");
    }

    #[test]
    fn object_image_prompt_includes_style_reference_and_exclusion_rules() {
        let object = SceneObjectSpec {
            id: "sofa_curved".to_string(),
            label: "curved sofa".to_string(),
            aliases: vec![],
            bbox: [0.2, 0.3, 0.9, 0.95],
            instances: Vec::new(),
            representative_instance_id: None,
            reuse_group: None,
            instance_count: 1,
            object_prompt: "A tan curved upholstered sectional sofa.".to_string(),
            camera_hint: Some("high oblique".to_string()),
            rotation_hint_degrees: Some(35.0),
            target_footprint_m: None,
        };
        let prompt = object_image_prompt(Path::new("docs/input_chair.jpg"), &object);
        assert!(prompt.contains("docs/input_chair.jpg"));
        assert!(prompt.contains("Source-preserving edit requirement"));
        assert!(prompt.contains("clean isolated"));
        assert!(prompt.contains("Do not include the room"));
        assert!(prompt.contains("curved sofa"));
        assert!(prompt.contains("solid matte cobalt-blue background"));
        assert!(prompt.contains("ring-like"));
        assert!(prompt.contains("visible straight run mostly straight"));
        assert!(prompt.contains("Target yaw/rotation hint: 35.0 degrees"));
    }

    #[test]
    fn object_image_prompt_protects_thin_white_table_geometry() {
        let object = SceneObjectSpec {
            id: "table".to_string(),
            label: "white coffee table".to_string(),
            aliases: vec!["low white table".to_string()],
            bbox: [0.3, 0.3, 0.7, 0.6],
            instances: Vec::new(),
            representative_instance_id: None,
            reuse_group: None,
            instance_count: 1,
            object_prompt: "A glossy white rectangular coffee table with slim white metal frame."
                .to_string(),
            camera_hint: None,
            rotation_hint_degrees: None,
            target_footprint_m: None,
        };
        let prompt = object_image_prompt(Path::new("docs/input_chair.jpg"), &object);
        assert!(prompt.contains("solid matte cobalt-blue background"));
        assert!(prompt.contains("Do not omit thin legs"));
        assert!(prompt.contains("Do not merge the tabletop into the background"));
        assert!(prompt.contains("no floor plane"));
    }

    #[test]
    fn object_image_prompt_prioritizes_chair_geometry_over_table_context() {
        let object = SceneObjectSpec {
            id: "black_mesh_conference_chair_group".to_string(),
            label: "black mesh conference chair".to_string(),
            aliases: vec!["chair".to_string()],
            bbox: [0.1, 0.2, 0.3, 0.7],
            instances: Vec::new(),
            representative_instance_id: None,
            reuse_group: Some("black_mesh_conference_chair".to_string()),
            instance_count: 6,
            object_prompt:
                "one reusable chair observed around the conference table; scale smaller than tabletop"
                    .to_string(),
            camera_hint: None,
            rotation_hint_degrees: None,
            target_footprint_m: None,
        };

        let prompt = object_image_prompt(Path::new("docs/input_chair.jpg"), &object);

        assert!(prompt.contains("preserve one complete high-back segmented chair"));
        assert!(prompt.contains("Do not generate multiple chairs"));
        assert!(!prompt.contains("preserve a flat rectangular tabletop"));
    }

    #[test]
    fn generated_image_suitability_penalizes_low_contrast_background() {
        let mut image = image::RgbImage::from_pixel(128, 128, image::Rgb([222, 222, 222]));
        for y in 50..72 {
            for x in 20..108 {
                image.put_pixel(x, y, image::Rgb([234, 234, 232]));
            }
        }
        for x in [24, 104] {
            for y in 72..105 {
                image.put_pixel(x, y, image::Rgb([236, 236, 234]));
            }
        }
        let score = score_generated_object_rgb(&image);
        assert!(
            score.score < 0.35,
            "low-contrast white-on-gray image should be a poor TRELLIS/RMBG candidate: {score:?}"
        );
    }

    #[test]
    fn generated_image_suitability_accepts_high_contrast_object() {
        let mut image = image::RgbImage::from_pixel(128, 128, image::Rgb([31, 95, 214]));
        for y in 32..96 {
            for x in 24..104 {
                image.put_pixel(x, y, image::Rgb([238, 238, 232]));
            }
        }
        let score = score_generated_object_rgb(&image);
        assert!(
            score.score > 0.90,
            "high-contrast object/matte image should rank well: {score:?}"
        );
    }

    #[test]
    fn object_image_generation_policy_retries_until_candidate_passes_guardrail() {
        let dir = tempfile::tempdir().unwrap();
        let source_path = dir.path().join("scene.png");
        image::RgbImage::from_pixel(512, 256, image::Rgb([64, 64, 64]))
            .save(&source_path)
            .unwrap();
        let provider = RetryImageProvider::new(vec![
            low_contrast_candidate_png(),
            high_contrast_candidate_png(),
        ]);
        let config = SceneBuildConfig {
            source_scene_path: source_path.clone(),
            object_reference_image_path: source_path.clone(),
            output_dir: dir.path().join("run"),
            candidate_count: 1,
            quality_profile: SceneQualityProfile::Draft,
            reasoning_model: "test-reasoning".to_string(),
            image_model: "test-image".to_string(),
            allow_catalog_reuse: false,
        };
        let pipeline = ScenePipeline::new(config, provider);
        let request = ObjectImageRequest {
            object: SceneObjectSpec {
                id: "green_chair".to_string(),
                label: "green chair".to_string(),
                aliases: Vec::new(),
                bbox: [0.2, 0.2, 0.4, 0.8],
                instances: Vec::new(),
                representative_instance_id: None,
                reuse_group: None,
                instance_count: 1,
                object_prompt: "dark green padded chair".to_string(),
                camera_hint: None,
                rotation_hint_degrees: None,
                target_footprint_m: None,
            },
            source_scene_path: source_path.display().to_string(),
            source_crop_path: source_path.display().to_string(),
            object_reference_image_path: source_path.display().to_string(),
            prompt: "generate chair".to_string(),
            candidate_count: 1,
            size: "1024x1024".to_string(),
            quality: "medium".to_string(),
        };

        let report = pipeline
            .generate_object_candidates_with_policy(
                &[request],
                ObjectImageGenerationPolicy {
                    min_score: 0.80,
                    max_attempts_per_object: 2,
                    candidates_per_attempt: 1,
                },
            )
            .unwrap();

        assert_eq!(report.attempts.len(), 2);
        assert!(!report.attempts[0].accepted);
        assert!(report.attempts[1].accepted);
        assert_eq!(report.candidates.len(), 2);
        assert_eq!(report.selected_candidates.len(), 1);
        assert!(report.rejected_objects.is_empty());
        assert_eq!(report.selected_candidates[0].candidate_index, 1);
        assert!(
            dir.path()
                .join("run/objects/generated/green_chair_candidate_0.png")
                .exists()
        );
        assert!(
            dir.path()
                .join("run/objects/generated/green_chair_candidate_1.png")
                .exists()
        );
    }

    #[test]
    fn generated_candidate_matte_writes_transparent_background() {
        let mut image = image::RgbImage::from_pixel(128, 128, image::Rgb([31, 95, 214]));
        for y in 32..96 {
            for x in 24..104 {
                image.put_pixel(x, y, image::Rgb([238, 238, 232]));
            }
        }
        let suitability = score_generated_object_rgb(&image);
        let (matted, stats) = matte_generated_object_rgb(&image, suitability);
        assert_eq!(matted.get_pixel(0, 0).0[3], 0);
        assert_eq!(matted.get_pixel(64, 64).0[3], 255);
        assert!(
            (0.20..0.50).contains(&stats.alpha_coverage),
            "matte alpha should cover the object, not the whole frame: {stats:?}"
        );
        assert_eq!(stats.alpha_bbox, Some([24, 32, 104, 96]));
    }

    #[test]
    fn schemas_are_strict_objects() {
        assert_eq!(
            object_manifest_schema()["additionalProperties"],
            json!(false)
        );
        assert_eq!(scene_bsn_schema()["additionalProperties"], json!(false));
    }

    #[test]
    fn image_data_url_uses_source_pixels_and_mime_type() {
        let dir = tempfile::tempdir().unwrap();
        let image_path = dir.path().join("input.jpg");
        fs::write(&image_path, [1u8, 2, 3, 4]).unwrap();
        let data_url = image_data_url(&image_path).unwrap();
        assert!(data_url.starts_with("data:image/jpeg;base64,"));
        assert!(data_url.ends_with("AQIDBA=="));
    }

    #[test]
    fn resize_image_for_api_bounds_large_inputs() {
        let dir = tempfile::tempdir().unwrap();
        let input_path = dir.path().join("large.png");
        let output_path = dir.path().join("large_1024.jpg");
        let image = image::RgbImage::from_pixel(2048, 1024, image::Rgb([128, 64, 32]));
        image.save(&input_path).unwrap();
        resize_image_for_api(&input_path, &output_path).unwrap();
        let resized = image::open(&output_path).unwrap();
        assert_eq!(resized.width(), 1024);
        assert_eq!(resized.height(), 512);
    }
}

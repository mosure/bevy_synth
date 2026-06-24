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
    pub objects: Vec<SceneObjectSpec>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct SceneObjectSpec {
    pub id: String,
    pub label: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    pub bbox: [f32; 4],
    #[serde(default)]
    pub reuse_group: Option<String>,
    #[serde(default = "default_instance_count")]
    pub instance_count: usize,
    pub object_prompt: String,
    #[serde(default)]
    pub camera_hint: Option<String>,
    #[serde(default)]
    pub rotation_hint_degrees: Option<f32>,
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

#[derive(Clone, Debug, Deserialize, Serialize)]
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
    #[serde(default)]
    pub provenance: Option<SceneAssetProvenance>,
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
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ScenePreparation {
    pub run_id: String,
    pub output_dir: String,
    pub source_scene_path: String,
    pub object_reference_image_path: String,
    pub object_manifest_schema: Value,
    pub scene_bsn_schema: Value,
    pub object_image_style_prompt: String,
}

pub trait SceneAiProvider {
    fn plan_objects(&self, request: &SceneReasoningRequest) -> SceneResult<SceneObjectManifest>;
    fn generate_object_images(&self, request: &ObjectImageRequest) -> SceneResult<Vec<Vec<u8>>>;
    fn plan_scene_bsn(&self, request: &SceneBsnRequest) -> SceneResult<String>;
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
            let crop_path = crop_scene_object(
                &self.config.source_scene_path,
                object,
                &crop_dir.join(format!("{}_crop_1024.jpg", object.id)),
            )?;
            requests.push(ObjectImageRequest {
                object: object.clone(),
                source_scene_path: api_scene_path.display().to_string(),
                source_crop_path: crop_path.display().to_string(),
                object_reference_image_path: api_reference_path.display().to_string(),
                prompt: object_image_prompt(&self.config.object_reference_image_path, object),
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
            let started = unix_ms();
            write_metric(
                &self.config.output_dir,
                "openai.object_image.start",
                json!({
                    "object_id": request.object.id,
                    "candidate_count": request.candidate_count,
                    "quality": request.quality,
                    "size": request.size,
                    "source_scene_path": request.source_scene_path,
                    "source_crop_path": request.source_crop_path,
                    "object_reference_image_path": request.object_reference_image_path,
                }),
            )?;
            eprintln!(
                "burn_synth_scene: generating {} object image candidate(s) for {}",
                request.candidate_count, request.object.id
            );
            let images = self.provider.generate_object_images(request)?;
            write_metric(
                &self.config.output_dir,
                "openai.object_image",
                json!({
                    "object_id": request.object.id,
                    "candidate_count": images.len(),
                    "elapsed_ms": unix_ms().saturating_sub(started),
                    "quality": request.quality,
                    "size": request.size,
                }),
            )?;
            for (candidate_index, bytes) in images.into_iter().enumerate() {
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
                fs::write(&raw_image_path, &bytes)?;
                matted.save(&image_path)?;
                write_metric(
                    &self.config.output_dir,
                    "openai.object_image.candidate",
                    json!({
                        "object_id": request.object.id,
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
                    }),
                )?;
                candidates.push(ObjectImageCandidate {
                    object_id: request.object.id.clone(),
                    candidate_index,
                    image_path: image_path.display().to_string(),
                    raw_image_path: Some(raw_image_path.display().to_string()),
                    prompt_hash: stable_hash_hex(&request.prompt),
                    score: suitability.score,
                    provider_request_id: None,
                });
            }
        }
        Ok(candidates)
    }

    pub fn plan_scene_bsn(
        &self,
        manifest: SceneObjectManifest,
        asset_bindings: Vec<SceneAssetBinding>,
    ) -> SceneResult<String> {
        self.provider.plan_scene_bsn(&SceneBsnRequest {
            source_scene_path: self.config.source_scene_path.clone(),
            object_manifest: manifest.clone(),
            asset_bindings: asset_bindings.clone(),
            prompt: scene_bsn_prompt(&manifest, &asset_bindings),
        })
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
}

impl OpenAiSceneProvider {
    pub fn from_env(mut config: OpenAiProviderConfig) -> SceneResult<Self> {
        config.api_key = env::var("OPENAI_API_KEY").map_err(|_| {
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
        Ok(Self { config, client })
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
        extract_structured_output(value)
    }
}

impl SceneAiProvider for OpenAiSceneProvider {
    fn plan_objects(&self, request: &SceneReasoningRequest) -> SceneResult<SceneObjectManifest> {
        let mut value = self.post_responses_schema(
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
}

pub fn object_manifest_prompt(
    scene_path: &Path,
    reference_path: &Path,
    allow_catalog_reuse: bool,
) -> String {
    format!(
        "Analyze the source scene image at `{}` and produce a strict object manifest for 3D reconstruction. \
Use the reference image `{}` as the expected clean isolated object-image style: single object, centered, full visible silhouette, neutral background, 3/4 camera. \
For the furniture demo prefer reusable object groups: one curved sofa, one coffee table, one reusable chair group with instance_count for repeated chairs, and no generated cube/proxy furniture. \
Normalized bboxes must be [x_min,y_min,x_max,y_max]. allow_catalog_reuse={}.",
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
        "{}\n{}\nReference style image: `{}`.\nObject id: {}.\nObject label: {}.\nSource crop bbox: [{:.4},{:.4},{:.4},{:.4}].\nCamera/orientation: {} {}\nGenerate a clean isolated product-style image of exactly this object for 3D reconstruction. Preserve the source object geometry, material, color, scale proportions, and camera angle. Do not include the room, rug, table clutter, extra chairs, people, walls, text, shadows cast by the original scene, or background furniture. Do not replace the object with a proxy, cube, simplified block, alternate furniture type, or stylized approximation. Full object visible, no truncation. {}\n{}",
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
    "Input: source scene image + source object crop + docs/input_chair.jpg style reference. Output: clean centered isolated object image suitable for RMBG and TRELLIS, on a flat high-contrast matte background with crisp object/background separation. Preserve object geometry/material/camera; remove scene context.".to_string()
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
    let descriptor = format!(
        "{} {} {}",
        object.label,
        object.aliases.join(" "),
        object.object_prompt
    )
    .to_ascii_lowercase();
    if descriptor.contains("sofa")
        || descriptor.contains("couch")
        || descriptor.contains("sectional")
    {
        "Geometry constraints: preserve the observed sectional sofa plan shape as a smooth open arc/curved L segment, not a closed ring, spiral, horseshoe, or exaggerated bend. Keep straight cushion runs straight, rounded ends rounded, seat thickness uniform, back panels vertical, and legs small/dark where visible."
    } else if descriptor.contains("table") {
        "Geometry constraints: preserve a flat rectangular tabletop with real thickness, four straight vertical legs and/or a slim rectangular metal frame. Do not merge the tabletop into the background. Do not omit thin legs, rails, or feet. Keep all frame lines straight and parallel."
    } else if descriptor.contains("chair") {
        "Geometry constraints: preserve one complete high-back segmented chair with stacked horizontal back cushions, padded seat, two metal loop arms, central pedestal, and five-star base. Do not generate multiple chairs in one image."
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

pub fn object_manifest_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["source_scene_path", "objects"],
        "properties": {
            "source_scene_path": { "type": "string" },
            "objects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["id", "label", "aliases", "bbox", "reuse_group", "instance_count", "object_prompt", "camera_hint", "rotation_hint_degrees"],
                    "properties": {
                        "id": { "type": "string" },
                        "label": { "type": "string" },
                        "aliases": { "type": "array", "items": { "type": "string" } },
                        "bbox": { "type": "array", "items": { "type": "number" }, "minItems": 4, "maxItems": 4 },
                        "reuse_group": { "type": ["string", "null"] },
                        "instance_count": { "type": "integer" },
                        "object_prompt": { "type": "string" },
                        "camera_hint": { "type": ["string", "null"] },
                        "rotation_hint_degrees": { "type": ["number", "null"] }
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
    let mut index = 5;
    while index + 1 < tokens.len() {
        match tokens[index].as_str() {
            "yaw" => yaw = Some(parse_f32_token(&tokens[index + 1], line)?),
            "pitch" => pitch = Some(parse_f32_token(&tokens[index + 1], line)?),
            "radius" => radius = Some(parse_f32_token(&tokens[index + 1], line)?),
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
            objects: vec![],
        };
        let prompt = scene_bsn_prompt(&manifest, &[chair_asset()]);
        assert!(prompt.contains("Every statement must be on exactly one line"));
        assert!(prompt.contains("asset <asset_id> = \"generated:<asset_id>\";"));
        assert!(prompt.contains("spawn <entity_id> uses <asset_id>"));
    }

    #[test]
    fn object_image_prompt_includes_style_reference_and_exclusion_rules() {
        let object = SceneObjectSpec {
            id: "sofa_curved".to_string(),
            label: "curved sofa".to_string(),
            aliases: vec![],
            bbox: [0.2, 0.3, 0.9, 0.95],
            reuse_group: None,
            instance_count: 1,
            object_prompt: "A tan curved upholstered sectional sofa.".to_string(),
            camera_hint: Some("high oblique".to_string()),
            rotation_hint_degrees: Some(35.0),
        };
        let prompt = object_image_prompt(Path::new("docs/input_chair.jpg"), &object);
        assert!(prompt.contains("docs/input_chair.jpg"));
        assert!(prompt.contains("clean isolated"));
        assert!(prompt.contains("Do not include the room"));
        assert!(prompt.contains("curved sofa"));
        assert!(prompt.contains("solid matte cobalt-blue background"));
        assert!(prompt.contains("not a closed ring"));
        assert!(prompt.contains("straight cushion runs straight"));
        assert!(prompt.contains("Target yaw/rotation hint: 35.0 degrees"));
    }

    #[test]
    fn object_image_prompt_protects_thin_white_table_geometry() {
        let object = SceneObjectSpec {
            id: "table".to_string(),
            label: "white coffee table".to_string(),
            aliases: vec!["low white table".to_string()],
            bbox: [0.3, 0.3, 0.7, 0.6],
            reuse_group: None,
            instance_count: 1,
            object_prompt: "A glossy white rectangular coffee table with slim white metal frame."
                .to_string(),
            camera_hint: None,
            rotation_hint_degrees: None,
        };
        let prompt = object_image_prompt(Path::new("docs/input_chair.jpg"), &object);
        assert!(prompt.contains("solid matte cobalt-blue background"));
        assert!(prompt.contains("Do not omit thin legs"));
        assert!(prompt.contains("Do not merge the tabletop into the background"));
        assert!(prompt.contains("no floor plane"));
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

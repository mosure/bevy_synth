use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use reqwest::blocking::multipart::{Form, Part};
use serde_json::{Value, json};

use crate::bsn::{extract_structured_output, image_data_url, image_mime_type, redact_openai_value};
use crate::*;

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
    request_log: Mutex<Vec<Value>>,
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
            request_log: Mutex::new(Vec::new()),
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
        let usage = value.get("usage").cloned().unwrap_or(Value::Null);
        self.record_request(json!({
            "operation": operation,
            "api": "responses",
            "requested_model": self.config.reasoning_model,
            "response_id": value.get("id").and_then(Value::as_str),
            "response_model": response_model,
            "image_count": image_paths.len(),
            "usage": usage,
        }));
        extract_structured_output(value)
    }

    fn record_request(&self, value: Value) {
        if let Ok(mut request_log) = self.request_log.lock() {
            request_log.push(value);
        }
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

    fn generate_object_images_once(
        &self,
        request: &ObjectImageRequest,
    ) -> SceneResult<Vec<Vec<u8>>> {
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
            request.source_crop_path.as_str(),
            request.source_scene_path.as_str(),
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
        let body = response
            .text()
            .map_err(|err| SceneError::Http(format!("read image response body: {err}")))?;
        let value: Value = serde_json::from_str(&body).map_err(|err| {
            SceneError::Http(format!(
                "decode image response body: {err}; body_prefix={}",
                body.chars().take(256).collect::<String>()
            ))
        })?;
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
        let usage = value.get("usage").cloned().unwrap_or(Value::Null);
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
            "usage": usage,
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
        const MAX_ATTEMPTS: usize = 3;
        let mut last_error = None;
        for attempt in 0..MAX_ATTEMPTS {
            match self.generate_object_images_once(request) {
                Ok(images) => return Ok(images),
                Err(err) if attempt + 1 < MAX_ATTEMPTS && image_error_is_retryable(&err) => {
                    eprintln!(
                        "burn_synth_scene: generate_object_images retry {}/{} for {} after {}",
                        attempt + 2,
                        MAX_ATTEMPTS,
                        request.object.id,
                        err
                    );
                    last_error = Some(err);
                    std::thread::sleep(Duration::from_millis(500 * (attempt as u64 + 1)));
                }
                Err(err) => return Err(err),
            }
        }
        Err(last_error
            .unwrap_or_else(|| SceneError::Http("image generation retry exhausted".to_string())))
    }

    fn plan_scene_bsn(&self, request: &SceneBsnRequest) -> SceneResult<String> {
        let value = self.post_responses_schema(
            "plan_scene_bsn",
            &request.prompt,
            scene_bsn_schema(),
            std::slice::from_ref(&request.source_scene_path),
        )?;
        let bsn = value
            .get("bsn")
            .and_then(Value::as_str)
            .ok_or_else(|| SceneError::Provider("scene plan missing bsn field".to_string()))?;
        let _ = parse_scene_bsn(bsn, &request.asset_bindings)?;
        Ok(bsn.to_string())
    }

    fn select_rotation_candidates(
        &self,
        request: &SceneRotationSelectionRequest,
    ) -> SceneResult<SceneRotationSelectionResponse> {
        let value = self.post_responses_schema(
            "select_rotation_candidates",
            &request.prompt,
            rotation_selection_schema(),
            &request.image_paths,
        )?;
        serde_json::from_value(value).map_err(|err| SceneError::Provider(err.to_string()))
    }

    fn provider_metadata(&self) -> Value {
        let requests = self
            .request_log
            .lock()
            .map(|request_log| request_log.clone())
            .unwrap_or_default();
        json!({
            "provider": "openai",
            "base_url": self.config.base_url,
            "project_id_set": self.config.project_id.is_some(),
            "requested_reasoning_model": self.config.reasoning_model,
            "requested_image_model": self.config.image_model,
            "requests": requests,
        })
    }
}

pub(crate) fn image_error_is_retryable(err: &SceneError) -> bool {
    match err {
        SceneError::Http(message) => {
            let message = message.trim_start();
            !message.starts_with("status 4")
        }
        _ => false,
    }
}

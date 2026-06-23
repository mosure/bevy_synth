#![recursion_limit = "256"]

mod scene_layout;

use std::fs;
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use burn_synth::{
    AssetBatchItem, AssetBatchRequest, ForegroundRequest, ImageSource, Mesh, ModelSelection,
    RuntimeBatchPolicy, RuntimeConfig, SynthRuntime, SynthesisAsset, write_glb_mesh,
};
use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use scene_layout::{
    SceneComposeArgs, SceneComposePlan, SceneValidateArgs, compose_scene_layout,
    validate_scene_layout,
};

const DEFAULT_PROTOCOL_VERSION: &str = "2025-06-18";
static NEXT_SCENE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ForegroundModel {
    Rmbg14,
    Rmbg2,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SynthesisModel {
    Triposg,
    Trellis,
    Triposplat,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InferenceBackend {
    Cpu,
    Wgpu,
    Cuda,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TrellisQuality {
    Low,
    Medium,
    High,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QualityPreset {
    Fast,
    Balanced,
    Full,
}

#[derive(Clone, Copy, Debug)]
struct QualityDefaults {
    num_steps: usize,
    num_tokens: usize,
    guidance_scale: f32,
    flash_octree_depth: usize,
    flash_min_resolution: usize,
    flash_mini_grid_num: usize,
    flash_num_chunks: usize,
}

impl QualityPreset {
    fn defaults(self) -> QualityDefaults {
        match self {
            Self::Fast => QualityDefaults {
                num_steps: 12,
                num_tokens: 512,
                guidance_scale: 7.0,
                flash_octree_depth: 7,
                flash_min_resolution: 31,
                flash_mini_grid_num: 2,
                flash_num_chunks: 4096,
            },
            Self::Balanced => QualityDefaults {
                num_steps: 20,
                num_tokens: 1024,
                guidance_scale: 7.0,
                flash_octree_depth: 8,
                flash_min_resolution: 31,
                flash_mini_grid_num: 4,
                flash_num_chunks: 8192,
            },
            Self::Full => QualityDefaults {
                num_steps: 50,
                num_tokens: 2048,
                guidance_scale: 7.0,
                flash_octree_depth: 9,
                flash_min_resolution: 63,
                flash_mini_grid_num: 4,
                flash_num_chunks: 10_000,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MeshOutputFormat {
    Obj,
    Gltf,
    Glb,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AssetOutputFormat {
    Auto,
    Glb,
    Splat,
    Ply,
}

#[derive(Parser, Debug, Clone)]
#[command(
    name = "burn_synth_mcp",
    version,
    about = "burn_synth MCP stdio server"
)]
pub struct ServerArgs {
    #[arg(long, value_enum, default_value_t = ForegroundModel::Rmbg2)]
    pub rmbg_model: ForegroundModel,

    #[arg(
        long,
        value_enum,
        value_delimiter = ',',
        default_values_t = [SynthesisModel::Triposg]
    )]
    /// Synthesis backends, ordered by preference (first is preferred).
    pub synthesis_models: Vec<SynthesisModel>,

    #[arg(long, value_enum, default_value_t = InferenceBackend::Wgpu)]
    pub backend: InferenceBackend,

    #[arg(long)]
    pub weights_root: Option<PathBuf>,

    #[arg(long)]
    pub trellis_weights_root: Option<PathBuf>,

    #[arg(long)]
    pub trellis_image_large_root: Option<PathBuf>,

    #[arg(long)]
    pub trellis_python_bin: Option<PathBuf>,

    #[arg(long)]
    pub trellis_bridge_script: Option<PathBuf>,

    #[arg(long, value_enum, default_value_t = TrellisQuality::Medium)]
    pub trellis_quality: TrellisQuality,

    /// Quality preset (fast, balanced, full). Individual flags override this preset.
    #[arg(long, value_enum, default_value_t = QualityPreset::Balanced)]
    pub quality: QualityPreset,

    #[arg(long)]
    pub bg_weights_root: Option<PathBuf>,

    #[arg(long)]
    pub num_steps: Option<usize>,

    #[arg(long)]
    pub num_tokens: Option<usize>,

    #[arg(long)]
    pub guidance_scale: Option<f32>,

    /// Batch chunk size for image generation tools. Use 0 for auto.
    #[arg(long, default_value_t = 0)]
    pub batch_size: usize,

    /// Explicit VRAM budget in MB for auto batch planning.
    #[arg(long)]
    pub batch_vram_mb: Option<u64>,

    /// Bevy scene command file path for scene_* tools.
    #[arg(long)]
    pub scene_control_path: Option<PathBuf>,

    /// Bevy scene status file path. Defaults to <scene-control-path>.status.json.
    #[arg(long)]
    pub scene_status_path: Option<PathBuf>,

    /// Timeout for scene command acknowledgements.
    #[arg(long, default_value_t = 5000)]
    pub scene_timeout_ms: u64,
}

#[derive(Clone, Debug)]
pub struct ServerConfig {
    pub default_rmbg_model: ForegroundModel,
    pub default_synthesis_models: Vec<SynthesisModel>,
    pub default_backend: InferenceBackend,
    pub weights_root: Option<PathBuf>,
    pub trellis_weights_root: Option<PathBuf>,
    pub trellis_image_large_root: Option<PathBuf>,
    pub trellis_python_bin: Option<PathBuf>,
    pub trellis_bridge_script: Option<PathBuf>,
    pub trellis_quality: TrellisQuality,
    pub quality: QualityPreset,
    pub bg_weights_root: Option<PathBuf>,
    pub num_steps: usize,
    pub num_tokens: usize,
    pub guidance_scale: f32,
    pub flash_octree_depth: usize,
    pub flash_min_resolution: usize,
    pub flash_mini_grid_num: usize,
    pub flash_num_chunks: usize,
    pub batch_size: Option<usize>,
    pub batch_vram_mb: Option<u64>,
    pub scene_control_path: Option<PathBuf>,
    pub scene_status_path: Option<PathBuf>,
    pub scene_timeout: Duration,
}

impl ServerConfig {
    pub fn from_args(args: ServerArgs) -> Self {
        let quality = args.quality;
        let defaults = quality.defaults();
        Self {
            default_rmbg_model: args.rmbg_model,
            default_synthesis_models: sanitize_synthesis_models(args.synthesis_models),
            default_backend: args.backend,
            weights_root: args.weights_root,
            trellis_weights_root: args.trellis_weights_root,
            trellis_image_large_root: args.trellis_image_large_root,
            trellis_python_bin: args.trellis_python_bin,
            trellis_bridge_script: args.trellis_bridge_script,
            trellis_quality: args.trellis_quality,
            quality,
            bg_weights_root: args.bg_weights_root,
            num_steps: args.num_steps.unwrap_or(defaults.num_steps),
            num_tokens: args.num_tokens.unwrap_or(defaults.num_tokens),
            guidance_scale: args.guidance_scale.unwrap_or(defaults.guidance_scale),
            flash_octree_depth: defaults.flash_octree_depth,
            flash_min_resolution: defaults.flash_min_resolution,
            flash_mini_grid_num: defaults.flash_mini_grid_num,
            flash_num_chunks: defaults.flash_num_chunks,
            batch_size: (args.batch_size > 0).then_some(args.batch_size),
            batch_vram_mb: args.batch_vram_mb,
            scene_status_path: args.scene_status_path.or_else(|| {
                args.scene_control_path
                    .as_ref()
                    .map(|path| path.with_extension("status.json"))
            }),
            scene_control_path: args.scene_control_path,
            scene_timeout: Duration::from_millis(args.scene_timeout_ms.max(1)),
        }
    }

    fn runtime_config(&self) -> RuntimeConfig {
        let mut config = RuntimeConfig {
            model_selection: ModelSelection::new(
                self.default_synthesis_models
                    .iter()
                    .copied()
                    .map(Into::into),
                self.default_rmbg_model.into(),
            ),
            backend: self.default_backend.into(),
            weights_root: self.weights_root.clone(),
            trellis_weights_root: self.trellis_weights_root.clone(),
            trellis_image_large_root: self.trellis_image_large_root.clone(),
            trellis_python_bin: self.trellis_python_bin.clone(),
            trellis_bridge_script: self.trellis_bridge_script.clone(),
            trellis_quality: self.trellis_quality.into(),
            bg_weights_root: self.bg_weights_root.clone(),
            num_steps: self.num_steps,
            num_tokens: self.num_tokens,
            guidance_scale: self.guidance_scale,
            ..RuntimeConfig::default()
        };
        config.flash_extract.octree_depth = self.flash_octree_depth;
        config.flash_extract.min_resolution = self.flash_min_resolution;
        config.flash_extract.mini_grid_num = self.flash_mini_grid_num;
        config.flash_extract.num_chunks = self.flash_num_chunks;
        config
    }
}

pub fn run_from_args(args: ServerArgs) -> Result<(), String> {
    run_stdio_server(ServerConfig::from_args(args))
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

struct McpServer {
    config: ServerConfig,
    runtime: SynthRuntime,
    should_exit: bool,
}

impl McpServer {
    fn new(config: ServerConfig) -> Self {
        let runtime = SynthRuntime::new(config.runtime_config());
        Self {
            config,
            runtime,
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
            "scene_status" => match self.call_scene_status() {
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

    fn call_images_to_assets(&mut self, args: ImagesToAssetsToolArgs) -> Result<Value, String> {
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

        let batch = self
            .runtime
            .synthesize_assets_batch(AssetBatchRequest {
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
            })
            .map_err(|err| err.to_string())?;

        let mut items = Vec::with_capacity(batch.items.len());
        for (batch_item, input_path) in batch.items.into_iter().zip(args.input_image_paths.iter()) {
            let output = batch_item.output.map_err(|err| err.to_string())?;
            let item = write_asset_output(
                input_path,
                args.output_dir.as_deref(),
                args.output_paths
                    .as_ref()
                    .and_then(|paths| paths.get(batch_item.item_index).cloned()),
                args.output_format.unwrap_or(AssetOutputFormat::Auto),
                output.asset,
                args.target_faces,
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
                "target_faces": args.target_faces.filter(|value| *value > 0),
                "material": item.material,
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
            "dry_run": args.dry_run,
        }))
    }

    fn call_scene_status(&self) -> Result<Value, String> {
        let status_path = self
            .config
            .scene_status_path
            .as_ref()
            .ok_or_else(|| "scene_status_path is not configured".to_string())?;
        read_scene_status(status_path)
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
        })])
    }

    fn call_scene_capture(&self, args: SceneCaptureArgs) -> Result<Value, String> {
        let path = args.output_path;
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

    fn send_scene_commands(&self, commands: Vec<Value>) -> Result<Value, String> {
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
        atomic_write_json(control_path, &envelope)?;

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

fn scene_commands_from_plan(plan: &SceneComposePlan) -> Result<Vec<Value>, String> {
    let mut commands = Vec::with_capacity(plan.placements.len());
    if plan.clear_existing {
        commands.push(json!({ "type": "clear_scene" }));
    }
    for placement in &plan.placements {
        if let Some(path) = placement.path.as_ref() {
            commands.push(json!({
                "type": "spawn_path",
                "path": path,
                "cache_key": placement.cache_key,
                "translation": placement.translation,
                "rotation": placement.rotation,
                "scale": placement.scale,
                "select": placement.select,
            }));
        } else if let Some(cache_key) = placement.cache_key.as_ref() {
            commands.push(json!({
                "type": "spawn_cached",
                "cache_key": cache_key,
                "translation": placement.translation,
                "rotation": placement.rotation,
                "scale": placement.scale,
                "select": placement.select,
            }));
        } else {
            return Err(format!(
                "placement for '{}' has neither path nor cache_key",
                placement.label
            ));
        }
    }
    Ok(commands)
}

fn sanitize_synthesis_models(models: Vec<SynthesisModel>) -> Vec<SynthesisModel> {
    let mut out = Vec::new();
    for model in models {
        if !out.contains(&model) {
            out.push(model);
        }
    }
    if out.is_empty() {
        out.push(SynthesisModel::Triposg);
    }
    out
}

fn default_output_path(input: &Path, suffix: &str, ext: &str) -> PathBuf {
    let parent = input.parent().unwrap_or_else(|| Path::new("."));
    let stem = input
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("output");
    parent.join(format!("{stem}{suffix}.{ext}"))
}

#[derive(Debug)]
struct WrittenAsset {
    output_path: PathBuf,
    output_format: AssetOutputFormat,
    asset_kind: &'static str,
    vertices: Option<usize>,
    faces: Option<usize>,
    gaussians: Option<usize>,
    material: Option<Value>,
}

fn write_asset_output(
    input_path: &Path,
    output_dir: Option<&Path>,
    explicit_output: Option<PathBuf>,
    requested_format: AssetOutputFormat,
    asset: SynthesisAsset,
    target_faces: Option<usize>,
) -> Result<WrittenAsset, String> {
    match asset {
        SynthesisAsset::Mesh(mesh) => {
            if matches!(
                requested_format,
                AssetOutputFormat::Splat | AssetOutputFormat::Ply
            ) {
                return Err(format!(
                    "mesh synthesis cannot be written as {}",
                    requested_format.as_str()
                ));
            }
            let mesh = apply_mesh_decimation(mesh, target_faces)
                .map_err(|err| format!("mesh decimation failed: {err}"))?;
            let output_path =
                resolve_asset_output_path(input_path, output_dir, explicit_output, "_mesh", "glb");
            write_glb_mesh(output_path.as_path(), &mesh)?;
            let material = mesh.material.map(|value| {
                json!({
                    "base_color": value.base_color,
                    "metallic": value.metallic,
                    "roughness": value.roughness,
                    "alpha": value.alpha,
                })
            });
            Ok(WrittenAsset {
                output_path,
                output_format: AssetOutputFormat::Glb,
                asset_kind: "mesh",
                vertices: Some(mesh.vertices.len()),
                faces: Some(mesh.faces.len()),
                gaussians: None,
                material,
            })
        }
        SynthesisAsset::GaussianSplat(splats) => {
            if matches!(requested_format, AssetOutputFormat::Glb) {
                return Err("Gaussian splats cannot be written as glb".to_string());
            }
            let output_format = match requested_format {
                AssetOutputFormat::Ply => AssetOutputFormat::Ply,
                _ => AssetOutputFormat::Splat,
            };
            let output_path = resolve_asset_output_path(
                input_path,
                output_dir,
                explicit_output,
                "_splat",
                output_format.as_str(),
            );
            write_splat_asset(output_path.as_path(), &splats, output_format)?;
            Ok(WrittenAsset {
                output_path,
                output_format,
                asset_kind: "gaussian_splat",
                vertices: None,
                faces: None,
                gaussians: Some(splats.len()),
                material: None,
            })
        }
    }
}

fn resolve_asset_output_path(
    input_path: &Path,
    output_dir: Option<&Path>,
    explicit_output: Option<PathBuf>,
    suffix: &str,
    ext: &str,
) -> PathBuf {
    if let Some(path) = explicit_output {
        if path.extension().is_none() || path.is_dir() {
            let stem = input_path
                .file_stem()
                .and_then(|value| value.to_str())
                .unwrap_or("asset");
            return path.join(format!("{stem}{suffix}.{ext}"));
        }
        return path;
    }
    if let Some(dir) = output_dir {
        let stem = input_path
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("asset");
        return dir.join(format!("{stem}{suffix}.{ext}"));
    }
    default_output_path(input_path, suffix, ext)
}

fn write_splat_asset(
    path: &Path,
    splats: &burn_synth::triposplat::GaussianSplatCloud,
    format: AssetOutputFormat,
) -> Result<(), String> {
    ensure_parent_dir(path).map_err(|err| err.to_string())?;
    match format {
        AssetOutputFormat::Ply => splats.write_ply(path),
        AssetOutputFormat::Splat | AssetOutputFormat::Auto => splats.write_splat(path),
        AssetOutputFormat::Glb => Err("Gaussian splats cannot be written as glb".to_string()),
    }
}

fn apply_mesh_decimation(mesh: Mesh, target_faces: Option<usize>) -> Result<Mesh, String> {
    let target_faces = target_faces.filter(|value| *value > 0);
    let Some(target) = target_faces else {
        return Ok(mesh);
    };
    if mesh.faces.len() <= target {
        return Ok(mesh);
    }
    decimate_mesh(&mesh, target)
}

fn decimate_mesh(mesh: &Mesh, target_faces: usize) -> Result<Mesh, String> {
    if target_faces == 0 || mesh.faces.len() <= target_faces {
        return Ok(mesh.clone());
    }
    if mesh.faces.is_empty() || mesh.vertices.is_empty() {
        return Ok(mesh.clone());
    }

    let mut indices = Vec::with_capacity(mesh.faces.len() * 3);
    for face in &mesh.faces {
        indices.push(face[0]);
        indices.push(face[1]);
        indices.push(face[2]);
    }
    let target_index_count = (target_faces.saturating_mul(3)).min(indices.len());
    if target_index_count < 3 {
        return Err("target face count too small for decimation".to_string());
    }

    let vertices_bytes = meshopt::typed_to_bytes(mesh.vertices.as_slice());
    let adapter =
        meshopt::VertexDataAdapter::new(vertices_bytes, std::mem::size_of::<[f32; 3]>(), 0)
            .map_err(|err| format!("meshopt vertex adapter: {err}"))?;

    let mut result_error = 0.0f32;
    let mut simplified = meshopt::simplify(
        &indices,
        &adapter,
        target_index_count,
        1.0,
        meshopt::SimplifyOptions::None,
        Some(&mut result_error),
    );
    if simplified.len() > target_index_count {
        simplified = meshopt::simplify_sloppy(&indices, &adapter, target_index_count, 1.0, None);
    }
    if simplified.len() < 3 {
        return Err("meshopt simplification produced empty mesh".to_string());
    }

    let (vertex_count, remap) =
        meshopt::generate_vertex_remap(mesh.vertices.as_slice(), Some(&simplified));
    let vertices = meshopt::remap_vertex_buffer(mesh.vertices.as_slice(), vertex_count, &remap);
    let uvs = if mesh.uvs.len() == mesh.vertices.len() && !mesh.uvs.is_empty() {
        meshopt::remap_vertex_buffer(mesh.uvs.as_slice(), vertex_count, &remap)
    } else {
        Vec::new()
    };
    let indices = meshopt::remap_index_buffer(Some(&simplified), vertex_count, &remap);
    if indices.len() < 3 {
        return Err("meshopt remap produced empty mesh".to_string());
    }

    let faces = indices
        .chunks_exact(3)
        .map(|chunk| [chunk[0], chunk[1], chunk[2]])
        .collect::<Vec<[u32; 3]>>();
    Ok(Mesh {
        vertices,
        faces,
        uvs,
        material: mesh.material,
        pbr_textures: mesh.pbr_textures.clone(),
    })
}

fn ensure_parent_dir(path: &Path) -> io::Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn next_scene_sequence() -> u64 {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(u128::from(u64::MAX)) as u64;
    let mut current = NEXT_SCENE_SEQUENCE.load(Ordering::Relaxed);
    loop {
        let next = now.max(current.saturating_add(1));
        match NEXT_SCENE_SEQUENCE.compare_exchange_weak(
            current,
            next,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => return next,
            Err(value) => current = value,
        }
    }
}

fn atomic_write_json(path: &Path, value: &Value) -> Result<(), String> {
    ensure_parent_dir(path).map_err(|err| err.to_string())?;
    let tmp = path.with_extension("json.tmp");
    let bytes = serde_json::to_vec_pretty(value)
        .map_err(|err| format!("failed to serialize scene command: {err}"))?;
    fs::write(&tmp, bytes).map_err(|err| format!("failed to write {}: {err}", tmp.display()))?;
    fs::rename(&tmp, path).map_err(|err| {
        format!(
            "failed to atomically replace scene command file {}: {err}",
            path.display()
        )
    })
}

fn read_scene_status(path: &Path) -> Result<Value, String> {
    let content = fs::read_to_string(path)
        .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    serde_json::from_str(&content)
        .map_err(|err| format!("failed to parse scene status {}: {err}", path.display()))
}

fn wait_scene_status(path: &Path, sequence: u64, timeout: Duration) -> Result<Value, String> {
    let started = Instant::now();
    let mut last_error = None;
    while started.elapsed() < timeout {
        match read_scene_status(path) {
            Ok(status) => {
                let acknowledged = status
                    .get("last_sequence")
                    .and_then(Value::as_u64)
                    .map(|last| last >= sequence)
                    .unwrap_or(false);
                if acknowledged {
                    return Ok(status);
                }
            }
            Err(err) => last_error = Some(err),
        }
        thread::sleep(Duration::from_millis(25));
    }
    Err(format!(
        "timed out waiting for scene status {} to acknowledge sequence {sequence}{}",
        path.display(),
        last_error
            .map(|err| format!("; last read error: {err}"))
            .unwrap_or_default()
    ))
}

fn success_response(id: Option<Value>, result: Value) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id.unwrap_or(Value::Null),
        "result": result,
    })
}

fn error_response(id: Option<Value>, code: i32, message: String) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id.unwrap_or(Value::Null),
        "error": {
            "code": code,
            "message": message,
        }
    })
}

fn success_tool_result(payload: Value) -> Value {
    let text = serde_json::to_string_pretty(&payload)
        .unwrap_or_else(|_| "{\"error\":\"failed to render tool payload\"}".to_string());
    json!({
        "content": [
            {
                "type": "text",
                "text": text,
            }
        ],
        "structuredContent": payload,
    })
}

fn error_tool_result(message: String) -> Value {
    json!({
        "isError": true,
        "content": [
            {
                "type": "text",
                "text": message,
            }
        ]
    })
}

fn tool_defs() -> Vec<Value> {
    vec![
        json!({
            "name": "image_to_foreground",
            "description": "Extract foreground alpha from an input image and write a PNG with transparency.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_image_path": { "type": "string", "description": "Path to input image file." },
                    "output_image_path": { "type": "string", "description": "Optional output path (defaults to *_foreground.png)." },
                    "rmbg_model": { "type": "string", "enum": ["rmbg14", "rmbg2"], "description": "Optional RMBG model override." },
                    "dry_run": { "type": "boolean", "description": "Skip model inference and just write a pass-through output image." }
                },
                "required": ["input_image_path"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "image_to_mesh",
            "description": "Run image-to-mesh synthesis and write a GLB mesh output.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_image_path": { "type": "string", "description": "Path to input image file." },
                    "output_mesh_path": { "type": "string", "description": "Optional output GLB path (defaults to *_mesh.glb)." },
                    "rmbg_model": { "type": "string", "enum": ["rmbg14", "rmbg2"], "description": "Optional RMBG model override." },
                    "synthesis_models": { "type": "array", "items": { "type": "string", "enum": ["triposg", "trellis"] }, "description": "Optional mesh synthesis model list override, ordered by preference." },
                    "backend": { "type": "string", "enum": ["cpu", "wgpu", "cuda"], "description": "Optional backend override." },
                    "target_faces": { "type": "integer", "description": "Optional target face count for mesh simplification." },
                    "dry_run": { "type": "boolean", "description": "Skip model inference and emit a canonical cube mesh." }
                },
                "required": ["input_image_path"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "image_to_splat",
            "description": "Run TripoSplat image-to-Gaussian-splat synthesis and write a .splat or .ply output.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_image_path": { "type": "string", "description": "Path to input image file." },
                    "output_splat_path": { "type": "string", "description": "Optional output path (defaults to *_splat.splat)." },
                    "output_format": { "type": "string", "enum": ["splat", "ply"], "description": "Optional splat output format." },
                    "rmbg_model": { "type": "string", "enum": ["rmbg14", "rmbg2"], "description": "Optional RMBG model override." },
                    "backend": { "type": "string", "enum": ["cpu", "wgpu", "cuda"], "description": "Optional backend override." },
                    "dry_run": { "type": "boolean", "description": "Skip model inference and emit a canonical debug splat cloud." }
                },
                "required": ["input_image_path"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "images_to_assets",
            "description": "Run batched image-to-asset synthesis over multiple images with shared model loading and chunk planning.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_image_paths": { "type": "array", "items": { "type": "string" }, "description": "Input image paths to process in one batch request." },
                    "output_dir": { "type": "string", "description": "Optional output directory for per-input output names." },
                    "output_paths": { "type": "array", "items": { "type": "string" }, "description": "Optional explicit output path per input." },
                    "output_format": { "type": "string", "enum": ["auto", "glb", "splat", "ply"], "description": "Optional output format. Auto writes GLB for meshes and .splat for Gaussian splats." },
                    "rmbg_model": { "type": "string", "enum": ["rmbg14", "rmbg2"], "description": "Optional RMBG model override." },
                    "synthesis_models": { "type": "array", "items": { "type": "string", "enum": ["triposg", "trellis", "triposplat"] }, "description": "Optional synthesis model list override, ordered by preference." },
                    "backend": { "type": "string", "enum": ["cpu", "wgpu", "cuda"], "description": "Optional backend override." },
                    "target_faces": { "type": "integer", "description": "Optional target face count for mesh simplification." },
                    "batch_size": { "type": "integer", "description": "Optional explicit chunk size; omit for server default/auto." },
                    "batch_vram_mb": { "type": "integer", "description": "Optional VRAM budget in MB for auto chunking." },
                    "dry_run": { "type": "boolean", "description": "Skip model inference and emit canonical debug assets." }
                },
                "required": ["input_image_paths"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "scene_status",
            "description": "Read the latest Bevy scene bridge status, including cache entries, world items, camera, and screenshots.",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }
        }),
        json!({
            "name": "scene_list_assets",
            "description": "List cached assets and spawned world items from the latest Bevy scene bridge status.",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }
        }),
        json!({
            "name": "scene_spawn_cached",
            "description": "Spawn an asset already present in the Bevy mesh/splat cache.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "cache_key": { "type": "string", "description": "Cache key to spawn." },
                    "translation": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 },
                    "rotation": { "type": "array", "items": { "type": "number" }, "minItems": 4, "maxItems": 4 },
                    "scale": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 },
                    "select": { "type": "boolean", "description": "Select the spawned entity." }
                },
                "required": ["cache_key"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "scene_spawn_path",
            "description": "Spawn a GLB mesh asset file directly into the Bevy scene.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "GLB mesh path to spawn." },
                    "translation": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 },
                    "rotation": { "type": "array", "items": { "type": "number" }, "minItems": 4, "maxItems": 4 },
                    "scale": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 },
                    "select": { "type": "boolean", "description": "Select the spawned entity." }
                },
                "required": ["path"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "scene_delete",
            "description": "Delete a spawned cached asset by cache key, delete the selection, or clear selection.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "cache_key": { "type": "string", "description": "Cache key to delete." },
                    "selected": { "type": "boolean", "description": "Delete the current selection when true; clear selection when false and no cache key is provided." }
                },
                "additionalProperties": false
            }
        }),
        json!({
            "name": "scene_clear",
            "description": "Clear all spawned cache-backed scene items from the Bevy scene.",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }
        }),
        json!({
            "name": "scene_set_camera",
            "description": "Set the Bevy scene camera transform and optional orbit state.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "translation": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 },
                    "rotation": { "type": "array", "items": { "type": "number" }, "minItems": 4, "maxItems": 4 },
                    "focus": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 },
                    "yaw": { "type": "number" },
                    "pitch": { "type": "number" },
                    "radius": { "type": "number" }
                },
                "required": ["translation", "rotation"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "scene_save",
            "description": "Flush the Bevy scene cache/world state.",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }
        }),
        json!({
            "name": "scene_capture",
            "description": "Capture a screenshot from the Bevy primary window and wait for the image file.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_path": { "type": "string", "description": "Screenshot path to write." }
                },
                "required": ["output_path"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "scene_compose_assets",
            "description": "Create deterministic Bevy placements from source-image object boxes and generated asset bindings; optionally apply them to the live scene.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "reference_objects": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": { "type": "string" },
                                "label": { "type": "string" },
                                "aliases": { "type": "array", "items": { "type": "string" } },
                                "bbox": { "type": "array", "items": { "type": "number" }, "minItems": 4, "maxItems": 4, "description": "Normalized source-image box [x_min, y_min, x_max, y_max]." }
                            },
                            "required": ["label", "bbox"],
                            "additionalProperties": false
                        }
                    },
                    "assets": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "reference_id": { "type": "string" },
                                "label": { "type": "string" },
                                "aliases": { "type": "array", "items": { "type": "string" } },
                                "path": { "type": "string" },
                                "cache_key": { "type": "string" },
                                "select": { "type": "boolean" }
                            },
                            "additionalProperties": false
                        }
                    },
            "apply": { "type": "boolean", "description": "When true, send spawn commands to the configured Bevy scene bridge." },
                    "clear_existing": { "type": "boolean", "description": "When true, clear existing scene instances before placing generated assets." },
                    "layout_width": { "type": "number" },
                    "layout_depth": { "type": "number" },
                    "y": { "type": "number" },
                    "min_scale": { "type": "number" },
                    "scale_multiplier": { "type": "number" }
                },
                "required": ["reference_objects", "assets"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "scene_validate_layout",
            "description": "Validate a composed Bevy scene against source-image object boxes using semantic label matching, object counts, normalized layout, and optional screenshot image similarity.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "reference_objects": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": { "type": "string" },
                                "label": { "type": "string" },
                                "aliases": { "type": "array", "items": { "type": "string" } },
                                "bbox": { "type": "array", "items": { "type": "number" }, "minItems": 4, "maxItems": 4 }
                            },
                            "required": ["label", "bbox"],
                            "additionalProperties": false
                        }
                    },
                    "scene_status": { "type": "object", "description": "Optional scene status JSON. Omit to read the configured scene_status_path." },
                    "source_image_path": { "type": "string" },
                    "rendered_image_path": { "type": "string" },
                    "thresholds": {
                        "type": "object",
                        "properties": {
                            "min_semantic_score": { "type": "number" },
                            "min_layout_score": { "type": "number" },
                            "min_overall_score": { "type": "number" },
                            "max_extra_objects": { "type": "integer" },
                            "min_image_similarity": { "type": "number" }
                        },
                        "additionalProperties": false
                    }
                },
                "required": ["reference_objects"],
                "additionalProperties": false
            }
        }),
    ]
}

fn read_framed_json<R: BufRead + Read>(reader: &mut R) -> io::Result<Option<Value>> {
    let mut content_length = None;
    let mut saw_header = false;

    loop {
        let mut line = String::new();
        let bytes = reader.read_line(&mut line)?;
        if bytes == 0 {
            if !saw_header {
                return Ok(None);
            }
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "unexpected EOF while reading MCP headers",
            ));
        }
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            break;
        }
        saw_header = true;
        if let Some((name, value)) = trimmed.split_once(':')
            && name.eq_ignore_ascii_case("Content-Length")
        {
            let parsed = value.trim().parse::<usize>().map_err(|err| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid Content-Length header: {err}"),
                )
            })?;
            content_length = Some(parsed);
        }
    }

    let content_length = content_length.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "missing Content-Length header in MCP message",
        )
    })?;
    let mut payload = vec![0u8; content_length];
    reader.read_exact(&mut payload)?;
    let value = serde_json::from_slice::<Value>(&payload).map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid MCP JSON payload: {err}"),
        )
    })?;
    Ok(Some(value))
}

fn write_framed_json<W: Write>(writer: &mut W, value: &Value) -> io::Result<()> {
    let payload = serde_json::to_vec(value).map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("failed to serialize MCP JSON payload: {err}"),
        )
    })?;
    write!(writer, "Content-Length: {}\r\n\r\n", payload.len())?;
    writer.write_all(&payload)?;
    writer.flush()
}

#[derive(Debug, Deserialize)]
struct RpcRequest {
    pub id: Option<Value>,
    pub method: String,
    #[serde(default)]
    pub params: Option<Value>,
}

#[derive(Debug, Default, Deserialize)]
struct InitializeParams {
    #[serde(rename = "protocolVersion")]
    pub protocol_version: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ToolsCallParams {
    pub name: String,
    #[serde(default)]
    pub arguments: Value,
}

#[derive(Debug, Deserialize)]
struct ForegroundToolArgs {
    #[serde(alias = "image_path")]
    pub input_image_path: PathBuf,
    #[serde(default, alias = "output_path")]
    pub output_image_path: Option<PathBuf>,
    #[serde(default)]
    pub rmbg_model: Option<ForegroundModel>,
    #[serde(default)]
    pub dry_run: bool,
}

#[derive(Debug, Deserialize)]
struct MeshToolArgs {
    #[serde(alias = "image_path")]
    pub input_image_path: PathBuf,
    #[serde(default, alias = "output_path")]
    pub output_mesh_path: Option<PathBuf>,
    #[serde(default)]
    pub output_format: Option<MeshOutputFormat>,
    #[serde(default)]
    pub rmbg_model: Option<ForegroundModel>,
    #[serde(default)]
    pub synthesis_models: Option<Vec<SynthesisModel>>,
    #[serde(default)]
    pub backend: Option<InferenceBackend>,
    #[serde(default)]
    pub target_faces: Option<usize>,
    #[serde(default)]
    pub dry_run: bool,
}

#[derive(Debug, Deserialize)]
struct SplatToolArgs {
    #[serde(alias = "image_path")]
    pub input_image_path: PathBuf,
    #[serde(default, alias = "output_path")]
    pub output_splat_path: Option<PathBuf>,
    #[serde(default)]
    pub output_format: Option<AssetOutputFormat>,
    #[serde(default)]
    pub rmbg_model: Option<ForegroundModel>,
    #[serde(default)]
    pub backend: Option<InferenceBackend>,
    #[serde(default)]
    pub dry_run: bool,
}

#[derive(Debug, Deserialize)]
struct ImagesToAssetsToolArgs {
    #[serde(default, alias = "image_paths")]
    pub input_image_paths: Vec<PathBuf>,
    #[serde(default)]
    pub output_dir: Option<PathBuf>,
    #[serde(default)]
    pub output_paths: Option<Vec<PathBuf>>,
    #[serde(default)]
    pub output_format: Option<AssetOutputFormat>,
    #[serde(default)]
    pub rmbg_model: Option<ForegroundModel>,
    #[serde(default)]
    pub synthesis_models: Option<Vec<SynthesisModel>>,
    #[serde(default)]
    pub backend: Option<InferenceBackend>,
    #[serde(default)]
    pub target_faces: Option<usize>,
    #[serde(default)]
    pub batch_size: Option<usize>,
    #[serde(default)]
    pub batch_vram_mb: Option<u64>,
    #[serde(default)]
    pub dry_run: bool,
}

#[derive(Debug, Deserialize)]
struct SceneSpawnCachedArgs {
    pub cache_key: String,
    #[serde(default)]
    pub translation: Option<[f32; 3]>,
    #[serde(default)]
    pub rotation: Option<[f32; 4]>,
    #[serde(default)]
    pub scale: Option<[f32; 3]>,
    #[serde(default)]
    pub select: bool,
}

#[derive(Debug, Deserialize)]
struct SceneSpawnPathArgs {
    pub path: PathBuf,
    #[serde(default)]
    pub translation: Option<[f32; 3]>,
    #[serde(default)]
    pub rotation: Option<[f32; 4]>,
    #[serde(default)]
    pub scale: Option<[f32; 3]>,
    #[serde(default)]
    pub select: bool,
}

#[derive(Debug, Default, Deserialize)]
struct SceneDeleteArgs {
    #[serde(default)]
    pub cache_key: Option<String>,
    #[serde(default)]
    pub selected: bool,
}

#[derive(Debug, Deserialize)]
struct SceneSetCameraArgs {
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    #[serde(default)]
    pub focus: Option<[f32; 3]>,
    #[serde(default)]
    pub yaw: Option<f32>,
    #[serde(default)]
    pub pitch: Option<f32>,
    #[serde(default)]
    pub radius: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct SceneCaptureArgs {
    #[serde(alias = "path")]
    pub output_path: PathBuf,
}

impl ForegroundModel {
    fn as_str(self) -> &'static str {
        match self {
            ForegroundModel::Rmbg14 => "rmbg14",
            ForegroundModel::Rmbg2 => "rmbg2",
        }
    }
}

impl SynthesisModel {
    fn as_str(self) -> &'static str {
        match self {
            SynthesisModel::Triposg => "triposg",
            SynthesisModel::Trellis => "trellis",
            SynthesisModel::Triposplat => "triposplat",
        }
    }
}

impl InferenceBackend {
    fn as_str(self) -> &'static str {
        match self {
            InferenceBackend::Cpu => "cpu",
            InferenceBackend::Wgpu => "wgpu",
            InferenceBackend::Cuda => "cuda",
        }
    }
}

impl MeshOutputFormat {
    fn as_str(self) -> &'static str {
        match self {
            MeshOutputFormat::Obj => "obj",
            MeshOutputFormat::Gltf => "gltf",
            MeshOutputFormat::Glb => "glb",
        }
    }
}

impl AssetOutputFormat {
    fn as_str(self) -> &'static str {
        match self {
            AssetOutputFormat::Auto => "auto",
            AssetOutputFormat::Glb => "glb",
            AssetOutputFormat::Splat => "splat",
            AssetOutputFormat::Ply => "ply",
        }
    }
}

fn runtime_foreground_model_str(value: burn_synth::ForegroundModel) -> &'static str {
    match value {
        burn_synth::ForegroundModel::Rmbg14 => "rmbg14",
        burn_synth::ForegroundModel::Rmbg2 => "rmbg2",
    }
}

fn runtime_synthesis_model_str(value: burn_synth::SynthesisModel) -> &'static str {
    match value {
        burn_synth::SynthesisModel::Triposg => "triposg",
        burn_synth::SynthesisModel::Trellis => "trellis",
        burn_synth::SynthesisModel::Triposplat => "triposplat",
    }
}

fn runtime_backend_str(value: burn_synth::InferenceBackend) -> &'static str {
    match value {
        burn_synth::InferenceBackend::Cpu => "cpu",
        burn_synth::InferenceBackend::Wgpu => "wgpu",
        burn_synth::InferenceBackend::Cuda => "cuda",
    }
}

impl From<ForegroundModel> for burn_synth::ForegroundModel {
    fn from(value: ForegroundModel) -> Self {
        match value {
            ForegroundModel::Rmbg14 => Self::Rmbg14,
            ForegroundModel::Rmbg2 => Self::Rmbg2,
        }
    }
}

impl From<SynthesisModel> for burn_synth::SynthesisModel {
    fn from(value: SynthesisModel) -> Self {
        match value {
            SynthesisModel::Triposg => Self::Triposg,
            SynthesisModel::Trellis => Self::Trellis,
            SynthesisModel::Triposplat => Self::Triposplat,
        }
    }
}

impl From<InferenceBackend> for burn_synth::InferenceBackend {
    fn from(value: InferenceBackend) -> Self {
        match value {
            InferenceBackend::Cpu => Self::Cpu,
            InferenceBackend::Wgpu => Self::Wgpu,
            InferenceBackend::Cuda => Self::Cuda,
        }
    }
}

impl From<TrellisQuality> for burn_synth::TrellisQuality {
    fn from(value: TrellisQuality) -> Self {
        match value {
            TrellisQuality::Low => Self::Low,
            TrellisQuality::Medium => Self::Medium,
            TrellisQuality::High => Self::High,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn server_args_default_to_balanced_quality_defaults() {
        let args = ServerArgs::parse_from(["burn_synth_mcp"]);
        let config = ServerConfig::from_args(args);
        assert_eq!(config.quality, QualityPreset::Balanced);
        assert_eq!(config.num_steps, 20);
        assert_eq!(config.num_tokens, 1024);
        assert_eq!(config.guidance_scale, 7.0);
        assert_eq!(config.flash_octree_depth, 8);
        assert_eq!(config.flash_min_resolution, 31);
        assert_eq!(config.flash_mini_grid_num, 4);
        assert_eq!(config.flash_num_chunks, 8192);
    }

    #[test]
    fn server_args_quality_and_explicit_overrides_map_to_runtime_config() {
        let args = ServerArgs::parse_from([
            "burn_synth_mcp",
            "--quality",
            "fast",
            "--num-steps",
            "18",
            "--guidance-scale",
            "6.5",
        ]);
        let config = ServerConfig::from_args(args);
        assert_eq!(config.quality, QualityPreset::Fast);
        assert_eq!(config.num_steps, 18);
        assert_eq!(config.num_tokens, 512);
        assert_eq!(config.guidance_scale, 6.5);
        assert_eq!(config.flash_octree_depth, 7);
        assert_eq!(config.flash_min_resolution, 31);
        assert_eq!(config.flash_mini_grid_num, 2);
        assert_eq!(config.flash_num_chunks, 4096);

        let runtime = config.runtime_config();
        assert_eq!(runtime.num_steps, 18);
        assert_eq!(runtime.num_tokens, 512);
        assert_eq!(runtime.guidance_scale, 6.5);
        assert_eq!(runtime.flash_extract.octree_depth, 7);
        assert_eq!(runtime.flash_extract.min_resolution, 31);
        assert_eq!(runtime.flash_extract.mini_grid_num, 2);
        assert_eq!(runtime.flash_extract.num_chunks, 4096);
    }

    #[test]
    fn tool_list_includes_batch_splat_and_scene_tools() {
        let tools = tool_defs();
        let names = tools
            .iter()
            .filter_map(|tool| tool["name"].as_str())
            .collect::<Vec<_>>();
        for expected in [
            "images_to_assets",
            "image_to_splat",
            "scene_status",
            "scene_spawn_cached",
            "scene_spawn_path",
            "scene_clear",
            "scene_capture",
            "scene_compose_assets",
            "scene_validate_layout",
        ] {
            assert!(names.contains(&expected), "missing tool {expected}");
        }
    }

    #[test]
    fn scene_compose_plan_generates_spawn_commands_with_validation_keys() {
        let plan = compose_scene_layout(SceneComposeArgs {
            reference_objects: vec![scene_layout::SceneReferenceObject {
                id: Some("chair_1".to_string()),
                label: "chair".to_string(),
                aliases: Vec::new(),
                bbox: [0.1, 0.2, 0.3, 0.6],
            }],
            assets: vec![scene_layout::SceneAssetBinding {
                reference_id: Some("chair_1".to_string()),
                label: Some("chair".to_string()),
                aliases: Vec::new(),
                path: Some(PathBuf::from("/tmp/chair.glb")),
                cache_key: None,
                select: true,
            }],
            apply: false,
            clear_existing: true,
            layout_width: 6.0,
            layout_depth: 4.0,
            y: 0.0,
            min_scale: 0.35,
            scale_multiplier: 1.0,
        })
        .expect("compose plan");
        let commands = scene_commands_from_plan(&plan).expect("scene commands");
        assert_eq!(commands.len(), 2);
        assert_eq!(commands[0]["type"], "clear_scene");
        assert_eq!(commands[1]["type"], "spawn_path");
        assert_eq!(commands[1]["cache_key"], "path:/tmp/chair.glb");
        assert_eq!(commands[1]["select"], true);
    }

    #[test]
    fn scene_sequence_is_strictly_monotonic() {
        let first = next_scene_sequence();
        let second = next_scene_sequence();
        assert!(second > first);
    }

    #[test]
    fn scene_command_waits_for_matching_status_sequence() {
        let root = unique_test_dir("scene_bridge");
        fs::create_dir_all(&root).expect("create temp dir");
        let command_path = root.join("scene_commands.json");
        let status_path = command_path.with_extension("status.json");
        let config = ServerConfig {
            scene_control_path: Some(command_path.clone()),
            scene_status_path: Some(status_path.clone()),
            scene_timeout: Duration::from_secs(1),
            ..ServerConfig::from_args(ServerArgs::parse_from(["burn_synth_mcp"]))
        };
        let server = McpServer::new(config);
        let status_path_for_thread = status_path.clone();
        let command_path_for_thread = command_path.clone();
        let handle = thread::spawn(move || {
            let started = Instant::now();
            loop {
                if command_path_for_thread.exists() {
                    let command = read_scene_status(&command_path_for_thread)
                        .expect("command JSON should parse");
                    let sequence = command["sequence"].as_u64().expect("sequence");
                    atomic_write_json(
                        &status_path_for_thread,
                        &json!({
                            "last_sequence": sequence,
                            "ok": true,
                            "cache_entries": [],
                            "world_items": [],
                            "camera": null,
                            "screenshots": [],
                        }),
                    )
                    .expect("write status");
                    return;
                }
                assert!(started.elapsed() < Duration::from_secs(1));
                thread::sleep(Duration::from_millis(10));
            }
        });

        let response = server
            .send_scene_commands(vec![json!({ "type": "clear_selection" })])
            .expect("scene command should be acknowledged");
        handle.join().expect("status writer thread");
        assert_eq!(response["acknowledged"], true);
        assert!(response["status"]["last_sequence"].as_u64().is_some());
        fs::remove_dir_all(root).expect("remove temp dir");
    }

    fn unique_test_dir(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "burn_synth_mcp_{label}_{}_{}",
            std::process::id(),
            nanos
        ))
    }
}

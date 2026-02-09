#![recursion_limit = "256"]

use std::borrow::Cow;
use std::fs;
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use burn_synth::{
    ForegroundRequest, ImageSource, Mesh, MeshRequest, ModelSelection, RuntimeConfig, SynthRuntime,
};
use clap::{Parser, ValueEnum};
use image::ImageEncoder;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

const DEFAULT_PROTOCOL_VERSION: &str = "2025-06-18";
const DEFAULT_NUM_STEPS: usize = 50;
const DEFAULT_NUM_TOKENS: usize = 2048;
const DEFAULT_GUIDANCE_SCALE: f32 = 7.0;

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
pub enum MeshOutputFormat {
    Obj,
    Gltf,
    Glb,
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

    #[arg(long)]
    pub bg_weights_root: Option<PathBuf>,

    #[arg(long)]
    pub num_steps: Option<usize>,

    #[arg(long)]
    pub num_tokens: Option<usize>,

    #[arg(long)]
    pub guidance_scale: Option<f32>,
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
    pub bg_weights_root: Option<PathBuf>,
    pub num_steps: usize,
    pub num_tokens: usize,
    pub guidance_scale: f32,
}

impl ServerConfig {
    pub fn from_args(args: ServerArgs) -> Self {
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
            bg_weights_root: args.bg_weights_root,
            num_steps: args.num_steps.unwrap_or(DEFAULT_NUM_STEPS),
            num_tokens: args.num_tokens.unwrap_or(DEFAULT_NUM_TOKENS),
            guidance_scale: args.guidance_scale.unwrap_or(DEFAULT_GUIDANCE_SCALE),
        }
    }

    fn runtime_config(&self) -> RuntimeConfig {
        RuntimeConfig {
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
        }
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

    loop {
        let Some(message) = read_framed_json(&mut reader).map_err(|err| err.to_string())? else {
            break;
        };
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
        let output_path = resolve_glb_output_path(args.output_mesh_path, &input_path);
        ensure_parent_dir(&output_path).map_err(|err| err.to_string())?;

        let selected_rmbg = args.rmbg_model.unwrap_or(self.config.default_rmbg_model);
        let selected_backend = args.backend.unwrap_or(self.config.default_backend);
        let selected_synthesis_models = args
            .synthesis_models
            .map(sanitize_synthesis_models)
            .unwrap_or_else(|| self.config.default_synthesis_models.clone());

        let mut mesh_output = self
            .runtime
            .synthesize_mesh(MeshRequest {
                image: ImageSource::from_path(input_path.clone()),
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
            })
            .map_err(|err| err.to_string())?;

        mesh_output.mesh = apply_mesh_decimation(mesh_output.mesh, args.target_faces)
            .map_err(|err| format!("mesh decimation failed: {err}"))?;
        let vertices = mesh_output.mesh.vertices.len();
        let faces = mesh_output.mesh.faces.len();
        write_glb(&output_path, &mesh_output.mesh).map_err(|err| {
            format!(
                "failed to write GLB output {}: {err}",
                output_path.display()
            )
        })?;
        let material = mesh_output.mesh.material;

        Ok(json!({
            "tool": "image_to_mesh",
            "input_image_path": input_path.display().to_string(),
            "output_mesh_path": output_path.display().to_string(),
            "output_format": "glb",
            "vertices": vertices,
            "faces": faces,
            "target_faces": args.target_faces.filter(|value| *value > 0),
            "material": material.map(|value| {
                json!({
                    "base_color": value.base_color,
                    "metallic": value.metallic,
                    "roughness": value.roughness,
                    "alpha": value.alpha,
                })
            }),
            "rmbg_model": selected_rmbg.as_str(),
            "synthesis_models": selected_synthesis_models.iter().map(|m| m.as_str()).collect::<Vec<_>>(),
            "backend": selected_backend.as_str(),
            "dry_run": args.dry_run,
        }))
    }
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

fn resolve_glb_output_path(output_mesh_path: Option<PathBuf>, input_path: &Path) -> PathBuf {
    let Some(path) = output_mesh_path else {
        return default_output_path(input_path, "_mesh", "glb");
    };
    if path.extension().is_none() || path.is_dir() {
        let stem = input_path
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("mesh");
        return path.join(format!("{stem}_mesh.glb"));
    }
    if path
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| value.eq_ignore_ascii_case("glb"))
        .unwrap_or(false)
    {
        path
    } else {
        path.with_extension("glb")
    }
}

#[derive(Clone, Debug)]
struct MeshBinaryLayout {
    buffer: Vec<u8>,
    positions_byte_offset: usize,
    positions_byte_length: usize,
    indices_byte_offset: usize,
    indices_byte_length: usize,
    uvs_byte_offset: Option<usize>,
    uvs_byte_length: Option<usize>,
    base_color_image_view: Option<(usize, usize)>,
    metallic_roughness_image_view: Option<(usize, usize)>,
    normal_image_view: Option<(usize, usize)>,
    emissive_image_view: Option<(usize, usize)>,
    occlusion_image_view: Option<(usize, usize)>,
    min: [f32; 3],
    max: [f32; 3],
}

fn write_glb(path: &Path, mesh: &Mesh) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let glb = mesh_to_glb_bytes(mesh)?;
    fs::write(path, glb)?;
    Ok(())
}

fn mesh_to_glb_bytes(mesh: &Mesh) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let layout = build_mesh_binary_layout(mesh)?;
    let gltf = gltf_json(mesh, &layout);
    let json_bytes = serde_json::to_vec(&gltf)?;
    let glb = gltf::Glb {
        header: gltf::binary::Header {
            magic: *b"glTF",
            version: 2,
            length: 0,
        },
        json: Cow::Owned(json_bytes),
        bin: Some(Cow::Owned(layout.buffer)),
    }
    .to_vec()?;
    Ok(glb)
}

fn build_mesh_binary_layout(mesh: &Mesh) -> Result<MeshBinaryLayout, Box<dyn std::error::Error>> {
    if mesh.vertices.is_empty() {
        return Err(Box::new(io::Error::new(
            io::ErrorKind::InvalidInput,
            "cannot export empty mesh",
        )));
    }

    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for vertex in &mesh.vertices {
        for axis in 0..3 {
            min[axis] = min[axis].min(vertex[axis]);
            max[axis] = max[axis].max(vertex[axis]);
        }
    }

    let mut buffer = Vec::with_capacity(mesh.vertices.len() * 12 + mesh.faces.len() * 12 + 8192);
    let positions_byte_offset = buffer.len();
    for vertex in &mesh.vertices {
        for component in vertex {
            buffer.extend_from_slice(&component.to_le_bytes());
        }
    }
    let positions_byte_length = buffer.len();

    let mut uvs_byte_offset = None;
    let mut uvs_byte_length = None;
    if mesh.uvs.len() == mesh.vertices.len() && !mesh.uvs.is_empty() {
        pad_buffer_4(&mut buffer);
        let offset = buffer.len();
        for uv in &mesh.uvs {
            buffer.extend_from_slice(&uv[0].to_le_bytes());
            buffer.extend_from_slice(&uv[1].to_le_bytes());
        }
        uvs_byte_offset = Some(offset);
        uvs_byte_length = Some(buffer.len() - offset);
    }

    pad_buffer_4(&mut buffer);
    let indices_byte_offset = buffer.len();
    for face in &mesh.faces {
        for index in face {
            buffer.extend_from_slice(&index.to_le_bytes());
        }
    }
    let indices_byte_length = buffer.len() - indices_byte_offset;

    let mut base_color_image_view = None;
    let mut metallic_roughness_image_view = None;
    let mut normal_image_view = None;
    let mut emissive_image_view = None;
    let mut occlusion_image_view = None;
    if let Some(pbr) = mesh.pbr_textures.as_ref() {
        let base_png = encode_rgba_texture_png(&pbr.base_color)?;
        let mr_png = encode_rgba_texture_png(&pbr.metallic_roughness)?;
        pad_buffer_4(&mut buffer);
        let base_offset = buffer.len();
        buffer.extend_from_slice(base_png.as_slice());
        base_color_image_view = Some((base_offset, base_png.len()));
        pad_buffer_4(&mut buffer);
        let mr_offset = buffer.len();
        buffer.extend_from_slice(mr_png.as_slice());
        metallic_roughness_image_view = Some((mr_offset, mr_png.len()));
        if let Some(normal) = pbr.normal.as_ref() {
            let normal_png = encode_rgba_texture_png(normal)?;
            pad_buffer_4(&mut buffer);
            let normal_offset = buffer.len();
            buffer.extend_from_slice(normal_png.as_slice());
            normal_image_view = Some((normal_offset, normal_png.len()));
        }
        if let Some(emissive) = pbr.emissive.as_ref() {
            let emissive_png = encode_rgba_texture_png(emissive)?;
            pad_buffer_4(&mut buffer);
            let emissive_offset = buffer.len();
            buffer.extend_from_slice(emissive_png.as_slice());
            emissive_image_view = Some((emissive_offset, emissive_png.len()));
        }
        if let Some(occlusion) = pbr.occlusion.as_ref() {
            let occlusion_png = encode_rgba_texture_png(occlusion)?;
            pad_buffer_4(&mut buffer);
            let occlusion_offset = buffer.len();
            buffer.extend_from_slice(occlusion_png.as_slice());
            occlusion_image_view = Some((occlusion_offset, occlusion_png.len()));
        }
    }

    Ok(MeshBinaryLayout {
        buffer,
        positions_byte_offset,
        positions_byte_length,
        indices_byte_offset,
        indices_byte_length,
        uvs_byte_offset,
        uvs_byte_length,
        base_color_image_view,
        metallic_roughness_image_view,
        normal_image_view,
        emissive_image_view,
        occlusion_image_view,
        min,
        max,
    })
}

fn gltf_json(mesh: &Mesh, layout: &MeshBinaryLayout) -> Value {
    let mut primitive = json!({
        "attributes": {
            "POSITION": 0
        },
        "indices": 1,
        "mode": 4
    });
    if mesh.uvs.len() == mesh.vertices.len() && !mesh.uvs.is_empty() {
        primitive["attributes"]["TEXCOORD_0"] = json!(2);
    }

    let buffers = vec![json!({
        "byteLength": layout.buffer.len(),
    })];

    let mut buffer_views = Vec::new();
    buffer_views.push(json!({
        "buffer": 0,
        "byteOffset": layout.positions_byte_offset,
        "byteLength": layout.positions_byte_length,
        "target": 34962
    }));
    buffer_views.push(json!({
        "buffer": 0,
        "byteOffset": layout.indices_byte_offset,
        "byteLength": layout.indices_byte_length,
        "target": 34963
    }));
    if let (Some(uv_offset), Some(uv_len)) = (layout.uvs_byte_offset, layout.uvs_byte_length) {
        buffer_views.push(json!({
            "buffer": 0,
            "byteOffset": uv_offset,
            "byteLength": uv_len,
            "target": 34962
        }));
    }

    let mut accessors = Vec::new();
    accessors.push(json!({
        "bufferView": 0,
        "componentType": 5126,
        "count": mesh.vertices.len(),
        "type": "VEC3",
        "min": layout.min,
        "max": layout.max
    }));
    accessors.push(json!({
        "bufferView": 1,
        "componentType": 5125,
        "count": mesh.faces.len() * 3,
        "type": "SCALAR"
    }));
    if mesh.uvs.len() == mesh.vertices.len() && !mesh.uvs.is_empty() {
        accessors.push(json!({
            "bufferView": 2,
            "componentType": 5126,
            "count": mesh.uvs.len(),
            "type": "VEC2"
        }));
    }

    let mut images = Vec::new();
    let mut textures = Vec::new();
    let mut materials = Vec::new();
    let mut pbr_mr = json!({});
    let mut push_texture_image = |byte_offset: usize, byte_length: usize| -> usize {
        let view_index = buffer_views.len();
        buffer_views.push(json!({
            "buffer": 0,
            "byteOffset": byte_offset,
            "byteLength": byte_length
        }));
        let image_index = images.len();
        images.push(json!({
            "bufferView": view_index,
            "mimeType": "image/png"
        }));
        let texture_index = textures.len();
        textures.push(json!({ "source": image_index }));
        texture_index
    };
    if let Some(material) = mesh.material {
        pbr_mr = json!({
            "baseColorFactor": [
                material.base_color[0],
                material.base_color[1],
                material.base_color[2],
                material.alpha.clamp(0.0, 1.0)
            ],
            "metallicFactor": material.metallic.clamp(0.0, 1.0),
            "roughnessFactor": material.roughness.clamp(0.0, 1.0)
        });
    }
    if let Some((base_offset, base_len)) = layout.base_color_image_view {
        let texture_index = push_texture_image(base_offset, base_len);
        pbr_mr["baseColorTexture"] = json!({ "index": texture_index });
    }
    if let Some((mr_offset, mr_len)) = layout.metallic_roughness_image_view {
        let texture_index = push_texture_image(mr_offset, mr_len);
        pbr_mr["metallicRoughnessTexture"] = json!({ "index": texture_index });
    }

    if mesh.material.is_some() || mesh.pbr_textures.is_some() {
        let alpha = mesh
            .material
            .map(|value| value.alpha)
            .unwrap_or(1.0)
            .clamp(0.0, 1.0);
        let material_index = materials.len();
        let mut material = json!({
            "pbrMetallicRoughness": pbr_mr,
            "alphaMode": if alpha < 0.995 { "BLEND" } else { "OPAQUE" },
            "doubleSided": true
        });
        if let Some((normal_offset, normal_len)) = layout.normal_image_view {
            let texture_index = push_texture_image(normal_offset, normal_len);
            material["normalTexture"] = json!({ "index": texture_index });
        }
        if let Some((emissive_offset, emissive_len)) = layout.emissive_image_view {
            let texture_index = push_texture_image(emissive_offset, emissive_len);
            material["emissiveTexture"] = json!({ "index": texture_index });
            material["emissiveFactor"] = json!([1.0, 1.0, 1.0]);
        }
        if let Some((occlusion_offset, occlusion_len)) = layout.occlusion_image_view {
            let texture_index = push_texture_image(occlusion_offset, occlusion_len);
            material["occlusionTexture"] = json!({ "index": texture_index });
        }
        materials.push(material);
        primitive["material"] = json!(material_index);
    }

    let mut gltf = json!({
        "asset": {
            "version": "2.0",
            "generator": "burn_synth_mcp"
        },
        "scene": 0,
        "scenes": [
            { "nodes": [0] }
        ],
        "nodes": [
            { "mesh": 0 }
        ],
        "meshes": [
            {
                "primitives": [
                    primitive
                ]
            }
        ],
        "buffers": buffers,
        "bufferViews": buffer_views,
        "accessors": accessors
    });
    if !materials.is_empty() {
        gltf["materials"] = Value::Array(materials);
    }
    if !images.is_empty() {
        gltf["images"] = Value::Array(images);
    }
    if !textures.is_empty() {
        gltf["textures"] = Value::Array(textures);
    }
    gltf
}

fn pad_buffer_4(buffer: &mut Vec<u8>) {
    while !buffer.len().is_multiple_of(4) {
        buffer.push(0);
    }
}

fn encode_rgba_texture_png(
    texture: &burn_synth::MeshTexture,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let expected = texture.width as usize * texture.height as usize * 4;
    if texture.rgba8.len() != expected {
        return Err(Box::new(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "texture byte length mismatch: expected {}, got {}",
                expected,
                texture.rgba8.len()
            ),
        )));
    }
    let mut out = Vec::new();
    let encoder = image::codecs::png::PngEncoder::new(&mut out);
    encoder.write_image(
        texture.rgba8.as_slice(),
        texture.width,
        texture.height,
        image::ColorType::Rgba8.into(),
    )?;
    Ok(out)
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
                    "synthesis_models": { "type": "array", "items": { "type": "string", "enum": ["triposg", "trellis"] }, "description": "Optional synthesis model list override, ordered by preference." },
                    "backend": { "type": "string", "enum": ["cpu", "wgpu", "cuda"], "description": "Optional backend override." },
                    "target_faces": { "type": "integer", "description": "Optional target face count for mesh simplification." },
                    "dry_run": { "type": "boolean", "description": "Skip model inference and emit a canonical cube mesh." }
                },
                "required": ["input_image_path"],
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

impl From<TrellisQuality> for burn_synth::trellis::TrellisQuality {
    fn from(value: TrellisQuality) -> Self {
        match value {
            TrellisQuality::Low => Self::Low,
            TrellisQuality::Medium => Self::Medium,
            TrellisQuality::High => Self::High,
        }
    }
}

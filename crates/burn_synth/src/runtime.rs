use std::fmt::{Display, Formatter};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use burn::backend::NdArray;
use burn::prelude::Backend;
use burn_foreground::pipeline::{
    PrepareImageConfig, PreparedImageData, RmbgPipeline, prepare_image_data,
};
use burn_foreground::rmbg2::Rmbg2Pipeline;
use burn_foreground::rmbg2::import::resolve_rmbg2_weights_root;
use burn_foreground::rmbg14::import::resolve_rmbg_weights_root;
use burn_trellis::TrellisQuality;
use burn_trellis::pipeline::{
    Trellis2Pipeline, Trellis2PipelineConfig, TrellisDevice, TrellisRunOptions,
};
use burn_tripo::paths::resolve_triposg_weights_root;
use burn_tripo::pipeline::geometry::FlashExtractConfig;
use burn_tripo::pipeline::triposg::TripoSGPipeline;
use image::{ImageFormat, RgbaImage};

use crate::io::ImageSource;
use crate::mesh::Mesh;
use crate::pipeline::{ForegroundModel, ModelSelection, SynthesisModel, sanitize_synthesis_models};

const DEFAULT_BOUNDS: [f32; 6] = [-1.005, -1.005, -1.005, 1.005, 1.005, 1.005];
const DEFAULT_NUM_STEPS: usize = 50;
const DEFAULT_NUM_TOKENS: usize = 2048;
const DEFAULT_GUIDANCE_SCALE: f32 = 7.0;
const DEFAULT_FLASH_OCTREE_DEPTH: usize = 9;
const DEFAULT_FLASH_MIN_RESOLUTION: usize = 63;
const DEFAULT_FLASH_MINI_GRID_NUM: usize = 4;
const DEFAULT_FLASH_NUM_CHUNKS: usize = 10_000;
const DEFAULT_SEED: u64 = 42;

#[cfg(feature = "wgpu")]
type WgpuBackend = burn_wgpu::Wgpu<f32, i32, u32>;

#[cfg(feature = "cuda")]
type CudaBackend = burn_cuda::Cuda<f32, i32, u32>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InferenceBackend {
    Cpu,
    Wgpu,
    Cuda,
}

impl Default for InferenceBackend {
    fn default() -> Self {
        Self::Wgpu
    }
}

impl InferenceBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Wgpu => "wgpu",
            Self::Cuda => "cuda",
        }
    }
}

#[derive(Clone, Debug)]
pub struct RuntimeConfig {
    pub model_selection: ModelSelection,
    pub backend: InferenceBackend,
    /// TripoSG weights root.
    pub weights_root: Option<PathBuf>,
    /// Trellis2 weights root.
    pub trellis_weights_root: Option<PathBuf>,
    /// Optional local root for TRELLIS-image-large assets.
    pub trellis_image_large_root: Option<PathBuf>,
    /// Legacy field retained for CLI compatibility; ignored by Trellis2 Rust runtime.
    pub trellis_python_bin: Option<PathBuf>,
    /// Legacy field retained for CLI compatibility; ignored by Trellis2 Rust runtime.
    pub trellis_bridge_script: Option<PathBuf>,
    /// Trellis high-level quality selection.
    pub trellis_quality: TrellisQuality,
    pub bg_weights_root: Option<PathBuf>,
    pub num_steps: usize,
    pub num_tokens: usize,
    pub guidance_scale: f32,
    pub seed: Option<u64>,
    pub flash_extract: FlashExtractConfig,
    pub mesh_prepare: PrepareImageConfig,
    pub foreground_prepare: PrepareImageConfig,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            model_selection: ModelSelection::default(),
            backend: InferenceBackend::default(),
            weights_root: None,
            trellis_weights_root: None,
            trellis_image_large_root: None,
            trellis_python_bin: None,
            trellis_bridge_script: None,
            trellis_quality: TrellisQuality::Medium,
            bg_weights_root: None,
            num_steps: DEFAULT_NUM_STEPS,
            num_tokens: DEFAULT_NUM_TOKENS,
            guidance_scale: DEFAULT_GUIDANCE_SCALE,
            seed: Some(DEFAULT_SEED),
            flash_extract: default_flash_config(),
            mesh_prepare: PrepareImageConfig::default(),
            foreground_prepare: PrepareImageConfig {
                max_dimension: usize::MAX,
                ..PrepareImageConfig::default()
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct ForegroundRequest {
    pub image: ImageSource,
    pub model: Option<ForegroundModel>,
}

impl ForegroundRequest {
    pub fn from_image(image: ImageSource) -> Self {
        Self { image, model: None }
    }
}

#[derive(Debug)]
pub struct ForegroundOutput {
    pub image: RgbaImage,
    pub width: u32,
    pub height: u32,
    pub model: ForegroundModel,
}

#[derive(Debug, Clone)]
pub struct MeshRequest {
    pub image: ImageSource,
    pub foreground_model: Option<ForegroundModel>,
    pub synthesis_models: Option<Vec<SynthesisModel>>,
    pub backend: Option<InferenceBackend>,
    pub dry_run: bool,
}

impl MeshRequest {
    pub fn from_image(image: ImageSource) -> Self {
        Self {
            image,
            foreground_model: None,
            synthesis_models: None,
            backend: None,
            dry_run: false,
        }
    }
}

#[derive(Debug)]
pub struct MeshOutput {
    pub mesh: Mesh,
    pub foreground_model: ForegroundModel,
    pub synthesis_models: Vec<SynthesisModel>,
    pub synthesis_backend: SynthesisModel,
    pub backend: InferenceBackend,
}

#[derive(Debug, Clone)]
pub struct RuntimeError {
    message: String,
}

impl RuntimeError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Display for RuntimeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for RuntimeError {}

type RuntimeResult<T> = Result<T, RuntimeError>;

pub struct SynthRuntime {
    config: RuntimeConfig,
    foreground: ForegroundRuntime,
    synthesis: SynthesisRuntime,
}

impl SynthRuntime {
    pub fn new(config: RuntimeConfig) -> Self {
        Self {
            config,
            foreground: ForegroundRuntime::default(),
            synthesis: SynthesisRuntime::default(),
        }
    }

    pub fn config(&self) -> &RuntimeConfig {
        &self.config
    }

    pub fn extract_foreground(
        &mut self,
        request: ForegroundRequest,
    ) -> RuntimeResult<ForegroundOutput> {
        let selected_model = request
            .model
            .unwrap_or(self.config.model_selection.foreground_model);
        let materialized = MaterializedImageInput::from_source(&request.image)?;
        let source = image::open(materialized.path())
            .map_err(|err| RuntimeError::new(format!("failed to open input image: {err}")))?
            .to_rgba8();
        let (width, height) = source.dimensions();
        let alpha_mask = self.compute_alpha_mask(materialized.path(), selected_model)?;
        let expected = width as usize * height as usize;
        if alpha_mask.len() != expected {
            return Err(RuntimeError::new(format!(
                "foreground mask size mismatch: expected {expected}, got {}",
                alpha_mask.len()
            )));
        }

        let mut output = source;
        for (idx, pixel) in output.pixels_mut().enumerate() {
            let alpha = (alpha_mask[idx].clamp(0.0, 1.0) * 255.0).round() as u8;
            pixel.0[3] = alpha;
        }

        Ok(ForegroundOutput {
            image: output,
            width,
            height,
            model: selected_model,
        })
    }

    pub fn synthesize_mesh(&mut self, request: MeshRequest) -> RuntimeResult<MeshOutput> {
        let selected_foreground = request
            .foreground_model
            .unwrap_or(self.config.model_selection.foreground_model);
        let selected_synthesis = request
            .synthesis_models
            .map(sanitize_synthesis_models)
            .unwrap_or_else(|| self.config.model_selection.synthesis_models.clone());
        let selected_backend = request.backend.unwrap_or(self.config.backend);
        let preferred_synthesis = selected_synthesis
            .first()
            .copied()
            .unwrap_or(SynthesisModel::Triposg);

        let (mesh, synthesis_backend) = if request.dry_run {
            (canonical_cube_mesh(), preferred_synthesis)
        } else {
            let materialized = MaterializedImageInput::from_source(&request.image)?;
            self.infer_mesh(
                materialized.path(),
                selected_foreground,
                selected_backend,
                &selected_synthesis,
            )?
        };

        Ok(MeshOutput {
            mesh,
            foreground_model: selected_foreground,
            synthesis_models: selected_synthesis,
            synthesis_backend,
            backend: selected_backend,
        })
    }

    fn infer_mesh(
        &mut self,
        input_image_path: &Path,
        foreground_model: ForegroundModel,
        backend: InferenceBackend,
        synthesis_models: &[SynthesisModel],
    ) -> RuntimeResult<(Mesh, SynthesisModel)> {
        let preferred = synthesis_models
            .first()
            .copied()
            .unwrap_or(SynthesisModel::Triposg);

        match preferred {
            SynthesisModel::Triposg => {
                match self.infer_mesh_triposg(input_image_path, foreground_model, backend) {
                    Ok(mesh) => return Ok((mesh, SynthesisModel::Triposg)),
                    Err(err) if synthesis_models.contains(&SynthesisModel::Trellis) => {
                        eprintln!(
                            "burn_synth runtime: TripoSG failed ({err}); falling back to Trellis2."
                        );
                        match self.infer_mesh_trellis(input_image_path, foreground_model, backend) {
                            Ok(mesh) => return Ok((mesh, SynthesisModel::Trellis)),
                            Err(trellis_err) => {
                                return Err(RuntimeError::new(format!(
                                    "TripoSG failed ({err}); Trellis2 fallback failed ({trellis_err})"
                                )));
                            }
                        }
                    }
                    Err(err) => return Err(err),
                }
            }
            SynthesisModel::Trellis => {
                match self.infer_mesh_trellis(input_image_path, foreground_model, backend) {
                    Ok(mesh) => return Ok((mesh, SynthesisModel::Trellis)),
                    Err(err) if synthesis_models.contains(&SynthesisModel::Triposg) => {
                        eprintln!(
                            "burn_synth runtime: Trellis2 failed ({err}); falling back to TripoSG."
                        );
                        match self.infer_mesh_triposg(input_image_path, foreground_model, backend) {
                            Ok(mesh) => return Ok((mesh, SynthesisModel::Triposg)),
                            Err(triposg_err) => {
                                return Err(RuntimeError::new(format!(
                                    "Trellis2 failed ({err}); TripoSG fallback failed ({triposg_err})"
                                )));
                            }
                        }
                    }
                    Err(err) => return Err(err),
                }
            }
        }
    }

    fn infer_mesh_triposg(
        &mut self,
        input_image_path: &Path,
        foreground_model: ForegroundModel,
        backend: InferenceBackend,
    ) -> RuntimeResult<Mesh> {
        let prepared = self.prepare_image_for_mesh(input_image_path, foreground_model)?;
        match backend {
            InferenceBackend::Cpu => {
                let state = self.synthesis.ensure_cpu(&self.config)?;
                run_backend_inference(state, &prepared, &self.config)
            }
            InferenceBackend::Wgpu => {
                #[cfg(feature = "wgpu")]
                {
                    let state = self.synthesis.ensure_wgpu(&self.config)?;
                    run_backend_inference(state, &prepared, &self.config)
                }
                #[cfg(not(feature = "wgpu"))]
                {
                    Err(RuntimeError::new(
                        "wgpu backend not enabled; build with burn_synth feature `wgpu`",
                    ))
                }
            }
            InferenceBackend::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    let state = self.synthesis.ensure_cuda(&self.config)?;
                    run_backend_inference(state, &prepared, &self.config)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err(RuntimeError::new(
                        "cuda backend not enabled; build with burn_synth feature `cuda`",
                    ))
                }
            }
        }
    }

    fn infer_mesh_trellis(
        &mut self,
        input_image_path: &Path,
        foreground_model: ForegroundModel,
        backend: InferenceBackend,
    ) -> RuntimeResult<Mesh> {
        let prepared = self.extract_foreground(ForegroundRequest {
            image: ImageSource::from_path(input_image_path.to_path_buf()),
            model: Some(foreground_model),
        })?;
        let temp_input = unique_temp_png_path();
        prepared.image.save(&temp_input).map_err(|err| {
            RuntimeError::new(format!(
                "failed to persist Trellis input image {}: {err}",
                temp_input.display()
            ))
        })?;

        let pipeline = self.synthesis.ensure_trellis(&self.config)?;
        let trellis_device = match backend {
            InferenceBackend::Cpu => TrellisDevice::Cpu,
            InferenceBackend::Wgpu => TrellisDevice::Wgpu,
            InferenceBackend::Cuda => TrellisDevice::Cuda,
        };
        let options = TrellisRunOptions {
            quality: self.config.trellis_quality,
            device: trellis_device,
            seed: self.config.seed,
            hook_output: None,
        };

        let mesh = pipeline
            .infer_mesh(&temp_input, &options)
            .map_err(|err| RuntimeError::new(format!("Trellis2 inference failed: {err}")))?;
        let _ = std::fs::remove_file(temp_input);
        Ok(mesh.into())
    }

    fn compute_alpha_mask(
        &mut self,
        input_path: &Path,
        selected_model: ForegroundModel,
    ) -> RuntimeResult<Vec<f32>> {
        if let Ok(prepared) =
            prepare_image_data::<NdArray<f32>>(input_path, None, &self.config.foreground_prepare)
            && let Some(alpha) = prepared.alpha_mask
        {
            return Ok(alpha);
        }

        match selected_model {
            ForegroundModel::Rmbg14 => {
                let root = resolve_foreground_weights_root(
                    self.config.bg_weights_root.as_deref(),
                    selected_model,
                );
                let pipeline = self.foreground.ensure_rmbg14(&root)?;
                let prepared =
                    prepare_image_data(input_path, Some(pipeline), &self.config.foreground_prepare)
                        .map_err(|err| RuntimeError::new(format!("RMBG-1.4 failed: {err}")))?;
                prepared
                    .alpha_mask
                    .ok_or_else(|| RuntimeError::new("RMBG-1.4 did not produce an alpha mask"))
            }
            ForegroundModel::Rmbg2 => {
                let root = resolve_foreground_weights_root(
                    self.config.bg_weights_root.as_deref(),
                    selected_model,
                );
                let pipeline = self.foreground.ensure_rmbg2(&root)?;
                let prepared = pipeline
                    .prepare_image_data(input_path, &self.config.foreground_prepare)
                    .map_err(|err| RuntimeError::new(format!("RMBG-2.0 failed: {err}")))?;
                prepared
                    .alpha_mask
                    .ok_or_else(|| RuntimeError::new("RMBG-2.0 did not produce an alpha mask"))
            }
        }
    }

    fn prepare_image_for_mesh(
        &mut self,
        input_path: &Path,
        selected_model: ForegroundModel,
    ) -> RuntimeResult<PreparedImageData> {
        if let Ok(prepared) =
            prepare_image_data::<NdArray<f32>>(input_path, None, &self.config.mesh_prepare)
        {
            return Ok(prepared);
        }

        match selected_model {
            ForegroundModel::Rmbg14 => {
                let root = resolve_foreground_weights_root(
                    self.config.bg_weights_root.as_deref(),
                    selected_model,
                );
                let pipeline = self.foreground.ensure_rmbg14(&root)?;
                prepare_image_data(input_path, Some(pipeline), &self.config.mesh_prepare).map_err(
                    |err| RuntimeError::new(format!("RMBG-1.4 preprocessing failed: {err}")),
                )
            }
            ForegroundModel::Rmbg2 => {
                let root = resolve_foreground_weights_root(
                    self.config.bg_weights_root.as_deref(),
                    selected_model,
                );
                let pipeline = self.foreground.ensure_rmbg2(&root)?;
                pipeline
                    .prepare_image_data(input_path, &self.config.mesh_prepare)
                    .map_err(|err| {
                        RuntimeError::new(format!("RMBG-2.0 preprocessing failed: {err}"))
                    })
            }
        }
    }
}

#[derive(Default)]
struct ForegroundRuntime {
    rmbg14: Option<RmbgPipeline<NdArray<f32>>>,
    rmbg2: Option<Rmbg2Pipeline>,
}

impl ForegroundRuntime {
    fn ensure_rmbg14(&mut self, root: &Path) -> RuntimeResult<&RmbgPipeline<NdArray<f32>>> {
        if self.rmbg14.is_none() {
            let device = <NdArray<f32> as Backend>::Device::default();
            let pipeline = RmbgPipeline::from_pretrained(root, &device).map_err(|err| {
                RuntimeError::new(format!(
                    "failed to load RMBG-1.4 at {}: {err}",
                    root.display()
                ))
            })?;
            self.rmbg14 = Some(pipeline);
        }
        self.rmbg14
            .as_ref()
            .ok_or_else(|| RuntimeError::new("RMBG-1.4 pipeline unavailable"))
    }

    fn ensure_rmbg2(&mut self, root: &Path) -> RuntimeResult<&Rmbg2Pipeline> {
        if self.rmbg2.is_none() {
            let pipeline = Rmbg2Pipeline::from_pretrained(root).map_err(|err| {
                RuntimeError::new(format!(
                    "failed to load RMBG-2.0 at {}: {err}",
                    root.display()
                ))
            })?;
            self.rmbg2 = Some(pipeline);
        }
        self.rmbg2
            .as_ref()
            .ok_or_else(|| RuntimeError::new("RMBG-2.0 pipeline unavailable"))
    }
}

struct BackendSynthesisState<B: Backend> {
    device: B::Device,
    pipeline: TripoSGPipeline<B>,
}

#[derive(Default)]
struct SynthesisRuntime {
    cpu: Option<BackendSynthesisState<NdArray<f32>>>,
    #[cfg(feature = "wgpu")]
    wgpu: Option<BackendSynthesisState<WgpuBackend>>,
    #[cfg(feature = "cuda")]
    cuda: Option<BackendSynthesisState<CudaBackend>>,
    trellis: Option<Trellis2Pipeline>,
}

impl SynthesisRuntime {
    fn ensure_cpu(
        &mut self,
        config: &RuntimeConfig,
    ) -> RuntimeResult<&mut BackendSynthesisState<NdArray<f32>>> {
        if self.cpu.is_none() {
            self.cpu = Some(load_backend_state::<NdArray<f32>>(config)?);
        }
        self.cpu
            .as_mut()
            .ok_or_else(|| RuntimeError::new("CPU synthesis backend unavailable"))
    }

    fn ensure_trellis(&mut self, config: &RuntimeConfig) -> RuntimeResult<&mut Trellis2Pipeline> {
        if self.trellis.is_none() {
            let mut trellis_config = Trellis2PipelineConfig::default();
            if let Some(root) = config.trellis_weights_root.as_ref() {
                trellis_config.weights_root = root.clone();
            }
            trellis_config.image_large_root = config.trellis_image_large_root.clone();
            let pipeline = Trellis2Pipeline::new(trellis_config).map_err(|err| {
                RuntimeError::new(format!("failed to initialize Trellis2: {err}"))
            })?;
            pipeline
                .validate_runtime()
                .map_err(|err| RuntimeError::new(format!("Trellis2 runtime unavailable: {err}")))?;
            self.trellis = Some(pipeline);
        }
        self.trellis
            .as_mut()
            .ok_or_else(|| RuntimeError::new("Trellis2 synthesis backend unavailable"))
    }

    #[cfg(feature = "wgpu")]
    fn ensure_wgpu(
        &mut self,
        config: &RuntimeConfig,
    ) -> RuntimeResult<&mut BackendSynthesisState<WgpuBackend>> {
        if self.wgpu.is_none() {
            self.wgpu = Some(load_backend_state::<WgpuBackend>(config)?);
        }
        self.wgpu
            .as_mut()
            .ok_or_else(|| RuntimeError::new("WGPU synthesis backend unavailable"))
    }

    #[cfg(feature = "cuda")]
    fn ensure_cuda(
        &mut self,
        config: &RuntimeConfig,
    ) -> RuntimeResult<&mut BackendSynthesisState<CudaBackend>> {
        if self.cuda.is_none() {
            self.cuda = Some(load_backend_state::<CudaBackend>(config)?);
        }
        self.cuda
            .as_mut()
            .ok_or_else(|| RuntimeError::new("CUDA synthesis backend unavailable"))
    }
}

fn load_backend_state<B: Backend>(
    config: &RuntimeConfig,
) -> RuntimeResult<BackendSynthesisState<B>> {
    let device = B::Device::default();
    if let Some(seed) = config.seed {
        B::seed(&device, seed);
    }
    let weights_root = resolve_triposg_weights_root(config.weights_root.as_deref());
    let pipeline =
        TripoSGPipeline::<B>::from_pretrained(&weights_root, &device).map_err(|err| {
            RuntimeError::new(format!(
                "failed to load TripoSG weights at {}: {err}",
                weights_root.display()
            ))
        })?;
    Ok(BackendSynthesisState { device, pipeline })
}

fn run_backend_inference<B: Backend>(
    state: &mut BackendSynthesisState<B>,
    prepared: &PreparedImageData,
    config: &RuntimeConfig,
) -> RuntimeResult<Mesh> {
    if let Some(seed) = config.seed {
        B::seed(&state.device, seed);
    }
    let image = prepared.to_tensor::<B>(&state.device);
    let output = state
        .pipeline
        .sample_mesh_flash(
            image,
            config.num_steps,
            config.num_tokens,
            config.guidance_scale,
            &config.flash_extract,
            None,
        )
        .map_err(|err| RuntimeError::new(format!("TripoSG inference failed: {err}")))?;
    let mesh = output
        .mesh
        .ok_or_else(|| RuntimeError::new("inference returned an empty mesh"))?;
    Ok(mesh.into())
}

fn resolve_foreground_weights_root(explicit: Option<&Path>, model: ForegroundModel) -> PathBuf {
    if let Some(path) = explicit
        && let Some(root) = normalize_foreground_root(path, model)
    {
        return root;
    }
    match model {
        ForegroundModel::Rmbg14 => resolve_rmbg_weights_root(),
        ForegroundModel::Rmbg2 => resolve_rmbg2_weights_root(),
    }
}

fn normalize_foreground_root(path: &Path, model: ForegroundModel) -> Option<PathBuf> {
    if path.is_dir() {
        let nested = path.join(match model {
            ForegroundModel::Rmbg14 => "RMBG-1.4",
            ForegroundModel::Rmbg2 => "RMBG-2.0",
        });
        if nested.exists() {
            return Some(nested);
        }
        return Some(path.to_path_buf());
    }

    if path.is_file() {
        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("");
        if file_name == "model.safetensors" || file_name.ends_with(".bpk") {
            return path.parent().map(Path::to_path_buf);
        }
        if file_name.ends_with(".onnx") {
            let parent = path.parent()?;
            if parent.file_name().and_then(|name| name.to_str()) == Some("onnx") {
                return parent.parent().map(Path::to_path_buf);
            }
            return Some(parent.to_path_buf());
        }
    }

    None
}

fn canonical_cube_mesh() -> Mesh {
    let vertices = vec![
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ];
    let faces = vec![
        [0, 1, 2],
        [0, 2, 3],
        [4, 6, 5],
        [4, 7, 6],
        [0, 4, 5],
        [0, 5, 1],
        [1, 5, 6],
        [1, 6, 2],
        [2, 6, 7],
        [2, 7, 3],
        [3, 7, 4],
        [3, 4, 0],
    ];
    Mesh { vertices, faces }
}

fn default_flash_config() -> FlashExtractConfig {
    FlashExtractConfig {
        bounds: DEFAULT_BOUNDS,
        octree_depth: DEFAULT_FLASH_OCTREE_DEPTH,
        num_chunks: DEFAULT_FLASH_NUM_CHUNKS,
        mc_level: 0.0,
        min_resolution: DEFAULT_FLASH_MIN_RESOLUTION,
        mini_grid_num: DEFAULT_FLASH_MINI_GRID_NUM,
    }
}

struct MaterializedImageInput {
    path: PathBuf,
    cleanup: Option<PathBuf>,
}

impl MaterializedImageInput {
    fn from_source(source: &ImageSource) -> RuntimeResult<Self> {
        match source {
            ImageSource::Path(path) => Ok(Self {
                path: path.clone(),
                cleanup: None,
            }),
            ImageSource::Bytes(bytes) => {
                let decoded = image::load_from_memory(bytes).map_err(|err| {
                    RuntimeError::new(format!("failed to decode image bytes: {err}"))
                })?;
                let path = unique_temp_png_path();
                decoded
                    .save_with_format(&path, ImageFormat::Png)
                    .map_err(|err| {
                        RuntimeError::new(format!(
                            "failed to materialize image bytes at {}: {err}",
                            path.display()
                        ))
                    })?;
                Ok(Self {
                    path: path.clone(),
                    cleanup: Some(path),
                })
            }
        }
    }

    fn path(&self) -> &Path {
        self.path.as_path()
    }
}

impl Drop for MaterializedImageInput {
    fn drop(&mut self) {
        if let Some(path) = self.cleanup.as_ref() {
            let _ = std::fs::remove_file(path);
        }
    }
}

fn unique_temp_png_path() -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let counter = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!(
        "burn_synth_input_{}_{}_{}.png",
        std::process::id(),
        nanos,
        counter
    ))
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use image::{DynamicImage, Rgba};

    use super::*;

    #[test]
    fn foreground_passthrough_alpha_from_bytes() {
        let mut input = RgbaImage::new(10, 10);
        for y in 0..10 {
            for x in 0..10 {
                input.put_pixel(x, y, Rgba([120, 140, 200, 255]));
            }
        }
        input.put_pixel(0, 0, Rgba([120, 140, 200, 0]));

        let mut encoded = Cursor::new(Vec::<u8>::new());
        DynamicImage::ImageRgba8(input)
            .write_to(&mut encoded, ImageFormat::Png)
            .expect("failed to encode PNG");
        let bytes = encoded.into_inner();

        let mut runtime = SynthRuntime::new(RuntimeConfig {
            backend: InferenceBackend::Cpu,
            ..RuntimeConfig::default()
        });
        let output = runtime
            .extract_foreground(ForegroundRequest::from_image(ImageSource::from_bytes(
                bytes,
            )))
            .expect("foreground extraction should succeed");

        assert_eq!(output.width, 10);
        assert_eq!(output.height, 10);
        assert_eq!(output.image.get_pixel(0, 0).0[3], 0);
        assert_eq!(output.image.get_pixel(9, 9).0[3], 255);
    }

    #[test]
    fn mesh_dry_run_returns_canonical_cube() {
        let mut runtime = SynthRuntime::new(RuntimeConfig {
            backend: InferenceBackend::Cpu,
            ..RuntimeConfig::default()
        });
        let output = runtime
            .synthesize_mesh(MeshRequest {
                image: ImageSource::from_path("unused.png"),
                foreground_model: Some(ForegroundModel::Rmbg2),
                synthesis_models: Some(vec![SynthesisModel::Trellis]),
                backend: Some(InferenceBackend::Cpu),
                dry_run: true,
            })
            .expect("dry-run mesh should succeed");

        assert_eq!(output.mesh.vertices.len(), 8);
        assert_eq!(output.mesh.faces.len(), 12);
        assert_eq!(output.synthesis_models, vec![SynthesisModel::Trellis]);
        assert_eq!(output.backend, InferenceBackend::Cpu);
        assert_eq!(output.foreground_model, ForegroundModel::Rmbg2);
    }
}

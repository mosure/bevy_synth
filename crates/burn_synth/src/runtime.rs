use std::fmt::{Display, Formatter};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use burn::backend::NdArray;
use burn::prelude::Backend;
use burn::tensor::Tensor;
use burn_foreground::pipeline::{
    PrepareImageConfig, PreparedImageData, RmbgPipeline, prepare_image_data,
};
use burn_foreground::rmbg2::Rmbg2Pipeline;
use burn_foreground::rmbg2::import::resolve_rmbg2_weights_root;
#[cfg(target_arch = "wasm32")]
use burn_foreground::rmbg14::import::resolve_rmbg_weights_root;
use burn_foreground::rmbg14::set_rmbg_strict_interp_override;
#[cfg(feature = "trellis")]
use burn_trellis::pipeline::{
    Trellis2Pipeline, Trellis2PipelineConfig, TrellisDevice, TrellisRunOptions,
};
use burn_tripo::model::triposg::image_encoder::DinoImageProcessor;
use burn_tripo::model::triposg::image_encoder::import::{
    load_dinov2_processor, load_triposg_dinov2_with_policy,
};
use burn_tripo::paths::resolve_triposg_weights_root;
use burn_tripo::pipeline::geometry::FlashExtractConfig;
use burn_tripo::pipeline::mesh::{Mesh as TripoMesh, sdf_to_mesh_diff_dmc};
use burn_tripo::pipeline::runtime_parity::{
    DinoBackendChoice, decimate_tripo_mesh, should_use_cpu_dino_backend, triposg_runtime_profile,
};
use burn_tripo::pipeline::triposg::{
    TripoSGLoadOptions, TripoSGPipeline, TripoSGSamplerProgress, deterministic_latents_from_seed,
};
use image::{ImageFormat, RgbaImage};

use crate::io::ImageSource;
use crate::mesh::Mesh;
#[cfg(not(target_arch = "wasm32"))]
use crate::native_model_bootstrap::{
    resolve_or_bootstrap_rmbg14_root, resolve_or_bootstrap_triposg_root,
};
use crate::pipeline::{ForegroundModel, ModelSelection, SynthesisModel, sanitize_synthesis_models};
use crate::progress::{RuntimeProgressEvent, RuntimeProgressObserver};

const DEFAULT_BOUNDS: [f32; 6] = [-1.005, -1.005, -1.005, 1.005, 1.005, 1.005];
const DEFAULT_NUM_STEPS: usize = 50;
const DEFAULT_NUM_TOKENS: usize = 2048;
const DEFAULT_GUIDANCE_SCALE: f32 = 7.0;
const DEFAULT_FLASH_OCTREE_DEPTH: usize = 9;
const DEFAULT_FLASH_MIN_RESOLUTION: usize = 63;
const DEFAULT_FLASH_MINI_GRID_NUM: usize = 4;
const DEFAULT_FLASH_NUM_CHUNKS: usize = 10_000;
const DEFAULT_SEED: u64 = 42;
const DEFAULT_TARGET_FACES: usize = 10_000;

#[cfg(feature = "wgpu")]
type WgpuBackend = burn_wgpu::Wgpu<f32, i32, u32>;

#[cfg(feature = "cuda")]
type CudaBackend = burn_cuda::Cuda<f32, i32>;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum InferenceBackend {
    Cpu,
    #[default]
    Wgpu,
    Cuda,
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum DinoBackend {
    #[default]
    Auto,
    Cpu,
    Gpu,
}

impl DinoBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Gpu => "gpu",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum TrellisQuality {
    Low,
    #[default]
    Medium,
    High,
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
    pub dino_backend: DinoBackend,
    pub target_faces: Option<usize>,
    pub flash_extract: FlashExtractConfig,
    pub mesh_prepare: PrepareImageConfig,
    pub foreground_prepare: PrepareImageConfig,
    pub progress: RuntimeProgressObserver,
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
            dino_backend: DinoBackend::Auto,
            target_faces: Some(DEFAULT_TARGET_FACES),
            flash_extract: default_flash_config(),
            mesh_prepare: PrepareImageConfig::default(),
            foreground_prepare: PrepareImageConfig {
                max_dimension: usize::MAX,
                ..PrepareImageConfig::default()
            },
            progress: RuntimeProgressObserver::default(),
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

struct ProgressRun {
    observer: RuntimeProgressObserver,
    run: &'static str,
    started: Instant,
}

impl ProgressRun {
    fn new(observer: &RuntimeProgressObserver, run: &'static str, detail: Option<String>) -> Self {
        let this = Self {
            observer: observer.clone(),
            run,
            started: Instant::now(),
        };
        if this.observer.emits_stages() {
            this.observer
                .emit(RuntimeProgressEvent::RunStarted { run, detail });
        }
        this
    }

    fn stage_started(
        &self,
        stage: &'static str,
        total_steps: Option<usize>,
        detail: Option<String>,
    ) {
        if self.observer.emits_stages() {
            self.observer.emit(RuntimeProgressEvent::StageStarted {
                run: self.run,
                stage,
                total_steps,
                detail,
            });
        }
    }

    fn stage_completed(
        &self,
        stage: &'static str,
        total_steps: Option<usize>,
        elapsed_ms: f64,
        detail: Option<String>,
    ) {
        if self.observer.emits_stages() {
            self.observer.emit(RuntimeProgressEvent::StageCompleted {
                run: self.run,
                stage,
                total_steps,
                elapsed_ms,
                detail,
            });
        }
    }

    fn step(
        &self,
        stage: &'static str,
        progress: TripoSGSamplerProgress,
        elapsed_ms: f64,
        detail: Option<String>,
    ) {
        if !self
            .observer
            .should_emit_step(progress.step_index, progress.total_steps)
        {
            return;
        }
        let avg_step_ms = if progress.step_index > 0 {
            elapsed_ms / progress.step_index as f64
        } else {
            progress.step_ms
        };
        let remaining = progress.total_steps.saturating_sub(progress.step_index) as f64;
        let eta_ms = if remaining > 0.0 {
            Some(avg_step_ms * remaining)
        } else {
            Some(0.0)
        };
        self.observer.emit(RuntimeProgressEvent::Step {
            run: self.run,
            stage,
            step: progress.step_index,
            total_steps: progress.total_steps,
            step_ms: progress.step_ms,
            elapsed_ms,
            eta_ms,
            detail,
        });
    }

    fn warn(&self, message: impl Into<String>) {
        self.observer.emit(RuntimeProgressEvent::Warning {
            run: self.run,
            message: message.into(),
        });
    }

    fn complete(&self, detail: Option<String>) {
        if self.observer.emits_stages() {
            self.observer.emit(RuntimeProgressEvent::RunCompleted {
                run: self.run,
                elapsed_ms: self.started.elapsed().as_secs_f64() * 1000.0,
                detail,
            });
        }
    }
}

pub struct SynthRuntime {
    config: RuntimeConfig,
    foreground: ForegroundRuntime,
    synthesis: SynthesisRuntime,
}

impl SynthRuntime {
    pub fn new(config: RuntimeConfig) -> Self {
        let parity = triposg_runtime_profile(Some(config.mesh_prepare.max_dimension));
        set_rmbg_strict_interp_override(Some(parity.strict_rmbg_interp));
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
        let progress = ProgressRun::new(
            &self.config.progress,
            "foreground",
            Some(format!("model={}", foreground_model_label(selected_model))),
        );
        progress.stage_started("foreground.materialize_input", None, None);
        let materialize_start = Instant::now();
        let materialized = MaterializedImageInput::from_source(&request.image)?;
        progress.stage_completed(
            "foreground.materialize_input",
            None,
            materialize_start.elapsed().as_secs_f64() * 1000.0,
            Some(format!("path={}", materialized.path().display())),
        );
        progress.stage_started("foreground.load_image", None, None);
        let load_start = Instant::now();
        let source = image::open(materialized.path())
            .map_err(|err| RuntimeError::new(format!("failed to open input image: {err}")))?
            .to_rgba8();
        progress.stage_completed(
            "foreground.load_image",
            None,
            load_start.elapsed().as_secs_f64() * 1000.0,
            None,
        );
        let (width, height) = source.dimensions();
        progress.stage_started(
            "foreground.alpha_mask",
            None,
            Some(format!("model={}", foreground_model_label(selected_model))),
        );
        let alpha_start = Instant::now();
        let alpha_mask = self.compute_alpha_mask(materialized.path(), selected_model)?;
        progress.stage_completed(
            "foreground.alpha_mask",
            None,
            alpha_start.elapsed().as_secs_f64() * 1000.0,
            Some(format!("pixels={}", alpha_mask.len())),
        );
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

        let output = ForegroundOutput {
            image: output,
            width,
            height,
            model: selected_model,
        };
        progress.complete(Some(format!(
            "width={} height={} model={}",
            output.width,
            output.height,
            foreground_model_label(output.model)
        )));
        Ok(output)
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
        let progress = ProgressRun::new(
            &self.config.progress,
            "mesh",
            Some(format!(
                "foreground_model={} backend={} dino_backend={} target_faces={:?} synthesis_models={}",
                foreground_model_label(selected_foreground),
                selected_backend.as_str(),
                self.config.dino_backend.as_str(),
                self.config.target_faces,
                synthesis_models_label(&selected_synthesis)
            )),
        );

        let (mut mesh, synthesis_backend) = if request.dry_run {
            progress.stage_started("mesh.dry_run", None, None);
            let dry_start = Instant::now();
            let mesh = canonical_cube_mesh();
            progress.stage_completed(
                "mesh.dry_run",
                None,
                dry_start.elapsed().as_secs_f64() * 1000.0,
                Some(format!(
                    "vertices={} faces={}",
                    mesh.vertices.len(),
                    mesh.faces.len()
                )),
            );
            (mesh, preferred_synthesis)
        } else {
            let materialized = MaterializedImageInput::from_source(&request.image)?;
            self.infer_mesh(
                materialized.path(),
                selected_foreground,
                selected_backend,
                &selected_synthesis,
                &progress,
            )?
        };

        if !request.dry_run
            && matches!(synthesis_backend, SynthesisModel::Triposg)
            && self
                .config
                .target_faces
                .filter(|faces| *faces > 0)
                .is_some()
        {
            progress.stage_started(
                "mesh.decimate",
                None,
                Some(format!(
                    "target_faces={}",
                    self.config.target_faces.unwrap_or_default()
                )),
            );
            let decimate_start = Instant::now();
            let before_faces = mesh.faces.len();
            let before_vertices = mesh.vertices.len();
            mesh = decimate_mesh(mesh, self.config.target_faces)
                .map_err(|err| RuntimeError::new(format!("mesh decimation failed: {err}")))?;
            progress.stage_completed(
                "mesh.decimate",
                None,
                decimate_start.elapsed().as_secs_f64() * 1000.0,
                Some(format!(
                    "vertices={} faces={} (from vertices={} faces={})",
                    mesh.vertices.len(),
                    mesh.faces.len(),
                    before_vertices,
                    before_faces
                )),
            );
        }

        let output = MeshOutput {
            mesh,
            foreground_model: selected_foreground,
            synthesis_models: selected_synthesis,
            synthesis_backend,
            backend: selected_backend,
        };
        progress.complete(Some(format!(
            "vertices={} faces={} synthesis_backend={}",
            output.mesh.vertices.len(),
            output.mesh.faces.len(),
            synthesis_model_label(output.synthesis_backend)
        )));
        Ok(output)
    }

    fn infer_mesh(
        &mut self,
        input_image_path: &Path,
        foreground_model: ForegroundModel,
        backend: InferenceBackend,
        synthesis_models: &[SynthesisModel],
        progress: &ProgressRun,
    ) -> RuntimeResult<(Mesh, SynthesisModel)> {
        let preferred = synthesis_models
            .first()
            .copied()
            .unwrap_or(SynthesisModel::Triposg);

        match preferred {
            SynthesisModel::Triposg => {
                match self.infer_mesh_triposg(input_image_path, foreground_model, backend, progress)
                {
                    Ok(mesh) => Ok((mesh, SynthesisModel::Triposg)),
                    Err(err) if synthesis_models.contains(&SynthesisModel::Trellis) => {
                        progress.warn(format!("TripoSG failed ({err}); falling back to Trellis2"));
                        match self.infer_mesh_trellis(
                            input_image_path,
                            foreground_model,
                            backend,
                            progress,
                        ) {
                            Ok(mesh) => Ok((mesh, SynthesisModel::Trellis)),
                            Err(trellis_err) => Err(RuntimeError::new(format!(
                                "TripoSG failed ({err}); Trellis2 fallback failed ({trellis_err})"
                            ))),
                        }
                    }
                    Err(err) => Err(err),
                }
            }
            SynthesisModel::Trellis => {
                match self.infer_mesh_trellis(input_image_path, foreground_model, backend, progress)
                {
                    Ok(mesh) => Ok((mesh, SynthesisModel::Trellis)),
                    Err(err) if synthesis_models.contains(&SynthesisModel::Triposg) => {
                        progress.warn(format!("Trellis2 failed ({err}); falling back to TripoSG"));
                        match self.infer_mesh_triposg(
                            input_image_path,
                            foreground_model,
                            backend,
                            progress,
                        ) {
                            Ok(mesh) => Ok((mesh, SynthesisModel::Triposg)),
                            Err(triposg_err) => Err(RuntimeError::new(format!(
                                "Trellis2 failed ({err}); TripoSG fallback failed ({triposg_err})"
                            ))),
                        }
                    }
                    Err(err) => Err(err),
                }
            }
        }
    }

    fn infer_mesh_triposg(
        &mut self,
        input_image_path: &Path,
        foreground_model: ForegroundModel,
        backend: InferenceBackend,
        progress: &ProgressRun,
    ) -> RuntimeResult<Mesh> {
        progress.stage_started(
            "mesh.preprocess_foreground",
            None,
            Some(format!(
                "model={}",
                foreground_model_label(foreground_model)
            )),
        );
        let preprocess_start = Instant::now();
        let prepared = self.prepare_image_for_mesh(input_image_path, foreground_model, backend)?;
        progress.stage_completed(
            "mesh.preprocess_foreground",
            None,
            preprocess_start.elapsed().as_secs_f64() * 1000.0,
            Some(format!("size={}x{}", prepared.width, prepared.height)),
        );
        match backend {
            InferenceBackend::Cpu => {
                progress.stage_started("triposg.load_backend", None, Some("backend=cpu".into()));
                let load_start = Instant::now();
                let state = self.synthesis.ensure_cpu(&self.config)?;
                progress.stage_completed(
                    "triposg.load_backend",
                    None,
                    load_start.elapsed().as_secs_f64() * 1000.0,
                    Some("backend=cpu".into()),
                );
                run_backend_inference(state, &prepared, &self.config, progress)
            }
            InferenceBackend::Wgpu => {
                #[cfg(feature = "wgpu")]
                {
                    progress.stage_started(
                        "triposg.load_backend",
                        None,
                        Some("backend=wgpu".into()),
                    );
                    let load_start = Instant::now();
                    let state = self.synthesis.ensure_wgpu(&self.config)?;
                    progress.stage_completed(
                        "triposg.load_backend",
                        None,
                        load_start.elapsed().as_secs_f64() * 1000.0,
                        Some("backend=wgpu".into()),
                    );
                    run_backend_inference(state, &prepared, &self.config, progress)
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
                    progress.stage_started(
                        "triposg.load_backend",
                        None,
                        Some("backend=cuda".into()),
                    );
                    let load_start = Instant::now();
                    let state = self.synthesis.ensure_cuda(&self.config)?;
                    progress.stage_completed(
                        "triposg.load_backend",
                        None,
                        load_start.elapsed().as_secs_f64() * 1000.0,
                        Some("backend=cuda".into()),
                    );
                    run_backend_inference(state, &prepared, &self.config, progress)
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

    #[cfg(feature = "trellis")]
    fn infer_mesh_trellis(
        &mut self,
        input_image_path: &Path,
        foreground_model: ForegroundModel,
        backend: InferenceBackend,
        progress: &ProgressRun,
    ) -> RuntimeResult<Mesh> {
        progress.stage_started(
            "trellis.preprocess_foreground",
            None,
            Some(format!(
                "model={}",
                foreground_model_label(foreground_model)
            )),
        );
        let preprocess_start = Instant::now();
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
        progress.stage_completed(
            "trellis.preprocess_foreground",
            None,
            preprocess_start.elapsed().as_secs_f64() * 1000.0,
            Some(format!(
                "size={}x{} temp={}",
                prepared.width,
                prepared.height,
                temp_input.display()
            )),
        );

        progress.stage_started("trellis.load_backend", None, None);
        let load_start = Instant::now();
        let pipeline = self.synthesis.ensure_trellis(&self.config)?;
        progress.stage_completed(
            "trellis.load_backend",
            None,
            load_start.elapsed().as_secs_f64() * 1000.0,
            Some(format!(
                "weights_root={}",
                pipeline.config().weights_root.display()
            )),
        );
        let trellis_device = match backend {
            InferenceBackend::Cpu => TrellisDevice::Cpu,
            InferenceBackend::Wgpu => TrellisDevice::Wgpu,
            InferenceBackend::Cuda => TrellisDevice::Cuda,
        };
        let options = TrellisRunOptions {
            quality: map_trellis_quality(self.config.trellis_quality),
            device: trellis_device,
            seed: self.config.seed,
            hook_output: None,
            noise_overrides_hook: None,
        };
        progress.stage_started(
            "trellis.infer",
            None,
            Some(format!(
                "quality={:?} device={}",
                self.config.trellis_quality,
                trellis_device.as_str()
            )),
        );
        let infer_start = Instant::now();
        let profiled = pipeline
            .infer_mesh_profile(&temp_input, &options)
            .map_err(|err| RuntimeError::new(format!("Trellis2 inference failed: {err}")))?;
        progress.stage_completed(
            "trellis.infer",
            None,
            infer_start.elapsed().as_secs_f64() * 1000.0,
            Some(format!(
                "total_ms={:.1} host_readbacks={} host_readback_elements={}",
                profiled.timings.total_ms,
                profiled.timings.host_readback_count,
                profiled.timings.host_readback_elements
            )),
        );
        progress.stage_completed(
            "trellis.sparse",
            Some(profiled.step_counts.sparse),
            profiled.timings.sparse_ms,
            Some(avg_step_detail(
                profiled.timings.sparse_ms,
                profiled.step_counts.sparse,
                profiled.sparse_source.as_str(),
            )),
        );
        progress.stage_completed(
            "trellis.shape_slat",
            Some(profiled.step_counts.shape_slat),
            profiled.timings.shape_slat_ms,
            Some(avg_step_detail(
                profiled.timings.shape_slat_ms,
                profiled.step_counts.shape_slat,
                "runtime",
            )),
        );
        progress.stage_completed(
            "trellis.tex_slat",
            Some(profiled.step_counts.tex_slat),
            profiled.timings.tex_slat_ms,
            Some(avg_step_detail(
                profiled.timings.tex_slat_ms,
                profiled.step_counts.tex_slat,
                "runtime",
            )),
        );
        progress.stage_completed("trellis.decode", None, profiled.timings.decode_ms, None);
        let _ = std::fs::remove_file(temp_input);
        Ok(profiled.mesh.into())
    }

    #[cfg(not(feature = "trellis"))]
    fn infer_mesh_trellis(
        &mut self,
        _input_image_path: &Path,
        _foreground_model: ForegroundModel,
        _backend: InferenceBackend,
        _progress: &ProgressRun,
    ) -> RuntimeResult<Mesh> {
        Err(RuntimeError::new(
            "Trellis backend not enabled; build with burn_synth feature `trellis`",
        ))
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
                )?;
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
                )?;
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
        backend: InferenceBackend,
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
                )?;
                match backend {
                    InferenceBackend::Cpu => {
                        let pipeline = self.foreground.ensure_rmbg14(&root)?;
                        prepare_image_data(input_path, Some(pipeline), &self.config.mesh_prepare)
                            .map_err(|err| {
                                RuntimeError::new(format!("RMBG-1.4 preprocessing failed: {err}"))
                            })
                    }
                    InferenceBackend::Wgpu => {
                        #[cfg(feature = "wgpu")]
                        {
                            let pipeline = self.foreground.ensure_rmbg14_wgpu(&root)?;
                            prepare_image_data(
                                input_path,
                                Some(pipeline),
                                &self.config.mesh_prepare,
                            )
                            .map_err(|err| {
                                RuntimeError::new(format!("RMBG-1.4 preprocessing failed: {err}"))
                            })
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
                            let pipeline = self.foreground.ensure_rmbg14_cuda(&root)?;
                            prepare_image_data(
                                input_path,
                                Some(pipeline),
                                &self.config.mesh_prepare,
                            )
                            .map_err(|err| {
                                RuntimeError::new(format!("RMBG-1.4 preprocessing failed: {err}"))
                            })
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
            ForegroundModel::Rmbg2 => {
                let root = resolve_foreground_weights_root(
                    self.config.bg_weights_root.as_deref(),
                    selected_model,
                )?;
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
    #[cfg(feature = "wgpu")]
    rmbg14_wgpu: Option<RmbgPipeline<WgpuBackend>>,
    #[cfg(feature = "cuda")]
    rmbg14_cuda: Option<RmbgPipeline<CudaBackend>>,
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

    #[cfg(feature = "wgpu")]
    fn ensure_rmbg14_wgpu(&mut self, root: &Path) -> RuntimeResult<&RmbgPipeline<WgpuBackend>> {
        if self.rmbg14_wgpu.is_none() {
            let device = <WgpuBackend as Backend>::Device::default();
            let pipeline = RmbgPipeline::from_pretrained(root, &device).map_err(|err| {
                RuntimeError::new(format!(
                    "failed to load RMBG-1.4 (wgpu) at {}: {err}",
                    root.display()
                ))
            })?;
            self.rmbg14_wgpu = Some(pipeline);
        }
        self.rmbg14_wgpu
            .as_ref()
            .ok_or_else(|| RuntimeError::new("RMBG-1.4 WGPU pipeline unavailable"))
    }

    #[cfg(feature = "cuda")]
    fn ensure_rmbg14_cuda(&mut self, root: &Path) -> RuntimeResult<&RmbgPipeline<CudaBackend>> {
        if self.rmbg14_cuda.is_none() {
            let device = <CudaBackend as Backend>::Device::default();
            let pipeline = RmbgPipeline::from_pretrained(root, &device).map_err(|err| {
                RuntimeError::new(format!(
                    "failed to load RMBG-1.4 (cuda) at {}: {err}",
                    root.display()
                ))
            })?;
            self.rmbg14_cuda = Some(pipeline);
        }
        self.rmbg14_cuda
            .as_ref()
            .ok_or_else(|| RuntimeError::new("RMBG-1.4 CUDA pipeline unavailable"))
    }
}

struct BackendSynthesisState<B: Backend> {
    device: B::Device,
    pipeline: TripoSGPipeline<B>,
    cpu_dino: Option<CpuDinoState>,
}

struct CpuDinoState {
    device: <NdArray<f32> as Backend>::Device,
    encoder: burn_tripo::model::triposg::image_encoder::TripoSGImageEncoder<NdArray<f32>>,
    processor: DinoImageProcessor,
}

#[derive(Default)]
struct SynthesisRuntime {
    cpu: Option<BackendSynthesisState<NdArray<f32>>>,
    #[cfg(feature = "wgpu")]
    wgpu: Option<BackendSynthesisState<WgpuBackend>>,
    #[cfg(feature = "cuda")]
    cuda: Option<BackendSynthesisState<CudaBackend>>,
    #[cfg(feature = "trellis")]
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

    #[cfg(feature = "trellis")]
    fn ensure_trellis(&mut self, config: &RuntimeConfig) -> RuntimeResult<&mut Trellis2Pipeline> {
        if self.trellis.is_none() {
            let mut trellis_config = Trellis2PipelineConfig::default();
            if let Some(root) = config.trellis_weights_root.as_ref() {
                trellis_config.weights_root = root.clone();
            }
            if let Some(root) = config.trellis_image_large_root.as_ref() {
                trellis_config.image_large_root = Some(root.clone());
            }
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
    let parity = triposg_runtime_profile(Some(config.mesh_prepare.max_dimension));
    let weights_root = resolve_triposg_runtime_weights_root(
        config.weights_root.as_deref(),
        parity.burnpack_policy.precision.prefer_f16(),
    )?;
    let use_cpu_dino = should_use_cpu_dino_backend::<B>(map_dino_backend(config.dino_backend));
    let load_options = TripoSGLoadOptions {
        burnpack_policy: parity.burnpack_policy,
        load_image_encoder: !use_cpu_dino,
        strict_dino_preprocess: Some(parity.strict_dino_preprocess),
        ..TripoSGLoadOptions::default()
    };
    let pipeline =
        TripoSGPipeline::<B>::from_pretrained_with_options(&weights_root, &device, load_options)
            .map_err(|err| {
                RuntimeError::new(format!(
                    "failed to load TripoSG weights at {}: {err}",
                    weights_root.display()
                ))
            })?;
    let cpu_dino = if use_cpu_dino {
        let cpu_device = <NdArray<f32> as Backend>::Device::default();
        let encoder = load_triposg_dinov2_with_policy(
            &cpu_device,
            weights_root.join("image_encoder_dinov2/model.safetensors"),
            parity.burnpack_policy,
        )
        .map_err(|err| {
            RuntimeError::new(format!(
                "failed to load CPU DINO encoder at {}: {err}",
                weights_root.display()
            ))
        })?;
        let mut processor = load_dinov2_processor(&weights_root).map_err(|err| {
            RuntimeError::new(format!(
                "failed to load DINO processor config at {}: {err}",
                weights_root.display()
            ))
        })?;
        processor.set_strict_preprocess(parity.strict_dino_preprocess);
        Some(CpuDinoState {
            device: cpu_device,
            encoder,
            processor,
        })
    } else {
        None
    };
    Ok(BackendSynthesisState {
        device,
        pipeline,
        cpu_dino,
    })
}

fn resolve_triposg_runtime_weights_root(
    explicit: Option<&Path>,
    prefer_f16: bool,
) -> RuntimeResult<PathBuf> {
    if let Some(path) = explicit {
        return Ok(resolve_triposg_weights_root(Some(path)));
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        resolve_or_bootstrap_triposg_root(prefer_f16).map_err(|err| {
            RuntimeError::new(format!("failed to prepare TripoSG cache bootstrap: {err}"))
        })
    }

    #[cfg(target_arch = "wasm32")]
    {
        let _ = prefer_f16;
        Ok(resolve_triposg_weights_root(None))
    }
}

fn map_dino_backend(value: DinoBackend) -> DinoBackendChoice {
    match value {
        DinoBackend::Auto => DinoBackendChoice::Auto,
        DinoBackend::Cpu => DinoBackendChoice::Cpu,
        DinoBackend::Gpu => DinoBackendChoice::Gpu,
    }
}

#[cfg(feature = "trellis")]
fn map_trellis_quality(value: TrellisQuality) -> burn_trellis::TrellisQuality {
    match value {
        TrellisQuality::Low => burn_trellis::TrellisQuality::Low,
        TrellisQuality::Medium => burn_trellis::TrellisQuality::Medium,
        TrellisQuality::High => burn_trellis::TrellisQuality::High,
    }
}

fn foreground_model_label(model: ForegroundModel) -> &'static str {
    match model {
        ForegroundModel::Rmbg14 => "rmbg14",
        ForegroundModel::Rmbg2 => "rmbg2",
    }
}

fn synthesis_model_label(model: SynthesisModel) -> &'static str {
    match model {
        SynthesisModel::Triposg => "triposg",
        SynthesisModel::Trellis => "trellis",
    }
}

fn synthesis_models_label(models: &[SynthesisModel]) -> String {
    models
        .iter()
        .map(|model| synthesis_model_label(*model))
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(feature = "trellis")]
fn avg_step_detail(elapsed_ms: f64, total_steps: usize, source: &str) -> String {
    if total_steps == 0 {
        return format!("source={source} avg_step_ms={elapsed_ms:.1}");
    }
    let avg_step_ms = elapsed_ms / total_steps as f64;
    format!("source={source} avg_step_ms={avg_step_ms:.1}")
}

fn run_backend_inference<B: Backend>(
    state: &mut BackendSynthesisState<B>,
    prepared: &PreparedImageData,
    config: &RuntimeConfig,
    progress: &ProgressRun,
) -> RuntimeResult<Mesh> {
    if let Some(seed) = config.seed {
        B::seed(&state.device, seed);
    }
    progress.stage_started(
        "triposg.prepare_tensor",
        None,
        Some(format!("image={}x{}", prepared.width, prepared.height)),
    );
    let prepare_start = Instant::now();
    let image = if state.cpu_dino.is_some() {
        None
    } else {
        Some(prepared.to_tensor::<B>(&state.device))
    };
    progress.stage_completed(
        "triposg.prepare_tensor",
        None,
        prepare_start.elapsed().as_secs_f64() * 1000.0,
        None,
    );

    progress.stage_started(
        "triposg.encode_image",
        None,
        Some(format!(
            "dino_backend={}",
            if state.cpu_dino.is_some() {
                "cpu"
            } else {
                "active"
            }
        )),
    );
    let encode_start = Instant::now();
    let (image_embeds, batch_size) = if let Some(cpu_dino) = state.cpu_dino.as_ref() {
        let cpu_image = prepared.to_tensor::<NdArray<f32>>(&cpu_dino.device);
        let processed = cpu_dino.processor.preprocess(cpu_image);
        let cpu_embeds = cpu_dino.encoder.forward(processed);
        let embeds = convert_embeddings_to_backend::<B>(cpu_embeds, &state.device)?;
        let batch = embeds.shape().dims::<3>()[0];
        (embeds, batch)
    } else {
        if state.pipeline.image_processor.is_strict_preprocess() {
            // Keep strict preprocessing numerics while avoiding a backend->CPU readback:
            // preprocess on CPU first, then upload once to the active backend.
            let cpu_device = <NdArray<f32> as Backend>::Device::default();
            let cpu_image = prepared.to_tensor::<NdArray<f32>>(&cpu_device);
            let cpu_processed = state.pipeline.image_processor.preprocess(cpu_image);
            let batch = cpu_processed.shape().dims::<4>()[0];
            let processed = convert_image_to_backend::<B>(cpu_processed, &state.device)?;
            let embeds = state
                .pipeline
                .image_encoder
                .as_ref()
                .ok_or_else(|| RuntimeError::new("TripoSG image encoder unavailable"))?
                .forward(processed);
            (embeds, batch)
        } else {
            let image = image.expect("image tensor should exist when CPU DINO is disabled");
            let batch = image.shape().dims::<4>()[0];
            let embeds = state.pipeline.encode_image(image);
            (embeds, batch)
        }
    };
    progress.stage_completed(
        "triposg.encode_image",
        None,
        encode_start.elapsed().as_secs_f64() * 1000.0,
        None,
    );

    progress.stage_started(
        "triposg.sample",
        Some(config.num_steps),
        Some(format!(
            "num_tokens={} guidance_scale={:.3}",
            config.num_tokens, config.guidance_scale
        )),
    );
    let sample_start = Instant::now();
    let latents = config.seed.map(|seed| {
        deterministic_latents_from_seed::<B>(
            seed,
            batch_size,
            config.num_tokens,
            state.pipeline.transformer.config().in_channels,
            &state.device,
        )
    });
    let output = state.pipeline.sample_from_embeds_with_progress(
        image_embeds,
        batch_size,
        config.num_steps,
        config.num_tokens,
        config.guidance_scale,
        None,
        latents,
        |step| {
            let elapsed_ms = sample_start.elapsed().as_secs_f64() * 1000.0;
            progress.step(
                "triposg.sample",
                step,
                elapsed_ms,
                Some(format!("timestep={:.6}", step.timestep)),
            );
        },
    );
    let sample_elapsed_ms = sample_start.elapsed().as_secs_f64() * 1000.0;
    let avg_step_ms = if config.num_steps > 0 {
        sample_elapsed_ms / config.num_steps as f64
    } else {
        sample_elapsed_ms
    };
    progress.stage_completed(
        "triposg.sample",
        Some(config.num_steps),
        sample_elapsed_ms,
        Some(format!("avg_step_ms={avg_step_ms:.1}")),
    );

    progress.stage_started(
        "triposg.flash_extract",
        None,
        Some(format!(
            "octree_depth={} min_resolution={} mini_grid_num={} num_chunks={}",
            config.flash_extract.octree_depth,
            config.flash_extract.min_resolution,
            config.flash_extract.mini_grid_num,
            config.flash_extract.num_chunks
        )),
    );
    let extract_start = Instant::now();
    let grid = state
        .pipeline
        .extract_flash_grid_from_latents(output.latents.clone(), &config.flash_extract)
        .map_err(|err| RuntimeError::new(format!("TripoSG geometry extraction failed: {err}")))?;
    progress.stage_completed(
        "triposg.flash_extract",
        None,
        extract_start.elapsed().as_secs_f64() * 1000.0,
        None,
    );

    progress.stage_started("triposg.mesh_extract", None, None);
    let mesh_start = Instant::now();
    let mesh = sdf_to_mesh_diff_dmc(&grid)
        .ok_or_else(|| RuntimeError::new("TripoSG mesh extraction returned an empty mesh"))?;
    progress.stage_completed(
        "triposg.mesh_extract",
        None,
        mesh_start.elapsed().as_secs_f64() * 1000.0,
        Some(format!(
            "vertices={} faces={}",
            mesh.vertices.len(),
            mesh.faces.len()
        )),
    );
    Ok(mesh.into())
}

fn convert_embeddings_to_backend<B: Backend>(
    embeddings: Tensor<NdArray<f32>, 3>,
    device: &B::Device,
) -> RuntimeResult<Tensor<B, 3>> {
    let shape = embeddings.shape().dims::<3>();
    let data = embeddings
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|err| RuntimeError::new(format!("failed to read CPU DINO embeddings: {err:?}")))?;
    let flat = Tensor::<B, 1>::from_floats(data.as_slice(), device);
    Ok(flat.reshape([shape[0] as i32, shape[1] as i32, shape[2] as i32]))
}

fn convert_image_to_backend<B: Backend>(
    image: Tensor<NdArray<f32>, 4>,
    device: &B::Device,
) -> RuntimeResult<Tensor<B, 4>> {
    let shape = image.shape().dims::<4>();
    let data = image
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|err| RuntimeError::new(format!("failed to read CPU image tensor: {err:?}")))?;
    let flat = Tensor::<B, 1>::from_floats(data.as_slice(), device);
    Ok(flat.reshape([
        shape[0] as i32,
        shape[1] as i32,
        shape[2] as i32,
        shape[3] as i32,
    ]))
}

fn decimate_mesh(mut mesh: Mesh, target_faces: Option<usize>) -> Result<Mesh, String> {
    let Some(target_faces) = target_faces.filter(|value| *value > 0) else {
        return Ok(mesh);
    };
    if mesh.faces.len() <= target_faces {
        return Ok(mesh);
    }
    if mesh.faces.is_empty() || mesh.vertices.is_empty() {
        return Ok(mesh);
    }
    if !mesh.uvs.is_empty() || mesh.material.is_some() || mesh.pbr_textures.is_some() {
        return Ok(mesh);
    }

    let decimated = decimate_tripo_mesh(
        &TripoMesh {
            vertices: std::mem::take(&mut mesh.vertices),
            faces: std::mem::take(&mut mesh.faces),
        },
        target_faces,
    )?;
    mesh.vertices = decimated.vertices;
    mesh.faces = decimated.faces;
    Ok(mesh)
}

fn resolve_foreground_weights_root(
    explicit: Option<&Path>,
    model: ForegroundModel,
) -> RuntimeResult<PathBuf> {
    if let Some(path) = explicit
        && let Some(root) = normalize_foreground_root(path, model)
    {
        return Ok(root);
    }
    match model {
        ForegroundModel::Rmbg14 => resolve_rmbg14_runtime_weights_root(),
        ForegroundModel::Rmbg2 => Ok(resolve_rmbg2_weights_root()),
    }
}

fn resolve_rmbg14_runtime_weights_root() -> RuntimeResult<PathBuf> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        resolve_or_bootstrap_rmbg14_root(true)
            .map_err(|err| RuntimeError::new(format!("failed to prepare RMBG-1.4 cache: {err}")))
    }

    #[cfg(target_arch = "wasm32")]
    {
        Ok(resolve_rmbg_weights_root())
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
    Mesh {
        vertices,
        faces,
        uvs: Vec::new(),
        material: None,
        pbr_textures: None,
    }
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

    #[cfg(not(feature = "trellis"))]
    #[test]
    fn trellis_request_errors_when_feature_disabled() {
        let mut runtime = SynthRuntime::new(RuntimeConfig {
            backend: InferenceBackend::Cpu,
            ..RuntimeConfig::default()
        });
        let err = runtime
            .synthesize_mesh(MeshRequest {
                image: ImageSource::from_path("unused.png"),
                foreground_model: Some(ForegroundModel::Rmbg14),
                synthesis_models: Some(vec![SynthesisModel::Trellis]),
                backend: Some(InferenceBackend::Cpu),
                dry_run: false,
            })
            .expect_err("trellis requests should fail when feature is disabled");
        assert!(
            err.to_string().contains("feature `trellis`"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn decimate_mesh_reduces_face_count() {
        let mut mesh = Mesh {
            vertices: Vec::new(),
            faces: Vec::new(),
            uvs: Vec::new(),
            material: None,
            pbr_textures: None,
        };
        let n = 24usize;
        for y in 0..=n {
            for x in 0..=n {
                mesh.vertices.push([x as f32, y as f32, 0.0]);
            }
        }
        for y in 0..n {
            for x in 0..n {
                let i0 = (y * (n + 1) + x) as u32;
                let i1 = i0 + 1;
                let i2 = i0 + (n + 1) as u32;
                let i3 = i2 + 1;
                mesh.faces.push([i0, i1, i3]);
                mesh.faces.push([i0, i3, i2]);
            }
        }
        let original_faces = mesh.faces.len();
        let simplified = decimate_mesh(mesh, Some(200)).expect("decimation should succeed");
        assert!(simplified.faces.len() <= 200);
        assert!(!simplified.faces.is_empty());
        assert!(simplified.faces.len() < original_faces);
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn auto_dino_backend_uses_gpu_on_wgpu() {
        assert!(!should_use_cpu_dino_backend::<WgpuBackend>(
            DinoBackendChoice::Auto
        ));
    }

    #[test]
    fn parity_profile_keeps_f16_preference_and_fallback_dimension() {
        let profile = triposg_runtime_profile(Some(777));
        assert!(profile.strict_dino_preprocess);
        assert!(profile.strict_rmbg_interp);
        assert_eq!(profile.max_image_dim, Some(777));
        assert!(profile.burnpack_policy.precision.prefer_f16());
    }
}

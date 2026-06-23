use std::fmt::{Display, Formatter};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use burn::backend::NdArray;
use burn::tensor::Tensor;
use burn::tensor::backend::{Backend, BackendTypes};
use burn::tensor::ops::InterpolateMode;
use burn_foreground::pipeline::{
    PrepareImageConfig, PreparedImageData, RmbgPipeline, prepare_image_data,
};
use burn_foreground::resize::resize_chw_align_corners_false;
use burn_foreground::rmbg2::Rmbg2Pipeline;
use burn_foreground::rmbg2::import::resolve_rmbg2_weights_root;
use burn_foreground::rmbg14::import::resolve_rmbg_weights_root;
use burn_foreground::rmbg14::set_rmbg_strict_interp_override;
#[cfg(feature = "trellis")]
use burn_trellis::paths::{resolve_trellis2_image_large_root, resolve_trellis2_weights_root};
#[cfg(feature = "trellis")]
use burn_trellis::pipeline::{
    Trellis2Pipeline, Trellis2PipelineConfig, TrellisDevice, TrellisRunOptions,
};
#[cfg(feature = "trellis")]
use burn_trellis::staged_pipeline::TrellisDecodeOutputMode;
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
use burn_triposplat::{
    CfgPredictionMode, GaussianSplatCloud, TripoSplatArtifactSet, TripoSplatBurnpackPrecision,
    TripoSplatOptions, TripoSplatPipeline, TripoSplatPipelineConfig, TripoSplatRuntimeComponents,
    normalize_num_gaussians,
};
use image::{ImageFormat, RgbaImage};
use sha2::{Digest, Sha256};

use crate::io::ImageSource;
use crate::mesh::Mesh;
#[cfg(all(not(target_arch = "wasm32"), feature = "trellis"))]
use crate::native_model_bootstrap::resolve_or_bootstrap_trellis_roots;
#[cfg(not(target_arch = "wasm32"))]
use crate::native_model_bootstrap::{
    resolve_or_bootstrap_rmbg14_root, resolve_or_bootstrap_triposg_root,
    resolve_or_bootstrap_triposplat_root,
};
use crate::pipeline::{
    ForegroundModel, ModelSelection, SynthesisAsset, SynthesisModel, sanitize_synthesis_models,
};
use crate::progress::{RuntimeProgressEvent, RuntimeProgressObserver};
use crate::quality::{
    DEFAULT_SEED as QUALITY_DEFAULT_SEED, DEFAULT_TRIPOSG_GUIDANCE_SCALE,
    DEFAULT_TRIPOSG_TARGET_FACES,
};
use crate::triposplat_preprocess::{TRIPOSPLAT_CANVAS_SIZE, triposplat_prepare_image_config};

const DEFAULT_BOUNDS: [f32; 6] = [-1.005, -1.005, -1.005, 1.005, 1.005, 1.005];
const DEFAULT_NUM_STEPS: usize = 50;
const DEFAULT_NUM_TOKENS: usize = 2048;
const DEFAULT_GUIDANCE_SCALE: f32 = DEFAULT_TRIPOSG_GUIDANCE_SCALE;
const DEFAULT_FLASH_OCTREE_DEPTH: usize = 9;
const DEFAULT_FLASH_MIN_RESOLUTION: usize = 63;
const DEFAULT_FLASH_MINI_GRID_NUM: usize = 4;
const DEFAULT_FLASH_NUM_CHUNKS: usize = 10_000;
const DEFAULT_SEED: u64 = QUALITY_DEFAULT_SEED;
const DEFAULT_TARGET_FACES: usize = DEFAULT_TRIPOSG_TARGET_FACES;
pub const DEFAULT_TRELLIS_PBR_TEXTURE_SIZE: usize = 1024;
const DEFAULT_TRIPOSPLAT_NUM_GAUSSIANS: usize = 262_144;
const DEFAULT_TRIPOSPLAT_SHIFT: f32 = 3.0;
const DEFAULT_TRIPOSPLAT_ERODE_RADIUS: usize = 1;

#[cfg(feature = "wgpu")]
type WgpuBackend = burn_wgpu::Wgpu<f32, i32, u32>;

#[cfg(feature = "wgpu")]
type WgpuTripoSplatBackend = burn_wgpu::Wgpu<f32, i32, u32>;

#[cfg(feature = "wgpu")]
type WgpuTripoSplatBackendF16 = burn_wgpu::Wgpu<burn::tensor::f16, i32, u32>;

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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum TrellisComputeProfile {
    #[default]
    ReferenceF32,
    WgpuFastMixedF16,
    WgpuFastSparseSelfF16,
    WgpuFastSparseCrossF16,
    WgpuFastF16Tail1F32,
    WgpuFastF16Tail2F32,
    WgpuFastF16Tail4F32,
    WgpuFastF16Tail6F32,
    WgpuFastF16,
}

#[derive(Clone, Debug)]
pub struct RuntimeConfig {
    pub model_selection: ModelSelection,
    pub backend: InferenceBackend,
    /// Optional externally initialized WGPU device. Native Bevy wrappers use this
    /// to share the render device with Burn/CubeCL instead of creating a second
    /// WGPU device for inference.
    #[cfg(feature = "wgpu")]
    pub wgpu_device: Option<burn_wgpu::WgpuDevice>,
    /// TripoSG weights root.
    pub weights_root: Option<PathBuf>,
    /// Trellis2 weights root.
    pub trellis_weights_root: Option<PathBuf>,
    /// TripoSplat weights root.
    pub triposplat_weights_root: Option<PathBuf>,
    /// Native TripoSplat BurnPack precision. `None` means auto-select from
    /// available artifacts; explicit values fail fast when unavailable.
    pub triposplat_weights_precision: Option<TripoSplatBurnpackPrecision>,
    /// Optional local root for TRELLIS-image-large assets.
    pub trellis_image_large_root: Option<PathBuf>,
    /// Legacy field retained for CLI compatibility; ignored by Trellis2 Rust runtime.
    pub trellis_python_bin: Option<PathBuf>,
    /// Legacy field retained for CLI compatibility; ignored by Trellis2 Rust runtime.
    pub trellis_bridge_script: Option<PathBuf>,
    /// Optional hook with stage-noise/coord overrides for Trellis runtime parity/debug.
    pub trellis_noise_overrides_hook: Option<PathBuf>,
    /// Optional explicit sparse-coordinate cap for Trellis decode.
    pub trellis_max_sparse_coords: Option<usize>,
    /// Optional native Trellis PBR texture size for Rust GLB export.
    pub trellis_pbr_texture_size: Option<usize>,
    /// Whether Trellis should bake UVs/material textures in the native GLB path.
    pub trellis_pbr_enabled: bool,
    /// Trellis high-level quality selection.
    pub trellis_quality: TrellisQuality,
    /// Trellis runtime compute profile.
    pub trellis_compute_profile: TrellisComputeProfile,
    pub bg_weights_root: Option<PathBuf>,
    pub num_steps: usize,
    pub num_tokens: usize,
    pub guidance_scale: f32,
    pub triposplat_num_steps: usize,
    pub triposplat_guidance_scale: f32,
    pub triposplat_shift: f32,
    pub triposplat_num_gaussians: usize,
    pub triposplat_erode_radius: usize,
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
            #[cfg(feature = "wgpu")]
            wgpu_device: None,
            weights_root: None,
            trellis_weights_root: None,
            triposplat_weights_root: None,
            triposplat_weights_precision: Some(TripoSplatBurnpackPrecision::F32),
            trellis_image_large_root: None,
            trellis_python_bin: None,
            trellis_bridge_script: None,
            trellis_noise_overrides_hook: None,
            trellis_max_sparse_coords: None,
            trellis_pbr_texture_size: Some(DEFAULT_TRELLIS_PBR_TEXTURE_SIZE),
            trellis_pbr_enabled: false,
            trellis_quality: TrellisQuality::Low,
            trellis_compute_profile: TrellisComputeProfile::default(),
            bg_weights_root: None,
            num_steps: DEFAULT_NUM_STEPS,
            num_tokens: DEFAULT_NUM_TOKENS,
            guidance_scale: DEFAULT_GUIDANCE_SCALE,
            triposplat_num_steps: burn_triposplat::DEFAULT_NUM_STEPS,
            triposplat_guidance_scale: burn_triposplat::DEFAULT_GUIDANCE_SCALE,
            triposplat_shift: DEFAULT_TRIPOSPLAT_SHIFT,
            triposplat_num_gaussians: DEFAULT_TRIPOSPLAT_NUM_GAUSSIANS,
            triposplat_erode_radius: DEFAULT_TRIPOSPLAT_ERODE_RADIUS,
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
pub struct AssetRequest {
    pub image: ImageSource,
    pub foreground_model: Option<ForegroundModel>,
    pub synthesis_models: Option<Vec<SynthesisModel>>,
    pub backend: Option<InferenceBackend>,
    pub dry_run: bool,
}

impl AssetRequest {
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
pub struct AssetOutput {
    pub asset: SynthesisAsset,
    pub foreground_model: ForegroundModel,
    pub synthesis_models: Vec<SynthesisModel>,
    pub synthesis_backend: SynthesisModel,
    pub backend: InferenceBackend,
}

#[derive(Debug, Clone)]
pub struct SplatRequest {
    pub image: ImageSource,
    pub foreground_model: Option<ForegroundModel>,
    pub backend: Option<InferenceBackend>,
    pub num_gaussians: Vec<usize>,
    pub dry_run: bool,
}

impl SplatRequest {
    pub fn from_image(image: ImageSource) -> Self {
        Self {
            image,
            foreground_model: None,
            backend: None,
            num_gaussians: Vec::new(),
            dry_run: false,
        }
    }
}

#[derive(Debug)]
pub struct SplatOutput {
    pub splats: Vec<GaussianSplatCloud>,
    pub num_gaussians: Vec<usize>,
    pub foreground_model: ForegroundModel,
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

#[cfg(feature = "trellis")]
fn trellis_decode_output_mode(pbr_enabled: bool) -> TrellisDecodeOutputMode {
    if pbr_enabled {
        TrellisDecodeOutputMode::NativePbr
    } else {
        TrellisDecodeOutputMode::NativeMesh
    }
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

    pub fn config_mut(&mut self) -> &mut RuntimeConfig {
        &mut self.config
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

    pub fn synthesize_asset(&mut self, request: AssetRequest) -> RuntimeResult<AssetOutput> {
        let selected_foreground = request
            .foreground_model
            .unwrap_or(self.config.model_selection.foreground_model);
        let selected_synthesis = request
            .synthesis_models
            .clone()
            .map(sanitize_synthesis_models)
            .unwrap_or_else(|| self.config.model_selection.synthesis_models.clone());
        let selected_backend = request.backend.unwrap_or(self.config.backend);
        let preferred_synthesis = selected_synthesis
            .first()
            .copied()
            .unwrap_or(SynthesisModel::Triposg);

        if matches!(preferred_synthesis, SynthesisModel::Triposplat) {
            let splat_output = self.synthesize_splats_inner(
                SplatRequest {
                    image: request.image,
                    foreground_model: Some(selected_foreground),
                    backend: Some(selected_backend),
                    num_gaussians: vec![self.config.triposplat_num_gaussians],
                    dry_run: request.dry_run,
                },
                "asset",
            )?;
            let splats = splat_output
                .splats
                .into_iter()
                .next()
                .ok_or_else(|| RuntimeError::new("TripoSplat produced no splat clouds"))?;
            let output = AssetOutput {
                asset: SynthesisAsset::GaussianSplat(splats),
                foreground_model: selected_foreground,
                synthesis_models: selected_synthesis,
                synthesis_backend: SynthesisModel::Triposplat,
                backend: selected_backend,
            };
            return Ok(output);
        }

        let mesh_output = self.synthesize_mesh(MeshRequest {
            image: request.image,
            foreground_model: Some(selected_foreground),
            synthesis_models: Some(selected_synthesis),
            backend: Some(selected_backend),
            dry_run: request.dry_run,
        })?;
        Ok(AssetOutput {
            asset: SynthesisAsset::Mesh(mesh_output.mesh),
            foreground_model: mesh_output.foreground_model,
            synthesis_models: mesh_output.synthesis_models,
            synthesis_backend: mesh_output.synthesis_backend,
            backend: mesh_output.backend,
        })
    }

    pub fn synthesize_splats(&mut self, request: SplatRequest) -> RuntimeResult<SplatOutput> {
        self.synthesize_splats_inner(request, "splat")
    }

    fn synthesize_splats_inner(
        &mut self,
        request: SplatRequest,
        run: &'static str,
    ) -> RuntimeResult<SplatOutput> {
        let selected_foreground = request
            .foreground_model
            .unwrap_or(self.config.model_selection.foreground_model);
        let selected_backend = request.backend.unwrap_or(self.config.backend);
        let num_gaussians = normalize_triposplat_counts(
            &request.num_gaussians,
            self.config.triposplat_num_gaussians,
        )?;
        let progress = ProgressRun::new(
            &self.config.progress,
            run,
            Some(format!(
                "synthesis_model=triposplat backend={} num_gaussians={} steps={} guidance_scale={:.3}",
                selected_backend.as_str(),
                triposplat_counts_label(&num_gaussians),
                self.config.triposplat_num_steps,
                self.config.triposplat_guidance_scale
            )),
        );
        let splats = if request.dry_run {
            if num_gaussians.len() != 1 {
                return Err(RuntimeError::new(
                    "TripoSplat dry-run emits one canonical debug cloud; request one gaussian count",
                ));
            }
            progress.stage_started("triposplat.dry_run", None, None);
            let dry_start = Instant::now();
            let splats = self.debug_triposplat_splats_with_count(num_gaussians[0])?;
            progress.stage_completed(
                "triposplat.dry_run",
                None,
                dry_start.elapsed().as_secs_f64() * 1000.0,
                Some(format!("splats={}", splats.len())),
            );
            vec![splats]
        } else {
            let materialized = MaterializedImageInput::from_source(&request.image)?;
            self.infer_splats_triposplat_many(
                materialized.path(),
                selected_foreground,
                selected_backend,
                &num_gaussians,
                &progress,
            )?
        };
        progress.complete(Some(triposplat_splats_detail(&splats)));
        Ok(SplatOutput {
            splats,
            num_gaussians,
            foreground_model: selected_foreground,
            synthesis_backend: SynthesisModel::Triposplat,
            backend: selected_backend,
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
            SynthesisModel::Triposplat => Err(RuntimeError::new(
                "TripoSplat produces Gaussian splats, not meshes; use synthesize_asset or the CLI splat command",
            )),
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
            compute_profile: map_trellis_compute_profile(self.config.trellis_compute_profile),
            seed: self.config.seed,
            hook_output: None,
            noise_overrides_hook: self.config.trellis_noise_overrides_hook.clone(),
            max_sparse_coords: self.config.trellis_max_sparse_coords,
            target_faces: self.config.target_faces,
            pbr_texture_size: self.config.trellis_pbr_texture_size,
            decode_output_mode: trellis_decode_output_mode(self.config.trellis_pbr_enabled),
            runtime_stage_debug: false,
            runtime_attention_debug: false,
            runtime_decoder_conv_telemetry: false,
            runtime_stage_fence: false,
            sampler_overrides: Default::default(),
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
        if let Err(err) = validate_trellis_runtime_sources(
            trellis_device,
            profiled.sparse_source.as_str(),
            profiled.decode_source.as_str(),
        ) {
            let _ = std::fs::remove_file(&temp_input);
            return Err(err);
        }
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
                match self.config.backend {
                    InferenceBackend::Cpu => {
                        let pipeline = self.foreground.ensure_rmbg14(&root)?;
                        let prepared = prepare_image_data(
                            input_path,
                            Some(pipeline),
                            &self.config.foreground_prepare,
                        )
                        .map_err(|err| RuntimeError::new(format!("RMBG-1.4 failed: {err}")))?;
                        prepared.alpha_mask.ok_or_else(|| {
                            RuntimeError::new("RMBG-1.4 did not produce an alpha mask")
                        })
                    }
                    InferenceBackend::Wgpu => {
                        #[cfg(feature = "wgpu")]
                        {
                            let pipeline = self
                                .foreground
                                .ensure_rmbg14_wgpu(&root, self.config.wgpu_device.as_ref())?;
                            let prepared = prepare_image_data(
                                input_path,
                                Some(pipeline),
                                &self.config.foreground_prepare,
                            )
                            .map_err(|err| RuntimeError::new(format!("RMBG-1.4 failed: {err}")))?;
                            prepared.alpha_mask.ok_or_else(|| {
                                RuntimeError::new("RMBG-1.4 did not produce an alpha mask")
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
                            let prepared = prepare_image_data(
                                input_path,
                                Some(pipeline),
                                &self.config.foreground_prepare,
                            )
                            .map_err(|err| RuntimeError::new(format!("RMBG-1.4 failed: {err}")))?;
                            prepared.alpha_mask.ok_or_else(|| {
                                RuntimeError::new("RMBG-1.4 did not produce an alpha mask")
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
                let prepared = pipeline
                    .prepare_image_data(input_path, &self.config.foreground_prepare)
                    .map_err(|err| RuntimeError::new(format!("RMBG-2.0 failed: {err}")))?;
                prepared
                    .alpha_mask
                    .ok_or_else(|| RuntimeError::new("RMBG-2.0 did not produce an alpha mask"))
            }
        }
    }

    fn infer_splats_triposplat_many(
        &mut self,
        input_image_path: &Path,
        foreground_model: ForegroundModel,
        backend: InferenceBackend,
        num_gaussians: &[usize],
        progress: &ProgressRun,
    ) -> RuntimeResult<Vec<GaussianSplatCloud>> {
        validate_triposplat_backend_support(backend, self.config.triposplat_weights_precision)?;
        #[cfg(feature = "cuda")]
        if matches!(backend, InferenceBackend::Cuda) && self.synthesis.triposplat_cuda.is_none() {
            progress.stage_started(
                "triposplat.cuda_preflight",
                None,
                Some("matmul/readback".to_string()),
            );
            let preflight_start = Instant::now();
            preflight_triposplat_cuda_backend()?;
            progress.stage_completed(
                "triposplat.cuda_preflight",
                None,
                preflight_start.elapsed().as_secs_f64() * 1000.0,
                None,
            );
        }
        progress.stage_started(
            "triposplat.preprocess_foreground",
            None,
            Some(format!(
                "model={}",
                foreground_model_label(foreground_model)
            )),
        );
        let preprocess_start = Instant::now();
        let prepared =
            self.prepare_image_for_triposplat(input_image_path, foreground_model, backend)?;
        let prepared = resize_prepared_image_for_triposplat(prepared, TRIPOSPLAT_CANVAS_SIZE);
        progress.stage_completed(
            "triposplat.preprocess_foreground",
            None,
            preprocess_start.elapsed().as_secs_f64() * 1000.0,
            Some(format!(
                "size={}x{} erode_radius={}",
                prepared.width, prepared.height, self.config.triposplat_erode_radius
            )),
        );

        progress.stage_started(
            "triposplat.load_backend",
            None,
            Some(format!("backend={}", backend.as_str())),
        );
        let load_start = Instant::now();
        let mut options = triposplat_options_from_config(&self.config);
        if let Some(first) = num_gaussians.first().copied() {
            options.num_gaussians = first;
        }
        match backend {
            InferenceBackend::Cpu => {
                let state = self.synthesis.ensure_triposplat_cpu(&self.config)?;
                progress.stage_completed(
                    "triposplat.load_backend",
                    None,
                    load_start.elapsed().as_secs_f64() * 1000.0,
                    Some(state.load_detail()),
                );
                run_triposplat_preprocessed_many(state, &prepared, num_gaussians, options, progress)
            }
            InferenceBackend::Wgpu => {
                #[cfg(feature = "wgpu")]
                {
                    let state = self.synthesis.ensure_triposplat_wgpu(&self.config)?;
                    progress.stage_completed(
                        "triposplat.load_backend",
                        None,
                        load_start.elapsed().as_secs_f64() * 1000.0,
                        Some(state.load_detail()),
                    );
                    state.run_preprocessed_many(&prepared, num_gaussians, options, progress)
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
                    let state = self.synthesis.ensure_triposplat_cuda(&self.config)?;
                    progress.stage_completed(
                        "triposplat.load_backend",
                        None,
                        load_start.elapsed().as_secs_f64() * 1000.0,
                        Some(state.load_detail()),
                    );
                    run_triposplat_preprocessed_many(
                        state,
                        &prepared,
                        num_gaussians,
                        options,
                        progress,
                    )
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

    fn debug_triposplat_splats_with_count(
        &self,
        num_gaussians: usize,
    ) -> RuntimeResult<GaussianSplatCloud> {
        let mut options = triposplat_options_from_config(&self.config);
        options.num_gaussians = num_gaussians;
        Ok(TripoSplatPipeline::debug_output(options).splats)
    }

    fn prepare_image_for_mesh(
        &mut self,
        input_path: &Path,
        selected_model: ForegroundModel,
        backend: InferenceBackend,
    ) -> RuntimeResult<PreparedImageData> {
        let config = self.config.mesh_prepare.clone();
        self.prepare_image_with_config(input_path, selected_model, backend, &config)
    }

    fn prepare_image_for_triposplat(
        &mut self,
        input_path: &Path,
        selected_model: ForegroundModel,
        backend: InferenceBackend,
    ) -> RuntimeResult<PreparedImageData> {
        let config = triposplat_prepare_image_config(self.config.triposplat_erode_radius);
        self.prepare_image_with_config(input_path, selected_model, backend, &config)
    }

    fn prepare_image_with_config(
        &mut self,
        input_path: &Path,
        selected_model: ForegroundModel,
        backend: InferenceBackend,
        config: &PrepareImageConfig,
    ) -> RuntimeResult<PreparedImageData> {
        if let Ok(prepared) = prepare_image_data::<NdArray<f32>>(input_path, None, config) {
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
                        prepare_image_data(input_path, Some(pipeline), config).map_err(|err| {
                            RuntimeError::new(format!("RMBG-1.4 preprocessing failed: {err}"))
                        })
                    }
                    InferenceBackend::Wgpu => {
                        #[cfg(feature = "wgpu")]
                        {
                            let pipeline = self
                                .foreground
                                .ensure_rmbg14_wgpu(&root, self.config.wgpu_device.as_ref())?;
                            prepare_image_data(input_path, Some(pipeline), config).map_err(|err| {
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
                            prepare_image_data(input_path, Some(pipeline), config).map_err(|err| {
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
                    .prepare_image_data(input_path, config)
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
            let device = <NdArray<f32> as BackendTypes>::Device::default();
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
    fn ensure_rmbg14_wgpu(
        &mut self,
        root: &Path,
        device: Option<&burn_wgpu::WgpuDevice>,
    ) -> RuntimeResult<&RmbgPipeline<WgpuBackend>> {
        if self.rmbg14_wgpu.is_none() {
            let device = device.cloned().unwrap_or_default();
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
            let device = <CudaBackend as BackendTypes>::Device::default();
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

struct BackendTripoSplatState<B: Backend> {
    device: B::Device,
    components: TripoSplatRuntimeComponents<B>,
    weights_root: PathBuf,
    precision: TripoSplatBurnpackPrecision,
    decoder_compute_label: &'static str,
    latent_cache: Option<TripoSplatLatentCache<B>>,
}

struct TripoSplatLatentCache<B: Backend> {
    key: TripoSplatLatentCacheKey,
    latent: Tensor<B, 3>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TripoSplatLatentCacheKey {
    image_hash: [u8; 32],
    seed: u64,
    steps: usize,
    guidance_scale_bits: u32,
    shift_bits: u32,
    cfg_mode: CfgPredictionMode,
    attention_query_chunk_tokens: Option<usize>,
}

impl<B: Backend> BackendTripoSplatState<B> {
    fn load_detail(&self) -> String {
        format!(
            "weights_root={} precision={} compute={} decoder_compute={}",
            self.weights_root.display(),
            self.precision.as_str(),
            backend_float_label::<B>(),
            self.decoder_compute_label,
        )
    }
}

#[cfg(feature = "wgpu")]
enum WgpuTripoSplatState {
    F32(BackendTripoSplatState<WgpuTripoSplatBackend>),
    F16(BackendTripoSplatState<WgpuTripoSplatBackendF16>),
}

#[cfg(feature = "wgpu")]
impl WgpuTripoSplatState {
    fn load_detail(&self) -> String {
        match self {
            Self::F32(state) => state.load_detail(),
            Self::F16(state) => state.load_detail(),
        }
    }

    fn run_preprocessed_many(
        &mut self,
        prepared: &PreparedImageData,
        num_gaussians: &[usize],
        options: TripoSplatOptions,
        progress: &ProgressRun,
    ) -> RuntimeResult<Vec<GaussianSplatCloud>> {
        match self {
            Self::F32(state) => {
                run_triposplat_preprocessed_many(state, prepared, num_gaussians, options, progress)
            }
            Self::F16(state) => {
                run_triposplat_preprocessed_many(state, prepared, num_gaussians, options, progress)
            }
        }
    }
}

struct CpuDinoState {
    device: <NdArray<f32> as BackendTypes>::Device,
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
    triposplat_cpu: Option<BackendTripoSplatState<NdArray<f32>>>,
    #[cfg(feature = "wgpu")]
    triposplat_wgpu: Option<WgpuTripoSplatState>,
    #[cfg(feature = "cuda")]
    triposplat_cuda: Option<BackendTripoSplatState<CudaBackend>>,
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
            let (weights_root, image_large_root) = resolve_trellis_runtime_roots(
                config.trellis_weights_root.as_deref(),
                config.trellis_image_large_root.as_deref(),
            )?;
            trellis_config.weights_root = weights_root;
            trellis_config.image_large_root = image_large_root;
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
            self.wgpu = Some(load_wgpu_backend_state(config)?);
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

    fn ensure_triposplat_cpu(
        &mut self,
        config: &RuntimeConfig,
    ) -> RuntimeResult<&mut BackendTripoSplatState<NdArray<f32>>> {
        if self.triposplat_cpu.is_none() {
            self.triposplat_cpu = Some(load_triposplat_state::<NdArray<f32>>(
                config,
                InferenceBackend::Cpu,
            )?);
        }
        self.triposplat_cpu
            .as_mut()
            .ok_or_else(|| RuntimeError::new("TripoSplat synthesis backend unavailable"))
    }

    #[cfg(feature = "wgpu")]
    fn ensure_triposplat_wgpu(
        &mut self,
        config: &RuntimeConfig,
    ) -> RuntimeResult<&mut WgpuTripoSplatState> {
        if self.triposplat_wgpu.is_none() {
            self.triposplat_wgpu = Some(load_wgpu_triposplat_state(config)?);
        }
        self.triposplat_wgpu
            .as_mut()
            .ok_or_else(|| RuntimeError::new("TripoSplat WGPU synthesis backend unavailable"))
    }

    #[cfg(feature = "cuda")]
    fn ensure_triposplat_cuda(
        &mut self,
        config: &RuntimeConfig,
    ) -> RuntimeResult<&mut BackendTripoSplatState<CudaBackend>> {
        if self.triposplat_cuda.is_none() {
            self.triposplat_cuda = Some(load_cuda_triposplat_state(config)?);
        }
        self.triposplat_cuda
            .as_mut()
            .ok_or_else(|| RuntimeError::new("TripoSplat CUDA synthesis backend unavailable"))
    }
}

fn load_backend_state<B: Backend>(
    config: &RuntimeConfig,
) -> RuntimeResult<BackendSynthesisState<B>> {
    let device = B::Device::default();
    load_backend_state_with_device(config, device)
}

fn load_backend_state_with_device<B: Backend>(
    config: &RuntimeConfig,
    device: B::Device,
) -> RuntimeResult<BackendSynthesisState<B>> {
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
        let cpu_device = <NdArray<f32> as BackendTypes>::Device::default();
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

#[cfg(feature = "wgpu")]
fn load_wgpu_backend_state(
    config: &RuntimeConfig,
) -> RuntimeResult<BackendSynthesisState<WgpuBackend>> {
    load_backend_state_with_device(config, wgpu_runtime_device(config))
}

fn load_triposplat_state<B: Backend>(
    config: &RuntimeConfig,
    backend: InferenceBackend,
) -> RuntimeResult<BackendTripoSplatState<B>> {
    let device = B::Device::default();
    load_triposplat_state_with_device(config, device, backend)
}

fn load_triposplat_state_with_device<B: Backend>(
    config: &RuntimeConfig,
    device: B::Device,
    backend: InferenceBackend,
) -> RuntimeResult<BackendTripoSplatState<B>> {
    if let Some(seed) = config.seed {
        B::seed(&device, seed);
    }
    let weights_root = resolve_triposplat_runtime_weights_root(
        config.triposplat_weights_root.as_deref(),
        config.triposplat_weights_precision,
    )?;
    let precision =
        resolve_triposplat_precision(&weights_root, config.triposplat_weights_precision)?;
    validate_triposplat_backend_support(backend, Some(precision))?;
    let pipeline = TripoSplatPipeline::new(TripoSplatPipelineConfig {
        weights_root: weights_root.clone(),
        precision,
    })
    .map_err(|err| RuntimeError::new(format!("failed to initialize TripoSplat: {err}")))?;
    let mut components = pipeline
        .load_runtime_components(&device)
        .map_err(|err| RuntimeError::new(format!("failed to load TripoSplat components: {err}")))?;
    let decoder_compute_label = if matches!(backend, InferenceBackend::Wgpu)
        && matches!(precision, TripoSplatBurnpackPrecision::F16)
        && backend_float_label::<B>() == "f16"
    {
        components.decoder = burn_triposplat::import::cast_module_float_dtype(
            components.decoder,
            burn::tensor::FloatDType::F32,
        );
        "f32"
    } else {
        backend_float_label::<B>()
    };
    Ok(BackendTripoSplatState {
        device,
        components,
        weights_root,
        precision,
        decoder_compute_label,
        latent_cache: None,
    })
}

#[cfg(feature = "cuda")]
fn load_cuda_triposplat_state(
    config: &RuntimeConfig,
) -> RuntimeResult<BackendTripoSplatState<CudaBackend>> {
    let device = <CudaBackend as BackendTypes>::Device::default();
    if let Some(seed) = config.seed {
        CudaBackend::seed(&device, seed);
    }
    let weights_root = resolve_triposplat_runtime_weights_root(
        config.triposplat_weights_root.as_deref(),
        config.triposplat_weights_precision,
    )?;
    let precision =
        resolve_triposplat_precision(&weights_root, config.triposplat_weights_precision)?;
    validate_triposplat_backend_support(InferenceBackend::Cuda, Some(precision))?;
    let pipeline = TripoSplatPipeline::new(TripoSplatPipelineConfig {
        weights_root: weights_root.clone(),
        precision,
    })
    .map_err(|err| RuntimeError::new(format!("failed to initialize TripoSplat: {err}")))?;
    let artifacts = TripoSplatArtifactSet::new(&weights_root, precision);
    let components =
        burn_triposplat::import::load_triposplat_runtime_components_with_compute_dtype_and_callback::<
            CudaBackend,
            _,
        >(
            &device,
            &artifacts,
            triposplat_compute_dtype_for_precision(precision),
            |event| flush_cuda_upload_queue_after_runtime_triposplat_load_event(&device, event),
        )
        .map_err(|err| RuntimeError::new(format!("failed to load TripoSplat components: {err}")))?;
    Ok(BackendTripoSplatState {
        device,
        components,
        weights_root: pipeline.config().weights_root.clone(),
        precision,
        decoder_compute_label: backend_float_label::<CudaBackend>(),
        latent_cache: None,
    })
}

#[cfg(feature = "wgpu")]
fn load_wgpu_triposplat_state(config: &RuntimeConfig) -> RuntimeResult<WgpuTripoSplatState> {
    let weights_root = resolve_triposplat_runtime_weights_root(
        config.triposplat_weights_root.as_deref(),
        config.triposplat_weights_precision,
    )?;
    let precision =
        resolve_triposplat_precision(&weights_root, config.triposplat_weights_precision)?;
    validate_triposplat_backend_support(InferenceBackend::Wgpu, Some(precision))?;
    match precision {
        TripoSplatBurnpackPrecision::F16 => {
            let mut f16_config = config.clone();
            f16_config.triposplat_weights_root = Some(weights_root);
            f16_config.triposplat_weights_precision = Some(TripoSplatBurnpackPrecision::F16);
            Ok(WgpuTripoSplatState::F16(load_triposplat_state_with_device(
                &f16_config,
                wgpu_runtime_device(config),
                InferenceBackend::Wgpu,
            )?))
        }
        TripoSplatBurnpackPrecision::F32 => {
            let mut f32_config = config.clone();
            f32_config.triposplat_weights_root = Some(weights_root);
            f32_config.triposplat_weights_precision = Some(TripoSplatBurnpackPrecision::F32);
            Ok(WgpuTripoSplatState::F32(load_triposplat_state_with_device(
                &f32_config,
                wgpu_runtime_device(config),
                InferenceBackend::Wgpu,
            )?))
        }
    }
}

#[cfg(feature = "wgpu")]
fn wgpu_runtime_device(config: &RuntimeConfig) -> burn_wgpu::WgpuDevice {
    config.wgpu_device.clone().unwrap_or_default()
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

fn resolve_triposplat_runtime_weights_root(
    explicit: Option<&Path>,
    precision: Option<TripoSplatBurnpackPrecision>,
) -> RuntimeResult<PathBuf> {
    if let Some(path) = explicit {
        return burn_triposplat::resolve_triposplat_weights_root(Some(path))
            .map_err(RuntimeError::new);
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let prefer_f16 = !matches!(precision, Some(TripoSplatBurnpackPrecision::F32));
        resolve_or_bootstrap_triposplat_root(prefer_f16).map_err(|err| {
            RuntimeError::new(format!(
                "failed to prepare TripoSplat cache bootstrap: {err}"
            ))
        })
    }

    #[cfg(target_arch = "wasm32")]
    {
        let _ = precision;
        burn_triposplat::resolve_triposplat_weights_root(None).map_err(RuntimeError::new)
    }
}

fn resolve_triposplat_precision(
    root: &Path,
    requested: Option<TripoSplatBurnpackPrecision>,
) -> RuntimeResult<TripoSplatBurnpackPrecision> {
    if let Some(precision) = requested {
        TripoSplatArtifactSet::new(root, precision)
            .validate_burnpacks()
            .map_err(RuntimeError::new)?;
        return Ok(precision);
    }

    let f32 = TripoSplatArtifactSet::new(root, TripoSplatBurnpackPrecision::F32);
    if f32.validate_burnpacks().is_ok() {
        return Ok(TripoSplatBurnpackPrecision::F32);
    }
    let f16 = TripoSplatArtifactSet::new(root, TripoSplatBurnpackPrecision::F16);
    if f16.validate_burnpacks().is_ok() {
        return Ok(TripoSplatBurnpackPrecision::F16);
    }
    f16.validate_burnpacks().map_err(RuntimeError::new)?;
    Ok(TripoSplatBurnpackPrecision::F16)
}

fn validate_triposplat_backend_support(
    backend: InferenceBackend,
    precision: Option<TripoSplatBurnpackPrecision>,
) -> RuntimeResult<()> {
    let _ = backend;
    #[cfg(not(feature = "cuda"))]
    let _ = precision;
    #[cfg(feature = "cuda")]
    if matches!(backend, InferenceBackend::Cuda)
        && matches!(precision, Some(TripoSplatBurnpackPrecision::F16))
    {
        return Err(RuntimeError::new(
            "TripoSplat native CUDA f16 artifacts remain disabled for now; use f32 artifacts for the validated CUDA path",
        ));
    }
    Ok(())
}

fn backend_float_label<B: Backend>() -> &'static str {
    let name = std::any::type_name::<B>().to_ascii_lowercase();
    if name.contains("f16") {
        "f16"
    } else if name.contains("bf16") {
        "bf16"
    } else {
        "f32"
    }
}

#[cfg(feature = "cuda")]
fn triposplat_compute_dtype_for_precision(
    precision: TripoSplatBurnpackPrecision,
) -> Option<burn::tensor::FloatDType> {
    match precision {
        TripoSplatBurnpackPrecision::F32 => None,
        TripoSplatBurnpackPrecision::F16 => Some(burn::tensor::FloatDType::F32),
    }
}

#[cfg(feature = "cuda")]
fn flush_cuda_upload_queue_after_runtime_triposplat_load_event(
    device: &<CudaBackend as BackendTypes>::Device,
    event: burn_triposplat::import::TripoSplatRuntimeLoadEvent,
) -> Result<(), Box<dyn std::error::Error>> {
    use cubecl::Runtime;

    let label = event.label();
    cubecl::cuda::CudaRuntime::client(device).flush().map_err(
        |err| -> Box<dyn std::error::Error> {
            format!("failed to flush CUDA upload queue after {label}: {err}").into()
        },
    )?;
    log::debug!("flushed CUDA upload queue after TripoSplat {label}");
    Ok(())
}

#[cfg(feature = "cuda")]
fn preflight_triposplat_cuda_backend() -> RuntimeResult<()> {
    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let device = <CudaBackend as BackendTypes>::Device::default();
        let lhs = Tensor::<CudaBackend, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
        let rhs = Tensor::<CudaBackend, 2>::from_floats([[5.0, 6.0], [7.0, 8.0]], &device);
        lhs.matmul(rhs)
            .try_into_data()
            .map_err(|err| {
                format!(
                    "failed to execute/read CUDA preflight tensor: {}",
                    summarize_cuda_execution_error(&format!("{err:?}"))
                )
            })?
            .convert::<f32>()
            .to_vec::<f32>()
            .map_err(|err| format!("failed to read CUDA preflight tensor: {err:?}"))
    }));
    std::panic::set_hook(previous_hook);

    let values = match result {
        Ok(Ok(values)) => values,
        Ok(Err(err)) => {
            return Err(RuntimeError::new(format!(
                "TripoSplat CUDA runtime preflight failed before model loading: {err}"
            )));
        }
        Err(payload) => {
            let detail = summarize_cuda_execution_error(&panic_payload_message(&payload));
            return Err(RuntimeError::new(format!(
                "TripoSplat CUDA runtime preflight failed before model loading: {detail}"
            )));
        }
    };

    let expected = [19.0, 22.0, 43.0, 50.0];
    if values.len() != expected.len() {
        return Err(RuntimeError::new(format!(
            "TripoSplat CUDA runtime preflight returned {} values, expected {}",
            values.len(),
            expected.len()
        )));
    }
    for (value, expected) in values.iter().zip(expected.iter()) {
        if (value - expected).abs() > 1e-2 {
            return Err(RuntimeError::new(format!(
                "TripoSplat CUDA runtime preflight produced {value}, expected {expected}"
            )));
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn panic_payload_message(payload: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        return (*message).to_string();
    }
    if let Some(message) = payload.downcast_ref::<String>() {
        return message.clone();
    }
    "panic payload was not a string".to_string()
}

fn triposplat_options_from_config(config: &RuntimeConfig) -> TripoSplatOptions {
    TripoSplatOptions {
        steps: config.triposplat_num_steps,
        guidance_scale: config.triposplat_guidance_scale,
        shift: config.triposplat_shift,
        seed: config.seed.unwrap_or(DEFAULT_SEED),
        num_gaussians: config.triposplat_num_gaussians,
        erode_radius: config.triposplat_erode_radius,
        cfg_mode: CfgPredictionMode::default(),
        attention_query_chunk_tokens: None,
    }
}

fn normalize_triposplat_counts(raw: &[usize], default_count: usize) -> RuntimeResult<Vec<usize>> {
    let counts = if raw.is_empty() {
        vec![default_count]
    } else {
        raw.to_vec()
    };
    let counts = counts
        .into_iter()
        .map(|count| normalize_num_gaussians(count).map_err(RuntimeError::new))
        .collect::<RuntimeResult<Vec<_>>>()?;
    if counts.is_empty() {
        Err(RuntimeError::new("num_gaussians list must not be empty"))
    } else {
        Ok(counts)
    }
}

#[cfg(feature = "trellis")]
fn resolve_trellis_runtime_roots(
    explicit_weights: Option<&Path>,
    explicit_image_large: Option<&Path>,
) -> RuntimeResult<(PathBuf, Option<PathBuf>)> {
    let resolved_explicit_weights =
        explicit_weights.map(|path| resolve_trellis2_weights_root(Some(path)));
    let resolved_explicit_image_large =
        explicit_image_large.map(|path| resolve_trellis2_image_large_root(Some(path)));

    if resolved_explicit_weights.is_some() || resolved_explicit_image_large.is_some() {
        let weights_root = resolved_explicit_weights.unwrap_or_else(|| {
            if let Some(path) = explicit_weights {
                path.to_path_buf()
            } else {
                resolve_trellis2_weights_root(None)
            }
        });
        let image_large_root = resolved_explicit_image_large.or_else(|| {
            let resolved = resolve_trellis2_image_large_root(None);
            resolved.exists().then_some(resolved)
        });
        return Ok((weights_root, image_large_root));
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        return resolve_or_bootstrap_trellis_roots(true).map_err(|err| {
            RuntimeError::new(format!("failed to prepare Trellis2 cache bootstrap: {err}"))
        });
    }

    #[cfg(target_arch = "wasm32")]
    {
        let weights_root = resolve_trellis2_weights_root(None);
        let image_large_root = resolve_trellis2_image_large_root(None);
        let image_large_root = image_large_root.exists().then_some(image_large_root);
        Ok((weights_root, image_large_root))
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

#[cfg(feature = "trellis")]
fn map_trellis_compute_profile(
    value: TrellisComputeProfile,
) -> burn_trellis::TrellisComputeProfile {
    match value {
        TrellisComputeProfile::ReferenceF32 => burn_trellis::TrellisComputeProfile::ReferenceF32,
        TrellisComputeProfile::WgpuFastMixedF16 => {
            burn_trellis::TrellisComputeProfile::WgpuFastMixedF16
        }
        TrellisComputeProfile::WgpuFastSparseSelfF16 => {
            burn_trellis::TrellisComputeProfile::WgpuFastSparseSelfF16
        }
        TrellisComputeProfile::WgpuFastSparseCrossF16 => {
            burn_trellis::TrellisComputeProfile::WgpuFastSparseCrossF16
        }
        TrellisComputeProfile::WgpuFastF16Tail1F32 => {
            burn_trellis::TrellisComputeProfile::WgpuFastF16Tail1F32
        }
        TrellisComputeProfile::WgpuFastF16Tail2F32 => {
            burn_trellis::TrellisComputeProfile::WgpuFastF16Tail2F32
        }
        TrellisComputeProfile::WgpuFastF16Tail4F32 => {
            burn_trellis::TrellisComputeProfile::WgpuFastF16Tail4F32
        }
        TrellisComputeProfile::WgpuFastF16Tail6F32 => {
            burn_trellis::TrellisComputeProfile::WgpuFastF16Tail6F32
        }
        TrellisComputeProfile::WgpuFastF16 => burn_trellis::TrellisComputeProfile::WgpuFastF16,
    }
}

#[cfg(feature = "trellis")]
fn validate_trellis_runtime_sources(
    requested_device: TrellisDevice,
    sparse_source: &str,
    decode_source: &str,
) -> RuntimeResult<()> {
    if sparse_source == "synthetic" {
        return Err(RuntimeError::new(format!(
            "Trellis2 runtime entered synthetic sparse fallback (requested_device={}, sparse_source={}, decode_source={}); refusing degraded output.",
            requested_device.as_str(),
            sparse_source,
            decode_source
        )));
    }

    if matches!(requested_device, TrellisDevice::Wgpu) && sparse_source != "runtime_model_wgpu" {
        return Err(RuntimeError::new(format!(
            "Trellis2 runtime sparse source mismatch for WGPU request (requested_device=wgpu, sparse_source={}, decode_source={}); refusing silent fallback.",
            sparse_source, decode_source
        )));
    }

    if decode_source != "runtime" {
        return Err(RuntimeError::new(format!(
            "Trellis2 runtime entered decode fallback path (requested_device={}, sparse_source={}, decode_source={}); refusing degraded output.",
            requested_device.as_str(),
            sparse_source,
            decode_source
        )));
    }

    Ok(())
}

#[cfg(feature = "cuda")]
fn summarize_cuda_execution_error(raw: &str) -> String {
    if raw.contains("invalid value for --gpu-architecture") {
        return "NVRTC rejected CubeCL's target GPU architecture (`invalid value for --gpu-architecture (-arch)`). This usually means the active CUDA/NVRTC runtime is too old for the installed GPU; on Blackwell, put a CUDA 12.9+ NVRTC runtime first on LD_LIBRARY_PATH and set CUDA_PATH to the matching toolkit before retrying.".to_string();
    }

    let lines = raw
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .filter(|line| !line.starts_with("[Source]"))
        .take(8)
        .collect::<Vec<_>>();
    if lines.is_empty() {
        return "CUDA execution failed without a diagnostic message".to_string();
    }
    let mut summary = lines.join("; ");
    const MAX_LEN: usize = 700;
    if summary.len() > MAX_LEN {
        summary = summary.chars().take(MAX_LEN).collect();
        summary.push_str("...");
    }
    summary
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
        SynthesisModel::Triposplat => "triposplat",
    }
}

fn synthesis_models_label(models: &[SynthesisModel]) -> String {
    models
        .iter()
        .map(|model| synthesis_model_label(*model))
        .collect::<Vec<_>>()
        .join(",")
}

fn triposplat_counts_label(counts: &[usize]) -> String {
    counts
        .iter()
        .map(|count| count.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn triposplat_splats_detail(splats: &[GaussianSplatCloud]) -> String {
    if splats.len() == 1 {
        return format!("splats={}", splats[0].len());
    }
    let counts = splats
        .iter()
        .map(|cloud| cloud.len().to_string())
        .collect::<Vec<_>>()
        .join(",");
    format!("splats=[{counts}]")
}

fn triposplat_latent_cache_key(
    prepared: &PreparedImageData,
    options: &TripoSplatOptions,
) -> TripoSplatLatentCacheKey {
    let mut hasher = Sha256::new();
    hasher.update((prepared.width as u64).to_le_bytes());
    hasher.update((prepared.height as u64).to_le_bytes());
    update_f32_slice_hash(&mut hasher, prepared.data.as_slice());
    update_optional_f32_slice_hash(&mut hasher, prepared.alpha_mask.as_deref());
    update_optional_f32_slice_hash(&mut hasher, prepared.alpha_probs.as_deref());
    TripoSplatLatentCacheKey {
        image_hash: hasher.finalize().into(),
        seed: options.seed,
        steps: options.steps,
        guidance_scale_bits: options.guidance_scale.to_bits(),
        shift_bits: options.shift.to_bits(),
        cfg_mode: options.cfg_mode,
        attention_query_chunk_tokens: options.attention_query_chunk_tokens,
    }
}

fn update_optional_f32_slice_hash(hasher: &mut Sha256, values: Option<&[f32]>) {
    match values {
        Some(values) => {
            hasher.update([1]);
            update_f32_slice_hash(hasher, values);
        }
        None => hasher.update([0]),
    }
}

fn update_f32_slice_hash(hasher: &mut Sha256, values: &[f32]) {
    hasher.update((values.len() as u64).to_le_bytes());
    for value in values {
        hasher.update(value.to_le_bytes());
    }
}

fn resize_prepared_image_for_triposplat(
    prepared: PreparedImageData,
    canvas_size: usize,
) -> PreparedImageData {
    let in_height = prepared.height;
    let in_width = prepared.width;
    let same_size = in_width == canvas_size && in_height == canvas_size;
    let data = if same_size {
        prepared.data
    } else {
        resize_chw_align_corners_false(
            &prepared.data,
            3,
            in_height,
            in_width,
            canvas_size,
            canvas_size,
            InterpolateMode::Lanczos3,
        )
    };
    let alpha_mask = resize_prepared_alpha_channel_for_triposplat(
        prepared.alpha_mask,
        in_height,
        in_width,
        canvas_size,
    );
    let alpha_probs = resize_prepared_alpha_channel_for_triposplat(
        prepared.alpha_probs,
        in_height,
        in_width,
        canvas_size,
    );
    normalize_prepared_image_rgb_0_1(PreparedImageData {
        data,
        width: canvas_size,
        height: canvas_size,
        alpha_mask,
        alpha_probs,
        bbox: if same_size { prepared.bbox } else { None },
    })
}

fn resize_prepared_alpha_channel_for_triposplat(
    alpha: Option<Vec<f32>>,
    in_height: usize,
    in_width: usize,
    canvas_size: usize,
) -> Option<Vec<f32>> {
    let alpha = alpha?;
    if alpha.len() != in_height.saturating_mul(in_width) {
        return None;
    }
    if in_width == canvas_size && in_height == canvas_size {
        return Some(alpha);
    }
    Some(resize_chw_align_corners_false(
        &alpha,
        1,
        in_height,
        in_width,
        canvas_size,
        canvas_size,
        InterpolateMode::Lanczos3,
    ))
}

fn normalize_prepared_image_rgb_0_1(mut prepared: PreparedImageData) -> PreparedImageData {
    for value in &mut prepared.data {
        *value = (*value / 255.0).clamp(0.0, 1.0);
    }
    prepared
}

fn run_triposplat_preprocessed_many<B: Backend>(
    state: &mut BackendTripoSplatState<B>,
    prepared: &PreparedImageData,
    num_gaussians: &[usize],
    mut options: TripoSplatOptions,
    progress: &ProgressRun,
) -> RuntimeResult<Vec<GaussianSplatCloud>> {
    let counts = normalize_triposplat_counts(num_gaussians, options.num_gaussians)?;
    options.num_gaussians = counts[0];
    let cache_key = triposplat_latent_cache_key(prepared, &options);

    let latent = if let Some(cache) = state
        .latent_cache
        .as_ref()
        .filter(|cache| cache.key == cache_key)
    {
        progress.stage_started(
            "triposplat.sample",
            Some(0),
            Some(format!(
                "cache=hit steps={} guidance_scale={:.3} shift={:.3}",
                options.steps, options.guidance_scale, options.shift
            )),
        );
        progress.stage_completed(
            "triposplat.sample",
            Some(0),
            0.0,
            Some("cache=hit reused_latent=true".to_string()),
        );
        cache.latent.clone()
    } else {
        progress.stage_started(
            "triposplat.prepare_tensor",
            None,
            Some(format!("image={}x{}", prepared.width, prepared.height)),
        );
        let prepare_start = Instant::now();
        let image = prepared.to_tensor(&state.device);
        progress.stage_completed(
            "triposplat.prepare_tensor",
            None,
            prepare_start.elapsed().as_secs_f64() * 1000.0,
            None,
        );
        B::memory_cleanup(&state.device);

        progress.stage_started(
            "triposplat.encode",
            None,
            Some(format!("seed={}", options.seed)),
        );
        let encode_start = Instant::now();
        B::seed(&state.device, options.seed);
        let condition = state.components.encode_preprocessed_image_random(image);
        progress.stage_completed(
            "triposplat.encode",
            None,
            encode_start.elapsed().as_secs_f64() * 1000.0,
            Some(format!(
                "feature1_tokens={} feature2={}",
                condition.feature1.dims()[1],
                condition
                    .feature2
                    .as_ref()
                    .map(|feature2| format!("tokens={}", feature2.dims()[1]))
                    .unwrap_or_else(|| "none".to_string())
            )),
        );
        B::memory_cleanup(&state.device);

        progress.stage_started(
            "triposplat.sample",
            Some(options.steps),
            Some(format!(
                "cache=miss guidance_scale={:.3} shift={:.3}",
                options.guidance_scale, options.shift
            )),
        );
        let sample_start = Instant::now();
        let latent = state
            .components
            .sample_latent_random(condition, options)
            .latent;
        let sample_elapsed_ms = sample_start.elapsed().as_secs_f64() * 1000.0;
        progress.stage_completed(
            "triposplat.sample",
            Some(options.steps),
            sample_elapsed_ms,
            Some(avg_step_detail(sample_elapsed_ms, options.steps, "flow")),
        );
        B::memory_cleanup(&state.device);
        state.latent_cache = Some(TripoSplatLatentCache {
            key: cache_key,
            latent: latent.clone(),
        });
        latent
    };

    progress.stage_started(
        "triposplat.decode",
        None,
        Some(format!(
            "num_gaussians={} guidance_scale={:.3} shift={:.3} erode_radius={}",
            triposplat_counts_label(&counts),
            options.guidance_scale,
            options.shift,
            options.erode_radius
        )),
    );
    let decode_start = Instant::now();
    let output = state
        .components
        .decode_latent_many(latent, counts.iter().copied(), options)
        .map_err(|err| RuntimeError::new(format!("TripoSplat inference failed: {err}")))?;
    let decode_readbacks = output.decode_readbacks;
    let splats = output.splats;
    progress.stage_completed(
        "triposplat.decode",
        None,
        decode_start.elapsed().as_secs_f64() * 1000.0,
        Some(format!(
            "{} readbacks={} sync_readbacks={} async_readbacks={} readback_bytes={}",
            triposplat_splats_detail(&splats),
            decode_readbacks.total_readbacks(),
            decode_readbacks.sync_readbacks,
            decode_readbacks.async_readbacks,
            decode_readbacks.bytes
        )),
    );
    B::memory_cleanup(&state.device);
    Ok(splats)
}

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
            let cpu_device = <NdArray<f32> as BackendTypes>::Device::default();
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
        .extract_flash_grid_from_latents(&output.latents, &config.flash_extract)
        .map_err(|err| RuntimeError::new(format!("TripoSG geometry extraction failed: {err}")))?;
    drop(output);
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
        // Keep RMBG-1.4 bootstrap on f32 artifacts for runtime stability: current
        // burn foreground inference paths use f32 backends and can fail or diverge
        // with cache roots that only contain f16 part shards.
        let cache_root = resolve_or_bootstrap_rmbg14_root(false)
            .map_err(|err| RuntimeError::new(format!("failed to prepare RMBG-1.4 cache: {err}")))?;
        if rmbg14_root_has_full_burnpack(cache_root.as_path()) {
            return Ok(cache_root);
        }
        let fallback = resolve_rmbg_weights_root();
        if rmbg14_root_has_full_burnpack(fallback.as_path()) {
            return Ok(fallback);
        }
        Ok(cache_root)
    }

    #[cfg(target_arch = "wasm32")]
    {
        Ok(resolve_rmbg_weights_root())
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn rmbg14_root_has_full_burnpack(root: &Path) -> bool {
    root.join("model.bpk").exists() || root.join("model_f16.bpk").exists()
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
    use std::{
        fs,
        io::Cursor,
        path::{Path, PathBuf},
        sync::{Arc, Mutex},
        time::{SystemTime, UNIX_EPOCH},
    };

    use image::{DynamicImage, Rgba};

    use super::*;

    #[cfg(any(feature = "wgpu", feature = "cuda"))]
    fn assert_backend_type_uses_fusion<B>(label: &str) {
        let type_name = std::any::type_name::<B>();
        assert!(
            type_name.contains("burn_fusion"),
            "{label} must use Burn fusion, got backend type {type_name}"
        );
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn wgpu_backend_alias_uses_burn_fusion() {
        assert_backend_type_uses_fusion::<WgpuBackend>("WGPU backend");
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn triposplat_wgpu_backend_alias_uses_burn_fusion() {
        assert_backend_type_uses_fusion::<WgpuTripoSplatBackend>("TripoSplat WGPU backend");
        assert_backend_type_uses_fusion::<WgpuTripoSplatBackendF16>("TripoSplat WGPU f16 backend");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_backend_alias_uses_burn_fusion() {
        assert_backend_type_uses_fusion::<CudaBackend>("CUDA backend");
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn wgpu_tensor_smoke() {
        if std::env::var("BURN_WGPU_SMOKE").is_err() {
            eprintln!("skipping: set BURN_WGPU_SMOKE=1 to run WGPU tensor smoke");
            return;
        }

        let device = <WgpuBackend as BackendTypes>::Device::default();
        eprintln!(
            "[wgpu_tensor_smoke] backend={}",
            <WgpuBackend as Backend>::name(&device)
        );
        let lhs = Tensor::<WgpuBackend, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
        let rhs = Tensor::<WgpuBackend, 2>::from_floats([[5.0, 6.0], [7.0, 8.0]], &device);
        let values = (lhs + rhs)
            .try_into_data()
            .expect("WGPU tensor add should execute")
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("WGPU tensor add values should read back");

        assert_eq!(values, vec![6.0, 8.0, 10.0, 12.0]);
        <WgpuBackend as Backend>::sync(&device).expect("WGPU backend should sync cleanly");
    }

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
    fn shared_preprocess_accepts_odd_resolution_rgba_for_mesh_and_triposplat() {
        let path = unique_temp_png_path();
        let mut input = RgbaImage::new(37, 23);
        for y in 0..23 {
            for x in 0..37 {
                let alpha = if (5..32).contains(&x) && (3..20).contains(&y) {
                    255
                } else {
                    0
                };
                input.put_pixel(x, y, Rgba([120, 140, 200, alpha]));
            }
        }
        DynamicImage::ImageRgba8(input)
            .save_with_format(&path, ImageFormat::Png)
            .expect("failed to write odd-resolution RGBA fixture");

        let mut runtime = SynthRuntime::new(RuntimeConfig {
            backend: InferenceBackend::Cpu,
            ..RuntimeConfig::default()
        });
        let mesh_prepared = runtime
            .prepare_image_for_mesh(&path, ForegroundModel::Rmbg14, InferenceBackend::Cpu)
            .expect("mesh preprocessing should use RGBA alpha without RMBG");
        let splat_prepared = runtime
            .prepare_image_for_triposplat(&path, ForegroundModel::Rmbg14, InferenceBackend::Cpu)
            .expect("TripoSplat preprocessing should use the same RGBA alpha path");
        let splat_resized = resize_prepared_image_for_triposplat(splat_prepared, 16);

        assert_eq!(
            mesh_prepared
                .alpha_mask
                .as_ref()
                .expect("mesh alpha mask")
                .len(),
            37 * 23
        );
        assert_eq!(splat_resized.width, 16);
        assert_eq!(splat_resized.height, 16);
        assert_eq!(splat_resized.data.len(), 3 * 16 * 16);

        let _ = fs::remove_file(path);
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

    #[test]
    fn triposplat_asset_dry_run_returns_gaussian_splats() {
        let mut runtime = SynthRuntime::new(RuntimeConfig {
            backend: InferenceBackend::Cpu,
            ..RuntimeConfig::default()
        });
        let output = runtime
            .synthesize_asset(AssetRequest {
                image: ImageSource::from_path("unused.png"),
                foreground_model: Some(ForegroundModel::Rmbg14),
                synthesis_models: Some(vec![SynthesisModel::Triposplat]),
                backend: Some(InferenceBackend::Cpu),
                dry_run: true,
            })
            .expect("dry-run TripoSplat asset should succeed");

        match output.asset {
            SynthesisAsset::GaussianSplat(splats) => assert!(!splats.is_empty()),
            SynthesisAsset::Mesh(_) => panic!("TripoSplat asset dry-run returned a mesh"),
        }
        assert_eq!(output.synthesis_backend, SynthesisModel::Triposplat);
        assert_eq!(output.backend, InferenceBackend::Cpu);
    }

    #[test]
    fn triposplat_splat_dry_run_normalizes_gaussian_count() {
        let mut runtime = SynthRuntime::new(RuntimeConfig {
            backend: InferenceBackend::Cpu,
            ..RuntimeConfig::default()
        });
        let output = runtime
            .synthesize_splats(SplatRequest {
                image: ImageSource::from_path("unused.png"),
                foreground_model: Some(ForegroundModel::Rmbg14),
                backend: Some(InferenceBackend::Cpu),
                num_gaussians: vec![32_769],
                dry_run: true,
            })
            .expect("dry-run TripoSplat splats should succeed");

        assert_eq!(output.num_gaussians, vec![32_768]);
        assert_eq!(output.splats.len(), 1);
        assert!(!output.splats[0].is_empty());
        assert_eq!(output.synthesis_backend, SynthesisModel::Triposplat);
        assert_eq!(output.backend, InferenceBackend::Cpu);
    }

    #[test]
    fn triposplat_splat_dry_run_rejects_multi_density() {
        let mut runtime = SynthRuntime::new(RuntimeConfig {
            backend: InferenceBackend::Cpu,
            ..RuntimeConfig::default()
        });
        let err = runtime
            .synthesize_splats(SplatRequest {
                image: ImageSource::from_path("unused.png"),
                foreground_model: Some(ForegroundModel::Rmbg14),
                backend: Some(InferenceBackend::Cpu),
                num_gaussians: vec![32_768, 65_536],
                dry_run: true,
            })
            .expect_err("multi-density dry-run should fail explicitly");

        assert!(
            err.to_string().contains("dry-run emits one"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn triposplat_native_runtime_emits_stage_level_progress() {
        type TestBackend = NdArray<f32>;

        let device = <TestBackend as BackendTypes>::Device::default();
        let flow_config = burn_triposplat::LatentSeqMmFlowModelConfig {
            cond_channels: 64,
            cond2_channels: Some(128),
            ..burn_triposplat::LatentSeqMmFlowModelConfig::tiny_for_tests()
        };
        let components = burn_triposplat::TripoSplatRuntimeComponents::<TestBackend> {
            dinov3: burn_dino::model::dinov3::DinoV3Config::tiny_for_tests(32, 16).init(&device),
            flux2_vae_encoder: burn_flux::Flux2VaeEncoderConfig::flux2().init(&device),
            flow: flow_config.init(&device),
            decoder: burn_triposplat::OctreeGaussianDecoder::new(
                &device,
                burn_triposplat::OctreeProbabilityFixedlenDecoderConfig::tiny_for_tests(),
                burn_triposplat::ElasticGaussianFixedlenDecoderConfig::tiny_for_tests(),
            ),
        };
        let mut state = BackendTripoSplatState {
            device,
            components,
            weights_root: PathBuf::from("unused"),
            precision: TripoSplatBurnpackPrecision::F32,
            decoder_compute_label: "f32",
            latent_cache: None,
        };
        let prepared = PreparedImageData {
            data: vec![0.5; 3 * 32 * 32],
            width: 32,
            height: 32,
            alpha_mask: None,
            alpha_probs: None,
            bbox: None,
        };
        let events = Arc::new(Mutex::new(Vec::new()));
        let observer = RuntimeProgressObserver::with_callback(
            crate::progress::ProgressVerbosity::Stages,
            1,
            {
                let events = events.clone();
                Arc::new(move |event| {
                    events
                        .lock()
                        .expect("progress events lock")
                        .push(event.clone());
                })
            },
        );
        let progress = ProgressRun::new(&observer, "splat", None);

        let splats = run_triposplat_preprocessed_many(
            &mut state,
            &prepared,
            &[32_768],
            TripoSplatOptions {
                steps: 1,
                num_gaussians: 32_768,
                ..TripoSplatOptions::default()
            },
            &progress,
        )
        .expect("tiny TripoSplat runtime should produce splats");

        assert_eq!(splats.len(), 1);
        assert_eq!(splats[0].len(), 32_768);
        let completed = events
            .lock()
            .expect("progress events lock")
            .iter()
            .filter_map(|event| match event {
                RuntimeProgressEvent::StageCompleted { stage, .. } => Some(*stage),
                _ => None,
            })
            .collect::<Vec<_>>();
        for expected in [
            "triposplat.prepare_tensor",
            "triposplat.encode",
            "triposplat.sample",
            "triposplat.decode",
        ] {
            assert!(
                completed.contains(&expected),
                "missing TripoSplat stage progress event {expected}; got {completed:?}"
            );
        }
        assert!(
            !completed.contains(&"triposplat.infer"),
            "TripoSplat runtime should expose canonical stage timings instead of one monolithic infer stage"
        );
    }

    #[test]
    fn triposplat_latent_cache_key_ignores_decode_only_gaussian_count() {
        let prepared = PreparedImageData {
            data: vec![0.5; 3 * 2 * 2],
            width: 2,
            height: 2,
            alpha_mask: None,
            alpha_probs: None,
            bbox: None,
        };
        let mut first = TripoSplatOptions {
            steps: 5,
            guidance_scale: 3.0,
            shift: 3.0,
            seed: 7,
            num_gaussians: 32_768,
            ..TripoSplatOptions::default()
        };
        let mut second = first;
        second.num_gaussians = 262_144;

        assert_eq!(
            triposplat_latent_cache_key(&prepared, &first),
            triposplat_latent_cache_key(&prepared, &second),
            "Gaussian count is a decode-only setting and must not invalidate sampled latent cache"
        );

        first.guidance_scale = 4.0;
        assert_ne!(
            triposplat_latent_cache_key(&prepared, &first),
            triposplat_latent_cache_key(&prepared, &second),
            "Guidance affects sampling and must invalidate sampled latent cache"
        );

        let mut third = second;
        third.cfg_mode = CfgPredictionMode::Separate;
        assert_ne!(
            triposplat_latent_cache_key(&prepared, &second),
            triposplat_latent_cache_key(&prepared, &third),
            "CFG execution mode affects sampling and must invalidate sampled latent cache"
        );
    }

    #[test]
    fn triposplat_prepared_image_is_normalized_to_unit_rgb() {
        let prepared = PreparedImageData {
            data: vec![0.0, 127.5, 255.0],
            width: 1,
            height: 1,
            alpha_mask: None,
            alpha_probs: None,
            bbox: None,
        };

        let normalized = resize_prepared_image_for_triposplat(prepared, 1);

        assert_eq!(normalized.data, vec![0.0, 0.5, 1.0]);
    }

    #[test]
    fn triposplat_prepared_resize_drops_original_size_alpha_metadata() {
        let prepared = PreparedImageData {
            data: vec![255.0; 3 * 3 * 2],
            width: 3,
            height: 2,
            alpha_mask: Some(vec![1.0; 2 * 2]),
            alpha_probs: Some(vec![0.5; 2 * 2]),
            bbox: Some([0, 0, 2, 2]),
        };

        let resized = resize_prepared_image_for_triposplat(prepared, 2);

        assert_eq!(resized.width, 2);
        assert_eq!(resized.height, 2);
        assert_eq!(resized.data.len(), 3 * 2 * 2);
        assert!(
            resized.alpha_mask.is_none(),
            "alpha_mask is original-image metadata and must not be resized with prepared RGB dimensions"
        );
        assert!(
            resized.alpha_probs.is_none(),
            "alpha_probs is original-image metadata and must not be resized with prepared RGB dimensions"
        );
    }

    #[test]
    fn triposplat_prepare_config_matches_upstream_contract() {
        let config = triposplat_prepare_image_config(1);

        assert_eq!(config.bg_color, [0.0, 0.0, 0.0]);
        assert_eq!(config.padding_ratio, 0.1);
        assert_eq!(config.max_dimension, usize::MAX);
        assert_eq!(config.resize_shorter_to, Some(TRIPOSPLAT_CANVAS_SIZE));
        assert_eq!(config.alpha_erode_radius, 1);
        assert_eq!(
            TRIPOSPLAT_CANVAS_SIZE,
            burn_triposplat::TRIPOSPLAT_CANONICAL_CANVAS_SIZE
        );
        assert_eq!(burn_triposplat::TRIPOSPLAT_FAST_VAE_TOKEN_LENGTH, 4096);
        assert_eq!(burn_triposplat::TRIPOSPLAT_FAST_DINOV3_TOKEN_LENGTH, 4101);
        assert_eq!(burn_triposplat::DEFAULT_Q_TOKEN_LENGTH, 8192);
    }

    #[test]
    fn runtime_config_has_triposplat_specific_upstream_defaults() {
        let config = RuntimeConfig::default();

        assert_eq!(config.num_steps, DEFAULT_NUM_STEPS);
        assert_eq!(config.guidance_scale, DEFAULT_GUIDANCE_SCALE);
        assert_eq!(
            config.triposplat_num_steps,
            burn_triposplat::DEFAULT_NUM_STEPS
        );
        assert_eq!(
            config.triposplat_guidance_scale,
            burn_triposplat::DEFAULT_GUIDANCE_SCALE
        );

        let options = triposplat_options_from_config(&config);
        assert_eq!(options.steps, burn_triposplat::DEFAULT_NUM_STEPS);
        assert_eq!(
            options.guidance_scale,
            burn_triposplat::DEFAULT_GUIDANCE_SCALE
        );
    }

    #[test]
    fn triposplat_mesh_request_errors_without_fallback() {
        let mut runtime = SynthRuntime::new(RuntimeConfig {
            backend: InferenceBackend::Cpu,
            ..RuntimeConfig::default()
        });
        let progress = ProgressRun::new(&runtime.config.progress, "mesh", None);
        let err = runtime
            .infer_mesh(
                Path::new("unused.png"),
                ForegroundModel::Rmbg14,
                InferenceBackend::Cpu,
                &[SynthesisModel::Triposplat, SynthesisModel::Triposg],
                &progress,
            )
            .expect_err("TripoSplat mesh requests should fail instead of substituting TripoSG");
        assert!(
            err.to_string().contains("Gaussian splats, not meshes"),
            "unexpected error: {err}"
        );
        assert!(
            !err.to_string().contains("falling back"),
            "unexpected fallback wording: {err}"
        );
    }

    #[test]
    fn triposplat_precision_default_prefers_f32_when_both_are_available() {
        let root = fake_triposplat_root(&[
            TripoSplatBurnpackPrecision::F16,
            TripoSplatBurnpackPrecision::F32,
        ]);
        let precision = resolve_triposplat_precision(
            &root,
            RuntimeConfig::default().triposplat_weights_precision,
        )
        .expect("default TripoSplat precision should resolve");

        assert_eq!(precision, TripoSplatBurnpackPrecision::F32);
        fs::remove_dir_all(root).expect("cleanup fake TripoSplat root");
    }

    #[test]
    fn runtime_config_defaults_to_reference_trellis_profile() {
        assert_eq!(
            RuntimeConfig::default().trellis_compute_profile,
            TrellisComputeProfile::ReferenceF32
        );
    }

    #[test]
    fn triposplat_precision_explicit_f16_is_respected() {
        let root = fake_triposplat_root(&[
            TripoSplatBurnpackPrecision::F16,
            TripoSplatBurnpackPrecision::F32,
        ]);
        let precision = resolve_triposplat_precision(&root, Some(TripoSplatBurnpackPrecision::F16))
            .expect("explicit f16 TripoSplat precision should resolve");

        assert_eq!(precision, TripoSplatBurnpackPrecision::F16);
        fs::remove_dir_all(root).expect("cleanup fake TripoSplat root");
    }

    #[test]
    fn triposplat_native_wgpu_is_supported_by_backend_validation() {
        validate_triposplat_backend_support(
            InferenceBackend::Wgpu,
            Some(TripoSplatBurnpackPrecision::F16),
        )
        .expect("native WGPU TripoSplat f16 artifacts should be accepted by validation");
        validate_triposplat_backend_support(
            InferenceBackend::Wgpu,
            Some(TripoSplatBurnpackPrecision::F32),
        )
        .expect("native WGPU TripoSplat f32 artifacts should be accepted by validation");
        #[cfg(feature = "cuda")]
        validate_triposplat_backend_support(
            InferenceBackend::Cuda,
            Some(TripoSplatBurnpackPrecision::F16),
        )
        .expect_err("native CUDA TripoSplat f16 artifacts should fail fast until validated");
        #[cfg(feature = "cuda")]
        validate_triposplat_backend_support(
            InferenceBackend::Cuda,
            Some(TripoSplatBurnpackPrecision::F32),
        )
        .expect("native CUDA TripoSplat f32 artifacts should use the validated CUDA path");
        validate_triposplat_backend_support(
            InferenceBackend::Cpu,
            Some(TripoSplatBurnpackPrecision::F16),
        )
        .expect("CPU f16 storage path should remain loadable");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn triposplat_cuda_preflight_smoke() {
        if std::env::var("BURN_CUDA_SMOKE").is_err() {
            eprintln!("skipping: set BURN_CUDA_SMOKE=1 to run TripoSplat CUDA preflight smoke");
            return;
        }

        preflight_triposplat_cuda_backend().expect("TripoSplat CUDA preflight should pass");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn triposplat_cuda_large_attention_smoke() {
        if std::env::var("BURN_CUDA_SMOKE").is_err() {
            eprintln!(
                "skipping: set BURN_CUDA_SMOKE=1 to run TripoSplat CUDA large-attention smoke"
            );
            return;
        }

        let device = <CudaBackend as BackendTypes>::Device::default();
        let q = Tensor::<CudaBackend, 4>::zeros([1, 32_768, 8, 64], &device);
        let k = Tensor::<CudaBackend, 4>::zeros([1, 32_768, 8, 64], &device);
        let v = Tensor::<CudaBackend, 4>::zeros([1, 32_768, 8, 64], &device);
        let out = burn_triposplat::components::scaled_dot_product_attention(q, k, v, 64);

        assert_eq!(out.dims(), [1, 32_768, 8, 64]);
        let sample = out
            .slice([0..1, 0..1, 0..1, 0..1])
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("attention output sample");
        assert_eq!(sample, vec![0.0]);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn triposplat_cuda_decoder_diagnostic_reference_latent() {
        if std::env::var("TRIPOSPLAT_CUDA_DECODER_DIAGNOSTIC").is_err() {
            eprintln!(
                "skipping: set TRIPOSPLAT_CUDA_DECODER_DIAGNOSTIC=1 to run TripoSplat CUDA decoder diagnostics"
            );
            return;
        }

        let stage_tensors = std::env::var("TRIPOSPLAT_STAGE_TENSORS")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(
                    "tmp/runs/20260604T074500Z_triposplat_cuda_alpha_reference/stage_tensors_f32.safetensors",
                )
            });
        let stage_tensors = workspace_relative_path(stage_tensors);
        assert!(
            stage_tensors.exists(),
            "missing TripoSplat stage tensor file {}",
            stage_tensors.display()
        );
        let weights_root = std::env::var("TRIPOSPLAT_WEIGHTS_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("crates/burn_triposplat/assets/models/TripoSplat"));
        let weights_root = workspace_relative_path(weights_root);
        let decoder_path = weights_root.join("vae/triposplat_vae_decoder.bpk");
        assert!(
            decoder_path.exists(),
            "missing TripoSplat f32 decoder burnpack {}",
            decoder_path.display()
        );

        let device = <CudaBackend as BackendTypes>::Device::default();
        let latent = read_f32_safetensor_3d::<CudaBackend>(&stage_tensors, "latent", &device)
            .expect("read upstream TripoSplat latent tensor");
        let decoder =
            burn_triposplat::import::load_triposplat_decoder_from_burnpack_file::<CudaBackend>(
                &device,
                decoder_path,
                &burn_triposplat::OctreeProbabilityFixedlenDecoderConfig::triposplat(),
                &burn_triposplat::ElasticGaussianFixedlenDecoderConfig::triposplat(),
            )
            .expect("load TripoSplat f32 decoder burnpack");

        match decoder.decode_to_cloud_with_seed_checked(latent, 32_768, 42) {
            Ok(cloud) => {
                eprintln!(
                    "[triposplat_cuda_decoder_diagnostic] decoded_splats={}",
                    cloud.len()
                );
                assert_eq!(cloud.len(), 32_768);
            }
            Err(err) => {
                eprintln!("[triposplat_cuda_decoder_diagnostic] first_failure={err}");
                assert!(
                    err.contains("gaussian_decoder") || err.contains("octree_gaussian_decoder"),
                    "diagnostic should report a checked decoder stage, got: {err}"
                );
            }
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn triposplat_cuda_stage_parity_reference_tensors() {
        if std::env::var("TRIPOSPLAT_CUDA_STAGE_PARITY").is_err() {
            eprintln!(
                "skipping: set TRIPOSPLAT_CUDA_STAGE_PARITY=1 to run TripoSplat CUDA stage parity diagnostics"
            );
            return;
        }

        let stage_tensors = default_triposplat_stage_tensors_path();
        assert!(
            stage_tensors.exists(),
            "missing TripoSplat stage tensor file {}",
            stage_tensors.display()
        );
        let weights_root = std::env::var("TRIPOSPLAT_WEIGHTS_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("crates/burn_triposplat/assets/models/TripoSplat"));
        let weights_root = workspace_relative_path(weights_root);
        assert!(
            weights_root.exists(),
            "missing TripoSplat weights root {}",
            weights_root.display()
        );
        preflight_triposplat_cuda_backend()
            .expect("TripoSplat CUDA stage parity requires a working CUDA runtime");

        let device = <CudaBackend as BackendTypes>::Device::default();
        let flow_only = std::env::var("TRIPOSPLAT_FLOW_ONLY").is_ok();
        let compute_dtype = std::env::var("TRIPOSPLAT_COMPUTE_DTYPE")
            .ok()
            .map(|value| match value.as_str() {
                "bf16" => burn::tensor::FloatDType::BF16,
                "f16" => burn::tensor::FloatDType::F16,
                "f32" => burn::tensor::FloatDType::F32,
                other => panic!(
                    "TRIPOSPLAT_COMPUTE_DTYPE must be bf16, f16, or f32 for diagnostics, got {other}"
                ),
            })
            .unwrap_or(burn::tensor::FloatDType::F32);
        let weights_precision = std::env::var("TRIPOSPLAT_WEIGHTS_PRECISION")
            .ok()
            .map(|value| match value.as_str() {
                "f16" => TripoSplatBurnpackPrecision::F16,
                "f32" => TripoSplatBurnpackPrecision::F32,
                other => panic!(
                    "TRIPOSPLAT_WEIGHTS_PRECISION must be f16 or f32 for diagnostics, got {other}"
                ),
            })
            .unwrap_or(TripoSplatBurnpackPrecision::F16);
        eprintln!(
            "[triposplat_cuda_stage_parity] weights_precision={} compute_dtype={}",
            weights_precision.as_str(),
            match compute_dtype {
                burn::tensor::FloatDType::BF16 => "bf16",
                burn::tensor::FloatDType::F16 => "f16",
                burn::tensor::FloatDType::F32 => "f32",
                _ => "other",
            }
        );
        let artifacts = TripoSplatArtifactSet::new(&weights_root, weights_precision);
        let components =
            burn_triposplat::import::load_triposplat_runtime_components_with_compute_dtype_and_callback::<
                CudaBackend,
                _,
            >(
                &device,
                &artifacts,
                Some(compute_dtype),
                |event| flush_cuda_upload_queue_after_triposplat_load_event(&device, event),
            )
            .expect("load TripoSplat runtime components for CUDA parity diagnostic");

        let rust_condition = if flow_only {
            eprintln!(
                "[triposplat_cuda_stage_parity] flow_only=1; skipping Rust DINO/Flux conditioning replay"
            );
            None
        } else {
            let image =
                read_f32_safetensor_4d::<CudaBackend>(&stage_tensors, "image_rgb_0_1", &device)
                    .expect("read upstream TripoSplat preprocessed image tensor");
            let expected_feature1 = read_f32_safetensor_vec(&stage_tensors, "feature1")
                .expect("read feature1 reference");
            let expected_feature2 = read_f32_safetensor_vec(&stage_tensors, "feature2")
                .expect("read feature2 reference");
            let condition = match read_f32_safetensor_4d::<CudaBackend>(
                &stage_tensors,
                "vae_noise",
                &device,
            ) {
                Ok(noise) => {
                    eprintln!(
                        "[triposplat_cuda_stage_parity] using upstream vae_noise for conditioning replay"
                    );
                    let flux_trace_image = image.clone();
                    let flux_trace_noise = noise.clone();
                    let diagnostics =
                        components.conditioning_diagnostics_with_vae_noise(image, noise);
                    if std::env::var("TRIPOSPLAT_FLUX_TRACE").is_ok() {
                        let flux_image = (flux_trace_image * 2.0 - 1.0).cast(compute_dtype);
                        let trace = components.flux2_vae_encoder.encode_with_noise_trace(
                            flux_image,
                            flux_trace_noise.cast(compute_dtype),
                        );
                        print_flux_trace_diffs(&stage_tensors, trace);
                    }
                    if let Ok(expected_dinov3_raw) =
                        read_f32_safetensor_vec(&stage_tensors, "dinov3_raw")
                    {
                        print_stage_diff(
                            "dinov3_raw",
                            diagnostics.dinov3_raw.clone(),
                            &expected_dinov3_raw.shape,
                            &expected_dinov3_raw.values,
                        );
                    }
                    if let Ok(expected_vae_mean) =
                        read_f32_safetensor_vec(&stage_tensors, "vae_mean")
                    {
                        print_stage_diff(
                            "vae_mean",
                            diagnostics.vae_mean.clone(),
                            &expected_vae_mean.shape,
                            &expected_vae_mean.values,
                        );
                    }
                    if let Ok(expected_vae_logvar) =
                        read_f32_safetensor_vec(&stage_tensors, "vae_logvar")
                    {
                        print_stage_diff(
                            "vae_logvar",
                            diagnostics.vae_logvar.clone(),
                            &expected_vae_logvar.shape,
                            &expected_vae_logvar.values,
                        );
                    }
                    diagnostics.into_condition()
                }
                Err(err) => {
                    eprintln!(
                        "[triposplat_cuda_stage_parity] upstream vae_noise unavailable ({err}); using seeded Rust VAE noise"
                    );
                    components.encode_preprocessed_image(image, DEFAULT_SEED)
                }
            };
            print_stage_diff(
                "feature1",
                condition.feature1.clone(),
                &expected_feature1.shape,
                &expected_feature1.values,
            );
            let actual_feature2 = condition
                .feature2
                .clone()
                .expect("TripoSplat condition should include Flux2 feature2");
            print_stage_diff(
                "feature2",
                actual_feature2,
                &expected_feature2.shape,
                &expected_feature2.values,
            );
            Some(condition)
        };
        if std::env::var("TRIPOSPLAT_CONDITION_ONLY").is_ok() {
            eprintln!("[triposplat_cuda_stage_parity] condition_only=1; skipping flow replay");
            return;
        }
        let expected_latent =
            read_f32_safetensor_vec(&stage_tensors, "latent").expect("read latent reference");
        let expected_camera =
            read_f32_safetensor_vec(&stage_tensors, "camera").expect("read camera reference");

        let reference_condition = burn_triposplat::TripoSplatCondition {
            feature1: read_f32_safetensor_3d::<CudaBackend>(&stage_tensors, "feature1", &device)
                .expect("read upstream feature1 tensor"),
            feature2: Some(
                read_f32_safetensor_3d::<CudaBackend>(&stage_tensors, "feature2", &device)
                    .expect("read upstream feature2 tensor"),
            ),
            rng_normals_consumed: 0,
        };
        let options = TripoSplatOptions {
            num_gaussians: 32_768,
            ..TripoSplatOptions::default()
        };
        let reference_flow_noise =
            read_f32_safetensor_3d::<CudaBackend>(&stage_tensors, "flow_noise_latent", &device)
                .ok()
                .map(|latent_noise| burn_triposplat::FlowState {
                    latent: latent_noise,
                    camera: read_f32_safetensor_3d::<CudaBackend>(
                        &stage_tensors,
                        "flow_noise_camera",
                        &device,
                    )
                    .ok(),
                });
        if let Some(noise) = reference_flow_noise.clone() {
            if let Ok(expected_pred_000) =
                read_f32_safetensor_vec(&stage_tensors, "flow_pred_000_latent")
            {
                let pred = components.flow_prediction_from_noise_at_step(
                    reference_condition.clone(),
                    noise.clone(),
                    options,
                    0,
                );
                print_stage_diff(
                    "flow_pred_000_latent",
                    pred.latent,
                    &expected_pred_000.shape,
                    &expected_pred_000.values,
                );
                if let (Some(camera), Ok(expected_camera)) = (
                    pred.camera,
                    read_f32_safetensor_vec(&stage_tensors, "flow_pred_000_camera"),
                ) {
                    print_stage_diff(
                        "flow_pred_000_camera",
                        camera,
                        &expected_camera.shape,
                        &expected_camera.values,
                    );
                }
            }
            if let Ok(expected_step_001) =
                read_f32_safetensor_vec(&stage_tensors, "flow_step_001_latent")
            {
                let step = components.sample_latent_prefix_from_noise(
                    reference_condition.clone(),
                    noise,
                    options,
                    1,
                );
                print_stage_diff(
                    "latent_after_flow_step_001",
                    step.latent,
                    &expected_step_001.shape,
                    &expected_step_001.values,
                );
            }
            if std::env::var("TRIPOSPLAT_PREFIX_ONLY").is_ok() {
                eprintln!(
                    "[triposplat_cuda_stage_parity] prefix_only=1; skipping full flow replay"
                );
                return;
            }
        }
        let sampled = match reference_flow_noise.clone() {
            Some(noise) => {
                eprintln!(
                    "[triposplat_cuda_stage_parity] using upstream flow_noise_latent for flow replay"
                );
                components.sample_latent_from_noise(reference_condition, noise, options)
            }
            None => {
                eprintln!(
                    "[triposplat_cuda_stage_parity] upstream flow noise unavailable; using seeded Rust flow noise"
                );
                components.sample_latent(reference_condition, options)
            }
        };
        let replay_latent_for_decode = sampled.latent.clone();
        print_stage_diff(
            "latent_from_reference_condition",
            sampled.latent,
            &expected_latent.shape,
            &expected_latent.values,
        );
        let camera = sampled
            .camera
            .expect("TripoSplat flow should emit camera channels");
        print_stage_diff(
            "camera_from_reference_condition",
            camera,
            &expected_camera.shape,
            &expected_camera.values,
        );
        decode_replayed_triposplat_latent(
            &components,
            "reference_condition",
            replay_latent_for_decode,
            options,
        );
        if std::env::var("TRIPOSPLAT_RUST_CONDITION_FLOW").is_ok()
            && let (Some(noise), Some(condition)) = (reference_flow_noise, rust_condition)
        {
            eprintln!(
                "[triposplat_cuda_stage_parity] using Rust conditioning with upstream flow noise"
            );
            let sampled = components.sample_latent_from_noise(condition, noise, options);
            let replay_latent_for_decode = sampled.latent.clone();
            print_stage_diff(
                "latent_from_rust_condition",
                sampled.latent,
                &expected_latent.shape,
                &expected_latent.values,
            );
            let camera = sampled
                .camera
                .expect("TripoSplat flow should emit camera channels");
            print_stage_diff(
                "camera_from_rust_condition",
                camera,
                &expected_camera.shape,
                &expected_camera.values,
            );
            decode_replayed_triposplat_latent(
                &components,
                "rust_condition",
                replay_latent_for_decode,
                options,
            );
        }
    }

    fn fake_triposplat_root(precisions: &[TripoSplatBurnpackPrecision]) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "burn_synth_triposplat_precision_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock should be after unix epoch")
                .as_nanos()
        ));
        for precision in precisions {
            for artifact in burn_triposplat::artifact::TRIPOSPLAT_ARTIFACTS
                .into_iter()
                .filter(|artifact| artifact.is_triposplat_runtime_required())
            {
                let path = artifact.burnpack_path(&root, *precision);
                fs::create_dir_all(path.parent().expect("artifact parent"))
                    .expect("create fake artifact parent");
                fs::write(path, []).expect("write fake artifact");
            }
        }
        root
    }

    #[cfg(feature = "cuda")]
    fn default_triposplat_stage_tensors_path() -> PathBuf {
        let path = std::env::var("TRIPOSPLAT_STAGE_TENSORS")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(
                    "tmp/runs/20260604T074500Z_triposplat_cuda_alpha_reference/stage_tensors_f32.safetensors",
                )
            });
        workspace_relative_path(path)
    }

    #[cfg(feature = "cuda")]
    fn flush_cuda_upload_queue_after_triposplat_load_event(
        device: &<CudaBackend as BackendTypes>::Device,
        event: burn_triposplat::import::TripoSplatRuntimeLoadEvent,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use cubecl::Runtime;

        let label = event.label();
        cubecl::cuda::CudaRuntime::client(device).flush().map_err(
            |err| -> Box<dyn std::error::Error> {
                format!("failed to flush CUDA upload queue after {label}: {err}").into()
            },
        )?;
        eprintln!("[triposplat_cuda_stage_parity] flushed CUDA upload queue after {label}");
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn decode_replayed_triposplat_latent(
        components: &TripoSplatRuntimeComponents<CudaBackend>,
        label: &str,
        latent: Tensor<CudaBackend, 3>,
        options: TripoSplatOptions,
    ) {
        if std::env::var("TRIPOSPLAT_DECODE_REPLAY").is_err() {
            return;
        }
        let cloud = components
            .decoder
            .decode_to_cloud_with_seed_checked(latent, options.num_gaussians, DEFAULT_SEED)
            .unwrap_or_else(|err| {
                panic!("[triposplat_cuda_stage_parity] {label} decode failed: {err}")
            });
        eprintln!(
            "[triposplat_cuda_stage_parity] {label}_decoded_splats={}",
            cloud.len()
        );
        assert_eq!(cloud.len(), options.num_gaussians);
        if let Ok(output_dir) = std::env::var("TRIPOSPLAT_REPLAY_SPLAT_DIR") {
            let output_dir = workspace_relative_path(PathBuf::from(output_dir));
            fs::create_dir_all(&output_dir).expect("create TripoSplat replay splat directory");
            let path = output_dir.join(format!("{}_{}.splat", label, options.num_gaussians));
            cloud
                .write_splat(&path)
                .expect("write TripoSplat replay splat output");
            eprintln!(
                "[triposplat_cuda_stage_parity] {label}_splat_path={}",
                path.display()
            );
        }
    }

    #[cfg(feature = "cuda")]
    struct F32Safetensor {
        shape: Vec<usize>,
        values: Vec<f32>,
    }

    #[cfg(feature = "cuda")]
    fn read_f32_safetensor_vec(
        path: &Path,
        name: &str,
    ) -> Result<F32Safetensor, Box<dyn std::error::Error>> {
        let bytes = fs::read(path)?;
        let tensors = safetensors::SafeTensors::deserialize(&bytes)?;
        let view = tensors.tensor(name)?;
        if view.dtype() != safetensors::tensor::Dtype::F32 {
            return Err(format!("{name} must be F32, got {:?}", view.dtype()).into());
        }
        let chunks = view.data().chunks_exact(4);
        if !chunks.remainder().is_empty() {
            return Err(format!("{name} F32 byte length is not divisible by 4").into());
        }
        let values = chunks
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<_>>();
        Ok(F32Safetensor {
            shape: view.shape().to_vec(),
            values,
        })
    }

    #[cfg(feature = "cuda")]
    fn read_f32_safetensor_3d<B: Backend>(
        path: &Path,
        name: &str,
        device: &B::Device,
    ) -> Result<Tensor<B, 3>, Box<dyn std::error::Error>> {
        let tensor = read_f32_safetensor_vec(path, name)?;
        let shape = tensor.shape.as_slice();
        if shape.len() != 3 {
            return Err(format!("{name} must be rank 3, got shape {shape:?}").into());
        }
        Ok(
            Tensor::<B, 1>::from_floats(tensor.values.as_slice(), device)
                .reshape([shape[0], shape[1], shape[2]]),
        )
    }

    #[cfg(feature = "cuda")]
    fn read_f32_safetensor_4d<B: Backend>(
        path: &Path,
        name: &str,
        device: &B::Device,
    ) -> Result<Tensor<B, 4>, Box<dyn std::error::Error>> {
        let tensor = read_f32_safetensor_vec(path, name)?;
        let shape = tensor.shape.as_slice();
        if shape.len() != 4 {
            return Err(format!("{name} must be rank 4, got shape {shape:?}").into());
        }
        Ok(
            Tensor::<B, 1>::from_floats(tensor.values.as_slice(), device)
                .reshape([shape[0], shape[1], shape[2], shape[3]]),
        )
    }

    #[cfg(feature = "cuda")]
    fn print_stage_diff<B: Backend, const D: usize>(
        label: &str,
        actual: Tensor<B, D>,
        expected_shape: &[usize],
        expected: &[f32],
    ) {
        let actual_shape = actual.dims().to_vec();
        assert_eq!(
            actual_shape, expected_shape,
            "{label} shape mismatch: actual={actual_shape:?} expected={expected_shape:?}"
        );
        let actual = actual
            .try_into_data()
            .unwrap_or_else(|err| {
                panic!(
                    "{label} failed to execute/read CUDA tensor: {}",
                    summarize_cuda_execution_error(&format!("{err:?}"))
                )
            })
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("stage tensor values");
        let summary = diff_summary(&actual, expected);
        eprintln!(
            "[triposplat_cuda_stage_parity] {label} elements={} max_abs={:.6e} mean_abs={:.6e} rms={:.6e} actual_min={:.6e} actual_max={:.6e} expected_min={:.6e} expected_max={:.6e}",
            actual.len(),
            summary.max_abs,
            summary.mean_abs,
            summary.rms,
            summary.actual_min,
            summary.actual_max,
            summary.expected_min,
            summary.expected_max
        );
        assert!(
            summary.all_finite,
            "{label} contains non-finite values in actual or expected tensor"
        );
        if let Some(thresholds) = StageDiffThresholds::from_env() {
            thresholds.assert_within(label, &summary);
        }
    }

    #[cfg(feature = "cuda")]
    fn print_flux_trace_diffs(path: &Path, trace: burn_flux::Flux2VaeEncodeTrace<CudaBackend>) {
        print_optional_stage_diff(path, "flux2_conv_in", trace.encoder.conv_in);
        print_optional_stage_diff(path, "flux2_down_0_resnet_0", trace.encoder.down_0_resnet_0);
        print_optional_stage_diff(path, "flux2_down_0_resnet_1", trace.encoder.down_0_resnet_1);
        print_optional_stage_diff(path, "flux2_down_0_sampler", trace.encoder.down_0_sampler);
        print_optional_stage_diff(path, "flux2_down_1_resnet_0", trace.encoder.down_1_resnet_0);
        print_optional_stage_diff(path, "flux2_down_1_resnet_1", trace.encoder.down_1_resnet_1);
        print_optional_stage_diff(path, "flux2_down_1_sampler", trace.encoder.down_1_sampler);
        print_optional_stage_diff(path, "flux2_down_2_resnet_0", trace.encoder.down_2_resnet_0);
        print_optional_stage_diff(path, "flux2_down_2_resnet_1", trace.encoder.down_2_resnet_1);
        print_optional_stage_diff(path, "flux2_down_2_sampler", trace.encoder.down_2_sampler);
        print_optional_stage_diff(path, "flux2_down_3_resnet_0", trace.encoder.down_3_resnet_0);
        print_optional_stage_diff(path, "flux2_down_3_resnet_1", trace.encoder.down_3_resnet_1);
        print_optional_stage_diff(path, "flux2_mid_resnet_0", trace.encoder.mid_resnet_0);
        print_optional_stage_diff(path, "flux2_mid_attn", trace.encoder.mid_attn);
        print_optional_stage_diff(path, "flux2_mid_resnet_1", trace.encoder.mid_resnet_1);
        print_optional_stage_diff(path, "flux2_encoder_out", trace.encoder.encoder_out);
        print_optional_stage_diff(path, "flux2_moments", trace.moments);
        print_optional_stage_diff(path, "flux2_latents", trace.latents);
        print_optional_stage_diff(path, "flux2_unshuffled", trace.unshuffled);
        print_optional_stage_diff(path, "flux2_normalized", trace.normalized);
        print_optional_stage_diff(path, "flux2_tokens", trace.tokens);
    }

    #[cfg(feature = "cuda")]
    fn print_optional_stage_diff<const D: usize>(
        path: &Path,
        label: &str,
        actual: Tensor<CudaBackend, D>,
    ) {
        match read_f32_safetensor_vec(path, label) {
            Ok(expected) => {
                print_stage_diff(label, actual, &expected.shape, &expected.values);
            }
            Err(err) => {
                eprintln!("[triposplat_cuda_stage_parity] missing optional {label}: {err}");
            }
        }
    }

    #[cfg(feature = "cuda")]
    struct DiffSummary {
        max_abs: f64,
        mean_abs: f64,
        rms: f64,
        actual_min: f64,
        actual_max: f64,
        expected_min: f64,
        expected_max: f64,
        all_finite: bool,
    }

    #[cfg(feature = "cuda")]
    struct StageDiffThresholds {
        max_abs: f64,
        mean_abs: f64,
        rms: f64,
    }

    #[cfg(feature = "cuda")]
    impl StageDiffThresholds {
        fn from_env() -> Option<Self> {
            if std::env::var("TRIPOSPLAT_CUDA_STAGE_ASSERT").is_err() {
                return None;
            }
            Some(Self {
                max_abs: parse_stage_threshold_env("TRIPOSPLAT_CUDA_STAGE_MAX_ABS", 1.0e-2),
                mean_abs: parse_stage_threshold_env("TRIPOSPLAT_CUDA_STAGE_MEAN_ABS", 1.0e-3),
                rms: parse_stage_threshold_env("TRIPOSPLAT_CUDA_STAGE_RMS", 2.0e-3),
            })
        }

        fn assert_within(&self, label: &str, summary: &DiffSummary) {
            assert!(
                summary.max_abs <= self.max_abs,
                "{label} max_abs {:.6e} exceeds threshold {:.6e}",
                summary.max_abs,
                self.max_abs
            );
            assert!(
                summary.mean_abs <= self.mean_abs,
                "{label} mean_abs {:.6e} exceeds threshold {:.6e}",
                summary.mean_abs,
                self.mean_abs
            );
            assert!(
                summary.rms <= self.rms,
                "{label} rms {:.6e} exceeds threshold {:.6e}",
                summary.rms,
                self.rms
            );
        }
    }

    #[cfg(feature = "cuda")]
    fn parse_stage_threshold_env(name: &str, default: f64) -> f64 {
        std::env::var(name)
            .ok()
            .map(|value| {
                value
                    .parse::<f64>()
                    .unwrap_or_else(|err| panic!("{name} must be a float, got {value}: {err}"))
            })
            .unwrap_or(default)
    }

    #[cfg(feature = "cuda")]
    fn diff_summary(actual: &[f32], expected: &[f32]) -> DiffSummary {
        assert_eq!(
            actual.len(),
            expected.len(),
            "stage tensor length mismatch: actual={} expected={}",
            actual.len(),
            expected.len()
        );
        let mut max_abs = 0.0f64;
        let mut sum_abs = 0.0f64;
        let mut sum_sq = 0.0f64;
        let mut actual_min = f64::INFINITY;
        let mut actual_max = f64::NEG_INFINITY;
        let mut expected_min = f64::INFINITY;
        let mut expected_max = f64::NEG_INFINITY;
        let mut all_finite = true;
        for (&actual, &expected) in actual.iter().zip(expected.iter()) {
            let actual = actual as f64;
            let expected = expected as f64;
            all_finite &= actual.is_finite() && expected.is_finite();
            actual_min = actual_min.min(actual);
            actual_max = actual_max.max(actual);
            expected_min = expected_min.min(expected);
            expected_max = expected_max.max(expected);
            let abs = (actual - expected).abs();
            max_abs = max_abs.max(abs);
            sum_abs += abs;
            sum_sq += abs * abs;
        }
        let len = actual.len().max(1) as f64;
        DiffSummary {
            max_abs,
            mean_abs: sum_abs / len,
            rms: (sum_sq / len).sqrt(),
            actual_min,
            actual_max,
            expected_min,
            expected_max,
            all_finite,
        }
    }

    #[cfg(feature = "cuda")]
    fn workspace_relative_path(path: PathBuf) -> PathBuf {
        if path.is_absolute() {
            return path;
        }
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(path)
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

    #[cfg(feature = "trellis")]
    #[test]
    fn trellis_runtime_source_validation_rejects_synthetic_sparse() {
        let err = validate_trellis_runtime_sources(TrellisDevice::Wgpu, "synthetic", "runtime")
            .expect_err("synthetic sparse source must fail fast");
        assert!(
            err.to_string().contains("synthetic sparse fallback"),
            "unexpected error: {err}"
        );
    }

    #[cfg(feature = "trellis")]
    #[test]
    fn trellis_runtime_source_validation_rejects_decode_fallback() {
        let err = validate_trellis_runtime_sources(
            TrellisDevice::Wgpu,
            "runtime_model_wgpu",
            "fallback_runtime_error",
        )
        .expect_err("decode fallback source must fail fast");
        assert!(
            err.to_string().contains("decode fallback path"),
            "unexpected error: {err}"
        );
    }

    #[cfg(feature = "trellis")]
    #[test]
    fn trellis_runtime_source_validation_accepts_runtime_wgpu() {
        validate_trellis_runtime_sources(TrellisDevice::Wgpu, "runtime_model_wgpu", "runtime")
            .expect("runtime wgpu sources should be accepted");
    }

    #[cfg(feature = "trellis")]
    #[test]
    fn trellis_pbr_off_uses_native_mesh_postprocess_not_hook_export() {
        assert_eq!(
            trellis_decode_output_mode(false),
            TrellisDecodeOutputMode::NativeMesh
        );
        assert_eq!(
            trellis_decode_output_mode(true),
            TrellisDecodeOutputMode::NativePbr
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

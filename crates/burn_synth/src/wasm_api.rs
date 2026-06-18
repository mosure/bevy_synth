#![cfg(all(target_arch = "wasm32", feature = "wasm-api"))]

use std::cell::RefCell;
use std::sync::Once;
#[cfg(feature = "wasm-api-wgpu")]
use std::sync::atomic::{AtomicBool, Ordering};

use burn::backend::NdArray;
use burn::prelude::*;
use burn::tensor::{FloatDType, backend::BackendTypes};
use burn_foreground::pipeline::{
    PrepareImageConfig, PreparedImageData, RmbgPipeline, prepare_image_data_from_bytes_async,
};
use burn_foreground::rmbg14::BriaRmbg;
use burn_foreground::rmbg14::import::{
    apply_rmbg_burnpack_part_bytes, load_rmbg_config_from_json_bytes,
};
use burn_foreground::rmbg14::set_rmbg_strict_interp_override;
#[cfg(feature = "trellis")]
use burn_trellis::pipeline::{
    Trellis2Pipeline, Trellis2PipelineConfig, TrellisDevice, TrellisRunOptions,
};
#[cfg(feature = "trellis")]
use burn_trellis::trellis_config::TrellisPipelineConfig;
#[cfg(feature = "trellis")]
use burn_trellis::virtual_fs as trellis_virtual_fs;
use burn_tripo::model::triposg::dit::import::apply_triposg_dit_burnpack_part_bytes;
use burn_tripo::model::triposg::dit::{TripoSGDiT, TripoSGDiTConfig};
use burn_tripo::model::triposg::image_encoder::import::{
    apply_triposg_dinov2_burnpack_part_bytes, default_dinov2_config, init_triposg_dinov2_model,
};
use burn_tripo::model::triposg::image_encoder::{DinoImageProcessor, TripoSGImageEncoder};
use burn_tripo::model::triposg::scheduler::RectifiedFlowSchedulerConfig;
use burn_tripo::model::triposg::vae::TripoSGVae;
use burn_tripo::model::triposg::vae::TripoSGVaeConfig;
use burn_tripo::model::triposg::vae::import::apply_triposg_vae_decoder_burnpack_part_bytes;
use burn_tripo::pipeline::geometry::FlashExtractConfig;
use burn_tripo::pipeline::mesh::Mesh as TripoMesh;
use burn_tripo::pipeline::runtime_parity::{
    decimate_tripo_mesh, should_prefer_f16_triposg_weights, triposg_runtime_profile,
};
use burn_tripo::pipeline::triposg::{TripoSGPipeline, deterministic_latents_from_seed};
use burn_triposplat::artifact::{TRIPOSPLAT_ARTIFACTS, TripoSplatArtifact};
use burn_triposplat::{
    ElasticGaussianFixedlenDecoderConfig, GaussianSplatCloud, LatentSeqMmFlowModelConfig,
    OCTREE_MAX_VOXEL_LEVEL, OctreeGaussianDecoder, OctreeProbabilityFixedlenDecoderConfig,
    TripoSplatOptions, TripoSplatRuntimeComponents, normalize_num_gaussians,
};
#[cfg(feature = "wasm-api-wgpu")]
use js_sys::{Function, Promise};
use js_sys::{Reflect, Uint8Array};
use sha2::{Digest, Sha256};
#[cfg(feature = "wasm-api-wgpu")]
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
#[cfg(feature = "wasm-api-wgpu")]
use wasm_bindgen_futures::JsFuture;

use crate::mesh::Mesh;
use crate::mesh_to_glb_bytes;
use crate::model_loader::{
    candidate_burnpack_names, parse_parts_manifest_bytes, resolve_manifest_entry_uri,
};
use crate::triposplat_preprocess::triposplat_prepare_image_config;
use crate::wasm::{DEFAULT_WASM_FLASH_NUM_CHUNKS, WasmInferencePreset};
use crate::wasm_loader::{
    DownloadTotals, WasmHostMemoryBudget, download_binary_with_status, fetch_optional_text,
    fetch_optional_text_candidates, join_web_path, web_max_burnpack_bytes, web_max_host_ram_bytes,
};

#[cfg(feature = "wasm-api-wgpu")]
type WgpuBackendF16 = burn_wgpu::Wgpu<burn::tensor::f16, i32, u32>;
#[cfg(feature = "wasm-api-wgpu")]
type WgpuBackendF32 = burn_wgpu::Wgpu<f32, i32, u32>;
#[cfg(feature = "wasm-api-wgpu")]
type WgpuTripoSplatBackendF16 = burn_wgpu::Wgpu<burn::tensor::f16, i32, u32>;
#[cfg(feature = "wasm-api-wgpu")]
type WgpuTripoSplatBackendF32 = burn_wgpu::Wgpu<f32, i32, u32>;
#[cfg(feature = "wasm-api-wgpu")]
type WgpuRmbgBackend = burn_wgpu::Wgpu<f32, i32, u32>;

const DEFAULT_GUIDANCE_SCALE: f32 = 7.0;
const DEFAULT_BOUNDS: [f32; 6] = [-1.005, -1.005, -1.005, 1.005, 1.005, 1.005];
const DEFAULT_MODEL_BASE_URL: &str = "https://aberration.technology/model";
const DEFAULT_LOCAL_MODEL_BASE_URL: &str = "assets/models";
const DINO_CONFIG_RELPATHS: [&str; 2] = [
    "image_encoder_dinov2/config.json",
    "image_encoder_2/config.json",
];
const ROOT_TRIPOSG: &str = "MIDI-3D";
const ROOT_TRIPOSPLAT: &str = "TripoSplat";
const ROOT_RMBG14: &str = "RMBG-1.4";
#[cfg(feature = "trellis")]
const ROOT_TRELLIS: &str = "TRELLIS.2-4B";
#[cfg(feature = "trellis")]
const ROOT_TRELLIS_IMAGE_LARGE: &str = "TRELLIS-image-large";
const CANONICAL_DINO_SHORT_EDGE: usize = 256;
const CANONICAL_DINO_CROP: usize = 224;

static PANIC_HOOK_ONCE: Once = Once::new();

struct WasmPipelineState<BTriposg: Backend, BRmbg: Backend> {
    triposg_device: BTriposg::Device,
    rmbg: Option<RmbgPipeline<BRmbg>>,
    triposg: TripoSGPipeline<BTriposg>,
}

struct WasmTripoSplatPipelineState<BTripoSplat: Backend, BRmbg: Backend> {
    triposplat_device: BTripoSplat::Device,
    rmbg: Option<RmbgPipeline<BRmbg>>,
    components: TripoSplatRuntimeComponents<BTripoSplat>,
}

#[cfg(feature = "trellis")]
struct WasmTrellisPipelineState {
    trellis: Trellis2Pipeline,
}

#[cfg(feature = "wasm-api-wgpu")]
enum CachedWasmPipeline {
    WgpuF32 {
        preset: WasmInferencePreset,
        state: WasmPipelineState<WgpuBackendF32, WgpuRmbgBackend>,
    },
    WgpuF16 {
        preset: WasmInferencePreset,
        state: WasmPipelineState<WgpuBackendF16, WgpuRmbgBackend>,
    },
}

#[cfg(feature = "wasm-api-wgpu")]
enum CachedWasmTripoSplatPipeline {
    WgpuF32 {
        preset: WasmInferencePreset,
        state: WasmTripoSplatPipelineState<WgpuTripoSplatBackendF32, WgpuRmbgBackend>,
    },
    WgpuF16 {
        preset: WasmInferencePreset,
        state: WasmTripoSplatPipelineState<WgpuTripoSplatBackendF16, WgpuRmbgBackend>,
    },
}

#[cfg(feature = "wasm-api-wgpu")]
thread_local! {
    static CACHED_WASM_PIPELINE: RefCell<Option<CachedWasmPipeline>> = const { RefCell::new(None) };
}

#[cfg(feature = "wasm-api-wgpu")]
thread_local! {
    static CACHED_WASM_TRIPOSPLAT_PIPELINE: RefCell<Option<CachedWasmTripoSplatPipeline>> = const { RefCell::new(None) };
}

#[cfg(all(feature = "wasm-api-wgpu", feature = "trellis"))]
thread_local! {
    static CACHED_WASM_TRELLIS_PIPELINE: RefCell<Option<(WasmInferencePreset, WasmTrellisPipelineState)>> = const { RefCell::new(None) };
}

#[derive(Clone, Copy, Debug)]
struct TripoWasmLoadOptions {
    strict_dino_preprocess: bool,
    strict_precision: bool,
    prefer_f16_vae: bool,
    prefer_f16_dit: bool,
    prefer_f16_dino: bool,
}

struct WasmLoadContext<'a, F: FnMut(String)> {
    totals: &'a mut DownloadTotals,
    host_ram_budget: &'a mut WasmHostMemoryBudget,
    on_status: &'a mut F,
}

impl<F: FnMut(String)> WasmLoadContext<'_, F> {
    fn status(&mut self, message: String) {
        (self.on_status)(message);
    }
}

#[cfg(feature = "wasm-api-wgpu")]
#[derive(Clone, Debug, Default)]
struct WebGpuAdapterProfile {
    shader_f16_supported: bool,
}

#[wasm_bindgen]
#[derive(Clone, Debug, Default)]
pub struct WasmInferOptions {
    synthesis_model: Option<String>,
    rmbg_model: Option<String>,
    quality: Option<String>,
    num_steps: u32,
    num_tokens: u32,
    guidance_scale: Option<f32>,
    triposplat_shift: Option<f32>,
    triposplat_num_gaussians: Option<u32>,
    triposplat_erode_radius: Option<u32>,
    resolution: u32,
    faces: Option<u32>,
    seed: Option<u64>,
    backend: Option<String>,
    dino_backend: Option<String>,
    weights_precision: Option<String>,
    rmbg_weights_precision: Option<String>,
}

#[wasm_bindgen]
impl WasmInferOptions {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_num_steps(&mut self, value: u32) {
        self.num_steps = value;
    }

    pub fn set_synthesis_model(&mut self, value: String) {
        self.synthesis_model = Some(value);
    }

    pub fn clear_synthesis_model(&mut self) {
        self.synthesis_model = None;
    }

    pub fn set_rmbg_model(&mut self, value: String) {
        self.rmbg_model = Some(value);
    }

    pub fn clear_rmbg_model(&mut self) {
        self.rmbg_model = None;
    }

    pub fn set_quality(&mut self, value: String) {
        self.quality = Some(value);
    }

    pub fn clear_quality(&mut self) {
        self.quality = None;
    }

    pub fn set_num_tokens(&mut self, value: u32) {
        self.num_tokens = value;
    }

    pub fn set_guidance_scale(&mut self, value: f32) {
        self.guidance_scale = Some(value);
    }

    pub fn clear_guidance_scale(&mut self) {
        self.guidance_scale = None;
    }

    pub fn set_triposplat_shift(&mut self, value: f32) {
        self.triposplat_shift = Some(value);
    }

    pub fn clear_triposplat_shift(&mut self) {
        self.triposplat_shift = None;
    }

    pub fn set_triposplat_num_gaussians(&mut self, value: u32) {
        self.triposplat_num_gaussians = Some(value);
    }

    pub fn clear_triposplat_num_gaussians(&mut self) {
        self.triposplat_num_gaussians = None;
    }

    pub fn set_triposplat_erode_radius(&mut self, value: u32) {
        self.triposplat_erode_radius = Some(value);
    }

    pub fn clear_triposplat_erode_radius(&mut self) {
        self.triposplat_erode_radius = None;
    }

    pub fn set_resolution(&mut self, value: u32) {
        self.resolution = value;
    }

    pub fn set_faces(&mut self, value: u32) {
        self.faces = Some(value);
    }

    pub fn clear_faces(&mut self) {
        self.faces = None;
    }

    pub fn set_seed(&mut self, value: u64) {
        self.seed = Some(value);
    }

    pub fn clear_seed(&mut self) {
        self.seed = None;
    }

    pub fn set_backend(&mut self, value: String) {
        self.backend = Some(value);
    }

    pub fn clear_backend(&mut self) {
        self.backend = None;
    }

    pub fn set_dino_backend(&mut self, value: String) {
        self.dino_backend = Some(value);
    }

    pub fn clear_dino_backend(&mut self) {
        self.dino_backend = None;
    }

    pub fn set_weights_precision(&mut self, value: String) {
        self.weights_precision = Some(value);
    }

    pub fn clear_weights_precision(&mut self) {
        self.weights_precision = None;
    }

    pub fn set_rmbg_weights_precision(&mut self, value: String) {
        self.rmbg_weights_precision = Some(value);
    }

    pub fn clear_rmbg_weights_precision(&mut self) {
        self.rmbg_weights_precision = None;
    }
}

impl WasmInferOptions {
    pub fn from_preset(preset: &WasmInferencePreset) -> Self {
        Self {
            synthesis_model: Some(preset.synthesis_model.to_string()),
            rmbg_model: Some(preset.rmbg_model.to_string()),
            quality: Some(preset.quality.to_string()),
            num_steps: preset.num_steps as u32,
            num_tokens: preset.num_tokens as u32,
            guidance_scale: Some(preset.guidance_scale),
            triposplat_shift: Some(preset.triposplat_shift),
            triposplat_num_gaussians: Some(preset.triposplat_num_gaussians as u32),
            triposplat_erode_radius: Some(preset.triposplat_erode_radius as u32),
            resolution: preset.resolution as u32,
            faces: Some(preset.faces as u32),
            seed: Some(preset.seed),
            backend: Some(preset.backend.to_string()),
            dino_backend: Some(preset.dino_backend.to_string()),
            weights_precision: Some(preset.weights_precision.to_string()),
            rmbg_weights_precision: Some(preset.rmbg_weights_precision.to_string()),
        }
    }

    fn apply_to_preset(&self, preset: &mut WasmInferencePreset) {
        if let Some(value) = self.synthesis_model.as_ref() {
            preset.synthesis_model = if value.eq_ignore_ascii_case("trellis")
                || value.eq_ignore_ascii_case("trellis2")
                || value.eq_ignore_ascii_case("trellis.2")
            {
                "trellis"
            } else if value.eq_ignore_ascii_case("triposplat")
                || value.eq_ignore_ascii_case("tripo-splat")
                || value.eq_ignore_ascii_case("splat")
            {
                "triposplat"
            } else {
                "triposg"
            };
        }
        if let Some(value) = self.rmbg_model.as_ref() {
            preset.rmbg_model = if value.eq_ignore_ascii_case("rmbg2")
                || value.eq_ignore_ascii_case("rmbg-2")
                || value.eq_ignore_ascii_case("rmbg-2.0")
            {
                "rmbg2"
            } else if value.eq_ignore_ascii_case("none")
                || value.eq_ignore_ascii_case("off")
                || value.eq_ignore_ascii_case("disabled")
                || value.eq_ignore_ascii_case("passthrough")
            {
                "none"
            } else {
                "rmbg14"
            };
        }
        if let Some(value) = self.quality.as_ref() {
            preset.quality =
                if value.eq_ignore_ascii_case("fast") || value.eq_ignore_ascii_case("low") {
                    "fast"
                } else if value.eq_ignore_ascii_case("full") || value.eq_ignore_ascii_case("high") {
                    "full"
                } else {
                    "balanced"
                };
        }
        if self.num_steps > 0 {
            preset.num_steps = self.num_steps as usize;
        }
        if self.num_tokens > 0 {
            preset.num_tokens = self.num_tokens as usize;
        }
        if let Some(value) = self.guidance_scale
            && value.is_finite()
            && value > 0.0
        {
            preset.guidance_scale = value;
        }
        if let Some(value) = self.triposplat_shift
            && value.is_finite()
            && value > 0.0
        {
            preset.triposplat_shift = value;
        }
        if let Some(value) = self.triposplat_num_gaussians {
            preset.triposplat_num_gaussians = value as usize;
        }
        if let Some(value) = self.triposplat_erode_radius {
            preset.triposplat_erode_radius = value as usize;
        }
        if self.resolution > 0 {
            preset.resolution = self.resolution as usize;
        }
        if let Some(value) = self.faces {
            preset.faces = value as usize;
        }
        if let Some(value) = self.seed {
            preset.seed = value;
        }
        if let Some(value) = self.backend.as_ref() {
            preset.backend = if value.eq_ignore_ascii_case("cpu") {
                "cpu"
            } else {
                "wgpu"
            };
        }
        if let Some(value) = self.dino_backend.as_ref() {
            preset.dino_backend = if value.eq_ignore_ascii_case("cpu") {
                "cpu"
            } else if value.eq_ignore_ascii_case("gpu") {
                "gpu"
            } else {
                "auto"
            };
        }
        if let Some(value) = self.weights_precision.as_ref() {
            preset.weights_precision = if value.eq_ignore_ascii_case("f16") {
                "f16"
            } else if value.eq_ignore_ascii_case("auto") {
                "auto"
            } else {
                "f32"
            };
        }
        if let Some(value) = self.rmbg_weights_precision.as_ref() {
            preset.rmbg_weights_precision = if value.eq_ignore_ascii_case("f16") {
                "f16"
            } else if value.eq_ignore_ascii_case("f32") {
                "f32"
            } else {
                "auto"
            };
        }
    }
}

#[cfg(feature = "wasm-api-wgpu")]
fn resolve_wgpu_precision_for_preset(
    preset: &WasmInferencePreset,
    shader_f16_supported: bool,
) -> &'static str {
    if preset.weights_precision.eq_ignore_ascii_case("f16")
        || preset.weights_precision.eq_ignore_ascii_case("auto")
    {
        return if shader_f16_supported { "f16" } else { "f32" };
    }
    "f32"
}

#[cfg(feature = "wasm-api-wgpu")]
fn resolve_triposplat_wgpu_precision_for_preset(
    preset: &WasmInferencePreset,
    shader_f16_supported: bool,
) -> Result<&'static str, String> {
    if !shader_f16_supported {
        return Err(
            "TripoSplat wasm currently requires a WebGPU adapter with shader-f16; the f32 browser path exceeds WebGPU memory limits during octree decode. Use a shader-f16-capable browser/GPU for TripoSplat wasm."
                .to_string(),
        );
    }
    if preset.weights_precision.eq_ignore_ascii_case("f32") {
        return Err(
            "TripoSplat wasm f32 precision is disabled because it exceeds WebGPU memory limits during octree decode; use weights_precision=auto or f16 on a shader-f16-capable adapter."
                .to_string(),
        );
    }
    Ok("f16")
}

fn validate_wasm_preset_supported(preset: &WasmInferencePreset) -> Result<(), String> {
    let synthesis_model = preset.synthesis_model.trim().to_ascii_lowercase();
    let triposg = matches!(synthesis_model.as_str(), "triposg" | "tripo");
    let trellis = matches!(
        synthesis_model.as_str(),
        "trellis" | "trellis2" | "trellis.2"
    );
    let triposplat = matches!(
        synthesis_model.as_str(),
        "triposplat" | "tripo-splat" | "splat"
    );

    if !triposg && !trellis && !triposplat {
        return Err(format!(
            "unsupported wasm synthesis model '{}'; expected triposg or triposplat",
            preset.synthesis_model
        ));
    }

    if trellis {
        return Err(
            "Trellis wasm loading is disabled until the chunked model loader is fully async; use TripoSG wasm or native Trellis."
                .to_string(),
        );
    }

    if !preset.rmbg_model.eq_ignore_ascii_case("rmbg14") && !wasm_preset_skips_rmbg(preset) {
        return Err(format!(
            "wasm build currently supports rmbg_model=rmbg14 or none (received '{}')",
            preset.rmbg_model
        ));
    }
    Ok(())
}

fn wasm_preset_skips_rmbg(preset: &WasmInferencePreset) -> bool {
    matches!(
        preset.rmbg_model.trim().to_ascii_lowercase().as_str(),
        "none" | "off" | "disabled" | "passthrough"
    )
}

fn wasm_preset_is_triposplat(preset: &WasmInferencePreset) -> bool {
    matches!(
        preset.synthesis_model.trim().to_ascii_lowercase().as_str(),
        "triposplat" | "tripo-splat" | "splat"
    )
}

fn wasm_preset_is_trellis(preset: &WasmInferencePreset) -> bool {
    matches!(
        preset.synthesis_model.trim().to_ascii_lowercase().as_str(),
        "trellis" | "trellis2" | "trellis.2"
    )
}

#[cfg(feature = "wasm-api-wgpu")]
pub async fn warmup_pipeline_for_preset(preset: &WasmInferencePreset) -> Result<(), String> {
    warmup_pipeline_for_preset_with_status(preset, |message| {
        web_sys::console::log_1(&message.into());
    })
    .await
}

#[cfg(feature = "wasm-api-wgpu")]
pub async fn warmup_pipeline_for_preset_with_status<F>(
    preset: &WasmInferencePreset,
    on_status: F,
) -> Result<(), String>
where
    F: FnMut(String),
{
    validate_wasm_preset_supported(preset)?;
    if !preset.backend.eq_ignore_ascii_case("wgpu") {
        return Err("wasm synthesis supports backend=wgpu only".to_string());
    }
    if wasm_preset_is_triposplat(preset) {
        return Box::pin(warmup_triposplat_pipeline_for_preset_with_status(
            preset, on_status,
        ))
        .await;
    }
    if wasm_preset_is_trellis(preset) {
        #[cfg(feature = "trellis")]
        {
            return Box::pin(warmup_trellis_pipeline_for_preset_with_status(
                preset, on_status,
            ))
            .await;
        }
        #[cfg(not(feature = "trellis"))]
        {
            return Err(
                "this wasm build does not include trellis support (`trellis-wgpu` feature missing)."
                    .to_string(),
            );
        }
    }
    Box::pin(warmup_triposg_pipeline_for_preset_with_status(
        preset, on_status,
    ))
    .await
}

#[cfg(feature = "wasm-api-wgpu")]
async fn warmup_triposg_pipeline_for_preset_with_status<F>(
    preset: &WasmInferencePreset,
    mut on_status: F,
) -> Result<(), String>
where
    F: FnMut(String),
{
    let Some(adapter_profile) = wasm_webgpu_adapter_profile().await else {
        return Err(
            "WebGPU is unavailable in this browser/runtime; CPU fallback is disabled for TripoSG wasm."
                .to_string(),
        );
    };
    initialize_wgpu_runtime_for_wasm().await?;
    if preset.weights_precision.eq_ignore_ascii_case("f16") && !adapter_profile.shader_f16_supported
    {
        return Err(
            "weights_precision=f16 requested, but this WebGPU adapter lacks shader-f16; use weights_precision=auto or f32."
                .to_string(),
        );
    }
    let precision = resolve_wgpu_precision_for_preset(preset, adapter_profile.shader_f16_supported);
    if preset.weights_precision.eq_ignore_ascii_case("f16") && !adapter_profile.shader_f16_supported
    {
        on_status(
            "WebGPU adapter lacks shader-f16; running TripoSG on f32 backend while preferring f16 model weights."
                .to_string(),
        );
    }

    let cache_hit = CACHED_WASM_PIPELINE.with(|cache| {
        let guard = cache.borrow();
        match (&*guard, precision) {
            (Some(CachedWasmPipeline::WgpuF32 { preset: cached, .. }), "f32") => cached == preset,
            (Some(CachedWasmPipeline::WgpuF16 { preset: cached, .. }), "f16") => cached == preset,
            _ => false,
        }
    });
    if cache_hit {
        on_status("Model weights already loaded (cache hit).".to_string());
        return Ok(());
    }

    let loaded = match precision {
        "f16" => CachedWasmPipeline::WgpuF16 {
            preset: preset.clone(),
            state: load_pipeline_state::<WgpuBackendF16, WgpuRmbgBackend, _>(
                preset,
                &mut on_status,
            )
            .await?,
        },
        _ => CachedWasmPipeline::WgpuF32 {
            preset: preset.clone(),
            state: load_pipeline_state::<WgpuBackendF32, WgpuRmbgBackend, _>(
                preset,
                &mut on_status,
            )
            .await?,
        },
    };

    CACHED_WASM_PIPELINE.with(|cache| {
        *cache.borrow_mut() = Some(loaded);
    });
    Ok(())
}

#[cfg(not(feature = "wasm-api-wgpu"))]
pub async fn warmup_pipeline_for_preset(_preset: &WasmInferencePreset) -> Result<(), String> {
    Err(
        "this build does not include wasm WebGPU support (`wasm-api-wgpu` feature missing)."
            .to_string(),
    )
}

#[cfg(not(feature = "wasm-api-wgpu"))]
pub async fn warmup_pipeline_for_preset_with_status<F>(
    _preset: &WasmInferencePreset,
    _on_status: F,
) -> Result<(), String>
where
    F: FnMut(String),
{
    Err(
        "this build does not include wasm WebGPU support (`wasm-api-wgpu` feature missing)."
            .to_string(),
    )
}

#[cfg(feature = "wasm-api-wgpu")]
pub async fn infer_glb_from_image_bytes_with_preset_cached(
    image_bytes: &[u8],
    preset: &WasmInferencePreset,
) -> Result<Vec<u8>, String> {
    validate_wasm_preset_supported(preset)?;
    if image_bytes.is_empty() {
        return Err("image bytes are empty".to_string());
    }
    if wasm_preset_is_triposplat(preset) {
        return Err(
            "TripoSplat produces Gaussian splat assets, not GLB meshes; use infer_splat_from_image_bytes_with_options or infer_ply_from_image_bytes_with_options."
                .to_string(),
        );
    }
    if preset.synthesis_model.eq_ignore_ascii_case("trellis")
        || preset.synthesis_model.eq_ignore_ascii_case("trellis2")
        || preset.synthesis_model.eq_ignore_ascii_case("trellis.2")
    {
        #[cfg(feature = "trellis")]
        {
            return infer_trellis_glb_from_image_bytes_with_preset_cached(image_bytes, preset)
                .await;
        }
        #[cfg(not(feature = "trellis"))]
        {
            return Err(
                "this wasm build does not include trellis support (`trellis-wgpu` feature missing)."
                    .to_string(),
            );
        }
    }
    let mut active_preset = preset.clone();
    let mut attempted_f32_fallback = false;

    loop {
        warmup_triposg_pipeline_for_preset_with_status(&active_preset, |message| {
            web_sys::console::log_1(&message.into());
        })
        .await?;

        let mut cached = CACHED_WASM_PIPELINE.with(|cache| cache.borrow_mut().take());
        let (result, used_fp16_backend) = match cached.as_mut() {
            Some(CachedWasmPipeline::WgpuF32 {
                preset: cached_preset,
                state,
            }) if cached_preset == &active_preset => (
                run_inference_once(state, image_bytes, &active_preset).await,
                false,
            ),
            Some(CachedWasmPipeline::WgpuF16 {
                preset: cached_preset,
                state,
            }) if cached_preset == &active_preset => (
                run_inference_once(state, image_bytes, &active_preset).await,
                true,
            ),
            Some(_) => (
                Err("cached wasm pipeline preset mismatch".to_string()),
                false,
            ),
            None => (
                Err("cached wasm pipeline unavailable after warmup".to_string()),
                false,
            ),
        };
        CACHED_WASM_PIPELINE.with(|cache| {
            *cache.borrow_mut() = cached;
        });

        match result {
            Ok(bytes) => return Ok(bytes),
            Err(err) => {
                if !attempted_f32_fallback
                    && used_fp16_backend
                    && !active_preset.weights_precision.eq_ignore_ascii_case("f32")
                    && should_retry_with_f32_for_alignment_error(&err)
                {
                    attempted_f32_fallback = true;
                    active_preset.weights_precision = "f32";
                    web_sys::console::warn_1(
                        &format!(
                            "burn_synth wasm infer: fp16 WebGPU alignment failure detected; retrying with f32 backend ({err})"
                        )
                        .into(),
                    );
                    continue;
                }
                return Err(err);
            }
        }
    }
}

#[cfg(feature = "wasm-api-wgpu")]
pub async fn infer_splat_from_image_bytes_with_preset_cached(
    image_bytes: &[u8],
    preset: &WasmInferencePreset,
) -> Result<Vec<u8>, String> {
    infer_triposplat_bytes_from_image_bytes_with_preset_cached(image_bytes, preset, false).await
}

#[cfg(feature = "wasm-api-wgpu")]
pub async fn infer_ply_from_image_bytes_with_preset_cached(
    image_bytes: &[u8],
    preset: &WasmInferencePreset,
) -> Result<Vec<u8>, String> {
    infer_triposplat_bytes_from_image_bytes_with_preset_cached(image_bytes, preset, true).await
}

#[cfg(not(feature = "wasm-api-wgpu"))]
pub async fn infer_glb_from_image_bytes_with_preset_cached(
    _image_bytes: &[u8],
    _preset: &WasmInferencePreset,
) -> Result<Vec<u8>, String> {
    Err(
        "this build does not include wasm WebGPU support (`wasm-api-wgpu` feature missing)."
            .to_string(),
    )
}

#[cfg(not(feature = "wasm-api-wgpu"))]
pub async fn infer_splat_from_image_bytes_with_preset_cached(
    _image_bytes: &[u8],
    _preset: &WasmInferencePreset,
) -> Result<Vec<u8>, String> {
    Err(
        "this build does not include wasm WebGPU support (`wasm-api-wgpu` feature missing)."
            .to_string(),
    )
}

#[cfg(not(feature = "wasm-api-wgpu"))]
pub async fn infer_ply_from_image_bytes_with_preset_cached(
    _image_bytes: &[u8],
    _preset: &WasmInferencePreset,
) -> Result<Vec<u8>, String> {
    Err(
        "this build does not include wasm WebGPU support (`wasm-api-wgpu` feature missing)."
            .to_string(),
    )
}

#[wasm_bindgen]
pub async fn infer_glb_from_image_bytes(
    image_bytes: Vec<u8>,
    _file_name: Option<String>,
) -> Result<Uint8Array, JsValue> {
    infer_glb_from_image_bytes_with_options(image_bytes, None, None).await
}

#[wasm_bindgen]
pub async fn infer_glb_from_image_bytes_with_options(
    image_bytes: Vec<u8>,
    _file_name: Option<String>,
    options: Option<WasmInferOptions>,
) -> Result<Uint8Array, JsValue> {
    PANIC_HOOK_ONCE.call_once(console_error_panic_hook::set_once);

    let mut preset = WasmInferencePreset::default();
    if let Some(options) = options.as_ref() {
        options.apply_to_preset(&mut preset);
    }
    let bytes = Box::pin(infer_glb_from_image_bytes_with_preset_cached(
        image_bytes.as_slice(),
        &preset,
    ))
    .await
    .map_err(|err| JsValue::from_str(&err))?;
    Ok(Uint8Array::from(bytes.as_slice()))
}

#[wasm_bindgen]
pub async fn infer_splat_from_image_bytes(
    image_bytes: Vec<u8>,
    _file_name: Option<String>,
) -> Result<Uint8Array, JsValue> {
    infer_splat_from_image_bytes_with_options(image_bytes, None, None).await
}

#[wasm_bindgen]
pub async fn infer_splat_from_image_bytes_with_options(
    image_bytes: Vec<u8>,
    _file_name: Option<String>,
    options: Option<WasmInferOptions>,
) -> Result<Uint8Array, JsValue> {
    PANIC_HOOK_ONCE.call_once(console_error_panic_hook::set_once);

    web_sys::console::log_1(&"burn_synth wasm TripoSplat infer entry: build preset".into());
    let mut preset = WasmInferencePreset::default();
    preset.synthesis_model = "triposplat";
    if let Some(options) = options.as_ref() {
        options.apply_to_preset(&mut preset);
    }
    if !wasm_preset_is_triposplat(&preset) {
        return Err(JsValue::from_str(
            "infer_splat_from_image_bytes requires synthesis_model=triposplat",
        ));
    }
    web_sys::console::log_1(&"burn_synth wasm TripoSplat infer entry: dispatch".into());
    let bytes = Box::pin(infer_splat_from_image_bytes_with_preset_cached(
        image_bytes.as_slice(),
        &preset,
    ))
    .await
    .map_err(|err| JsValue::from_str(&err))?;
    web_sys::console::log_1(&"burn_synth wasm TripoSplat infer entry: complete".into());
    Ok(Uint8Array::from(bytes.as_slice()))
}

#[wasm_bindgen]
pub async fn infer_ply_from_image_bytes_with_options(
    image_bytes: Vec<u8>,
    _file_name: Option<String>,
    options: Option<WasmInferOptions>,
) -> Result<Uint8Array, JsValue> {
    PANIC_HOOK_ONCE.call_once(console_error_panic_hook::set_once);

    let mut preset = WasmInferencePreset::default();
    preset.synthesis_model = "triposplat";
    if let Some(options) = options.as_ref() {
        options.apply_to_preset(&mut preset);
    }
    if !wasm_preset_is_triposplat(&preset) {
        return Err(JsValue::from_str(
            "infer_ply_from_image_bytes requires synthesis_model=triposplat",
        ));
    }
    let bytes = Box::pin(infer_ply_from_image_bytes_with_preset_cached(
        image_bytes.as_slice(),
        &preset,
    ))
    .await
    .map_err(|err| JsValue::from_str(&err))?;
    Ok(Uint8Array::from(bytes.as_slice()))
}

#[wasm_bindgen]
pub async fn webgpu_available() -> bool {
    #[cfg(feature = "wasm-api-wgpu")]
    {
        wasm_webgpu_available().await
    }
    #[cfg(not(feature = "wasm-api-wgpu"))]
    {
        false
    }
}

#[cfg(feature = "wasm-api-wgpu")]
fn should_retry_with_f32_for_alignment_error(message: &str) -> bool {
    let message = message.to_ascii_lowercase();
    message.contains("multiple of 4")
        || (message.contains("binding") && message.contains("alignment"))
        || (message.contains("binding") && message.contains("size 514"))
}

#[cfg(feature = "wasm-api-wgpu")]
async fn infer_triposplat_bytes_from_image_bytes_with_preset_cached(
    image_bytes: &[u8],
    preset: &WasmInferencePreset,
    ply: bool,
) -> Result<Vec<u8>, String> {
    let splats =
        infer_triposplat_cloud_from_image_bytes_with_preset_cached(image_bytes, preset).await?;
    if ply {
        splats.to_ply_bytes()
    } else {
        splats.to_splat_bytes()
    }
}

#[cfg(feature = "wasm-api-wgpu")]
pub async fn infer_triposplat_cloud_from_image_bytes_with_preset_cached(
    image_bytes: &[u8],
    preset: &WasmInferencePreset,
) -> Result<GaussianSplatCloud, String> {
    web_sys::console::log_1(&"burn_synth wasm TripoSplat infer: validate preset".into());
    validate_wasm_preset_supported(preset)?;
    if image_bytes.is_empty() {
        return Err("image bytes are empty".to_string());
    }
    if !wasm_preset_is_triposplat(preset) {
        return Err("TripoSplat cloud inference requires synthesis_model=triposplat".to_string());
    }

    let mut active_preset = preset.clone();
    let mut attempted_f32_fallback = false;
    loop {
        web_sys::console::log_1(&"burn_synth wasm TripoSplat infer: warmup start".into());
        warmup_triposplat_pipeline_for_preset_with_status(&active_preset, |message| {
            web_sys::console::log_1(&message.into());
        })
        .await?;
        web_sys::console::log_1(&"burn_synth wasm TripoSplat infer: warmup done".into());

        let mut cached = CACHED_WASM_TRIPOSPLAT_PIPELINE.with(|cache| cache.borrow_mut().take());
        let (result, used_fp16_backend) = match cached.as_mut() {
            Some(CachedWasmTripoSplatPipeline::WgpuF32 {
                preset: cached_preset,
                state,
            }) if cached_preset == &active_preset => (
                run_triposplat_inference_once(state, image_bytes, &active_preset).await,
                false,
            ),
            Some(CachedWasmTripoSplatPipeline::WgpuF16 {
                preset: cached_preset,
                state,
            }) if cached_preset == &active_preset => (
                run_triposplat_inference_once(state, image_bytes, &active_preset).await,
                true,
            ),
            Some(_) => (
                Err("cached TripoSplat wasm pipeline preset mismatch".to_string()),
                false,
            ),
            None => (
                Err("cached TripoSplat wasm pipeline unavailable after warmup".to_string()),
                false,
            ),
        };
        CACHED_WASM_TRIPOSPLAT_PIPELINE.with(|cache| {
            *cache.borrow_mut() = cached;
        });

        match result {
            Ok(bytes) => return Ok(bytes),
            Err(err) => {
                if !attempted_f32_fallback
                    && used_fp16_backend
                    && !active_preset.weights_precision.eq_ignore_ascii_case("f32")
                    && should_retry_with_f32_for_alignment_error(&err)
                {
                    attempted_f32_fallback = true;
                    active_preset.weights_precision = "f32";
                    web_sys::console::warn_1(
                        &format!(
                            "burn_synth wasm TripoSplat infer: fp16 WebGPU alignment failure detected; retrying with f32 backend ({err})"
                        )
                        .into(),
                    );
                    continue;
                }
                return Err(err);
            }
        }
    }
}

#[cfg(feature = "wasm-api-wgpu")]
async fn run_triposplat_inference_once<BTripoSplat: Backend, BRmbg: Backend>(
    state: &mut WasmTripoSplatPipelineState<BTripoSplat, BRmbg>,
    image_bytes: &[u8],
    preset: &WasmInferencePreset,
) -> Result<GaussianSplatCloud, String> {
    web_sys::console::log_1(&"burn_synth wasm TripoSplat infer: prepare_image_data start".into());
    let prepared = prepare_image_data_from_bytes_async::<BRmbg>(
        image_bytes,
        state.rmbg.as_ref(),
        &triposplat_prepare_image_config(preset.triposplat_erode_radius),
    )
    .await
    .map_err(|err| format!("failed to prepare image tensor: {err}"))?;
    web_sys::console::log_1(&"burn_synth wasm TripoSplat infer: prepare_image_data done".into());

    BTripoSplat::seed(&state.triposplat_device, preset.seed);
    let image = prepared.to_tensor::<BTripoSplat>(&state.triposplat_device);
    let options = TripoSplatOptions {
        steps: preset.num_steps.max(1),
        guidance_scale: preset.guidance_scale,
        shift: preset.triposplat_shift,
        seed: preset.seed,
        num_gaussians: preset.triposplat_num_gaussians,
        erode_radius: preset.triposplat_erode_radius,
        cfg_mode: Default::default(),
        attention_query_chunk_tokens: None,
    };
    web_sys::console::log_1(
        &format!(
            "burn_synth wasm TripoSplat infer: run start (steps={} guidance_scale={:.3} gaussians={} shift={:.3})",
            options.steps, options.guidance_scale, options.num_gaussians, options.shift
        )
        .into(),
    );
    let encode_start = js_sys::Date::now();
    web_sys::console::log_1(&"burn_synth wasm TripoSplat infer: encode start".into());
    let condition = state.components.encode_preprocessed_image_random(image);
    web_sys::console::log_1(
        &format!(
            "burn_synth wasm TripoSplat infer: encode done (elapsed_ms={:.1})",
            js_sys::Date::now() - encode_start
        )
        .into(),
    );
    BTripoSplat::memory_cleanup(&state.triposplat_device);

    let sample_start = js_sys::Date::now();
    web_sys::console::log_1(&"burn_synth wasm TripoSplat infer: sample start".into());
    let latent = state
        .components
        .sample_latent_random(condition, options)
        .latent;
    BTripoSplat::memory_cleanup(&state.triposplat_device);
    web_sys::console::log_1(
        &format!(
            "burn_synth wasm TripoSplat infer: sample done (elapsed_ms={:.1})",
            js_sys::Date::now() - sample_start
        )
        .into(),
    );

    let decode_start = js_sys::Date::now();
    web_sys::console::log_1(&"burn_synth wasm TripoSplat infer: decode start".into());
    let num_gaussians = normalize_num_gaussians(options.num_gaussians)
        .map_err(|err| format!("TripoSplat wasm inference failed: {err}"))?;
    let num_points = (num_gaussians / state.components.decoder.gaussians_per_point()).max(1);
    web_sys::console::log_1(
        &format!(
            "burn_synth wasm TripoSplat infer: decode octree sample start (points={} level={})",
            num_points, OCTREE_MAX_VOXEL_LEVEL
        )
        .into(),
    );
    let octree_start = js_sys::Date::now();
    let sample = state
        .components
        .decoder
        .octree
        .sample_systematic_host_async(
            latent.clone(),
            num_points,
            OCTREE_MAX_VOXEL_LEVEL,
            options.seed,
        )
        .await
        .map_err(|err| format!("TripoSplat wasm inference failed: {err}"))?;
    web_sys::console::log_1(
        &format!(
            "burn_synth wasm TripoSplat infer: decode octree sample done (elapsed_ms={:.1})",
            js_sys::Date::now() - octree_start
        )
        .into(),
    );
    BTripoSplat::memory_cleanup(&state.triposplat_device);
    web_sys::console::log_1(
        &"burn_synth wasm TripoSplat infer: decode gaussian forward start".into(),
    );
    let gaussian_start = js_sys::Date::now();
    let features = state.components.decoder.gs.forward(&sample, latent);
    web_sys::console::log_1(
        &format!(
            "burn_synth wasm TripoSplat infer: decode gaussian forward done (elapsed_ms={:.1})",
            js_sys::Date::now() - gaussian_start
        )
        .into(),
    );
    BTripoSplat::memory_cleanup(&state.triposplat_device);
    web_sys::console::log_1(&"burn_synth wasm TripoSplat infer: decode build cloud start".into());
    let build_start = js_sys::Date::now();
    let splats = state
        .components
        .decoder
        .gs
        .build_cloud_async(&sample, features)
        .await
        .map_err(|err| format!("TripoSplat wasm inference failed: {err}"))?;
    web_sys::console::log_1(
        &format!(
            "burn_synth wasm TripoSplat infer: decode build cloud done (elapsed_ms={:.1})",
            js_sys::Date::now() - build_start
        )
        .into(),
    );
    web_sys::console::log_1(
        &format!(
            "burn_synth wasm TripoSplat infer: decode done (elapsed_ms={:.1})",
            js_sys::Date::now() - decode_start
        )
        .into(),
    );
    web_sys::console::log_1(
        &format!(
            "burn_synth wasm TripoSplat infer: done (splats={})",
            splats.len()
        )
        .into(),
    );
    Ok(splats)
}

async fn run_inference_once<BTriposg: Backend, BRmbg: Backend>(
    state: &mut WasmPipelineState<BTriposg, BRmbg>,
    image_bytes: &[u8],
    preset: &WasmInferencePreset,
) -> Result<Vec<u8>, String> {
    web_sys::console::log_1(&"burn_synth wasm infer: prepare_image_data start".into());
    let prepared = prepare_image_data_from_bytes_async::<BRmbg>(
        image_bytes,
        state.rmbg.as_ref(),
        &prepare_image_config_for_backend::<BRmbg>(),
    )
    .await
    .map_err(|err| format!("failed to prepare image tensor: {err}"))?;
    web_sys::console::log_1(&"burn_synth wasm infer: prepare_image_data done".into());

    BTriposg::seed(&state.triposg_device, preset.seed);
    web_sys::console::log_1(&"burn_synth wasm infer: encode_image_embeds start".into());
    let image_embeds = encode_image_embeds_for_wasm(state, &prepared)?;
    web_sys::console::log_1(&"burn_synth wasm infer: encode_image_embeds done".into());
    let batch_size = image_embeds.shape().dims::<3>()[0];
    let latents = Some(deterministic_latents_from_seed::<BTriposg>(
        preset.seed,
        batch_size,
        preset.num_tokens.max(64),
        state.triposg.transformer.config().in_channels,
        &state.triposg_device,
    ));

    let requested_flash_chunks = preset.flash_num_chunks.max(1);
    let effective_flash_chunks = if backend_uses_f16::<BTriposg>() {
        requested_flash_chunks.min(DEFAULT_WASM_FLASH_NUM_CHUNKS)
    } else {
        requested_flash_chunks
    };
    if effective_flash_chunks != requested_flash_chunks {
        web_sys::console::log_1(
            &format!(
                "burn_synth wasm infer: capping flash_num_chunks from {} to {} for fp16 WebGPU portability",
                requested_flash_chunks, effective_flash_chunks
            )
            .into(),
        );
    }

    let flash = FlashExtractConfig {
        bounds: DEFAULT_BOUNDS,
        octree_depth: preset.flash_octree_depth.max(1),
        num_chunks: effective_flash_chunks,
        mc_level: 0.0,
        min_resolution: preset.resolution.max(2),
        mini_grid_num: preset.flash_mini_grid_num.max(1),
    };
    web_sys::console::log_1(
        &format!(
            "burn_synth wasm infer: flash_extract start (steps={} tokens={} octree_depth={} min_resolution={} mini_grid_num={} num_chunks={} faces={})",
            preset.num_steps.max(1),
            preset.num_tokens.max(64),
            flash.octree_depth,
            flash.min_resolution,
            flash.mini_grid_num,
            flash.num_chunks,
            preset.faces
        )
        .into(),
    );
    let flash_output = state
        .triposg
        .sample_mesh_flash_from_embeds_async_wasm(
            image_embeds,
            preset.num_steps.max(1),
            preset.num_tokens.max(64),
            DEFAULT_GUIDANCE_SCALE,
            &flash,
            latents,
        )
        .await
        .map_err(|err| format!("TripoSG flash geometry extraction failed: {err}"))?;
    web_sys::console::log_1(&"burn_synth wasm infer: flash_extract done".into());

    let mut mesh = flash_output
        .mesh
        .ok_or_else(|| "TripoSG mesh extraction returned an empty mesh".to_string())?;

    if preset.faces > 0 && mesh.faces.len() > preset.faces {
        web_sys::console::log_1(
            &format!(
                "burn_synth wasm infer: decimate start (from_faces={} target_faces={})",
                mesh.faces.len(),
                preset.faces
            )
            .into(),
        );
        mesh = decimate_tripo_mesh(&mesh, preset.faces)
            .map_err(|err| format!("mesh decimation failed: {err}"))?;
        web_sys::console::log_1(
            &format!(
                "burn_synth wasm infer: decimate done (to_faces={})",
                mesh.faces.len()
            )
            .into(),
        );
    }

    let mesh = tripo_mesh_to_mesh(mesh);
    web_sys::console::log_1(&"burn_synth wasm infer: serialize_glb start".into());
    mesh_to_glb_bytes(&mesh).map_err(|err| format!("failed to serialize GLB: {err}"))
}

#[cfg(feature = "trellis")]
fn trellis_quality_from_wasm_preset(preset: &WasmInferencePreset) -> burn_trellis::TrellisQuality {
    match preset.quality.trim().to_ascii_lowercase().as_str() {
        "fast" => burn_trellis::TrellisQuality::Low,
        "full" => burn_trellis::TrellisQuality::High,
        _ => burn_trellis::TrellisQuality::Medium,
    }
}

#[cfg(all(feature = "wasm-api-wgpu", feature = "trellis"))]
async fn warmup_trellis_pipeline_for_preset_with_status<F>(
    preset: &WasmInferencePreset,
    mut on_status: F,
) -> Result<(), String>
where
    F: FnMut(String),
{
    let Some(_adapter_profile) = wasm_webgpu_adapter_profile().await else {
        return Err(
            "WebGPU is unavailable in this browser/runtime; CPU fallback is disabled for Trellis wasm."
                .to_string(),
        );
    };
    initialize_wgpu_runtime_for_wasm().await?;

    let cache_hit = CACHED_WASM_TRELLIS_PIPELINE.with(|cache| {
        cache
            .borrow()
            .as_ref()
            .is_some_and(|(cached, _)| cached == preset)
    });
    if cache_hit {
        on_status("Trellis model weights already loaded (cache hit).".to_string());
        return Ok(());
    }

    let state = load_trellis_pipeline_state_wasm(preset, &mut on_status).await?;
    CACHED_WASM_TRELLIS_PIPELINE.with(|cache| {
        *cache.borrow_mut() = Some((preset.clone(), state));
    });
    Ok(())
}

#[cfg(all(feature = "wasm-api-wgpu", feature = "trellis"))]
async fn load_trellis_pipeline_state_wasm<F>(
    _preset: &WasmInferencePreset,
    on_status: &mut F,
) -> Result<WasmTrellisPipelineState, String>
where
    F: FnMut(String),
{
    on_status(format!(
        "WASM model asset root: {}",
        resolve_wasm_model_base_url()
    ));

    let weights_root_url = wasm_model_root(ROOT_TRELLIS);
    let image_large_root_url = wasm_model_root(ROOT_TRELLIS_IMAGE_LARGE);
    let weights_root = std::path::PathBuf::from(DEFAULT_LOCAL_MODEL_BASE_URL).join(ROOT_TRELLIS);
    let image_large_root =
        std::path::PathBuf::from(DEFAULT_LOCAL_MODEL_BASE_URL).join(ROOT_TRELLIS_IMAGE_LARGE);

    trellis_virtual_fs::clear_virtual_files();

    let mut totals = DownloadTotals::default();
    let mut host_ram_budget = WasmHostMemoryBudget::new(web_max_host_ram_bytes());

    let pipeline_url = join_web_path(&weights_root_url, "pipeline.json");
    let pipeline_bytes = download_binary_with_status(
        &pipeline_url,
        "Trellis pipeline config",
        16 * 1024 * 1024,
        &mut totals,
        &mut host_ram_budget,
        on_status,
    )
    .await?;
    let pipeline_virtual_path = weights_root.join("pipeline.json");
    trellis_virtual_fs::register_virtual_file(&pipeline_virtual_path, pipeline_bytes.clone());
    let pipeline = TrellisPipelineConfig::from_json_bytes(&pipeline_bytes).map_err(|err| {
        format!(
            "failed to parse Trellis pipeline config '{}': {err}",
            pipeline_url
        )
    })?;

    for (model_key, model_stem) in pipeline.args.models.iter() {
        let config_url = trellis_resolve_model_url(
            model_stem,
            "json",
            weights_root_url.as_str(),
            image_large_root_url.as_str(),
        );
        let config_virtual_path = trellis_resolve_model_virtual_path(
            model_stem,
            "json",
            weights_root.as_path(),
            image_large_root.as_path(),
        );
        trellis_virtual_fs::register_virtual_url(&config_virtual_path, config_url.clone());

        let safetensors_url = trellis_resolve_model_url(
            model_stem,
            "safetensors",
            weights_root_url.as_str(),
            image_large_root_url.as_str(),
        );
        let safetensors_virtual_path = trellis_resolve_model_virtual_path(
            model_stem,
            "safetensors",
            weights_root.as_path(),
            image_large_root.as_path(),
        );

        let prefer_f16 = trellis_model_prefers_f16(model_key.as_str());
        let candidate_urls = candidate_burnpack_names(&safetensors_url, prefer_f16);
        let candidate_virtuals = candidate_burnpack_names(
            safetensors_virtual_path.to_string_lossy().as_ref(),
            prefer_f16,
        );
        let mut manifest_found = false;
        for (candidate_url, candidate_virtual) in
            candidate_urls.iter().zip(candidate_virtuals.iter())
        {
            let manifest_url = format!("{candidate_url}.parts.json");
            if fetch_optional_text(&manifest_url).await?.is_none() {
                continue;
            }
            let manifest_virtual_path =
                std::path::PathBuf::from(format!("{candidate_virtual}.parts.json"));
            trellis_virtual_fs::register_virtual_url(&manifest_virtual_path, manifest_url);
            manifest_found = true;
            break;
        }
        if !manifest_found {
            return Err(format!(
                "missing Trellis burnpack parts manifest for model key '{}' (stem '{}')",
                model_key, model_stem
            ));
        }
    }

    if let Some(image_cond_model) = pipeline.args.image_cond_model.as_ref() {
        let model_name = image_cond_model.args.model_name.trim();
        if !model_name.is_empty() {
            let config_url = join_web_path(
                &weights_root_url,
                format!("{model_name}/config.json").as_str(),
            );
            let config_virtual =
                weights_root.join(std::path::Path::new(model_name).join("config.json"));
            trellis_virtual_fs::register_virtual_url(config_virtual, config_url);

            let model_safetensors_url = join_web_path(
                &weights_root_url,
                format!("{model_name}/model.safetensors").as_str(),
            );
            let model_safetensors_virtual =
                weights_root.join(std::path::Path::new(model_name).join("model.safetensors"));
            let candidate_urls = candidate_burnpack_names(&model_safetensors_url, false);
            let candidate_virtuals = candidate_burnpack_names(
                model_safetensors_virtual.to_string_lossy().as_ref(),
                false,
            );
            let mut manifest_found = false;
            for (candidate_url, candidate_virtual) in
                candidate_urls.iter().zip(candidate_virtuals.iter())
            {
                let manifest_url = format!("{candidate_url}.parts.json");
                if fetch_optional_text(&manifest_url).await?.is_none() {
                    continue;
                }
                let manifest_virtual =
                    std::path::PathBuf::from(format!("{candidate_virtual}.parts.json"));
                trellis_virtual_fs::register_virtual_url(manifest_virtual, manifest_url);
                manifest_found = true;
                break;
            }
            if !manifest_found {
                return Err(format!(
                    "missing Trellis image-conditioning burnpack parts manifest for '{}'",
                    model_name
                ));
            }
        }
    }

    let trellis = Trellis2Pipeline::new(Trellis2PipelineConfig {
        weights_root,
        image_large_root: Some(image_large_root),
    })
    .map_err(|err| format!("failed to initialize Trellis2 pipeline: {err}"))?;
    trellis
        .validate_runtime()
        .map_err(|err| format!("failed to validate Trellis2 runtime assets: {err}"))?;

    Ok(WasmTrellisPipelineState { trellis })
}

#[cfg(all(feature = "wasm-api-wgpu", feature = "trellis"))]
fn trellis_resolve_model_url(
    stem: &str,
    ext: &str,
    weights_root: &str,
    image_large_root: &str,
) -> String {
    if stem.starts_with("ckpts/") {
        return join_web_path(weights_root, format!("{stem}.{ext}").as_str());
    }
    if let Some((_, suffix)) = stem.split_once("/ckpts/") {
        return join_web_path(image_large_root, format!("ckpts/{suffix}.{ext}").as_str());
    }
    join_web_path(weights_root, format!("{stem}.{ext}").as_str())
}

#[cfg(all(feature = "wasm-api-wgpu", feature = "trellis"))]
fn trellis_resolve_model_virtual_path(
    stem: &str,
    ext: &str,
    weights_root: &std::path::Path,
    image_large_root: &std::path::Path,
) -> std::path::PathBuf {
    if stem.starts_with("ckpts/") {
        return weights_root.join(format!("{stem}.{ext}"));
    }
    if let Some((_, suffix)) = stem.split_once("/ckpts/") {
        return image_large_root.join(format!("ckpts/{suffix}.{ext}"));
    }
    weights_root.join(format!("{stem}.{ext}"))
}

#[cfg(all(feature = "wasm-api-wgpu", feature = "trellis"))]
fn trellis_model_prefers_f16(model_key: &str) -> bool {
    matches!(model_key, "shape_slat_decoder" | "tex_slat_decoder")
}

#[cfg(all(feature = "wasm-api-wgpu", feature = "trellis"))]
async fn infer_trellis_glb_from_image_bytes_with_preset_cached(
    image_bytes: &[u8],
    preset: &WasmInferencePreset,
) -> Result<Vec<u8>, String> {
    warmup_trellis_pipeline_for_preset_with_status(preset, |message| {
        web_sys::console::log_1(&message.into());
    })
    .await?;

    let mut cached = CACHED_WASM_TRELLIS_PIPELINE.with(|cache| cache.borrow_mut().take());
    let result = match cached.as_mut() {
        Some((cached_preset, state)) if cached_preset == preset => {
            run_trellis_inference_once(state, image_bytes, preset).await
        }
        Some(_) => Err("cached wasm Trellis pipeline preset mismatch".to_string()),
        None => Err("cached wasm Trellis pipeline unavailable after warmup".to_string()),
    };
    CACHED_WASM_TRELLIS_PIPELINE.with(|cache| {
        *cache.borrow_mut() = cached;
    });
    result
}

#[cfg(all(feature = "wasm-api-wgpu", feature = "trellis"))]
async fn run_trellis_inference_once(
    state: &mut WasmTrellisPipelineState,
    image_bytes: &[u8],
    preset: &WasmInferencePreset,
) -> Result<Vec<u8>, String> {
    let source = image::load_from_memory(image_bytes)
        .map_err(|err| format!("failed to decode image bytes for Trellis: {err}"))?
        .to_rgba8();

    let options = TrellisRunOptions {
        quality: trellis_quality_from_wasm_preset(preset),
        device: TrellisDevice::Wgpu,
        seed: Some(preset.seed),
        target_faces: (preset.faces > 0).then_some(preset.faces),
        ..TrellisRunOptions::default()
    };
    let mesh = state
        .trellis
        .infer_mesh_from_image(image::DynamicImage::ImageRgba8(source), &options)
        .map_err(|err| format!("Trellis2 wasm inference failed: {err}"))?;
    let mesh: Mesh = mesh.into();
    mesh_to_glb_bytes(&mesh).map_err(|err| format!("failed to serialize Trellis GLB: {err}"))
}

async fn load_pipeline_state<BTriposg: Backend, BRmbg: Backend, F>(
    preset: &WasmInferencePreset,
    on_status: &mut F,
) -> Result<WasmPipelineState<BTriposg, BRmbg>, String>
where
    F: FnMut(String),
{
    let parity = triposg_runtime_profile(Some(preset.resolution));
    set_rmbg_strict_interp_override(Some(parity.strict_rmbg_interp));
    let prefer_f16_default = should_prefer_f16_triposg_weights(parity);
    let use_wgpu = is_wgpu_backend::<BTriposg>();
    let backend_is_f16 = backend_uses_f16::<BTriposg>();
    let requested_tripo_precision = if preset.weights_precision.eq_ignore_ascii_case("f16") {
        "f16"
    } else if preset.weights_precision.eq_ignore_ascii_case("auto") {
        "auto"
    } else {
        "f32"
    };
    let requested_rmbg_precision = if preset.rmbg_weights_precision.eq_ignore_ascii_case("f16") {
        "f16"
    } else if preset.rmbg_weights_precision.eq_ignore_ascii_case("f32") {
        "f32"
    } else {
        "auto"
    };
    let auto_prefer_f16 = if use_wgpu {
        backend_is_f16
    } else {
        prefer_f16_default
    };
    let effective_prefer_f16 = match requested_tripo_precision {
        "f16" => true,
        "f32" => false,
        _ => auto_prefer_f16,
    };
    let allow_cross_precision_fallback = requested_tripo_precision == "auto";
    let strict_precision = !allow_cross_precision_fallback;
    let precision_reason = match requested_tripo_precision {
        "f16" => "forced by options (f16)",
        "f32" => "forced by options (f32)",
        _ => {
            if use_wgpu {
                if backend_is_f16 {
                    "auto (wasm WebGPU backend-aligned fp16)"
                } else {
                    "auto (wasm WebGPU backend-aligned fp32)"
                }
            } else {
                "auto (runtime parity profile)"
            }
        }
    };
    let precision_label = if effective_prefer_f16 { "f16" } else { "f32" };
    on_status(format!(
        "TripoSG weight precision policy: {precision_label} ({})",
        precision_reason
    ));
    on_status(format!(
        "WASM model asset root: {}",
        resolve_wasm_model_base_url()
    ));
    let rmbg_backend_is_f16 = backend_uses_f16::<BRmbg>();
    let prefer_f16_rmbg = match requested_rmbg_precision {
        "f16" => true,
        "f32" => false,
        _ => rmbg_backend_is_f16,
    };
    let allow_cross_precision_rmbg = requested_rmbg_precision == "auto";
    on_status(format!(
        "RMBG weight precision policy: {} ({})",
        if prefer_f16_rmbg { "f16" } else { "f32" },
        if requested_rmbg_precision == "auto" {
            if rmbg_backend_is_f16 {
                "auto (wasm RMBG backend-aligned fp16)"
            } else {
                "auto (wasm RMBG backend-aligned fp32)"
            }
        } else if prefer_f16_rmbg {
            "forced by options (f16)"
        } else {
            "forced by options (f32)"
        }
    ));
    // CPU wasm path cannot fit full-f32 model footprints under the 4 GiB host cap.
    // Keep CPU fallback on f16, and prefer f16 burnpacks for wasm WebGPU fp16 runtime.
    let prefer_f16_vae = if use_wgpu { effective_prefer_f16 } else { true };
    let prefer_f16_dit = if use_wgpu { effective_prefer_f16 } else { true };
    let prefer_f16_dino = if use_wgpu { effective_prefer_f16 } else { true };

    let triposg_device = BTriposg::Device::default();
    let rmbg_device = BRmbg::Device::default();
    let mut totals = DownloadTotals::default();
    let mut host_ram_budget = WasmHostMemoryBudget::new(web_max_host_ram_bytes());

    let options = TripoWasmLoadOptions {
        strict_dino_preprocess: parity.strict_dino_preprocess,
        strict_precision,
        prefer_f16_vae,
        prefer_f16_dit,
        prefer_f16_dino,
    };
    let mut load_ctx = WasmLoadContext {
        totals: &mut totals,
        host_ram_budget: &mut host_ram_budget,
        on_status,
    };

    let rmbg = if wasm_preset_skips_rmbg(preset) {
        load_ctx.status("Skipping RMBG component because rmbg_model=none.".to_string());
        None
    } else {
        Some(
            load_rmbg14_pipeline_wasm(
                &rmbg_device,
                prefer_f16_rmbg,
                allow_cross_precision_rmbg,
                &mut load_ctx,
            )
            .await?,
        )
    };
    let triposg = load_triposg_pipeline_wasm(&triposg_device, options, &mut load_ctx).await?;

    Ok(WasmPipelineState {
        triposg_device,
        rmbg,
        triposg,
    })
}

#[cfg(feature = "wasm-api-wgpu")]
async fn warmup_triposplat_pipeline_for_preset_with_status<F>(
    preset: &WasmInferencePreset,
    mut on_status: F,
) -> Result<(), String>
where
    F: FnMut(String),
{
    web_sys::console::log_1(&"burn_synth wasm TripoSplat warmup: adapter profile start".into());
    let Some(adapter_profile) = wasm_webgpu_adapter_profile().await else {
        return Err(
            "WebGPU is unavailable in this browser/runtime; CPU fallback is disabled for TripoSplat wasm."
                .to_string(),
        );
    };
    web_sys::console::log_1(&"burn_synth wasm TripoSplat warmup: adapter profile done".into());
    web_sys::console::log_1(&"burn_synth wasm TripoSplat warmup: runtime init start".into());
    initialize_wgpu_runtime_for_wasm().await?;
    web_sys::console::log_1(&"burn_synth wasm TripoSplat warmup: runtime init done".into());
    let precision =
        resolve_triposplat_wgpu_precision_for_preset(preset, adapter_profile.shader_f16_supported)?;
    let prefer_f16_artifacts = true;
    on_status(format!(
        "TripoSplat wasm uses {} WebGPU compute and {} artifacts{}.",
        precision,
        if prefer_f16_artifacts { "f16" } else { "f32" },
        if preset.weights_precision.eq_ignore_ascii_case("auto")
            && !adapter_profile.shader_f16_supported
        {
            " because this adapter lacks shader-f16"
        } else {
            ""
        }
    ));

    let cache_hit = CACHED_WASM_TRIPOSPLAT_PIPELINE.with(|cache| {
        let guard = cache.borrow();
        match (&*guard, precision) {
            (Some(CachedWasmTripoSplatPipeline::WgpuF32 { preset: cached, .. }), "f32") => {
                cached == preset
            }
            (Some(CachedWasmTripoSplatPipeline::WgpuF16 { preset: cached, .. }), "f16") => {
                cached == preset
            }
            _ => false,
        }
    });
    if cache_hit {
        on_status("TripoSplat model weights already loaded (cache hit).".to_string());
        return Ok(());
    }

    let loaded = match precision {
        "f16" => CachedWasmTripoSplatPipeline::WgpuF16 {
            preset: preset.clone(),
            state: load_triposplat_pipeline_state::<WgpuTripoSplatBackendF16, WgpuRmbgBackend, _>(
                preset,
                prefer_f16_artifacts,
                &mut on_status,
            )
            .await?,
        },
        _ => CachedWasmTripoSplatPipeline::WgpuF32 {
            preset: preset.clone(),
            state: load_triposplat_pipeline_state::<WgpuTripoSplatBackendF32, WgpuRmbgBackend, _>(
                preset,
                prefer_f16_artifacts,
                &mut on_status,
            )
            .await?,
        },
    };

    CACHED_WASM_TRIPOSPLAT_PIPELINE.with(|cache| {
        *cache.borrow_mut() = Some(loaded);
    });
    Ok(())
}

#[cfg(feature = "wasm-api-wgpu")]
async fn load_triposplat_pipeline_state<BTripoSplat: Backend, BRmbg: Backend, F>(
    preset: &WasmInferencePreset,
    prefer_f16_triposplat_artifacts: bool,
    on_status: &mut F,
) -> Result<WasmTripoSplatPipelineState<BTripoSplat, BRmbg>, String>
where
    F: FnMut(String),
{
    let backend_is_f16 = backend_uses_f16::<BTripoSplat>();
    let requested_precision = if preset.weights_precision.eq_ignore_ascii_case("f16") {
        "f16"
    } else if preset.weights_precision.eq_ignore_ascii_case("auto") {
        "auto"
    } else {
        "f32"
    };
    let prefer_f16 = prefer_f16_triposplat_artifacts;
    let triposplat_compute_dtype = if prefer_f16 && !backend_is_f16 {
        Some(FloatDType::F32)
    } else {
        None
    };
    let allow_cross_precision_fallback = requested_precision == "auto";
    let requested_rmbg_precision = if preset.rmbg_weights_precision.eq_ignore_ascii_case("f16") {
        "f16"
    } else if preset.rmbg_weights_precision.eq_ignore_ascii_case("f32") {
        "f32"
    } else {
        "auto"
    };
    let rmbg_backend_is_f16 = backend_uses_f16::<BRmbg>();
    let prefer_f16_rmbg = match requested_rmbg_precision {
        "f16" => true,
        "f32" => false,
        _ => rmbg_backend_is_f16,
    };

    on_status(format!(
        "TripoSplat weight precision policy: {} ({})",
        if prefer_f16 { "f16" } else { "f32" },
        if requested_precision == "auto" {
            "auto (wasm shader-f16 policy)"
        } else {
            "forced by options"
        }
    ));
    on_status(format!(
        "TripoSplat compute precision policy: {} ({})",
        if matches!(triposplat_compute_dtype, Some(FloatDType::F32)) {
            "f32"
        } else if backend_is_f16 {
            "backend f16"
        } else {
            "backend f32"
        },
        if prefer_f16 && backend_is_f16 {
            "f16 artifacts remain fp16 for upstream-style wasm memory/perf"
        } else if prefer_f16 {
            "f16 artifacts are promoted after load for backend compatibility"
        } else {
            "artifact dtype"
        }
    ));
    on_status(format!(
        "RMBG weight precision policy: {} ({})",
        if prefer_f16_rmbg { "f16" } else { "f32" },
        if requested_rmbg_precision == "auto" {
            if rmbg_backend_is_f16 {
                "auto (wasm RMBG backend-aligned fp16)"
            } else {
                "auto (wasm RMBG backend-aligned fp32)"
            }
        } else if prefer_f16_rmbg {
            "forced by options (f16)"
        } else {
            "forced by options (f32)"
        }
    ));
    on_status(format!(
        "WASM model asset root: {}",
        resolve_wasm_model_base_url()
    ));

    let triposplat_device = BTripoSplat::Device::default();
    let rmbg_device = BRmbg::Device::default();
    let mut totals = DownloadTotals::default();
    let mut host_ram_budget = WasmHostMemoryBudget::new(web_max_host_ram_bytes());
    let mut load_ctx = WasmLoadContext {
        totals: &mut totals,
        host_ram_budget: &mut host_ram_budget,
        on_status,
    };

    let rmbg = if wasm_preset_skips_rmbg(preset) {
        load_ctx.status("Skipping TripoSplat RMBG component because rmbg_model=none.".to_string());
        None
    } else {
        load_ctx.status("Loading TripoSplat RMBG component...".to_string());
        let rmbg = load_rmbg14_pipeline_wasm(
            &rmbg_device,
            prefer_f16_rmbg,
            requested_rmbg_precision == "auto",
            &mut load_ctx,
        )
        .await?;
        load_ctx.status("Loaded TripoSplat RMBG component.".to_string());
        Some(rmbg)
    };
    load_ctx.status("Loading TripoSplat runtime components...".to_string());
    let components = load_triposplat_components_wasm(
        &triposplat_device,
        prefer_f16,
        triposplat_compute_dtype,
        allow_cross_precision_fallback,
        &mut load_ctx,
    )
    .await?;
    load_ctx.status("Loaded TripoSplat runtime components.".to_string());

    Ok(WasmTripoSplatPipelineState {
        triposplat_device,
        rmbg,
        components,
    })
}

async fn load_rmbg14_pipeline_wasm<B: Backend, F>(
    device: &B::Device,
    prefer_f16: bool,
    allow_cross_precision_fallback: bool,
    load_ctx: &mut WasmLoadContext<'_, F>,
) -> Result<RmbgPipeline<B>, String>
where
    F: FnMut(String),
{
    let rmbg_root = wasm_model_root(ROOT_RMBG14);
    load_ctx.status(format!("Loading RMBG config from {rmbg_root}..."));
    let base_safetensors_url = join_web_path(&rmbg_root, "model.safetensors");
    let config_json = fetch_optional_text(&join_web_path(&rmbg_root, "config.json")).await?;

    let config = if let Some(json) = config_json.as_ref() {
        load_rmbg_config_from_json_bytes(json.as_bytes())
            .map_err(|err| format!("failed to parse RMBG config: {err}"))?
    } else {
        burn_foreground::rmbg14::RmbgConfig::rmbg_1_4()
    };
    let processor = burn_foreground::preprocess::RmbgImageProcessor::default();

    if let Some(model) = try_load_model_from_parts_wasm(
        &base_safetensors_url,
        "RMBG",
        prefer_f16,
        allow_cross_precision_fallback,
        load_ctx,
        || BriaRmbg::new(device, config.clone()),
        |model, part_bytes| {
            apply_rmbg_burnpack_part_bytes(model, part_bytes)
                .map_err(|err| format!("failed to apply RMBG burnpack part bytes: {err}"))
        },
    )
    .await?
    {
        return Ok(RmbgPipeline::new(model, processor));
    }

    Err(format!(
        "RMBG wasm loader requires burnpack parts manifests under {rmbg_root}; missing *.bpk.parts.json for requested precision."
    ))
}

#[cfg(feature = "wasm-api-wgpu")]
async fn load_triposplat_components_wasm<B: Backend, F>(
    device: &B::Device,
    prefer_f16: bool,
    compute_dtype: Option<FloatDType>,
    allow_cross_precision_fallback: bool,
    load_ctx: &mut WasmLoadContext<'_, F>,
) -> Result<TripoSplatRuntimeComponents<B>, String>
where
    F: FnMut(String),
{
    load_ctx.status("Loading TripoSplat DINOv3 component...".to_string());
    let mut dinov3 =
        load_triposplat_dinov3_wasm(device, prefer_f16, allow_cross_precision_fallback, load_ctx)
            .await?;
    if let Some(dtype) = compute_dtype {
        dinov3 = burn_triposplat::import::cast_module_float_dtype(dinov3, dtype);
        load_ctx.status("Cast TripoSplat DINOv3 component to f32 compute.".to_string());
    }
    cleanup_wasm_backend_memory::<B, _>(device, "TripoSplat DINOv3 load", load_ctx)?;
    load_ctx.status("Loaded TripoSplat DINOv3 component.".to_string());
    load_ctx.status("Loading TripoSplat Flux2 VAE encoder component...".to_string());
    let mut flux2_vae_encoder =
        load_triposplat_flux2_wasm(device, prefer_f16, allow_cross_precision_fallback, load_ctx)
            .await?;
    if let Some(dtype) = compute_dtype {
        flux2_vae_encoder =
            burn_triposplat::import::cast_module_float_dtype(flux2_vae_encoder, dtype);
        load_ctx.status("Cast TripoSplat Flux2 VAE encoder component to f32 compute.".to_string());
    }
    cleanup_wasm_backend_memory::<B, _>(device, "TripoSplat Flux2 VAE load", load_ctx)?;
    load_ctx.status("Loaded TripoSplat Flux2 VAE encoder component.".to_string());
    load_ctx.status("Loading TripoSplat flow component...".to_string());
    let mut flow =
        load_triposplat_flow_wasm(device, prefer_f16, allow_cross_precision_fallback, load_ctx)
            .await?;
    if let Some(dtype) = compute_dtype {
        flow = burn_triposplat::import::cast_module_float_dtype(flow, dtype);
        flow.reset_canonical_pos_pe(device);
        load_ctx.status("Cast TripoSplat flow component to f32 compute.".to_string());
    }
    cleanup_wasm_backend_memory::<B, _>(device, "TripoSplat flow load", load_ctx)?;
    load_ctx.status("Loaded TripoSplat flow component.".to_string());
    load_ctx.status("Loading TripoSplat decoder component...".to_string());
    let mut decoder =
        load_triposplat_decoder_wasm(device, prefer_f16, allow_cross_precision_fallback, load_ctx)
            .await?;
    if let Some(dtype) = compute_dtype {
        decoder = burn_triposplat::import::cast_module_float_dtype(decoder, dtype);
        load_ctx.status("Cast TripoSplat decoder component to f32 compute.".to_string());
    }
    cleanup_wasm_backend_memory::<B, _>(device, "TripoSplat decoder load", load_ctx)?;
    load_ctx.status("Loaded TripoSplat decoder component.".to_string());

    Ok(TripoSplatRuntimeComponents {
        dinov3,
        flux2_vae_encoder,
        flow,
        decoder,
    })
}

#[cfg(feature = "wasm-api-wgpu")]
fn cleanup_wasm_backend_memory<B: Backend, F>(
    device: &B::Device,
    label: &str,
    load_ctx: &mut WasmLoadContext<'_, F>,
) -> Result<(), String>
where
    F: FnMut(String),
{
    B::memory_cleanup(device);
    load_ctx.status(format!("Cleaned wasm backend memory after {label}."));
    Ok(())
}

#[cfg(feature = "wasm-api-wgpu")]
async fn load_triposplat_dinov3_wasm<B: Backend, F>(
    device: &B::Device,
    prefer_f16: bool,
    allow_cross_precision_fallback: bool,
    load_ctx: &mut WasmLoadContext<'_, F>,
) -> Result<burn_dino::model::dinov3::DinoV3ViT<B>, String>
where
    F: FnMut(String),
{
    let config = burn_dino::model::dinov3::DinoV3Config::vit_h_16_plus(None);
    let artifact = triposplat_required_artifact("dino_v3_vit_h")?;
    try_load_triposplat_artifact_from_parts_wasm(
        artifact,
        "TripoSplat DINOv3",
        prefer_f16,
        allow_cross_precision_fallback,
        load_ctx,
        || config.clone().init(device),
        |model, part_bytes| {
            burn_dino::model::dinov3::import::apply_dinov3_burnpack_part_bytes(model, part_bytes)
                .map_err(|err| format!("failed to apply TripoSplat DINOv3 burnpack part: {err}"))
        },
    )
    .await?
    .ok_or_else(|| triposplat_missing_wasm_artifact_message(artifact, "TripoSplat DINOv3"))
}

#[cfg(feature = "wasm-api-wgpu")]
async fn load_triposplat_flux2_wasm<B: Backend, F>(
    device: &B::Device,
    prefer_f16: bool,
    allow_cross_precision_fallback: bool,
    load_ctx: &mut WasmLoadContext<'_, F>,
) -> Result<burn_flux::Flux2VaeEncoder<B>, String>
where
    F: FnMut(String),
{
    let config = burn_flux::Flux2VaeEncoderConfig::flux2();
    let artifact = triposplat_required_artifact("flux2_vae_encoder")?;
    try_load_triposplat_artifact_from_parts_wasm(
        artifact,
        "TripoSplat Flux2 VAE encoder",
        prefer_f16,
        allow_cross_precision_fallback,
        load_ctx,
        || config.clone().init(device),
        |model, part_bytes| {
            burn_flux::flux2_import::apply_flux2_vae_encoder_burnpack_part_bytes(model, part_bytes)
                .map_err(|err| format!("failed to apply TripoSplat Flux2 VAE burnpack part: {err}"))
        },
    )
    .await?
    .ok_or_else(|| triposplat_missing_wasm_artifact_message(artifact, "TripoSplat Flux2 VAE"))
}

#[cfg(feature = "wasm-api-wgpu")]
async fn load_triposplat_flow_wasm<B: Backend, F>(
    device: &B::Device,
    prefer_f16: bool,
    allow_cross_precision_fallback: bool,
    load_ctx: &mut WasmLoadContext<'_, F>,
) -> Result<burn_triposplat::LatentSeqMmFlowModel<B>, String>
where
    F: FnMut(String),
{
    let config = LatentSeqMmFlowModelConfig::triposplat();
    let artifact = triposplat_required_artifact("triposplat_flow")?;
    try_load_triposplat_artifact_from_parts_wasm(
        artifact,
        "TripoSplat flow",
        prefer_f16,
        allow_cross_precision_fallback,
        load_ctx,
        || config.clone().init(device),
        |model, part_bytes| {
            burn_triposplat::import::apply_triposplat_flow_burnpack_part_bytes(model, part_bytes)
                .map_err(|err| format!("failed to apply TripoSplat flow burnpack part: {err}"))
        },
    )
    .await?
    .ok_or_else(|| triposplat_missing_wasm_artifact_message(artifact, "TripoSplat flow"))
}

#[cfg(feature = "wasm-api-wgpu")]
async fn load_triposplat_decoder_wasm<B: Backend, F>(
    device: &B::Device,
    prefer_f16: bool,
    allow_cross_precision_fallback: bool,
    load_ctx: &mut WasmLoadContext<'_, F>,
) -> Result<OctreeGaussianDecoder<B>, String>
where
    F: FnMut(String),
{
    let octree_config = OctreeProbabilityFixedlenDecoderConfig::triposplat();
    let gs_config = ElasticGaussianFixedlenDecoderConfig::triposplat();
    let artifact = triposplat_required_artifact("triposplat_vae_decoder")?;
    try_load_triposplat_artifact_from_parts_wasm(
        artifact,
        "TripoSplat decoder",
        prefer_f16,
        allow_cross_precision_fallback,
        load_ctx,
        || OctreeGaussianDecoder::new(device, octree_config.clone(), gs_config.clone()),
        |model, part_bytes| {
            burn_triposplat::import::apply_triposplat_decoder_burnpack_part_bytes(model, part_bytes)
                .map_err(|err| format!("failed to apply TripoSplat decoder burnpack part: {err}"))
        },
    )
    .await?
    .ok_or_else(|| triposplat_missing_wasm_artifact_message(artifact, "TripoSplat decoder"))
}

async fn load_triposg_pipeline_wasm<B: Backend, F>(
    device: &B::Device,
    options: TripoWasmLoadOptions,
    load_ctx: &mut WasmLoadContext<'_, F>,
) -> Result<TripoSGPipeline<B>, String>
where
    F: FnMut(String),
{
    let root = wasm_model_root(ROOT_TRIPOSG);

    let vae_config_json = fetch_optional_text(&join_web_path(&root, "vae/config.json")).await?;
    let dit_config_json =
        fetch_optional_text(&join_web_path(&root, "transformer/config.json")).await?;
    let scheduler_config_json =
        fetch_optional_text(&join_web_path(&root, "scheduler/scheduler_config.json")).await?;
    let dino_config_candidates = DINO_CONFIG_RELPATHS
        .iter()
        .map(|rel| join_web_path(&root, rel))
        .collect::<Vec<_>>();
    let dino_config_json = fetch_optional_text_candidates(&dino_config_candidates).await?;

    let vae_config = if let Some(json) = vae_config_json.as_ref() {
        TripoSGVaeConfig::from_config_bytes(json.as_bytes())
            .map_err(|err| format!("failed to parse TripoSG VAE config: {err}"))?
    } else {
        TripoSGVaeConfig::midi_3d()
    };
    let dit_config = if let Some(json) = dit_config_json.as_ref() {
        TripoSGDiTConfig::from_config_bytes(json.as_bytes())
            .map_err(|err| format!("failed to parse TripoSG DiT config: {err}"))?
    } else {
        TripoSGDiTConfig::triposg_pretrained()
    };
    let scheduler_config = if let Some(json) = scheduler_config_json.as_ref() {
        RectifiedFlowSchedulerConfig::from_config_bytes(json.as_bytes())
            .map_err(|err| format!("failed to parse TripoSG scheduler config: {err}"))?
    } else {
        RectifiedFlowSchedulerConfig::midi_3d()
    };

    let parsed_dino_config = dino_config_json.as_ref().and_then(|json| {
        burn_tripo::model::triposg::image_encoder::import::load_dinov2_config_from_json_bytes(
            json.as_bytes(),
        )
    });
    let mut dino_config = parsed_dino_config
        .clone()
        .unwrap_or_else(default_dinov2_config);
    let dino_processor =
        default_wasm_dino_processor().with_strict_preprocess(options.strict_dino_preprocess);
    if let Some(target_size) =
        dino_processor_target_size(&dino_processor, Some(CANONICAL_DINO_CROP))
    {
        let patch = dino_config.patch_size.max(1);
        let grid = target_size / patch;
        if grid > 0 {
            dino_config.positional_encoding_interpolate.output_size = Some([grid, grid]);
        }
    }

    let vae_base_safetensors_url = join_web_path(&root, "vae/diffusion_pytorch_model.safetensors");
    let vae = if let Some(model) = try_load_model_from_parts_wasm(
        &vae_base_safetensors_url,
        "TripoSG VAE",
        options.prefer_f16_vae,
        !options.strict_precision,
        load_ctx,
        || TripoSGVae::new_decode_only(device, vae_config.clone()),
        |model, part_bytes| {
            apply_triposg_vae_decoder_burnpack_part_bytes(model, part_bytes).map_err(|err| {
                format!("failed to apply TripoSG VAE decoder burnpack part bytes: {err}")
            })
        },
    )
    .await?
    {
        model
    } else {
        return Err(format!(
            "TripoSG VAE wasm loader requires burnpack parts manifests under {root}/vae; missing *.bpk.parts.json for requested precision."
        ));
    };

    let dino_base_safetensors_url = join_web_path(&root, "image_encoder_dinov2/model.safetensors");
    let image_encoder = if let Some(model) = try_load_model_from_parts_wasm(
        &dino_base_safetensors_url,
        "DINOv2",
        options.prefer_f16_dino,
        !options.strict_precision,
        load_ctx,
        || init_triposg_dinov2_model(device, dino_config.clone()),
        |model: &mut TripoSGImageEncoder<B>, part_bytes| {
            apply_triposg_dinov2_burnpack_part_bytes(model, part_bytes)
                .map_err(|err| format!("failed to apply DINOv2 burnpack part bytes: {err}"))
        },
    )
    .await?
    {
        model
    } else {
        return Err(format!(
            "DINOv2 wasm loader requires burnpack parts manifests under {root}/image_encoder_dinov2; missing *.bpk.parts.json for requested precision."
        ));
    };

    // Keep wasm load peak bounded by loading model components incrementally (parts-first).
    // TripoSG DiT remains the largest component and is loaded last.
    let dit = load_triposg_dit_wasm(
        device,
        &root,
        &dit_config,
        options.prefer_f16_dit,
        !options.strict_precision,
        load_ctx,
    )
    .await?;

    let scheduler = scheduler_config.init();

    Ok(TripoSGPipeline::new_with_optional_image_encoder(
        vae,
        dit,
        scheduler,
        Some(image_encoder),
        dino_processor,
    ))
}

async fn load_triposg_dit_wasm<B: Backend, F>(
    device: &B::Device,
    root: &str,
    dit_config: &TripoSGDiTConfig,
    prefer_f16: bool,
    allow_cross_precision_fallback: bool,
    load_ctx: &mut WasmLoadContext<'_, F>,
) -> Result<TripoSGDiT<B>, String>
where
    F: FnMut(String),
{
    let base_safetensors_url = join_web_path(
        &join_web_path(root, "transformer"),
        "diffusion_pytorch_model.safetensors",
    );
    if let Some(model) = try_load_triposg_dit_from_parts_wasm(
        device,
        dit_config,
        &base_safetensors_url,
        prefer_f16,
        allow_cross_precision_fallback,
        load_ctx,
    )
    .await?
    {
        return Ok(model);
    }

    Err(format!(
        "TripoSG DiT wasm loader requires burnpack parts manifests under {root}/transformer; missing *.bpk.parts.json for requested precision."
    ))
}

async fn try_load_triposg_dit_from_parts_wasm<B: Backend, F>(
    device: &B::Device,
    dit_config: &TripoSGDiTConfig,
    base_safetensors_url: &str,
    prefer_f16: bool,
    allow_cross_precision_fallback: bool,
    load_ctx: &mut WasmLoadContext<'_, F>,
) -> Result<Option<TripoSGDiT<B>>, String>
where
    F: FnMut(String),
{
    try_load_model_from_parts_wasm(
        base_safetensors_url,
        "TripoSG DiT",
        prefer_f16,
        allow_cross_precision_fallback,
        load_ctx,
        || TripoSGDiT::new(device, dit_config.clone()),
        |model, part_bytes| {
            apply_triposg_dit_burnpack_part_bytes(model, part_bytes)
                .map_err(|err| format!("failed to apply TripoSG DiT burnpack part bytes: {err}"))
        },
    )
    .await
}

async fn try_load_model_from_parts_wasm<M, F, Init, Apply>(
    base_safetensors_url: &str,
    label: &str,
    prefer_f16: bool,
    allow_cross_precision_fallback: bool,
    load_ctx: &mut WasmLoadContext<'_, F>,
    init_model: Init,
    apply_part: Apply,
) -> Result<Option<M>, String>
where
    F: FnMut(String),
    Init: FnMut() -> M,
    Apply: FnMut(&mut M, Vec<u8>) -> Result<(), String>,
{
    let mut candidates = candidate_burnpack_names(base_safetensors_url, prefer_f16);
    if !allow_cross_precision_fallback {
        candidates.truncate(1);
    }
    try_load_model_from_burnpack_candidates_wasm(
        candidates, label, load_ctx, init_model, apply_part,
    )
    .await
}

#[cfg(feature = "wasm-api-wgpu")]
async fn try_load_triposplat_artifact_from_parts_wasm<M, F, Init, Apply>(
    artifact: TripoSplatArtifact,
    label: &str,
    prefer_f16: bool,
    allow_cross_precision_fallback: bool,
    load_ctx: &mut WasmLoadContext<'_, F>,
    init_model: Init,
    apply_part: Apply,
) -> Result<Option<M>, String>
where
    F: FnMut(String),
    Init: FnMut() -> M,
    Apply: FnMut(&mut M, Vec<u8>) -> Result<(), String>,
{
    let mut candidates = triposplat_burnpack_candidate_urls(artifact, prefer_f16);
    if !allow_cross_precision_fallback {
        candidates.truncate(1);
    }
    try_load_model_from_burnpack_candidates_wasm(
        candidates, label, load_ctx, init_model, apply_part,
    )
    .await
}

#[cfg(feature = "wasm-api-wgpu")]
async fn try_load_model_from_burnpack_candidates_wasm<M, F, Init, Apply>(
    candidates: Vec<String>,
    label: &str,
    load_ctx: &mut WasmLoadContext<'_, F>,
    mut init_model: Init,
    mut apply_part: Apply,
) -> Result<Option<M>, String>
where
    F: FnMut(String),
    Init: FnMut() -> M,
    Apply: FnMut(&mut M, Vec<u8>) -> Result<(), String>,
{
    let max_bytes = web_max_burnpack_bytes();
    for candidate in candidates {
        let manifest_url = format!("{candidate}.parts.json");
        load_ctx.status(format!("Checking {label} manifest {manifest_url}..."));
        let Some(manifest_text) = fetch_optional_text(&manifest_url).await? else {
            load_ctx.status(format!(
                "Missing {label} manifest {manifest_url}; trying next candidate."
            ));
            continue;
        };
        load_ctx.status(format!("Found {label} manifest {manifest_url}."));
        let manifest = parse_parts_manifest_bytes(manifest_text.as_bytes(), &manifest_url)?;
        if manifest.parts.is_empty() {
            return Err(format!(
                "burnpack parts manifest {manifest_url} contains no parts"
            ));
        }

        load_ctx.status(format!(
            "Loading {label} from {} burnpack parts...",
            manifest.parts.len()
        ));
        let mut model = init_model();
        for (index, part) in manifest.parts.iter().enumerate() {
            let part_url = resolve_manifest_entry_uri(&manifest_url, &part.path);
            let part_label = format!("{label} part {}/{}", index + 1, manifest.parts.len());
            let bytes = download_binary_with_status(
                &part_url,
                &part_label,
                max_bytes,
                load_ctx.totals,
                load_ctx.host_ram_budget,
                load_ctx.on_status,
            )
            .await?;
            if part.bytes > 0 && bytes.len() as u64 != part.bytes {
                return Err(format!(
                    "{label} part {} expected {} bytes but downloaded {} bytes",
                    part.path,
                    part.bytes,
                    bytes.len()
                ));
            }
            let verify_part_checksum = should_verify_wasm_part_checksums();
            if verify_part_checksum && !part.sha256.trim().is_empty() {
                load_ctx.status(format!(
                    "Verifying checksum for {label} part {}/{}...",
                    index + 1,
                    manifest.parts.len()
                ));
                let actual_sha = sha256_hex(&bytes);
                if !actual_sha.eq_ignore_ascii_case(part.sha256.trim()) {
                    return Err(format!(
                        "{label} part {} checksum mismatch: expected {}, got {}",
                        part.path,
                        part.sha256.trim(),
                        actual_sha
                    ));
                }
                load_ctx.status(format!(
                    "Verified checksum for {label} part {}/{}",
                    index + 1,
                    manifest.parts.len()
                ));
            } else if !verify_part_checksum {
                load_ctx.status(format!(
                    "Skipping checksum verification for {label} part {}/{} in release wasm build",
                    index + 1,
                    manifest.parts.len()
                ));
            }
            load_ctx.status(format!(
                "Applying {label} part {}/{}...",
                index + 1,
                manifest.parts.len()
            ));
            apply_part(&mut model, bytes)?;
            load_ctx.status(format!(
                "Applied {label} part {}/{}",
                index + 1,
                manifest.parts.len()
            ));
        }

        return Ok(Some(model));
    }

    Ok(None)
}

#[cfg(feature = "wasm-api-wgpu")]
fn triposplat_required_artifact(stem: &str) -> Result<TripoSplatArtifact, String> {
    TRIPOSPLAT_ARTIFACTS
        .into_iter()
        .find(|artifact| artifact.burnpack_stem == stem)
        .ok_or_else(|| format!("missing TripoSplat artifact metadata for {stem}"))
}

#[cfg(feature = "wasm-api-wgpu")]
fn triposplat_burnpack_candidate_urls(
    artifact: TripoSplatArtifact,
    prefer_f16: bool,
) -> Vec<String> {
    let root = wasm_model_root(ROOT_TRIPOSPLAT);
    let component_root = join_web_path(&root, artifact.component);
    let f32 = join_web_path(&component_root, &format!("{}.bpk", artifact.burnpack_stem));
    let f16 = join_web_path(
        &component_root,
        &format!("{}_f16.bpk", artifact.burnpack_stem),
    );
    if prefer_f16 {
        vec![f16, f32]
    } else {
        vec![f32, f16]
    }
}

#[cfg(feature = "wasm-api-wgpu")]
fn triposplat_missing_wasm_artifact_message(artifact: TripoSplatArtifact, label: &str) -> String {
    let checked = triposplat_burnpack_candidate_urls(artifact, true)
        .into_iter()
        .map(|candidate| format!("{candidate}.parts.json"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("{label} wasm loader requires burnpack parts manifests; checked {checked}")
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn should_verify_wasm_part_checksums() -> bool {
    cfg!(debug_assertions)
}

fn prepare_image_config_for_backend<B: Backend>() -> PrepareImageConfig {
    let _ = std::any::type_name::<B>();
    PrepareImageConfig::default()
}

fn encode_image_embeds_for_wasm<BTriposg: Backend, BRmbg: Backend>(
    state: &WasmPipelineState<BTriposg, BRmbg>,
    prepared: &PreparedImageData,
) -> Result<Tensor<BTriposg, 3>, String> {
    let processed = if state.triposg.image_processor.is_strict_preprocess() {
        let cpu_device = <NdArray<f32> as BackendTypes>::Device::default();
        let cpu_image = prepared.to_tensor::<NdArray<f32>>(&cpu_device);
        let cpu_processed = state.triposg.image_processor.preprocess(cpu_image);
        convert_image_to_backend::<BTriposg>(cpu_processed, &state.triposg_device)?
    } else {
        let image = prepared.to_tensor::<BTriposg>(&state.triposg_device);
        state.triposg.image_processor.preprocess(image)
    };
    state
        .triposg
        .image_encoder
        .as_ref()
        .ok_or_else(|| "TripoSG image encoder is unavailable".to_string())
        .map(|encoder| encoder.forward(processed))
}

fn convert_image_to_backend<B: Backend>(
    image: Tensor<NdArray<f32>, 4>,
    device: &B::Device,
) -> Result<Tensor<B, 4>, String> {
    let shape = image.shape().dims::<4>();
    let data = image
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|err| format!("failed to read CPU image tensor: {err:?}"))?;
    let flat = Tensor::<B, 1>::from_floats(data.as_slice(), device);
    Ok(flat.reshape([
        shape[0] as i32,
        shape[1] as i32,
        shape[2] as i32,
        shape[3] as i32,
    ]))
}

fn dino_processor_target_size(
    processor: &DinoImageProcessor,
    fallback_size: Option<usize>,
) -> Option<usize> {
    processor
        .crop_size
        .map(|[height, width]| height.min(width))
        .or(processor.size_shortest_edge)
        .or(fallback_size)
        .filter(|size| *size > 0)
}

fn default_wasm_dino_processor() -> DinoImageProcessor {
    DinoImageProcessor {
        do_resize: true,
        size_shortest_edge: Some(CANONICAL_DINO_SHORT_EDGE),
        do_center_crop: true,
        crop_size: Some([CANONICAL_DINO_CROP, CANONICAL_DINO_CROP]),
        ..DinoImageProcessor::default()
    }
}

fn tripo_mesh_to_mesh(mesh: TripoMesh) -> Mesh {
    Mesh {
        vertices: mesh.vertices,
        faces: mesh.faces,
        uvs: Vec::new(),
        material: None,
        pbr_textures: None,
    }
}

fn is_wgpu_backend<B: Backend>() -> bool {
    std::any::type_name::<B>()
        .to_ascii_lowercase()
        .contains("wgpu")
}

fn backend_uses_f16<B: Backend>() -> bool {
    std::any::type_name::<B>()
        .to_ascii_lowercase()
        .contains("f16")
}

fn wasm_model_root(rel_root: &str) -> String {
    join_web_path(&resolve_wasm_model_base_url(), rel_root)
}

fn resolve_wasm_model_base_url() -> String {
    if let Some(value) = wasm_query_param_value("model_base_url")
        .or_else(|| wasm_query_param_value("model_base"))
        .or_else(|| wasm_query_param_value("model_root"))
        .or_else(|| wasm_query_param_value("model_url"))
    {
        return value;
    }

    if let Some(value) = wasm_query_param_value("model_source")
        .or_else(|| wasm_query_param_value("models"))
        .or_else(|| wasm_query_param_value("model_origin"))
    {
        match value.trim().to_ascii_lowercase().as_str() {
            "local" | "assets" | "bundle" | "bundled" => {
                return DEFAULT_LOCAL_MODEL_BASE_URL.to_string();
            }
            "cdn" | "cloud" | "remote" => {
                return DEFAULT_MODEL_BASE_URL.to_string();
            }
            _ => {}
        }
    }

    if let Some(value) = option_env!("MODEL_BASE_URL")
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .or_else(|| {
            option_env!("BURN_SYNTH_WEB_ASSET_ROOT")
                .map(str::trim)
                .filter(|value| !value.is_empty())
        })
    {
        return value.to_string();
    }

    let (protocol, hostname) = wasm_location_protocol_and_hostname();
    if is_local_dev_host(protocol.as_deref(), hostname.as_deref()) {
        return DEFAULT_LOCAL_MODEL_BASE_URL.to_string();
    }

    DEFAULT_MODEL_BASE_URL.to_string()
}

fn wasm_query_param_value(key: &str) -> Option<String> {
    let search = wasm_location_search()?;
    query_param_value(&search, key)
}

fn query_param_value(search: &str, key: &str) -> Option<String> {
    for pair in search.trim_start_matches('?').split('&') {
        if pair.is_empty() {
            continue;
        }
        let mut parts = pair.splitn(2, '=');
        let Some(raw_key) = parts.next() else {
            continue;
        };
        if !raw_key.eq_ignore_ascii_case(key) {
            continue;
        }
        let value = parts.next().unwrap_or_default().trim();
        if !value.is_empty() {
            return Some(value.to_string());
        }
    }
    None
}

fn wasm_location_search() -> Option<String> {
    wasm_window_location_field("search")
}

fn wasm_location_protocol_and_hostname() -> (Option<String>, Option<String>) {
    (
        wasm_window_location_field("protocol"),
        wasm_window_location_field("hostname"),
    )
}

fn wasm_window_location_field(field: &str) -> Option<String> {
    let window = web_sys::window()?;
    let window_js: wasm_bindgen::JsValue = window.into();
    let location = Reflect::get(&window_js, &wasm_bindgen::JsValue::from_str("location")).ok()?;
    let value = Reflect::get(&location, &wasm_bindgen::JsValue::from_str(field)).ok()?;
    value.as_string()
}

fn is_local_dev_host(protocol: Option<&str>, hostname: Option<&str>) -> bool {
    let protocol = protocol.unwrap_or_default().trim().to_ascii_lowercase();
    let host = hostname.unwrap_or_default().trim().to_ascii_lowercase();
    if protocol == "file:" {
        return true;
    }
    if host.is_empty() {
        return false;
    }
    if matches!(
        host.as_str(),
        "localhost" | "127.0.0.1" | "0.0.0.0" | "::1" | "[::1]"
    ) {
        return true;
    }
    host.ends_with(".localhost") || host.ends_with(".local")
}

#[cfg(feature = "wasm-api-wgpu")]
async fn wasm_webgpu_available() -> bool {
    wasm_webgpu_adapter_profile().await.is_some()
}

#[cfg(feature = "wasm-api-wgpu")]
async fn wasm_webgpu_adapter_profile() -> Option<WebGpuAdapterProfile> {
    let adapter = wasm_webgpu_request_adapter().await?;
    Some(webgpu_adapter_profile_from_value(adapter))
}

#[cfg(feature = "wasm-api-wgpu")]
fn webgpu_adapter_profile_from_value(adapter: wasm_bindgen::JsValue) -> WebGpuAdapterProfile {
    let shader_f16_supported = webgpu_adapter_supports_feature(&adapter, "shader-f16");
    WebGpuAdapterProfile {
        shader_f16_supported,
    }
}

#[cfg(feature = "wasm-api-wgpu")]
fn webgpu_adapter_supports_feature(adapter: &wasm_bindgen::JsValue, feature_name: &str) -> bool {
    let features = match Reflect::get(adapter, &wasm_bindgen::JsValue::from_str("features")) {
        Ok(value) if !value.is_null() && !value.is_undefined() => value,
        _ => return false,
    };
    let has_method = match Reflect::get(&features, &wasm_bindgen::JsValue::from_str("has")) {
        Ok(value) => value,
        Err(_) => return false,
    };
    let has_method = match has_method.dyn_into::<Function>() {
        Ok(func) => func,
        Err(_) => return false,
    };
    match has_method.call1(&features, &wasm_bindgen::JsValue::from_str(feature_name)) {
        Ok(value) => value.as_bool().unwrap_or(false),
        Err(_) => false,
    }
}

#[cfg(feature = "wasm-api-wgpu")]
async fn wasm_webgpu_request_adapter() -> Option<wasm_bindgen::JsValue> {
    let window = web_sys::window()?;
    let window_js: wasm_bindgen::JsValue = window.into();
    let navigator = match Reflect::get(&window_js, &wasm_bindgen::JsValue::from_str("navigator")) {
        Ok(value) if !value.is_undefined() && !value.is_null() => value,
        _ => return None,
    };
    let gpu = match Reflect::get(&navigator, &wasm_bindgen::JsValue::from_str("gpu")) {
        Ok(value) if !value.is_undefined() && !value.is_null() => value,
        _ => return None,
    };
    let request_adapter =
        match Reflect::get(&gpu, &wasm_bindgen::JsValue::from_str("requestAdapter")) {
            Ok(value) => value,
            Err(_) => return None,
        };
    let request_adapter = match request_adapter.dyn_into::<Function>() {
        Ok(func) => func,
        Err(_) => return None,
    };
    let promise = match request_adapter.call0(&gpu) {
        Ok(value) => value,
        Err(_) => return None,
    };
    let promise = match promise.dyn_into::<Promise>() {
        Ok(promise) => promise,
        Err(_) => return None,
    };
    match JsFuture::from(promise).await {
        Ok(adapter) if !adapter.is_null() && !adapter.is_undefined() => Some(adapter),
        _ => None,
    }
}

#[cfg(feature = "wasm-api-wgpu")]
async fn initialize_wgpu_runtime_for_wasm() -> Result<(), String> {
    static INIT_DONE: AtomicBool = AtomicBool::new(false);
    if INIT_DONE.load(Ordering::Acquire) {
        return Ok(());
    }
    let device = burn_wgpu::WgpuDevice::default();
    // Keep wasm runtime memory pressure bounded for mixed render+compute workloads
    // (for example bevy_synth WebGL rendering + Burn WebGPU inference in one tab).
    // RuntimeOptions::default() keeps CubeCL's default memory policy (SubSlices) which
    // is materially lower-footprint than forced ExclusivePages on web.
    let options = burn_wgpu::RuntimeOptions {
        tasks_max: 8,
        ..burn_wgpu::RuntimeOptions::default()
    };
    burn_wgpu::init_setup_async::<burn_wgpu::graphics::WebGpu>(&device, options).await;
    INIT_DONE.store(true, Ordering::Release);
    Ok(())
}

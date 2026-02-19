#![cfg(all(target_arch = "wasm32", feature = "wasm-api"))]

use std::cell::RefCell;
use std::sync::Once;
#[cfg(feature = "wasm-api-wgpu")]
use std::sync::atomic::{AtomicBool, Ordering};

use burn::backend::NdArray;
use burn::prelude::*;
use burn_foreground::pipeline::{
    PrepareImageConfig, PreparedImageData, RmbgPipeline, prepare_image_data_from_bytes_async,
};
use burn_foreground::rmbg14::BriaRmbg;
use burn_foreground::rmbg14::import::{
    apply_rmbg_burnpack_part_bytes, load_rmbg_config_from_json_bytes,
};
use burn_foreground::rmbg14::set_rmbg_strict_interp_override;
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
use js_sys::Uint8Array;
#[cfg(feature = "wasm-api-wgpu")]
use js_sys::{Function, Promise, Reflect};
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
use crate::wasm::WasmInferencePreset;
use crate::wasm_loader::{
    DownloadTotals, WasmHostMemoryBudget, download_binary_with_status, fetch_optional_text,
    fetch_optional_text_candidates, join_web_path, web_max_burnpack_bytes, web_max_host_ram_bytes,
};

#[cfg(feature = "wasm-api-wgpu")]
type WgpuBackendF16 = burn_wgpu::Wgpu<burn::tensor::f16, i32, u32>;
#[cfg(feature = "wasm-api-wgpu")]
type WgpuBackendF32 = burn_wgpu::Wgpu<f32, i32, u32>;
#[cfg(feature = "wasm-api-wgpu")]
type WgpuRmbgBackend = burn_wgpu::Wgpu<f32, i32, u32>;

const DEFAULT_GUIDANCE_SCALE: f32 = 7.0;
const DEFAULT_BOUNDS: [f32; 6] = [-1.005, -1.005, -1.005, 1.005, 1.005, 1.005];
const DEFAULT_MODEL_BASE_URL: &str = "https://aberration.technology/model";
const DINO_CONFIG_RELPATHS: [&str; 2] = [
    "image_encoder_dinov2/config.json",
    "image_encoder_2/config.json",
];
const ROOT_TRIPOSG: &str = "models/MIDI-3D";
const ROOT_RMBG14: &str = "models/RMBG-1.4";
const DEFAULT_FLASH_OCTREE_DEPTH: usize = 9;
const DEFAULT_FLASH_NUM_CHUNKS: usize = 10_000;
const DEFAULT_FLASH_MINI_GRID_NUM: usize = 4;
const CANONICAL_DINO_SHORT_EDGE: usize = 256;
const CANONICAL_DINO_CROP: usize = 224;

static PANIC_HOOK_ONCE: Once = Once::new();

struct WasmPipelineState<BTriposg: Backend, BRmbg: Backend> {
    triposg_device: BTriposg::Device,
    rmbg: RmbgPipeline<BRmbg>,
    triposg: TripoSGPipeline<BTriposg>,
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
thread_local! {
    static CACHED_WASM_PIPELINE: RefCell<Option<CachedWasmPipeline>> = const { RefCell::new(None) };
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

#[wasm_bindgen]
#[derive(Clone, Debug, Default)]
pub struct WasmInferOptions {
    num_steps: u32,
    num_tokens: u32,
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

    pub fn set_num_tokens(&mut self, value: u32) {
        self.num_tokens = value;
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
            num_steps: preset.num_steps as u32,
            num_tokens: preset.num_tokens as u32,
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
        if self.num_steps > 0 {
            preset.num_steps = self.num_steps as usize;
        }
        if self.num_tokens > 0 {
            preset.num_tokens = self.num_tokens as usize;
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
) -> Result<&'static str, String> {
    if preset.weights_precision.eq_ignore_ascii_case("f16") {
        return Ok(if shader_f16_supported { "f16" } else { "f32" });
    }
    if preset.weights_precision.eq_ignore_ascii_case("auto") {
        return Ok(if shader_f16_supported { "f16" } else { "f32" });
    }
    Ok("f32")
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
    mut on_status: F,
) -> Result<(), String>
where
    F: FnMut(String),
{
    if !preset.backend.eq_ignore_ascii_case("wgpu") {
        return Err("wasm TripoSG supports backend=wgpu only".to_string());
    }
    if !wasm_webgpu_available().await {
        return Err(
            "WebGPU is unavailable in this browser/runtime; CPU fallback is disabled for TripoSG wasm."
                .to_string(),
        );
    }
    initialize_wgpu_runtime_for_wasm().await?;
    let shader_f16_supported = wasm_webgpu_shader_f16_supported().await;
    let precision = resolve_wgpu_precision_for_preset(preset, shader_f16_supported)?;
    if preset.weights_precision.eq_ignore_ascii_case("f16") && !shader_f16_supported {
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
    if image_bytes.is_empty() {
        return Err("image bytes are empty".to_string());
    }
    warmup_pipeline_for_preset(preset).await?;

    let mut cached = CACHED_WASM_PIPELINE.with(|cache| cache.borrow_mut().take());
    let result = match cached.as_mut() {
        Some(CachedWasmPipeline::WgpuF32 {
            preset: cached_preset,
            state,
        }) if cached_preset == preset => run_inference_once(state, image_bytes, preset).await,
        Some(CachedWasmPipeline::WgpuF16 {
            preset: cached_preset,
            state,
        }) if cached_preset == preset => run_inference_once(state, image_bytes, preset).await,
        Some(_) => Err("cached wasm pipeline preset mismatch".to_string()),
        None => Err("cached wasm pipeline unavailable after warmup".to_string()),
    };
    CACHED_WASM_PIPELINE.with(|cache| {
        *cache.borrow_mut() = cached;
    });
    result
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
    let bytes = infer_glb_from_image_bytes_with_preset_cached(image_bytes.as_slice(), &preset)
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

async fn run_inference_once<BTriposg: Backend, BRmbg: Backend>(
    state: &mut WasmPipelineState<BTriposg, BRmbg>,
    image_bytes: &[u8],
    preset: &WasmInferencePreset,
) -> Result<Vec<u8>, String> {
    web_sys::console::log_1(&"burn_synth wasm infer: prepare_image_data start".into());
    let prepared = prepare_image_data_from_bytes_async::<BRmbg>(
        image_bytes,
        Some(&state.rmbg),
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

    let flash = FlashExtractConfig {
        bounds: DEFAULT_BOUNDS,
        octree_depth: DEFAULT_FLASH_OCTREE_DEPTH,
        num_chunks: DEFAULT_FLASH_NUM_CHUNKS,
        mc_level: 0.0,
        min_resolution: preset.resolution.max(2),
        mini_grid_num: DEFAULT_FLASH_MINI_GRID_NUM,
    };
    web_sys::console::log_1(
        &format!(
            "burn_synth wasm infer: flash_extract start (steps={} tokens={} min_resolution={} faces={})",
            preset.num_steps.max(1),
            preset.num_tokens.max(64),
            flash.min_resolution,
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
    let prefer_f16_rmbg = match requested_rmbg_precision {
        "f16" => true,
        "f32" => false,
        // Favor f16 by default on wasm Bevy runtimes to reduce startup pressure
        // while render resources are resident.
        _ => true,
    };
    let allow_cross_precision_rmbg = requested_rmbg_precision == "auto";
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

    let rmbg = load_rmbg14_pipeline_wasm(
        &rmbg_device,
        prefer_f16_rmbg,
        allow_cross_precision_rmbg,
        &mut load_ctx,
    )
    .await?;
    let triposg = load_triposg_pipeline_wasm(&triposg_device, options, &mut load_ctx).await?;

    Ok(WasmPipelineState {
        triposg_device,
        rmbg,
        triposg,
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
    mut init_model: Init,
    mut apply_part: Apply,
) -> Result<Option<M>, String>
where
    F: FnMut(String),
    Init: FnMut() -> M,
    Apply: FnMut(&mut M, Vec<u8>) -> Result<(), String>,
{
    let max_bytes = web_max_burnpack_bytes();
    let mut candidates = candidate_burnpack_names(base_safetensors_url, prefer_f16);
    if !allow_cross_precision_fallback {
        candidates.truncate(1);
    }
    for candidate in candidates {
        let manifest_url = format!("{candidate}.parts.json");
        let Some(manifest_text) = fetch_optional_text(&manifest_url).await? else {
            continue;
        };
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
        let cpu_device = <NdArray<f32> as Backend>::Device::default();
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
    let root = option_env!("MODEL_BASE_URL")
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .or_else(|| {
            option_env!("BURN_SYNTH_WEB_ASSET_ROOT")
                .map(str::trim)
                .filter(|value| !value.is_empty())
        })
        .unwrap_or(DEFAULT_MODEL_BASE_URL);
    join_web_path(root, rel_root)
}

#[cfg(feature = "wasm-api-wgpu")]
async fn wasm_webgpu_available() -> bool {
    wasm_webgpu_request_adapter().await.is_some()
}

#[cfg(feature = "wasm-api-wgpu")]
async fn wasm_webgpu_shader_f16_supported() -> bool {
    let Some(adapter) = wasm_webgpu_request_adapter().await else {
        return false;
    };
    let features = match Reflect::get(&adapter, &wasm_bindgen::JsValue::from_str("features")) {
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
    match has_method.call1(&features, &wasm_bindgen::JsValue::from_str("shader-f16")) {
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

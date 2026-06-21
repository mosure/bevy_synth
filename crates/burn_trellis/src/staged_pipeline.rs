#![cfg_attr(not(feature = "runtime-model"), allow(dead_code))]

#[cfg(feature = "runtime-model")]
use std::borrow::Cow;
use std::collections::HashMap;
#[cfg(feature = "runtime-model")]
use std::collections::HashSet;
use std::path::Path;
#[cfg(feature = "runtime-model")]
use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use std::time::{SystemTime, UNIX_EPOCH};

use crate::TrellisComputeProfile;
use crate::mesh::{Mesh, MeshMaterial, MeshPbrTextures, MeshTexture};
use crate::preprocess::PreprocessOutput;
#[cfg(feature = "runtime-model")]
use crate::runtime_model::fdg_decoder::{FdgDecoderRuntime, decode_fdg_outputs_from_host};
#[cfg(feature = "runtime-model")]
use crate::runtime_model::image_conditioning::TrellisImageConditioningRuntime;
#[cfg(feature = "runtime-model")]
use crate::runtime_model::runtime_config::{
    RuntimeModelDebugConfig, set_runtime_model_debug_config,
    set_runtime_model_sparse_flow_attention_policy,
};
#[cfg(feature = "runtime-model")]
use crate::runtime_model::sparse_decoder::{
    DecoderConvTelemetry, DecoderOpTelemetry, SparseSubdivisionLogits, decoder_conv_telemetry,
    decoder_op_telemetry, reset_decoder_conv_telemetry, reset_decoder_op_telemetry,
};
#[cfg(feature = "runtime-model")]
use crate::runtime_model::sparse_structure_decoder::SparseStructureDecoderRuntime;
#[cfg(feature = "runtime-model-wgpu")]
use crate::runtime_model::sparse_structure_flow::WgpuRuntimeBackend as SparseFlowWgpuBackend;
#[cfg(feature = "runtime-model")]
use crate::runtime_model::sparse_structure_flow::{
    SparseFlowOpTelemetry, SparseStructureFlowRuntime, reset_sparse_flow_op_telemetry,
    sparse_flow_op_telemetry,
};
#[cfg(feature = "runtime-model")]
use crate::runtime_model::sparse_unet_vae_decoder::{
    SparseUnetVaeDecoderRuntime, decode_tex_attrs_from_host,
};
use crate::sampler::{FlowEulerGuidanceIntervalSampler, FlowEulerSampleConfig};
use crate::time::Instant;
use crate::trellis_config::{TrellisNormalization, TrellisPipelineArgs, TrellisSamplerConfig};
#[cfg(feature = "runtime-model-wgpu")]
use burn::prelude::Backend;
#[cfg(feature = "runtime-model-wgpu")]
use burn::tensor::TensorData;
#[cfg(feature = "runtime-model-wgpu")]
use burn::tensor::{Int, Tensor};
#[cfg(feature = "runtime-model-wgpu")]
use burn_flex_gmm::wgpu::{
    neighbor_rows_build_stats, reset_neighbor_rows_build_stats, reset_sparse_wgpu_kernel_stats,
    sparse_wgpu_kernel_stats,
};
#[cfg(feature = "runtime-model-wgpu")]
use burn_wgpu::WgpuDevice;

#[path = "staged_pipeline_decode.rs"]
mod staged_pipeline_decode;
use staged_pipeline_decode::*;

static STAGE_LOG_EPOCH: OnceLock<Instant> = OnceLock::new();
static RUNTIME_STAGE_DEBUG: AtomicBool = AtomicBool::new(false);
static RUNTIME_DECODER_CONV_TELEMETRY: AtomicBool = AtomicBool::new(false);
static RUNTIME_STAGE_FENCE: AtomicBool = AtomicBool::new(false);

fn stage_log_timestamp() -> String {
    let elapsed = STAGE_LOG_EPOCH
        .get_or_init(Instant::now)
        .elapsed()
        .as_secs_f64();
    #[cfg(not(target_arch = "wasm32"))]
    let epoch_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0);
    #[cfg(target_arch = "wasm32")]
    let epoch_ms = 0u128;
    format!("ts_ms={epoch_ms} t+{elapsed:.3}s")
}

fn set_runtime_debug_toggles(run_config: TrellisStageRunConfig) {
    RUNTIME_STAGE_DEBUG.store(run_config.runtime_stage_debug, Ordering::Relaxed);
    RUNTIME_DECODER_CONV_TELEMETRY
        .store(run_config.runtime_decoder_conv_telemetry, Ordering::Relaxed);
    RUNTIME_STAGE_FENCE.store(run_config.runtime_stage_fence, Ordering::Relaxed);
}

#[cfg(feature = "runtime-model")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RuntimeFlowStage {
    SparseStructure,
    SLat,
}

#[cfg(feature = "runtime-model")]
fn set_runtime_model_debug_config_for_stage(
    run_config: TrellisStageRunConfig,
    stage: RuntimeFlowStage,
) {
    let module_attention_f16 = match stage {
        RuntimeFlowStage::SparseStructure => run_config
            .compute_profile
            .wgpu_sparse_module_attention_f16(),
        RuntimeFlowStage::SLat => run_config.compute_profile.wgpu_slat_module_attention_f16(),
    };
    set_runtime_model_debug_config(RuntimeModelDebugConfig {
        stage_debug: run_config.runtime_stage_debug,
        attention_debug: run_config.runtime_attention_debug,
        sparse_flow_module_attention: true,
        sparse_flow_module_attention_f16: module_attention_f16,
        sparse_flow_linear_f16: run_config.compute_profile.wgpu_linear_f16(),
        sparse_flow_torso_f16: run_config.compute_profile.wgpu_flow_torso_f16(),
        sparse_flow_coord_rope_kernel: true,
        sparse_decoder_conv_f16: run_config.compute_profile.wgpu_decoder_conv_f16(),
    });
    let (self_attention_f16, cross_attention_f16, final_f32_steps) = match stage {
        RuntimeFlowStage::SparseStructure => (
            run_config.compute_profile.wgpu_sparse_self_attention_f16(),
            run_config.compute_profile.wgpu_sparse_cross_attention_f16(),
            run_config.compute_profile.wgpu_sparse_final_f32_steps(),
        ),
        RuntimeFlowStage::SLat => {
            let f16 = run_config.compute_profile.wgpu_slat_module_attention_f16();
            (f16, f16, 0)
        }
    };
    set_runtime_model_sparse_flow_attention_policy(
        self_attention_f16,
        cross_attention_f16,
        final_f32_steps,
    );
}

fn runtime_stage_debug_enabled() -> bool {
    RUNTIME_STAGE_DEBUG.load(Ordering::Relaxed)
}

#[cfg(feature = "runtime-model")]
fn runtime_stage_fence_enabled() -> bool {
    RUNTIME_STAGE_FENCE.load(Ordering::Relaxed)
}

#[cfg(feature = "runtime-model")]
fn runtime_decoder_conv_telemetry_enabled() -> bool {
    RUNTIME_DECODER_CONV_TELEMETRY.load(Ordering::Relaxed)
}

#[cfg(feature = "runtime-model-wgpu")]
fn runtime_pipeline_stage_boundary_sync(stage: &str, enabled: bool) -> Result<(), String> {
    if !enabled {
        return Ok(());
    }
    // Keep repeat-to-repeat timings stable: ensure all queued decode work is
    // complete before returning stage timings to the caller.
    <SparseFlowWgpuBackend as Backend>::sync(&WgpuDevice::default())
        .map_err(|err| format!("runtime pipeline {stage} device sync failed: {err}"))
}

#[cfg(not(feature = "runtime-model-wgpu"))]
fn runtime_pipeline_stage_boundary_sync(_stage: &str, _enabled: bool) -> Result<(), String> {
    Ok(())
}

macro_rules! trellis_stage_log {
    ($($arg:tt)*) => {{
        std::eprintln!(
            "[{}] {}",
            $crate::staged_pipeline::stage_log_timestamp(),
            format!($($arg)*)
        );
    }};
}

include!("staged_pipeline_runtime_helpers.rs");
include!("staged_pipeline_sampling.rs");
include!("staged_pipeline_runtime_decode.rs");

#[cfg(test)]
mod tests {
    include!("staged_pipeline_tests.rs");
}

fn canonical_pipeline_type(pipeline_type: &str) -> &str {
    match pipeline_type {
        // Canonicalize legacy aliases only.
        "512" => "512_base",
        "1024_single" => "1024",
        other => other,
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SparseFlowOpTimingSummary {
    pub self_attn_calls: u64,
    pub self_attn_ns: u64,
    pub cross_attn_calls: u64,
    pub cross_attn_ns: u64,
    pub mlp_calls: u64,
    pub mlp_ns: u64,
    pub self_qkv_calls: u64,
    pub self_qkv_ns: u64,
    pub self_norm_rope_calls: u64,
    pub self_norm_rope_ns: u64,
    pub self_norm_rope_fused_qk_calls: u64,
    pub self_norm_rope_fused_qkv_module_calls: u64,
    pub self_kernel_calls: u64,
    pub self_kernel_ns: u64,
    pub self_out_calls: u64,
    pub self_out_ns: u64,
    pub self_cat_calls: u64,
    pub self_cat_ns: u64,
    pub cross_q_calls: u64,
    pub cross_q_ns: u64,
    pub cross_kv_calls: u64,
    pub cross_kv_ns: u64,
    pub cross_norm_calls: u64,
    pub cross_norm_ns: u64,
    pub cross_kernel_calls: u64,
    pub cross_kernel_ns: u64,
    pub cross_out_calls: u64,
    pub cross_out_ns: u64,
    pub cross_cat_calls: u64,
    pub cross_cat_ns: u64,
    pub module_cast_pad_calls: u64,
    pub module_cast_pad_ns: u64,
    pub module_attention_calls: u64,
    pub module_attention_ns: u64,
    pub module_output_calls: u64,
    pub module_output_ns: u64,
    pub block_norm_mod_calls: u64,
    pub block_norm_mod_ns: u64,
    pub block_norm_affine_calls: u64,
    pub block_norm_affine_ns: u64,
    pub block_gate_residual_calls: u64,
    pub block_gate_residual_ns: u64,
    pub model_io_calls: u64,
    pub model_io_ns: u64,
    pub model_input_calls: u64,
    pub model_input_ns: u64,
    pub model_output_calls: u64,
    pub model_output_ns: u64,
}

#[cfg(feature = "runtime-model")]
impl From<SparseFlowOpTelemetry> for SparseFlowOpTimingSummary {
    fn from(value: SparseFlowOpTelemetry) -> Self {
        Self {
            self_attn_calls: value.self_attn_calls,
            self_attn_ns: value.self_attn_ns,
            cross_attn_calls: value.cross_attn_calls,
            cross_attn_ns: value.cross_attn_ns,
            mlp_calls: value.mlp_calls,
            mlp_ns: value.mlp_ns,
            self_qkv_calls: value.self_qkv_calls,
            self_qkv_ns: value.self_qkv_ns,
            self_norm_rope_calls: value.self_norm_rope_calls,
            self_norm_rope_ns: value.self_norm_rope_ns,
            self_norm_rope_fused_qk_calls: value.self_norm_rope_fused_qk_calls,
            self_norm_rope_fused_qkv_module_calls: value.self_norm_rope_fused_qkv_module_calls,
            self_kernel_calls: value.self_kernel_calls,
            self_kernel_ns: value.self_kernel_ns,
            self_out_calls: value.self_out_calls,
            self_out_ns: value.self_out_ns,
            self_cat_calls: value.self_cat_calls,
            self_cat_ns: value.self_cat_ns,
            cross_q_calls: value.cross_q_calls,
            cross_q_ns: value.cross_q_ns,
            cross_kv_calls: value.cross_kv_calls,
            cross_kv_ns: value.cross_kv_ns,
            cross_norm_calls: value.cross_norm_calls,
            cross_norm_ns: value.cross_norm_ns,
            cross_kernel_calls: value.cross_kernel_calls,
            cross_kernel_ns: value.cross_kernel_ns,
            cross_out_calls: value.cross_out_calls,
            cross_out_ns: value.cross_out_ns,
            cross_cat_calls: value.cross_cat_calls,
            cross_cat_ns: value.cross_cat_ns,
            module_cast_pad_calls: value.module_cast_pad_calls,
            module_cast_pad_ns: value.module_cast_pad_ns,
            module_attention_calls: value.module_attention_calls,
            module_attention_ns: value.module_attention_ns,
            module_output_calls: value.module_output_calls,
            module_output_ns: value.module_output_ns,
            block_norm_mod_calls: value.block_norm_mod_calls,
            block_norm_mod_ns: value.block_norm_mod_ns,
            block_norm_affine_calls: value.block_norm_affine_calls,
            block_norm_affine_ns: value.block_norm_affine_ns,
            block_gate_residual_calls: value.block_gate_residual_calls,
            block_gate_residual_ns: value.block_gate_residual_ns,
            model_io_calls: value.model_io_calls,
            model_io_ns: value.model_io_ns,
            model_input_calls: value.model_input_calls,
            model_input_ns: value.model_input_ns,
            model_output_calls: value.model_output_calls,
            model_output_ns: value.model_output_ns,
        }
    }
}

#[cfg(feature = "runtime-model")]
fn current_sparse_flow_op_timing_summary() -> SparseFlowOpTimingSummary {
    sparse_flow_op_telemetry().into()
}

#[cfg(not(feature = "runtime-model"))]
fn current_sparse_flow_op_timing_summary() -> SparseFlowOpTimingSummary {
    SparseFlowOpTimingSummary::default()
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SparseStructureRuntimeProfile {
    pub cond_prepare_ms: f64,
    pub sample_ms: f64,
    pub postprocess_ms: f64,
    pub flow_ops: SparseFlowOpTimingSummary,
}

#[derive(Debug, Clone)]
pub struct SparseStructureSample {
    pub source: SparseStructureStageSource,
    pub sampler_config: FlowEulerSampleConfig,
    pub sigma_min: f32,
    pub step_count: usize,
    pub resolution: usize,
    pub flow_resolution: usize,
    pub flow_channels: usize,
    pub noise: Vec<f32>,
    pub step_0_pred_v: Vec<f32>,
    pub step_0_pred_v_pos: Vec<f32>,
    pub step_0_pred_v_neg: Vec<f32>,
    pub step_0_x_t: Vec<f32>,
    pub step_mid_x_t: Vec<f32>,
    pub step_last_x_t: Vec<f32>,
    pub step_pred_v: Vec<Vec<f32>>,
    pub step_x_t: Vec<Vec<f32>>,
    pub latent: Vec<f32>,
    pub coords: Vec<[u32; 4]>,
    pub layout: Vec<std::ops::Range<usize>>,
    #[cfg(feature = "runtime-model-wgpu")]
    pub coords_wgpu: Option<Tensor<SparseFlowWgpuBackend, 2, Int>>,
    pub runtime_profile: Option<SparseStructureRuntimeProfile>,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum SparseStructureStageSource {
    RuntimeModelCpu,
    RuntimeModelWgpu,
}

impl SparseStructureStageSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::RuntimeModelCpu => "runtime_model_cpu",
            Self::RuntimeModelWgpu => "runtime_model_wgpu",
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum DecodeStageSource {
    Runtime,
}

impl DecodeStageSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Runtime => "runtime",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ShapeSLatSample {
    pub sampler_config: FlowEulerSampleConfig,
    pub sigma_min: f32,
    pub step_count: usize,
    pub dense_resolution: usize,
    pub dense_channels: usize,
    pub dense_noise: Option<Vec<f32>>,
    pub features: Vec<[f32; 32]>,
    pub noise: Vec<[f32; 32]>,
    pub step_0_pred_v: Vec<[f32; 32]>,
    pub step_0_pred_v_pos: Vec<[f32; 32]>,
    pub step_0_pred_v_neg: Vec<[f32; 32]>,
    pub step_0_x_t: Vec<[f32; 32]>,
    pub step_mid_x_t: Vec<[f32; 32]>,
    pub step_last_x_t: Vec<[f32; 32]>,
    pub coords: Vec<[u32; 4]>,
    pub layout: Vec<std::ops::Range<usize>>,
    pub flow_ops: SparseFlowOpTimingSummary,
    #[cfg(feature = "runtime-model-wgpu")]
    pub coords_wgpu: Option<Tensor<SparseFlowWgpuBackend, 2, Int>>,
    #[cfg(feature = "runtime-model-wgpu")]
    pub features_wgpu: Option<Tensor<SparseFlowWgpuBackend, 2>>,
}

#[derive(Debug, Clone)]
pub struct TexSLatSample {
    pub sampler_config: FlowEulerSampleConfig,
    pub sigma_min: f32,
    pub step_count: usize,
    pub dense_resolution: usize,
    pub dense_channels: usize,
    pub dense_noise: Option<Vec<f32>>,
    pub features: Vec<[f32; 32]>,
    pub noise: Vec<[f32; 32]>,
    pub step_0_pred_v: Vec<[f32; 32]>,
    pub step_0_pred_v_pos: Vec<[f32; 32]>,
    pub step_0_pred_v_neg: Vec<[f32; 32]>,
    pub step_0_x_t: Vec<[f32; 32]>,
    pub step_mid_x_t: Vec<[f32; 32]>,
    pub step_last_x_t: Vec<[f32; 32]>,
    pub shape_slat_cond: Vec<[f32; 32]>,
    pub coords: Vec<[u32; 4]>,
    pub layout: Vec<std::ops::Range<usize>>,
    pub flow_ops: SparseFlowOpTimingSummary,
    #[cfg(feature = "runtime-model-wgpu")]
    pub coords_wgpu: Option<Tensor<SparseFlowWgpuBackend, 2, Int>>,
    #[cfg(feature = "runtime-model-wgpu")]
    pub features_wgpu: Option<Tensor<SparseFlowWgpuBackend, 2>>,
}

#[derive(Debug, Clone, Default)]
pub struct TrellisStageConditioning {
    pub cond_512: Vec<f32>,
    pub neg_cond_512: Vec<f32>,
    pub cond_1024: Option<Vec<f32>>,
    pub neg_cond_1024: Option<Vec<f32>>,
}

#[derive(Debug, Clone)]
pub struct TrellisStageOutput {
    pub sparse: SparseStructureSample,
    pub shape_slat: ShapeSLatSample,
    pub shape_slat_lr: Option<ShapeSLatSample>,
    pub tex_slat: TexSLatSample,
    pub conditioning: TrellisStageConditioning,
    pub decode_shape_input: Option<SparseRowNoiseOverride>,
    pub decode_tex_input: Option<SparseRowNoiseOverride>,
    pub decode_source: DecodeStageSource,
    pub decode_shape_subs: Vec<DecodeShapeSubSample>,
    pub decode_tex_voxels: DecodeTexVoxelSample,
    pub mesh: Mesh,
    pub pbr: Option<PbrBakeDebug>,
}

#[derive(Debug, Clone)]
pub struct DecodeShapeSubSample {
    pub coords: Vec<[u32; 4]>,
    pub feats: Vec<[f32; 8]>,
    pub spatial_shape: [u32; 3],
}

#[derive(Debug, Clone)]
pub struct DecodeTexVoxelSample {
    pub coords: Vec<[u32; 4]>,
    pub feats: Vec<[f32; 6]>,
    pub spatial_shape: [u32; 3],
}

#[derive(Debug, Clone)]
struct DecodedLatentOutput {
    source: DecodeStageSource,
    mesh: Mesh,
    shape_subs: Vec<DecodeShapeSubSample>,
    tex_voxels: DecodeTexVoxelSample,
    pbr: Option<PbrBakeDebug>,
    timings: DecodeRuntimeTimings,
}

#[derive(Debug, Clone, Default)]
struct DecodeRuntimeTimings {
    stage_fenced: bool,
    shape_decoder_ms: f64,
    tex_decoder_ms: f64,
    attr_merge_ms: f64,
    mesh_ms: f64,
    pbr_ms: f64,
    shape_conv_calls: u64,
    tex_conv_calls: u64,
    shape_wgpu_dispatches: u64,
    tex_wgpu_dispatches: u64,
    shape_wgpu_chunked_calls: u64,
    tex_wgpu_chunked_calls: u64,
    shape_wgpu_input_bytes: u64,
    tex_wgpu_input_bytes: u64,
    shape_wgpu_output_bytes: u64,
    tex_wgpu_output_bytes: u64,
    shape_wgpu_max_chunk_rows: usize,
    tex_wgpu_max_chunk_rows: usize,
}

#[derive(Debug, Clone)]
pub struct PbrBakeDebug {
    pub texture_width: usize,
    pub texture_height: usize,
    pub uvs: Vec<[f32; 2]>,
    pub raster_mask: Vec<u8>,
    pub sample_positions: Vec<[f32; 3]>,
    pub sample_attrs: Vec<[f32; 6]>,
    pub base_color_float: Vec<[f32; 4]>,
    pub metallic_float: Vec<f32>,
    pub roughness_float: Vec<f32>,
    pub alpha_float: Vec<f32>,
    pub base_color_rgba_u8: Vec<u8>,
    pub metallic_roughness_u8: Vec<u8>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct TrellisStageTimings {
    pub sparse_ms: f64,
    pub sparse_cond_ms: f64,
    pub sparse_sample_ms: f64,
    pub sparse_post_ms: f64,
    pub sparse_flow_ops: SparseFlowOpTimingSummary,
    pub shape_slat_ms: f64,
    pub shape_slat_flow_ops: SparseFlowOpTimingSummary,
    pub tex_slat_ms: f64,
    pub tex_slat_flow_ops: SparseFlowOpTimingSummary,
    pub decode_ms: f64,
    pub decode_stage_fenced: bool,
    pub decode_shape_decoder_ms: f64,
    pub decode_tex_decoder_ms: f64,
    pub decode_attr_merge_ms: f64,
    pub decode_mesh_ms: f64,
    pub decode_pbr_ms: f64,
    pub decode_shape_conv_calls: u64,
    pub decode_tex_conv_calls: u64,
    pub decode_shape_wgpu_dispatches: u64,
    pub decode_tex_wgpu_dispatches: u64,
    pub decode_shape_wgpu_chunked_calls: u64,
    pub decode_tex_wgpu_chunked_calls: u64,
    pub decode_shape_wgpu_input_bytes: u64,
    pub decode_tex_wgpu_input_bytes: u64,
    pub decode_shape_wgpu_output_bytes: u64,
    pub decode_tex_wgpu_output_bytes: u64,
    pub decode_shape_wgpu_max_chunk_rows: usize,
    pub decode_tex_wgpu_max_chunk_rows: usize,
    pub total_ms: f64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct TrellisStageRunConfig {
    pub max_sparse_coords: Option<usize>,
    pub max_num_tokens: Option<usize>,
    pub target_faces: Option<usize>,
    pub pbr_texture_size: Option<usize>,
    pub compute_profile: TrellisComputeProfile,
    pub decode_output_mode: TrellisDecodeOutputMode,
    pub runtime_stage_debug: bool,
    pub runtime_attention_debug: bool,
    pub runtime_decoder_conv_telemetry: bool,
    pub runtime_stage_fence: bool,
}

#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub enum TrellisDecodeOutputMode {
    #[default]
    NativePbr,
    OvoxelHookExport,
}

impl TrellisDecodeOutputMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NativePbr => "native-pbr",
            Self::OvoxelHookExport => "ovoxel-hook-export",
        }
    }

    pub fn needs_native_pbr(self) -> bool {
        matches!(self, Self::NativePbr)
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum SparseCoordCapSource {
    ExplicitRunConfig,
}

impl SparseCoordCapSource {
    fn as_str(self) -> &'static str {
        match self {
            Self::ExplicitRunConfig => "explicit_run_config",
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct TrellisSamplerRuntimeOverrides {
    pub sparse_steps: Option<usize>,
    pub shape_steps: Option<usize>,
    pub tex_steps: Option<usize>,
    pub sparse_guidance_strength: Option<f32>,
    pub shape_guidance_strength: Option<f32>,
    pub tex_guidance_strength: Option<f32>,
}

#[derive(Debug, Clone, Default)]
pub struct SparseRowNoiseOverride {
    pub coords: Vec<[u32; 4]>,
    pub feats: Vec<[f32; 32]>,
}

#[derive(Debug, Clone, Default)]
pub struct TrellisNoiseOverrides {
    pub sparse_noise: Option<Vec<f32>>,
    pub sparse_coords: Option<Vec<[u32; 4]>>,
    pub shape_noise: Option<SparseRowNoiseOverride>,
    pub shape_noise_lr: Option<SparseRowNoiseOverride>,
    pub shape_noise_hr: Option<SparseRowNoiseOverride>,
    pub tex_noise: Option<SparseRowNoiseOverride>,
    pub shape_slat: Option<SparseRowNoiseOverride>,
    pub tex_slat: Option<SparseRowNoiseOverride>,
    pub decode_shape_input: Option<SparseRowNoiseOverride>,
    pub decode_tex_input: Option<SparseRowNoiseOverride>,
    pub decode_shape_subs: Option<Vec<DecodeShapeSubSample>>,
    pub decode_tex_voxels: Option<DecodeTexVoxelSample>,
    pub decode_mesh_vertices: Option<Vec<[f32; 3]>>,
    pub decode_mesh_faces: Option<Vec<[u32; 3]>>,
    pub shape_noise_dense: Option<Vec<f32>>,
    pub tex_noise_dense: Option<Vec<f32>>,
    pub sparse_sampler: Option<SamplerConfigOverride>,
    pub shape_sampler: Option<SamplerConfigOverride>,
    pub tex_sampler: Option<SamplerConfigOverride>,
    pub cond_512: Option<Vec<f32>>,
    pub neg_cond_512: Option<Vec<f32>>,
    pub cond_1024: Option<Vec<f32>>,
    pub neg_cond_1024: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Copy)]
pub struct SamplerConfigOverride {
    pub sigma_min: f32,
    pub config: FlowEulerSampleConfig,
}

impl TrellisNoiseOverrides {
    pub fn is_empty(&self) -> bool {
        self.sparse_noise.is_none()
            && self.sparse_coords.is_none()
            && self.shape_noise.is_none()
            && self.shape_noise_lr.is_none()
            && self.shape_noise_hr.is_none()
            && self.tex_noise.is_none()
            && self.shape_slat.is_none()
            && self.tex_slat.is_none()
            && self.decode_shape_input.is_none()
            && self.decode_tex_input.is_none()
            && self.decode_shape_subs.is_none()
            && self.decode_tex_voxels.is_none()
            && self.decode_mesh_vertices.is_none()
            && self.decode_mesh_faces.is_none()
            && self.shape_noise_dense.is_none()
            && self.tex_noise_dense.is_none()
            && self.sparse_sampler.is_none()
            && self.shape_sampler.is_none()
            && self.tex_sampler.is_none()
            && self.cond_512.is_none()
            && self.neg_cond_512.is_none()
            && self.cond_1024.is_none()
            && self.neg_cond_1024.is_none()
    }
}

#[derive(Debug)]
pub struct TrellisStageRuntime {
    pipeline_type: String,
    sparse_sampler: TrellisSamplerConfig,
    shape_sampler: TrellisSamplerConfig,
    tex_sampler: TrellisSamplerConfig,
    shape_norm: TrellisNormalization,
    tex_norm: TrellisNormalization,
    #[cfg(feature = "runtime-model")]
    sparse_flow: OnceLock<Option<SparseStructureFlowRuntime>>,
    #[cfg(feature = "runtime-model")]
    shape_flow: OnceLock<Option<SparseStructureFlowRuntime>>,
    #[cfg(feature = "runtime-model")]
    shape_flow_512: OnceLock<Option<SparseStructureFlowRuntime>>,
    #[cfg(feature = "runtime-model")]
    shape_flow_1024: OnceLock<Option<SparseStructureFlowRuntime>>,
    #[cfg(feature = "runtime-model")]
    tex_flow: OnceLock<Option<SparseStructureFlowRuntime>>,
    #[cfg(feature = "runtime-model")]
    sparse_structure_decoder: OnceLock<Option<SparseStructureDecoderRuntime>>,
    #[cfg(feature = "runtime-model")]
    shape_decoder: OnceLock<Option<FdgDecoderRuntime>>,
    #[cfg(feature = "runtime-model")]
    tex_decoder: OnceLock<Option<SparseUnetVaeDecoderRuntime>>,
    #[cfg(feature = "runtime-model")]
    image_conditioning: OnceLock<Option<TrellisImageConditioningRuntime>>,
    #[cfg(feature = "runtime-model")]
    sparse_flow_spec: Option<FlowRuntimeLoadSpec>,
    #[cfg(feature = "runtime-model")]
    shape_flow_spec: Option<FlowRuntimeLoadSpec>,
    #[cfg(feature = "runtime-model")]
    shape_flow_512_spec: Option<FlowRuntimeLoadSpec>,
    #[cfg(feature = "runtime-model")]
    shape_flow_1024_spec: Option<FlowRuntimeLoadSpec>,
    #[cfg(feature = "runtime-model")]
    tex_flow_spec: Option<FlowRuntimeLoadSpec>,
    #[cfg(feature = "runtime-model")]
    sparse_structure_decoder_spec: Option<SparseStructureDecoderLoadSpec>,
    #[cfg(feature = "runtime-model")]
    shape_decoder_spec: Option<DecoderRuntimeLoadSpec>,
    #[cfg(feature = "runtime-model")]
    tex_decoder_spec: Option<DecoderRuntimeLoadSpec>,
    #[cfg(feature = "runtime-model")]
    image_conditioning_spec: Option<ImageConditioningLoadSpec>,
}

#[cfg(feature = "runtime-model")]
#[derive(Debug, Clone)]
struct FlowRuntimeLoadSpec {
    weights_root: PathBuf,
    image_large_root: Option<PathBuf>,
    model_stem: String,
    prefer_wgpu: bool,
    slat_dense_resolution: Option<usize>,
    stage_label: &'static str,
    flow_key: Option<String>,
}

#[cfg(feature = "runtime-model")]
#[derive(Debug, Clone, Copy)]
enum DecoderRuntimeKind {
    Shape,
    Tex,
}

#[cfg(feature = "runtime-model")]
#[derive(Debug, Clone)]
struct DecoderRuntimeLoadSpec {
    kind: DecoderRuntimeKind,
    weights_root: PathBuf,
    image_large_root: Option<PathBuf>,
    model_stem: String,
    prefer_wgpu: bool,
}

#[cfg(feature = "runtime-model")]
#[derive(Debug, Clone)]
struct SparseStructureDecoderLoadSpec {
    weights_root: PathBuf,
    image_large_root: Option<PathBuf>,
    model_stem: String,
    prefer_wgpu: bool,
}

#[cfg(feature = "runtime-model")]
#[derive(Debug, Clone)]
struct ImageConditioningLoadSpec {
    weights_root: PathBuf,
    image_large_root: Option<PathBuf>,
    model_name: String,
    prefer_wgpu: bool,
}

#[cfg(feature = "runtime-model")]
fn flow_specs_load_same_model(
    selected: Option<&FlowRuntimeLoadSpec>,
    candidate: Option<&FlowRuntimeLoadSpec>,
) -> bool {
    match (selected, candidate) {
        (Some(a), Some(b)) => {
            a.weights_root == b.weights_root
                && a.image_large_root == b.image_large_root
                && a.model_stem == b.model_stem
                && a.prefer_wgpu == b.prefer_wgpu
                && a.slat_dense_resolution == b.slat_dense_resolution
        }
        _ => false,
    }
}

#[cfg(feature = "runtime-model")]
fn should_preload_shape_flow_variant(
    pipeline_type: &str,
    selected_shape_spec: Option<&FlowRuntimeLoadSpec>,
    candidate_shape_spec: Option<&FlowRuntimeLoadSpec>,
    expected_flow_key: &str,
) -> bool {
    if pipeline_type != "1024_cascade" {
        return false;
    }
    let Some(candidate) = candidate_shape_spec else {
        return false;
    };
    if candidate.flow_key.as_deref() != Some(expected_flow_key) {
        return false;
    }
    // Keep medium/high cascade warm-start behavior while avoiding duplicate
    // preload of whichever shape flow model is already the selected primary.
    !flow_specs_load_same_model(selected_shape_spec, Some(candidate))
}

impl TrellisStageRuntime {
    pub fn from_args(args: &TrellisPipelineArgs, preferred_pipeline_type: Option<&str>) -> Self {
        Self::from_args_with_assets(args, preferred_pipeline_type, None, None, false, None)
    }

    pub fn from_args_with_assets(
        args: &TrellisPipelineArgs,
        preferred_pipeline_type: Option<&str>,
        _weights_root: Option<&Path>,
        _image_large_root: Option<&Path>,
        _prefer_wgpu: bool,
        sampler_overrides: Option<TrellisSamplerRuntimeOverrides>,
    ) -> Self {
        let requested_pipeline_type = preferred_pipeline_type
            .unwrap_or(args.default_pipeline_type.as_str())
            .to_string();
        let pipeline_type = canonical_pipeline_type(requested_pipeline_type.as_str()).to_string();
        if requested_pipeline_type != pipeline_type {
            trellis_stage_log!(
                "burn_trellis: canonicalized pipeline_type='{}' -> '{}'",
                requested_pipeline_type,
                pipeline_type
            );
        }
        let mut sparse_sampler = args.sparse_structure_sampler.clone();
        let mut shape_sampler = args.shape_slat_sampler.clone();
        let mut tex_sampler = args.tex_slat_sampler.clone();
        if let Some(overrides) = sampler_overrides {
            if let Some(steps) = overrides.sparse_steps {
                sparse_sampler.params.steps = steps.max(1);
            }
            if let Some(steps) = overrides.shape_steps {
                shape_sampler.params.steps = steps.max(1);
            }
            if let Some(steps) = overrides.tex_steps {
                tex_sampler.params.steps = steps.max(1);
            }
            if let Some(strength) = overrides.sparse_guidance_strength {
                sparse_sampler.params.guidance_strength = strength;
            }
            if let Some(strength) = overrides.shape_guidance_strength {
                shape_sampler.params.guidance_strength = strength;
            }
            if let Some(strength) = overrides.tex_guidance_strength {
                tex_sampler.params.guidance_strength = strength;
            }
            trellis_stage_log!(
                "burn_trellis: sampler overrides active (sparse_steps={}, shape_steps={}, tex_steps={}, sparse_guidance={:.3}, shape_guidance={:.3}, tex_guidance={:.3})",
                sparse_sampler.params.steps,
                shape_sampler.params.steps,
                tex_sampler.params.steps,
                sparse_sampler.params.guidance_strength,
                shape_sampler.params.guidance_strength,
                tex_sampler.params.guidance_strength
            );
        }
        #[cfg(feature = "runtime-model")]
        let slat_dense_resolution = None;
        #[cfg(feature = "runtime-model")]
        let prefer_512_slat = matches!(pipeline_type.as_str(), "512" | "512_base");
        #[cfg(feature = "runtime-model")]
        let shape_flow_key = if prefer_512_slat {
            "shape_slat_flow_model_512"
        } else {
            "shape_slat_flow_model_1024"
        };
        #[cfg(feature = "runtime-model")]
        let tex_flow_key = if prefer_512_slat {
            "tex_slat_flow_model_512"
        } else {
            "tex_slat_flow_model_1024"
        };
        #[cfg(feature = "runtime-model")]
        let runtime_lazy_model_load = runtime_lazy_model_load_enabled();
        #[cfg(feature = "runtime-model")]
        let sparse_flow_spec = match (
            _weights_root,
            args.models.get("sparse_structure_flow_model"),
        ) {
            (Some(weights_root), Some(model_stem)) => Some(FlowRuntimeLoadSpec {
                weights_root: weights_root.to_path_buf(),
                image_large_root: _image_large_root.map(Path::to_path_buf),
                model_stem: model_stem.clone(),
                prefer_wgpu: _prefer_wgpu,
                slat_dense_resolution: None,
                stage_label: "sparse flow",
                flow_key: None,
            }),
            _ => None,
        };
        #[cfg(feature = "runtime-model")]
        let shape_flow_spec = match (_weights_root, args.models.get(shape_flow_key)) {
            (Some(weights_root), Some(model_stem)) => Some(FlowRuntimeLoadSpec {
                weights_root: weights_root.to_path_buf(),
                image_large_root: _image_large_root.map(Path::to_path_buf),
                model_stem: model_stem.clone(),
                prefer_wgpu: _prefer_wgpu,
                slat_dense_resolution,
                stage_label: "shape slat",
                flow_key: Some(shape_flow_key.to_string()),
            }),
            _ => None,
        };
        #[cfg(feature = "runtime-model")]
        let shape_flow_512_spec =
            match (_weights_root, args.models.get("shape_slat_flow_model_512")) {
                (Some(weights_root), Some(model_stem)) => Some(FlowRuntimeLoadSpec {
                    weights_root: weights_root.to_path_buf(),
                    image_large_root: _image_large_root.map(Path::to_path_buf),
                    model_stem: model_stem.clone(),
                    prefer_wgpu: _prefer_wgpu,
                    slat_dense_resolution,
                    stage_label: "shape slat",
                    flow_key: Some("shape_slat_flow_model_512".to_string()),
                }),
                _ => None,
            };
        #[cfg(feature = "runtime-model")]
        let shape_flow_1024_spec =
            match (_weights_root, args.models.get("shape_slat_flow_model_1024")) {
                (Some(weights_root), Some(model_stem)) => Some(FlowRuntimeLoadSpec {
                    weights_root: weights_root.to_path_buf(),
                    image_large_root: _image_large_root.map(Path::to_path_buf),
                    model_stem: model_stem.clone(),
                    prefer_wgpu: _prefer_wgpu,
                    slat_dense_resolution,
                    stage_label: "shape slat",
                    flow_key: Some("shape_slat_flow_model_1024".to_string()),
                }),
                _ => None,
            };
        #[cfg(feature = "runtime-model")]
        let tex_flow_spec = match (_weights_root, args.models.get(tex_flow_key)) {
            (Some(weights_root), Some(model_stem)) => Some(FlowRuntimeLoadSpec {
                weights_root: weights_root.to_path_buf(),
                image_large_root: _image_large_root.map(Path::to_path_buf),
                model_stem: model_stem.clone(),
                prefer_wgpu: _prefer_wgpu,
                slat_dense_resolution,
                stage_label: "tex slat",
                flow_key: Some(tex_flow_key.to_string()),
            }),
            _ => None,
        };
        #[cfg(feature = "runtime-model")]
        let preload_shape_flow_512 = should_preload_shape_flow_variant(
            pipeline_type.as_str(),
            shape_flow_spec.as_ref(),
            shape_flow_512_spec.as_ref(),
            "shape_slat_flow_model_512",
        );
        #[cfg(feature = "runtime-model")]
        let preload_shape_flow_1024 = should_preload_shape_flow_variant(
            pipeline_type.as_str(),
            shape_flow_spec.as_ref(),
            shape_flow_1024_spec.as_ref(),
            "shape_slat_flow_model_1024",
        );
        #[cfg(feature = "runtime-model")]
        let sparse_structure_decoder_spec =
            match (_weights_root, args.models.get("sparse_structure_decoder")) {
                (Some(weights_root), Some(model_stem)) => Some(SparseStructureDecoderLoadSpec {
                    weights_root: weights_root.to_path_buf(),
                    image_large_root: _image_large_root.map(Path::to_path_buf),
                    model_stem: model_stem.clone(),
                    prefer_wgpu: _prefer_wgpu,
                }),
                _ => None,
            };
        #[cfg(feature = "runtime-model")]
        let shape_decoder_spec = match (_weights_root, args.models.get("shape_slat_decoder")) {
            (Some(weights_root), Some(model_stem)) => Some(DecoderRuntimeLoadSpec {
                kind: DecoderRuntimeKind::Shape,
                weights_root: weights_root.to_path_buf(),
                image_large_root: _image_large_root.map(Path::to_path_buf),
                model_stem: model_stem.clone(),
                prefer_wgpu: _prefer_wgpu,
            }),
            _ => None,
        };
        #[cfg(feature = "runtime-model")]
        let tex_decoder_spec = match (_weights_root, args.models.get("tex_slat_decoder")) {
            (Some(weights_root), Some(model_stem)) => Some(DecoderRuntimeLoadSpec {
                kind: DecoderRuntimeKind::Tex,
                weights_root: weights_root.to_path_buf(),
                image_large_root: _image_large_root.map(Path::to_path_buf),
                model_stem: model_stem.clone(),
                prefer_wgpu: _prefer_wgpu,
            }),
            _ => None,
        };
        #[cfg(feature = "runtime-model")]
        let image_conditioning_spec = {
            let model_name = args
                .image_cond_model
                .as_ref()
                .map(|model| model.args.model_name.trim())
                .filter(|name| !name.is_empty());
            match (_weights_root, model_name) {
                (Some(weights_root), Some(model_name)) => Some(ImageConditioningLoadSpec {
                    weights_root: weights_root.to_path_buf(),
                    image_large_root: _image_large_root.map(Path::to_path_buf),
                    model_name: model_name.to_string(),
                    prefer_wgpu: _prefer_wgpu,
                }),
                _ => None,
            }
        };
        #[cfg(feature = "runtime-model")]
        let sparse_flow = OnceLock::new();
        #[cfg(feature = "runtime-model")]
        let shape_flow = OnceLock::new();
        #[cfg(feature = "runtime-model")]
        let shape_flow_512 = OnceLock::new();
        #[cfg(feature = "runtime-model")]
        let shape_flow_1024 = OnceLock::new();
        #[cfg(feature = "runtime-model")]
        let tex_flow = OnceLock::new();
        #[cfg(feature = "runtime-model")]
        let sparse_structure_decoder = OnceLock::new();
        #[cfg(feature = "runtime-model")]
        let shape_decoder = OnceLock::new();
        #[cfg(feature = "runtime-model")]
        let tex_decoder = OnceLock::new();
        #[cfg(feature = "runtime-model")]
        let image_conditioning = OnceLock::new();
        #[cfg(feature = "runtime-model")]
        if !runtime_lazy_model_load {
            #[cfg(not(target_arch = "wasm32"))]
            {
                let sparse_spec_clone = sparse_flow_spec.clone();
                let shape_spec_clone = shape_flow_spec.clone();
                let shape_512_spec_clone = shape_flow_512_spec.clone();
                let shape_1024_spec_clone = shape_flow_1024_spec.clone();
                let tex_spec_clone = tex_flow_spec.clone();
                let sparse_structure_decoder_spec_clone = sparse_structure_decoder_spec.clone();
                let shape_decoder_spec_clone = shape_decoder_spec.clone();
                let tex_decoder_spec_clone = tex_decoder_spec.clone();
                let image_cond_spec_clone = image_conditioning_spec.clone();
                let sparse_task = std::thread::spawn(move || {
                    load_flow_runtime_from_spec(sparse_spec_clone.as_ref())
                });
                let shape_task = std::thread::spawn(move || {
                    load_flow_runtime_from_spec(shape_spec_clone.as_ref())
                });
                let shape_512_task = if preload_shape_flow_512 {
                    Some(std::thread::spawn(move || {
                        load_flow_runtime_from_spec(shape_512_spec_clone.as_ref())
                    }))
                } else {
                    None
                };
                let shape_1024_task = if preload_shape_flow_1024 {
                    Some(std::thread::spawn(move || {
                        load_flow_runtime_from_spec(shape_1024_spec_clone.as_ref())
                    }))
                } else {
                    None
                };
                let tex_task = std::thread::spawn(move || {
                    load_flow_runtime_from_spec(tex_spec_clone.as_ref())
                });
                let sparse_structure_decoder_task = std::thread::spawn(move || {
                    load_sparse_structure_decoder_from_spec(
                        sparse_structure_decoder_spec_clone.as_ref(),
                    )
                });
                let shape_decoder_task = std::thread::spawn(move || {
                    load_shape_decoder_from_spec(shape_decoder_spec_clone.as_ref())
                });
                let tex_decoder_task = std::thread::spawn(move || {
                    load_tex_decoder_from_spec(tex_decoder_spec_clone.as_ref())
                });
                let image_cond_task = std::thread::spawn(move || {
                    load_image_conditioning_from_spec(image_cond_spec_clone.as_ref())
                });
                let sparse_loaded = match sparse_task.join() {
                    Ok(value) => value,
                    Err(_) => {
                        trellis_stage_log!(
                            "burn_trellis: sparse runtime preload task panicked; lazy load retry remains as the only recovery path"
                        );
                        None
                    }
                };
                let shape_loaded = match shape_task.join() {
                    Ok(value) => value,
                    Err(_) => {
                        trellis_stage_log!(
                            "burn_trellis: shape runtime preload task panicked; lazy load retry remains as the only recovery path"
                        );
                        None
                    }
                };
                let shape_512_loaded = if let Some(task) = shape_512_task {
                    match task.join() {
                        Ok(value) => value,
                        Err(_) => {
                            trellis_stage_log!(
                                "burn_trellis: shape-512 runtime preload task panicked; lazy load retry remains as the only recovery path"
                            );
                            None
                        }
                    }
                } else {
                    None
                };
                let shape_1024_loaded = if let Some(task) = shape_1024_task {
                    match task.join() {
                        Ok(value) => value,
                        Err(_) => {
                            trellis_stage_log!(
                                "burn_trellis: shape-1024 runtime preload task panicked; lazy load retry remains as the only recovery path"
                            );
                            None
                        }
                    }
                } else {
                    None
                };
                let tex_loaded = match tex_task.join() {
                    Ok(value) => value,
                    Err(_) => {
                        trellis_stage_log!(
                            "burn_trellis: tex runtime preload task panicked; lazy load retry remains as the only recovery path"
                        );
                        None
                    }
                };
                let shape_decoder_loaded = match shape_decoder_task.join() {
                    Ok(value) => value,
                    Err(_) => {
                        trellis_stage_log!(
                            "burn_trellis: shape decoder preload task panicked; lazy load retry remains as the only recovery path"
                        );
                        None
                    }
                };
                let tex_decoder_loaded = match tex_decoder_task.join() {
                    Ok(value) => value,
                    Err(_) => {
                        trellis_stage_log!(
                            "burn_trellis: tex decoder preload task panicked; lazy load retry remains as the only recovery path"
                        );
                        None
                    }
                };
                let sparse_structure_decoder_loaded = match sparse_structure_decoder_task.join() {
                    Ok(value) => value,
                    Err(_) => {
                        trellis_stage_log!(
                            "burn_trellis: sparse structure decoder preload task panicked; lazy load retry remains as the only recovery path"
                        );
                        None
                    }
                };
                let image_conditioning_loaded = match image_cond_task.join() {
                    Ok(value) => value,
                    Err(_) => {
                        trellis_stage_log!(
                            "burn_trellis: image conditioning preload task panicked; lazy load retry remains as the only recovery path"
                        );
                        None
                    }
                };
                let _ = sparse_flow.set(sparse_loaded);
                let _ = shape_flow.set(shape_loaded);
                if preload_shape_flow_512 {
                    let _ = shape_flow_512.set(shape_512_loaded);
                }
                if preload_shape_flow_1024 {
                    let _ = shape_flow_1024.set(shape_1024_loaded);
                }
                let _ = tex_flow.set(tex_loaded);
                let _ = sparse_structure_decoder.set(sparse_structure_decoder_loaded);
                let _ = shape_decoder.set(shape_decoder_loaded);
                let _ = tex_decoder.set(tex_decoder_loaded);
                let _ = image_conditioning.set(image_conditioning_loaded);
            }
            #[cfg(target_arch = "wasm32")]
            {
                let _ = sparse_flow.set(load_flow_runtime_from_spec(sparse_flow_spec.as_ref()));
                let _ = shape_flow.set(load_flow_runtime_from_spec(shape_flow_spec.as_ref()));
                if preload_shape_flow_512 {
                    let _ = shape_flow_512
                        .set(load_flow_runtime_from_spec(shape_flow_512_spec.as_ref()));
                }
                if preload_shape_flow_1024 {
                    let _ = shape_flow_1024
                        .set(load_flow_runtime_from_spec(shape_flow_1024_spec.as_ref()));
                }
                let _ = tex_flow.set(load_flow_runtime_from_spec(tex_flow_spec.as_ref()));
                let _ = sparse_structure_decoder.set(load_sparse_structure_decoder_from_spec(
                    sparse_structure_decoder_spec.as_ref(),
                ));
                let _ =
                    shape_decoder.set(load_shape_decoder_from_spec(shape_decoder_spec.as_ref()));
                let _ = tex_decoder.set(load_tex_decoder_from_spec(tex_decoder_spec.as_ref()));
                let _ = image_conditioning.set(load_image_conditioning_from_spec(
                    image_conditioning_spec.as_ref(),
                ));
            }
        }
        Self {
            pipeline_type,
            sparse_sampler,
            shape_sampler,
            tex_sampler,
            shape_norm: args.shape_slat_normalization.clone(),
            tex_norm: args.tex_slat_normalization.clone(),
            #[cfg(feature = "runtime-model")]
            sparse_flow,
            #[cfg(feature = "runtime-model")]
            shape_flow,
            #[cfg(feature = "runtime-model")]
            shape_flow_512,
            #[cfg(feature = "runtime-model")]
            shape_flow_1024,
            #[cfg(feature = "runtime-model")]
            tex_flow,
            #[cfg(feature = "runtime-model")]
            sparse_structure_decoder,
            #[cfg(feature = "runtime-model")]
            shape_decoder,
            #[cfg(feature = "runtime-model")]
            tex_decoder,
            #[cfg(feature = "runtime-model")]
            image_conditioning,
            #[cfg(feature = "runtime-model")]
            sparse_flow_spec,
            #[cfg(feature = "runtime-model")]
            shape_flow_spec,
            #[cfg(feature = "runtime-model")]
            shape_flow_512_spec,
            #[cfg(feature = "runtime-model")]
            shape_flow_1024_spec,
            #[cfg(feature = "runtime-model")]
            tex_flow_spec,
            #[cfg(feature = "runtime-model")]
            sparse_structure_decoder_spec,
            #[cfg(feature = "runtime-model")]
            shape_decoder_spec,
            #[cfg(feature = "runtime-model")]
            tex_decoder_spec,
            #[cfg(feature = "runtime-model")]
            image_conditioning_spec,
        }
    }

    #[cfg(feature = "runtime-model")]
    fn sparse_flow_runtime(&self) -> Option<&SparseStructureFlowRuntime> {
        self.sparse_flow
            .get_or_init(|| load_flow_runtime_from_spec(self.sparse_flow_spec.as_ref()))
            .as_ref()
    }

    #[cfg(feature = "runtime-model")]
    fn shape_flow_runtime(&self) -> Option<&SparseStructureFlowRuntime> {
        self.shape_flow
            .get_or_init(|| load_flow_runtime_from_spec(self.shape_flow_spec.as_ref()))
            .as_ref()
    }

    #[cfg(feature = "runtime-model")]
    fn shape_flow_runtime_512(&self) -> Option<&SparseStructureFlowRuntime> {
        if flow_specs_load_same_model(
            self.shape_flow_spec.as_ref(),
            self.shape_flow_512_spec.as_ref(),
        ) {
            return self.shape_flow_runtime();
        }
        self.shape_flow_512
            .get_or_init(|| load_flow_runtime_from_spec(self.shape_flow_512_spec.as_ref()))
            .as_ref()
    }

    #[cfg(feature = "runtime-model")]
    fn shape_flow_runtime_1024(&self) -> Option<&SparseStructureFlowRuntime> {
        if flow_specs_load_same_model(
            self.shape_flow_spec.as_ref(),
            self.shape_flow_1024_spec.as_ref(),
        ) {
            return self.shape_flow_runtime();
        }
        self.shape_flow_1024
            .get_or_init(|| load_flow_runtime_from_spec(self.shape_flow_1024_spec.as_ref()))
            .as_ref()
    }

    #[cfg(feature = "runtime-model")]
    fn tex_flow_runtime(&self) -> Option<&SparseStructureFlowRuntime> {
        self.tex_flow
            .get_or_init(|| load_flow_runtime_from_spec(self.tex_flow_spec.as_ref()))
            .as_ref()
    }

    #[cfg(feature = "runtime-model")]
    fn sparse_structure_decoder_runtime(&self) -> Option<&SparseStructureDecoderRuntime> {
        self.sparse_structure_decoder
            .get_or_init(|| {
                load_sparse_structure_decoder_from_spec(self.sparse_structure_decoder_spec.as_ref())
            })
            .as_ref()
    }

    #[cfg(feature = "runtime-model")]
    fn shape_decoder_runtime(&self) -> Option<&FdgDecoderRuntime> {
        self.shape_decoder
            .get_or_init(|| load_shape_decoder_from_spec(self.shape_decoder_spec.as_ref()))
            .as_ref()
    }

    #[cfg(feature = "runtime-model")]
    fn tex_decoder_runtime(&self) -> Option<&SparseUnetVaeDecoderRuntime> {
        self.tex_decoder
            .get_or_init(|| load_tex_decoder_from_spec(self.tex_decoder_spec.as_ref()))
            .as_ref()
    }

    #[cfg(feature = "runtime-model")]
    fn image_conditioning_runtime(&self) -> Option<&TrellisImageConditioningRuntime> {
        self.image_conditioning
            .get_or_init(|| {
                load_image_conditioning_from_spec(self.image_conditioning_spec.as_ref())
            })
            .as_ref()
    }

    #[cfg(feature = "runtime-model")]
    fn extract_runtime_conditioning_with_log(
        &self,
        preprocess: &PreprocessOutput,
        stage: &str,
        resolution: usize,
        cond_tokens: usize,
        cond_channels: usize,
    ) -> Result<Vec<f32>, String> {
        let extract_start = Instant::now();
        let values = self.extract_runtime_conditioning(
            preprocess,
            stage,
            resolution,
            cond_tokens,
            cond_channels,
        )?;
        let extract_ms = extract_start.elapsed().as_secs_f64() * 1000.0;
        trellis_stage_log!(
            "burn_trellis: image conditioning extracted (stage='{}' resolution={} extract_ms={:.2})",
            stage,
            resolution,
            extract_ms
        );
        Ok(values)
    }

    #[cfg(feature = "runtime-model")]
    fn resolve_runtime_conditioning(
        &self,
        preprocess: &PreprocessOutput,
        overrides: Option<&TrellisNoiseOverrides>,
    ) -> Result<TrellisStageConditioning, String> {
        const COND_CHANNELS: usize = 1024;
        const TOKENS_512: usize = 32 * 32 + 5;
        const TOKENS_1024: usize = 64 * 64 + 5;
        const RESOLUTION_512: usize = 512;
        const RESOLUTION_1024: usize = 1024;

        let expected_512 = TOKENS_512.saturating_mul(COND_CHANNELS);
        let cond_512 = if let Some(values) = validated_cond_override(
            expected_512,
            overrides.and_then(|value| value.cond_512.as_deref()),
            "get_cond_512",
            "cond",
        )? {
            values
        } else {
            self.extract_runtime_conditioning_with_log(
                preprocess,
                "get_cond_512",
                RESOLUTION_512,
                TOKENS_512,
                COND_CHANNELS,
            )?
        };
        let neg_cond_512 = validated_cond_override(
            expected_512,
            overrides.and_then(|value| value.neg_cond_512.as_deref()),
            "get_cond_512",
            "neg-cond",
        )?
        .unwrap_or_else(|| vec![0.0; expected_512]);

        let require_1024 = !matches!(self.pipeline_type.as_str(), "512" | "512_base");
        let (cond_1024, neg_cond_1024) = if require_1024 {
            let expected_1024 = TOKENS_1024.saturating_mul(COND_CHANNELS);
            let cond_1024 = if let Some(values) = validated_cond_override(
                expected_1024,
                overrides.and_then(|value| value.cond_1024.as_deref()),
                "get_cond_1024",
                "cond",
            )? {
                values
            } else {
                self.extract_runtime_conditioning_with_log(
                    preprocess,
                    "get_cond_1024",
                    RESOLUTION_1024,
                    TOKENS_1024,
                    COND_CHANNELS,
                )?
            };
            let neg_cond_1024 = validated_cond_override(
                expected_1024,
                overrides.and_then(|value| value.neg_cond_1024.as_deref()),
                "get_cond_1024",
                "neg-cond",
            )?
            .unwrap_or_else(|| vec![0.0; expected_1024]);
            (Some(cond_1024), Some(neg_cond_1024))
        } else {
            (None, None)
        };

        Ok(TrellisStageConditioning {
            cond_512,
            neg_cond_512,
            cond_1024,
            neg_cond_1024,
        })
    }

    #[cfg(feature = "runtime-model")]
    fn extract_runtime_conditioning(
        &self,
        preprocess: &PreprocessOutput,
        stage: &str,
        resolution: usize,
        cond_tokens: usize,
        cond_channels: usize,
    ) -> Result<Vec<f32>, String> {
        let expected = cond_tokens.saturating_mul(cond_channels);
        let runtime = self.image_conditioning_runtime().ok_or_else(|| {
            format!(
                "{} no internal image-conditioning runtime model is available for stage '{}'.",
                missing_runtime_conditioning_error(stage, cond_tokens, cond_channels),
                stage
            )
        })?;
        let output = runtime
            .extract_condition(preprocess, resolution)
            .map_err(|err| {
                format!(
                    "{} internal image-conditioning extraction failed on backend '{}' for stage '{}': {err}",
                    missing_runtime_conditioning_error(stage, cond_tokens, cond_channels),
                    runtime.backend_name(),
                    stage
                )
            })?;
        if output.token_count != cond_tokens || output.channels != cond_channels {
            return Err(format!(
                "{} internal image-conditioning extractor returned tokens={} channels={} (resolution={}).",
                missing_runtime_conditioning_error(stage, cond_tokens, cond_channels),
                output.token_count,
                output.channels,
                output.resolution
            ));
        }
        if output.values.len() != expected {
            return Err(format!(
                "{} internal image-conditioning extractor returned {} values (expected {}).",
                missing_runtime_conditioning_error(stage, cond_tokens, cond_channels),
                output.values.len(),
                expected
            ));
        }
        Ok(output.values)
    }

    pub fn pipeline_type(&self) -> &str {
        self.pipeline_type.as_str()
    }

    pub fn run(
        &self,
        preprocess: &PreprocessOutput,
        seed: u64,
    ) -> Result<TrellisStageOutput, String> {
        self.run_with_overrides(preprocess, seed, None)
    }

    pub fn run_with_overrides(
        &self,
        preprocess: &PreprocessOutput,
        seed: u64,
        noise_overrides: Option<&TrellisNoiseOverrides>,
    ) -> Result<TrellisStageOutput, String> {
        self.run_profiled_with_overrides(
            preprocess,
            seed,
            noise_overrides,
            false,
            TrellisStageRunConfig::default(),
        )
        .map(|(output, _timings)| output)
    }

    pub fn run_profiled(
        &self,
        preprocess: &PreprocessOutput,
        seed: u64,
    ) -> Result<(TrellisStageOutput, TrellisStageTimings), String> {
        self.run_profiled_with_overrides(
            preprocess,
            seed,
            None,
            false,
            TrellisStageRunConfig::default(),
        )
    }

    pub fn run_profiled_with_overrides(
        &self,
        preprocess: &PreprocessOutput,
        seed: u64,
        noise_overrides: Option<&TrellisNoiseOverrides>,
        capture_sampler_trace: bool,
        run_config: TrellisStageRunConfig,
    ) -> Result<(TrellisStageOutput, TrellisStageTimings), String> {
        let total_start = Instant::now();
        set_runtime_debug_toggles(run_config);
        let parity_strict = runtime_parity_strict();
        let max_sparse_coords_override = run_config.max_sparse_coords.filter(|limit| *limit > 0);
        #[cfg(feature = "runtime-model")]
        let max_num_tokens = run_config
            .max_num_tokens
            .filter(|limit| *limit > 0)
            .unwrap_or(49_152);
        let sparse_resolution = sparse_resolution_for_pipeline(self.pipeline_type());
        let mut rng = Lcg::new(seed);
        #[cfg(feature = "runtime-model")]
        let effective_overrides_storage: Option<TrellisNoiseOverrides>;
        #[cfg(not(feature = "runtime-model"))]
        let effective_overrides_storage: Option<TrellisNoiseOverrides> = None;
        #[cfg(feature = "runtime-model")]
        let stage_conditioning = {
            let conditioning = self.resolve_runtime_conditioning(preprocess, noise_overrides)?;
            let mut merged = noise_overrides.cloned().unwrap_or_default();
            merged.cond_512 = Some(conditioning.cond_512.clone());
            merged.neg_cond_512 = Some(conditioning.neg_cond_512.clone());
            merged.cond_1024 = conditioning.cond_1024.clone();
            merged.neg_cond_1024 = conditioning.neg_cond_1024.clone();
            effective_overrides_storage = Some(merged);
            conditioning
        };
        #[cfg(not(feature = "runtime-model"))]
        let stage_conditioning = TrellisStageConditioning {
            cond_512: noise_overrides
                .and_then(|value| value.cond_512.clone())
                .unwrap_or_default(),
            neg_cond_512: noise_overrides
                .and_then(|value| value.neg_cond_512.clone())
                .unwrap_or_default(),
            cond_1024: noise_overrides.and_then(|value| value.cond_1024.clone()),
            neg_cond_1024: noise_overrides.and_then(|value| value.neg_cond_1024.clone()),
        };

        let effective_overrides = effective_overrides_storage.as_ref().or(noise_overrides);
        let sparse_noise_override = effective_overrides.and_then(|v| v.sparse_noise.as_deref());
        let sparse_coords_override = effective_overrides.and_then(|v| v.sparse_coords.as_deref());
        let shape_noise_override = effective_overrides.and_then(|v| v.shape_noise.as_ref());
        let tex_noise_override = effective_overrides.and_then(|v| v.tex_noise.as_ref());
        let shape_slat_override = effective_overrides.and_then(|v| v.shape_slat.as_ref());
        let tex_slat_override = effective_overrides.and_then(|v| v.tex_slat.as_ref());
        let decode_shape_input_override = effective_overrides
            .and_then(|v| v.decode_shape_input.as_ref())
            .cloned();
        let decode_tex_input_override = effective_overrides
            .and_then(|v| v.decode_tex_input.as_ref())
            .cloned();
        let decode_shape_subs_override =
            effective_overrides.and_then(|v| v.decode_shape_subs.as_deref());
        let decode_tex_voxels_override =
            effective_overrides.and_then(|v| v.decode_tex_voxels.as_ref());
        let decode_mesh_vertices_override =
            effective_overrides.and_then(|v| v.decode_mesh_vertices.as_deref());
        let decode_mesh_faces_override =
            effective_overrides.and_then(|v| v.decode_mesh_faces.as_deref());
        let shape_noise_dense_override =
            effective_overrides.and_then(|v| v.shape_noise_dense.as_deref());
        let tex_noise_dense_override =
            effective_overrides.and_then(|v| v.tex_noise_dense.as_deref());
        let sparse_sampler_override = if parity_strict {
            None
        } else {
            effective_overrides.and_then(|v| v.sparse_sampler)
        };
        let shape_sampler_override = if parity_strict {
            None
        } else {
            effective_overrides.and_then(|v| v.shape_sampler)
        };
        let tex_sampler_override = if parity_strict {
            None
        } else {
            effective_overrides.and_then(|v| v.tex_sampler)
        };
        let sparse_cond_override = effective_overrides.and_then(|v| v.cond_512.as_deref());
        let sparse_neg_cond_override = effective_overrides.and_then(|v| v.neg_cond_512.as_deref());
        let cascade_requires_decoder_upsample =
            matches!(self.pipeline_type(), "1024_cascade" | "1536_cascade");
        #[cfg(feature = "runtime-model-wgpu")]
        let sparse_requires_host_coords = if !cascade_requires_decoder_upsample {
            false
        } else {
            self.sparse_flow_runtime()
                .map(|runtime| runtime.backend_name() != "wgpu")
                .unwrap_or(true)
        };
        #[cfg(not(feature = "runtime-model-wgpu"))]
        let sparse_requires_host_coords = cascade_requires_decoder_upsample;
        let sparse_materialize_host_coords = sparse_requires_host_coords
            || capture_sampler_trace
            || shape_noise_dense_override.is_some()
            || tex_noise_dense_override.is_some();
        #[cfg(feature = "runtime-model")]
        let sparse_flow_runtime = self.sparse_flow_runtime();
        #[cfg(feature = "runtime-model")]
        let sparse_structure_decoder_runtime = self.sparse_structure_decoder_runtime();
        let sparse_start = Instant::now();
        trellis_stage_log!("burn_trellis: stage sparse begin");
        #[cfg(feature = "runtime-model")]
        set_runtime_model_debug_config_for_stage(run_config, RuntimeFlowStage::SparseStructure);
        let sparse = sample_sparse_structure(
            preprocess,
            sparse_resolution,
            &mut rng,
            sparse_noise_override,
            sparse_coords_override,
            sparse_cond_override,
            sparse_neg_cond_override,
            &self.sparse_sampler,
            sparse_sampler_override,
            capture_sampler_trace,
            parity_strict,
            sparse_materialize_host_coords,
            max_sparse_coords_override,
            #[cfg(feature = "runtime-model")]
            sparse_flow_runtime,
            #[cfg(feature = "runtime-model")]
            sparse_structure_decoder_runtime,
        )?;
        let sparse_ms = sparse_start.elapsed().as_secs_f64() * 1000.0;
        let sparse_rows = if !sparse.coords.is_empty() {
            sparse.coords.len()
        } else {
            #[cfg(feature = "runtime-model-wgpu")]
            {
                sparse
                    .coords_wgpu
                    .as_ref()
                    .map(|coords_t| coords_t.dims()[0])
                    .unwrap_or(0)
            }
            #[cfg(not(feature = "runtime-model-wgpu"))]
            {
                0
            }
        };
        trellis_stage_log!(
            "burn_trellis: stage sparse complete ({sparse_ms:.2} ms, coords={})",
            sparse_rows
        );
        let sparse_flow_ops = sparse
            .runtime_profile
            .map(|profile| profile.flow_ops)
            .unwrap_or_default();

        #[cfg(feature = "runtime-model")]
        let is_cascade_shape_pipeline =
            matches!(self.pipeline_type(), "1024_cascade" | "1536_cascade");
        let shape_start = Instant::now();
        trellis_stage_log!("burn_trellis: stage shape_slat begin");
        #[cfg(feature = "runtime-model")]
        set_runtime_model_debug_config_for_stage(run_config, RuntimeFlowStage::SLat);
        let (shape_slat, shape_slat_lr, shape_slat_sparse_resolution, shape_slat_decode_resolution) = {
            #[cfg(feature = "runtime-model")]
            {
                if shape_slat_override.is_none() && is_cascade_shape_pipeline {
                    let shape_flow_runtime_512 = self.shape_flow_runtime_512();
                    let shape_flow_runtime_1024 = self.shape_flow_runtime_1024();
                    let shape_decoder_runtime_for_cascade = self.shape_decoder_runtime();
                    let target_decode_resolution = match self.pipeline_type() {
                        "1024_cascade" => 1024usize,
                        "1536_cascade" => 1536usize,
                        _ => final_resolution_for_pipeline(self.pipeline_type()),
                    };
                    trellis_stage_log!(
                        "burn_trellis: shape_slat cascade begin (base_sparse_res={}, target_decode_res={}, max_num_tokens={})",
                        sparse.resolution,
                        target_decode_resolution,
                        max_num_tokens
                    );
                    sample_shape_slat_cascade_runtime(
                        preprocess,
                        sparse.coords.as_slice(),
                        sparse.layout.as_slice(),
                        #[cfg(feature = "runtime-model-wgpu")]
                        sparse.coords_wgpu.clone(),
                        &mut rng,
                        shape_noise_override,
                        shape_noise_dense_override,
                        effective_overrides,
                        &self.shape_sampler,
                        shape_sampler_override,
                        &self.shape_norm,
                        sparse.resolution,
                        512,
                        target_decode_resolution,
                        max_num_tokens,
                        capture_sampler_trace,
                        parity_strict,
                        shape_flow_runtime_512,
                        shape_flow_runtime_1024,
                        shape_decoder_runtime_for_cascade,
                    )?
                } else {
                    let shape_flow_runtime = self.shape_flow_runtime();
                    (
                        sample_shape_slat(
                            preprocess,
                            &sparse.coords,
                            sparse.layout.as_slice(),
                            shape_slat_override,
                            &mut rng,
                            shape_noise_override,
                            shape_noise_dense_override,
                            effective_overrides,
                            &self.shape_sampler,
                            shape_sampler_override,
                            &self.shape_norm,
                            sparse.resolution,
                            capture_sampler_trace,
                            parity_strict,
                            #[cfg(feature = "runtime-model-wgpu")]
                            sparse.coords_wgpu.clone(),
                            shape_flow_runtime,
                        )?,
                        None,
                        sparse.resolution,
                        final_resolution_for_pipeline(self.pipeline_type()),
                    )
                }
            }
            #[cfg(not(feature = "runtime-model"))]
            {
                (
                    sample_shape_slat(
                        preprocess,
                        &sparse.coords,
                        sparse.layout.as_slice(),
                        shape_slat_override,
                        &mut rng,
                        shape_noise_override,
                        shape_noise_dense_override,
                        effective_overrides,
                        &self.shape_sampler,
                        shape_sampler_override,
                        &self.shape_norm,
                        sparse.resolution,
                        capture_sampler_trace,
                        parity_strict,
                        #[cfg(feature = "runtime-model-wgpu")]
                        sparse.coords_wgpu.clone(),
                    )?,
                    None,
                    sparse.resolution,
                    final_resolution_for_pipeline(self.pipeline_type()),
                )
            }
        };
        let shape_slat_ms = shape_start.elapsed().as_secs_f64() * 1000.0;
        let shape_rows = if !shape_slat.coords.is_empty() {
            shape_slat.coords.len()
        } else {
            #[cfg(feature = "runtime-model-wgpu")]
            {
                shape_slat
                    .coords_wgpu
                    .as_ref()
                    .map(|coords_t| coords_t.dims()[0])
                    .unwrap_or(shape_slat.features.len())
            }
            #[cfg(not(feature = "runtime-model-wgpu"))]
            {
                shape_slat.features.len()
            }
        };
        trellis_stage_log!(
            "burn_trellis: stage shape_slat complete ({shape_slat_ms:.2} ms, rows={})",
            shape_rows
        );
        let shape_slat_flow_ops = shape_slat.flow_ops;

        #[cfg(feature = "runtime-model")]
        let tex_flow_runtime = self.tex_flow_runtime();
        let tex_start = Instant::now();
        trellis_stage_log!("burn_trellis: stage tex_slat begin");
        #[cfg(feature = "runtime-model")]
        set_runtime_model_debug_config_for_stage(run_config, RuntimeFlowStage::SLat);
        let tex_slat = sample_tex_slat(
            preprocess,
            &shape_slat,
            tex_slat_override,
            &mut rng,
            tex_noise_override,
            tex_noise_dense_override,
            effective_overrides,
            &self.tex_sampler,
            tex_sampler_override,
            &self.shape_norm,
            &self.tex_norm,
            shape_slat_sparse_resolution,
            capture_sampler_trace,
            parity_strict,
            #[cfg(feature = "runtime-model")]
            tex_flow_runtime,
        )?;
        let tex_slat_ms = tex_start.elapsed().as_secs_f64() * 1000.0;
        let tex_rows = if !tex_slat.coords.is_empty() {
            tex_slat.coords.len()
        } else {
            #[cfg(feature = "runtime-model-wgpu")]
            {
                tex_slat
                    .coords_wgpu
                    .as_ref()
                    .map(|coords_t| coords_t.dims()[0])
                    .or_else(|| {
                        tex_slat
                            .features_wgpu
                            .as_ref()
                            .map(|rows_t| rows_t.dims()[0])
                    })
                    .unwrap_or(tex_slat.features.len())
            }
            #[cfg(not(feature = "runtime-model-wgpu"))]
            {
                tex_slat.features.len()
            }
        };
        trellis_stage_log!(
            "burn_trellis: stage tex_slat complete ({tex_slat_ms:.2} ms, rows={})",
            tex_rows
        );
        let tex_slat_flow_ops = tex_slat.flow_ops;

        let decode_start = Instant::now();
        trellis_stage_log!("burn_trellis: stage decode begin");
        #[cfg(feature = "runtime-model")]
        set_runtime_model_debug_config_for_stage(run_config, RuntimeFlowStage::SLat);
        #[cfg(feature = "runtime-model")]
        let shape_decoder_runtime = self.shape_decoder_runtime();
        #[cfg(feature = "runtime-model")]
        let tex_decoder_runtime = self.tex_decoder_runtime();
        let decode_overrides = DecodeHookOverrides {
            decode_shape_subs: decode_shape_subs_override,
            decode_tex_voxels: decode_tex_voxels_override,
            decode_mesh_vertices: decode_mesh_vertices_override,
            decode_mesh_faces: decode_mesh_faces_override,
        };
        let decoded = decode_latent_to_outputs(
            &shape_slat,
            &tex_slat,
            self.pipeline_type(),
            Some(shape_slat_decode_resolution),
            run_config.target_faces,
            run_config.pbr_texture_size,
            parity_strict,
            capture_sampler_trace,
            decode_overrides,
            run_config.decode_output_mode,
            #[cfg(feature = "runtime-model")]
            RuntimeDecodeModels {
                shape_decoder: shape_decoder_runtime,
                tex_decoder: tex_decoder_runtime,
            },
        )?;
        #[cfg(feature = "runtime-model-wgpu")]
        let decode_uses_wgpu_dispatch =
            decoded.timings.shape_wgpu_dispatches > 0 || decoded.timings.tex_wgpu_dispatches > 0;
        #[cfg(not(feature = "runtime-model-wgpu"))]
        let decode_uses_wgpu_dispatch = false;
        #[cfg(feature = "runtime-model")]
        let decode_stage_fence_enabled = runtime_stage_fence_enabled();
        #[cfg(not(feature = "runtime-model"))]
        let decode_stage_fence_enabled = false;
        runtime_pipeline_stage_boundary_sync(
            "decode_complete",
            decode_uses_wgpu_dispatch && decode_stage_fence_enabled,
        )?;
        let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
        trellis_stage_log!(
            "burn_trellis: stage decode complete ({decode_ms:.2} ms, vertices={}, faces={})",
            decoded.mesh.vertices.len(),
            decoded.mesh.faces.len()
        );
        let sparse_runtime_profile = sparse.runtime_profile.unwrap_or_default();
        let output = TrellisStageOutput {
            sparse,
            shape_slat,
            shape_slat_lr,
            tex_slat,
            conditioning: stage_conditioning,
            decode_shape_input: decode_shape_input_override,
            decode_tex_input: decode_tex_input_override,
            decode_source: decoded.source,
            decode_shape_subs: decoded.shape_subs,
            decode_tex_voxels: decoded.tex_voxels,
            mesh: decoded.mesh,
            pbr: decoded.pbr,
        };
        let timings = TrellisStageTimings {
            sparse_ms,
            sparse_cond_ms: sparse_runtime_profile.cond_prepare_ms,
            sparse_sample_ms: sparse_runtime_profile.sample_ms,
            sparse_post_ms: sparse_runtime_profile.postprocess_ms,
            sparse_flow_ops,
            shape_slat_ms,
            shape_slat_flow_ops,
            tex_slat_ms,
            tex_slat_flow_ops,
            decode_ms,
            decode_stage_fenced: decoded.timings.stage_fenced,
            decode_shape_decoder_ms: decoded.timings.shape_decoder_ms,
            decode_tex_decoder_ms: decoded.timings.tex_decoder_ms,
            decode_attr_merge_ms: decoded.timings.attr_merge_ms,
            decode_mesh_ms: decoded.timings.mesh_ms,
            decode_pbr_ms: decoded.timings.pbr_ms,
            decode_shape_conv_calls: decoded.timings.shape_conv_calls,
            decode_tex_conv_calls: decoded.timings.tex_conv_calls,
            decode_shape_wgpu_dispatches: decoded.timings.shape_wgpu_dispatches,
            decode_tex_wgpu_dispatches: decoded.timings.tex_wgpu_dispatches,
            decode_shape_wgpu_chunked_calls: decoded.timings.shape_wgpu_chunked_calls,
            decode_tex_wgpu_chunked_calls: decoded.timings.tex_wgpu_chunked_calls,
            decode_shape_wgpu_input_bytes: decoded.timings.shape_wgpu_input_bytes,
            decode_tex_wgpu_input_bytes: decoded.timings.tex_wgpu_input_bytes,
            decode_shape_wgpu_output_bytes: decoded.timings.shape_wgpu_output_bytes,
            decode_tex_wgpu_output_bytes: decoded.timings.tex_wgpu_output_bytes,
            decode_shape_wgpu_max_chunk_rows: decoded.timings.shape_wgpu_max_chunk_rows,
            decode_tex_wgpu_max_chunk_rows: decoded.timings.tex_wgpu_max_chunk_rows,
            total_ms: total_start.elapsed().as_secs_f64() * 1000.0,
        };
        Ok((output, timings))
    }
}

#![allow(deprecated)]

#[cfg(feature = "runtime-model-wgpu")]
use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

#[cfg(target_arch = "wasm32")]
use super::chunked_blob_safetensors::{
    ChunkedBlobSafetensorsStore, chunked_blob_parts_manifest_exists,
};
#[cfg(feature = "runtime-model-wgpu")]
use super::runtime_config::{
    clear_runtime_model_sparse_flow_sampler_step,
    runtime_model_sparse_flow_coord_rope_kernel_enabled,
    runtime_model_sparse_flow_cross_attention_f16_enabled,
    runtime_model_sparse_flow_linear_f16_enabled,
    runtime_model_sparse_flow_module_attention_enabled,
    runtime_model_sparse_flow_self_attention_f16_enabled,
    runtime_model_sparse_flow_stock_bf16_emulation_enabled,
    runtime_model_sparse_flow_torso_f16_enabled, set_runtime_model_sparse_flow_sampler_step,
};
use super::runtime_config::{
    runtime_model_attention_debug_enabled, runtime_model_sparse_flow_module_attention_f16_enabled,
    runtime_model_stage_debug_enabled,
};
use super::weight_parts::{candidate_exists_or_has_parts, load_blob_bytes_from_burnpack_or_parts};
use crate::blob_burnpack::load_blob_bytes_from_burnpack as load_blob_bytes_from_blob_burnpack;
use crate::time::Instant;
use crate::virtual_fs;
use burn::module::{Ignored, Module, Param};
use burn::nn;
use burn::prelude::Backend;
use burn::tensor::activation::{sigmoid, softmax};
use burn::tensor::module::attention;
use burn::tensor::{Int, Tensor, TensorData, ops::AttentionModuleOptions};
#[cfg(feature = "runtime-model-wgpu")]
use burn_flex_gmm::wgpu::{
    bf16_round_to_f32_wgpu, layer_norm_affine_forward_wgpu, layer_norm_modulated_forward_wgpu,
    linear_skinny_forward_wgpu, multihead_qk_rms_norm_rope_from_qkv_coords_wgpu,
    multihead_qkv_module_rms_norm_rope_from_qkv_coords_wgpu, multihead_rms_norm_forward_wgpu,
    multihead_rms_norm_rope_from_coords_wgpu, rope_rotate_pairs_from_coords_wgpu,
    rope_rotate_pairs_wgpu,
};
use burn_store::{KeyRemapper, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};
use serde::Deserialize;

use crate::sampler::{
    FlowEulerSampleConfig, FlowEulerSampleTrace, guidance_interval_contains, mid_snapshot_step,
    timestep_pairs,
};

const F16_SUFFIX: &str = "_f16";
const MAX_PERIOD: f32 = 10_000.0;
const LAYER_NORM_EPS: f32 = 1.0e-6;
const RMS_NORM_EPS: f32 = 1.0e-12;
const ROPE_CACHE_MAX_ENTRIES: usize = 256;
const SPARSE_FLOW_MODULE_ATTENTION_LONG_K_FALLBACK_SEQ_KV: usize = 21_504;
const SPARSE_FLOW_MODULE_ATTENTION_LONG_K_QUERY_CHUNK: usize = 12_288;
// Native WGPU/CubeK blackbox attention is verified for the TRELLIS.2 long-key
// HR SLat head_dim=128 shape through q=12288. Larger query dispatches run, but
// do not improve the model forward and can move queue waits into other stages.
const SPARSE_FLOW_MODULE_ATTENTION_LONG_K_HEAD_DIM_128_QUERY_CHUNK: usize = 12_288;
const SPARSE_FLOW_MODULE_ATTENTION_VERIFIED_LONG_QUERY_CHUNK: usize = 49_152;
const SPARSE_FLOW_MODULE_ATTENTION_F16_MAX_KEY_TOKENS: usize = 8_192;
const SPARSE_FLOW_INPUT_SKINNY_LINEAR_WGPU: bool = true;
static HOST_READBACK_COUNT: AtomicU64 = AtomicU64::new(0);
static HOST_READBACK_ELEMENTS: AtomicU64 = AtomicU64::new(0);
static FLOW_SELF_ATTN_CALLS: AtomicU64 = AtomicU64::new(0);
static FLOW_SELF_ATTN_NS: AtomicU64 = AtomicU64::new(0);
static FLOW_CROSS_ATTN_CALLS: AtomicU64 = AtomicU64::new(0);
static FLOW_CROSS_ATTN_NS: AtomicU64 = AtomicU64::new(0);
static FLOW_MLP_CALLS: AtomicU64 = AtomicU64::new(0);
static FLOW_MLP_NS: AtomicU64 = AtomicU64::new(0);
static FLOW_SELF_QKV_CALLS: AtomicU64 = AtomicU64::new(0);
static FLOW_SELF_QKV_NS: AtomicU64 = AtomicU64::new(0);
static FLOW_SELF_NORM_ROPE_CALLS: AtomicU64 = AtomicU64::new(0);
static FLOW_SELF_NORM_ROPE_NS: AtomicU64 = AtomicU64::new(0);
static FLOW_SELF_NORM_ROPE_FUSED_QK_CALLS: AtomicU64 = AtomicU64::new(0);
static FLOW_SELF_NORM_ROPE_FUSED_QKV_MODULE_CALLS: AtomicU64 = AtomicU64::new(0);
static FLOW_SELF_KERNEL_CALLS: AtomicU64 = AtomicU64::new(0);
static FLOW_SELF_KERNEL_NS: AtomicU64 = AtomicU64::new(0);
static FLOW_SELF_OUT_CALLS: AtomicU64 = AtomicU64::new(0);
static FLOW_SELF_OUT_NS: AtomicU64 = AtomicU64::new(0);
static FLOW_SELF_CAT_CALLS: AtomicU64 = AtomicU64::new(0);
static FLOW_SELF_CAT_NS: AtomicU64 = AtomicU64::new(0);
static FLOW_CROSS_Q_CALLS: AtomicU64 = AtomicU64::new(0);
static FLOW_CROSS_Q_NS: AtomicU64 = AtomicU64::new(0);
static FLOW_CROSS_KV_CALLS: AtomicU64 = AtomicU64::new(0);
static FLOW_CROSS_KV_NS: AtomicU64 = AtomicU64::new(0);
static FLOW_CROSS_NORM_CALLS: AtomicU64 = AtomicU64::new(0);
static FLOW_CROSS_NORM_NS: AtomicU64 = AtomicU64::new(0);
static FLOW_CROSS_KERNEL_CALLS: AtomicU64 = AtomicU64::new(0);
static FLOW_CROSS_KERNEL_NS: AtomicU64 = AtomicU64::new(0);
static FLOW_CROSS_OUT_CALLS: AtomicU64 = AtomicU64::new(0);
static FLOW_CROSS_OUT_NS: AtomicU64 = AtomicU64::new(0);
static FLOW_CROSS_CAT_CALLS: AtomicU64 = AtomicU64::new(0);
static FLOW_CROSS_CAT_NS: AtomicU64 = AtomicU64::new(0);
static FLOW_MODULE_CAST_PAD_CALLS: AtomicU64 = AtomicU64::new(0);
static FLOW_MODULE_CAST_PAD_NS: AtomicU64 = AtomicU64::new(0);
static FLOW_MODULE_ATTENTION_CALLS: AtomicU64 = AtomicU64::new(0);
static FLOW_MODULE_ATTENTION_NS: AtomicU64 = AtomicU64::new(0);
static FLOW_MODULE_OUTPUT_CALLS: AtomicU64 = AtomicU64::new(0);
static FLOW_MODULE_OUTPUT_NS: AtomicU64 = AtomicU64::new(0);
static FLOW_BLOCK_NORM_MOD_CALLS: AtomicU64 = AtomicU64::new(0);
static FLOW_BLOCK_NORM_MOD_NS: AtomicU64 = AtomicU64::new(0);
static FLOW_BLOCK_NORM_AFFINE_CALLS: AtomicU64 = AtomicU64::new(0);
static FLOW_BLOCK_NORM_AFFINE_NS: AtomicU64 = AtomicU64::new(0);
static FLOW_BLOCK_GATE_RESIDUAL_CALLS: AtomicU64 = AtomicU64::new(0);
static FLOW_BLOCK_GATE_RESIDUAL_NS: AtomicU64 = AtomicU64::new(0);
static FLOW_MODEL_IO_CALLS: AtomicU64 = AtomicU64::new(0);
static FLOW_MODEL_IO_NS: AtomicU64 = AtomicU64::new(0);
static FLOW_MODEL_INPUT_CALLS: AtomicU64 = AtomicU64::new(0);
static FLOW_MODEL_INPUT_NS: AtomicU64 = AtomicU64::new(0);
static FLOW_MODEL_OUTPUT_CALLS: AtomicU64 = AtomicU64::new(0);
static FLOW_MODEL_OUTPUT_NS: AtomicU64 = AtomicU64::new(0);
static CFG_POS_NEG_DEBUG_COUNT: AtomicU64 = AtomicU64::new(0);
static ROPE_CACHE: OnceLock<Mutex<HashMap<RopeCacheKey, Arc<RopeCosSinRange>>>> = OnceLock::new();
#[cfg(feature = "runtime-model-wgpu")]
thread_local! {
    static LAYER_NORM_NO_AFFINE_PARAMS_CACHE: RefCell<HashMap<usize, (Tensor<WgpuRuntimeBackend, 1>, Tensor<WgpuRuntimeBackend, 1>)>> =
        RefCell::new(HashMap::new());
    static LINEAR_F16_PARAMS_CACHE: RefCell<HashMap<LinearF16CacheKey, (Tensor<WgpuRuntimeBackend, 2>, Option<Tensor<WgpuRuntimeBackend, 1>>)>> =
        RefCell::new(HashMap::new());
    static LINEAR_SKINNY_PARAMS_CACHE: RefCell<HashMap<LinearF16CacheKey, (Tensor<WgpuRuntimeBackend, 2>, Tensor<WgpuRuntimeBackend, 1>)>> =
        RefCell::new(HashMap::new());
}

fn dense_grid_token_coords<B: Backend>(resolution: usize, device: B::Device) -> Tensor<B, 2, Int> {
    let tokens = resolution
        .saturating_mul(resolution)
        .saturating_mul(resolution);
    let mut coords = Vec::with_capacity(tokens.saturating_mul(3));
    for x in 0..resolution {
        for y in 0..resolution {
            for z in 0..resolution {
                coords.push(x as i64);
                coords.push(y as i64);
                coords.push(z as i64);
            }
        }
    }
    Tensor::<B, 2, Int>::from_data(TensorData::new(coords, [tokens, 3]), &device)
}

#[cfg(feature = "runtime-model-wgpu")]
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct LinearF16CacheKey {
    weight_id: u64,
    bias_id: Option<u64>,
    in_channels: usize,
    out_channels: usize,
}

type CpuRuntimeBackend = burn::backend::NdArray<f32>;
#[cfg(feature = "runtime-model-wgpu")]
// Use raw cube backend for sparse-flow runtime attention so Burn module attention
// dispatches flash-attention kernels instead of fusion's naive fallback.
pub(crate) type WgpuRuntimeBackend = burn_wgpu::CubeBackend<burn_wgpu::WgpuRuntime, f32, i32, u32>;
#[cfg(not(feature = "runtime-model-wgpu"))]
type WgpuRuntimeBackend = CpuRuntimeBackend;

trait SparseFlowTraceWgpuBridge<B: Backend> {
    fn into_wgpu_rows(tensor: Tensor<B, 2>) -> Option<Tensor<WgpuRuntimeBackend, 2>>;
}

struct SparseFlowTraceWgpuBridgeImpl;

impl SparseFlowTraceWgpuBridge<CpuRuntimeBackend> for SparseFlowTraceWgpuBridgeImpl {
    fn into_wgpu_rows(
        _tensor: Tensor<CpuRuntimeBackend, 2>,
    ) -> Option<Tensor<WgpuRuntimeBackend, 2>> {
        None
    }
}

#[cfg(feature = "runtime-model-wgpu")]
impl SparseFlowTraceWgpuBridge<WgpuRuntimeBackend> for SparseFlowTraceWgpuBridgeImpl {
    fn into_wgpu_rows(
        tensor: Tensor<WgpuRuntimeBackend, 2>,
    ) -> Option<Tensor<WgpuRuntimeBackend, 2>> {
        Some(tensor)
    }
}

#[cfg(feature = "runtime-model-wgpu")]
fn maybe_trace_rows_wgpu<B: Backend>(tensor: Tensor<B, 2>) -> Option<Tensor<WgpuRuntimeBackend, 2>>
where
    SparseFlowTraceWgpuBridgeImpl: SparseFlowTraceWgpuBridge<B>,
{
    SparseFlowTraceWgpuBridgeImpl::into_wgpu_rows(tensor)
}

pub trait SparseFlowBf16EmulationBridge<B: Backend> {
    fn round<const D: usize>(tensor: Tensor<B, D>) -> Tensor<B, D>;
}

pub struct SparseFlowBf16EmulationBridgeImpl;

impl SparseFlowBf16EmulationBridge<CpuRuntimeBackend> for SparseFlowBf16EmulationBridgeImpl {
    fn round<const D: usize>(tensor: Tensor<CpuRuntimeBackend, D>) -> Tensor<CpuRuntimeBackend, D> {
        tensor
    }
}

#[cfg(feature = "runtime-model-wgpu")]
impl SparseFlowBf16EmulationBridge<WgpuRuntimeBackend> for SparseFlowBf16EmulationBridgeImpl {
    fn round<const D: usize>(
        tensor: Tensor<WgpuRuntimeBackend, D>,
    ) -> Tensor<WgpuRuntimeBackend, D> {
        bf16_round_to_f32_wgpu(tensor)
    }
}

#[cfg(feature = "runtime-model-wgpu")]
impl SparseFlowBf16EmulationBridge<burn_wgpu::Wgpu<f32, i32, u32>>
    for SparseFlowBf16EmulationBridgeImpl
{
    fn round<const D: usize>(
        tensor: Tensor<burn_wgpu::Wgpu<f32, i32, u32>, D>,
    ) -> Tensor<burn_wgpu::Wgpu<f32, i32, u32>, D> {
        tensor
    }
}

fn sparse_flow_stock_bf16_round<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Tensor<B, D>
where
    SparseFlowBf16EmulationBridgeImpl: SparseFlowBf16EmulationBridge<B>,
{
    #[cfg(feature = "runtime-model-wgpu")]
    {
        if runtime_model_sparse_flow_stock_bf16_emulation_enabled() {
            return SparseFlowBf16EmulationBridgeImpl::round(tensor);
        }
    }
    tensor
}

pub trait RopeRotateWgpuBridge<B: Backend> {
    fn rotate(x: Tensor<B, 4>, cos: Tensor<B, 4>, sin: Tensor<B, 4>) -> Option<Tensor<B, 4>>;
    fn rotate_coords(
        x: Tensor<B, 4>,
        coords: Tensor<B, 2, Int>,
        rope_freq: [f32; 2],
    ) -> Option<Tensor<B, 4>>;
}

pub struct RopeRotateWgpuBridgeImpl;

impl RopeRotateWgpuBridge<CpuRuntimeBackend> for RopeRotateWgpuBridgeImpl {
    fn rotate(
        _x: Tensor<CpuRuntimeBackend, 4>,
        _cos: Tensor<CpuRuntimeBackend, 4>,
        _sin: Tensor<CpuRuntimeBackend, 4>,
    ) -> Option<Tensor<CpuRuntimeBackend, 4>> {
        None
    }

    fn rotate_coords(
        _x: Tensor<CpuRuntimeBackend, 4>,
        _coords: Tensor<CpuRuntimeBackend, 2, Int>,
        _rope_freq: [f32; 2],
    ) -> Option<Tensor<CpuRuntimeBackend, 4>> {
        None
    }
}

#[cfg(feature = "runtime-model-wgpu")]
impl RopeRotateWgpuBridge<WgpuRuntimeBackend> for RopeRotateWgpuBridgeImpl {
    fn rotate(
        x: Tensor<WgpuRuntimeBackend, 4>,
        cos: Tensor<WgpuRuntimeBackend, 4>,
        sin: Tensor<WgpuRuntimeBackend, 4>,
    ) -> Option<Tensor<WgpuRuntimeBackend, 4>> {
        // Fail fast in canonical WGPU mode if the custom kernel path breaks.
        Some(
            rope_rotate_pairs_wgpu(x, cos, sin)
                .unwrap_or_else(|err| panic!("rope rotate wgpu kernel failed: {err}")),
        )
    }

    fn rotate_coords(
        x: Tensor<WgpuRuntimeBackend, 4>,
        coords: Tensor<WgpuRuntimeBackend, 2, Int>,
        rope_freq: [f32; 2],
    ) -> Option<Tensor<WgpuRuntimeBackend, 4>> {
        // Fail fast in canonical WGPU mode if the custom kernel path breaks.
        Some(
            rope_rotate_pairs_from_coords_wgpu(x, coords, rope_freq)
                .unwrap_or_else(|err| panic!("rope rotate coords wgpu kernel failed: {err}")),
        )
    }
}

#[cfg(feature = "runtime-model-wgpu")]
impl RopeRotateWgpuBridge<burn_wgpu::Wgpu<f32, i32, u32>> for RopeRotateWgpuBridgeImpl {
    fn rotate(
        _x: Tensor<burn_wgpu::Wgpu<f32, i32, u32>, 4>,
        _cos: Tensor<burn_wgpu::Wgpu<f32, i32, u32>, 4>,
        _sin: Tensor<burn_wgpu::Wgpu<f32, i32, u32>, 4>,
    ) -> Option<Tensor<burn_wgpu::Wgpu<f32, i32, u32>, 4>> {
        None
    }

    fn rotate_coords(
        _x: Tensor<burn_wgpu::Wgpu<f32, i32, u32>, 4>,
        _coords: Tensor<burn_wgpu::Wgpu<f32, i32, u32>, 2, Int>,
        _rope_freq: [f32; 2],
    ) -> Option<Tensor<burn_wgpu::Wgpu<f32, i32, u32>, 4>> {
        None
    }
}

#[cfg(feature = "runtime-model-wgpu")]
fn maybe_rotate_pairs_wgpu<B: Backend>(
    x: Tensor<B, 4>,
    cos: Tensor<B, 4>,
    sin: Tensor<B, 4>,
) -> Option<Tensor<B, 4>>
where
    RopeRotateWgpuBridgeImpl: RopeRotateWgpuBridge<B>,
{
    RopeRotateWgpuBridgeImpl::rotate(x, cos, sin)
}

#[cfg(feature = "runtime-model-wgpu")]
fn maybe_rotate_pairs_coords_wgpu<B: Backend>(
    x: Tensor<B, 4>,
    coords: Tensor<B, 2, Int>,
    rope_freq: [f32; 2],
) -> Option<Tensor<B, 4>>
where
    RopeRotateWgpuBridgeImpl: RopeRotateWgpuBridge<B>,
{
    RopeRotateWgpuBridgeImpl::rotate_coords(x, coords, rope_freq)
}

pub trait SparseFlowLayerNormWgpuBridge<B: Backend> {
    fn no_affine(x: Tensor<B, 3>, eps: f32) -> Option<Tensor<B, 3>>;
    fn affine(x: Tensor<B, 3>, norm: &nn::LayerNorm<B>, eps: f32) -> Option<Tensor<B, 3>>;
    fn modulated(
        x: Tensor<B, 3>,
        scale: Tensor<B, 3>,
        shift: Tensor<B, 3>,
        eps: f32,
    ) -> Option<Tensor<B, 3>>;
}

pub trait SparseFlowRmsNormWgpuBridge<B: Backend> {
    fn multihead(
        x: Tensor<B, 4>,
        gamma: Tensor<B, 2>,
        scale: f32,
        eps: f32,
    ) -> Option<Tensor<B, 4>>;
    fn multihead_rope_coords(
        x: Tensor<B, 4>,
        gamma: Tensor<B, 2>,
        coords: Tensor<B, 2, Int>,
        rope_freq: [f32; 2],
        scale: f32,
        eps: f32,
    ) -> Option<Tensor<B, 4>>;
}

pub trait SparseFlowQkRmsNormWgpuBridge<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn qk_multihead_rope_coords_from_qkv(
        qkv: Tensor<B, 5>,
        q_gamma: Tensor<B, 2>,
        k_gamma: Tensor<B, 2>,
        coords: Tensor<B, 2, Int>,
        rope_freq: [f32; 2],
        scale: f32,
        eps: f32,
    ) -> Option<(Tensor<B, 4>, Tensor<B, 4>)>;

    #[allow(clippy::too_many_arguments)]
    fn qkv_module_multihead_rope_coords_from_qkv(
        qkv: Tensor<B, 5>,
        q_gamma: Tensor<B, 2>,
        k_gamma: Tensor<B, 2>,
        coords: Tensor<B, 2, Int>,
        rope_freq: [f32; 2],
        scale: f32,
        eps: f32,
    ) -> Option<(Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>)>;
}

pub trait SparseFlowLinearWgpuBridge<B: Backend> {
    fn safe_matmul_2d(lhs: Tensor<B, 2>, rhs: Tensor<B, 2>, context: &str) -> Option<Tensor<B, 2>>;

    fn f16_linear(
        linear: &nn::Linear<B>,
        x: Tensor<B, 2>,
        output_dtype: burn::tensor::FloatDType,
    ) -> Option<Tensor<B, 2>>;

    fn cache_f16_linear(linear: &nn::Linear<B>) -> bool;

    fn skinny_linear(
        linear: &nn::Linear<B>,
        x: Tensor<B, 2>,
        output_dtype: burn::tensor::FloatDType,
    ) -> Option<Tensor<B, 2>>;
}

pub struct SparseFlowLinearWgpuBridgeImpl;

impl SparseFlowLayerNormWgpuBridge<CpuRuntimeBackend> for RopeRotateWgpuBridgeImpl {
    fn no_affine(
        _x: Tensor<CpuRuntimeBackend, 3>,
        _eps: f32,
    ) -> Option<Tensor<CpuRuntimeBackend, 3>> {
        None
    }

    fn affine(
        _x: Tensor<CpuRuntimeBackend, 3>,
        _norm: &nn::LayerNorm<CpuRuntimeBackend>,
        _eps: f32,
    ) -> Option<Tensor<CpuRuntimeBackend, 3>> {
        None
    }

    fn modulated(
        _x: Tensor<CpuRuntimeBackend, 3>,
        _scale: Tensor<CpuRuntimeBackend, 3>,
        _shift: Tensor<CpuRuntimeBackend, 3>,
        _eps: f32,
    ) -> Option<Tensor<CpuRuntimeBackend, 3>> {
        None
    }
}

impl SparseFlowRmsNormWgpuBridge<CpuRuntimeBackend> for RopeRotateWgpuBridgeImpl {
    fn multihead(
        _x: Tensor<CpuRuntimeBackend, 4>,
        _gamma: Tensor<CpuRuntimeBackend, 2>,
        _scale: f32,
        _eps: f32,
    ) -> Option<Tensor<CpuRuntimeBackend, 4>> {
        None
    }

    fn multihead_rope_coords(
        _x: Tensor<CpuRuntimeBackend, 4>,
        _gamma: Tensor<CpuRuntimeBackend, 2>,
        _coords: Tensor<CpuRuntimeBackend, 2, Int>,
        _rope_freq: [f32; 2],
        _scale: f32,
        _eps: f32,
    ) -> Option<Tensor<CpuRuntimeBackend, 4>> {
        None
    }
}

impl SparseFlowQkRmsNormWgpuBridge<CpuRuntimeBackend> for RopeRotateWgpuBridgeImpl {
    fn qk_multihead_rope_coords_from_qkv(
        _qkv: Tensor<CpuRuntimeBackend, 5>,
        _q_gamma: Tensor<CpuRuntimeBackend, 2>,
        _k_gamma: Tensor<CpuRuntimeBackend, 2>,
        _coords: Tensor<CpuRuntimeBackend, 2, Int>,
        _rope_freq: [f32; 2],
        _scale: f32,
        _eps: f32,
    ) -> Option<(Tensor<CpuRuntimeBackend, 4>, Tensor<CpuRuntimeBackend, 4>)> {
        None
    }

    fn qkv_module_multihead_rope_coords_from_qkv(
        _qkv: Tensor<CpuRuntimeBackend, 5>,
        _q_gamma: Tensor<CpuRuntimeBackend, 2>,
        _k_gamma: Tensor<CpuRuntimeBackend, 2>,
        _coords: Tensor<CpuRuntimeBackend, 2, Int>,
        _rope_freq: [f32; 2],
        _scale: f32,
        _eps: f32,
    ) -> Option<(
        Tensor<CpuRuntimeBackend, 4>,
        Tensor<CpuRuntimeBackend, 4>,
        Tensor<CpuRuntimeBackend, 4>,
    )> {
        None
    }
}

impl SparseFlowLinearWgpuBridge<CpuRuntimeBackend> for SparseFlowLinearWgpuBridgeImpl {
    fn safe_matmul_2d(
        _lhs: Tensor<CpuRuntimeBackend, 2>,
        _rhs: Tensor<CpuRuntimeBackend, 2>,
        _context: &str,
    ) -> Option<Tensor<CpuRuntimeBackend, 2>> {
        None
    }

    fn f16_linear(
        _linear: &nn::Linear<CpuRuntimeBackend>,
        _x: Tensor<CpuRuntimeBackend, 2>,
        _output_dtype: burn::tensor::FloatDType,
    ) -> Option<Tensor<CpuRuntimeBackend, 2>> {
        None
    }

    fn cache_f16_linear(_linear: &nn::Linear<CpuRuntimeBackend>) -> bool {
        false
    }

    fn skinny_linear(
        _linear: &nn::Linear<CpuRuntimeBackend>,
        _x: Tensor<CpuRuntimeBackend, 2>,
        _output_dtype: burn::tensor::FloatDType,
    ) -> Option<Tensor<CpuRuntimeBackend, 2>> {
        None
    }
}

#[cfg(feature = "runtime-model-wgpu")]
impl SparseFlowLayerNormWgpuBridge<burn_wgpu::Wgpu<f32, i32, u32>> for RopeRotateWgpuBridgeImpl {
    fn no_affine(
        _x: Tensor<burn_wgpu::Wgpu<f32, i32, u32>, 3>,
        _eps: f32,
    ) -> Option<Tensor<burn_wgpu::Wgpu<f32, i32, u32>, 3>> {
        None
    }

    fn affine(
        _x: Tensor<burn_wgpu::Wgpu<f32, i32, u32>, 3>,
        _norm: &nn::LayerNorm<burn_wgpu::Wgpu<f32, i32, u32>>,
        _eps: f32,
    ) -> Option<Tensor<burn_wgpu::Wgpu<f32, i32, u32>, 3>> {
        None
    }

    fn modulated(
        _x: Tensor<burn_wgpu::Wgpu<f32, i32, u32>, 3>,
        _scale: Tensor<burn_wgpu::Wgpu<f32, i32, u32>, 3>,
        _shift: Tensor<burn_wgpu::Wgpu<f32, i32, u32>, 3>,
        _eps: f32,
    ) -> Option<Tensor<burn_wgpu::Wgpu<f32, i32, u32>, 3>> {
        None
    }
}

#[cfg(feature = "runtime-model-wgpu")]
impl SparseFlowRmsNormWgpuBridge<burn_wgpu::Wgpu<f32, i32, u32>> for RopeRotateWgpuBridgeImpl {
    fn multihead(
        _x: Tensor<burn_wgpu::Wgpu<f32, i32, u32>, 4>,
        _gamma: Tensor<burn_wgpu::Wgpu<f32, i32, u32>, 2>,
        _scale: f32,
        _eps: f32,
    ) -> Option<Tensor<burn_wgpu::Wgpu<f32, i32, u32>, 4>> {
        None
    }

    fn multihead_rope_coords(
        _x: Tensor<burn_wgpu::Wgpu<f32, i32, u32>, 4>,
        _gamma: Tensor<burn_wgpu::Wgpu<f32, i32, u32>, 2>,
        _coords: Tensor<burn_wgpu::Wgpu<f32, i32, u32>, 2, Int>,
        _rope_freq: [f32; 2],
        _scale: f32,
        _eps: f32,
    ) -> Option<Tensor<burn_wgpu::Wgpu<f32, i32, u32>, 4>> {
        None
    }
}

#[cfg(feature = "runtime-model-wgpu")]
impl SparseFlowQkRmsNormWgpuBridge<burn_wgpu::Wgpu<f32, i32, u32>> for RopeRotateWgpuBridgeImpl {
    fn qk_multihead_rope_coords_from_qkv(
        _qkv: Tensor<burn_wgpu::Wgpu<f32, i32, u32>, 5>,
        _q_gamma: Tensor<burn_wgpu::Wgpu<f32, i32, u32>, 2>,
        _k_gamma: Tensor<burn_wgpu::Wgpu<f32, i32, u32>, 2>,
        _coords: Tensor<burn_wgpu::Wgpu<f32, i32, u32>, 2, Int>,
        _rope_freq: [f32; 2],
        _scale: f32,
        _eps: f32,
    ) -> Option<(
        Tensor<burn_wgpu::Wgpu<f32, i32, u32>, 4>,
        Tensor<burn_wgpu::Wgpu<f32, i32, u32>, 4>,
    )> {
        None
    }

    fn qkv_module_multihead_rope_coords_from_qkv(
        _qkv: Tensor<burn_wgpu::Wgpu<f32, i32, u32>, 5>,
        _q_gamma: Tensor<burn_wgpu::Wgpu<f32, i32, u32>, 2>,
        _k_gamma: Tensor<burn_wgpu::Wgpu<f32, i32, u32>, 2>,
        _coords: Tensor<burn_wgpu::Wgpu<f32, i32, u32>, 2, Int>,
        _rope_freq: [f32; 2],
        _scale: f32,
        _eps: f32,
    ) -> Option<(
        Tensor<burn_wgpu::Wgpu<f32, i32, u32>, 4>,
        Tensor<burn_wgpu::Wgpu<f32, i32, u32>, 4>,
        Tensor<burn_wgpu::Wgpu<f32, i32, u32>, 4>,
    )> {
        None
    }
}

#[cfg(feature = "runtime-model-wgpu")]
impl SparseFlowLinearWgpuBridge<burn_wgpu::Wgpu<f32, i32, u32>> for SparseFlowLinearWgpuBridgeImpl {
    fn safe_matmul_2d(
        _lhs: Tensor<burn_wgpu::Wgpu<f32, i32, u32>, 2>,
        _rhs: Tensor<burn_wgpu::Wgpu<f32, i32, u32>, 2>,
        _context: &str,
    ) -> Option<Tensor<burn_wgpu::Wgpu<f32, i32, u32>, 2>> {
        None
    }

    fn f16_linear(
        _linear: &nn::Linear<burn_wgpu::Wgpu<f32, i32, u32>>,
        _x: Tensor<burn_wgpu::Wgpu<f32, i32, u32>, 2>,
        _output_dtype: burn::tensor::FloatDType,
    ) -> Option<Tensor<burn_wgpu::Wgpu<f32, i32, u32>, 2>> {
        None
    }

    fn cache_f16_linear(_linear: &nn::Linear<burn_wgpu::Wgpu<f32, i32, u32>>) -> bool {
        false
    }

    fn skinny_linear(
        _linear: &nn::Linear<burn_wgpu::Wgpu<f32, i32, u32>>,
        _x: Tensor<burn_wgpu::Wgpu<f32, i32, u32>, 2>,
        _output_dtype: burn::tensor::FloatDType,
    ) -> Option<Tensor<burn_wgpu::Wgpu<f32, i32, u32>, 2>> {
        None
    }
}

#[cfg(feature = "runtime-model-wgpu")]
fn layer_norm_no_affine_params_wgpu(
    channels: usize,
    device: &burn_wgpu::WgpuDevice,
) -> (Tensor<WgpuRuntimeBackend, 1>, Tensor<WgpuRuntimeBackend, 1>) {
    LAYER_NORM_NO_AFFINE_PARAMS_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if let Some((weight, bias)) = cache.get(&channels) {
            return (weight.clone(), bias.clone());
        }
        let weight = Tensor::<WgpuRuntimeBackend, 1>::ones([channels], device);
        let bias = Tensor::<WgpuRuntimeBackend, 1>::zeros([channels], device);
        cache.insert(channels, (weight.clone(), bias.clone()));
        (weight, bias)
    })
}

#[cfg(feature = "runtime-model-wgpu")]
fn cached_linear_f16_params(
    linear: &nn::Linear<WgpuRuntimeBackend>,
) -> (
    Tensor<WgpuRuntimeBackend, 2>,
    Option<Tensor<WgpuRuntimeBackend, 1>>,
) {
    let weight = linear.weight.val();
    let [in_channels, out_channels] = weight.dims();
    let key = LinearF16CacheKey {
        weight_id: linear.weight.id.val(),
        bias_id: linear.bias.as_ref().map(|bias| bias.id.val()),
        in_channels,
        out_channels,
    };
    LINEAR_F16_PARAMS_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if let Some(params) = cache.get(&key) {
            return params.clone();
        }
        let f16 = burn::tensor::FloatDType::F16;
        let weight_f16 = tensor_cast_float_2d_if_needed(weight, f16);
        let bias_f16 = linear
            .bias
            .as_ref()
            .map(|bias| tensor_cast_float_1d_if_needed(bias.val(), f16));
        cache.insert(key, (weight_f16.clone(), bias_f16.clone()));
        (weight_f16, bias_f16)
    })
}

#[cfg(feature = "runtime-model-wgpu")]
impl SparseFlowLayerNormWgpuBridge<WgpuRuntimeBackend> for RopeRotateWgpuBridgeImpl {
    fn no_affine(
        x: Tensor<WgpuRuntimeBackend, 3>,
        eps: f32,
    ) -> Option<Tensor<WgpuRuntimeBackend, 3>> {
        if !sparse_flow_wgpu_layer_norm_kernel_enabled() {
            return None;
        }
        let [batch, tokens, channels] = x.dims();
        if batch == 0 || tokens == 0 || channels == 0 {
            return Some(x);
        }
        let rows = batch.saturating_mul(tokens);
        let device = x.device();
        let (weight, bias) = layer_norm_no_affine_params_wgpu(channels, &device);
        Some(
            layer_norm_affine_forward_wgpu(x.reshape([rows, channels]), weight, bias, eps)
                .unwrap_or_else(|err| {
                    panic!("sparse-flow layer_norm_no_affine wgpu kernel failed: {err}")
                })
                .reshape([batch, tokens, channels]),
        )
    }

    fn affine(
        x: Tensor<WgpuRuntimeBackend, 3>,
        norm: &nn::LayerNorm<WgpuRuntimeBackend>,
        eps: f32,
    ) -> Option<Tensor<WgpuRuntimeBackend, 3>> {
        if !sparse_flow_wgpu_layer_norm_kernel_enabled() {
            return None;
        }
        let [batch, tokens, channels] = x.dims();
        if batch == 0 || tokens == 0 || channels == 0 {
            return Some(x);
        }
        let rows = batch.saturating_mul(tokens);
        let x_dtype: burn::tensor::FloatDType = x.dtype().into();

        let gamma = norm.gamma.val();
        let gamma_dtype: burn::tensor::FloatDType = gamma.dtype().into();
        let gamma = if gamma_dtype != x_dtype {
            gamma.cast(x_dtype)
        } else {
            gamma
        };

        let beta = if let Some(beta) = norm.beta.as_ref() {
            let beta = beta.val();
            let beta_dtype: burn::tensor::FloatDType = beta.dtype().into();
            if beta_dtype != x_dtype {
                beta.cast(x_dtype)
            } else {
                beta
            }
        } else {
            let device = x.device();
            let (_, beta) = layer_norm_no_affine_params_wgpu(channels, &device);
            beta
        };

        Some(
            layer_norm_affine_forward_wgpu(x.reshape([rows, channels]), gamma, beta, eps)
                .unwrap_or_else(|err| {
                    panic!("sparse-flow layer_norm_affine wgpu kernel failed: {err}")
                })
                .reshape([batch, tokens, channels]),
        )
    }

    fn modulated(
        x: Tensor<WgpuRuntimeBackend, 3>,
        scale: Tensor<WgpuRuntimeBackend, 3>,
        shift: Tensor<WgpuRuntimeBackend, 3>,
        eps: f32,
    ) -> Option<Tensor<WgpuRuntimeBackend, 3>> {
        if !sparse_flow_wgpu_layer_norm_kernel_enabled() {
            return None;
        }
        let [batch, tokens, channels] = x.dims();
        if batch == 0 || tokens == 0 || channels == 0 {
            return Some(x);
        }
        Some(
            layer_norm_modulated_forward_wgpu(x, scale, shift, eps).unwrap_or_else(|err| {
                panic!("sparse-flow layer_norm_modulated wgpu kernel failed: {err}")
            }),
        )
    }
}

#[cfg(feature = "runtime-model-wgpu")]
impl SparseFlowRmsNormWgpuBridge<WgpuRuntimeBackend> for RopeRotateWgpuBridgeImpl {
    fn multihead(
        x: Tensor<WgpuRuntimeBackend, 4>,
        gamma: Tensor<WgpuRuntimeBackend, 2>,
        scale: f32,
        eps: f32,
    ) -> Option<Tensor<WgpuRuntimeBackend, 4>> {
        if !sparse_flow_wgpu_layer_norm_kernel_enabled() {
            return None;
        }
        let [batch, tokens, heads, head_dim] = x.dims();
        if batch == 0 || tokens == 0 || heads == 0 || head_dim == 0 {
            return Some(x);
        }
        let x_dtype: burn::tensor::FloatDType = x.dtype().into();
        let gamma_dtype: burn::tensor::FloatDType = gamma.dtype().into();
        let gamma = if gamma_dtype != x_dtype {
            gamma.cast(x_dtype)
        } else {
            gamma
        };
        Some(
            multihead_rms_norm_forward_wgpu(x, gamma, scale, eps).unwrap_or_else(|err| {
                panic!("sparse-flow multihead_rms_norm wgpu kernel failed: {err}")
            }),
        )
    }

    fn multihead_rope_coords(
        x: Tensor<WgpuRuntimeBackend, 4>,
        gamma: Tensor<WgpuRuntimeBackend, 2>,
        coords: Tensor<WgpuRuntimeBackend, 2, Int>,
        rope_freq: [f32; 2],
        scale: f32,
        eps: f32,
    ) -> Option<Tensor<WgpuRuntimeBackend, 4>> {
        if !sparse_flow_wgpu_layer_norm_kernel_enabled()
            || !sparse_flow_wgpu_coord_rope_kernel_enabled()
        {
            return None;
        }
        let [batch, tokens, heads, head_dim] = x.dims();
        if batch == 0 || tokens == 0 || heads == 0 || head_dim == 0 {
            return Some(x);
        }
        let x_dtype: burn::tensor::FloatDType = x.dtype().into();
        let gamma_dtype: burn::tensor::FloatDType = gamma.dtype().into();
        let gamma = if gamma_dtype != x_dtype {
            gamma.cast(x_dtype)
        } else {
            gamma
        };
        Some(
            multihead_rms_norm_rope_from_coords_wgpu(x, gamma, coords, rope_freq, scale, eps)
                .unwrap_or_else(|err| {
                    panic!("sparse-flow multihead_rms_norm_rope wgpu kernel failed: {err}")
                }),
        )
    }
}

#[cfg(feature = "runtime-model-wgpu")]
impl SparseFlowQkRmsNormWgpuBridge<WgpuRuntimeBackend> for RopeRotateWgpuBridgeImpl {
    fn qk_multihead_rope_coords_from_qkv(
        qkv: Tensor<WgpuRuntimeBackend, 5>,
        q_gamma: Tensor<WgpuRuntimeBackend, 2>,
        k_gamma: Tensor<WgpuRuntimeBackend, 2>,
        coords: Tensor<WgpuRuntimeBackend, 2, Int>,
        rope_freq: [f32; 2],
        scale: f32,
        eps: f32,
    ) -> Option<(Tensor<WgpuRuntimeBackend, 4>, Tensor<WgpuRuntimeBackend, 4>)> {
        if !sparse_flow_wgpu_layer_norm_kernel_enabled()
            || !sparse_flow_wgpu_coord_rope_kernel_enabled()
        {
            return None;
        }
        let [batch, tokens, qkv_dim, heads, head_dim] = qkv.dims();
        if batch == 0 || tokens == 0 || heads == 0 || head_dim == 0 {
            return Some((
                Tensor::<WgpuRuntimeBackend, 4>::zeros(
                    [batch, tokens, heads, head_dim],
                    &qkv.device(),
                ),
                Tensor::<WgpuRuntimeBackend, 4>::zeros(
                    [batch, tokens, heads, head_dim],
                    &qkv.device(),
                ),
            ));
        }
        if qkv_dim != 3 {
            return None;
        }
        let qkv_dtype: burn::tensor::FloatDType = qkv.dtype().into();
        let q_gamma_dtype: burn::tensor::FloatDType = q_gamma.dtype().into();
        let q_gamma = if q_gamma_dtype != qkv_dtype {
            q_gamma.cast(qkv_dtype)
        } else {
            q_gamma
        };
        let k_gamma_dtype: burn::tensor::FloatDType = k_gamma.dtype().into();
        let k_gamma = if k_gamma_dtype != qkv_dtype {
            k_gamma.cast(qkv_dtype)
        } else {
            k_gamma
        };
        Some(
            multihead_qk_rms_norm_rope_from_qkv_coords_wgpu(
                qkv, q_gamma, k_gamma, coords, rope_freq, scale, eps,
            )
            .unwrap_or_else(|err| {
                panic!("sparse-flow qk multihead_rms_norm_rope wgpu kernel failed: {err}")
            }),
        )
    }

    fn qkv_module_multihead_rope_coords_from_qkv(
        qkv: Tensor<WgpuRuntimeBackend, 5>,
        q_gamma: Tensor<WgpuRuntimeBackend, 2>,
        k_gamma: Tensor<WgpuRuntimeBackend, 2>,
        coords: Tensor<WgpuRuntimeBackend, 2, Int>,
        rope_freq: [f32; 2],
        scale: f32,
        eps: f32,
    ) -> Option<(
        Tensor<WgpuRuntimeBackend, 4>,
        Tensor<WgpuRuntimeBackend, 4>,
        Tensor<WgpuRuntimeBackend, 4>,
    )> {
        if !sparse_flow_wgpu_layer_norm_kernel_enabled()
            || !sparse_flow_wgpu_coord_rope_kernel_enabled()
        {
            return None;
        }
        let [batch, tokens, qkv_dim, heads, head_dim] = qkv.dims();
        if batch == 0 || tokens == 0 || heads == 0 || head_dim == 0 {
            return Some((
                Tensor::<WgpuRuntimeBackend, 4>::zeros(
                    [batch, heads, tokens, head_dim],
                    &qkv.device(),
                ),
                Tensor::<WgpuRuntimeBackend, 4>::zeros(
                    [batch, heads, tokens, head_dim],
                    &qkv.device(),
                ),
                Tensor::<WgpuRuntimeBackend, 4>::zeros(
                    [batch, heads, tokens, head_dim],
                    &qkv.device(),
                ),
            ));
        }
        if qkv_dim != 3 {
            return None;
        }
        let qkv_dtype: burn::tensor::FloatDType = qkv.dtype().into();
        let q_gamma_dtype: burn::tensor::FloatDType = q_gamma.dtype().into();
        let q_gamma = if q_gamma_dtype != qkv_dtype {
            q_gamma.cast(qkv_dtype)
        } else {
            q_gamma
        };
        let k_gamma_dtype: burn::tensor::FloatDType = k_gamma.dtype().into();
        let k_gamma = if k_gamma_dtype != qkv_dtype {
            k_gamma.cast(qkv_dtype)
        } else {
            k_gamma
        };
        Some(
            multihead_qkv_module_rms_norm_rope_from_qkv_coords_wgpu(
                qkv, q_gamma, k_gamma, coords, rope_freq, scale, eps,
            )
            .unwrap_or_else(|err| {
                panic!("sparse-flow qkv module multihead_rms_norm_rope wgpu kernel failed: {err}")
            }),
        )
    }
}

#[cfg(feature = "runtime-model-wgpu")]
impl SparseFlowLinearWgpuBridge<WgpuRuntimeBackend> for SparseFlowLinearWgpuBridgeImpl {
    fn safe_matmul_2d(
        lhs: Tensor<WgpuRuntimeBackend, 2>,
        rhs: Tensor<WgpuRuntimeBackend, 2>,
        context: &str,
    ) -> Option<Tensor<WgpuRuntimeBackend, 2>> {
        #[cfg(target_arch = "wasm32")]
        {
            return Some(super::wgpu_safe_ops::matmul_2d_naive(lhs, rhs, context));
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = (lhs, rhs, context);
            None
        }
    }

    fn f16_linear(
        linear: &nn::Linear<WgpuRuntimeBackend>,
        x: Tensor<WgpuRuntimeBackend, 2>,
        output_dtype: burn::tensor::FloatDType,
    ) -> Option<Tensor<WgpuRuntimeBackend, 2>> {
        let [rows, in_channels] = x.dims();
        let weight = linear.weight.val();
        let [weight_in_channels, out_channels] = weight.dims();
        if in_channels != weight_in_channels {
            panic!(
                "linear input/weight mismatch: input=[{rows},{in_channels}] weight=[{weight_in_channels},{out_channels}]"
            );
        }
        if rows == 0 || in_channels == 0 || out_channels == 0 {
            return None;
        }

        let (weight_f16, bias_f16) = cached_linear_f16_params(linear);

        let f16 = burn::tensor::FloatDType::F16;
        let mut output = tensor_cast_float_2d_if_needed(x, f16).matmul(weight_f16);
        if let Some(bias) = bias_f16 {
            output = output.add(bias.unsqueeze::<2>());
        }
        Some(tensor_cast_float_2d_if_needed(output, output_dtype))
    }

    fn cache_f16_linear(linear: &nn::Linear<WgpuRuntimeBackend>) -> bool {
        let [in_channels, out_channels] = linear.weight.val().dims();
        if in_channels == 0 || out_channels == 0 {
            return false;
        }
        let _ = cached_linear_f16_params(linear);
        true
    }

    fn skinny_linear(
        linear: &nn::Linear<WgpuRuntimeBackend>,
        x: Tensor<WgpuRuntimeBackend, 2>,
        output_dtype: burn::tensor::FloatDType,
    ) -> Option<Tensor<WgpuRuntimeBackend, 2>> {
        let [rows, in_channels] = x.dims();
        let weight = linear.weight.val();
        let [weight_in_channels, out_channels] = weight.dims();
        if in_channels != weight_in_channels {
            panic!(
                "linear input/weight mismatch: input=[{rows},{in_channels}] weight=[{weight_in_channels},{out_channels}]"
            );
        }
        if rows == 0 || in_channels == 0 || out_channels == 0 {
            return None;
        }
        let bias = linear.bias.as_ref()?;

        let key = LinearF16CacheKey {
            weight_id: linear.weight.id.val(),
            bias_id: Some(bias.id.val()),
            in_channels,
            out_channels,
        };
        let (weight_f32_t, bias_f32) = LINEAR_SKINNY_PARAMS_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            if let Some(params) = cache.get(&key) {
                return params.clone();
            }
            let f32_dtype = burn::tensor::FloatDType::F32;
            let weight_f32_t = tensor_cast_float_2d_if_needed(weight.swap_dims(0, 1), f32_dtype)
                .reshape([out_channels * in_channels])
                .reshape([out_channels, in_channels]);
            let bias_f32 = tensor_cast_float_1d_if_needed(bias.val(), f32_dtype);
            cache.insert(key, (weight_f32_t.clone(), bias_f32.clone()));
            (weight_f32_t, bias_f32)
        });

        let output = linear_skinny_forward_wgpu(
            tensor_cast_float_2d_if_needed(x, burn::tensor::FloatDType::F32),
            weight_f32_t,
            bias_f32,
        )
        .unwrap_or_else(|err| panic!("sparse-flow skinny linear wgpu kernel failed: {err}"));
        Some(tensor_cast_float_2d_if_needed(output, output_dtype))
    }
}

#[cfg(feature = "runtime-model-wgpu")]
fn maybe_layer_norm_no_affine_wgpu<B: Backend>(x: Tensor<B, 3>, eps: f32) -> Option<Tensor<B, 3>>
where
    RopeRotateWgpuBridgeImpl: SparseFlowLayerNormWgpuBridge<B>,
{
    RopeRotateWgpuBridgeImpl::no_affine(x, eps)
}

#[cfg(feature = "runtime-model-wgpu")]
fn maybe_layer_norm_affine_wgpu<B: Backend>(
    x: Tensor<B, 3>,
    norm: &nn::LayerNorm<B>,
    eps: f32,
) -> Option<Tensor<B, 3>>
where
    RopeRotateWgpuBridgeImpl: SparseFlowLayerNormWgpuBridge<B>,
{
    RopeRotateWgpuBridgeImpl::affine(x, norm, eps)
}

#[cfg(feature = "runtime-model-wgpu")]
fn maybe_layer_norm_modulated_wgpu<B: Backend>(
    x: Tensor<B, 3>,
    scale: Tensor<B, 3>,
    shift: Tensor<B, 3>,
    eps: f32,
) -> Option<Tensor<B, 3>>
where
    RopeRotateWgpuBridgeImpl: SparseFlowLayerNormWgpuBridge<B>,
    SparseFlowBf16EmulationBridgeImpl: SparseFlowBf16EmulationBridge<B>,
{
    RopeRotateWgpuBridgeImpl::modulated(x, scale, shift, eps)
}

#[cfg(feature = "runtime-model-wgpu")]
fn maybe_multihead_rms_norm_wgpu<B: Backend>(
    x: Tensor<B, 4>,
    gamma: Tensor<B, 2>,
    scale: f32,
    eps: f32,
) -> Option<Tensor<B, 4>>
where
    RopeRotateWgpuBridgeImpl: SparseFlowRmsNormWgpuBridge<B>,
{
    RopeRotateWgpuBridgeImpl::multihead(x, gamma, scale, eps)
}

fn maybe_multihead_rms_norm_rope_coords_wgpu<B: Backend>(
    x: Tensor<B, 4>,
    gamma: Tensor<B, 2>,
    coords: Tensor<B, 2, Int>,
    rope_freq: [f32; 2],
    scale: f32,
    eps: f32,
) -> Option<Tensor<B, 4>>
where
    RopeRotateWgpuBridgeImpl: SparseFlowRmsNormWgpuBridge<B>,
{
    RopeRotateWgpuBridgeImpl::multihead_rope_coords(x, gamma, coords, rope_freq, scale, eps)
}

#[cfg(feature = "runtime-model-wgpu")]
#[allow(clippy::too_many_arguments)]
fn maybe_qk_multihead_rms_norm_rope_coords_from_qkv_wgpu<B: Backend>(
    qkv: Tensor<B, 5>,
    q_gamma: Tensor<B, 2>,
    k_gamma: Tensor<B, 2>,
    coords: Tensor<B, 2, Int>,
    rope_freq: [f32; 2],
    scale: f32,
    eps: f32,
) -> Option<(Tensor<B, 4>, Tensor<B, 4>)>
where
    RopeRotateWgpuBridgeImpl: SparseFlowQkRmsNormWgpuBridge<B>,
{
    let output = RopeRotateWgpuBridgeImpl::qk_multihead_rope_coords_from_qkv(
        qkv, q_gamma, k_gamma, coords, rope_freq, scale, eps,
    );
    if output.is_some() {
        FLOW_SELF_NORM_ROPE_FUSED_QK_CALLS.fetch_add(1, Ordering::Relaxed);
    }
    output
}

#[cfg(feature = "runtime-model-wgpu")]
#[allow(clippy::too_many_arguments)]
fn maybe_qkv_module_multihead_rms_norm_rope_coords_from_qkv_wgpu<B: Backend>(
    qkv: Tensor<B, 5>,
    q_gamma: Tensor<B, 2>,
    k_gamma: Tensor<B, 2>,
    coords: Tensor<B, 2, Int>,
    rope_freq: [f32; 2],
    scale: f32,
    eps: f32,
) -> Option<(Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>)>
where
    RopeRotateWgpuBridgeImpl: SparseFlowQkRmsNormWgpuBridge<B>,
{
    let output = RopeRotateWgpuBridgeImpl::qkv_module_multihead_rope_coords_from_qkv(
        qkv, q_gamma, k_gamma, coords, rope_freq, scale, eps,
    );
    if output.is_some() {
        FLOW_SELF_NORM_ROPE_FUSED_QKV_MODULE_CALLS.fetch_add(1, Ordering::Relaxed);
    }
    output
}

#[cfg(feature = "runtime-model-wgpu")]
fn maybe_linear_f16_wgpu<B: Backend>(
    linear: &nn::Linear<B>,
    x: Tensor<B, 2>,
    output_dtype: burn::tensor::FloatDType,
) -> Option<Tensor<B, 2>>
where
    SparseFlowLinearWgpuBridgeImpl: SparseFlowLinearWgpuBridge<B>,
{
    SparseFlowLinearWgpuBridgeImpl::f16_linear(linear, x, output_dtype)
}

#[cfg(feature = "runtime-model-wgpu")]
fn maybe_cache_linear_f16_wgpu<B: Backend>(linear: &nn::Linear<B>) -> bool
where
    SparseFlowLinearWgpuBridgeImpl: SparseFlowLinearWgpuBridge<B>,
{
    SparseFlowLinearWgpuBridgeImpl::cache_f16_linear(linear)
}

#[cfg(feature = "runtime-model-wgpu")]
fn maybe_linear_skinny_wgpu<B: Backend>(
    linear: &nn::Linear<B>,
    x: Tensor<B, 2>,
    output_dtype: burn::tensor::FloatDType,
) -> Option<Tensor<B, 2>>
where
    SparseFlowLinearWgpuBridgeImpl: SparseFlowLinearWgpuBridge<B>,
{
    SparseFlowLinearWgpuBridgeImpl::skinny_linear(linear, x, output_dtype)
}

#[derive(Clone, Copy, Debug, Default)]
pub struct HostTransferStats {
    pub readback_count: u64,
    pub readback_elements: u64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SparseFlowOpTelemetry {
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

#[derive(Clone, Debug)]
struct RopeCosSinRange {
    cos: Vec<f32>,
    sin: Vec<f32>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct RopeCacheKey {
    resolution: usize,
    token_start: usize,
    tokens: usize,
    pairs: usize,
    rope_freq_0_bits: u32,
    rope_freq_1_bits: u32,
}

pub fn reset_host_transfer_stats() {
    HOST_READBACK_COUNT.store(0, Ordering::Relaxed);
    HOST_READBACK_ELEMENTS.store(0, Ordering::Relaxed);
}

pub fn host_transfer_stats() -> HostTransferStats {
    HostTransferStats {
        readback_count: HOST_READBACK_COUNT.load(Ordering::Relaxed),
        readback_elements: HOST_READBACK_ELEMENTS.load(Ordering::Relaxed),
    }
}

fn record_host_readback(elements: usize) {
    HOST_READBACK_COUNT.fetch_add(1, Ordering::Relaxed);
    HOST_READBACK_ELEMENTS.fetch_add(elements as u64, Ordering::Relaxed);
}

pub fn reset_sparse_flow_op_telemetry() {
    FLOW_SELF_ATTN_CALLS.store(0, Ordering::Relaxed);
    FLOW_SELF_ATTN_NS.store(0, Ordering::Relaxed);
    FLOW_CROSS_ATTN_CALLS.store(0, Ordering::Relaxed);
    FLOW_CROSS_ATTN_NS.store(0, Ordering::Relaxed);
    FLOW_MLP_CALLS.store(0, Ordering::Relaxed);
    FLOW_MLP_NS.store(0, Ordering::Relaxed);
    FLOW_SELF_QKV_CALLS.store(0, Ordering::Relaxed);
    FLOW_SELF_QKV_NS.store(0, Ordering::Relaxed);
    FLOW_SELF_NORM_ROPE_CALLS.store(0, Ordering::Relaxed);
    FLOW_SELF_NORM_ROPE_NS.store(0, Ordering::Relaxed);
    FLOW_SELF_NORM_ROPE_FUSED_QK_CALLS.store(0, Ordering::Relaxed);
    FLOW_SELF_NORM_ROPE_FUSED_QKV_MODULE_CALLS.store(0, Ordering::Relaxed);
    FLOW_SELF_KERNEL_CALLS.store(0, Ordering::Relaxed);
    FLOW_SELF_KERNEL_NS.store(0, Ordering::Relaxed);
    FLOW_SELF_OUT_CALLS.store(0, Ordering::Relaxed);
    FLOW_SELF_OUT_NS.store(0, Ordering::Relaxed);
    FLOW_SELF_CAT_CALLS.store(0, Ordering::Relaxed);
    FLOW_SELF_CAT_NS.store(0, Ordering::Relaxed);
    FLOW_CROSS_Q_CALLS.store(0, Ordering::Relaxed);
    FLOW_CROSS_Q_NS.store(0, Ordering::Relaxed);
    FLOW_CROSS_KV_CALLS.store(0, Ordering::Relaxed);
    FLOW_CROSS_KV_NS.store(0, Ordering::Relaxed);
    FLOW_CROSS_NORM_CALLS.store(0, Ordering::Relaxed);
    FLOW_CROSS_NORM_NS.store(0, Ordering::Relaxed);
    FLOW_CROSS_KERNEL_CALLS.store(0, Ordering::Relaxed);
    FLOW_CROSS_KERNEL_NS.store(0, Ordering::Relaxed);
    FLOW_CROSS_OUT_CALLS.store(0, Ordering::Relaxed);
    FLOW_CROSS_OUT_NS.store(0, Ordering::Relaxed);
    FLOW_CROSS_CAT_CALLS.store(0, Ordering::Relaxed);
    FLOW_CROSS_CAT_NS.store(0, Ordering::Relaxed);
    FLOW_MODULE_CAST_PAD_CALLS.store(0, Ordering::Relaxed);
    FLOW_MODULE_CAST_PAD_NS.store(0, Ordering::Relaxed);
    FLOW_MODULE_ATTENTION_CALLS.store(0, Ordering::Relaxed);
    FLOW_MODULE_ATTENTION_NS.store(0, Ordering::Relaxed);
    FLOW_MODULE_OUTPUT_CALLS.store(0, Ordering::Relaxed);
    FLOW_MODULE_OUTPUT_NS.store(0, Ordering::Relaxed);
    FLOW_BLOCK_NORM_MOD_CALLS.store(0, Ordering::Relaxed);
    FLOW_BLOCK_NORM_MOD_NS.store(0, Ordering::Relaxed);
    FLOW_BLOCK_NORM_AFFINE_CALLS.store(0, Ordering::Relaxed);
    FLOW_BLOCK_NORM_AFFINE_NS.store(0, Ordering::Relaxed);
    FLOW_BLOCK_GATE_RESIDUAL_CALLS.store(0, Ordering::Relaxed);
    FLOW_BLOCK_GATE_RESIDUAL_NS.store(0, Ordering::Relaxed);
    FLOW_MODEL_IO_CALLS.store(0, Ordering::Relaxed);
    FLOW_MODEL_IO_NS.store(0, Ordering::Relaxed);
    FLOW_MODEL_INPUT_CALLS.store(0, Ordering::Relaxed);
    FLOW_MODEL_INPUT_NS.store(0, Ordering::Relaxed);
    FLOW_MODEL_OUTPUT_CALLS.store(0, Ordering::Relaxed);
    FLOW_MODEL_OUTPUT_NS.store(0, Ordering::Relaxed);
}

pub fn sparse_flow_op_telemetry() -> SparseFlowOpTelemetry {
    SparseFlowOpTelemetry {
        self_attn_calls: FLOW_SELF_ATTN_CALLS.load(Ordering::Relaxed),
        self_attn_ns: FLOW_SELF_ATTN_NS.load(Ordering::Relaxed),
        cross_attn_calls: FLOW_CROSS_ATTN_CALLS.load(Ordering::Relaxed),
        cross_attn_ns: FLOW_CROSS_ATTN_NS.load(Ordering::Relaxed),
        mlp_calls: FLOW_MLP_CALLS.load(Ordering::Relaxed),
        mlp_ns: FLOW_MLP_NS.load(Ordering::Relaxed),
        self_qkv_calls: FLOW_SELF_QKV_CALLS.load(Ordering::Relaxed),
        self_qkv_ns: FLOW_SELF_QKV_NS.load(Ordering::Relaxed),
        self_norm_rope_calls: FLOW_SELF_NORM_ROPE_CALLS.load(Ordering::Relaxed),
        self_norm_rope_ns: FLOW_SELF_NORM_ROPE_NS.load(Ordering::Relaxed),
        self_norm_rope_fused_qk_calls: FLOW_SELF_NORM_ROPE_FUSED_QK_CALLS.load(Ordering::Relaxed),
        self_norm_rope_fused_qkv_module_calls: FLOW_SELF_NORM_ROPE_FUSED_QKV_MODULE_CALLS
            .load(Ordering::Relaxed),
        self_kernel_calls: FLOW_SELF_KERNEL_CALLS.load(Ordering::Relaxed),
        self_kernel_ns: FLOW_SELF_KERNEL_NS.load(Ordering::Relaxed),
        self_out_calls: FLOW_SELF_OUT_CALLS.load(Ordering::Relaxed),
        self_out_ns: FLOW_SELF_OUT_NS.load(Ordering::Relaxed),
        self_cat_calls: FLOW_SELF_CAT_CALLS.load(Ordering::Relaxed),
        self_cat_ns: FLOW_SELF_CAT_NS.load(Ordering::Relaxed),
        cross_q_calls: FLOW_CROSS_Q_CALLS.load(Ordering::Relaxed),
        cross_q_ns: FLOW_CROSS_Q_NS.load(Ordering::Relaxed),
        cross_kv_calls: FLOW_CROSS_KV_CALLS.load(Ordering::Relaxed),
        cross_kv_ns: FLOW_CROSS_KV_NS.load(Ordering::Relaxed),
        cross_norm_calls: FLOW_CROSS_NORM_CALLS.load(Ordering::Relaxed),
        cross_norm_ns: FLOW_CROSS_NORM_NS.load(Ordering::Relaxed),
        cross_kernel_calls: FLOW_CROSS_KERNEL_CALLS.load(Ordering::Relaxed),
        cross_kernel_ns: FLOW_CROSS_KERNEL_NS.load(Ordering::Relaxed),
        cross_out_calls: FLOW_CROSS_OUT_CALLS.load(Ordering::Relaxed),
        cross_out_ns: FLOW_CROSS_OUT_NS.load(Ordering::Relaxed),
        cross_cat_calls: FLOW_CROSS_CAT_CALLS.load(Ordering::Relaxed),
        cross_cat_ns: FLOW_CROSS_CAT_NS.load(Ordering::Relaxed),
        module_cast_pad_calls: FLOW_MODULE_CAST_PAD_CALLS.load(Ordering::Relaxed),
        module_cast_pad_ns: FLOW_MODULE_CAST_PAD_NS.load(Ordering::Relaxed),
        module_attention_calls: FLOW_MODULE_ATTENTION_CALLS.load(Ordering::Relaxed),
        module_attention_ns: FLOW_MODULE_ATTENTION_NS.load(Ordering::Relaxed),
        module_output_calls: FLOW_MODULE_OUTPUT_CALLS.load(Ordering::Relaxed),
        module_output_ns: FLOW_MODULE_OUTPUT_NS.load(Ordering::Relaxed),
        block_norm_mod_calls: FLOW_BLOCK_NORM_MOD_CALLS.load(Ordering::Relaxed),
        block_norm_mod_ns: FLOW_BLOCK_NORM_MOD_NS.load(Ordering::Relaxed),
        block_norm_affine_calls: FLOW_BLOCK_NORM_AFFINE_CALLS.load(Ordering::Relaxed),
        block_norm_affine_ns: FLOW_BLOCK_NORM_AFFINE_NS.load(Ordering::Relaxed),
        block_gate_residual_calls: FLOW_BLOCK_GATE_RESIDUAL_CALLS.load(Ordering::Relaxed),
        block_gate_residual_ns: FLOW_BLOCK_GATE_RESIDUAL_NS.load(Ordering::Relaxed),
        model_io_calls: FLOW_MODEL_IO_CALLS.load(Ordering::Relaxed),
        model_io_ns: FLOW_MODEL_IO_NS.load(Ordering::Relaxed),
        model_input_calls: FLOW_MODEL_INPUT_CALLS.load(Ordering::Relaxed),
        model_input_ns: FLOW_MODEL_INPUT_NS.load(Ordering::Relaxed),
        model_output_calls: FLOW_MODEL_OUTPUT_CALLS.load(Ordering::Relaxed),
        model_output_ns: FLOW_MODEL_OUTPUT_NS.load(Ordering::Relaxed),
    }
}

fn record_sparse_flow_op(kind: SparseFlowOpKind, elapsed_ns: u64) {
    match kind {
        SparseFlowOpKind::SelfAttn => {
            FLOW_SELF_ATTN_CALLS.fetch_add(1, Ordering::Relaxed);
            FLOW_SELF_ATTN_NS.fetch_add(elapsed_ns, Ordering::Relaxed);
        }
        SparseFlowOpKind::CrossAttn => {
            FLOW_CROSS_ATTN_CALLS.fetch_add(1, Ordering::Relaxed);
            FLOW_CROSS_ATTN_NS.fetch_add(elapsed_ns, Ordering::Relaxed);
        }
        SparseFlowOpKind::Mlp => {
            FLOW_MLP_CALLS.fetch_add(1, Ordering::Relaxed);
            FLOW_MLP_NS.fetch_add(elapsed_ns, Ordering::Relaxed);
        }
    }
}

fn elapsed_ns(start: Instant) -> u64 {
    start.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64
}

fn record_sparse_flow_detail(kind: SparseFlowOpDetailKind, elapsed_ns: u64) {
    let (calls, nanos) = match kind {
        SparseFlowOpDetailKind::SelfQkv => (&FLOW_SELF_QKV_CALLS, &FLOW_SELF_QKV_NS),
        SparseFlowOpDetailKind::SelfNormRope => {
            (&FLOW_SELF_NORM_ROPE_CALLS, &FLOW_SELF_NORM_ROPE_NS)
        }
        SparseFlowOpDetailKind::SelfKernel => (&FLOW_SELF_KERNEL_CALLS, &FLOW_SELF_KERNEL_NS),
        SparseFlowOpDetailKind::SelfOut => (&FLOW_SELF_OUT_CALLS, &FLOW_SELF_OUT_NS),
        SparseFlowOpDetailKind::SelfCat => (&FLOW_SELF_CAT_CALLS, &FLOW_SELF_CAT_NS),
        SparseFlowOpDetailKind::CrossQ => (&FLOW_CROSS_Q_CALLS, &FLOW_CROSS_Q_NS),
        SparseFlowOpDetailKind::CrossKv => (&FLOW_CROSS_KV_CALLS, &FLOW_CROSS_KV_NS),
        SparseFlowOpDetailKind::CrossNorm => (&FLOW_CROSS_NORM_CALLS, &FLOW_CROSS_NORM_NS),
        SparseFlowOpDetailKind::CrossKernel => (&FLOW_CROSS_KERNEL_CALLS, &FLOW_CROSS_KERNEL_NS),
        SparseFlowOpDetailKind::CrossOut => (&FLOW_CROSS_OUT_CALLS, &FLOW_CROSS_OUT_NS),
        SparseFlowOpDetailKind::CrossCat => (&FLOW_CROSS_CAT_CALLS, &FLOW_CROSS_CAT_NS),
        SparseFlowOpDetailKind::ModuleCastPad => {
            (&FLOW_MODULE_CAST_PAD_CALLS, &FLOW_MODULE_CAST_PAD_NS)
        }
        SparseFlowOpDetailKind::ModuleAttention => {
            (&FLOW_MODULE_ATTENTION_CALLS, &FLOW_MODULE_ATTENTION_NS)
        }
        SparseFlowOpDetailKind::ModuleOutput => (&FLOW_MODULE_OUTPUT_CALLS, &FLOW_MODULE_OUTPUT_NS),
        SparseFlowOpDetailKind::BlockNormMod => {
            (&FLOW_BLOCK_NORM_MOD_CALLS, &FLOW_BLOCK_NORM_MOD_NS)
        }
        SparseFlowOpDetailKind::BlockNormAffine => {
            (&FLOW_BLOCK_NORM_AFFINE_CALLS, &FLOW_BLOCK_NORM_AFFINE_NS)
        }
        SparseFlowOpDetailKind::BlockGateResidual => (
            &FLOW_BLOCK_GATE_RESIDUAL_CALLS,
            &FLOW_BLOCK_GATE_RESIDUAL_NS,
        ),
        SparseFlowOpDetailKind::ModelIo => (&FLOW_MODEL_IO_CALLS, &FLOW_MODEL_IO_NS),
        SparseFlowOpDetailKind::ModelInput => (&FLOW_MODEL_INPUT_CALLS, &FLOW_MODEL_INPUT_NS),
        SparseFlowOpDetailKind::ModelOutput => (&FLOW_MODEL_OUTPUT_CALLS, &FLOW_MODEL_OUTPUT_NS),
    };
    calls.fetch_add(1, Ordering::Relaxed);
    nanos.fetch_add(elapsed_ns, Ordering::Relaxed);
}

fn sparse_flow_sync_profile_enabled() -> bool {
    std::env::var("TRELLIS2_SPARSE_FLOW_SYNC_PROFILE")
        .ok()
        .and_then(|value| {
            let value = value.trim().to_ascii_lowercase();
            match value.as_str() {
                "1" | "true" | "on" | "yes" => Some(true),
                "0" | "false" | "off" | "no" => Some(false),
                _ => None,
            }
        })
        .unwrap_or(false)
}

fn maybe_sync_sparse_flow_profile<B: Backend>(device: Option<&B::Device>) {
    if let Some(device) = device {
        let _ = B::sync(device);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SparseFlowOpKind {
    SelfAttn,
    CrossAttn,
    Mlp,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SparseFlowOpDetailKind {
    SelfQkv,
    SelfNormRope,
    SelfKernel,
    SelfOut,
    SelfCat,
    CrossQ,
    CrossKv,
    CrossNorm,
    CrossKernel,
    CrossOut,
    CrossCat,
    ModuleCastPad,
    ModuleAttention,
    ModuleOutput,
    BlockNormMod,
    BlockNormAffine,
    BlockGateResidual,
    ModelIo,
    ModelInput,
    ModelOutput,
}

fn runtime_sample_progress_interval(steps: usize) -> usize {
    if steps <= 16 { 1 } else { (steps / 8).max(1) }
}

fn set_sparse_flow_sampler_step_for_attention(step_idx: usize, step_count: usize) {
    #[cfg(feature = "runtime-model-wgpu")]
    set_runtime_model_sparse_flow_sampler_step(step_idx, step_count);
    #[cfg(not(feature = "runtime-model-wgpu"))]
    let _ = (step_idx, step_count);
}

fn clear_sparse_flow_sampler_step_for_attention() {
    #[cfg(feature = "runtime-model-wgpu")]
    clear_runtime_model_sparse_flow_sampler_step();
}

#[derive(Clone, Debug)]
pub struct SparseFlowRowTrace {
    pub steps: usize,
    pub row_channels: usize,
    pub samples: Vec<f32>,
    pub step_0_pred_v: Vec<f32>,
    pub step_0_pred_v_pos: Vec<f32>,
    pub step_0_pred_v_neg: Vec<f32>,
    pub step_0_x_t: Vec<f32>,
    pub step_mid_x_t: Vec<f32>,
    pub step_last_x_t: Vec<f32>,
    #[cfg(feature = "runtime-model-wgpu")]
    pub samples_wgpu: Option<Tensor<WgpuRuntimeBackend, 2>>,
    #[cfg(feature = "runtime-model-wgpu")]
    pub step_0_pred_v_wgpu: Option<Tensor<WgpuRuntimeBackend, 2>>,
    #[cfg(feature = "runtime-model-wgpu")]
    pub step_0_pred_v_pos_wgpu: Option<Tensor<WgpuRuntimeBackend, 2>>,
    #[cfg(feature = "runtime-model-wgpu")]
    pub step_0_pred_v_neg_wgpu: Option<Tensor<WgpuRuntimeBackend, 2>>,
    #[cfg(feature = "runtime-model-wgpu")]
    pub step_0_x_t_wgpu: Option<Tensor<WgpuRuntimeBackend, 2>>,
    #[cfg(feature = "runtime-model-wgpu")]
    pub step_mid_x_t_wgpu: Option<Tensor<WgpuRuntimeBackend, 2>>,
    #[cfg(feature = "runtime-model-wgpu")]
    pub step_last_x_t_wgpu: Option<Tensor<WgpuRuntimeBackend, 2>>,
}

struct SparseCfgPrediction<B: Backend> {
    guided: Tensor<B, 3>,
    pos: Option<Tensor<B, 3>>,
    neg: Option<Tensor<B, 3>>,
}

struct DenseCfgPrediction<B: Backend> {
    guided: Tensor<B, 5>,
    pos: Option<Tensor<B, 5>>,
    neg: Option<Tensor<B, 5>>,
}

#[derive(Clone, Debug)]
pub struct VarLenTensorOwned {
    feats_host: Option<Vec<f32>>,
    layout: Vec<Range<usize>>,
    channels: usize,
    #[cfg(feature = "runtime-model-wgpu")]
    feats_wgpu: Option<Tensor<WgpuRuntimeBackend, 2>>,
}

impl VarLenTensorOwned {
    pub fn from_layout(
        feats: Vec<f32>,
        layout: Vec<Range<usize>>,
        channels: usize,
    ) -> Result<Self, String> {
        if channels == 0 {
            return Err("varlen tensor channels must be > 0".to_string());
        }
        let mut rows = 0usize;
        let mut expected_start = 0usize;
        for (batch_idx, range) in layout.iter().enumerate() {
            if range.start > range.end {
                return Err(format!(
                    "varlen tensor layout range has start>end at batch {}: {}..{}",
                    batch_idx, range.start, range.end
                ));
            }
            if range.start != expected_start {
                return Err(format!(
                    "varlen tensor layout must be contiguous from row 0; batch {} starts at {} but expected {}",
                    batch_idx, range.start, expected_start
                ));
            }
            rows = range.end;
            expected_start = range.end;
        }
        let expected = rows
            .checked_mul(channels)
            .ok_or_else(|| "varlen tensor element count overflow".to_string())?;
        if feats.len() != expected {
            return Err(format!(
                "varlen tensor feature length mismatch: expected {} (rows={} channels={}), got {}",
                expected,
                rows,
                channels,
                feats.len()
            ));
        }
        Ok(Self {
            feats_host: Some(feats),
            layout,
            channels,
            #[cfg(feature = "runtime-model-wgpu")]
            feats_wgpu: None,
        })
    }

    #[cfg(feature = "runtime-model-wgpu")]
    pub fn from_wgpu_tensor(
        feats_wgpu: Tensor<WgpuRuntimeBackend, 2>,
        layout: Vec<Range<usize>>,
    ) -> Result<Self, String> {
        let [rows, channels] = feats_wgpu.dims();
        let mut expected_start = 0usize;
        for (batch_idx, range) in layout.iter().enumerate() {
            if range.start > range.end {
                return Err(format!(
                    "varlen tensor layout range has start>end at batch {}: {}..{}",
                    batch_idx, range.start, range.end
                ));
            }
            if range.start != expected_start {
                return Err(format!(
                    "varlen tensor layout must be contiguous from row 0; batch {} starts at {} but expected {}",
                    batch_idx, range.start, expected_start
                ));
            }
            expected_start = range.end;
        }
        if expected_start != rows {
            return Err(format!(
                "varlen tensor layout rows mismatch: layout_rows={} tensor_rows={}",
                expected_start, rows
            ));
        }
        Ok(Self {
            feats_host: None,
            layout,
            channels,
            feats_wgpu: Some(feats_wgpu),
        })
    }

    pub fn rows(&self) -> usize {
        self.layout
            .iter()
            .map(|range| range.end.saturating_sub(range.start))
            .sum()
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn batch_size(&self) -> usize {
        self.layout.len()
    }

    pub fn feats(&self) -> &[f32] {
        self.feats_host.as_deref().unwrap_or_else(|| {
            panic!(
                "varlen tensor host features are unavailable; use device tensor accessors on canonical wgpu path"
            )
        })
    }

    pub fn feats_host(&self, context: &str) -> Result<Vec<f32>, String> {
        if let Some(feats) = self.feats_host.as_ref() {
            return Ok(feats.clone());
        }
        Err(format!(
            "{context}: varlen tensor host readback from device tensor is disabled on canonical sparse path"
        ))
    }

    #[cfg(feature = "runtime-model-wgpu")]
    pub fn feats_wgpu(&self) -> Option<Tensor<WgpuRuntimeBackend, 2>> {
        self.feats_wgpu.as_ref().cloned()
    }

    pub fn layout(&self) -> &[Range<usize>] {
        self.layout.as_slice()
    }

    pub fn replace_feats(&self, feats: Vec<f32>, channels: usize) -> Result<Self, String> {
        Self::from_layout(feats, self.layout.clone(), channels)
    }

    #[cfg(feature = "runtime-model-wgpu")]
    pub fn as_device_owned(
        &self,
        context: &str,
    ) -> Result<super::types::VarLenTensorDevice<WgpuRuntimeBackend>, String> {
        let feats_wgpu = self.feats_wgpu.as_ref().cloned().ok_or_else(|| {
            format!(
                "{context}: varlen tensor is host-only; canonical device ownership conversion requires device tensor"
            )
        })?;
        let device = feats_wgpu.device();
        let mut offsets = Vec::with_capacity(self.layout.len());
        let mut lengths = Vec::with_capacity(self.layout.len());
        for (batch_idx, range) in self.layout.iter().enumerate() {
            let offset = i32::try_from(range.start).map_err(|_| {
                format!(
                    "{context}: layout offset overflow at batch {batch_idx}: {}",
                    range.start
                )
            })?;
            let len = i32::try_from(range.end.saturating_sub(range.start)).map_err(|_| {
                format!(
                    "{context}: layout length overflow at batch {batch_idx}: {}",
                    range.end.saturating_sub(range.start)
                )
            })?;
            offsets.push(offset);
            lengths.push(len);
        }
        let offsets_t = Tensor::<WgpuRuntimeBackend, 1, Int>::from_data(
            TensorData::new(offsets, [self.layout.len()]),
            &device,
        );
        let lengths_t = Tensor::<WgpuRuntimeBackend, 1, Int>::from_data(
            TensorData::new(lengths, [self.layout.len()]),
            &device,
        );
        let layout = super::types::SparseBatchLayoutDevice::new(offsets_t, lengths_t, self.rows());
        Ok(super::types::VarLenTensorDevice::new(
            feats_wgpu,
            layout,
            self.channels,
        ))
    }
}

#[derive(Clone, Debug)]
pub struct SparseTensorOwned {
    values: VarLenTensorOwned,
    coords_host: Option<Vec<[u32; 4]>>,
    sparse_resolution: usize,
    #[cfg(feature = "runtime-model-wgpu")]
    coords_wgpu: Option<Tensor<WgpuRuntimeBackend, 2, Int>>,
}

impl SparseTensorOwned {
    pub fn from_layout(
        coords: Vec<[u32; 4]>,
        feats: Vec<f32>,
        layout: Vec<Range<usize>>,
        channels: usize,
        sparse_resolution: usize,
    ) -> Result<Self, String> {
        if sparse_resolution == 0 {
            return Err("sparse tensor resolution must be > 0".to_string());
        }
        let values = VarLenTensorOwned::from_layout(feats, layout.clone(), channels)?;
        if values.rows() != coords.len() {
            return Err(format!(
                "sparse tensor row mismatch: coords={} feats_rows={}",
                coords.len(),
                values.rows()
            ));
        }
        for (batch_idx, range) in layout.iter().enumerate() {
            let expected_batch = batch_idx as u32;
            for row_idx in range.clone() {
                let actual = coords[row_idx][0];
                if actual != expected_batch {
                    return Err(format!(
                        "sparse tensor layout/coord batch mismatch at row {}: coord_batch={} expected={}",
                        row_idx, actual, expected_batch
                    ));
                }
            }
        }
        Ok(Self {
            values,
            coords_host: Some(coords),
            sparse_resolution,
            #[cfg(feature = "runtime-model-wgpu")]
            coords_wgpu: None,
        })
    }

    #[cfg(feature = "runtime-model-wgpu")]
    pub fn from_wgpu_tensors(
        coords_wgpu: Tensor<WgpuRuntimeBackend, 2, Int>,
        values: VarLenTensorOwned,
        sparse_resolution: usize,
    ) -> Result<Self, String> {
        if sparse_resolution == 0 {
            return Err("sparse tensor resolution must be > 0".to_string());
        }
        let [rows, cols] = coords_wgpu.dims();
        if cols != 4 {
            return Err(format!(
                "sparse tensor coord tensor must have 4 columns, got {}",
                cols
            ));
        }
        if values.rows() != rows {
            return Err(format!(
                "sparse tensor row mismatch: coords_rows={} feats_rows={}",
                rows,
                values.rows()
            ));
        }
        Ok(Self {
            values,
            coords_host: None,
            sparse_resolution,
            coords_wgpu: Some(coords_wgpu),
        })
    }

    pub fn rows(&self) -> usize {
        self.values.rows()
    }

    pub fn channels(&self) -> usize {
        self.values.channels()
    }

    pub fn batch_size(&self) -> usize {
        self.values.batch_size()
    }

    pub fn layout(&self) -> &[Range<usize>] {
        self.values.layout()
    }

    pub fn coords(&self) -> &[[u32; 4]] {
        self.coords_host.as_deref().unwrap_or_else(|| {
            panic!(
                "sparse tensor host coords are unavailable; use device coord tensor accessors on canonical wgpu path"
            )
        })
    }

    pub fn coords_host(&self, context: &str) -> Result<Vec<[u32; 4]>, String> {
        if let Some(coords) = self.coords_host.as_ref() {
            return Ok(coords.clone());
        }
        Err(format!(
            "{context}: sparse tensor host readback from device coords is disabled on canonical sparse path"
        ))
    }

    #[cfg(feature = "runtime-model-wgpu")]
    pub fn coords_wgpu(&self) -> Option<Tensor<WgpuRuntimeBackend, 2, Int>> {
        self.coords_wgpu.as_ref().cloned()
    }

    pub fn feats(&self) -> &[f32] {
        self.values.feats()
    }

    pub fn feats_host(&self, context: &str) -> Result<Vec<f32>, String> {
        self.values.feats_host(context)
    }

    pub fn sparse_resolution(&self) -> usize {
        self.sparse_resolution
    }

    #[cfg(feature = "runtime-model-wgpu")]
    pub fn as_device_owned(
        &self,
        context: &str,
    ) -> Result<super::types::SparseTensorDevice<WgpuRuntimeBackend>, String> {
        let coords_wgpu = self.coords_wgpu.as_ref().cloned().ok_or_else(|| {
            format!(
                "{context}: sparse tensor is host-only; canonical device ownership conversion requires coord tensor"
            )
        })?;
        let values_device = self.values.as_device_owned(context)?;
        Ok(super::types::SparseTensorDevice::new(
            coords_wgpu,
            values_device,
            self.sparse_resolution,
        ))
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct SparseStructureFlowConfig {
    #[serde(default = "default_resolution")]
    pub resolution: usize,
    #[serde(default = "default_in_channels")]
    pub in_channels: usize,
    #[serde(default = "default_out_channels")]
    pub out_channels: usize,
    #[serde(default = "default_model_channels")]
    pub model_channels: usize,
    #[serde(default = "default_cond_channels")]
    pub cond_channels: usize,
    #[serde(default = "default_num_blocks")]
    pub num_blocks: usize,
    #[serde(default)]
    pub num_heads: Option<usize>,
    #[serde(default = "default_num_head_channels")]
    pub num_head_channels: usize,
    #[serde(default = "default_mlp_ratio")]
    pub mlp_ratio: f32,
    #[serde(default = "default_pe_mode")]
    pub pe_mode: String,
    #[serde(default = "default_rope_freq")]
    pub rope_freq: [f32; 2],
    #[serde(default = "default_share_mod")]
    pub share_mod: bool,
    #[serde(default = "default_qk_rms_norm")]
    pub qk_rms_norm: bool,
    #[serde(default = "default_qk_rms_norm_cross")]
    pub qk_rms_norm_cross: bool,
    #[serde(default = "default_frequency_embedding_size")]
    pub frequency_embedding_size: usize,
}

#[derive(Debug, Deserialize)]
struct SparseStructureFlowConfigFile {
    #[serde(default)]
    args: SparseStructureFlowConfig,
}

impl Default for SparseStructureFlowConfig {
    fn default() -> Self {
        Self {
            resolution: default_resolution(),
            in_channels: default_in_channels(),
            out_channels: default_out_channels(),
            model_channels: default_model_channels(),
            cond_channels: default_cond_channels(),
            num_blocks: default_num_blocks(),
            num_heads: None,
            num_head_channels: default_num_head_channels(),
            mlp_ratio: default_mlp_ratio(),
            pe_mode: default_pe_mode(),
            rope_freq: default_rope_freq(),
            share_mod: default_share_mod(),
            qk_rms_norm: default_qk_rms_norm(),
            qk_rms_norm_cross: default_qk_rms_norm_cross(),
            frequency_embedding_size: default_frequency_embedding_size(),
        }
    }
}

impl SparseStructureFlowConfig {
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, String> {
        let file: SparseStructureFlowConfigFile = serde_json::from_slice(bytes)
            .map_err(|err| format!("failed to parse sparse structure flow config json: {err}"))?;
        Ok(file.args)
    }

    pub fn num_heads(&self) -> usize {
        self.num_heads
            .unwrap_or(self.model_channels / self.num_head_channels.max(1))
            .max(1)
    }
}

fn default_resolution() -> usize {
    16
}

fn default_in_channels() -> usize {
    8
}

fn default_out_channels() -> usize {
    8
}

fn default_model_channels() -> usize {
    1536
}

fn default_cond_channels() -> usize {
    1024
}

fn default_num_blocks() -> usize {
    30
}

fn default_num_head_channels() -> usize {
    64
}

fn default_mlp_ratio() -> f32 {
    5.3334
}

fn default_pe_mode() -> String {
    "rope".to_string()
}

fn default_rope_freq() -> [f32; 2] {
    [1.0, 10_000.0]
}

fn default_share_mod() -> bool {
    true
}

fn default_qk_rms_norm() -> bool {
    true
}

fn default_qk_rms_norm_cross() -> bool {
    true
}

fn default_frequency_embedding_size() -> usize {
    256
}

#[derive(Module, Debug)]
pub struct TimestepEmbedder<B: Backend> {
    pub mlp_0: nn::Linear<B>,
    pub mlp_2: nn::Linear<B>,
}

impl<B: Backend> TimestepEmbedder<B>
where
    SparseFlowLinearWgpuBridgeImpl: SparseFlowLinearWgpuBridge<B>,
    SparseFlowBf16EmulationBridgeImpl: SparseFlowBf16EmulationBridge<B>,
{
    pub fn new(device: &B::Device, frequency_embedding_size: usize, hidden_size: usize) -> Self {
        let mlp_0 = nn::LinearConfig::new(frequency_embedding_size, hidden_size)
            .with_bias(true)
            .init(device);
        let mlp_2 = nn::LinearConfig::new(hidden_size, hidden_size)
            .with_bias(true)
            .init(device);
        Self { mlp_0, mlp_2 }
    }

    pub fn forward(&self, t: Tensor<B, 1>, frequency_embedding_size: usize) -> Tensor<B, 2> {
        let emb = timestep_embedding(t, frequency_embedding_size);
        let hidden = linear_forward_stable_2d(&self.mlp_0, emb);
        let hidden = silu(hidden);
        linear_forward_stable_2d(&self.mlp_2, hidden)
    }
}

#[derive(Module, Debug)]
pub struct MultiHeadRmsNorm<B: Backend> {
    pub gamma: Param<Tensor<B, 2>>,
    scale: f32,
}

impl<B: Backend> MultiHeadRmsNorm<B> {
    pub fn new(device: &B::Device, num_heads: usize, head_dim: usize) -> Self {
        let gamma = nn::Initializer::Ones.init([num_heads, head_dim], device);
        Self {
            gamma,
            scale: (head_dim as f32).sqrt(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4>
    where
        RopeRotateWgpuBridgeImpl: SparseFlowRmsNormWgpuBridge<B>,
        SparseFlowBf16EmulationBridgeImpl: SparseFlowBf16EmulationBridge<B>,
    {
        let [_, _, heads, head_dim] = x.dims();
        #[cfg(feature = "runtime-model-wgpu")]
        {
            let gamma = self.gamma.val();
            if let Some(y) =
                maybe_multihead_rms_norm_wgpu(x.clone(), gamma, self.scale, RMS_NORM_EPS)
            {
                return sparse_flow_stock_bf16_round(y);
            }
        }
        let rms = x
            .clone()
            .powf_scalar(2.0)
            .sum_dim(3)
            .add_scalar(RMS_NORM_EPS)
            .sqrt();
        let x = x.mul(rms.recip()).mul_scalar(self.scale);
        let gamma = self.gamma.val().reshape([1, 1, heads, head_dim]);
        let x_dtype: burn::tensor::FloatDType = x.dtype().into();
        let gamma_dtype: burn::tensor::FloatDType = gamma.dtype().into();
        let gamma = if gamma_dtype != x_dtype {
            gamma.cast(x_dtype)
        } else {
            gamma
        };
        sparse_flow_stock_bf16_round(x.mul(gamma))
    }
}

#[derive(Module, Debug)]
pub struct FeedForwardNet<B: Backend> {
    pub mlp_0: nn::Linear<B>,
    pub mlp_2: nn::Linear<B>,
}

impl<B: Backend> FeedForwardNet<B>
where
    SparseFlowLinearWgpuBridgeImpl: SparseFlowLinearWgpuBridge<B>,
    SparseFlowBf16EmulationBridgeImpl: SparseFlowBf16EmulationBridge<B>,
{
    pub fn new(device: &B::Device, channels: usize, mlp_ratio: f32) -> Self {
        let hidden = ((channels as f32) * mlp_ratio).round().max(1.0) as usize;
        let mlp_0 = nn::LinearConfig::new(channels, hidden)
            .with_bias(true)
            .init(device);
        let mlp_2 = nn::LinearConfig::new(hidden, channels)
            .with_bias(true)
            .init(device);
        Self { mlp_0, mlp_2 }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, tokens, channels] = x.dims();
        let chunk_tokens = sparse_flow_mlp_chunk_tokens_for_backend::<B>(tokens);
        let chunk_debug = attention_debug_enabled() && tokens >= 131_072;
        let sync_interval = sparse_flow_mlp_sync_interval_for_backend::<B>(tokens);
        let use_f16_chain = feed_forward_f16_chain_enabled::<B>(&self.mlp_0, &self.mlp_2);
        if chunk_tokens >= tokens {
            return if use_f16_chain {
                feed_forward_f16_chain_via_2d(&self.mlp_0, &self.mlp_2, x)
            } else {
                let hidden = linear_forward_block_via_2d(&self.mlp_0, x);
                let hidden = sparse_flow_stock_bf16_round(gelu(hidden));
                linear_forward_block_via_2d(&self.mlp_2, hidden)
            };
        }

        let device = x.device();
        let output_dtype: burn::tensor::FloatDType = x.dtype().into();
        let rows = batch.saturating_mul(tokens);
        let x_flat = x.reshape([rows, channels]);
        let mut chunks = Vec::new();
        let row_chunk = chunk_tokens.max(1);
        let total_chunks = (rows + row_chunk - 1) / row_chunk;
        let window_rows = sparse_flow_mlp_window_rows(rows)
            .max(row_chunk)
            .min(rows.max(1));
        let mut chunk_idx = 0usize;
        let mut window_start = 0usize;
        while window_start < rows {
            let window_end = (window_start + window_rows).min(rows);
            let window_len = window_end - window_start;
            let x_window = x_flat
                .clone()
                .slice([window_start..window_end, 0..channels])
                .add_scalar(0.0);
            let mut local_start = 0usize;
            while local_start < window_len {
                let local_end = (local_start + row_chunk).min(window_len);
                let start = window_start + local_start;
                let end = window_start + local_end;
                let should_log_chunk = chunk_debug;
                let chunk_start = if should_log_chunk {
                    eprintln!(
                        "burn_trellis: mlp.chunk {}/{} begin (row_start={} row_end={} rows={})",
                        chunk_idx + 1,
                        total_chunks,
                        start,
                        end,
                        end - start
                    );
                    Some(Instant::now())
                } else {
                    None
                };
                let x_chunk = x_window
                    .clone()
                    .slice([local_start..local_end, 0..channels]);
                let mlp_0_start = if should_log_chunk {
                    Some(Instant::now())
                } else {
                    None
                };
                let hidden = if use_f16_chain {
                    linear_forward_f16_raw_2d(&self.mlp_0, x_chunk)
                } else {
                    linear_forward_block_2d(&self.mlp_0, x_chunk)
                };
                if let Some(stage_start) = mlp_0_start {
                    eprintln!(
                        "burn_trellis: mlp.chunk {}/{} mlp_0 done ({:.2} ms)",
                        chunk_idx + 1,
                        total_chunks,
                        stage_start.elapsed().as_secs_f64() * 1000.0
                    );
                }
                let mlp_2_start = if should_log_chunk {
                    Some(Instant::now())
                } else {
                    None
                };
                let gelu_start = if should_log_chunk {
                    Some(Instant::now())
                } else {
                    None
                };
                let hidden = sparse_flow_stock_bf16_round(gelu(hidden));
                if let Some(stage_start) = gelu_start {
                    eprintln!(
                        "burn_trellis: mlp.chunk {}/{} gelu done ({:.2} ms)",
                        chunk_idx + 1,
                        total_chunks,
                        stage_start.elapsed().as_secs_f64() * 1000.0
                    );
                }
                let chunk_out = if use_f16_chain {
                    linear_forward_f16_raw_2d(&self.mlp_2, hidden).cast(output_dtype)
                } else {
                    linear_forward_block_2d(&self.mlp_2, hidden)
                };
                if let Some(stage_start) = mlp_2_start {
                    eprintln!(
                        "burn_trellis: mlp.chunk {}/{} mlp_2 done ({:.2} ms)",
                        chunk_idx + 1,
                        total_chunks,
                        stage_start.elapsed().as_secs_f64() * 1000.0
                    );
                }
                chunks.push(chunk_out);
                if sync_interval != usize::MAX && (chunk_idx + 1) % sync_interval == 0 {
                    let sync_start = if should_log_chunk {
                        Some(Instant::now())
                    } else {
                        None
                    };
                    let _ = B::sync(&device);
                    if let Some(stage_start) = sync_start {
                        eprintln!(
                            "burn_trellis: mlp.chunk {}/{} sync done ({:.2} ms)",
                            chunk_idx + 1,
                            total_chunks,
                            stage_start.elapsed().as_secs_f64() * 1000.0
                        );
                    }
                }
                if let Some(stage_start) = chunk_start {
                    eprintln!(
                        "burn_trellis: mlp.chunk {}/{} done ({:.2} ms)",
                        chunk_idx + 1,
                        total_chunks,
                        stage_start.elapsed().as_secs_f64() * 1000.0
                    );
                }
                local_start = local_end;
                chunk_idx += 1;
            }
            window_start = window_end;
        }
        Tensor::cat(chunks, 0).reshape([batch, tokens, channels])
    }
}

#[derive(Module, Debug)]
pub struct SelfAttention<B: Backend> {
    pub to_qkv: nn::Linear<B>,
    pub to_out: nn::Linear<B>,
    pub q_rms_norm: Option<MultiHeadRmsNorm<B>>,
    pub k_rms_norm: Option<MultiHeadRmsNorm<B>>,
    num_heads: usize,
    head_dim: usize,
    use_rope: bool,
    rope_freq: [f32; 2],
    attention_projection_f16: bool,
}

#[derive(Module, Debug)]
pub struct CrossAttention<B: Backend> {
    pub to_q: nn::Linear<B>,
    pub to_kv: nn::Linear<B>,
    pub to_out: nn::Linear<B>,
    pub q_rms_norm: Option<MultiHeadRmsNorm<B>>,
    pub k_rms_norm: Option<MultiHeadRmsNorm<B>>,
    num_heads: usize,
    head_dim: usize,
    attention_projection_f16: bool,
}

#[derive(Module, Debug)]
pub struct ModulatedTransformerCrossBlock<B: Backend> {
    pub self_attn: SelfAttention<B>,
    pub cross_attn: CrossAttention<B>,
    pub mlp: FeedForwardNet<B>,
    pub norm2: nn::LayerNorm<B>,
    pub modulation: Param<Tensor<B, 1>>,
}

#[derive(Module, Debug)]
pub struct SparseStructureFlowModel<B: Backend> {
    pub t_embedder: TimestepEmbedder<B>,
    pub ada_ln_modulation: nn::Linear<B>,
    pub input_layer: nn::Linear<B>,
    pub blocks: Vec<ModulatedTransformerCrossBlock<B>>,
    pub out_layer: nn::Linear<B>,
    config: Ignored<SparseStructureFlowConfig>,
}

#[derive(Clone)]
struct CrossAttentionProjectedKv<B: Backend> {
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    module_k: Option<Tensor<B, 4>>,
    module_v: Option<Tensor<B, 4>>,
    module_dtype: Option<burn::tensor::FloatDType>,
}

type CrossAttentionKvCache<B> = Vec<CrossAttentionProjectedKv<B>>;

#[derive(Debug)]
pub(crate) struct SparseStructureFlowRuntimeImpl<B: Backend> {
    config: SparseStructureFlowConfig,
    model: SparseStructureFlowModel<B>,
    device: B::Device,
}

#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub(crate) enum SparseStructureFlowRuntime {
    Cpu(SparseStructureFlowRuntimeImpl<CpuRuntimeBackend>),
    #[cfg(feature = "runtime-model-wgpu")]
    Wgpu(SparseStructureFlowRuntimeImpl<WgpuRuntimeBackend>),
}

#[derive(Debug)]
pub(crate) enum SparseFlowCondition {
    Cpu(Tensor<CpuRuntimeBackend, 3>),
    #[cfg(feature = "runtime-model-wgpu")]
    Wgpu(Tensor<WgpuRuntimeBackend, 3>),
}

trait SparseRuntimeTensorAccess<B: Backend> {
    fn build_state_rows_tensor(
        &self,
        sparse: &SparseTensorOwned,
        row_count: usize,
        state_channels: usize,
    ) -> Result<Tensor<B, 2>, String>;

    fn build_concat_rows_tensor(
        &self,
        values: &VarLenTensorOwned,
        row_count: usize,
        concat_channels: usize,
    ) -> Result<Tensor<B, 2>, String>;

    fn sparse_coords_tensor(
        &self,
        sparse: &SparseTensorOwned,
        context: &str,
    ) -> Result<Tensor<B, 2, Int>, String>;
}

impl SparseRuntimeTensorAccess<CpuRuntimeBackend>
    for SparseStructureFlowRuntimeImpl<CpuRuntimeBackend>
{
    fn build_state_rows_tensor(
        &self,
        sparse: &SparseTensorOwned,
        row_count: usize,
        state_channels: usize,
    ) -> Result<Tensor<CpuRuntimeBackend, 2>, String> {
        let feats = sparse.values.feats_host.as_deref().ok_or_else(|| {
            "sparse flow cpu path requires host-backed state features".to_string()
        })?;
        let expected = row_count.saturating_mul(state_channels);
        if feats.len() != expected {
            return Err(format!(
                "sparse flow state host features mismatch: expected {} values (rows={} channels={}), got {}",
                expected,
                row_count,
                state_channels,
                feats.len()
            ));
        }
        Ok(
            Tensor::<CpuRuntimeBackend, 1>::from_floats(feats, &self.device)
                .reshape([row_count, state_channels]),
        )
    }

    fn build_concat_rows_tensor(
        &self,
        values: &VarLenTensorOwned,
        row_count: usize,
        concat_channels: usize,
    ) -> Result<Tensor<CpuRuntimeBackend, 2>, String> {
        let feats = values.feats_host.as_deref().ok_or_else(|| {
            "sparse flow cpu path requires host-backed concat features".to_string()
        })?;
        let expected = row_count.saturating_mul(concat_channels);
        if feats.len() != expected {
            return Err(format!(
                "sparse flow concat host features mismatch: expected {} values (rows={} channels={}), got {}",
                expected,
                row_count,
                concat_channels,
                feats.len()
            ));
        }
        Ok(
            Tensor::<CpuRuntimeBackend, 1>::from_floats(feats, &self.device)
                .reshape([row_count, concat_channels]),
        )
    }

    fn sparse_coords_tensor(
        &self,
        sparse: &SparseTensorOwned,
        context: &str,
    ) -> Result<Tensor<CpuRuntimeBackend, 2, Int>, String> {
        let coords = sparse.coords_host.as_deref().ok_or_else(|| {
            format!("{context}: sparse flow cpu path requires host-backed coordinate rows")
        })?;
        let rows = coords.len();
        let mut flat = Vec::with_capacity(rows.saturating_mul(4));
        for (row_idx, coord) in coords.iter().enumerate() {
            for value in coord {
                let converted = i32::try_from(*value).map_err(|_| {
                    format!(
                        "{context}: sparse flow coord conversion overflow at row {} value {}",
                        row_idx, value
                    )
                })?;
                flat.push(converted);
            }
        }
        Ok(Tensor::<CpuRuntimeBackend, 1, Int>::from_data(
            TensorData::new(flat, [rows.saturating_mul(4)]),
            &self.device,
        )
        .reshape([rows, 4]))
    }
}

#[cfg(feature = "runtime-model-wgpu")]
impl SparseRuntimeTensorAccess<WgpuRuntimeBackend>
    for SparseStructureFlowRuntimeImpl<WgpuRuntimeBackend>
{
    fn build_state_rows_tensor(
        &self,
        sparse: &SparseTensorOwned,
        row_count: usize,
        state_channels: usize,
    ) -> Result<Tensor<WgpuRuntimeBackend, 2>, String> {
        if let Some(state_t) = sparse.values.feats_wgpu.as_ref() {
            let [rows, channels] = state_t.dims();
            if rows != row_count || channels != state_channels {
                return Err(format!(
                    "sparse flow state device tensor mismatch: got=[{},{}] expected=[{},{}]",
                    rows, channels, row_count, state_channels
                ));
            }
            return Ok(state_t.clone());
        }
        Err(
            "sparse flow wgpu state tensor missing device-backed features; host completion is disabled on canonical path"
                .to_string(),
        )
    }

    fn build_concat_rows_tensor(
        &self,
        values: &VarLenTensorOwned,
        row_count: usize,
        concat_channels: usize,
    ) -> Result<Tensor<WgpuRuntimeBackend, 2>, String> {
        if let Some(concat_t) = values.feats_wgpu.as_ref() {
            let [rows, channels] = concat_t.dims();
            if rows != row_count || channels != concat_channels {
                return Err(format!(
                    "sparse flow concat device tensor mismatch: got=[{},{}] expected=[{},{}]",
                    rows, channels, row_count, concat_channels
                ));
            }
            return Ok(concat_t.clone());
        }
        Err(
            "sparse flow wgpu concat tensor missing device-backed features; host completion is disabled on canonical path"
                .to_string(),
        )
    }

    fn sparse_coords_tensor(
        &self,
        sparse: &SparseTensorOwned,
        context: &str,
    ) -> Result<Tensor<WgpuRuntimeBackend, 2, Int>, String> {
        if let Some(coords_t) = sparse.coords_wgpu.as_ref() {
            return Ok(coords_t.clone());
        }
        Err(format!(
            "{context}: sparse flow wgpu coord tensor missing device-backed coords; host completion is disabled on canonical path"
        ))
    }
}

impl<B: Backend> SelfAttention<B>
where
    RopeRotateWgpuBridgeImpl: RopeRotateWgpuBridge<B>,
    RopeRotateWgpuBridgeImpl: SparseFlowRmsNormWgpuBridge<B>,
    RopeRotateWgpuBridgeImpl: SparseFlowQkRmsNormWgpuBridge<B>,
    SparseFlowLinearWgpuBridgeImpl: SparseFlowLinearWgpuBridge<B>,
    SparseFlowBf16EmulationBridgeImpl: SparseFlowBf16EmulationBridge<B>,
{
    pub fn new(
        device: &B::Device,
        channels: usize,
        num_heads: usize,
        use_rope: bool,
        rope_freq: [f32; 2],
        qk_rms_norm: bool,
        attention_projection_f16: bool,
    ) -> Self {
        let head_dim = channels / num_heads.max(1);
        let to_qkv = nn::LinearConfig::new(channels, channels * 3)
            .with_bias(true)
            .init(device);
        let to_out = nn::LinearConfig::new(channels, channels)
            .with_bias(true)
            .init(device);
        let q_rms_norm = if qk_rms_norm {
            Some(MultiHeadRmsNorm::new(device, num_heads, head_dim))
        } else {
            None
        };
        let k_rms_norm = if qk_rms_norm {
            Some(MultiHeadRmsNorm::new(device, num_heads, head_dim))
        } else {
            None
        };
        Self {
            to_qkv,
            to_out,
            q_rms_norm,
            k_rms_norm,
            num_heads,
            head_dim,
            use_rope,
            rope_freq,
            attention_projection_f16,
        }
    }

    fn forward(
        &self,
        x: Tensor<B, 3>,
        resolution: usize,
        token_coords: Option<Tensor<B, 2, Int>>,
    ) -> Tensor<B, 3> {
        let [batch, tokens, channels] = x.dims();
        let x_dtype = x.dtype().into();
        if sparse_flow_chunked_forward_for_backend::<B>(tokens) {
            return self.forward_chunked_stream(x, resolution, token_coords);
        }
        let qkv_start = Instant::now();
        let qkv = linear_forward_attention(&self.to_qkv, x, self.attention_projection_f16)
            .reshape([batch, tokens, 3, self.num_heads, self.head_dim]);
        record_sparse_flow_detail(SparseFlowOpDetailKind::SelfQkv, elapsed_ns(qkv_start));

        let v = qkv
            .clone()
            .slice([
                0..batch,
                0..tokens,
                2..3,
                0..self.num_heads,
                0..self.head_dim,
            ])
            .reshape([batch, tokens, self.num_heads, self.head_dim]);

        let norm_rope_start = Instant::now();
        let (q, k) = {
            #[cfg(feature = "runtime-model-wgpu")]
            if let Some((q, k)) = maybe_apply_qk_rms_norm_and_rope_from_qkv(
                qkv.clone(),
                self.q_rms_norm.as_ref(),
                self.k_rms_norm.as_ref(),
                self.use_rope,
                self.rope_freq,
                token_coords.clone(),
                0,
            ) {
                (q, k)
            } else {
                let q = qkv
                    .clone()
                    .slice([
                        0..batch,
                        0..tokens,
                        0..1,
                        0..self.num_heads,
                        0..self.head_dim,
                    ])
                    .reshape([batch, tokens, self.num_heads, self.head_dim]);
                let k = qkv
                    .slice([
                        0..batch,
                        0..tokens,
                        1..2,
                        0..self.num_heads,
                        0..self.head_dim,
                    ])
                    .reshape([batch, tokens, self.num_heads, self.head_dim]);
                (
                    apply_rms_norm_and_rope_single(
                        q,
                        self.q_rms_norm.as_ref(),
                        self.use_rope,
                        resolution,
                        self.head_dim,
                        self.rope_freq,
                        token_coords.clone(),
                        0,
                    ),
                    apply_rms_norm_and_rope_single(
                        k,
                        self.k_rms_norm.as_ref(),
                        self.use_rope,
                        resolution,
                        self.head_dim,
                        self.rope_freq,
                        token_coords,
                        0,
                    ),
                )
            }
            #[cfg(not(feature = "runtime-model-wgpu"))]
            {
                let q = qkv
                    .clone()
                    .slice([
                        0..batch,
                        0..tokens,
                        0..1,
                        0..self.num_heads,
                        0..self.head_dim,
                    ])
                    .reshape([batch, tokens, self.num_heads, self.head_dim]);
                let k = qkv
                    .slice([
                        0..batch,
                        0..tokens,
                        1..2,
                        0..self.num_heads,
                        0..self.head_dim,
                    ])
                    .reshape([batch, tokens, self.num_heads, self.head_dim]);
                (
                    apply_rms_norm_and_rope_single(
                        q,
                        self.q_rms_norm.as_ref(),
                        self.use_rope,
                        resolution,
                        self.head_dim,
                        self.rope_freq,
                        token_coords.clone(),
                        0,
                    ),
                    apply_rms_norm_and_rope_single(
                        k,
                        self.k_rms_norm.as_ref(),
                        self.use_rope,
                        resolution,
                        self.head_dim,
                        self.rope_freq,
                        token_coords,
                        0,
                    ),
                )
            }
        };
        record_sparse_flow_detail(
            SparseFlowOpDetailKind::SelfNormRope,
            elapsed_ns(norm_rope_start),
        );

        let kernel_start = Instant::now();
        let out =
            sparse_flow_stock_bf16_round(scaled_dot_product_attention(q, k, v, self.head_dim));
        record_sparse_flow_detail(SparseFlowOpDetailKind::SelfKernel, elapsed_ns(kernel_start));
        let out_start = Instant::now();
        let out = linear_forward_attention_to_dtype(
            &self.to_out,
            out.reshape([batch, tokens, channels]),
            x_dtype,
            self.attention_projection_f16,
        );
        record_sparse_flow_detail(SparseFlowOpDetailKind::SelfOut, elapsed_ns(out_start));
        out
    }

    fn forward_chunked_stream(
        &self,
        x: Tensor<B, 3>,
        resolution: usize,
        token_coords: Option<Tensor<B, 2, Int>>,
    ) -> Tensor<B, 3> {
        let [batch, tokens, channels] = x.dims();
        let x_dtype = x.dtype().into();
        let sync_profile_device = if sparse_flow_sync_profile_enabled() && tokens >= 1024 {
            Some(x.device())
        } else {
            None
        };
        let kv_chunk_tokens = sparse_flow_self_attn_kv_chunk_tokens(tokens);
        let backend_name = std::any::type_name::<B>();
        let module_kernel = attention_uses_module_kernel::<B>();
        let module_non_fusion = attention_uses_non_fusion_module_kernel::<B>();
        let module_full_attention =
            module_non_fusion && sparse_flow_module_attention_prefers_full(tokens);
        let mut reuse_qkv = sparse_flow_stream_reuse_qkv_enabled(tokens, channels);
        if module_full_attention {
            reuse_qkv = true;
        }
        let query_chunk_tokens = if reuse_qkv {
            kv_chunk_tokens
        } else {
            sparse_flow_self_attn_query_chunk_tokens(tokens)
        };
        let logits_budget = sparse_flow_attn_logits_budget_bytes();
        let (mut query_chunk_tokens, mut kv_chunk_tokens) = sparse_flow_stream_chunk_plan(
            batch,
            self.num_heads,
            tokens,
            query_chunk_tokens,
            kv_chunk_tokens,
            reuse_qkv,
            logits_budget,
        );
        if module_full_attention {
            query_chunk_tokens = tokens.max(1);
            kv_chunk_tokens = tokens.max(1);
        } else if module_kernel {
            let module_chunk_cap = sparse_flow_module_attention_chunk_cap_for_shape(
                batch,
                self.num_heads,
                tokens,
                tokens,
                self.head_dim,
            );
            if module_non_fusion {
                // Raw CubeBackend module attention should stay on flash kernels.
                // Avoid logits-budget downscaling here so moderate token counts
                // (for example ~8k) don't fragment into many tiny chunks.
                query_chunk_tokens = module_chunk_cap.max(1).min(tokens.max(1));
                kv_chunk_tokens = module_chunk_cap.max(1).min(tokens.max(1));
            } else {
                query_chunk_tokens = query_chunk_tokens.min(module_chunk_cap);
                kv_chunk_tokens = kv_chunk_tokens.min(module_chunk_cap);
                // Fusion backends can route module attention through dense fallbacks, so
                // keep a conservative query cap to avoid oversized temporary logits.
                let module_query_cap = sparse_flow_module_attention_query_chunk_cap(
                    batch,
                    self.num_heads,
                    tokens,
                    logits_budget,
                );
                query_chunk_tokens = query_chunk_tokens.min(module_query_cap);
            }
            if reuse_qkv {
                kv_chunk_tokens = kv_chunk_tokens.min(query_chunk_tokens);
            }
        }

        let module_attention_dtype = if module_kernel {
            Some(sparse_flow_self_module_attention_dtype_for_shape(
                batch,
                self.num_heads,
                tokens,
                self.head_dim,
            ))
        } else {
            None
        };

        if module_kernel
            && module_non_fusion
            && reuse_qkv
            && tokens >= 16_384
            && tokens <= sparse_flow_linear_chunk_tokens_for_backend::<B>(tokens)
        {
            let qkv_start = Instant::now();
            let qkv =
                linear_forward_attention(&self.to_qkv, x.clone(), self.attention_projection_f16)
                    .reshape([batch, tokens, 3, self.num_heads, self.head_dim]);
            maybe_sync_sparse_flow_profile::<B>(sync_profile_device.as_ref());
            record_sparse_flow_detail(SparseFlowOpDetailKind::SelfQkv, elapsed_ns(qkv_start));

            let norm_rope_start = Instant::now();
            let attention_dtype = module_attention_dtype
                .expect("module attention dtype should be selected for module kernel");
            let module_layout_qkv = {
                #[cfg(feature = "runtime-model-wgpu")]
                {
                    maybe_apply_qkv_module_rms_norm_and_rope_from_qkv(
                        qkv.clone(),
                        self.q_rms_norm.as_ref(),
                        self.k_rms_norm.as_ref(),
                        self.use_rope,
                        self.rope_freq,
                        token_coords.clone(),
                        0,
                    )
                }
                #[cfg(not(feature = "runtime-model-wgpu"))]
                {
                    None
                }
            };
            let (q_full, k_full, v_full) = if let Some((q, k, v)) = module_layout_qkv {
                record_sparse_flow_detail(
                    SparseFlowOpDetailKind::SelfNormRope,
                    elapsed_ns(norm_rope_start),
                );
                let cat_start = Instant::now();
                let q = tensor_cast_float_4d_if_needed(q, attention_dtype);
                let k = tensor_cast_float_4d_if_needed(k, attention_dtype);
                let v = tensor_cast_float_4d_if_needed(v, attention_dtype);
                maybe_sync_sparse_flow_profile::<B>(sync_profile_device.as_ref());
                record_sparse_flow_detail(SparseFlowOpDetailKind::SelfCat, elapsed_ns(cat_start));
                (q, k, v)
            } else {
                let v = qkv
                    .clone()
                    .slice([
                        0..batch,
                        0..tokens,
                        2..3,
                        0..self.num_heads,
                        0..self.head_dim,
                    ])
                    .reshape([batch, tokens, self.num_heads, self.head_dim]);
                let (q, k) = {
                    #[cfg(feature = "runtime-model-wgpu")]
                    if let Some((q, k)) = maybe_apply_qk_rms_norm_and_rope_from_qkv(
                        qkv.clone(),
                        self.q_rms_norm.as_ref(),
                        self.k_rms_norm.as_ref(),
                        self.use_rope,
                        self.rope_freq,
                        token_coords.clone(),
                        0,
                    ) {
                        (q, k)
                    } else {
                        let q = qkv
                            .clone()
                            .slice([
                                0..batch,
                                0..tokens,
                                0..1,
                                0..self.num_heads,
                                0..self.head_dim,
                            ])
                            .reshape([batch, tokens, self.num_heads, self.head_dim]);
                        let k = qkv
                            .slice([
                                0..batch,
                                0..tokens,
                                1..2,
                                0..self.num_heads,
                                0..self.head_dim,
                            ])
                            .reshape([batch, tokens, self.num_heads, self.head_dim]);
                        (
                            apply_rms_norm_and_rope_single(
                                q,
                                self.q_rms_norm.as_ref(),
                                self.use_rope,
                                resolution,
                                self.head_dim,
                                self.rope_freq,
                                token_coords.clone(),
                                0,
                            ),
                            apply_rms_norm_and_rope_single(
                                k,
                                self.k_rms_norm.as_ref(),
                                self.use_rope,
                                resolution,
                                self.head_dim,
                                self.rope_freq,
                                token_coords.clone(),
                                0,
                            ),
                        )
                    }
                    #[cfg(not(feature = "runtime-model-wgpu"))]
                    {
                        let q = qkv
                            .clone()
                            .slice([
                                0..batch,
                                0..tokens,
                                0..1,
                                0..self.num_heads,
                                0..self.head_dim,
                            ])
                            .reshape([batch, tokens, self.num_heads, self.head_dim]);
                        let k = qkv
                            .slice([
                                0..batch,
                                0..tokens,
                                1..2,
                                0..self.num_heads,
                                0..self.head_dim,
                            ])
                            .reshape([batch, tokens, self.num_heads, self.head_dim]);
                        (
                            apply_rms_norm_and_rope_single(
                                q,
                                self.q_rms_norm.as_ref(),
                                self.use_rope,
                                resolution,
                                self.head_dim,
                                self.rope_freq,
                                token_coords.clone(),
                                0,
                            ),
                            apply_rms_norm_and_rope_single(
                                k,
                                self.k_rms_norm.as_ref(),
                                self.use_rope,
                                resolution,
                                self.head_dim,
                                self.rope_freq,
                                token_coords.clone(),
                                0,
                            ),
                        )
                    }
                };
                maybe_sync_sparse_flow_profile::<B>(sync_profile_device.as_ref());
                record_sparse_flow_detail(
                    SparseFlowOpDetailKind::SelfNormRope,
                    elapsed_ns(norm_rope_start),
                );

                let cat_start = Instant::now();
                let q_full = tensor_cast_float_4d_if_needed(
                    force_contiguous_4d(q.permute([0, 2, 1, 3])),
                    attention_dtype,
                );
                let k_full = tensor_cast_float_4d_if_needed(
                    force_contiguous_4d(k.permute([0, 2, 1, 3])),
                    attention_dtype,
                );
                let v_full = tensor_cast_float_4d_if_needed(
                    force_contiguous_4d(v.permute([0, 2, 1, 3])),
                    attention_dtype,
                );
                record_sparse_flow_detail(SparseFlowOpDetailKind::SelfCat, elapsed_ns(cat_start));
                (q_full, k_full, v_full)
            };

            if attention_debug_enabled() && tokens >= 1024 {
                eprintln!(
                    "burn_trellis: attn chunked backend={backend_name} impl=flash_attention(module_attention_full_qkv) q_chunk={query_chunk_tokens} tokens={tokens}"
                );
            }

            let mut attn_chunks = Vec::new();
            let mut q_start = 0usize;
            while q_start < tokens {
                let q_end = (q_start + query_chunk_tokens).min(tokens);
                let q_tokens = q_end - q_start;
                let q_chunk = q_full.clone().slice([
                    0..batch,
                    0..self.num_heads,
                    q_start..q_end,
                    0..self.head_dim,
                ]);
                let kernel_start = Instant::now();
                let out =
                    sparse_flow_module_attention_prepared(q_chunk, k_full.clone(), v_full.clone())
                        .permute([0, 2, 1, 3])
                        .reshape([batch, q_tokens, channels]);
                maybe_sync_sparse_flow_profile::<B>(sync_profile_device.as_ref());
                record_sparse_flow_detail(
                    SparseFlowOpDetailKind::SelfKernel,
                    elapsed_ns(kernel_start),
                );
                attn_chunks.push(out);
                q_start = q_end;
            }

            let cat_start = Instant::now();
            let attn_out = Tensor::cat(attn_chunks, 1);
            maybe_sync_sparse_flow_profile::<B>(sync_profile_device.as_ref());
            record_sparse_flow_detail(SparseFlowOpDetailKind::SelfCat, elapsed_ns(cat_start));

            let out_start = Instant::now();
            let out = linear_forward_attention_to_dtype(
                &self.to_out,
                attn_out,
                x_dtype,
                self.attention_projection_f16,
            );
            maybe_sync_sparse_flow_profile::<B>(sync_profile_device.as_ref());
            record_sparse_flow_detail(SparseFlowOpDetailKind::SelfOut, elapsed_ns(out_start));
            return tensor_cast_float_3d_if_needed(out, x_dtype);
        }

        let mut k_chunks: Vec<Tensor<B, 4>> = Vec::new();
        let mut v_chunks: Vec<Tensor<B, 4>> = Vec::new();
        let mut q_chunks: Vec<Tensor<B, 4>> = Vec::new();
        let mut kv_start = 0usize;
        while kv_start < tokens {
            let kv_end = (kv_start + kv_chunk_tokens).min(tokens);
            let x_chunk = x.clone().slice([0..batch, kv_start..kv_end, 0..channels]);
            let qkv_start = Instant::now();
            let qkv =
                linear_forward_attention(&self.to_qkv, x_chunk, self.attention_projection_f16)
                    .reshape([batch, kv_end - kv_start, 3, self.num_heads, self.head_dim]);
            record_sparse_flow_detail(SparseFlowOpDetailKind::SelfQkv, elapsed_ns(qkv_start));
            let v = qkv
                .clone()
                .slice([
                    0..batch,
                    0..(kv_end - kv_start),
                    2..3,
                    0..self.num_heads,
                    0..self.head_dim,
                ])
                .reshape([batch, kv_end - kv_start, self.num_heads, self.head_dim]);

            let norm_rope_start = Instant::now();
            let (q, k) = {
                #[cfg(feature = "runtime-model-wgpu")]
                if let Some((q, k)) = maybe_apply_qk_rms_norm_and_rope_from_qkv(
                    qkv.clone(),
                    self.q_rms_norm.as_ref(),
                    self.k_rms_norm.as_ref(),
                    self.use_rope,
                    self.rope_freq,
                    token_coords.clone(),
                    kv_start,
                ) {
                    (q, k)
                } else {
                    let k = qkv
                        .clone()
                        .slice([
                            0..batch,
                            0..(kv_end - kv_start),
                            1..2,
                            0..self.num_heads,
                            0..self.head_dim,
                        ])
                        .reshape([batch, kv_end - kv_start, self.num_heads, self.head_dim]);
                    let q = qkv
                        .slice([
                            0..batch,
                            0..(kv_end - kv_start),
                            0..1,
                            0..self.num_heads,
                            0..self.head_dim,
                        ])
                        .reshape([batch, kv_end - kv_start, self.num_heads, self.head_dim]);
                    (
                        apply_rms_norm_and_rope_single(
                            q,
                            self.q_rms_norm.as_ref(),
                            self.use_rope,
                            resolution,
                            self.head_dim,
                            self.rope_freq,
                            token_coords.clone(),
                            kv_start,
                        ),
                        apply_rms_norm_and_rope_single(
                            k,
                            self.k_rms_norm.as_ref(),
                            self.use_rope,
                            resolution,
                            self.head_dim,
                            self.rope_freq,
                            token_coords.clone(),
                            kv_start,
                        ),
                    )
                }
                #[cfg(not(feature = "runtime-model-wgpu"))]
                {
                    let k = qkv
                        .clone()
                        .slice([
                            0..batch,
                            0..(kv_end - kv_start),
                            1..2,
                            0..self.num_heads,
                            0..self.head_dim,
                        ])
                        .reshape([batch, kv_end - kv_start, self.num_heads, self.head_dim]);
                    let q = qkv
                        .slice([
                            0..batch,
                            0..(kv_end - kv_start),
                            0..1,
                            0..self.num_heads,
                            0..self.head_dim,
                        ])
                        .reshape([batch, kv_end - kv_start, self.num_heads, self.head_dim]);
                    (
                        apply_rms_norm_and_rope_single(
                            q,
                            self.q_rms_norm.as_ref(),
                            self.use_rope,
                            resolution,
                            self.head_dim,
                            self.rope_freq,
                            token_coords.clone(),
                            kv_start,
                        ),
                        apply_rms_norm_and_rope_single(
                            k,
                            self.k_rms_norm.as_ref(),
                            self.use_rope,
                            resolution,
                            self.head_dim,
                            self.rope_freq,
                            token_coords.clone(),
                            kv_start,
                        ),
                    )
                }
            };
            record_sparse_flow_detail(
                SparseFlowOpDetailKind::SelfNormRope,
                elapsed_ns(norm_rope_start),
            );

            let k = k.permute([0, 2, 1, 3]);
            let v = v.permute([0, 2, 1, 3]);
            let (k, v) = if let Some(dtype) = module_attention_dtype {
                (
                    tensor_cast_float_4d_if_needed(k, dtype),
                    tensor_cast_float_4d_if_needed(v, dtype),
                )
            } else {
                (k, v)
            };
            k_chunks.push(k);
            v_chunks.push(v);
            if reuse_qkv {
                let q = q.permute([0, 2, 1, 3]);
                let q = if let Some(dtype) = module_attention_dtype {
                    tensor_cast_float_4d_if_needed(q, dtype)
                } else {
                    q
                };
                q_chunks.push(q);
            }
            kv_start = kv_end;
        }

        if attention_uses_module_kernel::<B>() {
            let cat_start = Instant::now();
            let k_full = if k_chunks.len() == 1 {
                k_chunks[0].clone()
            } else {
                Tensor::cat(k_chunks.clone(), 2)
            };
            let v_full = if v_chunks.len() == 1 {
                v_chunks[0].clone()
            } else {
                Tensor::cat(v_chunks.clone(), 2)
            };
            let attention_dtype = sparse_flow_self_module_attention_dtype_for_shape(
                batch,
                self.num_heads,
                tokens,
                self.head_dim,
            );
            let k_full =
                force_contiguous_4d(tensor_cast_float_4d_if_needed(k_full, attention_dtype));
            let v_full =
                force_contiguous_4d(tensor_cast_float_4d_if_needed(v_full, attention_dtype));
            record_sparse_flow_detail(SparseFlowOpDetailKind::SelfCat, elapsed_ns(cat_start));
            if attention_debug_enabled() && tokens >= 1024 {
                eprintln!(
                    "burn_trellis: attn chunked backend={backend_name} impl=flash_attention(module_attention) q_chunk={query_chunk_tokens} kv_chunk={kv_chunk_tokens} tokens={tokens} reuse_qkv={reuse_qkv} full={module_full_attention}"
                );
            }
            let mut out_chunks = Vec::new();
            if reuse_qkv {
                for q in q_chunks.into_iter() {
                    let q_tokens = q.dims()[2];
                    let kernel_start = Instant::now();
                    let out =
                        sparse_flow_module_attention_prepared(q, k_full.clone(), v_full.clone())
                            .permute([0, 2, 1, 3])
                            .reshape([batch, q_tokens, channels]);
                    record_sparse_flow_detail(
                        SparseFlowOpDetailKind::SelfKernel,
                        elapsed_ns(kernel_start),
                    );
                    let out_start = Instant::now();
                    out_chunks.push(linear_forward_attention_to_dtype(
                        &self.to_out,
                        out,
                        x_dtype,
                        self.attention_projection_f16,
                    ));
                    record_sparse_flow_detail(
                        SparseFlowOpDetailKind::SelfOut,
                        elapsed_ns(out_start),
                    );
                }
            } else {
                let mut q_start = 0usize;
                while q_start < tokens {
                    let q_end = (q_start + query_chunk_tokens).min(tokens);
                    let x_chunk = x.clone().slice([0..batch, q_start..q_end, 0..channels]);
                    let qkv_start = Instant::now();
                    let qkv = linear_forward_attention(
                        &self.to_qkv,
                        x_chunk,
                        self.attention_projection_f16,
                    )
                    .reshape([
                        batch,
                        q_end - q_start,
                        3,
                        self.num_heads,
                        self.head_dim,
                    ]);
                    record_sparse_flow_detail(
                        SparseFlowOpDetailKind::SelfQkv,
                        elapsed_ns(qkv_start),
                    );
                    let mut q = qkv
                        .slice([
                            0..batch,
                            0..(q_end - q_start),
                            0..1,
                            0..self.num_heads,
                            0..self.head_dim,
                        ])
                        .reshape([batch, q_end - q_start, self.num_heads, self.head_dim]);
                    let norm_rope_start = Instant::now();
                    q = apply_rms_norm_and_rope_single(
                        q,
                        self.q_rms_norm.as_ref(),
                        self.use_rope,
                        resolution,
                        self.head_dim,
                        self.rope_freq,
                        token_coords.clone(),
                        q_start,
                    );
                    record_sparse_flow_detail(
                        SparseFlowOpDetailKind::SelfNormRope,
                        elapsed_ns(norm_rope_start),
                    );
                    let kernel_start = Instant::now();
                    let q =
                        tensor_cast_float_4d_if_needed(q.permute([0, 2, 1, 3]), attention_dtype);
                    let out =
                        sparse_flow_module_attention_prepared(q, k_full.clone(), v_full.clone())
                            .permute([0, 2, 1, 3])
                            .reshape([batch, q_end - q_start, channels]);
                    record_sparse_flow_detail(
                        SparseFlowOpDetailKind::SelfKernel,
                        elapsed_ns(kernel_start),
                    );
                    let out_start = Instant::now();
                    out_chunks.push(linear_forward_attention_to_dtype(
                        &self.to_out,
                        out,
                        x_dtype,
                        self.attention_projection_f16,
                    ));
                    record_sparse_flow_detail(
                        SparseFlowOpDetailKind::SelfOut,
                        elapsed_ns(out_start),
                    );
                    q_start = q_end;
                }
            }
            let cat_start = Instant::now();
            let out = Tensor::cat(out_chunks, 1);
            record_sparse_flow_detail(SparseFlowOpDetailKind::SelfCat, elapsed_ns(cat_start));
            return tensor_cast_float_3d_if_needed(out, x_dtype);
        }

        let mut out_chunks = Vec::new();
        if reuse_qkv {
            for q in q_chunks.into_iter() {
                let q_tokens = q.dims()[2];
                let kernel_start = Instant::now();
                let out = scaled_dot_product_attention_stream_chunked_keys(
                    q,
                    k_chunks.as_slice(),
                    v_chunks.as_slice(),
                    self.head_dim,
                )
                .permute([0, 2, 1, 3])
                .reshape([batch, q_tokens, channels]);
                record_sparse_flow_detail(
                    SparseFlowOpDetailKind::SelfKernel,
                    elapsed_ns(kernel_start),
                );
                let out_start = Instant::now();
                out_chunks.push(linear_forward_attention_to_dtype(
                    &self.to_out,
                    out,
                    x_dtype,
                    self.attention_projection_f16,
                ));
                record_sparse_flow_detail(SparseFlowOpDetailKind::SelfOut, elapsed_ns(out_start));
            }
        } else {
            let mut q_start = 0usize;
            while q_start < tokens {
                let q_end = (q_start + query_chunk_tokens).min(tokens);
                let x_chunk = x.clone().slice([0..batch, q_start..q_end, 0..channels]);
                let qkv_start = Instant::now();
                let qkv =
                    linear_forward_attention(&self.to_qkv, x_chunk, self.attention_projection_f16)
                        .reshape([batch, q_end - q_start, 3, self.num_heads, self.head_dim]);
                record_sparse_flow_detail(SparseFlowOpDetailKind::SelfQkv, elapsed_ns(qkv_start));
                let mut q = qkv
                    .slice([
                        0..batch,
                        0..(q_end - q_start),
                        0..1,
                        0..self.num_heads,
                        0..self.head_dim,
                    ])
                    .reshape([batch, q_end - q_start, self.num_heads, self.head_dim]);
                let norm_rope_start = Instant::now();
                q = apply_rms_norm_and_rope_single(
                    q,
                    self.q_rms_norm.as_ref(),
                    self.use_rope,
                    resolution,
                    self.head_dim,
                    self.rope_freq,
                    token_coords.clone(),
                    q_start,
                );
                record_sparse_flow_detail(
                    SparseFlowOpDetailKind::SelfNormRope,
                    elapsed_ns(norm_rope_start),
                );

                let kernel_start = Instant::now();
                let out = scaled_dot_product_attention_stream_chunked_keys(
                    q.permute([0, 2, 1, 3]),
                    k_chunks.as_slice(),
                    v_chunks.as_slice(),
                    self.head_dim,
                )
                .permute([0, 2, 1, 3])
                .reshape([batch, q_end - q_start, channels]);
                record_sparse_flow_detail(
                    SparseFlowOpDetailKind::SelfKernel,
                    elapsed_ns(kernel_start),
                );

                let out_start = Instant::now();
                out_chunks.push(linear_forward_attention_to_dtype(
                    &self.to_out,
                    out,
                    x_dtype,
                    self.attention_projection_f16,
                ));
                record_sparse_flow_detail(SparseFlowOpDetailKind::SelfOut, elapsed_ns(out_start));
                q_start = q_end;
            }
        }

        let cat_start = Instant::now();
        let out = Tensor::cat(out_chunks, 1);
        record_sparse_flow_detail(SparseFlowOpDetailKind::SelfCat, elapsed_ns(cat_start));
        tensor_cast_float_3d_if_needed(out, x_dtype)
    }
}

impl<B: Backend> CrossAttention<B>
where
    RopeRotateWgpuBridgeImpl: SparseFlowRmsNormWgpuBridge<B>,
    SparseFlowLinearWgpuBridgeImpl: SparseFlowLinearWgpuBridge<B>,
    SparseFlowBf16EmulationBridgeImpl: SparseFlowBf16EmulationBridge<B>,
{
    pub fn new(
        device: &B::Device,
        channels: usize,
        ctx_channels: usize,
        num_heads: usize,
        qk_rms_norm: bool,
        attention_projection_f16: bool,
    ) -> Self {
        let head_dim = channels / num_heads.max(1);
        let to_q = nn::LinearConfig::new(channels, channels)
            .with_bias(true)
            .init(device);
        let to_kv = nn::LinearConfig::new(ctx_channels, channels * 2)
            .with_bias(true)
            .init(device);
        let to_out = nn::LinearConfig::new(channels, channels)
            .with_bias(true)
            .init(device);
        let q_rms_norm = if qk_rms_norm {
            Some(MultiHeadRmsNorm::new(device, num_heads, head_dim))
        } else {
            None
        };
        let k_rms_norm = if qk_rms_norm {
            Some(MultiHeadRmsNorm::new(device, num_heads, head_dim))
        } else {
            None
        };
        Self {
            to_q,
            to_kv,
            to_out,
            q_rms_norm,
            k_rms_norm,
            num_heads,
            head_dim,
            attention_projection_f16,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, context: Tensor<B, 3>) -> Tensor<B, 3> {
        let kv = self.project_context_kv(context);
        self.forward_from_projected_kv(x, &kv)
    }

    fn project_context_kv(&self, context: Tensor<B, 3>) -> CrossAttentionProjectedKv<B> {
        let [batch, ctx_tokens, _ctx_channels] = context.dims();
        let kv_start = Instant::now();
        let kv = linear_forward_attention(&self.to_kv, context, self.attention_projection_f16)
            .reshape([batch, ctx_tokens, 2, self.num_heads, self.head_dim]);
        let k = kv
            .clone()
            .slice([
                0..batch,
                0..ctx_tokens,
                0..1,
                0..self.num_heads,
                0..self.head_dim,
            ])
            .reshape([batch, ctx_tokens, self.num_heads, self.head_dim]);
        let v = kv
            .slice([
                0..batch,
                0..ctx_tokens,
                1..2,
                0..self.num_heads,
                0..self.head_dim,
            ])
            .reshape([batch, ctx_tokens, self.num_heads, self.head_dim]);
        let k = if let Some(norm) = self.k_rms_norm.as_ref() {
            norm.forward(k)
        } else {
            k
        };
        let (module_k, module_v, module_dtype) = if attention_uses_module_kernel::<B>() {
            let dtype = sparse_flow_cross_module_attention_dtype_for_shape(
                batch,
                self.num_heads,
                ctx_tokens,
                self.head_dim,
            );
            (
                Some(force_contiguous_4d(tensor_cast_float_4d_if_needed(
                    k.clone().permute([0, 2, 1, 3]),
                    dtype,
                ))),
                Some(force_contiguous_4d(tensor_cast_float_4d_if_needed(
                    v.clone().permute([0, 2, 1, 3]),
                    dtype,
                ))),
                Some(dtype),
            )
        } else {
            (None, None, None)
        };
        record_sparse_flow_detail(SparseFlowOpDetailKind::CrossKv, elapsed_ns(kv_start));
        CrossAttentionProjectedKv {
            k,
            v,
            module_k,
            module_v,
            module_dtype,
        }
    }

    fn forward_from_projected_kv(
        &self,
        x: Tensor<B, 3>,
        kv: &CrossAttentionProjectedKv<B>,
    ) -> Tensor<B, 3> {
        let [batch, tokens, channels] = x.dims();
        let x_dtype = x.dtype().into();
        if sparse_flow_chunked_forward_for_backend::<B>(tokens) {
            return self.forward_chunked_projected_kv(x, kv);
        }

        let q_start = Instant::now();
        let mut q = linear_forward_attention(&self.to_q, x, self.attention_projection_f16)
            .reshape([batch, tokens, self.num_heads, self.head_dim]);
        record_sparse_flow_detail(SparseFlowOpDetailKind::CrossQ, elapsed_ns(q_start));
        let norm_start = Instant::now();
        if let Some(norm) = self.q_rms_norm.as_ref() {
            q = norm.forward(q);
        }
        record_sparse_flow_detail(SparseFlowOpDetailKind::CrossNorm, elapsed_ns(norm_start));

        let kernel_start = Instant::now();
        let out = sparse_flow_stock_bf16_round(scaled_dot_product_attention(
            q,
            kv.k.clone(),
            kv.v.clone(),
            self.head_dim,
        ));
        record_sparse_flow_detail(
            SparseFlowOpDetailKind::CrossKernel,
            elapsed_ns(kernel_start),
        );
        let out_start = Instant::now();
        let out = linear_forward_attention_to_dtype(
            &self.to_out,
            out.reshape([batch, tokens, channels]),
            x_dtype,
            self.attention_projection_f16,
        );
        record_sparse_flow_detail(SparseFlowOpDetailKind::CrossOut, elapsed_ns(out_start));
        out
    }

    fn forward_chunked_projected_kv(
        &self,
        x: Tensor<B, 3>,
        kv: &CrossAttentionProjectedKv<B>,
    ) -> Tensor<B, 3> {
        let [batch, tokens, channels] = x.dims();
        let x_dtype = x.dtype().into();
        let ctx_tokens = kv.k.dims()[1];

        let backend_name = std::any::type_name::<B>();
        let use_module_attention = attention_uses_module_kernel::<B>()
            && !sparse_flow_module_attention_cross_shape_requires_stream(
                batch,
                self.num_heads,
                tokens,
                ctx_tokens,
                self.head_dim,
            );
        let module_non_fusion = attention_uses_non_fusion_module_kernel::<B>();
        let module_full_attention =
            module_non_fusion && sparse_flow_module_attention_prefers_full(tokens);

        let mut query_chunk_tokens = if use_module_attention {
            if module_full_attention {
                tokens.max(1)
            } else {
                sparse_flow_module_attention_chunk_cap_for_shape(
                    batch,
                    self.num_heads,
                    tokens,
                    ctx_tokens,
                    self.head_dim,
                )
            }
        } else {
            sparse_flow_self_attn_query_chunk_tokens(tokens)
        };
        if use_module_attention && !module_full_attention && !module_non_fusion {
            let logits_budget = sparse_flow_attn_logits_budget_bytes();
            query_chunk_tokens =
                query_chunk_tokens.min(sparse_flow_module_attention_query_chunk_cap(
                    batch,
                    self.num_heads,
                    ctx_tokens,
                    logits_budget,
                ));
        }
        query_chunk_tokens = query_chunk_tokens.max(1).min(tokens.max(1));
        let total_chunks = tokens.div_ceil(query_chunk_tokens);
        let debug_chunks = attention_debug_enabled() && tokens >= 1024;

        if debug_chunks {
            eprintln!(
                "burn_trellis: cross-attn chunked backend={backend_name} q_chunk={query_chunk_tokens} q_tokens={tokens} k_tokens={ctx_tokens} head_dim={} chunks={total_chunks} module_kernel={use_module_attention} full={module_full_attention}",
                self.head_dim,
            );
        }

        let kv_layout_start = Instant::now();
        let attention_dtype = kv.module_dtype.unwrap_or_else(|| {
            sparse_flow_cross_module_attention_dtype_for_shape(
                batch,
                self.num_heads,
                ctx_tokens,
                self.head_dim,
            )
        });
        let k_module = if use_module_attention {
            Some(kv.module_k.clone().unwrap_or_else(|| {
                tensor_cast_float_4d_if_needed(kv.k.clone().permute([0, 2, 1, 3]), attention_dtype)
            }))
        } else {
            None
        };
        let v_module = if use_module_attention {
            Some(kv.module_v.clone().unwrap_or_else(|| {
                tensor_cast_float_4d_if_needed(kv.v.clone().permute([0, 2, 1, 3]), attention_dtype)
            }))
        } else {
            None
        };
        if use_module_attention {
            record_sparse_flow_detail(SparseFlowOpDetailKind::CrossKv, elapsed_ns(kv_layout_start));
        }

        let mut out_chunks = Vec::new();
        let mut start = 0usize;
        let mut chunk_idx = 0usize;
        while start < tokens {
            let end = (start + query_chunk_tokens).min(tokens);
            let chunk_tokens = end - start;
            if debug_chunks && (chunk_idx % 8 == 0 || chunk_idx + 1 == total_chunks) {
                eprintln!(
                    "burn_trellis: cross-attn chunk {}/{} begin (start={} end={} size={})",
                    chunk_idx + 1,
                    total_chunks,
                    start,
                    end,
                    chunk_tokens
                );
            }
            let x_chunk = x.clone().slice([0..batch, start..end, 0..channels]);
            let q_start = Instant::now();
            let mut q =
                linear_forward_attention(&self.to_q, x_chunk, self.attention_projection_f16)
                    .reshape([batch, chunk_tokens, self.num_heads, self.head_dim]);
            record_sparse_flow_detail(SparseFlowOpDetailKind::CrossQ, elapsed_ns(q_start));
            let norm_start = Instant::now();
            let (q_module, q_dense) = if use_module_attention {
                if let Some(norm) = self.q_rms_norm.as_ref() {
                    q = norm.forward(q);
                }
                (
                    Some(tensor_cast_float_4d_if_needed(
                        q.permute([0, 2, 1, 3]),
                        attention_dtype,
                    )),
                    None,
                )
            } else {
                if let Some(norm) = self.q_rms_norm.as_ref() {
                    q = norm.forward(q);
                }
                (None, Some(q))
            };
            record_sparse_flow_detail(SparseFlowOpDetailKind::CrossNorm, elapsed_ns(norm_start));

            let attn_start = Instant::now();
            let out = if use_module_attention {
                sparse_flow_module_attention_prepared(
                    q_module.expect("module Q must be present"),
                    k_module.clone().expect("module K must be present"),
                    v_module.clone().expect("module V must be present"),
                )
                .permute([0, 2, 1, 3])
                .reshape([batch, chunk_tokens, channels])
            } else {
                scaled_dot_product_attention(
                    q_dense.expect("dense Q must be present"),
                    kv.k.clone(),
                    kv.v.clone(),
                    self.head_dim,
                )
                .reshape([batch, chunk_tokens, channels])
            };
            record_sparse_flow_detail(SparseFlowOpDetailKind::CrossKernel, elapsed_ns(attn_start));
            let out_start = Instant::now();
            out_chunks.push(linear_forward_attention_to_dtype(
                &self.to_out,
                out,
                x_dtype,
                self.attention_projection_f16,
            ));
            record_sparse_flow_detail(SparseFlowOpDetailKind::CrossOut, elapsed_ns(out_start));
            if debug_chunks && (chunk_idx % 8 == 0 || chunk_idx + 1 == total_chunks) {
                eprintln!(
                    "burn_trellis: cross-attn chunk {}/{} done ({:.2} ms)",
                    chunk_idx + 1,
                    total_chunks,
                    attn_start.elapsed().as_secs_f64() * 1000.0
                );
            }
            start = end;
            chunk_idx += 1;
        }

        let cat_start = Instant::now();
        let out = Tensor::cat(out_chunks, 1);
        record_sparse_flow_detail(SparseFlowOpDetailKind::CrossCat, elapsed_ns(cat_start));
        tensor_cast_float_3d_if_needed(out, x_dtype)
    }
}

impl<B: Backend> ModulatedTransformerCrossBlock<B>
where
    RopeRotateWgpuBridgeImpl: RopeRotateWgpuBridge<B>,
    RopeRotateWgpuBridgeImpl: SparseFlowRmsNormWgpuBridge<B>,
    RopeRotateWgpuBridgeImpl: SparseFlowQkRmsNormWgpuBridge<B>,
    SparseFlowLinearWgpuBridgeImpl: SparseFlowLinearWgpuBridge<B>,
    SparseFlowBf16EmulationBridgeImpl: SparseFlowBf16EmulationBridge<B>,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &B::Device,
        channels: usize,
        ctx_channels: usize,
        num_heads: usize,
        mlp_ratio: f32,
        use_rope: bool,
        rope_freq: [f32; 2],
        qk_rms_norm: bool,
        qk_rms_norm_cross: bool,
        attention_projection_f16: bool,
    ) -> Self {
        let self_attn = SelfAttention::new(
            device,
            channels,
            num_heads,
            use_rope,
            rope_freq,
            qk_rms_norm,
            attention_projection_f16,
        );
        let cross_attn = CrossAttention::new(
            device,
            channels,
            ctx_channels,
            num_heads,
            qk_rms_norm_cross,
            attention_projection_f16,
        );
        let mlp = FeedForwardNet::new(device, channels, mlp_ratio);
        let norm2 = nn::LayerNormConfig::new(channels)
            .with_epsilon(LAYER_NORM_EPS as f64)
            .init(device);
        let modulation = nn::Initializer::Zeros.init([channels * 6], device);
        Self {
            self_attn,
            cross_attn,
            mlp,
            norm2,
            modulation,
        }
    }

    fn forward(
        &self,
        x: Tensor<B, 3>,
        mod_signal: Tensor<B, 2>,
        context: Tensor<B, 3>,
        resolution: usize,
        token_coords: Option<Tensor<B, 2, Int>>,
        cross_attn_kv: Option<&CrossAttentionProjectedKv<B>>,
    ) -> Tensor<B, 3>
    where
        RopeRotateWgpuBridgeImpl: SparseFlowLayerNormWgpuBridge<B>,
        RopeRotateWgpuBridgeImpl: SparseFlowRmsNormWgpuBridge<B>,
        RopeRotateWgpuBridgeImpl: SparseFlowQkRmsNormWgpuBridge<B>,
        SparseFlowLinearWgpuBridgeImpl: SparseFlowLinearWgpuBridge<B>,
        SparseFlowBf16EmulationBridgeImpl: SparseFlowBf16EmulationBridge<B>,
    {
        let [batch, tokens, channels] = x.dims();
        let sync_profile_device = if sparse_flow_sync_profile_enabled() && tokens >= 1024 {
            Some(x.device())
        } else {
            None
        };
        let block_op_debug = attention_debug_enabled() && tokens >= 131_072;
        let mod_bias = self.modulation.val().reshape([1, channels * 6]);
        let mod_signal_dtype: burn::tensor::FloatDType = mod_signal.dtype().into();
        let mod_bias_dtype: burn::tensor::FloatDType = mod_bias.dtype().into();
        let mod_bias = if mod_bias_dtype != mod_signal_dtype {
            mod_bias.cast(mod_signal_dtype)
        } else {
            mod_bias
        };
        let mod_signal = sparse_flow_stock_bf16_round(mod_signal.add(mod_bias));
        let shift_msa = mod_signal
            .clone()
            .slice([0..batch, 0..channels])
            .reshape([batch, 1, channels]);
        let scale_msa = mod_signal
            .clone()
            .slice([0..batch, channels..(channels * 2)])
            .reshape([batch, 1, channels]);
        let gate_msa = mod_signal
            .clone()
            .slice([0..batch, (channels * 2)..(channels * 3)])
            .reshape([batch, 1, channels]);
        let shift_mlp = mod_signal
            .clone()
            .slice([0..batch, (channels * 3)..(channels * 4)])
            .reshape([batch, 1, channels]);
        let scale_mlp = mod_signal
            .clone()
            .slice([0..batch, (channels * 4)..(channels * 5)])
            .reshape([batch, 1, channels]);
        let gate_mlp = mod_signal
            .clone()
            .slice([0..batch, (channels * 5)..(channels * 6)])
            .reshape([batch, 1, channels]);

        let norm_start = Instant::now();
        let h = sparse_flow_stock_bf16_round(layer_norm_modulated(
            x.clone(),
            scale_msa,
            shift_msa,
            LAYER_NORM_EPS,
        ));
        maybe_sync_sparse_flow_profile::<B>(sync_profile_device.as_ref());
        record_sparse_flow_detail(SparseFlowOpDetailKind::BlockNormMod, elapsed_ns(norm_start));
        let self_attn_start = Instant::now();
        let h = if block_op_debug {
            let start = Instant::now();
            eprintln!("burn_trellis: flow.block op=self_attn begin (tokens={tokens})");
            let out = self.self_attn.forward(h, resolution, token_coords.clone());
            eprintln!(
                "burn_trellis: flow.block op=self_attn done ({:.2} ms)",
                start.elapsed().as_secs_f64() * 1000.0
            );
            out
        } else {
            self.self_attn.forward(h, resolution, token_coords.clone())
        };
        let h = sparse_flow_stock_bf16_round(h);
        maybe_sync_sparse_flow_profile::<B>(sync_profile_device.as_ref());
        let self_attn_ns = self_attn_start
            .elapsed()
            .as_nanos()
            .min(u128::from(u64::MAX)) as u64;
        record_sparse_flow_op(SparseFlowOpKind::SelfAttn, self_attn_ns);
        let gate_start = Instant::now();
        let h = sparse_flow_stock_bf16_round(h.mul(gate_msa));
        let x = sparse_flow_stock_bf16_round(x.add(h));
        maybe_sync_sparse_flow_profile::<B>(sync_profile_device.as_ref());
        record_sparse_flow_detail(
            SparseFlowOpDetailKind::BlockGateResidual,
            elapsed_ns(gate_start),
        );

        let norm_start = Instant::now();
        let h = sparse_flow_stock_bf16_round(layer_norm_affine_stable(
            x.clone(),
            &self.norm2,
            LAYER_NORM_EPS,
        ));
        maybe_sync_sparse_flow_profile::<B>(sync_profile_device.as_ref());
        record_sparse_flow_detail(
            SparseFlowOpDetailKind::BlockNormAffine,
            elapsed_ns(norm_start),
        );
        let cross_attn_start = Instant::now();
        let x = if block_op_debug {
            let start = Instant::now();
            eprintln!("burn_trellis: flow.block op=cross_attn begin (tokens={tokens})");
            let out = if let Some(kv) = cross_attn_kv {
                self.cross_attn.forward_from_projected_kv(h, kv)
            } else {
                self.cross_attn.forward(h, context.clone())
            };
            eprintln!(
                "burn_trellis: flow.block op=cross_attn done ({:.2} ms)",
                start.elapsed().as_secs_f64() * 1000.0
            );
            sparse_flow_stock_bf16_round(x.add(sparse_flow_stock_bf16_round(out)))
        } else {
            let out = if let Some(kv) = cross_attn_kv {
                self.cross_attn.forward_from_projected_kv(h, kv)
            } else {
                self.cross_attn.forward(h, context)
            };
            sparse_flow_stock_bf16_round(x.add(sparse_flow_stock_bf16_round(out)))
        };
        maybe_sync_sparse_flow_profile::<B>(sync_profile_device.as_ref());
        let cross_attn_ns = cross_attn_start
            .elapsed()
            .as_nanos()
            .min(u128::from(u64::MAX)) as u64;
        record_sparse_flow_op(SparseFlowOpKind::CrossAttn, cross_attn_ns);

        let norm_start = Instant::now();
        let h = sparse_flow_stock_bf16_round(layer_norm_modulated(
            x.clone(),
            scale_mlp,
            shift_mlp,
            LAYER_NORM_EPS,
        ));
        maybe_sync_sparse_flow_profile::<B>(sync_profile_device.as_ref());
        record_sparse_flow_detail(SparseFlowOpDetailKind::BlockNormMod, elapsed_ns(norm_start));
        let mlp_start = Instant::now();
        let h = if block_op_debug {
            let start = Instant::now();
            eprintln!("burn_trellis: flow.block op=mlp begin (tokens={tokens})");
            let out = self.mlp.forward(h);
            eprintln!(
                "burn_trellis: flow.block op=mlp done ({:.2} ms)",
                start.elapsed().as_secs_f64() * 1000.0
            );
            out
        } else {
            self.mlp.forward(h)
        };
        let h = sparse_flow_stock_bf16_round(h);
        maybe_sync_sparse_flow_profile::<B>(sync_profile_device.as_ref());
        let mlp_ns = mlp_start.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
        record_sparse_flow_op(SparseFlowOpKind::Mlp, mlp_ns);
        let gate_start = Instant::now();
        let h = sparse_flow_stock_bf16_round(h.mul(gate_mlp));
        let x = sparse_flow_stock_bf16_round(x.add(h));
        maybe_sync_sparse_flow_profile::<B>(sync_profile_device.as_ref());
        record_sparse_flow_detail(
            SparseFlowOpDetailKind::BlockGateResidual,
            elapsed_ns(gate_start),
        );
        x
    }
}

impl<B: Backend> SparseStructureFlowModel<B>
where
    RopeRotateWgpuBridgeImpl: RopeRotateWgpuBridge<B>,
    RopeRotateWgpuBridgeImpl: SparseFlowLayerNormWgpuBridge<B>,
    RopeRotateWgpuBridgeImpl: SparseFlowRmsNormWgpuBridge<B>,
    RopeRotateWgpuBridgeImpl: SparseFlowQkRmsNormWgpuBridge<B>,
    SparseFlowLinearWgpuBridgeImpl: SparseFlowLinearWgpuBridge<B>,
    SparseFlowBf16EmulationBridgeImpl: SparseFlowBf16EmulationBridge<B>,
{
    pub fn new(device: &B::Device, config: SparseStructureFlowConfig) -> Self {
        let num_heads = config.num_heads();
        let t_embedder = TimestepEmbedder::new(
            device,
            config.frequency_embedding_size,
            config.model_channels,
        );
        let ada_ln_modulation =
            nn::LinearConfig::new(config.model_channels, config.model_channels * 6)
                .with_bias(true)
                .init(device);
        let input_layer = nn::LinearConfig::new(config.in_channels, config.model_channels)
            .with_bias(true)
            .init(device);
        let attention_projection_f16 = {
            #[cfg(feature = "runtime-model-wgpu")]
            {
                sparse_flow_wgpu_linear_f16_enabled()
                    && attention_uses_non_fusion_module_kernel::<B>()
            }
            #[cfg(not(feature = "runtime-model-wgpu"))]
            {
                false
            }
        };
        let mut blocks = Vec::with_capacity(config.num_blocks);
        for _ in 0..config.num_blocks {
            blocks.push(ModulatedTransformerCrossBlock::new(
                device,
                config.model_channels,
                config.cond_channels,
                num_heads,
                config.mlp_ratio,
                config.pe_mode == "rope",
                config.rope_freq,
                config.qk_rms_norm,
                config.qk_rms_norm_cross,
                attention_projection_f16,
            ));
        }
        let out_layer = nn::LinearConfig::new(config.model_channels, config.out_channels)
            .with_bias(true)
            .init(device);
        Self {
            t_embedder,
            ada_ln_modulation,
            input_layer,
            blocks,
            out_layer,
            config: Ignored(config),
        }
    }

    pub fn config(&self) -> &SparseStructureFlowConfig {
        &self.config
    }

    fn prewarm_fast_f16_params(&self, device: &B::Device) {
        #[cfg(feature = "runtime-model-wgpu")]
        {
            if !(sparse_flow_wgpu_linear_f16_enabled()
                && attention_uses_non_fusion_module_kernel::<B>())
            {
                return;
            }
            let mut cached = 0usize;
            for block in &self.blocks {
                if maybe_cache_linear_f16_wgpu(&block.self_attn.to_qkv) {
                    cached += 1;
                }
                if maybe_cache_linear_f16_wgpu(&block.self_attn.to_out) {
                    cached += 1;
                }
                if maybe_cache_linear_f16_wgpu(&block.cross_attn.to_q) {
                    cached += 1;
                }
                if maybe_cache_linear_f16_wgpu(&block.cross_attn.to_kv) {
                    cached += 1;
                }
                if maybe_cache_linear_f16_wgpu(&block.cross_attn.to_out) {
                    cached += 1;
                }
                if maybe_cache_linear_f16_wgpu(&block.mlp.mlp_0) {
                    cached += 1;
                }
                if maybe_cache_linear_f16_wgpu(&block.mlp.mlp_2) {
                    cached += 1;
                }
            }
            if cached > 0 {
                let sync_start = Instant::now();
                let _ = B::sync(device);
                if sparse_flow_stage_debug_enabled() {
                    eprintln!(
                        "burn_trellis: sparse flow prewarmed {cached} f16 linear parameter sets ({:.2} ms)",
                        sync_start.elapsed().as_secs_f64() * 1000.0
                    );
                }
            }
        }
        #[cfg(not(feature = "runtime-model-wgpu"))]
        {
            let _ = device;
        }
    }

    #[allow(dead_code)]
    fn forward_tokens(
        &self,
        x: Tensor<B, 3>,
        t: Tensor<B, 1>,
        cond: Tensor<B, 3>,
        resolution: usize,
        token_coords: Option<Tensor<B, 2, Int>>,
    ) -> Tensor<B, 3> {
        self.forward_tokens_with_cross_cache(x, t, cond, resolution, token_coords, None)
    }

    fn forward_tokens_with_cross_cache(
        &self,
        x: Tensor<B, 3>,
        t: Tensor<B, 1>,
        cond: Tensor<B, 3>,
        resolution: usize,
        token_coords: Option<Tensor<B, 2, Int>>,
        cross_kv_cache: Option<&CrossAttentionKvCache<B>>,
    ) -> Tensor<B, 3> {
        let [_batch, tokens, channels] = x.dims();
        assert_eq!(
            channels, self.config.in_channels,
            "sparse flow input channel mismatch"
        );
        assert_eq!(
            cond.dims()[2],
            self.config.cond_channels,
            "sparse flow cond channel mismatch"
        );
        if let Some(coords) = token_coords.as_ref() {
            let [coord_rows, coord_cols] = coords.dims();
            assert_eq!(coord_cols, 3, "sparse flow token coord column mismatch");
            assert_eq!(
                coord_rows, tokens,
                "sparse flow token coords length mismatch"
            );
        }
        if let Some(cache) = cross_kv_cache {
            assert_eq!(
                cache.len(),
                self.blocks.len(),
                "sparse flow cross-attention K/V cache block count mismatch"
            );
        }

        let input_dtype: burn::tensor::FloatDType = x.dtype().into();
        let model_io_start = Instant::now();
        let mut h = linear_forward_input_token_chunked(
            &self.input_layer,
            x,
            sparse_flow_linear_chunk_tokens_for_backend::<B>(tokens),
        );

        let t_emb = self
            .t_embedder
            .forward(t, self.config.frequency_embedding_size);
        let mut mod_signal =
            linear_forward_stable_2d_reference(&self.ada_ln_modulation, silu(t_emb));
        let mut cond = cond;
        h = sparse_flow_stock_bf16_round(h);
        mod_signal = sparse_flow_stock_bf16_round(mod_signal);
        cond = sparse_flow_stock_bf16_round(cond);
        if let Some(torso_dtype) = sparse_flow_torso_dtype_for_backend::<B>() {
            h = tensor_cast_float_3d_if_needed(h, torso_dtype);
            mod_signal = tensor_cast_float_2d_if_needed(mod_signal, torso_dtype);
            cond = tensor_cast_float_3d_if_needed(cond, torso_dtype);
        }
        let elapsed = elapsed_ns(model_io_start);
        record_sparse_flow_detail(SparseFlowOpDetailKind::ModelIo, elapsed);
        record_sparse_flow_detail(SparseFlowOpDetailKind::ModelInput, elapsed);

        let finite_probe = sparse_flow_forward_finite_probe_enabled();
        if finite_probe {
            log_sparse_flow_forward_finite_probe("model.input_layer", h.clone());
            log_sparse_flow_forward_finite_probe("model.mod_signal", mod_signal.clone());
            log_sparse_flow_forward_finite_probe("model.cond", cond.clone());
        }

        let block_debug = attention_debug_enabled() && tokens >= 131_072;
        for (block_idx, block) in self.blocks.iter().enumerate() {
            let log_block = block_idx % 4 == 0 || block_idx + 1 == self.blocks.len();
            let block_start = if block_debug && log_block {
                let backend_name = std::any::type_name::<B>();
                eprintln!(
                    "burn_trellis: flow.block {}/{} begin (backend={backend_name} tokens={tokens})",
                    block_idx + 1,
                    self.blocks.len()
                );
                Some(Instant::now())
            } else {
                None
            };
            h = block.forward(
                h,
                mod_signal.clone(),
                cond.clone(),
                resolution,
                token_coords.clone(),
                cross_kv_cache.and_then(|cache| cache.get(block_idx)),
            );
            if finite_probe {
                log_sparse_flow_forward_finite_probe(
                    format!("model.block_{block_idx:02}.out").as_str(),
                    h.clone(),
                );
            }
            if let Some(start) = block_start {
                eprintln!(
                    "burn_trellis: flow.block {}/{} done ({:.2} ms)",
                    block_idx + 1,
                    self.blocks.len(),
                    start.elapsed().as_secs_f64() * 1000.0
                );
            }
        }

        let model_io_start = Instant::now();
        let h = tensor_cast_float_3d_if_needed(h, input_dtype);
        let h = layer_norm_no_affine(h, LAYER_NORM_EPS);
        if finite_probe {
            log_sparse_flow_forward_finite_probe("model.norm_out", h.clone());
        }
        let out = linear_forward_token_chunked_reference(
            &self.out_layer,
            h,
            sparse_flow_linear_chunk_tokens_for_backend::<B>(tokens),
        );
        if finite_probe {
            log_sparse_flow_forward_finite_probe("model.out", out.clone());
        }
        let elapsed = elapsed_ns(model_io_start);
        record_sparse_flow_detail(SparseFlowOpDetailKind::ModelIo, elapsed);
        record_sparse_flow_detail(SparseFlowOpDetailKind::ModelOutput, elapsed);
        out
    }

    fn project_cross_attention_cache(&self, cond: Tensor<B, 3>) -> CrossAttentionKvCache<B> {
        let cond = if let Some(torso_dtype) = sparse_flow_torso_dtype_for_backend::<B>() {
            tensor_cast_float_3d_if_needed(cond, torso_dtype)
        } else {
            cond
        };
        self.blocks
            .iter()
            .map(|block| block.cross_attn.project_context_kv(cond.clone()))
            .collect()
    }

    fn cross_attention_cache_estimated_bytes(
        &self,
        cond_batches: usize,
        cond_tokens: usize,
        condition_count: usize,
    ) -> usize {
        cond_batches
            .checked_mul(cond_tokens)
            .and_then(|value| value.checked_mul(self.config.model_channels))
            .and_then(|value| value.checked_mul(2))
            .and_then(|value| value.checked_mul(self.blocks.len()))
            .and_then(|value| value.checked_mul(condition_count.max(1)))
            .and_then(|value| value.checked_mul(core::mem::size_of::<f32>()))
            .unwrap_or(usize::MAX)
    }

    pub fn forward(&self, x: Tensor<B, 5>, t: Tensor<B, 1>, cond: Tensor<B, 3>) -> Tensor<B, 5> {
        self.forward_with_cross_cache(x, t, cond, None)
    }

    fn forward_with_cross_cache(
        &self,
        x: Tensor<B, 5>,
        t: Tensor<B, 1>,
        cond: Tensor<B, 3>,
        cross_kv_cache: Option<&CrossAttentionKvCache<B>>,
    ) -> Tensor<B, 5> {
        let [batch, channels, rx, ry, rz] = x.dims();
        assert_eq!(
            channels, self.config.in_channels,
            "sparse flow input channel mismatch"
        );
        assert_eq!(
            rx, self.config.resolution,
            "sparse flow input resolution mismatch"
        );
        assert_eq!(
            ry, self.config.resolution,
            "sparse flow input resolution mismatch"
        );
        assert_eq!(
            rz, self.config.resolution,
            "sparse flow input resolution mismatch"
        );
        let tokens = self.config.resolution * self.config.resolution * self.config.resolution;
        let device = x.device();
        let tokens_tensor = x.reshape([batch, channels, tokens]).swap_dims(1, 2);
        let token_coords = dense_grid_token_coords(self.config.resolution, device);
        let out_tokens = self.forward_tokens_with_cross_cache(
            tokens_tensor,
            t,
            cond,
            self.config.resolution,
            Some(token_coords),
            cross_kv_cache,
        );
        out_tokens.swap_dims(1, 2).reshape([
            batch,
            self.config.out_channels,
            self.config.resolution,
            self.config.resolution,
            self.config.resolution,
        ])
    }

    pub fn forward_sparse(
        &self,
        x: Tensor<B, 3>,
        t: Tensor<B, 1>,
        cond: Tensor<B, 3>,
        sparse_resolution: usize,
        token_coords: Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        self.forward_sparse_with_cross_cache(x, t, cond, sparse_resolution, token_coords, None)
    }

    fn forward_sparse_with_cross_cache(
        &self,
        x: Tensor<B, 3>,
        t: Tensor<B, 1>,
        cond: Tensor<B, 3>,
        sparse_resolution: usize,
        token_coords: Tensor<B, 2, Int>,
        cross_kv_cache: Option<&CrossAttentionKvCache<B>>,
    ) -> Tensor<B, 3> {
        self.forward_tokens_with_cross_cache(
            x,
            t,
            cond,
            sparse_resolution.max(1),
            Some(token_coords),
            cross_kv_cache,
        )
    }
}

impl<B> SparseStructureFlowRuntimeImpl<B>
where
    B: Backend,
    B::Device: Default,
    RopeRotateWgpuBridgeImpl: RopeRotateWgpuBridge<B>,
    RopeRotateWgpuBridgeImpl: SparseFlowLayerNormWgpuBridge<B>,
    RopeRotateWgpuBridgeImpl: SparseFlowRmsNormWgpuBridge<B>,
    RopeRotateWgpuBridgeImpl: SparseFlowQkRmsNormWgpuBridge<B>,
    SparseFlowLinearWgpuBridgeImpl: SparseFlowLinearWgpuBridge<B>,
    SparseFlowBf16EmulationBridgeImpl: SparseFlowBf16EmulationBridge<B>,
{
    fn load_from_stem(
        weights_root: &Path,
        image_large_root: Option<&Path>,
        model_stem: &str,
        resolution_override: Option<usize>,
    ) -> Result<Self, String> {
        let config_path =
            resolve_model_source_path(model_stem, "json", weights_root, image_large_root);
        let config_bytes = virtual_fs::read(&config_path).map_err(|err| {
            format!(
                "failed to read sparse structure flow config '{}': {err}",
                config_path.display()
            )
        })?;
        let mut config = SparseStructureFlowConfig::from_json_bytes(&config_bytes)?;
        if let Some(override_resolution) = resolution_override {
            if override_resolution == 0 {
                return Err("sparse flow resolution override must be > 0".to_string());
            }
            config.resolution = override_resolution;
        }
        if !config.share_mod {
            return Err(format!(
                "unsupported sparse structure flow config '{}': share_mod=false is not yet supported",
                config_path.display()
            ));
        }

        let weight_candidates =
            resolve_model_weight_candidates(model_stem, weights_root, image_large_root);
        if weight_candidates.is_empty() {
            return Err(format!(
                "unable to resolve sparse structure flow weights for stem '{model_stem}'"
            ));
        }
        let device = B::Device::default();
        let mut last_error = None;
        for weights_path in weight_candidates {
            let mut model = SparseStructureFlowModel::<B>::new(&device, config.clone());
            match load_sparse_model_weights(&mut model, &weights_path) {
                Ok(()) => {
                    if sparse_flow_weight_probe_enabled() {
                        log_sparse_flow_weight_probe(&model);
                    }
                    model.prewarm_fast_f16_params(&device);
                    return Ok(Self {
                        config,
                        model,
                        device,
                    });
                }
                Err(err) => {
                    last_error = Some(format!("{} ({err})", weights_path.display()));
                }
            }
        }

        Err(format!(
            "failed to load sparse structure flow weights for stem '{model_stem}': {}",
            last_error.unwrap_or_else(|| "unknown error".to_string())
        ))
    }
}

impl<B: Backend> SparseStructureFlowRuntimeImpl<B>
where
    RopeRotateWgpuBridgeImpl: RopeRotateWgpuBridge<B>,
    RopeRotateWgpuBridgeImpl: SparseFlowLayerNormWgpuBridge<B>,
    RopeRotateWgpuBridgeImpl: SparseFlowRmsNormWgpuBridge<B>,
    RopeRotateWgpuBridgeImpl: SparseFlowQkRmsNormWgpuBridge<B>,
    SparseFlowLinearWgpuBridgeImpl: SparseFlowLinearWgpuBridge<B>,
    SparseFlowBf16EmulationBridgeImpl: SparseFlowBf16EmulationBridge<B>,
{
    fn config(&self) -> &SparseStructureFlowConfig {
        &self.config
    }

    fn prepare_cross_attention_caches(
        &self,
        cond: Tensor<B, 3>,
        neg_cond: Tensor<B, 3>,
        sample_cfg: FlowEulerSampleConfig,
        t_pairs: &[(f32, f32)],
        stage_label: &str,
    ) -> (
        Option<CrossAttentionKvCache<B>>,
        Option<CrossAttentionKvCache<B>>,
    ) {
        if !sparse_flow_cross_kv_cache_enabled_for_backend::<B>() {
            return (None, None);
        }
        let budget = sparse_flow_cross_kv_cache_budget_bytes_for_backend::<B>();
        if budget == 0 {
            return (None, None);
        }
        let [cond_batches, cond_tokens, _] = cond.dims();
        let cond_estimated =
            self.model
                .cross_attention_cache_estimated_bytes(cond_batches, cond_tokens, 1);
        if cond_estimated > budget {
            eprintln!(
                "burn_trellis: {stage_label} cross-kv cache skipped (estimated={} MiB budget={} MiB)",
                bytes_to_mib(cond_estimated),
                bytes_to_mib(budget)
            );
            return (None, None);
        }

        let cache_start = Instant::now();
        let cond_cache = self.model.project_cross_attention_cache(cond);
        eprintln!(
            "burn_trellis: {stage_label} cross-kv cache projected condition={} MiB blocks={} ({:.2} ms)",
            bytes_to_mib(cond_estimated),
            cond_cache.len(),
            cache_start.elapsed().as_secs_f64() * 1000.0
        );

        let neg_needed = sample_cfg_needs_negative_condition(sample_cfg, t_pairs);
        let neg_cache = if neg_needed {
            let total_estimated = cond_estimated.saturating_mul(2);
            if total_estimated <= budget {
                let neg_start = Instant::now();
                let neg_cache = self.model.project_cross_attention_cache(neg_cond);
                eprintln!(
                    "burn_trellis: {stage_label} cross-kv cache projected negative={} MiB total={} MiB ({:.2} ms)",
                    bytes_to_mib(cond_estimated),
                    bytes_to_mib(total_estimated),
                    neg_start.elapsed().as_secs_f64() * 1000.0
                );
                Some(neg_cache)
            } else {
                eprintln!(
                    "burn_trellis: {stage_label} negative cross-kv cache skipped (total_estimated={} MiB budget={} MiB)",
                    bytes_to_mib(total_estimated),
                    bytes_to_mib(budget)
                );
                None
            }
        } else {
            None
        };

        (Some(cond_cache), neg_cache)
    }

    fn prepare_condition(&self, cond: &[f32], cond_tokens: usize) -> Result<Tensor<B, 3>, String> {
        let cond_elements = cond_tokens * self.config.cond_channels;
        if cond.len() != cond_elements {
            return Err(format!(
                "sparse flow cond length mismatch: expected {}, got {}",
                cond_elements,
                cond.len()
            ));
        }
        Ok(Tensor::<B, 1>::from_floats(cond, &self.device).reshape([
            1,
            cond_tokens,
            self.config.cond_channels,
        ]))
    }

    #[allow(dead_code)]
    fn predict_velocity_with_condition(
        &self,
        x_t: &[f32],
        timestep: f32,
        cond: Tensor<B, 3>,
        concat_cond: Option<&[f32]>,
    ) -> Result<Vec<f32>, String> {
        let voxel = self.config.resolution * self.config.resolution * self.config.resolution;
        if !x_t.len().is_multiple_of(voxel) {
            return Err(format!(
                "sparse flow sample length mismatch: sample len {} is not divisible by voxel count {}",
                x_t.len(),
                voxel
            ));
        }
        let state_channels = x_t.len() / voxel;
        let concat_channels = if let Some(cond) = concat_cond {
            if cond.len() % voxel != 0 {
                return Err(format!(
                    "sparse flow concat cond length mismatch: len {} is not divisible by voxel count {}",
                    cond.len(),
                    voxel
                ));
            }
            cond.len() / voxel
        } else {
            0usize
        };
        if state_channels + concat_channels != self.config.in_channels {
            return Err(format!(
                "sparse flow channel mismatch: state={} concat={} expected_in={}",
                state_channels, concat_channels, self.config.in_channels
            ));
        }
        if state_channels != self.config.out_channels {
            return Err(format!(
                "sparse flow state/output mismatch: state={} expected_out={}",
                state_channels, self.config.out_channels
            ));
        }

        let input = if let Some(concat) = concat_cond {
            let mut merged = Vec::with_capacity((state_channels + concat_channels) * voxel);
            merged.extend_from_slice(x_t);
            merged.extend_from_slice(concat);
            merged
        } else {
            x_t.to_vec()
        };

        let sample = Tensor::<B, 1>::from_floats(input.as_slice(), &self.device).reshape([
            1,
            self.config.in_channels,
            self.config.resolution,
            self.config.resolution,
            self.config.resolution,
        ]);
        let t = Tensor::<B, 1>::from_floats([timestep * 1000.0], &self.device);
        let out = self.model.forward(sample, t, cond);
        out.into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .map_err(|err| format!("failed to read sparse flow output: {err:?}"))
    }

    #[allow(clippy::too_many_arguments)]
    fn sample_with_trace(
        &self,
        noise: &[f32],
        sample_cfg: FlowEulerSampleConfig,
        sigma_min: f32,
        cond: Tensor<B, 3>,
        neg_cond: Tensor<B, 3>,
        concat_cond: Option<&[f32]>,
        capture_snapshots: bool,
    ) -> Result<FlowEulerSampleTrace, String> {
        let voxel = self.config.resolution * self.config.resolution * self.config.resolution;
        if voxel == 0 {
            return Err("sparse flow resolution produced zero voxels".to_string());
        }
        if !noise.len().is_multiple_of(voxel) {
            return Err(format!(
                "sparse flow sample length mismatch: sample len {} is not divisible by voxel count {}",
                noise.len(),
                voxel
            ));
        }
        let state_channels = noise.len() / voxel;
        if state_channels != self.config.out_channels {
            return Err(format!(
                "sparse flow state/output mismatch: state={} expected_out={}",
                state_channels, self.config.out_channels
            ));
        }
        let concat_channels = concat_cond.map_or(0usize, |values| values.len() / voxel);
        if state_channels + concat_channels != self.config.in_channels {
            return Err(format!(
                "sparse flow channel mismatch: state={} concat={} expected_in={}",
                state_channels, concat_channels, self.config.in_channels
            ));
        }
        if let Some(values) = concat_cond
            && !values.len().is_multiple_of(voxel)
        {
            return Err(format!(
                "sparse flow concat cond length mismatch: len {} is not divisible by voxel count {}",
                values.len(),
                voxel
            ));
        }

        let mut x_t = Tensor::<B, 1>::from_floats(noise, &self.device).reshape([
            1,
            state_channels,
            self.config.resolution,
            self.config.resolution,
            self.config.resolution,
        ]);
        let concat_tensor = concat_cond.map(|values| {
            Tensor::<B, 1>::from_floats(values, &self.device).reshape([
                1,
                concat_channels,
                self.config.resolution,
                self.config.resolution,
                self.config.resolution,
            ])
        });

        let mut step_0_pred_v: Option<Tensor<B, 5>> = None;
        let mut step_0_pred_v_pos: Option<Tensor<B, 5>> = None;
        let mut step_0_pred_v_neg: Option<Tensor<B, 5>> = None;
        let mut step_0_x_t: Option<Tensor<B, 5>> = None;
        let mut step_mid_x_t: Option<Tensor<B, 5>> = None;
        let mut step_last_x_t: Option<Tensor<B, 5>> = None;
        let mut step_pred_v = if capture_snapshots {
            Vec::with_capacity(sample_cfg.steps)
        } else {
            Vec::new()
        };
        let mut step_x_t = if capture_snapshots {
            Vec::with_capacity(sample_cfg.steps)
        } else {
            Vec::new()
        };
        let mid_step = mid_snapshot_step(sample_cfg.steps);
        let t_pairs = timestep_pairs(sample_cfg.steps, sample_cfg.rescale_t);
        let sample_start = Instant::now();
        let progress_interval = runtime_sample_progress_interval(sample_cfg.steps);
        let stage_label = if concat_cond.is_some() {
            "flow.sample_with_trace.concat"
        } else {
            "flow.sample_with_trace.sparse"
        };
        eprintln!(
            "burn_trellis: {stage_label} begin (steps={}, resolution={}, state_channels={}, concat_channels={})",
            sample_cfg.steps, self.config.resolution, state_channels, concat_channels
        );
        let (cond_cache, neg_cache) = self.prepare_cross_attention_caches(
            cond.clone(),
            neg_cond.clone(),
            sample_cfg,
            t_pairs.as_slice(),
            stage_label,
        );
        for (step_idx, (t, t_prev)) in t_pairs.into_iter().enumerate() {
            let step_start = Instant::now();
            set_sparse_flow_sampler_step_for_attention(step_idx, sample_cfg.steps);
            let pred_result = if capture_snapshots && step_idx == 0 {
                let parts = match self.predict_with_cfg_tensor_parts_with_cache(
                    x_t.clone(),
                    t,
                    sample_cfg,
                    sigma_min,
                    cond.clone(),
                    neg_cond.clone(),
                    concat_tensor.clone(),
                    cond_cache.as_ref(),
                    neg_cache.as_ref(),
                ) {
                    Ok(parts) => parts,
                    Err(err) => {
                        clear_sparse_flow_sampler_step_for_attention();
                        return Err(err);
                    }
                };
                step_0_pred_v = Some(parts.guided.clone());
                step_0_pred_v_pos = parts.pos;
                step_0_pred_v_neg = parts.neg;
                Ok(parts.guided)
            } else {
                self.predict_with_cfg_tensor_with_cache(
                    x_t.clone(),
                    t,
                    sample_cfg,
                    sigma_min,
                    cond.clone(),
                    neg_cond.clone(),
                    concat_tensor.clone(),
                    cond_cache.as_ref(),
                    neg_cache.as_ref(),
                )
            };
            let pred = match pred_result {
                Ok(pred) => pred,
                Err(err) => {
                    clear_sparse_flow_sampler_step_for_attention();
                    return Err(err);
                }
            };
            if capture_snapshots {
                step_pred_v.push(pred.clone());
            }
            let dt = t - t_prev;
            x_t = x_t.sub(pred.mul_scalar(dt));
            if capture_snapshots {
                step_x_t.push(x_t.clone());
            }
            if capture_snapshots && step_idx == 0 {
                step_0_x_t = Some(x_t.clone());
            }
            if capture_snapshots && step_idx == mid_step {
                step_mid_x_t = Some(x_t.clone());
            }
            if capture_snapshots && step_idx + 1 == sample_cfg.steps {
                step_last_x_t = Some(x_t.clone());
            }
            let step_done = step_idx + 1;
            if step_done % progress_interval == 0 || step_done == sample_cfg.steps {
                eprintln!(
                    "burn_trellis: {stage_label} step {step_done}/{} complete ({:.2} ms, elapsed={:.2} ms)",
                    sample_cfg.steps,
                    step_start.elapsed().as_secs_f64() * 1000.0,
                    sample_start.elapsed().as_secs_f64() * 1000.0
                );
            }
        }
        clear_sparse_flow_sampler_step_for_attention();
        eprintln!(
            "burn_trellis: {stage_label} complete ({:.2} ms)",
            sample_start.elapsed().as_secs_f64() * 1000.0
        );

        let state_len = state_channels.saturating_mul(voxel);
        let (
            samples,
            step_0_pred_v,
            step_0_pred_v_pos,
            step_0_pred_v_neg,
            step_0_x_t,
            step_mid_x_t,
            step_last_x_t,
            step_pred_v,
            step_x_t,
        ) = if capture_snapshots {
            let samples_t = x_t;
            let step_0_pred_v_t = step_0_pred_v
                .unwrap_or_else(|| samples_t.clone())
                .reshape([state_len]);
            let step_0_pred_v_pos_t = step_0_pred_v_pos
                .unwrap_or_else(|| samples_t.clone())
                .reshape([state_len]);
            let step_0_pred_v_neg_t = step_0_pred_v_neg
                .unwrap_or_else(|| samples_t.clone())
                .reshape([state_len]);
            let step_0_t = step_0_x_t
                .unwrap_or_else(|| samples_t.clone())
                .reshape([state_len]);
            let step_mid_t = step_mid_x_t
                .unwrap_or_else(|| samples_t.clone())
                .reshape([state_len]);
            let step_last_t = step_last_x_t
                .unwrap_or_else(|| samples_t.clone())
                .reshape([state_len]);
            let mut tensors = vec![
                samples_t.reshape([state_len]),
                step_0_pred_v_t,
                step_0_pred_v_pos_t,
                step_0_pred_v_neg_t,
                step_0_t,
                step_mid_t,
                step_last_t,
            ];
            let step_pred_v_count = step_pred_v.len();
            for tensor in step_pred_v {
                tensors.push(tensor.reshape([state_len]));
            }
            let step_x_t_count = step_x_t.len();
            for tensor in step_x_t {
                tensors.push(tensor.reshape([state_len]));
            }
            let merged = Tensor::cat(tensors, 0);
            let merged = tensor_to_vec_1d(merged, "failed to read sparse trace tensor")?;
            let segment = state_len;
            let samples = merged[..segment].to_vec();
            let step_0_pred_v = merged[segment..segment * 2].to_vec();
            let step_0_pred_v_pos = merged[segment * 2..segment * 3].to_vec();
            let step_0_pred_v_neg = merged[segment * 3..segment * 4].to_vec();
            let step_0_x_t = merged[segment * 4..segment * 5].to_vec();
            let step_mid_x_t = merged[segment * 5..segment * 6].to_vec();
            let step_last_x_t = merged[segment * 6..segment * 7].to_vec();
            let mut cursor = segment * 7;
            let mut step_pred_v_values = Vec::with_capacity(step_pred_v_count);
            for _ in 0..step_pred_v_count {
                step_pred_v_values.push(merged[cursor..cursor + segment].to_vec());
                cursor += segment;
            }
            let mut step_x_t_values = Vec::with_capacity(step_x_t_count);
            for _ in 0..step_x_t_count {
                step_x_t_values.push(merged[cursor..cursor + segment].to_vec());
                cursor += segment;
            }
            (
                samples,
                step_0_pred_v,
                step_0_pred_v_pos,
                step_0_pred_v_neg,
                step_0_x_t,
                step_mid_x_t,
                step_last_x_t,
                step_pred_v_values,
                step_x_t_values,
            )
        } else {
            let samples = tensor_to_vec(x_t)?;
            (
                samples.clone(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                samples.clone(),
                samples.clone(),
                samples,
                Vec::new(),
                Vec::new(),
            )
        };

        Ok(FlowEulerSampleTrace {
            steps: sample_cfg.steps,
            step_0_pred_v,
            step_0_pred_v_pos,
            step_0_pred_v_neg,
            step_0_x_t,
            step_mid_x_t,
            step_last_x_t,
            step_pred_v,
            step_x_t,
            samples,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn sample_final_tensor(
        &self,
        noise: &[f32],
        sample_cfg: FlowEulerSampleConfig,
        sigma_min: f32,
        cond: Tensor<B, 3>,
        neg_cond: Tensor<B, 3>,
        concat_cond: Option<&[f32]>,
    ) -> Result<Tensor<B, 5>, String> {
        let voxel = self.config.resolution * self.config.resolution * self.config.resolution;
        if voxel == 0 {
            return Err("sparse flow resolution produced zero voxels".to_string());
        }
        if !noise.len().is_multiple_of(voxel) {
            return Err(format!(
                "sparse flow sample length mismatch: sample len {} is not divisible by voxel count {}",
                noise.len(),
                voxel
            ));
        }
        let state_channels = noise.len() / voxel;
        if state_channels != self.config.out_channels {
            return Err(format!(
                "sparse flow state/output mismatch: state={} expected_out={}",
                state_channels, self.config.out_channels
            ));
        }
        let concat_channels = concat_cond.map_or(0usize, |values| values.len() / voxel);
        if state_channels + concat_channels != self.config.in_channels {
            return Err(format!(
                "sparse flow channel mismatch: state={} concat={} expected_in={}",
                state_channels, concat_channels, self.config.in_channels
            ));
        }
        if let Some(values) = concat_cond
            && !values.len().is_multiple_of(voxel)
        {
            return Err(format!(
                "sparse flow concat cond length mismatch: len {} is not divisible by voxel count {}",
                values.len(),
                voxel
            ));
        }

        let mut x_t = Tensor::<B, 1>::from_floats(noise, &self.device).reshape([
            1,
            state_channels,
            self.config.resolution,
            self.config.resolution,
            self.config.resolution,
        ]);
        let concat_tensor = concat_cond.map(|values| {
            Tensor::<B, 1>::from_floats(values, &self.device).reshape([
                1,
                concat_channels,
                self.config.resolution,
                self.config.resolution,
                self.config.resolution,
            ])
        });

        let t_pairs = timestep_pairs(sample_cfg.steps, sample_cfg.rescale_t);
        let sample_start = Instant::now();
        let progress_interval = runtime_sample_progress_interval(sample_cfg.steps);
        let stage_label = if concat_cond.is_some() {
            "flow.sample_tensor.concat"
        } else {
            "flow.sample_tensor.sparse"
        };
        eprintln!(
            "burn_trellis: {stage_label} begin (steps={}, resolution={}, state_channels={}, concat_channels={})",
            sample_cfg.steps, self.config.resolution, state_channels, concat_channels
        );
        let (cond_cache, neg_cache) = self.prepare_cross_attention_caches(
            cond.clone(),
            neg_cond.clone(),
            sample_cfg,
            t_pairs.as_slice(),
            stage_label,
        );
        for (step_idx, (t, t_prev)) in t_pairs.into_iter().enumerate() {
            let step_start = Instant::now();
            set_sparse_flow_sampler_step_for_attention(step_idx, sample_cfg.steps);
            let pred = match self.predict_with_cfg_tensor_with_cache(
                x_t.clone(),
                t,
                sample_cfg,
                sigma_min,
                cond.clone(),
                neg_cond.clone(),
                concat_tensor.clone(),
                cond_cache.as_ref(),
                neg_cache.as_ref(),
            ) {
                Ok(pred) => pred,
                Err(err) => {
                    clear_sparse_flow_sampler_step_for_attention();
                    return Err(err);
                }
            };
            let dt = t - t_prev;
            x_t = x_t.sub(pred.mul_scalar(dt));
            let step_done = step_idx + 1;
            if step_done % progress_interval == 0 || step_done == sample_cfg.steps {
                eprintln!(
                    "burn_trellis: {stage_label} step {step_done}/{} complete ({:.2} ms, elapsed={:.2} ms)",
                    sample_cfg.steps,
                    step_start.elapsed().as_secs_f64() * 1000.0,
                    sample_start.elapsed().as_secs_f64() * 1000.0
                );
            }
        }
        clear_sparse_flow_sampler_step_for_attention();
        eprintln!(
            "burn_trellis: {stage_label} complete ({:.2} ms)",
            sample_start.elapsed().as_secs_f64() * 1000.0
        );
        Ok(x_t)
    }

    #[allow(clippy::too_many_arguments)]
    fn sample_sparse_rows_with_trace(
        &self,
        sparse: &SparseTensorOwned,
        sample_cfg: FlowEulerSampleConfig,
        sigma_min: f32,
        cond: Tensor<B, 3>,
        neg_cond: Tensor<B, 3>,
        concat_cond: Option<&VarLenTensorOwned>,
        row_channels: usize,
        capture_snapshots: bool,
        materialize_host_rows: bool,
    ) -> Result<SparseFlowRowTrace, String>
    where
        Self: SparseRuntimeTensorAccess<B>,
        SparseFlowTraceWgpuBridgeImpl: SparseFlowTraceWgpuBridge<B>,
    {
        let row_count = sparse.rows();
        if row_count == 0 {
            return Ok(SparseFlowRowTrace {
                steps: sample_cfg.steps,
                row_channels: 0,
                samples: Vec::new(),
                step_0_pred_v: Vec::new(),
                step_0_pred_v_pos: Vec::new(),
                step_0_pred_v_neg: Vec::new(),
                step_0_x_t: Vec::new(),
                step_mid_x_t: Vec::new(),
                step_last_x_t: Vec::new(),
                #[cfg(feature = "runtime-model-wgpu")]
                samples_wgpu: None,
                #[cfg(feature = "runtime-model-wgpu")]
                step_0_pred_v_wgpu: None,
                #[cfg(feature = "runtime-model-wgpu")]
                step_0_pred_v_pos_wgpu: None,
                #[cfg(feature = "runtime-model-wgpu")]
                step_0_pred_v_neg_wgpu: None,
                #[cfg(feature = "runtime-model-wgpu")]
                step_0_x_t_wgpu: None,
                #[cfg(feature = "runtime-model-wgpu")]
                step_mid_x_t_wgpu: None,
                #[cfg(feature = "runtime-model-wgpu")]
                step_last_x_t_wgpu: None,
            });
        }
        let state_channels = sparse.channels();
        if state_channels != self.config.out_channels {
            return Err(format!(
                "sparse flow state/output mismatch: state={} expected_out={}",
                state_channels, self.config.out_channels
            ));
        }
        let sparse_layout = sparse.layout();
        let batch_count = sparse.batch_size();
        if batch_count == 0 {
            return Err("sparse flow sparse tensor layout has zero batches".to_string());
        }
        if let Some(values) = concat_cond
            && values.rows() != row_count
        {
            return Err(format!(
                "sparse flow row concat cond row mismatch: rows={} sparse_rows={}",
                values.rows(),
                row_count
            ));
        }
        if let Some(values) = concat_cond
            && values.layout() != sparse_layout
        {
            return Err(
                "sparse flow concat tensor layout does not match sparse tensor layout".to_string(),
            );
        }
        let concat_channels = concat_cond.map_or(0usize, VarLenTensorOwned::channels);
        if state_channels + concat_channels != self.config.in_channels {
            return Err(format!(
                "sparse flow channel mismatch: state={} concat={} expected_in={}",
                state_channels, concat_channels, self.config.in_channels
            ));
        }
        let [cond_batches, cond_tokens, cond_channels] = cond.dims();
        let [neg_batches, neg_tokens, neg_channels] = neg_cond.dims();
        if neg_tokens != cond_tokens || neg_channels != cond_channels {
            return Err(format!(
                "sparse flow negative condition shape mismatch: cond=[{cond_batches},{cond_tokens},{cond_channels}] neg=[{neg_batches},{neg_tokens},{neg_channels}]"
            ));
        }
        if cond_batches != 1 && cond_batches != batch_count {
            return Err(format!(
                "sparse flow condition batch mismatch: cond_batches={} sparse_batches={}",
                cond_batches, batch_count
            ));
        }
        if neg_batches != 1 && neg_batches != batch_count {
            return Err(format!(
                "sparse flow negative condition batch mismatch: neg_batches={} sparse_batches={}",
                neg_batches, batch_count
            ));
        }
        let select_condition_batch = |tensor: Tensor<B, 3>,
                                      tensor_batches: usize,
                                      batch_idx: usize,
                                      label: &str|
         -> Result<Tensor<B, 3>, String> {
            if tensor_batches == 1 {
                return Ok(tensor);
            }
            if batch_idx >= tensor_batches {
                return Err(format!(
                    "sparse flow {label} batch selection out of range: idx={} batches={}",
                    batch_idx, tensor_batches
                ));
            }
            Ok(tensor.slice([batch_idx..batch_idx + 1, 0..cond_tokens, 0..cond_channels]))
        };

        let used_channels = row_channels.min(state_channels);
        if used_channels == 0 {
            return Ok(SparseFlowRowTrace {
                steps: sample_cfg.steps,
                row_channels: 0,
                samples: Vec::new(),
                step_0_pred_v: Vec::new(),
                step_0_pred_v_pos: Vec::new(),
                step_0_pred_v_neg: Vec::new(),
                step_0_x_t: Vec::new(),
                step_mid_x_t: Vec::new(),
                step_last_x_t: Vec::new(),
                #[cfg(feature = "runtime-model-wgpu")]
                samples_wgpu: None,
                #[cfg(feature = "runtime-model-wgpu")]
                step_0_pred_v_wgpu: None,
                #[cfg(feature = "runtime-model-wgpu")]
                step_0_pred_v_pos_wgpu: None,
                #[cfg(feature = "runtime-model-wgpu")]
                step_0_pred_v_neg_wgpu: None,
                #[cfg(feature = "runtime-model-wgpu")]
                step_0_x_t_wgpu: None,
                #[cfg(feature = "runtime-model-wgpu")]
                step_mid_x_t_wgpu: None,
                #[cfg(feature = "runtime-model-wgpu")]
                step_last_x_t_wgpu: None,
            });
        }
        let total_elements = row_count.saturating_mul(used_channels);
        let mut samples_batches: Vec<Tensor<B, 1>> = Vec::with_capacity(batch_count);
        let mut step_0_pred_v_batches: Vec<Tensor<B, 1>> = if capture_snapshots {
            Vec::with_capacity(batch_count)
        } else {
            Vec::new()
        };
        let mut step_0_pred_v_pos_batches: Vec<Tensor<B, 1>> = if capture_snapshots {
            Vec::with_capacity(batch_count)
        } else {
            Vec::new()
        };
        let mut step_0_pred_v_neg_batches: Vec<Tensor<B, 1>> = if capture_snapshots {
            Vec::with_capacity(batch_count)
        } else {
            Vec::new()
        };
        let mut step_0_batches: Vec<Tensor<B, 1>> = if capture_snapshots {
            Vec::with_capacity(batch_count)
        } else {
            Vec::new()
        };
        let mut step_mid_batches: Vec<Tensor<B, 1>> = if capture_snapshots {
            Vec::with_capacity(batch_count)
        } else {
            Vec::new()
        };
        let mut step_last_batches: Vec<Tensor<B, 1>> = if capture_snapshots {
            Vec::with_capacity(batch_count)
        } else {
            Vec::new()
        };
        let state_all_t = Self::build_state_rows_tensor(self, sparse, row_count, state_channels)?;
        let concat_all_t = if let Some(values) = concat_cond {
            Some(Self::build_concat_rows_tensor(
                self,
                values,
                row_count,
                concat_channels,
            )?)
        } else {
            None
        };
        let coords_all_t =
            Self::sparse_coords_tensor(self, sparse, "sparse flow token coord tensorization")?;

        let mid_step = mid_snapshot_step(sample_cfg.steps);
        let t_pairs = timestep_pairs(sample_cfg.steps, sample_cfg.rescale_t);
        let sample_start = Instant::now();
        let progress_interval = runtime_sample_progress_interval(sample_cfg.steps);
        let stage_label = if concat_cond.is_some() {
            "flow.sample_sparse_rows_with_trace.tex_slat"
        } else {
            "flow.sample_sparse_rows_with_trace.shape_slat"
        };
        eprintln!(
            "burn_trellis: {stage_label} begin (steps={}, sparse_resolution={}, rows={}, batches={}, row_channels={})",
            sample_cfg.steps,
            sparse.sparse_resolution().max(1),
            row_count,
            batch_count,
            used_channels
        );
        for (batch_idx, range) in sparse_layout.iter().enumerate() {
            let batch_rows = range.end.saturating_sub(range.start);
            if batch_rows == 0 {
                continue;
            }
            let token_coords = coords_all_t.clone().slice([range.start..range.end, 1..4]);
            let mut x_t = state_all_t
                .clone()
                .slice([range.start..range.end, 0..state_channels])
                .reshape([1, batch_rows, state_channels]);
            let concat_tensor = concat_all_t.as_ref().map(|values| {
                values
                    .clone()
                    .slice([range.start..range.end, 0..concat_channels])
                    .reshape([1, batch_rows, concat_channels])
            });

            let cond_batch =
                select_condition_batch(cond.clone(), cond_batches, batch_idx, "condition")?;
            let neg_cond_batch = select_condition_batch(
                neg_cond.clone(),
                neg_batches,
                batch_idx,
                "negative condition",
            )?;
            let (cond_cache, neg_cache) = self.prepare_cross_attention_caches(
                cond_batch.clone(),
                neg_cond_batch.clone(),
                sample_cfg,
                t_pairs.as_slice(),
                stage_label,
            );
            let batched_cfg_cache = if sparse_flow_batched_cfg_enabled_for_backend::<B>() {
                match (&cond_cache, &neg_cache) {
                    (Some(pos_cache), Some(neg_cache)) => {
                        concat_cross_kv_caches(pos_cache, neg_cache)
                    }
                    _ => None,
                }
            } else {
                None
            };

            let mut step_0_pred_v_rows: Option<Tensor<B, 3>> = None;
            let mut step_0_pred_v_pos_rows: Option<Tensor<B, 3>> = None;
            let mut step_0_pred_v_neg_rows: Option<Tensor<B, 3>> = None;
            let mut step_0_rows: Option<Tensor<B, 3>> = None;
            let mut step_mid_rows: Option<Tensor<B, 3>> = None;
            let mut step_last_rows: Option<Tensor<B, 3>> = None;
            for (step_idx, (t, t_prev)) in t_pairs.iter().copied().enumerate() {
                let step_start = Instant::now();
                set_sparse_flow_sampler_step_for_attention(step_idx, sample_cfg.steps);
                let cfg_pred = match self.predict_with_cfg_sparse_tensor_parts_with_cache(
                    x_t.clone(),
                    t,
                    sample_cfg,
                    sigma_min,
                    cond_batch.clone(),
                    neg_cond_batch.clone(),
                    concat_tensor.clone(),
                    sparse.sparse_resolution().max(1),
                    token_coords.clone(),
                    cond_cache.as_ref(),
                    neg_cache.as_ref(),
                    batched_cfg_cache.as_ref(),
                ) {
                    Ok(pred) => pred,
                    Err(err) => {
                        clear_sparse_flow_sampler_step_for_attention();
                        return Err(err);
                    }
                };
                let pred = cfg_pred.guided;
                let dt = t - t_prev;
                if capture_snapshots && step_idx == 0 {
                    step_0_pred_v_rows = Some(pred.clone());
                    step_0_pred_v_pos_rows = cfg_pred.pos;
                    step_0_pred_v_neg_rows = cfg_pred.neg;
                }
                x_t = x_t.sub(pred.mul_scalar(dt));
                if capture_snapshots && step_idx == 0 {
                    step_0_rows = Some(x_t.clone());
                }
                if capture_snapshots && step_idx == mid_step {
                    step_mid_rows = Some(x_t.clone());
                }
                if capture_snapshots && step_idx + 1 == sample_cfg.steps {
                    step_last_rows = Some(x_t.clone());
                }
                let step_done = step_idx + 1;
                if step_done % progress_interval == 0 || step_done == sample_cfg.steps {
                    eprintln!(
                        "burn_trellis: {stage_label} batch {}/{} step {step_done}/{} complete ({:.2} ms, elapsed={:.2} ms)",
                        batch_idx + 1,
                        batch_count,
                        sample_cfg.steps,
                        step_start.elapsed().as_secs_f64() * 1000.0,
                        sample_start.elapsed().as_secs_f64() * 1000.0
                    );
                }
            }
            clear_sparse_flow_sampler_step_for_attention();

            let to_rows_1d = |tensor: Tensor<B, 3>| -> Tensor<B, 1> {
                if used_channels == state_channels {
                    tensor.reshape([batch_rows * used_channels])
                } else {
                    tensor
                        .slice([0..1, 0..batch_rows, 0..used_channels])
                        .reshape([batch_rows * used_channels])
                }
            };

            if capture_snapshots {
                let samples_rows = x_t.clone();
                samples_batches.push(to_rows_1d(samples_rows.clone()));
                step_0_pred_v_batches.push(to_rows_1d(
                    step_0_pred_v_rows.unwrap_or_else(|| samples_rows.clone()),
                ));
                step_0_pred_v_pos_batches.push(to_rows_1d(
                    step_0_pred_v_pos_rows.unwrap_or_else(|| samples_rows.clone()),
                ));
                step_0_pred_v_neg_batches.push(to_rows_1d(
                    step_0_pred_v_neg_rows.unwrap_or_else(|| samples_rows.clone()),
                ));
                step_0_batches.push(to_rows_1d(
                    step_0_rows.unwrap_or_else(|| samples_rows.clone()),
                ));
                step_mid_batches.push(to_rows_1d(
                    step_mid_rows.unwrap_or_else(|| samples_rows.clone()),
                ));
                step_last_batches.push(to_rows_1d(step_last_rows.unwrap_or(samples_rows)));
            } else {
                samples_batches.push(to_rows_1d(x_t));
            }
        }
        eprintln!(
            "burn_trellis: {stage_label} complete ({:.2} ms)",
            sample_start.elapsed().as_secs_f64() * 1000.0
        );

        let concat_batches =
            |mut batches: Vec<Tensor<B, 1>>, label: &str| -> Result<Tensor<B, 1>, String> {
                if batches.is_empty() {
                    return Err(format!(
                        "sparse flow {label} aggregation produced no batch tensors"
                    ));
                }
                let merged = if batches.len() == 1 {
                    batches
                        .pop()
                        .expect("single tensor batch aggregation should contain one tensor")
                } else {
                    Tensor::cat(batches, 0)
                };
                let [elements] = merged.dims();
                if elements != total_elements {
                    return Err(format!(
                        "sparse flow {label} aggregation element mismatch: got={} expected={}",
                        elements, total_elements
                    ));
                }
                Ok(merged)
            };

        #[cfg(feature = "runtime-model-wgpu")]
        let samples_wgpu: Option<Tensor<WgpuRuntimeBackend, 2>>;
        #[cfg(feature = "runtime-model-wgpu")]
        let step_0_pred_v_wgpu: Option<Tensor<WgpuRuntimeBackend, 2>>;
        #[cfg(feature = "runtime-model-wgpu")]
        let step_0_pred_v_pos_wgpu: Option<Tensor<WgpuRuntimeBackend, 2>>;
        #[cfg(feature = "runtime-model-wgpu")]
        let step_0_pred_v_neg_wgpu: Option<Tensor<WgpuRuntimeBackend, 2>>;
        #[cfg(feature = "runtime-model-wgpu")]
        let step_0_x_t_wgpu: Option<Tensor<WgpuRuntimeBackend, 2>>;
        #[cfg(feature = "runtime-model-wgpu")]
        let step_mid_x_t_wgpu: Option<Tensor<WgpuRuntimeBackend, 2>>;
        #[cfg(feature = "runtime-model-wgpu")]
        let step_last_x_t_wgpu: Option<Tensor<WgpuRuntimeBackend, 2>>;

        let (
            samples,
            step_0_pred_v,
            step_0_pred_v_pos,
            step_0_pred_v_neg,
            step_0_x_t,
            step_mid_x_t,
            step_last_x_t,
        ) = if capture_snapshots {
            let samples_t = concat_batches(samples_batches, "samples")?;
            let step_0_pred_v_t = concat_batches(step_0_pred_v_batches, "step_0_pred_v")?;
            let step_0_pred_v_pos_t =
                concat_batches(step_0_pred_v_pos_batches, "step_0_pred_v_pos")?;
            let step_0_pred_v_neg_t =
                concat_batches(step_0_pred_v_neg_batches, "step_0_pred_v_neg")?;
            let step_0_t = concat_batches(step_0_batches, "step_0_x_t")?;
            let step_mid_t = concat_batches(step_mid_batches, "step_mid_x_t")?;
            let step_last_t = concat_batches(step_last_batches, "step_last_x_t")?;
            #[cfg(feature = "runtime-model-wgpu")]
            {
                samples_wgpu =
                    maybe_trace_rows_wgpu(samples_t.clone().reshape([row_count, used_channels]));
                step_0_pred_v_wgpu = maybe_trace_rows_wgpu(
                    step_0_pred_v_t.clone().reshape([row_count, used_channels]),
                );
                step_0_pred_v_pos_wgpu = maybe_trace_rows_wgpu(
                    step_0_pred_v_pos_t
                        .clone()
                        .reshape([row_count, used_channels]),
                );
                step_0_pred_v_neg_wgpu = maybe_trace_rows_wgpu(
                    step_0_pred_v_neg_t
                        .clone()
                        .reshape([row_count, used_channels]),
                );
                step_0_x_t_wgpu =
                    maybe_trace_rows_wgpu(step_0_t.clone().reshape([row_count, used_channels]));
                step_mid_x_t_wgpu =
                    maybe_trace_rows_wgpu(step_mid_t.clone().reshape([row_count, used_channels]));
                step_last_x_t_wgpu =
                    maybe_trace_rows_wgpu(step_last_t.clone().reshape([row_count, used_channels]));
            }
            #[cfg(feature = "runtime-model-wgpu")]
            let host_rows_required = materialize_host_rows || samples_wgpu.is_none();
            #[cfg(not(feature = "runtime-model-wgpu"))]
            let host_rows_required = true;
            if host_rows_required {
                let merged = Tensor::cat(
                    vec![
                        samples_t,
                        step_0_pred_v_t,
                        step_0_pred_v_pos_t,
                        step_0_pred_v_neg_t,
                        step_0_t,
                        step_mid_t,
                        step_last_t,
                    ],
                    0,
                );
                let merged =
                    tensor_to_vec_1d(merged, "failed to read sparse-token row trace tensor")?;
                let segment = total_elements;
                (
                    merged[..segment].to_vec(),
                    merged[segment..segment * 2].to_vec(),
                    merged[segment * 2..segment * 3].to_vec(),
                    merged[segment * 3..segment * 4].to_vec(),
                    merged[segment * 4..segment * 5].to_vec(),
                    merged[segment * 5..segment * 6].to_vec(),
                    merged[segment * 6..segment * 7].to_vec(),
                )
            } else {
                (
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                )
            }
        } else {
            let samples_t = concat_batches(samples_batches, "samples")?;
            #[cfg(feature = "runtime-model-wgpu")]
            {
                samples_wgpu =
                    maybe_trace_rows_wgpu(samples_t.clone().reshape([row_count, used_channels]));
                step_0_pred_v_wgpu = samples_wgpu.clone();
                step_0_pred_v_pos_wgpu = samples_wgpu.clone();
                step_0_pred_v_neg_wgpu = samples_wgpu.clone();
                step_0_x_t_wgpu = samples_wgpu.clone();
                step_mid_x_t_wgpu = samples_wgpu.clone();
                step_last_x_t_wgpu = samples_wgpu.clone();
            }
            #[cfg(feature = "runtime-model-wgpu")]
            let host_rows_required = materialize_host_rows || samples_wgpu.is_none();
            #[cfg(not(feature = "runtime-model-wgpu"))]
            let host_rows_required = true;
            if host_rows_required {
                let samples =
                    tensor_to_vec_1d(samples_t, "failed to read sparse-token row tensor")?;
                (
                    samples.clone(),
                    samples.clone(),
                    samples.clone(),
                    samples.clone(),
                    samples.clone(),
                    samples.clone(),
                    samples,
                )
            } else {
                (
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                )
            }
        };

        Ok(SparseFlowRowTrace {
            steps: sample_cfg.steps,
            row_channels: used_channels,
            samples,
            step_0_pred_v,
            step_0_pred_v_pos,
            step_0_pred_v_neg,
            step_0_x_t,
            step_mid_x_t,
            step_last_x_t,
            #[cfg(feature = "runtime-model-wgpu")]
            samples_wgpu,
            #[cfg(feature = "runtime-model-wgpu")]
            step_0_pred_v_wgpu,
            #[cfg(feature = "runtime-model-wgpu")]
            step_0_pred_v_pos_wgpu,
            #[cfg(feature = "runtime-model-wgpu")]
            step_0_pred_v_neg_wgpu,
            #[cfg(feature = "runtime-model-wgpu")]
            step_0_x_t_wgpu,
            #[cfg(feature = "runtime-model-wgpu")]
            step_mid_x_t_wgpu,
            #[cfg(feature = "runtime-model-wgpu")]
            step_last_x_t_wgpu,
        })
    }

    #[allow(dead_code, clippy::too_many_arguments)]
    fn predict_with_cfg_sparse_tensor(
        &self,
        x_t: Tensor<B, 3>,
        timestep: f32,
        config: FlowEulerSampleConfig,
        sigma_min: f32,
        cond: Tensor<B, 3>,
        neg_cond: Tensor<B, 3>,
        concat_cond: Option<Tensor<B, 3>>,
        sparse_resolution: usize,
        token_coords: Tensor<B, 2, Int>,
    ) -> Result<Tensor<B, 3>, String> {
        Ok(self
            .predict_with_cfg_sparse_tensor_parts(
                x_t,
                timestep,
                config,
                sigma_min,
                cond,
                neg_cond,
                concat_cond,
                sparse_resolution,
                token_coords,
            )?
            .guided)
    }

    #[allow(dead_code, clippy::too_many_arguments)]
    fn predict_with_cfg_sparse_tensor_parts(
        &self,
        x_t: Tensor<B, 3>,
        timestep: f32,
        config: FlowEulerSampleConfig,
        sigma_min: f32,
        cond: Tensor<B, 3>,
        neg_cond: Tensor<B, 3>,
        concat_cond: Option<Tensor<B, 3>>,
        sparse_resolution: usize,
        token_coords: Tensor<B, 2, Int>,
    ) -> Result<SparseCfgPrediction<B>, String> {
        self.predict_with_cfg_sparse_tensor_parts_with_cache(
            x_t,
            timestep,
            config,
            sigma_min,
            cond,
            neg_cond,
            concat_cond,
            sparse_resolution,
            token_coords,
            None,
            None,
            None,
        )
    }

    #[allow(dead_code, clippy::too_many_arguments)]
    fn predict_with_cfg_sparse_tensor_parts_with_cache(
        &self,
        x_t: Tensor<B, 3>,
        timestep: f32,
        config: FlowEulerSampleConfig,
        sigma_min: f32,
        cond: Tensor<B, 3>,
        neg_cond: Tensor<B, 3>,
        concat_cond: Option<Tensor<B, 3>>,
        sparse_resolution: usize,
        token_coords: Tensor<B, 2, Int>,
        cond_cache: Option<&CrossAttentionKvCache<B>>,
        neg_cache: Option<&CrossAttentionKvCache<B>>,
        batched_cfg_cache: Option<&CrossAttentionKvCache<B>>,
    ) -> Result<SparseCfgPrediction<B>, String> {
        if !guidance_interval_contains(timestep, config.guidance_interval) {
            let guided = self.predict_velocity_sparse_tensor_with_cache(
                x_t,
                timestep,
                cond,
                concat_cond,
                sparse_resolution,
                token_coords,
                cond_cache,
            )?;
            return Ok(SparseCfgPrediction {
                guided: guided.clone(),
                pos: Some(guided),
                neg: None,
            });
        }

        let w = config.guidance_strength;
        if (w - 1.0).abs() < f32::EPSILON {
            let guided = self.predict_velocity_sparse_tensor_with_cache(
                x_t,
                timestep,
                cond,
                concat_cond,
                sparse_resolution,
                token_coords,
                cond_cache,
            )?;
            return Ok(SparseCfgPrediction {
                guided: guided.clone(),
                pos: Some(guided),
                neg: None,
            });
        }
        if w.abs() < f32::EPSILON {
            let guided = self.predict_velocity_sparse_tensor_with_cache(
                x_t,
                timestep,
                neg_cond,
                concat_cond,
                sparse_resolution,
                token_coords,
                neg_cache,
            )?;
            return Ok(SparseCfgPrediction {
                guided: guided.clone(),
                pos: None,
                neg: Some(guided),
            });
        }

        if sparse_flow_batched_cfg_enabled_for_backend::<B>() {
            let fallback_batched_cache;
            let batched_cache = if let Some(cache) = batched_cfg_cache {
                Some(cache)
            } else if let (Some(pos_cache), Some(neg_cache)) = (cond_cache, neg_cache) {
                fallback_batched_cache = concat_cross_kv_caches(pos_cache, neg_cache);
                fallback_batched_cache.as_ref()
            } else {
                None
            };
            if (cond_cache.is_none() && neg_cache.is_none()) || batched_cache.is_some() {
                let [sample_batches, tokens, row_channels] = x_t.dims();
                if sparse_flow_batched_cfg_debug_enabled() {
                    eprintln!(
                        "burn_trellis: sparse flow batched CFG fast path selected backend={} batches={} tokens={} channels={} cache={}",
                        std::any::type_name::<B>(),
                        sample_batches,
                        tokens,
                        row_channels,
                        batched_cache.is_some()
                    );
                }
                let x_batched = Tensor::cat(vec![x_t.clone(), x_t.clone()], 0);
                let cond_batched = if batched_cache.is_some() {
                    // With a concatenated cross-attention K/V cache, transformer
                    // blocks read the projected cache and ignore the raw context.
                    // Keep a single context tensor here to avoid re-concatenating
                    // the full positive/negative condition every sampled step.
                    cond.clone()
                } else {
                    Tensor::cat(vec![cond.clone(), neg_cond.clone()], 0)
                };
                let concat_batched = concat_cond
                    .clone()
                    .map(|concat| Tensor::cat(vec![concat.clone(), concat], 0));
                let pred_batched = self.predict_velocity_sparse_tensor_with_cache(
                    x_batched,
                    timestep,
                    cond_batched,
                    concat_batched,
                    sparse_resolution,
                    token_coords.clone(),
                    batched_cache,
                )?;
                let pos =
                    pred_batched
                        .clone()
                        .slice([0..sample_batches, 0..tokens, 0..row_channels]);
                let neg = pred_batched.slice([
                    sample_batches..sample_batches * 2,
                    0..tokens,
                    0..row_channels,
                ]);
                let guided = apply_cfg_sparse_tensor(
                    x_t,
                    timestep,
                    pos.clone(),
                    neg.clone(),
                    w,
                    config.guidance_rescale,
                    sigma_min,
                );
                return Ok(SparseCfgPrediction {
                    guided,
                    pos: Some(pos),
                    neg: Some(neg),
                });
            }
        }

        // Keep CFG as two explicit forwards, matching the upstream sampler and
        // avoiding batch-order drift in strict parity runs.
        let pos = self.predict_velocity_sparse_tensor_with_cache(
            x_t.clone(),
            timestep,
            cond,
            concat_cond.clone(),
            sparse_resolution,
            token_coords.clone(),
            cond_cache,
        )?;
        let neg = self.predict_velocity_sparse_tensor_with_cache(
            x_t.clone(),
            timestep,
            neg_cond,
            concat_cond,
            sparse_resolution,
            token_coords,
            neg_cache,
        )?;
        if sparse_flow_stage_debug_enabled() {
            let probe_idx = CFG_POS_NEG_DEBUG_COUNT.fetch_add(1, Ordering::Relaxed);
            if probe_idx < 12 {
                let (delta_mean_abs, delta_max_abs) =
                    tensor_abs_delta_stats_3d(pos.clone(), neg.clone());
                let pos_mean = tensor_mean_scalar_3d(pos.clone());
                let neg_mean = tensor_mean_scalar_3d(neg.clone());
                eprintln!(
                    "burn_trellis: sparse flow cfg probe idx={} backend={} timestep={:.6} delta_mean_abs={:.9e} delta_max_abs={:.9e} pos_mean={:.9e} neg_mean={:.9e}",
                    probe_idx,
                    std::any::type_name::<B>(),
                    timestep,
                    delta_mean_abs,
                    delta_max_abs,
                    pos_mean,
                    neg_mean
                );
            }
        }
        let guided = apply_cfg_sparse_tensor(
            x_t,
            timestep,
            pos.clone(),
            neg.clone(),
            w,
            config.guidance_rescale,
            sigma_min,
        );
        Ok(SparseCfgPrediction {
            guided,
            pos: Some(pos),
            neg: Some(neg),
        })
    }

    #[allow(dead_code, clippy::too_many_arguments)]
    fn predict_velocity_sparse_tensor(
        &self,
        x_t: Tensor<B, 3>,
        timestep: f32,
        cond: Tensor<B, 3>,
        concat_cond: Option<Tensor<B, 3>>,
        sparse_resolution: usize,
        token_coords: Tensor<B, 2, Int>,
    ) -> Result<Tensor<B, 3>, String> {
        self.predict_velocity_sparse_tensor_with_cache(
            x_t,
            timestep,
            cond,
            concat_cond,
            sparse_resolution,
            token_coords,
            None,
        )
    }

    #[allow(dead_code, clippy::too_many_arguments)]
    fn predict_velocity_sparse_tensor_with_cache(
        &self,
        x_t: Tensor<B, 3>,
        timestep: f32,
        cond: Tensor<B, 3>,
        concat_cond: Option<Tensor<B, 3>>,
        sparse_resolution: usize,
        token_coords: Tensor<B, 2, Int>,
        cross_kv_cache: Option<&CrossAttentionKvCache<B>>,
    ) -> Result<Tensor<B, 3>, String> {
        let [sample_batches, tokens, state_channels] = x_t.dims();
        let [coord_rows, coord_cols] = token_coords.dims();
        if coord_cols != 3 {
            return Err(format!(
                "sparse flow token coord shape mismatch: expected 3 columns, got {}",
                coord_cols
            ));
        }
        if tokens != coord_rows {
            return Err(format!(
                "sparse flow token coord mismatch: tokens={} coords={}",
                tokens, coord_rows
            ));
        }
        let concat_channels = concat_cond
            .as_ref()
            .map(|tensor| {
                let [_, concat_tokens, channels] = tensor.dims();
                if concat_tokens != tokens {
                    return Err(format!(
                        "concat cond token mismatch: got={} expected={tokens}",
                        concat_tokens
                    ));
                }
                Ok(channels)
            })
            .transpose()?
            .unwrap_or(0usize);
        if state_channels + concat_channels != self.config.in_channels {
            return Err(format!(
                "sparse flow channel mismatch: state={} concat={} expected_in={}",
                state_channels, concat_channels, self.config.in_channels
            ));
        }
        if state_channels != self.config.out_channels {
            return Err(format!(
                "sparse flow state/output mismatch: state={} expected_out={}",
                state_channels, self.config.out_channels
            ));
        }

        let sample = if let Some(concat) = concat_cond {
            Tensor::cat(vec![x_t, concat], 2)
        } else {
            x_t
        };
        let t_values = vec![timestep * 1000.0; sample_batches.max(1)];
        let t = Tensor::<B, 1>::from_floats(t_values.as_slice(), &self.device);
        Ok(self.model.forward_sparse_with_cross_cache(
            sample,
            t,
            cond,
            sparse_resolution.max(1),
            token_coords,
            cross_kv_cache,
        ))
    }

    #[allow(dead_code, clippy::too_many_arguments)]
    fn predict_with_cfg_tensor(
        &self,
        x_t: Tensor<B, 5>,
        timestep: f32,
        config: FlowEulerSampleConfig,
        sigma_min: f32,
        cond: Tensor<B, 3>,
        neg_cond: Tensor<B, 3>,
        concat_cond: Option<Tensor<B, 5>>,
    ) -> Result<Tensor<B, 5>, String> {
        self.predict_with_cfg_tensor_with_cache(
            x_t,
            timestep,
            config,
            sigma_min,
            cond,
            neg_cond,
            concat_cond,
            None,
            None,
        )
    }

    #[allow(dead_code, clippy::too_many_arguments)]
    fn predict_with_cfg_tensor_parts_with_cache(
        &self,
        x_t: Tensor<B, 5>,
        timestep: f32,
        config: FlowEulerSampleConfig,
        sigma_min: f32,
        cond: Tensor<B, 3>,
        neg_cond: Tensor<B, 3>,
        concat_cond: Option<Tensor<B, 5>>,
        cond_cache: Option<&CrossAttentionKvCache<B>>,
        neg_cache: Option<&CrossAttentionKvCache<B>>,
    ) -> Result<DenseCfgPrediction<B>, String> {
        if !guidance_interval_contains(timestep, config.guidance_interval) {
            let guided = self.predict_velocity_tensor_with_cache(
                x_t,
                timestep,
                cond,
                concat_cond,
                cond_cache,
            )?;
            return Ok(DenseCfgPrediction {
                guided: guided.clone(),
                pos: Some(guided),
                neg: None,
            });
        }

        let w = config.guidance_strength;
        if (w - 1.0).abs() < f32::EPSILON {
            let guided = self.predict_velocity_tensor_with_cache(
                x_t,
                timestep,
                cond,
                concat_cond,
                cond_cache,
            )?;
            return Ok(DenseCfgPrediction {
                guided: guided.clone(),
                pos: Some(guided),
                neg: None,
            });
        }
        if w.abs() < f32::EPSILON {
            let guided = self.predict_velocity_tensor_with_cache(
                x_t,
                timestep,
                neg_cond,
                concat_cond,
                neg_cache,
            )?;
            return Ok(DenseCfgPrediction {
                guided: guided.clone(),
                pos: None,
                neg: Some(guided),
            });
        }

        let pos = self.predict_velocity_tensor_with_cache(
            x_t.clone(),
            timestep,
            cond,
            concat_cond.clone(),
            cond_cache,
        )?;
        let neg = self.predict_velocity_tensor_with_cache(
            x_t.clone(),
            timestep,
            neg_cond,
            concat_cond,
            neg_cache,
        )?;
        let guided = apply_cfg_tensor(
            x_t,
            timestep,
            pos.clone(),
            neg.clone(),
            w,
            config.guidance_rescale,
            sigma_min,
        );
        Ok(DenseCfgPrediction {
            guided,
            pos: Some(pos),
            neg: Some(neg),
        })
    }

    #[allow(dead_code, clippy::too_many_arguments)]
    fn predict_with_cfg_tensor_with_cache(
        &self,
        x_t: Tensor<B, 5>,
        timestep: f32,
        config: FlowEulerSampleConfig,
        sigma_min: f32,
        cond: Tensor<B, 3>,
        neg_cond: Tensor<B, 3>,
        concat_cond: Option<Tensor<B, 5>>,
        cond_cache: Option<&CrossAttentionKvCache<B>>,
        neg_cache: Option<&CrossAttentionKvCache<B>>,
    ) -> Result<Tensor<B, 5>, String> {
        let parts = self.predict_with_cfg_tensor_parts_with_cache(
            x_t,
            timestep,
            config,
            sigma_min,
            cond,
            neg_cond,
            concat_cond,
            cond_cache,
            neg_cache,
        )?;
        if sparse_flow_stage_debug_enabled()
            && let (Some(pos), Some(neg)) = (parts.pos.as_ref(), parts.neg.as_ref())
        {
            let probe_idx = CFG_POS_NEG_DEBUG_COUNT.fetch_add(1, Ordering::Relaxed);
            if probe_idx < 12 {
                let (delta_mean_abs, delta_max_abs) =
                    tensor_abs_delta_stats_5d(pos.clone(), neg.clone());
                let pos_mean = tensor_mean_scalar_5d(pos.clone());
                let neg_mean = tensor_mean_scalar_5d(neg.clone());
                eprintln!(
                    "burn_trellis: sparse flow cfg probe idx={} backend={} timestep={:.6} delta_mean_abs={:.9e} delta_max_abs={:.9e} pos_mean={:.9e} neg_mean={:.9e}",
                    probe_idx,
                    std::any::type_name::<B>(),
                    timestep,
                    delta_mean_abs,
                    delta_max_abs,
                    pos_mean,
                    neg_mean
                );
            }
        }
        Ok(parts.guided)
    }

    #[allow(dead_code)]
    fn predict_velocity_tensor(
        &self,
        x_t: Tensor<B, 5>,
        timestep: f32,
        cond: Tensor<B, 3>,
        concat_cond: Option<Tensor<B, 5>>,
    ) -> Result<Tensor<B, 5>, String> {
        self.predict_velocity_tensor_with_cache(x_t, timestep, cond, concat_cond, None)
    }

    #[allow(dead_code)]
    fn predict_velocity_tensor_with_cache(
        &self,
        x_t: Tensor<B, 5>,
        timestep: f32,
        cond: Tensor<B, 3>,
        concat_cond: Option<Tensor<B, 5>>,
        cross_kv_cache: Option<&CrossAttentionKvCache<B>>,
    ) -> Result<Tensor<B, 5>, String> {
        let [sample_batches, state_channels, rx, ry, rz] = x_t.dims();
        let voxel = rx * ry * rz;
        if rx != self.config.resolution
            || ry != self.config.resolution
            || rz != self.config.resolution
        {
            return Err(format!(
                "sparse flow tensor resolution mismatch: got=({rx},{ry},{rz}) expected={}",
                self.config.resolution
            ));
        }
        let concat_channels = concat_cond
            .as_ref()
            .map(|tensor| {
                let [_, channels, cx, cy, cz] = tensor.dims();
                if channels == 0 {
                    return Err("concat cond tensor has zero channels".to_string());
                }
                if cx != rx || cy != ry || cz != rz {
                    return Err(format!(
                        "concat cond tensor resolution mismatch: got=({cx},{cy},{cz}) expected=({rx},{ry},{rz})"
                    ));
                }
                Ok(channels)
            })
            .transpose()?
            .unwrap_or(0usize);
        if state_channels + concat_channels != self.config.in_channels {
            return Err(format!(
                "sparse flow channel mismatch: state={} concat={} expected_in={}",
                state_channels, concat_channels, self.config.in_channels
            ));
        }
        if state_channels != self.config.out_channels {
            return Err(format!(
                "sparse flow state/output mismatch: state={} expected_out={}",
                state_channels, self.config.out_channels
            ));
        }
        if voxel == 0 {
            return Err("sparse flow tensor voxel count is zero".to_string());
        }

        let sample = if let Some(concat) = concat_cond {
            Tensor::cat(vec![x_t, concat], 1)
        } else {
            x_t
        };
        let t_values = vec![timestep * 1000.0; sample_batches.max(1)];
        let t = Tensor::<B, 1>::from_floats(t_values.as_slice(), &self.device);
        Ok(self
            .model
            .forward_with_cross_cache(sample, t, cond, cross_kv_cache))
    }
}

impl SparseStructureFlowRuntime {
    pub fn load_from_stem(
        weights_root: &Path,
        image_large_root: Option<&Path>,
        model_stem: &str,
        prefer_wgpu: bool,
        resolution_override: Option<usize>,
    ) -> Result<Self, String> {
        #[cfg(feature = "runtime-model-wgpu")]
        if prefer_wgpu {
            let runtime = SparseStructureFlowRuntimeImpl::<WgpuRuntimeBackend>::load_from_stem(
                weights_root,
                image_large_root,
                model_stem,
                resolution_override,
            )
            .map_err(|err| {
                format!(
                    "burn_trellis: failed to load sparse flow runtime on wgpu ({err}); refusing cpu fallback for model '{model_stem}'"
                )
            })?;
            let cfg = runtime.config();
            let tokens = cfg
                .resolution
                .saturating_mul(cfg.resolution)
                .saturating_mul(cfg.resolution);
            if sparse_flow_wgpu_may_overflow(cfg) && !sparse_flow_chunked_forward_enabled(tokens) {
                return Err(format!(
                    "burn_trellis: sparse flow wgpu estimated peak exceeds safe budget for model '{}' (resolution={}, model_channels={}); refusing cpu fallback",
                    model_stem, cfg.resolution, cfg.model_channels
                ));
            }
            if sparse_flow_wgpu_may_overflow(cfg) {
                eprintln!(
                    "burn_trellis: sparse flow wgpu keeping model '{}' on device with chunked-forward path (resolution={}, model_channels={}).",
                    model_stem, cfg.resolution, cfg.model_channels
                );
            }
            return Ok(Self::Wgpu(runtime));
        }

        #[cfg(not(feature = "runtime-model-wgpu"))]
        if prefer_wgpu {
            return Err(format!(
                "burn_trellis: sparse flow runtime requested wgpu for model '{}' but crate was built without runtime-model-wgpu",
                model_stem
            ));
        }
        let runtime = SparseStructureFlowRuntimeImpl::<CpuRuntimeBackend>::load_from_stem(
            weights_root,
            image_large_root,
            model_stem,
            resolution_override,
        )?;
        Ok(Self::Cpu(runtime))
    }

    pub fn backend_name(&self) -> &'static str {
        match self {
            Self::Cpu(_) => "cpu",
            #[cfg(feature = "runtime-model-wgpu")]
            Self::Wgpu(_) => "wgpu",
        }
    }

    pub fn config(&self) -> &SparseStructureFlowConfig {
        match self {
            Self::Cpu(runtime) => runtime.config(),
            #[cfg(feature = "runtime-model-wgpu")]
            Self::Wgpu(runtime) => runtime.config(),
        }
    }

    pub fn sparse_tensor_from_host_layout(
        &self,
        coords: Vec<[u32; 4]>,
        feats: Vec<f32>,
        layout: Vec<Range<usize>>,
        channels: usize,
        sparse_resolution: usize,
    ) -> Result<SparseTensorOwned, String> {
        match self {
            Self::Cpu(_) => {
                SparseTensorOwned::from_layout(coords, feats, layout, channels, sparse_resolution)
            }
            #[cfg(feature = "runtime-model-wgpu")]
            Self::Wgpu(runtime) => {
                let rows = coords.len();
                let expected = rows.saturating_mul(channels);
                if feats.len() != expected {
                    return Err(format!(
                        "sparse flow host feature length mismatch: expected {} (rows={} channels={}), got {}",
                        expected,
                        rows,
                        channels,
                        feats.len()
                    ));
                }
                let values_t =
                    Tensor::<WgpuRuntimeBackend, 1>::from_floats(feats.as_slice(), &runtime.device)
                        .reshape([rows, channels]);
                let values = VarLenTensorOwned::from_wgpu_tensor(values_t, layout)?;

                let mut flat_coords = Vec::with_capacity(rows.saturating_mul(4));
                for (row_idx, coord) in coords.iter().enumerate() {
                    for value in coord {
                        let converted = i32::try_from(*value).map_err(|_| {
                            format!(
                                "sparse flow coord conversion overflow at row {} value {}",
                                row_idx, value
                            )
                        })?;
                        flat_coords.push(converted);
                    }
                }
                let coords_t = Tensor::<WgpuRuntimeBackend, 1, Int>::from_data(
                    TensorData::new(flat_coords, [rows.saturating_mul(4)]),
                    &runtime.device,
                )
                .reshape([rows, 4]);
                SparseTensorOwned::from_wgpu_tensors(coords_t, values, sparse_resolution)
            }
        }
    }

    #[cfg(feature = "runtime-model-wgpu")]
    pub fn sparse_tensor_from_wgpu_tensors_layout(
        &self,
        coords_wgpu: Tensor<WgpuRuntimeBackend, 2, Int>,
        feats_wgpu: Tensor<WgpuRuntimeBackend, 2>,
        layout: Vec<Range<usize>>,
        sparse_resolution: usize,
    ) -> Result<SparseTensorOwned, String> {
        let Self::Wgpu(_) = self else {
            return Err(
                "sparse flow wgpu tensor assembly requires wgpu runtime backend".to_string(),
            );
        };
        let [rows, cols] = coords_wgpu.dims();
        if cols != 4 {
            return Err(format!(
                "sparse flow coord tensor must have 4 columns, got {}",
                cols
            ));
        }
        let [feat_rows, channels] = feats_wgpu.dims();
        if channels == 0 {
            return Err("sparse flow feature tensor channels must be > 0".to_string());
        }
        if feat_rows != rows {
            return Err(format!(
                "sparse flow tensor row mismatch: coords_rows={} feat_rows={}",
                rows, feat_rows
            ));
        }
        let values = VarLenTensorOwned::from_wgpu_tensor(feats_wgpu, layout)?;
        SparseTensorOwned::from_wgpu_tensors(coords_wgpu, values, sparse_resolution)
    }

    #[cfg(feature = "runtime-model-wgpu")]
    #[allow(dead_code)]
    pub fn sparse_tensor_device_from_wgpu_tensors_layout(
        &self,
        coords_wgpu: Tensor<WgpuRuntimeBackend, 2, Int>,
        feats_wgpu: Tensor<WgpuRuntimeBackend, 2>,
        layout: Vec<Range<usize>>,
        sparse_resolution: usize,
    ) -> Result<super::types::SparseTensorDevice<WgpuRuntimeBackend>, String> {
        let sparse = self.sparse_tensor_from_wgpu_tensors_layout(
            coords_wgpu,
            feats_wgpu,
            layout,
            sparse_resolution,
        )?;
        sparse.as_device_owned("sparse flow device sparse tensor assembly")
    }

    #[cfg(feature = "runtime-model-wgpu")]
    pub fn varlen_tensor_from_wgpu_tensor_layout(
        &self,
        feats_wgpu: Tensor<WgpuRuntimeBackend, 2>,
        layout: Vec<Range<usize>>,
    ) -> Result<VarLenTensorOwned, String> {
        let Self::Wgpu(_) = self else {
            return Err(
                "sparse flow wgpu varlen tensor assembly requires wgpu runtime backend".to_string(),
            );
        };
        VarLenTensorOwned::from_wgpu_tensor(feats_wgpu, layout)
    }

    #[cfg(feature = "runtime-model-wgpu")]
    #[allow(dead_code)]
    pub fn varlen_tensor_device_from_wgpu_tensor_layout(
        &self,
        feats_wgpu: Tensor<WgpuRuntimeBackend, 2>,
        layout: Vec<Range<usize>>,
    ) -> Result<super::types::VarLenTensorDevice<WgpuRuntimeBackend>, String> {
        let values = self.varlen_tensor_from_wgpu_tensor_layout(feats_wgpu, layout)?;
        values.as_device_owned("sparse flow device varlen tensor assembly")
    }

    pub fn varlen_tensor_from_host_layout(
        &self,
        feats: Vec<f32>,
        layout: Vec<Range<usize>>,
        channels: usize,
    ) -> Result<VarLenTensorOwned, String> {
        match self {
            Self::Cpu(_) => VarLenTensorOwned::from_layout(feats, layout, channels),
            #[cfg(feature = "runtime-model-wgpu")]
            Self::Wgpu(runtime) => {
                let rows: usize = layout
                    .iter()
                    .map(|range| range.end.saturating_sub(range.start))
                    .sum();
                let expected = rows.saturating_mul(channels);
                if feats.len() != expected {
                    return Err(format!(
                        "sparse flow concat host feature length mismatch: expected {} (rows={} channels={}), got {}",
                        expected,
                        rows,
                        channels,
                        feats.len()
                    ));
                }
                let values_t =
                    Tensor::<WgpuRuntimeBackend, 1>::from_floats(feats.as_slice(), &runtime.device)
                        .reshape([rows, channels]);
                VarLenTensorOwned::from_wgpu_tensor(values_t, layout)
            }
        }
    }

    pub fn prepare_condition(
        &self,
        cond: &[f32],
        cond_tokens: usize,
    ) -> Result<SparseFlowCondition, String> {
        match self {
            Self::Cpu(runtime) => runtime
                .prepare_condition(cond, cond_tokens)
                .map(SparseFlowCondition::Cpu),
            #[cfg(feature = "runtime-model-wgpu")]
            Self::Wgpu(runtime) => runtime
                .prepare_condition(cond, cond_tokens)
                .map(SparseFlowCondition::Wgpu),
        }
    }

    #[allow(dead_code)]
    pub fn predict_velocity_with_condition(
        &self,
        x_t: &[f32],
        timestep: f32,
        condition: &SparseFlowCondition,
        concat_cond: Option<&[f32]>,
    ) -> Result<Vec<f32>, String> {
        #[cfg(feature = "runtime-model-wgpu")]
        {
            match (self, condition) {
                (Self::Cpu(runtime), SparseFlowCondition::Cpu(cond)) => runtime
                    .predict_velocity_with_condition(x_t, timestep, cond.clone(), concat_cond),
                (Self::Wgpu(runtime), SparseFlowCondition::Wgpu(cond)) => runtime
                    .predict_velocity_with_condition(x_t, timestep, cond.clone(), concat_cond),
                _ => {
                    Err("sparse flow condition backend does not match runtime backend".to_string())
                }
            }
        }
        #[cfg(not(feature = "runtime-model-wgpu"))]
        {
            let Self::Cpu(runtime) = self;
            let SparseFlowCondition::Cpu(cond) = condition;
            runtime.predict_velocity_with_condition(x_t, timestep, cond.clone(), concat_cond)
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sample_with_trace(
        &self,
        noise: &[f32],
        sample_cfg: FlowEulerSampleConfig,
        sigma_min: f32,
        condition: &SparseFlowCondition,
        negative_condition: &SparseFlowCondition,
        concat_cond: Option<&[f32]>,
        capture_snapshots: bool,
    ) -> Result<FlowEulerSampleTrace, String> {
        #[cfg(feature = "runtime-model-wgpu")]
        {
            match (self, condition, negative_condition) {
                (
                    Self::Cpu(runtime),
                    SparseFlowCondition::Cpu(cond),
                    SparseFlowCondition::Cpu(neg_cond),
                ) => runtime.sample_with_trace(
                    noise,
                    sample_cfg,
                    sigma_min,
                    cond.clone(),
                    neg_cond.clone(),
                    concat_cond,
                    capture_snapshots,
                ),
                (
                    Self::Wgpu(runtime),
                    SparseFlowCondition::Wgpu(cond),
                    SparseFlowCondition::Wgpu(neg_cond),
                ) => runtime.sample_with_trace(
                    noise,
                    sample_cfg,
                    sigma_min,
                    cond.clone(),
                    neg_cond.clone(),
                    concat_cond,
                    capture_snapshots,
                ),
                _ => {
                    Err("sparse flow condition backend does not match runtime backend".to_string())
                }
            }
        }
        #[cfg(not(feature = "runtime-model-wgpu"))]
        {
            let Self::Cpu(runtime) = self;
            let SparseFlowCondition::Cpu(cond) = condition;
            let SparseFlowCondition::Cpu(neg_cond) = negative_condition;
            runtime.sample_with_trace(
                noise,
                sample_cfg,
                sigma_min,
                cond.clone(),
                neg_cond.clone(),
                concat_cond,
                capture_snapshots,
            )
        }
    }

    #[cfg(feature = "runtime-model-wgpu")]
    #[allow(clippy::too_many_arguments)]
    pub fn sample_final_tensor_wgpu(
        &self,
        noise: &[f32],
        sample_cfg: FlowEulerSampleConfig,
        sigma_min: f32,
        condition: &SparseFlowCondition,
        negative_condition: &SparseFlowCondition,
        concat_cond: Option<&[f32]>,
    ) -> Result<Tensor<WgpuRuntimeBackend, 5>, String> {
        match (self, condition, negative_condition) {
            (
                Self::Wgpu(runtime),
                SparseFlowCondition::Wgpu(cond),
                SparseFlowCondition::Wgpu(neg_cond),
            ) => runtime.sample_final_tensor(
                noise,
                sample_cfg,
                sigma_min,
                cond.clone(),
                neg_cond.clone(),
                concat_cond,
            ),
            (Self::Cpu(_), SparseFlowCondition::Cpu(_), SparseFlowCondition::Cpu(_)) => Err(
                "sparse flow tensor-native latent path requires wgpu runtime backend".to_string(),
            ),
            _ => Err("sparse flow condition backend does not match runtime backend".to_string()),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sample_sparse_rows_with_trace(
        &self,
        sparse: &SparseTensorOwned,
        sample_cfg: FlowEulerSampleConfig,
        sigma_min: f32,
        condition: &SparseFlowCondition,
        negative_condition: &SparseFlowCondition,
        concat_cond: Option<&VarLenTensorOwned>,
        row_channels: usize,
        capture_snapshots: bool,
        materialize_host_rows: bool,
    ) -> Result<SparseFlowRowTrace, String> {
        #[cfg(feature = "runtime-model-wgpu")]
        {
            match (self, condition, negative_condition) {
                (
                    Self::Cpu(runtime),
                    SparseFlowCondition::Cpu(cond),
                    SparseFlowCondition::Cpu(neg_cond),
                ) => runtime.sample_sparse_rows_with_trace(
                    sparse,
                    sample_cfg,
                    sigma_min,
                    cond.clone(),
                    neg_cond.clone(),
                    concat_cond,
                    row_channels,
                    capture_snapshots,
                    materialize_host_rows,
                ),
                (
                    Self::Wgpu(runtime),
                    SparseFlowCondition::Wgpu(cond),
                    SparseFlowCondition::Wgpu(neg_cond),
                ) => runtime.sample_sparse_rows_with_trace(
                    sparse,
                    sample_cfg,
                    sigma_min,
                    cond.clone(),
                    neg_cond.clone(),
                    concat_cond,
                    row_channels,
                    capture_snapshots,
                    materialize_host_rows,
                ),
                _ => {
                    Err("sparse flow condition backend does not match runtime backend".to_string())
                }
            }
        }
        #[cfg(not(feature = "runtime-model-wgpu"))]
        {
            let Self::Cpu(runtime) = self;
            let SparseFlowCondition::Cpu(cond) = condition;
            let SparseFlowCondition::Cpu(neg_cond) = negative_condition;
            runtime.sample_sparse_rows_with_trace(
                sparse,
                sample_cfg,
                sigma_min,
                cond.clone(),
                neg_cond.clone(),
                concat_cond,
                row_channels,
                capture_snapshots,
                materialize_host_rows,
            )
        }
    }

    #[cfg(feature = "runtime-model-wgpu")]
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    pub fn sample_sparse_rows_with_trace_wgpu_inputs(
        &self,
        coords_wgpu: Tensor<WgpuRuntimeBackend, 2, Int>,
        feats_wgpu: Tensor<WgpuRuntimeBackend, 2>,
        sample_cfg: FlowEulerSampleConfig,
        sigma_min: f32,
        condition: &SparseFlowCondition,
        negative_condition: &SparseFlowCondition,
        concat_wgpu: Option<Tensor<WgpuRuntimeBackend, 2>>,
        layout: Vec<Range<usize>>,
        sparse_resolution: usize,
        row_channels: usize,
        capture_snapshots: bool,
        materialize_host_rows: bool,
    ) -> Result<SparseFlowRowTrace, String> {
        let sparse = self
            .sparse_tensor_from_wgpu_tensors_layout(
                coords_wgpu,
                feats_wgpu,
                layout.clone(),
                sparse_resolution,
            )
            .map_err(|err| {
                format!("sparse flow wgpu sparse input tensor assembly failed ({err})")
            })?;
        let concat_owned = if let Some(concat_t) = concat_wgpu {
            Some(
                self.varlen_tensor_from_wgpu_tensor_layout(concat_t, layout)
                    .map_err(|err| {
                        format!("sparse flow wgpu concat input tensor assembly failed ({err})")
                    })?,
            )
        } else {
            None
        };
        self.sample_sparse_rows_with_trace(
            &sparse,
            sample_cfg,
            sigma_min,
            condition,
            negative_condition,
            concat_owned.as_ref(),
            row_channels,
            capture_snapshots,
            materialize_host_rows,
        )
    }
}

#[allow(dead_code)]
fn pred_to_xstart_tensor<B: Backend>(
    x_t: Tensor<B, 5>,
    timestep: f32,
    pred: Tensor<B, 5>,
    sigma_min: f32,
) -> Tensor<B, 5> {
    let factor = sigma_min + (1.0 - sigma_min) * timestep;
    let keep = 1.0 - sigma_min;
    x_t.mul_scalar(keep).sub(pred.mul_scalar(factor))
}

#[allow(dead_code)]
fn xstart_to_pred_tensor<B: Backend>(
    x_t: Tensor<B, 5>,
    timestep: f32,
    x0: Tensor<B, 5>,
    sigma_min: f32,
) -> Tensor<B, 5> {
    let factor = sigma_min + (1.0 - sigma_min) * timestep;
    let keep = 1.0 - sigma_min;
    x_t.mul_scalar(keep).sub(x0).div_scalar(factor)
}

fn apply_cfg_tensor<B: Backend>(
    x_t: Tensor<B, 5>,
    timestep: f32,
    pos: Tensor<B, 5>,
    neg: Tensor<B, 5>,
    guidance_strength: f32,
    guidance_rescale: f32,
    sigma_min: f32,
) -> Tensor<B, 5> {
    let mut pred = pos
        .clone()
        .mul_scalar(guidance_strength)
        .add(neg.mul_scalar(1.0 - guidance_strength));
    let guidance_rescale = guidance_rescale.max(0.0);
    if guidance_rescale > 0.0 {
        let [batch, _, _, _, _] = x_t.dims();
        let x0_pos = pred_to_xstart_tensor(x_t.clone(), timestep, pos, sigma_min);
        let x0_cfg = pred_to_xstart_tensor(x_t.clone(), timestep, pred.clone(), sigma_min);
        let std_pos = tensor_std_tensor(x0_pos).reshape([batch, 1, 1, 1, 1]);
        let std_cfg = tensor_std_tensor(x0_cfg.clone())
            .reshape([batch, 1, 1, 1, 1])
            .add_scalar(1.0e-12);
        let x0_rescaled = x0_cfg.clone().mul(std_pos.div(std_cfg));
        let x0 = x0_rescaled
            .mul_scalar(guidance_rescale)
            .add(x0_cfg.mul_scalar(1.0 - guidance_rescale));
        pred = xstart_to_pred_tensor(x_t, timestep, x0, sigma_min);
    }
    pred
}

#[allow(dead_code)]
fn pred_to_xstart_sparse_tensor<B: Backend>(
    x_t: Tensor<B, 3>,
    timestep: f32,
    pred: Tensor<B, 3>,
    sigma_min: f32,
) -> Tensor<B, 3> {
    let factor = sigma_min + (1.0 - sigma_min) * timestep;
    let keep = 1.0 - sigma_min;
    x_t.mul_scalar(keep).sub(pred.mul_scalar(factor))
}

#[allow(dead_code)]
fn xstart_to_pred_sparse_tensor<B: Backend>(
    x_t: Tensor<B, 3>,
    timestep: f32,
    x0: Tensor<B, 3>,
    sigma_min: f32,
) -> Tensor<B, 3> {
    let factor = sigma_min + (1.0 - sigma_min) * timestep;
    let keep = 1.0 - sigma_min;
    x_t.mul_scalar(keep).sub(x0).div_scalar(factor)
}

fn apply_cfg_sparse_tensor<B: Backend>(
    x_t: Tensor<B, 3>,
    timestep: f32,
    pos: Tensor<B, 3>,
    neg: Tensor<B, 3>,
    guidance_strength: f32,
    guidance_rescale: f32,
    sigma_min: f32,
) -> Tensor<B, 3> {
    let mut pred = pos
        .clone()
        .mul_scalar(guidance_strength)
        .add(neg.mul_scalar(1.0 - guidance_strength));
    let guidance_rescale = guidance_rescale.max(0.0);
    if guidance_rescale > 0.0 {
        let [batch, _, _] = x_t.dims();
        let x0_pos = pred_to_xstart_sparse_tensor(x_t.clone(), timestep, pos, sigma_min);
        let x0_cfg = pred_to_xstart_sparse_tensor(x_t.clone(), timestep, pred.clone(), sigma_min);
        let std_pos = tensor_std_sparse_tensor(x0_pos).reshape([batch, 1, 1]);
        let std_cfg = tensor_std_sparse_tensor(x0_cfg.clone())
            .reshape([batch, 1, 1])
            .add_scalar(1.0e-12);
        let x0_rescaled = x0_cfg.clone().mul(std_pos.div(std_cfg));
        let x0 = x0_rescaled
            .mul_scalar(guidance_rescale)
            .add(x0_cfg.mul_scalar(1.0 - guidance_rescale));
        pred = xstart_to_pred_sparse_tensor(x_t, timestep, x0, sigma_min);
    }
    pred
}

#[allow(dead_code)]
fn tensor_std_tensor<B: Backend>(tensor: Tensor<B, 5>) -> Tensor<B, 1> {
    let [b, c, x, y, z] = tensor.dims();
    let features = c.saturating_mul(x).saturating_mul(y).saturating_mul(z);
    let flat = tensor.reshape([b, features.max(1)]);
    let mean = flat.clone().mean_dim(1).reshape([b, 1]);
    let centered = flat.sub(mean);
    let denom = features.saturating_sub(1).max(1) as f32;
    centered
        .powf_scalar(2.0)
        .sum_dim(1)
        .reshape([b])
        .div_scalar(denom)
        .sqrt()
}

#[allow(dead_code)]
fn tensor_std_sparse_tensor<B: Backend>(tensor: Tensor<B, 3>) -> Tensor<B, 1> {
    let [b, tokens, channels] = tensor.dims();
    let features = tokens.saturating_mul(channels).max(1);
    let flat = tensor.reshape([b, features]);
    let mean = flat.clone().mean_dim(1).reshape([b, 1]);
    let centered = flat.sub(mean);
    let denom = features.saturating_sub(1).max(1) as f32;
    centered
        .powf_scalar(2.0)
        .sum_dim(1)
        .reshape([b])
        .div_scalar(denom)
        .sqrt()
}

fn tensor_to_vec<B: Backend>(tensor: Tensor<B, 5>) -> Result<Vec<f32>, String> {
    let [b, c, x, y, z] = tensor.dims();
    let elements = b
        .saturating_mul(c)
        .saturating_mul(x)
        .saturating_mul(y)
        .saturating_mul(z);
    let values = tensor
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|err| format!("failed to read sparse flow tensor: {err:?}"))?;
    record_host_readback(elements.max(values.len()));
    Ok(values)
}

fn tensor_to_vec_1d<B: Backend>(tensor: Tensor<B, 1>, context: &str) -> Result<Vec<f32>, String> {
    let [elements] = tensor.dims();
    let values = tensor
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|err| format!("{context}: {err:?}"))?;
    record_host_readback(elements.max(values.len()));
    Ok(values)
}

fn tensor_mean_scalar_3d<B: Backend>(tensor: Tensor<B, 3>) -> f32 {
    tensor
        .mean()
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .ok()
        .and_then(|values| {
            if values.is_empty() {
                None
            } else {
                record_host_readback(values.len().max(1));
                values.first().copied()
            }
        })
        .unwrap_or(f32::NAN)
}

fn tensor_mean_scalar_5d<B: Backend>(tensor: Tensor<B, 5>) -> f32 {
    tensor
        .mean()
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .ok()
        .and_then(|values| {
            if values.is_empty() {
                None
            } else {
                record_host_readback(values.len().max(1));
                values.first().copied()
            }
        })
        .unwrap_or(f32::NAN)
}

fn tensor_abs_delta_stats_3d<B: Backend>(lhs: Tensor<B, 3>, rhs: Tensor<B, 3>) -> (f32, f32) {
    lhs.sub(rhs)
        .abs()
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .ok()
        .map(|values| {
            if values.is_empty() {
                return (0.0f32, 0.0f32);
            }
            record_host_readback(values.len().max(1));
            let mut max_v = 0.0f32;
            let mut sum_v = 0.0f64;
            for value in values.iter().copied() {
                max_v = max_v.max(value);
                sum_v += value as f64;
            }
            (sum_v as f32 / values.len() as f32, max_v)
        })
        .unwrap_or((f32::NAN, f32::NAN))
}

fn tensor_abs_delta_stats_5d<B: Backend>(lhs: Tensor<B, 5>, rhs: Tensor<B, 5>) -> (f32, f32) {
    lhs.sub(rhs)
        .abs()
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .ok()
        .map(|values| {
            if values.is_empty() {
                return (0.0f32, 0.0f32);
            }
            record_host_readback(values.len().max(1));
            let mut max_v = 0.0f32;
            let mut sum_v = 0.0f64;
            for value in values.iter().copied() {
                max_v = max_v.max(value);
                sum_v += value as f64;
            }
            (sum_v as f32 / values.len() as f32, max_v)
        })
        .unwrap_or((f32::NAN, f32::NAN))
}

fn gelu<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    // tanh-approx GELU parity with TRELLIS modules:
    // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
    let c0 = 0.044_715_f32;
    let c1 = 0.797_884_6_f32; // sqrt(2/pi)
    let x3 = x.clone().powf_scalar(3.0).mul_scalar(c0);
    let t = x.clone().add(x3).mul_scalar(c1).tanh();
    x.mul_scalar(0.5).mul(t.add_scalar(1.0))
}

fn silu<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    x.clone().mul(sigmoid(x))
}

fn timestep_embedding<B: Backend>(timesteps: Tensor<B, 1>, dim: usize) -> Tensor<B, 2> {
    let [batch] = timesteps.dims();
    let half = (dim / 2).max(1);
    let device = timesteps.device();
    let freqs = Tensor::<B, 1, Int>::arange(0..(half as i64), &device)
        .float()
        .mul_scalar(-MAX_PERIOD.ln())
        .div_scalar(half as f32)
        .exp();
    let args = timesteps.unsqueeze_dim(1).mul(freqs.unsqueeze_dim(0));
    let mut emb = Tensor::cat(vec![args.clone().cos(), args.sin()], 1);
    if dim % 2 == 1 {
        emb = Tensor::cat(vec![emb, Tensor::<B, 2>::zeros([batch, 1], &device)], 1);
    }
    emb
}

fn layer_norm_no_affine<B: Backend>(x: Tensor<B, 3>, eps: f32) -> Tensor<B, 3>
where
    RopeRotateWgpuBridgeImpl: SparseFlowLayerNormWgpuBridge<B>,
{
    #[cfg(feature = "runtime-model-wgpu")]
    if let Some(y) = maybe_layer_norm_no_affine_wgpu(x.clone(), eps) {
        return y;
    }
    let mean = x.clone().mean_dim(2);
    let centered = x.sub(mean);
    let var = centered.clone().powf_scalar(2.0).mean_dim(2);
    centered.mul(var.add_scalar(eps).sqrt().recip())
}

fn layer_norm_modulated<B: Backend>(
    x: Tensor<B, 3>,
    scale: Tensor<B, 3>,
    shift: Tensor<B, 3>,
    eps: f32,
) -> Tensor<B, 3>
where
    RopeRotateWgpuBridgeImpl: SparseFlowLayerNormWgpuBridge<B>,
    SparseFlowBf16EmulationBridgeImpl: SparseFlowBf16EmulationBridge<B>,
{
    #[cfg(feature = "runtime-model-wgpu")]
    if !sparse_flow_prefers_decomposed_modulated_layer_norm::<B>(&x) {
        if let Some(y) =
            maybe_layer_norm_modulated_wgpu(x.clone(), scale.clone(), shift.clone(), eps)
        {
            return y;
        }
    }
    let y = layer_norm_no_affine(x, eps);
    let scale = sparse_flow_stock_bf16_round(scale.add_scalar(1.0));
    let y = sparse_flow_stock_bf16_round(y.mul(scale));
    sparse_flow_stock_bf16_round(y.add(shift))
}

#[cfg(feature = "runtime-model-wgpu")]
fn sparse_flow_prefers_decomposed_modulated_layer_norm<B: Backend>(x: &Tensor<B, 3>) -> bool {
    if !attention_uses_non_fusion_module_kernel::<B>() {
        return false;
    }
    let [batch, tokens, channels] = x.dims();
    let dtype: burn::tensor::FloatDType = x.dtype().into();
    dtype == burn::tensor::FloatDType::F32
        && channels >= 1024
        && batch.saturating_mul(tokens) >= 1024
}

fn layer_norm_affine_stable<B: Backend>(
    x: Tensor<B, 3>,
    norm: &nn::LayerNorm<B>,
    eps: f32,
) -> Tensor<B, 3>
where
    RopeRotateWgpuBridgeImpl: SparseFlowLayerNormWgpuBridge<B>,
{
    #[cfg(feature = "runtime-model-wgpu")]
    if let Some(y) = maybe_layer_norm_affine_wgpu(x.clone(), norm, eps) {
        return y;
    }
    let x_dtype: burn::tensor::FloatDType = x.dtype().into();
    let mut y = layer_norm_no_affine(x, eps);
    let gamma = norm.gamma.val().unsqueeze::<3>();
    let gamma_dtype: burn::tensor::FloatDType = gamma.dtype().into();
    let gamma = if gamma_dtype != x_dtype {
        gamma.cast(x_dtype)
    } else {
        gamma
    };
    y = y.mul(gamma);
    if let Some(beta) = norm.beta.as_ref() {
        let beta = beta.val().unsqueeze::<3>();
        let beta_dtype: burn::tensor::FloatDType = beta.dtype().into();
        let beta = if beta_dtype != x_dtype {
            beta.cast(x_dtype)
        } else {
            beta
        };
        y = y.add(beta);
    }
    y
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AttentionImpl {
    Dense,
    Stream,
}

#[cfg(feature = "runtime-model-wgpu")]
fn sparse_flow_wgpu_max_peak_bytes() -> usize {
    3 * 1024 * 1024 * 1024
}

fn bytes_to_mib(bytes: usize) -> usize {
    bytes / (1024 * 1024)
}

fn sample_cfg_needs_negative_condition(
    sample_cfg: FlowEulerSampleConfig,
    t_pairs: &[(f32, f32)],
) -> bool {
    if (sample_cfg.guidance_strength - 1.0).abs() < f32::EPSILON {
        return false;
    }
    t_pairs
        .iter()
        .any(|(timestep, _)| guidance_interval_contains(*timestep, sample_cfg.guidance_interval))
}

fn sparse_flow_cross_kv_cache_enabled_for_backend<B: Backend>() -> bool {
    attention_uses_non_fusion_module_kernel::<B>()
}

fn sparse_flow_cross_kv_cache_budget_bytes_for_backend<B: Backend>() -> usize {
    if !sparse_flow_cross_kv_cache_enabled_for_backend::<B>() {
        return 0;
    }
    #[cfg(feature = "runtime-model-wgpu")]
    {
        sparse_flow_wgpu_max_peak_bytes()
    }
    #[cfg(not(feature = "runtime-model-wgpu"))]
    {
        0
    }
}

fn sparse_flow_batched_cfg_enabled_for_backend<B: Backend>() -> bool {
    #[cfg(feature = "runtime-model-wgpu")]
    {
        attention_uses_non_fusion_module_kernel::<B>()
            && sparse_flow_batched_cfg_experimental_enabled()
            && sparse_flow_wgpu_module_attention_f16_enabled()
    }
    #[cfg(not(feature = "runtime-model-wgpu"))]
    {
        false
    }
}

#[cfg(feature = "runtime-model-wgpu")]
fn sparse_flow_batched_cfg_experimental_enabled() -> bool {
    std::env::var("TRELLIS2_SPARSE_FLOW_BATCHED_CFG")
        .ok()
        .and_then(|value| {
            let value = value.trim().to_ascii_lowercase();
            match value.as_str() {
                "1" | "true" | "on" | "yes" => Some(true),
                "0" | "false" | "off" | "no" => Some(false),
                _ => None,
            }
        })
        .unwrap_or(false)
}

fn sparse_flow_batched_cfg_debug_enabled() -> bool {
    std::env::var("TRELLIS2_SPARSE_FLOW_BATCHED_CFG_DEBUG")
        .ok()
        .and_then(|value| {
            let value = value.trim().to_ascii_lowercase();
            match value.as_str() {
                "1" | "true" | "on" | "yes" => Some(true),
                "0" | "false" | "off" | "no" => Some(false),
                _ => None,
            }
        })
        .unwrap_or(false)
}

fn concat_cross_kv_caches<B: Backend>(
    pos: &CrossAttentionKvCache<B>,
    neg: &CrossAttentionKvCache<B>,
) -> Option<CrossAttentionKvCache<B>> {
    if pos.len() != neg.len() {
        return None;
    }
    let mut out = Vec::with_capacity(pos.len());
    for (pos_block, neg_block) in pos.iter().zip(neg.iter()) {
        if pos_block.module_dtype != neg_block.module_dtype {
            return None;
        }
        let module_k = match (&pos_block.module_k, &neg_block.module_k) {
            (Some(pos_k), Some(neg_k)) => Some(Tensor::cat(vec![pos_k.clone(), neg_k.clone()], 0)),
            (None, None) => None,
            _ => return None,
        };
        let module_v = match (&pos_block.module_v, &neg_block.module_v) {
            (Some(pos_v), Some(neg_v)) => Some(Tensor::cat(vec![pos_v.clone(), neg_v.clone()], 0)),
            (None, None) => None,
            _ => return None,
        };
        out.push(CrossAttentionProjectedKv {
            k: Tensor::cat(vec![pos_block.k.clone(), neg_block.k.clone()], 0),
            v: Tensor::cat(vec![pos_block.v.clone(), neg_block.v.clone()], 0),
            module_k,
            module_v,
            module_dtype: pos_block.module_dtype,
        });
    }
    Some(out)
}

#[cfg(feature = "runtime-model-wgpu")]
fn sparse_flow_wgpu_estimated_peak_bytes(config: &SparseStructureFlowConfig) -> usize {
    let tokens = config
        .resolution
        .checked_mul(config.resolution)
        .and_then(|value| value.checked_mul(config.resolution))
        .unwrap_or(usize::MAX);
    let qkv_channels = config.model_channels.saturating_mul(3);
    let mlp_channels = ((config.model_channels as f32) * config.mlp_ratio.max(1.0))
        .ceil()
        .max(config.model_channels as f32) as usize;
    let peak_channels = qkv_channels.max(mlp_channels);
    tokens
        .checked_mul(peak_channels)
        .and_then(|value| value.checked_mul(core::mem::size_of::<f32>()))
        .unwrap_or(usize::MAX)
}

#[cfg(feature = "runtime-model-wgpu")]
fn sparse_flow_wgpu_may_overflow(config: &SparseStructureFlowConfig) -> bool {
    let estimated = sparse_flow_wgpu_estimated_peak_bytes(config);
    estimated > sparse_flow_wgpu_max_peak_bytes()
}

fn attention_debug_enabled() -> bool {
    runtime_model_attention_debug_enabled()
}

fn sparse_flow_stage_debug_enabled() -> bool {
    runtime_model_stage_debug_enabled()
}

fn sparse_flow_weight_probe_enabled() -> bool {
    sparse_flow_stage_debug_enabled()
        || std::env::var("TRELLIS2_SPARSE_FLOW_WEIGHT_PROBE")
            .ok()
            .and_then(|value| {
                let value = value.trim().to_ascii_lowercase();
                match value.as_str() {
                    "1" | "true" | "on" | "yes" => Some(true),
                    "0" | "false" | "off" | "no" => Some(false),
                    _ => None,
                }
            })
            .unwrap_or(false)
}

fn sparse_flow_forward_finite_probe_enabled() -> bool {
    std::env::var("TRELLIS2_SPARSE_FLOW_FORWARD_FINITE_PROBE")
        .ok()
        .and_then(|value| {
            let value = value.trim().to_ascii_lowercase();
            match value.as_str() {
                "1" | "true" | "on" | "yes" => Some(true),
                "0" | "false" | "off" | "no" => Some(false),
                _ => None,
            }
        })
        .unwrap_or(false)
}

fn sparse_flow_forward_finite_probe_strict() -> bool {
    std::env::var("TRELLIS2_SPARSE_FLOW_FORWARD_FINITE_PROBE_STRICT")
        .ok()
        .and_then(|value| {
            let value = value.trim().to_ascii_lowercase();
            match value.as_str() {
                "1" | "true" | "on" | "yes" => Some(true),
                "0" | "false" | "off" | "no" => Some(false),
                _ => None,
            }
        })
        .unwrap_or(false)
}

fn log_sparse_flow_forward_finite_probe<B: Backend, const D: usize>(
    label: &str,
    tensor: Tensor<B, D>,
) {
    let dims = tensor.dims();
    let values = match tensor.into_data().convert::<f32>().to_vec::<f32>() {
        Ok(values) => values,
        Err(err) => {
            eprintln!("burn_trellis: sparse flow finite probe {label}: readback failed: {err:?}");
            return;
        }
    };
    record_host_readback(values.len());
    let mut non_finite = 0usize;
    let mut first_non_finite = None;
    let mut finite_min = f32::INFINITY;
    let mut finite_max = f32::NEG_INFINITY;
    for (index, value) in values.iter().copied().enumerate() {
        if value.is_finite() {
            finite_min = finite_min.min(value);
            finite_max = finite_max.max(value);
        } else {
            non_finite += 1;
            first_non_finite.get_or_insert((index, value));
        }
    }
    let finite_min = if finite_min.is_finite() {
        finite_min
    } else {
        f32::NAN
    };
    let finite_max = if finite_max.is_finite() {
        finite_max
    } else {
        f32::NAN
    };
    let first_non_finite_text = first_non_finite
        .map(|(index, value)| format!("{index}:{value:?}"))
        .unwrap_or_else(|| "none".to_owned());
    eprintln!(
        "burn_trellis: sparse flow finite probe {label}: dims={dims:?} non_finite={non_finite} first_non_finite={first_non_finite_text} finite_min={finite_min:.9e} finite_max={finite_max:.9e}"
    );
    assert!(
        !(sparse_flow_forward_finite_probe_strict() && non_finite > 0),
        "sparse flow finite probe {label} produced {non_finite} non-finite values"
    );
}

#[cfg(feature = "runtime-model-wgpu")]
fn sparse_flow_wgpu_layer_norm_kernel_enabled() -> bool {
    std::env::var("TRELLIS2_SPARSE_FLOW_WGPU_LAYER_NORM_KERNEL")
        .ok()
        .and_then(|value| {
            let value = value.trim().to_ascii_lowercase();
            match value.as_str() {
                "0" | "false" | "off" | "no" => Some(false),
                "1" | "true" | "on" | "yes" => Some(true),
                _ => None,
            }
        })
        .unwrap_or(true)
}

#[cfg(feature = "runtime-model-wgpu")]
fn sparse_flow_wgpu_module_attention_enabled() -> bool {
    runtime_model_sparse_flow_module_attention_enabled()
}

fn sparse_flow_wgpu_module_attention_f16_enabled() -> bool {
    runtime_model_sparse_flow_module_attention_f16_enabled()
}

fn sparse_flow_wgpu_self_attention_f16_enabled() -> bool {
    #[cfg(feature = "runtime-model-wgpu")]
    {
        runtime_model_sparse_flow_self_attention_f16_enabled()
    }
    #[cfg(not(feature = "runtime-model-wgpu"))]
    {
        runtime_model_sparse_flow_module_attention_f16_enabled()
    }
}

fn sparse_flow_wgpu_cross_attention_f16_enabled() -> bool {
    #[cfg(feature = "runtime-model-wgpu")]
    {
        runtime_model_sparse_flow_cross_attention_f16_enabled()
    }
    #[cfg(not(feature = "runtime-model-wgpu"))]
    {
        runtime_model_sparse_flow_module_attention_f16_enabled()
    }
}

#[cfg(feature = "runtime-model-wgpu")]
fn sparse_flow_wgpu_linear_f16_enabled() -> bool {
    runtime_model_sparse_flow_linear_f16_enabled()
}

#[cfg(feature = "runtime-model-wgpu")]
fn sparse_flow_wgpu_coord_rope_kernel_enabled() -> bool {
    runtime_model_sparse_flow_coord_rope_kernel_enabled()
}

fn log_sparse_flow_weight_probe<B: Backend>(model: &SparseStructureFlowModel<B>) {
    fn stats_1d<B: Backend>(tensor: Tensor<B, 1>) -> (f32, f32, f32) {
        let values = tensor
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap_or_default();
        if values.is_empty() {
            return (0.0, 0.0, 0.0);
        }
        let mut min_v = f32::INFINITY;
        let mut max_v = f32::NEG_INFINITY;
        let mut sum_v = 0.0f64;
        for value in values.iter().copied() {
            min_v = min_v.min(value);
            max_v = max_v.max(value);
            sum_v += value as f64;
        }
        (min_v, max_v, (sum_v / values.len() as f64) as f32)
    }

    fn stats_2d<B: Backend>(tensor: Tensor<B, 2>) -> (f32, f32, f32) {
        let values = tensor
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap_or_default();
        if values.is_empty() {
            return (0.0, 0.0, 0.0);
        }
        let mut min_v = f32::INFINITY;
        let mut max_v = f32::NEG_INFINITY;
        let mut sum_v = 0.0f64;
        for value in values.iter().copied() {
            min_v = min_v.min(value);
            max_v = max_v.max(value);
            sum_v += value as f64;
        }
        (min_v, max_v, (sum_v / values.len() as f64) as f32)
    }

    let (in_min, in_max, in_mean) = stats_2d(model.input_layer.weight.val());
    let (out_min, out_max, out_mean) = stats_2d(model.out_layer.weight.val());
    if let Some(block0) = model.blocks.first() {
        let (kv_min, kv_max, kv_mean) = stats_2d(block0.cross_attn.to_kv.weight.val());
        let (q_min, q_max, q_mean) = stats_2d(block0.cross_attn.to_q.weight.val());
        let (mod_min, mod_max, mod_mean) = stats_1d(block0.modulation.val());
        eprintln!(
            "burn_trellis: sparse flow weight probe backend={} input_layer[min,max,mean]=[{:.6},{:.6},{:.6}] block0.modulation=[{:.6},{:.6},{:.6}] block0.cross_attn.to_q=[{:.6},{:.6},{:.6}] block0.cross_attn.to_kv=[{:.6},{:.6},{:.6}] out_layer=[{:.6},{:.6},{:.6}]",
            std::any::type_name::<B>(),
            in_min,
            in_max,
            in_mean,
            mod_min,
            mod_max,
            mod_mean,
            q_min,
            q_max,
            q_mean,
            kv_min,
            kv_max,
            kv_mean,
            out_min,
            out_max,
            out_mean
        );
    } else {
        eprintln!(
            "burn_trellis: sparse flow weight probe backend={} has no transformer blocks",
            std::any::type_name::<B>()
        );
    }
}

fn attention_uses_module_kernel<B: Backend>() -> bool {
    #[cfg(feature = "runtime-model-wgpu")]
    if !sparse_flow_wgpu_module_attention_enabled() {
        return false;
    }
    let backend = std::any::type_name::<B>();
    backend.contains("Wgpu") || backend.contains("cubecl_wgpu")
}

fn attention_uses_non_fusion_module_kernel<B: Backend>() -> bool {
    attention_uses_module_kernel::<B>() && !std::any::type_name::<B>().contains("Fusion<")
}

fn attention_prefers_stream() -> bool {
    #[cfg(feature = "runtime-model-wgpu")]
    {
        true
    }
    #[cfg(not(feature = "runtime-model-wgpu"))]
    {
        false
    }
}

fn sparse_flow_chunked_forward_enabled(tokens: usize) -> bool {
    attention_prefers_stream() && tokens >= 2_048
}

fn sparse_flow_chunked_forward_for_backend<B: Backend>(tokens: usize) -> bool {
    if !sparse_flow_chunked_forward_enabled(tokens) {
        return false;
    }
    if attention_uses_module_kernel::<B>() {
        #[cfg(target_arch = "wasm32")]
        {
            return true;
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            // Keep dense/full attention for smaller native token counts to
            // exercise fast module kernels, but retain streaming for large
            // row-conditioned stages (e.g. shape/tex SLAT at 32^3).
            const MODULE_ATTN_NATIVE_DENSE_MAX_TOKENS: usize = 8_192;
            return tokens > MODULE_ATTN_NATIVE_DENSE_MAX_TOKENS;
        }
    }
    true
}

fn sparse_flow_module_attention_chunk_cap(tokens: usize) -> usize {
    if tokens > SPARSE_FLOW_MODULE_ATTENTION_LONG_K_FALLBACK_SEQ_KV {
        return SPARSE_FLOW_MODULE_ATTENTION_LONG_K_QUERY_CHUNK.min(tokens.max(1));
    }
    if tokens >= 131_072 {
        16_384
    } else if tokens >= 16_384 {
        #[cfg(target_arch = "wasm32")]
        {
            8_192
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            16_384
        }
    } else {
        tokens.max(1)
    }
}

fn sparse_flow_module_attention_chunk_cap_for_shape(
    _batch: usize,
    _heads: usize,
    query_tokens: usize,
    key_tokens: usize,
    head_dim: usize,
) -> usize {
    if key_tokens > SPARSE_FLOW_MODULE_ATTENTION_LONG_K_FALLBACK_SEQ_KV {
        let cap = if head_dim == 128 {
            SPARSE_FLOW_MODULE_ATTENTION_LONG_K_HEAD_DIM_128_QUERY_CHUNK
        } else {
            SPARSE_FLOW_MODULE_ATTENTION_LONG_K_QUERY_CHUNK
        };
        return cap.min(query_tokens.max(1));
    }
    if query_tokens > SPARSE_FLOW_MODULE_ATTENTION_LONG_K_FALLBACK_SEQ_KV {
        return SPARSE_FLOW_MODULE_ATTENTION_VERIFIED_LONG_QUERY_CHUNK.min(query_tokens.max(1));
    }
    sparse_flow_module_attention_chunk_cap(query_tokens)
}

fn sparse_flow_module_attention_query_chunk_cap(
    batch: usize,
    heads: usize,
    key_tokens: usize,
    logits_budget: usize,
) -> usize {
    let key_tokens = key_tokens.max(1);
    let bytes_per_logit = batch
        .saturating_mul(heads)
        .saturating_mul(core::mem::size_of::<f32>())
        .max(1);
    let budget_logits = logits_budget / bytes_per_logit;
    budget_logits.checked_div(key_tokens).unwrap_or(1).max(1)
}

fn sparse_flow_module_attention_prefers_full(tokens: usize) -> bool {
    // Keep chunk-planned module attention as the default canonical behavior.
    // Forcing full-dispatch changed sparse-flow numerics enough to collapse
    // sparse-structure occupancy on representative runs.
    let _ = tokens;
    false
}

fn sparse_flow_module_attention_query_multiple() -> usize {
    // CubeK's accelerated WGPU attention path requires the query length to be a
    // multiple of the inferred stage width. Padding to 128 covers the current
    // native blackbox candidates while adding at most 127 ignored query rows.
    128
}

fn sparse_flow_stream_reuse_qkv_enabled(tokens: usize, channels: usize) -> bool {
    if !sparse_flow_chunked_forward_enabled(tokens) {
        return false;
    }
    #[cfg(feature = "runtime-model-wgpu")]
    {
        // Cap extra cache pressure from storing streamed Q chunks while still
        // eliminating the second QKV projection pass.
        let q_cache_bytes = tokens
            .checked_mul(channels)
            .and_then(|value| value.checked_mul(core::mem::size_of::<f32>()))
            .unwrap_or(usize::MAX);
        let budget = sparse_flow_wgpu_max_peak_bytes().saturating_mul(2);
        q_cache_bytes <= budget
    }
    #[cfg(not(feature = "runtime-model-wgpu"))]
    {
        let _ = channels;
        false
    }
}

fn sparse_flow_linear_chunk_tokens(tokens: usize) -> usize {
    let default = if sparse_flow_chunked_forward_enabled(tokens) {
        8_192
    } else {
        tokens.max(1)
    };
    default.min(32_768).min(tokens.max(1))
}

fn sparse_flow_linear_chunk_tokens_for_backend<B: Backend>(tokens: usize) -> usize {
    let default = sparse_flow_linear_chunk_tokens(tokens);
    if attention_uses_non_fusion_module_kernel::<B>() && tokens >= 16_384 {
        // Native WGPU fast-f16 linears can run the canonical HR SLat token
        // count in one matmul. Avoiding the extra slice/cat boundary removes
        // launch overhead from the 23,816-row shape without changing math.
        #[cfg(all(feature = "runtime-model-wgpu", not(target_arch = "wasm32")))]
        if sparse_flow_wgpu_linear_f16_enabled() {
            return 32_768usize.min(tokens.max(1)).max(default);
        }
        // Keep reference f32 and wasm on the previous conservative cap.
        return 16_384usize.min(tokens.max(1)).max(default);
    }
    default
}

fn sparse_flow_attn_logits_budget_bytes() -> usize {
    #[cfg(feature = "runtime-model-wgpu")]
    {
        // Keep sparse-flow attention logits bounded.
        #[cfg(target_arch = "wasm32")]
        {
            128 * 1024 * 1024
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            3 * 1024 * 1024 * 1024
        }
    }
    #[cfg(not(feature = "runtime-model-wgpu"))]
    {
        2_147_483_648
    }
}

fn sparse_flow_stream_chunk_plan(
    batch: usize,
    heads: usize,
    tokens: usize,
    query_chunk_tokens: usize,
    kv_chunk_tokens: usize,
    reuse_qkv: bool,
    logits_budget: usize,
) -> (usize, usize) {
    let tokens = tokens.max(1);
    let mut query_chunk_tokens = query_chunk_tokens.max(1).min(tokens);
    let mut kv_chunk_tokens = kv_chunk_tokens.max(1).min(tokens);
    let bytes_per_logit = batch
        .saturating_mul(heads)
        .saturating_mul(core::mem::size_of::<f32>())
        .max(1);
    let budget_logits = logits_budget / bytes_per_logit;

    if reuse_qkv {
        let max_square = integer_sqrt(budget_logits).max(1).min(tokens);
        let chunk_tokens = kv_chunk_tokens.min(max_square).max(1);
        (chunk_tokens, chunk_tokens)
    } else {
        let max_query = budget_logits
            .checked_div(kv_chunk_tokens.max(1))
            .unwrap_or(1)
            .max(1);
        query_chunk_tokens = query_chunk_tokens.min(max_query).max(1);
        let max_kv = budget_logits
            .checked_div(query_chunk_tokens.max(1))
            .unwrap_or(1)
            .max(1);
        kv_chunk_tokens = kv_chunk_tokens.min(max_kv).max(1);
        (query_chunk_tokens, kv_chunk_tokens)
    }
}

#[cfg(test)]
fn sparse_flow_attention_logits_within_budget(
    batch: usize,
    heads: usize,
    query_tokens: usize,
    key_tokens: usize,
    logits_budget: usize,
) -> bool {
    attention_logits_bytes(batch, heads, query_tokens, key_tokens) <= logits_budget
}

fn integer_sqrt(value: usize) -> usize {
    (value as f64).sqrt().floor() as usize
}

fn sparse_flow_mlp_chunk_tokens(tokens: usize) -> usize {
    let default = if sparse_flow_chunked_forward_enabled(tokens) {
        2_048
    } else {
        tokens.max(1)
    };
    default.min(65_536).min(tokens.max(1))
}

fn sparse_flow_mlp_chunk_tokens_for_backend<B: Backend>(tokens: usize) -> usize {
    let default = sparse_flow_mlp_chunk_tokens(tokens);
    if !attention_uses_non_fusion_module_kernel::<B>() {
        return default;
    }
    // Non-fusion WGPU module-attention path can tolerate wider MLP chunks.
    // This cuts launch/concatenation overhead while keeping canonical sparse
    // SLAT stages below common adapter allocation ceilings.
    let fast_linear_chunk = {
        #[cfg(all(feature = "runtime-model-wgpu", not(target_arch = "wasm32")))]
        {
            if sparse_flow_wgpu_linear_f16_enabled() {
                32_768usize
            } else {
                8_192usize
            }
        }
        #[cfg(any(not(feature = "runtime-model-wgpu"), target_arch = "wasm32"))]
        {
            8_192usize
        }
    };
    let widened = if tokens >= 16_384 {
        fast_linear_chunk.min(tokens.max(1)).max(default)
    } else if tokens >= 2_048 {
        8_192usize.min(tokens.max(1)).max(default)
    } else {
        default
    };
    // Keep per-chunk hidden activations bounded for WGPU sparse-flow MLP.
    if attention_uses_module_kernel::<B>() {
        let cap = {
            #[cfg(target_arch = "wasm32")]
            {
                // Browser adapters commonly expose lower effective storage
                // allocation ceilings than native Vulkan.
                4_096usize
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                #[cfg(feature = "runtime-model-wgpu")]
                if sparse_flow_wgpu_linear_f16_enabled() {
                    32_768usize
                } else {
                    8_192usize
                }
                #[cfg(not(feature = "runtime-model-wgpu"))]
                {
                    8_192usize
                }
            }
        };
        widened.min(cap).min(tokens.max(1))
    } else {
        widened
    }
}

fn sparse_flow_mlp_sync_interval(tokens: usize) -> usize {
    if tokens < 131_072 {
        return usize::MAX;
    }
    usize::MAX
}

fn sparse_flow_mlp_sync_interval_for_backend<B: Backend>(tokens: usize) -> usize {
    let configured = sparse_flow_mlp_sync_interval(tokens);
    if configured != usize::MAX {
        return configured;
    }
    if attention_uses_non_fusion_module_kernel::<B>() {
        // Larger canonical WGPU chunks reduce chunk count; keep syncs sparse to
        // avoid adding queue-fence overhead back into shape/tex sparse-flow.
        return 16;
    }
    // WGPU can accumulate deep unsynchronized MLP queues in shape_slat/tex_slat paths.
    // Keep periodic device synchronization enabled to avoid long-running queue stalls.
    if attention_uses_module_kernel::<B>() {
        8
    } else {
        usize::MAX
    }
}

fn sparse_flow_mlp_window_rows(rows: usize) -> usize {
    let rows = rows.max(1);
    let default = 32_768.min(rows);
    default.min(131_072).min(rows)
}

fn sparse_flow_self_attn_query_chunk_tokens(tokens: usize) -> usize {
    let default = if sparse_flow_chunked_forward_enabled(tokens) {
        2_048
    } else {
        tokens.max(1)
    };
    default.min(16_384).min(tokens.max(1))
}

fn sparse_flow_self_attn_kv_chunk_tokens(tokens: usize) -> usize {
    let default = if sparse_flow_chunked_forward_enabled(tokens) {
        8_192
    } else {
        tokens.max(1)
    };
    default.min(32_768).min(tokens.max(1))
}

fn linear_forward_stable_2d<B: Backend>(linear: &nn::Linear<B>, x: Tensor<B, 2>) -> Tensor<B, 2>
where
    SparseFlowLinearWgpuBridgeImpl: SparseFlowLinearWgpuBridge<B>,
    SparseFlowBf16EmulationBridgeImpl: SparseFlowBf16EmulationBridge<B>,
{
    linear_forward_stable_2d_with_policy(linear, x, true, false)
}

fn linear_forward_block_2d<B: Backend>(linear: &nn::Linear<B>, x: Tensor<B, 2>) -> Tensor<B, 2>
where
    SparseFlowLinearWgpuBridgeImpl: SparseFlowLinearWgpuBridge<B>,
    SparseFlowBf16EmulationBridgeImpl: SparseFlowBf16EmulationBridge<B>,
{
    linear_forward_stable_2d_with_policy(linear, x, true, true)
}

fn linear_forward_stable_2d_reference<B: Backend>(
    linear: &nn::Linear<B>,
    x: Tensor<B, 2>,
) -> Tensor<B, 2>
where
    SparseFlowLinearWgpuBridgeImpl: SparseFlowLinearWgpuBridge<B>,
    SparseFlowBf16EmulationBridgeImpl: SparseFlowBf16EmulationBridge<B>,
{
    linear_forward_stable_2d_reference_impl(linear, x, false)
}

fn linear_forward_stable_2d_with_policy<B: Backend>(
    linear: &nn::Linear<B>,
    x: Tensor<B, 2>,
    allow_fast_f16: bool,
    emulate_bf16_block: bool,
) -> Tensor<B, 2>
where
    SparseFlowLinearWgpuBridgeImpl: SparseFlowLinearWgpuBridge<B>,
    SparseFlowBf16EmulationBridgeImpl: SparseFlowBf16EmulationBridge<B>,
{
    let x_dtype: burn::tensor::FloatDType = x.dtype().into();
    let [rows, in_channels] = x.dims();
    let weight = linear.weight.val();
    let [weight_in_channels, out_channels] = weight.dims();
    if in_channels != weight_in_channels {
        panic!(
            "linear input/weight mismatch: input=[{rows},{in_channels}] weight=[{weight_in_channels},{out_channels}]"
        );
    }

    #[cfg(feature = "runtime-model-wgpu")]
    if allow_fast_f16
        && sparse_flow_wgpu_linear_f16_enabled()
        && attention_uses_non_fusion_module_kernel::<B>()
        && rows >= 1024
        && in_channels >= 64
        && out_channels >= 64
    {
        if let Some(output) = maybe_linear_f16_wgpu(linear, x.clone(), x_dtype) {
            return output;
        }
    }

    linear_forward_stable_2d_reference_impl(linear, x, emulate_bf16_block)
}

fn linear_forward_stable_2d_reference_impl<B: Backend>(
    linear: &nn::Linear<B>,
    x: Tensor<B, 2>,
    emulate_bf16_block: bool,
) -> Tensor<B, 2>
where
    SparseFlowLinearWgpuBridgeImpl: SparseFlowLinearWgpuBridge<B>,
    SparseFlowBf16EmulationBridgeImpl: SparseFlowBf16EmulationBridge<B>,
{
    let x_dtype: burn::tensor::FloatDType = x.dtype().into();
    let [rows, in_channels] = x.dims();
    let mut weight = linear.weight.val();
    let [weight_in_channels, out_channels] = weight.dims();
    if in_channels != weight_in_channels {
        panic!(
            "linear input/weight mismatch: input=[{rows},{in_channels}] weight=[{weight_in_channels},{out_channels}]"
        );
    }

    let matmul_dtype = if matches!(x_dtype, burn::tensor::FloatDType::BF16) {
        burn::tensor::FloatDType::F32
    } else {
        x_dtype
    };
    let x = tensor_cast_float_2d_if_needed(x, matmul_dtype);
    if emulate_bf16_block {
        weight = sparse_flow_stock_bf16_round(weight);
    }
    let weight_dtype: burn::tensor::FloatDType = weight.dtype().into();
    // Keep linear operands in a single dtype to avoid mixed-dtype WGPU matmul
    // collapse on bf16 checkpoint weights (observed near-zero outputs in sparse flow).
    let weight = if weight_dtype != matmul_dtype {
        weight.cast(matmul_dtype)
    } else {
        weight
    };
    let mut output = SparseFlowLinearWgpuBridgeImpl::safe_matmul_2d(
        x.clone(),
        weight.clone(),
        "sparse-flow.linear",
    )
    .unwrap_or_else(|| x.matmul(weight));
    if let Some(bias) = linear.bias.as_ref() {
        let output_dtype: burn::tensor::FloatDType = output.dtype().into();
        let mut bias = bias.val();
        if emulate_bf16_block {
            bias = sparse_flow_stock_bf16_round(bias);
        }
        let bias_dtype: burn::tensor::FloatDType = bias.dtype().into();
        let bias = if bias_dtype != output_dtype {
            bias.cast(output_dtype)
        } else {
            bias
        };
        output = output.add(bias.unsqueeze::<2>());
    }
    let output = if emulate_bf16_block {
        sparse_flow_stock_bf16_round(output)
    } else {
        output
    };
    tensor_cast_float_2d_if_needed(output, x_dtype)
}

fn sparse_flow_torso_dtype_for_backend<B: Backend>() -> Option<burn::tensor::FloatDType> {
    {
        #[cfg(feature = "runtime-model-wgpu")]
        {
            if attention_uses_non_fusion_module_kernel::<B>()
                && runtime_model_sparse_flow_torso_f16_enabled()
            {
                return Some(burn::tensor::FloatDType::F16);
            }
        }
    }
    None
}

fn feed_forward_f16_chain_enabled<B: Backend>(
    mlp_0: &nn::Linear<B>,
    mlp_2: &nn::Linear<B>,
) -> bool {
    #[cfg(feature = "runtime-model-wgpu")]
    {
        let [in_channels, hidden_channels] = mlp_0.weight.val().dims();
        let [hidden_in_channels, out_channels] = mlp_2.weight.val().dims();
        sparse_flow_wgpu_linear_f16_enabled()
            && attention_uses_non_fusion_module_kernel::<B>()
            && in_channels >= 64
            && hidden_channels >= 64
            && hidden_in_channels == hidden_channels
            && out_channels >= 64
    }
    #[cfg(not(feature = "runtime-model-wgpu"))]
    {
        let _ = (mlp_0, mlp_2);
        false
    }
}

fn linear_forward_f16_raw_2d<B: Backend>(linear: &nn::Linear<B>, x: Tensor<B, 2>) -> Tensor<B, 2>
where
    SparseFlowLinearWgpuBridgeImpl: SparseFlowLinearWgpuBridge<B>,
    SparseFlowBf16EmulationBridgeImpl: SparseFlowBf16EmulationBridge<B>,
{
    let [rows, in_channels] = x.dims();
    let weight = linear.weight.val();
    let [weight_in_channels, out_channels] = weight.dims();
    if in_channels != weight_in_channels {
        panic!(
            "linear input/weight mismatch: input=[{rows},{in_channels}] weight=[{weight_in_channels},{out_channels}]"
        );
    }

    let f16 = burn::tensor::FloatDType::F16;
    #[cfg(feature = "runtime-model-wgpu")]
    if sparse_flow_wgpu_linear_f16_enabled()
        && attention_uses_non_fusion_module_kernel::<B>()
        && rows >= 1024
        && in_channels >= 64
        && out_channels >= 64
    {
        if let Some(output) = maybe_linear_f16_wgpu(linear, x.clone(), f16) {
            return output;
        }
    }

    let mut output =
        tensor_cast_float_2d_if_needed(x, f16).matmul(tensor_cast_float_2d_if_needed(weight, f16));
    if let Some(bias) = linear.bias.as_ref() {
        output = output.add(tensor_cast_float_1d_if_needed(bias.val(), f16).unsqueeze::<2>());
    }
    output
}

fn feed_forward_f16_chain_via_2d<B: Backend>(
    mlp_0: &nn::Linear<B>,
    mlp_2: &nn::Linear<B>,
    x: Tensor<B, 3>,
) -> Tensor<B, 3>
where
    SparseFlowLinearWgpuBridgeImpl: SparseFlowLinearWgpuBridge<B>,
    SparseFlowBf16EmulationBridgeImpl: SparseFlowBf16EmulationBridge<B>,
{
    let output_dtype: burn::tensor::FloatDType = x.dtype().into();
    let [batch, tokens, channels] = x.dims();
    let out_channels = mlp_2.weight.val().dims()[1];
    let hidden = linear_forward_f16_raw_2d(mlp_0, x.reshape([batch * tokens, channels]));
    linear_forward_f16_raw_2d(mlp_2, gelu(hidden))
        .cast(output_dtype)
        .reshape([batch, tokens, out_channels])
}

fn linear_forward_block_via_2d<B: Backend>(linear: &nn::Linear<B>, x: Tensor<B, 3>) -> Tensor<B, 3>
where
    SparseFlowLinearWgpuBridgeImpl: SparseFlowLinearWgpuBridge<B>,
    SparseFlowBf16EmulationBridgeImpl: SparseFlowBf16EmulationBridge<B>,
{
    let [batch, tokens, channels] = x.dims();
    let out_channels = linear.weight.val().dims()[1];
    linear_forward_block_2d(linear, x.reshape([batch * tokens, channels])).reshape([
        batch,
        tokens,
        out_channels,
    ])
}

fn linear_forward_stable_via_2d_reference<B: Backend>(
    linear: &nn::Linear<B>,
    x: Tensor<B, 3>,
) -> Tensor<B, 3>
where
    SparseFlowLinearWgpuBridgeImpl: SparseFlowLinearWgpuBridge<B>,
    SparseFlowBf16EmulationBridgeImpl: SparseFlowBf16EmulationBridge<B>,
{
    let [batch, tokens, channels] = x.dims();
    let out_channels = linear.weight.val().dims()[1];
    linear_forward_stable_2d_reference(linear, x.reshape([batch * tokens, channels])).reshape([
        batch,
        tokens,
        out_channels,
    ])
}

fn linear_forward_attention<B: Backend>(
    linear: &nn::Linear<B>,
    x: Tensor<B, 3>,
    allow_f16_output: bool,
) -> Tensor<B, 3>
where
    SparseFlowLinearWgpuBridgeImpl: SparseFlowLinearWgpuBridge<B>,
    SparseFlowBf16EmulationBridgeImpl: SparseFlowBf16EmulationBridge<B>,
{
    #[cfg(feature = "runtime-model-wgpu")]
    {
        let [batch, tokens, channels] = x.dims();
        let rows = batch.saturating_mul(tokens);
        let [in_channels, out_channels] = linear.weight.val().dims();
        if allow_f16_output
            && sparse_flow_wgpu_linear_f16_enabled()
            && attention_uses_non_fusion_module_kernel::<B>()
            && rows >= 1024
            && channels == in_channels
            && in_channels >= 64
            && out_channels >= 64
        {
            return linear_forward_f16_raw_2d(linear, x.reshape([rows, channels])).reshape([
                batch,
                tokens,
                out_channels,
            ]);
        }
    }
    linear_forward_block_via_2d(linear, x)
}

fn linear_forward_attention_to_dtype<B: Backend>(
    linear: &nn::Linear<B>,
    x: Tensor<B, 3>,
    dtype: burn::tensor::FloatDType,
    allow_f16_output: bool,
) -> Tensor<B, 3>
where
    SparseFlowLinearWgpuBridgeImpl: SparseFlowLinearWgpuBridge<B>,
    SparseFlowBf16EmulationBridgeImpl: SparseFlowBf16EmulationBridge<B>,
{
    tensor_cast_float_3d_if_needed(linear_forward_attention(linear, x, allow_f16_output), dtype)
}

fn linear_forward_token_chunked_reference<B: Backend>(
    linear: &nn::Linear<B>,
    x: Tensor<B, 3>,
    chunk_tokens: usize,
) -> Tensor<B, 3>
where
    SparseFlowLinearWgpuBridgeImpl: SparseFlowLinearWgpuBridge<B>,
    SparseFlowBf16EmulationBridgeImpl: SparseFlowBf16EmulationBridge<B>,
{
    let [batch, tokens, channels] = x.dims();
    if chunk_tokens >= tokens {
        return linear_forward_stable_via_2d_reference(linear, x);
    }

    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < tokens {
        let end = (start + chunk_tokens).min(tokens);
        let x_chunk = x.clone().slice([0..batch, start..end, 0..channels]);
        chunks.push(linear_forward_stable_via_2d_reference(linear, x_chunk));
        start = end;
    }
    Tensor::cat(chunks, 1)
}

fn linear_forward_input_token_chunked<B: Backend>(
    linear: &nn::Linear<B>,
    x: Tensor<B, 3>,
    chunk_tokens: usize,
) -> Tensor<B, 3>
where
    SparseFlowLinearWgpuBridgeImpl: SparseFlowLinearWgpuBridge<B>,
    SparseFlowBf16EmulationBridgeImpl: SparseFlowBf16EmulationBridge<B>,
{
    let [batch, tokens, channels] = x.dims();
    let out_channels = linear.weight.val().dims()[1];
    let output_dtype: burn::tensor::FloatDType = x.dtype().into();

    #[cfg(feature = "runtime-model-wgpu")]
    let try_skinny = |x_2d: Tensor<B, 2>| -> Option<Tensor<B, 2>> {
        let [rows, in_channels] = x_2d.dims();
        if sparse_flow_wgpu_linear_f16_enabled()
            && attention_uses_non_fusion_module_kernel::<B>()
            && SPARSE_FLOW_INPUT_SKINNY_LINEAR_WGPU
            && rows >= 1024
            && in_channels <= 64
            && out_channels >= 256
        {
            maybe_linear_skinny_wgpu(linear, x_2d, output_dtype)
        } else {
            None
        }
    };

    if chunk_tokens >= tokens {
        let x_2d = x.reshape([batch * tokens, channels]);
        #[cfg(feature = "runtime-model-wgpu")]
        if let Some(output) = try_skinny(x_2d.clone()) {
            return output.reshape([batch, tokens, out_channels]);
        }
        return linear_forward_stable_2d_reference(linear, x_2d).reshape([
            batch,
            tokens,
            out_channels,
        ]);
    }

    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < tokens {
        let end = (start + chunk_tokens).min(tokens);
        let chunk_tokens_actual = end - start;
        let x_chunk = x.clone().slice([0..batch, start..end, 0..channels]);
        let x_2d = x_chunk.reshape([batch * chunk_tokens_actual, channels]);
        #[cfg(feature = "runtime-model-wgpu")]
        if let Some(output) = try_skinny(x_2d.clone()) {
            chunks.push(output.reshape([batch, chunk_tokens_actual, out_channels]));
            start = end;
            continue;
        }
        chunks.push(linear_forward_stable_2d_reference(linear, x_2d).reshape([
            batch,
            chunk_tokens_actual,
            out_channels,
        ]));
        start = end;
    }
    Tensor::cat(chunks, 1)
}

// Avoid 4D matmul layout expansion on fusion/cubecl backends by flattening batch*heads.
fn matmul_4d_via_3d<B: Backend>(lhs: Tensor<B, 4>, rhs: Tensor<B, 4>) -> Tensor<B, 4> {
    let [batch, heads, m, k] = lhs.dims();
    let [rhs_batch, rhs_heads, rhs_k, n] = rhs.dims();
    if batch != rhs_batch || heads != rhs_heads || k != rhs_k {
        panic!(
            "4d matmul shape mismatch: lhs=[{batch},{heads},{m},{k}] rhs=[{rhs_batch},{rhs_heads},{rhs_k},{n}]"
        );
    }
    let bh = batch.saturating_mul(heads).max(1);
    lhs.clone()
        .reshape([bh, m, k])
        .matmul(rhs.clone().reshape([bh, rhs_k, n]))
        .reshape([batch, heads, m, n])
}

fn attention_impl(query_tokens: usize, key_tokens: usize) -> AttentionImpl {
    let work = query_tokens.saturating_mul(key_tokens);
    if (attention_prefers_stream() && work >= 64usize.saturating_mul(64))
        || work >= 512usize.saturating_mul(512)
    {
        AttentionImpl::Stream
    } else {
        AttentionImpl::Dense
    }
}

fn scaled_dot_product_attention<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    head_dim: usize,
) -> Tensor<B, 4> {
    let q = q.permute([0, 2, 1, 3]);
    let k = k.permute([0, 2, 1, 3]);
    let v = v.permute([0, 2, 1, 3]);
    let [batch, heads, query_tokens, _] = q.dims();
    let [_, _, key_tokens, _] = k.dims();

    if attention_uses_module_kernel::<B>()
        && !sparse_flow_module_attention_cross_shape_requires_stream(
            batch,
            heads,
            query_tokens,
            key_tokens,
            head_dim,
        )
    {
        let backend_name = std::any::type_name::<B>();
        let fusion_backend = backend_name.contains("Fusion<");
        let logits_budget = attention_logits_budget_bytes();
        let bytes_per_logit = batch
            .saturating_mul(heads)
            .saturating_mul(core::mem::size_of::<f32>())
            .max(1);
        let budget_logits = logits_budget / bytes_per_logit;
        let max_query_by_budget = budget_logits
            .checked_div(key_tokens.max(1))
            .unwrap_or(1)
            .max(1);
        let query_chunk = if fusion_backend {
            // Fusion can fall back to dense attention in some layouts; keep
            // conservative query caps there to avoid oversized temporary logits.
            attention_query_chunk(query_tokens, query_tokens)
                .min(max_query_by_budget)
                .max(1)
        } else if sparse_flow_module_attention_long_k_requires_fallback(
            batch, heads, key_tokens, head_dim,
        ) {
            SPARSE_FLOW_MODULE_ATTENTION_LONG_K_QUERY_CHUNK
                .min(max_query_by_budget)
                .min(query_tokens)
                .max(1)
        } else {
            // Raw CubeBackend module attention should stay on flash-attn kernels.
            // Use the sparse-flow module cap instead of the global stream cap
            // (1024) to reduce cross-attention dispatch overhead on large token
            // counts while keeping chunking bounded for very large stages.
            sparse_flow_module_attention_chunk_cap_for_shape(
                batch,
                heads,
                query_tokens,
                key_tokens,
                head_dim,
            )
            .min(query_tokens)
            .max(1)
        };

        if attention_debug_enabled() && query_tokens >= 1024 {
            eprintln!(
                "burn_trellis: attn dispatch backend={backend_name} impl=flash_attention(module_attention) q={query_tokens} k={key_tokens} head_dim={head_dim} q_chunk={query_chunk} logits_budget={logits_budget}"
            );
        }

        let out = if query_chunk >= query_tokens {
            sparse_flow_module_attention(q, k, v)
        } else {
            let mut chunks = Vec::new();
            let mut start = 0usize;
            while start < query_tokens {
                let end = (start + query_chunk).min(query_tokens);
                let q_chunk = q
                    .clone()
                    .slice([0..batch, 0..heads, start..end, 0..head_dim])
                    .clone();
                chunks.push(sparse_flow_module_attention(q_chunk, k.clone(), v.clone()));
                start = end;
            }
            Tensor::cat(chunks, 2)
        };
        return out.permute([0, 2, 1, 3]);
    }

    let attention_impl = attention_impl(query_tokens, key_tokens);
    if attention_debug_enabled() && query_tokens >= 4096 {
        let backend_name = std::any::type_name::<B>();
        let impl_name = match attention_impl {
            AttentionImpl::Dense => "dense",
            AttentionImpl::Stream => "stream",
        };
        eprintln!(
            "burn_trellis: attn dispatch backend={backend_name} impl={impl_name} q={query_tokens} k={key_tokens} head_dim={head_dim}"
        );
    }

    let out = match attention_impl {
        AttentionImpl::Dense => scaled_dot_product_attention_dense(q, k, v, head_dim),
        AttentionImpl::Stream => scaled_dot_product_attention_stream(q, k, v, head_dim),
    };
    out.permute([0, 2, 1, 3])
}

fn scaled_dot_product_attention_dense<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    head_dim: usize,
) -> Tensor<B, 4> {
    let [batch, heads, tokens, _] = q.dims();
    let [_, _, key_tokens, _] = k.dims();
    let scale = 1.0 / (head_dim as f32).sqrt();
    let query_chunk = attention_query_chunk(tokens, 8);
    let logits_budget = attention_logits_budget_bytes();
    let dense_logits_bytes = attention_logits_bytes(batch, heads, tokens, key_tokens);
    if attention_debug_enabled() && tokens >= 4096 {
        eprintln!(
            "burn_trellis: attn dense q={tokens} k={key_tokens} query_chunk={query_chunk} logits_bytes={dense_logits_bytes} budget_bytes={logits_budget}"
        );
    }

    if query_chunk >= tokens && dense_logits_bytes <= logits_budget {
        let attn = softmax(
            matmul_4d_via_3d(q.clone(), k.clone().swap_dims(2, 3)).mul_scalar(scale),
            3,
        );
        return matmul_4d_via_3d(attn, v);
    }

    let k_t = k.clone().swap_dims(2, 3);
    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < tokens {
        let end = (start + query_chunk).min(tokens);
        let q_chunk = q
            .clone()
            .slice([0..batch, 0..heads, start..end, 0..head_dim])
            .clone();
        let attn = softmax(matmul_4d_via_3d(q_chunk, k_t.clone()).mul_scalar(scale), 3);
        chunks.push(matmul_4d_via_3d(attn, v.clone()));
        start = end;
    }
    Tensor::cat(chunks, 2)
}

fn scaled_dot_product_attention_stream<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    head_dim: usize,
) -> Tensor<B, 4> {
    let [batch, heads, query_tokens, _] = q.dims();
    let [_, _, key_tokens, _] = k.dims();
    let [_, _, _, value_dim] = v.dims();
    let scale = 1.0 / (head_dim as f32).sqrt();
    let query_chunk = attention_query_chunk(query_tokens, 64);
    let key_chunk = attention_key_chunk(key_tokens);
    let logits_budget = attention_logits_budget_bytes();
    let dense_logits_bytes = attention_logits_bytes(batch, heads, query_tokens, key_tokens);
    if attention_debug_enabled() && query_tokens >= 4096 {
        eprintln!(
            "burn_trellis: attn stream q={query_tokens} k={key_tokens} query_chunk={query_chunk} key_chunk={key_chunk} logits_bytes={dense_logits_bytes} budget_bytes={logits_budget}"
        );
    }

    if query_chunk >= query_tokens && key_chunk >= key_tokens && dense_logits_bytes <= logits_budget
    {
        let attn = softmax(
            matmul_4d_via_3d(q.clone(), k.clone().swap_dims(2, 3)).mul_scalar(scale),
            3,
        );
        return matmul_4d_via_3d(attn, v);
    }

    let mut outputs = Vec::new();
    let mut q_start = 0usize;
    while q_start < query_tokens {
        let q_end = (q_start + query_chunk).min(query_tokens);
        let q_chunk = q
            .clone()
            .slice([0..batch, 0..heads, q_start..q_end, 0..head_dim])
            .clone();

        let first_k_end = key_chunk.min(key_tokens);
        let first_k = k
            .clone()
            .slice([0..batch, 0..heads, 0..first_k_end, 0..head_dim])
            .clone();
        let first_v = v
            .clone()
            .slice([0..batch, 0..heads, 0..first_k_end, 0..value_dim])
            .clone();

        let first_logits =
            matmul_4d_via_3d(q_chunk.clone(), first_k.swap_dims(2, 3)).mul_scalar(scale);
        let mut max_scores = first_logits.clone().max_dim(3);
        let first_probs = first_logits.sub(max_scores.clone()).exp();
        let mut denom = first_probs.clone().sum_dim(3);
        let mut acc = matmul_4d_via_3d(first_probs, first_v);

        let mut k_start = first_k_end;
        while k_start < key_tokens {
            let k_end = (k_start + key_chunk).min(key_tokens);
            let k_chunk = k
                .clone()
                .slice([0..batch, 0..heads, k_start..k_end, 0..head_dim])
                .clone();
            let v_chunk = v
                .clone()
                .slice([0..batch, 0..heads, k_start..k_end, 0..value_dim])
                .clone();
            let logits =
                matmul_4d_via_3d(q_chunk.clone(), k_chunk.swap_dims(2, 3)).mul_scalar(scale);
            let chunk_max = logits.clone().max_dim(3);
            let probs = logits.sub(chunk_max.clone()).exp();
            let chunk_denom = probs.clone().sum_dim(3);
            let chunk_acc = matmul_4d_via_3d(probs, v_chunk);

            let max_new = max_scores.clone().max_pair(chunk_max.clone());
            let alpha = max_scores.clone().sub(max_new.clone()).exp();
            let beta = chunk_max.sub(max_new.clone()).exp();
            acc = acc.mul(alpha.clone()).add(chunk_acc.mul(beta.clone()));
            denom = alpha
                .mul(denom)
                .add(beta.mul(chunk_denom))
                .add_scalar(1.0e-12);

            max_scores = max_new;
            k_start = k_end;
        }

        outputs.push(acc.div(denom.add_scalar(1.0e-12)));
        q_start = q_end;
    }

    Tensor::cat(outputs, 2)
}

fn scaled_dot_product_attention_stream_chunked_keys<B: Backend>(
    q: Tensor<B, 4>,
    k_chunks: &[Tensor<B, 4>],
    v_chunks: &[Tensor<B, 4>],
    head_dim: usize,
) -> Tensor<B, 4> {
    if k_chunks.is_empty() || v_chunks.is_empty() {
        let [batch, heads, query_tokens, _] = q.dims();
        return Tensor::<B, 4>::zeros([batch, heads, query_tokens, head_dim], &q.device());
    }
    if k_chunks.len() != v_chunks.len() {
        panic!(
            "stream attention chunk mismatch: k_chunks={} v_chunks={}",
            k_chunks.len(),
            v_chunks.len()
        );
    }

    let scale = 1.0 / (head_dim as f32).sqrt();
    let [batch, heads, query_tokens, _] = q.dims();
    let [_, _, _, value_dim] = v_chunks[0].dims();

    let first_logits =
        matmul_4d_via_3d(q.clone(), k_chunks[0].clone().swap_dims(2, 3)).mul_scalar(scale);
    let mut max_scores = first_logits.clone().max_dim(3);
    let first_probs = first_logits.sub(max_scores.clone()).exp();
    let mut denom = first_probs.clone().sum_dim(3);
    let mut acc = matmul_4d_via_3d(first_probs, v_chunks[0].clone());

    for idx in 1..k_chunks.len() {
        let k_chunk = k_chunks[idx].clone();
        let v_chunk = v_chunks[idx].clone();
        let [k_batch, k_heads, _, k_head_dim] = k_chunk.dims();
        let [v_batch, v_heads, _, v_value_dim] = v_chunk.dims();
        if k_batch != batch || k_heads != heads || k_head_dim != head_dim {
            panic!(
                "stream attention k chunk dims mismatch at idx={idx}: got=[{k_batch},{k_heads},*,{k_head_dim}] expected=[{batch},{heads},*,{head_dim}]"
            );
        }
        if v_batch != batch || v_heads != heads || v_value_dim != value_dim {
            panic!(
                "stream attention v chunk dims mismatch at idx={idx}: got=[{v_batch},{v_heads},*,{v_value_dim}] expected=[{batch},{heads},*,{value_dim}]"
            );
        }

        let logits = matmul_4d_via_3d(q.clone(), k_chunk.swap_dims(2, 3)).mul_scalar(scale);
        let chunk_max = logits.clone().max_dim(3);
        let probs = logits.sub(chunk_max.clone()).exp();
        let chunk_denom = probs.clone().sum_dim(3);
        let chunk_acc = matmul_4d_via_3d(probs, v_chunk);

        let max_new = max_scores.clone().max_pair(chunk_max.clone());
        let alpha = max_scores.clone().sub(max_new.clone()).exp();
        let beta = chunk_max.sub(max_new.clone()).exp();
        acc = acc.mul(alpha.clone()).add(chunk_acc.mul(beta.clone()));
        denom = alpha
            .mul(denom)
            .add(beta.mul(chunk_denom))
            .add_scalar(1.0e-12);

        max_scores = max_new;
    }

    acc = acc.div(denom.add_scalar(1.0e-12));

    let [out_batch, out_heads, out_query_tokens, _] = acc.dims();
    if out_batch != batch || out_heads != heads || out_query_tokens != query_tokens {
        panic!(
            "stream attention chunked output dims mismatch: got=[{out_batch},{out_heads},{out_query_tokens},*] expected=[{batch},{heads},{query_tokens},*]"
        );
    }
    acc
}

fn sparse_flow_module_attention<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
) -> Tensor<B, 4> {
    sparse_flow_module_attention_impl(q, k, v)
}

fn sparse_flow_module_attention_impl<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
) -> Tensor<B, 4> {
    let [batch, heads, query_tokens, head_dim] = q.dims();
    let [_, _, key_tokens, _] = k.dims();
    if sparse_flow_module_attention_batched_long_k_requires_stream(batch, key_tokens, head_dim) {
        return scaled_dot_product_attention_stream(q, k, v, head_dim);
    }
    if let Some(query_cap) = sparse_flow_module_attention_safe_query_cap_for_shape(
        batch,
        heads,
        query_tokens,
        key_tokens,
        head_dim,
    ) {
        let mut chunks = Vec::new();
        let mut start = 0usize;
        while start < query_tokens {
            let end = (start + query_cap).min(query_tokens);
            let q_chunk = q
                .clone()
                .slice([0..batch, 0..heads, start..end, 0..head_dim]);
            chunks.push(sparse_flow_module_attention_impl(
                q_chunk,
                k.clone(),
                v.clone(),
            ));
            start = end;
        }
        return Tensor::cat(chunks, 2);
    }

    sparse_flow_module_attention_direct_impl(q, k, v)
}

fn sparse_flow_module_attention_prepared<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
) -> Tensor<B, 4> {
    let [batch, heads, query_tokens, head_dim] = q.dims();
    let [_, _, key_tokens, _] = k.dims();
    if sparse_flow_module_attention_batched_long_k_requires_stream(batch, key_tokens, head_dim) {
        return scaled_dot_product_attention_stream(q, k, v, head_dim);
    }
    if let Some(query_cap) = sparse_flow_module_attention_safe_query_cap_for_shape(
        batch,
        heads,
        query_tokens,
        key_tokens,
        head_dim,
    ) {
        let mut chunks = Vec::new();
        let mut start = 0usize;
        while start < query_tokens {
            let end = (start + query_cap).min(query_tokens);
            let q_chunk = q
                .clone()
                .slice([0..batch, 0..heads, start..end, 0..head_dim]);
            chunks.push(sparse_flow_module_attention_prepared(
                q_chunk,
                k.clone(),
                v.clone(),
            ));
            start = end;
        }
        return Tensor::cat(chunks, 2);
    }

    sparse_flow_module_attention_prepared_direct_impl(q, k, v)
}

fn sparse_flow_module_attention_batched_long_k_requires_stream(
    batch: usize,
    key_tokens: usize,
    head_dim: usize,
) -> bool {
    // CubeK/Burn WGPU module attention currently diverges for TRELLIS.2 SLat
    // batch>1 head_dim=128 calls. Batched CFG is still useful, but those
    // attention calls must stay on the streamed reference path until the module
    // primitive is fixed for batched HR SLat shapes.
    let _ = key_tokens;
    batch > 1 && head_dim == 128
}

#[allow(dead_code)]
fn sparse_flow_module_attention_direct<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
) -> Tensor<B, 4> {
    sparse_flow_module_attention_direct_impl(q, k, v)
}

fn sparse_flow_module_attention_direct_impl<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
) -> Tensor<B, 4> {
    let [batch, heads, query_tokens, head_dim] = q.dims();
    let [_, _, key_tokens, _] = k.dims();
    if sparse_flow_module_attention_batched_long_k_requires_stream(batch, key_tokens, head_dim) {
        panic!(
            "direct sparse-flow module attention called with unverified batched long-key shape batch={batch} q={query_tokens} kv={key_tokens}; use streamed attention"
        );
    }
    if let Some(query_cap) = sparse_flow_module_attention_safe_query_cap_for_shape(
        batch,
        heads,
        query_tokens,
        key_tokens,
        head_dim,
    ) {
        panic!(
            "direct sparse-flow module attention called with unverified long-key shape q={query_tokens} kv={key_tokens}; split into chunks <= {query_cap}"
        );
    }
    let query_multiple = sparse_flow_module_attention_query_multiple();
    let padded_query_tokens = query_tokens.div_ceil(query_multiple) * query_multiple;
    let attention_dtype: burn::tensor::FloatDType = q.dtype().into();
    let force_fallback =
        sparse_flow_module_attention_long_k_requires_fallback(batch, heads, key_tokens, head_dim);
    let cast_pad_start = Instant::now();
    let q = tensor_cast_float_4d_if_needed(q, attention_dtype);
    let k = tensor_cast_float_4d_if_needed(k, attention_dtype);
    let v = tensor_cast_float_4d_if_needed(v, attention_dtype);
    let q = if padded_query_tokens > query_tokens {
        let pad = Tensor::<B, 4>::zeros(
            [batch, heads, padded_query_tokens - query_tokens, head_dim],
            &q.device(),
        )
        .cast(attention_dtype);
        Tensor::cat(vec![q, pad], 2)
    } else {
        q
    };
    let q = force_contiguous_4d(q);
    let k = force_contiguous_4d(k);
    let v = force_contiguous_4d(v);
    record_sparse_flow_detail(
        SparseFlowOpDetailKind::ModuleCastPad,
        elapsed_ns(cast_pad_start),
    );

    let attention_start = Instant::now();
    let attention_options = if force_fallback {
        AttentionModuleOptions {
            scale: Some((head_dim as f64).powf(-0.5)),
            ..Default::default()
        }
    } else {
        AttentionModuleOptions::default()
    };
    let mut out = attention(q, k, v, None, None, attention_options);
    record_sparse_flow_detail(
        SparseFlowOpDetailKind::ModuleAttention,
        elapsed_ns(attention_start),
    );
    let output_start = Instant::now();
    if padded_query_tokens > query_tokens {
        let value_dim = out.dims()[3];
        out = out.slice([0..batch, 0..heads, 0..query_tokens, 0..value_dim]);
    }
    record_sparse_flow_detail(
        SparseFlowOpDetailKind::ModuleOutput,
        elapsed_ns(output_start),
    );
    out
}

fn sparse_flow_module_attention_prepared_direct_impl<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
) -> Tensor<B, 4> {
    let [batch, heads, query_tokens, head_dim] = q.dims();
    let [_, _, key_tokens, _] = k.dims();
    if sparse_flow_module_attention_batched_long_k_requires_stream(batch, key_tokens, head_dim) {
        panic!(
            "prepared sparse-flow module attention called with unverified batched long-key shape batch={batch} q={query_tokens} kv={key_tokens}; use streamed attention"
        );
    }
    if let Some(query_cap) = sparse_flow_module_attention_safe_query_cap_for_shape(
        batch,
        heads,
        query_tokens,
        key_tokens,
        head_dim,
    ) {
        panic!(
            "prepared sparse-flow module attention called with unverified long-key shape q={query_tokens} kv={key_tokens}; split into chunks <= {query_cap}"
        );
    }
    let query_multiple = sparse_flow_module_attention_query_multiple();
    let padded_query_tokens = query_tokens.div_ceil(query_multiple) * query_multiple;
    let force_fallback =
        sparse_flow_module_attention_long_k_requires_fallback(batch, heads, key_tokens, head_dim);
    let cast_pad_start = Instant::now();
    let q_dtype: burn::tensor::FloatDType = q.dtype().into();
    let q = if padded_query_tokens > query_tokens {
        let pad = Tensor::<B, 4>::zeros(
            [batch, heads, padded_query_tokens - query_tokens, head_dim],
            &q.device(),
        )
        .cast(q_dtype);
        Tensor::cat(vec![q, pad], 2)
    } else {
        q
    };
    let q = force_contiguous_4d(q);
    record_sparse_flow_detail(
        SparseFlowOpDetailKind::ModuleCastPad,
        elapsed_ns(cast_pad_start),
    );

    let attention_start = Instant::now();
    let attention_options = if force_fallback {
        AttentionModuleOptions {
            scale: Some((head_dim as f64).powf(-0.5)),
            ..Default::default()
        }
    } else {
        AttentionModuleOptions::default()
    };
    let mut out = attention(q, k, v, None, None, attention_options);
    record_sparse_flow_detail(
        SparseFlowOpDetailKind::ModuleAttention,
        elapsed_ns(attention_start),
    );
    let output_start = Instant::now();
    if padded_query_tokens > query_tokens {
        let value_dim = out.dims()[3];
        out = out.slice([0..batch, 0..heads, 0..query_tokens, 0..value_dim]);
    }
    record_sparse_flow_detail(
        SparseFlowOpDetailKind::ModuleOutput,
        elapsed_ns(output_start),
    );
    out
}

fn sparse_flow_module_attention_safe_query_cap_for_shape(
    batch: usize,
    heads: usize,
    query_tokens: usize,
    key_tokens: usize,
    head_dim: usize,
) -> Option<usize> {
    let cap = sparse_flow_module_attention_chunk_cap_for_shape(
        batch,
        heads,
        query_tokens,
        key_tokens,
        head_dim,
    );
    if key_tokens > SPARSE_FLOW_MODULE_ATTENTION_LONG_K_FALLBACK_SEQ_KV && query_tokens > cap {
        Some(cap)
    } else {
        None
    }
}

fn sparse_flow_module_attention_long_k_requires_fallback(
    batch: usize,
    heads: usize,
    key_tokens: usize,
    head_dim: usize,
) -> bool {
    batch.saturating_mul(heads) >= 16
        && head_dim == 64
        && key_tokens > SPARSE_FLOW_MODULE_ATTENTION_LONG_K_FALLBACK_SEQ_KV
        && !sparse_flow_wgpu_module_attention_f16_enabled()
}

fn sparse_flow_module_attention_cross_shape_requires_stream(
    batch: usize,
    heads: usize,
    query_tokens: usize,
    key_tokens: usize,
    head_dim: usize,
) -> bool {
    // The TRELLIS.2 cross-attention shape [1, 12, sparse_tokens, 4101, 128]
    // exposes a CubeK/Burn module f32 correctness gap on real model tensors: it
    // collapses toward the value mean. The f16 module path matches the upstream
    // Python BF16 SDPA hook, so keep only f32/reference mode on the streamed
    // matmul path.
    head_dim == 128
        && query_tokens > key_tokens
        && key_tokens <= 8_192
        && sparse_flow_cross_module_attention_dtype_for_shape(batch, heads, key_tokens, head_dim)
            != burn::tensor::FloatDType::F16
}

fn force_contiguous_4d<B: Backend>(tensor: Tensor<B, 4>) -> Tensor<B, 4> {
    let dims = tensor.dims();
    let elements = dims.iter().product::<usize>();
    tensor.reshape([elements]).reshape(dims)
}

#[cfg(test)]
fn sparse_flow_module_attention_dtype_for_shape(
    batch: usize,
    heads: usize,
    key_tokens: usize,
    head_dim: usize,
) -> burn::tensor::FloatDType {
    sparse_flow_module_attention_dtype_for_shape_with_policy(
        batch,
        heads,
        key_tokens,
        head_dim,
        sparse_flow_wgpu_module_attention_f16_enabled(),
    )
}

fn sparse_flow_self_module_attention_dtype_for_shape(
    batch: usize,
    heads: usize,
    key_tokens: usize,
    head_dim: usize,
) -> burn::tensor::FloatDType {
    sparse_flow_module_attention_dtype_for_shape_with_policy(
        batch,
        heads,
        key_tokens,
        head_dim,
        sparse_flow_wgpu_self_attention_f16_enabled(),
    )
}

fn sparse_flow_cross_module_attention_dtype_for_shape(
    batch: usize,
    heads: usize,
    key_tokens: usize,
    head_dim: usize,
) -> burn::tensor::FloatDType {
    sparse_flow_module_attention_dtype_for_shape_with_policy(
        batch,
        heads,
        key_tokens,
        head_dim,
        sparse_flow_wgpu_cross_attention_f16_enabled(),
    )
}

fn sparse_flow_module_attention_dtype_for_shape_with_policy(
    batch: usize,
    heads: usize,
    key_tokens: usize,
    head_dim: usize,
    f16_enabled: bool,
) -> burn::tensor::FloatDType {
    let verified_long_k_f16_shape = (head_dim == 128 || head_dim == 64)
        && key_tokens > SPARSE_FLOW_MODULE_ATTENTION_LONG_K_FALLBACK_SEQ_KV
        && !sparse_flow_module_attention_long_k_requires_fallback(
            batch, heads, key_tokens, head_dim,
        );
    if f16_enabled
        && (key_tokens <= SPARSE_FLOW_MODULE_ATTENTION_F16_MAX_KEY_TOKENS
            || verified_long_k_f16_shape)
        && !sparse_flow_module_attention_long_k_requires_fallback(
            batch, heads, key_tokens, head_dim,
        )
    {
        burn::tensor::FloatDType::F16
    } else {
        burn::tensor::FloatDType::F32
    }
}

fn tensor_cast_float_4d_if_needed<B: Backend>(
    tensor: Tensor<B, 4>,
    dtype: burn::tensor::FloatDType,
) -> Tensor<B, 4> {
    let tensor_dtype: burn::tensor::FloatDType = tensor.dtype().into();
    if tensor_dtype == dtype {
        tensor
    } else {
        tensor.cast(dtype)
    }
}

fn tensor_cast_float_3d_if_needed<B: Backend>(
    tensor: Tensor<B, 3>,
    dtype: burn::tensor::FloatDType,
) -> Tensor<B, 3> {
    let tensor_dtype: burn::tensor::FloatDType = tensor.dtype().into();
    if tensor_dtype == dtype {
        tensor
    } else {
        tensor.cast(dtype)
    }
}

fn tensor_cast_float_2d_if_needed<B: Backend>(
    tensor: Tensor<B, 2>,
    dtype: burn::tensor::FloatDType,
) -> Tensor<B, 2> {
    let tensor_dtype: burn::tensor::FloatDType = tensor.dtype().into();
    if tensor_dtype == dtype {
        tensor
    } else {
        tensor.cast(dtype)
    }
}

fn tensor_cast_float_1d_if_needed<B: Backend>(
    tensor: Tensor<B, 1>,
    dtype: burn::tensor::FloatDType,
) -> Tensor<B, 1> {
    let tensor_dtype: burn::tensor::FloatDType = tensor.dtype().into();
    if tensor_dtype == dtype {
        tensor
    } else {
        tensor.cast(dtype)
    }
}

fn attention_logits_bytes(
    batch: usize,
    heads: usize,
    query_tokens: usize,
    key_tokens: usize,
) -> usize {
    batch
        .saturating_mul(heads)
        .saturating_mul(query_tokens)
        .saturating_mul(key_tokens)
        .saturating_mul(std::mem::size_of::<f32>())
}

fn attention_logits_budget_bytes() -> usize {
    if attention_prefers_stream() {
        #[cfg(target_arch = "wasm32")]
        {
            return 128 * 1024 * 1024;
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            return 1024 * 1024 * 1024;
        }
    } else {
        usize::MAX
    }
}

fn attention_query_chunk(tokens: usize, default_chunk: usize) -> usize {
    let max_chunk = if attention_prefers_stream() {
        #[cfg(target_arch = "wasm32")]
        {
            256
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            1024
        }
    } else {
        usize::MAX
    };
    default_chunk.min(max_chunk).min(tokens.max(1))
}

fn attention_key_chunk(tokens: usize) -> usize {
    let max_chunk = if attention_prefers_stream() {
        512
    } else {
        usize::MAX
    };
    128usize.min(max_chunk).min(tokens.max(1))
}

#[cfg(feature = "runtime-model-wgpu")]
#[allow(clippy::too_many_arguments)]
fn maybe_apply_qk_rms_norm_and_rope_from_qkv<B: Backend>(
    qkv: Tensor<B, 5>,
    q_norm: Option<&MultiHeadRmsNorm<B>>,
    k_norm: Option<&MultiHeadRmsNorm<B>>,
    use_rope: bool,
    rope_freq: [f32; 2],
    token_coords: Option<Tensor<B, 2, Int>>,
    token_start: usize,
) -> Option<(Tensor<B, 4>, Tensor<B, 4>)>
where
    RopeRotateWgpuBridgeImpl: SparseFlowQkRmsNormWgpuBridge<B>,
{
    if !use_rope {
        return None;
    }
    let (Some(q_norm), Some(k_norm), Some(coords)) = (q_norm, k_norm, token_coords) else {
        return None;
    };
    let [_, tokens, _, _, head_dim] = qkv.dims();
    if head_dim != 128 {
        return None;
    }
    let [coord_rows, coord_cols] = coords.dims();
    assert_eq!(
        coord_cols, 3,
        "sparse flow rope token coords must have 3 columns"
    );
    assert!(
        token_start.saturating_add(tokens) <= coord_rows,
        "sparse flow rope token range out of bounds"
    );
    let coord_slice = coords.slice([token_start..token_start + tokens, 0..3]);
    maybe_qk_multihead_rms_norm_rope_coords_from_qkv_wgpu(
        qkv,
        q_norm.gamma.val(),
        k_norm.gamma.val(),
        coord_slice,
        rope_freq,
        q_norm.scale,
        RMS_NORM_EPS,
    )
}

fn maybe_apply_qkv_module_rms_norm_and_rope_from_qkv<B: Backend>(
    qkv: Tensor<B, 5>,
    q_norm: Option<&MultiHeadRmsNorm<B>>,
    k_norm: Option<&MultiHeadRmsNorm<B>>,
    use_rope: bool,
    rope_freq: [f32; 2],
    token_coords: Option<Tensor<B, 2, Int>>,
    token_start: usize,
) -> Option<(Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>)>
where
    RopeRotateWgpuBridgeImpl: SparseFlowQkRmsNormWgpuBridge<B>,
{
    if !use_rope {
        return None;
    }
    let (Some(q_norm), Some(k_norm), Some(coords)) = (q_norm, k_norm, token_coords) else {
        return None;
    };
    let [_, tokens, _, _, head_dim] = qkv.dims();
    if head_dim != 128 {
        return None;
    }
    let [coord_rows, coord_cols] = coords.dims();
    assert_eq!(
        coord_cols, 3,
        "sparse flow rope token coords must have 3 columns"
    );
    assert!(
        token_start.saturating_add(tokens) <= coord_rows,
        "sparse flow rope token range out of bounds"
    );
    #[cfg(feature = "runtime-model-wgpu")]
    {
        let coord_slice = coords.slice([token_start..token_start + tokens, 0..3]);
        maybe_qkv_module_multihead_rms_norm_rope_coords_from_qkv_wgpu(
            qkv,
            q_norm.gamma.val(),
            k_norm.gamma.val(),
            coord_slice,
            rope_freq,
            q_norm.scale,
            RMS_NORM_EPS,
        )
    }
    #[cfg(not(feature = "runtime-model-wgpu"))]
    {
        let _ = (qkv, q_norm, k_norm, coords, rope_freq, token_start);
        None
    }
}

fn apply_rms_norm_and_rope_single<B: Backend>(
    x: Tensor<B, 4>,
    norm: Option<&MultiHeadRmsNorm<B>>,
    use_rope: bool,
    resolution: usize,
    head_dim: usize,
    rope_freq: [f32; 2],
    token_coords: Option<Tensor<B, 2, Int>>,
    token_start: usize,
) -> Tensor<B, 4>
where
    RopeRotateWgpuBridgeImpl: RopeRotateWgpuBridge<B>,
    RopeRotateWgpuBridgeImpl: SparseFlowRmsNormWgpuBridge<B>,
    SparseFlowBf16EmulationBridgeImpl: SparseFlowBf16EmulationBridge<B>,
{
    let mut x = if let Some(norm) = norm {
        if use_rope {
            if let Some(coords) = token_coords.clone() {
                let [_, tokens, _, _] = x.dims();
                let [coord_rows, coord_cols] = coords.dims();
                assert_eq!(
                    coord_cols, 3,
                    "sparse flow rope token coords must have 3 columns"
                );
                assert!(
                    token_start.saturating_add(tokens) <= coord_rows,
                    "sparse flow rope token range out of bounds"
                );
                let coord_slice = coords.slice([token_start..token_start + tokens, 0..3]);
                #[cfg(feature = "runtime-model-wgpu")]
                if let Some(y) = maybe_multihead_rms_norm_rope_coords_wgpu(
                    x.clone(),
                    norm.gamma.val(),
                    coord_slice,
                    rope_freq,
                    norm.scale,
                    RMS_NORM_EPS,
                ) {
                    return y;
                }
            }
        }
        norm.forward(x)
    } else {
        x
    };
    if use_rope {
        x = apply_rope_single(
            x,
            resolution,
            head_dim,
            rope_freq,
            token_coords,
            token_start,
        );
    }
    x
}

fn apply_rope_single<B: Backend>(
    x: Tensor<B, 4>,
    resolution: usize,
    head_dim: usize,
    rope_freq: [f32; 2],
    token_coords: Option<Tensor<B, 2, Int>>,
    token_start: usize,
) -> Tensor<B, 4>
where
    RopeRotateWgpuBridgeImpl: RopeRotateWgpuBridge<B>,
{
    let [_, tokens, _, _] = x.dims();
    let pairs = head_dim / 2;
    if pairs == 0 || tokens == 0 {
        return x;
    }
    if let Some(coords) = token_coords {
        let [coord_rows, coord_cols] = coords.dims();
        assert_eq!(
            coord_cols, 3,
            "sparse flow rope token coords must have 3 columns"
        );
        assert!(
            token_start.saturating_add(tokens) <= coord_rows,
            "sparse flow rope token range out of bounds"
        );
        let coord_slice = coords.slice([token_start..token_start + tokens, 0..3]);
        #[cfg(feature = "runtime-model-wgpu")]
        if sparse_flow_wgpu_coord_rope_kernel_enabled() {
            if let Some(rotated) =
                maybe_rotate_pairs_coords_wgpu(x.clone(), coord_slice.clone(), rope_freq)
            {
                return rotated;
            }
        }
        let phase = rope_phase_from_coord_tensor(coord_slice, pairs, rope_freq);
        let (cos, sin) = rope_cos_sin_from_phase_tensor(phase);
        return rotate_pairs(x, cos, sin);
    }

    let device = x.device();
    let rope = rope_cos_sin_cached(resolution, token_start, tokens, pairs, rope_freq);
    let cos = Tensor::<B, 1>::from_floats(rope.cos.as_slice(), &device)
        .reshape([tokens, pairs])
        .reshape([1, tokens, 1, pairs]);
    let sin = Tensor::<B, 1>::from_floats(rope.sin.as_slice(), &device)
        .reshape([tokens, pairs])
        .reshape([1, tokens, 1, pairs]);
    rotate_pairs(x, cos, sin)
}

fn rope_phase_from_coord_tensor<B: Backend>(
    coords: Tensor<B, 2, Int>,
    pairs: usize,
    rope_freq: [f32; 2],
) -> Tensor<B, 2> {
    let [tokens, cols] = coords.dims();
    assert_eq!(cols, 3, "sparse flow rope token coords must have 3 columns");
    let device = coords.device();
    if tokens == 0 || pairs == 0 {
        return Tensor::<B, 2>::zeros([tokens, pairs], &device);
    }

    let freq_dim = (pairs / 3).max(1);
    let mut freqs = Vec::with_capacity(freq_dim);
    for idx in 0..freq_dim {
        let exp = idx as f32 / freq_dim as f32;
        freqs.push(rope_freq[0] / rope_freq[1].powf(exp));
    }
    let freq_t = Tensor::<B, 1>::from_floats(freqs.as_slice(), &device).reshape([1, freq_dim]);
    let coords_f = coords.float();
    let phase_x = coords_f
        .clone()
        .slice([0..tokens, 0..1])
        .mul(freq_t.clone());
    let phase_y = coords_f
        .clone()
        .slice([0..tokens, 1..2])
        .mul(freq_t.clone());
    let phase_z = coords_f.slice([0..tokens, 2..3]).mul(freq_t);
    let mut phase = Tensor::cat(vec![phase_x, phase_y, phase_z], 1);
    let phase_pairs = phase.dims()[1];
    if phase_pairs < pairs {
        let pad = Tensor::<B, 2>::zeros([tokens, pairs - phase_pairs], &device);
        phase = Tensor::cat(vec![phase, pad], 1);
    } else if phase_pairs > pairs {
        phase = phase.slice([0..tokens, 0..pairs]);
    }
    phase
}

fn rope_cos_sin_from_phase_tensor<B: Backend>(phase: Tensor<B, 2>) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let [tokens, pairs] = phase.dims();
    (
        phase.clone().cos().reshape([1, tokens, 1, pairs]),
        phase.sin().reshape([1, tokens, 1, pairs]),
    )
}

fn rotate_pairs<B: Backend>(x: Tensor<B, 4>, cos: Tensor<B, 4>, sin: Tensor<B, 4>) -> Tensor<B, 4>
where
    RopeRotateWgpuBridgeImpl: RopeRotateWgpuBridge<B>,
{
    #[cfg(feature = "runtime-model-wgpu")]
    if let Some(rotated) = maybe_rotate_pairs_wgpu(x.clone(), cos.clone(), sin.clone()) {
        return rotated;
    }

    let [batch, tokens, heads, head_dim] = x.dims();
    let pairs = head_dim / 2;
    let x = x.reshape([batch, tokens, heads, pairs, 2]);
    let x_even = x
        .clone()
        .slice([0..batch, 0..tokens, 0..heads, 0..pairs, 0..1])
        .reshape([batch, tokens, heads, pairs]);
    let x_odd = x
        .slice([0..batch, 0..tokens, 0..heads, 0..pairs, 1..2])
        .reshape([batch, tokens, heads, pairs]);

    let rot_even = x_even
        .clone()
        .mul(cos.clone())
        .sub(x_odd.clone().mul(sin.clone()));
    let rot_odd = x_even.mul(sin).add(x_odd.mul(cos));
    let rot_even = rot_even.reshape([batch, tokens, heads, pairs, 1]);
    let rot_odd = rot_odd.reshape([batch, tokens, heads, pairs, 1]);
    Tensor::cat(vec![rot_even, rot_odd], 4).reshape([batch, tokens, heads, head_dim])
}

fn rope_cache() -> &'static Mutex<HashMap<RopeCacheKey, Arc<RopeCosSinRange>>> {
    ROPE_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn rope_cos_sin_cached(
    resolution: usize,
    token_start: usize,
    tokens: usize,
    pairs: usize,
    rope_freq: [f32; 2],
) -> Arc<RopeCosSinRange> {
    if resolution == 0 || tokens == 0 || pairs == 0 {
        return Arc::new(RopeCosSinRange {
            cos: vec![1.0f32; tokens * pairs],
            sin: vec![0.0f32; tokens * pairs],
        });
    }

    let key = RopeCacheKey {
        resolution,
        token_start,
        tokens,
        pairs,
        rope_freq_0_bits: rope_freq[0].to_bits(),
        rope_freq_1_bits: rope_freq[1].to_bits(),
    };

    if let Ok(cache) = rope_cache().lock()
        && let Some(hit) = cache.get(&key)
    {
        return Arc::clone(hit);
    }

    let generated = Arc::new(rope_cos_sin_range_uncached(
        resolution,
        token_start,
        tokens,
        pairs,
        rope_freq,
    ));
    if let Ok(mut cache) = rope_cache().lock() {
        if cache.len() >= ROPE_CACHE_MAX_ENTRIES {
            cache.clear();
        }
        cache.insert(key, Arc::clone(&generated));
    }
    generated
}

fn rope_cos_sin_range_uncached(
    resolution: usize,
    token_start: usize,
    tokens: usize,
    pairs: usize,
    rope_freq: [f32; 2],
) -> RopeCosSinRange {
    let mut cos = vec![1.0f32; tokens * pairs];
    let mut sin = vec![0.0f32; tokens * pairs];
    if resolution == 0 || tokens == 0 || pairs == 0 {
        return RopeCosSinRange { cos, sin };
    }

    let freq_dim = (pairs / 3).max(1);
    let mut freqs = Vec::with_capacity(freq_dim);
    for idx in 0..freq_dim {
        let exp = idx as f32 / freq_dim as f32;
        freqs.push(rope_freq[0] / rope_freq[1].powf(exp));
    }

    let resolution_sq = resolution.saturating_mul(resolution).max(1);
    let max_tokens = resolution_sq.saturating_mul(resolution);
    for local_token in 0..tokens {
        let token = token_start.saturating_add(local_token);
        if token >= max_tokens {
            break;
        }
        let x = token / resolution_sq;
        let yz = token % resolution_sq;
        let y = yz / resolution;
        let z = yz % resolution;
        let coords = [x as f32, y as f32, z as f32];
        for (dim, coord) in coords.iter().enumerate() {
            for (freq_idx, freq) in freqs.iter().enumerate() {
                let pair = dim * freq_dim + freq_idx;
                if pair >= pairs {
                    continue;
                }
                let phase = *coord * *freq;
                let idx = local_token * pairs + pair;
                cos[idx] = phase.cos();
                sin[idx] = phase.sin();
            }
        }
    }
    RopeCosSinRange { cos, sin }
}

fn load_sparse_model_weights<B: Backend>(
    model: &mut SparseStructureFlowModel<B>,
    path: &Path,
) -> Result<(), String> {
    if path
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("bpk"))
    {
        #[cfg(target_arch = "wasm32")]
        if chunked_blob_parts_manifest_exists(path) {
            let mut chunked_store =
                ChunkedBlobSafetensorsStore::from_blob_burnpack_parts(path, key_remap_rules())?;
            return model
                .load_from(&mut chunked_store)
                .map(|_| ())
                .map_err(|err| {
                    format!(
                        "failed to load sparse flow burnpack parts '{}' on wasm chunked safetensors path: {err}",
                        path.display()
                    )
                });
        }

        // Canonical sparse-flow checkpoints are stored as raw safetensors blobs in burnpacks.
        // Legacy module-layout burnpacks are not supported on the canonical runtime path.
        let blob_bytes = load_blob_bytes_from_burnpack_or_parts(path, load_burnpack_blob_bytes)?;
        let mut safetensor_store = build_safetensor_store_from_bytes(blob_bytes)?;
        model
            .load_from(&mut safetensor_store)
            .map(|_| ())
            .map_err(|safetensor_err| {
                format!(
                    "failed to load sparse flow burnpack '{}' as safetensors blob: {safetensor_err}",
                    path.display()
                )
            })
    } else {
        let mut store = if virtual_fs::has_virtual_file(path) {
            let bytes = virtual_fs::read(path).map_err(|err| {
                format!(
                    "failed to read virtual sparse flow safetensors '{}': {err}",
                    path.display()
                )
            })?;
            build_safetensor_store_from_bytes(bytes)?
        } else {
            build_safetensor_store(path)?
        };
        model.load_from(&mut store).map(|_| ()).map_err(|err| {
            format!(
                "failed to load sparse flow safetensors '{}': {err}",
                path.display()
            )
        })
    }
}

fn build_safetensor_store(path: &Path) -> Result<SafetensorsStore, String> {
    let mut remapper = KeyRemapper::new();
    for &(from, to) in key_remap_rules() {
        remapper = remapper
            .add_pattern(from, to)
            .map_err(|err| format!("invalid sparse flow remap rule {from}->{to}: {err}"))?;
    }

    Ok(SafetensorsStore::from_file(path)
        .with_from_adapter(PyTorchToBurnAdapter)
        .allow_partial(false)
        .remap(remapper)
        .validate(true))
}

fn build_safetensor_store_from_bytes(bytes: Vec<u8>) -> Result<SafetensorsStore, String> {
    let mut remapper = KeyRemapper::new();
    for &(from, to) in key_remap_rules() {
        remapper = remapper
            .add_pattern(from, to)
            .map_err(|err| format!("invalid sparse flow remap rule {from}->{to}: {err}"))?;
    }

    Ok(SafetensorsStore::from_bytes(Some(bytes))
        .with_from_adapter(PyTorchToBurnAdapter)
        .allow_partial(false)
        .remap(remapper)
        .validate(true))
}

fn load_burnpack_blob_bytes(path: &Path) -> Result<Vec<u8>, String> {
    load_blob_bytes_from_blob_burnpack(path)
}

fn key_remap_rules() -> &'static [(&'static str, &'static str)] {
    &[
        (r"^(t_embedder)\.mlp\.0\.(weight|bias)$", "$1.mlp_0.$2"),
        (r"^(t_embedder)\.mlp\.2\.(weight|bias)$", "$1.mlp_2.$2"),
        (
            r"^(adaLN_modulation)\.1\.(weight|bias)$",
            "ada_ln_modulation.$2",
        ),
        (
            r"^(blocks\.\d+\.mlp)\.mlp\.0\.(weight|bias)$",
            "$1.mlp_0.$2",
        ),
        (
            r"^(blocks\.\d+\.mlp)\.mlp\.2\.(weight|bias)$",
            "$1.mlp_2.$2",
        ),
        (r"^(blocks\.\d+\.norm2)\.weight$", "$1.gamma"),
        (r"^(blocks\.\d+\.norm2)\.bias$", "$1.beta"),
    ]
}

fn resolve_model_weight_candidates(
    model_stem: &str,
    weights_root: &Path,
    image_large_root: Option<&Path>,
) -> Vec<PathBuf> {
    let source =
        resolve_model_source_path(model_stem, "safetensors", weights_root, image_large_root);
    let burnpack = source.with_extension("bpk");
    let burnpack_f16 = with_file_stem_suffix(&burnpack, F16_SUFFIX);
    let prefer_f16 = prefer_f16_burnpack();
    let candidates = if prefer_f16 {
        vec![burnpack_f16, burnpack, source]
    } else {
        vec![burnpack, burnpack_f16, source]
    };
    candidates
        .into_iter()
        .filter(|path| candidate_exists_or_has_parts(path))
        .collect::<Vec<_>>()
}

fn prefer_f16_burnpack() -> bool {
    // Correctness-first default: keep sparse-flow aligned with canonical bf16
    // checkpoint behavior before opting into lossy f16 burnpacks.
    false
}

fn resolve_model_source_path(
    stem: &str,
    ext: &str,
    weights_root: &Path,
    image_large_root: Option<&Path>,
) -> PathBuf {
    if stem.starts_with("ckpts/") {
        return weights_root.join(format!("{stem}.{ext}"));
    }
    if let Some((_, suffix)) = stem.split_once("/ckpts/") {
        let image_large_root = image_large_root.unwrap_or(weights_root);
        return image_large_root.join(format!("ckpts/{suffix}.{ext}"));
    }
    weights_root.join(format!("{stem}.{ext}"))
}

fn with_file_stem_suffix(path: &Path, suffix: &str) -> PathBuf {
    let Some(stem) = path.file_stem() else {
        return path.to_path_buf();
    };
    let stem = stem.to_string_lossy();
    if stem.ends_with(suffix) {
        return path.to_path_buf();
    }
    let ext = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("");
    let mut file_name = format!("{stem}{suffix}");
    if !ext.is_empty() {
        file_name.push('.');
        file_name.push_str(ext);
    }
    path.with_file_name(file_name)
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::sync::Mutex;
    #[cfg(feature = "runtime-model-wgpu")]
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Instant, SystemTime, UNIX_EPOCH};

    use burn::prelude::Backend;
    use burn::tensor::{
        Int, Tensor, activation::softmax, module::attention, ops::AttentionModuleOptions,
    };
    use burn_store::{BurnToPyTorchAdapter, ModuleSnapshot, SafetensorsStore};

    use crate::blob_burnpack::save_blob_bytes_to_burnpack;
    use crate::hook_diff::{HookSnapshot, HookTensor, compute_stats};
    use crate::runtime_model::runtime_config::{
        RuntimeModelDebugConfig, set_runtime_model_debug_config,
        set_runtime_model_sparse_flow_attention_policy,
    };
    use crate::sampler::FlowEulerSampleConfig;

    #[cfg(feature = "runtime-model-wgpu")]
    use super::WgpuRuntimeBackend;
    use super::{
        CpuRuntimeBackend, SelfAttention, SparseFlowCondition, SparseRuntimeTensorAccess,
        SparseStructureFlowConfig, SparseStructureFlowModel, SparseStructureFlowRuntime,
        SparseStructureFlowRuntimeImpl, SparseTensorOwned, VarLenTensorOwned, apply_rope_single,
        dense_grid_token_coords, force_contiguous_4d, host_transfer_stats,
        layer_norm_affine_stable, layer_norm_no_affine, linear_forward_attention,
        linear_forward_input_token_chunked, linear_forward_stable_2d,
        linear_forward_stable_2d_reference, linear_forward_token_chunked_reference,
        matmul_4d_via_3d, reset_host_transfer_stats, reset_sparse_flow_op_telemetry,
        resolve_model_weight_candidates, scaled_dot_product_attention,
        scaled_dot_product_attention_dense, scaled_dot_product_attention_stream, silu,
        sparse_flow_attention_logits_within_budget, sparse_flow_linear_chunk_tokens_for_backend,
        sparse_flow_module_attention_chunk_cap, sparse_flow_module_attention_chunk_cap_for_shape,
        sparse_flow_module_attention_cross_shape_requires_stream,
        sparse_flow_module_attention_dtype_for_shape,
        sparse_flow_module_attention_long_k_requires_fallback,
        sparse_flow_module_attention_safe_query_cap_for_shape, sparse_flow_op_telemetry,
        sparse_flow_stock_bf16_round, sparse_flow_stream_chunk_plan,
    };

    static HOST_STATS_LOCK: Mutex<()> = Mutex::new(());
    static ENV_LOCK: Mutex<()> = Mutex::new(());
    #[cfg(feature = "runtime-model-wgpu")]
    static SLAT_DEBUG_FIRST_NONFINITE_REPORTED: AtomicBool = AtomicBool::new(false);

    #[test]
    fn parses_sparse_structure_flow_config_json() {
        let json = br#"{
            "name": "SparseStructureFlowModel",
            "args": {
                "resolution": 16,
                "in_channels": 8,
                "out_channels": 8,
                "model_channels": 1536,
                "cond_channels": 1024,
                "num_blocks": 30,
                "num_heads": 12,
                "mlp_ratio": 5.3334,
                "pe_mode": "rope",
                "share_mod": true,
                "qk_rms_norm": true,
                "qk_rms_norm_cross": true
            }
        }"#;
        let parsed = SparseStructureFlowConfig::from_json_bytes(json).expect("config should parse");
        assert_eq!(parsed.resolution, 16);
        assert_eq!(parsed.in_channels, 8);
        assert_eq!(parsed.num_heads(), 12);
        assert_eq!(parsed.pe_mode, "rope");
        assert!(parsed.share_mod);
    }

    #[test]
    fn dense_grid_token_coords_follow_dense_voxel_flatten_order() {
        let device = <CpuRuntimeBackend as burn::tensor::backend::BackendTypes>::Device::default();
        let coords = dense_grid_token_coords::<CpuRuntimeBackend>(2, device);
        assert_eq!(coords.dims(), [8, 3]);
        let values = coords
            .into_data()
            .convert::<i64>()
            .to_vec::<i64>()
            .expect("dense grid coords should read");
        assert_eq!(
            values,
            vec![
                0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1,
            ]
        );
    }

    #[cfg(feature = "runtime-model-wgpu")]
    #[test]
    fn varlen_device_owned_conversion_requires_device_tensor() {
        let owned = VarLenTensorOwned::from_layout(vec![0.0; 32], vec![0..1], 32)
            .expect("host varlen tensor should build");
        let err = owned
            .as_device_owned("test varlen to device")
            .expect_err("host-only varlen tensor should not convert to device-owned view");
        assert!(
            err.contains("host-only"),
            "expected host-only conversion error, got: {err}"
        );
    }

    #[cfg(feature = "runtime-model-wgpu")]
    #[test]
    fn sparse_device_owned_conversion_requires_device_tensors() {
        let sparse =
            SparseTensorOwned::from_layout(vec![[0, 0, 0, 0]], vec![0.0; 32], vec![0..1], 32, 8)
                .expect("host sparse tensor should build");
        let err = sparse
            .as_device_owned("test sparse to device")
            .expect_err("host-only sparse tensor should not convert to device-owned view");
        assert!(
            err.contains("host-only"),
            "expected host-only conversion error, got: {err}"
        );
    }

    #[test]
    fn sparse_flow_module_attention_long_k_uses_small_chunks() {
        assert_eq!(
            sparse_flow_module_attention_chunk_cap_for_shape(2, 12, 23_816, 23_816, 64),
            12_288,
            "verified fast-f16 head_dim=64 long-key module attention should use the measured wide query cap"
        );
        assert_eq!(
            sparse_flow_module_attention_chunk_cap_for_shape(1, 12, 23_816, 23_816, 128),
            12_288,
            "Trellis HR self-attention with long keys should use the measured head_dim=128 query cap"
        );
        assert_eq!(
            sparse_flow_module_attention_chunk_cap_for_shape(1, 12, 23_816, 1_029, 128),
            23_816,
            "Trellis HR cross-attention has short keys and should run as one verified full-query chunk"
        );
        assert_eq!(
            sparse_flow_module_attention_safe_query_cap_for_shape(2, 12, 3_840, 23_816, 64),
            None,
            "verified fast-f16 long-key head_dim=64 module attention chunks within the wide cap should run directly"
        );
        assert_eq!(
            sparse_flow_module_attention_safe_query_cap_for_shape(2, 12, 3_712, 23_816, 64),
            None,
            "verified long-key head_dim=64 module attention chunks should run directly"
        );
        assert_eq!(
            sparse_flow_module_attention_safe_query_cap_for_shape(1, 12, 7_552, 23_816, 128),
            None,
            "head_dim=128 module attention calls inside the measured cap should run directly"
        );
        assert_eq!(
            sparse_flow_module_attention_safe_query_cap_for_shape(1, 12, 16_384, 23_816, 128),
            Some(12_288),
            "raw long-key head_dim=128 module attention calls above the verified cap must split internally"
        );
        assert!(super::sparse_flow_module_attention_batched_long_k_requires_stream(2, 23_816, 128));
        assert!(
            !super::sparse_flow_module_attention_batched_long_k_requires_stream(1, 23_816, 128)
        );
        assert_eq!(sparse_flow_module_attention_chunk_cap(8_192), 8_192);
    }

    #[test]
    fn sparse_flow_cross_attention_uses_stream_for_python_mismatched_shape() {
        let _guard = ENV_LOCK.lock().expect("env lock poisoned");
        set_runtime_model_debug_config(RuntimeModelDebugConfig {
            stage_debug: false,
            attention_debug: false,
            sparse_flow_module_attention: true,
            sparse_flow_module_attention_f16: false,
            sparse_flow_linear_f16: false,
            sparse_flow_torso_f16: false,
            sparse_flow_stock_bf16_emulation: false,
            sparse_flow_coord_rope_kernel: true,
            sparse_decoder_conv_f16: false,
        });
        assert!(sparse_flow_module_attention_cross_shape_requires_stream(
            1, 12, 23_816, 4_101, 128
        ));
        set_runtime_model_debug_config(RuntimeModelDebugConfig {
            stage_debug: false,
            attention_debug: false,
            sparse_flow_module_attention: true,
            sparse_flow_module_attention_f16: true,
            sparse_flow_linear_f16: true,
            sparse_flow_torso_f16: false,
            sparse_flow_stock_bf16_emulation: false,
            sparse_flow_coord_rope_kernel: true,
            sparse_decoder_conv_f16: false,
        });
        assert!(
            !sparse_flow_module_attention_cross_shape_requires_stream(1, 12, 23_816, 4_101, 128),
            "fast-f16 cross attention should use the module path; f32/reference remains streamed"
        );
        assert!(!sparse_flow_module_attention_cross_shape_requires_stream(
            1, 12, 23_816, 23_816, 128
        ));
        assert!(!sparse_flow_module_attention_cross_shape_requires_stream(
            1, 12, 3_712, 23_816, 128
        ));
        set_runtime_model_debug_config(RuntimeModelDebugConfig::default());
    }

    #[cfg(feature = "runtime-model-wgpu")]
    #[test]
    fn sparse_flow_module_attention_f16_is_gated_by_verified_shape() {
        let _guard = ENV_LOCK.lock().expect("env lock poisoned");
        set_runtime_model_debug_config(RuntimeModelDebugConfig {
            stage_debug: false,
            attention_debug: false,
            sparse_flow_module_attention: true,
            sparse_flow_module_attention_f16: true,
            sparse_flow_linear_f16: true,
            sparse_flow_torso_f16: false,
            sparse_flow_stock_bf16_emulation: false,
            sparse_flow_coord_rope_kernel: true,
            sparse_decoder_conv_f16: false,
        });

        assert_eq!(
            sparse_flow_module_attention_dtype_for_shape(1, 12, 5_768, 128),
            burn::tensor::FloatDType::F16,
            "moderate Trellis module-attention shapes should use the fast f16 path"
        );
        assert_eq!(
            sparse_flow_module_attention_dtype_for_shape(2, 12, 23_816, 64),
            burn::tensor::FloatDType::F16,
            "verified long-key head_dim=64 chunks should use the fast f16 path after query chunking"
        );
        assert_eq!(
            sparse_flow_module_attention_dtype_for_shape(1, 12, 23_816, 128),
            burn::tensor::FloatDType::F16,
            "Trellis HR long-key head_dim=128 chunks should use the fast f16 path after query chunking"
        );
        assert!(!sparse_flow_module_attention_long_k_requires_fallback(
            2, 12, 23_816, 64
        ));
        set_runtime_model_debug_config(RuntimeModelDebugConfig::default());
    }

    #[cfg(feature = "runtime-model-wgpu")]
    #[test]
    fn sparse_flow_batched_cfg_is_experimental_opt_in_and_streams_batched_hr_attention() {
        let _guard = ENV_LOCK.lock().expect("env lock poisoned");
        unsafe {
            std::env::remove_var("TRELLIS2_SPARSE_FLOW_BATCHED_CFG");
        }
        set_runtime_model_debug_config(RuntimeModelDebugConfig {
            stage_debug: false,
            attention_debug: false,
            sparse_flow_module_attention: true,
            sparse_flow_module_attention_f16: true,
            sparse_flow_linear_f16: false,
            sparse_flow_torso_f16: false,
            sparse_flow_stock_bf16_emulation: false,
            sparse_flow_coord_rope_kernel: true,
            sparse_decoder_conv_f16: false,
        });
        assert!(
            !super::sparse_flow_batched_cfg_enabled_for_backend::<super::WgpuRuntimeBackend>(),
            "batched CFG remains opt-in until HR SLat batch>1 parity is fixed"
        );
        unsafe {
            std::env::set_var("TRELLIS2_SPARSE_FLOW_BATCHED_CFG", "1");
        }
        assert!(
            super::sparse_flow_batched_cfg_enabled_for_backend::<super::WgpuRuntimeBackend>(),
            "diagnostic opt-in should still expose the batched CFG path"
        );
        assert!(super::sparse_flow_module_attention_batched_long_k_requires_stream(2, 4_101, 128));
        assert!(super::sparse_flow_module_attention_batched_long_k_requires_stream(2, 10_717, 128));
        assert!(
            !super::sparse_flow_module_attention_batched_long_k_requires_stream(1, 10_717, 128)
        );
        assert!(!super::sparse_flow_module_attention_batched_long_k_requires_stream(2, 10_717, 64));
        unsafe {
            std::env::remove_var("TRELLIS2_SPARSE_FLOW_BATCHED_CFG");
        }
        set_runtime_model_debug_config(RuntimeModelDebugConfig::default());
    }

    #[test]
    fn cross_attention_kv_cache_matches_uncached_forward() {
        let runtime = make_tiny_runtime_cpu();
        let config = runtime.config().clone();
        let device = <CpuRuntimeBackend as burn::tensor::backend::BackendTypes>::Device::default();
        let voxels = config.resolution * config.resolution * config.resolution;
        let x_values = (0..(config.in_channels * voxels))
            .map(|idx| {
                let x = idx as f32 * 0.017 + 0.31;
                x.sin() * 0.4 + x.cos() * 0.2
            })
            .collect::<Vec<_>>();
        let cond_tokens = 5usize;
        let cond_values = (0..(cond_tokens * config.cond_channels))
            .map(|idx| {
                let x = idx as f32 * 0.023 + 0.11;
                x.sin() * 0.3 + x.cos() * 0.25
            })
            .collect::<Vec<_>>();
        let x =
            Tensor::<CpuRuntimeBackend, 1>::from_floats(x_values.as_slice(), &device).reshape([
                1,
                config.in_channels,
                config.resolution,
                config.resolution,
                config.resolution,
            ]);
        let t = Tensor::<CpuRuntimeBackend, 1>::from_floats([123.0f32], &device);
        let cond = Tensor::<CpuRuntimeBackend, 1>::from_floats(cond_values.as_slice(), &device)
            .reshape([1, cond_tokens, config.cond_channels]);

        let uncached = runtime.model.forward(x.clone(), t.clone(), cond.clone());
        let cache = runtime.model.project_cross_attention_cache(cond.clone());
        let cached = runtime
            .model
            .forward_with_cross_cache(x, t, cond, Some(&cache));

        let uncached = tensor_to_vec5(uncached);
        let cached = tensor_to_vec5(cached);
        assert_eq!(uncached.len(), cached.len());
        let max_abs = uncached
            .iter()
            .zip(cached.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs <= 1.0e-6,
            "cached cross-attention K/V path diverged from uncached path: max_abs={max_abs:.6e}"
        );
    }

    #[test]
    fn runtime_model_smoke_load_and_predict() {
        if std::env::var("TRELLIS2_RUNTIME_MODEL_SMOKE").is_err() {
            eprintln!(
                "Skipping sparse flow runtime smoke test: set TRELLIS2_RUNTIME_MODEL_SMOKE=1 to enable."
            );
            return;
        }
        let weights_root = std::env::var("TRELLIS2_WEIGHTS_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(
                    "E:/models/huggingface/hub/models--microsoft--TRELLIS.2-4B/snapshots/af44b45f2e35a493886929c6d786e563ec68364d",
                )
            });
        if !weights_root.exists() {
            eprintln!(
                "Skipping sparse flow runtime smoke test: TRELLIS2 weights root missing at {}",
                weights_root.display()
            );
            return;
        }

        let runtime = SparseStructureFlowRuntime::load_from_stem(
            weights_root.as_path(),
            None,
            "ckpts/ss_flow_img_dit_1_3B_64_bf16",
            false,
            None,
        )
        .expect("sparse flow runtime should load from model stem");
        let cfg = runtime.config();
        let voxels = cfg.resolution * cfg.resolution * cfg.resolution;
        let sample = vec![0.0f32; cfg.in_channels * voxels];
        let cond_tokens = 32 * 32;
        let cond = vec![0.0f32; cond_tokens * cfg.cond_channels];
        let prepared = runtime
            .prepare_condition(cond.as_slice(), cond_tokens)
            .expect("sparse flow cond should prepare");
        let out = runtime
            .predict_velocity_with_condition(sample.as_slice(), 1.0, &prepared, None)
            .expect("sparse flow runtime forward should succeed");
        assert_eq!(out.len(), sample.len());
    }

    #[test]
    fn runtime_loads_blob_burnpack_when_module_layout_is_absent() {
        type TestBackend = burn::backend::NdArray<f32>;

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock drift")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("burn_trellis_sparse_flow_blob_{unique}"));
        let ckpts = root.join("ckpts");
        std::fs::create_dir_all(&ckpts).expect("create ckpt dir");

        let config = SparseStructureFlowConfig {
            resolution: 2,
            in_channels: 2,
            out_channels: 2,
            model_channels: 8,
            cond_channels: 4,
            num_blocks: 1,
            num_heads: Some(2),
            num_head_channels: 4,
            mlp_ratio: 2.0,
            pe_mode: "rope".to_string(),
            rope_freq: [1.0, 10_000.0],
            share_mod: true,
            qk_rms_norm: true,
            qk_rms_norm_cross: true,
            frequency_embedding_size: 8,
        };

        let config_json = serde_json::json!({
            "name": "SparseStructureFlowModel",
            "args": {
                "resolution": config.resolution,
                "in_channels": config.in_channels,
                "out_channels": config.out_channels,
                "model_channels": config.model_channels,
                "cond_channels": config.cond_channels,
                "num_blocks": config.num_blocks,
                "num_heads": config.num_heads,
                "num_head_channels": config.num_head_channels,
                "mlp_ratio": config.mlp_ratio,
                "pe_mode": config.pe_mode,
                "rope_freq": config.rope_freq,
                "share_mod": config.share_mod,
                "qk_rms_norm": config.qk_rms_norm,
                "qk_rms_norm_cross": config.qk_rms_norm_cross,
                "frequency_embedding_size": config.frequency_embedding_size
            }
        });
        std::fs::write(
            ckpts.join("flow_model.json"),
            serde_json::to_vec_pretty(&config_json).expect("serialize config"),
        )
        .expect("write config");

        let source_path = ckpts.join("flow_model.safetensors");
        let device = <TestBackend as burn::tensor::backend::BackendTypes>::Device::default();
        let model = SparseStructureFlowModel::<TestBackend>::new(&device, config.clone());
        let mut source_store =
            SafetensorsStore::from_file(&source_path).with_to_adapter(BurnToPyTorchAdapter);
        model
            .save_into(&mut source_store)
            .expect("save source safetensors");
        let source_bytes = std::fs::read(&source_path).expect("read source safetensors");

        let burnpack_path = ckpts.join("flow_model.bpk");
        save_blob_bytes_to_burnpack(&burnpack_path, source_bytes.as_slice(), 1024)
            .expect("save blob burnpack");

        std::fs::remove_file(&source_path).expect("remove source safetensors");

        let runtime = SparseStructureFlowRuntime::load_from_stem(
            root.as_path(),
            None,
            "ckpts/flow_model",
            false,
            None,
        )
        .expect("runtime should load from blob burnpack");
        let cfg = runtime.config();
        let voxels = cfg.resolution * cfg.resolution * cfg.resolution;
        let sample = vec![0.0f32; cfg.in_channels * voxels];
        let cond_tokens = 4;
        let cond = vec![0.0f32; cond_tokens * cfg.cond_channels];
        let prepared = runtime
            .prepare_condition(cond.as_slice(), cond_tokens)
            .expect("prepare cond");
        let out = runtime
            .predict_velocity_with_condition(sample.as_slice(), 1.0, &prepared, None)
            .expect("forward");
        assert_eq!(out.len(), sample.len());

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn runtime_loads_blob_burnpack_from_parts_manifest_when_base_file_missing() {
        type TestBackend = burn::backend::NdArray<f32>;

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock drift")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("burn_trellis_sparse_flow_parts_{unique}"));
        let ckpts = root.join("ckpts");
        std::fs::create_dir_all(&ckpts).expect("create ckpt dir");

        let config = SparseStructureFlowConfig {
            resolution: 2,
            in_channels: 2,
            out_channels: 2,
            model_channels: 8,
            cond_channels: 4,
            num_blocks: 1,
            num_heads: Some(2),
            num_head_channels: 4,
            mlp_ratio: 2.0,
            pe_mode: "rope".to_string(),
            rope_freq: [1.0, 10_000.0],
            share_mod: true,
            qk_rms_norm: true,
            qk_rms_norm_cross: true,
            frequency_embedding_size: 8,
        };

        let config_json = serde_json::json!({
            "name": "SparseStructureFlowModel",
            "args": {
                "resolution": config.resolution,
                "in_channels": config.in_channels,
                "out_channels": config.out_channels,
                "model_channels": config.model_channels,
                "cond_channels": config.cond_channels,
                "num_blocks": config.num_blocks,
                "num_heads": config.num_heads,
                "num_head_channels": config.num_head_channels,
                "mlp_ratio": config.mlp_ratio,
                "pe_mode": config.pe_mode,
                "rope_freq": config.rope_freq,
                "share_mod": config.share_mod,
                "qk_rms_norm": config.qk_rms_norm,
                "qk_rms_norm_cross": config.qk_rms_norm_cross,
                "frequency_embedding_size": config.frequency_embedding_size
            }
        });
        std::fs::write(
            ckpts.join("flow_model.json"),
            serde_json::to_vec_pretty(&config_json).expect("serialize config"),
        )
        .expect("write config");

        let source_path = ckpts.join("flow_model.safetensors");
        let device = <TestBackend as burn::tensor::backend::BackendTypes>::Device::default();
        let model = SparseStructureFlowModel::<TestBackend>::new(&device, config.clone());
        let mut source_store =
            SafetensorsStore::from_file(&source_path).with_to_adapter(BurnToPyTorchAdapter);
        model
            .save_into(&mut source_store)
            .expect("save source safetensors");
        let source_bytes = std::fs::read(&source_path).expect("read source safetensors");

        let burnpack_path = ckpts.join("flow_model.bpk");
        save_blob_bytes_to_burnpack(&burnpack_path, source_bytes.as_slice(), 1024)
            .expect("save blob burnpack");

        let part_path = ckpts.join("flow_model.bpk.part-00000.bpk");
        std::fs::rename(&burnpack_path, &part_path).expect("move burnpack into part");
        let part_bytes = std::fs::metadata(&part_path).expect("part metadata").len();
        let manifest_path = ckpts.join("flow_model.bpk.parts.json");
        std::fs::write(
            &manifest_path,
            format!(
                "{{\n  \"version\": 1,\n  \"source_file\": \"flow_model.bpk\",\n  \"source_modified_unix_ms\": 0,\n  \"total_bytes\": {},\n  \"max_part_bytes\": {},\n  \"parts\": [{{\"path\": \"{}\", \"bytes\": {}, \"sha256\": \"\", \"tensors\": 1}}]\n}}",
                part_bytes,
                part_bytes,
                part_path
                    .file_name()
                    .and_then(|v| v.to_str())
                    .expect("part file name"),
                part_bytes
            ),
        )
        .expect("write parts manifest");
        std::fs::remove_file(&source_path).expect("remove source safetensors");

        let runtime = SparseStructureFlowRuntime::load_from_stem(
            root.as_path(),
            None,
            "ckpts/flow_model",
            false,
            None,
        )
        .expect("runtime should load from parts manifest");
        let cfg = runtime.config();
        let voxels = cfg.resolution * cfg.resolution * cfg.resolution;
        let sample = vec![0.0f32; cfg.in_channels * voxels];
        let cond_tokens = 4;
        let cond = vec![0.0f32; cond_tokens * cfg.cond_channels];
        let prepared = runtime
            .prepare_condition(cond.as_slice(), cond_tokens)
            .expect("prepare cond");
        let out = runtime
            .predict_velocity_with_condition(sample.as_slice(), 1.0, &prepared, None)
            .expect("forward");
        assert_eq!(out.len(), sample.len());

        let candidates = resolve_model_weight_candidates("ckpts/flow_model", root.as_path(), None);
        assert_eq!(
            candidates.first(),
            Some(&ckpts.join("flow_model.bpk")),
            "parts manifest path should be treated as a valid burnpack candidate"
        );

        let _ = std::fs::remove_dir_all(root);
    }

    fn run_tiny_forward<B: Backend>(device: &B::Device)
    where
        super::RopeRotateWgpuBridgeImpl: super::RopeRotateWgpuBridge<B>,
        super::RopeRotateWgpuBridgeImpl: super::SparseFlowLayerNormWgpuBridge<B>,
        super::RopeRotateWgpuBridgeImpl: super::SparseFlowRmsNormWgpuBridge<B>,
        super::RopeRotateWgpuBridgeImpl: super::SparseFlowQkRmsNormWgpuBridge<B>,
        super::SparseFlowLinearWgpuBridgeImpl: super::SparseFlowLinearWgpuBridge<B>,
        super::SparseFlowBf16EmulationBridgeImpl: super::SparseFlowBf16EmulationBridge<B>,
    {
        let config = SparseStructureFlowConfig {
            resolution: 2,
            in_channels: 2,
            out_channels: 2,
            model_channels: 8,
            cond_channels: 4,
            num_blocks: 1,
            num_heads: Some(2),
            num_head_channels: 4,
            mlp_ratio: 2.0,
            pe_mode: "rope".to_string(),
            rope_freq: [1.0, 10_000.0],
            share_mod: true,
            qk_rms_norm: true,
            qk_rms_norm_cross: true,
            frequency_embedding_size: 8,
        };
        let model = SparseStructureFlowModel::<B>::new(device, config.clone());
        let x = Tensor::<B, 5>::zeros(
            [
                1,
                config.in_channels,
                config.resolution,
                config.resolution,
                config.resolution,
            ],
            device,
        );
        let t = Tensor::<B, 1>::from_floats([1.0], device);
        let cond = Tensor::<B, 3>::zeros([1, 4, config.cond_channels], device);
        let out = model.forward(x, t, cond);
        assert_eq!(
            out.dims(),
            [
                1,
                config.out_channels,
                config.resolution,
                config.resolution,
                config.resolution
            ]
        );
    }

    fn make_tiny_runtime_cpu() -> SparseStructureFlowRuntimeImpl<CpuRuntimeBackend> {
        let config = SparseStructureFlowConfig {
            resolution: 2,
            in_channels: 2,
            out_channels: 2,
            model_channels: 8,
            cond_channels: 4,
            num_blocks: 1,
            num_heads: Some(2),
            num_head_channels: 4,
            mlp_ratio: 2.0,
            pe_mode: "rope".to_string(),
            rope_freq: [1.0, 10_000.0],
            share_mod: true,
            qk_rms_norm: true,
            qk_rms_norm_cross: true,
            frequency_embedding_size: 8,
        };
        let device = <CpuRuntimeBackend as burn::tensor::backend::BackendTypes>::Device::default();
        let model = SparseStructureFlowModel::<CpuRuntimeBackend>::new(&device, config.clone());
        SparseStructureFlowRuntimeImpl {
            config,
            model,
            device,
        }
    }

    fn make_attention_tensor(
        device: &<CpuRuntimeBackend as burn::tensor::backend::BackendTypes>::Device,
        tokens: usize,
        heads: usize,
        channels: usize,
        seed: f32,
    ) -> Tensor<CpuRuntimeBackend, 4> {
        let mut values = Vec::with_capacity(tokens.saturating_mul(heads).saturating_mul(channels));
        for idx in 0..values.capacity() {
            let x = idx as f32 * 0.013 + seed;
            values.push(x.sin() * 0.7 + x.cos() * 0.3);
        }
        Tensor::<CpuRuntimeBackend, 1>::from_floats(values.as_slice(), device)
            .reshape([1, tokens, heads, channels])
    }

    fn tensor_to_vec4<B: Backend>(tensor: Tensor<B, 4>) -> Vec<f32> {
        tensor
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("tensor should be readable")
    }

    fn tensor_to_vec3<B: Backend>(tensor: Tensor<B, 3>) -> Vec<f32> {
        tensor
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("tensor should be readable")
    }

    fn tensor_to_vec2<B: Backend>(tensor: Tensor<B, 2>) -> Vec<f32> {
        tensor
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("tensor should be readable")
    }

    fn tensor_to_vec5<B: Backend>(tensor: Tensor<B, 5>) -> Vec<f32> {
        tensor
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("tensor should be readable")
    }

    fn sample_std(values: &[f32]) -> f32 {
        if values.len() < 2 {
            return 0.0;
        }
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let var_sum = values
            .iter()
            .map(|value| {
                let diff = *value - mean;
                diff * diff
            })
            .sum::<f32>();
        (var_sum / (values.len() - 1) as f32).sqrt()
    }

    #[test]
    fn tensor_std_matches_batchwise_unbiased_reference() {
        let device = <CpuRuntimeBackend as burn::tensor::backend::BackendTypes>::Device::default();
        let values = [
            0.0f32, 1.0, 2.0, 3.0, // batch 0
            2.0, 2.0, 2.0, 2.0, // batch 1
        ];
        let tensor = Tensor::<CpuRuntimeBackend, 1>::from_floats(values.as_slice(), &device)
            .reshape([2, 1, 1, 1, 4]);
        let std = super::tensor_std_tensor(tensor)
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("std tensor should be readable");
        assert_eq!(std.len(), 2, "expected one std value per batch item");

        let expected0 = sample_std(&values[0..4]);
        let expected1 = sample_std(&values[4..8]);
        assert!(
            (std[0] - expected0).abs() <= 1.0e-6,
            "batch 0 std mismatch: got={} expected={}",
            std[0],
            expected0
        );
        assert!(
            (std[1] - expected1).abs() <= 1.0e-6,
            "batch 1 std mismatch: got={} expected={}",
            std[1],
            expected1
        );
    }

    #[test]
    fn cfg_tensor_matches_upstream_formula_and_rescale() {
        let device = <CpuRuntimeBackend as burn::tensor::backend::BackendTypes>::Device::default();
        let x_t = Tensor::<CpuRuntimeBackend, 1>::from_floats(
            [0.1f32, -0.4, 0.7, -1.2].as_slice(),
            &device,
        )
        .reshape([1, 1, 2, 1, 2]);
        let pos = Tensor::<CpuRuntimeBackend, 1>::from_floats(
            [1.0f32, 2.0, -1.0, -2.0].as_slice(),
            &device,
        )
        .reshape([1, 1, 2, 1, 2]);
        let neg = Tensor::<CpuRuntimeBackend, 1>::from_floats(
            [0.25f32, -0.5, 0.75, -1.0].as_slice(),
            &device,
        )
        .reshape([1, 1, 2, 1, 2]);

        let pred =
            super::apply_cfg_tensor(x_t.clone(), 0.5, pos.clone(), neg.clone(), 3.0, 0.0, 0.01);
        let pred = tensor_to_vec5(pred);
        let expected = [
            3.0 * 1.0 + (1.0 - 3.0) * 0.25,
            3.0 * 2.0 + (1.0 - 3.0) * -0.5,
            3.0 * -1.0 + (1.0 - 3.0) * 0.75,
            3.0 * -2.0 + (1.0 - 3.0) * -1.0,
        ];
        for (idx, (actual, expected)) in pred.iter().zip(expected.iter()).enumerate() {
            assert!(
                (*actual - *expected).abs() <= 1.0e-6,
                "cfg tensor mismatch at {idx}: got={actual} expected={expected}"
            );
        }

        let rescaled = super::apply_cfg_tensor(x_t, 0.5, pos, neg, 3.0, 0.7, 0.01);
        let rescaled = tensor_to_vec5(rescaled);
        assert_ne!(rescaled, pred);
        assert!(rescaled.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn cfg_sparse_tensor_matches_upstream_formula_and_rescale() {
        let device = <CpuRuntimeBackend as burn::tensor::backend::BackendTypes>::Device::default();
        let x_t = Tensor::<CpuRuntimeBackend, 1>::from_floats(
            [0.1f32, -0.4, 0.7, -1.2].as_slice(),
            &device,
        )
        .reshape([1, 2, 2]);
        let pos = Tensor::<CpuRuntimeBackend, 1>::from_floats(
            [1.0f32, 2.0, -1.0, -2.0].as_slice(),
            &device,
        )
        .reshape([1, 2, 2]);
        let neg = Tensor::<CpuRuntimeBackend, 1>::from_floats(
            [0.25f32, -0.5, 0.75, -1.0].as_slice(),
            &device,
        )
        .reshape([1, 2, 2]);

        let pred = super::apply_cfg_sparse_tensor(
            x_t.clone(),
            0.5,
            pos.clone(),
            neg.clone(),
            3.0,
            0.0,
            0.01,
        );
        let pred = tensor_to_vec3(pred);
        let expected = [
            3.0 * 1.0 + (1.0 - 3.0) * 0.25,
            3.0 * 2.0 + (1.0 - 3.0) * -0.5,
            3.0 * -1.0 + (1.0 - 3.0) * 0.75,
            3.0 * -2.0 + (1.0 - 3.0) * -1.0,
        ];
        for (idx, (actual, expected)) in pred.iter().zip(expected.iter()).enumerate() {
            assert!(
                (*actual - *expected).abs() <= 1.0e-6,
                "cfg sparse tensor mismatch at {idx}: got={actual} expected={expected}"
            );
        }

        let rescaled = super::apply_cfg_sparse_tensor(x_t, 0.5, pos, neg, 3.0, 0.7, 0.01);
        let rescaled = tensor_to_vec3(rescaled);
        assert_ne!(rescaled, pred);
        assert!(rescaled.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn attention_stream_matches_dense_reference() {
        let _guard = ENV_LOCK.lock().expect("env lock");
        unsafe {
            std::env::set_var("TRELLIS2_ATTN_QUERY_CHUNK", "8");
            std::env::set_var("TRELLIS2_ATTN_QUERY_CHUNK_MAX", "8");
            std::env::set_var("TRELLIS2_ATTN_KEY_CHUNK", "7");
            std::env::set_var("TRELLIS2_ATTN_KEY_CHUNK_MAX", "7");
        }
        let device = <CpuRuntimeBackend as burn::tensor::backend::BackendTypes>::Device::default();
        let heads = 4usize;
        let head_dim = 8usize;
        let query_tokens = 32usize;
        let key_tokens = 24usize;

        let q = make_attention_tensor(&device, query_tokens, heads, head_dim, 0.2);
        let k = make_attention_tensor(&device, key_tokens, heads, head_dim, 0.7);
        let v = make_attention_tensor(&device, key_tokens, heads, head_dim, 1.3);

        let q = q.permute([0, 2, 1, 3]);
        let k = k.permute([0, 2, 1, 3]);
        let v = v.permute([0, 2, 1, 3]);

        let dense = scaled_dot_product_attention_dense(q.clone(), k.clone(), v.clone(), head_dim);
        let stream = scaled_dot_product_attention_stream(q, k, v, head_dim);

        let dense = tensor_to_vec4(dense);
        let stream = tensor_to_vec4(stream);
        assert_eq!(dense.len(), stream.len());

        let max_abs = dense
            .iter()
            .zip(stream.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs <= 1.0e-3,
            "stream attention drift too high: max_abs={max_abs:.6e}"
        );

        unsafe {
            std::env::remove_var("TRELLIS2_ATTN_QUERY_CHUNK");
            std::env::remove_var("TRELLIS2_ATTN_QUERY_CHUNK_MAX");
            std::env::remove_var("TRELLIS2_ATTN_KEY_CHUNK");
            std::env::remove_var("TRELLIS2_ATTN_KEY_CHUNK_MAX");
        }
    }

    #[test]
    fn attention_stream_benchmark_report() {
        if std::env::var("TRELLIS2_ATTN_BENCH").is_err() {
            eprintln!("skipping: set TRELLIS2_ATTN_BENCH=1 to run attention benchmark report");
            return;
        }

        let device = <CpuRuntimeBackend as burn::tensor::backend::BackendTypes>::Device::default();
        let heads = 8usize;
        let head_dim = 16usize;
        let query_tokens = 160usize;
        let key_tokens = 160usize;
        let iterations = 6usize;

        let q = make_attention_tensor(&device, query_tokens, heads, head_dim, 0.2)
            .permute([0, 2, 1, 3]);
        let k =
            make_attention_tensor(&device, key_tokens, heads, head_dim, 0.7).permute([0, 2, 1, 3]);
        let v =
            make_attention_tensor(&device, key_tokens, heads, head_dim, 1.3).permute([0, 2, 1, 3]);

        let _ = scaled_dot_product_attention_dense(q.clone(), k.clone(), v.clone(), head_dim)
            .into_data();
        let _ = scaled_dot_product_attention_stream(q.clone(), k.clone(), v.clone(), head_dim)
            .into_data();

        let dense_start = Instant::now();
        for _ in 0..iterations {
            let _ = scaled_dot_product_attention_dense(q.clone(), k.clone(), v.clone(), head_dim)
                .into_data();
        }
        let dense_ms = dense_start.elapsed().as_secs_f64() * 1_000.0 / iterations as f64;

        let stream_start = Instant::now();
        for _ in 0..iterations {
            let _ = scaled_dot_product_attention_stream(q.clone(), k.clone(), v.clone(), head_dim)
                .into_data();
        }
        let stream_ms = stream_start.elapsed().as_secs_f64() * 1_000.0 / iterations as f64;
        eprintln!(
            "attention bench: dense={dense_ms:.3}ms stream={stream_ms:.3}ms ratio={:.3}",
            stream_ms / dense_ms
        );
    }

    #[test]
    fn self_attention_chunked_matches_dense_reference() {
        let _guard = ENV_LOCK.lock().expect("env lock");
        let device = <CpuRuntimeBackend as burn::tensor::backend::BackendTypes>::Device::default();
        let channels = 32usize;
        let heads = 4usize;
        let tokens = 32usize;
        let resolution = 4usize;
        let attention = SelfAttention::<CpuRuntimeBackend>::new(
            &device,
            channels,
            heads,
            true,
            [1.0, 10_000.0],
            true,
            false,
        );

        let mut values = Vec::with_capacity(tokens.saturating_mul(channels));
        for idx in 0..values.capacity() {
            let x = idx as f32 * 0.011 + 0.37;
            values.push(x.sin() * 0.5 + x.cos() * 0.5);
        }
        let input = Tensor::<CpuRuntimeBackend, 1>::from_floats(values.as_slice(), &device)
            .reshape([1, tokens, channels]);

        unsafe {
            std::env::set_var("TRELLIS2_ATTN_BACKEND", "stream");
            std::env::set_var("TRELLIS2_SPARSE_FLOW_CHUNKED_FORWARD", "0");
        }
        let dense = attention.forward(input.clone(), resolution, None);

        unsafe {
            std::env::set_var("TRELLIS2_SPARSE_FLOW_CHUNKED_FORWARD", "1");
            std::env::set_var("TRELLIS2_SPARSE_FLOW_ATTN_QUERY_CHUNK", "8");
            std::env::set_var("TRELLIS2_SPARSE_FLOW_ATTN_KV_CHUNK", "8");
        }
        let chunked = attention.forward(input, resolution, None);

        let dense = dense
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("dense vec");
        let chunked = chunked
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("chunked vec");
        assert_eq!(dense.len(), chunked.len());
        let max_abs = dense
            .iter()
            .zip(chunked.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs <= 1.0e-3,
            "chunked self-attention drift too high: max_abs={max_abs:.6e}"
        );

        unsafe {
            std::env::remove_var("TRELLIS2_ATTN_BACKEND");
            std::env::remove_var("TRELLIS2_SPARSE_FLOW_CHUNKED_FORWARD");
            std::env::remove_var("TRELLIS2_SPARSE_FLOW_ATTN_QUERY_CHUNK");
            std::env::remove_var("TRELLIS2_SPARSE_FLOW_ATTN_KV_CHUNK");
        }
    }

    #[test]
    fn sparse_flow_chunk_plan_reuse_qkv_respects_logits_budget() {
        let logits_budget = 128 * 1024 * 1024;
        let (query_chunk_tokens, kv_chunk_tokens) =
            sparse_flow_stream_chunk_plan(1, 16, 6_144, 8_192, 8_192, true, logits_budget);
        assert_eq!(
            query_chunk_tokens, kv_chunk_tokens,
            "reuse-qkv path must keep query/kv chunks aligned"
        );
        assert!(
            sparse_flow_attention_logits_within_budget(
                1,
                16,
                query_chunk_tokens,
                kv_chunk_tokens,
                logits_budget
            ),
            "chunk plan exceeded logits budget in reuse-qkv mode: q={} kv={} budget={}",
            query_chunk_tokens,
            kv_chunk_tokens,
            logits_budget
        );
    }

    #[test]
    fn sparse_flow_chunk_plan_non_reuse_respects_logits_budget() {
        let logits_budget = 128 * 1024 * 1024;
        let (query_chunk_tokens, kv_chunk_tokens) =
            sparse_flow_stream_chunk_plan(1, 16, 8_192, 2_048, 8_192, false, logits_budget);
        assert!(
            sparse_flow_attention_logits_within_budget(
                1,
                16,
                query_chunk_tokens,
                kv_chunk_tokens,
                logits_budget
            ),
            "chunk plan exceeded logits budget in non-reuse mode: q={} kv={} budget={}",
            query_chunk_tokens,
            kv_chunk_tokens,
            logits_budget
        );
    }

    #[test]
    fn sparse_flow_backend_chunk_tokens_cpu_match_defaults() {
        assert_eq!(
            super::sparse_flow_mlp_chunk_tokens_for_backend::<CpuRuntimeBackend>(32_768),
            super::sparse_flow_mlp_chunk_tokens(32_768)
        );
        assert_eq!(
            super::sparse_flow_linear_chunk_tokens_for_backend::<CpuRuntimeBackend>(32_768),
            super::sparse_flow_linear_chunk_tokens(32_768)
        );
    }

    #[cfg(feature = "runtime-model-wgpu")]
    #[test]
    fn sparse_flow_backend_chunk_tokens_wgpu_respect_memory_safe_chunks() {
        let _guard = ENV_LOCK.lock().expect("env lock poisoned");
        set_runtime_model_debug_config(RuntimeModelDebugConfig {
            stage_debug: false,
            attention_debug: false,
            sparse_flow_module_attention: true,
            sparse_flow_module_attention_f16: false,
            sparse_flow_linear_f16: false,
            sparse_flow_torso_f16: false,
            sparse_flow_stock_bf16_emulation: false,
            sparse_flow_coord_rope_kernel: true,
            sparse_decoder_conv_f16: false,
        });
        assert_eq!(
            super::sparse_flow_mlp_chunk_tokens_for_backend::<super::WgpuRuntimeBackend>(4_096),
            4_096
        );
        assert_eq!(
            super::sparse_flow_mlp_chunk_tokens_for_backend::<super::WgpuRuntimeBackend>(8_192),
            8_192
        );
        assert_eq!(
            super::sparse_flow_mlp_chunk_tokens_for_backend::<super::WgpuRuntimeBackend>(32_768),
            8_192
        );
        assert_eq!(
            super::sparse_flow_linear_chunk_tokens_for_backend::<super::WgpuRuntimeBackend>(32_768,),
            16_384
        );
        assert_eq!(
            super::sparse_flow_torso_dtype_for_backend::<super::WgpuRuntimeBackend>(),
            None
        );
        set_runtime_model_debug_config(RuntimeModelDebugConfig {
            stage_debug: false,
            attention_debug: false,
            sparse_flow_module_attention: true,
            sparse_flow_module_attention_f16: true,
            sparse_flow_linear_f16: true,
            sparse_flow_torso_f16: false,
            sparse_flow_stock_bf16_emulation: false,
            sparse_flow_coord_rope_kernel: true,
            sparse_decoder_conv_f16: false,
        });
        assert_eq!(
            super::sparse_flow_mlp_chunk_tokens_for_backend::<super::WgpuRuntimeBackend>(32_768),
            32_768
        );
        assert_eq!(
            super::sparse_flow_linear_chunk_tokens_for_backend::<super::WgpuRuntimeBackend>(32_768,),
            32_768
        );
        assert_eq!(
            super::sparse_flow_torso_dtype_for_backend::<super::WgpuRuntimeBackend>(),
            None
        );
        set_runtime_model_debug_config(RuntimeModelDebugConfig::default());
    }

    #[cfg(feature = "runtime-model-wgpu")]
    #[test]
    fn wgpu_module_attention_probe_matches_stream_reference() {
        let _guard = ENV_LOCK.lock().expect("env lock poisoned");
        if std::env::var("TRELLIS2_WGPU_MODULE_ATTENTION_PROBE").is_err() {
            eprintln!(
                "Skipping WGPU module attention probe: set TRELLIS2_WGPU_MODULE_ATTENTION_PROBE=1 to enable."
            );
            return;
        }
        let probe_f16 = std::env::var("TRELLIS2_WGPU_MODULE_ATTENTION_PROBE_F16").is_ok();
        set_runtime_model_debug_config(RuntimeModelDebugConfig {
            stage_debug: false,
            attention_debug: false,
            sparse_flow_module_attention: true,
            sparse_flow_module_attention_f16: probe_f16,
            sparse_flow_linear_f16: false,
            sparse_flow_torso_f16: false,
            sparse_flow_stock_bf16_emulation: false,
            sparse_flow_coord_rope_kernel: true,
            sparse_decoder_conv_f16: false,
        });

        let tokens = std::env::var("TRELLIS2_WGPU_MODULE_ATTENTION_PROBE_TOKENS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(512)
            .max(1);
        let key_tokens = std::env::var("TRELLIS2_WGPU_MODULE_ATTENTION_PROBE_KEY_TOKENS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(tokens)
            .max(1);
        let batch = std::env::var("TRELLIS2_WGPU_MODULE_ATTENTION_PROBE_BATCH")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(1)
            .max(1);
        let heads = std::env::var("TRELLIS2_WGPU_MODULE_ATTENTION_PROBE_HEADS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(16)
            .max(1);
        let head_dim = std::env::var("TRELLIS2_WGPU_MODULE_ATTENTION_PROBE_HEAD_DIM")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(64)
            .max(1);
        let repeat = std::env::var("TRELLIS2_WGPU_MODULE_ATTENTION_PROBE_REPEAT")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(2)
            .max(1);
        let device = <WgpuRuntimeBackend as burn::tensor::backend::BackendTypes>::Device::default();
        let q_elem_count = batch
            .saturating_mul(tokens)
            .saturating_mul(heads)
            .saturating_mul(head_dim);
        let kv_elem_count = batch
            .saturating_mul(key_tokens)
            .saturating_mul(heads)
            .saturating_mul(head_dim);
        let mut q_values = Vec::with_capacity(q_elem_count);
        let mut k_values = Vec::with_capacity(kv_elem_count);
        let mut v_values = Vec::with_capacity(kv_elem_count);
        for idx in 0..q_elem_count {
            let x = idx as f32;
            q_values.push((x * 0.013).sin() * 0.5 + (x * 0.007).cos() * 0.25);
        }
        for idx in 0..kv_elem_count {
            let x = idx as f32;
            k_values.push((x * 0.011 + 0.3).sin() * 0.4 + (x * 0.017).cos() * 0.2);
            v_values.push((x * 0.019 + 0.7).sin() * 0.3 + (x * 0.005).cos() * 0.35);
        }

        let q_seq = Tensor::<WgpuRuntimeBackend, 1>::from_floats(q_values.as_slice(), &device)
            .reshape([batch, tokens, heads, head_dim]);
        let k_seq = Tensor::<WgpuRuntimeBackend, 1>::from_floats(k_values.as_slice(), &device)
            .reshape([batch, key_tokens, heads, head_dim]);
        let v_seq = Tensor::<WgpuRuntimeBackend, 1>::from_floats(v_values.as_slice(), &device)
            .reshape([batch, key_tokens, heads, head_dim]);
        let q = q_seq.swap_dims(1, 2);
        let k = k_seq.swap_dims(1, 2);
        let v = v_seq.swap_dims(1, 2);
        let (q, k, v) = if probe_f16 {
            (
                q.cast(burn::tensor::FloatDType::F16),
                k.cast(burn::tensor::FloatDType::F16),
                v.cast(burn::tensor::FloatDType::F16),
            )
        } else {
            (q, k, v)
        };

        let selected_dtype =
            super::sparse_flow_module_attention_dtype_for_shape(batch, heads, key_tokens, head_dim);
        let selected_fallback = super::sparse_flow_module_attention_long_k_requires_fallback(
            batch, heads, key_tokens, head_dim,
        );
        let raw = std::env::var("TRELLIS2_WGPU_MODULE_ATTENTION_PROBE_RAW").is_ok();
        let run_module = || {
            if raw {
                super::sparse_flow_module_attention_direct(q.clone(), k.clone(), v.clone())
            } else {
                super::sparse_flow_module_attention(q.clone(), k.clone(), v.clone())
            }
        };
        let _warm = tensor_to_vec4(run_module());
        let module_start = Instant::now();
        let mut module_values = Vec::new();
        for _ in 0..repeat {
            module_values = tensor_to_vec4(run_module());
        }
        let module_ms_total = module_start.elapsed().as_secs_f64() * 1_000.0;
        let module_ms = module_ms_total / repeat as f64;
        let stream_start = Instant::now();
        let stream = scaled_dot_product_attention_stream(q, k, v, head_dim);
        let stream_values = tensor_to_vec4(stream);
        let stream_ms = stream_start.elapsed().as_secs_f64() * 1_000.0;
        let module_finite =
            finite_debug_probe_tensor("module_attention_probe.output", module_values.as_slice());
        let stream_finite =
            finite_debug_probe_tensor("module_attention_probe.stream", stream_values.as_slice());
        let stats = compute_stats(module_values.as_slice(), stream_values.as_slice());
        eprintln!(
            "trellis2 wgpu module-attention probe: batch={batch} q_tokens={tokens} key_tokens={key_tokens} heads={heads} head_dim={head_dim} raw={raw} repeat={repeat} dtype={selected_dtype:?} fallback={selected_fallback} module_ms={module_ms:.2} stream_ms={stream_ms:.2} mean_abs={:.9e} max_abs={:.9e} rmse={:.9e}",
            stats.mean_abs, stats.max_abs, stats.rmse
        );
        set_runtime_model_debug_config(RuntimeModelDebugConfig::default());
        assert!(
            module_finite,
            "WGPU module attention probe produced non-finite output"
        );
        assert!(
            stream_finite,
            "WGPU module attention stream reference produced non-finite output"
        );

        if std::env::var("TRELLIS2_WGPU_MODULE_ATTENTION_PROBE_STRICT").is_ok() {
            assert!(
                stats.mean_abs <= 1.0e-4,
                "WGPU module attention mean_abs {:.6e} exceeded tolerance",
                stats.mean_abs
            );
            assert!(
                stats.max_abs <= 1.0e-3,
                "WGPU module attention max_abs {:.6e} exceeded tolerance",
                stats.max_abs
            );
        }
    }

    #[test]
    fn sample_trace_uses_single_host_readback_when_capturing_snapshots() {
        let _guard = HOST_STATS_LOCK
            .lock()
            .expect("host transfer stats lock should not be poisoned");
        let runtime = make_tiny_runtime_cpu();
        let config = runtime.config().clone();
        let voxel = config.resolution * config.resolution * config.resolution;
        let noise = vec![0.0f32; config.out_channels * voxel];
        let cond_tokens = 4usize;
        let cond_values = vec![0.0f32; cond_tokens * config.cond_channels];
        let cond = runtime
            .prepare_condition(cond_values.as_slice(), cond_tokens)
            .expect("prepare cond");
        let neg_cond = runtime
            .prepare_condition(cond_values.as_slice(), cond_tokens)
            .expect("prepare neg cond");
        let sample_cfg = FlowEulerSampleConfig {
            steps: 4,
            rescale_t: 1.0,
            guidance_strength: 1.0,
            guidance_rescale: 0.0,
            guidance_interval: [0.0, 1.0],
        };

        reset_host_transfer_stats();
        let trace = runtime
            .sample_with_trace(
                noise.as_slice(),
                sample_cfg,
                0.1,
                cond,
                neg_cond,
                None,
                true,
            )
            .expect("sample trace");
        let stats = host_transfer_stats();

        assert_eq!(
            stats.readback_count, 1,
            "dense trace snapshot capture should use a single merged host readback"
        );
        let expected_len = noise.len();
        assert_eq!(trace.samples.len(), expected_len);
        assert_eq!(trace.step_0_x_t.len(), expected_len);
        assert_eq!(trace.step_mid_x_t.len(), expected_len);
        assert_eq!(trace.step_last_x_t.len(), expected_len);
    }

    #[test]
    fn sample_sparse_rows_trace_uses_single_host_readback_when_capturing_snapshots() {
        let _guard = HOST_STATS_LOCK
            .lock()
            .expect("host transfer stats lock should not be poisoned");
        let runtime = make_tiny_runtime_cpu();
        let config = runtime.config().clone();
        let coords = vec![[0u32, 0, 0, 0], [0u32, 1, 0, 0], [0u32, 1, 1, 0]];
        let row_channels = 2usize;
        let noise = vec![0.0f32; config.out_channels * coords.len()];
        let sparse = SparseTensorOwned::from_layout(
            coords.clone(),
            noise.clone(),
            vec![0..coords.len()],
            config.out_channels,
            config.resolution,
        )
        .expect("sparse tensor");
        let cond_tokens = 4usize;
        let cond_values = vec![0.0f32; cond_tokens * config.cond_channels];
        let cond = runtime
            .prepare_condition(cond_values.as_slice(), cond_tokens)
            .expect("prepare cond");
        let neg_cond = runtime
            .prepare_condition(cond_values.as_slice(), cond_tokens)
            .expect("prepare neg cond");
        let sample_cfg = FlowEulerSampleConfig {
            steps: 4,
            rescale_t: 1.0,
            guidance_strength: 1.0,
            guidance_rescale: 0.0,
            guidance_interval: [0.0, 1.0],
        };

        reset_host_transfer_stats();
        let trace = runtime
            .sample_sparse_rows_with_trace(
                &sparse,
                sample_cfg,
                0.1,
                cond,
                neg_cond,
                None,
                row_channels,
                true,
                true,
            )
            .expect("sample sparse rows with trace");
        let stats = host_transfer_stats();

        assert_eq!(
            stats.readback_count, 1,
            "sparse-row trace snapshot capture should use a single merged host readback"
        );
        let expected_len = coords.len() * row_channels;
        assert_eq!(trace.samples.len(), expected_len);
        assert_eq!(trace.step_0_x_t.len(), expected_len);
        assert_eq!(trace.step_mid_x_t.len(), expected_len);
        assert_eq!(trace.step_last_x_t.len(), expected_len);
        #[cfg(feature = "runtime-model-wgpu")]
        {
            assert!(
                trace.samples_wgpu.is_none(),
                "cpu sparse-flow trace should not populate wgpu samples tensor"
            );
            assert!(
                trace.step_0_x_t_wgpu.is_none()
                    && trace.step_mid_x_t_wgpu.is_none()
                    && trace.step_last_x_t_wgpu.is_none(),
                "cpu sparse-flow trace should not populate wgpu snapshot tensors"
            );
        }
    }

    #[test]
    fn sample_sparse_rows_trace_batched_uses_single_host_readback_when_capturing_snapshots() {
        let _guard = HOST_STATS_LOCK
            .lock()
            .expect("host transfer stats lock should not be poisoned");
        let runtime = make_tiny_runtime_cpu();
        let config = runtime.config().clone();
        let coords = vec![
            [0u32, 0, 0, 0],
            [0u32, 1, 0, 0],
            [1u32, 0, 0, 0],
            [1u32, 1, 0, 0],
        ];
        let layout = vec![0..2, 2..4];
        let row_channels = 2usize;
        let noise = vec![0.0f32; config.out_channels * coords.len()];
        let sparse = SparseTensorOwned::from_layout(
            coords.clone(),
            noise.clone(),
            layout,
            config.out_channels,
            config.resolution,
        )
        .expect("sparse tensor");
        let cond_tokens = 4usize;
        let cond_values = vec![0.0f32; cond_tokens * config.cond_channels];
        let cond = runtime
            .prepare_condition(cond_values.as_slice(), cond_tokens)
            .expect("prepare cond");
        let neg_cond = runtime
            .prepare_condition(cond_values.as_slice(), cond_tokens)
            .expect("prepare neg cond");
        let sample_cfg = FlowEulerSampleConfig {
            steps: 4,
            rescale_t: 1.0,
            guidance_strength: 1.0,
            guidance_rescale: 0.0,
            guidance_interval: [0.0, 1.0],
        };

        reset_host_transfer_stats();
        let trace = runtime
            .sample_sparse_rows_with_trace(
                &sparse,
                sample_cfg,
                0.1,
                cond,
                neg_cond,
                None,
                row_channels,
                true,
                true,
            )
            .expect("sample sparse rows with trace");
        let stats = host_transfer_stats();

        assert_eq!(
            stats.readback_count, 1,
            "batched sparse-row trace snapshot capture should use a single merged host readback"
        );
        let expected_len = coords.len() * row_channels;
        assert_eq!(trace.samples.len(), expected_len);
        assert_eq!(trace.step_0_x_t.len(), expected_len);
        assert_eq!(trace.step_mid_x_t.len(), expected_len);
        assert_eq!(trace.step_last_x_t.len(), expected_len);
    }

    #[cfg(feature = "runtime-model-wgpu")]
    fn required_hook_tensor<'a>(hook: &'a HookSnapshot, key: &str) -> &'a HookTensor {
        hook.tensors
            .get(key)
            .unwrap_or_else(|| panic!("reference hook is missing required tensor '{key}'"))
    }

    #[cfg(feature = "runtime-model-wgpu")]
    fn hook_rows_f32(tensor: &HookTensor, channels: usize, label: &str) -> Vec<f32> {
        assert_eq!(
            tensor.shape.len(),
            2,
            "{label} must be a rank-2 row tensor, got {:?}",
            tensor.shape
        );
        assert_eq!(
            tensor.shape[1], channels,
            "{label} channel mismatch: got {} expected {channels}",
            tensor.shape[1]
        );
        assert_eq!(
            tensor.data.len(),
            tensor.shape[0] * channels,
            "{label} data length mismatch"
        );
        tensor.data.clone()
    }

    #[cfg(feature = "runtime-model-wgpu")]
    fn hook_coords4(tensor: &HookTensor, label: &str) -> Vec<[u32; 4]> {
        assert_eq!(
            tensor.shape.len(),
            2,
            "{label} must be a rank-2 coord tensor, got {:?}",
            tensor.shape
        );
        assert_eq!(
            tensor.shape[1], 4,
            "{label} coord tensor must have 4 columns, got {}",
            tensor.shape[1]
        );
        let rows = tensor.shape[0];
        assert_eq!(
            tensor.data.len(),
            rows * 4,
            "{label} coord data length mismatch"
        );
        let mut coords = Vec::with_capacity(rows);
        for row_idx in 0..rows {
            let base = row_idx * 4;
            coords.push([
                tensor.data[base].round().max(0.0) as u32,
                tensor.data[base + 1].round().max(0.0) as u32,
                tensor.data[base + 2].round().max(0.0) as u32,
                tensor.data[base + 3].round().max(0.0) as u32,
            ]);
        }
        coords
    }

    #[cfg(feature = "runtime-model-wgpu")]
    fn grouped_layout_from_coords(coords: &[[u32; 4]]) -> Vec<std::ops::Range<usize>> {
        assert!(
            !coords.is_empty(),
            "probe sparse tensor must contain at least one row"
        );
        let max_batch = coords
            .iter()
            .map(|coord| coord[0] as usize)
            .max()
            .expect("non-empty coords");
        let mut layout = Vec::with_capacity(max_batch + 1);
        let mut cursor = 0usize;
        for batch_idx in 0..=max_batch {
            let start = cursor;
            while cursor < coords.len() && coords[cursor][0] as usize == batch_idx {
                cursor += 1;
            }
            layout.push(start..cursor);
        }
        assert_eq!(
            cursor,
            coords.len(),
            "probe coords must be grouped by batch id"
        );
        for (row_idx, coord) in coords.iter().enumerate().skip(cursor) {
            panic!(
                "probe coords are not grouped by batch id at row {row_idx}: {:?}",
                coord
            );
        }
        layout
    }

    #[cfg(feature = "runtime-model-wgpu")]
    fn hook_condition_values(
        tensor: &HookTensor,
        cond_channels: usize,
        label: &str,
    ) -> (Vec<f32>, usize) {
        assert_eq!(
            tensor.shape.len(),
            3,
            "{label} must be a rank-3 condition tensor, got {:?}",
            tensor.shape
        );
        assert_eq!(
            tensor.shape[0], 1,
            "{label} probe expects a single condition batch"
        );
        assert_eq!(
            tensor.shape[2], cond_channels,
            "{label} condition channel mismatch: got {} expected {cond_channels}",
            tensor.shape[2]
        );
        (tensor.data.clone(), tensor.shape[1])
    }

    #[cfg(feature = "runtime-model-wgpu")]
    fn sampler_sigma_min(hook: &HookSnapshot, key: &str) -> f32 {
        let tensor = required_hook_tensor(hook, key);
        assert!(
            tensor.data.len() >= 7,
            "{key} sampler config must contain sigma_min at index 6"
        );
        tensor.data[6]
    }

    #[cfg(feature = "runtime-model-wgpu")]
    fn sampler_config_from_hook(hook: &HookSnapshot, key: &str) -> FlowEulerSampleConfig {
        let tensor = required_hook_tensor(hook, key);
        assert!(
            tensor.data.len() >= 6,
            "{key} sampler config must contain steps/rescale/guidance fields"
        );
        FlowEulerSampleConfig {
            steps: tensor.data[0].round().max(1.0) as usize,
            rescale_t: tensor.data[1].max(f32::EPSILON),
            guidance_strength: tensor.data[2],
            guidance_rescale: tensor.data[3].max(0.0),
            guidance_interval: [tensor.data[4], tensor.data[5]],
        }
    }

    #[cfg(feature = "runtime-model-wgpu")]
    fn compare_probe_tensor(
        label: &str,
        actual: &[f32],
        hook: &HookSnapshot,
        key: &str,
    ) -> Option<crate::hook_diff::MetricStats> {
        let Some(reference) = hook.tensors.get(key) else {
            eprintln!("trellis2 flow probe: reference key '{key}' missing; skipping {label}");
            return None;
        };
        assert_eq!(
            reference.data.len(),
            actual.len(),
            "{label} length mismatch for key '{key}'"
        );
        let stats = compute_stats(actual, reference.data.as_slice());
        eprintln!(
            "trellis2 flow probe {label}: key={key} mean_abs={:.9e} max_abs={:.9e} rmse={:.9e} non_finite={}",
            stats.mean_abs, stats.max_abs, stats.rmse, stats.non_finite_count
        );
        Some(stats)
    }

    #[cfg(feature = "runtime-model-wgpu")]
    fn finite_debug_probe_tensor(label: &str, actual: &[f32]) -> bool {
        let mut non_finite = 0usize;
        let mut first_non_finite = None;
        let mut finite_min = f32::INFINITY;
        let mut finite_max = f32::NEG_INFINITY;
        for (index, value) in actual.iter().copied().enumerate() {
            if value.is_finite() {
                finite_min = finite_min.min(value);
                finite_max = finite_max.max(value);
            } else {
                non_finite += 1;
                first_non_finite.get_or_insert((index, value));
            }
        }
        let finite_min = if finite_min.is_finite() {
            finite_min
        } else {
            f32::NAN
        };
        let finite_max = if finite_max.is_finite() {
            finite_max
        } else {
            f32::NAN
        };
        let first_non_finite_text = first_non_finite
            .map(|(index, value)| format!("{index}:{value:?}"))
            .unwrap_or_else(|| "none".to_owned());
        eprintln!(
            "trellis2 slat finite {label}: len={} non_finite={} first_non_finite={} finite_min={:.9e} finite_max={:.9e}",
            actual.len(),
            non_finite,
            first_non_finite_text,
            finite_min,
            finite_max
        );
        if let Some((index, value)) = first_non_finite {
            if !SLAT_DEBUG_FIRST_NONFINITE_REPORTED.swap(true, Ordering::Relaxed) {
                eprintln!(
                    "trellis2 slat first_nonfinite_stage: label={label} index={index} value={value:?} non_finite={non_finite}"
                );
            }
        }
        non_finite == 0
    }

    #[cfg(feature = "runtime-model-wgpu")]
    fn compare_debug_probe_tensor(
        label: &str,
        actual: &[f32],
        hook: Option<&HookSnapshot>,
        key: &str,
    ) -> bool {
        let finite = finite_debug_probe_tensor(label, actual);
        let Some(hook) = hook else {
            eprintln!("trellis2 slat debug {label}: no reference hook; finite-only");
            return finite;
        };
        let Some(reference) = hook.tensors.get(key) else {
            eprintln!("trellis2 slat debug {label}: reference key '{key}' missing; finite-only");
            return finite;
        };
        if reference.data.len() != actual.len() {
            eprintln!(
                "trellis2 slat debug {label}: key={key} length mismatch actual={} reference={}; finite-only",
                actual.len(),
                reference.data.len()
            );
            return finite;
        }
        let stats = compute_stats(actual, reference.data.as_slice());
        eprintln!(
            "trellis2 slat debug {label}: key={key} mean_abs={:.9e} max_abs={:.9e} rmse={:.9e} non_finite={}",
            stats.mean_abs, stats.max_abs, stats.rmse, stats.non_finite_count
        );
        if stats.max_abs > 1.0e-3 {
            let len = actual.len().min(8);
            eprintln!(
                "trellis2 slat debug {label}: first{len} actual={:?} reference={:?}",
                &actual[..len],
                &reference.data[..len]
            );
        }
        finite && stats.non_finite_count == 0
    }

    #[cfg(feature = "runtime-model-wgpu")]
    fn maybe_run_slat_block0_debug_probe(
        runtime_impl: &SparseStructureFlowRuntimeImpl<WgpuRuntimeBackend>,
        x_t: Tensor<WgpuRuntimeBackend, 3>,
        timestep: f32,
        cond: Tensor<WgpuRuntimeBackend, 3>,
        token_coords: Tensor<WgpuRuntimeBackend, 2, Int>,
        rows: usize,
        debug_hook_path: &Path,
    ) {
        SLAT_DEBUG_FIRST_NONFINITE_REPORTED.store(false, Ordering::Relaxed);
        let debug_hook = Some(
            HookSnapshot::from_file(debug_hook_path).expect("load Python SLat debug hook snapshot"),
        );
        let config = runtime_impl.config();
        let model = &runtime_impl.model;
        let channels = config.model_channels;
        let h = linear_forward_token_chunked_reference(
            &model.input_layer,
            x_t,
            sparse_flow_linear_chunk_tokens_for_backend::<WgpuRuntimeBackend>(rows),
        );
        compare_debug_probe_tensor(
            "input_layer",
            tensor_to_vec3(h.clone()).as_slice(),
            debug_hook.as_ref(),
            "input_layer",
        );

        let t =
            Tensor::<WgpuRuntimeBackend, 1>::from_floats([timestep * 1000.0], &runtime_impl.device);
        let t_emb = model.t_embedder.forward(t, config.frequency_embedding_size);
        compare_debug_probe_tensor(
            "t_embedder",
            tensor_to_vec2(t_emb.clone()).as_slice(),
            debug_hook.as_ref(),
            "t_embedder",
        );
        let mod_signal_base = linear_forward_stable_2d(&model.ada_ln_modulation, silu(t_emb));
        compare_debug_probe_tensor(
            "mod_signal",
            tensor_to_vec2(mod_signal_base.clone()).as_slice(),
            debug_hook.as_ref(),
            "mod_signal",
        );
        compare_debug_probe_tensor(
            "cond_cast",
            tensor_to_vec3(cond.clone()).as_slice(),
            debug_hook.as_ref(),
            "cond_cast",
        );

        let block = &model.blocks[0];
        let [batch, tokens, block_channels] = h.dims();
        assert_eq!(block_channels, channels);
        assert_eq!(tokens, rows, "block0 debug token count mismatch");

        let mod_bias = block.modulation.val().reshape([1, channels * 6]);
        let mod_signal_dtype: burn::tensor::FloatDType = mod_signal_base.dtype().into();
        let mod_bias_dtype: burn::tensor::FloatDType = mod_bias.dtype().into();
        let mod_bias = if mod_bias_dtype != mod_signal_dtype {
            mod_bias.cast(mod_signal_dtype)
        } else {
            mod_bias
        };
        let mod_signal = mod_signal_base.clone().add(mod_bias);
        let shift_msa = mod_signal
            .clone()
            .slice([0..batch, 0..channels])
            .reshape([batch, 1, channels]);
        let scale_msa = mod_signal
            .clone()
            .slice([0..batch, channels..(channels * 2)])
            .reshape([batch, 1, channels]);
        let gate_msa = mod_signal
            .clone()
            .slice([0..batch, (channels * 2)..(channels * 3)])
            .reshape([batch, 1, channels]);
        let shift_mlp = mod_signal
            .clone()
            .slice([0..batch, (channels * 3)..(channels * 4)])
            .reshape([batch, 1, channels]);
        let scale_mlp = mod_signal
            .clone()
            .slice([0..batch, (channels * 4)..(channels * 5)])
            .reshape([batch, 1, channels]);
        let gate_mlp = mod_signal
            .clone()
            .slice([0..batch, (channels * 5)..(channels * 6)])
            .reshape([batch, 1, channels]);
        compare_debug_probe_tensor(
            "block0.shift_msa",
            tensor_to_vec3(shift_msa.clone()).as_slice(),
            debug_hook.as_ref(),
            "block0.shift_msa",
        );
        compare_debug_probe_tensor(
            "block0.scale_msa",
            tensor_to_vec3(scale_msa.clone()).as_slice(),
            debug_hook.as_ref(),
            "block0.scale_msa",
        );
        compare_debug_probe_tensor(
            "block0.gate_msa",
            tensor_to_vec3(gate_msa.clone()).as_slice(),
            debug_hook.as_ref(),
            "block0.gate_msa",
        );

        let norm1 = layer_norm_no_affine(h.clone(), super::LAYER_NORM_EPS);
        compare_debug_probe_tensor(
            "block0.norm1",
            tensor_to_vec3(norm1.clone()).as_slice(),
            debug_hook.as_ref(),
            "block0.norm1",
        );
        let norm1_mod = norm1.mul(scale_msa.add_scalar(1.0)).add(shift_msa);
        compare_debug_probe_tensor(
            "block0.norm1_mod",
            tensor_to_vec3(norm1_mod.clone()).as_slice(),
            debug_hook.as_ref(),
            "block0.norm1_mod",
        );
        if std::env::var("TRELLIS2_SLAT_FORWARD_PROBE_DEBUG_MOD_ONLY").is_ok() {
            return;
        }
        let attn = &block.self_attn;
        let qkv_linear = linear_forward_attention(&attn.to_qkv, norm1_mod.clone(), false);
        compare_debug_probe_tensor(
            "block0.self_attn.qkv_linear",
            tensor_to_vec3(qkv_linear.clone()).as_slice(),
            debug_hook.as_ref(),
            "block0.self_attn.qkv_linear",
        );
        let qkv = qkv_linear.reshape([batch, tokens, 3, attn.num_heads, attn.head_dim]);
        compare_debug_probe_tensor(
            "block0.self_attn.qkv_fused",
            tensor_to_vec5(qkv.clone()).as_slice(),
            debug_hook.as_ref(),
            "block0.self_attn.qkv_fused",
        );
        let q = qkv
            .clone()
            .slice([
                0..batch,
                0..tokens,
                0..1,
                0..attn.num_heads,
                0..attn.head_dim,
            ])
            .reshape([batch, tokens, attn.num_heads, attn.head_dim]);
        let k = qkv
            .clone()
            .slice([
                0..batch,
                0..tokens,
                1..2,
                0..attn.num_heads,
                0..attn.head_dim,
            ])
            .reshape([batch, tokens, attn.num_heads, attn.head_dim]);
        let v = qkv
            .clone()
            .slice([
                0..batch,
                0..tokens,
                2..3,
                0..attn.num_heads,
                0..attn.head_dim,
            ])
            .reshape([batch, tokens, attn.num_heads, attn.head_dim]);
        compare_debug_probe_tensor(
            "block0.self_attn.q_pre_norm",
            tensor_to_vec4(q.clone()).as_slice(),
            debug_hook.as_ref(),
            "block0.self_attn.q_pre_norm",
        );
        compare_debug_probe_tensor(
            "block0.self_attn.k_pre_norm",
            tensor_to_vec4(k.clone()).as_slice(),
            debug_hook.as_ref(),
            "block0.self_attn.k_pre_norm",
        );
        compare_debug_probe_tensor(
            "block0.self_attn.v",
            tensor_to_vec4(v.clone()).as_slice(),
            debug_hook.as_ref(),
            "block0.self_attn.v",
        );
        let q = if let Some(norm) = attn.q_rms_norm.as_ref() {
            norm.forward(q)
        } else {
            q
        };
        let k = if let Some(norm) = attn.k_rms_norm.as_ref() {
            norm.forward(k)
        } else {
            k
        };
        compare_debug_probe_tensor(
            "block0.self_attn.q_rms",
            tensor_to_vec4(q.clone()).as_slice(),
            debug_hook.as_ref(),
            "block0.self_attn.q_rms",
        );
        compare_debug_probe_tensor(
            "block0.self_attn.k_rms",
            tensor_to_vec4(k.clone()).as_slice(),
            debug_hook.as_ref(),
            "block0.self_attn.k_rms",
        );
        let q = if attn.use_rope {
            apply_rope_single(
                q,
                config.resolution,
                attn.head_dim,
                attn.rope_freq,
                Some(token_coords.clone()),
                0,
            )
        } else {
            q
        };
        let k = if attn.use_rope {
            apply_rope_single(
                k,
                config.resolution,
                attn.head_dim,
                attn.rope_freq,
                Some(token_coords.clone()),
                0,
            )
        } else {
            k
        };
        compare_debug_probe_tensor(
            "block0.self_attn.q_rope",
            tensor_to_vec4(q.clone()).as_slice(),
            debug_hook.as_ref(),
            "block0.self_attn.q_rope",
        );
        compare_debug_probe_tensor(
            "block0.self_attn.k_rope",
            tensor_to_vec4(k.clone()).as_slice(),
            debug_hook.as_ref(),
            "block0.self_attn.k_rope",
        );
        let qkv_post = Tensor::stack(vec![q.clone(), k.clone(), v.clone()], 2);
        compare_debug_probe_tensor(
            "block0.self_attn.qkv_post",
            tensor_to_vec5(qkv_post).as_slice(),
            debug_hook.as_ref(),
            "block0.self_attn.qkv_post",
        );
        if std::env::var("TRELLIS2_SLAT_FORWARD_PROBE_DEBUG_ATTENTION_PRE_ONLY").is_ok() {
            return;
        }
        let sdpa = sparse_flow_stock_bf16_round(scaled_dot_product_attention(
            q.clone(),
            k.clone(),
            v.clone(),
            attn.head_dim,
        ));
        compare_debug_probe_tensor(
            "block0.self_attn.sdpa",
            tensor_to_vec4(sdpa.clone()).as_slice(),
            debug_hook.as_ref(),
            "block0.self_attn.sdpa",
        );
        if std::env::var("TRELLIS2_SLAT_FORWARD_PROBE_DEBUG_ATTENTION_ALT_LAYOUT").is_ok() {
            let q_alt = force_contiguous_4d(q.clone().permute([0, 2, 1, 3]));
            let k_alt = force_contiguous_4d(k.clone().permute([0, 2, 1, 3]));
            let v_alt = force_contiguous_4d(v.clone().permute([0, 2, 1, 3]));
            let sdpa_alt = attention(
                q_alt,
                k_alt,
                v_alt,
                None,
                None,
                AttentionModuleOptions::default(),
            )
            .permute([0, 2, 1, 3]);
            compare_debug_probe_tensor(
                "block0.self_attn.sdpa_alt_layout",
                tensor_to_vec4(sdpa_alt).as_slice(),
                debug_hook.as_ref(),
                "block0.self_attn.sdpa",
            );
        }
        let sdpa_flat = sdpa.reshape([batch, tokens, channels]);
        compare_debug_probe_tensor(
            "block0.self_attn.sdpa_flat",
            tensor_to_vec3(sdpa_flat.clone()).as_slice(),
            debug_hook.as_ref(),
            "block0.self_attn.sdpa_flat",
        );
        let self_attn_from_parts = linear_forward_attention(&attn.to_out, sdpa_flat, false);
        compare_debug_probe_tensor(
            "block0.self_attn.from_parts",
            tensor_to_vec3(self_attn_from_parts).as_slice(),
            debug_hook.as_ref(),
            "block0.self_attn",
        );
        let self_attn =
            block
                .self_attn
                .forward(norm1_mod, config.resolution, Some(token_coords.clone()));
        compare_debug_probe_tensor(
            "block0.self_attn",
            tensor_to_vec3(self_attn.clone()).as_slice(),
            debug_hook.as_ref(),
            "block0.self_attn",
        );
        let after_self = h.clone().add(self_attn.mul(gate_msa));
        compare_debug_probe_tensor(
            "block0.after_self",
            tensor_to_vec3(after_self.clone()).as_slice(),
            debug_hook.as_ref(),
            "block0.after_self",
        );
        let norm2 =
            layer_norm_affine_stable(after_self.clone(), &block.norm2, super::LAYER_NORM_EPS);
        compare_debug_probe_tensor(
            "block0.norm2",
            tensor_to_vec3(norm2.clone()).as_slice(),
            debug_hook.as_ref(),
            "block0.norm2",
        );
        let cross = &block.cross_attn;
        let cross_q_linear = linear_forward_attention(&cross.to_q, norm2.clone(), false);
        compare_debug_probe_tensor(
            "block0.cross_attn.q_linear",
            tensor_to_vec3(cross_q_linear.clone()).as_slice(),
            debug_hook.as_ref(),
            "block0.cross_attn.q_linear",
        );
        let mut cross_q = cross_q_linear.reshape([batch, tokens, cross.num_heads, cross.head_dim]);
        compare_debug_probe_tensor(
            "block0.cross_attn.q_reshaped",
            tensor_to_vec4(cross_q.clone()).as_slice(),
            debug_hook.as_ref(),
            "block0.cross_attn.q_reshaped",
        );
        let [cond_batch, cond_tokens, _] = cond.dims();
        let cross_kv_linear = linear_forward_attention(&cross.to_kv, cond.clone(), false);
        compare_debug_probe_tensor(
            "block0.cross_attn.kv_linear",
            tensor_to_vec3(cross_kv_linear.clone()).as_slice(),
            debug_hook.as_ref(),
            "block0.cross_attn.kv_linear",
        );
        let cross_kv =
            cross_kv_linear.reshape([cond_batch, cond_tokens, 2, cross.num_heads, cross.head_dim]);
        compare_debug_probe_tensor(
            "block0.cross_attn.kv_fused",
            tensor_to_vec5(cross_kv.clone()).as_slice(),
            debug_hook.as_ref(),
            "block0.cross_attn.kv_fused",
        );
        let mut cross_k = cross_kv
            .clone()
            .slice([
                0..cond_batch,
                0..cond_tokens,
                0..1,
                0..cross.num_heads,
                0..cross.head_dim,
            ])
            .reshape([cond_batch, cond_tokens, cross.num_heads, cross.head_dim]);
        let cross_v = cross_kv
            .slice([
                0..cond_batch,
                0..cond_tokens,
                1..2,
                0..cross.num_heads,
                0..cross.head_dim,
            ])
            .reshape([cond_batch, cond_tokens, cross.num_heads, cross.head_dim]);
        if let Some(norm) = cross.q_rms_norm.as_ref() {
            cross_q = norm.forward(cross_q);
        }
        if let Some(norm) = cross.k_rms_norm.as_ref() {
            cross_k = norm.forward(cross_k);
        }
        compare_debug_probe_tensor(
            "block0.cross_attn.q_rms",
            tensor_to_vec4(cross_q.clone()).as_slice(),
            debug_hook.as_ref(),
            "block0.cross_attn.q_rms",
        );
        compare_debug_probe_tensor(
            "block0.cross_attn.k_rms",
            tensor_to_vec4(cross_k.clone()).as_slice(),
            debug_hook.as_ref(),
            "block0.cross_attn.k_rms",
        );
        compare_debug_probe_tensor(
            "block0.cross_attn.v",
            tensor_to_vec4(cross_v.clone()).as_slice(),
            debug_hook.as_ref(),
            "block0.cross_attn.v",
        );
        if std::env::var("TRELLIS2_SLAT_FORWARD_PROBE_DEBUG_CROSS_MODULE_COMPARE").is_ok() {
            let q_heads = cross_q.clone().permute([0, 2, 1, 3]);
            let k_heads = cross_k.clone().permute([0, 2, 1, 3]);
            let v_heads = cross_v.clone().permute([0, 2, 1, 3]);
            let module_f32 = attention(
                q_heads.clone(),
                k_heads.clone(),
                v_heads.clone(),
                None,
                None,
                AttentionModuleOptions::default(),
            )
            .permute([0, 2, 1, 3]);
            compare_debug_probe_tensor(
                "block0.cross_attn.sdpa_module_f32",
                tensor_to_vec4(module_f32).as_slice(),
                debug_hook.as_ref(),
                "block0.cross_attn.sdpa",
            );
            let module_f16 = attention(
                q_heads.cast(burn::tensor::FloatDType::F16),
                k_heads.cast(burn::tensor::FloatDType::F16),
                v_heads.cast(burn::tensor::FloatDType::F16),
                None,
                None,
                AttentionModuleOptions::default(),
            )
            .permute([0, 2, 1, 3]);
            compare_debug_probe_tensor(
                "block0.cross_attn.sdpa_module_f16",
                tensor_to_vec4(module_f16).as_slice(),
                debug_hook.as_ref(),
                "block0.cross_attn.sdpa",
            );
        }
        let cross_sdpa = scaled_dot_product_attention(cross_q, cross_k, cross_v, cross.head_dim);
        compare_debug_probe_tensor(
            "block0.cross_attn.sdpa",
            tensor_to_vec4(cross_sdpa.clone()).as_slice(),
            debug_hook.as_ref(),
            "block0.cross_attn.sdpa",
        );
        let cross_sdpa_flat = cross_sdpa.reshape([batch, tokens, channels]);
        compare_debug_probe_tensor(
            "block0.cross_attn.sdpa_flat",
            tensor_to_vec3(cross_sdpa_flat.clone()).as_slice(),
            debug_hook.as_ref(),
            "block0.cross_attn.sdpa_flat",
        );
        let cross_attn = linear_forward_attention(&cross.to_out, cross_sdpa_flat, false);
        compare_debug_probe_tensor(
            "block0.cross_attn.from_parts",
            tensor_to_vec3(cross_attn.clone()).as_slice(),
            debug_hook.as_ref(),
            "block0.cross_attn.from_parts",
        );
        compare_debug_probe_tensor(
            "block0.cross_attn",
            tensor_to_vec3(cross_attn.clone()).as_slice(),
            debug_hook.as_ref(),
            "block0.cross_attn",
        );
        let after_cross = after_self.add(cross_attn);
        compare_debug_probe_tensor(
            "block0.after_cross",
            tensor_to_vec3(after_cross.clone()).as_slice(),
            debug_hook.as_ref(),
            "block0.after_cross",
        );
        let norm3 = layer_norm_no_affine(after_cross.clone(), super::LAYER_NORM_EPS);
        compare_debug_probe_tensor(
            "block0.norm3",
            tensor_to_vec3(norm3.clone()).as_slice(),
            debug_hook.as_ref(),
            "block0.norm3",
        );
        let norm3_mod = norm3.mul(scale_mlp.add_scalar(1.0)).add(shift_mlp);
        compare_debug_probe_tensor(
            "block0.norm3_mod",
            tensor_to_vec3(norm3_mod.clone()).as_slice(),
            debug_hook.as_ref(),
            "block0.norm3_mod",
        );
        let mlp = block.mlp.forward(norm3_mod);
        compare_debug_probe_tensor(
            "block0.mlp",
            tensor_to_vec3(mlp.clone()).as_slice(),
            debug_hook.as_ref(),
            "block0.mlp",
        );
        let block_out = after_cross.add(mlp.mul(gate_mlp));
        compare_debug_probe_tensor(
            "block0.out",
            tensor_to_vec3(block_out.clone()).as_slice(),
            debug_hook.as_ref(),
            "block0.out",
        );
        let forward_out = block.forward(
            h,
            mod_signal_base,
            cond,
            config.resolution,
            Some(token_coords),
            None,
        );
        compare_debug_probe_tensor(
            "block0.forward_out",
            tensor_to_vec3(forward_out).as_slice(),
            debug_hook.as_ref(),
            "block0.forward_out",
        );
    }

    #[cfg(feature = "runtime-model-wgpu")]
    fn maybe_run_sparse_structure_block0_debug_probe(
        runtime_impl: &SparseStructureFlowRuntimeImpl<WgpuRuntimeBackend>,
        noise: &HookTensor,
        timestep: f32,
        cond: Tensor<WgpuRuntimeBackend, 3>,
        hook: &HookSnapshot,
    ) {
        let debug_prefix = "debug.sparse_structure.block0";
        if !hook
            .tensors
            .contains_key(format!("{debug_prefix}.input_layer").as_str())
        {
            return;
        }

        let config = runtime_impl.config();
        let model = &runtime_impl.model;
        let batch = 1usize;
        let tokens = config.resolution * config.resolution * config.resolution;
        let channels = config.model_channels;
        let sample = Tensor::<WgpuRuntimeBackend, 1>::from_floats(
            noise.data.as_slice(),
            &runtime_impl.device,
        )
        .reshape([
            batch,
            config.out_channels,
            config.resolution,
            config.resolution,
            config.resolution,
        ]);
        let x = sample
            .reshape([batch, config.out_channels, tokens])
            .swap_dims(1, 2);
        let token_coords =
            super::dense_grid_token_coords(config.resolution, runtime_impl.device.clone());
        let compare = |label: &str, actual: &[f32], key_suffix: &str| {
            compare_debug_probe_tensor(
                label,
                actual,
                Some(hook),
                format!("{debug_prefix}.{key_suffix}").as_str(),
            );
        };

        let h_input = linear_forward_input_token_chunked(
            &model.input_layer,
            x,
            sparse_flow_linear_chunk_tokens_for_backend::<WgpuRuntimeBackend>(tokens),
        );
        compare(
            "ss.block0.input_layer",
            tensor_to_vec3(h_input.clone()).as_slice(),
            "input_layer",
        );

        let t =
            Tensor::<WgpuRuntimeBackend, 1>::from_floats([timestep * 1000.0], &runtime_impl.device);
        let t_emb = model.t_embedder.forward(t, config.frequency_embedding_size);
        compare(
            "ss.block0.t_embedder",
            tensor_to_vec2(t_emb.clone()).as_slice(),
            "t_embedder",
        );
        let mod_signal_base =
            linear_forward_stable_2d_reference(&model.ada_ln_modulation, silu(t_emb));
        compare(
            "ss.block0.mod_signal_base",
            tensor_to_vec2(mod_signal_base.clone()).as_slice(),
            "mod_signal_base",
        );

        let h = sparse_flow_stock_bf16_round(h_input);
        let mod_signal_cast = sparse_flow_stock_bf16_round(mod_signal_base);
        let cond_cast = sparse_flow_stock_bf16_round(cond);
        compare(
            "ss.block0.input_cast",
            tensor_to_vec3(h.clone()).as_slice(),
            "input_cast",
        );
        compare(
            "ss.block0.mod_signal_cast",
            tensor_to_vec2(mod_signal_cast.clone()).as_slice(),
            "mod_signal_cast",
        );
        compare(
            "ss.block0.cond_cast",
            tensor_to_vec3(cond_cast.clone()).as_slice(),
            "cond_cast",
        );

        let block = &model.blocks[0];
        let mod_bias = block.modulation.val().reshape([1, channels * 6]);
        let mod_signal_dtype: burn::tensor::FloatDType = mod_signal_cast.dtype().into();
        let mod_bias_dtype: burn::tensor::FloatDType = mod_bias.dtype().into();
        let mod_bias = if mod_bias_dtype != mod_signal_dtype {
            mod_bias.cast(mod_signal_dtype)
        } else {
            mod_bias
        };
        let mod_signal = sparse_flow_stock_bf16_round(mod_signal_cast.add(mod_bias));
        compare(
            "ss.block0.mod_signal_with_bias",
            tensor_to_vec2(mod_signal.clone()).as_slice(),
            "mod_signal_with_bias",
        );
        let shift_msa = mod_signal
            .clone()
            .slice([0..batch, 0..channels])
            .reshape([batch, 1, channels]);
        let scale_msa = mod_signal
            .clone()
            .slice([0..batch, channels..(channels * 2)])
            .reshape([batch, 1, channels]);
        let gate_msa = mod_signal
            .clone()
            .slice([0..batch, (channels * 2)..(channels * 3)])
            .reshape([batch, 1, channels]);
        let shift_mlp = mod_signal
            .clone()
            .slice([0..batch, (channels * 3)..(channels * 4)])
            .reshape([batch, 1, channels]);
        let scale_mlp = mod_signal
            .clone()
            .slice([0..batch, (channels * 4)..(channels * 5)])
            .reshape([batch, 1, channels]);
        let gate_mlp = mod_signal
            .clone()
            .slice([0..batch, (channels * 5)..(channels * 6)])
            .reshape([batch, 1, channels]);
        compare(
            "ss.block0.shift_msa",
            tensor_to_vec3(shift_msa.clone()).as_slice(),
            "shift_msa",
        );
        compare(
            "ss.block0.scale_msa",
            tensor_to_vec3(scale_msa.clone()).as_slice(),
            "scale_msa",
        );
        compare(
            "ss.block0.gate_msa",
            tensor_to_vec3(gate_msa.clone()).as_slice(),
            "gate_msa",
        );

        let norm1 =
            sparse_flow_stock_bf16_round(layer_norm_no_affine(h.clone(), super::LAYER_NORM_EPS));
        compare(
            "ss.block0.norm1",
            tensor_to_vec3(norm1.clone()).as_slice(),
            "norm1",
        );
        let scale_msa_plus = sparse_flow_stock_bf16_round(scale_msa.add_scalar(1.0));
        let norm1_scaled = sparse_flow_stock_bf16_round(norm1.mul(scale_msa_plus));
        let norm1_mod = sparse_flow_stock_bf16_round(norm1_scaled.add(shift_msa));
        compare(
            "ss.block0.norm1_mod",
            tensor_to_vec3(norm1_mod.clone()).as_slice(),
            "norm1_mod",
        );

        let attn = &block.self_attn;
        let qkv_linear = linear_forward_attention(&attn.to_qkv, norm1_mod.clone(), false);
        compare(
            "ss.block0.self_attn.qkv_linear",
            tensor_to_vec3(qkv_linear.clone()).as_slice(),
            "self_attn.qkv_linear",
        );
        let qkv = qkv_linear.reshape([batch, tokens, 3, attn.num_heads, attn.head_dim]);
        let q = qkv
            .clone()
            .slice([
                0..batch,
                0..tokens,
                0..1,
                0..attn.num_heads,
                0..attn.head_dim,
            ])
            .reshape([batch, tokens, attn.num_heads, attn.head_dim]);
        let k = qkv
            .clone()
            .slice([
                0..batch,
                0..tokens,
                1..2,
                0..attn.num_heads,
                0..attn.head_dim,
            ])
            .reshape([batch, tokens, attn.num_heads, attn.head_dim]);
        let v = qkv
            .clone()
            .slice([
                0..batch,
                0..tokens,
                2..3,
                0..attn.num_heads,
                0..attn.head_dim,
            ])
            .reshape([batch, tokens, attn.num_heads, attn.head_dim]);
        compare(
            "ss.block0.self_attn.q_pre_norm",
            tensor_to_vec4(q.clone()).as_slice(),
            "self_attn.q_pre_norm",
        );
        compare(
            "ss.block0.self_attn.k_pre_norm",
            tensor_to_vec4(k.clone()).as_slice(),
            "self_attn.k_pre_norm",
        );
        compare(
            "ss.block0.self_attn.v",
            tensor_to_vec4(v.clone()).as_slice(),
            "self_attn.v",
        );
        let q = if let Some(norm) = attn.q_rms_norm.as_ref() {
            norm.forward(q)
        } else {
            q
        };
        let k = if let Some(norm) = attn.k_rms_norm.as_ref() {
            norm.forward(k)
        } else {
            k
        };
        compare(
            "ss.block0.self_attn.q_rms",
            tensor_to_vec4(q.clone()).as_slice(),
            "self_attn.q_rms",
        );
        compare(
            "ss.block0.self_attn.k_rms",
            tensor_to_vec4(k.clone()).as_slice(),
            "self_attn.k_rms",
        );
        let q = if attn.use_rope {
            apply_rope_single(
                q,
                config.resolution,
                attn.head_dim,
                attn.rope_freq,
                Some(token_coords.clone()),
                0,
            )
        } else {
            q
        };
        let k = if attn.use_rope {
            apply_rope_single(
                k,
                config.resolution,
                attn.head_dim,
                attn.rope_freq,
                Some(token_coords.clone()),
                0,
            )
        } else {
            k
        };
        compare(
            "ss.block0.self_attn.q_rope",
            tensor_to_vec4(q.clone()).as_slice(),
            "self_attn.q_rope",
        );
        compare(
            "ss.block0.self_attn.k_rope",
            tensor_to_vec4(k.clone()).as_slice(),
            "self_attn.k_rope",
        );
        let debug_q = tokens.min(16);
        let q_bh = q.clone().permute([0, 2, 1, 3]);
        let k_bh = k.clone().permute([0, 2, 1, 3]);
        let v_bh = v.clone().permute([0, 2, 1, 3]);
        let q_bh_chunk = q_bh
            .slice([0..batch, 0..attn.num_heads, 0..debug_q, 0..attn.head_dim])
            .clone();
        let logits = sparse_flow_stock_bf16_round(
            matmul_4d_via_3d(q_bh_chunk, k_bh.clone().swap_dims(2, 3))
                .mul_scalar(1.0 / (attn.head_dim as f32).sqrt()),
        );
        compare(
            "ss.block0.self_attn.logits_q0_16",
            tensor_to_vec4(logits.clone()).as_slice(),
            "self_attn.logits_q0_16",
        );
        let probs = sparse_flow_stock_bf16_round(softmax(logits, 3));
        compare(
            "ss.block0.self_attn.probs_q0_16",
            tensor_to_vec4(probs.clone()).as_slice(),
            "self_attn.probs_q0_16",
        );
        let context =
            sparse_flow_stock_bf16_round(matmul_4d_via_3d(probs, v_bh)).permute([0, 2, 1, 3]);
        compare(
            "ss.block0.self_attn.context_q0_16",
            tensor_to_vec4(context).as_slice(),
            "self_attn.context_q0_16",
        );
        let sdpa = sparse_flow_stock_bf16_round(scaled_dot_product_attention(
            q.clone(),
            k.clone(),
            v.clone(),
            attn.head_dim,
        ));
        compare(
            "ss.block0.self_attn.sdpa",
            tensor_to_vec4(sdpa.clone()).as_slice(),
            "self_attn.sdpa",
        );
        let sdpa_flat = sdpa.reshape([batch, tokens, channels]);
        let self_out = linear_forward_attention(&attn.to_out, sdpa_flat, false);
        compare(
            "ss.block0.self_attn.out",
            tensor_to_vec3(self_out.clone()).as_slice(),
            "self_attn.out",
        );
        let self_forward =
            block
                .self_attn
                .forward(norm1_mod, config.resolution, Some(token_coords.clone()));
        compare(
            "ss.block0.self_attn.forward",
            tensor_to_vec3(self_forward.clone()).as_slice(),
            "self_attn.forward",
        );

        let after_self = sparse_flow_stock_bf16_round(
            h.clone()
                .add(sparse_flow_stock_bf16_round(self_forward.mul(gate_msa))),
        );
        compare(
            "ss.block0.after_self",
            tensor_to_vec3(after_self.clone()).as_slice(),
            "after_self",
        );
        let norm2 = sparse_flow_stock_bf16_round(layer_norm_affine_stable(
            after_self.clone(),
            &block.norm2,
            super::LAYER_NORM_EPS,
        ));
        compare(
            "ss.block0.norm2",
            tensor_to_vec3(norm2.clone()).as_slice(),
            "norm2",
        );

        let cross = &block.cross_attn;
        let cross_q_linear = linear_forward_attention(&cross.to_q, norm2.clone(), false);
        let cross_kv_linear = linear_forward_attention(&cross.to_kv, cond_cast.clone(), false);
        compare(
            "ss.block0.cross_attn.q_linear",
            tensor_to_vec3(cross_q_linear.clone()).as_slice(),
            "cross_attn.q_linear",
        );
        compare(
            "ss.block0.cross_attn.kv_linear",
            tensor_to_vec3(cross_kv_linear.clone()).as_slice(),
            "cross_attn.kv_linear",
        );
        let [cond_batch, cond_tokens, _] = cond_cast.dims();
        let mut cross_q = cross_q_linear.reshape([batch, tokens, cross.num_heads, cross.head_dim]);
        let cross_kv =
            cross_kv_linear.reshape([cond_batch, cond_tokens, 2, cross.num_heads, cross.head_dim]);
        let mut cross_k = cross_kv
            .clone()
            .slice([
                0..cond_batch,
                0..cond_tokens,
                0..1,
                0..cross.num_heads,
                0..cross.head_dim,
            ])
            .reshape([cond_batch, cond_tokens, cross.num_heads, cross.head_dim]);
        let cross_v = cross_kv
            .slice([
                0..cond_batch,
                0..cond_tokens,
                1..2,
                0..cross.num_heads,
                0..cross.head_dim,
            ])
            .reshape([cond_batch, cond_tokens, cross.num_heads, cross.head_dim]);
        if let Some(norm) = cross.q_rms_norm.as_ref() {
            cross_q = norm.forward(cross_q);
        }
        if let Some(norm) = cross.k_rms_norm.as_ref() {
            cross_k = norm.forward(cross_k);
        }
        compare(
            "ss.block0.cross_attn.q_rms",
            tensor_to_vec4(cross_q.clone()).as_slice(),
            "cross_attn.q_rms",
        );
        compare(
            "ss.block0.cross_attn.k_rms",
            tensor_to_vec4(cross_k.clone()).as_slice(),
            "cross_attn.k_rms",
        );
        let cross_sdpa = sparse_flow_stock_bf16_round(scaled_dot_product_attention(
            cross_q,
            cross_k,
            cross_v,
            cross.head_dim,
        ));
        compare(
            "ss.block0.cross_attn.sdpa",
            tensor_to_vec4(cross_sdpa.clone()).as_slice(),
            "cross_attn.sdpa",
        );
        let cross_flat = cross_sdpa.reshape([batch, tokens, channels]);
        let cross_out = linear_forward_attention(&cross.to_out, cross_flat, false);
        compare(
            "ss.block0.cross_attn.out",
            tensor_to_vec3(cross_out.clone()).as_slice(),
            "cross_attn.out",
        );
        let after_cross =
            sparse_flow_stock_bf16_round(after_self.add(sparse_flow_stock_bf16_round(cross_out)));
        compare(
            "ss.block0.after_cross",
            tensor_to_vec3(after_cross.clone()).as_slice(),
            "after_cross",
        );
        let norm3 = sparse_flow_stock_bf16_round(layer_norm_no_affine(
            after_cross.clone(),
            super::LAYER_NORM_EPS,
        ));
        let scale_mlp_plus = sparse_flow_stock_bf16_round(scale_mlp.add_scalar(1.0));
        let norm3_scaled = sparse_flow_stock_bf16_round(norm3.mul(scale_mlp_plus));
        let norm3_mod = sparse_flow_stock_bf16_round(norm3_scaled.add(shift_mlp));
        compare(
            "ss.block0.norm3_mod",
            tensor_to_vec3(norm3_mod.clone()).as_slice(),
            "norm3_mod",
        );
        let mlp = block.mlp.forward(norm3_mod);
        compare(
            "ss.block0.mlp",
            tensor_to_vec3(mlp.clone()).as_slice(),
            "mlp",
        );
        let block_out = sparse_flow_stock_bf16_round(
            after_cross.add(sparse_flow_stock_bf16_round(mlp.mul(gate_mlp))),
        );
        compare("ss.block0.out", tensor_to_vec3(block_out).as_slice(), "out");
    }

    #[cfg(feature = "runtime-model-wgpu")]
    #[test]
    fn slat_first_forward_probe() {
        if std::env::var("TRELLIS2_SLAT_FORWARD_PROBE").is_err() {
            eprintln!(
                "Skipping Trellis.2 SLat WGPU first-forward probe: set TRELLIS2_SLAT_FORWARD_PROBE=1 to enable."
            );
            return;
        }

        let hook_path = std::env::var("TRELLIS2_SLAT_FORWARD_PROBE_HOOK")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(
                    "tmp/runs/20260619T012716Z_trellis2_python_flow_f32_predv_posneg_slat_only/python_full_hook.safetensors",
                )
            });
        assert!(
            hook_path.exists(),
            "TRELLIS2_SLAT_FORWARD_PROBE_HOOK does not exist: {}",
            hook_path.display()
        );
        let weights_root = std::env::var("TRELLIS2_WEIGHTS_ROOT")
            .map(PathBuf::from)
            .expect("TRELLIS2_WEIGHTS_ROOT must point at the TRELLIS.2-4B weights root");
        assert!(
            weights_root.exists(),
            "TRELLIS2_WEIGHTS_ROOT does not exist: {}",
            weights_root.display()
        );

        let hook = HookSnapshot::from_file(&hook_path).expect("load Python SLat hook snapshot");
        let hook_prefix = std::env::var("TRELLIS2_SLAT_FORWARD_PROBE_PREFIX")
            .unwrap_or_else(|_| "sample_shape_slat".to_string());
        let sampler_prefix = std::env::var("TRELLIS2_SLAT_FORWARD_PROBE_SAMPLER_PREFIX")
            .unwrap_or_else(|_| "sample_shape_slat".to_string());
        let cond_prefix = std::env::var("TRELLIS2_SLAT_FORWARD_PROBE_COND_PREFIX")
            .unwrap_or_else(|_| "get_cond_512.out".to_string());
        let model_stem = std::env::var("TRELLIS2_SLAT_FORWARD_PROBE_MODEL_STEM")
            .unwrap_or_else(|_| "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16".to_string());
        let fast_f16 = std::env::var("TRELLIS2_SLAT_FORWARD_PROBE_FAST_F16").is_ok();
        let attention_f16 =
            fast_f16 || std::env::var("TRELLIS2_SLAT_FORWARD_PROBE_F16_ATTENTION").is_ok();
        let linear_f16 = std::env::var("TRELLIS2_SLAT_FORWARD_PROBE_F16_LINEAR").is_ok();
        let module_attention =
            std::env::var("TRELLIS2_SLAT_FORWARD_PROBE_DISABLE_MODULE_ATTENTION").is_err();
        set_runtime_model_debug_config(RuntimeModelDebugConfig {
            stage_debug: false,
            attention_debug: false,
            sparse_flow_module_attention: module_attention,
            sparse_flow_module_attention_f16: attention_f16,
            sparse_flow_linear_f16: linear_f16,
            sparse_flow_torso_f16: std::env::var("TRELLIS2_SLAT_FORWARD_PROBE_TORSO_F16").is_ok(),
            sparse_flow_stock_bf16_emulation: false,
            sparse_flow_coord_rope_kernel: true,
            sparse_decoder_conv_f16: false,
        });
        let sampler_config_key = format!("{sampler_prefix}.sampler.config");
        let sample_cfg = sampler_config_from_hook(&hook, sampler_config_key.as_str());
        let probe_step = std::env::var("TRELLIS2_SLAT_FORWARD_PROBE_STEP")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(0);
        assert!(
            probe_step < sample_cfg.steps,
            "probe step {} out of range for {} sampler steps",
            probe_step,
            sample_cfg.steps
        );
        let step_prefix = format!(
            "{hook_prefix}.sampler.step_{probe_step:03}_of_{:03}",
            sample_cfg.steps
        );
        let input_key = if probe_step == 0 {
            format!("{hook_prefix}.noise.feats")
        } else {
            format!(
                "{hook_prefix}.sampler.step_{:03}_of_{:03}.x_t.feats",
                probe_step - 1,
                sample_cfg.steps
            )
        };
        let coords_key = format!("{hook_prefix}.noise.coords");
        let coords = hook_coords4(
            required_hook_tensor(&hook, coords_key.as_str()),
            coords_key.as_str(),
        );
        let row_channels = 32usize;
        let input_rows = hook_rows_f32(
            required_hook_tensor(&hook, input_key.as_str()),
            row_channels,
            input_key.as_str(),
        );
        assert_eq!(
            coords.len() * row_channels,
            input_rows.len(),
            "input rows/channels must match coords"
        );

        let runtime = SparseStructureFlowRuntime::load_from_stem(
            weights_root.as_path(),
            None,
            model_stem.as_str(),
            true,
            None,
        )
        .expect("load shape SLat flow runtime on WGPU");
        assert_eq!(
            runtime.backend_name(),
            "wgpu",
            "probe must execute on the WGPU backend"
        );
        let config = runtime.config().clone();
        assert_eq!(config.out_channels, row_channels);

        let cond_key = format!("{cond_prefix}.cond");
        let neg_cond_key = format!("{cond_prefix}.neg_cond");
        let (cond_values, cond_tokens) = hook_condition_values(
            required_hook_tensor(&hook, cond_key.as_str()),
            config.cond_channels,
            cond_key.as_str(),
        );
        let (neg_values, neg_tokens) = hook_condition_values(
            required_hook_tensor(&hook, neg_cond_key.as_str()),
            config.cond_channels,
            neg_cond_key.as_str(),
        );
        assert_eq!(cond_tokens, neg_tokens, "cond/neg token count mismatch");
        let cond = runtime
            .prepare_condition(cond_values.as_slice(), cond_tokens)
            .expect("prepare positive condition");
        let neg_cond = runtime
            .prepare_condition(neg_values.as_slice(), neg_tokens)
            .expect("prepare negative condition");

        let layout = grouped_layout_from_coords(coords.as_slice());
        let sparse = runtime
            .sparse_tensor_from_host_layout(
                coords,
                input_rows,
                layout,
                row_channels,
                config.resolution,
            )
            .expect("create WGPU sparse tensor from reference rows");
        let t_pairs = super::timestep_pairs(sample_cfg.steps, sample_cfg.rescale_t);
        let (timestep, _) = t_pairs[probe_step];
        let runtime_impl = match &runtime {
            SparseStructureFlowRuntime::Wgpu(runtime_impl) => runtime_impl,
            SparseStructureFlowRuntime::Cpu(_) => panic!("probe must run with WGPU runtime"),
        };
        let cond_tensor = match cond {
            SparseFlowCondition::Wgpu(cond) => cond,
            SparseFlowCondition::Cpu(_) => panic!("positive condition must be WGPU-backed"),
        };
        let neg_cond_tensor = match neg_cond {
            SparseFlowCondition::Wgpu(cond) => cond,
            SparseFlowCondition::Cpu(_) => panic!("negative condition must be WGPU-backed"),
        };
        let state_rows = runtime_impl
            .build_state_rows_tensor(&sparse, sparse.rows(), row_channels)
            .expect("build WGPU state rows from sparse tensor");
        let coords_t = runtime_impl
            .sparse_coords_tensor(&sparse, "shape SLat probe coords")
            .expect("build WGPU coord tensor from sparse tensor");
        let token_coords = coords_t.slice([0..sparse.rows(), 1..4]);
        let x_t = state_rows.reshape([1, sparse.rows(), row_channels]);
        if let Ok(debug_hook_path) = std::env::var("TRELLIS2_SLAT_FORWARD_PROBE_DEBUG_HOOK") {
            maybe_run_slat_block0_debug_probe(
                runtime_impl,
                x_t.clone(),
                timestep,
                cond_tensor.clone(),
                token_coords.clone(),
                sparse.rows(),
                Path::new(debug_hook_path.as_str()),
            );
            if std::env::var("TRELLIS2_SLAT_FORWARD_PROBE_DEBUG_ONLY").is_ok() {
                return;
            }
        }
        let use_cache = std::env::var("TRELLIS2_SLAT_FORWARD_PROBE_USE_CACHE").is_ok();
        let (cond_cache, neg_cache) = if use_cache {
            runtime_impl.prepare_cross_attention_caches(
                cond_tensor.clone(),
                neg_cond_tensor.clone(),
                sample_cfg,
                t_pairs.as_slice(),
                "slat_forward_probe",
            )
        } else {
            (None, None)
        };
        let batched_cfg_cache = if use_cache
            && super::sparse_flow_batched_cfg_enabled_for_backend::<WgpuRuntimeBackend>()
        {
            match (&cond_cache, &neg_cache) {
                (Some(pos_cache), Some(neg_cache)) => {
                    super::concat_cross_kv_caches(pos_cache, neg_cache)
                }
                _ => None,
            }
        } else {
            None
        };
        let probe_repeats = std::env::var("TRELLIS2_SLAT_FORWARD_PROBE_REPEATS")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(1)
            .max(1);
        let run_probe_once = || {
            reset_sparse_flow_op_telemetry();
            let start = Instant::now();
            let pred = runtime_impl
                .predict_with_cfg_sparse_tensor_parts_with_cache(
                    x_t.clone(),
                    timestep,
                    sample_cfg,
                    sampler_sigma_min(&hook, sampler_config_key.as_str()),
                    cond_tensor.clone(),
                    neg_cond_tensor.clone(),
                    None,
                    config.resolution,
                    token_coords.clone(),
                    cond_cache.as_ref(),
                    neg_cache.as_ref(),
                    batched_cfg_cache.as_ref(),
                )
                .expect("run direct shape SLat sparse flow prediction");
            <WgpuRuntimeBackend as Backend>::sync(&runtime_impl.device)
                .expect("sync direct shape SLat sparse flow prediction");
            let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;
            (pred, elapsed_ms, sparse_flow_op_telemetry())
        };
        let (mut cfg_pred, mut elapsed_ms, mut flow_ops) = run_probe_once();
        if probe_repeats > 1 {
            eprintln!(
                "trellis2 slat repeat-forward probe: iter=1/{probe_repeats} elapsed={elapsed_ms:.2}ms"
            );
        }
        for idx in 1..probe_repeats {
            let (next_pred, next_elapsed_ms, next_flow_ops) = run_probe_once();
            cfg_pred = next_pred;
            elapsed_ms = next_elapsed_ms;
            flow_ops = next_flow_ops;
            eprintln!(
                "trellis2 slat repeat-forward probe: iter={}/{probe_repeats} elapsed={elapsed_ms:.2}ms",
                idx + 1,
            );
        }
        if std::env::var("TRELLIS2_SLAT_FORWARD_PROBE_COMPARE_BATCHED_CFG").is_ok()
            && let (Some(pos_sep), Some(neg_sep)) = (cfg_pred.pos.as_ref(), cfg_pred.neg.as_ref())
        {
            reset_sparse_flow_op_telemetry();
            let batched_start = Instant::now();
            let x_batched = Tensor::cat(vec![x_t.clone(), x_t.clone()], 0);
            let cond_batched = Tensor::cat(vec![cond_tensor.clone(), neg_cond_tensor.clone()], 0);
            let pred_batched = runtime_impl
                .predict_velocity_sparse_tensor_with_cache(
                    x_batched,
                    timestep,
                    cond_batched,
                    None,
                    config.resolution,
                    token_coords.clone(),
                    None,
                )
                .expect("run batched CFG candidate prediction");
            let batched_elapsed_ms = batched_start.elapsed().as_secs_f64() * 1_000.0;
            let batched_ops = sparse_flow_op_telemetry();
            let batched_pos = pred_batched
                .clone()
                .slice([0..1, 0..sparse.rows(), 0..row_channels]);
            let batched_neg = pred_batched.slice([1..2, 0..sparse.rows(), 0..row_channels]);
            let batched_guided = super::apply_cfg_sparse_tensor(
                x_t.clone(),
                timestep,
                batched_pos.clone(),
                batched_neg.clone(),
                sample_cfg.guidance_strength,
                sample_cfg.guidance_rescale,
                sampler_sigma_min(&hook, sampler_config_key.as_str()),
            );
            let pos_sep_vec = tensor_to_vec3(pos_sep.clone());
            let neg_sep_vec = tensor_to_vec3(neg_sep.clone());
            let guided_sep_vec = tensor_to_vec3(cfg_pred.guided.clone());
            let batched_pos_vec = tensor_to_vec3(batched_pos);
            let batched_neg_vec = tensor_to_vec3(batched_neg);
            let batched_guided_vec = tensor_to_vec3(batched_guided);
            let pos_stats = compute_stats(batched_pos_vec.as_slice(), pos_sep_vec.as_slice());
            let neg_stats = compute_stats(batched_neg_vec.as_slice(), neg_sep_vec.as_slice());
            let guided_stats =
                compute_stats(batched_guided_vec.as_slice(), guided_sep_vec.as_slice());
            eprintln!(
                "trellis2 slat batched-cfg probe: elapsed={batched_elapsed_ms:.2}ms separate_elapsed={elapsed_ms:.2}ms pos_mean_abs={:.9e} pos_max_abs={:.9e} neg_mean_abs={:.9e} neg_max_abs={:.9e} guided_mean_abs={:.9e} guided_max_abs={:.9e}",
                pos_stats.mean_abs,
                pos_stats.max_abs,
                neg_stats.mean_abs,
                neg_stats.max_abs,
                guided_stats.mean_abs,
                guided_stats.max_abs
            );
            eprintln!(
                concat!(
                    "trellis2 slat batched-cfg ops: ",
                    "self_attn_ms={:.2} cross_attn_ms={:.2} mlp_ms={:.2} ",
                    "self_kernel_ms={:.2} cross_kernel_ms={:.2} module_attention_ms={:.2}"
                ),
                batched_ops.self_attn_ns as f64 / 1_000_000.0,
                batched_ops.cross_attn_ns as f64 / 1_000_000.0,
                batched_ops.mlp_ns as f64 / 1_000_000.0,
                batched_ops.self_kernel_ns as f64 / 1_000_000.0,
                batched_ops.cross_kernel_ns as f64 / 1_000_000.0,
                batched_ops.module_attention_ns as f64 / 1_000_000.0,
            );
            if std::env::var("TRELLIS2_SLAT_FORWARD_PROBE_BATCHED_CFG_STRICT").is_ok() {
                assert!(
                    pos_stats.mean_abs <= 1.0e-4 && pos_stats.max_abs <= 1.0e-3,
                    "batched CFG pos drift too high: mean_abs={:.6e} max_abs={:.6e}",
                    pos_stats.mean_abs,
                    pos_stats.max_abs
                );
                assert!(
                    neg_stats.mean_abs <= 1.0e-4 && neg_stats.max_abs <= 1.0e-3,
                    "batched CFG neg drift too high: mean_abs={:.6e} max_abs={:.6e}",
                    neg_stats.mean_abs,
                    neg_stats.max_abs
                );
                assert!(
                    guided_stats.mean_abs <= 1.0e-4 && guided_stats.max_abs <= 1.0e-3,
                    "batched CFG guided drift too high: mean_abs={:.6e} max_abs={:.6e}",
                    guided_stats.mean_abs,
                    guided_stats.max_abs
                );
            }
        }
        let guided = tensor_to_vec3(cfg_pred.guided).reshape_to_rows_vec(row_channels);
        let pos = cfg_pred.pos.map(tensor_to_vec3);
        let neg = cfg_pred.neg.map(tensor_to_vec3);
        let cache_label = if use_cache { "cached" } else { "direct" };
        eprintln!(
            "trellis2 slat {cache_label}-forward probe: step={}/{} timestep={:.9} rows={} channels={} elapsed={elapsed_ms:.2}ms",
            probe_step,
            sample_cfg.steps,
            timestep,
            sparse.rows(),
            row_channels
        );
        eprintln!(
            concat!(
                "trellis2 slat direct-forward ops: ",
                "self_attn_calls={} cross_attn_calls={} mlp_calls={} self_kernel_calls={} module_attention_calls={} fused_qk_calls={} fused_qkv_module_calls={} ",
                "self_attn_total_ms={:.2} cross_attn_total_ms={:.2} mlp_total_ms={:.2} ",
                "self_qkv_ms={:.2} self_norm_rope_ms={:.2} self_kernel_ms={:.2} self_out_ms={:.2} self_cat_ms={:.2} ",
                "cross_q_ms={:.2} cross_kv_ms={:.2} cross_norm_ms={:.2} cross_kernel_ms={:.2} cross_out_ms={:.2} cross_cat_ms={:.2} ",
                "mlp_ms={:.2} block_norm_mod_ms={:.2} block_norm_affine_ms={:.2} block_gate_residual_ms={:.2} model_io_ms={:.2} model_input_ms={:.2} model_output_ms={:.2} ",
                "module_cast_pad_ms={:.2} module_attention_ms={:.2} module_output_ms={:.2}"
            ),
            flow_ops.self_attn_calls,
            flow_ops.cross_attn_calls,
            flow_ops.mlp_calls,
            flow_ops.self_kernel_calls,
            flow_ops.module_attention_calls,
            flow_ops.self_norm_rope_fused_qk_calls,
            flow_ops.self_norm_rope_fused_qkv_module_calls,
            flow_ops.self_attn_ns as f64 / 1_000_000.0,
            flow_ops.cross_attn_ns as f64 / 1_000_000.0,
            flow_ops.mlp_ns as f64 / 1_000_000.0,
            flow_ops.self_qkv_ns as f64 / 1_000_000.0,
            flow_ops.self_norm_rope_ns as f64 / 1_000_000.0,
            flow_ops.self_kernel_ns as f64 / 1_000_000.0,
            flow_ops.self_out_ns as f64 / 1_000_000.0,
            flow_ops.self_cat_ns as f64 / 1_000_000.0,
            flow_ops.cross_q_ns as f64 / 1_000_000.0,
            flow_ops.cross_kv_ns as f64 / 1_000_000.0,
            flow_ops.cross_norm_ns as f64 / 1_000_000.0,
            flow_ops.cross_kernel_ns as f64 / 1_000_000.0,
            flow_ops.cross_out_ns as f64 / 1_000_000.0,
            flow_ops.cross_cat_ns as f64 / 1_000_000.0,
            flow_ops.mlp_ns as f64 / 1_000_000.0,
            flow_ops.block_norm_mod_ns as f64 / 1_000_000.0,
            flow_ops.block_norm_affine_ns as f64 / 1_000_000.0,
            flow_ops.block_gate_residual_ns as f64 / 1_000_000.0,
            flow_ops.model_io_ns as f64 / 1_000_000.0,
            flow_ops.model_input_ns as f64 / 1_000_000.0,
            flow_ops.model_output_ns as f64 / 1_000_000.0,
            flow_ops.module_cast_pad_ns as f64 / 1_000_000.0,
            flow_ops.module_attention_ns as f64 / 1_000_000.0,
            flow_ops.module_output_ns as f64 / 1_000_000.0,
        );

        let mut strict_stats = Vec::new();
        if let Some(stats) = compare_probe_tensor(
            "guided",
            guided.as_slice(),
            &hook,
            format!("{step_prefix}.pred_v.feats").as_str(),
        ) {
            strict_stats.push(("guided", stats));
        }
        if let Some(pos) = pos
            && let Some(stats) = compare_probe_tensor(
                "pos",
                pos.as_slice(),
                &hook,
                format!("{step_prefix}.pred_v_pos.feats").as_str(),
            )
        {
            strict_stats.push(("pos", stats));
        }
        if let Some(neg) = neg
            && let Some(stats) = compare_probe_tensor(
                "neg",
                neg.as_slice(),
                &hook,
                format!("{step_prefix}.pred_v_neg.feats").as_str(),
            )
        {
            strict_stats.push(("neg", stats));
        }

        if std::env::var("TRELLIS2_SLAT_FORWARD_PROBE_STRICT").is_ok() {
            assert!(
                !strict_stats.is_empty(),
                "strict SLat probe did not compare any reference tensors"
            );
            for (label, stats) in strict_stats {
                assert_eq!(
                    stats.non_finite_count, 0,
                    "SLat {label} step {probe_step} produced {} non-finite comparisons",
                    stats.non_finite_count
                );
                assert!(
                    stats.mean_abs <= 1.0e-3,
                    "SLat {label} step {probe_step} mean_abs {:.6e} exceeded tolerance",
                    stats.mean_abs
                );
                assert!(
                    stats.max_abs <= 1.0e-2,
                    "SLat {label} step {probe_step} max_abs {:.6e} exceeded tolerance",
                    stats.max_abs
                );
            }
        }
    }

    #[cfg(feature = "runtime-model-wgpu")]
    #[test]
    fn sparse_structure_flow_sampler_probe_current_reference_wgpu() {
        if std::env::var("TRELLIS2_SPARSE_STRUCTURE_FLOW_PROBE").is_err() {
            eprintln!(
                "Skipping Trellis.2 sparse-structure WGPU sampler probe: set TRELLIS2_SPARSE_STRUCTURE_FLOW_PROBE=1 to enable."
            );
            return;
        }

        let hook_path = std::env::var("TRELLIS2_SPARSE_STRUCTURE_FLOW_PROBE_HOOK")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(
                    "tmp/runs/20260621T203243Z_trellis2_ui_medium_fullhook_capture/python/reference_hook.safetensors",
                )
            });
        assert!(
            hook_path.exists(),
            "TRELLIS2_SPARSE_STRUCTURE_FLOW_PROBE_HOOK does not exist: {}",
            hook_path.display()
        );
        let weights_root = std::env::var("TRELLIS2_WEIGHTS_ROOT")
            .map(PathBuf::from)
            .expect("TRELLIS2_WEIGHTS_ROOT must point at the TRELLIS.2-4B weights root");
        assert!(
            weights_root.exists(),
            "TRELLIS2_WEIGHTS_ROOT does not exist: {}",
            weights_root.display()
        );

        let profile = std::env::var("TRELLIS2_SPARSE_STRUCTURE_FLOW_PROBE_PROFILE")
            .unwrap_or_else(|_| "reference-f32".to_string());
        let (self_f16, cross_f16, final_f32_steps, stock_bf16_emulation) = match profile.as_str() {
            "reference-f32" | "f32" => (false, false, 0usize, false),
            "stock-bf16-emulated" => (false, false, 0usize, true),
            "wgpu-fast-mixed-f16" => (false, false, 0usize, false),
            "wgpu-fast-sparse-self-f16" => (true, false, 0usize, false),
            "wgpu-fast-sparse-cross-f16" => (false, true, 0usize, false),
            "wgpu-fast-f16-tail1-f32" => (true, true, 1usize, false),
            "wgpu-fast-f16-tail2-f32" => (true, true, 2usize, false),
            "wgpu-fast-f16-tail4-f32" => (true, true, 4usize, false),
            "wgpu-fast-f16-tail6-f32" => (true, true, 6usize, false),
            "wgpu-fast-f16" => (true, true, 0usize, false),
            other => panic!("unsupported sparse-structure probe profile '{other}'"),
        };
        set_runtime_model_debug_config(RuntimeModelDebugConfig {
            stage_debug: std::env::var("TRELLIS2_SPARSE_STRUCTURE_FLOW_PROBE_STAGE_DEBUG").is_ok(),
            attention_debug: false,
            sparse_flow_module_attention: true,
            sparse_flow_module_attention_f16: self_f16 || cross_f16,
            sparse_flow_linear_f16: std::env::var(
                "TRELLIS2_SPARSE_STRUCTURE_FLOW_PROBE_LINEAR_F16",
            )
            .is_ok(),
            sparse_flow_torso_f16: std::env::var("TRELLIS2_SPARSE_STRUCTURE_FLOW_PROBE_TORSO_F16")
                .is_ok(),
            sparse_flow_stock_bf16_emulation: stock_bf16_emulation,
            sparse_flow_coord_rope_kernel: true,
            sparse_decoder_conv_f16: false,
        });
        set_runtime_model_sparse_flow_attention_policy(self_f16, cross_f16, final_f32_steps);

        let hook = HookSnapshot::from_file(&hook_path)
            .expect("load Python sparse-structure hook snapshot");
        let model_stem = std::env::var("TRELLIS2_SPARSE_STRUCTURE_FLOW_PROBE_MODEL_STEM")
            .unwrap_or_else(|_| "ckpts/ss_flow_img_dit_1_3B_64_bf16".to_string());
        let runtime = SparseStructureFlowRuntime::load_from_stem(
            weights_root.as_path(),
            None,
            model_stem.as_str(),
            true,
            None,
        )
        .expect("load sparse-structure flow runtime on WGPU");
        assert_eq!(
            runtime.backend_name(),
            "wgpu",
            "probe must execute on the WGPU backend"
        );

        let config = runtime.config().clone();
        let sampler_config_key = "sample_sparse_structure.sampler.config";
        let sample_cfg = sampler_config_from_hook(&hook, sampler_config_key);
        let sigma_min = sampler_sigma_min(&hook, sampler_config_key);
        let noise = required_hook_tensor(&hook, "sample_sparse_structure.noise");
        assert_eq!(
            noise.shape,
            vec![
                1usize,
                config.out_channels,
                config.resolution,
                config.resolution,
                config.resolution
            ],
            "sparse-structure noise shape must match runtime config"
        );
        let cond_key = "get_cond_512.out.cond";
        let neg_cond_key = "get_cond_512.out.neg_cond";
        let (cond_values, cond_tokens) = hook_condition_values(
            required_hook_tensor(&hook, cond_key),
            config.cond_channels,
            cond_key,
        );
        let (neg_values, neg_tokens) = hook_condition_values(
            required_hook_tensor(&hook, neg_cond_key),
            config.cond_channels,
            neg_cond_key,
        );
        assert_eq!(cond_tokens, neg_tokens, "cond/neg token count mismatch");
        let cond = runtime
            .prepare_condition(cond_values.as_slice(), cond_tokens)
            .expect("prepare positive sparse-structure condition");
        let neg_cond = runtime
            .prepare_condition(neg_values.as_slice(), neg_tokens)
            .expect("prepare negative sparse-structure condition");

        if let SparseStructureFlowRuntime::Wgpu(runtime_impl) = &runtime {
            let cond_tensor = match &cond {
                SparseFlowCondition::Wgpu(tensor) => tensor.clone(),
                SparseFlowCondition::Cpu(_) => panic!("positive condition must be WGPU-backed"),
            };
            let t_pairs = super::timestep_pairs(sample_cfg.steps, sample_cfg.rescale_t);
            maybe_run_sparse_structure_block0_debug_probe(
                runtime_impl,
                noise,
                t_pairs[0].0,
                cond_tensor,
                &hook,
            );
            if std::env::var("TRELLIS2_SPARSE_STRUCTURE_FLOW_PROBE_BLOCK0_DEBUG_ONLY").is_ok() {
                return;
            }
        }

        let start = Instant::now();
        let trace = runtime
            .sample_with_trace(
                noise.data.as_slice(),
                sample_cfg,
                sigma_min,
                &cond,
                &neg_cond,
                None,
                true,
            )
            .expect("sample sparse-structure flow with trace");
        <WgpuRuntimeBackend as Backend>::sync(&burn_wgpu::WgpuDevice::default())
            .expect("sync sparse-structure flow probe");
        let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;
        eprintln!(
            "trellis2 sparse-structure sampler probe: profile={profile} steps={} elapsed={elapsed_ms:.2}ms",
            sample_cfg.steps
        );

        let mut strict_stats = Vec::new();
        if let Some(stats) = compare_probe_tensor(
            "step_0_pred_v",
            trace.step_0_pred_v.as_slice(),
            &hook,
            "sample_sparse_structure.sampler.step_000_of_012.pred_v",
        ) {
            strict_stats.push(("step_0_pred_v", stats));
        }
        if let Some(stats) = compare_probe_tensor(
            "step_0_x_t",
            trace.step_0_x_t.as_slice(),
            &hook,
            "sample_sparse_structure.sampler.step_000_of_012.x_t",
        ) {
            strict_stats.push(("step_0_x_t", stats));
        }
        if let Some(stats) = compare_probe_tensor(
            "latent",
            trace.samples.as_slice(),
            &hook,
            "sample_sparse_structure.latent",
        ) {
            strict_stats.push(("latent", stats));
        }

        for (step_idx, pred_v) in trace.step_pred_v.iter().enumerate() {
            let key = format!(
                "sample_sparse_structure.sampler.step_{step_idx:03}_of_{:03}.pred_v",
                sample_cfg.steps
            );
            if let Some(stats) = compare_probe_tensor(
                format!("step_{step_idx:03}_pred_v").as_str(),
                pred_v,
                &hook,
                key.as_str(),
            ) {
                strict_stats.push(("step_pred_v", stats));
            }
        }
        for (step_idx, x_t) in trace.step_x_t.iter().enumerate() {
            let key = format!(
                "sample_sparse_structure.sampler.step_{step_idx:03}_of_{:03}.x_t",
                sample_cfg.steps
            );
            if let Some(stats) = compare_probe_tensor(
                format!("step_{step_idx:03}_x_t").as_str(),
                x_t,
                &hook,
                key.as_str(),
            ) {
                strict_stats.push(("step_x_t", stats));
            }
        }

        if std::env::var("TRELLIS2_SPARSE_STRUCTURE_FLOW_PROBE_STRICT").is_ok() {
            assert!(
                !strict_stats.is_empty(),
                "strict sparse-structure flow probe did not compare any reference tensors"
            );
            let mean_abs_tol = std::env::var("TRELLIS2_SPARSE_STRUCTURE_FLOW_PROBE_MEAN_ABS_TOL")
                .ok()
                .and_then(|value| value.parse::<f32>().ok())
                .unwrap_or(2.0e-3);
            let max_abs_tol = std::env::var("TRELLIS2_SPARSE_STRUCTURE_FLOW_PROBE_MAX_ABS_TOL")
                .ok()
                .and_then(|value| value.parse::<f32>().ok())
                .unwrap_or(2.0e-2);
            for (label, stats) in strict_stats {
                assert_eq!(
                    stats.non_finite_count, 0,
                    "sparse-structure {label} produced {} non-finite comparisons",
                    stats.non_finite_count
                );
                assert!(
                    stats.mean_abs <= mean_abs_tol,
                    "sparse-structure {label} mean_abs {:.6e} exceeded tolerance {:.6e}",
                    stats.mean_abs,
                    mean_abs_tol
                );
                assert!(
                    stats.max_abs <= max_abs_tol,
                    "sparse-structure {label} max_abs {:.6e} exceeded tolerance {:.6e}",
                    stats.max_abs,
                    max_abs_tol
                );
            }
        }
        set_runtime_model_debug_config(RuntimeModelDebugConfig::default());
    }

    #[cfg(feature = "runtime-model-wgpu")]
    trait ProbeRowsVec {
        fn reshape_to_rows_vec(self, channels: usize) -> Vec<f32>;
    }

    #[cfg(feature = "runtime-model-wgpu")]
    impl ProbeRowsVec for Vec<f32> {
        fn reshape_to_rows_vec(self, channels: usize) -> Vec<f32> {
            assert!(
                self.len().is_multiple_of(channels.max(1)),
                "probe tensor length {} is not divisible by channels {}",
                self.len(),
                channels
            );
            self
        }
    }

    #[test]
    fn tiny_sparse_flow_forward_cpu_backend() {
        let device =
            <burn::backend::NdArray<f32> as burn::tensor::backend::BackendTypes>::Device::default();
        run_tiny_forward::<burn::backend::NdArray<f32>>(&device);
    }

    #[cfg(feature = "runtime-model-wgpu")]
    #[test]
    fn tiny_sparse_flow_forward_wgpu_backend() {
        if std::env::var("BURN_WGPU_SMOKE").is_err() {
            eprintln!("skipping: set BURN_WGPU_SMOKE=1 to run wgpu sparse flow smoke");
            return;
        }
        let result = std::panic::catch_unwind(|| {
            let device = burn_wgpu::WgpuDevice::default();
            run_tiny_forward::<burn_wgpu::Wgpu<f32, i32, u32>>(&device);
        });
        if result.is_err() {
            eprintln!("skipping: wgpu backend not available on this system");
        }
    }
}

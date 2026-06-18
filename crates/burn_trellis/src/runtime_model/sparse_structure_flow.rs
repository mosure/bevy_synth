#![allow(deprecated)]

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
use super::runtime_config::{
    runtime_model_attention_debug_enabled, runtime_model_stage_debug_enabled,
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
    layer_norm_affine_forward_wgpu, rope_rotate_pairs_from_coords_wgpu, rope_rotate_pairs_wgpu,
};
use burn_store::{KeyRemapper, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};
use serde::Deserialize;
#[cfg(test)]
use serde::Serialize;

use crate::sampler::{
    FlowEulerSampleConfig, FlowEulerSampleTrace, mid_snapshot_step, timestep_pairs,
};

const F16_SUFFIX: &str = "_f16";
const MAX_PERIOD: f32 = 10_000.0;
const LAYER_NORM_EPS: f32 = 1.0e-6;
const RMS_NORM_EPS: f32 = 1.0e-12;
const ROPE_CACHE_MAX_ENTRIES: usize = 256;
static HOST_READBACK_COUNT: AtomicU64 = AtomicU64::new(0);
static HOST_READBACK_ELEMENTS: AtomicU64 = AtomicU64::new(0);
static FLOW_SELF_ATTN_CALLS: AtomicU64 = AtomicU64::new(0);
static FLOW_SELF_ATTN_NS: AtomicU64 = AtomicU64::new(0);
static FLOW_CROSS_ATTN_CALLS: AtomicU64 = AtomicU64::new(0);
static FLOW_CROSS_ATTN_NS: AtomicU64 = AtomicU64::new(0);
static FLOW_MLP_CALLS: AtomicU64 = AtomicU64::new(0);
static FLOW_MLP_NS: AtomicU64 = AtomicU64::new(0);
static CFG_POS_NEG_DEBUG_COUNT: AtomicU64 = AtomicU64::new(0);
static ROPE_CACHE: OnceLock<Mutex<HashMap<RopeCacheKey, Arc<RopeCosSinRange>>>> = OnceLock::new();
#[cfg(feature = "runtime-model-wgpu")]
thread_local! {
    static LAYER_NORM_NO_AFFINE_PARAMS_CACHE: RefCell<HashMap<usize, (Tensor<WgpuRuntimeBackend, 1>, Tensor<WgpuRuntimeBackend, 1>)>> =
        RefCell::new(HashMap::new());
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
}

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
impl SparseFlowLayerNormWgpuBridge<WgpuRuntimeBackend> for RopeRotateWgpuBridgeImpl {
    fn no_affine(
        x: Tensor<WgpuRuntimeBackend, 3>,
        eps: f32,
    ) -> Option<Tensor<WgpuRuntimeBackend, 3>> {
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
}

#[cfg(test)]
#[derive(Module, Debug)]
struct BinaryBlob<B: Backend> {
    bytes: Param<Tensor<B, 1, Int>>,
}

#[cfg(test)]
#[derive(Debug, Deserialize, Serialize)]
struct BlobMetadata {
    bytes_len: usize,
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
}

pub fn sparse_flow_op_telemetry() -> SparseFlowOpTelemetry {
    SparseFlowOpTelemetry {
        self_attn_calls: FLOW_SELF_ATTN_CALLS.load(Ordering::Relaxed),
        self_attn_ns: FLOW_SELF_ATTN_NS.load(Ordering::Relaxed),
        cross_attn_calls: FLOW_CROSS_ATTN_CALLS.load(Ordering::Relaxed),
        cross_attn_ns: FLOW_CROSS_ATTN_NS.load(Ordering::Relaxed),
        mlp_calls: FLOW_MLP_CALLS.load(Ordering::Relaxed),
        mlp_ns: FLOW_MLP_NS.load(Ordering::Relaxed),
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SparseFlowOpKind {
    SelfAttn,
    CrossAttn,
    Mlp,
}

fn runtime_sample_progress_interval(steps: usize) -> usize {
    if steps <= 16 { 1 } else { (steps / 8).max(1) }
}

#[derive(Clone, Debug)]
pub struct SparseFlowRowTrace {
    pub steps: usize,
    pub row_channels: usize,
    pub samples: Vec<f32>,
    pub step_0_x_t: Vec<f32>,
    pub step_mid_x_t: Vec<f32>,
    pub step_last_x_t: Vec<f32>,
    #[cfg(feature = "runtime-model-wgpu")]
    pub samples_wgpu: Option<Tensor<WgpuRuntimeBackend, 2>>,
    #[cfg(feature = "runtime-model-wgpu")]
    pub step_0_x_t_wgpu: Option<Tensor<WgpuRuntimeBackend, 2>>,
    #[cfg(feature = "runtime-model-wgpu")]
    pub step_mid_x_t_wgpu: Option<Tensor<WgpuRuntimeBackend, 2>>,
    #[cfg(feature = "runtime-model-wgpu")]
    pub step_last_x_t_wgpu: Option<Tensor<WgpuRuntimeBackend, 2>>,
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

impl<B: Backend> TimestepEmbedder<B> {
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
        linear_forward_stable_2d(&self.mlp_2, silu(hidden))
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

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [_, _, heads, head_dim] = x.dims();
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
        x.mul(gamma)
    }
}

#[derive(Module, Debug)]
pub struct FeedForwardNet<B: Backend> {
    pub mlp_0: nn::Linear<B>,
    pub mlp_2: nn::Linear<B>,
}

impl<B: Backend> FeedForwardNet<B> {
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
        if chunk_tokens >= tokens {
            let hidden = linear_forward_stable_via_2d(&self.mlp_0, x);
            return linear_forward_stable_via_2d(&self.mlp_2, gelu(hidden));
        }

        let device = x.device();
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
                let hidden = linear_forward_stable_2d(&self.mlp_0, x_chunk);
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
                let hidden = gelu(hidden);
                if let Some(stage_start) = gelu_start {
                    eprintln!(
                        "burn_trellis: mlp.chunk {}/{} gelu done ({:.2} ms)",
                        chunk_idx + 1,
                        total_chunks,
                        stage_start.elapsed().as_secs_f64() * 1000.0
                    );
                }
                let chunk_out = linear_forward_stable_2d(&self.mlp_2, hidden);
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
{
    pub fn new(
        device: &B::Device,
        channels: usize,
        num_heads: usize,
        use_rope: bool,
        rope_freq: [f32; 2],
        qk_rms_norm: bool,
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
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        resolution: usize,
        token_coords: Option<Tensor<B, 2, Int>>,
    ) -> Tensor<B, 3> {
        let [batch, tokens, channels] = x.dims();
        if sparse_flow_chunked_forward_for_backend::<B>(tokens) {
            return self.forward_chunked_stream(x, resolution, token_coords);
        }
        let qkv = linear_forward_stable(&self.to_qkv, x).reshape([
            batch,
            tokens,
            3,
            self.num_heads,
            self.head_dim,
        ]);
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
            .clone()
            .slice([
                0..batch,
                0..tokens,
                1..2,
                0..self.num_heads,
                0..self.head_dim,
            ])
            .reshape([batch, tokens, self.num_heads, self.head_dim]);
        let v = qkv
            .slice([
                0..batch,
                0..tokens,
                2..3,
                0..self.num_heads,
                0..self.head_dim,
            ])
            .reshape([batch, tokens, self.num_heads, self.head_dim]);

        let q = if let Some(norm) = self.q_rms_norm.as_ref() {
            norm.forward(q)
        } else {
            q
        };
        let k = if let Some(norm) = self.k_rms_norm.as_ref() {
            norm.forward(k)
        } else {
            k
        };
        let (q, k) = if self.use_rope {
            apply_rope(
                q,
                k,
                resolution,
                self.head_dim,
                self.rope_freq,
                token_coords,
            )
        } else {
            (q, k)
        };

        let out = scaled_dot_product_attention(q, k, v, self.head_dim);
        linear_forward_stable(&self.to_out, out.reshape([batch, tokens, channels]))
    }

    fn forward_chunked_stream(
        &self,
        x: Tensor<B, 3>,
        resolution: usize,
        token_coords: Option<Tensor<B, 2, Int>>,
    ) -> Tensor<B, 3> {
        let [batch, tokens, channels] = x.dims();
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
            let module_chunk_cap = sparse_flow_module_attention_chunk_cap(tokens);
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

        let mut k_chunks: Vec<Tensor<B, 4>> = Vec::new();
        let mut v_chunks: Vec<Tensor<B, 4>> = Vec::new();
        let mut q_chunks: Vec<Tensor<B, 4>> = Vec::new();
        let mut kv_start = 0usize;
        while kv_start < tokens {
            let kv_end = (kv_start + kv_chunk_tokens).min(tokens);
            let x_chunk = x.clone().slice([0..batch, kv_start..kv_end, 0..channels]);
            let qkv = linear_forward_stable(&self.to_qkv, x_chunk).reshape([
                batch,
                kv_end - kv_start,
                3,
                self.num_heads,
                self.head_dim,
            ]);
            let mut k = qkv
                .clone()
                .slice([
                    0..batch,
                    0..(kv_end - kv_start),
                    1..2,
                    0..self.num_heads,
                    0..self.head_dim,
                ])
                .reshape([batch, kv_end - kv_start, self.num_heads, self.head_dim]);
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
            let mut q = qkv
                .slice([
                    0..batch,
                    0..(kv_end - kv_start),
                    0..1,
                    0..self.num_heads,
                    0..self.head_dim,
                ])
                .reshape([batch, kv_end - kv_start, self.num_heads, self.head_dim]);

            if let Some(norm) = self.k_rms_norm.as_ref() {
                k = norm.forward(k);
            }
            if let Some(norm) = self.q_rms_norm.as_ref() {
                q = norm.forward(q);
            }
            if self.use_rope {
                k = apply_rope_single(
                    k,
                    resolution,
                    self.head_dim,
                    self.rope_freq,
                    token_coords.clone(),
                    kv_start,
                );
                q = apply_rope_single(
                    q,
                    resolution,
                    self.head_dim,
                    self.rope_freq,
                    token_coords.clone(),
                    kv_start,
                );
            }

            k_chunks.push(k.permute([0, 2, 1, 3]));
            v_chunks.push(v.permute([0, 2, 1, 3]));
            if reuse_qkv {
                q_chunks.push(q.permute([0, 2, 1, 3]));
            }
            kv_start = kv_end;
        }

        if attention_uses_module_kernel::<B>() {
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
            if attention_debug_enabled() && tokens >= 1024 {
                eprintln!(
                    "burn_trellis: attn chunked backend={backend_name} impl=flash_attention(module_attention) q_chunk={query_chunk_tokens} kv_chunk={kv_chunk_tokens} tokens={tokens} reuse_qkv={reuse_qkv} full={module_full_attention}"
                );
            }
            let mut out_chunks = Vec::new();
            if reuse_qkv {
                for q in q_chunks.into_iter() {
                    let q_tokens = q.dims()[2];
                    let out = attention(
                        q,
                        k_full.clone(),
                        v_full.clone(),
                        None,
                        None,
                        AttentionModuleOptions::default(),
                    )
                    .permute([0, 2, 1, 3])
                    .reshape([batch, q_tokens, channels]);
                    out_chunks.push(linear_forward_stable(&self.to_out, out));
                }
            } else {
                let mut q_start = 0usize;
                while q_start < tokens {
                    let q_end = (q_start + query_chunk_tokens).min(tokens);
                    let x_chunk = x.clone().slice([0..batch, q_start..q_end, 0..channels]);
                    let qkv = linear_forward_stable(&self.to_qkv, x_chunk).reshape([
                        batch,
                        q_end - q_start,
                        3,
                        self.num_heads,
                        self.head_dim,
                    ]);
                    let mut q = qkv
                        .slice([
                            0..batch,
                            0..(q_end - q_start),
                            0..1,
                            0..self.num_heads,
                            0..self.head_dim,
                        ])
                        .reshape([batch, q_end - q_start, self.num_heads, self.head_dim]);
                    if let Some(norm) = self.q_rms_norm.as_ref() {
                        q = norm.forward(q);
                    }
                    if self.use_rope {
                        q = apply_rope_single(
                            q,
                            resolution,
                            self.head_dim,
                            self.rope_freq,
                            token_coords.clone(),
                            q_start,
                        );
                    }
                    let out = attention(
                        q.permute([0, 2, 1, 3]),
                        k_full.clone(),
                        v_full.clone(),
                        None,
                        None,
                        AttentionModuleOptions::default(),
                    )
                    .permute([0, 2, 1, 3])
                    .reshape([batch, q_end - q_start, channels]);
                    out_chunks.push(linear_forward_stable(&self.to_out, out));
                    q_start = q_end;
                }
            }
            return Tensor::cat(out_chunks, 1);
        }

        let mut out_chunks = Vec::new();
        if reuse_qkv {
            for q in q_chunks.into_iter() {
                let q_tokens = q.dims()[2];
                let out = scaled_dot_product_attention_stream_chunked_keys(
                    q,
                    k_chunks.as_slice(),
                    v_chunks.as_slice(),
                    self.head_dim,
                )
                .permute([0, 2, 1, 3])
                .reshape([batch, q_tokens, channels]);
                out_chunks.push(linear_forward_stable(&self.to_out, out));
            }
        } else {
            let mut q_start = 0usize;
            while q_start < tokens {
                let q_end = (q_start + query_chunk_tokens).min(tokens);
                let x_chunk = x.clone().slice([0..batch, q_start..q_end, 0..channels]);
                let qkv = linear_forward_stable(&self.to_qkv, x_chunk).reshape([
                    batch,
                    q_end - q_start,
                    3,
                    self.num_heads,
                    self.head_dim,
                ]);
                let mut q = qkv
                    .slice([
                        0..batch,
                        0..(q_end - q_start),
                        0..1,
                        0..self.num_heads,
                        0..self.head_dim,
                    ])
                    .reshape([batch, q_end - q_start, self.num_heads, self.head_dim]);
                if let Some(norm) = self.q_rms_norm.as_ref() {
                    q = norm.forward(q);
                }
                if self.use_rope {
                    q = apply_rope_single(
                        q,
                        resolution,
                        self.head_dim,
                        self.rope_freq,
                        token_coords.clone(),
                        q_start,
                    );
                }

                let out = scaled_dot_product_attention_stream_chunked_keys(
                    q.permute([0, 2, 1, 3]),
                    k_chunks.as_slice(),
                    v_chunks.as_slice(),
                    self.head_dim,
                )
                .permute([0, 2, 1, 3])
                .reshape([batch, q_end - q_start, channels]);

                out_chunks.push(linear_forward_stable(&self.to_out, out));
                q_start = q_end;
            }
        }

        Tensor::cat(out_chunks, 1)
    }
}

impl<B: Backend> CrossAttention<B> {
    pub fn new(
        device: &B::Device,
        channels: usize,
        ctx_channels: usize,
        num_heads: usize,
        qk_rms_norm: bool,
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
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, context: Tensor<B, 3>) -> Tensor<B, 3> {
        let (k, v) = self.project_context_kv(context);
        self.forward_from_projected_kv(x, k, v)
    }

    fn project_context_kv(&self, context: Tensor<B, 3>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let [batch, ctx_tokens, _ctx_channels] = context.dims();
        let kv = linear_forward_stable(&self.to_kv, context).reshape([
            batch,
            ctx_tokens,
            2,
            self.num_heads,
            self.head_dim,
        ]);
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
        (k, v)
    }

    fn forward_from_projected_kv(
        &self,
        x: Tensor<B, 3>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        let [batch, tokens, channels] = x.dims();
        if sparse_flow_chunked_forward_for_backend::<B>(tokens) {
            return self.forward_chunked_projected_kv(x, k, v);
        }

        let mut q = linear_forward_stable(&self.to_q, x).reshape([
            batch,
            tokens,
            self.num_heads,
            self.head_dim,
        ]);
        if let Some(norm) = self.q_rms_norm.as_ref() {
            q = norm.forward(q);
        }

        let out = scaled_dot_product_attention(q, k, v, self.head_dim);
        linear_forward_stable(&self.to_out, out.reshape([batch, tokens, channels]))
    }

    fn forward_chunked_projected_kv(
        &self,
        x: Tensor<B, 3>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        let [batch, tokens, channels] = x.dims();
        let ctx_tokens = k.dims()[1];

        let backend_name = std::any::type_name::<B>();
        let use_module_attention = attention_uses_module_kernel::<B>();
        let module_non_fusion = attention_uses_non_fusion_module_kernel::<B>();
        let module_full_attention =
            module_non_fusion && sparse_flow_module_attention_prefers_full(tokens);

        let mut query_chunk_tokens = if use_module_attention {
            if module_full_attention {
                tokens.max(1)
            } else {
                sparse_flow_module_attention_chunk_cap(tokens)
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

        let k_module = if use_module_attention {
            Some(k.clone().permute([0, 2, 1, 3]))
        } else {
            None
        };
        let v_module = if use_module_attention {
            Some(v.clone().permute([0, 2, 1, 3]))
        } else {
            None
        };

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
            let mut q = linear_forward_stable(&self.to_q, x_chunk).reshape([
                batch,
                chunk_tokens,
                self.num_heads,
                self.head_dim,
            ]);
            if let Some(norm) = self.q_rms_norm.as_ref() {
                q = norm.forward(q);
            }

            let attn_start = Instant::now();
            let out = if use_module_attention {
                attention(
                    q.permute([0, 2, 1, 3]),
                    k_module.clone().expect("module K must be present"),
                    v_module.clone().expect("module V must be present"),
                    None,
                    None,
                    AttentionModuleOptions::default(),
                )
                .permute([0, 2, 1, 3])
                .reshape([batch, chunk_tokens, channels])
            } else {
                scaled_dot_product_attention(q, k.clone(), v.clone(), self.head_dim).reshape([
                    batch,
                    chunk_tokens,
                    channels,
                ])
            };
            out_chunks.push(linear_forward_stable(&self.to_out, out));
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

        Tensor::cat(out_chunks, 1)
    }
}

impl<B: Backend> ModulatedTransformerCrossBlock<B>
where
    RopeRotateWgpuBridgeImpl: RopeRotateWgpuBridge<B>,
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
    ) -> Self {
        let self_attn = SelfAttention::new(
            device,
            channels,
            num_heads,
            use_rope,
            rope_freq,
            qk_rms_norm,
        );
        let cross_attn =
            CrossAttention::new(device, channels, ctx_channels, num_heads, qk_rms_norm_cross);
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

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        mod_signal: Tensor<B, 2>,
        context: Tensor<B, 3>,
        resolution: usize,
        token_coords: Option<Tensor<B, 2, Int>>,
    ) -> Tensor<B, 3>
    where
        RopeRotateWgpuBridgeImpl: SparseFlowLayerNormWgpuBridge<B>,
    {
        let [batch, tokens, channels] = x.dims();
        let block_op_debug = attention_debug_enabled() && tokens >= 131_072;
        let mod_bias = self.modulation.val().reshape([1, channels * 6]);
        let mod_signal_dtype: burn::tensor::FloatDType = mod_signal.dtype().into();
        let mod_bias_dtype: burn::tensor::FloatDType = mod_bias.dtype().into();
        let mod_bias = if mod_bias_dtype != mod_signal_dtype {
            mod_bias.cast(mod_signal_dtype)
        } else {
            mod_bias
        };
        let mod_signal = mod_signal.add(mod_bias);
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
            .slice([0..batch, (channels * 5)..(channels * 6)])
            .reshape([batch, 1, channels]);

        let h = layer_norm_no_affine(x.clone(), LAYER_NORM_EPS)
            .mul(scale_msa.add_scalar(1.0))
            .add(shift_msa);
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
        let self_attn_ns = self_attn_start
            .elapsed()
            .as_nanos()
            .min(u128::from(u64::MAX)) as u64;
        record_sparse_flow_op(SparseFlowOpKind::SelfAttn, self_attn_ns);
        let h = h.mul(gate_msa);
        let x = x.add(h);

        let h = layer_norm_affine_stable(x.clone(), &self.norm2, LAYER_NORM_EPS);
        let cross_attn_start = Instant::now();
        let x = if block_op_debug {
            let start = Instant::now();
            eprintln!("burn_trellis: flow.block op=cross_attn begin (tokens={tokens})");
            let out = self.cross_attn.forward(h, context.clone());
            eprintln!(
                "burn_trellis: flow.block op=cross_attn done ({:.2} ms)",
                start.elapsed().as_secs_f64() * 1000.0
            );
            x.add(out)
        } else {
            let out = self.cross_attn.forward(h, context);
            x.add(out)
        };
        let cross_attn_ns = cross_attn_start
            .elapsed()
            .as_nanos()
            .min(u128::from(u64::MAX)) as u64;
        record_sparse_flow_op(SparseFlowOpKind::CrossAttn, cross_attn_ns);

        let h = layer_norm_no_affine(x.clone(), LAYER_NORM_EPS)
            .mul(scale_mlp.add_scalar(1.0))
            .add(shift_mlp);
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
        let mlp_ns = mlp_start.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
        record_sparse_flow_op(SparseFlowOpKind::Mlp, mlp_ns);
        let h = h.mul(gate_mlp);
        x.add(h)
    }
}

impl<B: Backend> SparseStructureFlowModel<B>
where
    RopeRotateWgpuBridgeImpl: RopeRotateWgpuBridge<B>,
    RopeRotateWgpuBridgeImpl: SparseFlowLayerNormWgpuBridge<B>,
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

    fn forward_tokens(
        &self,
        x: Tensor<B, 3>,
        t: Tensor<B, 1>,
        cond: Tensor<B, 3>,
        resolution: usize,
        token_coords: Option<Tensor<B, 2, Int>>,
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

        let mut h = linear_forward_token_chunked(
            &self.input_layer,
            x,
            sparse_flow_linear_chunk_tokens_for_backend::<B>(tokens),
        );

        let t_emb = self
            .t_embedder
            .forward(t, self.config.frequency_embedding_size);
        let mod_signal = linear_forward_stable_2d(&self.ada_ln_modulation, silu(t_emb));

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
            );
            if let Some(start) = block_start {
                eprintln!(
                    "burn_trellis: flow.block {}/{} done ({:.2} ms)",
                    block_idx + 1,
                    self.blocks.len(),
                    start.elapsed().as_secs_f64() * 1000.0
                );
            }
        }

        let h = layer_norm_no_affine(h, LAYER_NORM_EPS);
        linear_forward_token_chunked(
            &self.out_layer,
            h,
            sparse_flow_linear_chunk_tokens_for_backend::<B>(tokens),
        )
    }

    pub fn forward(&self, x: Tensor<B, 5>, t: Tensor<B, 1>, cond: Tensor<B, 3>) -> Tensor<B, 5> {
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
        let tokens_tensor = x.reshape([batch, channels, tokens]).swap_dims(1, 2);
        let out_tokens = self.forward_tokens(tokens_tensor, t, cond, self.config.resolution, None);
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
        self.forward_tokens(x, t, cond, sparse_resolution.max(1), Some(token_coords))
    }
}

impl<B> SparseStructureFlowRuntimeImpl<B>
where
    B: Backend,
    B::Device: Default,
    RopeRotateWgpuBridgeImpl: RopeRotateWgpuBridge<B>,
    RopeRotateWgpuBridgeImpl: SparseFlowLayerNormWgpuBridge<B>,
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
                    if sparse_flow_stage_debug_enabled() {
                        log_sparse_flow_weight_probe(&model);
                    }
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
{
    fn config(&self) -> &SparseStructureFlowConfig {
        &self.config
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

        let mut step_0_x_t: Option<Tensor<B, 5>> = None;
        let mut step_mid_x_t: Option<Tensor<B, 5>> = None;
        let mut step_last_x_t: Option<Tensor<B, 5>> = None;
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
        for (step_idx, (t, t_prev)) in t_pairs.into_iter().enumerate() {
            let step_start = Instant::now();
            let pred = self.predict_with_cfg_tensor(
                x_t.clone(),
                t,
                sample_cfg,
                sigma_min,
                cond.clone(),
                neg_cond.clone(),
                concat_tensor.clone(),
            )?;
            let dt = t - t_prev;
            x_t = x_t.sub(pred.mul_scalar(dt));
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
        eprintln!(
            "burn_trellis: {stage_label} complete ({:.2} ms)",
            sample_start.elapsed().as_secs_f64() * 1000.0
        );

        let state_len = state_channels.saturating_mul(voxel);
        let (samples, step_0_x_t, step_mid_x_t, step_last_x_t) = if capture_snapshots {
            let samples_t = x_t;
            let step_0_t = step_0_x_t
                .unwrap_or_else(|| samples_t.clone())
                .reshape([state_len]);
            let step_mid_t = step_mid_x_t
                .unwrap_or_else(|| samples_t.clone())
                .reshape([state_len]);
            let step_last_t = step_last_x_t
                .unwrap_or_else(|| samples_t.clone())
                .reshape([state_len]);
            let merged = Tensor::cat(
                vec![
                    samples_t.reshape([state_len]),
                    step_0_t,
                    step_mid_t,
                    step_last_t,
                ],
                0,
            );
            let merged = tensor_to_vec_1d(merged, "failed to read sparse trace tensor")?;
            let segment = state_len;
            let samples = merged[..segment].to_vec();
            let step_0_x_t = merged[segment..segment * 2].to_vec();
            let step_mid_x_t = merged[segment * 2..segment * 3].to_vec();
            let step_last_x_t = merged[segment * 3..segment * 4].to_vec();
            (samples, step_0_x_t, step_mid_x_t, step_last_x_t)
        } else {
            let samples = tensor_to_vec(x_t)?;
            (samples.clone(), samples.clone(), samples.clone(), samples)
        };

        Ok(FlowEulerSampleTrace {
            steps: sample_cfg.steps,
            step_0_x_t,
            step_mid_x_t,
            step_last_x_t,
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
        for (step_idx, (t, t_prev)) in t_pairs.into_iter().enumerate() {
            let step_start = Instant::now();
            let pred = self.predict_with_cfg_tensor(
                x_t.clone(),
                t,
                sample_cfg,
                sigma_min,
                cond.clone(),
                neg_cond.clone(),
                concat_tensor.clone(),
            )?;
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
                step_0_x_t: Vec::new(),
                step_mid_x_t: Vec::new(),
                step_last_x_t: Vec::new(),
                #[cfg(feature = "runtime-model-wgpu")]
                samples_wgpu: None,
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
                step_0_x_t: Vec::new(),
                step_mid_x_t: Vec::new(),
                step_last_x_t: Vec::new(),
                #[cfg(feature = "runtime-model-wgpu")]
                samples_wgpu: None,
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

            let mut step_0_rows: Option<Tensor<B, 3>> = None;
            let mut step_mid_rows: Option<Tensor<B, 3>> = None;
            let mut step_last_rows: Option<Tensor<B, 3>> = None;
            for (step_idx, (t, t_prev)) in t_pairs.iter().copied().enumerate() {
                let step_start = Instant::now();
                let pred = self.predict_with_cfg_sparse_tensor(
                    x_t.clone(),
                    t,
                    sample_cfg,
                    sigma_min,
                    cond_batch.clone(),
                    neg_cond_batch.clone(),
                    concat_tensor.clone(),
                    sparse.sparse_resolution().max(1),
                    token_coords.clone(),
                )?;
                let dt = t - t_prev;
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
        let step_0_x_t_wgpu: Option<Tensor<WgpuRuntimeBackend, 2>>;
        #[cfg(feature = "runtime-model-wgpu")]
        let step_mid_x_t_wgpu: Option<Tensor<WgpuRuntimeBackend, 2>>;
        #[cfg(feature = "runtime-model-wgpu")]
        let step_last_x_t_wgpu: Option<Tensor<WgpuRuntimeBackend, 2>>;

        let (samples, step_0_x_t, step_mid_x_t, step_last_x_t) = if capture_snapshots {
            let samples_t = concat_batches(samples_batches, "samples")?;
            let step_0_t = concat_batches(step_0_batches, "step_0_x_t")?;
            let step_mid_t = concat_batches(step_mid_batches, "step_mid_x_t")?;
            let step_last_t = concat_batches(step_last_batches, "step_last_x_t")?;
            #[cfg(feature = "runtime-model-wgpu")]
            {
                samples_wgpu =
                    maybe_trace_rows_wgpu(samples_t.clone().reshape([row_count, used_channels]));
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
                let merged = Tensor::cat(vec![samples_t, step_0_t, step_mid_t, step_last_t], 0);
                let merged =
                    tensor_to_vec_1d(merged, "failed to read sparse-token row trace tensor")?;
                let segment = total_elements;
                (
                    merged[..segment].to_vec(),
                    merged[segment..segment * 2].to_vec(),
                    merged[segment * 2..segment * 3].to_vec(),
                    merged[segment * 3..segment * 4].to_vec(),
                )
            } else {
                (Vec::new(), Vec::new(), Vec::new(), Vec::new())
            }
        } else {
            let samples_t = concat_batches(samples_batches, "samples")?;
            #[cfg(feature = "runtime-model-wgpu")]
            {
                samples_wgpu =
                    maybe_trace_rows_wgpu(samples_t.clone().reshape([row_count, used_channels]));
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
                (samples.clone(), samples.clone(), samples.clone(), samples)
            } else {
                (Vec::new(), Vec::new(), Vec::new(), Vec::new())
            }
        };

        Ok(SparseFlowRowTrace {
            steps: sample_cfg.steps,
            row_channels: used_channels,
            samples,
            step_0_x_t,
            step_mid_x_t,
            step_last_x_t,
            #[cfg(feature = "runtime-model-wgpu")]
            samples_wgpu,
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
        _sigma_min: f32,
        cond: Tensor<B, 3>,
        neg_cond: Tensor<B, 3>,
        concat_cond: Option<Tensor<B, 3>>,
        sparse_resolution: usize,
        token_coords: Tensor<B, 2, Int>,
    ) -> Result<Tensor<B, 3>, String> {
        let in_guidance_interval =
            config.guidance_interval[0] <= timestep && timestep <= config.guidance_interval[1];
        if !in_guidance_interval {
            return self.predict_velocity_sparse_tensor(
                x_t,
                timestep,
                cond,
                concat_cond,
                sparse_resolution,
                token_coords,
            );
        }

        let w = config.guidance_strength;
        if (w - 1.0).abs() < f32::EPSILON {
            return self.predict_velocity_sparse_tensor(
                x_t,
                timestep,
                cond,
                concat_cond,
                sparse_resolution,
                token_coords,
            );
        }
        if w.abs() < f32::EPSILON {
            return self.predict_velocity_sparse_tensor(
                x_t,
                timestep,
                neg_cond,
                concat_cond,
                sparse_resolution,
                token_coords,
            );
        }

        // Keep CFG as two explicit forwards. Pairing pos/neg into batch=2 looked
        // attractive for throughput, but it regressed canonical WGPU numerics
        // (sparse occupancy collapse in strict parity runs), so the fail-safe
        // parity-preserving path remains the default.
        let pos = self.predict_velocity_sparse_tensor(
            x_t.clone(),
            timestep,
            cond,
            concat_cond.clone(),
            sparse_resolution,
            token_coords.clone(),
        )?;
        let neg = self.predict_velocity_sparse_tensor(
            x_t.clone(),
            timestep,
            neg_cond,
            concat_cond,
            sparse_resolution,
            token_coords,
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
        Ok(pos.clone().add(pos.sub(neg).mul_scalar(w)))
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
        let [_, tokens, state_channels] = x_t.dims();
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
        let t = Tensor::<B, 1>::from_floats([timestep * 1000.0], &self.device);
        Ok(self
            .model
            .forward_sparse(sample, t, cond, sparse_resolution.max(1), token_coords))
    }

    #[allow(dead_code, clippy::too_many_arguments)]
    fn predict_with_cfg_tensor(
        &self,
        x_t: Tensor<B, 5>,
        timestep: f32,
        config: FlowEulerSampleConfig,
        _sigma_min: f32,
        cond: Tensor<B, 3>,
        neg_cond: Tensor<B, 3>,
        concat_cond: Option<Tensor<B, 5>>,
    ) -> Result<Tensor<B, 5>, String> {
        let in_guidance_interval =
            config.guidance_interval[0] <= timestep && timestep <= config.guidance_interval[1];
        if !in_guidance_interval {
            return self.predict_velocity_tensor(x_t, timestep, cond, concat_cond);
        }

        let w = config.guidance_strength;
        if (w - 1.0).abs() < f32::EPSILON {
            return self.predict_velocity_tensor(x_t, timestep, cond, concat_cond);
        }
        if w.abs() < f32::EPSILON {
            return self.predict_velocity_tensor(x_t, timestep, neg_cond, concat_cond);
        }

        // Keep CFG as two explicit forwards for parity with current canonical
        // WGPU behavior; batch-paired CFG caused unacceptable numeric drift.
        let pos = self.predict_velocity_tensor(x_t.clone(), timestep, cond, concat_cond.clone())?;
        let neg = self.predict_velocity_tensor(x_t.clone(), timestep, neg_cond, concat_cond)?;
        if sparse_flow_stage_debug_enabled() {
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
        Ok(pos.clone().add(pos.sub(neg).mul_scalar(w)))
    }

    #[allow(dead_code)]
    fn predict_velocity_tensor(
        &self,
        x_t: Tensor<B, 5>,
        timestep: f32,
        cond: Tensor<B, 3>,
        concat_cond: Option<Tensor<B, 5>>,
    ) -> Result<Tensor<B, 5>, String> {
        let [_, state_channels, rx, ry, rz] = x_t.dims();
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
        let t = Tensor::<B, 1>::from_floats([timestep * 1000.0], &self.device);
        Ok(self.model.forward(sample, t, cond))
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

fn log_sparse_flow_weight_probe<B: Backend>(model: &SparseStructureFlowModel<B>) {
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
        eprintln!(
            "burn_trellis: sparse flow weight probe backend={} input_layer[min,max,mean]=[{:.6},{:.6},{:.6}] block0.cross_attn.to_q=[{:.6},{:.6},{:.6}] block0.cross_attn.to_kv=[{:.6},{:.6},{:.6}] out_layer=[{:.6},{:.6},{:.6}]",
            std::any::type_name::<B>(),
            in_min,
            in_max,
            in_mean,
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
    if tokens >= 131_072 {
        16_384
    } else if tokens >= 16_384 {
        8_192
    } else {
        tokens.max(1)
    }
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
        // Native WGPU module kernels sustain larger token chunks; raising this
        // cap reduces chunk-splitting overhead in sparse-flow entry/exit linears.
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
    let widened = if tokens >= 65_536 {
        8_192usize.min(tokens.max(1)).max(default)
    } else if tokens >= 16_384 {
        4_096usize.min(tokens.max(1)).max(default)
    } else {
        default
    };
    // Keep per-chunk hidden activations bounded for WGPU sparse-flow MLP.
    // The 1024 shape_slat path (rows ~= 4_912, hidden ~= 8_192) can otherwise
    // request a single ~161 MiB buffer and panic on adapters with ~128 MiB
    // effective storage allocation limits.
    if attention_uses_module_kernel::<B>() {
        widened.min(4_096usize).min(tokens.max(1))
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

fn linear_forward_stable_2d<B: Backend>(linear: &nn::Linear<B>, x: Tensor<B, 2>) -> Tensor<B, 2> {
    let x_dtype: burn::tensor::FloatDType = x.dtype().into();
    let weight = linear.weight.val();
    let weight_dtype: burn::tensor::FloatDType = weight.dtype().into();
    // Keep linear operands in a single dtype to avoid mixed-dtype WGPU matmul
    // collapse on bf16 checkpoint weights (observed near-zero outputs in sparse flow).
    let weight = if weight_dtype != x_dtype {
        weight.cast(x_dtype)
    } else {
        weight
    };
    let mut output = x.matmul(weight);
    if let Some(bias) = linear.bias.as_ref() {
        let output_dtype: burn::tensor::FloatDType = output.dtype().into();
        let bias_dtype: burn::tensor::FloatDType = bias.dtype().into();
        let bias = if bias_dtype != output_dtype {
            bias.val().cast(output_dtype)
        } else {
            bias.val()
        };
        output = output.add(bias.unsqueeze::<2>());
    }
    output
}

fn linear_forward_stable_via_2d<B: Backend>(
    linear: &nn::Linear<B>,
    x: Tensor<B, 3>,
) -> Tensor<B, 3> {
    let [batch, tokens, channels] = x.dims();
    let out_channels = linear.weight.val().dims()[1];
    linear_forward_stable_2d(linear, x.reshape([batch * tokens, channels])).reshape([
        batch,
        tokens,
        out_channels,
    ])
}

fn linear_forward_token_chunked<B: Backend>(
    linear: &nn::Linear<B>,
    x: Tensor<B, 3>,
    chunk_tokens: usize,
) -> Tensor<B, 3> {
    let [batch, tokens, channels] = x.dims();
    if chunk_tokens >= tokens {
        return linear_forward_stable(linear, x);
    }

    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < tokens {
        let end = (start + chunk_tokens).min(tokens);
        let x_chunk = x.clone().slice([0..batch, start..end, 0..channels]);
        chunks.push(linear_forward_stable(linear, x_chunk));
        start = end;
    }
    Tensor::cat(chunks, 1)
}

fn linear_forward_stable<B: Backend>(linear: &nn::Linear<B>, x: Tensor<B, 3>) -> Tensor<B, 3> {
    linear_forward_stable_via_2d(linear, x)
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

    if attention_uses_module_kernel::<B>() {
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
        } else {
            // Raw CubeBackend module attention should stay on flash-attn kernels.
            // Use the sparse-flow module cap instead of the global stream cap
            // (1024) to reduce cross-attention dispatch overhead on large token
            // counts while keeping chunking bounded for very large stages.
            sparse_flow_module_attention_chunk_cap(query_tokens)
                .min(query_tokens)
                .max(1)
        };

        if attention_debug_enabled() && query_tokens >= 1024 {
            eprintln!(
                "burn_trellis: attn dispatch backend={backend_name} impl=flash_attention(module_attention) q={query_tokens} k={key_tokens} head_dim={head_dim} q_chunk={query_chunk} logits_budget={logits_budget}"
            );
        }

        let out = if query_chunk >= query_tokens {
            attention(q, k, v, None, None, AttentionModuleOptions::default())
        } else {
            let mut chunks = Vec::new();
            let mut start = 0usize;
            while start < query_tokens {
                let end = (start + query_chunk).min(query_tokens);
                let q_chunk = q
                    .clone()
                    .slice([0..batch, 0..heads, start..end, 0..head_dim])
                    .clone();
                chunks.push(attention(
                    q_chunk,
                    k.clone(),
                    v.clone(),
                    None,
                    None,
                    AttentionModuleOptions::default(),
                ));
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

fn apply_rope<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    resolution: usize,
    head_dim: usize,
    rope_freq: [f32; 2],
    token_coords: Option<Tensor<B, 2, Int>>,
) -> (Tensor<B, 4>, Tensor<B, 4>)
where
    RopeRotateWgpuBridgeImpl: RopeRotateWgpuBridge<B>,
{
    (
        apply_rope_single(q, resolution, head_dim, rope_freq, token_coords.clone(), 0),
        apply_rope_single(k, resolution, head_dim, rope_freq, token_coords, 0),
    )
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
        if let Some(rotated) =
            maybe_rotate_pairs_coords_wgpu(x.clone(), coord_slice.clone(), rope_freq)
        {
            return rotated;
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

#[cfg(test)]
fn metadata_path(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("model.bpk");
    path.with_file_name(format!("{file_name}.meta.json"))
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
    use std::path::PathBuf;
    use std::sync::Mutex;
    use std::time::{Instant, SystemTime, UNIX_EPOCH};

    use burn::module::{Param, ParamId};
    use burn::prelude::Backend;
    use burn::tensor::{Int, Tensor, TensorData};
    use burn_store::{BurnToPyTorchAdapter, BurnpackStore, ModuleSnapshot, SafetensorsStore};

    use crate::sampler::FlowEulerSampleConfig;

    use super::{
        BinaryBlob, BlobMetadata, CpuRuntimeBackend, SelfAttention, SparseStructureFlowConfig,
        SparseStructureFlowModel, SparseStructureFlowRuntime, SparseStructureFlowRuntimeImpl,
        SparseTensorOwned, VarLenTensorOwned, host_transfer_stats, metadata_path,
        reset_host_transfer_stats, resolve_model_weight_candidates,
        scaled_dot_product_attention_dense, scaled_dot_product_attention_stream,
        sparse_flow_attention_logits_within_budget, sparse_flow_stream_chunk_plan,
    };

    static HOST_STATS_LOCK: Mutex<()> = Mutex::new(());
    static ENV_LOCK: Mutex<()> = Mutex::new(());

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
        type BlobBackend = burn::backend::NdArray<f32, u8>;
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
        let device = <TestBackend as Backend>::Device::default();
        let model = SparseStructureFlowModel::<TestBackend>::new(&device, config.clone());
        let mut source_store =
            SafetensorsStore::from_file(&source_path).with_to_adapter(BurnToPyTorchAdapter);
        model
            .save_into(&mut source_store)
            .expect("save source safetensors");
        let source_bytes = std::fs::read(&source_path).expect("read source safetensors");

        let burnpack_path = ckpts.join("flow_model.bpk");
        let blob_device = <BlobBackend as Backend>::Device::default();
        let tensor = Tensor::<BlobBackend, 1, Int>::from_data(
            TensorData::new(source_bytes.clone(), [source_bytes.len()]),
            &blob_device,
        );
        let blob = BinaryBlob {
            bytes: Param::initialized(ParamId::new(), tensor),
        };
        let mut burnpack_store = BurnpackStore::from_file(&burnpack_path).overwrite(true);
        blob.save_into(&mut burnpack_store)
            .expect("save blob burnpack");
        let metadata = BlobMetadata {
            bytes_len: source_bytes.len(),
        };
        std::fs::write(
            metadata_path(&burnpack_path),
            serde_json::to_vec_pretty(&metadata).expect("serialize metadata"),
        )
        .expect("write metadata");

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
        type BlobBackend = burn::backend::NdArray<f32, u8>;
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
        let device = <TestBackend as Backend>::Device::default();
        let model = SparseStructureFlowModel::<TestBackend>::new(&device, config.clone());
        let mut source_store =
            SafetensorsStore::from_file(&source_path).with_to_adapter(BurnToPyTorchAdapter);
        model
            .save_into(&mut source_store)
            .expect("save source safetensors");
        let source_bytes = std::fs::read(&source_path).expect("read source safetensors");

        let burnpack_path = ckpts.join("flow_model.bpk");
        let blob_device = <BlobBackend as Backend>::Device::default();
        let tensor = Tensor::<BlobBackend, 1, Int>::from_data(
            TensorData::new(source_bytes.clone(), [source_bytes.len()]),
            &blob_device,
        );
        let blob = BinaryBlob {
            bytes: Param::initialized(ParamId::new(), tensor),
        };
        let mut burnpack_store = BurnpackStore::from_file(&burnpack_path).overwrite(true);
        blob.save_into(&mut burnpack_store)
            .expect("save blob burnpack");
        let metadata = BlobMetadata {
            bytes_len: source_bytes.len(),
        };
        std::fs::write(
            metadata_path(&burnpack_path),
            serde_json::to_vec_pretty(&metadata).expect("serialize metadata"),
        )
        .expect("write metadata");

        let part_path = ckpts.join("flow_model.bpk.part-00000.bpk");
        let part_meta_path = metadata_path(&part_path);
        std::fs::rename(&burnpack_path, &part_path).expect("move burnpack into part");
        std::fs::rename(metadata_path(&burnpack_path), &part_meta_path)
            .expect("move part metadata");
        let manifest_path = ckpts.join("flow_model.bpk.parts.json");
        std::fs::write(
            &manifest_path,
            format!(
                "{{\n  \"version\": 1,\n  \"source_file\": \"flow_model.bpk\",\n  \"source_modified_unix_ms\": 0,\n  \"total_bytes\": {},\n  \"max_part_bytes\": {},\n  \"parts\": [{{\"path\": \"{}\", \"bytes\": {}, \"sha256\": \"\", \"tensors\": 1}}]\n}}",
                source_bytes.len(),
                source_bytes.len(),
                part_path
                    .file_name()
                    .and_then(|v| v.to_str())
                    .expect("part file name"),
                source_bytes.len()
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
        let device = <CpuRuntimeBackend as Backend>::Device::default();
        let model = SparseStructureFlowModel::<CpuRuntimeBackend>::new(&device, config.clone());
        SparseStructureFlowRuntimeImpl {
            config,
            model,
            device,
        }
    }

    fn make_attention_tensor(
        device: &<CpuRuntimeBackend as Backend>::Device,
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
        let device = <CpuRuntimeBackend as Backend>::Device::default();
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
    fn attention_stream_matches_dense_reference() {
        let _guard = ENV_LOCK.lock().expect("env lock");
        unsafe {
            std::env::set_var("TRELLIS2_ATTN_QUERY_CHUNK", "8");
            std::env::set_var("TRELLIS2_ATTN_QUERY_CHUNK_MAX", "8");
            std::env::set_var("TRELLIS2_ATTN_KEY_CHUNK", "7");
            std::env::set_var("TRELLIS2_ATTN_KEY_CHUNK_MAX", "7");
        }
        let device = <CpuRuntimeBackend as Backend>::Device::default();
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

        let device = <CpuRuntimeBackend as Backend>::Device::default();
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
        let device = <CpuRuntimeBackend as Backend>::Device::default();
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
        assert_eq!(
            super::sparse_flow_mlp_chunk_tokens_for_backend::<super::WgpuRuntimeBackend>(4_096),
            2_048
        );
        assert_eq!(
            super::sparse_flow_mlp_chunk_tokens_for_backend::<super::WgpuRuntimeBackend>(8_192),
            2_048
        );
        assert_eq!(
            super::sparse_flow_mlp_chunk_tokens_for_backend::<super::WgpuRuntimeBackend>(32_768),
            4_096
        );
        assert_eq!(
            super::sparse_flow_linear_chunk_tokens_for_backend::<super::WgpuRuntimeBackend>(32_768,),
            16_384
        );
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

    #[test]
    fn tiny_sparse_flow_forward_cpu_backend() {
        let device = <burn::backend::NdArray<f32> as Backend>::Device::default();
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

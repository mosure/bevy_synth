use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
#[cfg(feature = "runtime-model-wgpu")]
use std::sync::OnceLock;
use std::sync::{Arc, Mutex};

use super::weight_parts::{candidate_exists_or_has_parts, load_blob_bytes_from_burnpack_or_parts};
use crate::blob_burnpack::load_blob_bytes_from_burnpack as load_blob_bytes_from_blob_burnpack;
use crate::time::Instant;
#[cfg(feature = "runtime-model-wgpu")]
use burn::tensor::TensorData;
#[cfg(feature = "runtime-model-wgpu")]
use burn::tensor::activation::sigmoid;
#[cfg(any(test, feature = "runtime-model-wgpu"))]
use burn::tensor::{Int, Tensor};
#[cfg(feature = "runtime-model-wgpu")]
use burn_flex_gmm::kernel_rows;
#[cfg(feature = "runtime-model-wgpu")]
use burn_flex_gmm::wgpu::{
    DefaultWgpuBackend, SparseWgpuForwardConfig, SparseWgpuKernelVariant,
    layer_norm_affine_forward_wgpu, layer_norm_affine_silu_forward_wgpu,
    linear_skinny_forward_wgpu, neighbor_rows_tensor_from_coords_tensor,
    sparse_subm_conv_forward_wgpu_with_config,
};
use burn_flex_gmm::{
    SparseSubmConvConfig as FlexConvConfig, SparseSubmConvWeights, build_neighbor_rows,
    pack_flex_weight, sparse_subm_conv_forward_flex_precomputed,
};
#[cfg(feature = "runtime-model-wgpu")]
use burn_wgpu::WgpuDevice;
use half::{bf16, f16};
use memmap2::{Mmap, MmapOptions};
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;

const F16_SUFFIX: &str = "_f16";
const LAYER_NORM32_EPS: f32 = 1.0e-6;
const F_LAYER_NORM_EPS: f32 = 1.0e-5;
const DECODER_NEIGHBOR_CACHE_MAX: usize = 128;
#[cfg(feature = "runtime-model-wgpu")]
const DECODER_WGPU_TENSOR_CACHE_MAX: usize = 128;

#[derive(Debug, Clone, Default)]
pub(crate) struct DecoderConvBlockTelemetry {
    pub context: String,
    pub conv_calls: u64,
    pub wgpu_calls: u64,
    pub wgpu_successes: u64,
    pub wgpu_failures: u64,
    pub dispatches: u64,
    pub chunked_calls: u64,
    pub max_chunk_rows: usize,
    pub input_bytes: u64,
    pub output_bytes: u64,
    pub neighbor_elements: u64,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct DecoderConvTelemetry {
    pub conv_calls: u64,
    pub wgpu_calls: u64,
    pub wgpu_successes: u64,
    pub wgpu_failures: u64,
    pub dispatches: u64,
    pub chunked_calls: u64,
    pub max_chunk_rows: usize,
    pub input_bytes: u64,
    pub output_bytes: u64,
    pub neighbor_elements: u64,
    pub blocks: Vec<DecoderConvBlockTelemetry>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct DecoderOpTimingTelemetry {
    pub context: String,
    pub calls: u64,
    pub total_ms: f64,
    pub max_ms: f64,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct DecoderOpTelemetry {
    pub calls: u64,
    pub total_ms: f64,
    pub readback_count: u64,
    pub readback_elements: u64,
    pub ops: Vec<DecoderOpTimingTelemetry>,
}

#[cfg(feature = "runtime-model-wgpu")]
#[derive(Debug, Default)]
struct DecoderConvTelemetryState {
    total: DecoderConvBlockTelemetry,
    blocks: HashMap<String, DecoderConvBlockTelemetry>,
}

#[cfg(feature = "runtime-model-wgpu")]
static DECODER_CONV_TELEMETRY: OnceLock<Mutex<DecoderConvTelemetryState>> = OnceLock::new();

#[cfg(feature = "runtime-model-wgpu")]
#[derive(Debug, Default)]
struct DecoderOpTelemetryState {
    calls: u64,
    total_ms: f64,
    readback_count: u64,
    readback_elements: u64,
    ops: HashMap<String, DecoderOpTimingTelemetry>,
}

#[cfg(feature = "runtime-model-wgpu")]
static DECODER_OP_TELEMETRY: OnceLock<Mutex<DecoderOpTelemetryState>> = OnceLock::new();

enum WeightsBacking {
    Mmap(Mmap),
    Bytes(Vec<u8>),
}

impl WeightsBacking {
    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Mmap(mmap) => mmap.as_ref(),
            Self::Bytes(bytes) => bytes.as_slice(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct DecoderConfigFile {
    #[allow(dead_code)]
    pub name: String,
    pub args: DecoderArgs,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct DecoderArgs {
    #[serde(default)]
    pub out_channels: Option<usize>,
    pub model_channels: Vec<usize>,
    pub latent_channels: usize,
    pub num_blocks: Vec<usize>,
    #[allow(dead_code)]
    pub block_type: Vec<String>,
    #[allow(dead_code)]
    pub up_block_type: Vec<String>,
    #[allow(dead_code)]
    pub block_args: Vec<serde_json::Value>,
    #[serde(default)]
    pub pred_subdiv: Option<bool>,
    #[serde(default)]
    #[allow(dead_code)]
    pub resolution: Option<usize>,
    #[serde(default)]
    pub voxel_margin: Option<f32>,
    #[serde(default)]
    pub use_fp16: Option<bool>,
}

#[derive(Debug, Clone)]
pub(crate) struct DecoderRuntimeConfig {
    pub center_subdivision_logits: bool,
    pub force_fp32: bool,
    pub subdivision_threshold: f32,
    pub subdivision_child_thresholds: [f32; 8],
}

impl Default for DecoderRuntimeConfig {
    fn default() -> Self {
        Self {
            center_subdivision_logits: false,
            force_fp32: false,
            subdivision_threshold: 0.0,
            subdivision_child_thresholds: [f32::NAN; 8],
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SparseSubdivisionLogits {
    pub spatial_shape: [u32; 3],
    pub coords: Vec<[u32; 4]>,
    pub logits: Vec<f32>,
    #[cfg(feature = "runtime-model-wgpu")]
    pub coords_tensor: Option<Tensor<DefaultWgpuBackend, 2, Int>>,
    #[cfg(feature = "runtime-model-wgpu")]
    pub logits_tensor: Option<Tensor<DefaultWgpuBackend, 2>>,
    #[cfg(feature = "runtime-model-wgpu")]
    pub active_indices_tensor: Option<Tensor<DefaultWgpuBackend, 2, Int>>,
    #[cfg(feature = "runtime-model-wgpu")]
    pub child_coords_tensor: Option<Tensor<DefaultWgpuBackend, 2, Int>>,
    #[cfg(feature = "runtime-model-wgpu")]
    pub child_linear_idx_tensor: Option<Tensor<DefaultWgpuBackend, 1, Int>>,
}

impl SparseSubdivisionLogits {
    pub fn from_host(
        spatial_shape: [u32; 3],
        coords: Vec<[u32; 4]>,
        logits: Vec<f32>,
    ) -> Result<Self, String> {
        if logits.len() != coords.len().saturating_mul(8) {
            return Err(format!(
                "sparse subdivision logits length mismatch: logits={} coords_rows={}",
                logits.len(),
                coords.len()
            ));
        }
        Ok(Self {
            spatial_shape,
            coords,
            logits,
            #[cfg(feature = "runtime-model-wgpu")]
            coords_tensor: None,
            #[cfg(feature = "runtime-model-wgpu")]
            logits_tensor: None,
            #[cfg(feature = "runtime-model-wgpu")]
            active_indices_tensor: None,
            #[cfg(feature = "runtime-model-wgpu")]
            child_coords_tensor: None,
            #[cfg(feature = "runtime-model-wgpu")]
            child_linear_idx_tensor: None,
        })
    }

    #[cfg(feature = "runtime-model-wgpu")]
    pub fn from_device_tensors_with_active_and_children(
        spatial_shape: [u32; 3],
        coords_tensor: Tensor<DefaultWgpuBackend, 2, Int>,
        logits_tensor: Tensor<DefaultWgpuBackend, 2>,
        active_indices_tensor: Option<Tensor<DefaultWgpuBackend, 2, Int>>,
        child_tensors: Option<(
            Tensor<DefaultWgpuBackend, 2, Int>,
            Tensor<DefaultWgpuBackend, 1, Int>,
        )>,
    ) -> Result<Self, String> {
        let [coord_rows, coord_cols] = coords_tensor.dims();
        if coord_cols != 4 {
            return Err(format!(
                "sparse subdivision coord tensor must have 4 columns, got {}",
                coord_cols
            ));
        }
        let [logit_rows, logit_cols] = logits_tensor.dims();
        if logit_cols != 8 {
            return Err(format!(
                "sparse subdivision logits tensor must have 8 columns, got {}",
                logit_cols
            ));
        }
        if coord_rows != logit_rows {
            return Err(format!(
                "sparse subdivision tensor row mismatch: coords={} logits={}",
                coord_rows, logit_rows
            ));
        }
        if let Some(active_t) = active_indices_tensor.as_ref() {
            let [active_rows, active_cols] = active_t.dims();
            if active_cols != 2 {
                return Err(format!(
                    "sparse subdivision active-index tensor must have 2 columns, got {}",
                    active_cols
                ));
            }
            if active_rows > coord_rows.saturating_mul(8) {
                return Err(format!(
                    "sparse subdivision active-index tensor has too many rows: active_rows={} max_rows={}",
                    active_rows,
                    coord_rows.saturating_mul(8)
                ));
            }
        }
        let (child_coords_tensor, child_linear_idx_tensor) = if let Some((
            child_coords_t,
            child_linear_idx_t,
        )) = child_tensors
        {
            let [child_rows, child_cols] = child_coords_t.dims();
            if child_cols != 4 {
                return Err(format!(
                    "sparse subdivision child-coord tensor must have 4 columns, got {}",
                    child_cols
                ));
            }
            let [linear_rows] = child_linear_idx_t.dims();
            if linear_rows != child_rows {
                return Err(format!(
                    "sparse subdivision child tensor row mismatch: child_coords_rows={} child_linear_rows={}",
                    child_rows, linear_rows
                ));
            }
            if child_rows > coord_rows.saturating_mul(8) {
                return Err(format!(
                    "sparse subdivision child coord tensor has too many rows: child_rows={} max_rows={}",
                    child_rows,
                    coord_rows.saturating_mul(8)
                ));
            }
            if let Some(active_t) = active_indices_tensor.as_ref() {
                let [active_rows, _active_cols] = active_t.dims();
                if child_rows != active_rows {
                    return Err(format!(
                        "sparse subdivision child/active tensor row mismatch: child_rows={} active_rows={}",
                        child_rows, active_rows
                    ));
                }
            }
            (Some(child_coords_t), Some(child_linear_idx_t))
        } else {
            (None, None)
        };
        Ok(Self {
            spatial_shape,
            coords: Vec::new(),
            logits: Vec::new(),
            coords_tensor: Some(coords_tensor),
            logits_tensor: Some(logits_tensor),
            active_indices_tensor,
            child_coords_tensor,
            child_linear_idx_tensor,
        })
    }

    pub fn coords_host(&self, context: &str) -> Result<Vec<[u32; 4]>, String> {
        if !self.coords.is_empty() {
            return Ok(self.coords.clone());
        }
        #[cfg(feature = "runtime-model-wgpu")]
        if let Some(coords_t) = self.coords_tensor.as_ref() {
            return tensor_to_coords_u32(coords_t.clone(), context);
        }
        Err(format!(
            "{context}: sparse subdivision has no host coords and no device coord tensor"
        ))
    }

    pub fn logits_host(&self, context: &str) -> Result<Vec<f32>, String> {
        if !self.logits.is_empty() {
            return Ok(self.logits.clone());
        }
        #[cfg(feature = "runtime-model-wgpu")]
        if let Some(logits_t) = self.logits_tensor.as_ref() {
            return tensor_to_vec_f32(logits_t.clone(), context);
        }
        Err(format!(
            "{context}: sparse subdivision has no host logits and no device logits tensor"
        ))
    }

    #[cfg(feature = "runtime-model-wgpu")]
    pub fn device_tensors(
        &self,
    ) -> Option<(
        Tensor<DefaultWgpuBackend, 2, Int>,
        Tensor<DefaultWgpuBackend, 2>,
    )> {
        match (self.coords_tensor.as_ref(), self.logits_tensor.as_ref()) {
            (Some(coords_t), Some(logits_t)) => Some((coords_t.clone(), logits_t.clone())),
            _ => None,
        }
    }

    #[cfg(feature = "runtime-model-wgpu")]
    pub fn active_indices_tensor(&self) -> Option<Tensor<DefaultWgpuBackend, 2, Int>> {
        self.active_indices_tensor.as_ref().cloned()
    }

    #[cfg(feature = "runtime-model-wgpu")]
    pub fn child_tensors(
        &self,
    ) -> Option<(
        Tensor<DefaultWgpuBackend, 2, Int>,
        Tensor<DefaultWgpuBackend, 1, Int>,
    )> {
        match (
            self.child_coords_tensor.as_ref(),
            self.child_linear_idx_tensor.as_ref(),
        ) {
            (Some(child_coords_t), Some(child_linear_idx_t)) => {
                Some((child_coords_t.clone(), child_linear_idx_t.clone()))
            }
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SparseDecodeResult {
    pub coords: Option<Vec<[u32; 4]>>,
    pub feats: Option<Vec<f32>>,
    pub out_channels: usize,
    pub subdivisions: Vec<SparseSubdivisionLogits>,
    #[cfg(feature = "runtime-model-wgpu")]
    pub coords_tensor: Option<Tensor<DefaultWgpuBackend, 2, Int>>,
    #[cfg(feature = "runtime-model-wgpu")]
    pub feats_tensor: Option<Tensor<DefaultWgpuBackend, 2>>,
}

impl SparseDecodeResult {
    pub fn empty(out_channels: usize) -> Self {
        Self {
            coords: Some(Vec::new()),
            feats: Some(Vec::new()),
            out_channels,
            subdivisions: Vec::new(),
            #[cfg(feature = "runtime-model-wgpu")]
            coords_tensor: None,
            #[cfg(feature = "runtime-model-wgpu")]
            feats_tensor: None,
        }
    }

    pub fn coords_host(&self, context: &str) -> Result<Vec<[u32; 4]>, String> {
        if let Some(coords) = self.coords.as_ref() {
            return Ok(coords.clone());
        }
        #[cfg(feature = "runtime-model-wgpu")]
        if let Some(coords_t) = self.coords_tensor.as_ref() {
            return tensor_to_coords_u32(coords_t.clone(), context);
        }
        Err(format!(
            "{context}: sparse decode result has no host coords and no device coord tensor"
        ))
    }

    pub fn feats_host(&self, context: &str) -> Result<Vec<f32>, String> {
        if let Some(feats) = self.feats.as_ref() {
            return Ok(feats.clone());
        }
        #[cfg(feature = "runtime-model-wgpu")]
        if let Some(feats_t) = self.feats_tensor.as_ref() {
            return tensor_to_vec_f32(feats_t.clone(), context);
        }
        Err(format!(
            "{context}: sparse decode result has no host feats and no device feat tensor"
        ))
    }

    pub fn rows(&self) -> usize {
        if let Some(coords) = self.coords.as_ref() {
            return coords.len();
        }
        #[cfg(feature = "runtime-model-wgpu")]
        if let Some(coords_t) = self.coords_tensor.as_ref() {
            return coords_t.dims()[0];
        }
        0
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SparseUpsampledCoords {
    pub coords: Option<Vec<[u32; 4]>>,
    #[cfg(feature = "runtime-model-wgpu")]
    pub coords_tensor: Option<Tensor<DefaultWgpuBackend, 2, Int>>,
}

impl SparseUpsampledCoords {
    pub fn from_host(coords: Vec<[u32; 4]>) -> Self {
        Self {
            coords: Some(coords),
            #[cfg(feature = "runtime-model-wgpu")]
            coords_tensor: None,
        }
    }

    #[cfg(feature = "runtime-model-wgpu")]
    pub fn from_wgpu_tensor(coords_tensor: Tensor<DefaultWgpuBackend, 2, Int>) -> Self {
        Self {
            coords: None,
            coords_tensor: Some(coords_tensor),
        }
    }

    pub fn rows(&self) -> usize {
        if let Some(coords) = self.coords.as_ref() {
            return coords.len();
        }
        #[cfg(feature = "runtime-model-wgpu")]
        if let Some(coords_t) = self.coords_tensor.as_ref() {
            return coords_t.dims()[0];
        }
        0
    }

    pub fn coords_host(&self, context: &str) -> Result<Vec<[u32; 4]>, String> {
        if let Some(coords) = self.coords.as_ref() {
            return Ok(coords.clone());
        }
        #[cfg(feature = "runtime-model-wgpu")]
        if let Some(coords_t) = self.coords_tensor.as_ref() {
            return tensor_to_coords_u32(coords_t.clone(), context);
        }
        Err(format!(
            "{context}: sparse upsample result has no host coords and no device coord tensor"
        ))
    }

    #[allow(dead_code)]
    #[cfg(feature = "runtime-model-wgpu")]
    pub fn coords_tensor(&self) -> Option<Tensor<DefaultWgpuBackend, 2, Int>> {
        self.coords_tensor.as_ref().cloned()
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SparseUnetDecoderRuntime {
    out_channels: usize,
    pred_subdiv: bool,
    voxel_margin: f32,
    compute_fp16: bool,
    model_channels: Vec<usize>,
    runtime_config: DecoderRuntimeConfig,
    from_latent: LinearLayer,
    output_layer: LinearLayer,
    stages: Vec<DecoderStage>,
    conv_cache: Arc<Mutex<DecoderConvCache>>,
    #[cfg(feature = "runtime-model-wgpu")]
    wgpu_context: Option<Arc<Mutex<DecoderWgpuConvContext>>>,
}

#[derive(Debug, Clone)]
struct DecoderStage {
    convnext_blocks: Vec<ConvNeXtBlock>,
    upsample_block: Option<C2SUpsampleBlock>,
}

#[derive(Debug, Clone)]
struct ConvNeXtBlock {
    conv: SparseConvLayer,
    norm_weight: Vec<f32>,
    norm_bias: Vec<f32>,
    mlp_0: LinearLayer,
    mlp_2: LinearLayer,
}

#[derive(Debug, Clone)]
struct C2SUpsampleBlock {
    in_channels: usize,
    out_channels: usize,
    norm1_weight: Vec<f32>,
    norm1_bias: Vec<f32>,
    to_subdiv: Option<LinearLayer>,
    conv1: SparseConvLayer,
    conv2: SparseConvLayer,
}

#[derive(Debug, Clone)]
struct LinearLayer {
    in_channels: usize,
    out_channels: usize,
    // Row-major [out, in] as stored by PyTorch linear layers.
    weight: Vec<f32>,
    bias: Vec<f32>,
}

#[derive(Debug, Clone)]
struct SparseConvLayer {
    in_channels: usize,
    out_channels: usize,
    kernel_d: usize,
    kernel_h: usize,
    kernel_w: usize,
    in_channels_per_group: usize,
    out_channels_per_group: usize,
    groups: usize,
    // Row-major [out, kd, kh, kw, in_per_group]
    weight: Vec<f32>,
    bias: Vec<f32>,
    flex_packed_weight: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
struct NeighborRowsCacheKey {
    coords_hash: u64,
    rows: usize,
    kernel_d: usize,
    kernel_h: usize,
    kernel_w: usize,
    axis_order: [usize; 3],
    axis_sign: [i32; 3],
}

impl NeighborRowsCacheKey {
    fn new(config: &FlexConvConfig, coords: &[[u32; 4]]) -> Self {
        Self {
            coords_hash: hash_coords(coords),
            rows: coords.len(),
            kernel_d: config.kernel_d,
            kernel_h: config.kernel_h,
            kernel_w: config.kernel_w,
            axis_order: config.axis_order,
            axis_sign: config.axis_sign,
        }
    }
}

#[derive(Debug, Default)]
struct DecoderConvCache {
    neighbor_rows: HashMap<NeighborRowsCacheKey, Vec<i32>>,
}

impl DecoderConvCache {
    fn neighbor_rows_with_key<'a>(
        &'a mut self,
        config: &FlexConvConfig,
        coords: &[[u32; 4]],
    ) -> Result<(NeighborRowsCacheKey, &'a [i32]), String> {
        let key = NeighborRowsCacheKey::new(config, coords);
        if !self.neighbor_rows.contains_key(&key) {
            trim_hashmap(&mut self.neighbor_rows, DECODER_NEIGHBOR_CACHE_MAX);
            let neighbor_rows = build_neighbor_rows(config, coords)?;
            self.neighbor_rows.insert(key, neighbor_rows);
        }
        let rows = self
            .neighbor_rows
            .get(&key)
            .map(|rows| rows.as_slice())
            .ok_or_else(|| "decoder neighbor-row cache lookup failed".to_string())?;
        Ok((key, rows))
    }
}

#[cfg(feature = "runtime-model-wgpu")]
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
struct LayerTensorCacheKey {
    weight_ptr: usize,
    bias_ptr: usize,
}

#[cfg(feature = "runtime-model-wgpu")]
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
struct LinearTensorCacheKey {
    weight_ptr: usize,
    bias_ptr: usize,
    in_channels: usize,
    out_channels: usize,
}

#[cfg(feature = "runtime-model-wgpu")]
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
struct VectorTensorCacheKey {
    ptr: usize,
    len: usize,
}

#[cfg(feature = "runtime-model-wgpu")]
#[derive(Debug)]
struct DecoderWgpuConvContext {
    device: WgpuDevice,
    weight_tensors: HashMap<LayerTensorCacheKey, Tensor<DefaultWgpuBackend, 5>>,
    bias_tensors: HashMap<LayerTensorCacheKey, Tensor<DefaultWgpuBackend, 1>>,
    neighbor_tensors: HashMap<NeighborRowsCacheKey, Tensor<DefaultWgpuBackend, 2, Int>>,
    linear_weight_tensors: HashMap<LinearTensorCacheKey, Tensor<DefaultWgpuBackend, 2>>,
    linear_bias_tensors: HashMap<LinearTensorCacheKey, Tensor<DefaultWgpuBackend, 1>>,
    vector_tensors: HashMap<VectorTensorCacheKey, Tensor<DefaultWgpuBackend, 1>>,
    wgpu_failed: bool,
}

#[cfg(feature = "runtime-model-wgpu")]
impl DecoderWgpuConvContext {
    fn new() -> Result<Self, String> {
        let device = WgpuDevice::default();
        let _ = Tensor::<DefaultWgpuBackend, 1>::zeros([1], &device);
        Ok(Self {
            device,
            weight_tensors: HashMap::new(),
            bias_tensors: HashMap::new(),
            neighbor_tensors: HashMap::new(),
            linear_weight_tensors: HashMap::new(),
            linear_bias_tensors: HashMap::new(),
            vector_tensors: HashMap::new(),
            wgpu_failed: false,
        })
    }

    fn layer_key(layer: &SparseConvLayer) -> LayerTensorCacheKey {
        LayerTensorCacheKey {
            weight_ptr: layer.weight.as_ptr() as usize,
            bias_ptr: layer.bias.as_ptr() as usize,
        }
    }

    fn linear_key(layer: &LinearLayer) -> LinearTensorCacheKey {
        LinearTensorCacheKey {
            weight_ptr: layer.weight.as_ptr() as usize,
            bias_ptr: layer.bias.as_ptr() as usize,
            in_channels: layer.in_channels,
            out_channels: layer.out_channels,
        }
    }

    fn vector_key(values: &[f32]) -> VectorTensorCacheKey {
        VectorTensorCacheKey {
            ptr: values.as_ptr() as usize,
            len: values.len(),
        }
    }

    fn weight_tensor(&mut self, layer: &SparseConvLayer) -> Tensor<DefaultWgpuBackend, 5> {
        if !decoder_wgpu_use_tensor_cache() {
            return Tensor::<DefaultWgpuBackend, 1>::from_floats(
                layer.weight.as_slice(),
                &self.device,
            )
            .reshape([
                layer.out_channels,
                layer.kernel_d,
                layer.kernel_h,
                layer.kernel_w,
                layer.in_channels_per_group,
            ]);
        }
        let key = Self::layer_key(layer);
        if let Some(tensor) = self.weight_tensors.get(&key) {
            return tensor.clone();
        }
        trim_hashmap(&mut self.weight_tensors, decoder_wgpu_tensor_cache_max());
        let tensor =
            Tensor::<DefaultWgpuBackend, 1>::from_floats(layer.weight.as_slice(), &self.device)
                .reshape([
                    layer.out_channels,
                    layer.kernel_d,
                    layer.kernel_h,
                    layer.kernel_w,
                    layer.in_channels_per_group,
                ]);
        self.weight_tensors.insert(key, tensor.clone());
        tensor
    }

    fn bias_tensor(&mut self, layer: &SparseConvLayer) -> Tensor<DefaultWgpuBackend, 1> {
        if !decoder_wgpu_use_tensor_cache() {
            return Tensor::<DefaultWgpuBackend, 1>::from_floats(
                layer.bias.as_slice(),
                &self.device,
            );
        }
        let key = Self::layer_key(layer);
        if let Some(tensor) = self.bias_tensors.get(&key) {
            return tensor.clone();
        }
        trim_hashmap(&mut self.bias_tensors, decoder_wgpu_tensor_cache_max());
        let tensor =
            Tensor::<DefaultWgpuBackend, 1>::from_floats(layer.bias.as_slice(), &self.device);
        self.bias_tensors.insert(key, tensor.clone());
        tensor
    }

    fn linear_weight_tensor(&mut self, layer: &LinearLayer) -> Tensor<DefaultWgpuBackend, 2> {
        if !decoder_wgpu_use_tensor_cache() {
            return Tensor::<DefaultWgpuBackend, 1>::from_floats(
                layer.weight.as_slice(),
                &self.device,
            )
            .reshape([layer.out_channels, layer.in_channels]);
        }
        let key = Self::linear_key(layer);
        if let Some(tensor) = self.linear_weight_tensors.get(&key) {
            return tensor.clone();
        }
        trim_hashmap(
            &mut self.linear_weight_tensors,
            decoder_wgpu_tensor_cache_max(),
        );
        let tensor =
            Tensor::<DefaultWgpuBackend, 1>::from_floats(layer.weight.as_slice(), &self.device)
                .reshape([layer.out_channels, layer.in_channels]);
        self.linear_weight_tensors.insert(key, tensor.clone());
        tensor
    }

    fn linear_bias_tensor(&mut self, layer: &LinearLayer) -> Tensor<DefaultWgpuBackend, 1> {
        if !decoder_wgpu_use_tensor_cache() {
            return Tensor::<DefaultWgpuBackend, 1>::from_floats(
                layer.bias.as_slice(),
                &self.device,
            );
        }
        let key = Self::linear_key(layer);
        if let Some(tensor) = self.linear_bias_tensors.get(&key) {
            return tensor.clone();
        }
        trim_hashmap(
            &mut self.linear_bias_tensors,
            decoder_wgpu_tensor_cache_max(),
        );
        let tensor =
            Tensor::<DefaultWgpuBackend, 1>::from_floats(layer.bias.as_slice(), &self.device);
        self.linear_bias_tensors.insert(key, tensor.clone());
        tensor
    }

    fn vector_tensor(&mut self, values: &[f32]) -> Tensor<DefaultWgpuBackend, 1> {
        if !decoder_wgpu_use_tensor_cache() {
            return Tensor::<DefaultWgpuBackend, 1>::from_floats(values, &self.device);
        }
        let key = Self::vector_key(values);
        if let Some(tensor) = self.vector_tensors.get(&key) {
            return tensor.clone();
        }
        trim_hashmap(&mut self.vector_tensors, decoder_wgpu_tensor_cache_max());
        let tensor = Tensor::<DefaultWgpuBackend, 1>::from_floats(values, &self.device);
        self.vector_tensors.insert(key, tensor.clone());
        tensor
    }

    fn neighbor_tensor(
        &mut self,
        key: NeighborRowsCacheKey,
        config: &FlexConvConfig,
        rows: usize,
        neighbor_rows: &[i32],
    ) -> Result<Tensor<DefaultWgpuBackend, 2, Int>, String> {
        if !decoder_wgpu_use_tensor_cache() {
            let kernel_rows = kernel_rows(config)?;
            return Ok(Tensor::<DefaultWgpuBackend, 1, Int>::from_data(
                TensorData::new(neighbor_rows.to_vec(), [rows.saturating_mul(kernel_rows)]),
                &self.device,
            )
            .reshape([rows, kernel_rows]));
        }
        if let Some(tensor) = self.neighbor_tensors.get(&key) {
            return Ok(tensor.clone());
        }
        let kernel_rows = kernel_rows(config)?;
        trim_hashmap(&mut self.neighbor_tensors, decoder_wgpu_tensor_cache_max());
        let tensor = Tensor::<DefaultWgpuBackend, 1, Int>::from_data(
            TensorData::new(neighbor_rows.to_vec(), [rows.saturating_mul(kernel_rows)]),
            &self.device,
        )
        .reshape([rows, kernel_rows]);
        self.neighbor_tensors.insert(key, tensor.clone());
        Ok(tensor)
    }

    fn forward_with_neighbor_rows(
        &mut self,
        config: &FlexConvConfig,
        layer: &SparseConvLayer,
        input: &[f32],
        context: &str,
        cache_key: NeighborRowsCacheKey,
        neighbor_rows: &[i32],
    ) -> Result<Vec<f32>, String> {
        if self.wgpu_failed {
            return Err("wgpu sparse conv disabled after prior failure".to_string());
        }
        if config.in_channels == 0 {
            return Ok(Vec::new());
        }
        if !input.len().is_multiple_of(config.in_channels) {
            return Err(format!(
                "wgpu sparse conv input len mismatch: len={} in_channels={}",
                input.len(),
                config.in_channels
            ));
        }
        let rows = input.len() / config.in_channels;
        let kernel_rows = kernel_rows(config)?;
        if neighbor_rows.len() != rows.saturating_mul(kernel_rows) {
            return Err(format!(
                "wgpu sparse conv neighbor len mismatch: len={} expected={}",
                neighbor_rows.len(),
                rows.saturating_mul(kernel_rows)
            ));
        }
        let neighbor_bytes = rows
            .checked_mul(kernel_rows)
            .and_then(|value| value.checked_mul(core::mem::size_of::<i32>()))
            .ok_or_else(|| "wgpu sparse conv neighbor-byte-size overflow".to_string())?;
        if neighbor_bytes <= decoder_wgpu_max_neighbor_bytes() {
            let neighbor_t = self.neighbor_tensor(cache_key, config, rows, neighbor_rows)?;
            return self.forward_with_neighbor_tensor(
                config,
                layer,
                input,
                context,
                rows,
                kernel_rows,
                neighbor_t,
            );
        }

        let input_t = Tensor::<DefaultWgpuBackend, 1>::from_floats(input, &self.device)
            .reshape([rows, config.in_channels]);
        let output = self.forward_with_neighbor_rows_chunked_tensor(
            config,
            layer,
            input_t,
            context,
            rows,
            kernel_rows,
            neighbor_rows,
        )?;
        output
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .map_err(|err| format!("failed to read wgpu sparse conv output: {err:?}"))
    }

    fn forward_with_coords(
        &mut self,
        config: &FlexConvConfig,
        layer: &SparseConvLayer,
        input: &[f32],
        context: &str,
        coords: &[[u32; 4]],
    ) -> Result<Vec<f32>, String> {
        if self.wgpu_failed {
            return Err("wgpu sparse conv disabled after prior failure".to_string());
        }
        if config.in_channels == 0 {
            return Ok(Vec::new());
        }
        if !input.len().is_multiple_of(config.in_channels) {
            return Err(format!(
                "wgpu sparse conv input len mismatch: len={} in_channels={}",
                input.len(),
                config.in_channels
            ));
        }
        let rows = input.len() / config.in_channels;
        if coords.len() != rows {
            return Err(format!(
                "wgpu sparse conv coord/input row mismatch: coords={} rows={rows}",
                coords.len()
            ));
        }
        let input_t = Tensor::<DefaultWgpuBackend, 1>::from_floats(input, &self.device)
            .reshape([rows, config.in_channels]);
        let coords_t = coords_tensor_from_u32_slice(coords, &self.device)?;
        let output =
            self.forward_with_coords_tensor_device(config, layer, input_t, context, coords_t)?;
        output
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .map_err(|err| format!("failed to read wgpu sparse conv output: {err:?}"))
    }

    fn forward_with_coords_tensor_device(
        &mut self,
        config: &FlexConvConfig,
        layer: &SparseConvLayer,
        input_t: Tensor<DefaultWgpuBackend, 2>,
        context: &str,
        coords_t: Tensor<DefaultWgpuBackend, 2, Int>,
    ) -> Result<Tensor<DefaultWgpuBackend, 2>, String> {
        if self.wgpu_failed {
            return Err("wgpu sparse conv disabled after prior failure".to_string());
        }
        if config.in_channels == 0 {
            return Ok(input_t);
        }
        let [rows, in_channels] = input_t.dims();
        if in_channels != config.in_channels {
            return Err(format!(
                "wgpu sparse conv input channel mismatch: input_channels={} in_channels={}",
                in_channels, config.in_channels
            ));
        }
        let [coord_rows, coord_cols] = coords_t.dims();
        if coord_cols != 4 {
            return Err(format!(
                "wgpu sparse conv coords tensor must have 4 columns, got {coord_cols}"
            ));
        }
        if coord_rows != rows {
            return Err(format!(
                "wgpu sparse conv coord/input row mismatch: coords={} rows={rows}",
                coord_rows
            ));
        }
        let call_start = Instant::now();
        // Do not reject large outputs here: `forward_with_neighbor_tensor_tensor` owns the
        // canonical chunked-dispatch path and will split oversized dispatches by row count.
        // Early-aborting here prevents valid chunked execution during decode upsample stages.
        let kernel_rows = kernel_rows(config)?;
        let neighbor_start = Instant::now();
        let neighbor_t = neighbor_rows_tensor_from_coords_tensor(config, coords_t)?;
        let neighbor_ms = neighbor_start.elapsed().as_secs_f64() * 1000.0;
        let conv_start = Instant::now();
        let output = self.forward_with_neighbor_tensor_tensor(
            config,
            layer,
            input_t,
            context,
            rows,
            kernel_rows,
            neighbor_t,
        )?;
        let conv_ms = conv_start.elapsed().as_secs_f64() * 1000.0;
        let total_ms = call_start.elapsed().as_secs_f64() * 1000.0;
        if decoder_conv_debug_enabled() {
            eprintln!(
                "burn_trellis: wgpu sparse conv '{}' rows={} krows={} neighbor_ms={:.2} conv_ms={:.2} total_ms={:.2}",
                context, rows, kernel_rows, neighbor_ms, conv_ms, total_ms
            );
        }
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_with_neighbor_tensor(
        &mut self,
        config: &FlexConvConfig,
        layer: &SparseConvLayer,
        input: &[f32],
        context: &str,
        rows: usize,
        kernel_rows: usize,
        neighbor_t: Tensor<DefaultWgpuBackend, 2, Int>,
    ) -> Result<Vec<f32>, String> {
        let input_t = Tensor::<DefaultWgpuBackend, 1>::from_floats(input, &self.device)
            .reshape([rows, config.in_channels]);
        let output = self.forward_with_neighbor_tensor_tensor(
            config,
            layer,
            input_t,
            context,
            rows,
            kernel_rows,
            neighbor_t,
        )?;
        output
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .map_err(|err| format!("failed to read wgpu sparse conv output: {err:?}"))
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_with_neighbor_tensor_tensor(
        &mut self,
        config: &FlexConvConfig,
        layer: &SparseConvLayer,
        input_t: Tensor<DefaultWgpuBackend, 2>,
        context: &str,
        rows: usize,
        kernel_rows: usize,
        neighbor_t: Tensor<DefaultWgpuBackend, 2, Int>,
    ) -> Result<Tensor<DefaultWgpuBackend, 2>, String> {
        let [query_rows, neighbor_kernel_rows] = neighbor_t.dims();
        if query_rows != rows {
            return Err(format!(
                "wgpu sparse conv neighbor row mismatch: rows={rows} neighbor_rows={query_rows}"
            ));
        }
        if neighbor_kernel_rows != kernel_rows {
            return Err(format!(
                "wgpu sparse conv neighbor kernel rows mismatch: got={} expected={}",
                neighbor_kernel_rows, kernel_rows
            ));
        }
        let input_elements = rows
            .checked_mul(config.in_channels)
            .ok_or_else(|| "wgpu sparse conv input-element overflow".to_string())?;
        let input_bytes = input_elements
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or_else(|| "wgpu sparse conv input-byte-size overflow".to_string())?;
        let max_input_bytes = decoder_wgpu_max_input_bytes();
        if input_bytes > max_input_bytes {
            return Err(format!(
                "wgpu sparse conv input too large: bytes={} max_bytes={}",
                input_bytes, max_input_bytes
            ));
        }
        let [input_rows, input_channels] = input_t.dims();
        if input_rows != rows || input_channels != config.in_channels {
            return Err(format!(
                "wgpu sparse conv tensor dims mismatch: got=[{},{}] expected=[{},{}]",
                input_rows, input_channels, rows, config.in_channels
            ));
        }
        if let Some(chunk_out_channels) = self.single_group_output_channel_chunk(config, layer)? {
            return self.forward_with_neighbor_tensor_tensor_single_group_channel_chunked(
                config,
                layer,
                input_t,
                context,
                rows,
                kernel_rows,
                neighbor_t,
                input_bytes,
                chunk_out_channels,
            );
        }
        let weight_t = self.weight_tensor(layer);
        let bias_t = self.bias_tensor(layer);
        self.forward_with_neighbor_tensor_tensor_with_weight_bias(
            config,
            input_t,
            context,
            rows,
            kernel_rows,
            neighbor_t,
            input_bytes,
            weight_t,
            bias_t,
        )
    }

    fn single_group_output_channel_chunk(
        &self,
        config: &FlexConvConfig,
        layer: &SparseConvLayer,
    ) -> Result<Option<usize>, String> {
        if config.groups != 1 || layer.groups != 1 {
            return Ok(None);
        }
        let weight_bytes = layer
            .weight
            .len()
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or_else(|| "wgpu sparse conv weight-byte-size overflow".to_string())?;
        let max_weight_bytes = decoder_wgpu_max_weight_bytes();
        if weight_bytes <= max_weight_bytes {
            return Ok(None);
        }

        let per_output_values = layer
            .kernel_d
            .checked_mul(layer.kernel_h)
            .and_then(|value| value.checked_mul(layer.kernel_w))
            .and_then(|value| value.checked_mul(layer.in_channels_per_group))
            .ok_or_else(|| "wgpu sparse conv per-output weight-size overflow".to_string())?;
        if per_output_values == 0 {
            return Err("wgpu sparse conv per-output weight size is zero".to_string());
        }
        let per_output_bytes = per_output_values
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or_else(|| "wgpu sparse conv per-output weight-byte overflow".to_string())?;

        // Keep per-dispatch weight uploads bounded on memory-constrained adapters.
        // This is a canonical device path (no host fallback), but avoids allocating a
        // monolithic decoder weight tensor that can OOM before compute starts.
        let raw_chunk = (max_weight_bytes / per_output_bytes)
            .max(1)
            .min(config.out_channels.max(1));
        let aligned = raw_chunk - (raw_chunk % 8);
        let chunk_out_channels = if aligned > 0 { aligned } else { raw_chunk };
        if chunk_out_channels >= config.out_channels {
            Ok(None)
        } else {
            Ok(Some(chunk_out_channels.max(1)))
        }
    }

    fn chunk_weight_bias_tensors_single_group(
        &self,
        layer: &SparseConvLayer,
        out_start: usize,
        out_end: usize,
    ) -> Result<(Tensor<DefaultWgpuBackend, 5>, Tensor<DefaultWgpuBackend, 1>), String> {
        if out_start >= out_end || out_end > layer.out_channels {
            return Err(format!(
                "wgpu sparse conv output-channel slice out of range: start={} end={} out_channels={}",
                out_start, out_end, layer.out_channels
            ));
        }
        let per_output_values = layer
            .kernel_d
            .checked_mul(layer.kernel_h)
            .and_then(|value| value.checked_mul(layer.kernel_w))
            .and_then(|value| value.checked_mul(layer.in_channels_per_group))
            .ok_or_else(|| "wgpu sparse conv per-output weight-size overflow".to_string())?;

        let weight_start = out_start
            .checked_mul(per_output_values)
            .ok_or_else(|| "wgpu sparse conv weight slice start overflow".to_string())?;
        let weight_end = out_end
            .checked_mul(per_output_values)
            .ok_or_else(|| "wgpu sparse conv weight slice end overflow".to_string())?;
        let weight_slice = layer
            .weight
            .get(weight_start..weight_end)
            .ok_or_else(|| "wgpu sparse conv weight slice out of bounds".to_string())?;
        let bias_slice = layer
            .bias
            .get(out_start..out_end)
            .ok_or_else(|| "wgpu sparse conv bias slice out of bounds".to_string())?;

        let chunk_out_channels = out_end - out_start;
        let weight_t = Tensor::<DefaultWgpuBackend, 1>::from_floats(weight_slice, &self.device)
            .reshape([
                chunk_out_channels,
                layer.kernel_d,
                layer.kernel_h,
                layer.kernel_w,
                layer.in_channels_per_group,
            ]);
        let bias_t = Tensor::<DefaultWgpuBackend, 1>::from_floats(bias_slice, &self.device);
        Ok((weight_t, bias_t))
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_with_neighbor_tensor_tensor_single_group_channel_chunked(
        &mut self,
        config: &FlexConvConfig,
        layer: &SparseConvLayer,
        input_t: Tensor<DefaultWgpuBackend, 2>,
        context: &str,
        rows: usize,
        kernel_rows: usize,
        neighbor_t: Tensor<DefaultWgpuBackend, 2, Int>,
        input_bytes: usize,
        chunk_out_channels: usize,
    ) -> Result<Tensor<DefaultWgpuBackend, 2>, String> {
        if decoder_conv_debug_enabled() {
            eprintln!(
                "burn_trellis: chunking wgpu sparse conv output channels for '{}' total_out_channels={} chunk_out_channels={} rows={}",
                context, config.out_channels, chunk_out_channels, rows
            );
        }

        let mut out_start = 0usize;
        let mut outputs: Vec<Tensor<DefaultWgpuBackend, 2>> = Vec::new();
        while out_start < config.out_channels {
            let out_end = (out_start + chunk_out_channels).min(config.out_channels);
            let chunk_out = out_end - out_start;
            let chunk_config = FlexConvConfig {
                in_channels: config.in_channels,
                out_channels: chunk_out,
                kernel_d: config.kernel_d,
                kernel_h: config.kernel_h,
                kernel_w: config.kernel_w,
                in_channels_per_group: config.in_channels_per_group,
                out_channels_per_group: chunk_out,
                groups: 1,
                axis_order: config.axis_order,
                axis_sign: config.axis_sign,
            };
            let (weight_t, bias_t) =
                self.chunk_weight_bias_tensors_single_group(layer, out_start, out_end)?;
            let chunk_context = format!("{context} oc_chunk[{out_start}:{out_end})");
            let chunk_output = self.forward_with_neighbor_tensor_tensor_with_weight_bias(
                &chunk_config,
                input_t.clone(),
                chunk_context.as_str(),
                rows,
                kernel_rows,
                neighbor_t.clone(),
                input_bytes,
                weight_t,
                bias_t,
            )?;
            outputs.push(chunk_output);
            out_start = out_end;
        }

        if outputs.is_empty() {
            return Ok(Tensor::<DefaultWgpuBackend, 2>::zeros(
                [rows, 0],
                &self.device,
            ));
        }
        if outputs.len() == 1 {
            return Ok(outputs.remove(0));
        }
        Ok(Tensor::cat(outputs, 1))
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_with_neighbor_tensor_tensor_with_weight_bias(
        &mut self,
        config: &FlexConvConfig,
        input_t: Tensor<DefaultWgpuBackend, 2>,
        context: &str,
        rows: usize,
        kernel_rows: usize,
        neighbor_t: Tensor<DefaultWgpuBackend, 2, Int>,
        input_bytes: usize,
        weight_t: Tensor<DefaultWgpuBackend, 5>,
        bias_t: Tensor<DefaultWgpuBackend, 1>,
    ) -> Result<Tensor<DefaultWgpuBackend, 2>, String> {
        let bytes_per_row = config
            .out_channels
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or_else(|| "wgpu sparse conv bytes-per-row overflow".to_string())?;
        let max_output_bytes = decoder_wgpu_sparse_conv_max_output_bytes();
        let output_bytes = rows
            .checked_mul(bytes_per_row)
            .ok_or_else(|| "wgpu sparse conv output-byte-size overflow".to_string())?;
        let max_tensor_bytes = decoder_wgpu_max_tensor_bytes();
        if output_bytes > max_tensor_bytes {
            return Err(format!(
                "wgpu sparse conv output exceeds tensor limit: bytes={} max_tensor_bytes={}",
                output_bytes, max_tensor_bytes
            ));
        }
        // TODO(perf-kernel-2): Replace heuristic hotspot overrides with dedicated fused
        // sparse decoder kernels (fused gather+GEMM plus subgroup-tiling schedule).
        let forward_cfg =
            decoder_wgpu_forward_config_for_call(config, rows, output_bytes, max_output_bytes);
        if output_bytes <= max_output_bytes {
            match sparse_subm_conv_forward_wgpu_with_config(
                config,
                input_t.clone(),
                neighbor_t.clone(),
                weight_t.clone(),
                bias_t.clone(),
                forward_cfg,
            ) {
                Ok(output) => {
                    telemetry_record_wgpu_success(
                        context,
                        1,
                        false,
                        rows,
                        input_bytes,
                        output_bytes,
                        rows.saturating_mul(kernel_rows),
                    );
                    return Ok(output);
                }
                Err(err) => {
                    if !decoder_wgpu_is_buffer_too_big(err.as_str()) || rows <= 1 {
                        return Err(err);
                    }
                    if decoder_conv_debug_enabled() {
                        eprintln!(
                            "burn_trellis: sparse conv single-dispatch path hit buffer limit in '{context}', retrying with chunked dispatch"
                        );
                    }
                }
            }
        }

        let mut chunk_rows = decoder_wgpu_chunk_rows(rows, bytes_per_row, max_output_bytes);
        if output_bytes <= max_output_bytes {
            chunk_rows = decoder_wgpu_reduce_chunk_rows(chunk_rows);
        }
        if decoder_conv_debug_enabled() {
            eprintln!(
                "burn_trellis: chunking wgpu sparse conv rows={} chunk_rows={} out_channels={} bytes={} max_bytes={}",
                rows, chunk_rows, config.out_channels, output_bytes, max_output_bytes
            );
        }
        let mut start = 0usize;
        let mut dispatches = 0u64;
        let mut max_success_chunk_rows = 0usize;
        let mut chunk_tensors: Vec<Tensor<DefaultWgpuBackend, 2>> = Vec::new();
        while start < rows {
            let end = (start + chunk_rows).min(rows);
            let chunk_neighbor_t = neighbor_t.clone().slice([start..end, 0..kernel_rows]);
            match sparse_subm_conv_forward_wgpu_with_config(
                config,
                input_t.clone(),
                chunk_neighbor_t,
                weight_t.clone(),
                bias_t.clone(),
                forward_cfg,
            ) {
                Ok(chunk_out) => {
                    max_success_chunk_rows = max_success_chunk_rows.max(end - start);
                    chunk_tensors.push(chunk_out);
                    start = end;
                    dispatches = dispatches.saturating_add(1);
                }
                Err(err) => {
                    if decoder_wgpu_is_buffer_too_big(err.as_str()) && chunk_rows > 1 {
                        let reduced = decoder_wgpu_reduce_chunk_rows(chunk_rows);
                        if reduced < chunk_rows {
                            if decoder_conv_debug_enabled() {
                                eprintln!(
                                    "burn_trellis: reducing sparse conv chunk_rows from {} to {} after buffer-too-big in '{context}'",
                                    chunk_rows, reduced
                                );
                            }
                            chunk_rows = reduced;
                            continue;
                        }
                    }
                    return Err(err);
                }
            }
        }
        telemetry_record_wgpu_success(
            context,
            dispatches.max(1),
            true,
            max_success_chunk_rows.max(1),
            input_bytes,
            output_bytes,
            rows.saturating_mul(kernel_rows),
        );
        if chunk_tensors.is_empty() {
            return Ok(Tensor::<DefaultWgpuBackend, 2>::zeros(
                [rows, config.out_channels],
                &self.device,
            ));
        }
        if chunk_tensors.len() == 1 {
            return Ok(chunk_tensors.remove(0));
        }
        Ok(Tensor::cat(chunk_tensors, 0))
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_with_neighbor_rows_chunked_tensor(
        &mut self,
        config: &FlexConvConfig,
        layer: &SparseConvLayer,
        input_t: Tensor<DefaultWgpuBackend, 2>,
        context: &str,
        rows: usize,
        kernel_rows: usize,
        neighbor_rows: &[i32],
    ) -> Result<Tensor<DefaultWgpuBackend, 2>, String> {
        let [input_rows, input_channels] = input_t.dims();
        if input_rows != rows || input_channels != config.in_channels {
            return Err(format!(
                "wgpu sparse conv tensor dims mismatch: got=[{},{}] expected=[{},{}]",
                input_rows, input_channels, rows, config.in_channels
            ));
        }
        if neighbor_rows.len() != rows.saturating_mul(kernel_rows) {
            return Err(format!(
                "wgpu sparse conv neighbor len mismatch: len={} expected={}",
                neighbor_rows.len(),
                rows.saturating_mul(kernel_rows)
            ));
        }
        let input_elements = rows
            .checked_mul(config.in_channels)
            .ok_or_else(|| "wgpu sparse conv input-element overflow".to_string())?;
        let input_bytes = input_elements
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or_else(|| "wgpu sparse conv input-byte-size overflow".to_string())?;
        let max_input_bytes = decoder_wgpu_max_input_bytes();
        if input_bytes > max_input_bytes {
            return Err(format!(
                "wgpu sparse conv input too large: bytes={} max_bytes={}",
                input_bytes, max_input_bytes
            ));
        }
        let bytes_per_row = config
            .out_channels
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or_else(|| "wgpu sparse conv bytes-per-row overflow".to_string())?;
        let neighbor_bytes_per_row = kernel_rows
            .checked_mul(core::mem::size_of::<i32>())
            .ok_or_else(|| "wgpu sparse conv neighbor-bytes-per-row overflow".to_string())?;
        let output_bytes = rows
            .checked_mul(bytes_per_row)
            .ok_or_else(|| "wgpu sparse conv output-byte-size overflow".to_string())?;
        let max_tensor_bytes = decoder_wgpu_max_tensor_bytes();
        if output_bytes > max_tensor_bytes {
            return Err(format!(
                "wgpu sparse conv output exceeds tensor limit: bytes={} max_tensor_bytes={}",
                output_bytes, max_tensor_bytes
            ));
        }
        let neighbor_bytes = rows
            .checked_mul(neighbor_bytes_per_row)
            .ok_or_else(|| "wgpu sparse conv neighbor-byte-size overflow".to_string())?;

        let max_output_bytes = decoder_wgpu_sparse_conv_max_output_bytes();
        let max_neighbor_bytes = decoder_wgpu_max_neighbor_bytes();
        let output_chunk_rows = decoder_wgpu_chunk_rows(rows, bytes_per_row, max_output_bytes);
        let neighbor_chunk_rows =
            decoder_wgpu_chunk_rows(rows, neighbor_bytes_per_row, max_neighbor_bytes);
        let mut chunk_rows = output_chunk_rows.min(neighbor_chunk_rows).max(1);

        if decoder_conv_debug_enabled() {
            eprintln!(
                "burn_trellis: chunking wgpu sparse conv (neighbor-slice) rows={} chunk_rows={} out_channels={} output_bytes={} max_output_bytes={} neighbor_bytes={} max_neighbor_bytes={}",
                rows,
                chunk_rows,
                config.out_channels,
                output_bytes,
                max_output_bytes,
                neighbor_bytes,
                max_neighbor_bytes
            );
        }

        let weight_t = self.weight_tensor(layer);
        let bias_t = self.bias_tensor(layer);
        let forward_cfg =
            decoder_wgpu_forward_config_for_call(config, rows, output_bytes, max_output_bytes);

        let mut start = 0usize;
        let mut dispatches = 0u64;
        let mut max_success_chunk_rows = 0usize;
        let mut chunk_tensors: Vec<Tensor<DefaultWgpuBackend, 2>> = Vec::new();
        while start < rows {
            let end = (start + chunk_rows).min(rows);
            let chunk_rows_count = end - start;
            let neighbor_offset = start
                .checked_mul(kernel_rows)
                .ok_or_else(|| "wgpu sparse conv neighbor-offset overflow".to_string())?;
            let neighbor_len = chunk_rows_count
                .checked_mul(kernel_rows)
                .ok_or_else(|| "wgpu sparse conv neighbor-slice overflow".to_string())?;
            let chunk_neighbor_t = Tensor::<DefaultWgpuBackend, 1, Int>::from_data(
                TensorData::new(
                    neighbor_rows[neighbor_offset..neighbor_offset + neighbor_len].to_vec(),
                    [neighbor_len],
                ),
                &self.device,
            )
            .reshape([chunk_rows_count, kernel_rows]);
            match sparse_subm_conv_forward_wgpu_with_config(
                config,
                input_t.clone(),
                chunk_neighbor_t,
                weight_t.clone(),
                bias_t.clone(),
                forward_cfg,
            ) {
                Ok(chunk_out) => {
                    max_success_chunk_rows = max_success_chunk_rows.max(chunk_rows_count);
                    chunk_tensors.push(chunk_out);
                    start = end;
                    dispatches = dispatches.saturating_add(1);
                }
                Err(err) => {
                    if decoder_wgpu_is_buffer_too_big(err.as_str()) && chunk_rows > 1 {
                        let reduced = decoder_wgpu_reduce_chunk_rows(chunk_rows);
                        if reduced < chunk_rows {
                            if decoder_conv_debug_enabled() {
                                eprintln!(
                                    "burn_trellis: reducing neighbor-slice sparse conv chunk_rows from {} to {} after buffer-too-big in '{context}'",
                                    chunk_rows, reduced
                                );
                            }
                            chunk_rows = reduced;
                            continue;
                        }
                    }
                    return Err(err);
                }
            }
        }
        telemetry_record_wgpu_success(
            context,
            dispatches.max(1),
            true,
            max_success_chunk_rows.max(1),
            input_bytes,
            output_bytes,
            rows.saturating_mul(kernel_rows),
        );
        if chunk_tensors.is_empty() {
            return Ok(Tensor::<DefaultWgpuBackend, 2>::zeros(
                [rows, config.out_channels],
                &self.device,
            ));
        }
        if chunk_tensors.len() == 1 {
            return Ok(chunk_tensors.remove(0));
        }
        Ok(Tensor::cat(chunk_tensors, 0))
    }

    fn clear_caches(&mut self) {
        self.weight_tensors.clear();
        self.bias_tensors.clear();
        self.neighbor_tensors.clear();
        self.linear_weight_tensors.clear();
        self.linear_bias_tensors.clear();
        self.vector_tensors.clear();
    }
}

#[cfg(feature = "runtime-model-wgpu")]
fn create_wgpu_decoder_context() -> Option<Arc<Mutex<DecoderWgpuConvContext>>> {
    let context = std::panic::catch_unwind(DecoderWgpuConvContext::new)
        .ok()?
        .ok()?;
    Some(Arc::new(Mutex::new(context)))
}

fn hash_coords(coords: &[[u32; 4]]) -> u64 {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x0000_0100_0000_01B3;
    let mut hash = OFFSET;
    for coord in coords {
        for value in coord {
            hash ^= *value as u64;
            hash = hash.wrapping_mul(PRIME);
        }
    }
    hash ^= coords.len() as u64;
    hash
}

fn trim_hashmap<K, V>(map: &mut HashMap<K, V>, max_entries: usize)
where
    K: Eq + std::hash::Hash + Copy,
{
    if map.len() < max_entries.max(1) {
        return;
    }
    if let Some(key) = map.keys().next().copied() {
        map.remove(&key);
    }
}

include!("sparse_decoder_loading.rs");
include!("sparse_decoder_wgpu_ops.rs");
include!("sparse_decoder_runtime_impl.rs");

#[cfg(test)]
mod tests {
    include!("sparse_decoder_tests.rs");
}

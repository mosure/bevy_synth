use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
#[cfg(feature = "runtime-model-wgpu")]
use std::sync::OnceLock;
use std::sync::{Arc, Mutex};

use burn::module::{Module, Param, ParamId};
use burn::prelude::*;
#[cfg(feature = "runtime-model-wgpu")]
use burn::tensor::TensorData;
#[cfg(feature = "runtime-model-wgpu")]
use burn::tensor::activation::sigmoid;
#[cfg(feature = "runtime-model-wgpu")]
use burn_flex_gmm::kernel_rows;
#[cfg(feature = "runtime-model-wgpu")]
use burn_flex_gmm::wgpu::{
    DefaultWgpuBackend, neighbor_rows_tensor_from_coords, sparse_subm_conv_forward_wgpu,
};
use burn_flex_gmm::{
    SparseSubmConvConfig as FlexConvConfig, SparseSubmConvWeights, build_neighbor_rows,
    pack_flex_weight, sparse_subm_conv_forward_flex_precomputed,
};
use burn_store::{BurnpackStore, ModuleSnapshot};
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
const DECODER_WGPU_TENSOR_CACHE_MAX: usize = 64;

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

#[cfg(feature = "runtime-model-wgpu")]
#[derive(Debug, Default)]
struct DecoderConvTelemetryState {
    total: DecoderConvBlockTelemetry,
    blocks: HashMap<String, DecoderConvBlockTelemetry>,
}

#[cfg(feature = "runtime-model-wgpu")]
static DECODER_CONV_TELEMETRY: OnceLock<Mutex<DecoderConvTelemetryState>> = OnceLock::new();

#[derive(Module, Debug)]
struct BinaryBlob<B: Backend> {
    bytes: Param<Tensor<B, 1, Int>>,
}

#[derive(Debug, Clone, Deserialize)]
struct BlobMetadata {
    bytes_len: usize,
}

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
pub(crate) struct SparseSubdivisionLogits {
    pub coords: Vec<[u32; 4]>,
    pub logits: Vec<f32>,
    pub spatial_shape: [u32; 3],
}

#[derive(Debug, Clone)]
pub(crate) struct SparseDecodeResult {
    pub coords: Vec<[u32; 4]>,
    pub feats: Vec<f32>,
    pub out_channels: usize,
    pub subdivisions: Vec<SparseSubdivisionLogits>,
}

#[derive(Debug, Clone)]
pub(crate) struct SparseUnetDecoderRuntime {
    out_channels: usize,
    pred_subdiv: bool,
    voxel_margin: f32,
    compute_fp16: bool,
    model_channels: Vec<usize>,
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
        let neighbor_t = self.neighbor_tensor(cache_key, config, rows, neighbor_rows)?;
        self.forward_with_neighbor_tensor(
            config,
            layer,
            input,
            context,
            rows,
            kernel_rows,
            neighbor_t,
        )
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
        let kernel_rows = kernel_rows(config)?;
        let neighbor_t = neighbor_rows_tensor_from_coords(config, coords, &self.device)?;
        self.forward_with_neighbor_tensor(
            config,
            layer,
            input,
            context,
            rows,
            kernel_rows,
            neighbor_t,
        )
    }

    fn forward_with_coords_tensor(
        &mut self,
        config: &FlexConvConfig,
        layer: &SparseConvLayer,
        input_t: Tensor<DefaultWgpuBackend, 2>,
        context: &str,
        coords: &[[u32; 4]],
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
        if coords.len() != rows {
            return Err(format!(
                "wgpu sparse conv coord/input row mismatch: coords={} rows={rows}",
                coords.len()
            ));
        }
        let output_bytes = rows
            .checked_mul(config.out_channels)
            .and_then(|value| value.checked_mul(core::mem::size_of::<f32>()))
            .ok_or_else(|| "wgpu sparse conv output-byte-size overflow".to_string())?;
        let max_output_bytes = decoder_wgpu_max_output_bytes();
        if output_bytes > max_output_bytes {
            return Err(format!(
                "wgpu sparse conv tensor output exceeds per-dispatch guard: bytes={} max_bytes={}",
                output_bytes, max_output_bytes
            ));
        }
        let kernel_rows = kernel_rows(config)?;
        let neighbor_t = neighbor_rows_tensor_from_coords(config, coords, &self.device)?;
        self.forward_with_neighbor_tensor_tensor(
            config,
            layer,
            input_t,
            context,
            rows,
            kernel_rows,
            neighbor_t,
        )
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
        let weight_t = self.weight_tensor(layer);
        let bias_t = self.bias_tensor(layer);
        let bytes_per_row = config
            .out_channels
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or_else(|| "wgpu sparse conv bytes-per-row overflow".to_string())?;
        let max_output_bytes = decoder_wgpu_max_output_bytes();
        let output_bytes = rows
            .checked_mul(bytes_per_row)
            .ok_or_else(|| "wgpu sparse conv output-byte-size overflow".to_string())?;
        if output_bytes <= max_output_bytes {
            let output =
                sparse_subm_conv_forward_wgpu(config, input_t, neighbor_t, weight_t, bias_t)?;
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

        let chunk_rows = decoder_wgpu_chunk_rows(rows, bytes_per_row, max_output_bytes);
        if decoder_conv_debug_enabled() {
            eprintln!(
                "burn_trellis: chunking wgpu sparse conv rows={} chunk_rows={} out_channels={} bytes={} max_bytes={}",
                rows, chunk_rows, config.out_channels, output_bytes, max_output_bytes
            );
        }
        let mut start = 0usize;
        let mut dispatches = 0u64;
        let mut chunk_tensors: Vec<Tensor<DefaultWgpuBackend, 2>> = Vec::new();
        while start < rows {
            let end = (start + chunk_rows).min(rows);
            let chunk_neighbor_t = neighbor_t.clone().slice([start..end, 0..kernel_rows]);
            let chunk_out = sparse_subm_conv_forward_wgpu(
                config,
                input_t.clone(),
                chunk_neighbor_t,
                weight_t.clone(),
                bias_t.clone(),
            )?;
            chunk_tensors.push(chunk_out);
            start = end;
            dispatches = dispatches.saturating_add(1);
        }
        telemetry_record_wgpu_success(
            context,
            dispatches.max(1),
            true,
            chunk_rows,
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
    if env_flag("TRELLIS2_DECODER_DISABLE_WGPU") {
        return None;
    }
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

impl SparseUnetDecoderRuntime {
    pub fn load_from_stem(
        weights_root: &Path,
        image_large_root: Option<&Path>,
        model_stem: &str,
    ) -> Result<Self, String> {
        let config_path =
            resolve_model_source_path(model_stem, "json", weights_root, image_large_root);
        let config_bytes = std::fs::read(&config_path).map_err(|err| {
            format!(
                "failed to read sparse decoder config '{}': {err}",
                config_path.display()
            )
        })?;
        let parsed: DecoderConfigFile = serde_json::from_slice(&config_bytes).map_err(|err| {
            format!(
                "failed to parse sparse decoder config '{}': {err}",
                config_path.display()
            )
        })?;
        if parsed.args.model_channels.is_empty() {
            return Err(format!(
                "sparse decoder config '{}' has empty model_channels",
                config_path.display()
            ));
        }
        if parsed.args.num_blocks.len() != parsed.args.model_channels.len() {
            return Err(format!(
                "sparse decoder config '{}' has mismatched num_blocks/model_channels lengths",
                config_path.display()
            ));
        }

        let weight_path =
            resolve_model_weight_candidates(model_stem, weights_root, image_large_root)
                .into_iter()
                .next()
                .ok_or_else(|| {
                    format!("unable to resolve decoder weights for stem '{model_stem}'")
                })?;

        let weight_backing = load_weight_backing(&weight_path)?;
        let safetensors = SafeTensors::deserialize(weight_backing.as_slice()).map_err(|err| {
            format!(
                "failed to deserialize sparse decoder weights '{}' as safetensors: {err}",
                weight_path.display()
            )
        })?;

        let out_channels = parsed.args.out_channels.unwrap_or_else(|| {
            if parsed.name == "FlexiDualGridVaeDecoder" {
                7
            } else {
                6
            }
        });

        let from_latent = load_linear(
            &safetensors,
            "from_latent.weight",
            "from_latent.bias",
            parsed.args.latent_channels,
            parsed.args.model_channels[0],
        )?;
        let output_layer = load_linear(
            &safetensors,
            "output_layer.weight",
            "output_layer.bias",
            *parsed
                .args
                .model_channels
                .last()
                .expect("checked non-empty model_channels"),
            out_channels,
        )?;

        let mut stages = Vec::with_capacity(parsed.args.num_blocks.len());
        for stage_idx in 0..parsed.args.num_blocks.len() {
            let stage_channels = parsed.args.model_channels[stage_idx];
            let mut convnext_blocks = Vec::with_capacity(parsed.args.num_blocks[stage_idx]);
            for block_idx in 0..parsed.args.num_blocks[stage_idx] {
                let prefix = format!("blocks.{stage_idx}.{block_idx}");
                convnext_blocks.push(ConvNeXtBlock {
                    conv: load_sparse_conv(
                        &safetensors,
                        format!("{prefix}.conv.weight").as_str(),
                        format!("{prefix}.conv.bias").as_str(),
                        stage_channels,
                        stage_channels,
                    )?,
                    norm_weight: load_vector(
                        &safetensors,
                        format!("{prefix}.norm.weight").as_str(),
                        stage_channels,
                    )?,
                    norm_bias: load_vector(
                        &safetensors,
                        format!("{prefix}.norm.bias").as_str(),
                        stage_channels,
                    )?,
                    mlp_0: load_linear_dynamic(
                        &safetensors,
                        format!("{prefix}.mlp.0.weight").as_str(),
                        format!("{prefix}.mlp.0.bias").as_str(),
                        stage_channels,
                    )?,
                    mlp_2: load_linear_dynamic(
                        &safetensors,
                        format!("{prefix}.mlp.2.weight").as_str(),
                        format!("{prefix}.mlp.2.bias").as_str(),
                        0,
                    )?,
                });
            }

            let upsample_block = if stage_idx + 1 < parsed.args.model_channels.len() {
                let up_idx = parsed.args.num_blocks[stage_idx];
                let prefix = format!("blocks.{stage_idx}.{up_idx}");
                let in_channels = parsed.args.model_channels[stage_idx];
                let out_channels = parsed.args.model_channels[stage_idx + 1];
                let conv1_out = out_channels
                    .checked_mul(8)
                    .ok_or_else(|| "conv1_out channels overflow".to_string())?;
                let to_subdiv = match parsed.args.pred_subdiv.unwrap_or(true) {
                    true => Some(load_linear(
                        &safetensors,
                        format!("{prefix}.to_subdiv.weight").as_str(),
                        format!("{prefix}.to_subdiv.bias").as_str(),
                        in_channels,
                        8,
                    )?),
                    false => None,
                };

                Some(C2SUpsampleBlock {
                    in_channels,
                    out_channels,
                    norm1_weight: load_vector(
                        &safetensors,
                        format!("{prefix}.norm1.weight").as_str(),
                        in_channels,
                    )?,
                    norm1_bias: load_vector(
                        &safetensors,
                        format!("{prefix}.norm1.bias").as_str(),
                        in_channels,
                    )?,
                    to_subdiv,
                    conv1: load_sparse_conv(
                        &safetensors,
                        format!("{prefix}.conv1.weight").as_str(),
                        format!("{prefix}.conv1.bias").as_str(),
                        in_channels,
                        conv1_out,
                    )?,
                    conv2: load_sparse_conv(
                        &safetensors,
                        format!("{prefix}.conv2.weight").as_str(),
                        format!("{prefix}.conv2.bias").as_str(),
                        out_channels,
                        out_channels,
                    )?,
                })
            } else {
                None
            };

            stages.push(DecoderStage {
                convnext_blocks,
                upsample_block,
            });
        }

        Ok(Self {
            out_channels,
            pred_subdiv: parsed.args.pred_subdiv.unwrap_or(true),
            voxel_margin: parsed.args.voxel_margin.unwrap_or(0.5),
            compute_fp16: parsed.args.use_fp16.unwrap_or(false) && !decoder_force_fp32(),
            model_channels: parsed.args.model_channels,
            from_latent,
            output_layer,
            stages,
            conv_cache: Arc::new(Mutex::new(DecoderConvCache::default())),
            #[cfg(feature = "runtime-model-wgpu")]
            wgpu_context: create_wgpu_decoder_context(),
        })
    }

    pub fn out_channels(&self) -> usize {
        self.out_channels
    }

    pub fn pred_subdiv(&self) -> bool {
        self.pred_subdiv
    }

    pub fn voxel_margin(&self) -> f32 {
        self.voxel_margin
    }

    pub fn decode(
        &self,
        coords: &[[u32; 4]],
        rows: &[[f32; 32]],
        guide_subdivisions: Option<&[SparseSubdivisionLogits]>,
    ) -> Result<SparseDecodeResult, String> {
        let count = coords.len().min(rows.len());
        if count == 0 {
            return Ok(SparseDecodeResult {
                coords: Vec::new(),
                feats: Vec::new(),
                out_channels: self.out_channels,
                subdivisions: Vec::new(),
            });
        }

        let mut state_coords = coords[..count].to_vec();
        let mut state_feats = flatten_rows_32(&rows[..count]);
        #[cfg(feature = "runtime-model-wgpu")]
        let mut state_feats_wgpu: Option<Tensor<DefaultWgpuBackend, 2>> = None;
        let mut conv_cache = self
            .conv_cache
            .lock()
            .map_err(|_| "decoder conv cache lock poisoned".to_string())?;
        #[cfg(feature = "runtime-model-wgpu")]
        let mut wgpu_context = if let Some(context) = self.wgpu_context.as_ref() {
            Some(
                context
                    .lock()
                    .map_err(|_| "decoder wgpu context lock poisoned".to_string())?,
            )
        } else {
            None
        };
        state_feats = linear_forward(
            state_feats.as_slice(),
            count,
            &self.from_latent,
            "from_latent",
        )?;
        if self.compute_fp16 {
            quantize_f16_inplace(state_feats.as_mut_slice());
        }

        let mut subdivisions = Vec::new();
        for (stage_idx, stage) in self.stages.iter().enumerate() {
            let stage_channels = self.model_channels[stage_idx];
            #[allow(unused_mut)]
            let mut convnext_device_complete = false;
            #[cfg(feature = "runtime-model-wgpu")]
            if decoder_wgpu_device_math_enabled()
                && (!self.compute_fp16 || decoder_wgpu_device_math_allow_fp16())
                && !stage.convnext_blocks.is_empty()
                && let Some(context_gpu) = wgpu_context.as_deref_mut()
            {
                let row_count = state_coords.len();
                let state_t = if let Some(state_t) = state_feats_wgpu.take() {
                    let [rows_device, channels_device] = state_t.dims();
                    if rows_device == row_count && channels_device == stage_channels {
                        state_t
                    } else {
                        Tensor::<DefaultWgpuBackend, 1>::from_floats(
                            state_feats.as_slice(),
                            &context_gpu.device,
                        )
                        .reshape([row_count, stage_channels])
                    }
                } else {
                    Tensor::<DefaultWgpuBackend, 1>::from_floats(
                        state_feats.as_slice(),
                        &context_gpu.device,
                    )
                    .reshape([row_count, stage_channels])
                };
                match convnext_blocks_forward_wgpu_tensor(
                    context_gpu,
                    state_coords.as_slice(),
                    state_t,
                    stage_idx,
                    stage_channels,
                    stage.convnext_blocks.as_slice(),
                ) {
                    Ok(next_state_feats) => {
                        state_feats_wgpu = Some(next_state_feats);
                        convnext_device_complete = true;
                    }
                    Err(err) => {
                        state_feats_wgpu = None;
                        if decoder_conv_debug_enabled() {
                            eprintln!(
                                "burn_trellis: wgpu convnext stage fallback to cpu stage={} reason={err}",
                                stage_idx
                            );
                        }
                    }
                }
            }
            if !convnext_device_complete {
                #[cfg(feature = "runtime-model-wgpu")]
                if let Some(state_t) = state_feats_wgpu.take() {
                    let context =
                        format!("decoder stage {stage_idx} convnext fallback state readback");
                    state_feats = tensor_to_vec_f32(state_t, context.as_str())?;
                }
                for (block_idx, block) in stage.convnext_blocks.iter().enumerate() {
                    let row_count = state_coords.len();
                    if row_count == 0 {
                        break;
                    }
                    let residual = state_feats.clone();
                    let mut h = sparse_subm_conv_forward(
                        state_coords.as_slice(),
                        state_feats.as_slice(),
                        &block.conv,
                        format!("stage {stage_idx} block {block_idx} conv").as_str(),
                        &mut conv_cache,
                        #[cfg(feature = "runtime-model-wgpu")]
                        wgpu_context.as_deref_mut(),
                    )?;
                    if self.compute_fp16 {
                        quantize_f16_inplace(h.as_mut_slice());
                    }
                    layer_norm_inplace(
                        h.as_mut_slice(),
                        row_count,
                        stage_channels,
                        Some(block.norm_weight.as_slice()),
                        Some(block.norm_bias.as_slice()),
                        LAYER_NORM32_EPS,
                    )?;
                    if self.compute_fp16 {
                        quantize_f16_inplace(h.as_mut_slice());
                    }
                    h = linear_forward(
                        h.as_slice(),
                        row_count,
                        &block.mlp_0,
                        format!("stage {stage_idx} block {block_idx} mlp_0").as_str(),
                    )?;
                    if self.compute_fp16 {
                        quantize_f16_inplace(h.as_mut_slice());
                    }
                    silu_inplace(h.as_mut_slice());
                    if self.compute_fp16 {
                        quantize_f16_inplace(h.as_mut_slice());
                    }
                    h = linear_forward(
                        h.as_slice(),
                        row_count,
                        &block.mlp_2,
                        format!("stage {stage_idx} block {block_idx} mlp_2").as_str(),
                    )?;
                    if self.compute_fp16 {
                        quantize_f16_inplace(h.as_mut_slice());
                    }
                    add_inplace(h.as_mut_slice(), residual.as_slice());
                    if self.compute_fp16 {
                        quantize_f16_inplace(h.as_mut_slice());
                    }
                    state_feats = h;
                }
            }

            if let Some(up) = stage.upsample_block.as_ref() {
                #[cfg(feature = "runtime-model-wgpu")]
                if let Some(state_t) = state_feats_wgpu.take() {
                    let context = format!("decoder stage {stage_idx} upsample state readback");
                    state_feats = tensor_to_vec_f32(state_t, context.as_str())?;
                }
                let parent_coords = state_coords.clone();
                let parent_feats = state_feats.clone();
                let parent_rows = parent_coords.len();
                if parent_rows == 0 {
                    continue;
                }

                let subdiv_logits = if let Some(to_subdiv) = up.to_subdiv.as_ref() {
                    let mut logits = linear_forward(
                        parent_feats.as_slice(),
                        parent_rows,
                        to_subdiv,
                        format!("stage {stage_idx} to_subdiv").as_str(),
                    )?;
                    if self.compute_fp16 {
                        quantize_f16_inplace(logits.as_mut_slice());
                    }
                    if should_center_subdivision_logits() {
                        row_center_logits(logits.as_mut_slice(), parent_rows);
                    }
                    logits
                } else {
                    let guide = guide_subdivisions
                        .and_then(|levels| levels.get(stage_idx))
                        .ok_or_else(|| {
                            format!(
                                "decoder stage {stage_idx} requires guide_subdivisions but none were provided"
                            )
                        })?;
                    map_guide_subdivision_logits(parent_coords.as_slice(), guide)?
                };

                let subdivision_mask =
                    logits_to_mask(subdiv_logits.as_slice(), parent_rows, false)?;
                if self.pred_subdiv {
                    subdivisions.push(SparseSubdivisionLogits {
                        spatial_shape: spatial_shape_from_coords(parent_coords.as_slice()),
                        coords: parent_coords.clone(),
                        logits: subdiv_logits.clone(),
                    });
                }

                let mut h_norm = parent_feats.clone();
                layer_norm_inplace(
                    h_norm.as_mut_slice(),
                    parent_rows,
                    up.in_channels,
                    Some(up.norm1_weight.as_slice()),
                    Some(up.norm1_bias.as_slice()),
                    LAYER_NORM32_EPS,
                )?;
                if self.compute_fp16 {
                    quantize_f16_inplace(h_norm.as_mut_slice());
                }
                silu_inplace(h_norm.as_mut_slice());
                if self.compute_fp16 {
                    quantize_f16_inplace(h_norm.as_mut_slice());
                }
                let h_conv1 = sparse_subm_conv_forward(
                    parent_coords.as_slice(),
                    h_norm.as_slice(),
                    &up.conv1,
                    format!("stage {stage_idx} up conv1").as_str(),
                    &mut conv_cache,
                    #[cfg(feature = "runtime-model-wgpu")]
                    wgpu_context.as_deref_mut(),
                )?;
                let mut h_conv1 = h_conv1;
                if self.compute_fp16 {
                    quantize_f16_inplace(h_conv1.as_mut_slice());
                }
                let (child_coords, mut h_up) = channel2spatial(
                    parent_coords.as_slice(),
                    h_conv1.as_slice(),
                    up.out_channels
                        .checked_mul(8)
                        .ok_or_else(|| "up.out_channels * 8 overflow".to_string())?,
                    subdivision_mask.as_slice(),
                )?;
                let (child_coords_skip, x_up) = channel2spatial(
                    parent_coords.as_slice(),
                    parent_feats.as_slice(),
                    up.in_channels,
                    subdivision_mask.as_slice(),
                )?;
                if child_coords != child_coords_skip {
                    return Err(format!(
                        "decoder stage {stage_idx} channel2spatial coord mismatch between conv and skip branches"
                    ));
                }

                let skip_in_channels = up.in_channels / 8;
                if skip_in_channels == 0 || up.out_channels % skip_in_channels != 0 {
                    return Err(format!(
                        "decoder stage {stage_idx} invalid skip channel ratio (in={}, out={})",
                        up.in_channels, up.out_channels
                    ));
                }
                let repeat_factor = up.out_channels / skip_in_channels;
                let skip = repeat_interleave_channels(
                    x_up.as_slice(),
                    child_coords.len(),
                    skip_in_channels,
                    repeat_factor,
                );

                let child_rows = child_coords.len();
                #[allow(unused_mut)]
                let mut upsample_device_complete = false;
                #[cfg(feature = "runtime-model-wgpu")]
                if decoder_wgpu_device_math_enabled()
                    && (!self.compute_fp16 || decoder_wgpu_device_math_allow_fp16())
                    && child_rows > 0
                    && let Some(context_gpu) = wgpu_context.as_deref_mut()
                {
                    let h_up_t =
                        Tensor::<DefaultWgpuBackend, 1>::from_floats(h_up.as_slice(), &context_gpu.device)
                            .reshape([child_rows, up.out_channels]);
                    let h_up_t = layer_norm_wgpu(
                        context_gpu,
                        h_up_t,
                        child_rows,
                        up.out_channels,
                        None,
                        None,
                        LAYER_NORM32_EPS,
                    )?;
                    let h_up_t = silu_wgpu(h_up_t);
                    let config = flex_config_for_layer(&up.conv2);
                    match context_gpu.forward_with_coords_tensor(
                        &config,
                        &up.conv2,
                        h_up_t,
                        format!("stage {stage_idx} up conv2(wgpu_math)").as_str(),
                        child_coords.as_slice(),
                    ) {
                        Ok(h_t) => {
                            let skip_t = Tensor::<DefaultWgpuBackend, 1>::from_floats(
                                skip.as_slice(),
                                &context_gpu.device,
                            )
                            .reshape([child_rows, up.out_channels]);
                            state_feats_wgpu = Some(h_t.add(skip_t));
                            upsample_device_complete = true;
                        }
                        Err(err) => {
                            state_feats_wgpu = None;
                            if decoder_conv_debug_enabled() {
                                eprintln!(
                                    "burn_trellis: wgpu upsample conv2 fallback to cpu stage={} reason={err}",
                                    stage_idx
                                );
                            }
                        }
                    }
                }

                if !upsample_device_complete {
                    layer_norm_inplace(
                        h_up.as_mut_slice(),
                        child_rows,
                        up.out_channels,
                        None,
                        None,
                        LAYER_NORM32_EPS,
                    )?;
                    if self.compute_fp16 {
                        quantize_f16_inplace(h_up.as_mut_slice());
                    }
                    silu_inplace(h_up.as_mut_slice());
                    if self.compute_fp16 {
                        quantize_f16_inplace(h_up.as_mut_slice());
                    }
                    let mut h = sparse_subm_conv_forward(
                        child_coords.as_slice(),
                        h_up.as_slice(),
                        &up.conv2,
                        format!("stage {stage_idx} up conv2").as_str(),
                        &mut conv_cache,
                        #[cfg(feature = "runtime-model-wgpu")]
                        wgpu_context.as_deref_mut(),
                    )?;
                    if self.compute_fp16 {
                        quantize_f16_inplace(h.as_mut_slice());
                    }
                    add_inplace(h.as_mut_slice(), skip.as_slice());
                    if self.compute_fp16 {
                        quantize_f16_inplace(h.as_mut_slice());
                    }
                    state_feats = h;
                    #[cfg(feature = "runtime-model-wgpu")]
                    {
                        state_feats_wgpu = None;
                    }
                } else {
                    state_feats.clear();
                }
                state_coords = child_coords;
            }
        }

        let rows_final = state_coords.len();
        let final_channels = *self
            .model_channels
            .last()
            .expect("checked non-empty model_channels");
        let state_feats = {
            #[cfg(feature = "runtime-model-wgpu")]
            {
                if let Some(state_t) = state_feats_wgpu.take() {
                    if decoder_wgpu_device_math_enabled()
                        && (!self.compute_fp16 || decoder_wgpu_device_math_allow_fp16())
                    {
                        if let Some(context_gpu) = wgpu_context.as_deref_mut() {
                            let state_t = layer_norm_wgpu(
                                context_gpu,
                                state_t,
                                rows_final,
                                final_channels,
                                None,
                                None,
                                F_LAYER_NORM_EPS,
                            )?;
                            let state_t = linear_forward_wgpu(
                                context_gpu,
                                state_t,
                                &self.output_layer,
                                "output_layer(wgpu_math)",
                            )?;
                            tensor_to_vec_f32(state_t, "output_layer(wgpu_math)")?
                        } else {
                            let mut state_feats =
                                tensor_to_vec_f32(state_t, "output_layer state readback")?;
                            layer_norm_inplace(
                                state_feats.as_mut_slice(),
                                rows_final,
                                final_channels,
                                None,
                                None,
                                F_LAYER_NORM_EPS,
                            )?;
                            linear_forward(
                                state_feats.as_slice(),
                                rows_final,
                                &self.output_layer,
                                "output_layer",
                            )?
                        }
                    } else {
                        let mut state_feats =
                            tensor_to_vec_f32(state_t, "output_layer state readback")?;
                        layer_norm_inplace(
                            state_feats.as_mut_slice(),
                            rows_final,
                            final_channels,
                            None,
                            None,
                            F_LAYER_NORM_EPS,
                        )?;
                        linear_forward(
                            state_feats.as_slice(),
                            rows_final,
                            &self.output_layer,
                            "output_layer",
                        )?
                    }
                } else {
                    layer_norm_inplace(
                        state_feats.as_mut_slice(),
                        rows_final,
                        final_channels,
                        None,
                        None,
                        F_LAYER_NORM_EPS,
                    )?;
                    linear_forward(
                        state_feats.as_slice(),
                        rows_final,
                        &self.output_layer,
                        "output_layer",
                    )?
                }
            }
            #[cfg(not(feature = "runtime-model-wgpu"))]
            {
                layer_norm_inplace(
                    state_feats.as_mut_slice(),
                    rows_final,
                    final_channels,
                    None,
                    None,
                    F_LAYER_NORM_EPS,
                )?;
                linear_forward(
                    state_feats.as_slice(),
                    rows_final,
                    &self.output_layer,
                    "output_layer",
                )?
            }
        };

        #[cfg(feature = "runtime-model-wgpu")]
        if decoder_wgpu_clear_cache_after_decode()
            && let Some(context) = wgpu_context.as_deref_mut()
        {
            context.clear_caches();
        }

        Ok(SparseDecodeResult {
            coords: state_coords,
            feats: state_feats,
            out_channels: self.out_channels,
            subdivisions,
        })
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn stage0_subdivision_logits(
        &self,
        coords: &[[u32; 4]],
        rows: &[[f32; 32]],
    ) -> Result<SparseSubdivisionLogits, String> {
        if self.stages.is_empty() {
            return Err("decoder has no stages".to_string());
        }
        let stage = &self.stages[0];
        let up = stage
            .upsample_block
            .as_ref()
            .ok_or_else(|| "decoder stage0 has no upsample block".to_string())?;
        let to_subdiv = up
            .to_subdiv
            .as_ref()
            .ok_or_else(|| "decoder stage0 has no to_subdiv head".to_string())?;

        let count = coords.len().min(rows.len());
        if count == 0 {
            return Ok(SparseSubdivisionLogits {
                coords: Vec::new(),
                logits: Vec::new(),
                spatial_shape: [1, 1, 1],
            });
        }

        let state_coords = coords[..count].to_vec();
        let mut state_feats = flatten_rows_32(&rows[..count]);
        #[cfg(feature = "runtime-model-wgpu")]
        let mut state_feats_wgpu: Option<Tensor<DefaultWgpuBackend, 2>> = None;
        let mut conv_cache = self
            .conv_cache
            .lock()
            .map_err(|_| "decoder conv cache lock poisoned".to_string())?;
        #[cfg(feature = "runtime-model-wgpu")]
        let mut wgpu_context = if let Some(context) = self.wgpu_context.as_ref() {
            Some(
                context
                    .lock()
                    .map_err(|_| "decoder wgpu context lock poisoned".to_string())?,
            )
        } else {
            None
        };
        state_feats = linear_forward(
            state_feats.as_slice(),
            count,
            &self.from_latent,
            "from_latent(stage0)",
        )?;
        if self.compute_fp16 {
            quantize_f16_inplace(state_feats.as_mut_slice());
        }

        let stage_channels = self.model_channels[0];
        #[allow(unused_mut)]
        let mut convnext_device_complete = false;
        #[cfg(feature = "runtime-model-wgpu")]
        if decoder_wgpu_device_math_enabled()
            && (!self.compute_fp16 || decoder_wgpu_device_math_allow_fp16())
            && !stage.convnext_blocks.is_empty()
            && let Some(context_gpu) = wgpu_context.as_deref_mut()
        {
            let row_count = state_coords.len();
            let state_t = if let Some(state_t) = state_feats_wgpu.take() {
                let [rows_device, channels_device] = state_t.dims();
                if rows_device == row_count && channels_device == stage_channels {
                    state_t
                } else {
                    Tensor::<DefaultWgpuBackend, 1>::from_floats(
                        state_feats.as_slice(),
                        &context_gpu.device,
                    )
                    .reshape([row_count, stage_channels])
                }
            } else {
                Tensor::<DefaultWgpuBackend, 1>::from_floats(
                    state_feats.as_slice(),
                    &context_gpu.device,
                )
                .reshape([row_count, stage_channels])
            };
            match convnext_blocks_forward_wgpu_tensor(
                context_gpu,
                state_coords.as_slice(),
                state_t,
                0,
                stage_channels,
                stage.convnext_blocks.as_slice(),
            ) {
                Ok(next_state_feats) => {
                    state_feats_wgpu = Some(next_state_feats);
                    convnext_device_complete = true;
                }
                Err(err) => {
                    state_feats_wgpu = None;
                    if decoder_conv_debug_enabled() {
                        eprintln!("burn_trellis: wgpu stage0 convnext fallback to cpu reason={err}");
                    }
                }
            }
        }

        if !convnext_device_complete {
            #[cfg(feature = "runtime-model-wgpu")]
            if let Some(state_t) = state_feats_wgpu.take() {
                state_feats = tensor_to_vec_f32(state_t, "stage0 convnext fallback readback")?;
            }
            for (block_idx, block) in stage.convnext_blocks.iter().enumerate() {
                let row_count = state_coords.len();
                if row_count == 0 {
                    break;
                }
                let residual = state_feats.clone();
                let mut h = sparse_subm_conv_forward(
                    state_coords.as_slice(),
                    state_feats.as_slice(),
                    &block.conv,
                    format!("stage0 block {block_idx} conv(stage0)").as_str(),
                    &mut conv_cache,
                    #[cfg(feature = "runtime-model-wgpu")]
                    wgpu_context.as_deref_mut(),
                )?;
                if self.compute_fp16 {
                    quantize_f16_inplace(h.as_mut_slice());
                }
                layer_norm_inplace(
                    h.as_mut_slice(),
                    row_count,
                    stage_channels,
                    Some(block.norm_weight.as_slice()),
                    Some(block.norm_bias.as_slice()),
                    LAYER_NORM32_EPS,
                )?;
                if self.compute_fp16 {
                    quantize_f16_inplace(h.as_mut_slice());
                }
                h = linear_forward(
                    h.as_slice(),
                    row_count,
                    &block.mlp_0,
                    format!("stage0 block {block_idx} mlp_0(stage0)").as_str(),
                )?;
                if self.compute_fp16 {
                    quantize_f16_inplace(h.as_mut_slice());
                }
                silu_inplace(h.as_mut_slice());
                if self.compute_fp16 {
                    quantize_f16_inplace(h.as_mut_slice());
                }
                h = linear_forward(
                    h.as_slice(),
                    row_count,
                    &block.mlp_2,
                    format!("stage0 block {block_idx} mlp_2(stage0)").as_str(),
                )?;
                if self.compute_fp16 {
                    quantize_f16_inplace(h.as_mut_slice());
                }
                add_inplace(h.as_mut_slice(), residual.as_slice());
                if self.compute_fp16 {
                    quantize_f16_inplace(h.as_mut_slice());
                }
                state_feats = h;
            }
        }

        let mut subdiv_logits = {
            #[cfg(feature = "runtime-model-wgpu")]
            {
                if let Some(state_t) = state_feats_wgpu.take() {
                    if decoder_wgpu_device_math_enabled()
                        && (!self.compute_fp16 || decoder_wgpu_device_math_allow_fp16())
                    {
                        if let Some(context_gpu) = wgpu_context.as_deref_mut() {
                            let logits_t = linear_forward_wgpu(
                                context_gpu,
                                state_t,
                                to_subdiv,
                                "stage0 to_subdiv(wgpu_math)",
                            )?;
                            tensor_to_vec_f32(logits_t, "stage0 to_subdiv(wgpu_math)")?
                        } else {
                            let host = tensor_to_vec_f32(state_t, "stage0 to_subdiv readback")?;
                            linear_forward(
                                host.as_slice(),
                                state_coords.len(),
                                to_subdiv,
                                "stage0 to_subdiv",
                            )?
                        }
                    } else {
                        let host = tensor_to_vec_f32(state_t, "stage0 to_subdiv readback")?;
                        linear_forward(
                            host.as_slice(),
                            state_coords.len(),
                            to_subdiv,
                            "stage0 to_subdiv",
                        )?
                    }
                } else {
                    linear_forward(
                        state_feats.as_slice(),
                        state_coords.len(),
                        to_subdiv,
                        "stage0 to_subdiv",
                    )?
                }
            }
            #[cfg(not(feature = "runtime-model-wgpu"))]
            {
                linear_forward(
                    state_feats.as_slice(),
                    state_coords.len(),
                    to_subdiv,
                    "stage0 to_subdiv",
                )?
            }
        };
        if self.compute_fp16 {
            quantize_f16_inplace(subdiv_logits.as_mut_slice());
        }
        if should_center_subdivision_logits() {
            row_center_logits(subdiv_logits.as_mut_slice(), state_coords.len());
        }

        #[cfg(feature = "runtime-model-wgpu")]
        if decoder_wgpu_clear_cache_after_decode()
            && let Some(context) = wgpu_context.as_deref_mut()
        {
            context.clear_caches();
        }

        Ok(SparseSubdivisionLogits {
            spatial_shape: spatial_shape_from_coords(state_coords.as_slice()),
            coords: state_coords,
            logits: subdiv_logits,
        })
    }
}

fn load_linear(
    safetensors: &SafeTensors<'_>,
    weight_key: &str,
    bias_key: &str,
    expected_in: usize,
    expected_out: usize,
) -> Result<LinearLayer, String> {
    let (w_shape, w_data) = load_tensor_f32(safetensors, weight_key)?;
    if w_shape.len() != 2 {
        return Err(format!(
            "tensor '{weight_key}' expected rank=2, got rank={}",
            w_shape.len()
        ));
    }
    let out_channels = w_shape[0];
    let in_channels = w_shape[1];
    if expected_in > 0 && in_channels != expected_in {
        return Err(format!(
            "tensor '{weight_key}' expected in_channels={expected_in}, got {in_channels}"
        ));
    }
    if expected_out > 0 && out_channels != expected_out {
        return Err(format!(
            "tensor '{weight_key}' expected out_channels={expected_out}, got {out_channels}"
        ));
    }

    let (b_shape, bias) = load_tensor_f32(safetensors, bias_key)?;
    if b_shape.len() != 1 || b_shape[0] != out_channels {
        return Err(format!(
            "tensor '{bias_key}' expected shape=[{out_channels}], got {:?}",
            b_shape
        ));
    }

    let weight = w_data;

    Ok(LinearLayer {
        in_channels,
        out_channels,
        weight,
        bias,
    })
}

fn load_linear_dynamic(
    safetensors: &SafeTensors<'_>,
    weight_key: &str,
    bias_key: &str,
    expected_in: usize,
) -> Result<LinearLayer, String> {
    let (w_shape, w_data) = load_tensor_f32(safetensors, weight_key)?;
    if w_shape.len() != 2 {
        return Err(format!(
            "tensor '{weight_key}' expected rank=2, got rank={}",
            w_shape.len()
        ));
    }
    let out_channels = w_shape[0];
    let in_channels = w_shape[1];
    if expected_in > 0 && in_channels != expected_in {
        return Err(format!(
            "tensor '{weight_key}' expected in_channels={expected_in}, got {in_channels}"
        ));
    }

    let (b_shape, bias) = load_tensor_f32(safetensors, bias_key)?;
    if b_shape.len() != 1 || b_shape[0] != out_channels {
        return Err(format!(
            "tensor '{bias_key}' expected shape=[{out_channels}], got {:?}",
            b_shape
        ));
    }

    let weight = w_data;

    Ok(LinearLayer {
        in_channels,
        out_channels,
        weight,
        bias,
    })
}

fn load_sparse_conv(
    safetensors: &SafeTensors<'_>,
    weight_key: &str,
    bias_key: &str,
    expected_in: usize,
    expected_out: usize,
) -> Result<SparseConvLayer, String> {
    let (w_shape, weight) = load_tensor_f32(safetensors, weight_key)?;
    if w_shape.len() != 5 {
        return Err(format!(
            "tensor '{weight_key}' expected rank=5, got rank={}",
            w_shape.len()
        ));
    }
    let out_channels = w_shape[0];
    let kd = w_shape[1];
    let kh = w_shape[2];
    let kw = w_shape[3];
    let in_channels_per_group = w_shape[4];
    if kd == 0 || kh == 0 || kw == 0 {
        return Err(format!(
            "tensor '{weight_key}' has invalid kernel dims ({kd},{kh},{kw})"
        ));
    }
    if in_channels_per_group == 0 {
        return Err(format!(
            "tensor '{weight_key}' has invalid in_channels_per_group=0"
        ));
    }
    if expected_out > 0 && out_channels != expected_out {
        return Err(format!(
            "tensor '{weight_key}' expected out_channels={expected_out}, got {out_channels}"
        ));
    }
    let in_channels = if expected_in > 0 {
        expected_in
    } else {
        in_channels_per_group
    };
    if in_channels < in_channels_per_group || !in_channels.is_multiple_of(in_channels_per_group) {
        return Err(format!(
            "tensor '{weight_key}' expected_in={in_channels} is incompatible with in_per_group={in_channels_per_group}"
        ));
    }
    let groups = in_channels / in_channels_per_group;
    if groups == 0 || !out_channels.is_multiple_of(groups) {
        return Err(format!(
            "tensor '{weight_key}' has incompatible grouped channels (groups={groups}, out_channels={out_channels})"
        ));
    }
    let out_channels_per_group = out_channels / groups;

    let (b_shape, bias) = load_tensor_f32(safetensors, bias_key)?;
    if b_shape.len() != 1 || b_shape[0] != out_channels {
        return Err(format!(
            "tensor '{bias_key}' expected shape=[{out_channels}], got {:?}",
            b_shape
        ));
    }

    let expected_weight_len = out_channels
        .checked_mul(kd)
        .and_then(|value| value.checked_mul(kh))
        .and_then(|value| value.checked_mul(kw))
        .and_then(|value| value.checked_mul(in_channels_per_group))
        .ok_or_else(|| format!("tensor '{weight_key}' weight shape product overflow"))?;
    if weight.len() != expected_weight_len {
        return Err(format!(
            "tensor '{weight_key}' element count mismatch: expected {expected_weight_len}, got {}",
            weight.len()
        ));
    }

    let flex_pack_config = FlexConvConfig {
        in_channels,
        out_channels,
        kernel_d: kd,
        kernel_h: kh,
        kernel_w: kw,
        in_channels_per_group,
        out_channels_per_group,
        groups,
        axis_order: [0, 1, 2],
        axis_sign: [1, 1, 1],
    };
    let flex_packed_weight = Some(pack_flex_weight(&flex_pack_config, weight.as_slice())?);

    Ok(SparseConvLayer {
        in_channels,
        out_channels,
        kernel_d: kd,
        kernel_h: kh,
        kernel_w: kw,
        in_channels_per_group,
        out_channels_per_group,
        groups,
        weight,
        bias,
        flex_packed_weight,
    })
}

fn load_vector(
    safetensors: &SafeTensors<'_>,
    key: &str,
    expected_len: usize,
) -> Result<Vec<f32>, String> {
    let (shape, data) = load_tensor_f32(safetensors, key)?;
    if shape.len() != 1 {
        return Err(format!(
            "tensor '{key}' expected rank=1, got rank={}",
            shape.len()
        ));
    }
    if expected_len > 0 && shape[0] != expected_len {
        return Err(format!(
            "tensor '{key}' expected len={expected_len}, got len={}",
            shape[0]
        ));
    }
    Ok(data)
}

fn load_tensor_f32(
    safetensors: &SafeTensors<'_>,
    key: &str,
) -> Result<(Vec<usize>, Vec<f32>), String> {
    let view = safetensors
        .tensor(key)
        .map_err(|err| format!("missing tensor '{key}' in safetensors: {err}"))?;
    let shape = view.shape().to_vec();
    let data = match view.dtype() {
        Dtype::F32 => bytes_to_f32(view.data())?,
        Dtype::F16 => bytes_to_f16(view.data())?,
        Dtype::BF16 => bytes_to_bf16(view.data())?,
        other => {
            return Err(format!(
                "tensor '{key}' has unsupported dtype {other:?}; expected f32/f16/bf16"
            ));
        }
    };
    let expected = shape
        .iter()
        .try_fold(1usize, |acc, value| acc.checked_mul(*value))
        .ok_or_else(|| format!("tensor '{key}' shape product overflow: {:?}", shape))?;
    if data.len() != expected {
        return Err(format!(
            "tensor '{key}' element count mismatch: expected {expected}, got {}",
            data.len()
        ));
    }
    Ok((shape, data))
}

fn bytes_to_f32(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if !bytes.len().is_multiple_of(4) {
        return Err(format!(
            "invalid f32 tensor payload byte length {}; must be divisible by 4",
            bytes.len()
        ));
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

fn bytes_to_f16(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if !bytes.len().is_multiple_of(2) {
        return Err(format!(
            "invalid f16 tensor payload byte length {}; must be divisible by 2",
            bytes.len()
        ));
    }
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        out.push(f16::from_bits(bits).to_f32());
    }
    Ok(out)
}

fn bytes_to_bf16(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if !bytes.len().is_multiple_of(2) {
        return Err(format!(
            "invalid bf16 tensor payload byte length {}; must be divisible by 2",
            bytes.len()
        ));
    }
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        out.push(bf16::from_bits(bits).to_f32());
    }
    Ok(out)
}

fn flatten_rows_32(rows: &[[f32; 32]]) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows.len() * 32);
    for row in rows {
        out.extend_from_slice(row);
    }
    out
}

fn linear_forward(
    input: &[f32],
    rows: usize,
    layer: &LinearLayer,
    context: &str,
) -> Result<Vec<f32>, String> {
    if rows == 0 {
        return Ok(Vec::new());
    }
    let expected = rows
        .checked_mul(layer.in_channels)
        .ok_or_else(|| format!("{context}: input size overflow"))?;
    if input.len() != expected {
        return Err(format!(
            "{context}: invalid input len {}, expected {} (rows={} in_channels={})",
            input.len(),
            expected,
            rows,
            layer.in_channels
        ));
    }
    if layer.bias.len() != layer.out_channels {
        return Err(format!(
            "{context}: bias len {} does not match out_channels {}",
            layer.bias.len(),
            layer.out_channels
        ));
    }
    let mut output = vec![0.0f32; rows * layer.out_channels];
    for row_idx in 0..rows {
        let base = row_idx * layer.out_channels;
        output[base..base + layer.out_channels].copy_from_slice(layer.bias.as_slice());
    }
    unsafe {
        matrixmultiply::sgemm(
            rows,
            layer.in_channels,
            layer.out_channels,
            1.0,
            input.as_ptr(),
            layer.in_channels as isize,
            1,
            layer.weight.as_ptr(),
            1,
            layer.in_channels as isize,
            1.0,
            output.as_mut_ptr(),
            layer.out_channels as isize,
            1,
        );
    }
    Ok(output)
}

#[cfg(feature = "runtime-model-wgpu")]
fn linear_forward_wgpu(
    context_gpu: &mut DecoderWgpuConvContext,
    input: Tensor<DefaultWgpuBackend, 2>,
    layer: &LinearLayer,
    context: &str,
) -> Result<Tensor<DefaultWgpuBackend, 2>, String> {
    let [rows, in_channels] = input.dims();
    if in_channels != layer.in_channels {
        return Err(format!(
            "{context}: invalid input channels {}, expected {}",
            in_channels, layer.in_channels
        ));
    }
    if layer.bias.len() != layer.out_channels {
        return Err(format!(
            "{context}: bias len {} does not match out_channels {}",
            layer.bias.len(),
            layer.out_channels
        ));
    }
    if rows == 0 {
        return Ok(Tensor::<DefaultWgpuBackend, 2>::zeros(
            [0, layer.out_channels],
            &context_gpu.device,
        ));
    }
    let weight_t = context_gpu.linear_weight_tensor(layer).swap_dims(0, 1);
    let bias_t = context_gpu
        .linear_bias_tensor(layer)
        .reshape([1, layer.out_channels]);
    Ok(input.matmul(weight_t).add(bias_t))
}

#[cfg(feature = "runtime-model-wgpu")]
fn layer_norm_wgpu(
    context_gpu: &mut DecoderWgpuConvContext,
    input: Tensor<DefaultWgpuBackend, 2>,
    rows: usize,
    channels: usize,
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    eps: f32,
) -> Result<Tensor<DefaultWgpuBackend, 2>, String> {
    if rows == 0 || channels == 0 {
        return Ok(input);
    }
    let [input_rows, input_channels] = input.dims();
    if input_rows != rows || input_channels != channels {
        return Err(format!(
            "layer_norm_wgpu: invalid input dims [{},{}], expected [{rows},{channels}]",
            input_rows, input_channels
        ));
    }
    if let Some(weight) = weight
        && weight.len() != channels
    {
        return Err(format!(
            "layer_norm_wgpu: invalid weight len {}, expected {}",
            weight.len(),
            channels
        ));
    }
    if let Some(bias) = bias
        && bias.len() != channels
    {
        return Err(format!(
            "layer_norm_wgpu: invalid bias len {}, expected {}",
            bias.len(),
            channels
        ));
    }

    let mean = input.clone().mean_dim(1);
    let centered = input.sub(mean);
    let var = centered.clone().powf_scalar(2.0).mean_dim(1);
    let mut normalized = centered.mul(var.add_scalar(eps).sqrt().recip());
    if let Some(weight) = weight {
        let weight_t = context_gpu.vector_tensor(weight).reshape([1, channels]);
        normalized = normalized.mul(weight_t);
    }
    if let Some(bias) = bias {
        let bias_t = context_gpu.vector_tensor(bias).reshape([1, channels]);
        normalized = normalized.add(bias_t);
    }
    Ok(normalized)
}

#[cfg(feature = "runtime-model-wgpu")]
fn silu_wgpu(input: Tensor<DefaultWgpuBackend, 2>) -> Tensor<DefaultWgpuBackend, 2> {
    input.clone().mul(sigmoid(input))
}

#[cfg(feature = "runtime-model-wgpu")]
fn tensor_to_vec_f32(
    tensor: Tensor<DefaultWgpuBackend, 2>,
    context: &str,
) -> Result<Vec<f32>, String> {
    tensor
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|err| format!("{context}: failed to read wgpu tensor output: {err:?}"))
}

#[cfg(feature = "runtime-model-wgpu")]
fn convnext_blocks_forward_wgpu_tensor(
    context_gpu: &mut DecoderWgpuConvContext,
    coords: &[[u32; 4]],
    mut state_t: Tensor<DefaultWgpuBackend, 2>,
    stage_idx: usize,
    stage_channels: usize,
    blocks: &[ConvNeXtBlock],
) -> Result<Tensor<DefaultWgpuBackend, 2>, String> {
    let rows = coords.len();
    let [state_rows, state_channels] = state_t.dims();
    if state_rows != rows || state_channels != stage_channels {
        return Err(format!(
            "decoder wgpu convnext tensor dims mismatch: got=[{},{}] expected=[{},{}]",
            state_rows, state_channels, rows, stage_channels
        ));
    }
    for (block_idx, block) in blocks.iter().enumerate() {
        let residual = state_t.clone();
        let config = flex_config_for_layer(&block.conv);
        state_t = context_gpu.forward_with_coords_tensor(
            &config,
            &block.conv,
            state_t,
            format!("stage {stage_idx} block {block_idx} conv(wgpu_math)").as_str(),
            coords,
        )?;
        state_t = layer_norm_wgpu(
            context_gpu,
            state_t,
            rows,
            stage_channels,
            Some(block.norm_weight.as_slice()),
            Some(block.norm_bias.as_slice()),
            LAYER_NORM32_EPS,
        )?;
        state_t = linear_forward_wgpu(
            context_gpu,
            state_t,
            &block.mlp_0,
            format!("stage {stage_idx} block {block_idx} mlp_0(wgpu_math)").as_str(),
        )?;
        state_t = silu_wgpu(state_t);
        state_t = linear_forward_wgpu(
            context_gpu,
            state_t,
            &block.mlp_2,
            format!("stage {stage_idx} block {block_idx} mlp_2(wgpu_math)").as_str(),
        )?;
        state_t = state_t.add(residual);
    }
    Ok(state_t)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DecoderConvImpl {
    Legacy,
    FlexGmm,
    #[cfg(feature = "runtime-model-wgpu")]
    Wgpu,
}

fn decoder_conv_impl() -> DecoderConvImpl {
    match std::env::var("TRELLIS2_DECODER_CONV_IMPL")
        .ok()
        .map(|value| value.trim().to_ascii_lowercase())
        .as_deref()
    {
        Some("legacy") => DecoderConvImpl::Legacy,
        Some("flex") | Some("flex_gmm") => DecoderConvImpl::FlexGmm,
        #[cfg(feature = "runtime-model-wgpu")]
        Some("wgpu") => DecoderConvImpl::Wgpu,
        Some("auto") | None => {
            if env_flag("TRELLIS2_PARITY_STRICT") || env_flag("TRELLIS2_E2E_STRICT") {
                DecoderConvImpl::Legacy
            } else if !env_flag("TRELLIS2_DECODER_DISABLE_WGPU") {
                #[cfg(feature = "runtime-model-wgpu")]
                {
                    DecoderConvImpl::Wgpu
                }
                #[cfg(not(feature = "runtime-model-wgpu"))]
                {
                    DecoderConvImpl::FlexGmm
                }
            } else {
                DecoderConvImpl::FlexGmm
            }
        }
        Some(_) => {
            #[cfg(feature = "runtime-model-wgpu")]
            {
                if !env_flag("TRELLIS2_DECODER_DISABLE_WGPU") {
                    DecoderConvImpl::Wgpu
                } else {
                    DecoderConvImpl::FlexGmm
                }
            }
            #[cfg(not(feature = "runtime-model-wgpu"))]
            {
                DecoderConvImpl::FlexGmm
            }
        }
    }
}

fn decoder_conv_debug_enabled() -> bool {
    std::env::var("TRELLIS2_DECODER_CONV_DEBUG")
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

pub(crate) fn reset_decoder_conv_telemetry() {
    #[cfg(feature = "runtime-model-wgpu")]
    {
        if let Ok(mut state) = decoder_conv_telemetry_state().lock() {
            *state = DecoderConvTelemetryState::default();
        }
    }
}

pub(crate) fn decoder_conv_telemetry() -> DecoderConvTelemetry {
    #[cfg(feature = "runtime-model-wgpu")]
    {
        let Ok(state) = decoder_conv_telemetry_state().lock() else {
            return DecoderConvTelemetry::default();
        };
        let mut blocks = state.blocks.values().cloned().collect::<Vec<_>>();
        blocks.sort_by(|lhs, rhs| {
            rhs.dispatches
                .cmp(&lhs.dispatches)
                .then_with(|| rhs.wgpu_calls.cmp(&lhs.wgpu_calls))
                .then_with(|| rhs.conv_calls.cmp(&lhs.conv_calls))
                .then_with(|| lhs.context.cmp(&rhs.context))
        });
        DecoderConvTelemetry {
            conv_calls: state.total.conv_calls,
            wgpu_calls: state.total.wgpu_calls,
            wgpu_successes: state.total.wgpu_successes,
            wgpu_failures: state.total.wgpu_failures,
            dispatches: state.total.dispatches,
            chunked_calls: state.total.chunked_calls,
            max_chunk_rows: state.total.max_chunk_rows,
            input_bytes: state.total.input_bytes,
            output_bytes: state.total.output_bytes,
            neighbor_elements: state.total.neighbor_elements,
            blocks,
        }
    }
    #[cfg(not(feature = "runtime-model-wgpu"))]
    {
        DecoderConvTelemetry::default()
    }
}

#[cfg(feature = "runtime-model-wgpu")]
fn decoder_conv_telemetry_state() -> &'static Mutex<DecoderConvTelemetryState> {
    DECODER_CONV_TELEMETRY.get_or_init(|| Mutex::new(DecoderConvTelemetryState::default()))
}

#[cfg(feature = "runtime-model-wgpu")]
fn telemetry_update<F>(context: &str, mut update: F)
where
    F: FnMut(&mut DecoderConvBlockTelemetry),
{
    let Ok(mut state) = decoder_conv_telemetry_state().lock() else {
        return;
    };
    update(&mut state.total);
    let block =
        state
            .blocks
            .entry(context.to_string())
            .or_insert_with(|| DecoderConvBlockTelemetry {
                context: context.to_string(),
                ..DecoderConvBlockTelemetry::default()
            });
    update(block);
}

#[cfg(feature = "runtime-model-wgpu")]
fn telemetry_record_conv_call(context: &str) {
    telemetry_update(context, |stats| {
        stats.conv_calls += 1;
    });
}

#[cfg(feature = "runtime-model-wgpu")]
fn telemetry_record_wgpu_call(context: &str) {
    telemetry_update(context, |stats| {
        stats.wgpu_calls += 1;
    });
}

#[cfg(feature = "runtime-model-wgpu")]
fn telemetry_record_wgpu_failure(context: &str) {
    telemetry_update(context, |stats| {
        stats.wgpu_failures += 1;
    });
}

#[cfg(feature = "runtime-model-wgpu")]
fn telemetry_record_wgpu_success(
    context: &str,
    dispatches: u64,
    chunked: bool,
    max_chunk_rows: usize,
    input_bytes: usize,
    output_bytes: usize,
    neighbor_elements: usize,
) {
    telemetry_update(context, |stats| {
        stats.wgpu_successes += 1;
        stats.dispatches += dispatches;
        if chunked {
            stats.chunked_calls += 1;
        }
        stats.max_chunk_rows = stats.max_chunk_rows.max(max_chunk_rows);
        stats.input_bytes = stats.input_bytes.saturating_add(input_bytes as u64);
        stats.output_bytes = stats.output_bytes.saturating_add(output_bytes as u64);
        stats.neighbor_elements = stats
            .neighbor_elements
            .saturating_add(neighbor_elements as u64);
    });
}

#[cfg(feature = "runtime-model-wgpu")]
fn decoder_wgpu_neighbor_from_coords() -> bool {
    let Some(raw) = std::env::var("TRELLIS2_DECODER_WGPU_NEIGHBOR_SOURCE").ok() else {
        return true;
    };
    !matches!(raw.trim().to_ascii_lowercase().as_str(), "rows" | "host")
}

#[cfg(feature = "runtime-model-wgpu")]
fn decoder_wgpu_clear_cache_after_decode() -> bool {
    let Some(raw) = std::env::var("TRELLIS2_DECODER_WGPU_CLEAR_CACHE_AFTER_DECODE").ok() else {
        // Persist decoder tensors by default to avoid per-inference cache rebuild stalls.
        return false;
    };
    !matches!(
        raw.trim().to_ascii_lowercase().as_str(),
        "0" | "false" | "no" | "off"
    )
}

#[cfg(feature = "runtime-model-wgpu")]
fn decoder_wgpu_tensor_cache_max() -> usize {
    std::env::var("TRELLIS2_DECODER_WGPU_TENSOR_CACHE_MAX")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DECODER_WGPU_TENSOR_CACHE_MAX)
}

#[cfg(feature = "runtime-model-wgpu")]
fn decoder_wgpu_use_tensor_cache() -> bool {
    if decoder_wgpu_clear_cache_after_decode() {
        return false;
    }
    decoder_wgpu_tensor_cache_max() > 0
}

#[cfg(feature = "runtime-model-wgpu")]
fn decoder_wgpu_device_math_enabled() -> bool {
    if decoder_conv_impl() != DecoderConvImpl::Wgpu {
        return false;
    }
    std::env::var("TRELLIS2_DECODER_WGPU_DEVICE_MATH")
        .ok()
        .map(|raw| {
            !matches!(
                raw.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "no" | "off"
            )
        })
        .unwrap_or(true)
}

#[cfg(feature = "runtime-model-wgpu")]
fn decoder_wgpu_device_math_allow_fp16() -> bool {
    std::env::var("TRELLIS2_DECODER_WGPU_DEVICE_MATH_FP16")
        .ok()
        .map(|raw| {
            !matches!(
                raw.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "no" | "off"
            )
        })
        .unwrap_or(true)
}

fn flex_config_for_layer(layer: &SparseConvLayer) -> FlexConvConfig {
    FlexConvConfig {
        in_channels: layer.in_channels,
        out_channels: layer.out_channels,
        kernel_d: layer.kernel_d,
        kernel_h: layer.kernel_h,
        kernel_w: layer.kernel_w,
        in_channels_per_group: layer.in_channels_per_group,
        out_channels_per_group: layer.out_channels_per_group,
        groups: layer.groups,
        axis_order: conv_kernel_axis_order(),
        axis_sign: conv_kernel_axis_signs(),
    }
}

fn sparse_subm_conv_forward(
    coords: &[[u32; 4]],
    input: &[f32],
    layer: &SparseConvLayer,
    context: &str,
    conv_cache: &mut DecoderConvCache,
    #[cfg(feature = "runtime-model-wgpu")] wgpu_context: Option<&mut DecoderWgpuConvContext>,
) -> Result<Vec<f32>, String> {
    #[cfg(feature = "runtime-model-wgpu")]
    telemetry_record_conv_call(context);

    let config = flex_config_for_layer(layer);
    let weights = SparseSubmConvWeights {
        weight: layer.weight.as_slice(),
        bias: layer.bias.as_slice(),
    };
    let conv_impl = decoder_conv_impl();

    #[cfg(feature = "runtime-model-wgpu")]
    if conv_impl == DecoderConvImpl::Wgpu
        && let Some(context_gpu) = wgpu_context
    {
        telemetry_record_wgpu_call(context);
        let wgpu_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if decoder_wgpu_neighbor_from_coords() {
                context_gpu.forward_with_coords(&config, layer, input, context, coords)
            } else {
                let (neighbor_key, neighbor_rows) =
                    conv_cache.neighbor_rows_with_key(&config, coords)?;
                context_gpu.forward_with_neighbor_rows(
                    &config,
                    layer,
                    input,
                    context,
                    neighbor_key,
                    neighbor_rows,
                )
            }
        }));
        match wgpu_result {
            Ok(Ok(output)) => return Ok(output),
            Ok(Err(err)) => {
                telemetry_record_wgpu_failure(context);
                if err.contains("BufferTooBig") {
                    context_gpu.wgpu_failed = true;
                    if decoder_conv_debug_enabled() {
                        eprintln!(
                            "burn_trellis: wgpu conv disabling after buffer-too-big in '{context}': {err}"
                        );
                    }
                } else if decoder_conv_debug_enabled() {
                    eprintln!(
                        "burn_trellis: wgpu conv fallback to flex/legacy in '{context}': {err}"
                    );
                }
            }
            Err(payload) => {
                telemetry_record_wgpu_failure(context);
                context_gpu.wgpu_failed = true;
                if decoder_conv_debug_enabled() {
                    let panic_message = panic_payload_to_string(payload);
                    eprintln!(
                        "burn_trellis: wgpu conv panicked in '{context}', fallback to flex/legacy: {panic_message}"
                    );
                }
            }
        }
    }

    if conv_impl != DecoderConvImpl::Legacy {
        let (_neighbor_key, neighbor_rows) = conv_cache.neighbor_rows_with_key(&config, coords)?;
        match sparse_subm_conv_forward_flex_precomputed(
            &config,
            weights,
            input,
            neighbor_rows,
            layer.flex_packed_weight.as_deref(),
        ) {
            Ok(output) => return Ok(output),
            Err(err) => {
                if decoder_conv_debug_enabled() {
                    eprintln!("burn_trellis: flex conv fallback to legacy in '{context}': {err}");
                }
            }
        }
    }

    sparse_subm_conv_forward_legacy(coords, input, layer, context)
}

fn sparse_subm_conv_forward_legacy(
    coords: &[[u32; 4]],
    input: &[f32],
    layer: &SparseConvLayer,
    context: &str,
) -> Result<Vec<f32>, String> {
    let rows = coords.len();
    if rows == 0 {
        return Ok(Vec::new());
    }
    let expected = rows
        .checked_mul(layer.in_channels)
        .ok_or_else(|| format!("{context}: input size overflow"))?;
    if input.len() != expected {
        return Err(format!(
            "{context}: invalid input len {}, expected {} (rows={} in_channels={})",
            input.len(),
            expected,
            rows,
            layer.in_channels
        ));
    }
    if layer.bias.len() != layer.out_channels {
        return Err(format!(
            "{context}: bias len {} does not match out_channels {}",
            layer.bias.len(),
            layer.out_channels
        ));
    }

    let mut output = vec![0.0f32; rows * layer.out_channels];
    for row_idx in 0..rows {
        let base = row_idx * layer.out_channels;
        output[base..base + layer.out_channels].copy_from_slice(layer.bias.as_slice());
    }

    let mut coord_to_row = HashMap::with_capacity(rows.saturating_mul(2));
    for (row_idx, coord) in coords.iter().copied().enumerate() {
        coord_to_row.insert(coord, row_idx);
    }

    let center_d = (layer.kernel_d / 2) as i32;
    let center_h = (layer.kernel_h / 2) as i32;
    let center_w = (layer.kernel_w / 2) as i32;
    let axis_order = conv_kernel_axis_order();
    let axis_sign = conv_kernel_axis_signs();
    for (out_row_idx, out_coord) in coords.iter().copied().enumerate().take(rows) {
        let batch = out_coord[0];
        let ox = out_coord[1] as i32;
        let oy = out_coord[2] as i32;
        let oz = out_coord[3] as i32;
        let out_base = out_row_idx * layer.out_channels;

        for kd_idx in 0..layer.kernel_d {
            for kh_idx in 0..layer.kernel_h {
                for kw_idx in 0..layer.kernel_w {
                    let deltas = [
                        axis_sign[0] * (kd_idx as i32 - center_d),
                        axis_sign[1] * (kh_idx as i32 - center_h),
                        axis_sign[2] * (kw_idx as i32 - center_w),
                    ];
                    let mut spatial = [ox, oy, oz];
                    spatial[axis_order[0]] += deltas[0];
                    spatial[axis_order[1]] += deltas[1];
                    spatial[axis_order[2]] += deltas[2];
                    if spatial[0] < 0 || spatial[1] < 0 || spatial[2] < 0 {
                        continue;
                    }
                    let neighbor = [
                        batch,
                        spatial[0] as u32,
                        spatial[1] as u32,
                        spatial[2] as u32,
                    ];
                    let Some(in_row_idx) = coord_to_row.get(&neighbor).copied() else {
                        continue;
                    };
                    let in_row = &input
                        [in_row_idx * layer.in_channels..(in_row_idx + 1) * layer.in_channels];
                    for group_idx in 0..layer.groups {
                        let in_group_base = group_idx * layer.in_channels_per_group;
                        let out_group_base = group_idx * layer.out_channels_per_group;
                        for out_local in 0..layer.out_channels_per_group {
                            let out_idx = out_group_base + out_local;
                            let weight_base =
                                (((out_idx * layer.kernel_d + kd_idx) * layer.kernel_h + kh_idx)
                                    * layer.kernel_w
                                    + kw_idx)
                                    * layer.in_channels_per_group;
                            let mut accum = 0.0f32;
                            for in_local in 0..layer.in_channels_per_group {
                                accum += in_row[in_group_base + in_local]
                                    * layer.weight[weight_base + in_local];
                            }
                            output[out_base + out_idx] += accum;
                        }
                    }
                }
            }
        }
    }

    Ok(output)
}

fn conv_kernel_axis_order() -> [usize; 3] {
    if let Some(value) = std::env::var("TRELLIS2_CONV_AXIS_ORDER")
        .ok()
        .map(|value| value.trim().to_ascii_lowercase())
    {
        return match value.as_str() {
            "xzy" => [0, 2, 1],
            "yxz" => [1, 0, 2],
            "yzx" => [1, 2, 0],
            "zxy" => [2, 0, 1],
            "zyx" => [2, 1, 0],
            _ => [0, 1, 2],
        };
    }

    if env_flag("TRELLIS2_PARITY_STRICT") {
        // Empirically closest to current TRELLIS2 decoder hook traces.
        [1, 0, 2]
    } else {
        [0, 1, 2]
    }
}

fn conv_kernel_axis_signs() -> [i32; 3] {
    let raw = std::env::var("TRELLIS2_CONV_AXIS_SIGN")
        .ok()
        .map(|value| value.trim().to_ascii_lowercase())
        .unwrap_or_else(|| {
            if env_flag("TRELLIS2_PARITY_STRICT") {
                "---".to_string()
            } else {
                "+++".to_string()
            }
        });
    let mut signs = [1i32, 1, 1];
    for (idx, ch) in raw.chars().take(3).enumerate() {
        signs[idx] = if ch == '-' { -1 } else { 1 };
    }
    signs
}

fn layer_norm_inplace(
    data: &mut [f32],
    rows: usize,
    channels: usize,
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    eps: f32,
) -> Result<(), String> {
    if rows == 0 || channels == 0 {
        return Ok(());
    }
    if data.len() != rows * channels {
        return Err(format!(
            "layer_norm_inplace: invalid data len {}, expected {}",
            data.len(),
            rows * channels
        ));
    }
    if let Some(weight) = weight
        && weight.len() != channels
    {
        return Err(format!(
            "layer_norm_inplace: invalid weight len {}, expected {}",
            weight.len(),
            channels
        ));
    }
    if let Some(bias) = bias
        && bias.len() != channels
    {
        return Err(format!(
            "layer_norm_inplace: invalid bias len {}, expected {}",
            bias.len(),
            channels
        ));
    }

    for row_idx in 0..rows {
        let base = row_idx * channels;
        let row = &mut data[base..base + channels];
        let mean = row.iter().copied().sum::<f32>() / channels as f32;
        let var = row
            .iter()
            .map(|value| {
                let centered = *value - mean;
                centered * centered
            })
            .sum::<f32>()
            / channels as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        for ch in 0..channels {
            let mut value = (row[ch] - mean) * inv_std;
            if let Some(weight) = weight {
                value *= weight[ch];
            }
            if let Some(bias) = bias {
                value += bias[ch];
            }
            row[ch] = value;
        }
    }
    Ok(())
}

fn silu_inplace(data: &mut [f32]) {
    for value in data {
        *value = *value / (1.0 + (-*value).exp());
    }
}

fn quantize_f16_inplace(data: &mut [f32]) {
    for value in data {
        *value = f16::from_f32(*value).to_f32();
    }
}

fn row_center_logits(data: &mut [f32], rows: usize) {
    if rows == 0 {
        return;
    }
    if data.len() != rows * 8 {
        return;
    }
    for row_idx in 0..rows {
        let row = &mut data[row_idx * 8..(row_idx + 1) * 8];
        let mean = row.iter().copied().sum::<f32>() / 8.0;
        for value in row {
            *value -= mean;
        }
    }
}

fn should_center_subdivision_logits() -> bool {
    std::env::var("TRELLIS2_DECODER_CENTER_SUBDIV_LOGITS")
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn decoder_force_fp32() -> bool {
    std::env::var("TRELLIS2_DECODER_FORCE_FP32")
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn add_inplace(lhs: &mut [f32], rhs: &[f32]) {
    if lhs.len() != rhs.len() {
        return;
    }
    for (left, right) in lhs.iter_mut().zip(rhs.iter()) {
        *left += *right;
    }
}

fn logits_to_mask(
    logits: &[f32],
    rows: usize,
    enforce_non_empty: bool,
) -> Result<Vec<[bool; 8]>, String> {
    if logits.len() != rows * 8 {
        return Err(format!(
            "subdivision logits len {} does not match rows*8={}",
            logits.len(),
            rows * 8
        ));
    }
    let mut out = Vec::with_capacity(rows);
    let max_children = decoder_max_children_per_parent();
    for row_idx in 0..rows {
        let mut mask = [false; 8];
        let row = &logits[row_idx * 8..(row_idx + 1) * 8];
        for child in 0..8 {
            mask[child] = row[child] > 0.0;
        }
        if let Some(max_children) = max_children {
            let selected = mask.iter().filter(|flag| **flag).count();
            if selected > max_children {
                let mut order = (0..8usize).collect::<Vec<_>>();
                order.sort_by(|a, b| {
                    row[*b]
                        .partial_cmp(&row[*a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let mut limited = [false; 8];
                for idx in order.into_iter().take(max_children) {
                    limited[idx] = true;
                }
                mask = limited;
            }
        }
        if enforce_non_empty && !mask.iter().any(|flag| *flag) {
            let mut best_idx = 0usize;
            let mut best_val = row[0];
            for (idx, value) in row.iter().enumerate().skip(1) {
                if *value > best_val {
                    best_val = *value;
                    best_idx = idx;
                }
            }
            mask[best_idx] = true;
        }
        out.push(mask);
    }
    Ok(out)
}

fn decoder_max_children_per_parent() -> Option<usize> {
    if let Ok(value) = std::env::var("TRELLIS2_DECODER_MAX_CHILDREN_PER_PARENT")
        && let Ok(parsed) = value.trim().parse::<usize>()
    {
        if parsed == 0 {
            return None;
        }
        return Some(parsed.min(8));
    }

    if env_flag("TRELLIS2_PARITY_STRICT") || env_flag("TRELLIS2_DECODER_UNCAPPED") {
        return None;
    }
    // Default to uncapped subdivision for decoder parity.
    None
}

#[cfg(feature = "runtime-model-wgpu")]
fn panic_payload_to_string(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(msg) = payload.downcast_ref::<&str>() {
        (*msg).to_string()
    } else if let Some(msg) = payload.downcast_ref::<String>() {
        msg.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

#[cfg(feature = "runtime-model-wgpu")]
fn decoder_wgpu_max_output_bytes() -> usize {
    std::env::var("TRELLIS2_DECODER_WGPU_MAX_OUTPUT_BYTES")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(256 * 1024 * 1024)
}

#[cfg(feature = "runtime-model-wgpu")]
fn decoder_wgpu_max_input_bytes() -> usize {
    std::env::var("TRELLIS2_DECODER_WGPU_MAX_INPUT_BYTES")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(1024 * 1024 * 1024)
}

#[cfg(feature = "runtime-model-wgpu")]
fn decoder_wgpu_chunk_rows(rows: usize, bytes_per_row: usize, max_output_bytes: usize) -> usize {
    if rows == 0 {
        return 1;
    }
    if let Some(explicit) = std::env::var("TRELLIS2_DECODER_WGPU_CHUNK_ROWS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
    {
        return explicit.min(rows).max(1);
    }
    if bytes_per_row == 0 {
        return rows;
    }
    let by_bytes = (max_output_bytes / bytes_per_row).max(1).min(rows);
    let aligned = by_bytes - (by_bytes % 64);
    if aligned > 0 { aligned } else { by_bytes }
}

fn env_flag(key: &str) -> bool {
    std::env::var(key)
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn channel2spatial(
    coords: &[[u32; 4]],
    feats: &[f32],
    in_channels: usize,
    subdivision_mask: &[[bool; 8]],
) -> Result<(Vec<[u32; 4]>, Vec<f32>), String> {
    let rows = coords.len();
    if rows == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    if feats.len() != rows * in_channels {
        return Err(format!(
            "channel2spatial: invalid feats len {}, expected {}",
            feats.len(),
            rows * in_channels
        ));
    }
    if !in_channels.is_multiple_of(8) {
        return Err(format!(
            "channel2spatial: in_channels={} is not divisible by 8",
            in_channels
        ));
    }
    if subdivision_mask.len() != rows {
        return Err(format!(
            "channel2spatial: subdivision rows {} do not match coords rows {}",
            subdivision_mask.len(),
            rows
        ));
    }

    let out_channels = in_channels / 8;
    let mut out_coords = Vec::new();
    let mut out_feats = Vec::new();
    for row_idx in 0..rows {
        let coord = coords[row_idx];
        let row_feats = &feats[row_idx * in_channels..(row_idx + 1) * in_channels];
        for (child, selected) in subdivision_mask[row_idx].iter().enumerate().take(8usize) {
            if !*selected {
                continue;
            }
            let cx = (child & 1) as u32;
            let cy = ((child >> 1) & 1) as u32;
            let cz = ((child >> 2) & 1) as u32;
            out_coords.push([
                coord[0],
                coord[1].saturating_mul(2).saturating_add(cx),
                coord[2].saturating_mul(2).saturating_add(cy),
                coord[3].saturating_mul(2).saturating_add(cz),
            ]);
            let child_base = child * out_channels;
            out_feats.extend_from_slice(&row_feats[child_base..child_base + out_channels]);
        }
    }
    Ok((out_coords, out_feats))
}

fn repeat_interleave_channels(
    feats: &[f32],
    rows: usize,
    in_channels: usize,
    repeat_factor: usize,
) -> Vec<f32> {
    if rows == 0 || in_channels == 0 || repeat_factor == 0 {
        return Vec::new();
    }
    let out_channels = in_channels * repeat_factor;
    let mut out = Vec::with_capacity(rows * out_channels);
    for row_idx in 0..rows {
        let row = &feats[row_idx * in_channels..(row_idx + 1) * in_channels];
        for value in row {
            for _ in 0..repeat_factor {
                out.push(*value);
            }
        }
    }
    out
}

fn map_guide_subdivision_logits(
    coords: &[[u32; 4]],
    guide: &SparseSubdivisionLogits,
) -> Result<Vec<f32>, String> {
    if guide.logits.len() != guide.coords.len() * 8 {
        return Err(format!(
            "guide subdivision logits invalid length: logits={} coords={}",
            guide.logits.len(),
            guide.coords.len()
        ));
    }
    let mut map = HashMap::with_capacity(guide.coords.len() * 2);
    for (idx, coord) in guide.coords.iter().enumerate() {
        let row = &guide.logits[idx * 8..(idx + 1) * 8];
        map.insert(*coord, row.to_vec());
    }

    let mut out = Vec::with_capacity(coords.len() * 8);
    let strict = env_flag("TRELLIS2_PARITY_STRICT");
    for coord in coords {
        if let Some(row) = map.get(coord) {
            out.extend_from_slice(row);
        } else if strict {
            return Err(format!(
                "guide subdivision logits missing coord {:?} in parity strict mode",
                coord
            ));
        } else {
            // If the guide row is missing, keep all children disabled.
            out.extend_from_slice(&[-1.0; 8]);
        }
    }
    Ok(out)
}

fn spatial_shape_from_coords(coords: &[[u32; 4]]) -> [u32; 3] {
    if coords.is_empty() {
        return [1, 1, 1];
    }
    let mut max_x = 0u32;
    let mut max_y = 0u32;
    let mut max_z = 0u32;
    for coord in coords {
        max_x = max_x.max(coord[1]);
        max_y = max_y.max(coord[2]);
        max_z = max_z.max(coord[3]);
    }
    [
        max_x.saturating_add(1),
        max_y.saturating_add(1),
        max_z.saturating_add(1),
    ]
}

fn load_weight_backing(path: &Path) -> Result<WeightsBacking, String> {
    if path
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("bpk"))
    {
        let bytes = load_burnpack_blob_bytes(path)?;
        return Ok(WeightsBacking::Bytes(bytes));
    }

    let file = File::open(path).map_err(|err| {
        format!(
            "failed to open sparse decoder weights '{}': {err}",
            path.display()
        )
    })?;
    let mmap = unsafe { MmapOptions::new().map(&file) }.map_err(|err| {
        format!(
            "failed to mmap sparse decoder weights '{}': {err}",
            path.display()
        )
    })?;
    Ok(WeightsBacking::Mmap(mmap))
}

fn load_burnpack_blob_bytes(path: &Path) -> Result<Vec<u8>, String> {
    let metadata_path = metadata_path(path);
    let metadata: BlobMetadata =
        serde_json::from_slice(&std::fs::read(&metadata_path).map_err(|err| {
            format!(
                "failed to read burnpack metadata '{}': {err}",
                metadata_path.display()
            )
        })?)
        .map_err(|err| {
            format!(
                "failed to parse burnpack metadata '{}': {err}",
                metadata_path.display()
            )
        })?;

    match load_blob_bytes_with_backend::<burn::backend::NdArray<f32, u8>>(path, metadata.bytes_len)
    {
        Ok(bytes) => Ok(bytes),
        Err(u8_err) => load_blob_bytes_with_backend::<burn::backend::NdArray<f32, i64>>(
            path,
            metadata.bytes_len,
        )
        .map_err(|i64_err| {
            format!(
                "failed to load blob burnpack '{}' (u8 backend: {u8_err}; i64 fallback: {i64_err})",
                path.display()
            )
        }),
    }
}

fn load_blob_bytes_with_backend<B: Backend>(
    path: &Path,
    bytes_len: usize,
) -> Result<Vec<u8>, String>
where
    B::Device: Default,
{
    let device = <B as Backend>::Device::default();
    let zeros = Tensor::<B, 1, Int>::zeros([bytes_len], &device);
    let mut blob = BinaryBlob {
        bytes: Param::initialized(ParamId::new(), zeros),
    };

    let mut store = BurnpackStore::from_file(path).validate(true);
    blob.load_from(&mut store)
        .map_err(|err| format!("failed to load burnpack '{}': {err}", path.display()))?;

    let bytes = blob
        .bytes
        .val()
        .into_data()
        .convert::<u8>()
        .to_vec::<u8>()
        .map_err(|err| format!("failed to materialize burnpack bytes: {err:?}"))?;

    if bytes.len() != bytes_len {
        return Err(format!(
            "burnpack byte length mismatch for '{}': expected {}, got {}",
            path.display(),
            bytes_len,
            bytes.len()
        ));
    }
    Ok(bytes)
}

fn metadata_path(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("model.bpk");
    path.with_file_name(format!("{file_name}.meta.json"))
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
    let source_f16 = with_file_stem_suffix(&source, F16_SUFFIX);
    let prefer_f16 = prefer_f16_burnpack();
    let candidates = if prefer_f16 {
        vec![burnpack_f16, burnpack, source_f16, source]
    } else {
        vec![burnpack, burnpack_f16, source, source_f16]
    };
    candidates
        .into_iter()
        .filter(|path| path.exists())
        .collect::<Vec<_>>()
}

fn prefer_f16_burnpack() -> bool {
    let precision = std::env::var("TRELLIS2_BPK_PRECISION")
        .ok()
        .or_else(|| std::env::var("BURN_SYNTH_BPK_PRECISION").ok());
    match precision
        .as_deref()
        .map(str::trim)
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("f32" | "fp32" | "float32" | "32") => false,
        Some("f16" | "fp16" | "float16" | "half" | "16") => true,
        Some(_) | None => true,
    }
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
    use std::fs;
    use std::sync::Mutex;
    use std::time::Instant;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[cfg(feature = "runtime-model-wgpu")]
    use super::decoder_wgpu_neighbor_from_coords;
    use super::{
        DecoderConvCache, DecoderConvImpl, LinearLayer, SparseConvLayer, decoder_conv_impl,
        linear_forward, logits_to_mask, resolve_model_weight_candidates, sparse_subm_conv_forward,
        sparse_subm_conv_forward_legacy,
    };
    #[cfg(feature = "runtime-model-wgpu")]
    use super::{
        decoder_wgpu_clear_cache_after_decode, decoder_wgpu_device_math_allow_fp16,
        decoder_wgpu_device_math_enabled, decoder_wgpu_tensor_cache_max,
        decoder_wgpu_use_tensor_cache,
    };

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn make_unit_conv_3x1x1(weight: [f32; 3]) -> SparseConvLayer {
        SparseConvLayer {
            in_channels: 1,
            out_channels: 1,
            kernel_d: 3,
            kernel_h: 1,
            kernel_w: 1,
            in_channels_per_group: 1,
            out_channels_per_group: 1,
            groups: 1,
            weight: weight.to_vec(),
            bias: vec![0.0],
            flex_packed_weight: None,
        }
    }

    #[derive(Clone)]
    struct Lcg {
        state: u64,
    }

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self { state: seed | 1 }
        }

        fn next_f32(&mut self) -> f32 {
            self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let bits = ((self.state >> 40) as u32) | 1;
            (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
        }
    }

    #[test]
    fn sparse_conv_uses_neighbor_voxels() {
        let _guard = ENV_LOCK.lock().expect("lock env");
        unsafe {
            std::env::remove_var("TRELLIS2_PARITY_STRICT");
            std::env::set_var("TRELLIS2_CONV_AXIS_ORDER", "xyz");
            std::env::set_var("TRELLIS2_CONV_AXIS_SIGN", "+++");
        }
        let coords = vec![[0, 0, 0, 0], [0, 1, 0, 0]];
        let input = vec![1.0f32, 2.0f32];
        // kernel offsets: [-1, 0, +1]
        let layer = make_unit_conv_3x1x1([10.0, 1.0, 100.0]);

        let output = sparse_subm_conv_forward(
            coords.as_slice(),
            input.as_slice(),
            &layer,
            "test conv",
            &mut DecoderConvCache::default(),
            #[cfg(feature = "runtime-model-wgpu")]
            None,
        )
        .expect("sparse conv should succeed");
        assert_eq!(output.len(), 2);
        // x=0: center(1*1) + right-neighbor(2*100)
        assert!((output[0] - 201.0).abs() < 1.0e-5);
        // x=1: left-neighbor(1*10) + center(2*1)
        assert!((output[1] - 12.0).abs() < 1.0e-5);
        unsafe {
            std::env::remove_var("TRELLIS2_CONV_AXIS_ORDER");
            std::env::remove_var("TRELLIS2_CONV_AXIS_SIGN");
        }
    }

    #[test]
    fn sparse_conv_flex_matches_legacy_path() {
        let _guard = ENV_LOCK.lock().expect("lock env");
        unsafe {
            std::env::remove_var("TRELLIS2_PARITY_STRICT");
            std::env::set_var("TRELLIS2_CONV_AXIS_ORDER", "xyz");
            std::env::set_var("TRELLIS2_CONV_AXIS_SIGN", "+++");
            std::env::set_var("TRELLIS2_DECODER_CONV_IMPL", "flex_gmm");
        }
        let mut rng = Lcg::new(123);
        let layer = SparseConvLayer {
            in_channels: 4,
            out_channels: 6,
            kernel_d: 3,
            kernel_h: 1,
            kernel_w: 1,
            in_channels_per_group: 2,
            out_channels_per_group: 3,
            groups: 2,
            weight: (0..(6 * 3 * 2)).map(|_| rng.next_f32()).collect(),
            bias: (0..6).map(|_| rng.next_f32()).collect(),
            flex_packed_weight: None,
        };
        let coords: Vec<[u32; 4]> = (0..32u32).map(|x| [0, x, 0, 0]).collect();
        let input: Vec<f32> = (0..coords.len() * layer.in_channels)
            .map(|_| rng.next_f32())
            .collect();
        let legacy =
            sparse_subm_conv_forward_legacy(coords.as_slice(), input.as_slice(), &layer, "legacy")
                .expect("legacy conv");
        let fused = sparse_subm_conv_forward(
            coords.as_slice(),
            input.as_slice(),
            &layer,
            "fused",
            &mut DecoderConvCache::default(),
            #[cfg(feature = "runtime-model-wgpu")]
            None,
        )
        .expect("fused conv");
        assert_eq!(legacy.len(), fused.len());
        for (idx, (lhs, rhs)) in legacy.iter().zip(fused.iter()).enumerate() {
            let diff = (lhs - rhs).abs();
            assert!(
                diff <= 1.0e-5,
                "mismatch idx={idx}: legacy={lhs} fused={rhs} diff={diff}"
            );
        }
        unsafe {
            std::env::remove_var("TRELLIS2_DECODER_CONV_IMPL");
            std::env::remove_var("TRELLIS2_CONV_AXIS_ORDER");
            std::env::remove_var("TRELLIS2_CONV_AXIS_SIGN");
        }
    }

    #[test]
    fn decoder_neighbor_cache_reuses_across_coord_allocations() {
        let _guard = ENV_LOCK.lock().expect("lock env");
        unsafe {
            std::env::remove_var("TRELLIS2_CONV_AXIS_ORDER");
            std::env::remove_var("TRELLIS2_CONV_AXIS_SIGN");
        }
        let layer = make_unit_conv_3x1x1([0.1, 0.2, 0.3]);
        let config = super::flex_config_for_layer(&layer);
        let coords_a: Vec<[u32; 4]> = (0..16u32).map(|x| [0, x, 0, 0]).collect();
        let coords_b = coords_a.clone();
        let mut cache = DecoderConvCache::default();

        let key_a = {
            let (key, rows) = cache
                .neighbor_rows_with_key(&config, coords_a.as_slice())
                .expect("cache build");
            assert_eq!(rows.len(), coords_a.len() * 3);
            key
        };
        let len_after_a = cache.neighbor_rows.len();
        let key_b = {
            let (key, rows) = cache
                .neighbor_rows_with_key(&config, coords_b.as_slice())
                .expect("cache hit");
            assert_eq!(rows.len(), coords_b.len() * 3);
            key
        };

        assert_eq!(key_a, key_b);
        assert_eq!(cache.neighbor_rows.len(), len_after_a);
    }

    #[test]
    fn decoder_neighbor_cache_reuse_reduces_repeated_conv_time() {
        let _guard = ENV_LOCK.lock().expect("lock env");
        unsafe {
            std::env::set_var("TRELLIS2_DECODER_CONV_IMPL", "flex_gmm");
            std::env::set_var("TRELLIS2_CONV_AXIS_ORDER", "xyz");
            std::env::set_var("TRELLIS2_CONV_AXIS_SIGN", "+++");
        }
        let mut rng = Lcg::new(991);
        let layer = SparseConvLayer {
            in_channels: 64,
            out_channels: 128,
            kernel_d: 3,
            kernel_h: 3,
            kernel_w: 3,
            in_channels_per_group: 64,
            out_channels_per_group: 128,
            groups: 1,
            weight: (0..(128 * 3 * 3 * 3 * 64))
                .map(|_| rng.next_f32())
                .collect(),
            bias: (0..128).map(|_| rng.next_f32()).collect(),
            flex_packed_weight: None,
        };
        let coords: Vec<[u32; 4]> = (0..4096u32).map(|x| [0, x, 0, 0]).collect();
        let input: Vec<f32> = (0..coords.len() * layer.in_channels)
            .map(|_| rng.next_f32())
            .collect();
        let iterations = 12usize;

        let cold_start = Instant::now();
        for _ in 0..iterations {
            let _ = sparse_subm_conv_forward(
                coords.as_slice(),
                input.as_slice(),
                &layer,
                "cold",
                &mut DecoderConvCache::default(),
                #[cfg(feature = "runtime-model-wgpu")]
                None,
            )
            .expect("cold conv");
        }
        let cold = cold_start.elapsed();

        let mut warm_cache = DecoderConvCache::default();
        let warm_start = Instant::now();
        for _ in 0..iterations {
            let _ = sparse_subm_conv_forward(
                coords.as_slice(),
                input.as_slice(),
                &layer,
                "warm",
                &mut warm_cache,
                #[cfg(feature = "runtime-model-wgpu")]
                None,
            )
            .expect("warm conv");
        }
        let warm = warm_start.elapsed();
        eprintln!(
            "decoder cache perf: cold={:?} warm={:?} ratio={:.3}",
            cold,
            warm,
            warm.as_secs_f64() / cold.as_secs_f64().max(1.0e-12)
        );
        assert!(
            warm <= cold,
            "expected persistent neighbor cache to be no slower than rebuilding; cold={cold:?} warm={warm:?}"
        );

        unsafe {
            std::env::remove_var("TRELLIS2_DECODER_CONV_IMPL");
            std::env::remove_var("TRELLIS2_CONV_AXIS_ORDER");
            std::env::remove_var("TRELLIS2_CONV_AXIS_SIGN");
        }
    }

    #[test]
    fn decoder_default_child_cap_is_uncapped_without_strict_mode() {
        let _guard = ENV_LOCK.lock().expect("lock env");
        unsafe {
            std::env::remove_var("TRELLIS2_PARITY_STRICT");
            std::env::remove_var("TRELLIS2_DECODER_UNCAPPED");
            std::env::remove_var("TRELLIS2_DECODER_MAX_CHILDREN_PER_PARENT");
        }
        let logits = vec![1.0f32; 8];
        let mask = logits_to_mask(logits.as_slice(), 1, true).expect("mask");
        let selected = mask[0].iter().filter(|flag| **flag).count();
        assert_eq!(selected, 8);
    }

    #[test]
    fn parity_strict_defaults_to_uncapped_children() {
        let _guard = ENV_LOCK.lock().expect("lock env");
        unsafe {
            std::env::set_var("TRELLIS2_PARITY_STRICT", "1");
            std::env::remove_var("TRELLIS2_DECODER_UNCAPPED");
            std::env::remove_var("TRELLIS2_DECODER_MAX_CHILDREN_PER_PARENT");
        }
        let logits = vec![1.0f32; 8];
        let mask = logits_to_mask(logits.as_slice(), 1, true).expect("mask");
        let selected = mask[0].iter().filter(|flag| **flag).count();
        assert_eq!(selected, 8);
        unsafe {
            std::env::remove_var("TRELLIS2_PARITY_STRICT");
        }
    }

    #[test]
    fn explicit_zero_child_cap_env_means_uncapped() {
        let _guard = ENV_LOCK.lock().expect("lock env");
        unsafe {
            std::env::remove_var("TRELLIS2_PARITY_STRICT");
            std::env::remove_var("TRELLIS2_DECODER_UNCAPPED");
            std::env::set_var("TRELLIS2_DECODER_MAX_CHILDREN_PER_PARENT", "0");
        }
        let logits = vec![1.0f32; 8];
        let mask = logits_to_mask(logits.as_slice(), 1, true).expect("mask");
        let selected = mask[0].iter().filter(|flag| **flag).count();
        assert_eq!(selected, 8);
        unsafe {
            std::env::remove_var("TRELLIS2_DECODER_MAX_CHILDREN_PER_PARENT");
        }
    }

    #[test]
    fn decoder_conv_auto_defaults_to_flex() {
        let _guard = ENV_LOCK.lock().expect("lock env");
        unsafe {
            std::env::remove_var("TRELLIS2_DECODER_CONV_IMPL");
            std::env::remove_var("TRELLIS2_PARITY_STRICT");
            std::env::remove_var("TRELLIS2_E2E_STRICT");
            std::env::remove_var("TRELLIS2_DECODER_DISABLE_WGPU");
        }
        #[cfg(feature = "runtime-model-wgpu")]
        assert_eq!(decoder_conv_impl(), DecoderConvImpl::Wgpu);
        #[cfg(not(feature = "runtime-model-wgpu"))]
        assert_eq!(decoder_conv_impl(), DecoderConvImpl::FlexGmm);
    }

    #[test]
    fn decoder_conv_auto_uses_legacy_in_strict_mode() {
        let _guard = ENV_LOCK.lock().expect("lock env");
        unsafe {
            std::env::remove_var("TRELLIS2_DECODER_CONV_IMPL");
            std::env::set_var("TRELLIS2_E2E_STRICT", "1");
        }
        assert_eq!(decoder_conv_impl(), DecoderConvImpl::Legacy);
        unsafe {
            std::env::remove_var("TRELLIS2_E2E_STRICT");
        }
    }

    #[cfg(feature = "runtime-model-wgpu")]
    #[test]
    fn decoder_wgpu_neighbor_source_defaults_to_coords() {
        let _guard = ENV_LOCK.lock().expect("lock env");
        unsafe {
            std::env::remove_var("TRELLIS2_DECODER_WGPU_NEIGHBOR_SOURCE");
        }
        assert!(decoder_wgpu_neighbor_from_coords());
        unsafe {
            std::env::set_var("TRELLIS2_DECODER_WGPU_NEIGHBOR_SOURCE", "rows");
        }
        assert!(!decoder_wgpu_neighbor_from_coords());
        unsafe {
            std::env::set_var("TRELLIS2_DECODER_WGPU_NEIGHBOR_SOURCE", "coords");
        }
        assert!(decoder_wgpu_neighbor_from_coords());
        unsafe {
            std::env::remove_var("TRELLIS2_DECODER_WGPU_NEIGHBOR_SOURCE");
        }
    }

    #[cfg(feature = "runtime-model-wgpu")]
    #[test]
    fn decoder_wgpu_cache_controls_have_expected_defaults() {
        let _guard = ENV_LOCK.lock().expect("lock env");
        unsafe {
            std::env::remove_var("TRELLIS2_DECODER_WGPU_CLEAR_CACHE_AFTER_DECODE");
            std::env::remove_var("TRELLIS2_DECODER_WGPU_TENSOR_CACHE_MAX");
        }
        assert!(!decoder_wgpu_clear_cache_after_decode());
        assert_eq!(decoder_wgpu_tensor_cache_max(), 64);
        assert!(decoder_wgpu_use_tensor_cache());
        unsafe {
            std::env::set_var("TRELLIS2_DECODER_WGPU_CLEAR_CACHE_AFTER_DECODE", "1");
            std::env::set_var("TRELLIS2_DECODER_WGPU_TENSOR_CACHE_MAX", "8");
        }
        assert!(decoder_wgpu_clear_cache_after_decode());
        assert_eq!(decoder_wgpu_tensor_cache_max(), 8);
        assert!(!decoder_wgpu_use_tensor_cache());
        unsafe {
            std::env::remove_var("TRELLIS2_DECODER_WGPU_CLEAR_CACHE_AFTER_DECODE");
            std::env::remove_var("TRELLIS2_DECODER_WGPU_TENSOR_CACHE_MAX");
        }
    }

    #[cfg(feature = "runtime-model-wgpu")]
    #[test]
    fn decoder_wgpu_device_math_control_defaults_enabled() {
        let _guard = ENV_LOCK.lock().expect("lock env");
        unsafe {
            std::env::remove_var("TRELLIS2_DECODER_WGPU_DEVICE_MATH");
            std::env::remove_var("TRELLIS2_DECODER_WGPU_DEVICE_MATH_FP16");
            std::env::set_var("TRELLIS2_DECODER_CONV_IMPL", "wgpu");
            std::env::remove_var("TRELLIS2_DECODER_DISABLE_WGPU");
        }
        assert!(decoder_wgpu_device_math_enabled());
        assert!(decoder_wgpu_device_math_allow_fp16());
        unsafe {
            std::env::set_var("TRELLIS2_DECODER_WGPU_DEVICE_MATH", "0");
        }
        assert!(!decoder_wgpu_device_math_enabled());
        unsafe {
            std::env::set_var("TRELLIS2_DECODER_WGPU_DEVICE_MATH", "1");
            std::env::set_var("TRELLIS2_DECODER_WGPU_DEVICE_MATH_FP16", "0");
        }
        assert!(!decoder_wgpu_device_math_allow_fp16());
        unsafe {
            std::env::set_var("TRELLIS2_DECODER_WGPU_DEVICE_MATH_FP16", "1");
            std::env::set_var("TRELLIS2_DECODER_CONV_IMPL", "legacy");
        }
        assert!(!decoder_wgpu_device_math_enabled());
        unsafe {
            std::env::remove_var("TRELLIS2_DECODER_WGPU_DEVICE_MATH");
            std::env::remove_var("TRELLIS2_DECODER_WGPU_DEVICE_MATH_FP16");
            std::env::remove_var("TRELLIS2_DECODER_CONV_IMPL");
        }
    }

    #[test]
    fn linear_forward_matches_naive_matmul() {
        let layer = LinearLayer {
            in_channels: 3,
            out_channels: 2,
            // [out, in]
            weight: vec![
                1.0, 2.0, 3.0, // out0
                -1.0, 0.5, 4.0, // out1
            ],
            bias: vec![0.25, -0.5],
        };
        let input = vec![
            2.0, -1.0, 0.5, // row0
            -3.0, 4.0, 1.0, // row1
        ];
        let output = linear_forward(input.as_slice(), 2, &layer, "test linear")
            .expect("linear forward should succeed");
        assert_eq!(output.len(), 4);

        let mut expected = Vec::new();
        for row in 0..2 {
            let x = &input[row * 3..(row + 1) * 3];
            // out0
            expected.push(layer.bias[0] + x[0] * 1.0 + x[1] * 2.0 + x[2] * 3.0);
            // out1
            expected.push(layer.bias[1] - x[0] + x[1] * 0.5 + x[2] * 4.0);
        }
        for (got, want) in output.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1.0e-5, "got={got} want={want}");
        }
    }

    #[test]
    fn model_weight_candidates_prefer_bpk_variants() {
        let _guard = ENV_LOCK.lock().expect("lock env");
        unsafe {
            std::env::remove_var("TRELLIS2_BPK_PRECISION");
            std::env::remove_var("BURN_SYNTH_BPK_PRECISION");
        }

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock drift")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("burn_trellis_decoder_candidates_{unique}"));
        let ckpts = root.join("ckpts");
        fs::create_dir_all(&ckpts).expect("create ckpts");
        fs::write(ckpts.join("shape.safetensors"), b"safe").expect("write safetensors");
        fs::write(ckpts.join("shape.bpk"), b"bpk").expect("write bpk");
        fs::write(ckpts.join("shape_f16.bpk"), b"bpk_f16").expect("write f16 bpk");

        let candidates = resolve_model_weight_candidates("ckpts/shape", root.as_path(), None);
        assert!(!candidates.is_empty(), "expected weight candidates");
        assert_eq!(candidates[0], ckpts.join("shape_f16.bpk"));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn model_weight_candidates_honor_f32_preference() {
        let _guard = ENV_LOCK.lock().expect("lock env");
        unsafe {
            std::env::set_var("TRELLIS2_BPK_PRECISION", "f32");
            std::env::remove_var("BURN_SYNTH_BPK_PRECISION");
        }

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock drift")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("burn_trellis_decoder_candidates_{unique}"));
        let ckpts = root.join("ckpts");
        fs::create_dir_all(&ckpts).expect("create ckpts");
        fs::write(ckpts.join("shape.safetensors"), b"safe").expect("write safetensors");
        fs::write(ckpts.join("shape.bpk"), b"bpk").expect("write bpk");
        fs::write(ckpts.join("shape_f16.bpk"), b"bpk_f16").expect("write f16 bpk");

        let candidates = resolve_model_weight_candidates("ckpts/shape", root.as_path(), None);
        assert!(!candidates.is_empty(), "expected weight candidates");
        assert_eq!(candidates[0], ckpts.join("shape.bpk"));

        unsafe {
            std::env::remove_var("TRELLIS2_BPK_PRECISION");
        }
        let _ = fs::remove_dir_all(root);
    }
}

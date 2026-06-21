use std::fs::File;
use std::path::{Path, PathBuf};

use super::runtime_config::runtime_model_stage_debug_enabled;
use super::types::extraction::tensor_i32_to_vec;
use super::weight_parts::{candidate_exists_or_has_parts, load_blob_bytes_from_burnpack_or_parts};
use crate::blob_burnpack::load_blob_bytes_from_burnpack as load_blob_bytes_from_blob_burnpack;
use crate::virtual_fs;
use burn::prelude::Backend;
use burn::tensor::activation::sigmoid;
use burn::tensor::module::conv3d;
use burn::tensor::ops::ConvOptions;
use burn::tensor::{Int, Tensor};
use half::{bf16, f16};
use memmap2::{Mmap, MmapOptions};
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;

const F16_SUFFIX: &str = "_f16";
const LAYER_NORM_EPS: f64 = 1.0e-5;

type CpuRuntimeBackend = burn::backend::NdArray<f32>;
#[cfg(feature = "runtime-model-wgpu")]
type WgpuRuntimeBackend = burn_wgpu::CubeBackend<burn_wgpu::WgpuRuntime, f32, i32, u32>;

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

#[derive(Debug, Deserialize)]
struct SparseStructureDecoderConfigFile {
    args: SparseStructureDecoderConfig,
}

#[derive(Debug, Clone, Deserialize)]
struct SparseStructureDecoderConfig {
    out_channels: usize,
    latent_channels: usize,
    num_res_blocks: usize,
    channels: Vec<usize>,
    #[serde(default = "default_num_res_blocks_middle")]
    num_res_blocks_middle: usize,
    #[serde(default = "default_norm_type")]
    norm_type: String,
    #[serde(default)]
    use_fp16: bool,
}

fn default_num_res_blocks_middle() -> usize {
    2
}

fn default_norm_type() -> String {
    "layer".to_string()
}

#[derive(Debug)]
struct Conv3dLayer<B: Backend> {
    weight: Tensor<B, 5>,
    bias: Tensor<B, 1>,
    stride: [usize; 3],
    padding: [usize; 3],
    dilation: [usize; 3],
    groups: usize,
}

impl<B: Backend> Conv3dLayer<B> {
    fn forward(&self, input: Tensor<B, 5>) -> Tensor<B, 5> {
        conv3d(
            input,
            self.weight.clone(),
            Some(self.bias.clone()),
            ConvOptions::new(self.stride, self.padding, self.dilation, self.groups),
        )
    }
}

#[derive(Debug)]
struct ChannelLayerNorm<B: Backend> {
    gamma: Tensor<B, 1>,
    beta: Tensor<B, 1>,
}

impl<B: Backend> ChannelLayerNorm<B> {
    fn forward(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        let [batch, channels, depth, height, width] = x.dims();
        let x = x.permute([0, 2, 3, 4, 1]);
        let (var, mean) = x.clone().var_mean_bias(4);
        let normalized = x
            .sub(mean)
            .div(var.add_scalar(LAYER_NORM_EPS).sqrt())
            .mul(self.gamma.clone().reshape([1, 1, 1, 1, channels]))
            .add(self.beta.clone().reshape([1, 1, 1, 1, channels]));
        let _ = (batch, depth, height, width);
        normalized.permute([0, 4, 1, 2, 3])
    }
}

#[derive(Debug)]
struct SparseResBlock3d<B: Backend> {
    norm1: ChannelLayerNorm<B>,
    norm2: ChannelLayerNorm<B>,
    conv1: Conv3dLayer<B>,
    conv2: Conv3dLayer<B>,
}

impl<B: Backend> SparseResBlock3d<B> {
    fn forward(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        let residual = x.clone();
        let h = self.conv1.forward(silu_3d(self.norm1.forward(x)));
        let h = self.conv2.forward(silu_3d(self.norm2.forward(h)));
        h.add(residual)
    }
}

#[derive(Debug)]
struct SparseUpsampleBlock3d<B: Backend> {
    conv: Conv3dLayer<B>,
    out_channels: usize,
}

impl<B: Backend> SparseUpsampleBlock3d<B> {
    fn forward(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        let h = self.conv.forward(x);
        pixel_shuffle_3d(h, 2, self.out_channels)
    }
}

#[derive(Debug)]
enum SparseDecoderBlock<B: Backend> {
    Res(Box<SparseResBlock3d<B>>),
    Upsample(Box<SparseUpsampleBlock3d<B>>),
}

impl<B: Backend> SparseDecoderBlock<B> {
    fn forward(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        match self {
            Self::Res(block) => block.forward(x),
            Self::Upsample(block) => block.forward(x),
        }
    }
}

#[derive(Debug)]
struct SparseStructureDecoderModel<B: Backend> {
    input_layer: Conv3dLayer<B>,
    middle_block: Vec<SparseResBlock3d<B>>,
    blocks: Vec<SparseDecoderBlock<B>>,
    out_norm: ChannelLayerNorm<B>,
    out_conv: Conv3dLayer<B>,
}

impl<B: Backend> SparseStructureDecoderModel<B> {
    fn forward(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        let mut h = self.input_layer.forward(x);
        for block in self.middle_block.iter() {
            h = block.forward(h);
        }
        for block in self.blocks.iter() {
            h = block.forward(h);
        }
        let h = silu_3d(self.out_norm.forward(h));
        self.out_conv.forward(h)
    }
}

fn silu_3d<B: Backend>(x: Tensor<B, 5>) -> Tensor<B, 5> {
    x.clone().mul(sigmoid(x))
}

fn pixel_shuffle_3d<B: Backend>(
    x: Tensor<B, 5>,
    scale_factor: usize,
    out_channels: usize,
) -> Tensor<B, 5> {
    let [batch, channels, depth, height, width] = x.dims();
    let expected = out_channels
        .saturating_mul(scale_factor)
        .saturating_mul(scale_factor)
        .saturating_mul(scale_factor);
    if channels != expected {
        panic!(
            "pixel_shuffle_3d channel mismatch: got {}, expected {}",
            channels, expected
        );
    }
    x.reshape([
        batch,
        out_channels,
        scale_factor,
        scale_factor,
        scale_factor,
        depth,
        height,
        width,
    ])
    .permute([0, 1, 5, 2, 6, 3, 7, 4])
    .reshape([
        batch,
        out_channels,
        depth * scale_factor,
        height * scale_factor,
        width * scale_factor,
    ])
}

#[derive(Debug)]
pub(crate) struct SparseStructureDecoderRuntimeImpl<B: Backend> {
    model: SparseStructureDecoderModel<B>,
    device: B::Device,
    latent_channels: usize,
    output_channels: usize,
}

impl<B: Backend> SparseStructureDecoderRuntimeImpl<B>
where
    B::Device: Default,
{
    fn load_from_stem(
        weights_root: &Path,
        image_large_root: Option<&Path>,
        model_stem: &str,
    ) -> Result<Self, String> {
        let config_path =
            resolve_model_source_path(model_stem, "json", weights_root, image_large_root);
        let config_bytes = virtual_fs::read(&config_path).map_err(|err| {
            format!(
                "failed to read sparse structure decoder config '{}': {err}",
                config_path.display()
            )
        })?;
        let parsed: SparseStructureDecoderConfigFile =
            serde_json::from_slice(config_bytes.as_slice()).map_err(|err| {
                format!(
                    "failed to parse sparse structure decoder config '{}': {err}",
                    config_path.display()
                )
            })?;
        if parsed.args.channels.is_empty() {
            return Err(format!(
                "sparse structure decoder config '{}' has empty channels list",
                config_path.display()
            ));
        }
        if !parsed.args.norm_type.eq_ignore_ascii_case("layer") {
            return Err(format!(
                "unsupported sparse structure decoder norm_type '{}' (only 'layer' is currently supported)",
                parsed.args.norm_type
            ));
        }

        let weight_path =
            resolve_model_weight_candidates(model_stem, weights_root, image_large_root)
                .into_iter()
                .next()
                .ok_or_else(|| {
                    format!("unable to resolve sparse structure decoder weights for '{model_stem}'")
                })?;
        let weight_backing = load_weight_backing(&weight_path)?;
        let safetensors = SafeTensors::deserialize(weight_backing.as_slice()).map_err(|err| {
            format!(
                "failed to deserialize sparse structure decoder weights '{}' as safetensors: {err}",
                weight_path.display()
            )
        })?;

        let device = B::Device::default();
        let model = build_model::<B>(&parsed.args, &safetensors, &device)?;
        let _ = parsed.args.use_fp16;

        Ok(Self {
            model,
            device,
            latent_channels: parsed.args.latent_channels,
            output_channels: parsed.args.out_channels,
        })
    }

    fn decode_to_coord_tensor(
        &self,
        latent: &[f32],
        latent_resolution: usize,
        target_resolution: usize,
        max_sparse_coords: Option<usize>,
    ) -> Result<Tensor<B, 2, Int>, String> {
        if latent_resolution == 0 {
            return Err("sparse structure decoder latent resolution must be > 0".to_string());
        }
        if target_resolution == 0 {
            return Err("sparse structure decoder target resolution must be > 0".to_string());
        }
        let voxel_count = latent_resolution
            .checked_mul(latent_resolution)
            .and_then(|value| value.checked_mul(latent_resolution))
            .ok_or_else(|| {
                format!("sparse structure decoder latent voxel count overflow: {latent_resolution}")
            })?;
        let expected = self
            .latent_channels
            .checked_mul(voxel_count)
            .ok_or_else(|| "sparse structure decoder latent size overflow".to_string())?;
        if latent.len() != expected {
            return Err(format!(
                "sparse structure decoder latent length mismatch: expected {}, got {} (channels={} resolution={})",
                expected,
                latent.len(),
                self.latent_channels,
                latent_resolution
            ));
        }
        if self.output_channels != 1 {
            return Err(format!(
                "sparse structure decoder expects out_channels=1, got {}",
                self.output_channels
            ));
        }

        let latent_tensor = Tensor::<B, 1>::from_floats(latent, &self.device).reshape([
            1,
            self.latent_channels,
            latent_resolution,
            latent_resolution,
            latent_resolution,
        ]);
        self.decode_to_coord_tensor_from_latent_tensor(
            latent_tensor,
            target_resolution,
            max_sparse_coords,
        )
    }

    fn decode_to_coord_tensor_from_latent_tensor(
        &self,
        latent_tensor: Tensor<B, 5>,
        target_resolution: usize,
        max_sparse_coords: Option<usize>,
    ) -> Result<Tensor<B, 2, Int>, String> {
        if target_resolution == 0 {
            return Err("sparse structure decoder target resolution must be > 0".to_string());
        }
        let [
            latent_batch,
            latent_channels,
            latent_depth,
            latent_height,
            latent_width,
        ] = latent_tensor.dims();
        if latent_batch != 1 {
            return Err(format!(
                "sparse structure decoder latent batch mismatch: expected 1, got {latent_batch}"
            ));
        }
        if latent_channels != self.latent_channels {
            return Err(format!(
                "sparse structure decoder latent channel mismatch: expected {}, got {}",
                self.latent_channels, latent_channels
            ));
        }
        if latent_depth == 0 || latent_depth != latent_height || latent_depth != latent_width {
            return Err(format!(
                "sparse structure decoder latent tensor must be cubic with non-zero resolution, got [{latent_depth},{latent_height},{latent_width}]"
            ));
        }

        let logits = self.model.forward(latent_tensor);
        let [batch, channels, depth, height, width] = logits.dims();
        if batch != 1 || channels != 1 {
            return Err(format!(
                "sparse structure decoder output shape mismatch: expected [1,1,D,H,W], got [{batch},{channels},{depth},{height},{width}]"
            ));
        }
        if depth != height || depth != width {
            return Err(format!(
                "sparse structure decoder output must be cubic, got [{depth},{height},{width}]"
            ));
        }
        if depth < target_resolution || !depth.is_multiple_of(target_resolution) {
            return Err(format!(
                "sparse structure decoder cannot downsample occupancy from {} to {} (ratio must be integer and >= 1)",
                depth, target_resolution
            ));
        }
        let ratio = (depth / target_resolution).max(1);

        let reduced_logits = if ratio > 1 {
            logits
                .reshape([
                    target_resolution,
                    ratio,
                    target_resolution,
                    ratio,
                    target_resolution,
                    ratio,
                ])
                .max_dim(5)
                .max_dim(3)
                .max_dim(1)
                .reshape([target_resolution, target_resolution, target_resolution])
        } else {
            logits.reshape([depth, height, width])
        };
        select_positive_coord_tensor(
            reduced_logits,
            target_resolution,
            max_sparse_coords,
            sparse_structure_stage_debug_enabled(),
        )
    }
}

fn select_positive_coord_tensor<B: Backend>(
    reduced_logits: Tensor<B, 3>,
    target_resolution: usize,
    max_sparse_coords: Option<usize>,
    stage_debug: bool,
) -> Result<Tensor<B, 2, Int>, String> {
    let [reduced_depth, reduced_height, reduced_width] = reduced_logits.dims();
    if reduced_depth != target_resolution
        || reduced_height != target_resolution
        || reduced_width != target_resolution
    {
        return Err(format!(
            "sparse structure decoder reduced logits shape mismatch: expected [{0},{0},{0}], got [{1},{2},{3}]",
            target_resolution, reduced_depth, reduced_height, reduced_width
        ));
    }

    let device = reduced_logits.device();
    let mut positive = reduced_logits.greater_elem(0.0).argwhere();
    let [positive_count, positive_cols] = positive.dims();
    if positive_cols != 3 {
        return Err(format!(
            "sparse structure decoder argwhere output mismatch: expected [N,3], got [N,{}]",
            positive_cols
        ));
    }
    if stage_debug {
        let reduced_elements = reduced_depth
            .saturating_mul(reduced_height)
            .saturating_mul(reduced_width);
        eprintln!(
            "burn_trellis: sparse structure logits stats backend={} depth={} target={} total={} positive={}",
            std::any::type_name::<B>(),
            reduced_depth,
            target_resolution,
            reduced_elements,
            positive_count
        );
    }

    if positive_count > 1 {
        let resolution_i64 = i64::try_from(target_resolution).map_err(|_| {
            format!(
                "sparse structure decoder target resolution {} exceeds i64 range",
                target_resolution
            )
        })?;
        let max_key = resolution_i64
            .checked_mul(resolution_i64)
            .and_then(|value| value.checked_mul(resolution_i64))
            .and_then(|value| value.checked_sub(1))
            .ok_or_else(|| {
                format!(
                    "sparse structure decoder key-space overflow for resolution {}",
                    target_resolution
                )
            })?;
        let resolution_i32 = i32::try_from(target_resolution).map_err(|_| {
            format!(
                "sparse structure decoder target resolution {} exceeds i32 range",
                target_resolution
            )
        })?;
        let resolution_sq_i32 = resolution_i32.checked_mul(resolution_i32).ok_or_else(|| {
            format!(
                "sparse structure decoder key stride overflow for resolution {}",
                target_resolution
            )
        })?;
        if max_key > i32::MAX as i64 {
            return Err(format!(
                "sparse structure decoder key-space {} exceeds i32 range for resolution {}",
                max_key, target_resolution
            ));
        }
        let z_col = positive
            .clone()
            .slice([0..positive_count, 0..1])
            .squeeze_dim(1);
        let y_col = positive
            .clone()
            .slice([0..positive_count, 1..2])
            .squeeze_dim(1);
        let x_col = positive
            .clone()
            .slice([0..positive_count, 2..3])
            .squeeze_dim(1);
        let key_t = z_col
            .mul_scalar(resolution_sq_i32)
            .add(y_col.mul_scalar(resolution_i32))
            .add(x_col);
        let (_sorted_keys, sorted_idx) = key_t.sort_with_indices(0);
        positive = positive.select(0, sorted_idx);
    }

    let mut selected_count = positive_count;
    if let Some(limit) = max_sparse_coords.filter(|limit| *limit > 0)
        && positive_count > limit
    {
        selected_count = limit;
        positive = positive.slice([0..selected_count, 0..3]);
    }

    // Burn cubecl int cat currently panics on zero-row cat due an internal
    // overlap check underflow; return an explicit empty coord tensor instead.
    if selected_count == 0 {
        return Ok(Tensor::<B, 2, Int>::zeros([0, 4], &device));
    }

    let x_col = positive
        .clone()
        .slice([0..selected_count, 2..3])
        .reshape([selected_count, 1]);
    let y_col = positive
        .clone()
        .slice([0..selected_count, 1..2])
        .reshape([selected_count, 1]);
    let z_col = positive
        .slice([0..selected_count, 0..1])
        .reshape([selected_count, 1]);
    let batch_col = Tensor::<B, 2, Int>::zeros([selected_count, 1], &device);
    Ok(Tensor::cat(vec![batch_col, z_col, y_col, x_col], 1))
}

fn tensor_to_coords_u32<B: Backend>(
    tensor: Tensor<B, 2, Int>,
    context: &str,
) -> Result<Vec<[u32; 4]>, String> {
    let [rows, cols] = tensor.dims();
    if cols != 4 {
        return Err(format!(
            "{context}: coord tensor must have 4 columns, got {cols}"
        ));
    }
    let values = tensor_i32_to_vec(tensor, context)?;
    if values.len() != rows.saturating_mul(4) {
        return Err(format!(
            "{context}: coord tensor length mismatch: got={} expected={}",
            values.len(),
            rows.saturating_mul(4)
        ));
    }
    let mut out = Vec::with_capacity(rows);
    for row_idx in 0..rows {
        let base = row_idx.saturating_mul(4);
        let to_u32 = |value: i32| -> Result<u32, String> {
            u32::try_from(value).map_err(|_| {
                format!("{context}: negative coordinate value {value} at row {row_idx}")
            })
        };
        out.push([
            to_u32(values[base])?,
            to_u32(values[base + 1])?,
            to_u32(values[base + 2])?,
            to_u32(values[base + 3])?,
        ]);
    }
    Ok(out)
}

fn sparse_structure_stage_debug_enabled() -> bool {
    runtime_model_stage_debug_enabled()
}

#[derive(Debug, Clone)]
pub(crate) struct SparseStructureCoords {
    coords: Option<Vec<[u32; 4]>>,
    #[cfg(feature = "runtime-model-wgpu")]
    coords_tensor: Option<Tensor<WgpuRuntimeBackend, 2, Int>>,
}

impl SparseStructureCoords {
    fn from_host(coords: Vec<[u32; 4]>) -> Self {
        Self {
            coords: Some(coords),
            #[cfg(feature = "runtime-model-wgpu")]
            coords_tensor: None,
        }
    }

    #[cfg(feature = "runtime-model-wgpu")]
    fn from_wgpu_tensor(coords_tensor: Tensor<WgpuRuntimeBackend, 2, Int>) -> Self {
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
            "{context}: sparse structure coords have no host values and no device tensor"
        ))
    }

    #[cfg(feature = "runtime-model-wgpu")]
    pub fn coords_tensor(&self) -> Option<Tensor<WgpuRuntimeBackend, 2, Int>> {
        self.coords_tensor.as_ref().cloned()
    }
}

#[derive(Debug)]
pub(crate) enum SparseStructureDecoderRuntime {
    Cpu(Box<SparseStructureDecoderRuntimeImpl<CpuRuntimeBackend>>),
    #[cfg(feature = "runtime-model-wgpu")]
    Wgpu(Box<SparseStructureDecoderRuntimeImpl<WgpuRuntimeBackend>>),
}

impl SparseStructureDecoderRuntime {
    pub fn load_from_stem(
        weights_root: &Path,
        image_large_root: Option<&Path>,
        model_stem: &str,
        prefer_wgpu: bool,
    ) -> Result<Self, String> {
        #[cfg(feature = "runtime-model-wgpu")]
        if prefer_wgpu {
            let runtime = SparseStructureDecoderRuntimeImpl::<WgpuRuntimeBackend>::load_from_stem(
                weights_root,
                image_large_root,
                model_stem,
            )?;
            return Ok(Self::Wgpu(Box::new(runtime)));
        }

        #[cfg(not(feature = "runtime-model-wgpu"))]
        if prefer_wgpu {
            return Err(
                "sparse structure decoder requested wgpu runtime, but runtime-model-wgpu feature is disabled"
                    .to_string(),
            );
        }

        let runtime = SparseStructureDecoderRuntimeImpl::<CpuRuntimeBackend>::load_from_stem(
            weights_root,
            image_large_root,
            model_stem,
        )?;
        Ok(Self::Cpu(Box::new(runtime)))
    }

    pub fn backend_name(&self) -> &'static str {
        match self {
            Self::Cpu(_) => "cpu",
            #[cfg(feature = "runtime-model-wgpu")]
            Self::Wgpu(_) => "wgpu",
        }
    }

    pub fn decode_to_sparse_coords(
        &self,
        latent: &[f32],
        latent_resolution: usize,
        target_resolution: usize,
        max_sparse_coords: Option<usize>,
    ) -> Result<SparseStructureCoords, String> {
        match self {
            Self::Cpu(runtime) => runtime
                .decode_to_coord_tensor(
                    latent,
                    latent_resolution,
                    target_resolution,
                    max_sparse_coords,
                )
                .and_then(|coords_t| {
                    tensor_to_coords_u32(
                        coords_t,
                        "failed to read sparse structure decoder positive coords",
                    )
                })
                .map(SparseStructureCoords::from_host),
            #[cfg(feature = "runtime-model-wgpu")]
            Self::Wgpu(runtime) => runtime
                .decode_to_coord_tensor(
                    latent,
                    latent_resolution,
                    target_resolution,
                    max_sparse_coords,
                )
                .map(SparseStructureCoords::from_wgpu_tensor),
        }
    }

    #[cfg(feature = "runtime-model-wgpu")]
    pub fn decode_to_sparse_coords_wgpu_latent_tensor(
        &self,
        latent_tensor: Tensor<WgpuRuntimeBackend, 5>,
        target_resolution: usize,
        max_sparse_coords: Option<usize>,
    ) -> Result<SparseStructureCoords, String> {
        match self {
            Self::Cpu(_) => Err(
                "sparse structure decoder tensor-native latent path requires wgpu runtime backend"
                    .to_string(),
            ),
            Self::Wgpu(runtime) => runtime
                .decode_to_coord_tensor_from_latent_tensor(
                    latent_tensor,
                    target_resolution,
                    max_sparse_coords,
                )
                .map(SparseStructureCoords::from_wgpu_tensor),
        }
    }
}

fn build_model<B: Backend>(
    config: &SparseStructureDecoderConfig,
    safetensors: &SafeTensors<'_>,
    device: &B::Device,
) -> Result<SparseStructureDecoderModel<B>, String> {
    let input_layer = load_conv3d(
        safetensors,
        "input_layer.weight",
        "input_layer.bias",
        config.latent_channels,
        config.channels[0],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        1,
        device,
    )?;

    let mut middle_block = Vec::with_capacity(config.num_res_blocks_middle);
    for idx in 0..config.num_res_blocks_middle {
        let prefix = format!("middle_block.{idx}");
        middle_block.push(load_res_block(
            safetensors,
            prefix.as_str(),
            config.channels[0],
            device,
        )?);
    }

    let mut blocks = Vec::new();
    let mut block_idx = 0usize;
    for (stage_idx, channels) in config.channels.iter().copied().enumerate() {
        for _ in 0..config.num_res_blocks {
            let prefix = format!("blocks.{block_idx}");
            let res = load_res_block(safetensors, prefix.as_str(), channels, device)?;
            blocks.push(SparseDecoderBlock::Res(Box::new(res)));
            block_idx += 1;
        }
        if stage_idx + 1 < config.channels.len() {
            let next_channels = config.channels[stage_idx + 1];
            let prefix = format!("blocks.{block_idx}");
            let upsample = load_upsample_block(
                safetensors,
                prefix.as_str(),
                channels,
                next_channels,
                device,
            )?;
            blocks.push(SparseDecoderBlock::Upsample(Box::new(upsample)));
            block_idx += 1;
        }
    }

    let out_norm = load_channel_layer_norm(
        safetensors,
        "out_layer.0",
        config.channels.last().copied().unwrap_or(0),
        device,
    )?;
    let out_conv = load_conv3d(
        safetensors,
        "out_layer.2.weight",
        "out_layer.2.bias",
        config.channels.last().copied().unwrap_or(0),
        config.out_channels,
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        1,
        device,
    )?;

    Ok(SparseStructureDecoderModel {
        input_layer,
        middle_block,
        blocks,
        out_norm,
        out_conv,
    })
}

fn load_res_block<B: Backend>(
    safetensors: &SafeTensors<'_>,
    prefix: &str,
    channels: usize,
    device: &B::Device,
) -> Result<SparseResBlock3d<B>, String> {
    let norm1 = load_channel_layer_norm(
        safetensors,
        format!("{prefix}.norm1").as_str(),
        channels,
        device,
    )?;
    let norm2 = load_channel_layer_norm(
        safetensors,
        format!("{prefix}.norm2").as_str(),
        channels,
        device,
    )?;
    let conv1 = load_conv3d(
        safetensors,
        format!("{prefix}.conv1.weight").as_str(),
        format!("{prefix}.conv1.bias").as_str(),
        channels,
        channels,
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        1,
        device,
    )?;
    let conv2 = load_conv3d(
        safetensors,
        format!("{prefix}.conv2.weight").as_str(),
        format!("{prefix}.conv2.bias").as_str(),
        channels,
        channels,
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        1,
        device,
    )?;
    Ok(SparseResBlock3d {
        norm1,
        norm2,
        conv1,
        conv2,
    })
}

fn load_upsample_block<B: Backend>(
    safetensors: &SafeTensors<'_>,
    prefix: &str,
    in_channels: usize,
    out_channels: usize,
    device: &B::Device,
) -> Result<SparseUpsampleBlock3d<B>, String> {
    let conv = load_conv3d(
        safetensors,
        format!("{prefix}.conv.weight").as_str(),
        format!("{prefix}.conv.bias").as_str(),
        in_channels,
        out_channels
            .checked_mul(8)
            .ok_or_else(|| "upsample out_channels overflow".to_string())?,
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        1,
        device,
    )?;
    Ok(SparseUpsampleBlock3d { conv, out_channels })
}

fn load_channel_layer_norm<B: Backend>(
    safetensors: &SafeTensors<'_>,
    prefix: &str,
    channels: usize,
    device: &B::Device,
) -> Result<ChannelLayerNorm<B>, String> {
    let (weight_shape, weight) = load_tensor_f32(safetensors, format!("{prefix}.weight").as_str())?;
    let (bias_shape, bias) = load_tensor_f32(safetensors, format!("{prefix}.bias").as_str())?;
    if weight_shape.len() != 1 || weight_shape[0] != channels {
        return Err(format!(
            "tensor '{}.weight' expected shape [{}], got {:?}",
            prefix, channels, weight_shape
        ));
    }
    if bias_shape.len() != 1 || bias_shape[0] != channels {
        return Err(format!(
            "tensor '{}.bias' expected shape [{}], got {:?}",
            prefix, channels, bias_shape
        ));
    }
    Ok(ChannelLayerNorm {
        gamma: Tensor::<B, 1>::from_floats(weight.as_slice(), device),
        beta: Tensor::<B, 1>::from_floats(bias.as_slice(), device),
    })
}

#[allow(clippy::too_many_arguments)]
fn load_conv3d<B: Backend>(
    safetensors: &SafeTensors<'_>,
    weight_key: &str,
    bias_key: &str,
    expected_in_channels: usize,
    expected_out_channels: usize,
    stride: [usize; 3],
    padding: [usize; 3],
    dilation: [usize; 3],
    groups: usize,
    device: &B::Device,
) -> Result<Conv3dLayer<B>, String> {
    let (weight_shape, weight) = load_tensor_f32(safetensors, weight_key)?;
    if weight_shape.len() != 5 {
        return Err(format!(
            "tensor '{weight_key}' expected rank=5, got rank={}",
            weight_shape.len()
        ));
    }
    if weight_shape[0] != expected_out_channels {
        return Err(format!(
            "tensor '{weight_key}' expected out_channels={}, got {}",
            expected_out_channels, weight_shape[0]
        ));
    }
    if weight_shape[1] != expected_in_channels / groups.max(1) {
        return Err(format!(
            "tensor '{weight_key}' expected in_channels/group={}, got {}",
            expected_in_channels / groups.max(1),
            weight_shape[1]
        ));
    }
    let (bias_shape, bias) = load_tensor_f32(safetensors, bias_key)?;
    if bias_shape.len() != 1 || bias_shape[0] != expected_out_channels {
        return Err(format!(
            "tensor '{bias_key}' expected shape [{}], got {:?}",
            expected_out_channels, bias_shape
        ));
    }
    let weight = Tensor::<B, 1>::from_floats(weight.as_slice(), device).reshape([
        weight_shape[0],
        weight_shape[1],
        weight_shape[2],
        weight_shape[3],
        weight_shape[4],
    ]);
    let bias = Tensor::<B, 1>::from_floats(bias.as_slice(), device);
    Ok(Conv3dLayer {
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
    })
}

fn load_tensor_f32(
    safetensors: &SafeTensors<'_>,
    key: &str,
) -> Result<(Vec<usize>, Vec<f32>), String> {
    let view = safetensors
        .tensor(key)
        .map_err(|err| format!("missing tensor '{key}' in safetensors: {err}"))?;
    let shape = view.shape().to_vec();
    let values = match view.dtype() {
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
    if values.len() != expected {
        return Err(format!(
            "tensor '{key}' element count mismatch: expected {expected}, got {}",
            values.len()
        ));
    }
    if sparse_structure_stage_debug_enabled()
        && matches!(
            key,
            "input_layer.weight" | "input_layer.bias" | "out_layer.2.weight" | "out_layer.2.bias"
        )
    {
        let mut min_v = f32::INFINITY;
        let mut max_v = f32::NEG_INFINITY;
        let mut sum_v = 0.0f64;
        for value in values.iter().copied() {
            min_v = min_v.min(value);
            max_v = max_v.max(value);
            sum_v += value as f64;
        }
        let mean_v = if values.is_empty() {
            0.0
        } else {
            (sum_v / values.len() as f64) as f32
        };
        eprintln!(
            "burn_trellis: sparse structure weight stats key={} shape={:?} min={:.6} max={:.6} mean={:.6}",
            key, shape, min_v, max_v, mean_v
        );
    }
    Ok((shape, values))
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

fn load_weight_backing(path: &Path) -> Result<WeightsBacking, String> {
    if path
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("bpk"))
    {
        let bytes = load_blob_bytes_from_burnpack_or_parts(path, load_burnpack_blob_bytes)?;
        return Ok(WeightsBacking::Bytes(bytes));
    }

    if virtual_fs::has_virtual_file(path) {
        let bytes = virtual_fs::read(path).map_err(|err| {
            format!(
                "failed to read virtual sparse structure decoder weights '{}': {err}",
                path.display()
            )
        })?;
        return Ok(WeightsBacking::Bytes(bytes));
    }

    let file = File::open(path).map_err(|err| {
        format!(
            "failed to open sparse structure decoder weights '{}': {err}",
            path.display()
        )
    })?;
    let mmap = unsafe { MmapOptions::new().map(&file) }.map_err(|err| {
        format!(
            "failed to mmap sparse structure decoder weights '{}': {err}",
            path.display()
        )
    })?;
    Ok(WeightsBacking::Mmap(mmap))
}

fn load_burnpack_blob_bytes(path: &Path) -> Result<Vec<u8>, String> {
    load_blob_bytes_from_blob_burnpack(path)
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
    let candidates = vec![burnpack, burnpack_f16, source, source_f16];
    candidates
        .into_iter()
        .filter(|path| candidate_exists_or_has_parts(path))
        .collect::<Vec<_>>()
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
        let relative = format!("ckpts/{suffix}.{ext}");
        if let Some(image_large_root) = image_large_root {
            let image_candidate = image_large_root.join(&relative);
            if image_candidate.exists() {
                return image_candidate;
            }
            let weights_candidate = weights_root.join(&relative);
            if weights_candidate.exists() {
                return weights_candidate;
            }
            return image_candidate;
        }
        return weights_root.join(relative);
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
    use super::*;
    use crate::hook_diff::HookSnapshot;
    use crate::paths::{resolve_trellis2_image_large_root, resolve_trellis2_weights_root};
    #[cfg(feature = "runtime-model-wgpu")]
    use crate::runtime_model::types::extraction::tensor_f32_to_vec;
    use std::collections::BTreeSet;

    #[test]
    fn sparse_structure_coord_select_cap_boundary_parity() {
        let device = <CpuRuntimeBackend as burn::tensor::backend::BackendTypes>::Device::default();
        let logits = Tensor::<CpuRuntimeBackend, 1>::from_floats(
            [-1.0, 0.9, 0.8, -0.2, 0.7, -0.3, 0.6, 0.5],
            &device,
        )
        .reshape([2, 2, 2]);

        let coords_eq = select_positive_coord_tensor(logits.clone(), 2, Some(5), false)
            .expect("coord select with cap==count should succeed");
        let coords_eq_host = tensor_to_coords_u32(coords_eq, "coords_eq")
            .expect("coord tensor should materialize for test assertion");
        assert_eq!(
            coords_eq_host,
            vec![
                [0, 0, 0, 1],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 1, 1, 0],
                [0, 1, 1, 1],
            ]
        );

        let coords_cap = select_positive_coord_tensor(logits, 2, Some(4), false)
            .expect("coord select with cap<count should succeed");
        let coords_cap_host = tensor_to_coords_u32(coords_cap, "coords_cap")
            .expect("coord tensor should materialize for test assertion");
        assert_eq!(
            coords_cap_host,
            vec![[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 1, 0]]
        );
    }

    #[test]
    fn sparse_structure_coord_select_token_cap_boundary_parity() {
        // Canonical alias test so roadmap gate names map to a concrete test target.
        sparse_structure_coord_select_cap_boundary_parity();
    }

    #[test]
    fn sparse_structure_coord_select_empty_mask_returns_empty_coords() {
        let device = <CpuRuntimeBackend as burn::tensor::backend::BackendTypes>::Device::default();
        let logits = Tensor::<CpuRuntimeBackend, 1>::from_floats(
            [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3],
            &device,
        )
        .reshape([2, 2, 2]);
        let coords = select_positive_coord_tensor(logits, 2, Some(8), false)
            .expect("coord select should succeed for all-negative mask");
        let [rows, cols] = coords.dims();
        assert_eq!(rows, 0);
        assert_eq!(cols, 4);
    }

    #[test]
    fn image_large_stem_weight_candidates_fall_back_to_weights_root_source() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock drift")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "burn_trellis_sparse_structure_decoder_candidates_{unique}"
        ));
        let weights_ckpts = root.join("weights/ckpts");
        let image_ckpts = root.join("image/ckpts");
        std::fs::create_dir_all(&weights_ckpts).expect("create weights ckpts");
        std::fs::create_dir_all(&image_ckpts).expect("create image ckpts");
        std::fs::write(weights_ckpts.join("ss_dec.safetensors"), b"safe")
            .expect("write weights safetensors");
        std::fs::write(image_ckpts.join("ss_dec.bpk"), b"stale image burnpack")
            .expect("write stale image burnpack");

        let candidates = resolve_model_weight_candidates(
            "microsoft/TRELLIS-image-large/ckpts/ss_dec",
            root.join("weights").as_path(),
            Some(root.join("image").as_path()),
        );

        assert_eq!(candidates, vec![weights_ckpts.join("ss_dec.safetensors")]);

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn sparse_structure_decoder_matches_reference_latent_coord_count_cpu() {
        if std::env::var("TRELLIS2_DECODER_REFERENCE_TEST").is_err() {
            eprintln!(
                "skipping: set TRELLIS2_DECODER_REFERENCE_TEST=1 to run sparse structure decoder reference latent parity test"
            );
            return;
        }

        let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let hook_path =
            manifest_dir.join("assets/hooks/trellis2_full_reference_alpha_512.safetensors");
        if !hook_path.exists() {
            eprintln!(
                "skipping: missing sparse structure reference hook {}",
                hook_path.display()
            );
            return;
        }

        let snapshot = HookSnapshot::from_file(&hook_path)
            .unwrap_or_else(|err| panic!("failed to load sparse reference hook: {err}"));
        let latent = snapshot
            .tensors
            .get("sample_sparse_structure.latent")
            .expect("missing sample_sparse_structure.latent");
        assert_eq!(
            latent.shape.as_slice(),
            &[1, 8, 16, 16, 16],
            "unexpected sparse latent shape in reference hook"
        );
        let reference_coords = snapshot
            .tensors
            .get("sample_sparse_structure.coords")
            .expect("missing sample_sparse_structure.coords");
        assert_eq!(
            reference_coords.shape.len(),
            2,
            "reference sparse coords tensor must be rank-2"
        );
        assert_eq!(
            reference_coords.shape[1], 4,
            "reference sparse coords tensor must be [N,4]"
        );
        let reference_rows = reference_coords.shape[0];
        assert!(
            reference_rows > 0,
            "reference sparse coords must contain positive rows"
        );

        let weights_root = resolve_trellis2_weights_root(None);
        let image_large_root = resolve_trellis2_image_large_root(None);
        if !weights_root.exists() || !image_large_root.exists() {
            eprintln!(
                "skipping: missing Trellis roots (weights={} image_large={})",
                weights_root.display(),
                image_large_root.display()
            );
            return;
        }

        let runtime = SparseStructureDecoderRuntimeImpl::<CpuRuntimeBackend>::load_from_stem(
            weights_root.as_path(),
            Some(image_large_root.as_path()),
            "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16",
        )
        .unwrap_or_else(|err| panic!("failed to load sparse structure decoder runtime: {err}"));
        let coords_t = runtime
            .decode_to_coord_tensor(latent.data.as_slice(), 16, 32, None)
            .unwrap_or_else(|err| panic!("failed to decode sparse reference latent: {err}"));
        let [rows, cols] = coords_t.dims();
        assert_eq!(cols, 4, "decoded sparse coords tensor must be [N,4]");
        assert_eq!(
            rows, reference_rows,
            "decoded sparse coord row count mismatch vs reference hook"
        );
    }

    #[test]
    fn sparse_structure_decoder_probe_current_reference_cpu_wgpu() {
        if std::env::var("TRELLIS2_SPARSE_DECODER_PROBE").is_err() {
            eprintln!(
                "skipping: set TRELLIS2_SPARSE_DECODER_PROBE=1 and TRELLIS2_SPARSE_DECODER_PROBE_HOOK=<hook.safetensors> to compare sparse decoder coords"
            );
            return;
        }

        let hook_path = std::env::var("TRELLIS2_SPARSE_DECODER_PROBE_HOOK")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("assets/hooks/trellis2_full_reference_alpha_512.safetensors")
            });
        if !hook_path.exists() {
            panic!(
                "TRELLIS2_SPARSE_DECODER_PROBE_HOOK does not exist: {}",
                hook_path.display()
            );
        }

        let snapshot = HookSnapshot::from_file(&hook_path)
            .unwrap_or_else(|err| panic!("failed to load sparse decoder probe hook: {err}"));
        let latent = snapshot
            .tensors
            .get("sample_sparse_structure.latent")
            .expect("probe hook missing sample_sparse_structure.latent");
        assert_eq!(
            latent.shape.len(),
            5,
            "sample_sparse_structure.latent must be rank-5 [B,C,D,H,W]"
        );
        assert_eq!(
            latent.shape[0], 1,
            "sparse decoder probe expects batch=1 latent"
        );
        assert_eq!(
            latent.shape[2], latent.shape[3],
            "sparse decoder probe expects cubic latent"
        );
        assert_eq!(
            latent.shape[2], latent.shape[4],
            "sparse decoder probe expects cubic latent"
        );
        let latent_resolution = latent.shape[2];
        let target_resolution = sparse_decoder_probe_target_resolution(&snapshot);
        let reference_coords =
            sparse_decoder_probe_reference_coords(&snapshot, "sample_sparse_structure.coords");

        let weights_root =
            sparse_decoder_probe_root("TRELLIS2_WEIGHTS_ROOT", resolve_trellis2_weights_root(None));
        let image_large_root = sparse_decoder_probe_root(
            "TRELLIS2_IMAGE_LARGE_ROOT",
            resolve_trellis2_image_large_root(None),
        );
        if !weights_root.exists() || !image_large_root.exists() {
            panic!(
                "missing Trellis roots for sparse decoder probe (weights={} image_large={})",
                weights_root.display(),
                image_large_root.display()
            );
        }
        let model_stem = std::env::var("TRELLIS2_SPARSE_DECODER_MODEL_STEM").unwrap_or_else(|_| {
            "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16".to_string()
        });

        #[cfg(feature = "runtime-model-wgpu")]
        let wgpu_coords_for_cpu_compare = {
            let wgpu_runtime =
                SparseStructureDecoderRuntimeImpl::<WgpuRuntimeBackend>::load_from_stem(
                    weights_root.as_path(),
                    Some(image_large_root.as_path()),
                    model_stem.as_str(),
                )
                .unwrap_or_else(|err| panic!("failed to load WGPU sparse decoder runtime: {err}"));
            let wgpu_coords_t = wgpu_runtime
                .decode_to_coord_tensor(
                    latent.data.as_slice(),
                    latent_resolution,
                    target_resolution,
                    None,
                )
                .unwrap_or_else(|err| panic!("WGPU sparse decoder failed: {err}"));
            let wgpu_coords = tensor_to_coords_u32(wgpu_coords_t, "probe WGPU sparse coords")
                .unwrap_or_else(|err| panic!("failed to read WGPU sparse decoder coords: {err}"));
            eprintln!(
                "sparse decoder probe: wgpu rows={} latent_resolution={} target_resolution={} hook={}",
                wgpu_coords.len(),
                latent_resolution,
                target_resolution,
                hook_path.display()
            );
            if let Some(reference_coords) = reference_coords.as_ref() {
                let report = sparse_decoder_probe_coord_report(reference_coords, &wgpu_coords);
                let reduced_logits = sparse_decoder_probe_reduced_logits(
                    &wgpu_runtime,
                    latent.data.as_slice(),
                    latent_resolution,
                    target_resolution,
                    "probe WGPU reduced sparse logits",
                )
                .unwrap_or_else(|err| panic!("failed to read WGPU reduced sparse logits: {err}"));
                eprintln!(
                    "sparse decoder probe: reference vs wgpu overlap={} missing={} extra={} first_missing={:?} first_extra={:?} missing_logits={:?} extra_logits={:?}",
                    report.overlap,
                    report.missing.len(),
                    report.extra.len(),
                    report.missing.first(),
                    report.extra.first(),
                    sparse_decoder_probe_logits_for_coords(
                        &reduced_logits,
                        target_resolution,
                        &report.missing,
                        8,
                    ),
                    sparse_decoder_probe_logits_for_coords(
                        &reduced_logits,
                        target_resolution,
                        &report.extra,
                        8,
                    )
                );
                assert!(
                    report.missing.is_empty() && report.extra.is_empty(),
                    "WGPU sparse decoder coords differ from reference: reference={} actual={} overlap={} missing={} extra={} first_missing={:?} first_extra={:?}",
                    report.reference_count,
                    report.actual_count,
                    report.overlap,
                    report.missing.len(),
                    report.extra.len(),
                    report.missing.first(),
                    report.extra.first()
                );
            }
            Some(wgpu_coords)
        };

        let run_cpu = std::env::var("TRELLIS2_SPARSE_DECODER_PROBE_CPU").is_ok()
            || cfg!(not(feature = "runtime-model-wgpu"));
        if run_cpu {
            let cpu_runtime =
                SparseStructureDecoderRuntimeImpl::<CpuRuntimeBackend>::load_from_stem(
                    weights_root.as_path(),
                    Some(image_large_root.as_path()),
                    model_stem.as_str(),
                )
                .unwrap_or_else(|err| panic!("failed to load CPU sparse decoder runtime: {err}"));
            let cpu_coords_t = cpu_runtime
                .decode_to_coord_tensor(
                    latent.data.as_slice(),
                    latent_resolution,
                    target_resolution,
                    None,
                )
                .unwrap_or_else(|err| panic!("CPU sparse decoder failed: {err}"));
            let cpu_coords = tensor_to_coords_u32(cpu_coords_t, "probe CPU sparse coords")
                .unwrap_or_else(|err| panic!("failed to read CPU sparse decoder coords: {err}"));
            eprintln!(
                "sparse decoder probe: cpu rows={} latent_resolution={} target_resolution={} hook={}",
                cpu_coords.len(),
                latent_resolution,
                target_resolution,
                hook_path.display()
            );
            if let Some(reference_coords) = reference_coords.as_ref() {
                let report = sparse_decoder_probe_coord_report(reference_coords, &cpu_coords);
                eprintln!(
                    "sparse decoder probe: reference vs cpu overlap={} missing={} extra={} first_missing={:?} first_extra={:?}",
                    report.overlap,
                    report.missing.len(),
                    report.extra.len(),
                    report.missing.first(),
                    report.extra.first()
                );
                assert!(
                    report.missing.is_empty() && report.extra.is_empty(),
                    "CPU sparse decoder coords differ from reference: reference={} actual={} overlap={} missing={} extra={} first_missing={:?} first_extra={:?}",
                    report.reference_count,
                    report.actual_count,
                    report.overlap,
                    report.missing.len(),
                    report.extra.len(),
                    report.missing.first(),
                    report.extra.first()
                );
            }
            #[cfg(feature = "runtime-model-wgpu")]
            if let Some(wgpu_coords) = wgpu_coords_for_cpu_compare.as_ref() {
                let cpu_wgpu = sparse_decoder_probe_coord_report(&cpu_coords, wgpu_coords);
                eprintln!(
                    "sparse decoder probe: cpu vs wgpu rows_cpu={} rows_wgpu={} overlap={} missing_in_wgpu={} extra_in_wgpu={} first_missing={:?} first_extra={:?}",
                    cpu_coords.len(),
                    wgpu_coords.len(),
                    cpu_wgpu.overlap,
                    cpu_wgpu.missing.len(),
                    cpu_wgpu.extra.len(),
                    cpu_wgpu.missing.first(),
                    cpu_wgpu.extra.first()
                );
                assert!(
                    cpu_wgpu.missing.is_empty() && cpu_wgpu.extra.is_empty(),
                    "WGPU sparse decoder coords differ from CPU: cpu={} wgpu={} overlap={} missing_in_wgpu={} extra_in_wgpu={} first_missing={:?} first_extra={:?}",
                    cpu_wgpu.reference_count,
                    cpu_wgpu.actual_count,
                    cpu_wgpu.overlap,
                    cpu_wgpu.missing.len(),
                    cpu_wgpu.extra.len(),
                    cpu_wgpu.missing.first(),
                    cpu_wgpu.extra.first()
                );
            }
        }
    }

    #[cfg(feature = "runtime-model-wgpu")]
    fn sparse_decoder_probe_reduced_logits<B: Backend>(
        runtime: &SparseStructureDecoderRuntimeImpl<B>,
        latent: &[f32],
        latent_resolution: usize,
        target_resolution: usize,
        context: &str,
    ) -> Result<Vec<f32>, String> {
        let latent_tensor = Tensor::<B, 1>::from_floats(latent, &runtime.device).reshape([
            1,
            runtime.latent_channels,
            latent_resolution,
            latent_resolution,
            latent_resolution,
        ]);
        let logits = runtime.model.forward(latent_tensor);
        let [batch, channels, depth, height, width] = logits.dims();
        if batch != 1 || channels != 1 || depth != height || depth != width {
            return Err(format!(
                "{context}: unexpected decoder output shape [{batch},{channels},{depth},{height},{width}]"
            ));
        }
        if depth < target_resolution || !depth.is_multiple_of(target_resolution) {
            return Err(format!(
                "{context}: cannot reduce decoder output resolution {depth} to target {target_resolution}"
            ));
        }
        let ratio = (depth / target_resolution).max(1);
        let reduced = if ratio > 1 {
            logits
                .reshape([
                    target_resolution,
                    ratio,
                    target_resolution,
                    ratio,
                    target_resolution,
                    ratio,
                ])
                .max_dim(5)
                .max_dim(3)
                .max_dim(1)
                .reshape([target_resolution, target_resolution, target_resolution])
        } else {
            logits.reshape([depth, height, width])
        };
        tensor_f32_to_vec(
            reduced.reshape([
                target_resolution
                    .saturating_mul(target_resolution)
                    .saturating_mul(target_resolution),
                1,
            ]),
            context,
        )
    }

    #[cfg(feature = "runtime-model-wgpu")]
    fn sparse_decoder_probe_logits_for_coords(
        reduced_logits: &[f32],
        resolution: usize,
        coords: &[[u32; 4]],
        limit: usize,
    ) -> Vec<([u32; 4], f32)> {
        coords
            .iter()
            .take(limit)
            .filter_map(|coord| {
                let z = usize::try_from(coord[1]).ok()?;
                let y = usize::try_from(coord[2]).ok()?;
                let x = usize::try_from(coord[3]).ok()?;
                if z >= resolution || y >= resolution || x >= resolution {
                    return None;
                }
                let idx = z
                    .saturating_mul(resolution)
                    .saturating_mul(resolution)
                    .saturating_add(y.saturating_mul(resolution))
                    .saturating_add(x);
                reduced_logits
                    .get(idx)
                    .copied()
                    .map(|value| (*coord, value))
            })
            .collect()
    }

    fn sparse_decoder_probe_root(env_key: &str, fallback: PathBuf) -> PathBuf {
        std::env::var_os(env_key)
            .map(PathBuf::from)
            .unwrap_or(fallback)
    }

    fn sparse_decoder_probe_target_resolution(snapshot: &HookSnapshot) -> usize {
        if let Ok(value) = std::env::var("TRELLIS2_SPARSE_DECODER_TARGET_RESOLUTION")
            && let Ok(parsed) = value.parse::<usize>()
            && parsed > 0
        {
            return parsed;
        }
        if let Some(tensor) = snapshot.tensors.get("run.sparse_structure_resolution")
            && let Some(value) = tensor.data.first()
        {
            let rounded = value.round();
            if rounded > 0.0 {
                return rounded as usize;
            }
        }
        if let Some(value) = snapshot.metadata.get("sparse_resolution")
            && let Ok(parsed) = value.parse::<usize>()
            && parsed > 0
        {
            return parsed;
        }
        32
    }

    fn sparse_decoder_probe_reference_coords(
        snapshot: &HookSnapshot,
        key: &str,
    ) -> Option<Vec<[u32; 4]>> {
        let tensor = snapshot.tensors.get(key)?;
        assert_eq!(
            tensor.shape.len(),
            2,
            "{key} must be rank-2 coordinate tensor"
        );
        assert_eq!(tensor.shape[1], 4, "{key} must be [N,4]");
        let rows = tensor.shape[0];
        assert_eq!(
            tensor.data.len(),
            rows.saturating_mul(4),
            "{key} data length mismatch"
        );
        let mut coords = Vec::with_capacity(rows);
        for row_idx in 0..rows {
            let base = row_idx.saturating_mul(4);
            let mut row = [0u32; 4];
            for col in 0..4 {
                let value = tensor.data[base + col];
                let rounded = value.round();
                assert!(
                    (value - rounded).abs() <= 1.0e-3 && rounded >= 0.0,
                    "{key} coordinate value must be a non-negative integer, got {value} at row {row_idx} col {col}"
                );
                row[col] = rounded as u32;
            }
            coords.push(row);
        }
        Some(coords)
    }

    #[derive(Debug)]
    struct SparseDecoderProbeCoordReport {
        reference_count: usize,
        actual_count: usize,
        overlap: usize,
        missing: Vec<[u32; 4]>,
        extra: Vec<[u32; 4]>,
    }

    fn sparse_decoder_probe_coord_report(
        reference: &[[u32; 4]],
        actual: &[[u32; 4]],
    ) -> SparseDecoderProbeCoordReport {
        let reference_set = reference.iter().copied().collect::<BTreeSet<_>>();
        let actual_set = actual.iter().copied().collect::<BTreeSet<_>>();
        let missing = reference_set
            .difference(&actual_set)
            .copied()
            .collect::<Vec<_>>();
        let extra = actual_set
            .difference(&reference_set)
            .copied()
            .collect::<Vec<_>>();
        SparseDecoderProbeCoordReport {
            reference_count: reference_set.len(),
            actual_count: actual_set.len(),
            overlap: reference_set.intersection(&actual_set).count(),
            missing,
            extra,
        }
    }
}

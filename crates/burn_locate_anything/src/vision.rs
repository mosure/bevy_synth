use serde::{Deserialize, Serialize};

use crate::config::LocateAnythingModelConfig;
use crate::tensor_io::{LoadedTensorF32, load_required_tensors_from_safetensors_file};
use crate::{LocateAnythingError, LocateAnythingResult};

pub const LOCATE_ANYTHING_SAFE_IN_TOKEN_LIMIT: u32 = 4_096;
pub const LOCATE_ANYTHING_CHECKPOINT_IN_TOKEN_LIMIT: u32 = 25_600;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct VisionConfig {
    pub encoder: String,
    pub in_token_limit: u32,
    pub patch_size: u32,
    pub merge_kernel_size: [u32; 2],
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            encoder: "MoonViT-SO-400M".to_string(),
            in_token_limit: LOCATE_ANYTHING_SAFE_IN_TOKEN_LIMIT,
            patch_size: 14,
            merge_kernel_size: [2, 2],
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
        }
    }
}

impl VisionConfig {
    pub fn from_model_config(config: &LocateAnythingModelConfig, in_token_limit: u32) -> Self {
        let vision = &config.vision_config;
        Self {
            encoder: vision.model_type.clone(),
            in_token_limit,
            patch_size: vision.patch_size as u32,
            merge_kernel_size: [
                vision.merge_kernel_size[0] as u32,
                vision.merge_kernel_size[1] as u32,
            ],
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ImagePreprocessPlan {
    pub source_width: u32,
    pub source_height: u32,
    pub resized_width: u32,
    pub resized_height: u32,
    /// Upstream `image_grid_hws`: `[height / patch_size, width / patch_size]`.
    pub patch_grid: [u32; 2],
    pub merged_token_count: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreprocessedImagePatches {
    pub patches: Vec<f32>,
    pub patch_shape: [usize; 4],
    pub image_grid_hws: Vec<[usize; 2]>,
    pub plan: ImagePreprocessPlan,
}

pub fn plan_preprocess(width: u32, height: u32, config: &VisionConfig) -> ImagePreprocessPlan {
    let width = width.max(1);
    let height = height.max(1);
    let patch = config.patch_size.max(1);
    let token_area = (width / patch).saturating_mul(height / patch).max(1);
    let scale = if token_area > config.in_token_limit {
        (config.in_token_limit as f32 / token_area as f32).sqrt()
    } else {
        1.0
    };
    let scaled_width = (width as f32 * scale) as u32;
    let scaled_height = (height as f32 * scale) as u32;
    let target_multiple_w = config.merge_kernel_size[1].max(1) * patch;
    let target_multiple_h = config.merge_kernel_size[0].max(1) * patch;
    let resized_width = ceil_to_multiple(scaled_width.max(1), target_multiple_w);
    let resized_height = ceil_to_multiple(scaled_height.max(1), target_multiple_h);
    let grid_h = resized_height / patch;
    let grid_w = resized_width / patch;
    ImagePreprocessPlan {
        source_width: width,
        source_height: height,
        resized_width,
        resized_height,
        patch_grid: [grid_h, grid_w],
        merged_token_count: (grid_h / config.merge_kernel_size[0].max(1))
            .saturating_mul(grid_w / config.merge_kernel_size[1].max(1)),
    }
}

fn ceil_to_multiple(value: u32, multiple: u32) -> u32 {
    value.div_ceil(multiple) * multiple
}

pub fn preprocess_image_to_patches(
    image: &image::DynamicImage,
    config: &VisionConfig,
) -> LocateAnythingResult<PreprocessedImagePatches> {
    let rgb = image.to_rgb8();
    let plan = plan_preprocess(rgb.width(), rgb.height(), config);
    let resized = image::DynamicImage::ImageRgb8(rgb)
        .resize_exact(
            plan.resized_width,
            plan.resized_height,
            image::imageops::FilterType::CatmullRom,
        )
        .to_rgb8();
    let patch = config.patch_size as usize;
    let channels = 3usize;
    let grid_h = (plan.resized_height as usize) / patch;
    let grid_w = (plan.resized_width as usize) / patch;
    if grid_h == 0 || grid_w == 0 {
        return Err(LocateAnythingError::Config(format!(
            "LocateAnything preprocess produced empty patch grid [{grid_h}, {grid_w}] from {}x{}",
            plan.source_width, plan.source_height
        )));
    }
    if grid_h >= 512 || grid_w >= 512 {
        return Err(LocateAnythingError::Config(format!(
            "LocateAnything patch grid [{grid_h}, {grid_w}] exceeds MoonViT positional embedding limit"
        )));
    }
    let token_count = grid_h * grid_w;
    let mut patches = Vec::with_capacity(token_count * channels * patch * patch);
    for patch_y in 0..grid_h {
        for patch_x in 0..grid_w {
            for channel in 0..channels {
                for y in 0..patch {
                    for x in 0..patch {
                        let pixel = resized
                            .get_pixel((patch_x * patch + x) as u32, (patch_y * patch + y) as u32);
                        let value = pixel[channel] as f32 / 255.0;
                        patches.push((value - config.mean[channel]) / config.std[channel]);
                    }
                }
            }
        }
    }
    Ok(PreprocessedImagePatches {
        patches,
        patch_shape: [token_count, channels, patch, patch],
        image_grid_hws: vec![[grid_h, grid_w]],
        plan,
    })
}

#[derive(Clone, Debug, PartialEq)]
pub struct PatchEmbedWeights {
    pub out_dim: usize,
    pub patch_shape: [usize; 3],
    pub proj_weight: Vec<f32>,
    pub proj_bias: Vec<f32>,
    pub pos_emb_shape: [usize; 3],
    pub pos_emb_weight: Vec<f32>,
}

impl PatchEmbedWeights {
    pub fn from_safetensors_file(path: impl AsRef<std::path::Path>) -> LocateAnythingResult<Self> {
        let path = path.as_ref();
        let tensors = load_required_tensors_from_safetensors_file(
            path,
            &[
                "vision_model.patch_embed.proj.weight",
                "vision_model.patch_embed.proj.bias",
                "vision_model.patch_embed.pos_emb.weight",
            ],
        )?;
        Self::from_loaded_tensors(&tensors)
    }

    pub fn from_loaded_tensors(
        tensors: &[(String, LoadedTensorF32)],
    ) -> LocateAnythingResult<Self> {
        let proj_weight = find_tensor(tensors, "vision_model.patch_embed.proj.weight")?;
        let proj_bias = find_tensor(tensors, "vision_model.patch_embed.proj.bias")?;
        let pos_emb_weight = find_tensor(tensors, "vision_model.patch_embed.pos_emb.weight")?;
        let proj_shape: [usize; 4] =
            proj_weight
                .shape
                .clone()
                .try_into()
                .map_err(|shape: Vec<usize>| {
                    LocateAnythingError::Config(format!(
                        "patch proj weight expected rank 4, got {shape:?}"
                    ))
                })?;
        let out_dim = proj_shape[0];
        if proj_shape[2] != proj_shape[3] {
            return Err(LocateAnythingError::Config(format!(
                "patch proj kernel must be square, got {:?}",
                &proj_shape[2..]
            )));
        }
        let bias_len = expect_1d("vision_model.patch_embed.proj.bias", proj_bias)?;
        if bias_len != out_dim {
            return Err(LocateAnythingError::Config(format!(
                "patch proj bias len {bias_len} does not match out dim {out_dim}"
            )));
        }
        let pos_shape: [usize; 3] =
            pos_emb_weight
                .shape
                .clone()
                .try_into()
                .map_err(|shape: Vec<usize>| {
                    LocateAnythingError::Config(format!(
                        "patch positional embedding expected rank 3, got {shape:?}"
                    ))
                })?;
        if pos_shape[2] != out_dim {
            return Err(LocateAnythingError::Config(format!(
                "patch pos dim {} does not match out dim {out_dim}",
                pos_shape[2]
            )));
        }

        Ok(Self {
            out_dim,
            patch_shape: [proj_shape[1], proj_shape[2], proj_shape[3]],
            proj_weight: proj_weight.data.clone(),
            proj_bias: proj_bias.data.clone(),
            pos_emb_shape: pos_shape,
            pos_emb_weight: pos_emb_weight.data.clone(),
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MoonVitEncoderBlockWeights {
    pub layer_index: usize,
    pub hidden_dim: usize,
    pub mlp_dim: usize,
    pub num_heads: usize,
    pub norm0_weight: Vec<f32>,
    pub norm0_bias: Vec<f32>,
    pub norm1_weight: Vec<f32>,
    pub norm1_bias: Vec<f32>,
    pub wqkv_weight: Vec<f32>,
    pub wqkv_bias: Vec<f32>,
    pub wo_weight: Vec<f32>,
    pub wo_bias: Vec<f32>,
    pub mlp_fc0_weight: Vec<f32>,
    pub mlp_fc0_bias: Vec<f32>,
    pub mlp_fc1_weight: Vec<f32>,
    pub mlp_fc1_bias: Vec<f32>,
}

impl MoonVitEncoderBlockWeights {
    pub fn from_safetensors_file(
        path: impl AsRef<std::path::Path>,
        layer_index: usize,
    ) -> LocateAnythingResult<Self> {
        let prefix = format!("vision_model.encoder.blocks.{layer_index}");
        let keys = [
            format!("{prefix}.norm0.weight"),
            format!("{prefix}.norm0.bias"),
            format!("{prefix}.norm1.weight"),
            format!("{prefix}.norm1.bias"),
            format!("{prefix}.wqkv.weight"),
            format!("{prefix}.wqkv.bias"),
            format!("{prefix}.wo.weight"),
            format!("{prefix}.wo.bias"),
            format!("{prefix}.mlp.fc0.weight"),
            format!("{prefix}.mlp.fc0.bias"),
            format!("{prefix}.mlp.fc1.weight"),
            format!("{prefix}.mlp.fc1.bias"),
        ];
        let key_refs = keys.iter().map(String::as_str).collect::<Vec<_>>();
        let tensors = load_required_tensors_from_safetensors_file(path.as_ref(), &key_refs)?;
        Self::from_loaded_tensors(layer_index, &tensors)
    }

    pub fn from_loaded_tensors(
        layer_index: usize,
        tensors: &[(String, LoadedTensorF32)],
    ) -> LocateAnythingResult<Self> {
        let prefix = format!("vision_model.encoder.blocks.{layer_index}");
        let norm0_weight = find_tensor(tensors, &format!("{prefix}.norm0.weight"))?;
        let norm0_bias = find_tensor(tensors, &format!("{prefix}.norm0.bias"))?;
        let norm1_weight = find_tensor(tensors, &format!("{prefix}.norm1.weight"))?;
        let norm1_bias = find_tensor(tensors, &format!("{prefix}.norm1.bias"))?;
        let wqkv_weight = find_tensor(tensors, &format!("{prefix}.wqkv.weight"))?;
        let wqkv_bias = find_tensor(tensors, &format!("{prefix}.wqkv.bias"))?;
        let wo_weight = find_tensor(tensors, &format!("{prefix}.wo.weight"))?;
        let wo_bias = find_tensor(tensors, &format!("{prefix}.wo.bias"))?;
        let mlp_fc0_weight = find_tensor(tensors, &format!("{prefix}.mlp.fc0.weight"))?;
        let mlp_fc0_bias = find_tensor(tensors, &format!("{prefix}.mlp.fc0.bias"))?;
        let mlp_fc1_weight = find_tensor(tensors, &format!("{prefix}.mlp.fc1.weight"))?;
        let mlp_fc1_bias = find_tensor(tensors, &format!("{prefix}.mlp.fc1.bias"))?;

        let hidden_dim = expect_1d(&format!("{prefix}.norm0.weight"), norm0_weight)?;
        expect_1d_len(&format!("{prefix}.norm0.bias"), norm0_bias, hidden_dim)?;
        expect_1d_len(&format!("{prefix}.norm1.weight"), norm1_weight, hidden_dim)?;
        expect_1d_len(&format!("{prefix}.norm1.bias"), norm1_bias, hidden_dim)?;
        let [qkv_dim, qkv_input] = expect_2d(&format!("{prefix}.wqkv.weight"), wqkv_weight)?;
        if qkv_input != hidden_dim || qkv_dim != hidden_dim * 3 {
            return Err(LocateAnythingError::Config(format!(
                "{prefix}.wqkv.weight expected [{}, {hidden_dim}], got [{qkv_dim}, {qkv_input}]",
                hidden_dim * 3
            )));
        }
        expect_1d_len(&format!("{prefix}.wqkv.bias"), wqkv_bias, hidden_dim * 3)?;
        let [wo_dim, wo_input] = expect_2d(&format!("{prefix}.wo.weight"), wo_weight)?;
        if wo_dim != hidden_dim || wo_input != hidden_dim {
            return Err(LocateAnythingError::Config(format!(
                "{prefix}.wo.weight expected [{hidden_dim}, {hidden_dim}], got [{wo_dim}, {wo_input}]"
            )));
        }
        expect_1d_len(&format!("{prefix}.wo.bias"), wo_bias, hidden_dim)?;
        let [mlp_dim, mlp_input] = expect_2d(&format!("{prefix}.mlp.fc0.weight"), mlp_fc0_weight)?;
        if mlp_input != hidden_dim {
            return Err(LocateAnythingError::Config(format!(
                "{prefix}.mlp.fc0.weight input dim {mlp_input} does not match hidden dim {hidden_dim}"
            )));
        }
        expect_1d_len(&format!("{prefix}.mlp.fc0.bias"), mlp_fc0_bias, mlp_dim)?;
        let [mlp_out, mlp_hidden] = expect_2d(&format!("{prefix}.mlp.fc1.weight"), mlp_fc1_weight)?;
        if mlp_out != hidden_dim || mlp_hidden != mlp_dim {
            return Err(LocateAnythingError::Config(format!(
                "{prefix}.mlp.fc1.weight expected [{hidden_dim}, {mlp_dim}], got [{mlp_out}, {mlp_hidden}]"
            )));
        }
        expect_1d_len(&format!("{prefix}.mlp.fc1.bias"), mlp_fc1_bias, hidden_dim)?;

        let num_heads = 16;
        if hidden_dim % num_heads != 0 || (hidden_dim / num_heads) % 4 != 0 {
            return Err(LocateAnythingError::Config(format!(
                "MoonViT hidden dim {hidden_dim} is not compatible with {num_heads} heads and 2D RoPE"
            )));
        }

        Ok(Self {
            layer_index,
            hidden_dim,
            mlp_dim,
            num_heads,
            norm0_weight: norm0_weight.data.clone(),
            norm0_bias: norm0_bias.data.clone(),
            norm1_weight: norm1_weight.data.clone(),
            norm1_bias: norm1_bias.data.clone(),
            wqkv_weight: wqkv_weight.data.clone(),
            wqkv_bias: wqkv_bias.data.clone(),
            wo_weight: wo_weight.data.clone(),
            wo_bias: wo_bias.data.clone(),
            mlp_fc0_weight: mlp_fc0_weight.data.clone(),
            mlp_fc0_bias: mlp_fc0_bias.data.clone(),
            mlp_fc1_weight: mlp_fc1_weight.data.clone(),
            mlp_fc1_bias: mlp_fc1_bias.data.clone(),
        })
    }

    pub fn with_num_heads(mut self, num_heads: usize) -> LocateAnythingResult<Self> {
        if num_heads == 0 || !self.hidden_dim.is_multiple_of(num_heads) {
            return Err(LocateAnythingError::Config(format!(
                "MoonViT hidden dim {} must divide num_heads {num_heads}",
                self.hidden_dim
            )));
        }
        if !(self.hidden_dim / num_heads).is_multiple_of(4) {
            return Err(LocateAnythingError::Config(format!(
                "MoonViT head dim {} must be divisible by 4 for 2D RoPE",
                self.hidden_dim / num_heads
            )));
        }
        self.num_heads = num_heads;
        Ok(self)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MoonVitFinalNormWeights {
    pub hidden_dim: usize,
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

impl MoonVitFinalNormWeights {
    pub fn from_safetensors_file(path: impl AsRef<std::path::Path>) -> LocateAnythingResult<Self> {
        let tensors = load_required_tensors_from_safetensors_file(
            path.as_ref(),
            &[
                "vision_model.encoder.final_layernorm.weight",
                "vision_model.encoder.final_layernorm.bias",
            ],
        )?;
        Self::from_loaded_tensors(&tensors)
    }

    pub fn from_loaded_tensors(
        tensors: &[(String, LoadedTensorF32)],
    ) -> LocateAnythingResult<Self> {
        let weight = find_tensor(tensors, "vision_model.encoder.final_layernorm.weight")?;
        let bias = find_tensor(tensors, "vision_model.encoder.final_layernorm.bias")?;
        let hidden_dim = expect_1d("vision_model.encoder.final_layernorm.weight", weight)?;
        expect_1d_len(
            "vision_model.encoder.final_layernorm.bias",
            bias,
            hidden_dim,
        )?;
        Ok(Self {
            hidden_dim,
            weight: weight.data.clone(),
            bias: bias.data.clone(),
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MoonVitEncoderWeights {
    pub blocks: Vec<MoonVitEncoderBlockWeights>,
    pub final_norm: MoonVitFinalNormWeights,
}

impl MoonVitEncoderWeights {
    pub fn from_safetensors_file(
        path: impl AsRef<std::path::Path>,
        layer_count: usize,
        num_heads: usize,
    ) -> LocateAnythingResult<Self> {
        let path = path.as_ref();
        let blocks = (0..layer_count)
            .map(|layer| {
                MoonVitEncoderBlockWeights::from_safetensors_file(path, layer)?
                    .with_num_heads(num_heads)
            })
            .collect::<LocateAnythingResult<Vec<_>>>()?;
        let final_norm = MoonVitFinalNormWeights::from_safetensors_file(path)?;
        Ok(Self { blocks, final_norm })
    }
}

fn find_tensor<'a>(
    tensors: &'a [(String, LoadedTensorF32)],
    key: &str,
) -> LocateAnythingResult<&'a LoadedTensorF32> {
    tensors
        .iter()
        .find_map(|(name, tensor)| (name == key).then_some(tensor))
        .ok_or_else(|| LocateAnythingError::Runtime(format!("missing tensor `{key}`")))
}

fn expect_1d(key: &str, tensor: &LoadedTensorF32) -> LocateAnythingResult<usize> {
    if tensor.shape.len() != 1 {
        return Err(LocateAnythingError::Config(format!(
            "tensor `{key}` expected rank 1, got {:?}",
            tensor.shape
        )));
    }
    Ok(tensor.shape[0])
}

fn expect_1d_len(key: &str, tensor: &LoadedTensorF32, expected: usize) -> LocateAnythingResult<()> {
    let actual = expect_1d(key, tensor)?;
    if actual != expected {
        return Err(LocateAnythingError::Config(format!(
            "tensor `{key}` expected len {expected}, got {actual}"
        )));
    }
    Ok(())
}

fn expect_2d(key: &str, tensor: &LoadedTensorF32) -> LocateAnythingResult<[usize; 2]> {
    if tensor.shape.len() != 2 {
        return Err(LocateAnythingError::Config(format!(
            "tensor `{key}` expected rank 2, got {:?}",
            tensor.shape
        )));
    }
    Ok([tensor.shape[0], tensor.shape[1]])
}

#[cfg(any(
    feature = "backend_ndarray",
    feature = "backend_wgpu",
    feature = "backend_cuda"
))]
pub mod burn_vision {
    use burn::prelude::*;
    use burn::tensor::activation::softmax;
    use burn::tensor::module::interpolate;
    use burn::tensor::ops::{InterpolateMode, InterpolateOptions};

    use super::{
        MoonVitEncoderBlockWeights, MoonVitEncoderWeights, MoonVitFinalNormWeights,
        PatchEmbedWeights,
    };

    #[derive(Debug)]
    pub struct BurnPatchEmbed<B: Backend> {
        pub weights: PatchEmbedWeights,
        proj_weight: Tensor<B, 2>,
        proj_bias: Tensor<B, 1>,
        pos_emb_weight: Tensor<B, 3>,
    }

    impl<B: Backend> BurnPatchEmbed<B> {
        pub fn from_weights(weights: PatchEmbedWeights, device: &B::Device) -> Self {
            let flattened = weights.patch_shape.iter().product::<usize>();
            Self {
                proj_weight: tensor2(&weights.proj_weight, [weights.out_dim, flattened], device),
                proj_bias: tensor1(&weights.proj_bias, device),
                pos_emb_weight: tensor3(&weights.pos_emb_weight, weights.pos_emb_shape, device),
                weights,
            }
        }

        pub fn forward(&self, patches: Tensor<B, 4>, grid_hws: &[[usize; 2]]) -> Tensor<B, 2> {
            let pos = self.position_embeddings(grid_hws);
            self.forward_with_position_embeddings(patches, pos, grid_hws)
        }

        pub fn forward_with_position_embeddings(
            &self,
            patches: Tensor<B, 4>,
            position_embeddings: Tensor<B, 2>,
            grid_hws: &[[usize; 2]],
        ) -> Tensor<B, 2> {
            let [tokens, channels, height, width] = patches.dims();
            let expected_tokens = grid_hws
                .iter()
                .map(|[grid_h, grid_w]| grid_h * grid_w)
                .sum::<usize>();
            assert_eq!(tokens, expected_tokens);
            assert_eq!(position_embeddings.dims(), [tokens, self.weights.out_dim]);
            assert_eq!([channels, height, width], self.weights.patch_shape);
            let flattened = channels * height * width;
            let projected = patches
                .reshape([tokens, flattened])
                .matmul(self.proj_weight.clone().swap_dims(0, 1))
                + self.proj_bias.clone().reshape([1, self.weights.out_dim]);
            projected + position_embeddings
        }

        pub fn position_embeddings(&self, grid_hws: &[[usize; 2]]) -> Tensor<B, 2> {
            let [pos_h, pos_w, dim] = self.weights.pos_emb_shape;
            let pos = self
                .pos_emb_weight
                .clone()
                .permute([2, 0, 1])
                .reshape([1, dim, pos_h, pos_w]);
            let options =
                InterpolateOptions::new(InterpolateMode::Bicubic).with_align_corners(false);
            let mut segments = Vec::with_capacity(grid_hws.len());
            for &[grid_h, grid_w] in grid_hws {
                let interp = if grid_h == pos_h && grid_w == pos_w {
                    pos.clone()
                } else {
                    interpolate(pos.clone(), [grid_h, grid_w], options.clone())
                };
                segments.push(
                    interp
                        .reshape([dim, grid_h, grid_w])
                        .permute([1, 2, 0])
                        .reshape([grid_h * grid_w, dim]),
                );
            }
            Tensor::cat(segments, 0)
        }
    }

    fn tensor1<B: Backend>(values: &[f32], device: &B::Device) -> Tensor<B, 1> {
        Tensor::<B, 1>::from_floats(values, device)
    }

    fn tensor2<B: Backend>(values: &[f32], shape: [usize; 2], device: &B::Device) -> Tensor<B, 2> {
        Tensor::<B, 1>::from_floats(values, device).reshape(shape)
    }

    fn tensor3<B: Backend>(values: &[f32], shape: [usize; 3], device: &B::Device) -> Tensor<B, 3> {
        Tensor::<B, 1>::from_floats(values, device).reshape(shape)
    }

    #[derive(Debug)]
    pub struct BurnMoonVitEncoderBlock<B: Backend> {
        pub weights: MoonVitEncoderBlockWeights,
        norm0_weight: Tensor<B, 1>,
        norm0_bias: Tensor<B, 1>,
        norm1_weight: Tensor<B, 1>,
        norm1_bias: Tensor<B, 1>,
        wqkv_weight: Tensor<B, 2>,
        wqkv_bias: Tensor<B, 1>,
        wo_weight: Tensor<B, 2>,
        wo_bias: Tensor<B, 1>,
        mlp_fc0_weight: Tensor<B, 2>,
        mlp_fc0_bias: Tensor<B, 1>,
        mlp_fc1_weight: Tensor<B, 2>,
        mlp_fc1_bias: Tensor<B, 1>,
    }

    impl<B: Backend> BurnMoonVitEncoderBlock<B> {
        pub fn from_weights(weights: MoonVitEncoderBlockWeights, device: &B::Device) -> Self {
            let hidden = weights.hidden_dim;
            let mlp = weights.mlp_dim;
            Self {
                norm0_weight: tensor1(&weights.norm0_weight, device),
                norm0_bias: tensor1(&weights.norm0_bias, device),
                norm1_weight: tensor1(&weights.norm1_weight, device),
                norm1_bias: tensor1(&weights.norm1_bias, device),
                wqkv_weight: tensor2(&weights.wqkv_weight, [hidden * 3, hidden], device),
                wqkv_bias: tensor1(&weights.wqkv_bias, device),
                wo_weight: tensor2(&weights.wo_weight, [hidden, hidden], device),
                wo_bias: tensor1(&weights.wo_bias, device),
                mlp_fc0_weight: tensor2(&weights.mlp_fc0_weight, [mlp, hidden], device),
                mlp_fc0_bias: tensor1(&weights.mlp_fc0_bias, device),
                mlp_fc1_weight: tensor2(&weights.mlp_fc1_weight, [hidden, mlp], device),
                mlp_fc1_bias: tensor1(&weights.mlp_fc1_bias, device),
                weights,
            }
        }

        pub fn forward(
            &self,
            hidden_states: Tensor<B, 2>,
            grid_hws: &[[usize; 2]],
        ) -> Tensor<B, 2> {
            let [tokens, hidden] = hidden_states.dims();
            assert_eq!(hidden, self.weights.hidden_dim);
            let residual = hidden_states.clone();
            let normalized = layer_norm_2d(
                hidden_states,
                self.norm0_weight.clone(),
                self.norm0_bias.clone(),
                1.0e-5,
            );
            let attn_out = self.attention_qkvpacked(normalized, grid_hws);
            let hidden_states = residual + attn_out;

            let residual = hidden_states.clone();
            let mlp_in = layer_norm_2d(
                hidden_states,
                self.norm1_weight.clone(),
                self.norm1_bias.clone(),
                1.0e-5,
            );
            let mlp_hidden = mlp_in.matmul(self.mlp_fc0_weight.clone().swap_dims(0, 1))
                + self.mlp_fc0_bias.clone().reshape([1, self.weights.mlp_dim]);
            let mlp_out = gelu_tanh(mlp_hidden).matmul(self.mlp_fc1_weight.clone().swap_dims(0, 1))
                + self.mlp_fc1_bias.clone().reshape([1, hidden]);
            let out = residual + mlp_out;
            assert_eq!(out.dims(), [tokens, hidden]);
            out
        }

        pub fn attention_qkvpacked(
            &self,
            hidden_states: Tensor<B, 2>,
            grid_hws: &[[usize; 2]],
        ) -> Tensor<B, 2> {
            let [tokens, hidden] = hidden_states.dims();
            let heads = self.weights.num_heads;
            let head_dim = hidden / heads;
            let qkv = hidden_states.matmul(self.wqkv_weight.clone().swap_dims(0, 1))
                + self.wqkv_bias.clone().reshape([1, hidden * 3]);
            let qkv = qkv.reshape([tokens, 3, heads, head_dim]);
            let q = qkv
                .clone()
                .slice([0..tokens, 0..1, 0..heads, 0..head_dim])
                .reshape([tokens, heads, head_dim]);
            let k = qkv
                .clone()
                .slice([0..tokens, 1..2, 0..heads, 0..head_dim])
                .reshape([tokens, heads, head_dim]);
            let v = qkv
                .slice([0..tokens, 2..3, 0..heads, 0..head_dim])
                .reshape([tokens, heads, head_dim]);
            let (cos, sin) = rope_2d_cos_sin::<B>(grid_hws, head_dim, &self.wo_bias.device());
            let q = apply_rope_2d(q, cos.clone(), sin.clone());
            let k = apply_rope_2d(k, cos, sin);
            packed_segment_attention(q, k, v, grid_hws)
                .reshape([tokens, hidden])
                .matmul(self.wo_weight.clone().swap_dims(0, 1))
                + self.wo_bias.clone().reshape([1, hidden])
        }
    }

    #[derive(Debug)]
    pub struct BurnMoonVitFinalNorm<B: Backend> {
        pub weights: MoonVitFinalNormWeights,
        weight: Tensor<B, 1>,
        bias: Tensor<B, 1>,
    }

    impl<B: Backend> BurnMoonVitFinalNorm<B> {
        pub fn from_weights(weights: MoonVitFinalNormWeights, device: &B::Device) -> Self {
            Self {
                weight: tensor1(&weights.weight, device),
                bias: tensor1(&weights.bias, device),
                weights,
            }
        }

        pub fn forward(&self, hidden_states: Tensor<B, 2>) -> Tensor<B, 2> {
            layer_norm_2d(
                hidden_states,
                self.weight.clone(),
                self.bias.clone(),
                1.0e-5,
            )
        }
    }

    #[derive(Debug)]
    pub struct BurnMoonVitEncoder<B: Backend> {
        pub weights: MoonVitEncoderWeights,
        blocks: Vec<BurnMoonVitEncoderBlock<B>>,
        final_norm: BurnMoonVitFinalNorm<B>,
    }

    impl<B: Backend> BurnMoonVitEncoder<B> {
        pub fn from_weights(weights: MoonVitEncoderWeights, device: &B::Device) -> Self {
            let blocks = weights
                .blocks
                .iter()
                .cloned()
                .map(|block| BurnMoonVitEncoderBlock::from_weights(block, device))
                .collect();
            let final_norm = BurnMoonVitFinalNorm::from_weights(weights.final_norm.clone(), device);
            Self {
                weights,
                blocks,
                final_norm,
            }
        }

        pub fn forward(
            &self,
            mut hidden_states: Tensor<B, 2>,
            grid_hws: &[[usize; 2]],
        ) -> Tensor<B, 2> {
            for block in &self.blocks {
                hidden_states = block.forward(hidden_states, grid_hws);
            }
            self.final_norm.forward(hidden_states)
        }
    }

    pub fn layer_norm_2d<B: Backend>(
        input: Tensor<B, 2>,
        weight: Tensor<B, 1>,
        bias: Tensor<B, 1>,
        eps: f32,
    ) -> Tensor<B, 2> {
        let [_tokens, channels] = input.dims();
        let (var, mean) = input.clone().var_mean_bias(1);
        (input - mean) / var.add_scalar(eps).sqrt() * weight.reshape([1, channels])
            + bias.reshape([1, channels])
    }

    pub fn gelu_tanh<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
        let c0 = 0.044_715_f32;
        let c1 = 0.797_884_6_f32;
        let x3 = x.clone().powf_scalar(3.0).mul_scalar(c0);
        let t = x.clone().add(x3).mul_scalar(c1).tanh();
        x.mul_scalar(0.5).mul(t.add_scalar(1.0))
    }

    pub fn rope_2d_cos_sin<B: Backend>(
        grid_hws: &[[usize; 2]],
        head_dim: usize,
        device: &B::Device,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        assert_eq!(head_dim % 4, 0);
        let pairs = head_dim / 2;
        let quarter = head_dim / 4;
        let tokens = grid_hws.iter().map(|[h, w]| h * w).sum::<usize>();
        let mut cos = Vec::with_capacity(tokens * pairs);
        let mut sin = Vec::with_capacity(tokens * pairs);
        let inv_freq = (0..quarter)
            .map(|idx| 1.0 / 10_000_f32.powf((4 * idx) as f32 / head_dim as f32))
            .collect::<Vec<_>>();
        for &[height, width] in grid_hws {
            for y in 0..height {
                for x in 0..width {
                    for &freq in &inv_freq {
                        let phase = x as f32 * freq;
                        cos.push(phase.cos());
                        sin.push(phase.sin());
                        let phase = y as f32 * freq;
                        cos.push(phase.cos());
                        sin.push(phase.sin());
                    }
                }
            }
        }
        (
            Tensor::<B, 1>::from_floats(cos.as_slice(), device).reshape([tokens, 1, pairs]),
            Tensor::<B, 1>::from_floats(sin.as_slice(), device).reshape([tokens, 1, pairs]),
        )
    }

    pub fn apply_rope_2d<B: Backend>(
        input: Tensor<B, 3>,
        cos: Tensor<B, 3>,
        sin: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let [tokens, heads, head_dim] = input.dims();
        let pairs = head_dim / 2;
        let input = input.reshape([tokens, heads, pairs, 2]);
        let real = input
            .clone()
            .slice([0..tokens, 0..heads, 0..pairs, 0..1])
            .reshape([tokens, heads, pairs]);
        let imag = input
            .slice([0..tokens, 0..heads, 0..pairs, 1..2])
            .reshape([tokens, heads, pairs]);
        let out_real = real.clone() * cos.clone() - imag.clone() * sin.clone();
        let out_imag = real * sin + imag * cos;
        let out_real = out_real.unsqueeze_dim::<4>(3);
        let out_imag = out_imag.unsqueeze_dim::<4>(3);
        Tensor::cat(vec![out_real, out_imag], 3).reshape([tokens, heads, head_dim])
    }

    pub fn packed_segment_attention<B: Backend>(
        q: Tensor<B, 3>,
        k: Tensor<B, 3>,
        v: Tensor<B, 3>,
        grid_hws: &[[usize; 2]],
    ) -> Tensor<B, 3> {
        let [tokens, heads, head_dim] = q.dims();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut offset = 0usize;
        let mut segments = Vec::with_capacity(grid_hws.len());
        for &[height, width] in grid_hws {
            let len = height * width;
            let q_seg = q
                .clone()
                .slice([offset..offset + len, 0..heads, 0..head_dim])
                .permute([1, 0, 2]);
            let k_seg = k
                .clone()
                .slice([offset..offset + len, 0..heads, 0..head_dim])
                .permute([1, 0, 2]);
            let v_seg = v
                .clone()
                .slice([offset..offset + len, 0..heads, 0..head_dim])
                .permute([1, 0, 2]);
            let scores = q_seg.matmul(k_seg.swap_dims(1, 2)).mul_scalar(scale);
            let attn = softmax(scores, 2);
            segments.push(attn.matmul(v_seg).permute([1, 0, 2]));
            offset += len;
        }
        assert_eq!(offset, tokens);
        Tensor::cat(segments, 0)
    }

    pub fn patch_merger<B: Backend>(
        hidden_states: Tensor<B, 2>,
        grid_hws: &[[usize; 2]],
        merge_kernel_size: [usize; 2],
    ) -> Tensor<B, 2> {
        let [tokens, channels] = hidden_states.dims();
        let expected_tokens = grid_hws
            .iter()
            .map(|[grid_h, grid_w]| grid_h * grid_w)
            .sum::<usize>();
        assert_eq!(tokens, expected_tokens);
        let [kernel_h, kernel_w] = merge_kernel_size;
        let mut offset = 0usize;
        let mut segments = Vec::with_capacity(grid_hws.len());
        for &[height, width] in grid_hws {
            assert_eq!(height % kernel_h, 0);
            assert_eq!(width % kernel_w, 0);
            let len = height * width;
            let new_h = height / kernel_h;
            let new_w = width / kernel_w;
            let segment = hidden_states
                .clone()
                .slice([offset..offset + len, 0..channels])
                .reshape([new_h, kernel_h, new_w, kernel_w, channels])
                .permute([0, 2, 1, 3, 4])
                .reshape([new_h * new_w, kernel_h * kernel_w * channels]);
            segments.push(segment);
            offset += len;
        }
        Tensor::cat(segments, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_io::load_tensor_from_safetensors_file;

    #[test]
    fn preprocess_plan_matches_upstream_multiples() {
        let config = VisionConfig::default();
        let plan = plan_preprocess(1600, 900, &config);
        assert_eq!(plan.resized_width % 28, 0);
        assert_eq!(plan.resized_height % 28, 0);
        assert_eq!(plan.patch_grid, [50, 86]);
        assert_eq!(
            plan.patch_grid,
            [plan.resized_height / 14, plan.resized_width / 14]
        );
        assert!(plan.patch_grid[0] < 512);
        assert!(plan.patch_grid[1] < 512);
    }

    #[test]
    fn preprocess_plan_respects_token_limit() {
        let config = VisionConfig::default();
        let plan = plan_preprocess(8000, 4000, &config);
        assert!(plan.patch_grid[0] * plan.patch_grid[1] <= config.in_token_limit + 512);
        assert_eq!(
            plan.merged_token_count,
            (plan.patch_grid[0] / 2) * (plan.patch_grid[1] / 2)
        );
    }

    #[test]
    fn preprocess_plan_matches_wide_scene_safe_limit() {
        let config = VisionConfig::default();
        let plan = plan_preprocess(3839, 2157, &config);
        assert_eq!(plan.patch_grid, [48, 86]);
        assert_eq!(plan.merged_token_count, 1032);
    }

    #[test]
    fn preprocess_plan_can_represent_checkpoint_token_limit() {
        let config = VisionConfig {
            in_token_limit: LOCATE_ANYTHING_CHECKPOINT_IN_TOKEN_LIMIT,
            ..VisionConfig::default()
        };
        let plan = plan_preprocess(1600, 900, &config);
        assert_eq!(plan.patch_grid, [66, 116]);
        assert_eq!(plan.merged_token_count, 1914);
    }

    #[test]
    fn preprocess_patch_order_matches_torch_patchify_for_tiny_image() {
        let mut image = image::RgbImage::new(28, 28);
        for y in 0..28u32 {
            for x in 0..28u32 {
                image.put_pixel(x, y, image::Rgb([(x + y) as u8, x as u8, y as u8]));
            }
        }
        let out = preprocess_image_to_patches(
            &image::DynamicImage::ImageRgb8(image),
            &VisionConfig {
                in_token_limit: 4,
                ..VisionConfig::default()
            },
        )
        .unwrap();
        assert_eq!(out.patch_shape, [4, 3, 14, 14]);
        assert_eq!(out.image_grid_hws, vec![[2, 2]]);
        assert_eq!(out.plan.merged_token_count, 1);
        assert_eq!(out.patches[0], -1.0);
        assert_eq!(out.patches[14 * 14], -1.0);
        assert_eq!(out.patches[14 * 14 * 2], -1.0);
    }

    #[test]
    fn preprocess_matches_reference_grid_and_pixel_fixture_when_present() {
        let Some(root) = find_repo_root_for_test() else {
            eprintln!("skipping preprocess parity fixture; repo root not found");
            return;
        };
        let Some(image_path) =
            std::env::var_os("LOCATE_ANYTHING_PARITY_IMAGE").map(std::path::PathBuf::from)
        else {
            eprintln!(
                "skipping preprocess parity fixture; set LOCATE_ANYTHING_PARITY_IMAGE to the reference scene image"
            );
            return;
        };
        let fixture = root.join(
            "tmp/runs/20260626T020100Z_locateanything_patch_embed_parity_galaxy/preprocess.safetensors",
        );
        if !image_path.exists() || !fixture.exists() {
            eprintln!(
                "skipping preprocess parity fixture; missing {} or {}",
                image_path.display(),
                fixture.display()
            );
            return;
        }
        let image = image::open(&image_path).unwrap();
        let out = preprocess_image_to_patches(
            &image,
            &VisionConfig {
                in_token_limit: LOCATE_ANYTHING_SAFE_IN_TOKEN_LIMIT,
                ..VisionConfig::default()
            },
        )
        .unwrap();
        let reference_pixels = load_tensor_from_safetensors_file(&fixture, "pixel_values").unwrap();
        let reference_grid = load_tensor_from_safetensors_file(&fixture, "image_grid_hws").unwrap();
        assert_eq!(out.patch_shape, [4300, 3, 14, 14]);
        assert_eq!(out.image_grid_hws, vec![[50, 86]]);
        assert_eq!(
            reference_grid
                .data
                .iter()
                .map(|value| *value as usize)
                .collect::<Vec<_>>(),
            vec![50, 86]
        );
        assert_eq!(out.patches.len(), reference_pixels.data.len());
        let mut max_abs = 0.0f32;
        let mut sum_abs = 0.0f64;
        for (actual, expected) in out.patches.iter().zip(reference_pixels.data.iter()) {
            let diff = (actual - expected).abs();
            max_abs = max_abs.max(diff);
            sum_abs += diff as f64;
        }
        let mean_abs = sum_abs / out.patches.len() as f64;
        assert!(
            mean_abs < 0.02 && max_abs < 0.35,
            "preprocess diff too high: mean_abs={mean_abs:.6e} max_abs={max_abs:.6e}"
        );
    }

    fn find_repo_root_for_test() -> Option<std::path::PathBuf> {
        let mut dir = std::env::current_dir().ok()?;
        loop {
            if dir.join("Cargo.toml").exists() && dir.join("crates/burn_locate_anything").exists() {
                return Some(dir);
            }
            if !dir.pop() {
                return None;
            }
        }
    }
}

#[cfg(all(
    test,
    any(
        feature = "backend_ndarray",
        feature = "backend_wgpu",
        feature = "backend_cuda"
    )
))]
mod parity_tests {
    use std::time::Instant;

    use super::burn_vision::{BurnMoonVitEncoderBlock, BurnPatchEmbed};
    use super::*;
    use crate::tensor_io::load_tensor_from_safetensors_file;

    #[cfg(feature = "backend_ndarray")]
    #[test]
    fn patch_embed_matches_reference_hook_when_enabled() {
        use burn::tensor::backend::BackendTypes;

        let device = <burn::backend::NdArray<f32> as BackendTypes>::Device::default();
        run_patch_embed_parity::<burn::backend::NdArray<f32>>(&device);
    }

    #[cfg(feature = "backend_wgpu")]
    #[test]
    fn patch_embed_matches_reference_hook_wgpu_when_enabled() {
        if std::env::var("BURN_WGPU_CORRECTNESS").is_err() {
            eprintln!("skipping: set BURN_WGPU_CORRECTNESS=1 to run WGPU patch embed parity");
            return;
        }
        let device = burn_wgpu::WgpuDevice::default();
        run_patch_embed_parity::<burn_wgpu::Wgpu<f32, i32, u32>>(&device);
    }

    #[cfg(feature = "backend_ndarray")]
    #[test]
    fn patch_merger_matches_reference_hook_when_enabled() {
        use burn::tensor::backend::BackendTypes;

        let device = <burn::backend::NdArray<f32> as BackendTypes>::Device::default();
        run_patch_merger_parity::<burn::backend::NdArray<f32>>(&device);
    }

    #[cfg(feature = "backend_ndarray")]
    #[test]
    fn moonvit_block_zero_projections_preserve_residual_ndarray() {
        use burn::tensor::backend::BackendTypes;

        let device = <burn::backend::NdArray<f32> as BackendTypes>::Device::default();
        let weights = MoonVitEncoderBlockWeights {
            layer_index: 0,
            hidden_dim: 4,
            mlp_dim: 8,
            num_heads: 1,
            norm0_weight: vec![1.0; 4],
            norm0_bias: vec![0.0; 4],
            norm1_weight: vec![1.0; 4],
            norm1_bias: vec![0.0; 4],
            wqkv_weight: vec![0.0; 12 * 4],
            wqkv_bias: vec![0.0; 12],
            wo_weight: vec![0.0; 4 * 4],
            wo_bias: vec![0.0; 4],
            mlp_fc0_weight: vec![0.0; 8 * 4],
            mlp_fc0_bias: vec![0.0; 8],
            mlp_fc1_weight: vec![0.0; 4 * 8],
            mlp_fc1_bias: vec![0.0; 4],
        };
        let model =
            BurnMoonVitEncoderBlock::<burn::backend::NdArray<f32>>::from_weights(weights, &device);
        let input_values = [0.1, 0.2, 0.3, 0.4, -0.2, 0.5, 0.7, -0.8];
        let input = burn::prelude::Tensor::<burn::backend::NdArray<f32>, 1>::from_floats(
            input_values.as_slice(),
            &device,
        )
        .reshape([2, 4]);
        let output = model.forward(input, &[[1, 2]]);
        let output_values = output.into_data().convert::<f32>().to_vec::<f32>().unwrap();
        assert_eq!(output_values, input_values);
    }

    #[cfg(feature = "backend_wgpu")]
    #[test]
    fn patch_merger_matches_reference_hook_wgpu_when_enabled() {
        if std::env::var("BURN_WGPU_CORRECTNESS").is_err() {
            eprintln!("skipping: set BURN_WGPU_CORRECTNESS=1 to run WGPU patch merger parity");
            return;
        }
        let device = burn_wgpu::WgpuDevice::default();
        run_patch_merger_parity::<burn_wgpu::Wgpu<f32, i32, u32>>(&device);
    }

    #[cfg(feature = "backend_ndarray")]
    #[test]
    fn moonvit_block_matches_reference_hook_when_enabled() {
        use burn::tensor::backend::BackendTypes;

        let device = <burn::backend::NdArray<f32> as BackendTypes>::Device::default();
        run_moonvit_block_parity::<burn::backend::NdArray<f32>>(&device);
    }

    #[cfg(feature = "backend_wgpu")]
    #[test]
    fn moonvit_block_matches_reference_hook_wgpu_when_enabled() {
        if std::env::var("BURN_WGPU_CORRECTNESS").is_err() {
            eprintln!("skipping: set BURN_WGPU_CORRECTNESS=1 to run WGPU MoonViT block parity");
            return;
        }
        let device = burn_wgpu::WgpuDevice::default();
        run_moonvit_block_parity::<burn_wgpu::Wgpu<f32, i32, u32>>(&device);
    }

    fn run_patch_embed_parity<B: burn::prelude::Backend>(device: &B::Device) {
        if std::env::var("LOCATE_ANYTHING_PATCH_EMBED_PARITY").is_err() {
            eprintln!(
                "skipping: set LOCATE_ANYTHING_PATCH_EMBED_PARITY=1 with LOCATE_ANYTHING_PATCH_EMBED_WEIGHTS, LOCATE_ANYTHING_PREPROCESS_HOOKS, and LOCATE_ANYTHING_VISION_HOOKS"
            );
            return;
        }
        let weights_path = std::env::var("LOCATE_ANYTHING_PATCH_EMBED_WEIGHTS")
            .expect("LOCATE_ANYTHING_PATCH_EMBED_WEIGHTS");
        let preprocess_path = std::env::var("LOCATE_ANYTHING_PREPROCESS_HOOKS")
            .expect("LOCATE_ANYTHING_PREPROCESS_HOOKS");
        let hooks_path =
            std::env::var("LOCATE_ANYTHING_VISION_HOOKS").expect("LOCATE_ANYTHING_VISION_HOOKS");
        let warmup_iters = env_usize("LOCATE_ANYTHING_PATCH_EMBED_WARMUP_ITERS", 0);
        let cache_position_embeddings = std::env::var("LOCATE_ANYTHING_PATCH_EMBED_CACHE_POS")
            .ok()
            .is_some_and(|value| value != "0");
        let mean_tolerance = env_f32("LOCATE_ANYTHING_PATCH_EMBED_MEAN_TOLERANCE", 1.0e-4);
        let rms_tolerance = env_f32("LOCATE_ANYTHING_PATCH_EMBED_RMS_TOLERANCE", 2.0e-4);
        let max_tolerance = env_f32("LOCATE_ANYTHING_PATCH_EMBED_MAX_TOLERANCE", 2.0e-3);

        let weights = PatchEmbedWeights::from_safetensors_file(weights_path).unwrap();
        let pixel_values =
            load_tensor_from_safetensors_file(&preprocess_path, "pixel_values").unwrap();
        let grid_hws =
            load_tensor_from_safetensors_file(&preprocess_path, "image_grid_hws").unwrap();
        let reference =
            load_tensor_from_safetensors_file(&hooks_path, "vision.patch_embed").unwrap();
        let pixel_shape: [usize; 4] = pixel_values.shape.clone().try_into().unwrap();
        let grid_data = grid_hws
            .data
            .chunks_exact(2)
            .map(|chunk| [chunk[0] as usize, chunk[1] as usize])
            .collect::<Vec<_>>();
        let reference_shape: [usize; 2] = reference.shape.clone().try_into().unwrap();
        assert_eq!(pixel_shape[0], reference_shape[0]);
        assert_eq!(reference_shape[1], weights.out_dim);

        let model = BurnPatchEmbed::<B>::from_weights(weights, device);
        let patches =
            burn::prelude::Tensor::<B, 1>::from_floats(pixel_values.data.as_slice(), device)
                .reshape(pixel_shape);
        let cached_pos = cache_position_embeddings.then(|| model.position_embeddings(&grid_data));
        for _ in 0..warmup_iters {
            let output = if let Some(pos) = cached_pos.as_ref() {
                model.forward_with_position_embeddings(patches.clone(), pos.clone(), &grid_data)
            } else {
                model.forward(patches.clone(), &grid_data)
            };
            let _ = output.into_data();
        }
        let started = Instant::now();
        let output = if let Some(pos) = cached_pos {
            model.forward_with_position_embeddings(patches, pos, &grid_data)
        } else {
            model.forward(patches, &grid_data)
        };
        let output_data = output
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("tensor data");
        let forward_readback_ms = started.elapsed().as_secs_f64() * 1000.0;
        let stats = stats(&output_data, &reference.data);
        eprintln!(
            "patch embed parity tokens={} forward_readback_ms={forward_readback_ms:.3} mean_abs={:.6e} rms={:.6e} max_abs={:.6e}",
            reference_shape[0], stats.mean_abs, stats.rms, stats.max_abs
        );
        assert!(
            stats.mean_abs < mean_tolerance,
            "patch_embed mean_abs {:.6e} exceeded tolerance {:.6e}",
            stats.mean_abs,
            mean_tolerance
        );
        assert!(
            stats.rms < rms_tolerance,
            "patch_embed rms {:.6e} exceeded tolerance {:.6e}",
            stats.rms,
            rms_tolerance
        );
        assert!(
            stats.max_abs < max_tolerance,
            "patch_embed max_abs {:.6e} exceeded tolerance {:.6e}",
            stats.max_abs,
            max_tolerance
        );
    }

    fn run_patch_merger_parity<B: burn::prelude::Backend>(device: &B::Device) {
        if std::env::var("LOCATE_ANYTHING_PATCH_MERGER_PARITY").is_err() {
            eprintln!(
                "skipping: set LOCATE_ANYTHING_PATCH_MERGER_PARITY=1 with LOCATE_ANYTHING_PREPROCESS_HOOKS and LOCATE_ANYTHING_VISION_HOOKS"
            );
            return;
        }
        let preprocess_path = std::env::var("LOCATE_ANYTHING_PREPROCESS_HOOKS")
            .expect("LOCATE_ANYTHING_PREPROCESS_HOOKS");
        let hooks_path =
            std::env::var("LOCATE_ANYTHING_VISION_HOOKS").expect("LOCATE_ANYTHING_VISION_HOOKS");
        let warmup_iters = env_usize("LOCATE_ANYTHING_PATCH_MERGER_WARMUP_ITERS", 0);
        let mean_tolerance = env_f32("LOCATE_ANYTHING_PATCH_MERGER_MEAN_TOLERANCE", 1.0e-7);
        let rms_tolerance = env_f32("LOCATE_ANYTHING_PATCH_MERGER_RMS_TOLERANCE", 1.0e-7);
        let max_tolerance = env_f32("LOCATE_ANYTHING_PATCH_MERGER_MAX_TOLERANCE", 1.0e-6);

        let grid_hws =
            load_tensor_from_safetensors_file(&preprocess_path, "image_grid_hws").unwrap();
        let grid_data = grid_hws
            .data
            .chunks_exact(2)
            .map(|chunk| [chunk[0] as usize, chunk[1] as usize])
            .collect::<Vec<_>>();
        let input =
            load_tensor_from_safetensors_file(&hooks_path, "vision.final_layernorm").unwrap();
        let reference =
            load_tensor_from_safetensors_file(&hooks_path, "vision.merged_tokens").unwrap();
        let input_shape: [usize; 2] = input.shape.clone().try_into().unwrap();
        let reference_shape: [usize; 2] = reference.shape.clone().try_into().unwrap();
        assert_eq!(reference_shape[1], input_shape[1] * 4);

        let hidden = burn::prelude::Tensor::<B, 1>::from_floats(input.data.as_slice(), device)
            .reshape(input_shape);
        for _ in 0..warmup_iters {
            let _ =
                super::burn_vision::patch_merger(hidden.clone(), &grid_data, [2, 2]).into_data();
        }
        let started = Instant::now();
        let output = super::burn_vision::patch_merger(hidden, &grid_data, [2, 2]);
        let output_data = output
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("tensor data");
        let forward_readback_ms = started.elapsed().as_secs_f64() * 1000.0;
        let stats = stats(&output_data, &reference.data);
        eprintln!(
            "patch merger parity tokens={} forward_readback_ms={forward_readback_ms:.3} mean_abs={:.6e} rms={:.6e} max_abs={:.6e}",
            reference_shape[0], stats.mean_abs, stats.rms, stats.max_abs
        );
        assert!(
            stats.mean_abs < mean_tolerance,
            "patch_merger mean_abs {:.6e} exceeded tolerance {:.6e}",
            stats.mean_abs,
            mean_tolerance
        );
        assert!(
            stats.rms < rms_tolerance,
            "patch_merger rms {:.6e} exceeded tolerance {:.6e}",
            stats.rms,
            rms_tolerance
        );
        assert!(
            stats.max_abs < max_tolerance,
            "patch_merger max_abs {:.6e} exceeded tolerance {:.6e}",
            stats.max_abs,
            max_tolerance
        );
    }

    fn run_moonvit_block_parity<B: burn::prelude::Backend>(device: &B::Device) {
        if std::env::var("LOCATE_ANYTHING_MOONVIT_BLOCK_PARITY").is_err() {
            eprintln!(
                "skipping: set LOCATE_ANYTHING_MOONVIT_BLOCK_PARITY=1 with LOCATE_ANYTHING_MOONVIT_BLOCK_WEIGHTS and LOCATE_ANYTHING_MOONVIT_BLOCK_HOOKS"
            );
            return;
        }
        let weights_path = std::env::var("LOCATE_ANYTHING_MOONVIT_BLOCK_WEIGHTS")
            .expect("LOCATE_ANYTHING_MOONVIT_BLOCK_WEIGHTS");
        let hooks_path = std::env::var("LOCATE_ANYTHING_MOONVIT_BLOCK_HOOKS")
            .expect("LOCATE_ANYTHING_MOONVIT_BLOCK_HOOKS");
        let layer_index = env_usize("LOCATE_ANYTHING_MOONVIT_BLOCK_LAYER", 0);
        let warmup_iters = env_usize("LOCATE_ANYTHING_MOONVIT_BLOCK_WARMUP_ITERS", 0);
        let mean_tolerance = env_f32("LOCATE_ANYTHING_MOONVIT_BLOCK_MEAN_TOLERANCE", 5.0e-4);
        let rms_tolerance = env_f32("LOCATE_ANYTHING_MOONVIT_BLOCK_RMS_TOLERANCE", 1.0e-3);
        let max_tolerance = env_f32("LOCATE_ANYTHING_MOONVIT_BLOCK_MAX_TOLERANCE", 5.0e-2);

        let weights =
            MoonVitEncoderBlockWeights::from_safetensors_file(weights_path, layer_index).unwrap();
        let input = load_tensor_from_safetensors_file(&hooks_path, "vision.block_input").unwrap();
        let reference_key = format!("vision.block_{layer_index:02}");
        let reference = load_tensor_from_safetensors_file(&hooks_path, &reference_key).unwrap();
        let grid_hws = load_tensor_from_safetensors_file(&hooks_path, "image_grid_hws").unwrap();
        let input_shape: [usize; 2] = input.shape.clone().try_into().unwrap();
        let reference_shape: [usize; 2] = reference.shape.clone().try_into().unwrap();
        assert_eq!(input_shape, reference_shape);
        let grid_data = grid_hws
            .data
            .chunks_exact(2)
            .map(|chunk| [chunk[0] as usize, chunk[1] as usize])
            .collect::<Vec<_>>();
        let model = BurnMoonVitEncoderBlock::<B>::from_weights(weights, device);
        let hidden = burn::prelude::Tensor::<B, 1>::from_floats(input.data.as_slice(), device)
            .reshape(input_shape);
        for _ in 0..warmup_iters {
            let _ = model.forward(hidden.clone(), &grid_data).into_data();
        }
        let started = Instant::now();
        let output = model.forward(hidden, &grid_data);
        let output_data = output
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("tensor data");
        let forward_readback_ms = started.elapsed().as_secs_f64() * 1000.0;
        let stats = stats(&output_data, &reference.data);
        eprintln!(
            "MoonViT block parity layer={layer_index} tokens={} forward_readback_ms={forward_readback_ms:.3} mean_abs={:.6e} rms={:.6e} max_abs={:.6e}",
            reference_shape[0], stats.mean_abs, stats.rms, stats.max_abs
        );
        assert!(
            stats.mean_abs < mean_tolerance,
            "MoonViT block mean_abs {:.6e} exceeded tolerance {:.6e}",
            stats.mean_abs,
            mean_tolerance
        );
        assert!(
            stats.rms < rms_tolerance,
            "MoonViT block rms {:.6e} exceeded tolerance {:.6e}",
            stats.rms,
            rms_tolerance
        );
        assert!(
            stats.max_abs < max_tolerance,
            "MoonViT block max_abs {:.6e} exceeded tolerance {:.6e}",
            stats.max_abs,
            max_tolerance
        );
    }

    #[derive(Clone, Copy, Debug)]
    struct Stats {
        mean_abs: f32,
        rms: f32,
        max_abs: f32,
    }

    fn stats(lhs: &[f32], rhs: &[f32]) -> Stats {
        assert_eq!(lhs.len(), rhs.len());
        let mut sum_abs = 0.0;
        let mut sum_sq = 0.0;
        let mut max_abs = 0.0f32;
        for (&left, &right) in lhs.iter().zip(rhs.iter()) {
            let diff = left - right;
            let abs = diff.abs();
            sum_abs += abs;
            sum_sq += diff * diff;
            max_abs = max_abs.max(abs);
        }
        let len = lhs.len().max(1) as f32;
        Stats {
            mean_abs: sum_abs / len,
            rms: (sum_sq / len).sqrt(),
            max_abs,
        }
    }

    fn env_f32(name: &str, default: f32) -> f32 {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse::<f32>().ok())
            .unwrap_or(default)
    }

    fn env_usize(name: &str, default: usize) -> usize {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(default)
    }
}

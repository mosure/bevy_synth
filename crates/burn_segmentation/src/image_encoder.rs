use std::path::Path;

#[cfg(feature = "bpk")]
use crate::tensor_io::load_all_tensors_from_burnpack_file;
use crate::tensor_io::{LoadedTensorF32, load_all_tensors_from_safetensors_file};
use crate::{SegmentationError, SegmentationResult};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SamImageEncoderVariant {
    Sam2_1HieraTiny,
    Sam2_1HieraSmall,
    Sam2_1HieraBasePlus,
    Sam2_1HieraLarge,
}

impl SamImageEncoderVariant {
    pub fn label(self) -> &'static str {
        match self {
            Self::Sam2_1HieraTiny => "sam2.1-hiera-tiny",
            Self::Sam2_1HieraSmall => "sam2.1-hiera-small",
            Self::Sam2_1HieraBasePlus => "sam2.1-hiera-base-plus",
            Self::Sam2_1HieraLarge => "sam2.1-hiera-large",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SamImageEncoderConfig {
    pub variant: SamImageEncoderVariant,
    pub image_size: usize,
    pub embed_dim: usize,
    pub initial_heads: usize,
    pub output_dim: usize,
    pub stages: [usize; 4],
    pub global_att_blocks: [usize; 3],
    pub window_spec: [usize; 4],
    pub window_pos_spatial_size: [usize; 2],
    pub q_pool_stages: usize,
}

impl SamImageEncoderConfig {
    pub fn sam2_1_hiera_tiny() -> Self {
        Self {
            variant: SamImageEncoderVariant::Sam2_1HieraTiny,
            image_size: 1024,
            embed_dim: 96,
            initial_heads: 1,
            output_dim: 256,
            stages: [1, 2, 7, 2],
            global_att_blocks: [5, 7, 9],
            window_spec: [8, 4, 14, 7],
            window_pos_spatial_size: [7, 7],
            q_pool_stages: 3,
        }
    }

    pub fn sam2_1_hiera_small() -> Self {
        Self {
            variant: SamImageEncoderVariant::Sam2_1HieraSmall,
            image_size: 1024,
            embed_dim: 96,
            initial_heads: 1,
            output_dim: 256,
            stages: [1, 2, 11, 2],
            global_att_blocks: [7, 10, 13],
            window_spec: [8, 4, 14, 7],
            window_pos_spatial_size: [7, 7],
            q_pool_stages: 3,
        }
    }

    pub fn sam2_1_hiera_base_plus() -> Self {
        Self {
            variant: SamImageEncoderVariant::Sam2_1HieraBasePlus,
            image_size: 1024,
            embed_dim: 112,
            initial_heads: 2,
            output_dim: 256,
            stages: [2, 3, 16, 3],
            global_att_blocks: [12, 16, 20],
            window_spec: [8, 4, 14, 7],
            window_pos_spatial_size: [14, 14],
            q_pool_stages: 3,
        }
    }

    pub fn sam2_1_hiera_large() -> Self {
        Self {
            variant: SamImageEncoderVariant::Sam2_1HieraLarge,
            image_size: 1024,
            embed_dim: 144,
            initial_heads: 2,
            output_dim: 256,
            stages: [2, 6, 36, 4],
            global_att_blocks: [23, 33, 43],
            window_spec: [8, 4, 16, 8],
            window_pos_spatial_size: [7, 7],
            q_pool_stages: 3,
        }
    }

    pub fn depth(&self) -> usize {
        self.stages.iter().sum()
    }

    pub fn stage_ends(&self) -> [usize; 4] {
        [
            self.stages[0] - 1,
            self.stages[0] + self.stages[1] - 1,
            self.stages[0] + self.stages[1] + self.stages[2] - 1,
            self.depth() - 1,
        ]
    }

    fn q_pool_blocks(&self) -> Vec<usize> {
        self.stage_ends()
            .into_iter()
            .take(3)
            .map(|end| end + 1)
            .take(self.q_pool_stages)
            .collect()
    }

    fn block_specs(&self) -> Vec<SamImageBlockSpec> {
        let depth = self.depth();
        let stage_ends = self.stage_ends();
        let q_pool_blocks = self.q_pool_blocks();
        let mut specs = Vec::with_capacity(depth);
        let mut cur_stage = 1;
        let mut dim = self.embed_dim;
        let mut heads = self.initial_heads;
        for index in 0..depth {
            let mut dim_out = dim;
            let window_size = if self.global_att_blocks.contains(&index) {
                0
            } else {
                self.window_spec[cur_stage - 1]
            };
            if index
                .checked_sub(1)
                .is_some_and(|previous| stage_ends.contains(&previous))
            {
                dim_out = dim * 2;
                heads *= 2;
                cur_stage += 1;
            }
            specs.push(SamImageBlockSpec {
                dim,
                dim_out,
                heads,
                window_size,
                q_pool: q_pool_blocks.contains(&index),
            });
            dim = dim_out;
        }
        specs
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SamImageBlockSpec {
    dim: usize,
    dim_out: usize,
    heads: usize,
    window_size: usize,
    q_pool: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SamImageEncoderWeights {
    pub config: SamImageEncoderConfig,
    pub tensors: Vec<(String, LoadedTensorF32)>,
}

impl SamImageEncoderWeights {
    pub fn from_safetensors_file(path: impl AsRef<Path>) -> SegmentationResult<Self> {
        let tensors = load_all_tensors_from_safetensors_file(path.as_ref())?;
        let config = SamImageEncoderConfig::detect_from_tensors(&tensors)?;
        let weights = Self { config, tensors };
        weights.validate()?;
        Ok(weights)
    }

    #[cfg(feature = "bpk")]
    pub fn from_burnpack_file(path: impl AsRef<Path>) -> SegmentationResult<Self> {
        let tensors = load_all_tensors_from_burnpack_file(path.as_ref())?;
        let config = SamImageEncoderConfig::detect_from_tensors(&tensors)?;
        let weights = Self { config, tensors };
        weights.validate()?;
        Ok(weights)
    }

    fn validate(&self) -> SegmentationResult<()> {
        let config = &self.config;
        expect_shape(
            &self.tensors,
            "image_encoder.trunk.patch_embed.proj.weight",
            &[config.embed_dim, 3, 7, 7],
        )?;
        expect_shape(
            &self.tensors,
            "image_encoder.trunk.pos_embed",
            &[
                1,
                config.embed_dim,
                config.window_pos_spatial_size[0],
                config.window_pos_spatial_size[1],
            ],
        )?;
        expect_shape(
            &self.tensors,
            "image_encoder.trunk.pos_embed_window",
            &[
                1,
                config.embed_dim,
                config.window_spec[0],
                config.window_spec[0],
            ],
        )?;
        expect_shape(&self.tensors, "no_mem_embed", &[1, 1, config.output_dim])?;
        let specs = config.block_specs();
        for (index, spec) in specs.iter().enumerate() {
            expect_shape(
                &self.tensors,
                &format!("image_encoder.trunk.blocks.{index}.norm1.weight"),
                &[spec.dim],
            )?;
            expect_shape(
                &self.tensors,
                &format!("image_encoder.trunk.blocks.{index}.attn.qkv.weight"),
                &[spec.dim_out * 3, spec.dim],
            )?;
        }
        if find_tensor(
            &self.tensors,
            &format!("image_encoder.trunk.blocks.{}.norm1.weight", specs.len()),
        )
        .is_ok()
        {
            return Err(SegmentationError::Image(format!(
                "{} expected {} Hiera blocks but found additional block {}",
                config.variant.label(),
                specs.len(),
                specs.len()
            )));
        }
        Ok(())
    }
}

impl SamImageEncoderConfig {
    fn detect_from_tensors(tensors: &[(String, LoadedTensorF32)]) -> SegmentationResult<Self> {
        let patch = find_tensor(tensors, "image_encoder.trunk.patch_embed.proj.weight")?;
        let embed_dim =
            patch.shape.first().copied().ok_or_else(|| {
                SegmentationError::Image("empty patch embedding shape".to_string())
            })?;
        let depth = count_hiera_blocks(tensors);
        match (embed_dim, depth) {
            (96, 12) => Ok(Self::sam2_1_hiera_tiny()),
            (96, 16) => Ok(Self::sam2_1_hiera_small()),
            (112, 24) => Ok(Self::sam2_1_hiera_base_plus()),
            (144, 48) => Ok(Self::sam2_1_hiera_large()),
            _ => Err(SegmentationError::Image(format!(
                "unsupported SAM2.1 Hiera image encoder shape: embed_dim={embed_dim}, depth={depth}"
            ))),
        }
    }
}

fn count_hiera_blocks(tensors: &[(String, LoadedTensorF32)]) -> usize {
    tensors
        .iter()
        .filter_map(|(name, _)| {
            let rest = name.strip_prefix("image_encoder.trunk.blocks.")?;
            let (index, suffix) = rest.split_once('.')?;
            (suffix == "norm1.weight")
                .then(|| index.parse::<usize>().ok())
                .flatten()
        })
        .max()
        .map_or(0, |max_index| max_index + 1)
}

fn expect_shape(
    tensors: &[(String, LoadedTensorF32)],
    key: &str,
    expected: &[usize],
) -> SegmentationResult<()> {
    let actual = find_tensor(tensors, key)?.shape.as_slice();
    if actual != expected {
        return Err(SegmentationError::Image(format!(
            "{key} expected shape {expected:?}, got {actual:?}"
        )));
    }
    Ok(())
}

fn find_tensor<'a>(
    tensors: &'a [(String, LoadedTensorF32)],
    key: &str,
) -> SegmentationResult<&'a LoadedTensorF32> {
    tensors
        .iter()
        .find_map(|(name, tensor)| (name == key).then_some(tensor))
        .ok_or_else(|| SegmentationError::Image(format!("missing tensor `{key}`")))
}

#[cfg(any(feature = "backend_wgpu", feature = "backend_cuda"))]
pub mod burn_image_encoder {
    use std::collections::HashMap;

    use burn::prelude::*;
    use burn::tensor::activation::{gelu, softmax};
    use burn::tensor::module::{conv2d, interpolate, max_pool2d};
    use burn::tensor::ops::{ConvOptions, InterpolateMode, InterpolateOptions, PadMode};

    use super::*;

    #[derive(Debug)]
    pub struct BurnSamImageEncoder<B: Backend> {
        pub weights: SamImageEncoderWeights,
        tensors1: HashMap<String, Tensor<B, 1>>,
        tensors2: HashMap<String, Tensor<B, 2>>,
        tensors3: HashMap<String, Tensor<B, 3>>,
        tensors4: HashMap<String, Tensor<B, 4>>,
    }

    #[derive(Debug)]
    pub struct BurnSamImageEncoderOutput<B: Backend> {
        pub trunk_features: Vec<Tensor<B, 4>>,
        pub neck_features: Vec<Tensor<B, 4>>,
        pub high_res_features_raw: [Tensor<B, 4>; 2],
        pub image_embed: Tensor<B, 4>,
    }

    impl<B: Backend> BurnSamImageEncoder<B> {
        pub fn from_weights(weights: SamImageEncoderWeights, device: &B::Device) -> Self {
            let mut tensors1 = HashMap::new();
            let mut tensors2 = HashMap::new();
            let mut tensors3 = HashMap::new();
            let mut tensors4 = HashMap::new();
            for (name, tensor) in &weights.tensors {
                match tensor.shape.as_slice() {
                    [a] => {
                        tensors1.insert(
                            name.clone(),
                            Tensor::<B, 1>::from_floats(tensor.data.as_slice(), device)
                                .reshape([*a]),
                        );
                    }
                    [a, b] => {
                        tensors2.insert(
                            name.clone(),
                            Tensor::<B, 1>::from_floats(tensor.data.as_slice(), device)
                                .reshape([*a, *b]),
                        );
                    }
                    [a, b, c] => {
                        tensors3.insert(
                            name.clone(),
                            Tensor::<B, 1>::from_floats(tensor.data.as_slice(), device)
                                .reshape([*a, *b, *c]),
                        );
                    }
                    [a, b, c, d] => {
                        tensors4.insert(
                            name.clone(),
                            Tensor::<B, 1>::from_floats(tensor.data.as_slice(), device)
                                .reshape([*a, *b, *c, *d]),
                        );
                    }
                    _ => {}
                }
            }
            Self {
                weights,
                tensors1,
                tensors2,
                tensors3,
                tensors4,
            }
        }

        pub fn forward(&self, input: Tensor<B, 4>) -> BurnSamImageEncoderOutput<B> {
            let trunk_features = self.forward_trunk(input);
            let neck_features = self.forward_neck(&trunk_features);
            let image_embed = neck_features[2].clone() + self.no_mem_embed_nchw();
            BurnSamImageEncoderOutput {
                trunk_features,
                high_res_features_raw: [neck_features[0].clone(), neck_features[1].clone()],
                image_embed,
                neck_features,
            }
        }

        fn forward_trunk(&self, input: Tensor<B, 4>) -> Vec<Tensor<B, 4>> {
            let mut x = self.patch_embed(input);
            let [batch, height, width, channels] = x.dims();
            assert_eq!(
                [height, width, channels],
                [256, 256, self.weights.config.embed_dim]
            );
            x = x + self.pos_embed(height, width);
            let mut outputs = Vec::with_capacity(4);
            let stage_ends = self.weights.config.stage_ends();
            for (index, spec) in self.weights.config.block_specs().iter().enumerate() {
                x = self.block(index, *spec, x);
                if stage_ends.contains(&index) {
                    let [_b, h, w, c] = x.dims();
                    outputs.push(x.clone().permute([0, 3, 1, 2]).reshape([batch, c, h, w]));
                }
            }
            outputs
        }

        fn patch_embed(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
            conv2d(
                input,
                self.t4("image_encoder.trunk.patch_embed.proj.weight"),
                Some(self.t1("image_encoder.trunk.patch_embed.proj.bias")),
                ConvOptions::new([4, 4], [3, 3], [1, 1], 1),
            )
            .permute([0, 2, 3, 1])
        }

        fn pos_embed(&self, height: usize, width: usize) -> Tensor<B, 4> {
            let pos = interpolate(
                self.t4("image_encoder.trunk.pos_embed"),
                [height, width],
                InterpolateOptions::new(InterpolateMode::Bicubic).with_align_corners(false),
            );
            let window = self
                .t4("image_encoder.trunk.pos_embed_window")
                .reshape([
                    1,
                    self.weights.config.embed_dim,
                    1,
                    self.weights.config.window_spec[0],
                    1,
                    self.weights.config.window_spec[0],
                ])
                .repeat_dim(2, height / self.weights.config.window_spec[0])
                .repeat_dim(4, width / self.weights.config.window_spec[0])
                .reshape([1, self.weights.config.embed_dim, height, width]);
            (pos + window).permute([0, 2, 3, 1])
        }

        fn block(
            &self,
            index: usize,
            spec: SamImageBlockSpec,
            input: Tensor<B, 4>,
        ) -> Tensor<B, 4> {
            let prefix = format!("image_encoder.trunk.blocks.{index}");
            let shortcut = input.clone();
            let x_norm = self.layer_norm_bhwc(input, &format!("{prefix}.norm1"), spec.dim);
            let shortcut = if spec.dim != spec.dim_out {
                self.do_pool_bhwc(
                    self.linear_bhwc(x_norm.clone(), &format!("{prefix}.proj"), spec.dim_out),
                    spec.q_pool,
                )
            } else {
                shortcut
            };

            let mut window_size = spec.window_size;
            let (attn_input, pad_hw, original_hw) = if window_size > 0 {
                let [_b, h, w, _c] = x_norm.dims();
                let (windows, pad_hw) = self.window_partition(x_norm, window_size);
                (windows, pad_hw, [h, w])
            } else {
                let [_b, h, w, _c] = x_norm.dims();
                (x_norm, [h, w], [h, w])
            };
            let mut attn = self.multi_scale_attention(
                attn_input,
                &format!("{prefix}.attn"),
                spec.dim_out,
                spec.heads,
                spec.q_pool,
            );
            if spec.q_pool {
                window_size /= 2;
            }
            if spec.window_size > 0 {
                let [_b, h, w, _c] = shortcut.dims();
                let unpartition_pad_hw = if spec.q_pool {
                    let pad_h = (window_size - h % window_size) % window_size;
                    let pad_w = (window_size - w % window_size) % window_size;
                    [h + pad_h, w + pad_w]
                } else {
                    pad_hw
                };
                let target_hw = if spec.q_pool { [h, w] } else { original_hw };
                attn = self.window_unpartition(attn, window_size, unpartition_pad_hw, target_hw);
            }

            let x = shortcut + attn;
            let mlp_input =
                self.layer_norm_bhwc(x.clone(), &format!("{prefix}.norm2"), spec.dim_out);
            x + self.mlp_bhwc(
                mlp_input,
                &format!("{prefix}.mlp"),
                spec.dim_out,
                spec.dim_out * 4,
            )
        }

        fn multi_scale_attention(
            &self,
            input: Tensor<B, 4>,
            prefix: &str,
            dim_out: usize,
            heads: usize,
            q_pool: bool,
        ) -> Tensor<B, 4> {
            let [batch, height, width, _dim] = input.dims();
            let head_dim = dim_out / heads;
            let tokens = height * width;
            let qkv = self
                .linear_bhwc(input, &format!("{prefix}.qkv"), dim_out * 3)
                .reshape([batch, tokens, 3, heads, head_dim]);
            let mut q = qkv
                .clone()
                .slice([0..batch, 0..tokens, 0..1, 0..heads, 0..head_dim])
                .reshape([batch, tokens, heads, head_dim]);
            let k = qkv
                .clone()
                .slice([0..batch, 0..tokens, 1..2, 0..heads, 0..head_dim])
                .reshape([batch, tokens, heads, head_dim]);
            let v = qkv
                .slice([0..batch, 0..tokens, 2..3, 0..heads, 0..head_dim])
                .reshape([batch, tokens, heads, head_dim]);
            let mut q_h = height;
            let mut q_w = width;
            if q_pool {
                let pooled = self.do_pool_bhwc(q.reshape([batch, height, width, dim_out]), true);
                let [_b, h, w, _c] = pooled.dims();
                q_h = h;
                q_w = w;
                q = pooled.reshape([batch, h * w, heads, head_dim]);
            }
            let q_tokens = q_h * q_w;
            let q = q.permute([0, 2, 1, 3]);
            let k = k.permute([0, 2, 1, 3]);
            let v = v.permute([0, 2, 1, 3]);
            let scores = q
                .matmul(k.swap_dims(2, 3))
                .mul_scalar(1.0 / (head_dim as f64).sqrt());
            let out = softmax(scores, 3)
                .matmul(v)
                .permute([0, 2, 1, 3])
                .reshape([batch, q_tokens, dim_out])
                .reshape([batch, q_h, q_w, dim_out]);
            self.linear_bhwc(out, &format!("{prefix}.proj"), dim_out)
        }

        fn forward_neck(&self, trunk: &[Tensor<B, 4>]) -> Vec<Tensor<B, 4>> {
            assert_eq!(trunk.len(), 4);
            let mut out: Vec<Option<Tensor<B, 4>>> = vec![None, None, None, None];
            let mut prev: Option<Tensor<B, 4>> = None;
            for i in (0..4).rev() {
                let conv_index = 3 - i;
                let lateral = conv2d(
                    trunk[i].clone(),
                    self.t4(&format!(
                        "image_encoder.neck.convs.{conv_index}.conv.weight"
                    )),
                    Some(self.t1(&format!("image_encoder.neck.convs.{conv_index}.conv.bias"))),
                    ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
                );
                let next = if i == 2 || i == 3 {
                    if let Some(top) = prev {
                        let [_b, _c, h, w] = lateral.dims();
                        lateral + self.interpolate_nearest(top, [h, w])
                    } else {
                        lateral
                    }
                } else {
                    lateral
                };
                prev = Some(next.clone());
                out[i] = Some(next);
            }
            out.into_iter().map(Option::unwrap).collect()
        }

        fn interpolate_nearest(&self, input: Tensor<B, 4>, size: [usize; 2]) -> Tensor<B, 4> {
            interpolate(
                input,
                size,
                InterpolateOptions::new(InterpolateMode::Nearest),
            )
        }

        fn do_pool_bhwc(&self, input: Tensor<B, 4>, enabled: bool) -> Tensor<B, 4> {
            if !enabled {
                return input;
            }
            max_pool2d(
                input.permute([0, 3, 1, 2]),
                [2, 2],
                [2, 2],
                [0, 0],
                [1, 1],
                false,
            )
            .permute([0, 2, 3, 1])
        }

        fn window_partition(
            &self,
            input: Tensor<B, 4>,
            window_size: usize,
        ) -> (Tensor<B, 4>, [usize; 2]) {
            let [batch, height, width, channels] = input.dims();
            let pad_h = (window_size - height % window_size) % window_size;
            let pad_w = (window_size - width % window_size) % window_size;
            let padded = if pad_h > 0 || pad_w > 0 {
                input
                    .permute([0, 3, 1, 2])
                    .pad((0, pad_w, 0, pad_h), PadMode::Constant(0.0))
                    .permute([0, 2, 3, 1])
            } else {
                input
            };
            let hp = height + pad_h;
            let wp = width + pad_w;
            let windows = padded
                .reshape([
                    batch,
                    hp / window_size,
                    window_size,
                    wp / window_size,
                    window_size,
                    channels,
                ])
                .permute([0, 1, 3, 2, 4, 5])
                .reshape([
                    batch * (hp / window_size) * (wp / window_size),
                    window_size,
                    window_size,
                    channels,
                ]);
            (windows, [hp, wp])
        }

        fn window_unpartition(
            &self,
            windows: Tensor<B, 4>,
            window_size: usize,
            pad_hw: [usize; 2],
            hw: [usize; 2],
        ) -> Tensor<B, 4> {
            let [num_windows, _wh, _ww, channels] = windows.dims();
            let [hp, wp] = pad_hw;
            let [height, width] = hw;
            let windows_per_image = (hp / window_size) * (wp / window_size);
            let batch = num_windows / windows_per_image;
            let x = windows
                .reshape([
                    batch,
                    hp / window_size,
                    wp / window_size,
                    window_size,
                    window_size,
                    channels,
                ])
                .permute([0, 1, 3, 2, 4, 5])
                .reshape([batch, hp, wp, channels]);
            if hp > height || wp > width {
                x.slice([0..batch, 0..height, 0..width, 0..channels])
            } else {
                x
            }
        }

        fn mlp_bhwc(
            &self,
            input: Tensor<B, 4>,
            prefix: &str,
            output_dim: usize,
            hidden_dim: usize,
        ) -> Tensor<B, 4> {
            let hidden = self.linear_bhwc(input, &format!("{prefix}.layers.0"), hidden_dim);
            let hidden = gelu(hidden);
            self.linear_bhwc(hidden, &format!("{prefix}.layers.1"), output_dim)
        }

        fn linear_bhwc(
            &self,
            input: Tensor<B, 4>,
            prefix: &str,
            output_dim: usize,
        ) -> Tensor<B, 4> {
            let [batch, height, width, input_dim] = input.dims();
            (input
                .reshape([batch * height * width, input_dim])
                .matmul(self.t2(&format!("{prefix}.weight")).swap_dims(0, 1))
                + self.t1(&format!("{prefix}.bias")).reshape([1, output_dim]))
            .reshape([batch, height, width, output_dim])
        }

        fn layer_norm_bhwc(
            &self,
            input: Tensor<B, 4>,
            prefix: &str,
            channels: usize,
        ) -> Tensor<B, 4> {
            let (var, mean) = input.clone().var_mean_bias(3);
            (input - mean) / var.add_scalar(1.0e-6).sqrt()
                * self
                    .t1(&format!("{prefix}.weight"))
                    .reshape([1, 1, 1, channels])
                + self
                    .t1(&format!("{prefix}.bias"))
                    .reshape([1, 1, 1, channels])
        }

        fn no_mem_embed_nchw(&self) -> Tensor<B, 4> {
            self.t3("no_mem_embed")
                .reshape([1, 1, 256])
                .swap_dims(1, 2)
                .reshape([1, 256, 1, 1])
        }

        fn t1(&self, key: &str) -> Tensor<B, 1> {
            self.tensors1
                .get(key)
                .unwrap_or_else(|| panic!("missing rank-1 tensor `{key}`"))
                .clone()
        }

        fn t2(&self, key: &str) -> Tensor<B, 2> {
            self.tensors2
                .get(key)
                .unwrap_or_else(|| panic!("missing rank-2 tensor `{key}`"))
                .clone()
        }

        fn t3(&self, key: &str) -> Tensor<B, 3> {
            self.tensors3
                .get(key)
                .unwrap_or_else(|| panic!("missing rank-3 tensor `{key}`"))
                .clone()
        }

        fn t4(&self, key: &str) -> Tensor<B, 4> {
            self.tensors4
                .get(key)
                .unwrap_or_else(|| panic!("missing rank-4 tensor `{key}`"))
                .clone()
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "backend_wgpu")]
    use burn::prelude::*;

    use super::*;

    #[test]
    fn image_encoder_rejects_missing_required_weights() {
        let weights = SamImageEncoderWeights {
            config: SamImageEncoderConfig::sam2_1_hiera_tiny(),
            tensors: Vec::new(),
        };
        assert!(weights.validate().is_err());
    }

    #[test]
    fn sam2_hiera_variant_block_specs_match_upstream_configs() {
        let variants = [
            (
                SamImageEncoderConfig::sam2_1_hiera_tiny(),
                SamImageEncoderVariant::Sam2_1HieraTiny,
                [0, 2, 9, 11],
                [1, 2, 7, 2],
            ),
            (
                SamImageEncoderConfig::sam2_1_hiera_small(),
                SamImageEncoderVariant::Sam2_1HieraSmall,
                [0, 2, 13, 15],
                [1, 2, 11, 2],
            ),
            (
                SamImageEncoderConfig::sam2_1_hiera_base_plus(),
                SamImageEncoderVariant::Sam2_1HieraBasePlus,
                [1, 4, 20, 23],
                [2, 3, 16, 3],
            ),
            (
                SamImageEncoderConfig::sam2_1_hiera_large(),
                SamImageEncoderVariant::Sam2_1HieraLarge,
                [1, 7, 43, 47],
                [2, 6, 36, 4],
            ),
        ];
        for (config, variant, stage_ends, stages) in variants {
            assert_eq!(config.variant, variant);
            assert_eq!(config.stages, stages);
            assert_eq!(config.stage_ends(), stage_ends);
            let specs = config.block_specs();
            assert_eq!(specs.len(), config.depth());
            assert_eq!(specs[0].dim, config.embed_dim);
            assert_eq!(specs[0].heads, config.initial_heads);
            assert_eq!(specs[0].window_size, config.window_spec[0]);
            for global in config.global_att_blocks {
                assert_eq!(specs[global].window_size, 0);
            }
            for block in config.q_pool_blocks() {
                assert!(specs[block].q_pool);
            }
            assert_eq!(
                specs[*stage_ends.last().unwrap()].dim_out,
                config.embed_dim * 8
            );
        }
    }

    #[cfg(feature = "backend_wgpu")]
    #[test]
    fn sam2_image_encoder_wgpu_matches_reference_hook() {
        use burn_image_encoder::BurnSamImageEncoder;

        let weights_path = std::env::var("SAM2_IMAGE_ENCODER_WEIGHTS").unwrap_or_default();
        let reference_path = std::env::var("SAM2_REFERENCE_HOOK").unwrap_or_default();
        if weights_path.is_empty() || reference_path.is_empty() {
            eprintln!(
                "skipping: set SAM2_IMAGE_ENCODER_WEIGHTS and SAM2_REFERENCE_HOOK to run WGPU SAM2 image encoder parity"
            );
            return;
        }

        type B = burn_wgpu::Wgpu<f32, i32, u32>;
        let device = burn_wgpu::WgpuDevice::default();
        let weights = SamImageEncoderWeights::from_safetensors_file(&weights_path).unwrap();
        let config = weights.config.clone();
        if let Ok(expected_variant) = std::env::var("SAM2_IMAGE_ENCODER_VARIANT") {
            assert_eq!(config.variant.label(), expected_variant);
        }
        eprintln!("sam2_image_encoder variant={}", config.variant.label());
        let encoder = BurnSamImageEncoder::<B>::from_weights(weights, &device);
        let reference = crate::tensor_io::load_required_tensors_from_safetensors_file(
            Path::new(&reference_path),
            &[
                "image_encoder_input",
                "trunk_feat0",
                "trunk_feat1",
                "trunk_feat2",
                "trunk_feat3",
                "neck_feature0",
                "neck_feature1",
                "neck_feature2",
                "neck_feature3",
                "hook_image_embed",
            ],
        )
        .unwrap();
        let input = tensor_from_reference::<B>(&reference, "image_encoder_input", &device)
            .reshape([1, 3, 1024, 1024]);
        let output = encoder.forward(input);
        compare_tensor4(
            "trunk_feat0",
            output.trunk_features[0].clone(),
            &reference,
            [1, config.embed_dim, 256, 256],
            2.0e-3,
            2.0e-4,
        );
        compare_tensor4(
            "trunk_feat1",
            output.trunk_features[1].clone(),
            &reference,
            [1, config.embed_dim * 2, 128, 128],
            3.0e-3,
            3.0e-4,
        );
        compare_tensor4(
            "trunk_feat2",
            output.trunk_features[2].clone(),
            &reference,
            [1, config.embed_dim * 4, 64, 64],
            5.0e-3,
            5.0e-4,
        );
        compare_tensor4(
            "trunk_feat3",
            output.trunk_features[3].clone(),
            &reference,
            [1, config.embed_dim * 8, 32, 32],
            8.0e-3,
            8.0e-4,
        );
        compare_tensor4(
            "neck_feature0",
            output.neck_features[0].clone(),
            &reference,
            [1, 256, 256, 256],
            1.0e-2,
            1.0e-3,
        );
        compare_tensor4(
            "neck_feature1",
            output.neck_features[1].clone(),
            &reference,
            [1, 256, 128, 128],
            1.0e-2,
            1.0e-3,
        );
        compare_tensor4(
            "neck_feature2",
            output.neck_features[2].clone(),
            &reference,
            [1, 256, 64, 64],
            1.0e-2,
            1.0e-3,
        );
        compare_tensor4(
            "neck_feature3",
            output.neck_features[3].clone(),
            &reference,
            [1, 256, 32, 32],
            1.0e-2,
            1.0e-3,
        );
        compare_tensor4(
            "hook_image_embed",
            output.image_embed,
            &reference,
            [1, 256, 64, 64],
            1.0e-2,
            1.0e-3,
        );
    }

    #[cfg(feature = "backend_wgpu")]
    fn tensor_from_reference<B: burn::prelude::Backend>(
        tensors: &[(String, LoadedTensorF32)],
        key: &str,
        device: &B::Device,
    ) -> Tensor<B, 1> {
        let tensor = crate::tensor_io::find_tensor(tensors, key).unwrap();
        Tensor::<B, 1>::from_floats(tensor.data.as_slice(), device)
    }

    #[cfg(feature = "backend_wgpu")]
    fn compare_tensor4<B: burn::prelude::Backend>(
        key: &str,
        actual: Tensor<B, 4>,
        reference: &[(String, LoadedTensorF32)],
        shape: [usize; 4],
        max_threshold: f32,
        rms_threshold: f64,
    ) {
        let data = actual.into_data().convert::<f32>().to_vec::<f32>().unwrap();
        let expected = crate::tensor_io::find_tensor(reference, key).unwrap();
        assert_eq!(expected.shape, shape);
        assert_eq!(data.len(), expected.data.len());
        let mut max_abs = 0.0f32;
        let mut sum_sq = 0.0f64;
        for (left, right) in data.iter().zip(expected.data.iter()) {
            let delta = (left - right).abs();
            max_abs = max_abs.max(delta);
            sum_sq += (delta as f64) * (delta as f64);
        }
        let rms = (sum_sq / data.len() as f64).sqrt();
        eprintln!("sam2_image_encoder {key} max_abs={max_abs:.6e} rms={rms:.6e}");
        assert!(max_abs <= max_threshold, "{key} max_abs={max_abs}");
        assert!(rms <= rms_threshold, "{key} rms={rms}");
    }
}

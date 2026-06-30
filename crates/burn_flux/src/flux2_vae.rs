use burn::{
    module::Param,
    nn,
    nn::PaddingConfig2d,
    nn::conv::{Conv2d, Conv2dConfig},
    prelude::*,
    tensor::{
        Distribution, FloatDType, activation::sigmoid, module::attention as module_attention,
        ops::AttentionModuleOptions,
    },
};

#[derive(Config, Debug)]
pub struct Flux2VaeEncoderConfig {
    pub base_channels: usize,
    pub latent_moments_channels: usize,
    pub group_norm_groups: usize,
    pub group_norm_eps: f64,
    pub batch_norm_eps: f64,
}

impl Flux2VaeEncoderConfig {
    pub fn flux2() -> Self {
        Self {
            base_channels: 128,
            latent_moments_channels: 64,
            group_norm_groups: 32,
            group_norm_eps: 1.0e-6,
            batch_norm_eps: 1.0e-5,
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> Flux2VaeEncoder<B> {
        Flux2VaeEncoder::new(device, self.clone())
    }
}

#[derive(Module, Debug)]
pub struct Flux2ResnetBlock<B: Backend> {
    pub norm1: nn::GroupNorm<B>,
    pub conv1: Conv2d<B>,
    pub norm2: nn::GroupNorm<B>,
    pub conv2: Conv2d<B>,
    pub conv_shortcut: Option<Conv2d<B>>,
}

impl<B: Backend> Flux2ResnetBlock<B> {
    pub fn new(
        device: &B::Device,
        in_channels: usize,
        out_channels: usize,
        use_shortcut: bool,
        groups: usize,
        eps: f64,
    ) -> Self {
        let norm1 = nn::GroupNormConfig::new(groups, in_channels)
            .with_epsilon(eps)
            .init(device);
        let conv1 = conv3(device, in_channels, out_channels, 1);
        let norm2 = nn::GroupNormConfig::new(groups, out_channels)
            .with_epsilon(eps)
            .init(device);
        let conv2 = conv3(device, out_channels, out_channels, 1);
        let conv_shortcut = use_shortcut
            .then(|| Conv2dConfig::new([in_channels, out_channels], [1, 1]).init(device));
        Self {
            norm1,
            conv1,
            norm2,
            conv2,
            conv_shortcut,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let h = silu(group_norm_forward_accum_f32(&self.norm1, x.clone()));
        let h = self.conv1.forward(h);
        let h = silu(group_norm_forward_accum_f32(&self.norm2, h));
        let h = self.conv2.forward(h);
        let residual = if let Some(shortcut) = &self.conv_shortcut {
            shortcut.forward(x)
        } else {
            x
        };
        h + residual
    }
}

#[derive(Module, Debug)]
pub struct Flux2Downsampler<B: Backend> {
    pub conv: Conv2d<B>,
}

impl<B: Backend> Flux2Downsampler<B> {
    pub fn new(device: &B::Device, channels: usize) -> Self {
        let conv = Conv2dConfig::new([channels, channels], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(0, 0, 1, 1))
            .init(device);
        Self { conv }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let dtype: FloatDType = x.dtype().into();
        if matches!(dtype, FloatDType::F16 | FloatDType::BF16) {
            let mut conv = self.conv.clone();
            conv.padding = PaddingConfig2d::Valid;
            conv.forward(pad_bottom_right_4d(x))
        } else {
            self.conv.forward(x)
        }
    }
}

#[derive(Module, Debug)]
pub struct Flux2Attention<B: Backend> {
    pub group_norm: nn::GroupNorm<B>,
    pub to_q: nn::Linear<B>,
    pub to_k: nn::Linear<B>,
    pub to_v: nn::Linear<B>,
    pub to_out: nn::Linear<B>,
    channels: usize,
}

impl<B: Backend> Flux2Attention<B> {
    pub fn new(device: &B::Device, channels: usize, groups: usize, eps: f64) -> Self {
        Self {
            group_norm: nn::GroupNormConfig::new(groups, channels)
                .with_epsilon(eps)
                .init(device),
            to_q: nn::LinearConfig::new(channels, channels)
                .with_bias(true)
                .init(device),
            to_k: nn::LinearConfig::new(channels, channels)
                .with_bias(true)
                .init(device),
            to_v: nn::LinearConfig::new(channels, channels)
                .with_bias(true)
                .init(device),
            to_out: nn::LinearConfig::new(channels, channels)
                .with_bias(true)
                .init(device),
            channels,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch, channels, height, width] = x.dims();
        let tokens = height * width;
        let h = group_norm_forward_accum_f32(&self.group_norm, x.clone())
            .reshape([batch, channels, tokens])
            .swap_dims(1, 2);
        let q = self
            .to_q
            .forward(h.clone())
            .reshape([batch, tokens, 1, self.channels])
            .permute([0, 2, 1, 3]);
        let k = self
            .to_k
            .forward(h.clone())
            .reshape([batch, tokens, 1, self.channels])
            .permute([0, 2, 1, 3]);
        let v = self
            .to_v
            .forward(h)
            .reshape([batch, tokens, 1, self.channels])
            .permute([0, 2, 1, 3]);
        let out = module_attention(
            q,
            k,
            v,
            None,
            None,
            AttentionModuleOptions {
                scale: Some((self.channels as f64).powf(-0.5)),
                ..Default::default()
            },
        )
        .permute([0, 2, 1, 3])
        .reshape([batch, tokens, self.channels]);
        let out = self
            .to_out
            .forward(out)
            .swap_dims(1, 2)
            .reshape([batch, channels, height, width]);
        x + out
    }
}

#[derive(Module, Debug)]
pub struct Flux2Encoder<B: Backend> {
    pub conv_in: Conv2d<B>,
    pub down_0_resnets: Vec<Flux2ResnetBlock<B>>,
    pub down_0_sampler: Flux2Downsampler<B>,
    pub down_1_resnets: Vec<Flux2ResnetBlock<B>>,
    pub down_1_sampler: Flux2Downsampler<B>,
    pub down_2_resnets: Vec<Flux2ResnetBlock<B>>,
    pub down_2_sampler: Flux2Downsampler<B>,
    pub down_3_resnets: Vec<Flux2ResnetBlock<B>>,
    pub mid_attn: Flux2Attention<B>,
    pub mid_resnets: Vec<Flux2ResnetBlock<B>>,
    pub conv_norm_out: nn::GroupNorm<B>,
    pub conv_out: Conv2d<B>,
}

impl<B: Backend> Flux2Encoder<B> {
    pub fn new(device: &B::Device, config: &Flux2VaeEncoderConfig) -> Self {
        let c = config.base_channels;
        let groups = config.group_norm_groups;
        let eps = config.group_norm_eps;
        Self {
            conv_in: conv3(device, 3, c, 1),
            down_0_resnets: vec![
                Flux2ResnetBlock::new(device, c, c, false, groups, eps),
                Flux2ResnetBlock::new(device, c, c, false, groups, eps),
            ],
            down_0_sampler: Flux2Downsampler::new(device, c),
            down_1_resnets: vec![
                Flux2ResnetBlock::new(device, c, c * 2, true, groups, eps),
                Flux2ResnetBlock::new(device, c * 2, c * 2, false, groups, eps),
            ],
            down_1_sampler: Flux2Downsampler::new(device, c * 2),
            down_2_resnets: vec![
                Flux2ResnetBlock::new(device, c * 2, c * 4, true, groups, eps),
                Flux2ResnetBlock::new(device, c * 4, c * 4, false, groups, eps),
            ],
            down_2_sampler: Flux2Downsampler::new(device, c * 4),
            down_3_resnets: vec![
                Flux2ResnetBlock::new(device, c * 4, c * 4, false, groups, eps),
                Flux2ResnetBlock::new(device, c * 4, c * 4, false, groups, eps),
            ],
            mid_attn: Flux2Attention::new(device, c * 4, groups, eps),
            mid_resnets: vec![
                Flux2ResnetBlock::new(device, c * 4, c * 4, false, groups, eps),
                Flux2ResnetBlock::new(device, c * 4, c * 4, false, groups, eps),
            ],
            conv_norm_out: nn::GroupNormConfig::new(groups, c * 4)
                .with_epsilon(eps)
                .init(device),
            conv_out: conv3(device, c * 4, config.latent_moments_channels, 1),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut x = self.conv_in.forward(x);
        for block in &self.down_0_resnets {
            x = block.forward(x);
        }
        x = self.down_0_sampler.forward(x);
        for block in &self.down_1_resnets {
            x = block.forward(x);
        }
        x = self.down_1_sampler.forward(x);
        for block in &self.down_2_resnets {
            x = block.forward(x);
        }
        x = self.down_2_sampler.forward(x);
        for block in &self.down_3_resnets {
            x = block.forward(x);
        }
        x = self.mid_resnets[0].forward(x);
        x = self.mid_attn.forward(x);
        x = self.mid_resnets[1].forward(x);
        self.conv_out
            .forward(silu(group_norm_forward_accum_f32(&self.conv_norm_out, x)))
    }

    pub fn forward_trace(&self, x: Tensor<B, 4>) -> Flux2EncoderTrace<B> {
        let mut x = self.conv_in.forward(x);
        let conv_in = x.clone();

        x = self.down_0_resnets[0].forward(x);
        let down_0_resnet_0 = x.clone();
        x = self.down_0_resnets[1].forward(x);
        let down_0_resnet_1 = x.clone();
        x = self.down_0_sampler.forward(x);
        let down_0_sampler = x.clone();

        x = self.down_1_resnets[0].forward(x);
        let down_1_resnet_0 = x.clone();
        x = self.down_1_resnets[1].forward(x);
        let down_1_resnet_1 = x.clone();
        x = self.down_1_sampler.forward(x);
        let down_1_sampler = x.clone();

        x = self.down_2_resnets[0].forward(x);
        let down_2_resnet_0 = x.clone();
        x = self.down_2_resnets[1].forward(x);
        let down_2_resnet_1 = x.clone();
        x = self.down_2_sampler.forward(x);
        let down_2_sampler = x.clone();

        x = self.down_3_resnets[0].forward(x);
        let down_3_resnet_0 = x.clone();
        x = self.down_3_resnets[1].forward(x);
        let down_3_resnet_1 = x.clone();

        x = self.mid_resnets[0].forward(x);
        let mid_resnet_0 = x.clone();
        x = self.mid_attn.forward(x);
        let mid_attn = x.clone();
        x = self.mid_resnets[1].forward(x);
        let mid_resnet_1 = x.clone();
        let encoder_out = self
            .conv_out
            .forward(silu(group_norm_forward_accum_f32(&self.conv_norm_out, x)));

        Flux2EncoderTrace {
            conv_in,
            down_0_resnet_0,
            down_0_resnet_1,
            down_0_sampler,
            down_1_resnet_0,
            down_1_resnet_1,
            down_1_sampler,
            down_2_resnet_0,
            down_2_resnet_1,
            down_2_sampler,
            down_3_resnet_0,
            down_3_resnet_1,
            mid_resnet_0,
            mid_attn,
            mid_resnet_1,
            encoder_out,
        }
    }
}

#[derive(Debug)]
pub struct Flux2EncoderTrace<B: Backend> {
    pub conv_in: Tensor<B, 4>,
    pub down_0_resnet_0: Tensor<B, 4>,
    pub down_0_resnet_1: Tensor<B, 4>,
    pub down_0_sampler: Tensor<B, 4>,
    pub down_1_resnet_0: Tensor<B, 4>,
    pub down_1_resnet_1: Tensor<B, 4>,
    pub down_1_sampler: Tensor<B, 4>,
    pub down_2_resnet_0: Tensor<B, 4>,
    pub down_2_resnet_1: Tensor<B, 4>,
    pub down_2_sampler: Tensor<B, 4>,
    pub down_3_resnet_0: Tensor<B, 4>,
    pub down_3_resnet_1: Tensor<B, 4>,
    pub mid_resnet_0: Tensor<B, 4>,
    pub mid_attn: Tensor<B, 4>,
    pub mid_resnet_1: Tensor<B, 4>,
    pub encoder_out: Tensor<B, 4>,
}

#[derive(Module, Debug)]
pub struct FrozenBatchNorm1d<B: Backend> {
    pub running_mean: Param<Tensor<B, 1>>,
    pub running_var: Param<Tensor<B, 1>>,
    epsilon: f64,
}

impl<B: Backend> FrozenBatchNorm1d<B> {
    pub fn new(device: &B::Device, channels: usize, epsilon: f64) -> Self {
        Self {
            running_mean: nn::Initializer::Zeros.init([channels], device),
            running_var: nn::Initializer::Ones.init([channels], device),
            epsilon,
        }
    }

    pub fn forward_4d(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [_batch, channels, _height, _width] = x.dims();
        let mean = self.running_mean.val().reshape([1, channels as i32, 1, 1]);
        let std = self
            .running_var
            .val()
            .reshape([1, channels as i32, 1, 1])
            .add_scalar(self.epsilon)
            .sqrt();
        (x - mean) / std
    }
}

#[derive(Module, Debug)]
pub struct Flux2VaeEncoder<B: Backend> {
    pub encoder: Flux2Encoder<B>,
    pub quant_conv: Conv2d<B>,
    pub bn: FrozenBatchNorm1d<B>,
}

#[derive(Debug)]
pub struct Flux2VaeEncodeTrace<B: Backend> {
    pub encoder: Flux2EncoderTrace<B>,
    pub moments: Tensor<B, 4>,
    pub mean: Tensor<B, 4>,
    pub logvar: Tensor<B, 4>,
    pub latents: Tensor<B, 4>,
    pub unshuffled: Tensor<B, 4>,
    pub normalized: Tensor<B, 4>,
    pub tokens: Tensor<B, 3>,
}

impl<B: Backend> Flux2VaeEncoder<B> {
    pub fn new(device: &B::Device, config: Flux2VaeEncoderConfig) -> Self {
        Self {
            encoder: Flux2Encoder::new(device, &config),
            quant_conv: Conv2dConfig::new(
                [
                    config.latent_moments_channels,
                    config.latent_moments_channels,
                ],
                [1, 1],
            )
            .init(device),
            bn: FrozenBatchNorm1d::new(
                device,
                (config.latent_moments_channels / 2) * 4,
                config.batch_norm_eps,
            ),
        }
    }

    pub fn float_dtype(&self) -> FloatDType {
        self.encoder.conv_in.weight.val().dtype().into()
    }

    pub fn encode(&self, images: Tensor<B, 4>, deterministic: bool) -> Tensor<B, 3> {
        let (mean, logvar) = self.encode_moments(images);
        let latents = if deterministic {
            mean
        } else {
            let dtype: FloatDType = mean.dtype().into();
            let noise = Tensor::<B, 4>::random(
                mean.shape(),
                Distribution::Normal(0.0, 1.0),
                &mean.device(),
            )
            .cast(dtype);
            mean + logvar.mul_scalar(0.5).exp() * noise
        };
        self.encode_latents_to_tokens(latents)
    }

    pub fn encode_with_noise(&self, images: Tensor<B, 4>, noise: Tensor<B, 4>) -> Tensor<B, 3> {
        let (_mean, _logvar, tokens) = self.encode_with_noise_diagnostics(images, noise);
        tokens
    }

    pub fn encode_with_noise_diagnostics(
        &self,
        images: Tensor<B, 4>,
        noise: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 3>) {
        let (mean, logvar) = self.encode_moments(images);
        assert_eq!(
            mean.dims(),
            noise.dims(),
            "Flux2 VAE noise shape must match latent mean shape"
        );
        let tokens = self
            .encode_latents_to_tokens(mean.clone() + logvar.clone().mul_scalar(0.5).exp() * noise);
        (mean, logvar, tokens)
    }

    pub fn encode_with_noise_trace(
        &self,
        images: Tensor<B, 4>,
        noise: Tensor<B, 4>,
    ) -> Flux2VaeEncodeTrace<B> {
        let encoder = self.encoder.forward_trace(images);
        let moments = self.quant_conv.forward(encoder.encoder_out.clone());
        let [batch, channels, height, width] = moments.dims();
        let latent_channels = channels / 2;
        let mean = moments
            .clone()
            .slice([0..batch, 0..latent_channels, 0..height, 0..width]);
        let logvar =
            moments
                .clone()
                .slice([0..batch, latent_channels..channels, 0..height, 0..width]);
        assert_eq!(
            mean.dims(),
            noise.dims(),
            "Flux2 VAE noise shape must match latent mean shape"
        );
        let latents = mean.clone() + logvar.clone().mul_scalar(0.5).exp() * noise;
        let unshuffled = pixel_unshuffle_2(latents.clone());
        let normalized = self.bn.forward_4d(unshuffled.clone());
        let [batch, channels, height, width] = normalized.dims();
        let tokens: Tensor<B, 3> = normalized.clone().flatten(2, 3).swap_dims(1, 2);
        let tokens = tokens.reshape([batch, height * width, channels]);
        Flux2VaeEncodeTrace {
            encoder,
            moments,
            mean,
            logvar,
            latents,
            unshuffled,
            normalized,
            tokens,
        }
    }

    pub fn encode_with_seed(&self, images: Tensor<B, 4>, seed: u64) -> Tensor<B, 3> {
        let (mean, logvar) = self.encode_moments(images);
        let dtype: FloatDType = mean.dtype().into();
        let noise = deterministic_standard_normal_4d(seed, mean.dims(), &mean.device()).cast(dtype);
        self.encode_latents_to_tokens(mean + logvar.mul_scalar(0.5).exp() * noise)
    }

    pub fn encode_moments(&self, images: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let moments = self.quant_conv.forward(self.encoder.forward(images));
        let [batch, channels, height, width] = moments.dims();
        let latent_channels = channels / 2;
        let mean = moments
            .clone()
            .slice([0..batch, 0..latent_channels, 0..height, 0..width]);
        let logvar = moments.slice([0..batch, latent_channels..channels, 0..height, 0..width]);
        (mean, logvar)
    }

    fn encode_latents_to_tokens(&self, latents: Tensor<B, 4>) -> Tensor<B, 3> {
        let latents = pixel_unshuffle_2(latents);
        let latents = self.bn.forward_4d(latents);
        let [batch, channels, height, width] = latents.dims();
        let tokens: Tensor<B, 3> = latents.flatten(2, 3).swap_dims(1, 2);
        tokens.reshape([batch, height * width, channels])
    }
}

fn conv3<B: Backend>(
    device: &B::Device,
    in_channels: usize,
    out_channels: usize,
    padding: usize,
) -> Conv2d<B> {
    Conv2dConfig::new([in_channels, out_channels], [3, 3])
        .with_padding(PaddingConfig2d::Explicit(
            padding, padding, padding, padding,
        ))
        .init(device)
}

fn pixel_unshuffle_2<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [batch, channels, height, width] = x.dims();
    assert!(
        height.is_multiple_of(2) && width.is_multiple_of(2),
        "Flux2 VAE latent map must be divisible by 2 for pixel unshuffle, got {height}x{width}"
    );
    x.reshape([batch, channels, height / 2, 2, width / 2, 2])
        .permute([0, 1, 3, 5, 2, 4])
        .reshape([batch, channels * 4, height / 2, width / 2])
}

fn pad_bottom_right_4d<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [batch, channels, height, width] = x.dims();
    let device = x.device();
    let dtype: FloatDType = x.dtype().into();
    let right = Tensor::<B, 4>::zeros([batch, channels, height, 1], &device).cast(dtype);
    let x = Tensor::cat(vec![x, right], 3);
    let bottom = Tensor::<B, 4>::zeros([batch, channels, 1, width + 1], &device).cast(dtype);
    Tensor::cat(vec![x, bottom], 2)
}

fn group_norm_forward_accum_f32<B: Backend, const D: usize>(
    norm: &nn::GroupNorm<B>,
    input: Tensor<B, D>,
) -> Tensor<B, D> {
    if input.shape()[1] != norm.num_channels {
        panic!(
            "The number of channels in the input tensor should be equal to the number of channels in the GroupNorm module. Expected {}, got {}",
            norm.num_channels,
            input.shape()[1]
        );
    }

    let dtype: FloatDType = input.dtype().into();
    if !matches!(dtype, FloatDType::F16 | FloatDType::BF16) {
        return norm.forward(input);
    }
    if norm.affine && (norm.gamma.is_none() || norm.beta.is_none()) {
        panic!("Affine is set to true, but gamma or beta is None");
    }

    let shape = input.shape();
    if shape.num_elements() <= 2 {
        panic!(
            "input rank for GroupNorm should be at least 3, but got {}",
            shape.num_elements()
        );
    }

    let batch_size = shape[0];
    let num_channels = shape[1];
    let hidden_size = shape[2..].iter().product::<usize>() * num_channels / norm.num_groups;
    let input = input
        .cast(FloatDType::F32)
        .reshape([batch_size, norm.num_groups, hidden_size]);
    let mean = input.clone().sum_dim(2) / hidden_size as f64;
    let input = input.sub(mean);
    let var = input.clone().square().sum_dim(2) / hidden_size as f64;
    let input_normalized = input.div(var.add_scalar(norm.epsilon).sqrt());

    let output = if norm.affine {
        let mut affine_shape = [1; D];
        affine_shape[1] = num_channels;
        input_normalized
            .reshape(shape)
            .mul(
                norm.gamma
                    .as_ref()
                    .expect("group norm gamma should exist")
                    .val()
                    .cast(FloatDType::F32)
                    .reshape(affine_shape),
            )
            .add(
                norm.beta
                    .as_ref()
                    .expect("group norm beta should exist")
                    .val()
                    .cast(FloatDType::F32)
                    .reshape(affine_shape),
            )
    } else {
        input_normalized.reshape(shape)
    };
    output.cast(dtype)
}

fn silu<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    x.clone() * sigmoid(x)
}

fn deterministic_standard_normal_4d<B: Backend>(
    seed: u64,
    shape: [usize; 4],
    device: &B::Device,
) -> Tensor<B, 4> {
    let total = shape[0]
        .saturating_mul(shape[1])
        .saturating_mul(shape[2])
        .saturating_mul(shape[3]);
    if total == 0 {
        return Tensor::zeros(shape, device);
    }

    let mut state = seed;
    let mut values = Vec::with_capacity(total);
    while values.len() < total {
        let u1 = splitmix64_unit_f32(&mut state).max(f32::MIN_POSITIVE);
        let u2 = splitmix64_unit_f32(&mut state);
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = core::f32::consts::TAU * u2;
        values.push(radius * theta.cos());
        if values.len() < total {
            values.push(radius * theta.sin());
        }
    }

    Tensor::<B, 1>::from_floats(values.as_slice(), device).reshape(shape)
}

fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn splitmix64_unit_f32(state: &mut u64) -> f32 {
    let value = splitmix64_next(state) >> 40;
    (value as f32) / ((1u32 << 24) as f32)
}

#[cfg(feature = "import")]
pub fn flux2_vae_key_remap_rules() -> &'static [(&'static str, &'static str)] {
    &[
        (
            r"^encoder\.down_blocks\.(\d+)\.resnets\.(\d+)\.(.+)$",
            "encoder.down_${1}_resnets.$2.$3",
        ),
        (
            r"^encoder\.down_blocks\.(\d+)\.downsamplers\.0\.(.+)$",
            "encoder.down_${1}_sampler.$2",
        ),
        (
            r"^encoder\.mid_block\.resnets\.(\d+)\.(.+)$",
            "encoder.mid_resnets.$1.$2",
        ),
        (
            r"^encoder\.mid_block\.attentions\.0\.(.+)$",
            "encoder.mid_attn.$1",
        ),
        (r"^(encoder\.mid_attn\.to_out)\.0\.(weight|bias)$", "$1.$2"),
        (
            r"^encoder\.conv_norm_out\.weight$",
            "encoder.conv_norm_out.gamma",
        ),
        (
            r"^encoder\.conv_norm_out\.bias$",
            "encoder.conv_norm_out.beta",
        ),
        (r"^(.+)\.norm([12])\.weight$", "$1.norm$2.gamma"),
        (r"^(.+)\.norm([12])\.bias$", "$1.norm$2.beta"),
        (r"^(.+)\.group_norm\.weight$", "$1.group_norm.gamma"),
        (r"^(.+)\.group_norm\.bias$", "$1.group_norm.beta"),
        (r"^bn\.running_mean$", "bn.running_mean"),
        (r"^bn\.running_var$", "bn.running_var"),
    ]
}

#[cfg(feature = "import")]
pub mod import {
    use std::path::{Path, PathBuf};

    use burn::{
        module::{Module, ModuleMapper, Param},
        prelude::*,
        tensor::{Bytes, FloatDType},
    };
    use burn_store::{
        ApplyResult, BurnpackStore, KeyRemapper, ModuleSnapshot, PyTorchToBurnAdapter,
        SafetensorsStore,
    };

    use super::{Flux2VaeEncoder, Flux2VaeEncoderConfig, flux2_vae_key_remap_rules};

    pub fn load_flux2_vae_encoder_from_safetensors<B: Backend>(
        device: &B::Device,
        path: impl AsRef<Path>,
        config: &Flux2VaeEncoderConfig,
    ) -> Result<Flux2VaeEncoder<B>, Box<dyn std::error::Error>> {
        let mut model = config.clone().init(device);
        let mut store = build_store(path.as_ref())?;
        let result = model
            .load_from(&mut store)
            .map_err(|err| format!("failed to load Flux2 VAE encoder weights: {err}"))?;
        validate_nonempty_apply("Flux2 VAE encoder safetensors", &result)?;
        Ok(model)
    }

    pub fn load_flux2_vae_encoder_from_burnpack_file<B: Backend>(
        device: &B::Device,
        burnpack_path: impl AsRef<Path>,
        config: &Flux2VaeEncoderConfig,
    ) -> Result<Flux2VaeEncoder<B>, Box<dyn std::error::Error>> {
        let mut model = config.clone().init(device);
        let mut store =
            BurnpackStore::from_file(burnpack_path.as_ref()).validate(should_validate_burnpack());
        let result = model
            .load_from(&mut store)
            .map_err(|err| format!("failed to load Flux2 VAE encoder burnpack: {err}"))?;
        validate_nonempty_apply("Flux2 VAE encoder burnpack", &result)?;
        Ok(model)
    }

    pub fn apply_flux2_vae_encoder_burnpack_part_bytes<B: Backend>(
        model: &mut Flux2VaeEncoder<B>,
        burnpack_bytes: Vec<u8>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut store = BurnpackStore::from_bytes(Some(Bytes::from_bytes_vec(burnpack_bytes)))
            .allow_partial(true)
            .validate(should_validate_burnpack());
        let result = model
            .load_from(&mut store)
            .map_err(|err| format!("failed to apply Flux2 VAE encoder burnpack part: {err}"))?;
        validate_nonempty_apply("Flux2 VAE encoder burnpack part", &result)?;
        Ok(())
    }

    pub fn import_flux2_vae_encoder_burnpack_to_path<B: Backend>(
        device: &B::Device,
        source_path: impl AsRef<Path>,
        output_path: impl AsRef<Path>,
        config: &Flux2VaeEncoderConfig,
        use_f16: bool,
    ) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let mut model = load_flux2_vae_encoder_from_safetensors::<B>(device, source_path, config)?;
        let dtype = if use_f16 {
            FloatDType::F16
        } else {
            FloatDType::F32
        };
        model = cast_module_float_dtype(model, dtype);
        save_burnpack(&model, output_path.as_ref())?;
        Ok(output_path.as_ref().to_path_buf())
    }

    fn build_store(path: &Path) -> Result<SafetensorsStore, Box<dyn std::error::Error>> {
        let mut remapper = KeyRemapper::new();
        for &(from, to) in flux2_vae_key_remap_rules() {
            remapper = remapper
                .add_pattern(from, to)
                .map_err(|err| format!("invalid Flux2 VAE remap rule {from}->{to}: {err}"))?;
        }
        Ok(SafetensorsStore::from_file(path)
            .with_from_adapter(PyTorchToBurnAdapter)
            .allow_partial(false)
            .remap(remapper)
            .validate(true))
    }

    struct FloatDTypeMapper {
        dtype: FloatDType,
    }

    impl<B: Backend> ModuleMapper<B> for FloatDTypeMapper {
        fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
            let (id, tensor, mapper) = param.consume();
            Param::from_mapped_value(id, tensor.cast(self.dtype), mapper)
        }
    }

    fn cast_module_float_dtype<B: Backend, M: Module<B>>(module: M, dtype: FloatDType) -> M {
        let mut mapper = FloatDTypeMapper { dtype };
        module.map(&mut mapper)
    }

    fn validate_nonempty_apply(
        label: &str,
        result: &ApplyResult,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if result.applied.is_empty() {
            return Err(format!("{label} import did not apply any tensors").into());
        }
        Ok(())
    }

    fn save_burnpack<B: Backend>(
        model: &Flux2VaeEncoder<B>,
        path: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut store = BurnpackStore::from_file(path).overwrite(true);
        model
            .save_into(&mut store)
            .map_err(|err| format!("failed to save Flux2 VAE encoder burnpack: {err}"))?;
        Ok(())
    }

    fn should_validate_burnpack() -> bool {
        cfg!(all(not(target_arch = "wasm32"), debug_assertions))
    }
}

#[cfg(all(test, feature = "import"))]
mod import_tests {
    use super::*;

    fn remap(key: &str) -> String {
        let mut remapper = burn_store::KeyRemapper::new();
        for &(from, to) in flux2_vae_key_remap_rules() {
            remapper = remapper.add_pattern(from, to).unwrap();
        }
        let mut out = key.to_string();
        for (pattern, replacement) in &remapper.patterns {
            if pattern.is_match(&out) {
                out = pattern.replace_all(&out, replacement.as_str()).to_string();
            }
        }
        out
    }

    #[test]
    fn flux2_remaps_diffusers_blocks_and_attention_output() {
        assert_eq!(
            remap("encoder.down_blocks.2.resnets.1.norm1.weight"),
            "encoder.down_2_resnets.1.norm1.gamma"
        );
        assert_eq!(
            remap("encoder.mid_block.attentions.0.to_out.0.weight"),
            "encoder.mid_attn.to_out.weight"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestBackend = burn::backend::NdArray<f32>;

    #[test]
    fn flux2_vae_encoder_output_shape_matches_token_contract() {
        let device = Default::default();
        let config = Flux2VaeEncoderConfig::flux2();
        let model = config.init::<TestBackend>(&device);
        let input = Tensor::<TestBackend, 4>::zeros([1, 3, 32, 32], &device);
        let out = model.encode(input, true);
        assert_eq!(out.dims(), [1, 4, 128]);
    }

    #[test]
    fn flux2_downsampler_padding_matches_explicit_bottom_right_pad() {
        let device = Default::default();
        let downsampler = Flux2Downsampler::<TestBackend>::new(&device, 4);
        let input = Tensor::<TestBackend, 4>::random([1, 4, 8, 8], Distribution::Default, &device);
        let direct = downsampler.forward(input.clone());
        let mut valid_conv = downsampler.conv.clone();
        valid_conv.padding = PaddingConfig2d::Valid;
        let reference = valid_conv.forward(pad_bottom_right_4d(input));

        let direct = direct
            .to_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("direct vec");
        let reference = reference
            .to_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("reference vec");
        assert_eq!(direct.len(), reference.len());
        let max_abs = direct
            .iter()
            .zip(reference.iter())
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_abs <= 1.0e-6,
            "explicit conv padding changed downsampler output: max_abs={max_abs}"
        );
    }

    #[test]
    fn flux2_vae_zero_noise_matches_deterministic_encode() {
        let device = Default::default();
        let model = Flux2VaeEncoderConfig::flux2().init::<TestBackend>(&device);
        let input = Tensor::<TestBackend, 4>::zeros([1, 3, 32, 32], &device);
        let (mean, _logvar) = model.encode_moments(input.clone());
        let noise = Tensor::<TestBackend, 4>::zeros(mean.dims(), &device);

        let deterministic = model
            .encode(input.clone(), true)
            .to_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("deterministic vec");
        let with_zero_noise = model
            .encode_with_noise(input, noise)
            .to_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("noise vec");

        assert_eq!(deterministic, with_zero_noise);
    }

    #[test]
    fn flux2_vae_zero_noise_trace_matches_deterministic_encode() {
        let device = Default::default();
        let model = Flux2VaeEncoderConfig::flux2().init::<TestBackend>(&device);
        let input = Tensor::<TestBackend, 4>::zeros([1, 3, 32, 32], &device);
        let (mean, _logvar) = model.encode_moments(input.clone());
        let noise = Tensor::<TestBackend, 4>::zeros(mean.dims(), &device);

        let deterministic = model
            .encode(input.clone(), true)
            .to_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("deterministic vec");
        let traced = model
            .encode_with_noise_trace(input, noise)
            .tokens
            .to_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("trace vec");

        assert_eq!(deterministic, traced);
    }

    #[test]
    fn flux2_vae_seeded_stochastic_encode_is_reproducible() {
        let device = Default::default();
        let model = Flux2VaeEncoderConfig::flux2().init::<TestBackend>(&device);
        let input = Tensor::<TestBackend, 4>::zeros([1, 3, 32, 32], &device);
        let first = model
            .encode_with_seed(input.clone(), 1234)
            .to_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("first vec");
        let second = model
            .encode_with_seed(input, 1234)
            .to_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("second vec");

        assert_eq!(first, second);
    }
}

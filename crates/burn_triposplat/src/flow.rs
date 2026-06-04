use burn::{
    module::Param,
    nn,
    prelude::*,
    tensor::{Distribution, FloatDType},
};

use crate::components::{
    Mlp, PcdAbsolutePositionEmbedder, RePo3dRotaryEmbedding, SinusoidalEmbedder,
    UnifiedTransformerBlock, silu,
};
use crate::rng::{SplitMix64, deterministic_standard_normal_3d};

#[derive(Config, Debug)]
pub struct LatentSeqMmFlowModelConfig {
    pub q_token_length: usize,
    pub in_channels: usize,
    pub model_channels: usize,
    pub cond_channels: usize,
    pub out_channels: usize,
    pub num_blocks: usize,
    pub num_refiner_blocks: usize,
    pub num_heads: usize,
    pub cam_channels: Option<usize>,
    pub cond2_channels: Option<usize>,
    pub mlp_ratio: f32,
    pub share_mod: bool,
    pub qk_rms_norm: bool,
    pub use_shift_table: bool,
}

impl LatentSeqMmFlowModelConfig {
    pub fn triposplat() -> Self {
        Self {
            q_token_length: 8192,
            in_channels: 16,
            cam_channels: Some(5),
            out_channels: 16,
            model_channels: 1024,
            cond_channels: 1280,
            cond2_channels: Some(128),
            num_refiner_blocks: 2,
            num_blocks: 24,
            num_heads: 16,
            mlp_ratio: 4.0,
            qk_rms_norm: true,
            share_mod: true,
            use_shift_table: true,
        }
    }

    pub fn tiny_for_tests() -> Self {
        Self {
            q_token_length: 8,
            in_channels: 4,
            cam_channels: Some(3),
            out_channels: 4,
            model_channels: 32,
            cond_channels: 16,
            cond2_channels: Some(8),
            num_refiner_blocks: 1,
            num_blocks: 2,
            num_heads: 4,
            mlp_ratio: 2.0,
            qk_rms_norm: true,
            share_mod: true,
            use_shift_table: true,
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> LatentSeqMmFlowModel<B> {
        LatentSeqMmFlowModel::new(device, self.clone())
    }
}

#[derive(Clone, Debug)]
pub struct TripoSplatCondition<B: Backend> {
    pub feature1: Tensor<B, 3>,
    pub feature2: Option<Tensor<B, 3>>,
}

impl<B: Backend> TripoSplatCondition<B> {
    pub fn zeros_like(&self) -> Self {
        let feature1_dtype: FloatDType = self.feature1.dtype().into();
        Self {
            feature1: Tensor::<B, 3>::zeros(self.feature1.shape(), &self.feature1.device())
                .cast(feature1_dtype),
            feature2: self.feature2.as_ref().map(|tensor| {
                let dtype: FloatDType = tensor.dtype().into();
                Tensor::<B, 3>::zeros(tensor.shape(), &tensor.device()).cast(dtype)
            }),
        }
    }

    pub fn with_prefix_padded_feature2(self) -> Self {
        let feature2 = self
            .feature2
            .map(|feature2| prefix_pad_feature2(feature2, self.feature1.dims()[1]));
        Self {
            feature1: self.feature1,
            feature2,
        }
    }
}

#[derive(Clone, Debug)]
pub struct FlowState<B: Backend> {
    pub latent: Tensor<B, 3>,
    pub camera: Option<Tensor<B, 3>>,
}

impl<B: Backend> FlowState<B> {
    pub fn random(
        device: &B::Device,
        batch: usize,
        q_token_length: usize,
        in_channels: usize,
        cam_channels: Option<usize>,
    ) -> Self {
        Self {
            latent: Tensor::random(
                [batch, q_token_length, in_channels],
                Distribution::Normal(0.0, 1.0),
                device,
            ),
            camera: cam_channels.map(|channels| {
                Tensor::random([batch, 1, channels], Distribution::Normal(0.0, 1.0), device)
            }),
        }
    }

    pub fn deterministic_standard_normal(
        device: &B::Device,
        batch: usize,
        q_token_length: usize,
        in_channels: usize,
        cam_channels: Option<usize>,
        seed: u64,
    ) -> Self {
        let mut rng = SplitMix64::new(seed);
        Self {
            latent: deterministic_standard_normal_3d(
                &mut rng,
                [batch, q_token_length, in_channels],
                device,
            ),
            camera: cam_channels.map(|channels| {
                deterministic_standard_normal_3d(&mut rng, [batch, 1, channels], device)
            }),
        }
    }

    fn sub_scaled(self, velocity: FlowState<B>, dt: f32) -> Self {
        Self {
            latent: self.latent - velocity.latent.mul_scalar(dt),
            camera: match (self.camera, velocity.camera) {
                (Some(sample), Some(velocity)) => Some(sample - velocity.mul_scalar(dt)),
                (sample, _) => sample,
            },
        }
    }
}

struct FlowPredictionContext<'a, B: Backend> {
    dtype: FloatDType,
    device: &'a B::Device,
}

#[derive(Module, Debug)]
pub struct LatentSeqMmFlowModel<B: Backend> {
    pub t_embedder: SinusoidalEmbedder<B>,
    pub ada_ln_modulation: Option<nn::Linear<B>>,
    pub input_layer: nn::Linear<B>,
    pub cond_embedder: nn::Linear<B>,
    pub cond_embedder2: Option<nn::Linear<B>>,
    pub noise_repo_layers: Vec<RePo3dRotaryEmbedding<B>>,
    pub context_repo_layers: Vec<RePo3dRotaryEmbedding<B>>,
    pub repo_layers: Vec<RePo3dRotaryEmbedding<B>>,
    pub noise_refiner: Vec<UnifiedTransformerBlock<B>>,
    pub context_refiner: Vec<UnifiedTransformerBlock<B>>,
    pub cam_refiner: Option<Mlp<B>>,
    pub blocks: Vec<UnifiedTransformerBlock<B>>,
    pub shift_table: Option<Param<Tensor<B, 3>>>,
    pub out_layer: nn::Linear<B>,
    pub cam_out_layer: Option<nn::Linear<B>>,
    pub pos_pe: Param<Tensor<B, 2>>,
    config: LatentSeqMmFlowModelConfig,
}

impl<B: Backend> LatentSeqMmFlowModel<B> {
    pub fn new(device: &B::Device, config: LatentSeqMmFlowModelConfig) -> Self {
        let head_dim = config.model_channels / config.num_heads;
        let pos_pe = torch_sobol_seed123_dim3_positions(config.q_token_length, device);
        let block = |modulation, share_mod| {
            UnifiedTransformerBlock::new(
                device,
                config.model_channels,
                config.num_heads,
                config.mlp_ratio,
                true,
                config.qk_rms_norm,
                true,
                modulation,
                share_mod,
                config.use_shift_table,
            )
        };
        Self {
            t_embedder: SinusoidalEmbedder::new(
                device,
                config.model_channels,
                256,
                10_000.0,
                false,
            ),
            ada_ln_modulation: config.share_mod.then(|| {
                nn::LinearConfig::new(config.model_channels, 6 * config.model_channels)
                    .with_bias(true)
                    .init(device)
            }),
            input_layer: nn::LinearConfig::new(config.in_channels, config.model_channels)
                .with_bias(true)
                .init(device),
            cond_embedder: nn::LinearConfig::new(config.cond_channels, config.model_channels)
                .with_bias(true)
                .init(device),
            cond_embedder2: config.cond2_channels.map(|channels| {
                nn::LinearConfig::new(channels, config.model_channels)
                    .with_bias(true)
                    .init(device)
            }),
            noise_repo_layers: (0..config.num_refiner_blocks)
                .map(|_| {
                    RePo3dRotaryEmbedding::new(
                        device,
                        config.model_channels,
                        config.num_heads,
                        head_dim,
                    )
                })
                .collect(),
            context_repo_layers: (0..config.num_refiner_blocks)
                .map(|_| {
                    RePo3dRotaryEmbedding::new(
                        device,
                        config.model_channels,
                        config.num_heads,
                        head_dim,
                    )
                })
                .collect(),
            repo_layers: (0..config.num_blocks)
                .map(|_| {
                    RePo3dRotaryEmbedding::new(
                        device,
                        config.model_channels,
                        config.num_heads,
                        head_dim,
                    )
                })
                .collect(),
            noise_refiner: (0..config.num_refiner_blocks)
                .map(|_| block(true, config.share_mod))
                .collect(),
            context_refiner: (0..config.num_refiner_blocks)
                .map(|_| block(false, false))
                .collect(),
            cam_refiner: config.cam_channels.map(|channels| {
                Mlp::new(
                    device,
                    channels,
                    config.model_channels,
                    config.model_channels,
                    config.num_refiner_blocks,
                )
            }),
            blocks: (0..config.num_blocks)
                .map(|_| block(true, config.share_mod))
                .collect(),
            shift_table: config.use_shift_table.then(|| {
                nn::Initializer::Normal {
                    mean: 0.0,
                    std: (config.model_channels as f64).powf(-0.5),
                }
                .init([1, 2, config.model_channels], device)
            }),
            out_layer: nn::LinearConfig::new(config.model_channels, config.out_channels)
                .with_bias(true)
                .init(device),
            cam_out_layer: config.cam_channels.map(|channels| {
                nn::LinearConfig::new(config.model_channels, channels)
                    .with_bias(true)
                    .init(device)
            }),
            pos_pe,
            config,
        }
    }

    pub fn config(&self) -> &LatentSeqMmFlowModelConfig {
        &self.config
    }

    pub fn reset_canonical_pos_pe(&mut self, device: &B::Device) {
        self.pos_pe = torch_sobol_seed123_dim3_positions(self.config.q_token_length, device);
    }

    pub fn forward(
        &self,
        x_t: FlowState<B>,
        t: Tensor<B, 1>,
        cond: TripoSplatCondition<B>,
    ) -> FlowState<B> {
        let z = x_t.latent;
        let [batch, latent_tokens, _] = z.dims();
        let cond_tokens = cond.feature1.dims()[1];
        let mut h_x = self.input_layer.forward(z);
        let mut h_cond = self.cond_embedder.forward(cond.feature1);
        if let (Some(embedder2), Some(feature2)) = (&self.cond_embedder2, cond.feature2) {
            h_cond = h_cond + embedder2.forward(prefix_pad_feature2(feature2, cond_tokens));
        }
        let t_emb = self.t_embedder.forward(t);
        let t_mod = if self.config.share_mod {
            self.ada_ln_modulation
                .as_ref()
                .expect("shared adaLN modulation missing")
                .forward(silu(t_emb.clone()))
        } else {
            t_emb.clone()
        };

        let pos = PcdAbsolutePositionEmbedder::legacy(self.config.model_channels)
            .forward_2d(self.pos_pe.val())
            .unsqueeze_dim::<3>(0)
            .expand([batch as i64, -1, -1]);
        h_x = h_x + pos;

        for (index, block) in self.noise_refiner.iter().enumerate() {
            let rope = self.noise_repo_layers[index].forward(h_x.clone());
            h_x = block.forward(h_x, Some(t_mod.clone()), Some(&rope));
        }
        for (index, block) in self.context_refiner.iter().enumerate() {
            let rope = self.context_repo_layers[index].forward(h_cond.clone());
            h_cond = block.forward(h_cond, None, Some(&rope));
        }

        let h_cam = match (&self.cam_refiner, x_t.camera) {
            (Some(refiner), Some(camera)) => Some(refiner.forward(camera)),
            _ => None,
        };
        let mut parts = vec![h_x, h_cond];
        if let Some(camera) = h_cam.clone() {
            parts.push(camera);
        }
        let mut h = Tensor::cat(parts, 1);
        for (index, block) in self.blocks.iter().enumerate() {
            let rope = self.repo_layers[index].forward(h.clone());
            h = block.forward(h, Some(t_mod.clone()), Some(&rope));
        }

        let h_channels = h.dims()[2];
        let mut h_x = h.clone().slice([0..batch, 0..latent_tokens, 0..h_channels]);
        h_x = layer_norm_last(h_x, 1.0e-6);
        let mut h_cam = h_cam.map(|camera| {
            let cam_tokens = camera.dims()[1];
            let h_tokens = h.dims()[1];
            let mut h_cam =
                h.clone()
                    .slice([0..batch, h_tokens - cam_tokens..h_tokens, 0..h_channels]);
            h_cam = layer_norm_last(h_cam, 1.0e-6);
            h_cam
        });

        if let Some(shift_table) = &self.shift_table {
            let shifted = shift_table.val() + t_emb.unsqueeze_dim(1);
            let shift = shifted.clone().slice([0..batch, 0..1, 0..h_channels]);
            let scale = shifted.slice([0..batch, 1..2, 0..h_channels]);
            h_x = h_x * (scale.clone() + 1.0) + shift.clone();
            h_cam = h_cam.map(|cam| cam * (scale + 1.0) + shift);
        }

        FlowState {
            latent: self.out_layer.forward(h_x),
            camera: match (self.cam_out_layer.as_ref(), h_cam) {
                (Some(layer), Some(cam)) => Some(layer.forward(cam)),
                _ => None,
            },
        }
    }

    pub fn sample_euler_cfg(
        &self,
        noise: FlowState<B>,
        cond: TripoSplatCondition<B>,
        steps: usize,
        guidance_scale: f32,
        shift: f32,
    ) -> FlowState<B> {
        self.sample_euler_cfg_prefix(noise, cond, steps, steps, guidance_scale, shift)
    }

    pub fn sample_euler_cfg_prefix(
        &self,
        noise: FlowState<B>,
        cond: TripoSplatCondition<B>,
        total_steps: usize,
        prefix_steps: usize,
        guidance_scale: f32,
        shift: f32,
    ) -> FlowState<B> {
        let device = noise.latent.device();
        let dtype: FloatDType = noise.latent.dtype().into();
        let mut sample = noise;
        let neg_cond = cond.zeros_like();
        for index in 0..prefix_steps.min(total_steps) {
            let t = shifted_t(index, total_steps, shift);
            let t_prev = shifted_t(index + 1, total_steps, shift);
            let pred = self.euler_cfg_prediction(
                sample.clone(),
                t,
                cond.clone(),
                neg_cond.clone(),
                guidance_scale,
                FlowPredictionContext {
                    dtype,
                    device: &device,
                },
            );
            sample = sample.sub_scaled(pred, t - t_prev);
        }
        sample
    }

    pub fn euler_cfg_prediction_at_step(
        &self,
        sample: FlowState<B>,
        cond: TripoSplatCondition<B>,
        total_steps: usize,
        step: usize,
        guidance_scale: f32,
        shift: f32,
    ) -> FlowState<B> {
        let device = sample.latent.device();
        let dtype: FloatDType = sample.latent.dtype().into();
        let neg_cond = cond.zeros_like();
        let t = shifted_t(step, total_steps, shift);
        self.euler_cfg_prediction(
            sample,
            t,
            cond,
            neg_cond,
            guidance_scale,
            FlowPredictionContext {
                dtype,
                device: &device,
            },
        )
    }

    fn euler_cfg_prediction(
        &self,
        sample: FlowState<B>,
        t: f32,
        cond: TripoSplatCondition<B>,
        neg_cond: TripoSplatCondition<B>,
        guidance_scale: f32,
        context: FlowPredictionContext<'_, B>,
    ) -> FlowState<B> {
        let t_scaled =
            Tensor::<B, 1>::from_floats([1000.0 * t], context.device).cast(context.dtype);
        let pred = self.forward(sample.clone(), t_scaled.clone(), cond);
        if guidance_scale > 1.0 {
            let neg = self.forward(sample, t_scaled, neg_cond);
            FlowState {
                latent: pred.latent * guidance_scale - neg.latent * (guidance_scale - 1.0),
                camera: match (pred.camera, neg.camera) {
                    (Some(pred), Some(neg)) => {
                        Some(pred * guidance_scale - neg * (guidance_scale - 1.0))
                    }
                    (pred, _) => pred,
                },
            }
        } else {
            pred
        }
    }
}

fn shifted_t(index: usize, steps: usize, shift: f32) -> f32 {
    let base = 1.0 - index as f32 / steps.max(1) as f32;
    shift * base / (1.0 + (shift - 1.0) * base)
}

fn layer_norm_last<B: Backend>(x: Tensor<B, 3>, epsilon: f64) -> Tensor<B, 3> {
    let dtype: FloatDType = x.dtype().into();
    let x_acc = cast_low_precision_to_f32(x, dtype);
    let (var, mean) = x_acc.clone().var_mean_bias(2);
    cast_from_f32_accum((x_acc - mean) / var.add_scalar(epsilon).sqrt(), dtype)
}

fn cast_low_precision_to_f32<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    dtype: FloatDType,
) -> Tensor<B, D> {
    if matches!(dtype, FloatDType::F16 | FloatDType::BF16) {
        tensor.cast(FloatDType::F32)
    } else {
        tensor
    }
}

fn cast_from_f32_accum<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    dtype: FloatDType,
) -> Tensor<B, D> {
    if matches!(dtype, FloatDType::F16 | FloatDType::BF16) {
        tensor.cast(dtype)
    } else {
        tensor
    }
}

const TORCH_SOBOL_SEED123_DIM3_COUNT: usize = 8192;
const TORCH_SOBOL_SEED123_DIM3_BYTES: &[u8] =
    include_bytes!("torch_sobol_seed123_dim3_8192_f32le.bin");

fn torch_sobol_seed123_dim3_positions<B: Backend>(
    count: usize,
    device: &B::Device,
) -> Param<Tensor<B, 2>> {
    assert!(
        count <= TORCH_SOBOL_SEED123_DIM3_COUNT,
        "TripoSplat q_token_length {count} exceeds canonical PyTorch Sobol table length {TORCH_SOBOL_SEED123_DIM3_COUNT}"
    );
    let values = TORCH_SOBOL_SEED123_DIM3_BYTES
        .chunks_exact(4)
        .take(count * 3)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect::<Vec<_>>();
    Param::from_tensor(Tensor::<B, 1>::from_floats(values.as_slice(), device).reshape([count, 3]))
}

fn prefix_pad_feature2<B: Backend>(feature2: Tensor<B, 3>, target_tokens: usize) -> Tensor<B, 3> {
    let [batch, tokens, channels] = feature2.dims();
    assert!(
        tokens <= target_tokens,
        "TripoSplat feature2 token count ({tokens}) exceeds feature1 token count ({target_tokens})"
    );
    if tokens == target_tokens {
        feature2
    } else {
        let dtype: FloatDType = feature2.dtype().into();
        let prefix = Tensor::<B, 3>::zeros(
            [batch, target_tokens - tokens, channels],
            &feature2.device(),
        )
        .cast(dtype);
        Tensor::cat(vec![prefix, feature2], 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestBackend = burn::backend::NdArray<f32>;

    #[test]
    fn tiny_flow_forward_preserves_latent_and_camera_shapes() {
        let device = Default::default();
        let config = LatentSeqMmFlowModelConfig::tiny_for_tests();
        let model = config.clone().init::<TestBackend>(&device);
        let state = FlowState::random(
            &device,
            1,
            config.q_token_length,
            config.in_channels,
            config.cam_channels,
        );
        let cond = TripoSplatCondition {
            feature1: Tensor::zeros([1, 6, config.cond_channels], &device),
            feature2: Some(Tensor::zeros(
                [1, 4, config.cond2_channels.unwrap()],
                &device,
            )),
        };
        let t = Tensor::<TestBackend, 1>::from_floats([1000.0], &device);
        let out = model.forward(state, t, cond);
        assert_eq!(
            out.latent.dims(),
            [1, config.q_token_length, config.out_channels]
        );
        assert_eq!(
            out.camera.unwrap().dims(),
            [1, 1, config.cam_channels.unwrap()]
        );
    }

    #[test]
    fn condition_pads_feature2_prefix_to_match_dinov3_tokens() {
        let device = Default::default();
        let condition = TripoSplatCondition {
            feature1: Tensor::<TestBackend, 3>::ones([1, 9, 16], &device),
            feature2: Some(Tensor::ones([1, 4, 8], &device)),
        }
        .with_prefix_padded_feature2();

        assert_eq!(condition.feature2.unwrap().dims(), [1, 9, 8]);
    }

    #[test]
    fn torch_sobol_seed123_positions_match_pytorch_prefix() {
        let device = Default::default();
        let pos = torch_sobol_seed123_dim3_positions::<TestBackend>(8, &device)
            .val()
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("sobol prefix values");
        let expected = [
            0.0902513266,
            0.5811105371,
            0.4077436924,
            0.7700406313,
            0.2462926656,
            0.6401962042,
            0.6336019039,
            0.7657466531,
            0.0409966968,
            0.4376287460,
            0.4366661012,
            0.7577631474,
            0.3353308439,
            0.9191896319,
            0.5311235785,
            0.5308214426,
            0.2524206042,
            0.2986711860,
            0.8786804676,
            0.7327319980,
            0.8980799317,
            0.1984085888,
            0.0638877451,
            0.1813135445,
        ];
        for (index, (actual, expected)) in pos.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() <= 1.0e-7,
                "Sobol prefix mismatch at {index}: actual={actual} expected={expected}"
            );
        }
    }

    #[test]
    fn deterministic_flow_state_reuses_seeded_noise_stream() {
        let device = Default::default();
        let first =
            FlowState::<TestBackend>::deterministic_standard_normal(&device, 1, 8, 4, Some(5), 42);
        let second =
            FlowState::<TestBackend>::deterministic_standard_normal(&device, 1, 8, 4, Some(5), 42);

        let first_latent = first
            .latent
            .to_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("first latent");
        let second_latent = second
            .latent
            .to_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("second latent");
        let first_camera = first
            .camera
            .expect("first camera")
            .to_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("first camera vec");
        let second_camera = second
            .camera
            .expect("second camera")
            .to_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("second camera vec");

        assert_eq!(first_latent, second_latent);
        assert_eq!(first_camera, second_camera);
    }
}

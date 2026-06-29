use std::time::Instant;

use burn::{
    module::Param,
    nn,
    prelude::*,
    tensor::{DType, Distribution, FloatDType, TensorData},
};

use crate::components::{
    AttentionQkvCapture, AttentionQkvCaptureState, Mlp, PcdAbsolutePositionEmbedder,
    RePo3dRotaryEmbedding, SinusoidalEmbedder, TripoSplatProfileRecord, UnifiedTransformerBlock,
    default_attention_query_chunk_tokens, push_finite_debug_record, push_profile_record, silu,
    sync_elapsed_ms,
};
use crate::config::DEFAULT_Q_TOKEN_LENGTH;
use crate::rng::{SplitMix64, deterministic_standard_normal_3d, skip_standard_normals};

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
            q_token_length: DEFAULT_Q_TOKEN_LENGTH,
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
    pub rng_normals_consumed: usize,
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
            rng_normals_consumed: self.rng_normals_consumed,
        }
    }

    pub fn with_prefix_padded_feature2(self) -> Self {
        let feature2 = self
            .feature2
            .map(|feature2| prefix_pad_feature2(feature2, self.feature1.dims()[1]));
        Self {
            feature1: self.feature1,
            feature2,
            rng_normals_consumed: self.rng_normals_consumed,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CfgPredictionMode {
    Batched,
    #[default]
    BatchedMain,
    Separate,
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
        Self::deterministic_standard_normal_after_skipping(
            device,
            batch,
            q_token_length,
            in_channels,
            cam_channels,
            seed,
            0,
        )
    }

    pub fn deterministic_standard_normal_after_skipping(
        device: &B::Device,
        batch: usize,
        q_token_length: usize,
        in_channels: usize,
        cam_channels: Option<usize>,
        seed: u64,
        skip_normals: usize,
    ) -> Self {
        let mut rng = SplitMix64::new(seed);
        skip_standard_normals(&mut rng, skip_normals);
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
        let latent_dtype: FloatDType = self.latent.dtype().into();
        let velocity_latent = cast_tensor_dtype(velocity.latent, latent_dtype);
        Self {
            latent: self.latent - velocity_latent.mul_scalar(dt),
            camera: match (self.camera, velocity.camera) {
                (Some(sample), Some(velocity)) => {
                    let dtype: FloatDType = sample.dtype().into();
                    Some(sample - cast_tensor_dtype(velocity, dtype).mul_scalar(dt))
                }
                (sample, _) => sample,
            },
        }
    }
}

#[derive(Clone, Debug)]
pub struct FlowEulerTrace<B: Backend> {
    pub pred0: Option<FlowState<B>>,
    pub preds: Vec<FlowState<B>>,
    pub steps: Vec<FlowState<B>>,
}

#[derive(Clone, Debug)]
pub struct FlowPredictionProfile<B: Backend> {
    pub output: FlowState<B>,
    pub records: Vec<TripoSplatProfileRecord>,
}

#[derive(Clone, Debug)]
pub struct FlowPredictionQkvProfile<B: Backend> {
    pub output: FlowState<B>,
    pub records: Vec<TripoSplatProfileRecord>,
    pub qkv_capture: Option<AttentionQkvCapture<B>>,
}

#[derive(Clone, Copy)]
struct FlowPredictionContext<'a, B: Backend> {
    dtype: FloatDType,
    device: &'a B::Device,
}

#[derive(Clone, Debug)]
enum PreparedCfgContext<B: Backend> {
    Conditional {
        cond: Tensor<B, 3>,
        pos: Tensor<B, 3>,
    },
    Batched {
        cond: Tensor<B, 3>,
        pos: Tensor<B, 3>,
    },
    BatchedMain {
        cond: Tensor<B, 3>,
        neg: Tensor<B, 3>,
        pos: Tensor<B, 3>,
    },
    Separate {
        cond: Tensor<B, 3>,
        neg: Tensor<B, 3>,
        pos: Tensor<B, 3>,
    },
}

#[derive(Clone, Debug)]
struct PreparedFlowPrefix<B: Backend> {
    h_x: Tensor<B, 3>,
    h_cam: Option<Tensor<B, 3>>,
    t_emb: Tensor<B, 2>,
    t_mod: Tensor<B, 2>,
    batch: usize,
    latent_tokens: usize,
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
    #[module(skip)]
    latent_pos_embed: Tensor<B, 2>,
    config: LatentSeqMmFlowModelConfig,
}

impl<B: Backend> LatentSeqMmFlowModel<B> {
    pub fn new(device: &B::Device, config: LatentSeqMmFlowModelConfig) -> Self {
        let head_dim = config.model_channels / config.num_heads;
        let pos_pe = torch_sobol_seed123_dim3_positions(config.q_token_length, device);
        let latent_pos_embed = canonical_latent_position_embedding(
            config.q_token_length,
            config.model_channels,
            device,
            pos_pe.val(),
        );
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
            latent_pos_embed,
            config,
        }
    }

    pub fn config(&self) -> &LatentSeqMmFlowModelConfig {
        &self.config
    }

    pub fn float_dtype(&self) -> FloatDType {
        self.input_layer.weight.val().dtype().into()
    }

    pub fn reset_canonical_pos_pe(&mut self, device: &B::Device) {
        self.pos_pe = torch_sobol_seed123_dim3_positions(self.config.q_token_length, device);
        self.latent_pos_embed = canonical_latent_position_embedding(
            self.config.q_token_length,
            self.config.model_channels,
            device,
            self.pos_pe.val(),
        );
    }

    pub fn forward(
        &self,
        x_t: FlowState<B>,
        t: Tensor<B, 1>,
        cond: TripoSplatCondition<B>,
    ) -> FlowState<B> {
        let batch = x_t.latent.dims()[0];
        let h_cond = self.prepare_condition_context(cond);
        let pos = self.prepare_latent_position(batch);
        self.forward_with_prepared_condition_context(x_t, t, h_cond, pos)
    }

    fn prepare_condition_context(&self, cond: TripoSplatCondition<B>) -> Tensor<B, 3> {
        let cond_tokens = cond.feature1.dims()[1];
        let dtype = self.float_dtype();
        let mut h_cond = self
            .cond_embedder
            .forward(cast_tensor_dtype(cond.feature1, dtype));
        if let (Some(embedder2), Some(feature2)) = (&self.cond_embedder2, cond.feature2) {
            h_cond = h_cond
                + embedder2.forward(cast_tensor_dtype(
                    prefix_pad_feature2(feature2, cond_tokens),
                    dtype,
                ));
        }
        for (index, block) in self.context_refiner.iter().enumerate() {
            let rope = self.context_repo_layers[index].forward(h_cond.clone());
            h_cond = block.forward(h_cond, None, Some(&rope));
        }
        h_cond
    }

    fn prepare_latent_position(&self, batch: usize) -> Tensor<B, 3> {
        let dtype = self.float_dtype();
        self.latent_pos_embed
            .clone()
            .cast(dtype)
            .unsqueeze_dim::<3>(0)
            .expand([batch as i64, -1, -1])
    }

    fn prepare_cfg_context(
        &self,
        cond: TripoSplatCondition<B>,
        guidance_scale: f32,
        cfg_mode: CfgPredictionMode,
    ) -> PreparedCfgContext<B> {
        let batch = cond.feature1.dims()[0];
        if guidance_scale <= 1.0 {
            return PreparedCfgContext::Conditional {
                cond: self.prepare_condition_context(cond),
                pos: self.prepare_latent_position(batch),
            };
        }

        let neg_cond = cond.zeros_like();
        let cond = self.prepare_condition_context(cond);
        let neg = self.prepare_condition_context(neg_cond);
        match cfg_mode {
            CfgPredictionMode::Batched => PreparedCfgContext::Batched {
                cond: Tensor::cat(vec![cond, neg], 0),
                pos: self.prepare_latent_position(batch * 2),
            },
            CfgPredictionMode::BatchedMain => PreparedCfgContext::BatchedMain {
                cond,
                neg,
                pos: self.prepare_latent_position(batch),
            },
            CfgPredictionMode::Separate => PreparedCfgContext::Separate {
                cond,
                neg,
                pos: self.prepare_latent_position(batch),
            },
        }
    }

    fn forward_with_prepared_condition_context(
        &self,
        x_t: FlowState<B>,
        t: Tensor<B, 1>,
        h_cond: Tensor<B, 3>,
        pos: Tensor<B, 3>,
    ) -> FlowState<B> {
        self.forward_with_prepared_condition_context_optional_query_chunk_tokens(
            x_t, t, h_cond, pos, None,
        )
    }

    fn forward_with_prepared_condition_context_optional_query_chunk_tokens(
        &self,
        x_t: FlowState<B>,
        t: Tensor<B, 1>,
        h_cond: Tensor<B, 3>,
        pos: Tensor<B, 3>,
        query_chunk_tokens: Option<usize>,
    ) -> FlowState<B> {
        let prefix = self.prepare_forward_prefix_optional_query_chunk_tokens(
            x_t,
            t,
            pos,
            query_chunk_tokens,
        );
        self.forward_main_with_prepared_prefix_optional_query_chunk_tokens(
            prefix,
            h_cond,
            query_chunk_tokens,
        )
    }

    fn prepare_forward_prefix_optional_query_chunk_tokens(
        &self,
        x_t: FlowState<B>,
        t: Tensor<B, 1>,
        pos: Tensor<B, 3>,
        query_chunk_tokens: Option<usize>,
    ) -> PreparedFlowPrefix<B> {
        let dtype = self.float_dtype();
        let z = cast_tensor_dtype(x_t.latent, dtype);
        let [batch, latent_tokens, _] = z.dims();
        let mut h_x = self.input_layer.forward(z);
        let t_emb = self.t_embedder.forward(t);
        let t_mod = if self.config.share_mod {
            self.ada_ln_modulation
                .as_ref()
                .expect("shared adaLN modulation missing")
                .forward(silu(t_emb.clone()))
        } else {
            t_emb.clone()
        };

        h_x = h_x + pos;

        for (index, block) in self.noise_refiner.iter().enumerate() {
            let rope = self.noise_repo_layers[index].forward(h_x.clone());
            h_x = if let Some(query_chunk_tokens) = query_chunk_tokens {
                block.forward_with_query_chunk_tokens(
                    h_x,
                    Some(t_mod.clone()),
                    Some(&rope),
                    query_chunk_tokens,
                )
            } else {
                block.forward(h_x, Some(t_mod.clone()), Some(&rope))
            };
        }

        let h_cam = match (&self.cam_refiner, x_t.camera) {
            (Some(refiner), Some(camera)) => {
                Some(refiner.forward(cast_tensor_dtype(camera, dtype)))
            }
            _ => None,
        };
        PreparedFlowPrefix {
            h_x,
            h_cam,
            t_emb,
            t_mod,
            batch,
            latent_tokens,
        }
    }

    fn forward_main_with_prepared_prefix_optional_query_chunk_tokens(
        &self,
        prefix: PreparedFlowPrefix<B>,
        h_cond: Tensor<B, 3>,
        query_chunk_tokens: Option<usize>,
    ) -> FlowState<B> {
        let PreparedFlowPrefix {
            h_x,
            h_cam,
            t_emb,
            t_mod,
            batch,
            latent_tokens,
        } = prefix;
        let mut parts = vec![h_x, h_cond];
        if let Some(camera) = h_cam.clone() {
            parts.push(camera);
        }
        let mut h = Tensor::cat(parts, 1);
        for (index, block) in self.blocks.iter().enumerate() {
            let rope = self.repo_layers[index].forward(h.clone());
            h = if let Some(query_chunk_tokens) = query_chunk_tokens {
                block.forward_with_query_chunk_tokens(
                    h,
                    Some(t_mod.clone()),
                    Some(&rope),
                    query_chunk_tokens,
                )
            } else {
                block.forward(h, Some(t_mod.clone()), Some(&rope))
            };
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

    fn forward_main_batched_with_prepared_prefix_optional_query_chunk_tokens(
        &self,
        prefix: PreparedFlowPrefix<B>,
        cond: Tensor<B, 3>,
        neg: Tensor<B, 3>,
        query_chunk_tokens: Option<usize>,
    ) -> (FlowState<B>, FlowState<B>) {
        let PreparedFlowPrefix {
            h_x,
            h_cam,
            t_emb,
            t_mod,
            batch,
            latent_tokens,
        } = prefix;
        let batched_prefix = PreparedFlowPrefix {
            h_x: Tensor::cat(vec![h_x.clone(), h_x], 0),
            h_cam: h_cam.map(|camera| Tensor::cat(vec![camera.clone(), camera], 0)),
            t_emb: Tensor::cat(vec![t_emb.clone(), t_emb], 0),
            t_mod: Tensor::cat(vec![t_mod.clone(), t_mod], 0),
            batch: batch * 2,
            latent_tokens,
        };
        let batched = self.forward_main_with_prepared_prefix_optional_query_chunk_tokens(
            batched_prefix,
            Tensor::cat(vec![cond, neg], 0),
            query_chunk_tokens,
        );
        split_flow_state_batch(batched, batch)
    }

    fn profile_forward_main_with_prepared_prefix_optional_query_chunk_tokens(
        &self,
        label: &str,
        prefix: PreparedFlowPrefix<B>,
        h_cond: Tensor<B, 3>,
        query_chunk_tokens: usize,
        records: &mut Vec<TripoSplatProfileRecord>,
        mut qkv_capture: Option<&mut AttentionQkvCaptureState<B>>,
    ) -> FlowState<B> {
        let device = h_cond.device();
        let PreparedFlowPrefix {
            h_x,
            h_cam,
            t_emb,
            t_mod,
            batch,
            latent_tokens,
        } = prefix;
        let concat_start = Instant::now();
        let mut parts = vec![h_x, h_cond];
        if let Some(camera) = h_cam.clone() {
            parts.push(camera);
        }
        let mut h = Tensor::cat(parts, 1);
        push_profile_record(
            records,
            format!("{label}.concat_main_tokens"),
            batch,
            h.dims()[1],
            self.config.model_channels,
            sync_elapsed_ms::<B>(&device, concat_start),
        );
        push_finite_debug_record(records, format!("{label}.concat_main_tokens.out"), &h);

        for (index, block) in self.blocks.iter().enumerate() {
            let rope_start = Instant::now();
            let rope = self.repo_layers[index].forward(h.clone());
            push_profile_record(
                records,
                format!("{label}.main_{index:02}.repo"),
                batch,
                h.dims()[1],
                self.config.model_channels,
                sync_elapsed_ms::<B>(&device, rope_start),
            );
            h = block.forward_profiled_with_qkv_capture(
                &format!("{label}.main_{index:02}.block"),
                h,
                Some(t_mod.clone()),
                Some(&rope),
                query_chunk_tokens,
                records,
                qkv_capture.as_deref_mut(),
            );
            push_finite_debug_record(records, format!("{label}.main_{index:02}.out"), &h);
        }

        let output_start = Instant::now();
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

        let latent = self.out_layer.forward(h_x);
        let camera = match (self.cam_out_layer.as_ref(), h_cam) {
            (Some(layer), Some(cam)) => Some(layer.forward(cam)),
            _ => None,
        };
        push_profile_record(
            records,
            format!("{label}.output_projection"),
            batch,
            latent_tokens,
            self.config.out_channels,
            sync_elapsed_ms::<B>(&device, output_start),
        );
        push_finite_debug_record(
            records,
            format!("{label}.output_projection.latent"),
            &latent,
        );
        if let Some(camera) = &camera {
            push_finite_debug_record(records, format!("{label}.output_projection.camera"), camera);
        }
        FlowState { latent, camera }
    }

    #[allow(clippy::too_many_arguments)]
    fn profile_forward_main_batched_with_prepared_prefix_optional_query_chunk_tokens(
        &self,
        label: &str,
        prefix: PreparedFlowPrefix<B>,
        cond: Tensor<B, 3>,
        neg: Tensor<B, 3>,
        query_chunk_tokens: usize,
        records: &mut Vec<TripoSplatProfileRecord>,
        qkv_capture: Option<&mut AttentionQkvCaptureState<B>>,
    ) -> (FlowState<B>, FlowState<B>) {
        let PreparedFlowPrefix {
            h_x,
            h_cam,
            t_emb,
            t_mod,
            batch,
            latent_tokens,
        } = prefix;
        let batched_prefix = PreparedFlowPrefix {
            h_x: Tensor::cat(vec![h_x.clone(), h_x], 0),
            h_cam: h_cam.map(|camera| Tensor::cat(vec![camera.clone(), camera], 0)),
            t_emb: Tensor::cat(vec![t_emb.clone(), t_emb], 0),
            t_mod: Tensor::cat(vec![t_mod.clone(), t_mod], 0),
            batch: batch * 2,
            latent_tokens,
        };
        let batched = self.profile_forward_main_with_prepared_prefix_optional_query_chunk_tokens(
            label,
            batched_prefix,
            Tensor::cat(vec![cond, neg], 0),
            query_chunk_tokens,
            records,
            qkv_capture,
        );
        split_flow_state_batch(batched, batch)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn trace_euler_cfg_prediction_at_step_with_mode(
        &self,
        sample: FlowState<B>,
        cond: TripoSplatCondition<B>,
        total_steps: usize,
        step: usize,
        guidance_scale: f32,
        shift: f32,
        cfg_mode: CfgPredictionMode,
        token_limit: usize,
    ) -> Vec<(String, Tensor<B, 3>)> {
        let device = sample.latent.device();
        let dtype: FloatDType = sample.latent.dtype().into();
        let schedule = flow_schedule_step(step, total_steps, shift);
        let prepared_context = self.prepare_cfg_context(cond, guidance_scale, cfg_mode);
        let mut trace = Vec::new();
        match prepared_context {
            PreparedCfgContext::Conditional { cond, pos } => {
                let batch = sample.latent.dims()[0];
                let pred = self.trace_forward_with_prepared_condition_context(
                    "cfg.conditional.forward",
                    sample,
                    timestep_tensor(batch, schedule.t_scaled, &device, dtype),
                    cond,
                    pos,
                    token_limit,
                    &mut trace,
                );
                push_flow_trace_tensor3(
                    &mut trace,
                    "cfg.conditional.pred.latent",
                    &pred.latent,
                    token_limit,
                );
                if let Some(camera) = &pred.camera {
                    push_flow_trace_tensor3(
                        &mut trace,
                        "cfg.conditional.pred.camera",
                        camera,
                        token_limit,
                    );
                }
            }
            PreparedCfgContext::Batched { cond, pos } => {
                let batch = sample.latent.dims()[0];
                let pred = self.trace_forward_with_prepared_condition_context(
                    "cfg.batched.forward",
                    concat_flow_state_batch(sample),
                    timestep_tensor(batch * 2, schedule.t_scaled, &device, dtype),
                    cond,
                    pos,
                    token_limit,
                    &mut trace,
                );
                let (pred, neg) = split_flow_state_batch(pred, batch);
                let out = blend_cfg_prediction(pred, neg, guidance_scale);
                push_flow_trace_tensor3(
                    &mut trace,
                    "cfg.batched.blend.latent",
                    &out.latent,
                    token_limit,
                );
                if let Some(camera) = &out.camera {
                    push_flow_trace_tensor3(
                        &mut trace,
                        "cfg.batched.blend.camera",
                        camera,
                        token_limit,
                    );
                }
            }
            PreparedCfgContext::BatchedMain { cond, neg, pos } => {
                let batch = sample.latent.dims()[0];
                let t_scaled = timestep_tensor(batch, schedule.t_scaled, &device, dtype);
                let pred = self.trace_forward_with_prepared_condition_context(
                    "cfg.batched_main.cond_forward",
                    sample.clone(),
                    t_scaled.clone(),
                    cond,
                    pos.clone(),
                    token_limit,
                    &mut trace,
                );
                let neg = self.trace_forward_with_prepared_condition_context(
                    "cfg.batched_main.neg_forward",
                    sample,
                    t_scaled,
                    neg,
                    pos,
                    token_limit,
                    &mut trace,
                );
                let out = blend_cfg_prediction(pred, neg, guidance_scale);
                push_flow_trace_tensor3(
                    &mut trace,
                    "cfg.batched_main.blend.latent",
                    &out.latent,
                    token_limit,
                );
                if let Some(camera) = &out.camera {
                    push_flow_trace_tensor3(
                        &mut trace,
                        "cfg.batched_main.blend.camera",
                        camera,
                        token_limit,
                    );
                }
            }
            PreparedCfgContext::Separate { cond, neg, pos } => {
                let batch = sample.latent.dims()[0];
                let t_scaled = timestep_tensor(batch, schedule.t_scaled, &device, dtype);
                let pred = self.trace_forward_with_prepared_condition_context(
                    "cfg.separate.cond_forward",
                    sample.clone(),
                    t_scaled.clone(),
                    cond,
                    pos.clone(),
                    token_limit,
                    &mut trace,
                );
                let out = if guidance_scale > 1.0 {
                    let neg = self.trace_forward_with_prepared_condition_context(
                        "cfg.separate.neg_forward",
                        sample,
                        t_scaled,
                        neg,
                        pos,
                        token_limit,
                        &mut trace,
                    );
                    blend_cfg_prediction(pred, neg, guidance_scale)
                } else {
                    pred
                };
                push_flow_trace_tensor3(
                    &mut trace,
                    "cfg.separate.blend.latent",
                    &out.latent,
                    token_limit,
                );
                if let Some(camera) = &out.camera {
                    push_flow_trace_tensor3(
                        &mut trace,
                        "cfg.separate.blend.camera",
                        camera,
                        token_limit,
                    );
                }
            }
        }
        trace
    }

    #[allow(clippy::too_many_arguments)]
    fn trace_forward_with_prepared_condition_context(
        &self,
        label: &str,
        x_t: FlowState<B>,
        t: Tensor<B, 1>,
        h_cond: Tensor<B, 3>,
        pos: Tensor<B, 3>,
        token_limit: usize,
        trace: &mut Vec<(String, Tensor<B, 3>)>,
    ) -> FlowState<B> {
        let dtype = self.float_dtype();
        let z = cast_tensor_dtype(x_t.latent, dtype);
        let [batch, latent_tokens, _] = z.dims();
        let mut h_x = self.input_layer.forward(z);
        push_flow_trace_tensor3(trace, format!("{label}.input_layer.out"), &h_x, token_limit);
        let t_emb = self.t_embedder.forward(t);
        let t_mod = if self.config.share_mod {
            self.ada_ln_modulation
                .as_ref()
                .expect("shared adaLN modulation missing")
                .forward(silu(t_emb.clone()))
        } else {
            t_emb.clone()
        };
        push_flow_trace_tensor3(
            trace,
            format!("{label}.latent_position.out"),
            &pos,
            token_limit,
        );
        h_x = h_x + pos;
        push_flow_trace_tensor3(
            trace,
            format!("{label}.input_timestep_position.out"),
            &h_x,
            token_limit,
        );
        push_flow_trace_tensor3(
            trace,
            format!("{label}.condition_context.out"),
            &h_cond,
            token_limit,
        );

        for (index, block) in self.noise_refiner.iter().enumerate() {
            let rope = self.noise_repo_layers[index].forward(h_x.clone());
            h_x = block.forward_trace_selected(
                &format!("{label}.noise_refiner_{index:02}.block"),
                h_x,
                Some(t_mod.clone()),
                Some(&rope),
                token_limit,
                trace,
            );
            push_flow_trace_tensor3(
                trace,
                format!("{label}.noise_refiner_{index:02}.out"),
                &h_x,
                token_limit,
            );
        }

        let h_cam = match (&self.cam_refiner, x_t.camera) {
            (Some(refiner), Some(camera)) => {
                let h_cam = refiner.forward(cast_tensor_dtype(camera, dtype));
                push_flow_trace_tensor3(
                    trace,
                    format!("{label}.cam_refiner.out"),
                    &h_cam,
                    token_limit,
                );
                Some(h_cam)
            }
            _ => None,
        };
        let mut parts = vec![h_x, h_cond];
        if let Some(camera) = h_cam.clone() {
            parts.push(camera);
        }
        let mut h = Tensor::cat(parts, 1);
        push_flow_trace_tensor3(
            trace,
            format!("{label}.concat_main_tokens.out"),
            &h,
            token_limit,
        );
        for (index, block) in self.blocks.iter().enumerate() {
            let rope = self.repo_layers[index].forward(h.clone());
            h = block.forward_trace_selected(
                &format!("{label}.main_{index:02}.block"),
                h,
                Some(t_mod.clone()),
                Some(&rope),
                token_limit,
                trace,
            );
            push_flow_trace_tensor3(
                trace,
                format!("{label}.main_{index:02}.out"),
                &h,
                token_limit,
            );
        }

        let h_channels = h.dims()[2];
        let mut h_x = h.clone().slice([0..batch, 0..latent_tokens, 0..h_channels]);
        h_x = layer_norm_last(h_x, 1.0e-6);
        push_flow_trace_tensor3(
            trace,
            format!("{label}.output_norm.latent"),
            &h_x,
            token_limit,
        );
        let mut h_cam = h_cam.map(|camera| {
            let cam_tokens = camera.dims()[1];
            let h_tokens = h.dims()[1];
            let h_cam = h
                .clone()
                .slice([0..batch, h_tokens - cam_tokens..h_tokens, 0..h_channels]);
            layer_norm_last(h_cam, 1.0e-6)
        });
        if let Some(camera) = &h_cam {
            push_flow_trace_tensor3(
                trace,
                format!("{label}.output_norm.camera"),
                camera,
                token_limit,
            );
        }

        if let Some(shift_table) = &self.shift_table {
            let shifted = shift_table.val() + t_emb.unsqueeze_dim(1);
            let shift = shifted.clone().slice([0..batch, 0..1, 0..h_channels]);
            let scale = shifted.slice([0..batch, 1..2, 0..h_channels]);
            h_x = h_x * (scale.clone() + 1.0) + shift.clone();
            h_cam = h_cam.map(|cam| cam * (scale + 1.0) + shift);
            push_flow_trace_tensor3(
                trace,
                format!("{label}.output_shift.latent"),
                &h_x,
                token_limit,
            );
            if let Some(camera) = &h_cam {
                push_flow_trace_tensor3(
                    trace,
                    format!("{label}.output_shift.camera"),
                    camera,
                    token_limit,
                );
            }
        }

        let latent = self.out_layer.forward(h_x);
        push_flow_trace_tensor3(
            trace,
            format!("{label}.output_projection.latent"),
            &latent,
            token_limit,
        );
        let camera = match (self.cam_out_layer.as_ref(), h_cam) {
            (Some(layer), Some(cam)) => {
                let camera = layer.forward(cam);
                push_flow_trace_tensor3(
                    trace,
                    format!("{label}.output_projection.camera"),
                    &camera,
                    token_limit,
                );
                Some(camera)
            }
            _ => None,
        };
        FlowState { latent, camera }
    }

    #[allow(clippy::too_many_arguments)]
    fn profile_forward_with_prepared_condition_context(
        &self,
        label: &str,
        x_t: FlowState<B>,
        t: Tensor<B, 1>,
        h_cond: Tensor<B, 3>,
        pos: Tensor<B, 3>,
        query_chunk_tokens: usize,
        records: &mut Vec<TripoSplatProfileRecord>,
        mut qkv_capture: Option<&mut AttentionQkvCaptureState<B>>,
    ) -> FlowState<B> {
        let device = x_t.latent.device();
        B::sync(&device).expect("profile pre-sync failed");
        let total_start = Instant::now();
        let dtype = self.float_dtype();
        let z = cast_tensor_dtype(x_t.latent, dtype);
        let [batch, latent_tokens, _] = z.dims();
        let input_start = Instant::now();
        let mut h_x = self.input_layer.forward(z);
        push_finite_debug_record(records, format!("{label}.input_layer.out"), &h_x);
        let t_emb = self.t_embedder.forward(t);
        push_finite_debug_record(records, format!("{label}.t_embedder.out"), &t_emb);
        let t_mod = if self.config.share_mod {
            self.ada_ln_modulation
                .as_ref()
                .expect("shared adaLN modulation missing")
                .forward(silu(t_emb.clone()))
        } else {
            t_emb.clone()
        };
        push_finite_debug_record(records, format!("{label}.t_mod.out"), &t_mod);
        push_finite_debug_record(records, format!("{label}.latent_position.out"), &pos);
        h_x = h_x + pos;
        push_profile_record(
            records,
            format!("{label}.input_timestep_position"),
            batch,
            latent_tokens,
            self.config.model_channels,
            sync_elapsed_ms::<B>(&device, input_start),
        );
        push_finite_debug_record(
            records,
            format!("{label}.input_timestep_position.out"),
            &h_x,
        );

        for (index, block) in self.noise_refiner.iter().enumerate() {
            let rope_start = Instant::now();
            let rope = self.noise_repo_layers[index].forward(h_x.clone());
            push_profile_record(
                records,
                format!("{label}.noise_refiner_{index:02}.repo"),
                batch,
                h_x.dims()[1],
                self.config.model_channels,
                sync_elapsed_ms::<B>(&device, rope_start),
            );
            h_x = block.forward_profiled_with_qkv_capture(
                &format!("{label}.noise_refiner_{index:02}.block"),
                h_x,
                Some(t_mod.clone()),
                Some(&rope),
                query_chunk_tokens,
                records,
                qkv_capture.as_deref_mut(),
            );
            push_finite_debug_record(
                records,
                format!("{label}.noise_refiner_{index:02}.out"),
                &h_x,
            );
        }

        let h_cam = match (&self.cam_refiner, x_t.camera) {
            (Some(refiner), Some(camera)) => {
                let cam_start = Instant::now();
                let h_cam = refiner.forward(cast_tensor_dtype(camera, dtype));
                push_profile_record(
                    records,
                    format!("{label}.cam_refiner"),
                    batch,
                    h_cam.dims()[1],
                    self.config.model_channels,
                    sync_elapsed_ms::<B>(&device, cam_start),
                );
                push_finite_debug_record(records, format!("{label}.cam_refiner.out"), &h_cam);
                Some(h_cam)
            }
            _ => None,
        };
        let concat_start = Instant::now();
        let mut parts = vec![h_x, h_cond];
        if let Some(camera) = h_cam.clone() {
            parts.push(camera);
        }
        let mut h = Tensor::cat(parts, 1);
        push_profile_record(
            records,
            format!("{label}.concat_main_tokens"),
            batch,
            h.dims()[1],
            self.config.model_channels,
            sync_elapsed_ms::<B>(&device, concat_start),
        );
        push_finite_debug_record(records, format!("{label}.concat_main_tokens.out"), &h);
        for (index, block) in self.blocks.iter().enumerate() {
            let rope_start = Instant::now();
            let rope = self.repo_layers[index].forward(h.clone());
            push_profile_record(
                records,
                format!("{label}.main_{index:02}.repo"),
                batch,
                h.dims()[1],
                self.config.model_channels,
                sync_elapsed_ms::<B>(&device, rope_start),
            );
            h = block.forward_profiled_with_qkv_capture(
                &format!("{label}.main_{index:02}.block"),
                h,
                Some(t_mod.clone()),
                Some(&rope),
                query_chunk_tokens,
                records,
                qkv_capture.as_deref_mut(),
            );
            push_finite_debug_record(records, format!("{label}.main_{index:02}.out"), &h);
        }

        let output_start = Instant::now();
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

        let latent = self.out_layer.forward(h_x);
        let camera = match (self.cam_out_layer.as_ref(), h_cam) {
            (Some(layer), Some(cam)) => Some(layer.forward(cam)),
            _ => None,
        };
        push_profile_record(
            records,
            format!("{label}.output_projection"),
            batch,
            latent_tokens,
            self.config.out_channels,
            sync_elapsed_ms::<B>(&device, output_start),
        );
        push_finite_debug_record(
            records,
            format!("{label}.output_projection.latent"),
            &latent,
        );
        if let Some(camera) = &camera {
            push_finite_debug_record(records, format!("{label}.output_projection.camera"), camera);
        }
        push_profile_record(
            records,
            format!("{label}.total"),
            batch,
            latent_tokens,
            self.config.model_channels,
            sync_elapsed_ms::<B>(&device, total_start),
        );
        FlowState { latent, camera }
    }

    pub fn sample_euler_cfg(
        &self,
        noise: FlowState<B>,
        cond: TripoSplatCondition<B>,
        steps: usize,
        guidance_scale: f32,
        shift: f32,
    ) -> FlowState<B> {
        self.sample_euler_cfg_prefix_with_mode(
            noise,
            cond,
            steps,
            steps,
            guidance_scale,
            shift,
            CfgPredictionMode::Separate,
        )
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
        self.sample_euler_cfg_prefix_with_mode(
            noise,
            cond,
            total_steps,
            prefix_steps,
            guidance_scale,
            shift,
            CfgPredictionMode::Separate,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sample_euler_cfg_prefix_with_mode(
        &self,
        noise: FlowState<B>,
        cond: TripoSplatCondition<B>,
        total_steps: usize,
        prefix_steps: usize,
        guidance_scale: f32,
        shift: f32,
        cfg_mode: CfgPredictionMode,
    ) -> FlowState<B> {
        self.sample_euler_cfg_prefix_with_mode_optional_query_chunk_tokens(
            noise,
            cond,
            total_steps,
            prefix_steps,
            guidance_scale,
            shift,
            cfg_mode,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sample_euler_cfg_prefix_with_mode_and_query_chunk_tokens(
        &self,
        noise: FlowState<B>,
        cond: TripoSplatCondition<B>,
        total_steps: usize,
        prefix_steps: usize,
        guidance_scale: f32,
        shift: f32,
        cfg_mode: CfgPredictionMode,
        query_chunk_tokens: usize,
    ) -> FlowState<B> {
        self.sample_euler_cfg_prefix_with_mode_optional_query_chunk_tokens(
            noise,
            cond,
            total_steps,
            prefix_steps,
            guidance_scale,
            shift,
            cfg_mode,
            Some(query_chunk_tokens),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn sample_euler_cfg_prefix_with_mode_optional_query_chunk_tokens(
        &self,
        noise: FlowState<B>,
        cond: TripoSplatCondition<B>,
        total_steps: usize,
        prefix_steps: usize,
        guidance_scale: f32,
        shift: f32,
        cfg_mode: CfgPredictionMode,
        query_chunk_tokens: Option<usize>,
    ) -> FlowState<B> {
        let device = noise.latent.device();
        let dtype: FloatDType = noise.latent.dtype().into();
        let mut sample = noise;
        let prepared_context = self.prepare_cfg_context(cond, guidance_scale, cfg_mode);
        for index in 0..prefix_steps.min(total_steps) {
            let schedule = flow_schedule_step(index, total_steps, shift);
            let pred = self.euler_cfg_prediction_with_prepared_context_optional_query_chunk_tokens(
                sample.clone(),
                schedule.t_scaled,
                guidance_scale,
                &prepared_context,
                FlowPredictionContext {
                    dtype,
                    device: &device,
                },
                query_chunk_tokens,
            );
            sample = sample.sub_scaled(pred, schedule.dt);
        }
        sample
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sample_euler_cfg_trace_with_mode(
        &self,
        noise: FlowState<B>,
        cond: TripoSplatCondition<B>,
        total_steps: usize,
        prefix_steps: usize,
        guidance_scale: f32,
        shift: f32,
        cfg_mode: CfgPredictionMode,
    ) -> FlowEulerTrace<B> {
        let device = noise.latent.device();
        let dtype: FloatDType = noise.latent.dtype().into();
        let mut sample = noise;
        let prepared_context = self.prepare_cfg_context(cond, guidance_scale, cfg_mode);
        let mut steps = vec![sample.clone()];
        let mut pred0 = None;
        let mut preds = Vec::new();
        for index in 0..prefix_steps.min(total_steps) {
            let schedule = flow_schedule_step(index, total_steps, shift);
            let pred = self.euler_cfg_prediction_with_prepared_context(
                sample.clone(),
                schedule.t_scaled,
                guidance_scale,
                &prepared_context,
                FlowPredictionContext {
                    dtype,
                    device: &device,
                },
            );
            if index == 0 {
                pred0 = Some(pred.clone());
            }
            preds.push(pred.clone());
            sample = sample.sub_scaled(pred, schedule.dt);
            steps.push(sample.clone());
        }
        FlowEulerTrace {
            pred0,
            preds,
            steps,
        }
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
        self.euler_cfg_prediction_at_step_with_mode(
            sample,
            cond,
            total_steps,
            step,
            guidance_scale,
            shift,
            CfgPredictionMode::Separate,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn euler_cfg_prediction_at_step_with_mode(
        &self,
        sample: FlowState<B>,
        cond: TripoSplatCondition<B>,
        total_steps: usize,
        step: usize,
        guidance_scale: f32,
        shift: f32,
        cfg_mode: CfgPredictionMode,
    ) -> FlowState<B> {
        let device = sample.latent.device();
        let dtype: FloatDType = sample.latent.dtype().into();
        let neg_cond = cond.zeros_like();
        let schedule = flow_schedule_step(step, total_steps, shift);
        self.euler_cfg_prediction(
            sample,
            schedule.t_scaled,
            cond,
            neg_cond,
            guidance_scale,
            cfg_mode,
            FlowPredictionContext {
                dtype,
                device: &device,
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn profile_euler_cfg_prediction_at_step_with_mode(
        &self,
        sample: FlowState<B>,
        cond: TripoSplatCondition<B>,
        total_steps: usize,
        step: usize,
        guidance_scale: f32,
        shift: f32,
        cfg_mode: CfgPredictionMode,
    ) -> FlowPredictionProfile<B> {
        let query_chunk_tokens = default_attention_query_chunk_tokens(sample.latent.dtype().into());
        self.profile_euler_cfg_prediction_at_step_with_query_chunk_tokens(
            sample,
            cond,
            total_steps,
            step,
            guidance_scale,
            shift,
            cfg_mode,
            query_chunk_tokens,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn profile_euler_cfg_prediction_at_step_with_mode_and_qkv_capture(
        &self,
        sample: FlowState<B>,
        cond: TripoSplatCondition<B>,
        total_steps: usize,
        step: usize,
        guidance_scale: f32,
        shift: f32,
        cfg_mode: CfgPredictionMode,
        attention_label_filter: impl Into<String>,
    ) -> FlowPredictionQkvProfile<B> {
        let query_chunk_tokens = default_attention_query_chunk_tokens(sample.latent.dtype().into());
        self.profile_euler_cfg_prediction_at_step_with_qkv_capture(
            sample,
            cond,
            total_steps,
            step,
            guidance_scale,
            shift,
            cfg_mode,
            query_chunk_tokens,
            attention_label_filter,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn profile_euler_cfg_prediction_at_step_with_query_chunk_tokens(
        &self,
        sample: FlowState<B>,
        cond: TripoSplatCondition<B>,
        total_steps: usize,
        step: usize,
        guidance_scale: f32,
        shift: f32,
        cfg_mode: CfgPredictionMode,
        query_chunk_tokens: usize,
    ) -> FlowPredictionProfile<B> {
        let device = sample.latent.device();
        let dtype: FloatDType = sample.latent.dtype().into();
        let batch = sample.latent.dims()[0];
        let mut records = Vec::new();
        let prep_start = Instant::now();
        let prepared_context = self.prepare_cfg_context(cond, guidance_scale, cfg_mode);
        push_profile_record(
            &mut records,
            "cfg.prepare_context",
            batch,
            0,
            self.config.model_channels,
            sync_elapsed_ms::<B>(&device, prep_start),
        );
        let schedule = flow_schedule_step(step, total_steps, shift);
        let output = self.profile_euler_cfg_prediction_with_prepared_context(
            sample,
            schedule.t_scaled,
            guidance_scale,
            &prepared_context,
            FlowPredictionContext {
                dtype,
                device: &device,
            },
            query_chunk_tokens,
            &mut records,
            None,
        );
        FlowPredictionProfile { output, records }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn profile_euler_cfg_prediction_at_step_with_qkv_capture(
        &self,
        sample: FlowState<B>,
        cond: TripoSplatCondition<B>,
        total_steps: usize,
        step: usize,
        guidance_scale: f32,
        shift: f32,
        cfg_mode: CfgPredictionMode,
        query_chunk_tokens: usize,
        attention_label_filter: impl Into<String>,
    ) -> FlowPredictionQkvProfile<B> {
        let device = sample.latent.device();
        let dtype: FloatDType = sample.latent.dtype().into();
        let batch = sample.latent.dims()[0];
        let mut records = Vec::new();
        let prep_start = Instant::now();
        let prepared_context = self.prepare_cfg_context(cond, guidance_scale, cfg_mode);
        push_profile_record(
            &mut records,
            "cfg.prepare_context",
            batch,
            0,
            self.config.model_channels,
            sync_elapsed_ms::<B>(&device, prep_start),
        );
        let schedule = flow_schedule_step(step, total_steps, shift);
        let mut qkv_capture = AttentionQkvCaptureState::new(attention_label_filter);
        let output = self.profile_euler_cfg_prediction_with_prepared_context(
            sample,
            schedule.t_scaled,
            guidance_scale,
            &prepared_context,
            FlowPredictionContext {
                dtype,
                device: &device,
            },
            query_chunk_tokens,
            &mut records,
            Some(&mut qkv_capture),
        );
        FlowPredictionQkvProfile {
            output,
            records,
            qkv_capture: qkv_capture.into_captured(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn euler_cfg_prediction(
        &self,
        sample: FlowState<B>,
        t_scaled: f32,
        cond: TripoSplatCondition<B>,
        neg_cond: TripoSplatCondition<B>,
        guidance_scale: f32,
        cfg_mode: CfgPredictionMode,
        context: FlowPredictionContext<'_, B>,
    ) -> FlowState<B> {
        match (guidance_scale > 1.0, cfg_mode) {
            (true, CfgPredictionMode::Batched) => self.euler_cfg_prediction_batched(
                sample,
                t_scaled,
                cond,
                neg_cond,
                guidance_scale,
                context,
            ),
            (true, CfgPredictionMode::BatchedMain) => {
                let cond = self.prepare_condition_context(cond);
                let neg = self.prepare_condition_context(neg_cond);
                let pos = self.prepare_latent_position(sample.latent.dims()[0]);
                self.euler_cfg_prediction_batched_main_with_context_optional_query_chunk_tokens(
                    sample,
                    t_scaled,
                    cond,
                    neg,
                    pos,
                    guidance_scale,
                    context,
                    None,
                )
            }
            (true, CfgPredictionMode::Separate) => self.euler_cfg_prediction_separate(
                sample,
                t_scaled,
                cond,
                neg_cond,
                guidance_scale,
                context,
            ),
            (false, _) => {
                let batch = sample.latent.dims()[0];
                let t_scaled = timestep_tensor(batch, t_scaled, context.device, context.dtype);
                let h_cond = self.prepare_condition_context(cond);
                let pos = self.prepare_latent_position(batch);
                self.forward_with_prepared_condition_context(sample, t_scaled, h_cond, pos)
            }
        }
    }

    #[allow(clippy::option_as_ref_deref, clippy::too_many_arguments)]
    fn profile_euler_cfg_prediction_with_prepared_context(
        &self,
        sample: FlowState<B>,
        t_scaled: f32,
        guidance_scale: f32,
        prepared_context: &PreparedCfgContext<B>,
        context: FlowPredictionContext<'_, B>,
        query_chunk_tokens: usize,
        records: &mut Vec<TripoSplatProfileRecord>,
        mut qkv_capture: Option<&mut AttentionQkvCaptureState<B>>,
    ) -> FlowState<B> {
        match prepared_context {
            PreparedCfgContext::Conditional { cond, pos } => {
                let batch = sample.latent.dims()[0];
                let t_scaled = timestep_tensor(batch, t_scaled, context.device, context.dtype);
                self.profile_forward_with_prepared_condition_context(
                    "cfg.conditional.forward",
                    sample,
                    t_scaled,
                    cond.clone(),
                    pos.clone(),
                    query_chunk_tokens,
                    records,
                    qkv_capture.as_mut().map(|capture| &mut **capture),
                )
            }
            PreparedCfgContext::Batched { cond, pos } => {
                let device = sample.latent.device();
                let batch = sample.latent.dims()[0];
                let concat_start = Instant::now();
                let sample = concat_flow_state_batch(sample);
                push_profile_record(
                    records,
                    "cfg.batched.concat_sample",
                    batch * 2,
                    sample.latent.dims()[1],
                    self.config.in_channels,
                    sync_elapsed_ms::<B>(&device, concat_start),
                );
                let pred = self.profile_forward_with_prepared_condition_context(
                    "cfg.batched.forward",
                    sample,
                    timestep_tensor(batch * 2, t_scaled, context.device, context.dtype),
                    cond.clone(),
                    pos.clone(),
                    query_chunk_tokens,
                    records,
                    qkv_capture.as_deref_mut(),
                );
                let split_start = Instant::now();
                let (pred, neg) = split_flow_state_batch(pred, batch);
                push_profile_record(
                    records,
                    "cfg.batched.split",
                    batch,
                    pred.latent.dims()[1],
                    self.config.out_channels,
                    sync_elapsed_ms::<B>(&device, split_start),
                );
                let blend_start = Instant::now();
                let out = blend_cfg_prediction(pred, neg, guidance_scale);
                push_profile_record(
                    records,
                    "cfg.batched.blend",
                    batch,
                    out.latent.dims()[1],
                    self.config.out_channels,
                    sync_elapsed_ms::<B>(&device, blend_start),
                );
                out
            }
            PreparedCfgContext::BatchedMain { cond, neg, pos } => {
                let device = sample.latent.device();
                let batch = sample.latent.dims()[0];
                let t_scaled = timestep_tensor(batch, t_scaled, context.device, context.dtype);
                let prefix_start = Instant::now();
                let prefix = self.prepare_forward_prefix_optional_query_chunk_tokens(
                    sample,
                    t_scaled,
                    pos.clone(),
                    Some(query_chunk_tokens),
                );
                push_profile_record(
                    records,
                    "cfg.batched_main.prefix",
                    batch,
                    prefix.latent_tokens,
                    self.config.model_channels,
                    sync_elapsed_ms::<B>(&device, prefix_start),
                );
                let main_start = Instant::now();
                let (pred, neg) = self
                    .profile_forward_main_batched_with_prepared_prefix_optional_query_chunk_tokens(
                        "cfg.batched_main.main_forward",
                        prefix,
                        cond.clone(),
                        neg.clone(),
                        query_chunk_tokens,
                        records,
                        qkv_capture,
                    );
                push_profile_record(
                    records,
                    "cfg.batched_main.main",
                    batch * 2,
                    pred.latent.dims()[1],
                    self.config.out_channels,
                    sync_elapsed_ms::<B>(&device, main_start),
                );
                let blend_start = Instant::now();
                let out = blend_cfg_prediction(pred, neg, guidance_scale);
                push_profile_record(
                    records,
                    "cfg.batched_main.blend",
                    batch,
                    out.latent.dims()[1],
                    self.config.out_channels,
                    sync_elapsed_ms::<B>(&device, blend_start),
                );
                out
            }
            PreparedCfgContext::Separate { cond, neg, pos } => {
                let device = sample.latent.device();
                let batch = sample.latent.dims()[0];
                let t_scaled = timestep_tensor(batch, t_scaled, context.device, context.dtype);
                let pred = self.profile_forward_with_prepared_condition_context(
                    "cfg.separate.cond_forward",
                    sample.clone(),
                    t_scaled.clone(),
                    cond.clone(),
                    pos.clone(),
                    query_chunk_tokens,
                    records,
                    qkv_capture.as_deref_mut(),
                );
                if guidance_scale > 1.0 {
                    let neg = self.profile_forward_with_prepared_condition_context(
                        "cfg.separate.neg_forward",
                        sample,
                        t_scaled,
                        neg.clone(),
                        pos.clone(),
                        query_chunk_tokens,
                        records,
                        qkv_capture.as_mut().map(|capture| &mut **capture),
                    );
                    let blend_start = Instant::now();
                    let out = blend_cfg_prediction(pred, neg, guidance_scale);
                    push_profile_record(
                        records,
                        "cfg.separate.blend",
                        batch,
                        out.latent.dims()[1],
                        self.config.out_channels,
                        sync_elapsed_ms::<B>(&device, blend_start),
                    );
                    out
                } else {
                    pred
                }
            }
        }
    }

    fn euler_cfg_prediction_with_prepared_context(
        &self,
        sample: FlowState<B>,
        t_scaled: f32,
        guidance_scale: f32,
        prepared_context: &PreparedCfgContext<B>,
        context: FlowPredictionContext<'_, B>,
    ) -> FlowState<B> {
        self.euler_cfg_prediction_with_prepared_context_optional_query_chunk_tokens(
            sample,
            t_scaled,
            guidance_scale,
            prepared_context,
            context,
            None,
        )
    }

    fn euler_cfg_prediction_with_prepared_context_optional_query_chunk_tokens(
        &self,
        sample: FlowState<B>,
        t_scaled: f32,
        guidance_scale: f32,
        prepared_context: &PreparedCfgContext<B>,
        context: FlowPredictionContext<'_, B>,
        query_chunk_tokens: Option<usize>,
    ) -> FlowState<B> {
        match prepared_context {
            PreparedCfgContext::Conditional { cond, pos } => {
                let batch = sample.latent.dims()[0];
                let t_scaled = timestep_tensor(batch, t_scaled, context.device, context.dtype);
                self.forward_with_prepared_condition_context_optional_query_chunk_tokens(
                    sample,
                    t_scaled,
                    cond.clone(),
                    pos.clone(),
                    query_chunk_tokens,
                )
            }
            PreparedCfgContext::Batched { cond, pos } => self
                .euler_cfg_prediction_batched_with_context_optional_query_chunk_tokens(
                    sample,
                    t_scaled,
                    cond.clone(),
                    pos.clone(),
                    guidance_scale,
                    context,
                    query_chunk_tokens,
                ),
            PreparedCfgContext::BatchedMain { cond, neg, pos } => self
                .euler_cfg_prediction_batched_main_with_context_optional_query_chunk_tokens(
                    sample,
                    t_scaled,
                    cond.clone(),
                    neg.clone(),
                    pos.clone(),
                    guidance_scale,
                    context,
                    query_chunk_tokens,
                ),
            PreparedCfgContext::Separate { cond, neg, pos } => self
                .euler_cfg_prediction_separate_with_context_optional_query_chunk_tokens(
                    sample,
                    t_scaled,
                    cond.clone(),
                    neg.clone(),
                    pos.clone(),
                    guidance_scale,
                    context,
                    query_chunk_tokens,
                ),
        }
    }

    fn euler_cfg_prediction_batched(
        &self,
        sample: FlowState<B>,
        t_scaled: f32,
        cond: TripoSplatCondition<B>,
        neg_cond: TripoSplatCondition<B>,
        guidance_scale: f32,
        context: FlowPredictionContext<'_, B>,
    ) -> FlowState<B> {
        let cond = self.prepare_condition_context(cond);
        let neg = self.prepare_condition_context(neg_cond);
        let pos = self.prepare_latent_position(sample.latent.dims()[0] * 2);
        self.euler_cfg_prediction_batched_with_context(
            sample,
            t_scaled,
            Tensor::cat(vec![cond, neg], 0),
            pos,
            guidance_scale,
            context,
        )
    }

    fn euler_cfg_prediction_batched_with_context(
        &self,
        sample: FlowState<B>,
        t_scaled: f32,
        cond: Tensor<B, 3>,
        pos: Tensor<B, 3>,
        guidance_scale: f32,
        context: FlowPredictionContext<'_, B>,
    ) -> FlowState<B> {
        self.euler_cfg_prediction_batched_with_context_optional_query_chunk_tokens(
            sample,
            t_scaled,
            cond,
            pos,
            guidance_scale,
            context,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn euler_cfg_prediction_batched_with_context_optional_query_chunk_tokens(
        &self,
        sample: FlowState<B>,
        t_scaled: f32,
        cond: Tensor<B, 3>,
        pos: Tensor<B, 3>,
        guidance_scale: f32,
        context: FlowPredictionContext<'_, B>,
        query_chunk_tokens: Option<usize>,
    ) -> FlowState<B> {
        let batch = sample.latent.dims()[0];
        let pred = self.forward_with_prepared_condition_context_optional_query_chunk_tokens(
            concat_flow_state_batch(sample),
            timestep_tensor(batch * 2, t_scaled, context.device, context.dtype),
            cond,
            pos,
            query_chunk_tokens,
        );
        let (pred, neg) = split_flow_state_batch(pred, batch);
        blend_cfg_prediction(pred, neg, guidance_scale)
    }

    #[allow(clippy::too_many_arguments)]
    fn euler_cfg_prediction_batched_main_with_context_optional_query_chunk_tokens(
        &self,
        sample: FlowState<B>,
        t_scaled: f32,
        cond: Tensor<B, 3>,
        neg: Tensor<B, 3>,
        pos: Tensor<B, 3>,
        guidance_scale: f32,
        context: FlowPredictionContext<'_, B>,
        query_chunk_tokens: Option<usize>,
    ) -> FlowState<B> {
        let batch = sample.latent.dims()[0];
        let t_scaled = timestep_tensor(batch, t_scaled, context.device, context.dtype);
        let prefix = self.prepare_forward_prefix_optional_query_chunk_tokens(
            sample,
            t_scaled,
            pos,
            query_chunk_tokens,
        );
        let (pred, neg) = self
            .forward_main_batched_with_prepared_prefix_optional_query_chunk_tokens(
                prefix,
                cond,
                neg,
                query_chunk_tokens,
            );
        blend_cfg_prediction(pred, neg, guidance_scale)
    }

    fn euler_cfg_prediction_separate(
        &self,
        sample: FlowState<B>,
        t_scaled: f32,
        cond: TripoSplatCondition<B>,
        neg_cond: TripoSplatCondition<B>,
        guidance_scale: f32,
        context: FlowPredictionContext<'_, B>,
    ) -> FlowState<B> {
        let cond = self.prepare_condition_context(cond);
        let neg = self.prepare_condition_context(neg_cond);
        let pos = self.prepare_latent_position(sample.latent.dims()[0]);
        self.euler_cfg_prediction_separate_with_context(
            sample,
            t_scaled,
            cond,
            neg,
            pos,
            guidance_scale,
            context,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn euler_cfg_prediction_separate_with_context(
        &self,
        sample: FlowState<B>,
        t_scaled: f32,
        cond: Tensor<B, 3>,
        neg: Tensor<B, 3>,
        pos: Tensor<B, 3>,
        guidance_scale: f32,
        context: FlowPredictionContext<'_, B>,
    ) -> FlowState<B> {
        self.euler_cfg_prediction_separate_with_context_optional_query_chunk_tokens(
            sample,
            t_scaled,
            cond,
            neg,
            pos,
            guidance_scale,
            context,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn euler_cfg_prediction_separate_with_context_optional_query_chunk_tokens(
        &self,
        sample: FlowState<B>,
        t_scaled: f32,
        cond: Tensor<B, 3>,
        neg: Tensor<B, 3>,
        pos: Tensor<B, 3>,
        guidance_scale: f32,
        context: FlowPredictionContext<'_, B>,
        query_chunk_tokens: Option<usize>,
    ) -> FlowState<B> {
        let batch = sample.latent.dims()[0];
        let t_scaled = timestep_tensor(batch, t_scaled, context.device, context.dtype);
        let prefix = self.prepare_forward_prefix_optional_query_chunk_tokens(
            sample,
            t_scaled,
            pos,
            query_chunk_tokens,
        );
        let pred = self.forward_main_with_prepared_prefix_optional_query_chunk_tokens(
            prefix.clone(),
            cond,
            query_chunk_tokens,
        );
        if guidance_scale > 1.0 {
            let neg = self.forward_main_with_prepared_prefix_optional_query_chunk_tokens(
                prefix,
                neg,
                query_chunk_tokens,
            );
            blend_cfg_prediction(pred, neg, guidance_scale)
        } else {
            pred
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct FlowScheduleStep {
    t_scaled: f32,
    dt: f32,
}

fn flow_schedule_step(index: usize, steps: usize, shift: f32) -> FlowScheduleStep {
    let t = shifted_t64(index, steps, shift);
    let t_next = shifted_t64(index + 1, steps, shift);
    FlowScheduleStep {
        t_scaled: (1000.0 * t) as f32,
        dt: (t - t_next) as f32,
    }
}

fn shifted_t64(index: usize, steps: usize, shift: f32) -> f64 {
    let steps = steps.max(1) as f64;
    let base = 1.0 - index as f64 / steps;
    let shift = shift as f64;
    shift * base / (1.0 + (shift - 1.0) * base)
}

fn timestep_tensor<B: Backend>(
    batch: usize,
    t_scaled: f32,
    device: &B::Device,
    _dtype: FloatDType,
) -> Tensor<B, 1> {
    Tensor::<B, 1>::from_floats(vec![t_scaled; batch].as_slice(), device)
}

fn concat_flow_state_batch<B: Backend>(sample: FlowState<B>) -> FlowState<B> {
    FlowState {
        latent: Tensor::cat(vec![sample.latent.clone(), sample.latent], 0),
        camera: sample
            .camera
            .map(|camera| Tensor::cat(vec![camera.clone(), camera], 0)),
    }
}

fn split_flow_state_batch<B: Backend>(
    state: FlowState<B>,
    batch: usize,
) -> (FlowState<B>, FlowState<B>) {
    let latent_channels = state.latent.dims()[2];
    let pred_latent =
        state
            .latent
            .clone()
            .slice([0..batch, 0..state.latent.dims()[1], 0..latent_channels]);
    let neg_latent = state.latent.slice([
        batch..batch * 2,
        0..pred_latent.dims()[1],
        0..latent_channels,
    ]);
    let (pred_camera, neg_camera) = match state.camera {
        Some(camera) => {
            let camera_tokens = camera.dims()[1];
            let camera_channels = camera.dims()[2];
            (
                Some(
                    camera
                        .clone()
                        .slice([0..batch, 0..camera_tokens, 0..camera_channels]),
                ),
                Some(camera.slice([batch..batch * 2, 0..camera_tokens, 0..camera_channels])),
            )
        }
        None => (None, None),
    };
    (
        FlowState {
            latent: pred_latent,
            camera: pred_camera,
        },
        FlowState {
            latent: neg_latent,
            camera: neg_camera,
        },
    )
}

fn blend_cfg_prediction<B: Backend>(
    pred: FlowState<B>,
    neg: FlowState<B>,
    guidance_scale: f32,
) -> FlowState<B> {
    FlowState {
        latent: pred.latent * guidance_scale - neg.latent * (guidance_scale - 1.0),
        camera: match (pred.camera, neg.camera) {
            (Some(pred), Some(neg)) => Some(pred * guidance_scale - neg * (guidance_scale - 1.0)),
            (pred, _) => pred,
        },
    }
}

fn layer_norm_last<B: Backend>(x: Tensor<B, 3>, epsilon: f64) -> Tensor<B, 3> {
    let dtype: FloatDType = x.dtype().into();
    let x_acc = cast_low_precision_to_f32(x, dtype);
    let (var, mean) = x_acc.clone().var_mean_bias(2);
    cast_from_f32_accum((x_acc - mean) / var.add_scalar(epsilon).sqrt(), dtype)
}

fn push_flow_trace_tensor3<B: Backend>(
    trace: &mut Vec<(String, Tensor<B, 3>)>,
    label: impl Into<String>,
    tensor: &Tensor<B, 3>,
    token_limit: usize,
) {
    let [batch, tokens, channels] = tensor.dims();
    let end = token_limit.max(1).min(tokens);
    let clipped = if end < tokens {
        tensor.clone().slice([0..batch, 0..end, 0..channels])
    } else {
        tensor.clone()
    };
    trace.push((label.into(), clipped));
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

fn cast_tensor_dtype<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    dtype: FloatDType,
) -> Tensor<B, D> {
    let current: FloatDType = tensor.dtype().into();
    if current == dtype {
        tensor
    } else {
        tensor.cast(dtype)
    }
}

const TORCH_SOBOL_SEED123_DIM3_COUNT: usize = 8192;
const TORCH_SOBOL_SEED123_DIM3_BYTES: &[u8] =
    include_bytes!("torch_sobol_seed123_dim3_8192_f32le.bin");
const TORCH_LATENT_POSITION_SEED123_COUNT: usize = 8192;
const TORCH_LATENT_POSITION_CHANNELS: usize = 1024;
const TORCH_LATENT_POSITION_SEED123_BYTES: &[u8] =
    include_bytes!("torch_latent_position_seed123_8192x1024_f32le.bin");

fn torch_sobol_seed123_dim3_positions<B: Backend>(
    count: usize,
    device: &B::Device,
) -> Param<Tensor<B, 2>> {
    assert!(
        count <= TORCH_SOBOL_SEED123_DIM3_COUNT,
        "TripoSplat q_token_length {count} exceeds canonical PyTorch Sobol table length {TORCH_SOBOL_SEED123_DIM3_COUNT}"
    );
    let (sobol_chunks, sobol_remainder) = TORCH_SOBOL_SEED123_DIM3_BYTES.as_chunks::<4>();
    assert!(
        sobol_remainder.is_empty(),
        "canonical TripoSplat Sobol byte table must contain whole f32 values"
    );
    let values = sobol_chunks
        .iter()
        .take(count * 3)
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect::<Vec<_>>();
    Param::from_tensor(
        Tensor::<B, 1>::from_data(TensorData::new(values, [count * 3]), (device, DType::F32))
            .reshape([count, 3]),
    )
}

fn canonical_latent_position_embedding<B: Backend>(
    count: usize,
    channels: usize,
    device: &B::Device,
    pos_pe: Tensor<B, 2>,
) -> Tensor<B, 2> {
    if channels == TORCH_LATENT_POSITION_CHANNELS {
        assert!(
            count <= TORCH_LATENT_POSITION_SEED123_COUNT,
            "TripoSplat q_token_length {count} exceeds canonical PyTorch latent position table length {TORCH_LATENT_POSITION_SEED123_COUNT}"
        );
        let (position_chunks, position_remainder) =
            TORCH_LATENT_POSITION_SEED123_BYTES.as_chunks::<4>();
        assert!(
            position_remainder.is_empty(),
            "canonical TripoSplat latent-position byte table must contain whole f32 values"
        );
        let values = position_chunks
            .iter()
            .take(count * channels)
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect::<Vec<_>>();
        Tensor::<B, 1>::from_data(
            TensorData::new(values, [count * channels]),
            (device, DType::F32),
        )
        .reshape([count, channels])
    } else {
        PcdAbsolutePositionEmbedder::legacy(channels).forward_2d(pos_pe.cast(FloatDType::F32))
    }
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
            rng_normals_consumed: 0,
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
    fn triposplat_config_defaults_to_upstream_fast_latent_tokens() {
        let config = LatentSeqMmFlowModelConfig::triposplat();
        assert_eq!(config.q_token_length, crate::DEFAULT_Q_TOKEN_LENGTH);
        assert_eq!(
            config.q_token_length,
            crate::TRIPOSPLAT_FLOW_LATENT_TOKEN_LENGTH
        );
    }

    #[test]
    fn condition_pads_feature2_prefix_to_match_dinov3_tokens() {
        let device = Default::default();
        let condition = TripoSplatCondition {
            feature1: Tensor::<TestBackend, 3>::ones([1, 9, 16], &device),
            feature2: Some(Tensor::ones([1, 4, 8], &device)),
            rng_normals_consumed: 0,
        }
        .with_prefix_padded_feature2();

        assert_eq!(condition.feature2.unwrap().dims(), [1, 9, 8]);
    }

    #[test]
    fn timestep_tensor_remains_f32_for_low_precision_flow_weights() {
        let device = Default::default();
        let t = timestep_tensor::<TestBackend>(2, 875.123_5, &device, FloatDType::F16);

        assert_eq!(FloatDType::from(t.dtype()), FloatDType::F32);
        let values = t
            .to_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("timestep values");
        assert_eq!(values, vec![875.123_5, 875.123_5]);
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
            0.090_251_33,
            0.581_110_54,
            0.407_743_7,
            0.770_040_63,
            0.246_292_67,
            0.640_196_2,
            0.633_601_9,
            0.765_746_65,
            0.040_996_697,
            0.437_628_75,
            0.436_666_1,
            0.757_763_15,
            0.335_330_84,
            0.919_189_63,
            0.531_123_6,
            0.530_821_44,
            0.252_420_6,
            0.298_671_2,
            0.878_680_47,
            0.732_732,
            0.898_079_93,
            0.198_408_59,
            0.063_887_745,
            0.181_313_54,
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

    #[test]
    fn batched_cfg_prediction_matches_separate_cfg_prediction() {
        let device = Default::default();
        let config = LatentSeqMmFlowModelConfig::tiny_for_tests();
        let model = config.clone().init::<TestBackend>(&device);
        let sample = FlowState::<TestBackend>::deterministic_standard_normal(
            &device,
            1,
            config.q_token_length,
            config.in_channels,
            config.cam_channels,
            7,
        );
        let cond = TripoSplatCondition {
            feature1: deterministic_standard_normal_3d(
                &mut SplitMix64::new(11),
                [1, 6, config.cond_channels],
                &device,
            ),
            feature2: config.cond2_channels.map(|channels| {
                deterministic_standard_normal_3d(
                    &mut SplitMix64::new(13),
                    [1, 4, channels],
                    &device,
                )
            }),
            rng_normals_consumed: 0,
        }
        .with_prefix_padded_feature2();
        let neg_cond = cond.zeros_like();
        let context = FlowPredictionContext {
            dtype: FloatDType::F32,
            device: &device,
        };

        let separate = model.euler_cfg_prediction_separate(
            sample.clone(),
            0.75,
            cond.clone(),
            neg_cond.clone(),
            3.0,
            context,
        );
        let batched = model.euler_cfg_prediction_batched(
            sample.clone(),
            0.75,
            cond.clone(),
            neg_cond.clone(),
            3.0,
            context,
        );
        let batched_main = model.euler_cfg_prediction(
            sample,
            0.75,
            cond,
            neg_cond,
            3.0,
            CfgPredictionMode::BatchedMain,
            context,
        );

        assert_tensor_close("latent", batched.latent, separate.latent.clone(), 1.0e-4);
        assert_tensor_close(
            "camera",
            batched.camera.expect("batched camera"),
            separate.camera.clone().expect("separate camera"),
            1.0e-4,
        );
        assert_tensor_close(
            "batched-main latent",
            batched_main.latent,
            separate.latent,
            1.0e-4,
        );
        assert_tensor_close(
            "batched-main camera",
            batched_main.camera.expect("batched-main camera"),
            separate.camera.expect("separate camera"),
            1.0e-4,
        );
    }

    #[test]
    fn cached_cfg_sampling_matches_stepwise_prediction() {
        let device = Default::default();
        let config = LatentSeqMmFlowModelConfig::tiny_for_tests();
        let model = config.clone().init::<TestBackend>(&device);
        let sample = FlowState::<TestBackend>::deterministic_standard_normal(
            &device,
            1,
            config.q_token_length,
            config.in_channels,
            config.cam_channels,
            29,
        );
        let cond = TripoSplatCondition {
            feature1: deterministic_standard_normal_3d(
                &mut SplitMix64::new(31),
                [1, 6, config.cond_channels],
                &device,
            ),
            feature2: config.cond2_channels.map(|channels| {
                deterministic_standard_normal_3d(
                    &mut SplitMix64::new(37),
                    [1, 4, channels],
                    &device,
                )
            }),
            rng_normals_consumed: 0,
        }
        .with_prefix_padded_feature2();
        let total_steps = 4;
        let prefix_steps = 3;
        let guidance_scale = 3.0;
        let shift = 3.0;

        for cfg_mode in [
            CfgPredictionMode::Batched,
            CfgPredictionMode::BatchedMain,
            CfgPredictionMode::Separate,
        ] {
            let neg_cond = cond.zeros_like();
            let context = FlowPredictionContext {
                dtype: FloatDType::F32,
                device: &device,
            };
            let mut expected = sample.clone();
            for index in 0..prefix_steps {
                let schedule = flow_schedule_step(index, total_steps, shift);
                let pred = model.euler_cfg_prediction(
                    expected.clone(),
                    schedule.t_scaled,
                    cond.clone(),
                    neg_cond.clone(),
                    guidance_scale,
                    cfg_mode,
                    context,
                );
                expected = expected.sub_scaled(pred, schedule.dt);
            }

            let cached = model.sample_euler_cfg_prefix_with_mode(
                sample.clone(),
                cond.clone(),
                total_steps,
                prefix_steps,
                guidance_scale,
                shift,
                cfg_mode,
            );
            assert_tensor_close(
                &format!("cached {cfg_mode:?} latent"),
                cached.latent,
                expected.latent,
                1.0e-6,
            );
            assert_tensor_close(
                &format!("cached {cfg_mode:?} camera"),
                cached.camera.expect("cached camera"),
                expected.camera.expect("expected camera"),
                1.0e-6,
            );
        }
    }

    #[test]
    fn flow_trace_matches_prefix_sampling() {
        let device = Default::default();
        let config = LatentSeqMmFlowModelConfig::tiny_for_tests();
        let model = config.clone().init::<TestBackend>(&device);
        let sample = FlowState::<TestBackend>::deterministic_standard_normal(
            &device,
            1,
            config.q_token_length,
            config.in_channels,
            config.cam_channels,
            17,
        );
        let cond = TripoSplatCondition {
            feature1: deterministic_standard_normal_3d(
                &mut SplitMix64::new(19),
                [1, 6, config.cond_channels],
                &device,
            ),
            feature2: config.cond2_channels.map(|channels| {
                deterministic_standard_normal_3d(
                    &mut SplitMix64::new(23),
                    [1, 4, channels],
                    &device,
                )
            }),
            rng_normals_consumed: 0,
        }
        .with_prefix_padded_feature2();

        let trace = model.sample_euler_cfg_trace_with_mode(
            sample.clone(),
            cond.clone(),
            4,
            3,
            3.0,
            3.0,
            CfgPredictionMode::Separate,
        );

        assert_eq!(trace.steps.len(), 4);
        assert!(trace.pred0.is_some());
        for prefix in 0..=3 {
            let expected = model.sample_euler_cfg_prefix_with_mode(
                sample.clone(),
                cond.clone(),
                4,
                prefix,
                3.0,
                3.0,
                CfgPredictionMode::Separate,
            );
            assert_tensor_close(
                &format!("latent prefix {prefix}"),
                trace.steps[prefix].latent.clone(),
                expected.latent,
                1.0e-6,
            );
            assert_tensor_close(
                &format!("camera prefix {prefix}"),
                trace.steps[prefix].camera.clone().expect("trace camera"),
                expected.camera.expect("expected camera"),
                1.0e-6,
            );
        }
    }

    fn assert_tensor_close<const D: usize>(
        label: &str,
        actual: Tensor<TestBackend, D>,
        expected: Tensor<TestBackend, D>,
        tolerance: f32,
    ) {
        let actual = actual
            .to_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("actual tensor data");
        let expected = expected
            .to_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("expected tensor data");
        assert_eq!(actual.len(), expected.len(), "{label} length mismatch");
        for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "{label} mismatch at {index}: actual={actual} expected={expected}"
            );
        }
    }
}

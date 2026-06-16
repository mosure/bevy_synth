use burn::{prelude::*, tensor::FloatDType};

use crate::{
    CfgPredictionMode, FlowEulerTrace, FlowState, LatentSeqMmFlowModel, OctreeGaussianDecoder,
    TripoSplatCondition, TripoSplatMultiRunOutput, TripoSplatOptions, TripoSplatRunOutput,
    normalize_num_gaussians,
};

pub struct TripoSplatRuntimeComponents<B: Backend> {
    pub dinov3: burn_dino::model::dinov3::DinoV3ViT<B>,
    pub flux2_vae_encoder: burn_flux::Flux2VaeEncoder<B>,
    pub flow: LatentSeqMmFlowModel<B>,
    pub decoder: OctreeGaussianDecoder<B>,
}

pub struct TripoSplatConditioningDiagnostics<B: Backend> {
    pub dinov3_raw: Tensor<B, 3>,
    pub feature1: Tensor<B, 3>,
    pub vae_mean: Tensor<B, 4>,
    pub vae_logvar: Tensor<B, 4>,
    pub feature2: Tensor<B, 3>,
}

impl<B: Backend> TripoSplatConditioningDiagnostics<B> {
    pub fn into_condition(self) -> TripoSplatCondition<B> {
        let rng_normals_consumed = tensor_element_count_3d_dims(self.feature2.dims());
        TripoSplatCondition {
            feature1: self.feature1,
            feature2: Some(self.feature2),
            rng_normals_consumed,
        }
        .with_prefix_padded_feature2()
    }
}

impl<B: Backend> TripoSplatRuntimeComponents<B> {
    pub fn encode_preprocessed_image(
        &self,
        image_rgb_0_1: Tensor<B, 4>,
        seed: u64,
    ) -> TripoSplatCondition<B> {
        let dtype = self.float_dtype();
        let image_rgb_0_1 = cast_float_tensor(image_rgb_0_1, dtype);
        let dinov3_image = normalize_dinov3_image(image_rgb_0_1.clone(), dtype);
        let feature1 = layer_norm_last(self.dinov3.forward(dinov3_image), 1.0e-5);
        let flux_image = image_rgb_0_1 * 2.0 - 1.0;
        let feature2 = self.flux2_vae_encoder.encode_with_seed(flux_image, seed);
        let rng_normals_consumed = tensor_element_count_3d_dims(feature2.dims());
        TripoSplatCondition {
            feature1,
            feature2: Some(feature2),
            rng_normals_consumed,
        }
        .with_prefix_padded_feature2()
    }

    pub fn encode_preprocessed_image_random(
        &self,
        image_rgb_0_1: Tensor<B, 4>,
    ) -> TripoSplatCondition<B> {
        let dtype = self.float_dtype();
        let image_rgb_0_1 = cast_float_tensor(image_rgb_0_1, dtype);
        let dinov3_image = normalize_dinov3_image(image_rgb_0_1.clone(), dtype);
        let feature1 = layer_norm_last(self.dinov3.forward(dinov3_image), 1.0e-5);
        let flux_image = image_rgb_0_1 * 2.0 - 1.0;
        let feature2 = self.flux2_vae_encoder.encode(flux_image, false);
        TripoSplatCondition {
            feature1,
            feature2: Some(feature2),
            rng_normals_consumed: 0,
        }
        .with_prefix_padded_feature2()
    }

    pub fn encode_preprocessed_image_with_vae_noise(
        &self,
        image_rgb_0_1: Tensor<B, 4>,
        vae_noise: Tensor<B, 4>,
    ) -> TripoSplatCondition<B> {
        self.conditioning_diagnostics_with_vae_noise(image_rgb_0_1, vae_noise)
            .into_condition()
    }

    pub fn conditioning_diagnostics_with_vae_noise(
        &self,
        image_rgb_0_1: Tensor<B, 4>,
        vae_noise: Tensor<B, 4>,
    ) -> TripoSplatConditioningDiagnostics<B> {
        let dtype = self.float_dtype();
        let image_rgb_0_1 = cast_float_tensor(image_rgb_0_1, dtype);
        let dinov3_image = normalize_dinov3_image(image_rgb_0_1.clone(), dtype);
        let dinov3_raw = self.dinov3.forward(dinov3_image);
        let feature1 = layer_norm_last(dinov3_raw.clone(), 1.0e-5);
        let flux_image = image_rgb_0_1 * 2.0 - 1.0;
        let (vae_mean, vae_logvar, feature2) = self
            .flux2_vae_encoder
            .encode_with_noise_diagnostics(flux_image, cast_float_tensor(vae_noise, dtype));
        TripoSplatConditioningDiagnostics {
            dinov3_raw,
            feature1,
            vae_mean,
            vae_logvar,
            feature2,
        }
    }

    pub fn sample_latent(
        &self,
        condition: TripoSplatCondition<B>,
        options: TripoSplatOptions,
    ) -> FlowState<B> {
        let device = condition.feature1.device();
        let dtype = self.float_dtype();
        let noise = cast_flow_state(
            FlowState::deterministic_standard_normal_after_skipping(
                &device,
                condition.feature1.dims()[0],
                self.flow.config().q_token_length,
                self.flow.config().in_channels,
                self.flow.config().cam_channels,
                options.seed,
                condition.rng_normals_consumed,
            ),
            dtype,
        );
        self.flow.sample_euler_cfg(
            noise,
            condition,
            options.steps,
            options.guidance_scale,
            options.shift,
        )
    }

    pub fn sample_latent_random(
        &self,
        condition: TripoSplatCondition<B>,
        options: TripoSplatOptions,
    ) -> FlowState<B> {
        let device = condition.feature1.device();
        let dtype = self.float_dtype();
        let noise = cast_flow_state(
            FlowState::random(
                &device,
                condition.feature1.dims()[0],
                self.flow.config().q_token_length,
                self.flow.config().in_channels,
                self.flow.config().cam_channels,
            ),
            dtype,
        );
        self.flow.sample_euler_cfg(
            noise,
            condition,
            options.steps,
            options.guidance_scale,
            options.shift,
        )
    }

    pub fn sample_latent_from_noise(
        &self,
        condition: TripoSplatCondition<B>,
        noise: FlowState<B>,
        options: TripoSplatOptions,
    ) -> FlowState<B> {
        self.sample_latent_from_noise_with_cfg_mode(
            condition,
            noise,
            options,
            CfgPredictionMode::Batched,
        )
    }

    pub fn sample_latent_from_noise_with_cfg_mode(
        &self,
        condition: TripoSplatCondition<B>,
        noise: FlowState<B>,
        options: TripoSplatOptions,
        cfg_mode: CfgPredictionMode,
    ) -> FlowState<B> {
        let dtype = self.float_dtype();
        self.flow.sample_euler_cfg_prefix_with_mode(
            cast_flow_state(noise, dtype),
            cast_condition(condition, dtype),
            options.steps,
            options.steps,
            options.guidance_scale,
            options.shift,
            cfg_mode,
        )
    }

    pub fn sample_latent_prefix_from_noise(
        &self,
        condition: TripoSplatCondition<B>,
        noise: FlowState<B>,
        options: TripoSplatOptions,
        prefix_steps: usize,
    ) -> FlowState<B> {
        self.sample_latent_prefix_from_noise_with_cfg_mode(
            condition,
            noise,
            options,
            prefix_steps,
            CfgPredictionMode::Batched,
        )
    }

    pub fn sample_latent_prefix_from_noise_with_cfg_mode(
        &self,
        condition: TripoSplatCondition<B>,
        noise: FlowState<B>,
        options: TripoSplatOptions,
        prefix_steps: usize,
        cfg_mode: CfgPredictionMode,
    ) -> FlowState<B> {
        let dtype = self.float_dtype();
        self.flow.sample_euler_cfg_prefix_with_mode(
            cast_flow_state(noise, dtype),
            cast_condition(condition, dtype),
            options.steps,
            prefix_steps,
            options.guidance_scale,
            options.shift,
            cfg_mode,
        )
    }

    pub fn sample_latent_trace_from_noise_with_cfg_mode(
        &self,
        condition: TripoSplatCondition<B>,
        noise: FlowState<B>,
        options: TripoSplatOptions,
        prefix_steps: usize,
        cfg_mode: CfgPredictionMode,
    ) -> FlowEulerTrace<B> {
        let dtype = self.float_dtype();
        self.flow.sample_euler_cfg_trace_with_mode(
            cast_flow_state(noise, dtype),
            cast_condition(condition, dtype),
            options.steps,
            prefix_steps,
            options.guidance_scale,
            options.shift,
            cfg_mode,
        )
    }

    pub fn flow_prediction_from_noise_at_step(
        &self,
        condition: TripoSplatCondition<B>,
        noise: FlowState<B>,
        options: TripoSplatOptions,
        step: usize,
    ) -> FlowState<B> {
        self.flow_prediction_from_noise_at_step_with_cfg_mode(
            condition,
            noise,
            options,
            step,
            CfgPredictionMode::Batched,
        )
    }

    pub fn flow_prediction_from_noise_at_step_with_cfg_mode(
        &self,
        condition: TripoSplatCondition<B>,
        noise: FlowState<B>,
        options: TripoSplatOptions,
        step: usize,
        cfg_mode: CfgPredictionMode,
    ) -> FlowState<B> {
        let dtype = self.float_dtype();
        self.flow.euler_cfg_prediction_at_step_with_mode(
            cast_flow_state(noise, dtype),
            cast_condition(condition, dtype),
            options.steps,
            step,
            options.guidance_scale,
            options.shift,
            cfg_mode,
        )
    }

    fn float_dtype(&self) -> FloatDType {
        self.dinov3.patch_embed.proj.weight.val().dtype().into()
    }

    pub fn decode_latent(
        &self,
        latent: Tensor<B, 3>,
        options: TripoSplatOptions,
    ) -> Result<TripoSplatRunOutput, String> {
        let num_gaussians = normalize_num_gaussians(options.num_gaussians)?;
        let splats = self
            .decoder
            .decode_to_cloud_with_seed(latent, num_gaussians, options.seed)?;
        Ok(TripoSplatRunOutput { splats, options })
    }

    pub async fn decode_latent_async(
        &self,
        latent: Tensor<B, 3>,
        options: TripoSplatOptions,
    ) -> Result<TripoSplatRunOutput, String> {
        let num_gaussians = normalize_num_gaussians(options.num_gaussians)?;
        let splats = self
            .decoder
            .decode_to_cloud_with_seed_async(latent, num_gaussians, options.seed)
            .await?;
        Ok(TripoSplatRunOutput { splats, options })
    }

    pub fn decode_latent_many<I>(
        &self,
        latent: Tensor<B, 3>,
        num_gaussians: I,
        options: TripoSplatOptions,
    ) -> Result<TripoSplatMultiRunOutput, String>
    where
        I: IntoIterator<Item = usize>,
    {
        let counts = num_gaussians
            .into_iter()
            .map(normalize_num_gaussians)
            .collect::<Result<Vec<_>, _>>()?;
        if counts.is_empty() {
            return Err("num_gaussians list must not be empty".to_string());
        }

        let mut splats = Vec::with_capacity(counts.len());
        let mut output_options = Vec::with_capacity(counts.len());
        for count in counts {
            let mut count_options = options;
            count_options.num_gaussians = count;
            splats.push(self.decoder.decode_to_cloud_with_seed(
                latent.clone(),
                count,
                options.seed,
            )?);
            output_options.push(count_options);
        }
        Ok(TripoSplatMultiRunOutput {
            splats,
            options: output_options,
        })
    }

    pub fn infer_preprocessed_image(
        &self,
        image_rgb_0_1: Tensor<B, 4>,
        options: TripoSplatOptions,
    ) -> Result<TripoSplatRunOutput, String> {
        let condition = self.encode_preprocessed_image(image_rgb_0_1, options.seed);
        let latent = self.sample_latent(condition, options).latent;
        self.decode_latent(latent, options)
    }

    pub async fn infer_preprocessed_image_async(
        &self,
        image_rgb_0_1: Tensor<B, 4>,
        options: TripoSplatOptions,
    ) -> Result<TripoSplatRunOutput, String> {
        let condition = self.encode_preprocessed_image(image_rgb_0_1, options.seed);
        let latent = self.sample_latent(condition, options).latent;
        self.decode_latent_async(latent, options).await
    }

    pub fn infer_preprocessed_image_many<I>(
        &self,
        image_rgb_0_1: Tensor<B, 4>,
        num_gaussians: I,
        options: TripoSplatOptions,
    ) -> Result<TripoSplatMultiRunOutput, String>
    where
        I: IntoIterator<Item = usize>,
    {
        let condition = self.encode_preprocessed_image(image_rgb_0_1, options.seed);
        let latent = self.sample_latent(condition, options).latent;
        self.decode_latent_many(latent, num_gaussians, options)
    }
}

fn normalize_dinov3_image<B: Backend>(image: Tensor<B, 4>, dtype: FloatDType) -> Tensor<B, 4> {
    let device = image.device();
    let mean = Tensor::<B, 1>::from_floats([0.485, 0.456, 0.406], &device)
        .cast(dtype)
        .reshape([1, 3, 1, 1]);
    let std = Tensor::<B, 1>::from_floats([0.229, 0.224, 0.225], &device)
        .cast(dtype)
        .reshape([1, 3, 1, 1]);
    (image - mean) / std
}

fn cast_flow_state<B: Backend>(state: FlowState<B>, dtype: FloatDType) -> FlowState<B> {
    FlowState {
        latent: cast_float_tensor(state.latent, dtype),
        camera: state.camera.map(|camera| cast_float_tensor(camera, dtype)),
    }
}

fn cast_condition<B: Backend>(
    condition: TripoSplatCondition<B>,
    dtype: FloatDType,
) -> TripoSplatCondition<B> {
    TripoSplatCondition {
        feature1: cast_float_tensor(condition.feature1, dtype),
        feature2: condition
            .feature2
            .map(|feature2| cast_float_tensor(feature2, dtype)),
        rng_normals_consumed: condition.rng_normals_consumed,
    }
}

fn tensor_element_count_3d_dims(dims: [usize; 3]) -> usize {
    dims[0].saturating_mul(dims[1]).saturating_mul(dims[2])
}

fn cast_float_tensor<B: Backend, const D: usize>(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ElasticGaussianFixedlenDecoderConfig, LatentSeqMmFlowModelConfig, OctreeGaussianDecoder,
        OctreeProbabilityFixedlenDecoderConfig,
    };

    type TestBackend = burn::backend::NdArray<f32>;

    #[test]
    fn tiny_runtime_infers_from_preprocessed_tensor() {
        let device = Default::default();
        let flow_config = LatentSeqMmFlowModelConfig {
            cond_channels: 64,
            cond2_channels: Some(128),
            ..LatentSeqMmFlowModelConfig::tiny_for_tests()
        };
        let components = TripoSplatRuntimeComponents::<TestBackend> {
            dinov3: burn_dino::model::dinov3::DinoV3Config::tiny_for_tests(32, 16).init(&device),
            flux2_vae_encoder: burn_flux::Flux2VaeEncoderConfig::flux2().init(&device),
            flow: flow_config.clone().init(&device),
            decoder: OctreeGaussianDecoder::new(
                &device,
                OctreeProbabilityFixedlenDecoderConfig::tiny_for_tests(),
                ElasticGaussianFixedlenDecoderConfig::tiny_for_tests(),
            ),
        };
        let image = Tensor::<TestBackend, 4>::zeros([1, 3, 32, 32], &device);
        let options = TripoSplatOptions {
            steps: 1,
            num_gaussians: 32_768,
            ..Default::default()
        };
        let out = components
            .infer_preprocessed_image(image, options)
            .expect("tiny runtime output");
        assert_eq!(out.splats.len(), normalize_num_gaussians(32_768).unwrap());
    }

    #[test]
    fn tiny_runtime_replays_one_latent_for_multiple_gaussian_counts() {
        let device = Default::default();
        let flow_config = LatentSeqMmFlowModelConfig {
            cond_channels: 64,
            cond2_channels: Some(128),
            ..LatentSeqMmFlowModelConfig::tiny_for_tests()
        };
        let components = TripoSplatRuntimeComponents::<TestBackend> {
            dinov3: burn_dino::model::dinov3::DinoV3Config::tiny_for_tests(32, 16).init(&device),
            flux2_vae_encoder: burn_flux::Flux2VaeEncoderConfig::flux2().init(&device),
            flow: flow_config.clone().init(&device),
            decoder: OctreeGaussianDecoder::new(
                &device,
                OctreeProbabilityFixedlenDecoderConfig::tiny_for_tests(),
                ElasticGaussianFixedlenDecoderConfig::tiny_for_tests(),
            ),
        };
        let image = Tensor::<TestBackend, 4>::zeros([1, 3, 32, 32], &device);
        let options = TripoSplatOptions {
            steps: 1,
            num_gaussians: 32_768,
            ..Default::default()
        };
        let out = components
            .infer_preprocessed_image_many(image, [32_768, 32_800], options)
            .expect("tiny multi-density runtime output");
        assert_eq!(out.splats.len(), 2);
        assert_eq!(out.options[0].num_gaussians, 32_768);
        assert_eq!(out.options[1].num_gaussians, 32_800);
        assert_eq!(out.splats[0].len(), 32_768);
        assert_eq!(out.splats[1].len(), 32_800);
    }

    #[test]
    fn tiny_runtime_conditioning_pads_flux_tokens_to_dino_prefix_contract() {
        let device = Default::default();
        let components = TripoSplatRuntimeComponents::<TestBackend> {
            dinov3: burn_dino::model::dinov3::DinoV3Config::tiny_for_tests(32, 16).init(&device),
            flux2_vae_encoder: burn_flux::Flux2VaeEncoderConfig::flux2().init(&device),
            flow: LatentSeqMmFlowModelConfig::tiny_for_tests().init(&device),
            decoder: OctreeGaussianDecoder::new(
                &device,
                OctreeProbabilityFixedlenDecoderConfig::tiny_for_tests(),
                ElasticGaussianFixedlenDecoderConfig::tiny_for_tests(),
            ),
        };
        let image = Tensor::<TestBackend, 4>::zeros([1, 3, 32, 32], &device);
        let condition = components.encode_preprocessed_image(image, 42);
        assert_eq!(condition.feature1.dims(), [1, 7, 64]);
        assert_eq!(condition.feature2.expect("feature2").dims(), [1, 7, 128]);
    }
}

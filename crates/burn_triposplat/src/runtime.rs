use std::time::Instant;

use burn::{prelude::*, tensor::FloatDType};
use serde::Serialize;

use crate::{
    CfgPredictionMode, FlowEulerTrace, FlowState, LatentSeqMmFlowModel, OctreeGaussianDecoder,
    TripoSplatCondition, TripoSplatDecodeReadbackStats, TripoSplatMultiRunOutput,
    TripoSplatOptions, TripoSplatRunOutput, normalize_num_gaussians,
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

#[derive(Clone, Debug, Default, Serialize)]
pub struct TripoSplatEncodeTiming {
    pub input_cast_ms: f64,
    pub dinov3_normalize_ms: f64,
    pub dinov3_forward_ms: f64,
    pub dinov3_layer_norm_ms: f64,
    pub flux_image_ms: f64,
    pub vae_encode_ms: f64,
    pub condition_pack_ms: f64,
    pub total_ms: f64,
    pub feature1_shape: Vec<usize>,
    pub feature2_shape: Vec<usize>,
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
        let dinov3_dtype = self.dinov3_float_dtype();
        let vae_dtype = self.flux_vae_float_dtype();
        let image_rgb_0_1 = cast_float_tensor(image_rgb_0_1, dinov3_dtype);
        let dinov3_image = normalize_dinov3_image(image_rgb_0_1.clone(), dinov3_dtype);
        let feature1 = layer_norm_last(self.dinov3.forward(dinov3_image), 1.0e-5);
        let flux_image = cast_float_tensor(image_rgb_0_1, vae_dtype) * 2.0 - 1.0;
        let feature2 = self.flux2_vae_encoder.encode_with_seed(flux_image, seed);
        let rng_normals_consumed = tensor_element_count_3d_dims(feature2.dims());
        TripoSplatCondition {
            feature1,
            feature2: Some(feature2),
            rng_normals_consumed,
        }
        .with_prefix_padded_feature2()
    }

    pub fn encode_preprocessed_image_with_timing(
        &self,
        image_rgb_0_1: Tensor<B, 4>,
        seed: u64,
    ) -> (TripoSplatCondition<B>, TripoSplatEncodeTiming) {
        let device = image_rgb_0_1.device();
        B::sync(&device).expect("TripoSplat encode timing pre-sync failed");
        let total_start = Instant::now();
        let mut timing = TripoSplatEncodeTiming::default();

        let dinov3_dtype = self.dinov3_float_dtype();
        let vae_dtype = self.flux_vae_float_dtype();
        let stage_start = Instant::now();
        let image_rgb_0_1 = cast_float_tensor(image_rgb_0_1, dinov3_dtype);
        timing.input_cast_ms = sync_elapsed_ms::<B>(&device, stage_start, "input cast");

        let stage_start = Instant::now();
        let dinov3_image = normalize_dinov3_image(image_rgb_0_1.clone(), dinov3_dtype);
        timing.dinov3_normalize_ms = sync_elapsed_ms::<B>(&device, stage_start, "DINOv3 normalize");

        let stage_start = Instant::now();
        let dinov3_raw = self.dinov3.forward(dinov3_image);
        timing.dinov3_forward_ms = sync_elapsed_ms::<B>(&device, stage_start, "DINOv3 forward");

        let stage_start = Instant::now();
        let feature1 = layer_norm_last(dinov3_raw, 1.0e-5);
        timing.dinov3_layer_norm_ms =
            sync_elapsed_ms::<B>(&device, stage_start, "DINOv3 layer norm");

        let stage_start = Instant::now();
        let flux_image = cast_float_tensor(image_rgb_0_1, vae_dtype) * 2.0 - 1.0;
        timing.flux_image_ms = sync_elapsed_ms::<B>(&device, stage_start, "Flux image prep");

        let stage_start = Instant::now();
        let feature2 = self.flux2_vae_encoder.encode_with_seed(flux_image, seed);
        timing.vae_encode_ms = sync_elapsed_ms::<B>(&device, stage_start, "Flux VAE encode");

        let stage_start = Instant::now();
        let rng_normals_consumed = tensor_element_count_3d_dims(feature2.dims());
        let condition = TripoSplatCondition {
            feature1,
            feature2: Some(feature2),
            rng_normals_consumed,
        }
        .with_prefix_padded_feature2();
        timing.condition_pack_ms = sync_elapsed_ms::<B>(&device, stage_start, "condition packing");

        timing.feature1_shape = condition.feature1.dims().to_vec();
        timing.feature2_shape = condition
            .feature2
            .as_ref()
            .map(|feature2| feature2.dims().to_vec())
            .unwrap_or_default();
        timing.total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
        (condition, timing)
    }

    pub fn encode_preprocessed_image_random(
        &self,
        image_rgb_0_1: Tensor<B, 4>,
    ) -> TripoSplatCondition<B> {
        let dinov3_dtype = self.dinov3_float_dtype();
        let vae_dtype = self.flux_vae_float_dtype();
        let image_rgb_0_1 = cast_float_tensor(image_rgb_0_1, dinov3_dtype);
        let dinov3_image = normalize_dinov3_image(image_rgb_0_1.clone(), dinov3_dtype);
        let feature1 = layer_norm_last(self.dinov3.forward(dinov3_image), 1.0e-5);
        let flux_image = cast_float_tensor(image_rgb_0_1, vae_dtype) * 2.0 - 1.0;
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
        let dinov3_dtype = self.dinov3_float_dtype();
        let vae_dtype = self.flux_vae_float_dtype();
        let image_rgb_0_1 = cast_float_tensor(image_rgb_0_1, dinov3_dtype);
        let dinov3_image = normalize_dinov3_image(image_rgb_0_1.clone(), dinov3_dtype);
        let dinov3_raw = self.dinov3.forward(dinov3_image);
        let feature1 = layer_norm_last(dinov3_raw.clone(), 1.0e-5);
        let flux_image = cast_float_tensor(image_rgb_0_1, vae_dtype) * 2.0 - 1.0;
        let (vae_mean, vae_logvar, feature2) = self
            .flux2_vae_encoder
            .encode_with_noise_diagnostics(flux_image, cast_float_tensor(vae_noise, vae_dtype));
        TripoSplatConditioningDiagnostics {
            dinov3_raw,
            feature1,
            vae_mean,
            vae_logvar,
            feature2,
        }
    }

    pub fn conditioning_diagnostics_with_vae_noise_timed(
        &self,
        image_rgb_0_1: Tensor<B, 4>,
        vae_noise: Tensor<B, 4>,
    ) -> (TripoSplatConditioningDiagnostics<B>, TripoSplatEncodeTiming) {
        let device = image_rgb_0_1.device();
        B::sync(&device).expect("TripoSplat encode timing pre-sync failed");
        let total_start = Instant::now();
        let mut timing = TripoSplatEncodeTiming::default();

        let dinov3_dtype = self.dinov3_float_dtype();
        let vae_dtype = self.flux_vae_float_dtype();
        let stage_start = Instant::now();
        let image_rgb_0_1 = cast_float_tensor(image_rgb_0_1, dinov3_dtype);
        let vae_noise = cast_float_tensor(vae_noise, vae_dtype);
        timing.input_cast_ms = sync_elapsed_ms::<B>(&device, stage_start, "input cast");

        let stage_start = Instant::now();
        let dinov3_image = normalize_dinov3_image(image_rgb_0_1.clone(), dinov3_dtype);
        timing.dinov3_normalize_ms = sync_elapsed_ms::<B>(&device, stage_start, "DINOv3 normalize");

        let stage_start = Instant::now();
        let dinov3_raw = self.dinov3.forward(dinov3_image);
        timing.dinov3_forward_ms = sync_elapsed_ms::<B>(&device, stage_start, "DINOv3 forward");

        let stage_start = Instant::now();
        let feature1 = layer_norm_last(dinov3_raw.clone(), 1.0e-5);
        timing.dinov3_layer_norm_ms =
            sync_elapsed_ms::<B>(&device, stage_start, "DINOv3 layer norm");

        let stage_start = Instant::now();
        let flux_image = cast_float_tensor(image_rgb_0_1, vae_dtype) * 2.0 - 1.0;
        timing.flux_image_ms = sync_elapsed_ms::<B>(&device, stage_start, "Flux image prep");

        let stage_start = Instant::now();
        let (vae_mean, vae_logvar, feature2) = self
            .flux2_vae_encoder
            .encode_with_noise_diagnostics(flux_image, vae_noise);
        timing.vae_encode_ms = sync_elapsed_ms::<B>(&device, stage_start, "Flux VAE encode");

        let stage_start = Instant::now();
        let diagnostics = TripoSplatConditioningDiagnostics {
            dinov3_raw,
            feature1,
            vae_mean,
            vae_logvar,
            feature2,
        };
        timing.condition_pack_ms = sync_elapsed_ms::<B>(&device, stage_start, "condition packing");

        timing.feature1_shape = diagnostics.feature1.dims().to_vec();
        timing.feature2_shape = diagnostics.feature2.dims().to_vec();
        timing.total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
        (diagnostics, timing)
    }

    pub fn sample_latent(
        &self,
        condition: TripoSplatCondition<B>,
        options: TripoSplatOptions,
    ) -> FlowState<B> {
        let device = condition.feature1.device();
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
            FloatDType::F32,
        );
        if let Some(query_chunk_tokens) = options.attention_query_chunk_tokens {
            self.flow
                .sample_euler_cfg_prefix_with_mode_and_query_chunk_tokens(
                    noise,
                    condition,
                    options.steps,
                    options.steps,
                    options.guidance_scale,
                    options.shift,
                    options.cfg_mode,
                    query_chunk_tokens,
                )
        } else {
            self.flow.sample_euler_cfg_prefix_with_mode(
                noise,
                condition,
                options.steps,
                options.steps,
                options.guidance_scale,
                options.shift,
                options.cfg_mode,
            )
        }
    }

    pub fn sample_latent_random(
        &self,
        condition: TripoSplatCondition<B>,
        options: TripoSplatOptions,
    ) -> FlowState<B> {
        let device = condition.feature1.device();
        let noise = cast_flow_state(
            FlowState::random(
                &device,
                condition.feature1.dims()[0],
                self.flow.config().q_token_length,
                self.flow.config().in_channels,
                self.flow.config().cam_channels,
            ),
            FloatDType::F32,
        );
        if let Some(query_chunk_tokens) = options.attention_query_chunk_tokens {
            self.flow
                .sample_euler_cfg_prefix_with_mode_and_query_chunk_tokens(
                    noise,
                    condition,
                    options.steps,
                    options.steps,
                    options.guidance_scale,
                    options.shift,
                    options.cfg_mode,
                    query_chunk_tokens,
                )
        } else {
            self.flow.sample_euler_cfg_prefix_with_mode(
                noise,
                condition,
                options.steps,
                options.steps,
                options.guidance_scale,
                options.shift,
                options.cfg_mode,
            )
        }
    }

    pub fn sample_latent_from_noise(
        &self,
        condition: TripoSplatCondition<B>,
        noise: FlowState<B>,
        options: TripoSplatOptions,
    ) -> FlowState<B> {
        self.sample_latent_from_noise_with_cfg_mode(condition, noise, options, options.cfg_mode)
    }

    pub fn sample_latent_from_noise_with_cfg_mode(
        &self,
        condition: TripoSplatCondition<B>,
        noise: FlowState<B>,
        options: TripoSplatOptions,
        cfg_mode: CfgPredictionMode,
    ) -> FlowState<B> {
        let noise = cast_flow_state(noise, FloatDType::F32);
        if let Some(query_chunk_tokens) = options.attention_query_chunk_tokens {
            self.flow
                .sample_euler_cfg_prefix_with_mode_and_query_chunk_tokens(
                    noise,
                    condition,
                    options.steps,
                    options.steps,
                    options.guidance_scale,
                    options.shift,
                    cfg_mode,
                    query_chunk_tokens,
                )
        } else {
            self.flow.sample_euler_cfg_prefix_with_mode(
                noise,
                condition,
                options.steps,
                options.steps,
                options.guidance_scale,
                options.shift,
                cfg_mode,
            )
        }
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
            options.cfg_mode,
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
        let noise = cast_flow_state(noise, FloatDType::F32);
        if let Some(query_chunk_tokens) = options.attention_query_chunk_tokens {
            self.flow
                .sample_euler_cfg_prefix_with_mode_and_query_chunk_tokens(
                    noise,
                    condition,
                    options.steps,
                    prefix_steps,
                    options.guidance_scale,
                    options.shift,
                    cfg_mode,
                    query_chunk_tokens,
                )
        } else {
            self.flow.sample_euler_cfg_prefix_with_mode(
                noise,
                condition,
                options.steps,
                prefix_steps,
                options.guidance_scale,
                options.shift,
                cfg_mode,
            )
        }
    }

    pub fn sample_latent_trace_from_noise_with_cfg_mode(
        &self,
        condition: TripoSplatCondition<B>,
        noise: FlowState<B>,
        options: TripoSplatOptions,
        prefix_steps: usize,
        cfg_mode: CfgPredictionMode,
    ) -> FlowEulerTrace<B> {
        self.flow.sample_euler_cfg_trace_with_mode(
            cast_flow_state(noise, FloatDType::F32),
            condition,
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
            CfgPredictionMode::Separate,
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
        self.flow.euler_cfg_prediction_at_step_with_mode(
            cast_flow_state(noise, FloatDType::F32),
            condition,
            options.steps,
            step,
            options.guidance_scale,
            options.shift,
            cfg_mode,
        )
    }

    fn dinov3_float_dtype(&self) -> FloatDType {
        self.dinov3.patch_embed.proj.weight.val().dtype().into()
    }

    fn flux_vae_float_dtype(&self) -> FloatDType {
        self.flux2_vae_encoder.float_dtype()
    }

    pub fn decode_latent(
        &self,
        latent: Tensor<B, 3>,
        options: TripoSplatOptions,
    ) -> Result<TripoSplatRunOutput, String> {
        let num_gaussians = normalize_num_gaussians(options.num_gaussians)?;
        let latent = cast_float_tensor(latent, self.decoder.float_dtype());
        let (splats, decode_readbacks) = self.decoder.decode_to_cloud_with_seed_readback_stats(
            latent,
            num_gaussians,
            options.seed,
        )?;
        Ok(TripoSplatRunOutput {
            splats,
            options,
            decode_readbacks,
        })
    }

    pub async fn decode_latent_async(
        &self,
        latent: Tensor<B, 3>,
        options: TripoSplatOptions,
    ) -> Result<TripoSplatRunOutput, String> {
        let num_gaussians = normalize_num_gaussians(options.num_gaussians)?;
        let latent = cast_float_tensor(latent, self.decoder.float_dtype());
        let (splats, decode_readbacks) = self
            .decoder
            .decode_to_cloud_with_seed_async_readback_stats(latent, num_gaussians, options.seed)
            .await?;
        Ok(TripoSplatRunOutput {
            splats,
            options,
            decode_readbacks,
        })
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
        let mut decode_readbacks = TripoSplatDecodeReadbackStats::default();
        let latent = cast_float_tensor(latent, self.decoder.float_dtype());
        for count in counts {
            let mut count_options = options;
            count_options.num_gaussians = count;
            let (cloud, readbacks) = self.decoder.decode_to_cloud_with_seed_readback_stats(
                latent.clone(),
                count,
                options.seed,
            )?;
            decode_readbacks.sync_readbacks = decode_readbacks
                .sync_readbacks
                .saturating_add(readbacks.sync_readbacks);
            decode_readbacks.async_readbacks = decode_readbacks
                .async_readbacks
                .saturating_add(readbacks.async_readbacks);
            decode_readbacks.bytes = decode_readbacks.bytes.saturating_add(readbacks.bytes);
            splats.push(cloud);
            output_options.push(count_options);
        }
        Ok(TripoSplatMultiRunOutput {
            splats,
            options: output_options,
            decode_readbacks,
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

fn tensor_element_count_3d_dims(dims: [usize; 3]) -> usize {
    dims[0].saturating_mul(dims[1]).saturating_mul(dims[2])
}

fn sync_elapsed_ms<B: Backend>(device: &B::Device, start: Instant, label: &str) -> f64 {
    B::sync(device).unwrap_or_else(|err| {
        panic!("TripoSplat encode timing sync failed after {label}: {err:?}")
    });
    start.elapsed().as_secs_f64() * 1000.0
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

    #[test]
    fn default_runtime_sampling_uses_batched_main_cfg_mode() {
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
        let condition = components.encode_preprocessed_image(image, 42);
        let noise = FlowState {
            latent: Tensor::<TestBackend, 3>::zeros(
                [1, flow_config.q_token_length, flow_config.in_channels],
                &device,
            ),
            camera: flow_config
                .cam_channels
                .map(|channels| Tensor::<TestBackend, 3>::zeros([1, 1, channels], &device)),
        };
        let options = TripoSplatOptions {
            steps: 2,
            num_gaussians: 32_768,
            ..Default::default()
        };
        assert_eq!(options.cfg_mode, CfgPredictionMode::BatchedMain);

        let default_sample =
            components.sample_latent_from_noise(condition.clone(), noise.clone(), options);
        let explicit_sample = components.sample_latent_from_noise_with_cfg_mode(
            condition,
            noise,
            options,
            CfgPredictionMode::BatchedMain,
        );

        assert_runtime_tensor_close("latent", default_sample.latent, explicit_sample.latent);
        assert_runtime_tensor_close(
            "camera",
            default_sample.camera.expect("default camera"),
            explicit_sample.camera.expect("explicit camera"),
        );
    }

    fn assert_runtime_tensor_close<const D: usize>(
        label: &str,
        actual: Tensor<TestBackend, D>,
        expected: Tensor<TestBackend, D>,
    ) {
        let actual = actual
            .to_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("actual tensor vec");
        let expected = expected
            .to_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("expected tensor vec");
        assert_eq!(actual.len(), expected.len(), "{label} length mismatch");
        for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() <= 1.0e-6,
                "{label}[{index}] mismatch: actual={actual} expected={expected}"
            );
        }
    }
}

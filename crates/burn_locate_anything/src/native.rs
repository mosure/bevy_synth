use std::path::Path;

use image::DynamicImage;
use serde::{Deserialize, Serialize};

use crate::config::LocateAnythingModelConfig;
use crate::language::QwenLanguageConfig;
use crate::tokenizer::{LocateAnythingPromptInputs, QwenTokenizer};
use crate::vision::{PreprocessedImagePatches, VisionConfig, preprocess_image_to_patches};
use crate::{DetectionQuery, LocateAnythingError, LocateAnythingResult};

#[derive(Clone, Debug, PartialEq)]
pub struct LocateAnythingNativeBatchInputs {
    pub image: PreprocessedImagePatches,
    pub prompts: Vec<LocateAnythingNativePrompt>,
    pub vision_config: VisionConfig,
    pub language_config: QwenLanguageConfig,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct LocateAnythingNativePrompt {
    pub query: DetectionQuery,
    pub prompt: LocateAnythingPromptInputs,
}

#[cfg(any(
    feature = "backend_ndarray",
    feature = "backend_wgpu",
    feature = "backend_cuda"
))]
pub mod burn_native {
    use burn::prelude::*;

    use super::*;
    use crate::assets::weight_file_for_tensor;
    use crate::decode::LocateAnythingTokenIds;
    use crate::decode::{
        DecodeMode, ParallelBoxDecodeConfig, ParallelPatternKind, decode_parallel_box_from_logits,
    };
    use crate::language::burn_language::{BurnQwenDecoder, BurnQwenHead, QwenLayerKvCache};
    use crate::language::{
        LOCATE_ANYTHING_MTP_FUTURE_TOKENS, QwenAttentionMaskMode, QwenTokenEmbeddingWeights,
        qwen_ar_step_input_ids, qwen_ar_step_position_ids, qwen_mtp_step_input_ids,
        qwen_mtp_step_position_ids,
    };
    use crate::projector::ProjectorWeights;
    use crate::projector::burn_projector::BurnProjector;
    use crate::vision::burn_vision::{BurnMoonVitEncoder, BurnPatchEmbed, patch_merger};
    use crate::vision::{MoonVitEncoderWeights, PatchEmbedWeights};

    #[derive(Debug)]
    pub struct BurnLocateAnythingVisionProjector<B: Backend> {
        patch_embed: BurnPatchEmbed<B>,
        encoder: BurnMoonVitEncoder<B>,
        projector: BurnProjector<B>,
        merge_kernel_size: [usize; 2],
    }

    impl<B: Backend> BurnLocateAnythingVisionProjector<B> {
        pub fn from_model_root(
            model_root: impl AsRef<Path>,
            model_config: &LocateAnythingModelConfig,
            device: &B::Device,
        ) -> LocateAnythingResult<Self> {
            let model_root = model_root.as_ref();
            let vision_path =
                weight_file_for_tensor(model_root, "vision_model.patch_embed.proj.weight")?;
            let projector_path = weight_file_for_tensor(model_root, "mlp1.0.weight")?;
            let patch_embed = BurnPatchEmbed::from_weights(
                PatchEmbedWeights::from_safetensors_file(&vision_path)?,
                device,
            );
            let encoder = BurnMoonVitEncoder::from_weights(
                MoonVitEncoderWeights::from_safetensors_file(
                    &vision_path,
                    model_config.vision_config.num_hidden_layers,
                    model_config.vision_config.num_attention_heads,
                )?,
                device,
            );
            let projector = BurnProjector::from_weights(
                ProjectorWeights::from_safetensors_file(&projector_path)?,
                device,
            );
            Ok(Self {
                patch_embed,
                encoder,
                projector,
                merge_kernel_size: model_config.vision_config.merge_kernel_size,
            })
        }

        pub fn forward_preprocessed(
            &self,
            image: &PreprocessedImagePatches,
            device: &B::Device,
        ) -> Tensor<B, 2> {
            let patches = Tensor::<B, 1>::from_floats(image.patches.as_slice(), device)
                .reshape(image.patch_shape);
            let hidden = self.patch_embed.forward(patches, &image.image_grid_hws);
            let hidden = self.encoder.forward(hidden, &image.image_grid_hws);
            let merged = patch_merger(hidden, &image.image_grid_hws, self.merge_kernel_size);
            self.projector.forward(merged)
        }
    }

    #[derive(Debug)]
    pub struct BurnLocateAnythingLanguageBridge<B: Backend> {
        head: BurnQwenHead<B>,
    }

    impl<B: Backend> BurnLocateAnythingLanguageBridge<B> {
        pub fn from_model_root(
            model_root: impl AsRef<Path>,
            device: &B::Device,
        ) -> LocateAnythingResult<Self> {
            let model_root = model_root.as_ref();
            let embedding_path =
                weight_file_for_tensor(model_root, "language_model.model.embed_tokens.weight")?;
            let head_path = weight_file_for_tensor(model_root, "language_model.lm_head.weight")?;
            let head = BurnQwenHead::from_weights(
                QwenTokenEmbeddingWeights::from_safetensors_files(&embedding_path, &head_path)?,
                device,
            );
            Ok(Self { head })
        }

        pub fn embed_prompt_with_image_features(
            &self,
            prompt: &LocateAnythingPromptInputs,
            image_features: Tensor<B, 2>,
            device: &B::Device,
        ) -> LocateAnythingResult<Tensor<B, 3>> {
            self.head.embed_token_ids_with_image_features(
                &prompt.input_ids,
                &prompt.image_token_positions,
                image_features,
                device,
            )
        }
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub enum BurnQwenStepMode {
        Autoregressive,
        MultiTokenPrediction,
    }

    #[derive(Debug)]
    pub struct BurnQwenStepOutput<B: Backend> {
        pub logits: Tensor<B, 3>,
        pub input_ids: Vec<u32>,
        pub position_ids: Vec<usize>,
    }

    pub struct BurnQwenStepRequest<'a, B: Backend> {
        pub generated_ids: &'a [u32],
        pub prompt: &'a LocateAnythingPromptInputs,
        pub image_features: Option<Tensor<B, 2>>,
        pub caches: Option<&'a mut [QwenLayerKvCache<B>]>,
        pub past_len: usize,
        pub mode: BurnQwenStepMode,
    }

    #[derive(Debug)]
    pub struct BurnLocateAnythingQwen<B: Backend> {
        decoder: BurnQwenDecoder<B>,
        token_ids: LocateAnythingTokenIds,
        max_cached_tokens: usize,
    }

    #[derive(Clone, Debug, PartialEq)]
    pub struct BurnQwenGenerationConfig {
        pub decode_mode: DecodeMode,
        pub max_new_tokens: usize,
        pub repetition_penalty: f32,
        pub top_p: Option<f32>,
        pub top_k: Option<usize>,
    }

    impl Default for BurnQwenGenerationConfig {
        fn default() -> Self {
            Self {
                decode_mode: DecodeMode::Hybrid,
                max_new_tokens: 8192,
                repetition_penalty: 1.1,
                top_p: Some(0.9),
                top_k: None,
            }
        }
    }

    #[derive(Clone, Debug, PartialEq)]
    pub struct BurnQwenGenerationOutput {
        pub token_ids: Vec<u32>,
        pub elapsed_ms: f64,
        pub used_batched_initial_step: bool,
    }

    struct BurnQwenGenerationState<B: Backend> {
        generated: Vec<u32>,
        caches: Vec<QwenLayerKvCache<B>>,
        past_len: usize,
        use_mtp: bool,
        first_step: bool,
    }

    impl<B: Backend> BurnLocateAnythingQwen<B> {
        pub fn from_model_root(
            model_root: impl AsRef<Path>,
            model_config: &LocateAnythingModelConfig,
            device: &B::Device,
        ) -> LocateAnythingResult<Self> {
            let model_root = model_root.as_ref();
            let language_config = QwenLanguageConfig::from_model_config(model_config);
            let decoder = BurnQwenDecoder::from_model_root(model_root, language_config, device)?;
            Ok(Self {
                decoder,
                token_ids: LocateAnythingTokenIds::from_model_config(model_config),
                max_cached_tokens: model_config.text_config.max_position_embeddings,
            })
        }

        pub fn new_cache(&self) -> Vec<QwenLayerKvCache<B>> {
            self.decoder.new_cache(self.max_cached_tokens)
        }

        pub fn step_logits(
            &self,
            request: BurnQwenStepRequest<'_, B>,
            device: &B::Device,
        ) -> LocateAnythingResult<BurnQwenStepOutput<B>> {
            if request.image_features.is_some() && request.past_len != 0 {
                return Err(LocateAnythingError::Config(
                    "image features may only be inserted on the first LocateAnything Qwen step"
                        .to_string(),
                ));
            }
            let (input_ids, position_ids) = match request.mode {
                BurnQwenStepMode::Autoregressive => (
                    qwen_ar_step_input_ids(request.generated_ids, request.past_len)?,
                    qwen_ar_step_position_ids(request.generated_ids.len(), request.past_len)?,
                ),
                BurnQwenStepMode::MultiTokenPrediction => (
                    qwen_mtp_step_input_ids(
                        request.generated_ids,
                        request.past_len,
                        self.token_ids.default_mask_token_id,
                        LOCATE_ANYTHING_MTP_FUTURE_TOKENS,
                    )?,
                    qwen_mtp_step_position_ids(
                        request.generated_ids.len(),
                        request.past_len,
                        LOCATE_ANYTHING_MTP_FUTURE_TOKENS,
                    )?,
                ),
            };
            if input_ids.len() != position_ids.len() {
                return Err(LocateAnythingError::Config(format!(
                    "LocateAnything Qwen step input/position length mismatch: {} ids vs {} positions",
                    input_ids.len(),
                    position_ids.len()
                )));
            }

            let input_embeds = if let Some(image_features) = request.image_features {
                self.decoder.embed_token_ids_with_image_features(
                    &input_ids,
                    &request.prompt.image_token_positions,
                    image_features,
                    device,
                )?
            } else {
                self.decoder.embed_token_ids(&input_ids, device)?
            };
            let mask_mode = match request.mode {
                BurnQwenStepMode::Autoregressive => QwenAttentionMaskMode::Causal,
                BurnQwenStepMode::MultiTokenPrediction => QwenAttentionMaskMode::MtpWindow {
                    window_tokens: LOCATE_ANYTHING_MTP_FUTURE_TOKENS,
                },
            };
            let hidden = self.decoder.forward_hidden_with_attention_mask_mode(
                input_embeds,
                request.caches,
                Some(&position_ids),
                mask_mode,
            );
            let output_tokens = match request.mode {
                BurnQwenStepMode::Autoregressive => 1,
                BurnQwenStepMode::MultiTokenPrediction => LOCATE_ANYTHING_MTP_FUTURE_TOKENS,
            };
            let [batch, hidden_tokens, hidden_size] = hidden.dims();
            let logits_hidden = hidden.slice([
                0..batch,
                (hidden_tokens - output_tokens)..hidden_tokens,
                0..hidden_size,
            ]);
            let logits = self.decoder.logits(logits_hidden);
            Ok(BurnQwenStepOutput {
                logits,
                input_ids,
                position_ids,
            })
        }

        pub fn truncate_caches(caches: &mut [QwenLayerKvCache<B>], len: usize) {
            for cache in caches {
                cache.truncate(len);
            }
        }

        pub fn generate_token_ids(
            &self,
            prompt: &LocateAnythingPromptInputs,
            image_features: Tensor<B, 2>,
            decode_mode: DecodeMode,
            max_new_tokens: usize,
            device: &B::Device,
        ) -> LocateAnythingResult<Vec<u32>> {
            self.generate_token_ids_with_config(
                prompt,
                image_features,
                BurnQwenGenerationConfig {
                    decode_mode,
                    max_new_tokens,
                    ..BurnQwenGenerationConfig::default()
                },
                device,
            )
        }

        pub fn generate_token_ids_with_config(
            &self,
            prompt: &LocateAnythingPromptInputs,
            image_features: Tensor<B, 2>,
            config: BurnQwenGenerationConfig,
            device: &B::Device,
        ) -> LocateAnythingResult<Vec<u32>> {
            let use_mtp = matches!(
                config.decode_mode,
                DecodeMode::ParallelBox | DecodeMode::Hybrid
            );
            self.continue_generate_token_ids_with_config(
                prompt,
                Some(image_features),
                config,
                BurnQwenGenerationState {
                    generated: prompt.input_ids.clone(),
                    caches: self.new_cache(),
                    past_len: 0,
                    use_mtp,
                    first_step: true,
                },
                device,
            )
        }

        pub fn generate_token_ids_batch_with_config(
            &self,
            prompts: &[&LocateAnythingPromptInputs],
            image_features: Tensor<B, 2>,
            config: BurnQwenGenerationConfig,
            device: &B::Device,
        ) -> LocateAnythingResult<Vec<BurnQwenGenerationOutput>> {
            if !self.can_share_initial_mtp_step(prompts, &config) {
                return prompts
                    .iter()
                    .map(|prompt| {
                        let started = std::time::Instant::now();
                        let token_ids = self.generate_token_ids_with_config(
                            prompt,
                            image_features.clone(),
                            config.clone(),
                            device,
                        )?;
                        Ok(BurnQwenGenerationOutput {
                            token_ids,
                            elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
                            used_batched_initial_step: false,
                        })
                    })
                    .collect();
            }

            let batch_started = std::time::Instant::now();
            let prompt_len = prompts[0].input_ids.len();
            let position_ids =
                qwen_mtp_step_position_ids(prompt_len, 0, LOCATE_ANYTHING_MTP_FUTURE_TOKENS)?;
            let mut batched_embeds = Vec::with_capacity(prompts.len());
            for prompt in prompts {
                let input_ids = qwen_mtp_step_input_ids(
                    &prompt.input_ids,
                    0,
                    self.token_ids.default_mask_token_id,
                    LOCATE_ANYTHING_MTP_FUTURE_TOKENS,
                )?;
                batched_embeds.push(self.decoder.embed_token_ids_with_image_features(
                    &input_ids,
                    &prompt.image_token_positions,
                    image_features.clone(),
                    device,
                )?);
            }
            let input_embeds = Tensor::cat(batched_embeds, 0);
            let mut batched_caches = self.new_cache();
            let hidden = self.decoder.forward_hidden_with_attention_mask_mode(
                input_embeds,
                Some(&mut batched_caches),
                Some(&position_ids),
                QwenAttentionMaskMode::MtpWindow {
                    window_tokens: LOCATE_ANYTHING_MTP_FUTURE_TOKENS,
                },
            );
            let [batch, hidden_tokens, hidden_size] = hidden.dims();
            let logits_hidden = hidden.slice([
                0..batch,
                (hidden_tokens - LOCATE_ANYTHING_MTP_FUTURE_TOKENS)..hidden_tokens,
                0..hidden_size,
            ]);
            let logits = self
                .decoder
                .logits(logits_hidden)
                .into_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .map_err(|err| {
                    LocateAnythingError::Runtime(format!(
                        "failed to read batched LocateAnything Qwen logits: {err}"
                    ))
                })?;
            let first_step_ms = batch_started.elapsed().as_secs_f64() * 1000.0;
            let first_step_share_ms = first_step_ms / prompts.len().max(1) as f64;
            let row_stride = LOCATE_ANYTHING_MTP_FUTURE_TOKENS * self.decoder.vocab_size();
            let pbd_config = ParallelBoxDecodeConfig {
                generation_mode: config.decode_mode,
                ..ParallelBoxDecodeConfig::default()
            };
            let mut outputs = Vec::with_capacity(prompts.len());
            for (row, prompt) in prompts.iter().enumerate() {
                let mut generated = prompt.input_ids.clone();
                let mut row_logits = logits[row * row_stride..(row + 1) * row_stride].to_vec();
                self.filter_generation_logits(&mut row_logits, &generated, &config);
                let (out_tokens, terminal, switch_to_mtp) = self
                    .decode_generation_step_from_logits(
                        &row_logits,
                        BurnQwenStepMode::MultiTokenPrediction,
                        &config,
                        &pbd_config,
                    )?;
                generated.extend(out_tokens);
                let mut elapsed_ms = first_step_share_ms;
                let token_ids = if terminal {
                    generated[prompt_len..].to_vec()
                } else {
                    let continue_started = std::time::Instant::now();
                    let use_mtp = match config.decode_mode {
                        DecodeMode::ParallelBox => true,
                        DecodeMode::Hybrid => switch_to_mtp,
                        DecodeMode::Autoregressive => false,
                    };
                    let token_ids = self.continue_generate_token_ids_with_config(
                        prompt,
                        None,
                        config.clone(),
                        BurnQwenGenerationState {
                            generated,
                            caches: split_batched_cache_row(&batched_caches, row, prompt_len),
                            past_len: prompt_len,
                            use_mtp,
                            first_step: false,
                        },
                        device,
                    )?;
                    elapsed_ms += continue_started.elapsed().as_secs_f64() * 1000.0;
                    token_ids
                };
                outputs.push(BurnQwenGenerationOutput {
                    token_ids,
                    elapsed_ms,
                    used_batched_initial_step: true,
                });
            }
            Ok(outputs)
        }

        fn can_share_initial_mtp_step(
            &self,
            prompts: &[&LocateAnythingPromptInputs],
            config: &BurnQwenGenerationConfig,
        ) -> bool {
            if prompts.len() <= 1
                || !matches!(
                    config.decode_mode,
                    DecodeMode::ParallelBox | DecodeMode::Hybrid
                )
            {
                return false;
            }
            let first = prompts[0];
            prompts.iter().all(|prompt| {
                prompt.input_ids.len() == first.input_ids.len()
                    && prompt.image_token_positions == first.image_token_positions
            })
        }

        fn continue_generate_token_ids_with_config(
            &self,
            prompt: &LocateAnythingPromptInputs,
            image_features: Option<Tensor<B, 2>>,
            config: BurnQwenGenerationConfig,
            mut state: BurnQwenGenerationState<B>,
            device: &B::Device,
        ) -> LocateAnythingResult<Vec<u32>> {
            let prompt_len = prompt.input_ids.len();
            let total_gen_length = prompt_len.saturating_add(config.max_new_tokens);
            let pbd_config = ParallelBoxDecodeConfig {
                generation_mode: config.decode_mode,
                ..ParallelBoxDecodeConfig::default()
            };

            while state.generated.len() < total_gen_length {
                let mode = if state.use_mtp {
                    BurnQwenStepMode::MultiTokenPrediction
                } else {
                    BurnQwenStepMode::Autoregressive
                };
                let step = self.step_logits(
                    BurnQwenStepRequest {
                        generated_ids: &state.generated,
                        prompt,
                        image_features: state.first_step.then(|| image_features.clone()).flatten(),
                        caches: Some(&mut state.caches),
                        past_len: state.past_len,
                        mode,
                    },
                    device,
                )?;
                Self::truncate_caches(&mut state.caches, state.generated.len());
                state.past_len = state
                    .caches
                    .first()
                    .map(QwenLayerKvCache::len)
                    .unwrap_or_default();
                let mut logits = step
                    .logits
                    .into_data()
                    .convert::<f32>()
                    .to_vec::<f32>()
                    .map_err(|err| {
                        LocateAnythingError::Runtime(format!(
                            "failed to read LocateAnything Qwen logits: {err}"
                        ))
                    })?;
                self.filter_generation_logits(&mut logits, &state.generated, &config);

                let (out_tokens, terminal, switch_to_mtp) =
                    self.decode_generation_step_from_logits(&logits, mode, &config, &pbd_config)?;
                state.generated.extend(out_tokens);
                state.first_step = false;
                if terminal {
                    break;
                }
                if matches!(config.decode_mode, DecodeMode::Hybrid) {
                    state.use_mtp = switch_to_mtp;
                }
            }
            Ok(state.generated[prompt_len..].to_vec())
        }

        fn filter_generation_logits(
            &self,
            logits: &mut [f32],
            generated: &[u32],
            config: &BurnQwenGenerationConfig,
        ) {
            apply_repetition_penalty(
                logits,
                self.decoder.vocab_size(),
                generated,
                config.repetition_penalty,
            );
            if let Some(top_p) = config.top_p {
                apply_top_p_filter(logits, self.decoder.vocab_size(), top_p);
            }
            if let Some(top_k) = config.top_k {
                apply_top_k_filter(logits, self.decoder.vocab_size(), top_k);
            }
        }

        fn decode_generation_step_from_logits(
            &self,
            logits: &[f32],
            mode: BurnQwenStepMode,
            config: &BurnQwenGenerationConfig,
            pbd_config: &ParallelBoxDecodeConfig,
        ) -> LocateAnythingResult<(Vec<u32>, bool, bool)> {
            match mode {
                BurnQwenStepMode::MultiTokenPrediction => {
                    let decoded = decode_parallel_box_from_logits(
                        logits,
                        LOCATE_ANYTHING_MTP_FUTURE_TOKENS,
                        self.decoder.vocab_size(),
                        &self.token_ids,
                        pbd_config,
                    )?;
                    let switch_to_mtp = !matches!(decoded.kind, ParallelPatternKind::ErrorBox);
                    Ok((decoded.tokens, decoded.is_terminal, switch_to_mtp))
                }
                BurnQwenStepMode::Autoregressive => {
                    let token = row_argmax(logits) as u32;
                    let terminal = token == self.token_ids.im_end_token_id;
                    let switch_to_mtp = matches!(config.decode_mode, DecodeMode::Hybrid)
                        && token == self.token_ids.box_end_token_id;
                    Ok((vec![token], terminal, switch_to_mtp))
                }
            }
        }
    }

    const TOP_P_FAST_CANDIDATE_LIMIT: usize = 256;

    #[derive(Clone, Copy, Debug)]
    struct TopPCandidate {
        index: usize,
        value: f32,
    }

    impl PartialEq for TopPCandidate {
        fn eq(&self, other: &Self) -> bool {
            self.index == other.index && self.value.to_bits() == other.value.to_bits()
        }
    }

    impl Eq for TopPCandidate {}

    impl PartialOrd for TopPCandidate {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for TopPCandidate {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.value
                .total_cmp(&other.value)
                .then_with(|| other.index.cmp(&self.index))
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct TopPWorstFirstCandidate(TopPCandidate);

    impl PartialEq for TopPWorstFirstCandidate {
        fn eq(&self, other: &Self) -> bool {
            self.0 == other.0
        }
    }

    impl Eq for TopPWorstFirstCandidate {}

    impl PartialOrd for TopPWorstFirstCandidate {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for TopPWorstFirstCandidate {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            other.0.cmp(&self.0)
        }
    }

    pub(super) fn apply_top_p_filter(logits: &mut [f32], vocab_size: usize, top_p: f32) {
        apply_top_p_filter_with_candidate_limit(
            logits,
            vocab_size,
            top_p,
            TOP_P_FAST_CANDIDATE_LIMIT,
        );
    }

    pub(super) fn apply_top_p_filter_with_candidate_limit(
        logits: &mut [f32],
        vocab_size: usize,
        top_p: f32,
        candidate_limit: usize,
    ) {
        use std::cmp::Ordering;
        use std::collections::BinaryHeap;

        if !(0.0..1.0).contains(&top_p) || vocab_size == 0 {
            return;
        }
        for row in logits.chunks_mut(vocab_size) {
            let max = row
                .iter()
                .copied()
                .filter(|value| value.is_finite())
                .fold(f32::NEG_INFINITY, f32::max);
            if !max.is_finite() {
                continue;
            }
            let mut sum = 0.0f32;
            for value in row.iter().copied() {
                if value.is_finite() {
                    sum += (value - max).exp();
                }
            }
            if !(sum > 0.0 && sum.is_finite()) {
                continue;
            }
            let kept = top_p_fast_kept_indices(row, max, sum, top_p, candidate_limit)
                .unwrap_or_else(|| top_p_exact_kept_indices(row, max, sum, top_p));
            let kept_values = kept
                .into_iter()
                .map(|index| (index, row[index]))
                .collect::<Vec<_>>();
            row.fill(f32::NEG_INFINITY);
            for (index, value) in kept_values {
                row[index] = value;
            }
        }

        fn top_p_fast_kept_indices(
            row: &[f32],
            max: f32,
            sum: f32,
            top_p: f32,
            candidate_limit: usize,
        ) -> Option<Vec<usize>> {
            if candidate_limit == 0 || candidate_limit >= row.len() {
                return None;
            }
            let mut heap = BinaryHeap::<TopPWorstFirstCandidate>::with_capacity(candidate_limit);
            for (index, value) in row.iter().copied().enumerate() {
                if !value.is_finite() {
                    continue;
                }
                let candidate = TopPCandidate { index, value };
                if heap.len() < candidate_limit {
                    heap.push(TopPWorstFirstCandidate(candidate));
                } else if let Some(worst) = heap.peek()
                    && candidate.cmp(&worst.0) == Ordering::Greater
                {
                    heap.pop();
                    heap.push(TopPWorstFirstCandidate(candidate));
                }
            }
            let mut candidates = heap
                .into_iter()
                .map(|candidate| candidate.0)
                .collect::<Vec<_>>();
            candidates.sort_by(|left, right| right.cmp(left));
            let mut cumulative = 0.0f32;
            let mut kept = Vec::new();
            for candidate in candidates {
                kept.push(candidate.index);
                cumulative += (candidate.value - max).exp() / sum;
                if cumulative > top_p {
                    return Some(kept);
                }
            }
            None
        }

        fn top_p_exact_kept_indices(row: &[f32], max: f32, sum: f32, top_p: f32) -> Vec<usize> {
            let mut heap = BinaryHeap::from(
                row.iter()
                    .copied()
                    .enumerate()
                    .filter_map(|(index, value)| {
                        value.is_finite().then_some(TopPCandidate { index, value })
                    })
                    .collect::<Vec<_>>(),
            );
            let mut cumulative = 0.0f32;
            let mut kept = Vec::new();
            while let Some(candidate) = heap.pop() {
                kept.push(candidate.index);
                cumulative += (candidate.value - max).exp() / sum;
                if cumulative > top_p {
                    break;
                }
            }
            kept
        }
    }

    pub(super) fn apply_top_k_filter(logits: &mut [f32], vocab_size: usize, top_k: usize) {
        if top_k == 0 || top_k >= vocab_size || vocab_size == 0 {
            return;
        }
        for row in logits.chunks_mut(vocab_size) {
            let mut sorted = row
                .iter()
                .copied()
                .enumerate()
                .collect::<Vec<(usize, f32)>>();
            sorted.sort_by(|(_, a), (_, b)| b.total_cmp(a));
            let threshold = sorted[top_k - 1].1;
            for value in row.iter_mut() {
                if *value < threshold {
                    *value = f32::NEG_INFINITY;
                }
            }
        }
    }

    fn apply_repetition_penalty(
        logits: &mut [f32],
        vocab_size: usize,
        generated_ids: &[u32],
        penalty: f32,
    ) {
        if penalty == 1.0 || vocab_size == 0 {
            return;
        }
        let mut seen = generated_ids.to_vec();
        seen.sort_unstable();
        seen.dedup();
        for row in logits.chunks_mut(vocab_size) {
            for &token in &seen {
                let index = token as usize;
                if let Some(logit) = row.get_mut(index) {
                    if *logit > 0.0 {
                        *logit /= penalty;
                    } else {
                        *logit *= penalty;
                    }
                }
            }
        }
    }

    fn row_argmax(row: &[f32]) -> usize {
        row.iter()
            .copied()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(index, _)| index)
            .unwrap_or_default()
    }

    fn split_batched_cache_row<B: Backend>(
        caches: &[QwenLayerKvCache<B>],
        row: usize,
        len: usize,
    ) -> Vec<QwenLayerKvCache<B>> {
        caches
            .iter()
            .map(|cache| {
                let key = cache.key.as_ref().map(|key| {
                    let [batch, heads, tokens, head_dim] = key.dims();
                    assert!(row < batch);
                    let len = len.min(tokens);
                    key.clone()
                        .slice([row..row + 1, 0..heads, 0..len, 0..head_dim])
                });
                let value = cache.value.as_ref().map(|value| {
                    let [batch, heads, tokens, head_dim] = value.dims();
                    assert!(row < batch);
                    let len = len.min(tokens);
                    value
                        .clone()
                        .slice([row..row + 1, 0..heads, 0..len, 0..head_dim])
                });
                QwenLayerKvCache {
                    key,
                    value,
                    max_cached_tokens: cache.max_cached_tokens,
                }
            })
            .collect()
    }
}

pub fn prepare_native_batch_inputs(
    model_root: impl AsRef<Path>,
    model_config: &LocateAnythingModelConfig,
    in_token_limit: u32,
    image: &DynamicImage,
    queries: &[DetectionQuery],
) -> LocateAnythingResult<LocateAnythingNativeBatchInputs> {
    if queries.is_empty() {
        return Err(LocateAnythingError::Config(
            "native LocateAnything preparation requires at least one query".to_string(),
        ));
    }
    let vision_config = VisionConfig::from_model_config(model_config, in_token_limit);
    let language_config = QwenLanguageConfig::from_model_config(model_config);
    let image = preprocess_image_to_patches(image, &vision_config)?;
    let tokenizer = QwenTokenizer::from_model_root(model_root)?;
    let prompts = queries
        .iter()
        .cloned()
        .map(|query| {
            let prompt = tokenizer.build_prompt_inputs(&query.query, &image.image_grid_hws)?;
            Ok(LocateAnythingNativePrompt { query, prompt })
        })
        .collect::<LocateAnythingResult<Vec<_>>>()?;
    Ok(LocateAnythingNativeBatchInputs {
        image,
        prompts,
        vision_config,
        language_config,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_batch_preparation_matches_reference_fixture_when_present() {
        let Some(root) = find_repo_root_for_test() else {
            eprintln!("skipping native prep fixture; repo root not found");
            return;
        };
        let model_root = root.join("assets/models/LocateAnything-3B");
        let Some(image_path) =
            std::env::var_os("LOCATE_ANYTHING_PARITY_IMAGE").map(std::path::PathBuf::from)
        else {
            eprintln!(
                "skipping native prep fixture; set LOCATE_ANYTHING_PARITY_IMAGE to the reference scene image"
            );
            return;
        };
        if !model_root.join("config.json").exists() || !image_path.exists() {
            eprintln!(
                "skipping native prep fixture; missing {} or {}",
                model_root.display(),
                image_path.display()
            );
            return;
        }
        let model_config = LocateAnythingModelConfig::from_model_root(&model_root).unwrap();
        let image = image::open(&image_path).unwrap();
        let prepared = prepare_native_batch_inputs(
            &model_root,
            &model_config,
            crate::vision::LOCATE_ANYTHING_SAFE_IN_TOKEN_LIMIT,
            &image,
            &[DetectionQuery {
                query: "conference table".to_string(),
                label_hint: None,
            }],
        )
        .unwrap();
        assert_eq!(prepared.image.patch_shape, [4300, 3, 14, 14]);
        assert_eq!(prepared.image.image_grid_hws, vec![[50, 86]]);
        assert_eq!(prepared.prompts.len(), 1);
        assert_eq!(prepared.prompts[0].prompt.image_context_tokens, 1075);
        assert_eq!(prepared.prompts[0].prompt.image_token_positions.len(), 1075);
        assert_eq!(prepared.language_config.vocab_size, 152_681);
        assert_eq!(prepared.language_config.num_layers, 36);
    }

    #[cfg(any(
        feature = "backend_ndarray",
        feature = "backend_wgpu",
        feature = "backend_cuda"
    ))]
    #[test]
    fn top_p_filter_keeps_first_token_past_threshold_like_upstream() {
        let mut logits: Vec<f32> = vec![5.0, 4.0, 3.0, 2.0];
        super::burn_native::apply_top_p_filter(&mut logits, 4, 0.9);
        assert!(logits[0].is_finite());
        assert!(logits[1].is_finite());
        assert!(logits[2].is_finite());
        assert!(logits[3].is_infinite() && logits[3].is_sign_negative());
    }

    #[cfg(any(
        feature = "backend_ndarray",
        feature = "backend_wgpu",
        feature = "backend_cuda"
    ))]
    #[test]
    fn top_p_filter_preserves_only_argmax_when_argmax_exceeds_threshold() {
        let mut logits: Vec<f32> = vec![10.0, 1.0, 0.0, -1.0];
        super::burn_native::apply_top_p_filter(&mut logits, 4, 0.9);
        assert!(logits[0].is_finite());
        assert!(logits[1].is_infinite() && logits[1].is_sign_negative());
        assert!(logits[2].is_infinite() && logits[2].is_sign_negative());
        assert!(logits[3].is_infinite() && logits[3].is_sign_negative());
    }

    #[cfg(any(
        feature = "backend_ndarray",
        feature = "backend_wgpu",
        feature = "backend_cuda"
    ))]
    #[test]
    fn top_p_filter_fast_limit_matches_exact_filter() {
        let mut exact: Vec<f32> = vec![8.0, 7.5, 6.0, 2.0, 1.0, 0.5, -1.0, -2.0];
        let mut fast = exact.clone();
        super::burn_native::apply_top_p_filter_with_candidate_limit(&mut exact, 8, 0.9, 0);
        super::burn_native::apply_top_p_filter_with_candidate_limit(&mut fast, 8, 0.9, 4);
        assert_eq!(fast, exact);

        let mut exact: Vec<f32> = vec![5.0, 4.9, 4.8, 4.7, 4.6, 4.5, 4.4, 4.3];
        let mut fallback = exact.clone();
        super::burn_native::apply_top_p_filter_with_candidate_limit(&mut exact, 8, 0.9, 0);
        super::burn_native::apply_top_p_filter_with_candidate_limit(&mut fallback, 8, 0.9, 2);
        assert_eq!(fallback, exact);
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

#[cfg(all(test, feature = "backend_wgpu", not(target_arch = "wasm32")))]
mod wgpu_full_tests {
    use burn::prelude::*;

    use super::burn_native::{BurnLocateAnythingQwen, BurnQwenStepMode, BurnQwenStepRequest};
    use super::*;
    use crate::config::LocateAnythingModelConfig;
    use crate::tensor_io::load_tensor_from_safetensors_file;
    use crate::vision::LOCATE_ANYTHING_SAFE_IN_TOKEN_LIMIT;

    #[test]
    fn qwen_first_mtp_step_matches_reference_selected_logits_when_enabled() {
        if std::env::var("LOCATE_ANYTHING_QWEN_FULL_FORWARD_PARITY").is_err() {
            eprintln!(
                "skipping: set LOCATE_ANYTHING_QWEN_FULL_FORWARD_PARITY=1 to run full WGPU Qwen selected-logit parity"
            );
            return;
        }
        let Some(root) = find_repo_root_for_test() else {
            eprintln!("skipping full Qwen parity; repo root not found");
            return;
        };
        let model_root = root.join("assets/models/LocateAnything-3B");
        let hooks_path = std::env::var("LOCATE_ANYTHING_QWEN_FULL_FORWARD_HOOKS")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| {
                root.join(
                    "tmp/runs/20260626T034651Z_locateanything_f32_qwen3forward_galaxy/hooks.safetensors",
                )
            });
        let Some(image_path) =
            std::env::var_os("LOCATE_ANYTHING_PARITY_IMAGE").map(std::path::PathBuf::from)
        else {
            eprintln!(
                "skipping full Qwen parity; set LOCATE_ANYTHING_PARITY_IMAGE to the reference scene image"
            );
            return;
        };
        if !model_root.join("config.json").exists() || !hooks_path.exists() || !image_path.exists()
        {
            eprintln!(
                "skipping full Qwen parity; missing {}, {}, or {}",
                model_root.display(),
                hooks_path.display(),
                image_path.display()
            );
            return;
        }

        let device = burn_wgpu::WgpuDevice::default();
        let model_config = LocateAnythingModelConfig::from_model_root(&model_root).unwrap();
        let image = image::open(&image_path).unwrap();
        let prepared = prepare_native_batch_inputs(
            &model_root,
            &model_config,
            LOCATE_ANYTHING_SAFE_IN_TOKEN_LIMIT,
            &image,
            &[DetectionQuery {
                query: "conference table".to_string(),
                label_hint: None,
            }],
        )
        .unwrap();
        let image_features =
            load_tensor_from_safetensors_file(&hooks_path, "projector.mlp1").unwrap();
        let selected_token_ids =
            load_tensor_from_safetensors_file(&hooks_path, "selected_token_ids").unwrap();
        let reference_generated =
            load_tensor_from_safetensors_file(&hooks_path, "generated_token_ids").unwrap();
        let qwen = BurnLocateAnythingQwen::<burn_wgpu::Wgpu<f32, i32, u32>>::from_model_root(
            &model_root,
            &model_config,
            &device,
        )
        .unwrap();
        let features = Tensor::<burn_wgpu::Wgpu<f32, i32, u32>, 1>::from_floats(
            image_features.data.as_slice(),
            &device,
        )
        .reshape([image_features.shape[0], image_features.shape[1]]);
        let selected = selected_token_ids
            .data
            .iter()
            .map(|value| *value as usize)
            .collect::<Vec<_>>();

        let expected_generated = reference_generated
            .data
            .iter()
            .map(|value| *value as u32)
            .collect::<Vec<_>>();
        let expected_chunks = [
            expected_generated[0..4].to_vec(),
            expected_generated[4..10].to_vec(),
            expected_generated[10..11].to_vec(),
        ];
        let mut generated_context = prepared.prompts[0].prompt.input_ids.clone();
        let mut caches = qwen.new_cache();
        let mut past_len = 0usize;
        for (forward_index, expected_chunk) in expected_chunks.iter().enumerate() {
            let step = qwen
                .step_logits(
                    BurnQwenStepRequest {
                        generated_ids: &generated_context,
                        prompt: &prepared.prompts[0].prompt,
                        image_features: (forward_index == 0).then(|| features.clone()),
                        caches: Some(&mut caches),
                        past_len,
                        mode: BurnQwenStepMode::MultiTokenPrediction,
                    },
                    &device,
                )
                .unwrap();
            let reference = load_tensor_from_safetensors_file(
                &hooks_path,
                &format!("language_forward_{forward_index:02}_tail_selected_logits"),
            )
            .unwrap();
            let [batch, rows, vocab] = step.logits.dims();
            let logits = step
                .logits
                .into_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .unwrap();
            assert_eq!([batch, rows], [1, 6]);
            assert_eq!(reference.shape, vec![1, rows, selected.len()]);
            let (max_abs, mean_abs, rms, count) =
                selected_logit_error(&logits, &reference.data, rows, vocab, &selected);
            eprintln!(
                "LocateAnything Qwen WGPU forward {forward_index:02} selected-logit parity: count={count} max_abs={max_abs:.6e} mean_abs={mean_abs:.6e} rms={rms:.6e}"
            );
            assert!(
                max_abs < 5.0e-2 && rms < 5.0e-3,
                "selected-logit parity failed at forward {forward_index}: max_abs={max_abs:.6e}, rms={rms:.6e}"
            );
            BurnLocateAnythingQwen::truncate_caches(&mut caches, generated_context.len());
            past_len = caches
                .first()
                .map(crate::language::burn_language::QwenLayerKvCache::len)
                .unwrap_or_default();
            generated_context.extend(expected_chunk);
        }

        let generated = qwen
            .generate_token_ids(
                &prepared.prompts[0].prompt,
                features,
                crate::decode::DecodeMode::Hybrid,
                128,
                &device,
            )
            .unwrap();
        assert_eq!(generated, expected_generated);
    }

    fn selected_logit_error(
        logits: &[f32],
        reference: &[f32],
        rows: usize,
        vocab: usize,
        selected: &[usize],
    ) -> (f32, f32, f32, usize) {
        let mut max_abs = 0.0f32;
        let mut sum_abs = 0.0f32;
        let mut sum_sq = 0.0f32;
        let mut count = 0usize;
        for row in 0..rows {
            for (col, &token_id) in selected.iter().enumerate() {
                let actual = logits[row * vocab + token_id];
                let expected = reference[row * selected.len() + col];
                let delta = (actual - expected).abs();
                max_abs = max_abs.max(delta);
                sum_abs += delta;
                sum_sq += delta * delta;
                count += 1;
            }
        }
        (
            max_abs,
            sum_abs / count as f32,
            (sum_sq / count as f32).sqrt(),
            count,
        )
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

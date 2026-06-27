use std::path::Path;

use crate::tensor_io::{LoadedTensorF32, load_all_tensors_from_safetensors_file};
use crate::{SegmentationError, SegmentationResult};

#[derive(Clone, Debug, PartialEq)]
pub struct SamMaskDecoderConfig {
    pub transformer_dim: usize,
    pub transformer_depth: usize,
    pub num_heads: usize,
    pub mlp_dim: usize,
    pub attention_downsample_rate: usize,
    pub num_multimask_outputs: usize,
    pub pred_obj_scores: bool,
    pub use_high_res_features: bool,
}

impl SamMaskDecoderConfig {
    pub fn sam2_1024() -> Self {
        Self {
            transformer_dim: 256,
            transformer_depth: 2,
            num_heads: 8,
            mlp_dim: 2048,
            attention_downsample_rate: 2,
            num_multimask_outputs: 3,
            pred_obj_scores: true,
            use_high_res_features: true,
        }
    }

    pub fn num_mask_tokens(&self) -> usize {
        self.num_multimask_outputs + 1
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SamMaskDecoderWeights {
    pub config: SamMaskDecoderConfig,
    pub tensors: Vec<(String, LoadedTensorF32)>,
}

impl SamMaskDecoderWeights {
    pub fn from_safetensors_file(path: impl AsRef<Path>) -> SegmentationResult<Self> {
        let tensors = load_all_tensors_from_safetensors_file(path.as_ref())?;
        let weights = Self {
            config: SamMaskDecoderConfig::sam2_1024(),
            tensors,
        };
        weights.validate()?;
        Ok(weights)
    }

    fn validate(&self) -> SegmentationResult<()> {
        let dim = self.config.transformer_dim;
        expect_shape(
            &self.tensors,
            "sam_mask_decoder.iou_token.weight",
            &[1, dim],
        )?;
        expect_shape(
            &self.tensors,
            "sam_mask_decoder.mask_tokens.weight",
            &[self.config.num_mask_tokens(), dim],
        )?;
        if self.config.pred_obj_scores {
            expect_shape(
                &self.tensors,
                "sam_mask_decoder.obj_score_token.weight",
                &[1, dim],
            )?;
        }
        expect_shape(
            &self.tensors,
            "sam_mask_decoder.output_upscaling.0.weight",
            &[dim, dim / 4, 2, 2],
        )?;
        expect_shape(
            &self.tensors,
            "sam_mask_decoder.output_upscaling.3.weight",
            &[dim / 4, dim / 8, 2, 2],
        )?;
        Ok(())
    }
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
pub mod burn_mask_decoder {
    use std::collections::HashMap;

    use burn::prelude::*;
    use burn::tensor::activation::{gelu, relu, sigmoid, softmax};
    use burn::tensor::module::{conv_transpose2d, conv2d};
    use burn::tensor::ops::{ConvOptions, ConvTransposeOptions};

    use super::*;

    #[derive(Debug)]
    pub struct BurnSamMaskDecoder<B: Backend> {
        pub weights: SamMaskDecoderWeights,
        tensors1: HashMap<String, Tensor<B, 1>>,
        tensors2: HashMap<String, Tensor<B, 2>>,
        tensors4: HashMap<String, Tensor<B, 4>>,
    }

    #[derive(Debug)]
    pub struct BurnSamMaskDecoderOutput<B: Backend> {
        pub low_res_masks: Tensor<B, 4>,
        pub iou_predictions: Tensor<B, 2>,
        pub sam_tokens: Tensor<B, 3>,
        pub object_score_logits: Tensor<B, 2>,
    }

    #[derive(Debug)]
    pub struct BurnSamMaskDecoderRawOutput<B: Backend> {
        pub decoder_tokens: Tensor<B, 3>,
        pub decoder_src_input: Tensor<B, 4>,
        pub decoder_hs: Tensor<B, 3>,
        pub decoder_src_tokens: Tensor<B, 3>,
        pub decoder_iou_token_out: Tensor<B, 2>,
        pub decoder_mask_tokens_out: Tensor<B, 3>,
        pub decoder_upscaled_embedding: Tensor<B, 4>,
        pub decoder_hyper_in: Tensor<B, 3>,
        pub decoder_all_masks: Tensor<B, 4>,
        pub decoder_all_iou_predictions: Tensor<B, 2>,
        pub decoder_all_object_score_logits: Tensor<B, 2>,
    }

    impl<B: Backend> BurnSamMaskDecoder<B> {
        pub fn from_weights(weights: SamMaskDecoderWeights, device: &B::Device) -> Self {
            let mut tensors1 = HashMap::new();
            let mut tensors2 = HashMap::new();
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
                tensors4,
            }
        }

        #[allow(clippy::too_many_arguments)]
        pub fn forward(
            &self,
            image_embeddings: Tensor<B, 4>,
            image_pe: Tensor<B, 4>,
            sparse_prompt_embeddings: Tensor<B, 3>,
            dense_prompt_embeddings: Tensor<B, 4>,
            high_res_features: [Tensor<B, 4>; 2],
            multimask_output: bool,
            repeat_image: bool,
        ) -> BurnSamMaskDecoderOutput<B> {
            let raw = self.predict_masks_raw(
                image_embeddings,
                image_pe,
                sparse_prompt_embeddings,
                dense_prompt_embeddings,
                high_res_features,
                repeat_image,
            );
            let [_batch, _mask_count, height, width] = raw.decoder_all_masks.dims();
            let [_batch_iou, _iou_count] = raw.decoder_all_iou_predictions.dims();
            if multimask_output {
                let num = self.weights.config.num_multimask_outputs;
                BurnSamMaskDecoderOutput {
                    low_res_masks: raw.decoder_all_masks.slice([
                        0.._batch,
                        1..1 + num,
                        0..height,
                        0..width,
                    ]),
                    iou_predictions: raw
                        .decoder_all_iou_predictions
                        .slice([0.._batch_iou, 1..1 + num]),
                    sam_tokens: raw.decoder_mask_tokens_out.slice([
                        0.._batch,
                        1..1 + num,
                        0..self.weights.config.transformer_dim,
                    ]),
                    object_score_logits: raw.decoder_all_object_score_logits,
                }
            } else {
                BurnSamMaskDecoderOutput {
                    low_res_masks: raw.decoder_all_masks.slice([
                        0.._batch,
                        0..1,
                        0..height,
                        0..width,
                    ]),
                    iou_predictions: raw.decoder_all_iou_predictions.slice([0.._batch_iou, 0..1]),
                    sam_tokens: raw.decoder_mask_tokens_out.slice([
                        0.._batch,
                        0..1,
                        0..self.weights.config.transformer_dim,
                    ]),
                    object_score_logits: raw.decoder_all_object_score_logits,
                }
            }
        }

        pub fn predict_masks_raw(
            &self,
            image_embeddings: Tensor<B, 4>,
            image_pe: Tensor<B, 4>,
            sparse_prompt_embeddings: Tensor<B, 3>,
            dense_prompt_embeddings: Tensor<B, 4>,
            high_res_features: [Tensor<B, 4>; 2],
            repeat_image: bool,
        ) -> BurnSamMaskDecoderRawOutput<B> {
            let [prompt_batch, _prompt_tokens, dim] = sparse_prompt_embeddings.dims();
            assert_eq!(dim, self.weights.config.transformer_dim);
            let mut output_tokens = Vec::new();
            if self.weights.config.pred_obj_scores {
                output_tokens.push(self.t2("sam_mask_decoder.obj_score_token.weight"));
            }
            output_tokens.push(self.t2("sam_mask_decoder.iou_token.weight"));
            output_tokens.push(self.t2("sam_mask_decoder.mask_tokens.weight"));
            let output_tokens = Tensor::cat(output_tokens, 0)
                .reshape([1, self.output_token_count(), dim])
                .repeat_dim(0, prompt_batch);
            let tokens = Tensor::cat(vec![output_tokens, sparse_prompt_embeddings], 1);

            let src = if repeat_image {
                image_embeddings.repeat_dim(0, prompt_batch)
            } else {
                let [image_batch, _, _, _] = image_embeddings.dims();
                assert_eq!(image_batch, prompt_batch);
                image_embeddings
            } + dense_prompt_embeddings;
            let decoder_src_input = src.clone();
            let decoder_tokens = tokens.clone();
            let pos_src = image_pe.repeat_dim(0, prompt_batch);
            let [batch, channels, height, width] = src.dims();
            let (hs, src_tokens) = self.transformer(src, pos_src, tokens);
            let shift = usize::from(self.weights.config.pred_obj_scores);
            let iou_token_out = hs.clone().slice([0..batch, shift..shift + 1, 0..dim]);
            let iou_token_out = iou_token_out.reshape([batch, dim]);
            let mask_tokens_out = hs.clone().slice([
                0..batch,
                shift + 1..shift + 1 + self.weights.config.num_mask_tokens(),
                0..dim,
            ]);

            let src = src_tokens
                .clone()
                .swap_dims(1, 2)
                .reshape([batch, channels, height, width]);
            let upscaled_embedding = self.upscale(src, high_res_features);
            let [batch, up_channels, up_h, up_w] = upscaled_embedding.dims();
            let mut hyper = Vec::with_capacity(self.weights.config.num_mask_tokens());
            for index in 0..self.weights.config.num_mask_tokens() {
                let token = mask_tokens_out
                    .clone()
                    .slice([0..batch, index..index + 1, 0..dim])
                    .reshape([batch, dim]);
                hyper.push(self.mlp2(
                    token,
                    &format!("sam_mask_decoder.output_hypernetworks_mlps.{index}"),
                    3,
                    false,
                ));
            }
            let hyper = Tensor::cat(
                hyper
                    .into_iter()
                    .map(|tensor| tensor.reshape([batch, 1, up_channels]))
                    .collect(),
                1,
            );
            let masks = hyper.clone().matmul(upscaled_embedding.clone().reshape([
                batch,
                up_channels,
                up_h * up_w,
            ]));
            let masks = masks.reshape([batch, self.weights.config.num_mask_tokens(), up_h, up_w]);
            let iou_pred = self.mlp2(
                iou_token_out.clone(),
                "sam_mask_decoder.iou_prediction_head",
                3,
                true,
            );
            let object_score_logits = if self.weights.config.pred_obj_scores {
                let obj = hs
                    .clone()
                    .slice([0..batch, 0..1, 0..dim])
                    .reshape([batch, dim]);
                self.mlp2(obj, "sam_mask_decoder.pred_obj_score_head", 3, false)
            } else {
                iou_pred
                    .clone()
                    .slice([0..batch, 0..1])
                    .mul_scalar(0.0)
                    .add_scalar(10.0)
            };
            BurnSamMaskDecoderRawOutput {
                decoder_tokens,
                decoder_src_input,
                decoder_hs: hs,
                decoder_src_tokens: src_tokens,
                decoder_iou_token_out: iou_token_out,
                decoder_mask_tokens_out: mask_tokens_out,
                decoder_upscaled_embedding: upscaled_embedding,
                decoder_hyper_in: hyper,
                decoder_all_masks: masks,
                decoder_all_iou_predictions: iou_pred,
                decoder_all_object_score_logits: object_score_logits,
            }
        }

        fn transformer(
            &self,
            image_embedding: Tensor<B, 4>,
            image_pe: Tensor<B, 4>,
            point_embedding: Tensor<B, 3>,
        ) -> (Tensor<B, 3>, Tensor<B, 3>) {
            let [batch, channels, height, width] = image_embedding.dims();
            let mut keys = image_embedding
                .reshape([batch, channels, height * width])
                .swap_dims(1, 2);
            let key_pe = image_pe
                .reshape([batch, channels, height * width])
                .swap_dims(1, 2);
            let mut queries = point_embedding.clone();
            for layer in 0..self.weights.config.transformer_depth {
                let prefix = format!("sam_mask_decoder.transformer.layers.{layer}");
                if layer == 0 {
                    queries = self.attention(
                        &format!("{prefix}.self_attn"),
                        queries.clone(),
                        queries.clone(),
                        queries.clone(),
                    );
                    queries = self.layer_norm3(queries, &format!("{prefix}.norm1"));
                } else {
                    let q = queries.clone() + point_embedding.clone();
                    let attn = self.attention(
                        &format!("{prefix}.self_attn"),
                        q.clone(),
                        q,
                        queries.clone(),
                    );
                    queries = self.layer_norm3(queries + attn, &format!("{prefix}.norm1"));
                }

                let q = queries.clone() + point_embedding.clone();
                let k = keys.clone() + key_pe.clone();
                let attn = self.attention(
                    &format!("{prefix}.cross_attn_token_to_image"),
                    q,
                    k,
                    keys.clone(),
                );
                queries = self.layer_norm3(queries + attn, &format!("{prefix}.norm2"));

                let mlp = self.mlp3(queries.clone(), &format!("{prefix}.mlp"), 2);
                queries = self.layer_norm3(queries + mlp, &format!("{prefix}.norm3"));

                let q = keys.clone() + key_pe.clone();
                let k = queries.clone() + point_embedding.clone();
                let attn = self.attention(
                    &format!("{prefix}.cross_attn_image_to_token"),
                    q,
                    k,
                    queries.clone(),
                );
                keys = self.layer_norm3(keys + attn, &format!("{prefix}.norm4"));
            }

            let q = queries.clone() + point_embedding;
            let k = keys.clone() + key_pe;
            let attn = self.attention(
                "sam_mask_decoder.transformer.final_attn_token_to_image",
                q,
                k,
                keys.clone(),
            );
            let queries = self.layer_norm3(
                queries + attn,
                "sam_mask_decoder.transformer.norm_final_attn",
            );
            (queries, keys)
        }

        fn attention(
            &self,
            prefix: &str,
            q: Tensor<B, 3>,
            k: Tensor<B, 3>,
            v: Tensor<B, 3>,
        ) -> Tensor<B, 3> {
            let [batch, q_tokens, dim] = q.dims();
            let [_batch_k, k_tokens, _k_dim] = k.dims();
            let internal = self.t1(&format!("{prefix}.q_proj.bias")).dims()[0];
            let heads = self.weights.config.num_heads;
            let head_dim = internal / heads;
            let q = self
                .linear3(q, &format!("{prefix}.q_proj"), internal)
                .reshape([batch, q_tokens, heads, head_dim])
                .permute([0, 2, 1, 3]);
            let k = self
                .linear3(k, &format!("{prefix}.k_proj"), internal)
                .reshape([batch, k_tokens, heads, head_dim])
                .permute([0, 2, 1, 3]);
            let v = self
                .linear3(v, &format!("{prefix}.v_proj"), internal)
                .reshape([batch, k_tokens, heads, head_dim])
                .permute([0, 2, 1, 3]);
            let scores = q
                .matmul(k.swap_dims(2, 3))
                .mul_scalar(1.0 / (head_dim as f64).sqrt());
            let attn = softmax(scores, 3);
            let out = attn
                .matmul(v)
                .permute([0, 2, 1, 3])
                .reshape([batch, q_tokens, internal]);
            self.linear3(out, &format!("{prefix}.out_proj"), dim)
        }

        fn upscale(&self, src: Tensor<B, 4>, high_res_features: [Tensor<B, 4>; 2]) -> Tensor<B, 4> {
            let dc1 = conv_transpose2d(
                src,
                self.t4("sam_mask_decoder.output_upscaling.0.weight"),
                Some(self.t1("sam_mask_decoder.output_upscaling.0.bias")),
                ConvTransposeOptions::new([2, 2], [0, 0], [0, 0], [1, 1], 1),
            );
            let up = self.layer_norm2d(
                dc1 + high_res_features[1].clone(),
                "sam_mask_decoder.output_upscaling.1",
            );
            let up = gelu(up);
            let dc2 = conv_transpose2d(
                up,
                self.t4("sam_mask_decoder.output_upscaling.3.weight"),
                Some(self.t1("sam_mask_decoder.output_upscaling.3.bias")),
                ConvTransposeOptions::new([2, 2], [0, 0], [0, 0], [1, 1], 1),
            );
            gelu(dc2 + high_res_features[0].clone())
        }

        pub fn project_high_res_s0(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
            self.project_high_res_feature(input, "sam_mask_decoder.conv_s0")
        }

        pub fn project_high_res_s1(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
            self.project_high_res_feature(input, "sam_mask_decoder.conv_s1")
        }

        fn project_high_res_feature(&self, input: Tensor<B, 4>, prefix: &str) -> Tensor<B, 4> {
            conv2d(
                input,
                self.t4(&format!("{prefix}.weight")),
                Some(self.t1(&format!("{prefix}.bias"))),
                ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
            )
        }

        fn mlp2(
            &self,
            mut x: Tensor<B, 2>,
            prefix: &str,
            layers: usize,
            sigmoid_output: bool,
        ) -> Tensor<B, 2> {
            for layer in 0..layers {
                let key = format!("{prefix}.layers.{layer}");
                let out_dim = self.t1(&format!("{key}.bias")).dims()[0];
                x = self.linear2(x, &key, out_dim);
                if layer + 1 < layers {
                    x = relu(x);
                }
            }
            if sigmoid_output { sigmoid(x) } else { x }
        }

        fn mlp3(&self, mut x: Tensor<B, 3>, prefix: &str, layers: usize) -> Tensor<B, 3> {
            for layer in 0..layers {
                let key = format!("{prefix}.layers.{layer}");
                let out_dim = self.t1(&format!("{key}.bias")).dims()[0];
                x = self.linear3(x, &key, out_dim);
                if layer + 1 < layers {
                    x = relu(x);
                }
            }
            x
        }

        fn linear2(&self, input: Tensor<B, 2>, prefix: &str, out_dim: usize) -> Tensor<B, 2> {
            let [_batch, _in_dim] = input.dims();
            input.matmul(self.t2(&format!("{prefix}.weight")).swap_dims(0, 1))
                + self.t1(&format!("{prefix}.bias")).reshape([1, out_dim])
        }

        fn linear3(&self, input: Tensor<B, 3>, prefix: &str, out_dim: usize) -> Tensor<B, 3> {
            let [batch, tokens, in_dim] = input.dims();
            let flat = input.reshape([batch * tokens, in_dim]);
            self.linear2(flat, prefix, out_dim)
                .reshape([batch, tokens, out_dim])
        }

        fn layer_norm3(&self, input: Tensor<B, 3>, prefix: &str) -> Tensor<B, 3> {
            let [_batch, _tokens, dim] = input.dims();
            let (var, mean) = input.clone().var_mean_bias(2);
            (input - mean) / var.add_scalar(1.0e-5).sqrt()
                * self.t1(&format!("{prefix}.weight")).reshape([1, 1, dim])
                + self.t1(&format!("{prefix}.bias")).reshape([1, 1, dim])
        }

        fn layer_norm2d(&self, input: Tensor<B, 4>, prefix: &str) -> Tensor<B, 4> {
            let [_batch, channels, _height, _width] = input.dims();
            let (var, mean) = input.clone().var_mean_bias(1);
            (input - mean) / var.add_scalar(1.0e-6).sqrt()
                * self
                    .t1(&format!("{prefix}.weight"))
                    .reshape([1, channels, 1, 1])
                + self
                    .t1(&format!("{prefix}.bias"))
                    .reshape([1, channels, 1, 1])
        }

        fn output_token_count(&self) -> usize {
            usize::from(self.weights.config.pred_obj_scores)
                + 1
                + self.weights.config.num_mask_tokens()
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
    fn mask_decoder_rejects_missing_required_weights() {
        let weights = SamMaskDecoderWeights {
            config: SamMaskDecoderConfig::sam2_1024(),
            tensors: Vec::new(),
        };
        assert!(weights.validate().is_err());
    }

    #[cfg(feature = "backend_wgpu")]
    #[test]
    fn sam2_mask_decoder_wgpu_matches_reference_hook() {
        use burn_mask_decoder::BurnSamMaskDecoder;

        let weights_path = std::env::var("SAM2_MASK_DECODER_WEIGHTS").unwrap_or_default();
        let reference_path = std::env::var("SAM2_REFERENCE_HOOK").unwrap_or_default();
        if weights_path.is_empty() || reference_path.is_empty() {
            eprintln!(
                "skipping: set SAM2_MASK_DECODER_WEIGHTS and SAM2_REFERENCE_HOOK to run WGPU SAM2 decoder parity"
            );
            return;
        }

        type B = burn_wgpu::Wgpu<f32, i32, u32>;
        let device = burn_wgpu::WgpuDevice::default();
        let weights = SamMaskDecoderWeights::from_safetensors_file(&weights_path).unwrap();
        let decoder = BurnSamMaskDecoder::<B>::from_weights(weights, &device);
        let reference = crate::tensor_io::load_required_tensors_from_safetensors_file(
            Path::new(&reference_path),
            &[
                "image_embed",
                "dense_pe",
                "sparse_prompt_embeddings",
                "dense_prompt_embeddings",
                "high_res_feat0",
                "high_res_feat1",
                "low_res_masks",
                "iou_predictions",
                "sam_tokens",
                "object_score_logits",
                "decoder_tokens",
                "decoder_src_input",
                "decoder_hs",
                "decoder_src_tokens",
                "decoder_iou_token_out",
                "decoder_mask_tokens_out",
                "decoder_upscaled_embedding",
                "decoder_hyper_in",
                "decoder_all_masks",
                "decoder_all_iou_predictions",
                "decoder_all_object_score_logits",
            ],
        )
        .unwrap();
        let image_embed = tensor_from_reference::<B>(&reference, "image_embed", &device)
            .reshape([1, 256, 64, 64]);
        let dense_pe =
            tensor_from_reference::<B>(&reference, "dense_pe", &device).reshape([1, 256, 64, 64]);
        let sparse_prompt_embeddings =
            tensor_from_reference::<B>(&reference, "sparse_prompt_embeddings", &device)
                .reshape([1, 3, 256]);
        let dense_prompt_embeddings =
            tensor_from_reference::<B>(&reference, "dense_prompt_embeddings", &device)
                .reshape([1, 256, 64, 64]);
        let high_res_feat0 = tensor_from_reference::<B>(&reference, "high_res_feat0", &device)
            .reshape([1, 32, 256, 256]);
        let high_res_feat1 = tensor_from_reference::<B>(&reference, "high_res_feat1", &device)
            .reshape([1, 64, 128, 128]);
        let raw = decoder.predict_masks_raw(
            image_embed,
            dense_pe,
            sparse_prompt_embeddings,
            dense_prompt_embeddings,
            [high_res_feat0, high_res_feat1],
            false,
        );
        compare_tensor3::<B>(
            "decoder_tokens",
            raw.decoder_tokens,
            &reference,
            [1, 9, 256],
            2.0e-5,
            2.0e-6,
        );
        compare_tensor4::<B>(
            "decoder_src_input",
            raw.decoder_src_input,
            &reference,
            [1, 256, 64, 64],
            2.0e-5,
            2.0e-6,
        );
        compare_tensor3::<B>(
            "decoder_hs",
            raw.decoder_hs,
            &reference,
            [1, 9, 256],
            2.0e-4,
            2.0e-5,
        );
        compare_tensor3::<B>(
            "decoder_src_tokens",
            raw.decoder_src_tokens,
            &reference,
            [1, 4096, 256],
            2.0e-4,
            2.0e-5,
        );
        compare_tensor2::<B>(
            "decoder_iou_token_out",
            raw.decoder_iou_token_out,
            &reference,
            [1, 256],
            2.0e-4,
            2.0e-5,
        );
        compare_tensor3::<B>(
            "decoder_mask_tokens_out",
            raw.decoder_mask_tokens_out,
            &reference,
            [1, 4, 256],
            2.0e-4,
            2.0e-5,
        );
        compare_tensor4::<B>(
            "decoder_upscaled_embedding",
            raw.decoder_upscaled_embedding,
            &reference,
            [1, 32, 256, 256],
            2.0e-3,
            1.0e-4,
        );
        compare_tensor3::<B>(
            "decoder_hyper_in",
            raw.decoder_hyper_in,
            &reference,
            [1, 4, 32],
            2.0e-4,
            2.0e-5,
        );

        compare_tensor2::<B>(
            "decoder_all_iou_predictions",
            raw.decoder_all_iou_predictions,
            &reference,
            [1, 4],
            2.0e-4,
            2.0e-5,
        );
        compare_tensor2::<B>(
            "decoder_all_object_score_logits",
            raw.decoder_all_object_score_logits,
            &reference,
            [1, 1],
            2.0e-4,
            2.0e-5,
        );
        compare_tensor4::<B>(
            "decoder_all_masks",
            raw.decoder_all_masks,
            &reference,
            [1, 4, 256, 256],
            1.0e-2,
            1.5e-3,
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
    fn compare_tensor2<B: burn::prelude::Backend>(
        key: &str,
        actual: Tensor<B, 2>,
        reference: &[(String, LoadedTensorF32)],
        shape: [usize; 2],
        max_threshold: f32,
        rms_threshold: f64,
    ) {
        let data = actual.into_data().convert::<f32>().to_vec::<f32>().unwrap();
        compare_flat(
            key,
            &data,
            reference,
            shape.as_slice(),
            max_threshold,
            rms_threshold,
        );
    }

    #[cfg(feature = "backend_wgpu")]
    fn compare_tensor3<B: burn::prelude::Backend>(
        key: &str,
        actual: Tensor<B, 3>,
        reference: &[(String, LoadedTensorF32)],
        shape: [usize; 3],
        max_threshold: f32,
        rms_threshold: f64,
    ) {
        let data = actual.into_data().convert::<f32>().to_vec::<f32>().unwrap();
        compare_flat(
            key,
            &data,
            reference,
            shape.as_slice(),
            max_threshold,
            rms_threshold,
        );
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
        compare_flat(
            key,
            &data,
            reference,
            shape.as_slice(),
            max_threshold,
            rms_threshold,
        );
    }

    #[cfg(feature = "backend_wgpu")]
    fn compare_flat(
        key: &str,
        actual: &[f32],
        reference: &[(String, LoadedTensorF32)],
        shape: &[usize],
        max_threshold: f32,
        rms_threshold: f64,
    ) {
        let expected = crate::tensor_io::find_tensor(reference, key).unwrap();
        assert_eq!(expected.shape, shape);
        assert_eq!(actual.len(), expected.data.len());
        let mut max_abs = 0.0f32;
        let mut sum_sq = 0.0f64;
        for (left, right) in actual.iter().zip(expected.data.iter()) {
            let delta = (left - right).abs();
            max_abs = max_abs.max(delta);
            sum_sq += (delta as f64) * (delta as f64);
        }
        let rms = (sum_sq / actual.len() as f64).sqrt();
        eprintln!("sam2_mask_decoder {key} max_abs={max_abs:.6e} rms={rms:.6e}");
        assert!(max_abs <= max_threshold, "{key} max_abs={max_abs}");
        assert!(rms <= rms_threshold, "{key} rms={rms}");
    }
}

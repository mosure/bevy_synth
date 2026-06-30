use serde::{Deserialize, Serialize};

use crate::config::LocateAnythingModelConfig;
use crate::tensor_io::{LoadedTensorF32, load_required_tensors_from_safetensors_file};
use crate::{LocateAnythingError, LocateAnythingResult};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct QwenLanguageConfig {
    pub model: String,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub intermediate_size: usize,
    pub vocab_size: usize,
}

impl Default for QwenLanguageConfig {
    fn default() -> Self {
        Self {
            model: "Qwen2.5-3B-Instruct".to_string(),
            hidden_size: 2048,
            num_layers: 36,
            num_attention_heads: 16,
            num_key_value_heads: 2,
            max_position_embeddings: 25_600,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1.0e-6,
            intermediate_size: 11_008,
            vocab_size: 153_600,
        }
    }
}

impl QwenLanguageConfig {
    pub fn from_model_config(config: &LocateAnythingModelConfig) -> Self {
        let text = &config.text_config;
        Self {
            model: text
                .architectures
                .first()
                .cloned()
                .unwrap_or_else(|| "Qwen2ForCausalLM".to_string()),
            hidden_size: text.hidden_size,
            num_layers: text.num_hidden_layers,
            num_attention_heads: text.num_attention_heads,
            num_key_value_heads: text.num_key_value_heads,
            max_position_embeddings: text.max_position_embeddings,
            rope_theta: text.rope_theta,
            rms_norm_eps: text.rms_norm_eps,
            intermediate_size: text.intermediate_size,
            vocab_size: text.vocab_size,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct KvCachePolicy {
    pub enabled: bool,
    pub max_cached_tokens: usize,
}

impl Default for KvCachePolicy {
    fn default() -> Self {
        Self {
            enabled: true,
            max_cached_tokens: 25_600,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum QwenAttentionMaskMode {
    #[default]
    Causal,
    MtpWindow {
        window_tokens: usize,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct QwenDecoderLayerWeights {
    pub layer_index: usize,
    pub config: QwenLanguageConfig,
    pub input_layernorm_weight: Vec<f32>,
    pub post_attention_layernorm_weight: Vec<f32>,
    pub q_proj_weight: Vec<f32>,
    pub q_proj_bias: Vec<f32>,
    pub k_proj_weight: Vec<f32>,
    pub k_proj_bias: Vec<f32>,
    pub v_proj_weight: Vec<f32>,
    pub v_proj_bias: Vec<f32>,
    pub o_proj_weight: Vec<f32>,
    pub gate_proj_weight: Vec<f32>,
    pub up_proj_weight: Vec<f32>,
    pub down_proj_weight: Vec<f32>,
}

impl QwenDecoderLayerWeights {
    pub fn from_safetensors_file(
        path: impl AsRef<std::path::Path>,
        layer_index: usize,
        config: QwenLanguageConfig,
    ) -> LocateAnythingResult<Self> {
        let prefix = format!("language_model.model.layers.{layer_index}");
        let keys = [
            format!("{prefix}.input_layernorm.weight"),
            format!("{prefix}.post_attention_layernorm.weight"),
            format!("{prefix}.self_attn.q_proj.weight"),
            format!("{prefix}.self_attn.q_proj.bias"),
            format!("{prefix}.self_attn.k_proj.weight"),
            format!("{prefix}.self_attn.k_proj.bias"),
            format!("{prefix}.self_attn.v_proj.weight"),
            format!("{prefix}.self_attn.v_proj.bias"),
            format!("{prefix}.self_attn.o_proj.weight"),
            format!("{prefix}.mlp.gate_proj.weight"),
            format!("{prefix}.mlp.up_proj.weight"),
            format!("{prefix}.mlp.down_proj.weight"),
        ];
        let key_refs = keys.iter().map(String::as_str).collect::<Vec<_>>();
        let tensors = load_required_tensors_from_safetensors_file(path.as_ref(), &key_refs)?;
        Self::from_loaded_tensors(layer_index, config, &tensors)
    }

    pub fn from_loaded_tensors(
        layer_index: usize,
        config: QwenLanguageConfig,
        tensors: &[(String, LoadedTensorF32)],
    ) -> LocateAnythingResult<Self> {
        let prefix = format!("language_model.model.layers.{layer_index}");
        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;
        let head_dim = config.hidden_size / config.num_attention_heads;
        let kv_dim = config.num_key_value_heads * head_dim;

        let input_layernorm_weight =
            find_tensor(tensors, &format!("{prefix}.input_layernorm.weight"))?;
        let post_attention_layernorm_weight = find_tensor(
            tensors,
            &format!("{prefix}.post_attention_layernorm.weight"),
        )?;
        let q_proj_weight = find_tensor(tensors, &format!("{prefix}.self_attn.q_proj.weight"))?;
        let q_proj_bias = find_tensor(tensors, &format!("{prefix}.self_attn.q_proj.bias"))?;
        let k_proj_weight = find_tensor(tensors, &format!("{prefix}.self_attn.k_proj.weight"))?;
        let k_proj_bias = find_tensor(tensors, &format!("{prefix}.self_attn.k_proj.bias"))?;
        let v_proj_weight = find_tensor(tensors, &format!("{prefix}.self_attn.v_proj.weight"))?;
        let v_proj_bias = find_tensor(tensors, &format!("{prefix}.self_attn.v_proj.bias"))?;
        let o_proj_weight = find_tensor(tensors, &format!("{prefix}.self_attn.o_proj.weight"))?;
        let gate_proj_weight = find_tensor(tensors, &format!("{prefix}.mlp.gate_proj.weight"))?;
        let up_proj_weight = find_tensor(tensors, &format!("{prefix}.mlp.up_proj.weight"))?;
        let down_proj_weight = find_tensor(tensors, &format!("{prefix}.mlp.down_proj.weight"))?;

        expect_1d_len(
            &format!("{prefix}.input_layernorm.weight"),
            input_layernorm_weight,
            hidden,
        )?;
        expect_1d_len(
            &format!("{prefix}.post_attention_layernorm.weight"),
            post_attention_layernorm_weight,
            hidden,
        )?;
        expect_2d_shape(
            &format!("{prefix}.self_attn.q_proj.weight"),
            q_proj_weight,
            [hidden, hidden],
        )?;
        expect_1d_len(
            &format!("{prefix}.self_attn.q_proj.bias"),
            q_proj_bias,
            hidden,
        )?;
        expect_2d_shape(
            &format!("{prefix}.self_attn.k_proj.weight"),
            k_proj_weight,
            [kv_dim, hidden],
        )?;
        expect_1d_len(
            &format!("{prefix}.self_attn.k_proj.bias"),
            k_proj_bias,
            kv_dim,
        )?;
        expect_2d_shape(
            &format!("{prefix}.self_attn.v_proj.weight"),
            v_proj_weight,
            [kv_dim, hidden],
        )?;
        expect_1d_len(
            &format!("{prefix}.self_attn.v_proj.bias"),
            v_proj_bias,
            kv_dim,
        )?;
        expect_2d_shape(
            &format!("{prefix}.self_attn.o_proj.weight"),
            o_proj_weight,
            [hidden, hidden],
        )?;
        expect_2d_shape(
            &format!("{prefix}.mlp.gate_proj.weight"),
            gate_proj_weight,
            [intermediate, hidden],
        )?;
        expect_2d_shape(
            &format!("{prefix}.mlp.up_proj.weight"),
            up_proj_weight,
            [intermediate, hidden],
        )?;
        expect_2d_shape(
            &format!("{prefix}.mlp.down_proj.weight"),
            down_proj_weight,
            [hidden, intermediate],
        )?;

        Ok(Self {
            layer_index,
            config,
            input_layernorm_weight: input_layernorm_weight.data.clone(),
            post_attention_layernorm_weight: post_attention_layernorm_weight.data.clone(),
            q_proj_weight: q_proj_weight.data.clone(),
            q_proj_bias: q_proj_bias.data.clone(),
            k_proj_weight: k_proj_weight.data.clone(),
            k_proj_bias: k_proj_bias.data.clone(),
            v_proj_weight: v_proj_weight.data.clone(),
            v_proj_bias: v_proj_bias.data.clone(),
            o_proj_weight: o_proj_weight.data.clone(),
            gate_proj_weight: gate_proj_weight.data.clone(),
            up_proj_weight: up_proj_weight.data.clone(),
            down_proj_weight: down_proj_weight.data.clone(),
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct QwenTokenEmbeddingWeights {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub token_embedding_weight: Vec<f32>,
    pub lm_head_weight: Vec<f32>,
    pub final_norm_weight: Vec<f32>,
}

impl QwenTokenEmbeddingWeights {
    pub fn from_safetensors_files(
        embedding_path: impl AsRef<std::path::Path>,
        head_path: impl AsRef<std::path::Path>,
    ) -> LocateAnythingResult<Self> {
        let mut tensors = load_required_tensors_from_safetensors_file(
            embedding_path.as_ref(),
            &[
                "language_model.model.embed_tokens.weight",
                "language_model.model.norm.weight",
            ],
        )?;
        tensors.extend(load_required_tensors_from_safetensors_file(
            head_path.as_ref(),
            &["language_model.lm_head.weight"],
        )?);
        Self::from_loaded_tensors(&tensors)
    }

    pub fn from_loaded_tensors(
        tensors: &[(String, LoadedTensorF32)],
    ) -> LocateAnythingResult<Self> {
        let token_embedding_weight =
            find_tensor(tensors, "language_model.model.embed_tokens.weight")?;
        let final_norm_weight = find_tensor(tensors, "language_model.model.norm.weight")?;
        let lm_head_weight = find_tensor(tensors, "language_model.lm_head.weight")?;
        let [vocab_size, hidden_size] = expect_2d(
            "language_model.model.embed_tokens.weight",
            token_embedding_weight,
        )?;
        expect_1d_len(
            "language_model.model.norm.weight",
            final_norm_weight,
            hidden_size,
        )?;
        expect_2d_shape(
            "language_model.lm_head.weight",
            lm_head_weight,
            [vocab_size, hidden_size],
        )?;
        Ok(Self {
            vocab_size,
            hidden_size,
            token_embedding_weight: token_embedding_weight.data.clone(),
            lm_head_weight: lm_head_weight.data.clone(),
            final_norm_weight: final_norm_weight.data.clone(),
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct QwenDecoderWeights {
    pub layers: Vec<QwenDecoderLayerWeights>,
    pub head: QwenTokenEmbeddingWeights,
}

impl QwenDecoderWeights {
    pub fn from_safetensors_files(
        layer_paths: &[impl AsRef<std::path::Path>],
        embedding_path: impl AsRef<std::path::Path>,
        head_path: impl AsRef<std::path::Path>,
        config: QwenLanguageConfig,
    ) -> LocateAnythingResult<Self> {
        let mut layers = Vec::with_capacity(config.num_layers);
        for layer_index in 0..config.num_layers {
            let mut loaded = None;
            for path in layer_paths {
                match QwenDecoderLayerWeights::from_safetensors_file(
                    path.as_ref(),
                    layer_index,
                    config.clone(),
                ) {
                    Ok(weights) => {
                        loaded = Some(weights);
                        break;
                    }
                    Err(LocateAnythingError::Runtime(err)) if err.contains("missing tensor") => {}
                    Err(err) => return Err(err),
                }
            }
            layers.push(loaded.ok_or_else(|| {
                LocateAnythingError::Runtime(format!(
                    "missing Qwen layer {layer_index} in provided safetensors shards"
                ))
            })?);
        }
        let head = QwenTokenEmbeddingWeights::from_safetensors_files(embedding_path, head_path)?;
        Ok(Self { layers, head })
    }
}

pub const LOCATE_ANYTHING_MTP_FUTURE_TOKENS: usize = 6;

pub fn qwen_ar_step_position_ids(
    generated_len: usize,
    past_len: usize,
) -> LocateAnythingResult<Vec<usize>> {
    if past_len > generated_len {
        return Err(LocateAnythingError::Config(format!(
            "Qwen AR position plan has past_len={past_len} beyond generated_len={generated_len}"
        )));
    }
    Ok((past_len..generated_len).collect())
}

pub fn qwen_mtp_step_position_ids(
    generated_len: usize,
    past_len: usize,
    n_future_tokens: usize,
) -> LocateAnythingResult<Vec<usize>> {
    if n_future_tokens == 0 {
        return Err(LocateAnythingError::Config(
            "Qwen MTP position plan requires at least one future token".to_string(),
        ));
    }
    if past_len > generated_len {
        return Err(LocateAnythingError::Config(format!(
            "Qwen MTP position plan has past_len={past_len} beyond generated_len={generated_len}"
        )));
    }
    let generated_with_mask_len = generated_len + n_future_tokens;
    let mut position_ids = (past_len..generated_with_mask_len).collect::<Vec<_>>();
    let mtp_start = position_ids.len().saturating_sub(n_future_tokens);
    for position in &mut position_ids[mtp_start..] {
        *position = position.saturating_sub(1);
    }
    Ok(position_ids)
}

pub fn qwen_ar_step_input_ids(
    generated_ids: &[u32],
    past_len: usize,
) -> LocateAnythingResult<Vec<u32>> {
    if past_len > generated_ids.len() {
        return Err(LocateAnythingError::Config(format!(
            "Qwen AR input plan has past_len={past_len} beyond generated_len={}",
            generated_ids.len()
        )));
    }
    Ok(generated_ids[past_len..].to_vec())
}

pub fn qwen_mtp_step_input_ids(
    generated_ids: &[u32],
    past_len: usize,
    default_mask_token_id: u32,
    n_future_tokens: usize,
) -> LocateAnythingResult<Vec<u32>> {
    if generated_ids.is_empty() {
        return Err(LocateAnythingError::Config(
            "Qwen MTP input plan requires at least one generated token".to_string(),
        ));
    }
    if n_future_tokens == 0 {
        return Err(LocateAnythingError::Config(
            "Qwen MTP input plan requires at least one future token".to_string(),
        ));
    }
    let mut step_ids = Vec::with_capacity(generated_ids.len() + n_future_tokens);
    step_ids.extend_from_slice(generated_ids);
    step_ids.push(*generated_ids.last().unwrap());
    step_ids.extend(std::iter::repeat_n(
        default_mask_token_id,
        n_future_tokens - 1,
    ));
    if past_len > step_ids.len() {
        return Err(LocateAnythingError::Config(format!(
            "Qwen MTP input plan has past_len={past_len} beyond step input len={}",
            step_ids.len()
        )));
    }
    Ok(step_ids[past_len..].to_vec())
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

fn expect_2d_shape(
    key: &str,
    tensor: &LoadedTensorF32,
    expected: [usize; 2],
) -> LocateAnythingResult<()> {
    let actual = expect_2d(key, tensor)?;
    if actual != expected {
        return Err(LocateAnythingError::Config(format!(
            "tensor `{key}` expected shape {expected:?}, got {actual:?}"
        )));
    }
    Ok(())
}

#[cfg(any(
    feature = "backend_ndarray",
    feature = "backend_wgpu",
    feature = "backend_cuda"
))]
pub mod burn_language {
    use burn::prelude::*;
    use burn::tensor::{
        Int,
        activation::{sigmoid, softmax},
    };

    use super::{
        QwenAttentionMaskMode, QwenDecoderLayerWeights, QwenDecoderWeights, QwenLanguageConfig,
        QwenTokenEmbeddingWeights,
    };
    use crate::assets::weight_file_for_tensor;
    use crate::tensor_io::{LoadedTensorData, load_required_tensor_data_from_safetensors_file};
    use crate::{LocateAnythingError, LocateAnythingResult};

    #[derive(Clone, Debug)]
    pub struct QwenLayerKvCache<B: Backend> {
        pub key: Option<Tensor<B, 4>>,
        pub value: Option<Tensor<B, 4>>,
        pub max_cached_tokens: usize,
    }

    impl<B: Backend> QwenLayerKvCache<B> {
        pub fn new(max_cached_tokens: usize) -> Self {
            Self {
                key: None,
                value: None,
                max_cached_tokens,
            }
        }

        pub fn len(&self) -> usize {
            self.key.as_ref().map(|key| key.dims()[2]).unwrap_or(0)
        }

        pub fn is_empty(&self) -> bool {
            self.len() == 0
        }

        pub fn clear(&mut self) {
            self.key = None;
            self.value = None;
        }

        pub fn truncate(&mut self, len: usize) {
            if let Some(key) = self.key.take() {
                let [batch, heads, tokens, head_dim] = key.dims();
                self.key = Some(key.slice([0..batch, 0..heads, 0..len.min(tokens), 0..head_dim]));
            }
            if let Some(value) = self.value.take() {
                let [batch, heads, tokens, head_dim] = value.dims();
                self.value =
                    Some(value.slice([0..batch, 0..heads, 0..len.min(tokens), 0..head_dim]));
            }
        }
    }

    #[derive(Debug)]
    pub struct BurnQwenDecoderLayer<B: Backend> {
        config: QwenLanguageConfig,
        input_layernorm_weight: Tensor<B, 1>,
        post_attention_layernorm_weight: Tensor<B, 1>,
        q_proj_weight: Tensor<B, 2>,
        q_proj_bias: Tensor<B, 1>,
        k_proj_weight: Tensor<B, 2>,
        k_proj_bias: Tensor<B, 1>,
        v_proj_weight: Tensor<B, 2>,
        v_proj_bias: Tensor<B, 1>,
        o_proj_weight: Tensor<B, 2>,
        gate_proj_weight: Tensor<B, 2>,
        up_proj_weight: Tensor<B, 2>,
        down_proj_weight: Tensor<B, 2>,
    }

    impl<B: Backend> BurnQwenDecoderLayer<B> {
        pub fn from_weights(weights: QwenDecoderLayerWeights, device: &B::Device) -> Self {
            let hidden = weights.config.hidden_size;
            let intermediate = weights.config.intermediate_size;
            let head_dim = hidden / weights.config.num_attention_heads;
            let kv_dim = weights.config.num_key_value_heads * head_dim;
            let config = weights.config.clone();
            Self {
                input_layernorm_weight: tensor1(&weights.input_layernorm_weight, device),
                post_attention_layernorm_weight: tensor1(
                    &weights.post_attention_layernorm_weight,
                    device,
                ),
                q_proj_weight: tensor2(&weights.q_proj_weight, [hidden, hidden], device),
                q_proj_bias: tensor1(&weights.q_proj_bias, device),
                k_proj_weight: tensor2(&weights.k_proj_weight, [kv_dim, hidden], device),
                k_proj_bias: tensor1(&weights.k_proj_bias, device),
                v_proj_weight: tensor2(&weights.v_proj_weight, [kv_dim, hidden], device),
                v_proj_bias: tensor1(&weights.v_proj_bias, device),
                o_proj_weight: tensor2(&weights.o_proj_weight, [hidden, hidden], device),
                gate_proj_weight: tensor2(
                    &weights.gate_proj_weight,
                    [intermediate, hidden],
                    device,
                ),
                up_proj_weight: tensor2(&weights.up_proj_weight, [intermediate, hidden], device),
                down_proj_weight: tensor2(
                    &weights.down_proj_weight,
                    [hidden, intermediate],
                    device,
                ),
                config,
            }
        }

        pub fn from_safetensors_file(
            path: impl AsRef<std::path::Path>,
            layer_index: usize,
            config: QwenLanguageConfig,
            device: &B::Device,
        ) -> LocateAnythingResult<Self> {
            let keys = qwen_layer_keys(layer_index);
            let key_refs = keys.iter().map(String::as_str).collect::<Vec<_>>();
            let mut tensors =
                load_required_tensor_data_from_safetensors_file(path.as_ref(), &key_refs)?;
            Self::from_loaded_tensor_data(layer_index, config, &mut tensors, device)
        }

        pub fn from_model_root(
            model_root: impl AsRef<std::path::Path>,
            layer_index: usize,
            config: QwenLanguageConfig,
            device: &B::Device,
        ) -> LocateAnythingResult<Self> {
            let model_root = model_root.as_ref();
            let keys = qwen_layer_keys(layer_index);
            let mut grouped = std::collections::BTreeMap::<std::path::PathBuf, Vec<String>>::new();
            for key in &keys {
                grouped
                    .entry(weight_file_for_tensor(model_root, key)?)
                    .or_default()
                    .push(key.clone());
            }
            let mut tensors = Vec::with_capacity(keys.len());
            for (path, keys) in grouped {
                let key_refs = keys.iter().map(String::as_str).collect::<Vec<_>>();
                tensors.extend(load_required_tensor_data_from_safetensors_file(
                    &path, &key_refs,
                )?);
            }
            Self::from_loaded_tensor_data(layer_index, config, &mut tensors, device)
        }

        fn from_loaded_tensor_data(
            layer_index: usize,
            config: QwenLanguageConfig,
            tensors: &mut Vec<(String, LoadedTensorData)>,
            device: &B::Device,
        ) -> LocateAnythingResult<Self> {
            let prefix = format!("language_model.model.layers.{layer_index}");
            let hidden = config.hidden_size;
            let intermediate = config.intermediate_size;
            let head_dim = config.hidden_size / config.num_attention_heads;
            let kv_dim = config.num_key_value_heads * head_dim;
            Ok(Self {
                input_layernorm_weight: take_tensor1(
                    tensors,
                    &format!("{prefix}.input_layernorm.weight"),
                    hidden,
                    device,
                )?,
                post_attention_layernorm_weight: take_tensor1(
                    tensors,
                    &format!("{prefix}.post_attention_layernorm.weight"),
                    hidden,
                    device,
                )?,
                q_proj_weight: take_tensor2(
                    tensors,
                    &format!("{prefix}.self_attn.q_proj.weight"),
                    [hidden, hidden],
                    device,
                )?,
                q_proj_bias: take_tensor1(
                    tensors,
                    &format!("{prefix}.self_attn.q_proj.bias"),
                    hidden,
                    device,
                )?,
                k_proj_weight: take_tensor2(
                    tensors,
                    &format!("{prefix}.self_attn.k_proj.weight"),
                    [kv_dim, hidden],
                    device,
                )?,
                k_proj_bias: take_tensor1(
                    tensors,
                    &format!("{prefix}.self_attn.k_proj.bias"),
                    kv_dim,
                    device,
                )?,
                v_proj_weight: take_tensor2(
                    tensors,
                    &format!("{prefix}.self_attn.v_proj.weight"),
                    [kv_dim, hidden],
                    device,
                )?,
                v_proj_bias: take_tensor1(
                    tensors,
                    &format!("{prefix}.self_attn.v_proj.bias"),
                    kv_dim,
                    device,
                )?,
                o_proj_weight: take_tensor2(
                    tensors,
                    &format!("{prefix}.self_attn.o_proj.weight"),
                    [hidden, hidden],
                    device,
                )?,
                gate_proj_weight: take_tensor2(
                    tensors,
                    &format!("{prefix}.mlp.gate_proj.weight"),
                    [intermediate, hidden],
                    device,
                )?,
                up_proj_weight: take_tensor2(
                    tensors,
                    &format!("{prefix}.mlp.up_proj.weight"),
                    [intermediate, hidden],
                    device,
                )?,
                down_proj_weight: take_tensor2(
                    tensors,
                    &format!("{prefix}.mlp.down_proj.weight"),
                    [hidden, intermediate],
                    device,
                )?,
                config,
            })
        }

        pub fn forward(
            &self,
            hidden_states: Tensor<B, 3>,
            cache: Option<&mut QwenLayerKvCache<B>>,
        ) -> Tensor<B, 3> {
            self.forward_with_position_ids(hidden_states, cache, None)
        }

        pub fn forward_with_position_ids(
            &self,
            hidden_states: Tensor<B, 3>,
            cache: Option<&mut QwenLayerKvCache<B>>,
            position_ids: Option<&[usize]>,
        ) -> Tensor<B, 3> {
            self.forward_with_attention_mask_mode(
                hidden_states,
                cache,
                position_ids,
                QwenAttentionMaskMode::Causal,
            )
        }

        pub fn forward_with_attention_mask_mode(
            &self,
            hidden_states: Tensor<B, 3>,
            cache: Option<&mut QwenLayerKvCache<B>>,
            position_ids: Option<&[usize]>,
            mask_mode: QwenAttentionMaskMode,
        ) -> Tensor<B, 3> {
            self.forward_with_precomputed_attention_mask(
                hidden_states,
                cache,
                position_ids,
                mask_mode,
                None,
                None,
            )
        }

        fn forward_with_precomputed_attention_mask(
            &self,
            hidden_states: Tensor<B, 3>,
            cache: Option<&mut QwenLayerKvCache<B>>,
            position_ids: Option<&[usize]>,
            mask_mode: QwenAttentionMaskMode,
            attention_mask: Option<Tensor<B, 4>>,
            rope_cos_sin: Option<(Tensor<B, 4>, Tensor<B, 4>)>,
        ) -> Tensor<B, 3> {
            let residual = hidden_states.clone();
            let hidden_states = qwen_rms_norm(
                hidden_states,
                self.input_layernorm_weight.clone(),
                self.config.rms_norm_eps,
            );
            let attn_out = self.self_attention(
                hidden_states,
                cache,
                position_ids,
                mask_mode,
                attention_mask,
                rope_cos_sin,
            );
            let hidden_states = residual + attn_out;

            let residual = hidden_states.clone();
            let mlp_in = qwen_rms_norm(
                hidden_states,
                self.post_attention_layernorm_weight.clone(),
                self.config.rms_norm_eps,
            );
            let hidden = self.config.hidden_size;
            let [batch, tokens, _] = residual.dims();
            let mlp_flat = mlp_in.reshape([batch * tokens, hidden]);
            let gate = mlp_flat
                .clone()
                .matmul(self.gate_proj_weight.clone().swap_dims(0, 1));
            let up = mlp_flat.matmul(self.up_proj_weight.clone().swap_dims(0, 1));
            let down = silu(gate)
                .mul(up)
                .matmul(self.down_proj_weight.clone().swap_dims(0, 1));
            residual + down.reshape([batch, tokens, hidden])
        }

        fn self_attention(
            &self,
            hidden_states: Tensor<B, 3>,
            cache: Option<&mut QwenLayerKvCache<B>>,
            position_ids: Option<&[usize]>,
            mask_mode: QwenAttentionMaskMode,
            attention_mask: Option<Tensor<B, 4>>,
            rope_cos_sin: Option<(Tensor<B, 4>, Tensor<B, 4>)>,
        ) -> Tensor<B, 3> {
            let cfg = &self.config;
            let [batch, q_len, hidden] = hidden_states.dims();
            if let Some(position_ids) = position_ids {
                assert_eq!(
                    position_ids.len(),
                    q_len,
                    "Qwen explicit position ids must match query length"
                );
            }
            let heads = cfg.num_attention_heads;
            let kv_heads = cfg.num_key_value_heads;
            let head_dim = hidden / heads;
            let kv_dim = kv_heads * head_dim;
            let past_len = cache.as_ref().map(|cache| cache.len()).unwrap_or(0);

            let q = hidden_states
                .clone()
                .reshape([batch * q_len, hidden])
                .matmul(self.q_proj_weight.clone().swap_dims(0, 1))
                + self.q_proj_bias.clone().reshape([1, hidden]);
            let k = hidden_states
                .clone()
                .reshape([batch * q_len, hidden])
                .matmul(self.k_proj_weight.clone().swap_dims(0, 1))
                + self.k_proj_bias.clone().reshape([1, kv_dim]);
            let v = hidden_states
                .reshape([batch * q_len, hidden])
                .matmul(self.v_proj_weight.clone().swap_dims(0, 1))
                + self.v_proj_bias.clone().reshape([1, kv_dim]);

            let q = q.reshape([batch, q_len, heads, head_dim]).swap_dims(1, 2);
            let k = k
                .reshape([batch, q_len, kv_heads, head_dim])
                .swap_dims(1, 2);
            let v = v
                .reshape([batch, q_len, kv_heads, head_dim])
                .swap_dims(1, 2);

            let (cos, sin) = rope_cos_sin.unwrap_or_else(|| {
                if let Some(position_ids) = position_ids {
                    qwen_rope_cos_sin_for_positions::<B>(
                        position_ids,
                        head_dim,
                        cfg.rope_theta,
                        &self.q_proj_bias.device(),
                    )
                } else {
                    qwen_rope_cos_sin::<B>(
                        past_len,
                        q_len,
                        head_dim,
                        cfg.rope_theta,
                        &self.q_proj_bias.device(),
                    )
                }
            });
            let q = apply_qwen_rope(q, cos.clone(), sin.clone());
            let k = apply_qwen_rope(k, cos, sin);

            let (k, v, kv_len) = if let Some(cache) = cache {
                let k_all = if let Some(prev) = cache.key.take() {
                    Tensor::cat(vec![prev, k], 2)
                } else {
                    k
                };
                let v_all = if let Some(prev) = cache.value.take() {
                    Tensor::cat(vec![prev, v], 2)
                } else {
                    v
                };
                let kv_len = k_all.dims()[2];
                if kv_len > cache.max_cached_tokens {
                    panic!(
                        "Qwen KV cache exceeded max_cached_tokens: {kv_len} > {}",
                        cache.max_cached_tokens
                    );
                }
                cache.key = Some(k_all.clone());
                cache.value = Some(v_all.clone());
                (k_all, v_all, kv_len)
            } else {
                (k, v, q_len)
            };

            let key = repeat_kv(k, heads / kv_heads);
            let value = repeat_kv(v, heads / kv_heads);
            let mut scores = q
                .matmul(key.swap_dims(2, 3))
                .mul_scalar(1.0 / (head_dim as f32).sqrt());
            if let Some(mask) = attention_mask {
                scores = scores + mask;
            } else if let Some(mask) = qwen_attention_mask::<B>(
                q_len,
                kv_len,
                past_len,
                mask_mode,
                &self.q_proj_bias.device(),
            ) {
                scores = scores + mask;
            }
            let attn = softmax(scores, 3);
            let output = attn
                .matmul(value)
                .swap_dims(1, 2)
                .reshape([batch * q_len, hidden])
                .matmul(self.o_proj_weight.clone().swap_dims(0, 1));
            output.reshape([batch, q_len, hidden])
        }
    }

    #[derive(Debug)]
    pub struct BurnQwenHead<B: Backend> {
        vocab_size: usize,
        hidden_size: usize,
        token_embedding_weight: Tensor<B, 2>,
        lm_head_weight: Tensor<B, 2>,
        final_norm_weight: Tensor<B, 1>,
    }

    impl<B: Backend> BurnQwenHead<B> {
        pub fn from_weights(weights: QwenTokenEmbeddingWeights, device: &B::Device) -> Self {
            Self {
                token_embedding_weight: tensor2(
                    &weights.token_embedding_weight,
                    [weights.vocab_size, weights.hidden_size],
                    device,
                ),
                lm_head_weight: tensor2(
                    &weights.lm_head_weight,
                    [weights.vocab_size, weights.hidden_size],
                    device,
                ),
                final_norm_weight: tensor1(&weights.final_norm_weight, device),
                vocab_size: weights.vocab_size,
                hidden_size: weights.hidden_size,
            }
        }

        pub fn from_safetensors_files(
            embedding_path: impl AsRef<std::path::Path>,
            head_path: impl AsRef<std::path::Path>,
            device: &B::Device,
        ) -> LocateAnythingResult<Self> {
            let mut tensors = load_required_tensor_data_from_safetensors_file(
                embedding_path.as_ref(),
                &[
                    "language_model.model.embed_tokens.weight",
                    "language_model.model.norm.weight",
                ],
            )?;
            tensors.extend(load_required_tensor_data_from_safetensors_file(
                head_path.as_ref(),
                &["language_model.lm_head.weight"],
            )?);
            let [vocab_size, hidden_size] =
                expect_tensor2(&tensors, "language_model.model.embed_tokens.weight")?;
            expect_tensor1(&tensors, "language_model.model.norm.weight", hidden_size)?;
            expect_tensor2_shape(
                &tensors,
                "language_model.lm_head.weight",
                [vocab_size, hidden_size],
            )?;
            Ok(Self {
                token_embedding_weight: take_tensor2(
                    &mut tensors,
                    "language_model.model.embed_tokens.weight",
                    [vocab_size, hidden_size],
                    device,
                )?,
                lm_head_weight: take_tensor2(
                    &mut tensors,
                    "language_model.lm_head.weight",
                    [vocab_size, hidden_size],
                    device,
                )?,
                final_norm_weight: take_tensor1(
                    &mut tensors,
                    "language_model.model.norm.weight",
                    hidden_size,
                    device,
                )?,
                vocab_size,
                hidden_size,
            })
        }

        pub fn from_model_root(
            model_root: impl AsRef<std::path::Path>,
            device: &B::Device,
        ) -> LocateAnythingResult<Self> {
            let model_root = model_root.as_ref();
            let keys = [
                "language_model.model.embed_tokens.weight".to_string(),
                "language_model.model.norm.weight".to_string(),
                "language_model.lm_head.weight".to_string(),
            ];
            let mut grouped = std::collections::BTreeMap::<std::path::PathBuf, Vec<String>>::new();
            for key in &keys {
                grouped
                    .entry(weight_file_for_tensor(model_root, key)?)
                    .or_default()
                    .push(key.clone());
            }
            let mut tensors = Vec::with_capacity(keys.len());
            for (path, keys) in grouped {
                let key_refs = keys.iter().map(String::as_str).collect::<Vec<_>>();
                tensors.extend(load_required_tensor_data_from_safetensors_file(
                    &path, &key_refs,
                )?);
            }
            let [vocab_size, hidden_size] =
                expect_tensor2(&tensors, "language_model.model.embed_tokens.weight")?;
            expect_tensor1(&tensors, "language_model.model.norm.weight", hidden_size)?;
            expect_tensor2_shape(
                &tensors,
                "language_model.lm_head.weight",
                [vocab_size, hidden_size],
            )?;
            Ok(Self {
                token_embedding_weight: take_tensor2(
                    &mut tensors,
                    "language_model.model.embed_tokens.weight",
                    [vocab_size, hidden_size],
                    device,
                )?,
                lm_head_weight: take_tensor2(
                    &mut tensors,
                    "language_model.lm_head.weight",
                    [vocab_size, hidden_size],
                    device,
                )?,
                final_norm_weight: take_tensor1(
                    &mut tensors,
                    "language_model.model.norm.weight",
                    hidden_size,
                    device,
                )?,
                vocab_size,
                hidden_size,
            })
        }

        pub fn vocab_size(&self) -> usize {
            self.vocab_size
        }

        pub fn hidden_size(&self) -> usize {
            self.hidden_size
        }

        pub fn final_norm(&self, hidden_states: Tensor<B, 3>, eps: f32) -> Tensor<B, 3> {
            qwen_rms_norm(hidden_states, self.final_norm_weight.clone(), eps)
        }

        pub fn logits(&self, hidden_states: Tensor<B, 3>) -> Tensor<B, 3> {
            let [batch, tokens, hidden] = hidden_states.dims();
            hidden_states
                .reshape([batch * tokens, hidden])
                .matmul(self.lm_head_weight.clone().swap_dims(0, 1))
                .reshape([batch, tokens, self.vocab_size])
        }

        pub fn token_embedding_weight(&self) -> Tensor<B, 2> {
            self.token_embedding_weight.clone()
        }

        pub fn embed_token_ids(
            &self,
            token_ids: &[u32],
            device: &B::Device,
        ) -> LocateAnythingResult<Tensor<B, 3>> {
            if token_ids.is_empty() {
                return Err(LocateAnythingError::Config(
                    "cannot embed an empty LocateAnything prompt".to_string(),
                ));
            }
            for &token_id in token_ids {
                if token_id as usize >= self.vocab_size {
                    return Err(LocateAnythingError::Config(format!(
                        "token id {token_id} exceeds Qwen vocab size {}",
                        self.vocab_size
                    )));
                }
            }
            let ids = token_ids
                .iter()
                .map(|token_id| *token_id as i32)
                .collect::<Vec<_>>();
            let indices = Tensor::<B, 1, Int>::from_ints(ids.as_slice(), device);
            Ok(self
                .token_embedding_weight
                .clone()
                .select(0, indices)
                .reshape([1, token_ids.len(), self.hidden_size]))
        }

        pub fn embed_token_ids_with_image_features(
            &self,
            token_ids: &[u32],
            image_token_positions: &[usize],
            image_features: Tensor<B, 2>,
            device: &B::Device,
        ) -> LocateAnythingResult<Tensor<B, 3>> {
            let text_embeddings = self.embed_token_ids(token_ids, device)?;
            insert_image_features(text_embeddings, image_token_positions, image_features)
        }
    }

    #[derive(Debug)]
    pub struct BurnQwenDecoder<B: Backend> {
        config: QwenLanguageConfig,
        layers: Vec<BurnQwenDecoderLayer<B>>,
        head: BurnQwenHead<B>,
    }

    impl<B: Backend> BurnQwenDecoder<B> {
        pub fn from_weights(weights: QwenDecoderWeights, device: &B::Device) -> Self {
            let config = weights
                .layers
                .first()
                .map(|layer| layer.config.clone())
                .unwrap_or_default();
            let layers = weights
                .layers
                .into_iter()
                .map(|layer| BurnQwenDecoderLayer::from_weights(layer, device))
                .collect();
            let head = BurnQwenHead::from_weights(weights.head, device);
            Self {
                config,
                layers,
                head,
            }
        }

        pub fn from_safetensors_files(
            layer_paths: &[impl AsRef<std::path::Path>],
            embedding_path: impl AsRef<std::path::Path>,
            head_path: impl AsRef<std::path::Path>,
            config: QwenLanguageConfig,
            device: &B::Device,
        ) -> LocateAnythingResult<Self> {
            let mut layers = Vec::with_capacity(config.num_layers);
            for layer_index in 0..config.num_layers {
                let mut loaded = None;
                for path in layer_paths {
                    match BurnQwenDecoderLayer::from_safetensors_file(
                        path.as_ref(),
                        layer_index,
                        config.clone(),
                        device,
                    ) {
                        Ok(layer) => {
                            loaded = Some(layer);
                            break;
                        }
                        Err(LocateAnythingError::Runtime(err))
                            if err.contains("missing tensor") => {}
                        Err(err) => return Err(err),
                    }
                }
                layers.push(loaded.ok_or_else(|| {
                    LocateAnythingError::Runtime(format!(
                        "missing Qwen layer {layer_index} in provided safetensors shards"
                    ))
                })?);
            }
            let head = BurnQwenHead::from_safetensors_files(embedding_path, head_path, device)?;
            Ok(Self {
                config,
                layers,
                head,
            })
        }

        pub fn from_model_root(
            model_root: impl AsRef<std::path::Path>,
            config: QwenLanguageConfig,
            device: &B::Device,
        ) -> LocateAnythingResult<Self> {
            let model_root = model_root.as_ref();
            let mut layers = Vec::with_capacity(config.num_layers);
            for layer_index in 0..config.num_layers {
                layers.push(BurnQwenDecoderLayer::from_model_root(
                    model_root,
                    layer_index,
                    config.clone(),
                    device,
                )?);
            }
            let head = BurnQwenHead::from_model_root(model_root, device)?;
            Ok(Self {
                config,
                layers,
                head,
            })
        }

        pub fn vocab_size(&self) -> usize {
            self.head.vocab_size()
        }

        pub fn hidden_size(&self) -> usize {
            self.head.hidden_size()
        }

        pub fn new_cache(&self, max_cached_tokens: usize) -> Vec<QwenLayerKvCache<B>> {
            self.layers
                .iter()
                .map(|_| QwenLayerKvCache::new(max_cached_tokens))
                .collect()
        }

        pub fn forward_hidden(
            &self,
            hidden_states: Tensor<B, 3>,
            caches: Option<&mut [QwenLayerKvCache<B>]>,
        ) -> Tensor<B, 3> {
            self.forward_hidden_with_position_ids(hidden_states, caches, None)
        }

        pub fn forward_hidden_with_position_ids(
            &self,
            hidden_states: Tensor<B, 3>,
            caches: Option<&mut [QwenLayerKvCache<B>]>,
            position_ids: Option<&[usize]>,
        ) -> Tensor<B, 3> {
            self.forward_hidden_with_attention_mask_mode(
                hidden_states,
                caches,
                position_ids,
                QwenAttentionMaskMode::Causal,
            )
        }

        pub fn forward_hidden_with_attention_mask_mode(
            &self,
            mut hidden_states: Tensor<B, 3>,
            mut caches: Option<&mut [QwenLayerKvCache<B>]>,
            position_ids: Option<&[usize]>,
            mask_mode: QwenAttentionMaskMode,
        ) -> Tensor<B, 3> {
            if let Some(caches) = caches.as_ref() {
                assert_eq!(caches.len(), self.layers.len());
            }
            let [_, q_len, _] = hidden_states.dims();
            let past_len = caches
                .as_ref()
                .and_then(|caches| caches.first())
                .map(QwenLayerKvCache::len)
                .unwrap_or(0);
            if let Some(caches) = caches.as_ref() {
                for cache in caches.iter().skip(1) {
                    debug_assert_eq!(
                        cache.len(),
                        past_len,
                        "Qwen layer KV caches must have the same sequence length before a forward"
                    );
                }
            }
            let attention_mask = qwen_attention_mask::<B>(
                q_len,
                past_len + q_len,
                past_len,
                mask_mode,
                &hidden_states.device(),
            );
            let head_dim = self.config.hidden_size / self.config.num_attention_heads;
            let rope_cos_sin = if let Some(position_ids) = position_ids {
                Some(qwen_rope_cos_sin_for_positions::<B>(
                    position_ids,
                    head_dim,
                    self.config.rope_theta,
                    &hidden_states.device(),
                ))
            } else {
                Some(qwen_rope_cos_sin::<B>(
                    past_len,
                    q_len,
                    head_dim,
                    self.config.rope_theta,
                    &hidden_states.device(),
                ))
            };
            for (index, layer) in self.layers.iter().enumerate() {
                hidden_states = if let Some(caches) = caches.as_deref_mut() {
                    layer.forward_with_precomputed_attention_mask(
                        hidden_states,
                        Some(&mut caches[index]),
                        position_ids,
                        mask_mode,
                        attention_mask.clone(),
                        rope_cos_sin.clone(),
                    )
                } else {
                    layer.forward_with_precomputed_attention_mask(
                        hidden_states,
                        None,
                        position_ids,
                        mask_mode,
                        attention_mask.clone(),
                        rope_cos_sin.clone(),
                    )
                };
            }
            self.head
                .final_norm(hidden_states, self.config.rms_norm_eps)
        }

        pub fn logits(&self, hidden_states: Tensor<B, 3>) -> Tensor<B, 3> {
            self.head.logits(hidden_states)
        }

        pub fn embed_token_ids(
            &self,
            token_ids: &[u32],
            device: &B::Device,
        ) -> LocateAnythingResult<Tensor<B, 3>> {
            self.head.embed_token_ids(token_ids, device)
        }

        pub fn embed_token_ids_with_image_features(
            &self,
            token_ids: &[u32],
            image_token_positions: &[usize],
            image_features: Tensor<B, 2>,
            device: &B::Device,
        ) -> LocateAnythingResult<Tensor<B, 3>> {
            self.head.embed_token_ids_with_image_features(
                token_ids,
                image_token_positions,
                image_features,
                device,
            )
        }
    }

    pub fn insert_image_features<B: Backend>(
        text_embeddings: Tensor<B, 3>,
        image_token_positions: &[usize],
        image_features: Tensor<B, 2>,
    ) -> LocateAnythingResult<Tensor<B, 3>> {
        if image_token_positions.is_empty() {
            return Err(LocateAnythingError::Config(
                "LocateAnything prompt has no image token positions".to_string(),
            ));
        }
        let [batch, tokens, hidden] = text_embeddings.dims();
        if batch != 1 {
            return Err(LocateAnythingError::Config(format!(
                "image-feature insertion currently expects batch=1, got {batch}"
            )));
        }
        let [image_tokens, image_hidden] = image_features.dims();
        if image_tokens != image_token_positions.len() {
            return Err(LocateAnythingError::Config(format!(
                "image feature token count {image_tokens} does not match prompt image token count {}",
                image_token_positions.len()
            )));
        }
        if image_hidden != hidden {
            return Err(LocateAnythingError::Config(format!(
                "image feature hidden size {image_hidden} does not match Qwen hidden size {hidden}"
            )));
        }
        let start = image_token_positions[0];
        let end = start + image_tokens;
        if end > tokens {
            return Err(LocateAnythingError::Config(format!(
                "image token span {start}..{end} exceeds prompt length {tokens}"
            )));
        }
        for (offset, &position) in image_token_positions.iter().enumerate() {
            let expected = start + offset;
            if position != expected {
                return Err(LocateAnythingError::Config(format!(
                    "image token positions must be contiguous for upstream Qwen insertion; got position {position} at offset {offset}, expected {expected}"
                )));
            }
        }

        let mut chunks = Vec::with_capacity(3);
        if start > 0 {
            chunks.push(text_embeddings.clone().slice([0..1, 0..start, 0..hidden]));
        }
        chunks.push(image_features.reshape([1, image_tokens, hidden]));
        if end < tokens {
            chunks.push(text_embeddings.slice([0..1, end..tokens, 0..hidden]));
        }
        Ok(Tensor::cat(chunks, 1))
    }

    pub fn qwen_rms_norm<B: Backend>(
        input: Tensor<B, 3>,
        weight: Tensor<B, 1>,
        eps: f32,
    ) -> Tensor<B, 3> {
        let [batch, tokens, hidden] = input.dims();
        let variance = input.clone().powf_scalar(2.0).mean_dim(2);
        input
            * variance.add_scalar(eps).sqrt().recip()
            * weight
                .reshape([1, 1, hidden])
                .expand([batch as i64, tokens as i64, -1])
    }

    pub fn qwen_rope_cos_sin<B: Backend>(
        start_position: usize,
        seq_len: usize,
        head_dim: usize,
        theta: f32,
        device: &B::Device,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let half = head_dim / 2;
        let inv_freq = (0..half)
            .map(|idx| 1.0 / theta.powf((2 * idx) as f32 / head_dim as f32))
            .collect::<Vec<_>>();
        let mut cos = Vec::with_capacity(seq_len * head_dim);
        let mut sin = Vec::with_capacity(seq_len * head_dim);
        for pos in start_position..start_position + seq_len {
            let phases = inv_freq
                .iter()
                .map(|freq| pos as f32 * *freq)
                .collect::<Vec<_>>();
            for phase in phases.iter().chain(phases.iter()) {
                cos.push(phase.cos());
                sin.push(phase.sin());
            }
        }
        (
            Tensor::<B, 1>::from_floats(cos.as_slice(), device).reshape([1, 1, seq_len, head_dim]),
            Tensor::<B, 1>::from_floats(sin.as_slice(), device).reshape([1, 1, seq_len, head_dim]),
        )
    }

    pub fn qwen_rope_cos_sin_for_positions<B: Backend>(
        position_ids: &[usize],
        head_dim: usize,
        theta: f32,
        device: &B::Device,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let half = head_dim / 2;
        let inv_freq = (0..half)
            .map(|idx| 1.0 / theta.powf((2 * idx) as f32 / head_dim as f32))
            .collect::<Vec<_>>();
        let mut cos = Vec::with_capacity(position_ids.len() * head_dim);
        let mut sin = Vec::with_capacity(position_ids.len() * head_dim);
        for &pos in position_ids {
            let phases = inv_freq
                .iter()
                .map(|freq| pos as f32 * *freq)
                .collect::<Vec<_>>();
            for phase in phases.iter().chain(phases.iter()) {
                cos.push(phase.cos());
                sin.push(phase.sin());
            }
        }
        (
            Tensor::<B, 1>::from_floats(cos.as_slice(), device).reshape([
                1,
                1,
                position_ids.len(),
                head_dim,
            ]),
            Tensor::<B, 1>::from_floats(sin.as_slice(), device).reshape([
                1,
                1,
                position_ids.len(),
                head_dim,
            ]),
        )
    }

    pub fn apply_qwen_rope<B: Backend>(
        input: Tensor<B, 4>,
        cos: Tensor<B, 4>,
        sin: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        input.clone() * cos + rotate_half(input) * sin
    }

    pub fn rotate_half<B: Backend>(input: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch, heads, tokens, head_dim] = input.dims();
        let half = head_dim / 2;
        let first = input
            .clone()
            .slice([0..batch, 0..heads, 0..tokens, 0..half]);
        let second = input.slice([0..batch, 0..heads, 0..tokens, half..head_dim]);
        Tensor::cat(vec![second.mul_scalar(-1.0), first], 3)
    }

    pub fn repeat_kv<B: Backend>(input: Tensor<B, 4>, repeat: usize) -> Tensor<B, 4> {
        if repeat == 1 {
            return input;
        }
        let [batch, kv_heads, tokens, head_dim] = input.dims();
        input
            .reshape([batch, kv_heads, 1, tokens, head_dim])
            .expand([
                batch as i64,
                kv_heads as i64,
                repeat as i64,
                tokens as i64,
                head_dim as i64,
            ])
            .reshape([batch, kv_heads * repeat, tokens, head_dim])
    }

    pub fn causal_mask<B: Backend>(
        q_len: usize,
        kv_len: usize,
        past_len: usize,
        device: &B::Device,
    ) -> Tensor<B, 4> {
        let mut values = Vec::with_capacity(q_len * kv_len);
        for q in 0..q_len {
            let max_visible = past_len + q;
            for k in 0..kv_len {
                values.push(if k <= max_visible { 0.0 } else { -1.0e30 });
            }
        }
        Tensor::<B, 1>::from_floats(values.as_slice(), device).reshape([1, 1, q_len, kv_len])
    }

    pub fn qwen_attention_mask<B: Backend>(
        q_len: usize,
        kv_len: usize,
        past_len: usize,
        mask_mode: QwenAttentionMaskMode,
        device: &B::Device,
    ) -> Option<Tensor<B, 4>> {
        match mask_mode {
            QwenAttentionMaskMode::Causal => {
                (q_len > 1).then(|| causal_mask::<B>(q_len, kv_len, past_len, device))
            }
            QwenAttentionMaskMode::MtpWindow { window_tokens } => Some(mtp_inference_mask::<B>(
                q_len,
                kv_len,
                past_len,
                window_tokens,
                device,
            )),
        }
    }

    pub fn mtp_inference_mask<B: Backend>(
        q_len: usize,
        kv_len: usize,
        past_len: usize,
        window_tokens: usize,
        device: &B::Device,
    ) -> Tensor<B, 4> {
        let mut values = Vec::with_capacity(q_len * kv_len);
        for q in 0..q_len {
            let max_visible = past_len + q;
            for k in 0..kv_len {
                values.push(if k <= max_visible { 0.0 } else { -1.0e30 });
            }
        }

        let window = window_tokens.min(q_len).min(kv_len);
        if window > 0 {
            let row_start = q_len - window;
            let col_start = kv_len - window;
            for q in row_start..q_len {
                for k in col_start..kv_len {
                    values[q * kv_len + k] = 0.0;
                }
            }
            if kv_len > window {
                let previous_col = kv_len - window - 1;
                for q in row_start..q_len {
                    values[q * kv_len + previous_col] = -1.0e30;
                }
            }
        }

        Tensor::<B, 1>::from_floats(values.as_slice(), device).reshape([1, 1, q_len, kv_len])
    }

    pub fn silu<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
        x.clone() * sigmoid(x)
    }

    fn tensor1<B: Backend>(values: &[f32], device: &B::Device) -> Tensor<B, 1> {
        Tensor::<B, 1>::from_floats(values, device)
    }

    fn tensor2<B: Backend>(values: &[f32], shape: [usize; 2], device: &B::Device) -> Tensor<B, 2> {
        Tensor::<B, 1>::from_floats(values, device).reshape(shape)
    }

    fn qwen_layer_keys(layer_index: usize) -> Vec<String> {
        let prefix = format!("language_model.model.layers.{layer_index}");
        [
            format!("{prefix}.input_layernorm.weight"),
            format!("{prefix}.post_attention_layernorm.weight"),
            format!("{prefix}.self_attn.q_proj.weight"),
            format!("{prefix}.self_attn.q_proj.bias"),
            format!("{prefix}.self_attn.k_proj.weight"),
            format!("{prefix}.self_attn.k_proj.bias"),
            format!("{prefix}.self_attn.v_proj.weight"),
            format!("{prefix}.self_attn.v_proj.bias"),
            format!("{prefix}.self_attn.o_proj.weight"),
            format!("{prefix}.mlp.gate_proj.weight"),
            format!("{prefix}.mlp.up_proj.weight"),
            format!("{prefix}.mlp.down_proj.weight"),
        ]
        .into_iter()
        .collect()
    }

    fn take_tensor1<B: Backend>(
        tensors: &mut Vec<(String, LoadedTensorData)>,
        key: &str,
        expected_len: usize,
        device: &B::Device,
    ) -> LocateAnythingResult<Tensor<B, 1>> {
        let tensor = take_tensor_data(tensors, key)?;
        if tensor.shape.as_slice() != [expected_len] {
            return Err(LocateAnythingError::Config(format!(
                "tensor `{key}` expected shape [{expected_len}], got {:?}",
                tensor.shape
            )));
        }
        Ok(Tensor::<B, 1>::from_data(tensor.data, device))
    }

    fn take_tensor2<B: Backend>(
        tensors: &mut Vec<(String, LoadedTensorData)>,
        key: &str,
        expected_shape: [usize; 2],
        device: &B::Device,
    ) -> LocateAnythingResult<Tensor<B, 2>> {
        let tensor = take_tensor_data(tensors, key)?;
        if tensor.shape.as_slice() != expected_shape {
            return Err(LocateAnythingError::Config(format!(
                "tensor `{key}` expected shape {expected_shape:?}, got {:?}",
                tensor.shape
            )));
        }
        Ok(Tensor::<B, 2>::from_data(tensor.data, device))
    }

    fn take_tensor_data(
        tensors: &mut Vec<(String, LoadedTensorData)>,
        key: &str,
    ) -> LocateAnythingResult<LoadedTensorData> {
        let index = tensors
            .iter()
            .position(|(name, _)| name == key)
            .ok_or_else(|| LocateAnythingError::Runtime(format!("missing tensor `{key}`")))?;
        Ok(tensors.swap_remove(index).1)
    }

    fn expect_tensor1(
        tensors: &[(String, LoadedTensorData)],
        key: &str,
        expected_len: usize,
    ) -> LocateAnythingResult<()> {
        let tensor = find_tensor_data(tensors, key)?;
        if tensor.shape.as_slice() != [expected_len] {
            return Err(LocateAnythingError::Config(format!(
                "tensor `{key}` expected shape [{expected_len}], got {:?}",
                tensor.shape
            )));
        }
        Ok(())
    }

    fn expect_tensor2(
        tensors: &[(String, LoadedTensorData)],
        key: &str,
    ) -> LocateAnythingResult<[usize; 2]> {
        let tensor = find_tensor_data(tensors, key)?;
        if tensor.shape.len() != 2 {
            return Err(LocateAnythingError::Config(format!(
                "tensor `{key}` expected rank 2, got {:?}",
                tensor.shape
            )));
        }
        Ok([tensor.shape[0], tensor.shape[1]])
    }

    fn expect_tensor2_shape(
        tensors: &[(String, LoadedTensorData)],
        key: &str,
        expected: [usize; 2],
    ) -> LocateAnythingResult<()> {
        let actual = expect_tensor2(tensors, key)?;
        if actual != expected {
            return Err(LocateAnythingError::Config(format!(
                "tensor `{key}` expected shape {expected:?}, got {actual:?}"
            )));
        }
        Ok(())
    }

    fn find_tensor_data<'a>(
        tensors: &'a [(String, LoadedTensorData)],
        key: &str,
    ) -> LocateAnythingResult<&'a LoadedTensorData> {
        tensors
            .iter()
            .find_map(|(name, tensor)| (name == key).then_some(tensor))
            .ok_or_else(|| LocateAnythingError::Runtime(format!("missing tensor `{key}`")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "backend_ndarray")]
    use std::time::Instant;

    #[test]
    fn default_config_matches_public_checkpoint_shape() {
        let cfg = QwenLanguageConfig::default();
        assert_eq!(cfg.hidden_size / cfg.num_attention_heads, 128);
        assert_eq!(cfg.num_attention_heads / cfg.num_key_value_heads, 8);
        assert_eq!(cfg.intermediate_size, 11_008);
    }

    #[test]
    fn qwen_generation_position_plans_match_upstream_mtp_contract() {
        assert_eq!(
            qwen_ar_step_position_ids(10, 0).unwrap(),
            vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        );
        assert_eq!(qwen_ar_step_position_ids(10, 9).unwrap(), vec![9]);
        assert_eq!(
            qwen_mtp_step_position_ids(10, 0, LOCATE_ANYTHING_MTP_FUTURE_TOKENS).unwrap(),
            vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10, 11, 12, 13, 14]
        );
        assert_eq!(
            qwen_mtp_step_position_ids(16, 10, LOCATE_ANYTHING_MTP_FUTURE_TOKENS).unwrap(),
            vec![10, 11, 12, 13, 14, 15, 15, 16, 17, 18, 19, 20]
        );
    }

    #[test]
    fn qwen_generation_input_plans_match_upstream_mtp_contract() {
        let generated = [10, 20, 30, 40];
        assert_eq!(qwen_ar_step_input_ids(&generated, 0).unwrap(), generated);
        assert_eq!(qwen_ar_step_input_ids(&generated, 3).unwrap(), vec![40]);
        assert_eq!(
            qwen_mtp_step_input_ids(&generated, 0, 151_676, 6).unwrap(),
            vec![
                10, 20, 30, 40, 40, 151_676, 151_676, 151_676, 151_676, 151_676
            ]
        );
        assert_eq!(
            qwen_mtp_step_input_ids(&generated, generated.len(), 151_676, 6).unwrap(),
            vec![40, 151_676, 151_676, 151_676, 151_676, 151_676]
        );
    }

    #[cfg(feature = "backend_ndarray")]
    #[test]
    fn qwen_decode_step_updates_kv_cache_ndarray() {
        use burn::tensor::backend::BackendTypes;

        let device = <burn::backend::NdArray<f32> as BackendTypes>::Device::default();
        let config = QwenLanguageConfig {
            hidden_size: 4,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            intermediate_size: 8,
            max_position_embeddings: 16,
            vocab_size: 32,
            ..QwenLanguageConfig::default()
        };
        let weights = QwenDecoderLayerWeights {
            layer_index: 0,
            config,
            input_layernorm_weight: vec![1.0; 4],
            post_attention_layernorm_weight: vec![1.0; 4],
            q_proj_weight: vec![0.0; 4 * 4],
            q_proj_bias: vec![0.0; 4],
            k_proj_weight: vec![0.0; 2 * 4],
            k_proj_bias: vec![0.0; 2],
            v_proj_weight: vec![0.0; 2 * 4],
            v_proj_bias: vec![0.0; 2],
            o_proj_weight: vec![0.0; 4 * 4],
            gate_proj_weight: vec![0.0; 8 * 4],
            up_proj_weight: vec![0.0; 8 * 4],
            down_proj_weight: vec![0.0; 4 * 8],
        };
        let model =
            burn_language::BurnQwenDecoderLayer::<burn::backend::NdArray<f32>>::from_weights(
                weights, &device,
            );
        let mut cache = burn_language::QwenLayerKvCache::new(4);
        let first = burn::prelude::Tensor::<burn::backend::NdArray<f32>, 1>::from_floats(
            [0.1, 0.2, 0.3, 0.4].as_slice(),
            &device,
        )
        .reshape([1, 1, 4]);
        let second = burn::prelude::Tensor::<burn::backend::NdArray<f32>, 1>::from_floats(
            [0.4, 0.3, 0.2, 0.1].as_slice(),
            &device,
        )
        .reshape([1, 1, 4]);
        let out_first = model.forward(first, Some(&mut cache));
        assert_eq!(out_first.dims(), [1, 1, 4]);
        assert_eq!(cache.len(), 1);
        let out_second = model.forward(second, Some(&mut cache));
        assert_eq!(out_second.dims(), [1, 1, 4]);
        assert_eq!(cache.len(), 2);
    }

    #[cfg(feature = "backend_ndarray")]
    #[test]
    fn explicit_position_rope_matches_sequential_rope_ndarray() {
        use burn::tensor::backend::BackendTypes;

        let device = <burn::backend::NdArray<f32> as BackendTypes>::Device::default();
        let (seq_cos, seq_sin) = burn_language::qwen_rope_cos_sin::<burn::backend::NdArray<f32>>(
            3, 4, 8, 10_000.0, &device,
        );
        let (explicit_cos, explicit_sin) = burn_language::qwen_rope_cos_sin_for_positions::<
            burn::backend::NdArray<f32>,
        >(&[3, 4, 5, 6], 8, 10_000.0, &device);
        assert_eq!(
            seq_cos
                .into_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .unwrap(),
            explicit_cos
                .into_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .unwrap()
        );
        assert_eq!(
            seq_sin
                .into_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .unwrap(),
            explicit_sin
                .into_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .unwrap()
        );
    }

    #[cfg(feature = "backend_ndarray")]
    #[test]
    fn mtp_inference_mask_matches_upstream_one_window_update_ndarray() {
        use burn::tensor::backend::BackendTypes;

        let device = <burn::backend::NdArray<f32> as BackendTypes>::Device::default();
        let mask =
            burn_language::mtp_inference_mask::<burn::backend::NdArray<f32>>(4, 7, 3, 4, &device);
        let data = mask.into_data().convert::<f32>().to_vec::<f32>().unwrap();
        let blocked = |value: f32| value < -1.0e20;
        for row in 0..4 {
            assert!(!blocked(data[row * 7]));
            assert!(!blocked(data[row * 7 + 1]));
            assert!(blocked(data[row * 7 + 2]));
            for col in 3..7 {
                assert!(!blocked(data[row * 7 + col]));
            }
        }
    }

    #[cfg(feature = "backend_ndarray")]
    #[test]
    fn qwen_head_embeds_tokens_and_inserts_image_features_ndarray() {
        use burn::tensor::backend::BackendTypes;

        let device = <burn::backend::NdArray<f32> as BackendTypes>::Device::default();
        let weights = QwenTokenEmbeddingWeights {
            vocab_size: 6,
            hidden_size: 3,
            token_embedding_weight: (0..18).map(|value| value as f32).collect(),
            lm_head_weight: vec![0.0; 18],
            final_norm_weight: vec![1.0; 3],
        };
        let head = burn_language::BurnQwenHead::<burn::backend::NdArray<f32>>::from_weights(
            weights, &device,
        );
        let image_features = burn::prelude::Tensor::<burn::backend::NdArray<f32>, 1>::from_floats(
            [100.0, 101.0, 102.0, 200.0, 201.0, 202.0].as_slice(),
            &device,
        )
        .reshape([2, 3]);
        let embeddings = head
            .embed_token_ids_with_image_features(&[2, 4, 4, 5], &[1, 2], image_features, &device)
            .unwrap();
        assert_eq!(embeddings.dims(), [1, 4, 3]);
        let data = embeddings
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(
            data,
            vec![
                6.0, 7.0, 8.0, 100.0, 101.0, 102.0, 200.0, 201.0, 202.0, 15.0, 16.0, 17.0,
            ]
        );
    }

    #[cfg(feature = "backend_ndarray")]
    #[test]
    fn image_feature_insertion_rejects_non_contiguous_prompt_span_ndarray() {
        use burn::tensor::backend::BackendTypes;

        let device = <burn::backend::NdArray<f32> as BackendTypes>::Device::default();
        let text_embeddings = burn::prelude::Tensor::<burn::backend::NdArray<f32>, 1>::from_floats(
            [0.0; 15].as_slice(),
            &device,
        )
        .reshape([1, 5, 3]);
        let image_features = burn::prelude::Tensor::<burn::backend::NdArray<f32>, 1>::from_floats(
            [1.0; 6].as_slice(),
            &device,
        )
        .reshape([2, 3]);
        let err = burn_language::insert_image_features(text_embeddings, &[1, 3], image_features)
            .unwrap_err();
        assert!(err.to_string().contains("must be contiguous"));
    }

    #[cfg(feature = "backend_ndarray")]
    #[test]
    fn qwen_layer_matches_reference_hook_when_enabled() {
        if std::env::var("LOCATE_ANYTHING_QWEN_LAYER_PARITY").is_err() {
            eprintln!(
                "skipping: set LOCATE_ANYTHING_QWEN_LAYER_PARITY=1 with LOCATE_ANYTHING_QWEN_LAYER_WEIGHTS and LOCATE_ANYTHING_QWEN_LAYER_HOOKS"
            );
            return;
        }
        use crate::tensor_io::load_tensor_from_safetensors_file;

        let weights_path = std::env::var("LOCATE_ANYTHING_QWEN_LAYER_WEIGHTS")
            .expect("LOCATE_ANYTHING_QWEN_LAYER_WEIGHTS");
        let hooks_path = std::env::var("LOCATE_ANYTHING_QWEN_LAYER_HOOKS")
            .expect("LOCATE_ANYTHING_QWEN_LAYER_HOOKS");
        let layer_index = env_usize("LOCATE_ANYTHING_QWEN_LAYER_INDEX", 0);
        let mean_tolerance = env_f32("LOCATE_ANYTHING_QWEN_LAYER_MEAN_TOLERANCE", 2.0e-5);
        let rms_tolerance = env_f32("LOCATE_ANYTHING_QWEN_LAYER_RMS_TOLERANCE", 5.0e-5);
        let max_tolerance = env_f32("LOCATE_ANYTHING_QWEN_LAYER_MAX_TOLERANCE", 1.0e-3);

        let device =
            <burn::backend::NdArray<f32> as burn::tensor::backend::BackendTypes>::Device::default();
        let weights = QwenDecoderLayerWeights::from_safetensors_file(
            weights_path,
            layer_index,
            QwenLanguageConfig::default(),
        )
        .unwrap();
        let input = load_tensor_from_safetensors_file(&hooks_path, "language.layer_input").unwrap();
        let reference_key = format!("language.layer_{layer_index:02}");
        let reference = load_tensor_from_safetensors_file(&hooks_path, &reference_key).unwrap();
        let input_shape: [usize; 3] = input.shape.clone().try_into().unwrap();
        let reference_shape: [usize; 3] = reference.shape.clone().try_into().unwrap();
        assert_eq!(input_shape, reference_shape);
        let model =
            burn_language::BurnQwenDecoderLayer::<burn::backend::NdArray<f32>>::from_weights(
                weights, &device,
            );
        let hidden = burn::prelude::Tensor::<burn::backend::NdArray<f32>, 1>::from_floats(
            input.data.as_slice(),
            &device,
        )
        .reshape(input_shape);
        let started = Instant::now();
        let output = model.forward(hidden, None);
        let output_data = output
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("tensor data");
        let forward_readback_ms = started.elapsed().as_secs_f64() * 1000.0;
        let stats = stats(&output_data, &reference.data);
        eprintln!(
            "Qwen layer parity layer={layer_index} tokens={} forward_readback_ms={forward_readback_ms:.3} mean_abs={:.6e} rms={:.6e} max_abs={:.6e}",
            reference_shape[1], stats.mean_abs, stats.rms, stats.max_abs
        );
        assert!(
            stats.mean_abs < mean_tolerance,
            "Qwen layer mean_abs {:.6e} exceeded tolerance {:.6e}",
            stats.mean_abs,
            mean_tolerance
        );
        assert!(
            stats.rms < rms_tolerance,
            "Qwen layer rms {:.6e} exceeded tolerance {:.6e}",
            stats.rms,
            rms_tolerance
        );
        assert!(
            stats.max_abs < max_tolerance,
            "Qwen layer max_abs {:.6e} exceeded tolerance {:.6e}",
            stats.max_abs,
            max_tolerance
        );
    }

    #[cfg(feature = "backend_ndarray")]
    #[derive(Clone, Copy, Debug)]
    struct Stats {
        mean_abs: f32,
        rms: f32,
        max_abs: f32,
    }

    #[cfg(feature = "backend_ndarray")]
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

    #[cfg(feature = "backend_ndarray")]
    fn env_f32(name: &str, default: f32) -> f32 {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse::<f32>().ok())
            .unwrap_or(default)
    }

    #[cfg(feature = "backend_ndarray")]
    fn env_usize(name: &str, default: usize) -> usize {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(default)
    }
}

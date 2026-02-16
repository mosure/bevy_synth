use burn::{module::Ignored, nn, prelude::*, tensor::Int};

use super::{
    components::{CrossAttention, FeedForward, record_tensor},
    hooks::HookRecorder,
};

#[derive(Config, Debug)]
pub struct TripoSGDiTConfig {
    pub in_channels: usize,
    pub width: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub cross_attention_dim: usize,
    pub cross_attention_2_dim: Option<usize>,
}

impl TripoSGDiTConfig {
    pub fn midi_3d() -> Self {
        Self {
            in_channels: 64,
            width: 2048,
            num_layers: 21,
            num_attention_heads: 16,
            cross_attention_dim: 768,
            cross_attention_2_dim: Some(1024),
        }
    }

    pub fn triposg_pretrained() -> Self {
        Self {
            in_channels: 64,
            width: 2048,
            num_layers: 21,
            num_attention_heads: 16,
            cross_attention_dim: 1024,
            cross_attention_2_dim: None,
        }
    }

    #[cfg(feature = "import")]
    pub fn from_config_bytes(bytes: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        let config: TripoSGDiTConfigFile = serde_json::from_slice(bytes)?;
        Ok(Self {
            in_channels: config.in_channels.unwrap_or(64),
            width: config.width.unwrap_or(2048),
            num_layers: config.num_layers.unwrap_or(21),
            num_attention_heads: config.num_attention_heads.unwrap_or(16),
            cross_attention_dim: config.cross_attention_dim.unwrap_or(768),
            cross_attention_2_dim: config.cross_attention_2_dim,
        })
    }

    #[cfg(feature = "import")]
    pub fn from_config_file(
        path: impl AsRef<std::path::Path>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(path)?;
        Self::from_config_bytes(&bytes)
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> TripoSGDiT<B> {
        TripoSGDiT::new(device, self.clone())
    }
}

#[cfg(feature = "import")]
#[derive(serde::Deserialize)]
struct TripoSGDiTConfigFile {
    cross_attention_dim: Option<usize>,
    cross_attention_2_dim: Option<usize>,
    in_channels: Option<usize>,
    num_attention_heads: Option<usize>,
    num_layers: Option<usize>,
    width: Option<usize>,
}

#[derive(Module, Debug)]
pub struct TimestepEmbedding<B: Backend> {
    pub linear_1: nn::Linear<B>,
    pub linear_2: nn::Linear<B>,
    pub activation: nn::Gelu,
}

impl<B: Backend> TimestepEmbedding<B> {
    pub fn new(device: &B::Device, in_dim: usize, hidden_dim: usize, out_dim: usize) -> Self {
        let linear_1 = nn::LinearConfig::new(in_dim, hidden_dim)
            .with_bias(true)
            .init(device);
        let linear_2 = nn::LinearConfig::new(hidden_dim, out_dim)
            .with_bias(true)
            .init(device);
        let activation = nn::Gelu::new();

        Self {
            linear_1,
            linear_2,
            activation,
        }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear_1.forward(x);
        let x = self.activation.forward(x);
        self.linear_2.forward(x)
    }
}

fn timestep_embedding<B: Backend>(
    timesteps: Tensor<B, 1>,
    embedding_dim: usize,
    flip_sin_to_cos: bool,
    downscale_freq_shift: f32,
    scale: f32,
) -> Tensor<B, 2> {
    let [batch] = timesteps.shape().dims();
    let half = embedding_dim / 2;
    let device = timesteps.device();

    let exponent = Tensor::<B, 1, Int>::arange(0..half as i64, &device).float();
    let exponent = exponent
        .mul_scalar(-(10000.0_f32).ln())
        .div_scalar(half as f32 - downscale_freq_shift);
    let emb = exponent.exp();
    let emb = timesteps.clone().unsqueeze_dim(1).mul(emb.unsqueeze_dim(0));
    let emb = emb.mul_scalar(scale);

    let sin = emb.clone().sin();
    let cos = emb.cos();
    let mut out = if flip_sin_to_cos {
        Tensor::cat(vec![cos, sin], 1)
    } else {
        Tensor::cat(vec![sin, cos], 1)
    };

    if embedding_dim % 2 == 1 {
        let pad = Tensor::<B, 2>::zeros([batch, 1], &device);
        out = Tensor::cat(vec![out, pad], 1);
    }

    out
}

#[derive(Module, Debug)]
pub struct TripoSGDiTBlock<B: Backend> {
    pub norm1: nn::LayerNorm<B>,
    pub attn1: CrossAttention<B>,
    pub norm2: nn::LayerNorm<B>,
    pub attn2: CrossAttention<B>,
    pub norm2_2: Option<nn::LayerNorm<B>>,
    pub attn2_2: Option<CrossAttention<B>>,
    pub norm3: nn::LayerNorm<B>,
    pub ff: FeedForward<B>,
    pub skip_norm: Option<nn::LayerNorm<B>>,
    pub skip_linear: Option<nn::Linear<B>>,
    use_self_attention: bool,
    use_cross_attention: bool,
    use_cross_attention_2: bool,
    use_skip: bool,
    skip_concat_front: bool,
    skip_norm_last: bool,
}

impl<B: Backend> TripoSGDiTBlock<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &B::Device,
        dim: usize,
        num_heads: usize,
        cross_attention_dim: usize,
        cross_attention_2_dim: Option<usize>,
        use_self_attention: bool,
        use_cross_attention: bool,
        use_cross_attention_2: bool,
        use_skip: bool,
        skip_concat_front: bool,
        skip_norm_last: bool,
    ) -> Self {
        let norm1 = nn::LayerNormConfig::new(dim).init(device);
        let attn1 =
            CrossAttention::new(device, dim, dim, num_heads, false, true, false, true, false);
        let norm2 = nn::LayerNormConfig::new(dim).init(device);
        let attn2 = CrossAttention::new(
            device,
            dim,
            cross_attention_dim,
            num_heads,
            false,
            true,
            false,
            true,
            true,
        );
        let (norm2_2, attn2_2) = if use_cross_attention_2 {
            let dim2 = cross_attention_2_dim.expect("cross_attention_2_dim required");
            let norm2_2 = nn::LayerNormConfig::new(dim).init(device);
            let attn2_2 =
                CrossAttention::new(device, dim, dim2, num_heads, false, true, false, true, true);
            (Some(norm2_2), Some(attn2_2))
        } else {
            (None, None)
        };
        let norm3 = nn::LayerNormConfig::new(dim).init(device);
        let ff = FeedForward::new(device, dim, dim * 4);

        let (skip_norm, skip_linear) = if use_skip {
            let skip_norm = nn::LayerNormConfig::new(dim).init(device);
            let skip_linear = nn::LinearConfig::new(dim * 2, dim)
                .with_bias(true)
                .init(device);
            (Some(skip_norm), Some(skip_linear))
        } else {
            (None, None)
        };

        Self {
            norm1,
            attn1,
            norm2,
            attn2,
            norm2_2,
            attn2_2,
            norm3,
            ff,
            skip_norm,
            skip_linear,
            use_self_attention,
            use_cross_attention,
            use_cross_attention_2,
            use_skip,
            skip_concat_front,
            skip_norm_last,
        }
    }

    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        encoder_hidden_states: Tensor<B, 3>,
        encoder_hidden_states_2: Option<Tensor<B, 3>>,
        skip: Option<Tensor<B, 3>>,
        mut hook: Option<&mut HookRecorder>,
        idx: usize,
    ) -> Tensor<B, 3> {
        let prefix = format!("dit.blocks.{idx}");
        let mut hidden = hidden_states;

        if self.use_skip {
            let skip = skip.expect("skip tensor required for this block");
            let skip_norm = self
                .skip_norm
                .as_ref()
                .expect("skip_norm missing for skip block");
            let skip_linear = self
                .skip_linear
                .as_ref()
                .expect("skip_linear missing for skip block");
            let cat = if self.skip_concat_front {
                Tensor::cat(vec![skip, hidden], 2)
            } else {
                Tensor::cat(vec![hidden, skip], 2)
            };

            if self.skip_norm_last {
                let out = skip_linear.forward(cat);
                hidden = skip_norm.forward(out);
            } else {
                let out = skip_norm.forward(cat);
                hidden = skip_linear.forward(out);
            }

            record_tensor(&mut hook, &format!("{prefix}.skip"), &hidden);
        }

        if self.use_self_attention {
            let norm_hidden = self.norm1.forward(hidden.clone());
            record_tensor(&mut hook, &format!("{prefix}.norm1"), &norm_hidden);
            let attn = self.attn1.forward(
                norm_hidden.clone(),
                norm_hidden,
                hook.as_deref_mut(),
                &format!("{prefix}.attn1"),
            );
            hidden = hidden + attn;
            record_tensor(&mut hook, &format!("{prefix}.attn1_out"), &hidden);
        }

        if self.use_cross_attention {
            if self.use_cross_attention_2 {
                let norm_hidden = self.norm2.forward(hidden.clone());
                record_tensor(&mut hook, &format!("{prefix}.norm2"), &norm_hidden);
                let attn2 = self.attn2.forward(
                    norm_hidden,
                    encoder_hidden_states.clone(),
                    hook.as_deref_mut(),
                    &format!("{prefix}.attn2"),
                );

                let enc2 = encoder_hidden_states_2.expect("encoder_hidden_states_2 required");
                let norm2_2 = self.norm2_2.as_ref().expect("norm2_2 required");
                let attn2_2 = self.attn2_2.as_ref().expect("attn2_2 required");
                let norm_hidden = norm2_2.forward(hidden.clone());
                record_tensor(&mut hook, &format!("{prefix}.norm2_2"), &norm_hidden);
                let attn2_2 = attn2_2.forward(
                    norm_hidden,
                    enc2,
                    hook.as_deref_mut(),
                    &format!("{prefix}.attn2_2"),
                );

                hidden = hidden + attn2 + attn2_2;
                record_tensor(&mut hook, &format!("{prefix}.attn2_out"), &hidden);
            } else {
                let norm_hidden = self.norm2.forward(hidden.clone());
                record_tensor(&mut hook, &format!("{prefix}.norm2"), &norm_hidden);
                let attn = self.attn2.forward(
                    norm_hidden,
                    encoder_hidden_states,
                    hook.as_deref_mut(),
                    &format!("{prefix}.attn2"),
                );
                hidden = hidden + attn;
                record_tensor(&mut hook, &format!("{prefix}.attn2_out"), &hidden);
            }
        }

        let norm_hidden = self.norm3.forward(hidden.clone());
        record_tensor(&mut hook, &format!("{prefix}.norm3"), &norm_hidden);
        let ff = self
            .ff
            .forward(norm_hidden, hook.as_deref_mut(), &format!("{prefix}.ff"));
        let hidden = hidden + ff;
        record_tensor(&mut hook, &format!("{prefix}.out"), &hidden);
        hidden
    }
}

#[derive(Module, Debug)]
pub struct TripoSGDiT<B: Backend> {
    config: Ignored<TripoSGDiTConfig>,
    pub time_proj: TimestepEmbedding<B>,
    pub proj_in: nn::Linear<B>,
    pub blocks: Vec<TripoSGDiTBlock<B>>,
    pub norm_out: nn::LayerNorm<B>,
    pub proj_out: nn::Linear<B>,
    inner_dim: usize,
}

impl<B: Backend> TripoSGDiT<B> {
    pub fn new(device: &B::Device, config: TripoSGDiTConfig) -> Self {
        let inner_dim = config.width;
        let time_embed_dim = inner_dim * 4;
        let time_proj = TimestepEmbedding::new(device, inner_dim, time_embed_dim, inner_dim);
        let proj_in = nn::LinearConfig::new(config.in_channels, inner_dim)
            .with_bias(true)
            .init(device);

        let mut blocks = Vec::with_capacity(config.num_layers);
        let half = config.num_layers / 2;
        let use_cross_attention_2 = config.cross_attention_2_dim.is_some();
        for layer in 0..config.num_layers {
            let use_skip = layer > half;
            blocks.push(TripoSGDiTBlock::new(
                device,
                inner_dim,
                config.num_attention_heads,
                config.cross_attention_dim,
                config.cross_attention_2_dim,
                true,
                true,
                use_cross_attention_2,
                use_skip,
                true,
                true,
            ));
        }

        let norm_out = nn::LayerNormConfig::new(inner_dim).init(device);
        let proj_out = nn::LinearConfig::new(inner_dim, config.in_channels)
            .with_bias(true)
            .init(device);

        Self {
            config: Ignored(config),
            time_proj,
            proj_in,
            blocks,
            norm_out,
            proj_out,
            inner_dim,
        }
    }

    pub fn config(&self) -> &TripoSGDiTConfig {
        &self.config
    }

    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        timestep: Tensor<B, 1>,
        encoder_hidden_states: Tensor<B, 3>,
        encoder_hidden_states_2: Option<Tensor<B, 3>>,
        mut hook: Option<&mut HookRecorder>,
    ) -> Tensor<B, 3> {
        let [batch, n, _] = hidden_states.shape().dims();
        let temb = timestep_embedding(timestep, self.inner_dim, false, 0.0, 1.0);
        record_tensor(&mut hook, "dit.temb", &temb);
        let temb = self.time_proj.forward(temb);
        record_tensor(&mut hook, "dit.temb_proj", &temb);
        let temb = temb.unsqueeze_dim(1);

        let hidden = self.proj_in.forward(hidden_states);
        record_tensor(&mut hook, "dit.proj_in", &hidden);
        let mut hidden = Tensor::cat(vec![temb.clone(), hidden], 1);
        record_tensor(&mut hook, "dit.tokens", &hidden);

        let mut skips = Vec::new();
        let half = self.blocks.len() / 2;
        for (idx, block) in self.blocks.iter().enumerate() {
            let skip = if idx > half { skips.pop() } else { None };
            hidden = block.forward(
                hidden,
                encoder_hidden_states.clone(),
                encoder_hidden_states_2.clone(),
                skip,
                hook.as_deref_mut(),
                idx,
            );

            if idx < half {
                skips.push(hidden.clone());
            }
        }

        let hidden = self.norm_out.forward(hidden);
        record_tensor(&mut hook, "dit.norm_out", &hidden);
        let hidden = hidden.slice([0..batch, 1..(n + 1), 0..self.inner_dim]);
        let hidden = self.proj_out.forward(hidden);
        record_tensor(&mut hook, "dit.proj_out", &hidden);
        hidden
    }
}

#[cfg(feature = "import")]
pub mod import {
    use std::path::{Path, PathBuf};

    use burn::module::{Module, ModuleMapper, Param};
    use burn::prelude::*;
    use burn::tensor::Bytes;
    use burn::tensor::FloatDType;
    use burn_store::{
        BurnpackStore, KeyRemapper, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore,
    };

    use super::super::load_policy::{BurnpackLoadPolicy, burnpack_path, candidate_burnpack_paths};
    use super::{TripoSGDiT, TripoSGDiTConfig};

    pub fn load_triposg_dit<B: Backend>(
        config: &TripoSGDiTConfig,
        device: &B::Device,
        path: impl AsRef<Path>,
    ) -> Result<TripoSGDiT<B>, Box<dyn std::error::Error>> {
        load_triposg_dit_with_policy(config, device, path, default_burnpack_policy())
    }

    pub fn load_triposg_dit_with_policy<B: Backend>(
        config: &TripoSGDiTConfig,
        device: &B::Device,
        path: impl AsRef<Path>,
        policy: BurnpackLoadPolicy,
    ) -> Result<TripoSGDiT<B>, Box<dyn std::error::Error>> {
        let path = path.as_ref();
        let burnpack_candidates = candidate_burnpack_paths(path, policy);
        let burnpack_path = burnpack_candidates
            .iter()
            .find(|candidate| candidate.exists())
            .cloned();
        let Some(burnpack_path) = burnpack_path else {
            let checked = burnpack_candidates
                .iter()
                .map(|candidate| candidate.display().to_string())
                .collect::<Vec<_>>()
                .join(", ");
            return Err(format!(
                "Burnpack weights missing. Checked: {checked}. Run `triposg_import` to generate .bpk files."
            )
            .into());
        };

        let mut model = TripoSGDiT::new(device, config.clone());
        let mut store =
            BurnpackStore::from_file(&burnpack_path).validate(should_validate_burnpack());
        model
            .load_from(&mut store)
            .map_err(|err| format!("failed to load TripoSG DiT burnpack: {err}"))?;
        Ok(model)
    }

    pub fn load_triposg_dit_from_burnpack_bytes<B: Backend>(
        config: &TripoSGDiTConfig,
        device: &B::Device,
        burnpack_bytes: Vec<u8>,
    ) -> Result<TripoSGDiT<B>, Box<dyn std::error::Error>> {
        let mut model = TripoSGDiT::new(device, config.clone());
        let mut store = BurnpackStore::from_bytes(Some(Bytes::from_bytes_vec(burnpack_bytes)))
            .validate(should_validate_burnpack());
        model
            .load_from(&mut store)
            .map_err(|err| format!("failed to load TripoSG DiT burnpack bytes: {err}"))?;
        Ok(model)
    }

    pub fn apply_triposg_dit_burnpack_part_bytes<B: Backend>(
        model: &mut TripoSGDiT<B>,
        burnpack_bytes: Vec<u8>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut store = BurnpackStore::from_bytes(Some(Bytes::from_bytes_vec(burnpack_bytes)))
            .allow_partial(true)
            .validate(should_validate_burnpack());
        model
            .load_from(&mut store)
            .map_err(|err| format!("failed to load TripoSG DiT burnpack part bytes: {err}"))?;
        Ok(())
    }

    pub fn load_triposg_dit_from_burnpack_file<B: Backend>(
        config: &TripoSGDiTConfig,
        device: &B::Device,
        burnpack_path: impl AsRef<Path>,
    ) -> Result<TripoSGDiT<B>, Box<dyn std::error::Error>> {
        let mut model = TripoSGDiT::new(device, config.clone());
        let mut store =
            BurnpackStore::from_file(burnpack_path.as_ref()).validate(should_validate_burnpack());
        model
            .load_from(&mut store)
            .map_err(|err| format!("failed to load TripoSG DiT burnpack file: {err}"))?;
        Ok(model)
    }

    fn default_burnpack_policy() -> BurnpackLoadPolicy {
        BurnpackLoadPolicy::default()
    }

    fn should_validate_burnpack() -> bool {
        cfg!(all(not(target_arch = "wasm32"), debug_assertions))
    }

    pub fn load_triposg_dit_from_safetensors<B: Backend>(
        config: &TripoSGDiTConfig,
        device: &B::Device,
        path: impl AsRef<Path>,
    ) -> Result<TripoSGDiT<B>, Box<dyn std::error::Error>> {
        let mut model = TripoSGDiT::new(device, config.clone());
        let mut store = build_store(path.as_ref())?;
        model
            .load_from(&mut store)
            .map_err(|err| format!("failed to apply TripoSG DiT weights: {err}"))?;
        Ok(model)
    }

    pub fn import_triposg_dit_burnpack<B: Backend>(
        config: &TripoSGDiTConfig,
        device: &B::Device,
        path: impl AsRef<Path>,
        use_f16: bool,
    ) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let path = path.as_ref();
        let burnpack_path = burnpack_path(path, use_f16, BurnpackLoadPolicy::default().f16_suffix);
        let model = load_triposg_dit_from_safetensors::<B>(config, device, path)?;
        let model = if use_f16 {
            cast_module_float_dtype(model, FloatDType::F16)
        } else {
            model
        };
        save_burnpack(&model, &burnpack_path)?;
        Ok(burnpack_path)
    }

    struct FloatDTypeMapper {
        dtype: FloatDType,
    }

    impl<B: Backend> ModuleMapper<B> for FloatDTypeMapper {
        fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
            let (id, tensor, mapper) = param.consume();
            let tensor = tensor.cast(self.dtype);
            Param::from_mapped_value(id, tensor, mapper)
        }
    }

    fn cast_module_float_dtype<B: Backend, M: Module<B>>(module: M, dtype: FloatDType) -> M {
        let mut mapper = FloatDTypeMapper { dtype };
        module.map(&mut mapper)
    }

    fn save_burnpack<B: Backend>(
        model: &TripoSGDiT<B>,
        path: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut store = BurnpackStore::from_file(path).overwrite(true);
        model
            .save_into(&mut store)
            .map_err(|err| format!("failed to save TripoSG DiT burnpack: {err}"))?;
        Ok(())
    }

    fn build_store(path: &Path) -> Result<SafetensorsStore, Box<dyn std::error::Error>> {
        let mut remapper = KeyRemapper::new();
        for &(from, to) in key_remap_rules() {
            remapper = remapper
                .add_pattern(from, to)
                .map_err(|err| format!("invalid remap rule {from}->{to}: {err}"))?;
        }

        let store = SafetensorsStore::from_file(path)
            .with_from_adapter(PyTorchToBurnAdapter)
            .allow_partial(false)
            .remap(remapper)
            .validate(true);

        Ok(store)
    }

    fn key_remap_rules() -> &'static [(&'static str, &'static str)] {
        &[
            (r"^(blocks\.\d+\.attn1\.to_out)\.0\.(weight|bias)$", "$1.$2"),
            (r"^(blocks\.\d+\.attn2\.to_out)\.0\.(weight|bias)$", "$1.$2"),
            (
                r"^(blocks\.\d+\.attn2_2\.to_out)\.0\.(weight|bias)$",
                "$1.$2",
            ),
            (
                r"^(blocks\.\d+\.ff)\.net\.0\.proj\.(weight|bias)$",
                "$1.proj.$2",
            ),
            (r"^(blocks\.\d+\.ff)\.net\.2\.(weight|bias)$", "$1.out.$2"),
            (r"^(blocks\.\d+\.norm1)\.weight$", "$1.gamma"),
            (r"^(blocks\.\d+\.norm1)\.bias$", "$1.beta"),
            (r"^(blocks\.\d+\.norm2)\.weight$", "$1.gamma"),
            (r"^(blocks\.\d+\.norm2)\.bias$", "$1.beta"),
            (r"^(blocks\.\d+\.norm2_2)\.weight$", "$1.gamma"),
            (r"^(blocks\.\d+\.norm2_2)\.bias$", "$1.beta"),
            (r"^(blocks\.\d+\.norm3)\.weight$", "$1.gamma"),
            (r"^(blocks\.\d+\.norm3)\.bias$", "$1.beta"),
            (r"^(blocks\.\d+\.skip_norm)\.weight$", "$1.gamma"),
            (r"^(blocks\.\d+\.skip_norm)\.bias$", "$1.beta"),
            (r"^(blocks\.\d+\.attn1\.norm_q)\.weight$", "$1.gamma"),
            (r"^(blocks\.\d+\.attn1\.norm_k)\.weight$", "$1.gamma"),
            (r"^(blocks\.\d+\.attn2\.norm_q)\.weight$", "$1.gamma"),
            (r"^(blocks\.\d+\.attn2\.norm_k)\.weight$", "$1.gamma"),
            (r"^(blocks\.\d+\.attn2_2\.norm_q)\.weight$", "$1.gamma"),
            (r"^(blocks\.\d+\.attn2_2\.norm_k)\.weight$", "$1.gamma"),
            (r"^(norm_out)\.weight$", "$1.gamma"),
            (r"^(norm_out)\.bias$", "$1.beta"),
        ]
    }
}

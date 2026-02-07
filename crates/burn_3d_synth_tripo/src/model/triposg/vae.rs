use burn::{module::Ignored, nn, prelude::*};

use super::{
    components::{
        CrossAttention, DiagonalGaussianDistribution, FeedForward, FrequencyPositionalEmbedding,
        record_tensor,
    },
    hooks::HookRecorder,
};

#[derive(Config, Debug)]
pub struct TripoSGVaeConfig {
    pub embed_frequency: usize,
    pub embed_include_pi: bool,
    pub embedding_type: String,
    pub in_channels: usize,
    pub latent_channels: usize,
    pub num_attention_heads: usize,
    pub num_layers_decoder: usize,
    pub num_layers_encoder: usize,
    pub width_decoder: usize,
    pub width_encoder: usize,
}

impl TripoSGVaeConfig {
    pub fn midi_3d() -> Self {
        Self {
            embed_frequency: 8,
            embed_include_pi: false,
            embedding_type: "frequency".to_string(),
            in_channels: 3,
            latent_channels: 64,
            num_attention_heads: 8,
            num_layers_decoder: 16,
            num_layers_encoder: 8,
            width_decoder: 1024,
            width_encoder: 512,
        }
    }

    #[cfg(feature = "import")]
    pub fn from_config_bytes(bytes: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        let config: TripoSGVaeConfigFile = serde_json::from_slice(bytes)?;
        Ok(Self {
            embed_frequency: config.embed_frequency.unwrap_or(8),
            embed_include_pi: config.embed_include_pi.unwrap_or(false),
            embedding_type: config
                .embedding_type
                .unwrap_or_else(|| "frequency".to_string()),
            in_channels: config.in_channels.unwrap_or(3),
            latent_channels: config.latent_channels.unwrap_or(64),
            num_attention_heads: config.num_attention_heads.unwrap_or(8),
            num_layers_decoder: config.num_layers_decoder.unwrap_or(16),
            num_layers_encoder: config.num_layers_encoder.unwrap_or(8),
            width_decoder: config.width_decoder.unwrap_or(1024),
            width_encoder: config.width_encoder.unwrap_or(512),
        })
    }

    #[cfg(feature = "import")]
    pub fn from_config_file(
        path: impl AsRef<std::path::Path>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(path)?;
        Self::from_config_bytes(&bytes)
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> TripoSGVae<B> {
        TripoSGVae::new(device, self.clone())
    }
}

#[cfg(feature = "import")]
#[derive(serde::Deserialize)]
struct TripoSGVaeConfigFile {
    embed_frequency: Option<usize>,
    embed_include_pi: Option<bool>,
    embedding_type: Option<String>,
    in_channels: Option<usize>,
    latent_channels: Option<usize>,
    num_attention_heads: Option<usize>,
    num_layers_decoder: Option<usize>,
    num_layers_encoder: Option<usize>,
    width_decoder: Option<usize>,
    width_encoder: Option<usize>,
}

#[derive(Debug)]
pub struct TripoSGVaeOutput<B: Backend> {
    pub mean: Tensor<B, 3>,
    pub logvar: Tensor<B, 3>,
    pub latent: Tensor<B, 3>,
    pub decoded: Tensor<B, 3>,
}

#[derive(Module, Debug)]
pub struct TripoSGVae<B: Backend> {
    pub encoder: TripoSGEncoder<B>,
    pub decoder: TripoSGDecoder<B>,
    pub quant: nn::Linear<B>,
    pub post_quant: nn::Linear<B>,
    freq_embed: Ignored<FrequencyPositionalEmbedding>,
    in_channels: usize,
    latent_channels: usize,
}

impl<B: Backend> TripoSGVae<B> {
    pub fn new(device: &B::Device, config: TripoSGVaeConfig) -> Self {
        let freq_embed = FrequencyPositionalEmbedding {
            num_freq: config.embed_frequency,
            include_pi: config.embed_include_pi,
        };
        let embed_dim = freq_embed.embed_dim(3);
        let encoder_in_dim = embed_dim + config.in_channels;
        let decoder_in_dim = embed_dim;

        let encoder = TripoSGEncoder::new(
            device,
            encoder_in_dim,
            config.width_encoder,
            config.num_layers_encoder,
            config.num_attention_heads,
        );
        let decoder = TripoSGDecoder::new(
            device,
            decoder_in_dim,
            config.width_decoder,
            config.num_layers_decoder,
            config.num_attention_heads,
        );

        let quant = nn::LinearConfig::new(config.width_encoder, config.latent_channels * 2)
            .with_bias(true)
            .init(device);
        let post_quant = nn::LinearConfig::new(config.latent_channels, config.width_decoder)
            .with_bias(true)
            .init(device);

        Self {
            encoder,
            decoder,
            quant,
            post_quant,
            freq_embed: Ignored(freq_embed),
            in_channels: config.in_channels,
            latent_channels: config.latent_channels,
        }
    }

    pub fn encode(
        &self,
        coords: Tensor<B, 3>,
        features: Tensor<B, 3>,
        hook: Option<&mut HookRecorder>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        self.encode_with_kv(coords.clone(), features.clone(), coords, features, hook)
    }

    pub fn encode_with_kv(
        &self,
        coords_q: Tensor<B, 3>,
        features_q: Tensor<B, 3>,
        coords_kv: Tensor<B, 3>,
        features_kv: Tensor<B, 3>,
        hook: Option<&mut HookRecorder>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let mut hook = hook;
        let embedded_q = self.freq_embed.forward(coords_q);
        let embedded_kv = self.freq_embed.forward(coords_kv);
        let input_q = Tensor::cat(vec![embedded_q, features_q], 2);
        let input_kv = Tensor::cat(vec![embedded_kv, features_kv], 2);

        record_tensor(&mut hook, "encoder.input", &input_q);
        record_tensor(&mut hook, "encoder.input.kv", &input_kv);

        let hidden = self.encoder.forward(input_q, input_kv, hook.as_deref_mut());
        record_tensor(&mut hook, "encoder.hidden", &hidden);

        let stats = self.quant.forward(hidden);
        record_tensor(&mut hook, "encoder.quant", &stats);

        let [b, n, _] = stats.shape().dims();
        let mean = stats.clone().slice([0..b, 0..n, 0..self.latent_channels]);
        let logvar = stats.slice([0..b, 0..n, self.latent_channels..self.latent_channels * 2]);

        record_tensor(&mut hook, "encoder.mean", &mean);
        record_tensor(&mut hook, "encoder.logvar", &logvar);

        (mean, logvar)
    }

    pub fn prepare_latent_projection(
        &self,
        latents: Tensor<B, 3>,
        mut hook: Option<&mut HookRecorder>,
    ) -> Tensor<B, 3> {
        let latent_proj = self.post_quant.forward(latents);
        record_tensor(&mut hook, "decoder.post_quant", &latent_proj);
        latent_proj
    }

    pub fn build_kv_cache(
        &self,
        latent_proj: Tensor<B, 3>,
        hook: Option<&mut HookRecorder>,
    ) -> Tensor<B, 3> {
        self.decoder.build_kv_cache(latent_proj, hook)
    }

    pub fn decode_with_latent_projection(
        &self,
        query_coords: Tensor<B, 3>,
        latent_proj: Tensor<B, 3>,
        kv_cache: Option<Tensor<B, 3>>,
        mut hook: Option<&mut HookRecorder>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let query_embed = self.freq_embed.forward(query_coords);
        let query_tokens = self.decoder.proj_query.forward(query_embed);
        record_tensor(&mut hook, "decoder.query", &query_tokens);

        self.decoder
            .forward(latent_proj, query_tokens, kv_cache, hook)
    }

    pub fn decode(
        &self,
        query_coords: Tensor<B, 3>,
        latents: Tensor<B, 3>,
        mut hook: Option<&mut HookRecorder>,
    ) -> Tensor<B, 3> {
        let latent_proj = self.prepare_latent_projection(latents, hook.as_deref_mut());
        let (output, _kv_cache) =
            self.decode_with_latent_projection(query_coords, latent_proj, None, hook);
        output
    }

    pub fn forward(
        &self,
        coords: Tensor<B, 3>,
        features: Tensor<B, 3>,
        query_coords: Tensor<B, 3>,
        use_mean: bool,
        mut hook: Option<&mut HookRecorder>,
    ) -> TripoSGVaeOutput<B> {
        let (mean, logvar) = self.encode(coords, features, hook.as_deref_mut());
        let distribution = DiagonalGaussianDistribution::new(mean.clone(), logvar.clone());
        let latent = if use_mean {
            distribution.mode()
        } else {
            distribution.sample()
        };
        record_tensor(&mut hook, "latent.sample", &latent);

        let decoded = self.decode(query_coords, latent.clone(), hook);

        TripoSGVaeOutput {
            mean,
            logvar,
            latent,
            decoded,
        }
    }
}

#[derive(Module, Debug)]
pub struct TripoSGEncoder<B: Backend> {
    pub proj_in: nn::Linear<B>,
    pub blocks: Vec<TripoSGEncoderBlock<B>>,
    pub norm_out: nn::LayerNorm<B>,
}

impl<B: Backend> TripoSGEncoder<B> {
    pub fn new(
        device: &B::Device,
        in_dim: usize,
        width: usize,
        layers: usize,
        heads: usize,
    ) -> Self {
        let proj_in = nn::LinearConfig::new(in_dim, width)
            .with_bias(true)
            .init(device);
        let mut blocks = Vec::with_capacity(layers + 1);
        blocks.push(TripoSGEncoderBlock::new(device, width, heads, true));
        for _ in 0..layers {
            blocks.push(TripoSGEncoderBlock::new(device, width, heads, false));
        }
        let norm_out = nn::LayerNormConfig::new(width).init(device);

        Self {
            proj_in,
            blocks,
            norm_out,
        }
    }

    pub fn forward(
        &self,
        x_q: Tensor<B, 3>,
        x_kv: Tensor<B, 3>,
        mut hook: Option<&mut HookRecorder>,
    ) -> Tensor<B, 3> {
        let mut hidden = self.proj_in.forward(x_q);
        let kv = self.proj_in.forward(x_kv);

        record_tensor(&mut hook, "encoder.proj_in", &hidden);
        record_tensor(&mut hook, "encoder.proj_in.kv", &kv);

        for (idx, block) in self.blocks.iter().enumerate() {
            let context = if idx == 0 { kv.clone() } else { hidden.clone() };
            hidden = block.forward(hidden, context, hook.as_deref_mut(), idx);
        }

        let hidden = self.norm_out.forward(hidden);
        record_tensor(&mut hook, "encoder.norm_out", &hidden);
        hidden
    }
}

#[derive(Module, Debug)]
pub struct TripoSGEncoderBlock<B: Backend> {
    pub norm1: Option<nn::LayerNorm<B>>,
    pub attn1: Option<CrossAttention<B>>,
    pub norm2: Option<nn::LayerNorm<B>>,
    pub attn2: Option<CrossAttention<B>>,
    pub norm3: nn::LayerNorm<B>,
    pub ff: FeedForward<B>,
    use_cross: bool,
}

impl<B: Backend> TripoSGEncoderBlock<B> {
    pub fn new(device: &B::Device, width: usize, heads: usize, use_cross: bool) -> Self {
        let (norm1, attn1, norm2, attn2) = if use_cross {
            let norm2 = nn::LayerNormConfig::new(width).init(device);
            let attn2 =
                CrossAttention::new(device, width, width, heads, true, false, false, true, true);
            (None, None, Some(norm2), Some(attn2))
        } else {
            let norm1 = nn::LayerNormConfig::new(width).init(device);
            let attn1 = CrossAttention::new(
                device, width, width, heads, false, false, false, true, false,
            );
            (Some(norm1), Some(attn1), None, None)
        };
        let norm3 = nn::LayerNormConfig::new(width).init(device);
        let ff = FeedForward::new(device, width, width * 4);

        Self {
            norm1,
            attn1,
            norm2,
            attn2,
            norm3,
            ff,
            use_cross,
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        context: Tensor<B, 3>,
        mut hook: Option<&mut HookRecorder>,
        idx: usize,
    ) -> Tensor<B, 3> {
        let prefix = format!("encoder.blocks.{idx}");
        let attn = if self.use_cross {
            let norm2 = self.norm2.as_ref().expect("encoder cross norm2 missing");
            let attn2 = self.attn2.as_ref().expect("encoder cross attn2 missing");
            let x_norm = norm2.forward(x.clone());
            record_tensor(&mut hook, &format!("{prefix}.norm2"), &x_norm);
            attn2.forward(
                x_norm,
                context,
                hook.as_deref_mut(),
                &format!("{prefix}.attn2"),
            )
        } else {
            let norm1 = self.norm1.as_ref().expect("encoder self norm1 missing");
            let attn1 = self.attn1.as_ref().expect("encoder self attn1 missing");
            let x_norm = norm1.forward(x.clone());
            record_tensor(&mut hook, &format!("{prefix}.norm1"), &x_norm);
            attn1.forward(
                x_norm.clone(),
                x_norm,
                hook.as_deref_mut(),
                &format!("{prefix}.attn1"),
            )
        };
        let x = x + attn;
        record_tensor(&mut hook, &format!("{prefix}.attn_out"), &x);

        let x_norm = self.norm3.forward(x.clone());
        record_tensor(&mut hook, &format!("{prefix}.norm3"), &x_norm);
        let ff = self
            .ff
            .forward(x_norm, hook.as_deref_mut(), &format!("{prefix}.ff"));
        let x = x + ff;
        record_tensor(&mut hook, &format!("{prefix}.out"), &x);
        x
    }
}

#[derive(Module, Debug)]
pub struct TripoSGDecoder<B: Backend> {
    pub proj_query: nn::Linear<B>,
    pub blocks: Vec<TripoSGDecoderBlock<B>>,
    pub norm_out: nn::LayerNorm<B>,
    pub proj_out: nn::Linear<B>,
}

impl<B: Backend> TripoSGDecoder<B> {
    pub fn new(
        device: &B::Device,
        in_dim: usize,
        width: usize,
        layers: usize,
        heads: usize,
    ) -> Self {
        let proj_query = nn::LinearConfig::new(in_dim, width)
            .with_bias(true)
            .init(device);
        let mut blocks = Vec::with_capacity(layers + 1);
        for _ in 0..layers {
            blocks.push(TripoSGDecoderBlock::new(device, width, heads, false));
        }
        blocks.push(TripoSGDecoderBlock::new(device, width, heads, true));
        let norm_out = nn::LayerNormConfig::new(width).init(device);
        let proj_out = nn::LinearConfig::new(width, 1).with_bias(true).init(device);

        Self {
            proj_query,
            blocks,
            norm_out,
            proj_out,
        }
    }

    pub fn build_kv_cache(
        &self,
        sample: Tensor<B, 3>,
        mut hook: Option<&mut HookRecorder>,
    ) -> Tensor<B, 3> {
        let mut hidden = sample;
        let last_idx = self.blocks.len().saturating_sub(1);
        for (idx, block) in self.blocks.iter().enumerate().take(last_idx) {
            let context = hidden.clone();
            hidden = block.forward(hidden, context, hook.as_deref_mut(), idx);
        }
        record_tensor(&mut hook, "decoder.kv_cache", &hidden);
        hidden
    }

    pub fn forward(
        &self,
        sample: Tensor<B, 3>,
        queries: Tensor<B, 3>,
        kv_cache: Option<Tensor<B, 3>>,
        mut hook: Option<&mut HookRecorder>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let kv_cache = if let Some(cache) = kv_cache {
            cache
        } else {
            self.build_kv_cache(sample, hook.as_deref_mut())
        };

        let cross_idx = self.blocks.len().saturating_sub(1);
        let cross_block = &self.blocks[cross_idx];
        let hidden = cross_block.forward(queries, kv_cache.clone(), hook.as_deref_mut(), cross_idx);

        let hidden = self.norm_out.forward(hidden);
        record_tensor(&mut hook, "decoder.norm_out", &hidden);
        let hidden = self.proj_out.forward(hidden);
        record_tensor(&mut hook, "decoder.proj_out", &hidden);

        let output = hidden.mul_scalar(-1.0);
        record_tensor(&mut hook, "decoder.output", &output);

        (output, kv_cache)
    }
}

#[derive(Module, Debug)]
pub struct TripoSGDecoderBlock<B: Backend> {
    pub norm1: Option<nn::LayerNorm<B>>,
    pub attn1: Option<CrossAttention<B>>,
    pub norm2: Option<nn::LayerNorm<B>>,
    pub attn2: Option<CrossAttention<B>>,
    pub norm3: nn::LayerNorm<B>,
    pub ff: FeedForward<B>,
    use_cross: bool,
}

impl<B: Backend> TripoSGDecoderBlock<B> {
    pub fn new(device: &B::Device, width: usize, heads: usize, use_cross: bool) -> Self {
        let (norm1, attn1, norm2, attn2) = if use_cross {
            let norm2 = nn::LayerNormConfig::new(width).init(device);
            let attn2 =
                CrossAttention::new(device, width, width, heads, true, false, false, true, true);
            (None, None, Some(norm2), Some(attn2))
        } else {
            let norm1 = nn::LayerNormConfig::new(width).init(device);
            let attn1 = CrossAttention::new(
                device, width, width, heads, false, false, false, true, false,
            );
            (Some(norm1), Some(attn1), None, None)
        };
        let norm3 = nn::LayerNormConfig::new(width).init(device);
        let ff = FeedForward::new(device, width, width * 4);

        Self {
            norm1,
            attn1,
            norm2,
            attn2,
            norm3,
            ff,
            use_cross,
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        context: Tensor<B, 3>,
        mut hook: Option<&mut HookRecorder>,
        idx: usize,
    ) -> Tensor<B, 3> {
        let prefix = format!("decoder.blocks.{idx}");

        let attn = if self.use_cross {
            let norm2 = self.norm2.as_ref().expect("decoder cross norm2 missing");
            let attn2 = self.attn2.as_ref().expect("decoder cross attn2 missing");
            let x_norm = norm2.forward(x.clone());
            record_tensor(&mut hook, &format!("{prefix}.norm2"), &x_norm);
            attn2.forward(
                x_norm,
                context,
                hook.as_deref_mut(),
                &format!("{prefix}.attn2"),
            )
        } else {
            let norm1 = self.norm1.as_ref().expect("decoder self norm1 missing");
            let attn1 = self.attn1.as_ref().expect("decoder self attn1 missing");
            let x_norm = norm1.forward(x.clone());
            record_tensor(&mut hook, &format!("{prefix}.norm1"), &x_norm);
            attn1.forward(
                x_norm.clone(),
                x_norm,
                hook.as_deref_mut(),
                &format!("{prefix}.attn1"),
            )
        };
        let x = x + attn;
        record_tensor(&mut hook, &format!("{prefix}.attn_out"), &x);

        let x_norm = self.norm3.forward(x.clone());
        record_tensor(&mut hook, &format!("{prefix}.norm3"), &x_norm);
        let ff = self
            .ff
            .forward(x_norm, hook.as_deref_mut(), &format!("{prefix}.ff"));
        let x = x + ff;
        record_tensor(&mut hook, &format!("{prefix}.out"), &x);
        x
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

    use super::{TripoSGVae, TripoSGVaeConfig};

    const F16_SUFFIX: &str = "_f16";

    pub fn load_triposg_vae<B: Backend>(
        config: &TripoSGVaeConfig,
        device: &B::Device,
        path: impl AsRef<Path>,
    ) -> Result<TripoSGVae<B>, Box<dyn std::error::Error>> {
        let path = path.as_ref();
        let burnpack_candidates = candidate_burnpack_paths(path);
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

        let mut model = TripoSGVae::new(device, config.clone());
        let mut store = BurnpackStore::from_file(&burnpack_path).validate(true);
        model
            .load_from(&mut store)
            .map_err(|err| format!("failed to load TripoSG VAE burnpack: {err}"))?;
        Ok(model)
    }

    pub fn load_triposg_vae_from_burnpack_bytes<B: Backend>(
        config: &TripoSGVaeConfig,
        device: &B::Device,
        burnpack_bytes: Vec<u8>,
    ) -> Result<TripoSGVae<B>, Box<dyn std::error::Error>> {
        let mut model = TripoSGVae::new(device, config.clone());
        let mut store = BurnpackStore::from_bytes(Some(Bytes::from_bytes_vec(burnpack_bytes)))
            .validate(true);
        model
            .load_from(&mut store)
            .map_err(|err| format!("failed to load TripoSG VAE burnpack bytes: {err}"))?;
        Ok(model)
    }

    fn candidate_burnpack_paths(path: &Path) -> Vec<PathBuf> {
        let default = burnpack_path(path, false);
        let f16 = burnpack_path(path, true);
        if f16 == default {
            vec![default]
        } else if prefer_f16_burnpack() {
            vec![f16, default]
        } else {
            vec![default, f16]
        }
    }

    fn prefer_f16_burnpack() -> bool {
        preferred_precision_from_env("TRIPOSG_BPK_PRECISION", "BURN_3D_SYNTH_BPK_PRECISION")
    }

    fn preferred_precision_from_env(primary: &str, fallback: &str) -> bool {
        let value = std::env::var(primary)
            .ok()
            .or_else(|| std::env::var(fallback).ok());
        match value
            .as_deref()
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("f32" | "fp32" | "float32" | "32") => false,
            Some("f16" | "fp16" | "float16" | "half" | "16") => true,
            Some(_) | None => true,
        }
    }

    fn burnpack_path(path: &Path, use_f16: bool) -> PathBuf {
        let path = if path
            .extension()
            .map(|ext| ext.eq_ignore_ascii_case("bpk"))
            .unwrap_or(false)
        {
            path.to_path_buf()
        } else {
            path.with_extension("bpk")
        };

        if use_f16 {
            with_file_stem_suffix(&path, F16_SUFFIX)
        } else {
            path
        }
    }

    fn with_file_stem_suffix(path: &Path, suffix: &str) -> PathBuf {
        let Some(stem) = path.file_stem() else {
            return path.to_path_buf();
        };
        let stem = stem.to_string_lossy();
        if stem.ends_with(suffix) {
            return path.to_path_buf();
        }

        let ext = path.extension().and_then(|ext| ext.to_str()).unwrap_or("");
        let mut file_name = format!("{stem}{suffix}");
        if !ext.is_empty() {
            file_name.push('.');
            file_name.push_str(ext);
        }
        path.with_file_name(file_name)
    }

    pub fn load_triposg_vae_from_safetensors<B: Backend>(
        config: &TripoSGVaeConfig,
        device: &B::Device,
        path: impl AsRef<Path>,
    ) -> Result<TripoSGVae<B>, Box<dyn std::error::Error>> {
        let mut model = TripoSGVae::new(device, config.clone());
        let mut store = build_store(path.as_ref())?;
        model
            .load_from(&mut store)
            .map_err(|err| format!("failed to apply TripoSG VAE weights: {err}"))?;
        Ok(model)
    }

    pub fn import_triposg_vae_burnpack<B: Backend>(
        config: &TripoSGVaeConfig,
        device: &B::Device,
        path: impl AsRef<Path>,
        use_f16: bool,
    ) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let path = path.as_ref();
        let burnpack_path = burnpack_path(path, use_f16);
        let model = load_triposg_vae_from_safetensors::<B>(config, device, path)?;
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
        model: &TripoSGVae<B>,
        path: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut store = BurnpackStore::from_file(path).overwrite(true);
        model
            .save_into(&mut store)
            .map_err(|err| format!("failed to save TripoSG VAE burnpack: {err}"))?;
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
            (
                r"^(encoder\.blocks\.\d+\.attn1\.to_out)\.0\.(weight|bias)$",
                "$1.$2",
            ),
            (
                r"^(encoder\.blocks\.\d+\.attn2\.to_out)\.0\.(weight|bias)$",
                "$1.$2",
            ),
            (
                r"^(decoder\.blocks\.\d+\.attn1\.to_out)\.0\.(weight|bias)$",
                "$1.$2",
            ),
            (
                r"^(decoder\.blocks\.\d+\.attn2\.to_out)\.0\.(weight|bias)$",
                "$1.$2",
            ),
            (
                r"^(encoder\.blocks\.\d+\.ff)\.net\.0\.proj\.(weight|bias)$",
                "$1.proj.$2",
            ),
            (
                r"^(encoder\.blocks\.\d+\.ff)\.net\.2\.(weight|bias)$",
                "$1.out.$2",
            ),
            (
                r"^(decoder\.blocks\.\d+\.ff)\.net\.0\.proj\.(weight|bias)$",
                "$1.proj.$2",
            ),
            (
                r"^(decoder\.blocks\.\d+\.ff)\.net\.2\.(weight|bias)$",
                "$1.out.$2",
            ),
            (r"^(encoder\.blocks\.\d+\.norm\d)\.weight$", "$1.gamma"),
            (r"^(encoder\.blocks\.\d+\.norm\d)\.bias$", "$1.beta"),
            (r"^(decoder\.blocks\.\d+\.norm\d)\.weight$", "$1.gamma"),
            (r"^(decoder\.blocks\.\d+\.norm\d)\.bias$", "$1.beta"),
            (
                r"^(encoder\.blocks\.\d+\.attn2\.norm_cross)\.weight$",
                "$1.gamma",
            ),
            (
                r"^(encoder\.blocks\.\d+\.attn2\.norm_cross)\.bias$",
                "$1.beta",
            ),
            (
                r"^(decoder\.blocks\.\d+\.attn2\.norm_cross)\.weight$",
                "$1.gamma",
            ),
            (
                r"^(decoder\.blocks\.\d+\.attn2\.norm_cross)\.bias$",
                "$1.beta",
            ),
            (r"^(encoder\.norm_out)\.weight$", "$1.gamma"),
            (r"^(encoder\.norm_out)\.bias$", "$1.beta"),
            (r"^(decoder\.norm_out)\.weight$", "$1.gamma"),
            (r"^(decoder\.norm_out)\.bias$", "$1.beta"),
        ]
    }
}

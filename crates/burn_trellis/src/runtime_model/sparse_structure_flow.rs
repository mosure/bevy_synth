use std::path::{Path, PathBuf};

use burn::module::{Ignored, Module, Param};
use burn::nn;
use burn::prelude::Backend;
use burn::tensor::activation::{sigmoid, softmax};
use burn::tensor::{Int, Tensor};
use burn_store::{
    BurnpackStore, KeyRemapper, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore,
};
use serde::Deserialize;

use crate::sampler::{
    FlowEulerSampleConfig, FlowEulerSampleTrace, mid_snapshot_step, timestep_pairs,
};

const F16_SUFFIX: &str = "_f16";
const MAX_PERIOD: f32 = 10_000.0;
const LAYER_NORM_EPS: f32 = 1.0e-6;
const RMS_NORM_EPS: f32 = 1.0e-12;

type CpuRuntimeBackend = burn::backend::NdArray<f32>;
#[cfg(feature = "runtime-model-wgpu")]
type WgpuRuntimeBackend = burn_wgpu::Wgpu<f32, i32, u32>;

#[derive(Clone, Debug, Deserialize)]
pub struct SparseStructureFlowConfig {
    #[serde(default = "default_resolution")]
    pub resolution: usize,
    #[serde(default = "default_in_channels")]
    pub in_channels: usize,
    #[serde(default = "default_out_channels")]
    pub out_channels: usize,
    #[serde(default = "default_model_channels")]
    pub model_channels: usize,
    #[serde(default = "default_cond_channels")]
    pub cond_channels: usize,
    #[serde(default = "default_num_blocks")]
    pub num_blocks: usize,
    #[serde(default)]
    pub num_heads: Option<usize>,
    #[serde(default = "default_num_head_channels")]
    pub num_head_channels: usize,
    #[serde(default = "default_mlp_ratio")]
    pub mlp_ratio: f32,
    #[serde(default = "default_pe_mode")]
    pub pe_mode: String,
    #[serde(default = "default_rope_freq")]
    pub rope_freq: [f32; 2],
    #[serde(default = "default_share_mod")]
    pub share_mod: bool,
    #[serde(default = "default_qk_rms_norm")]
    pub qk_rms_norm: bool,
    #[serde(default = "default_qk_rms_norm_cross")]
    pub qk_rms_norm_cross: bool,
    #[serde(default = "default_frequency_embedding_size")]
    pub frequency_embedding_size: usize,
}

#[derive(Debug, Deserialize)]
struct SparseStructureFlowConfigFile {
    #[serde(default)]
    args: SparseStructureFlowConfig,
}

impl Default for SparseStructureFlowConfig {
    fn default() -> Self {
        Self {
            resolution: default_resolution(),
            in_channels: default_in_channels(),
            out_channels: default_out_channels(),
            model_channels: default_model_channels(),
            cond_channels: default_cond_channels(),
            num_blocks: default_num_blocks(),
            num_heads: None,
            num_head_channels: default_num_head_channels(),
            mlp_ratio: default_mlp_ratio(),
            pe_mode: default_pe_mode(),
            rope_freq: default_rope_freq(),
            share_mod: default_share_mod(),
            qk_rms_norm: default_qk_rms_norm(),
            qk_rms_norm_cross: default_qk_rms_norm_cross(),
            frequency_embedding_size: default_frequency_embedding_size(),
        }
    }
}

impl SparseStructureFlowConfig {
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, String> {
        let file: SparseStructureFlowConfigFile = serde_json::from_slice(bytes)
            .map_err(|err| format!("failed to parse sparse structure flow config json: {err}"))?;
        Ok(file.args)
    }

    pub fn num_heads(&self) -> usize {
        self.num_heads
            .unwrap_or(self.model_channels / self.num_head_channels.max(1))
            .max(1)
    }
}

fn default_resolution() -> usize {
    16
}

fn default_in_channels() -> usize {
    8
}

fn default_out_channels() -> usize {
    8
}

fn default_model_channels() -> usize {
    1536
}

fn default_cond_channels() -> usize {
    1024
}

fn default_num_blocks() -> usize {
    30
}

fn default_num_head_channels() -> usize {
    64
}

fn default_mlp_ratio() -> f32 {
    5.3334
}

fn default_pe_mode() -> String {
    "rope".to_string()
}

fn default_rope_freq() -> [f32; 2] {
    [1.0, 10_000.0]
}

fn default_share_mod() -> bool {
    true
}

fn default_qk_rms_norm() -> bool {
    true
}

fn default_qk_rms_norm_cross() -> bool {
    true
}

fn default_frequency_embedding_size() -> usize {
    256
}

#[derive(Module, Debug)]
pub struct TimestepEmbedder<B: Backend> {
    pub mlp_0: nn::Linear<B>,
    pub mlp_2: nn::Linear<B>,
}

impl<B: Backend> TimestepEmbedder<B> {
    pub fn new(device: &B::Device, frequency_embedding_size: usize, hidden_size: usize) -> Self {
        let mlp_0 = nn::LinearConfig::new(frequency_embedding_size, hidden_size)
            .with_bias(true)
            .init(device);
        let mlp_2 = nn::LinearConfig::new(hidden_size, hidden_size)
            .with_bias(true)
            .init(device);
        Self { mlp_0, mlp_2 }
    }

    pub fn forward(&self, t: Tensor<B, 1>, frequency_embedding_size: usize) -> Tensor<B, 2> {
        let emb = timestep_embedding(t, frequency_embedding_size);
        self.mlp_2.forward(silu(self.mlp_0.forward(emb)))
    }
}

#[derive(Module, Debug)]
pub struct MultiHeadRmsNorm<B: Backend> {
    pub gamma: Param<Tensor<B, 2>>,
    scale: f32,
}

impl<B: Backend> MultiHeadRmsNorm<B> {
    pub fn new(device: &B::Device, num_heads: usize, head_dim: usize) -> Self {
        let gamma = nn::Initializer::Ones.init([num_heads, head_dim], device);
        Self {
            gamma,
            scale: (head_dim as f32).sqrt(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [_, _, heads, head_dim] = x.dims();
        let rms = x
            .clone()
            .powf_scalar(2.0)
            .mean_dim(3)
            .add_scalar(RMS_NORM_EPS)
            .sqrt();
        let x = x.mul(rms.recip()).mul_scalar(self.scale);
        let gamma = self.gamma.val().reshape([1, 1, heads, head_dim]);
        x.mul(gamma)
    }
}

#[derive(Module, Debug)]
pub struct FeedForwardNet<B: Backend> {
    pub mlp_0: nn::Linear<B>,
    pub mlp_2: nn::Linear<B>,
}

impl<B: Backend> FeedForwardNet<B> {
    pub fn new(device: &B::Device, channels: usize, mlp_ratio: f32) -> Self {
        let hidden = ((channels as f32) * mlp_ratio).round().max(1.0) as usize;
        let mlp_0 = nn::LinearConfig::new(channels, hidden)
            .with_bias(true)
            .init(device);
        let mlp_2 = nn::LinearConfig::new(hidden, channels)
            .with_bias(true)
            .init(device);
        Self { mlp_0, mlp_2 }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.mlp_2.forward(gelu(self.mlp_0.forward(x)))
    }
}

#[derive(Module, Debug)]
pub struct SelfAttention<B: Backend> {
    pub to_qkv: nn::Linear<B>,
    pub to_out: nn::Linear<B>,
    pub q_rms_norm: Option<MultiHeadRmsNorm<B>>,
    pub k_rms_norm: Option<MultiHeadRmsNorm<B>>,
    num_heads: usize,
    head_dim: usize,
    use_rope: bool,
    rope_freq: [f32; 2],
}

#[derive(Module, Debug)]
pub struct CrossAttention<B: Backend> {
    pub to_q: nn::Linear<B>,
    pub to_kv: nn::Linear<B>,
    pub to_out: nn::Linear<B>,
    pub q_rms_norm: Option<MultiHeadRmsNorm<B>>,
    pub k_rms_norm: Option<MultiHeadRmsNorm<B>>,
    num_heads: usize,
    head_dim: usize,
}

#[derive(Module, Debug)]
pub struct ModulatedTransformerCrossBlock<B: Backend> {
    pub self_attn: SelfAttention<B>,
    pub cross_attn: CrossAttention<B>,
    pub mlp: FeedForwardNet<B>,
    pub norm2: nn::LayerNorm<B>,
    pub modulation: Param<Tensor<B, 1>>,
}

#[derive(Module, Debug)]
pub struct SparseStructureFlowModel<B: Backend> {
    pub t_embedder: TimestepEmbedder<B>,
    pub ada_ln_modulation: nn::Linear<B>,
    pub input_layer: nn::Linear<B>,
    pub blocks: Vec<ModulatedTransformerCrossBlock<B>>,
    pub out_layer: nn::Linear<B>,
    config: Ignored<SparseStructureFlowConfig>,
}

#[derive(Debug)]
pub(crate) struct SparseStructureFlowRuntimeImpl<B: Backend> {
    config: SparseStructureFlowConfig,
    model: SparseStructureFlowModel<B>,
    device: B::Device,
}

#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub(crate) enum SparseStructureFlowRuntime {
    Cpu(SparseStructureFlowRuntimeImpl<CpuRuntimeBackend>),
    #[cfg(feature = "runtime-model-wgpu")]
    Wgpu(SparseStructureFlowRuntimeImpl<WgpuRuntimeBackend>),
}

#[derive(Debug)]
pub(crate) enum SparseFlowCondition {
    Cpu(Tensor<CpuRuntimeBackend, 3>),
    #[cfg(feature = "runtime-model-wgpu")]
    Wgpu(Tensor<WgpuRuntimeBackend, 3>),
}

impl<B: Backend> SelfAttention<B> {
    pub fn new(
        device: &B::Device,
        channels: usize,
        num_heads: usize,
        use_rope: bool,
        rope_freq: [f32; 2],
        qk_rms_norm: bool,
    ) -> Self {
        let head_dim = channels / num_heads.max(1);
        let to_qkv = nn::LinearConfig::new(channels, channels * 3)
            .with_bias(true)
            .init(device);
        let to_out = nn::LinearConfig::new(channels, channels)
            .with_bias(true)
            .init(device);
        let q_rms_norm = if qk_rms_norm {
            Some(MultiHeadRmsNorm::new(device, num_heads, head_dim))
        } else {
            None
        };
        let k_rms_norm = if qk_rms_norm {
            Some(MultiHeadRmsNorm::new(device, num_heads, head_dim))
        } else {
            None
        };
        Self {
            to_qkv,
            to_out,
            q_rms_norm,
            k_rms_norm,
            num_heads,
            head_dim,
            use_rope,
            rope_freq,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, resolution: usize) -> Tensor<B, 3> {
        let [batch, tokens, channels] = x.dims();
        let qkv = self
            .to_qkv
            .forward(x)
            .reshape([batch, tokens, 3, self.num_heads, self.head_dim]);
        let q = qkv
            .clone()
            .slice([
                0..batch,
                0..tokens,
                0..1,
                0..self.num_heads,
                0..self.head_dim,
            ])
            .reshape([batch, tokens, self.num_heads, self.head_dim]);
        let k = qkv
            .clone()
            .slice([
                0..batch,
                0..tokens,
                1..2,
                0..self.num_heads,
                0..self.head_dim,
            ])
            .reshape([batch, tokens, self.num_heads, self.head_dim]);
        let v = qkv
            .slice([
                0..batch,
                0..tokens,
                2..3,
                0..self.num_heads,
                0..self.head_dim,
            ])
            .reshape([batch, tokens, self.num_heads, self.head_dim]);

        let q = if let Some(norm) = self.q_rms_norm.as_ref() {
            norm.forward(q)
        } else {
            q
        };
        let k = if let Some(norm) = self.k_rms_norm.as_ref() {
            norm.forward(k)
        } else {
            k
        };
        let (q, k) = if self.use_rope {
            apply_rope(q, k, resolution, self.head_dim, self.rope_freq)
        } else {
            (q, k)
        };

        let out = scaled_dot_product_attention(q, k, v, self.head_dim);
        self.to_out.forward(out.reshape([batch, tokens, channels]))
    }
}

impl<B: Backend> CrossAttention<B> {
    pub fn new(
        device: &B::Device,
        channels: usize,
        ctx_channels: usize,
        num_heads: usize,
        qk_rms_norm: bool,
    ) -> Self {
        let head_dim = channels / num_heads.max(1);
        let to_q = nn::LinearConfig::new(channels, channels)
            .with_bias(true)
            .init(device);
        let to_kv = nn::LinearConfig::new(ctx_channels, channels * 2)
            .with_bias(true)
            .init(device);
        let to_out = nn::LinearConfig::new(channels, channels)
            .with_bias(true)
            .init(device);
        let q_rms_norm = if qk_rms_norm {
            Some(MultiHeadRmsNorm::new(device, num_heads, head_dim))
        } else {
            None
        };
        let k_rms_norm = if qk_rms_norm {
            Some(MultiHeadRmsNorm::new(device, num_heads, head_dim))
        } else {
            None
        };
        Self {
            to_q,
            to_kv,
            to_out,
            q_rms_norm,
            k_rms_norm,
            num_heads,
            head_dim,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, context: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, tokens, channels] = x.dims();
        let ctx_tokens = context.dims()[1];
        let q = self
            .to_q
            .forward(x)
            .reshape([batch, tokens, self.num_heads, self.head_dim]);
        let kv = self.to_kv.forward(context).reshape([
            batch,
            ctx_tokens,
            2,
            self.num_heads,
            self.head_dim,
        ]);
        let k = kv
            .clone()
            .slice([
                0..batch,
                0..ctx_tokens,
                0..1,
                0..self.num_heads,
                0..self.head_dim,
            ])
            .reshape([batch, ctx_tokens, self.num_heads, self.head_dim]);
        let v = kv
            .slice([
                0..batch,
                0..ctx_tokens,
                1..2,
                0..self.num_heads,
                0..self.head_dim,
            ])
            .reshape([batch, ctx_tokens, self.num_heads, self.head_dim]);

        let q = if let Some(norm) = self.q_rms_norm.as_ref() {
            norm.forward(q)
        } else {
            q
        };
        let k = if let Some(norm) = self.k_rms_norm.as_ref() {
            norm.forward(k)
        } else {
            k
        };

        let out = scaled_dot_product_attention(q, k, v, self.head_dim);
        self.to_out.forward(out.reshape([batch, tokens, channels]))
    }
}

impl<B: Backend> ModulatedTransformerCrossBlock<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &B::Device,
        channels: usize,
        ctx_channels: usize,
        num_heads: usize,
        mlp_ratio: f32,
        use_rope: bool,
        rope_freq: [f32; 2],
        qk_rms_norm: bool,
        qk_rms_norm_cross: bool,
    ) -> Self {
        let self_attn = SelfAttention::new(
            device,
            channels,
            num_heads,
            use_rope,
            rope_freq,
            qk_rms_norm,
        );
        let cross_attn =
            CrossAttention::new(device, channels, ctx_channels, num_heads, qk_rms_norm_cross);
        let mlp = FeedForwardNet::new(device, channels, mlp_ratio);
        let norm2 = nn::LayerNormConfig::new(channels)
            .with_epsilon(LAYER_NORM_EPS as f64)
            .init(device);
        let modulation = nn::Initializer::Zeros.init([channels * 6], device);
        Self {
            self_attn,
            cross_attn,
            mlp,
            norm2,
            modulation,
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        mod_signal: Tensor<B, 2>,
        context: Tensor<B, 3>,
        resolution: usize,
    ) -> Tensor<B, 3> {
        let [batch, _, channels] = x.dims();
        let mod_bias = self.modulation.val().reshape([1, channels * 6]);
        let mod_signal = mod_signal.add(mod_bias);
        let shift_msa = mod_signal
            .clone()
            .slice([0..batch, 0..channels])
            .reshape([batch, 1, channels]);
        let scale_msa = mod_signal
            .clone()
            .slice([0..batch, channels..(channels * 2)])
            .reshape([batch, 1, channels]);
        let gate_msa = mod_signal
            .clone()
            .slice([0..batch, (channels * 2)..(channels * 3)])
            .reshape([batch, 1, channels]);
        let shift_mlp = mod_signal
            .clone()
            .slice([0..batch, (channels * 3)..(channels * 4)])
            .reshape([batch, 1, channels]);
        let scale_mlp = mod_signal
            .clone()
            .slice([0..batch, (channels * 4)..(channels * 5)])
            .reshape([batch, 1, channels]);
        let gate_mlp = mod_signal
            .slice([0..batch, (channels * 5)..(channels * 6)])
            .reshape([batch, 1, channels]);

        let h = layer_norm_no_affine(x.clone(), LAYER_NORM_EPS)
            .mul(scale_msa.add_scalar(1.0))
            .add(shift_msa);
        let h = self.self_attn.forward(h, resolution).mul(gate_msa);
        let x = x.add(h);

        let h = self.norm2.forward(x.clone());
        let x = x.add(self.cross_attn.forward(h, context));

        let h = layer_norm_no_affine(x.clone(), LAYER_NORM_EPS)
            .mul(scale_mlp.add_scalar(1.0))
            .add(shift_mlp);
        let h = self.mlp.forward(h).mul(gate_mlp);
        x.add(h)
    }
}

impl<B: Backend> SparseStructureFlowModel<B> {
    pub fn new(device: &B::Device, config: SparseStructureFlowConfig) -> Self {
        let num_heads = config.num_heads();
        let t_embedder = TimestepEmbedder::new(
            device,
            config.frequency_embedding_size,
            config.model_channels,
        );
        let ada_ln_modulation =
            nn::LinearConfig::new(config.model_channels, config.model_channels * 6)
                .with_bias(true)
                .init(device);
        let input_layer = nn::LinearConfig::new(config.in_channels, config.model_channels)
            .with_bias(true)
            .init(device);
        let mut blocks = Vec::with_capacity(config.num_blocks);
        for _ in 0..config.num_blocks {
            blocks.push(ModulatedTransformerCrossBlock::new(
                device,
                config.model_channels,
                config.cond_channels,
                num_heads,
                config.mlp_ratio,
                config.pe_mode == "rope",
                config.rope_freq,
                config.qk_rms_norm,
                config.qk_rms_norm_cross,
            ));
        }
        let out_layer = nn::LinearConfig::new(config.model_channels, config.out_channels)
            .with_bias(true)
            .init(device);
        Self {
            t_embedder,
            ada_ln_modulation,
            input_layer,
            blocks,
            out_layer,
            config: Ignored(config),
        }
    }

    pub fn config(&self) -> &SparseStructureFlowConfig {
        &self.config
    }

    pub fn forward(&self, x: Tensor<B, 5>, t: Tensor<B, 1>, cond: Tensor<B, 3>) -> Tensor<B, 5> {
        let [batch, channels, rx, ry, rz] = x.dims();
        assert_eq!(
            channels, self.config.in_channels,
            "sparse flow input channel mismatch"
        );
        assert_eq!(
            rx, self.config.resolution,
            "sparse flow input resolution mismatch"
        );
        assert_eq!(
            ry, self.config.resolution,
            "sparse flow input resolution mismatch"
        );
        assert_eq!(
            rz, self.config.resolution,
            "sparse flow input resolution mismatch"
        );
        assert_eq!(
            cond.dims()[2],
            self.config.cond_channels,
            "sparse flow cond channel mismatch"
        );

        let tokens = self.config.resolution * self.config.resolution * self.config.resolution;
        let mut h = x.reshape([batch, channels, tokens]).swap_dims(1, 2);
        h = self.input_layer.forward(h);

        let t_emb = self
            .t_embedder
            .forward(t, self.config.frequency_embedding_size);
        let mod_signal = self.ada_ln_modulation.forward(silu(t_emb));

        for block in &self.blocks {
            h = block.forward(h, mod_signal.clone(), cond.clone(), self.config.resolution);
        }

        let h = layer_norm_no_affine(h, LAYER_NORM_EPS);
        let h = self.out_layer.forward(h);
        h.swap_dims(1, 2).reshape([
            batch,
            self.config.out_channels,
            self.config.resolution,
            self.config.resolution,
            self.config.resolution,
        ])
    }
}

impl<B> SparseStructureFlowRuntimeImpl<B>
where
    B: Backend,
    B::Device: Default,
{
    fn load_from_stem(
        weights_root: &Path,
        image_large_root: Option<&Path>,
        model_stem: &str,
        resolution_override: Option<usize>,
    ) -> Result<Self, String> {
        let config_path =
            resolve_model_source_path(model_stem, "json", weights_root, image_large_root);
        let config_bytes = std::fs::read(&config_path).map_err(|err| {
            format!(
                "failed to read sparse structure flow config '{}': {err}",
                config_path.display()
            )
        })?;
        let mut config = SparseStructureFlowConfig::from_json_bytes(&config_bytes)?;
        if let Some(override_resolution) = resolution_override {
            if override_resolution == 0 {
                return Err("sparse flow resolution override must be > 0".to_string());
            }
            config.resolution = override_resolution;
        }
        if !config.share_mod {
            return Err(format!(
                "unsupported sparse structure flow config '{}': share_mod=false is not yet supported",
                config_path.display()
            ));
        }

        let weight_candidates =
            resolve_model_weight_candidates(model_stem, weights_root, image_large_root);
        if weight_candidates.is_empty() {
            return Err(format!(
                "unable to resolve sparse structure flow weights for stem '{model_stem}'"
            ));
        }
        let device = B::Device::default();
        let mut last_error = None;
        for weights_path in weight_candidates {
            let mut model = SparseStructureFlowModel::<B>::new(&device, config.clone());
            match load_sparse_model_weights(&mut model, &weights_path) {
                Ok(()) => {
                    return Ok(Self {
                        config,
                        model,
                        device,
                    });
                }
                Err(err) => {
                    last_error = Some(format!("{} ({err})", weights_path.display()));
                }
            }
        }

        Err(format!(
            "failed to load sparse structure flow weights for stem '{model_stem}': {}",
            last_error.unwrap_or_else(|| "unknown error".to_string())
        ))
    }
}

impl<B: Backend> SparseStructureFlowRuntimeImpl<B> {
    fn config(&self) -> &SparseStructureFlowConfig {
        &self.config
    }

    fn prepare_condition(&self, cond: &[f32], cond_tokens: usize) -> Result<Tensor<B, 3>, String> {
        let cond_elements = cond_tokens * self.config.cond_channels;
        if cond.len() != cond_elements {
            return Err(format!(
                "sparse flow cond length mismatch: expected {}, got {}",
                cond_elements,
                cond.len()
            ));
        }
        Ok(Tensor::<B, 1>::from_floats(cond, &self.device).reshape([
            1,
            cond_tokens,
            self.config.cond_channels,
        ]))
    }

    #[allow(dead_code)]
    fn predict_velocity_with_condition(
        &self,
        x_t: &[f32],
        timestep: f32,
        cond: Tensor<B, 3>,
        concat_cond: Option<&[f32]>,
    ) -> Result<Vec<f32>, String> {
        let voxel = self.config.resolution * self.config.resolution * self.config.resolution;
        if !x_t.len().is_multiple_of(voxel) {
            return Err(format!(
                "sparse flow sample length mismatch: sample len {} is not divisible by voxel count {}",
                x_t.len(),
                voxel
            ));
        }
        let state_channels = x_t.len() / voxel;
        let concat_channels = if let Some(cond) = concat_cond {
            if cond.len() % voxel != 0 {
                return Err(format!(
                    "sparse flow concat cond length mismatch: len {} is not divisible by voxel count {}",
                    cond.len(),
                    voxel
                ));
            }
            cond.len() / voxel
        } else {
            0usize
        };
        if state_channels + concat_channels != self.config.in_channels {
            return Err(format!(
                "sparse flow channel mismatch: state={} concat={} expected_in={}",
                state_channels, concat_channels, self.config.in_channels
            ));
        }
        if state_channels != self.config.out_channels {
            return Err(format!(
                "sparse flow state/output mismatch: state={} expected_out={}",
                state_channels, self.config.out_channels
            ));
        }

        let input = if let Some(concat) = concat_cond {
            let mut merged = Vec::with_capacity((state_channels + concat_channels) * voxel);
            merged.extend_from_slice(x_t);
            merged.extend_from_slice(concat);
            merged
        } else {
            x_t.to_vec()
        };

        let sample = Tensor::<B, 1>::from_floats(input.as_slice(), &self.device).reshape([
            1,
            self.config.in_channels,
            self.config.resolution,
            self.config.resolution,
            self.config.resolution,
        ]);
        let t = Tensor::<B, 1>::from_floats([timestep * 1000.0], &self.device);
        let out = self.model.forward(sample, t, cond);
        out.into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .map_err(|err| format!("failed to read sparse flow output: {err:?}"))
    }

    fn sample_with_trace(
        &self,
        noise: &[f32],
        sample_cfg: FlowEulerSampleConfig,
        sigma_min: f32,
        cond: Tensor<B, 3>,
        neg_cond: Tensor<B, 3>,
        concat_cond: Option<&[f32]>,
    ) -> Result<FlowEulerSampleTrace, String> {
        let voxel = self.config.resolution * self.config.resolution * self.config.resolution;
        if voxel == 0 {
            return Err("sparse flow resolution produced zero voxels".to_string());
        }
        if !noise.len().is_multiple_of(voxel) {
            return Err(format!(
                "sparse flow sample length mismatch: sample len {} is not divisible by voxel count {}",
                noise.len(),
                voxel
            ));
        }
        let state_channels = noise.len() / voxel;
        if state_channels != self.config.out_channels {
            return Err(format!(
                "sparse flow state/output mismatch: state={} expected_out={}",
                state_channels, self.config.out_channels
            ));
        }
        let concat_channels = concat_cond.map_or(0usize, |values| values.len() / voxel);
        if state_channels + concat_channels != self.config.in_channels {
            return Err(format!(
                "sparse flow channel mismatch: state={} concat={} expected_in={}",
                state_channels, concat_channels, self.config.in_channels
            ));
        }
        if let Some(values) = concat_cond
            && !values.len().is_multiple_of(voxel)
        {
            return Err(format!(
                "sparse flow concat cond length mismatch: len {} is not divisible by voxel count {}",
                values.len(),
                voxel
            ));
        }

        let mut x_t = Tensor::<B, 1>::from_floats(noise, &self.device).reshape([
            1,
            state_channels,
            self.config.resolution,
            self.config.resolution,
            self.config.resolution,
        ]);
        let concat_tensor = concat_cond.map(|values| {
            Tensor::<B, 1>::from_floats(values, &self.device).reshape([
                1,
                concat_channels,
                self.config.resolution,
                self.config.resolution,
                self.config.resolution,
            ])
        });

        let mut step_0_x_t: Option<Vec<f32>> = None;
        let mut step_mid_x_t: Option<Vec<f32>> = None;
        let mut step_last_x_t: Option<Vec<f32>> = None;
        let mid_step = mid_snapshot_step(sample_cfg.steps);
        let t_pairs = timestep_pairs(sample_cfg.steps, sample_cfg.rescale_t);
        for (step_idx, (t, t_prev)) in t_pairs.into_iter().enumerate() {
            let pred = self.predict_with_cfg_tensor(
                x_t.clone(),
                t,
                sample_cfg,
                sigma_min,
                cond.clone(),
                neg_cond.clone(),
                concat_tensor.clone(),
            )?;
            let dt = t - t_prev;
            x_t = x_t.sub(pred.mul_scalar(dt));
            if step_idx == 0 {
                step_0_x_t = Some(tensor_to_vec(x_t.clone())?);
            }
            if step_idx == mid_step {
                step_mid_x_t = Some(tensor_to_vec(x_t.clone())?);
            }
            if step_idx + 1 == sample_cfg.steps {
                step_last_x_t = Some(tensor_to_vec(x_t.clone())?);
            }
        }

        let samples = tensor_to_vec(x_t.clone())?;
        Ok(FlowEulerSampleTrace {
            steps: sample_cfg.steps,
            step_0_x_t: step_0_x_t.unwrap_or_else(|| samples.clone()),
            step_mid_x_t: step_mid_x_t.unwrap_or_else(|| samples.clone()),
            step_last_x_t: step_last_x_t.unwrap_or_else(|| samples.clone()),
            samples,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn predict_with_cfg_tensor(
        &self,
        x_t: Tensor<B, 5>,
        timestep: f32,
        config: FlowEulerSampleConfig,
        sigma_min: f32,
        cond: Tensor<B, 3>,
        neg_cond: Tensor<B, 3>,
        concat_cond: Option<Tensor<B, 5>>,
    ) -> Result<Tensor<B, 5>, String> {
        let in_guidance_interval =
            config.guidance_interval[0] <= timestep && timestep <= config.guidance_interval[1];
        if !in_guidance_interval {
            return self.predict_velocity_tensor(x_t, timestep, cond, concat_cond);
        }

        let w = config.guidance_strength;
        if (w - 1.0).abs() < f32::EPSILON {
            return self.predict_velocity_tensor(x_t, timestep, cond, concat_cond);
        }
        if w.abs() < f32::EPSILON {
            return self.predict_velocity_tensor(x_t, timestep, neg_cond, concat_cond);
        }

        let pos =
            self.predict_velocity_tensor(x_t.clone(), timestep, cond.clone(), concat_cond.clone())?;
        let neg = self.predict_velocity_tensor(x_t.clone(), timestep, neg_cond, concat_cond)?;
        let mut pred = pos.clone().mul_scalar(w).add(neg.mul_scalar(1.0 - w));
        if config.guidance_rescale <= 0.0 {
            return Ok(pred);
        }

        let x0_pos = pred_to_xstart_tensor(x_t.clone(), timestep, pos, sigma_min);
        let x0_cfg = pred_to_xstart_tensor(x_t.clone(), timestep, pred.clone(), sigma_min);
        let std_pos = tensor_std_scalar(x0_pos.clone())?;
        let std_cfg = tensor_std_scalar(x0_cfg.clone())?.max(1.0e-12);
        let scale = std_pos / std_cfg;
        let x0 = x0_cfg
            .clone()
            .mul_scalar(scale)
            .mul_scalar(config.guidance_rescale)
            .add(x0_cfg.mul_scalar(1.0 - config.guidance_rescale));
        pred = xstart_to_pred_tensor(x_t, timestep, x0, sigma_min);
        Ok(pred)
    }

    fn predict_velocity_tensor(
        &self,
        x_t: Tensor<B, 5>,
        timestep: f32,
        cond: Tensor<B, 3>,
        concat_cond: Option<Tensor<B, 5>>,
    ) -> Result<Tensor<B, 5>, String> {
        let [_, state_channels, rx, ry, rz] = x_t.dims();
        let voxel = rx * ry * rz;
        if rx != self.config.resolution
            || ry != self.config.resolution
            || rz != self.config.resolution
        {
            return Err(format!(
                "sparse flow tensor resolution mismatch: got=({rx},{ry},{rz}) expected={}",
                self.config.resolution
            ));
        }
        let concat_channels = concat_cond
            .as_ref()
            .map(|tensor| {
                let [_, channels, cx, cy, cz] = tensor.dims();
                if channels == 0 {
                    return Err("concat cond tensor has zero channels".to_string());
                }
                if cx != rx || cy != ry || cz != rz {
                    return Err(format!(
                        "concat cond tensor resolution mismatch: got=({cx},{cy},{cz}) expected=({rx},{ry},{rz})"
                    ));
                }
                Ok(channels)
            })
            .transpose()?
            .unwrap_or(0usize);
        if state_channels + concat_channels != self.config.in_channels {
            return Err(format!(
                "sparse flow channel mismatch: state={} concat={} expected_in={}",
                state_channels, concat_channels, self.config.in_channels
            ));
        }
        if state_channels != self.config.out_channels {
            return Err(format!(
                "sparse flow state/output mismatch: state={} expected_out={}",
                state_channels, self.config.out_channels
            ));
        }
        if voxel == 0 {
            return Err("sparse flow tensor voxel count is zero".to_string());
        }

        let sample = if let Some(concat) = concat_cond {
            Tensor::cat(vec![x_t, concat], 1)
        } else {
            x_t
        };
        let t = Tensor::<B, 1>::from_floats([timestep * 1000.0], &self.device);
        Ok(self.model.forward(sample, t, cond))
    }
}

impl SparseStructureFlowRuntime {
    pub fn load_from_stem(
        weights_root: &Path,
        image_large_root: Option<&Path>,
        model_stem: &str,
        _prefer_wgpu: bool,
        resolution_override: Option<usize>,
    ) -> Result<Self, String> {
        #[cfg(feature = "runtime-model-wgpu")]
        if _prefer_wgpu {
            match SparseStructureFlowRuntimeImpl::<WgpuRuntimeBackend>::load_from_stem(
                weights_root,
                image_large_root,
                model_stem,
                resolution_override,
            ) {
                Ok(runtime) => return Ok(Self::Wgpu(runtime)),
                Err(err) => {
                    eprintln!(
                        "burn_trellis: failed to load sparse flow runtime on wgpu ({err}); falling back to cpu."
                    );
                }
            }
        }
        let runtime = SparseStructureFlowRuntimeImpl::<CpuRuntimeBackend>::load_from_stem(
            weights_root,
            image_large_root,
            model_stem,
            resolution_override,
        )?;
        Ok(Self::Cpu(runtime))
    }

    pub fn backend_name(&self) -> &'static str {
        match self {
            Self::Cpu(_) => "cpu",
            #[cfg(feature = "runtime-model-wgpu")]
            Self::Wgpu(_) => "wgpu",
        }
    }

    pub fn config(&self) -> &SparseStructureFlowConfig {
        match self {
            Self::Cpu(runtime) => runtime.config(),
            #[cfg(feature = "runtime-model-wgpu")]
            Self::Wgpu(runtime) => runtime.config(),
        }
    }

    pub fn prepare_condition(
        &self,
        cond: &[f32],
        cond_tokens: usize,
    ) -> Result<SparseFlowCondition, String> {
        match self {
            Self::Cpu(runtime) => runtime
                .prepare_condition(cond, cond_tokens)
                .map(SparseFlowCondition::Cpu),
            #[cfg(feature = "runtime-model-wgpu")]
            Self::Wgpu(runtime) => runtime
                .prepare_condition(cond, cond_tokens)
                .map(SparseFlowCondition::Wgpu),
        }
    }

    #[allow(dead_code)]
    pub fn predict_velocity_with_condition(
        &self,
        x_t: &[f32],
        timestep: f32,
        condition: &SparseFlowCondition,
        concat_cond: Option<&[f32]>,
    ) -> Result<Vec<f32>, String> {
        #[cfg(feature = "runtime-model-wgpu")]
        {
            match (self, condition) {
                (Self::Cpu(runtime), SparseFlowCondition::Cpu(cond)) => runtime
                    .predict_velocity_with_condition(x_t, timestep, cond.clone(), concat_cond),
                (Self::Wgpu(runtime), SparseFlowCondition::Wgpu(cond)) => runtime
                    .predict_velocity_with_condition(x_t, timestep, cond.clone(), concat_cond),
                _ => {
                    Err("sparse flow condition backend does not match runtime backend".to_string())
                }
            }
        }
        #[cfg(not(feature = "runtime-model-wgpu"))]
        {
            let Self::Cpu(runtime) = self;
            let SparseFlowCondition::Cpu(cond) = condition;
            runtime.predict_velocity_with_condition(x_t, timestep, cond.clone(), concat_cond)
        }
    }

    pub fn sample_with_trace(
        &self,
        noise: &[f32],
        sample_cfg: FlowEulerSampleConfig,
        sigma_min: f32,
        condition: &SparseFlowCondition,
        negative_condition: &SparseFlowCondition,
        concat_cond: Option<&[f32]>,
    ) -> Result<FlowEulerSampleTrace, String> {
        #[cfg(feature = "runtime-model-wgpu")]
        {
            match (self, condition, negative_condition) {
                (
                    Self::Cpu(runtime),
                    SparseFlowCondition::Cpu(cond),
                    SparseFlowCondition::Cpu(neg_cond),
                ) => runtime.sample_with_trace(
                    noise,
                    sample_cfg,
                    sigma_min,
                    cond.clone(),
                    neg_cond.clone(),
                    concat_cond,
                ),
                (
                    Self::Wgpu(runtime),
                    SparseFlowCondition::Wgpu(cond),
                    SparseFlowCondition::Wgpu(neg_cond),
                ) => runtime.sample_with_trace(
                    noise,
                    sample_cfg,
                    sigma_min,
                    cond.clone(),
                    neg_cond.clone(),
                    concat_cond,
                ),
                _ => {
                    Err("sparse flow condition backend does not match runtime backend".to_string())
                }
            }
        }
        #[cfg(not(feature = "runtime-model-wgpu"))]
        {
            let Self::Cpu(runtime) = self;
            let SparseFlowCondition::Cpu(cond) = condition;
            let SparseFlowCondition::Cpu(neg_cond) = negative_condition;
            runtime.sample_with_trace(
                noise,
                sample_cfg,
                sigma_min,
                cond.clone(),
                neg_cond.clone(),
                concat_cond,
            )
        }
    }
}

fn pred_to_xstart_tensor<B: Backend>(
    x_t: Tensor<B, 5>,
    timestep: f32,
    pred: Tensor<B, 5>,
    sigma_min: f32,
) -> Tensor<B, 5> {
    let factor = sigma_min + (1.0 - sigma_min) * timestep;
    let keep = 1.0 - sigma_min;
    x_t.mul_scalar(keep).sub(pred.mul_scalar(factor))
}

fn xstart_to_pred_tensor<B: Backend>(
    x_t: Tensor<B, 5>,
    timestep: f32,
    x0: Tensor<B, 5>,
    sigma_min: f32,
) -> Tensor<B, 5> {
    let factor = sigma_min + (1.0 - sigma_min) * timestep;
    let keep = 1.0 - sigma_min;
    x_t.mul_scalar(keep).sub(x0).div_scalar(factor)
}

fn tensor_std_scalar<B: Backend>(tensor: Tensor<B, 5>) -> Result<f32, String> {
    let [b, c, x, y, z] = tensor.dims();
    let numel = b
        .saturating_mul(c)
        .saturating_mul(x)
        .saturating_mul(y)
        .saturating_mul(z)
        .max(1);
    let flat = tensor.reshape([numel]);
    let mean = flat.clone().mean_dim(0);
    let var = flat
        .sub(mean)
        .powf_scalar(2.0)
        .mean_dim(0)
        .sqrt()
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|err| format!("failed to read sparse flow std tensor: {err:?}"))?;
    Ok(var.first().copied().unwrap_or(0.0))
}

fn tensor_to_vec<B: Backend>(tensor: Tensor<B, 5>) -> Result<Vec<f32>, String> {
    tensor
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|err| format!("failed to read sparse flow tensor: {err:?}"))
}

fn gelu<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    // tanh-approx GELU parity with TRELLIS modules:
    // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
    let c0 = 0.044_715_f32;
    let c1 = 0.797_884_6_f32; // sqrt(2/pi)
    let x3 = x.clone().powf_scalar(3.0).mul_scalar(c0);
    let t = x.clone().add(x3).mul_scalar(c1).tanh();
    x.mul_scalar(0.5).mul(t.add_scalar(1.0))
}

fn silu<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    x.clone().mul(sigmoid(x))
}

fn timestep_embedding<B: Backend>(timesteps: Tensor<B, 1>, dim: usize) -> Tensor<B, 2> {
    let [batch] = timesteps.dims();
    let half = (dim / 2).max(1);
    let device = timesteps.device();
    let freqs = Tensor::<B, 1, Int>::arange(0..(half as i64), &device)
        .float()
        .mul_scalar(-MAX_PERIOD.ln())
        .div_scalar(half as f32)
        .exp();
    let args = timesteps.unsqueeze_dim(1).mul(freqs.unsqueeze_dim(0));
    let mut emb = Tensor::cat(vec![args.clone().cos(), args.sin()], 1);
    if dim % 2 == 1 {
        emb = Tensor::cat(vec![emb, Tensor::<B, 2>::zeros([batch, 1], &device)], 1);
    }
    emb
}

fn layer_norm_no_affine<B: Backend>(x: Tensor<B, 3>, eps: f32) -> Tensor<B, 3> {
    let mean = x.clone().mean_dim(2);
    let centered = x.sub(mean);
    let var = centered.clone().powf_scalar(2.0).mean_dim(2);
    centered.mul(var.add_scalar(eps).sqrt().recip())
}

fn scaled_dot_product_attention<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    head_dim: usize,
) -> Tensor<B, 4> {
    let q = q.permute([0, 2, 1, 3]);
    let k = k.permute([0, 2, 1, 3]);
    let v = v.permute([0, 2, 1, 3]);
    let [batch, heads, tokens, _] = q.dims();
    let scale = 1.0 / (head_dim as f32).sqrt();
    let chunk = attention_query_chunk(tokens);

    // Chunk attention over the query axis to avoid allocating a full [T, T] matrix
    // for large dense SLAT resolutions.
    if chunk >= tokens {
        let attn = softmax(q.matmul(k.swap_dims(2, 3)).mul_scalar(scale), 3);
        return attn.matmul(v).permute([0, 2, 1, 3]);
    }

    let k_t = k.clone().swap_dims(2, 3);
    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < tokens {
        let end = (start + chunk).min(tokens);
        let q_chunk = q
            .clone()
            .slice([0..batch, 0..heads, start..end, 0..head_dim]);
        let attn = softmax(q_chunk.matmul(k_t.clone()).mul_scalar(scale), 3);
        chunks.push(attn.matmul(v.clone()));
        start = end;
    }

    Tensor::cat(chunks, 2).permute([0, 2, 1, 3])
}

fn attention_query_chunk(tokens: usize) -> usize {
    let env_chunk = std::env::var("TRELLIS2_ATTN_QUERY_CHUNK")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(8);
    env_chunk.min(tokens.max(1))
}

fn apply_rope<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    resolution: usize,
    head_dim: usize,
    rope_freq: [f32; 2],
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let [_, tokens, _, _] = q.dims();
    let pairs = head_dim / 2;
    if pairs == 0 {
        return (q, k);
    }
    let device = q.device();
    let (rope_cos, rope_sin) = rope_cos_sin(resolution, tokens, pairs, rope_freq);
    let cos = Tensor::<B, 1>::from_floats(rope_cos.as_slice(), &device)
        .reshape([tokens, pairs])
        .reshape([1, tokens, 1, pairs]);
    let sin = Tensor::<B, 1>::from_floats(rope_sin.as_slice(), &device)
        .reshape([tokens, pairs])
        .reshape([1, tokens, 1, pairs]);

    (
        rotate_pairs(q, cos.clone(), sin.clone()),
        rotate_pairs(k, cos, sin),
    )
}

fn rotate_pairs<B: Backend>(x: Tensor<B, 4>, cos: Tensor<B, 4>, sin: Tensor<B, 4>) -> Tensor<B, 4> {
    let [batch, tokens, heads, head_dim] = x.dims();
    let pairs = head_dim / 2;
    let x = x.reshape([batch, tokens, heads, pairs, 2]);
    let x_even = x
        .clone()
        .slice([0..batch, 0..tokens, 0..heads, 0..pairs, 0..1])
        .reshape([batch, tokens, heads, pairs]);
    let x_odd = x
        .slice([0..batch, 0..tokens, 0..heads, 0..pairs, 1..2])
        .reshape([batch, tokens, heads, pairs]);

    let rot_even = x_even
        .clone()
        .mul(cos.clone())
        .sub(x_odd.clone().mul(sin.clone()));
    let rot_odd = x_even.mul(sin).add(x_odd.mul(cos));
    let rot_even = rot_even.reshape([batch, tokens, heads, pairs, 1]);
    let rot_odd = rot_odd.reshape([batch, tokens, heads, pairs, 1]);
    Tensor::cat(vec![rot_even, rot_odd], 4).reshape([batch, tokens, heads, head_dim])
}

fn rope_cos_sin(
    resolution: usize,
    tokens: usize,
    pairs: usize,
    rope_freq: [f32; 2],
) -> (Vec<f32>, Vec<f32>) {
    let mut cos = vec![1.0f32; tokens * pairs];
    let mut sin = vec![0.0f32; tokens * pairs];
    let freq_dim = (pairs / 3).max(1);
    let mut freqs = Vec::with_capacity(freq_dim);
    for idx in 0..freq_dim {
        let exp = idx as f32 / freq_dim as f32;
        freqs.push(rope_freq[0] / rope_freq[1].powf(exp));
    }

    let mut token = 0usize;
    for x in 0..resolution {
        for y in 0..resolution {
            for z in 0..resolution {
                if token >= tokens {
                    break;
                }
                let coords = [x as f32, y as f32, z as f32];
                for (dim, coord) in coords.iter().enumerate() {
                    for (freq_idx, freq) in freqs.iter().enumerate() {
                        let pair = dim * freq_dim + freq_idx;
                        if pair >= pairs {
                            continue;
                        }
                        let phase = *coord * *freq;
                        let idx = token * pairs + pair;
                        cos[idx] = phase.cos();
                        sin[idx] = phase.sin();
                    }
                }
                token += 1;
            }
        }
    }
    (cos, sin)
}

fn load_sparse_model_weights<B: Backend>(
    model: &mut SparseStructureFlowModel<B>,
    path: &Path,
) -> Result<(), String> {
    if path
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("bpk"))
    {
        let mut store = BurnpackStore::from_file(path).validate(true);
        model.load_from(&mut store).map(|_| ()).map_err(|err| {
            format!(
                "failed to load sparse flow burnpack '{}': {err}",
                path.display()
            )
        })
    } else {
        let mut store = build_safetensor_store(path)?;
        model.load_from(&mut store).map(|_| ()).map_err(|err| {
            format!(
                "failed to load sparse flow safetensors '{}': {err}",
                path.display()
            )
        })
    }
}

fn build_safetensor_store(path: &Path) -> Result<SafetensorsStore, String> {
    let mut remapper = KeyRemapper::new();
    for &(from, to) in key_remap_rules() {
        remapper = remapper
            .add_pattern(from, to)
            .map_err(|err| format!("invalid sparse flow remap rule {from}->{to}: {err}"))?;
    }

    Ok(SafetensorsStore::from_file(path)
        .with_from_adapter(PyTorchToBurnAdapter)
        .allow_partial(false)
        .remap(remapper)
        .validate(true))
}

fn key_remap_rules() -> &'static [(&'static str, &'static str)] {
    &[
        (r"^(t_embedder)\.mlp\.0\.(weight|bias)$", "$1.mlp_0.$2"),
        (r"^(t_embedder)\.mlp\.2\.(weight|bias)$", "$1.mlp_2.$2"),
        (
            r"^(adaLN_modulation)\.1\.(weight|bias)$",
            "ada_ln_modulation.$2",
        ),
        (
            r"^(blocks\.\d+\.mlp)\.mlp\.0\.(weight|bias)$",
            "$1.mlp_0.$2",
        ),
        (
            r"^(blocks\.\d+\.mlp)\.mlp\.2\.(weight|bias)$",
            "$1.mlp_2.$2",
        ),
        (r"^(blocks\.\d+\.norm2)\.weight$", "$1.gamma"),
        (r"^(blocks\.\d+\.norm2)\.bias$", "$1.beta"),
    ]
}

fn resolve_model_weight_candidates(
    model_stem: &str,
    weights_root: &Path,
    image_large_root: Option<&Path>,
) -> Vec<PathBuf> {
    let source =
        resolve_model_source_path(model_stem, "safetensors", weights_root, image_large_root);
    let burnpack = source.with_extension("bpk");
    let burnpack_f16 = with_file_stem_suffix(&burnpack, F16_SUFFIX);
    let prefer_f16 = prefer_f16_burnpack();
    let candidates = if prefer_f16 {
        vec![burnpack_f16, burnpack, source]
    } else {
        vec![burnpack, burnpack_f16, source]
    };
    candidates
        .into_iter()
        .filter(|path| path.exists())
        .collect::<Vec<_>>()
}

fn prefer_f16_burnpack() -> bool {
    let precision = std::env::var("TRELLIS2_BPK_PRECISION")
        .ok()
        .or_else(|| std::env::var("BURN_SYNTH_BPK_PRECISION").ok());
    match precision
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

fn resolve_model_source_path(
    stem: &str,
    ext: &str,
    weights_root: &Path,
    image_large_root: Option<&Path>,
) -> PathBuf {
    if stem.starts_with("ckpts/") {
        return weights_root.join(format!("{stem}.{ext}"));
    }
    if let Some((_, suffix)) = stem.split_once("/ckpts/") {
        let image_large_root = image_large_root.unwrap_or(weights_root);
        return image_large_root.join(format!("ckpts/{suffix}.{ext}"));
    }
    weights_root.join(format!("{stem}.{ext}"))
}

fn with_file_stem_suffix(path: &Path, suffix: &str) -> PathBuf {
    let Some(stem) = path.file_stem() else {
        return path.to_path_buf();
    };
    let stem = stem.to_string_lossy();
    if stem.ends_with(suffix) {
        return path.to_path_buf();
    }
    let ext = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("");
    let mut file_name = format!("{stem}{suffix}");
    if !ext.is_empty() {
        file_name.push('.');
        file_name.push_str(ext);
    }
    path.with_file_name(file_name)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use burn::prelude::Backend;
    use burn::tensor::Tensor;

    use super::{SparseStructureFlowConfig, SparseStructureFlowModel, SparseStructureFlowRuntime};

    #[test]
    fn parses_sparse_structure_flow_config_json() {
        let json = br#"{
            "name": "SparseStructureFlowModel",
            "args": {
                "resolution": 16,
                "in_channels": 8,
                "out_channels": 8,
                "model_channels": 1536,
                "cond_channels": 1024,
                "num_blocks": 30,
                "num_heads": 12,
                "mlp_ratio": 5.3334,
                "pe_mode": "rope",
                "share_mod": true,
                "qk_rms_norm": true,
                "qk_rms_norm_cross": true
            }
        }"#;
        let parsed = SparseStructureFlowConfig::from_json_bytes(json).expect("config should parse");
        assert_eq!(parsed.resolution, 16);
        assert_eq!(parsed.in_channels, 8);
        assert_eq!(parsed.num_heads(), 12);
        assert_eq!(parsed.pe_mode, "rope");
        assert!(parsed.share_mod);
    }

    #[test]
    fn runtime_model_smoke_load_and_predict() {
        if std::env::var("TRELLIS2_RUNTIME_MODEL_SMOKE").is_err() {
            eprintln!(
                "Skipping sparse flow runtime smoke test: set TRELLIS2_RUNTIME_MODEL_SMOKE=1 to enable."
            );
            return;
        }
        let weights_root = std::env::var("TRELLIS2_WEIGHTS_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(
                    "E:/models/huggingface/hub/models--microsoft--TRELLIS.2-4B/snapshots/af44b45f2e35a493886929c6d786e563ec68364d",
                )
            });
        if !weights_root.exists() {
            eprintln!(
                "Skipping sparse flow runtime smoke test: TRELLIS2 weights root missing at {}",
                weights_root.display()
            );
            return;
        }

        let runtime = SparseStructureFlowRuntime::load_from_stem(
            weights_root.as_path(),
            None,
            "ckpts/ss_flow_img_dit_1_3B_64_bf16",
            false,
            None,
        )
        .expect("sparse flow runtime should load from model stem");
        let cfg = runtime.config();
        let voxels = cfg.resolution * cfg.resolution * cfg.resolution;
        let sample = vec![0.0f32; cfg.in_channels * voxels];
        let cond_tokens = 32 * 32;
        let cond = vec![0.0f32; cond_tokens * cfg.cond_channels];
        let prepared = runtime
            .prepare_condition(cond.as_slice(), cond_tokens)
            .expect("sparse flow cond should prepare");
        let out = runtime
            .predict_velocity_with_condition(sample.as_slice(), 1.0, &prepared, None)
            .expect("sparse flow runtime forward should succeed");
        assert_eq!(out.len(), sample.len());
    }

    fn run_tiny_forward<B: Backend>(device: &B::Device) {
        let config = SparseStructureFlowConfig {
            resolution: 2,
            in_channels: 2,
            out_channels: 2,
            model_channels: 8,
            cond_channels: 4,
            num_blocks: 1,
            num_heads: Some(2),
            num_head_channels: 4,
            mlp_ratio: 2.0,
            pe_mode: "rope".to_string(),
            rope_freq: [1.0, 10_000.0],
            share_mod: true,
            qk_rms_norm: true,
            qk_rms_norm_cross: true,
            frequency_embedding_size: 8,
        };
        let model = SparseStructureFlowModel::<B>::new(device, config.clone());
        let x = Tensor::<B, 5>::zeros(
            [
                1,
                config.in_channels,
                config.resolution,
                config.resolution,
                config.resolution,
            ],
            device,
        );
        let t = Tensor::<B, 1>::from_floats([1.0], device);
        let cond = Tensor::<B, 3>::zeros([1, 4, config.cond_channels], device);
        let out = model.forward(x, t, cond);
        assert_eq!(
            out.dims(),
            [
                1,
                config.out_channels,
                config.resolution,
                config.resolution,
                config.resolution
            ]
        );
    }

    #[test]
    fn tiny_sparse_flow_forward_cpu_backend() {
        let device = <burn::backend::NdArray<f32> as Backend>::Device::default();
        run_tiny_forward::<burn::backend::NdArray<f32>>(&device);
    }

    #[cfg(feature = "runtime-model-wgpu")]
    #[test]
    fn tiny_sparse_flow_forward_wgpu_backend() {
        if std::env::var("BURN_WGPU_SMOKE").is_err() {
            eprintln!("skipping: set BURN_WGPU_SMOKE=1 to run wgpu sparse flow smoke");
            return;
        }
        let result = std::panic::catch_unwind(|| {
            let device = burn_wgpu::WgpuDevice::default();
            run_tiny_forward::<burn_wgpu::Wgpu<f32, i32, u32>>(&device);
        });
        if result.is_err() {
            eprintln!("skipping: wgpu backend not available on this system");
        }
    }
}

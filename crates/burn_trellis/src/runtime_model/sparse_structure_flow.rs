use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use burn::module::{Ignored, Module, Param, ParamId};
use burn::nn;
use burn::prelude::Backend;
use burn::tensor::activation::{sigmoid, softmax};
use burn::tensor::{Int, Tensor, TensorData};
use burn_store::{
    BurnpackStore, KeyRemapper, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore,
};
use serde::{Deserialize, Serialize};

use crate::sampler::{
    FlowEulerSampleConfig, FlowEulerSampleTrace, mid_snapshot_step, timestep_pairs,
};

const F16_SUFFIX: &str = "_f16";
const MAX_PERIOD: f32 = 10_000.0;
const LAYER_NORM_EPS: f32 = 1.0e-6;
const RMS_NORM_EPS: f32 = 1.0e-12;
static HOST_READBACK_COUNT: AtomicU64 = AtomicU64::new(0);
static HOST_READBACK_ELEMENTS: AtomicU64 = AtomicU64::new(0);

type CpuRuntimeBackend = burn::backend::NdArray<f32>;
#[cfg(feature = "runtime-model-wgpu")]
type WgpuRuntimeBackend = burn_wgpu::Wgpu<f32, i32, u32>;

#[derive(Clone, Copy, Debug, Default)]
pub struct HostTransferStats {
    pub readback_count: u64,
    pub readback_elements: u64,
}

#[derive(Module, Debug)]
struct BinaryBlob<B: Backend> {
    bytes: Param<Tensor<B, 1, Int>>,
}

#[derive(Debug, Deserialize, Serialize)]
struct BlobMetadata {
    bytes_len: usize,
}

#[derive(Clone)]
struct RowGatherPlan<B: Backend> {
    channels: usize,
    segment_len: usize,
    index_tensor: Tensor<B, 1, Int>,
}

pub fn reset_host_transfer_stats() {
    HOST_READBACK_COUNT.store(0, Ordering::Relaxed);
    HOST_READBACK_ELEMENTS.store(0, Ordering::Relaxed);
}

pub fn host_transfer_stats() -> HostTransferStats {
    HostTransferStats {
        readback_count: HOST_READBACK_COUNT.load(Ordering::Relaxed),
        readback_elements: HOST_READBACK_ELEMENTS.load(Ordering::Relaxed),
    }
}

fn record_host_readback(elements: usize) {
    HOST_READBACK_COUNT.fetch_add(1, Ordering::Relaxed);
    HOST_READBACK_ELEMENTS.fetch_add(elements as u64, Ordering::Relaxed);
}

#[derive(Clone, Debug)]
pub struct SparseFlowRowTrace {
    pub steps: usize,
    pub row_channels: usize,
    pub samples: Vec<f32>,
    pub step_0_x_t: Vec<f32>,
    pub step_mid_x_t: Vec<f32>,
    pub step_last_x_t: Vec<f32>,
}

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

    #[allow(clippy::too_many_arguments)]
    fn sample_with_trace(
        &self,
        noise: &[f32],
        sample_cfg: FlowEulerSampleConfig,
        sigma_min: f32,
        cond: Tensor<B, 3>,
        neg_cond: Tensor<B, 3>,
        concat_cond: Option<&[f32]>,
        capture_snapshots: bool,
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

        let mut step_0_x_t: Option<Tensor<B, 5>> = None;
        let mut step_mid_x_t: Option<Tensor<B, 5>> = None;
        let mut step_last_x_t: Option<Tensor<B, 5>> = None;
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
            if capture_snapshots && step_idx == 0 {
                step_0_x_t = Some(x_t.clone());
            }
            if capture_snapshots && step_idx == mid_step {
                step_mid_x_t = Some(x_t.clone());
            }
            if capture_snapshots && step_idx + 1 == sample_cfg.steps {
                step_last_x_t = Some(x_t.clone());
            }
        }

        let state_len = state_channels.saturating_mul(voxel);
        let (samples, step_0_x_t, step_mid_x_t, step_last_x_t) = if capture_snapshots {
            let samples_t = x_t;
            let step_0_t = step_0_x_t
                .unwrap_or_else(|| samples_t.clone())
                .reshape([state_len]);
            let step_mid_t = step_mid_x_t
                .unwrap_or_else(|| samples_t.clone())
                .reshape([state_len]);
            let step_last_t = step_last_x_t
                .unwrap_or_else(|| samples_t.clone())
                .reshape([state_len]);
            let merged = Tensor::cat(
                vec![
                    samples_t.reshape([state_len]),
                    step_0_t,
                    step_mid_t,
                    step_last_t,
                ],
                0,
            );
            let merged = tensor_to_vec_1d(merged, "failed to read sparse trace tensor")?;
            let segment = state_len;
            let samples = merged[..segment].to_vec();
            let step_0_x_t = merged[segment..segment * 2].to_vec();
            let step_mid_x_t = merged[segment * 2..segment * 3].to_vec();
            let step_last_x_t = merged[segment * 3..segment * 4].to_vec();
            (samples, step_0_x_t, step_mid_x_t, step_last_x_t)
        } else {
            let samples = tensor_to_vec(x_t)?;
            (samples.clone(), samples.clone(), samples.clone(), samples)
        };

        Ok(FlowEulerSampleTrace {
            steps: sample_cfg.steps,
            step_0_x_t,
            step_mid_x_t,
            step_last_x_t,
            samples,
        })
    }

    fn build_row_gather_plan(
        &self,
        state_channels: usize,
        voxel: usize,
        dense_indices: &[usize],
        row_channels: usize,
    ) -> Result<Option<RowGatherPlan<B>>, String> {
        let channels = row_channels.min(state_channels);
        if channels == 0 || dense_indices.is_empty() {
            return Ok(None);
        }
        let gather_len = channels.saturating_mul(dense_indices.len());
        let mut gather_indices = Vec::with_capacity(gather_len);
        for &dense_idx in dense_indices {
            if dense_idx >= voxel {
                return Err(format!(
                    "dense row index out of bounds: idx={} voxel_count={}",
                    dense_idx, voxel
                ));
            }
            for ch in 0..channels {
                let flat_idx = ch.saturating_mul(voxel).saturating_add(dense_idx);
                if flat_idx > i32::MAX as usize {
                    return Err(format!(
                        "dense row gather index overflow: idx={} > i32::MAX",
                        flat_idx
                    ));
                }
                gather_indices.push(flat_idx as i32);
            }
        }
        let gather_shape = [gather_indices.len()];
        let index_tensor = Tensor::<B, 1, Int>::from_data(
            TensorData::new(gather_indices, gather_shape).convert::<i32>(),
            &self.device,
        );
        Ok(Some(RowGatherPlan {
            channels,
            segment_len: gather_len,
            index_tensor,
        }))
    }

    fn gather_rows_tensor(
        &self,
        state: Tensor<B, 5>,
        plan: &RowGatherPlan<B>,
    ) -> Result<Tensor<B, 1>, String> {
        let [_, state_channels, rx, ry, rz] = state.dims();
        let voxel = rx.saturating_mul(ry).saturating_mul(rz).max(1);
        if plan.channels > state_channels {
            return Err(format!(
                "dense row gather channel mismatch: requested={} available={}",
                plan.channels, state_channels
            ));
        }
        let flat = state.reshape([state_channels.saturating_mul(voxel)]);
        Ok(flat.select(0, plan.index_tensor.clone()))
    }

    #[allow(clippy::too_many_arguments)]
    fn sample_rows_with_trace(
        &self,
        noise: &[f32],
        sample_cfg: FlowEulerSampleConfig,
        sigma_min: f32,
        cond: Tensor<B, 3>,
        neg_cond: Tensor<B, 3>,
        concat_cond: Option<&[f32]>,
        dense_indices: &[usize],
        row_channels: usize,
        capture_snapshots: bool,
    ) -> Result<SparseFlowRowTrace, String> {
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

        let gather_plan =
            self.build_row_gather_plan(state_channels, voxel, dense_indices, row_channels)?;
        let mut step_0_rows: Option<Tensor<B, 1>> = None;
        let mut step_mid_rows: Option<Tensor<B, 1>> = None;
        let mut step_last_rows: Option<Tensor<B, 1>> = None;
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
            if capture_snapshots
                && step_idx == 0
                && let Some(plan) = gather_plan.as_ref()
            {
                step_0_rows = Some(self.gather_rows_tensor(x_t.clone(), plan)?);
            }
            if capture_snapshots
                && step_idx == mid_step
                && let Some(plan) = gather_plan.as_ref()
            {
                step_mid_rows = Some(self.gather_rows_tensor(x_t.clone(), plan)?);
            }
            if capture_snapshots
                && step_idx + 1 == sample_cfg.steps
                && let Some(plan) = gather_plan.as_ref()
            {
                step_last_rows = Some(self.gather_rows_tensor(x_t.clone(), plan)?);
            }
        }

        let (samples, step_0_x_t, step_mid_x_t, step_last_x_t) = if let Some(plan) =
            gather_plan.as_ref()
        {
            let samples_t = self.gather_rows_tensor(x_t, plan)?;
            if capture_snapshots {
                let step_0_t = step_0_rows.unwrap_or_else(|| samples_t.clone());
                let step_mid_t = step_mid_rows.unwrap_or_else(|| samples_t.clone());
                let step_last_t = step_last_rows.unwrap_or_else(|| samples_t.clone());
                let merged = Tensor::cat(vec![samples_t, step_0_t, step_mid_t, step_last_t], 0);
                let merged = tensor_to_vec_1d(merged, "failed to read sparse-row trace tensor")?;
                let segment = plan.segment_len;
                let samples = merged[..segment].to_vec();
                let step_0_x_t = merged[segment..segment * 2].to_vec();
                let step_mid_x_t = merged[segment * 2..segment * 3].to_vec();
                let step_last_x_t = merged[segment * 3..segment * 4].to_vec();
                (samples, step_0_x_t, step_mid_x_t, step_last_x_t)
            } else {
                let samples = tensor_to_vec_1d(samples_t, "failed to read sparse-row tensor")?;
                (samples.clone(), samples.clone(), samples.clone(), samples)
            }
        } else {
            (Vec::new(), Vec::new(), Vec::new(), Vec::new())
        };

        Ok(SparseFlowRowTrace {
            steps: sample_cfg.steps,
            row_channels: row_channels.min(state_channels),
            samples,
            step_0_x_t,
            step_mid_x_t,
            step_last_x_t,
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

        let pos = self.predict_velocity_tensor(x_t.clone(), timestep, cond, concat_cond.clone())?;
        let neg = self.predict_velocity_tensor(x_t.clone(), timestep, neg_cond, concat_cond)?;
        let mut pred = pos.clone().mul_scalar(w).add(neg.mul_scalar(1.0 - w));
        if config.guidance_rescale <= 0.0 {
            return Ok(pred);
        }

        let x0_pos = pred_to_xstart_tensor(x_t.clone(), timestep, pos, sigma_min);
        let x0_cfg = pred_to_xstart_tensor(x_t.clone(), timestep, pred, sigma_min);
        let std_pos = tensor_std_tensor(x0_pos);
        let std_cfg = tensor_std_tensor(x0_cfg.clone()).add_scalar(1.0e-12);
        let scale = std_pos.div(std_cfg).reshape([1, 1, 1, 1, 1]);
        let x0 = x0_cfg
            .clone()
            .mul(scale)
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
                Ok(runtime) => {
                    if sparse_flow_wgpu_may_overflow(runtime.config()) {
                        eprintln!(
                            "burn_trellis: sparse flow wgpu disabled for model '{}' due estimated peak tensor bytes (resolution={}, model_channels={}); falling back to cpu.",
                            model_stem,
                            runtime.config().resolution,
                            runtime.config().model_channels
                        );
                    } else {
                        return Ok(Self::Wgpu(runtime));
                    }
                }
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

    #[allow(clippy::too_many_arguments)]
    pub fn sample_with_trace(
        &self,
        noise: &[f32],
        sample_cfg: FlowEulerSampleConfig,
        sigma_min: f32,
        condition: &SparseFlowCondition,
        negative_condition: &SparseFlowCondition,
        concat_cond: Option<&[f32]>,
        capture_snapshots: bool,
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
                    capture_snapshots,
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
                    capture_snapshots,
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
                capture_snapshots,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sample_rows_with_trace(
        &self,
        noise: &[f32],
        sample_cfg: FlowEulerSampleConfig,
        sigma_min: f32,
        condition: &SparseFlowCondition,
        negative_condition: &SparseFlowCondition,
        concat_cond: Option<&[f32]>,
        dense_indices: &[usize],
        row_channels: usize,
        capture_snapshots: bool,
    ) -> Result<SparseFlowRowTrace, String> {
        #[cfg(feature = "runtime-model-wgpu")]
        {
            match (self, condition, negative_condition) {
                (
                    Self::Cpu(runtime),
                    SparseFlowCondition::Cpu(cond),
                    SparseFlowCondition::Cpu(neg_cond),
                ) => runtime.sample_rows_with_trace(
                    noise,
                    sample_cfg,
                    sigma_min,
                    cond.clone(),
                    neg_cond.clone(),
                    concat_cond,
                    dense_indices,
                    row_channels,
                    capture_snapshots,
                ),
                (
                    Self::Wgpu(runtime),
                    SparseFlowCondition::Wgpu(cond),
                    SparseFlowCondition::Wgpu(neg_cond),
                ) => runtime.sample_rows_with_trace(
                    noise,
                    sample_cfg,
                    sigma_min,
                    cond.clone(),
                    neg_cond.clone(),
                    concat_cond,
                    dense_indices,
                    row_channels,
                    capture_snapshots,
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
            runtime.sample_rows_with_trace(
                noise,
                sample_cfg,
                sigma_min,
                cond.clone(),
                neg_cond.clone(),
                concat_cond,
                dense_indices,
                row_channels,
                capture_snapshots,
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

fn tensor_std_tensor<B: Backend>(tensor: Tensor<B, 5>) -> Tensor<B, 1> {
    let [b, c, x, y, z] = tensor.dims();
    let numel = b
        .saturating_mul(c)
        .saturating_mul(x)
        .saturating_mul(y)
        .saturating_mul(z)
        .max(1);
    let flat = tensor.reshape([numel]);
    let mean = flat.clone().mean_dim(0);
    flat.sub(mean).powf_scalar(2.0).mean_dim(0).sqrt()
}

fn tensor_to_vec<B: Backend>(tensor: Tensor<B, 5>) -> Result<Vec<f32>, String> {
    let [b, c, x, y, z] = tensor.dims();
    let elements = b
        .saturating_mul(c)
        .saturating_mul(x)
        .saturating_mul(y)
        .saturating_mul(z);
    let values = tensor
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|err| format!("failed to read sparse flow tensor: {err:?}"))?;
    record_host_readback(elements.max(values.len()));
    Ok(values)
}

fn tensor_to_vec_1d<B: Backend>(tensor: Tensor<B, 1>, context: &str) -> Result<Vec<f32>, String> {
    let [elements] = tensor.dims();
    let values = tensor
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|err| format!("{context}: {err:?}"))?;
    record_host_readback(elements.max(values.len()));
    Ok(values)
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AttentionImpl {
    Dense,
    Stream,
}

fn env_flag_enabled(key: &str) -> bool {
    std::env::var(key)
        .ok()
        .map(|raw| {
            matches!(
                raw.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

#[cfg(feature = "runtime-model-wgpu")]
fn sparse_flow_wgpu_max_peak_bytes() -> usize {
    std::env::var("TRELLIS2_SPARSE_FLOW_WGPU_MAX_PEAK_BYTES")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(3 * 1024 * 1024 * 1024)
}

#[cfg(feature = "runtime-model-wgpu")]
fn sparse_flow_wgpu_estimated_peak_bytes(config: &SparseStructureFlowConfig) -> usize {
    let tokens = config
        .resolution
        .checked_mul(config.resolution)
        .and_then(|value| value.checked_mul(config.resolution))
        .unwrap_or(usize::MAX);
    let qkv_channels = config.model_channels.saturating_mul(3);
    let mlp_channels = ((config.model_channels as f32) * config.mlp_ratio.max(1.0))
        .ceil()
        .max(config.model_channels as f32) as usize;
    let peak_channels = qkv_channels.max(mlp_channels);
    tokens
        .checked_mul(peak_channels)
        .and_then(|value| value.checked_mul(core::mem::size_of::<f32>()))
        .unwrap_or(usize::MAX)
}

#[cfg(feature = "runtime-model-wgpu")]
fn sparse_flow_wgpu_may_overflow(config: &SparseStructureFlowConfig) -> bool {
    let estimated = sparse_flow_wgpu_estimated_peak_bytes(config);
    estimated > sparse_flow_wgpu_max_peak_bytes()
}

fn attention_debug_enabled() -> bool {
    env_flag_enabled("TRELLIS2_ATTN_DEBUG")
}

fn attention_prefers_stream() -> bool {
    #[cfg(feature = "runtime-model-wgpu")]
    {
        true
    }
    #[cfg(not(feature = "runtime-model-wgpu"))]
    {
        false
    }
}

// Avoid 4D matmul layout expansion on fusion/cubecl backends by flattening batch*heads.
fn matmul_4d_via_3d<B: Backend>(lhs: Tensor<B, 4>, rhs: Tensor<B, 4>) -> Tensor<B, 4> {
    let [batch, heads, m, k] = lhs.dims();
    let [rhs_batch, rhs_heads, rhs_k, n] = rhs.dims();
    if batch != rhs_batch || heads != rhs_heads || k != rhs_k {
        panic!(
            "4d matmul shape mismatch: lhs=[{batch},{heads},{m},{k}] rhs=[{rhs_batch},{rhs_heads},{rhs_k},{n}]"
        );
    }
    let bh = batch.saturating_mul(heads).max(1);
    lhs.clone()
        .reshape([bh, m, k])
        .matmul(rhs.clone().reshape([bh, rhs_k, n]))
        .reshape([batch, heads, m, n])
}

fn attention_impl(query_tokens: usize, key_tokens: usize) -> AttentionImpl {
    let auto =
        if env_flag_enabled("TRELLIS2_PARITY_STRICT") || env_flag_enabled("TRELLIS2_E2E_STRICT") {
            AttentionImpl::Dense
        } else {
            let work = query_tokens.saturating_mul(key_tokens);
            if (attention_prefers_stream() && work >= 64usize.saturating_mul(64))
                || work >= 512usize.saturating_mul(512)
            {
                AttentionImpl::Stream
            } else {
                AttentionImpl::Dense
            }
        };

    match std::env::var("TRELLIS2_ATTN_BACKEND")
        .ok()
        .map(|value| value.trim().to_ascii_lowercase())
        .as_deref()
    {
        Some("dense") | Some("chunked") | Some("naive") | Some("sdpa") => AttentionImpl::Dense,
        Some("stream") | Some("flash") | Some("flash_like") | Some("varlen") => {
            AttentionImpl::Stream
        }
        Some("auto") | None => auto,
        Some(_) => auto,
    }
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
    let [_, _, query_tokens, _] = q.dims();
    let [_, _, key_tokens, _] = k.dims();

    let attention_impl = attention_impl(query_tokens, key_tokens);
    if attention_debug_enabled() && query_tokens >= 4096 {
        let backend_name = std::any::type_name::<B>();
        let impl_name = match attention_impl {
            AttentionImpl::Dense => "dense",
            AttentionImpl::Stream => "stream",
        };
        eprintln!(
            "burn_trellis: attn dispatch backend={backend_name} impl={impl_name} q={query_tokens} k={key_tokens} head_dim={head_dim}"
        );
    }

    let out = match attention_impl {
        AttentionImpl::Dense => scaled_dot_product_attention_dense(q, k, v, head_dim),
        AttentionImpl::Stream => scaled_dot_product_attention_stream(q, k, v, head_dim),
    };
    out.permute([0, 2, 1, 3])
}

fn scaled_dot_product_attention_dense<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    head_dim: usize,
) -> Tensor<B, 4> {
    let [batch, heads, tokens, _] = q.dims();
    let [_, _, key_tokens, _] = k.dims();
    let scale = 1.0 / (head_dim as f32).sqrt();
    let query_chunk = attention_query_chunk(tokens, 8);
    let logits_budget = attention_logits_budget_bytes();
    let dense_logits_bytes = attention_logits_bytes(batch, heads, tokens, key_tokens);
    if attention_debug_enabled() && tokens >= 4096 {
        eprintln!(
            "burn_trellis: attn dense q={tokens} k={key_tokens} query_chunk={query_chunk} logits_bytes={dense_logits_bytes} budget_bytes={logits_budget}"
        );
    }

    if query_chunk >= tokens && dense_logits_bytes <= logits_budget {
        let attn = softmax(
            matmul_4d_via_3d(q.clone(), k.clone().swap_dims(2, 3)).mul_scalar(scale),
            3,
        );
        return matmul_4d_via_3d(attn, v);
    }

    let k_t = k.clone().swap_dims(2, 3);
    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < tokens {
        let end = (start + query_chunk).min(tokens);
        let q_chunk = q
            .clone()
            .slice([0..batch, 0..heads, start..end, 0..head_dim])
            .clone();
        let attn = softmax(matmul_4d_via_3d(q_chunk, k_t.clone()).mul_scalar(scale), 3);
        chunks.push(matmul_4d_via_3d(attn, v.clone()));
        start = end;
    }
    Tensor::cat(chunks, 2)
}

fn scaled_dot_product_attention_stream<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    head_dim: usize,
) -> Tensor<B, 4> {
    let [batch, heads, query_tokens, _] = q.dims();
    let [_, _, key_tokens, _] = k.dims();
    let [_, _, _, value_dim] = v.dims();
    let scale = 1.0 / (head_dim as f32).sqrt();
    let query_chunk = attention_query_chunk(query_tokens, 64);
    let key_chunk = attention_key_chunk(key_tokens);
    let logits_budget = attention_logits_budget_bytes();
    let dense_logits_bytes = attention_logits_bytes(batch, heads, query_tokens, key_tokens);
    if attention_debug_enabled() && query_tokens >= 4096 {
        eprintln!(
            "burn_trellis: attn stream q={query_tokens} k={key_tokens} query_chunk={query_chunk} key_chunk={key_chunk} logits_bytes={dense_logits_bytes} budget_bytes={logits_budget}"
        );
    }

    if query_chunk >= query_tokens && key_chunk >= key_tokens && dense_logits_bytes <= logits_budget
    {
        let attn = softmax(
            matmul_4d_via_3d(q.clone(), k.clone().swap_dims(2, 3)).mul_scalar(scale),
            3,
        );
        return matmul_4d_via_3d(attn, v);
    }

    let mut outputs = Vec::new();
    let mut q_start = 0usize;
    while q_start < query_tokens {
        let q_end = (q_start + query_chunk).min(query_tokens);
        let q_chunk = q
            .clone()
            .slice([0..batch, 0..heads, q_start..q_end, 0..head_dim])
            .clone();

        let first_k_end = key_chunk.min(key_tokens);
        let first_k = k
            .clone()
            .slice([0..batch, 0..heads, 0..first_k_end, 0..head_dim])
            .clone();
        let first_v = v
            .clone()
            .slice([0..batch, 0..heads, 0..first_k_end, 0..value_dim])
            .clone();

        let first_logits =
            matmul_4d_via_3d(q_chunk.clone(), first_k.swap_dims(2, 3)).mul_scalar(scale);
        let mut max_scores = first_logits.clone().max_dim(3);
        let first_probs = first_logits.sub(max_scores.clone()).exp();
        let mut denom = first_probs.clone().sum_dim(3);
        let mut acc = matmul_4d_via_3d(first_probs, first_v);

        let mut k_start = first_k_end;
        while k_start < key_tokens {
            let k_end = (k_start + key_chunk).min(key_tokens);
            let k_chunk = k
                .clone()
                .slice([0..batch, 0..heads, k_start..k_end, 0..head_dim])
                .clone();
            let v_chunk = v
                .clone()
                .slice([0..batch, 0..heads, k_start..k_end, 0..value_dim])
                .clone();
            let logits =
                matmul_4d_via_3d(q_chunk.clone(), k_chunk.swap_dims(2, 3)).mul_scalar(scale);
            let chunk_max = logits.clone().max_dim(3);
            let probs = logits.sub(chunk_max.clone()).exp();
            let chunk_denom = probs.clone().sum_dim(3);
            let chunk_acc = matmul_4d_via_3d(probs, v_chunk);

            let max_new = max_scores.clone().max_pair(chunk_max.clone());
            let alpha = max_scores.clone().sub(max_new.clone()).exp();
            let beta = chunk_max.sub(max_new.clone()).exp();
            let denom_new = alpha
                .clone()
                .mul(denom.clone())
                .add(beta.clone().mul(chunk_denom.clone()))
                .add_scalar(1.0e-12);
            let prev_scale = alpha.mul(denom).div(denom_new.clone());
            let chunk_scale = beta.mul(chunk_denom).div(denom_new.clone());
            acc = acc.mul(prev_scale).add(chunk_acc.mul(chunk_scale));

            max_scores = max_new;
            denom = denom_new;
            k_start = k_end;
        }

        outputs.push(acc);
        q_start = q_end;
    }

    Tensor::cat(outputs, 2)
}

fn attention_logits_bytes(
    batch: usize,
    heads: usize,
    query_tokens: usize,
    key_tokens: usize,
) -> usize {
    batch
        .saturating_mul(heads)
        .saturating_mul(query_tokens)
        .saturating_mul(key_tokens)
        .saturating_mul(std::mem::size_of::<f32>())
}

fn attention_logits_budget_bytes() -> usize {
    std::env::var("TRELLIS2_ATTN_MAX_LOGITS_BYTES")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or_else(|| {
            if attention_prefers_stream() {
                512 * 1024 * 1024
            } else {
                usize::MAX
            }
        })
}

fn attention_query_chunk(tokens: usize, default_chunk: usize) -> usize {
    let env_chunk = std::env::var("TRELLIS2_ATTN_QUERY_CHUNK")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default_chunk);
    let max_chunk = std::env::var("TRELLIS2_ATTN_QUERY_CHUNK_MAX")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or_else(|| {
            if attention_prefers_stream() {
                256
            } else {
                usize::MAX
            }
        });
    env_chunk.min(max_chunk).min(tokens.max(1))
}

fn attention_key_chunk(tokens: usize) -> usize {
    let env_chunk = std::env::var("TRELLIS2_ATTN_KEY_CHUNK")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(128);
    let max_chunk = std::env::var("TRELLIS2_ATTN_KEY_CHUNK_MAX")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or_else(|| {
            if attention_prefers_stream() {
                512
            } else {
                usize::MAX
            }
        });
    env_chunk.min(max_chunk).min(tokens.max(1))
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
        match model.load_from(&mut store) {
            Ok(_) => Ok(()),
            Err(module_err) => {
                let blob_bytes = load_burnpack_blob_bytes(path).map_err(|blob_err| {
                    format!(
                        "failed to load sparse flow burnpack '{}' as module ({module_err}) or blob ({blob_err})",
                        path.display()
                    )
                })?;
                let mut safetensor_store = build_safetensor_store_from_bytes(blob_bytes)?;
                model
                    .load_from(&mut safetensor_store)
                    .map(|_| ())
                    .map_err(|safetensor_err| {
                        format!(
                            "failed to load sparse flow burnpack '{}' as safetensors blob after module load error ({module_err}): {safetensor_err}",
                            path.display()
                        )
                    })
            }
        }
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

fn build_safetensor_store_from_bytes(bytes: Vec<u8>) -> Result<SafetensorsStore, String> {
    let mut remapper = KeyRemapper::new();
    for &(from, to) in key_remap_rules() {
        remapper = remapper
            .add_pattern(from, to)
            .map_err(|err| format!("invalid sparse flow remap rule {from}->{to}: {err}"))?;
    }

    Ok(SafetensorsStore::from_bytes(Some(bytes))
        .with_from_adapter(PyTorchToBurnAdapter)
        .allow_partial(false)
        .remap(remapper)
        .validate(true))
}

fn load_burnpack_blob_bytes(path: &Path) -> Result<Vec<u8>, String> {
    let metadata_path = metadata_path(path);
    let metadata: BlobMetadata =
        serde_json::from_slice(&std::fs::read(&metadata_path).map_err(|err| {
            format!(
                "failed to read burnpack metadata '{}': {err}",
                metadata_path.display()
            )
        })?)
        .map_err(|err| {
            format!(
                "failed to parse burnpack metadata '{}': {err}",
                metadata_path.display()
            )
        })?;

    match load_blob_bytes_with_backend::<burn::backend::NdArray<f32, u8>>(path, metadata.bytes_len)
    {
        Ok(bytes) => Ok(bytes),
        Err(u8_err) => load_blob_bytes_with_backend::<burn::backend::NdArray<f32, i64>>(
            path,
            metadata.bytes_len,
        )
        .map_err(|i64_err| {
            format!(
                "failed to load blob burnpack '{}' (u8 backend: {u8_err}; i64 fallback: {i64_err})",
                path.display()
            )
        }),
    }
}

fn load_blob_bytes_with_backend<B: Backend>(
    path: &Path,
    bytes_len: usize,
) -> Result<Vec<u8>, String>
where
    B::Device: Default,
{
    let device = <B as Backend>::Device::default();
    let zeros = Tensor::<B, 1, Int>::zeros([bytes_len], &device);
    let mut blob = BinaryBlob {
        bytes: Param::initialized(ParamId::new(), zeros),
    };

    let mut store = BurnpackStore::from_file(path).validate(true);
    blob.load_from(&mut store)
        .map_err(|err| format!("failed to load burnpack '{}': {err}", path.display()))?;

    let bytes = blob
        .bytes
        .val()
        .into_data()
        .convert::<u8>()
        .to_vec::<u8>()
        .map_err(|err| format!("failed to materialize burnpack bytes: {err:?}"))?;

    if bytes.len() != bytes_len {
        return Err(format!(
            "burnpack byte length mismatch for '{}': expected {}, got {}",
            path.display(),
            bytes_len,
            bytes.len()
        ));
    }
    Ok(bytes)
}

fn metadata_path(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("model.bpk");
    path.with_file_name(format!("{file_name}.meta.json"))
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
    use std::sync::Mutex;
    use std::time::{Instant, SystemTime, UNIX_EPOCH};

    use burn::module::{Param, ParamId};
    use burn::prelude::Backend;
    use burn::tensor::{Int, Tensor, TensorData};
    use burn_store::{BurnToPyTorchAdapter, BurnpackStore, ModuleSnapshot, SafetensorsStore};

    use crate::sampler::FlowEulerSampleConfig;

    use super::{
        BinaryBlob, BlobMetadata, CpuRuntimeBackend, SparseStructureFlowConfig,
        SparseStructureFlowModel, SparseStructureFlowRuntime, SparseStructureFlowRuntimeImpl,
        host_transfer_stats, metadata_path, reset_host_transfer_stats,
        scaled_dot_product_attention_dense, scaled_dot_product_attention_stream,
    };

    static HOST_STATS_LOCK: Mutex<()> = Mutex::new(());

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

    #[test]
    fn runtime_loads_blob_burnpack_when_module_layout_is_absent() {
        type BlobBackend = burn::backend::NdArray<f32, u8>;
        type TestBackend = burn::backend::NdArray<f32>;

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock drift")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("burn_trellis_sparse_flow_blob_{unique}"));
        let ckpts = root.join("ckpts");
        std::fs::create_dir_all(&ckpts).expect("create ckpt dir");

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

        let config_json = serde_json::json!({
            "name": "SparseStructureFlowModel",
            "args": {
                "resolution": config.resolution,
                "in_channels": config.in_channels,
                "out_channels": config.out_channels,
                "model_channels": config.model_channels,
                "cond_channels": config.cond_channels,
                "num_blocks": config.num_blocks,
                "num_heads": config.num_heads,
                "num_head_channels": config.num_head_channels,
                "mlp_ratio": config.mlp_ratio,
                "pe_mode": config.pe_mode,
                "rope_freq": config.rope_freq,
                "share_mod": config.share_mod,
                "qk_rms_norm": config.qk_rms_norm,
                "qk_rms_norm_cross": config.qk_rms_norm_cross,
                "frequency_embedding_size": config.frequency_embedding_size
            }
        });
        std::fs::write(
            ckpts.join("flow_model.json"),
            serde_json::to_vec_pretty(&config_json).expect("serialize config"),
        )
        .expect("write config");

        let source_path = ckpts.join("flow_model.safetensors");
        let device = <TestBackend as Backend>::Device::default();
        let model = SparseStructureFlowModel::<TestBackend>::new(&device, config.clone());
        let mut source_store =
            SafetensorsStore::from_file(&source_path).with_to_adapter(BurnToPyTorchAdapter);
        model
            .save_into(&mut source_store)
            .expect("save source safetensors");
        let source_bytes = std::fs::read(&source_path).expect("read source safetensors");

        let burnpack_path = ckpts.join("flow_model.bpk");
        let blob_device = <BlobBackend as Backend>::Device::default();
        let tensor = Tensor::<BlobBackend, 1, Int>::from_data(
            TensorData::new(source_bytes.clone(), [source_bytes.len()]),
            &blob_device,
        );
        let blob = BinaryBlob {
            bytes: Param::initialized(ParamId::new(), tensor),
        };
        let mut burnpack_store = BurnpackStore::from_file(&burnpack_path).overwrite(true);
        blob.save_into(&mut burnpack_store)
            .expect("save blob burnpack");
        let metadata = BlobMetadata {
            bytes_len: source_bytes.len(),
        };
        std::fs::write(
            metadata_path(&burnpack_path),
            serde_json::to_vec_pretty(&metadata).expect("serialize metadata"),
        )
        .expect("write metadata");

        std::fs::remove_file(&source_path).expect("remove source safetensors");

        let runtime = SparseStructureFlowRuntime::load_from_stem(
            root.as_path(),
            None,
            "ckpts/flow_model",
            false,
            None,
        )
        .expect("runtime should load from blob burnpack");
        let cfg = runtime.config();
        let voxels = cfg.resolution * cfg.resolution * cfg.resolution;
        let sample = vec![0.0f32; cfg.in_channels * voxels];
        let cond_tokens = 4;
        let cond = vec![0.0f32; cond_tokens * cfg.cond_channels];
        let prepared = runtime
            .prepare_condition(cond.as_slice(), cond_tokens)
            .expect("prepare cond");
        let out = runtime
            .predict_velocity_with_condition(sample.as_slice(), 1.0, &prepared, None)
            .expect("forward");
        assert_eq!(out.len(), sample.len());

        let _ = std::fs::remove_dir_all(root);
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

    fn make_tiny_runtime_cpu() -> SparseStructureFlowRuntimeImpl<CpuRuntimeBackend> {
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
        let device = <CpuRuntimeBackend as Backend>::Device::default();
        let model = SparseStructureFlowModel::<CpuRuntimeBackend>::new(&device, config.clone());
        SparseStructureFlowRuntimeImpl {
            config,
            model,
            device,
        }
    }

    fn make_attention_tensor(
        device: &<CpuRuntimeBackend as Backend>::Device,
        tokens: usize,
        heads: usize,
        channels: usize,
        seed: f32,
    ) -> Tensor<CpuRuntimeBackend, 4> {
        let mut values = Vec::with_capacity(tokens.saturating_mul(heads).saturating_mul(channels));
        for idx in 0..values.capacity() {
            let x = idx as f32 * 0.013 + seed;
            values.push(x.sin() * 0.7 + x.cos() * 0.3);
        }
        Tensor::<CpuRuntimeBackend, 1>::from_floats(values.as_slice(), device)
            .reshape([1, tokens, heads, channels])
    }

    fn tensor_to_vec4<B: Backend>(tensor: Tensor<B, 4>) -> Vec<f32> {
        tensor
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("tensor should be readable")
    }

    #[test]
    fn attention_stream_matches_dense_reference() {
        let device = <CpuRuntimeBackend as Backend>::Device::default();
        let heads = 4usize;
        let head_dim = 8usize;
        let query_tokens = 32usize;
        let key_tokens = 24usize;

        let q = make_attention_tensor(&device, query_tokens, heads, head_dim, 0.2);
        let k = make_attention_tensor(&device, key_tokens, heads, head_dim, 0.7);
        let v = make_attention_tensor(&device, key_tokens, heads, head_dim, 1.3);

        let q = q.permute([0, 2, 1, 3]);
        let k = k.permute([0, 2, 1, 3]);
        let v = v.permute([0, 2, 1, 3]);

        let dense = scaled_dot_product_attention_dense(q.clone(), k.clone(), v.clone(), head_dim);
        let stream = scaled_dot_product_attention_stream(q, k, v, head_dim);

        let dense = tensor_to_vec4(dense);
        let stream = tensor_to_vec4(stream);
        assert_eq!(dense.len(), stream.len());

        let max_abs = dense
            .iter()
            .zip(stream.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs <= 1.0e-3,
            "stream attention drift too high: max_abs={max_abs:.6e}"
        );
    }

    #[test]
    fn attention_stream_benchmark_report() {
        if std::env::var("TRELLIS2_ATTN_BENCH").is_err() {
            eprintln!("skipping: set TRELLIS2_ATTN_BENCH=1 to run attention benchmark report");
            return;
        }

        let device = <CpuRuntimeBackend as Backend>::Device::default();
        let heads = 8usize;
        let head_dim = 16usize;
        let query_tokens = 160usize;
        let key_tokens = 160usize;
        let iterations = 6usize;

        let q = make_attention_tensor(&device, query_tokens, heads, head_dim, 0.2)
            .permute([0, 2, 1, 3]);
        let k =
            make_attention_tensor(&device, key_tokens, heads, head_dim, 0.7).permute([0, 2, 1, 3]);
        let v =
            make_attention_tensor(&device, key_tokens, heads, head_dim, 1.3).permute([0, 2, 1, 3]);

        let _ = scaled_dot_product_attention_dense(q.clone(), k.clone(), v.clone(), head_dim)
            .into_data();
        let _ = scaled_dot_product_attention_stream(q.clone(), k.clone(), v.clone(), head_dim)
            .into_data();

        let dense_start = Instant::now();
        for _ in 0..iterations {
            let _ = scaled_dot_product_attention_dense(q.clone(), k.clone(), v.clone(), head_dim)
                .into_data();
        }
        let dense_ms = dense_start.elapsed().as_secs_f64() * 1_000.0 / iterations as f64;

        let stream_start = Instant::now();
        for _ in 0..iterations {
            let _ = scaled_dot_product_attention_stream(q.clone(), k.clone(), v.clone(), head_dim)
                .into_data();
        }
        let stream_ms = stream_start.elapsed().as_secs_f64() * 1_000.0 / iterations as f64;
        eprintln!(
            "attention bench: dense={dense_ms:.3}ms stream={stream_ms:.3}ms ratio={:.3}",
            stream_ms / dense_ms
        );
    }

    #[test]
    fn sample_trace_uses_single_host_readback_when_capturing_snapshots() {
        let _guard = HOST_STATS_LOCK
            .lock()
            .expect("host transfer stats lock should not be poisoned");
        let runtime = make_tiny_runtime_cpu();
        let config = runtime.config().clone();
        let voxel = config.resolution * config.resolution * config.resolution;
        let noise = vec![0.0f32; config.out_channels * voxel];
        let cond_tokens = 4usize;
        let cond_values = vec![0.0f32; cond_tokens * config.cond_channels];
        let cond = runtime
            .prepare_condition(cond_values.as_slice(), cond_tokens)
            .expect("prepare cond");
        let neg_cond = runtime
            .prepare_condition(cond_values.as_slice(), cond_tokens)
            .expect("prepare neg cond");
        let sample_cfg = FlowEulerSampleConfig {
            steps: 4,
            rescale_t: 1.0,
            guidance_strength: 1.0,
            guidance_rescale: 0.0,
            guidance_interval: [0.0, 1.0],
        };

        reset_host_transfer_stats();
        let trace = runtime
            .sample_with_trace(
                noise.as_slice(),
                sample_cfg,
                0.1,
                cond,
                neg_cond,
                None,
                true,
            )
            .expect("sample trace");
        let stats = host_transfer_stats();

        assert_eq!(
            stats.readback_count, 1,
            "dense trace snapshot capture should use a single merged host readback"
        );
        let expected_len = noise.len();
        assert_eq!(trace.samples.len(), expected_len);
        assert_eq!(trace.step_0_x_t.len(), expected_len);
        assert_eq!(trace.step_mid_x_t.len(), expected_len);
        assert_eq!(trace.step_last_x_t.len(), expected_len);
    }

    #[test]
    fn sample_rows_trace_uses_single_host_readback_when_capturing_snapshots() {
        let _guard = HOST_STATS_LOCK
            .lock()
            .expect("host transfer stats lock should not be poisoned");
        let runtime = make_tiny_runtime_cpu();
        let config = runtime.config().clone();
        let voxel = config.resolution * config.resolution * config.resolution;
        let noise = vec![0.0f32; config.out_channels * voxel];
        let cond_tokens = 4usize;
        let cond_values = vec![0.0f32; cond_tokens * config.cond_channels];
        let cond = runtime
            .prepare_condition(cond_values.as_slice(), cond_tokens)
            .expect("prepare cond");
        let neg_cond = runtime
            .prepare_condition(cond_values.as_slice(), cond_tokens)
            .expect("prepare neg cond");
        let dense_indices = vec![0usize, 1usize, 3usize];
        let row_channels = 2usize;
        let sample_cfg = FlowEulerSampleConfig {
            steps: 4,
            rescale_t: 1.0,
            guidance_strength: 1.0,
            guidance_rescale: 0.0,
            guidance_interval: [0.0, 1.0],
        };

        reset_host_transfer_stats();
        let trace = runtime
            .sample_rows_with_trace(
                noise.as_slice(),
                sample_cfg,
                0.1,
                cond,
                neg_cond,
                None,
                dense_indices.as_slice(),
                row_channels,
                true,
            )
            .expect("sample rows with trace");
        let stats = host_transfer_stats();

        assert_eq!(
            stats.readback_count, 1,
            "row-trace snapshot capture should use a single merged host readback"
        );
        let expected_len = dense_indices.len() * row_channels;
        assert_eq!(trace.samples.len(), expected_len);
        assert_eq!(trace.step_0_x_t.len(), expected_len);
        assert_eq!(trace.step_mid_x_t.len(), expected_len);
        assert_eq!(trace.step_last_x_t.len(), expected_len);
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

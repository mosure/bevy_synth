use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use super::weight_parts::{candidate_exists_or_has_parts, load_blob_bytes_from_burnpack_or_parts};
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

fn runtime_sample_progress_interval(steps: usize) -> usize {
    if steps <= 16 { 1 } else { (steps / 8).max(1) }
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
            .sum_dim(3)
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
        let [batch, tokens, channels] = x.dims();
        let chunk_tokens = sparse_flow_mlp_chunk_tokens(tokens);
        if chunk_tokens >= tokens {
            return self.mlp_2.forward(gelu(self.mlp_0.forward(x)));
        }

        let mut chunks = Vec::new();
        let mut start = 0usize;
        while start < tokens {
            let end = (start + chunk_tokens).min(tokens);
            let x_chunk = x.clone().slice([0..batch, start..end, 0..channels]);
            chunks.push(self.mlp_2.forward(gelu(self.mlp_0.forward(x_chunk))));
            start = end;
        }
        Tensor::cat(chunks, 1)
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
        if sparse_flow_chunked_forward_enabled(tokens) {
            return self.forward_chunked_stream(x, resolution);
        }
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

    fn forward_chunked_stream(&self, x: Tensor<B, 3>, resolution: usize) -> Tensor<B, 3> {
        let [batch, tokens, channels] = x.dims();
        let kv_chunk_tokens = sparse_flow_self_attn_kv_chunk_tokens(tokens);
        let reuse_qkv = sparse_flow_stream_reuse_qkv_enabled(tokens, channels);
        let query_chunk_tokens = if reuse_qkv {
            kv_chunk_tokens
        } else {
            sparse_flow_self_attn_query_chunk_tokens(tokens)
        };
        let logits_budget = sparse_flow_attn_logits_budget_bytes();
        let (query_chunk_tokens, kv_chunk_tokens) = sparse_flow_stream_chunk_plan(
            batch,
            self.num_heads,
            tokens,
            query_chunk_tokens,
            kv_chunk_tokens,
            reuse_qkv,
            logits_budget,
        );

        let mut k_chunks: Vec<Tensor<B, 4>> = Vec::new();
        let mut v_chunks: Vec<Tensor<B, 4>> = Vec::new();
        let mut q_chunks: Vec<Tensor<B, 4>> = Vec::new();
        let mut kv_start = 0usize;
        while kv_start < tokens {
            let kv_end = (kv_start + kv_chunk_tokens).min(tokens);
            let x_chunk = x.clone().slice([0..batch, kv_start..kv_end, 0..channels]);
            let qkv = self.to_qkv.forward(x_chunk).reshape([
                batch,
                kv_end - kv_start,
                3,
                self.num_heads,
                self.head_dim,
            ]);
            let mut k = qkv
                .clone()
                .slice([
                    0..batch,
                    0..(kv_end - kv_start),
                    1..2,
                    0..self.num_heads,
                    0..self.head_dim,
                ])
                .reshape([batch, kv_end - kv_start, self.num_heads, self.head_dim]);
            let v = qkv
                .clone()
                .slice([
                    0..batch,
                    0..(kv_end - kv_start),
                    2..3,
                    0..self.num_heads,
                    0..self.head_dim,
                ])
                .reshape([batch, kv_end - kv_start, self.num_heads, self.head_dim]);
            let mut q = qkv
                .slice([
                    0..batch,
                    0..(kv_end - kv_start),
                    0..1,
                    0..self.num_heads,
                    0..self.head_dim,
                ])
                .reshape([batch, kv_end - kv_start, self.num_heads, self.head_dim]);

            if let Some(norm) = self.k_rms_norm.as_ref() {
                k = norm.forward(k);
            }
            if let Some(norm) = self.q_rms_norm.as_ref() {
                q = norm.forward(q);
            }
            if self.use_rope {
                k = apply_rope_single(k, resolution, self.head_dim, self.rope_freq, kv_start);
                q = apply_rope_single(q, resolution, self.head_dim, self.rope_freq, kv_start);
            }

            k_chunks.push(k.permute([0, 2, 1, 3]));
            v_chunks.push(v.permute([0, 2, 1, 3]));
            if reuse_qkv {
                q_chunks.push(q.permute([0, 2, 1, 3]));
            }
            kv_start = kv_end;
        }

        let mut out_chunks = Vec::new();
        if reuse_qkv {
            for q in q_chunks.into_iter() {
                let q_tokens = q.dims()[2];
                let out = scaled_dot_product_attention_stream_chunked_keys(
                    q,
                    k_chunks.as_slice(),
                    v_chunks.as_slice(),
                    self.head_dim,
                )
                .permute([0, 2, 1, 3])
                .reshape([batch, q_tokens, channels]);
                out_chunks.push(self.to_out.forward(out));
            }
        } else {
            let mut q_start = 0usize;
            while q_start < tokens {
                let q_end = (q_start + query_chunk_tokens).min(tokens);
                let x_chunk = x.clone().slice([0..batch, q_start..q_end, 0..channels]);
                let qkv = self.to_qkv.forward(x_chunk).reshape([
                    batch,
                    q_end - q_start,
                    3,
                    self.num_heads,
                    self.head_dim,
                ]);
                let mut q = qkv
                    .slice([
                        0..batch,
                        0..(q_end - q_start),
                        0..1,
                        0..self.num_heads,
                        0..self.head_dim,
                    ])
                    .reshape([batch, q_end - q_start, self.num_heads, self.head_dim]);
                if let Some(norm) = self.q_rms_norm.as_ref() {
                    q = norm.forward(q);
                }
                if self.use_rope {
                    q = apply_rope_single(q, resolution, self.head_dim, self.rope_freq, q_start);
                }

                let out = scaled_dot_product_attention_stream_chunked_keys(
                    q.permute([0, 2, 1, 3]),
                    k_chunks.as_slice(),
                    v_chunks.as_slice(),
                    self.head_dim,
                )
                .permute([0, 2, 1, 3])
                .reshape([batch, q_end - q_start, channels]);

                out_chunks.push(self.to_out.forward(out));
                q_start = q_end;
            }
        }

        Tensor::cat(out_chunks, 1)
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
        h = linear_forward_token_chunked(
            &self.input_layer,
            h,
            sparse_flow_linear_chunk_tokens(tokens),
        );

        let t_emb = self
            .t_embedder
            .forward(t, self.config.frequency_embedding_size);
        let mod_signal = self.ada_ln_modulation.forward(silu(t_emb));

        for block in &self.blocks {
            h = block.forward(h, mod_signal.clone(), cond.clone(), self.config.resolution);
        }

        let h = layer_norm_no_affine(h, LAYER_NORM_EPS);
        let h = linear_forward_token_chunked(
            &self.out_layer,
            h,
            sparse_flow_linear_chunk_tokens(tokens),
        );
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
        let sample_start = Instant::now();
        let progress_interval = runtime_sample_progress_interval(sample_cfg.steps);
        let stage_label = if concat_cond.is_some() {
            "flow.sample_with_trace.concat"
        } else {
            "flow.sample_with_trace.sparse"
        };
        eprintln!(
            "burn_trellis: {stage_label} begin (steps={}, resolution={}, state_channels={}, concat_channels={})",
            sample_cfg.steps, self.config.resolution, state_channels, concat_channels
        );
        for (step_idx, (t, t_prev)) in t_pairs.into_iter().enumerate() {
            let step_start = Instant::now();
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
            let step_done = step_idx + 1;
            if step_done % progress_interval == 0 || step_done == sample_cfg.steps {
                eprintln!(
                    "burn_trellis: {stage_label} step {step_done}/{} complete ({:.2} ms, elapsed={:.2} ms)",
                    sample_cfg.steps,
                    step_start.elapsed().as_secs_f64() * 1000.0,
                    sample_start.elapsed().as_secs_f64() * 1000.0
                );
            }
        }
        eprintln!(
            "burn_trellis: {stage_label} complete ({:.2} ms)",
            sample_start.elapsed().as_secs_f64() * 1000.0
        );

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
        let sample_start = Instant::now();
        let progress_interval = runtime_sample_progress_interval(sample_cfg.steps);
        let stage_label = if concat_cond.is_some() {
            "flow.sample_rows_with_trace.tex_slat"
        } else {
            "flow.sample_rows_with_trace.shape_slat"
        };
        eprintln!(
            "burn_trellis: {stage_label} begin (steps={}, resolution={}, rows={}, row_channels={})",
            sample_cfg.steps,
            self.config.resolution,
            dense_indices.len(),
            row_channels.min(state_channels)
        );
        for (step_idx, (t, t_prev)) in t_pairs.into_iter().enumerate() {
            let step_start = Instant::now();
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
            let step_done = step_idx + 1;
            if step_done % progress_interval == 0 || step_done == sample_cfg.steps {
                eprintln!(
                    "burn_trellis: {stage_label} step {step_done}/{} complete ({:.2} ms, elapsed={:.2} ms)",
                    sample_cfg.steps,
                    step_start.elapsed().as_secs_f64() * 1000.0,
                    sample_start.elapsed().as_secs_f64() * 1000.0
                );
            }
        }
        eprintln!(
            "burn_trellis: {stage_label} complete ({:.2} ms)",
            sample_start.elapsed().as_secs_f64() * 1000.0
        );

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

        let [batch, _, _, _, _] = x_t.dims();
        let x0_pos = pred_to_xstart_tensor(x_t.clone(), timestep, pos, sigma_min);
        let x0_cfg = pred_to_xstart_tensor(x_t.clone(), timestep, pred, sigma_min);
        let std_pos = tensor_std_tensor(x0_pos);
        let std_cfg = tensor_std_tensor(x0_cfg.clone()).add_scalar(1.0e-12);
        let scale = std_pos.div(std_cfg).reshape([batch, 1, 1, 1, 1]);
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
        prefer_wgpu: bool,
        resolution_override: Option<usize>,
    ) -> Result<Self, String> {
        #[cfg(feature = "runtime-model-wgpu")]
        if prefer_wgpu {
            let runtime = SparseStructureFlowRuntimeImpl::<WgpuRuntimeBackend>::load_from_stem(
                weights_root,
                image_large_root,
                model_stem,
                resolution_override,
            )
            .map_err(|err| {
                format!(
                    "burn_trellis: failed to load sparse flow runtime on wgpu ({err}); refusing cpu fallback for model '{model_stem}'"
                )
            })?;
            let cfg = runtime.config();
            let tokens = cfg
                .resolution
                .saturating_mul(cfg.resolution)
                .saturating_mul(cfg.resolution);
            if sparse_flow_wgpu_may_overflow(cfg) && !sparse_flow_chunked_forward_enabled(tokens) {
                return Err(format!(
                    "burn_trellis: sparse flow wgpu estimated peak exceeds safe budget for model '{}' (resolution={}, model_channels={}); refusing cpu fallback",
                    model_stem, cfg.resolution, cfg.model_channels
                ));
            }
            if sparse_flow_wgpu_may_overflow(cfg) {
                eprintln!(
                    "burn_trellis: sparse flow wgpu keeping model '{}' on device with chunked-forward path (resolution={}, model_channels={}).",
                    model_stem, cfg.resolution, cfg.model_channels
                );
            }
            return Ok(Self::Wgpu(runtime));
        }

        #[cfg(not(feature = "runtime-model-wgpu"))]
        if prefer_wgpu {
            return Err(format!(
                "burn_trellis: sparse flow runtime requested wgpu for model '{}' but crate was built without runtime-model-wgpu",
                model_stem
            ));
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
    let features = c.saturating_mul(x).saturating_mul(y).saturating_mul(z);
    let flat = tensor.reshape([b, features.max(1)]);
    let mean = flat.clone().mean_dim(1).reshape([b, 1]);
    let centered = flat.sub(mean);
    let denom = features.saturating_sub(1).max(1) as f32;
    centered
        .powf_scalar(2.0)
        .sum_dim(1)
        .reshape([b])
        .div_scalar(denom)
        .sqrt()
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

#[cfg(feature = "runtime-model-wgpu")]
fn sparse_flow_wgpu_max_peak_bytes() -> usize {
    3 * 1024 * 1024 * 1024
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
    false
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

fn env_chunk_tokens(key: &str, default: usize, max_chunk: usize, tokens: usize) -> usize {
    let _ = key;
    let requested = default.max(1);
    requested.min(max_chunk.max(1)).min(tokens.max(1))
}

fn sparse_flow_chunked_forward_enabled(tokens: usize) -> bool {
    attention_prefers_stream() && tokens >= 2_048
}

fn sparse_flow_stream_reuse_qkv_enabled(tokens: usize, channels: usize) -> bool {
    if !sparse_flow_chunked_forward_enabled(tokens) {
        return false;
    }
    #[cfg(feature = "runtime-model-wgpu")]
    {
        // Cap extra cache pressure from storing streamed Q chunks while still
        // eliminating the second QKV projection pass.
        let q_cache_bytes = tokens
            .checked_mul(channels)
            .and_then(|value| value.checked_mul(core::mem::size_of::<f32>()))
            .unwrap_or(usize::MAX);
        let budget = sparse_flow_wgpu_max_peak_bytes().saturating_mul(2);
        q_cache_bytes <= budget
    }
    #[cfg(not(feature = "runtime-model-wgpu"))]
    {
        let _ = channels;
        false
    }
}

fn sparse_flow_linear_chunk_tokens(tokens: usize) -> usize {
    let default = if sparse_flow_chunked_forward_enabled(tokens) {
        8_192
    } else {
        tokens.max(1)
    };
    env_chunk_tokens("TRELLIS2_SPARSE_FLOW_LINEAR_CHUNK", default, 32_768, tokens)
}

fn sparse_flow_attn_logits_budget_bytes() -> usize {
    #[cfg(feature = "runtime-model-wgpu")]
    {
        // Keep attention logits under conservative WGPU per-buffer limits.
        128 * 1024 * 1024
    }
    #[cfg(not(feature = "runtime-model-wgpu"))]
    {
        2_147_483_648
    }
}

fn sparse_flow_stream_chunk_plan(
    batch: usize,
    heads: usize,
    tokens: usize,
    query_chunk_tokens: usize,
    kv_chunk_tokens: usize,
    reuse_qkv: bool,
    logits_budget: usize,
) -> (usize, usize) {
    let tokens = tokens.max(1);
    let mut query_chunk_tokens = query_chunk_tokens.max(1).min(tokens);
    let mut kv_chunk_tokens = kv_chunk_tokens.max(1).min(tokens);
    let bytes_per_logit = batch
        .saturating_mul(heads)
        .saturating_mul(core::mem::size_of::<f32>())
        .max(1);
    let budget_logits = logits_budget / bytes_per_logit;

    if reuse_qkv {
        let max_square = integer_sqrt(budget_logits).max(1).min(tokens);
        let chunk_tokens = kv_chunk_tokens.min(max_square).max(1);
        (chunk_tokens, chunk_tokens)
    } else {
        let max_query = budget_logits
            .checked_div(kv_chunk_tokens.max(1))
            .unwrap_or(1)
            .max(1);
        query_chunk_tokens = query_chunk_tokens.min(max_query).max(1);
        let max_kv = budget_logits
            .checked_div(query_chunk_tokens.max(1))
            .unwrap_or(1)
            .max(1);
        kv_chunk_tokens = kv_chunk_tokens.min(max_kv).max(1);
        (query_chunk_tokens, kv_chunk_tokens)
    }
}

#[cfg(test)]
fn sparse_flow_attention_logits_within_budget(
    batch: usize,
    heads: usize,
    query_tokens: usize,
    key_tokens: usize,
    logits_budget: usize,
) -> bool {
    attention_logits_bytes(batch, heads, query_tokens, key_tokens) <= logits_budget
}

fn integer_sqrt(value: usize) -> usize {
    (value as f64).sqrt().floor() as usize
}

fn sparse_flow_mlp_chunk_tokens(tokens: usize) -> usize {
    let default = if sparse_flow_chunked_forward_enabled(tokens) {
        2_048
    } else {
        tokens.max(1)
    };
    env_chunk_tokens("TRELLIS2_SPARSE_FLOW_MLP_CHUNK", default, 16_384, tokens)
}

fn sparse_flow_self_attn_query_chunk_tokens(tokens: usize) -> usize {
    let default = if sparse_flow_chunked_forward_enabled(tokens) {
        2_048
    } else {
        tokens.max(1)
    };
    env_chunk_tokens(
        "TRELLIS2_SPARSE_FLOW_ATTN_QUERY_CHUNK",
        default,
        16_384,
        tokens,
    )
}

fn sparse_flow_self_attn_kv_chunk_tokens(tokens: usize) -> usize {
    let default = if sparse_flow_chunked_forward_enabled(tokens) {
        8_192
    } else {
        tokens.max(1)
    };
    env_chunk_tokens(
        "TRELLIS2_SPARSE_FLOW_ATTN_KV_CHUNK",
        default,
        32_768,
        tokens,
    )
}

fn linear_forward_token_chunked<B: Backend>(
    linear: &nn::Linear<B>,
    x: Tensor<B, 3>,
    chunk_tokens: usize,
) -> Tensor<B, 3> {
    let [batch, tokens, channels] = x.dims();
    if chunk_tokens >= tokens {
        return linear.forward(x);
    }

    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < tokens {
        let end = (start + chunk_tokens).min(tokens);
        let x_chunk = x.clone().slice([0..batch, start..end, 0..channels]);
        chunks.push(linear.forward(x_chunk));
        start = end;
    }
    Tensor::cat(chunks, 1)
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
    let work = query_tokens.saturating_mul(key_tokens);
    if (attention_prefers_stream() && work >= 64usize.saturating_mul(64))
        || work >= 512usize.saturating_mul(512)
    {
        AttentionImpl::Stream
    } else {
        AttentionImpl::Dense
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
            acc = acc.mul(alpha.clone()).add(chunk_acc.mul(beta.clone()));
            denom = alpha
                .mul(denom)
                .add(beta.mul(chunk_denom))
                .add_scalar(1.0e-12);

            max_scores = max_new;
            k_start = k_end;
        }

        outputs.push(acc.div(denom.add_scalar(1.0e-12)));
        q_start = q_end;
    }

    Tensor::cat(outputs, 2)
}

fn scaled_dot_product_attention_stream_chunked_keys<B: Backend>(
    q: Tensor<B, 4>,
    k_chunks: &[Tensor<B, 4>],
    v_chunks: &[Tensor<B, 4>],
    head_dim: usize,
) -> Tensor<B, 4> {
    if k_chunks.is_empty() || v_chunks.is_empty() {
        let [batch, heads, query_tokens, _] = q.dims();
        return Tensor::<B, 4>::zeros([batch, heads, query_tokens, head_dim], &q.device());
    }
    if k_chunks.len() != v_chunks.len() {
        panic!(
            "stream attention chunk mismatch: k_chunks={} v_chunks={}",
            k_chunks.len(),
            v_chunks.len()
        );
    }

    let scale = 1.0 / (head_dim as f32).sqrt();
    let [batch, heads, query_tokens, _] = q.dims();
    let [_, _, _, value_dim] = v_chunks[0].dims();

    let first_logits =
        matmul_4d_via_3d(q.clone(), k_chunks[0].clone().swap_dims(2, 3)).mul_scalar(scale);
    let mut max_scores = first_logits.clone().max_dim(3);
    let first_probs = first_logits.sub(max_scores.clone()).exp();
    let mut denom = first_probs.clone().sum_dim(3);
    let mut acc = matmul_4d_via_3d(first_probs, v_chunks[0].clone());

    for idx in 1..k_chunks.len() {
        let k_chunk = k_chunks[idx].clone();
        let v_chunk = v_chunks[idx].clone();
        let [k_batch, k_heads, _, k_head_dim] = k_chunk.dims();
        let [v_batch, v_heads, _, v_value_dim] = v_chunk.dims();
        if k_batch != batch || k_heads != heads || k_head_dim != head_dim {
            panic!(
                "stream attention k chunk dims mismatch at idx={idx}: got=[{k_batch},{k_heads},*,{k_head_dim}] expected=[{batch},{heads},*,{head_dim}]"
            );
        }
        if v_batch != batch || v_heads != heads || v_value_dim != value_dim {
            panic!(
                "stream attention v chunk dims mismatch at idx={idx}: got=[{v_batch},{v_heads},*,{v_value_dim}] expected=[{batch},{heads},*,{value_dim}]"
            );
        }

        let logits = matmul_4d_via_3d(q.clone(), k_chunk.swap_dims(2, 3)).mul_scalar(scale);
        let chunk_max = logits.clone().max_dim(3);
        let probs = logits.sub(chunk_max.clone()).exp();
        let chunk_denom = probs.clone().sum_dim(3);
        let chunk_acc = matmul_4d_via_3d(probs, v_chunk);

        let max_new = max_scores.clone().max_pair(chunk_max.clone());
        let alpha = max_scores.clone().sub(max_new.clone()).exp();
        let beta = chunk_max.sub(max_new.clone()).exp();
        acc = acc.mul(alpha.clone()).add(chunk_acc.mul(beta.clone()));
        denom = alpha
            .mul(denom)
            .add(beta.mul(chunk_denom))
            .add_scalar(1.0e-12);

        max_scores = max_new;
    }

    acc = acc.div(denom.add_scalar(1.0e-12));

    let [out_batch, out_heads, out_query_tokens, _] = acc.dims();
    if out_batch != batch || out_heads != heads || out_query_tokens != query_tokens {
        panic!(
            "stream attention chunked output dims mismatch: got=[{out_batch},{out_heads},{out_query_tokens},*] expected=[{batch},{heads},{query_tokens},*]"
        );
    }
    acc
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
    if attention_prefers_stream() {
        512 * 1024 * 1024
    } else {
        usize::MAX
    }
}

fn attention_query_chunk(tokens: usize, default_chunk: usize) -> usize {
    let max_chunk = if attention_prefers_stream() {
        256
    } else {
        usize::MAX
    };
    default_chunk.min(max_chunk).min(tokens.max(1))
}

fn attention_key_chunk(tokens: usize) -> usize {
    let max_chunk = if attention_prefers_stream() {
        512
    } else {
        usize::MAX
    };
    128usize.min(max_chunk).min(tokens.max(1))
}

fn apply_rope<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    resolution: usize,
    head_dim: usize,
    rope_freq: [f32; 2],
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    (
        apply_rope_single(q, resolution, head_dim, rope_freq, 0),
        apply_rope_single(k, resolution, head_dim, rope_freq, 0),
    )
}

fn apply_rope_single<B: Backend>(
    x: Tensor<B, 4>,
    resolution: usize,
    head_dim: usize,
    rope_freq: [f32; 2],
    token_start: usize,
) -> Tensor<B, 4> {
    let [_, tokens, _, _] = x.dims();
    let pairs = head_dim / 2;
    if pairs == 0 || tokens == 0 {
        return x;
    }
    let device = x.device();
    let (rope_cos, rope_sin) =
        rope_cos_sin_range(resolution, token_start, tokens, pairs, rope_freq);
    let cos = Tensor::<B, 1>::from_floats(rope_cos.as_slice(), &device)
        .reshape([tokens, pairs])
        .reshape([1, tokens, 1, pairs]);
    let sin = Tensor::<B, 1>::from_floats(rope_sin.as_slice(), &device)
        .reshape([tokens, pairs])
        .reshape([1, tokens, 1, pairs]);
    rotate_pairs(x, cos, sin)
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

fn rope_cos_sin_range(
    resolution: usize,
    token_start: usize,
    tokens: usize,
    pairs: usize,
    rope_freq: [f32; 2],
) -> (Vec<f32>, Vec<f32>) {
    let mut cos = vec![1.0f32; tokens * pairs];
    let mut sin = vec![0.0f32; tokens * pairs];
    if resolution == 0 || tokens == 0 || pairs == 0 {
        return (cos, sin);
    }

    let freq_dim = (pairs / 3).max(1);
    let mut freqs = Vec::with_capacity(freq_dim);
    for idx in 0..freq_dim {
        let exp = idx as f32 / freq_dim as f32;
        freqs.push(rope_freq[0] / rope_freq[1].powf(exp));
    }

    let resolution_sq = resolution.saturating_mul(resolution).max(1);
    let max_tokens = resolution_sq.saturating_mul(resolution);
    for local_token in 0..tokens {
        let token = token_start.saturating_add(local_token);
        if token >= max_tokens {
            break;
        }
        let x = token / resolution_sq;
        let yz = token % resolution_sq;
        let y = yz / resolution;
        let z = yz % resolution;
        let coords = [x as f32, y as f32, z as f32];
        for (dim, coord) in coords.iter().enumerate() {
            for (freq_idx, freq) in freqs.iter().enumerate() {
                let pair = dim * freq_dim + freq_idx;
                if pair >= pairs {
                    continue;
                }
                let phase = *coord * *freq;
                let idx = local_token * pairs + pair;
                cos[idx] = phase.cos();
                sin[idx] = phase.sin();
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
        // Canonical sparse-flow checkpoints are stored as raw safetensors blobs in burnpacks.
        // Prefer blob/parts loading first, then fall back to legacy module-layout burnpacks.
        match load_blob_bytes_from_burnpack_or_parts(path, load_burnpack_blob_bytes) {
            Ok(blob_bytes) => {
                let mut safetensor_store = build_safetensor_store_from_bytes(blob_bytes)?;
                model
                    .load_from(&mut safetensor_store)
                    .map(|_| ())
                    .map_err(|safetensor_err| {
                        format!(
                            "failed to load sparse flow burnpack '{}' as safetensors blob: {safetensor_err}",
                            path.display()
                        )
                    })
            }
            Err(blob_err) => {
                if !path.exists() {
                    return Err(format!(
                        "failed to load sparse flow burnpack '{}' as blob/parts ({blob_err})",
                        path.display()
                    ));
                }

                let mut store = BurnpackStore::from_file(path).validate(true);
                model.load_from(&mut store).map(|_| ()).map_err(|module_err| {
                    format!(
                        "failed to load sparse flow burnpack '{}' as blob/parts ({blob_err}) and legacy module layout ({module_err})",
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
        .filter(|path| candidate_exists_or_has_parts(path))
        .collect::<Vec<_>>()
}

fn prefer_f16_burnpack() -> bool {
    // Correctness-first default: keep sparse-flow aligned with canonical bf16
    // checkpoint behavior before opting into lossy f16 burnpacks.
    false
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
        BinaryBlob, BlobMetadata, CpuRuntimeBackend, SelfAttention, SparseStructureFlowConfig,
        SparseStructureFlowModel, SparseStructureFlowRuntime, SparseStructureFlowRuntimeImpl,
        host_transfer_stats, metadata_path, reset_host_transfer_stats,
        resolve_model_weight_candidates, scaled_dot_product_attention_dense,
        scaled_dot_product_attention_stream, sparse_flow_attention_logits_within_budget,
        sparse_flow_stream_chunk_plan,
    };

    static HOST_STATS_LOCK: Mutex<()> = Mutex::new(());
    static ENV_LOCK: Mutex<()> = Mutex::new(());

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

    #[test]
    fn runtime_loads_blob_burnpack_from_parts_manifest_when_base_file_missing() {
        type BlobBackend = burn::backend::NdArray<f32, u8>;
        type TestBackend = burn::backend::NdArray<f32>;

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock drift")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("burn_trellis_sparse_flow_parts_{unique}"));
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

        let part_path = ckpts.join("flow_model.bpk.part-00000.bpk");
        let part_meta_path = metadata_path(&part_path);
        std::fs::rename(&burnpack_path, &part_path).expect("move burnpack into part");
        std::fs::rename(metadata_path(&burnpack_path), &part_meta_path)
            .expect("move part metadata");
        let manifest_path = ckpts.join("flow_model.bpk.parts.json");
        std::fs::write(
            &manifest_path,
            format!(
                "{{\n  \"version\": 1,\n  \"source_file\": \"flow_model.bpk\",\n  \"source_modified_unix_ms\": 0,\n  \"total_bytes\": {},\n  \"max_part_bytes\": {},\n  \"parts\": [{{\"path\": \"{}\", \"bytes\": {}, \"sha256\": \"\", \"tensors\": 1}}]\n}}",
                source_bytes.len(),
                source_bytes.len(),
                part_path
                    .file_name()
                    .and_then(|v| v.to_str())
                    .expect("part file name"),
                source_bytes.len()
            ),
        )
        .expect("write parts manifest");
        std::fs::remove_file(&source_path).expect("remove source safetensors");

        let runtime = SparseStructureFlowRuntime::load_from_stem(
            root.as_path(),
            None,
            "ckpts/flow_model",
            false,
            None,
        )
        .expect("runtime should load from parts manifest");
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

        let candidates = resolve_model_weight_candidates("ckpts/flow_model", root.as_path(), None);
        assert_eq!(
            candidates.first(),
            Some(&ckpts.join("flow_model.bpk")),
            "parts manifest path should be treated as a valid burnpack candidate"
        );

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

    fn sample_std(values: &[f32]) -> f32 {
        if values.len() < 2 {
            return 0.0;
        }
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let var_sum = values
            .iter()
            .map(|value| {
                let diff = *value - mean;
                diff * diff
            })
            .sum::<f32>();
        (var_sum / (values.len() - 1) as f32).sqrt()
    }

    #[test]
    fn tensor_std_matches_batchwise_unbiased_reference() {
        let device = <CpuRuntimeBackend as Backend>::Device::default();
        let values = [
            0.0f32, 1.0, 2.0, 3.0, // batch 0
            2.0, 2.0, 2.0, 2.0, // batch 1
        ];
        let tensor = Tensor::<CpuRuntimeBackend, 1>::from_floats(values.as_slice(), &device)
            .reshape([2, 1, 1, 1, 4]);
        let std = super::tensor_std_tensor(tensor)
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("std tensor should be readable");
        assert_eq!(std.len(), 2, "expected one std value per batch item");

        let expected0 = sample_std(&values[0..4]);
        let expected1 = sample_std(&values[4..8]);
        assert!(
            (std[0] - expected0).abs() <= 1.0e-6,
            "batch 0 std mismatch: got={} expected={}",
            std[0],
            expected0
        );
        assert!(
            (std[1] - expected1).abs() <= 1.0e-6,
            "batch 1 std mismatch: got={} expected={}",
            std[1],
            expected1
        );
    }

    #[test]
    fn attention_stream_matches_dense_reference() {
        let _guard = ENV_LOCK.lock().expect("env lock");
        unsafe {
            std::env::set_var("TRELLIS2_ATTN_QUERY_CHUNK", "8");
            std::env::set_var("TRELLIS2_ATTN_QUERY_CHUNK_MAX", "8");
            std::env::set_var("TRELLIS2_ATTN_KEY_CHUNK", "7");
            std::env::set_var("TRELLIS2_ATTN_KEY_CHUNK_MAX", "7");
        }
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

        unsafe {
            std::env::remove_var("TRELLIS2_ATTN_QUERY_CHUNK");
            std::env::remove_var("TRELLIS2_ATTN_QUERY_CHUNK_MAX");
            std::env::remove_var("TRELLIS2_ATTN_KEY_CHUNK");
            std::env::remove_var("TRELLIS2_ATTN_KEY_CHUNK_MAX");
        }
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
    fn self_attention_chunked_matches_dense_reference() {
        let _guard = ENV_LOCK.lock().expect("env lock");
        let device = <CpuRuntimeBackend as Backend>::Device::default();
        let channels = 32usize;
        let heads = 4usize;
        let tokens = 32usize;
        let resolution = 4usize;
        let attention = SelfAttention::<CpuRuntimeBackend>::new(
            &device,
            channels,
            heads,
            true,
            [1.0, 10_000.0],
            true,
        );

        let mut values = Vec::with_capacity(tokens.saturating_mul(channels));
        for idx in 0..values.capacity() {
            let x = idx as f32 * 0.011 + 0.37;
            values.push(x.sin() * 0.5 + x.cos() * 0.5);
        }
        let input = Tensor::<CpuRuntimeBackend, 1>::from_floats(values.as_slice(), &device)
            .reshape([1, tokens, channels]);

        unsafe {
            std::env::set_var("TRELLIS2_ATTN_BACKEND", "stream");
            std::env::set_var("TRELLIS2_SPARSE_FLOW_CHUNKED_FORWARD", "0");
        }
        let dense = attention.forward(input.clone(), resolution);

        unsafe {
            std::env::set_var("TRELLIS2_SPARSE_FLOW_CHUNKED_FORWARD", "1");
            std::env::set_var("TRELLIS2_SPARSE_FLOW_ATTN_QUERY_CHUNK", "8");
            std::env::set_var("TRELLIS2_SPARSE_FLOW_ATTN_KV_CHUNK", "8");
        }
        let chunked = attention.forward(input, resolution);

        let dense = dense
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("dense vec");
        let chunked = chunked
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("chunked vec");
        assert_eq!(dense.len(), chunked.len());
        let max_abs = dense
            .iter()
            .zip(chunked.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs <= 1.0e-3,
            "chunked self-attention drift too high: max_abs={max_abs:.6e}"
        );

        unsafe {
            std::env::remove_var("TRELLIS2_ATTN_BACKEND");
            std::env::remove_var("TRELLIS2_SPARSE_FLOW_CHUNKED_FORWARD");
            std::env::remove_var("TRELLIS2_SPARSE_FLOW_ATTN_QUERY_CHUNK");
            std::env::remove_var("TRELLIS2_SPARSE_FLOW_ATTN_KV_CHUNK");
        }
    }

    #[test]
    fn sparse_flow_chunk_plan_reuse_qkv_respects_logits_budget() {
        let logits_budget = 128 * 1024 * 1024;
        let (query_chunk_tokens, kv_chunk_tokens) =
            sparse_flow_stream_chunk_plan(1, 16, 6_144, 8_192, 8_192, true, logits_budget);
        assert_eq!(
            query_chunk_tokens, kv_chunk_tokens,
            "reuse-qkv path must keep query/kv chunks aligned"
        );
        assert!(
            sparse_flow_attention_logits_within_budget(
                1,
                16,
                query_chunk_tokens,
                kv_chunk_tokens,
                logits_budget
            ),
            "chunk plan exceeded logits budget in reuse-qkv mode: q={} kv={} budget={}",
            query_chunk_tokens,
            kv_chunk_tokens,
            logits_budget
        );
    }

    #[test]
    fn sparse_flow_chunk_plan_non_reuse_respects_logits_budget() {
        let logits_budget = 128 * 1024 * 1024;
        let (query_chunk_tokens, kv_chunk_tokens) =
            sparse_flow_stream_chunk_plan(1, 16, 8_192, 2_048, 8_192, false, logits_budget);
        assert!(
            sparse_flow_attention_logits_within_budget(
                1,
                16,
                query_chunk_tokens,
                kv_chunk_tokens,
                logits_budget
            ),
            "chunk plan exceeded logits budget in non-reuse mode: q={} kv={} budget={}",
            query_chunk_tokens,
            kv_chunk_tokens,
            logits_budget
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

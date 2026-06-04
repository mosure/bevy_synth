use burn::{
    module::Param,
    nn,
    prelude::*,
    tensor::{
        FloatDType,
        activation::{sigmoid, softmax},
        module::attention as module_attention,
        ops::AttentionModuleOptions,
    },
};

const RMS_NORM_EPS: f32 = 1.0e-12;
const ATTENTION_SCORE_ELEMS_CHUNK_THRESHOLD: usize = 32 * 1024 * 1024;
const ATTENTION_QUERY_CHUNK_TOKENS: usize = 128;

pub fn silu<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    x.clone() * sigmoid(x)
}

pub fn gelu<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    nn::Gelu::new_approximate().forward(x)
}

#[derive(Module, Debug)]
pub struct MultiHeadRmsNorm<B: Backend> {
    pub gamma: Param<Tensor<B, 2>>,
    scale: f32,
}

impl<B: Backend> MultiHeadRmsNorm<B> {
    pub fn new(device: &B::Device, heads: usize, head_dim: usize) -> Self {
        Self {
            gamma: nn::Initializer::Ones.init([heads, head_dim], device),
            scale: (head_dim as f32).sqrt(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch, tokens, heads, head_dim] = x.dims();
        let dtype: FloatDType = x.dtype().into();
        let acc_dtype = accumulation_dtype(dtype);
        let x_acc = cast_low_precision_to_f32(x, dtype);
        let norm = x_acc
            .clone()
            .powf_scalar(2.0)
            .sum_dim(3)
            .add_scalar(RMS_NORM_EPS)
            .sqrt();
        let gamma = self
            .gamma
            .val()
            .cast(acc_dtype)
            .reshape([1, 1, heads, head_dim]);
        let out = x_acc
            .mul(norm.recip())
            .mul_scalar(self.scale)
            .mul(gamma.expand([batch as i64, tokens as i64, -1, -1]));
        cast_from_f32_accum(out, dtype)
    }
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

fn accumulation_dtype(dtype: FloatDType) -> FloatDType {
    if matches!(dtype, FloatDType::F16 | FloatDType::BF16) {
        FloatDType::F32
    } else {
        dtype
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

#[derive(Module, Debug)]
pub struct FeedForwardNet<B: Backend> {
    pub mlp_0: nn::Linear<B>,
    pub mlp_2: nn::Linear<B>,
}

impl<B: Backend> FeedForwardNet<B> {
    pub fn new(device: &B::Device, channels: usize, mlp_ratio: f32) -> Self {
        let hidden = ((channels as f32) * mlp_ratio).round().max(1.0) as usize;
        Self {
            mlp_0: nn::LinearConfig::new(channels, hidden)
                .with_bias(true)
                .init(device),
            mlp_2: nn::LinearConfig::new(hidden, channels)
                .with_bias(true)
                .init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.mlp_2.forward(gelu(self.mlp_0.forward(x)))
    }
}

#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    layers: Vec<nn::Linear<B>>,
}

impl<B: Backend> Mlp<B> {
    pub fn new(
        device: &B::Device,
        channels: usize,
        inner_channels: usize,
        channels_out: usize,
        layer_count: usize,
    ) -> Self {
        let mut layers = Vec::with_capacity(layer_count.max(1));
        for index in 0..layer_count {
            let input = if index == 0 { channels } else { inner_channels };
            let output = if index + 1 == layer_count {
                channels_out
            } else {
                inner_channels
            };
            layers.push(
                nn::LinearConfig::new(input, output)
                    .with_bias(true)
                    .init(device),
            );
        }
        Self { layers }
    }

    pub fn forward(&self, mut x: Tensor<B, 3>) -> Tensor<B, 3> {
        for (index, layer) in self.layers.iter().enumerate() {
            x = layer.forward(x);
            if index + 1 != self.layers.len() {
                x = gelu(x);
            }
        }
        x
    }
}

#[derive(Module, Debug)]
pub struct SinusoidalEmbedder<B: Backend> {
    pub mlp_0: nn::Linear<B>,
    pub mlp_2: nn::Linear<B>,
    frequency_embedding_size: usize,
    max_period: f32,
    multiply_2pi: bool,
}

impl<B: Backend> SinusoidalEmbedder<B> {
    pub fn new(
        device: &B::Device,
        hidden_size: usize,
        frequency_embedding_size: usize,
        max_period: f32,
        multiply_2pi: bool,
    ) -> Self {
        Self {
            mlp_0: nn::LinearConfig::new(frequency_embedding_size, hidden_size)
                .with_bias(true)
                .init(device),
            mlp_2: nn::LinearConfig::new(hidden_size, hidden_size)
                .with_bias(true)
                .init(device),
            frequency_embedding_size,
            max_period,
            multiply_2pi,
        }
    }

    pub fn forward(&self, t: Tensor<B, 1>) -> Tensor<B, 2> {
        let emb = sinusoidal_embedding(
            t,
            self.frequency_embedding_size,
            self.max_period,
            self.multiply_2pi,
        );
        self.mlp_2.forward(silu(self.mlp_0.forward(emb)))
    }
}

pub fn sinusoidal_embedding<B: Backend>(
    t: Tensor<B, 1>,
    dim: usize,
    max_period: f32,
    multiply_2pi: bool,
) -> Tensor<B, 2> {
    let [batch] = t.dims();
    let half = dim / 2;
    let device = t.device();
    let dtype: FloatDType = t.dtype().into();
    let mut freqs = Vec::with_capacity(half);
    for index in 0..half {
        freqs.push((-max_period.ln() * index as f32 / half as f32).exp());
    }
    let freqs = Tensor::<B, 1>::from_floats(freqs.as_slice(), &device).cast(dtype);
    let mut args = t.unsqueeze_dim(1) * freqs.unsqueeze_dim(0);
    if multiply_2pi {
        args = args.mul_scalar(core::f32::consts::TAU);
    }
    let mut out = Tensor::cat(vec![args.clone().cos(), args.sin()], 1);
    if dim % 2 == 1 {
        out = Tensor::cat(
            vec![out, Tensor::<B, 2>::zeros([batch, 1], &device).cast(dtype)],
            1,
        );
    }
    out
}

#[derive(Clone, Debug)]
pub struct PcdAbsolutePositionEmbedder {
    pub channels: usize,
    pub in_channels: usize,
    pub max_res: usize,
    pub linear_residual: bool,
}

impl PcdAbsolutePositionEmbedder {
    pub fn legacy(channels: usize) -> Self {
        Self {
            channels,
            in_channels: 3,
            max_res: 16,
            linear_residual: true,
        }
    }

    pub fn v2(channels: usize) -> Self {
        Self {
            channels,
            in_channels: 3,
            max_res: 10,
            linear_residual: false,
        }
    }

    pub fn forward_3d<B: Backend>(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, tokens, dims] = x.dims();
        self.forward_2d(x.reshape([batch * tokens, dims]))
            .reshape([batch, tokens, self.channels])
    }

    pub fn forward_2d<B: Backend>(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let [tokens, dims] = x.dims();
        let device = x.device();
        let dtype: FloatDType = x.dtype().into();
        let freq_dim = self.channels / self.in_channels / 2;
        let freqs = if self.linear_residual {
            legacy_freqs(freq_dim, self.max_res)
        } else {
            linspace_pow2(freq_dim, self.max_res)
        };
        let freqs = Tensor::<B, 1>::from_floats(freqs.as_slice(), &device).cast(dtype);
        let angle_scale = if self.linear_residual {
            core::f32::consts::TAU
        } else {
            core::f32::consts::PI
        };
        let scaled = x
            .unsqueeze_dim::<3>(2)
            .mul(freqs.reshape([1, 1, freq_dim]))
            .mul_scalar(angle_scale);
        let mut out = Tensor::cat(vec![scaled.clone().sin(), scaled.cos()], 2)
            .reshape([tokens, dims * freq_dim * 2]);
        if out.dims()[1] < self.channels {
            let pad =
                Tensor::<B, 2>::zeros([tokens, self.channels - out.dims()[1]], &device).cast(dtype);
            out = Tensor::cat(vec![out, pad], 1);
        }
        out
    }
}

fn legacy_freqs(freq_dim: usize, max_res: usize) -> Vec<f32> {
    let base = freq_dim.min(max_res);
    let mut freqs = Vec::with_capacity(freq_dim);
    for index in 0..base {
        freqs.push(2_f32.powi(index as i32));
    }
    let residual = freq_dim.saturating_sub(max_res);
    for index in 0..residual {
        freqs.push(2_f32.powf(index as f32 / residual.max(1) as f32 * max_res as f32));
    }
    freqs.truncate(freq_dim);
    freqs
}

fn linspace_pow2(freq_dim: usize, max_res: usize) -> Vec<f32> {
    if freq_dim <= 1 {
        return vec![1.0; freq_dim];
    }
    (0..freq_dim)
        .map(|index| 2_f32.powf(index as f32 / (freq_dim - 1) as f32 * max_res as f32))
        .collect()
}

#[derive(Module, Debug)]
pub struct RePo3dRotaryEmbedding<B: Backend> {
    pub norm: nn::LayerNorm<B>,
    pub gate_map: nn::Linear<B>,
    pub content_map: nn::Linear<B>,
    pub final_map: nn::Linear<B>,
    pub freqs_0: Param<Tensor<B, 1>>,
    pub freqs_1: Param<Tensor<B, 1>>,
    pub freqs_2: Param<Tensor<B, 1>>,
    num_heads: usize,
    dim_0: usize,
    dim_1: usize,
    dim_2: usize,
}

impl<B: Backend> RePo3dRotaryEmbedding<B> {
    pub fn new(
        device: &B::Device,
        model_channels: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Self {
        let repo_hidden_size = ((model_channels as f32) * 0.125).round().max(1.0) as usize;
        let dim_0 = 2 * (head_dim / 6);
        let dim_1 = 2 * (head_dim / 6);
        let dim_2 = head_dim - dim_0 - dim_1;
        let freqs_0 = repo_freqs::<B>(dim_0 / 2, device);
        let freqs_1 = repo_freqs::<B>(dim_1 / 2, device);
        let freqs_2 = repo_freqs::<B>(dim_2 / 2, device);
        Self {
            norm: nn::LayerNormConfig::new(model_channels)
                .with_epsilon(1.0e-6)
                .init(device),
            gate_map: nn::LinearConfig::new(model_channels, repo_hidden_size)
                .with_bias(false)
                .init(device),
            content_map: nn::LinearConfig::new(model_channels, repo_hidden_size)
                .with_bias(false)
                .init(device),
            final_map: nn::LinearConfig::new(repo_hidden_size, 3 * num_heads)
                .with_bias(false)
                .init(device),
            freqs_0,
            freqs_1,
            freqs_2,
            num_heads,
            dim_0,
            dim_1,
            dim_2,
        }
    }

    pub fn forward(&self, hidden_states: Tensor<B, 3>) -> RotaryAngles<B> {
        let h = self.norm.forward(hidden_states);
        let feat = silu(self.gate_map.forward(h.clone())) * self.content_map.forward(h);
        let out = self.final_map.forward(feat);
        let [batch, tokens, _] = out.dims();
        let delta = out.reshape([batch, tokens, self.num_heads, 3]);
        let d0 = delta
            .clone()
            .slice([0..batch, 0..tokens, 0..self.num_heads, 0..1]);
        let d1 = delta
            .clone()
            .slice([0..batch, 0..tokens, 0..self.num_heads, 1..2]);
        let d2 = delta.slice([0..batch, 0..tokens, 0..self.num_heads, 2..3]);
        let a0 = d0 * self.freqs_0.val().reshape([1, 1, 1, self.dim_0 / 2]);
        let a1 = d1 * self.freqs_1.val().reshape([1, 1, 1, self.dim_1 / 2]);
        let a2 = d2 * self.freqs_2.val().reshape([1, 1, 1, self.dim_2 / 2]);
        let angles = Tensor::cat(vec![a0, a1, a2], 3).mul_scalar(core::f32::consts::PI);
        RotaryAngles {
            cos: angles.clone().cos(),
            sin: angles.sin(),
        }
    }
}

fn repo_freqs<B: Backend>(freq_dim: usize, device: &B::Device) -> Param<Tensor<B, 1>> {
    let values = linspace_inclusive(1.0, 16.0, freq_dim);
    Param::from_tensor(Tensor::<B, 1>::from_floats(values.as_slice(), device))
}

fn linspace_inclusive(start: f32, end: f32, steps: usize) -> Vec<f32> {
    match steps {
        0 => Vec::new(),
        1 => vec![start],
        _ => (0..steps)
            .map(|index| start + (end - start) * index as f32 / (steps - 1) as f32)
            .collect(),
    }
}

#[derive(Clone, Debug)]
pub struct RotaryAngles<B: Backend> {
    pub cos: Tensor<B, 4>,
    pub sin: Tensor<B, 4>,
}

pub fn apply_rotary_emb<B: Backend>(
    hidden_states: Tensor<B, 4>,
    freqs: &RotaryAngles<B>,
) -> Tensor<B, 4> {
    let [batch, tokens, heads, head_dim] = hidden_states.dims();
    let pairs = head_dim / 2;
    let pair = hidden_states.reshape([batch, tokens, heads, pairs, 2]);
    let even = pair
        .clone()
        .slice([0..batch, 0..tokens, 0..heads, 0..pairs, 0..1]);
    let odd = pair.slice([0..batch, 0..tokens, 0..heads, 0..pairs, 1..2]);
    let cos = freqs.cos.clone().unsqueeze_dim::<5>(4);
    let sin = freqs.sin.clone().unsqueeze_dim::<5>(4);
    let out_even = even.clone() * cos.clone() - odd.clone() * sin.clone();
    let out_odd = even * sin + odd * cos;
    Tensor::cat(vec![out_even, out_odd], 4).reshape([batch, tokens, heads, head_dim])
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionKind {
    SelfAttention,
    CrossAttention,
}

#[derive(Module, Debug)]
pub struct MultiHeadAttention<B: Backend> {
    pub qkv: Option<nn::Linear<B>>,
    pub q: Option<nn::Linear<B>>,
    pub kv: Option<nn::Linear<B>>,
    pub out: nn::Linear<B>,
    pub q_norm: Option<MultiHeadRmsNorm<B>>,
    pub k_norm: Option<MultiHeadRmsNorm<B>>,
    kind: AttentionKind,
    num_heads: usize,
    head_dim: usize,
    channels: usize,
    context_channels: usize,
    use_rope: bool,
}

impl<B: Backend> MultiHeadAttention<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &B::Device,
        channels: usize,
        num_heads: usize,
        context_channels: Option<usize>,
        kind: AttentionKind,
        qkv_bias: bool,
        qk_rms_norm: bool,
        use_rope: bool,
    ) -> Self {
        let head_dim = channels / num_heads;
        let context_channels = context_channels.unwrap_or(channels);
        let (qkv, q, kv) = match kind {
            AttentionKind::SelfAttention => (
                Some(
                    nn::LinearConfig::new(channels, channels * 3)
                        .with_bias(qkv_bias)
                        .init(device),
                ),
                None,
                None,
            ),
            AttentionKind::CrossAttention => (
                None,
                Some(
                    nn::LinearConfig::new(channels, channels)
                        .with_bias(qkv_bias)
                        .init(device),
                ),
                Some(
                    nn::LinearConfig::new(context_channels, channels * 2)
                        .with_bias(qkv_bias)
                        .init(device),
                ),
            ),
        };
        Self {
            qkv,
            q,
            kv,
            out: nn::LinearConfig::new(channels, channels)
                .with_bias(true)
                .init(device),
            q_norm: qk_rms_norm.then(|| MultiHeadRmsNorm::new(device, num_heads, head_dim)),
            k_norm: qk_rms_norm.then(|| MultiHeadRmsNorm::new(device, num_heads, head_dim)),
            kind,
            num_heads,
            head_dim,
            channels,
            context_channels,
            use_rope,
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        context: Option<Tensor<B, 3>>,
        rope_emb: Option<&RotaryAngles<B>>,
    ) -> Tensor<B, 3> {
        let [batch, tokens, channels] = x.dims();
        let (mut q, mut k, v) = match self.kind {
            AttentionKind::SelfAttention => {
                let qkv = self
                    .qkv
                    .as_ref()
                    .expect("self attention qkv missing")
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
                (q, k, v)
            }
            AttentionKind::CrossAttention => {
                let context = context.expect("context required for cross attention");
                let context_tokens = context.dims()[1];
                let q = self
                    .q
                    .as_ref()
                    .expect("cross attention q missing")
                    .forward(x)
                    .reshape([batch, tokens, self.num_heads, self.head_dim]);
                let kv = self
                    .kv
                    .as_ref()
                    .expect("cross attention kv missing")
                    .forward(context)
                    .reshape([batch, context_tokens, 2, self.num_heads, self.head_dim]);
                let k = kv
                    .clone()
                    .slice([
                        0..batch,
                        0..context_tokens,
                        0..1,
                        0..self.num_heads,
                        0..self.head_dim,
                    ])
                    .reshape([batch, context_tokens, self.num_heads, self.head_dim]);
                let v = kv
                    .slice([
                        0..batch,
                        0..context_tokens,
                        1..2,
                        0..self.num_heads,
                        0..self.head_dim,
                    ])
                    .reshape([batch, context_tokens, self.num_heads, self.head_dim]);
                (q, k, v)
            }
        };
        if self.use_rope
            && let Some(rope) = rope_emb
        {
            q = apply_rotary_emb(q, rope);
            k = apply_rotary_emb(k, rope);
        }
        if let Some(norm) = &self.q_norm {
            q = norm.forward(q);
        }
        if let Some(norm) = &self.k_norm {
            k = norm.forward(k);
        }
        let out = scaled_dot_product_attention(q, k, v, self.head_dim);
        self.out.forward(out.reshape([batch, tokens, channels]))
    }
}

pub fn scaled_dot_product_attention<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    head_dim: usize,
) -> Tensor<B, 4> {
    let [batch, query_tokens, heads, _] = q.dims();
    let key_tokens = k.dims()[1];
    let score_elems = batch
        .saturating_mul(heads)
        .saturating_mul(query_tokens)
        .saturating_mul(key_tokens);

    if score_elems > ATTENTION_SCORE_ELEMS_CHUNK_THRESHOLD
        && query_tokens > ATTENTION_QUERY_CHUNK_TOKENS
    {
        return scaled_dot_product_attention_chunked(
            q,
            k,
            v,
            head_dim,
            ATTENTION_QUERY_CHUNK_TOKENS,
        );
    }

    scaled_dot_product_attention_dense(q, k, v, head_dim)
}

fn scaled_dot_product_attention_dense<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    head_dim: usize,
) -> Tensor<B, 4> {
    let q = q.permute([0, 2, 1, 3]);
    let k = k.permute([0, 2, 1, 3]);
    let v = v.permute([0, 2, 1, 3]);
    // Keep the canonical PyTorch scale explicit. CubeCL routes explicit-scale
    // attention through the generic fallback, so large TripoSplat shapes must be
    // split before reaching this helper to keep fallback score tensors bounded.
    let out = module_attention(
        q,
        k,
        v,
        None,
        None,
        AttentionModuleOptions {
            scale: Some((head_dim as f64).powf(-0.5)),
            ..Default::default()
        },
    );
    out.permute([0, 2, 1, 3])
}

fn scaled_dot_product_attention_chunked<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    _head_dim: usize,
    chunk_tokens: usize,
) -> Tensor<B, 4> {
    let [batch, query_tokens, heads, head_dim] = q.dims();
    let chunk_tokens = chunk_tokens.max(1);
    let mut chunks = Vec::with_capacity(query_tokens.div_ceil(chunk_tokens));

    for start in (0..query_tokens).step_by(chunk_tokens) {
        let end = start.saturating_add(chunk_tokens).min(query_tokens);
        let q_chunk = q
            .clone()
            .slice([0..batch, start..end, 0..heads, 0..head_dim]);
        chunks.push(scaled_dot_product_attention_dense(
            q_chunk,
            k.clone(),
            v.clone(),
            _head_dim,
        ));
    }

    Tensor::cat(chunks, 1)
}

#[derive(Module, Debug)]
pub struct UnifiedTransformerBlock<B: Backend> {
    pub norm1: nn::LayerNorm<B>,
    pub norm2: nn::LayerNorm<B>,
    pub attn: MultiHeadAttention<B>,
    pub mlp: FeedForwardNet<B>,
    pub ada_ln_modulation: Option<nn::Linear<B>>,
    pub shift_table: Option<Param<Tensor<B, 2>>>,
    modulation: bool,
    share_mod: bool,
    channels: usize,
}

impl<B: Backend> UnifiedTransformerBlock<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &B::Device,
        channels: usize,
        num_heads: usize,
        mlp_ratio: f32,
        use_rope: bool,
        qk_rms_norm: bool,
        qkv_bias: bool,
        modulation: bool,
        share_mod: bool,
        use_shift_table: bool,
    ) -> Self {
        Self {
            norm1: nn::LayerNormConfig::new(channels)
                .with_epsilon(1.0e-6)
                .init(device),
            norm2: nn::LayerNormConfig::new(channels)
                .with_epsilon(1.0e-6)
                .init(device),
            attn: MultiHeadAttention::new(
                device,
                channels,
                num_heads,
                None,
                AttentionKind::SelfAttention,
                qkv_bias,
                qk_rms_norm,
                use_rope,
            ),
            mlp: FeedForwardNet::new(device, channels, mlp_ratio),
            ada_ln_modulation: (modulation && !share_mod).then(|| {
                nn::LinearConfig::new(channels, 6 * channels)
                    .with_bias(true)
                    .init(device)
            }),
            shift_table: use_shift_table.then(|| {
                nn::Initializer::Normal {
                    mean: 0.0,
                    std: (channels as f64).powf(-0.5),
                }
                .init([1, 6 * channels], device)
            }),
            modulation,
            share_mod,
            channels,
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        mod_signal: Option<Tensor<B, 2>>,
        rope_emb: Option<&RotaryAngles<B>>,
    ) -> Tensor<B, 3> {
        if !self.modulation {
            let attn = self
                .attn
                .forward(self.norm1.forward(x.clone()), None, rope_emb);
            let x = x + attn;
            let mlp = self.mlp.forward(self.norm2.forward(x.clone()));
            return x + mlp;
        }
        let mut mod_signal = mod_signal.expect("modulated block requires modulation signal");
        if !self.share_mod {
            mod_signal = self
                .ada_ln_modulation
                .as_ref()
                .expect("adaLN modulation missing")
                .forward(silu(mod_signal));
        }
        if let Some(shift_table) = &self.shift_table {
            mod_signal = mod_signal + shift_table.val();
        }
        let [batch, _] = mod_signal.dims();
        let c = self.channels;
        let shift_msa = mod_signal.clone().slice([0..batch, 0..c]);
        let scale_msa = mod_signal.clone().slice([0..batch, c..2 * c]);
        let gate_msa = mod_signal.clone().slice([0..batch, 2 * c..3 * c]);
        let shift_mlp = mod_signal.clone().slice([0..batch, 3 * c..4 * c]);
        let scale_mlp = mod_signal.clone().slice([0..batch, 4 * c..5 * c]);
        let gate_mlp = mod_signal.slice([0..batch, 5 * c..6 * c]);
        let h = self.norm1.forward(x.clone()) * (scale_msa.unsqueeze_dim(1) + 1.0)
            + shift_msa.unsqueeze_dim(1);
        let h = self.attn.forward(h, None, rope_emb);
        let x = x + h * gate_msa.unsqueeze_dim(1);
        let h = self.norm2.forward(x.clone()) * (scale_mlp.unsqueeze_dim(1) + 1.0)
            + shift_mlp.unsqueeze_dim(1);
        x + self.mlp.forward(h) * gate_mlp.unsqueeze_dim(1)
    }
}

#[derive(Module, Debug)]
pub struct CrossOnlyBlock<B: Backend> {
    pub norm1: nn::LayerNorm<B>,
    pub norm2: nn::LayerNorm<B>,
    pub cross_attn: MultiHeadAttention<B>,
    pub mlp: FeedForwardNet<B>,
    pub ada_ln_modulation: Option<nn::Linear<B>>,
    share_mod: bool,
    channels: usize,
}

impl<B: Backend> CrossOnlyBlock<B> {
    pub fn new(
        device: &B::Device,
        channels: usize,
        context_channels: usize,
        num_heads: usize,
        mlp_ratio: f32,
        qk_rms_norm_cross: bool,
        share_mod: bool,
    ) -> Self {
        Self {
            norm1: nn::LayerNormConfig::new(channels)
                .with_epsilon(1.0e-6)
                .init(device),
            norm2: nn::LayerNormConfig::new(channels)
                .with_epsilon(1.0e-6)
                .init(device),
            cross_attn: MultiHeadAttention::new(
                device,
                channels,
                num_heads,
                Some(context_channels),
                AttentionKind::CrossAttention,
                true,
                qk_rms_norm_cross,
                false,
            ),
            mlp: FeedForwardNet::new(device, channels, mlp_ratio),
            ada_ln_modulation: (!share_mod).then(|| {
                nn::LinearConfig::new(channels, 6 * channels)
                    .with_bias(true)
                    .init(device)
            }),
            share_mod,
            channels,
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        mut mod_signal: Tensor<B, 2>,
        context: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        if !self.share_mod {
            mod_signal = self
                .ada_ln_modulation
                .as_ref()
                .expect("cross-only adaLN missing")
                .forward(silu(mod_signal));
        }
        let [batch, _] = mod_signal.dims();
        let c = self.channels;
        let shift_msa = mod_signal.clone().slice([0..batch, 0..c]);
        let scale_msa = mod_signal.clone().slice([0..batch, c..2 * c]);
        let gate_msa = mod_signal.clone().slice([0..batch, 2 * c..3 * c]);
        let shift_mlp = mod_signal.clone().slice([0..batch, 3 * c..4 * c]);
        let scale_mlp = mod_signal.clone().slice([0..batch, 4 * c..5 * c]);
        let gate_mlp = mod_signal.slice([0..batch, 5 * c..6 * c]);
        let h = self.norm1.forward(x.clone()) * (scale_msa.unsqueeze_dim(1) + 1.0)
            + shift_msa.unsqueeze_dim(1);
        let x = x + self.cross_attn.forward(h, Some(context), None) * gate_msa.unsqueeze_dim(1);
        let h = self.norm2.forward(x.clone()) * (scale_mlp.unsqueeze_dim(1) + 1.0)
            + shift_mlp.unsqueeze_dim(1);
        x + self.mlp.forward(h) * gate_mlp.unsqueeze_dim(1)
    }
}

pub fn softmax_last<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    softmax(x, D - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;

    type TestBackend = burn::backend::NdArray<f32>;

    #[test]
    fn pcd_v2_embedder_pads_to_requested_channels() {
        let device = Default::default();
        let embedder = PcdAbsolutePositionEmbedder::v2(10);
        let x = Tensor::<TestBackend, 2>::zeros([4, 3], &device);
        assert_eq!(embedder.forward_2d(x).dims(), [4, 10]);
    }

    #[test]
    fn rotary_embedding_preserves_input_shape() {
        let device = Default::default();
        let repo = RePo3dRotaryEmbedding::<TestBackend>::new(&device, 32, 4, 8);
        let x = Tensor::<TestBackend, 3>::zeros([2, 5, 32], &device);
        let angles = repo.forward(x.clone());
        let q = x.reshape([2, 5, 4, 8]);
        assert_eq!(apply_rotary_emb(q, &angles).dims(), [2, 5, 4, 8]);
    }

    #[test]
    fn repo_rotary_frequencies_match_upstream_initialization() {
        let device = Default::default();
        let repo = RePo3dRotaryEmbedding::<TestBackend>::new(&device, 64, 4, 16);
        let expected = [1.0, 6.0, 11.0, 16.0];
        let actual = repo
            .freqs_2
            .val()
            .to_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("repo freqs");
        assert_eq!(actual, expected);
    }

    #[test]
    fn attention_dense_matches_explicit_scale_reference() {
        let device = Default::default();
        let values = |offset: f32| {
            (0..24)
                .map(|index| (index as f32 + offset) / 17.0)
                .collect::<Vec<_>>()
        };
        let q = Tensor::<TestBackend, 4>::from_data(
            TensorData::new(values(-5.0), [1, 3, 2, 4]),
            &device,
        );
        let k = Tensor::<TestBackend, 4>::from_data(
            TensorData::new(values(3.0), [1, 3, 2, 4]),
            &device,
        );
        let v = Tensor::<TestBackend, 4>::from_data(
            TensorData::new(values(11.0), [1, 3, 2, 4]),
            &device,
        );

        let actual = scaled_dot_product_attention(q.clone(), k.clone(), v.clone(), 4);
        let expected = module_attention(
            q.permute([0, 2, 1, 3]),
            k.permute([0, 2, 1, 3]),
            v.permute([0, 2, 1, 3]),
            None,
            None,
            AttentionModuleOptions {
                scale: Some(4.0f64.powf(-0.5)),
                ..Default::default()
            },
        )
        .permute([0, 2, 1, 3]);

        let actual = actual
            .to_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("actual attention data");
        let expected = expected
            .to_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("expected attention data");
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!(
                (actual - expected).abs() <= 1.0e-5,
                "attention mismatch: actual={actual} expected={expected}"
            );
        }
    }

    #[test]
    fn attention_chunked_matches_dense_reference() {
        let device = Default::default();
        let values = |offset: f32| {
            (0..40)
                .map(|index| ((index as f32 + offset) / 19.0).sin())
                .collect::<Vec<_>>()
        };
        let q = Tensor::<TestBackend, 4>::from_data(
            TensorData::new(values(-3.0), [1, 5, 2, 4]),
            &device,
        );
        let k = Tensor::<TestBackend, 4>::from_data(
            TensorData::new(values(5.0), [1, 5, 2, 4]),
            &device,
        );
        let v = Tensor::<TestBackend, 4>::from_data(
            TensorData::new(values(13.0), [1, 5, 2, 4]),
            &device,
        );

        let actual = scaled_dot_product_attention_chunked(q.clone(), k.clone(), v.clone(), 4, 2);
        let expected = scaled_dot_product_attention_dense(q, k, v, 4);

        let actual = actual
            .to_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("chunked attention data");
        let expected = expected
            .to_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("dense attention data");
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!(
                (actual - expected).abs() <= 1.0e-5,
                "chunked attention mismatch: actual={actual} expected={expected}"
            );
        }
    }
}

use burn::{
    module::Param,
    nn,
    prelude::*,
    tensor::activation::softmax,
    tensor::Distribution,
};

use super::hooks::HookRecorder;

pub fn record_tensor<B: Backend, const D: usize>(
    hook: &mut Option<&mut HookRecorder>,
    name: &str,
    tensor: &Tensor<B, D>,
) {
    if let Some(hook) = hook.as_deref_mut() {
        hook.record_tensor(name, tensor);
    }
}

#[derive(Debug, Clone)]
pub struct FrequencyPositionalEmbedding {
    pub num_freq: usize,
    pub include_pi: bool,
}

impl FrequencyPositionalEmbedding {
    pub fn embed_dim(&self, input_dim: usize) -> usize {
        input_dim + input_dim * self.num_freq * 2
    }

    pub fn forward<B: Backend>(&self, coords: Tensor<B, 3>) -> Tensor<B, 3> {
        let scale_pi = if self.include_pi { core::f32::consts::PI } else { 1.0 };
        let device = coords.device();
        let mut freq_values = Vec::with_capacity(self.num_freq);
        for freq in 0..self.num_freq {
            freq_values.push(scale_pi * 2_f32.powi(freq as i32));
        }
        let freqs = Tensor::<B, 1>::from_floats(freq_values.as_slice(), &device);

        let [b, n, c] = coords.shape().dims();
        let freqs = freqs
            .reshape([1, 1, 1, self.num_freq])
            .expand([b as i64, n as i64, c as i64, -1]);
        let scaled = coords.clone().unsqueeze_dim::<4>(3).mul(freqs);
        let scaled = scaled.reshape([b, n, c * self.num_freq]);

        let sin = scaled.clone().sin();
        let cos = scaled.cos();
        Tensor::cat(vec![coords, sin, cos], 2)
    }
}

#[derive(Module, Debug)]
pub struct RmsNorm<B: Backend> {
    pub gamma: Param<Tensor<B, 1>>,
    epsilon: f32,
}

impl<B: Backend> RmsNorm<B> {
    pub fn new(d_model: usize, epsilon: f32, device: &B::Device) -> Self {
        let gamma = nn::Initializer::Ones.init([d_model], device);
        Self { gamma, epsilon }
    }

    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let variance = input.clone().powf_scalar(2.0).mean_dim(D - 1);
        let input_norm = input.mul(variance.add_scalar(self.epsilon).sqrt().recip());
        input_norm.mul(self.gamma.val().unsqueeze())
    }
}

#[derive(Module, Debug)]
pub struct CrossAttention<B: Backend> {
    pub to_q: nn::Linear<B>,
    pub to_k: nn::Linear<B>,
    pub to_v: nn::Linear<B>,
    pub to_out: nn::Linear<B>,
    pub norm_cross: Option<nn::LayerNorm<B>>,
    pub norm_q: Option<RmsNorm<B>>,
    pub norm_k: Option<RmsNorm<B>>,
    pub num_heads: usize,
    pub head_dim: usize,
    pub scale: f32,
    pub is_cross_attention: bool,
    pub use_triposg_split: bool,
}

impl<B: Backend> CrossAttention<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &B::Device,
        dim: usize,
        context_dim: usize,
        num_heads: usize,
        use_norm_cross: bool,
        qk_norm: bool,
        use_triposg_split: bool,
        is_cross_attention: bool,
    ) -> Self {
        let head_dim = dim / num_heads;
        let scale = (head_dim as f32).powf(-0.5);
        let to_q = nn::LinearConfig::new(dim, dim)
            .with_bias(false)
            .init(device);
        let to_k = nn::LinearConfig::new(context_dim, dim)
            .with_bias(false)
            .init(device);
        let to_v = nn::LinearConfig::new(context_dim, dim)
            .with_bias(false)
            .init(device);
        let to_out = nn::LinearConfig::new(dim, dim)
            .with_bias(true)
            .init(device);

        let norm_cross = if use_norm_cross {
            nn::LayerNormConfig::new(context_dim).init(device).into()
        } else {
            None
        };
        let norm_q = if qk_norm {
            RmsNorm::new(head_dim, 1e-6, device).into()
        } else {
            None
        };
        let norm_k = if qk_norm {
            RmsNorm::new(head_dim, 1e-6, device).into()
        } else {
            None
        };

        Self {
            to_q,
            to_k,
            to_v,
            to_out,
            norm_cross,
            norm_q,
            norm_k,
            num_heads,
            head_dim,
            scale,
            is_cross_attention,
            use_triposg_split,
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        context: Tensor<B, 3>,
        mut hook: Option<&mut HookRecorder>,
        hook_prefix: &str,
    ) -> Tensor<B, 3> {
        let [b, n, c] = x.shape().dims();

        let context = if let Some(norm_cross) = &self.norm_cross {
            norm_cross.forward(context)
        } else {
            context
        };

        let q = self.to_q.forward(x);
        let k = self.to_k.forward(context.clone());
        let v = self.to_v.forward(context);

        record_tensor(&mut hook, &format!("{hook_prefix}.q"), &q);
        record_tensor(&mut hook, &format!("{hook_prefix}.k"), &k);
        record_tensor(&mut hook, &format!("{hook_prefix}.v"), &v);

        let context_len = k.shape().dims::<3>()[1];
        let is_cross_attention = self.is_cross_attention || context_len != n;

        let (q, k, v) = if self.use_triposg_split {
            if is_cross_attention {
                let m = context_len;
                let q = q
                    .reshape([b, n, self.num_heads, self.head_dim])
                    .permute([0, 2, 1, 3]);
                let kv = Tensor::cat(vec![k, v], 2)
                    .reshape([b, m, self.num_heads, self.head_dim * 2]);
                let k = kv
                    .clone()
                    .slice([0..b, 0..m, 0..self.num_heads, 0..self.head_dim])
                    .permute([0, 2, 1, 3]);
                let v = kv
                    .slice([
                        0..b,
                        0..m,
                        0..self.num_heads,
                        self.head_dim..(self.head_dim * 2),
                    ])
                    .permute([0, 2, 1, 3]);
                (q, k, v)
            } else {
                let qkv = Tensor::cat(vec![q, k, v], 2)
                    .reshape([b, n, self.num_heads, self.head_dim * 3]);
                let q = qkv
                    .clone()
                    .slice([0..b, 0..n, 0..self.num_heads, 0..self.head_dim])
                    .permute([0, 2, 1, 3]);
                let k = qkv
                    .clone()
                    .slice([
                        0..b,
                        0..n,
                        0..self.num_heads,
                        self.head_dim..(self.head_dim * 2),
                    ])
                    .permute([0, 2, 1, 3]);
                let v = qkv
                    .slice([
                        0..b,
                        0..n,
                        0..self.num_heads,
                        (self.head_dim * 2)..(self.head_dim * 3),
                    ])
                    .permute([0, 2, 1, 3]);
                (q, k, v)
            }
        } else {
            let q = q
                .reshape([b, n, self.num_heads, self.head_dim])
                .permute([0, 2, 1, 3]);
            let k = k
                .reshape([b, context_len, self.num_heads, self.head_dim])
                .permute([0, 2, 1, 3]);
            let v = v
                .reshape([b, context_len, self.num_heads, self.head_dim])
                .permute([0, 2, 1, 3]);
            (q, k, v)
        };

        let q = if let Some(norm_q) = &self.norm_q {
            norm_q.forward(q)
        } else {
            q
        };
        let k = if let Some(norm_k) = &self.norm_k {
            norm_k.forward(k)
        } else {
            k
        };

        let attn = q.matmul(k.swap_dims(2, 3)).mul_scalar(self.scale);
        let attn = softmax(attn, 3);
        record_tensor(&mut hook, &format!("{hook_prefix}.attn"), &attn);

        let out = attn
            .matmul(v)
            .permute([0, 2, 1, 3])
            .reshape([b, n, c]);
        let out = self.to_out.forward(out);
        record_tensor(&mut hook, &format!("{hook_prefix}.out"), &out);
        out
    }
}

#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    pub proj: nn::Linear<B>,
    pub out: nn::Linear<B>,
    activation: nn::Gelu,
    dropout: nn::Dropout,
}

impl<B: Backend> FeedForward<B> {
    pub fn new(device: &B::Device, dim: usize, hidden_dim: usize) -> Self {
        let proj = nn::LinearConfig::new(dim, hidden_dim)
            .with_bias(true)
            .init(device);
        let out = nn::LinearConfig::new(hidden_dim, dim)
            .with_bias(true)
            .init(device);
        let activation = nn::Gelu::new();
        let dropout = nn::DropoutConfig::new(0.0).init();

        Self {
            proj,
            out,
            activation,
            dropout,
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        mut hook: Option<&mut HookRecorder>,
        hook_prefix: &str,
    ) -> Tensor<B, 3> {
        let x = self.proj.forward(x);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);
        let x = self.out.forward(x);
        let x = self.dropout.forward(x);
        record_tensor(&mut hook, hook_prefix, &x);
        x
    }
}

#[derive(Debug, Clone)]
pub struct DiagonalGaussianDistribution<B: Backend> {
    pub mean: Tensor<B, 3>,
    pub logvar: Tensor<B, 3>,
}

impl<B: Backend> DiagonalGaussianDistribution<B> {
    pub fn new(mean: Tensor<B, 3>, logvar: Tensor<B, 3>) -> Self {
        Self { mean, logvar }
    }

    pub fn sample(&self) -> Tensor<B, 3> {
        let std = self.logvar.clone().mul_scalar(0.5).exp();
        let noise = Tensor::<B, 3>::random(
            std.shape(),
            Distribution::Normal(0.0, 1.0),
            &std.device(),
        );
        self.mean.clone() + std * noise
    }

    pub fn mode(&self) -> Tensor<B, 3> {
        self.mean.clone()
    }
}

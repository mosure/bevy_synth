use std::time::Instant;

use burn::{
    module::Param,
    nn,
    prelude::*,
    tensor::{
        DType, FloatDType, TensorData,
        activation::{sigmoid, softmax},
        module::attention as module_attention,
        ops::AttentionModuleOptions,
    },
};

const RMS_NORM_EPS: f32 = 1.0e-12;
const ATTENTION_SCORE_ELEMS_CHUNK_THRESHOLD: usize = 32 * 1024 * 1024;
#[cfg(target_arch = "wasm32")]
const ATTENTION_QUERY_CHUNK_TOKENS_F32: usize = 128;
#[cfg(not(target_arch = "wasm32"))]
const ATTENTION_QUERY_CHUNK_TOKENS_F32: usize = 4096;
#[cfg(target_arch = "wasm32")]
const ATTENTION_QUERY_CHUNK_TOKENS_F16: usize = 128;
#[cfg(not(target_arch = "wasm32"))]
const ATTENTION_QUERY_CHUNK_TOKENS_F16: usize = 4096;
const WGPU_BATCHED_SAFE_ATTENTION_QUERY_CHUNK_TOKENS: usize = 2048;
const WGPU_BATCHED_SAFE_ATTENTION_QUERY_CHUNK_LIMIT: usize = 2560;
const WGPU_LONG_ATTENTION_MIN_FLASH_TOKENS: usize = 8192;
const WGPU_F32_UNIT_FLASH_UNSAFE_SCORE_ELEMS_PER_HEAD: usize = 1 << 26;
const WGPU_F32_UNIT_FLASH_QUERY_CHUNK_ALIGNMENT: usize = 64;
const WGPU_SAFE_ATTENTION_BINDING_BYTES: usize = 1_900_000_000;
const WGPU_DIRECT_PUBLIC_ATTENTION_QUERY_CHUNK_TOKENS: usize = 2048;
const WGPU_F16_BLACKBOX_ATTENTION_QUERY_PAD_MULTIPLE: usize = 128;

pub(crate) fn default_attention_query_chunk_tokens(dtype: FloatDType) -> usize {
    if matches!(dtype, FloatDType::F16 | FloatDType::BF16) {
        ATTENTION_QUERY_CHUNK_TOKENS_F16
    } else {
        ATTENTION_QUERY_CHUNK_TOKENS_F32
    }
}

fn resolved_attention_query_chunk_tokens(
    dtype: FloatDType,
    batch: usize,
    query_tokens: usize,
    key_tokens: usize,
    requested_query_chunk_tokens: usize,
    backend_name: Option<&str>,
) -> usize {
    let requested_query_chunk_tokens = requested_query_chunk_tokens.max(1);
    let default_query_chunk_tokens = default_attention_query_chunk_tokens(dtype);

    if cfg!(target_arch = "wasm32") && requested_query_chunk_tokens == default_query_chunk_tokens {
        default_query_chunk_tokens
    } else if native_wgpu_batched_attention_needs_safe_chunk(
        batch,
        query_tokens,
        key_tokens,
        requested_query_chunk_tokens,
        backend_name,
    ) {
        WGPU_BATCHED_SAFE_ATTENTION_QUERY_CHUNK_TOKENS
    } else {
        requested_query_chunk_tokens
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn native_wgpu_batched_attention_needs_safe_chunk(
    batch: usize,
    query_tokens: usize,
    key_tokens: usize,
    requested_query_chunk_tokens: usize,
    backend_name: Option<&str>,
) -> bool {
    if batch <= 1
        || query_tokens <= WGPU_BATCHED_SAFE_ATTENTION_QUERY_CHUNK_LIMIT
        || key_tokens <= WGPU_BATCHED_SAFE_ATTENTION_QUERY_CHUNK_LIMIT
        || requested_query_chunk_tokens <= WGPU_BATCHED_SAFE_ATTENTION_QUERY_CHUNK_LIMIT
    {
        return false;
    }
    let Some(backend_name) = backend_name else {
        return false;
    };
    let backend_name = backend_name.to_ascii_lowercase();
    backend_name.contains("wgpu") || backend_name.contains("spirv")
}

#[cfg(target_arch = "wasm32")]
fn native_wgpu_batched_attention_needs_safe_chunk(
    _batch: usize,
    _query_tokens: usize,
    _key_tokens: usize,
    _requested_query_chunk_tokens: usize,
    _backend_name: Option<&str>,
) -> bool {
    false
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct TripoSplatProfileRecord {
    pub label: String,
    pub batch: usize,
    pub tokens: usize,
    pub channels: usize,
    pub elapsed_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub key_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub heads: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub head_dim: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score_elems: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub query_chunk_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub query_chunks: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dense_calls: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dtype: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attention_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backend_hint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finite_checked: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nonfinite_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_nonfinite_index: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_finite: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_finite: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean_finite: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rms_finite: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_abs_finite: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finite_error: Option<String>,
}

#[derive(Clone, Debug)]
pub struct AttentionQkvCapture<B: Backend> {
    pub label: String,
    pub q: Tensor<B, 4>,
    pub k: Tensor<B, 4>,
    pub v: Tensor<B, 4>,
}

#[derive(Debug)]
pub struct AttentionQkvCaptureState<B: Backend> {
    label_filter: String,
    captured: Option<AttentionQkvCapture<B>>,
}

impl<B: Backend> AttentionQkvCaptureState<B> {
    pub fn new(label_filter: impl Into<String>) -> Self {
        Self {
            label_filter: label_filter.into(),
            captured: None,
        }
    }

    pub fn label_filter(&self) -> &str {
        &self.label_filter
    }

    pub fn captured(&self) -> Option<&AttentionQkvCapture<B>> {
        self.captured.as_ref()
    }

    pub fn into_captured(self) -> Option<AttentionQkvCapture<B>> {
        self.captured
    }

    pub fn try_capture(
        &mut self,
        label: &str,
        q: &Tensor<B, 4>,
        k: &Tensor<B, 4>,
        v: &Tensor<B, 4>,
    ) {
        if self.captured.is_some() || !label.contains(self.label_filter.as_str()) {
            return;
        }
        self.captured = Some(AttentionQkvCapture {
            label: label.to_string(),
            q: q.clone(),
            k: k.clone(),
            v: v.clone(),
        });
    }
}

pub(crate) fn sync_elapsed_ms<B: Backend>(device: &B::Device, start: Instant) -> f64 {
    B::sync(device).expect("profile sync failed");
    start.elapsed().as_secs_f64() * 1000.0
}

pub(crate) fn push_profile_record(
    records: &mut Vec<TripoSplatProfileRecord>,
    label: impl Into<String>,
    batch: usize,
    tokens: usize,
    channels: usize,
    elapsed_ms: f64,
) {
    records.push(TripoSplatProfileRecord {
        label: label.into(),
        batch,
        tokens,
        channels,
        elapsed_ms,
        key_tokens: None,
        heads: None,
        head_dim: None,
        score_elems: None,
        query_chunk_tokens: None,
        query_chunks: None,
        dense_calls: None,
        dtype: None,
        attention_path: None,
        backend_hint: None,
        finite_checked: None,
        nonfinite_count: None,
        first_nonfinite_index: None,
        min_finite: None,
        max_finite: None,
        mean_finite: None,
        rms_finite: None,
        max_abs_finite: None,
        finite_error: None,
    });
}

pub(crate) fn push_finite_debug_record<B: Backend, const D: usize>(
    records: &mut Vec<TripoSplatProfileRecord>,
    label: impl Into<String>,
    tensor: &Tensor<B, D>,
) {
    if !flow_finite_debug_enabled() {
        return;
    }

    let dims = tensor.dims();
    let (batch, tokens, channels) = profile_dims(dims.as_slice());
    let mut record = TripoSplatProfileRecord {
        label: label.into(),
        batch,
        tokens,
        channels,
        elapsed_ms: 0.0,
        key_tokens: None,
        heads: None,
        head_dim: None,
        score_elems: None,
        query_chunk_tokens: None,
        query_chunks: None,
        dense_calls: None,
        dtype: Some(format!("{:?}", tensor.dtype())),
        attention_path: None,
        backend_hint: None,
        finite_checked: Some(true),
        nonfinite_count: Some(0),
        first_nonfinite_index: None,
        min_finite: None,
        max_finite: None,
        mean_finite: None,
        rms_finite: None,
        max_abs_finite: Some(0.0),
        finite_error: None,
    };

    match tensor.clone().to_data().convert::<f32>().to_vec::<f32>() {
        Ok(values) => {
            let mut nonfinite_count = 0usize;
            let mut first_nonfinite_index = None;
            let mut max_abs_finite = 0.0f32;
            let mut min_finite = f32::INFINITY;
            let mut max_finite = f32::NEG_INFINITY;
            let mut sum_finite = 0.0f64;
            let mut sum_sq_finite = 0.0f64;
            let mut finite_count = 0usize;
            for (index, value) in values.iter().copied().enumerate() {
                if value.is_finite() {
                    max_abs_finite = max_abs_finite.max(value.abs());
                    min_finite = min_finite.min(value);
                    max_finite = max_finite.max(value);
                    sum_finite += value as f64;
                    sum_sq_finite += (value as f64) * (value as f64);
                    finite_count += 1;
                } else {
                    nonfinite_count += 1;
                    first_nonfinite_index.get_or_insert(index);
                }
            }
            record.nonfinite_count = Some(nonfinite_count);
            record.first_nonfinite_index = first_nonfinite_index;
            record.max_abs_finite = Some(max_abs_finite);
            if finite_count > 0 {
                record.min_finite = Some(min_finite);
                record.max_finite = Some(max_finite);
                record.mean_finite = Some((sum_finite / finite_count as f64) as f32);
                record.rms_finite = Some((sum_sq_finite / finite_count as f64).sqrt() as f32);
            }
        }
        Err(err) => {
            record.finite_error = Some(format!("{err:?}"));
        }
    }

    records.push(record);
}

fn flow_finite_debug_enabled() -> bool {
    std::env::var("TRIPOSPLAT_FLOW_FINITE_DEBUG")
        .ok()
        .is_some_and(|value| {
            let value = value.trim();
            !(value.is_empty()
                || value.eq_ignore_ascii_case("0")
                || value.eq_ignore_ascii_case("false")
                || value.eq_ignore_ascii_case("off"))
        })
}

fn profile_dims(dims: &[usize]) -> (usize, usize, usize) {
    match dims {
        [batch, tokens, channels] => (*batch, *tokens, *channels),
        [batch, tokens, heads, head_dim] => (*batch, *tokens, heads.saturating_mul(*head_dim)),
        [batch, channels] => (*batch, 1, *channels),
        [channels] => (1, 1, *channels),
        _ => (dims.first().copied().unwrap_or(0), 0, 0),
    }
}

#[derive(Clone, Debug)]
struct AttentionProfileMeta {
    batch: usize,
    query_tokens: usize,
    key_tokens: usize,
    heads: usize,
    head_dim: usize,
    score_elems: usize,
    query_chunk_tokens: usize,
    query_chunks: usize,
    dense_calls: usize,
    dtype: String,
    attention_path: String,
    backend_hint: String,
}

fn attention_profile_meta<B: Backend>(
    q: &Tensor<B, 4>,
    k: &Tensor<B, 4>,
    query_chunk_tokens: usize,
) -> AttentionProfileMeta {
    let [batch, query_tokens, heads, head_dim] = q.dims();
    let key_tokens = k.dims()[1];
    let backend_name = B::name(&q.device());
    let requested_query_chunk_tokens = query_chunk_tokens.max(1);
    let dtype = q.dtype().into();
    let default_query_chunk_tokens = default_attention_query_chunk_tokens(dtype);
    let requested_default_chunking = requested_query_chunk_tokens == default_query_chunk_tokens;
    let safe_flash_requested_query_chunk_tokens = if requested_default_chunking {
        query_tokens
    } else {
        requested_query_chunk_tokens
    };
    let safe_flash_query_chunk_tokens = native_wgpu_f32_unit_flash_safe_query_chunk_tokens(
        batch,
        heads,
        dtype,
        query_tokens,
        key_tokens,
        safe_flash_requested_query_chunk_tokens,
        Some(backend_name.as_str()),
    );
    let safe_binding_query_chunk_tokens = native_wgpu_attention_binding_safe_query_chunk_tokens(
        batch,
        heads,
        dtype,
        query_tokens,
        key_tokens,
        requested_query_chunk_tokens,
        Some(backend_name.as_str()),
    );
    let uses_module_flash_full = native_wgpu_f32_long_attention_prefers_module_flash(
        dtype,
        query_tokens,
        key_tokens,
        Some(backend_name.as_str()),
    ) && requested_default_chunking
        && safe_flash_query_chunk_tokens.is_none()
        && safe_binding_query_chunk_tokens.is_none();
    let score_elems = batch
        .saturating_mul(heads)
        .saturating_mul(query_tokens)
        .saturating_mul(key_tokens);
    let mut module_query_chunk_tokens = resolved_attention_query_chunk_tokens(
        dtype,
        batch,
        query_tokens,
        key_tokens,
        query_chunk_tokens,
        Some(backend_name.as_str()),
    );
    if let Some(safe_flash_query_chunk_tokens) = safe_flash_query_chunk_tokens {
        module_query_chunk_tokens = safe_flash_query_chunk_tokens;
    }
    if let Some(safe_binding_query_chunk_tokens) = safe_binding_query_chunk_tokens {
        module_query_chunk_tokens = module_query_chunk_tokens.min(safe_binding_query_chunk_tokens);
    }
    if uses_module_flash_full {
        module_query_chunk_tokens = query_tokens;
    }
    let direct_public_query_chunk_tokens = direct_public_attention_query_chunk_tokens(
        dtype,
        batch,
        heads,
        head_dim,
        query_tokens,
        key_tokens,
        module_query_chunk_tokens,
        Some(backend_name.as_str()),
    );
    let query_chunk_tokens = direct_public_query_chunk_tokens.unwrap_or(module_query_chunk_tokens);
    let explicit_chunk_limit =
        !uses_module_flash_full && query_chunk_tokens < default_query_chunk_tokens;
    let uses_chunking = !uses_module_flash_full
        && query_tokens > query_chunk_tokens
        && (explicit_chunk_limit || score_elems > ATTENTION_SCORE_ELEMS_CHUNK_THRESHOLD);
    let query_chunks = if uses_chunking {
        query_tokens.div_ceil(query_chunk_tokens.max(1))
    } else {
        1
    };
    let backend_type = std::any::type_name::<B>();
    let explicit_requested_limit = query_chunk_tokens < requested_query_chunk_tokens;
    let backend_hint = if backend_type.contains("burn_cubecl")
        || backend_type.contains("burn_wgpu")
        || backend_type.contains("CubeBackend")
        || backend_type.contains("Wgpu")
    {
        if uses_module_flash_full {
            "cubecl_module_flash_attention_f32_long_attention".to_string()
        } else if cfg!(feature = "backend_wgpu") {
            "cubecl_module_attention_default_autotune_eligible_contiguous_bhtd".to_string()
        } else {
            "cubecl_module_attention_default_contiguous_bhtd".to_string()
        }
    } else if backend_type.contains("NdArray") {
        "ndarray_attention_fallback".to_string()
    } else {
        "backend_attention_dispatch_unknown".to_string()
    };
    let blackbox_pad_multiple =
        native_wgpu_f16_blackbox_query_pad_multiple(dtype, Some(backend_name.as_str()));
    let backend_hint = if explicit_requested_limit {
        format!("{backend_hint}; explicit_query_chunk_limit")
    } else {
        backend_hint
    };
    let backend_hint = if let Some(pad_multiple) = blackbox_pad_multiple {
        format!("{backend_hint}; query_pad_multiple_for_f16_blackbox={pad_multiple}")
    } else {
        backend_hint
    };
    let dense_calls = if uses_module_flash_full {
        1
    } else {
        query_chunks
    };
    AttentionProfileMeta {
        batch,
        query_tokens,
        key_tokens,
        heads,
        head_dim,
        score_elems,
        query_chunk_tokens,
        query_chunks,
        dense_calls,
        dtype: format!("{:?}", q.dtype()),
        attention_path: if blackbox_pad_multiple.is_some() && uses_chunking {
            "chunked_padded_module_attention_f16_blackbox_eligible".to_string()
        } else if blackbox_pad_multiple.is_some() {
            "dense_padded_module_attention_f16_blackbox_eligible".to_string()
        } else if direct_public_query_chunk_tokens.is_some() && uses_chunking {
            "chunked_direct_public_primitives_attention".to_string()
        } else if direct_public_query_chunk_tokens.is_some() {
            "dense_direct_public_primitives_attention".to_string()
        } else if safe_flash_query_chunk_tokens.is_some() && uses_chunking {
            "chunked_module_flash_attention_safe_wgpu_f32".to_string()
        } else if safe_binding_query_chunk_tokens.is_some() && uses_chunking {
            "chunked_default_module_attention_safe_wgpu_binding".to_string()
        } else if uses_module_flash_full {
            "dense_module_flash_attention".to_string()
        } else if uses_chunking {
            "chunked_default_module_attention".to_string()
        } else {
            "dense_default_module_attention".to_string()
        },
        backend_hint,
    }
}

fn push_attention_profile_record(
    records: &mut Vec<TripoSplatProfileRecord>,
    label: impl Into<String>,
    channels: usize,
    elapsed_ms: f64,
    meta: AttentionProfileMeta,
) {
    records.push(TripoSplatProfileRecord {
        label: label.into(),
        batch: meta.batch,
        tokens: meta.query_tokens,
        channels,
        elapsed_ms,
        key_tokens: Some(meta.key_tokens),
        heads: Some(meta.heads),
        head_dim: Some(meta.head_dim),
        score_elems: Some(meta.score_elems),
        query_chunk_tokens: Some(meta.query_chunk_tokens),
        query_chunks: Some(meta.query_chunks),
        dense_calls: Some(meta.dense_calls),
        dtype: Some(meta.dtype),
        attention_path: Some(meta.attention_path),
        backend_hint: Some(meta.backend_hint),
        finite_checked: None,
        nonfinite_count: None,
        first_nonfinite_index: None,
        min_finite: None,
        max_finite: None,
        mean_finite: None,
        rms_finite: None,
        max_abs_finite: None,
        finite_error: None,
    });
}

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

fn cast_tensor_dtype<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    dtype: FloatDType,
) -> Tensor<B, D> {
    let current: FloatDType = tensor.dtype().into();
    if current == dtype {
        tensor
    } else {
        tensor.cast(dtype)
    }
}

fn f32_tensor_1d<B: Backend>(values: Vec<f32>, device: &B::Device) -> Tensor<B, 1> {
    let len = values.len();
    Tensor::<B, 1>::from_data(TensorData::new(values, [len]), (device, DType::F32))
}

fn force_contiguous_4d<B: Backend>(tensor: Tensor<B, 4>) -> Tensor<B, 4> {
    let dims = tensor.dims();
    tensor.flatten::<1>(0, 3).reshape(dims)
}

fn layer_norm32_3d<B: Backend>(norm: &nn::LayerNorm<B>, input: Tensor<B, 3>) -> Tensor<B, 3> {
    let dtype: FloatDType = input.dtype().into();
    let [_batch, _tokens, channels] = input.dims();
    let x_acc = cast_low_precision_to_f32(input, dtype);
    let (var, mean) = x_acc.clone().var_mean_bias(2);
    let mut out = (x_acc - mean) / var.add_scalar(1.0e-6).sqrt();
    out = out
        * norm
            .gamma
            .val()
            .cast(FloatDType::F32)
            .reshape([1, 1, channels]);
    if let Some(beta) = &norm.beta {
        out = out + beta.val().cast(FloatDType::F32).reshape([1, 1, channels]);
    }
    cast_tensor_dtype(out, dtype)
}

fn push_trace_tensor3<B: Backend>(
    trace: &mut Vec<(String, Tensor<B, 3>)>,
    label: impl Into<String>,
    tensor: &Tensor<B, 3>,
    token_limit: usize,
) {
    let [batch, tokens, channels] = tensor.dims();
    let end = token_limit.max(1).min(tokens);
    let clipped = if end < tokens {
        tensor.clone().slice([0..batch, 0..end, 0..channels])
    } else {
        tensor.clone()
    };
    trace.push((label.into(), clipped));
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
        let weight_dtype: FloatDType = self.mlp_0.weight.val().dtype().into();
        let emb = cast_tensor_dtype(emb, weight_dtype);
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
    let acc_dtype = accumulation_dtype(dtype);
    let t_acc = cast_low_precision_to_f32(t, dtype);
    let mut freqs = Vec::with_capacity(half);
    for index in 0..half {
        freqs.push((-max_period.ln() * index as f32 / half as f32).exp());
    }
    let freqs = f32_tensor_1d(freqs, &device).cast(acc_dtype);
    let mut args = t_acc.unsqueeze_dim(1) * freqs.unsqueeze_dim(0);
    if multiply_2pi {
        args = args.mul_scalar(core::f32::consts::TAU);
    }
    let mut out = Tensor::cat(vec![args.clone().cos(), args.sin()], 1);
    if dim % 2 == 1 {
        out = Tensor::cat(
            vec![
                out,
                Tensor::<B, 2>::zeros([batch, 1], &device).cast(acc_dtype),
            ],
            1,
        );
    }
    cast_from_f32_accum(out, dtype)
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
        let acc_dtype = accumulation_dtype(dtype);
        let x_acc = cast_low_precision_to_f32(x, dtype);
        let freq_dim = self.channels / self.in_channels / 2;
        let freqs = if self.linear_residual {
            legacy_freqs(freq_dim, self.max_res)
        } else {
            linspace_pow2(freq_dim, self.max_res)
        };
        let freqs = f32_tensor_1d(freqs, &device).cast(acc_dtype);
        let angle_scale = if self.linear_residual {
            core::f32::consts::TAU
        } else {
            core::f32::consts::PI
        };
        let phase = x_acc
            .unsqueeze_dim::<3>(2)
            .mul(freqs.reshape([1, 1, freq_dim]));
        let phase_period = if self.linear_residual { 1.0 } else { 2.0 };
        let reduced_phase = phase.clone()
            - (phase.clone() / phase_period)
                .floor()
                .mul_scalar(phase_period);
        let scaled = reduced_phase.mul_scalar(angle_scale);
        let mut out = Tensor::cat(vec![scaled.clone().sin(), scaled.cos()], 2)
            .reshape([tokens, dims * freq_dim * 2]);
        if out.dims()[1] < self.channels {
            let pad = Tensor::<B, 2>::zeros([tokens, self.channels - out.dims()[1]], &device)
                .cast(acc_dtype);
            out = Tensor::cat(vec![out, pad], 1);
        }
        cast_from_f32_accum(out, dtype)
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
        let h = layer_norm32_3d(&self.norm, hidden_states);
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
        let angles = Tensor::cat(vec![a0, a1, a2], 3)
            .mul_scalar(core::f32::consts::PI)
            .cast(FloatDType::F32);
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
    let dtype: FloatDType = hidden_states.dtype().into();
    let pairs = head_dim / 2;
    let pair = hidden_states
        .cast(FloatDType::F32)
        .reshape([batch, tokens, heads, pairs, 2]);
    let even = pair
        .clone()
        .slice([0..batch, 0..tokens, 0..heads, 0..pairs, 0..1]);
    let odd = pair.slice([0..batch, 0..tokens, 0..heads, 0..pairs, 1..2]);
    let cos = freqs.cos.clone().unsqueeze_dim::<5>(4);
    let sin = freqs.sin.clone().unsqueeze_dim::<5>(4);
    let out_even = even.clone() * cos.clone() - odd.clone() * sin.clone();
    let out_odd = even * sin + odd * cos;
    cast_tensor_dtype(
        Tensor::cat(vec![out_even, out_odd], 4).reshape([batch, tokens, heads, head_dim]),
        dtype,
    )
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
        let dtype: FloatDType = x.dtype().into();
        self.forward_with_query_chunk_tokens(
            x,
            context,
            rope_emb,
            default_attention_query_chunk_tokens(dtype),
        )
    }

    pub fn forward_with_query_chunk_tokens(
        &self,
        x: Tensor<B, 3>,
        context: Option<Tensor<B, 3>>,
        rope_emb: Option<&RotaryAngles<B>>,
        query_chunk_tokens: usize,
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
        let out = scaled_dot_product_attention_with_query_chunk_tokens(
            q,
            k,
            v,
            self.head_dim,
            query_chunk_tokens,
        );
        self.out.forward(out.reshape([batch, tokens, channels]))
    }

    pub fn forward_profiled(
        &self,
        label: &str,
        x: Tensor<B, 3>,
        context: Option<Tensor<B, 3>>,
        rope_emb: Option<&RotaryAngles<B>>,
        query_chunk_tokens: usize,
        records: &mut Vec<TripoSplatProfileRecord>,
    ) -> Tensor<B, 3> {
        self.forward_profiled_with_qkv_capture(
            label,
            x,
            context,
            rope_emb,
            query_chunk_tokens,
            records,
            None,
        )
    }

    pub fn forward_profiled_with_qkv_capture(
        &self,
        label: &str,
        x: Tensor<B, 3>,
        context: Option<Tensor<B, 3>>,
        rope_emb: Option<&RotaryAngles<B>>,
        query_chunk_tokens: usize,
        records: &mut Vec<TripoSplatProfileRecord>,
        mut qkv_capture: Option<&mut AttentionQkvCaptureState<B>>,
    ) -> Tensor<B, 3> {
        let device = x.device();
        let [batch, tokens, channels] = x.dims();
        let total_start = Instant::now();
        let (mut q, mut k, v) = match self.kind {
            AttentionKind::SelfAttention => {
                let qkv_start = Instant::now();
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
                push_profile_record(
                    records,
                    format!("{label}.qkv"),
                    batch,
                    tokens,
                    channels,
                    sync_elapsed_ms::<B>(&device, qkv_start),
                );
                push_finite_debug_record(records, format!("{label}.q"), &q);
                push_finite_debug_record(records, format!("{label}.k"), &k);
                push_finite_debug_record(records, format!("{label}.v"), &v);
                (q, k, v)
            }
            AttentionKind::CrossAttention => {
                let context = context.expect("context required for cross attention");
                let context_tokens = context.dims()[1];
                let q_start = Instant::now();
                let q = self
                    .q
                    .as_ref()
                    .expect("cross attention q missing")
                    .forward(x)
                    .reshape([batch, tokens, self.num_heads, self.head_dim]);
                push_profile_record(
                    records,
                    format!("{label}.q"),
                    batch,
                    tokens,
                    channels,
                    sync_elapsed_ms::<B>(&device, q_start),
                );
                push_finite_debug_record(records, format!("{label}.q.out"), &q);
                let kv_start = Instant::now();
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
                push_profile_record(
                    records,
                    format!("{label}.kv"),
                    batch,
                    context_tokens,
                    channels,
                    sync_elapsed_ms::<B>(&device, kv_start),
                );
                push_finite_debug_record(records, format!("{label}.k"), &k);
                push_finite_debug_record(records, format!("{label}.v"), &v);
                (q, k, v)
            }
        };
        if self.use_rope
            && let Some(rope) = rope_emb
        {
            let rope_start = Instant::now();
            q = apply_rotary_emb(q, rope);
            k = apply_rotary_emb(k, rope);
            push_profile_record(
                records,
                format!("{label}.rope"),
                batch,
                tokens,
                channels,
                sync_elapsed_ms::<B>(&device, rope_start),
            );
            push_finite_debug_record(records, format!("{label}.rope.q"), &q);
            push_finite_debug_record(records, format!("{label}.rope.k"), &k);
        }
        if let Some(norm) = &self.q_norm {
            let q_norm_start = Instant::now();
            q = norm.forward(q);
            push_profile_record(
                records,
                format!("{label}.q_norm"),
                batch,
                tokens,
                channels,
                sync_elapsed_ms::<B>(&device, q_norm_start),
            );
            push_finite_debug_record(records, format!("{label}.q_norm.out"), &q);
        }
        if let Some(norm) = &self.k_norm {
            let k_norm_start = Instant::now();
            k = norm.forward(k);
            push_profile_record(
                records,
                format!("{label}.k_norm"),
                batch,
                k.dims()[1],
                channels,
                sync_elapsed_ms::<B>(&device, k_norm_start),
            );
            push_finite_debug_record(records, format!("{label}.k_norm.out"), &k);
        }
        if let Some(capture) = qkv_capture.as_deref_mut() {
            capture.try_capture(label, &q, &k, &v);
        }
        let sdpa_start = Instant::now();
        let attention_meta = attention_profile_meta::<B>(&q, &k, query_chunk_tokens);
        let out = scaled_dot_product_attention_with_query_chunk_tokens(
            q,
            k,
            v,
            self.head_dim,
            query_chunk_tokens,
        );
        push_attention_profile_record(
            records,
            format!("{label}.sdpa"),
            channels,
            sync_elapsed_ms::<B>(&device, sdpa_start),
            attention_meta,
        );
        push_finite_debug_record(records, format!("{label}.sdpa.out"), &out);
        let out_start = Instant::now();
        let out = self.out.forward(out.reshape([batch, tokens, channels]));
        push_profile_record(
            records,
            format!("{label}.out"),
            batch,
            tokens,
            channels,
            sync_elapsed_ms::<B>(&device, out_start),
        );
        push_finite_debug_record(records, format!("{label}.out.out"), &out);
        push_profile_record(
            records,
            format!("{label}.total"),
            batch,
            tokens,
            channels,
            sync_elapsed_ms::<B>(&device, total_start),
        );
        out
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn native_wgpu_f32_long_attention_prefers_module_flash(
    dtype: FloatDType,
    query_tokens: usize,
    key_tokens: usize,
    backend_name: Option<&str>,
) -> bool {
    if !matches!(dtype, FloatDType::F32)
        || query_tokens <= WGPU_BATCHED_SAFE_ATTENTION_QUERY_CHUNK_LIMIT
        || key_tokens < WGPU_LONG_ATTENTION_MIN_FLASH_TOKENS
    {
        return false;
    }
    let Some(backend_name) = backend_name else {
        return false;
    };
    let backend_name = backend_name.to_ascii_lowercase();
    backend_name.contains("wgpu") || backend_name.contains("spirv")
}

#[cfg(target_arch = "wasm32")]
fn native_wgpu_f32_long_attention_prefers_module_flash(
    _dtype: FloatDType,
    _query_tokens: usize,
    _key_tokens: usize,
    _backend_name: Option<&str>,
) -> bool {
    false
}

#[cfg(not(target_arch = "wasm32"))]
fn native_wgpu_f32_unit_flash_safe_query_chunk_tokens(
    batch: usize,
    heads: usize,
    dtype: FloatDType,
    query_tokens: usize,
    key_tokens: usize,
    requested_query_chunk_tokens: usize,
    backend_name: Option<&str>,
) -> Option<usize> {
    if !native_wgpu_f32_long_attention_prefers_module_flash(
        dtype,
        query_tokens,
        key_tokens,
        backend_name,
    ) || batch.saturating_mul(heads) < 16
        || query_tokens.saturating_mul(key_tokens) < WGPU_F32_UNIT_FLASH_UNSAFE_SCORE_ELEMS_PER_HEAD
    {
        return None;
    }

    let max_safe_tokens =
        (WGPU_F32_UNIT_FLASH_UNSAFE_SCORE_ELEMS_PER_HEAD - 1).saturating_div(key_tokens.max(1));
    let aligned_safe_tokens = (max_safe_tokens / WGPU_F32_UNIT_FLASH_QUERY_CHUNK_ALIGNMENT)
        .saturating_mul(WGPU_F32_UNIT_FLASH_QUERY_CHUNK_ALIGNMENT)
        .max(1);
    Some(
        requested_query_chunk_tokens
            .max(1)
            .min(aligned_safe_tokens)
            .min(query_tokens),
    )
}

#[cfg(target_arch = "wasm32")]
fn native_wgpu_f32_unit_flash_safe_query_chunk_tokens(
    _batch: usize,
    _heads: usize,
    _dtype: FloatDType,
    _query_tokens: usize,
    _key_tokens: usize,
    _requested_query_chunk_tokens: usize,
    _backend_name: Option<&str>,
) -> Option<usize> {
    None
}

#[cfg(not(target_arch = "wasm32"))]
fn direct_public_attention_query_chunk_tokens(
    dtype: FloatDType,
    batch: usize,
    heads: usize,
    head_dim: usize,
    query_tokens: usize,
    key_tokens: usize,
    requested_query_chunk_tokens: usize,
    backend_name: Option<&str>,
) -> Option<usize> {
    if dtype != FloatDType::F32
        || batch == 0
        || batch > 2
        || heads != 16
        || head_dim != 64
        || query_tokens < WGPU_LONG_ATTENTION_MIN_FLASH_TOKENS
        || key_tokens < WGPU_LONG_ATTENTION_MIN_FLASH_TOKENS
    {
        return None;
    }
    let backend_name = backend_name?.to_ascii_lowercase();
    if !(backend_name.contains("wgpu") || backend_name.contains("spirv")) {
        return None;
    }
    Some(
        requested_query_chunk_tokens
            .max(1)
            .min(WGPU_DIRECT_PUBLIC_ATTENTION_QUERY_CHUNK_TOKENS)
            .min(query_tokens),
    )
}

#[cfg(target_arch = "wasm32")]
fn direct_public_attention_query_chunk_tokens(
    _dtype: FloatDType,
    _batch: usize,
    _heads: usize,
    _head_dim: usize,
    _query_tokens: usize,
    _key_tokens: usize,
    _requested_query_chunk_tokens: usize,
    _backend_name: Option<&str>,
) -> Option<usize> {
    None
}

pub fn scaled_dot_product_attention<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    head_dim: usize,
) -> Tensor<B, 4> {
    let query_chunk_tokens = default_attention_query_chunk_tokens(q.dtype().into());
    scaled_dot_product_attention_with_query_chunk_tokens(q, k, v, head_dim, query_chunk_tokens)
}

pub fn scaled_dot_product_attention_profiled<B: Backend>(
    label: &str,
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    head_dim: usize,
    query_chunk_tokens: usize,
    records: &mut Vec<TripoSplatProfileRecord>,
) -> Tensor<B, 4> {
    let device = q.device();
    let channels = q.dims()[2].saturating_mul(q.dims()[3]);
    let meta = attention_profile_meta::<B>(&q, &k, query_chunk_tokens);
    let start = Instant::now();
    let out =
        scaled_dot_product_attention_with_query_chunk_tokens(q, k, v, head_dim, query_chunk_tokens);
    push_attention_profile_record(
        records,
        label,
        channels,
        sync_elapsed_ms::<B>(&device, start),
        meta,
    );
    out
}

fn scaled_dot_product_attention_with_query_chunk_tokens<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    head_dim: usize,
    query_chunk_tokens: usize,
) -> Tensor<B, 4> {
    let [batch, query_tokens, heads, _] = q.dims();
    let key_tokens = k.dims()[1];
    let backend_name = B::name(&q.device());
    let dtype = q.dtype().into();
    let default_query_chunk_tokens = default_attention_query_chunk_tokens(dtype);
    let requested_query_chunk_tokens = query_chunk_tokens.max(1);
    let requested_default_chunking = requested_query_chunk_tokens == default_query_chunk_tokens;
    let safe_flash_requested_query_chunk_tokens = if requested_default_chunking {
        query_tokens
    } else {
        requested_query_chunk_tokens
    };
    let safe_flash_query_chunk_tokens = native_wgpu_f32_unit_flash_safe_query_chunk_tokens(
        batch,
        heads,
        dtype,
        query_tokens,
        key_tokens,
        safe_flash_requested_query_chunk_tokens,
        Some(backend_name.as_str()),
    );
    let safe_binding_query_chunk_tokens = native_wgpu_attention_binding_safe_query_chunk_tokens(
        batch,
        heads,
        dtype,
        query_tokens,
        key_tokens,
        requested_query_chunk_tokens,
        Some(backend_name.as_str()),
    );
    let use_module_flash_full = native_wgpu_f32_long_attention_prefers_module_flash(
        dtype,
        query_tokens,
        key_tokens,
        Some(backend_name.as_str()),
    ) && requested_default_chunking
        && safe_flash_query_chunk_tokens.is_none()
        && safe_binding_query_chunk_tokens.is_none();
    if use_module_flash_full {
        return scaled_dot_product_attention_dense(q, k, v, head_dim);
    }

    let mut query_chunk_tokens = resolved_attention_query_chunk_tokens(
        dtype,
        batch,
        query_tokens,
        key_tokens,
        query_chunk_tokens,
        Some(backend_name.as_str()),
    );
    if let Some(safe_flash_query_chunk_tokens) = safe_flash_query_chunk_tokens {
        query_chunk_tokens = safe_flash_query_chunk_tokens;
    }
    if let Some(safe_binding_query_chunk_tokens) = safe_binding_query_chunk_tokens {
        query_chunk_tokens = query_chunk_tokens.min(safe_binding_query_chunk_tokens);
    }
    let direct_public_query_chunk_tokens = direct_public_attention_query_chunk_tokens(
        dtype,
        batch,
        heads,
        q.dims()[3],
        query_tokens,
        key_tokens,
        query_chunk_tokens,
        Some(backend_name.as_str()),
    );
    let use_direct_primitives = direct_public_query_chunk_tokens.is_some();
    if let Some(direct_public_query_chunk_tokens) = direct_public_query_chunk_tokens {
        query_chunk_tokens = direct_public_query_chunk_tokens;
    }
    let score_elems = batch
        .saturating_mul(heads)
        .saturating_mul(query_tokens)
        .saturating_mul(key_tokens);

    let explicit_chunk_limit = query_chunk_tokens < default_query_chunk_tokens;
    if query_tokens > query_chunk_tokens
        && (explicit_chunk_limit || score_elems > ATTENTION_SCORE_ELEMS_CHUNK_THRESHOLD)
    {
        return scaled_dot_product_attention_chunked(
            q,
            k,
            v,
            head_dim,
            query_chunk_tokens,
            use_direct_primitives,
        );
    }

    if use_direct_primitives {
        return scaled_dot_product_attention_dense_direct(q, k, v, head_dim);
    }

    scaled_dot_product_attention_dense(q, k, v, head_dim)
}

fn scaled_dot_product_attention_dense<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    _head_dim: usize,
) -> Tensor<B, 4> {
    let backend_name = B::name(&q.device());
    let dtype = q.dtype();
    let float_dtype = dtype.into();
    let pad_multiple =
        native_wgpu_f16_blackbox_query_pad_multiple(float_dtype, Some(backend_name.as_str()));
    let original_query_tokens = q.dims()[1];
    let padded_query_tokens =
        padded_query_tokens(original_query_tokens, pad_multiple).max(original_query_tokens);
    let q = if padded_query_tokens > original_query_tokens {
        let [batch, _, heads, head_dim] = q.dims();
        let device = q.device();
        let pad_tokens = padded_query_tokens - original_query_tokens;
        let pad = Tensor::<B, 4>::zeros([batch, pad_tokens, heads, head_dim], &device).cast(dtype);
        Tensor::cat(vec![q, pad], 1)
    } else {
        q
    };
    let q = force_contiguous_4d(q.permute([0, 2, 1, 3]));
    let k = force_contiguous_4d(k.permute([0, 2, 1, 3]));
    let v = force_contiguous_4d(v.permute([0, 2, 1, 3]));
    let out = module_attention(q, k, v, None, None, AttentionModuleOptions::default());
    let out = out.permute([0, 2, 1, 3]);
    if padded_query_tokens > original_query_tokens {
        let [batch, _, heads, value_dim] = out.dims();
        out.slice([0..batch, 0..original_query_tokens, 0..heads, 0..value_dim])
    } else {
        out
    }
}

fn padded_query_tokens(query_tokens: usize, pad_multiple: Option<usize>) -> usize {
    let Some(pad_multiple) = pad_multiple else {
        return query_tokens;
    };
    let pad_multiple = pad_multiple.max(1);
    query_tokens.div_ceil(pad_multiple) * pad_multiple
}

fn native_wgpu_f16_blackbox_query_pad_multiple(
    dtype: FloatDType,
    backend_name: Option<&str>,
) -> Option<usize> {
    if !matches!(dtype, FloatDType::F16) {
        return None;
    }
    let Some(backend_name) = backend_name else {
        return None;
    };
    let backend_name = backend_name.to_ascii_lowercase();
    if backend_name.contains("wgpu") || backend_name.contains("spirv") {
        Some(WGPU_F16_BLACKBOX_ATTENTION_QUERY_PAD_MULTIPLE)
    } else {
        None
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn native_wgpu_attention_binding_safe_query_chunk_tokens(
    batch: usize,
    heads: usize,
    dtype: FloatDType,
    query_tokens: usize,
    key_tokens: usize,
    requested_query_chunk_tokens: usize,
    backend_name: Option<&str>,
) -> Option<usize> {
    let Some(backend_name) = backend_name else {
        return None;
    };
    let backend_name = backend_name.to_ascii_lowercase();
    if !(backend_name.contains("wgpu") || backend_name.contains("spirv")) {
        return None;
    }
    let elem_size = match dtype {
        FloatDType::F16 | FloatDType::BF16 => 2usize,
        _ => 4usize,
    };
    let bytes_per_query_token = batch
        .saturating_mul(heads)
        .saturating_mul(key_tokens)
        .saturating_mul(elem_size);
    if bytes_per_query_token == 0 {
        return None;
    }
    let requested_query_chunk_tokens = requested_query_chunk_tokens.max(1).min(query_tokens);
    let requested_bytes = bytes_per_query_token.saturating_mul(requested_query_chunk_tokens);
    if requested_bytes <= WGPU_SAFE_ATTENTION_BINDING_BYTES {
        return None;
    }
    let max_tokens = (WGPU_SAFE_ATTENTION_BINDING_BYTES / bytes_per_query_token).max(1);
    let aligned = (max_tokens / WGPU_F32_UNIT_FLASH_QUERY_CHUNK_ALIGNMENT)
        .saturating_mul(WGPU_F32_UNIT_FLASH_QUERY_CHUNK_ALIGNMENT)
        .max(1);
    Some(aligned.min(requested_query_chunk_tokens).min(query_tokens))
}

#[cfg(target_arch = "wasm32")]
fn native_wgpu_attention_binding_safe_query_chunk_tokens(
    _batch: usize,
    _heads: usize,
    _dtype: FloatDType,
    _query_tokens: usize,
    _key_tokens: usize,
    _requested_query_chunk_tokens: usize,
    _backend_name: Option<&str>,
) -> Option<usize> {
    None
}

#[cfg(test)]
fn scaled_dot_product_attention_dense_split_batch_heads<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    max_batch_heads_per_call: usize,
) -> Tensor<B, 4> {
    let [batch, query_tokens, heads, q_head_dim] = q.dims();
    let key_tokens = k.dims()[1];
    let k_head_dim = k.dims()[3];
    let value_dim = v.dims()[3];
    let total_batch_heads = batch.saturating_mul(heads);
    let max_chunk = max_batch_heads_per_call.max(1);

    let q = force_contiguous_4d(q.permute([0, 2, 1, 3])).reshape([
        total_batch_heads,
        1,
        query_tokens,
        q_head_dim,
    ]);
    let k = force_contiguous_4d(k.permute([0, 2, 1, 3])).reshape([
        total_batch_heads,
        1,
        key_tokens,
        k_head_dim,
    ]);
    let v = force_contiguous_4d(v.permute([0, 2, 1, 3])).reshape([
        total_batch_heads,
        1,
        key_tokens,
        value_dim,
    ]);
    let mut chunks = Vec::with_capacity(total_batch_heads.div_ceil(max_chunk));

    for start in (0..total_batch_heads).step_by(max_chunk) {
        let end = start.saturating_add(max_chunk).min(total_batch_heads);
        chunks.push(module_attention(
            q.clone()
                .slice([start..end, 0..1, 0..query_tokens, 0..q_head_dim]),
            k.clone()
                .slice([start..end, 0..1, 0..key_tokens, 0..k_head_dim]),
            v.clone()
                .slice([start..end, 0..1, 0..key_tokens, 0..value_dim]),
            None,
            None,
            AttentionModuleOptions::default(),
        ));
    }

    Tensor::cat(chunks, 0)
        .reshape([batch, heads, query_tokens, value_dim])
        .permute([0, 2, 1, 3])
}

fn scaled_dot_product_attention_dense_direct<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    head_dim: usize,
) -> Tensor<B, 4> {
    let [batch, query_tokens, heads, q_head_dim] = q.dims();
    let key_tokens = k.dims()[1];
    let value_dim = v.dims()[3];
    let flat_batch = batch.saturating_mul(heads);
    let q = q
        .permute([0, 2, 1, 3])
        .reshape([flat_batch, query_tokens, q_head_dim]);
    let k = k
        .permute([0, 2, 1, 3])
        .reshape([flat_batch, key_tokens, q_head_dim]);
    let v = v
        .permute([0, 2, 1, 3])
        .reshape([flat_batch, key_tokens, value_dim]);
    let scores = q
        .matmul(k.swap_dims(1, 2))
        .mul_scalar((head_dim as f64).powf(-0.5));
    softmax(scores, 2)
        .matmul(v)
        .reshape([batch, heads, query_tokens, value_dim])
        .permute([0, 2, 1, 3])
}

fn scaled_dot_product_attention_chunked<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    head_dim: usize,
    chunk_tokens: usize,
    use_direct_primitives: bool,
) -> Tensor<B, 4> {
    let [batch, query_tokens, heads, actual_head_dim] = q.dims();
    let chunk_tokens = chunk_tokens.max(1);
    let mut chunks = Vec::with_capacity(query_tokens.div_ceil(chunk_tokens));

    for start in (0..query_tokens).step_by(chunk_tokens) {
        let end = start.saturating_add(chunk_tokens).min(query_tokens);
        let q_chunk = q
            .clone()
            .slice([0..batch, start..end, 0..heads, 0..actual_head_dim]);
        let chunk = if use_direct_primitives {
            scaled_dot_product_attention_dense_direct(q_chunk, k.clone(), v.clone(), head_dim)
        } else {
            scaled_dot_product_attention_dense(q_chunk, k.clone(), v.clone(), head_dim)
        };
        chunks.push(chunk);
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
                .forward(layer_norm32_3d(&self.norm1, x.clone()), None, rope_emb);
            let x = x + attn;
            let mlp = self.mlp.forward(layer_norm32_3d(&self.norm2, x.clone()));
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
        let h = layer_norm32_3d(&self.norm1, x.clone()) * (scale_msa.unsqueeze_dim(1) + 1.0)
            + shift_msa.unsqueeze_dim(1);
        let h = self.attn.forward(h, None, rope_emb);
        let x = x + h * gate_msa.unsqueeze_dim(1);
        let h = layer_norm32_3d(&self.norm2, x.clone()) * (scale_mlp.unsqueeze_dim(1) + 1.0)
            + shift_mlp.unsqueeze_dim(1);
        x + self.mlp.forward(h) * gate_mlp.unsqueeze_dim(1)
    }

    pub fn forward_with_query_chunk_tokens(
        &self,
        x: Tensor<B, 3>,
        mod_signal: Option<Tensor<B, 2>>,
        rope_emb: Option<&RotaryAngles<B>>,
        query_chunk_tokens: usize,
    ) -> Tensor<B, 3> {
        if !self.modulation {
            let attn = self.attn.forward_with_query_chunk_tokens(
                layer_norm32_3d(&self.norm1, x.clone()),
                None,
                rope_emb,
                query_chunk_tokens,
            );
            let x = x + attn;
            let mlp = self.mlp.forward(layer_norm32_3d(&self.norm2, x.clone()));
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
        let h = layer_norm32_3d(&self.norm1, x.clone()) * (scale_msa.unsqueeze_dim(1) + 1.0)
            + shift_msa.unsqueeze_dim(1);
        let h = self
            .attn
            .forward_with_query_chunk_tokens(h, None, rope_emb, query_chunk_tokens);
        let x = x + h * gate_msa.unsqueeze_dim(1);
        let h = layer_norm32_3d(&self.norm2, x.clone()) * (scale_mlp.unsqueeze_dim(1) + 1.0)
            + shift_mlp.unsqueeze_dim(1);
        x + self.mlp.forward(h) * gate_mlp.unsqueeze_dim(1)
    }

    pub fn forward_trace_selected(
        &self,
        label: &str,
        x: Tensor<B, 3>,
        mod_signal: Option<Tensor<B, 2>>,
        rope_emb: Option<&RotaryAngles<B>>,
        token_limit: usize,
        trace: &mut Vec<(String, Tensor<B, 3>)>,
    ) -> Tensor<B, 3> {
        if !self.modulation {
            let h = layer_norm32_3d(&self.norm1, x.clone());
            push_trace_tensor3(trace, format!("{label}.norm1.out"), &h, token_limit);
            let attn = self.attn.forward(h, None, rope_emb);
            push_trace_tensor3(trace, format!("{label}.attn.out.out"), &attn, token_limit);
            let x = x + attn;
            push_trace_tensor3(trace, format!("{label}.attn_residual.out"), &x, token_limit);
            let h = layer_norm32_3d(&self.norm2, x.clone());
            push_trace_tensor3(trace, format!("{label}.norm2.out"), &h, token_limit);
            let mlp = self.mlp.forward(h);
            push_trace_tensor3(trace, format!("{label}.mlp.out"), &mlp, token_limit);
            let out = x + mlp;
            push_trace_tensor3(
                trace,
                format!("{label}.mlp_residual.out"),
                &out,
                token_limit,
            );
            return out;
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
        let h = layer_norm32_3d(&self.norm1, x.clone()) * (scale_msa.unsqueeze_dim(1) + 1.0)
            + shift_msa.unsqueeze_dim(1);
        push_trace_tensor3(trace, format!("{label}.norm1_mod.out"), &h, token_limit);
        let h = self.attn.forward(h, None, rope_emb);
        push_trace_tensor3(trace, format!("{label}.attn.out.out"), &h, token_limit);
        let x = x + h * gate_msa.unsqueeze_dim(1);
        push_trace_tensor3(trace, format!("{label}.attn_residual.out"), &x, token_limit);
        let h = layer_norm32_3d(&self.norm2, x.clone()) * (scale_mlp.unsqueeze_dim(1) + 1.0)
            + shift_mlp.unsqueeze_dim(1);
        push_trace_tensor3(trace, format!("{label}.norm2_mod.out"), &h, token_limit);
        let h = self.mlp.forward(h);
        push_trace_tensor3(trace, format!("{label}.mlp.out"), &h, token_limit);
        let out = x + h * gate_mlp.unsqueeze_dim(1);
        push_trace_tensor3(
            trace,
            format!("{label}.mlp_residual.out"),
            &out,
            token_limit,
        );
        out
    }

    pub fn forward_profiled(
        &self,
        label: &str,
        x: Tensor<B, 3>,
        mod_signal: Option<Tensor<B, 2>>,
        rope_emb: Option<&RotaryAngles<B>>,
        query_chunk_tokens: usize,
        records: &mut Vec<TripoSplatProfileRecord>,
    ) -> Tensor<B, 3> {
        self.forward_profiled_with_qkv_capture(
            label,
            x,
            mod_signal,
            rope_emb,
            query_chunk_tokens,
            records,
            None,
        )
    }

    pub fn forward_profiled_with_qkv_capture(
        &self,
        label: &str,
        x: Tensor<B, 3>,
        mod_signal: Option<Tensor<B, 2>>,
        rope_emb: Option<&RotaryAngles<B>>,
        query_chunk_tokens: usize,
        records: &mut Vec<TripoSplatProfileRecord>,
        mut qkv_capture: Option<&mut AttentionQkvCaptureState<B>>,
    ) -> Tensor<B, 3> {
        let device = x.device();
        let [batch, tokens, channels] = x.dims();
        let total_start = Instant::now();
        if !self.modulation {
            let norm1_start = Instant::now();
            let h = layer_norm32_3d(&self.norm1, x.clone());
            push_profile_record(
                records,
                format!("{label}.norm1"),
                batch,
                tokens,
                channels,
                sync_elapsed_ms::<B>(&device, norm1_start),
            );
            push_finite_debug_record(records, format!("{label}.norm1.out"), &h);
            let attn = self.attn.forward_profiled_with_qkv_capture(
                &format!("{label}.attn"),
                h,
                None,
                rope_emb,
                query_chunk_tokens,
                records,
                qkv_capture.as_deref_mut(),
            );
            let residual_start = Instant::now();
            let x = x + attn;
            push_profile_record(
                records,
                format!("{label}.attn_residual"),
                batch,
                tokens,
                channels,
                sync_elapsed_ms::<B>(&device, residual_start),
            );
            push_finite_debug_record(records, format!("{label}.attn_residual.out"), &x);
            let norm2_start = Instant::now();
            let h = layer_norm32_3d(&self.norm2, x.clone());
            push_profile_record(
                records,
                format!("{label}.norm2"),
                batch,
                tokens,
                channels,
                sync_elapsed_ms::<B>(&device, norm2_start),
            );
            push_finite_debug_record(records, format!("{label}.norm2.out"), &h);
            let mlp_start = Instant::now();
            let mlp = self.mlp.forward(h);
            push_profile_record(
                records,
                format!("{label}.mlp"),
                batch,
                tokens,
                channels,
                sync_elapsed_ms::<B>(&device, mlp_start),
            );
            push_finite_debug_record(records, format!("{label}.mlp.out"), &mlp);
            let residual_start = Instant::now();
            let out = x + mlp;
            push_profile_record(
                records,
                format!("{label}.mlp_residual"),
                batch,
                tokens,
                channels,
                sync_elapsed_ms::<B>(&device, residual_start),
            );
            push_finite_debug_record(records, format!("{label}.mlp_residual.out"), &out);
            push_profile_record(
                records,
                format!("{label}.total"),
                batch,
                tokens,
                channels,
                sync_elapsed_ms::<B>(&device, total_start),
            );
            return out;
        }
        let mut mod_signal = mod_signal.expect("modulated block requires modulation signal");
        let mod_start = Instant::now();
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
        push_profile_record(
            records,
            format!("{label}.mod"),
            batch,
            1,
            channels,
            sync_elapsed_ms::<B>(&device, mod_start),
        );
        let [batch, _] = mod_signal.dims();
        let c = self.channels;
        let shift_msa = mod_signal.clone().slice([0..batch, 0..c]);
        let scale_msa = mod_signal.clone().slice([0..batch, c..2 * c]);
        let gate_msa = mod_signal.clone().slice([0..batch, 2 * c..3 * c]);
        let shift_mlp = mod_signal.clone().slice([0..batch, 3 * c..4 * c]);
        let scale_mlp = mod_signal.clone().slice([0..batch, 4 * c..5 * c]);
        let gate_mlp = mod_signal.slice([0..batch, 5 * c..6 * c]);
        let norm1_start = Instant::now();
        let h = layer_norm32_3d(&self.norm1, x.clone()) * (scale_msa.unsqueeze_dim(1) + 1.0)
            + shift_msa.unsqueeze_dim(1);
        push_profile_record(
            records,
            format!("{label}.norm1_mod"),
            batch,
            tokens,
            channels,
            sync_elapsed_ms::<B>(&device, norm1_start),
        );
        push_finite_debug_record(records, format!("{label}.norm1_mod.out"), &h);
        let h = self.attn.forward_profiled_with_qkv_capture(
            &format!("{label}.attn"),
            h,
            None,
            rope_emb,
            query_chunk_tokens,
            records,
            qkv_capture.as_deref_mut(),
        );
        let residual_start = Instant::now();
        let x = x + h * gate_msa.unsqueeze_dim(1);
        push_profile_record(
            records,
            format!("{label}.attn_residual"),
            batch,
            tokens,
            channels,
            sync_elapsed_ms::<B>(&device, residual_start),
        );
        push_finite_debug_record(records, format!("{label}.attn_residual.out"), &x);
        let norm2_start = Instant::now();
        let h = layer_norm32_3d(&self.norm2, x.clone()) * (scale_mlp.unsqueeze_dim(1) + 1.0)
            + shift_mlp.unsqueeze_dim(1);
        push_profile_record(
            records,
            format!("{label}.norm2_mod"),
            batch,
            tokens,
            channels,
            sync_elapsed_ms::<B>(&device, norm2_start),
        );
        push_finite_debug_record(records, format!("{label}.norm2_mod.out"), &h);
        let mlp_start = Instant::now();
        let h = self.mlp.forward(h);
        push_profile_record(
            records,
            format!("{label}.mlp"),
            batch,
            tokens,
            channels,
            sync_elapsed_ms::<B>(&device, mlp_start),
        );
        push_finite_debug_record(records, format!("{label}.mlp.out"), &h);
        let residual_start = Instant::now();
        let out = x + h * gate_mlp.unsqueeze_dim(1);
        push_profile_record(
            records,
            format!("{label}.mlp_residual"),
            batch,
            tokens,
            channels,
            sync_elapsed_ms::<B>(&device, residual_start),
        );
        push_finite_debug_record(records, format!("{label}.mlp_residual.out"), &out);
        push_profile_record(
            records,
            format!("{label}.total"),
            batch,
            tokens,
            channels,
            sync_elapsed_ms::<B>(&device, total_start),
        );
        out
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
        mod_signal: Tensor<B, 2>,
        context: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let dtype: FloatDType = x.dtype().into();
        self.forward_with_query_chunk_tokens(
            x,
            mod_signal,
            context,
            default_attention_query_chunk_tokens(dtype),
        )
    }

    pub fn forward_with_query_chunk_tokens(
        &self,
        x: Tensor<B, 3>,
        mut mod_signal: Tensor<B, 2>,
        context: Tensor<B, 3>,
        query_chunk_tokens: usize,
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
        let x = x + self.cross_attn.forward_with_query_chunk_tokens(
            h,
            Some(context),
            None,
            query_chunk_tokens,
        ) * gate_msa.unsqueeze_dim(1);
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

    #[cfg(feature = "backend_wgpu")]
    #[test]
    fn pcd_legacy_wgpu_f16_uses_f32_runtime_frequencies() {
        if std::env::var("BURN_WGPU_CORRECTNESS").ok().as_deref() != Some("1") {
            eprintln!(
                "skipping pcd_legacy_wgpu_f16_uses_f32_runtime_frequencies; set BURN_WGPU_CORRECTNESS=1"
            );
            return;
        }

        type WgpuF16 = burn::backend::Wgpu<burn::tensor::f16, i32, u32>;
        let device = Default::default();
        let values = vec![
            0.123_456_7,
            0.456_789_1,
            0.987_654_3,
            0.031_25,
            0.515_625,
            0.9375,
        ];
        let x = Tensor::<WgpuF16, 2>::from_data(
            TensorData::new(values.clone(), [2, 3]),
            (&device, DType::F32),
        );
        let actual = PcdAbsolutePositionEmbedder::legacy(1024)
            .forward_2d(x)
            .cast(FloatDType::F16)
            .to_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("wgpu f16 pcd data");

        let expected = pcd_legacy_reference(values.as_slice(), 2, 3, 1024);
        let mut max_abs = 0.0f32;
        let mut mean_abs = 0.0f32;
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            let diff = (actual - expected).abs();
            max_abs = max_abs.max(diff);
            mean_abs += diff;
        }
        mean_abs /= actual.len() as f32;

        assert!(
            max_abs <= 1.0e-3 && mean_abs <= 1.0e-4,
            "pcd f16 positional embedding mismatch: max_abs={max_abs} mean_abs={mean_abs}"
        );
    }

    #[cfg(feature = "backend_wgpu")]
    fn pcd_legacy_reference(
        values: &[f32],
        tokens: usize,
        dims: usize,
        channels: usize,
    ) -> Vec<f32> {
        let freq_dim = channels / 3 / 2;
        let freqs = legacy_freqs(freq_dim, 16);
        let mut out = vec![0.0; tokens * channels];
        for token in 0..tokens {
            let mut dst = token * channels;
            for dim in 0..dims {
                let value = values[token * dims + dim];
                for &freq in freqs.iter() {
                    let phase = value * freq;
                    let reduced = phase - phase.floor();
                    out[dst] = (reduced * core::f32::consts::TAU).sin();
                    dst += 1;
                }
                for &freq in freqs.iter() {
                    let phase = value * freq;
                    let reduced = phase - phase.floor();
                    out[dst] = (reduced * core::f32::consts::TAU).cos();
                    dst += 1;
                }
            }
        }
        out
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
    fn attention_dense_matches_default_scale_reference() {
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
            force_contiguous_4d(q.permute([0, 2, 1, 3])),
            force_contiguous_4d(k.permute([0, 2, 1, 3])),
            force_contiguous_4d(v.permute([0, 2, 1, 3])),
            None,
            None,
            AttentionModuleOptions::default(),
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

        let actual =
            scaled_dot_product_attention_chunked(q.clone(), k.clone(), v.clone(), 4, 2, false);
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

    #[test]
    fn direct_primitive_attention_matches_dense_reference() {
        let device = Default::default();
        let values = |offset: f32| {
            (0..40)
                .map(|index| ((index as f32 + offset) / 23.0).cos())
                .collect::<Vec<_>>()
        };
        let q = Tensor::<TestBackend, 4>::from_data(
            TensorData::new(values(-7.0), [1, 5, 2, 4]),
            &device,
        );
        let k = Tensor::<TestBackend, 4>::from_data(
            TensorData::new(values(2.0), [1, 5, 2, 4]),
            &device,
        );
        let v = Tensor::<TestBackend, 4>::from_data(
            TensorData::new(values(17.0), [1, 5, 2, 4]),
            &device,
        );

        let actual =
            scaled_dot_product_attention_chunked(q.clone(), k.clone(), v.clone(), 4, 2, true);
        let expected = scaled_dot_product_attention_dense(q, k, v, 4);

        let actual = actual
            .to_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("direct attention data");
        let expected = expected
            .to_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("dense attention data");
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!(
                (actual - expected).abs() <= 1.0e-5,
                "direct attention mismatch: actual={actual} expected={expected}"
            );
        }
    }

    #[test]
    fn split_batch_head_attention_matches_dense_reference() {
        let device = Default::default();
        let values = |offset: f32| {
            (0..(2 * 5 * 4 * 3))
                .map(|index| ((index as f32 + offset) / 31.0).sin())
                .collect::<Vec<_>>()
        };
        let q = Tensor::<TestBackend, 4>::from_data(
            TensorData::new(values(-5.0), [2, 5, 4, 3]),
            &device,
        );
        let k = Tensor::<TestBackend, 4>::from_data(
            TensorData::new(values(3.0), [2, 5, 4, 3]),
            &device,
        );
        let v = Tensor::<TestBackend, 4>::from_data(
            TensorData::new(values(11.0), [2, 5, 4, 3]),
            &device,
        );

        let actual = scaled_dot_product_attention_dense_split_batch_heads(
            q.clone(),
            k.clone(),
            v.clone(),
            2,
        );
        let expected = scaled_dot_product_attention_dense(q, k, v, 3);

        let actual = actual
            .to_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("split attention data");
        let expected = expected
            .to_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("dense attention data");
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!(
                (actual - expected).abs() <= 1.0e-5,
                "split attention mismatch: actual={actual} expected={expected}"
            );
        }
    }

    #[test]
    fn native_wgpu_long_attention_respects_burn_default_autotune_shape() {
        let backend = Some("fusion<cubecl<wgpu<spirv>>>");
        assert_eq!(
            resolved_attention_query_chunk_tokens(
                FloatDType::F32,
                1,
                12_294,
                12_294,
                12_294,
                backend
            ),
            12_294
        );
        assert_eq!(
            resolved_attention_query_chunk_tokens(
                FloatDType::F16,
                1,
                12_294,
                12_294,
                12_294,
                backend
            ),
            12_294
        );
        assert_eq!(
            resolved_attention_query_chunk_tokens(
                FloatDType::F32,
                1,
                12_294,
                12_294,
                2048,
                backend
            ),
            2048,
            "explicit caller chunk limits remain available for diagnostics"
        );
    }

    #[test]
    fn native_attention_chunk_resolver_keeps_explicit_wgpu_diagnostic_limit() {
        let backend = Some("fusion<cubecl<wgpu<spirv>>>");
        assert_eq!(
            resolved_attention_query_chunk_tokens(
                FloatDType::F32,
                2,
                12_294,
                12_294,
                default_attention_query_chunk_tokens(FloatDType::F32),
                backend
            ),
            WGPU_BATCHED_SAFE_ATTENTION_QUERY_CHUNK_TOKENS
        );
        assert_eq!(
            resolved_attention_query_chunk_tokens(
                FloatDType::F16,
                2,
                12_294,
                12_294,
                default_attention_query_chunk_tokens(FloatDType::F16),
                backend
            ),
            WGPU_BATCHED_SAFE_ATTENTION_QUERY_CHUNK_TOKENS
        );
        assert_eq!(
            resolved_attention_query_chunk_tokens(
                FloatDType::F32,
                2,
                12_294,
                12_294,
                2048,
                backend
            ),
            2048,
            "safe explicit chunk limits remain available"
        );
        assert_eq!(ATTENTION_QUERY_CHUNK_TOKENS_F32, 4096);
        assert_eq!(ATTENTION_QUERY_CHUNK_TOKENS_F16, 4096);
        assert_eq!(WGPU_BATCHED_SAFE_ATTENTION_QUERY_CHUNK_TOKENS, 2048);
    }

    #[test]
    fn native_wgpu_f32_long_attention_uses_module_flash_policy() {
        let backend = Some("fusion<cubecl<wgpu<spirv>>>");
        assert!(native_wgpu_f32_long_attention_prefers_module_flash(
            FloatDType::F32,
            12_294,
            12_294,
            backend
        ));
        assert!(!native_wgpu_f32_long_attention_prefers_module_flash(
            FloatDType::F16,
            12_294,
            12_294,
            backend
        ));
        assert!(!native_wgpu_f32_long_attention_prefers_module_flash(
            FloatDType::F32,
            2048,
            12_294,
            backend
        ));
        assert!(!native_wgpu_f32_long_attention_prefers_module_flash(
            FloatDType::F32,
            12_294,
            12_294,
            Some("ndarray")
        ));

        assert_eq!(
            native_wgpu_f32_unit_flash_safe_query_chunk_tokens(
                1,
                15,
                FloatDType::F32,
                8192,
                8192,
                8192,
                backend
            ),
            None
        );
        assert_eq!(
            native_wgpu_f32_unit_flash_safe_query_chunk_tokens(
                1,
                16,
                FloatDType::F32,
                8192,
                8192,
                8192,
                backend
            ),
            Some(8128)
        );
        assert_eq!(
            native_wgpu_f32_unit_flash_safe_query_chunk_tokens(
                2,
                8,
                FloatDType::F32,
                8192,
                8192,
                8192,
                backend
            ),
            Some(8128)
        );
        assert_eq!(
            native_wgpu_f32_unit_flash_safe_query_chunk_tokens(
                1,
                16,
                FloatDType::F32,
                12_294,
                12_294,
                12_294,
                backend
            ),
            Some(5440)
        );
        assert_eq!(
            native_wgpu_f32_unit_flash_safe_query_chunk_tokens(
                1,
                16,
                FloatDType::F32,
                12_294,
                12_294,
                2048,
                backend
            ),
            Some(2048)
        );
        assert_eq!(
            native_wgpu_f32_unit_flash_safe_query_chunk_tokens(
                2,
                8,
                FloatDType::F32,
                8191,
                8192,
                8191,
                backend
            ),
            None
        );
    }

    #[test]
    fn native_gpu_long_attention_uses_safe_direct_public_chunk_caps() {
        let wgpu_backend = Some("fusion<cubecl<wgpu<spirv>>>");

        assert_eq!(
            direct_public_attention_query_chunk_tokens(
                FloatDType::F32,
                2,
                16,
                64,
                12_294,
                12_294,
                5440,
                wgpu_backend
            ),
            Some(WGPU_DIRECT_PUBLIC_ATTENTION_QUERY_CHUNK_TOKENS)
        );
        assert_eq!(
            direct_public_attention_query_chunk_tokens(
                FloatDType::F32,
                2,
                16,
                64,
                12_294,
                12_294,
                4096,
                Some("fusion<cubecl<cuda>>")
            ),
            None,
            "CUDA direct primitives are not a correctness-safe full-flow policy"
        );
        assert_eq!(
            direct_public_attention_query_chunk_tokens(
                FloatDType::F16,
                2,
                16,
                64,
                12_294,
                12_294,
                4096,
                Some("fusion<cubecl<cuda>>")
            ),
            None
        );
        assert_eq!(
            direct_public_attention_query_chunk_tokens(
                FloatDType::F32,
                2,
                16,
                64,
                4101,
                12_294,
                4096,
                wgpu_backend
            ),
            None
        );
    }

    #[test]
    fn native_wgpu_f16_blackbox_query_padding_is_explicit() {
        let backend = Some("fusion<cubecl<wgpu<spirv>>>");
        assert_eq!(
            native_wgpu_f16_blackbox_query_pad_multiple(FloatDType::F16, backend),
            Some(WGPU_F16_BLACKBOX_ATTENTION_QUERY_PAD_MULTIPLE)
        );
        assert_eq!(
            native_wgpu_f16_blackbox_query_pad_multiple(FloatDType::F32, backend),
            None,
            "f32 strict parity path must not inherit f16 blackbox padding"
        );
        assert_eq!(
            native_wgpu_f16_blackbox_query_pad_multiple(FloatDType::F16, Some("ndarray")),
            None
        );

        assert_eq!(padded_query_tokens(6, Some(128)), 128);
        assert_eq!(padded_query_tokens(512, Some(128)), 512);
        assert_eq!(padded_query_tokens(12_294, Some(128)), 12_416);
        assert_eq!(padded_query_tokens(12_294, None), 12_294);
    }

    #[test]
    fn native_wgpu_attention_binding_guard_chunks_oversized_score_bindings() {
        let backend = Some("fusion<cubecl<wgpu<spirv>>>");
        assert_eq!(
            native_wgpu_attention_binding_safe_query_chunk_tokens(
                1,
                40,
                FloatDType::F32,
                1526,
                23_842,
                default_attention_query_chunk_tokens(FloatDType::F32),
                backend
            ),
            Some(448)
        );
        assert_eq!(
            native_wgpu_attention_binding_safe_query_chunk_tokens(
                1,
                40,
                FloatDType::F16,
                1526,
                23_842,
                default_attention_query_chunk_tokens(FloatDType::F16),
                backend
            ),
            Some(960)
        );
        assert_eq!(
            native_wgpu_attention_binding_safe_query_chunk_tokens(
                1,
                40,
                FloatDType::F32,
                128,
                4096,
                default_attention_query_chunk_tokens(FloatDType::F32),
                backend
            ),
            None
        );
    }

    #[test]
    fn long_attention_chunk_cap_is_wgpu_specific() {
        assert_eq!(
            resolved_attention_query_chunk_tokens(FloatDType::F32, 1, 12_294, 12_294, 12_294, None),
            12_294
        );
        assert_eq!(
            resolved_attention_query_chunk_tokens(
                FloatDType::F32,
                1,
                12_294,
                12_294,
                12_294,
                Some("ndarray")
            ),
            12_294
        );
    }
}

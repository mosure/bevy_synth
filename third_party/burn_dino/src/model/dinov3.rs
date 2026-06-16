use burn::{
    module::Param,
    nn,
    nn::conv::{Conv2d, Conv2dConfig},
    prelude::*,
    tensor::{
        activation::sigmoid, module::attention as module_attention,
        ops::AttentionModuleOptions, FloatDType,
    },
};

use crate::layers::layer_norm::{LayerNorm, LayerNormConfig};

#[cfg(target_arch = "wasm32")]
const WASM_DINOV3_ATTENTION_QUERY_CHUNK_TOKENS: usize = 64;

#[derive(Config, Debug)]
pub struct DinoV3Config {
    pub image_size: usize,
    pub patch_size: usize,
    pub input_channels: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub num_register_tokens: usize,
    pub intermediate_size: usize,
    pub query_bias: bool,
    pub key_bias: bool,
    pub value_bias: bool,
    pub output_bias: bool,
    pub mlp_bias: bool,
    pub rope_theta: f32,
    pub layer_norm_eps: f64,
    pub layerscale_init: f64,
}

impl DinoV3Config {
    pub fn vit_h_16_plus(image_size: Option<usize>) -> Self {
        Self {
            image_size: image_size.unwrap_or(1024),
            patch_size: 16,
            input_channels: 3,
            hidden_size: 1280,
            num_heads: 20,
            num_layers: 32,
            num_register_tokens: 4,
            intermediate_size: 5120,
            query_bias: true,
            key_bias: false,
            value_bias: true,
            output_bias: true,
            mlp_bias: true,
            rope_theta: 100.0,
            layer_norm_eps: 1.0e-5,
            layerscale_init: 1.0,
        }
    }

    pub fn tiny_for_tests(image_size: usize, patch_size: usize) -> Self {
        Self {
            image_size,
            patch_size,
            input_channels: 3,
            hidden_size: 64,
            num_heads: 4,
            num_layers: 2,
            num_register_tokens: 2,
            intermediate_size: 192,
            query_bias: true,
            key_bias: false,
            value_bias: true,
            output_bias: true,
            mlp_bias: true,
            rope_theta: 100.0,
            layer_norm_eps: 1.0e-5,
            layerscale_init: 1.0,
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> DinoV3ViT<B> {
        DinoV3ViT::new(device, self.clone())
    }
}

#[derive(Module, Debug)]
pub struct DinoV3PatchEmbed<B: Backend> {
    pub proj: Conv2d<B>,
}

impl<B: Backend> DinoV3PatchEmbed<B> {
    pub fn new(device: &B::Device, patch_size: usize, in_channels: usize, hidden_size: usize) -> Self {
        let proj = Conv2dConfig::new([in_channels, hidden_size], [patch_size, patch_size])
            .with_stride([patch_size, patch_size])
            .with_bias(true)
            .init(device);
        Self { proj }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        self.proj.forward(x).flatten(2, 3).swap_dims(1, 2)
    }
}

#[derive(Module, Debug)]
pub struct DinoV3Attention<B: Backend> {
    pub q_proj: nn::Linear<B>,
    pub k_proj: nn::Linear<B>,
    pub v_proj: nn::Linear<B>,
    pub o_proj: nn::Linear<B>,
    num_heads: usize,
    head_dim: usize,
}

impl<B: Backend> DinoV3Attention<B> {
    pub fn new(device: &B::Device, config: &DinoV3Config) -> Self {
        let q_proj = nn::LinearConfig::new(config.hidden_size, config.hidden_size)
            .with_bias(config.query_bias)
            .init(device);
        let k_proj = nn::LinearConfig::new(config.hidden_size, config.hidden_size)
            .with_bias(config.key_bias)
            .init(device);
        let v_proj = nn::LinearConfig::new(config.hidden_size, config.hidden_size)
            .with_bias(config.value_bias)
            .init(device);
        let o_proj = nn::LinearConfig::new(config.hidden_size, config.hidden_size)
            .with_bias(config.output_bias)
            .init(device);
        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: config.num_heads,
            head_dim: config.hidden_size / config.num_heads,
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        cos: Tensor<B, 4>,
        sin: Tensor<B, 4>,
        num_prefix_tokens: usize,
    ) -> Tensor<B, 3> {
        let [batch, tokens, channels] = x.dims();
        let mut q = self
            .q_proj
            .forward(x.clone())
            .reshape([batch, tokens, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let mut k = self
            .k_proj
            .forward(x.clone())
            .reshape([batch, tokens, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = self
            .v_proj
            .forward(x)
            .reshape([batch, tokens, self.num_heads, self.head_dim])
            .swap_dims(1, 2);

        if num_prefix_tokens > 0 {
            q = apply_patch_rope(q, cos.clone(), sin.clone(), num_prefix_tokens);
            k = apply_patch_rope(k, cos, sin, num_prefix_tokens);
        } else {
            q = q.clone() * cos.clone() + rotate_half(q) * sin.clone();
            k = k.clone() * cos.clone() + rotate_half(k) * sin;
        }

        let out = dinov3_attention(q, k, v, self.head_dim);
        self.o_proj
            .forward(out.swap_dims(1, 2).reshape([batch, tokens, channels]))
    }
}

fn dinov3_attention<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    head_dim: usize,
) -> Tensor<B, 4> {
    #[cfg(target_arch = "wasm32")]
    {
        dinov3_attention_chunked(q, k, v, head_dim, WASM_DINOV3_ATTENTION_QUERY_CHUNK_TOKENS)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        dinov3_attention_dense(q, k, v, head_dim)
    }
}

fn dinov3_attention_dense<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    head_dim: usize,
) -> Tensor<B, 4> {
    module_attention(
        q,
        k,
        v,
        None,
        None,
        AttentionModuleOptions {
            scale: Some((head_dim as f64).powf(-0.5)),
            ..Default::default()
        },
    )
}

#[cfg(target_arch = "wasm32")]
fn dinov3_attention_chunked<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    head_dim: usize,
    query_chunk_tokens: usize,
) -> Tensor<B, 4> {
    let [batch, heads, query_tokens, _] = q.dims();
    if query_tokens <= query_chunk_tokens {
        return dinov3_attention_dense(q, k, v, head_dim);
    }

    let mut chunks = Vec::with_capacity(query_tokens.div_ceil(query_chunk_tokens));
    let mut start = 0;
    while start < query_tokens {
        let end = (start + query_chunk_tokens).min(query_tokens);
        let q_chunk = q
            .clone()
            .slice([0..batch, 0..heads, start..end, 0..head_dim]);
        chunks.push(dinov3_attention_dense(
            q_chunk,
            k.clone(),
            v.clone(),
            head_dim,
        ));
        start = end;
    }
    Tensor::cat(chunks, 2)
}

fn apply_patch_rope<B: Backend>(
    tokens: Tensor<B, 4>,
    cos: Tensor<B, 4>,
    sin: Tensor<B, 4>,
    num_prefix_tokens: usize,
) -> Tensor<B, 4> {
    let [batch, heads, token_count, head_dim] = tokens.dims();
    let prefix = tokens
        .clone()
        .slice([0..batch, 0..heads, 0..num_prefix_tokens, 0..head_dim]);
    let patch = tokens.slice([
        0..batch,
        0..heads,
        num_prefix_tokens..token_count,
        0..head_dim,
    ]);
    let patch = patch.clone() * cos.clone() + rotate_half(patch) * sin;
    Tensor::cat(vec![prefix, patch], 2)
}

fn rotate_half<B: Backend>(tokens: Tensor<B, 4>) -> Tensor<B, 4> {
    let [batch, heads, tokens_count, head_dim] = tokens.dims();
    let half = head_dim / 2;
    let first = tokens
        .clone()
        .slice([0..batch, 0..heads, 0..tokens_count, 0..half]);
    let second = tokens.slice([0..batch, 0..heads, 0..tokens_count, half..head_dim]);
    Tensor::cat(vec![second.mul_scalar(-1.0), first], 3)
}

#[derive(Module, Debug)]
pub struct DinoV3Mlp<B: Backend> {
    pub gate_proj: nn::Linear<B>,
    pub up_proj: nn::Linear<B>,
    pub down_proj: nn::Linear<B>,
}

impl<B: Backend> DinoV3Mlp<B> {
    pub fn new(device: &B::Device, config: &DinoV3Config) -> Self {
        let gate_proj = nn::LinearConfig::new(config.hidden_size, config.intermediate_size)
            .with_bias(config.mlp_bias)
            .init(device);
        let up_proj = nn::LinearConfig::new(config.hidden_size, config.intermediate_size)
            .with_bias(config.mlp_bias)
            .init(device);
        let down_proj = nn::LinearConfig::new(config.intermediate_size, config.hidden_size)
            .with_bias(config.mlp_bias)
            .init(device);
        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = silu(self.gate_proj.forward(x.clone()));
        let up = self.up_proj.forward(x);
        self.down_proj.forward(gate * up)
    }
}

#[derive(Module, Debug)]
pub struct DinoV3Block<B: Backend> {
    pub norm1: LayerNorm<B>,
    pub attn: DinoV3Attention<B>,
    pub ls1: Param<Tensor<B, 1>>,
    pub norm2: LayerNorm<B>,
    pub mlp: DinoV3Mlp<B>,
    pub ls2: Param<Tensor<B, 1>>,
}

impl<B: Backend> DinoV3Block<B> {
    pub fn new(device: &B::Device, config: &DinoV3Config) -> Self {
        let norm_config = LayerNormConfig::new(config.hidden_size)
            .with_epsilon(config.layer_norm_eps);
        Self {
            norm1: norm_config.clone().init(device),
            attn: DinoV3Attention::new(device, config),
            ls1: nn::Initializer::Constant {
                value: config.layerscale_init,
            }
            .init([config.hidden_size], device),
            norm2: norm_config.init(device),
            mlp: DinoV3Mlp::new(device, config),
            ls2: nn::Initializer::Constant {
                value: config.layerscale_init,
            }
            .init([config.hidden_size], device),
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        cos: Tensor<B, 4>,
        sin: Tensor<B, 4>,
        num_prefix_tokens: usize,
    ) -> Tensor<B, 3> {
        let attn = self
            .attn
            .forward(self.norm1.forward(x.clone()), cos, sin, num_prefix_tokens)
            * self.ls1.val().unsqueeze();
        let x = x + attn;
        let mlp = self.mlp.forward(self.norm2.forward(x.clone())) * self.ls2.val().unsqueeze();
        x + mlp
    }
}

#[derive(Module, Debug)]
pub struct DinoV3ViT<B: Backend> {
    pub patch_embed: DinoV3PatchEmbed<B>,
    pub cls_token: Param<Tensor<B, 3>>,
    pub register_tokens: Param<Tensor<B, 3>>,
    pub blocks: Vec<DinoV3Block<B>>,
    pub norm: LayerNorm<B>,
    patch_size: usize,
    hidden_size: usize,
    num_heads: usize,
    num_register_tokens: usize,
    rope_theta: f32,
}

impl<B: Backend> DinoV3ViT<B> {
    pub fn new(device: &B::Device, config: DinoV3Config) -> Self {
        let cls_token = nn::Initializer::Zeros.init([1, 1, config.hidden_size], device);
        let register_tokens = nn::Initializer::Zeros.init(
            [1, config.num_register_tokens, config.hidden_size],
            device,
        );
        let blocks = (0..config.num_layers)
            .map(|_| DinoV3Block::new(device, &config))
            .collect();
        Self {
            patch_embed: DinoV3PatchEmbed::new(
                device,
                config.patch_size,
                config.input_channels,
                config.hidden_size,
            ),
            cls_token,
            register_tokens,
            blocks,
            norm: LayerNormConfig::new(config.hidden_size)
                .with_epsilon(config.layer_norm_eps)
                .init(device),
            patch_size: config.patch_size,
            hidden_size: config.hidden_size,
            num_heads: config.num_heads,
            num_register_tokens: config.num_register_tokens,
            rope_theta: config.rope_theta,
        }
    }

    pub fn forward(&self, pixel_values: Tensor<B, 4>) -> Tensor<B, 3> {
        let [_batch, _channels, height, width] = pixel_values.dims();
        let patch_h = height / self.patch_size;
        let patch_w = width / self.patch_size;
        let mut x = self.patch_embed.forward(pixel_values);
        let dtype: FloatDType = x.dtype().into();
        let batch = x.dims()[0];
        let cos = rope_cos_sin::<B>(
            patch_h,
            patch_w,
            self.hidden_size / self.num_heads,
            self.rope_theta,
            &x.device(),
            true,
        )
        .cast(dtype);
        let sin = rope_cos_sin::<B>(
            patch_h,
            patch_w,
            self.hidden_size / self.num_heads,
            self.rope_theta,
            &x.device(),
            false,
        )
        .cast(dtype);
        let cls = self.cls_token.val().expand([batch as i64, -1, -1]);
        let registers = self
            .register_tokens
            .val()
            .expand([batch as i64, -1, -1]);
        x = Tensor::cat(vec![cls, registers, x], 1);
        let prefix = 1 + self.num_register_tokens;
        for block in &self.blocks {
            x = block.forward(x, cos.clone(), sin.clone(), prefix);
            cleanup_wasm_dino_memory(&x);
        }
        self.norm.forward(x)
    }
}

#[cfg(target_arch = "wasm32")]
fn cleanup_wasm_dino_memory<B: Backend, const D: usize>(tensor: &Tensor<B, D>) {
    B::memory_cleanup(&tensor.device());
}

#[cfg(not(target_arch = "wasm32"))]
fn cleanup_wasm_dino_memory<B: Backend, const D: usize>(_tensor: &Tensor<B, D>) {}

fn rope_cos_sin<B: Backend>(
    height: usize,
    width: usize,
    head_dim: usize,
    base: f32,
    device: &B::Device,
    cos: bool,
) -> Tensor<B, 4> {
    let mut values = Vec::with_capacity(height * width * head_dim);
    let inv_freq = inv_frequencies(head_dim, base);
    for y in 0..height {
        let y_coord = ((y as f32 + 0.5) / height as f32) * 2.0 - 1.0;
        for x in 0..width {
            let x_coord = ((x as f32 + 0.5) / width as f32) * 2.0 - 1.0;
            let mut half = Vec::with_capacity(head_dim / 2);
            for freq in &inv_freq {
                half.push(2.0 * core::f32::consts::PI * y_coord * *freq);
            }
            for freq in &inv_freq {
                half.push(2.0 * core::f32::consts::PI * x_coord * *freq);
            }
            for angle in half.iter().chain(half.iter()) {
                values.push(if cos { angle.cos() } else { angle.sin() });
            }
        }
    }
    Tensor::<B, 1>::from_floats(values.as_slice(), device).reshape([
        1,
        1,
        (height * width) as i32,
        head_dim as i32,
    ])
}

fn inv_frequencies(head_dim: usize, base: f32) -> Vec<f32> {
    let freq_count = head_dim / 4;
    (0..freq_count)
        .map(|i| {
            let exponent = (4 * i) as f32 / head_dim as f32;
            1.0 / base.powf(exponent)
        })
        .collect()
}

fn silu<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    x.clone() * sigmoid(x)
}

#[cfg(feature = "import")]
pub mod import {
    use std::{
        fs,
        path::{Path, PathBuf},
    };

    use burn::{
        module::{Module, ModuleMapper, Param},
        prelude::*,
        tensor::{Bytes, FloatDType},
    };
    use burn_store::{
        BurnpackStore, KeyRemapper, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore,
    };
    use safetensors::{
        Dtype, SafeTensors, serialize,
        tensor::{SafeTensorError, TensorView},
    };

    use super::{DinoV3Config, DinoV3ViT};

    pub fn load_dinov3_from_safetensors<B: Backend>(
        device: &B::Device,
        path: impl AsRef<Path>,
        config: &DinoV3Config,
    ) -> Result<DinoV3ViT<B>, Box<dyn std::error::Error>> {
        let mut model = config.clone().init(device);
        let mut store = build_store(path.as_ref())?;
        model
            .load_from(&mut store)
            .map_err(|err| format!("failed to load DINOv3 weights: {err}"))?;
        Ok(model)
    }

    pub fn load_dinov3_from_burnpack_file<B: Backend>(
        device: &B::Device,
        burnpack_path: impl AsRef<Path>,
        config: &DinoV3Config,
    ) -> Result<DinoV3ViT<B>, Box<dyn std::error::Error>> {
        let mut model = config.clone().init(device);
        let mut store =
            BurnpackStore::from_file(burnpack_path.as_ref()).validate(should_validate_burnpack());
        model
            .load_from(&mut store)
            .map_err(|err| format!("failed to load DINOv3 burnpack: {err}"))?;
        Ok(model)
    }

    pub fn apply_dinov3_burnpack_part_bytes<B: Backend>(
        model: &mut DinoV3ViT<B>,
        burnpack_bytes: Vec<u8>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut store =
            BurnpackStore::from_bytes(Some(Bytes::from_bytes_vec(burnpack_bytes)))
                .allow_partial(true)
                .validate(should_validate_burnpack());
        model
            .load_from(&mut store)
            .map_err(|err| format!("failed to apply DINOv3 burnpack part: {err}"))?;
        Ok(())
    }

    pub fn import_dinov3_burnpack_to_path<B: Backend>(
        device: &B::Device,
        source_path: impl AsRef<Path>,
        output_path: impl AsRef<Path>,
        config: &DinoV3Config,
        use_f16: bool,
    ) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let mut model = load_dinov3_from_safetensors::<B>(device, source_path, config)?;
        let dtype = if use_f16 {
            FloatDType::F16
        } else {
            FloatDType::F32
        };
        model = cast_module_float_dtype(model, dtype);
        save_burnpack(&model, output_path.as_ref())?;
        Ok(output_path.as_ref().to_path_buf())
    }

    pub fn dinov3_key_remap_rules() -> &'static [(&'static str, &'static str)] {
        &[
            (r"^embeddings\.patch_embeddings\.(.+)$", "patch_embed.proj.$1"),
            (r"^embeddings\.cls_token$", "cls_token"),
            (r"^embeddings\.register_tokens$", "register_tokens"),
            (r"^encoder\.layer\.(\d+)\.(.+)$", "layer.$1.$2"),
            (
                r"^layer\.(\d+)\.attention\.(q_proj|k_proj|v_proj|o_proj)\.(.+)$",
                "blocks.$1.attn.$2.$3",
            ),
            (
                r"^layer\.(\d+)\.layer_scale1\.lambda1$",
                "blocks.$1.ls1",
            ),
            (
                r"^layer\.(\d+)\.layer_scale2\.lambda1$",
                "blocks.$1.ls2",
            ),
            (r"^layer\.(\d+)\.norm([12])\.weight$", "blocks.$1.norm$2.gamma"),
            (r"^layer\.(\d+)\.norm([12])\.bias$", "blocks.$1.norm$2.beta"),
            (r"^layer\.(\d+)\.mlp\.(.+)$", "blocks.$1.mlp.$2"),
            (r"^(blocks\.\d+\.norm[12])\.weight$", "$1.gamma"),
            (r"^(blocks\.\d+\.norm[12])\.bias$", "$1.beta"),
            (r"^norm\.weight$", "norm.gamma"),
            (r"^norm\.bias$", "norm.beta"),
        ]
    }

    fn build_store(path: &Path) -> Result<SafetensorsStore, Box<dyn std::error::Error>> {
        let mut remapper = KeyRemapper::new();
        for &(from, to) in dinov3_key_remap_rules() {
            remapper = remapper
                .add_pattern(from, to)
                .map_err(|err| format!("invalid DINOv3 remap rule {from}->{to}: {err}"))?;
        }
        let store = match normalize_bf16_safetensors_for_ndarray(path)? {
            Some(bytes) => SafetensorsStore::from_bytes(Some(bytes)),
            None => SafetensorsStore::from_file(path),
        };
        Ok(store
            .with_from_adapter(PyTorchToBurnAdapter)
            .allow_partial(false)
            .remap(remapper)
            .validate(true))
    }

    struct OwnedTensor {
        name: String,
        shape: Vec<usize>,
        dtype: Dtype,
        data: Vec<u8>,
    }

    fn normalize_bf16_safetensors_for_ndarray(
        path: &Path,
    ) -> Result<Option<Vec<u8>>, Box<dyn std::error::Error>> {
        let source = fs::read(path)?;
        let tensors = SafeTensors::deserialize(&source)?;
        let has_bf16 = tensors
            .names()
            .iter()
            .any(|name| tensors.tensor(name).is_ok_and(|view| view.dtype() == Dtype::BF16));
        if !has_bf16 {
            return Ok(None);
        }

        let mut owned = Vec::with_capacity(tensors.names().len());
        for name in tensors.names() {
            let view = tensors.tensor(name)?;
            let (dtype, data) = if view.dtype() == Dtype::BF16 {
                (Dtype::F32, bf16_bytes_to_f32_bytes(view.data())?)
            } else {
                (view.dtype(), view.data().to_vec())
            };
            owned.push(OwnedTensor {
                name: name.to_string(),
                shape: view.shape().to_vec(),
                dtype,
                data,
            });
        }

        let views = owned
            .iter()
            .map(|tensor| {
                TensorView::new(tensor.dtype, tensor.shape.clone(), tensor.data.as_slice())
                    .map(|view| (tensor.name.as_str(), view))
            })
            .collect::<Result<Vec<_>, SafeTensorError>>()?;
        Ok(Some(serialize(views, None)?))
    }

    fn bf16_bytes_to_f32_bytes(bytes: &[u8]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        if bytes.len() % 2 != 0 {
            return Err(format!(
                "invalid BF16 tensor byte length {}; expected an even number of bytes",
                bytes.len()
            )
            .into());
        }
        let mut out = Vec::with_capacity(bytes.len() * 2);
        for chunk in bytes.chunks_exact(2) {
            let bf16 = u16::from_le_bytes([chunk[0], chunk[1]]);
            let value = f32::from_bits((bf16 as u32) << 16);
            out.extend_from_slice(&value.to_le_bytes());
        }
        Ok(out)
    }

    struct FloatDTypeMapper {
        dtype: FloatDType,
    }

    impl<B: Backend> ModuleMapper<B> for FloatDTypeMapper {
        fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
            let (id, tensor, mapper) = param.consume();
            Param::from_mapped_value(id, tensor.cast(self.dtype), mapper)
        }
    }

    fn cast_module_float_dtype<B: Backend, M: Module<B>>(module: M, dtype: FloatDType) -> M {
        let mut mapper = FloatDTypeMapper { dtype };
        module.map(&mut mapper)
    }

    fn save_burnpack<B: Backend>(
        model: &DinoV3ViT<B>,
        path: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut store = BurnpackStore::from_file(path).overwrite(true);
        model
            .save_into(&mut store)
            .map_err(|err| format!("failed to save DINOv3 burnpack: {err}"))?;
        Ok(())
    }

    fn should_validate_burnpack() -> bool {
        cfg!(all(not(target_arch = "wasm32"), debug_assertions))
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::time::{SystemTime, UNIX_EPOCH};

        #[test]
        fn bf16_safetensors_are_normalized_to_f32_for_ndarray_import() {
            let root = std::env::temp_dir().join(format!(
                "burn_dino_dinov3_bf16_normalize_{}",
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .expect("clock should be after unix epoch")
                    .as_nanos()
            ));
            fs::create_dir_all(&root).expect("create temp root");
            let path = root.join("dinov3_bf16.safetensors");
            let bf16 = vec![0x80, 0x3f, 0x00, 0xc0];
            let view = TensorView::new(Dtype::BF16, vec![2], bf16.as_slice())
                .expect("bf16 tensor view");
            let source =
                serialize(vec![("weight".to_string(), view)], None).expect("serialize bf16");
            fs::write(&path, source).expect("write source");

            let normalized = normalize_bf16_safetensors_for_ndarray(&path)
                .expect("normalize")
                .expect("bf16 should normalize");
            let parsed = SafeTensors::deserialize(normalized.as_slice()).expect("parse normalized");
            let tensor = parsed.tensor("weight").expect("weight tensor");
            assert_eq!(tensor.dtype(), Dtype::F32);
            assert_eq!(tensor.shape(), &[2]);
            let values = tensor
                .data()
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect::<Vec<_>>();
            assert_eq!(values, vec![1.0, -2.0]);

            fs::remove_dir_all(root).expect("cleanup temp root");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestBackend = burn::backend::NdArray<f32>;

    #[test]
    fn dinov3_tiny_forward_shape_matches_prefix_plus_patches() {
        let device = Default::default();
        let config = DinoV3Config::tiny_for_tests(32, 16);
        let model = config.clone().init::<TestBackend>(&device);
        let input = Tensor::<TestBackend, 4>::zeros([1, 3, 32, 32], &device);
        let out = model.forward(input);
        assert_eq!(
            out.dims(),
            [
                1,
                1 + config.num_register_tokens + (config.image_size / config.patch_size).pow(2),
                config.hidden_size
            ]
        );
    }

    #[test]
    fn dinov3_vith_config_matches_triposplat_reference() {
        let config = DinoV3Config::vit_h_16_plus(None);
        assert_eq!(config.hidden_size, 1280);
        assert_eq!(config.num_heads, 20);
        assert_eq!(config.num_layers, 32);
        assert_eq!(config.patch_size, 16);
        assert_eq!(config.num_register_tokens, 4);
        assert_eq!(config.intermediate_size, 5120);
        assert!(config.query_bias);
        assert!(!config.key_bias);
        assert!(config.value_bias);
    }
}

#[cfg(all(test, feature = "import"))]
mod import_tests {
    use super::import::dinov3_key_remap_rules;

    fn remap(key: &str) -> String {
        let mut remapper = burn_store::KeyRemapper::new();
        for &(from, to) in dinov3_key_remap_rules() {
            remapper = remapper.add_pattern(from, to).unwrap();
        }
        let mut out = key.to_string();
        for (pattern, replacement) in &remapper.patterns {
            if pattern.is_match(&out) {
                out = pattern.replace_all(&out, replacement.as_str()).to_string();
            }
        }
        out
    }

    #[test]
    fn dinov3_remaps_hf_attention_and_norm_keys() {
        assert_eq!(
            remap("encoder.layer.3.attention.q_proj.weight"),
            "blocks.3.attn.q_proj.weight"
        );
        assert_eq!(
            remap("encoder.layer.3.norm1.weight"),
            "blocks.3.norm1.gamma"
        );
        assert_eq!(remap("embeddings.cls_token"), "cls_token");
    }
}

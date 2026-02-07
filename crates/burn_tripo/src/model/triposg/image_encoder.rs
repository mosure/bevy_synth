use burn::prelude::*;
use burn::tensor::module::interpolate;
use burn::tensor::ops::{InterpolateMode, InterpolateOptions};
use image::{ImageBuffer, Rgb};

use burn_dino::model::dino::{DinoOutput, DinoVisionTransformer};

#[derive(Module, Debug)]
pub struct TripoSGImageEncoder<B: Backend> {
    pub dino: DinoVisionTransformer<B>,
}

impl<B: Backend> TripoSGImageEncoder<B> {
    pub fn new(dino: DinoVisionTransformer<B>) -> Self {
        Self { dino }
    }

    pub fn forward(&self, image: Tensor<B, 4>) -> Tensor<B, 3> {
        let output: DinoOutput<B> = self.dino.forward(image, None);
        let cls = output.x_norm_clstoken.unsqueeze_dim(1);
        Tensor::cat(vec![cls, output.x_norm_patchtokens], 1)
    }
}

#[derive(Debug, Clone)]
pub struct DinoImageProcessor {
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub rescale_factor: f32,
    pub do_rescale: bool,
    pub do_normalize: bool,
    pub do_resize: bool,
    pub size_shortest_edge: Option<usize>,
    pub do_center_crop: bool,
    pub crop_size: Option<[usize; 2]>,
    pub resize_mode: InterpolateMode,
}

impl Default for DinoImageProcessor {
    fn default() -> Self {
        Self {
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
            rescale_factor: 1.0 / 255.0,
            do_rescale: true,
            do_normalize: true,
            do_resize: false,
            size_shortest_edge: None,
            do_center_crop: false,
            crop_size: None,
            resize_mode: InterpolateMode::Bicubic,
        }
    }
}

impl DinoImageProcessor {
    pub fn preprocess<B: Backend>(&self, image: Tensor<B, 4>) -> Tensor<B, 4> {
        if !cfg!(target_arch = "wasm32") && std::env::var("DINO_STRICT_PREPROCESS").is_ok() {
            return self.preprocess_cpu(image);
        }
        let mut image = image;

        if self.do_resize
            && let Some(shortest_edge) = self.size_shortest_edge
        {
            let [_, _, height, width] = image.shape().dims();
            let min_edge = height.min(width);
            if min_edge > 0 && min_edge != shortest_edge {
                let scale = shortest_edge as f32 / min_edge as f32;
                let new_height = (height as f32 * scale).round() as usize;
                let new_width = (width as f32 * scale).round() as usize;
                let options = InterpolateOptions {
                    mode: self.resize_mode.clone(),
                };
                image = interpolate(image, [new_height, new_width], options);
            }
        }

        if self.do_center_crop
            && let Some([crop_height, crop_width]) = self.crop_size
        {
            let [batch, channels, height, width] = image.shape().dims();
            if height >= crop_height && width >= crop_width {
                let top = (height - crop_height) / 2;
                let left = (width - crop_width) / 2;
                image = image.slice([
                    0..batch,
                    0..channels,
                    top..(top + crop_height),
                    left..(left + crop_width),
                ]);
            }
        }

        if self.do_rescale {
            image = image.mul_scalar(self.rescale_factor);
        }

        if self.do_normalize {
            let device = image.device();
            let mean = Tensor::<B, 1>::from_floats(self.mean, &device).reshape([1, 3, 1, 1]);
            let std = Tensor::<B, 1>::from_floats(self.std, &device).reshape([1, 3, 1, 1]);
            image = image.sub(mean).div(std);
        }

        image
    }

    fn preprocess_cpu<B: Backend>(&self, image: Tensor<B, 4>) -> Tensor<B, 4> {
        let device = image.device();
        let [batch, channels, height, width] = image.shape().dims();
        let data = image
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("failed to read image tensor data");

        let mut output = Vec::new();
        let mut final_height = None;
        let mut final_width = None;
        let image_stride = channels * height * width;

        for b in 0..batch {
            let start = b * image_stride;
            let end = start + image_stride;
            let chw = &data[start..end];

            let mut hwc = Vec::with_capacity(height * width * 3);
            for y in 0..height {
                for x in 0..width {
                    for c in 0..3 {
                        let idx = c * height * width + y * width + x;
                        let value = chw[idx].clamp(0.0, 255.0) as u8;
                        hwc.push(value);
                    }
                }
            }

            let mut image = ImageBuffer::<Rgb<u8>, _>::from_vec(width as u32, height as u32, hwc)
                .expect("invalid image buffer");

            if self.do_resize
                && let Some(shortest) = self.size_shortest_edge
            {
                let (in_w, in_h) = (image.width() as usize, image.height() as usize);
                let (short, long) = if in_w <= in_h {
                    (in_w, in_h)
                } else {
                    (in_h, in_w)
                };
                if short > 0 && short != shortest {
                    let new_short = shortest;
                    let new_long = (new_short as f32 * long as f32 / short as f32) as usize;
                    let (new_h, new_w) = if in_w <= in_h {
                        (new_long, new_short)
                    } else {
                        (new_short, new_long)
                    };
                    image = image::imageops::resize(
                        &image,
                        new_w as u32,
                        new_h as u32,
                        image::imageops::FilterType::CatmullRom,
                    );
                }
            }

            if self.do_center_crop
                && let Some([crop_h, crop_w]) = self.crop_size
            {
                let (in_w, in_h) = (image.width() as usize, image.height() as usize);
                if in_h >= crop_h && in_w >= crop_w {
                    let top = (in_h - crop_h) / 2;
                    let left = (in_w - crop_w) / 2;
                    let cropped = image::imageops::crop_imm(
                        &image,
                        left as u32,
                        top as u32,
                        crop_w as u32,
                        crop_h as u32,
                    );
                    image = cropped.to_image();
                }
            }

            let (out_w, out_h) = (image.width() as usize, image.height() as usize);
            match (final_height, final_width) {
                (Some(h), Some(w)) => {
                    if h != out_h || w != out_w {
                        panic!(
                            "DINO preprocess produced inconsistent sizes: {h}x{w} vs {out_h}x{out_w}"
                        );
                    }
                }
                _ => {
                    final_height = Some(out_h);
                    final_width = Some(out_w);
                }
            }
            let pixels = out_h * out_w;
            let mut out_chw = vec![0.0f32; pixels * 3];
            for (idx, pixel) in image.pixels().enumerate() {
                let [r, g, b] = pixel.0;
                out_chw[idx] = r as f32;
                out_chw[pixels + idx] = g as f32;
                out_chw[pixels * 2 + idx] = b as f32;
            }

            if self.do_rescale {
                for value in &mut out_chw {
                    *value *= self.rescale_factor;
                }
            }

            if self.do_normalize {
                for c in 0..3 {
                    let mean = self.mean[c];
                    let std = self.std[c];
                    let offset = c * pixels;
                    for idx in 0..pixels {
                        let value = out_chw[offset + idx];
                        out_chw[offset + idx] = (value - mean) / std;
                    }
                }
            }

            output.extend(out_chw);
        }

        let flat = Tensor::<B, 1>::from_floats(output.as_slice(), &device);
        let out_height = final_height.unwrap_or(height);
        let out_width = final_width.unwrap_or(width);
        flat.reshape([batch as i32, 3, out_height as i32, out_width as i32])
    }
}

#[cfg(feature = "import")]
pub mod import {
    use std::{
        collections::BTreeMap,
        fs,
        path::{Path, PathBuf},
    };

    use burn::module::{Module, ModuleMapper, Param};
    use burn::prelude::*;
    use burn::tensor::Bytes;
    use burn::tensor::FloatDType;
    use burn::tensor::ops::InterpolateMode;
    use burn_store::{
        BurnpackStore, KeyRemapper, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore,
    };
    use safetensors::{
        Dtype, serialize,
        tensor::{SafeTensors, TensorView},
    };

    use super::{DinoImageProcessor, TripoSGImageEncoder};
    use burn_dino::model::dino::DinoVisionTransformerConfig;

    const F16_SUFFIX: &str = "_f16";

    #[derive(Debug)]
    pub struct Dinov2ImportError(pub String);

    impl std::fmt::Display for Dinov2ImportError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "Dinov2 import error: {}", self.0)
        }
    }

    impl std::error::Error for Dinov2ImportError {}

    pub fn load_triposg_dinov2<B: Backend>(
        device: &B::Device,
        weights_path: impl AsRef<Path>,
    ) -> Result<TripoSGImageEncoder<B>, Box<dyn std::error::Error>> {
        let weights_path = weights_path.as_ref();
        let mut config = load_dinov2_config(weights_path)
            .unwrap_or_else(|| DinoVisionTransformerConfig::vitl(None, None));
        if let Some(target_size) = load_dinov2_preprocess_size(weights_path) {
            let patch = config.patch_size.max(1);
            let grid = target_size / patch;
            if grid > 0 {
                config.positional_encoding_interpolate.output_size = Some([grid, grid]);
            }
        }
        let burnpack_candidates = candidate_burnpack_paths(weights_path);
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

        let mut model: burn_dino::model::dino::DinoVisionTransformer<B> =
            burn_dino::model::dino::DinoVisionTransformer::new(device, config);
        let mut store = BurnpackStore::from_file(&burnpack_path).validate(true);
        model
            .load_from(&mut store)
            .map_err(|err| format!("failed to load dinov2 burnpack: {err}"))?;

        Ok(TripoSGImageEncoder::new(model))
    }

    pub fn load_triposg_dinov2_from_burnpack_bytes<B: Backend>(
        device: &B::Device,
        config: DinoVisionTransformerConfig,
        burnpack_bytes: Vec<u8>,
    ) -> Result<TripoSGImageEncoder<B>, Box<dyn std::error::Error>> {
        let mut model: burn_dino::model::dino::DinoVisionTransformer<B> =
            burn_dino::model::dino::DinoVisionTransformer::new(device, config);
        let mut store =
            BurnpackStore::from_bytes(Some(Bytes::from_bytes_vec(burnpack_bytes))).validate(true);
        model
            .load_from(&mut store)
            .map_err(|err| format!("failed to load dinov2 burnpack bytes: {err}"))?;
        Ok(TripoSGImageEncoder::new(model))
    }

    pub fn load_dinov2_processor(
        weights_root: impl AsRef<Path>,
    ) -> Result<DinoImageProcessor, Box<dyn std::error::Error>> {
        let root = weights_root.as_ref();
        let path = root.join("feature_extractor_dinov2/preprocessor_config.json");
        let bytes = fs::read(path)?;
        let fallback_size = load_dinov2_image_size(root);
        load_dinov2_processor_from_json_bytes(&bytes, fallback_size)
    }

    pub fn load_dinov2_processor_from_json_bytes(
        bytes: &[u8],
        fallback_size: Option<usize>,
    ) -> Result<DinoImageProcessor, Box<dyn std::error::Error>> {
        let config: Dinov2ProcessorConfig = serde_json::from_slice(&bytes)?;
        let resize_mode = match config.resample.unwrap_or(3) {
            3 => InterpolateMode::Bicubic,
            2 => InterpolateMode::Bilinear,
            _ => InterpolateMode::Nearest,
        };

        let mut processor = DinoImageProcessor {
            mean: config.image_mean.unwrap_or([0.485, 0.456, 0.406]),
            std: config.image_std.unwrap_or([0.229, 0.224, 0.225]),
            rescale_factor: config.rescale_factor.unwrap_or(1.0 / 255.0),
            do_rescale: config.do_rescale.unwrap_or(true),
            do_normalize: config.do_normalize.unwrap_or(true),
            do_resize: config.do_resize.unwrap_or(false),
            size_shortest_edge: config.size.as_ref().and_then(|size| size.shortest_edge),
            do_center_crop: config.do_center_crop.unwrap_or(false),
            crop_size: config.crop_size.map(|size| [size.height, size.width]),
            resize_mode,
        };

        if processor.size_shortest_edge.is_none()
            && processor.crop_size.is_none()
            && let Some(target_size) = fallback_size
        {
            processor.do_resize = true;
            processor.size_shortest_edge = Some(target_size);
            processor.do_center_crop = true;
            processor.crop_size = Some([target_size, target_size]);
        }

        Ok(processor)
    }

    #[derive(serde::Deserialize)]
    struct Dinov2ProcessorConfig {
        image_mean: Option<[f32; 3]>,
        image_std: Option<[f32; 3]>,
        rescale_factor: Option<f32>,
        do_rescale: Option<bool>,
        do_normalize: Option<bool>,
        do_resize: Option<bool>,
        do_center_crop: Option<bool>,
        resample: Option<i64>,
        size: Option<Dinov2SizeConfig>,
        crop_size: Option<Dinov2CropConfig>,
    }

    #[derive(serde::Deserialize)]
    struct Dinov2SizeConfig {
        shortest_edge: Option<usize>,
    }

    #[derive(serde::Deserialize)]
    struct Dinov2CropConfig {
        height: usize,
        width: usize,
    }

    fn load_dinov2_image_size(weights_root: &Path) -> Option<usize> {
        let config_path = weights_root.join("image_encoder_dinov2/config.json");
        let bytes = fs::read(config_path).ok()?;
        load_dinov2_image_size_from_json_bytes(&bytes)
    }

    fn load_dinov2_preprocess_size(weights_path: &Path) -> Option<usize> {
        let weights_root = weights_path.parent()?.parent()?;
        let config_path = weights_root.join("feature_extractor_dinov2/preprocessor_config.json");
        let bytes = fs::read(config_path).ok()?;
        load_dinov2_preprocess_size_from_json_bytes(&bytes)
    }

    pub fn load_dinov2_preprocess_size_from_json_bytes(bytes: &[u8]) -> Option<usize> {
        let config: Dinov2ProcessorConfig = serde_json::from_slice(bytes).ok()?;
        if config.do_center_crop.unwrap_or(false)
            && let Some(crop) = config.crop_size
        {
            return Some(crop.height.min(crop.width));
        }
        if config.do_resize.unwrap_or(false)
            && let Some(size) = config.size.and_then(|size| size.shortest_edge)
        {
            return Some(size);
        }
        None
    }

    fn load_dinov2_config(weights_path: &Path) -> Option<DinoVisionTransformerConfig> {
        let config_path = weights_path.parent()?.join("config.json");
        let bytes = fs::read(config_path).ok()?;
        load_dinov2_config_from_json_bytes(&bytes)
    }

    fn load_dinov2_image_size_from_json_bytes(bytes: &[u8]) -> Option<usize> {
        let config: Dinov2Config = serde_json::from_slice(bytes).ok()?;
        config.image_size
    }

    pub fn default_dinov2_config() -> DinoVisionTransformerConfig {
        DinoVisionTransformerConfig::vitl(None, None)
    }

    pub fn load_dinov2_config_from_json_bytes(bytes: &[u8]) -> Option<DinoVisionTransformerConfig> {
        let config: Dinov2Config = serde_json::from_slice(bytes).ok()?;
        let image_size = config.image_size.unwrap_or(518);
        let patch_size = config.patch_size.unwrap_or(14);
        let mut dino = DinoVisionTransformerConfig::vitl(Some(image_size), Some(patch_size));
        if let Some(channels) = config.num_channels {
            dino.input_channels = channels;
        }
        Some(dino)
    }

    #[derive(serde::Deserialize)]
    struct Dinov2Config {
        image_size: Option<usize>,
        patch_size: Option<usize>,
        num_channels: Option<usize>,
    }

    fn build_store(bytes: Vec<u8>) -> Result<SafetensorsStore, Box<dyn std::error::Error>> {
        let mut remapper = KeyRemapper::new();
        for &(from, to) in key_remap_rules() {
            remapper = remapper
                .add_pattern(from, to)
                .map_err(|err| format!("invalid remap rule {from}->{to}: {err}"))?;
        }
        let store = SafetensorsStore::from_bytes(Some(bytes))
            .with_from_adapter(PyTorchToBurnAdapter)
            .allow_partial(true)
            .remap(remapper)
            .validate(true);
        Ok(store)
    }

    fn key_remap_rules() -> &'static [(&'static str, &'static str)] {
        &[
            (r"^(blocks\.\d+\.norm\d?)\.weight$", "$1.gamma"),
            (r"^(blocks\.\d+\.norm\d?)\.bias$", "$1.beta"),
            (r"^(norm)\.weight$", "$1.gamma"),
            (r"^(norm)\.bias$", "$1.beta"),
        ]
    }

    #[derive(Default)]
    struct QkvParts {
        q_weight: Option<Vec<f32>>,
        k_weight: Option<Vec<f32>>,
        v_weight: Option<Vec<f32>>,
        q_bias: Option<Vec<f32>>,
        k_bias: Option<Vec<f32>>,
        v_bias: Option<Vec<f32>>,
        dim: Option<usize>,
    }

    fn convert_hf_dinov2(weights_path: &Path) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let bytes = fs::read(weights_path)?;
        let tensors = SafeTensors::deserialize(&bytes)?;

        let mut owned = Vec::<OwnedTensor>::new();
        let mut qkv_parts: BTreeMap<usize, QkvParts> = BTreeMap::new();

        for name in tensors.names() {
            let view = tensors.tensor(name)?;
            if let Some(mapped_name) = map_tensor_name(name, &mut qkv_parts, &view)? {
                let data = view.data().to_vec();
                owned.push(OwnedTensor {
                    name: mapped_name,
                    shape: view.shape().to_vec(),
                    dtype: view.dtype(),
                    data,
                });
            }
        }

        for (layer, parts) in qkv_parts {
            let q = parts
                .q_weight
                .ok_or_else(|| Dinov2ImportError(format!("missing q weight for layer {layer}")))?;
            let k = parts
                .k_weight
                .ok_or_else(|| Dinov2ImportError(format!("missing k weight for layer {layer}")))?;
            let v = parts
                .v_weight
                .ok_or_else(|| Dinov2ImportError(format!("missing v weight for layer {layer}")))?;
            let dim = parts
                .dim
                .ok_or_else(|| Dinov2ImportError(format!("missing dim for layer {layer}")))?;
            let mut qkv = Vec::with_capacity(q.len() + k.len() + v.len());
            qkv.extend_from_slice(&q);
            qkv.extend_from_slice(&k);
            qkv.extend_from_slice(&v);
            owned.push(OwnedTensor {
                name: format!("blocks.{layer}.attn.qkv.weight"),
                shape: vec![dim * 3, dim],
                dtype: Dtype::F32,
                data: bytemuck::cast_slice(&qkv).to_vec(),
            });

            let qb = parts
                .q_bias
                .ok_or_else(|| Dinov2ImportError(format!("missing q bias for layer {layer}")))?;
            let kb = parts
                .k_bias
                .ok_or_else(|| Dinov2ImportError(format!("missing k bias for layer {layer}")))?;
            let vb = parts
                .v_bias
                .ok_or_else(|| Dinov2ImportError(format!("missing v bias for layer {layer}")))?;
            let mut qkv_bias = Vec::with_capacity(qb.len() + kb.len() + vb.len());
            qkv_bias.extend_from_slice(&qb);
            qkv_bias.extend_from_slice(&kb);
            qkv_bias.extend_from_slice(&vb);
            owned.push(OwnedTensor {
                name: format!("blocks.{layer}.attn.qkv.bias"),
                shape: vec![dim * 3],
                dtype: Dtype::F32,
                data: bytemuck::cast_slice(&qkv_bias).to_vec(),
            });
        }

        let views: Vec<(String, TensorView)> = owned
            .iter()
            .map(|tensor| {
                let view =
                    TensorView::new(tensor.dtype, tensor.shape.clone(), tensor.data.as_slice())
                        .expect("invalid tensor view");
                (tensor.name.clone(), view)
            })
            .collect();

        let data = serialize(views, None)?;
        Ok(data)
    }

    fn map_tensor_name(
        name: &str,
        qkv_parts: &mut BTreeMap<usize, QkvParts>,
        view: &TensorView<'_>,
    ) -> Result<Option<String>, Box<dyn std::error::Error>> {
        let mapped = match name {
            "embeddings.cls_token" => Some("cls_token".to_string()),
            "embeddings.mask_token" => Some("mask_token".to_string()),
            "embeddings.position_embeddings" => Some("pos_embed".to_string()),
            "embeddings.patch_embeddings.projection.weight" => {
                Some("patch_embed.proj.weight".to_string())
            }
            "embeddings.patch_embeddings.projection.bias" => {
                Some("patch_embed.proj.bias".to_string())
            }
            "layernorm.weight" => Some("norm.weight".to_string()),
            "layernorm.bias" => Some("norm.bias".to_string()),
            _ => None,
        };

        if mapped.is_some() {
            return Ok(mapped);
        }

        let parts: Vec<&str> = name.split('.').collect();
        if parts.len() < 4 {
            return Ok(None);
        }
        if parts[0] != "encoder" || parts[1] != "layer" {
            return Ok(None);
        }
        let layer: usize = parts[2]
            .parse()
            .map_err(|_| Dinov2ImportError(format!("invalid layer index in {name}")))?;

        match parts[3] {
            "norm1" | "norm2" => {
                let suffix = parts.get(4).copied().unwrap_or("");
                Ok(Some(format!("blocks.{layer}.{}.{}", parts[3], suffix)))
            }
            "mlp" => {
                if parts.len() >= 6 {
                    let fc = parts[4];
                    let suffix = parts[5];
                    Ok(Some(format!("blocks.{layer}.mlp.{fc}.{suffix}")))
                } else {
                    Ok(None)
                }
            }
            "layer_scale1" => Ok(Some(format!("blocks.{layer}.ls1.gamma"))),
            "layer_scale2" => Ok(Some(format!("blocks.{layer}.ls2.gamma"))),
            "attention" => {
                if parts.len() < 6 {
                    return Ok(None);
                }
                match (parts[4], parts[5]) {
                    ("output", "dense") => {
                        let suffix = parts.get(6).copied().unwrap_or("");
                        Ok(Some(format!("blocks.{layer}.attn.proj.{suffix}")))
                    }
                    ("attention", proj) => {
                        let suffix = parts.get(6).copied().unwrap_or("");
                        let data = tensor_view_to_vec(view)?;
                        let entry = qkv_parts.entry(layer).or_default();
                        if entry.dim.is_none() {
                            entry.dim = Some(view.shape()[0]);
                        }
                        match proj {
                            "query" => set_qkv(entry, &data, suffix, true)?,
                            "key" => set_qkv(entry, &data, suffix, false)?,
                            "value" => set_qkv_value(entry, &data, suffix)?,
                            _ => {}
                        }
                        Ok(None)
                    }
                    _ => Ok(None),
                }
            }
            _ => Ok(None),
        }
    }

    fn set_qkv(
        entry: &mut QkvParts,
        data: &[f32],
        suffix: &str,
        is_query: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        match suffix {
            "weight" => {
                if is_query {
                    entry.q_weight = Some(data.to_vec());
                } else {
                    entry.k_weight = Some(data.to_vec());
                }
            }
            "bias" => {
                if is_query {
                    entry.q_bias = Some(data.to_vec());
                } else {
                    entry.k_bias = Some(data.to_vec());
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn set_qkv_value(
        entry: &mut QkvParts,
        data: &[f32],
        suffix: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        match suffix {
            "weight" => entry.v_weight = Some(data.to_vec()),
            "bias" => entry.v_bias = Some(data.to_vec()),
            _ => {}
        }
        Ok(())
    }

    fn tensor_view_to_vec(view: &TensorView<'_>) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        if view.dtype() != Dtype::F32 {
            return Err(Box::new(Dinov2ImportError(format!(
                "unsupported dtype {:?}",
                view.dtype()
            ))));
        }
        let data = bytemuck::cast_slice::<u8, f32>(view.data());
        Ok(data.to_vec())
    }

    struct OwnedTensor {
        name: String,
        shape: Vec<usize>,
        dtype: Dtype,
        data: Vec<u8>,
    }

    pub fn resolve_triposg_weights_root() -> PathBuf {
        if let Ok(root) = std::env::var("TRIPOSG_WEIGHTS_ROOT") {
            return PathBuf::from(root);
        }
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/models/MIDI-3D")
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
        preferred_precision_from_env("TRIPOSG_BPK_PRECISION", "BURN_SYNTH_BPK_PRECISION")
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

    pub fn import_triposg_dinov2_burnpack<B: Backend>(
        device: &B::Device,
        weights_path: impl AsRef<Path>,
        use_f16: bool,
    ) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let weights_path = weights_path.as_ref();
        let mut config = load_dinov2_config(weights_path)
            .unwrap_or_else(|| DinoVisionTransformerConfig::vitl(None, None));
        if let Some(target_size) = load_dinov2_preprocess_size(weights_path) {
            let patch = config.patch_size.max(1);
            let grid = target_size / patch;
            if grid > 0 {
                config.positional_encoding_interpolate.output_size = Some([grid, grid]);
            }
        }
        let burnpack_path = burnpack_path(weights_path, use_f16);
        let mut model: burn_dino::model::dino::DinoVisionTransformer<B> =
            burn_dino::model::dino::DinoVisionTransformer::new(device, config);

        let converted = convert_hf_dinov2(weights_path)?;
        let mut store = build_store(converted)?;
        model
            .load_from(&mut store)
            .map_err(|err| format!("failed to load dinov2 weights: {err}"))?;
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
        model: &burn_dino::model::dino::DinoVisionTransformer<B>,
        path: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut store = BurnpackStore::from_file(path).overwrite(true);
        model
            .save_into(&mut store)
            .map_err(|err| format!("failed to save dinov2 burnpack: {err}"))?;
        Ok(())
    }
}

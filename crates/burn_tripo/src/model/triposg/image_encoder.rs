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
    pub strict_preprocess: Option<bool>,
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
            strict_preprocess: None,
        }
    }
}

impl DinoImageProcessor {
    pub fn with_strict_preprocess(mut self, strict: bool) -> Self {
        self.strict_preprocess = Some(strict);
        self
    }

    pub fn set_strict_preprocess(&mut self, strict: bool) {
        self.strict_preprocess = Some(strict);
    }

    fn strict_preprocess_enabled(&self) -> bool {
        self.strict_preprocess.unwrap_or(false)
    }

    /// Returns whether strict preprocessing is enabled.
    ///
    /// Strict mode uses the CPU reference resize/crop path to match the
    /// upstream TripoSG preprocessing behavior.
    pub fn is_strict_preprocess(&self) -> bool {
        self.strict_preprocess_enabled()
    }

    pub fn preprocess<B: Backend>(&self, image: Tensor<B, 4>) -> Tensor<B, 4> {
        if !cfg!(target_arch = "wasm32") && self.strict_preprocess_enabled() {
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
                    align_corners: true,
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
            if cfg!(target_arch = "wasm32") && is_wgpu_backend::<B>() {
                // WebGPU storage bindings require 4-byte alignment. Avoid creating tiny
                // 3-element f16 tensors for mean/std on wasm by normalizing per-channel
                // with scalar ops.
                let [batch, channels, height, width] = image.shape().dims();
                if channels == 3 {
                    let c0 = image
                        .clone()
                        .slice([0..batch, 0..1, 0..height, 0..width])
                        .sub_scalar(self.mean[0])
                        .div_scalar(self.std[0]);
                    let c1 = image
                        .clone()
                        .slice([0..batch, 1..2, 0..height, 0..width])
                        .sub_scalar(self.mean[1])
                        .div_scalar(self.std[1]);
                    let c2 = image
                        .slice([0..batch, 2..3, 0..height, 0..width])
                        .sub_scalar(self.mean[2])
                        .div_scalar(self.std[2]);
                    image = Tensor::cat(vec![c0, c1, c2], 1);
                } else {
                    let device = image.device();
                    let mean =
                        Tensor::<B, 1>::from_floats(self.mean, &device).reshape([1, 3, 1, 1]);
                    let std = Tensor::<B, 1>::from_floats(self.std, &device).reshape([1, 3, 1, 1]);
                    image = image.sub(mean).div(std);
                }
            } else {
                let device = image.device();
                let mean = Tensor::<B, 1>::from_floats(self.mean, &device).reshape([1, 3, 1, 1]);
                let std = Tensor::<B, 1>::from_floats(self.std, &device).reshape([1, 3, 1, 1]);
                image = image.sub(mean).div(std);
            }
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

fn is_wgpu_backend<B: Backend>() -> bool {
    std::any::type_name::<B>()
        .to_ascii_lowercase()
        .contains("wgpu")
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
        ApplyResult, BurnpackStore, KeyRemapper, ModuleSnapshot, PyTorchToBurnAdapter,
        SafetensorsStore,
    };
    use burn_synth_import::parts::load_model_from_burnpack_parts;
    use safetensors::{
        Dtype, serialize,
        tensor::{SafeTensors, TensorView},
    };

    use super::super::load_policy::{BurnpackLoadPolicy, burnpack_path, candidate_burnpack_paths};
    use super::{DinoImageProcessor, TripoSGImageEncoder};
    use burn_dino::model::dino::DinoVisionTransformerConfig;

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
        load_triposg_dinov2_with_policy(device, weights_path, default_burnpack_policy())
    }

    pub fn load_triposg_dinov2_with_policy<B: Backend>(
        device: &B::Device,
        weights_path: impl AsRef<Path>,
        policy: BurnpackLoadPolicy,
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
        let burnpack_candidates = candidate_burnpack_paths(weights_path, policy);
        if let Some(model) = load_model_from_burnpack_parts(
            &burnpack_candidates,
            "DINOv2",
            should_validate_burnpack(),
            || {
                let dino =
                    burn_dino::model::dino::DinoVisionTransformer::new(device, config.clone());
                TripoSGImageEncoder::new(dino)
            },
            |model, part_bytes| {
                apply_triposg_dinov2_burnpack_part_bytes(model, part_bytes)
                    .map_err(|err| format!("failed to apply DINOv2 burnpack part bytes: {err}"))
            },
        )? {
            return Ok(model);
        }
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
        let mut store =
            BurnpackStore::from_file(&burnpack_path).validate(should_validate_burnpack());
        let apply = model
            .load_from(&mut store)
            .map_err(|err| format!("failed to load dinov2 burnpack: {err}"))?;
        validate_apply_result("dinov2 burnpack", &apply)?;

        Ok(TripoSGImageEncoder::new(model))
    }

    pub fn load_triposg_dinov2_from_safetensors<B: Backend>(
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

        let mut model: burn_dino::model::dino::DinoVisionTransformer<B> =
            burn_dino::model::dino::DinoVisionTransformer::new(device, config);
        let converted = convert_hf_dinov2(weights_path)?;
        let mut store = build_store(converted)?;
        let apply = model
            .load_from(&mut store)
            .map_err(|err| format!("failed to load dinov2 safetensors: {err}"))?;
        validate_apply_result("dinov2 safetensors", &apply)?;
        Ok(TripoSGImageEncoder::new(model))
    }

    pub fn init_triposg_dinov2_model<B: Backend>(
        device: &B::Device,
        config: DinoVisionTransformerConfig,
    ) -> TripoSGImageEncoder<B> {
        let model: burn_dino::model::dino::DinoVisionTransformer<B> =
            burn_dino::model::dino::DinoVisionTransformer::new(device, config);
        TripoSGImageEncoder::new(model)
    }

    pub fn load_triposg_dinov2_from_burnpack_bytes<B: Backend>(
        device: &B::Device,
        config: DinoVisionTransformerConfig,
        burnpack_bytes: Vec<u8>,
    ) -> Result<TripoSGImageEncoder<B>, Box<dyn std::error::Error>> {
        let mut model: burn_dino::model::dino::DinoVisionTransformer<B> =
            burn_dino::model::dino::DinoVisionTransformer::new(device, config);
        let mut store = BurnpackStore::from_bytes(Some(Bytes::from_bytes_vec(burnpack_bytes)))
            .validate(should_validate_burnpack());
        let apply = model
            .load_from(&mut store)
            .map_err(|err| format!("failed to load dinov2 burnpack bytes: {err}"))?;
        validate_apply_result("dinov2 burnpack bytes", &apply)?;
        Ok(TripoSGImageEncoder::new(model))
    }

    pub fn apply_triposg_dinov2_burnpack_part_bytes<B: Backend>(
        model: &mut TripoSGImageEncoder<B>,
        burnpack_bytes: Vec<u8>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut store = BurnpackStore::from_bytes(Some(Bytes::from_bytes_vec(burnpack_bytes)))
            .allow_partial(true)
            .validate(should_validate_burnpack());
        model
            .dino
            .load_from(&mut store)
            .map_err(|err| format!("failed to load dinov2 burnpack part bytes: {err}"))?;
        Ok(())
    }

    pub fn load_triposg_dinov2_from_burnpack_file<B: Backend>(
        device: &B::Device,
        config: DinoVisionTransformerConfig,
        burnpack_path: impl AsRef<Path>,
    ) -> Result<TripoSGImageEncoder<B>, Box<dyn std::error::Error>> {
        let mut model: burn_dino::model::dino::DinoVisionTransformer<B> =
            burn_dino::model::dino::DinoVisionTransformer::new(device, config);
        let mut store =
            BurnpackStore::from_file(burnpack_path.as_ref()).validate(should_validate_burnpack());
        let apply = model
            .load_from(&mut store)
            .map_err(|err| format!("failed to load dinov2 burnpack file: {err}"))?;
        validate_apply_result("dinov2 burnpack file", &apply)?;
        Ok(TripoSGImageEncoder::new(model))
    }

    const CANONICAL_DINO_SHORT_EDGE: usize = 256;
    const CANONICAL_DINO_CROP: usize = 224;
    const CANONICAL_DINO_INPUT_CHANNELS: usize = 3;
    const LEGACY_DINO_SIZE_CAP: usize = 384;

    fn should_validate_burnpack() -> bool {
        cfg!(all(not(target_arch = "wasm32"), debug_assertions))
    }

    pub fn load_dinov2_processor(
        weights_root: impl AsRef<Path>,
    ) -> Result<DinoImageProcessor, Box<dyn std::error::Error>> {
        if !allow_legacy_dinov2_preprocessor() {
            return Ok(canonical_dinov2_processor());
        }

        let root = weights_root.as_ref();
        let fallback_size = load_dinov2_image_size(root);
        let mut legacy_processor = None;

        for (kind, path) in dinov2_preprocessor_config_paths(root) {
            let Ok(bytes) = fs::read(&path) else {
                continue;
            };
            let Ok(config) = load_dinov2_processor_config_from_json_bytes(&bytes) else {
                continue;
            };
            if !is_bit_image_processor(&config) {
                continue;
            }

            let processor = processor_from_config(config, fallback_size);
            if matches!(kind, Dinov2PreprocessorPathKind::Dedicated) {
                return Ok(processor);
            }
            if legacy_processor.is_none() {
                legacy_processor = Some(processor);
            }
        }

        if let Some(processor) = legacy_processor {
            if should_force_canonical_processor(root, &processor) {
                return Ok(canonical_dinov2_processor());
            }
            return Ok(processor);
        }

        if has_dinov2_weights_root(root) && !allow_legacy_dinov2_preprocessor() {
            return Ok(canonical_dinov2_processor());
        }

        let mut processor = DinoImageProcessor::default();
        if let Some(target_size) = fallback_size {
            processor.do_resize = true;
            processor.size_shortest_edge = Some(target_size);
            processor.do_center_crop = true;
            processor.crop_size = Some([target_size, target_size]);
        }
        Ok(processor)
    }

    pub fn load_dinov2_processor_from_json_bytes(
        bytes: &[u8],
        fallback_size: Option<usize>,
    ) -> Result<DinoImageProcessor, Box<dyn std::error::Error>> {
        let config = load_dinov2_processor_config_from_json_bytes(bytes)?;
        Ok(processor_from_config(config, fallback_size))
    }

    fn load_dinov2_processor_config_from_json_bytes(
        bytes: &[u8],
    ) -> Result<Dinov2ProcessorConfig, Box<dyn std::error::Error>> {
        Ok(serde_json::from_slice(bytes)?)
    }

    fn processor_from_config(
        config: Dinov2ProcessorConfig,
        fallback_size: Option<usize>,
    ) -> DinoImageProcessor {
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
            strict_preprocess: None,
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

        processor
    }

    #[derive(serde::Deserialize)]
    struct Dinov2ProcessorConfig {
        image_processor_type: Option<String>,
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

    #[derive(Clone, Copy)]
    enum Dinov2PreprocessorPathKind {
        Dedicated,
        LegacyBit,
        LegacyClip,
    }

    fn load_dinov2_image_size(weights_root: &Path) -> Option<usize> {
        for path in [
            weights_root.join("image_encoder_dinov2/config.json"),
            weights_root.join("image_encoder_2/config.json"),
        ] {
            if let Ok(bytes) = fs::read(path)
                && let Some(size) = load_dinov2_image_size_from_json_bytes(&bytes)
            {
                return Some(size);
            }
        }
        None
    }

    fn load_dinov2_preprocess_size(weights_path: &Path) -> Option<usize> {
        if !allow_legacy_dinov2_preprocessor() {
            return Some(CANONICAL_DINO_CROP);
        }

        let weights_root = weights_path.parent()?.parent()?;
        let mut legacy_size = None;
        for (kind, path) in dinov2_preprocessor_config_paths(weights_root) {
            let Ok(bytes) = fs::read(path) else {
                continue;
            };
            let Some(size) = load_dinov2_preprocess_size_from_json_bytes(&bytes) else {
                continue;
            };
            match kind {
                Dinov2PreprocessorPathKind::Dedicated => return Some(size),
                Dinov2PreprocessorPathKind::LegacyBit | Dinov2PreprocessorPathKind::LegacyClip => {
                    if legacy_size.is_none() {
                        legacy_size = Some(size);
                    }
                }
            }
        }
        if let Some(size) = legacy_size {
            if has_dinov2_weights_root(weights_root)
                && !allow_legacy_dinov2_preprocessor()
                && size > LEGACY_DINO_SIZE_CAP
            {
                return Some(CANONICAL_DINO_CROP);
            }
            return Some(size);
        }
        if has_dinov2_weights_root(weights_root) && !allow_legacy_dinov2_preprocessor() {
            return Some(CANONICAL_DINO_CROP);
        }
        None
    }

    fn dinov2_preprocessor_config_paths(
        weights_root: &Path,
    ) -> [(Dinov2PreprocessorPathKind, PathBuf); 3] {
        [
            (
                Dinov2PreprocessorPathKind::Dedicated,
                weights_root.join("feature_extractor_dinov2/preprocessor_config.json"),
            ),
            (
                Dinov2PreprocessorPathKind::LegacyBit,
                weights_root.join("feature_extractor_2/preprocessor_config.json"),
            ),
            (
                Dinov2PreprocessorPathKind::LegacyClip,
                weights_root.join("feature_extractor_1/preprocessor_config.json"),
            ),
        ]
    }

    pub fn load_dinov2_preprocess_size_from_json_bytes(bytes: &[u8]) -> Option<usize> {
        let config: Dinov2ProcessorConfig = serde_json::from_slice(bytes).ok()?;
        if !is_bit_image_processor(&config) {
            return None;
        }
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

    fn is_bit_image_processor(config: &Dinov2ProcessorConfig) -> bool {
        config
            .image_processor_type
            .as_deref()
            .map(|name| name.eq_ignore_ascii_case("BitImageProcessor"))
            .unwrap_or(true)
    }

    fn allow_legacy_dinov2_preprocessor() -> bool {
        false
    }

    fn has_dinov2_weights_root(weights_root: &Path) -> bool {
        let dino_dir = weights_root.join("image_encoder_dinov2");
        dino_dir.join("model.safetensors").exists()
            || dino_dir.join("model.bpk").exists()
            || dino_dir.join("model_f16.bpk").exists()
    }

    fn should_force_canonical_processor(
        weights_root: &Path,
        processor: &DinoImageProcessor,
    ) -> bool {
        if !has_dinov2_weights_root(weights_root) || allow_legacy_dinov2_preprocessor() {
            return false;
        }
        let size = processor
            .crop_size
            .map(|crop| crop[0].min(crop[1]))
            .or(processor.size_shortest_edge)
            .unwrap_or(0);
        size > LEGACY_DINO_SIZE_CAP
    }

    fn canonical_dinov2_processor() -> DinoImageProcessor {
        DinoImageProcessor {
            do_resize: true,
            size_shortest_edge: Some(CANONICAL_DINO_SHORT_EDGE),
            do_center_crop: true,
            crop_size: Some([CANONICAL_DINO_CROP, CANONICAL_DINO_CROP]),
            ..DinoImageProcessor::default()
        }
    }

    #[cfg(test)]
    mod tests {
        use std::fs;
        use std::path::PathBuf;
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::time::{SystemTime, UNIX_EPOCH};

        use super::{
            load_dinov2_config_from_json_bytes, load_dinov2_preprocess_size,
            load_dinov2_preprocess_size_from_json_bytes, load_dinov2_processor,
        };

        static TEST_NONCE: AtomicU64 = AtomicU64::new(0);

        fn make_temp_root(label: &str) -> PathBuf {
            let nonce = TEST_NONCE.fetch_add(1, Ordering::Relaxed);
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            std::env::temp_dir().join(format!(
                "burn_tripo_dino_processor_test_{}_{}_{}_{}",
                label,
                std::process::id(),
                nanos,
                nonce
            ))
        }

        #[test]
        fn loads_preprocessor_from_legacy_feature_extractor_directory() {
            let root = make_temp_root("legacy");
            let legacy = root.join("feature_extractor_2");
            fs::create_dir_all(&legacy).expect("create legacy preprocessor dir");
            fs::write(
                legacy.join("preprocessor_config.json"),
                r#"{
                    "crop_size": { "height": 512, "width": 512 },
                    "do_center_crop": true,
                    "do_normalize": true,
                    "do_rescale": true,
                    "do_resize": true,
                    "image_mean": [0.485, 0.456, 0.406],
                    "image_std": [0.229, 0.224, 0.225],
                    "resample": 3,
                    "rescale_factor": 0.00392156862745098,
                    "size": { "shortest_edge": 512 }
                }"#,
            )
            .expect("write legacy preprocessor config");

            let processor = load_dinov2_processor(&root).expect("load processor");
            assert!(processor.do_resize);
            assert_eq!(processor.size_shortest_edge, Some(256));
            assert!(processor.do_center_crop);
            assert_eq!(processor.crop_size, Some([224, 224]));

            let _ = fs::remove_dir_all(root);
        }

        #[test]
        fn canonicalizes_oversized_legacy_bit_preprocessor_for_dinov2_assets() {
            let root = make_temp_root("canonicalize");
            let legacy = root.join("feature_extractor_2");
            let dino_dir = root.join("image_encoder_dinov2");
            fs::create_dir_all(&legacy).expect("create legacy preprocessor dir");
            fs::create_dir_all(&dino_dir).expect("create dino dir");
            fs::write(
                legacy.join("preprocessor_config.json"),
                r#"{
                    "image_processor_type": "BitImageProcessor",
                    "crop_size": { "height": 512, "width": 512 },
                    "do_center_crop": true,
                    "do_resize": true,
                    "size": { "shortest_edge": 512 }
                }"#,
            )
            .expect("write legacy preprocessor config");

            let processor = load_dinov2_processor(&root).expect("load processor");
            assert_eq!(processor.size_shortest_edge, Some(256));
            assert_eq!(processor.crop_size, Some([224, 224]));

            let weights_path = dino_dir.join("model.safetensors");
            let preprocess_size = load_dinov2_preprocess_size(&weights_path);
            assert_eq!(preprocess_size, Some(224));

            let _ = fs::remove_dir_all(root);
        }

        #[test]
        fn ignores_clip_preprocessor_for_dinov2_size_selection() {
            let clip_json = br#"{
                "image_processor_type": "CLIPImageProcessor",
                "crop_size": { "height": 224, "width": 224 },
                "do_center_crop": true,
                "do_resize": true,
                "size": { "shortest_edge": 224 }
            }"#;
            assert_eq!(load_dinov2_preprocess_size_from_json_bytes(clip_json), None);
        }

        #[test]
        fn falls_back_to_default_processor_when_no_preprocessor_json_exists() {
            let root = make_temp_root("default");
            let dino_config_dir = root.join("image_encoder_dinov2");
            fs::create_dir_all(&dino_config_dir).expect("create dino config dir");
            fs::write(
                dino_config_dir.join("config.json"),
                r#"{
                    "image_size": 518,
                    "patch_size": 14,
                    "num_channels": 3
                }"#,
            )
            .expect("write dino config");

            let processor = load_dinov2_processor(&root).expect("load default processor");
            assert!(processor.do_resize);
            assert_eq!(processor.size_shortest_edge, Some(256));
            assert!(processor.do_center_crop);
            assert_eq!(processor.crop_size, Some([224, 224]));
            assert_eq!(processor.mean, [0.485, 0.456, 0.406]);
            assert_eq!(processor.std, [0.229, 0.224, 0.225]);

            let _ = fs::remove_dir_all(root);
        }

        #[test]
        fn canonicalizes_invalid_dino_num_channels_to_rgb() {
            let config = load_dinov2_config_from_json_bytes(
                br#"{
                    "image_size": 518,
                    "patch_size": 14,
                    "num_channels": 7
                }"#,
            )
            .expect("load dino config");
            assert_eq!(config.input_channels, 3);
        }
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
            // TripoSG DINOv2 consumes canonical RGB images; some exported configs
            // incorrectly report non-RGB channel counts (for example 7).
            dino.input_channels = if channels == CANONICAL_DINO_INPUT_CHANNELS {
                channels
            } else {
                CANONICAL_DINO_INPUT_CHANNELS
            };
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
            .allow_partial(false)
            .remap(remapper)
            .validate(true);
        Ok(store)
    }

    fn validate_apply_result(
        label: &str,
        result: &ApplyResult,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if result.missing.is_empty() && result.skipped.is_empty() && result.unused.is_empty() {
            return Ok(());
        }

        let mut parts = Vec::new();
        if !result.missing.is_empty() {
            let preview = result
                .missing
                .iter()
                .take(8)
                .map(|item| format!("{item:?}"))
                .collect::<Vec<_>>()
                .join(", ");
            parts.push(format!(
                "missing={} [{}{}]",
                result.missing.len(),
                preview,
                if result.missing.len() > 8 {
                    ", ..."
                } else {
                    ""
                }
            ));
        }
        if !result.skipped.is_empty() {
            let preview = result
                .skipped
                .iter()
                .take(8)
                .map(|item| format!("{item:?}"))
                .collect::<Vec<_>>()
                .join(", ");
            parts.push(format!(
                "skipped={} [{}{}]",
                result.skipped.len(),
                preview,
                if result.skipped.len() > 8 {
                    ", ..."
                } else {
                    ""
                }
            ));
        }
        if !result.unused.is_empty() {
            let preview = result
                .unused
                .iter()
                .take(8)
                .map(|item| format!("{item:?}"))
                .collect::<Vec<_>>()
                .join(", ");
            parts.push(format!(
                "unused={} [{}{}]",
                result.unused.len(),
                preview,
                if result.unused.len() > 8 { ", ..." } else { "" }
            ));
        }

        Err(format!("{label} import mismatch: {}", parts.join("; ")).into())
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
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/models/MIDI-3D")
    }

    fn default_burnpack_policy() -> BurnpackLoadPolicy {
        BurnpackLoadPolicy::default()
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
        let burnpack_path = burnpack_path(
            weights_path,
            use_f16,
            BurnpackLoadPolicy::default().f16_suffix,
        );
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

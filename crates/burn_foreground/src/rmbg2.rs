use std::path::Path;

use crate::pipeline::{PrepareImageConfig, PrepareImageError, PreparedImageData};
#[cfg(not(target_arch = "wasm32"))]
use crate::pipeline::{
    bbox_from_mask_all, build_prepared_image_from_alpha, is_valid_alpha, load_image_rgb,
    otsu_threshold, remove_small_objects,
};
#[cfg(not(target_arch = "wasm32"))]
use crate::preprocess::RmbgImageProcessor;
#[cfg(not(target_arch = "wasm32"))]
use crate::resize::resize_chw_align_corners_false;
#[cfg(not(target_arch = "wasm32"))]
use burn::tensor::ops::InterpolateMode;

#[cfg(all(not(target_arch = "wasm32"), feature = "import"))]
use ort::session::builder::GraphOptimizationLevel;
#[cfg(not(target_arch = "wasm32"))]
use ort::{session::Session, value::Tensor};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Mutex;

#[cfg(not(target_arch = "wasm32"))]
pub struct Rmbg2Pipeline {
    session: Mutex<Session>,
    pub processor: RmbgImageProcessor,
}

#[cfg(not(target_arch = "wasm32"))]
impl Rmbg2Pipeline {
    pub fn new(session: Session, processor: RmbgImageProcessor) -> Self {
        Self {
            session: Mutex::new(session),
            processor,
        }
    }

    #[cfg(feature = "import")]
    pub fn from_pretrained(
        weights_root: impl AsRef<Path>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        use self::import::{load_rmbg2_processor_config, resolve_rmbg2_model_path};

        let root = weights_root.as_ref();
        let model_path = resolve_rmbg2_model_path(root)?;
        let processor = load_rmbg2_processor_config(root)?;

        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Disable)?
            .commit_from_file(&model_path)?;

        Ok(Self::new(session, processor))
    }

    pub fn prepare_image_data(
        &self,
        path: &Path,
        config: &PrepareImageConfig,
    ) -> Result<PreparedImageData, PrepareImageError> {
        let loaded = load_image_rgb(path, config.max_dimension)?;
        let rgb = loaded.rgb;
        let width = loaded.width;
        let height = loaded.height;

        let alpha = if loaded.has_alpha {
            loaded.alpha.and_then(|alpha| {
                if is_valid_alpha(&alpha, width, height, 0.01) {
                    Some(alpha)
                } else {
                    None
                }
            })
        } else {
            None
        };

        let (alpha_mask, alpha_probs, bbox) = if let Some(alpha) = alpha {
            let alpha_mask = alpha
                .iter()
                .map(|value| *value as f32 / 255.0)
                .collect::<Vec<f32>>();
            let bbox = bbox_from_mask_all(&alpha, width, height)
                .ok_or_else(|| PrepareImageError("input image too small".to_string()))?;
            (alpha_mask, None, bbox)
        } else {
            let probs = self.infer_alpha_probs(&rgb, width, height)?;
            let mut alpha_u8 = probs
                .iter()
                .map(|value| (value * 255.0).clamp(0.0, 255.0) as u8)
                .collect::<Vec<u8>>();

            let mut thresh = otsu_threshold(&alpha_u8) as i32;
            if let Some(bias) = std::env::var("RMBG_OTSU_BIAS")
                .ok()
                .and_then(|v| v.parse::<i32>().ok())
            {
                thresh = (thresh + bias).clamp(0, 255);
            }
            let thresh = thresh as u8;
            for value in &mut alpha_u8 {
                *value = if *value > thresh { 255 } else { 0 };
            }

            let cleaned = remove_small_objects(&alpha_u8, width, height, config.min_component_size);
            let bbox = bbox_from_mask_all(&cleaned, width, height)
                .ok_or_else(|| PrepareImageError("input image too small".to_string()))?;
            let alpha_mask = cleaned
                .iter()
                .map(|value| if *value > 0 { 1.0 } else { 0.0 })
                .collect::<Vec<f32>>();
            (alpha_mask, Some(probs), bbox)
        };

        build_prepared_image_from_alpha(&rgb, width, height, alpha_mask, alpha_probs, bbox, config)
    }

    fn infer_alpha_probs(
        &self,
        rgb: &[u8],
        width: usize,
        height: usize,
    ) -> Result<Vec<f32>, PrepareImageError> {
        let pixels = width * height;
        let mut rgb_chw = Vec::with_capacity(pixels * 3);
        for c in 0..3 {
            for idx in 0..pixels {
                rgb_chw.push(rgb[idx * 3 + c] as f32);
            }
        }

        let processor = &self.processor;
        let target_size = if processor.do_resize {
            processor.size.unwrap_or([height, width])
        } else {
            [height, width]
        };
        let [target_height, target_width] = target_size;

        let mut input = if target_height != height || target_width != width {
            resize_chw_align_corners_false(
                &rgb_chw,
                3,
                height,
                width,
                target_height,
                target_width,
                processor.resize_mode.clone(),
            )
        } else {
            rgb_chw
        };

        if processor.do_rescale {
            for value in &mut input {
                *value *= processor.rescale_factor;
            }
        }

        if processor.do_normalize {
            let pixels = target_height * target_width;
            for c in 0..3 {
                let mean = processor.mean[c];
                let std = processor.std[c];
                let offset = c * pixels;
                for idx in 0..pixels {
                    let value = input[offset + idx];
                    input[offset + idx] = (value - mean) / std;
                }
            }
        }

        let tensor = Tensor::<f32>::from_array(([1, 3, target_height, target_width], input))
            .map_err(|err| {
                PrepareImageError(format!("failed to create RMBG2 input tensor: {err}"))
            })?;
        let mut session = self
            .session
            .lock()
            .map_err(|_| PrepareImageError("failed to lock RMBG2 session".to_string()))?;
        let outputs = session
            .run(ort::inputs![tensor])
            .map_err(|err| PrepareImageError(format!("RMBG2 inference failed: {err}")))?;
        let output = outputs.get("alphas").unwrap_or(&outputs[0]);
        let (_, alpha_probs) = output
            .try_extract_tensor::<f32>()
            .map_err(|err| PrepareImageError(format!("failed to extract RMBG2 output: {err}")))?;
        let mut data = alpha_probs.to_vec();

        if target_height != height || target_width != width {
            data = resize_chw_align_corners_false(
                &data,
                1,
                target_height,
                target_width,
                height,
                width,
                InterpolateMode::Bilinear,
            );
        }

        for value in &mut data {
            *value = value.clamp(0.0, 1.0);
        }

        Ok(data)
    }
}

#[cfg(target_arch = "wasm32")]
#[derive(Default)]
pub struct Rmbg2Pipeline;

#[cfg(target_arch = "wasm32")]
impl Rmbg2Pipeline {
    #[cfg(feature = "import")]
    pub fn from_pretrained(
        _weights_root: impl AsRef<Path>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Err("RMBG-2.0 ONNX pipeline is unavailable on wasm32 targets".into())
    }

    pub fn prepare_image_data(
        &self,
        _path: &Path,
        _config: &PrepareImageConfig,
    ) -> Result<PreparedImageData, PrepareImageError> {
        Err(PrepareImageError(
            "RMBG-2.0 ONNX pipeline is unavailable on wasm32 targets".to_string(),
        ))
    }
}

#[cfg(feature = "import")]
pub mod import {
    use std::{
        fs,
        path::{Path, PathBuf},
    };

    use burn::tensor::ops::InterpolateMode;

    use crate::preprocess::{RmbgImageProcessor, RmbgProcessorConfig};

    pub fn resolve_rmbg2_weights_root() -> PathBuf {
        if let Ok(root) = std::env::var("RMBG2_WEIGHTS_ROOT") {
            let path = PathBuf::from(root);
            if path.exists() {
                return path;
            }
        }
        if let Ok(root) = std::env::var("RMBG_WEIGHTS_ROOT") {
            let path = PathBuf::from(root);
            if path.exists() {
                return path;
            }
        }

        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let candidates = [
            manifest.join("../burn_tripo/assets/models/RMBG-2.0"),
            manifest.join("assets/models/RMBG-2.0"),
            manifest.join("../burn_tripo/assets/models/RMBG-1.4"),
            manifest.join("assets/models/RMBG-1.4"),
        ];
        for path in candidates {
            if path.exists() {
                return path;
            }
        }
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/models/RMBG-2.0")
    }

    pub fn resolve_rmbg2_model_path(root: impl AsRef<Path>) -> Result<PathBuf, String> {
        let root = root.as_ref();
        if root.is_file() && root.extension().and_then(|e| e.to_str()) == Some("onnx") {
            return Ok(root.to_path_buf());
        }

        if let Ok(explicit) = std::env::var("RMBG2_ONNX_FILE") {
            let explicit = PathBuf::from(explicit);
            let path = if explicit.is_relative() {
                root.join(explicit)
            } else {
                explicit
            };
            if path.exists() {
                return Ok(path);
            }
        }

        let candidates = [
            "onnx/model.onnx",
            "onnx/model_fp16.onnx",
            "onnx/model_int8.onnx",
            "onnx/model_q4f16.onnx",
            "onnx/model_q4.onnx",
            "onnx/model_quantized.onnx",
            "onnx/model_uint8.onnx",
            "model.onnx",
        ];
        for rel in candidates {
            let path = root.join(rel);
            if path.exists() {
                return Ok(path);
            }
        }

        Err(format!(
            "RMBG2 ONNX model not found under {} (checked: {})",
            root.display(),
            candidates.join(", ")
        ))
    }

    pub fn default_rmbg2_processor() -> RmbgImageProcessor {
        RmbgImageProcessor {
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
            rescale_factor: 1.0 / 255.0,
            do_rescale: true,
            do_normalize: true,
            do_resize: true,
            size: Some([1024, 1024]),
            resize_mode: InterpolateMode::Bilinear,
        }
    }

    pub fn load_rmbg2_processor_config(
        root: impl AsRef<Path>,
    ) -> Result<RmbgImageProcessor, Box<dyn std::error::Error>> {
        let path = root.as_ref().join("preprocessor_config.json");
        if !path.exists() {
            return Ok(default_rmbg2_processor());
        }
        let bytes = fs::read(path)?;
        load_rmbg2_processor_from_json_bytes(&bytes)
    }

    pub fn load_rmbg2_processor_from_json_bytes(
        bytes: &[u8],
    ) -> Result<RmbgImageProcessor, Box<dyn std::error::Error>> {
        let config: RmbgProcessorConfig = serde_json::from_slice(bytes)?;
        let mut processor = RmbgImageProcessor::from_config(config.clone());
        if config.image_mean.is_none() {
            processor.mean = [0.485, 0.456, 0.406];
        }
        if config.image_std.is_none() {
            processor.std = [0.229, 0.224, 0.225];
        }
        Ok(processor)
    }
}

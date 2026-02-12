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
        use self::import::{
            load_rmbg2_onnx_blob_from_preferred_burnpack, load_rmbg2_processor_config,
            resolve_rmbg2_model_path,
        };

        let root = weights_root.as_ref();
        let processor = load_rmbg2_processor_config(root)?;

        let session = if let Some(onnx_bytes) = load_rmbg2_onnx_blob_from_preferred_burnpack(root)?
        {
            Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Disable)?
                .commit_from_memory(&onnx_bytes)?
        } else {
            let model_path = resolve_rmbg2_model_path(root)?;
            Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Disable)?
                .commit_from_file(&model_path)?
        };

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

            let thresh = otsu_threshold(&alpha_u8) as u8;
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

    #[cfg(not(target_arch = "wasm32"))]
    use burn::module::{Param, ParamId};
    #[cfg(not(target_arch = "wasm32"))]
    use burn::prelude::*;
    use burn::tensor::ops::InterpolateMode;
    #[cfg(not(target_arch = "wasm32"))]
    use burn_store::{BurnpackStore, ModuleSnapshot};

    use crate::preprocess::{RmbgImageProcessor, RmbgProcessorConfig};

    const F16_SUFFIX: &str = "_f16";

    const MODEL_CANDIDATES_F16: &[&str] = &[
        "onnx/model_fp16.onnx",
        "onnx/model.onnx",
        "onnx/model_q4f16.onnx",
        "onnx/model_q4.onnx",
        "onnx/model_quantized.onnx",
        "onnx/model_uint8.onnx",
        "onnx/model_int8.onnx",
        "model_fp16.onnx",
        "model.onnx",
    ];

    const MODEL_CANDIDATES_F32: &[&str] = &[
        "onnx/model.onnx",
        "onnx/model_fp16.onnx",
        "onnx/model_q4f16.onnx",
        "onnx/model_q4.onnx",
        "onnx/model_quantized.onnx",
        "onnx/model_uint8.onnx",
        "onnx/model_int8.onnx",
        "model.onnx",
        "model_fp16.onnx",
    ];

    #[cfg(not(target_arch = "wasm32"))]
    #[derive(Module, Debug)]
    struct Rmbg2OnnxBlob<B: Backend> {
        bytes: Param<Tensor<B, 1, Int>>,
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    struct Rmbg2OnnxBlobMetadata {
        bytes_len: usize,
        source_path: String,
    }

    pub fn resolve_rmbg2_weights_root() -> PathBuf {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let candidates = [manifest.join("assets/models/RMBG-2.0")];
        for path in candidates {
            if path.exists() {
                return path;
            }
        }
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/models/RMBG-2.0")
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn import_rmbg2_burnpack(
        root: impl AsRef<Path>,
        use_f16: bool,
    ) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let root = root.as_ref();
        if !root.exists() {
            return Err(format!("RMBG-2.0 root does not exist: {}", root.display()).into());
        }

        let onnx_path = resolve_rmbg2_model_path_with_precision(root, use_f16)
            .map_err(|err| format!("failed to resolve RMBG-2.0 ONNX source: {err}"))?;
        let bytes = fs::read(&onnx_path)
            .map_err(|err| format!("failed to read ONNX model {}: {err}", onnx_path.display()))?;

        let output_path = burnpack_path(root, use_f16);
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let metadata = Rmbg2OnnxBlobMetadata {
            bytes_len: bytes.len(),
            source_path: onnx_path.display().to_string(),
        };
        save_rmbg2_onnx_blob_to_burnpack(&output_path, &bytes, &metadata)?;

        Ok(output_path)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_rmbg2_onnx_blob_from_preferred_burnpack(
        root: impl AsRef<Path>,
    ) -> Result<Option<Vec<u8>>, Box<dyn std::error::Error>> {
        let candidates = candidate_burnpack_paths(root.as_ref());
        let existing = candidates
            .iter()
            .filter(|path| path.exists())
            .cloned()
            .collect::<Vec<_>>();
        if existing.is_empty() {
            return Ok(None);
        }

        for path in existing {
            if let Ok(bytes) = load_rmbg2_onnx_blob_from_burnpack(&path) {
                return Ok(Some(bytes));
            }
        }

        Ok(None)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_rmbg2_onnx_blob_from_burnpack(
        path: impl AsRef<Path>,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        type BlobBackend = burn::backend::NdArray<f32>;

        let path = path.as_ref();
        let metadata_path = burnpack_metadata_path(path);
        let metadata_bytes = fs::read(&metadata_path).map_err(|err| {
            format!(
                "failed to read RMBG-2.0 burnpack metadata {}: {err}",
                metadata_path.display()
            )
        })?;
        let metadata: Rmbg2OnnxBlobMetadata =
            serde_json::from_slice(&metadata_bytes).map_err(|err| {
                format!(
                    "failed to parse RMBG-2.0 burnpack metadata {}: {err}",
                    metadata_path.display()
                )
            })?;

        let device = <BlobBackend as Backend>::Device::default();
        let zeros = Tensor::<BlobBackend, 1, Int>::zeros([metadata.bytes_len], &device);
        let mut blob = Rmbg2OnnxBlob {
            bytes: Param::initialized(ParamId::new(), zeros),
        };

        let mut store = BurnpackStore::from_file(path).validate(true);
        blob.load_from(&mut store)
            .map_err(|err| format!("failed to load RMBG-2.0 burnpack {}: {err}", path.display()))?;
        let bytes = blob
            .bytes
            .val()
            .into_data()
            .convert::<u8>()
            .to_vec::<u8>()
            .map_err(|err| format!("failed to materialize RMBG-2.0 burnpack bytes: {err:?}"))?;

        if bytes.len() != metadata.bytes_len {
            return Err(format!(
                "RMBG-2.0 burnpack byte length mismatch for {}: expected {}, got {}",
                path.display(),
                metadata.bytes_len,
                bytes.len()
            )
            .into());
        }

        Ok(bytes)
    }

    pub fn resolve_rmbg2_model_path(root: impl AsRef<Path>) -> Result<PathBuf, String> {
        let prefer_f16 = prefer_f16_burnpack();
        resolve_rmbg2_model_path_with_precision(root.as_ref(), prefer_f16)
    }

    fn resolve_rmbg2_model_path_with_precision(
        root: &Path,
        use_f16: bool,
    ) -> Result<PathBuf, String> {
        if root.is_file() && root.extension().and_then(|e| e.to_str()) == Some("onnx") {
            return Ok(root.to_path_buf());
        }

        let candidates = if use_f16 {
            MODEL_CANDIDATES_F16
        } else {
            MODEL_CANDIDATES_F32
        };
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

    fn candidate_burnpack_paths(root: &Path) -> Vec<PathBuf> {
        let default = burnpack_path(root, false);
        let f16 = burnpack_path(root, true);
        if f16 == default {
            vec![default]
        } else if prefer_f16_burnpack() {
            vec![f16, default]
        } else {
            vec![default, f16]
        }
    }

    fn prefer_f16_burnpack() -> bool {
        true
    }

    fn burnpack_path(root: &Path, use_f16: bool) -> PathBuf {
        let path = if root
            .extension()
            .map(|ext| ext.eq_ignore_ascii_case("bpk"))
            .unwrap_or(false)
        {
            root.to_path_buf()
        } else if root.is_dir() {
            root.join("model.bpk")
        } else {
            root.with_extension("bpk")
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

    fn burnpack_metadata_path(path: &Path) -> PathBuf {
        let file_name = path
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_else(|| "model.bpk".to_string());
        path.with_file_name(format!("{file_name}.meta.json"))
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn save_rmbg2_onnx_blob_to_burnpack(
        output_path: &Path,
        bytes: &[u8],
        metadata: &Rmbg2OnnxBlobMetadata,
    ) -> Result<(), Box<dyn std::error::Error>> {
        type BlobBackend = burn::backend::NdArray<f32>;
        let device = <BlobBackend as Backend>::Device::default();

        let tensor = Tensor::<BlobBackend, 1, Int>::from_data(
            TensorData::new(bytes.to_vec(), [bytes.len()]),
            &device,
        );
        let blob = Rmbg2OnnxBlob {
            bytes: Param::initialized(ParamId::new(), tensor),
        };

        let mut store = BurnpackStore::from_file(output_path).overwrite(true);
        blob.save_into(&mut store).map_err(|err| {
            format!(
                "failed to save RMBG-2.0 burnpack {}: {err}",
                output_path.display()
            )
        })?;

        let metadata_path = burnpack_metadata_path(output_path);
        let metadata_json = serde_json::to_vec_pretty(metadata)?;
        fs::write(&metadata_path, metadata_json).map_err(|err| {
            format!(
                "failed to write RMBG-2.0 burnpack metadata {}: {err}",
                metadata_path.display()
            )
        })?;

        Ok(())
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

    #[cfg(all(test, not(target_arch = "wasm32")))]
    mod tests {
        use super::*;
        use std::time::{SystemTime, UNIX_EPOCH};

        #[test]
        fn rmbg2_burnpack_blob_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
            let unique = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
            let root = std::env::temp_dir().join(format!("burn_foreground_rmbg2_{unique}"));
            let onnx_root = root.join("onnx");
            fs::create_dir_all(&onnx_root)?;

            let f32_bytes = b"fake_rmbg2_onnx_f32".to_vec();
            let f16_bytes = b"fake_rmbg2_onnx_f16".to_vec();
            fs::write(onnx_root.join("model.onnx"), &f32_bytes)?;
            fs::write(onnx_root.join("model_fp16.onnx"), &f16_bytes)?;

            let f32_bpk = import_rmbg2_burnpack(&root, false)?;
            let f16_bpk = import_rmbg2_burnpack(&root, true)?;

            assert!(f32_bpk.exists());
            assert!(f16_bpk.exists());
            assert!(burnpack_metadata_path(&f32_bpk).exists());
            assert!(burnpack_metadata_path(&f16_bpk).exists());

            let loaded_f32 = load_rmbg2_onnx_blob_from_burnpack(&f32_bpk)?;
            let loaded_f16 = load_rmbg2_onnx_blob_from_burnpack(&f16_bpk)?;
            assert_eq!(loaded_f32, f32_bytes);
            assert_eq!(loaded_f16, f16_bytes);

            fs::remove_dir_all(&root)?;
            Ok(())
        }
    }
}

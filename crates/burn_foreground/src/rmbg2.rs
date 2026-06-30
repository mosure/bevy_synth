use std::path::Path;

use burn::backend::NdArray;
#[cfg(feature = "import")]
use burn::backend::ndarray::NdArrayDevice;

use crate::pipeline::{
    PrepareImageConfig, PrepareImageError, PreparedImageData, RmbgPipeline, prepare_image_data,
};

/// Burn-only RMBG-2.0 compatibility pipeline.
///
/// NOTE:
/// This crate no longer depends on ORT/ONNX runtime. Until a dedicated Burn-native
/// RMBG-2.0 model implementation lands, this type reuses the Burn RMBG loader path.
#[derive(Debug)]
pub struct Rmbg2Pipeline {
    inner: RmbgPipeline<NdArray<f32>>,
}

impl Rmbg2Pipeline {
    pub fn new(inner: RmbgPipeline<NdArray<f32>>) -> Self {
        Self { inner }
    }

    #[cfg(feature = "import")]
    pub fn from_pretrained(
        weights_root: impl AsRef<Path>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        use crate::rmbg14::import::{load_rmbg, load_rmbg_config, load_rmbg_processor_config};

        let root = weights_root.as_ref();
        let device = NdArrayDevice::default();
        let config = load_rmbg_config(root)?;
        let weights_path = root.join("model.safetensors");
        let model = load_rmbg(&device, weights_path, &config)?;
        let processor = load_rmbg_processor_config(root)?;

        Ok(Self::new(RmbgPipeline::new(model, processor)))
    }

    pub fn prepare_image_data(
        &self,
        path: &Path,
        config: &PrepareImageConfig,
    ) -> Result<PreparedImageData, PrepareImageError> {
        prepare_image_data::<NdArray<f32>>(path, Some(&self.inner), config)
    }
}

#[cfg(feature = "import")]
pub mod import {
    use std::path::{Path, PathBuf};

    use burn::backend::NdArray;
    use burn::backend::ndarray::NdArrayDevice;

    use crate::preprocess::RmbgImageProcessor;
    use crate::rmbg14::RmbgConfig;
    use crate::rmbg14::import::{
        import_rmbg_burnpack, load_rmbg_config, load_rmbg_processor_config,
        load_rmbg_processor_from_json_bytes, resolve_rmbg_weights_root,
    };

    pub fn resolve_rmbg2_weights_root() -> PathBuf {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let rmbg2 = manifest.join("assets/models/RMBG-2.0");
        if has_burn_weights(&rmbg2) {
            return rmbg2;
        }

        let rmbg14 = resolve_rmbg_weights_root();
        if has_burn_weights(&rmbg14) {
            return rmbg14;
        }

        rmbg14
    }

    pub fn import_rmbg2_burnpack(
        root: impl AsRef<Path>,
        use_f16: bool,
    ) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let root = root.as_ref();
        let device = NdArrayDevice::default();
        let config = load_rmbg_config(root).unwrap_or_else(|_| RmbgConfig::rmbg_1_4());
        let weights_path = root.join("model.safetensors");
        import_rmbg_burnpack::<NdArray<f32>>(&device, weights_path, &config, use_f16)
    }

    pub fn load_rmbg2_processor_config(
        root: impl AsRef<Path>,
    ) -> Result<RmbgImageProcessor, Box<dyn std::error::Error>> {
        load_rmbg_processor_config(root)
    }

    pub fn load_rmbg2_processor_from_json_bytes(
        bytes: &[u8],
    ) -> Result<RmbgImageProcessor, Box<dyn std::error::Error>> {
        load_rmbg_processor_from_json_bytes(bytes)
    }

    fn has_burn_weights(root: &Path) -> bool {
        let candidates = [
            root.join("model.safetensors"),
            root.join("model.bpk"),
            root.join("model_f16.bpk"),
            root.join("model.bpk.parts.json"),
            root.join("model_f16.bpk.parts.json"),
        ];
        candidates.into_iter().any(|path| path.exists())
    }
}

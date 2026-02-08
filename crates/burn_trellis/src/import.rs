use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use burn::module::{Module, Param, ParamId};
use burn::prelude::*;
use burn_store::{BurnpackStore, ModuleSnapshot};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::paths::{resolve_trellis2_image_large_root, resolve_trellis2_weights_root};

const F16_SUFFIX: &str = "_f16";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QuantizationMode {
    F32,
    F16,
    Both,
}

impl QuantizationMode {
    pub fn include_f32(self) -> bool {
        matches!(self, Self::F32 | Self::Both)
    }

    pub fn include_f16(self) -> bool {
        matches!(self, Self::F16 | Self::Both)
    }
}

#[derive(Clone, Debug)]
pub struct TrellisImportOptions {
    pub weights_root: PathBuf,
    pub image_large_root: Option<PathBuf>,
    pub output_root: PathBuf,
    pub quantization: QuantizationMode,
    pub overwrite: bool,
}

impl Default for TrellisImportOptions {
    fn default() -> Self {
        Self {
            weights_root: resolve_trellis2_weights_root(None),
            image_large_root: Some(resolve_trellis2_image_large_root(None)),
            output_root: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("assets/models/TRELLIS.2-4B"),
            quantization: QuantizationMode::Both,
            overwrite: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportedBlobInfo {
    pub source: String,
    pub output: String,
    pub precision: String,
    pub bytes_len: usize,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrellisImportManifest {
    pub weights_root: String,
    pub image_large_root: Option<String>,
    pub imported_blobs: Vec<ImportedBlobInfo>,
    pub copied_json_files: Vec<String>,
    pub missing_sources: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct TrellisImportReport {
    pub manifest_path: PathBuf,
    pub manifest: TrellisImportManifest,
}

#[derive(Module, Debug)]
struct BinaryBlob<B: Backend> {
    bytes: Param<Tensor<B, 1, Int>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BlobMetadata {
    bytes_len: usize,
    source_path: String,
    sha256: String,
    precision: String,
}

pub fn import_trellis2_assets(
    options: &TrellisImportOptions,
) -> Result<TrellisImportReport, Box<dyn std::error::Error>> {
    let weights_root = resolve_trellis2_weights_root(Some(options.weights_root.as_path()));
    let image_large_root = options
        .image_large_root
        .as_ref()
        .map(|root| resolve_trellis2_image_large_root(Some(root.as_path())));
    let output_root = options.output_root.clone();
    fs::create_dir_all(&output_root)?;

    let pipeline_path = weights_root.join("pipeline.json");
    let pipeline_bytes = fs::read(&pipeline_path)?;
    let pipeline_json: Value = serde_json::from_slice(&pipeline_bytes)?;

    let mut copied_json_files = Vec::new();
    let mut imported_blobs = Vec::new();
    let mut missing_sources = Vec::new();

    let output_pipeline_path = output_root.join("pipeline.json");
    copy_if_needed(&pipeline_path, &output_pipeline_path, options.overwrite)?;
    copied_json_files.push(output_pipeline_path.display().to_string());

    for stem in collect_model_stems(&pipeline_json) {
        let source_json =
            resolve_model_source_path(&stem, "json", &weights_root, image_large_root.as_deref());
        let source_safetensors = resolve_model_source_path(
            &stem,
            "safetensors",
            &weights_root,
            image_large_root.as_deref(),
        );
        let relative_json = model_relative_path(&stem, "json");
        let relative_safetensors = model_relative_path(&stem, "safetensors");
        let output_json = output_root.join(relative_json);
        let output_bpk = output_root.join(relative_safetensors).with_extension("bpk");

        if source_json.exists() {
            copy_if_needed(&source_json, &output_json, options.overwrite)?;
            copied_json_files.push(output_json.display().to_string());
        } else {
            missing_sources.push(source_json.display().to_string());
        }

        if !source_safetensors.exists() {
            missing_sources.push(source_safetensors.display().to_string());
            continue;
        }

        if options.quantization.include_f32() {
            let info =
                import_blob_file(&source_safetensors, &output_bpk, "f32", options.overwrite)?;
            imported_blobs.push(info);
        }
        if options.quantization.include_f16() {
            let output_bpk_f16 = with_file_stem_suffix(&output_bpk, F16_SUFFIX);
            let info = import_blob_file(
                &source_safetensors,
                &output_bpk_f16,
                "f16",
                options.overwrite,
            )?;
            imported_blobs.push(info);
        }
    }

    let manifest = TrellisImportManifest {
        weights_root: weights_root.display().to_string(),
        image_large_root: image_large_root.map(|path| path.display().to_string()),
        imported_blobs,
        copied_json_files,
        missing_sources,
    };
    let manifest_path = output_root.join("trellis2_import_manifest.json");
    fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)?;

    Ok(TrellisImportReport {
        manifest_path,
        manifest,
    })
}

pub fn load_burnpack_blob_bytes(
    burnpack_path: impl AsRef<Path>,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    type BlobBackend = burn::backend::NdArray<f32>;

    let burnpack_path = burnpack_path.as_ref();
    let metadata_path = metadata_path(burnpack_path);
    let metadata: BlobMetadata = serde_json::from_slice(&fs::read(&metadata_path)?)?;

    let device = <BlobBackend as Backend>::Device::default();
    let zeros = Tensor::<BlobBackend, 1, Int>::zeros([metadata.bytes_len], &device);
    let mut blob = BinaryBlob {
        bytes: Param::initialized(ParamId::new(), zeros),
    };

    let mut store = BurnpackStore::from_file(burnpack_path).validate(true);
    blob.load_from(&mut store).map_err(|err| {
        format!(
            "failed to load burnpack '{}': {err}",
            burnpack_path.display()
        )
    })?;

    let bytes = blob
        .bytes
        .val()
        .into_data()
        .convert::<u8>()
        .to_vec::<u8>()
        .map_err(|err| format!("failed to materialize burnpack bytes: {err:?}"))?;

    if bytes.len() != metadata.bytes_len {
        return Err(format!(
            "burnpack byte length mismatch for '{}': expected {}, got {}",
            burnpack_path.display(),
            metadata.bytes_len,
            bytes.len()
        )
        .into());
    }
    Ok(bytes)
}

fn import_blob_file(
    source_path: &Path,
    burnpack_path: &Path,
    precision: &str,
    overwrite: bool,
) -> Result<ImportedBlobInfo, Box<dyn std::error::Error>> {
    if burnpack_path.exists() && !overwrite {
        let bytes = fs::read(source_path)?;
        return Ok(ImportedBlobInfo {
            source: source_path.display().to_string(),
            output: burnpack_path.display().to_string(),
            precision: precision.to_string(),
            bytes_len: bytes.len(),
            sha256: hex::encode(Sha256::digest(bytes.as_slice())),
        });
    }

    if let Some(parent) = burnpack_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let bytes = fs::read(source_path)?;
    save_blob_to_burnpack(burnpack_path, &bytes)?;
    let sha256 = hex::encode(Sha256::digest(bytes.as_slice()));
    let metadata = BlobMetadata {
        bytes_len: bytes.len(),
        source_path: source_path.display().to_string(),
        sha256: sha256.clone(),
        precision: precision.to_string(),
    };
    fs::write(
        metadata_path(burnpack_path),
        serde_json::to_vec_pretty(&metadata)?,
    )?;

    Ok(ImportedBlobInfo {
        source: source_path.display().to_string(),
        output: burnpack_path.display().to_string(),
        precision: precision.to_string(),
        bytes_len: bytes.len(),
        sha256,
    })
}

fn save_blob_to_burnpack(
    burnpack_path: &Path,
    bytes: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    type BlobBackend = burn::backend::NdArray<f32>;
    let device = <BlobBackend as Backend>::Device::default();
    let tensor = Tensor::<BlobBackend, 1, Int>::from_data(
        TensorData::new(bytes.to_vec(), [bytes.len()]),
        &device,
    );
    let blob = BinaryBlob {
        bytes: Param::initialized(ParamId::new(), tensor),
    };
    let mut store = BurnpackStore::from_file(burnpack_path).overwrite(true);
    blob.save_into(&mut store).map_err(|err| {
        format!(
            "failed to write burnpack '{}': {err}",
            burnpack_path.display()
        )
    })?;
    Ok(())
}

fn collect_model_stems(pipeline_json: &Value) -> Vec<String> {
    let mut stems = BTreeSet::new();
    let maybe_models = pipeline_json
        .get("args")
        .and_then(|value| value.get("models"))
        .and_then(Value::as_object);
    if let Some(models) = maybe_models {
        for value in models.values() {
            if let Some(stem) = value.as_str() {
                stems.insert(stem.to_string());
            }
        }
    }
    stems.into_iter().collect()
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

fn model_relative_path(stem: &str, ext: &str) -> PathBuf {
    if stem.starts_with("ckpts/") {
        return PathBuf::from(format!("{stem}.{ext}"));
    }
    if let Some((_, suffix)) = stem.split_once("/ckpts/") {
        return PathBuf::from(format!("ckpts/{suffix}.{ext}"));
    }
    PathBuf::from(format!("{stem}.{ext}"))
}

fn copy_if_needed(source: &Path, destination: &Path, overwrite: bool) -> std::io::Result<()> {
    if destination.exists() && !overwrite {
        return Ok(());
    }
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)?;
    }
    let _ = fs::copy(source, destination)?;
    Ok(())
}

fn metadata_path(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("model.bpk");
    path.with_file_name(format!("{file_name}.meta.json"))
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
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{
        QuantizationMode, TrellisImportOptions, import_trellis2_assets, load_burnpack_blob_bytes,
    };

    #[test]
    fn imports_pipeline_assets_and_roundtrips_blob_bytes() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock drift")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("burn_trellis_import_{unique}"));
        let output = root.join("out");
        let ckpts = root.join("ckpts");
        std::fs::create_dir_all(&ckpts).expect("failed to create ckpt dir");

        let pipeline = root.join("pipeline.json");
        let pipeline_json = r#"{
            "args": {
                "models": {
                    "shape": "ckpts/shape"
                }
            }
        }"#;
        std::fs::write(&pipeline, pipeline_json).expect("failed to write pipeline");
        std::fs::write(ckpts.join("shape.json"), "{}").expect("failed to write model json");
        let source_path = ckpts.join("shape.safetensors");
        let mut file = std::fs::File::create(&source_path).expect("failed to create source");
        file.write_all(b"fake_safetensor_bytes")
            .expect("failed to write source");

        let report = import_trellis2_assets(&TrellisImportOptions {
            weights_root: root.clone(),
            image_large_root: None,
            output_root: output.clone(),
            quantization: QuantizationMode::Both,
            overwrite: true,
        })
        .expect("import should succeed");
        assert!(report.manifest.missing_sources.is_empty());

        let f32_bpk = output.join("ckpts/shape.bpk");
        let f16_bpk = output.join("ckpts/shape_f16.bpk");
        assert!(f32_bpk.exists());
        assert!(f16_bpk.exists());
        let bytes_f32 = load_burnpack_blob_bytes(&f32_bpk).expect("load f32");
        let bytes_f16 = load_burnpack_blob_bytes(&f16_bpk).expect("load f16");
        assert_eq!(bytes_f32, b"fake_safetensor_bytes");
        assert_eq!(bytes_f16, b"fake_safetensor_bytes");

        let _ = std::fs::remove_dir_all(root);
    }
}

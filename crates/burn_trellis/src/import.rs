use std::borrow::Cow;
use std::cmp::Reverse;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

#[cfg(test)]
use burn::module::{Module, Param};
#[cfg(test)]
use burn::prelude::Backend;
#[cfg(test)]
use burn::tensor::{Int, Tensor};
use burn_synth_import::layout::stem_has_low_precision_marker;
use burn_synth_import::parts::{
    burnpack_parts_manifest_path, read_parts_manifest, resolve_part_entry_path,
};
use half::{bf16, f16};
use safetensors::tensor::{Dtype, SafeTensors, View, serialize};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::blob_burnpack::{
    DEFAULT_BLOB_CHUNK_BYTES, load_blob_bytes_from_burnpack as load_blob_bytes_from_blob_burnpack,
    save_blob_bytes_to_burnpack,
};
use crate::paths::{
    resolve_trellis2_weights_root, trellis2_repo_asset_root, trellis2_repo_image_large_root,
};
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
    pub image_large_output_root: Option<PathBuf>,
    pub quantization: QuantizationMode,
    pub overwrite: bool,
}

impl Default for TrellisImportOptions {
    fn default() -> Self {
        Self {
            weights_root: resolve_trellis2_weights_root(None),
            image_large_root: Some(trellis2_repo_image_large_root()),
            output_root: trellis2_repo_asset_root(),
            image_large_output_root: Some(trellis2_repo_image_large_root()),
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
    pub image_large_output_root: Option<String>,
    pub imported_blobs: Vec<ImportedBlobInfo>,
    pub copied_json_files: Vec<String>,
    pub missing_sources: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct TrellisImportReport {
    pub manifest_path: PathBuf,
    pub manifest: TrellisImportManifest,
}

#[cfg(test)]
#[derive(Module, Debug)]
struct BinaryBlob<B: Backend> {
    bytes: Param<Tensor<B, 1, Int>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ModelImportKind {
    SparseFlowModule,
    SafetensorsBlob,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BlobMetadata {
    bytes_len: usize,
    source_path: String,
    sha256: String,
    precision: String,
    #[serde(default)]
    chunk_sizes: Vec<usize>,
    #[serde(default)]
    chunk_bytes: usize,
}

#[derive(Debug, Clone)]
struct ImageConditioningSourceAssets {
    model_weights: PathBuf,
    config_json: Option<PathBuf>,
}

#[derive(Debug, Clone)]
struct OwnedTensorData {
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl View for &OwnedTensorData {
    fn dtype(&self) -> Dtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        self.shape.as_slice()
    }

    fn data(&self) -> Cow<'_, [u8]> {
        self.data.as_slice().into()
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

pub fn import_trellis2_assets(
    options: &TrellisImportOptions,
) -> Result<TrellisImportReport, Box<dyn std::error::Error>> {
    let weights_root = resolve_trellis2_weights_root(Some(options.weights_root.as_path()));
    let image_large_root = options
        .image_large_root
        .clone()
        .or_else(|| Some(trellis2_repo_image_large_root()));
    let output_root = options.output_root.clone();
    let image_large_output_root = options.image_large_output_root.clone().or_else(|| {
        if image_large_root.is_some() {
            Some(trellis2_repo_image_large_root())
        } else {
            None
        }
    });
    fs::create_dir_all(&output_root)?;
    if let Some(image_large_output_root) = image_large_output_root.as_ref() {
        fs::create_dir_all(image_large_output_root)?;
    }

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
        let model_in_image_large = is_image_large_stem(&stem);
        let source_json =
            resolve_model_source_path(&stem, "json", &weights_root, image_large_root.as_deref());
        let source_safetensors = resolve_model_source_path(
            &stem,
            "safetensors",
            &weights_root,
            image_large_root.as_deref(),
        );
        let model_output_root = if model_in_image_large {
            image_large_output_root.as_ref().unwrap_or(&output_root)
        } else {
            &output_root
        };
        let relative_json = model_relative_path(&stem, "json");
        let relative_safetensors = model_relative_path(&stem, "safetensors");
        let output_json = model_output_root.join(relative_json);
        let output_bpk = model_output_root
            .join(relative_safetensors)
            .with_extension("bpk");
        let import_kind = if source_json.exists() {
            model_import_kind(&source_json)
        } else {
            ModelImportKind::SafetensorsBlob
        };

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

        let source_low_precision = source_is_native_low_precision(source_safetensors.as_path());
        if source_low_precision {
            let redundant_suffix = with_file_stem_suffix(&output_bpk, F16_SUFFIX);
            if redundant_suffix != output_bpk {
                remove_burnpack_family_if_present(&redundant_suffix)?;
            }
        }
        let emit_f32 = options.quantization.include_f32() && !source_low_precision;
        let emit_f16 = options.quantization.include_f16()
            || (source_low_precision && options.quantization.include_f32());

        if emit_f32 {
            let info = import_model_file(
                &source_json,
                &source_safetensors,
                &output_bpk,
                import_kind,
                "f32",
                options.overwrite,
            )?;
            imported_blobs.push(info);
        }
        if emit_f16 {
            let output_bpk_f16 = if source_low_precision {
                output_bpk.clone()
            } else {
                with_file_stem_suffix(&output_bpk, F16_SUFFIX)
            };
            let info = import_model_file(
                &source_json,
                &source_safetensors,
                &output_bpk_f16,
                import_kind,
                "f16",
                options.overwrite,
            )?;
            imported_blobs.push(info);
        }
    }

    if let Some(model_name) = collect_image_conditioning_model_name(&pipeline_json) {
        match resolve_image_conditioning_source_assets(
            &weights_root,
            image_large_root.as_deref(),
            model_name.as_str(),
        ) {
            Some(source_assets) => {
                let output_model_dir = output_root.join(model_name.as_str());
                fs::create_dir_all(&output_model_dir)?;

                if let Some(source_config) = source_assets.config_json.as_ref() {
                    let output_config = output_model_dir.join("config.json");
                    copy_if_needed(source_config, &output_config, options.overwrite)?;
                    copied_json_files.push(output_config.display().to_string());
                }

                let output_bpk = output_model_dir.join("model.bpk");
                if options.quantization.include_f32() {
                    let info = import_blob_file(
                        source_assets.model_weights.as_path(),
                        &output_bpk,
                        "f32",
                        options.overwrite,
                    )?;
                    imported_blobs.push(info);
                }
                if options.quantization.include_f16() {
                    let output_bpk_f16 = with_file_stem_suffix(&output_bpk, F16_SUFFIX);
                    let info = import_blob_file(
                        source_assets.model_weights.as_path(),
                        &output_bpk_f16,
                        "f16",
                        options.overwrite,
                    )?;
                    imported_blobs.push(info);
                }
            }
            None => {
                missing_sources.push(format!(
                    "image conditioning weights for '{}' (expected model.safetensors under explicit/local paths or Hugging Face cache snapshots)",
                    model_name
                ));
            }
        }
    }

    let manifest = TrellisImportManifest {
        weights_root: weights_root.display().to_string(),
        image_large_root: image_large_root.map(|path| path.display().to_string()),
        image_large_output_root: image_large_output_root.map(|path| path.display().to_string()),
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

pub fn extract_blob_burnpack(
    burnpack_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
) -> Result<usize, Box<dyn std::error::Error>> {
    let burnpack_path = burnpack_path.as_ref();
    let output_path = output_path.as_ref();
    let bytes = load_blob_bytes_from_blob_burnpack(burnpack_path)?;
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(output_path, bytes.as_slice())?;
    Ok(bytes.len())
}

pub fn load_burnpack_blob_bytes(
    burnpack_path: impl AsRef<Path>,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let burnpack_path = burnpack_path.as_ref();
    let metadata = if metadata_path(burnpack_path).exists() {
        Some(serde_json::from_slice::<BlobMetadata>(
            fs::read(metadata_path(burnpack_path))?.as_slice(),
        )?)
    } else {
        None
    };

    let bytes = load_blob_bytes_from_blob_burnpack(burnpack_path).map_err(|err| {
        format!(
            "failed to load blob burnpack '{}': {err}",
            burnpack_path.display()
        )
    })?;
    if let Some(metadata) = metadata
        && metadata.bytes_len > 0
        && bytes.len() != metadata.bytes_len
    {
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

fn import_model_file(
    source_config_path: &Path,
    source_path: &Path,
    burnpack_path: &Path,
    kind: ModelImportKind,
    precision: &str,
    overwrite: bool,
) -> Result<ImportedBlobInfo, Box<dyn std::error::Error>> {
    if matches!(kind, ModelImportKind::SparseFlowModule) {
        return import_sparse_flow_module_file(
            source_config_path,
            source_path,
            burnpack_path,
            precision,
            overwrite,
        );
    }
    import_blob_file(source_path, burnpack_path, precision, overwrite)
}

fn import_blob_file(
    source_path: &Path,
    burnpack_path: &Path,
    precision: &str,
    overwrite: bool,
) -> Result<ImportedBlobInfo, Box<dyn std::error::Error>> {
    if burnpack_path.exists() && !overwrite {
        if let Some(from_metadata) =
            imported_blob_info_from_metadata(source_path, burnpack_path, precision)?
        {
            return Ok(from_metadata);
        }

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

    let source_bytes = fs::read(source_path)?;
    let bytes = prepare_blob_payload(source_bytes.as_slice(), precision);
    let chunk_sizes = save_blob_to_burnpack(burnpack_path, bytes.as_slice())?;
    let sha256 = hex::encode(Sha256::digest(bytes.as_slice()));
    let metadata = BlobMetadata {
        bytes_len: bytes.len(),
        source_path: source_path.display().to_string(),
        sha256: sha256.clone(),
        precision: precision.to_string(),
        chunk_bytes: chunk_sizes.iter().copied().max().unwrap_or(0),
        chunk_sizes,
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

fn imported_blob_info_from_metadata(
    source_path: &Path,
    burnpack_path: &Path,
    precision: &str,
) -> Result<Option<ImportedBlobInfo>, Box<dyn std::error::Error>> {
    let metadata_path = metadata_path(burnpack_path);
    if !metadata_path.exists() {
        return Ok(None);
    }

    let metadata: BlobMetadata = serde_json::from_slice(&fs::read(metadata_path)?)?;
    Ok(Some(ImportedBlobInfo {
        source: source_path.display().to_string(),
        output: burnpack_path.display().to_string(),
        precision: if metadata.precision.is_empty() {
            precision.to_string()
        } else {
            metadata.precision
        },
        bytes_len: metadata.bytes_len,
        sha256: metadata.sha256,
    }))
}

fn prepare_blob_payload(source_bytes: &[u8], precision: &str) -> Vec<u8> {
    if !precision.eq_ignore_ascii_case("f16") {
        return source_bytes.to_vec();
    }

    if safetensors_contains_low_precision_float(source_bytes).unwrap_or(false) {
        return source_bytes.to_vec();
    }

    convert_safetensors_blob_to_f16(source_bytes).unwrap_or_else(|_| source_bytes.to_vec())
}

fn safetensors_contains_low_precision_float(source_bytes: &[u8]) -> Result<bool, String> {
    let safetensors = SafeTensors::deserialize(source_bytes).map_err(|err| {
        format!("failed to deserialize safetensors for low-precision scan: {err}")
    })?;
    Ok(safetensors
        .tensors()
        .into_iter()
        .any(|(_, view)| matches!(view.dtype(), Dtype::F16 | Dtype::BF16)))
}

fn convert_safetensors_blob_to_f16(source_bytes: &[u8]) -> Result<Vec<u8>, String> {
    let safetensors = SafeTensors::deserialize(source_bytes)
        .map_err(|err| format!("failed to deserialize safetensors for f16 conversion: {err}"))?;
    let (_, metadata) = SafeTensors::read_metadata(source_bytes)
        .map_err(|err| format!("failed to read safetensors metadata for f16 conversion: {err}"))?;

    let mut converted = Vec::with_capacity(safetensors.len());
    for (name, view) in safetensors.tensors() {
        let (dtype, data) = match view.dtype() {
            Dtype::F16 => (Dtype::F16, view.data().to_vec()),
            Dtype::BF16 => (Dtype::F16, bf16_bytes_to_f16_bytes(view.data())?),
            Dtype::F32 => (Dtype::F16, f32_bytes_to_f16_bytes(view.data())?),
            Dtype::F64 => (Dtype::F16, f64_bytes_to_f16_bytes(view.data())?),
            other => (other, view.data().to_vec()),
        };
        converted.push((
            name,
            OwnedTensorData {
                dtype,
                shape: view.shape().to_vec(),
                data,
            },
        ));
    }

    serialize(
        converted
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor)),
        metadata.metadata().clone(),
    )
    .map_err(|err| format!("failed to serialize f16 safetensors: {err}"))
}

fn f32_bytes_to_f16_bytes(bytes: &[u8]) -> Result<Vec<u8>, String> {
    if !bytes.len().is_multiple_of(4) {
        return Err(format!(
            "invalid f32 tensor payload byte length {}; must be divisible by 4",
            bytes.len()
        ));
    }
    let mut out = Vec::with_capacity((bytes.len() / 4) * 2);
    for chunk in bytes.chunks_exact(4) {
        let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        out.extend_from_slice(&f16::from_f32(value).to_bits().to_le_bytes());
    }
    Ok(out)
}

fn f64_bytes_to_f16_bytes(bytes: &[u8]) -> Result<Vec<u8>, String> {
    if !bytes.len().is_multiple_of(8) {
        return Err(format!(
            "invalid f64 tensor payload byte length {}; must be divisible by 8",
            bytes.len()
        ));
    }
    let mut out = Vec::with_capacity((bytes.len() / 8) * 2);
    for chunk in bytes.chunks_exact(8) {
        let value = f64::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]) as f32;
        out.extend_from_slice(&f16::from_f32(value).to_bits().to_le_bytes());
    }
    Ok(out)
}

fn bf16_bytes_to_f16_bytes(bytes: &[u8]) -> Result<Vec<u8>, String> {
    if !bytes.len().is_multiple_of(2) {
        return Err(format!(
            "invalid bf16 tensor payload byte length {}; must be divisible by 2",
            bytes.len()
        ));
    }
    let mut out = Vec::with_capacity(bytes.len());
    for chunk in bytes.chunks_exact(2) {
        let value = bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32();
        out.extend_from_slice(&f16::from_f32(value).to_bits().to_le_bytes());
    }
    Ok(out)
}

fn save_blob_to_burnpack(
    burnpack_path: &Path,
    bytes: &[u8],
) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let chunk_sizes = save_blob_bytes_to_burnpack(burnpack_path, bytes, DEFAULT_BLOB_CHUNK_BYTES)
        .map_err(|err| {
        format!(
            "failed to write burnpack '{}': {err}",
            burnpack_path.display()
        )
    })?;
    Ok(chunk_sizes)
}

fn import_sparse_flow_module_file(
    source_config_path: &Path,
    source_safetensors_path: &Path,
    burnpack_path: &Path,
    precision: &str,
    overwrite: bool,
) -> Result<ImportedBlobInfo, Box<dyn std::error::Error>> {
    // Store sparse flow checkpoints as raw safetensor blobs inside burnpack to keep
    // footprint close to source files and avoid dtype widening inflation.
    let _ = source_config_path;
    import_blob_file(source_safetensors_path, burnpack_path, precision, overwrite)
}

fn model_import_kind(source_config_path: &Path) -> ModelImportKind {
    let Ok(bytes) = fs::read(source_config_path) else {
        return ModelImportKind::SafetensorsBlob;
    };
    let Ok(value) = serde_json::from_slice::<Value>(&bytes) else {
        return ModelImportKind::SafetensorsBlob;
    };
    let Some(name) = value.get("name").and_then(Value::as_str) else {
        return ModelImportKind::SafetensorsBlob;
    };

    if matches!(name, "SparseStructureFlowModel" | "SLatFlowModel") {
        ModelImportKind::SparseFlowModule
    } else {
        ModelImportKind::SafetensorsBlob
    }
}

fn source_is_native_low_precision(source_path: &Path) -> bool {
    source_path
        .file_stem()
        .and_then(|value| value.to_str())
        .is_some_and(stem_has_low_precision_marker)
}

fn remove_burnpack_family_if_present(
    burnpack_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    if burnpack_path.exists() {
        fs::remove_file(burnpack_path)?;
    }
    let meta = metadata_path(burnpack_path);
    if meta.exists() {
        fs::remove_file(meta)?;
    }
    let parts_manifest = burnpack_parts_manifest_path(burnpack_path);
    if parts_manifest.exists() {
        if let Ok(manifest) = read_parts_manifest(&parts_manifest) {
            for entry in manifest.parts {
                let part_path = resolve_part_entry_path(&parts_manifest, &entry.path)?;
                if part_path.exists() {
                    fs::remove_file(part_path)?;
                }
            }
        }
        fs::remove_file(parts_manifest)?;
    }
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

fn collect_image_conditioning_model_name(pipeline_json: &Value) -> Option<String> {
    pipeline_json
        .get("args")
        .and_then(|value| value.get("image_cond_model"))
        .and_then(|value| value.get("args"))
        .and_then(|value| value.get("model_name"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn resolve_image_conditioning_source_assets(
    weights_root: &Path,
    image_large_root: Option<&Path>,
    model_name: &str,
) -> Option<ImageConditioningSourceAssets> {
    let mut candidates = Vec::new();
    candidates.push(PathBuf::from(model_name));
    candidates.push(weights_root.join(model_name));
    if let Some(root) = image_large_root {
        candidates.push(root.join(model_name));
    }

    for candidate in candidates {
        if let Some(assets) = image_conditioning_assets_from_candidate_path(candidate.as_path()) {
            return Some(assets);
        }
    }

    resolve_hf_hub_snapshot_weights(model_name).map(|model_weights| ImageConditioningSourceAssets {
        config_json: model_weights.parent().map(|path| path.join("config.json")),
        model_weights,
    })
}

fn image_conditioning_assets_from_candidate_path(
    candidate: &Path,
) -> Option<ImageConditioningSourceAssets> {
    if !candidate.exists() {
        return None;
    }

    if candidate.is_file() {
        let extension = candidate.extension().and_then(|value| value.to_str())?;
        if !extension.eq_ignore_ascii_case("safetensors") {
            return None;
        }
        return Some(ImageConditioningSourceAssets {
            model_weights: candidate.to_path_buf(),
            config_json: candidate.parent().map(|path| path.join("config.json")),
        });
    }

    let weights = candidate.join("model.safetensors");
    if !weights.exists() {
        return None;
    }

    Some(ImageConditioningSourceAssets {
        model_weights: weights,
        config_json: Some(candidate.join("config.json")),
    })
}

fn resolve_hf_hub_snapshot_weights(model_name: &str) -> Option<PathBuf> {
    let mut matches = Vec::new();
    let suffix = model_name.rsplit('/').next().unwrap_or(model_name);

    for hub_root in huggingface_hub_roots() {
        let exact_repo = hub_root.join(format!("models--{}", model_name.replace('/', "--")));
        if let Some(path) = resolve_snapshot_weights_from_repo(exact_repo.as_path()) {
            let modified = path
                .metadata()
                .and_then(|metadata| metadata.modified())
                .unwrap_or(SystemTime::UNIX_EPOCH);
            matches.push((modified, path));
        }

        let Ok(entries) = fs::read_dir(&hub_root) else {
            continue;
        };
        for entry in entries.flatten() {
            let file_name = entry.file_name();
            let file_name = file_name.to_string_lossy();
            if !file_name.starts_with("models--") {
                continue;
            }
            if !file_name.ends_with(&format!("--{suffix}")) {
                continue;
            }
            if let Some(path) = resolve_snapshot_weights_from_repo(entry.path().as_path()) {
                let modified = path
                    .metadata()
                    .and_then(|metadata| metadata.modified())
                    .unwrap_or(SystemTime::UNIX_EPOCH);
                matches.push((modified, path));
            }
        }
    }

    matches.sort_by_key(|entry| Reverse(entry.0));
    matches.into_iter().next().map(|(_, path)| path)
}

fn resolve_snapshot_weights_from_repo(repo_root: &Path) -> Option<PathBuf> {
    if !repo_root.is_dir() {
        return None;
    }

    let refs_main = repo_root.join("refs/main");
    if let Ok(revision) = fs::read_to_string(refs_main.as_path()) {
        let revision = revision.trim();
        if !revision.is_empty() {
            let snapshot = repo_root.join("snapshots").join(revision);
            let weights = snapshot.join("model.safetensors");
            if weights.exists() {
                return Some(weights);
            }
        }
    }

    let snapshots = repo_root.join("snapshots");
    let mut candidates = Vec::new();
    let entries = fs::read_dir(snapshots).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        let weights = path.join("model.safetensors");
        if !weights.exists() {
            continue;
        }
        let modified = weights
            .metadata()
            .and_then(|metadata| metadata.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        candidates.push((modified, weights));
    }
    candidates.sort_by_key(|entry| Reverse(entry.0));
    candidates.into_iter().next().map(|(_, path)| path)
}

fn huggingface_hub_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();

    if let Some(path) = std::env::var_os("HF_HUB_CACHE") {
        push_unique_path(&mut roots, PathBuf::from(path));
    }
    if let Some(path) = std::env::var_os("HUGGINGFACE_HUB_CACHE") {
        push_unique_path(&mut roots, PathBuf::from(path));
    }
    if let Some(path) = std::env::var_os("HF_HOME") {
        push_unique_path(&mut roots, PathBuf::from(path).join("hub"));
    }
    if let Some(path) = std::env::var_os("HUGGINGFACE_HOME") {
        push_unique_path(&mut roots, PathBuf::from(path).join("hub"));
    }
    if let Some(path) = std::env::var_os("XDG_CACHE_HOME") {
        push_unique_path(
            &mut roots,
            PathBuf::from(path).join("huggingface").join("hub"),
        );
    }
    if let Some(path) = std::env::var_os("HOME") {
        push_unique_path(
            &mut roots,
            PathBuf::from(path)
                .join(".cache")
                .join("huggingface")
                .join("hub"),
        );
    }

    for base in ["/media", "/mnt"] {
        let Ok(entries) = fs::read_dir(base) else {
            continue;
        };
        for entry in entries.flatten() {
            let candidate = entry.path().join("models").join("huggingface").join("hub");
            push_unique_path(&mut roots, candidate);
        }
    }

    roots.into_iter().filter(|path| path.is_dir()).collect()
}

fn push_unique_path(paths: &mut Vec<PathBuf>, candidate: PathBuf) {
    if !paths.iter().any(|existing| existing == &candidate) {
        paths.push(candidate);
    }
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
        let relative = format!("ckpts/{suffix}.{ext}");
        if let Some(image_large_root) = image_large_root {
            let image_candidate = image_large_root.join(&relative);
            if image_candidate.exists() {
                return image_candidate;
            }
            let weights_candidate = weights_root.join(&relative);
            if weights_candidate.exists() {
                return weights_candidate;
            }
            return image_candidate;
        }
        return weights_root.join(relative);
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

fn is_image_large_stem(stem: &str) -> bool {
    !stem.starts_with("ckpts/") && stem.split_once("/ckpts/").is_some()
}

fn copy_if_needed(source: &Path, destination: &Path, overwrite: bool) -> std::io::Result<()> {
    if source == destination {
        return Ok(());
    }
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
    use crate::runtime_model::sparse_structure_flow::{
        SparseStructureFlowConfig, SparseStructureFlowModel,
    };
    use burn::module::{Param, ParamId};
    use burn::prelude::*;
    use burn_store::{BurnpackStore, ModuleSnapshot, SafetensorsStore};
    use half::{bf16, f16};
    use safetensors::tensor::{Dtype, SafeTensors, TensorView, serialize};
    use std::io::Write;
    use std::sync::{Mutex, MutexGuard};
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{
        BinaryBlob, BlobMetadata, QuantizationMode, TrellisImportOptions, import_trellis2_assets,
        load_burnpack_blob_bytes, metadata_path, prepare_blob_payload,
    };

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn env_lock_guard() -> MutexGuard<'static, ()> {
        ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

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
            image_large_output_root: None,
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

    #[test]
    fn low_precision_sources_do_not_emit_redundant_f32_or_f16_suffixed_names() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock drift")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("burn_trellis_import_low_precision_{unique}"));
        let output = root.join("out");
        let ckpts = root.join("ckpts");
        std::fs::create_dir_all(&ckpts).expect("failed to create ckpt dir");

        let pipeline = root.join("pipeline.json");
        let pipeline_json = r#"{
            "args": {
                "models": {
                    "shape": "ckpts/shape_bf16"
                }
            }
        }"#;
        std::fs::write(&pipeline, pipeline_json).expect("failed to write pipeline");
        std::fs::write(ckpts.join("shape_bf16.json"), "{}").expect("failed to write model json");
        std::fs::write(
            ckpts.join("shape_bf16.safetensors"),
            b"fake_safetensor_bytes",
        )
        .expect("failed to write source");

        let report = import_trellis2_assets(&TrellisImportOptions {
            weights_root: root.clone(),
            image_large_root: None,
            output_root: output.clone(),
            image_large_output_root: None,
            quantization: QuantizationMode::Both,
            overwrite: true,
        })
        .expect("import should succeed");
        assert!(report.manifest.missing_sources.is_empty());

        let canonical = output.join("ckpts/shape_bf16.bpk");
        let redundant_suffix = output.join("ckpts/shape_bf16_f16.bpk");
        assert!(
            canonical.exists(),
            "expected canonical low-precision burnpack"
        );
        assert!(
            !redundant_suffix.exists(),
            "redundant *_f16 low-precision burnpack should not be emitted"
        );
        assert_eq!(
            report
                .manifest
                .imported_blobs
                .iter()
                .filter(|entry| entry.output.ends_with("shape_bf16.bpk"))
                .count(),
            1,
            "expected only one low-precision artifact record"
        );

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn imports_image_conditioning_model_into_f32_and_f16_burnpacks() {
        let _guard = env_lock_guard();

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock drift")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("burn_trellis_import_dino_{unique}"));
        let output = root.join("out");
        let ckpts = root.join("ckpts");
        let hf_root = root.join("hf").join("hub");
        std::fs::create_dir_all(&ckpts).expect("create ckpts");
        std::fs::create_dir_all(&hf_root).expect("create hf root");

        let model_name = format!("facebook/dinov3-unit-test-{unique}");
        let pipeline = serde_json::json!({
            "args": {
                "models": {
                    "shape": "ckpts/shape"
                },
                "image_cond_model": {
                    "name": "DinoV3FeatureExtractor",
                    "args": {
                        "model_name": model_name
                    }
                }
            }
        });
        std::fs::write(
            root.join("pipeline.json"),
            serde_json::to_vec_pretty(&pipeline).expect("serialize pipeline"),
        )
        .expect("write pipeline");
        std::fs::write(ckpts.join("shape.json"), "{}").expect("write shape json");
        std::fs::write(ckpts.join("shape.safetensors"), b"shape_bytes")
            .expect("write shape safetensors");

        let repo_dir = hf_root.join(format!("models--{}", model_name.replace('/', "--")));
        let snapshot = repo_dir.join("snapshots").join("rev1");
        std::fs::create_dir_all(&snapshot).expect("create dino snapshot");
        std::fs::create_dir_all(repo_dir.join("refs")).expect("create refs");
        std::fs::write(repo_dir.join("refs/main"), b"rev1").expect("write refs/main");
        std::fs::write(snapshot.join("config.json"), b"{}").expect("write dino config");
        let dino_bytes = [1.0f32, -2.0f32]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        let view =
            TensorView::new(Dtype::F32, vec![2], dino_bytes.as_slice()).expect("tensor view");
        let source =
            serialize(vec![("dummy".to_string(), view)], None).expect("serialize dino safetensors");
        std::fs::write(snapshot.join("model.safetensors"), source).expect("write dino weights");

        let prev_hf_hub_cache = std::env::var_os("HF_HUB_CACHE");
        unsafe {
            std::env::set_var("HF_HUB_CACHE", &hf_root);
        }

        let report = import_trellis2_assets(&TrellisImportOptions {
            weights_root: root.clone(),
            image_large_root: None,
            output_root: output.clone(),
            image_large_output_root: None,
            quantization: QuantizationMode::Both,
            overwrite: true,
        })
        .expect("import should succeed");

        match prev_hf_hub_cache {
            Some(value) => unsafe {
                std::env::set_var("HF_HUB_CACHE", value);
            },
            None => unsafe {
                std::env::remove_var("HF_HUB_CACHE");
            },
        }

        assert!(
            report.manifest.missing_sources.is_empty(),
            "missing sources: {:?}",
            report.manifest.missing_sources
        );

        let output_model_dir = output.join(model_name);
        let dino_f32 = output_model_dir.join("model.bpk");
        let dino_f16 = output_model_dir.join("model_f16.bpk");
        assert!(dino_f32.exists(), "missing image-cond f32 burnpack");
        assert!(dino_f16.exists(), "missing image-cond f16 burnpack");
        assert!(
            output_model_dir.join("config.json").exists(),
            "missing image-cond config copy"
        );

        let bytes_f32 = load_burnpack_blob_bytes(&dino_f32).expect("read image-cond f32 burnpack");
        let bytes_f16 = load_burnpack_blob_bytes(&dino_f16).expect("read image-cond f16 burnpack");
        assert!(!bytes_f32.is_empty());
        assert!(!bytes_f16.is_empty());

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn loads_legacy_i64_blob_burnpack() {
        type LegacyBlobBackend = burn::backend::NdArray<f32, i64>;

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock drift")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("burn_trellis_import_legacy_{unique}"));
        std::fs::create_dir_all(&root).expect("failed to create tmp dir");
        let burnpack = root.join("legacy_blob.bpk");

        let source = b"legacy_blob_bytes";
        let device = <LegacyBlobBackend as burn::tensor::backend::BackendTypes>::Device::default();
        let tensor = Tensor::<LegacyBlobBackend, 1, Int>::from_data(
            TensorData::new(source.to_vec(), [source.len()]),
            &device,
        );
        let blob = BinaryBlob {
            bytes: Param::initialized(ParamId::new(), tensor),
        };
        let mut store = BurnpackStore::from_file(&burnpack).overwrite(true);
        blob.save_into(&mut store)
            .expect("failed to save legacy blob");

        let metadata = BlobMetadata {
            bytes_len: source.len(),
            source_path: "legacy".to_string(),
            sha256: "legacy".to_string(),
            precision: "f32".to_string(),
            chunk_sizes: Vec::new(),
            chunk_bytes: 0,
        };
        std::fs::write(
            metadata_path(&burnpack),
            serde_json::to_vec_pretty(&metadata).expect("metadata json"),
        )
        .expect("metadata write");

        let bytes = load_burnpack_blob_bytes(&burnpack).expect("load legacy blob");
        assert_eq!(bytes, source);

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn imports_sparse_flow_models_with_size_guardrails() {
        type TestBackend = burn::backend::NdArray<f32>;

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock drift")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("burn_trellis_import_flow_{unique}"));
        let output = root.join("out");
        let ckpts = root.join("ckpts");
        std::fs::create_dir_all(&ckpts).expect("failed to create ckpt dir");

        let config = SparseStructureFlowConfig {
            resolution: 2,
            in_channels: 2,
            out_channels: 2,
            model_channels: 8,
            cond_channels: 4,
            num_blocks: 1,
            num_heads: Some(2),
            num_head_channels: 4,
            mlp_ratio: 2.0,
            pe_mode: "rope".to_string(),
            rope_freq: [1.0, 10_000.0],
            share_mod: true,
            qk_rms_norm: true,
            qk_rms_norm_cross: true,
            frequency_embedding_size: 8,
        };

        let config_json = serde_json::json!({
            "name": "SparseStructureFlowModel",
            "args": {
                "resolution": config.resolution,
                "in_channels": config.in_channels,
                "out_channels": config.out_channels,
                "model_channels": config.model_channels,
                "cond_channels": config.cond_channels,
                "num_blocks": config.num_blocks,
                "num_heads": config.num_heads,
                "num_head_channels": config.num_head_channels,
                "mlp_ratio": config.mlp_ratio,
                "pe_mode": config.pe_mode,
                "rope_freq": config.rope_freq,
                "share_mod": config.share_mod,
                "qk_rms_norm": config.qk_rms_norm,
                "qk_rms_norm_cross": config.qk_rms_norm_cross,
                "frequency_embedding_size": config.frequency_embedding_size
            }
        });

        std::fs::write(
            root.join("pipeline.json"),
            r#"{
            "args": {
                "models": {
                    "flow": "ckpts/flow_model"
                }
            }
        }"#,
        )
        .expect("write pipeline");

        std::fs::write(
            ckpts.join("flow_model.json"),
            serde_json::to_vec_pretty(&config_json).expect("serialize config"),
        )
        .expect("write config");

        let device = <TestBackend as burn::tensor::backend::BackendTypes>::Device::default();
        let model = SparseStructureFlowModel::<TestBackend>::new(&device, config);
        let source_path = ckpts.join("flow_model.safetensors");
        let mut safetensor_store = SafetensorsStore::from_file(&source_path);
        model
            .save_into(&mut safetensor_store)
            .expect("save source safetensors");

        let report = import_trellis2_assets(&TrellisImportOptions {
            weights_root: root.clone(),
            image_large_root: None,
            output_root: output.clone(),
            image_large_output_root: None,
            quantization: QuantizationMode::Both,
            overwrite: true,
        })
        .expect("import should succeed");
        assert!(report.manifest.missing_sources.is_empty());

        let safe_size = std::fs::metadata(&source_path)
            .expect("source metadata")
            .len();
        let f32_bpk = output.join("ckpts/flow_model.bpk");
        let f16_bpk = output.join("ckpts/flow_model_f16.bpk");
        let f32_size = std::fs::metadata(&f32_bpk).expect("f32 metadata").len();
        let f16_size = std::fs::metadata(&f16_bpk).expect("f16 metadata").len();

        assert!(
            f32_size < safe_size * 4,
            "f32 burnpack unexpectedly inflated"
        );
        assert!(
            f16_size <= f32_size,
            "f16 burnpack should not exceed f32 size"
        );

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn imports_image_large_stems_into_dedicated_output_root() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock drift")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("burn_trellis_import_image_large_{unique}"));
        let weights_root = root.join("weights");
        let image_large_root = root.join("image_large");
        let output_root = root.join("out_main");
        let image_large_output_root = root.join("out_image_large");

        let weights_ckpts = weights_root.join("ckpts");
        let image_large_ckpts = image_large_root.join("ckpts");
        std::fs::create_dir_all(&weights_ckpts).expect("create weights ckpts");
        std::fs::create_dir_all(&image_large_ckpts).expect("create image-large ckpts");

        let pipeline = r#"{
            "args": {
                "models": {
                    "shape": "ckpts/shape",
                    "ss_dec": "microsoft/TRELLIS-image-large/ckpts/ss_dec"
                }
            }
        }"#;
        std::fs::write(weights_root.join("pipeline.json"), pipeline).expect("write pipeline");

        std::fs::write(weights_ckpts.join("shape.json"), "{}").expect("write shape json");
        std::fs::write(weights_ckpts.join("shape.safetensors"), b"shape_bytes")
            .expect("write shape safetensors");

        std::fs::write(image_large_ckpts.join("ss_dec.json"), "{}").expect("write ss_dec json");
        std::fs::write(
            image_large_ckpts.join("ss_dec.safetensors"),
            b"image_large_bytes",
        )
        .expect("write ss_dec safetensors");

        let report = import_trellis2_assets(&TrellisImportOptions {
            weights_root: weights_root.clone(),
            image_large_root: Some(image_large_root.clone()),
            output_root: output_root.clone(),
            image_large_output_root: Some(image_large_output_root.clone()),
            quantization: QuantizationMode::F32,
            overwrite: true,
        })
        .expect("import should succeed");

        assert!(
            report.manifest.missing_sources.is_empty(),
            "missing sources: {:?}",
            report.manifest.missing_sources
        );
        assert_eq!(
            report.manifest.image_large_output_root.as_deref(),
            Some(image_large_output_root.to_string_lossy().as_ref())
        );

        let shape_bpk = output_root.join("ckpts/shape.bpk");
        let ss_dec_bpk = image_large_output_root.join("ckpts/ss_dec.bpk");
        assert!(
            shape_bpk.exists(),
            "shape should remain in primary output root"
        );
        assert!(
            ss_dec_bpk.exists(),
            "image-large checkpoint should be written to dedicated output root"
        );
        assert!(
            !output_root.join("ckpts/ss_dec.bpk").exists(),
            "image-large checkpoint should not be written into primary output root"
        );

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn imports_image_large_stems_with_weights_root_source_fallback() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock drift")
            .as_nanos();
        let root =
            std::env::temp_dir().join(format!("burn_trellis_import_image_large_fallback_{unique}"));
        let weights_root = root.join("weights");
        let image_large_root = root.join("image_large_empty");
        let output_root = root.join("out_main");
        let image_large_output_root = root.join("out_image_large");

        let weights_ckpts = weights_root.join("ckpts");
        std::fs::create_dir_all(&weights_ckpts).expect("create weights ckpts");
        std::fs::create_dir_all(&image_large_root).expect("create empty image-large root");

        let pipeline = r#"{
            "args": {
                "models": {
                    "shape": "ckpts/shape",
                    "ss_dec": "microsoft/TRELLIS-image-large/ckpts/ss_dec"
                }
            }
        }"#;
        std::fs::write(weights_root.join("pipeline.json"), pipeline).expect("write pipeline");

        std::fs::write(weights_ckpts.join("shape.json"), "{}").expect("write shape json");
        std::fs::write(weights_ckpts.join("shape.safetensors"), b"shape_bytes")
            .expect("write shape safetensors");
        std::fs::write(weights_ckpts.join("ss_dec.json"), "{}").expect("write ss_dec json");
        std::fs::write(weights_ckpts.join("ss_dec.safetensors"), b"ss_dec_bytes")
            .expect("write ss_dec safetensors");

        let report = import_trellis2_assets(&TrellisImportOptions {
            weights_root: weights_root.clone(),
            image_large_root: Some(image_large_root.clone()),
            output_root: output_root.clone(),
            image_large_output_root: Some(image_large_output_root.clone()),
            quantization: QuantizationMode::F32,
            overwrite: true,
        })
        .expect("import should succeed");

        assert!(
            report.manifest.missing_sources.is_empty(),
            "missing sources: {:?}",
            report.manifest.missing_sources
        );
        assert!(output_root.join("ckpts/shape.bpk").exists());
        assert!(image_large_output_root.join("ckpts/ss_dec.bpk").exists());

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn f16_blob_conversion_rewrites_float_tensors_to_f16() {
        let float_values = [1.25f32, -2.5f32];
        let float_bytes = float_values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let int_values = [5i32, -3i32];
        let int_bytes = int_values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();

        let view_f32 =
            TensorView::new(Dtype::F32, vec![2], float_bytes.as_slice()).expect("f32 view");
        let view_i32 =
            TensorView::new(Dtype::I32, vec![2], int_bytes.as_slice()).expect("i32 view");
        let source = serialize(
            vec![("f32".to_string(), view_f32), ("i32".to_string(), view_i32)],
            None,
        )
        .expect("serialize source safetensors");

        let converted = prepare_blob_payload(source.as_slice(), "f16");
        let parsed = SafeTensors::deserialize(converted.as_slice()).expect("deserialize converted");

        let f32_tensor = parsed.tensor("f32").expect("f32 tensor");
        assert_eq!(f32_tensor.dtype(), Dtype::F16);
        let f32_data = f32_tensor
            .data()
            .chunks_exact(2)
            .map(|chunk| f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
            .collect::<Vec<_>>();
        assert!((f32_data[0] - float_values[0]).abs() <= 1.0e-3);
        assert!((f32_data[1] - float_values[1]).abs() <= 1.0e-3);

        let i32_tensor = parsed.tensor("i32").expect("i32 tensor");
        assert_eq!(i32_tensor.dtype(), Dtype::I32);
        assert_eq!(i32_tensor.data(), int_bytes.as_slice());
    }

    #[test]
    fn f16_blob_conversion_preserves_mixed_native_low_precision_payloads() {
        let float_values = [1.25f32, -2.5f32];
        let float_bytes = float_values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let bf16_values = [0.5f32, -7.0f32];
        let bf16_bytes = bf16_values
            .iter()
            .flat_map(|value| bf16::from_f32(*value).to_bits().to_le_bytes())
            .collect::<Vec<_>>();

        let view_f32 =
            TensorView::new(Dtype::F32, vec![2], float_bytes.as_slice()).expect("f32 view");
        let view_bf16 =
            TensorView::new(Dtype::BF16, vec![2], bf16_bytes.as_slice()).expect("bf16 view");
        let source = serialize(
            vec![
                ("f32".to_string(), view_f32),
                ("bf16".to_string(), view_bf16),
            ],
            None,
        )
        .expect("serialize mixed source safetensors");

        let converted = prepare_blob_payload(source.as_slice(), "f16");
        let parsed =
            SafeTensors::deserialize(converted.as_slice()).expect("deserialize preserved payload");

        let f32_tensor = parsed.tensor("f32").expect("f32 tensor");
        assert_eq!(f32_tensor.dtype(), Dtype::F32);
        assert_eq!(f32_tensor.data(), float_bytes.as_slice());

        let bf16_tensor = parsed.tensor("bf16").expect("bf16 tensor");
        assert_eq!(bf16_tensor.dtype(), Dtype::BF16);
        assert_eq!(bf16_tensor.data(), bf16_bytes.as_slice());
    }
}

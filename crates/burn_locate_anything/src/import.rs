use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::assets::{LocateAnythingAssetReport, inspect_model_assets};
use crate::blob_burnpack::{
    burnpack_parts_manifest_path, default_blob_chunk_bytes, write_blob_burnpack_from_file,
};
use crate::cdn::locate_anything_cdn_root_prefix;
use crate::config::LocateAnythingModelConfig;
use crate::{LocateAnythingError, LocateAnythingResult};

#[cfg(feature = "import")]
use burn_synth_import::parts::{resolve_part_entry_path, write_burnpack_parts_for_wasm};

const IMPORT_MANIFEST_FILE: &str = "locate_anything_import_manifest.json";
const METADATA_FILES: &[&str] = &[
    "added_tokens.json",
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.json",
];

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct LocateAnythingImportConfig {
    pub hf_root: PathBuf,
    pub output_dir: PathBuf,
    pub model_id: String,
    pub precision: LocateAnythingPrecision,
    pub shard_size_mib: Option<usize>,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum LocateAnythingPrecision {
    F32,
    F16,
    #[default]
    Bf16,
}

impl std::fmt::Display for LocateAnythingPrecision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::F32 => write!(f, "f32"),
            Self::F16 => write!(f, "f16"),
            Self::Bf16 => write!(f, "bf16"),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct LocateAnythingImportManifest {
    pub version: u32,
    pub model_id: String,
    pub precision: LocateAnythingPrecision,
    pub source_root: String,
    pub model_config: Option<LocateAnythingModelConfig>,
    pub asset_report: LocateAnythingAssetReport,
    pub files: Vec<LocateAnythingSourceFile>,
    pub required_burnpacks: Vec<LocateAnythingBurnpackArtifact>,
    pub cdn_layout: LocateAnythingCdnLayout,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct LocateAnythingSourceFile {
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct LocateAnythingBurnpackArtifact {
    pub component: String,
    pub path: String,
    pub parts_manifest: Option<String>,
    pub bytes: u64,
    pub sha256: String,
    pub source_path: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct LocateAnythingCdnLayout {
    pub root_prefix: String,
    pub files: Vec<LocateAnythingCdnFile>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct LocateAnythingCdnFile {
    pub kind: LocateAnythingCdnFileKind,
    pub local_path: String,
    pub cdn_path: String,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LocateAnythingCdnFileKind {
    Burnpack,
    PartsManifest,
    BurnpackPart,
    Metadata,
}

pub fn write_import_manifest(
    config: &LocateAnythingImportConfig,
) -> LocateAnythingResult<LocateAnythingImportManifest> {
    if !config.hf_root.exists() {
        return Err(LocateAnythingError::Import(format!(
            "HF root does not exist: {}",
            config.hf_root.display()
        )));
    }
    fs::create_dir_all(&config.output_dir)?;
    let files = collect_source_files(&config.hf_root)?;
    let asset_report = inspect_model_assets(&config.hf_root)?;
    let model_config = if asset_report.config_present {
        Some(LocateAnythingModelConfig::from_model_root(&config.hf_root)?)
    } else {
        None
    };
    copy_metadata_files(&config.hf_root, &config.output_dir)?;
    let required_burnpacks = write_required_burnpacks(config, &asset_report)?;
    let manifest = LocateAnythingImportManifest {
        version: 1,
        model_id: config.model_id.clone(),
        precision: config.precision,
        source_root: "hf_root".to_string(),
        model_config,
        asset_report,
        files,
        required_burnpacks: required_burnpacks.clone(),
        cdn_layout: collect_cdn_layout(&config.output_dir, &required_burnpacks)?,
    };
    let manifest_path = config.output_dir.join(IMPORT_MANIFEST_FILE);
    fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)?;
    Ok(manifest)
}

fn collect_source_files(root: &Path) -> LocateAnythingResult<Vec<LocateAnythingSourceFile>> {
    let mut out = Vec::new();
    collect_source_files_inner(root, root, &mut out)?;
    out.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(out)
}

fn collect_source_files_inner(
    root: &Path,
    path: &Path,
    out: &mut Vec<LocateAnythingSourceFile>,
) -> LocateAnythingResult<()> {
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let path = entry.path();
        let metadata = entry.metadata()?;
        if metadata.is_dir() {
            collect_source_files_inner(root, &path, out)?;
        } else if is_relevant_source_file(&path) {
            out.push(LocateAnythingSourceFile {
                path: path
                    .strip_prefix(root)
                    .unwrap_or(&path)
                    .display()
                    .to_string(),
                bytes: metadata.len(),
                sha256: sha256_file(&path)?,
            });
        }
    }
    Ok(())
}

fn is_relevant_source_file(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|value| value.to_str()),
        Some("json" | "safetensors" | "model" | "txt")
    )
}

fn copy_metadata_files(source_root: &Path, output_dir: &Path) -> LocateAnythingResult<()> {
    for relative in METADATA_FILES {
        let source = source_root.join(relative);
        if !source.exists() {
            continue;
        }
        let destination = output_dir.join(relative);
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent).map_err(|err| {
                LocateAnythingError::Io(format!("create {}: {err}", parent.display()))
            })?;
        }
        fs::copy(&source, &destination).map_err(|err| {
            LocateAnythingError::Io(format!(
                "copy LocateAnything metadata {} to {}: {err}",
                source.display(),
                destination.display()
            ))
        })?;
    }
    Ok(())
}

fn write_required_burnpacks(
    config: &LocateAnythingImportConfig,
    asset_report: &LocateAnythingAssetReport,
) -> LocateAnythingResult<Vec<LocateAnythingBurnpackArtifact>> {
    let mut artifacts = Vec::new();
    for source in &asset_report.weight_files {
        let source_path = config.hf_root.join(&source.path);
        if !source_path.exists() {
            return Err(LocateAnythingError::Import(format!(
                "LocateAnything weight shard missing: {}",
                source_path.display()
            )));
        }
        let file_name = burnpack_name_for_source_file(&source.path, config.precision)?;
        let burnpack_path = config.output_dir.join(&file_name);
        let chunk_bytes = default_blob_chunk_bytes(config.shard_size_mib);
        write_blob_burnpack_from_file(&source_path, &burnpack_path, chunk_bytes, false)?;
        let parts_report = write_parts_if_requested(&burnpack_path, config.shard_size_mib)?;
        artifacts.push(LocateAnythingBurnpackArtifact {
            component: source.path.clone(),
            path: file_name,
            parts_manifest: parts_report
                .as_ref()
                .map(|path| path.file_name().unwrap().to_string_lossy().to_string()),
            bytes: fs::metadata(&burnpack_path)
                .map_err(|err| {
                    LocateAnythingError::Io(format!("metadata {}: {err}", burnpack_path.display()))
                })?
                .len(),
            sha256: sha256_file(&burnpack_path)?,
            source_path: source.path.clone(),
        });
    }
    Ok(artifacts)
}

#[cfg(feature = "import")]
fn write_parts_if_requested(
    burnpack_path: &Path,
    shard_size_mib: Option<usize>,
) -> LocateAnythingResult<Option<PathBuf>> {
    let Some(shard_size_mib) = shard_size_mib else {
        return Ok(None);
    };
    write_burnpack_parts_for_wasm(burnpack_path, shard_size_mib as u64, false)
        .map(|report| report.map(|report| report.manifest_path))
        .map_err(LocateAnythingError::Import)
}

#[cfg(not(feature = "import"))]
fn write_parts_if_requested(
    burnpack_path: &Path,
    shard_size_mib: Option<usize>,
) -> LocateAnythingResult<Option<PathBuf>> {
    let _ = burnpack_path;
    if shard_size_mib.is_some() {
        return Err(LocateAnythingError::Unsupported(
            "LocateAnything burnpack sharding requires the import feature".to_string(),
        ));
    }
    Ok(None)
}

fn burnpack_name_for_source_file(
    source_file: &str,
    precision: LocateAnythingPrecision,
) -> LocateAnythingResult<String> {
    let Some(stem) = source_file.strip_suffix(".safetensors") else {
        return Err(LocateAnythingError::Import(format!(
            "LocateAnything source weight shard must be a .safetensors file: {source_file}"
        )));
    };
    Ok(format!("{stem}_{precision}.bpk"))
}

fn collect_cdn_layout(
    output_dir: &Path,
    burnpacks: &[LocateAnythingBurnpackArtifact],
) -> LocateAnythingResult<LocateAnythingCdnLayout> {
    let root_prefix = locate_anything_cdn_root_prefix();
    let mut files = Vec::new();
    for relative in METADATA_FILES {
        let path = output_dir.join(relative);
        if path.exists() {
            files.push(cdn_file(
                output_dir,
                &root_prefix,
                &path,
                LocateAnythingCdnFileKind::Metadata,
            )?);
        }
    }
    let manifest_path = output_dir.join(IMPORT_MANIFEST_FILE);
    if manifest_path.exists() {
        files.push(cdn_file(
            output_dir,
            &root_prefix,
            &manifest_path,
            LocateAnythingCdnFileKind::Metadata,
        )?);
    }
    for burnpack in burnpacks {
        let burnpack_path = output_dir.join(&burnpack.path);
        files.push(cdn_file(
            output_dir,
            &root_prefix,
            &burnpack_path,
            LocateAnythingCdnFileKind::Burnpack,
        )?);
        let parts_manifest = burnpack_parts_manifest_path(&burnpack_path);
        if parts_manifest.exists() {
            files.push(cdn_file(
                output_dir,
                &root_prefix,
                &parts_manifest,
                LocateAnythingCdnFileKind::PartsManifest,
            )?);
            #[cfg(feature = "import")]
            {
                let manifest = burn_synth_import::parts::read_parts_manifest(&parts_manifest)
                    .map_err(LocateAnythingError::Import)?;
                for part in manifest.parts {
                    let part_path = resolve_part_entry_path(&parts_manifest, &part.path)
                        .map_err(LocateAnythingError::Import)?;
                    files.push(cdn_file(
                        output_dir,
                        &root_prefix,
                        &part_path,
                        LocateAnythingCdnFileKind::BurnpackPart,
                    )?);
                }
            }
        }
    }
    files.sort_by(|left, right| left.cdn_path.cmp(&right.cdn_path));
    Ok(LocateAnythingCdnLayout { root_prefix, files })
}

fn cdn_file(
    output_dir: &Path,
    root_prefix: &str,
    path: &Path,
    kind: LocateAnythingCdnFileKind,
) -> LocateAnythingResult<LocateAnythingCdnFile> {
    let relative = path.strip_prefix(output_dir).unwrap_or(path);
    let relative_string = relative.display().to_string();
    Ok(LocateAnythingCdnFile {
        kind,
        local_path: relative_string.clone(),
        cdn_path: format!(
            "{}/{}",
            root_prefix.trim_end_matches('/'),
            relative_string.trim_start_matches('/')
        ),
        bytes: fs::metadata(path)
            .map_err(|err| LocateAnythingError::Io(format!("metadata {}: {err}", path.display())))?
            .len(),
        sha256: sha256_file(path)?,
    })
}

fn sha256_file(path: &Path) -> LocateAnythingResult<String> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precision_defaults_to_bf16_source_checkpoint_variant() {
        assert_eq!(
            LocateAnythingPrecision::default(),
            LocateAnythingPrecision::Bf16
        );
    }

    #[test]
    fn burnpack_names_preserve_source_shard_identity() {
        assert_eq!(
            burnpack_name_for_source_file(
                "model-00001-of-00002.safetensors",
                LocateAnythingPrecision::Bf16
            )
            .unwrap(),
            "model-00001-of-00002_bf16.bpk"
        );
    }
}

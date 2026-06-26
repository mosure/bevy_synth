use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::assets::{LocateAnythingAssetReport, inspect_model_assets};
use crate::config::LocateAnythingModelConfig;
use crate::{LocateAnythingError, LocateAnythingResult};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct LocateAnythingImportConfig {
    pub hf_root: PathBuf,
    pub output_dir: PathBuf,
    pub model_id: String,
    pub precision: LocateAnythingPrecision,
    pub shard_size_mib: Option<usize>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum LocateAnythingPrecision {
    F32,
    F16,
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
    let required_burnpacks = ["vision_encoder", "language_model", "projector", "tokenizer"]
        .into_iter()
        .map(|component| {
            let file_name = format!("{}_{}.bpk", component, config.precision);
            let burnpack_path = config.output_dir.join(&file_name);
            let parts_manifest = burnpack_parts_manifest_path(&burnpack_path);
            LocateAnythingBurnpackArtifact {
                component: component.to_string(),
                path: file_name,
                parts_manifest: parts_manifest.exists().then(|| {
                    parts_manifest
                        .file_name()
                        .unwrap()
                        .to_string_lossy()
                        .to_string()
                }),
            }
        })
        .collect::<Vec<_>>();
    let manifest = LocateAnythingImportManifest {
        version: 1,
        model_id: config.model_id.clone(),
        precision: config.precision,
        source_root: config.hf_root.display().to_string(),
        model_config,
        asset_report,
        files,
        required_burnpacks,
    };
    let manifest_path = config
        .output_dir
        .join("locate_anything_import_manifest.json");
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

fn burnpack_parts_manifest_path(burnpack_path: &Path) -> PathBuf {
    let file_name = burnpack_path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("model.bpk");
    burnpack_path.with_file_name(format!("{file_name}.parts.json"))
}

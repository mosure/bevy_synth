use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::blob_burnpack::burnpack_parts_manifest_path;
use crate::{LocateAnythingError, LocateAnythingResult};

static WEIGHT_MAP_CACHE: LazyLock<Mutex<BTreeMap<PathBuf, BTreeMap<String, String>>>> =
    LazyLock::new(|| Mutex::new(BTreeMap::new()));

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct LocateAnythingAssetReport {
    pub model_root: PathBuf,
    pub config_present: bool,
    pub tokenizer_present: bool,
    pub processor_present: bool,
    pub index_present: bool,
    pub weight_files: Vec<LocateAnythingWeightFileStatus>,
    pub missing_files: Vec<String>,
    pub total_weight_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct LocateAnythingWeightFileStatus {
    pub path: String,
    pub present: bool,
    pub bytes: u64,
}

impl LocateAnythingAssetReport {
    pub fn is_complete(&self) -> bool {
        self.config_present
            && self.tokenizer_present
            && self.processor_present
            && self.index_present
            && self.missing_files.is_empty()
            && self.weight_files.iter().all(|file| file.present)
    }
}

pub fn inspect_model_assets(
    model_root: impl AsRef<Path>,
) -> LocateAnythingResult<LocateAnythingAssetReport> {
    let model_root = model_root.as_ref();
    let required = [
        "config.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "preprocessor_config.json",
        "processor_config.json",
        "model.safetensors.index.json",
    ];
    let mut missing_files = Vec::new();
    for relative in required {
        if !model_root.join(relative).exists() {
            missing_files.push(relative.to_string());
        }
    }

    let index_path = model_root.join("model.safetensors.index.json");
    let weight_file_names = if index_path.exists() {
        collect_weight_file_names(&index_path)?
    } else {
        BTreeSet::new()
    };
    let mut weight_files = Vec::new();
    let mut total_weight_bytes = 0u64;
    for name in weight_file_names {
        let path = resolve_existing_weight_artifact(model_root, &name);
        let bytes = weight_artifact_bytes(&path)?;
        if bytes == 0 {
            missing_files.push(name.clone());
        }
        total_weight_bytes = total_weight_bytes.saturating_add(bytes);
        weight_files.push(LocateAnythingWeightFileStatus {
            path: name,
            present: bytes > 0,
            bytes,
        });
    }

    Ok(LocateAnythingAssetReport {
        model_root: model_root.to_path_buf(),
        config_present: model_root.join("config.json").exists(),
        tokenizer_present: model_root.join("tokenizer_config.json").exists()
            && model_root.join("vocab.json").exists()
            && model_root.join("merges.txt").exists(),
        processor_present: model_root.join("preprocessor_config.json").exists()
            && model_root.join("processor_config.json").exists(),
        index_present: index_path.exists(),
        weight_files,
        missing_files,
        total_weight_bytes,
    })
}

pub fn weight_file_for_tensor(
    model_root: impl AsRef<Path>,
    tensor_name: &str,
) -> LocateAnythingResult<PathBuf> {
    let model_root = model_root.as_ref();
    let index_path = model_root.join("model.safetensors.index.json");
    let weight_map = load_weight_map(&index_path)?;
    let file = weight_map
        .get(tensor_name)
        .map(String::as_str)
        .ok_or_else(|| {
            LocateAnythingError::Config(format!(
                "{} does not map tensor `{tensor_name}`",
                index_path.display()
            ))
        })?;
    Ok(resolve_existing_weight_artifact(model_root, file))
}

fn resolve_existing_weight_artifact(model_root: &Path, file: &str) -> PathBuf {
    let direct = model_root.join(file);
    if direct.exists() {
        return direct;
    }
    for candidate in burnpack_candidates_for_weight_file(model_root, file) {
        if candidate.exists() || burnpack_parts_manifest_path(&candidate).exists() {
            return candidate;
        }
    }
    direct
}

fn weight_artifact_bytes(path: &Path) -> LocateAnythingResult<u64> {
    if let Ok(metadata) = fs::metadata(path) {
        return Ok(metadata.len());
    }
    let manifest_path = burnpack_parts_manifest_path(path);
    if !manifest_path.exists() {
        return Ok(0);
    }
    let bytes = fs::read(&manifest_path).map_err(|err| {
        LocateAnythingError::Io(format!("read {}: {err}", manifest_path.display()))
    })?;
    let value = serde_json::from_slice::<Value>(&bytes).map_err(|err| {
        LocateAnythingError::Config(format!("parse {}: {err}", manifest_path.display()))
    })?;
    Ok(value
        .get("total_bytes")
        .and_then(Value::as_u64)
        .or_else(|| {
            value.get("parts").and_then(Value::as_array).map(|parts| {
                parts
                    .iter()
                    .filter_map(|part| part.get("bytes").and_then(Value::as_u64))
                    .sum()
            })
        })
        .unwrap_or(0))
}

fn burnpack_candidates_for_weight_file(model_root: &Path, file: &str) -> Vec<PathBuf> {
    let Some(stem) = file.strip_suffix(".safetensors") else {
        return vec![model_root.join(file)];
    };
    ["bf16", "f16", "f32"]
        .into_iter()
        .map(|precision| model_root.join(format!("{stem}_{precision}.bpk")))
        .chain(std::iter::once(model_root.join(format!("{stem}.bpk"))))
        .collect()
}

fn collect_weight_file_names(index_path: &Path) -> LocateAnythingResult<BTreeSet<String>> {
    Ok(load_weight_map(index_path)?.values().cloned().collect())
}

fn load_weight_map(index_path: &Path) -> LocateAnythingResult<BTreeMap<String, String>> {
    if let Some(cached) = WEIGHT_MAP_CACHE
        .lock()
        .map_err(|err| {
            LocateAnythingError::Runtime(format!(
                "failed to lock LocateAnything weight-map cache: {err}"
            ))
        })?
        .get(index_path)
        .cloned()
    {
        return Ok(cached);
    }

    let bytes = fs::read(index_path).map_err(|err| {
        LocateAnythingError::Config(format!("failed to read {}: {err}", index_path.display()))
    })?;
    let value = serde_json::from_slice::<Value>(&bytes).map_err(|err| {
        LocateAnythingError::Config(format!("failed to parse {}: {err}", index_path.display()))
    })?;
    let Some(weight_map) = value.get("weight_map").and_then(|value| value.as_object()) else {
        return Err(LocateAnythingError::Config(format!(
            "{} is missing `weight_map`",
            index_path.display()
        )));
    };
    let parsed = weight_map
        .iter()
        .filter_map(|(key, value)| {
            value
                .as_str()
                .map(|file| (key.to_string(), file.to_string()))
        })
        .collect::<BTreeMap<_, _>>();
    WEIGHT_MAP_CACHE
        .lock()
        .map_err(|err| {
            LocateAnythingError::Runtime(format!(
                "failed to lock LocateAnything weight-map cache: {err}"
            ))
        })?
        .insert(index_path.to_path_buf(), parsed.clone());
    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asset_report_detects_public_config_download() {
        let root = Path::new("assets/models/LocateAnything-3B");
        if !root.exists() {
            eprintln!("skipping asset report test; {} is missing", root.display());
            return;
        }
        let report = inspect_model_assets(root).unwrap();
        assert!(report.config_present);
        assert!(report.tokenizer_present);
        assert!(report.processor_present);
        assert!(report.index_present);
        assert_eq!(report.weight_files.len(), 2);
    }

    #[test]
    fn index_maps_known_component_tensors_when_present() {
        let root = Path::new("assets/models/LocateAnything-3B");
        if !root.join("model.safetensors.index.json").exists() {
            eprintln!("skipping index map test; {} is missing", root.display());
            return;
        }
        assert_eq!(
            weight_file_for_tensor(root, "vision_model.patch_embed.proj.weight")
                .unwrap()
                .file_name()
                .unwrap()
                .to_string_lossy(),
            "model-00001-of-00002.safetensors"
        );
        assert_eq!(
            weight_file_for_tensor(root, "mlp1.0.weight")
                .unwrap()
                .file_name()
                .unwrap()
                .to_string_lossy(),
            "model-00002-of-00002.safetensors"
        );
    }

    #[test]
    fn index_uses_bpk_bundle_when_safetensors_are_not_materialized() {
        let root =
            std::env::temp_dir().join(format!("locateanything_asset_bpk_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join("model.safetensors.index.json"),
            r#"{"weight_map":{"vision_model.patch_embed.proj.weight":"model-00001-of-00002.safetensors"}}"#,
        )
        .unwrap();
        fs::write(root.join("model-00001-of-00002_bf16.bpk.parts.json"), "{}").unwrap();
        let resolved =
            weight_file_for_tensor(&root, "vision_model.patch_embed.proj.weight").unwrap();
        assert_eq!(
            resolved.file_name().unwrap().to_string_lossy(),
            "model-00001-of-00002_bf16.bpk"
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn asset_report_accepts_bpk_parts_without_materialized_safetensors() {
        let root = std::env::temp_dir().join(format!(
            "locateanything_asset_report_bpk_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        for file in [
            "config.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "preprocessor_config.json",
            "processor_config.json",
        ] {
            fs::write(root.join(file), "{}").unwrap();
        }
        fs::write(
            root.join("model.safetensors.index.json"),
            r#"{"weight_map":{"vision_model.patch_embed.proj.weight":"model-00001-of-00002.safetensors"}}"#,
        )
        .unwrap();
        fs::write(
            root.join("model-00001-of-00002_bf16.bpk.parts.json"),
            r#"{"total_bytes":1234,"parts":[{"path":"model-00001-of-00002_bf16.bpk.part-00000.bpk","bytes":1234}]}"#,
        )
        .unwrap();
        let report = inspect_model_assets(&root).unwrap();
        assert!(report.is_complete(), "{report:#?}");
        assert_eq!(report.total_weight_bytes, 1234);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn asset_report_accepts_env_bpk_bundle_root() {
        let Some(root) = std::env::var_os("LOCATE_ANYTHING_BPK_BUNDLE_ROOT").map(PathBuf::from)
        else {
            eprintln!("skipping: set LOCATE_ANYTHING_BPK_BUNDLE_ROOT to a bpk bundle root");
            return;
        };
        let report = inspect_model_assets(&root).unwrap();
        assert!(report.is_complete(), "{report:#?}");
        assert_eq!(report.weight_files.len(), 2);
    }
}

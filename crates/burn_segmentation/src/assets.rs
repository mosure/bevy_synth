use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::cdn::{component_safetensors_file_name, component_safetensors_rel_path};
use crate::config::{
    SegmentationModelComponent, SegmentationPrecision, SegmentationQuantization,
    component_burnpack_file_name, optional_components, required_components,
};
use crate::{SegmentationError, SegmentationModelKind, SegmentationResult};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct SegmentationAssetReport {
    pub model_root: PathBuf,
    pub model: SegmentationModelKind,
    pub config_present: bool,
    pub processor_present: bool,
    pub index_present: bool,
    pub single_safetensors_present: bool,
    #[serde(default)]
    pub component_safetensors_present: bool,
    pub weight_files: Vec<SegmentationWeightFileStatus>,
    pub burnpacks: Vec<SegmentationBurnpackStatus>,
    pub missing_files: Vec<String>,
    pub total_weight_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct SegmentationWeightFileStatus {
    pub path: String,
    pub present: bool,
    pub bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct SegmentationBurnpackStatus {
    pub component: SegmentationModelComponent,
    pub path: String,
    pub present: bool,
    pub bytes: u64,
    pub required: bool,
    pub parts_manifest_present: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parts_manifest: Option<String>,
}

impl SegmentationAssetReport {
    pub fn is_complete_for_native_burn(&self) -> bool {
        if self.model == SegmentationModelKind::BboxPrompt {
            return true;
        }
        let required_burnpacks_present = self.burnpacks.iter().all(|artifact| {
            !artifact.required || artifact.present || artifact.parts_manifest_present
        });
        self.component_safetensors_present || required_burnpacks_present
    }

    pub fn has_source_weights(&self) -> bool {
        self.single_safetensors_present
            || self.index_present
            || self.component_safetensors_present
            || self.weight_files.iter().any(|file| file.present)
    }
}

pub fn inspect_model_assets(
    model: SegmentationModelKind,
    model_root: impl AsRef<Path>,
    precision: SegmentationPrecision,
    quantization: SegmentationQuantization,
) -> SegmentationResult<SegmentationAssetReport> {
    let model_root = model_root.as_ref();
    if model == SegmentationModelKind::BboxPrompt {
        return Ok(SegmentationAssetReport {
            model_root: model_root.to_path_buf(),
            model,
            config_present: true,
            processor_present: true,
            index_present: false,
            single_safetensors_present: false,
            component_safetensors_present: true,
            weight_files: Vec::new(),
            burnpacks: Vec::new(),
            missing_files: Vec::new(),
            total_weight_bytes: 0,
        });
    }

    let config_present = model_root.join("config.json").exists();
    let processor_present = [
        "preprocessor_config.json",
        "processor_config.json",
        "image_processor_config.json",
    ]
    .iter()
    .any(|name| model_root.join(name).exists());
    let index_path = model_root.join("model.safetensors.index.json");
    let index_present = index_path.exists();
    let single_safetensors_present = model_root.join("model.safetensors").exists();
    let component_file_names = collect_component_file_names(model_root, model);
    let component_safetensors_present = !required_components(model).is_empty()
        && required_components(model).iter().all(|component| {
            component_file_path(model_root, *component)
                .map(|path| path.exists())
                .unwrap_or(false)
        });
    let mut weight_file_names = if index_present {
        collect_weight_file_names(&index_path)?
    } else if single_safetensors_present {
        BTreeSet::from(["model.safetensors".to_string()])
    } else {
        BTreeSet::new()
    };
    weight_file_names.extend(component_file_names);
    let burnpacks = required_components(model)
        .iter()
        .copied()
        .map(|component| burnpack_status(model_root, component, precision, quantization, true))
        .chain(optional_components(model).iter().copied().map(|component| {
            burnpack_status(model_root, component, precision, quantization, false)
        }))
        .collect::<Vec<_>>();
    let mut missing_files = Vec::new();
    let required_burnpacks_present = burnpacks
        .iter()
        .all(|artifact| !artifact.required || artifact.present || artifact.parts_manifest_present);
    if !config_present && !component_safetensors_present && !required_burnpacks_present {
        missing_files.push("config.json".to_string());
    }
    if !processor_present && !component_safetensors_present && !required_burnpacks_present {
        missing_files.push(
            "preprocessor_config.json|processor_config.json|image_processor_config.json"
                .to_string(),
        );
    }
    if !index_present
        && !single_safetensors_present
        && !component_safetensors_present
        && !required_burnpacks_present
    {
        missing_files.push("model.safetensors or model.safetensors.index.json".to_string());
    }

    let mut total_weight_bytes = 0u64;
    let weight_files = weight_file_names
        .into_iter()
        .map(|name| {
            let path = model_root.join(&name);
            let bytes = fs::metadata(&path)
                .map(|metadata| metadata.len())
                .unwrap_or(0);
            if bytes == 0 {
                missing_files.push(name.clone());
            }
            total_weight_bytes = total_weight_bytes.saturating_add(bytes);
            SegmentationWeightFileStatus {
                path: name,
                present: bytes > 0,
                bytes,
            }
        })
        .collect::<Vec<_>>();

    for artifact in &burnpacks {
        if artifact.required
            && !artifact.present
            && !artifact.parts_manifest_present
            && !component_safetensors_present
        {
            missing_files.push(format!(
                "{} or {}",
                artifact.path,
                burnpack_parts_manifest_path(&model_root.join(&artifact.path))
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("model.bpk.parts.json")
            ));
        }
    }

    Ok(SegmentationAssetReport {
        model_root: model_root.to_path_buf(),
        model,
        config_present,
        processor_present,
        index_present,
        single_safetensors_present,
        component_safetensors_present,
        weight_files,
        burnpacks,
        missing_files,
        total_weight_bytes,
    })
}

fn collect_component_file_names(
    model_root: &Path,
    model: SegmentationModelKind,
) -> BTreeSet<String> {
    required_components(model)
        .iter()
        .chain(optional_components(model).iter())
        .filter_map(|component| {
            component_file_path(model_root, *component).and_then(|path| {
                path.exists().then(|| {
                    path.strip_prefix(model_root)
                        .unwrap_or(&path)
                        .display()
                        .to_string()
                })
            })
        })
        .collect()
}

fn component_file_path(
    model_root: &Path,
    component: SegmentationModelComponent,
) -> Option<PathBuf> {
    let component_rel = component_safetensors_rel_path(component);
    let direct = component_safetensors_file_name(component);
    [model_root.join(component_rel), model_root.join(direct)]
        .into_iter()
        .find(|path| path.exists())
}

pub fn burnpack_status(
    model_root: &Path,
    component: SegmentationModelComponent,
    precision: SegmentationPrecision,
    quantization: SegmentationQuantization,
    required: bool,
) -> SegmentationBurnpackStatus {
    let path = component_burnpack_file_name(component, precision, quantization);
    let burnpack_path = model_root.join(&path);
    let parts_manifest_path = burnpack_parts_manifest_path(&burnpack_path);
    let bytes = fs::metadata(&burnpack_path)
        .map(|metadata| metadata.len())
        .unwrap_or(0);
    SegmentationBurnpackStatus {
        component,
        path,
        present: bytes > 0,
        bytes,
        required,
        parts_manifest_present: parts_manifest_path.exists(),
        parts_manifest: parts_manifest_path.exists().then(|| {
            parts_manifest_path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .to_string()
        }),
    }
}

pub fn burnpack_parts_manifest_path(burnpack_path: &Path) -> PathBuf {
    let file_name = burnpack_path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("model.bpk");
    burnpack_path.with_file_name(format!("{file_name}.parts.json"))
}

fn collect_weight_file_names(index_path: &Path) -> SegmentationResult<BTreeSet<String>> {
    let bytes = fs::read(index_path).map_err(|err| {
        SegmentationError::Io(format!("failed to read {}: {err}", index_path.display()))
    })?;
    let value = serde_json::from_slice::<Value>(&bytes).map_err(|err| {
        SegmentationError::Image(format!("failed to parse {}: {err}", index_path.display()))
    })?;
    let Some(weight_map) = value.get("weight_map").and_then(Value::as_object) else {
        return Err(SegmentationError::Image(format!(
            "{} is missing weight_map",
            index_path.display()
        )));
    };
    Ok(weight_map
        .values()
        .filter_map(Value::as_str)
        .map(str::to_string)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bbox_prompt_assets_are_always_complete() {
        let report = inspect_model_assets(
            SegmentationModelKind::BboxPrompt,
            Path::new("/definitely/not/required"),
            SegmentationPrecision::F16,
            SegmentationQuantization::None,
        )
        .unwrap();
        assert!(report.is_complete_for_native_burn());
        assert!(report.burnpacks.is_empty());
    }

    #[test]
    fn sam_asset_report_tracks_required_burnpacks() {
        let root =
            std::env::temp_dir().join(format!("burn_segmentation_assets_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("config.json"), "{}").unwrap();
        fs::write(root.join("preprocessor_config.json"), "{}").unwrap();
        fs::write(root.join("model.safetensors"), b"weights").unwrap();
        fs::write(root.join("image_encoder_f16.bpk.parts.json"), "{}").unwrap();

        let report = inspect_model_assets(
            SegmentationModelKind::Sam2,
            &root,
            SegmentationPrecision::F16,
            SegmentationQuantization::None,
        )
        .unwrap();

        assert!(report.config_present);
        assert!(report.processor_present);
        assert!(report.has_source_weights());
        assert_eq!(
            report
                .burnpacks
                .iter()
                .filter(|artifact| artifact.required)
                .count(),
            3
        );
        assert!(
            report
                .burnpacks
                .iter()
                .any(|artifact| artifact.path == "image_encoder_f16.bpk"
                    && artifact.parts_manifest_present)
        );
        assert!(
            report
                .missing_files
                .iter()
                .any(|file| file.contains("prompt_encoder_f16.bpk"))
        );

        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn component_safetensors_are_complete_native_artifacts() {
        let root = std::env::temp_dir().join(format!(
            "burn_segmentation_component_assets_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("components")).unwrap();
        fs::write(root.join("config.json"), "{}").unwrap();
        fs::write(root.join("preprocessor_config.json"), "{}").unwrap();
        fs::write(root.join("components/image_encoder.safetensors"), b"image").unwrap();
        fs::write(
            root.join("components/prompt_encoder.safetensors"),
            b"prompt",
        )
        .unwrap();
        fs::write(root.join("components/mask_decoder.safetensors"), b"mask").unwrap();

        let report = inspect_model_assets(
            SegmentationModelKind::Sam2,
            &root,
            SegmentationPrecision::F16,
            SegmentationQuantization::None,
        )
        .unwrap();

        assert!(report.component_safetensors_present);
        assert!(report.has_source_weights());
        assert!(report.is_complete_for_native_burn());
        assert!(
            !report
                .missing_files
                .iter()
                .any(|path| path.contains(".bpk"))
        );
        fs::remove_dir_all(root).ok();
    }
}

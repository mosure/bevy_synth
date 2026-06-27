use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use burn_synth_import::parts::{
    read_parts_manifest, resolve_part_entry_path, write_burnpack_parts_for_wasm,
};

use crate::assets::{SegmentationAssetReport, burnpack_parts_manifest_path, inspect_model_assets};
use crate::config::{SegmentationPrecision, SegmentationQuantization};
use crate::{SegmentationError, SegmentationModelKind, SegmentationResult};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct SegmentationImportConfig {
    pub hf_root: PathBuf,
    pub output_dir: PathBuf,
    pub model_id: String,
    pub model: SegmentationModelKind,
    pub precision: SegmentationPrecision,
    pub quantization: SegmentationQuantization,
    pub shard_size_mib: Option<usize>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct SegmentationImportManifest {
    pub version: u32,
    pub model_id: String,
    pub model: SegmentationModelKind,
    pub precision: SegmentationPrecision,
    pub quantization: SegmentationQuantization,
    pub source_root: String,
    pub asset_report: SegmentationAssetReport,
    pub files: Vec<SegmentationSourceFile>,
    pub required_burnpacks: Vec<SegmentationBurnpackArtifact>,
    pub cdn_layout: SegmentationCdnLayout,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct SegmentationSourceFile {
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct SegmentationBurnpackArtifact {
    pub component: String,
    pub path: String,
    pub required: bool,
    pub parts_manifest: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct SegmentationCdnLayout {
    pub root_prefix: String,
    pub import_manifest_path: String,
    pub component_paths: Vec<String>,
    pub files: Vec<SegmentationCdnFile>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct SegmentationCdnFile {
    pub kind: SegmentationCdnFileKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub component: Option<String>,
    pub local_path: String,
    pub cdn_path: String,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum SegmentationCdnFileKind {
    Burnpack,
    PartsManifest,
    BurnpackPart,
}

pub fn write_import_manifest(
    config: &SegmentationImportConfig,
) -> SegmentationResult<SegmentationImportManifest> {
    if !config.hf_root.exists() {
        return Err(SegmentationError::Io(format!(
            "HF root does not exist: {}",
            config.hf_root.display()
        )));
    }
    fs::create_dir_all(&config.output_dir).map_err(|err| {
        SegmentationError::Io(format!("create {}: {err}", config.output_dir.display()))
    })?;
    let files = collect_source_files(&config.hf_root)?;
    let initial_asset_report = inspect_model_assets(
        config.model,
        &config.output_dir,
        config.precision,
        config.quantization,
    )?;
    if let Some(shard_size_mib) = config.shard_size_mib {
        for artifact in &initial_asset_report.burnpacks {
            let burnpack_path = config.output_dir.join(&artifact.path);
            if burnpack_path.exists() {
                write_burnpack_parts_for_wasm(&burnpack_path, shard_size_mib as u64, false)
                    .map_err(SegmentationError::Io)?;
            }
        }
    }
    let asset_report = inspect_model_assets(
        config.model,
        &config.output_dir,
        config.precision,
        config.quantization,
    )?;
    let required_burnpacks = asset_report
        .burnpacks
        .iter()
        .map(|artifact| {
            let burnpack_path = config.output_dir.join(&artifact.path);
            let parts_manifest = burnpack_parts_manifest_path(&burnpack_path);
            SegmentationBurnpackArtifact {
                component: artifact.component.label().to_string(),
                path: artifact.path.clone(),
                required: artifact.required,
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
    let root_prefix = format!(
        "models/segmentation/{}/{}/{}",
        config.model.label(),
        config.precision,
        config.quantization
    );
    let component_paths = required_burnpacks
        .iter()
        .map(|artifact| format!("{root_prefix}/{}", artifact.path))
        .chain(required_burnpacks.iter().filter_map(|artifact| {
            artifact
                .parts_manifest
                .as_ref()
                .map(|parts| format!("{root_prefix}/{parts}"))
        }))
        .collect::<Vec<_>>();
    let cdn_files = collect_cdn_files(&config.output_dir, &required_burnpacks, &root_prefix)?;
    let cdn_layout = SegmentationCdnLayout {
        root_prefix: root_prefix.clone(),
        import_manifest_path: format!("{root_prefix}/segmentation_import_manifest.json"),
        component_paths,
        files: cdn_files,
    };
    let manifest = SegmentationImportManifest {
        version: 1,
        model_id: config.model_id.clone(),
        model: config.model,
        precision: config.precision,
        quantization: config.quantization,
        source_root: source_root_label(&config.hf_root),
        asset_report,
        files,
        required_burnpacks,
        cdn_layout,
    };
    let manifest_path = config.output_dir.join("segmentation_import_manifest.json");
    fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest)
            .map_err(|err| SegmentationError::Image(format!("serialize import manifest: {err}")))?,
    )
    .map_err(|err| SegmentationError::Io(format!("write {}: {err}", manifest_path.display())))?;
    Ok(manifest)
}

fn collect_cdn_files(
    output_dir: &Path,
    artifacts: &[SegmentationBurnpackArtifact],
    root_prefix: &str,
) -> SegmentationResult<Vec<SegmentationCdnFile>> {
    let mut files = Vec::new();
    for artifact in artifacts {
        let component = Some(artifact.component.clone());
        let burnpack_path = output_dir.join(&artifact.path);
        if burnpack_path.exists() {
            files.push(cdn_file_entry(
                output_dir,
                &burnpack_path,
                root_prefix,
                SegmentationCdnFileKind::Burnpack,
                component.clone(),
            )?);
        }

        let parts_manifest_path = burnpack_parts_manifest_path(&burnpack_path);
        if parts_manifest_path.exists() {
            files.push(cdn_file_entry(
                output_dir,
                &parts_manifest_path,
                root_prefix,
                SegmentationCdnFileKind::PartsManifest,
                component.clone(),
            )?);
            let manifest = read_parts_manifest(&parts_manifest_path).map_err(|err| {
                SegmentationError::Image(format!(
                    "parse parts manifest {}: {err}",
                    parts_manifest_path.display()
                ))
            })?;
            for part in &manifest.parts {
                let part_path =
                    resolve_part_entry_path(&parts_manifest_path, &part.path).map_err(|err| {
                        SegmentationError::Image(format!(
                            "resolve parts manifest entry {}: {err}",
                            parts_manifest_path.display()
                        ))
                    })?;
                files.push(cdn_file_entry(
                    output_dir,
                    &part_path,
                    root_prefix,
                    SegmentationCdnFileKind::BurnpackPart,
                    component.clone(),
                )?);
            }
        }
    }
    files.sort_by(|left, right| left.cdn_path.cmp(&right.cdn_path));
    files.dedup_by(|left, right| left.cdn_path == right.cdn_path);
    Ok(files)
}

fn cdn_file_entry(
    output_dir: &Path,
    path: &Path,
    root_prefix: &str,
    kind: SegmentationCdnFileKind,
    component: Option<String>,
) -> SegmentationResult<SegmentationCdnFile> {
    let relative = path.strip_prefix(output_dir).unwrap_or(path);
    let local_path = relative.display().to_string();
    let bytes = fs::metadata(path)
        .map_err(|err| SegmentationError::Io(format!("metadata {}: {err}", path.display())))?
        .len();
    Ok(SegmentationCdnFile {
        kind,
        component,
        cdn_path: format!("{root_prefix}/{local_path}"),
        local_path,
        bytes,
        sha256: sha256_file(path)?,
    })
}

fn source_root_label(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.trim().is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| "hf_snapshot".to_string())
}

fn collect_source_files(root: &Path) -> SegmentationResult<Vec<SegmentationSourceFile>> {
    let mut out = Vec::new();
    collect_source_files_inner(root, root, &mut out)?;
    out.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(out)
}

fn collect_source_files_inner(
    root: &Path,
    path: &Path,
    out: &mut Vec<SegmentationSourceFile>,
) -> SegmentationResult<()> {
    for entry in fs::read_dir(path)
        .map_err(|err| SegmentationError::Io(format!("read_dir {}: {err}", path.display())))?
    {
        let entry = entry.map_err(|err| SegmentationError::Io(err.to_string()))?;
        let path = entry.path();
        let metadata = entry
            .metadata()
            .map_err(|err| SegmentationError::Io(format!("metadata {}: {err}", path.display())))?;
        if metadata.is_dir() {
            collect_source_files_inner(root, &path, out)?;
        } else if is_relevant_source_file(&path) {
            out.push(SegmentationSourceFile {
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
        Some("json" | "safetensors" | "model" | "txt" | "yaml" | "yml")
    )
}

fn sha256_file(path: &Path) -> SegmentationResult<String> {
    let mut file = fs::File::open(path)
        .map_err(|err| SegmentationError::Io(format!("open {}: {err}", path.display())))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|err| SegmentationError::Io(format!("read {}: {err}", path.display())))?;
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
    fn import_manifest_records_source_files_and_cdn_layout() {
        let root = std::env::temp_dir().join(format!(
            "burn_segmentation_import_src_{}",
            std::process::id()
        ));
        let out = std::env::temp_dir().join(format!(
            "burn_segmentation_import_out_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        let _ = fs::remove_dir_all(&out);
        fs::create_dir_all(&root).unwrap();
        fs::create_dir_all(&out).unwrap();
        fs::write(root.join("config.json"), "{}").unwrap();
        fs::write(root.join("model.safetensors"), b"weights").unwrap();
        fs::write(out.join("image_encoder_f16.bpk"), b"burnpack").unwrap();
        fs::write(out.join("image_encoder_f16.bpk.part-00000.bpk"), b"part").unwrap();
        fs::write(
            out.join("image_encoder_f16.bpk.parts.json"),
            r#"{
              "version": 1,
              "source_file": "image_encoder_f16.bpk",
              "source_modified_unix_ms": 1,
              "total_bytes": 8,
              "max_part_bytes": 4,
              "parts": [
                {
                  "path": "image_encoder_f16.bpk.part-00000.bpk",
                  "bytes": 4,
                  "sha256": "",
                  "tensors": 1
                }
              ]
            }"#,
        )
        .unwrap();

        let manifest = write_import_manifest(&SegmentationImportConfig {
            hf_root: root.clone(),
            output_dir: out.clone(),
            model_id: "facebook/sam2-test".to_string(),
            model: SegmentationModelKind::Sam2,
            precision: SegmentationPrecision::F16,
            quantization: SegmentationQuantization::None,
            shard_size_mib: None,
        })
        .unwrap();

        assert_eq!(manifest.model, SegmentationModelKind::Sam2);
        assert!(!Path::new(&manifest.source_root).is_absolute());
        assert!(manifest.files.iter().any(|file| file.path == "config.json"));
        assert!(manifest.cdn_layout.root_prefix.ends_with("sam2/f16/none"));
        assert!(
            manifest
                .cdn_layout
                .files
                .iter()
                .any(|file| file.kind == SegmentationCdnFileKind::PartsManifest
                    && file.cdn_path.ends_with("image_encoder_f16.bpk.parts.json"))
        );
        assert!(
            manifest
                .cdn_layout
                .files
                .iter()
                .any(|file| file.kind == SegmentationCdnFileKind::BurnpackPart
                    && file.cdn_path.contains("image_encoder_f16.bpk.part-"))
        );
        assert!(out.join("segmentation_import_manifest.json").exists());

        fs::remove_dir_all(root).ok();
        fs::remove_dir_all(out).ok();
    }
}

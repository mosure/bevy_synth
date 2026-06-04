use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::blob_burnpack::load_blob_bytes_from_burnpack_bytes;
use crate::virtual_fs;

#[derive(Debug, Deserialize)]
struct BurnpackPartEntry {
    path: String,
    #[serde(default)]
    bytes: u64,
}

#[derive(Debug, Deserialize)]
struct BurnpackPartsManifest {
    #[serde(default)]
    source_file: String,
    #[serde(default)]
    total_bytes: u64,
    #[serde(default)]
    parts: Vec<BurnpackPartEntry>,
}

pub(crate) fn burnpack_parts_manifest_path(burnpack_path: &Path) -> PathBuf {
    let file_name = burnpack_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("model.bpk");
    burnpack_path.with_file_name(format!("{file_name}.parts.json"))
}

pub(crate) fn candidate_exists_or_has_parts(path: &Path) -> bool {
    virtual_fs::exists(path) || virtual_fs::exists(&burnpack_parts_manifest_path(path))
}

pub(crate) fn load_blob_bytes_from_burnpack_or_parts<F>(
    burnpack_path: &Path,
    mut load_blob_from_burnpack: F,
) -> Result<Vec<u8>, String>
where
    F: FnMut(&Path) -> Result<Vec<u8>, String>,
{
    if virtual_fs::exists(burnpack_path) {
        return load_blob_from_burnpack(burnpack_path);
    }

    let manifest_path = burnpack_parts_manifest_path(burnpack_path);
    let manifest_bytes = virtual_fs::read(&manifest_path).map_err(|err| {
        format!(
            "failed to read burnpack parts manifest '{}': {err}",
            manifest_path.display()
        )
    })?;
    let manifest: BurnpackPartsManifest =
        serde_json::from_slice(&manifest_bytes).map_err(|err| {
            format!(
                "failed to parse burnpack parts manifest '{}': {err}",
                manifest_path.display()
            )
        })?;
    if manifest.parts.is_empty() {
        return Err(format!(
            "burnpack parts manifest '{}' contains no parts",
            manifest_path.display()
        ));
    }
    if !manifest.source_file.trim().is_empty()
        && burnpack_path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name != manifest.source_file.trim())
    {
        return Err(format!(
            "burnpack parts manifest '{}' source_file '{}' does not match requested '{}'",
            manifest_path.display(),
            manifest.source_file,
            burnpack_path.display()
        ));
    }

    let mut merged = Vec::new();
    let mut all_part_sizes_match_file_sizes = true;
    let mut all_part_sizes_match_payload_sizes = true;
    let mut file_size_total = 0u64;
    let manifest_source_url = virtual_fs::source_url(&manifest_path);
    for (index, part) in manifest.parts.iter().enumerate() {
        let part_path = resolve_manifest_part_path(&manifest_path, part.path.as_str())?;
        let part_file_size = if virtual_fs::has_virtual_file(&part_path) || part_path.exists() {
            virtual_fs::metadata_len(&part_path).map_err(|err| {
                format!(
                    "failed to stat burnpack part '{}': {err}",
                    part_path.display()
                )
            })?
        } else {
            part.bytes
        };
        file_size_total = file_size_total.saturating_add(part_file_size);
        let bytes = if virtual_fs::has_virtual_file(&part_path) {
            let burnpack_bytes = virtual_fs::read(&part_path).map_err(|err| {
                format!(
                    "failed to read virtual burnpack part {}/{} '{}' for '{}': {err}",
                    index + 1,
                    manifest.parts.len(),
                    part_path.display(),
                    burnpack_path.display()
                )
            })?;
            load_blob_bytes_from_burnpack_bytes(&burnpack_bytes).map_err(|err| {
                format!(
                    "failed to decode virtual burnpack part {}/{} '{}' for '{}': {err}",
                    index + 1,
                    manifest.parts.len(),
                    part_path.display(),
                    burnpack_path.display()
                )
            })?
        } else if let Some(manifest_url) = manifest_source_url.as_deref() {
            let part_url = resolve_manifest_part_url(manifest_url, part.path.as_str());
            let burnpack_bytes = virtual_fs::fetch_url(part_url.as_str()).map_err(|err| {
                format!(
                    "failed to fetch burnpack part {}/{} '{}' for '{}': {err}",
                    index + 1,
                    manifest.parts.len(),
                    part_url,
                    burnpack_path.display()
                )
            })?;
            load_blob_bytes_from_burnpack_bytes(&burnpack_bytes).map_err(|err| {
                format!(
                    "failed to decode fetched burnpack part {}/{} '{}' for '{}': {err}",
                    index + 1,
                    manifest.parts.len(),
                    part_url,
                    burnpack_path.display()
                )
            })?
        } else {
            load_blob_from_burnpack(&part_path).map_err(|err| {
                format!(
                    "failed to load burnpack part {}/{} '{}' for '{}': {err}",
                    index + 1,
                    manifest.parts.len(),
                    part_path.display(),
                    burnpack_path.display()
                )
            })?
        };
        if part.bytes > 0 {
            let payload_bytes = bytes.len() as u64;
            let matches_file_size = part_file_size == part.bytes;
            let matches_payload_size = payload_bytes == part.bytes;
            if !matches_file_size && !matches_payload_size {
                return Err(format!(
                    "burnpack part '{}' size mismatch: manifest={} file={} payload={}",
                    part_path.display(),
                    part.bytes,
                    part_file_size,
                    payload_bytes
                ));
            }
            all_part_sizes_match_file_sizes &= matches_file_size;
            all_part_sizes_match_payload_sizes &= matches_payload_size;
        } else {
            all_part_sizes_match_file_sizes = false;
            all_part_sizes_match_payload_sizes = false;
        }
        merged.extend_from_slice(bytes.as_slice());
    }

    if manifest.total_bytes > 0 {
        if all_part_sizes_match_file_sizes {
            if file_size_total != manifest.total_bytes {
                return Err(format!(
                    "assembled burnpack part file sizes for '{}' produced {} bytes, expected {}",
                    burnpack_path.display(),
                    file_size_total,
                    manifest.total_bytes
                ));
            }
        } else if all_part_sizes_match_payload_sizes || merged.len() as u64 != manifest.total_bytes
        {
            if merged.len() as u64 != manifest.total_bytes {
                return Err(format!(
                    "assembled burnpack parts for '{}' produced {} payload bytes, expected {}",
                    burnpack_path.display(),
                    merged.len(),
                    manifest.total_bytes
                ));
            }
        }
    }

    Ok(merged)
}

fn resolve_manifest_part_url(manifest_url: &str, entry: &str) -> String {
    if entry.contains("://") || entry.starts_with('/') {
        return entry.to_string();
    }
    let normalized = entry.replace('\\', "/");
    if let Some((parent, _)) = manifest_url.rsplit_once('/') {
        return format!("{}/{}", parent.trim_end_matches('/'), normalized);
    }
    normalized
}

fn resolve_manifest_part_path(manifest_path: &Path, entry: &str) -> Result<PathBuf, String> {
    let entry_path = Path::new(entry);
    if entry_path.is_absolute() {
        return Ok(entry_path.to_path_buf());
    }
    manifest_path
        .parent()
        .map(|parent| parent.join(entry_path))
        .ok_or_else(|| {
            format!(
                "invalid burnpack parts manifest path '{}'",
                manifest_path.display()
            )
        })
}

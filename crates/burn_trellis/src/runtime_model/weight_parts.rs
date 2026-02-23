use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

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
    path.exists() || burnpack_parts_manifest_path(path).exists()
}

pub(crate) fn load_blob_bytes_from_burnpack_or_parts<F>(
    burnpack_path: &Path,
    mut load_blob_from_burnpack: F,
) -> Result<Vec<u8>, String>
where
    F: FnMut(&Path) -> Result<Vec<u8>, String>,
{
    if burnpack_path.exists() {
        return load_blob_from_burnpack(burnpack_path);
    }

    let manifest_path = burnpack_parts_manifest_path(burnpack_path);
    let manifest_bytes = fs::read(&manifest_path).map_err(|err| {
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
    for (index, part) in manifest.parts.iter().enumerate() {
        let part_path = resolve_manifest_part_path(&manifest_path, part.path.as_str())?;
        let bytes = load_blob_from_burnpack(&part_path).map_err(|err| {
            format!(
                "failed to load burnpack part {}/{} '{}' for '{}': {err}",
                index + 1,
                manifest.parts.len(),
                part_path.display(),
                burnpack_path.display()
            )
        })?;
        if part.bytes > 0 && bytes.len() as u64 != part.bytes {
            return Err(format!(
                "burnpack part '{}' size mismatch: manifest={} loaded={}",
                part_path.display(),
                part.bytes,
                bytes.len()
            ));
        }
        merged.extend_from_slice(bytes.as_slice());
    }

    if manifest.total_bytes > 0 && merged.len() as u64 != manifest.total_bytes {
        return Err(format!(
            "assembled burnpack parts for '{}' produced {} bytes, expected {}",
            burnpack_path.display(),
            merged.len(),
            manifest.total_bytes
        ));
    }

    Ok(merged)
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

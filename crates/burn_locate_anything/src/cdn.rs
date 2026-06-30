use std::fs;
use std::path::{Path, PathBuf};

use crate::blob_burnpack::extract_blob_burnpack_or_parts_to_file;
use crate::import::LocateAnythingPrecision;
use crate::runtime::LocateAnythingRuntimeConfig;
use crate::{LocateAnythingError, LocateAnythingResult};

pub const LOCATE_ANYTHING_CDN_MODEL_DIR: &str = "LocateAnything-3B";
const IMPORT_MANIFEST_FILE: &str = "locate_anything_import_manifest.json";

const TOKENIZER_FILES: &[&str] = &[
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

pub fn locate_anything_cdn_root_prefix() -> String {
    format!("model/{LOCATE_ANYTHING_CDN_MODEL_DIR}")
}

pub fn locate_anything_cdn_root_url(base_url: &str) -> String {
    let trimmed = base_url.trim_end_matches('/');
    if trimmed.ends_with(LOCATE_ANYTHING_CDN_MODEL_DIR) {
        trimmed.to_string()
    } else if trimmed.ends_with("/model") || trimmed.ends_with("model") {
        join_url(trimmed, LOCATE_ANYTHING_CDN_MODEL_DIR)
    } else {
        join_url(trimmed, &locate_anything_cdn_root_prefix())
    }
}

pub fn resolve_or_download_model_root(
    config: &LocateAnythingRuntimeConfig,
) -> LocateAnythingResult<PathBuf> {
    if local_model_root_is_usable(&config.model_root) {
        return Ok(config.model_root.clone());
    }
    if !config.allow_download {
        return Ok(config.model_root.clone());
    }
    let Some(base_url) = config.cdn_base_url.as_deref() else {
        return Ok(config.model_root.clone());
    };
    let cache_root = config
        .cache_dir
        .clone()
        .unwrap_or_else(default_locate_anything_cache_dir)
        .join(LOCATE_ANYTHING_CDN_MODEL_DIR)
        .join(config.precision.to_string());
    fs::create_dir_all(&cache_root).map_err(|err| {
        LocateAnythingError::Io(format!("create {}: {err}", cache_root.display()))
    })?;
    let remote_root = locate_anything_cdn_root_url(base_url);
    download_metadata_files(&remote_root, &cache_root)?;
    download_and_extract_weight_shards(&remote_root, &cache_root, config.precision)?;
    Ok(cache_root)
}

pub fn default_locate_anything_cache_dir() -> PathBuf {
    std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))
        .unwrap_or_else(|| PathBuf::from(".cache"))
        .join("burn_synth")
        .join("models")
}

fn local_model_root_is_usable(root: &Path) -> bool {
    root.join("config.json").exists()
        && root.join("model.safetensors.index.json").exists()
        && root.join("vocab.json").exists()
        && root.join("merges.txt").exists()
}

fn download_metadata_files(remote_root: &str, cache_root: &Path) -> LocateAnythingResult<()> {
    for file in TOKENIZER_FILES {
        let destination = cache_root.join(file);
        if destination.exists() {
            continue;
        }
        download_required_file(&join_url(remote_root, file), &destination)?;
    }
    let import_manifest = cache_root.join(IMPORT_MANIFEST_FILE);
    if !import_manifest.exists()
        && let Ok(bytes) = download_bytes(&join_url(remote_root, IMPORT_MANIFEST_FILE))
    {
        write_file_atomically(&import_manifest, &bytes)?;
    }
    Ok(())
}

fn download_and_extract_weight_shards(
    remote_root: &str,
    cache_root: &Path,
    precision: LocateAnythingPrecision,
) -> LocateAnythingResult<()> {
    let index_path = cache_root.join("model.safetensors.index.json");
    let index = fs::read(&index_path)
        .map_err(|err| LocateAnythingError::Io(format!("read {}: {err}", index_path.display())))?;
    let value = serde_json::from_slice::<serde_json::Value>(&index).map_err(|err| {
        LocateAnythingError::Config(format!("parse {}: {err}", index_path.display()))
    })?;
    let mut sources = value
        .get("weight_map")
        .and_then(|value| value.as_object())
        .into_iter()
        .flat_map(|map| map.values())
        .filter_map(|value| value.as_str())
        .map(str::to_string)
        .collect::<Vec<_>>();
    sources.sort();
    sources.dedup();
    for source in sources {
        let safetensors_name = source
            .strip_suffix(".bpk")
            .map(|stem| format!("{stem}.safetensors"))
            .unwrap_or(source.clone());
        let destination = cache_root.join(&safetensors_name);
        if destination.exists() {
            continue;
        }
        let bpk_name = source
            .strip_suffix(".safetensors")
            .map(|stem| format!("{stem}_{}.bpk", precision))
            .unwrap_or_else(|| {
                source
                    .strip_suffix(".bpk")
                    .map(|stem| format!("{stem}.bpk"))
                    .unwrap_or_else(|| format!("{source}.bpk"))
            });
        let local_bpk = cache_root.join(&bpk_name);
        let manifest_name = format!("{bpk_name}.parts.json");
        let local_manifest = cache_root.join(&manifest_name);
        if !local_manifest.exists() {
            download_required_file(&join_url(remote_root, &manifest_name), &local_manifest)?;
        }
        let manifest = fs::read(&local_manifest).map_err(|err| {
            LocateAnythingError::Io(format!("read {}: {err}", local_manifest.display()))
        })?;
        let parts = serde_json::from_slice::<serde_json::Value>(&manifest)
            .map_err(|err| {
                LocateAnythingError::Config(format!("parse {}: {err}", local_manifest.display()))
            })?
            .get("parts")
            .and_then(|value| value.as_array())
            .cloned()
            .unwrap_or_default();
        for part in parts {
            let Some(path) = part.get("path").and_then(|value| value.as_str()) else {
                continue;
            };
            let part_destination = cache_root.join(path);
            let expected_bytes = part.get("bytes").and_then(|value| value.as_u64());
            let expected_sha256 = part.get("sha256").and_then(|value| value.as_str());
            if !part_file_is_valid(&part_destination, expected_bytes, expected_sha256)? {
                download_required_file(&join_url(remote_root, path), &part_destination)?;
            }
            validate_part_file(&part_destination, expected_bytes, expected_sha256)?;
        }
        extract_blob_burnpack_or_parts_to_file(&local_bpk, &destination)?;
    }
    Ok(())
}

fn download_required_file(url: &str, destination: &Path) -> LocateAnythingResult<()> {
    let bytes = download_bytes(url)?;
    write_file_atomically(destination, &bytes)
}

fn part_file_is_valid(
    path: &Path,
    expected_bytes: Option<u64>,
    expected_sha256: Option<&str>,
) -> LocateAnythingResult<bool> {
    if !path.exists() {
        return Ok(false);
    }
    match validate_part_file(path, expected_bytes, expected_sha256) {
        Ok(()) => Ok(true),
        Err(_) => {
            let _ = fs::remove_file(path);
            Ok(false)
        }
    }
}

fn validate_part_file(
    path: &Path,
    expected_bytes: Option<u64>,
    expected_sha256: Option<&str>,
) -> LocateAnythingResult<()> {
    if let Some(expected_bytes) = expected_bytes {
        let actual = fs::metadata(path)
            .map_err(|err| LocateAnythingError::Io(format!("metadata {}: {err}", path.display())))?
            .len();
        if actual != expected_bytes {
            return Err(LocateAnythingError::Io(format!(
                "LocateAnything part {} size mismatch: expected {expected_bytes}, got {actual}",
                path.display()
            )));
        }
    }
    if let Some(expected_sha256) = expected_sha256.filter(|value| !value.trim().is_empty()) {
        let bytes = fs::read(path)
            .map_err(|err| LocateAnythingError::Io(format!("read {}: {err}", path.display())))?;
        let actual = sha256_bytes(&bytes);
        if !actual.eq_ignore_ascii_case(expected_sha256.trim()) {
            return Err(LocateAnythingError::Io(format!(
                "LocateAnything part {} sha256 mismatch: expected {}, got {}",
                path.display(),
                expected_sha256.trim(),
                actual
            )));
        }
    }
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn download_bytes(url: &str) -> LocateAnythingResult<Vec<u8>> {
    let response = ureq::get(url).call().map_err(|err| {
        LocateAnythingError::Io(format!("download LocateAnything artifact {url}: {err}"))
    })?;
    let mut reader = response.into_reader();
    let mut bytes = Vec::new();
    std::io::Read::read_to_end(&mut reader, &mut bytes)
        .map_err(|err| LocateAnythingError::Io(format!("read {url}: {err}")))?;
    Ok(bytes)
}

#[cfg(target_arch = "wasm32")]
fn download_bytes(url: &str) -> LocateAnythingResult<Vec<u8>> {
    Err(LocateAnythingError::Unsupported(format!(
        "LocateAnything CDN download is not implemented inside wasm yet; preload {url}"
    )))
}

fn write_file_atomically(path: &Path, bytes: &[u8]) -> LocateAnythingResult<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            LocateAnythingError::Io(format!("create {}: {err}", parent.display()))
        })?;
    }
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, bytes)
        .map_err(|err| LocateAnythingError::Io(format!("write {}: {err}", tmp.display())))?;
    fs::rename(&tmp, path).map_err(|err| {
        LocateAnythingError::Io(format!(
            "move {} to {}: {err}",
            tmp.display(),
            path.display()
        ))
    })
}

fn sha256_bytes(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn join_url(base: &str, path: &str) -> String {
    format!(
        "{}/{}",
        base.trim_end_matches('/'),
        path.trim_start_matches('/')
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cdn_root_url_accepts_base_or_model_root() {
        assert_eq!(
            locate_anything_cdn_root_url("https://cdn.example.invalid"),
            "https://cdn.example.invalid/model/LocateAnything-3B"
        );
        assert_eq!(
            locate_anything_cdn_root_url("https://cdn.example.invalid/model"),
            "https://cdn.example.invalid/model/LocateAnything-3B"
        );
        assert_eq!(
            locate_anything_cdn_root_url("https://cdn.example.invalid/model/LocateAnything-3B"),
            "https://cdn.example.invalid/model/LocateAnything-3B"
        );
    }
}

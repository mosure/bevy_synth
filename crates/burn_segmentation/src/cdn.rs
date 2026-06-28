use std::fs;
#[cfg(not(target_arch = "wasm32"))]
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::config::component_burnpack_file_name;
#[cfg(not(target_arch = "wasm32"))]
use crate::config::required_components;
use crate::config::{SegmentationModelComponent, SegmentationPrecision, SegmentationQuantization};
use crate::{
    SegmentationError, SegmentationModelKind, SegmentationResult, SegmentationRuntimeConfig,
};

const ONE_MIB: u64 = 1024 * 1024;
#[cfg(not(target_arch = "wasm32"))]
const DOWNLOAD_CONNECT_TIMEOUT_SECS: u64 = 20;
#[cfg(not(target_arch = "wasm32"))]
const DOWNLOAD_READ_TIMEOUT_SECS: u64 = 45;
#[cfg(not(target_arch = "wasm32"))]
const DOWNLOAD_WRITE_TIMEOUT_SECS: u64 = 45;
#[cfg(not(target_arch = "wasm32"))]
const DOWNLOAD_MAX_ATTEMPTS: u32 = 4;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct SegmentationFilePartsManifest {
    #[serde(default = "default_manifest_version")]
    pub version: u32,
    #[serde(default)]
    pub source_file: String,
    #[serde(default)]
    pub total_bytes: u64,
    #[serde(default)]
    pub max_part_bytes: u64,
    #[serde(default)]
    pub parts: Vec<SegmentationFilePartEntry>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct SegmentationFilePartEntry {
    pub path: String,
    #[serde(default)]
    pub bytes: u64,
    #[serde(default)]
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct SegmentationFilePartsReport {
    pub manifest_path: PathBuf,
    pub part_paths: Vec<PathBuf>,
    pub total_bytes: u64,
}

const fn default_manifest_version() -> u32 {
    1
}

pub fn segmentation_cdn_root_prefix(
    model: SegmentationModelKind,
    _precision: SegmentationPrecision,
    _quantization: SegmentationQuantization,
) -> String {
    format!("model/{}", segmentation_cdn_model_dir(model))
}

pub fn segmentation_cdn_root_url(
    base_url: &str,
    model: SegmentationModelKind,
    precision: SegmentationPrecision,
    quantization: SegmentationQuantization,
) -> String {
    let prefix = segmentation_cdn_root_prefix(model, precision, quantization);
    let trimmed = base_url.trim_end_matches('/');
    if trimmed.ends_with(&prefix) || trimmed.ends_with(segmentation_cdn_model_dir(model)) {
        trimmed.to_string()
    } else if trimmed.ends_with("/model")
        || trimmed.ends_with("model")
        || trimmed.ends_with("/models")
        || trimmed.ends_with("models")
    {
        join_url(trimmed, segmentation_cdn_model_dir(model))
    } else {
        join_url(trimmed, &prefix)
    }
}

pub fn segmentation_cdn_model_dir(model: SegmentationModelKind) -> &'static str {
    match model {
        SegmentationModelKind::BboxPrompt => "BboxPrompt",
        SegmentationModelKind::Sam2 => "SAM2.1",
        SegmentationModelKind::Sam3 => "SAM3",
    }
}

pub fn component_burnpack_rel_path(
    component: SegmentationModelComponent,
    precision: SegmentationPrecision,
    quantization: SegmentationQuantization,
) -> String {
    component_burnpack_file_name(component, precision, quantization)
}

pub fn component_safetensors_file_name(component: SegmentationModelComponent) -> String {
    format!("{}.safetensors", component.label())
}

pub fn component_safetensors_rel_path(component: SegmentationModelComponent) -> String {
    format!("components/{}", component_safetensors_file_name(component))
}

pub fn file_parts_manifest_path(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("model.bin");
    path.with_file_name(format!("{file_name}.parts.json"))
}

pub fn read_file_parts_manifest(path: &Path) -> SegmentationResult<SegmentationFilePartsManifest> {
    let bytes = fs::read(path)
        .map_err(|err| SegmentationError::Io(format!("read {}: {err}", path.display())))?;
    serde_json::from_slice(&bytes)
        .map_err(|err| SegmentationError::Image(format!("parse {}: {err}", path.display())))
}

pub fn resolve_file_part_entry_path(
    manifest_path: &Path,
    entry_path: &str,
) -> SegmentationResult<PathBuf> {
    let entry = Path::new(entry_path);
    if entry.is_absolute() {
        return Ok(entry.to_path_buf());
    }
    let parent = manifest_path.parent().ok_or_else(|| {
        SegmentationError::Io(format!(
            "parts manifest has no parent directory: {}",
            manifest_path.display()
        ))
    })?;
    Ok(parent.join(entry))
}

pub fn write_file_parts_for_cdn(
    source_path: &Path,
    max_part_size_mib: u64,
    overwrite: bool,
) -> SegmentationResult<Option<SegmentationFilePartsReport>> {
    if !source_path.exists() {
        return Err(SegmentationError::Io(format!(
            "source file does not exist for parting: {}",
            source_path.display()
        )));
    }
    let max_part_bytes = max_part_size_mib.max(1).saturating_mul(ONE_MIB);
    let total_bytes = fs::metadata(source_path)
        .map_err(|err| SegmentationError::Io(format!("metadata {}: {err}", source_path.display())))?
        .len();
    let manifest_path = file_parts_manifest_path(source_path);
    if manifest_path.exists() && !overwrite && file_parts_manifest_is_complete(&manifest_path)? {
        let manifest = read_file_parts_manifest(&manifest_path)?;
        let part_paths = manifest
            .parts
            .iter()
            .map(|part| resolve_file_part_entry_path(&manifest_path, &part.path))
            .collect::<SegmentationResult<Vec<_>>>()?;
        return Ok(Some(SegmentationFilePartsReport {
            manifest_path,
            part_paths,
            total_bytes: manifest.total_bytes,
        }));
    }
    if let Some(parent) = manifest_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| SegmentationError::Io(format!("create {}: {err}", parent.display())))?;
    }
    if overwrite {
        cleanup_existing_file_parts(&manifest_path)?;
    }

    let source_file_name = source_path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| {
            SegmentationError::Io(format!(
                "invalid source file name: {}",
                source_path.display()
            ))
        })?;
    let mut source = fs::File::open(source_path)
        .map_err(|err| SegmentationError::Io(format!("open {}: {err}", source_path.display())))?;
    let mut parts = Vec::new();
    let mut part_paths = Vec::new();
    let mut buffer = vec![0_u8; max_part_bytes.min(16 * ONE_MIB) as usize];
    let mut index = 0usize;
    loop {
        let part_name = format!("{source_file_name}.part-{index:05}");
        let part_path = source_path.with_file_name(&part_name);
        let mut part = fs::File::create(&part_path).map_err(|err| {
            SegmentationError::Io(format!("create {}: {err}", part_path.display()))
        })?;
        let mut remaining = max_part_bytes;
        let mut written = 0u64;
        while remaining > 0 {
            let read_len = buffer.len().min(remaining as usize);
            let read = source.read(&mut buffer[..read_len]).map_err(|err| {
                SegmentationError::Io(format!("read {}: {err}", source_path.display()))
            })?;
            if read == 0 {
                break;
            }
            part.write_all(&buffer[..read]).map_err(|err| {
                SegmentationError::Io(format!("write {}: {err}", part_path.display()))
            })?;
            written = written.saturating_add(read as u64);
            remaining = remaining.saturating_sub(read as u64);
        }
        if written == 0 {
            fs::remove_file(&part_path).ok();
            break;
        }
        part.flush().map_err(|err| {
            SegmentationError::Io(format!("flush {}: {err}", part_path.display()))
        })?;
        parts.push(SegmentationFilePartEntry {
            path: part_name,
            bytes: written,
            sha256: sha256_file(&part_path)?,
        });
        part_paths.push(part_path);
        index += 1;
    }

    let manifest = SegmentationFilePartsManifest {
        version: 1,
        source_file: source_file_name.to_string(),
        total_bytes,
        max_part_bytes,
        parts,
    };
    fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest)
            .map_err(|err| SegmentationError::Image(format!("serialize parts manifest: {err}")))?,
    )
    .map_err(|err| SegmentationError::Io(format!("write {}: {err}", manifest_path.display())))?;
    Ok(Some(SegmentationFilePartsReport {
        manifest_path,
        part_paths,
        total_bytes,
    }))
}

pub fn file_parts_manifest_is_complete(manifest_path: &Path) -> SegmentationResult<bool> {
    if !manifest_path.exists() {
        return Ok(false);
    }
    let manifest = read_file_parts_manifest(manifest_path)?;
    if manifest.parts.is_empty() {
        return Ok(false);
    }
    let mut total = 0u64;
    for part in &manifest.parts {
        let part_path = resolve_file_part_entry_path(manifest_path, &part.path)?;
        if !part_path.exists() {
            return Ok(false);
        }
        let bytes = fs::metadata(&part_path)
            .map_err(|err| {
                SegmentationError::Io(format!("metadata {}: {err}", part_path.display()))
            })?
            .len();
        if part.bytes > 0 && bytes != part.bytes {
            return Ok(false);
        }
        if !part.sha256.trim().is_empty()
            && !sha256_file(&part_path)?.eq_ignore_ascii_case(part.sha256.trim())
        {
            return Ok(false);
        }
        total = total.saturating_add(bytes);
    }
    Ok(manifest.total_bytes == 0 || total == manifest.total_bytes)
}

pub fn assemble_file_parts(manifest_path: &Path, destination: &Path) -> SegmentationResult<()> {
    let manifest = read_file_parts_manifest(manifest_path)?;
    if manifest.parts.is_empty() {
        return Err(SegmentationError::Image(format!(
            "parts manifest {} contains no parts",
            manifest_path.display()
        )));
    }
    ensure_parent_dir(destination)?;
    let partial = partial_path(destination);
    let mut output = fs::File::create(&partial)
        .map_err(|err| SegmentationError::Io(format!("create {}: {err}", partial.display())))?;
    let mut total = 0u64;
    let mut buffer = vec![0_u8; 1024 * 1024];
    for part in &manifest.parts {
        let part_path = resolve_file_part_entry_path(manifest_path, &part.path)?;
        let mut input = fs::File::open(&part_path)
            .map_err(|err| SegmentationError::Io(format!("open {}: {err}", part_path.display())))?;
        let mut part_total = 0u64;
        loop {
            let read = input.read(&mut buffer).map_err(|err| {
                SegmentationError::Io(format!("read {}: {err}", part_path.display()))
            })?;
            if read == 0 {
                break;
            }
            output.write_all(&buffer[..read]).map_err(|err| {
                SegmentationError::Io(format!("write {}: {err}", partial.display()))
            })?;
            part_total = part_total.saturating_add(read as u64);
        }
        if part.bytes > 0 && part_total != part.bytes {
            return Err(SegmentationError::Io(format!(
                "part {} byte mismatch: got {}, expected {}",
                part_path.display(),
                part_total,
                part.bytes
            )));
        }
        total = total.saturating_add(part_total);
    }
    output
        .flush()
        .map_err(|err| SegmentationError::Io(format!("flush {}: {err}", partial.display())))?;
    if manifest.total_bytes > 0 && total != manifest.total_bytes {
        return Err(SegmentationError::Io(format!(
            "assembled {} bytes from {}, expected {}",
            total,
            manifest_path.display(),
            manifest.total_bytes
        )));
    }
    if destination.exists() {
        fs::remove_file(destination).map_err(|err| {
            SegmentationError::Io(format!("replace {}: {err}", destination.display()))
        })?;
    }
    fs::rename(&partial, destination)
        .map_err(|err| SegmentationError::Io(format!("rename {}: {err}", destination.display())))
}

#[cfg(not(target_arch = "wasm32"))]
pub fn resolve_or_download_model_root(
    config: &SegmentationRuntimeConfig,
) -> SegmentationResult<PathBuf> {
    if let Some(model_root) = config.model_root.as_ref() {
        if sam_component_files_present(
            model_root,
            config.model,
            config.precision,
            config.quantization,
        ) {
            return Ok(model_root.clone());
        }
        if !config.allow_download {
            return Ok(model_root.clone());
        }
    }
    if !config.allow_download {
        return config.model_root.clone().ok_or_else(|| {
            SegmentationError::Unsupported(
                "segmentation runtime requires model_root or allow_download=true with cdn_base_url"
                    .to_string(),
            )
        });
    }
    let base_url = config.cdn_base_url.as_deref().ok_or_else(|| {
        SegmentationError::Unsupported(
            "segmentation allow_download=true requires cdn_base_url".to_string(),
        )
    })?;
    let cache_root = config.cache_dir.clone().unwrap_or_else(default_cache_root);
    let target_root = config.model_root.clone().unwrap_or_else(|| {
        cache_root
            .join(segmentation_cdn_model_dir(config.model))
            .join(config.precision.label())
            .join(config.quantization.label())
    });
    fs::create_dir_all(&target_root)
        .map_err(|err| SegmentationError::Io(format!("create {}: {err}", target_root.display())))?;
    let remote_root = segmentation_cdn_root_url(
        base_url,
        config.model,
        config.precision,
        config.quantization,
    );
    sync_optional_file(
        &target_root,
        &remote_root,
        "segmentation_import_manifest.json",
    )?;
    sync_optional_file(&target_root, &remote_root, "config.json")?;
    sync_optional_file(&target_root, &remote_root, "preprocessor_config.json")?;
    sync_optional_file(&target_root, &remote_root, "processor_config.json")?;
    sync_optional_file(&target_root, &remote_root, "image_processor_config.json")?;
    for component in required_components(config.model) {
        ensure_component_burnpack(
            &target_root,
            &remote_root,
            *component,
            config.precision,
            config.quantization,
        )?;
    }
    Ok(target_root)
}

#[cfg(target_arch = "wasm32")]
pub fn resolve_or_download_model_root(
    config: &SegmentationRuntimeConfig,
) -> SegmentationResult<PathBuf> {
    if config.allow_download && config.cdn_base_url.is_some() {
        return Err(SegmentationError::Unsupported(
            "segmentation CDN bootstrap is not implemented for wasm yet; use preloaded artifacts"
                .to_string(),
        ));
    }
    config.model_root.clone().ok_or_else(|| {
        SegmentationError::Unsupported("segmentation runtime requires model_root".to_string())
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn sam_component_files_present(
    model_root: &Path,
    model: SegmentationModelKind,
    precision: SegmentationPrecision,
    quantization: SegmentationQuantization,
) -> bool {
    required_components(model).iter().all(|component| {
        component_burnpack_candidates(model_root, *component, precision, quantization)
            .into_iter()
            .any(|path| path.exists())
            || {
                let rel = component_safetensors_rel_path(*component);
                model_root.join(&rel).exists()
                    || model_root
                        .join(component_safetensors_file_name(*component))
                        .exists()
            }
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn ensure_component_burnpack(
    target_root: &Path,
    remote_root: &str,
    component: SegmentationModelComponent,
    precision: SegmentationPrecision,
    quantization: SegmentationQuantization,
) -> SegmentationResult<()> {
    let rel_path = component_burnpack_rel_path(component, precision, quantization);
    let destination = target_root.join(&rel_path);
    if destination.exists() {
        return Ok(());
    }
    let local_manifest = file_parts_manifest_path(&destination);
    let manifest_rel = format!("{rel_path}.parts.json");
    let manifest_url = join_url(remote_root, &manifest_rel);
    if !local_manifest.exists() {
        if let Some(bytes) = download_optional_bytes(&manifest_url)? {
            write_file_atomically(&local_manifest, &bytes)?;
        }
    }
    if local_manifest.exists() {
        let manifest = read_file_parts_manifest(&local_manifest)?;
        for part in &manifest.parts {
            let part_path = resolve_file_part_entry_path(&local_manifest, &part.path)?;
            if cached_part_ready(&part_path, part)? {
                continue;
            }
            let part_url = resolve_manifest_entry_url(&manifest_url, &part.path);
            download_file(
                &part_url,
                &part_path,
                Some(part.bytes),
                Some(part.sha256.as_str()),
            )?;
        }
        if file_parts_manifest_is_complete(&local_manifest)? {
            assemble_file_parts(&local_manifest, &destination)?;
            return Ok(());
        }
    }
    let direct_url = join_url(remote_root, &rel_path);
    download_file(&direct_url, &destination, None, None)
}

#[cfg(not(target_arch = "wasm32"))]
fn component_burnpack_candidates(
    model_root: &Path,
    component: SegmentationModelComponent,
    precision: SegmentationPrecision,
    quantization: SegmentationQuantization,
) -> Vec<PathBuf> {
    let requested = component_burnpack_rel_path(component, precision, quantization);
    let default = component_burnpack_rel_path(
        component,
        SegmentationPrecision::default(),
        SegmentationQuantization::default(),
    );
    let mut candidates = vec![model_root.join(&requested)];
    if requested != default {
        candidates.push(model_root.join(default));
    }
    candidates
}

#[cfg(not(target_arch = "wasm32"))]
fn sync_optional_file(
    target_root: &Path,
    remote_root: &str,
    rel_path: &str,
) -> SegmentationResult<()> {
    let destination = target_root.join(rel_path);
    if destination.exists() {
        return Ok(());
    }
    let url = join_url(remote_root, rel_path);
    let Some(bytes) = download_optional_bytes(&url)? else {
        return Ok(());
    };
    write_file_atomically(&destination, &bytes)
}

#[cfg(not(target_arch = "wasm32"))]
fn download_optional_bytes(url: &str) -> SegmentationResult<Option<Vec<u8>>> {
    let mut last_error = None;
    for _ in 0..DOWNLOAD_MAX_ATTEMPTS {
        match http_agent().get(url).call() {
            Ok(response) => {
                let mut reader = response.into_reader();
                let mut bytes = Vec::new();
                reader.read_to_end(&mut bytes).map_err(|err| {
                    SegmentationError::Io(format!("read response from {url}: {err}"))
                })?;
                return Ok(Some(bytes));
            }
            Err(ureq::Error::Status(404, _)) | Err(ureq::Error::Status(403, _)) => {
                return Ok(None);
            }
            Err(err) => last_error = Some(format_http_error(url, err)),
        }
    }
    Err(SegmentationError::Io(last_error.unwrap_or_else(|| {
        format!("download failed for {url}: unknown error")
    })))
}

#[cfg(not(target_arch = "wasm32"))]
fn download_file(
    url: &str,
    destination: &Path,
    expected_bytes: Option<u64>,
    expected_sha256: Option<&str>,
) -> SegmentationResult<()> {
    ensure_parent_dir(destination)?;
    if destination.exists() {
        if let Some(bytes) = expected_bytes {
            if fs::metadata(destination)
                .map_err(|err| {
                    SegmentationError::Io(format!("metadata {}: {err}", destination.display()))
                })?
                .len()
                != bytes
            {
                fs::remove_file(destination).ok();
            }
        }
        if destination.exists()
            && expected_sha256
                .filter(|value| !value.trim().is_empty())
                .is_none_or(|sha| {
                    sha256_file(destination).is_ok_and(|actual| actual.eq_ignore_ascii_case(sha))
                })
        {
            return Ok(());
        }
    }
    let partial = partial_path(destination);
    let mut last_error = None;
    for _ in 0..DOWNLOAD_MAX_ATTEMPTS {
        match download_file_once(url, &partial, expected_bytes, expected_sha256) {
            Ok(()) => {
                if destination.exists() {
                    fs::remove_file(destination).map_err(|err| {
                        SegmentationError::Io(format!("replace {}: {err}", destination.display()))
                    })?;
                }
                fs::rename(&partial, destination).map_err(|err| {
                    SegmentationError::Io(format!("rename {}: {err}", destination.display()))
                })?;
                return Ok(());
            }
            Err(err) => {
                fs::remove_file(&partial).ok();
                last_error = Some(err.to_string());
            }
        }
    }
    Err(SegmentationError::Io(last_error.unwrap_or_else(|| {
        format!("download failed for {url}: unknown error")
    })))
}

#[cfg(not(target_arch = "wasm32"))]
fn download_file_once(
    url: &str,
    partial: &Path,
    expected_bytes: Option<u64>,
    expected_sha256: Option<&str>,
) -> SegmentationResult<()> {
    let response = http_agent()
        .get(url)
        .call()
        .map_err(|err| SegmentationError::Io(format_http_error(url, err)))?;
    let mut reader = response.into_reader();
    let mut writer = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(partial)
        .map_err(|err| SegmentationError::Io(format!("create {}: {err}", partial.display())))?;
    let copied = std::io::copy(&mut reader, &mut writer)
        .map_err(|err| SegmentationError::Io(format!("write {}: {err}", partial.display())))?;
    writer
        .flush()
        .map_err(|err| SegmentationError::Io(format!("flush {}: {err}", partial.display())))?;
    if let Some(expected) = expected_bytes
        && copied != expected
    {
        return Err(SegmentationError::Io(format!(
            "downloaded {} bytes from {url}, expected {expected}",
            copied
        )));
    }
    if let Some(expected) = expected_sha256.filter(|value| !value.trim().is_empty()) {
        let actual = sha256_file(partial)?;
        if !actual.eq_ignore_ascii_case(expected) {
            return Err(SegmentationError::Io(format!(
                "checksum mismatch for {url}: expected {expected}, got {actual}"
            )));
        }
    }
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn cached_part_ready(path: &Path, part: &SegmentationFilePartEntry) -> SegmentationResult<bool> {
    if !path.exists() {
        return Ok(false);
    }
    let bytes = fs::metadata(path)
        .map_err(|err| SegmentationError::Io(format!("metadata {}: {err}", path.display())))?
        .len();
    if part.bytes > 0 && bytes != part.bytes {
        return Ok(false);
    }
    if !part.sha256.trim().is_empty()
        && !sha256_file(path)?.eq_ignore_ascii_case(part.sha256.trim())
    {
        return Ok(false);
    }
    Ok(true)
}

#[cfg(not(target_arch = "wasm32"))]
fn http_agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(DOWNLOAD_CONNECT_TIMEOUT_SECS))
        .timeout_read(Duration::from_secs(DOWNLOAD_READ_TIMEOUT_SECS))
        .timeout_write(Duration::from_secs(DOWNLOAD_WRITE_TIMEOUT_SECS))
        .build()
}

#[cfg(not(target_arch = "wasm32"))]
fn format_http_error(url: &str, err: ureq::Error) -> String {
    match err {
        ureq::Error::Status(code, response) => {
            format!("HTTP {code} ({}) for {url}", response.status_text())
        }
        ureq::Error::Transport(transport) => {
            format!("transport error while downloading {url}: {transport}")
        }
    }
}

pub fn join_url(root: &str, rel: &str) -> String {
    format!(
        "{}/{}",
        root.trim_end_matches('/'),
        rel.trim_start_matches('/')
    )
}

pub fn resolve_manifest_entry_url(manifest_url: &str, entry_path: &str) -> String {
    if entry_path.starts_with("http://") || entry_path.starts_with("https://") {
        return entry_path.to_string();
    }
    if let Some((parent, _)) = manifest_url.rsplit_once('/') {
        join_url(parent, entry_path)
    } else {
        entry_path.to_string()
    }
}

pub fn sha256_file(path: &Path) -> SegmentationResult<String> {
    let mut file = fs::File::open(path)
        .map_err(|err| SegmentationError::Io(format!("open {}: {err}", path.display())))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
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

fn cleanup_existing_file_parts(manifest_path: &Path) -> SegmentationResult<()> {
    if !manifest_path.exists() {
        return Ok(());
    }
    let manifest = read_file_parts_manifest(manifest_path)?;
    for part in manifest.parts {
        let path = resolve_file_part_entry_path(manifest_path, &part.path)?;
        fs::remove_file(path).ok();
    }
    fs::remove_file(manifest_path).ok();
    Ok(())
}

fn ensure_parent_dir(path: &Path) -> SegmentationResult<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| SegmentationError::Io(format!("create {}: {err}", parent.display())))?;
    }
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn write_file_atomically(path: &Path, bytes: &[u8]) -> SegmentationResult<()> {
    ensure_parent_dir(path)?;
    let partial = partial_path(path);
    fs::write(&partial, bytes)
        .map_err(|err| SegmentationError::Io(format!("write {}: {err}", partial.display())))?;
    if path.exists() {
        fs::remove_file(path)
            .map_err(|err| SegmentationError::Io(format!("replace {}: {err}", path.display())))?;
    }
    fs::rename(&partial, path)
        .map_err(|err| SegmentationError::Io(format!("rename {}: {err}", path.display())))
}

fn partial_path(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("download.bin");
    path.with_file_name(format!("{file_name}.partial"))
}

#[cfg(not(target_arch = "wasm32"))]
fn default_cache_root() -> PathBuf {
    std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))
        .unwrap_or_else(|| PathBuf::from(".cache"))
        .join("burn_segmentation")
        .join("models")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cdn_root_url_accepts_base_or_root_prefix() {
        assert_eq!(
            segmentation_cdn_root_url(
                "https://cdn.example.invalid",
                SegmentationModelKind::Sam2,
                SegmentationPrecision::F16,
                SegmentationQuantization::None,
            ),
            "https://cdn.example.invalid/model/SAM2.1"
        );
        assert_eq!(
            segmentation_cdn_root_url(
                "https://cdn.example.invalid/model",
                SegmentationModelKind::Sam2,
                SegmentationPrecision::F16,
                SegmentationQuantization::None,
            ),
            "https://cdn.example.invalid/model/SAM2.1"
        );
        assert_eq!(
            segmentation_cdn_root_url(
                "https://cdn.example.invalid/models",
                SegmentationModelKind::Sam2,
                SegmentationPrecision::F16,
                SegmentationQuantization::None,
            ),
            "https://cdn.example.invalid/models/SAM2.1"
        );
        assert_eq!(
            segmentation_cdn_root_url(
                "https://cdn.example.invalid/model/SAM2.1",
                SegmentationModelKind::Sam2,
                SegmentationPrecision::F16,
                SegmentationQuantization::None,
            ),
            "https://cdn.example.invalid/model/SAM2.1"
        );
    }

    #[test]
    fn file_parts_round_trip_and_assemble() {
        let root = std::env::temp_dir().join(format!(
            "burn_segmentation_file_parts_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let source = root.join("image_encoder_f16.bpk");
        fs::write(&source, b"abcdefghijklmnopqrstuvwxyz").unwrap();
        let report = write_file_parts_for_cdn(&source, 1, true)
            .unwrap()
            .expect("parts report");
        assert!(report.manifest_path.exists());
        assert!(!report.part_paths.is_empty());
        assert!(file_parts_manifest_is_complete(&report.manifest_path).unwrap());
        let assembled = root.join("assembled.safetensors");
        assemble_file_parts(&report.manifest_path, &assembled).unwrap();
        assert_eq!(fs::read(source).unwrap(), fs::read(assembled).unwrap());
        fs::remove_dir_all(root).ok();
    }
}

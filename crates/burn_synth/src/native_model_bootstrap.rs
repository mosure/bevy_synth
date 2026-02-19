#![cfg(all(feature = "runtime", not(target_arch = "wasm32")))]

use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::sleep;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use burn_foreground::rmbg14::import::resolve_rmbg_weights_root;
use burn_synth_import::parts::{BurnpackPartEntry, read_parts_manifest, resolve_part_entry_path};
use burn_tripo::paths::resolve_triposg_weights_root;
use log::{info, warn};
use sha2::{Digest, Sha256};

use crate::model_loader::{candidate_burnpack_names, parse_parts_manifest_bytes};

const DEFAULT_MODEL_BASE_URL: &str = "https://aberration.technology/model";
const CACHE_MODELS_DIR: &str = ".burn_synth/models";
const TRIPOSG_DIR: &str = "MIDI-3D";
const RMBG14_DIR: &str = "RMBG-1.4";
const DOWNLOAD_MAX_ATTEMPTS: u32 = 6;
const DOWNLOAD_RETRY_BASE_DELAY_MS: u64 = 800;
const DOWNLOAD_CONNECT_TIMEOUT_SECS: u64 = 20;
const DOWNLOAD_READ_TIMEOUT_SECS: u64 = 45;
const DOWNLOAD_WRITE_TIMEOUT_SECS: u64 = 45;
const DOWNLOAD_PROGRESS_LOG_EVERY_SECS: u64 = 10;

type BootstrapStatusCallback = Arc<dyn Fn(String) + Send + Sync + 'static>;
static BOOTSTRAP_STATUS_CALLBACK: OnceLock<Mutex<Option<BootstrapStatusCallback>>> =
    OnceLock::new();

const TRIPOSG_OPTIONAL_TEXT_RELPATHS: &[&str] = &[
    "vae/config.json",
    "transformer/config.json",
    "scheduler/scheduler_config.json",
    "image_encoder_dinov2/config.json",
    "image_encoder_2/config.json",
    "feature_extractor_dinov2/preprocessor_config.json",
    "feature_extractor_2/preprocessor_config.json",
    "feature_extractor_1/preprocessor_config.json",
];

const TRIPOSG_REQUIRED_PARTS_BASES: &[&str] = &[
    "vae/diffusion_pytorch_model.safetensors",
    "transformer/diffusion_pytorch_model.safetensors",
    "image_encoder_dinov2/model.safetensors",
];

const RMBG14_OPTIONAL_TEXT_RELPATHS: &[&str] = &["config.json", "preprocessor_config.json"];
const RMBG14_REQUIRED_PARTS_BASES: &[&str] = &["model.safetensors"];

pub fn set_bootstrap_status_callback(callback: Option<BootstrapStatusCallback>) {
    let lock = BOOTSTRAP_STATUS_CALLBACK.get_or_init(|| Mutex::new(None));
    if let Ok(mut guard) = lock.lock() {
        *guard = callback;
    }
}

fn emit_status(message: impl Into<String>) {
    let message = message.into();
    info!("{message}");
    let lock = BOOTSTRAP_STATUS_CALLBACK.get_or_init(|| Mutex::new(None));
    if let Ok(guard) = lock.lock()
        && let Some(callback) = guard.as_ref()
    {
        callback(message);
    }
}

pub(crate) fn resolve_or_bootstrap_triposg_root(prefer_f16: bool) -> Result<PathBuf, String> {
    let cache_root = default_cache_models_root();
    let target_root = cache_root.join(TRIPOSG_DIR);
    let remote_root = triposg_remote_root();

    match ensure_model_ready(
        &target_root,
        &remote_root,
        prefer_f16,
        TRIPOSG_OPTIONAL_TEXT_RELPATHS,
        TRIPOSG_REQUIRED_PARTS_BASES,
        "TripoSG",
    ) {
        Ok(()) => {
            ensure_triposg_metadata_aliases(&target_root)?;
            Ok(target_root)
        }
        Err(cache_err) => {
            let fallback = resolve_triposg_weights_root(None);
            if fallback.exists() {
                warn!(
                    "TripoSG cache bootstrap failed ({cache_err}); falling back to {}",
                    fallback.display()
                );
                Ok(fallback)
            } else {
                Err(cache_err)
            }
        }
    }
}

pub(crate) fn resolve_or_bootstrap_rmbg14_root(prefer_f16: bool) -> Result<PathBuf, String> {
    let cache_root = default_cache_models_root();
    let target_root = cache_root.join(RMBG14_DIR);
    let remote_root = rmbg14_remote_root();

    match ensure_model_ready(
        &target_root,
        &remote_root,
        prefer_f16,
        RMBG14_OPTIONAL_TEXT_RELPATHS,
        RMBG14_REQUIRED_PARTS_BASES,
        "RMBG-1.4",
    ) {
        Ok(()) => Ok(target_root),
        Err(cache_err) => {
            let fallback = resolve_rmbg_weights_root();
            if fallback.exists() {
                warn!(
                    "RMBG-1.4 cache bootstrap failed ({cache_err}); falling back to {}",
                    fallback.display()
                );
                Ok(fallback)
            } else {
                Err(cache_err)
            }
        }
    }
}

fn ensure_model_ready(
    local_root: &Path,
    remote_root: &str,
    prefer_f16: bool,
    optional_text_relpaths: &[&str],
    required_parts_bases: &[&str],
    label: &str,
) -> Result<(), String> {
    fs::create_dir_all(local_root).map_err(|err| {
        format!(
            "failed to create {label} cache directory {}: {err}",
            local_root.display()
        )
    })?;

    for rel in optional_text_relpaths {
        sync_optional_text_file(local_root, remote_root, rel)?;
    }

    for base in required_parts_bases {
        ensure_parts_bundle(local_root, remote_root, base, prefer_f16, label)?;
    }

    emit_status(format!(
        "{label} weights ready under {}",
        local_root.display()
    ));
    Ok(())
}

fn ensure_triposg_metadata_aliases(local_root: &Path) -> Result<(), String> {
    ensure_alias_text_file(
        local_root,
        "image_encoder_dinov2/config.json",
        &["image_encoder_2/config.json", "image_encoder_1/config.json"],
    )?;
    ensure_alias_text_file(
        local_root,
        "feature_extractor_dinov2/preprocessor_config.json",
        &[
            "feature_extractor_2/preprocessor_config.json",
            "feature_extractor_1/preprocessor_config.json",
        ],
    )?;
    Ok(())
}

fn ensure_alias_text_file(
    local_root: &Path,
    target_rel: &str,
    source_rels: &[&str],
) -> Result<(), String> {
    let target_path = local_root.join(target_rel);
    if target_path.exists() {
        return Ok(());
    }
    let Some(source_rel) = source_rels
        .iter()
        .find(|candidate| local_root.join(candidate).exists())
    else {
        return Ok(());
    };
    let source_path = local_root.join(source_rel);
    ensure_parent_dir(&target_path)?;
    fs::copy(&source_path, &target_path).map_err(|err| {
        format!(
            "failed to create metadata alias {} <- {}: {err}",
            target_path.display(),
            source_path.display()
        )
    })?;
    emit_status(format!(
        "Created metadata alias {} <- {}",
        target_path.display(),
        source_path.display()
    ));
    Ok(())
}

fn ensure_parts_bundle(
    local_root: &Path,
    remote_root: &str,
    base_safetensors_rel: &str,
    prefer_f16: bool,
    label: &str,
) -> Result<(), String> {
    let mut checked = Vec::new();
    let candidates = candidate_burnpack_names(base_safetensors_rel, prefer_f16);
    for candidate in candidates {
        let manifest_rel = format!("{candidate}.parts.json");
        let local_manifest_path = local_root.join(&manifest_rel);
        if manifest_is_complete(&local_manifest_path)? {
            return Ok(());
        }

        let manifest_url = join_url(remote_root, &manifest_rel);
        checked.push(manifest_url.clone());
        let Some(manifest_bytes) = download_optional_bytes(&manifest_url)? else {
            continue;
        };
        let manifest = parse_parts_manifest_bytes(&manifest_bytes, &manifest_url)?;
        if manifest.parts.is_empty() {
            return Err(format!(
                "parts manifest {manifest_url} for {label} is empty"
            ));
        }
        write_file_atomically(&local_manifest_path, &manifest_bytes)?;

        let part_count = manifest.parts.len();
        for (index, part) in manifest.parts.iter().enumerate() {
            let local_part_path = resolve_part_entry_path(&local_manifest_path, &part.path)?;
            if part_matches_cache(&local_part_path, part)? {
                continue;
            }
            let part_url = resolve_manifest_entry_url(&manifest_url, &part.path);
            let part_number = index + 1;
            emit_status(format!(
                "Downloading {label} part {part_number}/{part_count}: {} -> {} (expected_bytes={})",
                part_url,
                local_part_path.display(),
                part.bytes
            ));
            let downloaded =
                download_part_file(&part_url, &local_part_path, part).map_err(|err| {
                    format!(
                        "failed downloading {label} part {part_number}/{part_count} ({}): {err}",
                        part.path
                    )
                })?;
            emit_status(format!(
                "Downloaded {label} part {part_number}/{part_count}: {} bytes -> {}",
                downloaded,
                local_part_path.display()
            ));
        }

        if manifest_is_complete(&local_manifest_path)? {
            emit_status(format!("Downloaded {label} parts manifest {manifest_url}"));
            return Ok(());
        }
    }

    Err(format!(
        "{label} parts manifest missing for {base_safetensors_rel}; checked: {}",
        checked.join(", ")
    ))
}

fn manifest_is_complete(manifest_path: &Path) -> Result<bool, String> {
    if !manifest_path.exists() {
        return Ok(false);
    }
    let manifest = match read_parts_manifest(manifest_path) {
        Ok(manifest) => manifest,
        Err(_) => return Ok(false),
    };
    if manifest.parts.is_empty() {
        return Ok(false);
    }
    for part in &manifest.parts {
        let path = resolve_part_entry_path(manifest_path, &part.path)?;
        if !part_matches_cache(&path, part)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn part_matches_cache(path: &Path, part: &BurnpackPartEntry) -> Result<bool, String> {
    if !path.exists() {
        return Ok(false);
    }
    if part.bytes == 0 {
        return Ok(true);
    }
    let bytes = fs::metadata(path)
        .map_err(|err| format!("failed to read part metadata {}: {err}", path.display()))?
        .len();
    Ok(bytes == part.bytes)
}

fn sync_optional_text_file(
    local_root: &Path,
    remote_root: &str,
    rel_path: &str,
) -> Result<(), String> {
    let local_path = local_root.join(rel_path);
    if local_path.exists() {
        return Ok(());
    }

    let url = join_url(remote_root, rel_path);
    let Some(bytes) = download_optional_bytes(&url)? else {
        return Ok(());
    };
    write_file_atomically(&local_path, &bytes)?;
    emit_status(format!("Downloaded model metadata {url}"));
    Ok(())
}

fn download_part_file(
    url: &str,
    destination: &Path,
    part: &BurnpackPartEntry,
) -> Result<u64, String> {
    ensure_parent_dir(destination)?;
    let partial_path = partial_download_path(destination);
    let mut last_error = None;

    for attempt in 1..=DOWNLOAD_MAX_ATTEMPTS {
        match download_part_file_once(url, destination, &partial_path, part) {
            Ok(bytes) => return Ok(bytes),
            Err(err) => {
                if attempt == DOWNLOAD_MAX_ATTEMPTS {
                    return Err(format!(
                        "download failed after {DOWNLOAD_MAX_ATTEMPTS} attempts for {}: {err}",
                        destination.display()
                    ));
                }
                let delay = retry_delay(attempt);
                let retry_message = format!(
                    "Download attempt {attempt}/{DOWNLOAD_MAX_ATTEMPTS} failed for {}: {err}; retrying in {:.1}s",
                    destination.display(),
                    delay.as_secs_f64()
                );
                warn!("{retry_message}");
                emit_status(retry_message);
                last_error = Some(err);
                sleep(delay);
            }
        }
    }

    Err(last_error.unwrap_or_else(|| {
        format!(
            "download failed for {} with unknown error",
            destination.display()
        )
    }))
}

fn download_optional_bytes(url: &str) -> Result<Option<Vec<u8>>, String> {
    let mut last_error = None;
    for attempt in 1..=DOWNLOAD_MAX_ATTEMPTS {
        match download_optional_bytes_once(url) {
            Ok(value) => return Ok(value),
            Err(err) => {
                if attempt == DOWNLOAD_MAX_ATTEMPTS {
                    return Err(format!(
                        "failed downloading {url} after {DOWNLOAD_MAX_ATTEMPTS} attempts: {err}"
                    ));
                }
                let delay = retry_delay(attempt);
                let retry_message = format!(
                    "Metadata download attempt {attempt}/{DOWNLOAD_MAX_ATTEMPTS} failed for {url}: {err}; retrying in {:.1}s",
                    delay.as_secs_f64()
                );
                warn!("{retry_message}");
                emit_status(retry_message);
                last_error = Some(err);
                sleep(delay);
            }
        }
    }
    Err(last_error.unwrap_or_else(|| format!("failed downloading {url}: unknown error")))
}

fn format_download_error(url: &str, err: ureq::Error) -> String {
    match err {
        ureq::Error::Status(code, response) => {
            format!("HTTP {code} ({}) for {url}", response.status_text())
        }
        ureq::Error::Transport(transport) => {
            format!("transport error while downloading {url}: {transport}")
        }
    }
}

fn download_part_file_once(
    url: &str,
    destination: &Path,
    partial_path: &Path,
    part: &BurnpackPartEntry,
) -> Result<u64, String> {
    let mut resume_from = if partial_path.exists() {
        fs::metadata(partial_path)
            .map_err(|err| {
                format!(
                    "failed to stat partial file {}: {err}",
                    partial_path.display()
                )
            })?
            .len()
    } else {
        0
    };

    if part.bytes > 0 && resume_from > part.bytes {
        let message = format!(
            "Partial file {} exceeds expected size ({} > {}); restarting this part",
            partial_path.display(),
            resume_from,
            part.bytes
        );
        warn!("{message}");
        emit_status(message);
        fs::remove_file(partial_path).map_err(|err| {
            format!(
                "failed to remove stale partial file {}: {err}",
                partial_path.display()
            )
        })?;
        resume_from = 0;
    }
    if part.bytes > 0 && resume_from == part.bytes {
        let expected_sha = part.sha256.trim();
        if !expected_sha.is_empty() {
            let actual_sha = sha256_file(partial_path)?;
            if !actual_sha.eq_ignore_ascii_case(expected_sha) {
                let message = format!(
                    "Checksum mismatch for completed partial file {}; restarting this part",
                    partial_path.display()
                );
                warn!("{message}");
                emit_status(message);
                fs::remove_file(partial_path).map_err(|err| {
                    format!(
                        "failed to remove mismatched partial file {}: {err}",
                        partial_path.display()
                    )
                })?;
                resume_from = 0;
            }
        }
        if resume_from == part.bytes {
            if destination.exists() {
                fs::remove_file(destination).map_err(|err| {
                    format!(
                        "failed to replace stale part {}: {err}",
                        destination.display()
                    )
                })?;
            }
            fs::rename(partial_path, destination).map_err(|err| {
                format!(
                    "failed to move completed partial part into place {}: {err}",
                    destination.display()
                )
            })?;
            return Ok(part.bytes);
        }
    }

    if resume_from > 0 {
        emit_status(format!(
            "Resuming partial download for {} from byte {}",
            destination.display(),
            resume_from
        ));
    }

    let mut request = http_agent().get(url);
    if resume_from > 0 {
        request = request.set("Range", &format!("bytes={resume_from}-"));
    }

    let response = match request.call() {
        Ok(response) => response,
        Err(ureq::Error::Status(416, _)) if resume_from > 0 => {
            fs::remove_file(partial_path).map_err(|err| {
                format!(
                    "range request failed and stale partial file could not be removed {}: {err}",
                    partial_path.display()
                )
            })?;
            return Err(format!(
                "server rejected range request for {url}; cleared stale partial and will retry"
            ));
        }
        Err(err) => return Err(format_download_error(url, err)),
    };

    let status = response.status();
    let append = resume_from > 0 && status == 206;
    if resume_from > 0 && !append {
        let message = format!(
            "Server did not return 206 for resumed request (status={}): restarting {} from byte 0",
            status,
            destination.display()
        );
        warn!("{message}");
        emit_status(message);
        if partial_path.exists() {
            fs::remove_file(partial_path).map_err(|err| {
                format!(
                    "failed to clear partial file {} before restart: {err}",
                    partial_path.display()
                )
            })?;
        }
        resume_from = 0;
    }

    let mut writer = if append {
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(partial_path)
            .map_err(|err| {
                format!(
                    "failed to open partial file {} for append: {err}",
                    partial_path.display()
                )
            })?
    } else {
        fs::File::create(partial_path).map_err(|err| {
            format!(
                "failed to create partial file {}: {err}",
                partial_path.display()
            )
        })?
    };

    let mut reader = response.into_reader();
    let started = Instant::now();
    let mut last_progress_log = Instant::now();
    let mut bytes_written = resume_from;
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|err| format!("failed to read response body from {url}: {err}"))?;
        if read == 0 {
            break;
        }
        writer.write_all(&buffer[..read]).map_err(|err| {
            format!(
                "failed to write downloaded part to {}: {err}",
                partial_path.display()
            )
        })?;
        bytes_written = bytes_written.saturating_add(read as u64);

        if last_progress_log.elapsed() >= Duration::from_secs(DOWNLOAD_PROGRESS_LOG_EVERY_SECS) {
            let elapsed = started.elapsed().as_secs_f64().max(0.001);
            let throughput = (bytes_written.saturating_sub(resume_from)) as f64 / elapsed;
            if part.bytes > 0 {
                let percent =
                    ((bytes_written as f64 / part.bytes as f64) * 100.0).clamp(0.0, 100.0);
                emit_status(format!(
                    "Downloading {}: {:.1}% ({}/{}) {:.1} MiB/s",
                    destination.display(),
                    percent,
                    format_mebibytes(bytes_written),
                    format_mebibytes(part.bytes),
                    throughput / (1024.0 * 1024.0)
                ));
            } else {
                emit_status(format!(
                    "Downloading {}: {} ({:.1} MiB/s)",
                    destination.display(),
                    format_mebibytes(bytes_written),
                    throughput / (1024.0 * 1024.0)
                ));
            }
            last_progress_log = Instant::now();
        }
    }
    writer
        .flush()
        .map_err(|err| format!("failed to flush {}: {err}", partial_path.display()))?;

    if part.bytes > 0 {
        if bytes_written < part.bytes {
            return Err(format!(
                "download ended early for {}: got {} bytes, expected {}",
                destination.display(),
                bytes_written,
                part.bytes
            ));
        }
        if bytes_written > part.bytes {
            let _ = fs::remove_file(partial_path);
            return Err(format!(
                "downloaded too many bytes for {}: got {}, expected {}",
                destination.display(),
                bytes_written,
                part.bytes
            ));
        }
    }

    let expected_sha = part.sha256.trim();
    if !expected_sha.is_empty() {
        let actual_sha = sha256_file(partial_path)?;
        if !actual_sha.eq_ignore_ascii_case(expected_sha) {
            let _ = fs::remove_file(partial_path);
            return Err(format!(
                "checksum mismatch for {}: expected {}, got {}",
                destination.display(),
                expected_sha,
                actual_sha
            ));
        }
    }

    if destination.exists() {
        fs::remove_file(destination).map_err(|err| {
            format!(
                "failed to replace stale part {}: {err}",
                destination.display()
            )
        })?;
    }
    fs::rename(partial_path, destination).map_err(|err| {
        format!(
            "failed to move downloaded part into place {}: {err}",
            destination.display()
        )
    })?;
    Ok(bytes_written)
}

fn download_optional_bytes_once(url: &str) -> Result<Option<Vec<u8>>, String> {
    match http_agent().get(url).call() {
        Ok(response) => {
            let mut reader = response.into_reader();
            let mut bytes = Vec::new();
            reader
                .read_to_end(&mut bytes)
                .map_err(|err| format!("failed to read response from {url}: {err}"))?;
            Ok(Some(bytes))
        }
        Err(ureq::Error::Status(404, _)) | Err(ureq::Error::Status(403, _)) => Ok(None),
        Err(err) => Err(format_download_error(url, err)),
    }
}

fn http_agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(DOWNLOAD_CONNECT_TIMEOUT_SECS))
        .timeout_read(Duration::from_secs(DOWNLOAD_READ_TIMEOUT_SECS))
        .timeout_write(Duration::from_secs(DOWNLOAD_WRITE_TIMEOUT_SECS))
        .build()
}

fn retry_delay(attempt: u32) -> Duration {
    let exponent = attempt.saturating_sub(1).min(6);
    let factor = 1u64 << exponent;
    Duration::from_millis(DOWNLOAD_RETRY_BASE_DELAY_MS.saturating_mul(factor))
}

fn partial_download_path(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("download.bin");
    path.with_file_name(format!("{file_name}.partial"))
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = fs::File::open(path)
        .map_err(|err| format!("failed to open {} for checksum: {err}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|err| format!("failed to read {} for checksum: {err}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn format_mebibytes(bytes: u64) -> String {
    format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0))
}

fn write_file_atomically(path: &Path, bytes: &[u8]) -> Result<(), String> {
    ensure_parent_dir(path)?;
    let temp_path = temp_download_path(path);
    fs::write(&temp_path, bytes)
        .map_err(|err| format!("failed to write temp file {}: {err}", temp_path.display()))?;
    if path.exists() {
        fs::remove_file(path)
            .map_err(|err| format!("failed to replace stale file {}: {err}", path.display()))?;
    }
    fs::rename(&temp_path, path)
        .map_err(|err| format!("failed to move {} into place: {err}", path.display()))
}

fn ensure_parent_dir(path: &Path) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            format!(
                "failed to create parent directory {}: {err}",
                parent.display()
            )
        })
    } else {
        Ok(())
    }
}

fn default_cache_models_root() -> PathBuf {
    if let Some(home) = user_home_dir() {
        return home.join(CACHE_MODELS_DIR);
    }
    warn!(
        "HOME is not set; using relative model cache path {}",
        CACHE_MODELS_DIR
    );
    PathBuf::from(CACHE_MODELS_DIR)
}

fn user_home_dir() -> Option<PathBuf> {
    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        return Some(home);
    }
    #[cfg(target_os = "windows")]
    {
        if let Some(profile) = std::env::var_os("USERPROFILE").map(PathBuf::from) {
            return Some(profile);
        }
        let drive = std::env::var_os("HOMEDRIVE");
        let path = std::env::var_os("HOMEPATH");
        if let (Some(drive), Some(path)) = (drive, path) {
            return Some(PathBuf::from(format!(
                "{}{}",
                drive.to_string_lossy(),
                path.to_string_lossy()
            )));
        }
    }
    None
}

fn triposg_remote_root() -> String {
    option_env!("TRIPOSG_WEIGHTS_ROOT")
        .map(|value| value.to_string())
        .unwrap_or_else(|| format!("{}/{}", model_base_url(), TRIPOSG_DIR))
}

fn rmbg14_remote_root() -> String {
    option_env!("RMBG14_WEIGHTS_ROOT")
        .or(option_env!("RMBG_WEIGHTS_ROOT"))
        .map(|value| value.to_string())
        .unwrap_or_else(|| format!("{}/{}", model_base_url(), RMBG14_DIR))
}

fn model_base_url() -> String {
    option_env!("MODEL_BASE_URL")
        .unwrap_or(DEFAULT_MODEL_BASE_URL)
        .trim_end_matches('/')
        .to_string()
}

fn join_url(root: &str, rel: &str) -> String {
    let mut out = root.trim_end_matches('/').to_string();
    out.push('/');
    out.push_str(rel.trim_start_matches('/'));
    out
}

fn resolve_manifest_entry_url(manifest_url: &str, entry_url: &str) -> String {
    if entry_url.contains("://") || entry_url.starts_with('/') {
        return entry_url.to_string();
    }
    let normalized = entry_url.replace('\\', "/");
    if let Some((parent, _)) = manifest_url.rsplit_once('/') {
        return format!("{}/{}", parent.trim_end_matches('/'), normalized);
    }
    normalized
}

fn temp_download_path(path: &Path) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_nanos())
        .unwrap_or(0);
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("download.bin");
    path.with_file_name(format!("{file_name}.download-{nanos}.tmp"))
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use burn_synth_import::parts::{BurnpackPartEntry, BurnpackPartsManifest};

    use super::{manifest_is_complete, resolve_manifest_entry_url};

    fn unique_tmp_dir() -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("burn_synth_native_bootstrap_test_{nanos}"))
    }

    #[test]
    fn manifest_url_resolution_handles_relative_and_absolute_entries() {
        let manifest_url = "https://aberration.technology/model/MIDI-3D/vae/model.bpk.parts.json";
        assert_eq!(
            resolve_manifest_entry_url(manifest_url, "model.bpk.part-00000.bpk"),
            "https://aberration.technology/model/MIDI-3D/vae/model.bpk.part-00000.bpk"
        );
        assert_eq!(
            resolve_manifest_entry_url(manifest_url, "https://cdn.example/model.part"),
            "https://cdn.example/model.part"
        );
    }

    #[test]
    fn manifest_complete_requires_all_parts() {
        let root = unique_tmp_dir();
        fs::create_dir_all(&root).expect("create temp root");
        let manifest_path = root.join("model.bpk.parts.json");
        let part_path = root.join("model.bpk.part-00000.bpk");
        fs::write(&part_path, b"abc").expect("write part");

        let manifest = BurnpackPartsManifest {
            version: 1,
            source_file: "model.bpk".to_string(),
            source_modified_unix_ms: 0,
            total_bytes: 3,
            max_part_bytes: 3,
            parts: vec![BurnpackPartEntry {
                path: "model.bpk.part-00000.bpk".to_string(),
                bytes: 3,
                sha256: String::new(),
                tensors: 1,
            }],
        };
        fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).expect("serialize manifest"),
        )
        .expect("write manifest");
        assert!(
            manifest_is_complete(&manifest_path).expect("check complete"),
            "expected manifest to be complete"
        );

        fs::remove_file(&part_path).expect("remove part");
        assert!(
            !manifest_is_complete(&manifest_path).expect("check incomplete"),
            "expected manifest to be incomplete"
        );

        fs::remove_dir_all(root).expect("cleanup temp root");
    }
}

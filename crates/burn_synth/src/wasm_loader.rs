use std::path::Path;

use js_sys::{Date, Reflect, Uint8Array};
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use web_sys::{ReadableStreamDefaultReader, Response};

const ONE_GIB: u64 = 1024 * 1024 * 1024;
const DEFAULT_WEB_MAX_HOST_RAM_BYTES: u64 = 4 * ONE_GIB;

#[derive(Debug, Clone)]
pub struct WasmHostMemoryBudget {
    limit_bytes: u64,
    retained_bytes: u64,
    peak_bytes: u64,
}

impl WasmHostMemoryBudget {
    pub fn new(limit_bytes: u64) -> Self {
        Self {
            limit_bytes: limit_bytes.max(1),
            retained_bytes: 0,
            peak_bytes: 0,
        }
    }

    pub fn reserve_retained(&mut self, bytes: u64, context: &str) -> Result<(), String> {
        self.retained_bytes = self.retained_bytes.saturating_add(bytes);
        self.observe_total(self.retained_bytes, context)
    }

    pub fn release_retained(&mut self, bytes: u64) {
        self.retained_bytes = self.retained_bytes.saturating_sub(bytes);
    }

    pub fn observe_temporary(&mut self, temporary_bytes: u64, context: &str) -> Result<(), String> {
        let total = self.retained_bytes.saturating_add(temporary_bytes);
        self.observe_total(total, context)
    }

    fn observe_total(&mut self, total_bytes: u64, context: &str) -> Result<(), String> {
        self.peak_bytes = self.peak_bytes.max(total_bytes);
        if total_bytes > self.limit_bytes {
            return Err(format!(
                "wasm host RAM budget exceeded while {context}: {} used (limit {}). \
set build-time BURN_SYNTH_WEB_MAX_HOST_RAM_BYTES to raise",
                format_mebibytes(total_bytes),
                format_mebibytes(self.limit_bytes)
            ));
        }
        Ok(())
    }

    pub fn peak_bytes(&self) -> u64 {
        self.peak_bytes
    }

    pub fn limit_bytes(&self) -> u64 {
        self.limit_bytes
    }
}

#[derive(Default, Debug, Clone)]
pub struct DownloadTotals {
    pub known_total: u64,
    pub known_downloaded: u64,
    pub unknown_downloaded: u64,
}

pub fn normalize_web_path(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

pub fn join_web_path(root: &str, rel: &str) -> String {
    let mut out = root.trim_end_matches('/').to_string();
    out.push('/');
    out.push_str(rel.trim_start_matches('/'));
    out
}

pub fn format_mebibytes(bytes: u64) -> String {
    format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0))
}

pub fn web_max_host_ram_bytes() -> u64 {
    option_env!("BURN_SYNTH_WEB_MAX_HOST_RAM_BYTES")
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_WEB_MAX_HOST_RAM_BYTES)
}

pub fn web_max_burnpack_bytes() -> u64 {
    const DEFAULT_MAX_BPK_BYTES: u64 = ONE_GIB;
    option_env!("BURN_SYNTH_WEB_MAX_BPK_BYTES")
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_MAX_BPK_BYTES)
}

fn burnpack_too_large_error(url: &str, bytes: u64, limit_bytes: u64) -> String {
    format!(
        "burnpack at {url} is {} which exceeds browser limit {} (set build-time BURN_SYNTH_WEB_MAX_BPK_BYTES to raise)",
        format_mebibytes(bytes),
        format_mebibytes(limit_bytes),
    )
}

pub async fn fetch_optional_text(url: &str) -> Result<Option<String>, String> {
    match fetch_text(url).await {
        Ok(text) => Ok(Some(text)),
        Err(err) if err.contains("HTTP 404") => Ok(None),
        Err(err) => Err(err),
    }
}

pub async fn fetch_optional_text_candidates(urls: &[String]) -> Result<Option<String>, String> {
    for url in urls {
        match fetch_text(url).await {
            Ok(text) => return Ok(Some(text)),
            Err(err) if err.contains("HTTP 404") => continue,
            Err(err) => return Err(err),
        }
    }
    Ok(None)
}

async fn fetch_text(url: &str) -> Result<String, String> {
    let window = web_sys::window().ok_or_else(|| "window is unavailable".to_string())?;
    let response_value = JsFuture::from(window.fetch_with_str(url))
        .await
        .map_err(|err| format!("fetch failed for {url}: {err:?}"))?;
    let response: Response = response_value
        .dyn_into()
        .map_err(|_| format!("invalid response object for {url}"))?;
    if !response.ok() {
        return Err(format!("HTTP {} for {}", response.status(), url));
    }
    let text_promise = response
        .text()
        .map_err(|err| format!("failed to read text for {url}: {err:?}"))?;
    let text_value = JsFuture::from(text_promise)
        .await
        .map_err(|err| format!("failed to await text for {url}: {err:?}"))?;
    Ok(text_value.as_string().unwrap_or_default())
}

pub async fn download_binary_with_status<F>(
    url: &str,
    label: &str,
    max_bytes: u64,
    totals: &mut DownloadTotals,
    host_ram_budget: &mut WasmHostMemoryBudget,
    on_status: &mut F,
) -> Result<Vec<u8>, String>
where
    F: FnMut(String),
{
    const STATUS_MIN_INTERVAL_MS: f64 = 250.0;
    const STATUS_MIN_PROGRESS_BYTES: u64 = 4 * 1024 * 1024;

    let mut registered_total = false;
    let mut prev = 0u64;
    let mut last_status_bytes = 0u64;
    let mut last_status_ms = 0.0f64;
    let bytes = fetch_binary_with_progress(url, max_bytes, |loaded, total| {
        if let Some(total_bytes) = total
            && !registered_total
        {
            totals.known_total = totals.known_total.saturating_add(total_bytes);
            registered_total = true;
        }
        if registered_total {
            let delta = loaded.saturating_sub(prev);
            totals.known_downloaded = totals.known_downloaded.saturating_add(delta);
        } else {
            let delta = loaded.saturating_sub(prev);
            totals.unknown_downloaded = totals.unknown_downloaded.saturating_add(delta);
        }
        prev = loaded;

        let now_ms = Date::now();
        let reached_end = total.is_some_and(|total_bytes| loaded >= total_bytes);
        let should_emit = reached_end
            || loaded.saturating_sub(last_status_bytes) >= STATUS_MIN_PROGRESS_BYTES
            || (now_ms - last_status_ms) >= STATUS_MIN_INTERVAL_MS;
        if !should_emit {
            return;
        }
        last_status_bytes = loaded;
        last_status_ms = now_ms;

        let message = if totals.known_total > 0 {
            let percent = (totals.known_downloaded as f64 / totals.known_total as f64) * 100.0;
            format!(
                "Loading {label}... {percent:.1}% ({}/{})",
                format_mebibytes(totals.known_downloaded),
                format_mebibytes(totals.known_total)
            )
        } else {
            format!(
                "Loading {label}... {} downloaded",
                format_mebibytes(totals.unknown_downloaded)
            )
        };
        on_status(message);
    })
    .await?;
    host_ram_budget.observe_temporary(bytes.len() as u64, &format!("downloading {label}"))?;
    Ok(bytes)
}

async fn fetch_binary_with_progress<F>(
    url: &str,
    max_bytes: u64,
    mut on_progress: F,
) -> Result<Vec<u8>, String>
where
    F: FnMut(u64, Option<u64>),
{
    let window = web_sys::window().ok_or_else(|| "window is unavailable".to_string())?;
    let response_value = JsFuture::from(window.fetch_with_str(url))
        .await
        .map_err(|err| format!("fetch failed for {url}: {err:?}"))?;
    let response: Response = response_value
        .dyn_into()
        .map_err(|_| format!("invalid response object for {url}"))?;
    if !response.ok() {
        return Err(format!("HTTP {} for {}", response.status(), url));
    }

    let total = response
        .headers()
        .get("content-length")
        .ok()
        .flatten()
        .and_then(|value| value.parse::<u64>().ok());
    if let Some(total_bytes) = total
        && total_bytes > max_bytes
    {
        return Err(burnpack_too_large_error(url, total_bytes, max_bytes));
    }

    // Fast path for bounded responses: array_buffer avoids stream chunk plumbing overhead.
    // For unknown response sizes, prefer stream mode so max_bytes is enforced incrementally.
    let mut tried_array_buffer = false;
    if total.is_some() {
        tried_array_buffer = true;
        let buffer_promise = response
            .array_buffer()
            .map_err(|err| format!("failed to start array_buffer for {url}: {err:?}"))?;
        let buffer_result = JsFuture::from(buffer_promise).await;
        if let Ok(buffer) = buffer_result {
            let bytes = Uint8Array::new(&buffer);
            if (bytes.length() as u64) > max_bytes {
                return Err(burnpack_too_large_error(
                    url,
                    bytes.length() as u64,
                    max_bytes,
                ));
            }
            let mut output = vec![0u8; bytes.length() as usize];
            bytes.copy_to(&mut output);
            on_progress(output.len() as u64, total);
            return Ok(output);
        }
    }

    if let Some(body) = response.body() {
        let reader: ReadableStreamDefaultReader = body
            .get_reader()
            .dyn_into()
            .map_err(|_| format!("failed to create stream reader for {url}"))?;

        let mut output = Vec::new();
        let mut loaded = 0u64;
        loop {
            let chunk = JsFuture::from(reader.read())
                .await
                .map_err(|err| format!("stream read failed for {url}: {err:?}"))?;
            let done = Reflect::get(&chunk, &"done".into())
                .ok()
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            if done {
                break;
            }
            let value = Reflect::get(&chunk, &"value".into())
                .map_err(|_| format!("missing stream chunk value for {url}"))?;
            let chunk_bytes = Uint8Array::new(&value);
            let len = chunk_bytes.length() as usize;
            if loaded.saturating_add(len as u64) > max_bytes {
                return Err(burnpack_too_large_error(
                    url,
                    loaded.saturating_add(len as u64),
                    max_bytes,
                ));
            }
            let old_len = output.len();
            output.resize(old_len + len, 0);
            chunk_bytes.copy_to(&mut output[old_len..]);
            loaded = loaded.saturating_add(len as u64);
            on_progress(loaded, total);
        }
        on_progress(loaded, total);
        return Ok(output);
    }

    if !tried_array_buffer {
        let buffer_promise = response
            .array_buffer()
            .map_err(|err| format!("failed to start array_buffer for {url}: {err:?}"))?;
        let buffer = JsFuture::from(buffer_promise)
            .await
            .map_err(|err| format!("failed to await array_buffer for {url}: {err:?}"))?;
        let bytes = Uint8Array::new(&buffer);
        if (bytes.length() as u64) > max_bytes {
            return Err(burnpack_too_large_error(
                url,
                bytes.length() as u64,
                max_bytes,
            ));
        }
        let mut output = vec![0u8; bytes.length() as usize];
        bytes.copy_to(&mut output);
        on_progress(output.len() as u64, total);
        return Ok(output);
    }

    Err(format!(
        "failed to download {url}: neither array_buffer nor stream succeeded"
    ))
}

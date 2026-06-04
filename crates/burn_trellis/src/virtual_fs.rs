use std::path::Path;
#[cfg(target_arch = "wasm32")]
use std::path::PathBuf;

#[cfg(target_arch = "wasm32")]
use std::cell::RefCell;
#[cfg(target_arch = "wasm32")]
use std::collections::HashMap;

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Debug, Default)]
struct VirtualEntry {
    bytes: Option<Vec<u8>>,
    source_url: Option<String>,
}

#[cfg(target_arch = "wasm32")]
thread_local! {
    static VIRTUAL_FILES: RefCell<HashMap<PathBuf, VirtualEntry>> = RefCell::new(HashMap::new());
}

#[cfg(target_arch = "wasm32")]
fn has_virtual_descendant(path: &Path) -> bool {
    VIRTUAL_FILES.with(|files| {
        files
            .borrow()
            .keys()
            .any(|candidate| candidate.starts_with(path) && candidate != path)
    })
}

#[cfg(target_arch = "wasm32")]
pub fn clear_virtual_files() {
    VIRTUAL_FILES.with(|files| files.borrow_mut().clear());
}

#[cfg(not(target_arch = "wasm32"))]
pub fn clear_virtual_files() {}

#[cfg(target_arch = "wasm32")]
pub fn register_virtual_file(path: impl AsRef<Path>, bytes: Vec<u8>) {
    VIRTUAL_FILES.with(|files| {
        files.borrow_mut().insert(
            path.as_ref().to_path_buf(),
            VirtualEntry {
                bytes: Some(bytes),
                source_url: None,
            },
        );
    });
}

#[cfg(not(target_arch = "wasm32"))]
pub fn register_virtual_file(_path: impl AsRef<Path>, _bytes: Vec<u8>) {}

#[cfg(target_arch = "wasm32")]
pub fn register_virtual_url(path: impl AsRef<Path>, source_url: impl Into<String>) {
    VIRTUAL_FILES.with(|files| {
        files.borrow_mut().insert(
            path.as_ref().to_path_buf(),
            VirtualEntry {
                bytes: None,
                source_url: Some(source_url.into()),
            },
        );
    });
}

#[cfg(not(target_arch = "wasm32"))]
pub fn register_virtual_url(_path: impl AsRef<Path>, _source_url: impl Into<String>) {}

#[cfg(target_arch = "wasm32")]
pub fn has_virtual_file(path: &Path) -> bool {
    VIRTUAL_FILES.with(|files| files.borrow().contains_key(path))
}

#[cfg(not(target_arch = "wasm32"))]
pub fn has_virtual_file(_path: &Path) -> bool {
    false
}

#[cfg(target_arch = "wasm32")]
pub fn source_url(path: &Path) -> Option<String> {
    VIRTUAL_FILES.with(|files| {
        files
            .borrow()
            .get(path)
            .and_then(|entry| entry.source_url.clone())
    })
}

#[cfg(not(target_arch = "wasm32"))]
pub fn source_url(_path: &Path) -> Option<String> {
    None
}

pub fn exists(path: &Path) -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        if has_virtual_file(path) || has_virtual_descendant(path) {
            return true;
        }
    }
    path.exists()
}

pub fn is_file(path: &Path) -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        if has_virtual_file(path) {
            return true;
        }
    }
    path.is_file()
}

pub fn is_dir(path: &Path) -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        if has_virtual_descendant(path) {
            return true;
        }
    }
    path.is_dir()
}

#[cfg(target_arch = "wasm32")]
fn io_error(message: impl Into<String>) -> std::io::Error {
    std::io::Error::other(message.into())
}

#[cfg(target_arch = "wasm32")]
pub fn fetch_url(source_url: &str) -> Result<Vec<u8>, std::io::Error> {
    Err(io_error(format!(
        "virtual_fs URL fetch is intentionally disabled on wasm for '{}'; preload bytes asynchronously before registering virtual files",
        source_url
    )))
}

#[cfg(not(target_arch = "wasm32"))]
pub fn fetch_url(source_url: &str) -> Result<Vec<u8>, std::io::Error> {
    Err(std::io::Error::other(format!(
        "virtual_fs URL fetch is unavailable on non-wasm targets ({source_url})"
    )))
}

pub fn read(path: &Path) -> Result<Vec<u8>, std::io::Error> {
    #[cfg(target_arch = "wasm32")]
    {
        let entry = VIRTUAL_FILES.with(|files| files.borrow().get(path).cloned());
        if let Some(entry) = entry {
            if let Some(bytes) = entry.bytes {
                return Ok(bytes);
            }
            if let Some(url) = entry.source_url {
                return fetch_url(&url);
            }
            return Err(io_error(format!(
                "virtual file entry for '{}' has no bytes/url",
                path.display()
            )));
        }
    }
    std::fs::read(path)
}

pub fn read_to_string(path: &Path) -> Result<String, std::io::Error> {
    let bytes = read(path)?;
    String::from_utf8(bytes)
        .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err.to_string()))
}

pub fn metadata_len(path: &Path) -> Result<u64, std::io::Error> {
    #[cfg(target_arch = "wasm32")]
    {
        let entry = VIRTUAL_FILES.with(|files| files.borrow().get(path).cloned());
        if let Some(entry) = entry {
            if let Some(bytes) = entry.bytes {
                return Ok(bytes.len() as u64);
            }
            if let Some(url) = entry.source_url {
                let bytes = fetch_url(&url)?;
                return Ok(bytes.len() as u64);
            }
        }
    }
    Ok(std::fs::metadata(path)?.len())
}

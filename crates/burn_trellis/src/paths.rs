use std::path::{Path, PathBuf};

#[cfg(not(target_arch = "wasm32"))]
pub fn resolve_trellis2_weights_root(explicit: Option<&Path>) -> PathBuf {
    if let Some(path) = explicit
        && let Some(root) = normalize_root(path)
    {
        return root;
    }
    if let Ok(root) = std::env::var("TRELLIS2_WEIGHTS_ROOT") {
        let path = PathBuf::from(root);
        if let Some(root) = normalize_root(&path) {
            return root;
        }
    }
    let local = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/models/TRELLIS.2-4B");
    normalize_root(&local).unwrap_or(local)
}

#[cfg(target_arch = "wasm32")]
pub fn resolve_trellis2_weights_root(explicit: Option<&Path>) -> PathBuf {
    if let Some(path) = explicit {
        return path.to_path_buf();
    }
    if let Ok(root) = std::env::var("TRELLIS2_WEIGHTS_ROOT") {
        return PathBuf::from(root);
    }
    if let Some(root) = option_env!("TRELLIS2_WEIGHTS_ROOT") {
        return PathBuf::from(root);
    }
    web_asset_root().join("models/TRELLIS.2-4B")
}

#[cfg(not(target_arch = "wasm32"))]
pub fn resolve_trellis2_image_large_root(explicit: Option<&Path>) -> PathBuf {
    if let Some(path) = explicit
        && let Some(root) = normalize_root(path)
    {
        return root;
    }
    if let Ok(root) = std::env::var("TRELLIS2_IMAGE_LARGE_ROOT") {
        let path = PathBuf::from(root);
        if let Some(root) = normalize_root(&path) {
            return root;
        }
    }
    let local = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/models/TRELLIS-image-large");
    normalize_root(&local).unwrap_or(local)
}

#[cfg(target_arch = "wasm32")]
pub fn resolve_trellis2_image_large_root(explicit: Option<&Path>) -> PathBuf {
    if let Some(path) = explicit {
        return path.to_path_buf();
    }
    if let Ok(root) = std::env::var("TRELLIS2_IMAGE_LARGE_ROOT") {
        return PathBuf::from(root);
    }
    if let Some(root) = option_env!("TRELLIS2_IMAGE_LARGE_ROOT") {
        return PathBuf::from(root);
    }
    web_asset_root().join("models/TRELLIS-image-large")
}

#[cfg(not(target_arch = "wasm32"))]
fn normalize_root(path: &Path) -> Option<PathBuf> {
    if path.is_dir() {
        return Some(path.to_path_buf());
    }
    if path.is_file() {
        return path.parent().map(Path::to_path_buf);
    }
    None
}

#[cfg(target_arch = "wasm32")]
fn web_asset_root() -> PathBuf {
    if let Some(root) = option_env!("BURN_SYNTH_WEB_ASSET_ROOT") {
        let trimmed = root.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
    PathBuf::from("assets")
}

#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
use std::path::PathBuf;

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn resolve_triposg_root(explicit: Option<&PathBuf>) -> PathBuf {
    if let Some(path) = explicit
        && let Some(root) = normalize_weights_root(path)
    {
        return root;
    }
    if let Ok(root) = std::env::var("TRIPOSG_WEIGHTS_ROOT") {
        let path = PathBuf::from(root);
        if let Some(root) = normalize_weights_root(&path) {
            return root;
        }
    }
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let local = manifest_dir.join("../burn_3d_synth_tripo/assets/models/MIDI-3D");
    normalize_weights_root(&local).unwrap_or(local)
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn resolve_triposg_root(explicit: Option<&PathBuf>) -> PathBuf {
    if let Some(path) = explicit {
        return path.to_path_buf();
    }
    if let Some(root) = option_env!("TRIPOSG_WEIGHTS_ROOT") {
        return PathBuf::from(root);
    }
    web_asset_root().join("models/MIDI-3D")
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn resolve_rmbg_root(explicit: Option<&PathBuf>) -> PathBuf {
    if let Some(path) = explicit
        && let Some(root) = normalize_rmbg_root(path)
    {
        return root;
    }
    if let Ok(root) = std::env::var("RMBG_WEIGHTS_ROOT") {
        let path = PathBuf::from(root);
        if let Some(root) = normalize_rmbg_root(&path) {
            return root;
        }
    }
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let local = manifest_dir.join("../burn_3d_synth_tripo/assets/models/RMBG-1.4");
    normalize_rmbg_root(&local).unwrap_or(local)
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn resolve_rmbg_root(explicit: Option<&PathBuf>) -> PathBuf {
    if let Some(path) = explicit {
        return path.to_path_buf();
    }
    if let Some(root) = option_env!("RMBG_WEIGHTS_ROOT") {
        return PathBuf::from(root);
    }
    web_asset_root().join("models/RMBG-1.4")
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn resolve_scribble_root(explicit: Option<&PathBuf>) -> PathBuf {
    if let Some(path) = explicit
        && let Some(root) = normalize_weights_root(path)
    {
        return root;
    }
    if let Ok(root) = std::env::var("TRIPOSG_SCRIBBLE_WEIGHTS_ROOT") {
        let path = PathBuf::from(root);
        if let Some(root) = normalize_weights_root(&path) {
            return root;
        }
    }
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let local = manifest_dir.join("../burn_3d_synth_tripo/assets/models/TripoSG-scribble");
    normalize_weights_root(&local).unwrap_or(local)
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn resolve_scribble_root(explicit: Option<&PathBuf>) -> PathBuf {
    if let Some(path) = explicit {
        return path.to_path_buf();
    }
    if let Some(root) = option_env!("TRIPOSG_SCRIBBLE_WEIGHTS_ROOT") {
        return PathBuf::from(root);
    }
    web_asset_root().join("models/TripoSG-scribble")
}

#[cfg(not(target_arch = "wasm32"))]
fn normalize_weights_root(path: &Path) -> Option<PathBuf> {
    if path.is_dir() {
        return Some(path.to_path_buf());
    }
    if path.is_file()
        && let Some(parent) = path.parent().and_then(|p| p.parent())
    {
        return Some(parent.to_path_buf());
    }
    None
}

#[cfg(not(target_arch = "wasm32"))]
fn normalize_rmbg_root(path: &Path) -> Option<PathBuf> {
    if path.is_dir() {
        return Some(path.to_path_buf());
    }
    if path.is_file() && path.file_name().and_then(|n| n.to_str()) == Some("model.safetensors") {
        return path.parent().map(|p| p.to_path_buf());
    }
    None
}

#[cfg(target_arch = "wasm32")]
fn web_asset_root() -> PathBuf {
    if let Some(root) = option_env!("BURN_3D_SYNTH_WEB_ASSET_ROOT") {
        let trimmed = root.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
    PathBuf::from("assets")
}

#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
use std::path::PathBuf;

use crate::args::RmbgModel;

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn resolve_triposg_root(explicit: Option<&PathBuf>) -> PathBuf {
    burn_tripo::paths::resolve_triposg_weights_root(explicit.map(|path| path.as_path()))
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn resolve_triposg_root(explicit: Option<&PathBuf>) -> PathBuf {
    if let Some(path) = explicit {
        return path.to_path_buf();
    }
    web_asset_root().join("models/MIDI-3D")
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn resolve_rmbg_root(explicit: Option<&PathBuf>, model: RmbgModel) -> PathBuf {
    if let Some(path) = explicit
        && let Some(root) = normalize_rmbg_root(path)
    {
        return root;
    }
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let model_dir = match model {
        RmbgModel::Rmbg14 => "RMBG-1.4",
        RmbgModel::Rmbg2 => "RMBG-2.0",
    };
    let foreground_local =
        manifest_dir.join(format!("../burn_foreground/assets/models/{model_dir}"));
    if let Some(root) = normalize_rmbg_root(&foreground_local) {
        return root;
    }
    foreground_local
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn resolve_rmbg_root(explicit: Option<&PathBuf>, model: RmbgModel) -> PathBuf {
    if let Some(path) = explicit {
        return path.to_path_buf();
    }
    let model_dir = match model {
        RmbgModel::Rmbg14 => "RMBG-1.4",
        RmbgModel::Rmbg2 => "RMBG-2.0",
    };
    web_asset_root().join(format!("models/{model_dir}"))
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn resolve_scribble_root(explicit: Option<&PathBuf>) -> PathBuf {
    if let Some(path) = explicit
        && let Some(root) = normalize_weights_root(path)
    {
        return root;
    }
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let local = manifest_dir.join("../burn_tripo/assets/models/TripoSG-scribble");
    normalize_weights_root(&local).unwrap_or(local)
}

#[cfg(target_arch = "wasm32")]
#[allow(dead_code)]
pub(crate) fn resolve_scribble_root(explicit: Option<&PathBuf>) -> PathBuf {
    if let Some(path) = explicit {
        return path.to_path_buf();
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
    if path.is_file() {
        let file_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();
        if file_name == "model.safetensors" || file_name.ends_with(".bpk") {
            return path.parent().map(|p| p.to_path_buf());
        }
        if file_name.ends_with(".onnx") {
            let parent = path.parent()?;
            if parent.file_name().and_then(|n| n.to_str()) == Some("onnx") {
                return parent.parent().map(|p| p.to_path_buf());
            }
            return Some(parent.to_path_buf());
        }
    }
    None
}

#[cfg(target_arch = "wasm32")]
fn web_asset_root() -> PathBuf {
    PathBuf::from("assets")
}

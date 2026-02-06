use std::path::{Path, PathBuf};

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
    let fallback = PathBuf::from(r"E:\repos\TripoSG\pretrained_weights\TripoSG");
    if let Some(root) = normalize_weights_root(&fallback) {
        return root;
    }
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let local = manifest_dir.join("../burn_3d_synth_tripo/assets/models/MIDI-3D");
    normalize_weights_root(&local).unwrap_or(local)
}

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
    let fallback = PathBuf::from(r"E:\repos\TripoSG\pretrained_weights\RMBG-1.4");
    if let Some(root) = normalize_rmbg_root(&fallback) {
        return root;
    }
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let local = manifest_dir.join("../burn_3d_synth_bg_removal/assets/models/RMBG-1.4");
    normalize_rmbg_root(&local).unwrap_or(local)
}

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
    let fallback = PathBuf::from(r"E:\repos\TripoSG\pretrained_weights\TripoSG-scribble");
    if let Some(root) = normalize_weights_root(&fallback) {
        return root;
    }
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let local = manifest_dir.join("../burn_3d_synth_tripo/assets/models/TripoSG-scribble");
    normalize_weights_root(&local).unwrap_or(local)
}

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

fn normalize_rmbg_root(path: &Path) -> Option<PathBuf> {
    if path.is_dir() {
        return Some(path.to_path_buf());
    }
    if path.is_file() && path.file_name().and_then(|n| n.to_str()) == Some("model.safetensors") {
        return path.parent().map(|p| p.to_path_buf());
    }
    None
}

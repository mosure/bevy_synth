use std::path::{Path, PathBuf};

pub fn resolve_triposg_weights_root(explicit: Option<&Path>) -> PathBuf {
    if let Some(path) = explicit
        && let Some(root) = normalize_weights_root(path)
    {
        return root;
    }
    let local = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/models/MIDI-3D");
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

use std::path::{Path, PathBuf};

#[cfg(not(target_arch = "wasm32"))]
const TRELLIS2_WEIGHTS_ROOT_REPO_RELATIVE: &str = "crates/burn_trellis/assets/models/TRELLIS.2-4B";
#[cfg(not(target_arch = "wasm32"))]
const TRELLIS2_WEIGHTS_ROOT_CRATE_RELATIVE: &str = "assets/models/TRELLIS.2-4B";
#[cfg(not(target_arch = "wasm32"))]
const TRELLIS2_IMAGE_LARGE_ROOT_REPO_RELATIVE: &str =
    "crates/burn_trellis/assets/models/TRELLIS-image-large";
#[cfg(not(target_arch = "wasm32"))]
const TRELLIS2_IMAGE_LARGE_ROOT_CRATE_RELATIVE: &str = "assets/models/TRELLIS-image-large";

#[cfg(not(target_arch = "wasm32"))]
pub fn trellis2_repo_asset_root() -> PathBuf {
    preferred_local_asset_root(
        TRELLIS2_WEIGHTS_ROOT_REPO_RELATIVE,
        TRELLIS2_WEIGHTS_ROOT_CRATE_RELATIVE,
    )
}

#[cfg(target_arch = "wasm32")]
pub fn trellis2_repo_asset_root() -> PathBuf {
    web_asset_root().join("models/TRELLIS.2-4B")
}

#[cfg(not(target_arch = "wasm32"))]
pub fn trellis2_repo_image_large_root() -> PathBuf {
    preferred_local_asset_root(
        TRELLIS2_IMAGE_LARGE_ROOT_REPO_RELATIVE,
        TRELLIS2_IMAGE_LARGE_ROOT_CRATE_RELATIVE,
    )
}

#[cfg(target_arch = "wasm32")]
pub fn trellis2_repo_image_large_root() -> PathBuf {
    web_asset_root().join("models/TRELLIS-image-large")
}

#[cfg(not(target_arch = "wasm32"))]
pub fn resolve_trellis2_weights_root(explicit: Option<&Path>) -> PathBuf {
    if let Some(path) = explicit
        && let Some(root) = normalize_root(path)
    {
        return root;
    }
    let local = trellis2_repo_asset_root();
    select_bpk_preferred_root(None, local)
}

#[cfg(target_arch = "wasm32")]
pub fn resolve_trellis2_weights_root(explicit: Option<&Path>) -> PathBuf {
    if let Some(path) = explicit {
        return path.to_path_buf();
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
    let image_local_root = trellis2_repo_image_large_root();
    let image_selected = select_bpk_preferred_root(None, image_local_root);
    if root_contains_bpk(image_selected.as_path()) {
        return image_selected;
    }

    // Keep runtime/import functional even when TRELLIS-image-large is not materialized as a
    // separate directory by falling back to the main TRELLIS.2-4B root.
    let weights_local_root = trellis2_repo_asset_root();
    let weights_selected = select_bpk_preferred_root(None, weights_local_root);
    if root_contains_bpk(weights_selected.as_path()) {
        return weights_selected;
    }

    image_selected
}

#[cfg(not(target_arch = "wasm32"))]
pub fn trellis2_ovoxel_postprocess_script_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tool/trellis2_postprocess_from_hook.py")
}

#[cfg(target_arch = "wasm32")]
pub fn resolve_trellis2_image_large_root(explicit: Option<&Path>) -> PathBuf {
    if let Some(path) = explicit {
        return path.to_path_buf();
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

#[cfg(not(target_arch = "wasm32"))]
fn preferred_local_asset_root(repo_relative: &str, crate_relative: &str) -> PathBuf {
    for candidate in [PathBuf::from(repo_relative), PathBuf::from(crate_relative)] {
        if let Some(root) = normalize_root(candidate.as_path()) {
            return root;
        }
    }
    let manifest_fallback = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(crate_relative);
    normalize_root(manifest_fallback.as_path()).unwrap_or_else(|| PathBuf::from(repo_relative))
}

#[cfg(not(target_arch = "wasm32"))]
fn select_bpk_preferred_root(primary: Option<PathBuf>, fallback: PathBuf) -> PathBuf {
    let fallback_has_bpk = root_contains_bpk(fallback.as_path());
    if let Some(primary) = primary {
        if root_contains_bpk(primary.as_path()) {
            return primary;
        }
        if fallback_has_bpk {
            return fallback;
        }
        return primary;
    }
    fallback
}

#[cfg(not(target_arch = "wasm32"))]
fn root_contains_bpk(root: &Path) -> bool {
    let ckpts = root.join("ckpts");
    let Ok(entries) = std::fs::read_dir(ckpts) else {
        return false;
    };
    entries.filter_map(Result::ok).any(|entry| {
        entry
            .path()
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("bpk"))
    })
}

#[cfg(target_arch = "wasm32")]
fn web_asset_root() -> PathBuf {
    PathBuf::from("assets")
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::select_bpk_preferred_root;

    use std::fs;
    use std::path::Path;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn make_temp_root(label: &str) -> std::path::PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock drift")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("burn_trellis_paths_{label}_{unique}"));
        fs::create_dir_all(&root).expect("create temp root");
        root
    }

    fn write_bpk(root: &Path, name: &str) {
        let ckpts = root.join("ckpts");
        fs::create_dir_all(&ckpts).expect("create ckpts");
        fs::write(ckpts.join(name), b"bpk").expect("write bpk");
    }

    #[test]
    fn select_root_prefers_fallback_when_primary_has_no_bpk() {
        let primary = make_temp_root("primary_no_bpk");
        let fallback = make_temp_root("fallback_with_bpk");
        write_bpk(&fallback, "model_f16.bpk");

        let selected = select_bpk_preferred_root(Some(primary.clone()), fallback.clone());
        assert_eq!(selected, fallback);

        let _ = fs::remove_dir_all(primary);
        let _ = fs::remove_dir_all(fallback);
    }

    #[test]
    fn select_root_prefers_primary_when_primary_has_bpk() {
        let primary = make_temp_root("primary_with_bpk");
        let fallback = make_temp_root("fallback_with_bpk");
        write_bpk(&primary, "model_f16.bpk");
        write_bpk(&fallback, "model_f16.bpk");

        let selected = select_bpk_preferred_root(Some(primary.clone()), fallback.clone());
        assert_eq!(selected, primary);

        let _ = fs::remove_dir_all(primary);
        let _ = fs::remove_dir_all(fallback);
    }
}

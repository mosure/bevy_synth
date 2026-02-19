use burn_synth_import::layout::candidate_burnpack_names as shared_candidate_burnpack_names;
use burn_synth_import::parts::BurnpackPartsManifest;

/// Default artifact precision preference for generic model loader paths.
///
/// Note: do not use this to decide TripoSG runtime precision in parity-critical paths.
/// Use `burn_tripo::pipeline::runtime_parity::should_prefer_f16_triposg_weights` instead.
#[cfg(target_arch = "wasm32")]
pub fn prefer_f16_burnpack() -> bool {
    true
}

/// Default artifact precision preference for generic model loader paths.
///
/// Note: do not use this to decide TripoSG runtime precision in parity-critical paths.
/// Use `burn_tripo::pipeline::runtime_parity::should_prefer_f16_triposg_weights` instead.
#[cfg(not(target_arch = "wasm32"))]
pub fn prefer_f16_burnpack() -> bool {
    true
}

pub fn candidate_burnpack_names(base_safetensors_path: &str, prefer_f16: bool) -> Vec<String> {
    shared_candidate_burnpack_names(base_safetensors_path, prefer_f16)
}

pub fn parse_parts_manifest_bytes(
    manifest_bytes: &[u8],
    source: &str,
) -> Result<BurnpackPartsManifest, String> {
    serde_json::from_slice(manifest_bytes)
        .map_err(|err| format!("failed to parse burnpack parts manifest {source}: {err}"))
}

#[cfg(target_arch = "wasm32")]
pub fn resolve_manifest_entry_uri(manifest_uri: &str, entry_uri: &str) -> String {
    if entry_uri.contains("://") || entry_uri.starts_with('/') {
        return entry_uri.to_string();
    }
    let normalized = entry_uri.replace('\\', "/");
    if let Some((parent, _)) = manifest_uri.rsplit_once('/') {
        return format!("{}/{}", parent.trim_end_matches('/'), normalized);
    }
    normalized
}

#[cfg(not(target_arch = "wasm32"))]
use std::fs;
#[cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};

#[cfg(not(target_arch = "wasm32"))]
pub fn load_optional_text_from_root(root: &Path, rel: &str) -> Result<Option<String>, String> {
    let path = root.join(rel);
    if !path.exists() {
        return Ok(None);
    }
    fs::read_to_string(&path)
        .map(Some)
        .map_err(|err| format!("failed to read {}: {err}", path.display()))
}

#[cfg(not(target_arch = "wasm32"))]
pub fn load_optional_text_candidates_from_root(
    root: &Path,
    rel_paths: &[&str],
) -> Result<Option<String>, String> {
    for rel in rel_paths {
        if let Some(contents) = load_optional_text_from_root(root, rel)? {
            return Ok(Some(contents));
        }
    }
    Ok(None)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn resolve_burnpack_asset_path_from_root(
    root: &Path,
    base_safetensors_rel: &str,
) -> Result<PathBuf, String> {
    resolve_burnpack_asset_path_from_root_with_preference(
        root,
        base_safetensors_rel,
        prefer_f16_burnpack(),
    )
}

#[cfg(not(target_arch = "wasm32"))]
pub fn resolve_burnpack_asset_path_from_root_with_preference(
    root: &Path,
    base_safetensors_rel: &str,
    prefer_f16: bool,
) -> Result<PathBuf, String> {
    let candidates = candidate_burnpack_names(base_safetensors_rel, prefer_f16);
    let mut checked = Vec::new();
    for candidate in candidates {
        let candidate_path = root.join(Path::new(&candidate));
        checked.push(candidate_path.display().to_string());
        if candidate_path.exists() {
            return Ok(candidate_path);
        }
    }

    Err(format!(
        "failed to locate burnpack under '{}'; checked: {}",
        root.display(),
        checked.join(", "),
    ))
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{
        candidate_burnpack_names, prefer_f16_burnpack, resolve_burnpack_asset_path_from_root,
        resolve_burnpack_asset_path_from_root_with_preference,
    };

    fn unique_temp_dir() -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("burn_synth_loader_test_{nanos}"))
    }

    #[test]
    fn candidate_paths_respect_precision_preference() {
        let f16_first = candidate_burnpack_names("model.safetensors", true);
        assert_eq!(f16_first, vec!["model_f16.bpk", "model.bpk"]);

        let f32_first = candidate_burnpack_names("model.safetensors", false);
        assert_eq!(f32_first, vec!["model.bpk", "model_f16.bpk"]);
    }

    #[test]
    fn prefer_f16_default_is_true() {
        assert!(prefer_f16_burnpack());
    }

    #[test]
    fn resolve_prefers_f16_when_present() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).expect("failed to create temp root");
        fs::write(root.join("model_f16.bpk"), b"f16").expect("failed to write f16 burnpack");
        fs::write(root.join("model.bpk"), b"f32").expect("failed to write f32 burnpack");

        let resolved =
            resolve_burnpack_asset_path_from_root_with_preference(&root, "model.safetensors", true)
                .expect("failed to resolve burnpack");
        assert_eq!(resolved, root.join("model_f16.bpk"));

        fs::remove_dir_all(root).expect("failed to cleanup temp root");
    }

    #[test]
    fn resolve_falls_back_to_f32_when_f16_missing() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).expect("failed to create temp root");
        fs::write(root.join("model.bpk"), b"f32").expect("failed to write f32 burnpack");

        let resolved =
            resolve_burnpack_asset_path_from_root_with_preference(&root, "model.safetensors", true)
                .expect("failed to resolve burnpack");
        assert_eq!(resolved, root.join("model.bpk"));

        fs::remove_dir_all(root).expect("failed to cleanup temp root");
    }

    #[test]
    fn resolve_errors_when_burnpack_missing() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).expect("failed to create temp root");

        let err = resolve_burnpack_asset_path_from_root(&root, "model.safetensors")
            .expect_err("expected missing burnpack error");
        assert!(
            err.contains("model_f16.bpk") && err.contains("model.bpk"),
            "unexpected error message: {err}"
        );

        fs::remove_dir_all(root).expect("failed to cleanup temp root");
    }
}

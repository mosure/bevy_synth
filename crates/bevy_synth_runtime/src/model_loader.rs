use std::env;

use burn_synth_import::layout::{
    burnpack_manifest_candidates as shared_manifest_candidates,
    candidate_burnpack_names as shared_candidate_burnpack_names,
};
use burn_synth_import::shard::BurnpackShardManifest;

pub(crate) fn prefer_f16_burnpack(primary: &str) -> bool {
    let value = env::var(primary)
        .ok()
        .or_else(|| env::var("BURN_SYNTH_BPK_PRECISION").ok());
    match value
        .as_deref()
        .map(str::trim)
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("f32" | "fp32" | "float32" | "32") => false,
        Some("f16" | "fp16" | "float16" | "half" | "16") => true,
        Some(_) | None => true,
    }
}

pub(crate) fn candidate_burnpack_names(
    base_safetensors_path: &str,
    prefer_f16: bool,
) -> Vec<String> {
    shared_candidate_burnpack_names(base_safetensors_path, prefer_f16)
}

pub(crate) fn burnpack_manifest_candidates(candidate_burnpack_path: &str) -> [String; 2] {
    shared_manifest_candidates(candidate_burnpack_path)
}

pub(crate) fn parse_shard_manifest_bytes(
    manifest_bytes: &[u8],
    source: &str,
) -> Result<BurnpackShardManifest, String> {
    serde_json::from_slice(manifest_bytes)
        .map_err(|err| format!("failed to parse shard manifest {source}: {err}"))
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn reconstruct_burnpack_from_shard_manifest<F>(
    manifest: &BurnpackShardManifest,
    mut load_shard: F,
) -> Result<Vec<u8>, String>
where
    F: FnMut(&str) -> Result<Vec<u8>, String>,
{
    let shards = manifest.shard_entries();
    if shards.is_empty() {
        return Err("shard manifest contains no shard entries".to_string());
    }

    let mut output = Vec::with_capacity(manifest.total_bytes as usize);
    for shard in shards {
        let bytes = load_shard(shard.path())?;
        output.extend_from_slice(&bytes);
    }

    if manifest.total_bytes > 0 && output.len() as u64 != manifest.total_bytes {
        return Err(format!(
            "shard manifest expected {} bytes but reconstructed {} bytes",
            manifest.total_bytes,
            output.len()
        ));
    }

    Ok(output)
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn resolve_manifest_entry_uri(manifest_uri: &str, entry_uri: &str) -> String {
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
pub(crate) fn load_optional_text_from_root(
    root: &Path,
    rel: &str,
) -> Result<Option<String>, String> {
    let path = root.join(rel);
    if !path.exists() {
        return Ok(None);
    }
    fs::read_to_string(&path)
        .map(Some)
        .map_err(|err| format!("failed to read {}: {err}", path.display()))
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn load_optional_text_candidates_from_root(
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
pub(crate) fn load_burnpack_asset_from_root(
    root: &Path,
    base_safetensors_rel: &str,
    precision_env: &str,
) -> Result<Vec<u8>, String> {
    let candidates =
        candidate_burnpack_names(base_safetensors_rel, prefer_f16_burnpack(precision_env));
    let mut checked = Vec::new();
    for candidate in candidates {
        let candidate_path = root.join(Path::new(&candidate));
        checked.push(candidate_path.display().to_string());
        if candidate_path.exists() {
            return fs::read(&candidate_path)
                .map_err(|err| format!("failed to read {}: {err}", candidate_path.display()));
        }

        for manifest_path in burnpack_manifest_candidate_paths(&candidate_path) {
            checked.push(manifest_path.display().to_string());
            if !manifest_path.exists() {
                continue;
            }
            return load_burnpack_from_manifest(&manifest_path);
        }
    }

    Err(format!(
        "failed to locate burnpack under '{}'; checked: {}",
        root.display(),
        checked.join(", "),
    ))
}

#[cfg(not(target_arch = "wasm32"))]
fn burnpack_manifest_candidate_paths(candidate_path: &Path) -> [PathBuf; 2] {
    let text = candidate_path.to_string_lossy();
    let candidates = burnpack_manifest_candidates(&text);
    [PathBuf::from(&candidates[0]), PathBuf::from(&candidates[1])]
}

#[cfg(not(target_arch = "wasm32"))]
fn load_burnpack_from_manifest(manifest_path: &Path) -> Result<Vec<u8>, String> {
    let manifest_bytes = fs::read(manifest_path).map_err(|err| {
        format!(
            "failed to read shard manifest {}: {err}",
            manifest_path.display()
        )
    })?;
    let manifest =
        parse_shard_manifest_bytes(&manifest_bytes, &manifest_path.display().to_string())?;

    reconstruct_burnpack_from_shard_manifest(&manifest, |shard| {
        let shard_path = Path::new(shard);
        let full_path = if shard_path.is_absolute() {
            shard_path.to_path_buf()
        } else {
            manifest_path
                .parent()
                .map(|parent| parent.join(shard_path))
                .ok_or_else(|| {
                    format!("invalid shard manifest path '{}'", manifest_path.display())
                })?
        };
        fs::read(&full_path)
            .map_err(|err| format!("failed to read shard {}: {err}", full_path.display()))
    })
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use std::collections::HashMap;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{
        burnpack_manifest_candidates, candidate_burnpack_names, load_burnpack_asset_from_root,
        parse_shard_manifest_bytes, prefer_f16_burnpack, reconstruct_burnpack_from_shard_manifest,
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
    fn manifest_candidates_include_new_and_legacy_suffixes() {
        let candidates = burnpack_manifest_candidates("model_f16.bpk");
        assert_eq!(candidates[0], "model_f16.bpk.shards.json");
        assert_eq!(candidates[1], "model_f16.bpk.manifest.json");
    }

    #[test]
    fn prefer_f16_default_is_true() {
        assert!(prefer_f16_burnpack("BURN_SYNTH_TEST_BPK_PRECISION"));
    }

    #[test]
    fn loads_sharded_manifest_when_burnpack_missing() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).expect("failed to create temp root");

        let manifest = root.join("model_f16.bpk.shards.json");
        let shard_a = root.join("model_f16.bpk.shard-aa");
        let shard_b = root.join("model_f16.bpk.shard-ab");
        fs::write(&shard_a, b"abc").expect("failed to write shard a");
        fs::write(&shard_b, b"def").expect("failed to write shard b");
        fs::write(
            &manifest,
            r#"{"total_bytes":6,"shards":["model_f16.bpk.shard-aa","model_f16.bpk.shard-ab"]}"#,
        )
        .expect("failed to write manifest");

        let bytes =
            load_burnpack_asset_from_root(&root, "model.safetensors", "BURN_SYNTH_TEST_PRECISION")
                .expect("failed to load sharded burnpack");
        assert_eq!(bytes, b"abcdef");

        fs::remove_dir_all(root).expect("failed to cleanup temp root");
    }

    #[test]
    fn loads_legacy_manifest_suffix_when_shards_manifest_missing() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).expect("failed to create temp root");

        let manifest = root.join("model_f16.bpk.manifest.json");
        let shard_a = root.join("model_f16.bpk.shard-aa");
        let shard_b = root.join("model_f16.bpk.shard-ab");
        fs::write(&shard_a, b"abc").expect("failed to write shard a");
        fs::write(&shard_b, b"def").expect("failed to write shard b");
        fs::write(
            &manifest,
            r#"{"total_bytes":6,"shards":["model_f16.bpk.shard-aa","model_f16.bpk.shard-ab"]}"#,
        )
        .expect("failed to write manifest");

        let bytes =
            load_burnpack_asset_from_root(&root, "model.safetensors", "BURN_SYNTH_TEST_PRECISION")
                .expect("failed to load burnpack from legacy manifest suffix");
        assert_eq!(bytes, b"abcdef");

        fs::remove_dir_all(root).expect("failed to cleanup temp root");
    }

    #[test]
    fn loads_manifest_with_detailed_entries_and_nested_paths() {
        let root = unique_temp_dir();
        let shard_root = root.join("shards");
        fs::create_dir_all(&shard_root).expect("failed to create temp root");

        let manifest = root.join("model_f16.bpk.shards.json");
        fs::write(shard_root.join("a.bin"), b"abc").expect("failed to write shard a");
        fs::write(shard_root.join("b.bin"), b"def").expect("failed to write shard b");
        fs::write(
            &manifest,
            r#"{"total_bytes":6,"files":[{"path":"shards/a.bin","size":3},{"path":"shards/b.bin","bytes":3}]}"#,
        )
        .expect("failed to write manifest");

        let bytes =
            load_burnpack_asset_from_root(&root, "model.safetensors", "BURN_SYNTH_TEST_PRECISION")
                .expect("failed to load burnpack from nested detailed manifest");
        assert_eq!(bytes, b"abcdef");

        fs::remove_dir_all(root).expect("failed to cleanup temp root");
    }

    #[test]
    fn direct_burnpack_takes_precedence_over_manifest() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).expect("failed to create temp root");

        fs::write(root.join("model_f16.bpk"), b"direct-bpk").expect("failed to write direct bpk");
        fs::write(root.join("model_f16.bpk.shard-aa"), b"manifest").expect("write shard");
        fs::write(
            root.join("model_f16.bpk.shards.json"),
            r#"{"total_bytes":8,"shards":["model_f16.bpk.shard-aa"]}"#,
        )
        .expect("write manifest");

        let bytes =
            load_burnpack_asset_from_root(&root, "model.safetensors", "BURN_SYNTH_TEST_PRECISION")
                .expect("failed to load direct burnpack");
        assert_eq!(bytes, b"direct-bpk");

        fs::remove_dir_all(root).expect("failed to cleanup temp root");
    }

    #[test]
    fn reconstruct_rejects_empty_manifest_entries() {
        let manifest =
            parse_shard_manifest_bytes(br#"{"total_bytes":5}"#, "inline").expect("parse manifest");
        let err = reconstruct_burnpack_from_shard_manifest(&manifest, |_| {
            Err("loader should not be invoked".to_string())
        })
        .expect_err("expected empty manifest error");
        assert!(
            err.contains("no shard entries"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn reconstruct_rejects_total_byte_mismatch() {
        let manifest =
            parse_shard_manifest_bytes(br#"{"total_bytes":7,"shards":["a","b"]}"#, "inline")
                .expect("parse manifest");
        let mut shards = HashMap::new();
        shards.insert("a".to_string(), b"abc".to_vec());
        shards.insert("b".to_string(), b"def".to_vec());
        let err = reconstruct_burnpack_from_shard_manifest(&manifest, |name| {
            shards
                .get(name)
                .cloned()
                .ok_or_else(|| format!("missing shard: {name}"))
        })
        .expect_err("expected size mismatch");
        assert!(
            err.contains("expected 7 bytes"),
            "unexpected error message: {err}"
        );
    }
}

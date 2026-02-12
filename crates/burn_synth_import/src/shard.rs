use std::fs;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::io::ensure_parent_dir;
use crate::layout::burnpack_manifest_candidates;
use crate::plan::ArtifactPolicy;

const ONE_MIB: u64 = 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BurnpackShardManifest {
    #[serde(default = "default_manifest_version")]
    pub version: u32,
    #[serde(default)]
    pub source_file: String,
    #[serde(default)]
    pub total_bytes: u64,
    #[serde(default)]
    pub shard_size_bytes: u64,
    #[serde(default)]
    pub sha256: String,
    #[serde(default)]
    pub shards: Vec<BurnpackShardEntry>,
    #[serde(default)]
    pub files: Vec<BurnpackShardEntry>,
    #[serde(default)]
    pub parts: Vec<BurnpackShardEntry>,
}

const fn default_manifest_version() -> u32 {
    1
}

impl BurnpackShardManifest {
    pub fn shard_entries(&self) -> &[BurnpackShardEntry] {
        if !self.shards.is_empty() {
            &self.shards
        } else if !self.files.is_empty() {
            &self.files
        } else {
            &self.parts
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum BurnpackShardEntry {
    Path(String),
    Detailed {
        path: String,
        #[serde(default)]
        size: Option<u64>,
        #[serde(default)]
        bytes: Option<u64>,
        #[serde(default)]
        sha256: Option<String>,
    },
}

impl BurnpackShardEntry {
    pub fn path(&self) -> &str {
        match self {
            Self::Path(path) => path,
            Self::Detailed { path, .. } => path,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ShardWriteReport {
    pub manifest_path: PathBuf,
    pub shard_paths: Vec<PathBuf>,
    pub total_bytes: u64,
}

pub fn apply_artifact_policy(
    burnpack_path: &Path,
    policy: ArtifactPolicy,
    overwrite: bool,
) -> Result<Option<ShardWriteReport>, String> {
    let Some(shard_size_mib) = policy.shard_size_mib() else {
        return Ok(None);
    };
    write_shards_for_burnpack(burnpack_path, shard_size_mib, overwrite).map(Some)
}

pub fn write_shards_for_burnpack(
    burnpack_path: &Path,
    shard_size_mib: u64,
    overwrite: bool,
) -> Result<ShardWriteReport, String> {
    if !burnpack_path.exists() {
        return Err(format!(
            "burnpack does not exist for sharding: {}",
            burnpack_path.display()
        ));
    }
    if shard_size_mib == 0 {
        return Err("invalid shard size 0 MiB".to_string());
    }
    let shard_size_bytes = shard_size_mib
        .checked_mul(ONE_MIB)
        .ok_or_else(|| "shard size overflow".to_string())?;

    let manifest_path = shard_manifest_path(burnpack_path);
    if manifest_path.exists() && !overwrite && manifest_has_all_shards(&manifest_path) {
        let existing_manifest = read_shard_manifest(&manifest_path)?;
        let shard_paths = existing_manifest
            .shard_entries()
            .iter()
            .map(|entry| resolve_manifest_entry_path(&manifest_path, entry.path()))
            .collect::<Vec<_>>();
        return Ok(ShardWriteReport {
            manifest_path,
            shard_paths,
            total_bytes: existing_manifest.total_bytes,
        });
    }

    if overwrite {
        cleanup_existing_shards(&manifest_path)?;
    }

    ensure_parent_dir(&manifest_path).map_err(|err| {
        format!(
            "failed to create shard output directory '{}': {err}",
            manifest_path.display()
        )
    })?;

    let source_file_name = burnpack_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("invalid burnpack file name '{}'", burnpack_path.display()))?;

    let source_file = fs::File::open(burnpack_path).map_err(|err| {
        format!(
            "failed to open burnpack '{}': {err}",
            burnpack_path.display()
        )
    })?;
    let mut reader = BufReader::new(source_file);
    let mut full_hasher = Sha256::new();
    let mut total_bytes = 0u64;
    let mut buffer = vec![0u8; 1024 * 1024];

    let mut shard_entries = Vec::new();
    let mut shard_paths = Vec::new();
    let mut shard_index = 0usize;
    loop {
        let shard_file_name = format!("{source_file_name}.shard-{shard_index:05}");
        let shard_path = burnpack_path.with_file_name(&shard_file_name);
        let shard_file = fs::File::create(&shard_path)
            .map_err(|err| format!("failed to create shard '{}': {err}", shard_path.display()))?;
        let mut writer = BufWriter::new(shard_file);
        let mut shard_hasher = Sha256::new();
        let mut shard_written = 0u64;

        while shard_written < shard_size_bytes {
            let remaining = shard_size_bytes - shard_written;
            let to_read = remaining.min(buffer.len() as u64) as usize;
            let read = reader.read(&mut buffer[..to_read]).map_err(|err| {
                format!(
                    "failed while reading burnpack '{}' for sharding: {err}",
                    burnpack_path.display()
                )
            })?;
            if read == 0 {
                break;
            }
            writer.write_all(&buffer[..read]).map_err(|err| {
                format!(
                    "failed while writing shard '{}': {err}",
                    shard_path.display()
                )
            })?;
            shard_hasher.update(&buffer[..read]);
            full_hasher.update(&buffer[..read]);
            shard_written += read as u64;
            total_bytes += read as u64;
        }

        writer
            .flush()
            .map_err(|err| format!("failed to flush shard '{}': {err}", shard_path.display()))?;

        if shard_written == 0 {
            let _ = fs::remove_file(&shard_path);
            break;
        }

        let shard_entry = BurnpackShardEntry::Detailed {
            path: shard_file_name,
            size: Some(shard_written),
            bytes: Some(shard_written),
            sha256: Some(hex::encode(shard_hasher.finalize())),
        };
        shard_entries.push(shard_entry);
        shard_paths.push(shard_path);
        shard_index += 1;
    }

    if shard_entries.is_empty() {
        return Err(format!(
            "burnpack '{}' is empty; refusing to produce shard manifest",
            burnpack_path.display()
        ));
    }

    let manifest = BurnpackShardManifest {
        version: 1,
        source_file: source_file_name.to_string(),
        total_bytes,
        shard_size_bytes,
        sha256: hex::encode(full_hasher.finalize()),
        shards: shard_entries,
        files: Vec::new(),
        parts: Vec::new(),
    };
    let manifest_bytes = serde_json::to_vec_pretty(&manifest)
        .map_err(|err| format!("failed to serialize shard manifest: {err}"))?;
    fs::write(&manifest_path, manifest_bytes).map_err(|err| {
        format!(
            "failed to write shard manifest '{}': {err}",
            manifest_path.display()
        )
    })?;

    Ok(ShardWriteReport {
        manifest_path,
        shard_paths,
        total_bytes,
    })
}

pub fn shard_manifest_path(burnpack_path: &Path) -> PathBuf {
    let candidate = burnpack_path.to_string_lossy();
    PathBuf::from(burnpack_manifest_candidates(&candidate)[0].clone())
}

pub fn read_shard_manifest(path: &Path) -> Result<BurnpackShardManifest, String> {
    let bytes = fs::read(path)
        .map_err(|err| format!("failed to read shard manifest '{}': {err}", path.display()))?;
    serde_json::from_slice(&bytes)
        .map_err(|err| format!("failed to parse shard manifest '{}': {err}", path.display()))
}

fn manifest_has_all_shards(path: &Path) -> bool {
    let Ok(manifest) = read_shard_manifest(path) else {
        return false;
    };
    let entries = manifest.shard_entries();
    if entries.is_empty() {
        return false;
    }
    entries
        .iter()
        .all(|entry| resolve_manifest_entry_path(path, entry.path()).exists())
}

fn cleanup_existing_shards(manifest_path: &Path) -> Result<(), String> {
    if !manifest_path.exists() {
        return Ok(());
    }
    if let Ok(manifest) = read_shard_manifest(manifest_path) {
        for entry in manifest.shard_entries() {
            let shard_path = resolve_manifest_entry_path(manifest_path, entry.path());
            if shard_path.exists() {
                fs::remove_file(&shard_path).map_err(|err| {
                    format!(
                        "failed to remove stale shard '{}': {err}",
                        shard_path.display()
                    )
                })?;
            }
        }
    }
    fs::remove_file(manifest_path).map_err(|err| {
        format!(
            "failed to remove stale manifest '{}': {err}",
            manifest_path.display()
        )
    })
}

fn resolve_manifest_entry_path(manifest_path: &Path, shard: &str) -> PathBuf {
    let shard_path = Path::new(shard);
    if shard_path.is_absolute() {
        shard_path.to_path_buf()
    } else {
        manifest_path
            .parent()
            .map(|parent| parent.join(shard_path))
            .unwrap_or_else(|| shard_path.to_path_buf())
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::plan::ArtifactPolicy;

    use super::{
        BurnpackShardEntry, apply_artifact_policy, read_shard_manifest, shard_manifest_path,
        write_shards_for_burnpack,
    };

    fn unique_temp_dir() -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock drift")
            .as_nanos();
        std::env::temp_dir().join(format!("burn_synth_import_shard_{nanos}"))
    }

    fn patterned_bytes(len: usize) -> Vec<u8> {
        (0..len)
            .map(|idx| ((idx.wrapping_mul(37)).wrapping_add(11) % 251) as u8)
            .collect()
    }

    fn read_reconstructed_bytes(manifest_path: &Path) -> Vec<u8> {
        let manifest = read_shard_manifest(manifest_path).expect("read shard manifest");
        let mut out = Vec::new();
        for entry in manifest.shard_entries() {
            let shard_path = super::resolve_manifest_entry_path(manifest_path, entry.path());
            let bytes = std::fs::read(&shard_path).expect("read shard file");
            out.extend_from_slice(&bytes);
        }
        out
    }

    #[test]
    fn splits_burnpack_and_writes_manifest() {
        let root = unique_temp_dir();
        std::fs::create_dir_all(&root).expect("create temp dir");
        let burnpack = root.join("model.bpk");
        std::fs::write(&burnpack, vec![7u8; 3 * 1024 * 1024 + 13]).expect("write burnpack");

        let report = write_shards_for_burnpack(&burnpack, 1, true).expect("split burnpack");
        assert!(report.manifest_path.exists());
        assert!(report.shard_paths.len() >= 4);

        let manifest = read_shard_manifest(&report.manifest_path).expect("read manifest");
        assert_eq!(manifest.total_bytes, (3 * 1024 * 1024 + 13) as u64);
        assert_eq!(manifest.shard_entries().len(), report.shard_paths.len());

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn single_file_policy_skips_shard_output() {
        let root = unique_temp_dir();
        std::fs::create_dir_all(&root).expect("create temp dir");
        let burnpack = root.join("model.bpk");
        std::fs::write(&burnpack, patterned_bytes(1024 * 1024)).expect("write burnpack");

        let report = apply_artifact_policy(&burnpack, ArtifactPolicy::SingleFile, true)
            .expect("apply artifact policy");
        assert!(report.is_none());
        assert!(!shard_manifest_path(&burnpack).exists());

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn complete_manifest_is_reused_without_overwrite() {
        let root = unique_temp_dir();
        std::fs::create_dir_all(&root).expect("create temp dir");
        let burnpack = root.join("model.bpk");
        std::fs::write(&burnpack, patterned_bytes(2 * 1024 * 1024 + 17)).expect("write burnpack");

        let first = write_shards_for_burnpack(&burnpack, 1, true).expect("first split");
        let manifest_before = std::fs::read(&first.manifest_path).expect("read manifest before");
        let second = write_shards_for_burnpack(&burnpack, 1, false).expect("second split");
        let manifest_after = std::fs::read(&second.manifest_path).expect("read manifest after");

        assert_eq!(first.manifest_path, second.manifest_path);
        assert_eq!(first.shard_paths, second.shard_paths);
        assert_eq!(manifest_before, manifest_after);

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn missing_shard_forces_regeneration_without_overwrite() {
        let root = unique_temp_dir();
        std::fs::create_dir_all(&root).expect("create temp dir");
        let burnpack = root.join("model.bpk");
        let source = patterned_bytes(2 * 1024 * 1024 + 99);
        std::fs::write(&burnpack, source.as_slice()).expect("write burnpack");

        let first = write_shards_for_burnpack(&burnpack, 1, true).expect("first split");
        assert!(!first.shard_paths.is_empty());
        std::fs::remove_file(&first.shard_paths[0]).expect("remove shard to simulate corruption");

        let rebuilt =
            write_shards_for_burnpack(&burnpack, 1, false).expect("rebuild missing shard split");
        let manifest = read_shard_manifest(&rebuilt.manifest_path).expect("read rebuilt manifest");
        assert_eq!(manifest.total_bytes as usize, source.len());
        for entry in manifest.shard_entries() {
            let shard_path =
                super::resolve_manifest_entry_path(&rebuilt.manifest_path, entry.path());
            assert!(
                shard_path.exists(),
                "missing shard {}",
                shard_path.display()
            );
        }

        let reconstructed = read_reconstructed_bytes(&rebuilt.manifest_path);
        assert_eq!(reconstructed, source);

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn manifest_and_shard_hashes_match_source_content() {
        let root = unique_temp_dir();
        std::fs::create_dir_all(&root).expect("create temp dir");
        let burnpack = root.join("model.bpk");
        let source = patterned_bytes(3 * 1024 * 1024 + 31);
        std::fs::write(&burnpack, source.as_slice()).expect("write burnpack");

        let report = write_shards_for_burnpack(&burnpack, 1, true).expect("split burnpack");
        let manifest = read_shard_manifest(&report.manifest_path).expect("read manifest");
        assert_eq!(manifest.sha256, crate::io::sha256_bytes(source.as_slice()));
        assert_eq!(manifest.total_bytes as usize, source.len());

        for entry in manifest.shard_entries() {
            match entry {
                BurnpackShardEntry::Detailed {
                    path,
                    size,
                    bytes,
                    sha256,
                } => {
                    let shard_path =
                        super::resolve_manifest_entry_path(&report.manifest_path, path);
                    let shard_metadata = std::fs::metadata(&shard_path).expect("shard metadata");
                    let expected_len = shard_metadata.len();
                    assert_eq!(size.unwrap_or(expected_len), expected_len);
                    assert_eq!(bytes.unwrap_or(expected_len), expected_len);
                    let expected_sha = crate::io::sha256_file(&shard_path).expect("shard hash");
                    assert_eq!(sha256.as_deref(), Some(expected_sha.as_str()));
                }
                BurnpackShardEntry::Path(path) => {
                    panic!("writer should emit detailed shard entry, got path-only entry: {path}");
                }
            }
        }

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn overwrite_replaces_manifest_and_removes_stale_shards() {
        let root = unique_temp_dir();
        std::fs::create_dir_all(&root).expect("create temp dir");
        let burnpack = root.join("model.bpk");

        std::fs::write(&burnpack, patterned_bytes(3 * 1024 * 1024 + 19)).expect("write burnpack");
        let first = write_shards_for_burnpack(&burnpack, 1, true).expect("first split");
        let first_shards = first.shard_paths.clone();
        assert!(
            first_shards.len() >= 4,
            "expected multiple shards before overwrite"
        );

        // Rewrite with much smaller source and force overwrite to ensure stale shards get removed.
        std::fs::write(&burnpack, patterned_bytes(1024 * 1024 + 7)).expect("rewrite burnpack");
        let second = write_shards_for_burnpack(&burnpack, 1, true).expect("second split");
        let second_manifest = read_shard_manifest(&second.manifest_path).expect("read manifest");
        assert_eq!(second_manifest.total_bytes, (1024 * 1024 + 7) as u64);
        assert!(
            second.shard_paths.len() <= 2,
            "expected fewer shards after overwrite"
        );

        for stale in first_shards {
            if !second.shard_paths.contains(&stale) {
                assert!(
                    !stale.exists(),
                    "stale shard should be removed during overwrite: {}",
                    stale.display()
                );
            }
        }

        let reconstructed = read_reconstructed_bytes(&second.manifest_path);
        assert_eq!(reconstructed.len(), 1024 * 1024 + 7);

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn shard_entries_support_legacy_files_and_parts_fields() {
        let root = unique_temp_dir();
        std::fs::create_dir_all(&root).expect("create temp dir");

        let files_manifest = root.join("files_manifest.json");
        std::fs::write(
            &files_manifest,
            br#"{"total_bytes":3,"files":[{"path":"legacy-files.shard"}]}"#,
        )
        .expect("write files manifest");
        let files = read_shard_manifest(&files_manifest).expect("parse files manifest");
        assert_eq!(files.shard_entries().len(), 1);
        assert_eq!(files.shard_entries()[0].path(), "legacy-files.shard");

        let parts_manifest = root.join("parts_manifest.json");
        std::fs::write(
            &parts_manifest,
            br#"{"total_bytes":3,"parts":[{"path":"legacy-parts.shard"}]}"#,
        )
        .expect("write parts manifest");
        let parts = read_shard_manifest(&parts_manifest).expect("parse parts manifest");
        assert_eq!(parts.shard_entries().len(), 1);
        assert_eq!(parts.shard_entries()[0].path(), "legacy-parts.shard");

        let _ = std::fs::remove_dir_all(root);
    }
}

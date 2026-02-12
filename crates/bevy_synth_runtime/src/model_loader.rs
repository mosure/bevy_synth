use burn_synth_import::layout::{
    burnpack_manifest_candidates as shared_manifest_candidates,
    candidate_burnpack_names as shared_candidate_burnpack_names,
};
use burn_synth_import::shard::BurnpackShardManifest;

pub(crate) fn prefer_f16_burnpack(_primary: &str) -> bool {
    true
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

#[cfg(all(test, not(target_arch = "wasm32")))]
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
use std::io::{self, Write};
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
pub(crate) fn resolve_burnpack_asset_path_from_root(
    root: &Path,
    base_safetensors_rel: &str,
    precision_env: &str,
) -> Result<PathBuf, String> {
    resolve_burnpack_asset_path_from_root_with_preference(
        root,
        base_safetensors_rel,
        prefer_f16_burnpack(precision_env),
    )
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn resolve_burnpack_asset_path_from_root_with_preference(
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

        for manifest_path in burnpack_manifest_candidate_paths(&candidate_path) {
            checked.push(manifest_path.display().to_string());
            if !manifest_path.exists() {
                continue;
            }
            materialize_burnpack_from_manifest(&manifest_path, &candidate_path)?;
            return Ok(candidate_path);
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
fn resolve_manifest_entry_path(manifest_path: &Path, shard_path: &str) -> Result<PathBuf, String> {
    let shard_path = Path::new(shard_path);
    if shard_path.is_absolute() {
        return Ok(shard_path.to_path_buf());
    }
    manifest_path
        .parent()
        .map(|parent| parent.join(shard_path))
        .ok_or_else(|| format!("invalid shard manifest path '{}'", manifest_path.display()))
}

#[cfg(not(target_arch = "wasm32"))]
fn materialize_burnpack_from_manifest(
    manifest_path: &Path,
    destination_path: &Path,
) -> Result<(), String> {
    let manifest_bytes = fs::read(manifest_path).map_err(|err| {
        format!(
            "failed to read shard manifest {}: {err}",
            manifest_path.display()
        )
    })?;
    let manifest =
        parse_shard_manifest_bytes(&manifest_bytes, &manifest_path.display().to_string())?;
    let shards = manifest.shard_entries();
    if shards.is_empty() {
        return Err("shard manifest contains no shard entries".to_string());
    }

    if let Some(parent) = destination_path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            format!(
                "failed to create burnpack output directory {}: {err}",
                parent.display()
            )
        })?;
    }
    let destination_tmp = destination_path.with_extension(
        destination_path
            .extension()
            .map(|ext| format!("{}.tmp", ext.to_string_lossy()))
            .unwrap_or_else(|| "tmp".to_string()),
    );
    let mut output = fs::File::create(&destination_tmp).map_err(|err| {
        format!(
            "failed to create reconstructed burnpack {}: {err}",
            destination_tmp.display()
        )
    })?;
    let mut bytes_written = 0u64;
    for shard in shards {
        let full_path = resolve_manifest_entry_path(manifest_path, shard.path())?;
        let mut file = fs::File::open(&full_path)
            .map_err(|err| format!("failed to open shard {}: {err}", full_path.display()))?;
        bytes_written = bytes_written.saturating_add(
            io::copy(&mut file, &mut output)
                .map_err(|err| format!("failed to read shard {}: {err}", full_path.display()))?,
        );
    }
    output.flush().map_err(|err| {
        format!(
            "failed to flush reconstructed burnpack {}: {err}",
            destination_tmp.display()
        )
    })?;

    if manifest.total_bytes > 0 && bytes_written != manifest.total_bytes {
        return Err(format!(
            "shard manifest expected {} bytes but reconstructed {} bytes",
            manifest.total_bytes, bytes_written
        ));
    }

    if destination_path.exists() {
        fs::remove_file(destination_path).map_err(|err| {
            format!(
                "failed to replace stale burnpack {}: {err}",
                destination_path.display()
            )
        })?;
    }
    fs::rename(&destination_tmp, destination_path).map_err(|err| {
        format!(
            "failed to finalize reconstructed burnpack {}: {err}",
            destination_path.display()
        )
    })?;
    Ok(())
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use std::cmp;
    use std::collections::HashMap;
    use std::fs;
    use std::io::Write;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::thread;
    use std::time::Duration;
    use std::time::{SystemTime, UNIX_EPOCH};

    use burn_synth_import::io::{sha256_bytes, sha256_file};
    use burn_synth_import::shard::write_shards_for_burnpack;
    use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System, get_current_pid};

    use super::{
        burnpack_manifest_candidates, candidate_burnpack_names, parse_shard_manifest_bytes,
        prefer_f16_burnpack, reconstruct_burnpack_from_shard_manifest,
        resolve_burnpack_asset_path_from_root,
        resolve_burnpack_asset_path_from_root_with_preference,
    };

    const ONE_MIB: u64 = 1024 * 1024;

    struct ProcessMemoryMonitor {
        done: Arc<AtomicBool>,
        peak_bytes: Arc<AtomicU64>,
        join: Option<thread::JoinHandle<()>>,
    }

    impl ProcessMemoryMonitor {
        fn start(pid: Pid, baseline_bytes: u64) -> Self {
            let done = Arc::new(AtomicBool::new(false));
            let peak_bytes = Arc::new(AtomicU64::new(baseline_bytes));
            let done_thread = Arc::clone(&done);
            let peak_thread = Arc::clone(&peak_bytes);
            let join = thread::Builder::new()
                .name("burn_synth_rss_monitor".to_string())
                .spawn(move || {
                    let mut system = System::new();
                    while !done_thread.load(Ordering::Relaxed) {
                        if let Some(bytes) = refresh_process_rss_bytes(&mut system, pid) {
                            peak_thread.fetch_max(bytes, Ordering::Relaxed);
                        }
                        thread::sleep(Duration::from_millis(2));
                    }
                    if let Some(bytes) = refresh_process_rss_bytes(&mut system, pid) {
                        peak_thread.fetch_max(bytes, Ordering::Relaxed);
                    }
                })
                .expect("failed to spawn process memory monitor");
            Self {
                done,
                peak_bytes,
                join: Some(join),
            }
        }

        fn stop(mut self) -> u64 {
            self.done.store(true, Ordering::Relaxed);
            if let Some(join) = self.join.take() {
                let _ = join.join();
            }
            self.peak_bytes.load(Ordering::Relaxed)
        }
    }

    fn refresh_process_rss_bytes(system: &mut System, pid: Pid) -> Option<u64> {
        system.refresh_processes_specifics(
            ProcessesToUpdate::Some(&[pid]),
            true,
            ProcessRefreshKind::nothing().with_memory(),
        );
        system.process(pid).map(|process| process.memory())
    }

    fn measure_peak_rss_delta_bytes<F, T>(action: F) -> (T, u64)
    where
        F: FnOnce() -> T,
    {
        let pid = get_current_pid().expect("failed to get current process id");
        let mut system = System::new();
        let baseline_bytes = refresh_process_rss_bytes(&mut system, pid).unwrap_or(0);
        let monitor = ProcessMemoryMonitor::start(pid, baseline_bytes);
        let output = action();
        // Give allocator work a moment to settle so transient peaks are captured.
        thread::sleep(Duration::from_millis(20));
        let peak_bytes = monitor.stop();
        (output, peak_bytes.saturating_sub(baseline_bytes))
    }

    fn write_patterned_file(path: &std::path::Path, total_bytes: u64) {
        let mut file = fs::File::create(path).expect("failed to create source burnpack");
        let mut remaining = total_bytes;
        let mut seed = 0u8;
        let mut chunk = vec![0u8; ONE_MIB as usize];
        while remaining > 0 {
            let write_now = cmp::min(remaining, chunk.len() as u64) as usize;
            for byte in chunk.iter_mut().take(write_now) {
                *byte = seed;
                seed = seed.wrapping_add(17);
            }
            file.write_all(&chunk[..write_now])
                .expect("failed to write source burnpack chunk");
            remaining -= write_now as u64;
        }
        file.flush().expect("failed to flush source burnpack");
    }

    fn format_mebibytes(bytes: u64) -> String {
        format!("{:.1} MiB", bytes as f64 / ONE_MIB as f64)
    }

    fn load_burnpack_bytes_for_test(root: &std::path::Path, base_safetensors_rel: &str) -> Vec<u8> {
        let path = resolve_burnpack_asset_path_from_root(
            root,
            base_safetensors_rel,
            "BURN_SYNTH_TEST_PRECISION",
        )
        .expect("failed to resolve burnpack path");
        fs::read(&path).expect("failed to read burnpack bytes")
    }

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

        let bytes = load_burnpack_bytes_for_test(&root, "model.safetensors");
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

        let bytes = load_burnpack_bytes_for_test(&root, "model.safetensors");
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

        let bytes = load_burnpack_bytes_for_test(&root, "model.safetensors");
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

        let bytes = load_burnpack_bytes_for_test(&root, "model.safetensors");
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

    #[test]
    fn sharded_load_process_ram_spike_respects_total_plus_shard_cap() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).expect("failed to create temp root");

        let source_path = root.join("source_model_f16.bpk");
        let load_path = root.join("model_f16.bpk");
        let total_bytes = 96 * ONE_MIB;
        write_patterned_file(&source_path, total_bytes);
        let source_hash = sha256_file(&source_path).expect("failed to hash source burnpack");

        for shard_size_mib in [4_u64, 32_u64] {
            fs::copy(&source_path, &load_path).expect("failed to copy source burnpack");
            write_shards_for_burnpack(&load_path, shard_size_mib, true)
                .expect("failed to shard burnpack");
            fs::remove_file(&load_path).expect("failed to remove direct burnpack");

            let (loaded, peak_delta) = measure_peak_rss_delta_bytes(|| {
                load_burnpack_bytes_for_test(&root, "model.safetensors")
            });

            assert_eq!(loaded.len() as u64, total_bytes);
            assert_eq!(sha256_bytes(&loaded), source_hash);

            let shard_bytes = shard_size_mib * ONE_MIB;
            // Bound process RSS spike to reconstructed payload + one shard + allocator slack.
            let allowed_peak_delta = total_bytes + shard_bytes + (64 * ONE_MIB);
            assert!(
                peak_delta <= allowed_peak_delta,
                "peak host RAM delta {} exceeded bound {} for shard size {} MiB",
                format_mebibytes(peak_delta),
                format_mebibytes(allowed_peak_delta),
                shard_size_mib
            );
        }

        fs::remove_dir_all(root).expect("failed to cleanup temp root");
    }

    #[test]
    fn sharded_materialize_path_process_ram_spike_stays_under_128mib() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).expect("failed to create temp root");

        let source_path = root.join("source_model_f16.bpk");
        let load_path = root.join("model_f16.bpk");
        let total_bytes = 96 * ONE_MIB;
        write_patterned_file(&source_path, total_bytes);
        let source_hash = sha256_file(&source_path).expect("failed to hash source burnpack");

        for shard_size_mib in [4_u64, 32_u64] {
            fs::copy(&source_path, &load_path).expect("failed to copy source burnpack");
            write_shards_for_burnpack(&load_path, shard_size_mib, true)
                .expect("failed to shard burnpack");
            fs::remove_file(&load_path).expect("failed to remove direct burnpack");

            let (resolved, peak_delta) = measure_peak_rss_delta_bytes(|| {
                resolve_burnpack_asset_path_from_root_with_preference(
                    &root,
                    "model.safetensors",
                    true,
                )
                .expect("failed to resolve sharded burnpack path")
            });

            assert!(resolved.exists(), "resolved path should exist");
            assert_eq!(
                sha256_file(&resolved).expect("failed to hash resolved burnpack"),
                source_hash
            );

            // Streaming materialization should stay well below a full-model host allocation.
            let allowed_peak_delta = 128 * ONE_MIB;
            assert!(
                peak_delta <= allowed_peak_delta,
                "peak host RAM delta {} exceeded streaming budget {} for shard size {} MiB",
                format_mebibytes(peak_delta),
                format_mebibytes(allowed_peak_delta),
                shard_size_mib
            );
            fs::remove_file(&resolved).expect("failed to remove materialized burnpack");
        }

        // Validate the precision-env helper resolves identically.
        let resolved = resolve_burnpack_asset_path_from_root(
            &root,
            "model.safetensors",
            "BURN_SYNTH_TEST_PRECISION",
        )
        .expect("resolve via precision env helper");
        assert!(
            resolved.exists(),
            "precision helper should materialize file"
        );

        fs::remove_dir_all(root).expect("failed to cleanup temp root");
    }
}

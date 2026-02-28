use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use burn_synth_import::layout::stem_has_low_precision_marker;
use burn_synth_import::parts::{
    burnpack_parts_manifest_path, read_parts_manifest, remove_legacy_shard_artifacts_for_burnpack,
    resolve_part_entry_path, write_burnpack_parts_for_wasm,
};
use clap::Parser;

#[derive(Debug, Clone)]
struct OversizedManifestIssue {
    reason: String,
    unsplittable_single_tensor_blob: bool,
}

#[derive(Parser, Debug)]
#[command(
    about = "Ensure burnpack parts artifacts exist for wasm web model bundles",
    version
)]
struct Args {
    /// One or more directories to scan recursively for .bpk files.
    /// Defaults to www/assets/models/MIDI-3D and www/assets/models/RMBG-1.4.
    #[arg(long = "root")]
    roots: Vec<PathBuf>,

    /// Burnpack part size in MiB (used for wasm incremental loading).
    #[arg(long, default_value_t = 64)]
    part_size_mib: u64,

    /// Hard cap for per-part size validation.
    /// Defaults to `part_size_mib` when omitted.
    #[arg(long)]
    max_part_size_mib: Option<u64>,

    /// Overwrite existing manifests/parts.
    #[arg(long)]
    overwrite: bool,

    /// Keep legacy shard artifacts if present (`.bpk.shards.json`, `.bpk.manifest.json`, `.bpk.shard-*`).
    #[arg(long)]
    keep_legacy_shards: bool,

    /// Print planned work without writing artifacts.
    #[arg(long)]
    dry_run: bool,

    /// Allow model components to be present in only one precision (f32 or f16).
    #[arg(long)]
    allow_unpaired_precision: bool,

    /// Remove full `.bpk` files after parts manifests are validated/generated.
    /// This avoids storing duplicate full-file and parts-file payloads in web bundles.
    #[arg(long)]
    prune_full_bpk: bool,

    /// Optional hard cap (GiB) for each scanned root.
    /// Useful to catch accidental duplicate payload inflation in bundled web assets.
    #[arg(long)]
    max_root_size_gib: Option<u64>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let roots_explicit = !args.roots.is_empty();
    let roots = resolve_roots(args.roots);
    for root in &roots {
        if !root.exists() {
            if roots_explicit {
                return Err(format!("root does not exist: {}", root.display()).into());
            }
            println!(
                "[ARTIFACTS] skipping missing default root: {}",
                root.display()
            );
            continue;
        }
        ensure_triposg_metadata_aliases(root, args.dry_run)?;
    }
    let mut root_size_total = 0u64;
    let root_size_limit_bytes = args.max_root_size_gib.map(gib_to_bytes);
    for root in &roots {
        if !root.exists() {
            continue;
        }
        let root_size_bytes = directory_size_bytes(root)?;
        root_size_total = root_size_total.saturating_add(root_size_bytes);
        println!(
            "[ARTIFACTS] root {} size {:.2} GiB",
            root.display(),
            bytes_to_gib(root_size_bytes)
        );
        if let Some(limit) = root_size_limit_bytes
            && root_size_bytes > limit
        {
            return Err(format!(
                "root '{}' exceeds max size: {:.2} GiB > {:.2} GiB",
                root.display(),
                bytes_to_gib(root_size_bytes),
                bytes_to_gib(limit)
            )
            .into());
        }
    }
    println!(
        "[ARTIFACTS] aggregate root size {:.2} GiB across {} root(s)",
        bytes_to_gib(root_size_total),
        roots.len()
    );

    let mut burnpacks = Vec::new();
    let mut manifests = Vec::new();
    for root in &roots {
        if !root.exists() {
            continue;
        }
        collect_primary_burnpacks(root, &mut burnpacks)?;
        collect_parts_manifests(root, &mut manifests)?;
    }
    burnpacks.sort();
    burnpacks.dedup();
    manifests.sort();
    manifests.dedup();

    let burnpack_set = burnpacks.iter().cloned().collect::<BTreeSet<_>>();
    let mut manifest_only_burnpacks = Vec::new();
    let mut manifest_by_burnpack = BTreeMap::new();
    for manifest in manifests {
        let burnpack = burnpack_path_from_manifest_path(&manifest)?;
        if burnpack_set.contains(&burnpack) {
            continue;
        }
        manifest_by_burnpack.insert(burnpack.clone(), manifest);
        manifest_only_burnpacks.push(burnpack);
    }
    manifest_only_burnpacks.sort();
    manifest_only_burnpacks.dedup();

    let mut all_components = burnpacks.clone();
    all_components.extend(manifest_only_burnpacks.clone());
    all_components.sort();
    all_components.dedup();

    println!(
        "[ARTIFACTS] discovered {} source burnpack(s), {} manifest-only burnpack(s) across {} root(s)",
        burnpacks.len(),
        manifest_only_burnpacks.len(),
        roots.len()
    );
    if all_components.is_empty() {
        return Ok(());
    }

    if !args.allow_unpaired_precision {
        validate_precision_pairs(&all_components)?;
    }

    let part_size_mib = args.part_size_mib.max(1);
    let max_part_size_mib = args.max_part_size_mib.unwrap_or(part_size_mib).max(1);
    let max_part_size_bytes = max_part_size_mib.saturating_mul(1024 * 1024);

    let mut parts_manifest_count = 0usize;
    let mut part_file_count = 0usize;
    let mut removed_legacy_shard_count = 0usize;
    let mut repaired_oversized_manifest_count = 0usize;
    let mut normalized_low_precision_metadata_count = 0usize;
    let mut pruned_full_bpk_count = 0usize;
    for burnpack in &burnpacks {
        if normalize_low_precision_metadata_precision(burnpack, args.dry_run)? {
            normalized_low_precision_metadata_count += 1;
        }
        let parts_manifest = burnpack_parts_manifest_path(burnpack);
        let oversized_issue =
            detect_oversized_parts_manifest(&parts_manifest, max_part_size_bytes)?;

        if args.dry_run {
            println!("[ARTIFACTS][DRY RUN] {}", burnpack.display());
            if let Some(issue) = oversized_issue.as_ref() {
                if issue.unsplittable_single_tensor_blob {
                    println!(
                        "[ARTIFACTS][DRY RUN] oversized parts manifest for unsplittable single-tensor blob {} would be accepted: {}",
                        burnpack.display(),
                        issue.reason
                    );
                } else {
                    println!(
                        "[ARTIFACTS][DRY RUN] would repair oversized parts manifest for {}: {}",
                        burnpack.display(),
                        issue.reason
                    );
                }
            }
            if !args.keep_legacy_shards {
                println!(
                    "[ARTIFACTS][DRY RUN] would prune legacy shard artifacts for {}",
                    burnpack.display()
                );
            }
            if args.prune_full_bpk && burnpack.exists() {
                println!(
                    "[ARTIFACTS][DRY RUN] would remove full burnpack after parts validation: {}",
                    burnpack.display()
                );
            }
            continue;
        }

        let mut overwrite_parts = args.overwrite;
        if let Some(issue) = oversized_issue.as_ref()
            && !issue.unsplittable_single_tensor_blob
        {
            overwrite_parts = true;
            repaired_oversized_manifest_count += 1;
            println!(
                "[ARTIFACTS] repairing oversized parts manifest for {}: {}",
                burnpack.display(),
                issue.reason
            );
        }
        if let Some(issue) = oversized_issue.as_ref()
            && issue.unsplittable_single_tensor_blob
        {
            println!(
                "[ARTIFACTS] oversized part accepted for unsplittable single-tensor blob {}: {}",
                burnpack.display(),
                issue.reason
            );
        }

        if let Some(parts_report) =
            write_burnpack_parts_for_wasm(burnpack, part_size_mib, overwrite_parts)?
        {
            parts_manifest_count += 1;
            part_file_count += parts_report.part_paths.len();
        }

        if !parts_manifest.exists() {
            return Err(format!(
                "missing parts manifest after generation: {}",
                parts_manifest.display()
            )
            .into());
        }
        if let Some(issue) = detect_oversized_parts_manifest(&parts_manifest, max_part_size_bytes)?
        {
            if issue.unsplittable_single_tensor_blob {
                println!(
                    "[ARTIFACTS] oversized part remains for unsplittable single-tensor blob {}: {}",
                    burnpack.display(),
                    issue.reason
                );
            } else {
                return Err(format!(
                    "parts manifest still violates max part size ({max_part_size_mib} MiB) for {}: {}",
                    burnpack.display(),
                    issue.reason
                )
                .into());
            }
        }

        if !args.keep_legacy_shards {
            removed_legacy_shard_count += remove_legacy_shard_artifacts_for_burnpack(burnpack)?;
        }

        if args.prune_full_bpk && burnpack.exists() {
            fs::remove_file(burnpack)?;
            pruned_full_bpk_count += 1;
            let meta_path = burnpack.with_file_name(format!(
                "{}.meta.json",
                burnpack
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("model.bpk")
            ));
            if meta_path.exists() {
                fs::remove_file(meta_path)?;
            }
        }
    }

    for burnpack in &manifest_only_burnpacks {
        let Some(parts_manifest) = manifest_by_burnpack.get(burnpack) else {
            continue;
        };
        let oversized_issue = detect_oversized_parts_manifest(parts_manifest, max_part_size_bytes)?;

        if args.dry_run {
            println!(
                "[ARTIFACTS][DRY RUN] validate manifest-only burnpack {} via {}",
                burnpack.display(),
                parts_manifest.display()
            );
            if let Some(issue) = oversized_issue.as_ref() {
                if issue.unsplittable_single_tensor_blob {
                    println!(
                        "[ARTIFACTS][DRY RUN] manifest-only oversized check for unsplittable single-tensor blob {} would be accepted: {}",
                        burnpack.display(),
                        issue.reason
                    );
                } else {
                    println!(
                        "[ARTIFACTS][DRY RUN] manifest-only oversized check for {}: {}",
                        burnpack.display(),
                        issue.reason
                    );
                }
            }
            continue;
        }

        let manifest = read_parts_manifest(parts_manifest)?;
        if manifest.parts.is_empty() {
            return Err(format!(
                "manifest-only burnpack has zero parts: {}",
                parts_manifest.display()
            )
            .into());
        }
        if let Some(issue) = oversized_issue {
            if issue.unsplittable_single_tensor_blob {
                println!(
                    "[ARTIFACTS] oversized manifest-only part accepted for unsplittable single-tensor blob {}: {}",
                    burnpack.display(),
                    issue.reason
                );
            } else {
                return Err(format!(
                    "manifest-only burnpack '{}' violates max part size ({max_part_size_mib} MiB) and cannot auto-repair without source .bpk: {}",
                    burnpack.display(),
                    issue.reason
                )
                .into());
            }
        }
        for part in &manifest.parts {
            let part_path = resolve_part_entry_path(parts_manifest, &part.path)?;
            if !part_path.exists() {
                return Err(format!(
                    "manifest-only burnpack missing part file '{}' (manifest: {})",
                    part_path.display(),
                    parts_manifest.display()
                )
                .into());
            }
            let actual_bytes = fs::metadata(&part_path)?.len();
            if part.bytes > 0 && part.bytes != actual_bytes {
                return Err(format!(
                    "manifest-only part size mismatch '{}' expected {} found {}",
                    part_path.display(),
                    part.bytes,
                    actual_bytes
                )
                .into());
            }
            part_file_count += 1;
        }
        parts_manifest_count += 1;
    }

    if args.dry_run {
        println!("[ARTIFACTS][DRY RUN] complete");
        return Ok(());
    }

    println!(
        "[ARTIFACTS] generated/validated {} parts manifest(s), {} part file(s), repaired {} oversized manifest(s), normalized {} low-precision metadata file(s), removed {} legacy shard artifact(s), pruned {} full burnpack(s)",
        parts_manifest_count,
        part_file_count,
        repaired_oversized_manifest_count,
        normalized_low_precision_metadata_count,
        removed_legacy_shard_count,
        pruned_full_bpk_count
    );
    Ok(())
}

fn ensure_triposg_metadata_aliases(
    root: &Path,
    dry_run: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let dino_dir = root.join("image_encoder_dinov2");
    let feature_dir = root.join("feature_extractor_dinov2");
    let legacy_dino_2 = root.join("image_encoder_2/config.json");
    let legacy_dino_1 = root.join("image_encoder_1/config.json");
    let legacy_feature_2 = root.join("feature_extractor_2/preprocessor_config.json");
    let legacy_feature_1 = root.join("feature_extractor_1/preprocessor_config.json");

    ensure_alias_file(
        root,
        dino_dir.join("config.json"),
        &[legacy_dino_2, legacy_dino_1],
        dry_run,
    )?;
    ensure_alias_file(
        root,
        feature_dir.join("preprocessor_config.json"),
        &[legacy_feature_2, legacy_feature_1],
        dry_run,
    )?;
    Ok(())
}

fn ensure_alias_file(
    root: &Path,
    target: PathBuf,
    candidates: &[PathBuf],
    dry_run: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if target.exists() {
        return Ok(());
    }
    let Some(source) = candidates.iter().find(|candidate| candidate.exists()) else {
        return Ok(());
    };
    if dry_run {
        println!(
            "[ARTIFACTS][DRY RUN] alias metadata {} <- {}",
            target.display(),
            source.display()
        );
        return Ok(());
    }
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::copy(source, &target)?;
    println!(
        "[ARTIFACTS] created metadata alias {} <- {} (root: {})",
        target.display(),
        source.display(),
        root.display()
    );
    Ok(())
}

fn resolve_roots(roots: Vec<PathBuf>) -> Vec<PathBuf> {
    if !roots.is_empty() {
        return roots;
    }
    vec![
        PathBuf::from("www/assets/models/MIDI-3D"),
        PathBuf::from("www/assets/models/RMBG-1.4"),
        PathBuf::from("www/assets/models/TRELLIS.2-4B"),
        PathBuf::from("www/assets/models/TRELLIS-image-large"),
    ]
}

fn collect_parts_manifests(
    root: &Path,
    out: &mut Vec<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        let metadata = entry.metadata()?;
        if metadata.is_dir() {
            collect_parts_manifests(path.as_path(), out)?;
            continue;
        }
        if !metadata.is_file() {
            continue;
        }
        if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(".bpk.parts.json"))
        {
            out.push(path);
        }
    }
    Ok(())
}

fn collect_primary_burnpacks(
    root: &Path,
    out: &mut Vec<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        let metadata = entry.metadata()?;
        if metadata.is_dir() {
            collect_primary_burnpacks(path.as_path(), out)?;
            continue;
        }
        if !metadata.is_file() {
            continue;
        }
        if path.extension().and_then(|ext| ext.to_str()) != Some("bpk") {
            continue;
        }
        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("");
        if file_name.ends_with(".bpk.meta.json") || file_name.contains(".part-") {
            continue;
        }
        out.push(path);
    }
    Ok(())
}

fn validate_precision_pairs(burnpacks: &[PathBuf]) -> Result<(), Box<dyn std::error::Error>> {
    #[derive(Default)]
    struct PairState {
        has_f32: bool,
        has_f16: bool,
        low_precision: bool,
        redundant_low_precision_f16: bool,
    }

    let mut by_component: BTreeMap<String, PairState> = BTreeMap::new();
    for path in burnpacks {
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let Some(stem) = file_name.strip_suffix(".bpk") else {
            continue;
        };
        let (component_stem, is_f16) = if let Some(base) = stem.strip_suffix("_f16") {
            (base, true)
        } else {
            (stem, false)
        };
        let is_low_precision = stem_has_low_precision_marker(component_stem);
        let component_key = path
            .parent()
            .map(|parent| parent.join(component_stem))
            .unwrap_or_else(|| PathBuf::from(component_stem))
            .display()
            .to_string();
        let state = by_component.entry(component_key).or_default();
        state.low_precision = state.low_precision || is_low_precision;
        if is_f16 {
            state.has_f16 = true;
            if is_low_precision {
                state.redundant_low_precision_f16 = true;
            }
        } else {
            state.has_f32 = true;
        }
    }

    let redundant_low_precision = by_component
        .iter()
        .filter_map(|(component, state)| {
            state
                .redundant_low_precision_f16
                .then_some(component.to_string())
        })
        .collect::<Vec<_>>();
    if !redundant_low_precision.is_empty() {
        return Err(format!(
            "redundant low-precision variants detected (remove *_bf16_f16/*_fp16_f16 artifacts): {}",
            redundant_low_precision.join(", ")
        )
        .into());
    }

    let missing_pairs = by_component
        .into_iter()
        .filter_map(|(component, state)| {
            if state.low_precision {
                if state.has_f32 {
                    None
                } else {
                    Some(format!(
                        "{component} (native low-precision missing canonical .bpk)"
                    ))
                }
            } else if state.has_f32 && state.has_f16 {
                None
            } else {
                Some(format!(
                    "{component} (f32={}, f16={})",
                    state.has_f32, state.has_f16
                ))
            }
        })
        .collect::<Vec<_>>();

    if missing_pairs.is_empty() {
        return Ok(());
    }

    Err(format!(
        "missing paired f32/f16 burnpacks for component(s): {}",
        missing_pairs.join(", ")
    )
    .into())
}

fn detect_oversized_parts_manifest(
    manifest_path: &Path,
    max_part_size_bytes: u64,
) -> Result<Option<OversizedManifestIssue>, Box<dyn std::error::Error>> {
    if !manifest_path.exists() {
        return Ok(None);
    }
    let manifest = read_parts_manifest(manifest_path)?;
    if manifest.parts.is_empty() {
        return Ok(Some(OversizedManifestIssue {
            reason: "manifest has zero parts".to_string(),
            unsplittable_single_tensor_blob: false,
        }));
    }

    let unsplittable_single_tensor_blob = manifest.parts.len() == 1
        && manifest
            .parts
            .first()
            .is_some_and(|entry| entry.tensors <= 1 && entry.bytes == manifest.total_bytes);

    if manifest.max_part_bytes > max_part_size_bytes {
        return Ok(Some(OversizedManifestIssue {
            reason: format!(
                "manifest.max_part_bytes={} exceeds cap={}",
                manifest.max_part_bytes, max_part_size_bytes
            ),
            unsplittable_single_tensor_blob,
        }));
    }

    if let Some(entry) = manifest
        .parts
        .iter()
        .find(|entry| entry.bytes > max_part_size_bytes)
    {
        return Ok(Some(OversizedManifestIssue {
            reason: format!(
                "part '{}' bytes={} exceeds cap={}",
                entry.path, entry.bytes, max_part_size_bytes
            ),
            unsplittable_single_tensor_blob,
        }));
    }

    Ok(None)
}

fn burnpack_path_from_manifest_path(
    manifest_path: &Path,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let manifest = read_parts_manifest(manifest_path)?;
    let mut source_file = manifest.source_file.trim().to_string();
    if source_file.is_empty() {
        let file_name = manifest_path
            .file_name()
            .and_then(|value| value.to_str())
            .ok_or_else(|| format!("invalid parts manifest path '{}'", manifest_path.display()))?;
        source_file = file_name.trim_end_matches(".parts.json").to_string();
    }
    let source_file_name = Path::new(source_file.as_str())
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| {
            format!(
                "invalid source_file '{}' in manifest {}",
                source_file,
                manifest_path.display()
            )
        })?;
    Ok(manifest_path.with_file_name(source_file_name))
}

fn normalize_low_precision_metadata_precision(
    burnpack_path: &Path,
    dry_run: bool,
) -> Result<bool, Box<dyn std::error::Error>> {
    let Some(file_name) = burnpack_path.file_name().and_then(|value| value.to_str()) else {
        return Ok(false);
    };
    let Some(stem) = file_name.strip_suffix(".bpk") else {
        return Ok(false);
    };
    if !stem_has_low_precision_marker(stem) {
        return Ok(false);
    }
    let metadata_path = burnpack_path.with_file_name(format!("{file_name}.meta.json"));
    if !metadata_path.exists() {
        return Ok(false);
    }

    let mut json = serde_json::from_slice::<serde_json::Value>(&fs::read(&metadata_path)?)?;
    let current_precision = json
        .get("precision")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    if current_precision != "f32" {
        return Ok(false);
    }

    if dry_run {
        println!(
            "[ARTIFACTS][DRY RUN] would normalize low-precision metadata '{}' precision f32 -> f16",
            metadata_path.display()
        );
        return Ok(true);
    }

    json["precision"] = serde_json::Value::String("f16".to_string());
    fs::write(&metadata_path, serde_json::to_vec_pretty(&json)?)?;
    println!(
        "[ARTIFACTS] normalized low-precision metadata '{}' precision f32 -> f16",
        metadata_path.display()
    );
    Ok(true)
}

fn directory_size_bytes(root: &Path) -> Result<u64, Box<dyn std::error::Error>> {
    let mut total = 0u64;
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        let metadata = entry.metadata()?;
        if metadata.is_dir() {
            total = total.saturating_add(directory_size_bytes(path.as_path())?);
            continue;
        }
        if metadata.is_file() {
            total = total.saturating_add(metadata.len());
        }
    }
    Ok(total)
}

fn gib_to_bytes(gib: u64) -> u64 {
    gib.saturating_mul(1024 * 1024 * 1024)
}

fn bytes_to_gib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

#[cfg(test)]
mod tests {
    use super::{
        detect_oversized_parts_manifest, ensure_triposg_metadata_aliases, resolve_roots,
        validate_precision_pairs,
    };
    use burn_synth_import::parts::{BurnpackPartEntry, BurnpackPartsManifest};
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_tmp_dir() -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("ensure_web_artifacts_test_{nanos}"))
    }

    #[test]
    fn creates_dinov2_metadata_aliases_from_legacy_paths() {
        let root = unique_tmp_dir();
        fs::create_dir_all(root.join("image_encoder_2")).expect("create image_encoder_2");
        fs::create_dir_all(root.join("feature_extractor_2")).expect("create feature_extractor_2");
        fs::write(
            root.join("image_encoder_2/config.json"),
            br#"{"test":"image_encoder_2"}"#,
        )
        .expect("write legacy encoder config");
        fs::write(
            root.join("feature_extractor_2/preprocessor_config.json"),
            br#"{"test":"feature_extractor_2"}"#,
        )
        .expect("write legacy preprocessor config");

        ensure_triposg_metadata_aliases(&root, false).expect("ensure aliases");

        assert!(
            root.join("image_encoder_dinov2/config.json").exists(),
            "expected image_encoder_dinov2/config.json alias to exist"
        );
        assert!(
            root.join("feature_extractor_dinov2/preprocessor_config.json")
                .exists(),
            "expected feature_extractor_dinov2/preprocessor_config.json alias to exist"
        );

        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn preserves_existing_dinov2_metadata_files() {
        let root = unique_tmp_dir();
        fs::create_dir_all(root.join("image_encoder_dinov2")).expect("create image_encoder_dinov2");
        fs::create_dir_all(root.join("image_encoder_2")).expect("create image_encoder_2");
        let dedicated = br#"{"test":"dedicated"}"#;
        let legacy = br#"{"test":"legacy"}"#;
        fs::write(root.join("image_encoder_dinov2/config.json"), dedicated)
            .expect("write dedicated config");
        fs::write(root.join("image_encoder_2/config.json"), legacy).expect("write legacy config");

        ensure_triposg_metadata_aliases(&root, false).expect("ensure aliases");

        let bytes = fs::read(root.join("image_encoder_dinov2/config.json"))
            .expect("read dedicated config after ensure");
        assert_eq!(
            bytes, dedicated,
            "expected existing dedicated config to be preserved"
        );

        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn default_roots_include_trellis_bundles() {
        let roots = resolve_roots(Vec::new());
        assert!(
            roots
                .iter()
                .any(|path| path == Path::new("www/assets/models/TRELLIS.2-4B"))
        );
        assert!(
            roots
                .iter()
                .any(|path| path == Path::new("www/assets/models/TRELLIS-image-large"))
        );
    }

    #[test]
    fn detects_oversized_parts_manifest_entries() {
        let root = unique_tmp_dir();
        fs::create_dir_all(&root).expect("create root");
        let manifest_path = root.join("model.bpk.parts.json");
        let manifest = BurnpackPartsManifest {
            version: 1,
            source_file: "model.bpk".to_string(),
            source_modified_unix_ms: 0,
            total_bytes: 4096,
            max_part_bytes: 4096,
            parts: vec![BurnpackPartEntry {
                path: "model.bpk.part-00000.bpk".to_string(),
                bytes: 4096,
                sha256: String::new(),
                tensors: 1,
            }],
        };
        fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).expect("serialize manifest"),
        )
        .expect("write manifest");

        let issue = detect_oversized_parts_manifest(&manifest_path, 1024)
            .expect("detect issue")
            .expect("expected oversize issue");
        assert!(issue.reason.contains("exceeds cap"));
        assert!(issue.unsplittable_single_tensor_blob);

        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn precision_validation_allows_low_precision_singletons() {
        let burnpacks = vec![
            PathBuf::from("ckpts/shape_bf16.bpk"),
            PathBuf::from("facebook/dinov3/model.bpk"),
            PathBuf::from("facebook/dinov3/model_f16.bpk"),
        ];
        validate_precision_pairs(&burnpacks).expect("low-precision singleton should pass");
    }

    #[test]
    fn precision_validation_rejects_redundant_low_precision_f16_suffix() {
        let burnpacks = vec![
            PathBuf::from("ckpts/shape_bf16.bpk"),
            PathBuf::from("ckpts/shape_bf16_f16.bpk"),
        ];
        let err = validate_precision_pairs(&burnpacks).expect_err("expected validation error");
        let message = err.to_string();
        assert!(
            message.contains("redundant low-precision variants"),
            "unexpected error message: {message}"
        );
    }
}

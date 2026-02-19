use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use burn_synth_import::parts::{
    burnpack_parts_manifest_path, remove_legacy_shard_artifacts_for_burnpack,
    write_burnpack_parts_for_wasm,
};
use clap::Parser;

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
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let roots = resolve_roots(args.roots);
    let mut burnpacks = Vec::new();
    for root in &roots {
        if !root.exists() {
            return Err(format!("root does not exist: {}", root.display()).into());
        }
        collect_primary_burnpacks(root, &mut burnpacks)?;
    }
    burnpacks.sort();
    burnpacks.dedup();

    println!(
        "[ARTIFACTS] discovered {} burnpack(s) across {} root(s)",
        burnpacks.len(),
        roots.len()
    );
    if burnpacks.is_empty() {
        return Ok(());
    }

    if !args.allow_unpaired_precision {
        validate_precision_pairs(&burnpacks)?;
    }

    let part_size_mib = args.part_size_mib.max(1);

    let mut parts_manifest_count = 0usize;
    let mut part_file_count = 0usize;
    let mut removed_legacy_shard_count = 0usize;
    for burnpack in &burnpacks {
        if args.dry_run {
            println!("[ARTIFACTS][DRY RUN] {}", burnpack.display());
            if !args.keep_legacy_shards {
                println!(
                    "[ARTIFACTS][DRY RUN] would prune legacy shard artifacts for {}",
                    burnpack.display()
                );
            }
            continue;
        }

        if let Some(parts_report) =
            write_burnpack_parts_for_wasm(burnpack, part_size_mib, args.overwrite)?
        {
            parts_manifest_count += 1;
            part_file_count += parts_report.part_paths.len();
        }

        let parts_manifest = burnpack_parts_manifest_path(burnpack);
        if !parts_manifest.exists() {
            return Err(format!(
                "missing parts manifest after generation: {}",
                parts_manifest.display()
            )
            .into());
        }

        if !args.keep_legacy_shards {
            removed_legacy_shard_count += remove_legacy_shard_artifacts_for_burnpack(burnpack)?;
        }
    }

    if args.dry_run {
        println!("[ARTIFACTS][DRY RUN] complete");
        return Ok(());
    }

    println!(
        "[ARTIFACTS] generated/validated {} parts manifest(s), {} part file(s), removed {} legacy shard artifact(s)",
        parts_manifest_count, part_file_count, removed_legacy_shard_count
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
    ]
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
        let component_key = path
            .parent()
            .map(|parent| parent.join(component_stem))
            .unwrap_or_else(|| PathBuf::from(component_stem))
            .display()
            .to_string();
        let state = by_component.entry(component_key).or_default();
        if is_f16 {
            state.has_f16 = true;
        } else {
            state.has_f32 = true;
        }
    }

    let missing_pairs = by_component
        .into_iter()
        .filter_map(|(component, state)| {
            if state.has_f32 && state.has_f16 {
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

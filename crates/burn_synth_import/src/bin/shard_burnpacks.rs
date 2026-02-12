use std::fs;
use std::path::{Path, PathBuf};

use burn_synth_import::plan::ArtifactPolicy;
use burn_synth_import::shard::apply_artifact_policy;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(
    about = "Backfill burnpack shard manifests/shards for existing .bpk files",
    version
)]
struct Args {
    /// One or more directories to scan recursively for .bpk files.
    /// Defaults to the three crate asset model roots.
    #[arg(long = "root")]
    roots: Vec<PathBuf>,

    /// Shard size in MiB.
    #[arg(long, default_value_t = 64)]
    shard_size_mib: u64,

    /// Overwrite existing shard manifests/shards when present.
    #[arg(long)]
    overwrite: bool,

    /// Print planned work without writing shards.
    #[arg(long)]
    dry_run: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let roots = resolve_roots(args.roots);
    let mut burnpacks = Vec::new();
    for root in &roots {
        if !root.exists() {
            return Err(format!("root does not exist: {}", root.display()).into());
        }
        collect_burnpacks(root, &mut burnpacks)?;
    }

    burnpacks.sort();
    burnpacks.dedup();

    println!(
        "[SHARD] discovered {} burnpacks across {} root(s)",
        burnpacks.len(),
        roots.len()
    );
    if burnpacks.is_empty() {
        return Ok(());
    }

    if args.dry_run {
        for path in &burnpacks {
            println!("[SHARD][DRY RUN] {}", path.display());
        }
        return Ok(());
    }

    let policy = ArtifactPolicy::Both {
        shard_size_mib: args.shard_size_mib.max(1),
    };

    let mut manifest_count = 0usize;
    let mut shard_count = 0usize;
    for burnpack in &burnpacks {
        if let Some(report) = apply_artifact_policy(burnpack, policy, args.overwrite)? {
            manifest_count += 1;
            shard_count += report.shard_paths.len();
            println!(
                "[SHARD] {} -> {} ({} shards, {:.1} MiB)",
                burnpack.display(),
                report.manifest_path.display(),
                report.shard_paths.len(),
                report.total_bytes as f64 / (1024.0 * 1024.0),
            );
        }
    }

    println!(
        "[SHARD] generated {} manifest(s), {} shard file(s)",
        manifest_count, shard_count
    );
    Ok(())
}

fn resolve_roots(roots: Vec<PathBuf>) -> Vec<PathBuf> {
    if !roots.is_empty() {
        return roots;
    }
    vec![
        PathBuf::from("crates/burn_tripo/assets/models"),
        PathBuf::from("crates/burn_foreground/assets/models"),
        PathBuf::from("crates/burn_trellis/assets/models"),
    ]
}

fn collect_burnpacks(
    root: &Path,
    out: &mut Vec<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        let metadata = entry.metadata()?;
        if metadata.is_dir() {
            collect_burnpacks(path.as_path(), out)?;
            continue;
        }
        if !metadata.is_file() {
            continue;
        }
        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("");
        if path.extension().and_then(|ext| ext.to_str()) == Some("bpk")
            && !file_name.ends_with(".bpk.meta.json")
        {
            out.push(path);
        }
    }
    Ok(())
}

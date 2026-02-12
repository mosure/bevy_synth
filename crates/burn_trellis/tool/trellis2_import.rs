#![recursion_limit = "256"]

use std::path::PathBuf;

use burn_synth_import::plan::ArtifactPolicy;
use burn_synth_import::shard::apply_artifact_policy;
use burn_trellis::import::{QuantizationMode, TrellisImportOptions, import_trellis2_assets};
use burn_trellis::paths::{
    resolve_trellis2_weights_root, trellis2_repo_asset_root, trellis2_repo_image_large_root,
};
use clap::{Parser, ValueEnum};

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum Quantization {
    F32,
    F16,
    Both,
}

impl From<Quantization> for QuantizationMode {
    fn from(value: Quantization) -> Self {
        match value {
            Quantization::F32 => Self::F32,
            Quantization::F16 => Self::F16,
            Quantization::Both => Self::Both,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum ArtifactPolicyArg {
    SingleFile,
    Sharded,
    Both,
}

impl ArtifactPolicyArg {
    fn into_policy(self, shard_size_mib: u64) -> ArtifactPolicy {
        let shard_size_mib = shard_size_mib.max(1);
        match self {
            Self::SingleFile => ArtifactPolicy::SingleFile,
            Self::Sharded => ArtifactPolicy::Sharded { shard_size_mib },
            Self::Both => ArtifactPolicy::Both { shard_size_mib },
        }
    }
}

#[derive(Parser, Debug)]
#[command(
    about = "Import Trellis2 safetensors and config assets into Burnpack (.bpk) files",
    version
)]
struct Args {
    #[arg(long)]
    weights_root: Option<PathBuf>,

    #[arg(long)]
    image_large_root: Option<PathBuf>,

    #[arg(long)]
    output_root: Option<PathBuf>,

    #[arg(long)]
    image_large_output_root: Option<PathBuf>,

    #[arg(long, value_enum, default_value_t = Quantization::Both)]
    quantization: Quantization,

    #[arg(long)]
    overwrite: bool,

    #[arg(long, value_enum, default_value_t = ArtifactPolicyArg::Both)]
    artifact_policy: ArtifactPolicyArg,

    #[arg(long, default_value_t = 64)]
    shard_size_mib: u64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let weights_root = resolve_trellis2_weights_root(args.weights_root.as_deref());
    let image_large_root = Some(
        args.image_large_root
            .unwrap_or_else(trellis2_repo_image_large_root),
    );
    let output_root = args.output_root.unwrap_or_else(trellis2_repo_asset_root);
    let image_large_output_root = Some(
        args.image_large_output_root
            .unwrap_or_else(trellis2_repo_image_large_root),
    );
    let image_large_root_display = image_large_root
        .as_ref()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "<none>".to_string());
    let image_large_output_root_display = image_large_output_root
        .as_ref()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "<none>".to_string());

    let report = import_trellis2_assets(&TrellisImportOptions {
        weights_root: weights_root.clone(),
        image_large_root: image_large_root.clone(),
        output_root: output_root.clone(),
        image_large_output_root: image_large_output_root.clone(),
        quantization: args.quantization.into(),
        overwrite: args.overwrite,
    })?;
    let artifact_policy = args.artifact_policy.into_policy(args.shard_size_mib);
    let mut shard_manifest_count = 0usize;
    let mut shard_count = 0usize;
    if artifact_policy.wants_shards() {
        for imported in &report.manifest.imported_blobs {
            let output_path = PathBuf::from(&imported.output);
            if let Some(shard_report) =
                apply_artifact_policy(&output_path, artifact_policy, args.overwrite)?
            {
                shard_manifest_count += 1;
                shard_count += shard_report.shard_paths.len();
                println!(
                    "[IMPORT] sharded {} -> {} ({} shards, {:.1} MiB)",
                    output_path.display(),
                    shard_report.manifest_path.display(),
                    shard_report.shard_paths.len(),
                    shard_report.total_bytes as f64 / (1024.0 * 1024.0),
                );
            }
        }
    }

    println!(
        "[IMPORT] Trellis2 roots: weights='{}', output='{}', image_large_source='{}', image_large_output='{}'",
        weights_root.display(),
        output_root.display(),
        image_large_root_display,
        image_large_output_root_display
    );
    println!(
        "[IMPORT] copied json: {}, imported burnpacks: {}, missing sources: {}",
        report.manifest.copied_json_files.len(),
        report.manifest.imported_blobs.len(),
        report.manifest.missing_sources.len()
    );
    if artifact_policy.wants_shards() {
        println!(
            "[IMPORT] generated shard manifests: {}, total shards: {}",
            shard_manifest_count, shard_count
        );
    }
    if !report.manifest.missing_sources.is_empty() {
        println!("[IMPORT] Missing sources:");
        for source in &report.manifest.missing_sources {
            println!("  - {source}");
        }
        return Err("trellis2_import failed: missing required source files".into());
    }
    println!("[IMPORT] manifest: {}", report.manifest_path.display());

    Ok(())
}

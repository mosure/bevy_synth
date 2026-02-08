#![recursion_limit = "256"]

use std::path::PathBuf;

use burn_trellis::import::{QuantizationMode, TrellisImportOptions, import_trellis2_assets};
use burn_trellis::paths::{resolve_trellis2_image_large_root, resolve_trellis2_weights_root};
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

    #[arg(long, value_enum, default_value_t = Quantization::Both)]
    quantization: Quantization,

    #[arg(long)]
    overwrite: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let weights_root = resolve_trellis2_weights_root(args.weights_root.as_deref());
    let image_large_root = args
        .image_large_root
        .as_deref()
        .map(|path| resolve_trellis2_image_large_root(Some(path)));
    let output_root = args.output_root.unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/models/TRELLIS.2-4B")
    });

    let report = import_trellis2_assets(&TrellisImportOptions {
        weights_root: weights_root.clone(),
        image_large_root,
        output_root: output_root.clone(),
        quantization: args.quantization.into(),
        overwrite: args.overwrite,
    })?;

    println!(
        "[IMPORT] Trellis2 roots: weights='{}', output='{}'",
        weights_root.display(),
        output_root.display()
    );
    println!(
        "[IMPORT] copied json: {}, imported burnpacks: {}, missing sources: {}",
        report.manifest.copied_json_files.len(),
        report.manifest.imported_blobs.len(),
        report.manifest.missing_sources.len()
    );
    if !report.manifest.missing_sources.is_empty() {
        println!("[IMPORT] Missing sources:");
        for source in &report.manifest.missing_sources {
            println!("  - {source}");
        }
    }
    println!("[IMPORT] manifest: {}", report.manifest_path.display());

    Ok(())
}

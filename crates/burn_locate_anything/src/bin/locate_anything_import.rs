use std::path::PathBuf;

use burn_locate_anything::import::{
    LocateAnythingImportConfig, LocateAnythingPrecision, write_import_manifest,
};
use clap::{Parser, ValueEnum};

#[derive(Parser, Debug)]
#[command(
    name = "locate_anything_import",
    about = "Prepare a LocateAnything CDN-ready metadata + sharded blob-burnpack bundle"
)]
struct Args {
    #[arg(long)]
    hf_root: PathBuf,
    #[arg(long)]
    output_dir: PathBuf,
    #[arg(long, default_value = "nvidia/LocateAnything-3B")]
    model_id: String,
    #[arg(long, value_enum, default_value_t = PrecisionArg::Bf16)]
    precision: PrecisionArg,
    #[arg(long)]
    shard_size_mib: Option<usize>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum PrecisionArg {
    F32,
    F16,
    Bf16,
}

impl From<PrecisionArg> for LocateAnythingPrecision {
    fn from(value: PrecisionArg) -> Self {
        match value {
            PrecisionArg::F32 => Self::F32,
            PrecisionArg::F16 => Self::F16,
            PrecisionArg::Bf16 => Self::Bf16,
        }
    }
}

fn main() {
    let args = Args::parse();
    match write_import_manifest(&LocateAnythingImportConfig {
        hf_root: args.hf_root,
        output_dir: args.output_dir,
        model_id: args.model_id,
        precision: args.precision.into(),
        shard_size_mib: args.shard_size_mib,
    }) {
        Ok(manifest) => {
            println!("{}", serde_json::to_string_pretty(&manifest).unwrap());
        }
        Err(err) => {
            eprintln!("locate_anything_import error: {err}");
            std::process::exit(1);
        }
    }
}

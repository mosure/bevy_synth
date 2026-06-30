use std::path::PathBuf;

use burn_segmentation::SegmentationModelKind;
use burn_segmentation::config::{SegmentationPrecision, SegmentationQuantization};
use burn_segmentation::import::{SegmentationImportConfig, write_import_manifest};
use clap::{Parser, ValueEnum};

#[derive(Parser, Debug)]
#[command(
    name = "segmentation_import",
    about = "Prepare a segmentation import manifest and burnpack artifact plan"
)]
struct Args {
    #[arg(long)]
    hf_root: PathBuf,
    #[arg(long)]
    output_dir: PathBuf,
    #[arg(long, default_value = "facebook/sam2")]
    model_id: String,
    #[arg(long, value_enum)]
    model: ModelArg,
    #[arg(long, value_enum, default_value_t = PrecisionArg::F16)]
    precision: PrecisionArg,
    #[arg(long, value_enum, default_value_t = QuantizationArg::None)]
    quantization: QuantizationArg,
    #[arg(long)]
    shard_size_mib: Option<usize>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ModelArg {
    Sam2,
    Sam3,
}

impl From<ModelArg> for SegmentationModelKind {
    fn from(value: ModelArg) -> Self {
        match value {
            ModelArg::Sam2 => Self::Sam2,
            ModelArg::Sam3 => Self::Sam3,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum PrecisionArg {
    F32,
    F16,
    Bf16,
}

impl From<PrecisionArg> for SegmentationPrecision {
    fn from(value: PrecisionArg) -> Self {
        match value {
            PrecisionArg::F32 => Self::F32,
            PrecisionArg::F16 => Self::F16,
            PrecisionArg::Bf16 => Self::Bf16,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum QuantizationArg {
    None,
    Q8,
    Q4,
}

impl From<QuantizationArg> for SegmentationQuantization {
    fn from(value: QuantizationArg) -> Self {
        match value {
            QuantizationArg::None => Self::None,
            QuantizationArg::Q8 => Self::Q8,
            QuantizationArg::Q4 => Self::Q4,
        }
    }
}

fn main() {
    let args = Args::parse();
    match write_import_manifest(&SegmentationImportConfig {
        hf_root: args.hf_root,
        output_dir: args.output_dir,
        model_id: args.model_id,
        model: args.model.into(),
        precision: args.precision.into(),
        quantization: args.quantization.into(),
        shard_size_mib: args.shard_size_mib,
    }) {
        Ok(manifest) => println!("{}", serde_json::to_string_pretty(&manifest).unwrap()),
        Err(err) => {
            eprintln!("segmentation_import error: {err}");
            std::process::exit(1);
        }
    }
}

#![recursion_limit = "256"]

use std::path::PathBuf;

use burn_trellis::import::extract_blob_burnpack;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(
    about = "Extract a Trellis2 blob BurnPack payload back to its original bytes",
    version
)]
struct Args {
    /// Input blob .bpk file.
    #[arg(long)]
    input: PathBuf,

    /// Output path for the extracted payload.
    #[arg(long)]
    output: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let bytes = extract_blob_burnpack(&args.input, &args.output)?;
    println!(
        "[EXTRACT] {} -> {} ({} bytes)",
        args.input.display(),
        args.output.display(),
        bytes
    );
    Ok(())
}

use std::path::PathBuf;

use burn_segmentation::{compare_parity_fixture, read_parity_fixture, write_parity_summary};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(
    name = "segmentation_parity",
    about = "Compare Burn segmentation masks against a Python reference fixture"
)]
struct Args {
    #[arg(long)]
    fixture: PathBuf,
    #[arg(long)]
    output: Option<PathBuf>,
}

fn main() {
    let args = Args::parse();
    let result = read_parity_fixture(&args.fixture).and_then(|fixture| {
        let summary = compare_parity_fixture(&fixture)?;
        if let Some(output) = args.output.as_ref() {
            write_parity_summary(output, &summary)?;
        }
        Ok(summary)
    });
    match result {
        Ok(summary) => {
            println!("{}", serde_json::to_string_pretty(&summary).unwrap());
            if !summary.passed {
                std::process::exit(2);
            }
        }
        Err(err) => {
            eprintln!("segmentation_parity error: {err}");
            std::process::exit(1);
        }
    }
}

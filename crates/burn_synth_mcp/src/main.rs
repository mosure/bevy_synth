use clap::Parser;

use burn_synth_mcp::{ServerArgs, run_from_args};

fn main() {
    let args = ServerArgs::parse();
    if let Err(err) = run_from_args(args) {
        eprintln!("burn_synth_mcp error: {err}");
        std::process::exit(1);
    }
}

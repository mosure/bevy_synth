use std::process::ExitCode;

use burn_synth_scene::{Cli, run_cli};
use clap::Parser;

fn main() -> ExitCode {
    match run_cli(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("burn_synth_scene error: {err}");
            ExitCode::FAILURE
        }
    }
}

use clap::Parser;

use burn_synth_mcp::{ServerArgs, run_from_args};

fn main() {
    let result = std::thread::Builder::new()
        .name("burn_synth_mcp".to_string())
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            let args = ServerArgs::parse();
            run_from_args(args)
        })
        .expect("failed to spawn burn_synth_mcp main thread")
        .join();

    match result {
        Ok(Ok(())) => {}
        Ok(Err(err)) => {
            eprintln!("burn_synth_mcp error: {err}");
            std::process::exit(1);
        }
        Err(payload) => std::panic::resume_unwind(payload),
    }
}

use std::fs;
use std::path::PathBuf;

use bevy_args::{Deserialize, Parser, Serialize, parse_args};
use safetensors::SafeTensors;

#[derive(Parser, Serialize, Deserialize, Debug)]
#[command(about = "List keys in a safetensors file", version, long_about = None)]
struct Args {
    /// Path to the safetensors file.
    path: PathBuf,

    /// Optional substring filter for keys.
    #[arg(long)]
    contains: Option<String>,

    /// Maximum number of keys to print.
    #[arg(long)]
    limit: Option<usize>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args::<Args>();
    let bytes = fs::read(&args.path)?;
    let safetensors = SafeTensors::deserialize(&bytes)?;
    let mut keys = safetensors.names();
    keys.sort_unstable();

    let keys = if let Some(filter) = &args.contains {
        keys.into_iter()
            .filter(|name| name.contains(filter))
            .collect::<Vec<_>>()
    } else {
        keys
    };

    let total = keys.len();
    let limit = args.limit.unwrap_or(total);
    for name in keys.iter().take(limit) {
        println!("{name}");
    }

    if limit < total {
        println!("... ({}) more", total - limit);
    } else {
        println!("total keys: {total}");
    }

    Ok(())
}

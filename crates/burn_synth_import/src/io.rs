use std::fs;
use std::io::{BufReader, Read};
use std::path::Path;

use sha2::{Digest, Sha256};

pub fn ensure_parent_dir(path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

pub fn copy_if_needed(source: &Path, destination: &Path, overwrite: bool) -> std::io::Result<bool> {
    if source == destination {
        return Ok(false);
    }
    if destination.exists() && !overwrite {
        return Ok(false);
    }
    ensure_parent_dir(destination)?;
    let _ = fs::copy(source, destination)?;
    Ok(true)
}

pub fn sha256_bytes(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

pub fn sha256_file(path: &Path) -> std::io::Result<String> {
    let file = fs::File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex::encode(hasher.finalize()))
}

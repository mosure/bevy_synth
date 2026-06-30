use std::collections::BTreeMap;
use std::fs;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{LocateAnythingError, LocateAnythingResult};

const HEADER_SIZE: usize = 10;
const MAGIC_NUMBER: u32 = 0x4255_524E;
const VERSION: u16 = 1;
const TENSOR_ALIGNMENT: u64 = 256;
const DEFAULT_CHUNK_BYTES: u64 = 63 * 1024 * 1024;
const BLOB_CHUNK_PREFIX: &str = "blob_chunks.";

#[derive(Debug, Clone, Deserialize, Serialize)]
struct BlobBurnpackMetadata {
    tensors: BTreeMap<String, BlobTensorDescriptor>,
    #[serde(default)]
    metadata: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct BlobTensorDescriptor {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: (u64, u64),
}

#[derive(Debug, Clone)]
struct BlobSegment {
    ordinal: usize,
    start: u64,
    end: u64,
}

#[derive(Debug, Deserialize)]
struct PartsManifest {
    #[serde(default)]
    source_file: String,
    #[serde(default)]
    parts: Vec<PartEntry>,
}

#[derive(Debug, Deserialize)]
struct PartEntry {
    path: String,
    #[serde(default)]
    bytes: u64,
}

pub fn default_blob_chunk_bytes(max_part_size_mib: Option<usize>) -> u64 {
    let Some(max_part_size_mib) = max_part_size_mib else {
        return DEFAULT_CHUNK_BYTES;
    };
    let bytes = (max_part_size_mib.max(1) as u64).saturating_mul(1024 * 1024);
    bytes.saturating_sub(1024 * 1024).max(1024 * 1024)
}

pub fn write_blob_burnpack_from_file(
    source_path: &Path,
    burnpack_path: &Path,
    chunk_bytes: u64,
    overwrite: bool,
) -> LocateAnythingResult<bool> {
    if burnpack_path.exists() && !overwrite {
        return Ok(false);
    }
    let source_len = fs::metadata(source_path)
        .map_err(|err| {
            LocateAnythingError::Io(format!("metadata {}: {err}", source_path.display()))
        })?
        .len();
    if let Some(parent) = burnpack_path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            LocateAnythingError::Io(format!("create {}: {err}", parent.display()))
        })?;
    }

    let chunk_bytes = chunk_bytes.max(1);
    let mut tensors = BTreeMap::new();
    let mut offset = 0u64;
    let mut ordinal = 0usize;
    while offset < source_len || (source_len == 0 && ordinal == 0) {
        let len = if source_len == 0 {
            0
        } else {
            (source_len - offset).min(chunk_bytes)
        };
        let end = offset.saturating_add(len);
        tensors.insert(
            format!("{BLOB_CHUNK_PREFIX}{ordinal}"),
            BlobTensorDescriptor {
                dtype: "U8".to_string(),
                shape: vec![len],
                data_offsets: (offset, end),
            },
        );
        offset = end;
        ordinal += 1;
        if source_len == 0 {
            break;
        }
    }
    let mut metadata = BTreeMap::new();
    metadata.insert(
        "format".to_string(),
        "locate_anything_blob_burnpack".to_string(),
    );
    metadata.insert("source_file".to_string(), source_file_name(source_path)?);
    let metadata = BlobBurnpackMetadata { tensors, metadata };
    let mut metadata_bytes = Vec::new();
    ciborium::ser::into_writer(&metadata, &mut metadata_bytes).map_err(|err| {
        LocateAnythingError::Io(format!(
            "serialize burnpack metadata for {}: {err}",
            burnpack_path.display()
        ))
    })?;
    let data_start = aligned_data_section_start(metadata_bytes.len());
    let mut source = fs::File::open(source_path)
        .map_err(|err| LocateAnythingError::Io(format!("open {}: {err}", source_path.display())))?;
    let mut output = fs::File::create(burnpack_path).map_err(|err| {
        LocateAnythingError::Io(format!("create {}: {err}", burnpack_path.display()))
    })?;
    let mut header = [0u8; HEADER_SIZE];
    header[0..4].copy_from_slice(&MAGIC_NUMBER.to_le_bytes());
    header[4..6].copy_from_slice(&VERSION.to_le_bytes());
    let metadata_len = u32::try_from(metadata_bytes.len()).map_err(|_| {
        LocateAnythingError::Io(format!(
            "burnpack metadata too large for {}: {} bytes",
            burnpack_path.display(),
            metadata_bytes.len()
        ))
    })?;
    header[6..10].copy_from_slice(&metadata_len.to_le_bytes());
    output.write_all(&header).map_err(|err| {
        LocateAnythingError::Io(format!("write {} header: {err}", burnpack_path.display()))
    })?;
    output.write_all(&metadata_bytes).map_err(|err| {
        LocateAnythingError::Io(format!("write {} metadata: {err}", burnpack_path.display()))
    })?;
    let padding = data_start.saturating_sub((HEADER_SIZE + metadata_bytes.len()) as u64);
    if padding > 0 {
        write_zero_padding(&mut output, padding)?;
    }
    io::copy(&mut source, &mut output).map_err(|err| {
        LocateAnythingError::Io(format!(
            "copy {} into {}: {err}",
            source_path.display(),
            burnpack_path.display()
        ))
    })?;
    Ok(true)
}

pub fn extract_blob_burnpack_or_parts_to_file(
    burnpack_path: &Path,
    destination: &Path,
) -> LocateAnythingResult<()> {
    if burnpack_path.exists() {
        return extract_blob_burnpack_to_file(burnpack_path, destination, false);
    }
    let manifest_path = burnpack_parts_manifest_path(burnpack_path);
    extract_blob_burnpack_parts_to_file(&manifest_path, destination)
}

pub fn extract_blob_burnpack_parts_to_file(
    manifest_path: &Path,
    destination: &Path,
) -> LocateAnythingResult<()> {
    let bytes = fs::read(manifest_path).map_err(|err| {
        LocateAnythingError::Io(format!("read {}: {err}", manifest_path.display()))
    })?;
    let manifest = serde_json::from_slice::<PartsManifest>(&bytes).map_err(|err| {
        LocateAnythingError::Config(format!("parse {}: {err}", manifest_path.display()))
    })?;
    if manifest.parts.is_empty() {
        return Err(LocateAnythingError::Config(format!(
            "parts manifest {} contains no parts",
            manifest_path.display()
        )));
    }
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            LocateAnythingError::Io(format!("create {}: {err}", parent.display()))
        })?;
    }
    let tmp = destination.with_extension("tmp");
    let _ = fs::remove_file(&tmp);
    for (index, part) in manifest.parts.iter().enumerate() {
        let part_path = resolve_part_path(manifest_path, &part.path);
        if part.bytes > 0 {
            let actual = fs::metadata(&part_path)
                .map_err(|err| {
                    LocateAnythingError::Io(format!("metadata {}: {err}", part_path.display()))
                })?
                .len();
            if actual != part.bytes {
                return Err(LocateAnythingError::Io(format!(
                    "part {} size mismatch for {}: manifest={} actual={}",
                    index + 1,
                    part_path.display(),
                    part.bytes,
                    actual
                )));
            }
        }
        if !manifest.source_file.trim().is_empty()
            && index == 0
            && destination
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| {
                    name != manifest.source_file.trim().replace(".bpk", ".safetensors")
                })
        {
            // The destination may intentionally use the original safetensor name after extracting
            // from a .bpk source, so this is only an informational consistency check.
        }
        extract_blob_burnpack_to_file(&part_path, &tmp, true)?;
    }
    fs::rename(&tmp, destination).map_err(|err| {
        LocateAnythingError::Io(format!(
            "move {} to {}: {err}",
            tmp.display(),
            destination.display()
        ))
    })
}

pub fn burnpack_parts_manifest_path(burnpack_path: &Path) -> PathBuf {
    let file_name = burnpack_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("model.bpk");
    burnpack_path.with_file_name(format!("{file_name}.parts.json"))
}

fn extract_blob_burnpack_to_file(
    burnpack_path: &Path,
    destination: &Path,
    append: bool,
) -> LocateAnythingResult<()> {
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            LocateAnythingError::Io(format!("create {}: {err}", parent.display()))
        })?;
    }
    let mut input = fs::File::open(burnpack_path).map_err(|err| {
        LocateAnythingError::Io(format!("open {}: {err}", burnpack_path.display()))
    })?;
    let file_len = input
        .metadata()
        .map_err(|err| {
            LocateAnythingError::Io(format!("metadata {}: {err}", burnpack_path.display()))
        })?
        .len();
    let (data_start, metadata) = read_blob_metadata(&mut input, burnpack_path)?;
    let segments = resolve_blob_segments(&metadata)?;
    let mut output = if append {
        fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(destination)
    } else {
        fs::File::create(destination)
    }
    .map_err(|err| LocateAnythingError::Io(format!("open {}: {err}", destination.display())))?;
    let max_end = segments
        .iter()
        .map(|segment| segment.end)
        .max()
        .unwrap_or(0);
    let payload_start = blob_payload_start(data_start, file_len, max_end);
    let mut buffer = vec![0u8; 1024 * 1024];
    for segment in segments {
        if segment.end < segment.start {
            return Err(LocateAnythingError::Config(format!(
                "invalid blob offsets in {}: {}..{}",
                burnpack_path.display(),
                segment.start,
                segment.end
            )));
        }
        let mut remaining = segment.end - segment.start;
        input
            .seek(SeekFrom::Start(payload_start + segment.start))
            .map_err(|err| {
                LocateAnythingError::Io(format!("seek {}: {err}", burnpack_path.display()))
            })?;
        while remaining > 0 {
            let read_len = buffer.len().min(remaining as usize);
            input.read_exact(&mut buffer[..read_len]).map_err(|err| {
                LocateAnythingError::Io(format!("read {} payload: {err}", burnpack_path.display()))
            })?;
            output.write_all(&buffer[..read_len]).map_err(|err| {
                LocateAnythingError::Io(format!("write {}: {err}", destination.display()))
            })?;
            remaining -= read_len as u64;
        }
    }
    Ok(())
}

fn read_blob_metadata(
    input: &mut fs::File,
    path: &Path,
) -> LocateAnythingResult<(u64, BlobBurnpackMetadata)> {
    let mut header = [0u8; HEADER_SIZE];
    input
        .read_exact(&mut header)
        .map_err(|err| LocateAnythingError::Io(format!("read {} header: {err}", path.display())))?;
    let magic = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
    if magic != MAGIC_NUMBER {
        return Err(LocateAnythingError::Config(format!(
            "invalid burnpack magic in {}: expected {MAGIC_NUMBER:#x}, found {magic:#x}",
            path.display()
        )));
    }
    let metadata_size = u32::from_le_bytes([header[6], header[7], header[8], header[9]]) as usize;
    let mut metadata_bytes = vec![0u8; metadata_size];
    input.read_exact(&mut metadata_bytes).map_err(|err| {
        LocateAnythingError::Io(format!("read {} metadata: {err}", path.display()))
    })?;
    let metadata = ciborium::de::from_reader(metadata_bytes.as_slice()).map_err(|err| {
        LocateAnythingError::Config(format!("parse {} burnpack metadata: {err}", path.display()))
    })?;
    Ok((aligned_data_section_start(metadata_size), metadata))
}

fn resolve_blob_segments(
    metadata: &BlobBurnpackMetadata,
) -> LocateAnythingResult<Vec<BlobSegment>> {
    let mut segments = Vec::new();
    for (name, descriptor) in &metadata.tensors {
        if descriptor.dtype != "U8" {
            return Err(LocateAnythingError::Config(format!(
                "LocateAnything blob burnpack tensor {name} has dtype {}; expected U8",
                descriptor.dtype
            )));
        }
        let Some(suffix) = name.strip_prefix(BLOB_CHUNK_PREFIX) else {
            continue;
        };
        let ordinal = suffix
            .split('.')
            .next()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(usize::MAX);
        let len = descriptor.shape.iter().try_fold(1u64, |acc, dim| {
            acc.checked_mul(*dim).ok_or_else(|| {
                LocateAnythingError::Config(format!(
                    "LocateAnything blob tensor {name} shape product overflow"
                ))
            })
        })?;
        if descriptor
            .data_offsets
            .1
            .saturating_sub(descriptor.data_offsets.0)
            != len
        {
            return Err(LocateAnythingError::Config(format!(
                "LocateAnything blob tensor {name} shape {:?} does not match offsets {:?}",
                descriptor.shape, descriptor.data_offsets
            )));
        }
        segments.push(BlobSegment {
            ordinal,
            start: descriptor.data_offsets.0,
            end: descriptor.data_offsets.1,
        });
    }
    if segments.is_empty() {
        return Err(LocateAnythingError::Config(
            "burnpack is not a LocateAnything blob payload".to_string(),
        ));
    }
    segments.sort_by_key(|segment| (segment.ordinal, segment.start));
    Ok(segments)
}

fn aligned_data_section_start(metadata_size: usize) -> u64 {
    let unaligned_start = (HEADER_SIZE + metadata_size) as u64;
    unaligned_start.div_ceil(TENSOR_ALIGNMENT) * TENSOR_ALIGNMENT
}

fn blob_payload_start(data_start: u64, file_len: u64, max_end: u64) -> u64 {
    let inferred_payload_start = file_len.saturating_sub(max_end);
    if inferred_payload_start >= data_start {
        let tail_padding_shift = inferred_payload_start - data_start;
        if tail_padding_shift <= TENSOR_ALIGNMENT {
            return data_start;
        }
        return inferred_payload_start;
    }
    let legacy_alignment_gap = data_start - inferred_payload_start;
    if legacy_alignment_gap <= TENSOR_ALIGNMENT {
        inferred_payload_start
    } else {
        data_start
    }
}

fn resolve_part_path(manifest_path: &Path, part_path: &str) -> PathBuf {
    let part = Path::new(part_path);
    if part.is_absolute() {
        return part.to_path_buf();
    }
    manifest_path.parent().unwrap_or(Path::new(".")).join(part)
}

fn source_file_name(path: &Path) -> LocateAnythingResult<String> {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(str::to_string)
        .ok_or_else(|| {
            LocateAnythingError::Io(format!("invalid source file name: {}", path.display()))
        })
}

fn write_zero_padding(output: &mut fs::File, mut bytes: u64) -> LocateAnythingResult<()> {
    let zeros = [0u8; 4096];
    while bytes > 0 {
        let len = zeros.len().min(bytes as usize);
        output
            .write_all(&zeros[..len])
            .map_err(|err| LocateAnythingError::Io(format!("write burnpack padding: {err}")))?;
        bytes -= len as u64;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_blob_burnpack_roundtrips() {
        let root = std::env::temp_dir().join(format!(
            "burn_locate_anything_blob_bpk_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let source = root.join("source.safetensors");
        let burnpack = root.join("source.bpk");
        let extracted = root.join("extracted.safetensors");
        let bytes = (0..10_000)
            .map(|idx| ((idx * 31 + 7) % 251) as u8)
            .collect::<Vec<_>>();
        fs::write(&source, &bytes).unwrap();
        write_blob_burnpack_from_file(&source, &burnpack, 1024, true).unwrap();
        extract_blob_burnpack_or_parts_to_file(&burnpack, &extracted).unwrap();
        assert_eq!(fs::read(extracted).unwrap(), bytes);
        let _ = fs::remove_dir_all(root);
    }
}

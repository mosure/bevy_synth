use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use ciborium::Value;
use serde::{Deserialize, Serialize};

use crate::io::{ensure_parent_dir, sha256_file};

const ONE_MIB: u64 = 1024 * 1024;
const HEADER_SIZE: usize = 10;
const MAGIC_NUMBER: u32 = 0x4255_524E;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BurnpackPartsManifest {
    #[serde(default = "default_manifest_version")]
    pub version: u32,
    #[serde(default)]
    pub source_file: String,
    #[serde(default)]
    pub total_bytes: u64,
    #[serde(default)]
    pub max_part_bytes: u64,
    #[serde(default)]
    pub parts: Vec<BurnpackPartEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BurnpackPartEntry {
    pub path: String,
    #[serde(default)]
    pub bytes: u64,
    #[serde(default)]
    pub sha256: String,
    #[serde(default)]
    pub tensors: usize,
}

#[derive(Debug, Clone)]
pub struct BurnpackPartsReport {
    pub manifest_path: PathBuf,
    pub part_paths: Vec<PathBuf>,
    pub total_bytes: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct RawBurnpackMetadata {
    tensors: BTreeMap<String, RawTensorDescriptor>,
    #[serde(default)]
    metadata: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct RawTensorDescriptor {
    dtype: Value,
    shape: Vec<u64>,
    data_offsets: (u64, u64),
    #[serde(default, skip_serializing_if = "Option::is_none")]
    param_id: Option<u64>,
}

#[derive(Debug, Clone)]
struct TensorRecord {
    name: String,
    descriptor: RawTensorDescriptor,
}

const fn default_manifest_version() -> u32 {
    1
}

pub fn burnpack_parts_manifest_path(burnpack_path: &Path) -> PathBuf {
    let file_name = burnpack_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("model.bpk");
    burnpack_path.with_file_name(format!("{file_name}.parts.json"))
}

pub fn write_burnpack_parts_for_wasm(
    burnpack_path: &Path,
    max_part_size_mib: u64,
    overwrite: bool,
) -> Result<Option<BurnpackPartsReport>, String> {
    if !burnpack_path.exists() {
        return Err(format!(
            "burnpack does not exist for parting: {}",
            burnpack_path.display()
        ));
    }

    let max_part_bytes = max_part_size_mib
        .max(1)
        .checked_mul(ONE_MIB)
        .ok_or_else(|| "max part size overflow".to_string())?;

    let total_bytes = fs::metadata(burnpack_path)
        .map_err(|err| format!("failed to read {} metadata: {err}", burnpack_path.display()))?
        .len();
    if total_bytes <= max_part_bytes {
        return Ok(None);
    }

    let manifest_path = burnpack_parts_manifest_path(burnpack_path);
    if manifest_path.exists() && !overwrite && manifest_has_all_parts(&manifest_path) {
        let manifest = read_parts_manifest(&manifest_path)?;
        let part_paths = manifest
            .parts
            .iter()
            .map(|entry| resolve_part_entry_path(&manifest_path, &entry.path))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(Some(BurnpackPartsReport {
            manifest_path,
            part_paths,
            total_bytes: manifest.total_bytes,
        }));
    }

    if overwrite {
        cleanup_existing_parts(&manifest_path)?;
    }
    ensure_parent_dir(&manifest_path).map_err(|err| {
        format!(
            "failed to create burnpack parts directory '{}': {err}",
            manifest_path.display()
        )
    })?;

    let mut source = fs::File::open(burnpack_path)
        .map_err(|err| format!("failed to open burnpack {}: {err}", burnpack_path.display()))?;
    let (version, metadata_size, metadata) = read_burnpack_metadata(&mut source, burnpack_path)?;
    let data_start = HEADER_SIZE as u64 + metadata_size as u64;

    let mut tensor_records = metadata
        .tensors
        .iter()
        .map(|(name, descriptor)| TensorRecord {
            name: name.clone(),
            descriptor: descriptor.clone(),
        })
        .collect::<Vec<_>>();
    if tensor_records.is_empty() {
        return Err(format!(
            "burnpack '{}' contains no tensor descriptors",
            burnpack_path.display()
        ));
    }
    tensor_records.sort_by_key(|record| record.descriptor.data_offsets.0);
    let groups = split_tensor_records(tensor_records, max_part_bytes);

    let source_file_name = burnpack_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("invalid burnpack name '{}'", burnpack_path.display()))?;
    let mut part_entries = Vec::with_capacity(groups.len());
    let mut part_paths = Vec::with_capacity(groups.len());
    for (index, group) in groups.iter().enumerate() {
        let part_name = format!("{source_file_name}.part-{index:05}.bpk");
        let part_path = burnpack_path.with_file_name(&part_name);
        if part_path.exists() && overwrite {
            fs::remove_file(&part_path).map_err(|err| {
                format!(
                    "failed to replace stale burnpack part {}: {err}",
                    part_path.display()
                )
            })?;
        }

        write_burnpack_part(
            &mut source,
            &part_path,
            version,
            data_start,
            &metadata.metadata,
            group,
        )?;
        let bytes = fs::metadata(&part_path)
            .map_err(|err| {
                format!(
                    "failed to stat burnpack part {}: {err}",
                    part_path.display()
                )
            })?
            .len();
        let sha256 = sha256_file(&part_path).map_err(|err| {
            format!(
                "failed to hash burnpack part {}: {err}",
                part_path.display()
            )
        })?;
        part_entries.push(BurnpackPartEntry {
            path: part_name,
            bytes,
            sha256,
            tensors: group.len(),
        });
        part_paths.push(part_path);
    }

    let manifest = BurnpackPartsManifest {
        version: default_manifest_version(),
        source_file: source_file_name.to_string(),
        total_bytes,
        max_part_bytes,
        parts: part_entries,
    };
    let manifest_json = serde_json::to_string_pretty(&manifest)
        .map_err(|err| format!("failed to serialize parts manifest: {err}"))?;
    fs::write(&manifest_path, manifest_json).map_err(|err| {
        format!(
            "failed to write burnpack parts manifest {}: {err}",
            manifest_path.display()
        )
    })?;

    Ok(Some(BurnpackPartsReport {
        manifest_path,
        part_paths,
        total_bytes,
    }))
}

fn read_burnpack_metadata(
    source: &mut fs::File,
    burnpack_path: &Path,
) -> Result<(u16, u32, RawBurnpackMetadata), String> {
    source
        .seek(SeekFrom::Start(0))
        .map_err(|err| format!("failed to seek {}: {err}", burnpack_path.display()))?;
    let mut header = [0u8; HEADER_SIZE];
    source.read_exact(&mut header).map_err(|err| {
        format!(
            "failed to read burnpack header {}: {err}",
            burnpack_path.display()
        )
    })?;

    let magic = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
    if magic != MAGIC_NUMBER {
        return Err(format!(
            "invalid burnpack magic in {}: expected {MAGIC_NUMBER:#x}, found {magic:#x}",
            burnpack_path.display()
        ));
    }
    let version = u16::from_le_bytes([header[4], header[5]]);
    let metadata_size = u32::from_le_bytes([header[6], header[7], header[8], header[9]]);
    let mut metadata_bytes = vec![0u8; metadata_size as usize];
    source.read_exact(&mut metadata_bytes).map_err(|err| {
        format!(
            "failed to read burnpack metadata {}: {err}",
            burnpack_path.display()
        )
    })?;
    let metadata = ciborium::de::from_reader(metadata_bytes.as_slice()).map_err(|err| {
        format!(
            "failed to parse burnpack metadata {}: {err}",
            burnpack_path.display()
        )
    })?;
    Ok((version, metadata_size, metadata))
}

fn split_tensor_records(records: Vec<TensorRecord>, max_part_bytes: u64) -> Vec<Vec<TensorRecord>> {
    let mut groups = Vec::new();
    let mut current_group = Vec::new();
    let mut current_bytes = 0u64;

    for record in records {
        let tensor_bytes = record
            .descriptor
            .data_offsets
            .1
            .saturating_sub(record.descriptor.data_offsets.0);
        let would_exceed = !current_group.is_empty()
            && current_bytes.saturating_add(tensor_bytes) > max_part_bytes;
        if would_exceed {
            groups.push(current_group);
            current_group = Vec::new();
            current_bytes = 0;
        }
        current_bytes = current_bytes.saturating_add(tensor_bytes);
        current_group.push(record);
    }
    if !current_group.is_empty() {
        groups.push(current_group);
    }
    groups
}

fn write_burnpack_part(
    source: &mut fs::File,
    destination: &Path,
    version: u16,
    data_start: u64,
    source_metadata: &BTreeMap<String, String>,
    records: &[TensorRecord],
) -> Result<(), String> {
    let mut tensors = BTreeMap::new();
    let mut next_offset = 0u64;
    for record in records {
        let tensor_bytes = record
            .descriptor
            .data_offsets
            .1
            .saturating_sub(record.descriptor.data_offsets.0);
        let mut descriptor = record.descriptor.clone();
        descriptor.data_offsets = (next_offset, next_offset.saturating_add(tensor_bytes));
        next_offset = descriptor.data_offsets.1;
        tensors.insert(record.name.clone(), descriptor);
    }

    let metadata = RawBurnpackMetadata {
        tensors,
        metadata: source_metadata.clone(),
    };
    let mut metadata_bytes = Vec::new();
    ciborium::ser::into_writer(&metadata, &mut metadata_bytes)
        .map_err(|err| format!("failed to serialize burnpack part metadata: {err}"))?;
    let metadata_size = u32::try_from(metadata_bytes.len())
        .map_err(|_| "burnpack part metadata size exceeds u32".to_string())?;

    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
    }
    let mut out = fs::File::create(destination).map_err(|err| {
        format!(
            "failed to create burnpack part {}: {err}",
            destination.display()
        )
    })?;
    let mut header = [0u8; HEADER_SIZE];
    header[0..4].copy_from_slice(&MAGIC_NUMBER.to_le_bytes());
    header[4..6].copy_from_slice(&version.to_le_bytes());
    header[6..10].copy_from_slice(&metadata_size.to_le_bytes());
    out.write_all(&header).map_err(|err| {
        format!(
            "failed to write part header {}: {err}",
            destination.display()
        )
    })?;
    out.write_all(&metadata_bytes).map_err(|err| {
        format!(
            "failed to write part metadata {}: {err}",
            destination.display()
        )
    })?;

    let mut copy_buffer = vec![0u8; 1024 * 1024];
    for record in records {
        let start = record.descriptor.data_offsets.0;
        let end = record.descriptor.data_offsets.1;
        let mut remaining = end.saturating_sub(start);
        source
            .seek(SeekFrom::Start(data_start.saturating_add(start)))
            .map_err(|err| format!("failed to seek source burnpack: {err}"))?;
        while remaining > 0 {
            let chunk = remaining.min(copy_buffer.len() as u64) as usize;
            source
                .read_exact(&mut copy_buffer[..chunk])
                .map_err(|err| format!("failed to read source burnpack tensor bytes: {err}"))?;
            out.write_all(&copy_buffer[..chunk])
                .map_err(|err| format!("failed to write burnpack part tensor bytes: {err}"))?;
            remaining -= chunk as u64;
        }
    }
    out.flush().map_err(|err| {
        format!(
            "failed to flush burnpack part {}: {err}",
            destination.display()
        )
    })?;
    Ok(())
}

fn read_parts_manifest(path: &Path) -> Result<BurnpackPartsManifest, String> {
    let bytes = fs::read(path).map_err(|err| {
        format!(
            "failed to read burnpack parts manifest {}: {err}",
            path.display()
        )
    })?;
    serde_json::from_slice(&bytes).map_err(|err| {
        format!(
            "failed to parse burnpack parts manifest {}: {err}",
            path.display()
        )
    })
}

fn resolve_part_entry_path(manifest_path: &Path, entry_path: &str) -> Result<PathBuf, String> {
    let entry_path = Path::new(entry_path);
    if entry_path.is_absolute() {
        return Ok(entry_path.to_path_buf());
    }
    manifest_path
        .parent()
        .map(|parent| parent.join(entry_path))
        .ok_or_else(|| format!("invalid manifest path '{}'", manifest_path.display()))
}

fn manifest_has_all_parts(path: &Path) -> bool {
    let Ok(manifest) = read_parts_manifest(path) else {
        return false;
    };
    if manifest.parts.is_empty() {
        return false;
    }
    manifest
        .parts
        .iter()
        .all(|entry| resolve_part_entry_path(path, &entry.path).is_ok_and(|part| part.exists()))
}

fn cleanup_existing_parts(manifest_path: &Path) -> Result<(), String> {
    let manifest = match read_parts_manifest(manifest_path) {
        Ok(manifest) => manifest,
        Err(_) => return Ok(()),
    };
    for entry in &manifest.parts {
        let path = resolve_part_entry_path(manifest_path, &entry.path)?;
        if path.exists() {
            fs::remove_file(&path).map_err(|err| {
                format!(
                    "failed to remove old burnpack part {}: {err}",
                    path.display()
                )
            })?;
        }
    }
    Ok(())
}

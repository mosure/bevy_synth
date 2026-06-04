use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

#[cfg(any(feature = "import", test))]
use burn::module::{Module, Param, ParamId};
#[cfg(any(feature = "import", test))]
use burn::prelude::*;
#[cfg(any(feature = "import", test))]
use burn_store::{BurnpackStore, ModuleSnapshot};
use ciborium::Value;
use serde::Deserialize;

const BLOB_CHUNK_PREFIX: &str = "blob_chunks.";
const HEADER_SIZE: usize = 10;
const MAGIC_NUMBER: u32 = 0x4255_524E;
#[cfg(any(feature = "import", test))]
pub(crate) const DEFAULT_BLOB_CHUNK_BYTES: usize = 32 * 1024 * 1024;

#[cfg(any(feature = "import", test))]
#[derive(Module, Debug)]
struct BinaryBlob<B: Backend> {
    bytes: Param<Tensor<B, 1, Int>>,
}

#[cfg(any(feature = "import", test))]
#[derive(Module, Debug)]
struct BinaryBlobChunks<B: Backend> {
    blob_chunks: Vec<Param<Tensor<B, 1, Int>>>,
}

#[derive(Debug, Clone, Deserialize)]
struct RawBurnpackMetadata {
    tensors: BTreeMap<String, RawTensorDescriptor>,
}

#[derive(Debug, Clone, Deserialize)]
struct RawTensorDescriptor {
    #[allow(dead_code)]
    dtype: Value,
    #[allow(dead_code)]
    shape: Vec<u64>,
    data_offsets: (u64, u64),
}

#[derive(Debug, Clone)]
struct TensorSegment {
    ordinal: usize,
    start: u64,
    end: u64,
}

#[cfg(any(feature = "import", test))]
pub(crate) fn save_blob_bytes_to_burnpack(
    burnpack_path: &Path,
    bytes: &[u8],
    chunk_bytes: usize,
) -> Result<Vec<usize>, String> {
    type BlobBackend = burn::backend::NdArray<f32, u8>;
    let chunk_bytes = chunk_bytes.max(1);
    let mut chunk_sizes = Vec::new();
    if let Some(parent) = burnpack_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create '{}': {err}", parent.display()))?;
    }

    let device = <BlobBackend as Backend>::Device::default();
    let mut chunks = Vec::new();
    if bytes.is_empty() {
        let tensor = Tensor::<BlobBackend, 1, Int>::from_data(
            TensorData::new(Vec::<u8>::new(), [0usize]),
            &device,
        );
        chunks.push(Param::initialized(ParamId::new(), tensor));
        chunk_sizes.push(0usize);
    } else {
        for chunk in bytes.chunks(chunk_bytes) {
            let tensor = Tensor::<BlobBackend, 1, Int>::from_data(
                TensorData::new(chunk.to_vec(), [chunk.len()]),
                &device,
            );
            chunks.push(Param::initialized(ParamId::new(), tensor));
            chunk_sizes.push(chunk.len());
        }
    }

    let mut store = BurnpackStore::from_file(burnpack_path).overwrite(true);
    if chunk_sizes.len() == 1 {
        // Keep single-chunk blobs backward-compatible with the legacy field name.
        let module = BinaryBlob {
            bytes: chunks
                .into_iter()
                .next()
                .ok_or_else(|| "missing single blob chunk".to_string())?,
        };
        module.save_into(&mut store).map_err(|err| {
            format!(
                "failed to write burnpack '{}' (single chunk): {err}",
                burnpack_path.display()
            )
        })?;
    } else {
        let module = BinaryBlobChunks {
            blob_chunks: chunks,
        };
        module.save_into(&mut store).map_err(|err| {
            format!(
                "failed to write burnpack '{}' (chunked): {err}",
                burnpack_path.display()
            )
        })?;
    }

    Ok(chunk_sizes)
}

pub(crate) fn load_blob_bytes_from_burnpack(path: &Path) -> Result<Vec<u8>, String> {
    let (metadata_start, file_len, metadata) = read_burnpack_metadata(path)?;
    if metadata.tensors.is_empty() {
        return Err(format!("burnpack '{}' contains no tensors", path.display()));
    }

    let segments = resolve_blob_segments(path, &metadata)?;
    let max_end = segments
        .iter()
        .map(|segment| segment.end)
        .max()
        .unwrap_or(0);
    // Burnpack payload offsets are relative to tensor payload storage, but
    // burnpacks may include runtime-dependent padding between metadata and
    // payload bytes. Infer payload start from file length when possible.
    let inferred_payload_start = file_len.saturating_sub(max_end);
    let payload_start = inferred_payload_start.max(metadata_start);
    read_blob_segments(path, payload_start, &segments)
}

pub(crate) fn load_blob_bytes_from_burnpack_bytes(bytes: &[u8]) -> Result<Vec<u8>, String> {
    let (metadata_start, file_len, metadata) = read_burnpack_metadata_from_bytes(bytes)?;
    if metadata.tensors.is_empty() {
        return Err("burnpack bytes contain no tensors".to_string());
    }

    let segments = resolve_blob_segments(Path::new("<memory>"), &metadata)?;
    let max_end = segments
        .iter()
        .map(|segment| segment.end)
        .max()
        .unwrap_or(0);
    let inferred_payload_start = file_len.saturating_sub(max_end);
    let payload_start = inferred_payload_start.max(metadata_start);
    read_blob_segments_from_bytes(bytes, payload_start, &segments)
}

fn read_burnpack_metadata(path: &Path) -> Result<(u64, u64, RawBurnpackMetadata), String> {
    let mut file = fs::File::open(path)
        .map_err(|err| format!("failed to open burnpack '{}': {err}", path.display()))?;
    let file_len = file
        .metadata()
        .map_err(|err| format!("failed to stat burnpack '{}': {err}", path.display()))?
        .len();

    let mut header = [0u8; HEADER_SIZE];
    file.read_exact(&mut header)
        .map_err(|err| format!("failed to read burnpack header '{}': {err}", path.display()))?;
    let magic = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
    if magic != MAGIC_NUMBER {
        return Err(format!(
            "invalid burnpack magic in '{}': expected {MAGIC_NUMBER:#x}, found {magic:#x}",
            path.display()
        ));
    }
    let metadata_size = u32::from_le_bytes([header[6], header[7], header[8], header[9]]) as usize;
    let mut metadata_bytes = vec![0u8; metadata_size];
    file.read_exact(metadata_bytes.as_mut_slice())
        .map_err(|err| {
            format!(
                "failed to read burnpack metadata '{}' ({metadata_size} bytes): {err}",
                path.display()
            )
        })?;
    let metadata: RawBurnpackMetadata = ciborium::de::from_reader(metadata_bytes.as_slice())
        .map_err(|err| {
            format!(
                "failed to parse burnpack metadata '{}': {err}",
                path.display()
            )
        })?;
    Ok((
        HEADER_SIZE as u64 + metadata_size as u64,
        file_len,
        metadata,
    ))
}

fn read_burnpack_metadata_from_bytes(
    bytes: &[u8],
) -> Result<(u64, u64, RawBurnpackMetadata), String> {
    if bytes.len() < HEADER_SIZE {
        return Err(format!(
            "burnpack byte stream is too short ({} bytes)",
            bytes.len()
        ));
    }
    let header = &bytes[..HEADER_SIZE];
    let magic = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
    if magic != MAGIC_NUMBER {
        return Err(format!(
            "invalid burnpack magic in in-memory bytes: expected {MAGIC_NUMBER:#x}, found {magic:#x}",
        ));
    }
    let metadata_size = u32::from_le_bytes([header[6], header[7], header[8], header[9]]) as usize;
    let metadata_end = HEADER_SIZE.saturating_add(metadata_size);
    if metadata_end > bytes.len() {
        return Err(format!(
            "burnpack metadata out of bounds: header={} metadata={} total={}",
            HEADER_SIZE,
            metadata_size,
            bytes.len()
        ));
    }
    let metadata: RawBurnpackMetadata =
        ciborium::de::from_reader(&bytes[HEADER_SIZE..metadata_end])
            .map_err(|err| format!("failed to parse in-memory burnpack metadata: {err}"))?;
    Ok((metadata_end as u64, bytes.len() as u64, metadata))
}

fn resolve_blob_segments(
    path: &Path,
    metadata: &RawBurnpackMetadata,
) -> Result<Vec<TensorSegment>, String> {
    if let Some(descriptor) = metadata.tensors.get("bytes") {
        return Ok(vec![TensorSegment {
            ordinal: 0,
            start: descriptor.data_offsets.0,
            end: descriptor.data_offsets.1,
        }]);
    }

    let mut chunk_segments = Vec::new();
    for (name, descriptor) in &metadata.tensors {
        if let Some(suffix) = name.strip_prefix(BLOB_CHUNK_PREFIX) {
            let ordinal = suffix
                .split('.')
                .next()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(usize::MAX);
            chunk_segments.push(TensorSegment {
                ordinal,
                start: descriptor.data_offsets.0,
                end: descriptor.data_offsets.1,
            });
        }
    }

    if chunk_segments.is_empty() {
        return Err(format!(
            "burnpack '{}' is not a blob payload (expected legacy 'bytes' tensor or '{}' tensor prefix)",
            path.display(),
            BLOB_CHUNK_PREFIX
        ));
    }

    chunk_segments.sort_by(|lhs, rhs| {
        lhs.ordinal
            .cmp(&rhs.ordinal)
            .then_with(|| lhs.start.cmp(&rhs.start))
    });
    Ok(chunk_segments)
}

fn read_blob_segments(
    path: &Path,
    data_start: u64,
    segments: &[TensorSegment],
) -> Result<Vec<u8>, String> {
    let mut file = fs::File::open(path)
        .map_err(|err| format!("failed to open burnpack '{}': {err}", path.display()))?;
    let mut out = Vec::new();
    for segment in segments {
        if segment.end < segment.start {
            return Err(format!(
                "invalid burnpack tensor offsets in '{}': {}..{}",
                path.display(),
                segment.start,
                segment.end
            ));
        }
        let len = segment.end.saturating_sub(segment.start) as usize;
        if len == 0 {
            continue;
        }
        file.seek(SeekFrom::Start(data_start.saturating_add(segment.start)))
            .map_err(|err| {
                format!(
                    "failed to seek burnpack '{}' payload: {err}",
                    path.display()
                )
            })?;
        let start = out.len();
        out.resize(start + len, 0u8);
        file.read_exact(&mut out[start..]).map_err(|err| {
            format!(
                "failed to read burnpack '{}' payload slice {}..{}: {err}",
                path.display(),
                segment.start,
                segment.end
            )
        })?;
    }
    Ok(out)
}

fn read_blob_segments_from_bytes(
    bytes: &[u8],
    data_start: u64,
    segments: &[TensorSegment],
) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    for segment in segments {
        if segment.end < segment.start {
            return Err(format!(
                "invalid in-memory burnpack tensor offsets: {}..{}",
                segment.start, segment.end
            ));
        }
        let len = segment.end.saturating_sub(segment.start) as usize;
        if len == 0 {
            continue;
        }
        let start = data_start.saturating_add(segment.start) as usize;
        let end = start.saturating_add(len);
        if end > bytes.len() {
            return Err(format!(
                "in-memory burnpack payload slice out of bounds: {}..{} (total={})",
                start,
                end,
                bytes.len()
            ));
        }
        out.extend_from_slice(&bytes[start..end]);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_BLOB_CHUNK_BYTES, load_blob_bytes_from_burnpack, save_blob_bytes_to_burnpack,
    };
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn chunked_blob_roundtrips_through_burnpack() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock drift")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("burn_trellis_blob_burnpack_{unique}"));
        std::fs::create_dir_all(&root).expect("create root");
        let path = root.join("blob.bpk");

        let mut bytes = vec![0u8; DEFAULT_BLOB_CHUNK_BYTES + 137];
        for (idx, value) in bytes.iter_mut().enumerate() {
            *value = (idx % 251) as u8;
        }

        let chunk_sizes = save_blob_bytes_to_burnpack(&path, bytes.as_slice(), 1024 * 1024)
            .expect("save chunked blob");
        assert!(chunk_sizes.len() > 1, "expected chunked blob encoding");
        let (_metadata_start, _file_len, metadata) =
            super::read_burnpack_metadata(&path).expect("read metadata");
        assert!(
            metadata.tensors.len() > 1,
            "expected multiple tensors in chunked burnpack, found {}",
            metadata.tensors.len()
        );

        let loaded = load_blob_bytes_from_burnpack(&path).expect("load chunked blob");
        assert_eq!(loaded, bytes);

        let _ = std::fs::remove_dir_all(root);
    }
}

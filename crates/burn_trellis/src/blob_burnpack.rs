use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

#[cfg(any(feature = "import", test))]
use burn::module::{Module, Param, ParamId};
#[cfg(any(feature = "import", test))]
use burn::prelude::*;
use burn::tensor::DType;
#[cfg(any(feature = "import", test))]
use burn_store::{BurnpackStore, ModuleSnapshot};
use serde::Deserialize;

const BLOB_CHUNK_PREFIX: &str = "blob_chunks.";
const HEADER_SIZE: usize = 10;
const MAGIC_NUMBER: u32 = 0x4255_524E;
const TENSOR_ALIGNMENT: u64 = 256;
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
    dtype: DType,
    shape: Vec<u64>,
    data_offsets: (u64, u64),
}

#[derive(Debug, Clone)]
struct TensorSegment {
    ordinal: usize,
    dtype: DType,
    shape: Vec<u64>,
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

    let device = <BlobBackend as burn::tensor::backend::BackendTypes>::Device::default();
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
    let (data_section_start, file_len, metadata) = read_burnpack_metadata(path)?;
    if metadata.tensors.is_empty() {
        return Err(format!("burnpack '{}' contains no tensors", path.display()));
    }

    let segments = resolve_blob_segments(path, &metadata)?;
    let max_end = segments
        .iter()
        .map(|segment| segment.end)
        .max()
        .unwrap_or(0);
    let payload_start = blob_payload_start(data_section_start, file_len, max_end);
    read_blob_segments(path, payload_start, &segments)
}

pub(crate) fn load_blob_bytes_from_burnpack_bytes(bytes: &[u8]) -> Result<Vec<u8>, String> {
    let (data_section_start, file_len, metadata) = read_burnpack_metadata_from_bytes(bytes)?;
    if metadata.tensors.is_empty() {
        return Err("burnpack bytes contain no tensors".to_string());
    }

    let segments = resolve_blob_segments(Path::new("<memory>"), &metadata)?;
    let max_end = segments
        .iter()
        .map(|segment| segment.end)
        .max()
        .unwrap_or(0);
    let payload_start = blob_payload_start(data_section_start, file_len, max_end);
    read_blob_segments_from_bytes(bytes, payload_start, &segments)
}

fn blob_payload_start(data_section_start: u64, file_len: u64, max_end: u64) -> u64 {
    let inferred_payload_start = file_len.saturating_sub(max_end);
    if inferred_payload_start >= data_section_start {
        let tail_padding_shift = inferred_payload_start - data_section_start;
        if tail_padding_shift <= TENSOR_ALIGNMENT {
            return data_section_start;
        }
        return inferred_payload_start;
    }

    let legacy_alignment_gap = data_section_start - inferred_payload_start;
    if legacy_alignment_gap <= TENSOR_ALIGNMENT {
        inferred_payload_start
    } else {
        data_section_start
    }
}

fn aligned_data_section_start(metadata_size: usize) -> u64 {
    let unaligned_start = (HEADER_SIZE + metadata_size) as u64;
    unaligned_start.div_ceil(TENSOR_ALIGNMENT) * TENSOR_ALIGNMENT
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
        aligned_data_section_start(metadata_size),
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
    Ok((
        aligned_data_section_start(metadata_size),
        bytes.len() as u64,
        metadata,
    ))
}

fn resolve_blob_segments(
    path: &Path,
    metadata: &RawBurnpackMetadata,
) -> Result<Vec<TensorSegment>, String> {
    if let Some(descriptor) = metadata.tensors.get("bytes") {
        return Ok(vec![TensorSegment {
            ordinal: 0,
            dtype: descriptor.dtype,
            shape: descriptor.shape.clone(),
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
                dtype: descriptor.dtype,
                shape: descriptor.shape.clone(),
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
        let mut raw = vec![0u8; len];
        file.read_exact(&mut raw).map_err(|err| {
            format!(
                "failed to read burnpack '{}' payload slice {}..{}: {err}",
                path.display(),
                segment.start,
                segment.end
            )
        })?;
        append_blob_segment_bytes(&mut out, segment, raw.as_slice())?;
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
        append_blob_segment_bytes(&mut out, segment, &bytes[start..end])?;
    }
    Ok(out)
}

fn append_blob_segment_bytes(
    out: &mut Vec<u8>,
    segment: &TensorSegment,
    raw: &[u8],
) -> Result<(), String> {
    let element_size = segment.dtype.size();
    if element_size == 0 {
        return Err(format!("invalid zero-sized dtype {:?}", segment.dtype));
    }
    let element_count = segment.shape.iter().try_fold(1usize, |acc, dim| {
        let dim: usize = (*dim).try_into().map_err(|_| {
            format!(
                "blob tensor '{}' shape dimension {} does not fit usize",
                segment.ordinal, dim
            )
        })?;
        acc.checked_mul(dim).ok_or_else(|| {
            format!(
                "blob tensor '{}' shape product overflow: {:?}",
                segment.ordinal, segment.shape
            )
        })
    })?;
    let expected_raw_len = element_count.checked_mul(element_size).ok_or_else(|| {
        format!(
            "blob tensor '{}' raw byte count overflow: elements={} dtype={:?}",
            segment.ordinal, element_count, segment.dtype
        )
    })?;
    if expected_raw_len != raw.len() {
        return Err(format!(
            "blob tensor '{}' byte length mismatch for dtype {:?}: shape {:?} implies {} bytes, offsets contain {} bytes",
            segment.ordinal,
            segment.dtype,
            segment.shape,
            expected_raw_len,
            raw.len()
        ));
    }

    match segment.dtype {
        DType::U8 | DType::I8 | DType::Bool(_) => out.extend_from_slice(raw),
        DType::U16 | DType::I16 | DType::U32 | DType::I32 | DType::U64 | DType::I64 => {
            for lane in raw.chunks_exact(element_size) {
                out.push(lane[0]);
            }
        }
        other => {
            return Err(format!(
                "blob tensor '{}' uses unsupported dtype {:?}; expected byte or integer storage",
                segment.ordinal, other
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_BLOB_CHUNK_BYTES, load_blob_bytes_from_burnpack,
        load_blob_bytes_from_burnpack_bytes, save_blob_bytes_to_burnpack,
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

    #[test]
    fn in_memory_blob_burnpack_ignores_legacy_tail_padding() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock drift")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("burn_trellis_blob_burnpack_pad_{unique}"));
        std::fs::create_dir_all(&root).expect("create root");
        let path = root.join("blob.bpk");

        let bytes = (0..8193)
            .map(|idx| ((idx * 17 + 11) % 251) as u8)
            .collect::<Vec<_>>();
        save_blob_bytes_to_burnpack(&path, bytes.as_slice(), 2048).expect("save blob burnpack");

        let mut burnpack_bytes = std::fs::read(&path).expect("read burnpack");
        burnpack_bytes.extend(std::iter::repeat_n(0u8, 128));

        let loaded = load_blob_bytes_from_burnpack_bytes(burnpack_bytes.as_slice())
            .expect("load padded in-memory burnpack");
        assert_eq!(loaded, bytes);

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn in_memory_blob_burnpack_reads_legacy_unaligned_payload() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock drift")
            .as_nanos();
        let root =
            std::env::temp_dir().join(format!("burn_trellis_blob_burnpack_unaligned_{unique}"));
        std::fs::create_dir_all(&root).expect("create root");
        let path = root.join("blob.bpk");

        let bytes = (0..8193)
            .map(|idx| ((idx * 29 + 7) % 251) as u8)
            .collect::<Vec<_>>();
        save_blob_bytes_to_burnpack(&path, bytes.as_slice(), 2048).expect("save blob burnpack");

        let mut burnpack_bytes = std::fs::read(&path).expect("read burnpack");
        let metadata_size = u32::from_le_bytes([
            burnpack_bytes[6],
            burnpack_bytes[7],
            burnpack_bytes[8],
            burnpack_bytes[9],
        ]) as usize;
        let metadata_end = super::HEADER_SIZE + metadata_size;
        let data_start = super::aligned_data_section_start(metadata_size) as usize;
        assert!(
            data_start >= metadata_end,
            "expected aligned payload start after metadata"
        );
        if data_start > metadata_end {
            burnpack_bytes.drain(metadata_end..data_start);
        }

        let loaded = load_blob_bytes_from_burnpack_bytes(burnpack_bytes.as_slice())
            .expect("load legacy unaligned in-memory burnpack");
        assert_eq!(loaded, bytes);

        let _ = std::fs::remove_dir_all(root);
    }
}

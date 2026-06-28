use std::fs;
use std::path::Path;

#[cfg(feature = "bpk")]
use burn_core::module::ParamId;
#[cfg(feature = "bpk")]
use burn_store::{BurnpackStore, BurnpackWriter, ModuleStore, TensorSnapshot};
#[cfg(feature = "bpk")]
use burn_tensor::{DType, TensorData, bf16 as BurnBf16, f16 as BurnF16};
use half::{bf16, f16};
use memmap2::MmapOptions;
use safetensors::SafeTensors;
use safetensors::tensor::{Dtype, TensorView};

#[cfg(feature = "bpk")]
use crate::config::SegmentationPrecision;
use crate::{SegmentationError, SegmentationResult};

#[derive(Clone, Debug, PartialEq)]
pub struct LoadedTensorF32 {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

pub fn load_required_tensors_from_safetensors_file(
    path: &Path,
    keys: &[&str],
) -> SegmentationResult<Vec<(String, LoadedTensorF32)>> {
    let file = fs::File::open(path).map_err(|err| {
        SegmentationError::Io(format!("failed to open {}: {err}", path.display()))
    })?;
    let mmap = unsafe { MmapOptions::new().map(&file) }.map_err(|err| {
        SegmentationError::Io(format!("failed to mmap {}: {err}", path.display()))
    })?;
    let tensors = SafeTensors::deserialize(&mmap).map_err(|err| {
        SegmentationError::Image(format!(
            "failed to parse safetensors {}: {err}",
            path.display()
        ))
    })?;
    keys.iter()
        .map(|key| {
            let view = tensors.tensor(key).map_err(|err| {
                SegmentationError::Image(format!(
                    "missing tensor `{key}` in {}: {err}",
                    path.display()
                ))
            })?;
            Ok(((*key).to_string(), tensor_view_to_f32(&view)?))
        })
        .collect()
}

pub fn load_optional_tensor_from_safetensors_file(
    path: &Path,
    key: &str,
) -> SegmentationResult<Option<LoadedTensorF32>> {
    let file = fs::File::open(path).map_err(|err| {
        SegmentationError::Io(format!("failed to open {}: {err}", path.display()))
    })?;
    let mmap = unsafe { MmapOptions::new().map(&file) }.map_err(|err| {
        SegmentationError::Io(format!("failed to mmap {}: {err}", path.display()))
    })?;
    let tensors = SafeTensors::deserialize(&mmap).map_err(|err| {
        SegmentationError::Image(format!(
            "failed to parse safetensors {}: {err}",
            path.display()
        ))
    })?;
    match tensors.tensor(key) {
        Ok(view) => Ok(Some(tensor_view_to_f32(&view)?)),
        Err(_) => Ok(None),
    }
}

pub fn load_all_tensors_from_safetensors_file(
    path: &Path,
) -> SegmentationResult<Vec<(String, LoadedTensorF32)>> {
    let file = fs::File::open(path).map_err(|err| {
        SegmentationError::Io(format!("failed to open {}: {err}", path.display()))
    })?;
    let mmap = unsafe { MmapOptions::new().map(&file) }.map_err(|err| {
        SegmentationError::Io(format!("failed to mmap {}: {err}", path.display()))
    })?;
    let tensors = SafeTensors::deserialize(&mmap).map_err(|err| {
        SegmentationError::Image(format!(
            "failed to parse safetensors {}: {err}",
            path.display()
        ))
    })?;
    tensors
        .names()
        .into_iter()
        .map(|key| {
            let view = tensors.tensor(key).map_err(|err| {
                SegmentationError::Image(format!(
                    "missing tensor `{key}` in {}: {err}",
                    path.display()
                ))
            })?;
            Ok((key.to_string(), tensor_view_to_f32(&view)?))
        })
        .collect()
}

#[cfg(feature = "bpk")]
pub fn write_all_tensors_to_burnpack_file(
    path: &Path,
    tensors: &[(String, LoadedTensorF32)],
    precision: SegmentationPrecision,
    overwrite: bool,
) -> SegmentationResult<()> {
    if path.exists() && !overwrite {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| SegmentationError::Io(format!("create {}: {err}", parent.display())))?;
    }
    let snapshots = tensors
        .iter()
        .map(|(name, tensor)| {
            let data = tensor_data_for_precision(tensor, precision)?;
            Ok(TensorSnapshot::from_data(
                data,
                vec![name.clone()],
                vec!["Struct:SegmentationComponent".to_string()],
                ParamId::new(),
            ))
        })
        .collect::<SegmentationResult<Vec<_>>>()?;
    let writer = BurnpackWriter::new(snapshots)
        .with_metadata("format", "burnpack")
        .with_metadata("producer", "burn_segmentation")
        .with_metadata("precision", precision.label());
    writer
        .write_to_file(path)
        .map_err(|err| SegmentationError::Io(format!("write burnpack {}: {err}", path.display())))
}

#[cfg(feature = "bpk")]
pub fn load_all_tensors_from_burnpack_file(
    path: &Path,
) -> SegmentationResult<Vec<(String, LoadedTensorF32)>> {
    let mut store = BurnpackStore::from_file(path).auto_extension(false);
    let snapshots = store
        .get_all_snapshots()
        .map_err(|err| {
            SegmentationError::Image(format!("read burnpack {}: {err}", path.display()))
        })?
        .clone();
    snapshots
        .into_iter()
        .map(|(name, snapshot)| {
            let data = snapshot.to_data().map_err(|err| {
                SegmentationError::Image(format!(
                    "read tensor `{name}` from burnpack {}: {err}",
                    path.display()
                ))
            })?;
            Ok((name, tensor_data_to_f32(data)?))
        })
        .collect()
}

pub fn find_tensor<'a>(
    tensors: &'a [(String, LoadedTensorF32)],
    key: &str,
) -> SegmentationResult<&'a LoadedTensorF32> {
    tensors
        .iter()
        .find_map(|(name, tensor)| (name == key).then_some(tensor))
        .ok_or_else(|| SegmentationError::Image(format!("missing tensor `{key}`")))
}

pub fn expect_1d(key: &str, tensor: &LoadedTensorF32) -> SegmentationResult<usize> {
    if tensor.shape.len() != 1 {
        return Err(SegmentationError::Image(format!(
            "tensor `{key}` expected rank 1, got {:?}",
            tensor.shape
        )));
    }
    Ok(tensor.shape[0])
}

pub fn expect_2d(key: &str, tensor: &LoadedTensorF32) -> SegmentationResult<[usize; 2]> {
    if tensor.shape.len() != 2 {
        return Err(SegmentationError::Image(format!(
            "tensor `{key}` expected rank 2, got {:?}",
            tensor.shape
        )));
    }
    Ok([tensor.shape[0], tensor.shape[1]])
}

fn tensor_view_to_f32(view: &TensorView<'_>) -> SegmentationResult<LoadedTensorF32> {
    let shape = view.shape().to_vec();
    let data = match view.dtype() {
        Dtype::F32 => bytes_to_f32(view.data())?,
        Dtype::F64 => bytes_to_f64(view.data())?,
        Dtype::F16 => bytes_to_f16(view.data())?,
        Dtype::BF16 => bytes_to_bf16(view.data())?,
        Dtype::I64 => bytes_to_i64(view.data())?,
        Dtype::U64 => bytes_to_u64(view.data())?,
        Dtype::I32 => bytes_to_i32(view.data())?,
        Dtype::U32 => bytes_to_u32(view.data())?,
        Dtype::I16 => bytes_to_i16(view.data())?,
        Dtype::U16 => bytes_to_u16(view.data())?,
        Dtype::I8 => view
            .data()
            .iter()
            .map(|value| *value as i8 as f32)
            .collect(),
        Dtype::U8 => view.data().iter().map(|value| *value as f32).collect(),
        other => {
            return Err(SegmentationError::Image(format!(
                "unsupported tensor dtype {other:?}; expected numeric tensor"
            )));
        }
    };
    let expected = shape
        .iter()
        .try_fold(1usize, |acc, value| acc.checked_mul(*value))
        .ok_or_else(|| {
            SegmentationError::Image(format!("safetensors shape product overflow: {shape:?}"))
        })?;
    if data.len() != expected {
        return Err(SegmentationError::Image(format!(
            "tensor element count mismatch: shape {shape:?} implies {expected}, got {}",
            data.len()
        )));
    }
    Ok(LoadedTensorF32 { shape, data })
}

fn bytes_to_f32(bytes: &[u8]) -> SegmentationResult<Vec<f32>> {
    bytes_to_chunks(bytes, 4, "f32", |chunk| {
        f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
    })
}

fn bytes_to_f64(bytes: &[u8]) -> SegmentationResult<Vec<f32>> {
    bytes_to_chunks(bytes, 8, "f64", |chunk| {
        f64::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]) as f32
    })
}

fn bytes_to_f16(bytes: &[u8]) -> SegmentationResult<Vec<f32>> {
    bytes_to_chunks(bytes, 2, "f16", |chunk| {
        f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32()
    })
}

fn bytes_to_bf16(bytes: &[u8]) -> SegmentationResult<Vec<f32>> {
    bytes_to_chunks(bytes, 2, "bf16", |chunk| {
        bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32()
    })
}

fn bytes_to_i64(bytes: &[u8]) -> SegmentationResult<Vec<f32>> {
    bytes_to_chunks(bytes, 8, "i64", |chunk| {
        i64::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]) as f32
    })
}

fn bytes_to_u64(bytes: &[u8]) -> SegmentationResult<Vec<f32>> {
    bytes_to_chunks(bytes, 8, "u64", |chunk| {
        u64::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]) as f32
    })
}

fn bytes_to_i32(bytes: &[u8]) -> SegmentationResult<Vec<f32>> {
    bytes_to_chunks(bytes, 4, "i32", |chunk| {
        i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f32
    })
}

fn bytes_to_u32(bytes: &[u8]) -> SegmentationResult<Vec<f32>> {
    bytes_to_chunks(bytes, 4, "u32", |chunk| {
        u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f32
    })
}

fn bytes_to_i16(bytes: &[u8]) -> SegmentationResult<Vec<f32>> {
    bytes_to_chunks(bytes, 2, "i16", |chunk| {
        i16::from_le_bytes([chunk[0], chunk[1]]) as f32
    })
}

fn bytes_to_u16(bytes: &[u8]) -> SegmentationResult<Vec<f32>> {
    bytes_to_chunks(bytes, 2, "u16", |chunk| {
        u16::from_le_bytes([chunk[0], chunk[1]]) as f32
    })
}

fn bytes_to_chunks(
    bytes: &[u8],
    chunk_size: usize,
    dtype: &str,
    convert: impl Fn(&[u8]) -> f32,
) -> SegmentationResult<Vec<f32>> {
    if !bytes.len().is_multiple_of(chunk_size) {
        return Err(SegmentationError::Image(format!(
            "invalid {dtype} tensor byte length {}",
            bytes.len()
        )));
    }
    Ok(bytes.chunks_exact(chunk_size).map(convert).collect())
}

#[cfg(feature = "bpk")]
fn tensor_data_for_precision(
    tensor: &LoadedTensorF32,
    precision: SegmentationPrecision,
) -> SegmentationResult<TensorData> {
    let expected = tensor.shape.iter().try_fold(1usize, |acc, value| {
        acc.checked_mul(*value).ok_or_else(|| {
            SegmentationError::Image(format!(
                "burnpack tensor shape product overflow: {:?}",
                tensor.shape
            ))
        })
    })?;
    if expected != tensor.data.len() {
        return Err(SegmentationError::Image(format!(
            "burnpack tensor element count mismatch: shape {:?} implies {}, got {}",
            tensor.shape,
            expected,
            tensor.data.len()
        )));
    }
    Ok(match precision {
        SegmentationPrecision::F32 => TensorData::new(tensor.data.clone(), tensor.shape.clone()),
        SegmentationPrecision::F16 => TensorData::new(
            tensor
                .data
                .iter()
                .map(|value| BurnF16::from_f32(*value))
                .collect::<Vec<_>>(),
            tensor.shape.clone(),
        ),
        SegmentationPrecision::Bf16 => TensorData::new(
            tensor
                .data
                .iter()
                .map(|value| BurnBf16::from_f32(*value))
                .collect::<Vec<_>>(),
            tensor.shape.clone(),
        ),
    })
}

#[cfg(feature = "bpk")]
fn tensor_data_to_f32(data: TensorData) -> SegmentationResult<LoadedTensorF32> {
    let shape = data.shape.as_slice().to_vec();
    let values = match data.dtype {
        DType::F32 | DType::Flex32 => data
            .into_vec::<f32>()
            .map_err(|err| SegmentationError::Image(format!("read f32 burnpack tensor: {err}")))?,
        DType::F16 => data
            .into_vec::<BurnF16>()
            .map_err(|err| SegmentationError::Image(format!("read f16 burnpack tensor: {err}")))?
            .into_iter()
            .map(|value| value.to_f32())
            .collect(),
        DType::BF16 => data
            .into_vec::<BurnBf16>()
            .map_err(|err| SegmentationError::Image(format!("read bf16 burnpack tensor: {err}")))?
            .into_iter()
            .map(|value| value.to_f32())
            .collect(),
        DType::F64 => data
            .into_vec::<f64>()
            .map_err(|err| SegmentationError::Image(format!("read f64 burnpack tensor: {err}")))?
            .into_iter()
            .map(|value| value as f32)
            .collect(),
        DType::I64 => data
            .into_vec::<i64>()
            .map_err(|err| SegmentationError::Image(format!("read i64 burnpack tensor: {err}")))?
            .into_iter()
            .map(|value| value as f32)
            .collect(),
        DType::U64 => data
            .into_vec::<u64>()
            .map_err(|err| SegmentationError::Image(format!("read u64 burnpack tensor: {err}")))?
            .into_iter()
            .map(|value| value as f32)
            .collect(),
        DType::I32 => data
            .into_vec::<i32>()
            .map_err(|err| SegmentationError::Image(format!("read i32 burnpack tensor: {err}")))?
            .into_iter()
            .map(|value| value as f32)
            .collect(),
        DType::U32 => data
            .into_vec::<u32>()
            .map_err(|err| SegmentationError::Image(format!("read u32 burnpack tensor: {err}")))?
            .into_iter()
            .map(|value| value as f32)
            .collect(),
        DType::I16 => data
            .into_vec::<i16>()
            .map_err(|err| SegmentationError::Image(format!("read i16 burnpack tensor: {err}")))?
            .into_iter()
            .map(|value| value as f32)
            .collect(),
        DType::U16 => data
            .into_vec::<u16>()
            .map_err(|err| SegmentationError::Image(format!("read u16 burnpack tensor: {err}")))?
            .into_iter()
            .map(|value| value as f32)
            .collect(),
        DType::I8 => data
            .into_vec::<i8>()
            .map_err(|err| SegmentationError::Image(format!("read i8 burnpack tensor: {err}")))?
            .into_iter()
            .map(|value| value as f32)
            .collect(),
        DType::U8 => data
            .into_vec::<u8>()
            .map_err(|err| SegmentationError::Image(format!("read u8 burnpack tensor: {err}")))?
            .into_iter()
            .map(|value| value as f32)
            .collect(),
        other => {
            return Err(SegmentationError::Image(format!(
                "unsupported burnpack tensor dtype {other:?}"
            )));
        }
    };
    Ok(LoadedTensorF32 {
        shape,
        data: values,
    })
}

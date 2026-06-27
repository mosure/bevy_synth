use std::fs;
use std::path::Path;

use half::{bf16, f16};
use memmap2::MmapOptions;
use safetensors::SafeTensors;
use safetensors::tensor::{Dtype, TensorView};

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

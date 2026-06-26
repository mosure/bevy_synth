use std::fs;
use std::path::Path;

use half::{bf16, f16};
use memmap2::MmapOptions;
use safetensors::SafeTensors;
use safetensors::tensor::{Dtype, TensorView};

use crate::{LocateAnythingError, LocateAnythingResult};

#[derive(Clone, Debug, PartialEq)]
pub struct LoadedTensorF32 {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

#[cfg(any(
    feature = "backend_ndarray",
    feature = "backend_wgpu",
    feature = "backend_cuda"
))]
#[derive(Clone, Debug, PartialEq)]
pub struct LoadedTensorData {
    pub shape: Vec<usize>,
    pub data: burn::tensor::TensorData,
}

pub fn load_tensor_from_safetensors_file(
    path: impl AsRef<Path>,
    key: &str,
) -> LocateAnythingResult<LoadedTensorF32> {
    let mut tensors = load_required_tensors_from_safetensors_file(path.as_ref(), &[key])?;
    Ok(tensors.remove(0).1)
}

pub fn load_required_tensors_from_safetensors_file(
    path: &Path,
    keys: &[&str],
) -> LocateAnythingResult<Vec<(String, LoadedTensorF32)>> {
    let file = fs::File::open(path).map_err(|err| {
        LocateAnythingError::Io(format!("failed to open {}: {err}", path.display()))
    })?;
    // Safetensors files can be multi-GB checkpoints; mmap keeps fixture and
    // checkpoint inspection from doubling resident host memory.
    let mmap = unsafe { MmapOptions::new().map(&file) }.map_err(|err| {
        LocateAnythingError::Io(format!("failed to mmap {}: {err}", path.display()))
    })?;
    let tensors = SafeTensors::deserialize(&mmap).map_err(|err| {
        LocateAnythingError::Runtime(format!(
            "failed to parse safetensors {}: {err}",
            path.display()
        ))
    })?;
    keys.iter()
        .map(|key| {
            let view = tensors.tensor(key).map_err(|err| {
                LocateAnythingError::Runtime(format!(
                    "missing tensor `{key}` in {}: {err}",
                    path.display()
                ))
            })?;
            Ok(((*key).to_string(), tensor_view_to_f32(&view)?))
        })
        .collect()
}

#[cfg(any(
    feature = "backend_ndarray",
    feature = "backend_wgpu",
    feature = "backend_cuda"
))]
pub fn load_required_tensor_data_from_safetensors_file(
    path: &Path,
    keys: &[&str],
) -> LocateAnythingResult<Vec<(String, LoadedTensorData)>> {
    let file = fs::File::open(path).map_err(|err| {
        LocateAnythingError::Io(format!("failed to open {}: {err}", path.display()))
    })?;
    let mmap = unsafe { MmapOptions::new().map(&file) }.map_err(|err| {
        LocateAnythingError::Io(format!("failed to mmap {}: {err}", path.display()))
    })?;
    let tensors = SafeTensors::deserialize(&mmap).map_err(|err| {
        LocateAnythingError::Runtime(format!(
            "failed to parse safetensors {}: {err}",
            path.display()
        ))
    })?;
    keys.iter()
        .map(|key| {
            let view = tensors.tensor(key).map_err(|err| {
                LocateAnythingError::Runtime(format!(
                    "missing tensor `{key}` in {}: {err}",
                    path.display()
                ))
            })?;
            Ok(((*key).to_string(), tensor_view_to_tensor_data(&view)?))
        })
        .collect()
}

fn tensor_view_to_f32(view: &TensorView<'_>) -> LocateAnythingResult<LoadedTensorF32> {
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
            return Err(LocateAnythingError::Runtime(format!(
                "unsupported tensor dtype {other:?}; expected numeric f32/f16/bf16/int"
            )));
        }
    };
    let expected = shape
        .iter()
        .try_fold(1usize, |acc, value| acc.checked_mul(*value))
        .ok_or_else(|| {
            LocateAnythingError::Runtime(format!("safetensors shape product overflow: {shape:?}"))
        })?;
    if data.len() != expected {
        return Err(LocateAnythingError::Runtime(format!(
            "tensor element count mismatch: shape {shape:?} implies {expected}, got {}",
            data.len()
        )));
    }
    Ok(LoadedTensorF32 { shape, data })
}

#[cfg(any(
    feature = "backend_ndarray",
    feature = "backend_wgpu",
    feature = "backend_cuda"
))]
fn tensor_view_to_tensor_data(view: &TensorView<'_>) -> LocateAnythingResult<LoadedTensorData> {
    let shape = view.shape().to_vec();
    let dtype = safetensor_dtype_to_burn(view.dtype())?;
    let expected = shape
        .iter()
        .try_fold(dtype.size(), |acc, value| acc.checked_mul(*value))
        .ok_or_else(|| {
            LocateAnythingError::Runtime(format!("safetensors byte count overflow: {shape:?}"))
        })?;
    if view.data().len() != expected {
        return Err(LocateAnythingError::Runtime(format!(
            "tensor byte count mismatch: shape {shape:?} and dtype {dtype:?} imply {expected}, got {}",
            view.data().len()
        )));
    }
    Ok(LoadedTensorData {
        shape: shape.clone(),
        data: burn::tensor::TensorData::from_bytes_vec(view.data().to_vec(), shape, dtype),
    })
}

#[cfg(any(
    feature = "backend_ndarray",
    feature = "backend_wgpu",
    feature = "backend_cuda"
))]
fn safetensor_dtype_to_burn(dtype: Dtype) -> LocateAnythingResult<burn::tensor::DType> {
    use burn::tensor::{BoolDType, DType};
    Ok(match dtype {
        Dtype::F64 => DType::F64,
        Dtype::F32 => DType::F32,
        Dtype::F16 => DType::F16,
        Dtype::BF16 => DType::BF16,
        Dtype::I64 => DType::I64,
        Dtype::U64 => DType::U64,
        Dtype::I32 => DType::I32,
        Dtype::U32 => DType::U32,
        Dtype::I16 => DType::I16,
        Dtype::U16 => DType::U16,
        Dtype::I8 => DType::I8,
        Dtype::U8 => DType::U8,
        Dtype::BOOL => DType::Bool(BoolDType::Native),
        other => {
            return Err(LocateAnythingError::Runtime(format!(
                "unsupported tensor dtype {other:?}; expected numeric or bool safetensor"
            )));
        }
    })
}

fn bytes_to_f32(bytes: &[u8]) -> LocateAnythingResult<Vec<f32>> {
    if !bytes.len().is_multiple_of(4) {
        return Err(LocateAnythingError::Runtime(format!(
            "invalid f32 tensor byte length {}",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn bytes_to_f16(bytes: &[u8]) -> LocateAnythingResult<Vec<f32>> {
    if !bytes.len().is_multiple_of(2) {
        return Err(LocateAnythingError::Runtime(format!(
            "invalid f16 tensor byte length {}",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
        .collect())
}

fn bytes_to_f64(bytes: &[u8]) -> LocateAnythingResult<Vec<f32>> {
    if !bytes.len().is_multiple_of(8) {
        return Err(LocateAnythingError::Runtime(format!(
            "invalid f64 tensor byte length {}",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| {
            f64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]) as f32
        })
        .collect())
}

fn bytes_to_bf16(bytes: &[u8]) -> LocateAnythingResult<Vec<f32>> {
    if !bytes.len().is_multiple_of(2) {
        return Err(LocateAnythingError::Runtime(format!(
            "invalid bf16 tensor byte length {}",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
        .collect())
}

fn bytes_to_i64(bytes: &[u8]) -> LocateAnythingResult<Vec<f32>> {
    if !bytes.len().is_multiple_of(8) {
        return Err(LocateAnythingError::Runtime(format!(
            "invalid i64 tensor byte length {}",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| {
            i64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]) as f32
        })
        .collect())
}

fn bytes_to_u64(bytes: &[u8]) -> LocateAnythingResult<Vec<f32>> {
    if !bytes.len().is_multiple_of(8) {
        return Err(LocateAnythingError::Runtime(format!(
            "invalid u64 tensor byte length {}",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| {
            u64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]) as f32
        })
        .collect())
}

fn bytes_to_i32(bytes: &[u8]) -> LocateAnythingResult<Vec<f32>> {
    if !bytes.len().is_multiple_of(4) {
        return Err(LocateAnythingError::Runtime(format!(
            "invalid i32 tensor byte length {}",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f32)
        .collect())
}

fn bytes_to_u32(bytes: &[u8]) -> LocateAnythingResult<Vec<f32>> {
    if !bytes.len().is_multiple_of(4) {
        return Err(LocateAnythingError::Runtime(format!(
            "invalid u32 tensor byte length {}",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f32)
        .collect())
}

fn bytes_to_i16(bytes: &[u8]) -> LocateAnythingResult<Vec<f32>> {
    if !bytes.len().is_multiple_of(2) {
        return Err(LocateAnythingError::Runtime(format!(
            "invalid i16 tensor byte length {}",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]) as f32)
        .collect())
}

fn bytes_to_u16(bytes: &[u8]) -> LocateAnythingResult<Vec<f32>> {
    if !bytes.len().is_multiple_of(2) {
        return Err(LocateAnythingError::Runtime(format!(
            "invalid u16 tensor byte length {}",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]) as f32)
        .collect())
}

#[cfg(all(
    test,
    any(
        feature = "backend_ndarray",
        feature = "backend_wgpu",
        feature = "backend_cuda"
    )
))]
mod tests {
    use super::*;

    #[test]
    fn typed_loader_preserves_checkpoint_bf16_when_present() {
        let Some(root) = find_repo_root_for_test() else {
            eprintln!("skipping typed safetensor loader fixture; repo root not found");
            return;
        };
        let path = root
            .join("assets/models/LocateAnything-3B")
            .join("model-00002-of-00002.safetensors");
        if !path.exists() {
            eprintln!(
                "skipping typed safetensor loader fixture; missing {}",
                path.display()
            );
            return;
        }
        let tensors = load_required_tensor_data_from_safetensors_file(
            &path,
            &["language_model.model.norm.weight"],
        )
        .unwrap();
        let tensor = &tensors[0].1;
        assert_eq!(tensor.shape, vec![2048]);
        assert_eq!(tensor.data.dtype, burn::tensor::DType::BF16);
        assert_eq!(tensor.data.bytes.len(), 2048 * 2);
    }

    fn find_repo_root_for_test() -> Option<std::path::PathBuf> {
        let mut dir = std::env::current_dir().ok()?;
        loop {
            if dir.join("Cargo.toml").exists() && dir.join("crates/burn_locate_anything").exists() {
                return Some(dir);
            }
            if !dir.pop() {
                return None;
            }
        }
    }
}

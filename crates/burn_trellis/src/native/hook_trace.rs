use std::path::Path;

use safetensors::tensor::{Dtype, TensorView, serialize_to_file};

#[derive(Debug, Clone)]
struct HookTensor {
    key: String,
    dtype: Dtype,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

#[derive(Debug, Default, Clone)]
pub struct HookTrace {
    tensors: Vec<HookTensor>,
}

impl HookTrace {
    pub fn insert_u8(
        &mut self,
        key: impl Into<String>,
        shape: Vec<usize>,
        data: Vec<u8>,
    ) -> Result<(), String> {
        validate_shape_bytes(&shape, data.len(), 1)?;
        self.tensors.push(HookTensor {
            key: key.into(),
            dtype: Dtype::U8,
            shape,
            bytes: data,
        });
        Ok(())
    }

    pub fn insert_f32(
        &mut self,
        key: impl Into<String>,
        shape: Vec<usize>,
        data: Vec<f32>,
    ) -> Result<(), String> {
        validate_shape_bytes(&shape, data.len(), 1)?;
        let mut bytes = Vec::with_capacity(data.len() * std::mem::size_of::<f32>());
        for value in data {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        self.tensors.push(HookTensor {
            key: key.into(),
            dtype: Dtype::F32,
            shape,
            bytes,
        });
        Ok(())
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|err| {
                format!(
                    "failed to create hook directory '{}': {err}",
                    parent.display()
                )
            })?;
        }

        let mut views: Vec<(String, TensorView<'_>)> = Vec::with_capacity(self.tensors.len());
        for tensor in &self.tensors {
            let view = TensorView::new(tensor.dtype, tensor.shape.clone(), tensor.bytes.as_slice())
                .map_err(|err| format!("failed to encode hook tensor '{}': {err}", tensor.key))?;
            views.push((tensor.key.clone(), view));
        }

        serialize_to_file(
            views.iter().map(|(key, view)| (key.as_str(), view.clone())),
            None,
            path,
        )
        .map_err(|err| format!("failed to serialize hook trace '{}': {err}", path.display()))
    }
}

fn validate_shape_bytes(shape: &[usize], len: usize, element_size: usize) -> Result<(), String> {
    let expected = shape
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(|| "shape product overflow while building hook tensor".to_string())?;
    let expected_len = expected
        .checked_mul(element_size)
        .ok_or_else(|| "byte-length overflow while building hook tensor".to_string())?;
    if expected_len != len {
        return Err(format!(
            "hook tensor length mismatch: expected {}, got {}",
            expected_len, len
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use crate::hook_diff::HookSnapshot;

    use super::HookTrace;

    #[test]
    fn writes_u8_and_f32_hooks() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempdir()?;
        let out = dir.path().join("hook.safetensors");

        let mut trace = HookTrace::default();
        trace.insert_u8("image", vec![1, 2, 3], vec![1, 2, 3, 4, 5, 6])?;
        trace.insert_f32("value", vec![2], vec![1.5, -2.0])?;
        trace.save(&out)?;

        let snapshot = HookSnapshot::from_file(&out)?;
        assert_eq!(snapshot.tensors["image"].shape, vec![1, 2, 3]);
        assert_eq!(
            snapshot.tensors["image"].data,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
        assert_eq!(snapshot.tensors["value"].shape, vec![2]);
        assert!((snapshot.tensors["value"].data[0] - 1.5).abs() < 1e-6);
        assert!((snapshot.tensors["value"].data[1] + 2.0).abs() < 1e-6);
        Ok(())
    }
}

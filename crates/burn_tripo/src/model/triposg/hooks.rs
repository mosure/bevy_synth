use std::{collections::BTreeMap, fs, path::Path};

use burn::tensor::{Tensor, TensorData, backend::Backend};
use bytemuck::cast_slice;
use safetensors::{Dtype, serialize, tensor::TensorView};

#[derive(Debug, Clone)]
pub struct HookTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

#[derive(Debug, Default)]
pub struct HookRecorder {
    tensors: BTreeMap<String, HookTensor>,
}

impl HookRecorder {
    pub fn new() -> Self {
        Self {
            tensors: BTreeMap::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn tensors(&self) -> &BTreeMap<String, HookTensor> {
        &self.tensors
    }

    pub fn record_data(&mut self, name: &str, data: TensorData) {
        if self.tensors.contains_key(name) {
            panic!("hook tensor `{name}` recorded more than once");
        }
        let data = data.convert::<f32>();
        let values = data
            .to_vec::<f32>()
            .expect("hook tensor should convert to f32");
        let hook = HookTensor {
            shape: data.shape.clone(),
            data: values,
        };
        self.tensors.insert(name.to_string(), hook);
    }

    pub fn record_tensor<B: Backend, const D: usize>(&mut self, name: &str, tensor: &Tensor<B, D>) {
        let data = tensor.clone().into_data();
        self.record_data(name, data);
    }

    pub fn write_safetensors(&self, path: impl AsRef<Path>) -> Result<(), HookError> {
        let path = path.as_ref();
        let mut views = Vec::with_capacity(self.tensors.len());
        for (name, tensor) in &self.tensors {
            let view = TensorView::new(Dtype::F32, tensor.shape.clone(), cast_slice(&tensor.data))?;
            views.push((name.as_str(), view));
        }
        let data = serialize(views, None)?;
        fs::write(path, data)?;
        Ok(())
    }
}

#[derive(Debug)]
pub enum HookError {
    Io(std::io::Error),
    Safetensors(safetensors::SafeTensorError),
}

impl std::fmt::Display for HookError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(err) => write!(f, "io error: {err}"),
            Self::Safetensors(err) => write!(f, "safetensors error: {err}"),
        }
    }
}

impl std::error::Error for HookError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::Safetensors(err) => Some(err),
        }
    }
}

impl From<std::io::Error> for HookError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<safetensors::SafeTensorError> for HookError {
    fn from(err: safetensors::SafeTensorError) -> Self {
        Self::Safetensors(err)
    }
}

use burn::prelude::Backend;
use burn::tensor::{Int, Tensor};

use super::varlen_tensor_device::VarLenTensorDevice;

/// Canonical device-backed sparse tensor ownership for runtime-model flows.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct SparseTensorDevice<B: Backend> {
    pub coords: Tensor<B, 2, Int>,
    pub values: VarLenTensorDevice<B>,
    pub sparse_resolution: usize,
}

#[allow(dead_code)]
impl<B: Backend> SparseTensorDevice<B> {
    pub fn new(
        coords: Tensor<B, 2, Int>,
        values: VarLenTensorDevice<B>,
        sparse_resolution: usize,
    ) -> Self {
        Self {
            coords,
            values,
            sparse_resolution,
        }
    }

    pub fn rows(&self) -> usize {
        self.coords.dims()[0]
    }
}

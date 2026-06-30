use burn::prelude::Backend;
use burn::tensor::Tensor;

use super::sparse_batch_layout_device::SparseBatchLayoutDevice;

/// Device-backed varlen tensor wrapper used in sparse flow/decode paths.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct VarLenTensorDevice<B: Backend> {
    pub values: Tensor<B, 2>,
    pub layout: SparseBatchLayoutDevice<B>,
    pub channels: usize,
}

#[allow(dead_code)]
impl<B: Backend> VarLenTensorDevice<B> {
    pub fn new(values: Tensor<B, 2>, layout: SparseBatchLayoutDevice<B>, channels: usize) -> Self {
        Self {
            values,
            layout,
            channels,
        }
    }

    pub fn rows(&self) -> usize {
        self.values.dims()[0]
    }
}

use burn::prelude::Backend;
use burn::tensor::{Int, Tensor};

/// Device-backed sparse batch layout (`offsets`, `lengths`) used by varlen tensors.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct SparseBatchLayoutDevice<B: Backend> {
    pub offsets: Tensor<B, 1, Int>,
    pub lengths: Tensor<B, 1, Int>,
    pub rows: usize,
}

#[allow(dead_code)]
impl<B: Backend> SparseBatchLayoutDevice<B> {
    pub fn new(offsets: Tensor<B, 1, Int>, lengths: Tensor<B, 1, Int>, rows: usize) -> Self {
        Self {
            offsets,
            lengths,
            rows,
        }
    }

    pub fn batch_size(&self) -> usize {
        self.offsets.dims()[0]
    }
}

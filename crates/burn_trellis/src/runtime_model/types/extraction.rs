use burn::prelude::Backend;
use burn::tensor::{Int, Tensor};

/// Explicit host extraction boundary for canonical runtime-model paths.
///
/// Keep all `.into_data()` readbacks centralized in this module.
#[allow(dead_code)]
pub fn tensor_i32_to_vec<B: Backend>(
    tensor: Tensor<B, 2, Int>,
    context: &str,
) -> Result<Vec<i32>, String> {
    tensor
        .into_data()
        .convert::<i32>()
        .to_vec::<i32>()
        .map_err(|err| format!("{context}: failed tensor_i32 extraction: {err:?}"))
}

#[allow(dead_code)]
pub fn tensor_f32_to_vec<B: Backend>(
    tensor: Tensor<B, 2>,
    context: &str,
) -> Result<Vec<f32>, String> {
    tensor
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|err| format!("{context}: failed tensor_f32 extraction: {err:?}"))
}

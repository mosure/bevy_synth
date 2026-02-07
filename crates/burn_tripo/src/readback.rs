use burn::prelude::*;

#[cfg(target_arch = "wasm32")]
fn readback_not_supported() -> String {
    "tensor readback requires async handling on wasm32; \
     use an async readback path or disable GPU readback for this build"
        .to_string()
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn tensor_to_vec_f32<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
) -> Result<Vec<f32>, String> {
    tensor
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|_| "failed to read tensor data".to_string())
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn tensor_to_vec_f32<B: Backend, const D: usize>(
    _tensor: Tensor<B, D>,
) -> Result<Vec<f32>, String> {
    Err(readback_not_supported())
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn tensor_to_vec_i32<B: Backend, const D: usize>(
    tensor: Tensor<B, D, Int>,
) -> Result<Vec<i32>, String> {
    tensor
        .into_data()
        .convert::<i32>()
        .to_vec::<i32>()
        .map_err(|_| "failed to read tensor data".to_string())
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn tensor_to_vec_i32<B: Backend, const D: usize>(
    _tensor: Tensor<B, D, Int>,
) -> Result<Vec<i32>, String> {
    Err(readback_not_supported())
}

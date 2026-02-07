use burn::prelude::*;

fn smoke_backend<B: Backend>(device: &B::Device) {
    let a = Tensor::<B, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], device);
    let b = Tensor::<B, 2>::from_floats([[5.0, 6.0], [7.0, 8.0]], device);
    let c = a.matmul(b);

    let data = c
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .expect("failed to read tensor data");

    let expected = [19.0, 22.0, 43.0, 50.0];
    assert_eq!(data.len(), expected.len());
    for (value, expected) in data.iter().zip(expected.iter()) {
        let diff = (value - expected).abs();
        assert!(diff < 1e-2, "value {value} differs from {expected}");
    }
}

#[test]
fn smoke_ndarray_backend() {
    let device = Default::default();
    smoke_backend::<burn::backend::NdArray<f32>>(&device);
}

#[test]
fn smoke_wgpu_backend() {
    if std::env::var("BURN_WGPU_SMOKE").is_err() {
        eprintln!("skipping: set BURN_WGPU_SMOKE=1 to run wgpu smoke test");
        return;
    }

    let result = std::panic::catch_unwind(|| {
        let device = burn_wgpu::WgpuDevice::default();
        smoke_backend::<burn_wgpu::Wgpu<f32, i32, u32>>(&device);
    });

    if result.is_err() {
        eprintln!("skipping: wgpu backend not available on this system");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn smoke_cuda_backend() {
    if std::env::var("BURN_CUDA_SMOKE").is_err() {
        eprintln!("skipping: set BURN_CUDA_SMOKE=1 to run cuda smoke test");
        return;
    }

    let result = std::panic::catch_unwind(|| {
        let device = burn_cuda::CudaDevice::default();
        smoke_backend::<burn_cuda::Cuda<f32, i32>>(&device);
    });

    if result.is_err() {
        eprintln!("skipping: cuda backend not available on this system");
    }
}

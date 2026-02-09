use burn::tensor::{DType, Int, Shape, Tensor as BurnTensor, TensorData, TensorPrimitive};
use burn_cubecl::cubecl;
use burn_cubecl::cubecl::{calculate_cube_count_elemwise, prelude::*};
use burn_cubecl::{CubeRuntime, tensor::CubeTensor};

use crate::{SparseSubmConvConfig, build_neighbor_rows, kernel_rows};

/// Default WGPU backend type used by the tensor convenience wrappers.
pub type DefaultWgpuBackend = burn_wgpu::CubeBackend<burn_wgpu::WgpuRuntime, f32, i32, u32>;

#[cube(launch_unchecked)]
fn sparse_subm_conv_kernel(
    input: &Tensor<Line<f32>>,
    neighbor_rows: &Tensor<Line<i32>>,
    weight: &Tensor<Line<f32>>,
    bias: &Tensor<Line<f32>>,
    output: &mut Tensor<Line<f32>>,
    out_channels: &u32,
    kernel_rows: &u32,
    in_channels: &u32,
    in_channels_per_group: &u32,
    out_channels_per_group: &u32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let out_idx = ABSOLUTE_POS;
    let row = out_idx / *out_channels;
    let out_channel = out_idx % *out_channels;
    let group = out_channel / *out_channels_per_group;
    let in_group_base = group * *in_channels_per_group;

    let mut acc = bias[out_channel];
    for kernel_idx in 0..*kernel_rows {
        let neighbor = neighbor_rows[row * *kernel_rows + kernel_idx];
        let safe_neighbor = Max::max(neighbor, Line::new(0));
        let in_row = u32::cast_from(safe_neighbor);
        let input_base = in_row * *in_channels + in_group_base;
        let weight_base = (out_channel * *kernel_rows + kernel_idx) * *in_channels_per_group;
        let invalid = neighbor.equal(Line::new(-1));
        for in_local in 0..*in_channels_per_group {
            let input_value = input[input_base + in_local];
            let weight_value = weight[weight_base + in_local];
            let term = input_value * weight_value;
            acc += select_many(invalid, Line::new(0.0), term);
        }
    }

    output[out_idx] = acc;
}

/// Launch sparse submanifold convolution directly on CubeCL tensors.
///
/// All tensors stay device-resident during execution.
pub fn sparse_subm_conv_forward_cubecl<R: CubeRuntime>(
    config: &SparseSubmConvConfig,
    input: CubeTensor<R>,
    neighbor_rows: CubeTensor<R>,
    weight: CubeTensor<R>,
    bias: CubeTensor<R>,
) -> Result<CubeTensor<R>, String> {
    validate_tensor_shapes(config, &input, &neighbor_rows, &weight, &bias)?;

    let rows = input.shape.dims[0];
    let out_channels = config.out_channels;
    let output_elements = rows
        .checked_mul(out_channels)
        .ok_or_else(|| "sparse conv output size overflow".to_string())?;
    let output_bytes = output_elements
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or_else(|| "sparse conv output byte size overflow".to_string())?;

    let output = CubeTensor::new_contiguous(
        input.client.clone(),
        input.device.clone(),
        Shape::new([rows, out_channels]),
        input.client.empty(output_bytes),
        DType::F32,
    );

    let kernel_rows = kernel_rows(config)?;
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(output_elements, cube_dim);
    unsafe {
        sparse_subm_conv_kernel::launch_unchecked::<R>(
            &input.client,
            cube_count,
            cube_dim,
            input.as_tensor_arg::<f32>(1),
            neighbor_rows.as_tensor_arg::<i32>(1),
            weight.as_tensor_arg::<f32>(1),
            bias.as_tensor_arg::<f32>(1),
            output.as_tensor_arg::<f32>(1),
            ScalarArg::new(config.out_channels as u32),
            ScalarArg::new(kernel_rows as u32),
            ScalarArg::new(config.in_channels as u32),
            ScalarArg::new(config.in_channels_per_group as u32),
            ScalarArg::new(config.out_channels_per_group as u32),
        );
    }

    Ok(output)
}

/// Convenience wrapper for WGPU Burn tensors.
pub fn sparse_subm_conv_forward_wgpu(
    config: &SparseSubmConvConfig,
    input: BurnTensor<DefaultWgpuBackend, 2>,
    neighbor_rows: BurnTensor<DefaultWgpuBackend, 2, Int>,
    weight: BurnTensor<DefaultWgpuBackend, 5>,
    bias: BurnTensor<DefaultWgpuBackend, 1>,
) -> Result<BurnTensor<DefaultWgpuBackend, 2>, String> {
    let output = sparse_subm_conv_forward_cubecl(
        config,
        input.into_primitive().tensor(),
        neighbor_rows.into_primitive(),
        weight.into_primitive().tensor(),
        bias.into_primitive().tensor(),
    )?;
    Ok(BurnTensor::from_primitive(TensorPrimitive::Float(output)))
}

/// Build a device tensor containing sparse neighbor row indices.
pub fn neighbor_rows_tensor_from_coords(
    config: &SparseSubmConvConfig,
    coords: &[[u32; 4]],
    device: &burn_wgpu::WgpuDevice,
) -> Result<BurnTensor<DefaultWgpuBackend, 2, Int>, String> {
    let rows = coords.len();
    let kernel_rows = kernel_rows(config)?;
    let neighbor_rows = build_neighbor_rows(config, coords)?;
    if neighbor_rows.len() != rows * kernel_rows {
        return Err(format!(
            "neighbor row tensor size mismatch: got {} expected {}",
            neighbor_rows.len(),
            rows * kernel_rows
        ));
    }

    let tensor = BurnTensor::<DefaultWgpuBackend, 1, Int>::from_data(
        TensorData::new(neighbor_rows, [rows * kernel_rows]),
        device,
    )
    .reshape([rows, kernel_rows]);
    Ok(tensor)
}

fn validate_tensor_shapes<R: CubeRuntime>(
    config: &SparseSubmConvConfig,
    input: &CubeTensor<R>,
    neighbor_rows: &CubeTensor<R>,
    weight: &CubeTensor<R>,
    bias: &CubeTensor<R>,
) -> Result<(), String> {
    if input.dtype != DType::F32 {
        return Err(format!(
            "sparse conv input dtype must be F32 for kernel path, got {:?}",
            input.dtype
        ));
    }
    if weight.dtype != DType::F32 {
        return Err(format!(
            "sparse conv weight dtype must be F32 for kernel path, got {:?}",
            weight.dtype
        ));
    }
    if bias.dtype != DType::F32 {
        return Err(format!(
            "sparse conv bias dtype must be F32 for kernel path, got {:?}",
            bias.dtype
        ));
    }
    if neighbor_rows.dtype != DType::I32 {
        return Err(format!(
            "sparse conv neighbor_rows dtype must be I32 for kernel path, got {:?}",
            neighbor_rows.dtype
        ));
    }

    if input.shape.dims.len() != 2 {
        return Err(format!(
            "sparse conv input rank mismatch: got {} expected 2",
            input.shape.dims.len()
        ));
    }
    if neighbor_rows.shape.dims.len() != 2 {
        return Err(format!(
            "sparse conv neighbor_rows rank mismatch: got {} expected 2",
            neighbor_rows.shape.dims.len()
        ));
    }
    if weight.shape.dims.len() != 5 {
        return Err(format!(
            "sparse conv weight rank mismatch: got {} expected 5",
            weight.shape.dims.len()
        ));
    }
    if bias.shape.dims.len() != 1 {
        return Err(format!(
            "sparse conv bias rank mismatch: got {} expected 1",
            bias.shape.dims.len()
        ));
    }

    let rows = input.shape.dims[0];
    if input.shape.dims[1] != config.in_channels {
        return Err(format!(
            "sparse conv input channel mismatch: got {} expected {}",
            input.shape.dims[1], config.in_channels
        ));
    }
    if neighbor_rows.shape.dims[0] != rows {
        return Err(format!(
            "sparse conv neighbor row count mismatch: got {} expected {}",
            neighbor_rows.shape.dims[0], rows
        ));
    }
    let expected_kernel_rows = kernel_rows(config)?;
    if neighbor_rows.shape.dims[1] != expected_kernel_rows {
        return Err(format!(
            "sparse conv neighbor kernel rows mismatch: got {} expected {}",
            neighbor_rows.shape.dims[1], expected_kernel_rows
        ));
    }

    let expected_weight = [
        config.out_channels,
        config.kernel_d,
        config.kernel_h,
        config.kernel_w,
        config.in_channels_per_group,
    ];
    if weight.shape.dims.as_slice() != expected_weight.as_slice() {
        return Err(format!(
            "sparse conv weight shape mismatch: got {:?} expected {:?}",
            weight.shape.dims, expected_weight
        ));
    }
    if bias.shape.dims[0] != config.out_channels {
        return Err(format!(
            "sparse conv bias len mismatch: got {} expected {}",
            bias.shape.dims[0], config.out_channels
        ));
    }
    Ok(())
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use burn::tensor::Tensor;

    use crate::{SparseSubmConvConfig, SparseSubmConvWeights, sparse_subm_conv_forward_flex};

    use super::{
        DefaultWgpuBackend, neighbor_rows_tensor_from_coords, sparse_subm_conv_forward_wgpu,
    };

    #[derive(Clone)]
    struct Lcg {
        state: u64,
    }

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self { state: seed | 1 }
        }
        fn next_f32(&mut self) -> f32 {
            self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let bits = ((self.state >> 40) as u32) | 1;
            (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
        }
    }

    fn line_coords(count: usize) -> Vec<[u32; 4]> {
        (0..count as u32).map(|x| [0, x, 0, 0]).collect()
    }

    #[test]
    fn wgpu_kernel_matches_cpu_flex_path() {
        let cfg = SparseSubmConvConfig {
            in_channels: 8,
            out_channels: 12,
            kernel_d: 3,
            kernel_h: 1,
            kernel_w: 1,
            in_channels_per_group: 4,
            out_channels_per_group: 6,
            groups: 2,
            axis_order: [0, 1, 2],
            axis_sign: [1, 1, 1],
        };
        let coords = line_coords(96);
        let mut rng = Lcg::new(1234);
        let input: Vec<f32> = (0..coords.len() * cfg.in_channels)
            .map(|_| rng.next_f32())
            .collect();
        let weight_len = cfg.out_channels
            * cfg.kernel_d
            * cfg.kernel_h
            * cfg.kernel_w
            * cfg.in_channels_per_group;
        let weight: Vec<f32> = (0..weight_len).map(|_| rng.next_f32()).collect();
        let bias: Vec<f32> = (0..cfg.out_channels).map(|_| rng.next_f32()).collect();

        let expected = sparse_subm_conv_forward_flex(
            &cfg,
            SparseSubmConvWeights {
                weight: weight.as_slice(),
                bias: bias.as_slice(),
            },
            coords.as_slice(),
            input.as_slice(),
        )
        .expect("cpu flex path");

        let device = burn_wgpu::WgpuDevice::default();
        let input_t = Tensor::<DefaultWgpuBackend, 1>::from_floats(input.as_slice(), &device)
            .reshape([coords.len(), cfg.in_channels]);
        let weight_t = Tensor::<DefaultWgpuBackend, 1>::from_floats(weight.as_slice(), &device)
            .reshape([
                cfg.out_channels,
                cfg.kernel_d,
                cfg.kernel_h,
                cfg.kernel_w,
                cfg.in_channels_per_group,
            ]);
        let bias_t = Tensor::<DefaultWgpuBackend, 1>::from_floats(bias.as_slice(), &device);
        let neighbors_t =
            neighbor_rows_tensor_from_coords(&cfg, coords.as_slice(), &device).expect("neighbors");

        let output = sparse_subm_conv_forward_wgpu(&cfg, input_t, neighbors_t, weight_t, bias_t)
            .expect("wgpu kernel path");
        let output = output.to_data();
        let output = output.as_slice::<f32>().expect("f32 output");

        assert_eq!(output.len(), expected.len());
        for (idx, (lhs, rhs)) in output.iter().zip(expected.iter()).enumerate() {
            let diff = (lhs - rhs).abs();
            assert!(diff <= 1.0e-4, "mismatch at idx={idx}: lhs={lhs} rhs={rhs}");
        }
    }

    #[test]
    fn neighbor_rows_tensor_shape_is_consistent() {
        let cfg = SparseSubmConvConfig {
            in_channels: 2,
            out_channels: 2,
            kernel_d: 3,
            kernel_h: 1,
            kernel_w: 1,
            in_channels_per_group: 2,
            out_channels_per_group: 2,
            groups: 1,
            axis_order: [0, 1, 2],
            axis_sign: [1, 1, 1],
        };
        let coords = line_coords(5);
        let device = burn_wgpu::WgpuDevice::default();
        let neighbors =
            neighbor_rows_tensor_from_coords(&cfg, coords.as_slice(), &device).expect("neighbors");
        let data = neighbors.to_data();
        let [rows, kernel_rows] = neighbors.dims();
        assert_eq!(rows, coords.len());
        assert_eq!(kernel_rows, 3);
        let values = data.as_slice::<i32>().expect("i32");
        assert_eq!(values.len(), rows * kernel_rows);
    }
}

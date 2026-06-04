use burn::tensor::{DType, Int, Shape, Tensor as BurnTensor, TensorPrimitive};
use burn_cubecl::cubecl;
use burn_cubecl::cubecl::{calculate_cube_count_elemwise, prelude::*};
use burn_cubecl::tensor::CubeTensor;

use crate::{SparsePatchify3dConfig, validate_sparse_patchify3d_config};

/// Default WGPU backend type used by the sparse patchify tensor wrapper.
pub type DefaultWgpuBackend = burn_wgpu::CubeBackend<burn_wgpu::WgpuRuntime, f32, i32, u32>;

#[cube(launch_unchecked)]
fn sparse_patchify3d_kernel(
    input: &Array<f32>,
    coords: &Array<i32>,
    weight: &Array<f32>,
    bias: &Array<f32>,
    output: &mut Array<f32>,
    rows: &usize,
    batch_count: &usize,
    in_channels: &usize,
    out_channels: &usize,
    frames: &usize,
    height: &usize,
    width: &usize,
    tubelet_size: &usize,
    patch_h: &usize,
    patch_w: &usize,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }
    let idx = ABSOLUTE_POS;
    let row = idx / *out_channels;
    if row >= *rows {
        terminate!();
    }
    let out_channel = idx % *out_channels;
    let coord_base = row * 4;
    let batch_i = coords[coord_base];
    let tubelet_i = coords[coord_base + 1];
    let patch_row_i = coords[coord_base + 2];
    let patch_col_i = coords[coord_base + 3];

    let grid_t = *frames / *tubelet_size;
    let grid_h = *height / *patch_h;
    let grid_w = *width / *patch_w;
    if batch_i < 0 || tubelet_i < 0 || patch_row_i < 0 || patch_col_i < 0 {
        output[idx] = bias[out_channel];
        terminate!();
    }
    let batch = usize::cast_from(batch_i);
    let tubelet = usize::cast_from(tubelet_i);
    let patch_row = usize::cast_from(patch_row_i);
    let patch_col = usize::cast_from(patch_col_i);
    if batch >= *batch_count || tubelet >= grid_t || patch_row >= grid_h || patch_col >= grid_w {
        output[idx] = bias[out_channel];
        terminate!();
    }

    let t0 = tubelet * *tubelet_size;
    let y0 = patch_row * *patch_h;
    let x0 = patch_col * *patch_w;
    let mut acc = bias[out_channel];
    for c in 0..*in_channels {
        for dt in 0..*tubelet_size {
            let t = t0 + dt;
            for py in 0..*patch_h {
                let y = y0 + py;
                for px in 0..*patch_w {
                    let x = x0 + px;
                    let input_idx =
                        (((batch * *in_channels + c) * *frames + t) * *height + y) * *width + x;
                    let weight_idx =
                        ((((out_channel * *in_channels + c) * *tubelet_size + dt) * *patch_h + py)
                            * *patch_w)
                            + px;
                    acc += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    output[idx] = acc;
}

pub fn sparse_patchify3d_forward_wgpu(
    config: &SparsePatchify3dConfig,
    input: BurnTensor<DefaultWgpuBackend, 5>,
    coords: BurnTensor<DefaultWgpuBackend, 2, Int>,
    weight: BurnTensor<DefaultWgpuBackend, 5>,
    bias: BurnTensor<DefaultWgpuBackend, 1>,
) -> Result<BurnTensor<DefaultWgpuBackend, 2>, String> {
    validate_sparse_patchify3d_config(config)?;
    let [batch, in_channels, frames, height, width] = input.dims();
    if in_channels != config.in_channels
        || frames != config.frames
        || height != config.height
        || width != config.width
    {
        return Err(format!(
            "sparse patchify input dims mismatch: got=[{batch},{in_channels},{frames},{height},{width}] expected=[batch,{},{},{},{}]",
            config.in_channels, config.frames, config.height, config.width
        ));
    }
    let [rows, coord_cols] = coords.dims();
    if coord_cols != 4 {
        return Err(format!(
            "sparse patchify coords dims mismatch: got=[{rows},{coord_cols}] expected=[rows,4]"
        ));
    }
    let [weight_out, weight_in, weight_t, weight_h, weight_w] = weight.dims();
    if weight_out != config.out_channels
        || weight_in != config.in_channels
        || weight_t != config.tubelet_size
        || weight_h != config.patch_h
        || weight_w != config.patch_w
    {
        return Err(format!(
            "sparse patchify weight dims mismatch: got=[{weight_out},{weight_in},{weight_t},{weight_h},{weight_w}] expected=[{},{},{},{},{}]",
            config.out_channels,
            config.in_channels,
            config.tubelet_size,
            config.patch_h,
            config.patch_w
        ));
    }
    let [bias_channels] = bias.dims();
    if bias_channels != config.out_channels {
        return Err(format!(
            "sparse patchify bias mismatch: got {bias_channels} expected {}",
            config.out_channels
        ));
    }
    if rows == 0 {
        return Ok(BurnTensor::<DefaultWgpuBackend, 2>::zeros(
            [0, config.out_channels],
            &input.device(),
        ));
    }

    let output_elements = rows
        .checked_mul(config.out_channels)
        .ok_or_else(|| "sparse patchify output size overflow".to_string())?;
    let output_bytes = output_elements
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or_else(|| "sparse patchify output byte size overflow".to_string())?;

    let input_p = input.into_primitive().tensor();
    let coords_p = coords.reshape([rows * 4]).into_primitive();
    let weight_p = weight
        .reshape([config.out_channels
            * config.in_channels
            * config.tubelet_size
            * config.patch_h
            * config.patch_w])
        .into_primitive()
        .tensor();
    let bias_p = bias.into_primitive().tensor();
    let output = CubeTensor::new_contiguous(
        input_p.client.clone(),
        input_p.device.clone(),
        Shape::new([rows, config.out_channels]),
        input_p.client.empty(output_bytes),
        DType::F32,
    );
    let cube_dim = CubeDim::new(&input_p.client, output_elements);
    let cube_count = calculate_cube_count_elemwise(&input_p.client, output_elements, cube_dim);
    let client = input_p.client.clone();

    unsafe {
        sparse_patchify3d_kernel::launch_unchecked::<burn_wgpu::WgpuRuntime>(
            &client,
            cube_count,
            cube_dim,
            input_p.as_array_arg::<f32>(1),
            coords_p.as_array_arg::<i32>(1),
            weight_p.as_array_arg::<f32>(1),
            bias_p.as_array_arg::<f32>(1),
            output.as_array_arg::<f32>(1),
            ScalarArg::new(rows),
            ScalarArg::new(batch),
            ScalarArg::new(config.in_channels),
            ScalarArg::new(config.out_channels),
            ScalarArg::new(config.frames),
            ScalarArg::new(config.height),
            ScalarArg::new(config.width),
            ScalarArg::new(config.tubelet_size),
            ScalarArg::new(config.patch_h),
            ScalarArg::new(config.patch_w),
        )
        .map_err(|err| format!("sparse_patchify3d_kernel launch failed: {err:?}"))?;
    }

    Ok(BurnTensor::<DefaultWgpuBackend, 2>::from_primitive(
        TensorPrimitive::Float(output),
    ))
}

#[cfg(test)]
mod tests {
    use burn::tensor::{Tensor, TensorData};

    use crate::{SparsePatchify3dConfig, SparsePatchify3dWeights, sparse_patchify3d_forward_flex};

    use super::{DefaultWgpuBackend, sparse_patchify3d_forward_wgpu};

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

    #[test]
    fn wgpu_sparse_patchify3d_matches_cpu_flex_path() {
        let cfg = SparsePatchify3dConfig {
            in_channels: 3,
            out_channels: 7,
            frames: 4,
            height: 6,
            width: 8,
            tubelet_size: 2,
            patch_h: 3,
            patch_w: 4,
        };
        let coords = vec![[0, 0, 0, 0], [0, 1, 1, 1], [1, 0, 1, 0], [1, 1, 0, 1]];
        let mut rng = Lcg::new(82374);
        let per_batch = cfg.in_channels * cfg.frames * cfg.height * cfg.width;
        let input: Vec<f32> = (0..2 * per_batch).map(|_| rng.next_f32()).collect();
        let weight_len =
            cfg.out_channels * cfg.in_channels * cfg.tubelet_size * cfg.patch_h * cfg.patch_w;
        let weight: Vec<f32> = (0..weight_len).map(|_| rng.next_f32()).collect();
        let bias: Vec<f32> = (0..cfg.out_channels).map(|_| rng.next_f32()).collect();
        let expected = sparse_patchify3d_forward_flex(
            &cfg,
            SparsePatchify3dWeights {
                weight: weight.as_slice(),
                bias: bias.as_slice(),
            },
            coords.as_slice(),
            input.as_slice(),
        )
        .expect("cpu sparse patchify path");

        let device = burn_wgpu::WgpuDevice::default();
        let input_t = Tensor::<DefaultWgpuBackend, 1>::from_floats(input.as_slice(), &device)
            .reshape([2, cfg.in_channels, cfg.frames, cfg.height, cfg.width]);
        let coords_flat: Vec<i64> = coords
            .iter()
            .flat_map(|coord| coord.iter().map(|value| *value as i64))
            .collect();
        let coords_t = Tensor::<DefaultWgpuBackend, 1, burn::tensor::Int>::from_data(
            TensorData::new(coords_flat, [coords.len() * 4]),
            &device,
        )
        .reshape([coords.len(), 4]);
        let weight_t = Tensor::<DefaultWgpuBackend, 1>::from_floats(weight.as_slice(), &device)
            .reshape([
                cfg.out_channels,
                cfg.in_channels,
                cfg.tubelet_size,
                cfg.patch_h,
                cfg.patch_w,
            ]);
        let bias_t = Tensor::<DefaultWgpuBackend, 1>::from_floats(bias.as_slice(), &device);
        let output = sparse_patchify3d_forward_wgpu(&cfg, input_t, coords_t, weight_t, bias_t)
            .expect("wgpu sparse patchify path")
            .to_data();
        let output = output.as_slice::<f32>().expect("f32 output");

        assert_eq!(output.len(), expected.len());
        for (idx, (lhs, rhs)) in output.iter().zip(expected.iter()).enumerate() {
            let diff = (lhs - rhs).abs();
            assert!(
                diff <= 1.0e-4,
                "patchify mismatch at idx={idx}: lhs={lhs} rhs={rhs} diff={diff}"
            );
        }
    }
}

use super::sparse_conv_config::resolve_sparse_wgpu_forward_config_internal;
use super::sparse_conv_kernels::*;
use super::*;

fn sparse_subm_conv_forward_cubecl_impl<R: CubeRuntime>(
    config: &SparseSubmConvConfig,
    input: CubeTensor<R>,
    neighbor_rows: CubeTensor<R>,
    weight: CubeTensor<R>,
    bias: CubeTensor<R>,
    forward: SparseWgpuForwardConfig,
) -> Result<CubeTensor<R>, String> {
    validate_tensor_shapes(config, &input, &neighbor_rows, &weight, &bias)?;

    let query_rows = neighbor_rows.meta.shape[0];
    let out_channels = config.out_channels;
    let output_elements = query_rows
        .checked_mul(out_channels)
        .ok_or_else(|| "sparse conv output size overflow".to_string())?;
    let output_bytes = output_elements
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or_else(|| "sparse conv output byte size overflow".to_string())?;

    let output = CubeTensor::new_contiguous(
        input.client.clone(),
        input.device.clone(),
        Shape::new([query_rows, out_channels]),
        input.client.empty(output_bytes),
        DType::F32,
    );

    let kernel_rows = kernel_rows(config)?;
    let resolved =
        resolve_sparse_wgpu_forward_config_internal(config, query_rows, kernel_rows, forward);
    let split_k = resolved.split_k;
    let kernel_variant = resolved.kernel_variant;
    let cube_dim = resolve_cube_dim();
    if split_k <= 1 {
        match kernel_variant {
            SparseConvKernelVariant::Baseline => {
                let cube_count =
                    calculate_cube_count_elemwise(&input.client, output_elements, cube_dim);
                unsafe {
                    sparse_subm_conv_kernel::launch_unchecked::<R>(
                        &input.client,
                        cube_count,
                        cube_dim,
                        input.clone().into_array_arg(),
                        neighbor_rows.clone().into_array_arg(),
                        weight.clone().into_array_arg(),
                        bias.clone().into_array_arg(),
                        output.clone().into_array_arg(),
                        config.out_channels,
                        kernel_rows,
                        config.in_channels,
                        config.in_channels_per_group,
                        config.out_channels_per_group,
                    )
                    .map_err(|err| format!("sparse_subm_conv_kernel launch failed: {err:?}"))?;
                }
            }
            SparseConvKernelVariant::BaselineSingleGroup => {
                let cube_count =
                    calculate_cube_count_elemwise(&input.client, output_elements, cube_dim);
                unsafe {
                    sparse_subm_conv_single_group_kernel::launch_unchecked::<R>(
                        &input.client,
                        cube_count,
                        cube_dim,
                        input.clone().into_array_arg(),
                        neighbor_rows.clone().into_array_arg(),
                        weight.clone().into_array_arg(),
                        bias.clone().into_array_arg(),
                        output.clone().into_array_arg(),
                        config.out_channels,
                        kernel_rows,
                        config.in_channels,
                    )
                    .map_err(|err| {
                        format!("sparse_subm_conv_single_group_kernel launch failed: {err:?}")
                    })?;
                }
            }
            SparseConvKernelVariant::FusedOc4 => {
                let blocks_per_row = config.out_channels.div_ceil(FUSED_OC_TILE);
                let output_blocks = query_rows
                    .checked_mul(blocks_per_row)
                    .ok_or_else(|| "sparse conv fused output tile count overflow".to_string())?;
                let cube_count =
                    calculate_cube_count_elemwise(&input.client, output_blocks, cube_dim);
                unsafe {
                    sparse_subm_conv_fused_oc4_kernel::launch_unchecked::<R>(
                        &input.client,
                        cube_count,
                        cube_dim,
                        input.clone().into_array_arg(),
                        neighbor_rows.clone().into_array_arg(),
                        weight.clone().into_array_arg(),
                        bias.clone().into_array_arg(),
                        output.clone().into_array_arg(),
                        config.out_channels,
                        kernel_rows,
                        config.in_channels,
                        config.in_channels_per_group,
                        config.out_channels_per_group,
                    )
                    .map_err(|err| {
                        format!("sparse_subm_conv_fused_oc4_kernel launch failed: {err:?}")
                    })?;
                }
            }
            SparseConvKernelVariant::FusedOc4SingleGroup => {
                let blocks_per_row = config.out_channels.div_ceil(FUSED_OC_TILE);
                let output_blocks = query_rows
                    .checked_mul(blocks_per_row)
                    .ok_or_else(|| "sparse conv fused output tile count overflow".to_string())?;
                let cube_count =
                    calculate_cube_count_elemwise(&input.client, output_blocks, cube_dim);
                unsafe {
                    sparse_subm_conv_single_group_fused_oc4_kernel::launch_unchecked::<R>(
                        &input.client,
                        cube_count,
                        cube_dim,
                        input.clone().into_array_arg(),
                        neighbor_rows.clone().into_array_arg(),
                        weight.clone().into_array_arg(),
                        bias.clone().into_array_arg(),
                        output.clone().into_array_arg(),
                        config.out_channels,
                        kernel_rows,
                        config.in_channels,
                    )
                    .map_err(|err| {
                        format!(
                            "sparse_subm_conv_single_group_fused_oc4_kernel launch failed: {err:?}"
                        )
                    })?;
                }
            }
        }
    } else {
        let partial_elements = output_elements
            .checked_mul(split_k)
            .ok_or_else(|| "sparse conv split-k partial size overflow".to_string())?;
        let partial_bytes = partial_elements
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or_else(|| "sparse conv split-k partial byte size overflow".to_string())?;
        let partial = CubeTensor::new_contiguous(
            input.client.clone(),
            input.device.clone(),
            Shape::new([split_k, query_rows, out_channels]),
            input.client.empty(partial_bytes),
            DType::F32,
        );
        match kernel_variant {
            SparseConvKernelVariant::Baseline => {
                let partial_cube_count =
                    calculate_cube_count_elemwise(&input.client, partial_elements, cube_dim);
                unsafe {
                    sparse_subm_conv_splitk_partial_kernel::launch_unchecked::<R>(
                        &input.client,
                        partial_cube_count,
                        cube_dim,
                        input.clone().into_array_arg(),
                        neighbor_rows.clone().into_array_arg(),
                        weight.clone().into_array_arg(),
                        partial.clone().into_array_arg(),
                        config.out_channels,
                        kernel_rows,
                        config.in_channels,
                        config.in_channels_per_group,
                        config.out_channels_per_group,
                        output_elements,
                        split_k,
                    )
                    .map_err(|err| {
                        format!("sparse_subm_conv_splitk_partial_kernel launch failed: {err:?}")
                    })?;
                }
            }
            SparseConvKernelVariant::BaselineSingleGroup => {
                let partial_cube_count =
                    calculate_cube_count_elemwise(&input.client, partial_elements, cube_dim);
                unsafe {
                    sparse_subm_conv_splitk_partial_single_group_kernel::launch_unchecked::<R>(
                        &input.client,
                        partial_cube_count,
                        cube_dim,
                        input.clone().into_array_arg(),
                        neighbor_rows.clone().into_array_arg(),
                        weight.clone().into_array_arg(),
                        partial.clone().into_array_arg(),
                        config.out_channels,
                        kernel_rows,
                        config.in_channels,
                        output_elements,
                        split_k,
                    )
                    .map_err(|err| {
                        format!(
                            "sparse_subm_conv_splitk_partial_single_group_kernel launch failed: {err:?}"
                        )
                    })?;
                }
            }
            SparseConvKernelVariant::FusedOc4 => {
                let blocks_per_row = config.out_channels.div_ceil(FUSED_OC_TILE);
                let partial_blocks = query_rows
                    .checked_mul(blocks_per_row)
                    .and_then(|value| value.checked_mul(split_k))
                    .ok_or_else(|| "sparse conv fused split-k tile count overflow".to_string())?;
                let partial_cube_count =
                    calculate_cube_count_elemwise(&input.client, partial_blocks, cube_dim);
                unsafe {
                    sparse_subm_conv_splitk_partial_fused_oc4_kernel::launch_unchecked::<R>(
                        &input.client,
                        partial_cube_count,
                        cube_dim,
                        input.clone().into_array_arg(),
                        neighbor_rows.clone().into_array_arg(),
                        weight.clone().into_array_arg(),
                        partial.clone().into_array_arg(),
                        config.out_channels,
                        kernel_rows,
                        config.in_channels,
                        config.in_channels_per_group,
                        config.out_channels_per_group,
                        output_elements,
                        split_k,
                    )
                    .map_err(|err| {
                        format!(
                            "sparse_subm_conv_splitk_partial_fused_oc4_kernel launch failed: {err:?}"
                        )
                    })?;
                }
            }
            SparseConvKernelVariant::FusedOc4SingleGroup => {
                let blocks_per_row = config.out_channels.div_ceil(FUSED_OC_TILE);
                let partial_blocks = query_rows
                    .checked_mul(blocks_per_row)
                    .and_then(|value| value.checked_mul(split_k))
                    .ok_or_else(|| "sparse conv fused split-k tile count overflow".to_string())?;
                let partial_cube_count =
                    calculate_cube_count_elemwise(&input.client, partial_blocks, cube_dim);
                unsafe {
                    sparse_subm_conv_splitk_partial_single_group_fused_oc4_kernel::launch_unchecked::<R>(
                        &input.client,
                        partial_cube_count,
                        cube_dim,
                        input.clone().into_array_arg(),
                        neighbor_rows.clone().into_array_arg(),
                        weight.clone().into_array_arg(),
                        partial.clone().into_array_arg(),
                        config.out_channels,
                        kernel_rows,
                        config.in_channels,
                        output_elements,
                        split_k,
                    )
                    .map_err(|err| {
                        format!(
                            "sparse_subm_conv_splitk_partial_single_group_fused_oc4_kernel launch failed: {err:?}"
                        )
                    })?;
                }
            }
        }

        let finalize_cube_count =
            calculate_cube_count_elemwise(&input.client, output_elements, cube_dim);
        unsafe {
            sparse_subm_conv_splitk_finalize_kernel::launch_unchecked::<R>(
                &input.client,
                finalize_cube_count,
                cube_dim,
                partial.clone().into_array_arg(),
                bias.clone().into_array_arg(),
                output.clone().into_array_arg(),
                config.out_channels,
                output_elements,
                split_k,
            )
            .map_err(|err| {
                format!("sparse_subm_conv_splitk_finalize_kernel launch failed: {err:?}")
            })?;
        }
    }

    Ok(output)
}

pub fn sparse_subm_conv_forward_cubecl<R: CubeRuntime>(
    config: &SparseSubmConvConfig,
    input: CubeTensor<R>,
    neighbor_rows: CubeTensor<R>,
    weight: CubeTensor<R>,
    bias: CubeTensor<R>,
) -> Result<CubeTensor<R>, String> {
    sparse_subm_conv_forward_cubecl_impl(
        config,
        input,
        neighbor_rows,
        weight,
        bias,
        SparseWgpuForwardConfig::default(),
    )
}

/// Convenience wrapper for WGPU Burn tensors.
pub fn sparse_subm_conv_forward_wgpu(
    config: &SparseSubmConvConfig,
    input: BurnTensor<DefaultWgpuBackend, 2>,
    neighbor_rows: BurnTensor<DefaultWgpuBackend, 2, Int>,
    weight: BurnTensor<DefaultWgpuBackend, 5>,
    bias: BurnTensor<DefaultWgpuBackend, 1>,
) -> Result<BurnTensor<DefaultWgpuBackend, 2>, String> {
    let output = sparse_subm_conv_forward_cubecl_impl(
        config,
        input.into_primitive().tensor(),
        neighbor_rows.into_primitive(),
        weight.into_primitive().tensor(),
        bias.into_primitive().tensor(),
        SparseWgpuForwardConfig::default(),
    )?;
    Ok(BurnTensor::from_primitive(TensorPrimitive::Float(output)))
}

/// Convenience wrapper for WGPU Burn tensors with explicit kernel scheduling controls.
pub fn sparse_subm_conv_forward_wgpu_with_config(
    config: &SparseSubmConvConfig,
    input: BurnTensor<DefaultWgpuBackend, 2>,
    neighbor_rows: BurnTensor<DefaultWgpuBackend, 2, Int>,
    weight: BurnTensor<DefaultWgpuBackend, 5>,
    bias: BurnTensor<DefaultWgpuBackend, 1>,
    forward: SparseWgpuForwardConfig,
) -> Result<BurnTensor<DefaultWgpuBackend, 2>, String> {
    let query_rows = neighbor_rows.dims()[0];
    let kernel_rows_count = kernel_rows(config)?;
    let resolved =
        resolve_sparse_wgpu_forward_config_internal(config, query_rows, kernel_rows_count, forward);
    let split_k = resolved.split_k;
    let dispatches = if split_k > 1 {
        split_k.saturating_add(1)
    } else {
        1
    };
    let output_elements = query_rows.saturating_mul(config.out_channels);
    let conv_start = Instant::now();
    let output = sparse_subm_conv_forward_cubecl_impl(
        config,
        input.into_primitive().tensor(),
        neighbor_rows.into_primitive(),
        weight.into_primitive().tensor(),
        bias.into_primitive().tensor(),
        forward,
    )?;
    record_sparse_wgpu_conv_call(
        query_rows,
        output_elements,
        dispatches,
        split_k,
        resolved.kernel_variant,
        elapsed_ns_u64(conv_start),
    );
    Ok(BurnTensor::from_primitive(TensorPrimitive::Float(output)))
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

    let input_shape = input.meta.shape.as_ref();
    let neighbor_shape = neighbor_rows.meta.shape.as_ref();
    let weight_shape = weight.meta.shape.as_ref();
    let bias_shape = bias.meta.shape.as_ref();

    if input_shape.len() != 2 {
        return Err(format!(
            "sparse conv input rank mismatch: got {} expected 2",
            input_shape.len()
        ));
    }
    if neighbor_shape.len() != 2 {
        return Err(format!(
            "sparse conv neighbor_rows rank mismatch: got {} expected 2",
            neighbor_shape.len()
        ));
    }
    if weight_shape.len() != 5 {
        return Err(format!(
            "sparse conv weight rank mismatch: got {} expected 5",
            weight_shape.len()
        ));
    }
    if bias_shape.len() != 1 {
        return Err(format!(
            "sparse conv bias rank mismatch: got {} expected 1",
            bias_shape.len()
        ));
    }

    let input_rows = input_shape[0];
    let query_rows = neighbor_shape[0];
    if input_shape[1] != config.in_channels {
        return Err(format!(
            "sparse conv input channel mismatch: got {} expected {}",
            input_shape[1], config.in_channels
        ));
    }
    if query_rows > input_rows {
        return Err(format!(
            "sparse conv neighbor row count exceeds input rows: got {} input rows {}",
            query_rows, input_rows
        ));
    }
    let expected_kernel_rows = kernel_rows(config)?;
    if neighbor_shape[1] != expected_kernel_rows {
        return Err(format!(
            "sparse conv neighbor kernel rows mismatch: got {} expected {}",
            neighbor_shape[1], expected_kernel_rows
        ));
    }

    let expected_weight = [
        config.out_channels,
        config.kernel_d,
        config.kernel_h,
        config.kernel_w,
        config.in_channels_per_group,
    ];
    if weight_shape != expected_weight.as_slice() {
        return Err(format!(
            "sparse conv weight shape mismatch: got {:?} expected {:?}",
            weight_shape, expected_weight
        ));
    }
    if bias_shape[0] != config.out_channels {
        return Err(format!(
            "sparse conv bias len mismatch: got {} expected {}",
            bias_shape[0], config.out_channels
        ));
    }
    Ok(())
}

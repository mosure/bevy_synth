use super::*;

/// Sparse submanifold convolution through device gather + backend matmul.
///
/// This path is intended for single-group decoder hotspots where the arithmetic
/// intensity is high enough for the backend matmul to beat the scalar sparse
/// CubeCL kernels, even with the gathered im2col view.
pub fn sparse_subm_conv_forward_wgpu_im2col_matmul(
    config: &SparseSubmConvConfig,
    input: BurnTensor<DefaultWgpuBackend, 2>,
    neighbor_rows: BurnTensor<DefaultWgpuBackend, 2, Int>,
    weight: BurnTensor<DefaultWgpuBackend, 5>,
    bias: BurnTensor<DefaultWgpuBackend, 1>,
) -> Result<BurnTensor<DefaultWgpuBackend, 2>, String> {
    sparse_subm_conv_forward_wgpu_im2col_matmul_impl(
        config,
        input,
        neighbor_rows,
        weight,
        bias,
        None,
    )
}

pub fn sparse_subm_conv_forward_wgpu_im2col_matmul_fast_f16(
    config: &SparseSubmConvConfig,
    input: BurnTensor<DefaultWgpuBackend, 2>,
    neighbor_rows: BurnTensor<DefaultWgpuBackend, 2, Int>,
    weight: BurnTensor<DefaultWgpuBackend, 5>,
    bias: BurnTensor<DefaultWgpuBackend, 1>,
) -> Result<BurnTensor<DefaultWgpuBackend, 2>, String> {
    sparse_subm_conv_forward_wgpu_im2col_matmul_impl(
        config,
        input,
        neighbor_rows,
        weight,
        bias,
        Some(burn::tensor::FloatDType::F16),
    )
}

fn sparse_subm_conv_forward_wgpu_im2col_matmul_impl(
    config: &SparseSubmConvConfig,
    input: BurnTensor<DefaultWgpuBackend, 2>,
    neighbor_rows: BurnTensor<DefaultWgpuBackend, 2, Int>,
    weight: BurnTensor<DefaultWgpuBackend, 5>,
    bias: BurnTensor<DefaultWgpuBackend, 1>,
    matmul_dtype: Option<burn::tensor::FloatDType>,
) -> Result<BurnTensor<DefaultWgpuBackend, 2>, String> {
    if config.groups != 1
        || config.in_channels_per_group != config.in_channels
        || config.out_channels_per_group != config.out_channels
    {
        return Err("im2col sparse conv requires single-group config".to_string());
    }
    let [rows, kernel_rows] = neighbor_rows.dims();
    let [input_rows, in_channels] = input.dims();
    let [
        out_channels,
        kernel_d,
        kernel_h,
        kernel_w,
        weight_in_channels,
    ] = weight.dims();
    if in_channels != config.in_channels || weight_in_channels != config.in_channels {
        return Err(format!(
            "im2col sparse conv channel mismatch: input={} config={} weight={}",
            in_channels, config.in_channels, weight_in_channels
        ));
    }
    let expected_kernel_rows = kernel_d
        .checked_mul(kernel_h)
        .and_then(|value| value.checked_mul(kernel_w))
        .ok_or_else(|| "im2col sparse conv kernel-row overflow".to_string())?;
    if expected_kernel_rows != kernel_rows {
        return Err(format!(
            "im2col sparse conv kernel rows mismatch: neighbor={} weight={}",
            kernel_rows, expected_kernel_rows
        ));
    }
    let [bias_channels] = bias.dims();
    if bias_channels != out_channels {
        return Err(format!(
            "im2col sparse conv bias mismatch: bias={} out_channels={}",
            bias_channels, out_channels
        ));
    }
    if rows == 0 {
        return Ok(BurnTensor::<DefaultWgpuBackend, 2>::zeros(
            [0, out_channels],
            &input.device(),
        ));
    }
    if input_rows == 0 {
        return Ok(BurnTensor::<DefaultWgpuBackend, 2>::zeros(
            [rows, out_channels],
            &input.device(),
        ));
    }

    let max_input_row = i32::try_from(input_rows.saturating_sub(1))
        .map_err(|_| "im2col sparse conv input row count exceeds i32".to_string())?;
    let flat_neighbor_rows = neighbor_rows
        .clone()
        .clamp(0, max_input_row)
        .reshape([rows.saturating_mul(kernel_rows)]);
    let gathered = input
        .select(0, flat_neighbor_rows)
        .reshape([rows, kernel_rows, in_channels]);
    let valid_mask = neighbor_rows
        .greater_equal_elem(0)
        .float()
        .reshape([rows, kernel_rows, 1]);
    let cols = gathered
        .mul(valid_mask)
        .reshape([rows, kernel_rows.saturating_mul(in_channels)]);
    let weight_mat = weight
        .reshape([out_channels, kernel_rows.saturating_mul(in_channels)])
        .swap_dims(0, 1);
    let output = if let Some(dtype) = matmul_dtype {
        let cols = cast_float_tensor_if_needed(cols, dtype);
        let weight_mat = cast_float_tensor_if_needed(weight_mat, dtype);
        cols.matmul(weight_mat).cast(burn::tensor::FloatDType::F32)
    } else {
        cols.matmul(weight_mat)
    };
    Ok(output.add(bias.reshape([1, out_channels])))
}

fn resolve_split_k(
    config: &SparseSubmConvConfig,
    rows: usize,
    kernel_rows: usize,
    split_k_override: Option<usize>,
) -> usize {
    let max_split = 8usize;
    let mut split = if let Some(override_split) = split_k_override {
        override_split.clamp(1, max_split)
    } else {
        let k_in = kernel_rows.saturating_mul(config.in_channels_per_group);
        let work = rows
            .saturating_mul(config.out_channels_per_group)
            .saturating_mul(k_in);
        if work >= DEFAULT_SPARSE_WGPU_SPLIT_WORK_THRESHOLD_SPLIT4 {
            4
        } else if work >= DEFAULT_SPARSE_WGPU_SPLIT_WORK_THRESHOLD_SPLIT2 {
            2
        } else {
            1
        }
    };

    let output_elements = rows.saturating_mul(config.out_channels);
    let output_bytes = output_elements.saturating_mul(core::mem::size_of::<f32>());
    let max_partial_bytes = 256 * 1024 * 1024usize;
    // Large row-count decode convs are memory-bound; split-k partial/finalize
    // overhead dominates, so force single-pass kernels for these regimes.
    if rows >= DEFAULT_SPARSE_WGPU_SPLIT_CAP_ROWS {
        split = 1;
    }
    if rows >= DEFAULT_SPARSE_WGPU_SPLIT_CAP_HIGH_OC_ROWS
        && config.out_channels_per_group >= DEFAULT_SPARSE_WGPU_SPLIT_CAP_HIGH_OC_GROUP
    {
        split = 1;
    }
    if (DEFAULT_SPARSE_WGPU_SPLIT_CAP_VERY_HIGH_OC_MIN_ROWS
        ..DEFAULT_SPARSE_WGPU_SPLIT_CAP_HIGH_OC_ROWS)
        .contains(&rows)
        && config.out_channels_per_group >= DEFAULT_SPARSE_WGPU_SPLIT_CAP_VERY_HIGH_OC_GROUP
    {
        split = 1;
    }
    while split > 1 {
        let partial_bytes = output_bytes.saturating_mul(split);
        if partial_bytes <= max_partial_bytes {
            break;
        }
        split -= 1;
    }
    split.max(1)
}

fn use_single_group_specialization(config: &SparseSubmConvConfig) -> bool {
    config.groups == 1
        && config.in_channels_per_group == config.in_channels
        && config.out_channels_per_group == config.out_channels
}

fn resolve_sparse_conv_kernel_variant(
    config: &SparseSubmConvConfig,
    rows: usize,
    kernel_rows: usize,
    kernel_override: SparseWgpuKernelVariant,
) -> SparseConvKernelVariant {
    let single_group_specialized = use_single_group_specialization(config);
    match kernel_override {
        SparseWgpuKernelVariant::Baseline => {
            return if single_group_specialized {
                SparseConvKernelVariant::BaselineSingleGroup
            } else {
                SparseConvKernelVariant::Baseline
            };
        }
        SparseWgpuKernelVariant::FusedOc4 => {
            return if single_group_specialized {
                SparseConvKernelVariant::FusedOc4SingleGroup
            } else {
                SparseConvKernelVariant::FusedOc4
            };
        }
        SparseWgpuKernelVariant::Auto => {}
    }

    let inner_work = kernel_rows.saturating_mul(config.in_channels_per_group);
    let output_work = rows.saturating_mul(config.out_channels_per_group);
    if single_group_specialized
        && rows == DEFAULT_SPARSE_WGPU_FUSED_HOT_ROWS
        && inner_work <= DEFAULT_SPARSE_WGPU_FUSED_HOT_MAX_INNER_WORK
        && output_work >= DEFAULT_SPARSE_WGPU_FUSED_HOT_MIN_OUTPUT_WORK
        && config.out_channels_per_group <= DEFAULT_SPARSE_WGPU_FUSED_HOT_MAX_OC_GROUP
    {
        return SparseConvKernelVariant::FusedOc4SingleGroup;
    }
    if config.out_channels_per_group >= DEFAULT_SPARSE_WGPU_FUSED_AUTO_MIN_OC_GROUP
        && config.out_channels >= FUSED_OC_TILE
        && rows >= DEFAULT_SPARSE_WGPU_FUSED_AUTO_MIN_ROWS
        && inner_work >= DEFAULT_SPARSE_WGPU_FUSED_AUTO_MIN_INNER_WORK
        && config.in_channels_per_group <= DEFAULT_SPARSE_WGPU_FUSED_AUTO_MAX_IN_CHANNELS_PER_GROUP
        && output_work >= DEFAULT_SPARSE_WGPU_FUSED_AUTO_MIN_OUTPUT_WORK
    {
        if single_group_specialized {
            SparseConvKernelVariant::FusedOc4SingleGroup
        } else {
            SparseConvKernelVariant::FusedOc4
        }
    } else {
        if single_group_specialized {
            SparseConvKernelVariant::BaselineSingleGroup
        } else {
            SparseConvKernelVariant::Baseline
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ResolvedSparseWgpuForwardConfigInternal {
    pub(super) kernel_variant: SparseConvKernelVariant,
    pub(super) split_k: usize,
}

fn sparse_wgpu_kernel_variant_public(
    kernel_variant: SparseConvKernelVariant,
) -> SparseWgpuKernelVariant {
    match kernel_variant {
        SparseConvKernelVariant::Baseline | SparseConvKernelVariant::BaselineSingleGroup => {
            SparseWgpuKernelVariant::Baseline
        }
        SparseConvKernelVariant::FusedOc4 | SparseConvKernelVariant::FusedOc4SingleGroup => {
            SparseWgpuKernelVariant::FusedOc4
        }
    }
}

pub(super) fn resolve_sparse_wgpu_forward_config_internal(
    config: &SparseSubmConvConfig,
    rows: usize,
    kernel_rows: usize,
    forward: SparseWgpuForwardConfig,
) -> ResolvedSparseWgpuForwardConfigInternal {
    ResolvedSparseWgpuForwardConfigInternal {
        kernel_variant: resolve_sparse_conv_kernel_variant(
            config,
            rows,
            kernel_rows,
            forward.kernel_variant,
        ),
        split_k: resolve_split_k(config, rows, kernel_rows, forward.split_k),
    }
}

pub fn resolve_sparse_wgpu_forward_config(
    config: &SparseSubmConvConfig,
    rows: usize,
    forward: SparseWgpuForwardConfig,
) -> Result<SparseWgpuResolvedForwardConfig, String> {
    let kernel_rows = kernel_rows(config)?;
    let resolved = resolve_sparse_wgpu_forward_config_internal(config, rows, kernel_rows, forward);
    Ok(SparseWgpuResolvedForwardConfig {
        kernel_variant: sparse_wgpu_kernel_variant_public(resolved.kernel_variant),
        split_k: resolved.split_k,
    })
}

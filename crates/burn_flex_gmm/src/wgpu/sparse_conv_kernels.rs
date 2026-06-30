use super::*;

#[allow(clippy::useless_conversion)]
#[cube(launch_unchecked)]
pub(super) fn sparse_subm_conv_kernel(
    input: &Array<f32>,
    neighbor_rows: &Array<i32>,
    weight: &Array<f32>,
    bias: &Array<f32>,
    output: &mut Array<f32>,
    out_channels: &usize,
    kernel_rows: &usize,
    in_channels: &usize,
    in_channels_per_group: &usize,
    out_channels_per_group: &usize,
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
        let valid_neighbor = neighbor >= 0;
        let in_row_i32 = if valid_neighbor { neighbor } else { 0.into() };
        let in_row = usize::cast_from(in_row_i32);
        let input_base = in_row * *in_channels + in_group_base;
        let weight_base = (out_channel * *kernel_rows + kernel_idx) * *in_channels_per_group;
        for in_local in 0..*in_channels_per_group {
            let input_value = input[input_base + in_local];
            let weight_value = weight[weight_base + in_local];
            let term = input_value * weight_value;
            if valid_neighbor {
                acc += term;
            }
        }
    }

    output[out_idx] = acc;
}

#[allow(clippy::useless_conversion)]
#[cube(launch_unchecked)]
pub(super) fn sparse_subm_conv_single_group_kernel(
    input: &Array<f32>,
    neighbor_rows: &Array<i32>,
    weight: &Array<f32>,
    bias: &Array<f32>,
    output: &mut Array<f32>,
    out_channels: &usize,
    kernel_rows: &usize,
    in_channels: &usize,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let out_idx = ABSOLUTE_POS;
    let row = out_idx / *out_channels;
    let out_channel = out_idx % *out_channels;

    let mut acc = bias[out_channel];
    for kernel_idx in 0..*kernel_rows {
        let neighbor = neighbor_rows[row * *kernel_rows + kernel_idx];
        let valid_neighbor = neighbor >= 0;
        let in_row_i32 = if valid_neighbor { neighbor } else { 0.into() };
        let in_row = usize::cast_from(in_row_i32);
        let input_base = in_row * *in_channels;
        let weight_base = (out_channel * *kernel_rows + kernel_idx) * *in_channels;
        for in_local in 0..*in_channels {
            let input_value = input[input_base + in_local];
            let weight_value = weight[weight_base + in_local];
            let term = input_value * weight_value;
            if valid_neighbor {
                acc += term;
            }
        }
    }

    output[out_idx] = acc;
}

#[allow(clippy::useless_conversion)]
#[cube(launch_unchecked)]
pub(super) fn sparse_subm_conv_fused_oc4_kernel(
    input: &Array<f32>,
    neighbor_rows: &Array<i32>,
    weight: &Array<f32>,
    bias: &Array<f32>,
    output: &mut Array<f32>,
    out_channels: &usize,
    kernel_rows: &usize,
    in_channels: &usize,
    in_channels_per_group: &usize,
    out_channels_per_group: &usize,
) {
    let blocks_per_row = (*out_channels).div_ceil(FUSED_OC_TILE);
    let rows = neighbor_rows.len() / *kernel_rows;
    let output_blocks = rows * blocks_per_row;
    if ABSOLUTE_POS >= output_blocks {
        terminate!();
    }

    let tile_idx = ABSOLUTE_POS;
    let row = tile_idx / blocks_per_row;
    let block = tile_idx % blocks_per_row;
    let out_channel_0 = block * FUSED_OC_TILE;
    let out_channel_1 = out_channel_0 + 1;
    let out_channel_2 = out_channel_0 + 2;
    let out_channel_3 = out_channel_0 + 3;
    let valid_0 = out_channel_0 < *out_channels;
    let valid_1 = out_channel_1 < *out_channels;
    let valid_2 = out_channel_2 < *out_channels;
    let valid_3 = out_channel_3 < *out_channels;

    let mut acc_0 = 0.0;
    let mut acc_1 = 0.0;
    let mut acc_2 = 0.0;
    let mut acc_3 = 0.0;
    if valid_0 {
        acc_0 = bias[out_channel_0];
    }
    if valid_1 {
        acc_1 = bias[out_channel_1];
    }
    if valid_2 {
        acc_2 = bias[out_channel_2];
    }
    if valid_3 {
        acc_3 = bias[out_channel_3];
    }

    for kernel_idx in 0..*kernel_rows {
        let neighbor = neighbor_rows[row * *kernel_rows + kernel_idx];
        let valid_neighbor = neighbor >= 0;
        let in_row_i32 = if valid_neighbor { neighbor } else { 0.into() };
        let in_row = usize::cast_from(in_row_i32);
        if valid_0 {
            let group_0 = out_channel_0 / *out_channels_per_group;
            let in_group_base_0 = group_0 * *in_channels_per_group;
            let input_base_0 = in_row * *in_channels + in_group_base_0;
            let weight_base_0 =
                (out_channel_0 * *kernel_rows + kernel_idx) * *in_channels_per_group;
            for in_local in 0..*in_channels_per_group {
                let input_value_0 = input[input_base_0 + in_local];
                let weight_value_0 = weight[weight_base_0 + in_local];
                let term_0 = input_value_0 * weight_value_0;
                if valid_neighbor {
                    acc_0 += term_0;
                }
            }
        }
        if valid_1 {
            let group_1 = out_channel_1 / *out_channels_per_group;
            let in_group_base_1 = group_1 * *in_channels_per_group;
            let input_base_1 = in_row * *in_channels + in_group_base_1;
            let weight_base_1 =
                (out_channel_1 * *kernel_rows + kernel_idx) * *in_channels_per_group;
            for in_local in 0..*in_channels_per_group {
                let input_value_1 = input[input_base_1 + in_local];
                let weight_value_1 = weight[weight_base_1 + in_local];
                let term_1 = input_value_1 * weight_value_1;
                if valid_neighbor {
                    acc_1 += term_1;
                }
            }
        }
        if valid_2 {
            let group_2 = out_channel_2 / *out_channels_per_group;
            let in_group_base_2 = group_2 * *in_channels_per_group;
            let input_base_2 = in_row * *in_channels + in_group_base_2;
            let weight_base_2 =
                (out_channel_2 * *kernel_rows + kernel_idx) * *in_channels_per_group;
            for in_local in 0..*in_channels_per_group {
                let input_value_2 = input[input_base_2 + in_local];
                let weight_value_2 = weight[weight_base_2 + in_local];
                let term_2 = input_value_2 * weight_value_2;
                if valid_neighbor {
                    acc_2 += term_2;
                }
            }
        }
        if valid_3 {
            let group_3 = out_channel_3 / *out_channels_per_group;
            let in_group_base_3 = group_3 * *in_channels_per_group;
            let input_base_3 = in_row * *in_channels + in_group_base_3;
            let weight_base_3 =
                (out_channel_3 * *kernel_rows + kernel_idx) * *in_channels_per_group;
            for in_local in 0..*in_channels_per_group {
                let input_value_3 = input[input_base_3 + in_local];
                let weight_value_3 = weight[weight_base_3 + in_local];
                let term_3 = input_value_3 * weight_value_3;
                if valid_neighbor {
                    acc_3 += term_3;
                }
            }
        }
    }

    let row_base = row * *out_channels;
    if valid_0 {
        output[row_base + out_channel_0] = acc_0;
    }
    if valid_1 {
        output[row_base + out_channel_1] = acc_1;
    }
    if valid_2 {
        output[row_base + out_channel_2] = acc_2;
    }
    if valid_3 {
        output[row_base + out_channel_3] = acc_3;
    }
}

#[allow(clippy::useless_conversion)]
#[cube(launch_unchecked)]
pub(super) fn sparse_subm_conv_single_group_fused_oc4_kernel(
    input: &Array<f32>,
    neighbor_rows: &Array<i32>,
    weight: &Array<f32>,
    bias: &Array<f32>,
    output: &mut Array<f32>,
    out_channels: &usize,
    kernel_rows: &usize,
    in_channels: &usize,
) {
    let blocks_per_row = (*out_channels).div_ceil(FUSED_OC_TILE);
    let rows = neighbor_rows.len() / *kernel_rows;
    let output_blocks = rows * blocks_per_row;
    if ABSOLUTE_POS >= output_blocks {
        terminate!();
    }

    let tile_idx = ABSOLUTE_POS;
    let row = tile_idx / blocks_per_row;
    let block = tile_idx % blocks_per_row;
    let out_channel_0 = block * FUSED_OC_TILE;
    let out_channel_1 = out_channel_0 + 1;
    let out_channel_2 = out_channel_0 + 2;
    let out_channel_3 = out_channel_0 + 3;
    let valid_0 = out_channel_0 < *out_channels;
    let valid_1 = out_channel_1 < *out_channels;
    let valid_2 = out_channel_2 < *out_channels;
    let valid_3 = out_channel_3 < *out_channels;

    let mut acc_0 = 0.0;
    let mut acc_1 = 0.0;
    let mut acc_2 = 0.0;
    let mut acc_3 = 0.0;
    if valid_0 {
        acc_0 = bias[out_channel_0];
    }
    if valid_1 {
        acc_1 = bias[out_channel_1];
    }
    if valid_2 {
        acc_2 = bias[out_channel_2];
    }
    if valid_3 {
        acc_3 = bias[out_channel_3];
    }

    for kernel_idx in 0..*kernel_rows {
        let neighbor = neighbor_rows[row * *kernel_rows + kernel_idx];
        let valid_neighbor = neighbor >= 0;
        let in_row_i32 = if valid_neighbor { neighbor } else { 0.into() };
        let in_row = usize::cast_from(in_row_i32);
        if valid_0 {
            let input_base_0 = in_row * *in_channels;
            let weight_base_0 = (out_channel_0 * *kernel_rows + kernel_idx) * *in_channels;
            for in_local in 0..*in_channels {
                let input_value_0 = input[input_base_0 + in_local];
                let weight_value_0 = weight[weight_base_0 + in_local];
                let term_0 = input_value_0 * weight_value_0;
                if valid_neighbor {
                    acc_0 += term_0;
                }
            }
        }
        if valid_1 {
            let input_base_1 = in_row * *in_channels;
            let weight_base_1 = (out_channel_1 * *kernel_rows + kernel_idx) * *in_channels;
            for in_local in 0..*in_channels {
                let input_value_1 = input[input_base_1 + in_local];
                let weight_value_1 = weight[weight_base_1 + in_local];
                let term_1 = input_value_1 * weight_value_1;
                if valid_neighbor {
                    acc_1 += term_1;
                }
            }
        }
        if valid_2 {
            let input_base_2 = in_row * *in_channels;
            let weight_base_2 = (out_channel_2 * *kernel_rows + kernel_idx) * *in_channels;
            for in_local in 0..*in_channels {
                let input_value_2 = input[input_base_2 + in_local];
                let weight_value_2 = weight[weight_base_2 + in_local];
                let term_2 = input_value_2 * weight_value_2;
                if valid_neighbor {
                    acc_2 += term_2;
                }
            }
        }
        if valid_3 {
            let input_base_3 = in_row * *in_channels;
            let weight_base_3 = (out_channel_3 * *kernel_rows + kernel_idx) * *in_channels;
            for in_local in 0..*in_channels {
                let input_value_3 = input[input_base_3 + in_local];
                let weight_value_3 = weight[weight_base_3 + in_local];
                let term_3 = input_value_3 * weight_value_3;
                if valid_neighbor {
                    acc_3 += term_3;
                }
            }
        }
    }

    let row_base = row * *out_channels;
    if valid_0 {
        output[row_base + out_channel_0] = acc_0;
    }
    if valid_1 {
        output[row_base + out_channel_1] = acc_1;
    }
    if valid_2 {
        output[row_base + out_channel_2] = acc_2;
    }
    if valid_3 {
        output[row_base + out_channel_3] = acc_3;
    }
}

#[allow(clippy::useless_conversion)]
#[cube(launch_unchecked)]
pub(super) fn sparse_subm_conv_splitk_partial_kernel(
    input: &Array<f32>,
    neighbor_rows: &Array<i32>,
    weight: &Array<f32>,
    partial: &mut Array<f32>,
    out_channels: &usize,
    kernel_rows: &usize,
    in_channels: &usize,
    in_channels_per_group: &usize,
    out_channels_per_group: &usize,
    output_elements: &usize,
    split_k: &usize,
) {
    if ABSOLUTE_POS >= partial.len() {
        terminate!();
    }

    let partial_idx = ABSOLUTE_POS;
    let split_idx = partial_idx / *output_elements;
    let out_idx = partial_idx % *output_elements;
    if split_idx >= *split_k {
        partial[partial_idx] = 0.0;
        terminate!();
    }

    let row = out_idx / *out_channels;
    let out_channel = out_idx % *out_channels;
    let group = out_channel / *out_channels_per_group;
    let in_group_base = group * *in_channels_per_group;
    let chunk = (*kernel_rows).div_ceil(*split_k);
    let kernel_start = split_idx * chunk;
    let kernel_end = (kernel_start + chunk).min(*kernel_rows);

    let mut acc = 0.0;
    for kernel_idx in kernel_start..kernel_end {
        let neighbor = neighbor_rows[row * *kernel_rows + kernel_idx];
        let valid_neighbor = neighbor >= 0;
        let in_row_i32 = if valid_neighbor { neighbor } else { 0.into() };
        let in_row = usize::cast_from(in_row_i32);
        let input_base = in_row * *in_channels + in_group_base;
        let weight_base = (out_channel * *kernel_rows + kernel_idx) * *in_channels_per_group;
        for in_local in 0..*in_channels_per_group {
            let input_value = input[input_base + in_local];
            let weight_value = weight[weight_base + in_local];
            let term = input_value * weight_value;
            if valid_neighbor {
                acc += term;
            }
        }
    }

    partial[partial_idx] = acc;
}

#[allow(clippy::useless_conversion)]
#[cube(launch_unchecked)]
pub(super) fn sparse_subm_conv_splitk_partial_single_group_kernel(
    input: &Array<f32>,
    neighbor_rows: &Array<i32>,
    weight: &Array<f32>,
    partial: &mut Array<f32>,
    out_channels: &usize,
    kernel_rows: &usize,
    in_channels: &usize,
    output_elements: &usize,
    split_k: &usize,
) {
    if ABSOLUTE_POS >= partial.len() {
        terminate!();
    }

    let partial_idx = ABSOLUTE_POS;
    let split_idx = partial_idx / *output_elements;
    let out_idx = partial_idx % *output_elements;
    if split_idx >= *split_k {
        partial[partial_idx] = 0.0;
        terminate!();
    }

    let row = out_idx / *out_channels;
    let out_channel = out_idx % *out_channels;
    let chunk = (*kernel_rows).div_ceil(*split_k);
    let kernel_start = split_idx * chunk;
    let kernel_end = (kernel_start + chunk).min(*kernel_rows);

    let mut acc = 0.0;
    for kernel_idx in kernel_start..kernel_end {
        let neighbor = neighbor_rows[row * *kernel_rows + kernel_idx];
        let valid_neighbor = neighbor >= 0;
        let in_row_i32 = if valid_neighbor { neighbor } else { 0.into() };
        let in_row = usize::cast_from(in_row_i32);
        let input_base = in_row * *in_channels;
        let weight_base = (out_channel * *kernel_rows + kernel_idx) * *in_channels;
        for in_local in 0..*in_channels {
            let input_value = input[input_base + in_local];
            let weight_value = weight[weight_base + in_local];
            let term = input_value * weight_value;
            if valid_neighbor {
                acc += term;
            }
        }
    }

    partial[partial_idx] = acc;
}

#[allow(clippy::useless_conversion)]
#[cube(launch_unchecked)]
pub(super) fn sparse_subm_conv_splitk_partial_fused_oc4_kernel(
    input: &Array<f32>,
    neighbor_rows: &Array<i32>,
    weight: &Array<f32>,
    partial: &mut Array<f32>,
    out_channels: &usize,
    kernel_rows: &usize,
    in_channels: &usize,
    in_channels_per_group: &usize,
    out_channels_per_group: &usize,
    output_elements: &usize,
    split_k: &usize,
) {
    let blocks_per_row = (*out_channels).div_ceil(FUSED_OC_TILE);
    let rows = neighbor_rows.len() / *kernel_rows;
    let split_tiles = rows * blocks_per_row;
    if split_tiles == 0 {
        terminate!();
    }
    if ABSOLUTE_POS >= split_tiles * *split_k {
        terminate!();
    }

    let tile_idx = ABSOLUTE_POS;
    let split_idx = tile_idx / split_tiles;
    if split_idx >= *split_k {
        terminate!();
    }
    let split_tile = tile_idx % split_tiles;
    let row = split_tile / blocks_per_row;
    let block = split_tile % blocks_per_row;
    let out_channel_0 = block * FUSED_OC_TILE;
    let out_channel_1 = out_channel_0 + 1;
    let out_channel_2 = out_channel_0 + 2;
    let out_channel_3 = out_channel_0 + 3;
    let valid_0 = out_channel_0 < *out_channels;
    let valid_1 = out_channel_1 < *out_channels;
    let valid_2 = out_channel_2 < *out_channels;
    let valid_3 = out_channel_3 < *out_channels;

    let chunk = (*kernel_rows).div_ceil(*split_k);
    let kernel_start = split_idx * chunk;
    let kernel_end = (kernel_start + chunk).min(*kernel_rows);

    let mut acc_0 = 0.0;
    let mut acc_1 = 0.0;
    let mut acc_2 = 0.0;
    let mut acc_3 = 0.0;

    for kernel_idx in kernel_start..kernel_end {
        let neighbor = neighbor_rows[row * *kernel_rows + kernel_idx];
        let valid_neighbor = neighbor >= 0;
        let in_row_i32 = if valid_neighbor { neighbor } else { 0.into() };
        let in_row = usize::cast_from(in_row_i32);

        if valid_0 {
            let group_0 = out_channel_0 / *out_channels_per_group;
            let in_group_base_0 = group_0 * *in_channels_per_group;
            let input_base_0 = in_row * *in_channels + in_group_base_0;
            let weight_base_0 =
                (out_channel_0 * *kernel_rows + kernel_idx) * *in_channels_per_group;
            for in_local in 0..*in_channels_per_group {
                let input_value_0 = input[input_base_0 + in_local];
                let weight_value_0 = weight[weight_base_0 + in_local];
                let term_0 = input_value_0 * weight_value_0;
                if valid_neighbor {
                    acc_0 += term_0;
                }
            }
        }
        if valid_1 {
            let group_1 = out_channel_1 / *out_channels_per_group;
            let in_group_base_1 = group_1 * *in_channels_per_group;
            let input_base_1 = in_row * *in_channels + in_group_base_1;
            let weight_base_1 =
                (out_channel_1 * *kernel_rows + kernel_idx) * *in_channels_per_group;
            for in_local in 0..*in_channels_per_group {
                let input_value_1 = input[input_base_1 + in_local];
                let weight_value_1 = weight[weight_base_1 + in_local];
                let term_1 = input_value_1 * weight_value_1;
                if valid_neighbor {
                    acc_1 += term_1;
                }
            }
        }
        if valid_2 {
            let group_2 = out_channel_2 / *out_channels_per_group;
            let in_group_base_2 = group_2 * *in_channels_per_group;
            let input_base_2 = in_row * *in_channels + in_group_base_2;
            let weight_base_2 =
                (out_channel_2 * *kernel_rows + kernel_idx) * *in_channels_per_group;
            for in_local in 0..*in_channels_per_group {
                let input_value_2 = input[input_base_2 + in_local];
                let weight_value_2 = weight[weight_base_2 + in_local];
                let term_2 = input_value_2 * weight_value_2;
                if valid_neighbor {
                    acc_2 += term_2;
                }
            }
        }
        if valid_3 {
            let group_3 = out_channel_3 / *out_channels_per_group;
            let in_group_base_3 = group_3 * *in_channels_per_group;
            let input_base_3 = in_row * *in_channels + in_group_base_3;
            let weight_base_3 =
                (out_channel_3 * *kernel_rows + kernel_idx) * *in_channels_per_group;
            for in_local in 0..*in_channels_per_group {
                let input_value_3 = input[input_base_3 + in_local];
                let weight_value_3 = weight[weight_base_3 + in_local];
                let term_3 = input_value_3 * weight_value_3;
                if valid_neighbor {
                    acc_3 += term_3;
                }
            }
        }
    }

    let split_base = split_idx * *output_elements;
    let row_base = row * *out_channels;
    if valid_0 {
        partial[split_base + row_base + out_channel_0] = acc_0;
    }
    if valid_1 {
        partial[split_base + row_base + out_channel_1] = acc_1;
    }
    if valid_2 {
        partial[split_base + row_base + out_channel_2] = acc_2;
    }
    if valid_3 {
        partial[split_base + row_base + out_channel_3] = acc_3;
    }
}

#[allow(clippy::useless_conversion)]
#[cube(launch_unchecked)]
pub(super) fn sparse_subm_conv_splitk_partial_single_group_fused_oc4_kernel(
    input: &Array<f32>,
    neighbor_rows: &Array<i32>,
    weight: &Array<f32>,
    partial: &mut Array<f32>,
    out_channels: &usize,
    kernel_rows: &usize,
    in_channels: &usize,
    output_elements: &usize,
    split_k: &usize,
) {
    let blocks_per_row = (*out_channels).div_ceil(FUSED_OC_TILE);
    let rows = neighbor_rows.len() / *kernel_rows;
    let split_tiles = rows * blocks_per_row;
    if split_tiles == 0 {
        terminate!();
    }
    if ABSOLUTE_POS >= split_tiles * *split_k {
        terminate!();
    }

    let tile_idx = ABSOLUTE_POS;
    let split_idx = tile_idx / split_tiles;
    if split_idx >= *split_k {
        terminate!();
    }
    let split_tile = tile_idx % split_tiles;
    let row = split_tile / blocks_per_row;
    let block = split_tile % blocks_per_row;
    let out_channel_0 = block * FUSED_OC_TILE;
    let out_channel_1 = out_channel_0 + 1;
    let out_channel_2 = out_channel_0 + 2;
    let out_channel_3 = out_channel_0 + 3;
    let valid_0 = out_channel_0 < *out_channels;
    let valid_1 = out_channel_1 < *out_channels;
    let valid_2 = out_channel_2 < *out_channels;
    let valid_3 = out_channel_3 < *out_channels;

    let chunk = (*kernel_rows).div_ceil(*split_k);
    let kernel_start = split_idx * chunk;
    let kernel_end = (kernel_start + chunk).min(*kernel_rows);

    let mut acc_0 = 0.0;
    let mut acc_1 = 0.0;
    let mut acc_2 = 0.0;
    let mut acc_3 = 0.0;

    for kernel_idx in kernel_start..kernel_end {
        let neighbor = neighbor_rows[row * *kernel_rows + kernel_idx];
        let valid_neighbor = neighbor >= 0;
        let in_row_i32 = if valid_neighbor { neighbor } else { 0.into() };
        let in_row = usize::cast_from(in_row_i32);

        if valid_0 {
            let input_base_0 = in_row * *in_channels;
            let weight_base_0 = (out_channel_0 * *kernel_rows + kernel_idx) * *in_channels;
            for in_local in 0..*in_channels {
                let input_value_0 = input[input_base_0 + in_local];
                let weight_value_0 = weight[weight_base_0 + in_local];
                let term_0 = input_value_0 * weight_value_0;
                if valid_neighbor {
                    acc_0 += term_0;
                }
            }
        }
        if valid_1 {
            let input_base_1 = in_row * *in_channels;
            let weight_base_1 = (out_channel_1 * *kernel_rows + kernel_idx) * *in_channels;
            for in_local in 0..*in_channels {
                let input_value_1 = input[input_base_1 + in_local];
                let weight_value_1 = weight[weight_base_1 + in_local];
                let term_1 = input_value_1 * weight_value_1;
                if valid_neighbor {
                    acc_1 += term_1;
                }
            }
        }
        if valid_2 {
            let input_base_2 = in_row * *in_channels;
            let weight_base_2 = (out_channel_2 * *kernel_rows + kernel_idx) * *in_channels;
            for in_local in 0..*in_channels {
                let input_value_2 = input[input_base_2 + in_local];
                let weight_value_2 = weight[weight_base_2 + in_local];
                let term_2 = input_value_2 * weight_value_2;
                if valid_neighbor {
                    acc_2 += term_2;
                }
            }
        }
        if valid_3 {
            let input_base_3 = in_row * *in_channels;
            let weight_base_3 = (out_channel_3 * *kernel_rows + kernel_idx) * *in_channels;
            for in_local in 0..*in_channels {
                let input_value_3 = input[input_base_3 + in_local];
                let weight_value_3 = weight[weight_base_3 + in_local];
                let term_3 = input_value_3 * weight_value_3;
                if valid_neighbor {
                    acc_3 += term_3;
                }
            }
        }
    }

    let split_base = split_idx * *output_elements;
    let row_base = row * *out_channels;
    if valid_0 {
        partial[split_base + row_base + out_channel_0] = acc_0;
    }
    if valid_1 {
        partial[split_base + row_base + out_channel_1] = acc_1;
    }
    if valid_2 {
        partial[split_base + row_base + out_channel_2] = acc_2;
    }
    if valid_3 {
        partial[split_base + row_base + out_channel_3] = acc_3;
    }
}

#[cube(launch_unchecked)]
pub(super) fn sparse_subm_conv_splitk_finalize_kernel(
    partial: &Array<f32>,
    bias: &Array<f32>,
    output: &mut Array<f32>,
    out_channels: &usize,
    output_elements: &usize,
    split_k: &usize,
) {
    if ABSOLUTE_POS >= *output_elements {
        terminate!();
    }

    let out_idx = ABSOLUTE_POS;
    let out_channel = out_idx % *out_channels;
    let mut acc = bias[out_channel];
    for split_idx in 0..*split_k {
        acc += partial[split_idx * *output_elements + out_idx];
    }
    output[out_idx] = acc;
}
